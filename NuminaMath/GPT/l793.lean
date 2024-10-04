import Mathlib
import Mathlib.Algebra.GCD.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Lcm
import Mathlib.Algebra.ModularArithmetic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Group.Defs
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Fin.Int
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Lcm
import Mathlib.Data.Polynomial
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Log
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Init.Data.Nat.Basic
import Mathlib.Logic.Basic
import Mathlib.Order.AbsoluteValue
import Mathlib.Probability.Basic
import Mathlib.Probability.Conditional
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Sorry
import Mathlib.Topology.Constructions

namespace shaded_area_of_circles_l793_793877

theorem shaded_area_of_circles :
  let R := 10
  let r1 := R / 2
  let r2 := R / 2
  (π * R^2 - (π * r1^2 + π * r1^2 + π * r2^2)) = 25 * π :=
by
  sorry

end shaded_area_of_circles_l793_793877


namespace log_sum_real_coeffs_expansion_l793_793899

theorem log_sum_real_coeffs_expansion (x : ℂ) (h : x = I) : 
  ¬∃ (S : ℝ), S = (real.sum_of_real_coeffs (1 + ix) ^ 2010) ∧ real.log 2 S :=
by {
  sorry
}

end log_sum_real_coeffs_expansion_l793_793899


namespace percent_of_x_is_y_l793_793138

theorem percent_of_x_is_y (x y : ℝ) (h : 0.25 * (x - y) = 0.15 * (x + y)) : y / x = 0.25 :=
by
  -- proof omitted
  sorry

end percent_of_x_is_y_l793_793138


namespace line_param_func_l793_793969

theorem line_param_func (t : ℝ) : 
    ∃ f : ℝ → ℝ, (∀ t, (20 * t - 14) = 2 * (f t) - 30) ∧ (f t = 10 * t + 8) := by
  sorry

end line_param_func_l793_793969


namespace num_two_digit_numbers_with_digit_less_than_35_l793_793385

theorem num_two_digit_numbers_with_digit_less_than_35 : 
  let total_two_digit := 90 in
  let unsuitable_numbers := 35 in
  let suitable_numbers := total_two_digit - unsuitable_numbers
  in suitable_numbers = 55 :=
by 
  let total_two_digit := 90
  let unsuitable_numbers := 35
  let suitable_numbers := total_two_digit - unsuitable_numbers
  show suitable_numbers = 55, from sorry

end num_two_digit_numbers_with_digit_less_than_35_l793_793385


namespace sum_in_range_l793_793267

theorem sum_in_range (n : ℕ) (a : Fin n → ℝ) (h_nonneg : ∀ i, 0 ≤ a i)
    (h_bound : ∀ i, 1 ≤ i → a (i-1) ≤ a i ∧ a i ≤ 2 * a (i-1)) : 
    ∃ (b : Fin n → ℤ), (∀ i, b i = 1 ∨ b i = -1) ∧ 
    0 ≤ (∑ i, b i * a i) ∧ (∑ i, b i * a i) ≤ a 0 :=
by
  sorry

end sum_in_range_l793_793267


namespace probability_of_perfect_square_sum_l793_793611

def is_perfect_square (n : ℕ) : Prop :=
  n = 1*1 ∨ n = 2*2 ∨ n = 3*3 ∨ n = 4*4

theorem probability_of_perfect_square_sum :
  let total_outcomes := 64
  let perfect_square_sums := 12
  (perfect_square_sums / total_outcomes : ℚ) = 3 / 16 :=
by
  sorry

end probability_of_perfect_square_sum_l793_793611


namespace intersection_sets_l793_793270

-- defining sets A and B
def A : Set ℤ := {-1, 2, 4}
def B : Set ℤ := {0, 2, 6}

-- the theorem to be proved
theorem intersection_sets:
  A ∩ B = {2} :=
sorry

end intersection_sets_l793_793270


namespace number_of_correct_propositions_is_two_l793_793564

variable {α β γ : Type} [plane α] [plane β] [plane γ]
open Real

-- Proposition ①
def proposition1 (α : ℝ) : Prop :=
  let θ := atan (-sin α)
  θ ∈ [0, π / 4] ∪ [3 * π / 4, π]

-- Proposition ②
def proposition2 {A B C : triangle} (hacute : A + B + C = π) : Prop :=
  ∀ {a b c : ℝ}, sin a > cos b

-- Proposition ③
def proposition3 {x1 y1 k : ℝ} (x y : ℝ) : Prop :=
  y = y1 + k * (x - x1)

-- Proposition ④
def proposition4 : Prop :=
  (α ⟂ β) ∧ (β ⟂ γ) → ((α ⟂ γ) ∨ (α // γ))

def numberOfCorrectPropositions : ℕ :=
  let props := [proposition1, proposition2, proposition3, proposition4]
  props.count (λ p, p)

-- The statement we are going to prove
theorem number_of_correct_propositions_is_two :
  numberOfCorrectPropositions = 2 :=
sorry

end number_of_correct_propositions_is_two_l793_793564


namespace jessica_found_seashells_l793_793919

-- Define the given conditions
def mary_seashells : ℕ := 18
def total_seashells : ℕ := 59

-- Define the goal for the number of seashells Jessica found
def jessica_seashells (mary_seashells total_seashells : ℕ) : ℕ := total_seashells - mary_seashells

-- The theorem stating Jessica found 41 seashells
theorem jessica_found_seashells : jessica_seashells mary_seashells total_seashells = 41 := by
  -- We assume the conditions and skip the proof
  sorry

end jessica_found_seashells_l793_793919


namespace incorrect_statement_is_C_l793_793944

-- Definition of constants and conditions
def monthly_expenditure : ℝ := 5000
def passenger_profit_relationship (x : ℝ) : ℝ :=
  match x with
  | 1000 => -3000
  | 2000 => -1000
  | 3000 => 1000
  | 4000 => 3000
  | 5000 => 5000
  | _    => sorry  -- We fill the rest with a theorem or function that extrapolates linearly

-- Incorrect statement C is that at 2600 passengers, the bus incurs a loss
def incorrect_statement (x : ℝ) : Prop :=
  x = 2600 → passenger_profit_relationship x < 0

-- Proof that the incorrect statement is true
theorem incorrect_statement_is_C : incorrect_statement 2600 := sorry

end incorrect_statement_is_C_l793_793944


namespace ryan_quizzes_l793_793536

/-- Ryan aims to secure an A on at least 75% of his 60 quizzes this year.
By mid-year, he has achieved an A on 30 of the first 40 quizzes.
To meet his year-end goal, on at most how many of the remaining quizzes can
he earn a grade lower than an A? -/
theorem ryan_quizzes (total_quizzes: ℕ) (target_percent: ℝ) (midyear_quizzes: ℕ) (midyear_As: ℕ) :
    total_quizzes = 60 → target_percent = 0.75 → midyear_quizzes = 40 → midyear_As = 30 → 
    ∃ max_below_As : ℕ, max_below_As = 5 :=
by
  intros h1 h2 h3 h4
  have remaining_quizzes := total_quizzes - midyear_quizzes
  have required_As := target_percent * total_quizzes
  have remaining_As := required_As - midyear_As
  have max_below_As := remaining_quizzes - remaining_As
  use max_below_As
  norm_num at *
  sorry

end ryan_quizzes_l793_793536


namespace product_div_30_2_8_l793_793033

theorem product_div_30_2_8 : 
  let c := 2 in
  let d := 7 in
  let e := 6 in
  (c * d * e / 30) = 2.8 :=
by
  sorry

end product_div_30_2_8_l793_793033


namespace complement_A_complement_A_intersection_B_intersection_A_B_complement_intersection_A_B_l793_793508

def U : Set ℝ := {x | x ≥ -2}
def A : Set ℝ := {x | 2 < x ∧ x < 10}
def B : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}

theorem complement_A :
  (U \ A) = {x | -2 ≤ x ∧ x ≤ 2 ∨ x ≥ 10} :=
by sorry

theorem complement_A_intersection_B :
  (U \ A) ∩ B = {2} :=
by sorry

theorem intersection_A_B :
  A ∩ B = {x | 2 < x ∧ x ≤ 8} :=
by sorry

theorem complement_intersection_A_B :
  U \ (A ∩ B) = {x | -2 ≤ x ∧ x ≤ 2 ∨ x > 8} :=
by sorry

end complement_A_complement_A_intersection_B_intersection_A_B_complement_intersection_A_B_l793_793508


namespace sum_numerator_denominator_repeating_decimal_l793_793127

theorem sum_numerator_denominator_repeating_decimal :
  let x := 34 / 99 in
  x.denom * x + x.num = 133 :=
by
  let x : ℚ := 34 / 99
  sorry

end sum_numerator_denominator_repeating_decimal_l793_793127


namespace y_at_x_equals_8_l793_793402

theorem y_at_x_equals_8 (k : ℝ) (h1 : ∀ x y, y = k * x^(1/3))
    (h2 : 4 * real.sqrt 3 = k * 64^(1/3)) : k * 8^(1/3) = 2 * real.sqrt 3 :=
by
  sorry

end y_at_x_equals_8_l793_793402


namespace circle_y_axis_intersection_range_l793_793428

theorem circle_y_axis_intersection_range (m : ℝ) : (4 - 4 * (m + 6) > 0) → (-2 < 0) → (m + 6 > 0) → (-6 < m ∧ m < -5) :=
by 
  intros h1 h2 h3 
  sorry

end circle_y_axis_intersection_range_l793_793428


namespace num_sets_satisfying_union_l793_793822

theorem num_sets_satisfying_union : 
  ∃! (A : Set ℕ), ({1, 3} ∪ A = {1, 3, 5}) :=
by
  sorry

end num_sets_satisfying_union_l793_793822


namespace find_y_value_l793_793415

-- Define the given conditions and the final question in Lean
theorem find_y_value (k : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : y = k * x ^ (1/3)) 
  (h2 : y = 4 * real.sqrt 3)
  (x1 : x = 64) 
  : ∃ k, y = 2 * real.sqrt 3 :=
sorry

end find_y_value_l793_793415


namespace two_digit_numbers_less_than_35_l793_793357

theorem two_digit_numbers_less_than_35 : 
  let count : ℕ := (99 - 10 + 1) - (7 * 5)
  count = 55 :=
by
  -- definition of total number of two-digit numbers
  let total_two_digit_numbers : ℕ := 99 - 10 + 1
  -- definition of the number of unsuitable two-digit numbers
  let unsuitable_numbers : ℕ := 7 * 5
  -- definition of the count of suitable two-digit numbers
  let count : ℕ := total_two_digit_numbers - unsuitable_numbers
  -- verifying the final count
  exact rfl

end two_digit_numbers_less_than_35_l793_793357


namespace sector_perimeter_l793_793283

theorem sector_perimeter (R : ℝ) (α : ℝ) (A : ℝ) (P : ℝ) : 
  A = (1 / 2) * R^2 * α → 
  α = 4 → 
  A = 2 → 
  P = 2 * R + R * α → 
  P = 6 := 
by
  intros hArea hAlpha hA hP
  sorry

end sector_perimeter_l793_793283


namespace range_of_a_l793_793046

noncomputable def f (a x : ℝ) : ℝ :=
  (1/2) * x^2 + x - 2 * ln x + a

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Ioo 0 2, f a x = 0) ↔ (a = -3 / 2 ∨ a ≤ 2 * ln 2 - 4) :=
begin
  sorry
end

end range_of_a_l793_793046


namespace correct_operation_l793_793132

theorem correct_operation :
  (∀ a : ℝ, a^4 * a^3 = a^7)
  ∧ (∀ a : ℝ, (a^2)^3 ≠ a^5)
  ∧ (∀ a : ℝ, 3 * a^2 - a^2 ≠ 2)
  ∧ (∀ a b : ℝ, (a - b)^2 ≠ a^2 - b^2) :=
by {
  sorry
}

end correct_operation_l793_793132


namespace time_interval_for_birth_and_death_rates_l793_793446

theorem time_interval_for_birth_and_death_rates
  (birth_rate : ℝ)
  (death_rate : ℝ)
  (population_net_increase_per_day : ℝ)
  (number_of_minutes_per_day : ℝ)
  (net_increase_per_interval : ℝ)
  (time_intervals_per_day : ℝ)
  (time_interval_in_minutes : ℝ):

  birth_rate = 10 →
  death_rate = 2 →
  population_net_increase_per_day = 345600 →
  number_of_minutes_per_day = 1440 →
  net_increase_per_interval = birth_rate - death_rate →
  time_intervals_per_day = population_net_increase_per_day / net_increase_per_interval →
  time_interval_in_minutes = number_of_minutes_per_day / time_intervals_per_day →
  time_interval_in_minutes = 48 :=
by
  intros
  sorry

end time_interval_for_birth_and_death_rates_l793_793446


namespace triangle_perimeter_l793_793050

theorem triangle_perimeter (r AP PB x : ℕ) (h_r : r = 14) (h_AP : AP = 20) (h_PB : PB = 30) (h_BC_gt_AC : ∃ BC AC : ℝ, BC > AC)
: ∃ s : ℕ, s = (25 + x) → 2 * s = 50 + 2 * x :=
by
  sorry

end triangle_perimeter_l793_793050


namespace bells_toll_together_l793_793734

theorem bells_toll_together : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 3 5) 8) 11) 15) 20 = 1320 := by
  sorry

end bells_toll_together_l793_793734


namespace david_weighted_average_l793_793387

noncomputable def weighted_average (marks weights : List ℝ) : ℝ :=
  (List.sum (List.zipWith (*) marks weights)) / (List.sum weights)

theorem david_weighted_average :
  let marks := [86, 85, 82, 87, 85]
  let weights := [2, 3, 4, 3, 2]
  weighted_average marks weights ≈ 84.71 :=
by
  let marks := [86, 85, 82, 87, 85]
  let weights := [2, 3, 4, 3, 2]
  show weighted_average marks weights ≈ 84.71
  sorry

end david_weighted_average_l793_793387


namespace middle_three_cards_sum_l793_793487

-- Definitions from the conditions
def green_cards : List ℕ := [1, 2, 3, 4, 5, 6]
def blue_cards : List ℕ := [4, 5, 6, 7]
def alternates_and_divisibility_holds (stack : List ℕ) : Prop :=
  stack.length = 10 ∧
  (∀ (i : ℕ), i < (stack.length - 1) → 
    ((stack.get! i).toString().startsWith('G') ∧ (stack.get! (i+1)).toString().startsWith('B') ∨ 
     (stack.get! i).toString().startsWith('B') ∧ (stack.get! (i+1)).toString().startsWith('G')) ∧
     -- divisibility rule for green and blue numbers
     (stack.get! i).toString().startsWith('G') → ((stack.get! i) + 2) ∣ (stack.get! (i+1)))

-- Lean statement
theorem middle_three_cards_sum :
  ∃ (stack : List ℕ), alternates_and_divisibility_holds stack ∧ 
  stack.head! = 1 ∧
  (stack.drop 4).take 3 = [5, 7, 2] ∧
  (stack.drop 4).take 3.sum = 14 :=
by
  -- Construction of stack will be proof part
  sorry

end middle_three_cards_sum_l793_793487


namespace smallest_m_exists_l793_793266

theorem smallest_m_exists (n : ℕ) (hn : n ≥ 5) : 
  ∃ m (A B : finset ℤ), 
    A.card = n ∧ B.card = m ∧ A ⊆ B ∧ 
    (∀ x y ∈ B, x ≠ y → (x + y ∈ B ↔ x ∈ A ∧ y ∈ A)) ∧ m = 3*n - 3 :=
by
  sorry

end smallest_m_exists_l793_793266


namespace part1_part2_l793_793774

-- Definitions of points and conic sections
structure Point :=
(x : ℝ)
(y : ℝ)

structure ConicSection :=
(focus : Point)
-- Other parameters defining the conic, abstracted for now

def is_on_conic (P : Point) (conic : ConicSection) : Prop := sorry

def angle (P Q R : Point) : ℝ := sorry -- Placeholder for the definition of an angle

variable (F : Point) -- The focus
variable (P Q : Point) -- Moving points on the conic section
variable (conic : ConicSection) -- The conic section
variable (δ : ℝ) -- An acute angle

-- Conditions from the problem
axiom (h1 : conic.focus = F)
axiom (h2 : is_on_conic P conic)
axiom (h3 : is_on_conic Q conic)
axiom (h4 : angle P F Q = 2 * δ) -- Angle at focus F, between P and Q

-- Proof Problems (stored as theorems to be proved)
theorem part1 : ∃ conic' : ConicSection, conic'.focus = F ∧
  (∀ P Q : Point, is_on_conic P conic → is_on_conic Q conic → 
  intersection_locus P Q = conic') := sorry

theorem part2 : ∃ conic' : ConicSection, conic'.focus = F ∧ 
  (∀ P Q : Point, is_on_conic P conic → is_on_conic Q conic → 
  is_tangent_to_conic P Q conic') := sorry

end part1_part2_l793_793774


namespace arctan_sum_l793_793206

theorem arctan_sum (x y : ℝ) (hx : x = 3) (hy : y = 7) :
  Real.arctan (x / y) + Real.arctan (y / x) = Real.pi / 2 := 
by
  rw [hx, hy]
  sorry

end arctan_sum_l793_793206


namespace y_at_x_equals_8_l793_793406

theorem y_at_x_equals_8 (k : ℝ) (h1 : ∀ x y, y = k * x^(1/3))
    (h2 : 4 * real.sqrt 3 = k * 64^(1/3)) : k * 8^(1/3) = 2 * real.sqrt 3 :=
by
  sorry

end y_at_x_equals_8_l793_793406


namespace number_of_circles_for_third_vertex_l793_793615

theorem number_of_circles_for_third_vertex (v1 v2 : ℝ × ℝ × ℝ) 
  (is_vertex_of_cube : ∃ v3 v4 v5 v6 v7 v8 : ℝ × ℝ × ℝ, 
    set.of_cubes_with_vertices {v1, v2, v3, v4, v5, v6, v7, v8}) : 
  ∃ n : ℕ, n = 10 :=
by
  sorry

end number_of_circles_for_third_vertex_l793_793615


namespace find_y_l793_793060

-- Define the problem with the conditions
def rectangle_vertices := { v : ℝ × ℝ // 
  v = (-2, y) ∨ v = (10, y) ∨ v = (-2, 4) ∨ v = (10, 4) }

def rectangle_area (x1 y1 x2 y2 : ℝ) (h : { v : ℝ × ℝ // 
  v = (x1, y1) ∧ v = (x2, y1) ∧ v = (x1, y2) ∧ v = (x2, y2) }) : ℝ :=
  (x2 - x1) * (y2 - y1)

noncomputable def y_positive (y : ℝ) : Prop := y > 0

theorem find_y 
  (y : ℝ)
  (h1 : rectangle_vertices)
  (h2 : rectangle_area (-2) y 10 4 {
     val := (0, 0),
     property := sorry
  } = 108)
  (h3 : y_positive y) : 
  y = 13 := sorry

end find_y_l793_793060


namespace liquidX_percentage_l793_793185

variable (wA wB : ℝ) (pA pB : ℝ) (mA mB : ℝ)

-- Conditions
def weightA : ℝ := 200
def weightB : ℝ := 700
def percentA : ℝ := 0.8
def percentB : ℝ := 1.8

-- The question and answer.
theorem liquidX_percentage :
  (percentA / 100 * weightA + percentB / 100 * weightB) / (weightA + weightB) * 100 = 1.58 := by
  sorry

end liquidX_percentage_l793_793185


namespace hotel_rolls_l793_793664

theorem hotel_rolls (m n : ℕ) (rel_prime : Nat.gcd m n = 1) : 
  let num_nut_rolls := 3
  let num_cheese_rolls := 3
  let num_fruit_rolls := 3
  let total_rolls := 9
  let num_guests := 3
  let rolls_per_guest := 3
  let probability_first_guest := (3 / 9) * (3 / 8) * (3 / 7)
  let probability_second_guest := (2 / 6) * (2 / 5) * (2 / 4)
  let probability_third_guest := 1
  let overall_probability := probability_first_guest * probability_second_guest * probability_third_guest
  overall_probability = (9 / 70) → m = 9 ∧ n = 70 → m + n = 79 :=
by
  intros
  sorry

end hotel_rolls_l793_793664


namespace modulus_one_of_complex_eq_l793_793308

noncomputable def z := ℂ

def z9_relation (z : ℂ) : Prop :=
  z^9 = (11 - 10 * complex.I * z) / (11 * z + 10 * complex.I)

theorem modulus_one_of_complex_eq (z : ℂ) (h1 : 11 * z^10 + 10 * complex.I * z^9 + 10 * complex.I * z - 11 = 0) (h2 : z9_relation z) : abs z = 1 :=
sorry

end modulus_one_of_complex_eq_l793_793308


namespace beads_bracelet_rotational_symmetry_l793_793872

theorem beads_bracelet_rotational_symmetry :
  let n := 8
  let factorial := Nat.factorial
  (factorial n / n = 5040) := by
  sorry

end beads_bracelet_rotational_symmetry_l793_793872


namespace shirt_cost_l793_793400

variables (J S : ℝ)

theorem shirt_cost :
  (3 * J + 2 * S = 69) ∧
  (2 * J + 3 * S = 86) →
  S = 24 :=
by
  sorry

end shirt_cost_l793_793400


namespace relationship_l793_793451

-- Define sequences
variable (a b : ℕ → ℝ)

-- Define conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, 1 < m → m < n → a m = a 1 + (m - 1) * (a n - a 1) / (n - 1)

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, 1 < m → m < n → b m = b 1 * (b n / b 1)^(m - 1) / (n - 1)

noncomputable def sequences_conditions : Prop :=
  a 1 = b 1 ∧ a 1 > 0 ∧ ∀ n, a n = b n ∧ b n > 0

-- The main theorem
theorem relationship (h: sequences_conditions a b) : ∀ m n : ℕ, 1 < m → m < n → a m ≥ b m := 
by
  sorry

end relationship_l793_793451


namespace find_green_hats_l793_793140

variable (B G : ℕ)

theorem find_green_hats (h1 : B + G = 85) (h2 : 6 * B + 7 * G = 540) :
  G = 30 :=
by
  sorry

end find_green_hats_l793_793140


namespace compute_expression_l793_793715

section
variable (a : ℝ)

theorem compute_expression :
  (-a^2)^3 * a^3 = -a^9 :=
sorry
end

end compute_expression_l793_793715


namespace range_of_a_range_of_m_l793_793287

noncomputable def exists_extreme_points (a : ℝ) : Prop :=
  (∃ x1 x2 : ℝ, (f a x1) = 0 ∧ (f a x2) = 0 ∧ x1 ≠ x2 ∧ x1 * x2 > 1/2) 

noncomputable def satisfies_condition (a m : ℝ) : Prop :=
  ∀ x0 ∈ Set.Icc (1 + Real.sqrt 2 / 2) 2, 
    ((1/2 * a * x0^2 - 2 * a * x0 + Real.log x0) + Real.log (a + 1) > m * (a^2 - 1) - (a + 1) + 2 * Real.log 2)

theorem range_of_a (a : ℝ) : exists_extreme_points a → 1 < a ∧ a < 2 := sorry

theorem range_of_m (a : ℝ) (h : 1 < a ∧ a < 2) : satisfies_condition a m → m ≤ -1/4 := sorry

end range_of_a_range_of_m_l793_793287


namespace bankers_discount_l793_793039

theorem bankers_discount {TD S BD : ℝ} (hTD : TD = 66) (hS : S = 429) :
  BD = (TD * S) / (S - TD) → BD = 78 :=
by
  intros h
  rw [hTD, hS] at h
  sorry

end bankers_discount_l793_793039


namespace acute_angle_of_line_l793_793798

theorem acute_angle_of_line :
  (∃ t : ℝ, ∀ x y : ℝ, x = t - 3 ∧ y = sqrt 3 * t) → 
  ∃ α : ℝ, α = 60 :=
by
  sorry

end acute_angle_of_line_l793_793798


namespace adults_on_bus_l793_793071

-- Definitions
def total_passengers := 60
def children_percentage := 0.25
def adults_percentage := 1 - children_percentage
def adults := adults_percentage * total_passengers

-- Theorem statement
theorem adults_on_bus (h: total_passengers = 60) (h₁: children_percentage = 0.25) : adults = 45 := 
by
  -- (Insert proof here)
  sorry

end adults_on_bus_l793_793071


namespace total_marbles_after_2000th_move_l793_793002

/-- Prove the total number of marbles in the baskets after the 2000th move is 16.
Louis has an infinite number of marbles and empty baskets.
Each basket can hold up to five marbles.
Baskets are arranged from left to right in an infinite row.
Initially, Louis places one marble in the leftmost basket.
For each subsequent step, he places a marble in the leftmost basket that can still accept a marble then empties any basket to its left.
-/
theorem total_marbles_after_2000th_move : 
  let basket_capacity := 5 in
  let move_number := 2000 in
  let marbles_sum := (to_digits 6 (move_number)).foldr (λ x acc, x + acc) 0 in
  marbles_sum = 16 :=
by
  sorry

end total_marbles_after_2000th_move_l793_793002


namespace area_triangle_DMC_l793_793870

-- Define the equilateral triangle ABC with side length 4
variables {A B C D M : Type} 

-- Define the segment lengths and the relationship between the points on the segments AC and BC
variables {side_len : ℝ} (h_side_len : side_len = 4)
variables {AM_len MC_len BD_len DC_len : ℝ}
  (hAM : AM_len = 3) (hMC : MC_len = 1)
  (hBD : BD_len = 3) (hDC : DC_len = 1)

-- Define the area to prove and the specific angle
theorem area_triangle_DMC : ( (√3) / 4 ) = 
  (1 / 2) * MC_len * DC_len * real.sin (60 * (real.pi / 180)) :=
by 
  sorry

end area_triangle_DMC_l793_793870


namespace find_x_for_vectors_l793_793809

theorem find_x_for_vectors
  (x : ℝ)
  (h1 : x ∈ Set.Icc 0 Real.pi)
  (a : ℝ × ℝ := (Real.cos (3 * x / 2), Real.sin (3 * x / 2)))
  (b : ℝ × ℝ := (Real.cos (x / 2), -Real.sin (x / 2)))
  (h2 : (a.1 + b.1)^2 + (a.2 + b.2)^2 = 1) :
  x = Real.pi / 3 ∨ x = 2 * Real.pi / 3 :=
by
  sorry

end find_x_for_vectors_l793_793809


namespace sin_cos_sum_sin_double_minus_pi_over_4_l793_793759

variable (α : ℝ)
variable (h₀ : 0 < α ∧ α < π / 2)
variable (h₁ : cos (2 * π - α) - sin (π - α) = -sqrt 5 / 5)

theorem sin_cos_sum (h₀ : 0 < α ∧ α < π / 2) (h₁ : cos (2 * π - α) - sin (π - α) = -sqrt 5 / 5) :
  sin α + cos α = 3 * sqrt 5 / 5 :=
  sorry

theorem sin_double_minus_pi_over_4 (h₀ : 0 < α ∧ α < π / 2) (h₁ : cos (2 * π - α) - sin (π - α) = -sqrt 5 / 5) :
  sin (2 * α - π / 4) = 7 * sqrt 2 / 10 :=
  sorry

end sin_cos_sum_sin_double_minus_pi_over_4_l793_793759


namespace total_tiles_in_floor_l793_793453

theorem total_tiles_in_floor : 
  ∀ (n : ℕ) (a : ℕ), n = 9 → a = 53 → (Σ (i : ℕ) in (finset.range n), a - 2 * i) = 405 :=
by
  sorry

end total_tiles_in_floor_l793_793453


namespace ratio_of_perimeters_l793_793546

theorem ratio_of_perimeters (s : ℝ) :
  let folded_width := s / 2,
      smaller_rectangle_width := folded_width / 3,
      larger_rectangle_width := 2 * folded_width / 3,
      smaller_perimeter := 2 * (smaller_rectangle_width + s),
      larger_perimeter := 2 * (larger_rectangle_width + s)
  in smaller_perimeter / larger_perimeter = 7 / 8 :=
by
  sorry

end ratio_of_perimeters_l793_793546


namespace local_language_letters_l793_793860

theorem local_language_letters (n : ℕ) (h : 1 + 2 * n = 139) : n = 69 :=
by
  -- Proof skipped
  sorry

end local_language_letters_l793_793860


namespace repeating_decimal_sum_l793_793107

theorem repeating_decimal_sum (x : ℚ) (h : x = 34 / 999) : x.num + x.denom = 1033 := by 
  sorry

end repeating_decimal_sum_l793_793107


namespace variance_unchanged_by_constant_subtraction_l793_793956

def variance (l : List ℝ) : ℝ :=
  let mean := l.sum / l.length
  (l.map (λ x => (x - mean)^2)).sum / l.length

theorem variance_unchanged_by_constant_subtraction (ages : List ℝ) (c : ℝ) :
  variance (ages) = variance (ages.map (λ x => x - c)) := by
  sorry

end variance_unchanged_by_constant_subtraction_l793_793956


namespace cube_distinguishable_colorings_l793_793660

theorem cube_distinguishable_colorings (total_colors faces : ℕ) (h_colors : total_colors = 10) (h_faces : faces = 6) : 
  ∃ n : ℕ, n = 30240 :=
by
  use 30240
  sorry

end cube_distinguishable_colorings_l793_793660


namespace find_b_for_continuity_l793_793910

def f (x : ℝ) (b : ℝ) : ℝ := 
  if x ≤ 5 then 
    2 * x^2 + 3 * x + 1 
  else 
    b * x + 2

theorem find_b_for_continuity :
  (∃ b : ℝ, ∀ x : ℝ, (f x b)  = 
    if x ≤ 5 then 
      2 * x^2 + 3 * x + 1 
    else 
      b * x + 2) ∧ (∀ (h1 : ℝ), h1 ≤ 5 → (2 * 5^2 + 3 * 5 + 1) = (b * 5 + 2)) :=
  sorry

end find_b_for_continuity_l793_793910


namespace area_of_triangle_ABC_l793_793856

theorem area_of_triangle_ABC (A B C D M : Type*) 
  [IsTriangle (A, B, C)] (right_angle_at_C : ∠ACB = 90°)
  (median_AM : IsMedian A B C M)
  (altitude_BD : IsAltitude B A C D)
  (angle_AM_AC : ∠MAM = 15°)
  (area_AMD : Area (A, M, D) = K) :
  Area (A, B, C) = 4 * K / Real.cot 15 :=
sorry

end area_of_triangle_ABC_l793_793856


namespace range_of_m_l793_793433

theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ (x^3 - 3 * x + m = 0)) → (m ≥ -2 ∧ m ≤ 2) :=
sorry

end range_of_m_l793_793433


namespace cheryl_needed_second_material_l793_793051

noncomputable def cheryl_material : Prop :=
  let first_material := 3 / 8
  let leftover_material := 15 / 40
  let used_material := 0.33333333333333326
  let total_material_bought := used_material + leftover_material
  let second_material := total_material_bought - first_material
  second_material ≈ 0.3333333333333333

theorem cheryl_needed_second_material : cheryl_material :=
by
  sorry

end cheryl_needed_second_material_l793_793051


namespace multiplication_addition_l793_793650

theorem multiplication_addition :
  108 * 108 + 92 * 92 = 20128 :=
by
  sorry

end multiplication_addition_l793_793650


namespace tanya_clock_l793_793548

theorem tanya_clock (hours_passed : ℕ) (slow_rate : ℕ) : 
  (hours_passed = 6) → (slow_rate = 5) → 
  let correct_time := 6 * slow_rate + 6 * 60 in
  correct_time = 6 * 60 + 30 :=
by
  sorry

end tanya_clock_l793_793548


namespace atomic_weight_Ba_l793_793240

-- Definitions based on the conditions
def molecular_weight_BaCl2 : ℝ := 207
def atomic_weight_Cl : ℝ := 35.45

-- The question, rephrased as a theorem to prove
theorem atomic_weight_Ba : atomic_weight_Ba def molecular_weight_BaCl2 def atomic_weight_Cl := 136.1 :=
by
  -- Skip the proof
  sorry

end atomic_weight_Ba_l793_793240


namespace range_of_a_l793_793830

-- Definitions derived from conditions
def is_ellipse_with_foci_on_x_axis (a : ℝ) : Prop := a^2 > a + 6 ∧ a + 6 > 0

-- Theorem representing the proof problem
theorem range_of_a (a : ℝ) (h : is_ellipse_with_foci_on_x_axis a) :
  (a > 3) ∨ (-6 < a ∧ a < -2) :=
sorry

end range_of_a_l793_793830


namespace percentile_80_percent_data_is_8_l793_793035

def data_set : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def percentile (p : ℕ) (data : List ℕ) : ℕ :=
  let sorted := data.qsort (· ≤ ·)
  let pos := (p * data.length / 100).toNat
  sorted.getD (pos - 1) 0

theorem percentile_80_percent_data_is_8 :
  percentile 80 data_set = 8 :=
by
  sorry

end percentile_80_percent_data_is_8_l793_793035


namespace solve_equation_1_l793_793541

theorem solve_equation_1.05 :
  ∃ x : ℝ, (sqrt (9 * x - 4) + sqrt (4 * x - 3) = 3) ∧ (x = 1.05) :=
by
  sorry

end solve_equation_1_l793_793541


namespace molecular_weight_CaO_l793_793622

def atomic_weight_Ca : Float := 40.08
def atomic_weight_O : Float := 16.00

def molecular_weight (atoms : List (String × Float)) : Float :=
  atoms.foldr (fun (_, w) acc => w + acc) 0.0

theorem molecular_weight_CaO :
  molecular_weight [("Ca", atomic_weight_Ca), ("O", atomic_weight_O)] = 56.08 :=
by
  sorry

end molecular_weight_CaO_l793_793622


namespace incorrect_option_A_l793_793684

/-- Defining temperature sound speed relationship based on table data -/
def sound_speed_at_temp (t : ℝ) : ℝ :=
  if t = -20 then 318 else
  if t = -10 then 324 else
  if t = 0 then 330 else
  if t = 10 then 336 else
  if t = 20 then 342 else
  if t = 30 then 348 else 0

/-- Proving that Option A is incorrect: sound can travel 1740m in 5s at 20°C -/
theorem incorrect_option_A :
  sound_speed_at_temp 20 * 5 ≠ 1740 := sorry

end incorrect_option_A_l793_793684


namespace length_DE_is_20_l793_793474

noncomputable def length_DE (BC : ℝ) (angle_C_deg : ℝ) 
  (D : ℝ) (is_midpoint_D : D = BC / 2)
  (is_right_triangle : angle_C_deg = 45): ℝ := 
let DE := D in DE

theorem length_DE_is_20 : ∀ (BC : ℝ) (angle_C_deg : ℝ),
  BC = 40 → 
  angle_C_deg = 45 → 
  let D := BC / 2 in 
  let DE := D in 
  DE = 20 :=
by
  intros BC angle_C_deg hBC hAngle
  sorry

end length_DE_is_20_l793_793474


namespace line_relation_l793_793161

-- Definitions based on conditions
variables (l a b : Line)

def parallel (x y : Line) : Prop := ∃ (v : Vector), (x.direction = v ∧ y.direction = v)
def skew (x y : Line) : Prop := ¬ (parallel x y ∨ ∃ p, x.contains p ∧ y.contains p)

-- Given Conditions
axiom h1 : parallel l a
axiom h2 : skew a b

-- Proof goal
theorem line_relation (l a b : Line) (h1 : parallel l a) (h2 : skew a b) : 
  (∃ p, l.contains p ∧ b.contains p) ∨ skew l b :=
sorry

end line_relation_l793_793161


namespace exists_real_m_l793_793319

noncomputable def f (a : ℝ) (x : ℝ) := 4 * x + a * x ^ 2 - (2 / 3) * x ^ 3
noncomputable def g (x : ℝ) := 2 * x + (1 / 3) * x ^ 3

theorem exists_real_m (a : ℝ) (t : ℝ) (x1 x2 : ℝ) :
  (-1 : ℝ) ≤ a ∧ a ≤ 1 →
  (-1 : ℝ) ≤ t ∧ t ≤ 1 →
  f a x1 = g x1 ∧ f a x2 = g x2 →
  x1 ≠ 0 ∧ x2 ≠ 0 →
  x1 ≠ x2 →
  ∃ m : ℝ, (m ≥ 2 ∨ m ≤ -2) ∧ m^2 + t * m + 1 ≥ |x1 - x2| :=
sorry

end exists_real_m_l793_793319


namespace probability_dice_sum_perfect_square_l793_793601

def is_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def probability_perfect_square : ℚ :=
  12 / 64

theorem probability_dice_sum_perfect_square :
  -- Two standard 8-sided dice are rolled
  -- The probability that the sum rolled is a perfect square is 3/16
  (∃ dice1 dice2 : ℕ, 1 ≤ dice1 ∧ dice1 ≤ 8 ∧ 1 ≤ dice2 ∧ dice2 ≤ 8) →
  probability_perfect_square = 3 / 16 :=
sorry

end probability_dice_sum_perfect_square_l793_793601


namespace arrange_digits_divisible_by_2_to_18_l793_793176

theorem arrange_digits_divisible_by_2_to_18: 
  ∃ n: ℕ, 
  (∃ lst: List ℕ, lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0] ∧
   (lst.permutations.any (λ p, ∃ (m: ℕ) (h: m = p.foldl (λ acc d, acc * 10 + d) 0), 
    (∀ k in (list.range 17).map (+2), m % k = 0)))) := 
begin
  sorry
end

end arrange_digits_divisible_by_2_to_18_l793_793176


namespace num_two_digit_numbers_with_digit_less_than_35_l793_793382

theorem num_two_digit_numbers_with_digit_less_than_35 : 
  let total_two_digit := 90 in
  let unsuitable_numbers := 35 in
  let suitable_numbers := total_two_digit - unsuitable_numbers
  in suitable_numbers = 55 :=
by 
  let total_two_digit := 90
  let unsuitable_numbers := 35
  let suitable_numbers := total_two_digit - unsuitable_numbers
  show suitable_numbers = 55, from sorry

end num_two_digit_numbers_with_digit_less_than_35_l793_793382


namespace possible_value_of_theta_l793_793313

noncomputable def f (x θ : Real) := Real.sin (2 * x + θ) + Real.sqrt 3 * Real.cos (2 * x + θ)

theorem possible_value_of_theta :
  (∀ x : ℝ, 2015 ^ (f (-x) θ) = 1 / 2015 ^ (f x θ)) ∧
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ Real.pi / 4 → f x1 θ > f x2 θ)
  → θ = 2 * Real.pi / 3 :=
by
  sorry

end possible_value_of_theta_l793_793313


namespace always_possible_to_form_n_triangles_l793_793259

def three_n_points_are_not_collinear (n : ℕ) (points : Fin 3n → ℝ × ℝ) : Prop :=
  ∀ (i j k: Fin 3n), ∀ (hij : i ≠ j) (hik : i ≠ k) (hjk : j ≠ k),
    ¬collinear ℝ ({points i, points j, points k} : Set (ℝ × ℝ))

theorem always_possible_to_form_n_triangles (n : ℕ) (points : Fin 3n → ℝ × ℝ)
  (h : three_n_points_are_not_collinear n points) :
  ∃ triangles : Fin n → Fin 3n × Fin 3n × Fin 3n,
    (∀ i j : Fin n, i ≠ j → disjoint (triangles i).triplet (triangles j).triplet) ∧
    (∀ i : Fin 3n, ∃ j : Fin n, i ∈ (triangles j).triplet)
  :=
sorry

end always_possible_to_form_n_triangles_l793_793259


namespace complement_P_in_U_l793_793807

noncomputable def U := Set.univ

noncomputable def P := {y : ℝ | ∃ x : ℝ, 0 < x ∧ x < 1 ∧ y = 1 / x}

theorem complement_P_in_U : (U \ P) = {y : ℝ | y ≤ 1} :=
by
  sorry

end complement_P_in_U_l793_793807


namespace problem_1_problem_2_l793_793800

theorem problem_1 (n : ℕ) (h : n > 0) (a : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n, (n > 0) → 
    (∃ α β, α + β = β * α + 1 ∧ 
            α * β = 1 / a n ∧ 
            a n * α^2 - a (n+1) * α + 1 = 0 ∧ 
            a n * β^2 - a (n+1) * β + 1 = 0)) :
  a (n + 1) = a n + 1 := sorry

theorem problem_2 (n : ℕ) (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n, (n > 0) → a (n+1) = a n + 1) :
  a n = n := sorry

end problem_1_problem_2_l793_793800


namespace remaining_amount_l793_793736

def initial_amount : ℕ := 18
def spent_amount : ℕ := 16

theorem remaining_amount : initial_amount - spent_amount = 2 := 
by sorry

end remaining_amount_l793_793736


namespace barbata_interest_rate_l793_793182

theorem barbata_interest_rate (r : ℝ) : 
  let initial_investment := 2800
  let additional_investment := 1400
  let total_investment := initial_investment + additional_investment
  let annual_income := 0.06 * total_investment
  let additional_interest_rate := 0.08
  let income_from_initial := initial_investment * r
  let income_from_additional := additional_investment * additional_interest_rate
  income_from_initial + income_from_additional = annual_income → 
  r = 0.05 :=
by
  intros
  sorry

end barbata_interest_rate_l793_793182


namespace problem_l793_793771

-- Define the complex numbers and their conditions
variables (z1 z2 : ℂ)

-- Use the given conditions to create the definitions
def option_A := z1 * conj z1 ≠ abs z1
def option_B := (z1 * z2 = 0) → (z1 = 0 ∨ z2 = 0)
def option_C := conj z1 * conj z2 = conj (z1 * z2)
def option_D := (abs z1 = abs z2) → ¬ (z1^2 = z2^2)

-- Use the conditions to state the problem to prove
theorem problem : option_A z1 z2 ∧ option_B z1 z2 ∧ option_C z1 z2 ∧ option_D z1 z2 :=
by sorry

end problem_l793_793771


namespace number_of_factors_of_M_l793_793336

def M := 2 ^ 6 * 3 ^ 5 * 5 ^ 3 * 7 ^ 1 * 11 ^ 2

theorem number_of_factors_of_M :
  let num_factors := (6 + 1) * (5 + 1) * (3 + 1) * (1 + 1) * (2 + 1)
  in num_factors = 1008 := 
by
  let num_factors := (6 + 1) * (5 + 1) * (3 + 1) * (1 + 1) * (2 + 1)
  show num_factors = 1008,
  sorry

end number_of_factors_of_M_l793_793336


namespace arctan_sum_l793_793211

theorem arctan_sum {a b : ℝ} (h3 : a = 3) (h7 : b = 7) :
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 :=
by
  sorry

end arctan_sum_l793_793211


namespace limit_ln_frac_l793_793644

theorem limit_ln_frac:
  ∀ f : ℝ → ℝ,
  (∀ x, f x = (3^(2*x) - 5^(3*x)) / (arctan x + x^3)) →
  (filter.tendsto f (nhds 0) (nhds (Real.log (9/125)))) := by
  intros f hf
  sorry

end limit_ln_frac_l793_793644


namespace parabola_translation_l793_793057

theorem parabola_translation :
  ∀ (x y : ℝ), (y = 2 * (x - 3) ^ 2) ↔ ∃ t : ℝ, t = x - 3 ∧ y = 2 * t ^ 2 :=
by sorry

end parabola_translation_l793_793057


namespace find_x_for_prime_square_l793_793326

theorem find_x_for_prime_square (x p : ℤ) (hp : Prime p) (h : 2 * x^2 - x - 36 = p^2) : x = 13 ∧ p = 17 :=
by
  sorry

end find_x_for_prime_square_l793_793326


namespace unique_c1_c2_exists_l793_793503

theorem unique_c1_c2_exists (a_0 a_1 x_1 x_2 : ℝ) (h_distinct : x_1 ≠ x_2) : 
  ∃! (c_1 c_2 : ℝ), ∀ n : ℕ, a_n = c_1 * x_1^n + c_2 * x_2^n :=
sorry

end unique_c1_c2_exists_l793_793503


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l793_793116

theorem sum_of_numerator_and_denominator_of_repeating_decimal (x : ℚ) (h : x = 34 / 99) : (x.den + x.num : ℤ) = 133 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l793_793116


namespace time_to_pass_bridge_correct_l793_793638

-- Definitions for conditions
def length_of_train : ℝ := 360  -- in meters
def length_of_bridge : ℝ := 140 -- in meters
def speed_of_train_kmh : ℝ := 56 -- in km/h

-- Convert the speed of train from km/h to m/s
def kmh_to_ms (speed : ℝ) : ℝ := (speed * 1000) / 3600
def speed_of_train_ms : ℝ := kmh_to_ms speed_of_train_kmh

-- Calculate the total distance
def total_distance : ℝ := length_of_train + length_of_bridge

-- Calculate the time required to pass the bridge
def time_to_pass_bridge : ℝ := total_distance / speed_of_train_ms

-- The theorem to prove
theorem time_to_pass_bridge_correct : time_to_pass_bridge = 32.14 := 
by
  sorry

end time_to_pass_bridge_correct_l793_793638


namespace problem_I3_1_l793_793278

theorem problem_I3_1 (w x y z : ℝ) (h1 : w * x * y * z = 4) (h2 : w - x * y * z = 3) (h3 : w > 0) : 
  w = 4 :=
by
  sorry

end problem_I3_1_l793_793278


namespace find_nth_number_in_s_l793_793911

def s (k : ℕ) : ℕ := 8 * k + 5

theorem find_nth_number_in_s (n : ℕ) (number_in_s : ℕ) (h : number_in_s = 573) :
  ∃ k : ℕ, s k = number_in_s ∧ n = k + 1 := 
sorry

end find_nth_number_in_s_l793_793911


namespace maria_original_number_25_3_l793_793512

theorem maria_original_number_25_3 (x : ℚ) 
  (h : ((3 * (x + 3) - 4) / 3) = 10) : 
  x = 25 / 3 := 
by 
  sorry

end maria_original_number_25_3_l793_793512


namespace length_DE_l793_793461

-- Definition and given conditions
variables {A B C D E : Type}
variables [Point (Triangle ABC)]

-- Given BC = 40 and angle C = 45 degrees
constants (BC : Real) (angleC : Real)
constants (midpoint : BC → D) (perpendicular_bisector : BC → AC → E)
constant (triangle_CDE_454590 : Is454590Triangle C D E)

-- Definitions for points D and E
noncomputable def midpoint_of_BC (P: BC) : D :=
  midpoint P

noncomputable def intersection_perpendicular_bisector_AC (P: BC → AC) : E :=
  perpendicular_bisector P

-- Prove length of DE == 20
theorem length_DE : length DE = 20 := sorry

end length_DE_l793_793461


namespace vector_AD_eq_l793_793273

open Real

variables {Point : Type} [add_group Point] [module ℝ Point] 

def vec (P Q : Point) : Point := Q - P

variables (A B C D : Point)

theorem vector_AD_eq :
  vec A D = (2 / 3) • vec A B + (1 / 3) • vec A C :=
begin
  have h1 : vec B C = 3 • vec B D,
  { sorry },
  have h2 : vec B C = vec B A + vec A C,
  { sorry },
  -- other steps go here, concluding with the desired result
  sorry
end

end vector_AD_eq_l793_793273


namespace vector_magnitude_l793_793843

variable {V : Type*} [InnerProductSpace ℝ V]

theorem vector_magnitude
  (a b : V)
  (h1 : ‖a‖ = 3)
  (h2 : ‖a - b‖ = 5)
  (h3 : inner a b = 1) :
  ‖b‖ = 3 * Real.sqrt 2 :=
by
  sorry

end vector_magnitude_l793_793843


namespace part_a_part_b_part_c_l793_793667

-- Part (a)
theorem part_a (A B O : Point) :
  intersects_y_axis (Line.equation 2 3 12) A →
  intersects_x_axis (Line.equation 2 3 12) B →
  A = Point.mk 0 4 →
  B = Point.mk 6 0 →
  Triangle.area A B O = 12 :=
sorry

-- Part (b)
theorem part_b (D E O : Point) (c : ℝ) :
  intersects_y_axis (Line.equation 6 5 c) D →
  intersects_x_axis (Line.equation 6 5 c) E →
  area_of_triangle D E O = 240 →
  D = Point.mk 0 (c / 5) →
  E = Point.mk (c / 6) 0 →
  c = 120 :=
sorry

-- Part (c)
theorem part_c (P Q S R : Point) (m n : ℤ) :
  100 ≤ m ∧ m < n →
  intersects_y_axis (Line.equation (2 * m) 1 (4 * m)) P →
  intersects_x_axis (Line.equation (2 * m) 1 (4 * m)) Q →
  intersects_y_axis (Line.equation (7 * n) 4 (28 * n)) S →
  intersects_x_axis (Line.equation (7 * n) 4 (28 * n)) R →
  quadrilateral_area P Q S R = 2022 →
  (m, n) = (100, 173) ∨ (m, n) = (107, 175) :=
sorry

end part_a_part_b_part_c_l793_793667


namespace weight_of_cut_piece_is_4_l793_793576

noncomputable def weight_first_ingot : ℝ := 6
noncomputable def weight_second_ingot : ℝ := 12
noncomputable def fraction_tin_first_ingot : ℝ := a
noncomputable def fraction_tin_second_ingot : ℝ := b
noncomputable def weight_cut_piece : ℝ := x

-- Conditions
axiom h1 : a ≠ b

-- Equation derived from the equality of tin fractions in the new ingots
axiom h2 : (b * x + a * (weight_first_ingot - x)) / weight_first_ingot = 
          (a * x + b * (weight_second_ingot - x)) / weight_second_ingot

theorem weight_of_cut_piece_is_4 : x = 4 := 
sorry

end weight_of_cut_piece_is_4_l793_793576


namespace probability_cheryl_same_color_l793_793151

theorem probability_cheryl_same_color
    (R G Y : ℕ)
    (carol_draws : {marbles // marbles ⊆ {R, G, Y} ∧ marbles.card = 3})
    (claudia_draws : {marbles // marbles ⊆ (Set.diff {R, G, Y} carol_draws) ∧ marbles.card = 3})
    (cheryl_marble_color : {marbles // marbles = Set.diff (Set.diff {R, G, Y} carol_draws) claudia_draws ∧ marbles.card = 3}) :
    R = 3 → G = 3 → Y = 3 → 
    (Probability_cheryl_same_color cheryl_marble_color = 1 / 28) := 
by
  intros hR hG hY
  sorry

end probability_cheryl_same_color_l793_793151


namespace difference_between_shares_l793_793640

theorem difference_between_shares (ratio : ℕ → ℕ) (share_vasim rs : ℕ) :
  ratio 1 = 3 → ratio 2 = 5 → ratio 3 = 6 →
  share_vasim = 1500 →
  rs = 900 := by
slope

end difference_between_shares_l793_793640


namespace no_solution_fib_eq_l793_793974

def fib : ℕ → ℕ
| 0 := 1
| 1 := 1
| (n + 2) := fib (n + 1) + fib n

theorem no_solution_fib_eq :
  ∀ n : ℕ, n * fib n * fib (n - 1) ≠ (fib (n + 2) - 1)^2 :=
by sorry

end no_solution_fib_eq_l793_793974


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l793_793120

theorem sum_of_numerator_and_denominator_of_repeating_decimal (x : ℚ) (h : x = 34 / 99) : (x.den + x.num : ℤ) = 133 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l793_793120


namespace num_suitable_two_digit_numbers_l793_793376

/-- 
How many two-digit numbers have at least one digit that is smaller than the corresponding digit in the number 35?
-/
theorem num_suitable_two_digit_numbers : 
  let two_digit_numbers := { n : ℕ | 10 ≤ n ∧ n ≤ 99 },
      suitable_numbers := { n ∈ two_digit_numbers | (n / 10 < 3) ∨ (n % 10 < 5) } in
  suitable_numbers.card = 55 :=
by
  sorry

end num_suitable_two_digit_numbers_l793_793376


namespace probability_dice_sum_perfect_square_l793_793603

def is_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def probability_perfect_square : ℚ :=
  12 / 64

theorem probability_dice_sum_perfect_square :
  -- Two standard 8-sided dice are rolled
  -- The probability that the sum rolled is a perfect square is 3/16
  (∃ dice1 dice2 : ℕ, 1 ≤ dice1 ∧ dice1 ≤ 8 ∧ 1 ≤ dice2 ∧ dice2 ≤ 8) →
  probability_perfect_square = 3 / 16 :=
sorry

end probability_dice_sum_perfect_square_l793_793603


namespace good_carrots_total_l793_793718

-- Define the number of carrots picked by Carol and her mother
def carolCarrots := 29
def motherCarrots := 16

-- Define the number of bad carrots
def badCarrots := 7

-- Define the total number of carrots picked by Carol and her mother
def totalCarrots := carolCarrots + motherCarrots

-- Define the total number of good carrots
def goodCarrots := totalCarrots - badCarrots

-- The theorem to prove that the total number of good carrots is 38
theorem good_carrots_total : goodCarrots = 38 := by
  sorry

end good_carrots_total_l793_793718


namespace value_of_y_at_x8_l793_793409

theorem value_of_y_at_x8
  (k : ℝ)
  (y : ℝ → ℝ)
  (hx64 : y 64 = 4 * Real.sqrt 3)
  (hy_def : ∀ x, y x = k * x^(1 / 3)) :
  y 8 = 2 * Real.sqrt 3 :=
by {
  sorry,
}

end value_of_y_at_x8_l793_793409


namespace cricket_innings_count_l793_793155

theorem cricket_innings_count (n : ℕ) (h_avg_current : ∀ (total_runs : ℕ), total_runs = 32 * n)
  (h_runs_needed : ∀ (total_runs : ℕ), total_runs + 116 = 36 * (n + 1)) : n = 20 :=
by
  sorry

end cricket_innings_count_l793_793155


namespace set_subset_find_m_l793_793330

open Set

def A (m : ℝ) : Set ℝ := {1, 3, 2 * m + 3}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem set_subset_find_m (m : ℝ) : (B m ⊆ A m) → (m = 1 ∨ m = 3) :=
by 
  intro h
  sorry

end set_subset_find_m_l793_793330


namespace length_DE_is_20_l793_793457

open_locale real
open_locale complex_conjugate

noncomputable def in_triangle_config_with_angle (ABC : Type) [triangle ABC]
  (BC_length : ℝ) (angle_C : real.angle) (perp_bisector_intersect_D : point)
  (perp_bisector_intersect_E : point) : Prop :=
  -- define the conditions
  BC_length = 40 ∧
  angle_C = 45 ∧
  is_midpoint_of_BC ABC perp_bisector_intersect_D ∧
  is_perpendicular_bisector_intersects_AC ABC perp_bisector_intersect_D perp_bisector_intersect_E

noncomputable def length_DE (D E: point) : ℝ :=
  distance_between D E

theorem length_DE_is_20 {ABC : Type} [triangle ABC]
  (BC_length : ℝ) (angle_C : real.angle) (perp_bisector_intersect_D : point)
  (perp_bisector_intersect_E : point)
  (h : in_triangle_config_with_angle ABC BC_length angle_C perp_bisector_intersect_D perp_bisector_intersect_E):
  length_DE perp_bisector_intersect_D perp_bisector_intersect_E = 20 :=
begin
  sorry,
end

end length_DE_is_20_l793_793457


namespace length_intersection_chord_l793_793780

-- Definitions for hyperbola parameters
variables {a b : ℝ} (ha : 0 < a) (hb : 0 < b)

-- Definition of the given hyperbola
def hyperbola := ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)

-- Condition on the eccentricity of the hyperbola
def eccentricity := ∀ a b, sqrt (1 + b^2 / a^2) = sqrt 5

-- Definition of the given circle
def circle := ∀ x y : ℝ, (x - 2)^2 + (y - 3)^2 = 1

-- Proof problem to find the length of intersection chord
theorem length_intersection_chord (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (Hh : hyperbola a b) (He : eccentricity a b) (Hc : circle) : 
    ∀ AB : ℝ, AB = 4 * sqrt 5 / 5 :=
sorry

end length_intersection_chord_l793_793780


namespace intersection_P_Q_l793_793025

def P : Set ℝ := { x | x > 1 }
def Q : Set ℝ := { x | f x = Real.log (2 - x) }

theorem intersection_P_Q : P ∩ Q = { x | 1 < x ∧ x < 2 } :=
by
  sorry

end intersection_P_Q_l793_793025


namespace intersect_log5_dist_1_l793_793157

theorem intersect_log5_dist_1 (k a b : ℝ) (h1 : k = a + Real.sqrt b) 
  (h2 : (λ x, log 5 x) k = log 5 k) 
  (h3 : (λ x, log 5 (x + 8)) k = log 5 (k + 8))
  (h4 : abs (log 5 k - log 5 (k + 8)) = 1) 
  (a_int : a ∈ ℤ) (b_int : b ∈ ℤ) : 
  a + b = 2 :=
sorry

end intersect_log5_dist_1_l793_793157


namespace smallest_next_divisor_l793_793484

theorem smallest_next_divisor (m : ℕ) (h1 : 1000 ≤ m ∧ m < 10000) (h2 : m % 2 = 0) (h3 : 427 ∣ m) :
  ∃ d : ℕ, d > 427 ∧ d ∣ m ∧ ∀ e : ℕ, e > 427 → e ∣ m → d ≤ e ∧ e ≠ d → d = 434 :=
begin
  sorry

end smallest_next_divisor_l793_793484


namespace seq_relation_l793_793062

noncomputable def a_seq : ℕ → ℤ → ℤ → ℤ → ℕ → ℤ
| 0, α, β, γ => 1
| (n+1), α, β, γ => α * a_seq n α β γ + β * b_seq n α β γ

noncomputable def b_seq : ℕ → ℤ → ℤ → ℤ → ℕ → ℤ
| 0, α, β, γ => 1
| (n+1), α, β, γ => β * a_seq n α β γ + γ * b_seq n α β γ

theorem seq_relation 
  (α β γ : ℤ) (h1 : α < γ) (h2 : α * γ = β^2 + 1) :
  ∀ (m n : ℕ), a_seq (m+n) α β γ + b_seq (m+n) α β γ = a_seq m α β γ * a_seq n α β γ + b_seq m α β γ * b_seq n α β γ :=
by
  sorry

end seq_relation_l793_793062


namespace length_DE_is_20_l793_793471

noncomputable def length_DE (BC : ℝ) (angle_C_deg : ℝ) 
  (D : ℝ) (is_midpoint_D : D = BC / 2)
  (is_right_triangle : angle_C_deg = 45): ℝ := 
let DE := D in DE

theorem length_DE_is_20 : ∀ (BC : ℝ) (angle_C_deg : ℝ),
  BC = 40 → 
  angle_C_deg = 45 → 
  let D := BC / 2 in 
  let DE := D in 
  DE = 20 :=
by
  intros BC angle_C_deg hBC hAngle
  sorry

end length_DE_is_20_l793_793471


namespace number_of_correct_statements_l793_793555

theorem number_of_correct_statements :
  (¬ ∃ x : ℝ, sqrt x = x ∧ x ∉ ℚ) → 
  (∀ x : ℝ, -sqrt 2 < x ∧ x < sqrt 5 → (1 ≤ x ∧ x ≤ 2 ∨ 3 ≤ x ∧ x ≤ 4)) →
  (-3 = sqrt 9) →
  (∀ x y : ℝ, x ∉ ℝ → y ∉ ℝ → (x + y) ∉ ℝ) →
  (∃ x : ℝ, ¬(fractional x) ∧ x ∉ ℚ) →
  (∀ a : ℝ, sqrt (a^2) = a) →
  1 = 1 := by
  sorry

end number_of_correct_statements_l793_793555


namespace counterexample_exists_l793_793217

def is_prime (n : ℕ) : Prop :=
  ∀ d, d ∣ n → d = 1 ∨ d = n

def is_composite (n : ℕ) : Prop :=
  ¬is_prime n

theorem counterexample_exists : ∃ n ∈ ({10, 18, 22, 25, 29} : set ℕ), is_composite n ∧ is_composite (n - 3) := by
  sorry

end counterexample_exists_l793_793217


namespace conditional_probabilities_l793_793580

-- Define the events and probabilities
def event_A : set (ℕ × ℕ) := {(x, y) | x + y = 8}
def event_B : set (ℕ × ℕ) := {(x, y) | x < y}

def probability (s : set (ℕ × ℕ)) : ℚ :=
  (s.to_finset.card : ℚ) / 36

def conditional_probability (A B : set (ℕ × ℕ)) : ℚ :=
  probability (A ∩ B) / probability B

theorem conditional_probabilities :
  conditional_probability event_A event_B = 2 / 15 ∧ 
  conditional_probability event_B event_A = 2 / 5 :=
by
  sorry

end conditional_probabilities_l793_793580


namespace odd_divisibility_iff_l793_793937

variables {a b n : ℤ}

theorem odd_divisibility_iff (ha : a % 2 = 1) (hb : b % 2 = 1) (hn : n > 0) : 
  (2^n ∣ (a - b)) ↔ (2^n ∣ (a^3 - b^3)) :=
sorry

end odd_divisibility_iff_l793_793937


namespace solve_inequality_l793_793216

theorem solve_inequality (x : ℝ) :
  (x - 1)^2 < 12 - x ↔ 
  (Real.sqrt 5) ≠ 0 ∧
  (1 - 3 * (Real.sqrt 5)) / 2 < x ∧ 
  x < (1 + 3 * (Real.sqrt 5)) / 2 :=
sorry

end solve_inequality_l793_793216


namespace evaluate_power_function_at_5_l793_793289

-- Given conditions
def power_function (a : ℝ) (x : ℝ) : ℝ := x^a

-- An assumption that a power function passes through the point (2, sqrt 2)
def passes_through (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  f p.1 = p.2

-- The given power function passes through the point (2, sqrt 2)
def given_condition : Prop :=
  passes_through (power_function (1/2)) (2, Real.sqrt 2)

-- The goal to prove
theorem evaluate_power_function_at_5 (h : given_condition) : power_function (1/2) 5 = Real.sqrt 5 :=
sorry

end evaluate_power_function_at_5_l793_793289


namespace num_two_digit_numbers_with_digit_less_than_35_l793_793384

theorem num_two_digit_numbers_with_digit_less_than_35 : 
  let total_two_digit := 90 in
  let unsuitable_numbers := 35 in
  let suitable_numbers := total_two_digit - unsuitable_numbers
  in suitable_numbers = 55 :=
by 
  let total_two_digit := 90
  let unsuitable_numbers := 35
  let suitable_numbers := total_two_digit - unsuitable_numbers
  show suitable_numbers = 55, from sorry

end num_two_digit_numbers_with_digit_less_than_35_l793_793384


namespace tiling_remainder_336_l793_793652

-- Define the board size and colors
def board_length : ℕ := 8
inductive Color
| red
| blue
| green

-- Define the main problem
theorem tiling_remainder_336 :
  let M := num_tilings_with_all_colors board_length in 
  M % 1000 = 336 :=
begin
  sorry
end

-- Define the function to count tilings (placeholder)
-- Note: Here we need a more complex definition that we'd potentially build from combinatorics and Inclusion-Exclusion Principle.
noncomputable def num_tilings_with_all_colors (n : ℕ) : ℕ := sorry

end tiling_remainder_336_l793_793652


namespace experiment_analysis_l793_793687

-- Define the total structure of the problem
structure TossData :=
  (cumulative_tosses : List ℕ)
  (caps_up : List ℕ)
  (frequency : List ℝ)

-- Define the given experimental data
def experiment_data : TossData :=
  { cumulative_tosses := [50, 100, 200, 300, 500, 1000, 2000, 3000, 5000],
    caps_up := [28, 54, 106, 158, 264, 527, 1056, 1587, 2650],
    frequency := [0.5600, 0.5400, 0.5300, 0.5267, 0.5280, 0.5270, 0.5280, 0.5290, 0.5300] }

-- Define the claims to be proven as booleans
def claim_1 (data : TossData) : Prop :=
  ∃ ε > 0.03, ∀ f ∈ data.frequency.drop 5, |f - 0.53| < ε

def claim_2 (data : TossData) : Prop :=
  False

def claim_3 (data : TossData) : Prop :=
  ∀ n, n ≥ 100 ∧ n ∈ data.cumulative_tosses → abs (data.frequency.nth_le (data.cumulative_tosses.index_of n) sorry  - 0.53) < 0.03

-- Define the theorem to validate the problem
theorem experiment_analysis : claim_1 experiment_data ∧ ¬claim_2 experiment_data ∧ claim_3 experiment_data := sorry

end experiment_analysis_l793_793687


namespace centroid_projections_sum_l793_793882

theorem centroid_projections_sum (A B C G P Q R: Type) 
    (h1 : dist A B = 5)
    (h2 : dist A C = 5)
    (h3 : dist B C = 6)
    (hG : G = centroid A B C)
    (hP : P = proj G B C)
    (hQ : Q = proj G A C)
    (hR : R = proj G A B) : 
    dist G P + dist G Q + dist G R = 68/15 :=
by
  sorry

end centroid_projections_sum_l793_793882


namespace repeating_decimal_sum_l793_793108

theorem repeating_decimal_sum (x : ℚ) (h : x = 34 / 999) : x.num + x.denom = 1033 := by 
  sorry

end repeating_decimal_sum_l793_793108


namespace find_line_equation_l793_793292

def ellipse (x y : ℝ) : Prop := (x ^ 2) / 6 + (y ^ 2) / 3 = 1

def meets_first_quadrant (l : Line) : Prop :=
  ∃ A B : Point, ellipse A.x A.y ∧ ellipse B.x B.y ∧ 
  A.x > 0 ∧ A.y > 0 ∧ B.x > 0 ∧ B.y > 0 ∧ l.contains A ∧ l.contains B

def intersects_axes (l : Line) : Prop :=
  ∃ M N : Point, M.y = 0 ∧ N.x = 0 ∧ l.contains M ∧ l.contains N
  
def equal_distances (M N A B : Point) : Prop :=
  dist M A = dist N B

def distance_MN (M N : Point) : Prop :=
  dist M N = 2 * Real.sqrt 3

theorem find_line_equation (l : Line) (A B M N : Point)
  (h1 : meets_first_quadrant l)
  (h2 : intersects_axes l)
  (h3 : equal_distances M N A B)
  (h4 : distance_MN M N) :
  l.equation = "x + sqrt(2) * y - 2 * sqrt(2) = 0" :=
sorry

end find_line_equation_l793_793292


namespace number_of_ordered_pairs_l793_793498

variables (a b : ℤ)
noncomputable def omega : ℂ := Complex.I

theorem number_of_ordered_pairs : 
  (omega^4 = 1 ∧ omega.im ≠ 0 ∧ abs (a * omega + b) = Real.sqrt 2) → 
  (∃ (a b : ℤ), abs (a * omega + b) = Real.sqrt 2) → 
  ∃ (n : ℕ), n = 4 :=
by {
  sorry
}

end number_of_ordered_pairs_l793_793498


namespace sin_alpha_of_terminal_side_l793_793834

theorem sin_alpha_of_terminal_side (α : ℝ) (P : ℝ × ℝ) 
  (hP : P = (5, 12)) :
  Real.sin α = 12 / 13 := sorry

end sin_alpha_of_terminal_side_l793_793834


namespace arrange_solids_by_surface_area_l793_793980

theorem arrange_solids_by_surface_area (V : ℝ) :
  (surface_area (sphere_of_volume V) <= surface_area (cylinder_of_volume_with_square_cross V)) ∧
  (surface_area (cylinder_of_volume_with_square_cross V) <= surface_area (octahedron_of_volume V)) ∧
  (surface_area (octahedron_of_volume V) <= surface_area (cube_of_volume V)) ∧
  (surface_area (cube_of_volume V) <= surface_area (cone_of_volume_with_equilateral_triangle_cross V)) ∧
  (surface_area (cone_of_volume_with_equilateral_triangle_cross V) <= surface_area (tetrahedron_of_volume V)) :=
sorry

def surface_area : Solid → ℝ := sorry

def sphere_of_volume (V : ℝ) : Solid := sorry
def cylinder_of_volume_with_square_cross (V : ℝ) : Solid := sorry
def octahedron_of_volume (V : ℝ) : Solid := sorry
def cube_of_volume (V : ℝ) : Solid := sorry
def cone_of_volume_with_equilateral_triangle_cross (V : ℝ) : Solid := sorry
def tetrahedron_of_volume (V : ℝ) : Solid := sorry

structure Solid :=
(volume : ℝ)
(surface_area : ℝ)

end arrange_solids_by_surface_area_l793_793980


namespace arctan_sum_l793_793208

theorem arctan_sum {a b : ℝ} (h3 : a = 3) (h7 : b = 7) :
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 :=
by
  sorry

end arctan_sum_l793_793208


namespace two_digit_numbers_count_l793_793341

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem two_digit_numbers_count :
  { n : ℕ | 10 ≤ n ∧ n ≤ 99 ∧ (tens_digit n < 3 ∨ units_digit n < 5) }.to_finset.card = 55 :=
by
  sorry

end two_digit_numbers_count_l793_793341


namespace speed_of_racer_X_l793_793440

-- Definitions
variable (speedX speedY speedZ : ℝ)

-- Conditions
def condition1 : Prop := speedY = 8
def condition2 : Prop := 200 / speedX = 175 / speedY
def condition3 : Prop := 200 / speedY = 185 / speedZ
def condition4 : Prop := 200 / speedX = 165 / speedZ

-- The main theorem
theorem speed_of_racer_X (h1 : condition1 speedX speedY speedZ)
                         (h2 : condition2 speedX speedY speedZ)
                         (h3 : condition3 speedX speedY speedZ)
                         (h4 : condition4 speedX speedY speedZ) :
                         speedX = 9.14 :=
sorry

end speed_of_racer_X_l793_793440


namespace smallest_b_base_l793_793624

theorem smallest_b_base :
  ∃ b : ℕ, b^2 ≤ 25 ∧ 25 < b^3 ∧ (∀ c : ℕ, c < b → ¬(c^2 ≤ 25 ∧ 25 < c^3)) :=
sorry

end smallest_b_base_l793_793624


namespace two_digit_numbers_less_than_35_l793_793352

theorem two_digit_numbers_less_than_35 : 
  let count : ℕ := (99 - 10 + 1) - (7 * 5)
  count = 55 :=
by
  -- definition of total number of two-digit numbers
  let total_two_digit_numbers : ℕ := 99 - 10 + 1
  -- definition of the number of unsuitable two-digit numbers
  let unsuitable_numbers : ℕ := 7 * 5
  -- definition of the count of suitable two-digit numbers
  let count : ℕ := total_two_digit_numbers - unsuitable_numbers
  -- verifying the final count
  exact rfl

end two_digit_numbers_less_than_35_l793_793352


namespace joshua_total_payment_is_correct_l793_793890

noncomputable def total_cost : ℝ := 
  let t_shirt_price := 8
  let sweater_price := 18
  let jacket_price := 80
  let jeans_price := 35
  let shoes_price := 60
  let jacket_discount := 0.10
  let shoes_discount := 0.15
  let clothing_tax_rate := 0.05
  let shoes_tax_rate := 0.08

  let t_shirt_count := 6
  let sweater_count := 4
  let jacket_count := 5
  let jeans_count := 3
  let shoes_count := 2

  let t_shirts_subtotal := t_shirt_price * t_shirt_count
  let sweaters_subtotal := sweater_price * sweater_count
  let jackets_subtotal := jacket_price * jacket_count
  let jeans_subtotal := jeans_price * jeans_count
  let shoes_subtotal := shoes_price * shoes_count

  let jackets_discounted := jackets_subtotal * (1 - jacket_discount)
  let shoes_discounted := shoes_subtotal * (1 - shoes_discount)

  let total_before_tax := t_shirts_subtotal + sweaters_subtotal + jackets_discounted + jeans_subtotal + shoes_discounted

  let t_shirts_tax := t_shirts_subtotal * clothing_tax_rate
  let sweaters_tax := sweaters_subtotal * clothing_tax_rate
  let jackets_tax := jackets_discounted * clothing_tax_rate
  let jeans_tax := jeans_subtotal * clothing_tax_rate
  let shoes_tax := shoes_discounted * shoes_tax_rate

  total_before_tax + t_shirts_tax + sweaters_tax + jackets_tax + jeans_tax + shoes_tax

theorem joshua_total_payment_is_correct : total_cost = 724.41 := by
  sorry

end joshua_total_payment_is_correct_l793_793890


namespace purely_periodic_fraction_period_length_divisible_l793_793501

noncomputable def purely_periodic_fraction (p q n : ℕ) : Prop :=
  ∃ (r : ℕ), 10 ^ n - 1 = r * q ∧ (∃ (k : ℕ), q * (10 ^ (n * k)) ∣ p)

theorem purely_periodic_fraction_period_length_divisible
  (p q n : ℕ) (hq : ¬ (2 ∣ q) ∧ ¬ (5 ∣ q)) (hpq : p < q) (hn : 10 ^ n - 1 ∣ q) :
  purely_periodic_fraction p q n :=
by
  sorry

end purely_periodic_fraction_period_length_divisible_l793_793501


namespace circles_externally_tangent_implies_m_l793_793987

theorem circles_externally_tangent_implies_m (m : ℝ) :
  ∃ m : ℝ, (m = 2 ∨ m = -5) ∧
  ∃ c1 c2 r1 r2 : ℝ,
    c1 = (m, -2) ∧
    c2 = (-1, m) ∧
    r1 = 3 ∧
    r2 = 2 ∧
    (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2 :=
by
  sorry

end circles_externally_tangent_implies_m_l793_793987


namespace business_revenue_l793_793085

noncomputable theory

open Real

def canoe_rental_cost : ℝ := 12
def kayak_rental_cost : ℝ := 18
def canoes_to_kayaks_ratio : ℝ := 3 / 2
def extra_canoes : ℝ := 7

theorem business_revenue :
  ∃ (C K : ℝ), 
    C = canoes_to_kayaks_ratio * K ∧ 
    C = K + extra_canoes ∧
    revenue = (C * canoe_rental_cost) + (K * kayak_rental_cost) :=
begin
  let K := 14,
  let C := canoes_to_kayaks_ratio * K,
  have h1 : C = K + extra_canoes, sorry,
  have h2 : revenue = (C * canoe_rental_cost) + (K * kayak_rental_cost) := 504,
  exact ⟨C, K, h1, h2⟩,
end

end business_revenue_l793_793085


namespace find_third_angle_of_triangle_l793_793864

theorem find_third_angle_of_triangle (a b c : ℝ) (h₁ : a = 40) (h₂ : b = 3 * c) (h₃ : a + b + c = 180) : c = 35 := 
by sorry

end find_third_angle_of_triangle_l793_793864


namespace happy_snakes_not_purple_l793_793581

variable {Snake : Type}
variable [Fintype Snake]

-- Number of snakes
constant num_snakes : ℕ
constant purple_snakes : Finset Snake
constant happy_snakes : Finset Snake

-- Initial conditions
axiom h1 : num_snakes = 13
axiom h2 : purple_snakes.card = 4
axiom h3 : happy_snakes.card = 5

-- Predicate definitions
def CanAdd (s : Snake) : Prop := sorry
def CanSubtract (s : Snake) : Prop := sorry
def IsHappy (s : Snake) : Prop := s ∈ happy_snakes
def IsPurple (s : Snake) : Prop := s ∈ purple_snakes

-- Logical implications
axiom h4 : ∀ s, IsHappy s → CanAdd s
axiom h5 : ∀ s, IsPurple s → ¬ CanSubtract s
axiom h6 : ∀ s, ¬ CanSubtract s → ¬ CanAdd s

-- To prove: Happy snakes are not purple
theorem happy_snakes_not_purple : ∀ s, IsHappy s → ¬ IsPurple s :=
by sorry

end happy_snakes_not_purple_l793_793581


namespace book_arrangement_l793_793388

theorem book_arrangement : 
  let total_books := 7 in
  let identical_math_books := 3 in
  let identical_physics_books := 2 in
  fact total_books / (fact identical_math_books * fact identical_physics_books) = 420 :=
by {
  -- sorry for the proof section
  sorry
}

end book_arrangement_l793_793388


namespace enclosed_area_l793_793452

theorem enclosed_area {x : ℝ} (h0 : 0 < x) : 
  let y1 := 2 * x,
      y2 := (1 / 2) * x,
      y3 := 1 / x,
      a := ∫ (t : ℝ) in 0 .. (Real.sqrt 2 / 2), 2 * t,
      b := ∫ (t : ℝ) in (Real.sqrt 2 / 2) .. Real.sqrt 2, 1 / t,
      c := ∫ (t : ℝ) in 0 .. Real.sqrt 2, (1 / 2) * t
  in a + b - c = Real.log 2 := 
by
  sorry

end enclosed_area_l793_793452


namespace two_digit_numbers_less_than_35_l793_793364

theorem two_digit_numbers_less_than_35 : 
  ∃ n, n ∈ finset.range 90 ∧ (10 ≤ n + 10 ∧ n + 10 ≤ 99) ∧ 
  (let tens_digit := (n + 10) / 10 in 
   let units_digit := (n + 10) % 10 in 
   tens_digit < 3 ∨ units_digit < 5) → 
  nat.card {n | n ∈ finset.range 90 ∧ (10 ≤ n + 10 ∧ n + 10 ≤ 99) ∧ 
                    (let tens_digit := (n + 10) / 10 in 
                     let units_digit := (n + 10) % 10 in 
                     tens_digit < 3 ∨ units_digit < 5)} = 55
:= sorry

end two_digit_numbers_less_than_35_l793_793364


namespace find_line_equation_l793_793296

theorem find_line_equation (x y : ℝ) : 
  (∃ A B, (A.x^2 / 6 + A.y^2 / 3 = 1) ∧ (B.x^2 / 6 + B.y^2 / 3 = 1) ∧
  (A.x > 0 ∧ A.y > 0) ∧ (B.x > 0 ∧ B.y > 0) ∧
  let M := (-B.y, 0) in
  let N := (0, B.y) in 
  (abs (M.x - A.x) = abs (N.y - B.y)) ∧ 
  (abs (M.x - N.x + M.y - N.y) = 2 * sqrt 3)) →
  x + sqrt 2 * y - 2 * sqrt 2 = 0 := 
sorry

end find_line_equation_l793_793296


namespace magnitude_b_l793_793838

noncomputable theory

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Conditions
def mag_a : real := ‖a‖ = 3
def mag_diff_ab : real := ‖a - b‖ = 5
def dot_ab : real := inner a b = 1

-- Goal
theorem magnitude_b (h1 : ‖a‖ = 3) (h2 : ‖a - b‖ = 5) (h3 : inner a b = 1) : ‖b‖ = 3 * real.sqrt 2 :=
by 
  sorry

end magnitude_b_l793_793838


namespace two_digit_numbers_less_than_35_l793_793365

theorem two_digit_numbers_less_than_35 : 
  ∃ n, n ∈ finset.range 90 ∧ (10 ≤ n + 10 ∧ n + 10 ≤ 99) ∧ 
  (let tens_digit := (n + 10) / 10 in 
   let units_digit := (n + 10) % 10 in 
   tens_digit < 3 ∨ units_digit < 5) → 
  nat.card {n | n ∈ finset.range 90 ∧ (10 ≤ n + 10 ∧ n + 10 ≤ 99) ∧ 
                    (let tens_digit := (n + 10) / 10 in 
                     let units_digit := (n + 10) % 10 in 
                     tens_digit < 3 ∨ units_digit < 5)} = 55
:= sorry

end two_digit_numbers_less_than_35_l793_793365


namespace correct_statements_count_l793_793699

theorem correct_statements_count : 
  (if ($\frac{7}{2} \in \mathbb{R}$ then 1 else 0) 
  + if ($\pi \in \mathbb{Q}$ then 1 else 0) 
  + if ($\vert -3 \vert \notin \mathbb{N}$ then 1 else 0) 
  + if ($-\sqrt{4} \in \mathbb{Z}$ then 1 else 0)) = 2 :=
by
  -- Definitions of the conditions
  have h1 : $\frac{7}{2} \in \mathbb{R}$ := sorry,
  have h2 : $\pi \notin \mathbb{Q}$ := sorry,
  have h3 : $\vert -3 \vert = 3 \in \mathbb{N}$ := sorry,
  have h4 : $-\sqrt{4} = -2 \in \mathbb{Z}$ := sorry,

  -- Since evaluating the above conditions will result in two truths,
  -- thus making the sum of true conditions 2.
  sorry

end correct_statements_count_l793_793699


namespace boy_girl_team_combinations_l793_793942

theorem boy_girl_team_combinations (b g teamSize : ℕ) (H_b : b = 5) (H_g : g = 5) (H_t : teamSize = 3) :
  (nat.choose (b + g) teamSize - nat.choose b teamSize = 110) :=
by
  rw [H_b, H_g, H_t]
  sorry

end boy_girl_team_combinations_l793_793942


namespace milk_drinks_on_weekdays_l793_793916

-- Defining the number of boxes Lolita drinks on a weekday as a variable W
variable (W : ℕ)

-- Condition: Lolita drinks 30 boxes of milk per week.
axiom total_milk_per_week : 5 * W + 2 * W + 3 * W = 30

-- Proof (Statement) that Lolita drinks 15 boxes of milk on weekdays.
theorem milk_drinks_on_weekdays : 5 * W = 15 :=
by {
  -- Use the given axiom to derive the solution
  sorry
}

end milk_drinks_on_weekdays_l793_793916


namespace solution_inequality_l793_793950

noncomputable def problem (x : ℝ) : Prop :=
  x ∉ {1, 2, 3, 4} ∧
  (1 / (x - 1) - 3 / (x - 2) + 5 / (x - 3) - 2 / (x - 4) < 1 / 40)

theorem solution_inequality (x : ℝ) : problem x ↔
  x ∈ Set.interval (-∞ : ℝ) (-4) ∪ Set.Ioo (-1) 1 ∪ Set.Ioo 2 3 ∪ Set.Ioo 4 5 ∪ Set.Ioo 8 (∞ : ℝ) :=
by 
  sorry

end solution_inequality_l793_793950


namespace solve_system_of_inequalities_l793_793545

theorem solve_system_of_inequalities {x : ℝ} :
  (x + 3 ≥ 2) ∧ (2 * (x + 4) > 4 * x + 2) ↔ (-1 ≤ x ∧ x < 3) :=
by
  sorry

end solve_system_of_inequalities_l793_793545


namespace days_spent_on_Orbius5_l793_793527

-- Define the conditions
def days_per_year : Nat := 250
def seasons_per_year : Nat := 5
def length_of_season : Nat := days_per_year / seasons_per_year
def seasons_stayed : Nat := 3

-- Theorem statement
theorem days_spent_on_Orbius5 : (length_of_season * seasons_stayed = 150) :=
by 
  -- Proof is skipped
  sorry

end days_spent_on_Orbius5_l793_793527


namespace decimal_38_to_binary_l793_793725

noncomputable def decimal_to_binary (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  let quotient := n / 2 in
  let remainder := n % 2 in
  10 * decimal_to_binary quotient + remainder

theorem decimal_38_to_binary :
  decimal_to_binary 38 = 100110 := 
sorry

end decimal_38_to_binary_l793_793725


namespace problem1_problem2_l793_793773

noncomputable theory

open Set Real

-- Problem 1: Proving the union of sets A and B.
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def UnionAB : Set ℝ := {x | -2 ≤ x ∧ x < 6}

theorem problem1 : A ∪ B = UnionAB :=
by {
  sorry
}

-- Problem 2: Proving all subsets of set C.
def IntersectAB : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def C : Set ℝ := {x | x ∈ IntersectAB ∧ x ∈ Set.univ ∧ x ∈ {n : ℝ | IsInt n}} -- IsInt checks for integers
def subsetsC : Set (Set ℝ) := {{}, {2}, {3}, {2, 3}}

theorem problem2 : (∃ x, x = C) ∧ subsets C = subsetsC :=
by {
  sorry
}

end problem1_problem2_l793_793773


namespace sum_of_numerator_and_denominator_l793_793114

def repeating_decimal_represents_fraction : Prop :=
  let x := 34 / 99
  0.34̅ = x

/-- The sum of the numerator and denominator of 0.34̅ in its lowest terms is 133. -/
theorem sum_of_numerator_and_denominator
  (x : ℚ)
  (hx : repeating_decimal_represents_fraction) :
  (34 + 99 = 133) :=
begin
  sorry
end

end sum_of_numerator_and_denominator_l793_793114


namespace game_strategy_XiaoLiang_wins_l793_793927

/-- On a small table, there are 22 cards. The rules of the game are: 
- You can take at least 1 card and at most 2 cards at a time.
- The person who takes the last card loses.
- Xiao Liang and Xiao Xuan take turns to pick up cards, with Xiao Liang going first.
--/
theorem game_strategy_XiaoLiang_wins :
  let n := 22 in
  let can_take := {1, 2} in
  (∀ turns, turns % 3 ≠ 2 → 
    ∀ (XiaoLiang : ℕ) (XiaoXuan : ℕ),
    (XiaoLiang ∈ can_take) ∧ (XiaoXuan ∈ can_take) → 
    (XiaoLiang + turns ≠ (n - 1)))  :=
by
  sorry

end game_strategy_XiaoLiang_wins_l793_793927


namespace count_four_digit_even_integers_l793_793816

-- Defining the set of even digits
def even_digits := {0, 2, 4, 6, 8}

-- The problem statement in Lean 4
theorem count_four_digit_even_integers :
  (count(p : ℕ // 1000 ≤ p ∧ p < 10000 ∧ ∀ d ∈ to_digits 10 (p : ℕ), d ∈ even_digits)) = 500 :=
sorry

end count_four_digit_even_integers_l793_793816


namespace mary_money_left_l793_793513

theorem mary_money_left (p : ℝ) : 50 - (4 * p + 2 * p + 4 * p) = 50 - 10 * p := 
by 
  sorry

end mary_money_left_l793_793513


namespace seq_a1_seq_a2_seq_a3_seq_general_term_sum_Tn_l793_793307

def S (n : ℕ) : ℕ := 3^n - 1

def a (n : ℕ) : ℕ := if n = 1 then 2 else 2 * 3^(n-1)

def T (n : ℕ) : ℕ := (1 / 2).toNat + ((2 * n - 1) / 2).toNat * 3^n

theorem seq_a1 : a 1 = 2 :=
by simp [a]

theorem seq_a2 : a 2 = 6 :=
by simp [a]

theorem seq_a3 : a 3 = 18 :=
by simp [a]

theorem seq_general_term (n : ℕ) : a n = 2 * 3^(n-1) :=
by {
  cases n,
  { simp [a] },
  { simp [a, nat.add_sub_assoc, nat.succ_eq_one_add, nat.one_add] }
}

theorem sum_Tn (n : ℕ) : T n = (1 / 2).toNat + ((2 * n - 1) / 2).toNat * 3^n :=
by simp [T]

end seq_a1_seq_a2_seq_a3_seq_general_term_sum_Tn_l793_793307


namespace find_delta_l793_793776

theorem find_delta (p q Δ : ℕ) (h₁ : Δ + q = 73) (h₂ : 2 * (Δ + q) + p = 172) (h₃ : p = 26) : Δ = 12 :=
by
  sorry

end find_delta_l793_793776


namespace probability_of_perfect_square_sum_l793_793614

def is_perfect_square (n : ℕ) : Prop :=
  n = 1*1 ∨ n = 2*2 ∨ n = 3*3 ∨ n = 4*4

theorem probability_of_perfect_square_sum :
  let total_outcomes := 64
  let perfect_square_sums := 12
  (perfect_square_sums / total_outcomes : ℚ) = 3 / 16 :=
by
  sorry

end probability_of_perfect_square_sum_l793_793614


namespace f_at_5_l793_793391

noncomputable def f : ℝ → ℝ := sorry

theorem f_at_5 (h : ∀ x : ℝ, f (10^x) = x) : f 5 = log 10 5 :=
sorry

end f_at_5_l793_793391


namespace arctan_sum_l793_793210

theorem arctan_sum {a b : ℝ} (h3 : a = 3) (h7 : b = 7) :
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 :=
by
  sorry

end arctan_sum_l793_793210


namespace complex_division_l793_793787

theorem complex_division : (1 - 2 * complex.i) / (1 + complex.i) = (-1 - 3 * complex.i) / 2 := by
  sorry

end complex_division_l793_793787


namespace regular_dinosaurs_count_l793_793711

/-
Define the conditions:
1. w_regular := 800 -- weight of each regular dinosaur
2. w_barney := 800 * n + 1500 -- weight of Barney
3. total_weight := 9500 -- combined weight of Barney and the regular dinosaurs
-/

theorem regular_dinosaurs_count:
  ∃ (n : ℕ), 
  let w_regular := 800 in
  let w_barney := 800 * n + 1500 in
  let total_weight := 9500 in
  w_regular * n + w_barney = total_weight ∧ n = 5 :=
by {
  sorry
}

end regular_dinosaurs_count_l793_793711


namespace zarnin_staffing_ways_l793_793188

theorem zarnin_staffing_ways :
  let num_candidates := 15
  let num_positions := 5
  (finset.range num_candidates).card = num_candidates →
  ∏ i in finset.range num_positions, num_candidates - i = 360360 :=
by
  intro num_candidates num_positions h_card
  sorry

end zarnin_staffing_ways_l793_793188


namespace sum_of_valid_ns_l793_793751

theorem sum_of_valid_ns :
  ∃ n_list : List ℕ,
  (∀ n ∈ n_list, ∃ b : ℤ, |b| ≠ 4 ∧ (base_of (-4) n = base_of b n)) ∧ 
  n_list.sum = 1026 := 
sorry

end sum_of_valid_ns_l793_793751


namespace exist_monochromatic_equilateral_triangle_l793_793084

theorem exist_monochromatic_equilateral_triangle 
  (color : ℝ × ℝ → ℕ) 
  (h_color : ∀ p : ℝ × ℝ, color p = 0 ∨ color p = 1) : 
  ∃ (A B C : ℝ × ℝ), (dist A B = dist B C) ∧ (dist B C = dist C A) ∧ (color A = color B ∧ color B = color C) :=
sorry

end exist_monochromatic_equilateral_triangle_l793_793084


namespace min_value_eq_216_l793_793909

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (a^2 + 4*a + 1) * (b^2 + 4*b + 1) * (c^2 + 4*c + 1) / (a * b * c)

theorem min_value_eq_216 {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  min_value a b c = 216 :=
sorry

end min_value_eq_216_l793_793909


namespace triangle_side_length_l793_793884

noncomputable def sine (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180) -- Define sine function explicitly (degrees to radians)

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ)
  (hA : A = 30) (hC : C = 45) (ha : a = 4) :
  c = 4 * Real.sqrt 2 := by
  sorry

end triangle_side_length_l793_793884


namespace vasya_numbers_l793_793518

-- Define the problem in Lean
theorem vasya_numbers
  (n : ℕ)
  (a : Fin n → ℝ)
  (ha_pos : ∀ i, 0 < a i)  -- n is positive and each a_i is positive.
  (ha1_lt_2 : ∀ i, a i ≥ 1 ∧ a i < 2):  -- Assumption 1 ≤ a₁ ≤ a₂ ≤ ... ≤ aₙ < 2
  ∃ (b : Fin n → ℝ), 
  (∀ i, b i ≥ a i) ∧  -- Condition 1: b_i ≥ a_i for all i ≤ n
  (∀ i j, ∃ k : ℕ, b i = 2^k * b j) ∧  -- Condition 2: The ratio of b_is is a power of 2
  (∏ i, b i ≤ 2^((n-1)/2) * ∏ i, a i) -- Prove: b_1 * b_2 * ... * b_n ≤ 2^{(n-1)/2} * a_1 * a_2 * ... * a_n
sorry

end vasya_numbers_l793_793518


namespace remainder_degrees_l793_793627

theorem remainder_degrees (f : Polynomial ℝ) :
  let d := 7 in ∀ (r : Polynomial ℝ), degree r < d → degree r ∈ {0, 1, 2, 3, 4, 5, 6} :=
by
  intros d r hd
  sorry

end remainder_degrees_l793_793627


namespace cows_on_farm_l793_793521

theorem cows_on_farm (weekly_production_per_6_cows : ℕ) 
                     (production_over_5_weeks : ℕ) 
                     (number_of_weeks : ℕ) 
                     (cows : ℕ) :
  weekly_production_per_6_cows = 108 →
  production_over_5_weeks = 2160 →
  number_of_weeks = 5 →
  (cows * (weekly_production_per_6_cows / 6) * number_of_weeks = production_over_5_weeks) →
  cows = 24 :=
by
  intros h1 h2 h3 h4
  sorry

end cows_on_farm_l793_793521


namespace curve_equation_AB_ST_range_l793_793678

def C (x y : ℝ) : Prop := x^2 + y^2 = 4

def M_on_C (M : ℝ × ℝ) : Prop := C M.1 M.2

def perpendicular_x_axis (M : ℝ × ℝ) : ℝ × ℝ := (M.1, 0)

def P_condition (N M P : ℝ × ℝ) : Prop := 
  P.1 = N.1 + (M.1 - N.1) * (sqrt(3) / 2) ∧
  P.2 = N.2 + (M.2 - N.2) * (sqrt(3) / 2)

def Q : ℝ × ℝ := (0, 1)

def l_exists (Q l: ℝ) (A B : ℝ × ℝ) (k : ℝ) : Prop :=
  (A.2 = k * A.1 + 1) ∧ (B.2 = k * B.1 + 1) ∧ 
  (4 * k^2 + 3) * A.1^2 + 8 * k * A.1 - 8 = 0 ∧ 
  (4 * k^2 + 3) * B.1^2 + 8 * k * B.1 - 8 = 0

theorem curve_equation (M P : ℝ × ℝ) :
  M_on_C M →
  P_condition (perpendicular_x_axis M) M P → 
  (P.1^2 / 4 + P.2^2 / 3 = 1) :=
by sorry

theorem AB_ST_range (M S T A B : ℝ × ℝ) (k : ℝ) :
  M_on_C M →
  P_condition (perpendicular_x_axis M) M A →
  l_exists Q k A B →
  let d := 1 / sqrt(k^2 + 1) in
  let ST := 2 * sqrt(4 - d^2) in
  |A.1 - B.1| * |S.1 - T.1| = 8 * sqrt(3) →
  8 * sqrt(2) ≤ |A.1 - B.1| * |S.1 - T.1| ∧ |A.1 - B.1| * |S.1 - T.1| < 8 * sqrt(3) :=
by sorry

end curve_equation_AB_ST_range_l793_793678


namespace two_digit_numbers_count_l793_793340

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem two_digit_numbers_count :
  { n : ℕ | 10 ≤ n ∧ n ≤ 99 ∧ (tens_digit n < 3 ∨ units_digit n < 5) }.to_finset.card = 55 :=
by
  sorry

end two_digit_numbers_count_l793_793340


namespace arctan_sum_l793_793195

theorem arctan_sum (a b : ℝ) (h1 : a = 3) (h2 : b = 7) : 
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 := 
by 
  rw [h1, h2]
  sorry

end arctan_sum_l793_793195


namespace probability_of_perfect_square_sum_l793_793595

-- Define the conditions
def dice_faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def perfect_squares_in_range : Finset ℕ := {4, 9, 16}
def total_outcomes : ℕ := 8 * 8  -- total number of outcomes when rolling two dice

-- Calculate the number of favorable outcomes
def favorable_outcomes : ℕ :=
  (dice_faces.product dice_faces).filter (λ pair => perfect_squares_in_range.member (pair.1 + pair.2)).card

-- Calculate the probability of getting a perfect square sum
noncomputable def probability_perfect_square_sum : ℚ :=
  favorable_outcomes / total_outcomes

-- The statement to prove that the probability is 3/16
theorem probability_of_perfect_square_sum :
  probability_perfect_square_sum = 3 / 16 :=
begin
  sorry
end

end probability_of_perfect_square_sum_l793_793595


namespace radius_of_second_cylinder_l793_793662

theorem radius_of_second_cylinder :
  let r1 := 6
  let h1 := 4
  let h2 := 4
  let V1 := π * r1^2 * h1
  let V2 := 3 * V1
  let r2^2 := V2 / (π * h2)
  r2 = 6 * Real.sqrt 3 :=
by
  -- proof steps should go here
  sorry

end radius_of_second_cylinder_l793_793662


namespace find_y_value_l793_793418

-- Define the given conditions and the final question in Lean
theorem find_y_value (k : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : y = k * x ^ (1/3)) 
  (h2 : y = 4 * real.sqrt 3)
  (x1 : x = 64) 
  : ∃ k, y = 2 * real.sqrt 3 :=
sorry

end find_y_value_l793_793418


namespace complete_square_solution_l793_793630

theorem complete_square_solution (x: ℝ) : (x^2 + 8 * x - 3 = 0) -> ((x + 4)^2 = 19) := 
by
  sorry

end complete_square_solution_l793_793630


namespace measure_of_angle_C_l793_793704

theorem measure_of_angle_C (A B : ℝ) (h1 : A + B = 180) (h2 : A = 5 * B) : A = 150 := by
  sorry

end measure_of_angle_C_l793_793704


namespace polynomial_equation_solution_l793_793742

open Polynomial

noncomputable def exists_polynomial_form (P : Polynomial ℝ) : Prop :=
∀ a b c : ℝ, ab + bc + ca = 0 → P.eval (a - b) + P.eval (b - c) + P.eval (c - a) = 2 * P.eval (a + b + c)

noncomputable def polynomial_solution_form : Prop :=
∀ P : Polynomial ℝ, exists_polynomial_form P →
∃ (u v : ℝ), P = C u * X^4 + C v * X^2

theorem polynomial_equation_solution : polynomial_solution_form :=
sorry

end polynomial_equation_solution_l793_793742


namespace prop_equiv_l793_793058

theorem prop_equiv (a : ℝ) : (∀ x ∈ set.Icc (1 : ℝ) 3, x^2 - a ≤ 0) ↔ a ≥ 10 := by
  sorry

end prop_equiv_l793_793058


namespace non_intersecting_polygon_l793_793445

-- Formal statement in Lean 4
theorem non_intersecting_polygon (n : ℕ) (points : Fin n → Point) :
  (∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬ collinear (points i) (points j) (points k)) →
  ∃ perm : Fin n → Fin n, ¬ exists_intersections (perm_points points perm) :=
by
  -- proof steps would go here
  sorry

end non_intersecting_polygon_l793_793445


namespace probability_two_8sided_dice_sum_perfect_square_l793_793609

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def probabilityPerfectSquareSum (dice_sides : ℕ) (perfect_squares : List ℕ) : ℚ :=
  let outcomes := (dice_sides * dice_sides)
  let favorable_outcomes := perfect_squares.sum (λ ps, 
    (List.filter (λ (pair : ℕ × ℕ), pair.fst + pair.snd = ps) ((List.range dice_sides).product (List.range dice_sides))).length)
  favorable_outcomes /. outcomes

theorem probability_two_8sided_dice_sum_perfect_square :
  probabilityPerfectSquareSum 8 [4, 9, 16] = 3 / 16 := sorry

end probability_two_8sided_dice_sum_perfect_square_l793_793609


namespace triangle_area_l793_793560

-- Define the lines and the corresponding triangle.
def line1 (x : ℝ) := (4 : ℝ)
def line2 (x : ℝ) := (3 * x : ℝ)
def origin := (0, 0)
def pointP := (4, 0)
def pointQ := (4, 12)

-- Calculate the lengths of the segments OP and PQ.
def OP_length : ℝ := 4
def PQ_length : ℝ := 12

-- Define the area of the triangle using the lengths calculated.
def area_triangle : ℝ := 1/2 * OP_length * PQ_length

-- Prove that the area is 24
theorem triangle_area : area_triangle = 24 := by
  sorry

end triangle_area_l793_793560


namespace radius_of_scrap_cookie_l793_793924

theorem radius_of_scrap_cookie
  (r_cookies : ℝ) (n_cookies : ℕ) (radius_layout : Prop)
  (circle_diameter_twice_width : Prop) :
  (r_cookies = 0.5 ∧ n_cookies = 9 ∧ radius_layout ∧ circle_diameter_twice_width)
  →
  (∃ r_scrap : ℝ, r_scrap = Real.sqrt 6.75) :=
by
  sorry

end radius_of_scrap_cookie_l793_793924


namespace g_inv_undefined_at_one_l793_793394

-- Define the function g
def g (x : ℝ) : ℝ := (x - 5) / (x - 7)

-- Define the inverse function g_inv, specifying the domain where it is defined
noncomputable def g_inv (x : ℝ) : ℝ := (5 - 7*x) / (1 - x)

-- Theorem stating that the inverse function g_inv is undefined at x = 1
theorem g_inv_undefined_at_one : ¬∃ y : ℝ, g_inv 1 = y :=
by
  -- The proof calculations and logic would be added here to show the inverse is undefined at x = 1
  sorry

end g_inv_undefined_at_one_l793_793394


namespace probability_of_perfect_square_sum_on_two_8_sided_dice_l793_793594

-- Define the dice and possible sums
def is_sum_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def pairs_rolled : finset (ℕ × ℕ) :=
  finset.univ.filter (λ p, p.1 + p.2 ∈ {4, 9, 16})

theorem probability_of_perfect_square_sum_on_two_8_sided_dice :
  (pairs_rolled.card : ℚ) / 64 = 3 / 16 :=
by
  sorry

end probability_of_perfect_square_sum_on_two_8_sided_dice_l793_793594


namespace intersection_of_lines_l793_793223

theorem intersection_of_lines : 
  ∃ (x y : ℚ), 3 * y = -2 * x + 6 ∧ 4 * y = 3 * x - 4 ∧ x = 36 / 17 ∧ y = 10 / 17 :=
by {
  use (36 / 17, 10 / 17),
  sorry
}

end intersection_of_lines_l793_793223


namespace problem1_simplified_expression_l793_793027

theorem problem1_simplified_expression (x : ℝ) (k : ℤ) (h : x ≠ k * (π / 2)) :
  (1 + (Real.tan x - Real.cot x) * Real.sin (2 * x)) * Real.cos x + Real.cos (3 * x) = 0 :=
sorry

end problem1_simplified_expression_l793_793027


namespace cyclist_speed_l793_793589

theorem cyclist_speed (v : ℝ) (h : 0.7142857142857143 * (30 + v) = 50) : v = 40 :=
by
  sorry

end cyclist_speed_l793_793589


namespace range_of_a_l793_793784

/-- Given that the point (1, 1) is located inside the circle (x - a)^2 + (y + a)^2 = 4, 
    proving that the range of values for a is -1 < a < 1. -/
theorem range_of_a (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) → 
  (-1 < a ∧ a < 1) :=
by
  intro h
  sorry

end range_of_a_l793_793784


namespace correct_statements_l793_793133

-- Conditions extracted from the problem
def statement_A (x y : ℝ → ℝ) : Prop :=
  ∀ x, y x = 1.5 * x - 2 → ¬ (correlation x y < 0)

def statement_B : Prop :=
  ∀ residuals : ℝ → ℝ, uniformly_distributed residuals →
  (∀ x, abs (residuals x) < ε) → better_regression_model residuals

def statement_C (r : ℝ) : Prop :=
  abs r ≤ 1 ∧ abs r > 0 → strong_linear_correlation r

def statement_D (x y : list ℝ) (b a : ℝ) : Prop :=
  ∀ x y, regression_line b a x y → passes_through_center x y

-- Proof statement to be provided
theorem correct_statements (residuals : ℝ → ℝ) (r : ℝ) (x y : list ℝ) (b a : ℝ)
  (A : statement_A x y)
  (B : statement_B residuals)
  (C : statement_C r)
  (D : statement_D x y b a)
: B ∧ C :=
by
sorry

end correct_statements_l793_793133


namespace value_of_b_minus_a_l793_793285

-- Definition of the function and its domain and range
def domain (a b : ℝ) := a ≤ b
def range (f : ℝ → ℝ) := ∀ x ∈ set.Icc a b, f x ∈ set.Icc (-1/2) 1

-- The main statement we want to prove
theorem value_of_b_minus_a (a b : ℝ) 
  (h_dom : domain a b) 
  (h_rng : range (cos)) : 
  b - a ≠ π / 3 :=
by
  sorry

end value_of_b_minus_a_l793_793285


namespace empty_set_a_gt_nine_over_eight_singleton_set_a_values_at_most_one_element_set_a_range_l793_793329

noncomputable def A (a : ℝ) : Set ℝ := { x | a*x^2 - 3*x + 2 = 0 }

theorem empty_set_a_gt_nine_over_eight (a : ℝ) : A a = ∅ ↔ a > 9 / 8 :=
by
  sorry

theorem singleton_set_a_values (a : ℝ) : (∃ x, A a = {x}) ↔ (a = 0 ∨ a = 9 / 8) :=
by
  sorry

theorem at_most_one_element_set_a_range (a : ℝ) : (∀ x y, x ∈ A a → y ∈ A a → x = y) →
  (A a = ∅ ∨ ∃ x, A a = {x}) ↔ (a = 0 ∨ a ≥ 9 / 8) :=
by
  sorry

end empty_set_a_gt_nine_over_eight_singleton_set_a_values_at_most_one_element_set_a_range_l793_793329


namespace sum_numerator_denominator_l793_793095

-- Given the repeating decimal 0.343434...
def repeating_decimal : ℚ := 34 / 99

-- Prove that the decimal's fraction form in lowest terms adds up to 133.
theorem sum_numerator_denominator : 
  let num := repeating_decimal.num in 
  let denom := repeating_decimal.denom in 
  num + denom = 133 :=
by
  have x : repeating_decimal = 34 / 99 := by sorry
  sorry   -- Placeholder for the proof steps to demonstrate num and denom values lead to 133.

end sum_numerator_denominator_l793_793095


namespace probability_calculation_l793_793539

def balls := {1, 2, 3, 4, 5, 6, 7}

def is_odd (n : ℕ) : Prop := n % 2 = 1
def greater_than_nine (n : ℕ) : Prop := n > 9

def possible_sums := {n | ∃ (a b : ℕ), a ∈ balls ∧ b ∈ balls ∧ n = a + b ∧ is_odd n ∧ greater_than_nine n }

noncomputable def probability_odd_sum_and_greater_than_nine : ℚ :=
  possible_sums.to_finset.card / (balls.to_finset.card) ^ 2

theorem probability_calculation : probability_odd_sum_and_greater_than_nine = 6 / 49 :=
by sorry

end probability_calculation_l793_793539


namespace find_DE_length_l793_793469

noncomputable def triangle_DE_length (A B C D E : Type) (BC CD DE : ℝ) (C_angle : ℝ) : Prop :=
  BC = 40 ∧ C_angle = 45 ∧ CD = 20 ∧ DE = 20

theorem find_DE_length {A B C D E : Type} (BC CD DE : ℝ) (C_angle : ℝ) 
  (hBC : BC = 40) (hC_angle : C_angle = 45) (hCD : CD = 20) : DE = 20 :=
by {
  have hDE : DE = 20, sorry,
  exact hDE
}

end find_DE_length_l793_793469


namespace arctan_sum_l793_793207

theorem arctan_sum {a b : ℝ} (h3 : a = 3) (h7 : b = 7) :
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 :=
by
  sorry

end arctan_sum_l793_793207


namespace unique_solution_abc_l793_793251

theorem unique_solution_abc (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) 
(h1 : b ∣ 2^a - 1) 
(h2 : c ∣ 2^b - 1) 
(h3 : a ∣ 2^c - 1) : 
  a = 1 ∧ b = 1 ∧ c = 1 :=
sorry

end unique_solution_abc_l793_793251


namespace question_one_question_two_i_question_two_ii_l793_793315

section
variable (f : ℝ → ℝ) (a x : ℝ)

-- Conditions for all problems
def f_def : f x = a^x - exp(1) * x^2 := sorry
def a_pos : a > 0 := sorry
def a_ne_one : a ≠ 1 := sorry

-- Condition for Question (1)
def a_eq_e : a = exp(1) := sorry
def tangent_line_at_x1 : f'(1) * (x - 1) + f 1 = 0 := sorry

-- Proof problem for Question (1)
theorem question_one : (f'(1) = -exp(1) ∧ f 1 = 0) → 
  (exp(1)*x - exp(1) + y = 0) := sorry

-- Conditions for Question (2)
def a_gt_one : a > 1 := sorry

-- Condition for Question (2i)
def has_three_zeros (x1 x2 x3 : ℝ) : f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 := sorry

-- Proof problem for Question (2i)
theorem question_two_i : has_three_zeros x1 x2 x3 → 
  1 < a ∧ a < exp(2 / sqrt(exp(1))) := sorry

-- Conditions for Question (2ii)
def zero_order (x1 x2 x3 : ℝ) : x1 < x2 ∧ x2 < x3 := sorry

-- Proof problem for Question (2ii)
theorem question_two_ii : has_three_zeros x1 x2 x3 ∧ zero_order x1 x2 x3 →
  (x1 + 3 * x2 + x3 > (2 * exp(1) + 1) / sqrt(exp(1))) := sorry

end

end question_one_question_two_i_question_two_ii_l793_793315


namespace num_suitable_two_digit_numbers_l793_793377

/-- 
How many two-digit numbers have at least one digit that is smaller than the corresponding digit in the number 35?
-/
theorem num_suitable_two_digit_numbers : 
  let two_digit_numbers := { n : ℕ | 10 ≤ n ∧ n ≤ 99 },
      suitable_numbers := { n ∈ two_digit_numbers | (n / 10 < 3) ∨ (n % 10 < 5) } in
  suitable_numbers.card = 55 :=
by
  sorry

end num_suitable_two_digit_numbers_l793_793377


namespace probability_of_perfect_square_sum_l793_793612

def is_perfect_square (n : ℕ) : Prop :=
  n = 1*1 ∨ n = 2*2 ∨ n = 3*3 ∨ n = 4*4

theorem probability_of_perfect_square_sum :
  let total_outcomes := 64
  let perfect_square_sums := 12
  (perfect_square_sums / total_outcomes : ℚ) = 3 / 16 :=
by
  sorry

end probability_of_perfect_square_sum_l793_793612


namespace collinearity_condition_l793_793553

-- Definition of vertices of the regular hexagon in complex plane
def A : ℂ := 2
def B : ℂ := 2 * complex.exp (complex.I * π / 3)
def C : ℂ := 2 * complex.exp (complex.I * 2 * π / 3)
def D : ℂ := -2
def E : ℂ := 2 * complex.exp (complex.I * 4 * π / 3)
def F : ℂ := 2 * complex.exp (complex.I * -π / 3)

-- Definitions of points M and N
def M (r : ℝ) : ℂ := (1 - r) * A + r * C
def N (r : ℝ) : ℂ := (1 - r) * C + r * E

-- The main theorem statement
theorem collinearity_condition (r : ℝ) :
  (∃ r : ℝ, (M r).re * (N r).im = (N r).re * (M r).im) ↔ r = (real.sqrt 3) / 3 :=
sorry

end collinearity_condition_l793_793553


namespace a_minus_b_eq_eight_l793_793901

theorem a_minus_b_eq_eight
  (a b : ℕ)
  (h1 : Nat.coprime a b)
  (h2 : a > b)
  (h3 : (a^3 - b^3) / (a - b)^3 = 191 / 7) :
  a - b = 8 :=
sorry

end a_minus_b_eq_eight_l793_793901


namespace find_DE_length_l793_793468

noncomputable def triangle_DE_length (A B C D E : Type) (BC CD DE : ℝ) (C_angle : ℝ) : Prop :=
  BC = 40 ∧ C_angle = 45 ∧ CD = 20 ∧ DE = 20

theorem find_DE_length {A B C D E : Type} (BC CD DE : ℝ) (C_angle : ℝ) 
  (hBC : BC = 40) (hC_angle : C_angle = 45) (hCD : CD = 20) : DE = 20 :=
by {
  have hDE : DE = 20, sorry,
  exact hDE
}

end find_DE_length_l793_793468


namespace max_min_arc_sum_l793_793069

theorem max_min_arc_sum :
  ∀ (R B G : ℕ), R = 40 → B = 30 → G = 20 → R + B + G = 90 →
  (∀ a b c : ℕ, a = R * B ∧ b = R * G ∧ c = B * G → 
    (1 * a + 2 * b + 3 * c = 4600) ∧ (0 = 0)) :=
by {
  intros R B G hR hB hG hSum a b c hcounts,
  have h1 : a = 40 * 30, by { rw hR, rw hB, exact hcounts.1 },
  have h2 : b = 40 * 20, by { rw hR, rw hG, exact hcounts.2 },
  have h3 : c = 30 * 20, by { rw hB, rw hG, exact hcounts.3 },
  rw [h1, h2, h3], rw [mul_comm 1, mul_comm 2, mul_comm 3],
  have hcalc : 1 * (40 * 30) + 2 * (40 * 20) + 3 * (30 * 20) = 4600, by {
    rw [mul_assoc, mul_comm 1, mul_comm 2, mul_comm 3], numeral, exact hcounts,
  },
  exact λ a b c, by {rw hcalc, exact and.intro rfl rfl},
  sorry
}

end max_min_arc_sum_l793_793069


namespace radical_not_simplified_l793_793698

theorem radical_not_simplified (A : ℝ) (B : ℝ) (C : ℝ) (D : ℝ)
  (hA : A = real.sqrt 10)
  (hB : B = real.sqrt 8)
  (hC : C = real.sqrt 6)
  (hD : D = real.sqrt 2) : 
  ¬(B = 2 * real.sqrt 2) :=
sorry

end radical_not_simplified_l793_793698


namespace television_selection_l793_793757

theorem television_selection :
  let typeA := 4
  let typeB := 5
  (nat.choose typeA 2 * nat.choose typeB 1) + (nat.choose typeA 1 * nat.choose typeB 2) = 70 :=
by
  let typeA := 4
  let typeB := 5
  have h1 : nat.choose typeA 2 * nat.choose typeB 1 = 30 := by sorry
  have h2 : nat.choose typeA 1 * nat.choose typeB 2 = 40 := by sorry
  calc
    (nat.choose typeA 2 * nat.choose typeB 1) + (nat.choose typeA 1 * nat.choose typeB 2)
        = 30 + 40 : by rw [h1, h2]
    ... = 70 : by norm_num

end television_selection_l793_793757


namespace boxes_needed_l793_793067

theorem boxes_needed (balls : ℕ) (balls_per_box : ℕ) (h1 : balls = 10) (h2 : balls_per_box = 5) : balls / balls_per_box = 2 := by
  sorry

end boxes_needed_l793_793067


namespace find_m_in_range_l793_793390

theorem find_m_in_range :
  ∃ (m : ℤ), m ∈ set.Icc 150 200 ∧ (∃ (c d : ℤ), c ≡ 25 [MOD 53] ∧ d ≡ 88 [MOD 53] ∧ c - d ≡ m [MOD 53]) :=
sorry

end find_m_in_range_l793_793390


namespace y_at_x8_l793_793422

theorem y_at_x8 (k : ℝ) (y : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, y x = k * x^(1/3))
  (h2 : y 64 = 4 * Real.sqrt 3) :
  y 8 = 2 * Real.sqrt 3 :=
by
  sorry

end y_at_x8_l793_793422


namespace find_DE_length_l793_793466

noncomputable def triangle_DE_length (A B C D E : Type) (BC CD DE : ℝ) (C_angle : ℝ) : Prop :=
  BC = 40 ∧ C_angle = 45 ∧ CD = 20 ∧ DE = 20

theorem find_DE_length {A B C D E : Type} (BC CD DE : ℝ) (C_angle : ℝ) 
  (hBC : BC = 40) (hC_angle : C_angle = 45) (hCD : CD = 20) : DE = 20 :=
by {
  have hDE : DE = 20, sorry,
  exact hDE
}

end find_DE_length_l793_793466


namespace opposite_of_negative_fraction_l793_793971

theorem opposite_of_negative_fraction :
  -(-1 / 2023) = (1 / 2023) :=
by
  sorry

end opposite_of_negative_fraction_l793_793971


namespace magnitude_b_l793_793840

noncomputable theory

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Conditions
def mag_a : real := ‖a‖ = 3
def mag_diff_ab : real := ‖a - b‖ = 5
def dot_ab : real := inner a b = 1

-- Goal
theorem magnitude_b (h1 : ‖a‖ = 3) (h2 : ‖a - b‖ = 5) (h3 : inner a b = 1) : ‖b‖ = 3 * real.sqrt 2 :=
by 
  sorry

end magnitude_b_l793_793840


namespace part1_part2_l793_793333

open Real

-- Condition and definition for Part 1
variable (x : ℝ)
def m : ℝ × ℝ := (sqrt 3 * sin x, cos x)
def n : ℝ × ℝ := (cos x, cos x)
def p : ℝ × ℝ := (2 * sqrt 3, 1)

-- Part 1 statement
theorem part1 (h : (sqrt 3 * sin x) / cos x = 2 * sqrt 3) :
  m x.fst * n x.snd + m x.snd * n x.snd = (2 * sqrt 3 + 1) / 5 :=
sorry

-- Condition and definition for Part 2
def f : ℝ → ℝ := λ x, (sqrt 3 * sin x * cos x + cos x * cos x)

-- Part 2 statement
theorem part2 (h : x ∈ Ioc 0 (π / 3)) : 
  1 ≤ f x ∧ f x ≤ 3 / 2 :=
sorry

end part1_part2_l793_793333


namespace tangents_intersect_on_x_axis_l793_793754

noncomputable def hyperbola (a b x y : ℝ) : Prop := 
  b^2 * x^2 - a^2 * y^2 = a^2 * b^2

def pointA (a : ℝ) : ℝ × ℝ := (-a, 0)
def pointB (a : ℝ) : ℝ × ℝ := (a, 0)
def pointE (a b ϕ : ℝ) : ℝ × ℝ := (a * sec ϕ, b * tan ϕ)
def pointF (a b θ : ℝ) : ℝ × ℝ := (a * sec θ, b * tan θ)

theorem tangents_intersect_on_x_axis (a b ϕ θ : ℝ) :
  let E := pointE a b ϕ in
  let F := pointF a b θ in
  let P := (a * (tan θ - tan ϕ)) / (sec ϕ * tan θ - sec θ * tan ϕ) in
  let C := (-(a * ((sec ϕ + 1) * tan θ + (sec θ - 1) * tan ϕ)) / ((sec θ - 1) * tan ϕ - (sec ϕ + 1) * tan θ)) in
  P = C →
  ∀ x : ℝ, PC_perpendicular_AB (pointA a) (pointB a) (x, 0) :=
sorry -- Proof goes here

end tangents_intersect_on_x_axis_l793_793754


namespace find_line_equation_l793_793294

theorem find_line_equation (x y : ℝ) : 
  (∃ A B, (A.x^2 / 6 + A.y^2 / 3 = 1) ∧ (B.x^2 / 6 + B.y^2 / 3 = 1) ∧
  (A.x > 0 ∧ A.y > 0) ∧ (B.x > 0 ∧ B.y > 0) ∧
  let M := (-B.y, 0) in
  let N := (0, B.y) in 
  (abs (M.x - A.x) = abs (N.y - B.y)) ∧ 
  (abs (M.x - N.x + M.y - N.y) = 2 * sqrt 3)) →
  x + sqrt 2 * y - 2 * sqrt 2 = 0 := 
sorry

end find_line_equation_l793_793294


namespace probability_two_8sided_dice_sum_perfect_square_l793_793605

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def probabilityPerfectSquareSum (dice_sides : ℕ) (perfect_squares : List ℕ) : ℚ :=
  let outcomes := (dice_sides * dice_sides)
  let favorable_outcomes := perfect_squares.sum (λ ps, 
    (List.filter (λ (pair : ℕ × ℕ), pair.fst + pair.snd = ps) ((List.range dice_sides).product (List.range dice_sides))).length)
  favorable_outcomes /. outcomes

theorem probability_two_8sided_dice_sum_perfect_square :
  probabilityPerfectSquareSum 8 [4, 9, 16] = 3 / 16 := sorry

end probability_two_8sided_dice_sum_perfect_square_l793_793605


namespace arctan_sum_l793_793190

theorem arctan_sum : ∀ (a b : ℝ), 
  a = 3/7 → 
  b = 7/3 → 
  a * b = 1 → 
  a > 0 → 
  b > 0 → 
  Real.arctan a + Real.arctan b = Real.pi / 2 :=
by intros a b ha hb hab ha_pos hb_pos
   rw [ha, hb, hab]
   sorry

end arctan_sum_l793_793190


namespace compare_fx_l793_793789

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ :=
  x^2 - b * x + c

theorem compare_fx (b c : ℝ) (x : ℝ) (h1 : ∀ x : ℝ, f (1 - x) b c = f (1 + x) b c) (h2 : f 0 b c = 3) :
  f (2^x) b c ≤ f (3^x) b c :=
by
  sorry

end compare_fx_l793_793789


namespace coefficient_x9_is_zero_l793_793236

theorem coefficient_x9_is_zero: 
  (∃ k, k ∈ finset.range (10) ∧ 
        (binom 9 k * ((x^3 / 3) ^ (9 - k)) * ((-3 / x^2) ^ k)) = (0 : ℤ) ∧ 27 - 5 * k = 9) := 
by {
  sorry
}

end coefficient_x9_is_zero_l793_793236


namespace length_of_metallic_sheet_l793_793672

variable (L : ℝ) (width side volume : ℝ)

theorem length_of_metallic_sheet (h1 : width = 36) (h2 : side = 8) (h3 : volume = 5120) :
  ((L - 2 * side) * (width - 2 * side) * side = volume) → L = 48 := 
by
  intros h_eq
  sorry

end length_of_metallic_sheet_l793_793672


namespace Otimes_eval_l793_793218

-- Definitions based on conditions
def p : ℝ := 2
def q : ℝ := 5
def r : ℝ := 3
def s : ℝ := -1

-- Assertion to be proved
theorem Otimes_eval :
  (p ≠ r) → (q ≠ p) → 
  (\otimes (x y z : ℝ) : ℝ := x / (y - z)) ( 
    \otimes p q r
  ) (
    \otimes q r p
  ) (
    \otimes r p q
  ) = \otimes 1 5 -1 := 
   by sorry

end Otimes_eval_l793_793218


namespace exists_nat_N_l793_793490

-- Definitions
def is_divisor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

def smallest_mod (n m : ℕ) : Prop := ∀ d1 d2 ∈ (set_of (is_divisor n)), d1 ≠ d2 → (d1 % m) ≠ (d2 % m)

def f(n : ℕ) : ℕ := 
  Nat.find (smallest_mod n)

-- Main Statement
theorem exists_nat_N (ε : ℝ) (hε : ε > 0) : 
  ∃ (N : ℕ), ∀ n ≥ N, (f(n) : ℝ) ≤ (n : ℝ) ^ ε :=
sorry

end exists_nat_N_l793_793490


namespace arctan_sum_l793_793200

theorem arctan_sum (a b : ℝ) (h1 : a = 3) (h2 : b = 7) : 
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 := 
by 
  rw [h1, h2]
  sorry

end arctan_sum_l793_793200


namespace geometric_sequence_sum_l793_793878

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 3 + a 5 = 20)
  (h2 : a 4 = 8)
  (h3 : ∀ n, a (n + 1) = a n * q) :
  a 2 + a 6 = 34 := 
sorry

end geometric_sequence_sum_l793_793878


namespace parallel_line_no_intersect_l793_793426

open Set

variable {ℝ : Type} (a : Set ℝ) (α : Set (Set ℝ))

-- Definition of a line being parallel to a plane
def parallel (a : Set ℝ) (α : Set (Set ℝ)) : Prop :=
  ∀ (x : ℝ), (x ∈ a) → ∀ (y : Set ℝ), (y ∈ α) → (a ∩ y = ∅)

-- Statement of the main theorem
theorem parallel_line_no_intersect (a : Set ℝ) (α : Set (Set ℝ)) (h : parallel a α) : ∀ (y : Set ℝ), y ∈ α → (a ∩ y = ∅) :=
  by sorry

end parallel_line_no_intersect_l793_793426


namespace crayons_per_friend_l793_793729

theorem crayons_per_friend (total_crayons : ℕ) (num_friends : ℕ) (h1 : total_crayons = 210) (h2 : num_friends = 30) : total_crayons / num_friends = 7 :=
by
  sorry

end crayons_per_friend_l793_793729


namespace percentage_of_a_added_to_get_x_l793_793569

variable (a b x m : ℝ) (P : ℝ) (k : ℝ)
variable (h1 : a / b = 4 / 5)
variable (h2 : x = a * (1 + P / 100))
variable (h3 : m = b * 0.2)
variable (h4 : m / x = 0.14285714285714285)

theorem percentage_of_a_added_to_get_x :
  P = 75 :=
by
  sorry

end percentage_of_a_added_to_get_x_l793_793569


namespace student_distribution_arrangement_ways_l793_793443

-- Definitions for Part 1
def total_students := 9
def selection_probability := 16 / 21

-- Prove the number of male and female students given probability condition
theorem student_distribution (x : ℕ) (y : ℕ) 
  (h : x + y = total_students) 
  (h_probability : (∑ i in finset.range x, (nat.choose x 3 * nat.choose y (3 - i))) / (nat.choose total_students 3) = selection_probability) : 
  x = 3 ∧ y = 6 := 
sorry

-- Definitions for Part 2
def num_male_students := 6
def num_female_students := 3

-- Prove the total number of ways to arrange the students
theorem arrangement_ways: 
  num_male_students = 6 → 
  num_female_students = 3 → 
  num_ways := 15 * 144 * 8 
  17280 :=
sorry

end student_distribution_arrangement_ways_l793_793443


namespace one_figure_is_quadrilateral_l793_793645

theorem one_figure_is_quadrilateral (A B C K L : Point) (hABC : Triangle A B C)
  (hAKC : Triangle A K C)
  (hBLC : Triangle B L C)
  (hABL : Triangle A B L)
  (h_equal_areas : area hAKC = area hBLC ∧ area hBLC = area hABL)
  (h_divides_triangle : divides_triangle A B C K L (hAKC, hBLC, hABL, quadrilateral A K L B)) :
  is_quadrilateral (quadrilateral A K L B) :=
by
  sorry -- proof goes here

end one_figure_is_quadrilateral_l793_793645


namespace two_vertical_asymptotes_l793_793228

theorem two_vertical_asymptotes (k : ℝ) : 
  (∀ x : ℝ, (x ≠ 3 ∧ x ≠ -2) → 
           (∃ δ > 0, ∀ ε > 0, ∃ x' : ℝ, x + δ > x' ∧ x' > x - δ ∧ 
                             (x' ≠ 3 ∧ x' ≠ -2) → 
                             |(x'^2 + 2 * x' + k) / (x'^2 - x' - 6)| > 1/ε)) ↔ 
  (k ≠ -15 ∧ k ≠ 0) :=
sorry

end two_vertical_asymptotes_l793_793228


namespace cylinder_unoccupied_volume_l793_793991

theorem cylinder_unoccupied_volume (r h_cylinder h_cone : ℝ) 
  (h : r = 10 ∧ h_cylinder = 30 ∧ h_cone = 15) :
  (π * r^2 * h_cylinder - 2 * (1/3 * π * r^2 * h_cone) = 2000 * π) :=
by
  rcases h with ⟨rfl, rfl, rfl⟩
  simp
  sorry

end cylinder_unoccupied_volume_l793_793991


namespace sum_of_ages_l793_793534

-- Definitions for Robert's and Maria's current ages
variables (R M : ℕ)

-- Conditions based on the problem statement
theorem sum_of_ages
  (h1 : R = M + 8)
  (h2 : R + 5 = 3 * (M - 3)) :
  R + M = 30 :=
by
  sorry

end sum_of_ages_l793_793534


namespace telescoping_product_fib_l793_793898

noncomputable def fib : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n + 2) := fib n + fib (n + 1)

theorem telescoping_product_fib :
  ∏ k in finset.range(48).map (finset.nat_cast 3), 
    (fib k^2 / (fib (k - 1) * fib (k + 1)))^2 = 4 / (fib 51)^2 :=
sorry

end telescoping_product_fib_l793_793898


namespace line_relation_l793_793160

-- Definitions based on conditions
variables (l a b : Line)

def parallel (x y : Line) : Prop := ∃ (v : Vector), (x.direction = v ∧ y.direction = v)
def skew (x y : Line) : Prop := ¬ (parallel x y ∨ ∃ p, x.contains p ∧ y.contains p)

-- Given Conditions
axiom h1 : parallel l a
axiom h2 : skew a b

-- Proof goal
theorem line_relation (l a b : Line) (h1 : parallel l a) (h2 : skew a b) : 
  (∃ p, l.contains p ∧ b.contains p) ∨ skew l b :=
sorry

end line_relation_l793_793160


namespace smallest_y_l793_793090

theorem smallest_y : ∃ y : ℤ, (7 / 11 : ℝ) < (y / 17 : ℝ) ∧ ∀ k : ℤ, (7 / 11 : ℝ) < (k / 17 : ℝ) → k ≥ 11 :=
by
  let y := 11
  have : (7 / 11 : ℝ) < (y / 17 : ℝ) := by norm_num
  use y
  split
  {
    exact this
  }
  {
    intros k h_k
    have : (119 / 11 : ℝ) ≈ 10.8181818 := by norm_num
    exact Int.ofNatGe_of_le this
  }

end smallest_y_l793_793090


namespace remaining_gallons_to_fill_tank_l793_793181

-- Define the conditions as constants
def tank_capacity : ℕ := 50
def rate_seconds_per_gallon : ℕ := 20
def time_poured_minutes : ℕ := 6

-- Define the number of gallons poured per minute
def gallons_per_minute : ℕ := 60 / rate_seconds_per_gallon

def gallons_poured (minutes : ℕ) : ℕ :=
  minutes * gallons_per_minute

-- The main statement to prove the remaining gallons needed
theorem remaining_gallons_to_fill_tank : 
  tank_capacity - gallons_poured time_poured_minutes = 32 :=
by
  sorry

end remaining_gallons_to_fill_tank_l793_793181


namespace machine_p_takes_longer_l793_793004

variable (MachineP MachineQ MachineA : Type)
variable (s_prockets_per_hr : MachineA → ℝ)
variable (time_produce_s_prockets : MachineP → ℝ → ℝ)

noncomputable def machine_a_production : ℝ := 3
noncomputable def machine_q_production : ℝ := machine_a_production + 0.10 * machine_a_production

noncomputable def machine_q_time : ℝ := 330 / machine_q_production
noncomputable def additional_time : ℝ := sorry -- Since L is undefined

axiom machine_p_time : ℝ
axiom machine_p_time_eq_machine_q_time_plus_additional : machine_p_time = machine_q_time + additional_time

theorem machine_p_takes_longer : machine_p_time > machine_q_time := by
  rw [machine_p_time_eq_machine_q_time_plus_additional]
  exact lt_add_of_pos_right machine_q_time sorry  -- Need the exact L to conclude


end machine_p_takes_longer_l793_793004


namespace circle_area_through_incenter_l793_793079

noncomputable def area_of_circle_through_A_O_C (A O C : ℝ × ℝ) (AB AC BC : ℝ) (h_isosceles : AB = AC) (h_lengths : AB = 8 ∧ AC = 8 ∧ BC = 6) (incenter_O : is_incenter O A B C) : ℝ :=
  let r := (56 : ℝ) / 22 in
  let area := π * r^2 in
  area

theorem circle_area_through_incenter (A O C : ℝ × ℝ) (h_isosceles : AB = AC) (h_lengths : AB = 8 ∧ AC = 8 ∧ BC = 6) (incenter_O : is_incenter O A B C) : 
  area_of_circle_through_A_O_C A O C 8 8 6 h_isosceles h_lengths incenter_O = (392 / 121) * π := 
sorry

end circle_area_through_incenter_l793_793079


namespace y_at_x8_l793_793421

theorem y_at_x8 (k : ℝ) (y : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, y x = k * x^(1/3))
  (h2 : y 64 = 4 * Real.sqrt 3) :
  y 8 = 2 * Real.sqrt 3 :=
by
  sorry

end y_at_x8_l793_793421


namespace cost_price_of_article_l793_793637

theorem cost_price_of_article (SP : ℝ) (ProfitPercent : ℝ) (h_SP : SP = 100) (h_ProfitPercent : ProfitPercent = 0.10) :
  ∀ (CP : ℝ), CP = SP / (1 + ProfitPercent) → CP = 90.91 :=
by
  intro CP h_CP
  rw [h_SP, h_ProfitPercent] at h_CP
  simp at h_CP
  exact h_CP

# Check the theorem statement
# example usage:
# have := cost_price_of_article 100 0.10 rfl rfl 90.91 sorry

end cost_price_of_article_l793_793637


namespace infinite_set_contains_all_positive_integers_l793_793702

open Set

theorem infinite_set_contains_all_positive_integers {B : Set ℕ} (h₁ : Infinite B)
  (h₂ : ∀ a b ∈ B, a > b → (a - b) / Nat.gcd a b ∈ B) :
  ∀ n : ℕ, n > 0 → n ∈ B :=
by
  intro n hn
  sorry

end infinite_set_contains_all_positive_integers_l793_793702


namespace faye_coloring_books_left_l793_793739

noncomputable def remaining_coloring_books (initial: ℝ) (given_away1: ℝ) (given_away_percent: ℝ) : ℝ :=
  let remaining_after_first = initial - given_away1
  let given_away2 = given_away_percent * remaining_after_first
  remaining_after_first - given_away2

theorem faye_coloring_books_left :
  remaining_coloring_books 52.5 38.2 0.25 = 10.725 := by
  sorry

end faye_coloring_books_left_l793_793739


namespace probability_of_perfect_square_sum_on_two_8_sided_dice_l793_793592

-- Define the dice and possible sums
def is_sum_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def pairs_rolled : finset (ℕ × ℕ) :=
  finset.univ.filter (λ p, p.1 + p.2 ∈ {4, 9, 16})

theorem probability_of_perfect_square_sum_on_two_8_sided_dice :
  (pairs_rolled.card : ℚ) / 64 = 3 / 16 :=
by
  sorry

end probability_of_perfect_square_sum_on_two_8_sided_dice_l793_793592


namespace cell_growth_after_nine_days_l793_793481

theorem cell_growth_after_nine_days :
  let initial_cells := 5
  let cells_after_first_split := (initial_cells * 2 * 0.75).toInt
  let cells_after_second_split := (cells_after_first_split * 2 * 0.75).toInt
  let cells_after_third_split := (cells_after_second_split * 2 * 0.75).toInt
  cells_after_third_split = 15 := by
  let initial_cells := 5
  let cells_after_first_split := (initial_cells * 2 * 0.75).toInt
  let cells_after_second_split := (cells_after_first_split * 2 * 0.75).toInt
  let cells_after_third_split := (cells_after_second_split * 2 * 0.75).toInt
  show cells_after_third_split = 15,
  from sorry

end cell_growth_after_nine_days_l793_793481


namespace vector_magnitude_b_l793_793847

variables (a b : EuclideanSpace ℝ 3)
variables (h1 : ∥a∥ = 3) (h2 : ∥a - b∥ = 5) (h3 : inner a b = 1)

theorem vector_magnitude_b : ∥b∥ = 3 * real.sqrt 2 :=
by
  sorry

end vector_magnitude_b_l793_793847


namespace correct_diff_is_D_l793_793632

noncomputable def diff_A (x : ℝ) := (x / (Real.log x))'
noncomputable def diff_B (x : ℝ) := (x^2 + Real.exp x)'
noncomputable def diff_C (x : ℝ) := (x * Real.cos x)'
noncomputable def diff_D (x : ℝ) := (x - 1 / x)'

theorem correct_diff_is_D (x : ℝ) (h : x ≠ 0) :
(diff_A x = (Real.log x - 1) / (Real.log x)^2) ∨ 
(diff_B x = 2 * x + Real.exp x) ∨ 
(diff_C x = Real.cos x - x * Real.sin x) ∨ 
(diff_D x = 1 + 1 / x^2) :=
by {
    sorry
}

end correct_diff_is_D_l793_793632


namespace circle_M_equation_line_l_equation_circle_N_passes_fixed_points_l793_793806

noncomputable def A : ℝ × ℝ := (0, 2)
noncomputable def B : ℝ × ℝ := (0, 4)
noncomputable def C : ℝ × ℝ := (1, 3)
noncomputable def D : ℝ × ℝ := (1/2, 2)

theorem circle_M_equation :
  ∃ (D E F : ℝ), 
  D = 0 ∧ E = -6 ∧ F = 8 ∧
  ∀ (x y : ℝ), 
  (x, y) = A ∨ (x, y) = B ∨ (x, y) = C →
  x^2 + y^2 + D*x + E*y + F = 0
:= sorry

theorem line_l_equation :
  ∃ (k : ℝ), 
  (k = 0 ∧ (∀ (x y : ℝ), 2 * x = 1/2 ∨ 6 * x - 8 * y + 13 = 0)) ∨
  (k ≠ 0 ∧ (∀ (x y : ℝ), x/2 * k - y + 1/2 k - 2 = 0))
:= sorry

theorem circle_N_passes_fixed_points :
  ∀ (k : ℝ), 
  A ≠ B ∧ 
  ∀ (x : ℝ), 
  k ≠ 0 →
  let P := if k = 0 then B else (k, -1/k)
  let E := (x, 0) ;
  let F := (4*k, 0) ;
  let circle_N_equation := ∀ x y : ℝ, (x - 2*k - (-1/k))^2 + y^2 = (2*k + 1/k)^2 in
  circle_N_equation 0 2*sqrt(2) = 0 ∧ circle_N_equation 0 -2*sqrt(2) = 0
:= sorry

end circle_M_equation_line_l_equation_circle_N_passes_fixed_points_l793_793806


namespace divide_ratio_BB1_l793_793932

variables {a h : ℝ} (A B C A1 B1 C1 D M S N P : ℝ × ℝ × ℝ)
variable [line_function : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ) → ℝ → (ℝ × ℝ × ℝ)]

-- Condition 1: Point D is the midpoint of edge A1 C1
def is_midpoint (D A1 C1 : ℝ × ℝ × ℝ) : Prop :=
  D = ((A1.1 + C1.1) / 2, (A1.2 + C1.2) / 2, (A1.3 + C1.3) / 2)

-- Condition 2: Plane of SMNP coincides with plane of ABC (implicitly defined by coordinates)
-- Condition 3: Vertex M lies on the extension of AC with |CM| = 1/2 |AC|
def on_extension_of_AC (M A C : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t ≠ 0 ∧ M = (C.1 + t * (A.1 - C.1), C.2 + t * (A.2 - C.2), C.3 + t * (A.3 - C.3)) ∧
  dist C M = 0.5 * dist A C

-- Condition 4: Edge SN passes through point D
def passes_through (N S D : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, N = S + t • (D - S)

-- Condition 5: Edge SP intersects segment BB1
def intersects_SP_BB1 (P S B B1 : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, P = S + t • (B1 - B)

-- Proof goal: Ratio in which segment BB1 is divided by intersection point is 3/4
theorem divide_ratio_BB1 (hD_midpoint : is_midpoint D A1 C1) 
  (h_position_M : on_extension_of_AC M A C)
  (h_SN_pass_D : passes_through N S D)
  (h_SP_intersects_BB1 : intersects_SP_BB1 P S B B1) : 
  ∃ t : ℝ, t = 3 / 4 :=
sorry

end divide_ratio_BB1_l793_793932


namespace length_de_l793_793477

open Triangle

/-- In triangle ABC, BC = 40 and ∠C = 45°. Let the perpendicular bisector
of BC intersect BC and AC at D and E, respectively. Prove that DE = 10√2. -/

theorem length_de 
  (ABC : Triangle)
  (B C : Point)
  (BC_40 : dist B C = 40)
  (angle_C_45 : ∠(B, C, AC) = 45)
  (D : Point)
  (midpoint_D : is_midpoint B C D)
  (E : Point)
  (intersection_E : is_perpendicular_bisector_intersection B C D E) :
  dist D E = 10 * real.sqrt 2 :=
begin
  -- proof steps would go here
  sorry
end

end length_de_l793_793477


namespace count_four_digit_even_numbers_l793_793814

def digitSet := {0, 2, 4, 6, 8}

def validFirstDigitSet := {2, 4, 6, 8}

noncomputable def countValidFourDigitEvenNumbers : Nat :=
  Set.card validFirstDigitSet * (Set.card digitSet * Set.card digitSet * Set.card digitSet)

theorem count_four_digit_even_numbers : countValidFourDigitEvenNumbers = 500 :=
by
  rw [countValidFourDigitEvenNumbers]
  norm_num
  sorry

end count_four_digit_even_numbers_l793_793814


namespace count_two_digit_numbers_less_35_l793_793371

open Nat

theorem count_two_digit_numbers_less_35 : 
  let two_digit_numbers := (finset.Ico 10 100) -- The range of two-digit numbers
  let count_satisfying := two_digit_numbers.filter (λ n, (n / 10 < 3 ∨ n % 10 < 5)).card
  count_satisfying = 55 :=
by
  -- Placeholder for the actual proof.
  sorry

end count_two_digit_numbers_less_35_l793_793371


namespace num_suitable_two_digit_numbers_l793_793378

/-- 
How many two-digit numbers have at least one digit that is smaller than the corresponding digit in the number 35?
-/
theorem num_suitable_two_digit_numbers : 
  let two_digit_numbers := { n : ℕ | 10 ≤ n ∧ n ≤ 99 },
      suitable_numbers := { n ∈ two_digit_numbers | (n / 10 < 3) ∨ (n % 10 < 5) } in
  suitable_numbers.card = 55 :=
by
  sorry

end num_suitable_two_digit_numbers_l793_793378


namespace sum_of_squares_of_roots_eq_226_l793_793227

theorem sum_of_squares_of_roots_eq_226 (s_1 s_2 : ℝ) (h_eq : ∀ x, x^2 - 16 * x + 15 = 0 → (x = s_1 ∨ x = s_2)) :
  s_1^2 + s_2^2 = 226 := by
  sorry

end sum_of_squares_of_roots_eq_226_l793_793227


namespace goldfish_equal_in_3_months_l793_793184

theorem goldfish_equal_in_3_months 
  (Brent_growth : ∀ (n : ℕ), 3 ^ (n + 1) = 3 ^ (n + 4))
  : ∃ n : ℕ, 3 ^ (n + 1) = 3 ^ (n + 4) ∧ n = 3 :=
begin
  use 3,
  split,
  { exact Brent_growth 3 },
  { refl }
end

end goldfish_equal_in_3_months_l793_793184


namespace beer_drawing_time_l793_793653

theorem beer_drawing_time :
  let rate_A := 1 / 5
  let rate_C := 1 / 4
  let combined_rate := 9 / 20
  let extra_beer := 12
  let total_drawn := 48
  let t := total_drawn / combined_rate
  t = 48 * 20 / 9 :=
by {
  sorry -- proof not required
}

end beer_drawing_time_l793_793653


namespace police_force_females_l793_793013

def number_of_female_officers_on_police_force (total_officers_on_duty : ℕ) 
                                               (female_officers_on_duty_percent : ℝ)
                                               (male_to_female_ratio_on_duty : ℝ) : ℕ :=
  let F := total_officers_on_duty / (1 + male_to_female_ratio_on_duty)
  let T := F / female_officers_on_duty_percent
  T  -- We expect this to be 436 given the conditions

theorem police_force_females (total_officers_on_duty : ℕ) 
                              (female_officers_on_duty_percent : ℝ) 
                              (male_to_female_ratio_on_duty : ℝ) : 
                              number_of_female_officers_on_police_force total_officers_on_duty 
                                                                         female_officers_on_duty_percent 
                                                                         male_to_female_ratio_on_duty 
                                = 436 :=
  sorry

end police_force_females_l793_793013


namespace find_length_second_platform_l793_793148

noncomputable def length_second_platform : Prop :=
  let train_length := 500  -- in meters
  let time_cross_platform := 35  -- in seconds
  let time_cross_pole := 8  -- in seconds
  let second_train_length := 250  -- in meters
  let time_cross_second_train := 45  -- in seconds
  let platform1_scale := 0.75
  let time_cross_platform1 := 27  -- in seconds
  let train_speed := train_length / time_cross_pole
  let platform1_length := train_speed * time_cross_platform1 - train_length
  let platform2_length := platform1_length / platform1_scale
  platform2_length = 1583.33

/- The proof is omitted -/
theorem find_length_second_platform : length_second_platform := sorry

end find_length_second_platform_l793_793148


namespace sum_numerator_denominator_fraction_prob_l793_793254

theorem sum_numerator_denominator_fraction_prob :
  let S := finset.range 1000 in
  let prob_fitting (S : finset ℕ) : ℚ := 1 / 4 in
  (prob_fitting S).numerator + (prob_fitting S).denominator = 5 := 
by {
  sorry
}

end sum_numerator_denominator_fraction_prob_l793_793254


namespace measure_of_angle_C_l793_793705

theorem measure_of_angle_C (A B : ℝ) (h1 : A + B = 180) (h2 : A = 5 * B) : A = 150 := by
  sorry

end measure_of_angle_C_l793_793705


namespace problem_solution_l793_793826

-- Define the conditions
variables {a c b d x y z q : Real}
axiom h1 : a^x = c^q ∧ c^q = b
axiom h2 : c^y = a^z ∧ a^z = d

-- State the theorem
theorem problem_solution : xy = zq :=
by
  sorry

end problem_solution_l793_793826


namespace xyz_inequality_l793_793935

theorem xyz_inequality (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  3 * (x^2 * y^2 + x^2 * z^2 + y^2 * z^2) - 2 * x * y * z * (x + y + z) ≤ 3 := by
  sorry

end xyz_inequality_l793_793935


namespace find_r_l793_793556

noncomputable def geometric_series (a r n : ℕ) : ℚ :=
  a * r ^ n

noncomputable def sum_geometric_series (a r : ℚ) (n : ℕ) : ℚ :=
  if r = 1 then a * n else a * (1 - r ^ n) / (1 - r)

noncomputable def odd_powers_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (r * (1 - r ^ (2 * (n // 2))) / (1 - r ^ 2))

theorem find_r (a r : ℚ) (S S_odd : ℚ) (h1 : S = sum_geometric_series a r n)
  (h2 : S_odd = odd_powers_sum a r n) (h3 : S = 24) (h4 : S_odd = 9) :
  r = 3/5 :=
by
  sorry

end find_r_l793_793556


namespace area_calculation_l793_793661

variables {a b c d p q α β γ : ℝ}
variables (h1 : a ≠ 0) (h2 : α < β) (h3 : β < γ)
variables (h4 : ∀ x, (x = α ∨ x = β ∨ x = γ) → ax^3 + bx^2 + cx + d = px + q)

def area_between_cubic_and_line : ℝ :=
  (|a| / 12 * (γ - α)^3 * |2 * β - γ - α|)

theorem area_calculation :
  ∫ x in α .. γ, a * (x - α) * (x - β) * (x - γ) =
  |a| / 12 * (γ - α)^3 * |2 * β - γ - α| :=
sorry

end area_calculation_l793_793661


namespace solve_inequality_l793_793733

theorem solve_inequality (x : ℝ) : 3 * x^2 + 2 * x - 3 > 12 - 2 * x → x < -3 ∨ x > 5 / 3 :=
sorry

end solve_inequality_l793_793733


namespace arctan_sum_l793_793209

theorem arctan_sum {a b : ℝ} (h3 : a = 3) (h7 : b = 7) :
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 :=
by
  sorry

end arctan_sum_l793_793209


namespace length_DE_l793_793464

-- Definition and given conditions
variables {A B C D E : Type}
variables [Point (Triangle ABC)]

-- Given BC = 40 and angle C = 45 degrees
constants (BC : Real) (angleC : Real)
constants (midpoint : BC → D) (perpendicular_bisector : BC → AC → E)
constant (triangle_CDE_454590 : Is454590Triangle C D E)

-- Definitions for points D and E
noncomputable def midpoint_of_BC (P: BC) : D :=
  midpoint P

noncomputable def intersection_perpendicular_bisector_AC (P: BC → AC) : E :=
  perpendicular_bisector P

-- Prove length of DE == 20
theorem length_DE : length DE = 20 := sorry

end length_DE_l793_793464


namespace roots_of_polynomial_l793_793948

theorem roots_of_polynomial:
  ∃ x1 x2 x3 : ℝ, 
  (x1^3 - 5 * x1^2 - 9 * x1 + 45 = 0) ∧
  (x2^3 - 5 * x2^2 - 9 * x2 + 45 = 0) ∧
  (x3^3 - 5 * x3^2 - 9 * x3 + 45 = 0) ∧
  (x1 = 3) ∧ (x2 = -3) ∧ (x3 = 5) ∧
  (Polynomials.Root x1) ∧ 
  (Polynomials.Root x2) ∧ 
  (Polynomials.Root x3) :=
begin
  sorry
end

end roots_of_polynomial_l793_793948


namespace max_value_of_dot_product_l793_793332

variable (a b c : ℝ × ℝ)

def is_unit_vector (v : ℝ × ℝ) : Prop :=
  ∥v∥ = 1

def angle_is_sixty_degrees (a b : ℝ × ℝ) : Prop :=
  let θ := real.angle (vectorAngle a) (vectorAngle b)
  θ = real.pi / 3

theorem max_value_of_dot_product
  (unit_a : is_unit_vector a)
  (unit_b : is_unit_vector b)
  (unit_c : is_unit_vector c)
  (in_same_plane : ∃ α : ℝ, c = (cos α, -sin α))
  (sixty_degrees_between_ab : angle_is_sixty_degrees a b)
  : (∥a - b∥ * ∥a - 2*c∥) ≤ 5/2 := sorry

end max_value_of_dot_product_l793_793332


namespace minnie_slower_by_33_minutes_l793_793921

-- Define the speeds in kilometers per hour.
def minnie_speed_flat : ℝ := 25
def minnie_speed_downhill : ℝ := 35
def minnie_speed_uphill : ℝ := 10
def penny_speed_flat : ℝ := 35
def penny_speed_downhill : ℝ := 50
def penny_speed_uphill : ℝ := 15

-- Define the distances in kilometers.
def distance_ab : ℝ := 12
def distance_bc : ℝ := 18
def distance_ca : ℝ := 24

-- Define the time it takes for Minnie to complete each segment of the journey.
def minnie_time_uphill : ℝ := distance_ab / minnie_speed_uphill
def minnie_time_downhill : ℝ := distance_bc / minnie_speed_downhill
def minnie_time_flat : ℝ := distance_ca / minnie_speed_flat

-- Define the time it takes for Penny to complete each segment of the journey.
def penny_time_flat : ℝ := distance_ca / penny_speed_flat
def penny_time_uphill : ℝ := distance_bc / penny_speed_uphill
def penny_time_downhill : ℝ := distance_ab / penny_speed_downhill

-- Define the total times for both Minnie and Penny.
def minnie_total_time : ℝ := minnie_time_uphill + minnie_time_downhill + minnie_time_flat
def penny_total_time : ℝ := penny_time_flat + penny_time_uphill + penny_time_downhill

-- Define the time difference in minutes.
def time_difference : ℝ := (minnie_total_time - penny_total_time) * 60

-- Prove the time difference is 33 minutes.
theorem minnie_slower_by_33_minutes : time_difference = 33 :=
by
  sorry

end minnie_slower_by_33_minutes_l793_793921


namespace sum_of_fraction_numerator_denominator_l793_793102

theorem sum_of_fraction_numerator_denominator :
  let x := 0.343434...
  in let fraction := (34 / 99 : ℚ)
  in let sum := fraction.num + fraction.den 
  in (x : ℚ) = fraction ∧ fraction.isReduced → sum = 133 :=
by
  sorry

end sum_of_fraction_numerator_denominator_l793_793102


namespace angle_C_parallel_lines_l793_793915

theorem angle_C_parallel_lines (l m : line) (h_parallel: l ≠ m ∧ parallel l m)
  (angle_A angle_B : ℝ) (hA : angle_A = 100) (hB : angle_B = 130) :
  ∃ angle_C : ℝ, angle_C = 130 :=
by
  use 130
  sorry

end angle_C_parallel_lines_l793_793915


namespace smallest_divisor_100_factorial_gt_100_l793_793052

theorem smallest_divisor_100_factorial_gt_100 : 
  ∃ n > 100, (n ∣ Nat.factorial 100) ∧ ∀ m > 100, (m ∣ Nat.factorial 100) → n ≤ m :=
begin
  use 102,
  split,
  { exact lt_nat.mp (by decide), },
  split,
  { exact sorry, },
  { intros m h1 h2,
    exact sorry, },
end

end smallest_divisor_100_factorial_gt_100_l793_793052


namespace find_c_l793_793743

noncomputable def P (c : ℝ) : Polynomial ℝ := 3 * (Polynomial.X ^ 3) + c * (Polynomial.X ^ 2) - 8 * (Polynomial.X) + 50
def D : Polynomial ℝ := 3 * (Polynomial.X) + 5

theorem find_c (c : ℝ) :
  Polynomial.mod_by_monic (P c) (Polynomial.X - Polynomial.C (-5/3)) = Polynomial.C 7 → c = 18/25 :=
sorry

end find_c_l793_793743


namespace milk_to_water_ratio_after_combining_vessels_l793_793616

open Nat

def ratio_of_vessels (a b : ℕ) : ℕ × ℕ := (a, b)

def combine_ratios (r1 r2 : ℕ × ℕ) : ℕ × ℕ := 
  (r1.1 + r2.1, r1.2 + r2.2)

theorem milk_to_water_ratio_after_combining_vessels :
  let vessel1 := ratio_of_vessels 4 2 in
  let vessel2 := ratio_of_vessels 5 1 in
  combine_ratios vessel1 vessel2 = (9, 3) ∧ (9 / Nat.gcd 9 3 = 3 ∧ 3 / Nat.gcd 9 3 = 1) :=
by
  sorry

end milk_to_water_ratio_after_combining_vessels_l793_793616


namespace baker_still_has_45_pastries_l793_793710

def baker_pastries_remaining (pastries_made pastries_sold: ℕ) : ℕ :=
  pastries_made - pastries_sold

theorem baker_still_has_45_pastries
  (cakes_made pastries_made cakes_sold pastries_sold: ℕ)
  (h₁: cakes_made = 7)
  (h₂: pastries_made = 148)
  (h₃: cakes_sold = 15)
  (h₄: pastries_sold = 103) :
  baker_pastries_remaining pastries_made pastries_sold = 45 :=
by {
  unfold baker_pastries_remaining,
  rw [h₂, h₄],
  norm_num,
  sorry
}

end baker_still_has_45_pastries_l793_793710


namespace exam_test_plan_possible_l793_793044

-- Definitions based on given conditions
noncomputable def topics : ℕ := 25
noncomputable def questions_per_topic : ℕ := 8
noncomputable def questions_per_test : ℕ := 4
noncomputable def total_tests : ℕ := 50

-- Theorem statement
theorem exam_test_plan_possible :
  (∃ (tests : list (set (ℕ × ℕ))), 
    tests.length = total_tests ∧ 
    (∀ q ∈ tests, q.card = questions_per_test) ∧ 
    (∃ q_in_tests, ∀ t1 t2 ∈ (finset.range topics).image (λ t, finset.range questions_per_topic.map (λ q, (t, q))), q_in_tests t1 t2 ∧
    ∀ question_in_all_tests, (∃! t, q_to_tuple ∈ t) (0) (list.length_exclude_pair_val) (∀ q1 q2 ∈ tests, q_in_tests q1 q2))) :=
sorry

end exam_test_plan_possible_l793_793044


namespace simplify_fraction_l793_793540

theorem simplify_fraction :
  (21 / 25) * (35 / 45) * (75 / 63) = 35 / 9 :=
by
  sorry

end simplify_fraction_l793_793540


namespace product_evaluation_l793_793232

theorem product_evaluation (n : ℕ) (h : n = 5) : (n - 3) * (n - 2) * (n - 1) * n * (n + 1) = 720 :=
by
  rw h
  norm_num
  sorry

end product_evaluation_l793_793232


namespace intersection_line_circle_diameter_l793_793321

noncomputable def length_of_AB : ℝ := 2

theorem intersection_line_circle_diameter 
  (x y : ℝ)
  (h_line : x - 2*y - 1 = 0)
  (h_circle : (x - 1)^2 + y^2 = 1) :
  |(length_of_AB)| = 2 := 
sorry

end intersection_line_circle_diameter_l793_793321


namespace intersection_of_A_and_B_l793_793271

-- Definitions of the sets
def A : set ℝ := {x | 0 ≤ x ∧ x < 2}
def B : set ℤ := {-1, 0, 1, 2}

-- Proof statement that we need to prove
theorem intersection_of_A_and_B : {x : ℝ | x ∈ A ∧ x ∈ ↑B} = {0, 1} :=
by
  sorry

end intersection_of_A_and_B_l793_793271


namespace ratio_square_to_rectangle_l793_793639

variable (s : ℝ)  -- side length of the square
variable (x y : ℝ)  -- lengths AE and AG 
variable (abcd aegf : ℝ) 

def area_of_square (s : ℝ) : ℝ := s^2
def area_of_rectangle (x y : ℝ) : ℝ := x * y

def ratio (a b : ℝ) : ℝ := a / b

axioms (H1 : aegf = 0.25 * (area_of_square s))
       (H2 : x = 8 * y)

theorem ratio_square_to_rectangle : 
  ratio (area_of_square s) aegf = 4 := 
sorry

end ratio_square_to_rectangle_l793_793639


namespace negation_of_existence_l793_793502

theorem negation_of_existence :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by
  sorry

end negation_of_existence_l793_793502


namespace find_line_l_l793_793304

theorem find_line_l (A B M N : ℝ × ℝ)
  (h1 : ∃ x1 y1 x2 y2, A = (x1, y1) ∧ B = (x2, y2) ∧ x1 > 0 ∧ y1 > 0 ∧ x2 > 0 ∧ y2 > 0)
  (h2 : M = (x1, 0))
  (h3 : N = (0, y2))
  (h4 : abs (| M.1 - A.1 |) = abs (| N.2 - B.2 |))
  (h5 : dist M N = 2 * sqrt 3) :
  ∃ k m : ℝ, k < 0 ∧ m > 0 ∧ (∀ x y : ℝ, (y = k * x + m ↔ x + sqrt 2 * y - 2 * sqrt 2 = 0)) := sorry

end find_line_l_l793_793304


namespace bus_prob_at_least_two_days_on_time_l793_793306

noncomputable def bus_on_time (p : ℚ) : ℚ :=
  let q := 1 - p
  let P_X_2 := (3.choose 2) * (p^2) * (q^1)
  let P_X_3 := (3.choose 3) * (p^3) * (q^0)
  P_X_2 + P_X_3

theorem bus_prob_at_least_two_days_on_time :
  bus_on_time (3/5) = 81/125 :=
by
  sorry

end bus_prob_at_least_two_days_on_time_l793_793306


namespace college_potential_l793_793179

theorem college_potential (n : ℕ) (c r s : ℝ) (x : ℤ) (h1 : n = 40) (h2 : c = 3.975) (h3 : r = 3.995) (h4 : s = 4.0) : 
  (x : ℝ) = (r * (n + x) - c * n) / (s - r) := 
by 
  sorry

example : college_potential 40 3.975 3.995 4.0 160 :=
by 
  simp [college_potential]

end college_potential_l793_793179


namespace average_infection_per_round_l793_793229

theorem average_infection_per_round (x : ℝ) (h1 : 1 + x + x * (1 + x) = 100) : x = 9 :=
sorry

end average_infection_per_round_l793_793229


namespace expected_area_red_by_Sarah_l793_793538

-- Defining the initial circle radius
def O0_radius := 1

-- Defining the function f(r) for the area computation
def f (r : ℝ) (x : ℝ) : ℝ := (π * x) * r^2

-- Expected value of the area colored red by Sarah
theorem expected_area_red_by_Sarah 
    (O0_radius = 1)
    (O (n : ℕ) : set ℝ := { p | dist p 0 ≤ O0_radius }) :
    let x := 6 / 13 in
    f 1 x = π * (6 / 13) := by
  sorry

end expected_area_red_by_Sarah_l793_793538


namespace find_m_given_regression_and_values_l793_793309

theorem find_m_given_regression_and_values :
  let x_values : List ℝ := [1, 3, 4, 5, 7]
  let y_values : List (ℝ → ℝ) := [λ m, 1, λ m, m, λ m, 2 * m + 1, λ m, 2 * m + 3, λ m, 10]
  let regression_eq (x : ℝ) : ℝ := 1.3 * x + 0.8
  ∃ m : ℝ, 3 = m :=
by sorry

end find_m_given_regression_and_values_l793_793309


namespace correct_statements_l793_793247

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - 13 / 4 * Real.pi)

theorem correct_statements :
    (f (Real.pi / 8) = 0) ∧ 
    (∀ x, 2 * Real.sin (2 * (x - 5 / 8 * Real.pi)) = f x) :=
by
  sorry

end correct_statements_l793_793247


namespace transformed_sum_l793_793164

theorem transformed_sum (n : ℕ) (y : Fin n → ℝ) (s : ℝ) (h : s = (Finset.univ.sum (fun i => y i))) :
  Finset.univ.sum (fun i => 3 * (y i) + 30) = 3 * s + 30 * n :=
by 
  sorry

end transformed_sum_l793_793164


namespace selling_price_is_correct_l793_793510

-- Define the constants used in the problem
noncomputable def cost_price : ℝ := 540
noncomputable def markup_percentage : ℝ := 0.15
noncomputable def discount_percentage : ℝ := 26.570048309178745 / 100

-- Define the conditions in the problem
noncomputable def marked_price : ℝ := cost_price * (1 + markup_percentage)
noncomputable def discount_amount : ℝ := marked_price * discount_percentage
noncomputable def selling_price : ℝ := marked_price - discount_amount

-- Theorem stating the problem
theorem selling_price_is_correct : selling_price = 456 := by 
  sorry

end selling_price_is_correct_l793_793510


namespace find_y_value_l793_793419

-- Define the given conditions and the final question in Lean
theorem find_y_value (k : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : y = k * x ^ (1/3)) 
  (h2 : y = 4 * real.sqrt 3)
  (x1 : x = 64) 
  : ∃ k, y = 2 * real.sqrt 3 :=
sorry

end find_y_value_l793_793419


namespace translation_identity_l793_793966

def original_function (x : ℝ) : ℝ := sin (2 * x) + cos (2 * x)
def translated_function (x : ℝ) : ℝ := original_function (x + π / 4)
def expected_function (x : ℝ) : ℝ := cos (2 * x) - sin (2 * x)

theorem translation_identity : ∀ x : ℝ, translated_function x = expected_function x := by
  sorry

end translation_identity_l793_793966


namespace correct_statements_l793_793246

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - 13 / 4 * Real.pi)

theorem correct_statements :
    (f (Real.pi / 8) = 0) ∧ 
    (∀ x, 2 * Real.sin (2 * (x - 5 / 8 * Real.pi)) = f x) :=
by
  sorry

end correct_statements_l793_793246


namespace harrys_age_l793_793891

theorem harrys_age {K B J F H : ℕ} 
  (hKiarra : K = 30)
  (hKiarra_Bea : K = 2 * B)
  (hJob : J = 3 * B)
  (hFigaro : F = J + 7)
  (hHarry : H = F / 2) : 
  H = 26 := 
by 
  -- Definitions from the conditions
  have hBea : B = 15, from (by linarith : 15 = K / 2).symm,
  
  -- Continuing the proof using the provided conditions and calculating step by step
  sorry

end harrys_age_l793_793891


namespace vector_magnitude_b_l793_793848

variables (a b : EuclideanSpace ℝ 3)
variables (h1 : ∥a∥ = 3) (h2 : ∥a - b∥ = 5) (h3 : inner a b = 1)

theorem vector_magnitude_b : ∥b∥ = 3 * real.sqrt 2 :=
by
  sorry

end vector_magnitude_b_l793_793848


namespace maria_uses_666_blocks_l793_793511

theorem maria_uses_666_blocks :
  let original_volume := 15 * 12 * 7
  let interior_length := 15 - 2 * 1.5
  let interior_width := 12 - 2 * 1.5
  let interior_height := 7 - 1.5
  let interior_volume := interior_length * interior_width * interior_height
  let blocks_volume := original_volume - interior_volume
  blocks_volume = 666 :=
by
  sorry

end maria_uses_666_blocks_l793_793511


namespace minimum_cables_l793_793174

-- Definitions for the problem conditions
variable (Employee : Type) [Fintype Employee] [DecidableEq Employee]
variable (Brand : Type) [DecidableEq Brand]

-- Define the brands
def A : Brand := sorry
def B : Brand := sorry

-- Define the employees with respective brands
variable (f : Employee → Brand)

-- Predicates defining the number of employees with each type of computer
variable (numA : Fin 25 → Employee) (numB : Fin 10 → Employee)
axiom num_employees_A : ∀ i, f (numA i) = A
axiom num_employees_B : ∀ i, f (numB i) = B

-- Communication condition: if there is a cable, it must connect Brand A to Brand B
variable (Cable : Employee → Employee → Prop)
axiom cable_condition : ∀ (x y : Employee), Cable x y → f x = A ∧ f y = B ∨ f x = B ∧ f y = A

-- Connectivity requirement: everyone must be able to communicate
axiom connectivity : ∀ (x y : Employee), ∃ path: List Employee, path.head = x ∧ path.last = y ∧
  ∀ i, (i < path.length - 1) → Cable (path.nthLe i sorry) (path.nthLe (i + 1) sorry)

-- Proof statement that the minimum number of cables used to ensure full communication is 10
theorem minimum_cables : ∃ (cables : Finset (Employee × Employee)), cables.card = 10 ∧
  ∀ (x y : Employee), (∃ path: List (Employee × Employee), 
    ∀ i, (i < path.length - 1) → ((path.nthLe i sorry).snd, (path.nthLe (i + 1) sorry).fst) ∈ cables) →
    ∃ path: List Employee, path.head = x ∧ path.last = y ∧
  ∀ i, (i < path.length - 1) → Cable (path.nthLe i sorry) (path.nthLe (i + 1) sorry) := sorry

end minimum_cables_l793_793174


namespace unoccupied_volume_eq_l793_793988

-- Define the radii and heights of the cones and cylinder
variable (r_cone : ℝ) (h_cone : ℝ) (h_cylinder : ℝ)
variable (r_cone_eq : r_cone = 10) (h_cone_eq : h_cone = 15) (h_cylinder_eq : h_cylinder = 30)

-- Define the volumes of the cylinder and the two cones
def volume_cylinder (r h : ℝ) : ℝ := π * r ^ 2 * h
def volume_cone (r h : ℝ) : ℝ := 1 / 3 * π * r ^ 2 * h
def volume_unoccupied : ℝ := volume_cylinder r_cone h_cylinder - 2 * volume_cone r_cone h_cone

-- Expression of the final result
theorem unoccupied_volume_eq : volume_unoccupied r_cone h_cone h_cylinder = 2000 * π :=
by
  rw [r_cone_eq, h_cone_eq, h_cylinder_eq]
  unfold volume_cylinder volume_cone volume_unoccupied
  norm_num

end unoccupied_volume_eq_l793_793988


namespace perimeter_of_rectangle_l793_793681

theorem perimeter_of_rectangle (b l : ℝ) (h1 : l = 3 * b) (h2 : b * l = 75) : 2 * l + 2 * b = 40 := 
by 
  sorry

end perimeter_of_rectangle_l793_793681


namespace find_line_l_l793_793303

theorem find_line_l (A B M N : ℝ × ℝ)
  (h1 : ∃ x1 y1 x2 y2, A = (x1, y1) ∧ B = (x2, y2) ∧ x1 > 0 ∧ y1 > 0 ∧ x2 > 0 ∧ y2 > 0)
  (h2 : M = (x1, 0))
  (h3 : N = (0, y2))
  (h4 : abs (| M.1 - A.1 |) = abs (| N.2 - B.2 |))
  (h5 : dist M N = 2 * sqrt 3) :
  ∃ k m : ℝ, k < 0 ∧ m > 0 ∧ (∀ x y : ℝ, (y = k * x + m ↔ x + sqrt 2 * y - 2 * sqrt 2 = 0)) := sorry

end find_line_l_l793_793303


namespace complex_number_modulus_l793_793429

open Complex

theorem complex_number_modulus :
  ∀ x : ℂ, x + I = (2 - I) / I → abs x = Real.sqrt 10 := by
  sorry

end complex_number_modulus_l793_793429


namespace two_pow_n_plus_one_not_prime_l793_793401

theorem two_pow_n_plus_one_not_prime (n : ℕ) (h : ∃ m : ℕ, m > 1 ∧ m.Odd ∧ ∃ k : ℕ, n = k * m) :
  ¬ Nat.Prime (2^n + 1) := by
  sorry

end two_pow_n_plus_one_not_prime_l793_793401


namespace area_enclosed_by_curve_l793_793957

def abs_diff (a b : ℝ) : ℝ := abs (a - b)

def enclosed_area : Set (ℝ × ℝ) := {p | abs_diff p.1 1 + abs_diff p.2 1 = 1}

theorem area_enclosed_by_curve : MeasureTheory.Measure (Set) enclosed_area = 2 :=
sorry

end area_enclosed_by_curve_l793_793957


namespace average_age_across_rooms_l793_793022

theorem average_age_across_rooms
    (nA : ℕ) (avgA : ℕ)
    (nB : ℕ) (avgB : ℕ)
    (nC : ℕ) (avgC : ℕ)
    (h1 : nA = 8)
    (h2 : avgA = 35)
    (h3 : nB = 5)
    (h4 : avgB = 28)
    (h5 : nC = 3)
    (h6 : avgC = 42) :
        (nA * avgA + nB * avgB + nC * avgC) / (nA + nB + nC) = 34.125 :=
by
  sorry

end average_age_across_rooms_l793_793022


namespace most_likely_indicates_relationship_l793_793709

def indicate_relationship (a c : ℕ) : ℝ :=
  abs ((a : ℝ) / (a + 10) - (c : ℝ) / (c + 30))

theorem most_likely_indicates_relationship : 
  indicate_relationship 45 15 > indicate_relationship 40 20 ∧
  indicate_relationship 45 15 > indicate_relationship 35 25 ∧
  indicate_relationship 45 15 > indicate_relationship 30 30 :=
by
  sorry

end most_likely_indicates_relationship_l793_793709


namespace arctan_sum_l793_793196

theorem arctan_sum (a b : ℝ) (h1 : a = 3) (h2 : b = 7) : 
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 := 
by 
  rw [h1, h2]
  sorry

end arctan_sum_l793_793196


namespace number_of_four_digit_integers_with_thousands_digit_2_l793_793819

theorem number_of_four_digit_integers_with_thousands_digit_2 :
  {n : ℕ // 1000 ≤ n < 10000 ∧ (n / 1000 = 2)}.card = 1000 :=
by
  sorry

end number_of_four_digit_integers_with_thousands_digit_2_l793_793819


namespace correct_statements_about_f_l793_793248

def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - (13 / 4) * Real.pi)

theorem correct_statements_about_f :
  ((f (Real.pi / 8) = 0) ∧ (f (Real.pi / 8) = f (-(Real.pi / 8)))) ∧  -- Statement A
  (f(2 * (Real.pi / 8)) = 2 * Real.sin(2 * (Real.pi / 8) - (13 / 4) * Real.pi)) ∧  -- Statement B
  (f (2 * (Real.pi / 8) + 5/8 * Real.pi) = 2 * Real.sin (2 * (Real.pi / 8 + 5/8 * Real.pi) - 13 / 4 * Real.pi)) ∧ -- Statement C
  (f (2 * (Real.pi / 8) - 5/8 * Real.pi) = 2 * Real.sin (2 * (Real.pi / 8 - 5/8 * Real.pi) - 13 / 4 * Real.pi))  -- Statement D
:= sorry

end correct_statements_about_f_l793_793248


namespace max_sum_arithmetic_sequence_l793_793970

theorem max_sum_arithmetic_sequence : 
  let S : ℕ → ℤ := λ n, 25 * n - n * n in
  ∃ n : ℕ, S n = 156 ∧ (∀ m : ℕ, S m ≤ S n) :=
sorry

end max_sum_arithmetic_sequence_l793_793970


namespace minimum_distance_polar_coordinate_system_l793_793881

theorem minimum_distance_polar_coordinate_system :
  let M := (4, Real.pi / 3)
  let rectangular_coordinates_M := (2, 2 * Real.sqrt 3)
  let curve := λ ρ θ, ρ * Real.cos (θ - Real.pi / 3) = 2
  let rectangular_curve := λ x y, x + Real.sqrt 3 * y - 4 = 0
  ∃ d : ℝ, d = 2 ∧
    ∀ (x y : ℝ), rectangular_curve x y → distance (rectangular_coordinates_M.1, rectangular_coordinates_M.2) (x, y) ≥ d := 
by
  let d := 2
  exists d
  sorry -- The actual proof is not required as stated in the instructions

end minimum_distance_polar_coordinate_system_l793_793881


namespace max_point_f_l793_793277

-- Define the function f(x) = x^3 - 12x
def f (x : ℝ) : ℝ := x^3 - 12 * x

-- Maximum point of the function f is -2
theorem max_point_f : ∃ m, m = -2 ∧ ∀ x, f x ≤ f (-2) :=
by
  sorry

end max_point_f_l793_793277


namespace general_solution_of_differential_equation_l793_793744

theorem general_solution_of_differential_equation :
  ∀ (C₁ C₂ : ℝ), ∃ y : ℝ → ℝ,
    (∀ x, (deriv^[2] y) x - 6 * (deriv y x) + 9 * (y x) = 25 * real.exp x * real.sin x) ∧
    (y = λ x, (C₁ + C₂ * x) * real.exp (3 * x) + real.exp x * (4 * real.cos x + 3 * real.sin x)) :=
begin
  intros,
  use λ x, (C₁ + C₂ * x) * real.exp (3 * x) + real.exp x * (4 * real.cos x + 3 * real.sin x),
  split,
  { intro x,
    sorry },
  { refl }
end

end general_solution_of_differential_equation_l793_793744


namespace fraction_of_journey_by_bus_l793_793668

-- Define the total journey, the fraction of journey by rail, and the distance by foot
def total_journey : ℝ := 130
def journey_by_rail_fraction : ℝ := 3 / 5
def distance_on_foot : ℝ := 6.5

-- Calculate the distance by rail
def distance_by_rail := journey_by_rail_fraction * total_journey

-- Calculate the remaining distance by bus
def distance_by_bus := total_journey - (distance_by_rail + distance_on_foot)

-- Calculate the fraction of the journey by bus
def journey_by_bus_fraction := distance_by_bus / total_journey

-- The theorem to prove
theorem fraction_of_journey_by_bus :
  journey_by_bus_fraction = 45.5 / 130 :=
by
  sorry

end fraction_of_journey_by_bus_l793_793668


namespace gcd_digit_bound_l793_793895

theorem gcd_digit_bound (a b : ℕ) 
  (ha_len : nat.digits 10 a = 2019) 
  (hb_len : nat.digits 10 b = 2019) 
  (ha_nonzero : (nat.digits 10 a).countp (≠ 0) = 12) 
  (hb_nonzero : (nat.digits 10 b).countp (≠ 0) = 14) : 
  nat.digits 10 (nat.gcd a b) ≤ 14 := 
sorry

end gcd_digit_bound_l793_793895


namespace lucy_l793_793509

/-- A mathematical model to represent Lucy’s trip conditions and proving the correct graph representation --/

def city_traffic : ℕ → ℝ := sorry
def highway : ℕ → ℝ := sorry
def roadwork : ℕ → ℝ := sorry
def mall_shopping : ℕ → ℝ := 0

def lucy_trip_forward (t : ℕ) : ℝ :=
  if t < T₁ then city_traffic t
  else if t < T₂ then highway (t - T₁)
  else if t < T₃ then roadwork (t - T₂)
  else mall_shopping (t - T₃)

def lucy_trip_backward (t : ℕ) : ℝ :=
  if t < T₄ then roadwork (T₄ - t)
  else if t < T₅ then highway (T₄ - t - T₅)
  else city_traffic (T₆ - t)

/-- Representing Lucy's complete trip --/
def lucy_complete_trip (t : ℕ) : ℝ :=
  if t < total_forward_time then lucy_trip_forward t
  else lucy_trip_backward (t - total_forward_time)

theorem lucy's_trip_representation : lucy_complete_trip = graph_B :=
sorry

end lucy_l793_793509


namespace perimeter_of_rectangle_is_32_l793_793682

-- Given conditions
def side_length_of_square (s : ℝ) : Prop := 4 * s = 16
def rectangle_side_length (s : ℝ) : ℝ := 2 * s

-- Statement to prove
theorem perimeter_of_rectangle_is_32 (s : ℝ) (hs : side_length_of_square s) : 
  4 * rectangle_side_length s = 32 :=
by
  unfold side_length_of_square at hs
  unfold rectangle_side_length
  rw hs
  sorry

end perimeter_of_rectangle_is_32_l793_793682


namespace length_DE_is_20_l793_793473

noncomputable def length_DE (BC : ℝ) (angle_C_deg : ℝ) 
  (D : ℝ) (is_midpoint_D : D = BC / 2)
  (is_right_triangle : angle_C_deg = 45): ℝ := 
let DE := D in DE

theorem length_DE_is_20 : ∀ (BC : ℝ) (angle_C_deg : ℝ),
  BC = 40 → 
  angle_C_deg = 45 → 
  let D := BC / 2 in 
  let DE := D in 
  DE = 20 :=
by
  intros BC angle_C_deg hBC hAngle
  sorry

end length_DE_is_20_l793_793473


namespace inverse_matrix_correct_curve_C_equation_l793_793323

def matrix_A : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![2, -1],
  ![-4, 3]
]

noncomputable def inverse_matrix_A : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![3/2, 1/2],
  ![2, 1]
]

-- Given proof statement for inverse matrix of A
theorem inverse_matrix_correct : matrix_A.mul inverse_matrix_A = 1 :=
begin
  sorry
end

-- Definition of the transformation equations
def x' (x y : ℚ) := 2 * x - y
def y' (x y : ℚ) := -4 * x + 3 * y

-- Given statement for the equation of curve C
theorem curve_C_equation (x y : ℚ) : (x' x y) * (y' x y) = 1 → 8 * x ^ 2 - 10 * x * y + 3 * y ^ 2 + 1 = 0 :=
begin
  sorry
end

end inverse_matrix_correct_curve_C_equation_l793_793323


namespace length_DE_is_20_l793_793456

open_locale real
open_locale complex_conjugate

noncomputable def in_triangle_config_with_angle (ABC : Type) [triangle ABC]
  (BC_length : ℝ) (angle_C : real.angle) (perp_bisector_intersect_D : point)
  (perp_bisector_intersect_E : point) : Prop :=
  -- define the conditions
  BC_length = 40 ∧
  angle_C = 45 ∧
  is_midpoint_of_BC ABC perp_bisector_intersect_D ∧
  is_perpendicular_bisector_intersects_AC ABC perp_bisector_intersect_D perp_bisector_intersect_E

noncomputable def length_DE (D E: point) : ℝ :=
  distance_between D E

theorem length_DE_is_20 {ABC : Type} [triangle ABC]
  (BC_length : ℝ) (angle_C : real.angle) (perp_bisector_intersect_D : point)
  (perp_bisector_intersect_E : point)
  (h : in_triangle_config_with_angle ABC BC_length angle_C perp_bisector_intersect_D perp_bisector_intersect_E):
  length_DE perp_bisector_intersect_D perp_bisector_intersect_E = 20 :=
begin
  sorry,
end

end length_DE_is_20_l793_793456


namespace benny_picked_proof_l793_793183

-- Define the number of apples Dan picked
def dan_picked: ℕ := 9

-- Define the total number of apples picked
def total_apples: ℕ := 11

-- Define the number of apples Benny picked
def benny_picked (dan_picked total_apples: ℕ): ℕ :=
  total_apples - dan_picked

-- The theorem we need to prove
theorem benny_picked_proof: benny_picked dan_picked total_apples = 2 :=
by
  -- We calculate the number of apples Benny picked
  sorry

end benny_picked_proof_l793_793183


namespace crease_length_l793_793040

theorem crease_length (AB : ℝ) (h₁ : AB = 15)
  (h₂ : ∀ (area : ℝ) (folded_area : ℝ), folded_area = 0.25 * area) :
  ∃ (DE : ℝ), DE = 0.5 * AB :=
by
  use 7.5 -- DE
  sorry

end crease_length_l793_793040


namespace sum_of_fraction_numerator_denominator_l793_793098

theorem sum_of_fraction_numerator_denominator :
  let x := 0.343434...
  in let fraction := (34 / 99 : ℚ)
  in let sum := fraction.num + fraction.den 
  in (x : ℚ) = fraction ∧ fraction.isReduced → sum = 133 :=
by
  sorry

end sum_of_fraction_numerator_denominator_l793_793098


namespace probability_of_perfect_square_sum_l793_793613

def is_perfect_square (n : ℕ) : Prop :=
  n = 1*1 ∨ n = 2*2 ∨ n = 3*3 ∨ n = 4*4

theorem probability_of_perfect_square_sum :
  let total_outcomes := 64
  let perfect_square_sums := 12
  (perfect_square_sums / total_outcomes : ℚ) = 3 / 16 :=
by
  sorry

end probability_of_perfect_square_sum_l793_793613


namespace vector_magnitude_l793_793842

variable {V : Type*} [InnerProductSpace ℝ V]

theorem vector_magnitude
  (a b : V)
  (h1 : ‖a‖ = 3)
  (h2 : ‖a - b‖ = 5)
  (h3 : inner a b = 1) :
  ‖b‖ = 3 * Real.sqrt 2 :=
by
  sorry

end vector_magnitude_l793_793842


namespace probability_dice_sum_perfect_square_l793_793600

def is_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def probability_perfect_square : ℚ :=
  12 / 64

theorem probability_dice_sum_perfect_square :
  -- Two standard 8-sided dice are rolled
  -- The probability that the sum rolled is a perfect square is 3/16
  (∃ dice1 dice2 : ℕ, 1 ≤ dice1 ∧ dice1 ≤ 8 ∧ 1 ≤ dice2 ∧ dice2 ≤ 8) →
  probability_perfect_square = 3 / 16 :=
sorry

end probability_dice_sum_perfect_square_l793_793600


namespace marbles_of_different_color_l793_793441

noncomputable def red := 20
noncomputable def green := 3 * red
noncomputable def yellow := 0.20 * green
noncomputable def total_marbles := green + 3 * green
noncomputable def different_color_marbles := total_marbles - red - green - yellow

theorem marbles_of_different_color :
  different_color_marbles = 148 := by
  sorry

end marbles_of_different_color_l793_793441


namespace number_2digit_smaller_than_35_l793_793346

/--
Prove that the number of two-digit numbers where at least one digit
is smaller than the corresponding digit in 35 is exactly 55.
-/
theorem number_2digit_smaller_than_35 : 
  (Finset.filter (λ n : ℕ, let d1 := n / 10, d2 := n % 10 in d1 < 3 ∨ d2 < 5) (Finset.range' 10 100)).card = 55 := 
by
  sorry

end number_2digit_smaller_than_35_l793_793346


namespace num_two_digit_numbers_with_digit_less_than_35_l793_793380

theorem num_two_digit_numbers_with_digit_less_than_35 : 
  let total_two_digit := 90 in
  let unsuitable_numbers := 35 in
  let suitable_numbers := total_two_digit - unsuitable_numbers
  in suitable_numbers = 55 :=
by 
  let total_two_digit := 90
  let unsuitable_numbers := 35
  let suitable_numbers := total_two_digit - unsuitable_numbers
  show suitable_numbers = 55, from sorry

end num_two_digit_numbers_with_digit_less_than_35_l793_793380


namespace two_digit_numbers_count_l793_793344

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem two_digit_numbers_count :
  { n : ℕ | 10 ≤ n ∧ n ≤ 99 ∧ (tens_digit n < 3 ∨ units_digit n < 5) }.to_finset.card = 55 :=
by
  sorry

end two_digit_numbers_count_l793_793344


namespace log_interval_l793_793066

open Real

theorem log_interval (x : ℝ) : 
  x = (1 / (log 3 / log (1 / 2))) + (1 / (log 3 / log (1 / 5))) → 
  2 < x ∧ x < 3 :=
by
  intro h
  have h1 := (one_div (log 3 / log (1 / 2))).symm
  have h2 := (one_div (log 3 / log (1 / 5))).symm
  rw [log_inv, log_inv, neg_div_neg_eq, log_inv, neg_div_neg_eq] at h1 h2
  rw [h1, h2] at h
  have h3 : log 10 = 1 := by exact log_base_change log10_eq
  rw [div_self log3_pos, div_self log3_pos] at h
  refine ⟨_, _⟩
  { rw h, exact log10_lt3.inv_pos' },
  { rw h, exact log_base_change log10_eq }

end log_interval_l793_793066


namespace range_of_m_l793_793783

theorem range_of_m (ω : ℝ) (m : ℝ) (g : ℝ → ℝ) :
  (∀ x, g x = 4 * sin (2 * x - π / 6)) ∧ (g x = 4 * sin ω x + π / 6) ∧ (ω > 0) ∧ 
  (g has exactly three distinct zeros in the interval (-m, m)) →
  (frac 7 π 12 < m ∧ m ≤ frac 11 π 12) :=
begin
  sorry,
end

end range_of_m_l793_793783


namespace octagon_area_is_correct_l793_793659

-- Define the concept of a convex octagon inscribed in a circle with specified side lengths
structure convex_octagon (α : Type) :=
  (sides: fin 8 → ℝ)
  (inscribed_in_circle : Prop)
  (convex : Prop)

-- Define a specific convex octagon with given side lengths
def specific_convex_octagon (o : convex_octagon ℝ) : Prop :=
  (o.sides 0 = 3) ∧ (o.sides 1 = 3) ∧ (o.sides 2 = 3) ∧ (o.sides 3 = 3) ∧
  (o.sides 4 = 2) ∧ (o.sides 5 = 2) ∧ (o.sides 6 = 2) ∧ (o.sides 7 = 2) ∧
  o.inscribed_in_circle ∧ o.convex

noncomputable def octagon_area (α : Type) [real_scalar α] (o : convex_octagon α) : α :=
  13 + 12 * real.sqrt 2

-- State that the area of the specific convex octagon is 13 + 12sqrt(2)
theorem octagon_area_is_correct (o : convex_octagon ℝ) (h : specific_convex_octagon o) :
  octagon_area ℝ o = 13 + 12 * (Real.sqrt 2) :=
by
  sorry

end octagon_area_is_correct_l793_793659


namespace magnitude_b_l793_793839

noncomputable theory

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Conditions
def mag_a : real := ‖a‖ = 3
def mag_diff_ab : real := ‖a - b‖ = 5
def dot_ab : real := inner a b = 1

-- Goal
theorem magnitude_b (h1 : ‖a‖ = 3) (h2 : ‖a - b‖ = 5) (h3 : inner a b = 1) : ‖b‖ = 3 * real.sqrt 2 :=
by 
  sorry

end magnitude_b_l793_793839


namespace parallel_lines_are_equal_m_l793_793828

def l1 (m : ℝ) : ℝ → ℝ → Prop := λ x y, (2 * m + 1) * x - 4 * y + 3 * m = 0
def l2 (m : ℝ) : ℝ → ℝ → Prop := λ x y, x + (m + 5) * y - 3 * m = 0

theorem parallel_lines_are_equal_m (m : ℝ) : 
  (∀ x y, l1 m x y → ∃ x' y', l2 m x' y') → m = - 9 / 2 :=
by
  sorry

end parallel_lines_are_equal_m_l793_793828


namespace angle_ADB_is_90_l793_793703

open Real

noncomputable def is_isosceles {A B C : Point} (h : Triangle A B C) : Prop :=
  dist A B = dist A C

noncomputable def is_inscribed {A B C : Point} (h : Triangle A B C) (circle : Circle) : Prop :=
  circle.center = C ∧ dist C A = circle.radius

noncomputable def extend_through {A B D : Point} : Prop :=
  collinear A B D ∧ between B A D

noncomputable def is_right_angle {A B D : Point} : Prop :=
  angle A D B = π / 2

theorem angle_ADB_is_90 {A B C D : Point} {circle : Circle} (h1 : Triangle A B C)
    (hiso : is_isosceles h1) (hins : is_inscribed h1 circle)
    (hradius : circle.radius = 12) (hextend : extend_through A B D) : is_right_angle A D B :=
sorry

end angle_ADB_is_90_l793_793703


namespace total_sold_l793_793853

theorem total_sold (D C : ℝ) (h1 : D = 1.6 * C) (h2 : D = 168) : D + C = 273 :=
by
  sorry

end total_sold_l793_793853


namespace no_quaint_two_digit_integers_l793_793693

theorem no_quaint_two_digit_integers :
  ∀ x : ℕ, 10 ≤ x ∧ x < 100 ∧ (∃ a b : ℕ, x = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) →  ¬(10 * x.div 10 + x % 10 = (x.div 10) + (x % 10)^3) :=
by
  sorry

end no_quaint_two_digit_integers_l793_793693


namespace possible_measures_of_angle_X_l793_793561

theorem possible_measures_of_angle_X : 
  ∃ n : ℕ, n = 17 ∧ (∀ (X Y : ℕ), 
    X > 0 ∧ Y > 0 ∧ X + Y = 180 ∧ 
    ∃ m : ℕ, m ≥ 1 ∧ X = m * Y) :=
sorry

end possible_measures_of_angle_X_l793_793561


namespace original_price_eq_600_l793_793700

theorem original_price_eq_600 (P : ℝ) (h1 : 300 = P * 0.5) : 
  P = 600 :=
sorry

end original_price_eq_600_l793_793700


namespace gcd_of_polynomial_l793_793778

theorem gcd_of_polynomial (a : ℤ) (h : 720 ∣ a) : Int.gcd (a^2 + 8*a + 18) (a + 6) = 6 := 
by 
  sorry

end gcd_of_polynomial_l793_793778


namespace astronaut_days_on_orbius_l793_793525

noncomputable def days_in_year : ℕ := 250
noncomputable def seasons_in_year : ℕ := 5
noncomputable def seasons_stayed : ℕ := 3

theorem astronaut_days_on_orbius :
  (days_in_year / seasons_in_year) * seasons_stayed = 150 := by
  sorry

end astronaut_days_on_orbius_l793_793525


namespace radius_of_sphere_l793_793665

theorem radius_of_sphere (shadow_sphere shadow_stick height_stick : ℝ) (h1 : shadow_sphere = 16) (h2 : shadow_stick = 4) (h3 : height_stick = 2) : 
  ∃ r : ℝ, r = 8 :=
by
  let θ := Real.atan (height_stick / shadow_stick)
  have tan_sphere := r / shadow_sphere
  have tan_stick := height_stick / shadow_stick
  have h4 : tan_sphere = tan_stick := sorry
  use 16 / 2
  rfl
  sorry

end radius_of_sphere_l793_793665


namespace fg_eq_neg7_l793_793393

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^2 + 2

theorem fg_eq_neg7 : f (g 2) = -7 :=
  by
    sorry

end fg_eq_neg7_l793_793393


namespace zero_in_interval_l793_793981

noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

theorem zero_in_interval : ∃ x ∈ Ioo 2 3, f x = 0 :=
sorry

end zero_in_interval_l793_793981


namespace length_PJ_eq_2_sqrt_37_l793_793883

-- Definition of triangle PQR with given sides PQ, PR, and QR
variable (P Q R J X Y Z : Type)
variable [Real.metric_space P] [Real.metric_space Q] [Real.metric_space R] 
variable [Real.metric_space J] [Real.metric_space X] [Real.metric_space Y] [Real.metric_space Z]

-- Given conditions
variable (PQ PR QR : ℝ) 
variable (PQ_eq_15 : PQ = 15)
variable (PR_eq_17 : PR = 17)
variable (QR_eq_16 : QR = 16)

-- J is the incenter of triangle PQR and the incircle touches sides at X, Y, and Z
variable (is_incenter : incenter J PQR)
variable (incenter_touch_X : incircle_touch J QR X)
variable (incenter_touch_Y : incircle_touch J PR Y)
variable (incenter_touch_Z : incircle_touch J PQ Z)

-- The goal is to prove that the length of PJ equals 2√37
theorem length_PJ_eq_2_sqrt_37 : 
  ∃ (PJ : ℝ), PJ = 2 * Real.sqrt 37 :=
by
  sorry

end length_PJ_eq_2_sqrt_37_l793_793883


namespace xy_value_l793_793897

theorem xy_value : 
  ∀ (x y : ℝ),
  (∀ (A B C : ℝ × ℝ), A = (1, 8) ∧ B = (x, y) ∧ C = (6, 3) → 
  (C.1 = (A.1 + B.1) / 2) ∧ (C.2 = (A.2 + B.2) / 2)) → 
  x * y = -22 :=
sorry

end xy_value_l793_793897


namespace area_ratio_of_hexagon_dodecagon_l793_793767

variable (n m : ℝ)
variable (A : Type) [LinearOrderedField A] 

def is_ratio_of_areas (m n : ℝ) (r : ℝ) := 
  r = (m / n)

theorem area_ratio_of_hexagon_dodecagon (h : n ≠ 0) (h1 : is_ratio_of_areas m n (sqrt 3 - (3 / 2))) :
  m / n = sqrt 3 - (3 / 2) :=
  by
    rw is_ratio_of_areas at h1
    exact h1

end area_ratio_of_hexagon_dodecagon_l793_793767


namespace greatest_prime_factor_299_l793_793999

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_factors (n : ℕ) : List ℕ := 
  List.filter is_prime (List.range (n + 1)).tail.filter (λ m, m ∣ n)

theorem greatest_prime_factor_299 : List.maximum (prime_factors 299) = some 19 := 
  by
  sorry

end greatest_prime_factor_299_l793_793999


namespace value_of_y_at_x8_l793_793412

theorem value_of_y_at_x8
  (k : ℝ)
  (y : ℝ → ℝ)
  (hx64 : y 64 = 4 * Real.sqrt 3)
  (hy_def : ∀ x, y x = k * x^(1 / 3)) :
  y 8 = 2 * Real.sqrt 3 :=
by {
  sorry,
}

end value_of_y_at_x8_l793_793412


namespace y_at_x_equals_8_l793_793407

theorem y_at_x_equals_8 (k : ℝ) (h1 : ∀ x y, y = k * x^(1/3))
    (h2 : 4 * real.sqrt 3 = k * 64^(1/3)) : k * 8^(1/3) = 2 * real.sqrt 3 :=
by
  sorry

end y_at_x_equals_8_l793_793407


namespace general_term_formula_a_general_term_formula_b_l793_793762

noncomputable def S (n : ℕ) (a₁ d : ℝ) : ℝ :=
  n * a₁ + d * (n * (n - 1)) / 2

noncomputable def a (n : ℕ) (a₁ d : ℝ) : ℝ :=
  a₁ + (n - 1) * d

noncomputable def T (n : ℕ) (a₁ d : ℝ) : ℝ :=
  (a n a₁ d) ^ 2

noncomputable def b (n : ℕ) (a₁ d : ℝ) : ℝ :=
  if n = 1 then 49
  else 18 * n - 69

theorem general_term_formula_a
  (a₁ d : ℝ) 
  (h : S 5 a₁ d * S 6 a₁ d + 15 = 0)
  (h₀ : S 5 a₁ d ≠ 5) :
  ∀ n, a n a₁ d = 10 - 3 * n :=
sorry

theorem general_term_formula_b
  (a₁ d : ℝ) 
  (h : S 5 a₁ d * S 6 a₁ d + 15 = 0)
  (h₀ : S 5 a₁ d ≠ 5) :
  ∀ n, T n a₁ d = ∑ i in finset.range n, b i.succ a₁ d :=
sorry

end general_term_formula_a_general_term_formula_b_l793_793762


namespace calendar_three_consecutive_sum_l793_793444

theorem calendar_three_consecutive_sum {x : ℤ} (h1 : 8 ≤ x) (h2 : x ≤ 24) :
  ∃ s, s ∈ {18, 33, 38, 75} ∧ s = 3 * x :=
by
  use 3 * x
  split
  -- Proof of the sum being one of the given choices is omitted to focus only on the Lean statement
  sorry -- Placeholder for actual proof

end calendar_three_consecutive_sum_l793_793444


namespace carousel_seat_count_l793_793530

theorem carousel_seat_count
  (total_seats : ℕ)
  (colors : ℕ → Prop)
  (num_yellow num_blue num_red : ℕ)
  (num_colors : ∀ n, colors n → n = num_yellow ∨ n = num_blue ∨ n = num_red)
  (opposite_blue_red_7_3 : ∀ n, n = 7 ↔ n + 50 = 3)
  (opposite_yellow_red_7_23 : ∀ n, n = 7 ↔ n + 50 = 23)
  (total := 100)
 :
 (num_yellow = 34 ∧ num_blue = 20 ∧ num_red = 46) :=
by
  sorry

end carousel_seat_count_l793_793530


namespace cyclic_BDFG_l793_793867

open EuclideanGeometry

-- Defining the acute triangle ABC and other points and properties as per the problem statement.
theorem cyclic_BDFG
  (A B C P Q D E F G : Point)
  (hABCacute : ∀ (X : Point), is_acute_triangle A B C)
  (hBAC_gt_BCA : ∠BAC > ∠BCA)
  (hP_on_BC : Online P B C)
  (hAngle_PAB_eq_BCA : ∠PAB = ∠BCA)
  (hQ_circumcircle_APB : IsOnCircumcircle Q A P B)
  (hQ_on_AC : Online Q A C)
  (hD_on_AP : Online D A P)
  (hAngle_QDC_eq_CAP : ∠QDC = ∠CAP)
  (hE_on_BD : Online E B D)
  (hCE_eq_CD : CE = CD)
  (hCircumcircle_CQE : IsOnCircumcircle F C Q E)
  (hQF_on_BC : Online G Q F)
  (hQF_meets_BC : Meets Q F B C) :
  Concyclic B D F G :=
by
  sorry

end cyclic_BDFG_l793_793867


namespace probability_of_perfect_square_sum_l793_793596

-- Define the conditions
def dice_faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def perfect_squares_in_range : Finset ℕ := {4, 9, 16}
def total_outcomes : ℕ := 8 * 8  -- total number of outcomes when rolling two dice

-- Calculate the number of favorable outcomes
def favorable_outcomes : ℕ :=
  (dice_faces.product dice_faces).filter (λ pair => perfect_squares_in_range.member (pair.1 + pair.2)).card

-- Calculate the probability of getting a perfect square sum
noncomputable def probability_perfect_square_sum : ℚ :=
  favorable_outcomes / total_outcomes

-- The statement to prove that the probability is 3/16
theorem probability_of_perfect_square_sum :
  probability_perfect_square_sum = 3 / 16 :=
begin
  sorry
end

end probability_of_perfect_square_sum_l793_793596


namespace medicine_types_count_l793_793438

theorem medicine_types_count (n : ℕ) (hn : n = 5) : (Nat.choose n 2 = 10) :=
by
  sorry

end medicine_types_count_l793_793438


namespace precipitation_probability_correct_l793_793442

/-- 
 In a city's weather forecast, the "precipitation probability forecast" 
 means the probability of precipitation in the area for a given time. 
 Given the forecast "Tomorrow's precipitation probability is 90%," 
 we need to show that the correct interpretation is the possibility 
 of precipitation in the area tomorrow is 90%.
-/
theorem precipitation_probability_correct :
  (Tomorrow's precipitation probability is 90% means 
   that the possibility of precipitation in the area tomorrow is 90%) :=
sorry

end precipitation_probability_correct_l793_793442


namespace find_decreased_value_l793_793680

theorem find_decreased_value (x v : ℝ) (hx : x = 7)
  (h : x - v = 21 * (1 / x)) : v = 4 :=
by
  sorry

end find_decreased_value_l793_793680


namespace RX_eq_RY_l793_793929

-- Defining the required geometric entities and relations.
variables {A B C P Q R X Y : Type}
-- Define sides of the triangle and points on sides
variables (triangle_ABC : Triangle A B C)
variables (P_on_AB : Point_on_side P A B)
variables (Q_on_BC : Point_on_side Q B C)
variables (R_on_CA : Point_on_side R C A)
variables (AP_eq_CQ : Distance_eq AP CQ)
variables (RPBQ_cyclic : Cyclic_quadrilateral R P B Q)
variables (tangent_A : Tangent_at_point triangle_ABC.circumcircle A)
variables (tangent_C : Tangent_at_point triangle_ABC.circumcircle C)
variables (X_on_RP : Line_intersection tangent_A RP X)
variables (Y_on_RQ : Line_intersection tangent_C RQ Y)

-- The theorem to prove: RX equals RY
theorem RX_eq_RY : Distance R X = Distance R Y :=
by
  sorry

end RX_eq_RY_l793_793929


namespace max_visible_unit_cubes_l793_793147

def cube_size := 11
def total_unit_cubes := cube_size ^ 3

def visible_unit_cubes (n : ℕ) : ℕ :=
  (n * n) + (n * (n - 1)) + ((n - 1) * (n - 1))

theorem max_visible_unit_cubes : 
  visible_unit_cubes cube_size = 331 := by
  sorry

end max_visible_unit_cubes_l793_793147


namespace statement_B_statement_C_l793_793905

variable (z1 z2 : ℂ)

-- Statement B: z1 - conj(z1) is purely imaginary or zero
theorem statement_B : ∃ (a : ℝ), z1 - conj(z1) = a * I ∨ z1 - conj(z1) = 0 := sorry

-- Statement C: |z1 + z2| ≤ |z1| + |z2| holds for any complex numbers z1 and z2
theorem statement_C : abs (z1 + z2) ≤ abs z1 + abs z2 := sorry

end statement_B_statement_C_l793_793905


namespace volume_of_cube_l793_793752

/-
Problem: Given a cube, the distance from its diagonal to a non-intersecting edge is d. Prove that the volume of the cube is 2 * d^3 * sqrt(2).
-/

theorem volume_of_cube (d : ℝ) : 
  ∃ (a : ℝ), (a = d * sqrt 2) ∧ (volume : ℝ → ℝ) (a) = 2 * d^3 * (sqrt 2) := 
sorry

end volume_of_cube_l793_793752


namespace gcd_1855_1120_l793_793995

theorem gcd_1855_1120 : Int.gcd 1855 1120 = 35 :=
by
  sorry

end gcd_1855_1120_l793_793995


namespace train_passes_through_tunnel_l793_793690

theorem train_passes_through_tunnel :
  ∀ (length_train : ℕ) (speed_train_kmph : ℕ) (length_tunnel_km : ℕ),
  length_train = 100 →
  speed_train_kmph = 72 →
  length_tunnel_km = 3.5 →
  let speed_train_mpm := speed_train_kmph * 1000 / 60 in
  let length_tunnel_m := length_tunnel_km * 1000 in
  let total_distance := length_train + length_tunnel_m in
  total_distance / speed_train_mpm = 3 :=
by
  intros length_train speed_train_kmph length_tunnel_km h_length_train h_speed_train_kmph h_length_tunnel_km
  let speed_train_mpm := speed_train_kmph * 1000 / 60
  let length_tunnel_m := length_tunnel_km * 1000
  let total_distance := length_train + length_tunnel_m
  have h_speed_train : speed_train_mpm = 1200, sorry,
  have h_length_tunnel : length_tunnel_m = 3500, sorry,
  have h_total_distance : total_distance = 3600, sorry,
  have h_time : total_distance / speed_train_mpm = 3, by sorry,
  exact h_time

end train_passes_through_tunnel_l793_793690


namespace limit_ln_frac_l793_793643

theorem limit_ln_frac:
  ∀ f : ℝ → ℝ,
  (∀ x, f x = (3^(2*x) - 5^(3*x)) / (arctan x + x^3)) →
  (filter.tendsto f (nhds 0) (nhds (Real.log (9/125)))) := by
  intros f hf
  sorry

end limit_ln_frac_l793_793643


namespace two_digit_numbers_less_than_35_l793_793358

theorem two_digit_numbers_less_than_35 : 
  let count : ℕ := (99 - 10 + 1) - (7 * 5)
  count = 55 :=
by
  -- definition of total number of two-digit numbers
  let total_two_digit_numbers : ℕ := 99 - 10 + 1
  -- definition of the number of unsuitable two-digit numbers
  let unsuitable_numbers : ℕ := 7 * 5
  -- definition of the count of suitable two-digit numbers
  let count : ℕ := total_two_digit_numbers - unsuitable_numbers
  -- verifying the final count
  exact rfl

end two_digit_numbers_less_than_35_l793_793358


namespace find_angle_C_max_sin_A_plus_sin_B_l793_793334

variable {a b c A B C : ℝ}

-- Define the vectors and the dot product condition
def vec_m : ℝ × ℝ := (a + c, b)
def vec_n : ℝ × ℝ := (a - c, b - a)
def dot_product_condition : Prop := vec_m.1 * vec_n.1 + vec_m.2 * vec_n.2 = 0

-- Define the condition of vectors based on sides of the triangle
variable (hSides : a = sin A ∧ b = sin B ∧ c = sin C)

-- Main assertions to be proved
theorem find_angle_C (h : dot_product_condition ∧ 0 < C ∧ C < π) :
  C = π / 3 := by
    sorry

theorem max_sin_A_plus_sin_B (hC : C = π / 3) :
  sin A + sin B ≤ √3 := by
    sorry

end find_angle_C_max_sin_A_plus_sin_B_l793_793334


namespace find_a_l793_793965

theorem find_a (a : ℝ) :
  (∀ x : ℝ, x^2 - a = 0 ↔ 3 * x^4 - 48 = 0) → a = 4 :=
  by
    intros h
    sorry

end find_a_l793_793965


namespace union_sets_l793_793269

namespace Proof

def setA : Set ℝ := { x : ℝ | x * (x + 1) ≤ 0 }
def setB : Set ℝ := { x : ℝ | -1 < x ∧ x < 1 }

theorem union_sets : setA ∪ setB = { x : ℝ | -1 ≤ x ∧ x < 1 } :=
sorry

end Proof

end union_sets_l793_793269


namespace order_of_f0_f1_f_2_l793_793392

noncomputable def f (m x : ℝ) := (m-1) * x^2 + 6 * m * x + 2

theorem order_of_f0_f1_f_2 (m : ℝ) (h_even : ∀ x : ℝ, f m x = f m (-x)) :
  m = 0 → f m (-2) < f m 1 ∧ f m 1 < f m 0 :=
by 
  sorry

end order_of_f0_f1_f_2_l793_793392


namespace complex_magnitude_l793_793258

theorem complex_magnitude (z : ℂ) (h : z = (1 + Real.sqrt 3 * complex.i) * (3 - complex.i) ^ 2 / (3 - 4 * complex.i)) :
  z * complex.conj z = 16 := by
  sorry

end complex_magnitude_l793_793258


namespace sum_of_squares_inequality_l793_793018

theorem sum_of_squares_inequality {n : ℕ} (x y : Fin n → ℝ) (hx : ∀ i, 0 < x i) (hy : ∀ i, 0 < y i) :
  (∑ i, (x i + y i)^2) / (∑ i, (x i + y i)) ≤ 
  (∑ i, (x i)^2) / (∑ i, (x i)) + (∑ i, (y i)^2) / (∑ i, (y i)) :=
sorry

end sum_of_squares_inequality_l793_793018


namespace cevian_ratios_l793_793053

theorem cevian_ratios {A B C D E F S : Type} [LinearOrder S] 
    (h_AD: A < D) (h_BE: B < E) (h_CF: C < F) 
    (h1: (AS / DS) = 3 / 2) (h2: (BS / ES) = 4 / 3) : 
    (CS / FS) = 2 / 1 :=
sorry

end cevian_ratios_l793_793053


namespace line_parallel_to_skew_lines_condition_l793_793158

theorem line_parallel_to_skew_lines_condition {l1 l2 l3 : Type} [line l1] [line l2] [line l3] 
  (skew_l1_l2 : skew l1 l2) (parallel_l3_l1 : parallel l3 l1) : 
  intersect l3 l2 ∨ skew l3 l2 :=
sorry

end line_parallel_to_skew_lines_condition_l793_793158


namespace transformed_curve_l793_793238

theorem transformed_curve (x y : ℝ) :
  (∃ (x1 y1 : ℝ), x1 = 3*x ∧ y1 = 2*y ∧ (x1^2 / 9 + y1^2 / 4 = 1)) →
  x^2 + y^2 = 1 :=
by
  sorry

end transformed_curve_l793_793238


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l793_793121

theorem sum_of_numerator_and_denominator_of_repeating_decimal (x : ℚ) (h : x = 34 / 99) : (x.den + x.num : ℤ) = 133 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l793_793121


namespace john_books_sold_on_wednesday_l793_793889

theorem john_books_sold_on_wednesday
  (total_books : ℕ)
  (sold_monday : ℕ)
  (sold_tuesday : ℕ)
  (sold_thursday : ℕ)
  (sold_friday : ℕ)
  (percentage_not_sold : ℝ)
  (total_sold : ℕ) : 
  total_books = 1300 →
  sold_monday = 75 →
  sold_tuesday = 50 →
  sold_thursday = 78 →
  sold_friday = 135 →
  percentage_not_sold = 69.07692307692308 →
  total_sold = 402 →
  ∃ (W : ℕ), W + sold_monday + sold_tuesday + sold_thursday + sold_friday = total_sold ∧ W = 64 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  use 64
  split
  sorry -- The proof steps will go here, which are omitted according to instructions.
  sorry -- The proof steps will go here, which are omitted according to instructions.

end john_books_sold_on_wednesday_l793_793889


namespace brother_ate_2_more_cookies_than_mother_l793_793516

theorem brother_ate_2_more_cookies_than_mother :
  ∀ (total_cookies father_cookies remaining_cookies : ℕ) (mother_cookies brother_cookies : ℕ),
  total_cookies = 30 →
  father_cookies = 10 →
  mother_cookies = father_cookies / 2 →
  remaining_cookies = 8 →
  brother_cookies = total_cookies - remaining_cookies - father_cookies - mother_cookies →
  (brother_cookies - mother_cookies) = 2 :=
by
  intros total_cookies father_cookies remaining_cookies mother_cookies brother_cookies
  assume h_total h_father h_mother h_remaining h_brother
  sorry

end brother_ate_2_more_cookies_than_mother_l793_793516


namespace divisible_by_six_l793_793019

theorem divisible_by_six (m : ℕ) : 6 ∣ (m^3 + 11 * m) := 
sorry

end divisible_by_six_l793_793019


namespace minimum_cut_length_triangle_l793_793692

theorem minimum_cut_length_triangle (x : ℕ) (h1 : 0 < 9 - x) (h2 : 0 < 16 - x) (h3 : 0 < 18 - x)
  (h4 : (9 - x) + (16 - x) < (18 - x)) : x = 8 :=
begin
  sorry,
end

end minimum_cut_length_triangle_l793_793692


namespace tickets_sold_at_door_l793_793036

theorem tickets_sold_at_door :
  ∃ D : ℕ, ∃ A : ℕ, A + D = 800 ∧ (1450 * A + 2200 * D = 166400) ∧ D = 672 :=
by
  sorry

end tickets_sold_at_door_l793_793036


namespace sum_of_fraction_numerator_denominator_l793_793099

theorem sum_of_fraction_numerator_denominator :
  let x := 0.343434...
  in let fraction := (34 / 99 : ℚ)
  in let sum := fraction.num + fraction.den 
  in (x : ℚ) = fraction ∧ fraction.isReduced → sum = 133 :=
by
  sorry

end sum_of_fraction_numerator_denominator_l793_793099


namespace find_value_ab_l793_793568

noncomputable def equilateral_ab (a b : ℝ) : Prop :=
  let z_a := complex.mk a 15
  let z_b := complex.mk b 45
  let rot := complex.exp (complex.I * (2 * real.pi / 3))
  ∃ a b : ℝ, 
    (z_a * rot = z_b ∨ z_b * rot = z_a) ∧ ab = -1050

theorem find_value_ab :
  ∃ a b : ℝ, equilateral_ab a b :=
sorry

end find_value_ab_l793_793568


namespace men_in_group_initial_l793_793959

variable (M : ℕ)  -- Initial number of men in the group
variable (A : ℕ)  -- Initial average age of the group

theorem men_in_group_initial : (2 * 50 - (18 + 22) = 60) → ((M + 6) = 60 / 6) → (M = 10) :=
by
  sorry

end men_in_group_initial_l793_793959


namespace min_distance_eq_240_over_17_l793_793748

theorem min_distance_eq_240_over_17 : 
  ∀ x y : ℝ, 8 * x + 15 * y = 240 → sqrt (x^2 + y^2) = 240 / 17 := 
by
  intros x y h
  sorry

end min_distance_eq_240_over_17_l793_793748


namespace girls_in_blue_dresses_answered_affirmatively_l793_793072

theorem girls_in_blue_dresses_answered_affirmatively :
  ∃ (n : ℕ), n = 17 ∧
  ∀ (total_girls red_dresses blue_dresses answer_girls : ℕ),
  total_girls = 30 →
  red_dresses = 13 →
  blue_dresses = 17 →
  answer_girls = n →
  answer_girls = blue_dresses :=
sorry

end girls_in_blue_dresses_answered_affirmatively_l793_793072


namespace probability_abc_144_l793_793128

-- Define the set of possible outcomes for a standard die
def die_faces := {1, 2, 3, 4, 5, 6}

-- Define the event that the product of three dice is 144
def event (a b c : ℕ) := a * b * c = 144

-- Calculate the probability of the event
def probability_event := 
  (1 / 6) * (1 / 6) * (1 / 6)

theorem probability_abc_144 : 
  ∑ (a ∈ die_faces) ∑ (b ∈ die_faces) ∑ (c ∈ die_faces), if event a b c then probability_event else 0 = 1 / 72 :=
by
  sorry

end probability_abc_144_l793_793128


namespace at_least_4_digits_in_row_or_column_l793_793146

theorem at_least_4_digits_in_row_or_column 
  (grid : Fin 10 → Fin 10 → Fin 10)
  (digit_count : ∀ (d : Fin 10), (∑ (i : Fin 10) (j : Fin 10), if grid i j = d then 1 else 0) = 10) : 
  ∃ i : Fin 10, (∃ j : Fin 10, 4 ≤ (Finset.univ.filter (λ d, ∃ k : Fin 10, grid i k = d)).card) ∨ 
                 (∃ j : Fin 10, 4 ≤ (Finset.univ.filter (λ d, ∃ k : Fin 10, grid k j = d)).card) :=
sorry

end at_least_4_digits_in_row_or_column_l793_793146


namespace number_of_cows_on_farm_l793_793520

theorem number_of_cows_on_farm :
  (∀ (cows_per_week : ℤ) (six_cows_milk : ℤ) (total_milk : ℤ) (weeks : ℤ),
    cows_per_week = 6 → 
    six_cows_milk = 108 →
    total_milk = 2160 →
    weeks = 5 →
    (total_milk / (six_cows_milk / cows_per_week * weeks)) = 24) :=
by
  intros cows_per_week six_cows_milk total_milk weeks h1 h2 h3 h4
  have h_cow_milk_per_week : six_cows_milk / cows_per_week = 18 := by sorry
  have h_cow_milk_per_five_weeks : (six_cows_milk / cows_per_week) * weeks = 90 := by sorry
  have h_total_cows : total_milk / ((six_cows_milk / cows_per_week) * weeks) = 24 := by sorry
  exact h_total_cows

end number_of_cows_on_farm_l793_793520


namespace engineer_thought_of_l793_793172

def isProperDivisor (n k : ℕ) : Prop :=
  k ≠ 1 ∧ k ≠ n ∧ k ∣ n

def transformDivisors (n m : ℕ) : Prop :=
  ∀ k, isProperDivisor n k → isProperDivisor m (k + 1)

theorem engineer_thought_of (n : ℕ) :
  (∀ m : ℕ, n = 2^2 ∨ n = 2^3 → transformDivisors n m → (m % 2 = 1)) :=
by
  sorry

end engineer_thought_of_l793_793172


namespace count_two_digit_numbers_less_35_l793_793370

open Nat

theorem count_two_digit_numbers_less_35 : 
  let two_digit_numbers := (finset.Ico 10 100) -- The range of two-digit numbers
  let count_satisfying := two_digit_numbers.filter (λ n, (n / 10 < 3 ∨ n % 10 < 5)).card
  count_satisfying = 55 :=
by
  -- Placeholder for the actual proof.
  sorry

end count_two_digit_numbers_less_35_l793_793370


namespace sum_numerator_denominator_l793_793096

-- Given the repeating decimal 0.343434...
def repeating_decimal : ℚ := 34 / 99

-- Prove that the decimal's fraction form in lowest terms adds up to 133.
theorem sum_numerator_denominator : 
  let num := repeating_decimal.num in 
  let denom := repeating_decimal.denom in 
  num + denom = 133 :=
by
  have x : repeating_decimal = 34 / 99 := by sorry
  sorry   -- Placeholder for the proof steps to demonstrate num and denom values lead to 133.

end sum_numerator_denominator_l793_793096


namespace probability_smaller_area_l793_793679

-- Define the equilateral triangle and its properties
variables {P : Type*} [TopologicalSpace P]

structure EquilateralTriangle (P : Type*) :=
  (A B C : P)
  (is_equilateral : true) -- Assuming a placeholder for the property of being equilateral

def centroid (T : EquilateralTriangle P) : P := 
  sorry -- Assuming a definition of centroid

def area (a b c : P) : ℝ :=
  sorry -- Assuming a definition of the area of a triangle

def random_point_inside (T : EquilateralTriangle P) : P :=
  sorry -- Assuming a definition of a random point inside an equilateral triangle

-- Problem statement in Lean 4
theorem probability_smaller_area (T : EquilateralTriangle P) :
  let P := random_point_inside T in
  (area T.A T.B P < area T.A T.C P ∧ area T.A T.B P < area T.B T.C P) →
  (1 / 6) :=
by
  sorry

end probability_smaller_area_l793_793679


namespace unique_triangle_determination_l793_793634

-- Definitions for each type of triangle and their respective conditions
def isosceles_triangle (base_angle : ℝ) (altitude : ℝ) : Type := sorry
def vertex_base_isosceles_triangle (vertex_angle : ℝ) (base : ℝ) : Type := sorry
def circ_radius_side_equilateral_triangle (radius : ℝ) (side : ℝ) : Type := sorry
def leg_radius_right_triangle (leg : ℝ) (radius : ℝ) : Type := sorry
def angles_side_scalene_triangle (angle1 : ℝ) (angle2 : ℝ) (opp_side : ℝ) : Type := sorry

-- Condition: Option A does not uniquely determine a triangle
def option_A_does_not_uniquely_determine : Prop :=
  ∀ (base_angle altitude : ℝ), 
    (∃ t1 t2 : isosceles_triangle base_angle altitude, t1 ≠ t2)

-- Condition: Options B through E uniquely determine the triangle
def options_B_to_E_uniquely_determine : Prop :=
  (∀ (vertex_angle base : ℝ), ∃! t : vertex_base_isosceles_triangle vertex_angle base, true) ∧
  (∀ (radius side : ℝ), ∃! t : circ_radius_side_equilateral_triangle radius side, true) ∧
  (∀ (leg radius : ℝ), ∃! t : leg_radius_right_triangle leg radius, true) ∧
  (∀ (angle1 angle2 opp_side : ℝ), ∃! t : angles_side_scalene_triangle angle1 angle2 opp_side, true)

-- Main theorem combining both conditions
theorem unique_triangle_determination :
  option_A_does_not_uniquely_determine ∧ options_B_to_E_uniquely_determine :=
  sorry

end unique_triangle_determination_l793_793634


namespace prime_iff_even_and_power_of_two_l793_793907

theorem prime_iff_even_and_power_of_two (a n : ℕ) (h_pos_a : a > 1) (h_pos_n : n > 0) :
  Nat.Prime (a^n + 1) → (∃ k : ℕ, a = 2 * k) ∧ (∃ m : ℕ, n = 2^m) :=
by 
  sorry

end prime_iff_even_and_power_of_two_l793_793907


namespace count_two_digit_numbers_less_35_l793_793372

open Nat

theorem count_two_digit_numbers_less_35 : 
  let two_digit_numbers := (finset.Ico 10 100) -- The range of two-digit numbers
  let count_satisfying := two_digit_numbers.filter (λ n, (n / 10 < 3 ∨ n % 10 < 5)).card
  count_satisfying = 55 :=
by
  -- Placeholder for the actual proof.
  sorry

end count_two_digit_numbers_less_35_l793_793372


namespace length_de_l793_793480

open Triangle

/-- In triangle ABC, BC = 40 and ∠C = 45°. Let the perpendicular bisector
of BC intersect BC and AC at D and E, respectively. Prove that DE = 10√2. -/

theorem length_de 
  (ABC : Triangle)
  (B C : Point)
  (BC_40 : dist B C = 40)
  (angle_C_45 : ∠(B, C, AC) = 45)
  (D : Point)
  (midpoint_D : is_midpoint B C D)
  (E : Point)
  (intersection_E : is_perpendicular_bisector_intersection B C D E) :
  dist D E = 10 * real.sqrt 2 :=
begin
  -- proof steps would go here
  sorry
end

end length_de_l793_793480


namespace mushroom_drying_l793_793651

theorem mushroom_drying (M M' : ℝ) (m1 m2 : ℝ) :
  M = 100 ∧ m1 = 0.01 * M ∧ m2 = 0.02 * M' ∧ m1 = 1 → M' = 50 :=
by
  sorry

end mushroom_drying_l793_793651


namespace red_blue_cell_sum_equality_l793_793896

noncomputable def cell_value (i j : ℕ) : ℕ := i + j

theorem red_blue_cell_sum_equality (n : ℕ) (h_n : n ≥ 2)
  (initial_board : matrix (fin (4 * n)) (fin (4 * n)) ℕ := λ i j, cell_value i j)
  (moves_by_alex : fin (n^2) → (fin(4 * n) × fin (4 * n)))
  (valid_moves_by_alex : ∀ (m : fin (n^2)), (moves_by_alex m).1 ≠ (moves_by_alex m).2):

  ∃ (moves_by_jane : list ((fin (4 * n) × fin (4 * n)))),
    (∀ (m : (fin (n^2))), valid_moves_by_alex m) ∧
    (let S_R := Σ m in moves_by_alex, (cell_value (m.1).1.1 (m.1).2.1) in
     let S_B := Σ m in moves_by_alex, (cell_value (m.2).1.1 (m.2).2.1) in
         S_R = S_B) := sorry

end red_blue_cell_sum_equality_l793_793896


namespace student_chose_number_l793_793136

theorem student_chose_number (x : ℤ) (h : 2 * x - 138 = 104) : x = 121 := by
  sorry

end student_chose_number_l793_793136


namespace steve_can_answer_38_questions_l793_793982

theorem steve_can_answer_38_questions (total_questions S : ℕ) 
  (h1 : total_questions = 45)
  (h2 : total_questions - S = 7) :
  S = 38 :=
by {
  -- The proof goes here
  sorry
}

end steve_can_answer_38_questions_l793_793982


namespace repeating_decimal_sum_l793_793109

theorem repeating_decimal_sum (x : ℚ) (h : x = 34 / 999) : x.num + x.denom = 1033 := by 
  sorry

end repeating_decimal_sum_l793_793109


namespace two_digit_numbers_count_l793_793342

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem two_digit_numbers_count :
  { n : ℕ | 10 ≤ n ∧ n ≤ 99 ∧ (tens_digit n < 3 ∨ units_digit n < 5) }.to_finset.card = 55 :=
by
  sorry

end two_digit_numbers_count_l793_793342


namespace correct_operation_l793_793131

theorem correct_operation :
  (∀ a : ℝ, a^4 * a^3 = a^7)
  ∧ (∀ a : ℝ, (a^2)^3 ≠ a^5)
  ∧ (∀ a : ℝ, 3 * a^2 - a^2 ≠ 2)
  ∧ (∀ a b : ℝ, (a - b)^2 ≠ a^2 - b^2) :=
by {
  sorry
}

end correct_operation_l793_793131


namespace min_value_fraction_l793_793760

theorem min_value_fraction (a b : ℝ) (h1 : 2 * a + b = 3) (h2 : a > 0) (h3 : b > 0) (h4 : ∃ n : ℕ, b = n) : 
  (∃ a b : ℝ, 2 * a + b = 3 ∧ a > 0 ∧ b > 0 ∧ (∃ n : ℕ, b = n) ∧ ((1/(2*a) + 2/b) = 2)) := 
by
  sorry

end min_value_fraction_l793_793760


namespace area_AFC_l793_793011

-- Definitions based on the conditions
variable (A B C D E F : Point)
variable (S : ℝ)
variable (BD : Line)
variable (midpoint_D : is_midpoint D A C)
variable (on_BD: lies_on E BD)
variable (DE_fraction : DE / BD = 1/4)
variable (intersect_AF : ∃ (F : Point), lies_on F (line_through A E) ∧ lies_on F (line_through B C))

-- Prove the area of triangle AFC is 2/5 * S
theorem area_AFC (A B C D E F : Point) (S : ℝ)
  (BD : Line) (midpoint_D : is_midpoint D A C) 
  (on_BD: lies_on E BD) (DE_fraction : DE / BD = 1/4)
  (intersect_AF : ∃ (F : Point), lies_on F (line_through A E) ∧ lies_on F (line_through B C))
  : area (triangle A F C) = (2 / 5) * S :=
sorry

end area_AFC_l793_793011


namespace max_k_condition_l793_793825

noncomputable def max_possible_k (x y : ℝ) (hxy : x > 0 ∧ y > 0) : ℝ :=
  let k := (3 : ℝ) / 7
  in k

theorem max_k_condition (x y k : ℝ) (hxy : x > 0 ∧ y > 0) :
  4 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x) → k ≤ (3 : ℝ) / 7 :=
sorry

#eval max_possible_k 1 1 ⟨by norm_num, by norm_num⟩ -- Should return 3/7

end max_k_condition_l793_793825


namespace lunks_needed_for_apples_l793_793823

-- Define the conditions as constants and variables
constant lunk_to_kunk : ℝ := 4 / 2
constant kunk_to_apples : ℝ := 6 / 2
constant two_dozen_apples : ℝ := 24
constant kunks_for_two_dozen : ℝ := two_dozen_apples / kunk_to_apples
constant lunks_for_kunks : ℝ := kunks_for_two_dozen / (2 / lunk_to_kunk)

-- Theorem to prove that 16 lunks are needed for 24 apples
theorem lunks_needed_for_apples : lunks_for_kunks = 16 := by
  -- Here, we would proceed with the proof, but it's omitted as per instructions
  sorry

end lunks_needed_for_apples_l793_793823


namespace num_of_three_digit_with_two_consecutive_identical_digits_l793_793235

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_two_consecutive_identical_digits (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  (d1 = d2 ∧ d2 ≠ d3) ∨ (d2 = d3 ∧ d1 ≠ d2)

theorem num_of_three_digit_with_two_consecutive_identical_digits :
  { n : ℕ // is_three_digit_number n ∧ has_two_consecutive_identical_digits n }.card = 162 := by
  sorry

end num_of_three_digit_with_two_consecutive_identical_digits_l793_793235


namespace integer_solution_3_l793_793434

theorem integer_solution_3 (a : ℝ) : (∃ x : ℤ, ||(x : ℝ) - 2| - 1| = a) 
  → (∃ y : ℤ, ||(y : ℝ) - 2| - 1| = a) 
  → (∃ z : ℤ, ||(z : ℝ) - 2| - 1| = a) 
  → (¬ ∃ w : ℤ, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ ||(w : ℝ) - 2| - 1| = a) 
  → a = 1 :=
sorry

end integer_solution_3_l793_793434


namespace arctan_sum_l793_793204

theorem arctan_sum (x y : ℝ) (hx : x = 3) (hy : y = 7) :
  Real.arctan (x / y) + Real.arctan (y / x) = Real.pi / 2 := 
by
  rw [hx, hy]
  sorry

end arctan_sum_l793_793204


namespace digits_sum_squares_list_integers_sum_squares_even_two_digit_integers_sum_squares_exists_integer_sum_squares_odd_base_l793_793489

-- Definitions of the conditions and questions for Lean 4

-- Part (a)
theorem digits_sum_squares (b : ℕ) (hb : b ≥ 2) (N : ℕ) (hN : N = (digits b).sum_sq) :
  N = 1 ∨ has_two_digits b N := 
sorry

-- Part (b)
theorem list_integers_sum_squares (N : ℕ) (hN : N ≤ 50) (b : ℕ) (hb : b ≥ 2) :
  N ∈ [1, 9, 45] ↔ N = (digits b).sum_sq :=
sorry

-- Part (c)
theorem even_two_digit_integers_sum_squares (b : ℕ) (hb : b ≥ 2) :
  even (finset.filter (λ N, N = (digits b).sum_sq ∧ has_two_digits b N) (finset.range (b * b))).card :=
sorry

-- Part (d)
theorem exists_integer_sum_squares_odd_base (b : ℕ) (hb : b ≥ 2) :
  odd b → ∃ N ≠ 1, N = (digits b).sum_sq :=
sorry

-- Helper Definitions
def digits (b N : ℕ) : list ℕ :=
nat.digits b N

def sum_sq (l : list ℕ) : ℕ :=
list.sum (l.map (λ x, x * x))

def has_two_digits (b N : ℕ) : Prop :=
N < b * b ∧ N ≥ b

end digits_sum_squares_list_integers_sum_squares_even_two_digit_integers_sum_squares_exists_integer_sum_squares_odd_base_l793_793489


namespace sum_numerator_denominator_repeating_decimal_l793_793126

theorem sum_numerator_denominator_repeating_decimal :
  let x := 34 / 99 in
  x.denom * x + x.num = 133 :=
by
  let x : ℚ := 34 / 99
  sorry

end sum_numerator_denominator_repeating_decimal_l793_793126


namespace ord_vertex_of_quadratic_l793_793931

theorem ord_vertex_of_quadratic (a d : ℝ) (x1 x2 : ℝ) (hx : (x2 - x1).abs = d) : 
  ∃ c: ℝ, ∃ b: ℝ, (∀ x, a*x^2 + b*x + c = a*(x - (x1 + x2)/2)^2 - a*(d/2)^2) :=
by
  sorry

end ord_vertex_of_quadratic_l793_793931


namespace taxi_ride_cost_l793_793166

noncomputable def fixed_cost : ℝ := 2.00
noncomputable def cost_per_mile : ℝ := 0.30
noncomputable def distance_traveled : ℝ := 8

theorem taxi_ride_cost :
  fixed_cost + (cost_per_mile * distance_traveled) = 4.40 := by
  sorry

end taxi_ride_cost_l793_793166


namespace product_divisible_by_third_l793_793908

theorem product_divisible_by_third (a b c : Int)
    (h1 : (a + b + c)^2 = -(a * b + a * c + b * c))
    (h2 : a + b ≠ 0) (h3 : b + c ≠ 0) (h4 : a + c ≠ 0) :
    ((a + b) * (a + c) % (b + c) = 0) ∧ ((a + b) * (b + c) % (a + c) = 0) ∧ ((a + c) * (b + c) % (a + b) = 0) :=
  sorry

end product_divisible_by_third_l793_793908


namespace number_2digit_smaller_than_35_l793_793351

/--
Prove that the number of two-digit numbers where at least one digit
is smaller than the corresponding digit in 35 is exactly 55.
-/
theorem number_2digit_smaller_than_35 : 
  (Finset.filter (λ n : ℕ, let d1 := n / 10, d2 := n % 10 in d1 < 3 ∨ d2 < 5) (Finset.range' 10 100)).card = 55 := 
by
  sorry

end number_2digit_smaller_than_35_l793_793351


namespace sin_alpha_value_l793_793255

noncomputable def sin_alpha (α : ℝ) : ℝ :=
  sin α

theorem sin_alpha_value (α : ℝ) (h1 : sin (α - π / 4) = 7 * sqrt 2 / 10) (h2 : cos (2 * α) = 7 / 25) : sin_alpha α = 3 / 5 :=
  sorry

end sin_alpha_value_l793_793255


namespace decagon_intersection_points_l793_793697

theorem decagon_intersection_points : 
  let n := 10 in
  let k := 4 in
  finset.card (finset.powersetLen k (finset.range n)) = 210 :=
by
  sorry

end decagon_intersection_points_l793_793697


namespace age_ratio_7_9_l793_793537

/-- Definition of Sachin and Rahul's ages -/
def sachin_age : ℝ := 24.5
def rahul_age : ℝ := sachin_age + 7

/-- The ratio of Sachin's age to Rahul's age is 7:9 -/
theorem age_ratio_7_9 : sachin_age / rahul_age = 7 / 9 := by
  sorry

end age_ratio_7_9_l793_793537


namespace sum_numerator_denominator_repeating_decimal_l793_793123

theorem sum_numerator_denominator_repeating_decimal :
  let x := 34 / 99 in
  x.denom * x + x.num = 133 :=
by
  let x : ℚ := 34 / 99
  sorry

end sum_numerator_denominator_repeating_decimal_l793_793123


namespace line_intersects_ellipse_with_conditions_l793_793300

theorem line_intersects_ellipse_with_conditions :
  ∃ l : ℝ → ℝ, (∃ A B : ℝ × ℝ, 
  (A.fst^2/6 + A.snd^2/3 = 1 ∧ B.fst^2/6 + B.snd^2/3 = 1) ∧
  A.fst > 0 ∧ A.snd > 0 ∧ B.fst > 0 ∧ B.snd > 0 ∧
  (∃ M N : ℝ × ℝ, 
    M.snd = 0 ∧ N.fst = 0 ∧
    M.fst^2 + N.snd^2 = (2 * Real.sqrt 3)^2 ∧
    (M.snd - A.snd)^2 + (M.fst - A.fst)^2 = (N.fst - B.fst)^2 + (N.snd - B.snd)^2) ∧
    (∀ x, l x + Real.sqrt 2 * x - 2 * Real.sqrt 2 = 0)
) :=
sorry

end line_intersects_ellipse_with_conditions_l793_793300


namespace probability_two_8sided_dice_sum_perfect_square_l793_793607

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def probabilityPerfectSquareSum (dice_sides : ℕ) (perfect_squares : List ℕ) : ℚ :=
  let outcomes := (dice_sides * dice_sides)
  let favorable_outcomes := perfect_squares.sum (λ ps, 
    (List.filter (λ (pair : ℕ × ℕ), pair.fst + pair.snd = ps) ((List.range dice_sides).product (List.range dice_sides))).length)
  favorable_outcomes /. outcomes

theorem probability_two_8sided_dice_sum_perfect_square :
  probabilityPerfectSquareSum 8 [4, 9, 16] = 3 / 16 := sorry

end probability_two_8sided_dice_sum_perfect_square_l793_793607


namespace probability_at_least_one_female_math_group_probability_distribution_of_xi_expected_value_of_xi_l793_793984

noncomputable def prob_at_least_one_female_math_group : ℚ := 9 / 14

noncomputable def prob_distribution_xi : ℤ → ℚ
| 0 := 9 / 112
| 1 := 3 / 7
| 2 := 45 / 112
| 3 := 5 / 56
| _ := 0

noncomputable def expected_value_xi : ℚ := 3 / 2

theorem probability_at_least_one_female_math_group :
  prob_at_least_one_female_math_group = 9 / 14 := sorry

theorem probability_distribution_of_xi :
  (prob_distribution_xi 0 = 9 / 112) ∧
  (prob_distribution_xi 1 = 3 / 7) ∧
  (prob_distribution_xi 2 = 45 / 112) ∧
  (prob_distribution_xi 3 = 5 / 56) := sorry

theorem expected_value_of_xi :
  expected_value_xi = 3 / 2 := sorry

end probability_at_least_one_female_math_group_probability_distribution_of_xi_expected_value_of_xi_l793_793984


namespace notebooks_given_to_Tom_l793_793712

def initial_red_notebooks : ℕ := 15
def initial_blue_notebooks : ℕ := 17
def initial_white_notebooks : ℕ := 19
def notebooks_left : ℕ := 5

theorem notebooks_given_to_Tom :
  (initial_red_notebooks + initial_blue_notebooks + initial_white_notebooks - notebooks_left) = 46 :=
by
  calculate 
  sorry

end notebooks_given_to_Tom_l793_793712


namespace reflection_matrix_correct_l793_793494

def reflection_matrix (a b c : ℝ) : Matrix (Fin 3) (Fin 1) ℝ :=
  ![![ - (1 / 3) * a + (2 / 3) * b + (4 / 3) * c ],
    ![ (4 / 3) * a + (4 / 3) * b + (5 / 3) * c ],
    ![ (1 / 3) * a + (5 / 3) * b + (2 / 3) * c ]]

theorem reflection_matrix_correct :
  ∃ S : Matrix (Fin 3) (Fin 3) ℝ,
    (∀ (u : Vector ℝ), S * u = reflection_matrix u.vec_head u.vec_tail.vec_head u.vec_tail.vec_tail.vec_head) ∧
    S = ![![ - (1 / 3), (2 / 3), (4 / 3) ],
          ![ (4 / 3), (4 / 3), (5 / 3) ],
          ![ (1 / 3), (5 / 3), (2 / 3)] ] :=
by sorry

end reflection_matrix_correct_l793_793494


namespace number_2digit_smaller_than_35_l793_793348

/--
Prove that the number of two-digit numbers where at least one digit
is smaller than the corresponding digit in 35 is exactly 55.
-/
theorem number_2digit_smaller_than_35 : 
  (Finset.filter (λ n : ℕ, let d1 := n / 10, d2 := n % 10 in d1 < 3 ∨ d2 < 5) (Finset.range' 10 100)).card = 55 := 
by
  sorry

end number_2digit_smaller_than_35_l793_793348


namespace max_temperature_difference_required_time_interval_l793_793574

def temperature_function (t : ℝ) : ℝ := 10 - sqrt 3 * cos (π / 12 * t) - sin (π / 12 * t)

theorem max_temperature_difference : 
  ∃ max_diff : ℝ, max_diff = 4 ∧
  ∀ t ∈ set.Ico 0 24, 
    let f_t := temperature_function t in
    8 ≤ f_t ∧ f_t ≤ 12 :=
sorry

theorem required_time_interval :
  ∀ t ∈ set.Ico 0 24, 
  ¬ (10 < t ∧ t < 18) →
  temperature_function t ≤ 11 :=
sorry

end max_temperature_difference_required_time_interval_l793_793574


namespace greatest_50_supportive_X_correct_l793_793001

noncomputable def greatest_50_supportive_X (a : Fin 50 → ℝ) : ℝ :=
if ∃ X, ∀ a : Fin 50 → ℝ, (∑ i : Fin 50, a i).floor = ∑ i : Fin 50, a i ∧ 
  (∃ i, |a i - 0.5| >= X) 
then 0.01 else 0

theorem greatest_50_supportive_X_correct :
  ∀ a : Fin 50 → ℝ, (∑ i : Fin 50, a i).floor = ∑ i : Fin 50, a i → 
  (∃ i, |a i - 0.5| >= 0.01) :=
sorry

end greatest_50_supportive_X_correct_l793_793001


namespace additional_amount_each_payment_l793_793656

-- Define the given conditions and the quantity to be proved
def payments := List.append (List.replicate 12 410) (List.replicate 40 475) -- List of payments (12 of $410, 40 of $410 + $65)
def average_payment := (List.sum payments) / (List.length payments) -- Calculating the average payment

theorem additional_amount_each_payment :
  let x := 65 in
  average_payment = 460 → x = 65 := 
by
  sorry

end additional_amount_each_payment_l793_793656


namespace housewife_spending_l793_793683

theorem housewife_spending (P R M : ℝ) (h1 : R = 65) (h2 : R = 0.75 * P) (h3 : M / R - M / P = 5) :
  M = 1300 :=
by
  -- Proof steps will be added here.
  sorry

end housewife_spending_l793_793683


namespace parabola_intersections_l793_793042

theorem parabola_intersections :
  let y := fun x => x^2 - 3*x + 2 in
  (y 0 = 2) ∧ 
  ({x : ℝ | y x = 0} = {2, 1} : set ℝ) ∧
  ({(0, y 0)} = {(0, 2)} : set (ℝ × ℝ)) ∧
  ({(x, 0) | x ∈ {x : ℝ | y x = 0}} = {(2, 0), (1, 0)} : set (ℝ × ℝ)) :=
by
  sorry

end parabola_intersections_l793_793042


namespace base_length_of_isosceles_triangle_l793_793086

theorem base_length_of_isosceles_triangle
  (A : ℝ) (s : ℝ) (b : ℝ)
  (hA : A = 3) (hS : s = 25) 
  (hArea : A = (1 / 2) * b * real.sqrt (s^2 - (b / 2)^2)) :
  b = 48 ∨ b = 14 :=
by
  sorry

end base_length_of_isosceles_triangle_l793_793086


namespace find_volume_omega2_l793_793447

noncomputable def volume_omega2 : ℝ :=
  let Ω2 := { p : ℝ × ℝ × ℝ | p.1^2 + p.2^2 + (p.3^2 : ℝ) <= 1 }
  measure (volume : Measure (ℝ × ℝ × ℝ)) Ω2

theorem find_volume_omega2 : volume_omega2 = 7 :=
sorry

end find_volume_omega2_l793_793447


namespace color_preference_blue_percentage_l793_793962

theorem color_preference_blue_percentage : 
(40 + 70 + 30 + 50 + 30 + 40 = 260) → 
(70 / 260 * 100 = 27) := 
by 
  sorry

end color_preference_blue_percentage_l793_793962


namespace range_of_a_l793_793257

def condition_p (a : ℝ) : Prop :=
  ∀ k : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (k * x₁ + 1)^2 + (k * x₂ + 1)^2 = 1

def condition_q (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, 4^x₀ - 2^x₀ - a ≤ 0

theorem range_of_a (a : ℝ) : 
  (¬((condition_p a) ∧ (condition_q a)) ∧ ((condition_p a) ∨ (condition_q a))) ↔ (-1/4 ≤ a ∧ a ≤ 1) :=
begin
  sorry
end

end range_of_a_l793_793257


namespace sum_numerator_denominator_l793_793094

-- Given the repeating decimal 0.343434...
def repeating_decimal : ℚ := 34 / 99

-- Prove that the decimal's fraction form in lowest terms adds up to 133.
theorem sum_numerator_denominator : 
  let num := repeating_decimal.num in 
  let denom := repeating_decimal.denom in 
  num + denom = 133 :=
by
  have x : repeating_decimal = 34 / 99 := by sorry
  sorry   -- Placeholder for the proof steps to demonstrate num and denom values lead to 133.

end sum_numerator_denominator_l793_793094


namespace combination_identity_l793_793272

theorem combination_identity (n : ℕ) (h : nat.choose (n+1) 7 - nat.choose n 7 = nat.choose n 8) : n = 14 :=
sorry

end combination_identity_l793_793272


namespace transformed_equation_solutions_l793_793781

theorem transformed_equation_solutions :
  (∀ x : ℝ, x^2 + 2 * x - 3 = 0 → (x = 1 ∨ x = -3)) →
  (∀ x : ℝ, (x + 3)^2 + 2 * (x + 3) - 3 = 0 → (x = -2 ∨ x = -6)) :=
by
  intro h
  sorry

end transformed_equation_solutions_l793_793781


namespace lucas_50_mod_5_l793_793955

-- Define the Lucas sequence
def lucas : ℕ → ℕ
| 0       := 1
| 1       := 3
| (n + 2) := lucas n + lucas (n + 1)

-- Proof statement
theorem lucas_50_mod_5 : (lucas 50) % 5 = 3 :=
by sorry

end lucas_50_mod_5_l793_793955


namespace analytical_expression_of_f_value_of_b_in_triangle_l793_793790

theorem analytical_expression_of_f (ω : ℝ) (x : ℝ) (h_ω_pos : ω > 0) (hx : x ∈ ℝ) (h_period : ∀ x, f (x + π) = f x) :
  f(x) = 2 * sin (2 * x + π / 6) - 1 :=
sorry

theorem value_of_b_in_triangle {α a c : ℝ} (α_pos : α > 0) 
  (angle_B : ℝ) (h_dot : ac * cos B = 3 / 2) (h_sum : a + c = 4) : 
  let b := sqrt 7 in b :=
sorry

end analytical_expression_of_f_value_of_b_in_triangle_l793_793790


namespace inequality_solution_l793_793032

theorem inequality_solution (x : ℝ)
  (h : ∀ x, x^2 + 2 * x + 7 > 0) :
  (x - 3) / (x^2 + 2 * x + 7) ≥ 0 ↔ x ∈ Set.Ici 3 :=
by
  sorry

end inequality_solution_l793_793032


namespace inverse_of_f_at_3_l793_793912

noncomputable def f (x : ℝ) : ℝ := x^2 + 2

theorem inverse_of_f_at_3 :
  ∃ x : ℝ, -2 ≤ x ∧ x < 0 ∧ f x = 3 ∧ f⁻¹(3) = x :=
begin
  use -1,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num },
  sorry
end

end inverse_of_f_at_3_l793_793912


namespace num_suitable_two_digit_numbers_l793_793373

/-- 
How many two-digit numbers have at least one digit that is smaller than the corresponding digit in the number 35?
-/
theorem num_suitable_two_digit_numbers : 
  let two_digit_numbers := { n : ℕ | 10 ≤ n ∧ n ≤ 99 },
      suitable_numbers := { n ∈ two_digit_numbers | (n / 10 < 3) ∨ (n % 10 < 5) } in
  suitable_numbers.card = 55 :=
by
  sorry

end num_suitable_two_digit_numbers_l793_793373


namespace concyclic_points_l793_793865

-- Definitions and conditions
variables {A B C P Q D E F G : Type*}
variables [acute_triangle A B C] (h1 : ∠BAC > ∠BCA) (h2: is_point_on_side BC P (∠PAB = ∠BCA)) 
          (h3: circumcircle_meets APB AC Q) (h4: ∠QDC = ∠CAP)
          (h5: is_point_on_line BD E (CE = CD)) 
          (h6: circumcircle_meets CQE CD F) 
          (h7: line_meets BC QF G)

-- The theorem
theorem concyclic_points {A B C P Q D E F G : Type*}
  [acute_triangle A B C]
  (h1 : ∠BAC > ∠BCA)
  (h2 : is_point_on_side BC P (∠PAB = ∠BCA))
  (h3 : circumcircle_meets APB AC Q)
  (h4 : ∠QDC = ∠CAP)
  (h5 : is_point_on_line BD E (CE = CD))
  (h6 : circumcircle_meets CQE CD F)
  (h7 : line_meets BC QF G) :
  cyclic_points B D F G :=
sorry

end concyclic_points_l793_793865


namespace rubert_james_ratio_l793_793024

-- Definitions and conditions from a)
def adam_candies : ℕ := 6
def james_candies : ℕ := 3 * adam_candies
def rubert_candies (total_candies : ℕ) : ℕ := total_candies - (adam_candies + james_candies)
def total_candies : ℕ := 96

-- Statement to prove the ratio
theorem rubert_james_ratio : 
  (rubert_candies total_candies) / james_candies = 4 :=
by
  -- Proof is not required, so we leave it as sorry.
  sorry

end rubert_james_ratio_l793_793024


namespace y_at_x8_l793_793420

theorem y_at_x8 (k : ℝ) (y : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, y x = k * x^(1/3))
  (h2 : y 64 = 4 * Real.sqrt 3) :
  y 8 = 2 * Real.sqrt 3 :=
by
  sorry

end y_at_x8_l793_793420


namespace minimum_value_distance_sum_l793_793320

noncomputable def minimum_distance_sum : ℝ :=
  let l := {m // ∃ m, 3 * (-1) - 4 * 2 + m = 0} in
  let l_parametric := {t // (λ t, (-1) + 4 / 5 * t, 2 + 3 / 5 * t)} in
  let G := {p // (p.1 - 1) ^ 2 + (p.2 - 1) ^ 2 = 2} in
  let O := (0, 0 : ℝ × ℝ) in
  let A := (2, 0 : ℝ × ℝ) in
  let B := (2, 2 : ℝ × ℝ) in
  let C := (0, 2 : ℝ × ℝ) in
  ∀ P : ℝ × ℝ, P ∈ l_parametric →
    let distance_sum := (P.1)^2 + (P.2)^2 + (P.1 - 2)^2 + (P.2)^2 + (P.1 - 2)^2 + (P.2 - 2)^2 + (P.1)^2 + (P.2 - 2)^2 in
    distance_sum = 24

theorem minimum_value_distance_sum : minimum_distance_sum := by
  sorry

end minimum_value_distance_sum_l793_793320


namespace find_a_l793_793829

noncomputable def coefficient_of_x3_in_binomial_expansion (a : ℝ) : ℝ :=
  (- (1 / a))^3 * (finset.choose 6 3)

theorem find_a (a : ℝ) (h : coefficient_of_x3_in_binomial_expansion a = 5 / 2) :
  a = -2 :=
sorry

end find_a_l793_793829


namespace BobPiesNumber_l793_793076

structure PieBakingConditions where
  tom_radius : ℝ
  bob_leg1 : ℝ
  bob_leg2 : ℝ
  tom_pies_per_batch : ℕ

noncomputable def tom_pie_area (r : ℝ) : ℝ := π * r^2

noncomputable def bob_pie_area (a b : ℝ) : ℝ := (1/2) * a * b

theorem BobPiesNumber (conditions : PieBakingConditions) : (n : ℕ) :=
  let tom_total_area := (conditions.tom_pies_per_batch : ℝ) * tom_pie_area conditions.tom_radius
  let bob_area := bob_pie_area conditions.bob_leg1 conditions.bob_leg2
  n = Int.ofNat (Real.ceil (tom_total_area / bob_area)) := 50
:= sorry

-- Given conditions
def given_conditions : PieBakingConditions :=
  { tom_radius := 8,
    bob_leg1 := 6,
    bob_leg2 := 8,
    tom_pies_per_batch := 6 }

#eval BobPiesNumber given_conditions

end BobPiesNumber_l793_793076


namespace hexagon_numbers_zero_l793_793914

theorem hexagon_numbers_zero (n : ℕ) (a1 a2 a3 a4 a5 a6 : ℕ):
  let S := a1 + a2 + a3 + a4 + a5 + a6 in
  (S = 2 ∨ S % 2 = 1) → 
  ∃ (b1 b2 b3 b4 b5 b6 : ℕ), (b1 = 0 ∧ b2 = 0 ∧ b3 = 0 ∧ b4 = 0 ∧ b5 = 0 ∧ b6 = 0) := 
sorry

end hexagon_numbers_zero_l793_793914


namespace find_n_l793_793175

noncomputable def binom (n k : ℕ) := Nat.choose n k

theorem find_n 
  (n : ℕ)
  (h1 : (binom (n-6) 7) / binom n 7 = (6 * binom (n-7) 6) / binom n 7)
  : n = 48 := by
  sorry

end find_n_l793_793175


namespace value_of_a_l793_793805

theorem value_of_a (a : ℝ) :
  let M := {5, a^2 - 3a + 5}
  let N := {1, 3}
  M ∩ N ≠ ∅ → (a = 1 ∨ a = 2) :=
by {
  sorry
}

end value_of_a_l793_793805


namespace exists_locus_30deg_l793_793746

noncomputable def square_locus_30deg (square : set (ℝ × ℝ)) : set (ℝ × ℝ) :=
  let side_centers := {c : ℝ × ℝ | ∃ (p1 p2 : ℝ × ℝ), p1 ∈ square ∧ p2 ∈ square ∧ dist c p1 = dist c p2 ∧ ∠ p1 c p2 = π / 3}
  let diagonal_centers := {c : ℝ × ℝ | ∃ (p1 p2 : ℝ × ℝ), p1 ∈ square ∧ p2 ∈ square ∧ dist c p1 = dist c p2 ∧ ∠ p1 c p2 = 2 * π / 3}
  side_centers ∪ diagonal_centers

theorem exists_locus_30deg (square : set (ℝ × ℝ)) : 
  ∃ locus : set (ℝ × ℝ), ∀ P, P ∈ locus ↔ (∠ (top_left_vertex square) P (top_right_vertex square) = π / 6
                                   ∨ ∠ (top_right_vertex square) P (bottom_right_vertex square) = π / 6
                                   ∨ ∠ (bottom_right_vertex square) P (bottom_left_vertex square) = π / 6
                                   ∨ ∠ (bottom_left_vertex square) P (top_left_vertex square) = π / 6) :=
sorry

end exists_locus_30deg_l793_793746


namespace intersecting_lines_condition_l793_793562

theorem intersecting_lines_condition (λ : ℝ) (x y : ℝ) :
  (∃ m n : ℝ, m + n = 4 ∧ m * n = 3 ∧ n - m = λ) ↔ (λ = 2 ∨ λ = -2) :=
by {
  sorry
}

end intersecting_lines_condition_l793_793562


namespace strictly_increasing_intervals_l793_793571

def f (x : ℝ) : ℝ := 2 * Real.sin (π / 4 - x)

theorem strictly_increasing_intervals :
  ∀ k : ℤ,
    ∀ x : ℝ,
      ( (3 * π) / 4 + 2 * (k : ℝ) * π ≤ x ∧ x ≤ (7 * π) / 4 + 2 * (k : ℝ) * π) →
        ∀ x1 x2 : ℝ,
          (3 * π) / 4 + 2 * (k : ℝ) * π ≤ x1 ∧ 
          x1 < x2 ∧ 
          x2 ≤ (7 * π) / 4 + 2 * (k : ℝ) * π →
          f x1 < f x2 :=
by
  sorry

end strictly_increasing_intervals_l793_793571


namespace count_two_digit_numbers_less_35_l793_793368

open Nat

theorem count_two_digit_numbers_less_35 : 
  let two_digit_numbers := (finset.Ico 10 100) -- The range of two-digit numbers
  let count_satisfying := two_digit_numbers.filter (λ n, (n / 10 < 3 ∨ n % 10 < 5)).card
  count_satisfying = 55 :=
by
  -- Placeholder for the actual proof.
  sorry

end count_two_digit_numbers_less_35_l793_793368


namespace two_digit_numbers_less_than_35_l793_793362

theorem two_digit_numbers_less_than_35 : 
  ∃ n, n ∈ finset.range 90 ∧ (10 ≤ n + 10 ∧ n + 10 ≤ 99) ∧ 
  (let tens_digit := (n + 10) / 10 in 
   let units_digit := (n + 10) % 10 in 
   tens_digit < 3 ∨ units_digit < 5) → 
  nat.card {n | n ∈ finset.range 90 ∧ (10 ≤ n + 10 ∧ n + 10 ≤ 99) ∧ 
                    (let tens_digit := (n + 10) / 10 in 
                     let units_digit := (n + 10) % 10 in 
                     tens_digit < 3 ∨ units_digit < 5)} = 55
:= sorry

end two_digit_numbers_less_than_35_l793_793362


namespace fraction_of_journey_by_bus_l793_793670

theorem fraction_of_journey_by_bus:
  (total_journey distance_by_rail distance_on_foot distance_by_bus : ℝ)
  (h1: total_journey = 130)
  (h2: distance_by_rail = (3/5) * 130)
  (h3: distance_on_foot = 6.5)
  (h4: distance_by_bus = total_journey - (distance_by_rail + distance_on_foot)) :
  distance_by_bus / total_journey = 45.5 / 130 :=
by
  sorry

end fraction_of_journey_by_bus_l793_793670


namespace power_function_value_l793_793049

-- Define the power function that pass through (2, 1/4)
def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- Define the point condition
def passes_through (α : ℝ) : Prop := power_function α 2 = 1 / 4

-- The main theorem to be proven
theorem power_function_value (α : ℝ) (h : passes_through α) : power_function α (-3) = 1 / 9 :=
by sorry

end power_function_value_l793_793049


namespace two_digit_numbers_less_than_35_l793_793356

theorem two_digit_numbers_less_than_35 : 
  let count : ℕ := (99 - 10 + 1) - (7 * 5)
  count = 55 :=
by
  -- definition of total number of two-digit numbers
  let total_two_digit_numbers : ℕ := 99 - 10 + 1
  -- definition of the number of unsuitable two-digit numbers
  let unsuitable_numbers : ℕ := 7 * 5
  -- definition of the count of suitable two-digit numbers
  let count : ℕ := total_two_digit_numbers - unsuitable_numbers
  -- verifying the final count
  exact rfl

end two_digit_numbers_less_than_35_l793_793356


namespace at_least_two_dice_show_the_same_number_l793_793015

theorem at_least_two_dice_show_the_same_number :
    (∑ (d₁ d₂ d₃ d₄ d₅ d₆ : fin 6), 1) / (6^6) = 319 / 324 := sorry

end at_least_two_dice_show_the_same_number_l793_793015


namespace additional_toothpicks_for_staircase_l793_793888

def num_toothpicks (n : ℕ) : ℕ :=
  match n with
  | 1 => 6
  | _ => (num_toothpicks (n - 1) + 2*(n + 1))

theorem additional_toothpicks_for_staircase : num_toothpicks 7 - num_toothpicks 4 = 42 :=
by
  -- We do not provide the proof here, only the statement as required.
  sorry

end additional_toothpicks_for_staircase_l793_793888


namespace find_valid_sequences_l793_793495

def Point : Type := ℝ × ℝ

def Triangle := (Point × Point × Point)

def T : Triangle := ((0, 0), (4, 0), (0, 3))

def rotate_90 (p : Point) : Point := (-p.2, p.1)
def rotate_180 (p : Point) : Point := (-p.1, -p.2)
def rotate_270 (p : Point) : Point := (p.2, -p.1)
def reflect_x (p : Point) : Point := (p.1, -p.2)
def reflect_y (p : Point) : Point := (-p.1, p.2)
def scale_x2 (p : Point) : Point := (2 * p.1, p.2)

def transform (t : Point → Point) (T : Triangle) : Triangle :=
  (t T.1, t T.2, t T.3)

def transformations := [rotate_90, rotate_180, rotate_270, reflect_x, reflect_y, scale_x2]

def is_original_position (T1 T2 : Triangle) : Prop :=
  T1 = T2

theorem find_valid_sequences :
  let sequences := (list.zip transformations transformations).product transformations
  (sequences.filter (λ seq, 
    let (t1, t23) := seq in
    let (t2, t3) := t23 in
    is_original_position (transform t3 (transform t2 (transform t1 T))) T
  )).length = 15 :=
by sorry

end find_valid_sequences_l793_793495


namespace average_waiting_time_l793_793689

/-- 
A traffic light at a pedestrian crossing allows pedestrians to cross the street 
for one minute and prohibits crossing for two minutes. Prove that the average 
waiting time for a pedestrian who arrives at the intersection is 40 seconds.
-/ 
theorem average_waiting_time (pG : ℝ) (pR : ℝ) (eTG : ℝ) (eTR : ℝ) (cycle : ℝ) :
  pG = 1 / 3 ∧ pR = 2 / 3 ∧ eTG = 0 ∧ eTR = 1 ∧ cycle = 3 → 
  (eTG * pG + eTR * pR) * (60 / cycle) = 40 :=
by
  sorry

end average_waiting_time_l793_793689


namespace smallest_seven_consecutive_even_sum_588_l793_793565

theorem smallest_seven_consecutive_even_sum_588 : 
  ∃ (a : ℤ), (∀ i ∈ Finset.range 7, even (a + 2 * i)) ∧ (Finset.sum (Finset.range 7) (λ i, a + 2 * i) = 588) ∧ a = 78 := 
by
  sorry

end smallest_seven_consecutive_even_sum_588_l793_793565


namespace find_f_at_4_l793_793045

theorem find_f_at_4 (f : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → 3 * f x - 2 * f (1 / x) = x) : 
  f 4 = 5 / 2 :=
sorry

end find_f_at_4_l793_793045


namespace recorder_price_new_l793_793006

theorem recorder_price_new (a b : ℕ) (h1 : 10 * a + b < 50) (h2 : 10 * b + a = (10 * a + b) * 12 / 10) :
  10 * b + a = 54 :=
by
  sorry

end recorder_price_new_l793_793006


namespace first_term_geometric_sequence_l793_793978

theorem first_term_geometric_sequence (a r : ℚ) 
  (h3 : a * r^(3-1) = 24)
  (h4 : a * r^(4-1) = 36) :
  a = 32 / 3 :=
by
  sorry

end first_term_geometric_sequence_l793_793978


namespace shots_made_last_10_l793_793585

noncomputable def fraction (n : ℤ) (d : ℤ) : ℤ :=
  if d ≠ 0 then n / d else 0

theorem shots_made_last_10 {t_ini_shots t_additional_shots t_ini_percentage t_final_percentage : ℕ} :
  t_ini_shots = 30 →
  t_additional_shots = 10 →
  t_ini_percentage = 60 →
  t_final_percentage = 62 →
  let t_ini_made := (t_ini_percentage * t_ini_shots) / 100 in
  let t_total_shots := t_ini_shots + t_additional_shots in
  let t_final_made := (t_final_percentage * t_total_shots + 50) / 100 in
  (t_final_made - t_ini_made) = 7 :=
by
  intros _ _ _ _
  let t_ini_made := (t_ini_percentage * t_ini_shots) / 100
  let t_total_shots := t_ini_shots + t_additional_shots
  let t_final_made := (t_final_percentage * t_total_shots + 50) / 100
  have h := t_final_made - t_ini_made
  exact h ▸ rfl

end shots_made_last_10_l793_793585


namespace jan_keeps_on_hand_l793_793485

theorem jan_keeps_on_hand (total_length : ℕ) (section_length : ℕ) (friend_fraction : ℚ) (storage_fraction : ℚ) 
  (total_sections : ℕ) (sections_to_friend : ℕ) (remaining_sections : ℕ) (sections_in_storage : ℕ) (sections_on_hand : ℕ) :
  total_length = 1000 → section_length = 25 → friend_fraction = 1 / 4 → storage_fraction = 1 / 2 →
  total_sections = total_length / section_length →
  sections_to_friend = friend_fraction * total_sections →
  remaining_sections = total_sections - sections_to_friend →
  sections_in_storage = storage_fraction * remaining_sections →
  sections_on_hand = remaining_sections - sections_in_storage →
  sections_on_hand = 15 :=
by sorry

end jan_keeps_on_hand_l793_793485


namespace sequence_infinite_divisibility_l793_793492

theorem sequence_infinite_divisibility :
  ∃ (u : ℕ → ℤ), (∀ n, u (n + 2) = u (n + 1) ^ 2 - u n) ∧ u 1 = 39 ∧ u 2 = 45 ∧ (∀ N, ∃ k ≥ N, 1986 ∣ u k) := 
by
  sorry

end sequence_infinite_divisibility_l793_793492


namespace length_DE_is_20_l793_793458

open_locale real
open_locale complex_conjugate

noncomputable def in_triangle_config_with_angle (ABC : Type) [triangle ABC]
  (BC_length : ℝ) (angle_C : real.angle) (perp_bisector_intersect_D : point)
  (perp_bisector_intersect_E : point) : Prop :=
  -- define the conditions
  BC_length = 40 ∧
  angle_C = 45 ∧
  is_midpoint_of_BC ABC perp_bisector_intersect_D ∧
  is_perpendicular_bisector_intersects_AC ABC perp_bisector_intersect_D perp_bisector_intersect_E

noncomputable def length_DE (D E: point) : ℝ :=
  distance_between D E

theorem length_DE_is_20 {ABC : Type} [triangle ABC]
  (BC_length : ℝ) (angle_C : real.angle) (perp_bisector_intersect_D : point)
  (perp_bisector_intersect_E : point)
  (h : in_triangle_config_with_angle ABC BC_length angle_C perp_bisector_intersect_D perp_bisector_intersect_E):
  length_DE perp_bisector_intersect_D perp_bisector_intersect_E = 20 :=
begin
  sorry,
end

end length_DE_is_20_l793_793458


namespace integers_multiples_between_1_and_300_l793_793335

theorem integers_multiples_between_1_and_300 : 
  {n : ℕ | 1 ≤ n ∧ n ≤ 300 ∧ n % 30 = 0 ∧ n % 8 ≠ 0 ∧ n % 9 ≠ 0}.card = 5 := 
by
  sorry

end integers_multiples_between_1_and_300_l793_793335


namespace solve_equation_l793_793570

theorem solve_equation (x : ℝ) (h1 : 2 * x + 1 ≠ 0) (h2 : 4 * x ≠ 0) : 
  (3 / (2 * x + 1) = 5 / (4 * x)) ↔ (x = 2.5) :=
by 
  sorry

end solve_equation_l793_793570


namespace unique_polynomial_l793_793936

theorem unique_polynomial (n : ℤ) :
  ∃! (Q : Polynomial ℤ), (∀ coeff in Q.to_list, coeff ∈ finset.range 10) ∧ (Q.eval (-2) = n) ∧ (Q.eval (-5) = n) :=
sorry

end unique_polynomial_l793_793936


namespace pieces_of_cheese_per_student_l793_793063

theorem pieces_of_cheese_per_student
  (slices_per_pizza : ℕ) (number_of_pizzas : ℕ) (leftover_cheese : ℕ) (leftover_onion : ℕ) 
  (number_of_students : ℕ) (onion_per_student : ℕ) : 
  slices_per_pizza = 18 →
  number_of_pizzas = 6 →
  leftover_cheese = 8 →
  leftover_onion = 4 →
  number_of_students = 32 →
  onion_per_student = 1 →
  ((slices_per_pizza * number_of_pizzas - leftover_cheese - leftover_onion - 
  number_of_students * onion_per_student) / number_of_students = 2) :=
by
  intros h1 h2 h3 h4 h5 h6
  have total_slices : ℕ := slices_per_pizza * number_of_pizzas
  have used_slices : ℕ := total_slices - leftover_cheese - leftover_onion
  have total_onslices_used : ℕ := number_of_students * onion_per_student
  have total_cheese_used : ℕ := used_slices - total_onslices_used
  have cheese_per_student : ℕ := total_cheese_used / number_of_students
  have hypothesis := ((slices_per_pizza * number_of_pizzas - leftover_cheese - leftover_onion - 
  number_of_students * onion_per_student) / number_of_students)
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end pieces_of_cheese_per_student_l793_793063


namespace and_15_and_l793_793243

def x_and (x : ℝ) : ℝ := 8 - x
def and_x (x : ℝ) : ℝ := x - 8

theorem and_15_and : and_x (x_and 15) = -15 :=
by
  sorry

end and_15_and_l793_793243


namespace num_two_digit_numbers_with_digit_less_than_35_l793_793386

theorem num_two_digit_numbers_with_digit_less_than_35 : 
  let total_two_digit := 90 in
  let unsuitable_numbers := 35 in
  let suitable_numbers := total_two_digit - unsuitable_numbers
  in suitable_numbers = 55 :=
by 
  let total_two_digit := 90
  let unsuitable_numbers := 35
  let suitable_numbers := total_two_digit - unsuitable_numbers
  show suitable_numbers = 55, from sorry

end num_two_digit_numbers_with_digit_less_than_35_l793_793386


namespace total_DVDs_CDs_sold_l793_793854

theorem total_DVDs_CDs_sold (C D : ℕ) (h1 : D = 1.6 * C) (h2 : D = 168) : 
  D + C = 273 :=
by
  sorry

end total_DVDs_CDs_sold_l793_793854


namespace number_of_stones_per_bracelet_l793_793713

def total_number_of_stones : float := 88.0
def number_of_bracelets : float := 8.0

theorem number_of_stones_per_bracelet : total_number_of_stones / number_of_bracelets = 11.0 := by
  sorry

end number_of_stones_per_bracelet_l793_793713


namespace number_of_cows_on_farm_l793_793519

theorem number_of_cows_on_farm :
  (∀ (cows_per_week : ℤ) (six_cows_milk : ℤ) (total_milk : ℤ) (weeks : ℤ),
    cows_per_week = 6 → 
    six_cows_milk = 108 →
    total_milk = 2160 →
    weeks = 5 →
    (total_milk / (six_cows_milk / cows_per_week * weeks)) = 24) :=
by
  intros cows_per_week six_cows_milk total_milk weeks h1 h2 h3 h4
  have h_cow_milk_per_week : six_cows_milk / cows_per_week = 18 := by sorry
  have h_cow_milk_per_five_weeks : (six_cows_milk / cows_per_week) * weeks = 90 := by sorry
  have h_total_cows : total_milk / ((six_cows_milk / cows_per_week) * weeks) = 24 := by sorry
  exact h_total_cows

end number_of_cows_on_farm_l793_793519


namespace total_DVDs_CDs_sold_l793_793855

theorem total_DVDs_CDs_sold (C D : ℕ) (h1 : D = 1.6 * C) (h2 : D = 168) : 
  D + C = 273 :=
by
  sorry

end total_DVDs_CDs_sold_l793_793855


namespace concyclic_points_l793_793866

-- Definitions and conditions
variables {A B C P Q D E F G : Type*}
variables [acute_triangle A B C] (h1 : ∠BAC > ∠BCA) (h2: is_point_on_side BC P (∠PAB = ∠BCA)) 
          (h3: circumcircle_meets APB AC Q) (h4: ∠QDC = ∠CAP)
          (h5: is_point_on_line BD E (CE = CD)) 
          (h6: circumcircle_meets CQE CD F) 
          (h7: line_meets BC QF G)

-- The theorem
theorem concyclic_points {A B C P Q D E F G : Type*}
  [acute_triangle A B C]
  (h1 : ∠BAC > ∠BCA)
  (h2 : is_point_on_side BC P (∠PAB = ∠BCA))
  (h3 : circumcircle_meets APB AC Q)
  (h4 : ∠QDC = ∠CAP)
  (h5 : is_point_on_line BD E (CE = CD))
  (h6 : circumcircle_meets CQE CD F)
  (h7 : line_meets BC QF G) :
  cyclic_points B D F G :=
sorry

end concyclic_points_l793_793866


namespace cupcakes_left_after_distribution_l793_793727

theorem cupcakes_left_after_distribution : 
  ∀ (total_cupcakes students sick teacher aide : ℕ),
  total_cupcakes = 30 →
  students = 27 →
  sick = 3 →
  teacher = 1 →
  aide = 1 →
  total_cupcakes - (students - sick + teacher + aide) = 4 :=
by
  intros total_cupcakes students sick teacher aide hcups hstudents hsick hteacher haide
  rw [hcups, hstudents, hsick, hteacher, haide]
  norm_num
  exact rfl

end cupcakes_left_after_distribution_l793_793727


namespace slope_problem_l793_793395

theorem slope_problem (m : ℝ) (h1 : 0 < m)
  (h2 : ∀ x y, (x, y) = (2 * m, m + 1) ∨ (x, y) = (1, 2 * m)) :
  m = real.sqrt 2 / 2 :=
begin
  -- proof goes here
  sorry
end

end slope_problem_l793_793395


namespace y_at_x_equals_8_l793_793404

theorem y_at_x_equals_8 (k : ℝ) (h1 : ∀ x y, y = k * x^(1/3))
    (h2 : 4 * real.sqrt 3 = k * 64^(1/3)) : k * 8^(1/3) = 2 * real.sqrt 3 :=
by
  sorry

end y_at_x_equals_8_l793_793404


namespace number_2digit_smaller_than_35_l793_793345

/--
Prove that the number of two-digit numbers where at least one digit
is smaller than the corresponding digit in 35 is exactly 55.
-/
theorem number_2digit_smaller_than_35 : 
  (Finset.filter (λ n : ℕ, let d1 := n / 10, d2 := n % 10 in d1 < 3 ∨ d2 < 5) (Finset.range' 10 100)).card = 55 := 
by
  sorry

end number_2digit_smaller_than_35_l793_793345


namespace sum_numerator_denominator_l793_793092

-- Given the repeating decimal 0.343434...
def repeating_decimal : ℚ := 34 / 99

-- Prove that the decimal's fraction form in lowest terms adds up to 133.
theorem sum_numerator_denominator : 
  let num := repeating_decimal.num in 
  let denom := repeating_decimal.denom in 
  num + denom = 133 :=
by
  have x : repeating_decimal = 34 / 99 := by sorry
  sorry   -- Placeholder for the proof steps to demonstrate num and denom values lead to 133.

end sum_numerator_denominator_l793_793092


namespace moles_of_water_formed_l793_793750

theorem moles_of_water_formed {NH4NO3 NaOH : ℕ} (h1 : NH4NO3 = 2) (h2 : NaOH = 2) : (NH4NO3, NaOH) = (H2O) :=
by
  let balancedEq := (1, 1) |-> (1, 1)
  have step1 : balancedEq = ((2, 2) |-> (2))
  sorry

end moles_of_water_formed_l793_793750


namespace probability_event_interval_result_l793_793675

noncomputable def probability_event_interval (x: ℝ) : Prop := 0 ≤ x ∧ x ≤ 3 / 2

theorem probability_event_interval_result :
  let interval := set.Icc (0 : ℝ) (2 : ℝ) in
  let event_interval := set.Icc (0 : ℝ) (3/2 : ℝ) in
  (set.volume event_interval / set.volume interval) = 3 / 4 :=
by 
  sorry

end probability_event_interval_result_l793_793675


namespace max_edges_intersected_by_plane_l793_793056

-- Definitions and conditions
variables (P : Type) [convex_polyhedron P] (edges : P → ℕ) (intersects : plane → P → ℕ)
variable (plane : Type)

-- Given: The polyhedron has 99 edges
axiom polyhedron_has_99_edges : edges P = 99

-- Plane does not pass through any vertices
axiom plane_does_not_intersect_vertices : ∀ (p : plane) (v : vertex P), ¬intersects p v

-- We are required to prove that the greatest number of edges the plane can intersect is 66
theorem max_edges_intersected_by_plane : ∃ p : plane, intersects p P ≤ 66 :=
sorry

end max_edges_intersected_by_plane_l793_793056


namespace problem1_problem2_l793_793717

-- Problem (1) proof statement
theorem problem1 (a : ℝ) (h : a ≠ 0) : 
  3 * a^2 * a^3 + a^7 / a^2 = 4 * a^5 :=
by
  sorry

-- Problem (2) proof statement
theorem problem2 (x : ℝ) : 
  (x - 1)^2 - x * (x + 1) + (-2023)^0 = -3 * x + 2 :=
by
  sorry

end problem1_problem2_l793_793717


namespace line_parallel_to_skew_lines_condition_l793_793159

theorem line_parallel_to_skew_lines_condition {l1 l2 l3 : Type} [line l1] [line l2] [line l3] 
  (skew_l1_l2 : skew l1 l2) (parallel_l3_l1 : parallel l3 l1) : 
  intersect l3 l2 ∨ skew l3 l2 :=
sorry

end line_parallel_to_skew_lines_condition_l793_793159


namespace investment_rate_l793_793694

theorem investment_rate
  (I_total I1 I2 : ℝ)
  (r1 r2 : ℝ) :
  I_total = 12000 →
  I1 = 5000 →
  I2 = 4500 →
  r1 = 0.035 →
  r2 = 0.045 →
  ∃ r3 : ℝ, (I1 * r1 + I2 * r2 + (I_total - I1 - I2) * r3) = 600 ∧ r3 = 0.089 :=
by
  intro hI_total hI1 hI2 hr1 hr2
  sorry

end investment_rate_l793_793694


namespace extremum_value_when_a_is_negative_one_monotonicity_intervals_one_zero_value_of_a_l793_793793

-- Part 1: Prove the maximum value of f when a = -1
theorem extremum_value_when_a_is_negative_one (x : ℝ) : ∃ a : ℝ, a = -1 ∧ 
  (f x = -x^2 + x + ln x) → (∃ c : ℝ, (c = 1 ∧ f c = 0)) :=
begin
  sorry
end

-- Part 2: Prove the monotonicity intervals for different values of a
theorem monotonicity_intervals (a x : ℝ) : 
  ∀ f : ℝ → ℝ, 
    (f x = ax^2 + (2 - a^2)x - a * ln x) →
    (if a < 0 then 
      (∀ x, ∃ l u : ℝ, (0 < x < -1/a) → ∀ x, x ∈ l ∧ x ∈ u → increasing f x) ∧ 
      (∀ x, ∃ l u : ℝ, (-1/a < x < +∞) → ∀ x, x ∈ l ∧ x ∈ u → decreasing f x) else 
    if a = 0 then 
      (∀ x, (0 < x < +∞) → increasing f x) else 
    if a > 0 then 
      (∀ x, ∃ l u : ℝ, (0 < x < a/2) → ∀ x, x ∈ l ∧ x ∈ u → decreasing f x) ∧ 
      (∀ x, ∃ l u : ℝ, (a/2 < x < +∞) → ∀ x, x ∈ l ∧ x ∈ u → increasing f x)) :=
begin
  sorry
end

-- Part 3: Prove the values of a for which f has exactly one zero
theorem one_zero_value_of_a (a x : ℝ) : 
  (f x = ax^2 + (2 - a^2)x - a * ln x) → 
  (∃! x, f x = 0) ↔ (a = -1 ∨ a = 2) :=
begin
  sorry
end

end extremum_value_when_a_is_negative_one_monotonicity_intervals_one_zero_value_of_a_l793_793793


namespace arctan_sum_l793_793197

theorem arctan_sum (a b : ℝ) (h1 : a = 3) (h2 : b = 7) : 
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 := 
by 
  rw [h1, h2]
  sorry

end arctan_sum_l793_793197


namespace sin_35pi_over_6_l793_793648

theorem sin_35pi_over_6 : Real.sin (35 * Real.pi / 6) = -1 / 2 := by
  sorry

end sin_35pi_over_6_l793_793648


namespace probability_four_of_five_same_value_l793_793242

-- Define the conditions
def standard_six_sided_dice := {1, 2, 3, 4, 5, 6}
def initial_roll_condition (dices : Fin 5 → ℕ) : Prop :=
  ∃ num : ℕ, num ∈ standard_six_sided_dice ∧
             (∃ triplet : Finset (Fin 5), triplet.card = 3 ∧
               ∀ i ∈ triplet, dices i = num) ∧
             ∀ k : Fin 5, dices k ≠ num → ∃ j, dices j ≠ dices k

-- Define the probability space and the required outcome
noncomputable def probability_of_at_least_four_same_value :=
  @prob_set (Finset (Finset (Fin 6))) _ (λ outcomes, ∃ num ∈ standard_six_sided_dice, 
    filter (λ d, d = num) outcomes  ≥ 4)

-- State the theorem
theorem probability_four_of_five_same_value (dices : Fin 5 → ℕ) :
  initial_roll_condition dices →
  probability_of_at_least_four_same_value = 1/36 := by 
sorry

end probability_four_of_five_same_value_l793_793242


namespace problem1_problem2_problem3_l793_793312

-- Define the function f(x)
def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + a^2 - 1

-- 1. Prove that if f(1) = 3 then a = -1 or a = 3
theorem problem1 (a : ℝ) : f a 1 = 3 → (a = -1 ∨ a = 3) := sorry

-- 2. Prove that f(x) is monotonic in [0,2] implies a ≤ 0 or a ≥ 2
theorem problem2 (a : ℝ) : monotone_on (f a) (set.Icc 0 2) → (a ≤ 0 ∨ a ≥ 2) := sorry

-- 3. Find the minimum value of f(x) when x ∈ [-1,1] and define g(a)
def g (a : ℝ) : ℝ :=
  if a ≤ -1 then a^2 + 2 * a
  else if -1 < a ∧ a < 1 then -1
  else a^2 - 2 * a

-- Define g(a)
theorem problem3 (a : ℝ) (x : ℝ) : x ∈ set.Icc (-1) 1 → ∃ m, (m = g a) ∧ ∀ y, y ∈ set.Icc (-1) 1 → f a y ≥ m := sorry

end problem1_problem2_problem3_l793_793312


namespace first_factor_of_lcm_l793_793558

theorem first_factor_of_lcm (A B lcm hcf : ℕ) (h1 : hcf = 25) 
                            (h2 : A = 400) (h3 : lcm = hcf * X * 16) 
                            (h4 : lcm = nat.lcm A B) (h5 : nat.gcd A B = hcf) :
                            X = 1 :=
by
  -- Here we'd normally do the proof, but we'll skip it with sorry
  sorry

end first_factor_of_lcm_l793_793558


namespace part1_part2_l793_793314

noncomputable theory

-- Definitions and conditions
def f (x : ℝ) (a : ℝ) : ℝ := Math.sin (x + π / 6) + Math.sin (x - π / 6) + Math.cos x + a
def g (x : ℝ) : ℝ := Math.sin (2 * x + π / 6)

-- Problem statements in Lean 4
theorem part1 (a : ℝ) : (∀ x, f x a ≤ 1) → (a = -1) :=
by
  sorry

theorem part2 (A b c : ℝ) (aVal : ℝ) (angleA : ℝ) :
  (g A = 1 / 2) ∧ (aVal = 2) ∧ (angleA = π / 3) ∧ (Math.sin angleA = sqrt 3 / 2) →
  (let area := (sqrt 3) / 4 * b * c in area ≤ sqrt 3 ∧ area = sqrt 3) :=
by
  sorry

end part1_part2_l793_793314


namespace max_words_formula_l793_793552

-- Define a convex n-gon with distinct vertices
def convex_ngon (n : ℕ) := 
  {vertices : fin n → ℝ × ℝ // 
    ∀ i j k, (vertices i).fst ≠ (vertices j).fst ∧ 
             (vertices j).fst ≠ (vertices k).fst ∧
             (vertices k).fst ≠ (vertices i).fst ∧
             -- Convexity condition
             vertices i ≠ vertices j → 
             vertices j ≠ vertices k → 
             (vertices i).fst < (vertices j).fst < (vertices k).fst → 
             vertices i ≠ vertices k}

-- Maximum number of distinct n-letter words readable
def max_distinct_words (n : ℕ) (polygon : convex_ngon n) : ℕ :=
  1 / 12 * n * (n - 1) * (n ^ 2 - 5 * n + 18)

-- Problem statement
theorem max_words_formula (n : ℕ) (polygon : convex_ngon n) : 
  max_distinct_words n polygon = 1 / 12 * n * (n - 1) * (n ^ 2 - 5 * n + 18) :=
sorry

end max_words_formula_l793_793552


namespace digit_in_1173rd_place_of_fraction_8_17_l793_793631

theorem digit_in_1173rd_place_of_fraction_8_17 : 
  (decimal_representation (8/17)).digit 1173 = 2 :=
by
  -- Proof will go here
  sorry

end digit_in_1173rd_place_of_fraction_8_17_l793_793631


namespace mother_gave_dimes_l793_793515

noncomputable def initial_dimes : ℕ := 7
noncomputable def dad_dimes : ℕ := 8
noncomputable def final_dimes : ℕ := 19

theorem mother_gave_dimes : ∃ m : ℕ, m = final_dimes - (initial_dimes + dad_dimes) := by
  use (final_dimes - (initial_dimes + dad_dimes))
  simp [final_dimes, initial_dimes, dad_dimes]
  exact rfl

end mother_gave_dimes_l793_793515


namespace total_luggage_l793_793928

theorem total_luggage (ne nb nf : ℕ)
  (leconomy lbusiness lfirst : ℕ)
  (Heconomy : ne = 10) 
  (Hbusiness : nb = 7) 
  (Hfirst : nf = 3)
  (Heconomy_luggage : leconomy = 5)
  (Hbusiness_luggage : lbusiness = 8)
  (Hfirst_luggage : lfirst = 12) : 
  (ne * leconomy + nb * lbusiness + nf * lfirst) = 142 :=
by
  sorry

end total_luggage_l793_793928


namespace find_number_l793_793559

theorem find_number : 
  ∃ N : ℝ, ∀ (a b c d : ℕ), 
    (nat.lcm a (nat.lcm b (nat.lcm c d)) = 60) → 
    N + 11.000000000000014 = 60 ∧ 
    (a = 5 ∧ b = 6 ∧ c = 4 ∧ d = 3) → 
    N = 49 :=
by {
  sorry
}

end find_number_l793_793559


namespace max_ab_value_l793_793761

theorem max_ab_value (a b : ℝ) (h : ∀ x : ℝ, exp (x + 1) ≥ a * x + b) : a * b ≤ (exp 3) / 2 :=
sorry

end max_ab_value_l793_793761


namespace triangle_area_of_lines_l793_793620

theorem triangle_area_of_lines :
  let L1 (x : ℝ) := 3 * x - 6
  let L2 (x : ℝ) := -2 * x + 12
  let intersection := (18 / 5, 24 / 5)
  let y_intercept_L1 := (0, -6)
  let y_intercept_L2 := (0, 12)
  let base := 12 - (-6)
  let height := 18 / 5
  area(triangle(y_intercept_L1, y_intercept_L2, intersection)) = (1 / 2) * base * height :=
by
  sorry

end triangle_area_of_lines_l793_793620


namespace smallest_value_y_eq_8_l793_793753

theorem smallest_value_y_eq_8 :
  let y := 8 in
  (min (5 / (y - 1))
       (min (5 / (y + 1))
            (min (5 / y)
                 (min ((5 + y) / 10)
                      (y - 5))))) = 5 / 9 :=
by
  sorry

end smallest_value_y_eq_8_l793_793753


namespace triangle_inequality_l793_793938

variable {a b c : ℝ}

theorem triangle_inequality (h₁ : a + b + c = 2) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) (h₅ : a + b > c) (h₆ : a + c > b) (h₇ : b + c > a) : 
  a^2 + b^2 + c^2 < 2 * (1 - a * b * c) :=
by
  sorry

end triangle_inequality_l793_793938


namespace two_digit_numbers_less_than_35_l793_793363

theorem two_digit_numbers_less_than_35 : 
  ∃ n, n ∈ finset.range 90 ∧ (10 ≤ n + 10 ∧ n + 10 ≤ 99) ∧ 
  (let tens_digit := (n + 10) / 10 in 
   let units_digit := (n + 10) % 10 in 
   tens_digit < 3 ∨ units_digit < 5) → 
  nat.card {n | n ∈ finset.range 90 ∧ (10 ≤ n + 10 ∧ n + 10 ≤ 99) ∧ 
                    (let tens_digit := (n + 10) / 10 in 
                     let units_digit := (n + 10) % 10 in 
                     tens_digit < 3 ∨ units_digit < 5)} = 55
:= sorry

end two_digit_numbers_less_than_35_l793_793363


namespace find_y_value_l793_793416

-- Define the given conditions and the final question in Lean
theorem find_y_value (k : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : y = k * x ^ (1/3)) 
  (h2 : y = 4 * real.sqrt 3)
  (x1 : x = 64) 
  : ∃ k, y = 2 * real.sqrt 3 :=
sorry

end find_y_value_l793_793416


namespace value_of_other_number_l793_793399

theorem value_of_other_number (k : ℕ) (other_number : ℕ) (h1 : k = 2) (h2 : (5 + k) * (5 - k) = 5^2 - other_number) : other_number = 21 :=
  sorry

end value_of_other_number_l793_793399


namespace final_condition_met_l793_793686

-- Definitions representing the graph and degrees
structure Graph (V : Type) :=
(edges : V → V → Prop)
(symmetric : ∀ {v₁ v₂ : V}, edges v₁ v₂ → edges v₂ v₁)
(antisymmetric : ∀ {v₁ v₂ : V}, edges v₁ v₂ → v₁ ≠ v₂)

-- Vertex set
def V := Fin 2019

-- Social network represented as a graph
def socialNetwork : Graph V :=
{ edges := sorry,  -- initial conditions on edges
  symmetric := sorry,  -- adjacency is symmetric
  antisymmetric := sorry  -- no loops }

-- Condition on the degrees of the vertices
def degrees (G : Graph V) (v : V) : ℕ :=
Finset.card { w : Fin 2019 | G.edges v w }

def initialCondition (G : Graph V) : Prop :=
(Finset.card { v : V | degrees G v = 1009 } = 1010) ∧
(Finset.card { v : V | degrees G v = 1010 } = 1009)

-- Theorem stating the final condition
theorem final_condition_met (G : Graph V) (h : initialCondition G) :
  ∃ seq : List (Graph V), (∀ G' ∈ seq, (λ (G₁ G₂ : Graph V), ∃ A B C : V, G₁.edges A B ∧ G₁.edges A C ∧ ¬ G₁.edges B C ∧ G₂.edges B C ∧ ¬ G₂.edges A B ∧ ¬ G₂.edges A C) G G') ∧ (∀ v, degrees (List.getLast seq sorry) v ≤ 1) :=
sorry

end final_condition_met_l793_793686


namespace sequence_sums_l793_793264

noncomputable def a (n : ℕ) : ℕ := 2 * n - 1
noncomputable def b (n : ℕ) : ℕ := 3^(n - 1)
noncomputable def log3 (x : ℕ) : ℕ := x - 1 -- Using n-1 directly as given in log3(b n) = n - 1

noncomputable def S (n : ℕ) : ℕ :=
(n * (n - 1)) / 2

theorem sequence_sums (n : ℕ) : 
  let Sₙ := (Finset.range n).sum (λ i, log3 (b (i + 1))) in
  Sₙ = S n :=
by {
  sorry
}

end sequence_sums_l793_793264


namespace cylinder_unoccupied_volume_l793_793990

theorem cylinder_unoccupied_volume (r h_cylinder h_cone : ℝ) 
  (h : r = 10 ∧ h_cylinder = 30 ∧ h_cone = 15) :
  (π * r^2 * h_cylinder - 2 * (1/3 * π * r^2 * h_cone) = 2000 * π) :=
by
  rcases h with ⟨rfl, rfl, rfl⟩
  simp
  sorry

end cylinder_unoccupied_volume_l793_793990


namespace fraction_of_calls_processed_by_team_B_l793_793636

theorem fraction_of_calls_processed_by_team_B
  (C_B : ℕ) -- the number of calls processed by each member of team B
  (B : ℕ)  -- the number of call center agents in team B
  (C_A : ℕ := C_B / 5) -- each member of team A processes 1/5 the number of calls as each member of team B
  (A : ℕ := 5 * B / 8) -- team A has 5/8 as many agents as team B
: 
  (B * C_B) / ((A * C_A) + (B * C_B)) = (8 / 9 : ℚ) :=
sorry

end fraction_of_calls_processed_by_team_B_l793_793636


namespace find_line_equation_l793_793293

def ellipse (x y : ℝ) : Prop := (x ^ 2) / 6 + (y ^ 2) / 3 = 1

def meets_first_quadrant (l : Line) : Prop :=
  ∃ A B : Point, ellipse A.x A.y ∧ ellipse B.x B.y ∧ 
  A.x > 0 ∧ A.y > 0 ∧ B.x > 0 ∧ B.y > 0 ∧ l.contains A ∧ l.contains B

def intersects_axes (l : Line) : Prop :=
  ∃ M N : Point, M.y = 0 ∧ N.x = 0 ∧ l.contains M ∧ l.contains N
  
def equal_distances (M N A B : Point) : Prop :=
  dist M A = dist N B

def distance_MN (M N : Point) : Prop :=
  dist M N = 2 * Real.sqrt 3

theorem find_line_equation (l : Line) (A B M N : Point)
  (h1 : meets_first_quadrant l)
  (h2 : intersects_axes l)
  (h3 : equal_distances M N A B)
  (h4 : distance_MN M N) :
  l.equation = "x + sqrt(2) * y - 2 * sqrt(2) = 0" :=
sorry

end find_line_equation_l793_793293


namespace quadratic_roots_ratio_l793_793563

noncomputable def value_of_m (m : ℚ) : Prop :=
  ∃ r s : ℚ, r ≠ 0 ∧ s ≠ 0 ∧ (r / s = 3) ∧ (r + s = -9) ∧ (r * s = m)

theorem quadratic_roots_ratio (m : ℚ) (h : value_of_m m) : m = 243 / 16 :=
by
  sorry

end quadratic_roots_ratio_l793_793563


namespace computer_price_difference_l793_793064

-- Define the conditions as stated
def basic_computer_price := 1500
def total_price := 2500
def printer_price (P : ℕ) := basic_computer_price + P = total_price

def enhanced_computer_price (P E : ℕ) := P = (E + P) / 3

-- The theorem stating the proof problem
theorem computer_price_difference (P E : ℕ) 
  (h1 : printer_price P) 
  (h2 : enhanced_computer_price P E) : E - basic_computer_price = 500 :=
sorry

end computer_price_difference_l793_793064


namespace sum_of_fraction_numerator_denominator_l793_793103

theorem sum_of_fraction_numerator_denominator :
  let x := 0.343434...
  in let fraction := (34 / 99 : ℚ)
  in let sum := fraction.num + fraction.den 
  in (x : ℚ) = fraction ∧ fraction.isReduced → sum = 133 :=
by
  sorry

end sum_of_fraction_numerator_denominator_l793_793103


namespace degree_of_expanded_poly_l793_793621

noncomputable theory

def P1 : Polynomial ℤ := 3 * (Polynomial.X ^ 5) + 2 * (Polynomial.X ^ 4) - Polynomial.X + 7
def P2 : Polynomial ℤ := 4 * (Polynomial.X ^ 11) - 5 * (Polynomial.X ^ 8) + 2 * (Polynomial.X ^ 5) + 15
def P3 : Polynomial ℤ := (Polynomial.X ^ 3 + 4) ^ 6

theorem degree_of_expanded_poly : (P1 * P2 - P3).degree = 18 := 
by
  sorry

end degree_of_expanded_poly_l793_793621


namespace y_at_x8_l793_793425

theorem y_at_x8 (k : ℝ) (y : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, y x = k * x^(1/3))
  (h2 : y 64 = 4 * Real.sqrt 3) :
  y 8 = 2 * Real.sqrt 3 :=
by
  sorry

end y_at_x8_l793_793425


namespace correct_operation_is_a_l793_793129

theorem correct_operation_is_a (a b : ℝ) : 
  (a^4 * a^3 = a^7) ∧ 
  ((a^2)^3 ≠ a^5) ∧ 
  (3 * a^2 - a^2 ≠ 2) ∧ 
  ((a - b)^2 ≠ a^2 - b^2) := 
by {
  -- Here, you would fill in the proof
  sorry
}

end correct_operation_is_a_l793_793129


namespace greatest_50_supportive_X_correct_l793_793000

noncomputable def greatest_50_supportive_X (a : Fin 50 → ℝ) : ℝ :=
if ∃ X, ∀ a : Fin 50 → ℝ, (∑ i : Fin 50, a i).floor = ∑ i : Fin 50, a i ∧ 
  (∃ i, |a i - 0.5| >= X) 
then 0.01 else 0

theorem greatest_50_supportive_X_correct :
  ∀ a : Fin 50 → ℝ, (∑ i : Fin 50, a i).floor = ∑ i : Fin 50, a i → 
  (∃ i, |a i - 0.5| >= 0.01) :=
sorry

end greatest_50_supportive_X_correct_l793_793000


namespace div_by_73_l793_793017

theorem div_by_73 (n : ℕ) (h : 0 < n) : (2^(3*n + 6) + 3^(4*n + 2)) % 73 = 0 := sorry

end div_by_73_l793_793017


namespace polynomial_division_example_l793_793241

open Polynomial

theorem polynomial_division_example (x : ℝ) :
  (x^6 + 5 * x^3 - 8) /ₘ (X - C 2) = X^5 + 2 * X^4 + 4 * X^3 + 13 * X^2 + 26 * X + 52 ∧
  (x^6 + 5 * x^3 - 8) %ₘ (X - C 2) = 96 :=
by
  sorry

end polynomial_division_example_l793_793241


namespace y_at_x_equals_8_l793_793405

theorem y_at_x_equals_8 (k : ℝ) (h1 : ∀ x y, y = k * x^(1/3))
    (h2 : 4 * real.sqrt 3 = k * 64^(1/3)) : k * 8^(1/3) = 2 * real.sqrt 3 :=
by
  sorry

end y_at_x_equals_8_l793_793405


namespace correct_statements_about_f_l793_793249

def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - (13 / 4) * Real.pi)

theorem correct_statements_about_f :
  ((f (Real.pi / 8) = 0) ∧ (f (Real.pi / 8) = f (-(Real.pi / 8)))) ∧  -- Statement A
  (f(2 * (Real.pi / 8)) = 2 * Real.sin(2 * (Real.pi / 8) - (13 / 4) * Real.pi)) ∧  -- Statement B
  (f (2 * (Real.pi / 8) + 5/8 * Real.pi) = 2 * Real.sin (2 * (Real.pi / 8 + 5/8 * Real.pi) - 13 / 4 * Real.pi)) ∧ -- Statement C
  (f (2 * (Real.pi / 8) - 5/8 * Real.pi) = 2 * Real.sin (2 * (Real.pi / 8 - 5/8 * Real.pi) - 13 / 4 * Real.pi))  -- Statement D
:= sorry

end correct_statements_about_f_l793_793249


namespace conic_locus_of_N_l793_793796

variable {A B C D E F x0 y0 x' y' : ℝ}

-- Definitions of the coefficients
def a : ℝ := (B^2 - A*C) * y0^2 + 2 * (B*D - A*E) * y0 + D^2 - A*F
def b : ℝ := (A*C - B^2) * x0 * y0 + (A*E - B*D) * x0 + (C*D - B*E) * y0 + D*E - B*F
def c : ℝ := (B^2 - A*C) * x0^2 + 2 * (B*E - C*D) * x0 + E^2 - C*F
def d : ℝ := (A*E - B*D) * x0 * y0 + (B*E - C*D) * y0^2 + A*F - D^2 * x0 + B*F - D*E * y0
def e : ℝ := (B*D - A*E) * x0^2 + (C*D - B*E) * x0 * y0 + B*F - D*E * x0 + C*F - E^2 * y0
def f : ℝ := D^2 - A*F * x0^2 + 2 * (D*E - B*F) * x0 * y0 + E^2 - C*F * y0^2

theorem conic_locus_of_N :
  a * x' * x' + 2 * b * x' * y' + c * y' * y' + 2 * d * x' + 2 * e * y' + f = 0 := 
sorry

end conic_locus_of_N_l793_793796


namespace sum_of_roots_of_quadratic_l793_793625

theorem sum_of_roots_of_quadratic (x : ℝ) :
  (x^2 = 16 * x - 10) → ∑ x, x = 16 :=
by sorry

end sum_of_roots_of_quadratic_l793_793625


namespace angle_bisector_of_BKH_l793_793455

variable (a : ℝ) (H K E : EuclideanGeometry.Point)

def InSquare (p : EuclideanGeometry.Point) : Prop :=
  p.x ≥ 0 ∧ p.x ≤ a ∧ p.y ≥ 0 ∧ p.y ≤ a

def MidpointC_D (H : EuclideanGeometry.Point) : Prop :=
  H.x = a / 2 ∧ H.y = 0

def OnSideB_C (K : EuclideanGeometry.Point) : Prop :=
  K.x = a ∧ 0 ≤ K.y ∧ K.y ≤ a

def KCondition (K : EuclideanGeometry.Point) : Prop :=
  K.y = 2 * (a - K.y)

def ExtendBeyondB (E : EuclideanGeometry.Point) : Prop :=
  E.x = a ∧ E.y = a / 2

theorem angle_bisector_of_BKH
  (Hpt : MidpointC_D H)
  (Kpt : OnSideB_C K ∧ KCondition K)
  (Eext : ExtendBeyondB E)
  : EuclideanGeometry.is_angle_bisector K A B H :=
sorry

end angle_bisector_of_BKH_l793_793455


namespace min_chips_to_A10_l793_793997

theorem min_chips_to_A10 (n : ℕ) (A : ℕ → ℕ) (hA1 : A 1 = n) :
  (∃ (σ : ℕ → ℕ), 
    (∀ i, 1 ≤ i ∧ i < 10 → (σ i = A i - 2) ∧ (σ (i + 1) = A (i + 1) + 1)) ∨ 
    (∀ i, 1 ≤ i ∧ i < 9 → (σ (i + 1) = A (i + 1) - 2) ∧ (σ (i + 2) = A (i + 2) + 1) ∧ (σ i = A i + 1)) ∧ 
    (∃ (k : ℕ), k = 10 ∧ σ k = 1)) →
  n ≥ 46 := sorry

end min_chips_to_A10_l793_793997


namespace max_value_of_g_is_34_l793_793047
noncomputable def g : ℕ → ℕ
| n => if n < 15 then n + 20 else g (n - 7)

theorem max_value_of_g_is_34 : ∃ n, g n = 34 ∧ ∀ m, g m ≤ 34 :=
by
  sorry

end max_value_of_g_is_34_l793_793047


namespace smallest_positive_period_intervals_of_monotonic_increase_find_a_value_l793_793792

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := sin (2 * x + π / 6) + sin (2 * x - π / 6) + cos (2 * x) + a

theorem smallest_positive_period (a : ℝ) : ∃ (T : ℝ), T = π ∧ ∀ x : ℝ, f (x + T) a = f x a :=
by
  sorry

theorem intervals_of_monotonic_increase (a : ℝ) : ∃ k : ℤ, ∀ x : ℝ, (k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6) := 
by
  sorry

theorem find_a_value (a : ℝ) : (∃ x ∈ Icc (0 : ℝ) (π / 2) , f x a = -2) → a = -1 :=
by
  sorry

end smallest_positive_period_intervals_of_monotonic_increase_find_a_value_l793_793792


namespace lambda_range_l793_793262

noncomputable def a_seq : ℕ → ℝ
| 0       := 1
| (n + 1) := (a_seq n) / (a_seq n + 2)

noncomputable def b_seq (λ : ℝ) : ℕ → ℝ
| 0       := -λ
| (n + 1) := (n - λ) * (1 / a_seq n + 1)

theorem lambda_range (λ : ℝ) : (∀ n : ℕ, b_seq λ (n + 1) > b_seq λ n) ↔ (λ < 2) :=
  sorry

end lambda_range_l793_793262


namespace find_a_b_c_T_a_l793_793263

noncomputable def S (n : ℕ) : ℕ := 2 * a n - 1
noncomputable def b (n : ℕ) : ℕ := if n = 1 then 1 else (n - 1) * b (n - 1) + n * (n - 1)
noncomputable def c (n : ℕ) : ℕ := a n * Int.to_nat (Int.sqrt (b n))
noncomputable def T (n : ℕ) : ℕ := T_rec n

theorem find_a_b_c_T_a :
  (∀ n : ℕ, a (n + 1) = 2 * a n) ∧ 
  (∀ n : ℕ, b (n + 1) = (n + 1)^2) ∧ 
  (∀ n : ℕ, c n = 2^(n-1) * n) ∧ 
  (∀ n : ℕ, T n = (n - 1) * 2^n + 1) ∧ 
  (∀ n : ℕ, T n ≤ n * S n - a) → 
    a ≤ 0 :=
  sorry

end find_a_b_c_T_a_l793_793263


namespace root_equality_l793_793506

theorem root_equality (p q : ℝ) (h1 : 1 + p + q = (2 - 2 * q) / p) (h2 : 1 + p + q = (1 - p + q) / q) :
  p + q = 1 :=
sorry

end root_equality_l793_793506


namespace atomic_diameter_scientific_notation_l793_793043

theorem atomic_diameter_scientific_notation :
  ∀ (d : ℝ), d = 0.000000196 → (d * 0.001 = 2.0 * 10^(-10)) := 
by
  intros d h
  -- Proof goes here.
  sorry

end atomic_diameter_scientific_notation_l793_793043


namespace least_diff_geometric_arithmetic_l793_793943

noncomputable def geometricSeqA : List ℕ :=
  List.map (λ n => 3^n) ([0, 1, 2, 3, 4].filter (λ n => 3^n < 300))

noncomputable def arithmeticSeqB : List ℕ :=
  List.filter (λ k => 100 + 10*k < 300) (List.range 20)

theorem least_diff_geometric_arithmetic :
  ∃ (x ∈ geometricSeqA) (y ∈ arithmeticSeqB), (∀a ∈ geometricSeqA, ∀b ∈ arithmeticSeqB, abs (a - b) ≥ abs (243 - 240)) ∧ abs (243 - 240) = 3 :=
by
  sorry

end least_diff_geometric_arithmetic_l793_793943


namespace count_polynomials_degree_le_3_with_condition_l793_793215

def polynomial_solutions_eq_286 : Prop :=
  (finset.univ.filter (λ p : (fin 10) × (fin 10) × (fin 10) × (fin 10), 
    p.1.1 + p.1.2 + p.2.1 + p.2.2 = 10)).card = 286

theorem count_polynomials_degree_le_3_with_condition : polynomial_solutions_eq_286 :=
by sorry

end count_polynomials_degree_le_3_with_condition_l793_793215


namespace arctan_sum_l793_793193

theorem arctan_sum : ∀ (a b : ℝ), 
  a = 3/7 → 
  b = 7/3 → 
  a * b = 1 → 
  a > 0 → 
  b > 0 → 
  Real.arctan a + Real.arctan b = Real.pi / 2 :=
by intros a b ha hb hab ha_pos hb_pos
   rw [ha, hb, hab]
   sorry

end arctan_sum_l793_793193


namespace probability_two_8sided_dice_sum_perfect_square_l793_793608

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def probabilityPerfectSquareSum (dice_sides : ℕ) (perfect_squares : List ℕ) : ℚ :=
  let outcomes := (dice_sides * dice_sides)
  let favorable_outcomes := perfect_squares.sum (λ ps, 
    (List.filter (λ (pair : ℕ × ℕ), pair.fst + pair.snd = ps) ((List.range dice_sides).product (List.range dice_sides))).length)
  favorable_outcomes /. outcomes

theorem probability_two_8sided_dice_sum_perfect_square :
  probabilityPerfectSquareSum 8 [4, 9, 16] = 3 / 16 := sorry

end probability_two_8sided_dice_sum_perfect_square_l793_793608


namespace tens_digit_of_power_l793_793091

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

theorem tens_digit_of_power (n : ℕ) (h : n % 4 = 3) :
  (last_two_digits (8 ^ n) / 10) % 10 = 1 :=
by
  sorry

-- Using the conditions that the last two digits of 8^1 = 08, 8^2 = 64, 8^3 = 12, and 8^4 = 96,
-- and knowing the pattern repeats every 4, we aim to prove the tens digit of 8^23 is 1.
example : tens_digit_of_power 23 (by norm_num) := by
  simp [tens_digit_of_power, last_two_digits]

end tens_digit_of_power_l793_793091


namespace least_odd_prime_factor_of_2023_8_plus_1_l793_793731

theorem least_odd_prime_factor_of_2023_8_plus_1 :
  ∃ p : ℕ, Prime p ∧ p ≡ 1 [MOD 16] ∧ 2023^8 ≡ -1 [MOD p] ∧ (∀ q : ℕ, Prime q ∧ q < p → ¬ (q ≡ 1 [MOD 16] ∧ 2023^8 ≡ -1 [MOD q])) ∧ p = 97 :=
begin
  sorry
end

end least_odd_prime_factor_of_2023_8_plus_1_l793_793731


namespace y_at_x_equals_8_l793_793403

theorem y_at_x_equals_8 (k : ℝ) (h1 : ∀ x y, y = k * x^(1/3))
    (h2 : 4 * real.sqrt 3 = k * 64^(1/3)) : k * 8^(1/3) = 2 * real.sqrt 3 :=
by
  sorry

end y_at_x_equals_8_l793_793403


namespace minimum_crossings_required_l793_793588

-- Definitions representing the problem conditions.
def Person : Type := { a : Bool // a = true ∧ a = false } -- Define person type as either adult (true) or child (false)
def isAdult : Person → Bool := λ p => p.val
def isChild : Person → Bool := λ p => !p.val

def raftCapacity (p1 p2 : Person) : Bool := 
  if isAdult p1 then
    p2 = p1  -- only one adult can be on the raft
  else
    if isChild p1 ∧ isChild p2 then true else false -- two children can be on the raft

-- Problem statement as a theorem.
theorem minimum_crossings_required 
  (A1 A2 : Person) (hA1 : isAdult A1) (hA2 : isAdult A2) 
  (C1 C2 : Person) (hC1 : isChild C1) (hC2 : isChild C2) 
  (crossings : ℕ → List Person) -- sequence of crossings
  (valid_crossing : ∀ i, raftCapacity (crossings i).head (crossings i).tail.head) : 
  ∃ n : ℕ, n = 9 ∧ (List.length crossings) = n :=
sorry

end minimum_crossings_required_l793_793588


namespace positive_three_digit_integers_l793_793337

theorem positive_three_digit_integers (n : ℕ) :
  (n = {m : ℕ | ∃ (h : m ≥ 100 ∧ m < 1000) (d1 d2 : ℕ),
    m = 100 * d1 + 10 * d2 ∧ 3 < d1 ∧ 3 < d2 ∧ d1 < 10 ∧ d2 < 10 ∧ m % 10 = 0}.to_finset.card) →
  n = 36 :=
by
  sorry

end positive_three_digit_integers_l793_793337


namespace two_digit_numbers_less_than_35_l793_793360

theorem two_digit_numbers_less_than_35 : 
  ∃ n, n ∈ finset.range 90 ∧ (10 ≤ n + 10 ∧ n + 10 ≤ 99) ∧ 
  (let tens_digit := (n + 10) / 10 in 
   let units_digit := (n + 10) % 10 in 
   tens_digit < 3 ∨ units_digit < 5) → 
  nat.card {n | n ∈ finset.range 90 ∧ (10 ≤ n + 10 ∧ n + 10 ≤ 99) ∧ 
                    (let tens_digit := (n + 10) / 10 in 
                     let units_digit := (n + 10) % 10 in 
                     tens_digit < 3 ∨ units_digit < 5)} = 55
:= sorry

end two_digit_numbers_less_than_35_l793_793360


namespace cows_on_farm_l793_793522

theorem cows_on_farm (weekly_production_per_6_cows : ℕ) 
                     (production_over_5_weeks : ℕ) 
                     (number_of_weeks : ℕ) 
                     (cows : ℕ) :
  weekly_production_per_6_cows = 108 →
  production_over_5_weeks = 2160 →
  number_of_weeks = 5 →
  (cows * (weekly_production_per_6_cows / 6) * number_of_weeks = production_over_5_weeks) →
  cows = 24 :=
by
  intros h1 h2 h3 h4
  sorry

end cows_on_farm_l793_793522


namespace part1_fifth_numbers_part2_three_adjacent_sum_part3_difference_largest_smallest_l793_793008

-- Definitions for the sequences
def first_row (n : ℕ) : ℤ := (-3) ^ n
def second_row (n : ℕ) : ℤ := (-3) ^ n - 3
def third_row (n : ℕ) : ℤ := -((-3) ^ n) - 1

-- Statement for part 1
theorem part1_fifth_numbers:
  first_row 5 = -243 ∧ second_row 5 = -246 ∧ third_row 5 = 242 := sorry

-- Statement for part 2
theorem part2_three_adjacent_sum :
  ∃ n : ℕ, first_row (n-1) + first_row n + first_row (n+1) = -1701 ∧
           first_row (n-1) = -243 ∧ first_row n = 729 ∧ first_row (n+1) = -2187 := sorry

-- Statement for part 3
def sum_nth (n : ℕ) : ℤ := first_row n + second_row n + third_row n
theorem part3_difference_largest_smallest (n : ℕ) (m : ℤ) (hn : sum_nth n = m) :
  (∃ diff, (n % 2 = 1 → diff = -2 * m - 6) ∧ (n % 2 = 0 → diff = 2 * m + 9)) := sorry

end part1_fifth_numbers_part2_three_adjacent_sum_part3_difference_largest_smallest_l793_793008


namespace solve_inequality_1_find_range_of_a_l793_793764

def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

theorem solve_inequality_1 :
  {x : ℝ | f x ≥ 5} = {x : ℝ | x ≤ -3} ∪ {x : ℝ | x ≥ 2} :=
by
  sorry
  
theorem find_range_of_a (a : ℝ) :
  (∀ x : ℝ, f x > a^2 - 2 * a - 5) ↔ -2 < a ∧ a < 4 :=
by
  sorry

end solve_inequality_1_find_range_of_a_l793_793764


namespace sorting_five_rounds_l793_793081

def direct_sorting_method (l : List ℕ) : List ℕ := sorry

theorem sorting_five_rounds (initial_seq : List ℕ) :
  initial_seq = [49, 38, 65, 97, 76, 13, 27] →
  (direct_sorting_method ∘ direct_sorting_method ∘ direct_sorting_method ∘ direct_sorting_method ∘ direct_sorting_method) initial_seq = [97, 76, 65, 49, 38, 13, 27] :=
by
  intros h
  sorry

end sorting_five_rounds_l793_793081


namespace Louis_ate_whole_boxes_l793_793488

def package_size := 6
def total_lemon_heads := 54

def whole_boxes : ℕ := total_lemon_heads / package_size

theorem Louis_ate_whole_boxes :
  whole_boxes = 9 :=
by
  sorry

end Louis_ate_whole_boxes_l793_793488


namespace joey_speed_return_l793_793486

/--
Joey the postman takes 1 hour to run a 5-mile-long route every day, delivering packages along the way.
On his return, he must climb a steep hill covering 3 miles and then navigate a rough, muddy terrain spanning 2 miles.
If the average speed of the entire round trip is 8 miles per hour, prove that the speed with which Joey returns along the path is 20 miles per hour.
-/
theorem joey_speed_return
  (dist_out : ℝ := 5)
  (time_out : ℝ := 1)
  (dist_hill : ℝ := 3)
  (dist_terrain : ℝ := 2)
  (avg_speed_round : ℝ := 8)
  (total_dist : ℝ := dist_out * 2)
  (total_time : ℝ := total_dist / avg_speed_round)
  (time_return : ℝ := total_time - time_out)
  (dist_return : ℝ := dist_hill + dist_terrain) :
  (dist_return / time_return = 20) := 
sorry

end joey_speed_return_l793_793486


namespace solution_to_system_l793_793949

noncomputable def solve_system_of_equations : Prop :=
  ∃ (x y : ℝ), 
    (4 * x + 3 * y = 6.4) ∧ 
    (5 * x - 6 * y = -1.5) ∧ 
    (x = 0.8692) ∧ 
    (y = 0.9744)

theorem solution_to_system : solve_system_of_equations :=
by 
  use 0.8692
  use 0.9744
  split
  { norm_num }
  split
  { norm_num }
  split
  { refl }
  { refl }

end solution_to_system_l793_793949


namespace geometry_problem_l793_793177

-- Definitions of the geometric points and conditions
variables {A B C D E P K L : Point}
noncomputable def BC := (Real.sqrt (AB * DE)).val
noncomputable def CD := (Real.sqrt (AB * DE)).val

-- Hypotheses for the problem setup
hypothesis h1 : collinear [A, B, C, D, E]
hypothesis h2 : BC = CD
hypothesis h3 : PB = PD
hypothesis h4 : K ∈ PB
hypothesis h5 : L ∈ PD
hypothesis h6 : bisects KC ∠ BKE
hypothesis h7 : bisects LC ∠ ALD

-- Goal: Prove that points A, K, L, and E are concyclic (i.e., lie on a common circle)
theorem geometry_problem : concyclic [A, K, L, E] :=
by sorry

end geometry_problem_l793_793177


namespace value_of_y_at_x8_l793_793411

theorem value_of_y_at_x8
  (k : ℝ)
  (y : ℝ → ℝ)
  (hx64 : y 64 = 4 * Real.sqrt 3)
  (hy_def : ∀ x, y x = k * x^(1 / 3)) :
  y 8 = 2 * Real.sqrt 3 :=
by {
  sorry,
}

end value_of_y_at_x8_l793_793411


namespace duration_of_period_l793_793861

def birth_rate : ℕ := 4
def death_rate : ℕ := 2
def net_increase : ℕ := 86400

theorem duration_of_period (birth_rate death_rate net_increase: ℕ) (h_birth: birth_rate = 4) (h_death: death_rate = 2) (h_net: net_increase = 86400) : net_increase / (birth_rate - death_rate) / 3600 = 12 :=
by
  rw [h_birth, h_death, h_net]
  norm_num
  sorry

end duration_of_period_l793_793861


namespace find_line_equation_l793_793297

theorem find_line_equation (x y : ℝ) : 
  (∃ A B, (A.x^2 / 6 + A.y^2 / 3 = 1) ∧ (B.x^2 / 6 + B.y^2 / 3 = 1) ∧
  (A.x > 0 ∧ A.y > 0) ∧ (B.x > 0 ∧ B.y > 0) ∧
  let M := (-B.y, 0) in
  let N := (0, B.y) in 
  (abs (M.x - A.x) = abs (N.y - B.y)) ∧ 
  (abs (M.x - N.x + M.y - N.y) = 2 * sqrt 3)) →
  x + sqrt 2 * y - 2 * sqrt 2 = 0 := 
sorry

end find_line_equation_l793_793297


namespace simplify_sqrt_diff_l793_793945

theorem simplify_sqrt_diff :
  (81 ^ (1 / 2 : ℝ)) - (144 ^ (1 / 2 : ℝ)) = -3 := 
by
  sorry

end simplify_sqrt_diff_l793_793945


namespace range_of_a_l793_793799

-- Definitions derived from conditions
def q (a : ℝ) : Prop := ∃! x : ℝ, x^2 + 2*x + 2*a > 0
def p (a : ℝ) : Prop := false -- As derived from the analysis

-- The equivalent Lean statement
theorem range_of_a (a : ℝ) : (¬q a ∨ ¬p a) → a ∈ Set.Ioo (-∞) (-1) ∪ Set.Ioo (0) (1) := 
by
  intro h
  sorry -- proof goes here

end range_of_a_l793_793799


namespace ordered_pairs_count_l793_793224

theorem ordered_pairs_count :
  {p : Nat × Nat // 0 < p.1 ∧ 0 < p.2 ∧ (6 / p.1 : Rat) + (3 / p.2 : Rat) = 1}.to_list.length = 6 :=
by
  sorry

end ordered_pairs_count_l793_793224


namespace minimum_sum_of_factors_l793_793900

theorem minimum_sum_of_factors (a b : ℤ) (h : a * b = 144) : a + b ≥ -145 :=
sorry

example : ∃ a b : ℤ, a * b = 144 ∧ a + b = -145 :=
begin
  use [-1, -144],
  split,
  { norm_num },
  { norm_num }
end

end minimum_sum_of_factors_l793_793900


namespace values_of_a_b_f_is_odd_inequality_solution_l793_793316

def f (a b x : ℝ) : ℝ := b - 4 / (a^x + 1)

theorem values_of_a_b (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f a b x ∧ f a b x ≤ 1) :
  (a = 1/3 ∧ b = 3) ∨ (a = 3 ∧ b = 2) :=
sorry

theorem f_is_odd (a b : ℝ) (h : a > 0 ∧ a ≠ 1 ∧ a * b ≠ 1)
  (h_vals : (a = 1/3 ∧ b = 3) ∨ (a = 3 ∧ b = 2))
  : ∀ x, f a b (-x) = -f a b x :=
sorry

theorem inequality_solution (a b : ℝ) (h : a > 0 ∧ a ≠ 1 ∧ a * b ≠ 1)
  (h_vals : (a = 1/3 ∧ b = 3) ∨ (a = 3 ∧ b = 2)) :
  ∀ x, 5/3 < x → f a b (2*x - 1) + f a b (x - 4) > 0 :=
sorry

end values_of_a_b_f_is_odd_inequality_solution_l793_793316


namespace problem_I_problem_II_l793_793768

-- Given conditions
def S (n : ℕ) : ℕ := (n * (n + 1)) / 2

def a (n : ℕ) : ℕ := 
  if n = 0 then 1 else S n - S (n - 1)

def b (n : ℕ) : ℕ := a n + a (n + 1)

def c (n : ℕ) : ℕ := n * 2^(n + 1)

-- Problem statements
theorem problem_I (n : ℕ) (h : n > 0) : b n = 2 * n + 1 :=
sorry

theorem problem_II (n : ℕ) (h : n > 0) : 
  (∑ i in Finset.range (n + 1), c i) = (n - 1) * 2^(n + 2) + 4 :=
sorry

end problem_I_problem_II_l793_793768


namespace printer_to_enhanced_total_ratio_l793_793979

-- Define the conditions as variables
variables {basic_computer_price printer_price enhanced_computer_price total_price_basic total_price_enhanced : ℝ}

-- Initialize the conditions based on the problem statement
def basic_computer_price : ℝ := 2000
def total_price_basic : ℝ := 2500
def enhanced_computer_price : ℝ := basic_computer_price + 500

-- The ratio proof problem
theorem printer_to_enhanced_total_ratio :
  (∃ (printer_price: ℝ),
  basic_computer_price + printer_price = total_price_basic ∧
  (printer_price / (enhanced_computer_price + printer_price)) = (1 / 6)) :=
begin
  use 500,
  split,
  { -- Prove basic_computer_price + printer_price = total_price_basic
    rw [basic_computer_price, total_price_basic],
    norm_num },
  { -- Prove the ratio equals 1 / 6
    norm_num,
    rw [enhanced_computer_price],
    norm_num }
end

end printer_to_enhanced_total_ratio_l793_793979


namespace log_5_of_0_04_l793_793737

theorem log_5_of_0_04 : log 5 0.04 = -2 := by
  sorry

end log_5_of_0_04_l793_793737


namespace h_of_one_correct_l793_793721

noncomputable def given_polynomials (p q r : ℝ) (h : polynomial ℝ) : 
  polynomial ℝ :=
x^3 + p * x^2 + q * x + r

theorem h_of_one_correct (p q r : ℝ) (h : polynomial ℝ) (hpqr : p < q ∧ q < r) 
  (roots_h : ∀ (s : ℝ), s ∈ h.roots ↔ s⁻¹ ^ 2 ∈ (given_polynomials p q r).roots) :
  h.eval 1 = (1 - p + q - r) * (1 - q + p - r) * (1 - r + p - q) / r ^ 2 :=
by
  sorry

end h_of_one_correct_l793_793721


namespace circumscribed_circle_and_area_of_triangle_l793_793153

theorem circumscribed_circle_and_area_of_triangle 
  (ABC : Type*) [triangle ABC] 
  (B C : tri_point ABC) 
  (isosceles : is_isosceles ABC B C)
  (acute : is_acute_angle_tri ABC)
  (circumscribed : has_circumscribed_circle ABC)
  (T_on_arc : is_midpoint_of_arc_not_containing B) 
  (distance_T_AC : dist_to_line T_on_arc.1 equals_three)
  (distance_T_BC : dist_to_line T_on_arc.2 equals_seven):
  radius_of_circumscribed_circle equals_nine ∧ area_of_triangle equals_forty_sqrt_five :=
sorry

end circumscribed_circle_and_area_of_triangle_l793_793153


namespace cyl_volume_l793_793554

noncomputable def volume_of_cylinder (D H : ℝ) : ℝ := 
  let r := D / 2
  π * r^2 * H

theorem cyl_volume (D H : ℝ) (hD : D = 14) (hH : H = 2) : 
  volume_of_cylinder D H ≈ 307.87622 :=
by
  sorry

end cyl_volume_l793_793554


namespace length_DE_l793_793462

-- Definition and given conditions
variables {A B C D E : Type}
variables [Point (Triangle ABC)]

-- Given BC = 40 and angle C = 45 degrees
constants (BC : Real) (angleC : Real)
constants (midpoint : BC → D) (perpendicular_bisector : BC → AC → E)
constant (triangle_CDE_454590 : Is454590Triangle C D E)

-- Definitions for points D and E
noncomputable def midpoint_of_BC (P: BC) : D :=
  midpoint P

noncomputable def intersection_perpendicular_bisector_AC (P: BC → AC) : E :=
  perpendicular_bisector P

-- Prove length of DE == 20
theorem length_DE : length DE = 20 := sorry

end length_DE_l793_793462


namespace repeating_decimal_sum_l793_793105

theorem repeating_decimal_sum (x : ℚ) (h : x = 34 / 999) : x.num + x.denom = 1033 := by 
  sorry

end repeating_decimal_sum_l793_793105


namespace solve_inequality1_solve_inequality2_l793_793542

-- Problem 1: Solve the inequality (1)
theorem solve_inequality1 (x : ℝ) (h : x ≠ -4) : 
  (2 - x) / (x + 4) ≤ 0 ↔ (x ≥ 2 ∨ x < -4) := sorry

-- Problem 2: Solve the inequality (2) for different cases of a
theorem solve_inequality2 (x a : ℝ) : 
  (x^2 - 3 * a * x + 2 * a^2 ≥ 0) ↔
  (if a > 0 then (x ≥ 2 * a ∨ x ≤ a) 
   else if a < 0 then (x ≥ a ∨ x ≤ 2 * a) 
   else true) := sorry

end solve_inequality1_solve_inequality2_l793_793542


namespace find_m_range_l793_793318

noncomputable theory

variable {g : ℝ → ℝ} {f : ℝ → ℝ} {m : ℝ}

-- Define the conditions
def condition1 (x : ℝ) : Prop := g(x) = 2 * g(1 / x)
def condition2 (x : ℝ) : Prop := x ∈ Icc 1 3 → g(x) = log x

-- Define the function f in terms of g and m
def f (x : ℝ) : ℝ := g(x) - m * x

-- Define the range of m such that f(x) has three distinct zeros in [1/3, 3]
def valid_m_range : Set ℝ := Ico (log 3 / 3) (1 / Real.exp 1)

-- Lean 4 statement
theorem find_m_range (h1 : ∀ x, condition1 x) (h2 : ∀ x, condition2 x) :
  (∃ a b c ∈ Icc (1/3) 3, ∀ x ∈ Icc (1/3) 3, f(x) = 0 → (x = a ∨ x = b ∨ x = c)) ↔ m ∈ valid_m_range :=
by sorry

end find_m_range_l793_793318


namespace sqrt5_minus_2_power_2023_mul_sqrt5_plus_2_power_2023_eq_one_l793_793714

-- Defining the terms and the theorem
theorem sqrt5_minus_2_power_2023_mul_sqrt5_plus_2_power_2023_eq_one :
  (Real.sqrt 5 - 2) ^ 2023 * (Real.sqrt 5 + 2) ^ 2023 = 1 := 
by
  sorry

end sqrt5_minus_2_power_2023_mul_sqrt5_plus_2_power_2023_eq_one_l793_793714


namespace complex_expression_equals_neg3_l793_793034

noncomputable def nonreal_root_of_x4_eq_1 : Type :=
{ζ : ℂ // ζ^4 = 1 ∧ ζ.im ≠ 0}

theorem complex_expression_equals_neg3 (ζ : nonreal_root_of_x4_eq_1) :
  (1 - ζ.val + ζ.val^3)^4 + (1 + ζ.val^2 - ζ.val^3)^4 = -3 :=
sorry

end complex_expression_equals_neg3_l793_793034


namespace two_digit_numbers_less_than_35_l793_793361

theorem two_digit_numbers_less_than_35 : 
  ∃ n, n ∈ finset.range 90 ∧ (10 ≤ n + 10 ∧ n + 10 ≤ 99) ∧ 
  (let tens_digit := (n + 10) / 10 in 
   let units_digit := (n + 10) % 10 in 
   tens_digit < 3 ∨ units_digit < 5) → 
  nat.card {n | n ∈ finset.range 90 ∧ (10 ≤ n + 10 ∧ n + 10 ≤ 99) ∧ 
                    (let tens_digit := (n + 10) / 10 in 
                     let units_digit := (n + 10) % 10 in 
                     tens_digit < 3 ∨ units_digit < 5)} = 55
:= sorry

end two_digit_numbers_less_than_35_l793_793361


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l793_793119

theorem sum_of_numerator_and_denominator_of_repeating_decimal (x : ℚ) (h : x = 34 / 99) : (x.den + x.num : ℤ) = 133 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l793_793119


namespace find_DE_length_l793_793467

noncomputable def triangle_DE_length (A B C D E : Type) (BC CD DE : ℝ) (C_angle : ℝ) : Prop :=
  BC = 40 ∧ C_angle = 45 ∧ CD = 20 ∧ DE = 20

theorem find_DE_length {A B C D E : Type} (BC CD DE : ℝ) (C_angle : ℝ) 
  (hBC : BC = 40) (hC_angle : C_angle = 45) (hCD : CD = 20) : DE = 20 :=
by {
  have hDE : DE = 20, sorry,
  exact hDE
}

end find_DE_length_l793_793467


namespace cos_inequality_l793_793144

theorem cos_inequality {θ : ℝ} (h1 : θ ∈ Icc (0 : ℝ) π) :
  cos (2 * θ - (π / 6)) < (√3 / 2) ↔ (π / 6) < θ ∧ θ < π :=
begin
  sorry
end

end cos_inequality_l793_793144


namespace arc_length_eq_l793_793142

-- Definitions for x(t) and y(t) in the given parametric form
def x (t : ℝ) := (t^2 - 2) * sin t + 2 * t * cos t
def y (t : ℝ) := (2 - t^2) * cos t + 2 * t * sin t

-- Defining x'(t) and y'(t)
def dx_dt (t : ℝ) := ((differentiable_at ℝ (λ t, (t^2 - 2) * sin t + 2 * t * cos t))).deriv t
def dy_dt (t : ℝ) := ((differentiable_at ℝ (λ t, (2 - t^2) * cos t + 2 * t * sin t))).deriv t

-- Arc length formula in the specified interval
noncomputable def arc_length := ∫ t in (0 : ℝ)..π, (sqrt ((dx_dt t)^2 + (dy_dt t)^2))

-- The goal is to prove that the arc length is equal to (π^3 / 3)
theorem arc_length_eq : arc_length = π^3 / 3 := 
sorry

end arc_length_eq_l793_793142


namespace vector_magnitude_l793_793845

variable {V : Type*} [InnerProductSpace ℝ V]

theorem vector_magnitude
  (a b : V)
  (h1 : ‖a‖ = 3)
  (h2 : ‖a - b‖ = 5)
  (h3 : inner a b = 1) :
  ‖b‖ = 3 * Real.sqrt 2 :=
by
  sorry

end vector_magnitude_l793_793845


namespace union_complement_l793_793507

open Set

-- Definitions for the universal set U and subsets A, B
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 3, 4}
def B : Set ℕ := {1, 3}

-- Definition for the complement of A with respect to U
def CuA : Set ℕ := U \ A

-- Proof statement
theorem union_complement (U_def : U = {0, 1, 2, 3, 4})
                         (A_def : A = {0, 3, 4})
                         (B_def : B = {1, 3}) :
  (CuA ∪ B) = {1, 2, 3} := by
  sorry

end union_complement_l793_793507


namespace ice_cream_stacks_l793_793535

theorem ice_cream_stacks :
  let ice_cream_flavors := ["vanilla", "chocolate", "strawberry", "cherry", "banana"]
  let ways_to_stack := Nat.factorial ice_cream_flavors.length
  ways_to_stack = 120 :=
by
  let ice_cream_flavors := ["vanilla", "chocolate", "strawberry", "cherry", "banana"]
  let ways_to_stack := Nat.factorial ice_cream_flavors.length
  show (ways_to_stack = 120)
  sorry

end ice_cream_stacks_l793_793535


namespace part_one_part_two_l793_793504

noncomputable def f (x a : ℝ) := 5 - |x + a| - |x - 2|

-- First part
theorem part_one (a : ℝ) (h : a = 1) : 
  {x : ℝ | f x a ≥ 0} = set.Icc (-2 : ℝ) 3 :=
by
  sorry

-- Second part
theorem part_two (a : ℝ) : 
  (∀ x : ℝ, f x a ≤ 1) → (-6 : ℝ) ≤ a ∧ a ≤ 2 :=
by
  sorry

end part_one_part_two_l793_793504


namespace fraction_of_journey_by_bus_l793_793671

theorem fraction_of_journey_by_bus:
  (total_journey distance_by_rail distance_on_foot distance_by_bus : ℝ)
  (h1: total_journey = 130)
  (h2: distance_by_rail = (3/5) * 130)
  (h3: distance_on_foot = 6.5)
  (h4: distance_by_bus = total_journey - (distance_by_rail + distance_on_foot)) :
  distance_by_bus / total_journey = 45.5 / 130 :=
by
  sorry

end fraction_of_journey_by_bus_l793_793671


namespace evaluate_K_l793_793231

theorem evaluate_K : ∃ K : ℕ, 32^2 * 4^4 = 2^K ∧ K = 18 := by
  use 18
  sorry

end evaluate_K_l793_793231


namespace ned_initially_had_games_l793_793007

variable (G : ℕ)

theorem ned_initially_had_games (h1 : (3 / 4) * (2 / 3) * G = 6) : G = 12 := by
  sorry

end ned_initially_had_games_l793_793007


namespace length_DE_is_20_l793_793472

noncomputable def length_DE (BC : ℝ) (angle_C_deg : ℝ) 
  (D : ℝ) (is_midpoint_D : D = BC / 2)
  (is_right_triangle : angle_C_deg = 45): ℝ := 
let DE := D in DE

theorem length_DE_is_20 : ∀ (BC : ℝ) (angle_C_deg : ℝ),
  BC = 40 → 
  angle_C_deg = 45 → 
  let D := BC / 2 in 
  let DE := D in 
  DE = 20 :=
by
  intros BC angle_C_deg hBC hAngle
  sorry

end length_DE_is_20_l793_793472


namespace count_integers_between_cubes_l793_793820

theorem count_integers_between_cubes :
  let a := 10.1
  let b := 10.2 
  let x := a^3
  let y := b^3
  1030 < x ∧ x < 1062 ∧ 1030 < y ∧ y < 1062 → (31) :=
by
  have x_val : x = 1030.301 := by norm_num
  have y_val : y = 1061.208 := by norm_num
  sorry

end count_integers_between_cubes_l793_793820


namespace arrange_five_coins_l793_793756

noncomputable def arrange_coins (coins : ℕ) : Prop :=
coins = 4 ∧
∃ fifth_coin : ℕ,
  fifth_coin = 1 ∧
  ∀ (c : ℕ), c ≤ 4 → touching fifth_coin c

axiom touching : ℕ → ℕ → Prop

theorem arrange_five_coins : arrange_coins 4 :=
by
  sorry

end arrange_five_coins_l793_793756


namespace max_min_m_l793_793777

-- Define the given conditions and the definition of m
def satisfies_condition (x y : ℝ) : Prop :=
  sin x + sin y = 1 / 3

-- Define the expression m
def m (x y : ℝ) : ℝ :=
  sin x - cos y ^ 2

-- Define the problem statement
theorem max_min_m (x y : ℝ) (h : satisfies_condition x y) : 
  ∃ maximum minimum : ℝ,
  maximum = 4 / 9 ∧ minimum = -11 / 16 :=
sorry

end max_min_m_l793_793777


namespace harrys_age_l793_793894

-- Definitions of the ages
variable (Kiarra Bea Job Figaro Harry : ℕ)

-- Given conditions
variable (h1 : Kiarra = 2 * Bea)
variable (h2 : Job = 3 * Bea)
variable (h3 : Figaro = Job + 7)
variable (h4 : Harry = Figaro / 2)
variable (h5 : Kiarra = 30)

-- The statement to prove
theorem harrys_age : Harry = 26 := sorry

end harrys_age_l793_793894


namespace smallest_n_l793_793226

def is_perfect_fourth (m : ℕ) : Prop := ∃ x : ℕ, m = x^4
def is_perfect_fifth (m : ℕ) : Prop := ∃ y : ℕ, m = y^5

theorem smallest_n :
  ∃ n : ℕ, n > 0 ∧ is_perfect_fourth (3 * n) ∧ is_perfect_fifth (2 * n) ∧ n = 6912 :=
by {
  sorry
}

end smallest_n_l793_793226


namespace probability_of_perfect_square_sum_l793_793597

-- Define the conditions
def dice_faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def perfect_squares_in_range : Finset ℕ := {4, 9, 16}
def total_outcomes : ℕ := 8 * 8  -- total number of outcomes when rolling two dice

-- Calculate the number of favorable outcomes
def favorable_outcomes : ℕ :=
  (dice_faces.product dice_faces).filter (λ pair => perfect_squares_in_range.member (pair.1 + pair.2)).card

-- Calculate the probability of getting a perfect square sum
noncomputable def probability_perfect_square_sum : ℚ :=
  favorable_outcomes / total_outcomes

-- The statement to prove that the probability is 3/16
theorem probability_of_perfect_square_sum :
  probability_perfect_square_sum = 3 / 16 :=
begin
  sorry
end

end probability_of_perfect_square_sum_l793_793597


namespace shooting_competition_same_hits_l793_793993

-- Define the probabilities of hitting the target for A and B
def probability_A := 1 / 2
def probability_B := 2 / 3

-- Conditions given in the problem
axiom A_shoots_twice : true
axiom B_shoots_twice : true
axiom events_independent : true

-- Definition of the probability of hitting the target the same number of times for both A and B
def probability_same_hits : ℚ := 13 / 36

-- Statement we need to prove
theorem shooting_competition_same_hits : 
  (probability_A = 1 / 2) ∧ (probability_B = 2 / 3) ∧ 
  A_shoots_twice ∧ B_shoots_twice ∧ events_independent → 
  probability_same_hits = 13 / 36 :=
by
  sorry

end shooting_competition_same_hits_l793_793993


namespace arctan_sum_l793_793189

theorem arctan_sum : ∀ (a b : ℝ), 
  a = 3/7 → 
  b = 7/3 → 
  a * b = 1 → 
  a > 0 → 
  b > 0 → 
  Real.arctan a + Real.arctan b = Real.pi / 2 :=
by intros a b ha hb hab ha_pos hb_pos
   rw [ha, hb, hab]
   sorry

end arctan_sum_l793_793189


namespace ratio_length_breadth_l793_793968

theorem ratio_length_breadth
  (b : ℝ) (A : ℝ) (h_b : b = 11) (h_A : A = 363) :
  (∃ l : ℝ, A = l * b ∧ l / b = 3) :=
by
  sorry

end ratio_length_breadth_l793_793968


namespace bread_calories_l793_793985

theorem bread_calories (total_calories : Nat) (pb_calories : Nat) (pb_servings : Nat) (bread_pieces : Nat) (bread_calories : Nat)
  (h1 : total_calories = 500)
  (h2 : pb_calories = 200)
  (h3 : pb_servings = 2)
  (h4 : bread_pieces = 1)
  (h5 : total_calories = pb_servings * pb_calories + bread_pieces * bread_calories) : 
  bread_calories = 100 :=
by
  sorry

end bread_calories_l793_793985


namespace unique_solution_t_interval_l793_793252

theorem unique_solution_t_interval (x y z v t : ℝ) :
  (x + y + z + v = 0) →
  ((x * y + y * z + z * v) + t * (x * z + x * v + y * v) = 0) →
  (t > (3 - Real.sqrt 5) / 2) ∧ (t < (3 + Real.sqrt 5) / 2) :=
by
  intro h1 h2
  sorry

end unique_solution_t_interval_l793_793252


namespace average_after_19_innings_l793_793871

variables (X : ℕ) (A : ℕ)
-- Define the initial conditions
def cricketer_conditions := 
  (∃ (centuries : ℕ) (half_centuries : ℕ) (below_fifty : ℕ),
    centuries = 3 ∧ half_centuries = 5 ∧ below_fifty = 11 ∧
    ∀ i < 18, (total_score : ℕ) (score : ℕ) (20 ≤ score) ∧ (score ≤ 100))

-- Given statements
axiom initial_average (A : ℕ) : X = 18 * A
axiom runs_in_19th_inning : X + 95
axiom average_increased_by_4 (A : ℕ) : (X + 95) / 19 = A + 4

-- Prove the average after 19 innings
theorem average_after_19_innings : (X + 95) / 19 = 23 :=
begin
  sorry
end

end average_after_19_innings_l793_793871


namespace sum_of_fraction_numerator_denominator_l793_793100

theorem sum_of_fraction_numerator_denominator :
  let x := 0.343434...
  in let fraction := (34 / 99 : ℚ)
  in let sum := fraction.num + fraction.den 
  in (x : ℚ) = fraction ∧ fraction.isReduced → sum = 133 :=
by
  sorry

end sum_of_fraction_numerator_denominator_l793_793100


namespace number_of_valid_sets_of_women_l793_793696

-- Conditions
def daughter_relationships : Prop :=
  ∃(Alice : Type) (Daughters : Alice → Type) (GrandDaughters : ∀ (a : Alice), Daughters a → Type) (GreatGrandDaughters : ∀ (a : Alice) (d : Daughters a), GrandDaughters a d → Type),
    (∀ a : Alice, Fintype (Daughters a)) ∧
    (∀ (a : Alice) (d : Daughters a), Fintype (GrandDaughters a d)) ∧
    (∀ (a : Alice) (d : Daughters a) (g : GrandDaughters a d), Fintype (GreatGrandDaughters a d g)) ∧
    (∃ (a : Alice), Fintype.card (Daughters a) = 3 ∧
                      ∀ d, Fintype.card (GrandDaughters a d) = 2 ∧
                           ∀ g, Fintype.card (GreatGrandDaughters a d g) = 1)

-- Theorem statement
theorem number_of_valid_sets_of_women (h : daughter_relationships) : ∃ (n : ℕ), n = 793 := by 
  sorry

end number_of_valid_sets_of_women_l793_793696


namespace monotonic_intervals_when_a_neg1_range_of_a_and_proof_of_inequality_l793_793317

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x - (a * (x + 1)) / (x - 1)

theorem monotonic_intervals_when_a_neg1 :
  let f := f x (-1)
  -- The domain of f is (0, 1) ∪ (1, +∞)
  -- Increasing intervals of f: (0, 2 - √3) ∪ (2 + √3, +∞)
  -- Decreasing intervals of f: (2 - √3, 1) ∪ (1, 2 + √3)
  sorry

theorem range_of_a_and_proof_of_inequality (h : ∃ x1 x2 : ℝ, x1 < x2 ∧ f x1 a = 0 ∧ f x2 a = 0) :
  a > 0 ∧ ∀ x1 x2 : ℝ, (f x1 a = 0 ∧ f x2 a = 0 ∧ x1 < x2) → 
  ((1 / (log x1 + a)) + (1 / (log x2 + a)) < 0) :=
sorry

end monotonic_intervals_when_a_neg1_range_of_a_and_proof_of_inequality_l793_793317


namespace exponential_function_f1_l793_793311

theorem exponential_function_f1 (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) (h3 : a^3 = 8) : a^1 = 2 := by
  sorry

end exponential_function_f1_l793_793311


namespace count_two_digit_numbers_less_35_l793_793366

open Nat

theorem count_two_digit_numbers_less_35 : 
  let two_digit_numbers := (finset.Ico 10 100) -- The range of two-digit numbers
  let count_satisfying := two_digit_numbers.filter (λ n, (n / 10 < 3 ∨ n % 10 < 5)).card
  count_satisfying = 55 :=
by
  -- Placeholder for the actual proof.
  sorry

end count_two_digit_numbers_less_35_l793_793366


namespace probability_dice_sum_perfect_square_l793_793604

def is_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def probability_perfect_square : ℚ :=
  12 / 64

theorem probability_dice_sum_perfect_square :
  -- Two standard 8-sided dice are rolled
  -- The probability that the sum rolled is a perfect square is 3/16
  (∃ dice1 dice2 : ℕ, 1 ≤ dice1 ∧ dice1 ≤ 8 ∧ 1 ≤ dice2 ∧ dice2 ≤ 8) →
  probability_perfect_square = 3 / 16 :=
sorry

end probability_dice_sum_perfect_square_l793_793604


namespace grocer_initial_stock_l793_793156

theorem grocer_initial_stock 
  (x : ℝ) 
  (h1 : 0.20 * x + 70 = 0.30 * (x + 100)) : 
  x = 400 := by
  sorry

end grocer_initial_stock_l793_793156


namespace probability_dice_sum_perfect_square_l793_793602

def is_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def probability_perfect_square : ℚ :=
  12 / 64

theorem probability_dice_sum_perfect_square :
  -- Two standard 8-sided dice are rolled
  -- The probability that the sum rolled is a perfect square is 3/16
  (∃ dice1 dice2 : ℕ, 1 ≤ dice1 ∧ dice1 ≤ 8 ∧ 1 ≤ dice2 ∧ dice2 ≤ 8) →
  probability_perfect_square = 3 / 16 :=
sorry

end probability_dice_sum_perfect_square_l793_793602


namespace tracy_initial_candies_l793_793077

variable (x : ℕ)
variable (b : ℕ)

theorem tracy_initial_candies : 
  (x % 6 = 0) ∧
  (34 ≤ (1 / 2 * x)) ∧
  ((1 / 2 * x) ≤ 38) ∧
  (1 ≤ b) ∧
  (b ≤ 5) ∧
  (1 / 2 * x - 30 - b = 3) →
  x = 72 := 
sorry

end tracy_initial_candies_l793_793077


namespace common_solutions_form_segment_length_one_l793_793234

theorem common_solutions_form_segment_length_one (a : ℝ) (h₁ : ∀ x : ℝ, x^2 - 4 * x + 2 - a ≤ 0) 
  (h₂ : ∀ x : ℝ, x^2 - 5 * x + 2 * a + 8 ≤ 0) : 
  (a = -1 ∨ a = -7 / 4) :=
by
  sorry

end common_solutions_form_segment_length_one_l793_793234


namespace average_runs_l793_793961

/-- The average runs scored by the batsman in the first 20 matches is 40,
and in the next 10 matches is 30. We want to prove the average runs scored
by the batsman in all 30 matches is 36.67. --/
theorem average_runs (avg20 avg10 : ℕ) (num_matches_20 num_matches_10 : ℕ)
  (h1 : avg20 = 40) (h2 : avg10 = 30) (h3 : num_matches_20 = 20) (h4 : num_matches_10 = 10) :
  ((num_matches_20 * avg20 + num_matches_10 * avg10 : ℕ) : ℚ) / (num_matches_20 + num_matches_10 : ℕ) = 36.67 := by
  sorry

end average_runs_l793_793961


namespace age_of_35th_student_l793_793550

variables (n : ℕ) (average_age_all avg1 avg2 avg3 age4 : ℝ)
variables (n1 n2 n3 : ℕ)

def total_age (n : ℕ) (avg : ℝ) : ℝ := n * avg

theorem age_of_35th_student
  (h_n : n = 35)
  (h_avg_all : average_age_all = 16.5)
  (h_n1 : n1 = 10) (h_avg1 : avg1 = 15.3)
  (h_n2 : n2 = 17) (h_avg2 : avg2 = 16.7)
  (h_n3 : n3 = 6) (h_avg3 : avg3 = 18.4)
  (h_age4 : age4 = 14.7) :
  let total_age_all := total_age n average_age_all,
      total_age_34 := total_age n1 avg1 + total_age n2 avg2 + total_age n3 avg3 + age4
  in total_age_all - total_age_34 = 15.5 :=
sorry

end age_of_35th_student_l793_793550


namespace area_of_region_l793_793012

def plane_region (x y : ℝ) : Prop := |x| ≤ 1 ∧ |y| ≤ 1

def inequality_holds (a b : ℝ) : Prop := ∀ x y : ℝ, plane_region x y → a * x - 2 * b * y ≤ 2

theorem area_of_region (a b : ℝ) (h : inequality_holds a b) : 
  (-2 ≤ a ∧ a ≤ 2) ∧ (-1 ≤ b ∧ b ≤ 1) ∧ (4 * 2 = 8) :=
sorry

end area_of_region_l793_793012


namespace count_four_digit_even_numbers_l793_793813

def digitSet := {0, 2, 4, 6, 8}

def validFirstDigitSet := {2, 4, 6, 8}

noncomputable def countValidFourDigitEvenNumbers : Nat :=
  Set.card validFirstDigitSet * (Set.card digitSet * Set.card digitSet * Set.card digitSet)

theorem count_four_digit_even_numbers : countValidFourDigitEvenNumbers = 500 :=
by
  rw [countValidFourDigitEvenNumbers]
  norm_num
  sorry

end count_four_digit_even_numbers_l793_793813


namespace find_a_l793_793275

def a := ℝ 
def i := Complex.I
def lhs := (2 + a * i) / (1 + i)
def rhs := 3 + i

theorem find_a (a : ℝ) (h : lhs = rhs) : a = 4 := 
by
  sorry

end find_a_l793_793275


namespace length_de_l793_793476

open Triangle

/-- In triangle ABC, BC = 40 and ∠C = 45°. Let the perpendicular bisector
of BC intersect BC and AC at D and E, respectively. Prove that DE = 10√2. -/

theorem length_de 
  (ABC : Triangle)
  (B C : Point)
  (BC_40 : dist B C = 40)
  (angle_C_45 : ∠(B, C, AC) = 45)
  (D : Point)
  (midpoint_D : is_midpoint B C D)
  (E : Point)
  (intersection_E : is_perpendicular_bisector_intersection B C D E) :
  dist D E = 10 * real.sqrt 2 :=
begin
  -- proof steps would go here
  sorry
end

end length_de_l793_793476


namespace solution_exists_l793_793327

open Int

def isPrime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d ∣ p, d = 1 ∨ d = p

theorem solution_exists :
  ∃ (p : ℕ), isPrime p ∧ ∃ (x : ℤ), (2 * x^2 - x - 36 = p^2 ∧ x = 13) := sorry

end solution_exists_l793_793327


namespace two_digit_numbers_count_l793_793343

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem two_digit_numbers_count :
  { n : ℕ | 10 ≤ n ∧ n ≤ 99 ∧ (tens_digit n < 3 ∨ units_digit n < 5) }.to_finset.card = 55 :=
by
  sorry

end two_digit_numbers_count_l793_793343


namespace distance_between_tangency_points_l793_793994

theorem distance_between_tangency_points
  (circle_radius : ℝ) (M_distance : ℝ) (A_distance : ℝ) 
  (h1 : circle_radius = 7)
  (h2 : M_distance = 25)
  (h3 : A_distance = 7) :
  ∃ AB : ℝ, AB = 48 :=
by
  -- Definitions and proofs will go here.
  sorry

end distance_between_tangency_points_l793_793994


namespace david_produces_8_more_widgets_l793_793925

variable (w t : ℝ)

def widgets_monday (w t : ℝ) : ℝ :=
  w * t

def widgets_tuesday (w t : ℝ) : ℝ :=
  (w + 4) * (t - 2)

theorem david_produces_8_more_widgets (h : w = 2 * t) : 
  widgets_monday w t - widgets_tuesday w t = 8 :=
by
  sorry

end david_produces_8_more_widgets_l793_793925


namespace min_length_of_segment_PT_l793_793770

noncomputable def min_length_PT (a : ℝ) : ℝ :=
  let P := (2, -1)
  let PC := real.sqrt ((2 - 0) ^ 2 + ((-1) - a) ^ 2)
  let r := real.sqrt (a ^ 2 - 2)
  real.sqrt (PC^2 - r^2)

theorem min_length_of_segment_PT : 
  ∃ a : ℝ, (2 - 0) ^ 2 + ((-1) - a) ^ 2 ≥ 2 ∧ min_length_PT a = real.sqrt 2 :=
sorry

end min_length_of_segment_PT_l793_793770


namespace part_a_part_b_part_c_part_d_l793_793496

noncomputable def independent_random_variables (X Y : ℝ → ℝ) : Prop :=
sorry

noncomputable def continuous (Y : ℝ → ℝ) : Prop :=
∀ y : ℝ, MeasureTheory.ProbabilityTheory.prob_event (Y = y) = 0

noncomputable def density (Y : ℝ → ℝ) (f : ℝ → ℝ) : Prop :=
sorry

noncomputable def singular (X Y : ℝ → ℝ) : Prop :=
sorry

theorem part_a (X Y : ℝ → ℝ) (h_indep : independent_random_variables X Y)
  (h_cont : continuous Y) : continuous (λ t : ℝ, X t + Y t) :=
sorry

theorem part_b (X Y : ℝ → ℝ) (f : ℝ → ℝ) (h_indep : independent_random_variables X Y)
  (h_density : density Y f) : density (λ t : ℝ, X t + Y t) (λ z : ℝ, (MeasureTheory.Expectation (f (z - X t)))) :=
sorry

theorem part_c (X Y : ℝ → ℝ) (h_discrete_singular : singular X Y) : singular (λ t : ℝ, X t + Y t) :=
sorry

theorem part_d (X Y : ℝ → ℝ) (h_not_indep : ¬ independent_random_variables X Y) :
¬ (continuous (λ t : ℝ, X t + Y t) ∧ density (λ t : ℝ, X t + Y t) (λ z : ℝ, MeasureTheory.Expectation (λ x : ℝ, density (λ z : ℝ, X t + Y t - x)))) :=
sorry

end part_a_part_b_part_c_part_d_l793_793496


namespace arctan_sum_l793_793203

theorem arctan_sum (x y : ℝ) (hx : x = 3) (hy : y = 7) :
  Real.arctan (x / y) + Real.arctan (y / x) = Real.pi / 2 := 
by
  rw [hx, hy]
  sorry

end arctan_sum_l793_793203


namespace salary_increase_l793_793922

theorem salary_increase (S : ℝ) : 
  let new_salary := S * (1.05 ^ 4)
  let increase := (new_salary - S) / S * 100
  increase ≈ 21.5 :=
sorry

end salary_increase_l793_793922


namespace problem_1_problem_2_problem_3_problem_4_l793_793187

-- Define the problems as Lean theorems without proof (using sorry to skip the proof).

theorem problem_1 : (∛27 - (sqrt 2 * sqrt 6) / sqrt 3) = 1 := 
by sorry

theorem problem_2 : (2 * sqrt 32 - sqrt 50) = 3 * sqrt 2 := 
by sorry

theorem problem_3 : (sqrt 12 - sqrt 8 + sqrt (4/3) + 2 * sqrt (1/2)) = (8 * sqrt 3 / 3) - sqrt 2 := 
by sorry

theorem problem_4 : (sqrt 48 + sqrt 3 - sqrt (1/2) * sqrt 12 + sqrt 24) = 5 * sqrt 3 + sqrt 6 := 
by sorry

end problem_1_problem_2_problem_3_problem_4_l793_793187


namespace total_surface_area_proof_l793_793958

-- Define the radius and the areas provided
variables (r : ℝ) (base_area lid_area: ℝ)

-- Given conditions
axiom base_area_eq_144pi : base_area = 144 * Real.pi
axiom lid_same_radius : lid_area = base_area

-- Definition of the total surface area including the curved surface and the lid
noncomputable def total_surface_area (r : ℝ) : ℝ := 2 * Real.pi * r ^ 2 + lid_area

-- Radius r is calculated based on the given base area
lemma radius_from_base_area : r = Real.sqrt (base_area / Real.pi) := 
begin
  -- implicitly comes from translating given condition into how we compute the radius.
  sorry,
end

theorem total_surface_area_proof : total_surface_area r = 432 * Real.pi :=
by
  have hr : r = 12 := by sorry -- deriving r = 12 from base_area_eq_144pi
  have hlid : lid_area = 144 * Real.pi := by sorry -- given condition
  rw [hr],
  rw [hlid],
  -- Now combining curved surface area of the hemisphere and the lid
  have h_curved_surface_area : 2 * Real.pi * 12 ^ 2 = 288 * Real.pi := by sorry, 
  rw [h_curved_surface_area],
  norm_num,
  sorry

end total_surface_area_proof_l793_793958


namespace least_sum_of_exponents_for_400_l793_793983

theorem least_sum_of_exponents_for_400 :
  ∃ (exponents : List ℕ), (∀ e ∈ exponents, ∃ n, 2^n = 2^e) ∧ (List.sum exponents = 19) ∧ (List.sum (exponents.map (λ e, 2^e)) = 400) ∧ (2 ≤ exponents.length) :=
  sorry

end least_sum_of_exponents_for_400_l793_793983


namespace solve_log_inequality_l793_793951

theorem solve_log_inequality (a x : ℝ) : 
  (a > 0) ∧ (a ≠ 1) ∧ (log a (4 + 3 * x - x^2) - log a (2 * x - 1) > log a 2) → 
  ((a > 1) → (1 / 2 < x ∧ x < 2)) ∧ 
  ((0 < a ∧ a < 1) → (2 < x ∧ x < 4)) := 
by 
  sorry

end solve_log_inequality_l793_793951


namespace solve_inequality1_solve_inequality2_l793_793030

-- Proof problem 1
theorem solve_inequality1 (x : ℝ) : 
  2 < |2 * x - 5| → |2 * x - 5| ≤ 7 → -1 ≤ x ∧ x < (3 / 2) ∨ (7 / 2) < x ∧ x ≤ 6 :=
sorry

-- Proof problem 2
theorem solve_inequality2 (x : ℝ) : 
  (1 / (x - 1)) > (x + 1) → x < - Real.sqrt 2 ∨ (1 < x ∧ x < Real.sqrt 2) :=
sorry

end solve_inequality1_solve_inequality2_l793_793030


namespace length_of_wall_l793_793886

theorem length_of_wall (area width : ℝ) (h1 : area = 12.0) (h2 : width = 8) :
  ∃ length : ℝ, area = width * length ∧ length = 1.5 :=
by
  use 1.5
  split
  · rw [←h1, ←h2]
    norm_num
  · exact rfl

end length_of_wall_l793_793886


namespace length_of_metallic_sheet_l793_793673

variable (L : ℝ) (width side volume : ℝ)

theorem length_of_metallic_sheet (h1 : width = 36) (h2 : side = 8) (h3 : volume = 5120) :
  ((L - 2 * side) * (width - 2 * side) * side = volume) → L = 48 := 
by
  intros h_eq
  sorry

end length_of_metallic_sheet_l793_793673


namespace statement_B_statement_C_l793_793906

variable (z1 z2 : ℂ)

-- Statement B: z1 - conj(z1) is purely imaginary or zero
theorem statement_B : ∃ (a : ℝ), z1 - conj(z1) = a * I ∨ z1 - conj(z1) = 0 := sorry

-- Statement C: |z1 + z2| ≤ |z1| + |z2| holds for any complex numbers z1 and z2
theorem statement_C : abs (z1 + z2) ≤ abs z1 + abs z2 := sorry

end statement_B_statement_C_l793_793906


namespace sum_of_distinct_interior_angles_l793_793080

-- Definitions for the problem
structure Quadrilateral :=
  (A B C D : Point)
  (angle_sum : Real := 360)

-- Given quadrilaterals ABCD and PQRS with specific vertex coincidences
def ABCD : Quadrilateral := ⟨A, B, C, D⟩
def PQRS : Quadrilateral := ⟨P, Q, S, R⟩

structure OverlappingQuadrilaterals :=
  (ABCD PQRS : Quadrilateral)
  (C_eq_S : C = S)
  (D_eq_R : D = R)

-- The theorem to prove
theorem sum_of_distinct_interior_angles (overlap: OverlappingQuadrilaterals) : 
  (ABCD.angle_sum + PQRS.angle_sum - ∠(ABCD.C B A) - ∠(PQRS.R P Q)) = 540 := 
by
  sorry

end sum_of_distinct_interior_angles_l793_793080


namespace harrys_age_l793_793892

theorem harrys_age {K B J F H : ℕ} 
  (hKiarra : K = 30)
  (hKiarra_Bea : K = 2 * B)
  (hJob : J = 3 * B)
  (hFigaro : F = J + 7)
  (hHarry : H = F / 2) : 
  H = 26 := 
by 
  -- Definitions from the conditions
  have hBea : B = 15, from (by linarith : 15 = K / 2).symm,
  
  -- Continuing the proof using the provided conditions and calculating step by step
  sorry

end harrys_age_l793_793892


namespace find_k_l793_793666

-- We will define the points and the collinearity condition
def point (α : Type) := (α × α)

-- Given points
def p1 : point ℚ := (2, 8)
def p2 : point ℚ := (10, 32/7)
def p3 : point ℚ := (16, 2)

-- Slope calculation between two points
def slope (pA pB : point ℚ) : ℚ :=
  (pB.2 - pA.2) / (pB.1 - pA.1)

-- The main theorem stating that all three points are collinear if the slopes match
theorem find_k : 
  ∃ k : ℚ, slope (2, 8) (10, k) = slope (10, k) (16, 2) ∧ k = 32 / 7 :=
by
  use 32 / 7
  -- Proof of collinearity would go here
  sorry

end find_k_l793_793666


namespace probability_of_perfect_square_sum_on_two_8_sided_dice_l793_793593

-- Define the dice and possible sums
def is_sum_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def pairs_rolled : finset (ℕ × ℕ) :=
  finset.univ.filter (λ p, p.1 + p.2 ∈ {4, 9, 16})

theorem probability_of_perfect_square_sum_on_two_8_sided_dice :
  (pairs_rolled.card : ℚ) / 64 = 3 / 16 :=
by
  sorry

end probability_of_perfect_square_sum_on_two_8_sided_dice_l793_793593


namespace decagon_parallel_side_l793_793137

theorem decagon_parallel_side (A B C D E F G H I J : Point) (h : Inscribed (A ▸ B ▸ C ▸ D ▸ E ▸ F ▸ G ▸ H ▸ I ▸ J ▸ A)) :
  (Parallel (A, B) (F, G)) → (Parallel (B, C) (G, H)) → (Parallel (C, D) (H, I)) → (Parallel (D, E) (I, J)) → (Parallel (E, F) (J, A)) :=
begin
  sorry
end

end decagon_parallel_side_l793_793137


namespace two_digit_numbers_count_l793_793338

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem two_digit_numbers_count :
  { n : ℕ | 10 ≤ n ∧ n ≤ 99 ∧ (tens_digit n < 3 ∨ units_digit n < 5) }.to_finset.card = 55 :=
by
  sorry

end two_digit_numbers_count_l793_793338


namespace sum_of_numerator_and_denominator_l793_793112

def repeating_decimal_represents_fraction : Prop :=
  let x := 34 / 99
  0.34̅ = x

/-- The sum of the numerator and denominator of 0.34̅ in its lowest terms is 133. -/
theorem sum_of_numerator_and_denominator
  (x : ℚ)
  (hx : repeating_decimal_represents_fraction) :
  (34 + 99 = 133) :=
begin
  sorry
end

end sum_of_numerator_and_denominator_l793_793112


namespace number_of_red_balls_l793_793150

theorem number_of_red_balls (r : ℕ) (h : r ≠ 0 ∧ r ≤ 15) :
  ((r * (r-1)) = 30) → r = 6 :=
by {
    -- Set up the conditions and acknowledge the known probability.
    intro hr,
    have h2 : r * (r - 1) = 30 := hr,
    rw ←nat.mul_div_cancel' at h2,
    rw h2,
    have possible_values := nat.eq_or_eq_of_mul_eq_mul (by norm_num : 2 * 15 = 30) r (r - 1),
    cases possible_values,
    { exact possible_values.elim (λ h, nat.mul_self_inj.mp h) (λ h, by contradiction) },
    { intro rid, cases rid },
}

end number_of_red_balls_l793_793150


namespace quadrilateral_inequality_l793_793493

variable (a b c d S : ℝ)
variable (x y z w : ℝ)
variable (permutation : List ℝ → List ℝ)

noncomputable def is_permutation (l1 l2 : List ℝ) : Prop :=
  l1.perm l2

theorem quadrilateral_inequality
  (hS : S = (S : ℝ))
  (hperm : is_permutation [x, y, z, w] [a, b, c, d]) :
  S ≤ 1 / 2 * (x * y + z * w) :=
sorry

end quadrilateral_inequality_l793_793493


namespace problem_statement_l793_793286

noncomputable def f (x : ℝ) (b c : ℝ) := x^2 + b * x + c

theorem problem_statement (b c : ℝ) (h : ∀ x : ℝ, f (x - 1) b c = f (3 - x) b c) : f 0 b c < f (-2) b c ∧ f (-2) b c < f 5 b c := 
by sorry

end problem_statement_l793_793286


namespace triangle_third_side_length_l793_793448

theorem triangle_third_side_length :
  ∀ (a b : ℝ) (θ : ℝ), a = 10 → b = 11 → θ = real.pi * 5 / 6 →
  ∃ c : ℝ, c = real.sqrt (a^2 + b^2 - 2 * a * b * real.cos θ) :=
by
  intros a b θ ha hb hθ
  rw [ha, hb, hθ]
  use real.sqrt (10^2 + 11^2 + 2 * 10 * 11 * (real.sqrt 3 / 2))
  sorry

end triangle_third_side_length_l793_793448


namespace num_positive_integer_N_l793_793244

def num_valid_N : Nat := 7

theorem num_positive_integer_N (N : Nat) (h_pos : N > 0) :
  (∃ k : Nat, k > 3 ∧ N = k - 3 ∧ 48 % k = 0) ↔ (N < 45) ∧ (num_valid_N = 7) := 
by
sorry

end num_positive_integer_N_l793_793244


namespace power_of_point_eq_dist_squared_l793_793532

variables {V : Type*} [inner_product_space ℝ V]

-- Define the equation of a sphere
def sphere_eq (r q : V) (R : ℝ) : Prop := (r - q) • (r - q) = R^2

-- Define the distance squared
def dist_squared (OM OC : V) : ℝ := (OM - OC) • (OM - OC)

-- Power of a point with respect to the sphere
def power_of_point (r1 q : V) (R : ℝ) : ℝ := (r1 - q) • (r1 - q) - R^2

theorem power_of_point_eq_dist_squared {OM OC : V} (R : ℝ) :
  power_of_point OM OC R = dist_squared OM OC - R^2 :=
sorry

end power_of_point_eq_dist_squared_l793_793532


namespace opposite_of_negative_mixed_number_l793_793566

theorem opposite_of_negative_mixed_number :
  let x := -1 - 3/4 in -x = 1 + 3/4 := by
  let x := -7/4
  show -x = 7/4 from sorry

end opposite_of_negative_mixed_number_l793_793566


namespace complex_conjugate_l793_793276

noncomputable def z := {z : ℂ // (z / (1 - z) = complex.I)}

theorem complex_conjugate (h : ∀ z, z / (1 - z) = complex.I) : ∃ z : ℂ, z = (1 / 2 + complex.I / 2) ∧ complex.conj z = (1 / 2 - complex.I / 2) :=
by
  use (1 / 2 + complex.I / 2)
  split
  · refl
  · rw [complex.conj_of_real, complex.conj_I, complex.add_conj]
  sorry

end complex_conjugate_l793_793276


namespace leap_years_count_l793_793674

def is_candidate_year (y : ℕ) : Prop :=
  (y % 100 = 0) ∧ (y % 700 = 200 ∨ y % 700 = 500)

def is_in_range (y : ℕ) : Prop :=
  2000 ≤ y ∧ y ≤ 4500

def valid_years_count : ℕ :=
  {y : ℕ | is_candidate_year y ∧ is_in_range y}.to_finset.card

theorem leap_years_count :
  valid_years_count = 8 :=
by
  sorry

end leap_years_count_l793_793674


namespace number_2digit_smaller_than_35_l793_793349

/--
Prove that the number of two-digit numbers where at least one digit
is smaller than the corresponding digit in 35 is exactly 55.
-/
theorem number_2digit_smaller_than_35 : 
  (Finset.filter (λ n : ℕ, let d1 := n / 10, d2 := n % 10 in d1 < 3 ∨ d2 < 5) (Finset.range' 10 100)).card = 55 := 
by
  sorry

end number_2digit_smaller_than_35_l793_793349


namespace base_of_logarithm_is_four_l793_793427

theorem base_of_logarithm_is_four (x : ℝ) (b : ℝ) (h1 : x = 12) (h2 : log b x + log b (1/6) = 1/2) :
  b = 4 :=
by
  sorry

end base_of_logarithm_is_four_l793_793427


namespace smallest_positive_period_f_max_value_f_in_interval_min_value_f_in_interval_l793_793763

noncomputable def f (x : ℝ) : ℝ := 4 * cos x * sin (x + π / 6) - 1

theorem smallest_positive_period_f : ∀ x, f (x + π) = f x :=
by sorry

theorem max_value_f_in_interval : 
  ∀ x, -π/6 ≤ x ∧ x ≤ π/4 → f x ≤ 2 :=
by sorry

theorem min_value_f_in_interval :
  ∀ x, -π/6 ≤ x ∧ x ≤ π/4 → f x ≥ -1 :=
by sorry

end smallest_positive_period_f_max_value_f_in_interval_min_value_f_in_interval_l793_793763


namespace problem_1_problem_2_problem_3_problem_4_l793_793745

-- General solution to 2y'' - 3y' + y = 0
theorem problem_1 (C₁ C₂ : ℂ) (y : ℂ → ℂ) :
  (∀ x, 2 * (y x).derivative.derivative - 3 * (y x).derivative + y x = 0) ↔ 
  (y = λ x, C₁ * complex.exp x + C₂ * complex.exp (0.5 * x)) :=
sorry

-- General and particular solution to 9y'' + 12y' + 4y = 0 with initial conditions x = 1, y = 2, y' = 1
theorem problem_2 (C₁ C₂ : ℂ) (y : ℂ → ℂ) :
  (∀ x, 9 * (y x).derivative.derivative + 12 * (y x).derivative + 4 * y x = 0 ∧ y 1 = 2 ∧ (y.derivative 1) = 1) ↔ 
  (y = λ x, (C₁ + C₂ * x) * complex.exp (- (2/3) * x) ∧ y = λ x, (1/3) * (7 * x - 1) * complex.exp ((2/3) * (1 - x))) :=
sorry

-- General solution to y'' + 4y' + 5y = 0
theorem problem_3 (C₁ C₂ : ℂ) (y : ℂ → ℂ) :
  (∀ x, (y x).derivative.derivative + 4 * (y x).derivative + 5 * y x = 0) ↔ 
  (y = λ x, (complex.exp (-2 * x)) * (C₁ * complex.cos x + C₂ * complex.sin x)) :=
sorry

-- General and particular solution to 4y'' + 25y = 0 with initial conditions x = π, y = 2, y' = 1
theorem problem_4 (C₁ C₂ : ℂ) (y : ℂ → ℂ) :
  (∀ x, 4 * (y x).derivative.derivative + 25 * y x = 0 ∧ y real.pi = 2 ∧ (y.derivative real.pi) = 1) ↔ 
  (y = λ x, C₁ * complex.cos ((5/2) * x) + C₂ * complex.sin ((5/2) * x) ∧ y = λ x, (-2/5) * complex.cos ((5/2) * x) + 2 * complex.sin ((5/2) * x)) :=
sorry

end problem_1_problem_2_problem_3_problem_4_l793_793745


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l793_793117

theorem sum_of_numerator_and_denominator_of_repeating_decimal (x : ℚ) (h : x = 34 / 99) : (x.den + x.num : ℤ) = 133 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l793_793117


namespace arctan_sum_l793_793191

theorem arctan_sum : ∀ (a b : ℝ), 
  a = 3/7 → 
  b = 7/3 → 
  a * b = 1 → 
  a > 0 → 
  b > 0 → 
  Real.arctan a + Real.arctan b = Real.pi / 2 :=
by intros a b ha hb hab ha_pos hb_pos
   rw [ha, hb, hab]
   sorry

end arctan_sum_l793_793191


namespace find_line_equation_l793_793291

def ellipse (x y : ℝ) : Prop := (x ^ 2) / 6 + (y ^ 2) / 3 = 1

def meets_first_quadrant (l : Line) : Prop :=
  ∃ A B : Point, ellipse A.x A.y ∧ ellipse B.x B.y ∧ 
  A.x > 0 ∧ A.y > 0 ∧ B.x > 0 ∧ B.y > 0 ∧ l.contains A ∧ l.contains B

def intersects_axes (l : Line) : Prop :=
  ∃ M N : Point, M.y = 0 ∧ N.x = 0 ∧ l.contains M ∧ l.contains N
  
def equal_distances (M N A B : Point) : Prop :=
  dist M A = dist N B

def distance_MN (M N : Point) : Prop :=
  dist M N = 2 * Real.sqrt 3

theorem find_line_equation (l : Line) (A B M N : Point)
  (h1 : meets_first_quadrant l)
  (h2 : intersects_axes l)
  (h3 : equal_distances M N A B)
  (h4 : distance_MN M N) :
  l.equation = "x + sqrt(2) * y - 2 * sqrt(2) = 0" :=
sorry

end find_line_equation_l793_793291


namespace solve_for_x_l793_793028

theorem solve_for_x (x : ℝ) : 
  (2:ℝ)^(32^x) = (32:ℝ)^(2^x) ↔ x = (Real.log 5 / Real.log 2) / 4 :=
by
  sorry

end solve_for_x_l793_793028


namespace y_at_x8_l793_793423

theorem y_at_x8 (k : ℝ) (y : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, y x = k * x^(1/3))
  (h2 : y 64 = 4 * Real.sqrt 3) :
  y 8 = 2 * Real.sqrt 3 :=
by
  sorry

end y_at_x8_l793_793423


namespace maria_purse_value_l793_793918

def value_of_nickels (num_nickels : ℕ) : ℕ := num_nickels * 5
def value_of_dimes (num_dimes : ℕ) : ℕ := num_dimes * 10
def value_of_quarters (num_quarters : ℕ) : ℕ := num_quarters * 25
def total_value (num_nickels num_dimes num_quarters : ℕ) : ℕ := 
  value_of_nickels num_nickels + value_of_dimes num_dimes + value_of_quarters num_quarters
def percentage_of_dollar (value_cents : ℕ) : ℕ := value_cents * 100 / 100

theorem maria_purse_value : percentage_of_dollar (total_value 2 3 2) = 90 := by
  sorry

end maria_purse_value_l793_793918


namespace count_two_digit_numbers_less_35_l793_793369

open Nat

theorem count_two_digit_numbers_less_35 : 
  let two_digit_numbers := (finset.Ico 10 100) -- The range of two-digit numbers
  let count_satisfying := two_digit_numbers.filter (λ n, (n / 10 < 3 ∨ n % 10 < 5)).card
  count_satisfying = 55 :=
by
  -- Placeholder for the actual proof.
  sorry

end count_two_digit_numbers_less_35_l793_793369


namespace sum_of_m_satisfying_conditions_l793_793831

-- Definitions for conditions
def fractional_eq (m x : ℤ) : Prop := (m * x - 1) / (x - 2) = 2 + 3 / (2 - x)
def quadratic_intersects_x_axis (m : ℤ) : Prop := (2^2 - 4 * (m - 2) * 1) ≥ 0

-- Proofs that sum of m is 0
theorem sum_of_m_satisfying_conditions : 
  let S := {m : ℤ | ∃ x : ℤ, fractional_eq m x ∧ quadratic_intersects_x_axis m} in
  ∑ m in S, m = 0 :=
by
  -- Proof is skipped, just provide the statement for now
  sorry

end sum_of_m_satisfying_conditions_l793_793831


namespace linear_equation_a_is_minus_one_l793_793431

theorem linear_equation_a_is_minus_one (a : ℝ) (x : ℝ) :
  ((a - 1) * x ^ (2 - |a|) + 5 = 0) → (2 - |a| = 1) → (a ≠ 1) → a = -1 :=
by
  intros h1 h2 h3
  sorry

end linear_equation_a_is_minus_one_l793_793431


namespace angle_C_is_150_degrees_l793_793706

theorem angle_C_is_150_degrees
  (C D : ℝ)
  (h_supp : C + D = 180)
  (h_C_5D : C = 5 * D) :
  C = 150 :=
by
  sorry

end angle_C_is_150_degrees_l793_793706


namespace probability_point_not_above_x_axis_l793_793014

theorem probability_point_not_above_x_axis (A B C D : ℝ × ℝ) :
  A = (9, 4) →
  B = (3, -2) →
  C = (-3, -2) →
  D = (3, 4) →
  (1 / 2 : ℚ) = 1 / 2 := 
by 
  intros hA hB hC hD 
  sorry

end probability_point_not_above_x_axis_l793_793014


namespace swap_original_x_y_l793_793170

variables (x y z : ℕ)

theorem swap_original_x_y (x_original y_original : ℕ) 
  (step1 : z = x_original)
  (step2 : x = y_original)
  (step3 : y = z) :
  x = y_original ∧ y = x_original :=
sorry

end swap_original_x_y_l793_793170


namespace f_2048_eq_1728_l793_793059

noncomputable def f : ℤ → ℝ := sorry

theorem f_2048_eq_1728 (h : ∀ a b n : ℤ, a > 0 → b > 0 → a + b = 2^n → f a + f b = n^3) : f 2048 = 1728 :=
  sorry

end f_2048_eq_1728_l793_793059


namespace probability_two_8sided_dice_sum_perfect_square_l793_793606

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def probabilityPerfectSquareSum (dice_sides : ℕ) (perfect_squares : List ℕ) : ℚ :=
  let outcomes := (dice_sides * dice_sides)
  let favorable_outcomes := perfect_squares.sum (λ ps, 
    (List.filter (λ (pair : ℕ × ℕ), pair.fst + pair.snd = ps) ((List.range dice_sides).product (List.range dice_sides))).length)
  favorable_outcomes /. outcomes

theorem probability_two_8sided_dice_sum_perfect_square :
  probabilityPerfectSquareSum 8 [4, 9, 16] = 3 / 16 := sorry

end probability_two_8sided_dice_sum_perfect_square_l793_793606


namespace original_cost_price_correct_l793_793685

variable (SP : ℕ) (D : ℝ) (L : ℕ) (n : ℕ) (C : ℕ)

def total_loss (n L : ℕ) : ℕ := n * L

def cost_price (SP L : ℕ) : ℕ := SP + (total_loss n L)

def original_cost_per_metre (CP n : ℕ) : ℝ := CP / n

def solve_original_cost_price (SP : ℕ) (D : ℝ) (L : ℕ) (n : ℕ) : ℝ := 
  let CP := cost_price SP (total_loss n L)
  original_cost_per_metre CP n

theorem original_cost_price_correct :
  solve_original_cost_price 18000 0.1 5 200 = 95 := by
  sorry

end original_cost_price_correct_l793_793685


namespace seventh_observation_l793_793139

-- Definitions from the conditions
def avg_original (x : ℕ) := 13
def num_observations_original := 6
def total_original := num_observations_original * (avg_original 0) -- 6 * 13 = 78

def avg_new := 12
def num_observations_new := num_observations_original + 1 -- 7
def total_new := num_observations_new * avg_new -- 7 * 12 = 84

-- The proof goal statement
theorem seventh_observation : (total_new - total_original) = 6 := 
  by
    -- Placeholder for the proof
    sorry

end seventh_observation_l793_793139


namespace cylinder_radius_l793_793967

theorem cylinder_radius (h r: ℝ) (S: ℝ) (S_eq: S = 130 * Real.pi) (h_eq: h = 8) 
    (surface_area_eq: S = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) : 
    r = 5 :=
by {
  -- Placeholder for proof steps.
  sorry
}

end cylinder_radius_l793_793967


namespace estimate_total_weight_300_carps_l793_793253

noncomputable def average_weight (weights: List ℝ) : ℝ :=
  (weights.foldl (+) 0) / weights.length

noncomputable def total_weight_estimate (num_carps : ℕ) (weights: List ℝ) : ℝ :=
  num_carps * average_weight weights

theorem estimate_total_weight_300_carps :
  total_weight_estimate 300 [1.5, 1.6, 1.4, 1.6, 1.2, 1.7, 1.5, 1.8, 1.3, 1.4] = 450 :=
by
  sorry

end estimate_total_weight_300_carps_l793_793253


namespace minimum_value_MP_plus_MF_l793_793280

noncomputable def point := ℝ × ℝ

def parabola (p : point) : Prop := (p.2)^2 = 4 * p.1

def P : point := (3, 2)
def F : point := (1, 0)

def inside_parabola (p : point) : Prop := parabola (p)

def minimum_MP_plus_MF (M : point) : ℝ := 
  if parabola M 
  then real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2) + real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2)
  else 0

theorem minimum_value_MP_plus_MF : ∃ M, parabola M ∧ minimum_MP_plus_MF M = 4 := 
  sorry

end minimum_value_MP_plus_MF_l793_793280


namespace proof_m_t_product_l793_793500

def g (x : ℝ) : ℝ :=
sorry

axiom g_property : ∀ x y : ℝ, g (g x - y) = g x + g (g y - g (-x)) + 2 * x

def g_value (x : ℝ) : ℝ :=
g x

theorem proof_m_t_product : 
  let m := 1
  let t := g_value 4
  t = -8 →
  m * t = -8 :=
by
  intro h_t_eq_neg8
  rw [h_t_eq_neg8]
  norm_num

end proof_m_t_product_l793_793500


namespace tangency_equivalence_l793_793859

-- Define the points and objects in the problem
variables (A B C D P : Point)
variables (I1 I2 O H : Point)

-- Assume the given conditions hold
-- Convex quadrilateral with intersection points and centers as described
def conditions (h_convex : is_convex_quadrilateral A B C D)
               (h_inter_P : is_intersection P AD BC)
               (h_incenter_I1 : is_incenter I1 P A B )
               (h_incenter_I2 : is_incenter I2 P D C)
               (h_circumcenter_O : is_circumcenter O P A B)
               (h_orthocenter_H : is_orthocenter H P D C) : Prop := 
  true

-- The main theorem statement
theorem tangency_equivalence 
        {A B C D P : Point}
        {I1 I2 O H : Point}
        (h_convex : is_convex_quadrilateral A B C D)
        (h_inter_P : is_intersection P AD BC)
        (h_incenter_I1 : is_incenter I1 P A B)
        (h_incenter_I2 : is_incenter I2 P D C)
        (h_circumcenter_O : is_circumcenter O P A B)
        (h_orthocenter_H : is_orthocenter H P D C) : 
        (are_circumcircles_tangent (circumcircle A I1 B) (circumcircle D H C)) ↔ (are_circumcircles_tangent (circumcircle A O B) (circumcircle D I2 C)) :=
sorry

end tangency_equivalence_l793_793859


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l793_793118

theorem sum_of_numerator_and_denominator_of_repeating_decimal (x : ℚ) (h : x = 34 / 99) : (x.den + x.num : ℤ) = 133 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l793_793118


namespace sin_squared_sum_constant_cos_triple_sum_sin_triple_sum_l793_793143

variable (α β γ : ℝ)

-- Define the conditions
def cos_sum_zero : Prop := cos α + cos β + cos γ = 0
def sin_sum_zero : Prop := sin α + sin β + sin γ = 0

-- The main theorem statements:
theorem sin_squared_sum_constant (hcos : cos_sum_zero α β γ) (hsin : sin_sum_zero α β γ) :
  sin α ^ 2 + sin β ^ 2 + sin γ ^ 2 = 1.5 :=
sorry

theorem cos_triple_sum (hcos : cos_sum_zero α β γ) (hsin : sin_sum_zero α β γ) :
  cos (3 * α) + cos (3 * β) + cos (3 * γ) = 3 * cos (α + β + γ) :=
sorry

theorem sin_triple_sum (hcos : cos_sum_zero α β γ) (hsin : sin_sum_zero α β γ) :
  sin (3 * α) + sin (3 * β) + sin (3 * γ) = 3 * sin (α + β + γ) :=
sorry

end sin_squared_sum_constant_cos_triple_sum_sin_triple_sum_l793_793143


namespace sum_of_roots_eq_9_l793_793397

theorem sum_of_roots_eq_9 (x : ℝ) (h : x^2 - 9*x + 20 = 0) : x = 4 ∨ x = 5 → (4 + 5 = 9) :=
begin
  sorry
end

end sum_of_roots_eq_9_l793_793397


namespace disease_type_related_to_gender_expected_vaccine_cost_l793_793657

section DiseaseAnalysis

variable (n : ℕ) (m f : ℕ) (maleA maleB femaleA femaleB : ℕ)
variable (totalMale totalFemale totalA totalB : ℕ)
variable (probOneDoseSucceeds : ℚ) (doseCost : ℕ) (alpha : ℚ) (chi_squared : ℚ) (cost : ℚ)

/-- Define the given conditions for the contingency table and $\chi^2$ calculation  --/
def conditions : Prop :=
  m + f = 1800 ∧
  f = m / 2 ∧
  maleA = 2 * m / 3 ∧
  femaleA = 3 * f / 4 ∧
  maleB = m - maleA ∧
  femaleB = f - femaleA ∧
  totalMale = maleA + maleB ∧
  totalFemale = femaleA + femaleB ∧
  totalA = maleA + femaleA ∧
  totalB = maleB + femaleB ∧
  chi_squared = (n * (maleA * femaleB - maleB * femaleA) ^ 2) / (totalMale * totalFemale * totalA * totalB) ∧
  alpha = 0.001 ∧
  chi_squared > 10.828

/-- Define the conditions for expected vaccination cost calculation  --/
def vaccination_conditions : Prop :=
  doseCost = 9 ∧
  probOneDoseSucceeds = 2 / 3 ∧
  cost = 27 * (20 / 27) + 54 * (7 / 27)

-- Problem proof statements
theorem disease_type_related_to_gender (h : conditions 1800 1200 600 800 400 450 150 1200 600 1250 550 (2 / 3) 9 0.001 13.09 34) :
  chi_squared > 10.828 := by sorry

theorem expected_vaccine_cost (h : vaccination_conditions 9 (2 / 3) 34) : cost = 34 := by sorry

end DiseaseAnalysis

end disease_type_related_to_gender_expected_vaccine_cost_l793_793657


namespace find_line_l_l793_793302

theorem find_line_l (A B M N : ℝ × ℝ)
  (h1 : ∃ x1 y1 x2 y2, A = (x1, y1) ∧ B = (x2, y2) ∧ x1 > 0 ∧ y1 > 0 ∧ x2 > 0 ∧ y2 > 0)
  (h2 : M = (x1, 0))
  (h3 : N = (0, y2))
  (h4 : abs (| M.1 - A.1 |) = abs (| N.2 - B.2 |))
  (h5 : dist M N = 2 * sqrt 3) :
  ∃ k m : ℝ, k < 0 ∧ m > 0 ∧ (∀ x y : ℝ, (y = k * x + m ↔ x + sqrt 2 * y - 2 * sqrt 2 = 0)) := sorry

end find_line_l_l793_793302


namespace quadratic_roots_l793_793432

theorem quadratic_roots (m : ℝ) (h1 : m > 4) :
  (∃ x y : ℝ, x ≠ y ∧ (m-5) * x^2 - 2 * (m + 2) * x + m = 0 ∧ (m-5) * y^2 - 2 * (m + 2) * y + m = 0)
  ∨ (m = 5 ∧ ∃ x : ℝ, (m-5) * x^2 - 2 * (m + 2) * x + m = 0)
  ∨ (¬((∃ x y : ℝ, x ≠ y ∧ (m-5) * x^2 - 2 * (m + 2) * x + m = 0) ∨ (m = 5 ∧ ∃ x : ℝ, (m-5) * x^2 - 2 * (m + 2) * x + m = 0))) :=
by
  sorry

end quadratic_roots_l793_793432


namespace calculation_correct_l793_793186

theorem calculation_correct :
  (3 + 4) * (3^2 + 4^2) * (3^4 + 4^4) * (3^8 + 4^8) * (3^16 + 4^16) * (3^32 + 4^32) * (3^64 + 4^64) = 4^128 - 3^128 :=
by
  sorry

end calculation_correct_l793_793186


namespace sum_exterior_angles_pentagon_l793_793976

theorem sum_exterior_angles_pentagon
  (polygon_exterior_angle_sum : ∀ (n : ℕ) (hn : 3 ≤ n), (n : ℝ) = 5 → ∑ i in (finset.range n), exterior_angle (polygon i) = 360) :
  ∑ i in (finset.range 5), exterior_angle (polygon i) = 360 := 
by
  sorry

end sum_exterior_angles_pentagon_l793_793976


namespace how_many_times_l793_793810

theorem how_many_times (a b : ℝ) (h1 : a = 0.5) (h2 : b = 0.01) : a / b = 50 := 
by 
  sorry

end how_many_times_l793_793810


namespace travel_from_A_to_C_l793_793073

def num_ways_A_to_B : ℕ := 5 + 2  -- 5 buses and 2 trains
def num_ways_B_to_C : ℕ := 3 + 2  -- 3 buses and 2 ferries

theorem travel_from_A_to_C :
  num_ways_A_to_B * num_ways_B_to_C = 35 :=
by
  -- The proof environment will be added here. 
  -- We include 'sorry' here for now.
  sorry

end travel_from_A_to_C_l793_793073


namespace trihedral_angle_bisectors_are_coplanar_l793_793533

variables {V : Type*} [inner_product_space ℝ V]

theorem trihedral_angle_bisectors_are_coplanar
  (a b c : V) :
  (a + b) + (c - a) = b + c :=
by sorry

end trihedral_angle_bisectors_are_coplanar_l793_793533


namespace arctan_sum_l793_793199

theorem arctan_sum (a b : ℝ) (h1 : a = 3) (h2 : b = 7) : 
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 := 
by 
  rw [h1, h2]
  sorry

end arctan_sum_l793_793199


namespace harrys_age_l793_793893

-- Definitions of the ages
variable (Kiarra Bea Job Figaro Harry : ℕ)

-- Given conditions
variable (h1 : Kiarra = 2 * Bea)
variable (h2 : Job = 3 * Bea)
variable (h3 : Figaro = Job + 7)
variable (h4 : Harry = Figaro / 2)
variable (h5 : Kiarra = 30)

-- The statement to prove
theorem harrys_age : Harry = 26 := sorry

end harrys_age_l793_793893


namespace min_bola_na_trave_count_max_bola_na_trave_count_specific_bola_na_trave_count_1_specific_bola_na_trave_count_2_l793_793858

theorem min_bola_na_trave_count:
    (s: finset ℕ) (h: s = {1, 2, 3, 4, 5, 6}) :
    count_bola_na_trave(s) = 6 := sorry

theorem max_bola_na_trave_count:
    (s: finset ℕ) (h: s = {3, 6, 9, 12, 15, 18}) :
    count_bola_na_trave(s) = 728 := sorry

theorem specific_bola_na_trave_count_1:
    (s: finset ℕ) (h: s = {2, 3, 8, 11, 14, 17}) :
    count_bola_na_trave(s) = 485 := sorry

theorem specific_bola_na_trave_count_2:
    (s: finset ℕ) (h: s = {8, 10, 12, 14, 16, 18}) :
    count_bola_na_trave(s) = 376 := sorry

end min_bola_na_trave_count_max_bola_na_trave_count_specific_bola_na_trave_count_1_specific_bola_na_trave_count_2_l793_793858


namespace adam_candies_l793_793023

theorem adam_candies:
  ∃ (A : ℕ), let J := 3 * A, let R := 4 * J in A + J + R = 96 ∧ A = 6 :=
sorry

end adam_candies_l793_793023


namespace find_y_value_l793_793414

-- Define the given conditions and the final question in Lean
theorem find_y_value (k : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : y = k * x ^ (1/3)) 
  (h2 : y = 4 * real.sqrt 3)
  (x1 : x = 64) 
  : ∃ k, y = 2 * real.sqrt 3 :=
sorry

end find_y_value_l793_793414


namespace find_natural_n_for_sum_of_squares_l793_793145

theorem find_natural_n_for_sum_of_squares (n : ℕ) (h_n : n > 1) (p : ℕ) (h_p_prime : Nat.Prime p) (k : ℕ) :
  ∑ i in (Finset.range (n + 1)).filter (λ x, x > 1), i^2 = p^k ↔ n = 2 :=
by
  sorry

end find_natural_n_for_sum_of_squares_l793_793145


namespace fill_missing_digits_l793_793740

noncomputable def first_number (a : ℕ) : ℕ := a * 1000 + 2 * 100 + 5 * 10 + 7
noncomputable def second_number (b c : ℕ) : ℕ := 2 * 1000 + b * 100 + 9 * 10 + c

theorem fill_missing_digits (a b c : ℕ) : a = 1 ∧ b = 5 ∧ c = 6 → first_number a + second_number b c = 5842 :=
by
  intros
  sorry

end fill_missing_digits_l793_793740


namespace surface_area_of_circumscribed_sphere_l793_793833

noncomputable def length : ℝ := Real.sqrt 3
noncomputable def width : ℝ := Real.sqrt 2
def height : ℝ := 1

theorem surface_area_of_circumscribed_sphere : 
  let diagonal := Real.sqrt (length ^ 2 + width ^ 2 + height ^ 2)
  let radius := diagonal / 2
  4 * Real.pi * radius ^ 2 = 6 * Real.pi := 
  by 
    sorry

end surface_area_of_circumscribed_sphere_l793_793833


namespace find_n_l793_793641

theorem find_n (n : ℕ) (h : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^28) : n = 27 :=
sorry

end find_n_l793_793641


namespace simplify_cube_roots_l793_793026

-- Definition for cube root
noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

-- Useful lemmas
lemma cube_root_mul (x y : ℝ) : cube_root (x * y) = cube_root x * cube_root y :=
  by sorry

lemma cube_root_27 : cube_root 27 = 3 :=
  by sorry

-- Main theorem statement
theorem simplify_cube_roots :
  cube_root (27 - 8) * cube_root (9 - cube_root 27) = cube_root 114 :=
begin
  have h1 : cube_root (27 - 8) = cube_root 19,
  { sorry },
  have h2 : cube_root (9 - cube_root 27) = cube_root 6,
  { sorry },
  rw [h1, h2],
  exact cube_root_mul 19 6,
end

end simplify_cube_roots_l793_793026


namespace count_parallelograms_l793_793677

theorem count_parallelograms :
  let area := 500000
  let lattice_points_condition (b d : Int) (m : Int) := b > 0 ∧ d > 0 ∧ m > 2 ∧ {
    y_B := 2 * b,
    y_D := m * d,
    area_equation := b * (m - 1) * d - b^2 = area
  }
  ∑ d in divisors 500000, 
  count_valid_pairs := 420 :=
by
  sorry

end count_parallelograms_l793_793677


namespace complement_of_A_in_U_eq_l793_793331

def U : Set ℝ := {x | x > 0}
def A : Set ℝ := {x | x ≥ Real.exp 1}
def complement_U_A : Set ℝ := {x | 0 < x ∧ x ≤ Real.exp 1}

theorem complement_of_A_in_U_eq : 
  (U \ A) = complement_U_A := 
by
  sorry

end complement_of_A_in_U_eq_l793_793331


namespace count_two_digit_numbers_less_35_l793_793367

open Nat

theorem count_two_digit_numbers_less_35 : 
  let two_digit_numbers := (finset.Ico 10 100) -- The range of two-digit numbers
  let count_satisfying := two_digit_numbers.filter (λ n, (n / 10 < 3 ∨ n % 10 < 5)).card
  count_satisfying = 55 :=
by
  -- Placeholder for the actual proof.
  sorry

end count_two_digit_numbers_less_35_l793_793367


namespace lucy_total_earnings_l793_793003

def earnings_in_one_cycle : ℕ := 1 + 2 + 3 + 4 + 5 + 6

def complete_cycles (total_hours : ℕ) (hours_per_cycle : ℕ) : ℕ :=
  total_hours / hours_per_cycle

def remaining_hours (total_hours : ℕ) (hours_per_cycle : ℕ) : ℕ :=
  total_hours % hours_per_cycle

def earnings_from_remaining_hours (remaining_hrs : ℕ) : ℕ :=
  if remaining_hrs = 0 then 0 else Finset.sum (Finset.range remaining_hrs) (λ x, x + 1)

theorem lucy_total_earnings (total_hours : ℕ) (hours_per_cycle : ℕ) (earnings_per_cycle : ℕ) : ℕ :=
  let cycles := complete_cycles total_hours hours_per_cycle
  let remaining := remaining_hours total_hours hours_per_cycle
  (cycles * earnings_per_cycle) + (earnings_from_remaining_hours remaining)

example : lucy_total_earnings 45 6 21 = 153 := by
  sorry

end lucy_total_earnings_l793_793003


namespace arctan_sum_l793_793201

theorem arctan_sum (x y : ℝ) (hx : x = 3) (hy : y = 7) :
  Real.arctan (x / y) + Real.arctan (y / x) = Real.pi / 2 := 
by
  rw [hx, hy]
  sorry

end arctan_sum_l793_793201


namespace price_per_goat_correct_l793_793524

open Classical

noncomputable def price_per_goat (G S goats_sold sheep_sold amount_from_goats : ℕ) 
  (g_ratio_s_ratio : 5 * S = 7 * G)
  (total_animals : G + S = 360)
  (half_goats : goats_sold = G / 2)
  (third_sheep : sheep_sold = (2 * S) / 3)
  (total_amount : 7200 = (sheep_sold * 30) + amount_from_goats)
  (goats_sold_val : goats_sold = 75)
  (sheep_sold_val : sheep_sold = 140) : ℕ :=
  amount_from_goats / goats_sold

theorem price_per_goat_correct : 
  ∃ G S, (5 * S = 7 * G) ∧ (G + S = 360) ∧ 
         (goats_sold = G / 2) ∧ (sheep_sold = (2 * S) / 3) ∧
         (7200 = (sheep_sold * 30) + amount_from_goats) ∧
         (goats_sold = 75) ∧ (sheep_sold = 140) →
         price_per_goat G S 75 140 3000 5 360 (G / 2) ((2 * S) / 3) 7200 75 140 = 40 := 
by {
  sorry
}

end price_per_goat_correct_l793_793524


namespace triangle_area_of_lines_l793_793619

theorem triangle_area_of_lines :
  let L1 (x : ℝ) := 3 * x - 6
  let L2 (x : ℝ) := -2 * x + 12
  let intersection := (18 / 5, 24 / 5)
  let y_intercept_L1 := (0, -6)
  let y_intercept_L2 := (0, 12)
  let base := 12 - (-6)
  let height := 18 / 5
  area(triangle(y_intercept_L1, y_intercept_L2, intersection)) = (1 / 2) * base * height :=
by
  sorry

end triangle_area_of_lines_l793_793619


namespace find_DE_length_l793_793470

noncomputable def triangle_DE_length (A B C D E : Type) (BC CD DE : ℝ) (C_angle : ℝ) : Prop :=
  BC = 40 ∧ C_angle = 45 ∧ CD = 20 ∧ DE = 20

theorem find_DE_length {A B C D E : Type} (BC CD DE : ℝ) (C_angle : ℝ) 
  (hBC : BC = 40) (hC_angle : C_angle = 45) (hCD : CD = 20) : DE = 20 :=
by {
  have hDE : DE = 20, sorry,
  exact hDE
}

end find_DE_length_l793_793470


namespace happy_snakes_not_purple_l793_793583

variables (Snakes : Type) -- Define the type for snakes
variables (Happy Purple CanAdd CannotSubtract : Snakes → Prop) -- Define the properties of snakes

-- Given conditions
axiom Happy_implies_CanAdd : ∀ s, Happy s → CanAdd s
axiom Purple_implies_CannotSubtract : ∀ s, Purple s → CannotSubtract s
axiom CannotSubtract_implies_CannotAdd : ∀ s, CannotSubtract s → ¬ CanAdd s

-- Goal: Prove that happy snakes are not purple
theorem happy_snakes_not_purple : ∀ s, Happy s → ¬ Purple s := 
by 
  intro s,
  intro hs,
  intro ps,
  have cs := Purple_implies_CannotSubtract s ps,
  have nca := CannotSubtract_implies_CannotAdd s cs,
  have ca := Happy_implies_CanAdd s hs,
  contradiction

-- sorry

end happy_snakes_not_purple_l793_793583


namespace count_valid_x_l793_793749

def is_valid_x (x : ℕ) : Prop := 
  (x % 5 = 0) ∧ 
  (121 < x ∧ x < 1331) ∧ 
  let base11_digits := x.digits 11 in
  (base11_digits.head! < base11_digits.last!)

theorem count_valid_x : 
  (finset.range 1332).filter is_valid_x .card = 99 :=
by sorry

end count_valid_x_l793_793749


namespace union_of_A_and_B_is_R_l793_793268

open Set Real

def A := {x : ℝ | log x > 0}
def B := {x : ℝ | x ≤ 1}

theorem union_of_A_and_B_is_R : A ∪ B = univ := by
  sorry

end union_of_A_and_B_is_R_l793_793268


namespace football_team_total_members_l793_793954

-- Definitions from the problem conditions
def initialMembers : ℕ := 42
def newMembers : ℕ := 17

-- Mathematical equivalent proof problem
theorem football_team_total_members : initialMembers + newMembers = 59 := by
  sorry

end football_team_total_members_l793_793954


namespace find_y_l793_793741

theorem find_y (y : ℚ) (h : ⌊y⌋ + y = 5) : y = 7 / 3 :=
sorry

end find_y_l793_793741


namespace syrup_ratio_l793_793483

def James_initial_coffee : ℝ := 14
def Janet_initial_coffee : ℝ := 14

def James_syrup_added : ℝ := 3
def Janet_syrup_added : ℝ := 4

def James_drank : ℝ := 3
def Janet_drank : ℝ := 3

def James_final_syrup : ℝ := 3
def Janet_final_syrup : ℝ := (4 - (3 * (4 / (14 + 4))))

theorem syrup_ratio :
  (James_final_syrup / Janet_final_syrup) = 9 / 10 :=
by
  simp,
  sorry

end syrup_ratio_l793_793483


namespace instrument_costs_purchasing_plans_l793_793655

variable (x y : ℕ)
variable (a b : ℕ)

theorem instrument_costs : 
  (2 * x + 3 * y = 1700 ∧ 3 * x + y = 1500) →
  x = 400 ∧ y = 300 := 
by 
  intros h
  sorry

theorem purchasing_plans :
  (x = 400) → (y = 300) → (3 * a + 10 = b) →
  (400 * a + 300 * b ≤ 30000) →
  ((760 - 400) * a + (540 - 300) * b ≥ 21600) →
  (a = 18 ∧ b = 64 ∨ a = 19 ∧ b = 67 ∨ a = 20 ∧ b = 70) :=
by
  intros hx hy hab hcost hprofit
  sorry

end instrument_costs_purchasing_plans_l793_793655


namespace estimate_total_fish_l793_793517

theorem estimate_total_fish (marked : ℕ) (sample_size : ℕ) (marked_in_sample : ℕ) (x : ℝ) 
  (h1 : marked = 50) 
  (h2 : sample_size = 168) 
  (h3 : marked_in_sample = 8) 
  (h4 : sample_size * 50 = marked_in_sample * x) : 
  x = 1050 := 
sorry

end estimate_total_fish_l793_793517


namespace length_de_l793_793478

open Triangle

/-- In triangle ABC, BC = 40 and ∠C = 45°. Let the perpendicular bisector
of BC intersect BC and AC at D and E, respectively. Prove that DE = 10√2. -/

theorem length_de 
  (ABC : Triangle)
  (B C : Point)
  (BC_40 : dist B C = 40)
  (angle_C_45 : ∠(B, C, AC) = 45)
  (D : Point)
  (midpoint_D : is_midpoint B C D)
  (E : Point)
  (intersection_E : is_perpendicular_bisector_intersection B C D E) :
  dist D E = 10 * real.sqrt 2 :=
begin
  -- proof steps would go here
  sorry
end

end length_de_l793_793478


namespace sum_of_numerator_and_denominator_l793_793110

def repeating_decimal_represents_fraction : Prop :=
  let x := 34 / 99
  0.34̅ = x

/-- The sum of the numerator and denominator of 0.34̅ in its lowest terms is 133. -/
theorem sum_of_numerator_and_denominator
  (x : ℚ)
  (hx : repeating_decimal_represents_fraction) :
  (34 + 99 = 133) :=
begin
  sorry
end

end sum_of_numerator_and_denominator_l793_793110


namespace magnitude_b_l793_793836

noncomputable theory

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Conditions
def mag_a : real := ‖a‖ = 3
def mag_diff_ab : real := ‖a - b‖ = 5
def dot_ab : real := inner a b = 1

-- Goal
theorem magnitude_b (h1 : ‖a‖ = 3) (h2 : ‖a - b‖ = 5) (h3 : inner a b = 1) : ‖b‖ = 3 * real.sqrt 2 :=
by 
  sorry

end magnitude_b_l793_793836


namespace smallest_n_l793_793827

def n_expr (n : ℕ) : ℕ :=
  n * (2^7) * (3^2) * (7^3)

theorem smallest_n (n : ℕ) (h1: 25 ∣ n_expr n) (h2: 27 ∣ n_expr n) : n = 75 :=
sorry

end smallest_n_l793_793827


namespace rectangle_diagonal_angles_l793_793261

theorem rectangle_diagonal_angles 
    (A B C D : ℤ × ℤ) -- Vertices of the rectangle on integer coordinates
    (is_rect : ∃ O, (O = (A + C) / 2) ∧ (O = (B + D) / 2)) -- condition of rectangle
    (α : ℝ) -- angle between diagonals
    (cos_α_tan_α_rational : 
        (∃ cos_α : ℚ, cos_α = real.cos α) ∧ 
        (∃ tan_α : ℚ, tan_α = real.tan α)) :

    (∃ cos_α : ℚ, cos_α = real.cos α) ∧ 
    (∃ sin_α : ℚ, sin_α = real.sin α) :=
sorry

end rectangle_diagonal_angles_l793_793261


namespace expected_reachable_cells_l793_793577

theorem expected_reachable_cells :
  ∀ (height : ℕ) (inf_dir : bool) (locked_prob : ℚ), 
  height = 2 → inf_dir = tt → locked_prob = 1 / 2 →
  ExpectedCells (height, inf_dir, locked_prob) = 32 / 7 :=
by
  intros height inf_dir locked_prob h1 h2 h3
  sorry

end expected_reachable_cells_l793_793577


namespace sum_of_numerator_and_denominator_l793_793115

def repeating_decimal_represents_fraction : Prop :=
  let x := 34 / 99
  0.34̅ = x

/-- The sum of the numerator and denominator of 0.34̅ in its lowest terms is 133. -/
theorem sum_of_numerator_and_denominator
  (x : ℚ)
  (hx : repeating_decimal_represents_fraction) :
  (34 + 99 = 133) :=
begin
  sorry
end

end sum_of_numerator_and_denominator_l793_793115


namespace line_through_point_with_slope_l793_793048

theorem line_through_point_with_slope (x y : ℝ) (h : y - 2 = -3 * (x - 1)) : 3 * x + y - 5 = 0 :=
sorry

example : 3 * 1 + 2 - 5 = 0 := by sorry

end line_through_point_with_slope_l793_793048


namespace two_digit_numbers_count_l793_793339

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem two_digit_numbers_count :
  { n : ℕ | 10 ≤ n ∧ n ≤ 99 ∧ (tens_digit n < 3 ∨ units_digit n < 5) }.to_finset.card = 55 :=
by
  sorry

end two_digit_numbers_count_l793_793339


namespace convert_and_subtract_l793_793724

-- Condition definitions
def B : ℕ := 11
def one : ℕ := 1
def F : ℕ := 15

-- Lean statement for the proof problem
theorem convert_and_subtract : 
  let base10_value := B * 16^2 + one * 16 + F in
  base10_value - 432 = 2415 := 
by sorry

end convert_and_subtract_l793_793724


namespace dimension_of_V_l793_793941

/-- Definition of a polynomial being balanced. -/
def is_balanced (p : ℝ[X][X]) : Prop :=
  ∀ r, average_value_on_circle_centered_at_origin p r = 0

/-- Vector space of balanced polynomials of degree at most 2009 -/
def V : Type := {p : ℝ[X][X] // p.degree ≤ 2009 ∧ is_balanced p}

/-- Statement that the dimension of V is equal to 2020050 -/
theorem dimension_of_V : fintype.card V = 2020050 := 
sorry

end dimension_of_V_l793_793941


namespace repeating_decimal_sum_l793_793104

theorem repeating_decimal_sum (x : ℚ) (h : x = 34 / 999) : x.num + x.denom = 1033 := by 
  sorry

end repeating_decimal_sum_l793_793104


namespace conjugate_of_z_l793_793284

open complex

theorem conjugate_of_z (z : ℂ) (h : (z - 3*I)*(2 + I) = 5*I) : conj z = 2 - 5*I :=
sorry

end conjugate_of_z_l793_793284


namespace part1_question1_part1_question2_part2_l793_793735

-- Part 1
theorem part1_question1 (x : ℕ) (h : 5 ≤ x ∧ x ≤ 12) :
  let price_per_plant := (-0.3 : ℝ) * x + 4.5 in
  let price_per_pot := (-0.3 : ℝ) * (x * x) + 4.5 * x in
  price_per_plant = (-0.3 : ℝ) * x + 4.5 ∧ price_per_pot = (-0.3 : ℝ) * (x * x) + 4.5 * x :=
by
  -- proof would go here
  sorry

theorem part1_question2 (x : ℕ) (hx : -0.3 * (x * x) + 4.5 * x = 16.2) :
  x = 6 ∨ x = 9 :=
by
  -- proof would go here
  sorry

-- Part 2
theorem part2 (x : ℕ) :
  (5 ≤ x ∧ x ≤ 12 ∧ 30 * (-0.3 * x * x + 4.5 * x) - 40 * (2 + 0.3 * x) = 100)
  ∨ (12 < x ∧ x ≤ 18 ∧ 30 * 0.8 * x - 40 * (2 + 0.3 * x) = 100) →
  x = 12 ∨ x = 15 :=
by
  -- proof would go here
  sorry

end part1_question1_part1_question2_part2_l793_793735


namespace find_set_C_l793_793649

def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 2 = 0}
def C : Set ℝ := {a | B a ⊆ A}

theorem find_set_C : C = {0, 1, 2} :=
by
  sorry

end find_set_C_l793_793649


namespace greatest_x_satisfies_f_integer_l793_793088

def f (x : ℤ) : ℚ :=
  (x^2 + 2*x + 5) / (x - 3)

theorem greatest_x_satisfies_f_integer : 
  ∀ x : ℤ, (∃ k : ℤ, f x = k) → x ≤ 23 :=
begin
  -- Proof would be here
  sorry
end

end greatest_x_satisfies_f_integer_l793_793088


namespace find_investment_period_l793_793940

variable (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ)

theorem find_investment_period (hP : P = 12000)
                               (hr : r = 0.10)
                               (hn : n = 2)
                               (hA : A = 13230) :
                               ∃ t : ℝ, A = P * (1 + r / n)^(n * t) ∧ t = 1 := 
by
  sorry

end find_investment_period_l793_793940


namespace equilateral_triangle_l793_793975

theorem equilateral_triangle (a b c : ℝ) (h1 : b^2 = a * c) (h2 : 2 * b = a + c) : a = b ∧ b = c ∧ a = c := by
  sorry

end equilateral_triangle_l793_793975


namespace partI_OM_line_eq_partII_min_dist_l793_793875

noncomputable def polarToCartesian (r θ : ℝ) : ℝ × ℝ := (r * (Real.cos θ), r * (Real.sin θ))

structure Point :=
  (x : ℝ)
  (y : ℝ)

def M_polar : Point := ⟨4 * Real.sqrt 2 * Real.cos (Real.pi / 4), 4 * Real.sqrt 2 * Real.sin (Real.pi / 4)⟩

def C_parametric (α : ℝ) : Point := ⟨1 + Real.sqrt 2 * Real.cos α, Real.sqrt 2 * Real.sin α⟩

theorem partI_OM_line_eq : ∃ k b : ℝ, ∀ x : ℝ, (M_polar = ⟨x, k * x + b⟩) :=
by
  sorry

theorem partII_min_dist : ∃ d : ℝ, d = 5 - Real.sqrt 2 :=
by
  sorry

end partI_OM_line_eq_partII_min_dist_l793_793875


namespace arctan_sum_l793_793192

theorem arctan_sum : ∀ (a b : ℝ), 
  a = 3/7 → 
  b = 7/3 → 
  a * b = 1 → 
  a > 0 → 
  b > 0 → 
  Real.arctan a + Real.arctan b = Real.pi / 2 :=
by intros a b ha hb hab ha_pos hb_pos
   rw [ha, hb, hab]
   sorry

end arctan_sum_l793_793192


namespace height_of_third_tree_l793_793726

variables 
  (first_tree_height : ℕ) (first_tree_branches : ℕ)
  (second_tree_height : ℕ) (second_tree_branches : ℕ)
  (third_tree_branches : ℕ)
  (fourth_tree_height : ℕ) (fourth_tree_branches : ℕ)
  (average_branches_per_foot : ℕ)
  
variables 
  (first_tree_height_is : first_tree_height = 50)
  (first_tree_branches_is : first_tree_branches = 200)
  (second_tree_height_is : second_tree_height = 40)
  (second_tree_branches_is : second_tree_branches = 180)
  (third_tree_branches_is : third_tree_branches = 180)
  (fourth_tree_height_is : fourth_tree_height = 34)
  (fourth_tree_branches_is : fourth_tree_branches = 153)
  (average_branches_per_foot_is : average_branches_per_foot = 4)

theorem height_of_third_tree :
  third_tree_branches / average_branches_per_foot = 45 :=
by { rw [third_tree_branches_is, average_branches_per_foot_is], norm_num, }

end height_of_third_tree_l793_793726


namespace candy_cost_per_box_l793_793213

def ticket_cost : ℚ := 10
def combo_meal_cost : ℚ := 11
def total_spending : ℚ := 36
def num_boxes_candy : ℕ := 2

theorem candy_cost_per_box :
  let total_tickets_cost := 2 * ticket_cost in
  let total_cost := total_tickets_cost + combo_meal_cost in
  let remaining_cost := total_spending - total_cost in
  (remaining_cost / num_boxes_candy) = 2.50 :=
by
  sorry

end candy_cost_per_box_l793_793213


namespace happy_snakes_not_purple_l793_793582

variable {Snake : Type}
variable [Fintype Snake]

-- Number of snakes
constant num_snakes : ℕ
constant purple_snakes : Finset Snake
constant happy_snakes : Finset Snake

-- Initial conditions
axiom h1 : num_snakes = 13
axiom h2 : purple_snakes.card = 4
axiom h3 : happy_snakes.card = 5

-- Predicate definitions
def CanAdd (s : Snake) : Prop := sorry
def CanSubtract (s : Snake) : Prop := sorry
def IsHappy (s : Snake) : Prop := s ∈ happy_snakes
def IsPurple (s : Snake) : Prop := s ∈ purple_snakes

-- Logical implications
axiom h4 : ∀ s, IsHappy s → CanAdd s
axiom h5 : ∀ s, IsPurple s → ¬ CanSubtract s
axiom h6 : ∀ s, ¬ CanSubtract s → ¬ CanAdd s

-- To prove: Happy snakes are not purple
theorem happy_snakes_not_purple : ∀ s, IsHappy s → ¬ IsPurple s :=
by sorry

end happy_snakes_not_purple_l793_793582


namespace prove_length_of_bridge_l793_793149

-- Define the given conditions as constants
constant speed_of_train : ℝ := 15 -- meters/second
constant time_to_cross_bridge : ℝ := 480 -- seconds
constant length_of_train : ℝ := 600 -- meters

-- Define the total distance traveled as per the conditions
def total_distance_traveled := speed_of_train * time_to_cross_bridge

-- Define the length of the bridge
def length_of_bridge := total_distance_traveled - length_of_train

-- The goal statement
theorem prove_length_of_bridge : length_of_bridge = 6600 := by
  -- The proof is omitted as per instructions
  sorry

end prove_length_of_bridge_l793_793149


namespace correct_operation_is_a_l793_793130

theorem correct_operation_is_a (a b : ℝ) : 
  (a^4 * a^3 = a^7) ∧ 
  ((a^2)^3 ≠ a^5) ∧ 
  (3 * a^2 - a^2 ≠ 2) ∧ 
  ((a - b)^2 ≠ a^2 - b^2) := 
by {
  -- Here, you would fill in the proof
  sorry
}

end correct_operation_is_a_l793_793130


namespace cos_theta_sum_eq_2016_div_2017_l793_793791

-- Definitions based on conditions
def f (x : ℝ) : ℝ := 1 / (x + 1)
def O : ℝ × ℝ := (0, 0)
def A (n : ℕ) (hn : n > 0) : ℝ × ℝ := (n, f n)
def a : ℝ × ℝ := (0, 1)

def θ (n : ℕ) (hn : n > 0) : ℝ := 
  let OA_n := A n hn
  real.arccos ((OA_n.1 * a.1 + OA_n.2 * a.2) /
              (real.sqrt (OA_n.1^2 + OA_n.2^2) * real.sqrt (a.1^2 + a.2^2)))

-- Lean Statement
theorem cos_theta_sum_eq_2016_div_2017 :
  (finset.range 2016).sum (λ n, 
    let hn := nat.succ_pos n
    in
    (real.cos (θ n.succ hn) / real.sin (θ n.succ hn))) = 2016 / 2017 :=
by
  sorry

end cos_theta_sum_eq_2016_div_2017_l793_793791


namespace min_value_of_f_l793_793747

noncomputable def f (θ : ℝ) : ℝ := cos (θ / 2) * (1 - sin θ)

theorem min_value_of_f :
  ∃ θ, (π / 2 < θ ∧ θ < 3 * π / 2) ∧
  ∀ θ, (π / 2 < θ ∧ θ < 3 * π / 2) → f θ ≥ (sqrt 3 / 2) - (3 / 4) :=
sorry

end min_value_of_f_l793_793747


namespace arctan_sum_l793_793194

theorem arctan_sum : ∀ (a b : ℝ), 
  a = 3/7 → 
  b = 7/3 → 
  a * b = 1 → 
  a > 0 → 
  b > 0 → 
  Real.arctan a + Real.arctan b = Real.pi / 2 :=
by intros a b ha hb hab ha_pos hb_pos
   rw [ha, hb, hab]
   sorry

end arctan_sum_l793_793194


namespace part1_m_neg1_part2_B_subset_A_l793_793802

-- Define the sets A and B
def setA : Set ℝ := {x : ℝ | x > 1}
def setB (m : ℝ) : Set ℝ := {x : ℝ | m ≤ x ∧ x ≤ m + 3}

-- Define the problem statements
theorem part1_m_neg1 :
  let m := -1 in
  setB m = {x : ℝ | -1 ≤ x ∧ x ≤ 2} ∧
  setA ∩ setB m = {x : ℝ | 1 < x ∧ x ≤ 2} ∧
  setA ∪ setB m = {x : ℝ | x ≥ -1} :=
by
  sorry

theorem part2_B_subset_A (m : ℝ) :
  (setB m ⊆ setA) ↔ (m > 1) :=
by
  sorry

end part1_m_neg1_part2_B_subset_A_l793_793802


namespace max_sum_square_pyramid_addition_l793_793165

def square_pyramid_addition_sum (faces edges vertices : ℕ) : ℕ :=
  let new_faces := faces - 1 + 4
  let new_edges := edges + 4
  let new_vertices := vertices + 1
  new_faces + new_edges + new_vertices

theorem max_sum_square_pyramid_addition :
  square_pyramid_addition_sum 6 12 8 = 34 :=
by
  sorry

end max_sum_square_pyramid_addition_l793_793165


namespace lyssa_incorrect_percentage_l793_793917

theorem lyssa_incorrect_percentage (T : ℕ) (P_incorrect : ℕ) (L_more_correct : ℕ) 
  (hTotal : T = 75)
  (hP_incorrect : P_incorrect = 12)
  (hL_more_correct : L_more_correct = 3) :
  let P_correct := T - P_incorrect in
  let L_correct := P_correct + L_more_correct in
  let L_incorrect := T - L_correct in
  (L_incorrect * 100) / T = 12 :=
by
  sorry

end lyssa_incorrect_percentage_l793_793917


namespace ratio_problem_l793_793626

theorem ratio_problem (x : ℕ) : (20 / 1 : ℝ) = (x / 10 : ℝ) → x = 200 := by
  sorry

end ratio_problem_l793_793626


namespace f_is_odd_k_range_condition_l793_793765

def f (x : ℝ) : ℝ := log (1 + x) / log 3 - log (1 - x) / log 3

def g (x k : ℝ) : ℝ := 2 * log ((1 + x) / k) / log 3

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  sorry

theorem k_range_condition (x k : ℝ) (h₁ : 1 / 3 ≤ x) (h₂ : x ≤ 1 / 2) (h₃ : 0 < k) :
  log ((1 + x) / (1 - x)) / log 3 ≥ 2 * log ((1 + x) / k) / log 3 → k ≥ sqrt 3 / 2 := by
  sorry

end f_is_odd_k_range_condition_l793_793765


namespace arctan_sum_l793_793202

theorem arctan_sum (x y : ℝ) (hx : x = 3) (hy : y = 7) :
  Real.arctan (x / y) + Real.arctan (y / x) = Real.pi / 2 := 
by
  rw [hx, hy]
  sorry

end arctan_sum_l793_793202


namespace sum_of_inscribed_angles_l793_793963

theorem sum_of_inscribed_angles 
  (n : ℕ) 
  (total_degrees : ℝ)
  (arcs : ℕ)
  (x_arcs : ℕ)
  (y_arcs : ℕ) 
  (arc_angle : ℝ)
  (x_central_angle : ℝ)
  (y_central_angle : ℝ)
  (x_inscribed_angle : ℝ)
  (y_inscribed_angle : ℝ)
  (total_inscribed_angles : ℝ) :
  n = 18 →
  total_degrees = 360 →
  x_arcs = 3 →
  y_arcs = 5 →
  arc_angle = total_degrees / n →
  x_central_angle = x_arcs * arc_angle →
  y_central_angle = y_arcs * arc_angle →
  x_inscribed_angle = x_central_angle / 2 →
  y_inscribed_angle = y_central_angle / 2 →
  total_inscribed_angles = x_inscribed_angle + y_inscribed_angle →
  total_inscribed_angles = 80 := sorry

end sum_of_inscribed_angles_l793_793963


namespace total_worksheets_to_grade_l793_793168

theorem total_worksheets_to_grade (problems_per_worksheet : ℕ) (worksheets_graded : ℕ) (remaining_problems : ℕ) :
  problems_per_worksheet = 4 →
  worksheets_graded = 5 →
  remaining_problems = 16 →
  worksheets_graded + (remaining_problems / problems_per_worksheet) = 9 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_worksheets_to_grade_l793_793168


namespace remainder_of_55_pow_55_plus_15_mod_8_l793_793061

theorem remainder_of_55_pow_55_plus_15_mod_8 :
  (55^55 + 15) % 8 = 6 := by
  -- This statement does not include any solution steps.
  sorry

end remainder_of_55_pow_55_plus_15_mod_8_l793_793061


namespace gcd_set_divisors_of_integer_l793_793499

theorem gcd_set_divisors_of_integer (a b c d : ℕ) (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_not_eq : a * d ≠ b * c) (h_gcd : Nat.gcd (Nat.gcd a b) (Nat.gcd c d) = 1) :
  ∃ m : ℕ, ∀ n : ℕ, 0 < n → (∃ d, d ∣ m ∧ d = Nat.gcd (a * n + b) (c * n + d)) :=
begin
  sorry
end

end gcd_set_divisors_of_integer_l793_793499


namespace original_price_l793_793162

theorem original_price (SP : ℝ) (rate_of_profit : ℝ) (CP : ℝ) 
  (h1 : SP = 60) 
  (h2 : rate_of_profit = 0.20) 
  (h3 : SP = CP * (1 + rate_of_profit)) : CP = 50 := by
  sorry

end original_price_l793_793162


namespace value_of_y_at_x8_l793_793408

theorem value_of_y_at_x8
  (k : ℝ)
  (y : ℝ → ℝ)
  (hx64 : y 64 = 4 * Real.sqrt 3)
  (hy_def : ∀ x, y x = k * x^(1 / 3)) :
  y 8 = 2 * Real.sqrt 3 :=
by {
  sorry,
}

end value_of_y_at_x8_l793_793408


namespace negation_proposition_equivalence_l793_793933

theorem negation_proposition_equivalence : 
    (¬ ∃ x_0 : ℝ, (x_0^2 + 1 > 0) ∨ (x_0 > Real.sin x_0)) ↔ 
    (∀ x : ℝ, (x^2 + 1 ≤ 0) ∧ (x ≤ Real.sin x)) :=
by 
    sorry

end negation_proposition_equivalence_l793_793933


namespace geometric_sequence_properties_l793_793782

variables {n : ℕ} -- Natural number n for sequence terms
variables {a : ℕ → ℕ} -- Function definition a for sequence

-- Conditions
def is_geometric_sequence (a : ℕ → ℕ) (q : ℝ) := ∀ (n : ℕ), q > 0 ∧ a 1 = 1 ∧ a (n + 1) = a n * q
def condition := 4 * a 3 = a 2 * a 4

-- Correctness of common ratio q and a3 value
def common_ratio (q : ℝ) := q = 2
def a3_value := a 3 = 4

-- Sn and an for second part of the problem
def a_n (n : ℕ) : ℝ := 2 ^ (n - 1)
def S_n (n : ℕ) : ℝ := (2 ^ n - 1)

theorem geometric_sequence_properties (a : ℕ → ℕ) (q : ℝ) (n : ℕ) :
  is_geometric_sequence a q ∧ condition →
  common_ratio q ∧ a 3 = 4 ∧ (∃ Sn : ℕ, Sn = S_n n ∧ ∀ n, S_n n / a_n n < 2) := by
  sorry

end geometric_sequence_properties_l793_793782


namespace statement_B_statement_C_l793_793903

open Complex

theorem statement_B (z_1 : ℂ) : (z_1 - conj z_1).im = (2 * Complex.i * z_1.im).im :=
by sorry

theorem statement_C (z_1 z_2 : ℂ) : abs (z_1 + z_2) ≤ abs z_1 + abs z_2 :=
by sorry

end statement_B_statement_C_l793_793903


namespace add_words_to_meet_requirement_l793_793134

-- Definitions required by the problem
def yvonne_words : ℕ := 400
def janna_extra_words : ℕ := 150
def words_removed : ℕ := 20
def requirement : ℕ := 1000

-- Derived values based on the conditions
def janna_words : ℕ := yvonne_words + janna_extra_words
def initial_words : ℕ := yvonne_words + janna_words
def words_after_removal : ℕ := initial_words - words_removed
def words_added : ℕ := 2 * words_removed
def total_words_after_editing : ℕ := words_after_removal + words_added
def words_to_add : ℕ := requirement - total_words_after_editing

-- The theorem to prove
theorem add_words_to_meet_requirement : words_to_add = 30 := by
  sorry

end add_words_to_meet_requirement_l793_793134


namespace find_line_l_l793_793305

theorem find_line_l (A B M N : ℝ × ℝ)
  (h1 : ∃ x1 y1 x2 y2, A = (x1, y1) ∧ B = (x2, y2) ∧ x1 > 0 ∧ y1 > 0 ∧ x2 > 0 ∧ y2 > 0)
  (h2 : M = (x1, 0))
  (h3 : N = (0, y2))
  (h4 : abs (| M.1 - A.1 |) = abs (| N.2 - B.2 |))
  (h5 : dist M N = 2 * sqrt 3) :
  ∃ k m : ℝ, k < 0 ∧ m > 0 ∧ (∀ x y : ℝ, (y = k * x + m ↔ x + sqrt 2 * y - 2 * sqrt 2 = 0)) := sorry

end find_line_l_l793_793305


namespace total_distance_is_23_total_distance_is_23_l793_793654

variable (start c1 c2 end : ℤ)

-- Define the initial and intermediary positions of the bug
axiom h1 : start = 4
axiom h2 : c1 = -3
axiom h3 : c2 = 7
axiom h4 : end = 1

-- Calculate the distances between the points
def distance1 : ℤ := abs (c1 - start)
def distance2 : ℤ := abs (c2 - c1)
def distance3 : ℤ := abs (end - c2)

-- Define the total distance travelled by the bug
def total_distance : ℤ := distance1 + distance2 + distance3

-- Proof that the total distance is 23 units
theorem total_distance_is_23 : total_distance start c1 c2 end = 23 := by
  rw [distance1, distance2, distance3]
  have : distance1 = 7 := by
    rw [h1, h2]
    simp
  have : distance2 = 10 := by
    rw [h2, h3]
    simp
  have : distance3 = 6 := by
    rw [h3, h4]
    simp
  rw [this_1, this_2, this_3]
  simp

-- Skipping the detailed proof, leaving it with "sorry"
theorem total_distance_is_23 : total_distance start c1 c2 end = 23 := sorry

end total_distance_is_23_total_distance_is_23_l793_793654


namespace arc_length_parametric_curve_l793_793141

noncomputable def arcLength (x y : ℝ → ℝ) (t1 t2 : ℝ) : ℝ :=
  ∫ t in t1..t2, Real.sqrt ((deriv x t)^2 + (deriv y t)^2)

theorem arc_length_parametric_curve :
    (∫ t in (0 : ℝ)..(3 * Real.pi), 
        Real.sqrt ((deriv (fun t => (t ^ 2 - 2) * Real.sin t + 2 * t * Real.cos t) t) ^ 2 +
                   (deriv (fun t => (2 - t ^ 2) * Real.cos t + 2 * t * Real.sin t) t) ^ 2)) =
    9 * Real.pi ^ 3 :=
by
  -- The proof is omitted
  sorry

end arc_length_parametric_curve_l793_793141


namespace repeating_decimal_sum_l793_793106

theorem repeating_decimal_sum (x : ℚ) (h : x = 34 / 999) : x.num + x.denom = 1033 := by 
  sorry

end repeating_decimal_sum_l793_793106


namespace max_take_home_pay_l793_793857

-- Define the income in thousand dollars
def income (x : ℝ) := 1000 * x

-- Define the tax rate as given in the problem
def tax_rate (x : ℝ) := (x / 2 + 1) / 100

-- Define the tax amount for income of x thousand dollars
def tax_amount (x : ℝ) := tax_rate x * income x

-- Define the take-home pay
def take_home_pay (x : ℝ) := income x - tax_amount x

-- Statement: The income of 99 thousand dollars yields the greatest take-home pay
theorem max_take_home_pay : take_home_pay 99 = take_home_pay 99000 :=
by
  -- Skipping proof
  sorry

end max_take_home_pay_l793_793857


namespace correct_choice_D_l793_793450

-- Definitions based on provided conditions
variables {Point : Type} {Line Plane : Type} [linear_space Point Line Plane]
variables {l m: Line} {α β: Plane}

-- Non-intersecting conditions
variables (H1 : non_intersecting_lines l m)
variables (H2 : non_intersecting_planes α β)

-- Relationship conditions for choice D
variables (H3 : l ⊆ α)
variables (H4 : l ∥ m)
variables (H5 : α ∥ β)

-- Theorem to prove choice D is correct
theorem correct_choice_D : l ⊆ α → l ∥ m → α ∥ β → m ⊆ β :=
by
  sorry

end correct_choice_D_l793_793450


namespace semiperimeter_inequality_l793_793934

theorem semiperimeter_inequality (p R r : ℝ) (hp : p ≥ 0) (hR : R ≥ 0) (hr : r ≥ 0) :
  p ≥ (3 / 2) * Real.sqrt (6 * R * r) :=
sorry

end semiperimeter_inequality_l793_793934


namespace num_suitable_two_digit_numbers_l793_793379

/-- 
How many two-digit numbers have at least one digit that is smaller than the corresponding digit in the number 35?
-/
theorem num_suitable_two_digit_numbers : 
  let two_digit_numbers := { n : ℕ | 10 ≤ n ∧ n ≤ 99 },
      suitable_numbers := { n ∈ two_digit_numbers | (n / 10 < 3) ∨ (n % 10 < 5) } in
  suitable_numbers.card = 55 :=
by
  sorry

end num_suitable_two_digit_numbers_l793_793379


namespace measure_angle_BPC_l793_793876

-- Define the vertices of the square ABCD and its side length
def square_side := 5
def A := (0, 0)
def B := (square_side, 0)
def C := (square_side, square_side)
def D := (0, square_side)

-- Define point E such that triangle ABE is isosceles with AB = BE and ∠ABE = 70°
def E := sorry  -- E's exact position needs to satisfy the conditions

-- Define lines BE and AC and their intersection P
def AC := sorry  -- Line AC (extend from A to C)
def BE := sorry  -- Line BE (extend from B to E)
def P := sorry  -- Intersection point P of lines BE and AC

-- Define point Q on BC such that PQ ⊥ BC
def Q := sorry  -- Q's exact position on BC

-- Define ∠BPC and prove it
def angle_BPC := 115  -- Given by problem's solution step
theorem measure_angle_BPC : ∠ B P C = 115°
by 
  sorry

end measure_angle_BPC_l793_793876


namespace hyperbola_asymptote_eccentricity_l793_793964

-- Problem statement: We need to prove that the eccentricity of hyperbola 
-- given the specific asymptote is sqrt(5).

noncomputable def calc_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_asymptote_eccentricity 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_asymptote : b = 2 * a) :
  calc_eccentricity a b = Real.sqrt 5 := 
by
  -- Insert the proof step here
  sorry

end hyperbola_asymptote_eccentricity_l793_793964


namespace weight_of_new_person_l793_793038

theorem weight_of_new_person (avg_increase : ℝ) (num_persons : ℕ) (old_weight new_weight : ℝ) 
  (h_avg_increase : avg_increase = 1.5) (h_num_persons : num_persons = 9) (h_old_weight : old_weight = 65) 
  (h_new_weight_increase : new_weight = old_weight + num_persons * avg_increase) : 
  new_weight = 78.5 :=
sorry

end weight_of_new_person_l793_793038


namespace value_of_y_at_x8_l793_793410

theorem value_of_y_at_x8
  (k : ℝ)
  (y : ℝ → ℝ)
  (hx64 : y 64 = 4 * Real.sqrt 3)
  (hy_def : ∀ x, y x = k * x^(1 / 3)) :
  y 8 = 2 * Real.sqrt 3 :=
by {
  sorry,
}

end value_of_y_at_x8_l793_793410


namespace sum_of_numerator_and_denominator_l793_793113

def repeating_decimal_represents_fraction : Prop :=
  let x := 34 / 99
  0.34̅ = x

/-- The sum of the numerator and denominator of 0.34̅ in its lowest terms is 133. -/
theorem sum_of_numerator_and_denominator
  (x : ℚ)
  (hx : repeating_decimal_represents_fraction) :
  (34 + 99 = 133) :=
begin
  sorry
end

end sum_of_numerator_and_denominator_l793_793113


namespace largest_integer_n_satisfies_conditions_l793_793239

noncomputable def max_valid_n : ℕ := 9897969594939291909

theorem largest_integer_n_satisfies_conditions :
  ∀ n : ℕ,
    (∀ (d : ℕ), (d < 10) → (∃ (ds : list ℕ), ds.length = n ∧ ∀ i < ds.length - 1, ds.nth i ≠ ds.nth (i + 1))) ∧
    (∀ a b : ℕ, (a ≠ b ∧ a < 10 ∧ b < 10) → ¬ ∃ (ds : list ℕ), ds.length = n ∧ removes_abab a b ds) →
    n ≤ max_valid_n :=
by
  --
  sorry

end largest_integer_n_satisfies_conditions_l793_793239


namespace contradiction_proof_real_root_l793_793075

theorem contradiction_proof_real_root (a b : ℝ) :
  (∀ x : ℝ, x^3 + a * x + b ≠ 0) → (∃ x : ℝ, x + a * x + b = 0) :=
sorry

end contradiction_proof_real_root_l793_793075


namespace angles_of_triangle_KCC_l793_793691

-- Define the given triangle vertices
variables (A B C C' B' K : Point)
variables [is_triangle A B C]
variables [is_isosceles_triangle A B C' (angle_measure 120)] -- Isosceles triangle with angle 120 degrees
variables [is_constructed_on_opposite_side A B C' C] -- C' is on the opposite side of AB to C
variables [is_equilateral_triangle A C B'] -- Equilateral triangle ACB'
variables [is_constructed_on_same_side A C B' C] -- B' is on the same side of AC as C
variables [is_midpoint B B' K] -- K is the midpoint of BB'

-- Prove that the angles of triangle KCC' are 30, 60, and 90 degrees
theorem angles_of_triangle_KCC' :
  angles_of_triangle K C C' = (30, 60, 90) :=
sorry

end angles_of_triangle_KCC_l793_793691


namespace horner_rule_operations_l793_793996

theorem horner_rule_operations :
  let f : ℤ → ℤ := λ x, 4 * x^5 - 3 * x^4 + 6 * x - 9 in
  ∀ x : ℤ, x = -3 → 
  (num_mul_ops f x = 5 ∧ num_add_ops f x = 3) :=
by
  sorry

-- Definition for counting multiplicative operations
def num_mul_ops (f : ℤ → ℤ) (x : ℤ) : ℕ := 5 -- provided directly by conditions without solving

-- Definition for counting additive operations
def num_add_ops (f : ℤ → ℤ) (x : ℤ) : ℕ := 3 -- provided directly by conditions without solving

end horner_rule_operations_l793_793996


namespace number_2digit_smaller_than_35_l793_793347

/--
Prove that the number of two-digit numbers where at least one digit
is smaller than the corresponding digit in 35 is exactly 55.
-/
theorem number_2digit_smaller_than_35 : 
  (Finset.filter (λ n : ℕ, let d1 := n / 10, d2 := n % 10 in d1 < 3 ∨ d2 < 5) (Finset.range' 10 100)).card = 55 := 
by
  sorry

end number_2digit_smaller_than_35_l793_793347


namespace gross_pay_is_450_l793_793635

def net_pay : ℤ := 315
def taxes : ℤ := 135
def gross_pay : ℤ := net_pay + taxes

theorem gross_pay_is_450 : gross_pay = 450 := by
  sorry

end gross_pay_is_450_l793_793635


namespace sum_first_3m_l793_793573

theorem sum_first_3m {a : ℕ → ℝ} {m : ℕ}
  (S_m : ℝ) (S_2m : ℝ) (h1 : S_m = 30) (h2 : S_2m = 100)
  (h3 : ∀ (n : ℕ), S_n = n * (a 1 + a n) / 2) :
  S (3 * m) = 210 :=
sorry

end sum_first_3m_l793_793573


namespace sum_numerator_denominator_repeating_decimal_l793_793124

theorem sum_numerator_denominator_repeating_decimal :
  let x := 34 / 99 in
  x.denom * x + x.num = 133 :=
by
  let x : ℚ := 34 / 99
  sorry

end sum_numerator_denominator_repeating_decimal_l793_793124


namespace solve_inequality_l793_793031

theorem solve_inequality :
  ∀ x : ℝ, (x ≠ -4) → ((x^2 - 16) / (x + 4) < 0 ↔ x ∈ set.Ioo (-4 : ℝ) 4 ∨ x < -4) :=
by
  sorry

end solve_inequality_l793_793031


namespace prob1_prob2_maximum_prob2_minimum_l793_793436

def f (x: ℝ) : ℝ := 2*x^2 - x + 3

def g (x: ℝ) : ℝ := f (2^x)

theorem prob1:
  ∀ (x: ℝ),
  ∃ (a b c: ℝ), 
  f(x) = a * x^2 + b * x + c ∧ 
  (∀ x, f(x+1) - f(x) = 4 * x + 1) ∧ 
  f(0) = 3 :=
by
  use 2, -1, 3
  simp [f]
  sorry

theorem prob2_maximum :
  ∀ x ∈ Icc (-3:ℝ) 0, 
  g(x) ≤ 4 :=
by
  sorry

theorem prob2_minimum :
  ∀ x ∈ Icc (-3:ℝ) 0, 
  g(x) ≥ 23/8 :=
by
  sorry

end prob1_prob2_maximum_prob2_minimum_l793_793436


namespace length_DE_l793_793465

-- Definition and given conditions
variables {A B C D E : Type}
variables [Point (Triangle ABC)]

-- Given BC = 40 and angle C = 45 degrees
constants (BC : Real) (angleC : Real)
constants (midpoint : BC → D) (perpendicular_bisector : BC → AC → E)
constant (triangle_CDE_454590 : Is454590Triangle C D E)

-- Definitions for points D and E
noncomputable def midpoint_of_BC (P: BC) : D :=
  midpoint P

noncomputable def intersection_perpendicular_bisector_AC (P: BC → AC) : E :=
  perpendicular_bisector P

-- Prove length of DE == 20
theorem length_DE : length DE = 20 := sorry

end length_DE_l793_793465


namespace number_2digit_smaller_than_35_l793_793350

/--
Prove that the number of two-digit numbers where at least one digit
is smaller than the corresponding digit in 35 is exactly 55.
-/
theorem number_2digit_smaller_than_35 : 
  (Finset.filter (λ n : ℕ, let d1 := n / 10, d2 := n % 10 in d1 < 3 ∨ d2 < 5) (Finset.range' 10 100)).card = 55 := 
by
  sorry

end number_2digit_smaller_than_35_l793_793350


namespace length_de_l793_793479

open Triangle

/-- In triangle ABC, BC = 40 and ∠C = 45°. Let the perpendicular bisector
of BC intersect BC and AC at D and E, respectively. Prove that DE = 10√2. -/

theorem length_de 
  (ABC : Triangle)
  (B C : Point)
  (BC_40 : dist B C = 40)
  (angle_C_45 : ∠(B, C, AC) = 45)
  (D : Point)
  (midpoint_D : is_midpoint B C D)
  (E : Point)
  (intersection_E : is_perpendicular_bisector_intersection B C D E) :
  dist D E = 10 * real.sqrt 2 :=
begin
  -- proof steps would go here
  sorry
end

end length_de_l793_793479


namespace no_subset_with_length_at_least_60_l793_793873

-- Define the problem using Lean 4 statement
theorem no_subset_with_length_at_least_60
  (vectors : Fin 100 → E) [InnerProductSpace ℝ E]
  (h_unit : ∀ i, ‖vectors i‖ = 1)
  (h_sum : ∑ i, vectors i = 0) :
  ¬∃ (s : Finset (Fin 100)), ‖∑ i in s, vectors i‖ ≥ 60 := 
sorry

end no_subset_with_length_at_least_60_l793_793873


namespace guards_catch_monkey_lemma_l793_793863

inductive Vertex
| A | B | C

structure ZooPaths :=
  (equilateral_triangle : Prop)
  (mid_segments : Prop)

structure Situation :=
  (initial_guard_position : Vertex)
  (initial_monkey_position : Vertex)
  (visibility : Prop)
  (speed_ratio : ℕ)

def guards_can_catch_monkey (z : ZooPaths) (s : Situation) : Prop := ∃ t : ℕ, ∀ m_pos : Vertex, m_pos = s.initial_guard_position ∨ m_pos = s.initial_monkey_position → t < m_pos.speed_ratio.succ ∧ z.equilateral_triangle.succ ∧ z.mid_segments.succ ∉ t.succ → true

theorem guards_catch_monkey_lemma 
  (z : ZooPaths) (s : Situation) : guards_can_catch_monkey z s :=
begin
  -- Conditions
  have h1 : z.equilateral_triangle := sorry,
  have h2 : z.mid_segments := sorry,
  have h3 : s.visibility := sorry,
  have h4 : s.speed_ratio = 3 := sorry,
  
  -- Conclusion
  apply guards_can_catch_monkey,
  sorry
end

end guards_catch_monkey_lemma_l793_793863


namespace geometric_sequence_log_sum_l793_793454

theorem geometric_sequence_log_sum (a : ℕ → ℝ) (r : ℝ) 
  (h_pos : ∀ n, 0 < a n) (h_geom : ∀ n, a (n + 1) = r * a n) 
  (h_cond : a 9 * a 11 = 4) : 
  ∑ i in finset.range 19, real.logb 2 (a i) = 19 := 
by sorry

end geometric_sequence_log_sum_l793_793454


namespace Robert_ate_10_chocolates_l793_793021

def chocolates_eaten_by_Nickel : Nat := 5
def difference_between_Robert_and_Nickel : Nat := 5
def chocolates_eaten_by_Robert := chocolates_eaten_by_Nickel + difference_between_Robert_and_Nickel

theorem Robert_ate_10_chocolates : chocolates_eaten_by_Robert = 10 :=
by
  -- Proof omitted
  sorry

end Robert_ate_10_chocolates_l793_793021


namespace oblique_prism_volume_l793_793551

noncomputable def volume_of_oblique_prism 
  (a b c : ℝ) (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  : ℝ :=
  a * b * c / Real.sqrt (1 + (Real.cos α / Real.sin α)^2 + (Real.cos β / Real.sin β)^2)

theorem oblique_prism_volume 
  (a b c : ℝ) (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  : volume_of_oblique_prism a b c α β hα hβ = a * b * c / Real.sqrt (1 + (Real.cos α / Real.sin α)^2 + (Real.cos β / Real.sin β)^2) := 
by
  -- Proof will be completed here
  sorry

end oblique_prism_volume_l793_793551


namespace tax_for_3000_income_l793_793439

def taxable_amount (total_monthly_income : ℝ) : ℝ :=
  total_monthly_income - 800

def tax_amount (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 500 then 0.05 * x
  else if 500 < x ∧ x ≤ 2000 then 25 + 0.10 * (x - 500)
  else if 2000 < x ∧ x ≤ 5000 then 175 + 0.15 * (x - 2000)
  else 0 -- Assumes x is in [0, 5000], can extend for other ranges based on full problem

theorem tax_for_3000_income : tax_amount (taxable_amount 3000) = 205 := 
  by sorry

end tax_for_3000_income_l793_793439


namespace statement_B_statement_C_l793_793904

open Complex

theorem statement_B (z_1 : ℂ) : (z_1 - conj z_1).im = (2 * Complex.i * z_1.im).im :=
by sorry

theorem statement_C (z_1 z_2 : ℂ) : abs (z_1 + z_2) ≤ abs z_1 + abs z_2 :=
by sorry

end statement_B_statement_C_l793_793904


namespace length_DE_is_20_l793_793460

open_locale real
open_locale complex_conjugate

noncomputable def in_triangle_config_with_angle (ABC : Type) [triangle ABC]
  (BC_length : ℝ) (angle_C : real.angle) (perp_bisector_intersect_D : point)
  (perp_bisector_intersect_E : point) : Prop :=
  -- define the conditions
  BC_length = 40 ∧
  angle_C = 45 ∧
  is_midpoint_of_BC ABC perp_bisector_intersect_D ∧
  is_perpendicular_bisector_intersects_AC ABC perp_bisector_intersect_D perp_bisector_intersect_E

noncomputable def length_DE (D E: point) : ℝ :=
  distance_between D E

theorem length_DE_is_20 {ABC : Type} [triangle ABC]
  (BC_length : ℝ) (angle_C : real.angle) (perp_bisector_intersect_D : point)
  (perp_bisector_intersect_E : point)
  (h : in_triangle_config_with_angle ABC BC_length angle_C perp_bisector_intersect_D perp_bisector_intersect_E):
  length_DE perp_bisector_intersect_D perp_bisector_intersect_E = 20 :=
begin
  sorry,
end

end length_DE_is_20_l793_793460


namespace common_chord_circumcircle_l793_793449

noncomputable def common_chord_length : Real :=
  Real.sqrt (Real.sqrt (12)) * (2 - Real.sqrt 3) 

theorem common_chord_circumcircle 
(T : Type) [MetricSpace T]
{A B C : T}
(hiso : IsoscelesTriangle A B C)
(angleB : angle B = 120)
(sideAC : dist A C = 1)
: ∃ l, l = common_chord_length :=
sorry

end common_chord_circumcircle_l793_793449


namespace max_dominos_l793_793214

theorem max_dominos (n k : ℕ) (h1 : k ≤ n) (h2 : n < 2 * k) : 
  (n = k → (max_dominos n k = n)) ∧ 
  (n = 2 * k - 1 → (max_dominos n k = n)) ∧ 
  (k < n ∧ n < 2 * k - 1 → (max_dominos n k = 2 * n - 2 * k + 2)) :=
by
  sorry

end max_dominos_l793_793214


namespace sum_numerator_denominator_repeating_decimal_l793_793125

theorem sum_numerator_denominator_repeating_decimal :
  let x := 34 / 99 in
  x.denom * x + x.num = 133 :=
by
  let x : ℚ := 34 / 99
  sorry

end sum_numerator_denominator_repeating_decimal_l793_793125


namespace cyclic_BDFG_l793_793868

open EuclideanGeometry

-- Defining the acute triangle ABC and other points and properties as per the problem statement.
theorem cyclic_BDFG
  (A B C P Q D E F G : Point)
  (hABCacute : ∀ (X : Point), is_acute_triangle A B C)
  (hBAC_gt_BCA : ∠BAC > ∠BCA)
  (hP_on_BC : Online P B C)
  (hAngle_PAB_eq_BCA : ∠PAB = ∠BCA)
  (hQ_circumcircle_APB : IsOnCircumcircle Q A P B)
  (hQ_on_AC : Online Q A C)
  (hD_on_AP : Online D A P)
  (hAngle_QDC_eq_CAP : ∠QDC = ∠CAP)
  (hE_on_BD : Online E B D)
  (hCE_eq_CD : CE = CD)
  (hCircumcircle_CQE : IsOnCircumcircle F C Q E)
  (hQF_on_BC : Online G Q F)
  (hQF_meets_BC : Meets Q F B C) :
  Concyclic B D F G :=
by
  sorry

end cyclic_BDFG_l793_793868


namespace vector_magnitude_b_l793_793850

variables (a b : EuclideanSpace ℝ 3)
variables (h1 : ∥a∥ = 3) (h2 : ∥a - b∥ = 5) (h3 : inner a b = 1)

theorem vector_magnitude_b : ∥b∥ = 3 * real.sqrt 2 :=
by
  sorry

end vector_magnitude_b_l793_793850


namespace initial_markers_count_l793_793920

   -- Let x be the initial number of markers Megan had.
   variable (x : ℕ)

   -- Conditions:
   def robert_gave_109_markers : Prop := true
   def total_markers_after_adding : ℕ := 326
   def markers_added_by_robert : ℕ := 109

   -- The total number of markers Megan has now is 326.
   def total_markers_eq (x : ℕ) : Prop := x + markers_added_by_robert = total_markers_after_adding

   -- Prove that initially Megan had 217 markers.
   theorem initial_markers_count : total_markers_eq 217 := by
     sorry
   
end initial_markers_count_l793_793920


namespace probability_of_perfect_square_sum_l793_793599

-- Define the conditions
def dice_faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def perfect_squares_in_range : Finset ℕ := {4, 9, 16}
def total_outcomes : ℕ := 8 * 8  -- total number of outcomes when rolling two dice

-- Calculate the number of favorable outcomes
def favorable_outcomes : ℕ :=
  (dice_faces.product dice_faces).filter (λ pair => perfect_squares_in_range.member (pair.1 + pair.2)).card

-- Calculate the probability of getting a perfect square sum
noncomputable def probability_perfect_square_sum : ℚ :=
  favorable_outcomes / total_outcomes

-- The statement to prove that the probability is 3/16
theorem probability_of_perfect_square_sum :
  probability_perfect_square_sum = 3 / 16 :=
begin
  sorry
end

end probability_of_perfect_square_sum_l793_793599


namespace num_suitable_two_digit_numbers_l793_793375

/-- 
How many two-digit numbers have at least one digit that is smaller than the corresponding digit in the number 35?
-/
theorem num_suitable_two_digit_numbers : 
  let two_digit_numbers := { n : ℕ | 10 ≤ n ∧ n ≤ 99 },
      suitable_numbers := { n ∈ two_digit_numbers | (n / 10 < 3) ∨ (n % 10 < 5) } in
  suitable_numbers.card = 55 :=
by
  sorry

end num_suitable_two_digit_numbers_l793_793375


namespace min_value_symmetry_l793_793788

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem min_value_symmetry (a b c : ℝ) (h_a : a > 0) (h_symm : ∀ x : ℝ, quadratic a b c (2 + x) = quadratic a b c (2 - x)) : 
  quadratic a b c 2 < quadratic a b c 1 ∧ quadratic a b c 1 < quadratic a b c 4 := 
sorry

end min_value_symmetry_l793_793788


namespace curve_defined_by_theta_is_line_l793_793237

theorem curve_defined_by_theta_is_line : ∀ (θ : ℝ), θ = π / 4 → ∃ (r : ℝ), (r * cos θ, r * sin θ) lies_on_line :=
by
  sorry

end curve_defined_by_theta_is_line_l793_793237


namespace polynomial_expansion_correct_l793_793738

def polynomial1 (x : ℝ) := 3 * x^2 - 4 * x + 3
def polynomial2 (x : ℝ) := -2 * x^2 + 3 * x - 4

theorem polynomial_expansion_correct {x : ℝ} :
  (polynomial1 x) * (polynomial2 x) = -6 * x^4 + 17 * x^3 - 30 * x^2 + 25 * x - 12 :=
by
  sorry

end polynomial_expansion_correct_l793_793738


namespace arctan_sum_l793_793205

theorem arctan_sum (x y : ℝ) (hx : x = 3) (hy : y = 7) :
  Real.arctan (x / y) + Real.arctan (y / x) = Real.pi / 2 := 
by
  rw [hx, hy]
  sorry

end arctan_sum_l793_793205


namespace conclusion_six_l793_793775

def floor (x : ℝ) : ℤ := intFloor x

def f (x : ℝ) : ℤ := floor (Real.log (2^x + 1) / Real.log 2 - Real.log 9 / Real.log 2)

theorem conclusion_six (x : ℝ) (h1 : 12 < x) (h2 : x < 13) : f x = 9 :=
sorry

end conclusion_six_l793_793775


namespace trigonometric_identity_l793_793647

noncomputable def sin110cos40_minus_cos70sin40 : ℝ := 
  Real.sin (110 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) - 
  Real.cos (70 * Real.pi / 180) * Real.sin (40 * Real.pi / 180)

theorem trigonometric_identity : 
  sin110cos40_minus_cos70sin40 = 1 / 2 := 
by sorry

end trigonometric_identity_l793_793647


namespace coefficient_of_x3_in_expansion_l793_793730

theorem coefficient_of_x3_in_expansion :
  (∑ n in finset.range 6, (-1)^n * (nat.choose 5 n) * x^(5 - n) - 
   ∑ n in finset.range 7, (-1)^n * (nat.choose 6 n) * x^(6 - n)).coeff 3 = 10 :=
begin
  sorry
end

end coefficient_of_x3_in_expansion_l793_793730


namespace total_balloons_after_gift_l793_793986

-- Definitions for conditions
def initial_balloons := 26
def additional_balloons := 34

-- Proposition for the total number of balloons
theorem total_balloons_after_gift : initial_balloons + additional_balloons = 60 := 
by
  -- Proof omitted, adding sorry
  sorry

end total_balloons_after_gift_l793_793986


namespace vectors_are_not_coplanar_l793_793708

structure Vector3 :=
(x : ℝ) (y : ℝ) (z : ℝ)

def scalarTripleProduct (a b c : Vector3) : ℝ :=
  a.x * (b.y * c.z - b.z * c.y) -
  a.y * (b.x * c.z - b.z * c.x) +
  a.z * (b.x * c.y - b.y * c.x)

theorem vectors_are_not_coplanar :
  let a := Vector3.mk 4 2 2
  let b := Vector3.mk (-3) (-3) (-3)
  let c := Vector3.mk 2 1 2
  scalarTripleProduct a b c = -6 :=
by
  sorry

end vectors_are_not_coplanar_l793_793708


namespace problem_equivalent_proof_statement_l793_793633

-- Definition of a line with a definite slope
def has_definite_slope (m : ℝ) : Prop :=
  ∃ slope : ℝ, slope = -m 

-- Definition of the equation of a line passing through two points being correct
def line_through_two_points (x1 y1 x2 y2 : ℝ) (h : x1 ≠ x2) : Prop :=
  ∀ x y : ℝ, (y - y1 = ((y2 - y1) / (x2 - x1)) * (x - x1)) ↔ y = ((y2 - y1) * (x - x1) / (x2 - x1)) + y1 

-- Formalizing and proving the given conditions
theorem problem_equivalent_proof_statement : 
  (∀ m : ℝ, has_definite_slope m) ∧ 
  (∀ (x1 y1 x2 y2 : ℝ) (h : x1 ≠ x2), line_through_two_points x1 y1 x2 y2 h) :=
by 
  sorry

end problem_equivalent_proof_statement_l793_793633


namespace probability_of_perfect_square_sum_on_two_8_sided_dice_l793_793591

-- Define the dice and possible sums
def is_sum_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def pairs_rolled : finset (ℕ × ℕ) :=
  finset.univ.filter (λ p, p.1 + p.2 ∈ {4, 9, 16})

theorem probability_of_perfect_square_sum_on_two_8_sided_dice :
  (pairs_rolled.card : ℚ) / 64 = 3 / 16 :=
by
  sorry

end probability_of_perfect_square_sum_on_two_8_sided_dice_l793_793591


namespace trapezoid_area_correct_l793_793173

noncomputable def trapezoid_area (R a : ℝ) : ℝ :=
  -- Placeholder definition, the exact mathematical formulation of the area
  -- as given in the problem's solution.
  8 * R^3 / a

theorem trapezoid_area_correct (R a : ℝ) (hR : R > 0) (ha : a > 0) :
  let S := trapezoid_area R a
  in S = 8 * R^3 / a :=
by
  sorry

end trapezoid_area_correct_l793_793173


namespace calculate_expression_l793_793716

theorem calculate_expression :
  (-1 : ℤ)^2023 + real.cbrt 8 - 2 * real.sqrt (1/4) + abs (real.sqrt 3 - 2) = 2 - real.sqrt 3 :=
by
  sorry

end calculate_expression_l793_793716


namespace arithmetic_sequence_common_difference_l793_793952

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (a_1 : ℝ) (d : ℝ)
  (h1 : a 1 = a_1)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h_a5 : a 5 = 1)
  (h_sum : (∑ i in (finset.range 10).map (λ x, x + 1), a i) = 20) :
  d = 2 :=
sorry

end arithmetic_sequence_common_difference_l793_793952


namespace number_machine_output_l793_793676

def machine (x : ℕ) : ℕ := x + 15 - 6

theorem number_machine_output : machine 68 = 77 := by
  sorry

end number_machine_output_l793_793676


namespace sqrt_Q_is_fraction_l793_793065

theorem sqrt_Q_is_fraction (S S' Q : ℝ) (n m : ℕ)
  (h : Q = ((n * real.sqrt S) + (m * real.sqrt S')) / (n + m)) :
  real.sqrt Q = (n * real.sqrt S + m * real.sqrt S') / (n + m) :=
by
  sorry

end sqrt_Q_is_fraction_l793_793065


namespace line_intersects_ellipse_with_conditions_l793_793298

theorem line_intersects_ellipse_with_conditions :
  ∃ l : ℝ → ℝ, (∃ A B : ℝ × ℝ, 
  (A.fst^2/6 + A.snd^2/3 = 1 ∧ B.fst^2/6 + B.snd^2/3 = 1) ∧
  A.fst > 0 ∧ A.snd > 0 ∧ B.fst > 0 ∧ B.snd > 0 ∧
  (∃ M N : ℝ × ℝ, 
    M.snd = 0 ∧ N.fst = 0 ∧
    M.fst^2 + N.snd^2 = (2 * Real.sqrt 3)^2 ∧
    (M.snd - A.snd)^2 + (M.fst - A.fst)^2 = (N.fst - B.fst)^2 + (N.snd - B.snd)^2) ∧
    (∀ x, l x + Real.sqrt 2 * x - 2 * Real.sqrt 2 = 0)
) :=
sorry

end line_intersects_ellipse_with_conditions_l793_793298


namespace valid_conclusions_l793_793972

theorem valid_conclusions (a b c m : ℝ) (h1 : a ≠ 0) (h2 : a < 0) (h3 : -2 < m ∧ m < -1) (h4 : a + b + c = 0) (h5 : a * m^2 + b * m + c = 0) :
  (abc > 0) ∧ (a - b + c > 0) ∧ (a * (m + 1) - b + c > 0) ∧ ¬(a(x-m)(x-1) - 1 = 0 ∧ 4ac - b^2 > 4a) → true :=
begin
  sorry
end

end valid_conclusions_l793_793972


namespace range_of_a_l793_793832

theorem range_of_a (a : ℝ) :
  (∃ x ∈ set.Icc (-1 : ℝ) 2, e^2 * log a - e^2 * x + x - log a ≥ (2 * a / exp x) - 2) ↔ a ∈ set.Icc ((1 : ℝ) / exp 1) (exp (4 : ℝ)) :=
by sorry

end range_of_a_l793_793832


namespace triangle_solution_l793_793220

noncomputable def triangle_construction (a r d : ℝ) : Prop :=
  ∃ (T : Type) [triangle T], 
  let circumcircle := circle(center T, r) in
  let orthocenter := reflection(T, a) in
  distance(center(T), orthocenter) = d ∧ 
  ∃ (valid_triangle : T → Prop), valid_triangle(T) ∧
  (check_validity(a, r) ∧ 
   (number_of_solutions(a, r, d) = 2 ∨ 
    number_of_solutions(a, r, d) = 1 ∨ 
    number_of_solutions(a, r, d) = 0))

theorem triangle_solution (a r d : ℝ) : triangle_construction a r d := 
  sorry

end triangle_solution_l793_793220


namespace quadratic_intersects_at_3_points_l793_793435

theorem quadratic_intersects_at_3_points (m : ℝ) : 
  (exists x : ℝ, x^2 + 2*x + m = 0) ∧ (m ≠ 0) → m < 1 :=
by
  sorry

end quadratic_intersects_at_3_points_l793_793435


namespace b_profit_share_l793_793169

theorem b_profit_share (total_capital : ℝ) (profit : ℝ) (A_invest : ℝ) (B_invest : ℝ) (C_invest : ℝ) (D_invest : ℝ)
 (A_time : ℝ) (B_time : ℝ) (C_time : ℝ) (D_time : ℝ) :
  total_capital = 100000 ∧
  A_invest = B_invest + 10000 ∧
  B_invest = C_invest + 5000 ∧
  D_invest = A_invest + 8000 ∧
  A_time = 12 ∧
  B_time = 10 ∧
  C_time = 8 ∧
  D_time = 6 ∧
  profit = 50000 →
  (B_invest * B_time / (A_invest * A_time + B_invest * B_time + C_invest * C_time + D_invest * D_time)) * profit = 10925 :=
by
  sorry

end b_profit_share_l793_793169


namespace swim_depth_calculation_l793_793695

-- Definitions based on conditions
def Ron_height := 13
def Dean_height := Ron_height + 4
def max_depth_high_tide := 15 * Dean_height
def current_tide_depth := 0.75 * max_depth_high_tide
def additional_current_depth := 0.20 * current_tide_depth
def total_water_depth := current_tide_depth + additional_current_depth

-- Theorem to prove the total depth
theorem swim_depth_calculation : total_water_depth = 229.5 := by
  sorry

end swim_depth_calculation_l793_793695


namespace vector_magnitude_b_l793_793846

variables (a b : EuclideanSpace ℝ 3)
variables (h1 : ∥a∥ = 3) (h2 : ∥a - b∥ = 5) (h3 : inner a b = 1)

theorem vector_magnitude_b : ∥b∥ = 3 * real.sqrt 2 :=
by
  sorry

end vector_magnitude_b_l793_793846


namespace positive_real_solution_unique_l793_793821

theorem positive_real_solution_unique :
  let f := λ (x : ℝ), x^4 + 8*x^3 + 15*x^2 + 2023*x - 1500 in
  let f' := λ (x : ℝ), 4*x^3 + 24*x^2 + 30*x + 2023 in
  (∀ x : ℝ, x > 0 → f' x > 0) ∧ f 0 < 0 ∧ f 1 > 0 →
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
by
  let f := λ (x : ℝ), x^4 + 8*x^3 + 15*x^2 + 2023*x - 1500
  let f' := λ (x : ℝ), 4*x^3 + 24*x^2 + 30*x + 2023
  have Hmonotonic : ∀ x : ℝ, x > 0 → f' x > 0
  have Hsign_change : f 0 < 0 ∧ f 1 > 0
  sorry

end positive_real_solution_unique_l793_793821


namespace num_divisors_less_than_n_not_divide_n_l793_793902

theorem num_divisors_less_than_n_not_divide_n 
  (n : ℕ) (h : n = 2^35 * 3^21) :
  let n_squared_divisors_less_than_n := (∏ i in {1, 2}, (nat.factorization h).getOrDefault i 1 + 1) / 2 - 1,
      n_divisors := ∏ i in {1, 2}, (nat.factorization h).getOrDefault i 1 + 1 in
  (n_squared_divisors_less_than_n - n_divisors = 734) :=
by
  sorry

end num_divisors_less_than_n_not_divide_n_l793_793902


namespace find_numbers_in_progressions_l793_793579

theorem find_numbers_in_progressions (a b c d : ℝ) :
    (a + b + c = 114) ∧ -- Sum condition
    (b^2 = a * c) ∧ -- Geometric progression condition
    (b = a + 3 * d) ∧ -- Arithmetic progression first condition
    (c = a + 24 * d) -- Arithmetic progression second condition
    ↔ (a = 38 ∧ b = 38 ∧ c = 38) ∨ (a = 2 ∧ b = 14 ∧ c = 98) := by
  sorry

end find_numbers_in_progressions_l793_793579


namespace kite_parabola_l793_793567

theorem kite_parabola (a b : ℝ) (h1 : ∀ x : ℝ, ∀ y : ℝ, y = ax^2 + 3 ∨ y = 9 - bx^2 → 
  (x = 0 ∨ y = 0)) (h2 : 3 = 6 * (2 * sqrt (9 / b) / (2 * sqrt (-3 / a)))) : a + b = 2 / 3 :=
  sorry

end kite_parabola_l793_793567


namespace arctan_sum_l793_793198

theorem arctan_sum (a b : ℝ) (h1 : a = 3) (h2 : b = 7) : 
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 := 
by 
  rw [h1, h2]
  sorry

end arctan_sum_l793_793198


namespace y_at_x8_l793_793424

theorem y_at_x8 (k : ℝ) (y : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, y x = k * x^(1/3))
  (h2 : y 64 = 4 * Real.sqrt 3) :
  y 8 = 2 * Real.sqrt 3 :=
by
  sorry

end y_at_x8_l793_793424


namespace production_line_B_units_l793_793663

theorem production_line_B_units {x y z : ℕ} (h1 : x + y + z = 24000) (h2 : 2 * y = x + z) : y = 8000 :=
sorry

end production_line_B_units_l793_793663


namespace sempoltec_final_item_l793_793037

noncomputable def remaining_item (G P B : ℕ) (f : ℕ → ℕ → ℕ → (ℕ × ℕ × ℕ)) : String :=
  if G % 2 = 1 then "Gold"
  else if P % 2 = 1 then "Pearls"
  else "Beads"

theorem sempoltec_final_item :
  remaining_item 24 26 25 (λ G P B, (G, P, B)) = "Beads" :=
sorry

end sempoltec_final_item_l793_793037


namespace number_of_b_values_l793_793245

-- Let's define the conditions and the final proof required.
def inequations (x b : ℤ) : Prop := 
  (3 * x > 4 * x - 4) ∧
  (4 * x - b > -8) ∧
  (5 * x < b + 13)

theorem number_of_b_values :
  (∀ x : ℤ, 1 ≤ x → x ≠ 3 → ¬ inequations x b) →
  (∃ (b_values : Finset ℤ), 
      (∀ b ∈ b_values, inequations 3 b) ∧ 
      (b_values.card = 7)) :=
sorry

end number_of_b_values_l793_793245


namespace sqrt13_decomposition_ten_plus_sqrt3_decomposition_l793_793020

-- For the first problem
theorem sqrt13_decomposition :
  let a := 3
  let b := Real.sqrt 13 - 3
  a^2 + b - Real.sqrt 13 = 6 := by
sorry

-- For the second problem
theorem ten_plus_sqrt3_decomposition :
  let x := 11
  let y := Real.sqrt 3 - 1
  x - y = 12 - Real.sqrt 3 := by
sorry

end sqrt13_decomposition_ten_plus_sqrt3_decomposition_l793_793020


namespace area_of_triangle_correct_l793_793531

noncomputable def area_of_triangle (P F1 F2 : Point) : ℝ :=
  let angle := 30 * (Real.pi / 180)  -- converting 30 degrees to radians.
  let mid_dist := Real.sqrt 5  -- since the length of the major axis 2a = 2sqrt(5)
  let distance := 2 * Real.sqrt 5
  let len_of_focus := 1  -- c which is the distance to the focus
  let F1F2_length := 2 * len_of_focus
  ∃ (x_P : ℝ) (y_P : ℝ), 
   P.1 = x_P ∧ P.2 = y_P ∧
   F1.1 = 0 ∧ F1.2 = len_of_focus ∧
   F2.1 = 0 ∧ F2.2 = -len_of_focus ∧
   (y_P^2 / 5 + x_P^2 / 4 = 1) ∧ -- Condition that P is on the ellipse
   Real.arccos ((F1.2 - y_P) * (F2.2 - y_P) / (Real.sqrt ((F1.2 - y_P)^2 + (F1.1 - x_P)^2) * Real.sqrt ((F2.2 - y_P)^2 + (F2.1 - x_P)^2))) = angle ∧
   let side1 := Real.sqrt ((F1.2 - y_P)^2 + (F1.1 - x_P)^2) in
   let side2 := Real.sqrt ((F2.2 - y_P)^2 + (F2.1 - x_P)^2) in
   let area := (side1 * side2 * Real.sin angle) / 2 in
   area = 8 - 4 * Real.sqrt 3

theorem area_of_triangle_correct : area_of_triangle P F1 F2 :=
by sorry

end area_of_triangle_correct_l793_793531


namespace sqrt_S_floor_l793_793497

def floor (x : ℝ) : ℤ := Int.floor x

def S : ℤ :=
  ∑ i in (Finset.range 2020).filter (λ n, n > 0), floor (Real.sqrt i)

theorem sqrt_S_floor :
  floor (Real.sqrt (S : ℝ)) = 243 :=
sorry

end sqrt_S_floor_l793_793497


namespace simplify_absolute_value_l793_793947

theorem simplify_absolute_value : abs (-(5^2) + 6 * 2) = 13 := by
  sorry

end simplify_absolute_value_l793_793947


namespace distance_midpoint_larger_arc_l793_793575

/-- The vertices of a rectangle inscribed in a circle divide the circle into four arcs.
Given the sides of the rectangle are 24 cm and 7 cm, prove the distance from the midpoint
of one of the larger arcs to the vertices of the rectangle is 15 cm and 20 cm. -/
theorem distance_midpoint_larger_arc (a b : ℝ) (h₁ : a = 24) (h₂ : b = 7) :
  let d := real.sqrt (a ^ 2 + b ^ 2)
  let r := d / 2
  let mf := 12.5 - 3.5
  let mk := 12.5 + 3.5
  let bf := b / 2
  let ob := r
  let of := real.sqrt (ob ^ 2 - bf ^ 2)
  (real.sqrt (mf ^ 2 + bf ^ 2) = 15) ∧
  (real.sqrt (mk ^ 2 + bf ^ 2) = 20) := 
by
  sorry

end distance_midpoint_larger_arc_l793_793575


namespace num_suitable_two_digit_numbers_l793_793374

/-- 
How many two-digit numbers have at least one digit that is smaller than the corresponding digit in the number 35?
-/
theorem num_suitable_two_digit_numbers : 
  let two_digit_numbers := { n : ℕ | 10 ≤ n ∧ n ≤ 99 },
      suitable_numbers := { n ∈ two_digit_numbers | (n / 10 < 3) ∨ (n % 10 < 5) } in
  suitable_numbers.card = 55 :=
by
  sorry

end num_suitable_two_digit_numbers_l793_793374


namespace min_value_l793_793055

noncomputable def f (x : ℝ) : ℝ := x^4 - 4 * x + 3

theorem min_value (I : Set.Icc (-2 : ℝ) (3 : ℝ)) : 
  ∃ x ∈ I, ∀ y ∈ I, f x ≤ f y ∧ f x = 0 :=
begin
  -- Proof goes here
  sorry
end

end min_value_l793_793055


namespace inverse_proposition_l793_793772

-- Definitions of the propositions
def is_odd (a : ℕ) : Prop := ∃ k : ℕ, a = 2 * k + 1
def is_prime (a : ℕ) : Prop := a > 1 ∧ ∀ b c : ℕ, a = b * c → b = 1 ∨ c = 1

-- Initial proposition P: If a is odd, then a is prime
def P (a : ℕ) : Prop := is_odd a → is_prime a

-- The inverse proposition of P: If a is prime, then a is odd
def inverse_P (a : ℕ) : Prop := is_prime a → is_odd a

theorem inverse_proposition (a : ℕ) : inverse_P a = (is_prime a → is_odd a) :=
by {
  -- Proof goes here
  sorry
}

end inverse_proposition_l793_793772


namespace solve_problem_l793_793801

-- Define the set A (where A is constructed from an arithmetic sequence)
def A (a : ℕ → ℕ) := {x | ∃ k ∈ {1, 2, ..., 2016}, x = a k}

-- Define arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℕ) := ∃ d, d ≠ 0 ∧ ∀ n, n ≥ 1 → a (n + 1) = a n + d

-- Define L(A), the number of distinct sums of pairs from A
def L (a : ℕ → ℕ) : ℕ := (A a).to_finset.sum (λ x, ∃ i j, 1 ≤ i ∧ i < j ∧ j ≤ 2016 ∧ x = a i + a j)

-- Theorem statement to be proven
theorem solve_problem (a : ℕ → ℕ) (h1 : is_arithmetic_sequence a) : L a = 4029 :=
by
  sorry

end solve_problem_l793_793801


namespace count_four_digit_even_integers_l793_793815

-- Defining the set of even digits
def even_digits := {0, 2, 4, 6, 8}

-- The problem statement in Lean 4
theorem count_four_digit_even_integers :
  (count(p : ℕ // 1000 ≤ p ∧ p < 10000 ∧ ∀ d ∈ to_digits 10 (p : ℕ), d ∈ even_digits)) = 500 :=
sorry

end count_four_digit_even_integers_l793_793815


namespace sum_of_fraction_numerator_denominator_l793_793101

theorem sum_of_fraction_numerator_denominator :
  let x := 0.343434...
  in let fraction := (34 / 99 : ℚ)
  in let sum := fraction.num + fraction.den 
  in (x : ℚ) = fraction ∧ fraction.isReduced → sum = 133 :=
by
  sorry

end sum_of_fraction_numerator_denominator_l793_793101


namespace obtuse_angle_range_l793_793256

variables {λ : ℝ}

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (λ, 1)

theorem obtuse_angle_range (h : (a.1 * b.1 + a.2 * b.2) < 0) : λ < -1 ∨ (-1 < λ ∧ λ < 1) :=
by sorry

end obtuse_angle_range_l793_793256


namespace unoccupied_volume_eq_l793_793989

-- Define the radii and heights of the cones and cylinder
variable (r_cone : ℝ) (h_cone : ℝ) (h_cylinder : ℝ)
variable (r_cone_eq : r_cone = 10) (h_cone_eq : h_cone = 15) (h_cylinder_eq : h_cylinder = 30)

-- Define the volumes of the cylinder and the two cones
def volume_cylinder (r h : ℝ) : ℝ := π * r ^ 2 * h
def volume_cone (r h : ℝ) : ℝ := 1 / 3 * π * r ^ 2 * h
def volume_unoccupied : ℝ := volume_cylinder r_cone h_cylinder - 2 * volume_cone r_cone h_cone

-- Expression of the final result
theorem unoccupied_volume_eq : volume_unoccupied r_cone h_cone h_cylinder = 2000 * π :=
by
  rw [r_cone_eq, h_cone_eq, h_cylinder_eq]
  unfold volume_cylinder volume_cone volume_unoccupied
  norm_num

end unoccupied_volume_eq_l793_793989


namespace base5_digits_count_l793_793089

theorem base5_digits_count (n : ℕ) (h : n = 3125) : 
  let base5_digits := 6 in -- Number of digits in the base-5 representation of 3125.
  n.base5DigitLength = base5_digits := 
  sorry

end base5_digits_count_l793_793089


namespace average_eq_4x_minus_7_l793_793960

theorem average_eq_4x_minus_7 (x : ℝ) :
  (1 / 3) * ((x + 6) + (6x + 2) + (2x + 7)) = 4 * x - 7 → x = 12 :=
by
  sorry

end average_eq_4x_minus_7_l793_793960


namespace each_boy_makes_14_dollars_l793_793082

noncomputable def victor_shrimp_caught := 26
noncomputable def austin_shrimp_caught := victor_shrimp_caught - 8
noncomputable def brian_shrimp_caught := (victor_shrimp_caught + austin_shrimp_caught) / 2
noncomputable def total_shrimp_caught := victor_shrimp_caught + austin_shrimp_caught + brian_shrimp_caught
noncomputable def money_made := (total_shrimp_caught / 11) * 7
noncomputable def each_boys_earnings := money_made / 3

theorem each_boy_makes_14_dollars : each_boys_earnings = 14 := by
  sorry

end each_boy_makes_14_dollars_l793_793082


namespace length_DE_is_20_l793_793475

noncomputable def length_DE (BC : ℝ) (angle_C_deg : ℝ) 
  (D : ℝ) (is_midpoint_D : D = BC / 2)
  (is_right_triangle : angle_C_deg = 45): ℝ := 
let DE := D in DE

theorem length_DE_is_20 : ∀ (BC : ℝ) (angle_C_deg : ℝ),
  BC = 40 → 
  angle_C_deg = 45 → 
  let D := BC / 2 in 
  let DE := D in 
  DE = 20 :=
by
  intros BC angle_C_deg hBC hAngle
  sorry

end length_DE_is_20_l793_793475


namespace find_y_l793_793054

-- Define the given set
def nums (y : ℕ) : List ℕ := [2, 3, 4, 5, 5, 6, y]

-- Define the mode (as most frequent element)
noncomputable def mode (lst : List ℕ) : ℕ :=
lst.maxBy (fun x => lst.count x)

-- Define the median (middle value after sorting)
noncomputable def median (lst : List ℕ) : ℕ :=
let sorted := lst.qsort (≤)
sorted.nth (sorted.length / 2) |>.getD 0

-- Define the mean (average value)
noncomputable def mean (lst : List ℕ) : ℕ :=
lst.sum / lst.length

-- Theorem that proves y = 10 satisfies the conditions
theorem find_y : ∃ y : ℕ, let lst := nums y in mean lst = 5 ∧ median lst = 5 ∧ mode lst = 5 := by
  exists 10
  sorry

end find_y_l793_793054


namespace two_digit_numbers_less_than_35_l793_793354

theorem two_digit_numbers_less_than_35 : 
  let count : ℕ := (99 - 10 + 1) - (7 * 5)
  count = 55 :=
by
  -- definition of total number of two-digit numbers
  let total_two_digit_numbers : ℕ := 99 - 10 + 1
  -- definition of the number of unsuitable two-digit numbers
  let unsuitable_numbers : ℕ := 7 * 5
  -- definition of the count of suitable two-digit numbers
  let count : ℕ := total_two_digit_numbers - unsuitable_numbers
  -- verifying the final count
  exact rfl

end two_digit_numbers_less_than_35_l793_793354


namespace exists_perpendicular_PQ_l793_793482

noncomputable theory

open Set

def parabola1 (x : ℝ) : ℝ := x^2

def parabola2 (x : ℝ) : ℝ := 2 * x^2 - (7 / 2) * x + 57 / 16

theorem exists_perpendicular_PQ :
  ∃ (t u : ℝ), 
  ∃ (P Q : ℝ × ℝ),
  P = (t, parabola1 t) ∧ Q = (u, parabola2 u) 
  ∧ (deriv parabola1 t) * (deriv parabola2 u) = -1 :=
by
  sorry

end exists_perpendicular_PQ_l793_793482


namespace back_wheel_revolutions_l793_793926

noncomputable def circumference (r : ℝ) := 2 * Real.pi * r

noncomputable def distance_traveled (r : ℝ) (revolutions : ℝ) := 
  revolutions * circumference r

noncomputable def convert_meters_to_feet (meters : ℝ) := 
  meters * 3.28084

theorem back_wheel_revolutions :
  let r_front : ℝ := 1.5
  let r_back : ℝ := 0.5
  let front_revolutions : ℝ := 50
  let front_circumference := circumference r_front
  let distance_front_meters := distance_traveled r_front front_revolutions
  let distance_front_feet := convert_meters_to_feet distance_front_meters
  let back_circumference := circumference r_back
  let back_revolutions := distance_front_feet / back_circumference
  back_revolutions ≈ 492 := 
by 
  sorry

end back_wheel_revolutions_l793_793926


namespace probability_either_hits_l793_793281

noncomputable def P (E : Prop) : ℝ := sorry -- Placeholder definition for probability

variables (PA PB : ℝ)

-- The conditions
axiom shooter_A_hits : PA = 0.9
axiom shooter_B_hits : PB = 0.8

-- Definition of the event where A or B hits the target
def both_miss (PA PB : ℝ) : ℝ := (1 - PA) * (1 - PB)
def either_hits (PA PB : ℝ) : ℝ := 1 - both_miss PA PB

-- The proof statement
theorem probability_either_hits : PA = 0.9 → PB = 0.8 → either_hits PA PB = 0.98 := 
by
  intros PA_eq PB_eq
  rw [PA_eq, PB_eq]
  unfold either_hits
  unfold both_miss
  norm_num
  sorry

end probability_either_hits_l793_793281


namespace ball_travel_distance_l793_793701

theorem ball_travel_distance 
    (initial_height : ℕ)
    (half : ℕ → ℕ)
    (num_bounces : ℕ)
    (height_after_bounce : ℕ → ℕ)
    (total_distance : ℕ) :
    initial_height = 16 ∧ 
    (∀ n, half n = n / 2) ∧ 
    num_bounces = 4 ∧ 
    (height_after_bounce 0 = initial_height) ∧
    (∀ n, height_after_bounce (n + 1) = half (height_after_bounce n))
→ total_distance = 46 :=
by
  sorry

end ball_travel_distance_l793_793701


namespace angle_BCA_l793_793549

theorem angle_BCA (A B C H : Type) (hAB : A ≠ B) (hAC : A ≠ C) (hBC : B ≠ C) 
  (orthocenter_H : ∃ D E F, (line AD = ⊥ line BC) ∧ (line BE = ⊥ line AC) ∧ (line CF = ⊥ line AB) ∧ D ≠ E ∧ E ≠ F ∧ F ≠ D ∧ AD ∩ BE ∩ CF = {H} )
  (hABeqCH : distance A B = distance C H) :
  angle B C A = 45 :=
sorry

end angle_BCA_l793_793549


namespace Xiaoming_passwords_l793_793874

def num_passwords : ℕ :=
  let distinct_digits := [3, 4, 5, 9]
  let arrangements_of_distinct_digits := List.permutations distinct_digits |>.length
  let choose_spaces := Nat.choose 5 2
  arrangements_of_distinct_digits * choose_spaces

theorem Xiaoming_passwords :
  num_passwords = 240 := by
  sorry

end Xiaoming_passwords_l793_793874


namespace arctan_sum_l793_793212

theorem arctan_sum {a b : ℝ} (h3 : a = 3) (h7 : b = 7) :
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 :=
by
  sorry

end arctan_sum_l793_793212


namespace non_intersecting_segments_l793_793720

theorem non_intersecting_segments (n : ℕ) 
  (polygon_vertices : Finset (ℕ × ℕ)) 
  (h_vertices_count : polygon_vertices.card = 2 * n)
  (blue_points : Finset (ℕ × ℕ))
  (red_points : Finset (ℕ × ℕ))
  (h_blue_count : blue_points.card = n)
  (h_red_count : red_points.card = n)
  (h_all_blue_red : blue_points ∪ red_points = polygon_vertices)
  (convex_polygon : polygon_vertices.toList.cyclically_ordered) :
  ∃ (segments : Finset (Finset (ℕ × ℕ))),
    segments.card = n ∧ 
    (∀ segment ∈ segments, ∃ b r, segment = {b, r} ∧ b ∈ blue_points ∧ r ∈ red_points) ∧
    ∀ segment1 segment2 ∈ segments, segment1 ≠ segment2 → Disjoint segment1 segment2 :=
sorry

end non_intersecting_segments_l793_793720


namespace shaded_area_of_square_l793_793732

theorem shaded_area_of_square (side_square : ℝ) (leg_triangle : ℝ) (h1 : side_square = 40) (h2 : leg_triangle = 25) :
  let area_square := side_square ^ 2
  let area_triangle := (1 / 2) * leg_triangle * leg_triangle
  let total_area_triangles := 2 * area_triangle
  let shaded_area := area_square - total_area_triangles
  shaded_area = 975 :=
by
  sorry

end shaded_area_of_square_l793_793732


namespace restocked_bags_correct_l793_793688

def initial_stock := 55
def sold_bags := 23
def final_stock := 164

theorem restocked_bags_correct :
  (final_stock - (initial_stock - sold_bags)) = 132 :=
by
  -- The proof would go here, but we use sorry to skip it.
  sorry

end restocked_bags_correct_l793_793688


namespace complement_M_in_U_l793_793913

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_M_in_U :
  U \ M = {3, 5, 6} :=
by sorry

end complement_M_in_U_l793_793913


namespace right_triangle_angles_l793_793862

theorem right_triangle_angles (A B C D : Point)
(h1 : ∠ABC = 90)
(h2 : D is_midpoint_of (A, C))
(h3 : ∠(angle_bisector_of ∠ABC, median_to_hypotenuse) = 16) :
∠BAC = 61 ∧ ∠BCA = 29 ∧ ∠ABC = 90 :=
by
  sorry

end right_triangle_angles_l793_793862


namespace percent_kindergarten_combined_l793_793977

-- Define the constants provided in the problem
def studentsPinegrove : ℕ := 150
def studentsMaplewood : ℕ := 250

def percentKindergartenPinegrove : ℝ := 18.0
def percentKindergartenMaplewood : ℝ := 14.0

-- The proof statement
theorem percent_kindergarten_combined :
  (27.0 + 35.0) / (150.0 + 250.0) * 100.0 = 15.5 :=
by 
  sorry

end percent_kindergarten_combined_l793_793977


namespace inequality_solution_l793_793543

theorem inequality_solution (x : ℝ) : x ∈ Set.Ioo (-7 : ℝ) (7 : ℝ) ↔ (x^2 - 49) / (x + 7) < 0 :=
by 
  sorry

end inequality_solution_l793_793543


namespace molecular_weight_BaO_l793_793623

theorem molecular_weight_BaO {molecular_weight_6_moles : ℕ} (h : molecular_weight_6_moles = 918) : 
  let molecular_weight_1_mole := molecular_weight_6_moles / 6 in
  molecular_weight_1_mole = 153 :=
by
  sorry

end molecular_weight_BaO_l793_793623


namespace manhattan_dist_sum_min_max_l793_793880

theorem manhattan_dist_sum_min_max (a b : ℝ) (h : |a - 2| + |b| = 2) :
  (let expr := a^2 + b^2 - 4 * a in
   (λ E : ℝ, ∃ u v : ℝ, u = expr ∧ v = expr ∧ u ≤ v ∧ (u + v)) = (-2)) :=
by
  let expr := a^2 + b^2 - 4 * a
  have := h
  sorry

end manhattan_dist_sum_min_max_l793_793880


namespace parallel_lines_distance_l793_793779

theorem parallel_lines_distance (b c : ℝ) 
  (h1: b = 8) 
  (h2: (abs (10 - c) / (Real.sqrt (3^2 + 4^2))) = 3) :
  b + c = -12 ∨ b + c = 48 := by
 sorry

end parallel_lines_distance_l793_793779


namespace trapezoid_perimeter_correct_l793_793586

variable (AP PQ DQ AB BC CD AD perimeter : ℕ)

-- Define the given lengths
def AP_val := 24
def PQ_val := 32
def DQ_val := 18
def AB_val := 29
def BC_val := 32
def CD_val := 35
def AD_val := AP_val + PQ_val + DQ_val  -- Split of AD into segments AP, PQ, DQ
def perimeter_val := AB_val + BC_val + CD_val + AD_val

-- Calculate the perimeter and prove equality
theorem trapezoid_perimeter_correct : perimeter_val = 170 := by
  rw [AD_val, AP_val, PQ_val, DQ_val, AB_val, BC_val, CD_val]
  have h_ad : 24 + 32 + 18 = 74 := by norm_num
  rw [h_ad]
  have h_perimeter : 29 + 32 + 35 + 74 = 170 := by norm_num
  exact h_perimeter

end trapezoid_perimeter_correct_l793_793586


namespace length_DE_l793_793463

-- Definition and given conditions
variables {A B C D E : Type}
variables [Point (Triangle ABC)]

-- Given BC = 40 and angle C = 45 degrees
constants (BC : Real) (angleC : Real)
constants (midpoint : BC → D) (perpendicular_bisector : BC → AC → E)
constant (triangle_CDE_454590 : Is454590Triangle C D E)

-- Definitions for points D and E
noncomputable def midpoint_of_BC (P: BC) : D :=
  midpoint P

noncomputable def intersection_perpendicular_bisector_AC (P: BC → AC) : E :=
  perpendicular_bisector P

-- Prove length of DE == 20
theorem length_DE : length DE = 20 := sorry

end length_DE_l793_793463


namespace kaleb_tickets_l793_793180

variable (T : Nat)
variable (tickets_left : Nat) (ticket_cost : Nat) (total_spent : Nat)

theorem kaleb_tickets : tickets_left = 3 → ticket_cost = 9 → total_spent = 27 → T = 6 :=
by
  sorry

end kaleb_tickets_l793_793180


namespace find_x_l793_793803

open Set

-- Defining the conditions as sets.
def A (x : ℝ) := {1, 4, x}
def B (x : ℝ) := {1, x^2}

-- Proposition to be proven
theorem find_x (x : ℝ) (h : A x ∪ B x = A x) : x = 2 ∨ x = -2 ∨ x = 0 := 
by {
  sorry
}

end find_x_l793_793803


namespace happy_snakes_not_purple_l793_793584

variables (Snakes : Type) -- Define the type for snakes
variables (Happy Purple CanAdd CannotSubtract : Snakes → Prop) -- Define the properties of snakes

-- Given conditions
axiom Happy_implies_CanAdd : ∀ s, Happy s → CanAdd s
axiom Purple_implies_CannotSubtract : ∀ s, Purple s → CannotSubtract s
axiom CannotSubtract_implies_CannotAdd : ∀ s, CannotSubtract s → ¬ CanAdd s

-- Goal: Prove that happy snakes are not purple
theorem happy_snakes_not_purple : ∀ s, Happy s → ¬ Purple s := 
by 
  intro s,
  intro hs,
  intro ps,
  have cs := Purple_implies_CannotSubtract s ps,
  have nca := CannotSubtract_implies_CannotAdd s cs,
  have ca := Happy_implies_CanAdd s hs,
  contradiction

-- sorry

end happy_snakes_not_purple_l793_793584


namespace simplify_expression_l793_793946

theorem simplify_expression (x : ℝ) (h1 : sin (2 * x) = 2 * sin x * cos x)
  (h2 : cos (2 * x) = 2 * cos x * cos x - 1)
  (h3 : sin x * sin x + cos x * cos x = 1)
  (h4 : cos x ≠ 0) :
  (sin x * cos x + sin (2 * x)) / (cos x + cos (2 * x) + sin x * sin x) = 3 * sin x / (1 + 2 * cos x) :=
by
  sorry

end simplify_expression_l793_793946


namespace canoes_to_kayaks_ratio_l793_793618

theorem canoes_to_kayaks_ratio
  (canoe_cost kayak_cost total_revenue canoes_more_than_kayaks : ℕ)
  (H1 : canoe_cost = 14)
  (H2 : kayak_cost = 15)
  (H3 : total_revenue = 288)
  (H4 : ∃ C K : ℕ, C = K + canoes_more_than_kayaks ∧ 14 * C + 15 * K = 288) :
  ∃ (r : ℚ), r = 3 / 2 := by
  sorry

end canoes_to_kayaks_ratio_l793_793618


namespace assignment_methods_count_l793_793074

-- Define the number of teachers and question types.
def num_teachers : ℕ := 4
def num_question_types : ℕ := 3

-- State that each assignment maps each teacher to a unique question type.
theorem assignment_methods_count (h_teachers: num_teachers = 4) (h_types: num_question_types = 3) :
  ∃ count : ℕ, count = 36 :=
by
  use 36
  sorry

end assignment_methods_count_l793_793074


namespace two_digit_numbers_less_than_35_l793_793355

theorem two_digit_numbers_less_than_35 : 
  let count : ℕ := (99 - 10 + 1) - (7 * 5)
  count = 55 :=
by
  -- definition of total number of two-digit numbers
  let total_two_digit_numbers : ℕ := 99 - 10 + 1
  -- definition of the number of unsuitable two-digit numbers
  let unsuitable_numbers : ℕ := 7 * 5
  -- definition of the count of suitable two-digit numbers
  let count : ℕ := total_two_digit_numbers - unsuitable_numbers
  -- verifying the final count
  exact rfl

end two_digit_numbers_less_than_35_l793_793355


namespace arithmetic_geometric_sequences_properties_l793_793265

open Nat

noncomputable def a_n (n : ℕ) : ℕ := 3 * n

noncomputable def b_n (n : ℕ) : ℝ := 3 ^ (n - 1)

noncomputable def c_n (n k : ℕ) : ℝ := k + a_n n + log 3 (b_n n)

theorem arithmetic_geometric_sequences_properties :
  (∀ n : ℕ, a_n n = 3 * n) ∧
  (∀ n : ℕ, b_n n = 3 ^ (n - 1)) ∧
  (∀ k : ℕ, ∃ t : ℕ, t ≥ 3 ∧
    let c1 := c_n 1 k, c2 := c_n 2 k, ct := c_n t k
    in (2 / c2) = (1 / c1) + (1 / ct) ∧
       (k, t) ∈ {(2, 11), (3, 7), (5, 5), (9, 4)}){
sorry

end arithmetic_geometric_sequences_properties_l793_793265


namespace total_sold_l793_793852

theorem total_sold (D C : ℝ) (h1 : D = 1.6 * C) (h2 : D = 168) : D + C = 273 :=
by
  sorry

end total_sold_l793_793852


namespace find_v3_using_Horner_l793_793766

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 4 * x^5 - 3 * x^3 + 2 * x^2 + 5 * x + 1

-- Use Horner's method to compute intermediate values and v_3
theorem find_v3_using_Horner : 
  let x := 3,
      v0 := 4,
      v1 := v0 * x + 0,
      v2 := v1 * x - 3,
      v3 := v2 * x + 2
  in v3 = 101 :=
by
  let x := 3
  let v0 := 4
  let v1 := v0 * x + 0
  let v2 := v1 * x - 3
  let v3 := v2 * x + 2
  show v3 = 101
  sorry

end find_v3_using_Horner_l793_793766


namespace valid_statements_l793_793279

noncomputable def hyperbola := {P : ℝ × ℝ // ∃ (x y : ℝ), (x, y) = P ∧ x^2 / 16 - y^2 / 9 = 1}
def foci_dist : ℝ := 5
def area_triangle (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) : ℝ := 20
def on_hyperbola (P : ℝ × ℝ) : Prop := ∃ (x y : ℝ), (x, y) = P ∧ x^2 / 16 - y^2 / 9 = 1

theorem valid_statements (P : ℝ × ℝ) (F₁ F₂: ℝ × ℝ) (h1: on_hyperbola P) (h2: area_triangle P F₁ F₂ = 20) :
  (|y_P| = 4) ∧ (|PF₁| + |PF₂| = 50 / 3) := by
sorry

end valid_statements_l793_793279


namespace determine_f_value_l793_793041

-- Define initial conditions
def parabola_eqn (d e f : ℝ) (y : ℝ) : ℝ := d * y^2 + e * y + f
def vertex : (ℝ × ℝ) := (2, -3)
def point_on_parabola : (ℝ × ℝ) := (7, 0)

-- Prove that f = 7 given the conditions
theorem determine_f_value (d e f : ℝ) :
  (parabola_eqn d e f (vertex.snd) = vertex.fst) ∧
  (parabola_eqn d e f (point_on_parabola.snd) = point_on_parabola.fst) →
  f = 7 := 
by
  sorry 

end determine_f_value_l793_793041


namespace two_digit_numbers_less_than_35_l793_793359

theorem two_digit_numbers_less_than_35 : 
  ∃ n, n ∈ finset.range 90 ∧ (10 ≤ n + 10 ∧ n + 10 ≤ 99) ∧ 
  (let tens_digit := (n + 10) / 10 in 
   let units_digit := (n + 10) % 10 in 
   tens_digit < 3 ∨ units_digit < 5) → 
  nat.card {n | n ∈ finset.range 90 ∧ (10 ≤ n + 10 ∧ n + 10 ≤ 99) ∧ 
                    (let tens_digit := (n + 10) / 10 in 
                     let units_digit := (n + 10) % 10 in 
                     tens_digit < 3 ∨ units_digit < 5)} = 55
:= sorry

end two_digit_numbers_less_than_35_l793_793359


namespace line_intersects_ellipse_with_conditions_l793_793299

theorem line_intersects_ellipse_with_conditions :
  ∃ l : ℝ → ℝ, (∃ A B : ℝ × ℝ, 
  (A.fst^2/6 + A.snd^2/3 = 1 ∧ B.fst^2/6 + B.snd^2/3 = 1) ∧
  A.fst > 0 ∧ A.snd > 0 ∧ B.fst > 0 ∧ B.snd > 0 ∧
  (∃ M N : ℝ × ℝ, 
    M.snd = 0 ∧ N.fst = 0 ∧
    M.fst^2 + N.snd^2 = (2 * Real.sqrt 3)^2 ∧
    (M.snd - A.snd)^2 + (M.fst - A.fst)^2 = (N.fst - B.fst)^2 + (N.snd - B.snd)^2) ∧
    (∀ x, l x + Real.sqrt 2 * x - 2 * Real.sqrt 2 = 0)
) :=
sorry

end line_intersects_ellipse_with_conditions_l793_793299


namespace marbles_game_winning_strategy_l793_793529

theorem marbles_game_winning_strategy :
  ∃ k : ℕ, 1 < k ∧ k < 1024 ∧ (k = 4 ∨ k = 24 ∨ k = 40) := sorry

end marbles_game_winning_strategy_l793_793529


namespace perfect_four_digit_numbers_count_l793_793437

theorem perfect_four_digit_numbers_count :
  (number_of_perfect_four_digit_numbers_greater_than 2017 [0, 1, 2, 3, 4, 5, 6, 7] 10) = 71 :=
sorry

end perfect_four_digit_numbers_count_l793_793437


namespace trigonometric_identity_l793_793786

-- Declaration of the point and the calculations for sine and cosine
def point : (ℝ × ℝ) := (-5, 12)

-- Definition of radius
def radius (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

-- Calculations of sine and cosine
def sin_alpha : ℝ := point.snd / radius point.fst point.snd
def cos_alpha : ℝ := point.fst / radius point.fst point.snd

-- Statement that needs to be proven
theorem trigonometric_identity : sin_alpha + 2 * cos_alpha = 2/13 := by
  sorry

end trigonometric_identity_l793_793786


namespace maya_height_in_centimeters_maya_height_in_feet_l793_793514

def inches_to_centimeters (inches: ℝ): ℝ := inches * 2.54
def inches_to_feet (inches: ℝ): ℝ := inches / 12

theorem maya_height_in_centimeters: inches_to_centimeters 72 = 182.9 :=
by
  sorry

theorem maya_height_in_feet: inches_to_feet 72 = 6 :=
by
  sorry

end maya_height_in_centimeters_maya_height_in_feet_l793_793514


namespace ellipse_focal_length_l793_793430

theorem ellipse_focal_length (m : ℝ) (h : m > m - 1)
  (h_ellipse : ∀ (x y : ℝ), x^2 / (m - 1) + y^2 / m = 1) : 
  2 * real.sqrt (m - 1) = 2 :=
by
  sorry

end ellipse_focal_length_l793_793430


namespace right_handed_players_total_l793_793642

theorem right_handed_players_total (total_players throwers : ℕ) (non_throwers: ℕ := total_players - throwers)
  (left_handed_non_throwers : ℕ := non_throwers / 3)
  (right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers)
  (all_throwers_right_handed : throwers = 37)
  (total_players_55 : total_players = 55)
  (one_third_left_handed : left_handed_non_throwers = non_throwers / 3)
  (right_handed_total: ℕ := throwers + right_handed_non_throwers)
  : right_handed_total = 49 := by
  sorry

end right_handed_players_total_l793_793642


namespace star_photograph_l793_793879

variable {α : Type*} [LinearOrder α]

/--
Let I be a finite set of closed intervals on real numbers. 
Suppose that for any k > 1 intervals in I, at least 2 of them overlap. 
We can find k-1 points such that each interval in I contains at least one of these points.
-/
theorem star_photograph (I : finset (set.Icc α α)) 
  (k : ℕ) (hk : k > 1) (hI : ∀ s ⊆ I, s.card = k → ∃ (i j ∈ s), i ∩ j ≠ ∅) :
  ∃ P : finset α, P.card = k-1 ∧ ∀ i ∈ I, ∃ p ∈ P, p ∈ i :=
sorry

end star_photograph_l793_793879


namespace exists_zero_point_in_interval_l793_793794

def f (x : ℝ) : ℝ := 2^x + x / 3

theorem exists_zero_point_in_interval :
  ∃ x ∈ Ioo (-2 : ℝ) (-1 : ℝ), f x = 0 :=
by
  sorry

end exists_zero_point_in_interval_l793_793794


namespace x2004_y2004_l793_793396

theorem x2004_y2004 (x y : ℝ) (h1 : x - y = 2) (h2 : x^2 + y^2 = 4) : 
  x^2004 + y^2004 = 2^2004 := 
by
  sorry

end x2004_y2004_l793_793396


namespace triangle_angle_measure_triangle_area_l793_793769

-- Part (I): Proving the measure of angle A 
theorem triangle_angle_measure (A : ℝ) (h : sin (A - π / 6) = cos A) :
  A = π / 3 :=
by
  sorry

-- Part (II): Proving the area of the triangle
theorem triangle_area (a b c S A : ℝ) (ha : a = 1) (hs : b + c = 2) (hA : A = π / 3) :
  S = sqrt 3 / 4 :=
by
  sorry

end triangle_angle_measure_triangle_area_l793_793769


namespace find_line_equation_l793_793290

def ellipse (x y : ℝ) : Prop := (x ^ 2) / 6 + (y ^ 2) / 3 = 1

def meets_first_quadrant (l : Line) : Prop :=
  ∃ A B : Point, ellipse A.x A.y ∧ ellipse B.x B.y ∧ 
  A.x > 0 ∧ A.y > 0 ∧ B.x > 0 ∧ B.y > 0 ∧ l.contains A ∧ l.contains B

def intersects_axes (l : Line) : Prop :=
  ∃ M N : Point, M.y = 0 ∧ N.x = 0 ∧ l.contains M ∧ l.contains N
  
def equal_distances (M N A B : Point) : Prop :=
  dist M A = dist N B

def distance_MN (M N : Point) : Prop :=
  dist M N = 2 * Real.sqrt 3

theorem find_line_equation (l : Line) (A B M N : Point)
  (h1 : meets_first_quadrant l)
  (h2 : intersects_axes l)
  (h3 : equal_distances M N A B)
  (h4 : distance_MN M N) :
  l.equation = "x + sqrt(2) * y - 2 * sqrt(2) = 0" :=
sorry

end find_line_equation_l793_793290


namespace find_y_value_l793_793417

-- Define the given conditions and the final question in Lean
theorem find_y_value (k : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : y = k * x ^ (1/3)) 
  (h2 : y = 4 * real.sqrt 3)
  (x1 : x = 64) 
  : ∃ k, y = 2 * real.sqrt 3 :=
sorry

end find_y_value_l793_793417


namespace calculate_zorion_distance_l793_793973

noncomputable def zorion_halfway_distance : ℝ := 9

theorem calculate_zorion_distance :
  ∀ (orbit : Ellipse) (perigee apogee tilt : ℝ),
    perigee = 3 → apogee = 15 → tilt = 30 → 
    (orbit.semiMajorAxis = (perigee + apogee) / 2) → 
    let halfway_distance : ℝ := orbit.semiMajorAxis in
    halfway_distance = zorion_halfway_distance :=
begin
  sorry
end

end calculate_zorion_distance_l793_793973


namespace polynomial_coefficients_l793_793389

theorem polynomial_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) :
  (∀ (x : ℝ), (2 * x - 1) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) →
  (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 1 ∧ a_0 + a_2 + a_4 + a_6 = 365) :=
by 
  assume h : ∀ (x : ℝ), (2 * x - 1) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6,
  sorry

end polynomial_coefficients_l793_793389


namespace highest_annual_increase_in_sales_in_1994_l793_793557

def sales (year : Nat) : Real :=
  match year with
  | 1990 => 2
  | 1991 => 2.5
  | 1992 => 3
  | 1993 => 3.5
  | 1994 => 5
  | 1995 => 6
  | 1996 => 6.5
  | 1997 => 7
  | 1998 => 7.3
  | 1999 => 8
  | _    => 0

theorem highest_annual_increase_in_sales_in_1994 :
  ∃ year, (∀ y, y ≠ 1994 -> ((sales (y + 1) - sales y) ≤ (sales (1994 + 1) - sales 1994))) :=
by
  sorry

end highest_annual_increase_in_sales_in_1994_l793_793557


namespace probability_of_perfect_square_sum_l793_793610

def is_perfect_square (n : ℕ) : Prop :=
  n = 1*1 ∨ n = 2*2 ∨ n = 3*3 ∨ n = 4*4

theorem probability_of_perfect_square_sum :
  let total_outcomes := 64
  let perfect_square_sums := 12
  (perfect_square_sums / total_outcomes : ℚ) = 3 / 16 :=
by
  sorry

end probability_of_perfect_square_sum_l793_793610


namespace completing_the_square_sum_l793_793029

theorem completing_the_square_sum :
  ∃ (a b c : ℤ), 64 * (x : ℝ) ^ 2 + 96 * x - 81 = 0 ∧ a > 0 ∧ (8 * x + 6) ^ 2 = c ∧ a = 8 ∧ b = 6 ∧ a + b + c = 131 :=
by
  sorry

end completing_the_square_sum_l793_793029


namespace problem_statement_l793_793178

theorem problem_statement (f : ℝ → ℝ) (h_even : ∀ x : ℝ, f x = f (-x)) 
    (h_decreasing : ∀ x y : ℝ, 0 < x → 0 < y → x > y → f x < f y) :
    ∀ (x1 x2 : ℝ), x1 < 0 → x1 + x2 > 0 → f (-x1) > f (-x2) :=
by
  intros x1 x2 h1 h2
  have hx1_pos : 0 < -x1 := by linarith
  have hx2 : -x1 > x2 := by linarith
  have hx2_pos : 0 < x2 := by linarith [h2, h1]
  exact h_decreasing (-x1) x2 hx1_pos hx2_pos hx2

end problem_statement_l793_793178


namespace sum_numerator_denominator_l793_793097

-- Given the repeating decimal 0.343434...
def repeating_decimal : ℚ := 34 / 99

-- Prove that the decimal's fraction form in lowest terms adds up to 133.
theorem sum_numerator_denominator : 
  let num := repeating_decimal.num in 
  let denom := repeating_decimal.denom in 
  num + denom = 133 :=
by
  have x : repeating_decimal = 34 / 99 := by sorry
  sorry   -- Placeholder for the proof steps to demonstrate num and denom values lead to 133.

end sum_numerator_denominator_l793_793097


namespace numWaysToPaintDoors_l793_793083

-- Define the number of doors and choices per door
def numDoors : ℕ := 3
def numChoicesPerDoor : ℕ := 2

-- Theorem statement that we want to prove
theorem numWaysToPaintDoors : numChoicesPerDoor ^ numDoors = 8 := by
  sorry

end numWaysToPaintDoors_l793_793083


namespace bees_hatch_every_day_l793_793152

   /-- 
   Given:
   - The queen loses 900 bees every day.
   - The initial number of bees is 12500.
   - After 7 days, the total number of bees is 27201.
   
   Prove:
   - The number of bees hatching from the queen's eggs every day is 3001.
   -/
   
   theorem bees_hatch_every_day :
     ∃ x : ℕ, 12500 + 7 * (x - 900) = 27201 → x = 3001 :=
   sorry
   
end bees_hatch_every_day_l793_793152


namespace count_divisible_by_9_l793_793728

def alter_digit (digit : ℕ) : ℕ :=
  digit + 2

def f (n : ℕ) : ℕ :=
  (1 + n) * n / 2

def a_k (k : ℕ) : ℕ :=
  -- The function to compute the concatenated value which will be very complex. 
  -- For now, we mock it here for demonstration purposes.
  sorry 

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum 

def is_divisible_by_9 (n : ℕ) : Prop :=
  digit_sum n % 9 = 0

theorem count_divisible_by_9 : 
  ∃ count : ℕ, count = (Finset.range 100).count (λ k => is_divisible_by_9 (a_k k)) :=
sorry

end count_divisible_by_9_l793_793728


namespace magnitude_b_l793_793837

noncomputable theory

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Conditions
def mag_a : real := ‖a‖ = 3
def mag_diff_ab : real := ‖a - b‖ = 5
def dot_ab : real := inner a b = 1

-- Goal
theorem magnitude_b (h1 : ‖a‖ = 3) (h2 : ‖a - b‖ = 5) (h3 : inner a b = 1) : ‖b‖ = 3 * real.sqrt 2 :=
by 
  sorry

end magnitude_b_l793_793837


namespace remainder_of_3n_mod_9_l793_793233

theorem remainder_of_3n_mod_9 (n : ℕ) (h : n % 9 = 7) : (3 * n) % 9 = 3 :=
by
  sorry

end remainder_of_3n_mod_9_l793_793233


namespace solution_exists_l793_793328

open Int

def isPrime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d ∣ p, d = 1 ∨ d = p

theorem solution_exists :
  ∃ (p : ℕ), isPrime p ∧ ∃ (x : ℤ), (2 * x^2 - x - 36 = p^2 ∧ x = 13) := sorry

end solution_exists_l793_793328


namespace num_real_solutions_l793_793811

theorem num_real_solutions (x : ℝ) : 
  (∃ x1 x2 x3 x4 : ℝ, (x^2 - 7)^2 = 18 ∧ 
    x1 = sqrt(7 + 3 * sqrt(2)) ∧ x2 = -sqrt(7 + 3 * sqrt(2)) ∧ 
    x3 = sqrt(7 - 3 * sqrt(2)) ∧ x4 = -sqrt(7 - 3 * sqrt(2)) ∧ 
    x ∈ {x1, x2, x3, x4} ∧ 
    ({x1, x2, x3, x4}.card = 4)) := 
  sorry

end num_real_solutions_l793_793811


namespace constant_function_solution_l793_793646

theorem constant_function_solution (f : ℝ → ℝ)
  (h_diff : ∀ x, differentiable_at ℝ f x)
  (h_diff2 : ∀ x, differentiable_at ℝ (λ x => (deriv f) x) x)
  (h_cond : ∀ x, (deriv f x) ^ 2 + deriv (deriv f) x ≤ 0) :
  ∃ C : ℝ, ∀ x, f x = C :=
by
  sorry

end constant_function_solution_l793_793646


namespace problem1_problem2_l793_793322

-- Given line and ellipse conditions
variables (a b : ℝ) (A B : ℝ × ℝ)
variables (e : ℝ) (c : ℝ)
variables (y : ℝ → ℝ)

-- Given conditions
def line_eq (x : ℝ) : ℝ := -x + 1
def ellipse_eq (x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Specific conditions for problem 1
axiom a_val : a = sqrt 2
axiom c_val : c = 1
axiom b_val : b = 1

-- Specific conditions for problem 2
axiom e_range : ∀ e, 1/2 ≤ e ∧ e ≤ sqrt 2 / 2

-- Length of line segment AB is given by
def length_AB : ℝ := dist A B

-- Problem 1 statement
theorem problem1 : length_AB (A:= (4 / 3, -1 / 3)) (B:= (0, 1)) = (4 / 3) * sqrt 2 := sorry

-- Problem 2 statement
theorem problem2 : ∀ (max_length : ℝ), max_length = sqrt 6 → 
  (a^2 = (1/2) * (1 + (1 / (1 - e^2)))) → 
  e_range e →
  max_length = 2 * a := sorry

end problem1_problem2_l793_793322


namespace distributeCandies_l793_793887

-- Define the conditions as separate definitions.

-- Number of candies
def candies : ℕ := 10

-- Number of boxes
def boxes : ℕ := 5

-- Condition that each box gets at least one candy
def atLeastOne (candyDist : Fin boxes → ℕ) : Prop :=
  ∀ b, candyDist b > 0

-- Function to count the number of ways to distribute candies
noncomputable def countWaysToDistribute (candies : ℕ) (boxes : ℕ) : ℕ :=
  -- Function to compute the number of ways
  -- (assuming a correct implementation is provided)
  sorry -- Placeholder for the actual counting implementation

-- Theorem to prove the number of distributions
theorem distributeCandies : countWaysToDistribute candies boxes = 7 := 
by {
  -- Proof omitted
  sorry
}

end distributeCandies_l793_793887


namespace sum_numerator_denominator_repeating_decimal_l793_793122

theorem sum_numerator_denominator_repeating_decimal :
  let x := 34 / 99 in
  x.denom * x + x.num = 133 :=
by
  let x : ℚ := 34 / 99
  sorry

end sum_numerator_denominator_repeating_decimal_l793_793122


namespace bars_per_set_correct_l793_793250

-- Define the total number of metal bars and the number of sets
def total_metal_bars : ℕ := 14
def number_of_sets : ℕ := 2

-- Define the function to compute bars per set
def bars_per_set (total_bars : ℕ) (sets : ℕ) : ℕ :=
  total_bars / sets

-- The proof statement
theorem bars_per_set_correct : bars_per_set total_metal_bars number_of_sets = 7 := by
  sorry

end bars_per_set_correct_l793_793250


namespace vector_magnitude_l793_793841

variable {V : Type*} [InnerProductSpace ℝ V]

theorem vector_magnitude
  (a b : V)
  (h1 : ‖a‖ = 3)
  (h2 : ‖a - b‖ = 5)
  (h3 : inner a b = 1) :
  ‖b‖ = 3 * Real.sqrt 2 :=
by
  sorry

end vector_magnitude_l793_793841


namespace company_p_employees_december_l793_793719

theorem company_p_employees_december :
  let january_employees := 434.7826086956522
  let percent_more := 0.15
  let december_employees := january_employees + (percent_more * january_employees)
  december_employees = 500 :=
by
  sorry

end company_p_employees_december_l793_793719


namespace fraction_of_journey_by_bus_l793_793669

-- Define the total journey, the fraction of journey by rail, and the distance by foot
def total_journey : ℝ := 130
def journey_by_rail_fraction : ℝ := 3 / 5
def distance_on_foot : ℝ := 6.5

-- Calculate the distance by rail
def distance_by_rail := journey_by_rail_fraction * total_journey

-- Calculate the remaining distance by bus
def distance_by_bus := total_journey - (distance_by_rail + distance_on_foot)

-- Calculate the fraction of the journey by bus
def journey_by_bus_fraction := distance_by_bus / total_journey

-- The theorem to prove
theorem fraction_of_journey_by_bus :
  journey_by_bus_fraction = 45.5 / 130 :=
by
  sorry

end fraction_of_journey_by_bus_l793_793669


namespace European_to_American_swallow_ratio_l793_793953

theorem European_to_American_swallow_ratio (a e : ℝ) (n_E : ℕ) 
  (h1 : a = 5)
  (h2 : 2 * n_E + n_E = 90)
  (h3 : 60 * a + 30 * e = 600) :
  e / a = 2 := 
by
  sorry

end European_to_American_swallow_ratio_l793_793953


namespace number_of_distinct_products_l793_793818

def S : Set ℕ := {2, 3, 5, 7, 11, 13}

theorem number_of_distinct_products: 
  (∃ (P : Finset ℕ), P = 
    {p | ∃ (a b ∈ S), a ≠ b ∧ 
    p = a * b ∨ 
    (∃ (c ∈ S), c ≠ a ∧ c ≠ b ∧ p = a * b * c) ∨ 
    (∃ (d ∈ S), d ≠ a ∧ d ≠ b ∧ d ≠ c ∧ p = a * b * c * d) ∨ 
    (∃ (e ∈ S), e ≠ a ∧ e ≠ b ∧ e ≠ c ∧ e ≠ d ∧ p = a * b * c * d * e) ∨ 
    (∃ (f ∈ S), f ≠ a ∧ f ≠ b ∧ f ≠ c ∧ f ≠ d ∧ f ≠ e ∧ p = a * b * c * d * e * f) 
    } ∧ P.card = 57)
:= sorry

end number_of_distinct_products_l793_793818


namespace angle_C_is_150_degrees_l793_793707

theorem angle_C_is_150_degrees
  (C D : ℝ)
  (h_supp : C + D = 180)
  (h_C_5D : C = 5 * D) :
  C = 150 :=
by
  sorry

end angle_C_is_150_degrees_l793_793707


namespace probability_of_perfect_square_sum_l793_793598

-- Define the conditions
def dice_faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def perfect_squares_in_range : Finset ℕ := {4, 9, 16}
def total_outcomes : ℕ := 8 * 8  -- total number of outcomes when rolling two dice

-- Calculate the number of favorable outcomes
def favorable_outcomes : ℕ :=
  (dice_faces.product dice_faces).filter (λ pair => perfect_squares_in_range.member (pair.1 + pair.2)).card

-- Calculate the probability of getting a perfect square sum
noncomputable def probability_perfect_square_sum : ℚ :=
  favorable_outcomes / total_outcomes

-- The statement to prove that the probability is 3/16
theorem probability_of_perfect_square_sum :
  probability_perfect_square_sum = 3 / 16 :=
begin
  sorry
end

end probability_of_perfect_square_sum_l793_793598


namespace nathaniel_wins_probability_l793_793923

noncomputable def probability_nathaniel_wins : ℚ :=
  5 / 11

theorem nathaniel_wins_probability :
  (∃ (turns : ℕ → ℕ) (running_tally : ℕ → ℕ),
    running_tally 0 = 0 ∧
    (∀ n, running_tally (n + 1) = running_tally n + (turns n % 6 + 1)) ∧
    (∃ n, running_tally n % 7 = 0 ∧ ∀ m < n, running_tally m % 7 ≠ 0) ∧
    (∀ k, (∃ m, m % 7 = k) → k ∈ {1, 2, 3, 4, 5, 6, 0})
  ) →
  probability_nathaniel_wins = 5 / 11 :=
by sorry

end nathaniel_wins_probability_l793_793923


namespace find_line_equation_l793_793295

theorem find_line_equation (x y : ℝ) : 
  (∃ A B, (A.x^2 / 6 + A.y^2 / 3 = 1) ∧ (B.x^2 / 6 + B.y^2 / 3 = 1) ∧
  (A.x > 0 ∧ A.y > 0) ∧ (B.x > 0 ∧ B.y > 0) ∧
  let M := (-B.y, 0) in
  let N := (0, B.y) in 
  (abs (M.x - A.x) = abs (N.y - B.y)) ∧ 
  (abs (M.x - N.x + M.y - N.y) = 2 * sqrt 3)) →
  x + sqrt 2 * y - 2 * sqrt 2 = 0 := 
sorry

end find_line_equation_l793_793295


namespace count_lines_with_slope_conditions_l793_793797

theorem count_lines_with_slope_conditions :
  let s := {-3, -2, -1, 0, 1, 2}
  ∃ (n : ℕ), n = 16 ∧
  ∃ (lines : Finset (ℚ × ℚ)),
    lines.card = n ∧
    ∀ (a b : ℚ), (a, b) ∈ lines →
      a ∈ s ∧ b ∈ s ∧ a ≠ b ∧ |a / b| > Real.sqrt 3 :=
begin
  let s := Finset.ofList [-3, -2, -1, 0, 1, 2],
  use 16,
  refine ⟨_, _, _⟩,
  sorry,  -- Placeholder for the actual construction and proof
end

end count_lines_with_slope_conditions_l793_793797


namespace selection_ways_at_least_one_girl_l793_793758

def boys : Nat := 4
def girls : Nat := 3

theorem selection_ways_at_least_one_girl : 
  ∑ k in ({1, 2, 3} : Finset ℕ), (Nat.choose girls k) * (Nat.choose boys (4 - k)) = 34 :=
by 
  sorry

end selection_ways_at_least_one_girl_l793_793758


namespace shortest_chord_length_l793_793505

theorem shortest_chord_length {k : ℝ} :
  let l := fun t : ℝ => (1 + t, 1 + k * t) in
  let C := fun θ : ℝ => 2 * Real.cos θ + 4 * Real.sin θ in
  ∀ t : ℝ, (l t).fst = 1 + t ∧ (l t).snd = 1 + k * t →
  (∃ (x y : ℝ), ((x-1)^2 + (y-2)^2 = 5) ∧ (k*x - y - k = 0)) →
  (2 = 2) :=
sorry

end shortest_chord_length_l793_793505


namespace sampling_interval_divisor_l793_793835

theorem sampling_interval_divisor (P : ℕ) (hP : P = 524) (k : ℕ) (hk : k ∣ P) : k = 4 :=
by
  sorry

end sampling_interval_divisor_l793_793835


namespace expected_value_correct_l793_793171

noncomputable def even_outcome (n : ℕ) : ℕ :=
  if n % 2 = 0 then n else 0

noncomputable def probability (n : ℕ) : ℝ :=
  1 / 8

noncomputable def expected_value : ℝ :=
  ∑ i in (Finset.range 8).map ⟨λ n, n + 1, Nat.succ_injective⟩, (probability i) * (even_outcome i)

theorem expected_value_correct : expected_value = 2.50 := by
  sorry

end expected_value_correct_l793_793171


namespace rectangle_diagonal_angles_l793_793260

theorem rectangle_diagonal_angles 
    (A B C D : ℤ × ℤ) -- Vertices of the rectangle on integer coordinates
    (is_rect : ∃ O, (O = (A + C) / 2) ∧ (O = (B + D) / 2)) -- condition of rectangle
    (α : ℝ) -- angle between diagonals
    (cos_α_tan_α_rational : 
        (∃ cos_α : ℚ, cos_α = real.cos α) ∧ 
        (∃ tan_α : ℚ, tan_α = real.tan α)) :

    (∃ cos_α : ℚ, cos_α = real.cos α) ∧ 
    (∃ sin_α : ℚ, sin_α = real.sin α) :=
sorry

end rectangle_diagonal_angles_l793_793260


namespace num_sequences_l793_793078

open Function

def transformations_t :=
  {L : Equiv.Perm (Fin 3), R : Equiv.Perm (Fin 3), H : Equiv.Perm (Fin 3), V : Equiv.Perm (Fin 3)}

variables (t : transformations_t)
  (ALeqC : t.L (Fin.ofNat 0) = Fin.ofNat 2)
  (CLetB : t.L (Fin.ofNat 2) = Fin.ofNat 1)
  (BLetA : t.L (Fin.ofNat 1) = Fin.ofNat 0)
  (AReB : t.R (Fin.ofNat 0) = Fin.ofNat 1)
  (BReC : t.R (Fin.ofNat 1) = Fin.ofNat 2)
  (CReA : t.R (Fin.ofNat 2) = Fin.ofNat 0)
  (AHeC : t.H (Fin.ofNat 0) = Fin.ofNat 2)
  (CHeA : t.H (Fin.ofNat 2) = Fin.ofNat 0)
  (BHeB : t.H (Fin.ofNat 1) = Fin.ofNat 1)
  (AWeB : t.V (Fin.ofNat 0) = Fin.ofNat 1)
  (BWeA : t.V (Fin.ofNat 1) = Fin.ofNat 0)
  (CWeC : t.V (Fin.ofNat 2) = Fin.ofNat 2)

theorem num_sequences (t : transformations_t) : (∃ L R H V : ℕ, L + R + H + V = 30 ∧ 
  L % 3 = 0 ∧ R % 3 = 0 ∧ H % 2 = 0 ∧ V % 2 = 0 ∧ 
  L / 3 + R / 3 + H / 2 + V / 2 = 15 ∧ 
  nat.choose 18 3 = 816) :=
by
  sorry

end num_sequences_l793_793078


namespace nested_g_value_l793_793795

def g (x : ℝ) : ℝ := 1 / (x + 1)

theorem nested_g_value :
  g (g (g (g (g (g 5))))) = 33 / 53 :=
by sorry

end nested_g_value_l793_793795


namespace value_of_y_at_x8_l793_793413

theorem value_of_y_at_x8
  (k : ℝ)
  (y : ℝ → ℝ)
  (hx64 : y 64 = 4 * Real.sqrt 3)
  (hy_def : ∀ x, y x = k * x^(1 / 3)) :
  y 8 = 2 * Real.sqrt 3 :=
by {
  sorry,
}

end value_of_y_at_x8_l793_793413


namespace numOddFunctions_l793_793324

-- Define the set of exponents
def exponents : Set ℚ := {-2, -1, -1/2, 1/3, 1/2, 1, 2, 3}

-- Define the condition for a function to be odd
def isOddFunction (a : ℚ) : Prop := ∀ x : ℝ, -x ^ a = - (x ^ a)

-- Filter to get only the exponents that make the function odd
def oddExponents : Set ℚ := {a ∈ exponents | isOddFunction a}

-- The main theorem to be proven
theorem numOddFunctions : oddExponents.card = 4 :=
sorry

end numOddFunctions_l793_793324


namespace quadratic_polynomial_correct_l793_793617

noncomputable def roots (b c : ℝ) : list ℝ := 
let Δ := b^2 - 4 * 1 * c in
if h : Δ ≥ 0 then
  let sqrt_Δ := real.sqrt Δ in
  [(-b + sqrt_Δ) / (2 * 1), (-b - sqrt_Δ) / (2 * 1)]
else
  [] -- when Δ < 0, no real roots

theorem quadratic_polynomial_correct: 
  (∀ (b c : ℝ), 
  (roots b c).length = 2 → 
  b + c + (roots b c).sum = -3 ∧ (b * c * (roots b c)).prod = 36) ↔
  (b = 4 ∧ c = -3) :=
by
  sorry

end quadratic_polynomial_correct_l793_793617


namespace num_circular_arrangements_l793_793070

-- Define the number of keys
def num_keys : ℕ := 5

-- Define the proposition that the number of unique arrangements is 24
theorem num_circular_arrangements (h_keys: ℕ = num_keys) : 
  (num_keys - 1)! = 24 := 
by 
  sorry

end num_circular_arrangements_l793_793070


namespace right_triangle_area_l793_793930

theorem right_triangle_area (a b c r : ℝ) (h1 : a = 15) (h2 : r = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_right : a ^ 2 + b ^ 2 = c ^ 2) (h_incircle : r = (a + b - c) / 2) : 
  1 / 2 * a * b = 60 :=
by
  sorry

end right_triangle_area_l793_793930


namespace parallel_line_eq_l793_793087

theorem parallel_line_eq (a b c : ℝ) (p1 p2 : ℝ) :
  (∃ m b1 b2, 3 * a + 6 * b * p1 = 12 ∧ p2 = - (1 / 2) * p1 + b1 ∧
    - (1 / 2) * p1 - m * p1 = b2) → 
    (∃ b', p2 = - (1 / 2) * p1 + b' ∧ b' = 0) := 
sorry

end parallel_line_eq_l793_793087


namespace astronaut_days_on_orbius_l793_793526

noncomputable def days_in_year : ℕ := 250
noncomputable def seasons_in_year : ℕ := 5
noncomputable def seasons_stayed : ℕ := 3

theorem astronaut_days_on_orbius :
  (days_in_year / seasons_in_year) * seasons_stayed = 150 := by
  sorry

end astronaut_days_on_orbius_l793_793526


namespace cos_2_alpha_tan_alpha_minus_beta_l793_793274

noncomputable theory

variables (α β : ℝ)
-- Conditions
def acute_angles (α β : ℝ) : Prop := (0 < α ∧ α < π / 2) ∧ (0 < β ∧ β < π / 2)
def tan_alpha := tan α = 4 / 3
def cos_alpha_plus_beta := cos (α + β) = - sqrt 5 / 5

-- Proof statements
theorem cos_2_alpha (h1 : acute_angles α β) (h2: tan_alpha α) : cos (2 * α) = -7 / 25 :=
by sorry

theorem tan_alpha_minus_beta (h1 : acute_angles α β) (h2: tan_alpha α) (h3: cos_alpha_plus_beta α β) : tan (α - β) = -2 / 11 :=
by sorry

end cos_2_alpha_tan_alpha_minus_beta_l793_793274


namespace sum_of_numerator_and_denominator_l793_793111

def repeating_decimal_represents_fraction : Prop :=
  let x := 34 / 99
  0.34̅ = x

/-- The sum of the numerator and denominator of 0.34̅ in its lowest terms is 133. -/
theorem sum_of_numerator_and_denominator
  (x : ℚ)
  (hx : repeating_decimal_represents_fraction) :
  (34 + 99 = 133) :=
begin
  sorry
end

end sum_of_numerator_and_denominator_l793_793111


namespace train_cross_time_in_seconds_l793_793167

-- Definitions based on conditions
def train_speed_kph : ℚ := 60
def train_length_m : ℚ := 450

-- Statement: prove that the time to cross the pole is 27 seconds
theorem train_cross_time_in_seconds (train_speed_kph train_length_m : ℚ) :
  train_speed_kph = 60 →
  train_length_m = 450 →
  (train_length_m / (train_speed_kph * 1000 / 3600)) = 27 :=
by
  intros h_speed h_length
  rw [h_speed, h_length]
  sorry

end train_cross_time_in_seconds_l793_793167


namespace sum_numerator_denominator_l793_793093

-- Given the repeating decimal 0.343434...
def repeating_decimal : ℚ := 34 / 99

-- Prove that the decimal's fraction form in lowest terms adds up to 133.
theorem sum_numerator_denominator : 
  let num := repeating_decimal.num in 
  let denom := repeating_decimal.denom in 
  num + denom = 133 :=
by
  have x : repeating_decimal = 34 / 99 := by sorry
  sorry   -- Placeholder for the proof steps to demonstrate num and denom values lead to 133.

end sum_numerator_denominator_l793_793093


namespace find_other_roots_l793_793885

-- Defining the polynomial equation
def poly_eq (t : ℝ) : ℝ :=
  32 * t ^ 5 - 40 * t ^ 3 + 10 * t - Real.sqrt 3

-- Conditions given in the problem
axiom root1 : poly_eq (Real.cos (6 * Real.pi / 180)) = 0

-- Stating the theorem that needs to be proved
theorem find_other_roots :
  poly_eq (Real.cos (78 * Real.pi / 180)) = 0 ∧ 
  poly_eq (Real.cos (150 * Real.pi / 180)) = 0 ∧ 
  poly_eq (Real.cos (222 * Real.pi / 180)) = 0 ∧ 
  poly_eq (Real.cos (294 * Real.pi / 180)) = 0 :=
sorry

end find_other_roots_l793_793885


namespace triangle_area_DEF_l793_793998

def point : Type := ℝ × ℝ

def D : point := (-2, 2)
def E : point := (8, 2)
def F : point := (6, -4)

theorem triangle_area_DEF :
  let base : ℝ := abs (D.1 - E.1)
  let height : ℝ := abs (F.2 - 2)
  let area := 1/2 * base * height
  area = 30 := 
by 
  sorry

end triangle_area_DEF_l793_793998


namespace magnitude_of_difference_l793_793808

noncomputable theory
open_locale classical

variables (a b : ℝ × ℝ)
variables (h₁ : a = (1, real.sqrt 3))
variables (h₂ : real.sqrt (b.1 * b.1 + b.2 * b.2) = 3)
variables (h₃ : a.1 * (a.1 - 2 * b.1) + a.2 * (a.2 - 2 * b.2) = 0)

theorem magnitude_of_difference : real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 3 :=
sorry

end magnitude_of_difference_l793_793808


namespace central_angle_of_sector_l793_793282

-- Definitions and conditions
def area_of_sector := (3 * Real.pi) / 8
def radius := 1

-- Theorem statement
theorem central_angle_of_sector
  (h_area : area_of_sector = (3 * Real.pi) / 8)
  (h_radius : radius = 1):
  ∃ θ, (area_of_sector = (1/2) * radius^2 * θ) ∧ θ = (3 * Real.pi) / 4 := by
  sorry

end central_angle_of_sector_l793_793282


namespace zach_babysitting_hours_l793_793135

theorem zach_babysitting_hours :
  ∀ (bike_cost weekly_allowance mowing_pay babysitting_rate saved_amount needed_additional_amount hours_needed : ℕ),
    bike_cost = 100 →
    weekly_allowance = 5 →
    mowing_pay = 10 →
    babysitting_rate = 7 →
    saved_amount = 65 →
    needed_additional_amount = 6 →
    saved_amount + weekly_allowance + mowing_pay + hours_needed * babysitting_rate = bike_cost - needed_additional_amount →
    hours_needed = 2 :=
by
  intros bike_cost weekly_allowance mowing_pay babysitting_rate saved_amount needed_additional_amount hours_needed
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end zach_babysitting_hours_l793_793135


namespace no_real_solution_l793_793219

theorem no_real_solution (x y : ℝ) (hx : x^2 = 1 + 1 / y^2) (hy : y^2 = 1 + 1 / x^2) : false :=
by
  sorry

end no_real_solution_l793_793219


namespace isosceles_triangle_altitudes_l793_793992

theorem isosceles_triangle_altitudes (ABC : Triangle) (h₁ h₂ : ℕ)
  (h₁_eq_6 : h₁ = 6) (h₂_eq_18 : h₂ = 18) (isosceles : ABC.is_isosceles) :
  ∃ h₃ : ℕ, h₃ ≤ 6 ∧ ∀ h : ℕ, h > h₃ → ¬ABC.has_altitude_of_length h :=
by
  sorry

end isosceles_triangle_altitudes_l793_793992


namespace roberts_test_score_l793_793005

structure ClassState where
  num_students : ℕ
  avg_19_students : ℕ
  class_avg_20_students : ℕ

def calculate_roberts_score (s : ClassState) : ℕ :=
  let total_19_students := s.num_students * s.avg_19_students
  let total_20_students := (s.num_students + 1) * s.class_avg_20_students
  total_20_students - total_19_students

theorem roberts_test_score 
  (state : ClassState) 
  (h1 : state.num_students = 19) 
  (h2 : state.avg_19_students = 74)
  (h3 : state.class_avg_20_students = 75) : 
  calculate_roberts_score state = 94 := by
  sorry

end roberts_test_score_l793_793005


namespace max_distinct_sums_l793_793578

/-- Given 3 boys and 20 girls standing in a row, each child counts the number of girls to their 
left and the number of boys to their right and adds these two counts together. Prove that 
the maximum number of different sums that the children could have obtained is 20. -/
theorem max_distinct_sums (boys girls : ℕ) (total_children : ℕ) 
  (h_boys : boys = 3) (h_girls : girls = 20) (h_total : total_children = boys + girls) : 
  ∃ (max_sums : ℕ), max_sums = 20 := 
by 
  sorry

end max_distinct_sums_l793_793578


namespace probability_green_ball_eq_l793_793221

noncomputable def prob_green_ball : ℚ := 
  1 / 3 * (5 / 18) + 1 / 3 * (1 / 2) + 1 / 3 * (1 / 2)

theorem probability_green_ball_eq : 
  prob_green_ball = 23 / 54 := 
  by
  sorry

end probability_green_ball_eq_l793_793221


namespace min_value_of_a_plus_b_l793_793785

-- Definitions based on the conditions
variables (a b : ℝ)
def roots_real (a b : ℝ) : Prop := a^2 ≥ 8 * b ∧ b^2 ≥ a
def positive_vars (a b : ℝ) : Prop := a > 0 ∧ b > 0
def min_a_plus_b (a b : ℝ) : Prop := a + b = 6

-- Lean theorem statement
theorem min_value_of_a_plus_b (a b : ℝ) (hr : roots_real a b) (pv : positive_vars a b) : min_a_plus_b a b :=
sorry

end min_value_of_a_plus_b_l793_793785


namespace length_DE_is_20_l793_793459

open_locale real
open_locale complex_conjugate

noncomputable def in_triangle_config_with_angle (ABC : Type) [triangle ABC]
  (BC_length : ℝ) (angle_C : real.angle) (perp_bisector_intersect_D : point)
  (perp_bisector_intersect_E : point) : Prop :=
  -- define the conditions
  BC_length = 40 ∧
  angle_C = 45 ∧
  is_midpoint_of_BC ABC perp_bisector_intersect_D ∧
  is_perpendicular_bisector_intersects_AC ABC perp_bisector_intersect_D perp_bisector_intersect_E

noncomputable def length_DE (D E: point) : ℝ :=
  distance_between D E

theorem length_DE_is_20 {ABC : Type} [triangle ABC]
  (BC_length : ℝ) (angle_C : real.angle) (perp_bisector_intersect_D : point)
  (perp_bisector_intersect_E : point)
  (h : in_triangle_config_with_angle ABC BC_length angle_C perp_bisector_intersect_D perp_bisector_intersect_E):
  length_DE perp_bisector_intersect_D perp_bisector_intersect_E = 20 :=
begin
  sorry,
end

end length_DE_is_20_l793_793459


namespace vector_magnitude_b_l793_793849

variables (a b : EuclideanSpace ℝ 3)
variables (h1 : ∥a∥ = 3) (h2 : ∥a - b∥ = 5) (h3 : inner a b = 1)

theorem vector_magnitude_b : ∥b∥ = 3 * real.sqrt 2 :=
by
  sorry

end vector_magnitude_b_l793_793849


namespace max_segment_length_in_rectangle_l793_793722

def rectABCD_max_segment_length (AB BC : ℝ) (hAB : AB = 8) (hBC : BC = 6) : ℝ :=
let AC := Real.sqrt (AB^2 + BC^2) in
let midpoint := AC / 2 in
let max_length := AB / 2 in
max_length

theorem max_segment_length_in_rectangle :
  ∀ (AB BC : ℝ), AB = 8 → BC = 6 → 
  rectABCD_max_segment_length AB BC (by simp) (by simp) = 4 :=
by
  intros AB BC hAB hBC
  unfold rectABCD_max_segment_length
  simp [hAB, hBC]
  sorry

end max_segment_length_in_rectangle_l793_793722


namespace find_x_for_prime_square_l793_793325

theorem find_x_for_prime_square (x p : ℤ) (hp : Prime p) (h : 2 * x^2 - x - 36 = p^2) : x = 13 ∧ p = 17 :=
by
  sorry

end find_x_for_prime_square_l793_793325


namespace thousandth_chime_date_l793_793658

-- Constants/Definitions related to the conditions and question
def chimes_per_hour (hour : ℕ) : ℕ := hour
def extra_chimes : ℕ := 2  -- 15 minutes and 45 minutes past each hour
def hours_per_day : ℕ := 24
def minutes_per_hour : ℕ := 60

-- Calculate the total number of chimes from a given starting hour to midnight
def chimes_from_start_to_midnight (start_hour : ℕ) : ℕ :=
  let rec loop (hour : ℕ) (acc : ℕ) :=
    if hour > 12 then acc + ((hour - 12) * (chimes_per_hour hour + extra_chimes))
    else acc + (chimes_per_hour hour + extra_chimes)
  loop (start_hour+1) 0

-- Calculate total daily chimes
def total_daily_chimes : ℕ :=
  let hourly_chimes := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12
  (hourly_chimes * 2) + (extra_chimes * hours_per_day)

-- Date calculation based on chimes
def date_of_chime (initial_hour : ℕ) (initial_date : ℕ) (total_chimes : ℕ) : ℕ :=
  let initial_chimes = chimes_from_start_to_midnight initial_hour
  let remaining_chimes = total_chimes - initial_chimes
  let full_days = remaining_chimes / total_daily_chimes
  let extra_chimes = remaining_chimes % total_daily_chimes
  initial_date + full_days + (if extra_chimes == 0 then 0 else 1)

-- Define the proof statement
theorem thousandth_chime_date :
  date_of_chime 15 15 1000 = 23 := by
  sorry

end thousandth_chime_date_l793_793658


namespace line_intersects_ellipse_with_conditions_l793_793301

theorem line_intersects_ellipse_with_conditions :
  ∃ l : ℝ → ℝ, (∃ A B : ℝ × ℝ, 
  (A.fst^2/6 + A.snd^2/3 = 1 ∧ B.fst^2/6 + B.snd^2/3 = 1) ∧
  A.fst > 0 ∧ A.snd > 0 ∧ B.fst > 0 ∧ B.snd > 0 ∧
  (∃ M N : ℝ × ℝ, 
    M.snd = 0 ∧ N.fst = 0 ∧
    M.fst^2 + N.snd^2 = (2 * Real.sqrt 3)^2 ∧
    (M.snd - A.snd)^2 + (M.fst - A.fst)^2 = (N.fst - B.fst)^2 + (N.snd - B.snd)^2) ∧
    (∀ x, l x + Real.sqrt 2 * x - 2 * Real.sqrt 2 = 0)
) :=
sorry

end line_intersects_ellipse_with_conditions_l793_793301


namespace correctness_of_solution_set_l793_793544

-- Define the set of real numbers satisfying the inequality
def solution_set : Set ℝ := { x | 3 ≤ |5 - 2 * x| ∧ |5 - 2 * x| < 9 }

-- Define the expected solution set derived from the problem
def expected_solution_set : Set ℝ := { x | -1 < x ∧ x ≤ 1 } ∪ { x | 2.5 < x ∧ x < 4.5 }

-- The proof statement
theorem correctness_of_solution_set : solution_set = expected_solution_set :=
  sorry

end correctness_of_solution_set_l793_793544


namespace days_spent_on_Orbius5_l793_793528

-- Define the conditions
def days_per_year : Nat := 250
def seasons_per_year : Nat := 5
def length_of_season : Nat := days_per_year / seasons_per_year
def seasons_stayed : Nat := 3

-- Theorem statement
theorem days_spent_on_Orbius5 : (length_of_season * seasons_stayed = 150) :=
by 
  -- Proof is skipped
  sorry

end days_spent_on_Orbius5_l793_793528


namespace cistern_length_l793_793154

theorem cistern_length (W H A : ℝ) (hW : W = 4) (hH : H = 1.25) (hA : A = 62) :
  ∃ L : ℝ, L = 8 ∧ A = (L * W) + (2 * L * H) + (2 * W * H) :=
by
  use 8
  split
  · rfl
  rw [hW, hH, hA]
  norm_num
  sorry

end cistern_length_l793_793154


namespace count_four_digit_even_numbers_l793_793812

def digitSet := {0, 2, 4, 6, 8}

def validFirstDigitSet := {2, 4, 6, 8}

noncomputable def countValidFourDigitEvenNumbers : Nat :=
  Set.card validFirstDigitSet * (Set.card digitSet * Set.card digitSet * Set.card digitSet)

theorem count_four_digit_even_numbers : countValidFourDigitEvenNumbers = 500 :=
by
  rw [countValidFourDigitEvenNumbers]
  norm_num
  sorry

end count_four_digit_even_numbers_l793_793812


namespace a_628_th_term_is_negative_l793_793222

def a (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), Real.sin k

theorem a_628_th_term_is_negative : 
  ∃ n, a n < 0 ∧ n = 628 ∧ (∑ k in Finset.range 628, if a k < 0 then 1 else 0) = 100 :=
by
  sorry

end a_628_th_term_is_negative_l793_793222


namespace measure_angle_KDA_l793_793939

-- Define the problem setting
variables (A B C D M K : Point)
variables (AD AB KD MKC : ℚ)
variables (AMK : ℝ)

-- Conditions
def rectangle_property (hAD_eq_2AB : AD = 2 * AB) (hM_mid_AD : segment M A = segment M D) 
(h_angle_AMK_80 : amk = 80) (h_K_bisector_MKC : is_bisector K D (angle MKC (MKC / 2))

-- Goal: measure of ∠KDA is 35°
theorem measure_angle_KDA (hAD_eq_2AB : AD = 2 * AB) (hM_mid_AD : segment M A = segment M D) (h_angle_AMK_80 : angle A M K = 80) (h_K_bisector_MKC : is_angle_bisector K D (angle M K C)) :
  angle K D A = 35 :=
by
  sorry

end measure_angle_KDA_l793_793939


namespace tan_of_angle_in_third_quadrant_l793_793824

open Real

theorem tan_of_angle_in_third_quadrant 
  (α : ℝ) 
  (h1 : sin (π + α) = 3/5) 
  (h2 : π < α ∧ α < 3 * π / 2) : 
  tan α = 3 / 4 :=
by
  sorry

end tan_of_angle_in_third_quadrant_l793_793824


namespace vector_projection_constant_l793_793628

theorem vector_projection_constant:
  ∀ (a : ℝ) (x : ℝ), ∃ (d : ℝ) (c : ℝ), c = -4 * d →
  let v := (a, 4 * a - 3) in
  let w := (c, d) in
  let p : ℝ × ℝ := (12 / 17, -3 / 17) in
  ((v.1 * w.1 + v.2 * w.2) / (w.1^2 + w.2^2)) * w = p := 
by
  intros a x d c hc v w p,
  sorry

end vector_projection_constant_l793_793628


namespace k3_to_fourth_equals_81_l793_793547

theorem k3_to_fourth_equals_81
  (h k : ℝ → ℝ)
  (h_cond : ∀ x, x ≥ 1 → h (k x) = x^3)
  (k_cond : ∀ x, x ≥ 1 → k (h x) = x^4)
  (k_81 : k 81 = 81) :
  k 3 ^ 4 = 81 :=
sorry

end k3_to_fourth_equals_81_l793_793547


namespace intersection_M_N_l793_793804

def M : Set ℝ := { x | 0.2^x < 25 }
def N : Set ℝ := { x | Real.log (x - 1) / Real.log 3 < 1 }

theorem intersection_M_N :
  (M ∩ N) = { x | 1 < x ∧ x < 4 } :=
sorry

end intersection_M_N_l793_793804


namespace num_two_digit_numbers_with_digit_less_than_35_l793_793383

theorem num_two_digit_numbers_with_digit_less_than_35 : 
  let total_two_digit := 90 in
  let unsuitable_numbers := 35 in
  let suitable_numbers := total_two_digit - unsuitable_numbers
  in suitable_numbers = 55 :=
by 
  let total_two_digit := 90
  let unsuitable_numbers := 35
  let suitable_numbers := total_two_digit - unsuitable_numbers
  show suitable_numbers = 55, from sorry

end num_two_digit_numbers_with_digit_less_than_35_l793_793383


namespace knight_reach_and_max_steps_l793_793869

def inGrid (m n : ℕ) (pt : ℕ × ℕ) : Prop :=
  let (x, y) := pt in 1 ≤ x ∧ x ≤ m ∧ 1 ≤ y ∧ y ≤ n

def knightMove (pt1 pt2 : ℕ × ℕ) : Prop :=
  let (x1, y1) := pt1 in
  let (x2, y2) := pt2 in
  (abs (x1 - x2) = 2 ∧ abs (y1 - y2) = 1) ∨
  (abs (x1 - x2) = 1 ∧ abs (y1 - y2) = 2)

def reachEveryPoint (m n : ℕ) (start : ℕ × ℕ) : Prop :=
  ∀ pt, inGrid m n pt → ∃ k : ℕ, ∃ path : List (ℕ × ℕ), 
    path.head = start ∧ path.last = some pt ∧ ∀ i, i < path.length - 1 → knightMove (path.nthLe i sorry) (path.nthLe (i + 1) sorry)

def maxStepsAndPoints (m n : ℕ) (start : ℕ × ℕ) : Prop :=
  ∃ k : ℕ, k = 6 ∧ 
  ∀ pt, inGrid m n pt ∧ isMaxStep m n pt start k ↔ pt = (7, 9) ∨ pt = (8, 8) ∨ pt = (9, 7) ∨ pt = (9, 9)

-- Additional helper predicate for checking max steps.
def isMaxStep (m n : ℕ) (pt start : ℕ × ℕ) (k : ℕ) : Prop := 
  ∃ path : List (ℕ × ℕ), path.head = start ∧ path.last = some pt ∧ path.length - 1 = k ∧ ∀ i, i < path.length - 1 → knightMove (path.nthLe i sorry) (path.nthLe (i + 1) sorry)

theorem knight_reach_and_max_steps :
  reachEveryPoint 9 9 (1, 1) ∧ 
  maxStepsAndPoints 9 9 (1, 1) :=
sorry

end knight_reach_and_max_steps_l793_793869


namespace price_per_bag_of_cement_l793_793009

def number_of_bags_of_cement : Nat := 500
def number_of_lorries_of_sand : Nat := 20
def tons_of_sand_per_lorry : Nat := 10
def price_of_sand_per_ton : Nat := 40
def total_amount_paid : Nat := 13000

/-- The price per bag of cement given the conditions -/
theorem price_per_bag_of_cement : (let p_c := (total_amount_paid - (number_of_lorries_of_sand * tons_of_sand_per_lorry * price_of_sand_per_ton)) / number_of_bags_of_cement in p_c = 10) :=
by
  sorry

end price_per_bag_of_cement_l793_793009


namespace count_four_digit_even_integers_l793_793817

-- Defining the set of even digits
def even_digits := {0, 2, 4, 6, 8}

-- The problem statement in Lean 4
theorem count_four_digit_even_integers :
  (count(p : ℕ // 1000 ≤ p ∧ p < 10000 ∧ ∀ d ∈ to_digits 10 (p : ℕ), d ∈ even_digits)) = 500 :=
sorry

end count_four_digit_even_integers_l793_793817


namespace exponent_equality_l793_793398

theorem exponent_equality (s m : ℕ) (h : (2^16) * (25^s) = 5 * (10^m)) : m = 16 :=
by sorry

end exponent_equality_l793_793398


namespace probability_of_perfect_square_sum_on_two_8_sided_dice_l793_793590

-- Define the dice and possible sums
def is_sum_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def pairs_rolled : finset (ℕ × ℕ) :=
  finset.univ.filter (λ p, p.1 + p.2 ∈ {4, 9, 16})

theorem probability_of_perfect_square_sum_on_two_8_sided_dice :
  (pairs_rolled.card : ℚ) / 64 = 3 / 16 :=
by
  sorry

end probability_of_perfect_square_sum_on_two_8_sided_dice_l793_793590


namespace coefficient_of_tr_in_expansion_l793_793288

theorem coefficient_of_tr_in_expansion
  (n r : ℕ) :
  (∑ i in range (r + 1), (complex.of_real (((1 : ℝ) - complex.of_real (t)) ^ n) * t ^ r) = (complex.of_real (((1 : ℝ) - t) ^ n) * t ^ r)) →  (t ≠ 0) → (t < 1):
  dit : ℝ := binom r-1 n-1
  coefficient_of_tr_in_expansion (n r : ℕ) :
  :=
by
  sorry

end coefficient_of_tr_in_expansion_l793_793288


namespace jam_cost_in_dollars_l793_793230

theorem jam_cost_in_dollars (N B J : ℕ) (hN : N > 1) (hB : B > 0) (hJ : J > 0) 
  (h : N * (3 * B + 7 * J) = 378) : 7 * J * N / 100 = 2.52 :=
by
  sorry

end jam_cost_in_dollars_l793_793230


namespace coin_flipping_probability_l793_793491

theorem coin_flipping_probability (m n : ℕ) (h1 : p = 3 / 34) (h2 : Nat.coprime m n) : m + n = 37 :=
sorry

end coin_flipping_probability_l793_793491


namespace sum_of_non_visible_faces_l793_793755

theorem sum_of_non_visible_faces
    (d1 d2 d3 d4 : Fin 6 → Nat)
    (visible_faces : List Nat)
    (hv : visible_faces = [1, 2, 3, 4, 4, 5, 5, 6]) :
    let total_sum := 4 * 21
    let visible_sum := List.sum visible_faces
    total_sum - visible_sum = 54 := by
  sorry

end sum_of_non_visible_faces_l793_793755


namespace quadratic_min_value_l793_793723

theorem quadratic_min_value :
  (∀ x : ℝ, let f := (2 * x^2 + 6 * x + 5) in 
    (f 1 = 13)) ∧
    (∀ x : ℝ, ∃ y : ℝ, ∀ z : ℝ, f(z) ≥ y ∧ (f(-3/2) = 0.5)) :=
sorry

end quadratic_min_value_l793_793723


namespace remainder_is_one_l793_793163

theorem remainder_is_one (N : ℤ) (R : ℤ)
  (h1 : N % 100 = R)
  (h2 : N % R = 1) :
  R = 1 :=
by
  sorry

end remainder_is_one_l793_793163


namespace two_digit_numbers_less_than_35_l793_793353

theorem two_digit_numbers_less_than_35 : 
  let count : ℕ := (99 - 10 + 1) - (7 * 5)
  count = 55 :=
by
  -- definition of total number of two-digit numbers
  let total_two_digit_numbers : ℕ := 99 - 10 + 1
  -- definition of the number of unsuitable two-digit numbers
  let unsuitable_numbers : ℕ := 7 * 5
  -- definition of the count of suitable two-digit numbers
  let count : ℕ := total_two_digit_numbers - unsuitable_numbers
  -- verifying the final count
  exact rfl

end two_digit_numbers_less_than_35_l793_793353


namespace angle_C_in_triangle_l793_793851

theorem angle_C_in_triangle (AC BC : ℝ) (angleB : ℝ) (hAC : AC = Real.sqrt 6) (hBC : BC = 2) (hAngleB : angleB = 60) :
  let A := 180 - angleB - 45 in A = 75 := sorry

end angle_C_in_triangle_l793_793851


namespace sum_x_coords_of_X_l793_793587

theorem sum_x_coords_of_X 
  (X Y Z W V : ℝ × ℝ)
  (area_XYZ area_XWV : ℝ)
  (hy : Y = (0,0))
  (hz : Z = (150,0))
  (hw : W = (500, 300))
  (hv : V = (510,290))
  (h_area_XYZ : area_XYZ = 1200)
  (h_area_XWV : area_XWV = 3600)
  : let x_coords_sum := (4 * 800) in x_coords_sum = 3200 := by
    -- The proof is omitted.
    sorry

end sum_x_coords_of_X_l793_793587


namespace vector_magnitude_l793_793844

variable {V : Type*} [InnerProductSpace ℝ V]

theorem vector_magnitude
  (a b : V)
  (h1 : ‖a‖ = 3)
  (h2 : ‖a - b‖ = 5)
  (h3 : inner a b = 1) :
  ‖b‖ = 3 * Real.sqrt 2 :=
by
  sorry

end vector_magnitude_l793_793844


namespace num_two_digit_numbers_with_digit_less_than_35_l793_793381

theorem num_two_digit_numbers_with_digit_less_than_35 : 
  let total_two_digit := 90 in
  let unsuitable_numbers := 35 in
  let suitable_numbers := total_two_digit - unsuitable_numbers
  in suitable_numbers = 55 :=
by 
  let total_two_digit := 90
  let unsuitable_numbers := 35
  let suitable_numbers := total_two_digit - unsuitable_numbers
  show suitable_numbers = 55, from sorry

end num_two_digit_numbers_with_digit_less_than_35_l793_793381


namespace largest_prime_difference_130_l793_793572

theorem largest_prime_difference_130 :
  ∃ p q : ℕ, p ≠ q ∧ nat.prime p ∧ nat.prime q ∧ p + q = 130 ∧ (q - p = 124) :=
sorry

end largest_prime_difference_130_l793_793572


namespace max_bishops_arrangement_is_perfect_square_l793_793010

theorem max_bishops_arrangement_is_perfect_square :
  ∃ N k : ℕ, N = k^2 ∧ let max_bishops := λ (board : Fin 8 × Fin 8 → Prop), 
    ∀ i j : Fin 8, board i j → (∃ d : Int, i - j = d ∨ i + j = d) →
    (board i j → ∀ k l : Fin 8, (i ≠ k ∨ j ≠ l) → ¬ board k l) in
    (let board := fun xy => xy.1 < 8 ∧ xy.2 < 8 in max_bishops board) = N := sorry

end max_bishops_arrangement_is_perfect_square_l793_793010


namespace line_slope_l793_793225

theorem line_slope (x y : ℝ) : 3 * y - (1 / 2) * x = 9 → ∃ m, m = 1 / 6 :=
by
  sorry

end line_slope_l793_793225


namespace chickens_on_farm_l793_793068

theorem chickens_on_farm (x y : ℕ) (h1 : 2 * x + 4 * y = 38) (h2 : x + y = 12) : x = 5 :=
by
  have h3 : y = 12 - x :=
    by sorry
  have h4 : 2 * x + 4 * (12 - x) = 38 :=
    by sorry
  have h5 : 2 * x + 48 - 4 * x = 38 :=
    by sorry
  have h6 : -2 * x = -10 :=
    by sorry
  have h7 : x = 5 :=
    by sorry
  exact h7

end chickens_on_farm_l793_793068


namespace x_coordinate_of_tangent_point_l793_793310

-- Define the function f
def f (x : ℝ) : ℝ := x * Real.log x

-- Define the condition that the derivative f' equals 2
def slope_condition (x : ℝ) : Prop := Real.derivative f x = 2

-- State the theorem to find the x-coordinate where the slope condition holds
theorem x_coordinate_of_tangent_point : ∃ x : ℝ, slope_condition x ∧ x = Real.exp 1 :=
by
  sorry

end x_coordinate_of_tangent_point_l793_793310


namespace probability_of_less_than_three_on_eight_sided_die_l793_793629

theorem probability_of_less_than_three_on_eight_sided_die : 
  let total_outcomes := 8 in
  let favorable_outcomes := 2 in
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 4 := 
by
  sorry

end probability_of_less_than_three_on_eight_sided_die_l793_793629


namespace find_quotient_l793_793523

-- Define the given conditions
def dividend : ℤ := 144
def divisor : ℤ := 11
def remainder : ℤ := 1

-- Define the quotient logically derived from the given conditions
def quotient : ℤ := dividend / divisor

-- The theorem we need to prove
theorem find_quotient : quotient = 13 := by
  sorry

end find_quotient_l793_793523


namespace negation_of_quadratic_prop_l793_793016

theorem negation_of_quadratic_prop :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 0) ↔ ∃ x_0 : ℝ, x_0^2 + 1 < 0 :=
by
  sorry

end negation_of_quadratic_prop_l793_793016
