import Mathlib
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.Group.Defs
import Mathlib.Analysis.Calculus.Sqrt
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Composition
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Finite.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Perm
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.NumberTheory.ModularArithmetic.Basic
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.SolveByElim

namespace triangle_inequality_l801_801885

theorem triangle_inequality (a b c : ℝ) (α : ℝ) 
  (h_triangle_sides : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_cosine_rule : a^2 = b^2 + c^2 - 2 * b * c * Real.cos α) :
  (2 * b * c * Real.cos α) / (b + c) < (b + c - a) ∧ (b + c - a) < (2 * b * c) / a := 
sorry

end triangle_inequality_l801_801885


namespace problem_l801_801772

noncomputable def f (ω x : ℝ) : ℝ := 2 * sin(ω * x) * cos(ω * x) + 2 * sqrt 3 * (sin(ω * x))^2 - sqrt 3

theorem problem (
  ω > 0 : Prop
  (h1 : f(ω) has minimum positive period π)
  ω_eq : ω = 1
  decr_interval : ∀ k : ℤ, (k * π + (5 * π)/12 ≤ x ∧ x ≤ k * π + (11 * π)/12)
  (g : ℝ → ℝ) := g(x) = f(1, x + (π / 6)) + 1
  has_zeros : ∀ k : ℤ, k * π + (7 * π)/12 ∨ k * π + (11 * π)/12
  at_least_10_zeros (bₘin : ℝ), bₘin = (4 * π + (11 * π)/12)) 
  :
  bₘin = (59 * π) / 12 := sorry

end problem_l801_801772


namespace blowfish_stayed_own_tank_l801_801691

def number_clownfish : ℕ := 50
def number_blowfish : ℕ := 50
def number_clownfish_display_initial : ℕ := 24
def number_clownfish_display_final : ℕ := 16

theorem blowfish_stayed_own_tank : 
    (number_clownfish + number_blowfish = 100) ∧ 
    (number_clownfish = number_blowfish) ∧ 
    (number_clownfish_display_final = 2 / 3 * number_clownfish_display_initial) →
    ∀ (blowfish : ℕ), 
    blowfish = number_blowfish - number_clownfish_display_initial → 
    blowfish = 26 :=
sorry

end blowfish_stayed_own_tank_l801_801691


namespace find_remainder_l801_801872

theorem find_remainder : ∃ r : ℝ, r = 14 ∧ 13698 = (153.75280898876406 * 89) + r := 
by
  sorry

end find_remainder_l801_801872


namespace certain_number_calculation_l801_801445

theorem certain_number_calculation (x : ℝ) (h : (15 * x) / 100 = 0.04863) : x = 0.3242 :=
by
  sorry

end certain_number_calculation_l801_801445


namespace circumcircle_centers_distance_eq_AC_l801_801112

variable {A B C H C_2 A_1 A_2 : Point}

-- Assume the triangle ABC is acute
def is_acute_triangle (A B C : Point) : Prop := sorry

-- Define orthocenter
def is_orthocenter (H A B C : Point) : Prop := sorry

-- Distance between points
def distance (p q : Point) : Real := sorry

-- Assuming the points C_2, A_1, and A_2 are defined and have meaning in context
-- for instance, through perpendiculars and constructions related to the orthocenter and circumcircles

theorem circumcircle_centers_distance_eq_AC :
  is_acute_triangle A B C →
  is_orthocenter H A B C →
  distance (center_of_circumcircle C_2 H A_1) (center_of_circumcircle A_1 H A_2) = distance A C := sorry

end circumcircle_centers_distance_eq_AC_l801_801112


namespace multiply_by_15_is_225_l801_801654

-- Define the condition
def number : ℕ := 15

-- State the theorem with the conditions and the expected result
theorem multiply_by_15_is_225 : 15 * number = 225 := by
  -- Insert the proof here
  sorry

end multiply_by_15_is_225_l801_801654


namespace min_reciprocal_sum_of_sequence_l801_801381

theorem min_reciprocal_sum_of_sequence 
  (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℝ) 
  (h1 : ∀ n : ℕ, 2 ≤ n → a (n+1) + a n = (n+1) * real.cos (n * real.pi / 2))
  (h2 : S 2017 + m = 1012)
  (h3 : a 1 * m > 0) :
  (∃ (a1 m : ℝ), a1 = a 1 ∧ m > 0 ∧ m = m ∧ 
  min_value : ∃ (min_val : ℝ), 
    min_val = min (1 / a1 + 1 / m) ∧ 
    min_val = 1) :=
sorry

end min_reciprocal_sum_of_sequence_l801_801381


namespace exists_isosceles_triangle_in_20gon_l801_801458

theorem exists_isosceles_triangle_in_20gon (marked_vertices : Finset (Fin 20)) (h_marked : marked_vertices.card = 9) : 
  ∃ (a b c : Fin 20), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ marked_vertices ∧ b ∈ marked_vertices ∧ c ∈ marked_vertices ∧ 
  ((a < b ∧ b < c) ∨ (b < c ∧ c < a) ∨ (c < a ∧ a < b)) ∧ is_isosceles a b c := 
sorry

end exists_isosceles_triangle_in_20gon_l801_801458


namespace seq_10001_satisfies_C_l801_801656

-- Definitions from conditions
def is_zero_one_sequence (a : ℕ → ℕ) : Prop := ∀ i, a i ∈ {0, 1}

def has_period (a : ℕ → ℕ) (m : ℕ) : Prop := ∀ i, a (i + m) = a i

def C (a : ℕ → ℕ) (m k : ℕ) : ℚ :=
  (1 / m : ℚ) * (∑ i in Finset.range m, a i * a (i + k))

noncomputable def satisfies_C (a : ℕ → ℕ) (m : ℕ) : Prop :=
  ∀ k, k ∈ Finset.range (m - 1) → C a m k ≤ (1 / m : ℚ)

-- Sequence 10001 repeating
def seq_10001 : ℕ → ℕ
| 0 := 1
| 1 := 0
| 2 := 0
| 3 := 0
| 4 := 1
| (n + 5) := seq_10001 n

-- Theorem statement
theorem seq_10001_satisfies_C :
  is_zero_one_sequence seq_10001 ∧ has_period seq_10001 5 → satisfies_C seq_10001 5 := 
by sorry

end seq_10001_satisfies_C_l801_801656


namespace combined_experience_l801_801831

noncomputable def james_experience : ℕ := 20
noncomputable def john_experience_8_years_ago : ℕ := 2 * (james_experience - 8)
noncomputable def john_current_experience : ℕ := john_experience_8_years_ago + 8
noncomputable def mike_experience : ℕ := john_current_experience - 16

theorem combined_experience :
  james_experience + john_current_experience + mike_experience = 68 :=
by
  sorry

end combined_experience_l801_801831


namespace new_fish_received_l801_801485

def initial_fish := 14
def added_fish := 2
def eaten_fish := 6
def final_fish := 11

def current_fish := initial_fish + added_fish - eaten_fish
def returned_fish := 2
def exchanged_fish := final_fish - current_fish

theorem new_fish_received : exchanged_fish = 1 := by
  sorry

end new_fish_received_l801_801485


namespace cube_root_of_difference_l801_801097

variable (a m : ℝ)
axiom (h1 : a > 0)
axiom (h2 : sqrt(a) = m + 7 ∨ sqrt(a) = 2m - 1)

theorem cube_root_of_difference : real.cbrt (a - m) = 3 :=
by
  sorry

end cube_root_of_difference_l801_801097


namespace mean_of_xyz_l801_801198

variable (x y z : ℝ)

def arithmetic_mean (nums : List ℝ) : ℝ :=
  nums.sum / nums.length

theorem mean_of_xyz
  (h1 : arithmetic_mean [a1, a2, ..., a12] = 72)
  (h2 : arithmetic_mean (x :: y :: z :: [a1, a2, ..., a12]) = 80)
  : arithmetic_mean [x, y, z] = 112 :=
by
  sorry

end mean_of_xyz_l801_801198


namespace tan_sum_pi_over_12_l801_801531

theorem tan_sum_pi_over_12 : 
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12)) = 4 := 
sorry

end tan_sum_pi_over_12_l801_801531


namespace sum_of_roots_f_3x_eq_c_l801_801158

theorem sum_of_roots_f_3x_eq_c :
  (∀ x : ℝ, (f : ℝ → ℝ) = λ x, x^2 + x + 1 → ∃ d : ℝ, d = -1/9) :=
by
  sorry

end sum_of_roots_f_3x_eq_c_l801_801158


namespace tea_drinking_proof_l801_801995

theorem tea_drinking_proof :
  ∃ (k : ℝ), 
    (∃ (c_sunday t_sunday c_wednesday t_wednesday : ℝ),
      c_sunday = 8.5 ∧ 
      t_sunday = 4 ∧ 
      c_wednesday = 5 ∧ 
      t_sunday * c_sunday = k ∧ 
      t_wednesday * c_wednesday = k ∧ 
      t_wednesday = 6.8) :=
sorry

end tea_drinking_proof_l801_801995


namespace walking_rate_misses_train_l801_801799

theorem walking_rate_misses_train
  (v : ℝ) -- Walking rate at which the man misses the train
  (d : ℝ) -- Distance to the station
  (t1 : ℝ) -- Time to reach station at 5 kmph
  (t2 : ℝ) -- Time train arrives before he walks at 'v'
  (missed_time : ℝ) -- Time he misses the train by
  (v_correct : v = 4) -- The correct rate we're proving for
  (distance : d = 4) -- Distance is 4 km
  (speed_5_kmph : 5) -- Walking at 5 kmph
  (time_early : t1 = d / speed_5_kmph) -- Time walked at 5 kmph
  (arrives_early : t2 = t1 + (6 / 60)) -- Train arrival is 6 minutes later
  (misses_train : 1 = d / v) -- Time walked at 'v' accounts for 6 minutes missed
  : v = 4 := 
sorry

end walking_rate_misses_train_l801_801799


namespace required_circle_properties_l801_801355

-- Define the two given circles' equations
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x - 4 = 0

def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*y - 28 = 0

-- Define the line on which the center of the required circle lies
def line (x y : ℝ) : Prop :=
  x - y - 4 = 0

-- The equation of the required circle
def required_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - x + 7*y - 32 = 0

-- Prove that the required circle satisfies the conditions
theorem required_circle_properties (x y : ℝ) (hx : required_circle x y) :
  (∃ x y, circle1 x y ∧ circle2 x y ∧ required_circle x y) ∧
  (∃ x y, required_circle x y ∧ line x y) :=
by
  sorry

end required_circle_properties_l801_801355


namespace taequan_dice_game_prob_l801_801643

theorem taequan_dice_game_prob :
  let possible_combinations := { (2, 6), (3, 5), (4, 4), (5, 3), (6, 2) },
      total_outcomes := 6 * 6,
      winning_combinations := possible_combinations.card in
  winning_combinations / total_outcomes = 5 / 36 :=
by sorry

end taequan_dice_game_prob_l801_801643


namespace apples_handout_l801_801576

theorem apples_handout {total_apples pies_needed pies_count handed_out : ℕ}
  (h1 : total_apples = 51)
  (h2 : pies_needed = 5)
  (h3 : pies_count = 2)
  (han : handed_out = total_apples - (pies_needed * pies_count)) :
  handed_out = 41 :=
by {
  sorry
}

end apples_handout_l801_801576


namespace find_k_l801_801779

noncomputable def g (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem find_k (k : ℝ) (h_pos : 0 < k) (h_exists : ∃ x₀ : ℝ, 1 ≤ x₀ ∧ g x₀ ≤ k * (-x₀^2 + 3 * x₀)) : 
  k > (1 / 2) * (Real.exp 1 + 1 / Real.exp 1) :=
sorry

end find_k_l801_801779


namespace parallelogram_sides_product_l801_801816

theorem parallelogram_sides_product (w z : ℝ) (EF GH HE FG : ℝ)
  (hEF : EF = 42) (hGH : GH = 2 * w + 6) (hHE : HE = 32) (hFG : FG = 4 * z^3) :
  w * z = 36 := by
  have h1 : 42 = 2 * w + 6 := by rw [← hGH, hEF]
  have hw : w = 18 := by
    linarith

  have h2 : 4 * z^3 = 32 := by rw [← hFG, hHE]
  have hz : z = 2 := by
    field_simp
    norm_num at h2
    norm_num

  calc
    w * z = 18 * 2 : by rw [hw, hz]
    ... = 36 : by norm_num

end parallelogram_sides_product_l801_801816


namespace modified_region_perimeter_is_56_feet_l801_801822

-- Definitions of conditions
def all_angles_are_right : Prop :=
  ∀ (α : ℝ), α = 90

def congruent_sides (n : ℕ) (l : ℝ) : Prop :=
  ∀ (i : ℕ), i < n → length_of_side i = l

def region_area (A : ℝ) : Prop :=
  A = 140

def bottom_side_length (L : ℝ) : Prop :=
  L = 18

-- Main goal: Prove the perimeter of the modified region is 56 feet
theorem modified_region_perimeter_is_56_feet (A : ℝ) (L : ℝ) (n : ℕ) (l : ℝ)
  (h1 : all_angles_are_right)
  (h2 : congruent_sides n l)
  (h3 : region_area A)
  (h4 : bottom_side_length L) : 
  perimeter_of_modified_region A L n l = 56 :=
by
  sorry

end modified_region_perimeter_is_56_feet_l801_801822


namespace total_resistance_between_A_and_B_l801_801465

-- Define the conditions
def R1 : ℝ := 1 -- in Ohms
def R2 : ℝ := 2 -- in Ohms
def R3 : ℝ := 3 -- in Ohms
def R4 : ℝ := 4 -- in Ohms

-- Problem statement
theorem total_resistance_between_A_and_B :
  let R12 := (R1 * R2) / (R1 + R2) in
  let R34 := (R3 * R4) / (R3 + R4) in
  let R := R12 + R34 in
  R = 50 / 21 :=
by {
  sorry -- Proof to be filled in
}

end total_resistance_between_A_and_B_l801_801465


namespace sum_equals_value_l801_801848

def floor (x : ℝ) : ℤ := Int.ofNat (Real.floor x)

noncomputable def S : ℤ :=
  let main_contrib := ∑ k in range 1 45, ∑ n in range 1 (2*k + 1), floor (n / k)
  let rem_terms := ∑ n in range 1 37, floor (n / 45)
  main_contrib + rem_terms

theorem sum_equals_value :
  S = 1078 :=
  by
  sorry

end sum_equals_value_l801_801848


namespace one_incorrect_proposition_l801_801052

theorem one_incorrect_proposition :
  (¬ (p ∧ q) implies (¬ p ∧ ¬ q)) = false ∧
  (¬ (a > b ∧ 2^a ≤ 2^b - 1)) = true ∧
  (¬ (∀ x : ℝ, x^2 + 1 ≥ 0)) = false ∧
  (∀ A B : ℝ, A > B ↔ sin A > sin B) = true ↔
  1 = 1 :=
by sorry

end one_incorrect_proposition_l801_801052


namespace abs_diff_one_l801_801383

theorem abs_diff_one {a b c d : ℤ} (h : a + b + c + d = a * b + b * c + c * d + d * a + 1) :
  ∃ x y ∈ {a, b, c, d}, (|x - y| = 1) :=
sorry

end abs_diff_one_l801_801383


namespace part1_part2_l801_801060

noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (x : ℝ) (b : ℝ) := 0.5 * x^2 - b * x
noncomputable def h (x : ℝ) (b : ℝ) := f x + g x b

theorem part1 (b : ℝ) :
  (∃ (tangent_point : ℝ),
    tangent_point = 1 ∧
    deriv f tangent_point = 1 ∧
    f tangent_point = 0 ∧
    ∃ (y_tangent : ℝ → ℝ), (∀ (x : ℝ), y_tangent x = x - 1) ∧
    ∃ (tangent_for_g : ℝ), (∀ (x : ℝ), y_tangent x = g x b)
  ) → false :=
sorry 

theorem part2 (b : ℝ) :
  ¬ (∀ (x : ℝ) (hx : 0 < x), deriv (h x) b = 0 → deriv (h x) b < 0) →
  2 < b :=
sorry

end part1_part2_l801_801060


namespace possible_values_of_b2_l801_801302

theorem possible_values_of_b2 :
  let b : ℕ → ℕ := λ n, if n = 0 then 798 else if n = 1 then b2 else |b (n-1) - b (n-2)|
  in ∃ b2, b2 < 798 ∧ b2 % 2 = 0 ∧ Nat.gcd 798 b2 = 2 ∧
    (∃ S : Finset ℕ, S = { b2 | b2 < 798 ∧ b2 % 2 = 0 ∧ Nat.gcd 798 b2 = 2} ∧ S.card = 256) :=
begin
  sorry
end

end possible_values_of_b2_l801_801302


namespace movie_box_office_revenue_l801_801514

variable (x : ℝ)

theorem movie_box_office_revenue (h : 300 + 300 * (1 + x) + 300 * (1 + x)^2 = 1000) :
  3 + 3 * (1 + x) + 3 * (1 + x)^2 = 10 :=
by
  sorry

end movie_box_office_revenue_l801_801514


namespace count_pairs_l801_801712

theorem count_pairs : 
  ∃ (s : Finset (ℕ × ℕ)), 
    (∀ p ∈ s, let x := p.1, y := p.2 in x^2 + y^2 = x^3 ∧ (x - 1) % 3 = 0) ∧
    (∀ xy, let x := xy.1, y := xy.2 in x^2 + y^2 = x^3 ∧ (x - 1) % 3 = 0 → xy ∈ s) ∧
    s.card = 3 :=
by
  sorry

end count_pairs_l801_801712


namespace convex_quad_with_symmetry_inscribed_or_circumscribed_l801_801174

-- Definitions congruent with the given problem:
structure ConvexQuadrilateral (A B C D : Type) :=
(convex_quad : ∀ A B C D : Finset ℝ, (∃ E F G H : Point, convex_quad E F G H))

structure AxisSymmetry (A B C D : Type) :=
(axis_of_symmetry : ∀ A B C D : Finset ℝ, (∃ E F G H : Point, axis_of_symmetry E F G H))

-- Proposition coinciding with the problem statement:
theorem convex_quad_with_symmetry_inscribed_or_circumscribed (A B C D E F : Type) 
  (h1 : ConvexQuadrilateral A B C D) (h2 : AxisSymmetry A B C D) :
  (∃ Q : Circle, inscribed Q A B C D) ∨ (∃ P : Circle, circumscribed P A B C D) := 
sorry

end convex_quad_with_symmetry_inscribed_or_circumscribed_l801_801174


namespace tan_pi_over_12_plus_tan_5pi_over_12_eq_4_l801_801527

-- Definitions for given conditions
def cos_pi_over_12 : ℝ := (Real.sqrt 6 + Real.sqrt 2) / 4
def cos_5pi_over_12 : ℝ := (Real.sqrt 6 - Real.sqrt 2) / 4

-- The theorem to be proved
theorem tan_pi_over_12_plus_tan_5pi_over_12_eq_4 : 
  Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 :=
by sorry

end tan_pi_over_12_plus_tan_5pi_over_12_eq_4_l801_801527


namespace number_of_ways_to_assign_shifts_l801_801944

def workers : List String := ["A", "B", "C"]

theorem number_of_ways_to_assign_shifts :
  let shifts := ["day", "night"]
  (workers.length * (workers.length - 1)) = 6 := by
  sorry

end number_of_ways_to_assign_shifts_l801_801944


namespace right_triangle_sides_l801_801597

theorem right_triangle_sides (m : ℝ)
  (A B C : ℝ^3)
  (hABC: right_triangle A B C)
  (hCO_median: median_to_hypotenuse C O = m)
  (hangle_ratio: angle_ratio_1_2 (angle B C O) (angle A C O)) :
  triangle_sides A B C = (m, m * √3, 2 * m) :=
sorry

end right_triangle_sides_l801_801597


namespace geom_prog_find_side_c_l801_801806

noncomputable section
open Real

variables (a b c A B C : ℝ)
variables (triangle_ABC : Type)

-- Conditions
def condition1 := sin (A + C) = 2 * sin A * cos (A + B)
def condition2 := C = 3 * π / 4
def condition3 := (2 * (1 / 2) * a * b * sin C) = 2
def condition4 := b = sqrt 2 * a

-- Question Ⅰ
theorem geom_prog : 
  condition1 →
  condition2 →
  (a ≠ 0) →
  b^2 = 2 * a^2 :=
sorry

-- Question Ⅱ
theorem find_side_c :
  condition1 →
  condition2 →
  condition3 →
  condition4 →
  c = 2 * sqrt 5 :=
sorry

end geom_prog_find_side_c_l801_801806


namespace divisibility_by_seven_l801_801905

theorem divisibility_by_seven (n : ℤ) (b : ℤ) (a : ℤ) (h : n = 10 * a + b) 
  (hb : 0 ≤ b) (hb9 : b ≤ 9) (ha : 0 ≤ a) (d : ℤ) (hd : d = a - 2 * b) :
  (2 * n + d) % 7 = 0 ↔ n % 7 = 0 := 
by
  sorry

end divisibility_by_seven_l801_801905


namespace moles_ethane_and_hexachloroethane_l801_801072

-- Define the conditions
def balanced_eq (a b c d : ℕ) : Prop :=
  a * 6 = b ∧ d * 6 = c

-- The main theorem statement
theorem moles_ethane_and_hexachloroethane (moles_Cl2 : ℕ) :
  moles_Cl2 = 18 → balanced_eq 1 1 18 3 :=
by
  sorry

end moles_ethane_and_hexachloroethane_l801_801072


namespace symmetric_line_equation_l801_801356

-- Definitions of the given problem's conditions
def line1 (x y : ℝ) : Prop := 2 * x - y + 3 = 0
def symmetry_line (x y : ℝ) : Prop := x - y + 2 = 0
def symmetric_line (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- The theorem statement proving the required equation
theorem symmetric_line_equation :
  ∀ x y : ℝ, line1 x y → (∃ ! y', symmetric_line x y' ∧ symmetry_line x y') :=
by {
  sorry -- the proof would go here
}

end symmetric_line_equation_l801_801356


namespace rahul_deepak_present_ages_l801_801236

theorem rahul_deepak_present_ages (R D : ℕ) 
  (h1 : R / D = 4 / 3)
  (h2 : R + 6 = 26)
  (h3 : D + 6 = 1/2 * (R + (R + 6)))
  (h4 : (R + 11) + (D + 11) = 59) 
  : R = 20 ∧ D = 17 :=
sorry

end rahul_deepak_present_ages_l801_801236


namespace seven_lines_with_orthogonal_third_line_l801_801827

theorem seven_lines_with_orthogonal_third_line :
  ∃ (lines : Fin 7 → ℝ × ℝ × ℝ), 
    (∀ i, ∥lines i∥ = 1) ∧
    (∀ i j, i ≠ j → ∃ k, k ≠ i ∧ k ≠ j ∧ inner (lines i) (lines j) = 0 ∧ inner (lines i) (lines k) = 0 ∧ inner (lines j) (lines k) = 0) :=
by
  sorry

end seven_lines_with_orthogonal_third_line_l801_801827


namespace count_multiples_of_30_l801_801837

theorem count_multiples_of_30 (h1: ∃ n₁ : ℕ, n₁ * n₁ = 900)
                             (h2: ∃ n₂ : ℕ, n₂ * n₂ * n₂ = 27000):
  { k : ℕ // 900 ≤ 30 * k ∧ 30 * k ≤ 27000 }.toFinmap.card = 871 :=
sorry

end count_multiples_of_30_l801_801837


namespace hexagon_parabola_intercepts_l801_801819

/-- Given a regular hexagon ABCDEF with side length 2,
points E and F on the x-axis,
and points A, B, C, D lying on a parabola,
prove the distance between the x-intercepts of the parabola is 2√7. -/
theorem hexagon_parabola_intercepts 
  (A B C D E F : ℝ × ℝ)
  (h_hex : regular_hexagon A B C D E F)
  (h_EF_x_axis : E.2 = 0 ∧ F.2 = 0)
  (h_parabola : ∃ a b c : ℝ, ∀ p ∈ {A, B, C, D}, p.2 = a * p.1^2 + b * p.1 + c) :
  distance (parabola_x_intercepts a b c) = 2 * real.sqrt 7 :=
by
  sorry

end hexagon_parabola_intercepts_l801_801819


namespace power_of_point_invariant_l801_801168

variable {P : Type} [PlanePoint P] (S : Circle P) (P0 : P)
variable (line1 line2 : Line P)
variable {A B A1 B1 : P}

-- State that the points A, B are intersections of S with line1 through P0
def intersects_S_line1 (line1 : Line P) {S : Circle P} {P0 A B : P} :=
  line1.through P0 ∧ S.contains A ∧ S.contains B ∧ line1.contains A ∧ line1.contains B

-- State that the points A1, B1 are intersections of S with line2 through P0
def intersects_S_line2 (line2 : Line P) {S : Circle P} {P0 A1 B1 : P} :=
  line2.through P0 ∧ S.contains A1 ∧ S.contains B1 ∧ line2.contains A1 ∧ line2.contains B1

-- We need to prove the product PA.PB is constant for different lines through P0
theorem power_of_point_invariant
  (h1 : intersects_S_line1 S line1 P0 A B)
  (h2 : intersects_S_line2 S line2 P0 A1 B1) :
  distance P0 A * distance P0 B = distance P0 A1 * distance P0 B1 :=
  sorry

end power_of_point_invariant_l801_801168


namespace stratified_sampling_l801_801286

theorem stratified_sampling (total_samples : ℕ) (prod1 prod2 prod3 : ℕ) :
  total_samples = 46 → prod1 = 1200 → prod2 = 6000 → prod3 = 2000 → 
  let total_production := prod1 + prod2 + prod3 in
  let samples1 := (prod1 * total_samples) / total_production in
  let samples2 := (prod2 * total_samples) / total_production in
  let samples3 := (prod3 * total_samples) / total_production in
  samples1 = 6 ∧ samples2 = 30 ∧ samples3 = 10 :=
by
  intros h1 h2 h3 h4
  let total_production := prod1 + prod2 + prod3
  let samples1 := (prod1 * total_samples) / total_production
  let samples2 := (prod2 * total_samples) / total_production
  let samples3 := (prod3 * total_samples) / total_production
  have h5 : total_production = 1200 + 6000 + 2000, from sorry,
  have h6 : samples1 = (1200 * 46) / (1200 + 6000 + 2000), from sorry,
  have h7 : samples2 = (6000 * 46) / (1200 + 6000 + 2000), from sorry,
  have h8 : samples3 = (2000 * 46) / (1200 + 6000 + 2000), from sorry,
  have h9 : (1200 * 46) / (1200 + 6000 + 2000) = 6, from sorry,
  have h10 : (6000 * 46) / (1200 + 6000 + 2000) = 30, from sorry,
  have h11 : (2000 * 46) / (1200 + 6000 + 2000) = 10, from sorry,
  exact ⟨h9, h10, h11⟩

end stratified_sampling_l801_801286


namespace eve_spending_l801_801725

-- Definitions of the conditions
def cost_mitt : ℝ := 14.00
def cost_apron : ℝ := 16.00
def cost_utensils : ℝ := 10.00
def cost_knife : ℝ := 2 * cost_utensils -- Twice the amount of the utensils
def discount_rate : ℝ := 0.25
def num_nieces : ℝ := 3

-- Total cost before the discount for one kit
def total_cost_one_kit : ℝ :=
  cost_mitt + cost_apron + cost_utensils + cost_knife

-- Discount for one kit
def discount_one_kit : ℝ := 
  total_cost_one_kit * discount_rate

-- Discounted price for one kit
def discounted_cost_one_kit : ℝ :=
  total_cost_one_kit - discount_one_kit

-- Total cost for all kits
def total_cost_all_kits : ℝ :=
  num_nieces * discounted_cost_one_kit

-- The theorem statement
theorem eve_spending : total_cost_all_kits = 135.00 :=
by sorry

end eve_spending_l801_801725


namespace equal_ratios_l801_801875

variables {A B C A_1 B_1 C_1 A_2 B_2 C_2 : Type*}
          [add_comm_group A] [add_comm_group B] [add_comm_group C]
          [module ℝ A] [module ℝ B] [module ℝ C]

/- The conditions -/
def points_on_sides (A B C A_1 B_1 C_1 : Type*) :=
    True -- Placeholder: Define the exact geometric relationship regarding points lying on sides

def segments_intersect_at_points (BB_1 CC_1 AA_1 A_2 B_2 C_2 : Type*) :=
    True -- Placeholder: Define the exact geometric relationship regarding intersections

def vector_sum_condition (AA_2 BB_2 CC_2 : Type*) [has_vadd AA_2] [has_vadd BB_2] [has_vadd CC_2] :=
    (vector_sum AA_2 BB_2 CC_2 = 0 : Type*) -- Placeholder: Assume given vector sum condition

/- The theorem to prove -/
theorem equal_ratios (h1 : points_on_sides A B C A_1 B_1 C_1)
                    (h2 : segments_intersect_at_points BB_1 CC_1 AA_1 A_2 B_2 C_2)
                    (h3 : vector_sum_condition AA_2 BB_2 CC_2) :
  AB_1 B_1C = CA_1 A_1B ∧ CA_1 A_1B = BC_1 C_1A :=
sorry

end equal_ratios_l801_801875


namespace find_n_l801_801344

theorem find_n (n : ℕ) (h₁ : 2^6 * 3^3 * n = factorial 10) : n = 2100 :=
sorry

end find_n_l801_801344


namespace tangent_line_correct_l801_801919

noncomputable def tangent_line_equation : ℝ → ℝ := 
  fun x => (exp(1) / 4) * x + (exp(1) / 4)

theorem tangent_line_correct :
  (∀ x, (x ≠ 1 → tangent_line_equation x = (exp(1) / 4) * x + (exp(1) / 4)) ∧
       (∀ y, (y = 1 → tangent_line_equation 1 = exp(1) / 2)) ∧
       (∀ f, (f x = exp(x) / (x + 1) ∧ x = 1 → deriv f 1 = exp(1) / 4))) :=
by
  sorry

end tangent_line_correct_l801_801919


namespace circle_area_from_circumference_l801_801206

theorem circle_area_from_circumference
  (c : ℝ)    -- the circumference
  (hc : c = 36)    -- condition: circumference is 36 cm
  : 
  ∃ A : ℝ,   -- there exists an area A
    A = 324 / π :=   -- conclusion: area is 324/π
by
  sorry   -- proof goes here

end circle_area_from_circumference_l801_801206


namespace hyperbola_eccentricity_l801_801492

theorem hyperbola_eccentricity
  (a b : ℝ)
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0)
  (H : ∀ P : ℝ×ℝ, (|dist P (a, 0) - dist P (-a, 0)|)^2 = b^2 - 3 * a * b) :
  let e := (Real.sqrt (a^2 + b^2)) / a in
  e = Real.sqrt 17 :=
by
  sorry

end hyperbola_eccentricity_l801_801492


namespace intersection_complement_l801_801765

-- Defining the sets A and B
def setA : Set ℝ := { x | -3 < x ∧ x < 3 }
def setB : Set ℝ := { x | x < -2 }
def complementB : Set ℝ := { x | x ≥ -2 }

-- The theorem to be proved
theorem intersection_complement :
  setA ∩ complementB = { x | -2 ≤ x ∧ x < 3 } :=
by
  sorry

end intersection_complement_l801_801765


namespace triangle_altitude_from_rectangle_l801_801569

theorem triangle_altitude_from_rectangle (a b : ℕ) (A : ℕ) (h : ℕ) (H1 : a = 7) (H2 : b = 21) (H3 : A = 147) (H4 : a * b = A) (H5 : 2 * A = h * b) : h = 14 :=
sorry

end triangle_altitude_from_rectangle_l801_801569


namespace distinct_prime_factors_sum_divisors_450_l801_801352

theorem distinct_prime_factors_sum_divisors_450 :
  let n := 450
  let sigma := (1 + 2) * (1 + 3 + 3^2) * (1 + 5 + 5^2)
  nat.num_distinct_prime_factors sigma = 2 := by
  sorry

end distinct_prime_factors_sum_divisors_450_l801_801352


namespace parallelogram_area_proof_perpendicular_vectors_l801_801068

noncomputable def abs : ℝ → ℝ := abs

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def vector (P Q : Point3D) : Point3D :=
{ x := Q.x - P.x,
  y := Q.y - P.y,
  z := Q.z - P.z }

def magnitude (v : Point3D) : ℝ :=
Real.sqrt (v.x ^ 2 + v.y ^ 2 + v.z ^ 2)

def dot_product (v w : Point3D) : ℝ :=
v.x * w.x + v.y * w.y + v.z * w.z

def orthogonal (v w : Point3D) : Prop :=
dot_product v w = 0

def parallelogram_area (v w : Point3D) : ℝ :=
magnitude v * magnitude w

theorem parallelogram_area_proof :
  ∃ S, 
  let A := Point3D.mk (-1) 2 1;
      B := Point3D.mk 1 2 1;
      C := Point3D.mk (-1) 6 4;
      AB := vector A B;
      AC := vector A C in
  parallelogram_area AB AC = 10 :=
by
  let A := Point3D.mk (-1) 2 1;
  let B := Point3D.mk 1 2 1;
  let C := Point3D.mk (-1) 6 4;
  let AB := vector A B;
  let AC := vector A C;
  use 10;
  sorry

theorem perpendicular_vectors :
  ∃ a : Point3D,
  let A := Point3D.mk (-1) 2 1;
      B := Point3D.mk 1 2 1;
      C := Point3D.mk (-1) 6 4;
      AB := vector A B;
      AC := vector A C in
  orthogonal a AB ∧ 
  orthogonal a AC ∧
  magnitude a = 10 ∧ 
  (a = Point3D.mk 0 (-6) 8 ∨ a = Point3D.mk 0 6 (-8)) :=
by
  let A := Point3D.mk (-1) 2 1;
  let B := Point3D.mk 1 2 1;
  let C := Point3D.mk (-1) 6 4;
  let AB := vector A B;
  let AC := vector A C;
  have : orthogonal (Point3D.mk 0 (-6) 8) AB ∧ orthogonal (Point3D.mk 0 (-6) 8) AC ∧ magnitude (Point3D.mk 0 (-6) 8) = 10 := by sorry;
  have : orthogonal (Point3D.mk 0 6 (-8)) AB ∧ orthogonal (Point3D.mk 0 6 (-8)) AC ∧ magnitude (Point3D.mk 0 6 (-8)) = 10 := by sorry;
  exact ⟨Point3D.mk 0 (-6) 8, this⟩ ∨ ⟨Point3D.mk 0 6 (-8), this⟩;
  sorry

end parallelogram_area_proof_perpendicular_vectors_l801_801068


namespace circumcircle_tangency_l801_801203

variables {A B C D : Type*} [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C] [euclidean_geometry D]
variables (AD : line_segment A D) (ABC : triangle A B C)
variables (k : circle A) (BC : line_segment B C)
variable [tangent_to : touches_at_point k BC D]

theorem circumcircle_tangency
  (h1 : is_angle_bisector AD ABC)
  (h2 : passes_through k A)
  (h3 : touches k BC D) :
  tangents_at_point circumscribed_circle ABC k A := sorry

end circumcircle_tangency_l801_801203


namespace lines_of_symmetry_l801_801740

theorem lines_of_symmetry (f : ℝ → ℝ) (x : ℝ) (h : ∀ x, f x = x + 1 / x) :
  (∀ x, f x = (1 + real.sqrt 2) * x) ∨ (∀ x, f x = (1 - real.sqrt 2) * x) :=
sorry

end lines_of_symmetry_l801_801740


namespace pool_ratio_three_to_one_l801_801167

theorem pool_ratio_three_to_one (P : ℕ) (B B' : ℕ) (k : ℕ) :
  (P = 5 * B + 2) → (k * P = 5 * B' + 1) → k = 3 :=
by
  intros h1 h2
  sorry

end pool_ratio_three_to_one_l801_801167


namespace solve_system_l801_801563

theorem solve_system :
    (∃ x y z : ℝ, 5 * x^2 + 3 * y^2 + 3 * x * y + 2 * x * z - y * z - 10 * y + 5 = 0 ∧
                49 * x^2 + 65 * y^2 + 49 * z^2 - 14 * x * y - 98 * x * z + 14 * y * z - 182 * x - 102 * y + 182 * z + 233 =0
                ∧ ((x = 0 ∧ y = 1 ∧ z = -2)
                   ∨ (x = 2/7 ∧ y = 1 ∧ z = -12/7))) :=
by
  sorry

end solve_system_l801_801563


namespace partition_of_remaining_cells_l801_801747

/-- 
Define the initial dimensions and the cut-out cells, then show the partition is correct.
-/
def initial_dimensions : ℕ × ℕ := (5, 7)

def cut_out_cells : set (ℕ × ℕ) := { (1, 1), (2, 3), (4, 5) }

def remaining_cells (dims : ℕ × ℕ) (cut : set (ℕ × ℕ)) : set (ℕ × ℕ) :=
  { (x, y) | x < dims.1 ∧ y < dims.2 } \ cut

def part_size := 4

def num_parts := 8

-- the Lean theorem statement to be proved
theorem partition_of_remaining_cells :
  ∃ (parts : fin num_parts → set (ℕ × ℕ)),
    (∀ i, parts i ⊆ remaining_cells initial_dimensions cut_out_cells) ∧
    (∀ i, parts i.card = part_size) ∧
    (∀ i j, i ≠ j → disjoint (parts i) (parts j)) ∧
    (∀ i j, ∃ (f : (ℕ × ℕ) → (ℕ × ℕ)), (∀ x ∈ parts i, f x ∈ parts j)) :=
by sorry

end partition_of_remaining_cells_l801_801747


namespace surface_area_of_sphere_given_cube_volume_8_l801_801243

theorem surface_area_of_sphere_given_cube_volume_8 
  (volume_of_cube : ℝ)
  (h₁ : volume_of_cube = 8) :
  ∃ (surface_area_of_sphere : ℝ), 
  surface_area_of_sphere = 12 * Real.pi :=
by
  sorry

end surface_area_of_sphere_given_cube_volume_8_l801_801243


namespace circle_center_radius_sum_l801_801846

theorem circle_center_radius_sum : 
  ∀ (x y : ℝ), (x^2 - 3*y - 12 = -y^2 + 12*x + 72) → 
    (let a := 6 in
    let b := 3/2 in
    let r := (120.25 : ℝ).sqrt in
    a + b + r = 18.45) :=
by
  intros x y h
  let a := 6
  let b := 3/2
  let r := (120.25 : ℝ).sqrt
  have : a + b + r = 18.45 := sorry
  exact this

end circle_center_radius_sum_l801_801846


namespace max_value_unbounded_or_sqrt_l801_801148

noncomputable def max_value_expr (a b c : ℝ) (θ : ℝ) : ℝ := 
  a * Real.cos θ + b * Real.sin θ + c * Real.tan θ

theorem max_value_unbounded_or_sqrt (a b c : ℝ) :
  (∀ L, ∃ θ, max_value_expr a b c θ > L) ∨ 
  (c = 0 → ∀ θ, max_value_expr a b 0 θ ≤ sqrt (a^2 + b^2) ∧ 
                 ∃ θ, max_value_expr a b 0 θ = sqrt (a^2 + b^2)) :=
by
  sorry

end max_value_unbounded_or_sqrt_l801_801148


namespace complement_B_A_l801_801751

def A (x : ℝ) : Prop := 
  ∃ y : ℝ, y = sqrt (x - 4) / (|x| - 5) ∧ x ≥ 4 ∧ |x| ≠ 5

def B (y : ℝ) : Prop := 
  ∃ x : ℝ, y = sqrt (x^2 - 6 * x + 13) ∧ y ≥ 2

theorem complement_B_A :(∀ x, B x ↔ x ∈ set.Ici 2) → (∀ x, A x → x ∈ set.Icc 4 5 ∨ x ∈ set.Ioi 5) →
  set.compl {x | ∃ y, B y ∧ y = sqrt (x^2 - 6 * x + 13)} {x | ∃ y, A y ∧ y = sqrt (x - 4) / (|x| - 5)}
  = {x | x ∈ set.Ioi 2 ∩ set.Iio 4 ∪ set.Icc (5 : ℝ) (5 : ℝ)} :=
begin
  sorry
end

end complement_B_A_l801_801751


namespace max_value_equilateral_quadrilateral_l801_801721

open Set

theorem max_value_equilateral_quadrilateral
  (A B C D P Q R S : Point) 
  (equilateral_triangles_outside : 
    ∀ A B P : Point, ∀ B C R : Point, ∀ C D S : Point, ∀ D A P : Point,
      is_equilateral_triangle A B Q ∧ is_equilateral_triangle B C R ∧
      is_equilateral_triangle C D S ∧ is_equilateral_triangle D A P)
  (midpoints : ∀ X Y Z W : Point, 
    is_midpoint X P Q ∧ is_midpoint Y Q R ∧ is_midpoint Z R S ∧ is_midpoint W S P) :
  ∀ (AC BD : ℝ), 
  ∃ (XZ YW : ℝ), 
  XZ = dist X Z ∧ YW = dist Y W →
  XZ + YW ≤ (sqrt 3 + 1) / 2 * (AC + BD) :=
by
  sorry

end max_value_equilateral_quadrilateral_l801_801721


namespace domain_of_f_sqrt_log_is_interval_e_l801_801582

noncomputable def domain_of_sqrt_log (f : ℝ → ℝ) : set ℝ :=
  {x | x > 0 ∧ 1 - real.log x ≥ 0}

theorem domain_of_f_sqrt_log_is_interval_e :
  domain_of_sqrt_log (λ x, real.sqrt (1 - real.log x)) = set.Ioc 0 real.exp 1 := 
begin
  sorry
end

end domain_of_f_sqrt_log_is_interval_e_l801_801582


namespace correct_quotient_l801_801462

-- Definition of the problem in Lean 4
theorem correct_quotient (c d : ℕ) (hc : 10 ≤ c ∧ c < 100) (hd : d > 0)
  (h_swap : let c' := (c % 10) * 10 + (c / 10) in c' = 8 * d) :
  (c : ℚ) / d = 5.75 := 
sorry

end correct_quotient_l801_801462


namespace perimeter_triangle_DEF_is_340_l801_801593

noncomputable def perimeter_triangle (r : ℝ) (dp : ℝ) (pe : ℝ) : ℝ :=
  let y := ((94770 + real.sqrt (94770^2 - 4 * 819 * 952875)) / 1638) in
  let s := (dp + pe + y) in
  2 * s

theorem perimeter_triangle_DEF_is_340 :
  perimeter_triangle 15 36 29 = 340 := by
  sorry

end perimeter_triangle_DEF_is_340_l801_801593


namespace largest_even_digit_integer_l801_801263

theorem largest_even_digit_integer (n : ℕ) : n < 6000 ∧ ∀ d, d ∈ {0, 2, 4, 6, 8} → d ∈ digits(10 n) ∧ 4 ∈ digits(10 n) ∧ 8 ∣ n ↔ n = 5408 := 
sorry

end largest_even_digit_integer_l801_801263


namespace victor_candy_l801_801950

theorem victor_candy (total_candies : ℕ) (friends : ℕ) (desired_candies : ℕ) :
  total_candies = 379 → friends = 6 → 
  (exists k, total_candies = k * friends ∧ desired_candies = total_candies - k * friends) →
  desired_candies = 378 :=
by
  intros h_total h_friends h_exists
  rcases h_exists with ⟨ k, h_div, h_desired⟩
  simp at h_desired
  sorry

end victor_candy_l801_801950


namespace find_a_b_is_even_function_is_decreasing_on_neg_infty_0_l801_801402

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 2^x + 2^(a * x + b)

-- Problem (Ⅰ)
theorem find_a_b :
  (∃ a b, 
    (f 1 a b = 5 / 2) ∧ 
    (f 2 a b = 17 / 4) ∧ 
    a = -1 ∧ b = 0) := by
  sorry

def f_even (x : ℝ) : ℝ := 2^x + 2^(-x)

-- Problem (Ⅱ)
theorem is_even_function : ∀ x : ℝ, f_even x = f_even (-x) := by
  sorry

-- Problem (Ⅲ)
theorem is_decreasing_on_neg_infty_0 : ∀ (x₁ x₂ : ℝ), x₁ < x₂ → x₂ < 0 → f_even x₁ > f_even x₂ := by
  sorry

end find_a_b_is_even_function_is_decreasing_on_neg_infty_0_l801_801402


namespace irrational_sqrt2_less_than_4_l801_801960

theorem irrational_sqrt2_less_than_4 : ∃ x : ℝ, irrational x ∧ x = real.sqrt 2 ∧ x < 4 :=
by {
  use real.sqrt 2,
  split,
  { sorry }, -- Prove that real.sqrt 2 is irrational
  split,
  { refl }, -- Prove that x = real.sqrt 2
  { sorry } -- Prove that real.sqrt 2 < 4
}

end irrational_sqrt2_less_than_4_l801_801960


namespace inverse_point_F_is_C_l801_801602

noncomputable def F (a b : ℝ) : ℂ := a + b * complex.i

theorem inverse_point_F_is_C {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 > 1) :
  let invF := (a / (a^2 + b^2)) - (b / (a^2 + b^2)) * complex.i in
  complex.abs invF < 1 ∧ invF.re > 0 ∧ invF.im < 0 := 
by
  sorry

end inverse_point_F_is_C_l801_801602


namespace subset_A_if_inter_eq_l801_801803

variable {B : Set ℝ}

def A : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem subset_A_if_inter_eq:
  A ∩ B = B ↔ B = ∅ ∨ B = {1} ∨ B = { x | 0 < x ∧ x < 2 } :=
by
  sorry

end subset_A_if_inter_eq_l801_801803


namespace weekly_rental_cost_l801_801673

theorem weekly_rental_cost (W : ℝ) 
  (monthly_cost : ℝ := 40)
  (months_in_year : ℝ := 12)
  (weeks_in_year : ℝ := 52)
  (savings : ℝ := 40)
  (total_year_cost_month : ℝ := months_in_year * monthly_cost)
  (total_year_cost_week : ℝ := total_year_cost_month + savings) :
  (total_year_cost_week / weeks_in_year) = 10 :=
by 
  sorry

end weekly_rental_cost_l801_801673


namespace find_a_for_fx_inequality_l801_801777

theorem find_a_for_fx_inequality
  (a : ℝ)
  (f : ℝ → ℝ)
  (h : ∀ x, f x = sin (2 * x) + a * cos (2 * x))
  (h_max : ∀ x, |f x| ≤ f (π / 8))
  : a = 1 :=
sorry

end find_a_for_fx_inequality_l801_801777


namespace expression_values_count_l801_801085

theorem expression_values_count {E : ℝ} (m : ℝ) (h : m = | |E| - 2 |) : (m = 5) → (∃ a b : ℝ, (a = E ∨ b = E) ∧ (a ≠ b) ∧ m = 5) :=
by
  sorry

end expression_values_count_l801_801085


namespace area_of_circle_l801_801214
open Real

-- Define the circumference condition
def circumference (r : ℝ) : ℝ :=
  2 * π * r

-- Define the area formula
def area (r : ℝ) : ℝ :=
  π * r * r

-- The given radius derived from the circumference
def radius_given_circumference (C : ℝ) : ℝ :=
  C / (2 * π)

-- The target proof statement
theorem area_of_circle (C : ℝ) (h : C = 36) : (area (radius_given_circumference C)) = 324 / π :=
by
  sorry

end area_of_circle_l801_801214


namespace avg_of_last_6_numbers_l801_801572

theorem avg_of_last_6_numbers 
  (avg_11 : ℕ) (avg_first_6 : ℕ) (num_6 : ℕ) 
  (h_avg_11 : avg_11 = 60) (h_avg_first_6 : avg_first_6 = 98) (h_num_6 : num_6 = 318) 
  : (660 - (6 * 98 - 318)) / 6 = 65 :=
by
  rw [h_avg_11, h_avg_first_6, h_num_6]
  sorry

end avg_of_last_6_numbers_l801_801572


namespace power_function_sqrt2_l801_801090

theorem power_function_sqrt2 (n : ℝ) (f : ℝ → ℝ) (h1 : f = λ x, x ^ n) (h2 : f 2 = 1 / 4) : 
  f (real.sqrt 2) = 1 / 2 :=
by
  sorry

end power_function_sqrt2_l801_801090


namespace parallel_to_a_perpendicular_to_a_l801_801034

-- Definition of vectors a and b and conditions
def a : ℝ × ℝ := (3, 4)
def b (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Mathematical statement for Problem (1)
theorem parallel_to_a (x y : ℝ) (h : b x y) (h_parallel : 3 * y - 4 * x = 0) :
  (x = 3/5 ∧ y = 4/5) ∨ (x = -3/5 ∧ y = -4/5) := 
sorry

-- Mathematical statement for Problem (2)
theorem perpendicular_to_a (x y : ℝ) (h : b x y) (h_perpendicular : 3 * x + 4 * y = 0) :
  (x = -4/5 ∧ y = 3/5) ∨ (x = 4/5 ∧ y = -3/5) := 
sorry

end parallel_to_a_perpendicular_to_a_l801_801034


namespace number_of_functions_with_given_range_l801_801711

theorem number_of_functions_with_given_range : 
  let S := {2, 5, 10}
  let R (x : ℤ) := x^2 + 1
  ∃ f : ℤ → ℤ, (∀ y ∈ S, ∃ x : ℤ, f x = y) ∧ (f '' {x | R x ∈ S} = S) :=
    sorry

end number_of_functions_with_given_range_l801_801711


namespace circumradius_relationship_l801_801266

theorem circumradius_relationship 
  (a b c a' b' c' R : ℝ)
  (S S' p p' : ℝ)
  (h₁ : R = (a * b * c) / (4 * S))
  (h₂ : R = (a' * b' * c') / (4 * S'))
  (h₃ : S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (h₄ : S' = Real.sqrt (p' * (p' - a') * (p' - b') * (p' - c')))
  (h₅ : p = (a + b + c) / 2)
  (h₆ : p' = (a' + b' + c') / 2) :
  (a * b * c) / Real.sqrt (p * (p - a) * (p - b) * (p - c)) = 
  (a' * b' * c') / Real.sqrt (p' * (p' - a') * (p' - b') * (p' - c')) :=
by 
  sorry

end circumradius_relationship_l801_801266


namespace distance_midpoint_to_vertexA_circumradius_of_triangle_l801_801999

-- Define the sides of the triangle
def side_a : ℝ := 3
def side_b : ℝ := 4
def side_c : ℝ := 5

-- Define the property of the triangle being right-angled
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Define the midpoint of the hypotenuse
def midpoint_hypotenuse (c : ℝ) : (ℝ × ℝ) :=
  (c / 2, 0)

-- Define the coordinates of the vertices
def vertex_A : (ℝ × ℝ) := (side_a, side_b)

-- Function to compute distance between two points
def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

-- Define the circumradius of the triangle
def circumradius (c : ℝ) : ℝ := c / 2

-- Prove the properties
theorem distance_midpoint_to_vertexA :
  distance (midpoint_hypotenuse side_c) vertex_A = real.sqrt 65 / 2 := sorry

theorem circumradius_of_triangle :
  circumradius side_c = 2.5 := sorry

end distance_midpoint_to_vertexA_circumradius_of_triangle_l801_801999


namespace total_five_digit_odd_and_multiples_of_5_l801_801491

def count_odd_five_digit_numbers : ℕ :=
  let choices := 9 * 10 * 10 * 10 * 5
  choices

def count_multiples_of_5_five_digit_numbers : ℕ :=
  let choices := 9 * 10 * 10 * 10 * 2
  choices

theorem total_five_digit_odd_and_multiples_of_5 : count_odd_five_digit_numbers + count_multiples_of_5_five_digit_numbers = 63000 :=
by
  -- Proof Placeholder
  sorry

end total_five_digit_odd_and_multiples_of_5_l801_801491


namespace distance_home_to_school_l801_801640

theorem distance_home_to_school :
  ∃ (D : ℝ), D = 2 ∧
    (∀ (T : ℝ),
      4 * (T + (7 / 60)) = D ∧
      8 * (T - (8 / 60)) = D) :=
begin
  use 2,
  split,
  { refl },
  { intro T,
    split,
    { sorry },
    { sorry } }
end

end distance_home_to_school_l801_801640


namespace tangent_length_min_triangle_area_max_and_line_eq_line_equations_l801_801069

-- Definition for Circle and Line
structure Circle :=
  (x : ℝ)
  (y : ℝ)
  (r : ℝ)
  (hr : r > 0)

structure Line := 
  (a : ℝ)
  (b : ℝ)
  (h : a ≠ 0 ∨ b ≠ 0)
  (lhs (point: ℝ × ℝ) := a * point.1 / 9 - b * point.2 / 12)
  (rhs : ℝ := 1)

-- Definition for tangency condition
def externally_tangent (c1 c2 : Circle) : Prop := 
  real.sqrt ((c1.x - c2.x)^2 + (c1.y - c2.y)^2) = c1.r + c2.r

-- Proof Problem (1): Minimum tangent length
theorem tangent_length_min (a b : ℝ) (c1 c2 : Circle) (l : Line)
  (h1 : c1.x = 0 ∧ c1.y = 0 ∧ c1.r = 3)
  (h2 : c2.x = 3 ∧ c2.y = 4 ∧ externally_tangent c1 c2)
  (h3 : l.lhs (a, b) = l.rhs)
  (h4 : a = b + 3) :
  real.sqrt ((a - c2.x)^2 + (b - c2.y)^2 - c2.r^2) = 2 := sorry

-- Proof Problem (2): Maximum area and Equation of Line
theorem triangle_area_max_and_line_eq (c2 : Circle) 
  (h2 : c2.x = 3 ∧ c2.y = 4 ∧ c2.r = 2)
  (l1 : ℝ → ℝ)
  (P Q : ℝ × ℝ)
  (A : ℝ × ℝ := (1, 0))
  (h_line : ∃ k, ∀ x, l1 x = k *(x - A.1))
  (h_intersect : P ≠ Q ∧ 
    ((P.1 - c2.x)^2 + (P.2 - c2.y)^2 = c2.r^2) ∧
    ((Q.1 - c2.x)^2 + (Q.2 - c2.y)^2 = c2.r^2)) :
  let d := abs (c2.x * k - c2.y + b) / real.sqrt (k^2 + b^2) in
  real.sqrt (4 - d^2) ≤ 2 ∧ (d * real.sqrt (4 - d^2) ≤ 2) :=
  sorry

-- Additional part related to line equations
theorem line_equations (c2 : Circle) 
  (h2 : c2.x = 3 ∧ c2.y = 4 ∧ c2.r = 2)
  (A : ℝ × ℝ := (1, 0))
  : ∃ l1 : ℝ → ℝ, (∀ x, (l1 x = x - 1) ∨ (l1 x = 7 * x - 7)) :=
  sorry

end tangent_length_min_triangle_area_max_and_line_eq_line_equations_l801_801069


namespace more_even_products_l801_801949

theorem more_even_products :
  let S := {1, 2, 3, 4, 5}
  let pairs := { (a, b) | a ∈ S ∧ b ∈ S ∧ a ≤ b }
  let products := { a * b | (a, b) ∈ pairs }
  let even_products := { p | p ∈ products ∧ p % 2 = 0 }
  let odd_products := { p | p ∈ products ∧ ¬(p % 2 = 0) }
  even_products.card > odd_products.card :=
by
  let S := {1, 2, 3, 4, 5}
  let pairs := { (a, b) | a ∈ S ∧ b ∈ S ∧ a ≤ b }
  let products := { a * b | (a, b) ∈ pairs }
  let even_products := { p | p ∈ products ∧ p % 2 = 0 }
  let odd_products := { p | p ∈ products ∧ ¬(p % 2 = 0) }
  have h1 : even_products = {2, 4, 6, 8, 10, 12, 20} by sorry
  have h2 : odd_products = {3, 5, 15} by sorry
  have h3 : even_products.card = 7 by sorry
  have h4 : odd_products.card = 3 by sorry
  show even_products.card > odd_products.card, from
    have h : 7 > 3 by norm_num
    h

end more_even_products_l801_801949


namespace f_inequality_l801_801162

open Real

noncomputable def f : ℕ+ → ℝ :=
λ x, if x = 1 then 1/2 else sin (π / 2 * f (x - 1))

theorem f_inequality (x : ℕ+) (hx : x ≥ 2) : 1 - f x < π / 4 * (1 - f (x - 1)) :=
sorry

end f_inequality_l801_801162


namespace james_total_money_at_end_of_year_l801_801131

-- Definitions based on conditions
def weekly_investment : ℕ := 2000
def initial_amount : ℕ := 250000
def weeks_in_year : ℕ := 52
def windfall_rate : ℝ := 0.5

-- The total amount James has at the end of the year
theorem james_total_money_at_end_of_year 
  (weekly_investment : ℕ)
  (initial_amount : ℕ)
  (weeks_in_year : ℕ)
  (windfall_rate : ℝ) :
  let total_deposit := weekly_investment * weeks_in_year
      balance_before_windfall := initial_amount + total_deposit
      windfall := windfall_rate * balance_before_windfall
      total_balance := balance_before_windfall + windfall in
  total_balance = 885000 :=
by {
  let total_deposit := 2000 * 52
  let balance_before_windfall := 250000 + total_deposit
  let windfall := 0.5 * (balance_before_windfall : ℝ)
  let total_balance := balance_before_windfall + windfall
  show total_balance = 885000,
  sorry
}

end james_total_money_at_end_of_year_l801_801131


namespace period_of_function_l801_801628

theorem period_of_function : ∀ x, 2 * sin x - 2 * cos x = 2 * sin (x + 2 * π) - 2 * cos (x + 2 * π) :=
by
  intro x
  sorry

end period_of_function_l801_801628


namespace initial_investment_l801_801559

noncomputable def invested_amount (A : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  A / (1 + r) ^ n

theorem initial_investment
  (A : ℝ) (r : ℝ) (n : ℕ)
  (hA : A = 635.48)
  (hr : r = 0.12)
  (hn : n = 6) :
  invested_amount A r n ≈ 315.84 :=
sorry

end initial_investment_l801_801559


namespace problem1_correct_problem2_correct_l801_801320

noncomputable def problem1 : ℝ :=
  real.sqrt ((-2) ^ 2) + real.sqrt 2 * (1 - real.sqrt (1 / 2)) + abs (-real.sqrt 8)

noncomputable def expected1 : ℝ := 1 + 3 * real.sqrt 2

theorem problem1_correct : problem1 = expected1 := by
  -- Proof goes here
  sorry

noncomputable def problem2 : ℝ :=
  real.sqrt 18 - real.sqrt 8 + (real.sqrt 3 + 1) * (real.sqrt 3 - 1)

noncomputable def expected2 : ℝ := real.sqrt 2 + 2

theorem problem2_correct : problem2 = expected2 := by
  -- Proof goes here
  sorry

end problem1_correct_problem2_correct_l801_801320


namespace cricket_team_average_age_l801_801577

variables (A : ℝ) (a_c a_w : ℝ)
variables (R : ℝ) -- Age of the retiring player from the remaining group

-- Given conditions
def captain_age := a_c = 24
def wicket_keeper_age := a_w = a_c + 7
def average_remaining_players := (11 * A - a_c - a_w) / 9 = A - 1
def new_player_age := ∃ R, true -- R - the age of retiring player is universally quantified
def change_in_average_age := (11 * A - R + (R - 5)) / 11 = A - 0.5

-- To be proved
def new_average_age := (11 * A - 5) / 11 = 22.5

theorem cricket_team_average_age :
  captain_age ∧ wicket_keeper_age ∧ average_remaining_players ∧ (∃ R, change_in_average_age) → new_average_age :=
by
  intros,
  sorry -- proof goes here

end cricket_team_average_age_l801_801577


namespace nullity_of_matrix_l801_801780

noncomputable def matrix (d c : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![3, -1, 1],
  ![2, d, -2],
  ![1, 1, c]
]

theorem nullity_of_matrix {d c : ℝ} (h_inv : matrix d c * matrix d c = 1) : 
  Matrix.nullity (matrix d c) = 0 :=
sorry

end nullity_of_matrix_l801_801780


namespace probability_of_2_points_for_question_11_probability_of_total_7_points_l801_801508

-- Define the probabilities for selecting options
def prob_select_one := (1 : ℚ) / 3
def prob_select_two := (1 : ℚ) / 3
def prob_select_three := (1 : ℚ) / 3

-- Define the events and their probabilities
def prob_correct_selection_one := (1 : ℚ) / 2  -- Given one option is selected, the probability it is correct
def prob_correct_selection_two := (1 : ℚ) / 6  -- Probability of selecting the two correct options out of possible pairs

-- Part 1: Probability of getting 2 points for question 11
theorem probability_of_2_points_for_question_11 : 
  (prob_select_one * prob_correct_selection_one) = (1 : ℚ) / 6 := 
sorry

-- Part 2: Probability of scoring a total of 7 points for questions 11 and 12
theorem probability_of_total_7_points : 
  2 * ((prob_select_one * prob_correct_selection_one) * (prob_select_two * prob_correct_selection_two)) = (1 : ℚ) / 54 := 
sorry

end probability_of_2_points_for_question_11_probability_of_total_7_points_l801_801508


namespace wendy_chocolates_l801_801627

theorem wendy_chocolates (h : ℕ) : 
  let chocolates_per_4_hours := 1152
  let chocolates_per_hour := chocolates_per_4_hours / 4
  (chocolates_per_hour * h) = 288 * h :=
by
  sorry

end wendy_chocolates_l801_801627


namespace find_integer_n_l801_801358

-- Define the condition range for n
def is_in_range (n : ℤ) : Prop := n ≥ -180 ∧ n ≤ 180

-- State the main theorem using Lean syntax
theorem find_integer_n : ∃ n : ℤ, is_in_range n ∧ real.sin (n * real.pi / 180) = real.sin (720 * real.pi / 180) :=
by
  sorry  -- Proof to be completed.


end find_integer_n_l801_801358


namespace skew_pairs_of_edges_of_cube_l801_801098

-- Conditions provided based on problem statement
def is_edge_of_cube (l : ℕ) : Prop := l ≤ 12

def are_skew (l1 l2 : ℕ) : Prop := 
  ¬ (l1 = l2 ∨ intersects l1 l2 ∨ coplanar l1 l2)

-- The proof problem
theorem skew_pairs_of_edges_of_cube : 
  ∀ (edges : list ℕ), (∀ l ∈ edges, is_edge_of_cube l) → ∃ (n : ℕ), n = 24 
    ∧ (∃ pairs : list (ℕ × ℕ), 
      (∀ p ∈ pairs, are_skew (p.1) (p.2)) 
      ∧ length pairs = 24) :=
by 
  -- We'll use the fact that there are 12 edges in the cube
  sorry

end skew_pairs_of_edges_of_cube_l801_801098


namespace num_ways_5_balls_4_boxes_l801_801430

-- Define the function calculating number of ways to place n balls into m boxes
def num_ways (n : ℕ) (m : ℕ) : ℕ := m ^ n

-- The proof that placing 5 distinguishable balls into 4 distinguishable boxes
theorem num_ways_5_balls_4_boxes : num_ways 5 4 = 1024 := by
  -- Calculation of 4^5
  unfold num_ways
  -- Check if the calculation matches 1024
  norm_num
  exact rfl

end num_ways_5_balls_4_boxes_l801_801430


namespace find_k_and_f_min_total_cost_l801_801621

-- Define the conditions
def construction_cost (x : ℝ) : ℝ := 60 * x
def energy_consumption_cost (x : ℝ) : ℝ := 40 - 4 * x
def total_cost (x : ℝ) : ℝ := construction_cost x + 20 * energy_consumption_cost x

theorem find_k_and_f :
  (∀ x, 0 ≤ x ∧ x ≤ 10 → energy_consumption_cost 0 = 8 → energy_consumption_cost x = 40 - 4 * x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 10 → total_cost x = 800 - 74 * x) :=
by
  sorry

theorem min_total_cost :
  (∀ x, 0 ≤ x ∧ x ≤ 10 → 800 - 74 * x ≥ 70) ∧
  total_cost 5 = 70 :=
by
  sorry

end find_k_and_f_min_total_cost_l801_801621


namespace determine_eccentricity_l801_801061

def hyperbola_eccentricity (a b : ℝ) (h : a > 0 ∧ b > 0) (area_triangle : ℝ) : Prop :=
  let e := (√(a^2 + b^2)) / a in
  area_triangle = 4 → b = a →  e = √2

-- Statement of the problem
theorem determine_eccentricity (a b : ℝ) (h : a > 0 ∧ b > 0)
  (intersect_line : ∀ x, x = -2 → ∃ y, y = (b / a) * x ∨ y = -(b / a) * x)
  (area_triangle : ∃ area, area = 4) :
  (b = a) → ((√(a^2 + b^2)) / a) = √2 :=
begin
  sorry
end

end determine_eccentricity_l801_801061


namespace number_of_ways_to_place_balls_into_boxes_l801_801426

theorem number_of_ways_to_place_balls_into_boxes : 
  (number_of_ways (balls : ℕ) (boxes : ℕ) : ℕ) where balls = 5 ∧ boxes = 4 = 4^5 :=
begin
  sorry
end

end number_of_ways_to_place_balls_into_boxes_l801_801426


namespace value_of_kaftan_l801_801990

theorem value_of_kaftan (K : ℝ) (h : (7 / 12) * (12 + K) = 5 + K) : K = 4.8 :=
by
  sorry

end value_of_kaftan_l801_801990


namespace determine_linear_function_l801_801388

noncomputable theory -- To manage potential non-computability issues

open function

-- Define a structure to encapsulate the conditions of the linear function
structure linear_function (f : ℝ → ℝ) :=
(is_linear : ∃ (a b : ℝ), f = λ x, a*x + b)
(f_at_neg2 : f (-2) = -1)
(f_at_0_plus_f_at_2 : f 0 + f 2 = 10)

-- The main theorem to prove that the linear function according to the given conditions is f(x) = 2x + 3
theorem determine_linear_function (f : ℝ → ℝ) (h : linear_function f) : f = (λ x, 2*x + 3) :=
by sorry

end determine_linear_function_l801_801388


namespace num_ways_5_balls_4_boxes_l801_801429

-- Define the function calculating number of ways to place n balls into m boxes
def num_ways (n : ℕ) (m : ℕ) : ℕ := m ^ n

-- The proof that placing 5 distinguishable balls into 4 distinguishable boxes
theorem num_ways_5_balls_4_boxes : num_ways 5 4 = 1024 := by
  -- Calculation of 4^5
  unfold num_ways
  -- Check if the calculation matches 1024
  norm_num
  exact rfl

end num_ways_5_balls_4_boxes_l801_801429


namespace permutation_count_l801_801230

open Function

-- Definitions based on the conditions
def valid_permutation_condition (n : List ℕ) (i : ℕ) : Prop :=
  (n.take i) ≠ (List.range' 1 i)

def count_valid_permutations (arr : List ℕ) : ℕ :=
  (arr.permutations.filter (λ n, ∀ i, 1 ≤ i ∧ i ≤ 5 → valid_permutation_condition n i)).length

-- The main theorem we need to prove
theorem permutation_count : count_valid_permutations [1, 2, 3, 4, 5, 6] = 259 := by
  sorry

end permutation_count_l801_801230


namespace percent_increase_quarter_l801_801233

-- Define the profit changes over each month
def profit_march (P : ℝ) := P
def profit_april (P : ℝ) := 1.40 * P
def profit_may (P : ℝ) := 1.12 * P
def profit_june (P : ℝ) := 1.68 * P

-- Starting Lean theorem statement
theorem percent_increase_quarter (P : ℝ) (hP : P > 0) :
  ((profit_june P - profit_march P) / profit_march P) * 100 = 68 :=
  sorry

end percent_increase_quarter_l801_801233


namespace car_closest_to_C_l801_801937

/-- The theorems of this problem:
 - Circular track of length 20 km.
 - Points marked: B, C, D, E at respective distances from A.
 - A car travels 367 km starting from A.

Prove: The car is closest to point C after traveling 367 km starting from A. -/

def closest_point := 'C

theorem car_closest_to_C (
    track_length : ℕ := 20,
    A_to_B : ℕ := 5,
    B_to_C : ℕ := 3,
    C_to_D : ℕ := 4,
    D_to_E : ℕ := 5,
    total_distance : ℕ := 367
) : closest_point = 'C :=
by 
  -- sorry to skip the proof.
  sorry

end car_closest_to_C_l801_801937


namespace find_angle_BEC_l801_801996

theorem find_angle_BEC (A B C D E : Type) (angle_A angle_B angle_D angle_DEC angle_C angle_CED angle_BEC : ℝ) 
  (hA : angle_A = 50) (hB : angle_B = 90) (hD : angle_D = 70) (hDEC : angle_DEC = 20)
  (h_quadrilateral_sum: angle_A + angle_B + angle_C + angle_D = 360)
  (h_C : angle_C = 150)
  (h_CED : angle_CED = angle_C - angle_DEC)
  (h_BEC: angle_BEC = 180 - angle_B - angle_CED) : angle_BEC = 110 :=
by
  -- Definitions according to the given problem
  have h1 : angle_C = 360 - (angle_A + angle_B + angle_D) := by sorry
  have h2 : angle_CED = angle_C - angle_DEC := by sorry
  have h3 : angle_BEC = 180 - angle_B - angle_CED := by sorry

  -- Proving the required angle
  have h_goal : angle_BEC = 110 := by
    sorry  -- Actual proof steps go here

  exact h_goal

end find_angle_BEC_l801_801996


namespace orlando_weight_gain_l801_801314

def weight_gain_statement (x J F : ℝ) : Prop :=
  J = 2 * x + 2 ∧ F = 1/2 * J - 3 ∧ x + J + F = 20

theorem orlando_weight_gain :
  ∃ x J F : ℝ, weight_gain_statement x J F ∧ x = 5 :=
by {
  sorry
}

end orlando_weight_gain_l801_801314


namespace gail_has_two_ten_dollar_bills_l801_801016

-- Define the given conditions
def total_amount : ℕ := 100
def num_five_bills : ℕ := 4
def num_twenty_bills : ℕ := 3
def value_five_bill : ℕ := 5
def value_twenty_bill : ℕ := 20
def value_ten_bill : ℕ := 10

-- The function to determine the number of ten-dollar bills
noncomputable def num_ten_bills : ℕ := 
  (total_amount - (num_five_bills * value_five_bill + num_twenty_bills * value_twenty_bill)) / value_ten_bill

-- Proof statement
theorem gail_has_two_ten_dollar_bills : num_ten_bills = 2 := by
  sorry

end gail_has_two_ten_dollar_bills_l801_801016


namespace t_le_S_l801_801389

variables {a b : ℝ}

def t := a + 2 * b
def S := a + b^2 + 1

theorem t_le_S : t ≤ S :=
by sorry

end t_le_S_l801_801389


namespace range_of_a_l801_801785

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + 2*a*x + a > 0) → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l801_801785


namespace linear_system_solution_l801_801786

theorem linear_system_solution (k x y : ℝ) (h₁ : x + y = 5 * k) (h₂ : x - y = 9 * k) (h₃ : 2 * x + 3 * y = 6) :
  k = 3 / 4 :=
by
  sorry

end linear_system_solution_l801_801786


namespace unique_id_tags_div_five_l801_801287

/-- Define the set of available characters. -/
def available_characters := ['A', 'I', 'M', 'E', 'Z', '2', '0', '2', '3']

/-- Define the condition for the number of unique ID tags with given constraints. -/
def num_unique_tags (chars : List Char) : Nat := sorry

/-- The theorem statement proving the expected result. -/
theorem unique_id_tags_div_five :
  let N := num_unique_tags available_characters in
  N / 5 = 864 :=
sorry


end unique_id_tags_div_five_l801_801287


namespace initial_files_count_l801_801863

theorem initial_files_count (deleted_files folders files_per_folder total_files initial_files : ℕ)
    (h1 : deleted_files = 21)
    (h2 : folders = 9)
    (h3 : files_per_folder = 8)
    (h4 : total_files = folders * files_per_folder)
    (h5 : initial_files = total_files + deleted_files) :
    initial_files = 93 :=
by
  sorry

end initial_files_count_l801_801863


namespace medians_concurrent_altitudes_concurrent_l801_801852

variables {V : Type*} [inner_product_space ℝ V]
variables {A B C : V}

-- Problem 1: Show the medians are concurrent (Centroid)
theorem medians_concurrent {M : V} 
  (hM : (∃ A' B' : V, 
            A' = (B + C) / 2 ∧ 
            B' = (A + C) / 2 ∧ 
            2 * M = A + A' ∧ 
            2 * M = B + B')) :
  (∃ M : V, 
     (A - M) + (B - M) + (C - M) = (0 : V)) := 
sorry

-- Problem 2: Show the altitudes are concurrent (Orthocenter)
theorem altitudes_concurrent {H : V} 
  (hH : (∃ M : V, 
            (∀ M : V, 
                ((A - M) • (B - M) = (A - M) • (C - M) 
                ↔ 
                B - M = C - M)) 
                ∧ 
                (M - A) • (B - C) = 0)) :
  (∃ H : V, 
     ∀ M' : V, 
       (A - M') • (B - C) = 0 
       ∧ 
       (B - M') • (A - C) = 0 
       ∧ 
       (C - M') • (A - B) = 0) := 
sorry

end medians_concurrent_altitudes_concurrent_l801_801852


namespace progress_regress_rate_approximation_l801_801669

theorem progress_regress_rate_approximation :
  ∃ n : ℕ, 
  (1.03 / 0.97)^n = 1000 ∧ abs ((n : ℝ) - 115) ≤ 1 :=
by
  sorry

end progress_regress_rate_approximation_l801_801669


namespace find_k_l801_801506

open Function

variable {α : Type*} [LinearOrderedField α] (a : ℕ → α) (S : ℕ → α)

def is_arithmetic_seq : Prop :=
∀ (n m : ℕ), a (n + 1) - a n = a (m + 1) - a m

def S_def : Prop :=
∀ (n : ℕ), S n = (n * (a 1 + a n)) / 2

theorem find_k (h_arith : is_arithmetic_seq a) (h_S_def : S_def S)
  (h_S2014 : S 2014 > 0) (h_S2015 : S 2015 < 0)
  (h_abs : ∀ n : ℕ, ∀ k : ℕ, |a n| ≥ |a k|) :
  1008 = 1008 :=
sorry

end find_k_l801_801506


namespace find_abc_l801_801590

def f (x : ℝ) : ℝ :=
if x < 0 then -2 - x
else if x <= 2 then Real.sqrt (4 - (x - 2) ^ 2) - 2
else 2 * (x - 2)

noncomputable def g (a b c : ℝ) (x : ℝ) : ℝ := a * f (b * x + 1) + c

theorem find_abc : ∃ a b c, 
  (∀ x: ℝ, g a b c (x) = f (x / 3 + 1) - 3)
  ∧ a = 1 
  ∧ b = (1 : ℝ) / 3 
  ∧ c = -3 :=
by
  sorry

end find_abc_l801_801590


namespace krystiana_monthly_rent_l801_801840

theorem krystiana_monthly_rent :
  let rooms_1 := 5
  let rent_1 := 15
  let occupancy_1 := 0.80
  let rooms_2 := 6
  let rent_2 := 25
  let occupancy_2 := 0.75
  let rooms_3 := 9
  let rent_3 := 2 * rent_1
  let occupancy_3 := 0.50
  let occupied_rooms_1 := rooms_1 * occupancy_1
  let occupied_rent_1 := rent_1 * occupied_rooms_1
  let occupied_rooms_2 := rooms_2 * occupancy_2
  let occupied_rent_2 := rent_2 * occupied_rooms_2
  let occupied_rooms_3 := rooms_3 * occupancy_3
  let occupied_rent_3 := rent_3 * occupied_rooms_3
  let total_rent := occupied_rent_1 + occupied_rent_2 + occupied_rent_3 in
  total_rent = 280 :=
by
  sorry

end krystiana_monthly_rent_l801_801840


namespace find_x_pow_24_l801_801436

noncomputable def x := sorry
axiom h : x + 1/x = -Real.sqrt 3

theorem find_x_pow_24 : x ^ 24 = 1 := sorry

end find_x_pow_24_l801_801436


namespace fruits_turned_yellow_on_friday_l801_801612

theorem fruits_turned_yellow_on_friday :
  ∃ (F : ℕ), F + 2*F = 6 ∧ 14 - F - 2*F = 8 :=
by
  existsi 2
  sorry

end fruits_turned_yellow_on_friday_l801_801612


namespace eve_spending_l801_801724

-- Definitions of the conditions
def cost_mitt : ℝ := 14.00
def cost_apron : ℝ := 16.00
def cost_utensils : ℝ := 10.00
def cost_knife : ℝ := 2 * cost_utensils -- Twice the amount of the utensils
def discount_rate : ℝ := 0.25
def num_nieces : ℝ := 3

-- Total cost before the discount for one kit
def total_cost_one_kit : ℝ :=
  cost_mitt + cost_apron + cost_utensils + cost_knife

-- Discount for one kit
def discount_one_kit : ℝ := 
  total_cost_one_kit * discount_rate

-- Discounted price for one kit
def discounted_cost_one_kit : ℝ :=
  total_cost_one_kit - discount_one_kit

-- Total cost for all kits
def total_cost_all_kits : ℝ :=
  num_nieces * discounted_cost_one_kit

-- The theorem statement
theorem eve_spending : total_cost_all_kits = 135.00 :=
by sorry

end eve_spending_l801_801724


namespace dice_sum_probability_lt_10_l801_801254

theorem dice_sum_probability_lt_10 {n : ℕ} (h1 : 1 ≤ n) (h2 : n ≤ 6) :
  let pairs := [(1, 1), (1, 2), (2, 1), (1, 3), (2, 2), (3, 1), (1, 4), (2, 3), (3, 2), (4, 1),
                (1, 5), (2, 4), (3, 3), (4, 2), (5, 1), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1),
                (2, 6), (3, 5), (4, 4), (5, 3), (6, 2), (3, 6), (4, 5), (5, 4), (6, 3)] in
  let total_outcomes := 6 * 6 in
  (pairs.length : ℕ) / total_outcomes = 5/6 :=
sorry

end dice_sum_probability_lt_10_l801_801254


namespace sikh_percentage_correct_l801_801812

-- Define the given conditions
def total_boys : ℕ := 850
def muslim_percentage : ℝ := 0.44
def hindu_percentage : ℝ := 0.32
def other_communities_boys : ℕ := 119

-- Calculate intermediate values to ensure conditions are considered
def muslim_boys : ℕ := (muslim_percentage * total_boys).toInt
def hindu_boys : ℕ := (hindu_percentage * total_boys).toInt
def total_muslim_hindu_other_boys : ℕ := muslim_boys + hindu_boys + other_communities_boys
def sikh_boys : ℕ := total_boys - total_muslim_hindu_other_boys
def sikh_percentage : ℝ := (sikh_boys.toFloat / total_boys.toFloat) * 100

-- Lean statement to be proven
theorem sikh_percentage_correct :
  sikh_percentage = 10 := by
  sorry

end sikh_percentage_correct_l801_801812


namespace point_Q_motion_l801_801083

theorem point_Q_motion (ω t: ℝ) (hω : 0 < ω) :
  (P : ℝ×ℝ) → P = (cos(ω * t), sin(ω * t)) →
  (Q : ℝ×ℝ) → Q = (-2 * cos(ω * t) * sin(ω * t), sin(ω * t)^2 - cos(ω * t)^2) →
  ∃ t' (h : P = (cos(ω * t'), sin(ω * t'))), 
    Q = (cos(-2 * ω * t' + 3 * π / 2), sin(-2 * ω * t' + 2 * π / 3)) ∧
    ω *= 1 ∧ -2 * ω *= 2 * -ω :=
by
  sorry

end point_Q_motion_l801_801083


namespace tan_three_halves_pi_sub_alpha_l801_801756

theorem tan_three_halves_pi_sub_alpha (α : ℝ) (h : Real.cos (π - α) = -3/5) :
    Real.tan (3 * π / 2 - α) = 3/4 ∨ Real.tan (3 * π / 2 - α) = -3/4 := by
  sorry

end tan_three_halves_pi_sub_alpha_l801_801756


namespace perfect_square_expression_l801_801007

theorem perfect_square_expression (n : ℕ) (h : 7 ≤ n) : ∃ k : ℤ, (n + 2) ^ 2 = k ^ 2 :=
by 
  sorry

end perfect_square_expression_l801_801007


namespace equal_faces_iff_conditions_l801_801886

structure Tetrahedron (V : Type) [EuclideanSpace V] :=
(vertices : Fin 4 → V)
(is_tetrahedron : list.pairwise ((≠) on vertices))

def all_faces_equal {V : Type} [EuclideanSpace V] (T : Tetrahedron V) : Prop :=
  ∀ (i j k l : Fin 4), i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l →
  euclidean_distance (T.vertices i) (T.vertices j) =
  euclidean_distance (T.vertices k) (T.vertices l)

def condition_a {V : Type} [EuclideanSpace V] (T : Tetrahedron V) : Prop :=
-- Assuming some function that checks if extending the tetrahedron forms a rectangular parallelepiped
sorry

def condition_b {V : Type} [EuclideanSpace V] (T : Tetrahedron V) : Prop :=
-- Assuming some function that checks if midpoints of opposite edges are perpendicular
sorry

def condition_c {V : Type} [EuclideanSpace V] (T : Tetrahedron V) : Prop :=
-- Assuming some function that checks if all face areas are equal
sorry

def condition_d {V : Type} [EuclideanSpace V] (T : Tetrahedron V) : Prop :=
-- Assuming some function that checks if the center of mass and the center of the inscribed sphere coincide
sorry

theorem equal_faces_iff_conditions {V : Type} [EuclideanSpace V] (T : Tetrahedron V) :
  all_faces_equal T ↔ (condition_a T ∨ condition_b T ∨ condition_c T ∨ condition_d T) :=
sorry

end equal_faces_iff_conditions_l801_801886


namespace handshaking_remainder_l801_801811

-- Define the number of people in the group
def num_people : ℕ := 12

-- Each person shakes hands with three different individuals
def handshake_degree : ℕ := 3

-- Define the graph interpretation of the scenario
def handshaking_arrangements := -- Implementation for computing handshaking arrangements goes here
  sorry

-- The main theorem to prove
theorem handshaking_remainder :
  (handshaking_arrangements % 1000) = 680 :=
sorry

end handshaking_remainder_l801_801811


namespace find_x_l801_801418

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (3, 5)
def vec_b (x : ℝ) : ℝ × ℝ := (1, x)

-- Define what it means for two vectors to be parallel
def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (a.1 = k * b.1) ∧ (a.2 = k * b.2)

-- Given condition: vectors a and b are parallel
theorem find_x (x : ℝ) (h : vectors_parallel vec_a (vec_b x)) : x = 5 / 3 :=
by
  sorry

end find_x_l801_801418


namespace interval_intersection_l801_801706

theorem interval_intersection (x : ℝ) : 
  (1 < 4 * x ∧ 4 * x < 3) ∧ (2 < 6 * x ∧ 6 * x < 4) ↔ (1 / 3 < x ∧ x < 2 / 3) := 
by 
  sorry

end interval_intersection_l801_801706


namespace centroid_tetrahedron_equalities_l801_801854

variables {A1 A2 A3 A4 G : ℝ} 

-- Assume G is the centroid of the tetrahedron A1A2A3A4
variable (hG : G = ∑ (i : fin 4), Ai / 4)

theorem centroid_tetrahedron_equalities
  (h1 : G = (A1 + A2 + A3 + A4) / 4)
  (dist_squared : ℝ → ℝ → ℝ)
  :
  4 * dist_squared G A1 + dist_squared A2 A3 + dist_squared A2 A4 + dist_squared A3 A4 =
  4 * dist_squared G A2 + dist_squared A1 A3 + dist_squared A1 A4 + dist_squared A3 A4 ∧
  4 * dist_squared G A3 + dist_squared A1 A2 + dist_squared A1 A4 + dist_squared A2 A4 ∧
  4 * dist_squared G A4 + dist_squared A1 A2 + dist_squared A1 A3 + dist_squared A2 A3 ∧
  (4 * dist_squared G A1 + dist_squared A2 A3 + dist_squared A2 A4 + dist_squared A3 A4 = 
   4 * dist_squared G A2 + dist_squared A1 A3 + dist_squared A1 A4 + dist_squared A3 A4 ∧
   4 * dist_squared G A3 + dist_squared A1 A2 + dist_squared A1 A4 + dist_squared A2 A4 ∧
   4 * dist_squared G A4 + dist_squared A1 A2 + dist_squared A1 A3 + dist_squared A2 A3 =
   4 * dist_squared G A4 + dist_squared A1 A2 + dist_squared A1 A3 + dist_squared A2 A3) ∧
   (4 * dist_squared G A4 + dist_squared A1 A2 + dist_squared A1 A3 + dist_squared A2 A3 = 
   (3/4) * ∑ (1 ≤ i < j ≤ 4), dist_squared Ai Aj)
 :=
begin
  sorry -- The proof is skipped
end

end centroid_tetrahedron_equalities_l801_801854


namespace arithmetic_sequence_S20_l801_801464

variable {a : ℕ → ℝ}
variable {a_6 a_9 a_12 a_15 : ℝ}
variable {S : ℕ → ℝ}

-- Define general properties of an arithmetic sequence
axiom arithmetic_seq (a: ℕ → ℝ) : ∃ (d : ℝ), ∀ (n : ℕ), a (n + 1) = a n + d

-- Conditions
axiom condition_1 : a 6 + a 9 + a 12 + a 15 = 20
axiom condition_2 : a 6 + a 15 = a 9 + a 12
axiom condition_3 : a 9 + a 12 = a 1 + a 20

-- Sum of the first n terms of an arithmetic sequence
def sum_arithmetic_seq (a: ℕ → ℝ) (n : ℕ) (d : ℝ) : ℝ := n * (2 * a 1 + (n - 1) * d) / 2

-- Definition of S_20
def S_20 := sum_arithmetic_seq a 20

theorem arithmetic_sequence_S20 : S 20 = 100 :=
by {
  -- Here would go the proof, but we are only defining the statement
  sorry
}

end arithmetic_sequence_S20_l801_801464


namespace tan_sum_pi_over_12_l801_801530

theorem tan_sum_pi_over_12 : 
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12)) = 4 := 
sorry

end tan_sum_pi_over_12_l801_801530


namespace a_2017_eq_2_l801_801413

noncomputable def a : ℕ → ℚ
| 0       := 2
| (n + 1) := (a n - 1) / (a n + 1)

theorem a_2017_eq_2 : a 2017 = 2 :=
sorry

end a_2017_eq_2_l801_801413


namespace train_length_is_correct_l801_801670

-- Define all the specific conditions as constants
def jogger_speed_kmph : ℝ := 9
def train_speed_kmph : ℝ := 45
def head_start_m : ℝ := 200
def time_sec : ℝ := 40
def answer : ℝ := 200

-- Convert speeds from km/hr to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ := 
  speed_kmph * (1000 / 3600)

-- Define jogger and train speed in m/s
def jogger_speed_mps : ℝ := kmph_to_mps jogger_speed_kmph
def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

-- Compute the relative speed
def relative_speed : ℝ := train_speed_mps - jogger_speed_mps

-- Compute the distance covered by the train in the given time
def distance_covered : ℝ := relative_speed * time_sec

-- Define the length of the train
def length_of_train : ℝ := distance_covered - head_start_m

-- Statement to prove that the length of the train is 200 meters
theorem train_length_is_correct : length_of_train = answer := by
  sorry

end train_length_is_correct_l801_801670


namespace three_connected_iff_sequence_exists_l801_801792

theorem three_connected_iff_sequence_exists (G : Graph) :
  (∃ seq : List Graph, seq.head = K_4 ∧ seq.last = G ∧
    (∀ i < seq.length - 1, ∃ e, Graph.remove_edge (seq[i+1]) e = seq[i]) ∧
    (∀ g ∈ seq, Graph.is_three_connected g)) ↔
  Graph.is_three_connected G := 
sorry

end three_connected_iff_sequence_exists_l801_801792


namespace range_of_f_exists_k_l801_801380

-- Define the quadratic function
def f (x a : ℝ) : ℝ := x^2 + a * x + 1

-- Statement 1: The range of f(x) on [-1, 1]
theorem range_of_f (a : ℝ) : 
  (a ≤ -2 → (∀ x, -1 ≤ x ∧ x ≤ 1 → f x a ∈ set.Icc a (-a))) ∧ 
  (0 < a ∧ a ≤ 2 → (∀ x, -1 ≤ x ∧ x ≤ 1 → f x a ∈ set.Icc (1 - a^2 / 4) (2 - a))) ∧ 
  (-2 < a ∧ a ≤ 0 → (∀ x, -1 ≤ x ∧ x ≤ 1 → f x a ∈ set.Icc (1 - a^2 / 4) (2 + a))) ∧ 
  (a > 2 → (∀ x, -1 ≤ x ∧ x ≤ 1 → f x a ∈ set.Icc (-a) a)) :=
sorry

-- Statement 2: Existence of an integer k such that |f(k)| ≤ 1/4
theorem exists_k (a b x1 x2 : ℝ) (h1 : f x1 a = 0) (h2 : f x2 a = 0)
  (x_non_int : (∃ (m : ℤ), (m : ℝ) < x1 ∧ x1 < m + 1 ∧ (m : ℝ) < x2 ∧ x2 < m + 1)) :
  ∃ k : ℤ, |f k a| ≤ 1 / 4 :=
sorry

end range_of_f_exists_k_l801_801380


namespace code_random_l801_801808

def code_range : String := "12345"
def code_rand : String := "1236"

-- We assume 'o' and 'm' are coded as 7 and 8 respectively:
def code_o : Nat := 7
def code_m : Nat := 8

theorem code_random :
  (Substr code_range 0 1 = "1") ∧
  (Substr code_range 1 2 = "2") ∧
  (Substr code_range 2 3 = "3") ∧
  (Substr code_range 3 4 = "4") ∧
  (Substr code_range 4 5 = "5") ∧
  (Substr code_rand 0 1 = "1") ∧
  (Substr code_rand 1 2 = "2") ∧
  (Substr code_rand 2 3 = "3") ∧
  (Substr code_rand 3 4 = "6") →
  "123678" = "1" ++ "2" ++ "3" ++ "6" ++ toString code_o ++ toString code_m :=
by
  -- Code omitted: proof construction not required
  sorry

end code_random_l801_801808


namespace john_anna_ebook_readers_l801_801483

-- Definitions based on conditions
def anna_bought : ℕ := 50
def john_buy_diff : ℕ := 15
def john_lost : ℕ := 3

-- Main statement
theorem john_anna_ebook_readers :
  let john_bought := anna_bought - john_buy_diff in
  let john_remaining := john_bought - john_lost in
  john_remaining + anna_bought = 82 :=
by
  sorry

end john_anna_ebook_readers_l801_801483


namespace second_most_expensive_sandwich_price_l801_801694

noncomputable def Andy_purchase_total : ℝ :=
  let soda := 1.50
  let hamburgers := 4 * 2.80
  let keychain := 2.25
  let plant := 5.77
  let food_tax := 0.07
  let non_food_tax := 0.095
  (soda * (1 + food_tax)) + (hamburgers * (1 + food_tax)) +
  (keychain * (1 + non_food_tax)) + (plant * (1 + non_food_tax))

noncomputable def Bob_sandwich_prices : List ℝ :=
  [2.90, 3.20, 2.55, 3.10, 2.85, 3.25]

noncomputable def Bob_discounted_price (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

noncomputable def Bob_purchase_total : ℝ :=
  let sorted_prices := Bob_sandwich_prices.qsort (· > ·)
  let expensive_discounts := [0.12, 0.12, 0.12]
  let cheap_prices := sorted_prices.drop 3
  let cheap_discounts := [0.08, 0.08, 0.08]
  let magazine_price := 4
  let magazine_tax := 0.055
  let food_tax := 0.06
  List.zipWith Bob_discounted_price expensive_discounts ++
  List.zipWith Bob_discounted_price cheap_prices cheap_discounts |>.sum * (1 + food_tax) +
  magazine_price * (1 + magazine_tax)

theorem second_most_expensive_sandwich_price :
  let sorted_prices := Bob_sandwich_prices.qsort (· > ·)
  let second_most_expensive_price := sorted_prices.tail.head? |>.getD 0
  let discounted_price := Bob_discounted_price second_most_expensive_price 0.12
  discounted_price = 2.816 := sorry

end second_most_expensive_sandwich_price_l801_801694


namespace John_Anna_total_eBooks_l801_801480

variables (Anna_bought : ℕ) (eBooks_lost_by_John : ℕ) 

def John_bought (Anna_bought : ℕ) : ℕ := Anna_bought - 15
def John_left (Anna_bought : ℕ) (eBooks_lost_by_John : ℕ) : ℕ := John_bought Anna_bought - eBooks_lost_by_John

theorem John_Anna_total_eBooks (Anna_bought_eq_50 : Anna_bought = 50)
    (John_bought_eq_35 : John_bought Anna_bought = 35) (eBooks_lost_eq_3 : eBooks_lost_by_John = 3) :
    (Anna_bought + John_left Anna_bought eBooks_lost_by_John = 82) :=
by sorry

end John_Anna_total_eBooks_l801_801480


namespace total_students_correct_l801_801516

-- Define the number of students who play football, cricket, both and neither.
def play_football : ℕ := 325
def play_cricket : ℕ := 175
def play_both : ℕ := 90
def play_neither : ℕ := 50

-- Define the total number of students
def total_students : ℕ := play_football + play_cricket - play_both + play_neither

-- Prove that the total number of students is 460 given the conditions
theorem total_students_correct : total_students = 460 := by
  sorry

end total_students_correct_l801_801516


namespace tan_pi_over_12_plus_tan_5pi_over_12_eq_4_l801_801528

-- Definitions for given conditions
def cos_pi_over_12 : ℝ := (Real.sqrt 6 + Real.sqrt 2) / 4
def cos_5pi_over_12 : ℝ := (Real.sqrt 6 - Real.sqrt 2) / 4

-- The theorem to be proved
theorem tan_pi_over_12_plus_tan_5pi_over_12_eq_4 : 
  Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 :=
by sorry

end tan_pi_over_12_plus_tan_5pi_over_12_eq_4_l801_801528


namespace ratio_of_combined_total_surface_area_of_cubes_l801_801091

open Nat

theorem ratio_of_combined_total_surface_area_of_cubes 
  (x : ℕ) 
  (SA_a : ℕ := 6 * (3 * x)^2)
  (SA_b : ℕ := 6 * x^2)
  (SA_c : ℕ := 6 * (2 * x)^2) :
  (SA_a + SA_b + SA_c) / gcd (gcd SA_a SA_b) SA_c = 84 * x^2 / gcd 54 6 24 := 
  9 * x^2 + 1 * x^2 + 4 * x^2 / gcd 54 6 24 := 
  -- simplified ratio in terms of coefficients
  sorry

end ratio_of_combined_total_surface_area_of_cubes_l801_801091


namespace john_anna_ebook_readers_l801_801482

-- Definitions based on conditions
def anna_bought : ℕ := 50
def john_buy_diff : ℕ := 15
def john_lost : ℕ := 3

-- Main statement
theorem john_anna_ebook_readers :
  let john_bought := anna_bought - john_buy_diff in
  let john_remaining := john_bought - john_lost in
  john_remaining + anna_bought = 82 :=
by
  sorry

end john_anna_ebook_readers_l801_801482


namespace book_collection_example_l801_801645

theorem book_collection_example :
  ∃ (P C B : ℕ), 
    (P : ℚ) / C = 3 / 2 ∧ 
    (C : ℚ) / B = 4 / 3 ∧ 
    P + C + B = 3002 ∧ 
    P + C + B > 3000 :=
by
  sorry

end book_collection_example_l801_801645


namespace min_distance_ellipse_line_l801_801177

theorem min_distance_ellipse_line : 
  let ellipse := { p : ℝ × ℝ | p.1^2 / 3 + p.2^2 = 1 }
  let line := { q : ℝ × ℝ | q.1 + q.2 - 4 = 0 }
  ∃ P Q : ℝ × ℝ, P ∈ ellipse ∧ Q ∈ line ∧ 
    let dist := λ (p1 p2 : ℝ × ℝ), real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
    ∃ min_dist : ℝ, min_dist = sqrt(2) ∧ 
      P = (3 / 2, 1 / 2) ∧ 
      ∀ (P' ∈ ellipse) (Q' ∈ line), dist P Q ≤ dist P' Q' :=
by
  sorry

end min_distance_ellipse_line_l801_801177


namespace A_share_in_profit_l801_801686

theorem A_share_in_profit (a_investment b_investment c_investment total_profit : ℕ)
    (h₁ : a_investment = 6300) (h₂ : b_investment = 4200) (h₃ : c_investment = 10500) (h₄ : total_profit = 13600) :
    (total_profit * a_investment) / (a_investment + b_investment + c_investment) = 4080 :=
by
  -- Definitions and assumptions
  have total_investment := a_investment + b_investment + c_investment
  have a_ratio := a_investment * total_profit / total_investment
  show a_ratio = 4080
  sorry

end A_share_in_profit_l801_801686


namespace correct_total_annual_salary_expression_l801_801660

def initial_workers : ℕ := 8
def initial_salary : ℝ := 1.0 -- in ten thousand yuan
def new_workers : ℕ := 3
def new_worker_initial_salary : ℝ := 0.8 -- in ten thousand yuan
def salary_increase_rate : ℝ := 1.2 -- 20% increase each year

def total_annual_salary (n : ℕ) : ℝ :=
  (3 * n + 5) * salary_increase_rate^n + (new_workers * new_worker_initial_salary)

theorem correct_total_annual_salary_expression (n : ℕ) :
  total_annual_salary n = (3 * n + 5) * 1.2^n + 2.4 := 
by
  sorry

end correct_total_annual_salary_expression_l801_801660


namespace max_marbles_rolled_l801_801365

def points_per_hole : list ℕ := [1, 2, 3, 4, 5]
def max_score : ℕ := 23
def max_marbles : ℕ := 30  -- Since it's up to 10 marbles of each size, max 10 small + 10 medium + 10 large

-- Marble size definition
inductive MarbleSize
| small
| medium
| large

-- Function to determine valid holes for each marble size
def valid_holes (size : MarbleSize) : list ℕ :=
  match size with
  | MarbleSize.small => [1, 2, 3, 4, 5]
  | MarbleSize.medium => [3, 4, 5]
  | MarbleSize.large => [5]

-- The theorem to prove the maximum number of marbles rolled for a score of 23
theorem max_marbles_rolled (score : ℕ) (num_small num_medium num_large : ℕ)
  (score = max_score)
  (h1 : score = num_small * 1 + num_medium * 3 + num_large * 5)
  (h2 : num_small ≤ 10)
  (h3 : num_medium ≤ 10)
  (h4 : num_large ≤ 10)
  : num_small + num_medium + num_large ≤ 14 := by
  sorry

end max_marbles_rolled_l801_801365


namespace find_common_difference_l801_801608

def arithmetic_sequence (S_n : ℕ → ℝ) (d : ℝ) :=
  ∀ n, S_n n = (n / 2) * (2 * (S_n 1 / 1) + (n - 1) * d)

theorem find_common_difference (S_n : ℕ → ℝ) (d : ℝ) (h : ∀n, S_n n = (n / 2) * (2 * (S_n 1 / 1) + (n - 1) * d)) 
    (h_condition : S_n 3 / 3 - S_n 2 / 2 = 1) :
  d = 2 :=
sorry

end find_common_difference_l801_801608


namespace problem_f_of_f_one_fourth_l801_801057

def f : ℝ → ℝ :=
  λ x,
    if x > 0 then Real.log x / Real.log 2 -- log base 2
    else 3 ^ x

theorem problem_f_of_f_one_fourth : f (f (1 / 4)) = 1 / 9 :=
by
  -- Begin proof construction
  sorry

end problem_f_of_f_one_fourth_l801_801057


namespace solution_set_equality_of_abs_x_minus_2_gt_1_and_quadratic_gt_0_l801_801044

variable (a b : ℝ)

theorem solution_set_equality_of_abs_x_minus_2_gt_1_and_quadratic_gt_0 :
  (∀ x : ℝ, (|x - 2| > 1 ↔ x^2 + a * x + b > 0)) → a + b = -1 :=
by
  sorry

end solution_set_equality_of_abs_x_minus_2_gt_1_and_quadratic_gt_0_l801_801044


namespace min_value_sin_cos_l801_801598

theorem min_value_sin_cos (x : ℝ) (h : x ≥ -π / 6 ∧ x ≤ π / 2) :
  (sin x + 1) * (cos x + 1) ≥ (2 + sqrt 3) / 4 :=
sorry

end min_value_sin_cos_l801_801598


namespace mirror_area_l801_801720

/-- The outer dimensions of the frame are given as 100 cm by 140 cm,
and the frame width is 15 cm. We aim to prove that the area of the mirror
inside the frame is 7700 cm². -/
theorem mirror_area (W H F: ℕ) (hW : W = 100) (hH : H = 140) (hF : F = 15) :
  (W - 2 * F) * (H - 2 * F) = 7700 :=
by
  sorry

end mirror_area_l801_801720


namespace sousliks_problem_l801_801450

variable (V : Type) [Fintype V] [DecidableEq V]

def souslik_graph (G : SimpleGraph V) :=
  ∃ (friends : Finset V) (souslik_7 : Finset V), 
    souslik_7.card = 7 ∧ 
    (∀ (v : V), v ∈ souslik_7 → (G.degree v = 4)) ∧
    (∀ (u v : V), u ∈ friends → v ∈ friends → G.Adj u v) ∧
    ∃ T S : Finset V, T.card = 3 ∧ S.card = 3 ∧
    T ∩ S = ∅ ∧ 
    T ⊆ friends ∧
    S ⊆ friends ∧
    (∀ (u v : V), u ∈ T → v ∈ T → G.Adj u v) ∧
    (∀ (u v : V), u ∈ S → v ∈ S → G.Adj u v)

theorem sousliks_problem (G : SimpleGraph V) : souslik_graph G :=
sorry

end sousliks_problem_l801_801450


namespace hyperbola_standard_equation_equation_of_line_L_l801_801024

open Real

noncomputable def hyperbola (x y : ℝ) : Prop :=
  y^2 - x^2 / 3 = 1

noncomputable def focus_on_y_axis := ∃ c : ℝ, c = 2

noncomputable def asymptote (x y : ℝ) : Prop := 
  y = sqrt 3 / 3 * x ∨ y = - sqrt 3 / 3 * x

noncomputable def point_A := (1, 1 / 2)

noncomputable def line_L (x y : ℝ) : Prop :=
  4 * x - 6 * y - 1 = 0

theorem hyperbola_standard_equation :
  ∃ (x y: ℝ), hyperbola x y :=
sorry

theorem equation_of_line_L :
  ∀ (x y : ℝ), point_A = (1, 1 / 2) ∧ line_L x y :=
sorry

end hyperbola_standard_equation_equation_of_line_L_l801_801024


namespace relay_race_arrangements_l801_801176

noncomputable def number_of_arrangements (athletes : Finset ℕ) (a b : ℕ) : ℕ :=
  (athletes.erase a).card.factorial * ((athletes.erase b).card.factorial - 2) * (athletes.card.factorial / ((athletes.card - 4).factorial)) / 4

theorem relay_race_arrangements :
  let athletes := {0, 1, 2, 3, 4, 5}
  number_of_arrangements athletes 0 1 = 252 := 
by
  sorry

end relay_race_arrangements_l801_801176


namespace oranges_per_box_l801_801668

theorem oranges_per_box (total_oranges : ℝ) (total_boxes : ℝ) (h1 : total_oranges = 26500) (h2 : total_boxes = 2650) : 
  total_oranges / total_boxes = 10 :=
by 
  sorry

end oranges_per_box_l801_801668


namespace tan_sum_simplification_l801_801557
-- We start by importing the relevant Lean libraries that contain trigonometric functions and basic real analysis.

-- Define the statement to be proved in Lean.
theorem tan_sum_simplification :
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12) = 4 * Real.sqrt 2 - 4) :=
by
  sorry

end tan_sum_simplification_l801_801557


namespace find_dimes_l801_801256

-- Definitions for the conditions
def total_dollars : ℕ := 13
def dollar_bills_1 : ℕ := 2
def dollar_bills_5 : ℕ := 1
def quarters : ℕ := 13
def nickels : ℕ := 8
def pennies : ℕ := 35
def value_dollar_bill_1 : ℝ := 1.0
def value_dollar_bill_5 : ℝ := 5.0
def value_quarter : ℝ := 0.25
def value_nickel : ℝ := 0.05
def value_penny : ℝ := 0.01
def value_dime : ℝ := 0.10

-- Theorem statement
theorem find_dimes (total_dollars dollar_bills_1 dollar_bills_5 quarters nickels pennies : ℕ)
  (value_dollar_bill_1 value_dollar_bill_5 value_quarter value_nickel value_penny value_dime : ℝ) :
  (2 * value_dollar_bill_1 + 1 * value_dollar_bill_5 + 13 * value_quarter + 8 * value_nickel + 35 * value_penny) + 
  (20 * value_dime) = ↑total_dollars :=
sorry

end find_dimes_l801_801256


namespace product_mean_median_l801_801442

def s := {8, 16, 24, 32, 40, 48}

noncomputable def mean (s : Set ℕ) : ℕ :=
  s.to_list.sum / s.size

noncomputable def median (s : Set ℕ) : ℕ :=
  let sorted_list := s.to_list.qsort (· ≤ ·)
  if h : 2 ∣ s.size then
    (sorted_list.get ⟨s.size / 2 - 1, sorry⟩ + sorted_list.get ⟨s.size / 2, sorry⟩) / 2
  else
    sorted_list.get ⟨s.size / 2, sorry⟩

theorem product_mean_median (hs : s = {8, 16, 24, 32, 40, 48}) :
  (mean s) * (median s) = 784 := by
  sorry

end product_mean_median_l801_801442


namespace expected_value_is_correct_l801_801975

def probability_of_rolling_one : ℚ := 1 / 4

def probability_of_other_numbers : ℚ := 3 / 4 / 5

def win_amount : ℚ := 8

def loss_amount : ℚ := -3

def expected_value : ℚ := (probability_of_rolling_one * win_amount) + 
                          (probability_of_other_numbers * 5 * loss_amount)

theorem expected_value_is_correct : expected_value = -0.25 :=
by 
  unfold expected_value probability_of_rolling_one probability_of_other_numbers win_amount loss_amount
  sorry

end expected_value_is_correct_l801_801975


namespace finite_discontinuities_not_satisfy_condition1_l801_801157

noncomputable def has_finitely_many_discontinuities (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∃ S : set ℝ, S ⊆ set.Ioo a b ∧ S.finite ∧ ∀ x ∈ set.Ioo a b \ S, continuous_at f x

theorem finite_discontinuities_not_satisfy_condition1 (f : ℝ → ℝ) (a b : ℝ)
    (h : has_finitely_many_discontinuities f a b) : ¬ ( ∀ ε > 0, ∃ δ > 0, ∀ x y ∈ set.Ioo a b, abs (x - y) < δ → abs (f x - f y) < ε ) :=
sorry

end finite_discontinuities_not_satisfy_condition1_l801_801157


namespace simplify_tangent_sum_l801_801537

theorem simplify_tangent_sum :
  tan (Real.pi / 12) + tan (5 * Real.pi / 12) = Real.sqrt 6 - Real.sqrt 2 := 
sorry

end simplify_tangent_sum_l801_801537


namespace minimize_sum_AP_BP_l801_801616

def point := (ℝ × ℝ)

def A : point := (-1, 0)
def B : point := (1, 0)
def center : point := (3, 4)
def radius : ℝ := 2

def on_circle (P : point) : Prop := (P.1 - center.1)^2 + (P.2 - center.2)^2 = radius^2

def AP_squared (P : point) : ℝ := (P.1 - A.1)^2 + (P.2 - A.2)^2
def BP_squared (P : point) : ℝ := (P.1 - B.1)^2 + (P.2 - B.2)^2
def sum_AP_BP_squared (P : point) : ℝ := AP_squared P + BP_squared P

theorem minimize_sum_AP_BP :
  ∀ P : point, on_circle P → sum_AP_BP_squared P = AP_squared (9/5, 12/5) + BP_squared (9/5, 12/5) → 
  P = (9/5, 12/5) :=
sorry

end minimize_sum_AP_BP_l801_801616


namespace equation_of_tangent_line_maximum_value_of_h_l801_801058

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := (1 / 3) * x^3 + (1 / 2) * x^2 - x + 1 / 6
noncomputable def h (x : ℝ) : ℝ := f x - (g' x)

theorem equation_of_tangent_line : ∃ m n : ℝ, (tangent_line : ℝ → ℝ) := 
  (λ x, x - 1) ∧ g = (λ x, (1 / 3) * x ^ 3 + (1 / 2) * x ^ 2 - x + 1 / 6) := sorry

theorem maximum_value_of_h : ∃ (x : ℝ), x = 1 / 2 ∧ h x = Real.log (1 / 2) + 1 / 4 := sorry

end equation_of_tangent_line_maximum_value_of_h_l801_801058


namespace count_odd_S_eq_722_count_even_S_eq_816_abs_diff_odd_even_S_eq_94_l801_801371

def tau (n : ℕ) : ℕ := set.to_finset (set_of (λ d, d ∣ n)).card

def S (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), tau i

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

noncomputable def count_perfect_squares (n : ℕ) : ℕ :=
  nat.sqrt n

theorem count_odd_S_eq_722 : 
  (finset.range 1501).filter (λ n, is_perfect_square n ∧ n % 2 = 1).card = 722 := sorry

theorem count_even_S_eq_816 :
  (finset.range 1501).filter (λ n, is_perfect_square n ∧ n % 2 = 0).card = 816 := sorry

theorem abs_diff_odd_even_S_eq_94 :
  |((finset.range 1501).filter (λ n, S n % 2 = 1).card - 
   (finset.range 1501).filter (λ n, S n % 2 = 0).card)| = 94 := by
  have odd_count := count_odd_S_eq_722
  have even_count := count_even_S_eq_816
  exact (odd_count, even_count, sorry)

end count_odd_S_eq_722_count_even_S_eq_816_abs_diff_odd_even_S_eq_94_l801_801371


namespace angle_measure_is_ninety_l801_801182

noncomputable def measure_angle_AQH (A B C D E F G H Q : Type) [RegularOctagon A B C D E F G H] [ExtendedMeet A B G H Q] : ℝ :=
  90

-- Formalizing the statement of the problem in Lean
theorem angle_measure_is_ninety
  (A B C D E F G H Q : Type)
  [RegularOctagon A B C D E F G H]
  [ExtendedMeet A B G H Q] :
  measure_angle_AQH A B C D E F G H Q = 90 := 
sorry

end angle_measure_is_ninety_l801_801182


namespace find_n_l801_801343

theorem find_n (n : ℕ) (h : 2^6 * 3^3 * n = Nat.factorial 10) : n = 2100 :=
by
sorry

end find_n_l801_801343


namespace number_of_valid_ndigit_numbers_l801_801761

theorem number_of_valid_ndigit_numbers (n : ℕ) : 
  let S := (fin (3^n)),
      A_1 := (fin (2^n)),
      A_2 := (fin (2^n)),
      A_3 := (fin (2^n)),
      A_1_A_2 := 1,
      A_1_A_3 := 1,
      A_2_A_3 := 1,
      A_1_A_2_A_3 := 0
   in card S - (card A_1 + card A_2 + card A_3) + (A_1_A_2 + A_1_A_3 + A_2_A_3) - A_1_A_2_A_3 
   = 3^n - 3 * 2^n + 3 :=
begin
  /- The proof steps would go here -/
  sorry
end

end number_of_valid_ndigit_numbers_l801_801761


namespace sqrt_sub_cos_add_one_minus_sqrt_sq_is_three_plus_three_hlf_sqrt_two_l801_801700

noncomputable def sqrt (x : ℝ) := Real.sqrt x
noncomputable def cos_pi_over_4 := Real.cos (Math.pi / 4)
noncomputable def one_minus_sqrt_2_squared := (1 - Real.sqrt 2)^2

theorem sqrt_sub_cos_add_one_minus_sqrt_sq_is_three_plus_three_hlf_sqrt_two :
  sqrt 32 - cos_pi_over_4 + one_minus_sqrt_2_squared = 3 + 3 / 2 * Real.sqrt 2 :=
by
  sorry

end sqrt_sub_cos_add_one_minus_sqrt_sq_is_three_plus_three_hlf_sqrt_two_l801_801700


namespace combined_experience_l801_801834

theorem combined_experience : 
  ∀ (James John Mike : ℕ), 
  (James = 20) → 
  (∀ (years_ago : ℕ), (years_ago = 8) → (John = 2 * (James - years_ago) + years_ago)) → 
  (∀ (started : ℕ), (John - started = 16) → (Mike = 16)) → 
  James + John + Mike = 68 :=
begin
  intros James John Mike HJames HJohn HMike,
  rw HJames,
  have HJohn8 : John = 32, {
    rw HJohn,
    intros years_ago Hyears_ago,
    rw Hyears_ago,
    norm_num,
  },
  rw HJohn8 at HMike,
  norm_num at HMike,
  rw HJohn8,
  rw HMike,
  norm_num,
end

end combined_experience_l801_801834


namespace log_a1_13_l801_801118

variable (a : ℕ → ℝ)

-- Conditions
def a_9 := a 9 = 13 
def a_13 := a 13 = 1

-- Toplevel statement (goal)
theorem log_a1_13 :
  (∃ a : ℝ, (a 9 = 13 ∧ a 13 = 1)) → (log (a 1) 13 = 1 / 3) :=
sorry

end log_a1_13_l801_801118


namespace cube_root_of_a_minus_m_l801_801094

theorem cube_root_of_a_minus_m (m a : ℝ)
  (h1 : a > 0)
  (h2 : (m + 7) * (m + 7) = a)
  (h3 : (2 * m - 1) * (2 * m - 1) = a) :
  real.cbrt (a - m) = 3 :=
sorry

end cube_root_of_a_minus_m_l801_801094


namespace packs_of_yellow_balls_l801_801486

theorem packs_of_yellow_balls (Y : ℕ) : 
  3 * 19 + Y * 19 + 8 * 19 = 399 → Y = 10 :=
by sorry

end packs_of_yellow_balls_l801_801486


namespace largest_change_to_nine_l801_801050

theorem largest_change_to_nine (d : ℕ) (h1 : d = 27693) :
  (∀ m : ℕ, m = nat_of_digits 10 [9, 9, 6, 9, 3] →
              ∀ n : ℕ, n ∈ {nat_of_digits 10 [2, 9, 9, 9, 3], 
                            nat_of_digits 10 [2, 7, 9, 9, 9], 
                            nat_of_digits 10 [2, 7, 6, 9, 9]} →
              m > n) := by
  sorry

end largest_change_to_nine_l801_801050


namespace magic_square_solution_l801_801109

-- Definitions for the entries in the magic square
variables (x d e f g h S : ℤ)

-- The conditions given in the problem
def condition1 : Prop := x + 3 + f = S
def condition2 : Prop := 50 + d + f = S
def condition3 : Prop := x + (x - 47) + h = S
def condition4 : Prop := 50 + e + h = S
def condition5 : Prop := x + 21 + 50 = S
def condition6 : Prop := 3 + d + (2 * x - 97) = S

-- The theorem to prove
theorem magic_square_solution : 
  condition1 x S f → condition2 x d S f → condition3 x S h → condition4 x S e h → condition5 x S → condition6 x d S → x = 106 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end magic_square_solution_l801_801109


namespace correct_propositions_l801_801033

variable {α : Type*} [InnerProductSpace ℝ α]

variables (a b c : α)

-- Proposition A
def prop_A := (abs ⟪a, b⟫ = ∥a∥ * ∥b∥) ↔ (∃ k : ℝ, k ≠ 0 ∧ a = k • b)

-- Proposition B
def prop_B := (∃ k : ℝ, k < 0 ∧ a = k • b) ↔ (⟪a, b⟫ = -∥a∥ * ∥b∥)

-- Proposition C
def prop_C := (∥a + b∥ = ∥a - b∥) ↔ (⟪a, b⟫ = 0)

-- Proposition D
def prop_D := (∥a∥ = ∥b∥) ↔ (abs ⟪a, c⟫ = abs ⟪b, c⟫)

-- The main theorem to evaluate the propositions
theorem correct_propositions : prop_A a b ∧ prop_B a b ∧ prop_C a b ∧ ¬prop_D a b :=
by
  sorry

end correct_propositions_l801_801033


namespace find_position_vector_l801_801495

noncomputable def ratio := 3 / (3 + 4)

theorem find_position_vector (C D : ℝ^3) (Q : ℝ^3)
  (h1 : Q = (4 / (3 + 4)) • C + (3 / (3 + 4)) • D) :
  Q = (4 / 7) • C + (3 / 7) • D :=
sorry

end find_position_vector_l801_801495


namespace sum_of_squared_distances_eq_1152_l801_801338

theorem sum_of_squared_distances_eq_1152 :
  let ABC := {A B C : Point} -- Assume points for an equilateral triangle
  let s : Real := sqrt 144 -- Side length of the equilateral triangle
  let r : Real := sqrt 12 -- Distance from B to D1 and D2
  let D1 D2 : Point -- Points D1 and D2
  -- Each of the four triangles AD1E1, AD1E2, AD2E3, AD2E4 is congruent to ABC
  let t1 t2 t3 t4 : Triangle
  -- Distinct points for each triangle
  let E1 E2 E3 E4 : Point
  -- Distances BD1 and BD2 are given as sqrt 12
  in BD1 = r ∧ BD2 = r ∧
  -- Sum of squared distances from C to each E_k
  (CE1^2 + CE2^2 + CE3^2 + CE4^2 = 1152) :=
by
  sorry

end sum_of_squared_distances_eq_1152_l801_801338


namespace handshaking_pairs_l801_801456

-- Definition of the problem: Given 8 people, pair them up uniquely and count the ways modulo 1000
theorem handshaking_pairs (N : ℕ) (H : N=105) : (N % 1000) = 105 :=
by {
  -- The proof is omitted.
  sorry
}

end handshaking_pairs_l801_801456


namespace sum_x1_x2_range_l801_801850

variable {x₁ x₂ : ℝ}

-- Definition of x₁ being the real root of the equation x * 2^x = 1
def is_root_1 (x : ℝ) : Prop :=
  x * 2^x = 1

-- Definition of x₂ being the real root of the equation x * log_2 x = 1
def is_root_2 (x : ℝ) : Prop :=
  x * Real.log x / Real.log 2 = 1

theorem sum_x1_x2_range (hx₁ : is_root_1 x₁) (hx₂ : is_root_2 x₂) :
  2 < x₁ + x₂ :=
sorry

end sum_x1_x2_range_l801_801850


namespace num_license_plates_l801_801994

theorem num_license_plates : ∃ n : ℕ, n = 5 * 10^4 * 26^3 ∧ n = 878800000 :=
by
  use 5 * 10^4 * 26^3
  split
  case h_left =>
    exact rfl
  case h_right =>
    norm_num  -- This will simplify and check the calculation

end num_license_plates_l801_801994


namespace sum_of_interior_angles_of_polygon_l801_801493

theorem sum_of_interior_angles_of_polygon
  (Q : Type) (n : ℕ) (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_sides : n = 80)
  (h_inter_ext_relation : ∀ k : ℕ, k < n → a k = 8 * b k)
  (h_sum_exterior : ∑ k in Finset.range n, b k = 360) :
  ∑ k in Finset.range n, a k = 2880 :=
by
  sorry

end sum_of_interior_angles_of_polygon_l801_801493


namespace bisection_method_root_nearest_tenth_l801_801626

noncomputable def f : ℝ → ℝ := sorry -- Assume f is some noncomputable function

theorem bisection_method_root_nearest_tenth :
  f 0.64 < 0 ∧ f 0.68 < 0 ∧ f 0.72 > 0 → ∃ x ∈ set.Ioo 0.68 0.72, |x - 0.7| ≤ 0.05 ∧ f x = 0 :=
sorry

end bisection_method_root_nearest_tenth_l801_801626


namespace scissors_total_l801_801615

theorem scissors_total (initial_scissors : ℕ) (additional_scissors : ℕ) (h1 : initial_scissors = 54) (h2 : additional_scissors = 22) : 
  initial_scissors + additional_scissors = 76 :=
by
  sorry

end scissors_total_l801_801615


namespace vector_decomposition_l801_801272

theorem vector_decomposition 
  (x p q r : ℝ × ℝ × ℝ) 
  (α β γ : ℝ) 
  (hx : x = (23, -14, -30)) 
  (hp : p = (2, 1, 0)) 
  (hq : q = (1, -1, 0)) 
  (hr : r = (-3, 2, 5)) 
  (hγ : γ = -6) 
  (hβ : β = 3) 
  (hα : α = 1) 
  : x = α • p + β • q + γ • r := by
srry

end vector_decomposition_l801_801272


namespace sequence_term_l801_801412

theorem sequence_term (n : ℕ) (h₁ : 0 < n) : (3 * Real.sqrt 5 = Real.sqrt (2 * n - 1)) → (n = 23) :=
by 
  intros h2,
  sorry

end sequence_term_l801_801412


namespace product_a_n_equals_30603_over_100_factorial_l801_801369

def a_n (n : ℕ) (hn : n ≥ 5) : ℚ :=
  (n^2 + 3*n + 4) / (n^3 - 1)

theorem product_a_n_equals_30603_over_100_factorial :
  (∏ n in (finset.range 96).filter (λ n, n + 5 ≥ 5), a_n (n + 5) (by linarith)) = 30603 / 100! :=
by sorry

end product_a_n_equals_30603_over_100_factorial_l801_801369


namespace stamp_solutions_l801_801737

theorem stamp_solutions (n : ℕ) (h1 : ∀ (k : ℕ), k < 115 → ∃ (a b c : ℕ), 
  3 * a + n * b + (n + 1) * c = k) 
  (h2 : ¬ ∃ (a b c : ℕ), 3 * a + n * b + (n + 1) * c = 115) 
  (h3 : ∀ (k : ℕ), 116 ≤ k ∧ k ≤ 120 → ∃ (a b c : ℕ), 
  3 * a + n * b + (n + 1) * c = k) : 
  n = 59 :=
sorry

end stamp_solutions_l801_801737


namespace domain_tan_2x_plus_pi_over_3_l801_801857

noncomputable def domain_of_tangent (x : ℝ) : Prop :=
  ∀ k : ℤ, x ≠ (π / 12) + (k * π / 2)

theorem domain_tan_2x_plus_pi_over_3 :
  ∀ x : ℝ, domain_of_tangent x :=
by
  intro x
  sorry

end domain_tan_2x_plus_pi_over_3_l801_801857


namespace problem_AE_length_l801_801113

theorem problem_AE_length (BC CD DE : ℝ) (B_angle C_angle D_angle A_angle : ℝ) :
  BC = 3 → CD = 3 → DE = 3 → B_angle = 135 → C_angle = 120 → D_angle = 120 → A_angle = 90 →
  (∃ (AE_length : ℝ) (a b : ℕ), 
      AE_length = a + 3 * real.sqrt b ∧ a + b = 6) := 
by
  -- Given conditions
  intros BC_eq_three CD_eq_three DE_eq_three B_angle_135 
         C_angle_120 D_angle_120 A_angle_90,
  -- Proof steps and calculations would be filled here.
  sorry

end problem_AE_length_l801_801113


namespace ellipse_equation_line_intersection_fixed_point_l801_801767

-- Definitions for the problem
def ellipse_eq (x y : ℝ) (b : ℝ) : Prop :=
  x^2 / 4 + y^2 / b^2 = 1

def focus_left (b : ℝ) : ℝ × ℝ :=
  (-Real.sqrt (4 - b^2), 0)

def focus_right (b : ℝ) : ℝ × ℝ :=
  (Real.sqrt (4 - b^2), 0)

def max_pf1_pf2 (x y : ℝ) (b : ℝ) : Prop :=
  let _: ℝ × ℝ := focus_left b in
  let _: ℝ × ℝ := focus_right b in
  let pf1 := (-Real.sqrt (4 - b^2) - x, -y)
  let pf2 := (Real.sqrt (4 - b^2) - x, -y)
  pf1.1 * pf2.1 + pf1.2 * pf2.2 = 1

-- Statement for the first proof
theorem ellipse_equation (b : ℝ) :
  ellipse_eq x y b → max_pf1_pf2 x y b → b^2 = 1 :=
sorry

def line_eq (k : ℝ) : ℝ → ℝ := λ y, k * y - 1

def line_intersects_ellipse (k : ℝ) (b : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), line_eq k y₁ = x₁ ∧ line_eq k y₂ = x₂ ∧ ellipse_eq x₁ y₁ b ∧ ellipse_eq x₂ y₂ b :=
sorry

-- Statement for the second proof
theorem line_intersection_fixed_point (k b : ℝ) :
  b^2 = 1 →
  line_intersects_ellipse k b →
  ∀ x₁ y₁ x₂ y₂, 
    (y₁ = -y₂ ∧ line_eq k y₁ = x₁ ∧ line_eq k y₂ = x₂) → 
    let A := (x₁, y₁) in
    let B := (x₂, y₂) in
    let A' := (x₁, -y₁) in
    let A'B_line := (A'₁, B₂) in
    A'B_line.x = -4 :=
sorry

end ellipse_equation_line_intersection_fixed_point_l801_801767


namespace tan_pi_over_12_plus_tan_5pi_over_12_eq_4_l801_801525

-- Definitions for given conditions
def cos_pi_over_12 : ℝ := (Real.sqrt 6 + Real.sqrt 2) / 4
def cos_5pi_over_12 : ℝ := (Real.sqrt 6 - Real.sqrt 2) / 4

-- The theorem to be proved
theorem tan_pi_over_12_plus_tan_5pi_over_12_eq_4 : 
  Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 :=
by sorry

end tan_pi_over_12_plus_tan_5pi_over_12_eq_4_l801_801525


namespace solve_equation_l801_801188

variable {x : ℝ}

theorem solve_equation (h : (x^2 + 6 * x - 6)^2 + 6 * (x^2 + 6 * x - 6) - 6 = x) :
  x = -6 ∨ x = 1 ∨ x = (-7 + 3 * Real.sqrt(5)) / 2 ∨ x = (-7 - 3 * Real.sqrt(5)) / 2 :=
by
  sorry

end solve_equation_l801_801188


namespace base_6_addition_l801_801594

-- Definitions of base conversion and addition
def base_6_to_nat (n : ℕ) : ℕ :=
  n.div 100 * 36 + n.div 10 % 10 * 6 + n % 10

def nat_to_base_6 (n : ℕ) : ℕ :=
  let a := n.div 216
  let b := (n % 216).div 36
  let c := ((n % 216) % 36).div 6
  let d := n % 6
  a * 1000 + b * 100 + c * 10 + d

-- Conversion from base 6 to base 10 for the given numbers
def nat_256 := base_6_to_nat 256
def nat_130 := base_6_to_nat 130

-- The final theorem to prove
theorem base_6_addition : nat_to_base_6 (nat_256 + nat_130) = 1042 :=
by
  -- Proof omitted since it is not required
  sorry

end base_6_addition_l801_801594


namespace sock_pairs_l801_801453

open Nat

theorem sock_pairs (r g y : ℕ) (hr : r = 5) (hg : g = 6) (hy : y = 4) :
  (choose r 2) + (choose g 2) + (choose y 2) = 31 :=
by
  rw [hr, hg, hy]
  norm_num
  sorry

end sock_pairs_l801_801453


namespace intersection_of_A_and_B_l801_801490

theorem intersection_of_A_and_B {x : ℝ} :
  (x - 1 < 0) ∧ (log x / log 2 < 0) ↔ (0 < x ∧ x < 1) := by
  sorry

end intersection_of_A_and_B_l801_801490


namespace sum_of_remainders_mod_11_l801_801631

theorem sum_of_remainders_mod_11
    (a b c d : ℤ)
    (h₁ : a % 11 = 2)
    (h₂ : b % 11 = 4)
    (h₃ : c % 11 = 6)
    (h₄ : d % 11 = 8) :
    (a + b + c + d) % 11 = 9 :=
by
  sorry

end sum_of_remainders_mod_11_l801_801631


namespace number_of_ways_to_place_balls_into_boxes_l801_801425

theorem number_of_ways_to_place_balls_into_boxes : 
  (number_of_ways (balls : ℕ) (boxes : ℕ) : ℕ) where balls = 5 ∧ boxes = 4 = 4^5 :=
begin
  sorry
end

end number_of_ways_to_place_balls_into_boxes_l801_801425


namespace sum_ratio_l801_801847

variables (a : ℕ → ℚ) (S : ℕ → ℚ)

-- Assume {a_n} is an arithmetic sequence and S_n is the sum of the first n terms
axiom arithmetic_sequence (a : ℕ → ℚ) : ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d
axiom sum_arithmetic_sequence (S a : ℕ → ℚ) : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

-- Given condition
axiom ratio_condition : a 8 / a 7 = 13 / 5

theorem sum_ratio (a : ℕ → ℚ) (S : ℕ → ℚ) 
  [arithmetic_sequence a] [sum_arithmetic_sequence S a] : S 15 / S 13 = 3 :=
sorry

end sum_ratio_l801_801847


namespace determine_common_ratio_l801_801146

-- Definition of geometric sequence and sum of first n terms
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def sum_geometric_sequence (a : ℕ → ℝ) : ℕ → ℝ
  | 0       => a 0
  | (n + 1) => a (n + 1) + sum_geometric_sequence a n

-- Main theorem
theorem determine_common_ratio (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : is_geometric_sequence a q)
  (h3 : ∀ n, S n = sum_geometric_sequence a n)
  (h4 : 3 * (S 2 + a 2 + a 1 * q^2) = 8 * a 1 * q + 5 * a 1) :
  q = 2 :=
by 
  sorry

end determine_common_ratio_l801_801146


namespace digits_in_8_pow_12_times_5_pow_18_l801_801733

noncomputable def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else (Real.log10 n).ceil.toNat

theorem digits_in_8_pow_12_times_5_pow_18 :
  num_digits (8^12 * 5^18) = 24 := by
  sorry

end digits_in_8_pow_12_times_5_pow_18_l801_801733


namespace max_value_l801_801731

theorem max_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 9 * a^2 + 4 * b^2 + c^2 = 91) :
  a + 2 * b + 3 * c ≤ 30.333 :=
by
  sorry

end max_value_l801_801731


namespace collinear_points_l801_801664

noncomputable def point : Type := sorry -- Assuming that point is a type in the geometric context
noncomputable def isQuadrilateral (A B C D : point) : Prop := sorry -- Definition of a convex quadrilateral
noncomputable def isAcuteAngle (A B C : point) : Prop := sorry -- Definition of acute angle

noncomputable def circumcenter (A B C : point) : point := sorry -- Definition of the circumcenter of a triangle
noncomputable def areCollinear (A B C : point) : Prop := sorry -- Definition of collinearity

theorem collinear_points
  (A B C D E F O1 O2 : point)
  (h1 : isQuadrilateral A B C D)
  (h2 : ∠ B = ∠ D)
  (h3 : isAcuteAngle A B D)
  (h4 : lies_on E A B)
  (h5 : lies_on F A D)
  (h6 : distance C B = distance C E)
  (h7 : distance C F = distance C D)
  (h8 : O1 = circumcenter C E F)
  (h9 : O2 = circumcenter A B D) :
  areCollinear C O1 O2 :=
sorry

end collinear_points_l801_801664


namespace sphere_surface_area_l801_801820

theorem sphere_surface_area
  (a : ℝ)
  (expansion : (1 - 2 * 1 : ℝ)^6 = a)
  (a_value : a = 1) :
  4 * Real.pi * ((Real.sqrt (2^2 + 3^2 + a^2) / 2)^2) = 14 * Real.pi :=
by
  sorry

end sphere_surface_area_l801_801820


namespace find_tangent_line_l801_801916

-- Define the function
def f (x : ℝ) : ℝ := (Real.exp x) / (x + 1)

-- Define the point of tangency
def P : ℝ × ℝ := (1, Real.exp 1 / 2)

-- Define the tangent line equation
def tangent_line (x : ℝ) : ℝ := (Real.exp 1 / 4) * x + (Real.exp 1 / 4)

-- The main theorem stating the equation of the tangent line
theorem find_tangent_line :
  ∀ (x y : ℝ), P = (1, y) → y = f 1 → tangent_line x = (Real.exp 1 / 4) * x + (Real.exp 1 / 4) := by
  sorry

end find_tangent_line_l801_801916


namespace sequence_sum_13_terms_l801_801066

def seq_sum : ℕ → ℝ
| 0     := 0
| (n+1) := seq_sum n + (1 / ((2*n - 15) * (2*(n+1) - 15)))

theorem sequence_sum_13_terms (a : ℕ → ℝ)
    (h1 : a 1 = -13)
    (h2 : a 6 + a 8 = -2)
    (h3 : ∀ n ≥ 2, a (n-1) = 2 * a n - a (n+1))
    : seq_sum 13 = -1 / 13 :=
sorry

end sequence_sum_13_terms_l801_801066


namespace axis_of_symmetry_l801_801909

-- Define the given parabola
def parabola (x : ℝ) : ℝ := x^2 - 2 * x + 5

-- Define the statement that we need to prove
theorem axis_of_symmetry : (∃ (a : ℝ), ∀ x, parabola (x) = (x - a) ^ 2 + 4) ∧ 
                           (∃ (b : ℝ), b = 1) :=
by
  sorry

end axis_of_symmetry_l801_801909


namespace h_x_eq_x_is_51_over_2_l801_801500

theorem h_x_eq_x_is_51_over_2 (h : ℝ → ℝ) (H : ∀ x : ℝ, h (5 * x - 2) = 3 * x + 9) :
  ∃ x : ℝ, h x = x ∧ x = 51 / 2 :=
by
  use 51 / 2
  split
  · sorry
  · rfl

end h_x_eq_x_is_51_over_2_l801_801500


namespace pyramid_volume_l801_801575

theorem pyramid_volume (a : ℝ) (a_pos : 0 < a)
  (DH : a / 2 = height_DH)
  (base_area : (sqrt 3 / 4) * a^2 = area_base)
  (vol_formula : volume = (1 / 3) * area_base * height_DH)
  (a_eq_3 : a = 3) :
  volume = 9 * sqrt 3 / 8 := 
sorry

end pyramid_volume_l801_801575


namespace distance_between_foci_of_ellipse_zero_l801_801013

theorem distance_between_foci_of_ellipse_zero :
  ∀ (x y : ℝ), 9 * x ^ 2 + 36 * x + 4 * y ^ 2 - 8 * y + 20 = 0 → 
  let a := sqrt (20 / 9)
      b := sqrt 5
      c := sqrt (a ^ 2 - b ^ 2)
  in 2 * c = 0 :=
by
  intro x y h
  let a := sqrt (20 / 9)
  let b := sqrt 5
  let c := sqrt (a ^ 2 - b ^ 2)
  suffices h1 : 2 * c = 2 * sqrt (a ^ 2 - b ^ 2), from h1,
  suffices h2 : a = sqrt 5, from (by rw [h2, sqrt_sub_self] : sqrt (sqrt 5 ^ 2 - sqrt 5 ^ 2) = 0),
  suffices h3 : sqrt (20 / 9) = sqrt 5, from h3,
  sorry

end distance_between_foci_of_ellipse_zero_l801_801013


namespace park_area_l801_801237

variable (L B : ℕ)
variable (speed_km_hr speed_km_min perimeter distance_round : ℕ)
variable (minutes : ℕ := 6)
variable (hours : ℕ := 1)
variable (meter : ℕ := 1000) -- 1 km to meters

-- Conditions from the problem
def length_breadth_ratio := B = 2 * L
def speed := speed_km_hr = 6
def speed_conversion := speed_km_min = speed_km_hr * (hours / 60)
def perimeter_equation := perimeter = 2 * (L + B)
def perimeter_value := perimeter = 600
def distance_round_equation := distance_round = (speed_km_min * minutes) * meter
def perimeter_equals_distance := perimeter = distance_round / meter

-- Correct Answer
theorem park_area :
  length_breadth_ratio →
  speed →
  speed_conversion →
  perimeter_equation →
  perimeter_value →
  distance_round_equation →
  perimeter_equals_distance →
  L * B = 20000 :=
by
  intros h_ratio h_speed h_conversion h_perimeter h_perimeter_val h_distance h_perimeter_dist
  -- Proof is skipped, so we use sorry
  sorry

end park_area_l801_801237


namespace functional_equation_solution_l801_801710

theorem functional_equation_solution :
  ∀ (f : ℤ → ℤ), (∀ (m n : ℤ), f (m + f (f n)) = -f (f (m + 1)) - n) → (∀ (p : ℤ), f p = 1 - p) :=
by
  intro f h
  sorry

end functional_equation_solution_l801_801710


namespace partial_sum_lt_two_l801_801400

-- Define the partial sum of the geometric series
def partial_sum (n : ℕ) : ℝ :=
  (finset.range (n + 1)).sum (λ k, (1 : ℝ) / 2^k)

-- State the main theorem to prove under the given conditions
theorem partial_sum_lt_two (n : ℕ) : partial_sum n < 2 :=
sorry

end partial_sum_lt_two_l801_801400


namespace brick_height_l801_801976

theorem brick_height (l_w_wall w_m_wall h_m_wall : ℝ)
  (mortar_percentage : ℝ)
  (num_bricks : ℕ)
  (l_brick w_brick : ℝ)
  (h_brick : ℝ) :
  l_w_wall = 10 → w_m_wall = 4 → h_m_wall = 5 →
  mortar_percentage = 0.1 →
  num_bricks = 6000 →
  l_brick = 25 → w_brick = 15 →
  (let V_wall := l_w_wall * w_m_wall * h_m_wall in
   let V_bricks := (1 - mortar_percentage) * V_wall in
   let V_bricks_cm³ := V_bricks * 1000000 in
   let V_brick := l_brick * w_brick * h_brick in
   let total_V_bricks := V_brick * num_bricks in
   V_bricks_cm³ = total_V_bricks) →
  h_brick = 80 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end brick_height_l801_801976


namespace students_who_saw_l801_801697

variable (B G : ℕ)

theorem students_who_saw (h : B + G = 33) : (2 * G / 3) + (2 * B / 3) = 22 :=
by
  sorry

end students_who_saw_l801_801697


namespace hour_hand_rotation_l801_801227

theorem hour_hand_rotation (h1 h2 : ℕ) (angle_per_hour : ℕ) : 
  h1 = 6 → h2 = 9 → angle_per_hour = 30 → (h2 - h1) * angle_per_hour = 90 := by
  intros h1_eq h2_eq angle_eq
  rw [h1_eq, h2_eq, angle_eq]
  sorry

end hour_hand_rotation_l801_801227


namespace diana_win_strategy_diana_win_with_specific_config_l801_801110

def nim_value : ℕ → ℕ
| 0     := 0
| 1     := 1
| 2     := 2
| 3     := 3
| n     := if n % 2 = 0 then 2 else 3 -- Example nim-values for this problem

def nim_sum (configs : List ℕ) : ℕ :=
configs.foldl (λ acc x => acc + nim_value x) 0

theorem diana_win_strategy (configs : List ℕ) :
nim_sum configs = 0 → ∃ f : List ℕ → List ℕ, ∀ c ∈ configs, nim_sum (f configs) = 0 :=
begin
  sorry
end

-- Instantiate the theorem with the specific configuration
theorem diana_win_with_specific_config : 
  diana_win_strategy [6, 2, 1] :=
begin
  sorry
end

end diana_win_strategy_diana_win_with_specific_config_l801_801110


namespace sqrt_gt_one_sqrt_lt_one_l801_801887

variable (a n : ℕ)

theorem sqrt_gt_one (h_a : a > 1) (h_n : n > 0) : (nat.sqrt a n) > 1 := sorry

theorem sqrt_lt_one (h_a : a < 1) (h_n : n > 0) : (nat.sqrt a n) < 1 := sorry

end sqrt_gt_one_sqrt_lt_one_l801_801887


namespace John_Anna_total_eBooks_l801_801481

variables (Anna_bought : ℕ) (eBooks_lost_by_John : ℕ) 

def John_bought (Anna_bought : ℕ) : ℕ := Anna_bought - 15
def John_left (Anna_bought : ℕ) (eBooks_lost_by_John : ℕ) : ℕ := John_bought Anna_bought - eBooks_lost_by_John

theorem John_Anna_total_eBooks (Anna_bought_eq_50 : Anna_bought = 50)
    (John_bought_eq_35 : John_bought Anna_bought = 35) (eBooks_lost_eq_3 : eBooks_lost_by_John = 3) :
    (Anna_bought + John_left Anna_bought eBooks_lost_by_John = 82) :=
by sorry

end John_Anna_total_eBooks_l801_801481


namespace sqrt_inequality_solution_l801_801932

theorem sqrt_inequality_solution : 
  {x : Real | sqrt (x + 3) < 2} = Set.Ico (-3) 1 := 
by 
  sorry

end sqrt_inequality_solution_l801_801932


namespace concert_attendance_difference_l801_801868

noncomputable def first_concert : ℕ := 65899
noncomputable def second_concert : ℕ := 66018

theorem concert_attendance_difference :
  (second_concert - first_concert) = 119 :=
by
  sorry

end concert_attendance_difference_l801_801868


namespace angela_total_money_spent_l801_801695

theorem angela_total_money_spent (spent money_left total_money : ℕ) 
                                 (h1 : spent = 78) 
                                 (h2 : money_left = 12) :
                                 total_money = spent + money_left := 
begin
  -- Note: The proof is intentionally left as 'sorry' to indicate it's skipped
  sorry,
end

-- Apply the theorem to the specific case of Angela
lemma angela_money : angela_total_money_spent 78 12 90 :=
begin
  sorry,
end

end angela_total_money_spent_l801_801695


namespace smallest_m_for_divisibility_l801_801891

theorem smallest_m_for_divisibility : 
  ∃ (m : ℕ), 2^1990 ∣ 1989^m - 1 ∧ m = 2^1988 := 
sorry

end smallest_m_for_divisibility_l801_801891


namespace compute_ff4_l801_801379

def f : ℝ → ℝ :=
  λ x, if x > 0 then x^(1/2) - 2 else Real.exp x

theorem compute_ff4 : f (f 4) = 1 := by
  sorry

end compute_ff4_l801_801379


namespace point_inside_polygon_iff_odd_marked_vertices_l801_801382

variables {Polygon : Type} {Line : Type} {Point : Type} [Geometry Polygon Line Point]

def is_general_position (polygon : Polygon) (l : Line) (P : Point) : Prop :=
  ∀ side₁ side₂ ∈ sides polygon, side₁ ≠ side₂ → ∃! (x : Point), x ∈ (mk_segment (side₁, side₂)) ∧ x ≠ P

def marked_vertices_on_line (polygon : Polygon) (l : Line) (P : Point) : list Point :=
  {V ∈ vertices polygon | intersects_line (line_through V (outgoing_side V)) l ∧ intersects_line (line_through V (outgoing_side V)) l ≠ P}

def is_point_inside_polygon (polygon : Polygon) (P : Point) : Prop :=
  ∃ side ∈ sides polygon, P ∈ side

theorem point_inside_polygon_iff_odd_marked_vertices (polygon : Polygon) (l : Line) (P : Point)
  (hg : is_general_position polygon l P) :
  (is_point_inside_polygon polygon P) ↔ 
  (list.length (marked_vertices_on_line polygon l P) % 2 = 1) :=
sorry

end point_inside_polygon_iff_odd_marked_vertices_l801_801382


namespace number_of_puppies_with_4_spots_is_3_l801_801666

noncomputable def total_puppies : Nat := 10
noncomputable def puppies_with_5_spots : Nat := 6
noncomputable def puppies_with_2_spots : Nat := 1
noncomputable def puppies_with_4_spots : Nat := total_puppies - puppies_with_5_spots - puppies_with_2_spots

theorem number_of_puppies_with_4_spots_is_3 :
  puppies_with_4_spots = 3 := 
sorry

end number_of_puppies_with_4_spots_is_3_l801_801666


namespace max_value_f_area_triangle_ABC_l801_801017

noncomputable def f (x : ℝ) : ℝ := sin(2 * x + π / 6) + cos(2 * x - π / 3)

-- Problem I: Prove the maximum value of f(x)
theorem max_value_f : ∃ (x : ℝ), f(x) = 2 := sorry

-- Problem II: Given the conditions, prove the area of triangle ABC
variables (a b c A B C : ℝ)
variables (h_fC : f(C) = 1) (h_c : c = 2 * √3) (h_sin : sin A = 2 * sin B)

-- Using the Law of Sines and Law of Cosines
theorem area_triangle_ABC :
  let s := (a * b * sin C) / 2 in
  c = 2 * √3 ∧ sin A = 2 * sin B → s = 2 * √3 := sorry

end max_value_f_area_triangle_ABC_l801_801017


namespace power_binary_representation_zero_digit_l801_801890

theorem power_binary_representation_zero_digit
  (a n s : ℕ) (ha : a > 1) (hn : n > 1) (hs : s > 0) :
  a ^ n ≠ 2 ^ s - 1 :=
by
  sorry

end power_binary_representation_zero_digit_l801_801890


namespace tangent_line_at_point_l801_801915

open Real

def curve (x : ℝ) : ℝ := exp x / (x + 1)

def tangent_line (x : ℝ) : ℝ := (exp 1 / 4) * x + (exp 1 / 4)

theorem tangent_line_at_point :
  tangent_line = λ x, (exp 1 / 4) * x + (exp 1 / 4) :=
by
  sorry

end tangent_line_at_point_l801_801915


namespace quadratic_residue_one_mod_p_l801_801502

theorem quadratic_residue_one_mod_p (p : ℕ) [hp : Fact (Nat.Prime p)] (a : ℕ) :
  (a^2 % p = 1 % p) ↔ (a % p = 1 % p ∨ a % p = (p-1) % p) :=
sorry

end quadratic_residue_one_mod_p_l801_801502


namespace simplify_tangent_sum_l801_801540

theorem simplify_tangent_sum :
  tan (Real.pi / 12) + tan (5 * Real.pi / 12) = Real.sqrt 6 - Real.sqrt 2 := 
sorry

end simplify_tangent_sum_l801_801540


namespace find_a_if_sum_purely_imaginary_l801_801396

variable (a : ℝ)

def z1 : ℂ := a^2 - 2 - 3 * complex.I * a
def z2 : ℂ := a + (a^2 + 2) * complex.I

theorem find_a_if_sum_purely_imaginary 
  (h1 : (z1 + z2).re = 0) 
  (h2 : (z1 + z2).im ≠ 0) : 
  a = -2 :=
by 
  sorry

end find_a_if_sum_purely_imaginary_l801_801396


namespace bob_total_distance_travelled_l801_801699

theorem bob_total_distance_travelled 
(first_segment_time_minutes : Nat) (first_segment_speed_mph : Nat)
(second_segment_time_minutes : Nat) (second_segment_speed_mph : Nat)
(third_segment_time_minutes : Nat) (third_segment_speed_mph : Nat)
(fourth_segment_time_minutes : Nat) (fourth_segment_speed_mph : Nat)
: first_segment_time_minutes = 105 →
  first_segment_speed_mph = 65 →
  second_segment_time_minutes = 140 →
  second_segment_speed_mph = 45 →
  third_segment_time_minutes = 90 →
  third_segment_speed_mph = 35 →
  fourth_segment_time_minutes = 135 →
  fourth_segment_speed_mph = 50 →
  (first_segment_speed_mph * (first_segment_time_minutes / 60.0) +
   second_segment_speed_mph * (second_segment_time_minutes / 60.0) +
   third_segment_speed_mph * (third_segment_time_minutes / 60.0) +
   fourth_segment_speed_mph * (fourth_segment_time_minutes / 60.0) = 383.735) :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end bob_total_distance_travelled_l801_801699


namespace number_of_solutions_l801_801219

open Real

-- Define the main equation in terms of absolute values 
def equation (x : ℝ) : Prop := abs (x - abs (2 * x + 1)) = 3

-- Prove that there are exactly 2 distinct solutions to the equation
theorem number_of_solutions : 
  ∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ ∧ equation x₂ :=
sorry

end number_of_solutions_l801_801219


namespace intersection_nonempty_implies_a_gt_neg1_l801_801844

theorem intersection_nonempty_implies_a_gt_neg1 :
  (∀ x, (-1 ≤ x ∧ x < 2) ↔ x ∈ set { x | -1 ≤ x ∧ x < 2 }) →
  (∃ x : ℝ, -1 ≤ x ∧ x < 2 ∧ x < a) → a > -1 := by
  intros _ h
  rcases h with ⟨x, hx1, hx2, hx3⟩
  linarith

end intersection_nonempty_implies_a_gt_neg1_l801_801844


namespace value_of_three_inch_cube_l801_801309

theorem value_of_three_inch_cube (value_two_inch: ℝ) (volume_two_inch: ℝ) (volume_three_inch: ℝ) (cost_two_inch: ℝ):
  value_two_inch = cost_two_inch * ((volume_three_inch / volume_two_inch): ℝ) := 
by
  have volume_two_inch := 2^3 -- Volume of two-inch cube
  have volume_three_inch := 3^3 -- Volume of three-inch cube
  let volume_ratio := (volume_three_inch / volume_two_inch: ℝ)
  have := cost_two_inch * volume_ratio
  norm_num
  sorry

end value_of_three_inch_cube_l801_801309


namespace find_a_plus_b_l801_801045

theorem find_a_plus_b (a b : ℝ) :
  (∀ x : ℝ, |x - 2| > 1 ↔ x^2 + a * x + b > 0) →
  a + b = -1 :=
by {
  assume h : ∀ x : ℝ, |x - 2| > 1 ↔ x^2 + a * x + b > 0,
  sorry
}

end find_a_plus_b_l801_801045


namespace probability_of_selecting_red_books_is_3_div_14_l801_801939

-- Define the conditions
def total_books : ℕ := 8
def red_books : ℕ := 4
def blue_books : ℕ := 4
def books_selected : ℕ := 2

-- Define the calculation of the probability
def probability_red_books_selected : ℚ :=
  let total_outcomes := Nat.choose total_books books_selected
  let favorable_outcomes := Nat.choose red_books books_selected
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

-- State the theorem
theorem probability_of_selecting_red_books_is_3_div_14 :
  probability_red_books_selected = 3 / 14 :=
by
  sorry

end probability_of_selecting_red_books_is_3_div_14_l801_801939


namespace angles_sum_proof_l801_801447

-- Given that BD, FC, GC, and FE are straight lines
-- Angles a, b, c, d, e, f, g are given as the angles in the figure formed
-- Prove that the sum of these angles equals 540 degrees

def angles_sum (a b c d e f g : ℝ) : Prop :=
  a + b + c + d + e + f + g = 540

theorem angles_sum_proof (a b c d e f g : ℝ)
  (h_a : a ∈ set.Icc 0 360) (h_b : b ∈ set.Icc 0 360)
  (h_c : c ∈ set.Icc 0 360) (h_d : d ∈ set.Icc 0 360)
  (h_e : e ∈ set.Icc 0 360) (h_f : f ∈ set.Icc 0 360)
  (h_g : g ∈ set.Icc 0 360) :
  angles_sum a b c d e f g :=
begin
  sorry
end

end angles_sum_proof_l801_801447


namespace largest_prime_in_set_average_27_eq_139_l801_801201

-- Define the conditions
def is_prime_set (A : Set ℕ) : Prop :=
  ∀ a ∈ A, Nat.Prime a

def is_average_27 (A : Set ℕ) : Prop :=
  ∑ a in A, a = 27 * (Set.card A)

-- Define the problem statement
theorem largest_prime_in_set_average_27_eq_139 (A : Set ℕ) 
  (h1 : is_prime_set A)
  (h2 : is_average_27 A) :
  ∃ p ∈ A, p = 139 ∧ ∀ q ∈ A, q ≤ 139 := 
sorry

end largest_prime_in_set_average_27_eq_139_l801_801201


namespace min_young_rank_l801_801638

-- Conditions
def yuna_rank : ℕ := 6
def min_young_offset_after_yuna : ℕ := 5

-- Proof statement
theorem min_young_rank :
  min_young_rank = yuna_rank + min_young_offset_after_yuna := by
  -- We are given that:
  -- yuna_rank = 6
  -- min_young_offset_after_yuna = 5
  -- We need to find and prove Min-Young's rank:
  sorry

end min_young_rank_l801_801638


namespace range_of_a_l801_801860

-- Definitions based on the given conditions
def A (a : ℝ) : set ℝ := {x | x ≤ a}
def B : set ℝ := {x | x < 2}

-- The main proof statement
theorem range_of_a (a : ℝ) (h : A a ⊆ B) : a < 2 := sorry

end range_of_a_l801_801860


namespace trigonometric_identity_l801_801047

noncomputable def point := (-4, 3)
noncomputable def hypotenuse : ℝ := real.sqrt ((-4)^2 + 3^2)
noncomputable def sin_alpha : ℝ := 3 / hypotenuse
noncomputable def cos_alpha : ℝ := -4 / hypotenuse

theorem trigonometric_identity : 
  (sin_alpha - 2 * cos_alpha) / (sin_alpha + cos_alpha) = -11 :=
by
  sorry

end trigonometric_identity_l801_801047


namespace range_of_lambda_l801_801376

theorem range_of_lambda (x y : ℝ) (λ : ℝ) (hx : 0 < x) (hy : 0 < y)
  (a : ℝ := x + y) (b : ℝ := sqrt (x^2 - x * y + y^2)) (c : ℝ := λ * sqrt (x * y))
  (h_triangle : a > b ∧ b + c > a ∧ a + b > c) :
  1 < λ ∧ λ < 3 := by
  sorry

end range_of_lambda_l801_801376


namespace max_principals_in_8_years_l801_801718

theorem max_principals_in_8_years 
  (years_in_term : ℕ)
  (terms_in_given_period : ℕ)
  (term_length : ℕ)
  (term_length_eq : term_length = 4)
  (given_period : ℕ)
  (given_period_eq : given_period = 8) :
  terms_in_given_period = given_period / term_length :=
by
  rw [term_length_eq, given_period_eq]
  sorry

end max_principals_in_8_years_l801_801718


namespace total_pears_picked_l801_801166

def mikes_pears : Nat := 8
def jasons_pears : Nat := 7
def freds_apples : Nat := 6

theorem total_pears_picked : (mikes_pears + jasons_pears) = 15 :=
by
  sorry

end total_pears_picked_l801_801166


namespace simplify_trig_identity_l801_801184

variables {x y : ℝ}

theorem simplify_trig_identity : sin (x + y) * sin x + cos (x + y) * cos x = cos y :=
sorry

end simplify_trig_identity_l801_801184


namespace find_number_l801_801641

theorem find_number (x : ℝ) : 3 * (2 * x + 9) = 75 → x = 8 :=
by {
  sorry
}

end find_number_l801_801641


namespace x_power_24_l801_801435

theorem x_power_24 (x : ℝ) (h : x + 1/x = -real.sqrt 3) : x^24 = 390625 :=
by
  sorry

end x_power_24_l801_801435


namespace find_a_l801_801444

noncomputable def constant_term_condition : Prop :=
  let general_term (x a : ℝ) (r : ℕ) : ℝ := ((-a)^r * Nat.choose 8 r * x^((4-2*r)/3))
  (∀ (a : ℝ), 
   ∃ (r : ℕ), 
     (4-2*r)/3 = 0 ∧
     ∑ r in {r | (general_term x a r = 56)}, r = 2)
      
theorem find_a (x : ℝ) (a : ℝ) : 
  constant_term_condition →
  a = sqrt 2 ∨ a = -sqrt 2 :=
sorry

end find_a_l801_801444


namespace find_n_l801_801342

theorem find_n (n : ℕ) (h : 2^6 * 3^3 * n = Nat.factorial 10) : n = 2100 :=
by
sorry

end find_n_l801_801342


namespace expr_1989_eval_expr_1990_eval_l801_801558

def nestedExpr : ℕ → ℤ
| 0     => 0
| (n+1) => -1 - (nestedExpr n)

-- Conditions translated into Lean definitions:
def expr_1989 := nestedExpr 1989
def expr_1990 := nestedExpr 1990

-- The proof statements:
theorem expr_1989_eval : expr_1989 = -1 := sorry
theorem expr_1990_eval : expr_1990 = 0 := sorry

end expr_1989_eval_expr_1990_eval_l801_801558


namespace length_GG_l801_801499

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem length_GG'
  (A B C : ℝ)
  (AB BC CA : ℝ)
  (hAB : AB = 9)
  (hBC : BC = 10)
  (hCA : CA = 17)
  (B' : ℝ)
  (hB' : reflecting_over_line CA B = B')
  (G G' : ℝ)
  (hG : centroid_of_triangle B C A = G)
  (hG' : centroid_of_triangle B' C A = G') :
  dist G G' = 48 / 17 := 
sorry

end length_GG_l801_801499


namespace domain_of_function_l801_801391

theorem domain_of_function (f : ℝ → ℝ) :
  (∀ x, 0 ≤ x + 1 ∧ x + 1 ≤ 1 → True) →
  (∀ x, 1 ≤ x ∧ x ≤ 5 / 2 → f (sqrt (2 * x - 1)) ∈ f '' Icc 1 (5 / 2)) :=
by
  intro h
  sorry

end domain_of_function_l801_801391


namespace John_Anna_total_eBooks_l801_801479

variables (Anna_bought : ℕ) (eBooks_lost_by_John : ℕ) 

def John_bought (Anna_bought : ℕ) : ℕ := Anna_bought - 15
def John_left (Anna_bought : ℕ) (eBooks_lost_by_John : ℕ) : ℕ := John_bought Anna_bought - eBooks_lost_by_John

theorem John_Anna_total_eBooks (Anna_bought_eq_50 : Anna_bought = 50)
    (John_bought_eq_35 : John_bought Anna_bought = 35) (eBooks_lost_eq_3 : eBooks_lost_by_John = 3) :
    (Anna_bought + John_left Anna_bought eBooks_lost_by_John = 82) :=
by sorry

end John_Anna_total_eBooks_l801_801479


namespace solve_system_of_equations_l801_801561

theorem solve_system_of_equations (r : ℝ) (n : ℝ) (x : ℕ → ℝ) (k : ℕ) 
(h1 : x 1 + x 2 + x 3 + ... + x k = r / (n - 1)) 
(h2 : x 2 + x 3 + ... + x k = (r + x 1) / (n ^ 2 - 1))
(h3 : x 3 + x 4 + ... + x k = (r + x 1 + x 2) / (n ^ 3 - 1))
-- ...
(hk_minus_1 : x (k - 1) + x k = (r + x 1 + x 2 + ... + x (k - 2)) / (n ^ (k - 1) - 1))
(hk : x k = (r + x 1 + x 2 + ... + x (k - 1)) / (n ^ k - 1)) :
∀ i, i ∈ (Finset.range k).succ → x i = r / (n ^ i) := sorry

end solve_system_of_equations_l801_801561


namespace determine_value_of_omega_l801_801778

noncomputable def f (ω x : ℝ) : ℝ := sin (ω * x) + cos (ω * x)

theorem determine_value_of_omega (ω : ℝ) (h_ω_pos : ω > 0) 
    (h_mono_increase : ∀ x y, -ω < x → x < y → y < ω → f ω x ≤ f ω y)
    (h_symmetric : ∀ x, f ω x = f ω (2 * ω - x)) :
  ω = sqrt π / 2 :=
sorry

end determine_value_of_omega_l801_801778


namespace monotonicity_and_extrema_of_f_l801_801775

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem monotonicity_and_extrema_of_f :
  (∀ x : ℝ, x < -1 → deriv f x > 0) ∧
  (∀ x : ℝ, -1 < x ∧ x < 1 → deriv f x < 0) ∧
  (∀ x : ℝ, x > 1 → deriv f x > 0) ∧
  (∃ a b : ℝ, a ∈ set.Icc (-3 : ℝ) (2 : ℝ) ∧ b ∈ set.Icc (-3 : ℝ) (2 : ℝ) ∧
    (∀ x : ℝ, x ∈ set.Icc (-3) 2 → f x ≤ f a) ∧
    (∀ x : ℝ, x ∈ set.Icc (-3) 2 → f x ≥ f b) ∧
    f a = -18 ∧ f b = 2) := by
      sorry

end monotonicity_and_extrema_of_f_l801_801775


namespace pyramid_edges_sum_l801_801676

noncomputable def sum_of_pyramid_edges (s : ℝ) (h : ℝ) : ℝ :=
  let diagonal := s * Real.sqrt 2
  let half_diagonal := diagonal / 2
  let slant_height := Real.sqrt (half_diagonal^2 + h^2)
  4 * s + 4 * slant_height

theorem pyramid_edges_sum
  (s : ℝ) (h : ℝ)
  (hs : s = 15)
  (hh : h = 15) :
  sum_of_pyramid_edges s h = 135 :=
sorry

end pyramid_edges_sum_l801_801676


namespace conjugate_complex_number_l801_801049

def z : ℂ := 2 * complex.I / (3 + complex.I)

theorem conjugate_complex_number :
  complex.conj z = (1 / 5 : ℂ) - (3 / 5 : ℂ) * complex.I :=
by
  sorry

end conjugate_complex_number_l801_801049


namespace sum_func_y_1_to_100_l801_801406

def func_y (x : ℕ) : ℕ :=
  if x ≤ 2 ∨ x ≥ 98 then
    (x * x - 100 * x + 196)
  else
    0

theorem sum_func_y_1_to_100 : 
  ∑ k in Finset.range 100, func_y (k + 1) = 390 :=
  sorry

end sum_func_y_1_to_100_l801_801406


namespace area_of_circle_l801_801213
open Real

-- Define the circumference condition
def circumference (r : ℝ) : ℝ :=
  2 * π * r

-- Define the area formula
def area (r : ℝ) : ℝ :=
  π * r * r

-- The given radius derived from the circumference
def radius_given_circumference (C : ℝ) : ℝ :=
  C / (2 * π)

-- The target proof statement
theorem area_of_circle (C : ℝ) (h : C = 36) : (area (radius_given_circumference C)) = 324 / π :=
by
  sorry

end area_of_circle_l801_801213


namespace angle_complement_l801_801818

theorem angle_complement (x : ℝ) (h1 : ∠PQR = 90) 
  (h2 : ∠PQS = 3 * x) (h3 : ∠SQR = 2 * x) : x = 18 :=
by
  -- Proof begins here
  sorry

end angle_complement_l801_801818


namespace value_of_coat_l801_801991

noncomputable def coat_value := 4.8

theorem value_of_coat 
  (annual_rubles : ℝ) (months_worked : ℝ) (rubles_received : ℝ) (actual_payment : ℝ) (value_of_coat : ℝ) :
  annual_rubles = 12 →
  months_worked = 7 →
  rubles_received = 5 →
  actual_payment = 5 + value_of_coat →
  (7 / 12 * (12 + value_of_coat)) = actual_payment →
  value_of_coat = 4.8 :=
by {
  intros h1 h2 h3 h4 h5,
  assume h5,
  sorry
}

end value_of_coat_l801_801991


namespace interest_calculations_l801_801164

noncomputable def principal : ℝ := 30000
noncomputable def time : ℝ := 3
noncomputable def annual_rate : ℝ := 0.047
noncomputable def tax_rate : ℝ := 0.20

theorem interest_calculations :
  let pre_tax_interest := principal * annual_rate * time in
  let after_tax_interest := pre_tax_interest * (1 - tax_rate) in
  let total_withdrawal := principal + after_tax_interest in
  after_tax_interest = 3372 ∧ total_withdrawal = 33372 :=
by
  sorry

end interest_calculations_l801_801164


namespace james_total_money_at_end_of_year_l801_801130

-- Definitions based on conditions
def weekly_investment : ℕ := 2000
def initial_amount : ℕ := 250000
def weeks_in_year : ℕ := 52
def windfall_rate : ℝ := 0.5

-- The total amount James has at the end of the year
theorem james_total_money_at_end_of_year 
  (weekly_investment : ℕ)
  (initial_amount : ℕ)
  (weeks_in_year : ℕ)
  (windfall_rate : ℝ) :
  let total_deposit := weekly_investment * weeks_in_year
      balance_before_windfall := initial_amount + total_deposit
      windfall := windfall_rate * balance_before_windfall
      total_balance := balance_before_windfall + windfall in
  total_balance = 885000 :=
by {
  let total_deposit := 2000 * 52
  let balance_before_windfall := 250000 + total_deposit
  let windfall := 0.5 * (balance_before_windfall : ℝ)
  let total_balance := balance_before_windfall + windfall
  show total_balance = 885000,
  sorry
}

end james_total_money_at_end_of_year_l801_801130


namespace cos_half_pi_plus_double_alpha_l801_801753

theorem cos_half_pi_plus_double_alpha (α : ℝ) (h : Real.tan α = 1 / 3) : 
  Real.cos (Real.pi / 2 + 2 * α) = -3 / 5 :=
by
  sorry

end cos_half_pi_plus_double_alpha_l801_801753


namespace sum_and_product_of_divisors_l801_801849

theorem sum_and_product_of_divisors (m : ℕ) (h1 : m > 0) (h2 : 225 % m = 5) :
  (m = 22 ∨ m = 55) → (m ∈ {22, 55} → (22 + 55 = 77 ∧ 22 * 55 = 1210)) :=
by
  intro h
  cases h
  · have : m = 22 := h
    rw [this]
    split
    repeat sorry
  · have : m = 55 := h
    rw [this]
    split
    repeat sorry

end sum_and_product_of_divisors_l801_801849


namespace tan_sum_pi_over_12_l801_801533

theorem tan_sum_pi_over_12 : 
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12)) = 4 := 
sorry

end tan_sum_pi_over_12_l801_801533


namespace solve_problem_l801_801026

noncomputable def a_n : ℕ → ℕ
| 0     => 2
| (n+1) => 2 * a_n n

def S (n : ℕ) := 2 * a_n n - 2

def b_n (n : ℕ) := 10 - n

theorem solve_problem :
  (∀ n, S n = 2 * a_n n - 2) →
  (∀ n, b_n n = 10 - n) →
  (∃ n, (n = 9 ∨ n = 10) ∧ b_n n = 10 - n) :=
by
  intros hS hb
  use 9
  rw [hb]
  left
  refl
  -- sorry: for young proof you can use references like theorem statements including only the stuff you know from problem.

end solve_problem_l801_801026


namespace ring_arrangements_l801_801766

open Nat

theorem ring_arrangements : 
  let rings := 10
  let selected_rings := 6
  let fingers := 4
  let binom := Nat.choose rings selected_rings
  let perm := Nat.factorial selected_rings
  let dist := fingers ^ selected_rings
  n = binom * perm * dist :=
by
  let rings := 10
  let selected_rings := 6
  let fingers := 4
  let binom := Nat.choose rings selected_rings
  let perm := Nat.factorial selected_rings
  let dist := fingers ^ selected_rings
  have : n = 210 * 720 * 4096 := sorry
  exact this

end ring_arrangements_l801_801766


namespace left_shift_sin_cos_phi_l801_801042

theorem left_shift_sin_cos_phi (φ : ℝ) (h1 : φ ∈ set.Ioc 0 (2 * π)) :
  (∀ x, (sin x + sqrt 3 * cos x) = 2 * sin (x + π / 3)) ∧
  (∀ x, (sin x - sqrt 3 * cos x) = 2 * sin (x - π / 3)) ∧
  (∀ x, 2 * sin (x + π / 3) = 2 * sin (x - π / 3 + φ)) → 
  φ = (2 * π / 3) :=
by
  intro h
  sorry

end left_shift_sin_cos_phi_l801_801042


namespace faster_train_length_225_l801_801646

noncomputable def length_of_faster_train (speed_slower speed_faster : ℝ) (time : ℝ) : ℝ :=
  let relative_speed_kmph := speed_slower + speed_faster
  let relative_speed_mps := (relative_speed_kmph * 1000) / 3600
  relative_speed_mps * time

theorem faster_train_length_225 :
  length_of_faster_train 36 45 10 = 225 := by
  sorry

end faster_train_length_225_l801_801646


namespace constant_sequence_no_perfect_cubes_l801_801409

/-- Define the sequences x_n and y_n --/
def x_seq : ℕ → ℤ
| 0 := 3  -- initial condition x_1 = 3
| (n + 1) := 3 * x_seq n + 2 * y_seq n
with y_seq : ℕ → ℤ
| 0 := 4  -- initial condition y_1 = 4
| (n + 1):= 4 * x_seq n + 3 * y_seq n

/-- Prove that 2 * x_seq n^2 - y_seq n^2 is constant and equals to 2 --/
theorem constant_sequence : ∀ n : ℕ, 2 * (x_seq n) ^ 2 - (y_seq n) ^ 2 = 2 :=
by
  sorry

/-- Prove that there are no perfect cubes in the sequences x_seq and y_seq --/
theorem no_perfect_cubes : ∀ n : ℕ, ¬∃ k : ℤ, k ^ 3 = x_seq n ∨ k ^ 3 = y_seq n :=
by
  sorry

end constant_sequence_no_perfect_cubes_l801_801409


namespace number_of_divisors_of_polynomial_l801_801145

theorem number_of_divisors_of_polynomial (p : ℕ) (hp : Nat.Prime p)
    (div_int : ∃ k : ℕ, 28 ^ p - 1 = k * (2 * p ^ 2 + 2 * p + 1)) :
    Nat.divisor_count (2 * p ^ 2 + 2 * p + 1) = 2 :=
sorry

end number_of_divisors_of_polynomial_l801_801145


namespace max_sum_l801_801762

theorem max_sum (n : ℕ) (h : n ≥ 2) (a : fin (n+1) → ℝ) (h_cond : ∀ k : fin n, 2 ≤ k.val.succ → a k.succ ≥ ∑ i in finset.range k.val, a ⟨i, nat.lt_succ_iff.mp k.property⟩) :
  (∑ i in finset.range (n-1), a ⟨i, _⟩ / a ⟨i.succ, _⟩) ≤ (n : ℝ) / 2 :=
sorry

end max_sum_l801_801762


namespace solve_sin_equation_l801_801902

theorem solve_sin_equation (x y : ℝ) : 
  abs (sin x - sin y) + sin x * sin y = 0 → (∃ k n : ℤ, x = k * π ∧ y = n * π) :=
by
  sorry

end solve_sin_equation_l801_801902


namespace find_g_5_l801_801925

-- Define the function g and the condition it satisfies
variable {g : ℝ → ℝ}
variable (hg : ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = 2 * x)

-- The proof goal
theorem find_g_5 : g 5 = 206 / 35 :=
by
  -- To be proven using the given condition hg
  sorry

end find_g_5_l801_801925


namespace pyramid_edges_cannot_form_closed_polygon_l801_801141

/-- Given that a pyramid has 171 lateral edges and 171 edges in the base, 
    it is not possible to translate each of the 342 edges for them to form a 
    closed broken line in space. -/
theorem pyramid_edges_cannot_form_closed_polygon :
    ∀ (n : ℕ), n = 171 → 
    let total_edges := 2 * n in
    total_edges = 342 →
    ¬(∃ (translate_edge : fin total_edges → ℝ × ℝ × ℝ), 
        let points := fin total_edges in
        (∀ i, translate_edge (i + 1) = points : fin total_edges → ℝ × ℝ × ℝ) ∧ 
        (translate_edge 0 = translate_edge (total_edges - 1))) :=
by
  intros n hn htotal
  sorry

end pyramid_edges_cannot_form_closed_polygon_l801_801141


namespace three_c_plus_two_d_l801_801187

theorem three_c_plus_two_d (c d : ℝ)
  (h1 : c^2 - 6 * c + 15 = 25)
  (h2 : d^2 - 6 * d + 15 = 25)
  (h3 : c ≥ d) :
  3 * c + 2 * d = 15 + real.sqrt 19 :=
sorry

end three_c_plus_two_d_l801_801187


namespace find_BC_length_l801_801170

-- Declare variables for the lengths of the sides and the points.
variables (A B C D E M : Point) (AB BC AD CD ED EM BM : ℝ)

-- Given conditions in Lean 4 statement.
def given_conditions : Prop :=
  rectangle A B C D ∧
  E ∈ segment A D ∧
  M ∈ segment E C ∧
  AB = BM ∧
  AE = EM ∧
  length E D = 16 ∧
  length C D = 12

-- The theorem we want to prove.

theorem find_BC_length (h : given_conditions A B C D E M AB BC AD CD ED EM BM) : BC = 20 := sorry

end find_BC_length_l801_801170


namespace tablecloth_width_l801_801896

theorem tablecloth_width (length_tablecloth : ℕ) (napkins_count : ℕ) (napkin_length : ℕ) (napkin_width : ℕ) (total_material : ℕ) (width_tablecloth : ℕ) :
  length_tablecloth = 102 →
  napkins_count = 8 →
  napkin_length = 6 →
  napkin_width = 7 →
  total_material = 5844 →
  total_material = length_tablecloth * width_tablecloth + napkins_count * (napkin_length * napkin_width) →
  width_tablecloth = 54 :=
by
  intros h1 h2 h3 h4 h5 h_eq
  sorry

end tablecloth_width_l801_801896


namespace f_correct_l801_801232

-- Definitions of conditions
def A : ℝ := 2
def B : ℝ := 7
def ω : ℝ := π / 4
def φ : ℝ := -π / 4
def f (x : ℕ) : ℝ := A * Real.sin (ω * x + φ) + B

-- The proof statement
theorem f_correct :
  (∀ x, 1 ≤ x ∧ x ≤ 12 → ∃ y, y = f x) ∧ 
  (f 3 = 9000) ∧ 
  (f 7 = 5000) :=
by
  sorry

end f_correct_l801_801232


namespace mk_div_km_l801_801770

theorem mk_div_km 
  (m n k : ℕ) 
  (hm : 0 < m) 
  (hn : 0 < n) 
  (hk : 0 < k) 
  (h1 : m^n ∣ n^m) 
  (h2 : n^k ∣ k^n) : 
  m^k ∣ k^m := 
  sorry

end mk_div_km_l801_801770


namespace hiring_manager_acceptance_l801_801200

theorem hiring_manager_acceptance :
  let average_age := 31
  let std_dev := 9
  let max_diff_ages := 19
  let k := max_diff_ages / (2 * std_dev)
  k = 19 / 18 :=
by
  let average_age := 31
  let std_dev := 9
  let max_diff_ages := 19
  let k := max_diff_ages / (2 * std_dev)
  show k = 19 / 18
  sorry

end hiring_manager_acceptance_l801_801200


namespace concentration_of_resulting_mixture_l801_801662

-- Definitions based on conditions
def pure_water_volume : ℝ := 1
def salt_solution_volume : ℝ := 1
def salt_concentration_in_solution : ℝ := 0.40

-- Proof problem: Concentration of the resulting mixture
theorem concentration_of_resulting_mixture :
  let total_volume := pure_water_volume + salt_solution_volume,
      amount_of_salt := salt_solution_volume * salt_concentration_in_solution,
      resulting_concentration := amount_of_salt / total_volume
  in resulting_concentration = 0.20 :=
by
  sorry

end concentration_of_resulting_mixture_l801_801662


namespace athletes_leave_rate_l801_801291

theorem athletes_leave_rate (R : ℝ) (h : 300 - 4 * R + 105 = 307) : R = 24.5 :=
  sorry

end athletes_leave_rate_l801_801291


namespace sqrt_x2_plus_12x_plus_36_add_x_add_6_eq_0_has_infinite_solutions_l801_801220

theorem sqrt_x2_plus_12x_plus_36_add_x_add_6_eq_0_has_infinite_solutions :
  ∀ x : ℝ, (sqrt (x^2 + 12 * x + 36) + x + 6 = 0) ↔ (x = -6 ∨ x < -6) := by
  sorry

end sqrt_x2_plus_12x_plus_36_add_x_add_6_eq_0_has_infinite_solutions_l801_801220


namespace solution_l801_801981

variable {f : ℝ → ℝ}

-- f is continuous
axiom continuous_f : Continuous f
-- f satisfies the equation f(x + y) = f(x) + f(y)
axiom additivity : ∀ x y : ℝ, f (x + y) = f (x) + f (y)

theorem solution (C : ℝ) (hC : C = f 1) : ∀ x : ℝ, f x = C x :=
by
  sorry

end solution_l801_801981


namespace fence_section_count_l801_801140

theorem fence_section_count (posts_total : ℕ) (posts_fn : ℕ → ℕ) (n : ℕ)
  (h_posts_fn : ∀ n, posts_fn n = 2 * n + 1)
  (h_sum : ∑ i in finset.range n, posts_fn (i + 1) = posts_total) :
  posts_total = 435 → n = 21 :=
by {
  sorry
}

end fence_section_count_l801_801140


namespace root_product_identity_l801_801497

theorem root_product_identity (a b c : ℝ) (h1 : a * b * c = -8) (h2 : a * b + b * c + c * a = 20) (h3 : a + b + c = 15) :
    (1 + a) * (1 + b) * (1 + c) = 28 :=
by
  sorry

end root_product_identity_l801_801497


namespace relationship_abc_l801_801022

noncomputable def a (x : ℝ) (hx : 0 < x ∧ x < 1) : ℝ := (Real.sin x) / x
noncomputable def b (x : ℝ) (hx : 0 < x ∧ x < 1) : ℝ := (Real.sin (x^3)) / (x^3)
noncomputable def c (x : ℝ) (hx : 0 < x ∧ x < 1) : ℝ := ((Real.sin x)^3) / (x^3)

theorem relationship_abc (x : ℝ) (hx : 0 < x ∧ x < 1) : b x hx > a x hx ∧ a x hx > c x hx :=
by
  sorry

end relationship_abc_l801_801022


namespace exists_equidistant_points_l801_801953

namespace EquidistantPoints

-- Define points and a line in the plane
variables {Point : Type*} {Line : Type*}
variables (A B : Point) (l : Line)

-- Assumptions
axiom distinct_points : A ≠ B
axiom in_plane (p : Point) (l : Line) : Prop -- Here we assume that Point and Line are defined in the same plane and we have a function defining whether a point is in a plane

-- Definitions for equidistance and perpendicular bisector
def equidistant_from_point_and_line (p : Point) (l : Line) (A : Point) : Prop :=
  -- Assume a function that checks equidistance from a point and a line
  sorry

def equidistant_from_two_points (p : Point) (A B : Point) : Prop :=
  -- Assume a function that checks equidistance from two points
  sorry

def perpendicular_bisector (A B : Point) : Line :=
  -- Assume a function that returns the perpendicular bisector of segment AB
  sorry

-- The theorem we want to prove
theorem exists_equidistant_points
  (A B : Point) (l : Line)
  (hA_ne_B : A ≠ B) :
  ∃ (n : ℕ), n = 0 ∨ n = 1 ∨ n = 2 ∧
  (∀ (p : Point), (equidistant_from_point_and_line p l A → equidistant_from_point_and_line p l B → equidistant_from_two_points p A B) := sorry

end EquidistantPoints

end exists_equidistant_points_l801_801953


namespace penny_money_left_is_5_l801_801881

def penny_initial_money : ℤ := 20
def socks_pairs : ℤ := 4
def price_per_pair_of_socks : ℤ := 2
def price_of_hat : ℤ := 7

def total_cost_of_socks : ℤ := socks_pairs * price_per_pair_of_socks
def total_cost_of_hat_and_socks : ℤ := total_cost_of_socks + price_of_hat
def penny_money_left : ℤ := penny_initial_money - total_cost_of_hat_and_socks

theorem penny_money_left_is_5 : penny_money_left = 5 := by
  sorry

end penny_money_left_is_5_l801_801881


namespace last_two_videos_length_l801_801488

noncomputable def ad1 : ℕ := 45
noncomputable def ad2 : ℕ := 30
noncomputable def pause1 : ℕ := 45
noncomputable def pause2 : ℕ := 30
noncomputable def video1 : ℕ := 120
noncomputable def video2 : ℕ := 270
noncomputable def total_time : ℕ := 960

theorem last_two_videos_length : 
    ∃ v : ℕ, 
    v = 210 ∧ 
    total_time = ad1 + ad2 + video1 + video2 + pause1 + pause2 + 2 * v :=
by
  sorry

end last_two_videos_length_l801_801488


namespace price_difference_proof_l801_801105

variable (n : ℕ)

def price_X (n : ℕ) : ℝ := 4.20 + 0.40 * n
def price_Y (n : ℕ) : ℝ := 6.30 + 0.15 * n
def tax_rate (n : ℕ) : ℝ := 3 + 0.5 * n
def exchange_rate (n : ℕ) : ℝ := 2 - 0.02 * n

def final_price_X_LC (n : ℕ) : ℝ := (price_X n) * (1 + tax_rate n / 100)
def final_price_Y_LC (n : ℕ) : ℝ := (price_Y n) * (1 + tax_rate n / 100)

def final_price_X_FC (n : ℕ) : ℝ := final_price_X_LC n / exchange_rate n
def final_price_Y_FC (n : ℕ) : ℝ := final_price_Y_LC n / exchange_rate n

theorem price_difference_proof : ∃ n : ℕ,
    final_price_X_FC n - final_price_Y_FC n = 0.15 :=
sorry

end price_difference_proof_l801_801105


namespace exists_triangle_with_area_six_l801_801308

structure Triangle :=
  (a b c : ℕ)
  (perimeter_eq : a + b + c = 12)
  (triangle_ineq_1 : a + b > c)
  (triangle_ineq_2 : a + c > b)
  (triangle_ineq_3 : b + c > a)

noncomputable def area (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2 in
  (s * (s - t.a) * (s - t.b) * (s - t.c)).sqrt

theorem exists_triangle_with_area_six : ∃ t : Triangle, area t = 6 := by
  sorry

end exists_triangle_with_area_six_l801_801308


namespace tan_sum_pi_over_12_l801_801535

theorem tan_sum_pi_over_12 : 
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12)) = 4 := 
sorry

end tan_sum_pi_over_12_l801_801535


namespace cos_A_minus_B_l801_801384

variable {A B : ℝ}

-- Conditions
def cos_conditions (A B : ℝ) : Prop :=
  (Real.cos A + Real.cos B = 1 / 2)

def sin_conditions (A B : ℝ) : Prop :=
  (Real.sin A + Real.sin B = 3 / 2)

-- Mathematically equivalent proof problem
theorem cos_A_minus_B (h1 : cos_conditions A B) (h2 : sin_conditions A B) :
  Real.cos (A - B) = 1 / 4 := 
sorry

end cos_A_minus_B_l801_801384


namespace eve_total_spend_l801_801723

def hand_mitts_cost : ℝ := 14.00
def apron_cost : ℝ := 16.00
def utensils_cost : ℝ := 10.00
def knife_cost : ℝ := 2 * utensils_cost
def discount_percent : ℝ := 0.25
def nieces_count : ℕ := 3

def total_cost_before_discount : ℝ :=
  (hand_mitts_cost + apron_cost + utensils_cost + knife_cost) * nieces_count

def discount_amount : ℝ :=
  discount_percent * total_cost_before_discount

def total_cost_after_discount : ℝ :=
  total_cost_before_discount - discount_amount

theorem eve_total_spend : total_cost_after_discount = 135.00 := by
  sorry

end eve_total_spend_l801_801723


namespace find_f_neg1_l801_801393

-- Definition of odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Given conditions
variables (f : ℝ → ℝ) (h_odd : odd_function f) (h_f1 : f 1 = 2)

-- Theorem stating the necessary proof
theorem find_f_neg1 : f (-1) = -2 :=
by
  sorry

end find_f_neg1_l801_801393


namespace tan_sum_simplification_l801_801555
-- We start by importing the relevant Lean libraries that contain trigonometric functions and basic real analysis.

-- Define the statement to be proved in Lean.
theorem tan_sum_simplification :
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12) = 4 * Real.sqrt 2 - 4) :=
by
  sorry

end tan_sum_simplification_l801_801555


namespace roots_squared_sum_l801_801798

theorem roots_squared_sum (a b : ℝ) (h : a^2 - 8 * a + 8 = 0 ∧ b^2 - 8 * b + 8 = 0) : a^2 + b^2 = 48 := 
sorry

end roots_squared_sum_l801_801798


namespace tom_cost_cheaper_than_jane_l801_801948

def store_A_full_price : ℝ := 125
def store_A_discount_single : ℝ := 0.08
def store_A_discount_bulk : ℝ := 0.12
def store_A_tax_rate : ℝ := 0.07
def store_A_shipping_fee : ℝ := 10
def store_A_club_discount : ℝ := 0.05

def store_B_full_price : ℝ := 130
def store_B_discount_single : ℝ := 0.10
def store_B_discount_bulk : ℝ := 0.15
def store_B_tax_rate : ℝ := 0.05
def store_B_free_shipping_threshold : ℝ := 250
def store_B_club_discount : ℝ := 0.03

def tom_smartphones_qty : ℕ := 2
def jane_smartphones_qty : ℕ := 3

theorem tom_cost_cheaper_than_jane :
  let tom_cost := 
    let total := store_A_full_price * tom_smartphones_qty
    let discount := if tom_smartphones_qty ≥ 2 then store_A_discount_bulk else store_A_discount_single
    let price_after_discount := total * (1 - discount)
    let price_after_club_discount := price_after_discount * (1 - store_A_club_discount)
    let price_after_tax := price_after_club_discount * (1 + store_A_tax_rate) 
    price_after_tax + store_A_shipping_fee

  let jane_cost := 
    let total := store_B_full_price * jane_smartphones_qty
    let discount := if jane_smartphones_qty ≥ 3 then store_B_discount_bulk else store_B_discount_single
    let price_after_discount := total * (1 - discount)
    let price_after_club_discount := price_after_discount * (1 - store_B_club_discount)
    let price_after_tax := price_after_club_discount * (1 + store_B_tax_rate)
    let shipping_fee := if total > store_B_free_shipping_threshold then 0 else 0
    price_after_tax + shipping_fee
  
  jane_cost - tom_cost = 104.01 := 
by 
  sorry

end tom_cost_cheaper_than_jane_l801_801948


namespace find_n_l801_801048

-- Define the vectors \overrightarrow {AB}, \overrightarrow {BC}, and \overrightarrow {AC}
def vectorAB : ℝ × ℝ := (2, 4)
def vectorBC (n : ℝ) : ℝ × ℝ := (-2, 2 * n)
def vectorAC : ℝ × ℝ := (0, 2)

-- State the theorem and prove the value of n
theorem find_n (n : ℝ) (h : vectorAC = (vectorAB.1 + (vectorBC n).1, vectorAB.2 + (vectorBC n).2)) : n = -1 :=
by
  sorry

end find_n_l801_801048


namespace base8_integers_with_6_or_7_count_l801_801073

theorem base8_integers_with_6_or_7_count : 
  let numbers_in_base8_range := range 512
  let numbers_containing_6_or_7 := numbers_in_base8_range.filter (λ n, (6 ∈ n.digits 8) ∨ (7 ∈ n.digits 8))
  numbers_containing_6_or_7.length = 297 :=
begin
  sorry
end

end base8_integers_with_6_or_7_count_l801_801073


namespace andrew_subway_time_l801_801622

variable (S : ℝ) -- Let \( S \) be the time Andrew spends on the subway in hours

variable (total_time : ℝ)
variable (bike_time : ℝ)
variable (train_time : ℝ)

noncomputable def travel_conditions := 
  total_time = S + 2 * S + bike_time ∧ 
  total_time = 38 ∧ 
  bike_time = 8

theorem andrew_subway_time
  (S : ℝ)
  (total_time : ℝ)
  (bike_time : ℝ)
  (train_time : ℝ)
  (h : travel_conditions S total_time bike_time) : 
  S = 10 := 
sorry

end andrew_subway_time_l801_801622


namespace max_points_on_line_l801_801988

theorem max_points_on_line (A B C : Point) (ℓ : Line) (X : Set Point) :
  (ℓ ∈ intersects_segment A B C) →
  (∀ X ∈ ℓ, ∃ α : ℝ, ∠ B X C = 2 * ∠ A X C) →
  X.card = 4 :=
begin
  sorry
end

end max_points_on_line_l801_801988


namespace jose_profit_share_l801_801276

theorem jose_profit_share :
  ∀ (Tom_investment Jose_investment total_profit month_investment_tom month_investment_jose total_month_investment: ℝ),
    Tom_investment = 30000 →
    ∃ (months_tom months_jose : ℝ), months_tom = 12 ∧ months_jose = 10 →
      Jose_investment = 45000 →
      total_profit = 72000 →
      month_investment_tom = Tom_investment * months_tom →
      month_investment_jose = Jose_investment * months_jose →
      total_month_investment = month_investment_tom + month_investment_jose →
      (Jose_investment * months_jose / total_month_investment) * total_profit = 40000 :=
by
  sorry

end jose_profit_share_l801_801276


namespace selection_ways_l801_801689

-- The statement of the problem in Lean 4
theorem selection_ways :
  (Nat.choose 50 4) - (Nat.choose 47 4) = 
  (Nat.choose 3 1) * (Nat.choose 47 3) + 
  (Nat.choose 3 2) * (Nat.choose 47 2) + 
  (Nat.choose 3 3) * (Nat.choose 47 1) := 
sorry

end selection_ways_l801_801689


namespace smallest_positive_period_cos2x_minus_sin2x_l801_801713

theorem smallest_positive_period_cos2x_minus_sin2x : 
  ∃ T : ℝ, (T > 0) ∧ (∀ x : ℝ, cos (x + T) ^ 2 - sin (x + T) ^ 2 = cos x ^ 2 - sin x ^ 2) ∧ (T = π) :=
sorry

end smallest_positive_period_cos2x_minus_sin2x_l801_801713


namespace tangent_points_l801_801216

noncomputable def curve (x : ℝ) : ℝ := x^3 - x - 1

theorem tangent_points (x y : ℝ) (h : y = curve x) (slope_line : ℝ) (h_slope : slope_line = -1/2)
  (tangent_perpendicular : (3 * x^2 - 1) = 2) :
  (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -1) := sorry

end tangent_points_l801_801216


namespace area_ratio_of_triangle_and_parallelogram_l801_801114

theorem area_ratio_of_triangle_and_parallelogram 
  (AB CD : ℝ) (h_parallel : AB ∥ CD) (h_AB : AB = 10) (h_CD : CD = 21)
  (E : Point) (A B C D : Point) (parallelogram_ABCD : Parallelogram ABCD)
  (E_on_ext_AC_BD : E ∈ (Line AC ∪ Line BD)) :
  area_ratio (triangle EAB) (parallelogram ABCD) = 100 / 341 := 
sorry

end area_ratio_of_triangle_and_parallelogram_l801_801114


namespace chlorine_needed_l801_801359

variable (Methane moles_HCl moles_Cl₂ : ℕ)

-- Given conditions
def reaction_started_with_one_mole_of_methane : Prop :=
  Methane = 1

def reaction_produces_two_moles_of_HCl : Prop :=
  moles_HCl = 2

-- Question to be proved
def number_of_moles_of_Chlorine_combined : Prop :=
  moles_Cl₂ = 2

theorem chlorine_needed
  (h1 : reaction_started_with_one_mole_of_methane Methane)
  (h2 : reaction_produces_two_moles_of_HCl moles_HCl)
  : number_of_moles_of_Chlorine_combined moles_Cl₂ :=
sorry

end chlorine_needed_l801_801359


namespace largest_m_l801_801011

def pow (n : ℕ) : ℕ :=
  let p := (nat.factorization n).max_key id
  p ^ ((nat.factorization n).find p)

def product_pow (a b : ℕ) : ℕ :=
  (finset.range (b + 1)).filter (λ x, x ≥ a).prod pow

theorem largest_m (m : ℕ) :
  let n_range := finset.range (7001).filter (λ x, x ≥ 2)
  let L := finset.fold (λ acc n, acc * pow n) 1 n_range
  (4620 ^ m) ∣ L ↔ m = 698 :=
sorry

end largest_m_l801_801011


namespace exist_pos_poly_l801_801143

theorem exist_pos_poly (n : ℕ) (h : n > 0) :
  ∃ P Q : ℝ[X], (∀ x : ℝ, 0 ≤ P.eval x) ∧
                (∀ x : ℝ, 0 ≤ Q.eval x) ∧
                (∀ x : ℝ, 1 - x^n = (1 - x) * P.eval x + (1 + x) * Q.eval x) :=
by
  sorry

end exist_pos_poly_l801_801143


namespace soda_cans_purchase_l801_801567

-- Definition of conditions
def cansPerQuarter := 7 / 10
def discount := 0.10
def dollarsToQuarters (d: ℕ) : ℕ := d * 4
def quartersToCans (q: ℕ) : ℕ := (7 * q) / 10

-- Theorem statement: Given the provided conditions, prove the number of cans for $20 is 61, accounting for discount
theorem soda_cans_purchase: 
  let q := dollarsToQuarters 20 in
  let cans_without_discount := quartersToCans q in
  let cans_with_discount := (cans_without_discount * (1 + discount)) in
  q ≤ 80 ∧ q > 20 → 
  floor cans_with_discount = 61 :=
by
  sorry

end soda_cans_purchase_l801_801567


namespace minimum_segments_two_l801_801876

noncomputable def minimum_segments_coincide_edges (cube : Type) [TopologicalSpace cube] : ℕ :=
  sorry

theorem minimum_segments_two {cube : Type} [TopologicalSpace cube] :
  (∃ broken_line : ℕ → cube, 
  ∀ i, i < 8 → 
  (broken_line (i + 1) = broken_line i ∨ -- Segment is either an edge
   ∃ face : Set cube, broken_line i ∈ face ∧ broken_line (i + 1) ∈ face)) -- or a diagonal
   ∧ broken_line 8 = broken_line 0) -- and it's a closed broken line
  → minimum_segments_coincide_edges cube = 2 :=
sorry

end minimum_segments_two_l801_801876


namespace number_of_puppies_l801_801613

theorem number_of_puppies (P K : ℕ) (h1 : K = 2 * P + 14) (h2 : K = 78) : P = 32 :=
by sorry

end number_of_puppies_l801_801613


namespace triangle_area_l801_801121

variables {A B C a b c : ℝ}

/-- In triangle ABC, the sides opposite to angles A, B, and C are denoted as a, b, and c, respectively.
It is given that b * sin C + c * sin B = 4 * a * sin B * sin C and b^2 + c^2 - a^2 = 8.
Prove that the area of triangle ABC is 4 * sqrt 3 / 3. -/
theorem triangle_area (h1 : b * Real.sin C + c * Real.sin B = 4 * a * Real.sin B * Real.sin C)
  (h2 : b^2 + c^2 - a^2 = 8) :
  (1 / 2) * b * c * Real.sin A = 4 * Real.sqrt 3 / 3 :=
sorry

end triangle_area_l801_801121


namespace max_t_value_min_y_value_l801_801092

-- Part (1)
theorem max_t_value (x t : ℝ) (h : ∀ x : ℝ, |3*x + 2| + |3*x - 1| - t ≥ 0) : t ≤ 3 := 
by {
  -- Proof sketch: \[ \forall x, |3x + 2| + |3x - 1| \geq 3 \]
  sorry
}

-- Part (2)
theorem min_y_value (m n : ℝ) 
  (h₀ : 0 < m) (h₁ : 0 < n) (h₂ : 4*m + 5*n = 3) : 
  let y := 1/(m + 2*n) + 4/(3*m + 3*n) in y ≥ 3 :=
by {
  -- Proof sketch: AM-GM inequality
  sorry
}

end max_t_value_min_y_value_l801_801092


namespace value_of_kaftan_l801_801989

theorem value_of_kaftan (K : ℝ) (h : (7 / 12) * (12 + K) = 5 + K) : K = 4.8 :=
by
  sorry

end value_of_kaftan_l801_801989


namespace order_of_means_l801_801416

variables (a b : ℝ)
-- a and b are positive and unequal
axiom h1 : 0 < a
axiom h2 : 0 < b
axiom h3 : a ≠ b

-- Definitions of the means
noncomputable def AM : ℝ := (a + b) / 2
noncomputable def GM : ℝ := Real.sqrt (a * b)
noncomputable def HM : ℝ := (2 * a * b) / (a + b)
noncomputable def QM : ℝ := Real.sqrt ((a^2 + b^2) / 2)

-- The theorem to prove the order of the means
theorem order_of_means (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) :
  QM a b > AM a b ∧ AM a b > GM a b ∧ GM a b > HM a b :=
sorry

end order_of_means_l801_801416


namespace hot_water_bottles_sold_l801_801952

theorem hot_water_bottles_sold :
  ∃ (H : ℕ), 
    let T := 7 * H in 
    2 * T + 6 * H = 1200 ∧ H = 60 :=
by
  sorry

end hot_water_bottles_sold_l801_801952


namespace shortest_chord_length_l801_801773

theorem shortest_chord_length
  (x y : ℝ)
  (hx : x^2 + y^2 - 6 * x - 8 * y = 0)
  (point_on_circle : (3, 5) = (x, y)) :
  ∃ (length : ℝ), length = 4 * Real.sqrt 6 := 
by
  sorry

end shortest_chord_length_l801_801773


namespace rectangle_length_l801_801087

theorem rectangle_length (side_length_square : ℝ) (width_rectangle : ℝ) (area_equal : ℝ) 
  (square_area : side_length_square * side_length_square = area_equal) 
  (rectangle_area : width_rectangle * (width_rectangle * length) = area_equal) : 
  length = 24 :=
by 
  sorry

end rectangle_length_l801_801087


namespace factorization_correct_l801_801221

theorem factorization_correct:
  ∃ a b : ℤ, (25 * x^2 - 85 * x - 150 = (5 * x + a) * (5 * x + b)) ∧ (a + 2 * b = -24) :=
by
  sorry

end factorization_correct_l801_801221


namespace Helly_theorem_plane_l801_801652

theorem Helly_theorem_plane {n : ℕ} (h_n : n ≥ 3) (M : Fin n → Set ℝ²)
  (h_convex : ∀ i, Convex ℝ (M i))
  (h_intersect : ∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k → (M i ∩ M j ∩ M k).Nonempty) :
  (⋂ i, M i).Nonempty :=
sorry

end Helly_theorem_plane_l801_801652


namespace problem_solution_l801_801193

variable {x y z : ℝ}

/-- Suppose that x, y, and z are three positive numbers that satisfy the given conditions.
    Prove that z + 1/y = 13/77. --/
theorem problem_solution (h1 : x * y * z = 1)
                         (h2 : x + 1 / z = 8)
                         (h3 : y + 1 / x = 29) :
  z + 1 / y = 13 / 77 := 
  sorry

end problem_solution_l801_801193


namespace pencils_in_drawer_l801_801614

/-- 
If there were originally 2 pencils in the drawer and there are now 5 pencils in total, 
then Tim must have placed 3 pencils in the drawer.
-/
theorem pencils_in_drawer (original_pencils tim_pencils total_pencils : ℕ) 
  (h1 : original_pencils = 2) 
  (h2 : total_pencils = 5) 
  (h3 : total_pencils = original_pencils + tim_pencils) : 
  tim_pencils = 3 := 
by
  rw [h1, h2] at h3
  linarith

end pencils_in_drawer_l801_801614


namespace determine_b_value_l801_801404

theorem determine_b_value 
  (a : ℝ) 
  (b : ℝ) 
  (h₀ : a > 0) 
  (h₁ : a ≠ 1) 
  (h₂ : 2 * a^(2 - b) + 1 = 3) : 
  b = 2 := 
by 
  sorry

end determine_b_value_l801_801404


namespace range_of_a_eq_2_sqrt_5_plus_infty_l801_801764

-- Definitions and assumptions
def proposition_P (a : ℝ) : Prop :=
  ∀ x : ℝ, 2 < x ∧ x < 3 → x^2 + 5 ≤ a * x

-- Goal
theorem range_of_a_eq_2_sqrt_5_plus_infty :
  {a : ℝ | ¬ (proposition_P a)} = Set.Ici (2 * Real.sqrt 5) :=
begin
  sorry
end

end range_of_a_eq_2_sqrt_5_plus_infty_l801_801764


namespace decimal_to_binary_93_to_1011101_l801_801703

theorem decimal_to_binary_93_to_1011101 : let rec toBinary (n : ℕ) (acc : List ℕ) : List ℕ :=
  if h : n = 0 then acc.reverse else
  have : n / 2 < n := Nat.div_lt_self (by decide) (by decide)
  toBinary (n / 2) ((n % 2) :: acc)
in toBinary 93 [] = [1, 0, 1, 1, 1, 0, 1] :=
by
  let rec toBinary (n : ℕ) (acc : List ℕ) : List ℕ :=
    if h : n = 0 then acc.reverse else
    have : n / 2 < n := Nat.div_lt_self (by decide) (by decide)
    toBinary (n / 2) ((n % 2) :: acc)
  show toBinary 93 [] = [1, 0, 1, 1, 1, 0, 1]
  sorry

end decimal_to_binary_93_to_1011101_l801_801703


namespace penny_money_left_is_5_l801_801880

def penny_initial_money : ℤ := 20
def socks_pairs : ℤ := 4
def price_per_pair_of_socks : ℤ := 2
def price_of_hat : ℤ := 7

def total_cost_of_socks : ℤ := socks_pairs * price_per_pair_of_socks
def total_cost_of_hat_and_socks : ℤ := total_cost_of_socks + price_of_hat
def penny_money_left : ℤ := penny_initial_money - total_cost_of_hat_and_socks

theorem penny_money_left_is_5 : penny_money_left = 5 := by
  sorry

end penny_money_left_is_5_l801_801880


namespace polynomial_integer_roots_l801_801603

theorem polynomial_integer_roots (a b c : ℚ) (p q : ℤ) :
  (polynomial.eval (2 - real.sqrt 3) (polynomial.X ^ 4 + (a : ℝ) * polynomial.X ^ 2 + (b : ℝ) * polynomial.X + (c : ℝ)) = 0) ∧
  (polynomial.eval (2 + real.sqrt 3) (polynomial.X ^ 4 + (a : ℝ) * polynomial.X ^ 2 + (b : ℝ) * polynomial.X + (c : ℝ)) = 0) ∧
  polynomial.eval (p : ℝ) (polynomial.X ^ 4 + (a : ℝ) * polynomial.X ^ 2 + (b : ℝ) * polynomial.X + (c : ℝ)) = 0 ∧
  polynomial.eval (q : ℝ) (polynomial.X ^ 4 + (a : ℝ) * polynomial.X ^ 2 + (b : ℝ) * polynomial.X + (c : ℝ)) = 0 ∧
  p ≠ q ∧
  (p + q = -4) →
  (p = -1 ∧ q = -3) ∨ (p = -3 ∧ q = -1) :=
sorry

end polynomial_integer_roots_l801_801603


namespace days_to_clear_land_l801_801261

-- Definitions of all the conditions
def length_of_land := 200
def width_of_land := 900
def area_cleared_by_one_rabbit_per_day_square_yards := 10
def number_of_rabbits := 100
def conversion_square_yards_to_square_feet := 9
def total_area_of_land := length_of_land * width_of_land
def area_cleared_by_one_rabbit_per_day_square_feet := area_cleared_by_one_rabbit_per_day_square_yards * conversion_square_yards_to_square_feet
def area_cleared_by_all_rabbits_per_day := number_of_rabbits * area_cleared_by_one_rabbit_per_day_square_feet

-- Theorem to prove the number of days required to clear the land
theorem days_to_clear_land :
  total_area_of_land / area_cleared_by_all_rabbits_per_day = 20 := by
  sorry

end days_to_clear_land_l801_801261


namespace general_term_formula_tn_sum_formula_l801_801067

-- Given sequence definition
def S (n : ℕ) : ℕ := n^2 + 2 * n

-- General formula for the n-th term a_n
def a : ℕ → ℕ
| 1       := 3
| (n + 1) := S (n + 1) - S n

-- General formula to be proved: a_n = 2n + 1
theorem general_term_formula (n : ℕ) (h_n : n ≠ 0) : a n = 2 * n + 1 := sorry

-- Definition of b_n based on a_n
def b (n : ℕ) : ℕ := n

-- Definition of T_n
def T (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, (1 / (b k * b (k + 1)))

-- Sum formula to be proved: T_n = n / (n + 1)
theorem tn_sum_formula (n : ℕ) : T n = n / (n + 1) := sorry

end general_term_formula_tn_sum_formula_l801_801067


namespace sum_of_inradii_l801_801979

theorem sum_of_inradii (ABC : Triangle) (r r1 r2 r3 : ℝ)
  (h1 : ABC.inscribed_circle_radius = r)
  (h2 : ∀ Δ, Δ ∈ ABC.smaller_triangles_by_tangents → Δ.inscribed_circle_radius ∈ {r1, r2, r3}) :
  r1 + r2 + r3 = r := sorry

end sum_of_inradii_l801_801979


namespace zadam_win_probability_l801_801639

theorem zadam_win_probability :
  ∃ p : ℚ, p = 1 / 2 ∧
  (∃ (num_attempts success_attempts : ℕ), num_attempts ≥ 5 ∧ num_attempts ≤ 9 ∧ success_attempts = 5 ∧
    (∀ k : ℕ, k < num_attempts → 
      ∃ successful_rolls : finset (fin num_attempts), successful_rolls.card = success_attempts ∧
        (∀ i : fin num_attempts, i ∈ successful_rolls → 
          (1/2 : ℚ)) ∧
        (probability (λ outcome : finset (fin num_attempts), ∃ win_set, 
          win_set = successful_rolls ∧ win_set.card = success_attempts ∧ win_set.sum (λ x, (1/2 : ℚ)) = p
        ) = 1 / 2)
     )
  )

end zadam_win_probability_l801_801639


namespace solution_set_of_inequality_l801_801984

noncomputable def f : ℝ → ℝ := sorry

axiom f_deriv_gt_f : ∀ x : ℝ, deriv f x > f x
axiom f_at_zero : f 0 = 1

theorem solution_set_of_inequality : {x : ℝ | f x < Real.exp x} = Iio 0 := 
by 
  sorry

end solution_set_of_inequality_l801_801984


namespace trigonometric_identity_l801_801962

theorem trigonometric_identity (α : ℝ) :
    1 - 1/4 * (Real.sin (2 * α)) ^ 2 + Real.cos (2 * α) = (Real.cos α) ^ 2 + (Real.cos α) ^ 4 :=
by
  sorry

end trigonometric_identity_l801_801962


namespace bridge_length_correct_l801_801684

-- Define the given conditions
constant train_length : ℝ := 110
constant train_speed_kmph : ℝ := 72
constant crossing_time : ℝ := 12.099

-- Define the conversion function from km/h to m/s
def kmph_to_mps (speed: ℝ) : ℝ :=
  speed * (1000 / 3600)

-- Define the speed in m/s
constant train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

-- Define the total distance covered by the train
def total_distance_covered (speed: ℝ) (time: ℝ) : ℝ :=
  speed * time

-- Define the bridge length
def bridge_length (total_distance : ℝ) (train_length : ℝ) : ℝ :=
  total_distance - train_length

-- Prove that the bridge length is 131.98 
theorem bridge_length_correct :
  bridge_length (total_distance_covered train_speed_mps crossing_time) train_length = 131.98 := 
sorry

end bridge_length_correct_l801_801684


namespace unique_side_order_l801_801299

noncomputable def quadrilateral_sides := {a : ℝ // a = 5 ∨ a = 6 ∨ a = 8 ∨ a = 11}
noncomputable def quadrilateral_diagonals := {d : ℝ // d = 4.7 ∨ d = 7 ∨ d = 13.96}

-- Proving the uniqueness of side order given provided diagonals
theorem unique_side_order (sides : quadrilateral_sides) (diagonals : quadrilateral_diagonals)
(h1 : sides = {5, 6, 8, 11})
(h2 : diagonals = {4.7, 7, 13.96}) :
(sides = {5, 6, 8, 11} ∧ 
 (angles = {116.4, 49.7, 171.5, 22.4}) ∧ 
 (radius_of_smallest_covering_circle = 6.98)) :=
begin
  sorry
end

end unique_side_order_l801_801299


namespace seq_10001_satisfies_C_l801_801655

-- Definitions from conditions
def is_zero_one_sequence (a : ℕ → ℕ) : Prop := ∀ i, a i ∈ {0, 1}

def has_period (a : ℕ → ℕ) (m : ℕ) : Prop := ∀ i, a (i + m) = a i

def C (a : ℕ → ℕ) (m k : ℕ) : ℚ :=
  (1 / m : ℚ) * (∑ i in Finset.range m, a i * a (i + k))

noncomputable def satisfies_C (a : ℕ → ℕ) (m : ℕ) : Prop :=
  ∀ k, k ∈ Finset.range (m - 1) → C a m k ≤ (1 / m : ℚ)

-- Sequence 10001 repeating
def seq_10001 : ℕ → ℕ
| 0 := 1
| 1 := 0
| 2 := 0
| 3 := 0
| 4 := 1
| (n + 5) := seq_10001 n

-- Theorem statement
theorem seq_10001_satisfies_C :
  is_zero_one_sequence seq_10001 ∧ has_period seq_10001 5 → satisfies_C seq_10001 5 := 
by sorry

end seq_10001_satisfies_C_l801_801655


namespace combined_experience_l801_801832

noncomputable def james_experience : ℕ := 20
noncomputable def john_experience_8_years_ago : ℕ := 2 * (james_experience - 8)
noncomputable def john_current_experience : ℕ := john_experience_8_years_ago + 8
noncomputable def mike_experience : ℕ := john_current_experience - 16

theorem combined_experience :
  james_experience + john_current_experience + mike_experience = 68 :=
by
  sorry

end combined_experience_l801_801832


namespace dice_sum_probability_lt_10_l801_801253

theorem dice_sum_probability_lt_10 {n : ℕ} (h1 : 1 ≤ n) (h2 : n ≤ 6) :
  let pairs := [(1, 1), (1, 2), (2, 1), (1, 3), (2, 2), (3, 1), (1, 4), (2, 3), (3, 2), (4, 1),
                (1, 5), (2, 4), (3, 3), (4, 2), (5, 1), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1),
                (2, 6), (3, 5), (4, 4), (5, 3), (6, 2), (3, 6), (4, 5), (5, 4), (6, 3)] in
  let total_outcomes := 6 * 6 in
  (pairs.length : ℕ) / total_outcomes = 5/6 :=
sorry

end dice_sum_probability_lt_10_l801_801253


namespace simplify_tangent_sum_l801_801538

theorem simplify_tangent_sum :
  tan (Real.pi / 12) + tan (5 * Real.pi / 12) = Real.sqrt 6 - Real.sqrt 2 := 
sorry

end simplify_tangent_sum_l801_801538


namespace min_sets_cover_plane_l801_801759

/-- Define the set of points whose distance from a given point is irrational. -/
def SP (P : ℝ × ℝ) : set (ℝ × ℝ) :=
  {Q : ℝ × ℝ | irrational (dist P Q)}

/-- Lean statement for the proof problem: the smallest number of sets SP required to cover the entire plane is 3. -/
theorem min_sets_cover_plane : ∃ P Q R : ℝ × ℝ, 
  (SP P ∪ SP Q ∪ SP R) = set.univ :=
sorry

end min_sets_cover_plane_l801_801759


namespace tan_sum_pi_over_12_eq_4_l801_801547

theorem tan_sum_pi_over_12_eq_4 :
  tan (π / 12) + tan (5 * π / 12) = 4 := 
by
  have cos_pi_over_12 : cos (π / 12) = (real.sqrt 6 + real.sqrt 2) / 4 := sorry
  have cos_5pi_over_12 : cos (5 * π / 12) = (real.sqrt 6 - real.sqrt 2) / 4 := sorry
  have sin_pi_over_2 : sin (π / 2) = 1 := by
    exact real.sin_pi_div_two
  sorry

end tan_sum_pi_over_12_eq_4_l801_801547


namespace simplify_expression_l801_801522

theorem simplify_expression (z : ℝ) : (5 - 2*z^2) - (4*z^2 - 7) = 12 - 6*z^2 :=
by
  sorry

end simplify_expression_l801_801522


namespace smallest_multiple_of_1_to_10_l801_801955

theorem smallest_multiple_of_1_to_10 : 
  ∃ n : ℕ, (∀ x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, x ∣ n) ∧ n = 2520 :=
by
  sorry

end smallest_multiple_of_1_to_10_l801_801955


namespace find_a_of_symmetric_about_solve_inequality_for_f_l801_801587

-- Given that the function f(x) is symmetric about the point (1, 0), prove that a = 1/4
theorem find_a_of_symmetric_about {a : ℝ} (h : ∀ x, (1 / (2^x - 2) + a) = (1 / (2^(2 - x) - 2) + a)) :
  a = 1 / 4 :=
by sorry

-- Given a = 1/4, solve the inequality f(x) < 5/4
theorem solve_inequality_for_f {x : ℝ} (h : ∀ x, f x = 1 / (2^x - 2) + 1 / 4) :
  f x < 5 / 4 ↔ x > log 2 3 ∨ x < 1 :=
by sorry

end find_a_of_symmetric_about_solve_inequality_for_f_l801_801587


namespace binom_10_4_eq_210_l801_801701

theorem binom_10_4_eq_210 : Nat.choose 10 4 = 210 :=
  by sorry

end binom_10_4_eq_210_l801_801701


namespace joe_paint_usage_l801_801835

/--
Joe buys 720 gallons of paint and uses it over four weeks as follows:
1. During the first week, he uses 2/7 of all the paint.
2. During the second week, he uses 3/8 of the remaining paint.
3. In the third week, he uses 5/11 of the remaining paint.
4. In the fourth week, he uses 4/13 of what's left.

Prove that Joe has used approximately 598.620 gallons of paint after the fourth week.
-/
theorem joe_paint_usage :
  ∃ (total_paint_used : ℝ), total_paint_used ≈ 598.620 ∧ 
  (let initial_paint := 720.0 in
   let paint_first_week := 2/7 * initial_paint in
   let remaining_after_first := initial_paint - paint_first_week in
   let paint_second_week := 3/8 * remaining_after_first in
   let remaining_after_second := remaining_after_first - paint_second_week in
   let paint_third_week := 5/11 * remaining_after_second in
   let remaining_after_third := remaining_after_second - paint_third_week in
   let paint_fourth_week := 4/13 * remaining_after_third in
   let total_paint_used := paint_first_week + paint_second_week + paint_third_week + paint_fourth_week in
   true) sorry

end joe_paint_usage_l801_801835


namespace geometric_sum_S30_l801_801934

theorem geometric_sum_S30 (S : ℕ → ℝ) (h1 : S 10 = 10) (h2 : S 20 = 30) : S 30 = 70 := 
by 
  sorry

end geometric_sum_S30_l801_801934


namespace fans_received_all_items_l801_801368

theorem fans_received_all_items (n : ℕ) (h1 : (∀ k : ℕ, k * 45 ≤ n → (k * 45) ∣ n))
                                (h2 : (∀ k : ℕ, k * 50 ≤ n → (k * 50) ∣ n))
                                (h3 : (∀ k : ℕ, k * 100 ≤ n → (k * 100) ∣ n))
                                (capacity_full : n = 5000) :
  n / Nat.lcm 45 (Nat.lcm 50 100) = 5 :=
by
  sorry

end fans_received_all_items_l801_801368


namespace hyperbola_asymptote_check_l801_801634

def hasAsymptote (a b : ℝ) (h : ℝ) (k : ℝ → ℝ → Prop) := 
    k (a / b) (b / a)

theorem hyperbola_asymptote_check :
    (hasAsymptote 2 3 (λ x y, x = y ∨ x = -y) (λ x y, x*y=x ∨ x*y=-x)) → 
    (hasAsymptote 2 3 (λ x y, x = y ∨ x = -y) (λ x y, y*x=y ∨ y*x=-y)) → 
    ¬ (hasAsymptote 2 3 (λ x y, x = y ∨ x = -y) (λ x y, y^2/4 - x^2/9 = 0)) →
    (hasAsymptote 2 3 (λ x y, x = y ∨ x = -y) (λ x y, y^2/12 - x^2/27 = 0)) → True := sorry

end hyperbola_asymptote_check_l801_801634


namespace pieces_bound_l801_801463

open Finset

variable {n : ℕ} (B W : ℕ)

theorem pieces_bound (n : ℕ) (B W : ℕ) (hB : B ≤ n^2) (hW : W ≤ n^2) :
    B ≤ n^2 ∨ W ≤ n^2 := 
by
  sorry

end pieces_bound_l801_801463


namespace bianca_mean_correct_l801_801248

noncomputable def bianca_mean : ℝ :=
let total_sum := 80 + 82 + 88 + 90 + 91 + 95 in
let alain_mean := 86 in
let alain_scores_sum := 3 * alain_mean in
let bianca_scores_sum := total_sum - alain_scores_sum in
bianca_scores_sum / 3

theorem bianca_mean_correct : bianca_mean = 89 := by
  sorry

end bianca_mean_correct_l801_801248


namespace kimberly_strawberries_times_brother_l801_801487

theorem kimberly_strawberries_times_brother :
  (x times_brother_strawberries : ℕ) -- Define x as the times Kimberly picked strawberries relative to her brother
  (brother_strawberries : ℕ = 3 * 15) -- Brother picked 3 baskets each containing 15 strawberries
  (kimberly_strawberries : ℕ = x * brother_strawberries) -- Kimberly picked x times the amount her brother picked
  (parents_strawberries : ℕ = kimberly_strawberries - 93) -- Kimberly's parents picked 93 strawberries less than her
  (total_strawberries : ℕ = 672) -- Total strawberries is 672 
  (equal_strawberries_per_person : total_strawberries / 4 = 168) -- when divided equally, each family member gets 168 strawberries
  → x = 8 := 
sorry

end kimberly_strawberries_times_brother_l801_801487


namespace complex_point_coordinates_l801_801910

theorem complex_point_coordinates
  (z : ℂ)
  (H : z * complex.I = abs (1 / 2 - (real.sqrt 3) / 2 * complex.I)) :
    z = -complex.I := by
  sorry

end complex_point_coordinates_l801_801910


namespace penny_remaining_money_l801_801883

theorem penny_remaining_money (initial_money : ℤ) (socks_pairs : ℤ) (socks_cost_per_pair : ℤ) (hat_cost : ℤ) :
  initial_money = 20 → socks_pairs = 4 → socks_cost_per_pair = 2 → hat_cost = 7 → 
  initial_money - (socks_pairs * socks_cost_per_pair + hat_cost) = 5 := 
by
  intros h₁ h₂ h₃ h₄
  sorry

end penny_remaining_money_l801_801883


namespace simplify_and_evaluate_l801_801900

theorem simplify_and_evaluate :
  let x := 2 * Real.sqrt 3
  (x - Real.sqrt 2) * (x + Real.sqrt 2) + x * (x - 1) = 22 - 2 * Real.sqrt 3 := 
by
  let x := 2 * Real.sqrt 3
  sorry

end simplify_and_evaluate_l801_801900


namespace factorize_polynomial_l801_801239

variable (a x y : ℝ)

theorem factorize_polynomial (a x y : ℝ) :
  3 * a * x ^ 2 - 3 * a * y ^ 2 = 3 * a * (x + y) * (x - y) := by
  sorry

end factorize_polynomial_l801_801239


namespace geometric_seq_sum_bound_l801_801783

def seq_a (n : ℕ) : ℕ → ℝ
| 1       := 1 / 2
| (n + 1) := 2 * seq_a n / (1 + seq_a n)

def seq_b (n : ℕ) : ℕ → ℝ
| n := n / seq_a n

def sum_b (n : ℕ) : ℝ := (finset.range n).sum seq_b

theorem geometric_seq (n : ℕ) (h : n > 0) :
  ∃ (r : ℝ) (c : ℕ), (∀ n > 0, let x := (1 / seq_a n) - 1 in x = c * r^(n-1)) ∧
  (seq_a n = 2 ^ (n - 1) / (1 + 2 ^ (n - 1))) :=
sorry

theorem sum_bound (n : ℕ) (h : n ≥ 3) :
  sum_b n > (n^2)/2 + 4 :=
sorry

end geometric_seq_sum_bound_l801_801783


namespace tan_sum_simplification_l801_801554
-- We start by importing the relevant Lean libraries that contain trigonometric functions and basic real analysis.

-- Define the statement to be proved in Lean.
theorem tan_sum_simplification :
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12) = 4 * Real.sqrt 2 - 4) :=
by
  sorry

end tan_sum_simplification_l801_801554


namespace employee_n_weekly_wage_l801_801568

theorem employee_n_weekly_wage (Rm Rn : ℝ) (Hm Hn : ℝ) 
    (h1 : (Rm * Hm) + (Rn * Hn) = 770) 
    (h2 : (Rm * Hm) = 1.3 * (Rn * Hn)) :
    Rn * Hn = 335 :=
by
  sorry

end employee_n_weekly_wage_l801_801568


namespace sum_of_special_integers_l801_801947

theorem sum_of_special_integers :
  let m := 2 in   -- representing the smallest prime number
  let n := 121 in -- representing the largest integer < 150 with exactly three positive divisors
  m + n = 123 :=
by
  sorry

end sum_of_special_integers_l801_801947


namespace ball_distribution_l801_801424

theorem ball_distribution (n k : ℕ) (h₁ : n = 5) (h₂ : k = 4) :
  k^n = 1024 := by
  rw [h₁, h₂]
  norm_num
  sorry

end ball_distribution_l801_801424


namespace total_pages_eq_l801_801421

theorem total_pages_eq :
  ∀ (pages_per_booklet : ℕ) (number_of_booklets : ℕ),
    pages_per_booklet = 9 → number_of_booklets = 49 → 
    pages_per_booklet * number_of_booklets = 441 :=
by
  intros pages_per_booklet number_of_booklets h1 h2
  rw [h1, h2]
  norm_num

end total_pages_eq_l801_801421


namespace concert_songs_l801_801451

def total_songs (g : ℕ) : ℕ := (9 + 3 + 9 + g) / 3

theorem concert_songs 
  (g : ℕ) 
  (h1 : 9 + 3 + 9 + g = 3 * total_songs g) 
  (h2 : 3 + g % 4 = 0) 
  (h3 : 4 ≤ g ∧ g ≤ 9) 
  : total_songs g = 9 ∨ total_songs g = 10 := 
sorry

end concert_songs_l801_801451


namespace proof_problem_l801_801407

noncomputable def f (x : ℝ) : ℝ := Real.exp x

noncomputable def g (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x + b

theorem proof_problem
  (a : ℝ) (b : ℝ) (x : ℝ)
  (h₀ : 0 ≤ a)
  (h₁ : a ≤ 1 / 2)
  (h₂ : b = 1)
  (h₃ : 0 ≤ x) :
  (1 / f x) + (x / g x a b) ≥ 1 := by
    sorry

end proof_problem_l801_801407


namespace tan_alpha_value_cos2_minus_sin2_l801_801390

variable (α : Real) 

axiom is_internal_angle (angle : Real) : angle ∈ Set.Ico 0 Real.pi 

axiom sin_cos_sum (α : Real) : α ∈ Set.Ico 0 Real.pi → Real.sin α + Real.cos α = 1 / 5

theorem tan_alpha_value (h : α ∈ Set.Ico 0 Real.pi) : Real.tan α = -4 / 3 := by 
  sorry

theorem cos2_minus_sin2 (h : Real.tan α = -4 / 3) : 1 / (Real.cos α^2 - Real.sin α^2) = -25 / 7 := by 
  sorry

end tan_alpha_value_cos2_minus_sin2_l801_801390


namespace least_number_remainder_4_l801_801264

theorem least_number_remainder_4 (n : ℕ) :
  (n % 6 = 4) ∧ (n % 7 = 4) ∧ (n % 9 = 4) ∧ (n % 18 = 4) ↔ n = 130 :=
by
  sorry

end least_number_remainder_4_l801_801264


namespace arranging_six_letters_example_l801_801281

open Multiset 

theorem arranging_six_letters_example (A B C D E F  : ℕ) :
  ∃ (arrangements : ℕ), 
    (A, B, C, D, E, F) = (1, 2, 3, 4, 5, 6) →
    (A ≠ B) ∧ (A ≠ C) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (C ≠ E) ∧ (C ≠ F) →
    (A < C ∧ B < C ∨ A > C ∧ B > C) →
    arrangements = 480 := 
by 
  sorry

end arranging_six_letters_example_l801_801281


namespace range_of_a_l801_801405
noncomputable theory

def f (x a : ℝ) : ℝ := x * (a - exp (-x))
def f' (x a : ℝ) : ℝ := a + (x - 1) * exp (-x)
def y (x : ℝ) : ℝ := (1 - x) * exp (-x)
def y' (x : ℝ) : ℝ := (x - 2) * exp (-x)

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f' x1 a = 0 ∧ f' x2 a = 0) ↔ a ∈ Ioo (-1 / exp 2) 0 :=
sorry

end range_of_a_l801_801405


namespace find_g_values_l801_801923

open Function

-- Defining the function g and its properties
axiom g : ℝ → ℝ
axiom g_domain : ∀ x, 0 ≤ x → 0 ≤ g x
axiom g_proper : ∀ x, 0 ≤ x → 0 ≤ g (g x)
axiom g_func : ∀ x, 0 ≤ x → g (g x) = 3 * x / (x + 3)
axiom g_interval : ∀ x, 2 ≤ x ∧ x ≤ 3 → g x = (x + 1) / 2

-- Problem statement translating to Lean
theorem find_g_values :
  g 2021 = 2021.5 ∧ g (1 / 2021) = 6 := by {
  sorry 
}

end find_g_values_l801_801923


namespace janine_total_pages_l801_801134

-- Define the conditions
def books_last_month : ℕ := 5
def books_this_month : ℕ := 2 * books_last_month
def books_per_page : ℕ := 10

-- Define the total number of pages she read in two months
def total_pages : ℕ :=
  let total_books := books_last_month + books_this_month
  total_books * books_per_page

-- State the theorem to be proven
theorem janine_total_pages : total_pages = 150 :=
by
  sorry

end janine_total_pages_l801_801134


namespace angle_at_intersection_of_extended_sides_of_regular_octagon_l801_801181

theorem angle_at_intersection_of_extended_sides_of_regular_octagon
  (ABCDEFGH : Prop) -- Regular octagon
  (is_regular_octagon : ∀ (A B C D E F G H Q : ℝ), regular_octagon A B C D E F G H)
  (AB GH : ℝ) -- Sides AB and GH of the octagon
  (extended_to_Q : ∃ Q, extend_line AB Q ∧ extend_line GH Q)
  : angle_at_point Q = 90 :=
sorry

end angle_at_intersection_of_extended_sides_of_regular_octagon_l801_801181


namespace jack_weight_l801_801125

-- Define weights and conditions
def weight_of_rocks : ℕ := 5 * 4
def weight_of_anna : ℕ := 40
def weight_of_jack : ℕ := weight_of_anna - weight_of_rocks

-- Prove that Jack's weight is 20 pounds
theorem jack_weight : weight_of_jack = 20 := by
  sorry

end jack_weight_l801_801125


namespace cyclic_sum_inequality_l801_801367

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 + 3 * b^3) / (5 * a + b) + (b^3 + 3 * c^3) / (5 * b + c) + (c^3 + 3 * a^3) / (5 * c + a) ≥ (2 / 3) * (a^2 + b^2 + c^2) :=
  sorry

end cyclic_sum_inequality_l801_801367


namespace train_speed_is_30_kmh_l801_801685

-- Defining the conditions
def train_length : ℝ := 100 -- in meters
def time_to_cross_pole : ℝ := 12 -- in seconds

-- Conversion factor from m/s to km/hr
def ms_to_kmh : ℝ := 18 / 5

-- Speed calculation
noncomputable def speed_in_kmh := (train_length / time_to_cross_pole) * ms_to_kmh

-- Theorem statement
theorem train_speed_is_30_kmh : speed_in_kmh = 30 := 
by
  sorry

end train_speed_is_30_kmh_l801_801685


namespace exists_proper_web_sequence_l801_801680

noncomputable theory

open Real

variables {A B C D E F G H I J K : ℝ × ℝ} -- Assume 11 points in ℝ²

-- Define collinear condition for three points.
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p1.1 - p2.1) * (p1.2 - p3.2) = (p1.2 - p2.2) * (p1.1 - p3.1)

-- Sequence of points to form a proper web
variables proper_sequence : list (ℝ × ℝ)

-- Proper conditions definition
def proper_web (points : list (ℝ × ℝ)) : Prop :=
  points.length = 11 ∧
  (points.nodup) ∧
  (∀ p1 p2 p3 ∈ points, ¬collinear p1 p2 p3) ∧
  (points.head = points.last)

-- Define the main theorem statement.
theorem exists_proper_web_sequence : 
  ∃ (seq : list (ℝ × ℝ)), proper_web seq :=
sorry

end exists_proper_web_sequence_l801_801680


namespace num_distinct_equilateral_triangles_l801_801030

theorem num_distinct_equilateral_triangles :
  let vertices := {B : ℂ | ∃ k ∈ (finset.range 11), B = complex.exp (2 * real.pi * complex.I * k / 11)} in
  let all_triangles := {T : set ℂ | ∃ (A B C ∈ vertices), T = {A, B, C} ∧ (abs (A - B) = abs (B - C) ∧ abs (C - A) = abs (A - B))} in
  finset.card all_triangles = 92 :=
sorry

end num_distinct_equilateral_triangles_l801_801030


namespace find_height_of_box_l801_801838

-- Given the conditions
variables (h l w : ℝ)
variables (V : ℝ)

-- Conditions as definitions in Lean
def length_eq_height (h : ℝ) : ℝ := 3 * h
def length_eq_width (w : ℝ) : ℝ := 4 * w
def volume_eq (h l w : ℝ) : ℝ := l * w * h

-- The proof problem: Prove height of the box is 12 given the conditions
theorem find_height_of_box : 
  (∃ h l w, l = 3 * h ∧ l = 4 * w ∧ l * w * h = 3888) → h = 12 :=
by
  sorry

end find_height_of_box_l801_801838


namespace min_value_expr_l801_801802

theorem min_value_expr (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : x + 2 * y = 5) :
  (1 / (x - 1) + 1 / (y - 1)) = (3 / 2 + Real.sqrt 2) :=
sorry

end min_value_expr_l801_801802


namespace max_value_at_pi_over_six_l801_801004

noncomputable def y (x : ℝ) : ℝ := x + 2 * Real.cos x

theorem max_value_at_pi_over_six :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), ∀ y ∈ Set.Icc (0 : ℝ) (Real.pi / 2), y (x) ≥ y (y) :=
by
  let x₀ := Real.pi / 6
  use x₀
  split
  · exact ⟨Real.pi_nonneg, Real.pi_div_two_pos.le⟩
  intro y hy
  sorry

end max_value_at_pi_over_six_l801_801004


namespace find_a_values_l801_801727

def has_integer_root (p : ℤ → ℤ) : Prop :=
  ∃ r : ℤ, p r = 0

theorem find_a_values :
  ∀ a : ℤ, has_integer_root (λ x, x^3 + 5 * x^2 + a * x + 8) ↔ (a = -71 ∨ a = -42 ∨ a = -24 ∨ a = -14 ∨ a = 4 ∨ a = 14 ∨ a = 22 ∨ a = 41) :=
by
  sorry

end find_a_values_l801_801727


namespace area_triangle_max_area_quadrilateral_l801_801038

variables {A B C D : Type*}
variables {a b c : ℝ}

-- Part (1): Definitions for conditions
def side_lengths (a b c : ℝ) : Prop := b = 2
def angles_relation (B : ℝ) : Prop := 2 * real.sin B = real.sqrt 3 * b
def vector_dot_product (AC AB : ℝ) : Prop := AC * AB = AB ^ 2

-- Theorem for Part (1)
theorem area_triangle {A B C : ℝ} (h1 : side_lengths a b c) (h2 : angles_relation B) (h3 : vector_dot_product A B) :
  let area := (1 / 2) * a * c * real.sin B
  in area = (real.sqrt 3) / 2 :=
sorry

-- Part (2): Additional definitions for conditions
def point_D (AC AD CD : ℝ) : Prop := CD = 3 * AD ∧ CD = 6

-- Theorem for Part (2)
theorem max_area_quadrilateral {A B C D : ℝ} (h1 : side_lengths a b c) (h2 : angles_relation B) (h3 : vector_dot_product A B)
  (h4 : point_D A D D) :
  let S := (real.sqrt 3) / 8 * A * C + (1 / 2) * 2 * 6 * real.sin D
  in S = 5 * real.sqrt 3 + 3 * real.sqrt 7 :=
sorry

end area_triangle_max_area_quadrilateral_l801_801038


namespace sum_of_first_100_triangular_numbers_l801_801197

def triangular (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem sum_of_first_100_triangular_numbers :
  ∑ n in Finset.range 101, triangular n = 171700 := 
sorry

end sum_of_first_100_triangular_numbers_l801_801197


namespace g_at_2_eq_neg4_l801_801161

def f (x : ℝ) : ℝ :=
  if x < 0 then (1 / 2) ^ x else 
  if x = 0 then 0 else 
  g x

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

variable (g : ℝ → ℝ)
variable (h₀ : is_odd_function f)

theorem g_at_2_eq_neg4 : g 2 = -4 :=
by sorry

end g_at_2_eq_neg4_l801_801161


namespace solve_inequality_inequality_proof_l801_801280

-- Problem 1: Solve the inequality |2x+1| - |x-4| > 2
theorem solve_inequality (x : ℝ) :
  (|2 * x + 1| - |x - 4| > 2) ↔ (x < -7 ∨ x > (5/3)) :=
sorry

-- Problem 2: Prove the inequality given a > 0 and b > 0
theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a / Real.sqrt b + b / Real.sqrt a) ≥ (Real.sqrt a + Real.sqrt b) :=
sorry

end solve_inequality_inequality_proof_l801_801280


namespace ball_arrangements_l801_801748

theorem ball_arrangements :
  let balls := 5
  let selected_balls := 4
  let first_box := 1
  let second_box := 2
  let third_box := 1
  let total_ways := Nat.choose balls selected_balls * Nat.choose selected_balls first_box * 
                    Nat.choose (selected_balls - first_box) second_box *  Nat.choose 1 third_box
  in total_ways = 60
:= by
  -- Definitions
  let balls := 5
  let selected_balls := 4
  let first_box := 1
  let second_box := 2
  let third_box := 1
  let total_ways := Nat.choose balls selected_balls * Nat.choose selected_balls first_box * 
                    Nat.choose (selected_balls - first_box) second_box * Nat.choose 1 third_box
  
  -- The proof is omitted.
  sorry

end ball_arrangements_l801_801748


namespace true_statement_is_P3_l801_801620

-- Define the statements as propositions
variable (P1 P2 P3 : Prop)

-- Define the conditions
def statement_1 := P1
def statement_2 := P2
def statement_3 := P3

-- Define the assumption that exactly one of the statements is true
def exactly_one_true (P1 P2 P3 : Prop) : Prop :=
  (P1 ∧ ¬P2 ∧ ¬P3) ∨ (¬P1 ∧ P2 ∧ ¬P3) ∨ (¬P1 ∧ ¬P2 ∧ P3)

-- The goal: to prove that the third statement is the true one
theorem true_statement_is_P3 : exactly_one_true P1 P2 P3 → P3 :=
by
  intros h,
  cases h,
  { -- First case: P1 is true, which we know will lead to a contradiction
    exfalso,
    exact h.right.left h.left, * } |
  { -- Second case: P2 is true, which we know will lead to a contradiction
    exfalso,
    exact h.right.right h.left, * } |
  { -- Third case: The only consistent scenario, where P3 is true
    exact h.right_right,
  }
(val h)

-- Add sorry to skip the proof
sorry

end true_statement_is_P3_l801_801620


namespace abs_inequality_solution_set_l801_801241

theorem abs_inequality_solution_set (x : ℝ) : |x - 1| > 2 ↔ x > 3 ∨ x < -1 :=
by
  sorry

end abs_inequality_solution_set_l801_801241


namespace taller_tree_is_60_l801_801610

variable (height_taller_tree : ℕ) (height_shorter_tree : ℕ)

-- Conditions
def condition1 : Prop :=
  height_taller_tree = height_shorter_tree + 20

def condition2 : Prop :=
  height_shorter_tree * 3 = height_taller_tree * 2

-- Theorem: Prove the height of the taller tree is 60
theorem taller_tree_is_60 (h : height_taller_tree = 60) :
  condition1 → condition2 → height_taller_tree = 60 :=
by
  intros cond1 cond2
  exact h

end taller_tree_is_60_l801_801610


namespace relationship_among_abc_l801_801041

noncomputable theory
open Real

variables {f : ℝ → ℝ}

-- Given conditions
def even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def condition_on_f (f : ℝ → ℝ) := ∀ x, x ∈ Iio 0 → f x + deriv f x < 0

-- Constants definitions based on the problem
def a := 2 ^ 0.1 * f (2 ^ 0.1)
def b := log 2 * f (log 2)
def c := -3 * f (-3)  -- since log₂(1/8) = -3

-- Theorems to prove the relationship
theorem relationship_among_abc (hf_even : even_function f) (hf_cond : condition_on_f f) : 
  b < a ∧ a < c :=
sorry

end relationship_among_abc_l801_801041


namespace zoey_friends_l801_801961

theorem zoey_friends (money_won : ℕ) (h : money_won = 7348340) :
  ∃ n, n = money_won → nat.prime (money_won + 1) ∧ n = 7348340 := 
sorry

end zoey_friends_l801_801961


namespace magician_trick_correct_l801_801296

theorem magician_trick_correct : ∃ (N : ℕ), (∀ (seq : Fin N → Fin 10) 
  (i : Fin (N-1)), ∀ (d1 d2: Fin 10), covers_two_adjacent seq i d1 d2 →
  magician_guesses_correctly seq i d1 d2) ∧ N = 101 :=
begin
  sorry
end

-- Definitions required for the statement
def covers_two_adjacent (seq : Fin N → Fin 10) (i : Fin (N-1)) (d1 d2 : Fin 10) : Prop :=
  seq i = d1 ∧ seq ⟨i + 1, _⟩ = d2

def magician_guesses_correctly (seq : Fin N → Fin 10) (i : Fin (N-1)) (d1 d2 : Fin 10) : Prop :=
  ∀ seq_with_guesses : Fin N → Fin 10,
  (∀ j : Fin N, (j ≠ i ∧ j ≠ ⟨i + 1, _⟩) → seq_with_guesses j = seq j) 
  ∧ seq_with_guesses i = d1 ∧ seq_with_guesses ⟨i + 1, _⟩ = d2

end magician_trick_correct_l801_801296


namespace find_dividend_l801_801800

theorem find_dividend (x D : ℕ) (q r : ℕ) (h_q : q = 4) (h_r : r = 3)
  (h_div : D = x * q + r) (h_sum : D + x + q + r = 100) : D = 75 :=
by
  sorry

end find_dividend_l801_801800


namespace polynomial_min_value_l801_801008

theorem polynomial_min_value (n : ℕ) (h : 2 ≤ n) : 
  ∃ x : ℝ, (x = (n + 1) / 2) ∧ 
    ∀ y : ℝ, (λ x, (list.range n).sum (λ k, (x - (k + 1))^4)) y ≥ 
    (λ x, (list.range n).sum (λ k, (x - (k + 1))^4)) x :=
begin
  sorry
end

end polynomial_min_value_l801_801008


namespace solution_set_equality_of_abs_x_minus_2_gt_1_and_quadratic_gt_0_l801_801043

variable (a b : ℝ)

theorem solution_set_equality_of_abs_x_minus_2_gt_1_and_quadratic_gt_0 :
  (∀ x : ℝ, (|x - 2| > 1 ↔ x^2 + a * x + b > 0)) → a + b = -1 :=
by
  sorry

end solution_set_equality_of_abs_x_minus_2_gt_1_and_quadratic_gt_0_l801_801043


namespace pieces_of_mail_in_july_l801_801257

-- Given conditions encoded as definitions
def mail_sent (month : ℕ) : ℕ
| 1 := 5   -- April
| 2 := 10  -- May
| 3 := 20  -- June
| 5 := 80  -- August
| _ := sorry  -- values for other months are not required

theorem pieces_of_mail_in_july : mail_sent 4 = 40 :=
by
  -- problem conditions
  have apr_mail : mail_sent 1 = 5 := rfl,
  have may_mail : mail_sent 2 = 10 := rfl,
  have jun_mail : mail_sent 3 = 20 := rfl,
  have aug_mail : mail_sent 5 = 80 := rfl,
  
  -- proven part via pattern doubling which was in steps
  sorry

end pieces_of_mail_in_july_l801_801257


namespace tan_sum_pi_over_12_eq_4_l801_801545

theorem tan_sum_pi_over_12_eq_4 :
  tan (π / 12) + tan (5 * π / 12) = 4 := 
by
  have cos_pi_over_12 : cos (π / 12) = (real.sqrt 6 + real.sqrt 2) / 4 := sorry
  have cos_5pi_over_12 : cos (5 * π / 12) = (real.sqrt 6 - real.sqrt 2) / 4 := sorry
  have sin_pi_over_2 : sin (π / 2) = 1 := by
    exact real.sin_pi_div_two
  sorry

end tan_sum_pi_over_12_eq_4_l801_801545


namespace undefined_expression_iff_l801_801707

theorem undefined_expression_iff (x : ℝ) :
  (x^2 - 24 * x + 144 = 0) ↔ (x = 12) := 
sorry

end undefined_expression_iff_l801_801707


namespace probability_starting_vertex_l801_801998

-- Conditions of the problem:
def is_equilateral (v₁ v₂ v₃ : ℕ → ℕ → Prop) : Prop := 
  ∀ (a b c : ℕ), v₁ a b = v₂ b c ∧ v₂ b c = v₃ c a ∧ v₃ c a = v₁ a b

def ant_random_move (num_vertices : ℕ) : Prop := 
  num_vertices = 3

-- The proposition we want to prove:
theorem probability_starting_vertex (num_faces : ℕ)
  (is_regular_tetrahedron : num_faces = 4)
  (equi_faces : ∀ v₁ v₂ v₃ : ℕ → ℕ → Prop, is_equilateral v₁ v₂ v₃)
  (moves_randomly : ant_random_move 3):
  ∃ (p : ℚ), p = 1 / 6 :=
begin
  sorry
end

end probability_starting_vertex_l801_801998


namespace percentage_of_a_is_100_l801_801977

noncomputable theory

variable (a b c P: ℚ)

-- Condition 1: A certain percentage of a is 8
def percentage_of_a := P / 100 * a = 8

-- Condition 2: 2 is 8% of b
def eight_percent_of_b := 0.08 * b = 2

-- Condition 3: c equals b / a
def c_equals_b_over_a := c = b / a

-- Proof: What is the value of the percentage of a
theorem percentage_of_a_is_100 :
  percentage_of_a a P ∧ eight_percent_of_b (P / 100 * a = 8) (0.08 * b = 2) ∧ 
  c_equals_b_over_a (c = b / a) → P = 100 :=
sorry

end percentage_of_a_is_100_l801_801977


namespace max_val_expression_l801_801144

theorem max_val_expression (n : ℕ) (h : n > 5) (v : ℕ → ℕ) 
  (no_collinear_points : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → (v i) ≠ (v j) → (v j) ≠ (v k))
  (erase_points : ∀ i, 1 ≤ i ∧ i < n - 3 → (v (n-2) = 3)) :
  |v 1 - v 2| + |v 2 - v 3| + ... + |v (n-3) - v (n-2)| = 2 * n - 8 := 
sorry

end max_val_expression_l801_801144


namespace person_A_pays_15_sum_fees_30_l801_801570

noncomputable def fee_structure (hours : ℕ) : ℕ :=
  if hours = 0 then 0
  else if hours = 1 then 5
  else 5 + (hours-1)*10

def prob_pay_15 : ℚ := 1 - (1/2 + 1/6)

def pay_exactly_15 (prob : ℚ) : Prop :=
  prob = 1/3

def total_fees_30_events : list (ℕ × ℕ) :=
  [(5, 25), (15, 15), (25, 5)]

def prob_sum_30 (events : list (ℕ × ℕ)) (total_events : ℕ) : ℚ :=
  events.length / total_events.to_rat

def total_combinations : ℕ := 16

theorem person_A_pays_15 :
  pay_exactly_15 prob_pay_15 := by
  sorry

theorem sum_fees_30 :
  prob_sum_30 total_fees_30_events total_combinations = 3 / 16 := by
  sorry

end person_A_pays_15_sum_fees_30_l801_801570


namespace value_of_coat_l801_801992

noncomputable def coat_value := 4.8

theorem value_of_coat 
  (annual_rubles : ℝ) (months_worked : ℝ) (rubles_received : ℝ) (actual_payment : ℝ) (value_of_coat : ℝ) :
  annual_rubles = 12 →
  months_worked = 7 →
  rubles_received = 5 →
  actual_payment = 5 + value_of_coat →
  (7 / 12 * (12 + value_of_coat)) = actual_payment →
  value_of_coat = 4.8 :=
by {
  intros h1 h2 h3 h4 h5,
  assume h5,
  sorry
}

end value_of_coat_l801_801992


namespace tan_sum_simplification_l801_801552
-- We start by importing the relevant Lean libraries that contain trigonometric functions and basic real analysis.

-- Define the statement to be proved in Lean.
theorem tan_sum_simplification :
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12) = 4 * Real.sqrt 2 - 4) :=
by
  sorry

end tan_sum_simplification_l801_801552


namespace martha_total_butterflies_l801_801165

variable (Yellow Blue Black : ℕ)

def butterfly_equations (Yellow Blue Black : ℕ) : Prop :=
  (Blue = 2 * Yellow) ∧ (Blue = 6) ∧ (Black = 10)

theorem martha_total_butterflies 
  (h : butterfly_equations Yellow Blue Black) : 
  (Yellow + Blue + Black = 19) :=
by
  sorry

end martha_total_butterflies_l801_801165


namespace tan_three_halves_pi_sub_alpha_l801_801755

theorem tan_three_halves_pi_sub_alpha (α : ℝ) (h : Real.cos (π - α) = -3/5) :
    Real.tan (3 * π / 2 - α) = 3/4 ∨ Real.tan (3 * π / 2 - α) = -3/4 := by
  sorry

end tan_three_halves_pi_sub_alpha_l801_801755


namespace relationship_among_abc_l801_801079

noncomputable def a : ℝ := Real.log (1 / 2)
noncomputable def b : ℝ := (1 / 3) ^ 0.8
noncomputable def c : ℝ := 2 ^ (1 / 3)

theorem relationship_among_abc : c > b ∧ b > a := by
  sorry

end relationship_among_abc_l801_801079


namespace area_ratio_of_squares_l801_801192

theorem area_ratio_of_squares (s : ℝ) :
  let AI := 3 / 8 * s,
      IB := 5 / 8 * s,
      IJ := sqrt ((5 * s / 8)^2 + (3 * s / 8)^2) in
  IJ = sqrt 34 / 8 * s ∧ (IJ^2 / s^2) = 17 / 32 :=
by
sorry

end area_ratio_of_squares_l801_801192


namespace arc_length_parametric_curve_l801_801318

noncomputable def arcLength (f : ℝ → ℝ × ℝ) (a b : ℝ) : ℝ :=
  ∫ t in a..b, sqrt ((deriv (fun t => (f t).1) t)^2 + (deriv (fun t => (f t).2) t)^2)

def parametricCurve (t : ℝ) : ℝ × ℝ := (6 * (cos t + t * sin t), 6 * (sin t - t * cos t))

theorem arc_length_parametric_curve :
  arcLength parametricCurve 0 π = 3 * π^2 :=
by
  sorry

end arc_length_parametric_curve_l801_801318


namespace tan_pi_over_12_plus_tan_5pi_over_12_eq_4_l801_801526

-- Definitions for given conditions
def cos_pi_over_12 : ℝ := (Real.sqrt 6 + Real.sqrt 2) / 4
def cos_5pi_over_12 : ℝ := (Real.sqrt 6 - Real.sqrt 2) / 4

-- The theorem to be proved
theorem tan_pi_over_12_plus_tan_5pi_over_12_eq_4 : 
  Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 :=
by sorry

end tan_pi_over_12_plus_tan_5pi_over_12_eq_4_l801_801526


namespace equilateral_triangles_in_extended_hexagonal_lattice_l801_801717

-- Definitions based on conditions
def point := ℤ × ℤ

def distance (p1 p2 : point) : ℝ := sqrt (↑((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2))

-- Extended hexagonal lattice with an additional ring
def is_lattice_point (p : point) : Prop :=
  ∃ k: ℤ, (p.1, p.2) = (k, 0) ∨ (p.1, p.2) = (k * cos (π / 3), k * sin (π / 3)) ∨ 
  (p.1, p.2) = (k * cos (2 * π / 3), k * sin (2 * π / 3)) ∨ 
  (p.1, p.2) = (k * cos π, k * sin π) ∨ 
  (p.1, p.2) = (k * cos (4 * π / 3), k * sin (4 * π / 3)) ∨ 
  (p.1, p.2) = (k * cos (5 * π / 3), k * sin (5 * π / 3))

def is_equilateral_triangle (p1 p2 p3 : point) : Prop :=
  distance p1 p2 = distance p2 p3 ∧ distance p2 p3 = distance p3 p1 ∧ 
  distance p1 p2 > 0 ∧ 
  is_lattice_point p1 ∧ is_lattice_point p2 ∧ is_lattice_point p3

-- Main theorem statement
theorem equilateral_triangles_in_extended_hexagonal_lattice : 
  ∃ n : ℕ, n = 32 ∧ ∀ (t : set point), t.card = 3 →
  (∃ p1 p2 p3 ∈ t, is_equilateral_triangle p1 p2 p3) :=
sorry

end equilateral_triangles_in_extended_hexagonal_lattice_l801_801717


namespace geometric_series_l801_801520

/-- Mathematical induction problem: 
For x ≠ 1 and n ∈ ℕ+, prove that 1 + x + x^2 + ... + x^(n+2) = (1 - x^(n+3)) / (1 - x). -/
theorem geometric_series (x : ℝ) (hx : x ≠ 1) (n : ℕ) (hn : 0 < n) : 
  (∑ k in finset.range (n + 3), x ^ k) = (1 - x ^ (n + 3)) / (1 - x) :=
by
  sorry

end geometric_series_l801_801520


namespace tan_sum_pi_over_12_eq_4_l801_801549

theorem tan_sum_pi_over_12_eq_4 :
  tan (π / 12) + tan (5 * π / 12) = 4 := 
by
  have cos_pi_over_12 : cos (π / 12) = (real.sqrt 6 + real.sqrt 2) / 4 := sorry
  have cos_5pi_over_12 : cos (5 * π / 12) = (real.sqrt 6 - real.sqrt 2) / 4 := sorry
  have sin_pi_over_2 : sin (π / 2) = 1 := by
    exact real.sin_pi_div_two
  sorry

end tan_sum_pi_over_12_eq_4_l801_801549


namespace positive_integer_solutions_l801_801334

theorem positive_integer_solutions (k : ℕ) (h_pos : k > 0) :
  (∃ x : ℤ, k * x - 18 = 4 * k) ↔ k ∈ ({1, 2, 3, 6, 9, 18} : set ℕ) :=
by {
  split;
  intro h,
  {
    cases h with x hx,
    have key := nat.dvd_of_mod_eq_zero (int.coe_nat_dvd.mp ((nat.add_sub_cancel x k).symm ▸ show 
    ((k * (x - 4)).nat_abs = 18), from int.coe_nat_inj _)),
    apply key,
  },
  {
    cases x,
    {
      fin_cases h; existsi (nat.succ n) * 4 - 18; ring,
    },
    {
      fin_cases h; existsi (nat.pred n) * 4 - 18; ring,
    },
  }
}

end positive_integer_solutions_l801_801334


namespace range_of_m_l801_801415

-- Define the sets A, B, and C
def set_A : set ℝ := { x | x > 1 }
def set_B (x : ℝ) : set ℝ := { y | y = log10 (2 / (x + 1)) }
def set_C (m x : ℝ) : ℝ := (m^2 * x - 1) / (m * x + 1)

-- Define the condition C ⊆ B
def subset_condition (x m : ℝ) : Prop :=
  (set_C m x) < 0

-- Goal: Prove the range of m under the condition that C ⊆ B is (-∞, -1] ∪ {0}
theorem range_of_m (x : ℝ) (h : x ∈ set_A) (m : ℝ) : 
  (subset_condition x m ↔ m ≤ -1 ∨ m = 0) :=
by
  sorry

end range_of_m_l801_801415


namespace probability_three_correct_l801_801255

theorem probability_three_correct : 
  let pA := 3 / 4,
      pB := 2 / 3,
      not_pA := 1 - pA,
      not_pB := 1 - pB in
  (pA * pA * pA * pB * not_pB * not_pB * not_pB) +
  (pA * pA * pA * pB * not_pB * pB * not_pB) +
  (pA * pA * pA * pB * pB * not_pB * not_pB) +
  (pA * pA * pA * not_pB * pB * pB * not_pB) = 5 / 12 :=
by
  sorry

end probability_three_correct_l801_801255


namespace trajectory_of_Z_equation_of_line_l_area_of_triangle_PQR_l801_801859

-- Definitions from conditions
variable (z : ℂ) (x y : ℝ) (i : ℂ) (n : ℝ) (t : ℝ)
variable (P Q R : ℝ × ℝ)

-- Condition 1: Point Z in complex plane
def point_Z_condition := (z = x + y * i) ∧ (|z + 2| + |z - 2| = 6)

-- Condition 2: Hyperbola C₂ shares focus with C₁
def hyperbola_and_focus_condition := (c^2 = 1 + n) ∧ (c = 2)

-- Condition 3: Line l intersects asymptotes of C₂
def line_intersection_condition := 
  let A := (t / (sqrt 3 - 1), sqrt 3 * t / (sqrt 3 - 1)) in
  let B := (-t / (sqrt 3 + 1), sqrt 3 * t / (sqrt 3 + 1)) in
  (A, B) ∧ O = (0,0) ∧ (OA • OB = 2)

-- Function to convert complex to real part
def real_part (c : ℂ) := complex.re c

-- Function to convert real part to variable pair (x,y)
def real_to_rv (w : ℝ) := (complex_re w, complex_im w)

-- Triangle Vertices and Centroid
def centroid_condition := 
  let P := (3 * cos θ_1, sqrt 5 * sin θ_1), 
      Q := (3 * cos θ_2, sqrt 5 * sin θ_2), 
      R := (3 * cos θ_3, sqrt 5 * sin θ_3) in
  O = ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3)

-- Proof for trajectory equation of Z
theorem trajectory_of_Z :
  point_Z_condition → hyperbola_and_focus_condition → Equation := 
  sorry

-- Proof for equation of line l
theorem equation_of_line_l :
  line_intersection_condition → Equation := 
  sorry

-- Proof for area of triangle PQR
-- Provided O is the centroid
theorem area_of_triangle_PQR : 
  centroid_condition → real := 
  sorry


end trajectory_of_Z_equation_of_line_l_area_of_triangle_PQR_l801_801859


namespace circle_area_l801_801205

theorem circle_area (C : ℝ) (hC : C = 24) : ∃ (A : ℝ), A = 144 / π :=
by
  sorry

end circle_area_l801_801205


namespace haman_dropped_trays_l801_801420

def initial_trays_to_collect : ℕ := 10
def additional_trays : ℕ := 7
def eggs_sold : ℕ := 540
def eggs_per_tray : ℕ := 30

theorem haman_dropped_trays :
  ∃ dropped_trays : ℕ,
  (initial_trays_to_collect + additional_trays - dropped_trays)*eggs_per_tray = eggs_sold → dropped_trays = 8 :=
sorry

end haman_dropped_trays_l801_801420


namespace ceil_sqrt_of_900_l801_801339

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem ceil_sqrt_of_900 :
  isPerfectSquare 36 ∧ isPerfectSquare 25 ∧ (36 * 25 = 900) → 
  Int.ceil (Real.sqrt 900) = 30 :=
by
  intro h
  sorry

end ceil_sqrt_of_900_l801_801339


namespace probability_of_rolling_five_four_times_in_five_rolls_l801_801439

theorem probability_of_rolling_five_four_times_in_five_rolls :
  let p := ((1 / 6) ^ 4) * (5 / 6),
      total_prob := 5 * p in
  total_prob = 25 / 7776 := by
  sorry

end probability_of_rolling_five_four_times_in_five_rolls_l801_801439


namespace regular_ticket_cost_l801_801295

theorem regular_ticket_cost (total_tickets : ℕ) (senior_ticket_cost : ℕ) (total_sales : ℕ) (regular_tickets : ℕ) 
    (h1 : total_tickets = 65)
    (h2 : senior_ticket_cost = 10)
    (h3 : total_sales = 855)
    (h4 : regular_tickets = 41) : 
    (regular_ticket_cost : ℕ) :=
  have h5 : regular_ticket_cost = (total_sales - (total_tickets - regular_tickets) * senior_ticket_cost) / regular_tickets, by
    sorry,
  show regular_ticket_cost = 15, by
    exact h5

end regular_ticket_cost_l801_801295


namespace mul_18396_9999_l801_801642

theorem mul_18396_9999 :
  18396 * 9999 = 183941604 :=
by
  sorry

end mul_18396_9999_l801_801642


namespace average_greater_than_median_by_13_l801_801871

-- Define the weights of the six children
def weights := [4, 6, 8, 10, 12, 90]

-- Calculate the median of the weights
def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  (sorted.get! (l.length / 2 - 1) + sorted.get! (l.length / 2)) / 2

-- Calculate the average of the weights
def average (l : List ℕ) : ℕ :=
  l.sum / l.length

-- Define the proof goal:
theorem average_greater_than_median_by_13 :
  average weights = 9 + 13 :=
by
  sorry

end average_greater_than_median_by_13_l801_801871


namespace tangent_hyperbola_sin_inv_l801_801225

/-- The graph of x^2 - (y - 1)^2 = 1 has one tangent line with positive slope that passes through (0,0). 
    The point of tangency is (a, b). Prove that sin^(-1)(a/b) = π/4. -/
theorem tangent_hyperbola_sin_inv (a b : ℝ) 
  (h1 : a^2 - (b - 1)^2 = 1)
  (h2 : b = 2)
  (h3 : a = real.sqrt 2) :
  real.arcsin (a / b) = real.pi / 4 :=
begin
  sorry
end

end tangent_hyperbola_sin_inv_l801_801225


namespace find_term_number_l801_801782

theorem find_term_number :
  ∃ n : ℕ, (2 * (5 : ℝ)^(1/2) = (3 * (n : ℝ) - 1)^(1/2)) ∧ n = 7 :=
sorry

end find_term_number_l801_801782


namespace percentage_decrease_area_square_l801_801106

theorem percentage_decrease_area_square
  (area_tri_A : ℝ) (area_tri_C : ℝ) (area_square_B : ℝ) (side_EF_decrease_percent : ℝ) :
  area_tri_A = 18 * Real.sqrt 3 ∧ 
  area_tri_C = 72 * Real.sqrt 3 ∧ 
  area_square_B = 72 ∧ 
  side_EF_decrease_percent = 25 → 
  let new_side_length := (side_length_square_B : ℝ) :=
    Real.sqrt 72 * (1 - side_EF_decrease_percent / 100) in
  let new_area_square := new_side_length^2 in
  let decrease := area_square_B - new_area_square in
  let percentage_decrease := (decrease / area_square_B) * 100 in
  percentage_decrease = 43.75 :=
  by sorry

end percentage_decrease_area_square_l801_801106


namespace tan_sum_pi_over_12_eq_4_l801_801546

theorem tan_sum_pi_over_12_eq_4 :
  tan (π / 12) + tan (5 * π / 12) = 4 := 
by
  have cos_pi_over_12 : cos (π / 12) = (real.sqrt 6 + real.sqrt 2) / 4 := sorry
  have cos_5pi_over_12 : cos (5 * π / 12) = (real.sqrt 6 - real.sqrt 2) / 4 := sorry
  have sin_pi_over_2 : sin (π / 2) = 1 := by
    exact real.sin_pi_div_two
  sorry

end tan_sum_pi_over_12_eq_4_l801_801546


namespace max_area_of_triangle_l801_801278

theorem max_area_of_triangle (a b c : ℝ) (A B C : ℝ) (h1 : a + c = 4)
  (h2 : sin A * (1 + cos B) = (2 - cos A) * sin B) : 
  ∃ (area : ℝ), area = sqrt 3 ∧ area = (1 / 2) * a * c * sin B :=
begin
  sorry
end

end max_area_of_triangle_l801_801278


namespace simplify_tangent_sum_l801_801541

theorem simplify_tangent_sum :
  tan (Real.pi / 12) + tan (5 * Real.pi / 12) = Real.sqrt 6 - Real.sqrt 2 := 
sorry

end simplify_tangent_sum_l801_801541


namespace cost_per_unit_range_of_type_A_purchases_maximum_profit_l801_801446

-- Definitions of the problem conditions
def cost_type_A : ℕ := 15
def cost_type_B : ℕ := 20

def profit_type_A : ℕ := 3
def profit_type_B : ℕ := 4

def budget_min : ℕ := 2750
def budget_max : ℕ := 2850

def total_units : ℕ := 150
def profit_min : ℕ := 565

-- Main proof statements as Lean theorems
theorem cost_per_unit : 
  ∃ (x y : ℕ), 
    2 * x + 3 * y = 90 ∧ 
    3 * x + y = 65 ∧ 
    x = cost_type_A ∧ 
    y = cost_type_B := 
sorry

theorem range_of_type_A_purchases : 
  ∃ (a : ℕ), 
    30 ≤ a ∧ 
    a ≤ 50 ∧ 
    budget_min ≤ cost_type_A * a + cost_type_B * (total_units - a) ∧ 
    cost_type_A * a + cost_type_B * (total_units - a) ≤ budget_max := 
sorry

theorem maximum_profit : 
  ∃ (a : ℕ), 
    30 ≤ a ∧ 
    a ≤ 35 ∧ 
    profit_min ≤ profit_type_A * a + profit_type_B * (total_units - a) ∧ 
    ¬∃ (b : ℕ), 
      30 ≤ b ∧ 
      b ≤ 35 ∧ 
      b ≠ a ∧ 
      profit_type_A * b + profit_type_B * (total_units - b) > profit_type_A * a + profit_type_B * (total_units - a) :=
sorry

end cost_per_unit_range_of_type_A_purchases_maximum_profit_l801_801446


namespace product_of_integers_l801_801366

theorem product_of_integers (a b c d e : ℕ) (h₁ : ab + a + b = 624)
  (h₂ : bc + b + c = 234) (h₃ : cd + c + d = 156) (h₄ : de + d + e = 80)
  (h₅ : a * b * c * d * e = 10!) : a - e = 22 := 
sorry

end product_of_integers_l801_801366


namespace minimalPerimeterQuadrilateral_l801_801741

-- Define what it means for a quadrilateral to be cyclic
def isCyclic (A B C D : Point) : Prop := ∠A + ∠C = 180 ∧ ∠B + ∠D = 180

-- Define the conditions given in the problem
variable (A B C D : Point)

-- State the theorem with the conditions and the required proof
theorem minimalPerimeterQuadrilateral (h1 : ¬isCyclic A B C D) :
  ¬ ∃ P Q R S, (inscribedQuadrilateral P Q R S A B C D ∧ minimalPerimeter P Q R S) ∨
  (isCyclic A B C D → ∃ infiniteSolutionsForMinimalPerimeter A B C D) :=
sorry

end minimalPerimeterQuadrilateral_l801_801741


namespace find_t_l801_801081

variables (s t : ℚ)

theorem find_t (h1 : 12 * s + 7 * t = 154) (h2 : s = 2 * t - 3) : t = 190 / 31 :=
by
  sorry

end find_t_l801_801081


namespace coffees_harold_picked_l801_801512

theorem coffees_harold_picked:
  ∃ (C : ℝ) (mC : ℕ), 
    0.45 * 3 + C * mC = 4.91 ∧
    0.45 * 5 + C * 6 = 7.59 ∧ 
    mC = 4 :=
begin
  sorry
end

end coffees_harold_picked_l801_801512


namespace sum_of_first_100_digits_of_1_div_2222_l801_801957

theorem sum_of_first_100_digits_of_1_div_2222 : 
  (let repeating_block := [0, 0, 0, 4, 5];
  let sum_of_digits (lst : List ℕ) := lst.sum;
  let block_sum := sum_of_digits repeating_block;
  let num_blocks := 100 / 5;
  num_blocks * block_sum = 180) :=
by 
  let repeating_block := [0, 0, 0, 4, 5]
  let sum_of_digits (lst : List ℕ) := lst.sum
  let block_sum := sum_of_digits repeating_block
  let num_blocks := 100 / 5
  have h : num_blocks * block_sum = 180 := sorry
  exact h

end sum_of_first_100_digits_of_1_div_2222_l801_801957


namespace college_strength_l801_801449

-- Define variables for the problem
variables (C B S T : ℕ)
variables (CS ST SB CB CT BT CBST : ℕ)

-- Conditions given in the problem
def cricket := (C = 500)
def basketball := (B = 600)
def soccer := (S = 250)
def tennis := (T = 200)
def cricket_soccer := (CS = 100)
def soccer_tennis := (ST = 50)
def soccer_basketball := (SB = 75)
def cricket_basketball := (CB = 220)
def cricket_tennis := (CT = 150)
def basketball_tennis := (BT = 50)
def all_four := (CBST = 30)

-- The statement to be proved
theorem college_strength : 
  C = 500 →
  B = 600 →
  S = 250 →
  T = 200 →
  CS = 100 →
  ST = 50 →
  SB = 75 →
  CB = 220 →
  CT = 150 →
  BT = 50 →
  CBST = 30 →
  (C + B + S + T - CS - ST - SB - CB - CT - BT + 4 * CBST = 1025) :=
by
  intros hC hB hS hT hCS hST hSB hCB hCT hBT hCBST
  rw [hC, hB, hS, hT, hCS, hST, hSB, hCB, hCT, hBT, hCBST]
  sorry

end college_strength_l801_801449


namespace intercepts_of_line_l801_801592

theorem intercepts_of_line :
  (∀ x y : ℝ, (x = 4 ∨ y = -3) → (x / 4 - y / 3 = 1)) ∧ (∀ x y : ℝ, (x / 4 = 1 ∧ y = 0) ∧ (x = 0 ∧ y / 3 = -1)) :=
by
  sorry

end intercepts_of_line_l801_801592


namespace fraction_evaluation_l801_801635

def is_whole_number (x : ℝ) : Prop := floor x = x

theorem fraction_evaluation : is_whole_number (52 / 4) ∧
  ¬ is_whole_number (52 / 5) ∧
  ¬ is_whole_number (52 / 7) ∧
  ¬ is_whole_number (52 / 3) ∧
  ¬ is_whole_number (52 / 6) :=
by
  sorry

end fraction_evaluation_l801_801635


namespace smallest_abs_P0_exists_l801_801311

theorem smallest_abs_P0_exists (P : ℤ[X]) (h1 : P.eval (-10) = 145) (h2 : P.eval 9 = 164) :
  ∃ (n : ℤ), |P.eval 0| = n ∧ n = 25 :=
sorry

end smallest_abs_P0_exists_l801_801311


namespace range_of_a_range_of_m_l801_801279

-- Problem (1)
theorem range_of_a (a : ℝ) (h1 : ∀ x : ℝ, ¬(2 * x^2 - 3 * x + 1 ≤ 0) → ¬(x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0) ∧ ¬(x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0) → ¬(2 * x^2 - 3 * x + 1 ≤ 0)) :
  0 ≤ a ∧ a ≤ 0.5 :=
sorry

-- Problem (2)
theorem range_of_m (m : ℝ) (h1 : ∃ x ∈ (0, 1), ∃ y ∈ (2, 3), x^2 + (m - 3) * x + m = 0 ∧ y^2 + (m - 3) * y + m = 0 ∨ ∀ x : ℝ, ln (m * x^2 - 2 * x + 1) :=
  (0 < m ∧ m < 2 / 3) ∨ 1 < m :=
sorry

end range_of_a_range_of_m_l801_801279


namespace included_angle_between_vectors_l801_801787

noncomputable def a : ℝ × ℝ := (-Real.sqrt 3 / 3, 1)
noncomputable def b : ℝ × ℝ := (1, Real.sqrt 3)
noncomputable def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)
noncomputable def theta : ℝ := Real.arccos (dot a b / (norm a * norm b))

theorem included_angle_between_vectors : theta = Real.pi / 3 := by
  -- Proof goes here
  sorry

end included_angle_between_vectors_l801_801787


namespace books_selection_l801_801247

theorem books_selection (n m : ℕ) (hn : n = 5) (hm : m = 6) : n * m = 30 :=
by
  rw [hn, hm]
  exact Nat.mul_comm 5 6

end books_selection_l801_801247


namespace darius_drive_miles_l801_801328

theorem darius_drive_miles (total_miles : ℕ) (julia_miles : ℕ) (darius_miles : ℕ) 
  (h1 : total_miles = 1677) (h2 : julia_miles = 998) (h3 : total_miles = darius_miles + julia_miles) : 
  darius_miles = 679 :=
by
  sorry

end darius_drive_miles_l801_801328


namespace solve_inequality_l801_801903

noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

theorem solve_inequality (x : ℝ) (hx : x ≠ 0 ∧ 0 < x) :
  (64 + (log_b (1/5) (x^2))^3) / (log_b (1/5) (x^6) * log_b 5 (x^2) + 5 * log_b 5 (x^6) + 14 * log_b (1/5) (x^2) + 2) ≤ 0 ↔
  (x ∈ Set.Icc (-25 : ℝ) (- Real.sqrt 5)) ∨
  (x ∈ Set.Icc (- (Real.exp (Real.log 5 / 3))) 0) ∨
  (x ∈ Set.Icc 0 (Real.exp (Real.log 5 / 3))) ∨
  (x ∈ Set.Icc (Real.sqrt 5) 25) :=
by 
  sorry

end solve_inequality_l801_801903


namespace cos_value_tan_sum_identity_l801_801768

noncomputable def cos_theta (θ : ℝ) (h1 : Real.sin θ = -4/5) (h2 : Real.sin θ / Real.cos θ > 0) : ℝ :=
  -Real.sqrt (1 - (Real.sin θ) ^ 2)

theorem cos_value (θ : ℝ) (h1 : Real.sin θ = -4/5) (h2 : Real.sin θ / Real.cos θ > 0) :
  Real.cos θ = -3/5 :=
  by
    unfold cos_theta
    sorry

theorem tan_sum_identity (θ : ℝ) (h1 : Real.sin θ = -4/5) (h2 : Real.sin θ / Real.cos θ > 0) :
  Real.tan (θ + Real.pi / 4) = -7 :=
  by
    have h_cos : Real.cos θ = -3/5 := cos_value θ h1 h2
    have h_tan : Real.tan θ = 4/3 :=
      by
        unfold Real.tan
        rw [h1, h_cos]
        field_simp
        norm_num
    field_simp
    linarith
    -- Use the tangent sum identity here
    sorry

end cos_value_tan_sum_identity_l801_801768


namespace checkerboard_cannot_be_covered_l801_801702

theorem checkerboard_cannot_be_covered (m n : ℕ) :
  (m = 5) ∧ (n = 5) → ¬(even m ∧ even n ∧ even (m * n)) :=
by
  intros
  sorry

end checkerboard_cannot_be_covered_l801_801702


namespace total_cost_of_fencing_l801_801738

def costOfFencing (lengths rates : List ℝ) : ℝ :=
  List.sum (List.zipWith (· * ·) lengths rates)

theorem total_cost_of_fencing :
  costOfFencing [14, 20, 35, 40, 15, 30, 25]
                [2.50, 3.00, 3.50, 4.00, 2.75, 3.25, 3.75] = 610.00 :=
by
  sorry

end total_cost_of_fencing_l801_801738


namespace find_valid_tax_range_l801_801448

noncomputable def valid_tax_range (t : ℝ) : Prop :=
  let initial_consumption := 200000
  let price_per_cubic_meter := 240
  let consumption_reduction := 2.5 * t * 10^4
  let tax_revenue := (initial_consumption - consumption_reduction) * price_per_cubic_meter * (t / 100)
  tax_revenue >= 900000

theorem find_valid_tax_range (t : ℝ) : 3 ≤ t ∧ t ≤ 5 ↔ valid_tax_range t :=
sorry

end find_valid_tax_range_l801_801448


namespace incenter_on_common_chord_l801_801518

/--
Given a triangle ABC where points M and N lie on the line containing side AC
such that MA = AB, NC = CB, and the order of points on the line is M, A, C, N,
prove that the incenter of triangle ABC lies on the common chord of the circles
circumscribed around triangles MCB and NAB.
-/
theorem incenter_on_common_chord 
  (A B C M N I : Point)
  (h1 : Collinear M A C)
  (h2 : Collinear A C N)
  (h3 : distance M A = distance A B)
  (h4 : distance N C = distance C B)
  (h_order : Ordered M A C N)
  (hI : incenter I A B C) :
  I ∈ common_chord (circumcircle M C B) (circumcircle N A B) :=
sorry

end incenter_on_common_chord_l801_801518


namespace derivative_at_0_l801_801217

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x * Real.sin x - 7 * x

theorem derivative_at_0 : deriv f 0 = -6 := 
by
  sorry

end derivative_at_0_l801_801217


namespace side_length_of_inscribed_square_l801_801892

noncomputable def side_length_of_square {P Q R : Type} [metric_space P] [metric_space Q] [metric_space R]
  (leg1 leg2 : ℝ) (hypotenuse side_length : ℝ) : Prop :=
  let right_angle := π / 2 in
  let hypotenuse := real.sqrt (leg1^2 + leg2^2) in
  ∃ (side_sqr : ℝ), hypotenuse = real.sqrt (leg1^2 + leg2^2) ∧ side_sqr = side_length

theorem side_length_of_inscribed_square {P Q R : Type} [metric_space P] [metric_space Q] [metric_space R] :
  side_length_of_square 5 12 (real.sqrt ((5:ℝ)^2 + (12:ℝ)^2)) (96.205 / 20.385) :=
sorry

end side_length_of_inscribed_square_l801_801892


namespace g_min_value_l801_801653

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 2

def g (t : ℝ) : ℝ :=
  if t < 0 then t^2 + 1
  else if t < 1 then t
  else t^2 - 2 * t + 2

theorem g_min_value (t : ℝ) : 
  (∀ x ∈ set.Icc t (t+1), f x ≥ g t) ∧
  (∃ x ∈ set.Icc t (t+1), f x = g t) :=
begin
  sorry
end

end g_min_value_l801_801653


namespace converse_proposition_l801_801234

-- Define the propositions
def P : Prop := ∀ (Q : Type), rect Q → (diags_equal Q)
def Q : Prop := ∀ (Q : Type), (diags_equal Q) → (rect Q)

-- The problem statement given to prove that Q is the converse of P
theorem converse_proposition : (Q ↔ ¬P) → Q := by
  sorry

end converse_proposition_l801_801234


namespace max_value_ineq_l801_801969

open Real

theorem max_value_ineq (n : ℕ) (hₙ : n ≥ 2) 
  (a b : Fin n → ℝ) 
  (ha : ∀ i, a i > 0) (hb : ∀ i, b i ≥ 0) 
  (hsum : (∑ i, a i) + (∑ i, b i) = n)
  (hprod : (∏ i, a i) + (∏ i, b i) = 1 / 2) :
  (∏ i, a i) * (∑ i, b i / a i) ≤ 1 / 2 := 
sorry

end max_value_ineq_l801_801969


namespace solves_fn_eq_x_l801_801155

def f (x : ℝ) : ℝ := 1 + 2 / x
def f_n (n : ℕ) (x : ℝ) : ℝ := Nat.iterate f n x

theorem solves_fn_eq_x (n : ℕ) (hn : n > 0) : 
  ∃ x : ℝ, (x = f_n n x) ↔ (x = -1 ∨ x = 2) := 
    sorry

end solves_fn_eq_x_l801_801155


namespace ellipse_eq_proof_center_of_circle_P_proof_max_Y_value_proof_l801_801410

noncomputable def ellipse_eq : Prop :=
  let foci_1 := (-Real.sqrt 2, 0)
  let foci_2 := (Real.sqrt 2, 0)
  let A := (Real.sqrt 2, Real.sqrt 3 / 3)
  ∃ (a b : Real), (foci_1.1)^2 + foci_1.2^2 = 2 ∧ 
                   (foci_2.1)^2 + foci_2.2^2 = 2 ∧
                   a^2 = 3 ∧ b^2 = 1 ∧ 
                   (A.1)^2 / a^2 + (A.2)^2 / b^2 = 1

theorem ellipse_eq_proof : ellipse_eq :=
by
  -- proof here
  sorry

noncomputable def center_of_circle_P : Prop :=
  ∃ t : Real, -1 < t ∧ t < 1 ∧ 
              (t^2 = 3 * (1 - t^2)) ∧ 
              ((0, t) = (0, Real.sqrt 3 / 2) ∨ 
               (0, t) = (0, -Real.sqrt 3 / 2))

theorem center_of_circle_P_proof : center_of_circle_P :=
by
  -- proof here
  sorry

noncomputable def max_Y_value : Prop :=
  ∃ (t : Real) (Q : Real × Real), t ∈ 0..1 ∧
                                   Q.2 = t + Real.sqrt(3 * (1 - t^2)) ∧
                                   Q.2 ≤ 2

theorem max_Y_value_proof : max_Y_value :=
by
  -- proof here
  sorry

end ellipse_eq_proof_center_of_circle_P_proof_max_Y_value_proof_l801_801410


namespace construct_pairwise_tangent_circles_l801_801789

-- Define the three points A, B, and C in a 2D plane.
variables (A B C : EuclideanSpace ℝ (Fin 2))

/--
  Given three points A, B, and C in the plane, 
  it is possible to construct three circles that are pairwise tangent at these points.
-/
theorem construct_pairwise_tangent_circles (A B C : EuclideanSpace ℝ (Fin 2)) :
  ∃ (O1 O2 O3 : EuclideanSpace ℝ (Fin 2)) (r1 r2 r3 : ℝ),
    r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧
    dist O1 O2 = r1 + r2 ∧
    dist O2 O3 = r2 + r3 ∧
    dist O3 O1 = r3 + r1 ∧
    dist O1 A = r1 ∧ dist O2 B = r2 ∧ dist O3 C = r3 :=
sorry

end construct_pairwise_tangent_circles_l801_801789


namespace tangent_line_eqn_at_point_l801_801586

noncomputable def f (x : ℝ) : ℝ := x^2 + 1 / x

def point_of_tangency : ℝ × ℝ := (1, 2)

theorem tangent_line_eqn_at_point :
  let x := point_of_tangency.1
  let y := point_of_tangency.2 
  let dydx := (deriv f x)
  (dydx = 1) → 
  (let k := dydx in 
   y - 2 = k * (x - 1)) →
  x - y + 1 = 0 :=
by
  sorry

end tangent_line_eqn_at_point_l801_801586


namespace value_of_b_minus_a_l801_801056

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x / 2)

theorem value_of_b_minus_a (a b : ℝ) (h1 : ∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (-1 : ℝ) 2) (h2 : ∀ x, f x = 2 * Real.sin (x / 2)) : 
  b - a ≠ 14 * Real.pi / 3 :=
sorry

end value_of_b_minus_a_l801_801056


namespace solution_quadratic_1_solution_quadratic_2_l801_801562

noncomputable def solve_quadratic_1 : Set ℂ := { x : ℂ | x^2 - 5 * x + 6 = 0 }

theorem solution_quadratic_1 :
  solve_quadratic_1 = {2, 3} :=
sorry

noncomputable def solve_quadratic_2 : Set ℂ := { x : ℂ | 2 * x^2 - 4 * x - 1 = 0 }

theorem solution_quadratic_2 :
  solve_quadratic_2 = { (4 + Complex.sqrt 24) / 4, (4 - Complex.sqrt 24) / 4 } :=
sorry

end solution_quadratic_1_solution_quadratic_2_l801_801562


namespace find_missing_number_l801_801928

-- Define the known values
def numbers : List ℕ := [1, 22, 24, 25, 26, 27, 2]
def specified_mean : ℕ := 20
def total_counts : ℕ := 8

-- The theorem statement
theorem find_missing_number : (∀ (x : ℕ), (List.sum (x :: numbers) = specified_mean * total_counts) → x = 33) :=
by
  sorry

end find_missing_number_l801_801928


namespace expenditure_of_5_yuan_neg_l801_801911

theorem expenditure_of_5_yuan_neg :
  (∀ income : Int, expenditure : Int, (income = 7 → income = 7) → (∀ exp, exp = 5 → exp = -exp) → (5 = -5)) sorry

end expenditure_of_5_yuan_neg_l801_801911


namespace inscribed_square_in_triangle_l801_801681

theorem inscribed_square_in_triangle (a : ℝ) (h : ℝ) (x : ℝ) 
  (h1 : h * a = 12 * x^2)
  (h2 : x * (a + h) = a * h) :
  (x = (3 + √6) / 6 * a ∨ x = (3 - √6) / 6 * a) ∧ (h = (5 + 2 * √6) * a ∨ h = (5 - 2 * √6) * a) :=
by
  sorry

end inscribed_square_in_triangle_l801_801681


namespace angle_sum_heptagon_l801_801821

-- Declare the points as variables within the geometric structure.
variables (A B C D E F G O : Type) [metric_space G]
variables [metric_space_types.metric_space A B C D E F]
variables [affine_space.seconds_geom A B C D E F G]

-- Define the angles at vertices of the heptagon.
def angle_at_vertex (u v w : G) := ∠ u v w

-- Define the specific angles formed by lines through point O to vertices of the heptagon.
def internal_angles_at_O := [∠ O G A, ∠ O A B, ∠ O B C, ∠ O C D, ∠ O D E, ∠ O E F, ∠ O F G]

-- Define the desired angle sum as the theorem to be proven.
theorem angle_sum_heptagon :
  let angle_sum := ∑ (x : G), angle_at_vertex x x x
  angle_sum = 540 :=
by sorry

end angle_sum_heptagon_l801_801821


namespace possible_k_value_l801_801842

theorem possible_k_value (a n k : ℕ) (h1 : n > 1) (h2 : 10^(n-1) ≤ a ∧ a < 10^n)
    (h3 : b = a * (10^n + 1)) (h4 : k = b / a^2) (h5 : b = a * 10 ^n + a) :
  k = 7 := 
sorry

end possible_k_value_l801_801842


namespace janine_total_pages_l801_801135

-- Define the conditions
def books_last_month : ℕ := 5
def books_this_month : ℕ := 2 * books_last_month
def books_per_page : ℕ := 10

-- Define the total number of pages she read in two months
def total_pages : ℕ :=
  let total_books := books_last_month + books_this_month
  total_books * books_per_page

-- State the theorem to be proven
theorem janine_total_pages : total_pages = 150 :=
by
  sorry

end janine_total_pages_l801_801135


namespace total_amount_proof_l801_801579

-- Let's define the conditions of the problem as given.
variables (CI r t n : ℝ) (P : ℝ)

-- The initial conditions given in the problem
noncomputable def given_conditions : Prop := 
  CI = 420 ∧ 
  r = 0.10 ∧
  t = 2 ∧ 
  n = 1

-- Definition of the compound interest formula
noncomputable def compound_interest_formula (P r n t : ℝ) : ℝ :=
  P * ((1 + r / n) ^ (n * t))

-- Calculating the total amount at the end
noncomputable def total_amount (P CI r n t : ℝ) : ℝ :=
  P * ((1 + r / n) ^ (n * t)) + CI

-- The proof goal: show that the total amount received by Sunil is Rs. 2420
theorem total_amount_proof (CI r t n : ℝ) (P : ℝ) (h : given_conditions CI r t n) :
  total_amount P CI r n t = 2420 :=
by {
  -- Define constants from the conditions
  have h1 : CI = 420 := h.1,
  have h2 : r = 0.10 := h.2.1,
  have h3 : t = 2 := h.2.2.1,
  have h4 : n = 1 := h.2.2.2,

  -- Substitute the constants into the definition
  sorry
}

end total_amount_proof_l801_801579


namespace sufficient_but_not_necessary_condition_l801_801031

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a > 0) → (a^2 + a ≥ 0) ∧ ((a^2 + a ≥ 0) → (a > 0 ∨ a < -1)) :=
by
  intros h1  -- Assume a > 0
  have h2 : a^2 + a ≥ 0 := calc
    a^2 + a = a * (a + 1) : by ring
    ... ≥ 0 : by
      cases lt_trichotomy a (-1) with h3 h3,
      { left, exact (le_mul_iff_one_le_left (by linarith)).2 (by linarith) },
      cases h3 with h4 h4,
      { rw h4, norm_num },
      { right, exact (le_trans (mul_nonneg (by linarith) (by linarith)) (le_of_eq ((mul_one 1).symm))) }
  exact ⟨h1, λ h, or.inr (lt_of_le_of_ne (by linarith) (ne_of_lt (by linarith)))⟩ 

end sufficient_but_not_necessary_condition_l801_801031


namespace banker_l801_801202

noncomputable def present_worth (BG : ℝ) (n : ℝ) (r : ℝ) : ℝ :=
BG / ((1 + r)^n - (1 + r * n))

theorem banker's_gain_calculation :
  present_worth 120 4 0.15 ≈ 805.69 :=
by
  sorry

end banker_l801_801202


namespace number_of_integers_with_floor_sqrt_eq_seven_l801_801076

theorem number_of_integers_with_floor_sqrt_eq_seven :
  (finset.card (finset.filter (λ x : ℕ, (nat.floor (real.sqrt x)) = 7) (finset.range 64))) = 15 :=
by
  have h_min : 49 = 7 * 7 := by norm_num
  have h_max : 64 = 8 * 8 := by norm_num
  sorry

end number_of_integers_with_floor_sqrt_eq_seven_l801_801076


namespace system_of_equations_solution_exists_l801_801904

theorem system_of_equations_solution_exists :
  ∃ (x y : ℝ), 
    (4 * x^2 + 8 * x * y + 16 * y^2 + 2 * x + 20 * y = -7) ∧
    (2 * x^2 - 16 * x * y + 8 * y^2 - 14 * x + 20 * y = -11) ∧
    (x = 1/2) ∧ (y = -3/4) :=
by
  sorry

end system_of_equations_solution_exists_l801_801904


namespace length_segment_FG_l801_801466

-- Defining the problem and conditions.
variable (ABCD : Type) [Parallelogram ABCD]
variable (BE : ℝ)
variable (CD DE : ℝ)

-- Assumptions based on the problem conditions.
axiom parallelogram_opposite_sides_equal :
  ∀ (AB CD : ℝ) (AD BC : ℝ), Parallelogram ABCD →
    (AB = CD ∧ AD = BC ∧ parallel AB CD ∧ parallel AD BC)

axiom segment_BE_length : BE = 6
axiom segments_CD_DE_equal : CD = DE

-- Goal: Proving the length of segment FG.
theorem length_segment_FG : FG = 1 := 
sorry

end length_segment_FG_l801_801466


namespace parallel_line_eq_5_units_away_l801_801063

-- Given conditions: the equation of line y = 5/3 * x + 10, and a parallel line L that is 5 units away
def line_eq (x : ℝ) : ℝ := (5 / 3) * x + 10

theorem parallel_line_eq_5_units_away :
  ∃ c_2 : ℝ, (c_2 = 10 + (5 * real.sqrt 34 / 3) ∨ c_2 = 10 - (5 * real.sqrt 34 / 3)) ∧ 
  ∀ x : ℝ, (line_eq x - ((5 / 3) * x + c_2)) = 5 :=
begin
  sorry
end

end parallel_line_eq_5_units_away_l801_801063


namespace fraction_of_number_l801_801262

theorem fraction_of_number (a b c : ℝ) (h : c = a / b * 89473) :
  7 / 25 * 89473 = 25052.44 :=
by {
  apply h,
  sorry
}

end fraction_of_number_l801_801262


namespace zero_point_in_interval_2_3_l801_801333

noncomputable def f (x : ℝ) : ℝ := (2 / x) + real.log (1 / (x - 1))

theorem zero_point_in_interval_2_3 :
  ∃ x, (2 < x ∧ x < 3) ∧ f x = 0 :=
sorry

end zero_point_in_interval_2_3_l801_801333


namespace penny_remaining_money_l801_801882

theorem penny_remaining_money (initial_money : ℤ) (socks_pairs : ℤ) (socks_cost_per_pair : ℤ) (hat_cost : ℤ) :
  initial_money = 20 → socks_pairs = 4 → socks_cost_per_pair = 2 → hat_cost = 7 → 
  initial_money - (socks_pairs * socks_cost_per_pair + hat_cost) = 5 := 
by
  intros h₁ h₂ h₃ h₄
  sorry

end penny_remaining_money_l801_801882


namespace football_game_spectators_l801_801108

theorem football_game_spectators (total_wristbands wristbands_per_person : ℕ)
  (h1 : total_wristbands = 250) (h2 : wristbands_per_person = 2) : 
  total_wristbands / wristbands_per_person = 125 :=
by
  sorry

end football_game_spectators_l801_801108


namespace math_problem_solution_l801_801498

noncomputable def math_problem (g : ℝ → ℝ) :=
  ∀ x y : ℝ, g(1) = 1 ∧ g(x^3 - y^3) = (x - y) * (g(x) + g(y))

theorem math_problem_solution (g : ℝ → ℝ) (h: math_problem g) : (∀ x y : ℝ, g(2) = 1) := by
  sorry

end math_problem_solution_l801_801498


namespace find_s_l801_801231

theorem find_s (x y : Real -> Real) : 
  (x 2 = 2 ∧ y 2 = 5) ∧ 
  (x 6 = 6 ∧ y 6 = 17) ∧ 
  (x 10 = 10 ∧ y 10 = 29) ∧ 
  (∀ x, y x = 3 * x - 1) -> 
  (y 34 = 101) := 
by 
  sorry

end find_s_l801_801231


namespace cos_alpha_value_l801_801019

theorem cos_alpha_value (α : ℝ) (h₁ : 0 < α ∧ α < (π / 2)) (h₂ : cos (α + (π / 3)) = -2 / 3) :
  cos α = (sqrt 15 - 2) / 6 :=
sorry

end cos_alpha_value_l801_801019


namespace evaluate_expression_l801_801340

theorem evaluate_expression (x : ℝ) (h : |7 - 8 * (x - 12)| - |5 - 11| = 73) : x = 3 :=
  sorry

end evaluate_expression_l801_801340


namespace Q_is_orthocenter_of_BDF_l801_801519

-- Definitions of the problem
variables {A B C P Q D E F : Point}
variables {angle_ACP_angle_BCQ angle_CAP_angle_BAQ : Prop}
variables {feets_perpendiculars : is_perpendicular P BC D ∧ is_perpendicular P CA E ∧ is_perpendicular P AB F}
variables {angle_DEF_90 : ∠ D E F = 90}

-- The theorem to prove
theorem Q_is_orthocenter_of_BDF 
  (h1 : ∠ A C P = ∠ B C Q)
  (h2 : ∠ C A P = ∠ B A Q)
  (h3 : is_perpendicular P BC D)
  (h4 : is_perpendicular P CA E)
  (h5 : is_perpendicular P AB F)
  (h6 : ∠ D E F = 90) :
  is_orthocenter Q B D F :=
sorry

end Q_is_orthocenter_of_BDF_l801_801519


namespace good_functions_finite_zeros_l801_801142

def S : Type := ℤ → ℝ

def g (f : S) : S := λ x, f (x + 1) - f x

def isGood (f : S) : Prop := ∃ n : ℕ, iterate g n f = (λ x, 0)

theorem good_functions_finite_zeros (s t : S) (h_s : isGood s) (h_t : isGood t) (h_diff : s ≠ t) :
  { m : ℤ | s m = t m }.finite :=
sorry

end good_functions_finite_zeros_l801_801142


namespace max_special_pairs_l801_801006

open Finset

theorem max_special_pairs (n : ℕ) (hn_odd : Odd n) (hn_gt_one : n > 1) :
  ∃ p : Equiv.Perm (Fin n), (∃ S : ℕ, S = (n + 1) * (n + 3) / 8) :=
by
  sorry

end max_special_pairs_l801_801006


namespace remaining_pipes_l801_801246

noncomputable def largest_triangular_number_leq (n : ℕ) (total_pipes : ℕ) : ℕ :=
  let triangular_number (k : ℕ) := k * (k + 1) / 2
  if h : ∃ k : ℕ, k ≤ n ∧ triangular_number k ≤ total_pipes
  then (nat.find h)
  else 0

theorem remaining_pipes (total_pipes : ℕ) : total_pipes = 1200 → 
  (total_pipes - largest_triangular_number_leq 48 1200 * (largest_triangular_number_leq 48 1200 + 1) / 2 = 24) :=
by
  intros h
  sorry

end remaining_pipes_l801_801246


namespace edward_original_amount_l801_801719

theorem edward_original_amount (spent left total : ℕ) (h1 : spent = 13) (h2 : left = 6) (h3 : total = spent + left) : total = 19 := by 
  sorry

end edward_original_amount_l801_801719


namespace probability_at_least_one_pair_women_l801_801292

/-- There are 7 men and 7 women. Prove that the probability of having 
    at least one pair of women when the group is randomly divided 
    into pairs is approximately 0.96. -/
theorem probability_at_least_one_pair_women : 
  let total_ways := (Nat.factorial 14) / ((Nat.factorial 2)^7 * (Nat.factorial 7)),
      favorable_ways := nat.choose 7 2 * (Nat.factorial 12) / ((Nat.factorial 2)^6 * (Nat.factorial 6)),
      probability := favorable_ways.to_nat_cast / total_ways.to_nat_cast
  in probability ≈ 0.96 :=
by
  sorry

end probability_at_least_one_pair_women_l801_801292


namespace minimum_value_l801_801226

open Real

theorem minimum_value (a : ℝ) (m n : ℝ) (h_a : a > 0) (h_a_not_one : a ≠ 1) 
                      (h_mn : m * n > 0) (h_point : -m - n + 1 = 0) :
  (1 / m + 2 / n) = 3 + 2 * sqrt 2 :=
by
  -- proof should go here
  sorry

end minimum_value_l801_801226


namespace non_vegan_gluten_cupcakes_eq_28_l801_801993

def total_cupcakes : ℕ := 80
def gluten_free_cupcakes : ℕ := total_cupcakes / 2
def vegan_cupcakes : ℕ := 24
def vegan_gluten_free_cupcakes : ℕ := vegan_cupcakes / 2
def non_vegan_cupcakes : ℕ := total_cupcakes - vegan_cupcakes
def gluten_cupcakes : ℕ := total_cupcakes - gluten_free_cupcakes
def non_vegan_gluten_cupcakes : ℕ := gluten_cupcakes - vegan_gluten_free_cupcakes

theorem non_vegan_gluten_cupcakes_eq_28 :
  non_vegan_gluten_cupcakes = 28 := by
  sorry

end non_vegan_gluten_cupcakes_eq_28_l801_801993


namespace isabella_hair_length_after_haircut_cm_l801_801828

theorem isabella_hair_length_after_haircut_cm :
  let initial_length_in : ℝ := 18  -- initial length in inches
  let growth_rate_in_per_week : ℝ := 0.5  -- growth rate in inches per week
  let weeks : ℝ := 4  -- time in weeks
  let hair_trimmed_in : ℝ := 2.25  -- length of hair trimmed in inches
  let cm_per_inch : ℝ := 2.54  -- conversion factor from inches to centimeters
  let final_length_in := initial_length_in + growth_rate_in_per_week * weeks - hair_trimmed_in  -- final length in inches
  let final_length_cm := final_length_in * cm_per_inch  -- final length in centimeters
  final_length_cm = 45.085 := by
  sorry

end isabella_hair_length_after_haircut_cm_l801_801828


namespace time_to_pass_man_l801_801306

noncomputable def train_pass_time : ℝ :=
  let speed_kmph : ℝ := 54
  let speed_mps : ℝ := speed_kmph * (1000 / 3600)
  let platform_length : ℝ := 180.0144
  let time_to_pass_platform : ℝ := 32
  let total_length := speed_mps * time_to_pass_platform
  let train_length := total_length - platform_length
  train_length / speed_mps

theorem time_to_pass_man : train_pass_time ≈ 20 :=
by
  have speed_mps : ℝ := 54 * (1000 / 3600)
  have platform_length : ℝ := 180.0144
  have time_to_pass_platform : ℝ := 32
  have total_length := speed_mps * time_to_pass_platform
  have train_length := total_length - platform_length
  have time_to_pass := train_length / speed_mps
  show time_to_pass ≈ 20
  {
    exact_mod_cast (eq.trans (by ring) (by norm_num1))
  }
  sorry

end time_to_pass_man_l801_801306


namespace stack_crates_height_l801_801250

theorem stack_crates_height :
  ∀ a b c : ℕ, (3 * a + 4 * b + 5 * c = 50) ∧ (a + b + c = 12) → false :=
by
  sorry

end stack_crates_height_l801_801250


namespace tan_pi_over_12_plus_tan_5pi_over_12_eq_4_l801_801524

-- Definitions for given conditions
def cos_pi_over_12 : ℝ := (Real.sqrt 6 + Real.sqrt 2) / 4
def cos_5pi_over_12 : ℝ := (Real.sqrt 6 - Real.sqrt 2) / 4

-- The theorem to be proved
theorem tan_pi_over_12_plus_tan_5pi_over_12_eq_4 : 
  Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 :=
by sorry

end tan_pi_over_12_plus_tan_5pi_over_12_eq_4_l801_801524


namespace euler_line_acutetri_intersects_euler_line_obtusetri_intersects_l801_801637

-- Definitions of geometrical conditions
structure Triangle :=
(a b c : Point)
(angle_a angle_b angle_c : ℝ)

-- Definitions of acute-angled and obtuse-angled triangles
def acute_angled (T : Triangle) : Prop :=
T.angle_a < 90 ∧ T.angle_b < 90 ∧ T.angle_c < 90

def obtuse_angled (T : Triangle) : Prop :=
(T.angle_a > 90 ∨ T.angle_b > 90 ∨ T.angle_c > 90)

structure EulerLine :=
(O H : Point) -- Circumcenter and Orthocenter

-- Euler's line definition
def euler_line_intersects (T : Triangle) (line : EulerLine) (sides : Set (Triangle → (Point × Point))) : Prop :=
∀ side ∈ sides, line.O ∈ side ∧ line.H ∈ side

-- The proof problem:
theorem euler_line_acutetri_intersects {T : Triangle} (hT : acute_angled T) (line : EulerLine) :
  euler_line_intersects T line {side | side = (λ T, (T.a, T.b)) ∨ side = (λ T, (T.a, T.c))} :=
sorry

theorem euler_line_obtusetri_intersects {T : Triangle} (hT : obtuse_angled T) (line : EulerLine) :
  euler_line_intersects T line {side | side = (λ T, (T.a, T.b)) ∨ side = (λ T, (T.b, T.c))} :=
sorry

end euler_line_acutetri_intersects_euler_line_obtusetri_intersects_l801_801637


namespace candle_height_half_after_9_hours_l801_801624

-- Define the initial heights and burn rates
def initial_height_first : ℝ := 12
def burn_rate_first : ℝ := 2
def initial_height_second : ℝ := 15
def burn_rate_second : ℝ := 3

-- Define the height functions after t hours
def height_first (t : ℝ) : ℝ := initial_height_first - burn_rate_first * t
def height_second (t : ℝ) : ℝ := initial_height_second - burn_rate_second * t

-- Prove that at t = 9, the height of the first candle is half the height of the second candle
theorem candle_height_half_after_9_hours : height_first 9 = 0.5 * height_second 9 := by
  sorry

end candle_height_half_after_9_hours_l801_801624


namespace balloon_count_l801_801293

theorem balloon_count (total_balloons red_balloons blue_balloons black_balloons : ℕ) 
  (h_total : total_balloons = 180)
  (h_red : red_balloons = 3 * blue_balloons)
  (h_black : black_balloons = 2 * blue_balloons) :
  red_balloons = 90 ∧ blue_balloons = 30 ∧ black_balloons = 60 :=
by
  sorry

end balloon_count_l801_801293


namespace tan_sum_pi_over_12_l801_801532

theorem tan_sum_pi_over_12 : 
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12)) = 4 := 
sorry

end tan_sum_pi_over_12_l801_801532


namespace integer_solution_count_l801_801331

theorem integer_solution_count : 
  ∃! x : ℤ, (x - 3)^(27 - x^2) = 1 :=
sorry

end integer_solution_count_l801_801331


namespace calculator_keys_l801_801595

def clear_key : Type := String
def off_key : Type := String

def key_clear_screen (key : clear_key) : Prop := 
  key = "ON/C"

def key_off_function (key : off_key) : Prop := 
  key = "power off key"

theorem calculator_keys :
  ∃ (clear_screen_key : clear_key) (off_function_key : off_key), 
    key_clear_screen clear_screen_key ∧ key_off_function off_function_key := by
  exists "ON/C"
  exists "power off key"
  split
  . rfl
  . rfl

end calculator_keys_l801_801595


namespace purple_four_leaved_clovers_count_l801_801454

noncomputable def total_clovers := 850
noncomputable def four_leaf_probability := 0.273
noncomputable def ratio_red := 5.5
noncomputable def ratio_yellow := 7.3
noncomputable def ratio_purple := 9.2

theorem purple_four_leaved_clovers_count :
  let total_four_leaved := total_clovers * four_leaf_probability
  let total_four_leaved_int := Int.floor total_four_leaved
  let ratio_sum := ratio_red + ratio_yellow + ratio_purple
  let purple_proportion := ratio_purple / ratio_sum
  let total_purple_four_leaved := purple_proportion * total_four_leaved_int
  total_purple_four_leaved.floor = 97 := by
  sorry

end purple_four_leaved_clovers_count_l801_801454


namespace optimal_strategy_max_pair_product_l801_801698

theorem optimal_strategy_max_pair_product (x : Fin 100 → ℝ) (hx : ∀ i, 0 ≤ x i) (hsum : ∑ i, x i = 1) :
  ∃ p : (Fin 50 → (Fin 100 × Fin 100)), (∀ i j, i ≠ j → p i ≠ p j ∧ p i ≠ (p j).symm) →
    (maxProd : ℝ) (∀ i, (x (p i).fst) * (x (p i).snd)) ≤ (1 / 396) := 
  sorry

end optimal_strategy_max_pair_product_l801_801698


namespace largest_angle_in_PQR_l801_801100

theorem largest_angle_in_PQR 
  (B C : ℝ)
  (hB : B = 46)
  (hC : C = 48)
  (ABC : ∀ A, 180 - A - B - C = 0) :
  ∃ P Q R, (PQR_angle : ℝ) ∧ PQR_angle = 67 := 
sorry

end largest_angle_in_PQR_l801_801100


namespace complex_equality_l801_801020

theorem complex_equality (a b : ℝ) (i : ℂ) (h : i^2 = -1) (h_eq : a - b * i = (1 + i) * i^3) : a = 1 ∧ b = -1 :=
by sorry

end complex_equality_l801_801020


namespace domain_of_f_l801_801580

noncomputable def f (x : ℝ) : ℝ :=
  real.sqrt (1 - real.log x)

theorem domain_of_f :
  {x : ℝ | 0 < x ∧ x ≤ real.exp 1} = 
  {x : ℝ | 0 < x ∧ 1 - real.log x ≥ 0} :=
begin
  sorry
end

end domain_of_f_l801_801580


namespace remainder_13_plus_x_l801_801153

theorem remainder_13_plus_x (x : ℕ) (h1 : 7 * x % 31 = 1) : (13 + x) % 31 = 22 := 
by
  sorry

end remainder_13_plus_x_l801_801153


namespace problem_statement_l801_801688

noncomputable def square : ℝ := sorry -- We define a placeholder
noncomputable def pentagon : ℝ := sorry -- We define a placeholder

axiom eq1 : 2 * square + 4 * pentagon = 25
axiom eq2 : 3 * square + 3 * pentagon = 22

theorem problem_statement : 4 * pentagon = 20.67 := 
by
  sorry

end problem_statement_l801_801688


namespace exist_initial_numbers_exceed_2011_l801_801619

theorem exist_initial_numbers_exceed_2011 :
  ∃ (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 40) (h3 : 1 ≤ b) (h4 : b ≤ 40) (h5 : 1 ≤ c) (h6 : c ≤ 40),
  (∃ (operations : List (ℕ → ℕ)) (final_number : ℕ), operations.foldl (λ acc f, f acc) a > 2011 ∨ 
                                                   operations.foldl (λ acc f, f acc) b > 2011 ∨ 
                                                   operations.foldl (λ acc f, f acc) c > 2011) :=
begin
  -- This part of the problem requires a proof here.
  sorry
end

end exist_initial_numbers_exceed_2011_l801_801619


namespace numberOfGuppies_is_8_l801_801841

noncomputable def numberOfGuppies : ℕ :=
  let foodForGoldfish := 2 * 1 -- total food for 2 Goldfish
  let foodForSwordtails := 3 * 2 -- total food for 3 Swordtails
  let totalFoodForGoldfishAndSwordtails := foodForGoldfish + foodForSwordtails
  let remainingFood := 12 - totalFoodForGoldfishAndSwordtails
  remainingFood / 0.5

theorem numberOfGuppies_is_8 : numberOfGuppies = 8 :=
by
  sorry

end numberOfGuppies_is_8_l801_801841


namespace reflection_through_plane_l801_801494

open Matrix

/-- Define the normal vector n' -/
def n' : ℝ → ℝ → ℝ → (Vector ℝ) := 
  λ x y z, Vector.ofArray [2, -1, 1]

/-- Define the reflection matrix R' -/
def reflection_matrix : Matrix (Fin 3) (Fin 3) ℝ := 
  (λ i j, 
    match i, j with
    | 0, 0 => -1 / 3
    | 0, 1 => -1 / 3
    | 0, 2 => 1 / 3
    | 1, 0 => 1 / 3
    | 1, 1 => 2 / 3
    | 1, 2 => 4 / 3
    | 2, 0 => 1 / 3
    | 2, 1 => 2 / 3
    | 2, 2 => 2 / 3)

theorem reflection_through_plane (v : Vector ℝ) : 
  let v_proj : Vector ℝ := (reflection_matrix ⬝ v) in
  v_proj = reflection_matrix ⬝ v := 
  -- Proof steps would go here
  sorry

end reflection_through_plane_l801_801494


namespace tangent_line_at_point_l801_801914

open Real

def curve (x : ℝ) : ℝ := exp x / (x + 1)

def tangent_line (x : ℝ) : ℝ := (exp 1 / 4) * x + (exp 1 / 4)

theorem tangent_line_at_point :
  tangent_line = λ x, (exp 1 / 4) * x + (exp 1 / 4) :=
by
  sorry

end tangent_line_at_point_l801_801914


namespace cone_volume_divided_by_pi_l801_801288

noncomputable def sector_to_cone_volume_divided_by_pi (r : ℝ) (sector_angle : ℝ) : ℝ :=
  let circumference := 2 * π * r
  let arc_length := sector_angle / 360 * circumference
  let base_radius := arc_length / (2 * π)
  let height := real.sqrt (r^2 - base_radius^2)
  let volume := (1 / 3) * π * base_radius^2 * height
  volume / π

theorem cone_volume_divided_by_pi :
  sector_to_cone_volume_divided_by_pi 21 270 = 1146.32 := 
by 
  sorry

end cone_volume_divided_by_pi_l801_801288


namespace domain_of_sqrt_function_l801_801218
open Real

noncomputable def domain (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x, x ∈ D ↔ ∃ y, f x = y

def functionDomain := {x : ℝ | ∃ k : ℤ, (π / 3 + 2 * k * π <= x) ∧ (x <= 2 * π / 3 + 2 * k * π)}

theorem domain_of_sqrt_function :
  domain (λ x, sqrt (2 * sin x - sqrt 3)) functionDomain :=
by
  sorry

end domain_of_sqrt_function_l801_801218


namespace james_end_of_year_balance_l801_801129

theorem james_end_of_year_balance :
  let weekly_investment := 2000
  let starting_balance := 250000
  let weeks_in_year := 52
  let total_investment := weekly_investment * weeks_in_year
  let end_of_year_balance := starting_balance + total_investment
  let windfall := 0.50 * end_of_year_balance
  let final_balance := end_of_year_balance + windfall
  final_balance = 531000 :=
by
  let weekly_investment := 2000
  let starting_balance := 250000
  let weeks_in_year := 52
  let total_investment := weekly_investment * weeks_in_year
  let end_of_year_balance := starting_balance + total_investment
  let windfall := 0.50 * end_of_year_balance
  let final_balance := end_of_year_balance + windfall
  sorry

end james_end_of_year_balance_l801_801129


namespace banana_positions_count_l801_801510

def fruits : Type := ℕ → Prop
def banana_position (n : ℕ) : Prop :=
  (1 ≤ n ∧ n ≤ 40) ∨ (60 ≤ n ∧ n ≤ 99) ∨ (n = 50)

theorem banana_positions_count : 
  (finset.filter (banana_position) (finset.range 100)).card = 21 := by
  sorry

end banana_positions_count_l801_801510


namespace sum_of_vectors_zero_checkerboard_l801_801204

theorem sum_of_vectors_zero_checkerboard :
  let n := 2001
  ∀ (v : ℕ × ℕ → ℝ × ℝ), 
  (∀ i j, (i + j) % 2 = 0 → v (i, j) = (i, j)) →
  (∀ i j i' j', (i + j) % 2 = 0 ∧ (i' + j') % 2 = 1 → 
    v (i, j) - v (i', j') = 
    (↑i - ↑i', ↑j - ↑j')) →
  ∑ (i j i' j' : ℕ) in finset.range n ×ˢ finset.range n, v(i, j) - v(i', j') = (0, 0) := 
by
  sorry

end sum_of_vectors_zero_checkerboard_l801_801204


namespace ratio_a_b_eq_3_4_l801_801147

theorem ratio_a_b_eq_3_4 (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : (3 - 4 * complex.I) * (a + b * complex.I) = (3 * a + 4 * b)) : a/b = 3/4 :=
  sorry

end ratio_a_b_eq_3_4_l801_801147


namespace number_times_quarter_squared_eq_four_cubed_l801_801647

theorem number_times_quarter_squared_eq_four_cubed : 
  ∃ (number : ℕ), number * (1 / 4 : ℚ) ^ 2 = (4 : ℚ) ^ 3 ∧ number = 1024 :=
by 
  use 1024
  sorry

end number_times_quarter_squared_eq_four_cubed_l801_801647


namespace remainder_5_pow_2023_mod_11_l801_801001

theorem remainder_5_pow_2023_mod_11 : (5^2023) % 11 = 4 :=
by
  have h1 : 5^2 % 11 = 25 % 11 := sorry
  have h2 : 25 % 11 = 3 := sorry
  have h3 : (3^5) % 11 = 1 := sorry
  have h4 : 3^1011 % 11 = ((3^5)^202 * 3) % 11 := sorry
  have h5 : ((3^5)^202 * 3) % 11 = (1^202 * 3) % 11 := sorry
  have h6 : (1^202 * 3) % 11 = 3 % 11 := sorry
  have h7 : (5^2023) % 11 = (3 * 5) % 11 := sorry
  have h8 : (3 * 5) % 11 = 15 % 11 := sorry
  have h9 : 15 % 11 = 4 := sorry
  exact h9

end remainder_5_pow_2023_mod_11_l801_801001


namespace ball_distribution_l801_801423

theorem ball_distribution (n k : ℕ) (h₁ : n = 5) (h₂ : k = 4) :
  k^n = 1024 := by
  rw [h₁, h₂]
  norm_num
  sorry

end ball_distribution_l801_801423


namespace concert_attendance_difference_l801_801867

noncomputable def first_concert : ℕ := 65899
noncomputable def second_concert : ℕ := 66018

theorem concert_attendance_difference :
  (second_concert - first_concert) = 119 :=
by
  sorry

end concert_attendance_difference_l801_801867


namespace surface_area_ratio_l801_801630

-- Definitions of the conditions
variables (s p q r : ℝ) (A_cube A_rect : ℝ)

-- Given the side length of the cube, the surface area of the cube
def surface_area_cube : ℝ := 6 * s^2

-- Given the dimensions of the rectangular solid after scaling
def surface_area_rect : ℝ := 2 * s^2 * (p*q + p*r + q*r)

-- The ratio g of the surface area of the cube to the surface area of the rectangular solid
def ratio_g : ℝ := surface_area_cube / surface_area_rect

-- Proof statement
theorem surface_area_ratio (s p q r : ℝ) : 
  surface_area_cube s = 6 * s^2 →
  surface_area_rect s p q r = 2 * s^2 * (p * q + p * r + q * r) →
  ratio_g s p q r = 3 / (p * q + p * r + q * r) :=
by
  intros h_cube h_rect
  rw [surface_area_cube, surface_area_rect] at *
  sorry

end surface_area_ratio_l801_801630


namespace simplify_and_evaluate_expression_l801_801185

theorem simplify_and_evaluate_expression (a : ℂ) (h: a^2 + 4 * a + 1 = 0) :
  ( ( (a + 2) / (a^2 - 2 * a) + 8 / (4 - a^2) ) / ( (a^2 - 4) / a ) ) = 1 / 3 := by
  sorry

end simplify_and_evaluate_expression_l801_801185


namespace product_a_equiv_l801_801010

/-- Definition of a_n as given in the problem. -/
def a (n : ℕ) : ℚ := 
  if n >= 4 then (n^3 + 3*n^2 + 3*n + 7) / (n^4 - 1)
  else 0

/-- The product of a_n from n = 4 to n = 150 equals 5763601 / 150! -/
theorem product_a_equiv : 
  (∏ n in Finset.range 147 \ Finset.range 3, a (n+4)) = 5763601 / Nat.factorial 150 := 
by
  sorry

end product_a_equiv_l801_801010


namespace find_n_l801_801347

theorem find_n (n : ℕ) (h : 2^6 * 3^3 * n = nat.factorial 10) : n = 2100 :=
sorry

end find_n_l801_801347


namespace janine_read_pages_in_two_months_l801_801133

theorem janine_read_pages_in_two_months :
  (let books_last_month := 5
   let books_this_month := 2 * books_last_month
   let total_books := books_last_month + books_this_month
   let pages_per_book := 10
   total_books * pages_per_book = 150) := by
   sorry

end janine_read_pages_in_two_months_l801_801133


namespace equilateral_centers_form_triangle_l801_801515

-- Define the structure of the Triangle
structure Triangle :=
(a b c : ℝ) -- Sides of the triangle

-- Define the centers of the equilateral triangles
structure Center := 
(O1 O2 O3 : Triangle)

-- The main theorem to prove
theorem equilateral_centers_form_triangle (T : Triangle) (C : Center) : 
∃ (O1 O2 O3 : Point), 
  is_center_of_equilateral_triangle O1 (T.a) (T.b) ∧
  is_center_of_equilateral_triangle O2 (T.b) (T.c) ∧
  is_center_of_equilateral_triangle O3 (T.c) (T.a) ∧
  is_equilateral_triangle O1 O2 O3 :=
sorry

-- Additional necessary definitions/functions would be defined here 
-- (dummy definitions for context completeness)
def Point := (ℝ × ℝ)

def is_center_of_equilateral_triangle (O : Point) (a b : ℝ) : Prop := sorry
def is_equilateral_triangle (O1 O2 O3 : Point) : Prop := sorry

end equilateral_centers_form_triangle_l801_801515


namespace integral_inequality_l801_801926

variable {f g : ℝ → ℝ}
variable {I : set ℝ} (hf : ∀ x ∈ I, 0 < f x) (hg : ∀ y ∈ I, 0 < g y)
variable (cf : continuous_on f I) (cg : continuous_on g I)
variable (f_inc : ∀ a b ∈ I, a < b → f a < f b) (g_dec : ∀ a b ∈ I, a < b → g a > g b)

theorem integral_inequality :
  ∫ x in 0..1, f x * g x ≤ ∫ x in 0..1, f x * g (1 - x) :=
sorry

end integral_inequality_l801_801926


namespace gemstones_count_l801_801704

theorem gemstones_count (F B S W SN : ℕ) 
  (hS : S = 1)
  (hSpaatz : S = F / 2 - 2)
  (hBinkie : B = 4 * F)
  (hWhiskers : W = S + 3)
  (hSnowball : SN = 2 * W) :
  B = 24 :=
by
  sorry

end gemstones_count_l801_801704


namespace sparrow_swallow_equations_l801_801648

theorem sparrow_swallow_equations (x y : ℝ) : 
  (5 * x + 6 * y = 16) ∧ (4 * x + y = 5 * y + x) :=
  sorry

end sparrow_swallow_equations_l801_801648


namespace grid_fill_count_l801_801815

theorem grid_fill_count (n : ℕ) (h : n ≥ 3) : 
  let D := n + 1 in
  (∃ fill : (fin n) → (fin n) → ℕ, 
    -- Condition to ensure each cell is filled with numbers from 1 to n^2
    (∀ i j, 1 ≤ fill i j ∧ fill i j ≤ n^2) ∧
    -- Adjacent cells condition
    (∀ i j i' j', (i = i' ∧ abs (j - j') = 1) ∨ (j = j' ∧ abs (i - i') = 1) ∨
                  (abs (i - i') = 1 ∧ abs (j - j') = 1) →
                  abs (fill i j - fill i' j') ≤ D)) →
  -- Conclusion
  (∃ m : ℕ, m = 32) :=
begin
  intros D fill h_fill,
  existsi 32,
  sorry
end

end grid_fill_count_l801_801815


namespace purely_imaginary_z_implies_m_zero_l801_801797

theorem purely_imaginary_z_implies_m_zero (m : ℝ) :
  m * (m + 1) = 0 → m ≠ -1 := by sorry

end purely_imaginary_z_implies_m_zero_l801_801797


namespace triangle_area_of_tangent_circles_l801_801716

open Real

theorem triangle_area_of_tangent_circles (r : ℝ) (h : r > 0) :
    let A := 2r * (2 * sqrt 3 + 3) in
    (equilateral_triangle_area (2 * r * (sqrt 3 + 1)) = 2 * r^2 * (2 * sqrt 3 + 3)) := 
sorry

-- Auxiliary definitions for the equilateral triangle area calculation
def equilateral_triangle_area (a : ℝ) : ℝ := (a^2 * sqrt 3) / 4

end triangle_area_of_tangent_circles_l801_801716


namespace max_value_of_quadratic_l801_801744

def quadratic_max_val (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x : ℝ, (f x = a*x + b*x^2) → (a < 0) → (∃ y : ℝ, ∀ z : ℝ, f y ≥ f z)

theorem max_value_of_quadratic :
  quadratic_max_val (λ x : ℝ, 8 * x - 3 * x^2) 8 (-3) :=
by
  sorry

end max_value_of_quadratic_l801_801744


namespace find_other_endpoint_l801_801877

theorem find_other_endpoint (x₁ y₁ x y x_mid y_mid : ℝ) 
  (h1 : x₁ = 5) (h2 : y₁ = 2) (h3 : x_mid = 3) (h4 : y_mid = 10) 
  (hx : (x₁ + x) / 2 = x_mid) (hy : (y₁ + y) / 2 = y_mid) : 
  x = 1 ∧ y = 18 := by
  sorry

end find_other_endpoint_l801_801877


namespace days_to_clear_land_l801_801260

-- Definitions of all the conditions
def length_of_land := 200
def width_of_land := 900
def area_cleared_by_one_rabbit_per_day_square_yards := 10
def number_of_rabbits := 100
def conversion_square_yards_to_square_feet := 9
def total_area_of_land := length_of_land * width_of_land
def area_cleared_by_one_rabbit_per_day_square_feet := area_cleared_by_one_rabbit_per_day_square_yards * conversion_square_yards_to_square_feet
def area_cleared_by_all_rabbits_per_day := number_of_rabbits * area_cleared_by_one_rabbit_per_day_square_feet

-- Theorem to prove the number of days required to clear the land
theorem days_to_clear_land :
  total_area_of_land / area_cleared_by_all_rabbits_per_day = 20 := by
  sorry

end days_to_clear_land_l801_801260


namespace largest_angle_is_75_l801_801927

noncomputable def largest_angle_in_triangle := 
  let k := 180 / (3 + 4 + 5) in
  5 * k = 75

theorem largest_angle_is_75 :
  largest_angle_in_triangle :=
by
  sorry

end largest_angle_is_75_l801_801927


namespace max_area_isosceles_min_perimeter_isosceles_l801_801173

section TriangleOptimization

variables {a P A : ℝ} -- side a is fixed, P is the given perimeter, A is the given area
variables {b c : ℝ} -- other two sides of the triangle

-- Part (a): Among all triangles with a given side and a given perimeter, the isosceles triangle 
-- (with the given side as its base) has the largest area.
theorem max_area_isosceles (a P : ℝ) (hP : P > a) : ∃ (b c : ℝ), 
  b = c ∧ (by_calc √(P/2 * (P/2 - a) * (P/2 - b) * (P/2 - c))) -- isosceles triangle with side a maximizes area
:= sorry

-- Part (b): Among all triangles with a given side and a given area, the isosceles triangle 
-- (with the given side as its base) has the smallest perimeter.
theorem min_perimeter_isosceles (a A : ℝ) (hA : A > 0) : ∃ (b c : ℝ), 
  b = c ∧ (by_when_calc P = a + b + c ∧ (sqrt((s * (s - a) * (s - b) * (s - c)) = A))) -- isosceles triangle with side a minimizes perimeter
:= sorry

end TriangleOptimization

end max_area_isosceles_min_perimeter_isosceles_l801_801173


namespace boxed_boxed_13_l801_801743

def sum_of_factors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0) |>.sum

def boxed (n : ℕ) : ℕ := sum_of_factors n

theorem boxed_boxed_13 : boxed (boxed 13) = 24 := by
  sorry

end boxed_boxed_13_l801_801743


namespace find_angle_between_vectors_l801_801353

-- Define the vectors
def a : ℝ × ℝ × ℝ := (3, 0, -2)
def b : ℝ × ℝ × ℝ := (1, 4, 0)

noncomputable def angle_between_vectors (a b : ℝ × ℝ × ℝ) : ℝ :=
  real.arccos ((a.1 * b.1 + a.2 * b.2 + a.3 * b.3) /
             (real.sqrt (a.1^2 + a.2^2 + a.3^2) * real.sqrt (b.1^2 + b.2^2 + b.3^2)))

theorem find_angle_between_vectors :
  angle_between_vectors a b = real.arccos (3 / real.sqrt 221) := sorry

end find_angle_between_vectors_l801_801353


namespace geom_series_sum_equality_l801_801194

theorem geom_series_sum_equality (a : ℕ → ℝ) (h1 : ∀ n, log (a (n + 1)) = 1 + log (a n))
    (h2 : (∑ t in finset.range 10, a (2001 + t)) = 2013) :
    (∑ t in finset.range 10, a (2011 + t)) = 2013 * 10^10 := 
  sorry

end geom_series_sum_equality_l801_801194


namespace parallel_lines_a_value_l801_801070

theorem parallel_lines_a_value (a : ℝ) 
  (h_parallel: ∃ k : ℝ, ax + 2y = 0 ∧ x + (a - 1) y + a^2 - 1 = 0 ∧ (a: ℝ)/(1: ℝ) = (2: ℝ)/(a: ℝ - 1)) : 
    a = -1 :=
by
  sorry

end parallel_lines_a_value_l801_801070


namespace problem1_problem2i_problem2ii_l801_801468

-- Define the sequence \(a_n\) with the conditions given
noncomputable def a (n : ℕ) : ℕ :=
if n = 0 then 0 else -- since a_0 is not defined, we use 0
if n = 1 then 1 else
if n % 2 = 0 then a (n / 2 + 1) * a (n / 2 + 1) / 4 else a (n / 2 * 2 + 1) * 4

-- Problem 1: Prove the sum of the sequence given conditions
theorem problem1 (k : ℕ) : (∑ i in finset.range k, a (2 * i + 1)) = (1 / 3) * (4 ^ k - 1) :=
by
  sorry

-- Define the sequences b_k and d_k and their properties
def q (k : ℕ) : ℚ := 2 -- Common ratio q_k is fixed as 2

def b (k : ℕ) : ℚ := 1 / (q k - 1)

noncomputable def a_k (k : ℕ) := 
if k % 2 = 0 then a (k + 1) * q (k / 2 + 1) else a ((k + 1) / 2 * 2 - 1)

-- Problem 2(i): Prove that {b_k} is an arithmetic sequence with common difference 1.
theorem problem2i (k : ℕ) : b (k + 1) - b k = 1 :=
by
  sorry

-- Define d1 and the corresponding sequences for part 2(ii)
def d_k (k : ℕ) : ℚ := 
if k = 1 then 2 else 
if k % 2 = 0 then a_k (k + 1) - a_k k else a_k k - a_k (k - 1)

-- Problem 2(ii): Prove the sum of the first k terms of {d_k}
theorem problem2ii (k : ℕ) : ∑ i in finset.range k, d_k (i + 1) = if a (2) = 2 then k * (k + 3) / 2 else 2 * k^2 :=
by
  sorry

end problem1_problem2i_problem2ii_l801_801468


namespace problem1_problem2_l801_801319

theorem problem1 : 12 - (-18) + (-7) + (-15) = 8 := 
by
  sorry

theorem problem2 : -2^3 + (-5)^2 * (2 / 5) - | -3 | = -1 := 
by
  sorry

end problem1_problem2_l801_801319


namespace marble_count_geometric_sequence_l801_801507

theorem marble_count_geometric_sequence : ∃ k : ℕ, 5 * 3 ^ k > 500 ∧ k = 5 :=
by
  -- let's find the smallest k such that 5 * 3^k > 500
  have h0 : 5 * 3 ^ 0 = 5 := rfl
  have h1 : 5 * 3 ^ 1 = 15 := rfl
  have h2 : 5 * 3 ^ 2 = 45 := rfl
  have h3 : 5 * 3 ^ 3 = 135 := rfl
  have h4 : 5 * 3 ^ 4 = 405 := rfl
  have h5 : 5 * 3 ^ 5 = 1215 := rfl
  use 5
  split
  . exact h5.symm.trans (by norm_num : 5 * 3 ^ 5 > 500)
  . rfl
  sorry

end marble_count_geometric_sequence_l801_801507


namespace incorrect_statement_l801_801119

def class (k : ℕ) : Set ℤ :=
  {x : ℤ | ∃ n : ℤ, x = 5 * n + k}

def is_incorrect (k : ℕ) (x : ℤ) : Prop :=
  ¬ (x ∈ class k)

theorem incorrect_statement : is_incorrect 2 (-2) :=
sorry

end incorrect_statement_l801_801119


namespace heat_equation_solution_l801_801560

noncomputable def initial_condition (x : ℝ) : ℝ :=
  real.sin (2 * real.pi * x) ^ 3

def boundary_condition (t : ℝ) : ℝ := 0

def solution (x t : ℝ) : ℝ :=
  (3 / 4) * real.exp (-4 * real.pi ^ 2 * t) * real.sin (2 * real.pi * x) -
  (1 / 4) * real.exp (-36 * real.pi ^ 2 * t) * real.sin (6 * real.pi * x)

theorem heat_equation_solution (x t : ℝ) (h1 : 0 < x ∧ x < 1)
  (h2 : 0 < t):
  ∂t (solution x t) = ∂xx (solution x t)
  ∧ (solution x 0 = initial_condition x)
  ∧ (solution 0 t = boundary_condition t)
  ∧ (solution 1 t = boundary_condition t) :=
sorry

end heat_equation_solution_l801_801560


namespace triangle_inequality_range_isosceles_triangle_perimeter_l801_801471

-- Define the parameters for the triangle
variables (AB BC AC a : ℝ)
variables (h_AB : AB = 8) (h_BC : BC = 2 * a + 2) (h_AC : AC = 22)

-- Define the lean proof problem for the given conditions
theorem triangle_inequality_range (h_triangle : AB = 8 ∧ BC = 2 * a + 2 ∧ AC = 22) :
  6 < a ∧ a < 14 := sorry

-- Define the isosceles condition and perimeter calculation
theorem isosceles_triangle_perimeter (h_isosceles : BC = AC) :
  perimeter = 52 := sorry

end triangle_inequality_range_isosceles_triangle_perimeter_l801_801471


namespace solutionY_materialB_correct_l801_801186

open Real

-- Definitions and conditions from step a
def solutionX_materialA : ℝ := 0.20
def solutionX_materialB : ℝ := 0.80
def solutionY_materialA : ℝ := 0.30
def mixture_materialA : ℝ := 0.22
def solutionX_in_mixture : ℝ := 0.80
def solutionY_in_mixture : ℝ := 0.20

-- The conjecture to prove
theorem solutionY_materialB_correct (B_Y : ℝ) 
  (h1 : solutionX_materialA = 0.20)
  (h2 : solutionX_materialB = 0.80) 
  (h3 : solutionY_materialA = 0.30) 
  (h4 : mixture_materialA = 0.22)
  (h5 : solutionX_in_mixture = 0.80)
  (h6 : solutionY_in_mixture = 0.20) :
  B_Y = 1 - solutionY_materialA := by 
  sorry

end solutionY_materialB_correct_l801_801186


namespace length_DE_l801_801472

-- Definitions of points and segments
variables {A B C D E : Type} [MetricSpace A]

-- Definitions of segments and lengths
variables (AB AC BC DE : ℝ)
variable h_midpoints : midpoint A B D ∧ midpoint A C E
variable h_BC : segment_length B C = 10

theorem length_DE (h_midpoints : midpoint A B D ∧ midpoint A C E)
    (h_BC : segment_length B C = 10) : segment_length D E = 5 :=
sorry

end length_DE_l801_801472


namespace quadratic_congruence_solution_l801_801898

theorem quadratic_congruence_solution (p : ℕ) (hp : Nat.Prime p) : 
  ∃ n : ℕ, 6 * n^2 + 5 * n + 1 ≡ 0 [MOD p] := 
sorry

end quadratic_congruence_solution_l801_801898


namespace sum_of_digits_of_n_l801_801617

theorem sum_of_digits_of_n (n : ℕ) (h1 : 0 < n) (h2 : (n+1)! + (n+3)! = n! * 1320) : n.digits.sum = 1 := by
  sorry

end sum_of_digits_of_n_l801_801617


namespace eigenvalue_and_inverse_l801_801378

open Matrix

noncomputable def given_matrix (x : ℝ) := ![![1, x], ![0, 2]]
def eigenvector := ![0, 1]
def eigenvalue := 2
def expected_inverse := ![![1, 0], ![0, 1/2]]

theorem eigenvalue_and_inverse :
  ∀ x : ℝ,
    (∃ (λ : ℝ), (given_matrix x).mul_vec eigenvector = λ • eigenvector) ∧
    (λ = eigenvalue) ∧
    (inverse (given_matrix x) = expected_inverse) :=
by
  intros x
  split
  sorry -- Proof for the eigenvalue is 2
  sorry -- Proof for the inverse of the given matrix

end eigenvalue_and_inverse_l801_801378


namespace find_x_pow_24_l801_801437

noncomputable def x := sorry
axiom h : x + 1/x = -Real.sqrt 3

theorem find_x_pow_24 : x ^ 24 = 1 := sorry

end find_x_pow_24_l801_801437


namespace mobile_card_probability_l801_801460

theorem mobile_card_probability 
  (total_cards : ℕ)
  (mobile_cards : ℕ)
  (union_cards : ℕ)
  (P_both_mobile : ℚ)
  (h_total_cards : total_cards = 5)
  (h_mobile_cards : mobile_cards = 3)
  (h_union_cards : union_cards = 2)
  (h_P_both_mobile : P_both_mobile = 3 / 10) :
  ∃ (P_at_most_one_mobile : ℚ), P_at_most_one_mobile = 7 / 10 :=
by
  exists 7 / 10
  sorry

end mobile_card_probability_l801_801460


namespace negation_of_implication_l801_801599

theorem negation_of_implication (x : ℝ) :
  ¬ (x > 1 → x^2 > 1) ↔ (x ≤ 1 → x^2 ≤ 1) :=
by
  sorry

end negation_of_implication_l801_801599


namespace trajectory_of_vertex_A_l801_801064

theorem trajectory_of_vertex_A :
  ∀ (A B C : ℝ × ℝ),
  B = (0, -4) → 
  C = (0, 4) →
  let perimeter := 20 in
  let BC := 8 in
  let AB_plus_AC := perimeter - BC in
  (∃ a b : ℝ, a = 6 ∧ (b^2 = 20 ∧ (∀ x y : ℝ, (x ≠ 0) → ((x, y) ∈ {p | (p.1^2 / 20) + (p.2^2 / 36) = 1}))))
sorry

end trajectory_of_vertex_A_l801_801064


namespace exist_balanced_sequence_l801_801750

theorem exist_balanced_sequence (n : ℕ) (points : list bool) (h_length : points.length = 4 * n)
  (h_white : points.count tt = 2 * n) :
  ∃ k : ℕ, k ≤ 2 * n + 1 ∧ points.slice k (k + 2 * n).count tt = n := sorry

end exist_balanced_sequence_l801_801750


namespace expected_visible_colors_l801_801618

noncomputable def c (n : ℕ) : ℝ := sorry

theorem expected_visible_colors :
  tendsto (λ n, c n) at_top (𝓝 (4 * Real.pi)) := sorry

end expected_visible_colors_l801_801618


namespace proof_problem_l801_801035

variables {R : Type*} [CommRing R]

-- f is a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Variable definitions for the conditions
variables (h_odd : is_odd f)
(h_f1 : f 1 = 1)
(h_period : ∀ x, f (x + 6) = f x + f 3)

-- The proof problem statement
theorem proof_problem : f 2015 + f 2016 = -1 :=
by
  sorry

end proof_problem_l801_801035


namespace find_ab_integer_l801_801350

theorem find_ab_integer (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_neq : a ≠ b) :
    ∃ n : ℤ, (a^b + b^a) = n * (a^a - b^b) ↔ (a = 2 ∧ b = 1) ∨ (a = 1 ∧ b = 2) := 
sorry

end find_ab_integer_l801_801350


namespace domain_of_composite_function_l801_801039

theorem domain_of_composite_function (k : ℤ) :
  {x : ℝ | 0 < sin x ∧ sin x ≤ 1} = {x : ℝ | ∃ k : ℤ, (2 * k * π < x) ∧ (x < 2 * k * π + π)} :=
by
  sorry

end domain_of_composite_function_l801_801039


namespace correct_geometric_solids_definitions_l801_801271

theorem correct_geometric_solids_definitions :
  (∀ (s : geometric_solid), s.is_pyramid ↔ (s.has_one_polygon_face ∧ s.rest_triangles_share_vertex)) ∧
  (∀ (p : pyramid), p.is_frustum ↔ (p.section_between_base_parallel)) ∧
  (∀ (pr : prism), pr.lateral_faces_are_parallelograms) ∧
  (∀ (t : right_triangle), (rotate_about_one_side t).is_cone) :=
sorry

end correct_geometric_solids_definitions_l801_801271


namespace readers_in_group_l801_801457

theorem readers_in_group (S L B T : ℕ) (hS : S = 120) (hL : L = 90) (hB : B = 60) :
  T = S + L - B → T = 150 :=
by
  intro h₁
  rw [hS, hL, hB] at h₁
  linarith

end readers_in_group_l801_801457


namespace avg_age_combined_l801_801908

-- Define the conditions
def avg_age_roomA : ℕ := 45
def avg_age_roomB : ℕ := 20
def num_people_roomA : ℕ := 8
def num_people_roomB : ℕ := 3

-- Definition of the problem statement
theorem avg_age_combined :
  (num_people_roomA * avg_age_roomA + num_people_roomB * avg_age_roomB) / (num_people_roomA + num_people_roomB) = 38 :=
by
  sorry

end avg_age_combined_l801_801908


namespace determine_omega_l801_801588

-- Define the function f(x) and the constraints on ω
def f (ω x : ℝ) := 2 * Real.sin (ω * x)

-- Lean statement to prove the value of ω given the conditions
theorem determine_omega :
  ∃ (ω : ℝ), ω > 0 ∧
  (ω * π / 3 ≤ π / 2) ∧
  (2 * Real.sin (ω * π / 3) = Real.sqrt 2) ∧
  ω = 3 / 4 :=
sorry

end determine_omega_l801_801588


namespace circle_area_l801_801210

-- Let r be the radius of the circle
-- The circumference of the circle is given by 2 * π * r, which is 36 cm
-- We need to prove that given this condition, the area of the circle is 324/π square centimeters

theorem circle_area (r : Real) (h : 2 * Real.pi * r = 36) : Real.pi * r^2 = 324 / Real.pi :=
by
  sorry

end circle_area_l801_801210


namespace part_I_part_II_l801_801398

def t1 (x : ℕ) : ℕ → ℕ := λ r => binomial 5 r * 2^(5 - r) * x^(10 - 3*r)
def t2 (x : ℕ) (n : ℕ) : ℕ → ℕ := λ r => binomial n r * 4 * x^(n/2 - 3)

-- term containing 1/x^2 in expansion of (2*x^2 + 1/x)^5
theorem part_I (x : ℕ) : t1 x 4 = 10 / x^2 :=
  sorry

-- value of n when sum of binomial coefficients is 28 less than coefficient of third term in (sqrt(x) + 2/x)^n
theorem part_II (x : ℕ) (n : ℕ) : 2^5 = binomial n 2 * 4 - 28 → n = 6 :=
  sorry

end part_I_part_II_l801_801398


namespace evaluate_f_f_one_fourth_l801_801776

def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 2 else 3^x

theorem evaluate_f_f_one_fourth : f (f (1 / 4)) = 1 / 9 := by
  sorry

end evaluate_f_f_one_fourth_l801_801776


namespace polygon_area_28_l801_801922

-- Assuming basic properties of isosceles triangles
structure IsoscelesTriangle :=
  (area : ℝ)
  (vertex_angle : ℝ)
  (base_angle : ℝ)

-- Defining the identified conditions
def triangle_conditions : IsoscelesTriangle :=
  { area := 2, vertex_angle := 100, base_angle := 40 }

-- Defining the total polygon area given the number of triangles
def polygon_area (full_triangles half_triangles : ℕ) (triangle : IsoscelesTriangle) : ℝ :=
  (full_triangles * triangle.area) + (half_triangles / 2.0 * triangle.area)

-- The statement that needs to be proved
theorem polygon_area_28 :
  polygon_area 12 4 triangle_conditions = 28 := by
  sorry

end polygon_area_28_l801_801922


namespace r_plus_s_l801_801228

section Proof

variables {x y r s : ℚ}

/- Given lines -/
def line1 (x : ℚ) : ℚ := - (2 / 3) * x + 8
def line2 (x : ℚ) : ℚ := (3 / 2) * x - 9

/- Intercepts -/
def P : ℚ × ℚ := (12, 0)
def Q : ℚ × ℚ := (0, 8)

/- Intersection point T -/
def T : ℚ × ℚ := (r, s)

/- Area calculation -/
def triangle_area (a b c : ℚ × ℚ) : ℚ :=
  1 / 2 * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))

def area_POQ : ℚ := triangle_area P Q ⟨0, 0⟩
def area_TOP : ℚ := triangle_area T ⟨0, 0⟩ P

/- Given conditions -/
axiom T_intersection : line1 r = s ∧ line2 r = s
axiom area_cond : area_POQ = 2 * area_TOP

/- Prove the value of r + s -/
theorem r_plus_s : r + s = 138 / 13 :=
sorry

end Proof

end r_plus_s_l801_801228


namespace lateral_surface_area_of_rotated_square_l801_801895

noncomputable def lateralSurfaceAreaOfRotatedSquare (side_length : ℝ) : ℝ :=
  2 * Real.pi * side_length * side_length

theorem lateral_surface_area_of_rotated_square :
  lateralSurfaceAreaOfRotatedSquare 1 = 2 * Real.pi :=
by
  sorry

end lateral_surface_area_of_rotated_square_l801_801895


namespace problem_l801_801059

def f (a : ℝ) (x : ℝ) : ℝ := log a (3 - x) + log a (x + 1)

theorem problem (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (∀ x, f a x = f a (2 - x)) ∧
  (set.Ioo (-1 : ℝ) 3 = {x | -1 < x ∧ x < 3}) ∧
  (∀ x, f a x ≤ f a 1 → a = 2) :=
by
  sorry

end problem_l801_801059


namespace combined_experience_l801_801833

theorem combined_experience : 
  ∀ (James John Mike : ℕ), 
  (James = 20) → 
  (∀ (years_ago : ℕ), (years_ago = 8) → (John = 2 * (James - years_ago) + years_ago)) → 
  (∀ (started : ℕ), (John - started = 16) → (Mike = 16)) → 
  James + John + Mike = 68 :=
begin
  intros James John Mike HJames HJohn HMike,
  rw HJames,
  have HJohn8 : John = 32, {
    rw HJohn,
    intros years_ago Hyears_ago,
    rw Hyears_ago,
    norm_num,
  },
  rw HJohn8 at HMike,
  norm_num at HMike,
  rw HJohn8,
  rw HMike,
  norm_num,
end

end combined_experience_l801_801833


namespace min_value_x3_l801_801065

noncomputable def min_x3 (x1 x2 x3 : ℝ) : ℝ := -21 / 11

theorem min_value_x3 (x1 x2 x3 : ℝ) 
  (h1 : x1 + (1 / 2) * x2 + (1 / 3) * x3 = 1)
  (h2 : x1^2 + (1 / 2) * x2^2 + (1 / 3) * x3^2 = 3) 
  : x3 ≥ - (21 / 11) := 
by sorry

end min_value_x3_l801_801065


namespace tan_alpha_eq_neg3_g_alpha_eq_neg7div11_l801_801385

-- Proof that tan(α) = -3 given conditions
theorem tan_alpha_eq_neg3 (α : ℝ) (h1 : π / 2 < α ∧ α < π)
    (h2 : tan α - 1 / tan α = -8 / 3) : tan α = -3 :=
sorry

-- Proof for the value of g(α) given the conditions
noncomputable def g (α : ℝ) : ℝ := 
  (sin (π + α) + 4 * cos (2 * π + α)) / (sin ((π / 2) - α) - 4 * sin (-α))

theorem g_alpha_eq_neg7div11 (α : ℝ) (h1 : π / 2 < α ∧ α < π) (h3 : tan α = -3) : 
  g α = -7 / 11 :=
sorry

end tan_alpha_eq_neg3_g_alpha_eq_neg7div11_l801_801385


namespace largest_negative_integer_is_neg_one_l801_801636

def is_negative_integer (n : Int) : Prop := n < 0

def is_largest_negative_integer (n : Int) : Prop := 
  is_negative_integer n ∧ ∀ m : Int, is_negative_integer m → m ≤ n

theorem largest_negative_integer_is_neg_one : 
  is_largest_negative_integer (-1) := by
  sorry

end largest_negative_integer_is_neg_one_l801_801636


namespace count_divisors_2022_2022_l801_801794

noncomputable def num_divisors_2022_2022 : ℕ :=
  let fac2022 := 2022
  let factor_triplets := [(2, 3, 337), (3, 337, 2), (2, 337, 3), (337, 2, 3), (337, 3, 2), (3, 2, 337)]
  factor_triplets.length

theorem count_divisors_2022_2022 :
  num_divisors_2022_2022 = 6 :=
  by {
    sorry
  }

end count_divisors_2022_2022_l801_801794


namespace broken_glass_pieces_l801_801249

theorem broken_glass_pieces (x : ℕ) 
    (total_pieces : ℕ := 100) 
    (safe_fee : ℕ := 3) 
    (compensation : ℕ := 5) 
    (total_fee : ℕ := 260) 
    (h : safe_fee * (total_pieces - x) - compensation * x = total_fee) : x = 5 := by
  sorry

end broken_glass_pieces_l801_801249


namespace seashells_total_correct_l801_801179

def total_seashells (red_shells green_shells other_shells : ℕ) : ℕ :=
  red_shells + green_shells + other_shells

theorem seashells_total_correct :
  total_seashells 76 49 166 = 291 :=
by
  sorry

end seashells_total_correct_l801_801179


namespace solve_rational_equation_l801_801901

theorem solve_rational_equation (x : ℝ) (h : x ≠ (2/3)) : 
  (6*x + 4) / (3*x^2 + 6*x - 8) = 3*x / (3*x - 2) ↔ x = -4/3 ∨ x = 3 :=
sorry

end solve_rational_equation_l801_801901


namespace simplify_tangent_sum_l801_801539

theorem simplify_tangent_sum :
  tan (Real.pi / 12) + tan (5 * Real.pi / 12) = Real.sqrt 6 - Real.sqrt 2 := 
sorry

end simplify_tangent_sum_l801_801539


namespace hyperbola_real_axis_length_l801_801238

theorem hyperbola_real_axis_length : 
  (∃ (x y : ℝ), (x^2 / 2) - (y^2 / 4) = 1) → real_axis_length = 2 * Real.sqrt 2 :=
by
  -- Proof is omitted
  sorry

end hyperbola_real_axis_length_l801_801238


namespace area_hexagon_l801_801845

/-!
Let \(ABCD\) be a trapezoid with \(AB \parallel CD\), where \(AB = 9\), \(BC = 7\), \(CD = 23\), and \(DA = 5\). 
Bisectors of \( \angle A \) and \( \angle D \) meet at point \( P \), and bisectors of \( \angle B \) and \( \angle C \) meet at point \( Q \). 
We need to prove that the area of the hexagon \( ABQCDP \) is \( 13\sqrt{18.5} \).
-/

-- Defining the conditions
variables {A B C D P Q : Type}
variables {AB CD x : ℝ}
variables (BC DA : ℝ) (angleBisectorA angleBisectorD angleBisectorB angleBisectorC : ℝ)

-- Provided triangle and trapezoid properties
def is_trapezoid (A B C D : Type) (AB CD : ℝ) : Prop := AB = 9 ∧ CD = 23 ∧ AB ∥ CD
def side_lengths (BC DA : ℝ) : Prop := BC = 7 ∧ DA = 5
def bisector_meet_at_PQ (P Q : Type) : Prop :=
  ∃ x, bisector_distance A D P x ∧ bisector_distance B C Q x

-- Proving the area of hexagon ABQCDP
theorem area_hexagon (PQ : bisector_meet_at_PQ P Q) :
  let x := sqrt(18.5) / 2 in
  let area := 26 * x in
  area = 13 * sqrt(18.5) :=
sorry

end area_hexagon_l801_801845


namespace team_percentage_win_l801_801337

theorem team_percentage_win (P : ℕ) 
  (h1 : (85 : ℝ) = 0.85 * 100) 
  (h2 : 123 = int.floor (0.70 * 175)) 
  (h3 : 175 = 100 + 75)
  (h4 : P = int.floor ((38.0 / 75) * 100)) : 
  P = 51 := 
by
  sorry

end team_percentage_win_l801_801337


namespace equality_of_costs_l801_801298

variable (x : ℝ)
def C1 : ℝ := 50 + 0.35 * (x - 500)
def C2 : ℝ := 75 + 0.45 * (x - 1000)

theorem equality_of_costs : C1 x = C2 x → x = 2500 :=
by
  intro h
  sorry

end equality_of_costs_l801_801298


namespace abc_inequality_l801_801123

theorem abc_inequality 
  (a b c a₁ a₂ b₁ b₂ c₁ c₂ : ℝ)
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) 
  (h₄ : a₁ > 0) (h₅ : a₂ > 0) (h₆ : b₁ > 0) (h₇ : b₂ > 0)
  (h₈ : c₁ > 0) (h₉ : c₂ > 0) 
  (P₁ P₂ : PointInsideTriangle)
  (AP₁ : P₁.distance_to A = a₁)
  (AP₂ : P₂.distance_to A = a₂)
  (BP₁ : P₁.distance_to B = b₁)
  (BP₂ : P₂.distance_to B = b₂)
  (CP₁ : P₁.distance_to C = c₁)
  (CP₂ : P₂.distance_to C = c₂) : 
  a * a₁ * a₂ + b * b₁ * b₂ + c * c₁ * c₂ ≥ a * b * c := 
begin
  sorry
end

end abc_inequality_l801_801123


namespace dividend_approx_l801_801873

def quotient := 89
def remainder := 14
def divisor := 198.69662921348313
def dividend := (divisor * quotient) + remainder

theorem dividend_approx : abs (dividend - 17698) < 1 :=
by
  sorry

end dividend_approx_l801_801873


namespace returnable_value_l801_801829

namespace GiftCards

structure GiftCard :=
  (value : ℕ)  -- Value of the gift card in dollars
  (discount : Option ℕ := none) -- Optional discount as a percentage
  (expired_in_months : Option ℕ := none) -- Optional expiration in months
  (region_restriction : Option String := none) -- Optional region restriction

def BestBuyCards := [
  GiftCard.mk 500 none (some 3),
  GiftCard.mk 500 none (some 3),
  GiftCard.mk 500 none none,
  GiftCard.mk 500 none none,
  GiftCard.mk 500 none none
]

def TargetCards := [
  GiftCard.mk 250 (some 10) none,
  GiftCard.mk 250 none none,
  GiftCard.mk 250 none none
]

def WalmartCards := [
  GiftCard.mk 100 none (some 8),
  GiftCard.mk 100 none (some 14),
  GiftCard.mk 100 none none,
  GiftCard.mk 100 none none,
  GiftCard.mk 100 none none,
  GiftCard.mk 100 none none,
  GiftCard.mk 100 none none
]

def AmazonCards := [
  GiftCard.mk 1000 none none,
  GiftCard.mk 1000 none none
]

def remaining_value (cards: List GiftCard) (sent_codes: ℕ) : ℕ :=
  cards.drop sent_codes |>.sum (λ card, card.value) -- Sum the value of the remaining gift cards

theorem returnable_value :
  remaining_value BestBuyCards 1 +
  remaining_value TargetCards 0 +
  remaining_value WalmartCards 2 +
  remaining_value AmazonCards 1 = 4250 := by
  sorry

end GiftCards

end returnable_value_l801_801829


namespace john_finishes_fourth_task_at_12_40PM_l801_801138

noncomputable def john_finish_time
  (task_start : ℕ)
  (third_task_end : ℕ)
  (num_tasks : ℕ)
  (tasks_done : ℕ)
  (task_duration : ℕ := (third_task_end - task_start) / tasks_done) : ℕ :=
third_task_end + task_duration

theorem john_finishes_fourth_task_at_12_40PM :
  john_finish_time 8 11*60 + 30 4 3 = 12*60 + 40 :=
sorry

end john_finishes_fourth_task_at_12_40PM_l801_801138


namespace find_n_l801_801348

theorem find_n (n : ℕ) (h : 2^6 * 3^3 * n = nat.factorial 10) : n = 2100 :=
sorry

end find_n_l801_801348


namespace right_triangle_area_inscribed_circle_l801_801978

theorem right_triangle_area_inscribed_circle (r a b c : ℝ)
  (h_c : c = 6 + 7)
  (h_a : a = 6 + r)
  (h_b : b = 7 + r)
  (h_pyth : (6 + r)^2 + (7 + r)^2 = 13^2):
  (1 / 2) * (a * b) = 42 :=
by
  -- The necessary calculations have already been derived and verified
  sorry

end right_triangle_area_inscribed_circle_l801_801978


namespace ratio_sum_l801_801826

theorem ratio_sum (A B C D E : Type) (on_line_BC : D ∈ line B C) 
  (on_line_AB : E ∈ line A B) (BD_2DC : ∃ k: ℝ, k * BD = (2:ℝ) * DC) 
  (AE_2EB : ∃ l: ℝ, l * AE = (2:ℝ) * EB) :
  (AE / EB) + (BD / DC) = (7/3 : ℝ) :=
by
  sorry

end ratio_sum_l801_801826


namespace parabola_equation_standard_line_equation_l801_801758

theorem parabola_equation_standard (p : ℝ) :
  (let eqn := (λ x y : ℝ, y^2 = 2 * p * x) in (eqn 1 (-2)) ∧ p > 0) →
  (∀ x y : ℝ, y^2 = 4 * x) :=
by
  intro h
  let eqn := λ x y : ℝ, y^2 = 4 * x
  sorry

theorem line_equation (line_eq : ℝ → ℝ):
  (∀ x : ℝ, line_eq x = k * (x - 1)) → 
  (∃ k : ℝ, k ≠ 0 ∧ ( (line_eq = λ x, -x + 1) ∨ (line_eq = λ x, x - 1) )) :=
by
  intro h
  have area_condition : 2 * sqrt 2 = ∀ x1 y1 x2 y2 : ℝ, (1 / 2) * abs(x1 + x2 + 2 / k) * (abs(k) / sqrt(1 + k^2))
  sorry

end parabola_equation_standard_line_equation_l801_801758


namespace remaining_number_is_zero_l801_801273

theorem remaining_number_is_zero :
  (∑ i in finset.range 1988, i) % 7 = 0 →
  987 % 7 = 0 →
  ∃ x, (x + 987) % 7 = 0 ∧ x = 0 :=
by
  intros h_sum h_987
  use 0
  split
  -- Checking (x + 987) % 7 = 0 when x = 0
  exact h_987
  -- Confirm that x = 0
  refl

end remaining_number_is_zero_l801_801273


namespace edge_labeling_gcd_one_l801_801853

variables {V E : Type*} 
variables (G : SimpleGraph V) [Fintype V] [Fintype E]

theorem edge_labeling_gcd_one (hG : G.IsConnected) (k : ℕ) (hE : G.edgeFinset.card = k) :
  ∃ (f : E → ℕ), (∀ v ∈ G.V, G.degree v ≥ 2 → gcd (G.adjacentEdgeLabels f v) = 1) :=
sorry

end edge_labeling_gcd_one_l801_801853


namespace problem1_problem2_problem3_problem4_problem5_problem6_l801_801317

theorem problem1 : (-2.4) + (-3.7) + (-4.6) + 5.7 = -5 := by
  sorry

theorem problem2 : (-8) + (-6) - (+4) - (-10) = -8 := by
  sorry

theorem problem3 : (-1 / 2) - (- (13 / 4)) + (11 / 4) - (11 / 2) = 0 := by
  sorry

theorem problem4 : abs (-45) + (-71) + abs (-5 - (-9)) = -22 := by
  sorry

theorem problem5 : abs (-5 / 2) + (-3.7) + abs (-2.7) - abs (15 / 2) = -6 := by
  sorry

theorem problem6 : (-3) * (5 / 6) * (-4 / 5) * (-1 / 4) = -1 / 2 := by
  sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l801_801317


namespace sum_of_three_consecutive_odds_is_69_l801_801242

-- Definition for the smallest of three consecutive odd numbers
def smallest_consecutive_odd := 21

-- Define the three consecutive odd numbers based on the smallest one
def first_consecutive_odd := smallest_consecutive_odd
def second_consecutive_odd := smallest_consecutive_odd + 2
def third_consecutive_odd := smallest_consecutive_odd + 4

-- Calculate the sum of these three consecutive odd numbers
def sum_consecutive_odds := first_consecutive_odd + second_consecutive_odd + third_consecutive_odd

-- Theorem statement that the sum of these three consecutive odd numbers is 69
theorem sum_of_three_consecutive_odds_is_69 : 
  sum_consecutive_odds = 69 := by
    sorry

end sum_of_three_consecutive_odds_is_69_l801_801242


namespace parallelepiped_diagonal_inequality_l801_801149

theorem parallelepiped_diagonal_inequality 
  (a b c d : ℝ) 
  (h_d : d = Real.sqrt (a^2 + b^2 + c^2)) : 
  a^2 + b^2 + c^2 ≥ d^2 / 3 := 
by 
  sorry

end parallelepiped_diagonal_inequality_l801_801149


namespace smallest_nonfactor_product_of_48_l801_801946

noncomputable def is_factor_of (a b : ℕ) : Prop :=
  b % a = 0

theorem smallest_nonfactor_product_of_48
  (m n : ℕ)
  (h1 : m ≠ n)
  (h2 : is_factor_of m 48)
  (h3 : is_factor_of n 48)
  (h4 : ¬is_factor_of (m * n) 48) :
  m * n = 18 :=
sorry

end smallest_nonfactor_product_of_48_l801_801946


namespace number_of_students_l801_801139

-- Define John's total winnings
def john_total_winnings : ℤ := 155250

-- Define the proportion of winnings given to each student
def proportion_per_student : ℚ := 1 / 1000

-- Define the total amount received by students
def total_received_by_students : ℚ := 15525

-- Calculate the amount each student received
def amount_per_student : ℚ := john_total_winnings * proportion_per_student

-- Theorem to prove the number of students
theorem number_of_students : total_received_by_students / amount_per_student = 100 :=
by
  -- Lean will be expected to fill in this proof
  sorry

end number_of_students_l801_801139


namespace slope_of_line_AB_is_pm_4_3_l801_801036

noncomputable def slope_of_line_AB : ℝ := sorry

theorem slope_of_line_AB_is_pm_4_3 (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : y₁^2 = 4 * x₁)
  (h₂ : y₂^2 = 4 * x₂)
  (h₃ : (x₁, y₁) ≠ (x₂, y₂))
  (h₄ : (x₁ - 1, y₁) = -4 * (x₂ - 1, y₂)) :
  slope_of_line_AB = 4 / 3 ∨ slope_of_line_AB = -4 / 3 :=
sorry

end slope_of_line_AB_is_pm_4_3_l801_801036


namespace quartets_sung_songs_l801_801005

def friends := {Mary, Alina, Tina, Hanna, Zoe}

variables (z m a t h : ℕ)

noncomputable def songs_sung_by_quartet : ℕ :=
  (8 + 5 + a + t + h) / 4

theorem quartets_sung_songs (h₁ : z = 8)
  (h₂ : m = 5)
  (h₃ : 5 < a ∧ a < 8)
  (h₄ : 5 < t ∧ t < 8)
  (h₅ : 5 < h ∧ h < 8)
  (h₆ : friends.sum_by_quartet = 8)
  (h₇ : (8 + 5 + a + t + h) % 4 = 0) :
  songs_sung_by_quartet = 8 :=
sorry

end quartets_sung_songs_l801_801005


namespace prob_sum_is_nine_l801_801099

-- Definition of the probability problem
def throwing_two_dice_outcomes : Set (ℕ × ℕ) :=
  { (a, b) | 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 }

-- Definition of the specific event where the sum is 9
def sum_is_nine (outcome : ℕ × ℕ) : Prop :=
  outcome.1 + outcome.2 = 9

-- Definition of the probability of an event given the total number of outcomes
def probability (event : Set (ℕ × ℕ)) (total_outcomes : ℝ) : ℝ :=
  (event.to_finset.card : ℝ) / total_outcomes

-- Probability problem statement
theorem prob_sum_is_nine :
  probability {o ∈ throwing_two_dice_outcomes | sum_is_nine o} 36 = 1 / 9 :=
by
  sorry

end prob_sum_is_nine_l801_801099


namespace find_range_of_a_l801_801152

-- Definitions
def is_decreasing_function (a : ℝ) : Prop :=
  0 < a ∧ a < 1

def no_real_roots_of_poly (a : ℝ) : Prop :=
  4 * a < 1

def problem_statement (a : ℝ) : Prop :=
  (is_decreasing_function a ∨ no_real_roots_of_poly a) ∧ ¬ (is_decreasing_function a ∧ no_real_roots_of_poly a)

-- Main theorem
theorem find_range_of_a (a : ℝ) : problem_statement a ↔ (0 < a ∧ a ≤ 1 / 4) ∨ (a ≥ 1) :=
by
  -- Proof omitted
  sorry

end find_range_of_a_l801_801152


namespace quadrilateral_incircle_properties_l801_801023

variable {α : Type*} [EuclideanGeometry α]

/-- Given a convex quadrilateral ABCD with an incircle, and a point P outside ABCD such that
  ∠APB = ∠CPD, and the rays PB and PD lie within ∠APC.
  1. Prove that the incircles of triangles ABP, BCP, CDP, and DAP share a common tangent line.
  2. Prove that the centers of these incircles are concyclic. -/
theorem quadrilateral_incircle_properties 
  {A B C D P : α} 
  (h_convex : ConvexQuadrilateral A B C D) 
  (h_incirc : HasIncircle A B C D)
  (h_APB_eq_CPD : ∠A P B = ∠C P D)
  (h_rays : PB ⊆ ∠A P C ∧ PD ⊆ ∠A P C) : 
  ∃ l : Line α, (Incircle (Triangle.mk A B P)).Tangent l ∧
                (Incircle (Triangle.mk B C P)).Tangent l ∧
                (Incircle (Triangle.mk C D P)).Tangent l ∧
                (Incircle (Triangle.mk D A P)).Tangent l ∧
                Concyclic {I1, I2, I3, I4} :=
by sorry

end quadrilateral_incircle_properties_l801_801023


namespace page_cost_in_cents_l801_801830

theorem page_cost_in_cents (notebooks pages_per_notebook total_cost : ℕ)
  (h_notebooks : notebooks = 2)
  (h_pages_per_notebook : pages_per_notebook = 50)
  (h_total_cost : total_cost = 5 * 100) :
  (total_cost / (notebooks * pages_per_notebook)) = 5 :=
by
  sorry

end page_cost_in_cents_l801_801830


namespace part1_correct_part2_correct_part3_correct_l801_801810

-- Example survival rates data (provided conditions)
def survivalRatesA : List (Option Float) := [some 95.5, some 92, some 96.5, some 91.6, some 96.3, some 94.6, none, none, none, none]
def survivalRatesB : List (Option Float) := [some 95.1, some 91.6, some 93.2, some 97.8, some 95.6, some 92.3, some 96.6, none, none, none]
def survivalRatesC : List (Option Float) := [some 97, some 95.4, some 98.2, some 93.5, some 94.8, some 95.5, some 94.5, some 93.5, some 98, some 92.5]

-- Define high-quality project condition
def isHighQuality (rate : Float) : Bool := rate > 95.0

-- Problem 1: Probability of two high-quality years from farm B
noncomputable def probabilityTwoHighQualityB : Float := (4.0 * 3.0) / (7.0 * 6.0)

-- Problem 2: Distribution of high-quality projects from farms A, B, and C
structure DistributionX := 
(P0 : Float) -- probability of 0 high-quality years
(P1 : Float) -- probability of 1 high-quality year
(P2 : Float) -- probability of 2 high-quality years
(P3 : Float) -- probability of 3 high-quality years

noncomputable def distributionX : DistributionX := 
{ P0 := 3.0 / 28.0,
  P1 := 5.0 / 14.0,
  P2 := 11.0 / 28.0,
  P3 := 1.0 / 7.0 
}

-- Problem 3: Inference of average survival rate from high-quality project probabilities
structure AverageSurvivalRates := 
(avgB : Float) 
(avgC : Float)
(probHighQualityB : Float)
(probHighQualityC : Float)
(canInfer : Bool)

noncomputable def avgSurvivalRates : AverageSurvivalRates := 
{ avgB := (95.1 + 91.6 + 93.2 + 97.8 + 95.6 + 92.3 + 96.6) / 7.0,
  avgC := (97 + 95.4 + 98.2 + 93.5 + 94.8 + 95.5 + 94.5 + 93.5 + 98 + 92.5) / 10.0,
  probHighQualityB := 4.0 / 7.0,
  probHighQualityC := 5.0 / 10.0,
  canInfer := false
}

-- Definitions for proof statements indicating correctness
theorem part1_correct : probabilityTwoHighQualityB = (2.0 / 7.0) := sorry

theorem part2_correct : distributionX = 
{ P0 := 3.0 / 28.0,
  P1 := 5.0 / 14.0,
  P2 := 11.0 / 28.0,
  P3 := 1.0 / 7.0 
} := sorry

theorem part3_correct : avgSurvivalRates.canInfer = false := sorry

end part1_correct_part2_correct_part3_correct_l801_801810


namespace projection_onto_line_l801_801362

theorem projection_onto_line : 
  let v := (⟨5, -3, 2⟩ : ℝ × ℝ × ℝ),
      d := (⟨1, -1/2, 1/4⟩ : ℝ × ℝ × ℝ),
      dot_v_d := (v.1 * d.1 + v.2 * d.2 + v.3 * d.3),
      dot_d_d := (d.1 * d.1 + d.2 * d.2 + d.3 * d.3),
      scalar := dot_v_d / dot_d_d
  in (scalar * d.1, scalar * d.2, scalar * d.3) = (⟨16/3, -8/3, 4/3⟩ : ℝ × ℝ × ℝ) := 
by
  sorry

end projection_onto_line_l801_801362


namespace find_divisor_l801_801965

-- Definition of given conditions
def Dividend : ℝ := 63584
def Quotient : ℝ := 127.8
def Remainder : ℝ := 45.5

-- Goal: Prove the divisor using the given conditions
theorem find_divisor (Divisor : ℝ) : Dividend = (Divisor * Quotient) + Remainder → Divisor = 497.1 :=
 by {
   sorry, -- Proof skipped
 }

end find_divisor_l801_801965


namespace tan_sum_pi_over_12_l801_801536

theorem tan_sum_pi_over_12 : 
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12)) = 4 := 
sorry

end tan_sum_pi_over_12_l801_801536


namespace hours_per_shift_l801_801477

def hourlyWage : ℝ := 4.0
def tipRate : ℝ := 0.15
def shiftsWorked : ℕ := 3
def averageOrdersPerHour : ℝ := 40.0
def totalEarnings : ℝ := 240.0

theorem hours_per_shift :
  (hourlyWage + averageOrdersPerHour * tipRate) * (8 * shiftsWorked) = totalEarnings := 
sorry

end hours_per_shift_l801_801477


namespace sum_of_positive_integer_solutions_l801_801002

theorem sum_of_positive_integer_solutions :
  (∑ x in Finset.filter (λ x, ∃ t: ℕ, x ≠ 0 ∧ (x: ℤ) = t ∧ ∃ (d: ℤ), (x^2:ℕ) - 1 = d ∧ 120 % d = 0) (Finset.range 12).erase 0, x) = 25 :=
sorry

end sum_of_positive_integer_solutions_l801_801002


namespace min_even_integers_l801_801609

def three_sum (x y z : Int) := x + y + z = 36
def six_sum (x y z a b c : Int) := x + y + z + a + b + c = 60
def eight_sum (x y z a b c m n : Int) := x + y + z + a + b + c + m + n = 76
def mn_product (m n : Int) := m * n = 48
def mn_sum (m n : Int) := m + n = 16

theorem min_even_integers : ∃ x y z a b c m n : Int,
  three_sum x y z ∧
  six_sum x y z a b c ∧
  eight_sum x y z a b c m n ∧
  mn_product m n ∧
  mn_sum m n ∧
  (∀ xs : List Int, List.length (List.filter (λ x, Int.even x) xs) = 1 ↔
    xs = [x, y, z, a, b, c, m, n]) :=
sorry

end min_even_integers_l801_801609


namespace circle_area_from_circumference_l801_801208

theorem circle_area_from_circumference
  (c : ℝ)    -- the circumference
  (hc : c = 36)    -- condition: circumference is 36 cm
  : 
  ∃ A : ℝ,   -- there exists an area A
    A = 324 / π :=   -- conclusion: area is 324/π
by
  sorry   -- proof goes here

end circle_area_from_circumference_l801_801208


namespace domain_of_f_sqrt_log_is_interval_e_l801_801583

noncomputable def domain_of_sqrt_log (f : ℝ → ℝ) : set ℝ :=
  {x | x > 0 ∧ 1 - real.log x ≥ 0}

theorem domain_of_f_sqrt_log_is_interval_e :
  domain_of_sqrt_log (λ x, real.sqrt (1 - real.log x)) = set.Ioc 0 real.exp 1 := 
begin
  sorry
end

end domain_of_f_sqrt_log_is_interval_e_l801_801583


namespace lines_concurrent_l801_801154

variables {A B C H_B H_C E F P Q : Type*}
variables [triangle A B C] [altitude_foot B H_B] [altitude_foot C H_C]
variables [incircle_contact AC E] [incircle_contact AB F]
variables [angle_bisector_foot B P] [angle_bisector_foot C Q]

theorem lines_concurrent : concurrent (PQ) (EF) (H_B H_C) :=
sorry

end lines_concurrent_l801_801154


namespace probability_light_change_l801_801305

noncomputable def total_cycle_duration : ℕ := 45 + 5 + 50
def change_intervals : ℕ := 15

theorem probability_light_change :
  (15 : ℚ) / total_cycle_duration = 3 / 20 :=
by
  sorry

end probability_light_change_l801_801305


namespace eve_total_spend_l801_801722

def hand_mitts_cost : ℝ := 14.00
def apron_cost : ℝ := 16.00
def utensils_cost : ℝ := 10.00
def knife_cost : ℝ := 2 * utensils_cost
def discount_percent : ℝ := 0.25
def nieces_count : ℕ := 3

def total_cost_before_discount : ℝ :=
  (hand_mitts_cost + apron_cost + utensils_cost + knife_cost) * nieces_count

def discount_amount : ℝ :=
  discount_percent * total_cost_before_discount

def total_cost_after_discount : ℝ :=
  total_cost_before_discount - discount_amount

theorem eve_total_spend : total_cost_after_discount = 135.00 := by
  sorry

end eve_total_spend_l801_801722


namespace smallest_n_exists_l801_801736

noncomputable def gcd (a b : ℕ) : ℕ :=
  Nat.gcd a b

noncomputable def lcm (a b : ℕ) : ℕ :=
  Nat.lcm a b

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem smallest_n_exists :
  ∃ (n : ℕ) (a : Fin n → ℕ),
  (∀ i j, i < j → gcd (a i) (a j) > 1) ∧
  (lcm (Finset.image a Finset.univ).val = 2002) ∧
  (is_perfect_square (Finset.prod Finset.univ a)) ∧
  (32 ∣ Finset.prod Finset.univ a) ∧ n = 7 :=
sorry

end smallest_n_exists_l801_801736


namespace quadrilateral_parallelogram_l801_801107

-- Definitions of vertices and perpendicular vectors
variables {A B C D : Type*}
variables [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]
variables {n1 n2 n3 n4 : A} -- Assume these are unit vectors perpendicular to the edges

-- Definition stating sum of distances condition for a convex quadrilateral
def sum_of_distances (X : A) : ℝ :=
  distance X l1 + distance X l2 + distance X l3 + distance X l4

theorem quadrilateral_parallelogram
  (convex : (convex_quad A B C D))
  (equal_sum : ∀ X ∈ {A, B, C, D}, sum_of_distances X = k) :
  is_parallelogram A B C D :=
by
  sorry

end quadrilateral_parallelogram_l801_801107


namespace find_p_q_l801_801277

open_locale big_operators

-- Definitions of triangle sides and points
variables (P Q R X Y Z O_4 O_5 O_6 : Type)
variables [inner_product_space ℝ P] [inner_product_space ℝ Q] [inner_product_space ℝ R]
variables (PQ QR PR : ℝ)
variables (arc_PY arc_XQ arc_QZ arc_YR arc_PZ arc_YQ : ℝ)
variables (p q : ℕ)

-- Given conditions
axiom cond1 : PQ = 21
axiom cond2 : QR = 29
axiom cond3 : PR = 28
axiom cond4 : arc_PY = arc_XQ
axiom cond5 : arc_QZ = arc_YR
axiom cond6 : arc_PZ = arc_YQ

-- Main statement: Find p and q such that YQ can be written as p/q and prove p+q = 16.
theorem find_p_q : ∃ (p q : ℕ), (p/q : ℝ) = 15 ∧ nat.gcd p q = 1 ∧ p + q = 16 :=
by { 
  use [15, 1],
  split,
  { norm_num },
  split,
  { apply nat.gcd_div_gcd, dec_trivial, norm_num },
  { norm_num },
  sorry
}

end find_p_q_l801_801277


namespace circle_area_l801_801211

-- Let r be the radius of the circle
-- The circumference of the circle is given by 2 * π * r, which is 36 cm
-- We need to prove that given this condition, the area of the circle is 324/π square centimeters

theorem circle_area (r : Real) (h : 2 * Real.pi * r = 36) : Real.pi * r^2 = 324 / Real.pi :=
by
  sorry

end circle_area_l801_801211


namespace effective_annual_rate_l801_801963

theorem effective_annual_rate
  (i : ℝ) (n : ℕ) (t : ℝ) (h_i : i = 0.06) (h_n : n = 2) (h_t : t = 1) :
  (1 + i / n)^(n * t) - 1 = 0.0609 :=
by
  rw [h_i, h_n, h_t]
  norm_num
  sorry

end effective_annual_rate_l801_801963


namespace pasture_cows_variance_l801_801674

noncomputable def Dx {n : ℕ} {p : ℝ} (h₀ : n = 10) (h₁ : p = 0.02) : ℝ :=
n * p * (1 - p)

theorem pasture_cows_variance 
  (n : ℕ) (p : ℝ)
  (h₀ : n = 10) 
  (h₁ : p = 0.02) : 
  Dx h₀ h₁ = 0.196 :=
by 
  have h₂ : Dx h₀ h₁ = 10 * 0.02 * (1 - 0.02), 
  { rw [h₀, h₁] },
  rw h₂,
  norm_num,
  sorry

end pasture_cows_variance_l801_801674


namespace q_investment_time_l801_801644

variable {x t : ℕ}
variable (invest_p invest_q profit_p profit_q : ℕ)
variable (invest_p_ratio invest_q_ratio profit_p_ratio profit_q_ratio time_p : ℕ)

def investment_ratio (p_ratio q_ratio : ℕ) : Prop :=
  invest_p = p_ratio * x ∧ invest_q = q_ratio * x

def profit_ratio (p_ratio q_ratio : ℕ) : Prop := 
  profit_p = p_ratio * (invest_p * time_p) ∧ profit_q = q_ratio * (invest_q * t)

theorem q_investment_time (h1 : investment_ratio 7 5)
  (h2 : profit_ratio 7 11)
  (h3 : time_p = 5) : t = 55 :=
by
  sorry

end q_investment_time_l801_801644


namespace min_pyramid_sum_exists_l801_801297

noncomputable def pyramid_sum (B1 B2 B3 B4 B5 C : ℕ) : Prop :=
  (B5 + B1 + B2 = 13) ∧ (B3 + B4 + C = 13) ∧ (B1 + B3 = B2 + B4 ∨ B3 + B1 = B4 + B2) -- Added flexibility for pyramid property

theorem min_pyramid_sum_exists : ∃ B1 B2 B3 B4 B5 C, (B1 ∈ {1..10} ∧ B2 ∈ {1..10} ∧ B3 ∈ {1..10} ∧ B4 ∈ {1..10} ∧ B5 ∈ {1..10} ∧ C ∈ {1..10}) ∧
  pyramid_sum B1 B2 B3 B4 B5 C ∧
  (B1 ≠ B2 ∧ B1 ≠ B3 ∧ B1 ≠ B4 ∧ B1 ≠ B5 ∧ B1 ≠ C ∧ B2 ≠ B3 ∧ B2 ≠ B4 ∧ B2 ≠ B5 ∧ B2 ≠ C ∧
   B3 ≠ B4 ∧ B3 ≠ B5 ∧ B3 ≠ C ∧ B4 ≠ B5 ∧ B4 ≠ C ∧ B5 ≠ C) := sorry

end min_pyramid_sum_exists_l801_801297


namespace nick_paints_wall_in_fraction_l801_801793

theorem nick_paints_wall_in_fraction (nick_paint_time wall_paint_time : ℕ) (h1 : wall_paint_time = 60) (h2 : nick_paint_time = 12) : (nick_paint_time * 1 / wall_paint_time = 1 / 5) :=
by
  sorry

end nick_paints_wall_in_fraction_l801_801793


namespace Victoria_money_left_l801_801951

noncomputable def Victoria_initial_money : ℝ := 10000
noncomputable def jacket_price : ℝ := 250
noncomputable def trousers_price : ℝ := 180
noncomputable def purse_price : ℝ := 450
noncomputable def jackets_bought : ℕ := 8
noncomputable def trousers_bought : ℕ := 15
noncomputable def purses_bought : ℕ := 4
noncomputable def discount_rate : ℝ := 0.15
noncomputable def dinner_bill_inclusive : ℝ := 552.50
noncomputable def dinner_service_charge_rate : ℝ := 0.15

theorem Victoria_money_left : 
  Victoria_initial_money - 
  ((jackets_bought * jacket_price + trousers_bought * trousers_price) * (1 - discount_rate) + 
   purses_bought * purse_price + 
   dinner_bill_inclusive / (1 + dinner_service_charge_rate)) = 3725 := 
by 
  sorry

end Victoria_money_left_l801_801951


namespace ball_distribution_l801_801422

theorem ball_distribution (n k : ℕ) (h₁ : n = 5) (h₂ : k = 4) :
  k^n = 1024 := by
  rw [h₁, h₂]
  norm_num
  sorry

end ball_distribution_l801_801422


namespace necessary_but_not_sufficient_l801_801763

theorem necessary_but_not_sufficient (x : Real)
  (p : Prop := x < 1) 
  (q : Prop := x^2 + x - 2 < 0) 
  : p -> (q <-> x > -2 ∧ x < 1) ∧ (q -> p) → ¬ (p -> q) ∧ (x > -2 -> p) :=
by
  sorry

end necessary_but_not_sufficient_l801_801763


namespace midpoint_locus_is_nine_point_circle_l801_801650

noncomputable def midpoint_locus (A B C : Point) (P : Point) (circumcircle : Circle) (M : Point) :=
  ∀ (P : Point), P ∈ circumcircle → 
    midpoint (M, P) ∈ nine_point_circle (triangle.mk A B C)

theorem midpoint_locus_is_nine_point_circle (A B C M P : Point) (circumcircle : Circle) :
  let triangle_ABC := triangle.mk A B C in
  let orthocenter := orthocenter triangle_ABC in
  let nine_point_circle := nine_point_circle triangle_ABC in
  P ∈ circumcircle →
  midpoint (orthocenter, P) ∈ nine_point_circle :=
begin
  sorry
end

end midpoint_locus_is_nine_point_circle_l801_801650


namespace value_of_expression_l801_801956

theorem value_of_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -1) : -a^2 - b^2 + a * b = -21 := by
  sorry

end value_of_expression_l801_801956


namespace MN_value_l801_801884

-- Define the conditions:
def A := (0 : ℝ, 0 : ℝ)
def B := (841 : ℝ, 0 : ℝ)
def C := (0 : ℝ, 41 : ℝ)
def D := (0 : ℝ, 609 : ℝ)

-- Define centroids M and N:
def M := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
def N := ((A.1 + B.1 + D.1) / 3, (A.2 + B.2 + D.2) / 3)

-- Define the distance MN:
def distance (p q : ℝ × ℝ) := (Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2))
def MN := distance M N

-- Define the fraction 568 / 3:
def fraction := (568 / 3 : ℝ)

-- Prove that MN equals 568 / 3 and the sum of the numerator and denominator is 571:
theorem MN_value :
  MN = fraction → 568 + 3 = 571 :=
begin
  sorry,
end

end MN_value_l801_801884


namespace arrasta_um_proof_l801_801970

variable (n : ℕ)

def arrasta_um_possible_moves (n : ℕ) : ℕ :=
  6 * n - 8

theorem arrasta_um_proof (n : ℕ) (h : n ≥ 2) : arrasta_um_possible_moves n =
6 * n - 8 := by
  sorry

end arrasta_um_proof_l801_801970


namespace intersection_of_A_and_B_l801_801784

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def B : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }
def expected_intersection : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem intersection_of_A_and_B : A ∩ B = expected_intersection :=
by
  sorry

end intersection_of_A_and_B_l801_801784


namespace value_of_S5_l801_801503

noncomputable def S (x : ℝ) (m : ℕ) := x^m + x^(-m)

theorem value_of_S5 (x : ℝ) (h : x + x⁻¹ = 4) : S x 5 = 724 :=
by
  sorry

end value_of_S5_l801_801503


namespace donut_fundraiser_goal_not_met_l801_801476

theorem donut_fundraiser_goal_not_met : 
  ∀ (P_plain P_glazed P_chocolate S_plain S_glazed S_chocolate : ℝ)
    (n_d : ℕ),
    P_plain = 2.40 → P_glazed = 3.60 → P_chocolate = 4.80 → 
    S_plain = 1 → S_glazed = 1.50 → S_chocolate = 2 → 
    n_d = 6 → 
    (let profit_plain := (12 - P_plain),
         profit_glazed := (18 - P_glazed),
         profit_chocolate := (24 - P_chocolate),
         x := n_d / 3 in 
     (profit_plain * x + profit_glazed * x + profit_chocolate * x) < 96) := 
by {
  intro P_plain P_glazed P_chocolate S_plain S_glazed S_chocolate n_d,
  intros h1 h2 h3 h4 h5 h6 h7,
  let profit_plain := 12 - P_plain,
  let profit_glazed := 18 - P_glazed,
  let profit_chocolate := 24 - P_chocolate,
  have x_eq : n_d / 3 = 2, from calc
    n_d / 3 = 6 / 3 : by rw [h7]
    ... = 2 : by norm_num,
  have total_profit : profit_plain * 2 + profit_glazed * 2 + profit_chocolate * 2 = 86.40, from calc
    profit_plain * 2 + profit_glazed * 2 + profit_chocolate * 2 
    = (12 - P_plain) * 2 + (18 - P_glazed) * 2 + (24 - P_chocolate) * 2 : by congr,
    ... = 19.20 + 28.80 + 38.40 : by rw [h1, h2, h3]; norm_num,
    ... = 86.40 : by norm_num,
  have goal_not_met : 86.40 < 96, from by norm_num,
  exact goal_not_met,
  sorry
 }

end donut_fundraiser_goal_not_met_l801_801476


namespace rabbits_clear_land_in_21_days_l801_801258

theorem rabbits_clear_land_in_21_days (length_feet width_feet : ℝ) (rabbits : ℕ) (clear_per_rabbit_per_day : ℝ) : 
  length_feet = 900 → width_feet = 200 → rabbits = 100 → clear_per_rabbit_per_day = 10 →
  (⌈ (length_feet / 3 * width_feet / 3) / (rabbits * clear_per_rabbit_per_day) ⌉ = 21) := 
by
  intros
  sorry

end rabbits_clear_land_in_21_days_l801_801258


namespace thompson_children_divisibility_l801_801865

theorem thompson_children_divisibility : 
    ∀ n, 
    (n = 4440 ∨ n = 4443 ∨ n = 4446) ∧ 
    (∀ k ∈ {2, 3, 4, 6, 7, 8}, n % k = 0) → 
    n % 5 ≠ 0 :=
by sorry

end thompson_children_divisibility_l801_801865


namespace total_salaries_l801_801240

theorem total_salaries (A_salary B_salary : ℝ)
  (hA : A_salary = 1500)
  (hsavings : 0.05 * A_salary = 0.15 * B_salary) :
  A_salary + B_salary = 2000 :=
by {
  sorry
}

end total_salaries_l801_801240


namespace gain_percentage_is_8_l801_801679

variable (C S : ℝ) (D : ℝ)
variable (h1 : 20 * C * (1 - D / 100) = 12 * S)
variable (h2 : D ≥ 5 ∧ D ≤ 25)

theorem gain_percentage_is_8 :
  (12 * S * 1.08 - 20 * C * (1 - D / 100)) / (20 * C * (1 - D / 100)) * 100 = 8 :=
by
  sorry

end gain_percentage_is_8_l801_801679


namespace selection_methods_l801_801115

theorem selection_methods (students lectures : ℕ) (h_stu : students = 4) (h_lect : lectures = 3) : 
  (lectures ^ students) = 81 := 
by
  rw [h_stu, h_lect]
  rfl

end selection_methods_l801_801115


namespace coconuts_per_tree_correct_l801_801894

-- Definitions of the conditions
def farm_area : ℕ := 20
def trees_per_square_meter : ℕ := 2
def harvest_period_in_months : ℕ := 3
def price_per_coconut : ℝ := 0.50
def earnings_in_6_months : ℝ := 240

-- Calculated total number of trees
def total_trees : ℕ := farm_area * trees_per_square_meter

-- Earnings from one harvest
def earnings_per_harvest : ℝ := earnings_in_6_months / 2

-- Number of coconuts sold per harvest
def coconuts_per_harvest : ℕ := (earnings_per_harvest / price_per_coconut).toNat

-- Target value: number of coconuts per tree per harvest
def coconuts_per_tree_per_harvest : ℕ := coconuts_per_harvest / total_trees

theorem coconuts_per_tree_correct :
  coconuts_per_tree_per_harvest = 6 := by
  sorry

end coconuts_per_tree_correct_l801_801894


namespace second_percentage_increase_l801_801604

theorem second_percentage_increase :
  ∀ (P : ℝ) (x : ℝ), (P * 1.30 * (1 + x) = P * 1.5600000000000001) → x = 0.2 :=
by
  intros P x h
  sorry

end second_percentage_increase_l801_801604


namespace group_D_average_age_is_19_l801_801813

-- Definitions 
def total_students := 120
def overall_average_age := 16
def group_A_students := 0.20 * total_students
def group_B_students := 0.30 * total_students
def group_C_students := 0.40 * total_students
def group_D_students := total_students - (group_A_students + group_B_students + group_C_students)
def group_A_average_age := 14
def group_B_average_age := 15
def group_C_average_age := 17
def group_D_age_range := (18, 20)

-- Calculate total ages
def total_age_A := group_A_students * group_A_average_age
def total_age_B := group_B_students * group_B_average_age
def total_age_C := group_C_students * group_C_average_age
def overall_total_age := total_students * overall_average_age
def total_age_D := overall_total_age - (total_age_A + total_age_B + total_age_C)
def group_D_average_age := total_age_D / group_D_students

-- Proof problem
theorem group_D_average_age_is_19 :
  group_D_average_age = 19 := 
by
  sorry

end group_D_average_age_is_19_l801_801813


namespace concert_attendance_difference_l801_801869

theorem concert_attendance_difference :
  let first_concert := 65899
  let second_concert := 66018
  second_concert - first_concert = 119 :=
by
  sorry

end concert_attendance_difference_l801_801869


namespace max_min_values_l801_801054

-- Problem definitions
def f (x : ℝ) : ℝ := sqrt 3 * sin x * cos x - cos x ^ 2 - 1 / 2

-- Conditions
axiom domain_x : x ∈ set.Icc (-π / 12) (5 * π / 12)
axiom A_cond : f (A / 2) = -1
axiom a_eq : a = 2 * sqrt 3
axiom b_eq : b = 6

-- Proof problem
theorem max_min_values :
  (∀ x : ℝ, (x ∈ set.Icc (-π / 12) (5 * π / 12) → f x ≤ f (π / 3))) ∧
  (∀ x : ℝ, (x ∈ set.Icc (-π / 12) (5 * π / 12) → f (-π / 12) ≤ f x)) ∧
  (∀ c : ℝ, ∃ A : ℝ, (f (A / 2) = -1) ∧ (a = 2 * sqrt 3) ∧ (b = 6) → 
    (c = 2 * sqrt 3 ∨ c = 4 * sqrt 3)) := 
sorry

end max_min_values_l801_801054


namespace circle_area_l801_801209

-- Let r be the radius of the circle
-- The circumference of the circle is given by 2 * π * r, which is 36 cm
-- We need to prove that given this condition, the area of the circle is 324/π square centimeters

theorem circle_area (r : Real) (h : 2 * Real.pi * r = 36) : Real.pi * r^2 = 324 / Real.pi :=
by
  sorry

end circle_area_l801_801209


namespace volume_ratio_l801_801942

noncomputable theory

variables {A B B' C C' D D' : ℝ³} -- A, B, B', C, C', D, D' are points in ℝ³

-- Volumes of tetrahedrons
def volume_tetra (A B C D : ℝ³) : ℝ :=
  (1 / 6) * abs ((B - A) ⬝ ((C - A) × (D - A)))

theorem volume_ratio
  (hA : collinear {A, B, B'})
  (hB : collinear {A, C, C'})
  (hC : collinear {A, D, D'}) :
  volume_tetra A B C D / volume_tetra A B' C' D' =
    (dist A B * dist A C * dist A D) / (dist A B' * dist A C' * dist A D') :=
sorry

end volume_ratio_l801_801942


namespace quadratic_sum_roots_twice_difference_l801_801734

theorem quadratic_sum_roots_twice_difference
  (a b c x₁ x₂ : ℝ)
  (h_eq : a * x₁^2 + b * x₁ + c = 0)
  (h_eq2 : a * x₂^2 + b * x₂ + c = 0)
  (h_sum_twice_diff: x₁ + x₂ = 2 * (x₁ - x₂)) :
  3 * b^2 = 16 * a * c :=
sorry

end quadratic_sum_roots_twice_difference_l801_801734


namespace initial_bags_l801_801954

variable (b : ℕ)

theorem initial_bags (h : 5 * (b - 2) = 45) : b = 11 := 
by 
  sorry

end initial_bags_l801_801954


namespace ellipses_same_eccentricity_l801_801397

theorem ellipses_same_eccentricity 
  (a b : ℝ) (k : ℝ)
  (h1 : a > 0) 
  (h2 : b > 0)
  (h3 : k > 0)
  (e1_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ (x^2 / (a^2)) + (y^2 / (b^2)) = 1)
  (e2_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = k ↔ (x^2 / (ka^2)) + (y^2 / (kb^2)) = 1) :
  1 - (b^2 / a^2) = 1 - (b^2 / (ka^2)) :=
by
  sorry

end ellipses_same_eccentricity_l801_801397


namespace cab_ride_total_cost_for_all_l801_801316

-- Defining the conditions given in the problem

def off_peak_cost_per_mile : ℝ := 2.5
def peak_cost_per_mile : ℝ := 3.5
def distance_to_event : ℝ := 200
def participants : ℕ := 4
def discount : ℝ := 0.20
def days : ℕ := 7

-- Calculating the total cost for all participants

def single_trip_cost_off_peak : ℝ := off_peak_cost_per_mile * distance_to_event
def single_trip_cost_peak : ℝ := peak_cost_per_mile * distance_to_event
def single_day_cost : ℝ := single_trip_cost_off_peak + single_trip_cost_peak
def weekly_cost_without_discount : ℝ := single_day_cost * days
def total_discount : ℝ := weekly_cost_without_discount * discount
def total_cost_with_discount : ℝ := weekly_cost_without_discount - total_discount
def cost_per_participant : ℝ := total_cost_with_discount / participants

-- The theorem to prove

theorem cab_ride_total_cost_for_all :
  total_cost_with_discount = 6720 := by
  -- Omitted proof
  sorry

end cab_ride_total_cost_for_all_l801_801316


namespace similar_triangles_DE_length_l801_801651

theorem similar_triangles_DE_length
  (A B C D E : Type)
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  [MetricSpace D]
  [MetricSpace E]
  (BC BE : ℝ)
  (BC_eq : BC = 24)
  (BE_eq : BE = 18)
  (angle_B : ∠(B, C, A) = 60 ∧ ∠(B, E, D) = 60)
  (sim : similar_triangles A B C D B E) :
  ∃ DE : ℝ, DE = 18 :=
begin
  sorry
end

end similar_triangles_DE_length_l801_801651


namespace distance_equality_l801_801274

noncomputable def point (n : Type) := n

variables {A B C D M P : point ℝ}

def intersection_diagonals (A B C D M : point ℝ) : Prop :=
  -- Define condition stating M is the intersection of diagonals
  exists x y : ℝ, -- x and y are parameters in some coordinate system, for example
  -- Given a coordinate system where A, B, C, D are defined,
  -- M is the intersection of the line AC and line BD

def angle_condition (A P M D : point ℝ) : Prop :=
  -- Define the condition such that angle APM equals angle DPM
  ∠ APM = ∠ DPM

def distance_from_point_to_line (X : point ℝ) (line : set (point ℝ)) : ℝ := 
  -- Placeholder function to denote the distance from point X to a given line
  sorry

axiom d (C AP : point ℝ) : ℝ -- Distance from point C to line AP
axiom d_eq : ∀ (C AP B DP : point ℝ), distance_from_point_to_line C (set.range AP) = distance_from_point_to_line B (set.range DP)

theorem distance_equality (A B C D M P : point ℝ) 
  (h1 : intersection_diagonals A B C D M) 
  (h2 : angle_condition A P M D) : 
  d C (set.range (λ x : point ℝ, d_eq C (set.range (λ x, sorry))) = d B (set.range (λ x : point ℝ, d_eq B (set.range (λ x, sorry))) :=
sorry

end distance_equality_l801_801274


namespace find_m_max_min_FA_FB_l801_801062

open Real

-- Definition of the parametric line
def parametric_line (α t m : ℝ) : ℝ × ℝ :=
  (t * cos α + m, t * sin α)

-- Definition of the ellipse
def ellipse (φ : ℝ) : ℝ × ℝ :=
  (5 * cos φ, 3 * sin φ)

-- Focus of the ellipse
def right_focus : ℝ × ℝ :=
  (4, 0)

-- Prove that m = 4
theorem find_m (α t : ℝ) (cond_line : parametric_line α t m = right_focus) : m = 4 :=
  sorry

-- Prove the max and min values of |FA| * |FB|
theorem max_min_FA_FB (α : ℝ) :
  let expr := 81 / (9 + 16 * sin α ^ 2) in
  (∀ α, sin α = 0 → expr = 9) ∧ (∀ α, sin α = 1 ∨ sin α = -1 → expr = 81 / 25) :=
  sorry

end find_m_max_min_FA_FB_l801_801062


namespace james_end_of_year_balance_l801_801128

theorem james_end_of_year_balance :
  let weekly_investment := 2000
  let starting_balance := 250000
  let weeks_in_year := 52
  let total_investment := weekly_investment * weeks_in_year
  let end_of_year_balance := starting_balance + total_investment
  let windfall := 0.50 * end_of_year_balance
  let final_balance := end_of_year_balance + windfall
  final_balance = 531000 :=
by
  let weekly_investment := 2000
  let starting_balance := 250000
  let weeks_in_year := 52
  let total_investment := weekly_investment * weeks_in_year
  let end_of_year_balance := starting_balance + total_investment
  let windfall := 0.50 * end_of_year_balance
  let final_balance := end_of_year_balance + windfall
  sorry

end james_end_of_year_balance_l801_801128


namespace range_of_quadratic_expression_l801_801335

theorem range_of_quadratic_expression :
  (∃ x : ℝ, y = 2 * x^2 - 4 * x + 12) ↔ (y ≥ 10) :=
by
  sorry

end range_of_quadratic_expression_l801_801335


namespace correct_option_is_B_l801_801958

theorem correct_option_is_B :
  (∀ x : ℝ, x^2 * x^3 ≠ x^6) ∧
  (1 / (5 / 3) * (5 / 3) = 1) ∧
  (-3^2 ≠ 9) ∧
  (1 / |(-3)| ≠ 1 / (-3)) → 
  correct_option = "B" :=
by 
  intro h,
  sorry

end correct_option_is_B_l801_801958


namespace boat_return_time_l801_801714

-- Define the conditions
def distance_ahead := 70 -- km
def boat_speed := 28 -- km/h
def squadron_speed := 14 -- km/h

-- Time to return for the boat
def return_time : ℝ := 20 / 3 -- hours

-- Proof statement
theorem boat_return_time :
  let x := return_time in
  distance_ahead * 2 - squadron_speed * x = boat_speed * x :=
  by sorry

end boat_return_time_l801_801714


namespace dart_probability_l801_801290

-- Define the problem conditions
def dartboard_side_length : ℝ := 2
def random_throw : Prop := true
def distance_from_corner (x y : ℝ) : ℝ := real.sqrt (x^2 + y^2)

-- Define the regions
def quarter_circle_area (r : ℝ) : ℝ := (r^2 * real.pi) / 4
def right_triangle_area (a : ℝ) : ℝ := (a * a) / 2

-- Define the desired area
def desired_area (r a : ℝ) : ℝ := quarter_circle_area r - right_triangle_area a

-- Define the total probability
def total_area_of_quarter_square : ℝ := 1
def probability (desired_area total_area : ℝ) : ℝ := desired_area / total_area

-- Theorem statement
theorem dart_probability :
  probability (desired_area 1 1) total_area_of_quarter_square = (real.pi - 2) / 4 :=
by
  sorry

end dart_probability_l801_801290


namespace parabola_perpendicular_bisector_intersects_x_axis_l801_801411

theorem parabola_perpendicular_bisector_intersects_x_axis
  (x1 y1 x2 y2 : ℝ) 
  (A_on_parabola : y1^2 = 2 * x1)
  (B_on_parabola : y2^2 = 2 * x2) 
  (k m : ℝ) 
  (AB_line : ∀ x y, y = k * x + m)
  (k_not_zero : k ≠ 0) 
  (k_m_condition : (1 / k^2) - (m / k) > 0) :
  ∃ x0 : ℝ, x0 = (1 / k^2) - (m / k) + 1 ∧ x0 > 1 :=
by
  sorry

end parabola_perpendicular_bisector_intersects_x_axis_l801_801411


namespace remainder_5_pow_2023_mod_11_l801_801000

theorem remainder_5_pow_2023_mod_11 : (5^2023) % 11 = 4 :=
by
  have h1 : 5^2 % 11 = 25 % 11 := sorry
  have h2 : 25 % 11 = 3 := sorry
  have h3 : (3^5) % 11 = 1 := sorry
  have h4 : 3^1011 % 11 = ((3^5)^202 * 3) % 11 := sorry
  have h5 : ((3^5)^202 * 3) % 11 = (1^202 * 3) % 11 := sorry
  have h6 : (1^202 * 3) % 11 = 3 % 11 := sorry
  have h7 : (5^2023) % 11 = (3 * 5) % 11 := sorry
  have h8 : (3 * 5) % 11 = 15 % 11 := sorry
  have h9 : 15 % 11 = 4 := sorry
  exact h9

end remainder_5_pow_2023_mod_11_l801_801000


namespace candidates_count_l801_801304

theorem candidates_count (n : ℕ) (h : n * (n - 1) = 42) : n = 7 :=
by sorry

end candidates_count_l801_801304


namespace locate_2020_in_table_l801_801310

theorem locate_2020_in_table :
  ∃ r c : ℕ, r = 505 ∧ c = 508 ∧ 2020 = r * (r - 1) / 2 + c :=
begin
  -- Hypotheses and pattern conditions can be defined here
  sorry
end

end locate_2020_in_table_l801_801310


namespace max_x2y_l801_801377

noncomputable def maximum_value_x_squared_y (x y : ℝ) : ℝ :=
  if x ∈ Set.Ici 0 ∧ y ∈ Set.Ici 0 ∧ x^3 + y^3 + 3*x*y = 1 then x^2 * y else 0

theorem max_x2y (x y : ℝ) (h1 : x ∈ Set.Ici 0) (h2 : y ∈ Set.Ici 0) (h3 : x^3 + y^3 + 3*x*y = 1) :
  maximum_value_x_squared_y x y = 4 / 27 :=
sorry

end max_x2y_l801_801377


namespace three_tenths_of_number_l801_801086

theorem three_tenths_of_number (N : ℝ) (h : (1/3) * (1/4) * N = 15) : (3/10) * N = 54 :=
sorry

end three_tenths_of_number_l801_801086


namespace gamma_max_success_ratio_l801_801111

theorem gamma_max_success_ratio :
  ∀ (x y z w : ℕ),
    x > 0 → z > 0 →
    (5 * x < 3 * y) →
    (5 * z < 3 * w) →
    (y + w = 600) →
    (x + z ≤ 359) :=
by
  intros x y z w hx hz hxy hzw hyw
  sorry

end gamma_max_success_ratio_l801_801111


namespace rate_per_kg_of_grapes_l801_801693

-- Define the conditions 
namespace Problem

-- Given conditions
variables (G : ℝ) (rate_mangoes : ℝ := 55) (cost_paid : ℝ := 1055)
variables (kg_grapes : ℝ := 8) (kg_mangoes : ℝ := 9)

-- Statement to prove
theorem rate_per_kg_of_grapes : 8 * G + 9 * rate_mangoes = cost_paid → G = 70 := 
by
  intro h
  sorry -- proof goes here

end Problem

end rate_per_kg_of_grapes_l801_801693


namespace circle_area_from_circumference_l801_801207

theorem circle_area_from_circumference
  (c : ℝ)    -- the circumference
  (hc : c = 36)    -- condition: circumference is 36 cm
  : 
  ∃ A : ℝ,   -- there exists an area A
    A = 324 / π :=   -- conclusion: area is 324/π
by
  sorry   -- proof goes here

end circle_area_from_circumference_l801_801207


namespace dark_squares_exceed_light_by_one_l801_801659

theorem dark_squares_exceed_light_by_one :
  ∀ (n : ℕ), (n = 9) → 
    let num_squares := n * n in
    let dark_squares := 41 in
    let light_squares := 40 in
    dark_squares - light_squares = 1 :=
by
  intros n h
  subst h
  have num_squares := 9 * 9
  have dark_squares := 41
  have light_squares := 40
  have result := dark_squares - light_squares
  exact result
sorry

end dark_squares_exceed_light_by_one_l801_801659


namespace fish_count_l801_801938

theorem fish_count (fishbowls : ℕ) (fish_per_bowl : ℕ) (h1 : fishbowls = 261) (h2 : fish_per_bowl = 23) : fishbowls * fish_per_bowl = 6003 :=
by
  rw [h1, h2]
  norm_num
  sorry

end fish_count_l801_801938


namespace diplomats_count_l801_801511

theorem diplomats_count (D : ℝ)
  (hj : 20) 
  (hnr : 32)
  (hnr_percentage : 0.2 * D) 
  (hbjr_percentage : 0.1 * D) :
  D = 40 := 
by
  have hD : D - (20 + (D - 32) - 0.1 * D) = 0.2 * D := by sorry
  have hD_simplified : 12 - 0.1 * D = 0.2 * D := by sorry
  have hD_equation : 12 = 0.3 * D := by sorry
  exact sorry

end diplomats_count_l801_801511


namespace union_of_A_and_B_l801_801282

namespace SetUnionProof

def A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 1 }
def B : Set ℝ := { x | x ≤ 2 }
def C : Set ℝ := { x | x ≤ 2 }

theorem union_of_A_and_B : A ∪ B = C := by
  -- proof goes here
  sorry

end SetUnionProof

end union_of_A_and_B_l801_801282


namespace inverse_log_base3_l801_801375

theorem inverse_log_base3 (x : ℝ) (h : f x = logBase 3 (x + 3)) : 
  f⁻¹ 2 = 6 :=
by
  -- Define the function f
  let f (x : ℝ) := logBase 3 (x + 3)
  -- State that f⁻¹(2) = 6
  have h₁ : f (f⁻¹ 2) = 2 := by sorry
  have h₂ : f 6 = 2 := by sorry
  -- Conclude that f⁻¹(2) = 6
  exact eq.trans h₁ h₂.symm

end inverse_log_base3_l801_801375


namespace problem_l801_801163

open Set

def U := ℝ

def A := { x : ℝ | x * (x - 2) < 0 }

def B := { x : ℝ | ∃ y : ℝ, y = Real.log (1 - x) }

def complement_B_U := U \ B

theorem problem : A ∩ complement_B_U = { x : ℝ | 1 ≤ x ∧ x < 2 } := by
  sorry

end problem_l801_801163


namespace initial_percentage_female_workers_l801_801611

theorem initial_percentage_female_workers
  (E : ℕ) (P : ℕ)
  (hiring : E + 20 = 240)
  (percent_after_hiring : (55 / 100) * 240 = 132) :
  P = 60 :=
by
  have h1 : E = 220 := by linarith
  have female_initial = 132 := by linarith
  have percentage_calc : P = (132 / 220) * 100 := by sorry
  linarith

end initial_percentage_female_workers_l801_801611


namespace find_n_for_x_n_is_1995_l801_801745

def p (x : ℕ) : ℕ :=
  if x == 1 then 2 else
  well_founded.min Nat.lt_wfRel {p : ℕ | Nat.Prime p ∧ ¬ p ∣ x} sorry

def q (x : ℕ) : ℕ :=
  if p(x) == 2 then 1 else
  ∏ p in ({2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53} : Finset ℕ), 
  if p < p(x) then p else 1

def x_seq : ℕ → ℕ
| 0       := 1
| (n + 1) := x_seq n * p (x_seq n) / q (x_seq n)

theorem find_n_for_x_n_is_1995 :
  ∃ n : ℕ, x_seq n = 1995 :=
  sorry

end find_n_for_x_n_is_1995_l801_801745


namespace average_book_width_is_correct_l801_801836

theorem average_book_width_is_correct :
  let w1 := 6
  let w2 := 500 / 10
  let w3 := 1
  let w4 := 350 / 10
  let w5 := 3
  let w6 := 5
  let w7 := 750 / 10
  let w8 := 200 / 10
  average := (w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8) / 8
  in average = 24.375 := sorry

end average_book_width_is_correct_l801_801836


namespace min_shift_value_l801_801089

theorem min_shift_value (φ : ℝ) (hφ : φ > 0) :
  (∃ k : ℤ, φ = -k * π / 3 + π / 6) →
  ∃ φ_min : ℝ, φ_min = π / 6 ∧ (∀ φ', φ' > 0 → ∃ k' : ℤ, φ' = -k' * π / 3 + π / 6 → φ_min ≤ φ') :=
by
  intro h
  use π / 6
  constructor
  . sorry
  . sorry

end min_shift_value_l801_801089


namespace m_greater_than_p_l801_801159

theorem m_greater_than_p (p m n : ℕ) (hp : Nat.Prime p) (hm : 0 < m) (hn : 0 < n) (eq : p^2 + m^2 = n^2) : m > p :=
sorry

end m_greater_than_p_l801_801159


namespace value_of_g_neg3_l801_801223

def g (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem value_of_g_neg3 : g (-3) = 4 :=
by
  sorry

end value_of_g_neg3_l801_801223


namespace shortest_side_of_triangle_l801_801171

/-- One of the sides of a triangle is divided into segments of 7 and 9 units by
    the point of tangency of the inscribed circle. If the radius of the circle 
    is 5 units, and the sum of the other two sides is 36 units, then the 
    length of the shortest side is 14 units. -/
theorem shortest_side_of_triangle :
  ∀ (a b c r s Δ : ℝ),
    a = 7 →
    b = 9 →
    r = 5 →
    c = 36 - (a + b) →
    s = (a + b + c) / 2 →
    Δ = sqrt (s * (s - a) * (s - b) * (s - c)) →
    r = Δ / s →
    min (min a b) c = 14 :=
by
  sorry

end shortest_side_of_triangle_l801_801171


namespace cannot_form_triangle_l801_801690

theorem cannot_form_triangle (a b c : ℕ) (h1 : a = 3) (h2 : b = 5) (h3 : c = 2) : ¬ (a + b > c ∧ a + c > b ∧ b + c > a) :=
by {
  simp [h1, h2, h3],
  sorry
}

end cannot_form_triangle_l801_801690


namespace ladder_sliding_speed_l801_801671

theorem ladder_sliding_speed :
  ∀ (x dxdt L : ℝ) (hL : L = 5) (hdxdt : dxdt = 3) (hx : x = 1.4),
  let y := Real.sqrt (L^2 - x^2) in
  ∃ (dydt : ℝ), 2 * x * dxdt + 2 * y * dydt = 0 ∧ dydt = -0.875 :=
by
  intros x dxdt L hL hdxdt hx
  let y := Real.sqrt (L^2 - x^2)
  use (- (8.4 / (2 * y)))
  split
  {
    calc
      2 * x * dxdt + 2 * y * (- (8.4 / (2 * y)))
      = 2 * 1.4 * 3 + 2 * y * (- (8.4 / (2 * y))) : by rw [hx, hdxdt]
      ... = 0
  }
  {
    calc
      - (8.4 / (2 * y)) = - 0.875
  }
sorry

end ladder_sliding_speed_l801_801671


namespace solution_is_36_l801_801728

noncomputable def solve_equation (x : ℝ) : Prop :=
  real.cbrt (4 - x / 3) = -2

theorem solution_is_36 : solve_equation 36 :=
  by
  sorry

end solution_is_36_l801_801728


namespace mean_and_variance_transformed_set_l801_801742

-- Definitions of mean and variance
def mean (xs : List ℝ) : ℝ := (xs.sum) / (xs.length)

def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  ((xs.map (λ x => (x - μ) ^ 2)).sum) / (xs.length)

-- Problem statement
theorem mean_and_variance_transformed_set :
  let original_data := [1, 2, 3, 4, 5]
  let transformed_data := [11, 12, 13, 14, 15]
  mean transformed_data = mean original_data + 10 ∧ variance transformed_data = variance original_data := 
by
  sorry

end mean_and_variance_transformed_set_l801_801742


namespace compare_abc_l801_801021

noncomputable def a : ℝ := Real.log (1.2) / Real.log (0.7)
noncomputable def b : ℝ := 0.8^0.7
noncomputable def c : ℝ := 1.2^0.8

theorem compare_abc :
  c > b ∧ b > a :=
by
  have h₁ : a = Real.log (1.2) / Real.log (0.7) := rfl
  have h₂ : b = 0.8^0.7 := rfl
  have h₃ : c = 1.2^0.8 := rfl
  sorry

end compare_abc_l801_801021


namespace period_of_cos_div_x_over_2_l801_801629

theorem period_of_cos_div_x_over_2 :
  (∃ T, ∀ x, cos ((x + T)/2) = cos (x/2)) → T = 4 * π :=
sorry

end period_of_cos_div_x_over_2_l801_801629


namespace sufficient_material_for_box_l801_801473

theorem sufficient_material_for_box :
  ∃ (l w h : ℕ), l * w * h ≥ 1995 ∧ 2 * (l * w + w * h + h * l) ≤ 958 :=
  sorry

end sufficient_material_for_box_l801_801473


namespace necessary_but_not_sufficient_condition_arithmetic_sequence_l801_801029

theorem necessary_but_not_sufficient_condition_arithmetic_sequence 
  (S : ℕ → ℝ) (d : ℝ) (a₁ : ℝ) 
  (h₁ : a₁ = -20) 
  (h₂ : ∀ n : ℕ, S n = n/2 * (2 * a₁ + (n - 1) * d)) 
  (cond : 3 < d ∧ d < 5) 
  (min_sn_is_s6 : ∀ n : ℕ, (n ≠ 6 → S n ≥ S 6)) : 
  (10 / 3 < d ∧ d < 4) ∧ ¬((10 / 3 < d ∧ d < 4) → (3 < d ∧ d < 5)) :=
begin
  sorry
end

end necessary_but_not_sufficient_condition_arithmetic_sequence_l801_801029


namespace min_possible_value_of_x_l801_801169

theorem min_possible_value_of_x :
  ∀ (x y : ℝ),
  (69 + 53 + 69 + 71 + 78 + x + y) / 7 = 66 →
  (∀ y ≤ 100, x ≥ 0) →
  x ≥ 22 :=
by
  intros x y h_avg h_y 
  -- proof steps go here
  sorry

end min_possible_value_of_x_l801_801169


namespace converse_statement_l801_801215

theorem converse_statement (a : ℝ) : (a > 2018 → a > 2017) ↔ (a > 2017 → a > 2018) :=
by
  sorry

end converse_statement_l801_801215


namespace tan_to_sin_cos_l801_801754

theorem tan_to_sin_cos (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.cos α = 2 / 5 := 
sorry

end tan_to_sin_cos_l801_801754


namespace circle_geometry_problem_l801_801943

theorem circle_geometry_problem (A B C D : Point) (circle : Circle) (BC BD : Real)
  (angle_BAC : Real) (tangent : Line) (secant : Line)
  (hBC : BC = 4) (hBD : BD = 3) (h_angle_BAC : angle_BAC = arccos (1 / 3))
  (h_tangent : tangent.is_tangent_to circle at B)
  (h_secant : secant.is_secant_to circle at C and D)
  (h_AD_on_AC : D ∈ segment A C):
    AB = 12 / sqrt(17) ∧
    CD = 7 / sqrt(17) ∧
    R = (3 * sqrt(34)) / 4 := sorry

end circle_geometry_problem_l801_801943


namespace stock_percentage_l801_801732

theorem stock_percentage
  (total_investment : ℝ)
  (stock_price : ℝ)
  (annual_income : ℝ)
  (number_of_units : total_investment / stock_price = 50)
  (income_per_unit : annual_income / 50 = 30) :
  ((30 / stock_price) * 100 ≈ 22.06) :=
by
  have h1 : 136 = stock_price := rfl
  have h2 : 6800 = total_investment := rfl
  have h3 : 1500 = annual_income := rfl
  have h4 : 50 = 6800 / 136 := rfl
  have h5 : 30 = 1500 / 50 := rfl
  sorry

end stock_percentage_l801_801732


namespace find_a_even_function_l801_801151

-- Ensure the function is well-defined and given
def f (x : ℝ) (a : ℝ) : ℝ := x * (Real.exp x + a * Real.exp (-x))

-- Definition stating the function f is even
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem find_a_even_function :
  (is_even_function (f · a)) → (a = -1) :=
by
  -- to be proved
  sorry

end find_a_even_function_l801_801151


namespace no_int_k_sq_divides_a_k_plus_b_k_l801_801504

theorem no_int_k_sq_divides_a_k_plus_b_k 
  (a b : ℤ) (k : ℤ) (α : ℕ) 
  (h1 : odd a) (h2 : odd b) 
  (h3 : 1 < a) (h4 : 1 < b) 
  (h5 : a + b = 2^α) (h6 : α ≥ 1)
  : ¬ ∃ k, k > 1 ∧ k^2 ∣ a^k + b^k :=
begin
  sorry
end

end no_int_k_sq_divides_a_k_plus_b_k_l801_801504


namespace division_example_l801_801964

theorem division_example :
  100 / 0.25 = 400 :=
by sorry

end division_example_l801_801964


namespace find_n_l801_801346

theorem find_n (n : ℕ) (h₁ : 2^6 * 3^3 * n = factorial 10) : n = 2100 :=
sorry

end find_n_l801_801346


namespace sum_of_Ns_l801_801084

theorem sum_of_Ns (N R : ℝ) (hN_nonzero : N ≠ 0) (h_eq : N - 3 * N^2 = R) : 
  ∃ N1 N2 : ℝ, N1 ≠ 0 ∧ N2 ≠ 0 ∧ 3 * N1^2 - N1 + R = 0 ∧ 3 * N2^2 - N2 + R = 0 ∧ (N1 + N2) = 1 / 3 :=
sorry

end sum_of_Ns_l801_801084


namespace count_integers_negative_l801_801370

def f (x : ℤ) : ℤ := x^4 - 53 * x^2 + 150

theorem count_integers_negative (n : ℕ) (h : n = 12) :
  ∃ xs : Fin n → ℤ, ∀ x : ℤ, f x < 0 ↔ ∃ i, i < n ∧ x = xs i :=
by 
  sorry

end count_integers_negative_l801_801370


namespace smallest_M_l801_801363

theorem smallest_M : ∃ M : ℝ, M = 1 / 512 ∧ ∀ (a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℝ) 
  (h : a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 = 1) 
  (h_nonneg : ∀ i, 0 ≤ (list.nth_le [a1, a2, a3, a4, a5, a6, a7, a8, a9] i sorry)), -- ensure nonnegativity
  ∃ (grid : ℕ × ℕ → ℝ),
  (grid (0, 0) = a1 ∧ grid (0, 1) = a2 ∧ grid (0, 2) = a3 ∧
   grid (1, 0) = a4 ∧ grid (1, 1) = a5 ∧ grid (1, 2) = a6 ∧
   grid (2, 0) = a7 ∧ grid (2, 1) = a8 ∧ grid (2, 2) = a9) ∧
  (grid 0 0 * grid 0 1 * grid 0 2 ≤ M ∧ grid 1 0 * grid 1 1 * grid 1 2 ≤ M ∧ grid 2 0 * grid 2 1 * grid 2 2 ≤ M ∧
   grid 0 0 * grid 1 0 * grid 2 0 ≤ M ∧ grid 0 1 * grid 1 1 * grid 2 1 ≤ M ∧ grid 0 2 * grid 1 2 * grid 2 2 ≤ M) :=
begin
  use 1/512,
  split,
  { refl },
  { intros a1 a2 a3 a4 a5 a6 a7 a8 a9 h h_nonneg,
    use (λ p, match p with 
                | (0, 0) := a1 | (0, 1) := a2 | (0, 2) := a3
                | (1, 0) := a4 | (1, 1) := a5 | (1, 2) := a6
                | (2, 0) := a7 | (2, 1) := a8 | (2, 2) := a9
                | _        := 0
              end),
    split,
    { repeat { split }; try { refl } }, -- establishing the grid variable
    { sorry } -- actual provision of the example arrangement and proof that all products are ≤ 1/512
  }
end

end smallest_M_l801_801363


namespace part_I_part_II_l801_801102

-- Define the triangle and sides
structure Triangle :=
  (A B C : ℝ)   -- angles in the triangle
  (a b c : ℝ)   -- sides opposite to respective angles

-- Express given conditions in the problem
def conditions (T: Triangle) : Prop :=
  2 * (1 / (Real.tan T.A) + 1 / (Real.tan T.C)) = 1 / (Real.sin T.A) + 1 / (Real.sin T.C)

-- First theorem statement
theorem part_I (T : Triangle) : conditions T → (T.a + T.c = 2 * T.b) :=
sorry

-- Second theorem statement
theorem part_II (T : Triangle) : conditions T → (T.B ≤ Real.pi / 3) :=
sorry

end part_I_part_II_l801_801102


namespace total_cost_of_books_l801_801687

def book_cost (num_mathbooks num_artbooks num_sciencebooks cost_mathbook cost_artbook cost_sciencebook : ℕ) : ℕ :=
  (num_mathbooks * cost_mathbook) + (num_artbooks * cost_artbook) + (num_sciencebooks * cost_sciencebook)

theorem total_cost_of_books :
  let num_mathbooks := 2
  let num_artbooks := 3
  let num_sciencebooks := 6
  let cost_mathbook := 3
  let cost_artbook := 2
  let cost_sciencebook := 3
  book_cost num_mathbooks num_artbooks num_sciencebooks cost_mathbook cost_artbook cost_sciencebook = 30 :=
by
  sorry

end total_cost_of_books_l801_801687


namespace no_nat_n_nn_minus_6n_plus_5_prime_l801_801351

theorem no_nat_n_nn_minus_6n_plus_5_prime : 
  ∀ n : ℕ, ¬ prime (n^n - 6*n + 5) := 
by
  sorry

end no_nat_n_nn_minus_6n_plus_5_prime_l801_801351


namespace area_inside_Q_outside_P_and_R_l801_801327

noncomputable def circle (radius : ℝ) := { center : ℝ × ℝ }

def circleP : circle := circle 1
def circleQ : circle := circle 2
def circleR : circle := circle 1

-- Conditions
def condition1 := (center circleP = (0, 0))
def condition2 := (center circleR = (2, 0))
def condition3 := (center circleQ = (0, 0))
def condition4 := tangent externally circleQ circleR
def condition5 := tangent externally circleR circleP

-- Question Translation
def area_difference := area circleQ - area circleP - area circleR

theorem area_inside_Q_outside_P_and_R :
  (area_difference = 2 * π) :=
by
  sorry

end area_inside_Q_outside_P_and_R_l801_801327


namespace area_ABC_l801_801101

variable (A B C : Type) [EuclideanAffineSpace A]
variables {P Q R : A} (hA : ∠(P, Q, R) = 30) (hPQ : dist P Q = sqrt 3) (hQR : dist Q R = 1)

def area_of_triangle (P Q R : A) : ℝ :=
1 / 2 * (dist P Q) * (dist Q R) * sin 30

theorem area_ABC (hA : ∠(P, Q, R) = 30)
    (hPQ : dist P Q = sqrt 3) (hQR : dist Q R = 1) :
  area_of_triangle P Q R = sqrt 3 / 2 ∨ area_of_triangle P Q R = sqrt 3 / 4 :=
by
  sorry

end area_ABC_l801_801101


namespace smallest_positive_z_l801_801565

noncomputable def find_z (x z : ℝ) : Prop :=
  cos x = 0 ∧ cos (x + z) = -1 / 2 ∧ z > 0

theorem smallest_positive_z (x z : ℝ) (m n : ℤ) :
  find_z x z → z = 11 * pi / 6 :=
by
  sorry

end smallest_positive_z_l801_801565


namespace paint_faces_l801_801431

def cuboid_faces : ℕ := 6
def number_of_cuboids : ℕ := 8 
def total_faces_painted : ℕ := cuboid_faces * number_of_cuboids

theorem paint_faces (h1 : cuboid_faces = 6) (h2 : number_of_cuboids = 8) : total_faces_painted = 48 := by
  -- conditions are defined above
  sorry

end paint_faces_l801_801431


namespace max_tiles_to_spell_CMWMC_l801_801974

theorem max_tiles_to_spell_CMWMC {Cs Ms Ws : ℕ} (hC : Cs = 8) (hM : Ms = 8) (hW : Ws = 8) : 
  ∃ (max_draws : ℕ), max_draws = 18 :=
by
  -- Assuming we have 8 C's, 8 M's, and 8 W's in the bag
  sorry

end max_tiles_to_spell_CMWMC_l801_801974


namespace smallest_three_digit_divisible_by_4_and_5_l801_801269

theorem smallest_three_digit_divisible_by_4_and_5 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (n % 4 = 0) ∧ (n % 5 = 0) ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (m % 4 = 0) ∧ (m % 5 = 0) → m ≥ n →
n = 100 :=
sorry

end smallest_three_digit_divisible_by_4_and_5_l801_801269


namespace cos_double_x0_zero_of_f_x0_l801_801403

noncomputable def f (x : ℝ) : ℝ :=
  sin (x) ^ 2 + 2 * sqrt 3 * sin (x) * cos (x) + sin (x + π / 4) * sin (x - π / 4)

theorem cos_double_x0_zero_of_f_x0
  (x0 : ℝ) (h : 0 ≤ x0 ∧ x0 ≤ π / 2)
  (h_zero : f(x0) = 0) :
  cos (2 * x0) = (3 * sqrt 5 + 1) / 8 :=
begin
  sorry
end

end cos_double_x0_zero_of_f_x0_l801_801403


namespace win_percentage_of_people_with_envelopes_l801_801814

theorem win_percentage_of_people_with_envelopes (total_people : ℕ) (percent_with_envelopes : ℝ) (winners : ℕ) (num_with_envelopes : ℕ) : 
  total_people = 100 ∧ percent_with_envelopes = 0.40 ∧ num_with_envelopes = total_people * percent_with_envelopes ∧ winners = 8 → 
    (winners / num_with_envelopes) * 100 = 20 :=
by
  intros
  sorry

end win_percentage_of_people_with_envelopes_l801_801814


namespace max_column_sum_l801_801726

-- Definitions of the 5x5 grid and related constraints
def grid : Type := Matrix (Fin 5) (Fin 5) ℕ

def is_valid_grid (G : grid) : Prop :=
  (∀ i j, 1 ≤ G i j ∧ G i j ≤ 5) ∧  -- All numbers are within the range 1 to 5
  (∀ i j k, | G i j - G i k | ≤ 2) ∧  -- Absolute difference in columns is at most 2
  (∀ i, (Finset.univ.map (G i)).card = 5) ∧  -- Each row contains unique numbers 1 to 5
  (∀ j, (Finset.univ.map (G i)).card = 5)    -- Each column contains unique numbers 1 to 5

def column_sum (G : grid) (j : Fin 5) : ℕ := 
  (Finset.univ.sum (λ i => G i j))

-- The maximum possible value of M
def max_M (G : grid) : ℕ :=
  Finset.univ.min' (Finset.image (column_sum G) Finset.univ)

theorem max_column_sum : ∃ G : grid, is_valid_grid G ∧ max_M G = 10 := 
by 
  sorry

end max_column_sum_l801_801726


namespace exponent_of_5_in_40_factorial_l801_801824

theorem exponent_of_5_in_40_factorial :
  ∀ (n : ℕ), n = 40 → multiplicity 5 (factorial n) = 10 :=
by
  intro n
  intro h
  rw [h, nat.factorial]
  sorry

end exponent_of_5_in_40_factorial_l801_801824


namespace divide_square_into_three_shapes_with_given_perimeter_l801_801966

def divide_square_into_shapes (square_side_length : ℝ) (shape_perimeter : ℝ) : Prop :=
square_side_length = 4 ∧ shape_perimeter = 16 ∧
  ∃ (dividing_lines : list (ℝ × ℝ × ℝ × ℝ)), 
    (∀ (x1 y1 x2 y2 : ℝ), (x1, y1, x2, y2) ∈ dividing_lines →
      (step_type : ℝ) -- This ensures that each line is reasonable
      ∧ dividing_lines.length >= 3 -- Ensure at least 3 valid divisions
      ∃ (shape1 shape2 : set (ℝ × ℝ)), 
        ∃ (shape1_perimeter shape2_perimeter : ℝ), 
          (boundary shape1 shape1_perimeter) ∧ 
          (side_divider shape1 shape1_perimeter)
            ∧ (boundary shape2 shape2_perimeter) 
            ∧ (side_holder shape2 shape2_perimeter)
            (shape1_perimeter = 16)
            (shape2_perimeter = 16)

theorem divide_square_into_three_shapes_with_given_perimeter :
  divide_square_into_shapes 4 16 :=
begin
  sorry
end

end divide_square_into_three_shapes_with_given_perimeter_l801_801966


namespace rabbit_run_time_l801_801300

-- Define the constants and the problem
def rabbit_speed : ℝ := 5 -- in miles per hour
def distance : ℝ := 2     -- in miles
def time_in_minutes (s d : ℝ) : ℝ := (d / s) * 60 

-- The theorem we want to prove
theorem rabbit_run_time : time_in_minutes rabbit_speed distance = 24 :=
by
  -- Skip the proof for now
  sorry

end rabbit_run_time_l801_801300


namespace sqrt_48_not_integer_floor_sqrt_48_eq_6_f_sqrt_48_l801_801160

noncomputable def f (x : ℝ) : ℝ :=
if x % 1 = 0 then 7 * x + 6 else real.floor x + 7

theorem sqrt_48_not_integer : ¬(∃ n : ℤ, n = real.sqrt 48) := sorry

theorem floor_sqrt_48_eq_6 : real.floor (real.sqrt 48) = 6 := sorry

theorem f_sqrt_48 : f (real.sqrt 48) = 13 :=
by
  rw [f, sqrt_48_not_integer, floor_sqrt_48_eq_6]
  simp [f, if_neg]
  sorry

end sqrt_48_not_integer_floor_sqrt_48_eq_6_f_sqrt_48_l801_801160


namespace count_linear_equations_l801_801051

theorem count_linear_equations {x y : ℝ} :
  let eq1 := x = 1,
      eq2 := x - 2 = 12,
      eq3 := x^2 + x + 1 = 0,
      eq4 := x * y = 0,
      eq5 := 2 * x + y = 0 in
  (if eq1 then 1 else 0) + (if eq2 then 1 else 0) +
  (if eq3 then 0 else 0) + (if eq4 then 0 else 0) +
  (if eq5 then 0 else 0) = 2 :=
by { sorry }

end count_linear_equations_l801_801051


namespace problem_l801_801078

theorem problem (a b : ℝ) (h : a > b) (k : b > 0) : b * (a - b) > 0 := 
by
  sorry

end problem_l801_801078


namespace min_positive_period_and_symmetry_axis_l801_801229

noncomputable def f (x : ℝ) := - (Real.sin (x + Real.pi / 6)) * (Real.sin (x - Real.pi / 3))

theorem min_positive_period_and_symmetry_axis :
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧ (∃ k : ℤ, ∀ x : ℝ, f x = f (x + 1 / 2 * k * Real.pi + Real.pi / 12)) := by
  sorry

end min_positive_period_and_symmetry_axis_l801_801229


namespace max_dogs_and_fish_l801_801809

theorem max_dogs_and_fish (d c b p f : ℕ) (h_ratio : d / 7 = c / 7 ∧ d / 7 = b / 8 ∧ d / 7 = p / 3 ∧ d / 7 = f / 5)
  (h_dogs_bunnies : d + b = 330)
  (h_twice_fish : f ≥ 2 * c) :
  d = 154 ∧ f = 308 :=
by
  -- This is where the proof would go
  sorry

end max_dogs_and_fish_l801_801809


namespace max_production_term_l801_801661

-- Define the cumulative production function
def f (n : ℕ) : ℝ := (1 / 2) * n * (n + 1) * (2 * n + 1)

-- Define the annual production function
def annual_production (n : ℕ) : ℝ :=
  if n = 1 then
    f 1
  else
    f n - f (n - 1)

-- Define the maximum safe annual production limit
def max_annual_production : ℝ := 150

-- Theorem statement: the maximum production term such that the annual production does not exceed 150 tons is 7
theorem max_production_term : ∀ n : ℕ, annual_production n ≤ max_annual_production ↔ n ≤ 7 := sorry

end max_production_term_l801_801661


namespace prob_log_a_b_is_int_l801_801625

theorem prob_log_a_b_is_int : 
  (Set.Choose_in_Set 20 2 * 10 / ∑ (x : ℕ) in (Finset.range 20).filter (λ x, x > 0), 
  ( (Finset.range 20).filter (λ y, y > 0 ∧ y ≠ x ∧ y % x == 0 )).card ) = 47 / 190 
  sorry

end prob_log_a_b_is_int_l801_801625


namespace test_point_third_l801_801825

def interval := (1000, 2000)
def phi := 0.618
def x1 := 1000 + phi * (2000 - 1000)
def x2 := 1000 + 2000 - x1

-- By definition and given the conditions, x3 is computed in a specific manner
def x3 := x2 + 2000 - x1

theorem test_point_third : x3 = 1764 :=
by
  -- Skipping the proof for now
  sorry

end test_point_third_l801_801825


namespace triangle_midsegments_equal_l801_801336

variables {A B C A1 B1 C1 D : Point}

def midpoint (P Q R : Point) : Prop := dist P R = dist Q R

-- Declare the midpoints and conditions
variables (hA1 : midpoint A1 B C)
variables (hB1 : midpoint B1 A C)
variables (hC1 : midpoint C1 A B)
variables (hAD_parallel_BB1 : parallel (line_through A1 D) (line_through B1 B))

-- Prove the sides of the triangle AA1D are equal to the midsegments of ABC
theorem triangle_midsegments_equal (hA1 : midpoint A1 B C) (hB1 : midpoint B1 A C)
(hC1 : midpoint C1 A B) (hAD_parallel_BB1 : parallel (line_through A1 D) (line_through B1 B)) :
AD = dist C C1 := sorry

end triangle_midsegments_equal_l801_801336


namespace min_posts_required_l801_801301

theorem min_posts_required :
  ∀ (length width : ℕ) (post_spacing : ℕ),
  length = 50 →
  width = 30 →
  post_spacing = 10 →
  (length / post_spacing + 1) + 2 * (width / post_spacing + 1 - 1) = 12 :=
by
  intros length width post_spacing hlength hwidth hspacing
  rw [hlength, hwidth, hspacing]
  calc
    (50 / 10 + 1) + 2 * (30 / 10 + 1 - 1)
      = 5 + 1 + 2 * 3                : by norm_num
  ... = 6 + 6                        : by linarith
  ... = 12                           : by norm_num

end min_posts_required_l801_801301


namespace jill_spent_30_percent_on_food_l801_801513

variables (T F : ℝ)

theorem jill_spent_30_percent_on_food
  (h1 : 0.04 * T = 0.016 * T + 0.024 * T)
  (h2 : 0.40 + 0.30 + F = 1) :
  F = 0.30 :=
by 
  sorry

end jill_spent_30_percent_on_food_l801_801513


namespace probability_of_stopping_after_5_draws_l801_801940

def color := {red, yellow, blue}
def draws (n : ℕ) := vector color n

def valid_sequences (first4 : draws 4) (fifth : color) :=
  let first4_colors := first4.to_list in
  ∃ c1 c2 : color, c1 ≠ c2 ∧
    (c1 ∈ first4_colors) ∧ (c2 ∈ first4_colors) ∧ 
    c1 ≠ fifth ∧ c2 ≠ fifth ∧ 
    fifth ≠ c1 ∧ fifth ≠ c2

theorem probability_of_stopping_after_5_draws : 
  (∃ (first4 : draws 4) (fifth : color), valid_sequences first4 fifth) → 
  (3 ^ 4 - 4) / (3 ^ 5) = 4 / 27 := 
by sorry

end probability_of_stopping_after_5_draws_l801_801940


namespace cans_per_bag_l801_801897

def total_cans : ℕ := 42
def bags_saturday : ℕ := 4
def bags_sunday : ℕ := 3
def total_bags : ℕ := bags_saturday + bags_sunday

theorem cans_per_bag (h1 : total_cans = 42) (h2 : total_bags = 7) : total_cans / total_bags = 6 :=
by {
    -- proof body to be filled
    sorry
}

end cans_per_bag_l801_801897


namespace net_population_increase_per_day_l801_801275

def birth_rate : Nat := 4
def death_rate : Nat := 2
def seconds_per_day : Nat := 24 * 60 * 60

theorem net_population_increase_per_day : 
  (birth_rate - death_rate) * (seconds_per_day / 2) = 86400 := by
  sorry

end net_population_increase_per_day_l801_801275


namespace max_n_l801_801760

-- Definitions of given conditions
variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

axiom condition1 : ∀ n, a (n + 1) + a n = 2 * n + 1
axiom condition2 : S 2019 = ∑ i in range(64), a i
axiom condition3 : a 2 < 2

-- Goal
theorem max_n (n : ℕ) : a (n + 1) + a n = 2 * n + 1 → S 2019 = 2019 → a 2 < 2 → n <= 63 := 
sorry

end max_n_l801_801760


namespace monotonicity_decreasing_range_l801_801924

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

theorem monotonicity_decreasing_range (ω : ℝ) :
  (∀ x y : ℝ, (π / 2 < x ∧ x < π ∧ π / 2 < y ∧ y < π ∧ x < y) → f ω x > f ω y) ↔ (1 / 2 ≤ ω ∧ ω ≤ 5 / 4) :=
sorry

end monotonicity_decreasing_range_l801_801924


namespace minimum_length_of_AG_l801_801505

noncomputable def centroid (a b c : ℝ × ℝ) : ℝ × ℝ :=
  (1/3 * (a.1 + b.1 + c.1), 1/3 * (a.2 + b.2 + c.2))

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def length (u : ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1 * u.1 + u.2 * u.2)

theorem minimum_length_of_AG {A B C G : ℝ × ℝ}
  (hG : G = centroid A B C)
  (hAngleA : ∠(B, A, C) = 120)
  (hDot : dot_product (B - A) (C - A) = -1) :
  length (G - A) = Real.sqrt(2) / 3 :=
sorry

end minimum_length_of_AG_l801_801505


namespace expression_value_l801_801082

theorem expression_value (x y : ℤ) (h1 : x = 3) (h2 : y = 4) : 3 * x - 5 * y + 7 = -4 :=
by {
  rw [h1, h2],
  norm_num,
}

end expression_value_l801_801082


namespace complex_value_is_minus_sixteen_l801_801739

noncomputable def complex_value : ℂ :=
  (-1 + real.sqrt 3 * complex.I)^5 / (1 + real.sqrt 3 * complex.I)

theorem complex_value_is_minus_sixteen : complex_value = -16 := by
  sorry

end complex_value_is_minus_sixteen_l801_801739


namespace tangent_line_at_point_l801_801913

open Real

def curve (x : ℝ) : ℝ := exp x / (x + 1)

def tangent_line (x : ℝ) : ℝ := (exp 1 / 4) * x + (exp 1 / 4)

theorem tangent_line_at_point :
  tangent_line = λ x, (exp 1 / 4) * x + (exp 1 / 4) :=
by
  sorry

end tangent_line_at_point_l801_801913


namespace last_integer_in_sequence_l801_801931

theorem last_integer_in_sequence :
  ∃ (n : ℕ), (n = 250) ∧ ∃ (num_seq : ℕ → ℝ), 
  (num_seq 0 = 1024000) ∧ 
  (∀ (i : ℕ), num_seq (i + 1) = num_seq i / 4 ∧ num_seq (i + 1).denominator = 1) ∧
  (∀ (j : ℕ), num_seq (j + 1).denominator ≠ 1 → num_seq j = n) :=
sorry

end last_integer_in_sequence_l801_801931


namespace middle_sign_determination_l801_801509

theorem middle_sign_determination 
  (a : Fin 9 → ℝ)
  (H1 : ∀ i : Fin 8, a i + a ⟨i + 1, by linarith⟩ < 0)
  (H2 : (∑ i : Fin 9, a i) > 0) :
  a ⟨4, by linarith⟩ > 0 ∧ a ⟨3, by linarith⟩ < 0 ∧ a ⟨5, by linarith⟩ < 0 :=
by 
  sorry

end middle_sign_determination_l801_801509


namespace length_AC_l801_801805

variable {A B C : Type} [Field A] [Field B] [Field C]

-- Definitions for the problem conditions
noncomputable def length_AB : ℝ := 3
noncomputable def angle_A : ℝ := Real.pi * 120 / 180
noncomputable def area_ABC : ℝ := (15 * Real.sqrt 3) / 4

-- The theorem statement
theorem length_AC (b : ℝ) (h1 : b = length_AB) (h2 : angle_A = Real.pi * 120 / 180) (h3 : area_ABC = (15 * Real.sqrt 3) / 4) : b = 5 :=
sorry

end length_AC_l801_801805


namespace average_age_when_youngest_born_l801_801973

theorem average_age_when_youngest_born (total_people : ℕ) (average_current_age : ℕ) (age_youngest : ℕ) :
  total_people = 7 → average_current_age = 30 → age_youngest = 5 →
  (total_people * average_current_age - age_youngest) / (total_people - 1) = 34.17 :=
by
  sorry

end average_age_when_youngest_born_l801_801973


namespace rabbits_clear_land_in_21_days_l801_801259

theorem rabbits_clear_land_in_21_days (length_feet width_feet : ℝ) (rabbits : ℕ) (clear_per_rabbit_per_day : ℝ) : 
  length_feet = 900 → width_feet = 200 → rabbits = 100 → clear_per_rabbit_per_day = 10 →
  (⌈ (length_feet / 3 * width_feet / 3) / (rabbits * clear_per_rabbit_per_day) ⌉ = 21) := 
by
  intros
  sorry

end rabbits_clear_land_in_21_days_l801_801259


namespace work_completion_time_l801_801284

theorem work_completion_time :
  let A_rate := 1/4
  let B_rate := 1/8
  let C_rate := 1/6
  let combined_AB_rate := A_rate + B_rate
  let two_days_work := 2 * combined_AB_rate
  let remaining_work := 1 - two_days_work
  let combined_BC_rate := B_rate + C_rate
  (combined_BC_rate * (6 / 7) = remaining_work) :=
by
  let A_rate := (1 : ℚ) / 4
  let B_rate := (1: ℚ) / 8
  let C_rate := (1: ℚ) / 6
  let combined_AB_rate := A_rate + B_rate
  let two_days_work := 2 * combined_AB_rate
  let remaining_work := 1 - two_days_work
  let combined_BC_rate := B_rate + C_rate
  have h: combined_BC_rate * (6 / 7) = remaining_work := by
    sorry
  exact h

end work_completion_time_l801_801284


namespace max_airplane_companies_l801_801983

def number_of_cities := 100
def number_of_flights := 2018

theorem max_airplane_companies (n : ℕ) : 
  (exists (G : Graph number_of_cities), G.edges = number_of_flights ∧ G.diameter ≥ 3) →
  (∃ (k : ℕ), k = number_of_flights - number_of_cities + 2 ∧ k = n) :=
begin
  sorry
end

end max_airplane_companies_l801_801983


namespace intersection_line_slope_l801_801373

theorem intersection_line_slope (s : ℝ) :
  let l1 (x y : ℝ) := x + 3 * y = 8 * s + 4,
      l2 (x y : ℝ) := x - 2 * y = 3 * s - 1 in
  ∃ k : ℝ, k = 1 / 5 ∧
    ∀ x y : ℝ, (l1 x y ∧ l2 x y) → (y = k * x + 4 / 5) :=
sorry

end intersection_line_slope_l801_801373


namespace tan_sum_simplification_l801_801551
-- We start by importing the relevant Lean libraries that contain trigonometric functions and basic real analysis.

-- Define the statement to be proved in Lean.
theorem tan_sum_simplification :
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12) = 4 * Real.sqrt 2 - 4) :=
by
  sorry

end tan_sum_simplification_l801_801551


namespace dice_probability_l801_801804

/-- Probability that exactly two of the three fair 6-sided dice show even numbers 
and the third one shows an odd number less than 3 is 1/8. -/
theorem dice_probability : 
  let even_prob := 3 / 6
  let odd_prob := 1 / 6
  3 * (even_prob * even_prob * odd_prob) = 1 / 8 :=
by
  -- Definitions of probabilities
  let even_prob := 3 / 6
  let odd_prob := 1 / 6

  -- Calculated result
  have prob := 3 * (even_prob * even_prob * odd_prob)
  
  -- Simplify the probability expression
  calc
    prob = 3 * ((3 / 6) * (3 / 6) * (1 / 6)) : by rfl
    ... = 3 * (9 / 216) : by norm_num
    ... = 27 / 216 : by norm_num
    ... = 1 / 8 : by norm_num

end dice_probability_l801_801804


namespace probability_of_blue_face_l801_801289

theorem probability_of_blue_face :
  (cube_faces : Finset ℕ) (blue_faces : Finset ℕ) (total_faces : ℕ) (blue_count : ℕ)
  (h₁ : total_faces = 6) (h₂ : blue_count = 3) (h₃ : cube_faces.card = total_faces) (h₄ : blue_faces.card = blue_count) :
  (blue_faces.card.to_nat / cube_faces.card.to_nat) = 1/2 :=
sorry

end probability_of_blue_face_l801_801289


namespace fraction_blue_after_doubling_l801_801807

theorem fraction_blue_after_doubling (x : ℕ) (h1 : ∃ x, (2 : ℚ) / 3 * x + (1 : ℚ) / 3 * x = x) :
  ((2 * (2 / 3 * x)) / ((2 / 3 * x) + (1 / 3 * x))) = (4 / 5) := by
  sorry

end fraction_blue_after_doubling_l801_801807


namespace find_a_plus_b_l801_801046

theorem find_a_plus_b (a b : ℝ) :
  (∀ x : ℝ, |x - 2| > 1 ↔ x^2 + a * x + b > 0) →
  a + b = -1 :=
by {
  assume h : ∀ x : ℝ, |x - 2| > 1 ↔ x^2 + a * x + b > 0,
  sorry
}

end find_a_plus_b_l801_801046


namespace dot_product_x_pi_div_3_cos_2x_from_dot_product_l801_801419

open Real

noncomputable
def m (x : ℝ) : ℝ × ℝ := (sqrt 3 * cos x, -1)

noncomputable
def n (x : ℝ) : ℝ × ℝ := (sin x, cos (x)^2)

theorem dot_product_x_pi_div_3 :
  m (π / 3) • n (π / 3) = 1 / 2 := by
sorry

theorem cos_2x_from_dot_product {x : ℝ} (h1 : 0 ≤ x) (h2 : x ≤ π / 4)
  (h : m x • n x = sqrt 3 / 3 - 1 / 2) :
  cos (2 * x) = (3 * sqrt 2 - sqrt 3) / 6 := by
sorry

end dot_product_x_pi_div_3_cos_2x_from_dot_product_l801_801419


namespace altitudes_intersect_on_circle_l801_801578

theorem altitudes_intersect_on_circle (circle : Set Point) (A B C D M A₁ D₁ : Point)
  (h_circle : is_circle circle)
  (h_intersect : M ∈ circle)
  (h_AM_AC : dist A M = dist A C)
  (h_chords : chord A B circle ∧ chord C D circle)
  (h_altitudes : altitude A A₁ (triangle A C M) ∧ altitude D D₁ (triangle B D M)) :
  ∃ Q ∈ circle, lies_on_line A₁ Q ∧ lies_on_line D₁ Q :=
by
  sorry

end altitudes_intersect_on_circle_l801_801578


namespace tangent_line_correct_l801_801920

noncomputable def tangent_line_equation : ℝ → ℝ := 
  fun x => (exp(1) / 4) * x + (exp(1) / 4)

theorem tangent_line_correct :
  (∀ x, (x ≠ 1 → tangent_line_equation x = (exp(1) / 4) * x + (exp(1) / 4)) ∧
       (∀ y, (y = 1 → tangent_line_equation 1 = exp(1) / 2)) ∧
       (∀ f, (f x = exp(x) / (x + 1) ∧ x = 1 → deriv f 1 = exp(1) / 4))) :=
by
  sorry

end tangent_line_correct_l801_801920


namespace range_of_positive_integers_in_list_K_l801_801861

/-- List K consists of 12 consecutive integers,
with -5 being the least integer and the sum of all negative integers in the list being -21.
Prove that the range of the positive integers in list K is 5. -/
theorem range_of_positive_integers_in_list_K :
  ∃ K : List ℤ, K.length = 12 ∧ 
                (-5 = K.head) ∧ 
                (∑ k in K.filter (λ x, x < 0), x = -21) ∧ 
                (range_of (K.filter (λ x, x > 0)) = 5) :=
sorry

end range_of_positive_integers_in_list_K_l801_801861


namespace collinear_A_S_H_l801_801467

noncomputable def circlesIntersectAt (O₁ O₂ A B : Point) : Prop := sorry -- Definition of two circles intersecting at A and B
noncomputable def commonTangent (O₁ O₂ P Q : Point) : Prop := sorry -- Definition of a common tangent intersecting circles at P and Q
noncomputable def tangentsIntersectAt (A P Q S : Point) : Prop := sorry -- Definition that tangents at P and Q to the circumcircle of APQ intersect at S
noncomputable def reflectionAcrossLine (B PQ H : Point) : Prop := sorry -- Definition that H is the reflection of B across line PQ

theorem collinear_A_S_H
  (O₁ O₂ A B P Q S H : Point)
  (hc : circlesIntersectAt O₁ O₂ A B)
  (ht : commonTangent O₁ O₂ P Q)
  (hi : tangentsIntersectAt A P Q S)
  (hr : reflectionAcrossLine B PQ H) :
  collinear A S H :=
sorry

end collinear_A_S_H_l801_801467


namespace h_2023_sum_of_digits_l801_801156

noncomputable def f (x : ℝ) : ℝ := 2^(5*x)
noncomputable def g (x : ℝ) : ℝ := logBase 10 (x / 5)
noncomputable def h1 (x : ℝ) : ℝ := g (f x)
noncomputable def h : ℕ → ℝ → ℝ
| 1       , x := h1 x
| (n + 1) , x := h1 (h n x)

theorem h_2023_sum_of_digits :
  let h2023 := h 2023 1 in sum_of_digits h2023 = 2023 :=
by sorry

end h_2023_sum_of_digits_l801_801156


namespace playerA_winning_moves_l801_801116

-- Definitions of the game
-- Circles are labeled from 1 to 9
inductive Circle
| A | B | C1 | C2 | C3 | C4 | C5 | C6 | C7

inductive Player
| A | B

def StraightLine (c1 c2 c3 : Circle) : Prop := sorry
-- The straight line property between circles is specified by the game rules

-- Initial conditions
def initial_conditions (playerA_move playerB_move : Circle) : Prop :=
  playerA_move = Circle.A ∧ playerB_move = Circle.B

-- Winning condition
def winning_move (move : Circle) : Prop := sorry
-- This will check if a move leads to a win for Player A

-- Equivalent proof problem
theorem playerA_winning_moves : ∀ (move : Circle), initial_conditions Circle.A Circle.B → 
  (move = Circle.C2 ∨ move = Circle.C3 ∨ move = Circle.C4) → winning_move move :=
by
  sorry

end playerA_winning_moves_l801_801116


namespace total_sequences_l801_801866

theorem total_sequences (n1 n2 m1 m2: ℕ) (h1: n1 = 12) (h2: n2 = 9) (h3: m1 = 3) (h4: m2 = 2): 
  n1 ^ m1 * n2 ^ m2 = 139968 :=
by 
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_sequences_l801_801866


namespace total_weight_loss_l801_801178

def seth_loss : ℝ := 17.53
def jerome_loss : ℝ := 3 * seth_loss
def veronica_loss : ℝ := seth_loss + 1.56
def seth_veronica_loss : ℝ := seth_loss + veronica_loss
def maya_loss : ℝ := seth_veronica_loss - 0.25 * seth_veronica_loss
def total_loss : ℝ := seth_loss + jerome_loss + veronica_loss + maya_loss

theorem total_weight_loss : total_loss = 116.675 := by
  sorry

end total_weight_loss_l801_801178


namespace john_anna_ebook_readers_l801_801484

-- Definitions based on conditions
def anna_bought : ℕ := 50
def john_buy_diff : ℕ := 15
def john_lost : ℕ := 3

-- Main statement
theorem john_anna_ebook_readers :
  let john_bought := anna_bought - john_buy_diff in
  let john_remaining := john_bought - john_lost in
  john_remaining + anna_bought = 82 :=
by
  sorry

end john_anna_ebook_readers_l801_801484


namespace value_of_a10_l801_801386

def sequence (a : ℕ → ℚ) (n : ℕ) : Prop :=
  a 1 = 1/4 ∧ (∀ n > 1, a n = 1 - (1/a (n - 1)))

theorem value_of_a10 (a : ℕ → ℚ) (h : sequence a 10) : a 10 = 1/4 :=
by
  sorry

end value_of_a10_l801_801386


namespace probability_sum_less_than_10_l801_801252

-- Two fair, six-sided dice are rolled.
def fair_die := fin 6

-- Each die has 6 faces.
def num_faces_die : ℕ := 6

-- Total number of outcomes when two dice are rolled.
def total_outcomes : ℕ := num_faces_die * num_faces_die

-- The successful outcomes for the sum being less than 10
def successful_outcomes : set (fair_die × fair_die) :=
  { (d1, d2) | d1.1 + d2.1 + 2 < 10 }  -- Adjust for 1-based indexing of dice faces

-- Probability calculation
def probability (s : set (fair_die × fair_die)) : ℚ :=
  (finset.card s) / (total_outcomes)

theorem probability_sum_less_than_10 :
  probability successful_outcomes = 5 / 6 := by
  sorry

end probability_sum_less_than_10_l801_801252


namespace vector_dot_product_l801_801752

def a : ℤ × ℤ := (1, -2)
def b : ℤ × ℤ := (-3, 4)
def c : ℤ × ℤ := (3, 2)

theorem vector_dot_product :
  let a := (1, -2)
  let b := (-3, 4)
  let c := (3, 2)
  (2 * (a.1, a.2) + (b.1, b.2)) • (c.1, c.2) = -3 :=
by
  sorry

end vector_dot_product_l801_801752


namespace total_stickers_l801_801864

theorem total_stickers :
  (20.0 : ℝ) + (26.0 : ℝ) + (20.0 : ℝ) + (6.0 : ℝ) + (58.0 : ℝ) = 130.0 := by
  sorry

end total_stickers_l801_801864


namespace least_possible_value_l801_801399

-- Define the set of distinct numbers
def distinct_set : Set ℕ := {2, 3, 5, 7}

open Set

-- Define the expression
def expression (a b c d : ℕ) : ℝ :=
  ((a + b : ℝ) / (c - d : ℝ)) / 2

-- Define the problem
theorem least_possible_value :
  ∀ (a b c d : ℕ),
    a ∈ distinct_set ∧ b ∈ distinct_set ∧ c ∈ distinct_set ∧ d ∈ distinct_set ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    expression a b c d = (if a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 7 then -1.25 else sorry) :=
by 
  sorry

end least_possible_value_l801_801399


namespace rotate_A_90_degrees_clockwise_l801_801172

theorem rotate_A_90_degrees_clockwise :
  let A : ℝ × ℝ := (-4, 1)
  rotate_90_clockwise (A) = (1, 4) := 
  sorry

/-- Helper function to perform a 90-degree clockwise rotation on a given point (ℝ × ℝ) around origin -/
def rotate_90_clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.snd, -p.fst)

end rotate_A_90_degrees_clockwise_l801_801172


namespace scientific_notation_correct_l801_801195

theorem scientific_notation_correct : 
  let n := 2720000
  in ScientificNotation n (2.72, 6) := 
sorry

end scientific_notation_correct_l801_801195


namespace smallest_positive_angle_l801_801325

theorem smallest_positive_angle (y : ℝ) (h : 8 * sin y * cos y ^ 3 - 8 * sin y ^ 3 * cos y = 1) : y = 7.5 := 
sorry

end smallest_positive_angle_l801_801325


namespace hexagon_diagonal_length_proof_l801_801997

noncomputable def hexagon_diagonal_length : ℝ :=
  let a := 4
  let b := 6
  Real.sqrt (a^2 + b^2)

theorem hexagon_diagonal_length_proof :
  ∃ (AC BD : ℝ), AC = Real.sqrt 52 ∧ BD = Real.sqrt 52 :=
by
  let a := 4
  let b := 6
  use Real.sqrt (a^2 + b^2)
  use Real.sqrt (a^2 + b^2)
  simp only [hexagon_diagonal_length]
  exact ⟨rfl, rfl⟩

end hexagon_diagonal_length_proof_l801_801997


namespace shorter_train_length_l801_801623

noncomputable def speed1 := 60 -- km/hr
noncomputable def speed2 := 40 -- km/hr
noncomputable def longer_train_length := 170 -- meters
noncomputable def crossing_time := 11.159107271418288 -- seconds

def relative_speed (s1 s2 : Float) : Float := (s1 + s2) * 1000 / 3600
def distance_covered (v : Float) (t : Float) : Float := v * t

theorem shorter_train_length :
  ∀ L : Float,
  let v_rel := relative_speed speed1 speed2
  let dist := distance_covered v_rel crossing_time
  dist = longer_train_length + L →
  L = 140 := by
  intros L v_rel dist h
  sorry

end shorter_train_length_l801_801623


namespace correct_description_of_line_l801_801633

theorem correct_description_of_line :
  let line := λ x : ℝ, - (1 / 2) * x - 1 in
  (∃ y : ℝ, y = line 0 ∧ y = -1) :=
by
  let line := λ x : ℝ, - (1 / 2) * x - 1
  have intersection_y_axis := line 0
  show ∃ y : ℝ, y = intersection_y_axis ∧ y = -1
  -- Proof would follow here
  sorry

end correct_description_of_line_l801_801633


namespace minimum_shift_a_l801_801014

noncomputable def cosine_function (x : ℝ) : ℝ :=
  cos x * cos (x + (real.pi / 6))

theorem minimum_shift_a (a : ℝ) (h_a : a > 0) :
  (∀ x : ℝ, cosine_function (x + a) = cosine_function (-(x + a))) → a = real.pi * (5 / 12) :=
by
  sorry

end minimum_shift_a_l801_801014


namespace eccentricity_of_ellipse_l801_801584

-- Definitions and conditions
def a_squared : ℝ := 4
def b_squared : ℝ := 1
def c_squared : ℝ := a_squared - b_squared
def a : ℝ := Real.sqrt a_squared
def c : ℝ := Real.sqrt c_squared

-- Statement of the theorem
theorem eccentricity_of_ellipse : 
  c / a = sqrt 3 / 2 :=
by
  -- Proof would go here
  sorry

end eccentricity_of_ellipse_l801_801584


namespace classification_of_square_and_cube_roots_l801_801632

-- Define the three cases: positive, zero, and negative
inductive NumberCase
| positive 
| zero 
| negative 

-- Define the concept of "classification and discussion thinking"
def is_classification_and_discussion_thinking (cases : List NumberCase) : Prop :=
  cases = [NumberCase.positive, NumberCase.zero, NumberCase.negative]

-- The main statement to be proven
theorem classification_of_square_and_cube_roots :
  is_classification_and_discussion_thinking [NumberCase.positive, NumberCase.zero, NumberCase.negative] :=
by
  sorry

end classification_of_square_and_cube_roots_l801_801632


namespace find_a_value_l801_801589

variable {a : ℝ} (f : ℝ → ℝ)
def is_maximum_value_in_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f = λ x, a * x^2 + (2 * a - 1) * x - 3 ∧ 
  (∃ x_max ∈ Icc (-(3 : ℝ)/2) (2 : ℝ), f x_max = 1) ∧
  (∀ x ∈ Icc (-(3 : ℝ)/2) (2 : ℝ), f x ≤ 1)

theorem find_a_value : is_maximum_value_in_interval (λ x, a * x^2 + (2 * a - 1) * x - 3) a →
  (a = 3 / 4 ∨ a = (-3 - 2 * real.sqrt 2) / 2) :=
sorry

end find_a_value_l801_801589


namespace equal_squares_l801_801469

-- Define a square structure and points on the sides
structure square (A B C D K L M N : Type) :=
  (AB BC CD DA : ℝ)
  (angle_KLA angle_LAM angle_AMN : ℝ)
  (angle_condition : angle_KLA = 45 ∧ angle_LAM = 45 ∧ angle_AMN = 45)

-- Define our specific square ABCD and points K, L, M, N with the angles conditions
def ABCD_square (A B C D K L M N : Type) [square A B C D K L M N] : Prop := 
  ∃ (s : ℝ), (AB A B = s ∧ BC B C = s ∧ CD C D = s ∧ DA D A = s) ∧ 
             (angle_KLA A B C D K L M N = 45 ∧ 
              angle_LAM A B C D K L M N = 45 ∧ 
              angle_AMN A B C D K L M N = 45)

-- Theorem to prove the desired equality
theorem equal_squares {A B C D K L M N : Type} 
  [sqr: square A B C D K L M N] : 
  (sqr.angle_KLA = 45 ∧ sqr.angle_LAM = 45 ∧ sqr.angle_AMN = 45) →
  (ABCD_square A B C D K L M N) →
  KL^2 + AM^2 = LA^2 + MN^2 :=
sorry

end equal_squares_l801_801469


namespace harold_monthly_income_l801_801071

variable (M : ℕ)

def rent : ℕ := 700
def car_payment : ℕ := 300
def utilities : ℕ := car_payment / 2
def groceries : ℕ := 50

def total_expenses : ℕ := rent + car_payment + utilities + groceries
def remaining_money_after_expenses : ℕ := M - total_expenses
def retirement_saving_target : ℕ := 650
def required_remaining_money_pre_saving : ℕ := 2 * retirement_saving_target

theorem harold_monthly_income :
  remaining_money_after_expenses = required_remaining_money_pre_saving → M = 2500 :=
by
  sorry

end harold_monthly_income_l801_801071


namespace simplify_f_find_value_f_l801_801564

variables (α : Real) -- α is an angle

noncomputable def f (α : Real) : Real :=
  (sin (α - Real.pi / 2) * cos (3 * Real.pi / 2 + α) * tan (Real.pi - α)) / 
  (tan (-Real.pi - α) * sin (-Real.pi - α))

-- Proof for simplified form of f(α)
theorem simplify_f (hα : α > Real.pi ∧ α < 3 * Real.pi / 2) : f α = -cos α := 
by sorry

-- Proof for the specific value of f(α) when cos(α - 3π/2) = 1/5
theorem find_value_f (hα : α > Real.pi ∧ α < 3 * Real.pi / 2) 
  (h_cos_eq : cos (α - 3 * Real.pi / 2) = 1 / 5) : f α = 2 * Real.sqrt 6 / 5 := 
by sorry

end simplify_f_find_value_f_l801_801564


namespace total_practice_hours_l801_801667

def schedule : List ℕ := [6, 4, 5, 7, 3]

-- We define the conditions
def total_scheduled_hours : ℕ := schedule.sum

def average_daily_practice_time (total : ℕ) : ℕ := total / schedule.length

def rainy_day_lost_hours : ℕ := average_daily_practice_time total_scheduled_hours

def player_A_missed_hours : ℕ := 2

def player_B_missed_hours : ℕ := 3

def total_missed_hours : ℕ := player_A_missed_hours + player_B_missed_hours

def total_hours_practiced : ℕ := total_scheduled_hours - (rainy_day_lost_hours + total_missed_hours)

-- Now we state the theorem we want to prove
theorem total_practice_hours : total_hours_practiced = 15 := by
  -- omitted proof
  sorry

end total_practice_hours_l801_801667


namespace find_tangent_line_l801_801918

-- Define the function
def f (x : ℝ) : ℝ := (Real.exp x) / (x + 1)

-- Define the point of tangency
def P : ℝ × ℝ := (1, Real.exp 1 / 2)

-- Define the tangent line equation
def tangent_line (x : ℝ) : ℝ := (Real.exp 1 / 4) * x + (Real.exp 1 / 4)

-- The main theorem stating the equation of the tangent line
theorem find_tangent_line :
  ∀ (x y : ℝ), P = (1, y) → y = f 1 → tangent_line x = (Real.exp 1 / 4) * x + (Real.exp 1 / 4) := by
  sorry

end find_tangent_line_l801_801918


namespace count_ordered_7_tuples_l801_801360

theorem count_ordered_7_tuples :
  let exists_tuple (a : Fin 7 → ℕ) : Prop :=
    ∀ n : Fin 5, a n + a (n + 1) = a (n + 2) ∧ a ⟨5, sorry⟩ = 2005
  in 
  ∃ (S : Set (Fin 7 → ℕ)), (∀ a ∈ S, exists_tuple a) ∧ S.card = 133 :=
by
  sorry

end count_ordered_7_tuples_l801_801360


namespace find_real_solutions_l801_801968

theorem find_real_solutions (a : ℝ) (n : ℕ) (x : ℕ → ℝ)
  (h1 : ∀ i, 1 ≤ i ∧ i < n → x (i + 1) = (1 / 2) * (x i + a / x i))
  (h2 : x 1 = (1 / 2) * (x n + a / x n)) :
  (∀ i, 1 ≤ i ∧ i ≤ n → x i = sqrt a) ∨ (∀ i, 1 ≤ i ∧ i ≤ n → x i = -sqrt a) :=
sorry

end find_real_solutions_l801_801968


namespace lattice_triangle_area_ge_half_l801_801244

structure Point3D := 
  (x : ℤ)
  (y : ℤ)
  (z : ℤ)

noncomputable def triangle_area(lower : Point3D) (A B C : Point3D) : ℝ := sorry

theorem lattice_triangle_area_ge_half
  (A B C : Point3D) :
  triangle_area A B C ≥ (1 : ℝ) / 2 :=
sorry

end lattice_triangle_area_ge_half_l801_801244


namespace P_moves_perpendicular_to_AB_l801_801517

/-- Given triangle PAB with point P moving perpendicular to side AB,
    and M and N as midpoints of PA and PB respectively, prove that
    exactly three of the following quantities change as P moves:
    (a) length of segment MN
    (b) perimeter of triangle PAB
    (c) area of triangle PAB
    (d) length of diagonal PN in quadrilateral PABM. -/
theorem P_moves_perpendicular_to_AB :
  let P A B M N : Type := sorry, /-- Assume necessary geometric types -/
  ∀ (P A B M N : Type)
    (isMidpoint : M = (P + A) / 2 ∧ N = (P + B) / 2)
    (perpendicular : ∀ (P : Type), -- Assume this definition is well-formed
      P ∉ (line_through A B))
    (MN_length : length (segment M N))
    (perimeter_ΔPAB : perimeter (triangle P A B))
    (area_ΔPAB : area (triangle P A B))
    (PN_length : length (diagonal P N in quadrilateral P A B M)),
  exactly_three_change_as_P_moves P A B M N
  ∧ (MN_length does_not_change)
  ∧ (perimeter_ΔPAB changes)
  ∧ (area_ΔPAB changes)
  ∧ (PN_length changes) 
  := sorry

end P_moves_perpendicular_to_AB_l801_801517


namespace general_term_formula_l801_801224

-- Conditions: sequence \(\frac{1}{2}\), \(\frac{1}{3}\), \(\frac{1}{4}\), \(\frac{1}{5}, \ldots\)
-- Let seq be the sequence in question.

def seq (n : ℕ) : ℚ := 1 / (n + 1)

-- Question: prove the general term formula is \(\frac{1}{n+1}\)
theorem general_term_formula (n : ℕ) : seq n = 1 / (n + 1) :=
by
  -- Proof goes here
  sorry

end general_term_formula_l801_801224


namespace probability_sum_less_than_10_l801_801251

-- Two fair, six-sided dice are rolled.
def fair_die := fin 6

-- Each die has 6 faces.
def num_faces_die : ℕ := 6

-- Total number of outcomes when two dice are rolled.
def total_outcomes : ℕ := num_faces_die * num_faces_die

-- The successful outcomes for the sum being less than 10
def successful_outcomes : set (fair_die × fair_die) :=
  { (d1, d2) | d1.1 + d2.1 + 2 < 10 }  -- Adjust for 1-based indexing of dice faces

-- Probability calculation
def probability (s : set (fair_die × fair_die)) : ℚ :=
  (finset.card s) / (total_outcomes)

theorem probability_sum_less_than_10 :
  probability successful_outcomes = 5 / 6 := by
  sorry

end probability_sum_less_than_10_l801_801251


namespace range_of_k_l801_801018

theorem range_of_k (k : ℝ) :
  (2 ≤ ∫ x in (1 : ℝ) .. 2, k + 1) → (∫ x in (1 : ℝ) .. 2, k + 1 ≤ 4) → 1 ≤ k ∧ k ≤ 3 :=
by
  intros
  sorry

end range_of_k_l801_801018


namespace intersection_of_sets_l801_801414

open Set

variable (U : Set ℝ)
variable (A : Set ℝ)
variable (B : Set ℝ)

def complement (U : Set ℝ) (B : Set ℝ) := { x | x ∈ U ∧ x ∉ B }

theorem intersection_of_sets :
  let U := set.univ;
  let A := { x : ℝ | x^2 - 3 * x < 0 };
  let B := { x : ℝ | x > 2 };
  let C_U_B := complement U B;
  A ∩ C_U_B = { x | 0 < x ∧ x ≤ 2 } :=
by
  let U := set.univ;
  let A := { x | x^2 - 3 * x < 0 };
  let B := { x | x > 2 };
  let C_U_B := complement U B;
  have h : A ∩ C_U_B = { x | 0 < x ∧ x ≤ 2 }, from sorry;
  exact h

end intersection_of_sets_l801_801414


namespace ordered_pairs_unique_solution_l801_801361

theorem ordered_pairs_unique_solution :
  ∃! (x y : ℝ), 9^(x^2 + y) + 9^(x + y^2) = Real.sqrt 2 :=
by
  sorry

end ordered_pairs_unique_solution_l801_801361


namespace triangle_inscribed_relation_l801_801268

noncomputable def herons_area (p a b c : ℝ) : ℝ := (p * (p - a) * (p - b) * (p - c)).sqrt

theorem triangle_inscribed_relation
  (S S' p p' : ℝ)
  (a b c a' b' c' r : ℝ)
  (h1 : r = S / p)
  (h2 : r = S' / p')
  (h3 : S = herons_area p a b c)
  (h4 : S' = herons_area p' a' b' c') :
  (p - a) * (p - b) * (p - c) / p = (p' - a') * (p' - b') * (p' - c') / p' :=
by sorry

end triangle_inscribed_relation_l801_801268


namespace triangle_area_eq_40sqrt3_l801_801027

theorem triangle_area_eq_40sqrt3 (A B C : Point) (r : ℝ)
  (hA : ∠BAC = π / 3)
  (hAB_AC : ∥A - B∥ / ∥A - C∥ = 8 / 5)
  (hr_inradius : r = 2 * sqrt 3) :
  let AB := ∥A - B∥, AC := ∥A - C∥, BC := ∥B - C∥,
      s := (AB + AC + BC) / 2,
      Δ₁ := sqrt (s * (s - AB) * (s - AC) * (s - BC)),
      Δ₂ := s * r
  in Δ₁ = 40 * sqrt 3 ∧ Δ₂ = 40 * sqrt 3 :=
sorry

end triangle_area_eq_40sqrt3_l801_801027


namespace obtain_2020_from_20_and_21_l801_801601

theorem obtain_2020_from_20_and_21 :
  ∃ (a b : ℕ), 20 * a + 21 * b = 2020 :=
by
  -- We only need to construct the proof goal, leaving the proof itself out.
  sorry

end obtain_2020_from_20_and_21_l801_801601


namespace find_n_l801_801349

theorem find_n (n : ℕ) (h : 2^6 * 3^3 * n = nat.factorial 10) : n = 2100 :=
sorry

end find_n_l801_801349


namespace vector_Q_determination_l801_801122

open EuclideanGeometry

variables {A B C G H Q : Point}
variables {a b c x y z : ℝ}
variables [field ℝ] [add_comm_group (ℝ × ℝ)] [module ℝ (ℝ × ℝ)]

/--
  conditions: In triangle ABC, point G lies on line AC such that AG:GC = 3:2. Point H lies on line AB such that
  AH:HB = 3:1. Q is the intersection of BG and CH.
  goal: Determine the vector Q in terms of A, B, and C, where Q = x A + y B + z C and x + y + z = 1.
-/
theorem vector_Q_determination (ha : a = (3:ℝ)/(3 + 2)) (hb : b = (3:ℝ)/(3 + 1))
  (vG : G = a • A + (1 - a) • C) (vH : H = b • A + (1 - b) • B) 
  (vQ : Q = x • A + y • B + z • C) (hxyz : x + y + z = 1) :
  Q = (1/7) • A + (3/7) • B + (3/14) • C :=
sorry

end vector_Q_determination_l801_801122


namespace tan_sum_simplification_l801_801553
-- We start by importing the relevant Lean libraries that contain trigonometric functions and basic real analysis.

-- Define the statement to be proved in Lean.
theorem tan_sum_simplification :
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12) = 4 * Real.sqrt 2 - 4) :=
by
  sorry

end tan_sum_simplification_l801_801553


namespace conclusion_from_true_and_false_statements_l801_801899

theorem conclusion_from_true_and_false_statements :
  (5 = 5 → 25 = 25) ∧ (5 = -5 → 25 = 25) :=
by
  -- True statement case
  have h_true: 5 = 5 := sorry
  have conclude_from_true := calc 
    (5:ℤ) = 5: from h_true
    (5:ℤ)^2 = 5^2: by rw [h_true]
    (25:ℤ) = 25: by norm_num
  
  -- False statement case
  have h_false: 5 = -5 := sorry
  have conclude_from_false := calc
    (5:ℤ) = -5: from h_false
    (5:ℤ)^2 = (-5)^2: by rw [h_false]
    (25:ℤ) = 25: by norm_num
  
  exact  and.intro conclude_from_true conclude_from_false
  sorry

end conclusion_from_true_and_false_statements_l801_801899


namespace steel_bar_length_l801_801321

-- Defining the conditions
def billet_length : ℝ := 12.56
def billet_width : ℝ := 5
def billet_height : ℝ := 4
def cylinder_diameter : ℝ := 4

-- Defining the volume calculation for the billet
def billet_volume : ℝ := billet_length * billet_width * billet_height

-- Defining the radius of the cylinder
def cylinder_radius : ℝ := cylinder_diameter / 2

-- Defining the area of the base of the cylinder
def cylinder_base_area : ℝ := Float.pi * (cylinder_radius ^ 2)

-- Defining the expected length (height) of the cylinder
def expected_cylinder_length : ℝ := billet_volume / cylinder_base_area

-- The theorem to be proved
theorem steel_bar_length : expected_cylinder_length = 20 := by
  sorry

end steel_bar_length_l801_801321


namespace find_radius_l801_801395

noncomputable def radius_of_tangent_circle (r : ℝ) : Prop := 
  (r > 0) ∧ (∀ x y : ℝ, (3*x - 4*y + 20 = 0) → (x^2 + y^2 = r^2) → r = 4)

theorem find_radius : ∀ r : ℝ, radius_of_tangent_circle r → r = 4 :=
by
  intro r h
  cases h with r_pos tangent_condition
  let x := 0
  let y := 0
  have line_tangent := tangent_condition x y
  have equation_from_line : 3*x - 4*y + 20 = 0 := by sorry
  have equation_from_circle : x^2 + y^2 = r^2 := by sorry
  have distance_eq_r := by sorry
  exact distance_eq_r

end find_radius_l801_801395


namespace coins_3_kopecks_l801_801945

theorem coins_3_kopecks (n m : ℕ) (coins : Fin 101 → Fin 4) :
  (∀ i j : Fin 101, coins i = 1 → coins j = 1 → i ≠ j → abs (i - j) ≥ 1) →
  (∀ i j : Fin 101, coins i = 2 → coins j = 2 → i ≠ j → abs (i - j) ≥ 2) →
  (∀ i j : Fin 101, coins i = 3 → coins j = 3 → i ≠ j → abs (i - j) ≥ 3) →
  nat.sum n (λ i => if coins i = 3 then 1 else 0) = m →
  m ∈ {25, 26} :=
by
  sorry

end coins_3_kopecks_l801_801945


namespace sum_abs_coefficients_l801_801795

theorem sum_abs_coefficients :
  let f (x : ℝ) := (1 - x) ^ 9
  let a_0 := f 0
  let a_1 := -9
  let a_2 := 36
  let a_3 := -84
  let a_4 := 126
  let a_5 := -126
  let a_6 := 84
  let a_7 := -36
  let a_8 := 9
  let a_9 := -1 in
  |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| + |a_8| + |a_9| = 511 := by
{
  sorry
}

end sum_abs_coefficients_l801_801795


namespace percentage_of_female_students_l801_801104

theorem percentage_of_female_students {F : ℝ} (h1 : 200 > 0): ((200 * (F / 100)) * 0.5 * 0.5 = 30) → (F = 60) :=
by
  sorry

end percentage_of_female_students_l801_801104


namespace graph_passes_through_point_l801_801591

theorem graph_passes_through_point (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) : ∃ y, y = a^(1 - 1) + 2 ∧ y = 3 :=
by
  use 3
  have h: a^(1 - 1) = 1, from pow_zero a
  simp at h
  exact Eq.trans (by rw h) (by norm_num)
  done

end graph_passes_through_point_l801_801591


namespace solve_quadratic_1_solve_quadratic_2_l801_801190

theorem solve_quadratic_1 (x : ℝ) : 2 * x^2 - 7 * x - 1 = 0 ↔ 
  (x = (7 + Real.sqrt 57) / 4 ∨ x = (7 - Real.sqrt 57) / 4) := 
by 
  sorry

theorem solve_quadratic_2 (x : ℝ) : (2 * x - 3)^2 = 10 * x - 15 ↔ 
  (x = 3 / 2 ∨ x = 4) := 
by 
  sorry

end solve_quadratic_1_solve_quadratic_2_l801_801190


namespace vector_parallel_l801_801417

theorem vector_parallel (m : ℝ) :
  let a := (2, -3)
  let b := (m, 6)
  (b.1 * a.2 - a.1 * b.2 = 0) ↔ (m = -4) := 
by
  intros a b
  unfold a b
  simp [a, b]
  sorry

end vector_parallel_l801_801417


namespace find_cheap_coat_cost_l801_801839

-- Definitions based on the given conditions
def cost_expensive_coat : ℕ := 300
def years_expensive_coat_lasts : ℕ := 15
def years_cheap_coat_lasts : ℕ := 5
def savings_over_30_years : ℕ := 120
def years_of_consideration : ℕ := 30

-- We need to find the cost of the cheap coat
noncomputable def cost_cheap_coat : ℕ := 720 / 6

theorem find_cheap_coat_cost :
  (2 * cost_expensive_coat + savings_over_30_years = 6 * cost_cheap_coat) →
  cost_cheap_coat = 120 :=
by
  simp [cost_expensive_coat, savings_over_30_years, cost_cheap_coat]
  intro h
  exact h

#eval find_cheap_coat_cost sorry

end find_cheap_coat_cost_l801_801839


namespace vector_magnitude_l801_801906

noncomputable def a := (2:ℝ, 0:ℝ)
noncomputable def b_norm := 1
noncomputable def θ := Real.pi / 3 -- 60 degrees in radians
noncomputable def cos_θ := Real.cos θ -- should simplify to 1/2

theorem vector_magnitude :
  let b := (1:ℝ, Real.sin θ) in -- because vector b has norm 1 and angle θ with a
  ‖(a.1 + 2*b.1, a.2 + 2*b.2)‖ = 2 * Real.sqrt 3 :=
by
  sorry

end vector_magnitude_l801_801906


namespace mr_arevalo_change_l801_801196

-- Definitions for the costs of the food items
def cost_smoky_salmon : ℤ := 40
def cost_black_burger : ℤ := 15
def cost_chicken_katsu : ℤ := 25

-- Definitions for the service charge and tip percentages
def service_charge_percent : ℝ := 0.10
def tip_percent : ℝ := 0.05

-- Definition for the amount Mr. Arevalo pays
def amount_paid : ℤ := 100

-- Calculation for total food cost
def total_food_cost : ℤ := cost_smoky_salmon + cost_black_burger + cost_chicken_katsu

-- Calculation for service charge
def service_charge : ℝ := service_charge_percent * total_food_cost

-- Calculation for tip
def tip : ℝ := tip_percent * total_food_cost

-- Calculation for the final bill amount
def final_bill_amount : ℝ := total_food_cost + service_charge + tip

-- Calculation for the change
def change : ℝ := amount_paid - final_bill_amount

-- Proof statement
theorem mr_arevalo_change : change = 8 := by
  sorry

end mr_arevalo_change_l801_801196


namespace first_car_departure_time_l801_801312

variable (leave_time : Nat) -- in minutes past 8:00 am

def speed : Nat := 60 -- km/h
def firstCarTimeAt32 : Nat := 32 -- minutes since 8:00 am
def secondCarFactorAt32 : Nat := 3
def firstCarTimeAt39 : Nat := 39 -- minutes since 8:00 am
def secondCarFactorAt39 : Nat := 2

theorem first_car_departure_time :
  let firstCarSpeed := (60 / 60 : Nat) -- km/min
  let d1_32 := firstCarSpeed * firstCarTimeAt32
  let d2_32 := firstCarSpeed * (firstCarTimeAt32 - leave_time)
  let d1_39 := firstCarSpeed * firstCarTimeAt39
  let d2_39 := firstCarSpeed * (firstCarTimeAt39 - leave_time)
  d1_32 = secondCarFactorAt32 * d2_32 →
  d1_39 = secondCarFactorAt39 * d2_39 →
  leave_time = 11 :=
by
  intros h1 h2
  sorry

end first_car_departure_time_l801_801312


namespace pablo_puzzle_l801_801878

open Nat

theorem pablo_puzzle (pieces_per_hour : ℕ) (hours_per_day : ℕ) (days : ℕ) 
    (pieces_per_five_puzzles : ℕ) (num_five_puzzles : ℕ) (total_pieces : ℕ) 
    (num_eight_puzzles : ℕ) :

    pieces_per_hour = 100 →
    hours_per_day = 7 →
    days = 7 →
    pieces_per_five_puzzles = 500 →
    num_five_puzzles = 5 →
    num_eight_puzzles = 8 →
    total_pieces = (pieces_per_hour * hours_per_day * days) →
    num_eight_puzzles * (total_pieces - num_five_puzzles * pieces_per_five_puzzles) / num_eight_puzzles = 300 :=
by
  intros
  sorry

end pablo_puzzle_l801_801878


namespace equilateral_triangle_area_third_l801_801461

theorem equilateral_triangle_area_third 
  (ABC : Triangle) 
  (acute_ABC : ABC.acute) 
  (angle_A_eq_pi_div_3 : ABC.angleA = π / 3)
  (G : Set Point) 
  (P_inside_ABC : ∀ P ∈ G, ABC.contains P) 
  (PA_leq_PB : ∀ P ∈ G, dist P ABC.A ≤ dist P ABC.B) 
  (PA_leq_PC : ∀ P ∈ G, dist P ABC.A ≤ dist P ABC.C) 
  (area_G_eq_one_third_area_ABC : ABC.area / 3 = region_area G) :
  ABC.equilateral :=
sorry

end equilateral_triangle_area_third_l801_801461


namespace no_integer_roots_l801_801501

theorem no_integer_roots (n : ℕ) (p : Fin (2*n + 1) → ℤ)
  (non_zero : ∀ i, p i ≠ 0)
  (sum_non_zero : (Finset.univ.sum (λ i => p i)) ≠ 0) :
  ∃ P : ℤ → ℤ, ∀ x : ℤ, P x ≠ 0 → x > 1 ∨ x < -1 := sorry

end no_integer_roots_l801_801501


namespace find_function_equation_l801_801912

theorem find_function_equation :
  ∃ (k b : ℝ), k > 0 ∧ (∀ x, x ∈ set.Icc 0 1 → f x = k * x + b) ∧
  f 0 = -1 ∧ f 1 = 1 ∧ (∀ x, f x ∈ set.Icc (-1) 1)  → f(x) = 2 * x - 1 :=
by
  have h1 : f 0 = -1 := sorry
  have h2 : f 1 = 1 := sorry
  have h3 : f x = k * x + b := sorry
  use 2, -1
  split,
  exact sorry,
  split,
  exact sorry,
  split,
  sorry,
  sorry,
  sorry

end find_function_equation_l801_801912


namespace count_pairs_200_l801_801012

def count_valid_pairs (n : ℕ) : ℕ :=
  ∑ y in Finset.range (n - 1), (n - y - 1) / ((y + 1) * (y + 2) * (y + 3))

theorem count_pairs_200 : count_valid_pairs 200 = 78 := 
by
    sorry

end count_pairs_200_l801_801012


namespace proof_problem_l801_801705

-- Define the operation ♠ as a simple function
def spadesuit (a b : ℕ) : ℕ := abs (a - b)

-- State the property we want to prove
theorem proof_problem :
  (spadesuit (spadesuit 3 5) (spadesuit 2 (spadesuit 1 6)) = 1) :=
by
  sorry

end proof_problem_l801_801705


namespace find_n_l801_801341

theorem find_n (n : ℕ) (h : 2^6 * 3^3 * n = Nat.factorial 10) : n = 2100 :=
by
sorry

end find_n_l801_801341


namespace mean_noon_temperature_l801_801929

theorem mean_noon_temperature : 
  let temperatures := [75, 80, 78, 82, 85, 90, 87, 84, 88, 93] in
  (∑ temp in temperatures, temp) / (temperatures.length : ℝ) = (421 : ℝ) / 5 :=
by
  let temperatures : List ℝ := [75, 80, 78, 82, 85, 90, 87, 84, 88, 93]
  have h1 : ∑ temp in temperatures, temp = 842 := by sorry  -- Proof to be filled in
  have h2 : temperatures.length = 10 := by sorry  -- Proof to be filled in
  rw [h1, h2]
  norm_num
  -- Convert 842 / 10 to 421 / 5
  exact_rat_cast_to (421 : ℝ) / 5
  sorry

end mean_noon_temperature_l801_801929


namespace solve_equation_l801_801189

theorem solve_equation : 
  (x : ℝ) (h : (x^2 + 3 * x + 5) / (x + 6) = x + 7) → x = -37 / 10 := 
sorry

end solve_equation_l801_801189


namespace symmetric_scanning_codes_count_l801_801677

noncomputable section

-- Definition of symmetric scanning code conditions
def is_symmetric (grid : Fin 8 → Fin 8 → Bool) : Prop :=
  ∀ i j, grid i j = grid j (7 - i) ∧ grid i j = grid (7 - i) (7 - j) ∧ grid i j = grid (7 - j) i

-- The main theorem
theorem symmetric_scanning_codes_count : 
  (∀ grid : Fin 8 → Fin 8 → Bool, (∃ i j, grid i j = true) ∧ (∃ i j, grid i j = false) → is_symmetric grid → 2046) :=
sorry

end symmetric_scanning_codes_count_l801_801677


namespace remainder_when_divided_by_3_l801_801374

theorem remainder_when_divided_by_3 
  (a : ℕ → ℤ)
  (h : (2 * x + 4)^2010 = ∑ i in finset.range (2010 + 1), a i * x^i) :
  (∑ i in finset.range (2010 + 1), if i % 2 = 0 then a i else 0) % 3 = 2 :=
sorry

end remainder_when_divided_by_3_l801_801374


namespace prove_f_eight_minus_a_l801_801222

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 10^(1 - x) + 1 else Real.log10 (x + 2)

theorem prove_f_eight_minus_a :
  ∀ a : ℝ, f a = 1 → f (8 - a) = 11 :=
by
  intro a ha
  sorry

end prove_f_eight_minus_a_l801_801222


namespace quadratic_roots_difference_l801_801326

theorem quadratic_roots_difference (p q : ℝ) (hp : 0 < p) (hq : 0 < q) 
  (h_diff : ∀ (r₁ r₂ : ℝ), r₁ ≠ r₂ → (r₁ = (-p + sqrt(p^2 + 4 * q)) / 2) 
  ∧ (r₂ = (-p - sqrt(p^2 + 4 * q)) / 2) → |r₁ - r₂| = 2) : 
  p = sqrt(4 - 4 * q) :=
by
  sorry

end quadratic_roots_difference_l801_801326


namespace jimmy_paid_amount_l801_801478

theorem jimmy_paid_amount
  (pens_price : ℕ)
  (notebooks_price : ℕ)
  (folders_price : ℕ)
  (pens_bought : ℕ)
  (notebooks_bought : ℕ)
  (folders_bought : ℕ)
  (change_received : ℕ) :
  pens_price = 1 →
  notebooks_price = 3 →
  folders_price = 5 →
  pens_bought = 3 →
  notebooks_bought = 4 →
  folders_bought = 2 →
  change_received = 25 →
  let total_cost := (pens_bought * pens_price) + (notebooks_bought * notebooks_price) + (folders_bought * folders_price) in
  total_cost + change_received = 50 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  simp [h1, h2, h3, h4, h5, h6, h7]
  -- This will result in: total_cost + 25 = 50
  sorry

end jimmy_paid_amount_l801_801478


namespace find_n_l801_801345

theorem find_n (n : ℕ) (h₁ : 2^6 * 3^3 * n = factorial 10) : n = 2100 :=
sorry

end find_n_l801_801345


namespace arithmetic_mean_after_removal_l801_801907

theorem arithmetic_mean_after_removal (orig_mean : ℝ) (total_nums : ℕ) (removals : List ℝ) (new_mean : ℝ) :
  orig_mean = 45 →
  total_nums = 80 →
  removals = [50, 60, 70] →
  new_mean = (45 * 80 - list.sum removals) / (80 - removals.length) →
  new_mean = 44.4 :=
by
  sorry

end arithmetic_mean_after_removal_l801_801907


namespace product_of_integers_with_cubes_sum_189_l801_801600

theorem product_of_integers_with_cubes_sum_189 :
  ∃ a b : ℤ, a^3 + b^3 = 189 ∧ a * b = 20 :=
by
  -- The proof is omitted for brevity.
  sorry

end product_of_integers_with_cubes_sum_189_l801_801600


namespace pat_initial_stickers_l801_801874

def initial_stickers (s : ℕ) : ℕ := s  -- Number of stickers Pat had on the first day of the week

def stickers_earned : ℕ := 22  -- Stickers earned during the week

def stickers_end_week (s : ℕ) : ℕ := initial_stickers s + stickers_earned  -- Stickers at the end of the week

theorem pat_initial_stickers (s : ℕ) (h : stickers_end_week s = 61) : s = 39 :=
by
  sorry

end pat_initial_stickers_l801_801874


namespace unique_perpendicular_line_through_point_l801_801025

variable (α : Plane) (P : Point)

theorem unique_perpendicular_line_through_point : ∃! p : Line, (P ∈ p) ∧ (p ⊥ α) :=
sorry

end unique_perpendicular_line_through_point_l801_801025


namespace ratio_doctors_lawyers_l801_801455

theorem ratio_doctors_lawyers (d l : ℕ) (h1 : (45 * d + 60 * l) / (d + l) = 50) (h2 : d + l = 50) : d = 2 * l :=
by
  sorry

end ratio_doctors_lawyers_l801_801455


namespace diamond_eval_l801_801372

def diamond (x y : ℝ) : ℝ := (x^2 + y^2) / (x + y)

theorem diamond_eval {w x y z : ℝ} (hw : 0 < w) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : w + x + y + z = 10) :
  diamond (diamond (diamond w x) y) z = (w^2 + x^2 + y^2 + z^2) / 10 :=
sorry

end diamond_eval_l801_801372


namespace initial_average_weight_l801_801573

theorem initial_average_weight (a b c d e : ℝ) (A : ℝ) 
    (h1 : (a + b + c) / 3 = A) 
    (h2 : (a + b + c + d) / 4 = 80) 
    (h3 : e = d + 3) 
    (h4 : (b + c + d + e) / 4 = 79) 
    (h5 : a = 75) : A = 84 :=
sorry

end initial_average_weight_l801_801573


namespace tan_sum_pi_over_12_l801_801534

theorem tan_sum_pi_over_12 : 
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12)) = 4 := 
sorry

end tan_sum_pi_over_12_l801_801534


namespace number_of_tasty_sequences_l801_801843

-- Define what it means for a polynomial to be in a tasty sequence
def is_tasty_sequence (q : ℕ) (P : ℕ → polynomial ℤ) : Prop :=
  prime q ∧ q < 50 ∧      -- q is a prime number less than 50
  (∀ i : ℕ, i ≤ q^2 → P i .degree = i) ∧  -- P_i has degree i
  (∀ i : ℕ, i ≤ q^2 → ∀ j : ℕ, j ≤ q^2, 
    (∀ k : ℕ, P k.coeff.nat_degree ∈ set.Icc 0 (q-1))) ∧  -- coefficients of P_i are between 0 and q-1
  (∀ i : ℕ, i ≤ q^2 → ∀ j : ℕ, j ≤ q^2, 
    (P i).eval (P j) - (P j).eval (P i) ∈ ideal.span ({q} : set (polynomial ℤ)))  -- P_i(P_j(x)) - P_j(P_i(x)) has all coefficients divisible by q

-- The main theorem stating the number of tasty sequences
theorem number_of_tasty_sequences : 
  (∑ q in (finset.filter prime (finset.Ico 0 50)), 
    finset.card {P | is_tasty_sequence q P}) = 30416 := 
sorry

end number_of_tasty_sequences_l801_801843


namespace outer_boundary_diameter_l801_801980

-- Define the given conditions
def fountain_diameter : ℝ := 12
def walking_path_width : ℝ := 6
def garden_ring_width : ℝ := 10

-- Define what we need to prove
theorem outer_boundary_diameter :
  2 * (fountain_diameter / 2 + garden_ring_width + walking_path_width) = 44 :=
by
  sorry

end outer_boundary_diameter_l801_801980


namespace problem_part1_problem_part2_l801_801055

noncomputable def f (a : ℝ) (x : ℝ) := 2 * Real.log x + a / x
noncomputable def g (a : ℝ) (x : ℝ) := (x / 2) * f a x - a * x^2 - x

theorem problem_part1 (a : ℝ) :
  (∀ x : ℝ, f a x = 0 → x > 0) ↔ 0 < a ∧ a < 2/Real.exp 1 := sorry

theorem problem_part2 (a : ℝ) (x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : g a x₁ = 0) (h₃ : g a x₂ = 0) :
  0 < a ∧ a < 2/Real.exp 1 → Real.log x₁ + 2 * Real.log x₂ > 3 := sorry

end problem_part1_problem_part2_l801_801055


namespace number_of_mappings_l801_801790

theorem number_of_mappings (A : Fin 100 → ℝ) (B : Fin 50 → ℝ) (f : Fin 100 → Fin 50) 
  (hA : Set.toFinset (Set.range A) ⊇ (Finset.univ : Finset (Fin 100 → ℝ)))
  (hB : Set.toFinset (Set.range B) ⊇ (Finset.univ : Finset (Fin 50 → ℝ)))
  (hf : ∀ i j, i ≤ j → f i ≤ f j) :
  ∃ n, n = Nat.choose 99 49 := by
  sorry

end number_of_mappings_l801_801790


namespace tan_sum_pi_over_12_eq_4_l801_801544

theorem tan_sum_pi_over_12_eq_4 :
  tan (π / 12) + tan (5 * π / 12) = 4 := 
by
  have cos_pi_over_12 : cos (π / 12) = (real.sqrt 6 + real.sqrt 2) / 4 := sorry
  have cos_5pi_over_12 : cos (5 * π / 12) = (real.sqrt 6 - real.sqrt 2) / 4 := sorry
  have sin_pi_over_2 : sin (π / 2) = 1 := by
    exact real.sin_pi_div_two
  sorry

end tan_sum_pi_over_12_eq_4_l801_801544


namespace nth_term_is_2n_sum_bn_l801_801781

-- Condition 1: Sequence with sum formula
def S (n k : ℕ) := n^2 + k * n

-- Condition 2: Minimum value of S_n - 5kn is -4, implicating k = 1
axiom min_value_cond (n : ℕ) : S n 1 - 5 * 1 * n = -4 ↔ n = 2

-- Question 1: Prove the nth term a_n = 2n for k = 1 and S_n defined as above
theorem nth_term_is_2n (n : ℕ) : 
  let a : ℕ → ℕ := λ n, 2 * n in
  a n = 2 * n :=
by sorry

-- Definition of sequence b_n
def b (n : ℕ) := (2 * n) / (4^n)

-- Question 2: Prove the sum of the first n terms of sequence b_n
theorem sum_bn (n : ℕ) : 
  let T := Nat.sum (Finset.range n) (λ i, b (i + 1)) in
  T = 8 / 9 - (8 + 6 * n) / (9 * 4^n) :=
by sorry

end nth_term_is_2n_sum_bn_l801_801781


namespace find_tangent_line_l801_801917

-- Define the function
def f (x : ℝ) : ℝ := (Real.exp x) / (x + 1)

-- Define the point of tangency
def P : ℝ × ℝ := (1, Real.exp 1 / 2)

-- Define the tangent line equation
def tangent_line (x : ℝ) : ℝ := (Real.exp 1 / 4) * x + (Real.exp 1 / 4)

-- The main theorem stating the equation of the tangent line
theorem find_tangent_line :
  ∀ (x y : ℝ), P = (1, y) → y = f 1 → tangent_line x = (Real.exp 1 / 4) * x + (Real.exp 1 / 4) := by
  sorry

end find_tangent_line_l801_801917


namespace sum_of_roots_poly1_sum_of_roots_poly2_sum_of_roots_combined_l801_801364

def poly1 := 3 * x^3 - 9 * x^2 + 12 * x - 4
def poly2 := 4 * x^3 + 2 * x^2 - 3 * x + 1

noncomputable def sum_of_roots (p : Polynomial ℝ) : ℝ :=
  -((Polynomial.coeff p 2) / (Polynomial.coeff p 3))

theorem sum_of_roots_poly1 : sum_of_roots poly1 = 3 :=
by
  sorry

theorem sum_of_roots_poly2 : sum_of_roots poly2 = -0.5 :=
by
  sorry

theorem sum_of_roots_combined : sum_of_roots poly1 + sum_of_roots poly2 = 2.5 :=
by
  exact (sum_of_roots_poly1) + (sum_of_roots_poly2)

end sum_of_roots_poly1_sum_of_roots_poly2_sum_of_roots_combined_l801_801364


namespace solve_birthday_problem_l801_801137

variables (Joey Chloe Max : ℕ) (n : ℕ)
def next_multiple_of (a b : ℕ) : ℕ := sorry  -- This is a placeholder for the actual function that determines the next multiple

noncomputable def birthday_problem : Prop :=
  (Joey > Chloe) ∧ (Joey = Chloe + 2) ∧ (Max = 3) ∧ 
  (∃ k : ℕ, (0 ≤ k < 12 ∧ ∀ i : ℕ, (i ≤ k → (Chloe + i) % (Max + i) = 0))) ∧
  (let J' := next_multiple_of Joey (Max + k) in J' = 93 ∧ (J' / 10 + J' % 10) = 12)

theorem solve_birthday_problem : birthday_problem Joey Chloe Max n :=
sorry

end solve_birthday_problem_l801_801137


namespace solution_set_f_less_exp_l801_801757

noncomputable def f : ℝ → ℝ := sorry  -- placeholder for a differentiable function
axiom f_diff : differentiable ℝ f
axiom f_prime_less_than_f : ∀ x, deriv f x < f x
axiom f_symmetry : ∀ x, f (-x) = f (2 + x)
axiom f_at_2 : f 2 = 1

theorem solution_set_f_less_exp :
  {x | f x < real.exp x} = set.Ioi 0 := 
sorry

end solution_set_f_less_exp_l801_801757


namespace basketball_team_lineup_l801_801986

theorem basketball_team_lineup (n k : ℕ) (twin1 twin2 : ℕ)
  (h_twin1_twin2 : twin1 ≠ twin2) (h_twin1_le_n : twin1 ≤ n) (h_twin2_le_n : twin2 ≤ n) 
  (h_n_eq_12 : n = 12) (h_k_eq_5 : k = 5) : ∑ i in {twin1, twin2}, ∑ j in finset.range (n - 1) choose (k - 1) = 660 := 
by
  sorry

end basketball_team_lineup_l801_801986


namespace feed_days_l801_801136

theorem feed_days (morning_food evening_food total_food : ℕ) (h1 : morning_food = 1) (h2 : evening_food = 1) (h3 : total_food = 32)
: (total_food / (morning_food + evening_food)) = 16 := by
  sorry

end feed_days_l801_801136


namespace false_statement_A_true_statement_B_true_statement_C_false_statement_D_l801_801009

theorem false_statement_A (a b c : ℝ) (h1 : a > b) (h2 : c ≠ 0) (h3 : c < 0) : ¬ (ac > bc) :=
by sorry

theorem true_statement_B (a b c : ℝ) (h1 : ac^2 > bc^2) (h2 : c ≠ 0) : a > b :=
by sorry

theorem true_statement_C (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > ab ∧ ab > b^2 :=
by sorry

theorem false_statement_D (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : c < 0) (h5 : d < 0) : ¬ (ac > bd) :=
by sorry

end false_statement_A_true_statement_B_true_statement_C_false_statement_D_l801_801009


namespace circumradius_relationship_l801_801265

theorem circumradius_relationship 
  (a b c a' b' c' R : ℝ)
  (S S' p p' : ℝ)
  (h₁ : R = (a * b * c) / (4 * S))
  (h₂ : R = (a' * b' * c') / (4 * S'))
  (h₃ : S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (h₄ : S' = Real.sqrt (p' * (p' - a') * (p' - b') * (p' - c')))
  (h₅ : p = (a + b + c) / 2)
  (h₆ : p' = (a' + b' + c') / 2) :
  (a * b * c) / Real.sqrt (p * (p - a) * (p - b) * (p - c)) = 
  (a' * b' * c') / Real.sqrt (p' * (p' - a') * (p' - b') * (p' - c')) :=
by 
  sorry

end circumradius_relationship_l801_801265


namespace cost_per_page_proof_l801_801127

-- Defining parameters based on the problem's conditions
def num_notebooks : ℕ := 2
def pages_per_notebook : ℕ := 50
def unit_cost_dollars : ℝ := 40
def discount_rate : ℝ := 0.15
def sales_tax_rate : ℝ := 0.05
def exchange_rate : ℝ := 2.3

-- Calculating derived values
def total_pages := num_notebooks * pages_per_notebook
def discounted_cost := unit_cost_dollars * (1 - discount_rate)
def total_cost_with_tax := discounted_cost * (1 + sales_tax_rate)
def total_cost_local_currency := total_cost_with_tax * exchange_rate
def cost_per_page_local_currency := total_cost_local_currency / total_pages

-- Stating the theorem
theorem cost_per_page_proof : cost_per_page_local_currency = 0.8211 := by
  sorry

end cost_per_page_proof_l801_801127


namespace number_pattern_div_99_l801_801930

noncomputable def T (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.digits 2 n.flat_map (λ d, if d = 0 then [1] else [1, 0]).foldl (λ acc d, acc * 10 + d) 0

theorem number_pattern_div_99 (a : ℕ) (N : ℕ) (h_init : a = 0) (h_trans : ∀ n, a n = T(a (n-1))) (h_div_9 : 9 ∣ N) : 99 ∣ N := 
by
  sorry

end number_pattern_div_99_l801_801930


namespace minTransfers_l801_801452

-- Define the cities as a finite set of number of cities.
constant Cities : Type
constant card_cities : Fintype.card Cities = 20

-- Define the flight network as a relation (i.e., adjacency to represent two-way flights).
constant FlightNetwork : Cities → Cities → Prop
axiom two_way_flight : ∀ a b : Cities, FlightNetwork a b → FlightNetwork b a

-- Define the degree constraint for flight connections.
axiom max_flight_connections : ∀ c : Cities, 4 ≥ Fintype.card { d : Cities // FlightNetwork c d }

-- Define the smallest k needed for any city to any other city to be connected with at most k transfers.
constant k : ℕ
axiom min_k : k = 2

-- The theorem statement.
theorem minTransfers (H : ∀ a b : Cities, ∃ (p : List Cities), (List.length p - 1 ≤ k ∧ p.head = a ∧ p.last = b ∧ ∀ (x y : Cities), (x, y) ∈ List.zip p.tail p.dropLast.tail → FlightNetwork x y)) : 
  k = 2 := by
  sorry

end minTransfers_l801_801452


namespace side_length_of_square_l801_801657

-- Defining relevant constants and initial conditions
def length_rect := 10
def width_rect := 20
def area_rect := length_rect * width_rect

-- Theorem statement (to prove the length of the side of the square is 10√2)
theorem side_length_of_square :
  let area_square := area_rect in
  let side_of_square := Real.sqrt area_square in
  side_of_square = 10 * Real.sqrt 2 :=
by
  sorry

end side_length_of_square_l801_801657


namespace fraction_simplification_l801_801440

theorem fraction_simplification (x y : ℚ) (h1 : x = 4) (h2 : y = 5) : 
  (1 / y) / (1 / x) = 4 / 5 :=
by
  sorry

end fraction_simplification_l801_801440


namespace problem1_problem2_l801_801972

theorem problem1 (x k : ℝ) (hx : x ≥ 1) (hk : k ≥ 1) : 
  ln x ≤ k * sqrt (x - 1) :=
sorry

theorem problem2 (n : ℕ) : 
  let a (n : ℕ) := sqrt (2 / (2 * n - 1))
  let S (n : ℕ) := Σ k in finset.range n, a (k+1)
  S n ≥ ln (2 * ↑n + 1) :=
sorry

end problem1_problem2_l801_801972


namespace sqrt_sum_comparison_l801_801323

theorem sqrt_sum_comparison : (√2 + √10) < 2 * √6 := 
by
  sorry

end sqrt_sum_comparison_l801_801323


namespace slope_range_of_PF_l801_801015

-- Definition of the hyperbola and its properties
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- The focus is given as (1, 0) for the hyperbola term x^2 - y^2 = 1
def left_focus : ℝ × ℝ := (-1, 0)

-- Point P is any point in the third quadrant on the hyperbola
def point_P_in_third_quadrant (P : ℝ × ℝ) : Prop := 
  P.1^2 - P.2^2 = 1 ∧ P.1 < 0 ∧ P.2 < 0

-- The slope of the line PF
def slope_PF (P F : ℝ × ℝ) : ℝ :=
  if P.1 = F.1 then 0 else (P.2 - F.2) / (P.1 - F.1)

-- The theorem statement
theorem slope_range_of_PF (P : ℝ × ℝ) (hP : point_P_in_third_quadrant P) :
  let s := slope_PF P left_focus
  in s < 0 ∨ s > 1 :=
sorry

end slope_range_of_PF_l801_801015


namespace tangent_addition_formula_l801_801474

theorem tangent_addition_formula
  (α β p q : ℝ)
  (h1 : (Real.tan α) + (Real.tan β) = p)
  (h2 : (Real.cot α) + (Real.cot β) = q) :
  Real.tan (α + β) = (p * q) / (q - p) := 
sorry

end tangent_addition_formula_l801_801474


namespace tickets_spent_on_hat_l801_801313

def tickets_won_whack_a_mole := 32
def tickets_won_skee_ball := 25
def tickets_left := 50
def total_tickets := tickets_won_whack_a_mole + tickets_won_skee_ball

theorem tickets_spent_on_hat : 
  total_tickets - tickets_left = 7 :=
by
  sorry

end tickets_spent_on_hat_l801_801313


namespace max_length_PQ_l801_801971

-- Define the curve in polar coordinates
def curve (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

-- Definition of points P and Q lying on the curve
def point_on_curve (ρ θ : ℝ) (P : ℝ × ℝ) : Prop :=
  curve ρ θ ∧ P = (ρ * Real.cos θ, ρ * Real.sin θ)

def points_on_curve (P Q : ℝ × ℝ) : Prop :=
  ∃ θ₁ θ₂ ρ₁ ρ₂, point_on_curve ρ₁ θ₁ P ∧ point_on_curve ρ₂ θ₂ Q

-- The theorem stating the maximum length of PQ
theorem max_length_PQ {P Q : ℝ × ℝ} (h : points_on_curve P Q) : dist P Q ≤ 4 :=
sorry

end max_length_PQ_l801_801971


namespace B_value_sum_of_a_c_l801_801103

-- The given conditions and the corresponding results
theorem B_value (a b c : ℝ) (A B C : ℝ) (h1 : b = 2 * sqrt 3) 
  (h2 : 1/2 * b * b * sin B = 2 * sqrt 3) 
  (h3 : cos^2 A = cos^2 B + sin^2 C - sin A * sin C) : B = π / 3 := by
  sorry

theorem sum_of_a_c (a b c : ℝ) (A B C : ℝ) (h1 : b = 2 * sqrt 3) 
  (h2 : 1/2 * b * b * sin B = 2 * sqrt 3) 
  (h3 : cos^2 A = cos^2 B + sin^2 C - sin A * sin C) 
  (h4 : B = π / 3) : a + c = 6 := by 
  sorry

end B_value_sum_of_a_c_l801_801103


namespace tens_digits_divisible_by_8_l801_801708

theorem tens_digits_divisible_by_8 : set_card {d : ℕ // ∃ (n : ℕ), (n % 1000) % 8 = 0 ∧ d = (n % 100) / 10} = 10 :=
sorry

end tens_digits_divisible_by_8_l801_801708


namespace triangle_sine_of_angle_l801_801307

theorem triangle_sine_of_angle (A : ℝ) (a : ℝ) (m : ℝ) (θ : ℝ) 
  (hA : A = 50) (ha : a = 12) (hm : m = 13) :
  (1 / 2) * a * m * real.sin θ = 50 → 
  real.sin θ = 25 / 39 :=
by
  intros 
  sorry

end triangle_sine_of_angle_l801_801307


namespace f_2016_value_l801_801392

def f : ℝ → ℝ := sorry

axiom f_prop₁ : ∀ x : ℝ, (x + 6) + f x = 0
axiom f_symmetry : ∀ x : ℝ, f (-x) = -f x ∧ f 0 = 0

theorem f_2016_value : f 2016 = 0 :=
by
  sorry

end f_2016_value_l801_801392


namespace find_sequences_sum_sequence_c_l801_801387

-- Definitions for conditions
def quadratic_roots : Prop :=
  ∃ (a1 a5 : ℝ), a1 ≠ a5 ∧ (a1 * a1 - 12 * a1 + 27 = 0) ∧ (a5 * a5 - 12 * a5 + 27 = 0)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d > 0, ∀ n : ℕ, a (n+1) - a n = d

def sum_b_n (b : ℕ → ℝ) (T : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → T n = 1 - (1 / 2) * b n

-- Proving the sequences a_n and b_n
theorem find_sequences (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) 
  (H1 : quadratic_roots)
  (H2 : arithmetic_sequence a)
  (H3 : sum_b_n b T) :
  (∀ n : ℕ, a n = 2 * n - 1) ∧ (∀ n : ℕ, b n = 2 / 3^n) :=
by sorry

-- Proving the sum of the sequence c_n
theorem sum_sequence_c (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ) (S : ℕ → ℝ)
  (H1 : quadratic_roots)
  (H2 : arithmetic_sequence a)
  (H3 : sum_b_n b (λ n, 1 - (1 / 2) * b n))
  (H4 : ∀ n : ℕ, c n = a n * b n) :
  (∀ n : ℕ, S n = 2 - (2 * n + 2) / 3^n) :=
by sorry

end find_sequences_sum_sequence_c_l801_801387


namespace probability_of_white_ball_l801_801283

theorem probability_of_white_ball (red_balls white_balls : ℕ) (draws : ℕ)
    (h_red : red_balls = 4) (h_white : white_balls = 2) (h_draws : draws = 2) :
    ((4 * 2 + 1) / 15 : ℚ) = 3 / 5 := by sorry

end probability_of_white_ball_l801_801283


namespace num_elements_in_A_range_of_a_for_B_l801_801330

noncomputable def f (a x : ℝ) : ℝ := Real.exp (a * x)

def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x
def is_stable_point (f : ℝ → ℝ) (x : ℝ) : Prop := f (f x) = x

def A (f : ℝ → ℝ) : set ℝ := { x | is_fixed_point f x }
def B (f : ℝ → ℝ) : set ℝ := { x | is_stable_point f x }

axiom property1 (f : ℝ → ℝ) : A f ⊆ B f
axiom property2 (f : ℝ → ℝ) (hf : monotone f) : A f = B f

-- Problem 1 Statement
theorem num_elements_in_A (a : ℝ) (h_pos : 0 < a) (h_neq : a ≠ 0) :
  (A (f a)).card = if h : a = (1 : ℝ) / Real.exp 1 then 1 else if 0 < a ∧ a < 1 / Real.exp 1 then 2 else 0 :=
sorry

-- Problem 2 Statement
theorem range_of_a_for_B (a : ℝ) (h_set_B_one : (B (f a)).card = 1) :
  a ∈ [-Real.exp 1, 0) ∪ {1 / Real.exp 1} :=
sorry

end num_elements_in_A_range_of_a_for_B_l801_801330


namespace david_marks_in_biology_l801_801329

theorem david_marks_in_biology (english: ℕ) (math: ℕ) (physics: ℕ) (chemistry: ℕ) (average: ℕ) (biology: ℕ) :
  english = 81 ∧ math = 65 ∧ physics = 82 ∧ chemistry = 67 ∧ average = 76 → (biology = 85) :=
by
  sorry

end david_marks_in_biology_l801_801329


namespace equation_of_parabola_l801_801357

def parabola_vertex_form_vertex (a x y : ℝ) := y = a * (x - 3)^2 - 2
def parabola_passes_through_point (a : ℝ) := 1 = a * (0 - 3)^2 - 2
def parabola_equation (y x : ℝ) := y = (1/3) * x^2 - 2 * x + 1

theorem equation_of_parabola :
  ∃ a : ℝ,
    ∀ x y : ℝ,
      parabola_vertex_form_vertex a x y ∧
      parabola_passes_through_point a →
      parabola_equation y x :=
by
  sorry

end equation_of_parabola_l801_801357


namespace triangle_side_lengths_l801_801596

theorem triangle_side_lengths (a b c : ℕ) (p : ℕ) (h_prime : p.prime) :
  ∃ a b c : ℕ, (a = 13) ∧ (b = 14) ∧ (c = 15) ∧ 
  (∃ ha hb hc : ℕ, ha > 0 ∧ hb > 0 ∧ hc > 0) ∧
  (∃ r : ℕ, r = p) :=
sorry

end triangle_side_lengths_l801_801596


namespace student_avg_always_greater_l801_801682

theorem student_avg_always_greater (x y z : ℝ) (h1 : x < y) (h2 : y < z) : 
  ( ( (x + y) / 2 + z) / 2 ) > ( (x + y + z) / 3 ) :=
by
  sorry

end student_avg_always_greater_l801_801682


namespace side_length_of_square_l801_801303

theorem side_length_of_square (A : ℝ) (h : A = 81) : ∃ s : ℝ, s^2 = A ∧ s = 9 :=
by
  sorry

end side_length_of_square_l801_801303


namespace triangle_not_isosceles_l801_801040

noncomputable def M : set ℝ := {a, b, c}

variables {a b c : ℝ}

def distinct_elements (M : set ℝ) : Prop :=
  ∀ {x y : ℝ}, x ∈ M → y ∈ M → x ≠ y → x ≠ y

def is_not_isosceles_triangle (M : set ℝ) : Prop :=
  ∀ {a b c : ℝ}, a ∈ M → b ∈ M → c ∈ M → 
  a ≠ b → b ≠ c → a ≠ c → 
  ¬(a = b ∨ b = c ∨ a = c)

theorem triangle_not_isosceles {a b c : ℝ} (h_dist : distinct_elements M) :
  is_not_isosceles_triangle M := 
sorry

end triangle_not_isosceles_l801_801040


namespace angles_sum_correct_l801_801696

-- Definitions from the problem conditions
def identicalSquares (n : Nat) := n = 13

variable (α β γ δ ε ζ η θ : ℝ) -- Angles of interest

def anglesSum :=
  (α + β + γ + δ) + (ε + ζ + η + θ)

-- Lean 4 statement
theorem angles_sum_correct
  (h₁ : identicalSquares 13)
  (h₂ : α = 90) (h₃ : β = 90) (h₄ : γ = 90) (h₅ : δ = 90)
  (h₆ : ε = 90) (h₇ : ζ = 90) (h₈ : η = 45) (h₉ : θ = 45) :
  anglesSum α β γ δ ε ζ η θ = 405 :=
by
  simp [anglesSum]
  sorry

end angles_sum_correct_l801_801696


namespace complementary_and_supplementary_angles_l801_801441

def angle := ℝ

noncomputable def given_angle : angle := 46

def complementary_angle (a : angle) : angle := 90 - a
def supplementary_angle (a : angle) : angle := 180 - a

theorem complementary_and_supplementary_angles : 
  complementary_angle given_angle = 44 ∧ supplementary_angle given_angle = 134 :=
by
  sorry

end complementary_and_supplementary_angles_l801_801441


namespace x_power_24_l801_801434

theorem x_power_24 (x : ℝ) (h : x + 1/x = -real.sqrt 3) : x^24 = 390625 :=
by
  sorry

end x_power_24_l801_801434


namespace projection_of_a_onto_b_l801_801438

variables (a b : ℝ^3)
variable (θ : ℝ)
variable (ha : ‖a‖ = 8)
variable (hb : ‖b‖ = 12)
variable (hθ : θ = real.pi / 4)

theorem projection_of_a_onto_b :
  (vector_proj b a) = (real.sqrt 2 / 3) • b :=
sorry

end projection_of_a_onto_b_l801_801438


namespace axis_of_symmetry_l801_801574

theorem axis_of_symmetry (x : ℝ) : (∀ y : ℝ, y = (2 - x) * x) → (∃ l : ℝ, l = 1) :=
by
  intro h
  use 1
  sorry

end axis_of_symmetry_l801_801574


namespace magnitude_of_b_perp_to_a_l801_801791

-- Define the vectors
def a : ℝ × ℝ := (-1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Vector dot product
def dot_prod (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Magnitude of vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Main statement to prove
theorem magnitude_of_b_perp_to_a : dot_prod a (b 6) = 0 → magnitude (b 6) = 3 * Real.sqrt 5 :=
by
  sorry

end magnitude_of_b_perp_to_a_l801_801791


namespace time_to_cross_platform_l801_801658

-- Definition of the given conditions
def length_of_train : ℕ := 1500 -- in meters
def time_to_cross_tree : ℕ := 120 -- in seconds
def length_of_platform : ℕ := 500 -- in meters
def speed : ℚ := length_of_train / time_to_cross_tree -- speed in meters per second

-- Definition of the total distance to cross the platform
def total_distance : ℕ := length_of_train + length_of_platform

-- Theorem to prove the time taken to cross the platform
theorem time_to_cross_platform : (total_distance / speed) = 160 :=
by
  -- Placeholder for the proof
  sorry

end time_to_cross_platform_l801_801658


namespace smallest_approved_expenditure_k_l801_801982

-- Define number of deputies and expenditure items
def num_deputies : ℕ := 2000
def num_items : ℕ := 200

-- Define the total permissible expense amount
def S : ℕ

-- Define the condition where each deputy's proposed budget does not exceed S
def proposed_budget (d : ℕ) : ℕ → ℕ := sorry

-- Define the approved expenditure per item by at least k deputies
def approved_expenditure (k : ℕ) : ℕ → ℕ := sorry

-- Define the function to find the smallest k satisfying the condition
noncomputable def smallest_k (S : ℕ) : ℕ :=
  if h : ∃ k, (∀ i < num_items, approved_expenditure k i ≤ S) ∧ (∀ j < k, approved_expenditure j 0 > S) then
    classical.some h
  else
    0

-- Proving the smallest k is 1991
theorem smallest_approved_expenditure_k (S : ℕ) : smallest_k S = 1991 := sorry

end smallest_approved_expenditure_k_l801_801982


namespace divisors_72_l801_801729

theorem divisors_72 : 
  { d | d ∣ 72 ∧ 0 < d } = {1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72} := 
sorry

end divisors_72_l801_801729


namespace rectangle_area_coefficient_l801_801675

theorem rectangle_area_coefficient (length width d k : ℝ) 
(h1 : length / width = 5 / 2) 
(h2 : d^2 = length^2 + width^2) 
(h3 : k = 10 / 29) :
  (length * width = k * d^2) :=
by
  sorry

end rectangle_area_coefficient_l801_801675


namespace shop_owner_percentage_profit_l801_801678

theorem shop_owner_percentage_profit (cost_price_per_kg : ℝ) 
  (buying_cheat_percentage : ℝ) (selling_cheat_percentage : ℝ) : 
  (buying_cheat_percentage = 0.12) → 
  (selling_cheat_percentage = 0.30) →
  let buying_effective_amount := cost_price_per_kg * (1 + buying_cheat_percentage) in
  let selling_effective_amount := cost_price_per_kg * (1 - selling_cheat_percentage) in
  let profit := (buying_effective_amount / selling_effective_amount) * cost_price_per_kg - cost_price_per_kg in
  let percentage_profit := (profit / cost_price_per_kg) * 100 in
  percentage_profit = 60 :=
begin
  intros,
  sorry
end

end shop_owner_percentage_profit_l801_801678


namespace max_area_quadrilateral_cdfg_l801_801649

theorem max_area_quadrilateral_cdfg (s : ℝ) (x : ℝ)
  (h1 : s = 1) (h2 : x > 0) (h3 : x < s) (h4 : AE = x) (h5 : AF = x) : 
  ∃ x, x > 0 ∧ x < 1 ∧ (1 - x) * x ≤ 5 / 8 :=
sorry

end max_area_quadrilateral_cdfg_l801_801649


namespace f_K_monotonically_increasing_l801_801394

def f (x : ℝ) : ℝ := 2 ^ -|x|

def f_K (x : ℝ) : ℝ :=
  if x ≤ -1 then 2 ^ x
  else if -1 < x ∧ x < 1 then 1 / 2
  else (1 / 2) ^ x

theorem f_K_monotonically_increasing :
  ∀ x y : ℝ, x < y → x < -1 → y ≤ -1 → f_K x ≤ f_K y :=
by
  intros x y hx hxl hy
  -- Proof here
  sorry

end f_K_monotonically_increasing_l801_801394


namespace common_tangent_length_valid_l801_801235

noncomputable def common_tangent_lengths (a R r : ℝ) : Set ℝ :=
  {L : ℝ | sqrt (a^2 - (R + r)^2) ≤ L ∧ L ≤ sqrt (a^2 - (R - r)^2)}

theorem common_tangent_length_valid
  (a R r : ℝ)
  (h_a : a > 0)
  (h_R : R > 0)
  (h_r : r > 0)
  (h_d : a > R + r) :
  ∃ L, L ∈ common_tangent_lengths a R r :=
begin
  sorry
end

end common_tangent_length_valid_l801_801235


namespace problem_l801_801858

variable (f : ℝ → ℝ)
variable (x : ℝ)

def periodic_function := ∀ x, f(x + 2 * Real.pi) = f x

theorem problem (h1 : periodic_function f) (h2 : f 0 = 0) : f (4 * Real.pi) = 0 := by sorry

end problem_l801_801858


namespace lottery_prob_correct_l801_801459

def possibleMegaBalls : ℕ := 30
def possibleWinnerBalls : ℕ := 49
def drawnWinnerBalls : ℕ := 6

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def winningProbability : ℚ :=
  (1 : ℚ) / possibleMegaBalls * (1 : ℚ) / combination possibleWinnerBalls drawnWinnerBalls

theorem lottery_prob_correct :
  winningProbability = 1 / 419514480 := by
  sorry

end lottery_prob_correct_l801_801459


namespace pascal_odd_rows_count_l801_801332

theorem pascal_odd_rows_count :
  let rows := {n : ℕ | n ∈ (finset.range 31) \ {0, 1} ∧ (∃ k, n = 2^k)} in
  finset.card rows = 4 :=
by
  -- We state our goal in Lean
  sorry -- proof implementation goes here

end pascal_odd_rows_count_l801_801332


namespace factorial_plus_one_div_prime_l801_801889

theorem factorial_plus_one_div_prime (n : ℕ) (h : (n! + 1) % (n + 1) = 0) : Nat.Prime (n + 1) := 
sorry

end factorial_plus_one_div_prime_l801_801889


namespace product_xy_l801_801432

theorem product_xy 
  (x y : ℝ) 
  (h1 : 8^x / 4^(x + y) = 64) 
  (h2 : 16^(x + y) / 4^(4*y) = 256) :
  x * y = 8 := 
sorry

end product_xy_l801_801432


namespace tan_sum_pi_over_12_eq_4_l801_801548

theorem tan_sum_pi_over_12_eq_4 :
  tan (π / 12) + tan (5 * π / 12) = 4 := 
by
  have cos_pi_over_12 : cos (π / 12) = (real.sqrt 6 + real.sqrt 2) / 4 := sorry
  have cos_5pi_over_12 : cos (5 * π / 12) = (real.sqrt 6 - real.sqrt 2) / 4 := sorry
  have sin_pi_over_2 : sin (π / 2) = 1 := by
    exact real.sin_pi_div_two
  sorry

end tan_sum_pi_over_12_eq_4_l801_801548


namespace triangle_inscribed_relation_l801_801267

noncomputable def herons_area (p a b c : ℝ) : ℝ := (p * (p - a) * (p - b) * (p - c)).sqrt

theorem triangle_inscribed_relation
  (S S' p p' : ℝ)
  (a b c a' b' c' r : ℝ)
  (h1 : r = S / p)
  (h2 : r = S' / p')
  (h3 : S = herons_area p a b c)
  (h4 : S' = herons_area p' a' b' c') :
  (p - a) * (p - b) * (p - c) / p = (p' - a') * (p' - b') * (p' - c') / p' :=
by sorry

end triangle_inscribed_relation_l801_801267


namespace find_x_l801_801077

noncomputable def e_squared := Real.exp 2

theorem find_x (x : ℝ) (h : Real.log (x^2 - 5*x + 10) = 2) :
  x = 4.4 ∨ x = 0.6 :=
sorry

end find_x_l801_801077


namespace soccer_balls_distribution_l801_801571

theorem soccer_balls_distribution:
  let num_ways : ℕ := 
    (finset.range (10)).filter (λ xy : ℕ × ℕ, let x := xy.1, y := xy.2 in
      (0 ≤ xy.1) ∧ (0 ≤ xy.2) ∧ (0 ≤ 3 - xy.1 - xy.2)).card 
  in num_ways = 10 := by
  sorry

end soccer_balls_distribution_l801_801571


namespace cube_root_of_a_minus_m_l801_801095

theorem cube_root_of_a_minus_m (m a : ℝ)
  (h1 : a > 0)
  (h2 : (m + 7) * (m + 7) = a)
  (h3 : (2 * m - 1) * (2 * m - 1) = a) :
  real.cbrt (a - m) = 3 :=
sorry

end cube_root_of_a_minus_m_l801_801095


namespace area_of_circle_l801_801212
open Real

-- Define the circumference condition
def circumference (r : ℝ) : ℝ :=
  2 * π * r

-- Define the area formula
def area (r : ℝ) : ℝ :=
  π * r * r

-- The given radius derived from the circumference
def radius_given_circumference (C : ℝ) : ℝ :=
  C / (2 * π)

-- The target proof statement
theorem area_of_circle (C : ℝ) (h : C = 36) : (area (radius_given_circumference C)) = 324 / π :=
by
  sorry

end area_of_circle_l801_801212


namespace operation_three_six_l801_801433

theorem operation_three_six : (3 * 3 * 6) / (3 + 6) = 6 :=
by
  calc (3 * 3 * 6) / (3 + 6) = 6 := sorry

end operation_three_six_l801_801433


namespace angle_measure_is_ninety_l801_801183

noncomputable def measure_angle_AQH (A B C D E F G H Q : Type) [RegularOctagon A B C D E F G H] [ExtendedMeet A B G H Q] : ℝ :=
  90

-- Formalizing the statement of the problem in Lean
theorem angle_measure_is_ninety
  (A B C D E F G H Q : Type)
  [RegularOctagon A B C D E F G H]
  [ExtendedMeet A B G H Q] :
  measure_angle_AQH A B C D E F G H Q = 90 := 
sorry

end angle_measure_is_ninety_l801_801183


namespace probability_license_plate_EY9_l801_801823

def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U', 'Y'}

def letters_except_y : Finset Char := Finset.filter (λ c, c ≠ 'Y') (Finset.rangeSet 'A' 'Z')

def digits : Finset Char := Finset.rangeSet '0' '9'

def total_license_plates : Nat :=
  vowels.card * letters_except_y.card * digits.card

def favorable_outcomes : Nat := 1 -- Only one "EY9"

theorem probability_license_plate_EY9 :
  (favorable_outcomes : ℚ) / (total_license_plates : ℚ) = 1 / 1500 :=
by
  sorry

end probability_license_plate_EY9_l801_801823


namespace f_inequality_l801_801985

noncomputable def f : ℝ → ℝ := sorry

-- Condition 1: f(x+3) = -1 / f(x)
axiom f_prop1 : ∀ x : ℝ, f (x + 3) = -1 / f x

-- Condition 2: ∀ 3 ≤ x_1 < x_2 ≤ 6, f(x_1) < f(x_2)
axiom f_prop2 : ∀ x1 x2 : ℝ, 3 ≤ x1 → x1 < x2 → x2 ≤ 6 → f x1 < f x2

-- Condition 3: The graph of y = f(x + 3) is symmetric about the y-axis
axiom f_prop3 : ∀ x : ℝ, f (3 - x) = f (3 + x)

-- Theorem: f(3) < f(4.5) < f(7)
theorem f_inequality : f 3 < f 4.5 ∧ f 4.5 < f 7 := by
  sorry

end f_inequality_l801_801985


namespace tan_sum_simplification_l801_801556
-- We start by importing the relevant Lean libraries that contain trigonometric functions and basic real analysis.

-- Define the statement to be proved in Lean.
theorem tan_sum_simplification :
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12) = 4 * Real.sqrt 2 - 4) :=
by
  sorry

end tan_sum_simplification_l801_801556


namespace linear_dependent_vectors_l801_801606

variable (m : ℝ) (a b : ℝ) 

theorem linear_dependent_vectors :
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 
    a • (⟨2, 3⟩ : ℝ × ℝ) + b • (⟨5, m⟩ : ℝ × ℝ) = (⟨0, 0⟩ : ℝ × ℝ)) ↔ m = 15 / 2 :=
sorry

end linear_dependent_vectors_l801_801606


namespace cone_curved_surface_area_at_5_seconds_l801_801663

theorem cone_curved_surface_area_at_5_seconds :
  let l := λ t : ℝ => 10 + 2 * t
  let r := λ t : ℝ => 5 + 1 * t
  let CSA := λ t : ℝ => Real.pi * r t * l t
  CSA 5 = 160 * Real.pi :=
by
  -- Definitions and calculations in the problem ensure this statement
  sorry

end cone_curved_surface_area_at_5_seconds_l801_801663


namespace percentage_increase_correct_l801_801605

def original_price : ℝ := 300
def new_price : ℝ := 390
def price_increase_percentage : ℝ := (new_price - original_price) / original_price * 100

theorem percentage_increase_correct : price_increase_percentage = 30 := 
by 
  sorry

end percentage_increase_correct_l801_801605


namespace exists_edge_with_acute_angles_l801_801175

noncomputable def edge_forms_acute_angles_with_adjacent_edges (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] : Prop :=
  ∃ (u v : A × B × C × D), 
    ∀ (w : A × B × C × D), 
    Metric.angle u w < π / 2 ∧ Metric.angle v w < π / 2

theorem exists_edge_with_acute_angles (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] :
  edge_forms_acute_angles_with_adjacent_edges A B C D :=
sorry

end exists_edge_with_acute_angles_l801_801175


namespace probability_of_satisfying_conditions_l801_801692

-- Definitions for the problem's conditions
def is_between_1000_and_9999 (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0
def digits_are_distinct (n : ℕ) : Prop := 
  let digits := (n / 1000, (n % 1000) / 100, (n % 100) / 10, n % 10) in
  list.nodup [digits.1, digits.2, digits.3, digits.4]

-- Definition for a number satisfying all the conditions
def satisfies_conditions (n : ℕ) : Prop :=
  is_between_1000_and_9999 n ∧ is_even n ∧ is_divisible_by_5 n ∧ digits_are_distinct n

-- The proof statement
theorem probability_of_satisfying_conditions :
  (∑ n in (finset.range 10000).filter satisfies_conditions, 1 : ℕ) / 9000 = 9 / 125 := by sorry

end probability_of_satisfying_conditions_l801_801692


namespace hyperbola_asymptotes_equation_l801_801585

noncomputable def hyperbola_asymptotes (x y : ℝ) : Prop :=
  (x^2 / 4 - y^2 / 9 = 1) → (y = (3 / 2) * x) ∨ (y = -(3 / 2) * x)

-- Now we assert the theorem that states this
theorem hyperbola_asymptotes_equation :
  ∀ (x y : ℝ), hyperbola_asymptotes x y :=
by
  intros x y
  unfold hyperbola_asymptotes
  -- proof here
  sorry

end hyperbola_asymptotes_equation_l801_801585


namespace plane_intersects_cube_through_center_of_cube_l801_801665

theorem plane_intersects_cube_through_center_of_cube
  {A B C D E F : Point}
  (cube : Cube)
  (hexagon_intersection: Plane → Cube → Hexagon)
  (P: Point) (plane : Plane)
  (Hex_inter_geom_cond: ∀ (P : Point), P ∈ (hexagon_intersection plane cube) → 
    (∃ O : Point, O ∈ (diagonals (hexagon_intersection plane cube)) ∧ 
    ((AD (hexagon_intersection plane cube) O) ∧ 
    (BE (hexagon_intersection plane cube) O) ∧ 
    (CF (hexagon_intersection plane cube) O)))
  :
  passes_through_center (hexagon_intersection plane cube) cube :=
sorry

end plane_intersects_cube_through_center_of_cube_l801_801665


namespace number_of_ways_to_place_balls_into_boxes_l801_801427

theorem number_of_ways_to_place_balls_into_boxes : 
  (number_of_ways (balls : ℕ) (boxes : ℕ) : ℕ) where balls = 5 ∧ boxes = 4 = 4^5 :=
begin
  sorry
end

end number_of_ways_to_place_balls_into_boxes_l801_801427


namespace student_chose_number_l801_801683

theorem student_chose_number : ∃ x : ℤ, 2 * x - 152 = 102 ∧ x = 127 :=
by
  sorry

end student_chose_number_l801_801683


namespace inscribed_circle_radius_in_sector_l801_801521

noncomputable def sector_radius : ℝ := 4

noncomputable def inscribed_circle_radius : ℝ :=
  (4 * Real.sqrt 3) / 3

theorem inscribed_circle_radius_in_sector :
  let sector_fraction : ℝ := 1/3
  let tangent_circle_radius : ℝ := 
    -- One-third of a circle sector
    if sector_fraction = 1/3 then inscribed_circle_radius else sorry
  sector_radius = 4 →
  tangent_circle_radius = (4 * Real.sqrt 3) / 3 :=
by
  intros
  rw [sector_radius]
  rw [inscribed_circle_radius]
  sorry

end inscribed_circle_radius_in_sector_l801_801521


namespace option_c_is_odd_l801_801270

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def f (x : ℝ) : ℝ := -3 * sin (2 * x)

theorem option_c_is_odd : is_odd_function f :=
  sorry

end option_c_is_odd_l801_801270


namespace shaded_area_l801_801354

def radius (R : ℝ) : Prop := R > 0
def angle (α : ℝ) : Prop := α = 20 * (Real.pi / 180)

theorem shaded_area (R : ℝ) (hR : radius R) (hα : angle (20 * (Real.pi / 180))) :
  let S0 := Real.pi * R^2 / 2
  let sector_radius := 2 * R
  let sector_angle := 20 * (Real.pi / 180)
  (2 * sector_radius * sector_radius * sector_angle / 2) / sector_angle = 2 * Real.pi * R^2 / 9 :=
by
  sorry

end shaded_area_l801_801354


namespace max_possible_value_d_n_l801_801150

noncomputable def a_n (n : ℕ) : ℚ := (5^n - 1) / 4

def d_n (n : ℕ) : ℕ := Int.gcd (a_n n).num (a_n (n + 1)).num

theorem max_possible_value_d_n : ∀ n : ℕ, d_n n = 1 := by
  sorry

end max_possible_value_d_n_l801_801150


namespace proof_problem_l801_801401

-- Definitions of p1 and p2
def p1 : Prop := ∃ (a b : ℝ), a^2 - a * b + b^2 < 0
def p2 : Prop := ∀ {A B C : ℝ} (h : ∠A > ∠B), sin ∠A > sin ∠B

-- The theorem statement
theorem proof_problem : (¬p1) ∧ p2 :=
by
  sorry

end proof_problem_l801_801401


namespace tan_sum_pi_over_12_eq_4_l801_801550

theorem tan_sum_pi_over_12_eq_4 :
  tan (π / 12) + tan (5 * π / 12) = 4 := 
by
  have cos_pi_over_12 : cos (π / 12) = (real.sqrt 6 + real.sqrt 2) / 4 := sorry
  have cos_5pi_over_12 : cos (5 * π / 12) = (real.sqrt 6 - real.sqrt 2) / 4 := sorry
  have sin_pi_over_2 : sin (π / 2) = 1 := by
    exact real.sin_pi_div_two
  sorry

end tan_sum_pi_over_12_eq_4_l801_801550


namespace point_concyclic_l801_801028

theorem point_concyclic
  (ABC : triangle)
  (H : point)
  (hH : is_orthocenter H ABC)
  (M_B M_C M_A : point)
  (hM_B : is_midpoint M_B (BC_segment ABC))
  (hM_C : is_midpoint M_C (CA_segment ABC))
  (hM_A : is_midpoint M_A (AB_segment ABC))
  (circle_M_B : circle)
  (circle_M_C : circle)
  (circle_M_A : circle)
  (h_circle_M_B : is_centered_circle M_B (passing_through circle_M_B H))
  (h_circle_M_C : is_centered_circle M_C (passing_through circle_M_C H))
  (h_circle_M_A : is_centered_circle M_A (passing_through circle_M_A H))
  (A_1 A_2 B_1 B_2 C_1 C_2 : point)
  (hA_1 : intersects_circle A_1 circle_M_B)
  (hA_2 : intersects_circle A_2 circle_M_B)
  (hB_1 : intersects_circle B_1 circle_M_C)
  (hB_2 : intersects_circle B_2 circle_M_C)
  (hC_1 : intersects_circle C_1 circle_M_A)
  (hC_2 : intersects_circle C_2 circle_M_A) :
  concyclic {A_1, A_2, B_1, B_2, C_1, C_2} :=
sorry

end point_concyclic_l801_801028


namespace Janice_age_l801_801475

theorem Janice_age (Mark_birthYear : ℕ) (currentYear : ℕ)
  (h1 : Mark_birthYear = 1976) (h2 : currentYear = 2021)
  (Mark_age : ℕ := currentYear - Mark_birthYear)
  (Graham_age : ℕ := Mark_age - 3)
  (Janice_age : ℕ := Graham_age / 2) : Janice_age = 21 :=
by {
  -- Facts from given conditions
  rw [h1, h2],
  -- Let's calculate age
  sorry
}

end Janice_age_l801_801475


namespace complex_quadrant_l801_801566

theorem complex_quadrant :
  let i := complex.I
  let (a, b) ∈ ℝ × ℝ := (1/5, -3/10 - 1)
  let z := a + b * i
  in (z.re > 0) ∧ (z.im < 0) :=
by
  have a : ℝ := 1/5
  have b : ℝ := -3/10 - 1
  let i := complex.I
  let z := a + b * i
  exact (a > 0) ∧ (b * i < 0)
  sorry

end complex_quadrant_l801_801566


namespace max_elements_in_set_l801_801496

def no_pair_sum_divisible_by_5 (T : set ℕ) : Prop :=
  ∀ (x y ∈ T), x ≠ y → (x + y) % 5 ≠ 0

theorem max_elements_in_set (T : set ℕ) :
  (∀ x ∈ T, x ∈ finset.range 101) → no_pair_sum_divisible_by_5 T → T.to_finset.card ≤ 40 :=
by sorry

end max_elements_in_set_l801_801496


namespace equation_of_ellipse_area_of_triangle_l801_801771

def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)
def e : ℝ := 1 / 2

-- Proof for the equation of the ellipse
theorem equation_of_ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0):
  a = 2 → b^2 = a^2 - 1 → (b = sqrt 3) →
  (∀ x y : ℝ, (x^2 / (2^2) + y^2 / b^2 = 1) ↔ (x^2 / 4 + y^2 / 3 = 1)) := 
by
  sorry

-- Proof for the area of the triangle
theorem area_of_triangle (P : ℝ × ℝ) (angle_P : ∀ x1 y1 x2 y2 : ℝ, angle P (x1, y1) (x2, y2) = π/3) :
  ∃ (S : ℝ), S = sqrt 3 := 
by
  sorry

end equation_of_ellipse_area_of_triangle_l801_801771


namespace no_number_has_reversed_product_of_ones_l801_801967

theorem no_number_has_reversed_product_of_ones (n : ℕ) (hn : n > 1) :
  ¬ (∃ m : ℕ, (reverse_digits n) * n = (10^l - 1) / 9) :=
begin
  sorry
end

end no_number_has_reversed_product_of_ones_l801_801967


namespace equal_roots_of_quadratic_l801_801093

theorem equal_roots_of_quadratic (k : ℝ) : 
  (∃ x, (x^2 + 2 * x + k = 0) ∧ (x^2 + 2 * x + k) = 0) → k = 1 :=
by
  sorry

end equal_roots_of_quadratic_l801_801093


namespace find_m_l801_801746

theorem find_m (x1 x2 m : ℝ) (h1 : x1 + x2 = 4) (h2 : x1 + 3 * x2 = 5) : m = 7 / 4 :=
  sorry

end find_m_l801_801746


namespace quadrilateral_sides_equal_or_parallel_l801_801888

theorem quadrilateral_sides_equal_or_parallel
  (A B C D M1 M2 M3 M4 : Type)
  [affine_space ℝ A] [affine_space ℝ B] [affine_space ℝ C] [affine_space ℝ D]
  [affine_space ℝ M1] [affine_space ℝ M2] [affine_space ℝ M3] [affine_space ℝ M4]
  (midpoint_AB : midpoint A B = M1)
  (midpoint_CD : midpoint C D = M2)
  (midpoint_BC : midpoint B C = M3)
  (midpoint_AD : midpoint A D = M4)
  (midline : line_through M1 M2)
  (eq_angles : angle M1 M2 M3 = angle M1 M2 M4) :
  (is_parallel A D B C ∨ (distance A D = distance B C)) :=
by
  sorry

end quadrilateral_sides_equal_or_parallel_l801_801888


namespace janine_read_pages_in_two_months_l801_801132

theorem janine_read_pages_in_two_months :
  (let books_last_month := 5
   let books_this_month := 2 * books_last_month
   let total_books := books_last_month + books_this_month
   let pages_per_book := 10
   total_books * pages_per_book = 150) := by
   sorry

end janine_read_pages_in_two_months_l801_801132


namespace probability_T_H_E_equal_L_A_V_A_l801_801851

noncomputable def probability_condition : ℚ :=
  -- Number of total sample space (3^6)
  (3 ^ 6 : ℚ)

noncomputable def favorable_events_0 : ℚ :=
  -- Number of favorable outcomes where 𝑻 ⋅ 𝑯 ⋅ 𝑬 is 0 and 𝑳 ⋅ 𝑨 ⋅ 𝑽 ⋅ 𝑨 is 0
  26 * 19

noncomputable def favorable_events_1 : ℚ :=
  -- Number of favorable outcomes where 𝑻 ⋅ 𝑯 ⋅ 𝑬 is 1 and 𝑳 ⋅ 𝑨 ⋅ 𝑽 ⋅ 𝑨 is 1
  1

noncomputable def total_favorable_events : ℚ :=
  favorable_events_0 + favorable_events_1

theorem probability_T_H_E_equal_L_A_V_A :
  (total_favorable_events / probability_condition) = 55 / 81 :=
sorry

end probability_T_H_E_equal_L_A_V_A_l801_801851


namespace nat_triple_solution_l801_801730

theorem nat_triple_solution (x y n : ℕ) :
  (x! + y!) / n! = 3^n ↔ (x = 1 ∧ y = 2 ∧ n = 1) ∨ (x = 2 ∧ y = 1 ∧ n = 1) := 
by
  sorry

end nat_triple_solution_l801_801730


namespace Mark_water_balloon_spending_l801_801862

theorem Mark_water_balloon_spending :
  let budget := 24
  let small_bag_cost := 4
  let small_bag_balloons := 50
  let medium_bag_balloons := 75
  let extra_large_bag_cost := 12
  let extra_large_bag_balloons := 200
  let total_balloons := 400
  (2 * extra_large_bag_balloons = total_balloons) → (2 * extra_large_bag_cost = budget) :=
by
  intros
  sorry

end Mark_water_balloon_spending_l801_801862


namespace coin_difference_l801_801879

theorem coin_difference (h : ∃ x y z : ℕ, 5*x + 10*y + 20*z = 40) : (∃ x : ℕ, 5*x = 40) → (∃ y : ℕ, 20*y = 40) → 8 - 2 = 6 :=
by
  intros h1 h2
  exact rfl

end coin_difference_l801_801879


namespace boys_count_correct_l801_801941

def total_children : ℕ := 97
def number_of_girls : ℕ := 53
def number_of_boys : ℕ := total_children - number_of_girls

theorem boys_count_correct : number_of_boys = 44 := by
  unfold number_of_boys
  unfold total_children
  unfold number_of_girls
  simp
  sorry

end boys_count_correct_l801_801941


namespace simplify_tangent_sum_l801_801542

theorem simplify_tangent_sum :
  tan (Real.pi / 12) + tan (5 * Real.pi / 12) = Real.sqrt 6 - Real.sqrt 2 := 
sorry

end simplify_tangent_sum_l801_801542


namespace find_two_sets_l801_801322

theorem find_two_sets :
  ∃ (a1 a2 a3 a4 a5 b1 b2 b3 b4 b5 : ℕ),
    a1 + a2 + a3 + a4 + a5 = a1 * a2 * a3 * a4 * a5 ∧
    b1 + b2 + b3 + b4 + b5 = b1 * b2 * b3 * b4 * b5 ∧
    (a1, a2, a3, a4, a5) ≠ (b1, b2, b3, b4, b5) := by
  sorry

end find_two_sets_l801_801322


namespace tan_pi_over_12_plus_tan_5pi_over_12_eq_4_l801_801529

-- Definitions for given conditions
def cos_pi_over_12 : ℝ := (Real.sqrt 6 + Real.sqrt 2) / 4
def cos_5pi_over_12 : ℝ := (Real.sqrt 6 - Real.sqrt 2) / 4

-- The theorem to be proved
theorem tan_pi_over_12_plus_tan_5pi_over_12_eq_4 : 
  Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 :=
by sorry

end tan_pi_over_12_plus_tan_5pi_over_12_eq_4_l801_801529


namespace construct_triangle_l801_801788

-- Definitions of lines and points and their intersections
variables {Point Line : Type} (l₁ l₂ l₃ : Line) (A₁ : Point)

-- Assumptions about the given conditions
axiom intersect_at_single_point : ∃ (O : Point), 
  (O ∈ l₁) ∧ (O ∈ l₂) ∧ (O ∈ l₃)
axiom A₁_on_l₁ : A₁ ∈ l₁

-- Definitions needed for proving the existence of the triangle
def is_midpoint (M B C : Point) := 
  let d1 := dist M B in
  let d2 := dist M C in
  d1 = d2

def is_perpendicular_bisector (l : Line) (A B : Point) := 
  l ∈ perp_bisector A B

-- The statement we need to prove:
theorem construct_triangle (l₁ l₂ l₃ : Line) (A₁ : Point) :
  ∃ (A B C : Point),
    is_midpoint A₁ B C ∧
    is_perpendicular_bisector l₁ B C ∧
    is_perpendicular_bisector l₂ C A ∧
    is_perpendicular_bisector l₃ A B :=
sorry

end construct_triangle_l801_801788


namespace solve_inequality_l801_801191

theorem solve_inequality (a : ℝ) :
  (λ x : ℝ, x^2 - (a + 1) * x + a > 0) = 
  {x | if a < 1 then x < a ∨ x > 1 else if a = 1 then x ≠ 1 else x < 1 ∨ x > a} :=
by sorry

end solve_inequality_l801_801191


namespace smallest_coefficient_term_in_binomial_expansion_l801_801117

-- Define factorial, combinatorial, and binomial expansion if not already defined 
noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

lemma binomial_coeff (n k : ℕ) : ℕ :=
factorial n / (factorial k * factorial (n - k))

-- Define the binomial term in the expansion
def binomial_term (x y : ℝ) (n k : ℕ) : ℝ :=
binomial_coeff n k * (x^(n - k)) * ((-y)^k)

-- Formulate the proof problem in Lean 4
theorem smallest_coefficient_term_in_binomial_expansion (x y : ℝ) :
  ∃ (k : ℕ), k = 6 ∧
  (∀ m, (m ≠ 6 ∧ m ≤ 10) ->
    binomial_term x y 10 m > binomial_term x y 10 6) :=
sorry

end smallest_coefficient_term_in_binomial_expansion_l801_801117


namespace w_squared_approx_l801_801075

theorem w_squared_approx {w : ℝ} (h_eq : (w + 15)^2 = (4 * w + 9) * (3 * w + 6)) : w^2 ≈ 3.101 := 
sorry

end w_squared_approx_l801_801075


namespace steven_more_peaches_than_apples_l801_801126

-- Definitions
def apples_steven := 11
def peaches_steven := 18

-- Theorem statement
theorem steven_more_peaches_than_apples : (peaches_steven - apples_steven) = 7 := by 
  sorry

end steven_more_peaches_than_apples_l801_801126


namespace sum_of_squares_of_divisors_1800_l801_801003

def sum_of_squares_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (Finset.Ico 1 (n+1)).filter (λ x, n % x = 0), d^2

theorem sum_of_squares_of_divisors_1800 :
  sum_of_squares_of_divisors 1800 = 5035485 :=
by sorry

end sum_of_squares_of_divisors_1800_l801_801003


namespace product_max_min_a_l801_801037

theorem product_max_min_a : 
  ∀ (a b c : ℝ), a + b + c = 15 → a^2 + b^2 + c^2 = 100 →
    let max_a := 5 + 5 * Real.sqrt 6 / 3 in
    let min_a := 5 - 5 * Real.sqrt 6 / 3 in
    max_a * min_a = 25 / 3 :=
by
  sorry

end product_max_min_a_l801_801037


namespace num_ways_5_balls_4_boxes_l801_801428

-- Define the function calculating number of ways to place n balls into m boxes
def num_ways (n : ℕ) (m : ℕ) : ℕ := m ^ n

-- The proof that placing 5 distinguishable balls into 4 distinguishable boxes
theorem num_ways_5_balls_4_boxes : num_ways 5 4 = 1024 := by
  -- Calculation of 4^5
  unfold num_ways
  -- Check if the calculation matches 1024
  norm_num
  exact rfl

end num_ways_5_balls_4_boxes_l801_801428


namespace simplify_tangent_sum_l801_801543

theorem simplify_tangent_sum :
  tan (Real.pi / 12) + tan (5 * Real.pi / 12) = Real.sqrt 6 - Real.sqrt 2 := 
sorry

end simplify_tangent_sum_l801_801543


namespace volume_of_each_cube_is_correct_l801_801987

def box_length : ℕ := 12
def box_width : ℕ := 16
def box_height : ℕ := 6
def total_volume : ℕ := 1152
def number_of_cubes : ℕ := 384

theorem volume_of_each_cube_is_correct :
  (total_volume / number_of_cubes = 3) :=
by
  sorry

end volume_of_each_cube_is_correct_l801_801987


namespace smallest_n_exist_infinitely_many_rationals_l801_801735

theorem smallest_n_exist_infinitely_many_rationals (n : ℕ) (h1 : 0 < n) (h2 :
  ∃ᶠ (a : Fin n → ℚ) in (at_top : Filter (Fin n → ℚ)),
  (∀ i, 0 < a i) ∧ (∑ i, a i ∈ ℤ) ∧ (∑ i, (1 : ℚ) / a i ∈ ℤ))
  : n = 3 := sorry

end smallest_n_exist_infinitely_many_rationals_l801_801735


namespace riza_son_age_l801_801893

theorem riza_son_age (R S : ℕ) (h1 : R = S + 25) (h2 : R + S = 105) : S = 40 :=
by
  sorry

end riza_son_age_l801_801893


namespace Ilya_wins_l801_801124

def initial_pile_A : ℕ := 100
def initial_pile_B : ℕ := 101
def initial_pile_C : ℕ := 102

structure GameState :=
  (pile_A : ℕ)
  (pile_B : ℕ)
  (pile_C : ℕ)
  (previous_pile : Option (Fin 3))

def initial_game_state : GameState :=
  { pile_A := initial_pile_A, pile_B := initial_pile_B, pile_C := initial_pile_C, previous_pile := none }

def valid_move (s : GameState) (pile_choice : Fin 3) : Prop :=
  match s.previous_pile with
  | some p => p ≠ pile_choice
  | none => True

def make_move (s : GameState) (pile_choice : Fin 3) : GameState :=
  match pile_choice with
  | ⟨0, _⟩ => { s with pile_A := s.pile_A - 1, previous_pile := some ⟨0, _⟩ }
  | ⟨1, _⟩ => { s with pile_B := s.pile_B - 1, previous_pile := some ⟨1, _⟩ }
  | ⟨2, _⟩ => { s with pile_C := s.pile_C - 1, previous_pile := some ⟨2, _⟩ }

def no_more_moves (s : GameState) : Prop :=
  (s.pile_A = 0 ∨ match s.previous_pile with | some ⟨0, _⟩ => s.pile_A = 0 | _ => False) ∧
  (s.pile_B = 0 ∨ match s.previous_pile with | some ⟨1, _⟩ => s.pile_B = 0 | _ => False) ∧
  (s.pile_C = 0 ∨ match s.previous_pile with | some ⟨2, _⟩ => s.pile_C = 0 | _ => False)

theorem Ilya_wins : 
  ∃ strategy : (GameState → Fin 3), 
  ∀ s, valid_move s (strategy s) → 
       (no_more_moves s → no_more_moves (make_move s (strategy s))) ∧
       (no_more_moves s → ∃ s', make_move s (strategy s) = s' ∧ ¬no_more_moves s') :=
sorry

end Ilya_wins_l801_801124


namespace tan_XYZ_l801_801470

theorem tan_XYZ {X Y Z : Type} [euclidean_geometry X] [triangle X Y Z] 
  (angleZ : angle Z = 90) (XY : distance X Y = 10) (YZ : distance Y Z = Real.sqrt 51) :
  tan (angle X) = Real.sqrt 51 / 7 := 
  sorry

end tan_XYZ_l801_801470


namespace range_of_k_for_extremum_l801_801088

theorem range_of_k_for_extremum (k : ℝ) :
  (∃ x : ℝ, (∃ f' : ℝ → ℝ, f' x = 0 ∧ f' = λ x, cos x - k)) ↔ -1 < k ∧ k < 1 :=
begin
  sorry
end

end range_of_k_for_extremum_l801_801088


namespace combined_percent_of_6th_graders_l801_801936

theorem combined_percent_of_6th_graders (num_students_pineview : ℕ) 
                                        (percent_6th_pineview : ℝ) 
                                        (num_students_oakridge : ℕ)
                                        (percent_6th_oakridge : ℝ)
                                        (num_students_maplewood : ℕ)
                                        (percent_6th_maplewood : ℝ) 
                                        (total_students : ℝ) :
    num_students_pineview = 150 →
    percent_6th_pineview = 0.15 →
    num_students_oakridge = 180 →
    percent_6th_oakridge = 0.17 →
    num_students_maplewood = 170 →
    percent_6th_maplewood = 0.15 →
    total_students = 500 →
    ((percent_6th_pineview * num_students_pineview) + 
     (percent_6th_oakridge * num_students_oakridge) + 
     (percent_6th_maplewood * num_students_maplewood)) / 
    total_students * 100 = 15.72 :=
by
  sorry

end combined_percent_of_6th_graders_l801_801936


namespace KM_perp_KD_l801_801855

-- Definitions based on conditions from part c)
variables (L M K B A C D : Point)

-- Midpoints
def is_midpoint (L : Point) (C D : Point) := dist L C = dist L D

-- Medians
def is_median (M L C D : Point) := is_midpoint L C D ∧ dist M L = dist M (C + D)

-- Perpendicularity
def perp (M L : Point) (C D : Line) := ∃ l : Line, l.orthogonal M L ∧ l.orthogonal C D

-- Conditions
axiom midpoint_L_CD : is_midpoint L C D
axiom midpoint_M_AB : is_midpoint M A B
axiom median_ML_CD : perp M L (line_through C D)
axiom parallel_BK_CD : parallel (line_through B K) (line_through C D)
axiom parallel_KL_AD : parallel (line_through K L) (line_through A D)

-- The proof statement
theorem KM_perp_KD : perp K M (line_through K D) := sorry

end KM_perp_KD_l801_801855


namespace f_g_2_eq_neg_19_l801_801408

def f (x : ℝ) : ℝ := 5 - 4 * x

def g (x : ℝ) : ℝ := x^2 + 2

theorem f_g_2_eq_neg_19 : f (g 2) = -19 := 
by
  -- The proof is omitted
  sorry

end f_g_2_eq_neg_19_l801_801408


namespace exponent_property_l801_801796

theorem exponent_property (a : ℝ) (m n : ℝ) (h₁ : a^m = 4) (h₂ : a^n = 8) : a^(m + n) = 32 := 
by 
  sorry

end exponent_property_l801_801796


namespace monotonic_intervals_maximum_on_interval_ratio_inequality_l801_801053

variable {a x x1 x2 : ℝ}

def f (x : ℝ) (a : ℝ) : ℝ := x / a - Real.exp x

theorem monotonic_intervals (ha : 0 < a) :
  ∀ x : ℝ, (f x a).monotone_on (Iio (Real.log (1 / a))) → 
           (f x a).monotone_on (Ioi (Real.log (1 / a))) := 
sorry

theorem maximum_on_interval (ha : 0 < a) :
  (∃ x : ℝ, f x a = max {f 1 a, f 2 a, f (Real.log (1 / a)) a}.sup) :=
sorry

theorem ratio_inequality (ha : 0 < a) (hx1x2 : x1 < x2) (hx1f0 : f x1 a = 0) (hx2f0 : f x2 a = 0) :
  x1 / x2 < a * Real.exp 1 :=
sorry

end monotonic_intervals_maximum_on_interval_ratio_inequality_l801_801053


namespace inverse_proposition_P_l801_801032

variable (a : ℕ)

def P : Prop := a % 2 = 1 → nat.prime a

def inverse_P : Prop := nat.prime a → a % 2 = 1

theorem inverse_proposition_P : inverse_P a := sorry

end inverse_proposition_P_l801_801032


namespace concert_attendance_difference_l801_801870

theorem concert_attendance_difference :
  let first_concert := 65899
  let second_concert := 66018
  second_concert - first_concert = 119 :=
by
  sorry

end concert_attendance_difference_l801_801870


namespace fish_caught_by_dad_l801_801315

def total_fish_both : ℕ := 23
def fish_caught_morning : ℕ := 8
def fish_thrown_back : ℕ := 3
def fish_caught_afternoon : ℕ := 5
def fish_kept_brendan : ℕ := fish_caught_morning - fish_thrown_back + fish_caught_afternoon

theorem fish_caught_by_dad : total_fish_both - fish_kept_brendan = 13 := by
  sorry

end fish_caught_by_dad_l801_801315


namespace change_in_enthalpy_CaCO3_formation_l801_801709

-- Define the given bond dissociation energies
def CaO_bond_energy : ℝ := 1500 -- in kJ/mol
def CO_double_bond_energy : ℝ := 730 -- in kJ/mol
def CO_single_bond_energy : ℝ := 360 -- in kJ/mol
def OO_double_bond_energy : ℝ := 498 -- in kJ/mol

-- Define the formation reaction energy calculations
def ΔH_break : ℝ := (3/2) * OO_double_bond_energy
def ΔH_form : ℝ := CaO_bond_energy + CO_double_bond_energy + 2 * CO_single_bond_energy

-- Define the change in enthalpy (ΔH)
def ΔH : ℝ := ΔH_form - ΔH_break

-- The final theorem to prove the change in enthalpy is 2203 kJ/mol
theorem change_in_enthalpy_CaCO3_formation : ΔH = 2203 := by
  sorry

end change_in_enthalpy_CaCO3_formation_l801_801709


namespace distance_calculation_l801_801933

variables (D : ℝ)

def speed_boat_still_water : ℝ := 8
def speed_stream : ℝ := 2

def downstream_speed : ℝ := speed_boat_still_water + speed_stream
def upstream_speed : ℝ := speed_boat_still_water - speed_stream

def downstream_time (D : ℝ) : ℝ := D / downstream_speed
def upstream_time (D : ℝ) : ℝ := D / upstream_speed

def total_time (D : ℝ) : ℝ :=
  downstream_time D + upstream_time D

theorem distance_calculation : total_time D = 56 → D = 210 :=
sorry

end distance_calculation_l801_801933


namespace exists_g_l801_801489

variable {R : Type} [Field R]

-- Define the function f with the given condition
def f (x y : R) : R := sorry

-- The main theorem to prove the existence of g
theorem exists_g (f_condition: ∀ x y z : R, f x y + f y z + f z x = 0) : ∃ g : R → R, ∀ x y : R, f x y = g x - g y := 
by 
  sorry

end exists_g_l801_801489


namespace angle_at_intersection_of_extended_sides_of_regular_octagon_l801_801180

theorem angle_at_intersection_of_extended_sides_of_regular_octagon
  (ABCDEFGH : Prop) -- Regular octagon
  (is_regular_octagon : ∀ (A B C D E F G H Q : ℝ), regular_octagon A B C D E F G H)
  (AB GH : ℝ) -- Sides AB and GH of the octagon
  (extended_to_Q : ∃ Q, extend_line AB Q ∧ extend_line GH Q)
  : angle_at_point Q = 90 :=
sorry

end angle_at_intersection_of_extended_sides_of_regular_octagon_l801_801180


namespace compute_fraction_eq_2410_l801_801324

theorem compute_fraction_eq_2410 (x : ℕ) (hx : x = 7) : 
  (x^8 + 18 * x^4 + 81) / (x^4 + 9) = 2410 := 
by
  -- proof steps go here
  sorry

end compute_fraction_eq_2410_l801_801324


namespace planet_colonization_ways_l801_801074

theorem planet_colonization_ways :
  let earth_like := 7
  let mars_like := 15 - earth_like
  let total_units := 15
  let earth_unit := 3
  let mars_unit := 1
  let valid_b := {0, 3, 6, 9, 12, 15}

  ∃ (n : ℕ), n = 21 + 1960 + 980 :=
  sorry

end planet_colonization_ways_l801_801074


namespace area_trap_GHCD_l801_801120

-- Define the problem's conditions and question in Lean 4
variables (AB CD altitude : ℝ)
variables (mid_GH_altitude GH_base CD_base : ℝ)

-- Given conditions
def trap_ABCD :=
  AB = 10 ∧
  CD = 26 ∧
  altitude = 15 ∧
  mid_GH_altitude = 15 / 2 ∧
  GH_base = (10 + 26) / 2 ∧
  CD_base = 26

-- Define what we want to prove
def area_GHCD (GH_base CD_base mid_GH_altitude : ℝ) : ℝ :=
  mid_GH_altitude * ((GH_base + CD_base) / 2)

-- State the theorem
theorem area_trap_GHCD : trap_ABCD AB CD altitude mid_GH_altitude GH_base CD_base →
  area_GHCD GH_base CD_base mid_GH_altitude = 165 :=
begin
  -- Proof steps would go here
  sorry
end

end area_trap_GHCD_l801_801120


namespace domain_of_g_l801_801801

theorem domain_of_g :
  (∀ x : ℝ, x ∈ Set.Icc (-3 : ℝ) (1 : ℝ) → f x) →
  ∀ y : ℝ, (y ∈ Set.Icc (-1 : ℝ) (1 : ℝ)) ↔ ((g y = f y + f (-y)) → y ∈ Set.Icc (-1 : ℝ) (1 : ℝ)) :=
by
  intros hf y
  split
  · intro hy
    exact hy
  · intro hg
    split
    · have h_neg : -y ∈ Set.Icc (-3 : ℝ) (1 : ℝ) := by sorry
      sorry
    sorry
    sorry

end domain_of_g_l801_801801


namespace find_g_3_l801_801080

noncomputable def g (x : ℝ) : ℝ := sorry

theorem find_g_3 (h : ∀ x : ℝ, g (2 * x - 5) = 3 * x + 9) : g 3 = 21 :=
by
  sorry

end find_g_3_l801_801080


namespace tan_pi_over_12_plus_tan_5pi_over_12_eq_4_l801_801523

-- Definitions for given conditions
def cos_pi_over_12 : ℝ := (Real.sqrt 6 + Real.sqrt 2) / 4
def cos_5pi_over_12 : ℝ := (Real.sqrt 6 - Real.sqrt 2) / 4

-- The theorem to be proved
theorem tan_pi_over_12_plus_tan_5pi_over_12_eq_4 : 
  Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 :=
by sorry

end tan_pi_over_12_plus_tan_5pi_over_12_eq_4_l801_801523


namespace cube_root_of_difference_l801_801096

variable (a m : ℝ)
axiom (h1 : a > 0)
axiom (h2 : sqrt(a) = m + 7 ∨ sqrt(a) = 2m - 1)

theorem cube_root_of_difference : real.cbrt (a - m) = 3 :=
by
  sorry

end cube_root_of_difference_l801_801096


namespace line_equation_l801_801294

-- The conditions
def line_vector_eq (x y : ℝ) : Prop :=
  (3 * (x + 2) - 4 * (y - 8) = 0)

-- The proof statement to be shown
theorem line_equation (x y : ℝ) :
  line_vector_eq x y →
  (∃ m b : ℝ, m = 3 / 4 ∧ b = 19 / 2 ∧ y = m * x + b) :=
by
  intros h,
  use 3 / 4,
  use 19 / 2,
  split,
  { refl },
  split,
  { refl },
  { sorry }

end line_equation_l801_801294


namespace tangent_line_correct_l801_801921

noncomputable def tangent_line_equation : ℝ → ℝ := 
  fun x => (exp(1) / 4) * x + (exp(1) / 4)

theorem tangent_line_correct :
  (∀ x, (x ≠ 1 → tangent_line_equation x = (exp(1) / 4) * x + (exp(1) / 4)) ∧
       (∀ y, (y = 1 → tangent_line_equation 1 = exp(1) / 2)) ∧
       (∀ f, (f x = exp(x) / (x + 1) ∧ x = 1 → deriv f 1 = exp(1) / 4))) :=
by
  sorry

end tangent_line_correct_l801_801921


namespace funfair_initial_visitors_l801_801672

theorem funfair_initial_visitors {a : ℕ} (ha1 : 50 * a - 40 > 0) (ha2 : 90 - 20 * a > 0) (ha3 : 50 * a - 40 > 90 - 20 * a) :
  (50 * a - 40 = 60) ∨ (50 * a - 40 = 110) ∨ (50 * a - 40 = 160) :=
sorry

end funfair_initial_visitors_l801_801672


namespace monochromatic_triangle_probability_correct_l801_801715

noncomputable def monochromatic_triangle_probability (p : ℝ) : ℝ :=
  1 - (3 * (p^2) * (1 - p) + 3 * ((1 - p)^2) * p)^20

theorem monochromatic_triangle_probability_correct :
  monochromatic_triangle_probability (1/2) = 1 - (3/4)^20 :=
by
  sorry

end monochromatic_triangle_probability_correct_l801_801715


namespace find_mode_median_l801_801245
noncomputable def dataset : List ℕ := [38, 42, 35, 40, 36, 42, 75]

theorem find_mode_median (d : List ℕ) (h : d = dataset) : 
  (mode d = 42) ∧ (median d = 40) := 
sorry

end find_mode_median_l801_801245


namespace domain_of_f_l801_801581

noncomputable def f (x : ℝ) : ℝ :=
  real.sqrt (1 - real.log x)

theorem domain_of_f :
  {x : ℝ | 0 < x ∧ x ≤ real.exp 1} = 
  {x : ℝ | 0 < x ∧ 1 - real.log x ≥ 0} :=
begin
  sorry
end

end domain_of_f_l801_801581


namespace largest_angle_in_triangle_l801_801935

theorem largest_angle_in_triangle (A B C : ℝ) 
  (h_sum : A + B = 126) 
  (h_diff : B = A + 40) 
  (h_triangle : A + B + C = 180) : max A (max B C) = 83 := 
by
  sorry

end largest_angle_in_triangle_l801_801935


namespace min_distance_midpoint_l801_801817

-- Definitions of the curves and line
def C1_x (t : ℝ) : ℝ := -4 + Real.cos t
def C1_y (t : ℝ) : ℝ := 3 + Real.sin t

def C2_x (θ : ℝ) : ℝ := 8 * Real.cos θ
def C2_y (θ : ℝ) : ℝ := 3 * Real.sin θ

def C3 (r θ : ℝ) : Prop := r * (Real.cos θ - 2 * Real.sin θ) = 7

-- Prove the minimum distance from midpoint M of P and Q to the line C3
theorem min_distance_midpoint :
  ∀ θ : ℝ, 
  let P : ℝ × ℝ := (-4, 4),
      Q : ℝ × ℝ := (C2_x θ, C2_y θ),
      M : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) in
  let d : ℝ := (Real.sqrt 5 / 5) * Real.abs (4 * Real.cos θ - 3 * Real.sin θ - 13) in
  ∃ θ : ℝ, Real.cos θ = 4 / 5 ∧ Real.sin θ = -3 / 5 → d = 8 * Real.sqrt 5 / 5 :=
sorry

end min_distance_midpoint_l801_801817


namespace elongation_rate_improvement_l801_801285

-- Definitions of x_i and y_i
def xi : ℕ → ℝ
| 1 := 545 | 2 := 533 | 3 := 551 | 4 := 522 | 5 := 575 
| 6 := 544 | 7 := 541 | 8 := 568 | 9 := 596 | 10 := 548

def yi : ℕ → ℝ
| 1 := 536 | 2 := 527 | 3 := 543 | 4 := 530 | 5 := 560 
| 6 := 533 | 7 := 522 | 8 := 550 | 9 := 576 | 10 := 536

-- Definition of zi
def zi (i : ℕ) : ℝ := xi i - yi i

-- Mean and variance calculation for {z_i}
def mean_z : ℝ := (∑ i in Finset.range 10, zi (i + 1)) / 10

def variance_z : ℝ := (∑ i in Finset.range 10, (zi (i + 1) - mean_z) ^ 2) / 10

-- Significant improvement criterion
def significant_improvement : Prop := mean_z ≥ 2 * Real.sqrt (variance_z / 10)

theorem elongation_rate_improvement : 
  mean_z = 11 ∧ variance_z = 61 ∧ significant_improvement :=
by
  -- The proof steps would be inserted here.
  sorry

end elongation_rate_improvement_l801_801285


namespace prove_inequality_l801_801769

noncomputable def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d
noncomputable def geometric_sequence (b : ℕ → ℝ) := ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, b (n + 1) = b n * q

theorem prove_inequality 
  {a b : ℕ → ℝ}
  (h1 : arithmetic_sequence a)
  (h2 : geometric_sequence b)
  (h3 : a 11 = b 10) :
  a 13 + a 9 ≤ b 14 + b 6 :=
begin
  sorry
end

end prove_inequality_l801_801769


namespace num_possible_values_of_M_l801_801749

theorem num_possible_values_of_M :
  ∃ n : ℕ, n = 8 ∧
  ∃ (a b : ℕ), (10 <= 10*a + b) ∧ (10*a + b < 100) ∧ (9*(a - b) ∈ {k : ℕ | ∃ m : ℕ, k = m^2}) := sorry

end num_possible_values_of_M_l801_801749


namespace equivalent_function_l801_801959

theorem equivalent_function :
  (∀ x : ℝ, (76 * x ^ 6) ^ 7 = |x|) :=
by
  sorry

end equivalent_function_l801_801959


namespace probability_x_div_y_integer_l801_801607

def x_set : Set ℕ := {78, 910}
def y_set : Set ℕ := {23, 45}

theorem probability_x_div_y_integer : 
  (∃ (x ∈ x_set) (y ∈ y_set), (x / y : ℚ).den = 1) = false :=
by
  sorry

end probability_x_div_y_integer_l801_801607


namespace sum_of_first_4_terms_arithmetic_sequence_l801_801199

variable {a : ℕ → ℝ}

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d a1, (∀ n, a n = a1 + n * d) ∧ (a 3 - a 1 = 2) ∧ (a 5 = 5)

-- Define the sum S4 for the first 4 terms of the sequence
def sum_first_4_terms (a : ℕ → ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3

-- Define the Lean statement for the problem
theorem sum_of_first_4_terms_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a → sum_first_4_terms a = 10 :=
by
  sorry

end sum_of_first_4_terms_arithmetic_sequence_l801_801199


namespace magnitude_of_complex_l801_801443

theorem magnitude_of_complex (z : ℂ) (h : complex.I * z = 3 - 4 * complex.I) : complex.abs z = 5 :=
sorry

end magnitude_of_complex_l801_801443


namespace sequence_value_proof_l801_801856

theorem sequence_value_proof : 
  (∃ (a : ℕ → ℕ), 
    a 1 = 2 ∧ 
    (∀ n : ℕ, a (2 * n) = 2 * n * a n) ∧ 
    a (2^50) = 2^1276) :=
sorry

end sequence_value_proof_l801_801856


namespace domain_of_f_decreasing_on_interval_range_of_f_l801_801774

noncomputable def f (x : ℝ) : ℝ := Real.log (3 + 2 * x - x^2) / Real.log 2

theorem domain_of_f :
  ∀ x : ℝ, (3 + 2 * x - x^2 > 0) ↔ (-1 < x ∧ x < 3) :=
by
  sorry

theorem decreasing_on_interval :
  ∀ (x₁ x₂ : ℝ), (1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3) →
  f x₂ < f x₁ :=
by
  sorry

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, -1 < x ∧ x < 3 ∧ y = f x) ↔ y ≤ 2 :=
by
  sorry

end domain_of_f_decreasing_on_interval_range_of_f_l801_801774
