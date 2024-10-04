import Mathlib
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Prime
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Combinatorics
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Zmod.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Init.Data.Int.Basic
import Mathlib.Logic.Basic
import Mathlib.NumberTheory.Prime
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic

namespace maximize_area_CDFE_l331_331707

-- Define the problem conditions
def is_square (A B C D : Point) (s : ℝ) : Prop :=
  distance A B = s ∧ distance B C = s ∧ distance C D = s ∧ distance D A = s ∧ 
  distance A C = distance B D

def midpoint (A B : Point) : Point := (A + B) / 2

-- Main statement
theorem maximize_area_CDFE 
  {A B C D E F : Point} 
  (h_square : is_square A B C D 1) 
  (hE : E = midpoint A B) 
  (hF : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ F = A + x • (D - A)) : 
  ∃ x : ℝ, x = 0 ∧ maximizes (area C D F E) :=
sorry

end maximize_area_CDFE_l331_331707


namespace sasha_eventually_writes_div_by_101_l331_331778

theorem sasha_eventually_writes_div_by_101 (n : ℕ) (hn : n >= 10^2018 ∧ n < 10^2019) :
  ∃ m, (│concat_numbers_from n│ = m) ∧ m % 101 = 0 :=
by sorry

-- Helper function to define the concatenation of numbers from a given starting number
noncomputable def concat_numbers_from (n : ℕ) : ℕ :=
  sorry -- To be defined

end sasha_eventually_writes_div_by_101_l331_331778


namespace problem_statement_l331_331425

noncomputable def g : ℝ → ℝ := sorry

theorem problem_statement 
  (h : ∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - x * y^2 - x + 2) :
  ∃ (m t : ℕ), (m = 1) ∧ (t = 3) ∧ (m * t = 3) :=
sorry

end problem_statement_l331_331425


namespace village_lasts_6_weeks_l331_331568

def vampire_people_per_week (leader : ℕ) (others : ℕ) (vampires : ℕ) : ℕ :=
  leader + others * vampires

def werewolf_people_per_week (alpha : ℕ) (others : ℕ) (werewolves : ℕ) : ℕ :=
  alpha + others * werewolves

def ghost_people_per_week (ghost_rate : ℕ) : ℕ := ghost_rate
def witch_people_per_week (witches : ℕ) (sacrifice_rate : ℕ) : ℕ := witches * sacrifice_rate
def zombie_people_per_week (zombies : ℕ) (zombie_rate : ℕ) : ℕ := zombies * zombie_rate

def total_people_per_week (vampire_total : ℕ) (werewolf_total : ℕ) (ghost_total : ℕ) (witch_total : ℕ) (zombie_total : ℕ) : ℕ :=
  vampire_total + werewolf_total + ghost_total + witch_total + zombie_total

def village_weeks (village_population : ℕ) (consumption_per_week : ℕ) : ℕ :=
  village_population / consumption_per_week

theorem village_lasts_6_weeks : village_weeks 500 72 = 6 :=
by
  have vampire_total := vampire_people_per_week 5 5 3
  have werewolf_total := werewolf_people_per_week 7 5 4
  have ghost_total := ghost_people_per_week 2
  have witch_total := witch_people_per_week 4 3
  have zombie_total := zombie_people_per_week 20 1
  have consumption_per_week := total_people_per_week vampire_total werewolf_total ghost_total witch_total zombie_total
  exact village_weeks 500 consumption_per_week = 6

end village_lasts_6_weeks_l331_331568


namespace lines_are_skew_iff_l331_331948

def line1 (s : ℝ) (b : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 3 * s, 3 + 4 * s, b + 5 * s)

def line2 (v : ℝ) : ℝ × ℝ × ℝ :=
  (5 + 6 * v, 2 + 3 * v, 1 + 2 * v)

def lines_intersect (s v b : ℝ) : Prop :=
  line1 s b = line2 v

theorem lines_are_skew_iff (b : ℝ) : ¬ (∃ s v, lines_intersect s v b) ↔ b ≠ 9 :=
by
  sorry

end lines_are_skew_iff_l331_331948


namespace geometric_sum_S5_l331_331411

variable (a_n : ℕ → ℝ)
variable (S : ℕ → ℝ)

def geometric_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a_n (n+1) = a_n n * q

theorem geometric_sum_S5 (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (h_geom : geometric_sequence a_n)
  (h_cond1 : a_n 2 * a_n 3 = 8 * a_n 1)
  (h_cond2 : (a_n 4 + 2 * a_n 5) / 2 = 20) :
  S 5 = 31 :=
sorry

end geometric_sum_S5_l331_331411


namespace JB_eq_JM_l331_331262

variables {M A B C D E O J : Type}
variables [InnerProductSpace ℝ M]
variables [NormedSpace ℝ B] [NormedSpace ℝ C]
variables [hMAB_right_angle : IsOrtho A M B]
variables (ω : Circumcircle M A B) 

-- The setup and conditions
-- 1. Triangle MAB is right-angled at A
def triangle_MAB_right_angle : Prop :=
  ∃ (M A B : Type) [InnerProductSpace ℝ M], IsOrtho A M B

-- 2. ω is the circumcircle of triangle MAB
def is_circumcircle (ω : Type) : Prop :=
  is_circumcircle_of_triangle ω M A B

-- 3. Chord CD is perpendicular to AB
def chord_CD_perpendicular_AB (CD AB : Type) : Prop :=
  is_perpendicular CD AB

-- 4. Points A, C, B, D, M lie on ω in this order
def points_on_circle (A C B D M : Type) (ω : Type) : Prop :=
  points_on_circle_in_order A C B D M ω

-- 5. AC and MD intersect at point E
def intersect_AC_MD_at_E (AC MD E : Type) : Prop :=
  intersection_point AC MD = E

-- 6. O is the circumcenter of triangle EMC
def circumcenter_of_triangle_EMC (O E M C : Type) : Prop :=
  is_circumcenter_of_triangle O E M C

-- 7. J is the intersection of BC and OM
def intersection_of_BC_OM (J B C O M : Type) : Prop :=
  intersection_point BC OM = J

-- The main goal: Show JB = JM if the conditions hold
theorem JB_eq_JM (M A B C D E O J : Type) [InnerProductSpace ℝ M] [NormedSpace ℝ B] [NormedSpace ℝ C]
  (h1 : triangle_MAB_right_angle M A B)
  (h2 : is_circumcircle ω M A B)
  (h3 : chord_CD_perpendicular_AB C D A B)
  (h4 : points_on_circle A C B D M ω)
  (h5 : intersect_AC_MD_at_E A C M D E)
  (h6 : circumcenter_of_triangle_EMC O E M C)
  (h7 : intersection_of_BC_OM J B C O M) :
  distance J B = distance J M := 
sorry

end JB_eq_JM_l331_331262


namespace xy_square_diff_l331_331050

theorem xy_square_diff (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end xy_square_diff_l331_331050


namespace plums_added_l331_331534

-- Definitions of initial and final plum counts
def initial_plums : ℕ := 17
def final_plums : ℕ := 21

-- The mathematical statement to be proved
theorem plums_added (initial_plums final_plums : ℕ) : final_plums - initial_plums = 4 := by
  -- The proof will be inserted here
  sorry

end plums_added_l331_331534


namespace forty_percent_of_n_l331_331760

theorem forty_percent_of_n (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 16) : 0.40 * N = 384 :=
by
  sorry

end forty_percent_of_n_l331_331760


namespace find_a_plus_b_l331_331427

def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem find_a_plus_b
  (a b : ℝ)
  (z1 : ℂ := ⟨sqrt 3 * a - 1, sqrt 3 - b⟩)
  (z2 : ℂ := ⟨2 - sqrt 3 * a, b⟩)
  (h1 : abs z1 = abs z2)
  (h2 : is_purely_imaginary (z1 * conj z2)) :
  a + b = sqrt 3 + 1 ∨ a + b = sqrt 3 - 1 :=
sorry

end find_a_plus_b_l331_331427


namespace rationalize_denominator_l331_331775

theorem rationalize_denominator :
  let a := (∛5 : ℝ)
  let b := (∛3 : ℝ)
  let expr := 1 / (a - b)
  let num := a^2 + a*b + b^2
  ∃ A B C D : ℝ,
    expr * num = (∛A + ∛B + ∛C) / D ∧
    A = 25 ∧ B = 15 ∧ C = 9 ∧ D = 2 ∧
    A + B + C + D = 51 :=
by {
  let a := (∛5 : ℝ)
  let b := (∛3 : ℝ)
  let expr := 1 / (a - b)
  let num := a^2 + a*b + b^2

  exists (25 : ℝ)
  exists (15 : ℝ)
  exists (9 : ℝ)
  exists (2 : ℝ)

  split
  { sorry, }
  { split
    { exact rfl, }
    { split
      { exact rfl, }
      { split
        { exact rfl, }
        { split
          { exact rfl, }
          { norm_num }}}}
}

end rationalize_denominator_l331_331775


namespace trigonometric_identity_l331_331983

open Real

theorem trigonometric_identity (α : ℝ) (h1 : cos α = -4 / 5) (h2 : π < α ∧ α < (3 * π / 2)) :
    (1 + tan (α / 2)) / (1 - tan (α / 2)) = -1 / 2 := by
  sorry

end trigonometric_identity_l331_331983


namespace divisible_by_6_and_sum_15_l331_331058

theorem divisible_by_6_and_sum_15 (A B : ℕ) (h1 : A + B = 15) (h2 : (10 * A + B) % 6 = 0) :
  (A * B = 56) ∨ (A * B = 54) :=
by sorry

end divisible_by_6_and_sum_15_l331_331058


namespace correct_statement_D_l331_331185

-- Conditions expressed as definitions
def candidates_selected_for_analysis : ℕ := 500

def statement_A : Prop := candidates_selected_for_analysis = 500
def statement_B : Prop := "The mathematics scores of the 500 candidates selected are the sample size."
def statement_C : Prop := "The 500 candidates selected are individuals."
def statement_D : Prop := "The mathematics scores of all candidates in the city's high school entrance examination last year constitute the population."

-- Problem statement in Lean
theorem correct_statement_D :
  statement_D := sorry

end correct_statement_D_l331_331185


namespace liam_balloons_remainder_l331_331114

def balloons : Nat := 24 + 45 + 78 + 96
def friends : Nat := 10
def remainder := balloons % friends

theorem liam_balloons_remainder : remainder = 3 := by
  sorry

end liam_balloons_remainder_l331_331114


namespace gcd_lcm_sum_l331_331523

theorem gcd_lcm_sum :
  ∀ (a b c d : ℕ), gcd a b + lcm c d = 74 :=
by
  let a := 42
  let b := 70
  let c := 20
  let d := 15
  sorry

end gcd_lcm_sum_l331_331523


namespace total_percentage_failed_exam_l331_331698

theorem total_percentage_failed_exam :
  let total_candidates := 2000
  let general_candidates := 1000
  let obc_candidates := 600
  let sc_candidates := 300
  let st_candidates := total_candidates - (general_candidates + obc_candidates + sc_candidates)
  let general_pass_percentage := 0.35
  let obc_pass_percentage := 0.50
  let sc_pass_percentage := 0.25
  let st_pass_percentage := 0.30
  let general_failed := general_candidates - (general_candidates * general_pass_percentage)
  let obc_failed := obc_candidates - (obc_candidates * obc_pass_percentage)
  let sc_failed := sc_candidates - (sc_candidates * sc_pass_percentage)
  let st_failed := st_candidates - (st_candidates * st_pass_percentage)
  let total_failed := general_failed + obc_failed + sc_failed + st_failed
  let failed_percentage := (total_failed / total_candidates) * 100
  failed_percentage = 62.25 :=
by
  sorry

end total_percentage_failed_exam_l331_331698


namespace max_total_toads_l331_331181

variable (x y : Nat)
variable (frogs total_frogs : Nat)
variable (total_toads : Nat)

def pond1_frogs := 3 * x
def pond1_toads := 4 * x
def pond2_frogs := 5 * y
def pond2_toads := 6 * y

def all_frogs := pond1_frogs x + pond2_frogs y
def all_toads := pond1_toads x + pond2_toads y

theorem max_total_toads (h_frogs : all_frogs x y = 36) : all_toads x y = 46 := 
sorry

end max_total_toads_l331_331181


namespace solve_for_y_l331_331792

theorem solve_for_y (y : ℝ) :
  (1 / 8) ^ (3 * y + 12) = (32) ^ (3 * y + 7) → y = -71 / 24 :=
by
  intro h
  sorry

end solve_for_y_l331_331792


namespace unit_distance_path_length_at_least_l331_331923

/--An infinite tree graph with edges connecting lattice points such that
the distance between two connected points is at most 1998. Every lattice point
in the plane is a vertex of the graph. Prove that there exists a pair
of points in the plane at a unit distance from each other that are connected
by a path of length at least 10^1998.--/
theorem unit_distance_path_length_at_least :
  ∃ (A B : ℤ × ℤ), dist A B = 1 ∧ 
  ∃ (p : List (ℤ × ℤ)), List.Chain (λ u v, dist u v ≤ 1998) p ∧
  List.head p = A ∧ List.last p (by simp) = B ∧ p.length ≥ 10^1998 := sorry

end unit_distance_path_length_at_least_l331_331923


namespace students_liked_strawberries_l331_331140

theorem students_liked_strawberries : 
  let total_students := 450 
  let students_oranges := 70 
  let students_pears := 120 
  let students_apples := 147 
  let students_strawberries := total_students - (students_oranges + students_pears + students_apples)
  students_strawberries = 113 :=
by
  sorry

end students_liked_strawberries_l331_331140


namespace min_t_condition_l331_331330

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 2 * x
def I : Set ℝ := Set.Icc (-Real.pi) Real.pi

theorem min_t_condition :
  ∀ t : ℝ, (∀ x1 x2 ∈ I, |f x1 - f x2| ≤ t) ↔ t ≥ 4 * Real.pi := by
  sorry

end min_t_condition_l331_331330


namespace jose_investment_l331_331507

theorem jose_investment (P T : ℝ) (X : ℝ) (months_tom months_jose : ℝ) (profit_total profit_jose profit_tom : ℝ) :
  T = 30000 →
  months_tom = 12 →
  months_jose = 10 →
  profit_total = 54000 →
  profit_jose = 30000 →
  profit_tom = profit_total - profit_jose →
  profit_tom / profit_jose = (T * months_tom) / (X * months_jose) →
  X = 45000 :=
by sorry

end jose_investment_l331_331507


namespace house_orderings_l331_331701

/-- Ralph walks past five houses each painted in a different color: 
orange, red, blue, yellow, and green.
Conditions:
1. Ralph passed the orange house before the red house.
2. Ralph passed the blue house before the yellow house.
3. The blue house was not next to the yellow house.
4. Ralph passed the green house before the red house and after the blue house.
Given these conditions, prove that there are exactly 3 valid orderings of the houses.
-/
theorem house_orderings : 
  ∃ (orderings : Finset (List String)), 
  orderings.card = 3 ∧
  (∀ (o : List String), 
   o ∈ orderings ↔ 
    ∃ (idx_o idx_r idx_b idx_y idx_g : ℕ), 
    o = ["orange", "red", "blue", "yellow", "green"] ∧
    idx_o < idx_r ∧ 
    idx_b < idx_y ∧ 
    (idx_b + 1 < idx_y ∨ idx_y + 1 < idx_b) ∧ 
    idx_b < idx_g ∧ idx_g < idx_r) := sorry

end house_orderings_l331_331701


namespace part_A_part_B_part_C_l331_331002

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f(x * y) = y^2 * f(x) + x^2 * f(y)

theorem part_A : f(0) = 0 := sorry
theorem part_B : f(1) = 0 := sorry
theorem part_C : ∀ x : ℝ, f(x) = f(-x) := sorry

end part_A_part_B_part_C_l331_331002


namespace parabola_b_value_l331_331834

theorem parabola_b_value (a b c h k : ℝ) (hk : k ≠ 0)
  (vertex_form : ∀ x : ℝ, (a (x - h) ^ 2 + k) = ax^2 + bx + c)
  (passes_point : a (0 - h) ^ 2 + k = -k) :
  b = 4 * k / h :=
by
  sorry

end parabola_b_value_l331_331834


namespace range_of_m_solve_inequality_l331_331335

-- Part (1) proof problem statement
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) :
  (∀ x ∈ set.Icc 1 2, f x = (2 * m + 1) * x ^ 2 - m * x + 2 * m - 1) →
  (monotone_on f (set.Icc 1 2)) →
  m ∈ set.Icc (-Float.ofRat (4 / 7)) Float.infinity ∪ set.Icc Float.negInfinity (-Float.ofRat (2 / 3)) :=
sorry

--Part (2) proof problem statement
theorem solve_inequality (f : ℝ → ℝ) (m : ℝ) :
  (m ≤ -1) →
  (f = λ x, (2 * m + 1) * x ^ 2 - m * x + 2 * m - 1) →
  (∀ x : ℝ,
     if m < -2 then (-1 / (m + 1) ≤ x ∧ x ≤ 1) else
     if m = -2 then x = 1 else
     if -2 < m ∧ m < -1 then (1 ≤ x ∧ x ≤ -1 / (m + 1)) else
     if m = -1 then (1 ≤ x) else
     false) :=
sorry

end range_of_m_solve_inequality_l331_331335


namespace permutation_condition_l331_331981

theorem permutation_condition (n : ℕ) :
  (∃ (x : Fin n → Fin n), (∀ k, x k ≠ k) ∧ (∀ i j, i ≠ j → |x i - i| ≠ |x j - j|)) ↔ (n % 4 = 0 ∨ n % 4 = 1) :=
by
  sorry

end permutation_condition_l331_331981


namespace jean_calories_consumed_l331_331722

/-- Jean writes 20 pages and eats one donut per 3 pages, 
    and each donut has 180 calories. 
    We want to prove that she will consume 1260 calories. -/
theorem jean_calories_consumed : 
  (20 / 3).ceil * 180 = 1260 := 
by
  sorry

end jean_calories_consumed_l331_331722


namespace rhombus_construction_exists_l331_331989

theorem rhombus_construction_exists
  (quad : Type)
  (side_x side_y side_s side_r diag_l diag_k diag_p diag_q : ℝ)
  (hx : 0 < side_x) (hy : 0 < side_y) (hs : 0 < side_s) (hr : 0 < side_r)
  (hl : 0 < diag_l) (hk : 0 < diag_k) (hp : 0 < diag_p) (hq : 0 < diag_q)
  (parallel_sides : side_x / side_y = diag_l / diag_k ∧ side_r / side_s = diag_p / diag_q) :
  ∃ (rhombus : quad), 
    (rhombus ∈ quadrilateral) ∧ 
    (vertices rhombus Lie on Sides quad) ∧ 
    (sides rhombus parallel to diagonals quad) ∧ 
    (all_sides_equal rhombus) :=
sorry

end rhombus_construction_exists_l331_331989


namespace simplify_G_l331_331947

def F (x : ℝ) : ℝ := log ((1 + x) / (1 - x))
def y (x : ℝ) : ℝ := (4 * x + x^4) / (1 + 4 * x^3)
def G (x : ℝ) : ℝ := log ((1 + y x) / (1 - y x))

theorem simplify_G (x : ℝ) : G x = 4 * F x := sorry

end simplify_G_l331_331947


namespace probability_two_fives_from_twelve_dice_l331_331509

/-- 
Problem: What is the probability that exactly two of the dice show a 5? 
Conditions: Twelve standard 6-sided dice are rolled.
Answer: The probability is 0.179, rounded to the nearest thousandth.
-/

theorem probability_two_fives_from_twelve_dice : 
  let p : ℚ := (66 * (1 / 6) ^ 2 * (5 / 6) ^ 10 : ℚ)
  in p ≈ 0.179 := 
by 
  sorry


end probability_two_fives_from_twelve_dice_l331_331509


namespace final_amoeba_is_blue_l331_331699

-- We define the initial counts of each type of amoeba
def initial_red : ℕ := 26
def initial_blue : ℕ := 31
def initial_yellow : ℕ := 16

-- We define the final count of amoebas
def final_amoebas : ℕ := 1

-- The type of the final amoeba (we're proving it's 'blue')
inductive AmoebaColor
| Red
| Blue
| Yellow

-- Given initial counts, we aim to prove the final amoeba is blue
theorem final_amoeba_is_blue :
  initial_red = 26 ∧ initial_blue = 31 ∧ initial_yellow = 16 ∧ final_amoebas = 1 → 
  ∃ c : AmoebaColor, c = AmoebaColor.Blue :=
by sorry

end final_amoeba_is_blue_l331_331699


namespace AI_parallel_HO_iff_angle_BAC_120_l331_331728

noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def centroid (A B C : Point) : Point := sorry
noncomputable def circumcenter (A B C : Point) : Point := sorry

-- Points A, B, C are the vertices of the non-isosceles triangle ABC
variables (A B C : Point)

-- Definitions of I, H, O using the conditions
let I := incenter A B C
let H := centroid A B C
let O := circumcenter A B C

-- Non-isosceles triangle condition
axiom non_isosceles : ¬ is_isosceles_triangle A B C

-- The statement to be proven
theorem AI_parallel_HO_iff_angle_BAC_120 :
  ∠BAC = 120° ↔ line_through' A I ∥ line_through' H O :=
by
  -- Geometric proof to be filled
  sorry

end AI_parallel_HO_iff_angle_BAC_120_l331_331728


namespace area_ratio_of_point_in_triangle_l331_331403

variables {A B C P : Type} [AffSpace A]

noncomputable def vector_eq_condition (PA PB PC : Vec A) : Vec A :=
PA + 3 • PB + 4 • PC

theorem area_ratio_of_point_in_triangle
  (A B C P : Vec A)
  (h : vector_eq_condition (A - P) (B - P) (C - P) = 0) :
  (area_of_triangle A B C) / (area_of_triangle A P B) = 5 / 2 := 
sorry

end area_ratio_of_point_in_triangle_l331_331403


namespace list_of_21_numbers_l331_331894

theorem list_of_21_numbers (numbers : List ℝ) (n : ℝ) (h_length : numbers.length = 21) 
  (h_mem : n ∈ numbers) 
  (h_n_avg : n = 4 * (numbers.sum - n) / 20) 
  (h_n_sum : n = (numbers.sum) / 6) : numbers.length - 1 = 20 :=
by
  -- We provide the statement with the correct hypotheses
  -- the proof is yet to be filled in
  sorry

end list_of_21_numbers_l331_331894


namespace abs_expr_evaluation_l331_331291

theorem abs_expr_evaluation : abs (abs (-abs (-1 + 2) - 2) + 3) = 6 := by
  sorry

end abs_expr_evaluation_l331_331291


namespace series_sum_eq_14_div_15_l331_331279

theorem series_sum_eq_14_div_15 : 
  (∑ n in Finset.range 14, (1 : ℚ) / ((n + 1) * (n + 2))) = 14 / 15 := 
by
  sorry

end series_sum_eq_14_div_15_l331_331279


namespace probability_of_players_obtaining_odd_sum_l331_331189

theorem probability_of_players_obtaining_odd_sum :
  let total_ways := (Nat.choose 12 4) * (Nat.choose 8 4) * (Nat.choose 4 4)
  let valid_ways := 1350 + 90
  let prob := (valid_ways : ℚ) / total_ways
  let m := 16
  let n := 385
  (m + n = 401) :=
by
  let total_ways := (Nat.choose 12 4) * (Nat.choose 8 4) * (Nat.choose 4 4)
  let valid_ways := 1350 + 90
  let prob := (valid_ways : ℚ) / total_ways
  let m := 16
  let n := 385
  have prob_eq : prob = (16 / 385 : ℚ) := sorry
  have gcd_mn : Nat.gcd 16 385 = 1 := by decide
  have co_prime : (16 / 385).denom = 385 := by apply Rat.denom_eq_of_coprime; assumption
  show m + n = 401
  sorry

end probability_of_players_obtaining_odd_sum_l331_331189


namespace xy_square_diff_l331_331056

theorem xy_square_diff (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end xy_square_diff_l331_331056


namespace solve_for_x_l331_331483

noncomputable def infinite_power_tower (x : ℝ) : ℝ :=
  x^(x^(x^...))

theorem solve_for_x :
  infinite_power_tower x = 4 ↔ x = real.sqrt 2 :=
sorry

end solve_for_x_l331_331483


namespace ratio_of_fixing_times_is_two_l331_331719

noncomputable def time_per_shirt : ℝ := 1.5
noncomputable def number_of_shirts : ℕ := 10
noncomputable def number_of_pants : ℕ := 12
noncomputable def hourly_rate : ℝ := 30
noncomputable def total_cost : ℝ := 1530

theorem ratio_of_fixing_times_is_two :
  let total_hours := total_cost / hourly_rate
  let shirt_hours := number_of_shirts * time_per_shirt
  let pant_hours := total_hours - shirt_hours
  let time_per_pant := pant_hours / number_of_pants
  (time_per_pant / time_per_shirt) = 2 :=
by
  sorry

end ratio_of_fixing_times_is_two_l331_331719


namespace locker_number_problem_l331_331839

theorem locker_number_problem 
  (cost_per_digit : ℝ)
  (total_cost : ℝ)
  (one_digit_cost : ℝ)
  (two_digit_cost : ℝ)
  (three_digit_cost : ℝ) :
  cost_per_digit = 0.03 →
  one_digit_cost = 0.27 →
  two_digit_cost = 5.40 →
  three_digit_cost = 81.00 →
  total_cost = 206.91 →
  10 * cost_per_digit = six_cents →
  9 * cost_per_digit = three_cents →
  1 * 9 * cost_per_digit = one_digit_cost →
  2 * 45 * cost_per_digit = two_digit_cost →
  3 * 300 * cost_per_digit = three_digit_cost →
  (999 * 3 + x * 4 = 6880) →
  ∀ total_locker : ℕ, total_locker = 2001 := sorry

end locker_number_problem_l331_331839


namespace max_dot_product_val_l331_331410

noncomputable def max_dot_product 
  (a b c : EuclideanSpace ℝ (Fin 2)) 
  (ha : ∥a∥ = 1) 
  (hb : ∥b∥ = 1) 
  (hc : ∥c∥ = 1) 
  (h_angle_ab : real.angle a b = real.pi / 3) : ℝ := 
  let term := (c - a) • (c - b) in 
  term

theorem max_dot_product_val 
  (a b c : EuclideanSpace ℝ (Fin 2)) 
  (ha : ∥a∥ = 1) 
  (hb : ∥b∥ = 1) 
  (hc : ∥c∥ = 1) 
  (h_angle_ab : real.angle a b = real.pi / 3) :
  max_dot_product a b c ha hb hc h_angle_ab = (3 / 2 + real.sqrt 3) :=
sorry

end max_dot_product_val_l331_331410


namespace part_A_part_B_part_C_l331_331004

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f(x * y) = y^2 * f(x) + x^2 * f(y)

theorem part_A : f(0) = 0 := sorry
theorem part_B : f(1) = 0 := sorry
theorem part_C : ∀ x : ℝ, f(x) = f(-x) := sorry

end part_A_part_B_part_C_l331_331004


namespace rationalize_denominator_l331_331774

theorem rationalize_denominator :
  let a := (∛5 : ℝ)
  let b := (∛3 : ℝ)
  let expr := 1 / (a - b)
  let num := a^2 + a*b + b^2
  ∃ A B C D : ℝ,
    expr * num = (∛A + ∛B + ∛C) / D ∧
    A = 25 ∧ B = 15 ∧ C = 9 ∧ D = 2 ∧
    A + B + C + D = 51 :=
by {
  let a := (∛5 : ℝ)
  let b := (∛3 : ℝ)
  let expr := 1 / (a - b)
  let num := a^2 + a*b + b^2

  exists (25 : ℝ)
  exists (15 : ℝ)
  exists (9 : ℝ)
  exists (2 : ℝ)

  split
  { sorry, }
  { split
    { exact rfl, }
    { split
      { exact rfl, }
      { split
        { exact rfl, }
        { split
          { exact rfl, }
          { norm_num }}}}
}

end rationalize_denominator_l331_331774


namespace polynomial_remainder_l331_331025

-- Define the given polynomials
def p (x : ℝ) : ℝ := x^3 + 2 * x^2 + 3
def d (x : ℝ) : ℝ := x^2 - 2 * x + 4

-- Define the expected remainder
def remainder (x : ℝ) : ℝ := 4 * x - 13

-- State that the remainder when p(x) is divided by d(x) is remainder(x)
theorem polynomial_remainder : ∀ x : ℝ, polynomial.div_mod_by_monic p d x.snd = remainder x :=
sorry

end polynomial_remainder_l331_331025


namespace minimum_cost_for_Dorokhov_family_vacation_l331_331466

theorem minimum_cost_for_Dorokhov_family_vacation :
  let globus_cost := (25400 * 3) * 0.98,
      around_world_cost := (11400 + (23500 * 2)) * 1.01
  in
  min globus_cost around_world_cost = 58984 := by
  let globus_cost := (25400 * 3) * 0.98
  let around_world_cost := (11400 + (23500 * 2)) * 1.01
  sorry

end minimum_cost_for_Dorokhov_family_vacation_l331_331466


namespace shobha_current_age_l331_331928

theorem shobha_current_age (S B : ℕ) (h1 : S / B = 4 / 3) (h2 : S + 6 = 26) : B = 15 :=
by
  -- Here we would begin the proof
  sorry

end shobha_current_age_l331_331928


namespace blankets_avg_price_l331_331905

def average_price_blankets (cost1 : ℕ) (quantity1 : ℕ) (cost2 : ℕ) (quantity2 : ℕ) (total_cost_unknown : ℕ) (total_number_blankets : ℕ) : ℕ :=
  (cost1 * quantity1 + cost2 * quantity2 + total_cost_unknown) / total_number_blankets

theorem blankets_avg_price : 
  average_price_blankets 100 1 150 5 650 8 = 187.50 :=
by 
  sorry

end blankets_avg_price_l331_331905


namespace solution_correct_l331_331413

variable (a b c d : ℝ)

theorem solution_correct (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end solution_correct_l331_331413


namespace flushes_per_day_l331_331098

-- Definitions based on the conditions provided
def old_toilet_gallons_per_flush := 5
def new_toilet_ratio := 0.2
def water_saved_in_june := 1800
def days_in_june := 30

-- Theorem statement
theorem flushes_per_day (old_toilet_gallons_per_flush : ℕ) (new_toilet_ratio : ℚ) 
                        (water_saved_in_june : ℕ) (days_in_june : ℕ) : 
  (water_saved_in_june / (old_toilet_gallons_per_flush - old_toilet_gallons_per_flush * new_toilet_ratio.to_nat))/(days_in_june) = 15 :=
by
  sorry

end flushes_per_day_l331_331098


namespace solve_for_x_l331_331785

theorem solve_for_x : ∀ (x : ℤ), (5 * x - 2) * 4 = (3 * (6 * x - 6)) → x = -5 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l331_331785


namespace seaweed_percentage_l331_331937

theorem seaweed_percentage (x : ℝ) : 
  ∃ x, 
  let harvested := 400
  let livestock := 150
  let edible_percent := 0.25 
  let livestock_percent := 0.75 in
  0.75 * (1 - x / 100) * harvested = livestock  → 
  x = 50 :=
by {
  sorry
}

end seaweed_percentage_l331_331937


namespace meaningful_sqrt_x_minus_5_l331_331362

theorem meaningful_sqrt_x_minus_5 (x : ℝ) (h : sqrt (x - 5) ∈ ℝ) : x = 6 ∨ x ≥ 5 := by
  sorry

end meaningful_sqrt_x_minus_5_l331_331362


namespace max_a_plus_b_l331_331104

/-- Given real numbers a and b such that 5a + 3b <= 11 and 3a + 6b <= 12,
    the largest possible value of a + b is 23/9. -/
theorem max_a_plus_b (a b : ℝ) (h1 : 5 * a + 3 * b ≤ 11) (h2 : 3 * a + 6 * b ≤ 12) :
  a + b ≤ 23 / 9 :=
sorry

end max_a_plus_b_l331_331104


namespace product_multiple_of_four_probability_l331_331955

theorem product_multiple_of_four_probability :
  let box := [1, 2, 4, 5] in
  (∃ (chip1 chip2 : ℕ), chip1 ∈ box ∧ chip2 ∈ box ∧ (chip1 * chip2) % 4 = 0) ↔ true :=
by
  sorry

end product_multiple_of_four_probability_l331_331955


namespace speed_of_B_l331_331208

theorem speed_of_B 
  (v_A : ℝ := 7)  -- speed of A in km/h
  (t_A : ℝ := 0.5)  -- time A walks before B starts in hours
  (t_B : ℝ := 1.8)  -- time B walks before overtaking A in hours
  :
  B : ℝ := (v_A * t_A + v_A * t_B) / t_B
  :
  B = 8.944 :=
by
  sorry

end speed_of_B_l331_331208


namespace am_perpendicular_bc_l331_331578

noncomputable def Triangle := Type*
variables (A B C D E F G M : Triangle) (BC : Triangle → Triangle → Triangle)
variables (AB AC : Triangle → Triangle → Triangle) (DG EF : Triangle → Triangle → Triangle)
variables (perpendicular : Triangle → Triangle → Prop)
variables (intersect : Triangle → Triangle → Triangle)
variables (semicircle_diameter : Triangle → Triangle)

theorem am_perpendicular_bc 
  (H1 : BC A B = C)
  (H2 : semicircle_diameter BC = BC)
  (H3 : AB B D = AB)
  (H4 : AC C E = AC)
  (H5 : perpendicular D BC)
  (H6 : perpendicular E BC)
  (H7 : intersect DG EF = M) :
  perpendicular (AB M) BC := sorry

end am_perpendicular_bc_l331_331578


namespace number_of_subsets_of_set_A_eq_4_l331_331671

theorem number_of_subsets_of_set_A_eq_4 : 
  let A := {1, 2} in 
  ∃ x, x = set.powerset A ∧ x.card = 4 :=
by
  let A := {1, 2}
  existsi set.powerset A
  split
  exact rfl
  sorry

end number_of_subsets_of_set_A_eq_4_l331_331671


namespace range_of_a_l331_331412

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = Real.exp x + a * x) ∧ (∃ x, 0 < x ∧ (DifferentiableAt ℝ f x) ∧ (deriv f x = 0)) → a < -1 :=
by
  sorry

end range_of_a_l331_331412


namespace _l331_331912

noncomputable def proof_theorem 
   (S : Type) [MetricSpace S] [NormedAddCommGroup S] [InnerProductSpace ℝ S]
   (sphere : S) (circle_on_sphere : S) (P : S) (h : (P ∉ sphere)) : Prop :=
  ∀ (points_on_circle : Set S), (∀ p ∈ points_on_circle, p ∈ circle_on_sphere) 
  ∃ (circle : Set S), ∀ (p ∈ points_on_circle), 
  let intersection_point := (line_through P p).intersection sphere in
  (intersection_point \ P) ∈ circle

sorry

end _l331_331912


namespace angle_B_l331_331648

/-- 
  Given that the area of triangle ABC is (sqrt 3 / 2) 
  and the dot product of vectors AB and BC is 3, 
  prove that the measure of angle B is 5π/6. 
--/
theorem angle_B (A B C : ℝ) (a c : ℝ) (h1 : 0 ≤ B ∧ B ≤ π)
  (h_area : (1 / 2) * a * c * (Real.sin B) = (Real.sqrt 3 / 2))
  (h_dot : a * c * (Real.cos B) = -3) :
  B = 5 * Real.pi / 6 :=
sorry

end angle_B_l331_331648


namespace problem1_problem2_l331_331781

theorem problem1 : (-1/2 + 2/3 - 1/4) ÷ (-1/24) = 2 :=
  sorry

theorem problem2 : (3 + 1/2) * (-5/7) - (-5/7) * (2 + 1/2) - 5/7 * (-1/2) = -5/14 :=
  sorry

end problem1_problem2_l331_331781


namespace linear_regression_passes_through_centroid_l331_331829

noncomputable def linear_regression (a b : ℝ) (x : ℝ) : ℝ := a + b * x

theorem linear_regression_passes_through_centroid 
  (a b : ℝ) (x_bar y_bar : ℝ) 
  (h_centroid : ∀ (x y : ℝ), (x = x_bar ∧ y = y_bar) → y = linear_regression a b x) :
  linear_regression a b x_bar = y_bar :=
by
  -- proof omitted
  sorry

end linear_regression_passes_through_centroid_l331_331829


namespace part_1_part_2_l331_331309

theorem part_1 (a : ℕ → ℚ) (S : ℕ → ℚ) (h1 : ∀ n, S (n + 1) = 4 * a n - 2) (h2 : a 1 = 2) (n : ℕ) (hn_pos : 0 < n) : 
  a (n + 1) - 2 * a n = 0 :=
sorry

theorem part_2 (a : ℕ → ℚ) (b : ℕ → ℚ) (S : ℕ → ℚ) (h1 : ∀ n, S (n + 1) = 4 * a n - 2) (h2 : a 1 = 2) :
  (∀ n, b n = 1 / (a n * a (n + 1))) → ∀ n, S n = (1/6) * (1 - (1/4)^n) :=
sorry

end part_1_part_2_l331_331309


namespace convex_hull_vertices_ge_100_l331_331842

theorem convex_hull_vertices_ge_100 (n : ℕ) (h : n = 1000) :
  ∀ k (hk1 : k = 100) (hk2 : ∀ i : ℕ, i < n → is_regular_polygon (hk1 ∈ 100)),
  convex_hull_vertices (system_of_polygons n k hk1 hk2) ≥ 100 :=
sorry

end convex_hull_vertices_ge_100_l331_331842


namespace smallest_lcm_not_multiple_of_25_l331_331866

theorem smallest_lcm_not_multiple_of_25 (n : ℕ) (h1 : n % 36 = 0) (h2 : n % 45 = 0) (h3 : n % 25 ≠ 0) : n = 180 := 
by 
  sorry

end smallest_lcm_not_multiple_of_25_l331_331866


namespace error_percentage_approx_l331_331909

theorem error_percentage_approx (x : ℝ) (hx : 0 < x) :
  abs (x^2 - x / 8) / x^2 * 100 ≈ 88 :=
sorry

end error_percentage_approx_l331_331909


namespace part1_part2_l331_331642

def point := ℝ × ℝ × ℝ

def A : point := (0, 2, 3)
def B : point := (-2, 1, 6)
def C : point := (1, -1, 5)

def vector_sub (p1 p2 : point) : point := 
  (p1.1 - p2.1, p1.2 - p2.2, p1.3 - p2.3)

def dot_product (v1 v2 : point) : ℝ := 
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v : point) : ℝ := 
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def area_parallelogram : ℝ :=
  let ab := vector_sub B A
  let ac := vector_sub C A
  magnitude (ab ×ₙ ac)

theorem part1 :
  area_parallelogram = 7 * Real.sqrt 3 :=
sorry

def vector_a (a : ℝ) := (a, a, a)

theorem part2 (a : ℝ) (ha : Real.sqrt (a^2 + a^2 + a^2) = Real.sqrt 3) :
  vector_a a = (1, 1, 1) ∨ vector_a a = (-1, -1, -1) :=
sorry

end part1_part2_l331_331642


namespace min_value_l331_331661

noncomputable def f (a b x : ℝ) := (1 / 3) * a * x^3 + (1 / 2) * b * x^2 - x

theorem min_value (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_min : f a b 1 = (1 / 3) * a * 1^3 + (1 / 2) * b * 1^2 - 1) (h_derive : a + b = 1) :
  (1 / a) + (4 / b) = 9 :=
begin
  sorry
end

end min_value_l331_331661


namespace right_triangle_acute_angle_l331_331700

theorem right_triangle_acute_angle (a b : ℝ) (h1 : a + b = 90) (h2 : a = 55) : b = 35 := 
by sorry

end right_triangle_acute_angle_l331_331700


namespace find_a_l331_331799

def f (x : ℝ) : ℝ := x / 3 + 2
def g (x : ℝ) : ℝ := 5 - 2 * x

theorem find_a (a : ℝ) (h : f (g a) = 4) : a = -1/2 := by
  sorry

end find_a_l331_331799


namespace correctGraph_l331_331696

def workingFromHomePercentage
  (year: ℕ) : ℕ :=
  if year = 1995 then 10
  else if year = 2005 then 18
  else if year = 2015 then 35
  else if year = 2025 then 50
  else sorry

theorem correctGraph :
  (∀ year: ℕ,
  if year = 1995 then workingFromHomePercentage year = 10
  else if year = 2005 then workingFromHomePercentage year = 18
  else if year = 2015 then workingFromHomePercentage year = 35
  else if year = 2025 then workingFromHomePercentage year = 50)
  ∧
  (∀ year: ℕ,
    1995 ≤ year ∧ year ≤ 2025 →
    workingFromHomePercentage year ≥ workingFromHomePercentage (year - 10)) :=
sorry

end correctGraph_l331_331696


namespace remainder_of_power_of_three_l331_331290

open Nat

def lambda_1000 : ℕ := 100
def lambda_100 : ℕ := 20

theorem remainder_of_power_of_three : 3^(3^(3^3)) % 1000 = 387 := by
  have co_prime : gcd 3 1000 = 1 := by simp [gcd]
  have carmichael_property₁ : (3^lambda_1000) % 1000 = 1 := by sorry
  have three_power_27 := 3^(3^3)
  have three_power_27_mod_100 := three_power_27 % 100
  have carmichael_property₂ : (3^lambda_100) % 100 = 1 := by sorry
  have simplified_exponent := 27 % lambda_100
  have simplified_pow := 3^simplified_exponent
  have three_power_27_mod_100_calculated : simplified_pow % 100 = 87 := by sorry
  have final_exponent := 3^three_power_27_mod_100_calculated
  calc
    final_exponent % 1000 = ?...

end remainder_of_power_of_three_l331_331290


namespace carl_returns_to_start_l331_331585

noncomputable def prob_return_after_10_minutes : ℚ :=
  127 / 512

theorem carl_returns_to_start (pentagon_vertex: Type) [fintype pentagon_vertex] [fintype.card pentagon_vertex = 5] :
  probability (Carl randomly selects an adjacent vertex each minute and returns to start after 10 minutes) = prob_return_after_10_minutes := 
  sorry

end carl_returns_to_start_l331_331585


namespace MutualExclusivity_Of_A_C_l331_331619

-- Definitions of events using conditions from a)
def EventA (products : List Bool) : Prop :=
  products.all (λ p => p = true)

def EventB (products : List Bool) : Prop :=
  products.all (λ p => p = false)

def EventC (products : List Bool) : Prop :=
  products.any (λ p => p = false)

-- The main theorem using correct answer from b)
theorem MutualExclusivity_Of_A_C (products : List Bool) :
  EventA products → ¬ EventC products :=
by
  sorry

end MutualExclusivity_Of_A_C_l331_331619


namespace derivative_of_x_exp_x_l331_331153

theorem derivative_of_x_exp_x (x : ℝ) :
  (deriv (λ x : ℝ, x * exp x)) x = (1 + x) * exp x :=
sorry

end derivative_of_x_exp_x_l331_331153


namespace quadrilateral_probability_l331_331275

def sticks : List ℕ := [1, 3, 4, 6, 8, 9, 10, 12]

def isValidSet (s : Finset ℕ) : Prop :=
  s.card = 4 ∧ 
  ∀ (x ∈ s), (s.sum - x) > x

def validCombCount : ℕ := 
  (Finset.powerset (Finset.of_list sticks)).filter (λ s => isValidSet s).card

def totalCombCount : ℕ := Nat.choose 8 4

theorem quadrilateral_probability :
  (validCombCount : ℚ) / totalCombCount = 9 / 14 :=
by
  sorry

end quadrilateral_probability_l331_331275


namespace verify_smallest_x_l331_331865

noncomputable def smallest_positive_integer_for_product_multiple_of_576 : ℕ :=
  let x := 36 in
  x

theorem verify_smallest_x :
  ∃ x : ℕ, x = smallest_positive_integer_for_product_multiple_of_576 ∧ (400 * x) % 576 = 0 :=
by
  use 36
  split
  { refl }
  { show (400 * 36) % 576 = 0
    sorry }

end verify_smallest_x_l331_331865


namespace solve_for_y_l331_331793

theorem solve_for_y (y : ℝ) :
  (1 / 8) ^ (3 * y + 12) = (32) ^ (3 * y + 7) → y = -71 / 24 :=
by
  intro h
  sorry

end solve_for_y_l331_331793


namespace rectangles_containment_existence_l331_331856

theorem rectangles_containment_existence :
  (∃ (rects : ℕ → ℕ × ℕ), (∀ n : ℕ, (rects n).fst > 0 ∧ (rects n).snd > 0) ∧
   (∀ n m : ℕ, n ≠ m → ¬((rects n).fst ≤ (rects m).fst ∧ (rects n).snd ≤ (rects m).snd))) →
  false :=
by
  sorry

end rectangles_containment_existence_l331_331856


namespace triangle_area_ratio_l331_331402

open EuclideanGeometry

theorem triangle_area_ratio (A B C P : Point) 
  (h₁ : Let PA = vector_from P A) 
  (h₂ : Let PB = vector_from P B)
  (h₃ : Let PC = vector_from P C)
  (h₄ : PA + 3 * PB + 4 * PC = 0) :
  let area_ABC := area_of_triangle A B C
  let area_APB := area_of_triangle A P B
  area_ABC / area_APB = 5 / 2 :=
sorry

end triangle_area_ratio_l331_331402


namespace width_of_metallic_sheet_l331_331553

theorem width_of_metallic_sheet 
  (length : ℕ)
  (new_volume : ℕ) 
  (side_length_of_square : ℕ)
  (height_of_box : ℕ)
  (new_length : ℕ)
  (new_width : ℕ)
  (w : ℕ) : 
  length = 48 → 
  new_volume = 5120 → 
  side_length_of_square = 8 → 
  height_of_box = 8 → 
  new_length = length - 2 * side_length_of_square → 
  new_width = w - 2 * side_length_of_square → 
  new_volume = new_length * new_width * height_of_box → 
  w = 36 := 
by 
  intros _ _ _ _ _ _ _ 
  sorry

end width_of_metallic_sheet_l331_331553


namespace rationalize_denominator_correct_l331_331769

noncomputable def rationalize_denominator_sum : ℕ :=
  let a := real.root (5 : ℝ) 3;
  let b := real.root (3 : ℝ) 3;
  let A := real.root (25 : ℝ) 3;
  let B := real.root (15 : ℝ) 3;
  let C := real.root (9 : ℝ) 3;
  let D := 2;
  (25 + 15 + 9 + 2)

theorem rationalize_denominator_correct :
  rationalize_denominator_sum = 51 :=
  by sorry

end rationalize_denominator_correct_l331_331769


namespace zero_probability_divisible_by_15_l331_331694

/--
The sum of the digits 1, 2, 3, 0, 5, and 8 is 19. Since 19 is not divisible by 3,
no number formed by any arrangement of these digits can be divisible by 15.
Therefore, the probability that a number formed by any random arrangement of these digits is divisible by 15 is 0.
-/
theorem zero_probability_divisible_by_15 :
  let digits := [1, 2, 3, 0, 5, 8] in
  let sum_digits := digits.sum in
  let divisible_by_3 := sum_digits % 3 = 0 in
  let arrangement_divisible_by_15 := ∃ (arrangement : List ℕ),
    arrangement.perm digits ∧ sum_digits % 15 = 0 in
  ¬arrangement_divisible_by_15 :=
begin
  sorry
end

end zero_probability_divisible_by_15_l331_331694


namespace option_D_is_correct_option_A_is_incorrect_option_B_is_incorrect_option_C_is_incorrect_l331_331871

variable (a b x : ℝ)

theorem option_D_is_correct :
  (2 * x + 1) * (x - 2) = 2 * x^2 - 3 * x - 2 :=
by sorry

theorem option_A_is_incorrect :
  2 * a^2 * b * 3 * a^2 * b^2 ≠ 6 * a^6 * b^3 :=
by sorry

theorem option_B_is_incorrect :
  0.00076 ≠ 7.6 * 10^4 :=
by sorry

theorem option_C_is_incorrect :
  -2 * a * (a + b) ≠ -2 * a^2 + 2 * a * b :=
by sorry

end option_D_is_correct_option_A_is_incorrect_option_B_is_incorrect_option_C_is_incorrect_l331_331871


namespace average_age_and_variance_l331_331852

noncomputable def average_age_two_years_ago := 13
noncomputable def variance_two_years_ago := 3
noncomputable def years_passed := 2

theorem average_age_and_variance:
  let new_average_age := average_age_two_years_ago + years_passed in
  new_average_age = 15 ∧ variance_two_years_ago = 3 :=
by
  sorry

end average_age_and_variance_l331_331852


namespace Josh_lost_marbles_l331_331099

theorem Josh_lost_marbles :
  let original_marbles := 9.5
  let current_marbles := 4.25
  original_marbles - current_marbles = 5.25 :=
by
  sorry

end Josh_lost_marbles_l331_331099


namespace greatest_number_of_bouquets_l331_331936

def cherry_lollipops := 4
def orange_lollipops := 6
def raspberry_lollipops := 8
def lemon_lollipops := 10
def candy_canes := 12
def chocolate_coins := 14

theorem greatest_number_of_bouquets : 
  Nat.gcd cherry_lollipops (Nat.gcd orange_lollipops (Nat.gcd raspberry_lollipops (Nat.gcd lemon_lollipops (Nat.gcd candy_canes chocolate_coins)))) = 2 := 
by 
  sorry

end greatest_number_of_bouquets_l331_331936


namespace female_employees_count_l331_331179

theorem female_employees_count (E Male_E Female_E M : ℕ)
  (h1: M = (2 / 5) * E)
  (h2: 200 = (E - Male_E) * (2 / 5))
  (h3: M = (2 / 5) * Male_E + 200) :
  Female_E = 500 := by
{
  sorry
}

end female_employees_count_l331_331179


namespace correct_remove_parentheses_l331_331922

theorem correct_remove_parentheses (a b c d : ℝ) :
  (a - (5 * b - (2 * c - 1)) = a - 5 * b + 2 * c - 1) :=
by sorry

end correct_remove_parentheses_l331_331922


namespace smallest_product_of_set_l331_331271

def smallest_product (s : list ℤ) : ℤ :=
  s.product.map (λ p, p.fst * p.snd).minimum

theorem smallest_product_of_set :
  smallest_product [ -9, -7, -4, 2, 5, 7 ] = -63 :=
by sorry

end smallest_product_of_set_l331_331271


namespace sum_of_fractions_l331_331277

theorem sum_of_fractions : 
  (∑ n in Finset.range 14, (1 : ℚ) / ((n + 1) * (n + 2))) = 14 / 15 := 
by
  sorry

end sum_of_fractions_l331_331277


namespace line_through_center_and_perpendicular_l331_331820

theorem line_through_center_and_perpendicular 
(C : ℝ × ℝ) 
(HC : ∀ (x y : ℝ), x ^ 2 + (y - 1) ^ 2 = 4 → C = (0, 1))
(l : ℝ → ℝ)
(Hl : ∀ x y : ℝ, 3 * x + 2 * y + 1 = 0 → y = l x)
: ∃ k b : ℝ, (∀ x : ℝ, y = k * x + b ↔ 2 * x - 3 * y + 3 = 0) :=
by 
  sorry

end line_through_center_and_perpendicular_l331_331820


namespace height_of_other_end_of_rod_l331_331896

noncomputable def cylinder_height (radius length_to_line rod_length : ℝ) : ℝ :=
  let α := real.arctan (radius / length_to_line) in
  rod_length * real.sin (2 * α)

theorem height_of_other_end_of_rod :
  ∀ (radius length_to_line rod_length : ℝ),
  radius = 20 →
  length_to_line = 40 →
  rod_length = 50 →
  cylinder_height radius length_to_line rod_length = 40 :=
by
  intros radius length_to_line rod_length hradius hlength hrod
  simp [hradius, hlength, hrod, cylinder_height]
  sorry

end height_of_other_end_of_rod_l331_331896


namespace melissa_total_repair_time_l331_331122

def time_flat_shoes := 3 + 8 + 9
def time_sandals :=  4 + 5
def time_high_heels := 6 + 12 + 10

def first_session_flat_shoes := 6 * time_flat_shoes
def first_session_sandals := 4 * time_sandals
def first_session_high_heels := 3 * time_high_heels

def second_session_flat_shoes := 4 * time_flat_shoes
def second_session_sandals := 7 * time_sandals
def second_session_high_heels := 5 * time_high_heels

def total_first_session := first_session_flat_shoes + first_session_sandals + first_session_high_heels
def total_second_session := second_session_flat_shoes + second_session_sandals + second_session_high_heels

def break_time := 15

def total_repair_time := total_first_session + total_second_session
def total_time_including_break := total_repair_time + break_time

theorem melissa_total_repair_time : total_time_including_break = 538 := by
  sorry

end melissa_total_repair_time_l331_331122


namespace xy_square_diff_l331_331051

theorem xy_square_diff (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end xy_square_diff_l331_331051


namespace election_invalid_votes_percentage_l331_331078

theorem election_invalid_votes_percentage (x : ℝ) :
  (∀ (total_votes valid_votes_in_favor_of_A : ℝ),
    total_votes = 560000 →
    valid_votes_in_favor_of_A = 357000 →
    0.75 * ((1 - x / 100) * total_votes) = valid_votes_in_favor_of_A) →
  x = 15 :=
by
  intro h
  specialize h 560000 357000 (rfl : 560000 = 560000) (rfl : 357000 = 357000)
  sorry

end election_invalid_votes_percentage_l331_331078


namespace pages_read_first_day_l331_331571

-- Alexa is reading a Nancy Drew mystery with 95 pages.
def total_pages : ℕ := 95

-- She read 58 pages the next day.
def pages_read_second_day : ℕ := 58

-- She has 19 pages left to read.
def pages_left_to_read : ℕ := 19

-- How many pages did she read on the first day?
theorem pages_read_first_day : total_pages - pages_read_second_day - pages_left_to_read = 18 := by
  -- Proof is omitted as instructed
  sorry

end pages_read_first_day_l331_331571


namespace final_answer_l331_331011

def f (x : ℝ) : ℝ := sorry

axiom functional_eqn : ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem final_answer : f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : ℝ, f (-x) = f x) ∧ ¬ (∃ ε > 0, ∀ h : ℝ, abs h < ε → f h ≥ f 0) := 
by
  -- omit the proof steps that were provided in the solution
  sorry

end final_answer_l331_331011


namespace final_position_fuel_calculation_l331_331505

-- Definitions based on the conditions in the problem.
def journey : List ℤ := [15, -4, 13, -10, -12, 3, -13, -17]
def fuel_consumption_rate : ℚ := 0.4

-- Total displacement calculation
def total_displacement (moves : List ℤ) : ℤ := moves.sum

-- Total distance calculation (sum of absolute values of segments)
def total_distance (moves : List ℤ) : ℚ := moves.foldl (λ acc x, acc + |x|) 0

-- Fuel consumed calculation
def fuel_consumed (distance : ℚ) (rate : ℚ) : ℚ := distance * rate

-- Proof problem statements
theorem final_position (j : List ℤ) : 
  total_displacement j = -38 :=
by sorry

theorem fuel_calculation (j : List ℤ) (rate : ℚ) : 
  fuel_consumed (total_distance j) rate = 34.8 :=
by sorry

-- Apply the conditions from the problem
example : final_position journey :=
by sorry

example : fuel_calculation journey fuel_consumption_rate :=
by sorry

end final_position_fuel_calculation_l331_331505


namespace test_tube_full_with_two_amoebas_l331_331931

-- Definition: Each amoeba doubles in number every minute.
def amoeba_doubling (initial : Nat) (minutes : Nat) : Nat :=
  initial * 2 ^ minutes

-- Condition: Starting with one amoeba, the test tube is filled in 60 minutes.
def time_to_fill_one_amoeba := 60

-- Theorem: If two amoebas are placed in the test tube, it takes 59 minutes to fill.
theorem test_tube_full_with_two_amoebas : amoeba_doubling 2 59 = amoeba_doubling 1 time_to_fill_one_amoeba :=
by sorry

end test_tube_full_with_two_amoebas_l331_331931


namespace problem_statement_l331_331386

noncomputable def measure_of_angle_B (a b : ℝ) (sin_A cos_B : ℝ) (h : b * sin_A = sqrt 3 * a * cos_B)
    : ℝ :=
    if sin_A ≠ 0 then
      let tan_B := sqrt 3 * (cos_B / sin_A)
      if tan_B = sqrt 3 then
        pi / 3
      else
        0 -- This placeholder should cover other cases sensibly.
    else
      0 -- To handle the case sin_A = 0.

-- Check the correct conditions to determine the area of the triangle
noncomputable def area_of_triangle_condition_2 (a c b B : ℝ) (h1 : c - a = 1) (h2 : b = sqrt 7) (hB : B = pi / 3)
    : ℝ :=
    if a = 2 ∧ c = 3 then
        (1/2) * a * c * sin B
    else
        0 -- Placeholder for other cases.

theorem problem_statement (a b c : ℝ)
    (sin_A cos_B : ℝ)
    (h_general : b * sin_A = sqrt 3 * a * cos_B)
    (h2_c_minus_a : c - a = 1)
    (h2_b : b = sqrt 7) :
    measure_of_angle_B a b sin_A cos_B h_general = pi / 3 ∧
    area_of_triangle_condition_2 a c b (pi / 3) h2_c_minus_a h2_b (pi / 3) = (3 * sqrt 3) / 2 :=
sorry

end problem_statement_l331_331386


namespace triangle_area_l331_331093

noncomputable def area_of_triangle (A B C : ℝ) (BC : ℝ) : ℝ :=
  if A + B + C = 180 ∧ BC = 24 then 32 * Real.sqrt 3 else 0

theorem triangle_area (A B C : ℝ) (BC : ℝ)
  (h1 : BC = 24)
  (h2 : ∃ D E,
    D is the midpoint of BC ∧
    E is the foot of the perpendicular from A to BC ∧
    (∠(BAD) = ∠(DAE) = ∠(EAC) = ∠(CAD))) :
  area_of_triangle A B C BC = 32 * Real.sqrt 3 :=
by
  sorry

end triangle_area_l331_331093


namespace tan_theta_minus_pi_over_4_l331_331319

theorem tan_theta_minus_pi_over_4 (θ : Real) (h1 : θ ∈ Set.Ioc (-(π / 2)) 0)
  (h2 : Real.sin (θ + π / 4) = 3 / 5) : Real.tan (θ - π / 4) = - (4 / 3) :=
by
  /- Proof goes here -/
  sorry

end tan_theta_minus_pi_over_4_l331_331319


namespace max_factorable_m_l331_331609

theorem max_factorable_m :
  ∃ m : ℤ, (∀ C D : ℤ, 5 * C * D = 120 → m = 5 * D + C) ∧ m = 601 :=
begin
  sorry
end

end max_factorable_m_l331_331609


namespace arithmetic_sequence_length_of_50_l331_331682

theorem arithmetic_sequence_length_of_50:
  let a_1 := -3 in
  let d := 4 in
  let a_n := 50 in
  ∃ n, a_1 + (n - 1) * d ≤ a_n ∧ a_1 + n * d > a_n :=
sorry

end arithmetic_sequence_length_of_50_l331_331682


namespace cos_square_sub_exp_zero_l331_331881

theorem cos_square_sub_exp_zero : 
  (cos (30 * Real.pi / 180))^2 - (2 - Real.pi) ^ 0 = -1 / 4 := by
  sorry

end cos_square_sub_exp_zero_l331_331881


namespace sum_of_integer_coeffs_of_factorized_expression_l331_331958

theorem sum_of_integer_coeffs_of_factorized_expression :
  let expr := (125 : ℤ) * (x ^ 6) - (216 : ℤ) * (z ^ 6)
  let factored_expr := (5 * (x ^ 2) - 6 * (z ^ 2)) * (25 * (x ^ 4) + 30 * (x ^ 2) * (z ^ 2) + 36 * (z ^ 4))
  ∑ c in (factored_expr.coeffs), c = 90 :=
by
  let expr := (125 : ℤ) * (x ^ 6) - (216 : ℤ) * (z ^ 6)
  let factored_expr := (5 * (x ^ 2) - 6 * (z ^ 2)) * (25 * (x ^ 4) + 30 * (x ^ 2) * (z ^ 2) + 36 * (z ^ 4))
  have sum := 5 + (-6) + 25 + 30 + 36 -- sum of integer coefficients
  exact eq.refl 90 -- sum is 90
  sorry

end sum_of_integer_coeffs_of_factorized_expression_l331_331958


namespace expression_compute_eq_pi_expression_simplify_eq_5_l331_331260

-- Proof Problem 1
theorem expression_compute_eq_pi (pi : ℝ) :
  (-7 / 8 : ℝ)^0 + (8 : ℝ)^(1 / 3) + ((3 : ℝ) - pi)^4^(1 / 4) = pi :=
sorry

-- Proof Problem 2
theorem expression_simplify_eq_5 :
  log 3 (sqrt 27) - log 3 (sqrt 3) + log 10 25 + log 10 4 + Real.log (Real.exp 2) = 5 :=
sorry

end expression_compute_eq_pi_expression_simplify_eq_5_l331_331260


namespace find_a10_l331_331634

variable {n : ℕ}
variable (a : ℕ → ℝ)
variable (h_pos : ∀ (n : ℕ), 0 < a n)
variable (h_mul : ∀ (p q : ℕ), a (p + q) = a p * a q)
variable (h_a8 : a 8 = 16)

theorem find_a10 : a 10 = 32 :=
by
  sorry

end find_a10_l331_331634


namespace trapezoid_area_is_48_l331_331577

noncomputable def trapezoid_area (a b : ℝ) (h : ℝ) := (1 / 2) * (a + b) * h

theorem trapezoid_area_is_48 :
  ∀ (a b h : ℝ), 
    a = 20 → 
    b = (3/5) * 20 → 
    h = 3 →
    trapezoid_area a b h = 48 :=
by
  intros a b h a_eq b_eq h_eq
  unfold trapezoid_area
  rw [a_eq, b_eq, h_eq]
  norm_num
  sorry

end trapezoid_area_is_48_l331_331577


namespace smallest_4_digit_multiple_l331_331862

-- Define the numbers
def num1 := 25
def num2 := 40
def num3 := 75

-- Define the maximum 4-digit number
def max_4_digit_num := 9600

-- Define the smallest 4-digit number
theorem smallest_4_digit_multiple :
  ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % Nat.lcm (Nat.lcm num1 num2) num3 = 0 ∧
  (∀ m : ℕ, m ≥ 1000 ∧ m < n → m % Nat.lcm (Nat.lcm num1 num2) num3 ≠ 0) ∧
  max_4_digit_num % Nat.lcm (Nat.lcm num1 num2) num3 = 0 :=
begin
  sorry
end

end smallest_4_digit_multiple_l331_331862


namespace problem_l331_331041

theorem problem (m n : ℤ) (h : 496 = 2^m - 2^n) : m + n = 13 :=
sorry

end problem_l331_331041


namespace min_value_of_m_n_squared_l331_331302

theorem min_value_of_m_n_squared 
  (a b c : ℝ)
  (triangle_cond : a^2 + b^2 = c^2)
  (m n : ℝ)
  (line_cond : a * m + b * n + 3 * c = 0) 
  : m^2 + n^2 = 9 := 
by
  sorry

end min_value_of_m_n_squared_l331_331302


namespace calculate_perimeter_l331_331848

-- Definitions based on conditions
def num_posts : ℕ := 36
def post_width : ℕ := 2
def gap_width : ℕ := 4
def sides : ℕ := 4

-- Computations inferred from the conditions (not using solution steps directly)
def posts_per_side : ℕ := num_posts / sides
def gaps_per_side : ℕ := posts_per_side - 1
def side_length : ℕ := posts_per_side * post_width + gaps_per_side * gap_width

-- Theorem statement, proving the perimeter is 200 feet
theorem calculate_perimeter : 4 * side_length = 200 := by
  sorry

end calculate_perimeter_l331_331848


namespace symmetric_lines_intersect_on_circumcircle_l331_331904

open Set

noncomputable def orthocenter (A B C : Point) : Point := sorry

noncomputable def symmetric_line (l : Line) (s : Line) : Line := sorry

noncomputable def circumcircle (A B C : Point) : Circle := sorry

noncomputable def intersection_point (l1 l2 : Line) : Point := sorry

theorem symmetric_lines_intersect_on_circumcircle
  (A B C : Point)
  (h_acute : acute_triangle A B C)
  (H : Point := orthocenter A B C)
  (l : Line)
  (h_l : passes_through l H)
  (l_a := symmetric_line l (line_through B C))
  (l_b := symmetric_line l (line_through C A))
  (l_c := symmetric_line l (line_through A B))
  (P := intersection_point l_a l_b) :
  P ∈ circumcircle A B C ∧ 
  P = intersection_point l_b l_c ∧ 
  P = intersection_point l_c l_a := 
sorry

end symmetric_lines_intersect_on_circumcircle_l331_331904


namespace magnitude_of_w_l331_331396

noncomputable def z : ℂ := ( (-13 + 15 * Complex.i)^2 * (26 - 9 * Complex.i)^3 ) / (5 + 2 * Complex.i)
noncomputable def w : ℂ := Complex.conj z / z + 1

theorem magnitude_of_w : Complex.abs w = 2 := by
  sorry

end magnitude_of_w_l331_331396


namespace pizza_area_increase_l331_331762
open Real

noncomputable def area_of_circle (d : ℝ) : ℝ := π * (d / 2) ^ 2

theorem pizza_area_increase (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 20) :
  (area_of_circle d2 - area_of_circle d1) / area_of_circle d1 * 100 = 56.25 :=
by
  rw [h1, h2]
  simp[area_of_circle]
  -- rest of the proof steps here
  -- sorry

end pizza_area_increase_l331_331762


namespace star_eight_four_l331_331491

def star (a b : ℝ) : ℝ := a + (a / b) + 1

theorem star_eight_four : star 8 4 = 11 :=
by
  sorry

end star_eight_four_l331_331491


namespace bijective_continuous_fun_integral_property_l331_331283

theorem bijective_continuous_fun_integral_property :
  ∀ (f : ℝ → ℝ), (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x ∧ f x ≤ 1) →
  (Bijective f ∧ Continuous f) →
  (∀ g : ℝ → ℝ, ContinuousOn g (Set.Icc (0 : ℝ) 1) →
  (∫ x in 0..1, g (f x)) = ∫ x in 0..1, g x) →
  (∀ x, 0 ≤ x ∧ x ≤ 1 → (f x = x ∨ f x = 1 - x)) :=
by
  intros f hf_range hf_props hf_integral x hx
  sorry

end bijective_continuous_fun_integral_property_l331_331283


namespace general_term_formulas_l331_331324

-- Definitions of the sequences and sum formulas
def arithmetic_sequence (a_1 d : ℤ) : ℕ → ℤ
| 0     := a_1
| (n+1) := a_1 + (n+1) * d

def geometric_sequence (b_1 q : ℤ) : ℕ → ℤ
| 0     := b_1
| (n+1) := b_1 * q ^ (n+1)

def sum_arithmetic_sequence (a_1 d n : ℤ) : ℤ :=
(n * (2 * a_1 + (n - 1) * d)) / 2

def sum_geometric_sequence (b_1 q n : ℤ) : ℤ :=
if q = 1 then n * b_1 else b_1 * ((q ^ n - 1) / (q - 1))

-- Conditions
constants (a_1 b_1 : ℤ) (d q : ℤ) (n : ℕ)
axiom a1 : a_1 = 1
axiom b1 : b_1 = 3
axiom a3_b3_sum : arithmetic_sequence a_1 d 2 + geometric_sequence b_1 q 2 = 17
axiom T3_S3_diff : sum_geometric_sequence b_1 q 3 - sum_arithmetic_sequence a_1 d 3 = 12
axiom q_pos : q > 0

-- Proof of the general term formulas
theorem general_term_formulas :
  (∀ n, arithmetic_sequence a_1 d n = 2 * n - 1) ∧
  (∀ n, geometric_sequence b_1 q n = 3) :=
by
  sorry

end general_term_formulas_l331_331324


namespace geometric_sequence_common_ratio_l331_331654

theorem geometric_sequence_common_ratio {a : ℕ+ → ℝ} (q : ℝ) (h_geom : ∀ n, a (n + 1) = q * a n) 
  (h_a3 : a 3 = 1) (h_a5 : a 5 = 4) : q = 2 ∨ q = -2 :=
by
  sorry

end geometric_sequence_common_ratio_l331_331654


namespace problem1_solution_set_problem2_range_of_m_l331_331334

open Real

noncomputable def f (x : ℝ) := abs (x + 1) - abs (x - 2)

theorem problem1_solution_set :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} :=
sorry

theorem problem2_range_of_m (m : ℝ) :
  (∃ x : ℝ, f x ≥ x^2 - x + m) → m ≤ 5 / 4 :=
sorry

end problem1_solution_set_problem2_range_of_m_l331_331334


namespace red_balls_as_random_variable_l331_331070

noncomputable def number_of_red_balls_drawn_is_random_variable (black_balls red_balls balls_drawn : ℕ) (h_black : black_balls = 2) (h_red : red_balls = 6) (h_drawn : balls_drawn = 2) : Prop :=
  ∃ (X : Type*) [random_variable X] (possible_values : set ℕ), X = λ (d : balls_drawn), possible_values = {0, 1, 2} 

theorem red_balls_as_random_variable :
  number_of_red_balls_drawn_is_random_variable 2 6 2 := 
by
  sorry

end red_balls_as_random_variable_l331_331070


namespace abs_eq_sqrt_five_l331_331040

theorem abs_eq_sqrt_five (x : ℝ) (h : |x| = Real.sqrt 5) : x = Real.sqrt 5 ∨ x = -Real.sqrt 5 := 
sorry

end abs_eq_sqrt_five_l331_331040


namespace tangents_form_cyclic_quadrilateral_inscribed_and_circumscribed_quad_product_eq_r_squared_l331_331876

-- Problem 2.79(a)
theorem tangents_form_cyclic_quadrilateral {A B C D O : Type*} [Circle O] (h1 : Tangent A O) (h2 : Tangent B O) (h3 : Tangent C O) (h4 : Tangent D O) :
  CyclicQuadrilateral (Quadrilateral.mk A B C D) :=
sorry

-- Problem 2.79(b)
theorem inscribed_and_circumscribed_quad_product_eq_r_squared {K L M N O : Type*} [InscribedCircledQuadrilateral K L M N O]
  (A B : Type*) (r : ℝ) (h1 : PointOnCircle A K L) (h2 : PointOnCircle B L M) :
  AK * BM = r^2 :=
sorry

end tangents_form_cyclic_quadrilateral_inscribed_and_circumscribed_quad_product_eq_r_squared_l331_331876


namespace length_of_first_platform_l331_331240

theorem length_of_first_platform 
  (t1 t2 : ℝ) 
  (length_train : ℝ) 
  (length_second_platform : ℝ) 
  (time1 : ℝ) 
  (time2 : ℝ) 
  (speed_eq : (t1 + length_train) / time1 = (length_second_platform + length_train) / time2) 
  (time1_eq : time1 = 15) 
  (time2_eq : time2 = 20) 
  (length_train_eq : length_train = 100) 
  (length_second_platform_eq: length_second_platform = 500) :
  t1 = 350 := 
  by 
  sorry

end length_of_first_platform_l331_331240


namespace total_distance_is_correct_l331_331231

-- Define the given conditions
def speed_still_water := 15 -- Speed of the man in still water (km/hr)
def speed_current_same := 2.5 -- Speed of the current in the same direction (km/hr)
def time_same_direction := 2 -- Time in the same direction (hours)

def speed_current_opposite := 3 -- Speed of the current in the opposite direction (km/hr)
def time_opposite_direction := 1.5 -- Time in opposite direction (hours)

-- Define the effective speeds in each part
def speed_with_current := speed_still_water + speed_current_same
def speed_against_current := speed_still_water - speed_current_opposite

-- Define the distances covered in each part
def distance_with_current := speed_with_current * time_same_direction
def distance_against_current := speed_against_current * time_opposite_direction

-- Define the total distance covered
def total_distance := distance_with_current + distance_against_current

-- The main theorem to prove
theorem total_distance_is_correct : total_distance = 53 := by
  -- Definitions (from the conditions)
  let speed_still_water := 15
  let speed_current_same := 2.5
  let time_same_direction := 2
  let speed_current_opposite := 3
  let time_opposite_direction := 1.5

  -- Effective speeds
  let speed_with_current := speed_still_water + speed_current_same
  let speed_against_current := speed_still_water - speed_current_opposite

  -- Distances
  let distance_with_current := speed_with_current * time_same_direction
  let distance_against_current := speed_against_current * time_opposite_direction

  -- Total distance
  let total_distance := distance_with_current + distance_against_current

  -- Show that total distance is 53
  have dist_with_current : distance_with_current = 35 := by sorry
  have dist_against_current : distance_against_current = 18 := by sorry
  have total : total_distance = 53 := by sorry
  show total_distance = 53 from total

end total_distance_is_correct_l331_331231


namespace Stewarts_Theorem_l331_331134

theorem Stewarts_Theorem 
  {A B C P : Type*} [metric_space P] [linear_ordered_field P] 
  (AB AC AP BP PC BC : P) : Prop :=
  AB^2 * PC + AC^2 * BP = AP^2 * BC + BP * PC * BC := 
sorry

end Stewarts_Theorem_l331_331134


namespace percentage_invalid_votes_l331_331083

theorem percentage_invalid_votes 
    (total_votes : ℕ)
    (candidate_A_votes : ℕ)
    (candidate_A_percentage : ℝ)
    (total_valid_percentage : ℝ) :
    total_votes = 560000 ∧
    candidate_A_votes = 357000 ∧
    candidate_A_percentage = 0.75 ∧
    total_valid_percentage = 100 - x ∧
    (0.75 * (total_valid_percentage / 100) * 560000 = 357000) →
    x = 15 :=
by
  sorry

end percentage_invalid_votes_l331_331083


namespace not_true_statement_l331_331417

theorem not_true_statement
  (n : ℕ)
  (h₁ : n > 0)
  (h₂ : (1 / 2 + 1 / 4 + 1 / 5 + 1 / (n : ℝ)) ∈ ℤ) :
  ¬ n > 40 :=
by
  have h₃ : n = 20 := sorry
  exact sorry

end not_true_statement_l331_331417


namespace tim_biking_time_l331_331184

theorem tim_biking_time
  (work_days : ℕ := 5) 
  (distance_to_work : ℕ := 20) 
  (weekend_ride : ℕ := 200) 
  (speed : ℕ := 25) 
  (weekly_work_distance := 2 * distance_to_work * work_days)
  (total_distance := weekly_work_distance + weekend_ride) : 
  (total_distance / speed = 16) := 
by
  sorry

end tim_biking_time_l331_331184


namespace inequality_holds_for_all_m_l331_331971

theorem inequality_holds_for_all_m (m : ℝ) (h1 : ∀ (x : ℝ), x^2 - 8 * x + 20 > 0)
  (h2 : m < -1/2) : ∀ (x : ℝ), (x ^ 2 - 8 * x + 20) / (m * x ^ 2 + 2 * (m + 1) * x + 9 * m + 4) < 0 :=
by
  sorry

end inequality_holds_for_all_m_l331_331971


namespace probability_a_squared_plus_b_divisible_by_3_l331_331620

def is_divisible_by_3 (x : Int) : Prop :=
  x % 3 = 0

noncomputable def probability_divisible_by_3 : Real :=
  let count_all_pairs : Nat := 100
  let count_favorable_pairs : Nat := 30
  (count_favorable_pairs.toReal / count_all_pairs.toReal)

theorem probability_a_squared_plus_b_divisible_by_3 :
  probability_divisible_by_3 = 0.3 :=
by
  sorry

end probability_a_squared_plus_b_divisible_by_3_l331_331620


namespace arrangement_count_l331_331348

theorem arrangement_count : 
  let letters := ['B', 'A₁', 'A₂', 'A₃', 'N₁', 'N₂', 'N₃'] in
  letters.length = 7 ∧ 
  (∀ (x y : Char), x ∈ letters → y ∈ letters → x ≠ y → ∃! (p : letters.Perm x y), p x = y) →
  (letters.length.factorial = 5040) :=
by
  sorry

end arrangement_count_l331_331348


namespace first_player_wins_l331_331847

def initial_piles (p1 p2 : Nat) : Prop :=
  p1 = 33 ∧ p2 = 35

def winning_strategy (p1 p2 : Nat) : Prop :=
  ∃ moves : List (Nat × Nat), 
  (initial_piles p1 p2) →
  (∀ (p1' p2' : Nat), 
    (p1', p2') ∈ moves →
    p1' = 1 ∧ p2' = 1 ∨ p1' = 2 ∧ p2' = 1)

theorem first_player_wins : winning_strategy 33 35 :=
sorry

end first_player_wins_l331_331847


namespace tomatoes_price_per_pound_l331_331830

noncomputable def price_per_pound (cost_per_pound : ℝ) (loss_percent : ℝ) (profit_percent : ℝ) : ℝ :=
  let remaining_percent := 1 - loss_percent / 100
  let desired_total := (1 + profit_percent / 100) * cost_per_pound
  desired_total / remaining_percent

theorem tomatoes_price_per_pound :
  price_per_pound 0.80 15 8 = 1.02 :=
by
  sorry

end tomatoes_price_per_pound_l331_331830


namespace relationship_between_y1_y2_l331_331321

theorem relationship_between_y1_y2 (b y1 y2 : ℝ) 
  (h1 : y1 = -(-2) + b) 
  (h2 : y2 = -(3) + b) : 
  y1 > y2 := 
by {
  sorry
}

end relationship_between_y1_y2_l331_331321


namespace maxwell_distance_when_meeting_l331_331209

theorem maxwell_distance_when_meeting 
  (distance_between_homes : ℕ)
  (maxwell_speed : ℕ) 
  (brad_speed : ℕ) 
  (total_distance : ℕ) 
  (h : distance_between_homes = 36) 
  (h1 : maxwell_speed = 2)
  (h2 : brad_speed = 4) 
  (h3 : 6 * (total_distance / 6) = distance_between_homes) :
  total_distance = 12 :=
sorry

end maxwell_distance_when_meeting_l331_331209


namespace ratio_green_to_red_is_16_l331_331849

def diameter_red := 2
def diameter_middle := 6
def diameter_large := 10

def radius (d : ℝ) : ℝ := d / 2
def area (r : ℝ) : ℝ := Real.pi * r^2

def radius_red := radius diameter_red
def radius_middle := radius diameter_middle
def radius_large := radius diameter_large

def area_red := area radius_red
def area_middle := area radius_middle
def area_large := area radius_large

def area_green := area_large - area_middle
def ratio_green_to_red := area_green / area_red

theorem ratio_green_to_red_is_16 : ratio_green_to_red = 16 :=
by
  sorry

end ratio_green_to_red_is_16_l331_331849


namespace mark_jump_rope_hours_l331_331118

theorem mark_jump_rope_hours 
    (record : ℕ := 54000)
    (jump_per_second : ℕ := 3)
    (seconds_per_hour : ℕ := 3600)
    (total_jumps_to_break_record : ℕ := 54001)
    (jumps_per_hour : ℕ := jump_per_second * seconds_per_hour) 
    (hours_needed : ℕ := total_jumps_to_break_record / jumps_per_hour) 
    (round_up : ℕ := if total_jumps_to_break_record % jumps_per_hour = 0 then hours_needed else hours_needed + 1) :
    round_up = 5 :=
sorry

end mark_jump_rope_hours_l331_331118


namespace large_square_area_l331_331579

theorem large_square_area (l w : ℕ) (h1 : 2 * (l + w) = 28) : (l + w) * (l + w) = 196 :=
by {
  sorry
}

end large_square_area_l331_331579


namespace scientific_notation_2700000_l331_331470

theorem scientific_notation_2700000 :
  2700000 = 2.7 * 10^6 := 
begin
  sorry
end

end scientific_notation_2700000_l331_331470


namespace vector_dot_product_value_l331_331409

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 4)
def c : ℝ × ℝ := (3, 2)

def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem vector_dot_product_value : dot_product (add (scalar_mul 2 a) b) c = -3 := by
  sorry

end vector_dot_product_value_l331_331409


namespace sin_double_angle_l331_331622

theorem sin_double_angle (α : ℝ) (h : sin α - cos α = 4/3) : sin (2 * α) = -7/9 :=
by
  sorry

end sin_double_angle_l331_331622


namespace quadratic_m_value_l331_331990

theorem quadratic_m_value (m : ℤ) (hm1 : |m| = 2) (hm2 : m ≠ 2) : m = -2 :=
sorry

end quadratic_m_value_l331_331990


namespace greatest_prime_factor_377_l331_331199

theorem greatest_prime_factor_377 : ∃ p : ℕ, p ∈ {f | f.factor 377} ∧ ∀ q ∈ {f | f.factor 377}, q ≤ p := 
by
  sorry

end greatest_prime_factor_377_l331_331199


namespace polynomial_degree_rational_coefficients_l331_331802

theorem polynomial_degree_rational_coefficients :
  ∃ p : Polynomial ℚ,
    (Polynomial.aeval (2 - 3 * Real.sqrt 3) p = 0) ∧
    (Polynomial.aeval (-2 - 3 * Real.sqrt 3) p = 0) ∧
    (Polynomial.aeval (3 + Real.sqrt 11) p = 0) ∧
    (Polynomial.aeval (3 - Real.sqrt 11) p = 0) ∧
    p.degree = 6 :=
sorry

end polynomial_degree_rational_coefficients_l331_331802


namespace proof_b_minus_c_l331_331978

def a (n : ℕ) : ℝ := if n > 1 then real.log n / real.log 3003 else 0

noncomputable def b : ℝ := a 6 + a 7 + a 8 + a 9
noncomputable def c : ℝ := a 15 + a 16 + a 17 + a 18 + a 19

theorem proof_b_minus_c : b - c = -real.log 4610 / real.log 3003 :=
by sorry

end proof_b_minus_c_l331_331978


namespace sum_of_triangle_areas_in_prism_faces_l331_331934

def volume_of_rectangular_prism (length width height : ℕ) : ℕ := length * width * height

def area_of_right_triangle (base height : ℕ) : ℕ := (base * height) / 2

theorem sum_of_triangle_areas_in_prism_faces : 
  let len := 2
  let wid := 3
  let hei := 4
  let area_face1 := 2 * 3 / 2
  let area_face2 := 3 * 4 / 2
  let area_face3 := 2 * 4 / 2
  let total_area := 2 * (2 * area_face1 + 2 * area_face2 + 2 * area_face3)
  total_area = 52 := 
by 
  have len := 2
  have wid := 3
  have hei := 4
  let area_face1 := 2 * 3 / 2
  let area_face2 := 3 * 4 / 2
  let area_face3 := 2 * 4 / 2
  let total_area := 2 * (2 * area_face1 + 2 * area_face2 + 2 * area_face3)
  show total_area = 52 from sorry

end sum_of_triangle_areas_in_prism_faces_l331_331934


namespace quadratic_roots_difference_l331_331270

theorem quadratic_roots_difference 
  (h1 : ∃ k, ∃ r1 r2, 49 * r1^2 - 84 * r1 + 36 = 0 ∧ r2 = k * r1)
  : (let k := (3 + Real.sqrt 5) / 2 in
     let r1 := 24 / (17 + 7 * Real.sqrt 5) in
     let r2 := (12 / 7) - r1 in
     r2 - r1 = (17 * Real.sqrt 5 - 31) / (17 + 7 * Real.sqrt 5)) :=
sorry

end quadratic_roots_difference_l331_331270


namespace inequalities_correct_l331_331999

-- Define the basic conditions
variables {a b c d : ℝ}

-- Conditions given in the problem
axiom h1 : a > b
axiom h2 : b > 0
axiom h3 : 0 > c
axiom h4 : c > d

-- Correct answers to be proven
theorem inequalities_correct (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
  (a + c > b + d) ∧ (a * d^2 > b * c^2) ∧ (1 / (b * c) < 1 / (a * d)) :=
begin
  -- Proof part
  sorry
end

end inequalities_correct_l331_331999


namespace abs_diff_squares_l331_331859

theorem abs_diff_squares (a b : ℤ) (ha : a = 103) (hb : b = 97) : |a^2 - b^2| = 1200 :=
by
  sorry

end abs_diff_squares_l331_331859


namespace max_points_in_circle_with_minimum_distance_l331_331632

theorem max_points_in_circle_with_minimum_distance {O : Type} (r : ℝ) (N : set O) (n : ℕ) 
  (h_radius : ∀ p ∈ N, dist O p = r)
  (h_min_dist : ∀ p1 p2 ∈ N, p1 ≠ p2 → dist p1 p2 ≥ (real.sqrt 3) * r) :
  n ≤ 3 :=
sorry

end max_points_in_circle_with_minimum_distance_l331_331632


namespace min_value_of_f_l331_331344

-- Define the function f
def f (a b : ℝ) : ℝ :=
  max (-1:1) (λ x, abs (x^2 - a * x - b))

-- State the theorem to prove that the minimum value of f(a, b) is 1/2
theorem min_value_of_f :
  ∃ a b : ℝ, f a b = 1/2 ∧ (∀ a' b' : ℝ, f a' b' ≥ 1/2) :=
sorry

end min_value_of_f_l331_331344


namespace three_sum_zero_l331_331444

theorem three_sum_zero (n : ℕ) (s : Finset ℤ) (h_card : s.card = 2 * n + 1)
  (h_abs : ∀ x ∈ s, |x| ≤ 2 * n - 1) :
  ∃ a b c ∈ s, a + b + c = 0 := by
  sorry

end three_sum_zero_l331_331444


namespace single_interval_condition_l331_331516

-- Definitions: k and l are integers
variables (k l : ℤ)

-- Condition: The given condition for l
theorem single_interval_condition : l = Int.floor (k ^ 2 / 4) :=
sorry

end single_interval_condition_l331_331516


namespace trig_identity_l331_331258

theorem trig_identity : 2 * tan (10 * Real.pi / 180) + 3 * sin (10 * Real.pi / 180) = 5 * sin (10 * Real.pi / 180) :=
by
  sorry

end trig_identity_l331_331258


namespace jerry_feathers_ratio_l331_331094

theorem jerry_feathers_ratio:
  ∀ (hawk_feathers eagle_feathers total_feathers feathers_given feathers_left feathers_sold: ℕ),
  hawk_feathers = 6 -> 
  eagle_feathers = 17 * hawk_feathers ->
  total_feathers = hawk_feathers + eagle_feathers ->
  feathers_given = 10 ->
  feathers_left = total_feathers - feathers_given - feathers_sold ->
  feathers_left = 49 ->
  (feathers_sold : total_feathers - feathers_given) = 1 : 2 :=
by
   sorry

end jerry_feathers_ratio_l331_331094


namespace radius_of_circle_l331_331146

theorem radius_of_circle:
  (∃ (r: ℝ), 
    (∀ (x: ℝ), (x^2 + r - x) = 0 → 1 - 4 * r = 0)
  ) → r = 1 / 4 := 
sorry

end radius_of_circle_l331_331146


namespace solve_for_x_l331_331787

theorem solve_for_x (x : ℝ) : (5 * x - 2) / (6 * x - 6) = 3 / 4 ↔ x = -5 := by
  sorry

end solve_for_x_l331_331787


namespace tanya_completes_work_in_4_days_l331_331139

theorem tanya_completes_work_in_4_days (sakshi_days : ℕ) (tanya_efficiency : ℝ) :
  sakshi_days = 5 → tanya_efficiency = 1.25 → (1 / (1 / sakshi_days * tanya_efficiency)) = 4 :=
by
  intro h_sakshi_days h_tanya_efficiency
  subst h_sakshi_days
  subst h_tanya_efficiency
  have h_sakshi_rate : ℝ := 1 / 5
  have h_tanya_rate : ℝ := h_sakshi_rate * 1.25
  have h_tanya_days : ℝ := 1 / h_tanya_rate
  rw [h_sakshi_rate, h_tanya_rate, h_tanya_days]
  norm_num
  sorry

end tanya_completes_work_in_4_days_l331_331139


namespace new_years_day_more_frequent_l331_331725

-- Define conditions
def common_year_days : ℕ := 365
def leap_year_days : ℕ := 366
def century_is_leap_year (year : ℕ) : Prop := (year % 400 = 0)

-- Given: 23 October 1948 was a Saturday
def october_23_1948 : ℕ := 5 -- 5 corresponds to Saturday

-- Define the question proof statement
theorem new_years_day_more_frequent :
  (frequency_Sunday : ℕ) > (frequency_Monday : ℕ) :=
sorry

end new_years_day_more_frequent_l331_331725


namespace sin_B_value_midline_BC_length_l331_331320

-- Definitions for conditions
variables (ABC : Triangle)
variables (a b c : ℝ)
variables (A B C : Angle)
variables (k : ℝ)

-- Triangular relations
axiom sin_A : sin A = sqrt 7 / 4
axiom sin_C : sin C = 3 * sqrt 7 / 8
axiom vector_mag : ∥vector.plus ABC.AC ABC.BC∥ = 2 * sqrt 23

-- Assuming triangle sides for simplicity:
axiom sides_relation : (sin A)⁻¹ * a = (sin B)⁻¹ * b ∧ (sin B)⁻¹ * b = (sin C)⁻¹ * c

-- Midline length and to prove values
theorem sin_B_value : sin B = 5 * sqrt 7 / 16 := sorry
theorem midline_BC_length : midline_length ABC.BC = sqrt 53 := sorry

end sin_B_value_midline_BC_length_l331_331320


namespace total_bowling_balls_l331_331151

theorem total_bowling_balls (r g b : ℕ) (h1 : r = 30) (h2 : g = r + 6) (h3 : b = 2 * g) :
  r + g + b = 138 :=
by
  have g := by rw [h1] at h2; exact h2
  have b := by rw [h1, g] at h3; exact h3
  have total := by rw [h1, g, b]
  exact total

end total_bowling_balls_l331_331151


namespace expected_value_min_pow4_eq_m_plus_n_eq_1002_l331_331777

noncomputable def expected_value_min_pow4 (X : Fin 10 → ℝ) : ℝ :=
  ∫ y in 0..1, 10 * y^4 * (1 - y)^9

theorem expected_value_min_pow4_eq :
  ∀ (X : Fin 10 → ℝ) (h : ∀ i, X i ∈ Set.Icc (0 : ℝ) 1),
  10 * ∫ y in 0..1, y^4 * (1 - y)^9 = (1 / 1001) :=
by
  sorry

theorem m_plus_n_eq_1002 :
  ∀ (X : Fin 10 → ℝ) (h : ∀ i, X i ∈ Set.Icc (0 : ℝ) 1),
  let E := 10 * ∫ y in 0..1, y^4 * (1 - y)^9
  (m n : ℕ) (h_rel_prime : Nat.RelativelyPrime m n) (h_rat_eq : E = (m / n))
  in m + n = 1002 :=
by
  sorry

end expected_value_min_pow4_eq_m_plus_n_eq_1002_l331_331777


namespace heartsuit_fraction_l331_331023

def heartsuit (n m : ℕ) : ℕ := n ^ 4 * m ^ 3

theorem heartsuit_fraction :
  (heartsuit 3 5) / (heartsuit 5 3) = 3 / 5 :=
by
  sorry

end heartsuit_fraction_l331_331023


namespace joan_picked_apples_l331_331096

theorem joan_picked_apples (a b c : ℕ) (h1 : b = 27) (h2 : c = 70) (h3 : c = a + b) : a = 43 :=
by
  sorry

end joan_picked_apples_l331_331096


namespace total_flowers_in_3_hours_l331_331845

-- Constants representing the number of each type of flower
def roses : ℕ := 12
def sunflowers : ℕ := 15
def tulips : ℕ := 9
def daisies : ℕ := 18
def orchids : ℕ := 6
def total_flowers : ℕ := 60

-- Number of flowers each bee can pollinate in an hour
def bee_A_rate (roses sunflowers tulips: ℕ) : ℕ := 2 + 3 + 1
def bee_B_rate (daisies orchids: ℕ) : ℕ := 4 + 1
def bee_C_rate (roses sunflowers tulips daisies orchids: ℕ) : ℕ := 1 + 2 + 2 + 3 + 1

-- Total number of flowers pollinated by all bees in an hour
def total_bees_rate (bee_A_rate bee_B_rate bee_C_rate: ℕ) : ℕ := bee_A_rate + bee_B_rate + bee_C_rate

-- Proving the total flowers pollinated in 3 hours
theorem total_flowers_in_3_hours : total_bees_rate 6 5 9 * 3 = total_flowers := 
by {
  sorry
}

end total_flowers_in_3_hours_l331_331845


namespace ratio_age_difference_to_pencils_l331_331471

-- Definitions of the given problem conditions
def AsafAge : ℕ := 50
def SumOfAges : ℕ := 140
def AlexanderAge : ℕ := SumOfAges - AsafAge

def PencilDifference : ℕ := 60
def TotalPencils : ℕ := 220
def AsafPencils : ℕ := (TotalPencils - PencilDifference) / 2
def AlexanderPencils : ℕ := AsafPencils + PencilDifference

-- Define the age difference and the ratio
def AgeDifference : ℕ := AlexanderAge - AsafAge
def Ratio : ℚ := AgeDifference / AsafPencils

theorem ratio_age_difference_to_pencils : Ratio = 1 / 2 := by
  sorry

end ratio_age_difference_to_pencils_l331_331471


namespace female_employees_count_l331_331178

-- Define constants
def E : ℕ  -- Total number of employees
def M : ℕ := (2 / 5) * E  -- Total number of managers
def Male_E : ℕ  -- Total number of male employees
def Female_E : ℕ := E - Male_E  -- Total number of female employees
def Male_M : ℕ := (2 / 5) * Male_E  -- Total number of male managers
def Female_M : ℕ := 200  -- Total number of female managers

-- Given equation relating managers and employees
theorem female_employees_count : Female_E = 500 :=
by
  -- Required proof goes here
  sorry

end female_employees_count_l331_331178


namespace arith_sqrt_of_2m_minus_n_l331_331645

theorem arith_sqrt_of_2m_minus_n (m n : ℤ) (h1 : m = 3) (h2 : n = 2) : 
  sqrt (2 * m - n) = 2 := 
by
-- The proof goes here
sorry

end arith_sqrt_of_2m_minus_n_l331_331645


namespace cosine_angle_cube_regular_tetrahedron_l331_331246

noncomputable def cube_vertices (a : ℝ) : list (ℝ × ℝ × ℝ) := 
  [(0, 0, 0), (a, 0, 0), (0, a, 0), (a, a, 0), (0, 0, a), (a, 0, a), (0, a, a), (a, a, a)]

noncomputable def distance_from_point_to_plane (P : ℝ × ℝ × ℝ) (plane_normal : ℝ × ℝ × ℝ) (plane_point : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := P in
  let (a, b, c) := plane_normal in
  let (x0, y0, z0) := plane_point in
  abs (a * (x1 - x0) + b * (y1 - y0) + c * (z1 - z0)) / sqrt (a^2 + b^2 + c^2)

def cos_theta_between_lines (P : ℝ × ℝ × ℝ) (A1 : ℝ × ℝ × ℝ) (B : ℝ × ℝ × ℝ) (C1 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := P in
  let (x2, y2, z2) := A1 in
  let (x3, y3, z3) := B in
  let (x4, y4, z4) := C1 in
  let p1 := (x1 - x2, y1 - y2, z1 - z2) in
  let p2 := (x4 - x3, y4 - y3, z4 - z3) in
  let p1_mag := sqrt ((fst3 p1)^2 + (snd3 p1)^2 + (trd3 p1)^2) in
  let p2_mag := sqrt ((fst3 p2)^2 + (snd3 p2)^2 + (trd3 p2)^2) in
  let dot_product := (fst3 p1) * (fst3 p2) + (snd3 p1) * (snd3 p2) + (trd3 p1) * (trd3 p2) in
  dot_product / (p1_mag * p2_mag)

theorem cosine_angle_cube_regular_tetrahedron (a : ℝ) :
  let cube := cube_vertices a in
  let A := cube.nth 0 in
  let B := cube.nth 1 in
  let C1 := cube.nth 6 in
  let P := (a/2, a/2, 3*a/2) in
  let plane_normal := (0, 0, 1) in
  let A1 := cube.nth 4 in
  distance_from_point_to_plane P plane_normal (0, 0, 0) = 3/2 * a →
  cos_theta_between_lines P A1 (B.get_or_else (0, 0, 0)) C1.get_or_else (0, 0, 0) = sqrt 6 / 3 :=
sorry

end cosine_angle_cube_regular_tetrahedron_l331_331246


namespace solve_for_y_l331_331790

theorem solve_for_y (y : ℝ) : (1 / 8)^(3 * y + 12) = (32)^(3 * y + 7) → y = -71 / 24 := by
  sorry

end solve_for_y_l331_331790


namespace remainder_13_pow_150_mod_11_l331_331521

theorem remainder_13_pow_150_mod_11 : (13^150) % 11 = 1 := 
by 
  sorry

end remainder_13_pow_150_mod_11_l331_331521


namespace amount_given_to_last_set_l331_331441

theorem amount_given_to_last_set 
  (total_amount : ℕ) 
  (amount_first_set : ℕ) 
  (amount_second_set : ℕ) 
  (h_total : total_amount = 900) 
  (h_first : amount_first_set = 325) 
  (h_second : amount_second_set = 260) : 
  total_amount - amount_first_set - amount_second_set = 315 := 
by 
  rw [h_total, h_first, h_second]
  exact rfl

end amount_given_to_last_set_l331_331441


namespace sequence_sum_third_fifth_l331_331367

theorem sequence_sum_third_fifth :
  let a : Nat → ℚ := λ n, if n = 1 then 1 else (n : ℚ)^2 / ((n - 1) : ℚ)^2
  in a 3 + a 5 = 61 / 16 := 
by
  let a : Nat → ℚ := λ n, if n = 1 then 1 else (n : ℚ)^2 / ((n - 1) : ℚ)^2
  have h3 : a 3 = 9 / 4 := by sorry
  have h5 : a 5 = 25 / 16 := by sorry
  calc
    a 3 + a 5 = 9 / 4 + 25 / 16 := by rw [h3, h5]
    ... = 36 / 16 + 25 / 16 := by sorry
    ... = 61 / 16 := by sorry

end sequence_sum_third_fifth_l331_331367


namespace PE_parallel_BC_l331_331091

theorem PE_parallel_BC {ABC : Type*} [RightTriangle ABC] (B : Point) (angle_B : ∃ (θ : ℝ), θ = 90) 
  (incircle : Circle) (D E F P : Point) : 
  touches(incircle, BC) D → touches(incircle, CA) E → touches(incircle, AB) F → 
  intersects(AD, incircle) P → perpendicular(PC, PF) → parallel(PE, BC) := 
sorry

end PE_parallel_BC_l331_331091


namespace total_artworks_l331_331433

theorem total_artworks (students art_kits : ℕ)
    (students_per_kit : ℕ → ℕ → ℕ) 
    (artworks_group1 : ℕ → ℕ) 
    (artworks_group2 : ℕ → ℕ)
    (h_students : students = 10)
    (h_art_kits : art_kits = 20)
    (h_students_per_kit : students_per_kit students art_kits = 2)
    (h_group1_size : (students / 2) = 5)
    (h_group2_size : (students / 2) = 5)
    (h_artworks_group1 : (5 * 3) = artworks_group1 5)
    (h_artworks_group2 : (5 * 4) = artworks_group2 5)
    : (artworks_group1 5 + artworks_group2 5) = 35 := 
by 
  rw [h_students, h_art_kits, h_students_per_kit, h_group1_size, h_group2_size, h_artworks_group1, h_artworks_group2]
  sorry

end total_artworks_l331_331433


namespace ratio_of_small_square_to_shaded_area_l331_331809

theorem ratio_of_small_square_to_shaded_area :
  let small_square_area := 2 * 2
  let large_square_area := 5 * 5
  let shaded_area := (large_square_area / 2) - (small_square_area / 2)
  (small_square_area : ℚ) / shaded_area = 8 / 21 :=
by
  sorry

end ratio_of_small_square_to_shaded_area_l331_331809


namespace third_value_of_expression_l331_331846

theorem third_value_of_expression :
  let a := 2 ^ 11,
      b := 2 ^ 5,
      c := 2 in
    let values := [
      a + b + c,
      a + b - c,
      a - b + c,
      a - b - c,
      -a + b + c,
      -a + b - c,
      -a - b + c,
      -a - b - c
    ] in
  (values.bsort (≤)).nth 2 = some 2018 :=
by
  sorry

end third_value_of_expression_l331_331846


namespace xy_square_diff_l331_331053

theorem xy_square_diff (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end xy_square_diff_l331_331053


namespace base_conversion_b_eq_3_l331_331150

theorem base_conversion_b_eq_3 (b : ℕ) (hb : b > 0) :
  (3 * 6^1 + 5 * 6^0 = 23) →
  (1 * b^2 + 3 * b + 2 = 23) →
  b = 3 :=
by {
  sorry
}

end base_conversion_b_eq_3_l331_331150


namespace find_RS_l331_331508

noncomputable def triangle_defs (DE EF : ℝ) (Q R S N : ℝ → Prop) :=
  DE = 600 ∧ EF = 400 ∧ 
  (∃ (Q R : ℝ), DQ = 300 ∧ QE = 300) ∧ 
  (∃ (R : ℝ), ER_angle_bisector R) ∧ 
  (∃ (S : ℝ), intersection_FP_ER S) ∧
  midpoint_SN Q S N ∧ 
  DN = 240

theorem find_RS (DE EF Q DQ QE R ER_angle_bisector S intersection_FP_ER midpoint_SN DN : ℝ) :
  triangle_defs DE EF Q R S N →
  DN = 240 →
  RS = 240 :=
by {
  intros H1 H2,
  sorry
}

end find_RS_l331_331508


namespace alice_wins_if_n_ge_13_l331_331808

theorem alice_wins_if_n_ge_13 (n : ℕ) (paints_red_fields : ℕ) (paints_black_fields : ℕ) : 
  (∀ (bob_plays : list (ℕ × ℕ)), 4 ≤ bob_plays.length → 
    (∃ cell, cell ≠ bob_plays.foldr (λ (rc : ℕ × ℕ) (acc : set (ℕ × ℕ)), acc.insert rc) ∅)
  ) → n ≥ 13 :=
sorry

end alice_wins_if_n_ge_13_l331_331808


namespace parking_spaces_in_the_back_l331_331492

theorem parking_spaces_in_the_back
  (front_spaces : ℕ)
  (cars_parked : ℕ)
  (half_back_filled : ℕ → ℚ)
  (spaces_available : ℕ)
  (B : ℕ)
  (h1 : front_spaces = 52)
  (h2 : cars_parked = 39)
  (h3 : half_back_filled B = B / 2)
  (h4 : spaces_available = 32) :
  B = 38 :=
by
  -- Here you can provide the proof steps.
  sorry

end parking_spaces_in_the_back_l331_331492


namespace rectangular_plot_breadth_l331_331826

theorem rectangular_plot_breadth (b : ℝ) 
    (h1 : ∃ l : ℝ, l = 3 * b)
    (h2 : 432 = 3 * b * b) : b = 12 :=
by
  sorry

end rectangular_plot_breadth_l331_331826


namespace mary_final_weight_l331_331746

def initial_weight : Int := 99
def weight_lost_initially : Int := 12
def weight_gained_back_twice_initial : Int := 2 * weight_lost_initially
def weight_lost_thrice_initial : Int := 3 * weight_lost_initially
def weight_gained_back_half_dozen : Int := 12 / 2

theorem mary_final_weight :
  let final_weight := 
      initial_weight 
      - weight_lost_initially 
      + weight_gained_back_twice_initial 
      - weight_lost_thrice_initial 
      + weight_gained_back_half_dozen
  in final_weight = 78 :=
by
  sorry

end mary_final_weight_l331_331746


namespace range_of_a_l331_331030

-- Given conditions
variable (a : ℝ)

def A (a : ℝ) : set ℝ := {x : ℝ | a * x^2 - 3 * x + 2 = 0}

-- Statement of the proof problem
theorem range_of_a (h : ∀ x1 x2 ∈ A a, x1 = x2) : a ≥ 9 / 8 ∨ a = 0 := 
sorry

end range_of_a_l331_331030


namespace max_f0_is_3_l331_331727

def is_cosine_polynomial (f : ℝ → ℝ) (N : ℕ) : Prop :=
  ∃ (a : ℕ → ℝ), f = λ x, 1 + ∑ n in finset.range N, a n * real.cos (2 * real.pi * n * x) ∧ 
  (∀ n, n % 3 = 0 → a n = 0) ∧ (∀ x, f x ≥ 0)

def C : set (ℝ → ℝ) := { f | ∃ N, is_cosine_polynomial f N }

noncomputable def max_f0 : ℝ :=
  real.Sup { f 0 | f ∈ C }

theorem max_f0_is_3 : ∃ f ∈ C, f 0 = 3 ∧ max_f0 = 3 :=
begin
  sorry
end

end max_f0_is_3_l331_331727


namespace new_person_weight_l331_331530

theorem new_person_weight (N : ℝ) (h : N - 65 = 22.5) : N = 87.5 :=
by
  sorry

end new_person_weight_l331_331530


namespace ratio_OM_PC_l331_331716

open_locale euclidean_geometry

variables {A B C M P O : Point}
variables (h_midpoint_M : midpoint M A C)
variables (h_P_BC : collinear B P C)
variables (h_intersect_O : ∃ O, line_through A P ∧ line_through B M ∧ incidence O B)
variables (h_BO_BP : dist B O = dist B P)

theorem ratio_OM_PC (h_midpoint_M : midpoint M A C) (h_P_BC : collinear B P C) (h_intersect_O : ∃ O, line_through A P ∧ line_through B M ∧ incidence O B) (h_BO_BP : dist B O = dist B P) :
  dist O M = dist O (1 / 2 : ℝ) * (dist P C) :=
sorry -- Proof to be completed

end ratio_OM_PC_l331_331716


namespace part1_part2_l331_331665

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1
noncomputable def g (x : ℝ) : ℝ := f x / x

theorem part1 (t : ℝ) (ht : f t = tangent_line_through_origin f t) : 
  tangent_line_through_origin f t = λ (x : ℝ), x := 
sorry

theorem part2 (x : ℝ) : g x ≤ 1 :=
sorry

end part1_part2_l331_331665


namespace must_true_l331_331651

axiom p : Prop
axiom q : Prop
axiom h1 : ¬ (p ∧ q)
axiom h2 : p ∨ q

theorem must_true : (¬ p) ∨ (¬ q) := by
  sorry

end must_true_l331_331651


namespace max_volume_is_16_l331_331192

noncomputable def max_volume (width : ℝ) (material : ℝ) : ℝ :=
  let l := (material - 2 * width) / (2 + 2 * width)
  let h := (material - 2 * l) / (2 * width + 2 * l)
  l * width * h

theorem max_volume_is_16 :
  max_volume 2 32 = 16 :=
by
  sorry

end max_volume_is_16_l331_331192


namespace correct_statements_l331_331295

noncomputable def binomial_expansion (x : ℝ) : ℝ := (2 * x - 1 / x ^ 2) ^ 6

theorem correct_statements :
  (∀ x, binomial_expansion 1 = 1) ∧
  (∀ x, (∃ C, (2 * x - 1 / x ^ 2) ^ 6 = C * x^0) → C = 240) :=
by {
  sorry
}

end correct_statements_l331_331295


namespace integral_squared_ge_one_third_l331_331398

open Real Interval

theorem integral_squared_ge_one_third 
  (f : ℝ → ℝ) 
  (hc : ContinuousOn f (Icc 0 1))
  (h : ∀ (x ∈ Icc 0 1), ∫ t in Ioc x 1, f t ≥ (1 - x^2) / 2) :
  ∫ t in 0..1, (f t)^2 ≥ 1 / 3 :=
sorry

end integral_squared_ge_one_third_l331_331398


namespace complex_pure_imaginary_is_x_eq_2_l331_331059

theorem complex_pure_imaginary_is_x_eq_2
  (x : ℝ)
  (z : ℂ)
  (h : z = ⟨x^2 - 3 * x + 2, x - 1⟩)
  (pure_imaginary : z.re = 0) :
  x = 2 :=
by
  sorry

end complex_pure_imaginary_is_x_eq_2_l331_331059


namespace range_of_m_l331_331630

theorem range_of_m (m : ℝ) (h1 : |m + 3| = m + 3) (h2 : |3m + 9| ≥ 4m - 3) : -3 ≤ m ∧ m ≤ 12 :=
by
  sorry

end range_of_m_l331_331630


namespace identity_x_squared_minus_y_squared_l331_331042

theorem identity_x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : 
  x^2 - y^2 = 80 := by sorry

end identity_x_squared_minus_y_squared_l331_331042


namespace tiger_chase_speed_l331_331918

-- Conditions as definitions
def escape_time := 1 -- 1 AM
def notice_time := 4 -- 4 AM
def find_time := 8 -- 8 AM
def initial_speed := 25 -- mph
def post_slow_speed := 10 -- mph
def total_distance := 135 -- miles
def chase_time := 0.5 -- hours

-- Mathematically equivalent proof problem
theorem tiger_chase_speed :
  ∀ (t1 t2 t3 _: ℕ) (v1 v2 total d_chase : ℕ),
    t1 = 1 →                       -- escape time
    t2 = 4 →                       -- notice time
    t3 = 8 →                       -- find time
    v1 = 25 →                      -- initial speed
    v2 = 10 →                      -- post-slow speed
    total = 135 →                  -- total distance
    d_chase = 0.5 →                -- chase time (hours)
    let d1 := v1 * (t2 - t1),
        d2 := v2 * (t3 - t2),
        d_total := d1 + d2,
        chase_distance := total - d_total,
        chase_speed := chase_distance / d_chase
    in chase_speed = 40 :=         -- prove chase speed is 40 mph
by {
  intros t1 t2 t3 t4 v1 v2 total d_chase ht1 ht2 ht3 hv1 hv2 htotal hd_chase,
  let d1 := v1 * (t2 - t1),
  let d2 := v2 * (t3 - t2),
  let d_total := d1 + d2,
  let chase_distance := total - d_total,
  let chase_speed := chase_distance / d_chase,
  have : chase_speed = 40, 
  { sorry },
  exact this,
}

end tiger_chase_speed_l331_331918


namespace frustum_volume_fraction_l331_331915

theorem frustum_volume_fraction (base_edge_orig alt_orig : ℝ) (h_base : base_edge_orig = 24) (h_alt : alt_orig = 18) :
  let alt_small := alt_orig / 3
  in let scale_factor := alt_small / alt_orig
     in let vol_ratio := scale_factor ^ 3
        in let resultant_volume_fraction := 1 - vol_ratio
           in resultant_volume_fraction = 26 / 27 :=
by
  sorry

end frustum_volume_fraction_l331_331915


namespace cos_sqr_sub_power_zero_l331_331888

theorem cos_sqr_sub_power_zero :
  (cos (30 * Real.pi / 180))^2 - (2 - Real.pi)^0 = -1/4 :=
by
  sorry

end cos_sqr_sub_power_zero_l331_331888


namespace obtuse_angle_probability_l331_331132

-- Define the points P, Q, R, S, T, U
structure Point where
  x : ℝ
  y : ℝ

def P : Point := ⟨0, 3⟩
def R : Point := ⟨3, 0⟩
def S : Point := ⟨3 + 2 * π, 0⟩
def T : Point := ⟨3 + 2 * π, 5⟩
def U : Point := ⟨0, 5⟩

-- Probability calculation for angle PQR being obtuse
def probability_obtuse : ℝ :=
  (60 + 31 * π) / (60 + 40 * π)

-- Define the conditions and prove the result
theorem obtuse_angle_probability :
  ∃ Q : Point, 
  Q ∈ interior of pentagon P U S T R ∧ 
  obtuse (angle P Q R) →
  probability_obtuse =
  (60 + 31 * π) / (60 + 40 * π) :=
by
  sorry -- Proof is omitted

end obtuse_angle_probability_l331_331132


namespace opposite_face_of_3_l331_331833

def numbers_on_faces := {1, 2, 3, 4, 5, 6}

def sum_all_faces (faces : Set Nat) : Nat :=
  faces.foldl (· + ·) 0

def sum_lateral_faces (rolls : List (Set Nat)) : List Nat :=
  rolls.map sum_all_faces

theorem opposite_face_of_3 (r1 r2 : Set Nat) (number3 : Nat) (number6 : Nat)
  (cond1 : number3 = 3)
  (cond2 : number6 = 6)
  (cond3 : sum_all_faces numbers_on_faces = 21)
  (cond4 : sum_all_faces r1 = 12)
  (cond5 : sum_all_faces r2 = 15)
  (cond6 : ∀ x y z w : Nat, x ∈ r1 ∧ y ∈ r1 ∧ z ∈ r1 ∧ w ∈ r1 → x + y + z + w = 12)
  (cond7 : ∀ x y z w : Nat, x ∈ r2 ∧ y ∈ r2 ∧ z ∈ r2 ∧ w ∈ r2 → x + y + z + w = 15) :
  ∃ b : Bool, b = true ∧ ((¬b) → (number6 = number3.opposite)) :=
sorry

end opposite_face_of_3_l331_331833


namespace product_sum_identity_l331_331737

theorem product_sum_identity (n : ℕ) (x : Fin n → ℝ) (h_diff : ∀ i j : Fin n, i ≠ j → x i ≠ x j) : 
  (∑ i, ∏ j in Finset.univ \ {i}, (1 - x i * x j) / (x i - x j)) = 
  if even n then 0 else 1 := 
sorry

end product_sum_identity_l331_331737


namespace system_solution_l331_331495

theorem system_solution : 
  ∀ (x y Δ : ℕ), 
  x = 1 → y = 2 → 
  (2 * x + y = Δ ↔ Δ = 4) ∧ (x + y = 3) :=
by
  intros x y Δ hx hy
  split
  { intro h
    rw [hx, hy] at h
    exact eq_add_of_sub h rfl }
  { intro h
    rw [hx, hy] at h
    linarith }
  sorry

end system_solution_l331_331495


namespace relationship_among_solutions_l331_331731

theorem relationship_among_solutions : 
  let a := (∃ x, 2 * x + x = 1) in
  let b := (∃ x, 2 * x + x = 2) in
  let c := (∃ x, 3 * x + x = 2) in
  a < c ∧ c < b :=
by
  sorry

end relationship_among_solutions_l331_331731


namespace average_percentage_l331_331220

theorem average_percentage (average_15 : ℕ) (n_15 : ℕ) (n_10 : ℕ) (total_average : ℕ) (n : ℕ) :
  n = n_15 + n_10 →
  total_average * n = n_15 * average_15 + n_10 * (total_average * n - n_15 * average_15) / n_10 :=
by
  intros
  rw [← mul_comm n_10, ← mul_div_assoc, ← sub_eq_iff_eq_add]
  ring
  sorry

end average_percentage_l331_331220


namespace triangle_area_ratio_l331_331401

open EuclideanGeometry

theorem triangle_area_ratio (A B C P : Point) 
  (h₁ : Let PA = vector_from P A) 
  (h₂ : Let PB = vector_from P B)
  (h₃ : Let PC = vector_from P C)
  (h₄ : PA + 3 * PB + 4 * PC = 0) :
  let area_ABC := area_of_triangle A B C
  let area_APB := area_of_triangle A P B
  area_ABC / area_APB = 5 / 2 :=
sorry

end triangle_area_ratio_l331_331401


namespace set_representation_l331_331171

theorem set_representation : {x | 8 < x ∧ x < 12 ∧ x ∈ Nat} = {9, 10, 11} :=
by
  sorry

end set_representation_l331_331171


namespace mode_of_scores_is_97_l331_331837

-- Define the stem-and-leaf plot as lists of integers
def stemAndLeafPlot : List (Nat × List Nat) :=
[
  (7, [5, 5]),
  (8, [1, 1, 2, 2, 2, 9, 9]),
  (9, [3, 4, 4, 7, 7, 7, 7]),
  (10, [6]),
  (11, [2, 2, 4, 4, 4]),
  (12, [0])
]

-- Extract the scores from the stem-and-leaf plot
def scores : List Nat :=
  stemAndLeafPlot.bind (λ (s, ls) => ls.map (λ l => s * 10 + l))

-- Define the mode function
def mode (xs : List Nat) : Option Nat :=
  let freqMap := xs.foldl (λ m x => m.insert x (m.findD x 0 + 1)) Std.RBMap.empty
  freqMap.fold (λ acc key count =>
    match acc with
    | none => some key
    | some (k, c) => if count > c then some (key, count) else acc) none >>= (λ (k, _) => some k)

-- The proof statement
theorem mode_of_scores_is_97 : mode scores = some 97 := by
  sorry

end mode_of_scores_is_97_l331_331837


namespace problemA_problemB_l331_331929

-- Definition and statement for Problem (a)
def arrangementLine (n m : ℕ) : ℕ :=
  n.factorial * (nat.choose (n + 1) m) * m.factorial

theorem problemA (n m : ℕ) : 
  arrangementLine n m = n.factorial * (nat.choose (n + 1) m) * m.factorial := 
by
  rw [arrangementLine]
  sorry

-- Definition and statement for Problem (b)
def arrangementCircle (n m : ℕ) : ℕ :=
  (n - 1).factorial * (nat.choose n m) * m.factorial

theorem problemB (n m : ℕ) (h1 : 0 < n) (h2 : 0 < m ∧ m ≤ n) : 
  arrangementCircle n m = (n - 1).factorial * (nat.choose n m) * m.factorial := 
by
  rw [arrangementCircle]
  sorry

end problemA_problemB_l331_331929


namespace cubic_yards_to_cubic_feet_l331_331680

theorem cubic_yards_to_cubic_feet :
  (1 : ℝ) * 3^3 * 5 = 135 := by
sorry

end cubic_yards_to_cubic_feet_l331_331680


namespace meaningful_sqrt_x_minus_5_l331_331363

theorem meaningful_sqrt_x_minus_5 (x : ℝ) (h : sqrt (x - 5) ∈ ℝ) : x = 6 ∨ x ≥ 5 := by
  sorry

end meaningful_sqrt_x_minus_5_l331_331363


namespace problem1_problem2_l331_331301

variables {a m n : ℝ}

-- Conditions as Lean definitions
def condition1 : Prop := a > 0
def condition2 : Prop := a ≠ 1
def condition3 : Prop := a ^ m = 4
def condition4 : Prop := a ^ n = 3

-- Define the first proof problem for a^{-m/2}
theorem problem1 (h1 : condition1) (h2 : condition2) (h3 : condition3) : a ^ - (m / 2) = 1 / 2 :=
  sorry

-- Define the second proof problem for a^{2m-n}
theorem problem2 (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : a ^ (2 * m - n) = 16 / 3 :=
  sorry

end problem1_problem2_l331_331301


namespace ellipse_eq_proof_collinear_DNQ_l331_331638

-- Definitions as per conditions
def ellipse_eq (a b : ℝ) : (ℝ × ℝ) → Prop := fun p => (p.1 / a)^2 + (p.2 / b)^2 = 1

def A : (ℝ × ℝ) := (0, 1)
def B : (ℝ × ℝ) := (0, -1)
def e : ℝ := sqrt 3 / 2

-- Proving the equation of the ellipse 
theorem ellipse_eq_proof (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
(h3 : distance A B = 2) (h4 : e = sqrt 3 / 2)
(h5 : a ^ 2 = b ^ 2 + c ^ 2) : 
  ellipse_eq 2 1 = fun p => (p.1 / 2)^2 + (p.2 / 1)^2 = 1 := 
sorry

-- Proving collinearity of points D, N, Q
theorem collinear_DNQ (a b : ℝ) (h1 : a > b) (h2 : b > 0)
{P Q D N : (ℝ × ℝ)} (hP : ellipse_eq a b P)
(hQ : ellipse_eq a b Q) (hMN : midpoint O P = M) (hN : midpoint B P = N)
(hAD_intersect : ∃ D, line_through A M ∩ ellipse_eq a b = {A, D}) 
(hD_eq : D = (2 * (2 - P.2) / (5 - 4 * P.2), (-2 * (P.2)^2 + 4 * P.2 - 3) / (5 - 4 * P.2)))
(kQN : slope Q N = -(P.2 + 1) / (3 * P.1)) 
(kQD : slope Q D = -(P.2 + 1) / (3 * P.1)) : 
  collinear D N Q :=
sorry

end ellipse_eq_proof_collinear_DNQ_l331_331638


namespace total_fencing_length_l331_331899

/-- Given a garden in the shape of a square with an area of 784 square meters and an additional 
    10 meters of fencing required on each side, the total length of fencing needed 
    is 152 meters. -/
theorem total_fencing_length (area : ℕ) (extension : ℕ) 
  (h1 : area = 784) (h2 : extension = 10) : 
  4 * (Nat.sqrt area + extension) = 152 := 
begin
  -- Proof is omitted since only the Lean 4 statement is required
  sorry
end

end total_fencing_length_l331_331899


namespace tangent_triangle_perimeter_acute_tangent_triangle_perimeter_obtuse_l331_331714

theorem tangent_triangle_perimeter_acute (a b c: ℝ) (h1: a^2 + b^2 > c^2) (h2: b^2 + c^2 > a^2) (h3: c^2 + a^2 > b^2) :
  2 * a * b * c * (1 / (b^2 + c^2 - a^2) + 1 / (c^2 + a^2 - b^2) + 1 / (a^2 + b^2 - c^2)) = 
  2 * a * b * c * (1 / (b^2 + c^2 - a^2) + 1 / (c^2 + a^2 - b^2) + 1 / (a^2 + b^2 - c^2)) := 
by sorry -- proof goes here

theorem tangent_triangle_perimeter_obtuse (a b c: ℝ) (h1: a^2 > b^2 + c^2) :
  2 * a * b * c / (a^2 - b^2 - c^2) = 2 * a * b * c / (a^2 - b^2 - c^2) := 
by sorry -- proof goes here

end tangent_triangle_perimeter_acute_tangent_triangle_perimeter_obtuse_l331_331714


namespace inequality_abc_l331_331738

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b))) ≥ 3 / 2 := 
sorry

end inequality_abc_l331_331738


namespace greatest_divisor_remainders_l331_331607

theorem greatest_divisor_remainders (x : ℕ) (h1 : 1255 % x = 8) (h2 : 1490 % x = 11) : x = 29 :=
by
  -- The proof steps would go here, but for now, we use sorry.
  sorry

end greatest_divisor_remainders_l331_331607


namespace exists_face_insphere_l331_331857

-- Definitions in Lean
structure Tetrahedron :=
  (vertices : fin 4 → ℝ × ℝ × ℝ)
  (edges : fin 6 → fin 3 → fin 3)
  (insphere : ℝ × ℝ × ℝ → ℝ) -- Sphere touching all edges

-- Proposition to be proved
theorem exists_face_insphere (T : Tetrahedron) 
  (h : ∃ (S : ℝ × ℝ × ℝ → ℝ), ∃ (f : fin 3), S touches_edges (T.edges f) ∧ S touches_extensions (T.edges \ {f})) :
  ∀ f : fin 4, ∃ (S : ℝ × ℝ × ℝ → ℝ), S touches_edges (T.edges f) ∧ S touches_extensions (T.edges \ {f}) := 
sorry

end exists_face_insphere_l331_331857


namespace root_expression_value_l331_331419

-- Define p and q as the roots of the quadratic equation
variable (p q : ℝ)
variable h_root_p : 3 * p ^ 2 - 9 * p - 15 = 0
variable h_root_q : 3 * q ^ 2 - 9 * q - 15 = 0

-- Prove that (3p-5)(6q-10) is equal to -130 under the given conditions
theorem root_expression_value :
  (3 * p - 5) * (6 * q - 10) = -130 :=
by
  sorry

end root_expression_value_l331_331419


namespace Mitch_saved_amount_l331_331124

theorem Mitch_saved_amount :
  let boat_cost_per_foot := 1500
  let license_and_registration := 500
  let docking_fees := 3 * 500
  let longest_boat_length := 12
  let total_license_and_fees := license_and_registration + docking_fees
  let total_boat_cost := boat_cost_per_foot * longest_boat_length
  let total_saved := total_boat_cost + total_license_and_fees
  total_saved = 20000 :=
by
  sorry

end Mitch_saved_amount_l331_331124


namespace side_lengths_of_squares_l331_331481

theorem side_lengths_of_squares :
  let d1 := 2 * Real.sqrt 2
  let d2 := 2 * d1
  ∃ (s1 s2 : ℝ), d1 = s1 * Real.sqrt 2 ∧ d2 = s2 * Real.sqrt 2 ∧ s1 = 2 ∧ s2 = 4 :=
by 
  let d1 := 2 * Real.sqrt 2
  let d2 := 2 * d1
  use 2
  use 4
  split
  { calc d1 = 2 * Real.sqrt 2 : by rfl
       ... = 2 * (Real.sqrt 2) : by rfl },
  split
  { calc d2 = 2 * d1 : by rfl
       ... = 2 * (2 * Real.sqrt 2) : rfl
       ... = 4 * Real.sqrt 2 : by ring },
  split
  { rfl },
  { rfl }
  sorry  -- skipping proof completion

end side_lengths_of_squares_l331_331481


namespace geometric_sequence_min_value_l331_331105

theorem geometric_sequence_min_value (r : ℝ) (a1 a2 a3 : ℝ) 
  (h1 : a1 = 1) 
  (h2 : a2 = a1 * r) 
  (h3 : a3 = a2 * r) :
  4 * a2 + 5 * a3 ≥ -(4 / 5) :=
by
  sorry

end geometric_sequence_min_value_l331_331105


namespace logarithmic_expression_l331_331219

theorem logarithmic_expression : 
  (\lg (Real.sqrt 27) + \lg 8 - log 4 8) / ((1 / 2) * \lg 0.3 + \lg 2) = 3 :=
sorry

end logarithmic_expression_l331_331219


namespace mean_comparison_l331_331589

variables {a b : ℝ} (g : ℝ → ℝ)
-- assumptions
variables (ha : 0 < a) (hb : 0 < b)
variables (hpos : ∀ x, 0 < g x) (hderiv_pos : ∀ x, 0 < g' x) (hsec_pos : ∀ x, 0 < g'' x)

-- defining the arithmetic mean
def m (a b : ℝ) : ℝ :=
(a + b) / 2

-- defining the mean value μ with respect to g
noncomputable def μ (a b : ℝ) : ℝ :=
Classical.some (exists_unique (λ μ, 2 * g μ = g a + g b))

-- statement to prove
theorem mean_comparison : μ g a b ≥ m a b :=
sorry

end mean_comparison_l331_331589


namespace steps_to_return_l331_331349

def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def walking_game (n : ℕ) : ℤ :=
  if n = 1 then 0
  else if is_prime n then -1
  else 3

def total_movement : ℤ :=
  ((list.range 30).map (λ n, walking_game (n + 1))).sum

theorem steps_to_return : total_movement = 47 :=
  by sorry

end steps_to_return_l331_331349


namespace coefficient_x4_in_expansion_equal_coefficients_implies_r_eq_1_l331_331018

-- Problem (I)
theorem coefficient_x4_in_expansion :
  let T := (x - (2 / √x))^10 in
  ∃ k: ℕ, (10 - (3 / 2) * k) = 4 ∧ ((-2)^k * Nat.choose 10 k) = 3360 := by
  sorry

-- Problem (II)
theorem equal_coefficients_implies_r_eq_1 :
  let T := (x - (2 / √x))^10 in 
  ∃ r: ℕ, Nat.choose 10 (3 * r - 1) = Nat.choose 10 (r + 1) → r = 1 := by
  sorry

end coefficient_x4_in_expansion_equal_coefficients_implies_r_eq_1_l331_331018


namespace find_sin_θ_l331_331646

open Real

noncomputable def θ_in_range_and_sin_2θ (θ : ℝ) : Prop :=
  (θ ∈ Set.Icc (π / 4) (π / 2)) ∧ (sin (2 * θ) = 3 * sqrt 7 / 8)

theorem find_sin_θ (θ : ℝ) (h : θ_in_range_and_sin_2θ θ) : sin θ = 3 / 4 :=
  sorry

end find_sin_θ_l331_331646


namespace arithmetic_sequence_1001th_term_l331_331821

theorem arithmetic_sequence_1001th_term (p q : ℤ)
  (h1 : 9 - p = (2 * q - 5))
  (h2 : (3 * p - q + 7) - 9 = (2 * q - 5)) :
  p + (1000 * (2 * q - 5)) = 5004 :=
by
  sorry

end arithmetic_sequence_1001th_term_l331_331821


namespace find_k_l331_331400

theorem find_k (k : ℝ) (A B : ℝ × ℝ) 
  (hA : A = (2, 3)) (hB : B = (4, k)) 
  (hAB_parallel : A.2 = B.2) : k = 3 := 
by 
  have hA_def : A = (2, 3) := hA 
  have hB_def : B = (4, k) := hB 
  have parallel_condition: A.2 = B.2 := hAB_parallel
  simp at parallel_condition
  sorry

end find_k_l331_331400


namespace probability_is_correct_l331_331545

-- Define the ratios for the colors: red, yellow, blue, black
def red_ratio := 6
def yellow_ratio := 2
def blue_ratio := 1
def black_ratio := 4

-- Define the total ratio
def total_ratio := red_ratio + yellow_ratio + blue_ratio + black_ratio

-- Define the ratio of red or blue regions
def red_or_blue_ratio := red_ratio + blue_ratio

-- Define the probability of landing on a red or blue region
def probability_red_or_blue := red_or_blue_ratio / total_ratio

-- State the theorem to prove
theorem probability_is_correct : probability_red_or_blue = 7 / 13 := 
by 
  -- Proof will go here
  sorry

end probability_is_correct_l331_331545


namespace find_f_of_f_of_f_neg2_l331_331627

def f (x : ℝ) : ℝ :=
  if x > 0 then x + 2 else if x = 0 then Real.pi else 0

theorem find_f_of_f_of_f_neg2 : f (f (f (-2))) = Real.pi + 2 := by
  sorry

end find_f_of_f_of_f_neg2_l331_331627


namespace multiples_of_7_between_10_6_and_10_9_are_perfect_squares_l331_331347

theorem multiples_of_7_between_10_6_and_10_9_are_perfect_squares :
  (∃ n : ℕ, 10^6 ≤ 49 * n^2 ∧ 49 * n^2 ≤ 10^9 ∧ (4376 = (4517 - 142 + 1))) :=
begin
  sorry
end

end multiples_of_7_between_10_6_and_10_9_are_perfect_squares_l331_331347


namespace exists_4_good_quadruplet_maximal_k_good_quadruplet_l331_331233

def is_arithmetic_progression (a b c : ℕ) : Prop := 2 * b = a + c

def distinct_not_ap (a b c d : ℕ) : Prop := 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  ¬is_arithmetic_progression a b c ∧ ¬is_arithmetic_progression a b d ∧
  ¬is_arithmetic_progression a c d ∧ ¬is_arithmetic_progression b c d

def sum_is_ap (sums : List ℕ) : ℕ → Prop
| 0      := true
| n + 1  := match sums with
            | []      => false
            | [x]     => false
            | [x, y]  => false
            | x :: y :: z :: tl => is_arithmetic_progression x y z ∧ list_pairwise sum_is_ap tl n -- Helper function for pairwise processing


def is_k_good (a b c d k : ℕ) : Prop := 
  distinct_not_ap a b c d ∧
  ∃ sums : List ℕ, sums = [a+b, a+c, a+d, b+c, b+d, c+d] ∧ sum_is_ap sums k

theorem exists_4_good_quadruplet : ∃ a b c d, is_k_good a b c d 4 := 
by {
  use [1, 2, 4, 5],
  split,
  -- verifying distinct_not_ap (1, 2, 4, 5)
  split; simp [distinct_not_ap, is_arithmetic_progression],
  -- verifying sum_is_ap
  existsi [3, 5, 6, 7, 7, 9], 
  split; rfl, sorry
}

theorem maximal_k_good_quadruplet : ∀ a b c d k, is_k_good a b c d k → k ≤ 4 := sorry

end exists_4_good_quadruplet_maximal_k_good_quadruplet_l331_331233


namespace area_ratio_of_point_in_triangle_l331_331404

variables {A B C P : Type} [AffSpace A]

noncomputable def vector_eq_condition (PA PB PC : Vec A) : Vec A :=
PA + 3 • PB + 4 • PC

theorem area_ratio_of_point_in_triangle
  (A B C P : Vec A)
  (h : vector_eq_condition (A - P) (B - P) (C - P) = 0) :
  (area_of_triangle A B C) / (area_of_triangle A P B) = 5 / 2 := 
sorry

end area_ratio_of_point_in_triangle_l331_331404


namespace cost_of_date_book_l331_331234

theorem cost_of_date_book
  (cost_calendar : ℝ := 0.75)
  (num_calendars : ℕ := 300)
  (num_date_books : ℕ := 200)
  (total_cost : ℝ := 300) :
  let
    cost_date_book : ℝ := 0.375
  in
    num_calendars * cost_calendar + num_date_books * cost_date_book = total_cost :=
by
  sorry

end cost_of_date_book_l331_331234


namespace triangle_construction_iff_l331_331588

variable (b c α : ℝ)
variable (h1 : 0 < α ∧ α < π / 2)

theorem triangle_construction_iff :
  (b * Real.tan (α / 2) ≤ c ∧ c < b) ↔ 
  (∃ (A B C M : EucVec3), 
     dist A C = b ∧
     dist A B = c ∧
     ∠ A M B = α ∧
     midpoint M B C) := sorry

end triangle_construction_iff_l331_331588


namespace solve_for_x_l331_331786

theorem solve_for_x (x : ℝ) : (5 * x - 2) / (6 * x - 6) = 3 / 4 ↔ x = -5 := by
  sorry

end solve_for_x_l331_331786


namespace length_of_BD_l331_331706

theorem length_of_BD (b : ℝ) (h1 : ∠ ABC = 90) (h2 : ∠ ABD = 90) (BC : ℝ = 2) (AC : ℝ = b) (AD : ℝ = 3) :
  BD = real.sqrt (b^2 - 5) := sorry

end length_of_BD_l331_331706


namespace find_angle_BAC_l331_331375

-- Definitions based on conditions
def acute_triangle (A B C : Point) : Prop := ∀a b c: ℝ, 0 < a+b+c < π
def projection (A B C D : Point) : Prop := D ∈ line B C ∧ angle A D B = 90°
def midpoint (A C M : Point) : Prop := ∃b c, M ∈ line B C ∧ A B = A C ∧ B C = 2 * B M
def on_segment (P B M : Point) : Prop := P ∈ segment B M
def angle_equality (A P M B : Point) : Prop := angle P A M = angle M B A
def given_angles (A B P D : Point) : Prop := angle B A P = 41° ∧ angle P D B = 115°

-- Conclusion
theorem find_angle_BAC {A B C D M P : Point} 
  (h₁ : acute_triangle A B C) 
  (h₂ : projection A B C D) 
  (h₃ : midpoint A C M) 
  (h₄ : on_segment P B M) 
  (h₅ : angle_equality A P M B) 
  (h₆ : given_angles A B P D) : 
  angle B A C = 94° := sorry

end find_angle_BAC_l331_331375


namespace alcohol_solution_percentage_l331_331681

theorem alcohol_solution_percentage :
  let P : ℝ := 0.2 in
  40 * P + 60 * 0.7 = 100 * 0.5 :=
by
  sorry

end alcohol_solution_percentage_l331_331681


namespace smallest_positive_period_of_f_max_min_values_of_f_on_interval_l331_331329

noncomputable def f (x : ℝ) : ℝ := cos x * sin (x + (real.pi / 9)) - cos x^2

theorem smallest_positive_period_of_f :
  is_periodic f (real.pi) ∧ (∀ T > 0, is_periodic f T → T >= real.pi) :=
by sorry

theorem max_min_values_of_f_on_interval :
  ∃ max min,
    max = (1/4 : ℝ) ∧ min = (-1/4 : ℝ) ∧ 
    (∀ x ∈ (set.Icc 0 (real.pi / 2)), f x ≤ max ∧ f x ≥ min) :=
by sorry

end smallest_positive_period_of_f_max_min_values_of_f_on_interval_l331_331329


namespace head_start_time_l331_331537

-- Definitions based on conditions
def distance : ℕ := 1000 -- 1000 meters
def time : ℕ := 190     -- 190 seconds
def head_start_distance : ℕ := 50 -- 50 meters

-- The problem translated to Lean statement
theorem head_start_time :
  let speed := (distance : ℚ) / (time : ℚ),
      head_start_time := (head_start_distance : ℚ) / speed
  in head_start_time = 9.5 :=
by
  sorry

end head_start_time_l331_331537


namespace track_width_track_area_l331_331561

theorem track_width (r1 r2 : ℝ) (h1 : 2 * π * r1 - 2 * π * r2 = 24 * π) : r1 - r2 = 12 :=
by sorry

theorem track_area (r1 r2 : ℝ) (h1 : r1 = r2 + 12) : π * (r1^2 - r2^2) = π * (24 * r2 + 144) :=
by sorry

end track_width_track_area_l331_331561


namespace whiskers_count_l331_331621

variable (P C S : ℕ)

theorem whiskers_count :
  P = 14 →
  C = 2 * P - 6 →
  S = P + C + 8 →
  C = 22 ∧ S = 44 :=
by
  intros hP hC hS
  rw [hP] at hC
  rw [hP, hC] at hS
  exact ⟨hC, hS⟩

end whiskers_count_l331_331621


namespace not_line_parallel_to_any_line_in_plane_l331_331689

-- Definitions
variables {ℝ : Type} {P L1 L2 : set ℝ} [plane P] [line L1] [line L2]

-- Hypothesis: L1 is parallel to P
def line_parallel_plane (L1 : set ℝ) (P : set ℝ) : Prop := ∀ (p₁ p₂ ∈ P), (L1 ∩ line_through p₁ p₂) = ∅

-- Statement to prove: If a line is parallel to a plane, then it is parallel to any line in the plane is incorrect.
theorem not_line_parallel_to_any_line_in_plane 
  (h : line_parallel_plane L1 P) :
  ¬∀ (L2 : set ℝ), (L2 ⊆ P) → (L1 ∩ L2 ≠ ∅ → parallel L1 L2) :=
sorry

end not_line_parallel_to_any_line_in_plane_l331_331689


namespace ratio_of_fresh_produce_to_soda_l331_331595

noncomputable def weight_of_empty_truck := 12000
noncomputable def weight_of_soda_crate := 50
noncomputable def number_of_soda_crates := 20
noncomputable def weight_of_dryer := 3000
noncomputable def number_of_dryers := 3
noncomputable def weight_of_fully_loaded_truck := 24000

theorem ratio_of_fresh_produce_to_soda :
    let weight_of_soda := number_of_soda_crates * weight_of_soda_crate,
        weight_of_dryers := number_of_dryers * weight_of_dryer,
        total_weight_except_fresh_produce := weight_of_empty_truck + weight_of_soda + weight_of_dryers,
        weight_of_fresh_produce := weight_of_fully_loaded_truck - total_weight_except_fresh_produce,
        weight_of_fresh_produce_to_soda_ratio := weight_of_fresh_produce / weight_of_soda
    in weight_of_fresh_produce_to_soda_ratio = 2 :=
by
  sorry

end ratio_of_fresh_produce_to_soda_l331_331595


namespace find_width_of_metallic_sheet_l331_331551

noncomputable def width_of_metallic_sheet (w : ℝ) : Prop :=
  let length := 48
  let square_side := 8
  let new_length := length - 2 * square_side
  let new_width := w - 2 * square_side
  let height := square_side
  let volume := new_length * new_width * height
  volume = 5120

theorem find_width_of_metallic_sheet (w : ℝ) :
  width_of_metallic_sheet w -> w = 36 := 
sorry

end find_width_of_metallic_sheet_l331_331551


namespace increasing_on_interval_implies_a_le_7_l331_331625

noncomputable def f (a x : ℝ) : ℝ := x^3 + 2*x^2 - a*x + 1
noncomputable def f' (a x : ℝ) : ℝ := 3*x^2 + 4*x - a

theorem increasing_on_interval_implies_a_le_7 {a : ℝ} :
  (∀ x ∈ set.Icc (1:ℝ) (2:ℝ), 0 ≤ f' a x) → a ≤ 7 :=
by
  -- This is proved in the solution provided
  sorry

end increasing_on_interval_implies_a_le_7_l331_331625


namespace max_sphere_radius_l331_331991

open Real

def square_pyramid := Type

variables (M A B C D : square_pyramid)

def condition1 : Prop := ∀ M A D : square_pyramid, dist M A = dist M D
def condition2 : Prop := ∀ M A B : square_pyramid, angle M A B = π / 2
def condition3 : Prop := ∀ M A D : square_pyramid, area (triangle M A D) = 1

theorem max_sphere_radius (h1: condition1 M A D) (h2 : condition2 M A B) (h3 : condition3 M A D) : 
    Real.Sqrt 2 - 1 = 
sorry

end max_sphere_radius_l331_331991


namespace number_square_25_l331_331166

theorem number_square_25 (x : ℝ) : x^2 = 25 ↔ x = 5 ∨ x = -5 := 
sorry

end number_square_25_l331_331166


namespace domain_of_f_l331_331811

-- Mathematical definitions for the conditions
def condition1 (x : ℝ) : Prop := 2 ^ x - (1 / 2) ≥ 0
def condition2 (x : ℝ) : Prop := x + 1 ≠ 0

-- Problem statement: prove that if both conditions hold, then x > -1
theorem domain_of_f (x : ℝ) : (condition1 x ∧ condition2 x) → x > -1 :=
sorry

end domain_of_f_l331_331811


namespace election_invalid_votes_percentage_l331_331077

theorem election_invalid_votes_percentage (x : ℝ) :
  (∀ (total_votes valid_votes_in_favor_of_A : ℝ),
    total_votes = 560000 →
    valid_votes_in_favor_of_A = 357000 →
    0.75 * ((1 - x / 100) * total_votes) = valid_votes_in_favor_of_A) →
  x = 15 :=
by
  intro h
  specialize h 560000 357000 (rfl : 560000 = 560000) (rfl : 357000 = 357000)
  sorry

end election_invalid_votes_percentage_l331_331077


namespace sequence_general_formula_l331_331310

theorem sequence_general_formula (a : ℕ+ → ℝ) (h₀ : a 1 = 7 / 8)
  (h₁ : ∀ n : ℕ+, a (n + 1) = 1 / 2 * a n + 1 / 3) :
  ∀ n : ℕ+, a n = 5 / 24 * (1 / 2)^(n - 1 : ℕ) + 2 / 3 :=
by
  sorry

end sequence_general_formula_l331_331310


namespace arithmetic_sequence_geometric_condition_l331_331927

theorem arithmetic_sequence_geometric_condition :
  ∃ d : ℝ, d ≠ 0 ∧ (∀ (a_n : ℕ → ℝ), (a_n 1 = 1) ∧ 
    (a_n 3 = a_n 1 + 2 * d) ∧ (a_n 13 = a_n 1 + 12 * d) ∧ 
    (a_n 3 ^ 2 = a_n 1 * a_n 13) ↔ d = 2) :=
by 
  sorry

end arithmetic_sequence_geometric_condition_l331_331927


namespace rationalize_denominator_l331_331766

theorem rationalize_denominator :
  ∃ A B C D : ℝ, 
    (1 / (real.cbrt 5 - real.cbrt 3) = (real.cbrt A + real.cbrt B + real.cbrt C) / D) ∧ 
    A + B + C + D = 51 :=
  sorry

end rationalize_denominator_l331_331766


namespace long_side_length_l331_331874

theorem long_side_length (total_wire_length short_side_length : ℕ) 
  (h1 : total_wire_length = 30) 
  (h2 : short_side_length = 7) : 
  (2 * short_side_length + 2 * long_side_length = total_wire_length) → 
  long_side_length = 8 :=
by
  sorry

end long_side_length_l331_331874


namespace trig_identity_solution_l331_331300

theorem trig_identity_solution (m : ℝ) : 
  (sin (10 * real.pi / 180) + m * cos (10 * real.pi / 180) = 2 * cos (140 * real.pi / 180)) → 
  m = -real.sqrt 3 :=
sorry

end trig_identity_solution_l331_331300


namespace final_answer_l331_331006

def f (x : ℝ) : ℝ := sorry

axiom functional_eqn : ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem final_answer : f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : ℝ, f (-x) = f x) ∧ ¬ (∃ ε > 0, ∀ h : ℝ, abs h < ε → f h ≥ f 0) := 
by
  -- omit the proof steps that were provided in the solution
  sorry

end final_answer_l331_331006


namespace circle_log_exp_intersect_l331_331633

open Real

theorem circle_log_exp_intersect (x1 y1 x2 y2 : ℝ) (h_intersect1 : y1 = log 2015 x1) (h_intersect2 : y2 = 2015^x2) (h_circle1 : x1^2 + y1^2 = 2) (h_circle2 : x2^2 + y2^2 = 2) (h_symmetry : x1 = y2 ∧ x2 = y1) : x1^2 + x2^2 = 2 := by
  sorry

end circle_log_exp_intersect_l331_331633


namespace cannot_be_2008_times_greater_l331_331935

theorem cannot_be_2008_times_greater (n m k l : ℕ) (h1 : 2^k ≤ m) (h2 : m < 2^(k+1)) (h3 : 3^l ≤ m) (h4 : m < 3^(l+1)) :
  ¬ (least_common_multiple (finset.range (n + 1)) = 2008 * least_common_multiple (finset.range (m + 1))) :=
sorry

end cannot_be_2008_times_greater_l331_331935


namespace invalid_votes_percentage_l331_331080

def total_votes : ℕ := 560000
def valid_votes_A : ℕ := 357000
def percentage_A : ℝ := 0.75
def invalid_percentage (x : ℝ) : Prop := (percentage_A * (1 - x / 100) * total_votes = valid_votes_A)

theorem invalid_votes_percentage : ∃ x : ℝ, invalid_percentage x ∧ x = 15 :=
by 
  use 15
  unfold invalid_percentage
  sorry

end invalid_votes_percentage_l331_331080


namespace running_track_diameter_l331_331586

theorem running_track_diameter 
  (running_track_width : ℕ) 
  (garden_ring_width : ℕ) 
  (play_area_diameter : ℕ) 
  (h1 : running_track_width = 4) 
  (h2 : garden_ring_width = 6) 
  (h3 : play_area_diameter = 14) :
  (2 * ((play_area_diameter / 2) + garden_ring_width + running_track_width)) = 34 := 
by
  sorry

end running_track_diameter_l331_331586


namespace smallest_N_sequence_has_7_numbers_l331_331506

open Nat

-- Define the sequence function
def tomSeq (n : ℕ) (count : ℕ) : ℕ :=
  if n = 0 then count
  else let k := sqrt n;
       if k % 2 = 0 then
         tomSeq (n - k^2) (count + 1)
       else
         tomSeq (n - (k - 1)^2) (count + 1)

-- Definition of N that satisfies the problem conditions
def smallestN := 168

-- The theorem stating the smallest N having exactly 7 numbers in the sequence
theorem smallest_N_sequence_has_7_numbers : tomSeq smallestN 0 = 7 := by
  sorry

end smallest_N_sequence_has_7_numbers_l331_331506


namespace mult_closest_l331_331525

theorem mult_closest :
  0.0004 * 9000000 = 3600 := sorry

end mult_closest_l331_331525


namespace packing_peanuts_per_large_order_l331_331854

/-- Definitions of conditions as stated -/
def large_orders : ℕ := 3
def small_orders : ℕ := 4
def total_peanuts_used : ℕ := 800
def peanuts_per_small : ℕ := 50

/-- The statement to prove, ensuring all conditions are utilized in the definitions -/
theorem packing_peanuts_per_large_order : 
  ∃ L, large_orders * L + small_orders * peanuts_per_small = total_peanuts_used ∧ L = 200 := 
by
  use 200
  -- Adding the necessary proof steps
  have h1 : large_orders = 3 := rfl
  have h2 : small_orders = 4 := rfl
  have h3 : peanuts_per_small = 50 := rfl
  have h4 : total_peanuts_used = 800 := rfl
  sorry

end packing_peanuts_per_large_order_l331_331854


namespace true_propositions_l331_331326

-- Definitions of the propositions
def proposition_1 : Prop :=
  ∀ (L1 L2 : Line) (P1 P2 : Plane), (L1 ∈ P1 ∧ L2 ∈ P1) ∧ (L1 ∈ P2 ∧ L2 ∈ P2) → P1 = P2

def proposition_2 : Prop := 
  ∀ (P1 P2 : Plane) (L : Line), (P1 ⊥ L ∧ P2 ⊥ L) → P1 = P2

def proposition_3 : Prop := 
  ∀ (P1 P2 : Plane) (L : Line), (P1 ⊥ P2 ∧ L ∈ P1) → L ⊥ P2

def proposition_4 : Prop := 
  ∀ (P1 P2 : Plane) (L : Line), (P1 ∥ P2 ∧ L ∈ P1) → L ∥ P2

-- The target statement
theorem true_propositions :
  (proposition_1 = false) ∧
  (proposition_2 = true) ∧
  (proposition_3 = false) ∧
  (proposition_4 = true) :=
  by
    sorry

end true_propositions_l331_331326


namespace transformed_curve_l331_331017

def curve_C (x y : ℝ) := (x - y)^2 + y^2 = 1

theorem transformed_curve (x y : ℝ) (A : Matrix (Fin 2) (Fin 2) ℝ) :
    A = ![![2, -2], ![0, 1]] →
    (∃ (x0 y0 : ℝ), curve_C x0 y0 ∧ x = 2 * x0 - 2 * y0 ∧ y = y0) →
    (∃ (x y : ℝ), (x^2 / 4 + y^2 = 1)) :=
by
  -- Proof to be completed
  sorry

end transformed_curve_l331_331017


namespace area_of_EPGQ_l331_331138

/--
Rectangle EFGH is 10 cm by 6 cm. P is the midpoint of FG, and Q is the point on 
GH such that GQ:QH = 1:3. Prove that the area of region EPGQ is 37.5 square centimeters.
-/
theorem area_of_EPGQ (EF GH P Q : ℝ→ℝ) (EFGH_area : EF * GH = 60)
  (midpoint_P : P = (F + G) / 2) (ratio_Q : GQ / QH = 1 / 3) : 
  area_of_EPGQ = 37.5 :=
sorry

end area_of_EPGQ_l331_331138


namespace max_expression_value_l331_331969

theorem max_expression_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let expr := (|4 * a - 10 * b| + |2 * (a - b * sqrt 3) - 5 * (a * sqrt 3 + b)|) / sqrt (a^2 + b^2) in
  expr ≤ 2 * sqrt 87 :=
sorry

end max_expression_value_l331_331969


namespace projectile_height_l331_331819

theorem projectile_height (t : ℝ) (h : (-16 * t^2 + 80 * t = 100)) : t = 2.5 :=
sorry

end projectile_height_l331_331819


namespace angle_bisectors_inequality_l331_331803

-- Define the points and conditions
variables {A B C D E F P1 P2 P3 : Type}

-- Assuming AD, BE, CF are angle bisectors and P1, P2, P3 are intersections as described
variable (triangle : Triangle ABC)
variable (D_on_BC : Point_on_side D B C)
variable (E_on_CA : Point_on_side E C A)
variable (F_on_AB : Point_on_side F A B)
variable (AD_bisector : IsAngleBisector A D)
variable (BE_bisector : IsAngleBisector B E)
variable (CF_bisector : IsAngleBisector C F)
variable (P1_intersection : IsIntersectionPoint P1 A D E F)
variable (P2_intersection : IsIntersectionPoint P2 B E D F)
variable (P3_intersection : IsIntersectionPoint P3 C F D E)

-- The goal to prove
theorem angle_bisectors_inequality :
  (AD_length AD / AP1_length P1) + (BE_length BE / BP2_length P2) + (CF_length CF / CP3_length P3) ≥ 6 := 
by
  sorry 

end angle_bisectors_inequality_l331_331803


namespace number_of_random_events_l331_331921

/-- 
Determine the number of random events among the given scenarios:
1. In the school's track and field sports meeting to be held next year, student Zhang San wins the 100m sprint championship.
2. During the PE class, the PE teacher randomly selects a student to fetch the sports equipment, and Li Si is selected.
3. Wang Mazhi randomly picks one out of four numbered tags marked with 1, 2, 3, 4, and gets the number 1 tag.
-/
def is_random_event_1 : Prop := true --Assuming it describes a possible future event making it random
def is_random_event_2 : Prop := true -- The selection process is random
def is_random_event_3 : Prop := true -- The picking is random

theorem number_of_random_events : (is_random_event_1 ∧ is_random_event_2 ∧ is_random_event_3) = 3 :=
by
  -- Proof is omitted.
  sorry

end number_of_random_events_l331_331921


namespace find_cost_per_kg_l331_331479

-- Define the conditions given in the problem
def side_length : ℕ := 30
def coverage_per_kg : ℕ := 20
def total_cost : ℕ := 10800

-- The cost per kg we need to find
def cost_per_kg := total_cost / ((6 * side_length^2) / coverage_per_kg)

-- We need to prove that cost_per_kg = 40
theorem find_cost_per_kg : cost_per_kg = 40 := by
  sorry

end find_cost_per_kg_l331_331479


namespace initial_machines_l331_331890

theorem initial_machines (r : ℝ) (x : ℕ) (h1 : x * 42 * r = 7 * 36 * r) : x = 6 :=
by
  sorry

end initial_machines_l331_331890


namespace tan_beta_half_l331_331640

theorem tan_beta_half (α β : ℝ)
    (h1 : Real.tan α = 1 / 3)
    (h2 : Real.sin β = 2 * Real.cos (α + β) * Real.sin α) : 
    Real.tan β = 1 / 2 := 
sorry

end tan_beta_half_l331_331640


namespace geometric_sequence_product_l331_331306

theorem geometric_sequence_product (a : ℕ → ℝ) (h1 : a 5 * a 6 * a 7 = 8) 
  (h_geom : ∀ n, a (n + 1) = a n * (a 1 / a 0)) :
  (∏ i in Finset.range 11, a i) = 2 ^ 11 :=
by
  sorry

end geometric_sequence_product_l331_331306


namespace minimum_value_l331_331328

noncomputable def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem minimum_value (a b c d : ℝ) (h1 : a < (2 / 3) * b) 
  (h2 : ∀ x, 3 * a * x^2 + 2 * b * x + c ≥ 0) : 
  ∃ (x : ℝ), ∀ c, 2 * b - 3 * a ≠ 0 → (c = (b^2 / 3 / a)) → (c / (2 * b - 3 * a) ≥ 1) :=
by
  sorry

end minimum_value_l331_331328


namespace minimum_vacation_cost_l331_331462

-- Definitions based on the conditions in the problem.
def Polina_age : ℕ := 5
def parents_age : ℕ := 30 -- arbitrary age above the threshold, assuming adults

def Globus_cost_old : ℕ := 25400
def Globus_discount : ℝ := 0.02

def AroundWorld_cost_young : ℕ := 11400
def AroundWorld_cost_old : ℕ := 23500
def AroundWorld_commission : ℝ := 0.01

def globus_total_cost (num_adults num_children : ℕ) : ℕ :=
  let initial_cost := (num_adults + num_children) * Globus_cost_old
  let discount := Globus_discount * initial_cost
  initial_cost - discount.to_nat

def around_world_total_cost (num_adults num_children : ℕ) : ℕ :=
  let initial_cost := (num_adults * AroundWorld_cost_old) + (num_children * AroundWorld_cost_young)
  let commission := AroundWorld_commission * initial_cost
  initial_cost + commission.to_nat

-- Total costs calculated for the Dorokhov family with specific parameters.
def globus_final_cost : ℕ := globus_total_cost 2 1  -- 2 adults, 1 child
def around_world_final_cost : ℕ := around_world_total_cost 2 1  -- 2 adults, 1 child

theorem minimum_vacation_cost : around_world_final_cost = 58984 := by
  sorry

end minimum_vacation_cost_l331_331462


namespace right_triangle_discriminant_right_triangle_m_l331_331027

-- Define variables and functions for the first part
variables {a b c : ℝ}
def quadratic (x : ℝ) := a * x^2 + b * x + c
def discriminant : ℝ := b^2 - 4 * a * c

theorem right_triangle_discriminant (h_triangle : ∃ A B C : ℝ, 
  quadratic A = 0 ∧ quadratic B = 0 ∧ quadratic C = a * C^2 + b * C + c ∧
  (C = (A + B) / 2) ∧
  (∃ r : ℝ, 
    (A = C + r ∧ B = C - r) ∨ (A = C - r ∧ B = C + r) ∧
    ((C - A)^2 + (0 - quadratic A)^2 = (C - B)^2 ∨ (B - A)^2 + (0 - quadratic B)^2 = (C - A)^2 ∨ 
    (B - A)^2 + (C - quadratic A)^2 = (C - quadratic B)^2))) :
  discriminant = 4 :=
sorry

-- Define variables and functions for the second part
variables {m : ℝ}
def quadratic2 (x : ℝ) := x^2 - (2 * m + 2) * x + m^2 + 5 * m + 3
def linear (x : ℝ) := 3 * x - 1
def G := (m + 1, 3 * (m + 1) - 1)

theorem right_triangle_m (h_triangle : ∃ E F G : ℝ × ℝ, 
  quadratic2 E.1 = 0 ∧ quadratic2 F.1 = 0 ∧ linear (m + 1) = quadratic2 (m + 1) ∧ G = (m + 1, 3 * (m + 1) - 1) ∧ 
  ((E.1 - F.1)^2 + (E.2 - linear E.1)^2 = (F.1 - G.1)^2 ∨ 
   (E.1 - G.1)^2 + (E.2 - linear G.1)^2 = (F.1 - G.1)^2 ∨ 
   (F.1 - G.1)^2 + (F.2 - linear G.1)^2 = (E.1 - G.1)^2)) :
  m = -1 :=
sorry

end right_triangle_discriminant_right_triangle_m_l331_331027


namespace expected_value_coins_heads_l331_331907

noncomputable def expected_value_cents : ℝ :=
  let values := [1, 5, 10, 25, 50, 100]
  let probability_heads := 1 / 2
  probability_heads * (values.sum : ℝ)

theorem expected_value_coins_heads : expected_value_cents = 95.5 := by
  sorry

end expected_value_coins_heads_l331_331907


namespace problem_power_function_l331_331157

-- Defining the conditions
variable {f : ℝ → ℝ}
variable (a : ℝ)
variable (h₁ : ∀ x, f x = x^a)
variable (h₂ : f 2 = Real.sqrt 2)

-- Stating what we need to prove
theorem problem_power_function : f 4 = 2 :=
by sorry

end problem_power_function_l331_331157


namespace find_a_l331_331327

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x + 2

theorem find_a (a : ℝ) (h_a : a ≠ 0) (h_exist : ∀ x1 ∈ Set.Icc 1 Real.exp 1, ∃ x2 ∈ Set.Icc 1 Real.exp 1, f a x1 + f a x2 = 4) :
  a = Real.exp 1 + 1 :=
sorry

end find_a_l331_331327


namespace b_minus_c_l331_331979

def a (n : ℤ) (hn : n > 1) : ℝ := 1 / (Real.log 1001 / Real.log n)

def b : ℝ := a 3 dec_trivial + a 4 dec_trivial + a 5 dec_trivial + a 6 dec_trivial

def c : ℝ := a 15 dec_trivial + a 16 dec_trivial + a 17 dec_trivial + a 18 dec_trivial + a 19 dec_trivial

theorem b_minus_c : b - c = - Real.log 3876 / Real.log 1001 := by
  sorry

end b_minus_c_l331_331979


namespace identity_x_squared_minus_y_squared_l331_331047

theorem identity_x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : 
  x^2 - y^2 = 80 := by sorry

end identity_x_squared_minus_y_squared_l331_331047


namespace expression_equality_l331_331294

theorem expression_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : (1 / x + 1 / y) = 1 :=
by
  sorry

end expression_equality_l331_331294


namespace fruits_and_vegetables_wasted_is_15_pounds_l331_331391

-- Define the given conditions
def minimum_wage : ℝ := 8
def meat_wasted : ℝ := 20
def cost_per_pound_meat : ℝ := 5
def fruits_and_vegetables_cost_per_pound : ℝ := 4
def bread_wasted : ℝ := 60
def cost_per_pound_bread : ℝ := 1.5
def janitorial_hours : ℝ := 10
def janitors_hourly_wage : ℝ := 10
def time_and_a_half : ℝ := 1.5
def hours_worked : ℝ := 50

-- Translate remaining money and cost calculations
def total_earnings := hours_worked * minimum_wage
def cost_meat := meat_wasted * cost_per_pound_meat
def cost_bread := bread_wasted * cost_per_pound_bread
def cost_janitors := janitorial_hours * (janitors_hourly_wage * time_and_a_half)
def total_cost := cost_meat + cost_bread + cost_janitors
def remaining_money := total_earnings - total_cost
def fruits_and_vegetables_wasted := remaining_money / fruits_and_vegetables_cost_per_pound

-- State the proof problem equivalent
theorem fruits_and_vegetables_wasted_is_15_pounds :
  fruits_and_vegetables_wasted = 15 := sorry

end fruits_and_vegetables_wasted_is_15_pounds_l331_331391


namespace exists_lambda_divisible_by_2011_l331_331414

theorem exists_lambda_divisible_by_2011 (a : Fin 11 → ℤ) : 
  ∃ λ : Fin 11 → ℤ, (∀ i, λ i ∈ {-1, 0, 1}) ∧ (λ ≠ 0) ∧ (∑ i, λ i * a i) % 2011 = 0 :=
by
  sorry

end exists_lambda_divisible_by_2011_l331_331414


namespace dot_product_of_foci_triangle_area_one_is_zero_l331_331314

theorem dot_product_of_foci_triangle_area_one_is_zero
  (F1 F2 P : ℝ × ℝ)
  (h_ellipse : ∃ x y : ℝ, P = (x, y) ∧ x ^ 2 / 4 + y ^ 2 = 1)
  (h_foci : F1 = (-√3, 0) ∧ F2 = (√3, 0))
  (h_area : let PF1 := (fst P - fst F1, snd P - snd F1),
                PF2 := (fst P - fst F2, snd P - snd F2) in
            (fst PF1 * snd PF2 - snd PF1 * fst PF2).abs / 2 = 1)
  : (let PF1 := (fst P - fst F1, snd P - snd F1),
         PF2 := (fst P - fst F2, snd P - snd F2) in
     PF1.1 * PF2.1 + PF1.2 * PF2.2) = 0 :=
by
  sorry

end dot_product_of_foci_triangle_area_one_is_zero_l331_331314


namespace solve_for_y_l331_331034

theorem solve_for_y (y : ℝ) (h : 25^y = 125 * 5) : y = 2 := 
by 
  have p1 : 5^2 = 25 := by ring
  have p2 : 5^3 = 125 := by norm_num
  have p3 : 25^y = (5^2)^y := by ring
  have p4 : (5^2)^y = 5^(2 * y) := by rw ←pow_mul
  rw [←p1, ←p2, p3, p4] at h
  have p5 : 125 * 5 = 5^(3 + 1) := by rw [p2, pow_add]
  rw p5 at h
  apply pow_inj (by norm_num) h
  norm_num

end solve_for_y_l331_331034


namespace remove_terms_sum_eq_two_thirds_l331_331342

theorem remove_terms_sum_eq_two_thirds : 
  (∑ x in ([1/3, 1/6, 1/9, 1/12, 1/15, 1/18] : List ℚ), x) 
  - (1/12 + 1/15) = 2/3 := by
  sorry

end remove_terms_sum_eq_two_thirds_l331_331342


namespace rationalize_denominator_l331_331765

theorem rationalize_denominator :
  ∃ A B C D : ℝ, 
    (1 / (real.cbrt 5 - real.cbrt 3) = (real.cbrt A + real.cbrt B + real.cbrt C) / D) ∧ 
    A + B + C + D = 51 :=
  sorry

end rationalize_denominator_l331_331765


namespace rationalize_denominator_l331_331776

theorem rationalize_denominator :
  let a := (∛5 : ℝ)
  let b := (∛3 : ℝ)
  let expr := 1 / (a - b)
  let num := a^2 + a*b + b^2
  ∃ A B C D : ℝ,
    expr * num = (∛A + ∛B + ∛C) / D ∧
    A = 25 ∧ B = 15 ∧ C = 9 ∧ D = 2 ∧
    A + B + C + D = 51 :=
by {
  let a := (∛5 : ℝ)
  let b := (∛3 : ℝ)
  let expr := 1 / (a - b)
  let num := a^2 + a*b + b^2

  exists (25 : ℝ)
  exists (15 : ℝ)
  exists (9 : ℝ)
  exists (2 : ℝ)

  split
  { sorry, }
  { split
    { exact rfl, }
    { split
      { exact rfl, }
      { split
        { exact rfl, }
        { split
          { exact rfl, }
          { norm_num }}}}
}

end rationalize_denominator_l331_331776


namespace final_answer_l331_331009

def f (x : ℝ) : ℝ := sorry

axiom functional_eqn : ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem final_answer : f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : ℝ, f (-x) = f x) ∧ ¬ (∃ ε > 0, ∀ h : ℝ, abs h < ε → f h ≥ f 0) := 
by
  -- omit the proof steps that were provided in the solution
  sorry

end final_answer_l331_331009


namespace jackson_initial_meat_l331_331717

-- Define the conditions given in the problem
def used_for_meatballs (M : ℝ) : ℝ := (1/4) * M
def used_for_spring_rolls : ℝ := 3
def remaining_meat : ℝ := 12

-- Define the total initial meat Jackson had
def initial_meat (M : ℝ) : Prop :=
  M - used_for_meatballs M - used_for_spring_rolls = remaining_meat

-- The statement expressing the proof problem:
theorem jackson_initial_meat : ∃ M : ℝ, initial_meat M ∧ M = 20 := 
by
  sorry

end jackson_initial_meat_l331_331717


namespace cost_of_each_toy_l331_331872

theorem cost_of_each_toy (initial_money spent_money remaining_money toys_count toy_cost : ℕ) 
  (h1 : initial_money = 57)
  (h2 : spent_money = 27)
  (h3 : remaining_money = initial_money - spent_money)
  (h4 : toys_count = 5)
  (h5 : remaining_money / toys_count = toy_cost) :
  toy_cost = 6 :=
by
  sorry

end cost_of_each_toy_l331_331872


namespace good_walker_catch_up_l331_331378

theorem good_walker_catch_up :
  ∀ x y : ℕ, 
    (x = (100:ℕ) + y) ∧ (x = ((100:ℕ)/(60:ℕ) : ℚ) * y) := 
by
  sorry

end good_walker_catch_up_l331_331378


namespace complex_number_z_l331_331016

theorem complex_number_z (z : ℂ) (h : (3 + 1 * I) * z = 4 - 2 * I) : z = 1 - I :=
by
  sorry

end complex_number_z_l331_331016


namespace max_consecutive_sum_terms_l331_331163

theorem max_consecutive_sum_terms (S : ℤ) (n : ℕ) (H1 : S = 2015) (H2 : 0 < n) :
  (∃ a : ℤ, S = (a * n + (n * (n - 1)) / 2)) → n = 4030 :=
sorry

end max_consecutive_sum_terms_l331_331163


namespace correct_population_statement_l331_331187

def correct_statement :=
  "The mathematics scores of all candidates in the city's high school entrance examination last year constitute the population."

def sample_size : ℕ := 500

def is_correct (statement : String) : Prop :=
  statement = correct_statement

theorem correct_population_statement (scores : Fin 500 → ℝ) :
  is_correct "The mathematics scores of all candidates in the city's high school entrance examination last year constitute the population." :=
by
  sorry

end correct_population_statement_l331_331187


namespace identity_x_squared_minus_y_squared_l331_331045

theorem identity_x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : 
  x^2 - y^2 = 80 := by sorry

end identity_x_squared_minus_y_squared_l331_331045


namespace max_cars_with_ac_but_not_racing_stripes_l331_331369

def total_cars : Nat := 200
def cars_without_ac : Nat := 85
def cars_with_racing_stripes : Nat := 110

theorem max_cars_with_ac_but_not_racing_stripes : 
  ∀ (total_cars cars_without_ac cars_with_racing_stripes : Nat), 
  total_cars = 200 → 
  cars_without_ac = 85 → 
  cars_with_racing_stripes ≥ 110 → 
  115 - cars_with_racing_stripes = 5 := 
by {
  intros,
  sorry
}

end max_cars_with_ac_but_not_racing_stripes_l331_331369


namespace pure_imaginary_ratio_l331_331110

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : (∀ (z : ℂ), z = (3 - 5 * complex.I) * (a + b * complex.I) → z.re = 0)) : 
  a / b = -5 / 3 :=
sorry

end pure_imaginary_ratio_l331_331110


namespace bounces_to_below_30_cm_l331_331536

theorem bounces_to_below_30_cm :
  ∃ (b : ℕ), (256 * (3 / 4)^b < 30) ∧
            (∀ (k : ℕ), k < b -> 256 * (3 / 4)^k ≥ 30) :=
by 
  sorry

end bounces_to_below_30_cm_l331_331536


namespace exists_real_q_l331_331293

noncomputable def f (n : ℕ) : ℝ :=
  real.min (λ m : ℕ+, real.abs (real.sqrt 2 - (m : ℝ) / n))

theorem exists_real_q (seq : ℕ → ℕ) (h_inc : strict_mono seq) (λ : ℝ) (h_f : ∀ i, f (seq i) < λ / (seq i)^2) :
  ∃ q > 1, ∀ i, seq i ≥ q ^ (i - 1) :=
sorry

end exists_real_q_l331_331293


namespace marilyn_bottle_caps_l331_331116

theorem marilyn_bottle_caps : ∀ (caps_start caps_given : ℕ), caps_start = 51 → caps_given = 36 → (caps_start - caps_given) = 15 :=
by
  intros caps_start caps_given h1 h2
  rw [h1, h2]
  simp
  exact sorry

end marilyn_bottle_caps_l331_331116


namespace proof_problem_l331_331343

variables (p q r s u v : Prop)

-- Conditions
axiom h1 : p > q → r > s
axiom h2 : r = s → u < v
axiom h3 : p = q → s > r

-- The statement to prove.
theorem proof_problem : p ≠ q → s ≠ r := by
  intro hpq
  by_cases hpq_case : p = q
  -- Case: p = q
  have hsgr : s > r := h3 hpq_case
  intro hsr
  contradiction
  -- Case: p ≠ q
  intro hsr
  cases lt_or_gt_of_ne hsr with hrs hsr_ls
  contradiction
  contradiction
  sorry

end proof_problem_l331_331343


namespace general_formula_an_bound_Tn_l331_331992

variable (S : ℕ → ℕ) (T : ℕ → ℝ)

axiom S_5_eq_70 : S 5 = 70
axiom a2_a7_a22_geo_seq : (2*d + a1) * (21*d + a1) = (6*d + a1)^2
axiom d_nonzero : d ≠ 0

theorem general_formula_an (n : ℕ) : ∃ a1 d, ∀ n, S n = (n / 2) * (2*a1 + (n - 1)*d) → a1 = 6 ∨ a1 = 14 :=
sorry

theorem bound_Tn (n : ℕ) : ∃ a1 d, (a1 = 6 ∨ a1 = 14) → T n = (Σ i in range n, 1 / ((i + 1) * (a1 + i * d))) → (1/6 : ℝ) ≤ T n ∧ T n < (3/8 : ℝ) :=
sorry

end general_formula_an_bound_Tn_l331_331992


namespace measure_angle_C_l331_331365

-- Given: a^2 + b^2 - ab = c^2 in a triangle ABC
variable {a b c : ℝ} (h : a^2 + b^2 - ab = c^2)

-- Prove: ∠C = π / 3
theorem measure_angle_C (a b c : ℝ) (h : a^2 + b^2 - ab = c^2) : 
  ∠C = π / 3 :=
by
  -- Sorry is used to skip the proof steps
  sorry

end measure_angle_C_l331_331365


namespace common_point_of_symmetric_spheres_l331_331917

variables {R : ℝ} {O O_a O_b O_c O' P A B C S : Point}
noncomputable def symmetric_sphere_center (center : Point) (axis : Line) : Point :=
sorry

noncomputable def symmetric_sphere_center_plane (center : Point) (plane : Plane) : Point :=
sorry

theorem common_point_of_symmetric_spheres
  (SABC : Tetrahedron)
  (center : Point)
  (radius : ℝ)
  (symm_center_SA : symmetric_sphere_center center (Line.mk S A) = O_a)
  (symm_center_SB : symmetric_sphere_center center (Line.mk S B) = O_b)
  (symm_center_SC : symmetric_sphere_center center (Line.mk S C) = O_c)
  (symm_center_plane_ABC : symmetric_sphere_center_plane center (Plane.mk A B C) = O')
  (radius_eq : radius ≠ 0) :
  ∃ P : Point,
  (dist P O').v = radius ∧ (dist P O_a).v = radius ∧ 
  (dist P O_b).v = radius ∧ (dist P O_c).v = radius :=
begin
  sorry
end

end common_point_of_symmetric_spheres_l331_331917


namespace P_plus_Q_l331_331734

theorem P_plus_Q (P Q : ℝ) (h : (P / (x - 3) + Q * (x - 2)) = (-5 * x^2 + 18 * x + 27) / (x - 3)) : P + Q = 31 := 
by {
  sorry
}

end P_plus_Q_l331_331734


namespace least_n_difference_9_l331_331610

open Finset

theorem least_n_difference_9 (n : ℕ) (h : ∀ (A : Finset ℕ), A.card = n → (∃ a b ∈ A, a - b = 9 ∨ b - a = 9)) :
  n = 51 := 
begin
  sorry
end

end least_n_difference_9_l331_331610


namespace number_of_rows_in_theater_l331_331357

theorem number_of_rows_in_theater 
  (x : ℕ)
  (h1 : ∀ (students : ℕ), students = 30 → ∃ row : ℕ, row < x ∧ ∃ a b : ℕ, a ≠ b ∧ row = a ∧ row = b)
  (h2 : ∀ (students : ℕ), students = 26 → ∃ empties : ℕ, empties ≥ 3 ∧ x - students = empties)
  : x = 29 :=
by
  sorry

end number_of_rows_in_theater_l331_331357


namespace profit_percent_is_correct_l331_331230

theorem profit_percent_is_correct : 
  ∀ (marked_price cost_price selling_price : ℚ) 
  (num_pens : ℕ),
  num_pens = 120 →
  marked_price = 1 →
  cost_price = (100 : ℚ) / num_pens →
  selling_price = 95 / 100 * marked_price →
  (selling_price - cost_price) / cost_price * 100 = 70 := by
  intros marked_price cost_price selling_price num_pens 
  introduce h_num_pens h_marked_price h_cost_price h_selling_price
  sorry

end profit_percent_is_correct_l331_331230


namespace find_angle_A_l331_331066

theorem find_angle_A (A B C a b c : ℝ) 
  (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : a > 0)
  (h5 : b > 0)
  (h6 : c > 0)
  (sin_eq : Real.sin (C + π / 6) = b / (2 * a)) :
  A = π / 6 :=
sorry

end find_angle_A_l331_331066


namespace toy_factory_max_profit_l331_331565

theorem toy_factory_max_profit :
  ∃ x y : ℕ,    -- x: number of bears, y: number of cats
  15 * x + 10 * y ≤ 450 ∧    -- labor hours constraint
  20 * x + 5 * y ≤ 400 ∧     -- raw materials constraint
  80 * x + 45 * y = 2200 :=  -- total selling price
by
  sorry

end toy_factory_max_profit_l331_331565


namespace find_num_students_first_class_l331_331604

noncomputable def num_students_first_class (avg1 avg2 combined_avg : ℝ) (students_second_class : ℕ) :=
  let num := (avg2 - combined_avg) * (students_second_class : ℝ) / (combined_avg - avg1) in
  Int.ofNat ⟨num, by norm_num ; exact_mod_cast num⟩.natAbs

-- Define the conditions
constant (avg1 avg2 combined_avg : ℝ)
constant (students_second_class : ℕ)
constant (expected_num_students_first_class : ℤ)

-- Assign given values
def avg1_val : ℝ := 67
def avg2_val : ℝ := 82
def combined_avg_val : ℝ := 74.0909090909091
def students_second_class_val : ℕ := 52
def expected_result : ℤ := 58

-- Prove the statement
theorem find_num_students_first_class :
  num_students_first_class avg1_val avg2_val combined_avg_val students_second_class_val = expected_result :=
sorry

end find_num_students_first_class_l331_331604


namespace chicken_nuggets_order_l331_331438

theorem chicken_nuggets_order (cost_per_box : ℕ) (nuggets_per_box : ℕ) (total_amount_paid : ℕ) 
  (h1 : cost_per_box = 4) (h2 : nuggets_per_box = 20) (h3 : total_amount_paid = 20) : 
  total_amount_paid / cost_per_box * nuggets_per_box = 100 :=
by
  -- This is where the proof would go
  sorry

end chicken_nuggets_order_l331_331438


namespace sum_b_geq_4_div_n_plus_1_l331_331415

theorem sum_b_geq_4_div_n_plus_1
  (n : ℕ)
  (b : Fin n → ℝ)
  (b_pos : ∀ k, 0 < b k)
  (x : Fin n → ℝ)
  (x_nontrivial : ∃ k, x k ≠ 0)
  (h_eq : ∀ k, (k : ℕ) < n → 
    (if k = 0 then 0 else x ⟨k-1, by linarith⟩) 
    - 2 * x ⟨k, by linarith⟩ 
    + (if k = n-1 then 0 else x ⟨k+1, by linarith⟩) 
    + b k * x ⟨k, by linarith⟩ = 0)
  (h_zero : (x 0 = 0) ∧ (x ⟨n, by linarith⟩ = 0)) :
  ∑ k : Fin n, b k ≥ 4 / (n + 1) := 
sorry

end sum_b_geq_4_div_n_plus_1_l331_331415


namespace area_inside_T_l331_331420

noncomputable def five_presentable (z : ℂ) : Prop :=
∃ (w : ℂ), |w| = 5 ∧ z = w - 1 / w

noncomputable def T : set ℂ := {z : ℂ | five_presentable z}

theorem area_inside_T : ∃ (A : ℝ), A = (624 / 25) * real.pi := 
begin
  use (624 / 25) * real.pi,
  sorry
end

end area_inside_T_l331_331420


namespace mary_final_weight_l331_331750

theorem mary_final_weight :
    let initial_weight := 99
    let initial_loss := 12
    let first_gain := 2 * initial_loss
    let second_loss := 3 * initial_loss
    let final_gain := 6
    let weight_after_first_loss := initial_weight - initial_loss
    let weight_after_first_gain := weight_after_first_loss + first_gain
    let weight_after_second_loss := weight_after_first_gain - second_loss
    let final_weight := weight_after_second_loss + final_gain
    in final_weight = 81 :=
by
    sorry

end mary_final_weight_l331_331750


namespace minimum_cost_for_Dorokhov_family_vacation_l331_331465

theorem minimum_cost_for_Dorokhov_family_vacation :
  let globus_cost := (25400 * 3) * 0.98,
      around_world_cost := (11400 + (23500 * 2)) * 1.01
  in
  min globus_cost around_world_cost = 58984 := by
  let globus_cost := (25400 * 3) * 0.98
  let around_world_cost := (11400 + (23500 * 2)) * 1.01
  sorry

end minimum_cost_for_Dorokhov_family_vacation_l331_331465


namespace pizza_cost_per_slice_correct_l331_331095

noncomputable def pizza_cost_per_slice : ℝ :=
  let base_pizza_cost := 10.00
  let first_topping_cost := 2.00
  let next_two_toppings_cost := 2.00
  let remaining_toppings_cost := 2.00
  let total_cost := base_pizza_cost + first_topping_cost + next_two_toppings_cost + remaining_toppings_cost
  total_cost / 8

theorem pizza_cost_per_slice_correct :
  pizza_cost_per_slice = 2.00 :=
by
  unfold pizza_cost_per_slice
  sorry

end pizza_cost_per_slice_correct_l331_331095


namespace binet_formula_variant_l331_331137

/-- Binet's formula for Fibonacci numbers expressed using series. -/
theorem binet_formula_variant (n : ℕ) (F : ℕ → ℕ) [∀ n, F (n) = (Nat.fib n : ℕ)] :
  2^(n-1) * F(n) = ∑ k in Finset.range ((n-1) / 2 + 1), (Nat.choose n (2*k + 1)) * 5^k := 
sorry

end binet_formula_variant_l331_331137


namespace age_difference_is_16_l331_331475

variable (y : ℕ)  -- the present age of the younger person

-- Conditions
def elder_age_now : ℕ := 30
def elder_age_6_years_ago := elder_age_now - 6
def younger_age_6_years_ago := y - 6
def condition := elder_age_6_years_ago = 3 * younger_age_6_years_ago

-- Theorem to prove the difference in ages is 16 years
theorem age_difference_is_16 (h : condition y) : elder_age_now - y = 16 :=
by
  sorry

end age_difference_is_16_l331_331475


namespace area_of_original_triangle_l331_331558

variable (H : ℝ) (H' : ℝ := 0.65 * H) 
variable (A' : ℝ := 14.365)
variable (k : ℝ := 0.65) 
variable (A : ℝ)

theorem area_of_original_triangle (h₁ : H' = k * H) (h₂ : A' = 14.365) (h₃ : k = 0.65) : A = 34 := by
  sorry

end area_of_original_triangle_l331_331558


namespace sum_of_rel_prime_factors_eq_l331_331296

theorem sum_of_rel_prime_factors_eq :
  ∃ (a b c d : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 ∧ a * b * c * d = 14400 ∧ 
  Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd a d = 1 ∧ 
  Nat.gcd b c = 1 ∧ Nat.gcd b d = 1 ∧ Nat.gcd c d = 1 ∧ 
  a + b + c = 98 :=
by decide -- checking logical consistency only, proof not provided.

end sum_of_rel_prime_factors_eq_l331_331296


namespace parabola_directrix_eq_l331_331338

theorem parabola_directrix_eq (a : ℝ) (h : - a / 4 = - (1 : ℝ) / 4) : a = 1 := by
  sorry

end parabola_directrix_eq_l331_331338


namespace ellipse_focus_value_l331_331063

theorem ellipse_focus_value (m : ℝ) (h1 : m > 0) :
  (∃ (x y : ℝ), (x, y) = (-4, 0) ∧ (25 - m^2 = 16)) → m = 3 :=
by
  sorry

end ellipse_focus_value_l331_331063


namespace there_exists_triangle_l331_331373

def unit_square : set (ℝ × ℝ) := { p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 }

noncomputable def points : set (ℝ × ℝ) := sorry -- This represents the 101 points condition

axiom points_101 : points.finite ∧ points.card = 101
axiom no_three_collinear {p1 p2 p3 : ℝ × ℝ} (hp1 : p1 ∈ points) (hp2 : p2 ∈ points) (hp3 : p3 ∈ points) :
  ¬ collinear ({p1, p2, p3} : set (ℝ × ℝ))

theorem there_exists_triangle :
  ∃ (p1 p2 p3 : ℝ × ℝ), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
                          ¬collinear ({p1, p2, p3} : set (ℝ × ℝ)) ∧
                          triangle_area p1 p2 p3 ≤ 0.01 := sorry

end there_exists_triangle_l331_331373


namespace inverse_proportion_l331_331801

variable {x y x1 x2 y1 y2 : ℝ}
variable {k : ℝ}

theorem inverse_proportion {h1 : x1 ≠ 0} {h2 : x2 ≠ 0} {h3 : y1 ≠ 0} {h4 : y2 ≠ 0}
  (h5 : (∃ k, ∀ (x y : ℝ), x * y = k))
  (h6 : x1 / x2 = 4 / 5) : 
  y1 / y2 = 5 / 4 :=
sorry

end inverse_proportion_l331_331801


namespace intersection_and_area_l331_331022

theorem intersection_and_area (A B : ℝ × ℝ) (x y : ℝ):
  (x - 2 * y - 5 = 0) → (x ^ 2 + y ^ 2 = 50) →
  (A = (-5, -5) ∨ A = (7, 1)) → (B = (-5, -5) ∨ B = (7, 1)) →
  (A ≠ B) →
  ∃ (area : ℝ), area = 15 :=
by
  sorry

end intersection_and_area_l331_331022


namespace projectile_height_l331_331818

theorem projectile_height (t : ℝ) (h : (-16 * t^2 + 80 * t = 100)) : t = 2.5 :=
sorry

end projectile_height_l331_331818


namespace expected_value_of_area_std_dev_of_area_l331_331250

-- Define the variables and constants given in the conditions
def width_mean : ℝ := 2 -- meters
def length_mean : ℝ := 1 -- meter
def width_std_dev : ℝ := 0.003 -- meters
def length_std_dev : ℝ := 0.002 -- meters

-- Define the expectation and variance for the random variables X and Y
def E_X : ℝ := width_mean
def E_Y : ℝ := length_mean
def Var_X : ℝ := width_std_dev^2
def Var_Y : ℝ := length_std_dev^2

-- The expected value of the area of the rectangle
theorem expected_value_of_area : E_X * E_Y = 2 :=
by sorry

-- The standard deviation of the area of the rectangle in square centimeters
theorem std_dev_of_area : sqrt (Var_X * E_Y^2 + Var_Y * E_X^2 + Var_X * Var_Y) * 10000 = 50 :=
by sorry

end expected_value_of_area_std_dev_of_area_l331_331250


namespace black_white_ratio_l331_331975

theorem black_white_ratio :
  ∀ (r1 r2 r3 r4 r5 : ℝ), 
    r1 = 2 → r2 = 4 → r3 = 6 → r4 = 8 → r5 = 10 →
    let A1 := π * r1^2 in
    let A2 := π * r2^2 in
    let A3 := π * r3^2 in
    let A4 := π * r4^2 in
    let A5 := π * r5^2 in
    let white1 := A1 in
    let black1 := A2 - A1 in
    let white2 := A3 - A2 in
    let black2 := A4 - A3 in
    let white3 := A5 - A4 in
    let total_black := black1 + black2 in
    let total_white := white1 + white2 + white3 in
    total_black / total_white = 2 / 3 :=
by
  intros r1 r2 r3 r4 r5 hr1 hr2 hr3 hr4 hr5
  let A1 := π * r1^2
  let A2 := π * r2^2
  let A3 := π * r3^2
  let A4 := π * r4^2
  let A5 := π * r5^2
  let white1 := A1
  let black1 := A2 - A1
  let white2 := A3 - A2
  let black2 := A4 - A3
  let white3 := A5 - A4
  let total_black := black1 + black2
  let total_white := white1 + white2 + white3
  sorry -- Proof goes here

end black_white_ratio_l331_331975


namespace no_root_in_interval_l331_331952

open Set

def f (x : ℝ) : ℝ := x^5 - 3*x - 1

theorem no_root_in_interval : ¬∃ x ∈ Ioo 2 3, f x = 0 :=
begin
  sorry
end

end no_root_in_interval_l331_331952


namespace slope_of_line_l331_331692

theorem slope_of_line (a : ℝ) (h_eq_intercepts : (a - 2) / a = a - 2) : (a = 2 ∨ a = 1) → ((a = 2 → slope l = -1) ∧ (a = 1 → slope l = -2)) :=
begin
  sorry
end

end slope_of_line_l331_331692


namespace lucky_83rd_number_l331_331065

def is_lucky_number (a : ℕ) : Prop :=
  (a.digits 10).sum = 8

def lucky_numbers : List ℕ :=
  List.filter is_lucky_number (List.range 2016)

noncomputable def a (n : ℕ) : ℕ :=
  (lucky_numbers.sort (· ≤ ·)).nth (n - 1)

theorem lucky_83rd_number : a 83 = 2015 :=
  sorry

end lucky_83rd_number_l331_331065


namespace smaller_integer_l331_331477

noncomputable def m : ℕ := 1
noncomputable def n : ℕ := 1998 * m

lemma two_digit_number (m: ℕ) : 10 ≤ m ∧ m < 100 := by sorry
lemma three_digit_number (n: ℕ) : 100 ≤ n ∧ n < 1000 := by sorry

theorem smaller_integer 
  (two_digit_m: 10 ≤ m ∧ m < 100)
  (three_digit_n: 100 ≤ n ∧ n < 1000)
  (avg_eq_decimal: (m + n) / 2 = m + n / 1000)
  : m = 1 := by 
  sorry

end smaller_integer_l331_331477


namespace num_ways_divide_points_l331_331312

theorem num_ways_divide_points (n : ℕ) (h : n > 2) (no_three_collinear : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ≠ p2 → p2 ≠ p3 → p3 ≠ p1 → ¬ collinear ({p1, p2, p3} : set (ℝ × ℝ))) : 
  ∃ k, k = n * (n - 1) / 2 :=
by
  sorry

end num_ways_divide_points_l331_331312


namespace solve_for_y_l331_331791

theorem solve_for_y (y : ℝ) :
  (1 / 8) ^ (3 * y + 12) = (32) ^ (3 * y + 7) → y = -71 / 24 :=
by
  intro h
  sorry

end solve_for_y_l331_331791


namespace election_invalid_votes_percentage_l331_331079

theorem election_invalid_votes_percentage (x : ℝ) :
  (∀ (total_votes valid_votes_in_favor_of_A : ℝ),
    total_votes = 560000 →
    valid_votes_in_favor_of_A = 357000 →
    0.75 * ((1 - x / 100) * total_votes) = valid_votes_in_favor_of_A) →
  x = 15 :=
by
  intro h
  specialize h 560000 357000 (rfl : 560000 = 560000) (rfl : 357000 = 357000)
  sorry

end election_invalid_votes_percentage_l331_331079


namespace combined_area_of_walls_l331_331504

theorem combined_area_of_walls (A : ℕ) 
  (h1: ∃ (A : ℕ), A ≥ 0)
  (h2 : (A - 2 * 40 - 40 = 180)) :
  A = 300 := 
sorry

end combined_area_of_walls_l331_331504


namespace general_term_arithmetic_sequence_l331_331015

theorem general_term_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (a1 : a 1 = -1) 
  (d : ℤ) 
  (h : d = 4) : 
  ∀ n : ℕ, a n = 4 * n - 5 :=
by
  sorry

end general_term_arithmetic_sequence_l331_331015


namespace root_division_pow_l331_331484

theorem root_division_pow : (7 ^ (1 / 4)) / (7 ^ (1 / 3)) = 7 ^ (-1 / 12) :=
by 
  sorry

end root_division_pow_l331_331484


namespace average_speed_over_five_hours_l331_331174

theorem average_speed_over_five_hours :
  let speed1 := 50
  let speed2 := 60
  let speed3 := 55
  let speed4 := 70
  let speed5 := 65
  let total_distance := speed1 + speed2 + speed3 + speed4 + speed5
  let total_time := 5
  (total_distance / total_time) = 60 := by
  define speed1 := 50
  define speed2 := 60
  define speed3 := 55
  define speed4 := 70
  define speed5 := 65
  define total_distance := speed1 + speed2 + speed3 + speed4 + speed5
  define total_time := 5
  show (total_distance / total_time) = 60
  sorry

end average_speed_over_five_hours_l331_331174


namespace sum_distances_from_A_to_midpoints_l331_331946

open Real

def point := (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  √((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def midpoint (p1 p2 : point) : point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def A : point := (0, 0)
def B : point := (3, 0)
def C : point := (3, 4)
def D : point := (0, 4)

def M : point := midpoint A B
def N : point := midpoint B C
def O : point := midpoint C D
def P : point := midpoint D A

def AM : ℝ := distance A M
def AN : ℝ := distance A N
def AO : ℝ := distance A O
def AP : ℝ := distance A P

theorem sum_distances_from_A_to_midpoints
  (M == midpoint A B)
  (N == midpoint B C)
  (O == midpoint C D)
  (P == midpoint D A)
  (AM == distance A M)
  (AN == distance A N)
  (AO == distance A O)
  (AP == distance A P) :
  AM + AN + AO + AP = 3.5 + √13 + √18.25 := by
  -- proof steps here, if we were to write the proof
  sorry

end sum_distances_from_A_to_midpoints_l331_331946


namespace sum_of_squares_of_projections_constant_l331_331447

-- Defines a function that calculates the sum of the squares of the projections of the edges of a cube onto any plane.
def sum_of_squares_of_projections (a : ℝ) (n : ℝ × ℝ × ℝ) : ℝ :=
  let α := n.1
  let β := n.2.1
  let γ := n.2.2
  4 * (a^2) * (2)

-- Define the theorem statement that proves the sum of the squares of the projections is constant and equal to 8a^2
theorem sum_of_squares_of_projections_constant (a : ℝ) (n : ℝ × ℝ × ℝ) :
  sum_of_squares_of_projections a n = 8 * a^2 :=
by
  -- Since we assume the trigonometric identity holds, directly match the sum_of_squares_of_projections function result.
  sorry

end sum_of_squares_of_projections_constant_l331_331447


namespace solve_for_x_l331_331784

theorem solve_for_x : ∀ (x : ℤ), (5 * x - 2) * 4 = (3 * (6 * x - 6)) → x = -5 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l331_331784


namespace final_answer_l331_331010

def f (x : ℝ) : ℝ := sorry

axiom functional_eqn : ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem final_answer : f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : ℝ, f (-x) = f x) ∧ ¬ (∃ ε > 0, ∀ h : ℝ, abs h < ε → f h ≥ f 0) := 
by
  -- omit the proof steps that were provided in the solution
  sorry

end final_answer_l331_331010


namespace krish_spent_on_sweets_l331_331437

noncomputable def initial_amount := 200.50
noncomputable def amount_per_friend := 25.20
noncomputable def remaining_amount := 114.85

noncomputable def total_given_to_friends := amount_per_friend * 2
noncomputable def amount_before_sweets := initial_amount - total_given_to_friends
noncomputable def amount_spent_on_sweets := amount_before_sweets - remaining_amount

theorem krish_spent_on_sweets : amount_spent_on_sweets = 35.25 :=
by
  sorry

end krish_spent_on_sweets_l331_331437


namespace height_of_bottom_step_l331_331853

variable (h l w : ℝ)

theorem height_of_bottom_step
  (h l w : ℝ)
  (eq1 : l + h - w / 2 = 42)
  (eq2 : 2 * l + h = 38)
  (w_value : w = 4) : h = 34 := by
sorry

end height_of_bottom_step_l331_331853


namespace parabola_hyperbola_line_l331_331337

theorem parabola_hyperbola_line
    (p : ℝ) (hp : 0 < p) :
    let F := (p / 2, 0)
    let C := { (x, y) : ℝ × ℝ | y^2 = 2 * p * x }
    let H := { (x, y) : ℝ × ℝ | x^2 / 3 - y^2 = 1 }
    let d := dist (p / 2, 0) (x, sqrt 3 / 3 * x) = 1
    (k : ℝ) (A B : ℝ × ℝ) :
    lineThrough (p / 2, 0) k ∩ C = {A, B} →
    vector_notation (from p / 2 0 A) = 2 • (from A B) →
    k = 2 * sqrt 2 ∨ k = -2 * sqrt 2 :=
sorry

end parabola_hyperbola_line_l331_331337


namespace trains_meet_at_distance_360_km_l331_331566

-- Define the speeds of the trains
def speed_A : ℕ := 30 -- speed of train A in kmph
def speed_B : ℕ := 40 -- speed of train B in kmph
def speed_C : ℕ := 60 -- speed of train C in kmph

-- Define the head starts in hours for trains A and B
def head_start_A : ℕ := 9 -- head start for train A in hours
def head_start_B : ℕ := 3 -- head start for train B in hours

-- Define the distances traveled by trains A and B by the time train C starts at 6 p.m.
def distance_A_start : ℕ := speed_A * head_start_A -- distance traveled by train A by 6 p.m.
def distance_B_start : ℕ := speed_B * head_start_B -- distance traveled by train B by 6 p.m.

-- The formula to calculate the distance after t hours from 6 p.m. for each train
def distance_A (t : ℕ) : ℕ := distance_A_start + speed_A * t
def distance_B (t : ℕ) : ℕ := distance_B_start + speed_B * t
def distance_C (t : ℕ) : ℕ := speed_C * t

-- Problem statement to prove the point where all three trains meet
theorem trains_meet_at_distance_360_km : ∃ t : ℕ, distance_A t = 360 ∧ distance_B t = 360 ∧ distance_C t = 360 := by
  sorry

end trains_meet_at_distance_360_km_l331_331566


namespace odd_function_expression_l331_331318

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then -x * log (2 - x) / log 2 else -x * log (2 + x) / log 2

theorem odd_function_expression (x : ℝ) :
  (∀ x : ℝ, f (-x) = -f x) → (∀ x : ℝ, f x = if x < 0 then -x * log (2 - x) / log 2 else -x * log (2 + x) / log 2) :=
begin
  intro h_odd,
  funext,
  unfold f,
  split_ifs,
  { apply h_odd },
  { apply h_odd }
end

end odd_function_expression_l331_331318


namespace f_odd_and_increasing_l331_331659

noncomputable def f (x : ℝ) : ℝ := 3^x - (1/3)^x

theorem f_odd_and_increasing : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) := sorry

end f_odd_and_increasing_l331_331659


namespace correct_conclusions_count_l331_331482

theorem correct_conclusions_count : 
  let f : ℝ → ℝ := λ x, 2^(2*x) - 2^(x+1) + 2 in
  let M : Set ℝ := {x | x ≤ 1} in
  let P : Set ℝ := Icc 1 2 in
  (∀ x, f x ∈ P) → 
  (M = {x | x ≤ 1}) ∧ 
  ([0, 1] ⊆ M) ∧ 
  (M ⊆ {x | x ≤ 1}) ∧ 
  (1 ∈ M) ∧ 
  (-1 ∈ M) → 
  4 :=
by
  sorry

end correct_conclusions_count_l331_331482


namespace financial_outcome_l331_331440

-- Define the conditions given in the problem
def selling_price := 2.40
def profit_percentage_first := 30 / 100
def loss_percentage_second := 10 / 100

-- Define the cost prices C1 and C2
def cost_price_first := selling_price / (1 + profit_percentage_first)
def cost_price_second := selling_price / (1 - loss_percentage_second)

-- Define the total cost and total revenue
def total_cost := cost_price_first + cost_price_second
def total_revenue := 2 * selling_price

-- Define the net result
def net_result := total_revenue - total_cost

-- The theorem we want to prove
theorem financial_outcome : net_result = 0.29 :=
by
  -- Here we assume the proof steps would fill in to show that net_result equals 0.29
  sorry

end financial_outcome_l331_331440


namespace probabilityOfWearingSunglassesGivenCap_l331_331756

-- Define the conditions as Lean constants
def peopleWearingSunglasses : ℕ := 80
def peopleWearingCaps : ℕ := 60
def probabilityOfWearingCapGivenSunglasses : ℚ := 3 / 8
def peopleWearingBoth : ℕ := (3 / 8) * 80

-- Prove the desired probability
theorem probabilityOfWearingSunglassesGivenCap : (peopleWearingBoth / peopleWearingCaps = 1 / 2) :=
by
  -- sorry is used here to skip the proof
  sorry

end probabilityOfWearingSunglassesGivenCap_l331_331756


namespace mary_final_weight_l331_331747

def initial_weight : Int := 99
def weight_lost_initially : Int := 12
def weight_gained_back_twice_initial : Int := 2 * weight_lost_initially
def weight_lost_thrice_initial : Int := 3 * weight_lost_initially
def weight_gained_back_half_dozen : Int := 12 / 2

theorem mary_final_weight :
  let final_weight := 
      initial_weight 
      - weight_lost_initially 
      + weight_gained_back_twice_initial 
      - weight_lost_thrice_initial 
      + weight_gained_back_half_dozen
  in final_weight = 78 :=
by
  sorry

end mary_final_weight_l331_331747


namespace smallest_positive_angle_l331_331261

theorem smallest_positive_angle :
  ∃ y : ℝ, 0 < y ∧ y < 90 ∧ (6 * Real.sin y * (Real.cos y)^3 - 6 * (Real.sin y)^3 * Real.cos y = 3 / 2) ∧ y = 22.5 :=
by
  sorry

end smallest_positive_angle_l331_331261


namespace syntheticMethod_correct_l331_331497

-- Definition: The synthetic method leads from cause to effect.
def syntheticMethod (s : String) : Prop :=
  s = "The synthetic method leads from cause to effect, gradually searching for the necessary conditions that are known."

-- Question: Is the statement correct?
def question : String :=
  "The thought process of the synthetic method is to lead from cause to effect, gradually searching for the necessary conditions that are known."

-- Options given
def options : List String := ["Correct", "Incorrect", "", ""]

-- Correct answer is Option A - "Correct"
def correctAnswer : String := "Correct"

theorem syntheticMethod_correct :
  syntheticMethod question → options.head? = some correctAnswer :=
sorry

end syntheticMethod_correct_l331_331497


namespace sequence_sum_l331_331236

theorem sequence_sum :
  (∀ n : ℕ, (n ≥ 3) → a n = - a (n - 1) + 6 * a (n - 2)) →
  a 1 = 2 →
  a 2 = 1 →
  a 100 + 3 * a 99 = 7 * 2^98 :=
by
  sorry

end sequence_sum_l331_331236


namespace smallest_circles_covering_rectangle_l331_331524
-- Import necessary libraries

-- Define the problem
theorem smallest_circles_covering_rectangle :
  ∀ (r : ℝ) (w h n : ℕ),  -- Variables for radius, width, height, and number of circles
  r = real.sqrt 2 →        -- Radius condition
  w = 6 →                  -- Width of rectangle
  h = 3 →                  -- Height of rectangle
  (w * h ≤ n * real.pi * r ^ 2) →    -- Coverage condition for the circles
  n = 6 :=                 -- Number of circles needed
by
  intros r w h n hr hw hh hn,
  sorry 

end smallest_circles_covering_rectangle_l331_331524


namespace probability_seat_7_l331_331758

open ProbabilityTheory

noncomputable def probability_last_passenger_seat (n : ℕ) : ℝ :=
if h: n = 1 then 1 else 1 / (n:ℝ)

theorem probability_seat_7 : probability_last_passenger_seat 7 = 1 / 7 := by
  sorry

end probability_seat_7_l331_331758


namespace domain_of_f_is_open_interval_7_infty_l331_331951

noncomputable def f (x : ℝ) : ℝ := (2 * x - 3) / (Real.sqrt (x - 7))

def domain (x : ℝ) : Prop := x > 7

theorem domain_of_f_is_open_interval_7_infty : 
  ∀ x : ℝ, f x ∉ ℝ → (domain x) :=
by
  sorry

end domain_of_f_is_open_interval_7_infty_l331_331951


namespace standard_equation_of_hyperbola_l331_331674

def P1 : ℝ × ℝ := (3, -4 * Real.sqrt 2)
def P2 : ℝ × ℝ := (9 / 4, 5)

def hyperbola_standard_form {a b : ℝ} (a_pos : 0 < a) (b_pos : 0 < b) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

theorem standard_equation_of_hyperbola :
  ∃ (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b),
    (hyperbola_standard_form a_pos b_pos P1.1 P1.2) ∧ 
    (hyperbola_standard_form a_pos b_pos P2.1 P2.2) ∧ 
    (∀ x y, hyperbola_standard_form a_pos b_pos x y ↔ 
      (49 * x^2) / 113 - (7 * y^2) / 113 = 1) :=
by
  sorry

end standard_equation_of_hyperbola_l331_331674


namespace intersection_complement_eq_three_l331_331740

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem intersection_complement_eq_three : N ∩ (U \ M) = {3} := by
  sorry

end intersection_complement_eq_three_l331_331740


namespace total_balloons_l331_331298

theorem total_balloons (fred_balloons sam_balloons dan_balloons : ℕ) 
  (h1 : fred_balloons = 10) 
  (h2 : sam_balloons = 46) 
  (h3 : dan_balloons = 16) 
  (total : fred_balloons + sam_balloons + dan_balloons = 72) :
  fred_balloons + sam_balloons + dan_balloons = 72 := 
sorry

end total_balloons_l331_331298


namespace hyundai_to_dodge_ratio_l331_331742

theorem hyundai_to_dodge_ratio 
  (total_vehicles : ℕ)
  (half_dodge : total_vehicles / 2)
  (some_hyundai : ℕ)
  (kia_vehicles : ℕ)
  (htotal : total_vehicles = 400)
  (hdodge : half_dodge = 200)
  (hkia : kia_vehicles = 100) :
  let hyundai_vehicles := total_vehicles - (half_dodge + kia_vehicles)
  in (hyundai_vehicles : half_dodge) = (1 : 2) := 
by
  sorry

end hyundai_to_dodge_ratio_l331_331742


namespace total_area_enclosed_by_curve_l331_331868

noncomputable def region (x y : ℝ): Prop := x^2 + y^2 = sqrt (|x|) + sqrt (|y|)

theorem total_area_enclosed_by_curve : 
  let R := { p : ℝ × ℝ | region p.1 p.2 } in
  measure_theory.measure.region_area R = Real.pi :=
sorry

end total_area_enclosed_by_curve_l331_331868


namespace variance_of_xi_l331_331836

noncomputable def ξ : Type := {n : ℕ // n ≤ 2}

-- Define the probabilities
def P (x : ξ → ℝ) (event : ξ → Prop) : ℝ := 
  let prob_values := if event 0 then [1/5] else [] ++ 
                     if event 1 then [3/5] else [] ++ 
                     if event 2 then [1/5] else []
  prob_values.sum

-- Definition of expected value
def E (x : ξ → ℝ) : ℝ := 
  ∑ i, x i * P x (λ y, y = i)

-- Definition of variance
def D (x : ξ → ℝ) : ℝ := 
  E (λ n, (x n - E x)^2)

-- Define the random variable ξ
def ξ_values : ξ → ℝ
| 0 := 0
| 1 := 1
| 2 := 2

-- Proposition stating the problem
theorem variance_of_xi : D ξ_values = 2 / 5 := 
by 
  sorry

end variance_of_xi_l331_331836


namespace hiking_distance_l331_331244

theorem hiking_distance (d : ℝ) (h1 : d < 8) (h2 : d > 7) (h3 : d ≠ 5) : d ∈ set.Ioo 7 8 :=
by {
    sorry,
}

end hiking_distance_l331_331244


namespace quadrilateral_has_smallest_perimeter_when_parallelogram_l331_331136

-- Let us define some variables to represent the quadrilateral and its properties
variables {A B C D : Type} [quadrilateral A B C D]
variables {AC BD : ℝ} (theta : ℝ)

-- Definition of a quadrilateral with given diagonals and angle between them.
structure has_diagonals_and_angle (A B C D : Type) [quadrilateral A B C D] :=
(diagonal_AC_length : ℝ)
(diagonal_BD_length : ℝ)
(angle_between_diagonals : ℝ)

-- Given the quadrilateral with certain properties
variables [has_diagonals_and_angle A B C D] 

-- Statement: Prove that the perimeter of the quadrilateral is smallest when it forms a parallelogram.
theorem quadrilateral_has_smallest_perimeter_when_parallelogram 
  (h : has_diagonals_and_angle A B C D) :
  smallest_perimeter A B C D <-> is_parallelogram A B C D := 
  sorry

end quadrilateral_has_smallest_perimeter_when_parallelogram_l331_331136


namespace area_of_triangle_APB_is_sqrt_8_l331_331206

-- Given definitions and coordinates
def A := (0, 0, 0 : ℝ × ℝ × ℝ)
def B := (2, 0, 0 : ℝ × ℝ × ℝ)
def E := (0, 0, 2 : ℝ × ℝ × ℝ)
def F := (2, 0, 2 : ℝ × ℝ × ℝ)
def P := ((1, 0, 2 : ℝ × ℝ × ℝ))

-- Function to calculate the area of a triangle given coordinates of three points
noncomputable def triangle_area (A B C : ℝ × ℝ × ℝ) : ℝ :=
  1 / 2 * real.sqrt ((A.1 * ((B.2 - C.2) * (B.3 - C.3)) +
                      B.1 * ((C.2 - A.2) * (C.3 - A.3)) +
                      C.1 * ((A.2 - B.2) * (A.3 - B.3)))^2)

-- The theorem to prove
theorem area_of_triangle_APB_is_sqrt_8 : triangle_area A B P = real.sqrt 8 :=
by sorry

end area_of_triangle_APB_is_sqrt_8_l331_331206


namespace triangle_XYZ_equilateral_and_side_length_l331_331527

-- Definitions based on the conditions
variables {A B C X Y Z : Type}
variables (a r L : ℝ)
variables (hABC : (A, B, C) form_equilateral_triangle_with_side_length a)
variables (hCircles : circles_with_radius_r_intersect_at_points_outside_triangle A B C r X Y Z)
variables (hr_lt_a : r < a) (h2r_gt_a : 2 * r > a)

-- The proof problem consisting of the required proof of equilateral triangle and side length
theorem triangle_XYZ_equilateral_and_side_length :
  (is_equilateral_triangle X Y Z) ∧ (L = (a / 2) + (sqrt 3) * (sqrt (r^2 - (a^2 / 4))))
  :=
sorry

end triangle_XYZ_equilateral_and_side_length_l331_331527


namespace find_c_l331_331272

-- Definitions for the conditions
def line1 (x y : ℝ) : Prop := 4 * y + 2 * x + 6 = 0
def line2 (x y : ℝ) (c : ℝ) : Prop := 5 * y + c * x + 4 = 0
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Main theorem
theorem find_c (c : ℝ) : 
  (∀ x y : ℝ, line1 x y → y = -1/2 * x - 3/2) ∧ 
  (∀ x y : ℝ, line2 x y c → y = -c/5 * x - 4/5) ∧ 
  perpendicular (-1/2) (-c/5) → 
  c = -10 := by
  sorry

end find_c_l331_331272


namespace simplify_and_evaluate_expression_l331_331143

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.sqrt 3 + 1) :
  (1 - 1 / m) / ((m ^ 2 - 2 * m + 1) / m) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l331_331143


namespace football_players_count_l331_331443

def total_students : ℕ := 450
def play_cricket : ℕ := 175
def neither_sport : ℕ := 50
def both_sports : ℕ := 100
def play_football : ℕ := total_students - neither_sport + both_sports - play_cricket

theorem football_players_count : play_football = 325 := by
  calc
    play_football = (total_students - neither_sport) - play_cricket + both_sports : by sorry 
    ... = 325 : by sorry

end football_players_count_l331_331443


namespace trig_expression_value_l331_331942

noncomputable def cos_pi_over_3 := Real.cos (Real.pi / 3)
noncomputable def tan_pi_over_4 := Real.tan (Real.pi / 4)

theorem trig_expression_value : 
  Real.cos (11 * Real.pi / 3) + Real.tan (-3 * Real.pi / 4) = -1 / 2 :=
by
  sorry

end trig_expression_value_l331_331942


namespace smallest_four_digit_remainder_l331_331972

theorem smallest_four_digit_remainder :
  ∃ N : ℕ, (N % 6 = 5) ∧ (1000 ≤ N ∧ N ≤ 9999) ∧ (∀ M : ℕ, (M % 6 = 5) ∧ (1000 ≤ M ∧ M ≤ 9999) → N ≤ M) ∧ N = 1001 :=
by
  sorry

end smallest_four_digit_remainder_l331_331972


namespace seq_infinite_even_seq_infinite_odd_l331_331170

-- Definition of the sequence according to the given conditions
def seq (n : ℕ) : ℕ :=
  Nat.recOn n 2 (λ n a, Nat.floor (1.5 * a))

-- Proving that the sequence contains infinitely many even numbers
theorem seq_infinite_even : ∀ n : ℕ, ∃ m : ℕ, m > n ∧ seq m % 2 = 0 :=
by
  sorry

-- Proving that the sequence contains infinitely many odd numbers
theorem seq_infinite_odd : ∀ n : ℕ, ∃ m : ℕ, m > n ∧ seq m % 2 = 1 :=
by
  sorry

end seq_infinite_even_seq_infinite_odd_l331_331170


namespace range_of_a_l331_331629

noncomputable def p (x : ℝ) : Prop := (1 / (x - 3)) ≥ 1

noncomputable def q (x a : ℝ) : Prop := abs (x - a) < 1

theorem range_of_a (a : ℝ) : (∀ x, p x → q x a) ∧ (∃ x, ¬ (p x) ∧ (q x a)) ↔ (3 < a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l331_331629


namespace joan_and_karl_sofas_l331_331723

variable (J K : ℝ)

theorem joan_and_karl_sofas (hJ : J = 230) (hSum : J + K = 600) :
  2 * J - K = 90 :=
by
  sorry

end joan_and_karl_sofas_l331_331723


namespace mary_final_weight_l331_331745

theorem mary_final_weight : 
  let initial_weight := 99
  let weight_loss1 := 12
  let weight_gain1 := 2 * weight_loss1
  let weight_loss2 := 3 * weight_loss1
  let weight_gain2 := 6
  initial_weight - weight_loss1 + weight_gain1 - weight_loss2 + weight_gain2 = 81 := by 
  sorry

end mary_final_weight_l331_331745


namespace b_distance_behind_proof_l331_331073

-- Given conditions
def race_distance : ℕ := 1000
def a_time : ℕ := 40
def b_delay : ℕ := 10

def a_speed : ℕ := race_distance / a_time
def b_distance_behind : ℕ := a_speed * b_delay

theorem b_distance_behind_proof : b_distance_behind = 250 := by
  -- Prove that b_distance_behind = 250
  sorry

end b_distance_behind_proof_l331_331073


namespace initial_minutes_under_plan_A_l331_331539

theorem initial_minutes_under_plan_A (x : ℕ) (planA_initial : ℝ) (planA_rate : ℝ) (planB_rate : ℝ) (call_duration : ℕ) :
  planA_initial = 0.60 ∧ planA_rate = 0.06 ∧ planB_rate = 0.08 ∧ call_duration = 3 ∧
  (planA_initial + planA_rate * (call_duration - x) = planB_rate * call_duration) →
  x = 9 := 
by
  intros h
  obtain ⟨h1, h2, h3, h4, heq⟩ := h
  -- Skipping the proof
  sorry

end initial_minutes_under_plan_A_l331_331539


namespace problem_l331_331986

noncomputable def a := 2 ^ 0.3
noncomputable def b := Real.log 0.3 / Real.log 2 -- log base 2 of 0.3
def c := 0.3 ^ 2

theorem problem {a b c : ℝ} (ha : a = 2 ^ 0.3) (hb : b = log 0.3 / log 2) (hc : c = 0.3 ^ 2) : 
  b < c ∧ c < a :=
by
  rw [ha, hb, hc]
  -- The proof would go here
  sorry

end problem_l331_331986


namespace identity_x_squared_minus_y_squared_l331_331043

theorem identity_x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : 
  x^2 - y^2 = 80 := by sorry

end identity_x_squared_minus_y_squared_l331_331043


namespace coeff_sum_eq_781_l331_331299

theorem coeff_sum_eq_781
  (a : ℕ → ℕ) 
  (h : ∀ x, (x^2 + 1) * (2 * x + 1)^9 = ∑ i in finset.range 12, a i * (x + 1)^i) :
  a 1 + a 2 + a 11 = 781 := 
sorry

end coeff_sum_eq_781_l331_331299


namespace magnitude_of_resultant_vector_is_sqrt_5_l331_331345

-- We denote the vectors a and b
def a : ℝ × ℝ := (1, -2)
def b (y : ℝ) : ℝ × ℝ := (-2, y)

-- We encode the condition that vectors are parallel
def parallel_vectors (y : ℝ) : Prop := 1 * y = (-2) * (-2)

-- We calculate the resultant vector and its magnitude
def resultant_vector (y : ℝ) : ℝ × ℝ :=
  ((3 * 1 + 2 * -2), (3 * -2 + 2 * y))

def magnitude_square (v : ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2

-- The target statement
theorem magnitude_of_resultant_vector_is_sqrt_5 (y : ℝ) (hy : parallel_vectors y) :
  magnitude_square (resultant_vector y) = 5 := by
  sorry

end magnitude_of_resultant_vector_is_sqrt_5_l331_331345


namespace findOddPairs_l331_331602

noncomputable def oddPairs := { 
  (m, n) : ℕ × ℕ // (m % 2 = 1) ∧ (n % 2 = 1) ∧ (n ∣ 3 * m + 1) ∧ (m ∣ n ^ 2 + 3)
}

theorem findOddPairs : oddPairs = 
  { (1, 1), (43, 13), (49, 37) } :=
by {
  sorry
}

end findOddPairs_l331_331602


namespace stratified_sampling_third_grade_l331_331563

theorem stratified_sampling_third_grade 
  (N : ℕ) (N3 : ℕ) (S : ℕ) (x : ℕ)
  (h1 : N = 1600)
  (h2 : N3 = 400)
  (h3 : S = 80)
  (h4 : N3 / N = x / S) :
  x = 20 := 
by {
  sorry
}

end stratified_sampling_third_grade_l331_331563


namespace non_neg_int_solutions_eq_6_l331_331164

theorem non_neg_int_solutions_eq_6 :
  {p : ℕ × ℕ // p.1 + 4 * p.2 = 20}.card = 6 :=
sorry

end non_neg_int_solutions_eq_6_l331_331164


namespace base5_first_digit_627_l331_331518

theorem base5_first_digit_627 : 
  let n := 627 in 
  let b := 5 in 
  ∃ d, (n = d * b^4 + r ∧ r < b^4 ∧ d = 1) := 
sorry

end base5_first_digit_627_l331_331518


namespace planar_graph_orientation_l331_331446

-- Define a simple planar graph
variables {V : Type} {E : Type} [SimpleGraph V E]

-- Define the main theorem stating the goal
theorem planar_graph_orientation (G : SimpleGraph V E) [PlanarGraph G] :
  ∃ (O : SimpleGraph.Digraph V), (∀ v : V, O.outdegree v ≤ 3) :=
sorry

end planar_graph_orientation_l331_331446


namespace solve_system_of_equations_l331_331797

-- Define the given system of equations and conditions
theorem solve_system_of_equations (a b c x y z : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : yz / (y + z) = a) 
  (h2 : xz / (x + z) = b) 
  (h3 : xy / (x + y) = c) :
  x = 2 * a * b * c / (a * c + a * b - b * c) ∧ 
  y = 2 * a * b * c / (a * b + b * c - a * c) ∧ 
  z = 2 * a * b * c / (a * c + b * c - a * b) := sorry

end solve_system_of_equations_l331_331797


namespace smallest_positive_multiple_of_45_divisible_by_3_l331_331867

theorem smallest_positive_multiple_of_45_divisible_by_3 
  (x : ℕ) (hx: x > 0) : ∃ y : ℕ, y = 45 ∧ 45 ∣ y ∧ 3 ∣ y ∧ ∀ z : ℕ, (45 ∣ z ∧ 3 ∣ z ∧ z > 0) → z ≥ y :=
by
  sorry

end smallest_positive_multiple_of_45_divisible_by_3_l331_331867


namespace smallest_number_of_students_at_least_18_l331_331572

noncomputable def smallest_number_of_students (n : ℕ) : Prop :=
  let total_points : ℕ := 120
  let excellent_students : ℕ := 7
  let excellent_score : ℕ := 120
  let min_score : ℕ := 70
  let mean_score : ℕ := 90
  let total_sum : ℕ := excellent_students * excellent_score + (n - excellent_students) * min_score in
  excellent_students * excellent_score = 840 ∧
  total_sum / n = 90 ∧
  total_sum ≥ 840 + min_score * (n - excellent_students) ∧
  n >= 18

theorem smallest_number_of_students_at_least_18 : ∃ n, smallest_number_of_students n :=
  sorry

end smallest_number_of_students_at_least_18_l331_331572


namespace cost_price_is_975_l331_331875

-- Definitions from the conditions
def selling_price : ℝ := 1170
def profit_percentage : ℝ := 0.20

-- The proof statement
theorem cost_price_is_975 : (selling_price / (1 + profit_percentage)) = 975 := by
  sorry

end cost_price_is_975_l331_331875


namespace miles_difference_l331_331757

namespace Proof

-- Definitions based on conditions
def miles_day1 : ℕ := 135
def miles_day2 : ℕ := x
def miles_day3 : ℕ := 159
def miles_day4 : ℕ := 189
def charge_frequency : ℕ := 106
def total_charges : ℕ := 7

-- Given total miles she drove using charges
def total_miles : ℕ := total_charges * charge_frequency

-- Proving the correct difference in miles between the second and first day
theorem miles_difference (x : ℕ) (h : 135 + x + 159 + 189 = total_miles) : x - miles_day1 = 124 :=
by
  -- The proof part is omitted, this is where you would use steps from the solution
  sorry

end Proof

end miles_difference_l331_331757


namespace num_ordered_triples_l331_331108

noncomputable def S_n (p : Nat) (n : Nat) : Finset (Nat × Nat × Nat) := 
  { t | let (a, b, c) := t; a + b + c = n ∧ p ∣ ⇑(Finset.prod (Finset.range n) !) / (⇑(Finset.prod (Finset.range a) !) * ⇑(Finset.prod (Finset.range b) !) * ⇑(Finset.prod (Finset.range c) !)) }

theorem num_ordered_triples (p : Nat) (n : Nat) (n_i : Fin n.succ → Nat) (t : Nat)
  (h1 : p.prime)
  (h2 : ∀ i, 0 ≤ n_i i ∧ n_i i ≤ p - 1)
  (h3 : n = ∑ i in Finset.range t, n_i i * p ^ i) :
  (S_n p n).card = Finset.prod (Finset.range t) (fun i => Nat.choose (n_i i + 2) 2) := 
sorry

end num_ordered_triples_l331_331108


namespace smallest_three_digit_number_multiple_of_conditions_l331_331564

theorem smallest_three_digit_number_multiple_of_conditions :
  ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧
  (x % 2 = 0) ∧ ((x + 1) % 3 = 0) ∧ ((x + 2) % 4 = 0) ∧ ((x + 3) % 5 = 0) ∧ ((x + 4) % 6 = 0) 
  ∧ x = 122 := 
by
  sorry

end smallest_three_digit_number_multiple_of_conditions_l331_331564


namespace compare_abc_l331_331593

noncomputable def a := 0.31 ^ 2
noncomputable def b := Real.log 0.31 / Real.log 2
noncomputable def c := 2 ^ 0.31

theorem compare_abc : b < a ∧ a < c := by
  sorry

end compare_abc_l331_331593


namespace solve_for_y_l331_331796

theorem solve_for_y :
  (∀ y : ℝ, (1 / 8) ^ (3 * y + 12) = 32 ^ (3 * y + 7)) →
  y = -71 / 24 :=
begin
  sorry
end

end solve_for_y_l331_331796


namespace find_width_of_metallic_sheet_l331_331552

noncomputable def width_of_metallic_sheet (w : ℝ) : Prop :=
  let length := 48
  let square_side := 8
  let new_length := length - 2 * square_side
  let new_width := w - 2 * square_side
  let height := square_side
  let volume := new_length * new_width * height
  volume = 5120

theorem find_width_of_metallic_sheet (w : ℝ) :
  width_of_metallic_sheet w -> w = 36 := 
sorry

end find_width_of_metallic_sheet_l331_331552


namespace find_d_l331_331227

-- Conditions
def rectangle_area : ℝ := 6
def equal_area_region : ℝ := rectangle_area / 2

-- Equation of the line passing through (d, 0) and (4, 2)
def line_eq (d : ℝ) (x : ℝ) : ℝ := (2 / (4 - d)) * (x - d)

-- Area of the triangle formed by the line
def triangle_area (d : ℝ) : ℝ := (1 / 2) * (4 - d) * 2

theorem find_d : ∃ d : ℝ, triangle_area d = equal_area_region ∧ d = 1 :=
by { existsi 1, split, simp [triangle_area, equal_area_region], sorry }


end find_d_l331_331227


namespace find_angle_between_a_b_l331_331407

noncomputable section

variables {V : Type*} [inner_product_space ℝ V]

def is_unit_vector (v : V) : Prop := ⟪v, v⟫ = 1

variables (a b c : V)
variables (h1 : is_unit_vector a) (h2 : is_unit_vector b) (h3 : is_unit_vector c)
variables (h4 : a + b + 2 • c = 0)

theorem find_angle_between_a_b : real.angle a b = 0 :=
by
  -- Proof goes here
  sorry

end find_angle_between_a_b_l331_331407


namespace rocket_altitude_time_l331_331460

theorem rocket_altitude_time (a₁ d : ℕ) (n : ℕ) (h₁ : a₁ = 2) (h₂ : d = 2)
  (h₃ : n * a₁ + (n * (n - 1) * d) / 2 = 240) : n = 15 :=
by
  -- The proof is ignored as per instruction.
  sorry

end rocket_altitude_time_l331_331460


namespace sheila_hourly_wage_l331_331877

theorem sheila_hourly_wage (h1 : 8 * 3 = 24)
                           (h2 : 6 * 2 = 12)
                           (h3 : 24 + 12 = 36)
                           (h4 : 504 / 36 = 14) : 
                           sheila_earnings_per_hour (8 * 3 + 6 * 2 = 36) (504 = 36 * 14) : 
                           504 / 36 = 14 := 
begin
  sorry
end

end sheila_hourly_wage_l331_331877


namespace remainder_43_pow_43_plus_43_mod_44_l331_331211

theorem remainder_43_pow_43_plus_43_mod_44 : (43^43 + 43) % 44 = 42 :=
by 
    sorry

end remainder_43_pow_43_plus_43_mod_44_l331_331211


namespace sum_of_distances_l331_331408

-- Definitions of points and the focus of the parabola
def P1 : ℝ × ℝ := (-5, 25)
def P2 : ℝ × ℝ := (3, 9)
def P3 : ℝ × ℝ := (10, 100)
def P4 : ℝ × ℝ := (-8, 64)
def focus : ℝ × ℝ := (0, 1 / 4)

-- Definition of the Euclidean distance
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Proof that the sum of the distances from the focus is approximately 199.114
theorem sum_of_distances :
  dist focus P1 + dist focus P2 + dist focus P3 + dist focus P4 ≈ 199.114 :=
  by
  -- Proof omitted for brevity
  sorry

end sum_of_distances_l331_331408


namespace final_answer_l331_331005

def f (x : ℝ) : ℝ := sorry

axiom functional_eqn : ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem final_answer : f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : ℝ, f (-x) = f x) ∧ ¬ (∃ ε > 0, ∀ h : ℝ, abs h < ε → f h ≥ f 0) := 
by
  -- omit the proof steps that were provided in the solution
  sorry

end final_answer_l331_331005


namespace find_a_l331_331822

theorem find_a (a b c : ℤ) (h_vertex : ∀ x, (x - 2)*(x - 2) * a + 3 = a*x*x + b*x + c) 
  (h_point : (a*(3 - 2)*(3 -2) + 3 = 6)) : a = 3 :=
by
  sorry

end find_a_l331_331822


namespace age_difference_is_16_l331_331472

-- Variables
variables (y : ℕ) -- y represents the present age of the younger person

-- Conditions from the problem
def elder_present_age := 30
def elder_age_6_years_ago := elder_present_age - 6
def younger_age_6_years_ago := y - 6

-- Given condition 6 years ago:
def condition_6_years_ago := elder_age_6_years_ago = 3 * younger_age_6_years_ago

-- The theorem to prove the difference in ages is 16 years
theorem age_difference_is_16
  (h1 : elder_present_age = 30)
  (h2 : condition_6_years_ago) :
  elder_present_age - y = 16 :=
by sorry

end age_difference_is_16_l331_331472


namespace width_of_metallic_sheet_l331_331554

theorem width_of_metallic_sheet 
  (length : ℕ)
  (new_volume : ℕ) 
  (side_length_of_square : ℕ)
  (height_of_box : ℕ)
  (new_length : ℕ)
  (new_width : ℕ)
  (w : ℕ) : 
  length = 48 → 
  new_volume = 5120 → 
  side_length_of_square = 8 → 
  height_of_box = 8 → 
  new_length = length - 2 * side_length_of_square → 
  new_width = w - 2 * side_length_of_square → 
  new_volume = new_length * new_width * height_of_box → 
  w = 36 := 
by 
  intros _ _ _ _ _ _ _ 
  sorry

end width_of_metallic_sheet_l331_331554


namespace only_n_m_good_set_is_N_iff_l331_331399

-- Definitions for conditions
variables (n m : ℕ) (S : set ℕ)

def is_n_m_good (n m : ℕ) (S : set ℕ) : Prop :=
  m ∈ S ∧ 
  (∀ a ∈ S, ∀ d ∈ (finset.divisors a).to_set, d ∈ S) ∧ 
  (∀ (a b : ℕ), a ≠ b → a ∈ S → b ∈ S → (a^n + b^n) ∈ S)

-- The main theorem statement
theorem only_n_m_good_set_is_N_iff (n m : ℕ) :
  (∀ S : set ℕ, is_n_m_good n m S → S = set.univ) ↔ odd n :=
by
  sorry

end only_n_m_good_set_is_N_iff_l331_331399


namespace cosine_sum_l331_331455

noncomputable def ω : ℝ := Real.pi / 3

def f (x : ℝ) : ℝ := Real.cos (ω * x)

theorem cosine_sum :
    f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + 
    f 7 + f 8 + f 9 + f 10 + f 11 + f 12 +
    -- Summation continues up to f 2018
    f 2016 + f 2017 + f 2018 = 0 :=
by
  sorry

end cosine_sum_l331_331455


namespace derivative_of_f_l331_331352

-- Define the function f(x)
noncomputable def f (a x : ℝ) : ℝ := sqrt (1 + a^2) + sqrt (1 - x)

-- Define the derivative of f(x), using the correct answer from the solution
noncomputable def f_prime (x : ℝ) : ℝ := -1 / (2 * sqrt (1 - x))

-- The statement to be proven
theorem derivative_of_f (a x : ℝ) : (deriv (f a) x) = f_prime x := by
  sorry

end derivative_of_f_l331_331352


namespace isosceles_of_equal_segments_l331_331763

open Classical

variables {A B C M3 P A1 B1 : Type}

structure Triangle (A B C : Type) :=
(midpoint : A → B → C)
(median  : C → C → C → C → Prop)

def isosceles_triangle (ABC : Triangle A B C): Prop :=
∀ A B C : A, ∃ AB : Type, AB = AC

theorem isosceles_of_equal_segments (ABC : Triangle A B C)
  (h1 : ABC.median A B C M3)
  (h2 : ABC.midpoint A B M3)
  (h3 : ∀ P : C, P ∈ ABC.median A B C M3) 
  (h4 : ∀ P: C, ∃ A1, B1 : Type, ∀ A1 B1 ∈ P, AA1 = BB1 ) :
  isosceles_triangle ABC :=
sorry

end isosceles_of_equal_segments_l331_331763


namespace final_answer_l331_331026

def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)
def g (x : ℝ) : ℝ := 2 * Real.sin (2 * (x - Real.pi / 4) + Real.pi / 3)

-- Proposition p
def p : Prop :=
  ∀ x, g x = f (x - Real.pi / 4) ∧ (∀ x ∈ Set.Icc (-Real.pi / 3 : ℝ) (0 : ℝ), g x > g (x - 0.01))

-- Proposition q
def q : Prop :=
  ∀ x, f (-x) = f (x + 3)

-- Combined proposition
def combined_proposition : Prop := ¬p ∧ q

theorem final_answer : combined_proposition := by
  sorry

end final_answer_l331_331026


namespace problem1_problem2_l331_331067

-- Problem (1)
theorem problem1 (a b : ℝ) (h1 : c = 2) (h2 : C = π / 3) (h3 : (1 / 2) * a * b * (Real.sin C) = sqrt 3) : a = 2 ∧ b = 2 := 
sorry

-- Problem (2)
theorem problem2 (a b : ℝ) (h1 : c = 2) (h2 : C = π / 3) (h3 : Real.sin B = 2 * Real.sin A) :
  (1/2) * a * b * (Real.sin C) = (2 * Real.sqrt 3) / 3 :=
sorry

end problem1_problem2_l331_331067


namespace xy_square_diff_l331_331052

theorem xy_square_diff (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end xy_square_diff_l331_331052


namespace largest_number_with_digits_sum_13_l331_331863

-- Definition of valid number with conditions
def is_valid_number (n : ℕ) : Prop :=
  ∀ d ∈ digits 10 n, d = 7 ∨ d = 1

def digits_add_up_to_13 (n : ℕ) : Prop :=
  (digits 10 n).sum = 13

-- Proof statement
theorem largest_number_with_digits_sum_13 : ∃ n : ℕ, is_valid_number n ∧ digits_add_up_to_13 n ∧ n = 7111111 := by
  sorry

end largest_number_with_digits_sum_13_l331_331863


namespace largest_k_dividing_A_l331_331608

def A : ℤ := 1990^(1991^1992) + 1991^(1990^1992) + 1992^(1991^1990)

theorem largest_k_dividing_A :
  1991^(1991) ∣ A := sorry

end largest_k_dividing_A_l331_331608


namespace probability_at_least_one_third_correct_l331_331720

theorem probability_at_least_one_third_correct :
  ∑ k in finset.range (13) \ finset.range (5), 
    nat.choose 12 k * (1/2)^12 = 825 / 1024 :=
by sorry

end probability_at_least_one_third_correct_l331_331720


namespace second_bill_late_fee_l331_331925

def first_bill_amount : ℕ := 200
def first_bill_interest_rate : ℝ := 0.10
def first_bill_months : ℕ := 2
def second_bill_amount : ℕ := 130
def second_bill_months : ℕ := 6
def third_bill_first_month_fee : ℕ := 40
def third_bill_second_month_fee : ℕ := 80
def total_amount_owed : ℕ := 1234

theorem second_bill_late_fee (x : ℕ) 
(h : first_bill_amount * (first_bill_interest_rate * first_bill_months) + first_bill_amount + third_bill_first_month_fee + third_bill_second_month_fee + second_bill_amount + second_bill_months * x = total_amount_owed) : x = 124 :=
sorry

end second_bill_late_fee_l331_331925


namespace true_propositions_l331_331430

def f (x : ℝ) : ℝ := if x ∈ ℚ then 1 else 0

theorem true_propositions :
  (∀ x, f (f x) = 1) = false ∧
  (∀ x, f (-x) = f x) = true ∧
  (∀ x (T : ℚ), T ≠ 0 → f (x + T) = f x) = true ∧
  (∃ A B C : ℝ × ℝ, A.1 ∈ ℚ ∧ A.2 = 1 ∧ ((B.1 = A.1 - √3 / 3 ∧ B.2 = 0) ∧ (C.1 = A.1 + √3 / 3 ∧ C.2 = 0)) ∧
      (dist A B = dist B C ∧ dist B C = dist C A) = true :=
by
  sorry

end true_propositions_l331_331430


namespace find_M_l331_331730

def T : Set ℕ := { n | ∃ k : ℕ, k ≤ 7 ∧ n = 3^k }

def compute_M (s : Set ℕ) : ℕ :=
  let elements := s.toList
  let diffs := elements.product elements |>.filter (fun (x, y) => x > y) |>.map (fun (x, y) => x - y)
  diffs.foldl (· + ·) 0

theorem find_M : compute_M T = 18689 :=
  sorry

end find_M_l331_331730


namespace min_max_of_function_l331_331289

noncomputable def f (x : ℝ) : ℝ :=
  4^(x - 1/2) - 3 * 2^x + 5

theorem min_max_of_function :
  (∀ x ∈ Icc (0 : ℝ) 2, f x ≥ 1/2) ∧ (∃ x ∈ Icc (0 : ℝ) 2, f x = 1/2) ∧
  (∀ x ∈ Icc (0 : ℝ) 2, f x ≤ 5/2) ∧ (∃ x ∈ Icc (0 : ℝ) 2, f x = 5/2) :=
by sorry

end min_max_of_function_l331_331289


namespace distance_from_point_to_x_axis_l331_331377

def distance_to_x_axis (p : ℝ × ℝ) : ℝ :=
  |p.2|

theorem distance_from_point_to_x_axis :
  let p := (-2, -Real.sqrt 5)
  distance_to_x_axis p = Real.sqrt 5 := by
  sorry

end distance_from_point_to_x_axis_l331_331377


namespace min_tablets_to_extract_l331_331892

noncomputable def min_tablets_needed : ℕ :=
  let A := 10
  let B := 14
  let C := 18
  let D := 20
  let required_A := 3
  let required_B := 4
  let required_C := 3
  let required_D := 2
  let worst_case := B + C + D
  worst_case + required_A -- 14 + 18 + 20 + 3 = 55

theorem min_tablets_to_extract : min_tablets_needed = 55 :=
by {
  let A := 10
  let B := 14
  let C := 18
  let D := 20
  let required_A := 3
  let required_B := 4
  let required_C := 3
  let required_D := 2
  let worst_case := B + C + D
  have h : worst_case + required_A = 55 := by decide
  exact h
}

end min_tablets_to_extract_l331_331892


namespace fifth_largest_divisor_l331_331197

theorem fifth_largest_divisor (n : ℕ) (h : n = 2520000000) : by_A ∃ d, fifth_largest_divisor n d ∧ d = 105000000 :=
by sorry

end fifth_largest_divisor_l331_331197


namespace complex_problem_l331_331423

theorem complex_problem (a : ℝ) (b : ℝ) (h₁ : ∀ z : Complex, z = a + 4 * Complex.i → z / (z + b) = 4 * Complex.i) : b = 17 :=
sorry

end complex_problem_l331_331423


namespace angle_OP_AM_eq_π_div_2_l331_331090

-- Definitions based on the conditions
variables {A B C D A1 B1 C1 D1 M O P : Point}
variables [IsCube A B C D A1 B1 C1 D1]
variables : 
(M : Midpoint D D1)
(O : Center ABCD)
(P : EdgePoint A1 B1)
  
-- Prove that the angle between the lines OP and AM is π/2
theorem angle_OP_AM_eq_π_div_2 :
  ∠ (LineOP O P) (LineAM A M) = π / 2 :=
sorry

end angle_OP_AM_eq_π_div_2_l331_331090


namespace avg_weight_increase_l331_331478

theorem avg_weight_increase
    (initial_avg : ℝ) (n : ℝ := 6) (old_weight : ℝ := 65) (new_weight : ℝ := 80) :
    (new_weight - old_weight) / n = 2.5 :=
by
  have h1 : new_weight - old_weight = 15 := by norm_num
  have h2 : n = 6 := by norm_num
  rw [h1, h2]
  norm_num
  sorry

end avg_weight_increase_l331_331478


namespace monthly_rent_needed_l331_331569

-- Define the conditions
def investment : ℝ := 15000
def property_taxes : ℝ := 400
def desired_annual_roi := 0.07
def maintenance_fee_percentage := 0.15

-- Define the final statement to prove
theorem monthly_rent_needed : 
  ∃ (rent : ℝ), 
  (let annual_roi := desired_annual_roi * investment in
   let total_annual_earnings := annual_roi + property_taxes in
   let monthly_earnings_needed := total_annual_earnings / 12 in
   let effective_rent := monthly_earnings_needed / (1 - maintenance_fee_percentage) in
   effective_rent ≈ 142.16) :=
sorry

end monthly_rent_needed_l331_331569


namespace pencil_cost_l331_331276

-- Definitions of given conditions
def has_amount : ℝ := 5.00  -- Elizabeth has 5 dollars
def borrowed_amount : ℝ := 0.53  -- She borrowed 53 cents
def needed_amount : ℝ := 0.47  -- She needs 47 cents more

-- Theorem to prove the cost of the pencil
theorem pencil_cost : has_amount + borrowed_amount + needed_amount = 6.00 := by 
  sorry

end pencil_cost_l331_331276


namespace min_value_y_plus_PQ_l331_331024

variables {x y : ℝ}
def Q : ℝ × ℝ := (4, 0)
def parabola (x : ℝ) : ℝ := x^2 / 4 + 2
def P (x : ℝ) : ℝ × ℝ := (x, parabola x)
def distance (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem min_value_y_plus_PQ : 
    ∃ x : ℝ, y = parabola x → y + distance (P x) Q = 6 :=
begin
  sorry
end

end min_value_y_plus_PQ_l331_331024


namespace determine_speeds_l331_331930

structure Particle :=
  (speed : ℝ)

def distance : ℝ := 3.01 -- meters

def initial_distance (m1_speed : ℝ) : ℝ :=
  301 - 11 * m1_speed -- converted to cm

theorem determine_speeds :
  ∃ (m1 m2 : Particle), 
  m1.speed = 11 ∧ m2.speed = 7 ∧ 
  ∀ t : ℝ, (t = 10 ∨ t = 45) →
  (initial_distance m1.speed) = t * (m1.speed + m2.speed) ∧
  20 * m2.speed = 35 * (m1.speed - m2.speed) :=
by {
  sorry 
}

end determine_speeds_l331_331930


namespace minimum_value_expression_l331_331332

noncomputable def f (x : ℝ) : ℝ := exp x - exp (-x) + x^3 + 3 * x

theorem minimum_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : f (2 * a - 1) + f (b - 1) = 0) : 
  ∃ a b, (a > 0) ∧ (b > 0) ∧ (f (2 * a - 1) + f (b - 1) = 0) ∧
  (frac(2 * a^2) (a + 1) + frac(b^2 + 1) b = 9 / 4) := sorry

end minimum_value_expression_l331_331332


namespace scalene_triangle_division_l331_331268

-- Assuming Point is a valid type representing a point in the 2D plane
structure Triangle := 
  (A : Point)
  (B : Point)
  (C : Point)
  (scalene : True) -- indicating the triangle is scalene, this would involve additional proof in a full implementation

-- Assume two triangles are similar if their corresponding angles are equal and their sides are proportional
def triangles_similar (T1 T2 : Triangle) : Prop := sorry -- definition of similarity property

theorem scalene_triangle_division (T : Triangle) (scalene : T.scalene) :
  ∃ (K M P : Point), 
    (K ∈ segment T.A T.B) ∧ 
    (M ∈ segment T.B T.C) ∧
    (P ∈ segment T.C T.A) ∧
    let T1 := Triangle.mk T.A K P sorry in
    let T2 := Triangle.mk K T.B M sorry in
    (triangles_similar T1 T ∨ triangles_similar T2 T) ∧
    (triangles_similar T1 T ∨ triangles_similar T2 T) := sorry

end scalene_triangle_division_l331_331268


namespace daily_production_l331_331543

-- Define the conditions
def bottles_per_case : ℕ := 9
def num_cases : ℕ := 8000

-- State the theorem with the question and the calculated answer
theorem daily_production : bottles_per_case * num_cases = 72000 :=
by
  sorry

end daily_production_l331_331543


namespace problem_solution_l331_331303

def f (x : ℝ) : ℝ := 
 if x > 0 then x + 1 
 else if x = 0 then Real.pi 
 else 0

theorem problem_solution : f (f (-1)) = Real.pi :=
  sorry

end problem_solution_l331_331303


namespace sqrt_of_second_number_l331_331512

-- Given condition: the arithmetic square root of a natural number n is x
variable (x : ℕ)
def first_number := x ^ 2
def second_number := first_number + 1

-- The theorem statement we want to prove
theorem sqrt_of_second_number (x : ℕ) : Real.sqrt (x^2 + 1) = Real.sqrt (first_number x + 1) :=
by
  sorry

end sqrt_of_second_number_l331_331512


namespace trigonometric_identity_l331_331985

theorem trigonometric_identity (θ : ℝ) (h : sin θ - 2 * cos θ = 0) : cos θ ^ 2 + sin (2 * θ) = 1 := 
sorry

end trigonometric_identity_l331_331985


namespace unique_solution_l331_331603

theorem unique_solution (x y z : ℝ) (h₁ : x^2 + y^2 + z^2 = 2) (h₂ : x = z + 2) :
  x = 1 ∧ y = 0 ∧ z = -1 :=
by
  sorry

end unique_solution_l331_331603


namespace abs_diff_squares_l331_331860

theorem abs_diff_squares (a b : ℤ) (ha : a = 103) (hb : b = 97) : |a^2 - b^2| = 1200 :=
by
  sorry

end abs_diff_squares_l331_331860


namespace cube_volume_given_surface_area_l331_331840

theorem cube_volume_given_surface_area (A : ℝ) (V : ℝ) :
  A = 96 → V = 64 :=
by
  sorry

end cube_volume_given_surface_area_l331_331840


namespace median_from_A_l331_331487

theorem median_from_A (A B C : Point) (K P Q M : Point) (a : Real) 
  (hK : midpoint B C K) 
  (hP : midpoint A C P) 
  (hQ : midpoint A B Q) 
  (hM : centroid A B C M) 
  (hcyclic : cyclic A P Q M) :
  distance A M = a / Real.sqrt 3 :=
by 
  sorry

end median_from_A_l331_331487


namespace f_of_x_l331_331037

variable (f : ℝ → ℝ)

theorem f_of_x (x : ℝ) (h : f (x - 1 / x) = x^2 + 1 / x^2) : f x = x^2 + 2 :=
sorry

end f_of_x_l331_331037


namespace ratio_R_N_l331_331405

variables (P Q M N R : ℝ)

def condition1 : Prop := R = 0.40 * M
def condition2 : Prop := M = 0.25 * Q
def condition3 : Prop := Q = 0.30 * P
def condition4 : Prop := N = 0.60 * P

theorem ratio_R_N (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : 
  R / N = 1 / 20 :=
by
  sorry

end ratio_R_N_l331_331405


namespace prove_triangle_point_C_coords_are_correct_prove_triangle_area_is_correct_l331_331384

noncomputable def triangle_point_C_coords_are_correct : Prop :=
  let B := (4, 4) in
  let angle_bisector_eq := ∀ x, y = 0 in
  let altitude_eq := ∀ x, x - 2 * y + 2 = 0 in
  let C := (10, -8) in
  C = (10, -8)

noncomputable def triangle_area_is_correct : Prop :=
  let B := (4, 4) in
  let angle_bisector_eq := ∀ x, y = 0 in
  let altitude_eq := ∀ x, x - 2 * y + 2 = 0 in
  let area := 48 in
  area = 48

theorem prove_triangle_point_C_coords_are_correct :
  triangle_point_C_coords_are_correct := by
  sorry

theorem prove_triangle_area_is_correct :
  triangle_area_is_correct := by
  sorry

end prove_triangle_point_C_coords_are_correct_prove_triangle_area_is_correct_l331_331384


namespace basketball_game_total_points_l331_331370

theorem basketball_game_total_points :
  ∃ (a d b: ℕ) (r: ℝ), 
      a = b + 2 ∧     -- Eagles lead by 2 points at the end of the first quarter
      (a + d < 100) ∧ -- Points scored by Eagles in each quarter form an increasing arithmetic sequence
      (b * r < 100) ∧ -- Points scored by Lions in each quarter form an increasing geometric sequence
      (a + (a + d) + (a + 2 * d)) = b * (1 + r + r^2) ∧ -- Aggregate score tied at the end of the third quarter
      (a + (a + d) + (a + 2 * d) + (a + 3 * d) + b * (1 + r + r^2 + r^3) = 144) -- Total points scored by both teams 
   :=
sorry

end basketball_game_total_points_l331_331370


namespace mcpherson_rent_l331_331125

theorem mcpherson_rent (
    current_rent : ℝ := 1200,
    rent_increase_pct : ℝ := 0.05,
    monthly_expenses : ℝ := 100,
    expenses_increase_pct : ℝ := 0.03,
    mrs_mcpherson_pct : ℝ := 0.30
  ) : 
  let new_rent := current_rent + (current_rent * rent_increase_pct),
      new_monthly_expenses := monthly_expenses + (monthly_expenses * expenses_increase_pct),
      total_new_monthly_expenses := new_monthly_expenses * 12,
      total_amount_needed := new_rent + total_new_monthly_expenses,
      mrs_mcpherson_amount_raised := total_amount_needed * mrs_mcpherson_pct,
      mr_mcpherson_amount_raised := total_amount_needed - mrs_mcpherson_amount_raised
  in mr_mcpherson_amount_raised = 1747.20 := by
    sorry

end mcpherson_rent_l331_331125


namespace evaluate_fraction_l331_331282

theorem evaluate_fraction :
  1 + (2 / (3 + (6 / (7 + (8 / 9))))) = 409 / 267 :=
by
  sorry

end evaluate_fraction_l331_331282


namespace necessary_condition_l331_331631

theorem necessary_condition {x m : ℝ} 
  (p : |1 - (x - 1) / 3| ≤ 2)
  (q : x^2 - 2 * x + 1 - m^2 ≤ 0)
  (hm : m > 0)
  (h_np_nq : ¬(|1 - (x - 1) / 3| ≤ 2) → ¬(x^2 - 2 * x + 1 - m^2 ≤ 0))
  : m ≥ 9 :=
sorry

end necessary_condition_l331_331631


namespace computation_result_l331_331941

theorem computation_result : 143 - 13 + 31 + 17 = 178 := 
by
  sorry

end computation_result_l331_331941


namespace tan_105_eq_neg2_sub_sqrt3_l331_331943

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l331_331943


namespace range_of_a_l331_331323

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 4 → x^2 - 2 * x + 1 - a^2 < 0) ↔ (a < -3 ∨ a > 3) :=
sorry

end range_of_a_l331_331323


namespace geometric_properties_of_pentagon_l331_331075

open Real
open Geometry

def regular_pentagon (A B C D E : Point) : Prop :=
(A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ A) ∧
(distance A B = distance B C ∧ distance B C = distance C D ∧
 distance C D = distance D E ∧ distance D E = distance E A) ∧
(distance A C = distance B D ∧ distance A D = distance B E ∧ distance A E = distance C E)

def midpoint (F C D : Point) : Prop :=
distance C F = distance F D

theorem geometric_properties_of_pentagon 
  {A B C D E F : Point}
  (hpent : regular_pentagon A B C D E)
  (hmid : midpoint F C D) :
  (∀ (x y z : Point), x ≠ y → interior_angle x y z = 108) ∧
  (angle E C D = 36 ∧ angle C E D = 36) ∧
  angle A C D = 72 :=
sorry

end geometric_properties_of_pentagon_l331_331075


namespace sum_of_midpoint_coordinates_l331_331813

theorem sum_of_midpoint_coordinates :
  let x1 := 3
  let y1 := 4
  let x2 := 9
  let y2 := 18
  let midpoint := ((x1 + x2) / 2, (y1 + y2) / 2)
  let sum_of_coordinates := midpoint.fst + midpoint.snd
  sum_of_coordinates = 17 :=
by
  let x1 := 3
  let y1 := 4
  let x2 := 9
  let y2 := 18
  let midpoint := ((x1 + x2) / 2, (y1 + y2) / 2)
  let sum_of_coordinates := midpoint.fst + midpoint.snd
  show sum_of_coordinates = 17
  sorry

end sum_of_midpoint_coordinates_l331_331813


namespace value_of_B_range_of_expression_l331_331068

variables (A B C a b c : ℝ)
variables (triangle_ABC : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
variables (angles_in_triangle : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
variables (sum_angles : A + B + C = π)
variables (opposite_sides : a = (b * sin C / sin B) ∧ b = (c * sin A / sin C) ∧ c = (a * sin B / sin A))
variables (arithmetic_sequence : 2 * b * cos B = a * cos C + c * cos A)

-- Part 1: Prove B = π / 3
theorem value_of_B : B = π / 3 :=
begin
  sorry
end

-- Part 2: Determine the range of 2 * sin^2 A + cos (A - C):
theorem range_of_expression :
  ∃ (x : ℝ), x ∈ Icc (-1 / 2) (1 + sqrt 3) ∧ 2 * sin A ^ 2 + cos (A - C) = x :=
begin
  sorry
end

end value_of_B_range_of_expression_l331_331068


namespace find_z_l331_331356

variable {x y z : ℝ}

theorem find_z (h : (1/x + 1/y = 1/z)) : z = (x * y) / (x + y) :=
  sorry

end find_z_l331_331356


namespace difference_in_shares_l331_331457

theorem difference_in_shares:
  let suresh_investment := 18000 in
  let rohan_investment := 12000 in
  let sudhir_investment := 9000 in
  let priya_investment := 15000 in
  let akash_investment := 10000 in

  let suresh_time := 12 in
  let rohan_time := 9 in
  let sudhir_time := 8 in
  let priya_time := 6 in
  let akash_time := 6 in

  let total_profit := 5948 in

  let suresh_product := suresh_investment * suresh_time in
  let rohan_product := rohan_investment * rohan_time in
  let sudhir_product := sudhir_investment * sudhir_time in
  let priya_product := priya_investment * priya_time in
  let akash_product := akash_investment * akash_time in

  let total_investment_time := suresh_product + rohan_product + sudhir_product + priya_product + akash_product in

  let rohan_share := (rohan_product * total_profit) / total_investment_time in
  let sudhir_share := (sudhir_product * total_profit) / total_investment_time in

  rohan_share - sudhir_share = 393 := sorry

end difference_in_shares_l331_331457


namespace total_artworks_l331_331434

theorem total_artworks (students : ℕ) (group1_artworks : ℕ) (group2_artworks : ℕ) (total_students : students = 10) 
    (artwork_group1 : group1_artworks = 5 * 3) (artwork_group2 : group2_artworks = 5 * 4) : 
    group1_artworks + group2_artworks = 35 :=
by
  sorry

end total_artworks_l331_331434


namespace circle_tangent_angle_l331_331102

noncomputable def circle {α : Type} [EuclideanSpace α] := ℝ × α × ℝ -- representing a circle as (radius, center, radius squared)

variables (Ω₁ Ω₂ : circle ℝ) (A P X Y Q R : ℝ × ℝ)

def tangent_internal (Ω₁ Ω₂ : circle ℝ) (A : ℝ × ℝ) : Prop := sorry -- definition of tangent circles internally
def on_circle (Ω : circle ℝ) (P : ℝ × ℝ) : Prop := sorry -- definition of a point being on the circle
def tangent_point (P : ℝ × ℝ) (Ω : circle ℝ) (X : ℝ × ℝ) : Prop := sorry -- definition of a tangent point
def intersects_again (P ℝ) (Ω : circle ℝ) (X Q : ℝ × ℝ) : Prop := sorry -- definition of intersects the circle again

theorem circle_tangent_angle (Ω₁ Ω₂ : circle ℝ) (A P X Y Q R : ℝ × ℝ) 
  (h1 : tangent_internal Ω₁ Ω₂ A)
  (h2 : on_circle Ω₂ P)
  (h3 : tangent_point P Ω₁ X)
  (h4 : tangent_point P Ω₁ Y)
  (h5 : intersects_again P Ω₂ X Q)
  (h6 : intersects_again P Ω₂ Y R) :
  angle Q A R = 2 * angle X A Y := by
  sorry

end circle_tangent_angle_l331_331102


namespace final_answer_l331_331008

def f (x : ℝ) : ℝ := sorry

axiom functional_eqn : ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem final_answer : f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : ℝ, f (-x) = f x) ∧ ¬ (∃ ε > 0, ∀ h : ℝ, abs h < ε → f h ≥ f 0) := 
by
  -- omit the proof steps that were provided in the solution
  sorry

end final_answer_l331_331008


namespace tangent_eq_normal_eq_l331_331879

noncomputable def curve_x (t : ℝ) := (1 + real.log t) / t^2
noncomputable def curve_y (t : ℝ) := (3 + 2 * real.log t) / t

def tangent_line (x : ℝ) : ℝ := x + 2
def normal_line (x : ℝ) : ℝ := -x + 4

/-- The equation of the tangent line to the curve at t = 1 is y = x + 2 --/
theorem tangent_eq (x y : ℝ) (t0 : ℝ) (h : t0 = 1) :
  y = tangent_line x ↔ (∃ t, t = t0 ∧ curve_x t = x ∧ curve_y t = y) :=
sorry

/-- The equation of the normal line to the curve at t = 1 is y = -x + 4 --/
theorem normal_eq (x y : ℝ) (t0 : ℝ) (h : t0 = 1) :
  y = normal_line x ↔ (∃ t, t = t0 ∧ curve_x t = x ∧ curve_y t = y) :=
sorry

end tangent_eq_normal_eq_l331_331879


namespace min_value_fraction_l331_331624

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1) :
  ∃ c : ℝ, c = 16 ∧ (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1 / x + 9 / y ≥ c) :=
by
  use 16
  intros x y hx hy hxy
  have fact : x + y = 1 := hxy
  -- The code here skips the detailed proof steps and directly uses the result.
  sorry

end min_value_fraction_l331_331624


namespace coefficient_x_squared_l331_331962

theorem coefficient_x_squared (x : ℝ) (h : 0 ≤ x) : 
  (∀ n : ℕ, n ≥ 0 → n ≤ 5 → (∑ k in Finset.range (n + 1), (Nat.choose 5 k) * (1 : ℝ)^(5 - k) * ((-1) * Real.sqrt x)^k) * Real.sqrt x = -10 * x^2) :=
sorry

end coefficient_x_squared_l331_331962


namespace find_value_l331_331984

variable {a b : ℝ}

theorem find_value (h : 2 * a + b + 1 = 0) : 1 + 4 * a + 2 * b = -1 := 
by
  sorry

end find_value_l331_331984


namespace true_propositions_l331_331573

-- Propositions given in the problem
def conv1 := ∀ (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C],
  ∀ (AB AC BC : A ≃ B ≃ C), 
  (∠ ABC > ∠ ACB) → (dist AB > dist AC)

def neg2 := ∀ (a b : ℕ), (a * b ≠ 0) → (a = 0 ∨ b ≠ 0)

def contra3 := ∀ (a b : ℕ), 
  (a ≠ 0 ∧ b ≠ 0) → (a * b ≠ 0)

def conv4 := ∀ (P Q R S : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace S],
  ∀ (diagonalPQ diagonalQR diagonalRS diagonalSP : P ≃ Q ≃ R ≃ S),
  (bisects diagonalPQ diagonalSR ∧ bisects diagonalQR diagonalSP) → parallelogram P Q R S

-- Proof problem: Prove there are 4 true propositions among the given conditions.
theorem true_propositions : 
  conv1 ∧ neg2 ∧ contra3 ∧ conv4 ↔ true :=
by
  sorry

end true_propositions_l331_331573


namespace no_integer_solutions_to_system_l331_331949

theorem no_integer_solutions_to_system :
  ¬ ∃ (x y z : ℤ),
    x^2 - 2 * x * y + y^2 - z^2 = 17 ∧
    -x^2 + 3 * y * z + 3 * z^2 = 27 ∧
    x^2 - x * y + 5 * z^2 = 50 :=
by
  sorry

end no_integer_solutions_to_system_l331_331949


namespace millie_bracelets_left_l331_331123

def millie_bracelets_initial : ℕ := 9
def millie_bracelets_lost : ℕ := 2

theorem millie_bracelets_left : millie_bracelets_initial - millie_bracelets_lost = 7 := 
by
  sorry

end millie_bracelets_left_l331_331123


namespace meaningful_sqrt_l331_331361

theorem meaningful_sqrt (x : ℝ) (h : x - 5 ≥ 0) : x = 6 :=
sorry

end meaningful_sqrt_l331_331361


namespace simplify_fraction_l331_331599

def a : ℕ := 2016
def b : ℕ := 2017

theorem simplify_fraction :
  (a^4 - 2 * a^3 * b + 3 * a^2 * b^2 - a * b^3 + 1) / (a^2 * b^2) = 1 - 1 / b^2 :=
by
  sorry

end simplify_fraction_l331_331599


namespace range_of_a_l331_331029

theorem range_of_a (a : ℝ) 
  (h : ∀ x y, (a * x^2 - 3 * x + 2 = 0) ∧ (a * y^2 - 3 * y + 2 = 0) → x = y) :
  a = 0 ∨ a ≥ 9/8 :=
sorry

end range_of_a_l331_331029


namespace invalid_votes_percentage_l331_331081

def total_votes : ℕ := 560000
def valid_votes_A : ℕ := 357000
def percentage_A : ℝ := 0.75
def invalid_percentage (x : ℝ) : Prop := (percentage_A * (1 - x / 100) * total_votes = valid_votes_A)

theorem invalid_votes_percentage : ∃ x : ℝ, invalid_percentage x ∧ x = 15 :=
by 
  use 15
  unfold invalid_percentage
  sorry

end invalid_votes_percentage_l331_331081


namespace subject_choice_count_l331_331916

def num_subject_choices : ℕ := 18

theorem subject_choice_count :
  let P := {"Physics", "Chemistry", "Biology"}
  let Q := {"Politics", "History", "Geography"}
  ∃ (choices : finset string) (h : choices.card = 3), 
    (∃ s ∈ P, s ∈ choices) ∧ 
    (∃ t ∈ Q, t ∈ choices) ∧ 
    (choices ⊆ (P ∪ Q)) ∧ 
    (finset.card (choices ∩ P) ≥ 1) ∧ 
    (finset.card (choices ∩ Q) ≥ 1) →
  num_subject_choices = 18 := 
by {
  sorry
}

end subject_choice_count_l331_331916


namespace smallest_natural_number_l331_331503

theorem smallest_natural_number :
  ∃ n : ℕ, (n > 0) ∧ (7 * n % 10000 = 2012) ∧ ∀ m : ℕ, (7 * m % 10000 = 2012) → (n ≤ m) :=
sorry

end smallest_natural_number_l331_331503


namespace xy_square_diff_l331_331057

theorem xy_square_diff (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end xy_square_diff_l331_331057


namespace midpoint_divides_equal_ratio_l331_331366

noncomputable def point_divides_segment {A B C F E G : Type} [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq F] [DecidableEq E] [DecidableEq G] : Prop :=
  ∀ (triangle : A × B × C),
  ∀ (F_on_AC : F),
  ∀ (G_mid_AF : G),
  (segment_ratio : (2 / 3)),
  (G_midpoint : G = midpoint (A, F_on_AC)),
  (E_intersection : E = intersection (B, C, G_mid_AF)),
  (E_midpoint : E = midpoint (B, C)),
  divides_ratio (E, B, C) = (1, 1)

theorem midpoint_divides_equal_ratio (A B C F E G : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq F] [DecidableEq E] [DecidableEq G] :
  point_divides_segment (A, B, C) F G :=
by
  sorry

end midpoint_divides_equal_ratio_l331_331366


namespace series_sum_eq_14_div_15_l331_331280

theorem series_sum_eq_14_div_15 : 
  (∑ n in Finset.range 14, (1 : ℚ) / ((n + 1) * (n + 2))) = 14 / 15 := 
by
  sorry

end series_sum_eq_14_div_15_l331_331280


namespace cos_sq_minus_exp_equals_neg_one_fourth_l331_331886

theorem cos_sq_minus_exp_equals_neg_one_fourth :
  (Real.cos (30 * Real.pi / 180))^2 - (2 - Real.pi)^0 = -1 / 4 := by
sorry

end cos_sq_minus_exp_equals_neg_one_fourth_l331_331886


namespace eq_circle_equation_l331_331154

noncomputable def circle_equation (D E F : ℝ) : Polynomial ℝ :=
  Polynomial.C 1 * x^2 + Polynomial.C 1 * y^2 + Polynomial.C D * x + Polynomial.C E * y + Polynomial.C F

def passes_through (p : Polynomial ℝ) (x₀ y₀ : ℝ) : Prop :=
  p.eval₂ Polynomial.C Polynomial.C (x₀, y₀) = 0

theorem eq_circle_equation :
  ∃ D E F, passes_through (circle_equation D E F) 0 0 ∧ passes_through (circle_equation D E F) 1 1 ∧ passes_through (circle_equation D E F) 4 2 ∧
    (D = -8 ∧ E = 6 ∧ F = 0) :=
by
  -- Proof omitted
  sorry

end eq_circle_equation_l331_331154


namespace find_x_l331_331498

theorem find_x (x y k: ℝ) (h1: y * (sqrt x) = k) (h2: y = 8) (h3: ∀ x, ∃ k, y = 1/2 → x = 0.25 → k = 1/4) : x = 1/1024 :=
sorry

end find_x_l331_331498


namespace james_total_spending_l331_331542

noncomputable def total_spent_for_night (entry_fee : ℝ) (num_friends : ℕ) (num_rounds : ℕ) (price_per_cocktail : ℝ) (price_non_alcoholic : ℝ)
  (num_cocktails_james : ℕ) (num_non_alcoholic_james : ℕ) (price_burger : ℝ) (tip_percentage : ℝ) : ℝ :=
let friends_drinks := num_rounds * num_friends * price_per_cocktail in
let james_drinks := (num_cocktails_james * price_per_cocktail) + (num_non_alcoholic_james * price_non_alcoholic) in
let food := price_burger in
let total_before_tip := entry_fee + friends_drinks + james_drinks + food in
let tip := total_before_tip * tip_percentage in
total_before_tip + tip

theorem james_total_spending :
  total_spent_for_night 25 8 3 8 4 6 1 18 0.25 = 358.75 :=
by
  sorry

end james_total_spending_l331_331542


namespace digit_P_value_l331_331810

def is_divisible_by (n m : ℕ) : Prop := m ≠ 0 ∧ n % m = 0

theorem digit_P_value :
  ∃ (P Q R S T : ℕ),
    {P, Q, R, S, T} = {2, 3, 4, 5, 6} ∧
    is_divisible_by (100 * P + 10 * Q + R) 6 ∧
    is_divisible_by (100 * Q + 10 * R + S) 3 ∧
    is_divisible_by (100 * R + 10 * S + T) 9 ∧
    P = 3 :=
by
  sorry

end digit_P_value_l331_331810


namespace part1_part2_l331_331308

open BigOperators

-- Define the sequences and the problem conditions
def a_seq (n : ℕ) : ℝ := 3^(n - 1)

def S_n (n : ℕ) (t : ℝ) : ℝ :=
  t * (a_seq n) - 1 / 2

def b_seq (n : ℕ) : ℝ := Real.log (a_seq (2 * n)) / Real.log 3

def T_n (n : ℕ) : ℝ := ∑ i in Finset.range n, 1 / (b_seq i * b_seq (i + 1))

-- First part of the proof: Finding t and the general formula for a_seq
theorem part1 (t : ℝ) : t = 3 / 2 ∧ (∀ n > 0, a_seq n = 3 ^ (n - 1)) :=
by
  sorry

-- Second part of the proof: Finding the sum T_n
theorem part2 :
  ∀ (n : ℕ), T_n n = n / (2 * n + 1) :=
by
  sorry

end part1_part2_l331_331308


namespace possible_values_of_q_l331_331355

theorem possible_values_of_q {q : ℕ} (hq : q > 0) :
  (∃ k : ℕ, (5 * q + 35) = k * (3 * q - 7) ∧ k > 0) ↔
  q = 3 ∨ q = 4 ∨ q = 5 ∨ q = 7 ∨ q = 9 ∨ q = 15 ∨ q = 21 ∨ q = 31 :=
by
  sorry

end possible_values_of_q_l331_331355


namespace madeline_groceries_l331_331115

theorem madeline_groceries :
  let rent := 1200
  let medical := 200
  let utilities := 60
  let emergency := 200
  let wage_per_hour := 15
  let hours_worked := 138
  let total_expenses_without_groceries := rent + medical + utilities + emergency
  let total_income := wage_per_hour * hours_worked
  let money_left_for_groceries := total_income - total_expenses_without_groceries
  money_left_for_groceries = 410 :=
by
  let rent := 1200
  let medical := 200
  let utilities := 60
  let emergency := 200
  let wage_per_hour := 15
  let hours_worked := 138
  let total_expenses_without_groceries := rent + medical + utilities + emergency
  let total_income := wage_per_hour * hours_worked
  let money_left_for_groceries := total_income - total_expenses_without_groceries
  show money_left_for_groceries = 410 from sorry

end madeline_groceries_l331_331115


namespace probability_no_adjacent_birch_l331_331228

theorem probability_no_adjacent_birch:
  let total_trees := 12
  let maple_trees := 3
  let oak_trees := 4
  let birch_trees := 5

  let total_arrangements := Nat.factorial total_trees
  
  -- Arrangements of maples and oaks
  let non_birch_trees := maple_trees + oak_trees
  let arrangements_without_birch := Nat.factorial non_birch_trees
  
  -- Possible slots to place birch trees
  let slots_for_birch := non_birch_trees + 1
  let ways_to_choose_slots := Nat.choose slots_for_birch birch_trees

  -- Permutation of birch trees
  let permutation_of_birch := Nat.factorial birch_trees

  -- Favorable arrangements
  let favorable_arrangements := arrangements_without_birch * ways_to_choose_slots * permutation_of_birch

  -- Probability
  let probability := favorable_arrangements / total_arrangements
  probability = 7 / 99 :=
by {
  sorry
}

end probability_no_adjacent_birch_l331_331228


namespace probability_of_both_questions_chosen_l331_331697

def student_choices : Type := {p : Prop // p = "Each of the two students chooses one question from question 22 and question 23"}

def total_outcomes : ℕ := 4
def favorable_outcomes : ℕ := 2

theorem probability_of_both_questions_chosen (c : student_choices) : 
  favorable_outcomes / total_outcomes = 1 / 2 :=
by
  sorry

end probability_of_both_questions_chosen_l331_331697


namespace deposit_percentage_l331_331453

-- Define the conditions of the problem
def amount_deposited : ℕ := 5000
def monthly_income : ℕ := 25000

-- Define the percentage deposited formula
def percentage_deposited (amount_deposited monthly_income : ℕ) : ℚ :=
  (amount_deposited / monthly_income) * 100

-- State the theorem to be proved
theorem deposit_percentage :
  percentage_deposited amount_deposited monthly_income = 20 := by
  sorry

end deposit_percentage_l331_331453


namespace correct_operation_is_B_l331_331204

-- Definitions of the operations as conditions
def operation_A (x : ℝ) : Prop := 3 * x - x = 3
def operation_B (x : ℝ) : Prop := x^2 * x^3 = x^5
def operation_C (x : ℝ) : Prop := x^6 / x^2 = x^3
def operation_D (x : ℝ) : Prop := (x^2)^3 = x^5

-- Prove that the correct operation is B
theorem correct_operation_is_B (x : ℝ) : operation_B x :=
by
  show x^2 * x^3 = x^5
  sorry

end correct_operation_is_B_l331_331204


namespace determine_z_l331_331160

theorem determine_z (z : ℕ) (h1: z.factors.count = 18) (h2: 16 ∣ z) (h3: 18 ∣ z) : z = 288 := 
  by 
  sorry

end determine_z_l331_331160


namespace inequality_proof_l331_331880

theorem inequality_proof (a b c : ℝ) (hp : 0 < a ∧ 0 < b ∧ 0 < c) (hd : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
    (bc / a + ac / b + ab / c > a + b + c) :=
by
  sorry

end inequality_proof_l331_331880


namespace range_of_x_l331_331693

theorem range_of_x (a : ℝ) (x : ℝ) (h_a : 1 ≤ a ∧ a ≤ 3) (h : a * x^2 + (a - 2) * x - 2 > 0) :
  x < -1 ∨ x > 2 / 3 :=
sorry

end range_of_x_l331_331693


namespace xy_square_diff_l331_331054

theorem xy_square_diff (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end xy_square_diff_l331_331054


namespace proof_problem_l331_331316

variables (α : ℝ)

-- Condition: tan(α) = 2
def tan_condition : Prop := Real.tan α = 2

-- First expression: (sin α + 2 cos α) / (4 cos α - sin α) = 2
def expression1 : Prop := (Real.sin α + 2 * Real.cos α) / (4 * Real.cos α - Real.sin α) = 2

-- Second expression: sqrt(2) * sin(2α + π/4) + 1 = 6/5
def expression2 : Prop := Real.sqrt 2 * Real.sin (2 * α + Real.pi / 4) + 1 = 6 / 5

-- Theorem: Prove the expressions given the condition
theorem proof_problem :
  tan_condition α → expression1 α ∧ expression2 α :=
by
  intro tan_cond
  have h1 : expression1 α := sorry
  have h2 : expression2 α := sorry
  exact ⟨h1, h2⟩

end proof_problem_l331_331316


namespace find_baking_soda_boxes_l331_331222

-- Define the quantities and costs
def num_flour_boxes := 3
def cost_per_flour_box := 3
def num_egg_trays := 3
def cost_per_egg_tray := 10
def num_milk_liters := 7
def cost_per_milk_liter := 5
def baking_soda_cost_per_box := 3
def total_cost := 80

-- Define the total cost of flour, eggs, and milk
def total_flour_cost := num_flour_boxes * cost_per_flour_box
def total_egg_cost := num_egg_trays * cost_per_egg_tray
def total_milk_cost := num_milk_liters * cost_per_milk_liter

-- Define the total cost of non-baking soda items
def total_non_baking_soda_cost := total_flour_cost + total_egg_cost + total_milk_cost

-- Define the remaining cost for baking soda
def baking_soda_total_cost := total_cost - total_non_baking_soda_cost

-- Define the number of baking soda boxes
def num_baking_soda_boxes := baking_soda_total_cost / baking_soda_cost_per_box

theorem find_baking_soda_boxes : num_baking_soda_boxes = 2 :=
by
  sorry

end find_baking_soda_boxes_l331_331222


namespace simplify_expression_l331_331858

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : 
  (15 * x^2) * (6 * x) * (1 / (3 * x)^2) = 10 * x := 
by
  sorry

end simplify_expression_l331_331858


namespace ratio_second_to_first_l331_331501

theorem ratio_second_to_first (F S T : ℕ) 
  (hT : T = 2 * F)
  (havg : (F + S + T) / 3 = 77)
  (hmin : F = 33) :
  S / F = 4 :=
by
  sorry

end ratio_second_to_first_l331_331501


namespace inner_cube_surface_area_8_l331_331911

section CubeInSphere

variable (cube_surface_area : ℕ) (cube_side : Real) (sphere_diameter : Real) (inner_cube_diagonal : Real) (inner_cube_side : Real)

-- Define the conditions for the sequence of the cubes and sphere.
def cube.has_surface_area (c : cube_surface_area) : Prop := c = 24
def cube.side_length (c : cube_surface_area) (s : cube_side) : Prop := s^2 * 6 = c
def sphere.is_inscribed_in_cube (s : cube_side) (d : sphere_diameter) : Prop := d = s
def inner_cube.is_inscribed_in_sphere (d : sphere_diameter) (i : inner_cube_diagonal) : Prop := i = d
def inner_cube.diagonal_is_side_sqrt_3 (i : inner_cube_diagonal) (inner_side : inner_cube_side) : Prop := inner_side^2 + inner_side^2 + inner_side^2 = i^2
def inner_cube.surface_area (inner_side : inner_cube_side) : Real := 6 * inner_side^2

theorem inner_cube_surface_area_8:
  cube.has_surface_area cube_surface_area →
  cube.side_length cube_surface_area cube_side →
  sphere.is_inscribed_in_cube cube_side sphere_diameter →
  inner_cube.is_inscribed_in_sphere sphere_diameter inner_cube_diagonal →
  inner_cube.diagonal_is_side_sqrt_3 inner_cube_diagonal inner_cube_side →
  inner_cube.surface_area inner_cube_side = 8 := by
sorry

end CubeInSphere

end inner_cube_surface_area_8_l331_331911


namespace circle_center_radius_sum_l331_331594

noncomputable def circle_equation (x y : ℝ) : Prop :=
  (x + 4) ^ 2 + (y + 7) ^ 2 = 0

theorem circle_center_radius_sum :
  let a := -4
  let b := -7
  let r := 0
  a + b + r = -11 :=
by
  intro a b r
  simp only
  exact Eq.refl (-4 + (-7) + 0)

end circle_center_radius_sum_l331_331594


namespace count_real_solutions_l331_331683

theorem count_real_solutions : 
  (card {x : ℝ | x = (⌊x / 2⌋ : ℝ) + (⌊x / 3⌋ : ℝ) + (⌊x / 5⌋ : ℝ)} = 30) := 
sorry

end count_real_solutions_l331_331683


namespace correct_population_statement_l331_331188

def correct_statement :=
  "The mathematics scores of all candidates in the city's high school entrance examination last year constitute the population."

def sample_size : ℕ := 500

def is_correct (statement : String) : Prop :=
  statement = correct_statement

theorem correct_population_statement (scores : Fin 500 → ℝ) :
  is_correct "The mathematics scores of all candidates in the city's high school entrance examination last year constitute the population." :=
by
  sorry

end correct_population_statement_l331_331188


namespace find_focus_of_hyperbola_l331_331824

def hyperbola_foci :=
  ∃ f : ℝ × ℝ, f = (-1, -2 + (2 * Real.sqrt 3) / 3) ∧
  (let (x, y) := f in (3 * x^2 - y^2 + 6 * x - 4 * y + 8 = 0))

theorem find_focus_of_hyperbola :
  hyperbola_foci :=
sorry

end find_focus_of_hyperbola_l331_331824


namespace projectile_reaches_100_feet_l331_331814

noncomputable def projectile_height (t : ℝ) : ℝ :=
  -16 * t^2 + 80 * t

theorem projectile_reaches_100_feet :
  ∃ t : ℝ, t = 2.5 ∧ projectile_height t = 100 :=
by
  use 2.5
  sorry

end projectile_reaches_100_feet_l331_331814


namespace part1_part2_l331_331676
noncomputable theory

-- Part (1)
def vector_a1 (m : ℝ) : ℝ × ℝ := (1, m)
def vector_b1 (n : ℝ) : ℝ × ℝ := (2, n)
def orthogonal_condition (a b : ℝ × ℝ) (λ : ℝ) : Prop :=
  let a' := (a.1 + λ * b.1, a.2 + λ * b.2)
  in a.1 * a'.1 + a.2 * a'.2 = 0

theorem part1 (λ : ℝ) (h1 : λ = 10) : 
  let m := 3 in
  let n := -1 in
  let a := vector_a1 m in
  let b := vector_b1 n in
  orthogonal_condition a b λ :=
sorry

-- Part (2)
def vector_a2 (m : ℝ) : ℝ × ℝ := (1, m)
def vector_b2 (n : ℝ) : ℝ × ℝ := (2, n)
def angle_condition (a b : ℝ × ℝ) (θ : ℝ) : Prop :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let norm_a := real.sqrt (a.1 * a.1 + a.2 * a.2)
  let norm_b := real.sqrt (b.1 * b.1 + b.2 * b.2)
  in dot_product = norm_a * norm_b * real.cos θ

theorem part2 (n : ℝ) (h2 : n = 0) : 
  let m := 1 in
  let b := vector_b2 n in
  let a := vector_a2 m in
  angle_condition a b (real.pi / 4) :=
sorry

end part1_part2_l331_331676


namespace smallest_non_multiple_of_four_abundant_l331_331575

def is_proper_divisor (n : ℕ) (d : ℕ) : Prop :=
d ∣ n ∧ d < n

def sum_proper_divisors (n : ℕ) : ℕ :=
(n.divisors.filter (λ d, is_proper_divisor n d)).sum

def is_abundant_number (n : ℕ) : Prop :=
sum_proper_divisors n > n

def is_not_multiple_of_four (n : ℕ) : Prop :=
¬ 4 ∣ n

theorem smallest_non_multiple_of_four_abundant :
  ∃ n : ℕ, is_abundant_number n ∧ is_not_multiple_of_four n ∧
  ∀ m : ℕ, m < n → is_abundant_number m → ¬ is_not_multiple_of_four m :=
begin
  use 18,
  split,
  { -- Prove 18 is abundant
    sorry
  },
  split,
  { -- Prove 18 is not a multiple of 4
    sorry
  },
  { -- Prove 18 is the smallest one
    sorry
  }
end

end smallest_non_multiple_of_four_abundant_l331_331575


namespace hyperbola_focal_length_l331_331485

-- Define the hyperbola and the condition
def hyperbola_property (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 / (m^2 + 12) - y^2 / (4 - m^2) = 1) → (4 - m^2 > 0)

-- State the focal length problem
theorem hyperbola_focal_length (m : ℝ) (h : 4 - m^2 > 0) :
  let a := Real.sqrt (m^2 + 12),
      b := Real.sqrt (4 - m^2),
      c := Real.sqrt (a^2 + b^2) in
  2 * c = 8 :=
by
  sorry

end hyperbola_focal_length_l331_331485


namespace total_money_is_300_l331_331183

-- Definitions of initial amounts
variables {a b c : ℝ}

-- Conditions
def initial_conditions :=
  c = 50

def after_amys_redistribution := 
  let new_c := 2 * c,
      new_b := 2 * b,
      new_a := a - (c + b)
  in (new_c, new_b, new_a)

def after_bobs_redistribution (new_c new_b new_a: ℝ) := 
  let newer_c := 2 * new_c,
      newer_a := 2 * new_a,
      newer_b := new_b - (new_a + new_c)
  in (newer_c, newer_a, newer_b)

def after_cals_redistribution (newer_c newer_a newer_b: ℝ) := 
  let final_a := 2 * newer_a,
      final_b := 2 * newer_b,
      final_c := newer_c - (newer_a + newer_b)
  in (final_a, final_b, final_c)

def final_condition (newer_c newer_a newer_b: ℝ) := 
  let (_, _, final_c) := after_cals_redistribution newer_c newer_a newer_b
  in final_c = 100

-- Proof statement
theorem total_money_is_300 : 
  ∀ a b c : ℝ, initial_conditions → 
  (let (new_c, new_b, new_a) := after_amys_redistribution in
  let (newer_c, newer_a, newer_b) := after_bobs_redistribution new_c new_b new_a in
  final_condition newer_c newer_a newer_b) → 
  (a + b + c) = 300 := 
sorry

end total_money_is_300_l331_331183


namespace Karlson_max_candies_l331_331500

theorem Karlson_max_candies (f : Fin 25 → ℕ) (g : Fin 25 → Fin 25 → ℕ) :
  (∀ i, f i = 1) →
  (∀ i j, g i j = f i * f j) →
  (∃ (S : ℕ), S = 300) :=
by
  intros h1 h2
  sorry

end Karlson_max_candies_l331_331500


namespace total_artworks_l331_331435

theorem total_artworks (students : ℕ) (group1_artworks : ℕ) (group2_artworks : ℕ) (total_students : students = 10) 
    (artwork_group1 : group1_artworks = 5 * 3) (artwork_group2 : group2_artworks = 5 * 4) : 
    group1_artworks + group2_artworks = 35 :=
by
  sorry

end total_artworks_l331_331435


namespace rectangular_cards_are_squares_l331_331902

theorem rectangular_cards_are_squares
  (n : ℕ) (h : n > 1)
  (k l : ℕ) (hlk : l < k)
  (a : Fin n → ℕ) (b : Fin n → ℕ) (c : Fin n → ℕ)
  (total_squares_prime : ∀ i, l = (b i) * (a i) ∧ k = (c i) * (a i))
  (A : ℕ) (hA : A = ∑ i, (b i) * (c i)) (prime_A : Nat.Prime A) : k = l :=
by
  sorry

end rectangular_cards_are_squares_l331_331902


namespace angle_PEQ_is_180_l331_331449

theorem angle_PEQ_is_180 (A B C D : Point) (omega omega1 omega2 : Circle)
  (h1 : InscribedQuads A B C D omega)
  (h2 : CenterOnSide omega A B)
  (h3 : ExternallyTangent omega omega1 C)
  (h4 : TangentAtPoints omega omega2 D omega1 E)
  (h5 : IntersectAgainLine BC omega1 P)
  (h6 : IntersectAgainLine AD omega2 Q)
  (h7 : DistinctPoints P Q E) :
  angle P E Q = 180 :=
sorry

end angle_PEQ_is_180_l331_331449


namespace max_police_officers_needed_l331_331120

theorem max_police_officers_needed : 
  let streets := 10
  let non_parallel := true
  let curved_streets := 2
  let additional_intersections_per_curved := 3 
  streets = 10 ∧ 
  non_parallel = true ∧ 
  curved_streets = 2 ∧ 
  additional_intersections_per_curved = 3 → 
  ( (streets * (streets - 1) / 2) + (curved_streets * additional_intersections_per_curved) ) = 51 :=
by
  intros
  sorry

end max_police_officers_needed_l331_331120


namespace problem1_problem2_l331_331644

-- Define the conditions as noncomputable definitions
noncomputable def A : Real := sorry
noncomputable def tan_A : Real := 2
noncomputable def sin_A_plus_cos_A : Real := 1 / 5

-- Define the trigonometric identities
noncomputable def sin (x : Real) : Real := sorry
noncomputable def cos (x : Real) : Real := sorry
noncomputable def tan (x : Real) : Real := sin x / cos x

-- Ensure the conditions
axiom tan_A_condition : tan A = tan_A
axiom sin_A_plus_cos_A_condition : sin A + cos A = sin_A_plus_cos_A

-- Proof problem 1:
theorem problem1 : 
  (sin (π - A) + cos (-A)) / (sin A - sin (π / 2 + A)) = 3 := by
  sorry

-- Proof problem 2:
theorem problem2 : 
  sin A - cos A = 7 / 5 := by
  sorry

end problem1_problem2_l331_331644


namespace magic_square_y_value_l331_331074

theorem magic_square_y_value
  (a b c d e : ℤ)
  (h1 : y + 17 + 124 = 9 + a + b)
  (h2 : y + 9 + c = 17 + a + d)
  (h3 : 124 + b + e = 9 + a + b)
  (h4 : y + 9 + c = 124 + d)
  (h5 : sum of each row, column, and diagonal is same constant K)
   : y = 0 := by
  sorry

end magic_square_y_value_l331_331074


namespace problem1_problem2_l331_331032

noncomputable def vector_a (theta : ℝ) := (Real.cos (3 * theta / 2), Real.sin (3 * theta / 2))
noncomputable def vector_b (theta : ℝ) := (Real.cos (theta / 2), -Real.sin (theta / 2))

theorem problem1 (theta : ℝ) (hθ : 0 ≤ theta ∧ theta ≤ Real.pi / 3) : 
  let a := vector_a theta 
  let b := vector_b theta in
  let dot_product := a.1 * b.1 + a.2 * b.2 in
  let magnitude_sum := Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) in
  let expr := dot_product / magnitude_sum in
  -1/2 ≤ expr ∧ expr ≤ 1/2 := 
  sorry

theorem problem2 (theta : ℝ) (k : ℝ) (hθ : 0 ≤ theta ∧ theta ≤ Real.pi / 3) :
  let a := vector_a theta 
  let b := vector_b theta in
  (Real.sqrt ((k * a.1 + b.1)^2 + (k * a.2 + b.2)^2) = Real.sqrt 3 * Real.sqrt ((a.1 - k * b.1)^2 + (a.2 - k * b.2)^2)) → 
  (k = -1 ∨ (2 - Real.sqrt 3 ≤ k ∧ k ≤ 2 + Real.sqrt 3)) :=
  sorry

end problem1_problem2_l331_331032


namespace navi_wins_l331_331216

-- Function definition and game conditions
def f : ℕ → ℕ
| 0 := 2
| k := if ∃ x < 2016, f x = 2 ∨ (∃ y, k = 2014 * y ∧ f (k / 2014) = 2) then 1 else 2

theorem navi_wins (a b : ℕ) (h : a ≥ 2015) :
  (∀ n : ℕ, f (an + b) ≠ 2) →
  ∃ S : set ℕ, (∀ n ∈ S, f n = 2) ∧ S.card = 2013 :=
sorry

end navi_wins_l331_331216


namespace inverse_f_4_eq_4_div_79_l331_331663

def f (x : ℝ) : ℝ := Real.logb 3 ((4 / x) + 2)

theorem inverse_f_4_eq_4_div_79 : (∃ x, f x = 4) ∧ (f⁻¹ 4 = 4 / 79) := 
sorry

end inverse_f_4_eq_4_div_79_l331_331663


namespace correct_average_is_51_l331_331149

variable (sum_correct sum_incorrect : ℝ)
variable (n : ℕ)
variable (avg_incorrect avg_correct : ℝ)

-- Defining the initial average condition
def avg_incorrect := 46

-- Defining the number of elements
def n := 10

-- Defining the correct number which was mistaken
def mistaken_value := 25
def correct_value := 75

-- Calculating the incorrect sum using given average
def sum_incorrect := avg_incorrect * n

-- Adjusting the incorrect sum by the difference between the correct and mistaken value
def diff := correct_value - mistaken_value
def sum_correct := sum_incorrect + diff

-- Calculating the correct average using correct sum
def avg_correct := sum_correct / n

-- The theorem to prove
theorem correct_average_is_51 : avg_correct = 51 :=
by
  sorry

end correct_average_is_51_l331_331149


namespace selling_price_l331_331550

-- Definitions
variable (cp : ℝ) (gp : ℝ)

-- Given conditions as Lean definitions
def cost_price := 10
def gain_percent := 1.5 / 1

-- The statement we need to prove
theorem selling_price (h1 : cp = cost_price) (h2 : gp = gain_percent) : cp + (gp * cp) = 25 :=
by
  -- Proof content skipped
  sorry

end selling_price_l331_331550


namespace age_difference_is_16_l331_331474

variable (y : ℕ)  -- the present age of the younger person

-- Conditions
def elder_age_now : ℕ := 30
def elder_age_6_years_ago := elder_age_now - 6
def younger_age_6_years_ago := y - 6
def condition := elder_age_6_years_ago = 3 * younger_age_6_years_ago

-- Theorem to prove the difference in ages is 16 years
theorem age_difference_is_16 (h : condition y) : elder_age_now - y = 16 :=
by
  sorry

end age_difference_is_16_l331_331474


namespace pieces_eaten_first_night_l331_331976

-- Define the initial numbers of candies
def debby_candies : Nat := 32
def sister_candies : Nat := 42
def candies_left : Nat := 39

-- Calculate the initial total number of candies
def initial_total_candies : Nat := debby_candies + sister_candies

-- Define the number of candies eaten the first night
def candies_eaten : Nat := initial_total_candies - candies_left

-- The problem statement with the proof goal
theorem pieces_eaten_first_night : candies_eaten = 35 := by
  sorry

end pieces_eaten_first_night_l331_331976


namespace relationship_between_lifts_l331_331374

-- Define the conditions
def total_weight : ℝ := 900
def first_lift : ℝ := 400
def second_lift (total_weight first_lift : ℝ) : ℝ := total_weight - first_lift

-- Prove the relationship
theorem relationship_between_lifts (total_weight: ℝ) (first_lift: ℝ) (h1: total_weight = 900) (h2: first_lift = 400) : 
  second_lift total_weight first_lift = 500 :=
by
  rw [h1, h2]
  exact rfl

end relationship_between_lifts_l331_331374


namespace mutually_exclusive_not_opposite_l331_331954

-- Define the number of balls
def red_balls := 5
def black_balls := 2
def total_balls := red_balls + black_balls

-- Define the conditions for the events
def event_exactly_one_black_ball (drawn: Fin 3 → Fin total_balls) : Prop :=
  ∃(i: Fin 3), drawn i < 7 ∧ 5 ≤ drawn i -- Assumes 5 to 6 are black balls

def event_exactly_two_black_balls (drawn: Fin 3 → Fin total_balls) : Prop :=
  ∃(i j: Fin 3), i ≠ j ∧ drawn i < 7 ∧ drawn j < 7 ∧ 5 ≤ drawn i ∧ 5 ≤ drawn j -- Assumes 5 to 6 are black balls

theorem mutually_exclusive_not_opposite : 
  ∀ (drawn: Fin 3 → Fin total_balls),
  (event_exactly_one_black_ball drawn ∧ ¬event_exactly_two_black_balls drawn) ∨ 
  (¬event_exactly_one_black_ball drawn ∧ event_exactly_two_black_balls drawn) ∧ 
  ¬(event_exactly_one_black_ball drawn ∨ event_exactly_two_black_balls drawn) :=
by 
  sorry

end mutually_exclusive_not_opposite_l331_331954


namespace watermelons_eaten_l331_331452

theorem watermelons_eaten (original left : ℕ) (h1 : original = 4) (h2 : left = 1) :
  original - left = 3 :=
by {
  -- Providing the proof steps is not necessary as per the instructions
  sorry
}

end watermelons_eaten_l331_331452


namespace probability_of_dime_l331_331548

theorem probability_of_dime (quarters_value dimes_value pennies_value : ℝ)
  (value_quarter value_dime value_penny : ℝ)
  (num_quarters num_dimes num_pennies total_coins : ℕ) :
  quarters_value = 12.50 → dimes_value = 5.00 → pennies_value = 2.50 →
  value_quarter = 0.25 → value_dime = 0.10 → value_penny = 0.01 →
  num_quarters = (quarters_value / value_quarter).to_nat →
  num_dimes = (dimes_value / value_dime).to_nat →
  num_pennies = (pennies_value / value_penny).to_nat →
  total_coins = num_quarters + num_dimes + num_pennies →
  (num_dimes : ℝ) / (total_coins : ℝ) = 1 / 7 :=
by
  intros
  sorry

end probability_of_dime_l331_331548


namespace find_number_of_white_balls_l331_331703

theorem find_number_of_white_balls (n : ℕ) (h : 6 / (6 + n) = 2 / 5) : n = 9 :=
sorry

end find_number_of_white_balls_l331_331703


namespace coefficient_x2_term_is_20_l331_331710

noncomputable def coefficient_x2_term : ℕ :=
  let expansion := (1 + x) ^ 8 * (1 - x)
  in expansion.coeff(2)

theorem coefficient_x2_term_is_20 : coefficient_x2_term = 20 :=
sorry

end coefficient_x2_term_is_20_l331_331710


namespace rationalize_denominator_correct_l331_331771

noncomputable def rationalize_denominator_sum : ℕ :=
  let a := real.root (5 : ℝ) 3;
  let b := real.root (3 : ℝ) 3;
  let A := real.root (25 : ℝ) 3;
  let B := real.root (15 : ℝ) 3;
  let C := real.root (9 : ℝ) 3;
  let D := 2;
  (25 + 15 + 9 + 2)

theorem rationalize_denominator_correct :
  rationalize_denominator_sum = 51 :=
  by sorry

end rationalize_denominator_correct_l331_331771


namespace length_XY_l331_331549

-- Define the key points and lengths from the problem
variables {A B D X Y : Type}
variables [has_dist A X] [has_dist A Y] [has_dist B X] [has_dist D Y]

-- Given conditions as per the problem
variables (BX DY AB BC : ℝ)
variables (AB_dist_A_X : dist A X = BX / 2)
variables (AB_dist_B_X : dist B X = BX)
variables (DY_dist_A_Y : dist A Y = 2 * BX)
variables (DY_dist_D_Y : dist D Y = DY)
variables (BC_eq_2_AB : BC = 2 * AB)

-- The main theorem to prove
theorem length_XY : dist X Y = 13 :=
by
  -- Add main proof logic here (skipped with sorry for now)
  sorry

end length_XY_l331_331549


namespace distinct_intersection_points_l331_331617

-- Define the setup
constant num_lines : Nat
constant lines: Fin num_lines → Prop

-- Conditions
axiom h_num_lines : num_lines = 5
axiom h_distinct_lines : ∀ (i j: Fin num_lines), i ≠ j → ∃ p, (lines i) ∧ (lines j)
axiom h_no_three_concurrent : ∀ (i j k : Fin num_lines), i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬ ∃ p, (lines i) ∧ (lines j) ∧ (lines k)

-- Theorem to prove
theorem distinct_intersection_points : ∃ n, n = 10 :=
by { 
    -- We'll provide the proof here
    sorry
}

end distinct_intersection_points_l331_331617


namespace number_of_possible_ns_count_possible_ns_l331_331828

theorem number_of_possible_ns : ∃ (n : ℕ), 3 ≤ n ∧ n ≤ 3600 ∧ (log 2 (40 : ℝ)) + (log 2 (90 : ℝ)) > (log 2 n) ∧ (log 2 (90 : ℝ)) + (log 2 n) > (log 2 (40 : ℝ)) ∧ (log 2 (40 : ℝ)) + (log 2 n) > (log 2 (90 : ℝ)) :=
by { sorry }

theorem count_possible_ns : (number_of_possible_ns).some ≤ 3600 - 3 + 1 :=
by { sorry }

end number_of_possible_ns_count_possible_ns_l331_331828


namespace solve_log_eq_l331_331783

theorem solve_log_eq (x : ℝ) (h : log 2 (2 - x) + log 2 (3 - x) = log 2 12) (cond : x < 2) : x = -1 :=
sorry

end solve_log_eq_l331_331783


namespace solve_for_y_l331_331794

theorem solve_for_y :
  (∀ y : ℝ, (1 / 8) ^ (3 * y + 12) = 32 ^ (3 * y + 7)) →
  y = -71 / 24 :=
begin
  sorry
end

end solve_for_y_l331_331794


namespace euler_distance_formula_l331_331424

theorem euler_distance_formula 
  (d R r : ℝ) 
  (h₁ : d = distance_between_centers_of_inscribed_and_circumscribed_circles_of_triangle)
  (h₂ : R = circumradius_of_triangle)
  (h₃ : r = inradius_of_triangle) : 
  d^2 = R^2 - 2 * R * r := 
sorry

end euler_distance_formula_l331_331424


namespace number_of_terms_added_l331_331191

theorem number_of_terms_added (k : ℕ) (h : k > 1) : 
  (∑ i in (range ((2^(k+1))-1)).filter (λ x, x ≥ ((2^k)-1)), 1 / (i + 1)) = 2^k :=
by
  -- Proof goes here
  sorry

end number_of_terms_added_l331_331191


namespace transport_b_speed_l331_331850

theorem transport_b_speed :
  ∀ (v : ℝ),
    let speed_a := 60 in
    let time := 2.71875 in
    let distance := 348 in
    (speed_a + v) * time = distance → v = 68 :=
by
  intros v speed_a time distance h
  sorry

end transport_b_speed_l331_331850


namespace distinct_ordered_pair_count_l331_331325

theorem distinct_ordered_pair_count (x y : ℕ) (h1 : x + y = 50) (h2 : 1 ≤ x) (h3 : 1 ≤ y) : 
  ∃! (x y : ℕ), x + y = 50 ∧ 1 ≤ x ∧ 1 ≤ y :=
by
  sorry

end distinct_ordered_pair_count_l331_331325


namespace coprime_fraction_l331_331155

theorem coprime_fraction :
  let a := 2022
  let expr := (2023 : ℚ) / 2022 - 2022 / 2023
  ∃ (p q : ℕ), expr = p / q ∧ Nat.coprime p q ∧ p = 4045 := by sorry

end coprime_fraction_l331_331155


namespace sum_in_base3_l331_331161

def base3_repr (n : ℕ) : list ℕ :=
  if n = 0 then [0]
  else
    let rec aux (n : ℕ) (acc : list ℕ) : list ℕ :=
      if n = 0 then acc
      else aux (n / 3) (n % 3 :: acc)
    in aux n []

theorem sum_in_base3 :
  base3_repr (243 + 81) = [1, 1, 0, 0, 0, 0] :=
by sorry

end sum_in_base3_l331_331161


namespace compound_interest_principal_l331_331195

theorem compound_interest_principal 
  (CI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (hCI : CI = 315)
  (hR : R = 10)
  (hT : T = 2) :
  CI = P * ((1 + R / 100)^T - 1) → P = 1500 := by
  sorry

end compound_interest_principal_l331_331195


namespace mathematicians_ages_l331_331570

theorem mathematicians_ages (a b c d : ℕ) (A B C : ℕ) :
  (a * b + (a + 2) = (a + 4) * (a + 6)) → 
  (c * d + (c + 1) = (c + 2) * (c + 3)) →
  (C < A) →
  (C < B) →
  (A = 48) →
  (B = 56) →
  (C = 35) →
  B is_absent_minded :=
sorry

end mathematicians_ages_l331_331570


namespace coeff_x3_in_x_mul_one_plus_2x_pow_6_l331_331269

theorem coeff_x3_in_x_mul_one_plus_2x_pow_6 :
  let expansion := (x : ℕ) * (1 + 2 * x) ^ 6,
      coeff := coefficient expansion 3
  in coeff = 60 :=
by
  sorry

end coeff_x3_in_x_mul_one_plus_2x_pow_6_l331_331269


namespace intersection_points_count_l331_331266

theorem intersection_points_count :
  (∑ z in {3, 4, 5, 6, 7, 8}, 
    ∑ xy in { ⟨x, y⟩ | x^2 + y^2 ≤ (if z = 3 then 0
                                     else if z = 4 then 15
                                     else if z = 5 then 27
                                     else if z = 6 then 20
                                     else if z = 7 then 11
                                     else if z = 8 then 0
                                     else 0) }
              and x ∈ {-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6} 
              and y ∈ {-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6} }, 
  1) = 37 := by
  sorry

end intersection_points_count_l331_331266


namespace infinite_sum_evaluation_l331_331956

theorem infinite_sum_evaluation :
  (∑' n : ℕ, (n : ℚ) / ((n^2 - 2 * n + 2) * (n^2 + 2 * n + 4))) = 5 / 24 :=
sorry

end infinite_sum_evaluation_l331_331956


namespace g_sqrt_45_l331_331111

def g (x : ℝ) : ℝ :=
  if x ∈ set.Ico ⌊x⌋ (⌊x⌋ + 1) then ⌊x⌋ + 6 else sorry -- use the floor condition

theorem g_sqrt_45 : g (real.sqrt 45) = 12 :=
by 
  have h1 : ¬ (real.sqrt 45).is_integer := sorry -- root of non-perfect square is not an integer
  have h2 : ⌊real.sqrt 45⌋ = 6 := sorry --floor calculation
  sorry -- step to final conclusion

end g_sqrt_45_l331_331111


namespace circle_equation_value_l331_331060

theorem circle_equation_value (a : ℝ) :
  (∀ x y : ℝ, x^2 + (a + 2) * y^2 + 2 * a * x + a = 0 → False) → a = -1 :=
by
  intros h
  sorry

end circle_equation_value_l331_331060


namespace chessboard_max_label_sum_l331_331838

noncomputable def maxLabelSum := 8 / 12.5

theorem chessboard_max_label_sum : 
  let label (i j : ℕ) := 1 / (2 * i + j - 1)
  let chosen_squares (r : ℕ → ℕ) := ∑ i in Finset.range 8, label (r i) (i + 1)
  maxLabelSum = ∑ i in Finset.range 8, label (i + 1) (i + 1) := 
sorry

end chessboard_max_label_sum_l331_331838


namespace cos_sqr_sub_power_zero_l331_331889

theorem cos_sqr_sub_power_zero :
  (cos (30 * Real.pi / 180))^2 - (2 - Real.pi)^0 = -1/4 :=
by
  sorry

end cos_sqr_sub_power_zero_l331_331889


namespace traveler_never_returns_home_l331_331567

variable (City : Type)
variable (Distance : City → City → ℝ)

variables (A B C : City)
variables (C_i C_i_plus_one C_i_minus_one : City)

-- Given conditions
axiom travel_far_from_A : ∀ (C : City), C ≠ B → Distance A B > Distance A C
axiom travel_far_from_B : ∀ (D : City), D ≠ C → Distance B C > Distance B D
axiom increasing_distance : ∀ i : ℕ, Distance C_i C_i_plus_one > Distance C_i_minus_one C_i

-- Given condition that C is not A
axiom C_not_eq_A : C ≠ A

-- Proof statement
theorem traveler_never_returns_home : ∀ i : ℕ, C_i ≠ A := sorry

end traveler_never_returns_home_l331_331567


namespace problem_statement_l331_331333

open Real

-- Problem definition
def f (x a : ℝ) : ℝ := exp (x^3 - a * x)
def g (x a : ℝ) : ℝ := f x a - x^2

-- Statement of the proof
theorem problem_statement (a : ℝ) :
  (∀ x, (a ≤ 0 → deriv (f x a) ≥ 0) ∧
    (a > 0 → (∀ x, x < -sqrt (a / 3) ∨ x > sqrt (a / 3) → deriv (f x a) > 0) ∧ 
             (∀ x, -sqrt (a / 3) < x ∧ x < sqrt (a / 3) → deriv (f x a) < 0))) ∧
  (∃ m : ℝ, g m a = 0 ∧ g (-m) a = 0 → a ∈ Ioi 1) :=
sorry

end problem_statement_l331_331333


namespace factor_expression_l331_331217

variable (x y : ℝ)

theorem factor_expression :
  4 * x ^ 2 - 4 * x - y ^ 2 + 4 * y - 3 = (2 * x + y - 3) * (2 * x - y + 1) := by
  sorry

end factor_expression_l331_331217


namespace part1_part2a_part2b_l331_331660

-- Condition for part (1)
def function_f (x a b : ℝ) : ℝ :=
  x * |x - a| + b * x

theorem part1 (a b : ℝ) :
  (b = -1) ∧ (function_f (a + 1) a b = 0 ∧ function_f (a - 1) a b = 0 ∧ function_f 0 a b = 0) 
  → (a = 1 ∨ a = -1) :=
by
  sorry

-- Condition for part (2a)
theorem part2a (a : ℝ) :
  (∀ (x : ℝ), 1 ≤ x ∧ x ≤ 3 → function_f x a 1 / x ≤ 2 * sqrt (x + 1)) 
  → (0 ≤ a ∧ a ≤ 2 * sqrt 2) :=
by 
  sorry

-- Definitions and conditions for part (2b)
def g (a : ℝ) : ℝ :=
  if 0 < a ∧ a < 4 * sqrt 3 - 5 then
    6 - 2 * a
  else if 4 * sqrt 3 - 5 ≤ a ∧ a < 3 then
    (a + 1) ^ 2 / 4
  else if a ≥ 3 then
    2 * a - 2
  else
    0  -- This case shouldn't happen as per the problem statement

theorem part2b (a : ℝ) (H : 0 < a) :
  g a = function_f 2 a 1 :=
by
  sorry

end part1_part2a_part2b_l331_331660


namespace sum_of_values_of_x_l331_331428

noncomputable def g (x : ℝ) : ℝ :=
if x < 3 then 7 * x + 10 else 3 * x - 18

theorem sum_of_values_of_x (h : ∃ x : ℝ, g x = 5) :
  (∃ x1 x2 : ℝ, g x1 = 5 ∧ g x2 = 5) → (x1 + x2 = 18 / 7) :=
sorry

end sum_of_values_of_x_l331_331428


namespace sum_of_rel_prime_ints_l331_331167

theorem sum_of_rel_prime_ints (a b : ℕ) (h1 : a < 15) (h2 : b < 15) (h3 : a * b + a + b = 71)
    (h4 : Nat.gcd a b = 1) : a + b = 16 := by
  sorry

end sum_of_rel_prime_ints_l331_331167


namespace vector_decomposition_l331_331677

variable (a b c : ℝ × ℝ)
variable (x y : ℝ)

-- Defining vectors a, b, and c as given
def a := (1, 1)
def b := (1, -1)
def c := (-1, 2)

-- Defining the expression of c in terms of a and b
def expression := (1/2 : ℝ) • a + (-3/2 : ℝ) • b

-- Theorem to prove the given expression equals vector c
theorem vector_decomposition : c = expression := by
  sorry

end vector_decomposition_l331_331677


namespace ordered_pairs_count_l331_331612

theorem ordered_pairs_count :
  let conditions (a b : ℕ) :=
    prime (a + b) ∧ 1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100 ∧
    (∃ k : ℤ, (ab + 1) = k * (a + b))
  in (∃ n : ℕ, n = 91 ∧
        (∀ a b, conditions a b → a = 1 ∨ a = b - 1) →
        ((a + b).prime ∧ 2 ≤ a + b ∧ a + b ≤ 200)) :=
  n = 91 := sorry

end ordered_pairs_count_l331_331612


namespace inequality_problem_l331_331994

variable {a b c d : ℝ}

theorem inequality_problem (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
    a + c > b + d ∧ ad^2 > bc^2 ∧ (1 / bc) < (1 / ad) :=
by
  sorry

end inequality_problem_l331_331994


namespace find_x4_l331_331851

def f (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem find_x4 (x1 x2 : ℝ) (h1 : x1 = 4) (h2 : x2 = 16) (h3 : 0 < x1) (h4 : x1 < x2) :
    ∃ x4 : ℝ, x4 = 2 ^ (8 / 3) := by
  sorry

end find_x4_l331_331851


namespace minimum_vacation_cost_l331_331463

-- Definitions based on the conditions in the problem.
def Polina_age : ℕ := 5
def parents_age : ℕ := 30 -- arbitrary age above the threshold, assuming adults

def Globus_cost_old : ℕ := 25400
def Globus_discount : ℝ := 0.02

def AroundWorld_cost_young : ℕ := 11400
def AroundWorld_cost_old : ℕ := 23500
def AroundWorld_commission : ℝ := 0.01

def globus_total_cost (num_adults num_children : ℕ) : ℕ :=
  let initial_cost := (num_adults + num_children) * Globus_cost_old
  let discount := Globus_discount * initial_cost
  initial_cost - discount.to_nat

def around_world_total_cost (num_adults num_children : ℕ) : ℕ :=
  let initial_cost := (num_adults * AroundWorld_cost_old) + (num_children * AroundWorld_cost_young)
  let commission := AroundWorld_commission * initial_cost
  initial_cost + commission.to_nat

-- Total costs calculated for the Dorokhov family with specific parameters.
def globus_final_cost : ℕ := globus_total_cost 2 1  -- 2 adults, 1 child
def around_world_final_cost : ℕ := around_world_total_cost 2 1  -- 2 adults, 1 child

theorem minimum_vacation_cost : around_world_final_cost = 58984 := by
  sorry

end minimum_vacation_cost_l331_331463


namespace geometric_sequence_sum_l331_331711

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum {a : ℕ → ℝ}
  (h_geo : is_geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_condition : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) :
  a 3 + a 5 = 5 := 
sorry

end geometric_sequence_sum_l331_331711


namespace probability_at_least_twice_l331_331232

-- Define the conditions from a)
def probability_of_hitting_target (single_shot_prob : ℝ) : ℕ → ℝ
| 0 => 1 - single_shot_prob
| 1 => single_shot_prob
| 2 => (Mathlib.Combinatorics.choose 3 2) * single_shot_prob^2 * (1 - single_shot_prob)
| 3 => single_shot_prob^3

-- State the problem
theorem probability_at_least_twice :
  probability_of_hitting_target 0.6 2 + probability_of_hitting_target 0.6 3 = 0.648 :=
  sorry

end probability_at_least_twice_l331_331232


namespace olympic_training_partition_l331_331251

noncomputable def can_partition (G : SimpleGraph V) : Prop :=
  (∃ (A B : Finset V), disjoint A B ∧ A ∪ B = Finset.univ ∧ (∀ v ∈ A, (G.adj v).card = 1) ∧ (∀ v ∈ B, (G.adj v).card = 1))

theorem olympic_training_partition (V : Type) [Fintype V] (G : SimpleGraph V) (h : ∀ v : V, (G.adj v).card = 3) :
  can_partition G :=
sorry

end olympic_training_partition_l331_331251


namespace age_difference_is_16_l331_331473

-- Variables
variables (y : ℕ) -- y represents the present age of the younger person

-- Conditions from the problem
def elder_present_age := 30
def elder_age_6_years_ago := elder_present_age - 6
def younger_age_6_years_ago := y - 6

-- Given condition 6 years ago:
def condition_6_years_ago := elder_age_6_years_ago = 3 * younger_age_6_years_ago

-- The theorem to prove the difference in ages is 16 years
theorem age_difference_is_16
  (h1 : elder_present_age = 30)
  (h2 : condition_6_years_ago) :
  elder_present_age - y = 16 :=
by sorry

end age_difference_is_16_l331_331473


namespace neg_p_l331_331341

theorem neg_p :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by
  sorry

end neg_p_l331_331341


namespace sum_of_visible_lattice_points_l331_331257

def is_visible (x y : ℤ) : Prop :=
  Int.gcd x y = 1

def S (d : ℤ) : Finset (ℤ × ℤ) := 
  {p | let (x, y) := p in x^2 + y^2 = d^2 ∧ is_visible x y}

def D : Finset ℤ := {
  x | 1 ≤ x ∧ x ∣ (2021 * 2025)}

def sum_visible_points (d : ℤ) : ℕ := 
  (S d).card

theorem sum_of_visible_lattice_points : 
  (Finset.sum D sum_visible_points) = 20 := 
  sorry

end sum_of_visible_lattice_points_l331_331257


namespace arithmetic_sequence_geometric_sequence_sum_c_sequence_l331_331653

-- Define the arithmetic sequence {a_n}
def S (n : ℕ) : ℕ := n^2 + n
def a (n : ℕ) : ℕ := 2 * n

-- Define the geometric sequence {b_n}
def b (n : ℕ) : ℝ := (1/2)^n

-- Conditions for the geometric sequence
axiom b_condition1 : b 1 * b 3 = 1 / 16
axiom b_condition2 : b 5 = 1 / 32

-- Define the sequence {c_n} and the sum T_n
def c (n : ℕ) : ℝ := 2 * (a n) - b n

def T (n : ℕ) : ℝ :=
  ((n * (n + 1) : ℕ) * 2 + n * 2 - 1 + (1/2)^n : ℝ)

theorem arithmetic_sequence (n : ℕ) : S n = Σ i in finset.range n, a (i + 1) := sorry

theorem geometric_sequence (n : ℕ) : b n = (1/2)^n := sorry

theorem sum_c_sequence (n : ℕ) : T n = Σ i in finset.range n, c (i + 1) := sorry

end arithmetic_sequence_geometric_sequence_sum_c_sequence_l331_331653


namespace original_votes_for_Twilight_l331_331442

theorem original_votes_for_Twilight (T : ℕ) :
  (let V := 10 + T + 20 in
   let remaining_votes_Twilight := T / 2 in
   let remaining_votes_ArtOfDeal := 4 in
   let V' := 10 + remaining_votes_Twilight + remaining_votes_ArtOfDeal in
   10 = 0.5 * V' → T = 12) :=
sorry

end original_votes_for_Twilight_l331_331442


namespace express_15_ab_as_R_b_S_a_l331_331800

theorem express_15_ab_as_R_b_S_a 
  (a b : ℤ) : 
  let R := 3^a in
  let S := 5^b in
  15^(a * b) = (R^b) * (S^a) :=
by 
  let R := 3^a
  let S := 5^b
  calc
    15^(a * b) = (3 * 5)^(a * b) : by sorry
             ... = 3^(a * b) * 5^(a * b) : by sorry
             ... = (3^a)^b * (5^b)^a : by sorry
             ... = R^b * S^a : by sorry

end express_15_ab_as_R_b_S_a_l331_331800


namespace range_of_a_l331_331358

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - 2| ≥ a) → a ≤ 3 :=
by
  intro h
  sorry

end range_of_a_l331_331358


namespace radius_inscribed_circle_eq_l331_331520

noncomputable def calc_radius_inscribed_circle {α : Type*} [linear_ordered_field α] 
  (AB AC BC : α) (h1 : AB = 8) (h2 : AC = 15) (h3 : BC = 17) : α :=
  let s := (AB + AC + BC) / 2,
  K := (s * (s - AB) * (s - AC) * (s - BC)).sqrt in
    K / s

theorem radius_inscribed_circle_eq (r : ℝ) :
  calc_radius_inscribed_circle 8 15 17 (by rfl) (by rfl) (by rfl) = 3 := 
by sorry

end radius_inscribed_circle_eq_l331_331520


namespace next_numbers_for_sequence_a_next_numbers_for_sequence_b_next_numbers_for_sequence_c_next_numbers_for_sequence_d_next_numbers_for_sequence_e_l331_331596

def pattern_a (s : List ℕ) : Prop :=
  s = [19, 20, 22, 25, 29] ∧ (s.last! + 4 + 5 = 34 + 6) -- Dummy simplification for the pattern

def pattern_b (s : List ℕ) : Prop :=
  s = [5, 8, 14, 26, 50] ∧ (50 + 24 * 2 = 98) ∧ (98 + 48 * 2 = 194)

def pattern_c (s : List ℕ) : Prop :=
  s = [253, 238, 223, 208, 193] ∧ (193 - 15 = 178) ∧ (178 - 15 = 163)

def pattern_d (s : List ℕ) : Prop :=
  s = [12, 11, 16, 16, 20, 21, 24, 26] ∧
  ∃ a b, s.index a = (12, 16, 20, 24) ∧ s.index b = (11, 16, 21, 26) ∧ (28, 31)

def pattern_e (s : List ℕ) : Prop :=
  s = [15, 29, 56, 109, 214] ∧ (214 * 2 - 5 = 423) ∧ (423 * 2 - 6 = 840)

theorem next_numbers_for_sequence_a (s : List ℕ) (h : pattern_a s) :
  s.last? = some 34 ∧ (s.last? = some 40) := sorry

theorem next_numbers_for_sequence_b (s : List ℕ) (h : pattern_b s) :
  s.last? = some 98 ∧ (s.last? = some 194) := sorry

theorem next_numbers_for_sequence_c (s : List ℕ) (h : pattern_c s) :
  s.last? = some 178 ∧ (s.last? = some 163) := sorry

theorem next_numbers_for_sequence_d (s : List ℕ) (h : pattern_d s) :
  s.last? = some 28 ∧ (s.last? = some 31) := sorry

theorem next_numbers_for_sequence_e (s : List ℕ) (h : pattern_e s) :
  s.last? = some 423 ∧ (s.last? = some 840) := sorry

end next_numbers_for_sequence_a_next_numbers_for_sequence_b_next_numbers_for_sequence_c_next_numbers_for_sequence_d_next_numbers_for_sequence_e_l331_331596


namespace final_cash_and_asset_positions_l331_331439

def cash_after_transaction (initial_cash : Int) (transaction_amount : Int) (is_buy : Bool) : Int :=
  if is_buy then initial_cash - transaction_amount else initial_cash + transaction_amount

theorem final_cash_and_asset_positions 
(initial_cash_C : Int) (initial_cash_D : Int) (initial_car_value : Int)
(transaction1 : Int) (transaction2 : Int) (transaction3 : Int)
(C_final_cash : Int) (D_final_cash : Int) (C_has_car : Bool) (D_has_car : Bool):
  initial_cash_C = 15000 →
  initial_cash_D = 17000 →
  initial_car_value = 15000 →
  transaction1 = 16000 →
  transaction2 = 14000 →
  transaction3 = 15500 →
  C_final_cash = 32500 →
  D_final_cash = -500 →
  C_has_car = False →
  D_has_car = True →
  (cash_after_transaction (cash_after_transaction (cash_after_transaction initial_cash_C transaction1 False) transaction2 True) transaction3 False) = C_final_cash ∧
  (cash_after_transaction (cash_after_transaction (cash_after_transaction initial_cash_D transaction1 True) transaction2 False) transaction3 True) = D_final_cash ∧
  D_has_car ∧ ¬C_has_car :=
begin
  intros hC0 hD0 hV0 ht1 ht2 ht3 hC_fc hD_fc hC_car hD_car,
  split,
  { simp [cash_after_transaction, hC0, ht1, ht2, ht3, hC_fc] },
  split,
  { simp [cash_after_transaction, hD0, ht1, ht2, ht3, hD_fc] },
  { exact hD_car, },
  { exact hC_car, },
end

end final_cash_and_asset_positions_l331_331439


namespace projectile_reaches_100_feet_l331_331816

noncomputable def projectile_height (t : ℝ) : ℝ :=
  -16 * t^2 + 80 * t

theorem projectile_reaches_100_feet :
  ∃ t : ℝ, t = 2.5 ∧ projectile_height t = 100 :=
by
  use 2.5
  sorry

end projectile_reaches_100_feet_l331_331816


namespace two_af_eq_ab_minus_ac_l331_331678

open EuclideanGeometry Real

namespace TriangleProblem

-- Defining the triangle ABC and given conditions
variables {A B C E F : Point}
variable (T: Triangle A B C)
variable [nontrivial_ring ℝ]

-- Conditions
noncomputable def ab_gt_ac : Prop := T.ab > T.ac
noncomputable def ext_angle_bisector := T.ext_angle_bisector A E
noncomputable def ef_perpendicular_ab : Prop := is_perpendicular E F (Line.mk A B)

-- The theorem statement
theorem two_af_eq_ab_minus_ac
  (hab_gt_ac: ab_gt_ac)
  (ext_bis: ext_angle_bisector)
  (ef_perp: ef_perpendicular_ab):
  2 * T.af = T.ab - T.ac :=
sorry

end TriangleProblem

end two_af_eq_ab_minus_ac_l331_331678


namespace monotonicity_of_f_range_of_m_l331_331664

noncomputable def f (a : ℝ) (b : ℝ) (x : ℝ) : ℝ := a * x * (Real.log x) + b * x

theorem monotonicity_of_f (a b : ℝ) (h : a ≠ 0) (h' : f a b 1 = a + b) :
  (a > 0 → (∀ x ∈ Set.Ioo 0 1, f a b x < 0) ∧ (∀ x ∈ Set.Ioi 1, f a b x > 0)) ∧
  (a < 0 → (∀ x ∈ Set.Ioo 0 1, f a b x > 0) ∧ (∀ x ∈ Set.Ioi 1, f a b x < 0)) :=
by
  sorry

theorem range_of_m (a : ℝ) (h : a ∈ Set.Ioi Real.exp) (m : ℝ)
  (h' : ∀ x₁ x₂ ∈ Set.Icc (1 / 3 * Real.exp) (3 * Real.exp),
           abs (f a (-a) x₁ - f a (-a) x₂) < (m + Real.exp * Real.log 3) * a + 3 * Real.exp) :
  m > 2 * Real.exp * Real.log 3 - 2 :=
by
  sorry

end monotonicity_of_f_range_of_m_l331_331664


namespace f_2009_eq_1_l331_331156

def f : Real → Real
| x => if x ≤ 0 then Real.log2 (1 - x) else -f (x + 3)

theorem f_2009_eq_1 : f 2009 = 1 := 
by
  sorry  -- proof to be provided

end f_2009_eq_1_l331_331156


namespace discounted_price_correct_l331_331891

variable (originalPrice : ℝ) (discountRate : ℝ)

theorem discounted_price_correct :
  originalPrice = 1500 →
  discountRate = 0.15 →
  originalPrice * (1 - discountRate) = 1275 :=
by
  intro h1 h2
  rw [h1, h2]
  norm_num
  sorry

end discounted_price_correct_l331_331891


namespace percentage_of_normal_days_l331_331128

def prob_shark_appearance : ℚ := 1 / 30
def prob_detection_given_shark : ℚ := 3 / 4
def prob_no_detection_given_shark : ℚ := 1 - prob_detection_given_shark
def false_alarm_multiplier : ℚ := 10
def prob_false_alarm : ℚ := false_alarm_multiplier * prob_shark_appearance
def prob_no_alarm_given_no_shark : ℚ := 1 - prob_false_alarm
def prob_no_shark : ℚ := 1 - prob_shark_appearance

def prob_normal_day : ℚ := prob_no_alarm_given_no_shark * prob_no_shark

theorem percentage_of_normal_days : prob_normal_day * 100 ≈ 64.44 :=
by
  sorry

end percentage_of_normal_days_l331_331128


namespace cos_sq_minus_exp_equals_neg_one_fourth_l331_331884

theorem cos_sq_minus_exp_equals_neg_one_fourth :
  (Real.cos (30 * Real.pi / 180))^2 - (2 - Real.pi)^0 = -1 / 4 := by
sorry

end cos_sq_minus_exp_equals_neg_one_fourth_l331_331884


namespace find_other_number_l331_331496

theorem find_other_number (x y : ℕ) (h1 : x + y = 72) (h2 : y = x + 12) (h3 : y = 42) : x = 30 := by
  sorry

end find_other_number_l331_331496


namespace leo_score_l331_331071

-- Definitions for the conditions
def caroline_score : ℕ := 13
def anthony_score : ℕ := 19
def winning_score : ℕ := 21

-- Lean statement for the proof problem
theorem leo_score : ∃ (leo_score : ℕ), leo_score = winning_score := by
  have h_caroline := caroline_score
  have h_anthony := anthony_score
  have h_winning := winning_score
  use 21
  sorry

end leo_score_l331_331071


namespace number_of_right_handed_players_l331_331126

/-- 
Given:
(1) There are 70 players on a football team.
(2) 34 players are throwers.
(3) One third of the non-throwers are left-handed.
(4) All throwers are right-handed.
Prove:
The total number of right-handed players is 58.
-/
theorem number_of_right_handed_players 
  (total_players : ℕ) (throwers : ℕ) (non_throwers : ℕ) (left_handed_non_throwers : ℕ) (right_handed_non_throwers : ℕ) : 
  total_players = 70 ∧ throwers = 34 ∧ non_throwers = total_players - throwers ∧ left_handed_non_throwers = non_throwers / 3 ∧ right_handed_non_throwers = non_throwers - left_handed_non_throwers ∧ right_handed_non_throwers + throwers = 58 :=
by
  sorry

end number_of_right_handed_players_l331_331126


namespace power_difference_expression_l331_331957

theorem power_difference_expression : 
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * (30^1001) :=
by
  sorry

end power_difference_expression_l331_331957


namespace sphere_circumscribed_around_cone_radius_l331_331613

-- Definitions of the given conditions
variable (r h : ℝ)

-- Theorem statement (without the proof)
theorem sphere_circumscribed_around_cone_radius :
  ∃ R : ℝ, R = (Real.sqrt (r^2 + h^2)) / 2 :=
sorry

end sphere_circumscribed_around_cone_radius_l331_331613


namespace solution_set_I_range_of_m_l331_331020

noncomputable def f (x : ℝ) : ℝ := |2 * x + 3| + |2 * x - 1|

theorem solution_set_I (x : ℝ) : f x < 8 ↔ -5 / 2 < x ∧ x < 3 / 2 :=
sorry

theorem range_of_m (m : ℝ) (h : ∃ x, f x ≤ |3 * m + 1|) : m ≤ -5 / 3 ∨ m ≥ 1 :=
sorry

end solution_set_I_range_of_m_l331_331020


namespace rectangle_sides_l331_331827

theorem rectangle_sides (x y : ℕ) (h_diff : x ≠ y) (h_eq : x * y = 2 * x + 2 * y) : 
  (x = 3 ∧ y = 6) ∨ (x = 6 ∧ y = 3) :=
sorry

end rectangle_sides_l331_331827


namespace find_side_lengths_l331_331657

variable (a b : ℝ)

-- Conditions
def diff_side_lengths := a - b = 2
def diff_areas := a^2 - b^2 = 40

-- Theorem to prove
theorem find_side_lengths (h1 : diff_side_lengths a b) (h2 : diff_areas a b) :
  a = 11 ∧ b = 9 := by
  -- Proof skipped
  sorry

end find_side_lengths_l331_331657


namespace computer_additions_per_hour_l331_331895

theorem computer_additions_per_hour : 
  ∀ (initial_rate : ℕ) (increase_rate: ℚ) (intervals_per_hour : ℕ),
  initial_rate = 12000 → 
  increase_rate = 0.05 → 
  intervals_per_hour = 4 → 
  (12000 * 900) + (12000 * 1.05 * 900) + (12000 * 1.05^2 * 900) + (12000 * 1.05^3 * 900) = 46549350 := 
by
  intros initial_rate increase_rate intervals_per_hour h1 h2 h3
  have h4 : initial_rate = 12000 := h1
  have h5 : increase_rate = 0.05 := h2
  have h6 : intervals_per_hour = 4 := h3
  sorry

end computer_additions_per_hour_l331_331895


namespace max_value_p_l331_331600

/-- Given the conditions from the execution of the program flowchart -/
def program_flowchart : ℕ → ℕ
| 1 := 0
| 2 := 1
| 3 := 3
| 4 := 7
| 5 := 15
| _ := 0  -- Default case for simplicity

theorem max_value_p (p : ℕ) : p ≤ 15 := by
  -- Program reaches k = 5 where S = 15
  have h : program_flowchart 5 = 15 := rfl
  -- Since S (15) ≥ p and S = 15 when k = 5, the maximum p is 15
  exact le_refl 15

end max_value_p_l331_331600


namespace odd_function_neg_condition_l331_331626

theorem odd_function_neg_condition (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_pos : ∀ x, x > 0 → f x = -x * (1 + x)) :
  ∀ x, x < 0 → f x = x * (x - 1) :=
by
  intro x hx
  have h_neg := h_pos (-x) (neg_pos.mpr hx)
  rw [h_odd x] at h_neg
  rw [neg_mul, neg_add, neg_one_mul, neg_neg] at h_neg
  exact h_neg

end odd_function_neg_condition_l331_331626


namespace locker_digit_packages_needed_l331_331235

theorem locker_digit_packages_needed :
  let range1 := list.range' 300 51, -- Lockers from 300 to 350
  let range2 := list.range' 400 51, -- Lockers from 400 to 450
  let digit_occurrences h := h.div 100, 
  let digit_occurrences t := (t / 10) % 10,
  let digit_occurrences u := t % 10,
  let total_occurrences := list.join (range1 ++ range2).map (λ n, [digit_occurrences n, digit_occurrences t, digit_occurrences u]).flatten,
  let digit_count := list.map (λ d, list.count total_occurrences d) (list.range 10),
  let max_occurrence := list.maximum digit_count,
  max_occurrence = 73
:=
sorry

end locker_digit_packages_needed_l331_331235


namespace triangle_side_relation_l331_331476

variables {A B C : Type}
-- Consider A, B, C as points of type real (for simplicity).
variables {angle : Type → Type} 
-- Consider angle as a type representing an angle object.
variables [AddGroup angle] -- Assuming additive group structure for angle measures.
variable [Point : A]
variable [Point : B]
variable [Point : C]

-- Introduce the required lengths and angles associated with the triangle.
variables {AC BC AB : ℝ} -- Length of the sides.
variables {angle_A : angle A} {angle_B : angle B} -- Measures of the angles at A and B.

-- Given condition: angle A is twice angle B.
hypothesis (h1 : angle_A = 2 * angle_B)

-- Conclusion: |BC|^2 = (|AC| + |AB|) |AC|.
theorem triangle_side_relation
  (angle_A : 2*angle B)
  (AB AC BC : ℝ) :
   BC^2 = (AC + AB) * AC :=
sorry

end triangle_side_relation_l331_331476


namespace delicious_cake_exists_for_n_ge_5_l331_331292

noncomputable def delicious (n : ℕ) : Prop :=
  ∃ (cake : Array (Array Bool)),
  (∀ i j, (i = j ∨ i + j = n - 1) → cake[i][j] = true) ∧
  (∀ i, (Array.foldr (λ b acc => if b then acc + 1 else acc) 0 cake[i]) % 2 = 1) ∧
  (∀ j, (Array.foldr (λ b acc => if b then acc + 1 else acc) 0 (Array.map (λ row => row[j]) cake)) % 2 = 1) ∧
  let strawberries := Array.foldr (λ row acc => acc + Array.foldr Bool.intro 0 row) 0 cake
  in strawberries = (n * n + 1) / 2

theorem delicious_cake_exists_for_n_ge_5 :
  ∀ n, n ≥ 5 → delicious n :=
by
  intros n h
  sorry

end delicious_cake_exists_for_n_ge_5_l331_331292


namespace difference_of_sums_1000_l331_331861

def sum_first_n_even (n : ℕ) : ℕ :=
  n * (n + 1)

def sum_first_n_odd_not_divisible_by_5 (n : ℕ) : ℕ :=
  (n * n) - 5 * ((n / 5) * ((n / 5) + 1))

theorem difference_of_sums_1000 :
  (sum_first_n_even 1000) - (sum_first_n_odd_not_divisible_by_5 1000) = 51000 :=
by
  sorry

end difference_of_sums_1000_l331_331861


namespace minimize_product_l331_331643

theorem minimize_product
    (a b c : ℕ) 
    (h_positive: a > 0 ∧ b > 0 ∧ c > 0)
    (h_eq: 10 * a^2 - 3 * a * b + 7 * c^2 = 0) : 
    (gcd a b) * (gcd b c) * (gcd c a) = 3 :=
sorry

end minimize_product_l331_331643


namespace selling_price_per_pound_l331_331897

-- Definitions based on conditions
def cost_per_pound_type1 : ℝ := 2.00
def cost_per_pound_type2 : ℝ := 3.00
def weight_type1 : ℝ := 64
def weight_type2 : ℝ := 16
def total_weight : ℝ := 80

-- The selling price per pound of the mixture
theorem selling_price_per_pound :
  let total_cost := (weight_type1 * cost_per_pound_type1) + (weight_type2 * cost_per_pound_type2)
  (total_cost / total_weight) = 2.20 :=
by
  sorry

end selling_price_per_pound_l331_331897


namespace smallest_b_in_ap_l331_331732

-- Definition of an arithmetic progression
def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b

-- Problem statement in Lean
theorem smallest_b_in_ap (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h_ap : is_arithmetic_progression a b c) 
  (h_prod : a * b * c = 216) : 
  b ≥ 6 :=
by
  sorry

end smallest_b_in_ap_l331_331732


namespace triangle_side_length_l331_331541

theorem triangle_side_length 
  (O A B C : Point)
  (r : ℝ) 
  (s : ℝ) 
  (h_circle_area : π * r^2 = 100 * π) 
  (h_triangle_equilateral : equilateral_triangle A B C) 
  (h_bc_chord : on_chord B C O) 
  (h_OA_distance : dist O A = 5)
  (h_O_outside_triangle : outside_triangle O A B C)
  : s = 5 := 
sorry

end triangle_side_length_l331_331541


namespace sum_of_first_10_terms_is_350_l331_331147

-- Define the terms and conditions for the arithmetic sequence
variables (a d : ℤ)

-- Define the 4th and 8th terms of the sequence
def fourth_term := a + 3*d
def eighth_term := a + 7*d

-- Given conditions
axiom h1 : fourth_term a d = 23
axiom h2 : eighth_term a d = 55

-- Sum of the first 10 terms of the sequence
def sum_first_10_terms := 10 / 2 * (2*a + (10 - 1)*d)

-- Theorem to prove
theorem sum_of_first_10_terms_is_350 : sum_first_10_terms a d = 350 :=
by sorry

end sum_of_first_10_terms_is_350_l331_331147


namespace female_employees_count_l331_331180

theorem female_employees_count (E Male_E Female_E M : ℕ)
  (h1: M = (2 / 5) * E)
  (h2: 200 = (E - Male_E) * (2 / 5))
  (h3: M = (2 / 5) * Male_E + 200) :
  Female_E = 500 := by
{
  sorry
}

end female_employees_count_l331_331180


namespace meaningful_sqrt_l331_331360

theorem meaningful_sqrt (x : ℝ) (h : x - 5 ≥ 0) : x = 6 :=
sorry

end meaningful_sqrt_l331_331360


namespace one_product_success_profit_distribution_l331_331223

namespace CompanyResearch

-- Conditions
def probs : probability_space ℝ :=
{ sample_space := { successesA := 0.75, successesB := 0.6},
  prob := sorry }

def successA : events ℝ := { event := λ ω, ω.successesA}
def successB : events ℝ := { event := λ ω, ω.successesB}

-- (1) Probability that exactly one new product is successfully developed
theorem one_product_success (probs : ℝ) (successA successB : prob_event probs) : 
  (Pr[successA] = 3/4) → 
  (Pr[successB] = 3/5) → 
  (independent successA successB) → 
  Pr[successA ∩ successBᶜ ∪ successAᶜ ∩ successB] = 3/20 + 6/20 := 
sorry

-- (2) Distribution of the company's profit
inductive Profit
| neg_90 : Profit
| pos_50 : Profit
| pos_80 : Profit
| pos_220 : Profit

def profit_dist : probability_space Profit :=
{ sample_space := { Profit.neg_90, Profit.pos_50, Profit.pos_80, Profit.pos_220 },
  prob := λ x, match x with
    | Profit.neg_90 := 0.1
    | Profit.pos_50 := 0.15
    | Profit.pos_80 := 0.3
    | Profit.pos_220 := 0.45
  end := sorry }

theorem profit_distribution (probs : ℝ) (successA successB : prob_event probs) : 
  (Pr[successA] = 3/4) → 
  (Pr[successB] = 3/5) → 
  (independent successA successB) →
  immeasurable (profit_dist.ProbabilitySizeSpace) :=
sorry

end CompanyResearch

end one_product_success_profit_distribution_l331_331223


namespace cos_sum_identity_integral_cos_squared_l331_331445

theorem cos_sum_identity (n : ℕ) (hn : 1 ≤ n) :
  (1 : ℝ) / n * (∑ i in Finset.range n, (Real.cos (2 * (i + 1) * Real.pi / (2 * n + 1)))^2) = 
  (2 * n - 1) / (4 * n) := 
sorry

theorem integral_cos_squared :
  ∫ x in 0 .. Real.pi, (Real.cos x)^2 = Real.pi / 2 := 
sorry

end cos_sum_identity_integral_cos_squared_l331_331445


namespace cartesian_eq_and_intersection_sum_l331_331379

theorem cartesian_eq_and_intersection_sum
  (α θ ρ x y : ℝ)
  (C_param : x = 3 * Real.cos α ∧ y = Real.sin α)
  (l_param : ρ * Real.sin (θ - (Real.pi / 4)) = Real.sqrt 2)
  (P : x = 0 ∧ y = 2) :
  (∃ (x y : ℝ), x^2 / 9 + y^2 = 1) ∧
  (∃ (θ : ℝ), θ = Real.pi / 4) ∧
  (∃ (t1 t2 : ℝ), |(t1 * sqrt 2 / 2)| + |(t2 * sqrt 2 / 2)| = 18 * sqrt 2 / 5) :=
by {
  sorry
}

end cartesian_eq_and_intersection_sum_l331_331379


namespace total_cost_is_correct_l331_331130

-- Define the price of pizzas
def pizza_price : ℕ := 5

-- Define the count of triple cheese and meat lovers pizzas
def triple_cheese_pizzas : ℕ := 10
def meat_lovers_pizzas : ℕ := 9

-- Define the special offers
def buy1get1free (count : ℕ) : ℕ := count / 2 + count % 2
def buy2get1free (count : ℕ) : ℕ := (count / 3) * 2 + count % 3

-- Define the cost calculations using the special offers
def cost_triple_cheese : ℕ := buy1get1free triple_cheese_pizzas * pizza_price
def cost_meat_lovers : ℕ := buy2get1free meat_lovers_pizzas * pizza_price

-- Define the total cost calculation
def total_cost : ℕ := cost_triple_cheese + cost_meat_lovers

-- The theorem we need to prove
theorem total_cost_is_correct :
  total_cost = 55 := by
  sorry

end total_cost_is_correct_l331_331130


namespace Vieta_formulas_l331_331733

noncomputable def polynomial_coeff (n : ℕ) (a : ℕ → ℤ) : ℕ → ℤ
| 0      := a 0
| (k+1) := a (k+1) + polynomial_coeff k a

theorem Vieta_formulas (n : ℕ) (a : ℕ → ℤ) (x : ℕ → ℂ)
  (h1 : ∀ i : ℕ, i < n → (nat_degree (X - C (x i)).to_X :=
    (∏ i in finset.range n, (X - C (x i))) = polynomial_coeff n a
| : (∑ i in finset.range n, x i) = -a (n-1) ∧ 
    (∑ i in (finset.range (n)).image (λ (i : ℕ), x i * x (i + 1)) = a (n-2) ∧ 
    (∑ i in finset.attach (finset.range (n)).image (λ (i : ℕ), x i * x (i + 2)) = -a (n-3) ∧ 
    (∏ i in finset.range n, x i) = (-1) ^ n * (a 0) :=
sorry

end Vieta_formulas_l331_331733


namespace percentage_reduction_in_price_l331_331555

-- Definitions for the given conditions
def original_price : ℝ := 5.56
def total_cost : ℝ := 250
def additional_gallons : ℝ := 5

-- Definition for the reduced price (from the solution)
def reduced_price : ℝ := 5.00

-- The proof problem statement to show percentage reduction
theorem percentage_reduction_in_price :
  let percentage_reduction := (original_price - reduced_price) / original_price * 100 in
  abs (percentage_reduction - 10.07) < 0.01 := 
by
  sorry

end percentage_reduction_in_price_l331_331555


namespace max_expression_value_l331_331968

theorem max_expression_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let expr := (|4 * a - 10 * b| + |2 * (a - b * sqrt 3) - 5 * (a * sqrt 3 + b)|) / sqrt (a^2 + b^2) in
  expr ≤ 2 * sqrt 87 :=
sorry

end max_expression_value_l331_331968


namespace monotonic_increasing_intervals_l331_331831

def f (x : ℝ) : ℝ := abs (x^2 - 2 * x - 3)

theorem monotonic_increasing_intervals : 
  (∀ x y : ℝ, x ∈ Icc (-1 : ℝ) (1 : ℝ) → y ∈ Icc (-1 : ℝ) (1 : ℝ) → x ≤ y → f x ≤ f y) ∧
  (∀ x y : ℝ, x ∈ Ici (3 : ℝ) → y ∈ Ici (3 : ℝ) → x ≤ y → f x ≤ f y) := by
  sorry

end monotonic_increasing_intervals_l331_331831


namespace least_points_twelveth_game_l331_331376

def scores := (18, 25, 10, 22)
def total_after_eleven (total_first_seven : ℕ) :=
  total_first_seven + scores.1 + scores.2 + scores.3 + scores.4

theorem least_points_twelveth_game (total_first_seven : ℕ) (h1 : total_first_seven < total_after_eleven total_first_seven)
  (h2 : ∀ avg_total_points_after_twelve > 20) :
  ∃ pts_twelve : ℕ, pts_twelve >= 42 :=
by
  sorry

end least_points_twelveth_game_l331_331376


namespace ron_needs_to_drink_60_percent_l331_331215

theorem ron_needs_to_drink_60_percent :
  let V := 400 in
  let min_intake := 30 in
  let final_volume := 500 in
  let percentage_needed (volume : ℕ) :=
    (min_intake : ℚ) / (volume : ℚ) * 100 in
  let drinks := percentage_needed final_volume in
  drinks * 3 = 60 :=
by
  sorry

end ron_needs_to_drink_60_percent_l331_331215


namespace simplify_sqrt_l331_331145

-- Define the domain and main trigonometric properties
open Real

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  sqrt (1 - 2 * sin x * cos x)

-- Define the main theorem with given conditions
theorem simplify_sqrt {x : ℝ} (h1 : (5 / 4) * π < x) (h2 : x < (3 / 2) * π) (h3 : cos x > sin x) :
  simplify_expression x = cos x - sin x :=
  sorry

end simplify_sqrt_l331_331145


namespace F_2517_correct_number_of_interactive_lines_l331_331100

structure FourDigitNumber where
  a b c : ℕ
  condition_c_ne_zero : c ≠ 0
  value : ℕ := 1000 * a + 100 * b + 10 * 1 + c

def N (M : FourDigitNumber) : ℕ := 
  1000 * M.c + 100 * M.a + 10 * M.b + 1

noncomputable def F (M : FourDigitNumber) : ℕ := 
  (M.value + N M) / 11

theorem F_2517_correct : F {a := 2, b := 5, c := 7, condition_c_ne_zero := by decide} = 888 := 
  sorry

theorem number_of_interactive_lines (M : FourDigitNumber) : (F M % 6 = 0) → 
  ∃ n, n = 8 :=
  sorry

end F_2517_correct_number_of_interactive_lines_l331_331100


namespace find_smallest_r_l331_331615

noncomputable def smallest_r (x : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n → (∑ i in finset.range (n+1), x i) ≤ r * x n

theorem find_smallest_r :
  (∃ x : ℕ → ℝ, (∀ n, 1 ≤ n → 0 < x n) ∧ smallest_r x 4) ∧
  (∀ r : ℝ, (∃ x : ℕ → ℝ, (∀ n, 1 ≤ n → 0 < x n) ∧ smallest_r x r) → 4 ≤ r) :=
sorry

end find_smallest_r_l331_331615


namespace identical_functions_l331_331350

def f1 (x : ℝ) : ℝ := Real.sqrt ((x - 1)^2)
def g1 (x : ℝ) : ℝ := x - 1

def f2 (x : ℝ) : ℝ := x - 1
def g2 (t : ℝ) : ℝ := t - 1

def f3 (x : ℝ) : ℝ := Real.sqrt (x^2 - 1)
def g3 (x : ℝ) : ℝ := Real.sqrt (x + 1) * Real.sqrt (x - 1)

def f4 (x : ℝ) : ℝ := x
def g4 (x : ℝ) : ℝ := x^2 / x

theorem identical_functions : (∀ x : ℝ, f2 x = g2 x) 
                            ∧ (∀ x : ℝ, (∃ y : ℝ, x = y - 1) ↔ (f2 x ∈ ℝ)) 
                            ∧ (∀ x : ℝ, (∃ t : ℝ, x = t - 1) ↔ (g2 x ∈ ℝ)) :=
by
  sorry

end identical_functions_l331_331350


namespace power_function_solution_l331_331691

def power_function_does_not_pass_through_origin (m : ℝ) : Prop :=
  (m^2 - m - 2) ≤ 0

def condition (m : ℝ) : Prop :=
  m^2 - 3 * m + 3 = 1

theorem power_function_solution (m : ℝ) :
  power_function_does_not_pass_through_origin m ∧ condition m → (m = 1 ∨ m = 2) :=
by sorry

end power_function_solution_l331_331691


namespace main_l331_331113

theorem main (f : ℝ → ℝ) (f' : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f (2 + x) = f (2 - x)) 
  (h2 : ∀ x, 2 - x ≠ 0 → f' x / (2 - x) > 0)
  (h3 : 2 < a) (h4 : a < 4) : 
  f (2 ^ a) < f (Real.log2 a) ∧ f (Real.log2 a) < f 2 := 
sorry

end main_l331_331113


namespace find_a1_in_geometric_sequence_l331_331382

noncomputable def geometric_sequence_first_term (a : ℕ → ℝ) (r : ℝ) (h : ∀ n : ℕ, a (n + 1) = a n * r) : ℝ :=
  a 0

theorem find_a1_in_geometric_sequence (a : ℕ → ℝ) (h_geo : ∀ n : ℕ, a (n + 1) = a n * (1 / 2)) :
  a 2 = 16 → a 3 = 8 → geometric_sequence_first_term a (1 / 2) h_geo = 64 :=
by
  intros h2 h3
  -- Proof would go here
  sorry

end find_a1_in_geometric_sequence_l331_331382


namespace rationalize_denominator_correct_l331_331772

noncomputable def rationalize_denominator_sum : ℕ :=
  let a := real.root (5 : ℝ) 3;
  let b := real.root (3 : ℝ) 3;
  let A := real.root (25 : ℝ) 3;
  let B := real.root (15 : ℝ) 3;
  let C := real.root (9 : ℝ) 3;
  let D := 2;
  (25 + 15 + 9 + 2)

theorem rationalize_denominator_correct :
  rationalize_denominator_sum = 51 :=
  by sorry

end rationalize_denominator_correct_l331_331772


namespace no_solution_xy_l331_331454

theorem no_solution_xy (x y : ℕ) : ¬ (x * (x + 1) = 4 * y * (y + 1)) :=
sorry

end no_solution_xy_l331_331454


namespace teams_in_BIG_M_l331_331708

theorem teams_in_BIG_M (n : ℕ) (h : n * (n - 1) / 2 = 36) : n = 9 :=
sorry

end teams_in_BIG_M_l331_331708


namespace range_and_mode_l331_331636

open Multiset

noncomputable def data_set : Multiset ℕ := {15, 13, 15, 16, 17, 16, 14, 15}

theorem range_and_mode :
  (data_set.sup - data_set.inf = 4) ∧
  ((data_set.mode : Option ℕ) = some 15) := by
-- Proof here
sorry

end range_and_mode_l331_331636


namespace cos_sq_minus_exp_equals_neg_one_fourth_l331_331885

theorem cos_sq_minus_exp_equals_neg_one_fourth :
  (Real.cos (30 * Real.pi / 180))^2 - (2 - Real.pi)^0 = -1 / 4 := by
sorry

end cos_sq_minus_exp_equals_neg_one_fourth_l331_331885


namespace find_b_l331_331963

theorem find_b 
  (a b c x : ℝ)
  (h : (3 * x^2 - 4 * x + 5 / 2) * (a * x^2 + b * x + c) 
       = 6 * x^4 - 17 * x^3 + 11 * x^2 - 7 / 2 * x + 5 / 3) 
  (ha : 3 * a = 6) : b = -3 := 
by 
  sorry

end find_b_l331_331963


namespace gcd_seven_eight_fact_l331_331965

-- Definitions based on the problem conditions
def seven_fact : ℕ := 1 * 2 * 3 * 4 * 5 * 6 * 7
def eight_fact : ℕ := 8 * seven_fact

-- Statement of the theorem
theorem gcd_seven_eight_fact : Nat.gcd seven_fact eight_fact = seven_fact := by
  sorry

end gcd_seven_eight_fact_l331_331965


namespace div_by_90_l331_331388

def N : ℤ := 19^92 - 91^29

theorem div_by_90 : ∃ k : ℤ, N = 90 * k := 
sorry

end div_by_90_l331_331388


namespace distinct_elements_in_T_l331_331587

def sequence1 (k : ℕ) : ℤ := 3 * k - 1
def sequence2 (m : ℕ) : ℤ := 8 * m + 2

def setC : Finset ℤ := Finset.image sequence1 (Finset.range 3000)
def setD : Finset ℤ := Finset.image sequence2 (Finset.range 3000)
def setT : Finset ℤ := setC ∪ setD

theorem distinct_elements_in_T : setT.card = 3000 := by
  sorry

end distinct_elements_in_T_l331_331587


namespace find_an_l331_331655

variable (a : ℕ → ℕ)

axiom S_def (n : ℕ) : (∑ i in Finset.range(n+1), a i) = n^2 + 2*n

theorem find_an (n : ℕ) : a n = 2*n + 1 :=
by
  -- proof to be filled in later
  sorry

end find_an_l331_331655


namespace correct_statements_l331_331993

variable (a b : Line) (α : Plane) (P : Point)

-- Statement 2 condition and result
def statement2 := (a ∩ α = P) → (b ⊂ α) → ¬ (a ∥ b)

-- Statement 3 condition and result
def statement3 := (a ∥ b) → (b ⊥ α) → (a ⊥ α)

-- Proof of correct statements being statement 2 and statement 3
theorem correct_statements (h2 : statement2 a b α P) (h3 : statement3 a b α) : 
  (statement2 a b α P) ∧ (statement3 a b α) :=
by
  split
  case left => exact h2
  case right => exact h3

end correct_statements_l331_331993


namespace area_triangle_ACM_l331_331759

theorem area_triangle_ACM (R : ℝ) :
  let O := (0, 0)
  let A := (-R, 0)
  let B := (R, 0)
  let C := (3 * R, 0)
  let M := (R, R * sqrt 2)
  let AC := dist A C
  let CM := dist C M
  let sin_OCM := R / (3 * R)
  AC = 4 * R ∧ CM = 2 * R * sqrt 2 ∧ sin_OCM = 1 / 3 →
  area (A, C, M) = (4 * R^2 * sqrt 2) / 3 := by
  sorry

end area_triangle_ACM_l331_331759


namespace find_angle_A_l331_331715

theorem find_angle_A (a b : ℝ) (B : ℝ) (h1 : a = 4 * Real.sqrt 3)
    (h2 : b = 12) (h3 : B = 60) : 
    let angle_A : ℝ := 30
in sorry

end find_angle_A_l331_331715


namespace segment_length_at_M_minimum_segment_length_l331_331173

open Real

variables (a : ℝ) (M : ℝ) (x : ℝ)
noncomputable def segment_length_through_M (a : ℝ) : ℝ :=
sqrt (15) * a / 3

theorem segment_length_at_M (a : ℝ) :
  let BD := a * sqrt 2 in
  let DM := BD / 3 in
  let M := sqrt (15) * a / 3 in
  M = segment_length_through_M a :=
by {
  let BD := a * sqrt 2,
  let DM := BD / 3,
  let length := sqrt (15) * a / 3,
  exact rfl,
}

theorem minimum_segment_length (a : ℝ) :
  let min_length := sqrt (10) * a / 4 in
  ∀ M : ℝ, M = min_length \/
  M = segment_length_through_M a :=
by {
  let min_length := sqrt (10) * a / 4,
  intro M,
  right,
  exact rfl,
}

end segment_length_at_M_minimum_segment_length_l331_331173


namespace rationalize_denominator_l331_331767

theorem rationalize_denominator :
  ∃ A B C D : ℝ, 
    (1 / (real.cbrt 5 - real.cbrt 3) = (real.cbrt A + real.cbrt B + real.cbrt C) / D) ∧ 
    A + B + C + D = 51 :=
  sorry

end rationalize_denominator_l331_331767


namespace total_amount_l331_331535

variable (A B C : ℕ)
variable (h1 : C = 495)
variable (h2 : (A - 10) * 18 = (B - 20) * 11)
variable (h3 : (B - 20) * 24 = (C - 15) * 18)

theorem total_amount (A B C : ℕ) (h1 : C = 495)
  (h2 : (A - 10) * 18 = (B - 20) * 11)
  (h3 : (B - 20) * 24 = (C - 15) * 18) :
  A + B + C = 1105 :=
sorry

end total_amount_l331_331535


namespace perfect_square_sum_exists_l331_331107

theorem perfect_square_sum_exists {n : ℕ} (hn : n ≥ 15) (A B : Set ℕ) (hAB : A ∩ B = ∅) (hUnion : A ∪ B = {x | x ∈ set.univ ∧ x ≤ n ∧ x > 0}) :
  (∃ a b ∈ A, a ≠ b ∧ ∃ k, (a + b = k * k)) ∨ (∃ a b ∈ B, a ≠ b ∧ ∃ k, (a + b = k * k)) :=
sorry

end perfect_square_sum_exists_l331_331107


namespace integer_solutions_count_l331_331832

/-- The number of integer solutions to the inequality system 
    2x + 1 > -3 and -x + 3 ≥ 0 is 5. -/
theorem integer_solutions_count :
  (∃ x : ℤ, 2*x + 1 > -3 ∧ -x + 3 ≥ 0) →
  (finset.card ((finset.filter (λ x, 2*x + 1 > -3 ∧ -x + 3 ≥ 0) (finset.range 6)).lift ℤ) = 5) :=
sorry

end integer_solutions_count_l331_331832


namespace problem1_problem2_l331_331670

-- Problem 1
theorem problem1 (a : ℕ → ℕ) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 3)
  (h_rec : ∀ n, a (n + 2) = 3 * a (n + 1) - a n) :
  ∀ n, ∃ k : ℕ, 1 + a n * a (n + 2) = k^2 :=
by sorry

-- Problem 2
theorem problem2 (a : ℕ → ℕ)
  (h1 : a 1 = 1) 
  (h2 : a 2 = 3)
  (h_rec : ∀ n, a (n + 2) = 3 * a (n + 1) - a n)
  (b : ℕ → ℝ := λ n, let q := (a (n + 1)^2 / a n) in ⌊q⌋ * (q - ⌊q⌋)) :
  ∃ k : ℕ, k = ∏ k in (finset.range 2019).map (finset.nat.cast_embedding).succ, b k :=
by sorry

end problem1_problem2_l331_331670


namespace sequence_contains_one_iff_l331_331855

-- Define the twist function
def twist (s k : ℕ) : ℕ :=
  let a := k / s
  let b := k % s
  b * s + a

-- Define the sequence
def sequence (s n : ℕ) : ℕ → ℕ
| 0     := n
| (i+1) := twist s (sequence s n i)

-- Main theorem statement
theorem sequence_contains_one_iff (s n : ℕ) (h : s ≥ 2) :
  (∃ i, sequence s n i = 1) ↔ n % (s^2 - 1) = 1 ∨ n % (s^2 - 1) = s :=
by
  sorry

end sequence_contains_one_iff_l331_331855


namespace lily_ducks_l331_331451

variable (D G : ℕ)
variable (Rayden_ducks : ℕ := 3 * D)
variable (Rayden_geese : ℕ := 4 * G)
variable (Lily_geese : ℕ := 10) -- Given G = 10
variable (Rayden_extra : ℕ := 70) -- Given Rayden has 70 more ducks and geese

theorem lily_ducks (h : 3 * D + 4 * Lily_geese = D + Lily_geese + Rayden_extra) : D = 20 :=
by sorry

end lily_ducks_l331_331451


namespace sufficient_but_not_necessary_condition_l331_331628

-- Defining the function f(x)
def f (x φ : ℝ) : ℝ := Real.sin (x + φ)

-- Defining what it means for f(x) to be an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Proving the main statement
theorem sufficient_but_not_necessary_condition
  (φ : ℝ) : (φ = π / 2 → is_even_function (f x)) ∧ (is_even_function (f x) → φ = π / 2) :=
by
  sorry

end sufficient_but_not_necessary_condition_l331_331628


namespace distance_from_unselected_vertex_l331_331243

-- Define the problem statement
theorem distance_from_unselected_vertex
  (base length : ℝ) (area : ℝ) (h : ℝ) 
  (h_area : area = (base * h) / 2) 
  (h_base : base = 8) 
  (h_area_given : area = 24) : 
  h = 6 :=
by
  -- The proof here is skipped
  sorry

end distance_from_unselected_vertex_l331_331243


namespace digit_at_308th_pos_l331_331517

def repeating_sequence : List ℕ := [3, 2, 4]

def fraction_as_decimal := 12 / 37

def digit_at_pos (n : ℕ) : ℕ := repeating_sequence[(n % 3) - 1]

theorem digit_at_308th_pos : digit_at_pos 308 = 2 :=
by
  -- proof steps here
  sorry

end digit_at_308th_pos_l331_331517


namespace proof_problem_l331_331658

-- Definitions of the conditions
def line_l : ℝ → ℝ → Prop := λ x y, 3 * x - 4 * y + 2 = 0

def point_P : ℝ × ℝ := (-2, 2)

def line_p₁ : ℝ × ℝ → ℝ → ℝ → Prop := λ (pt : ℝ × ℝ) x y, 
  let (px, py) := pt in 4 * x + 3 * y + (4 * px + 3 * py) = 0

def line1 : ℝ → ℝ → Prop := λ x y, x - y - 1 = 0
def line2 : ℝ → ℝ → Prop := λ x y, 2 * x + y - 2 = 0

-- Correct answers to prove
def perp_line : ℝ → ℝ → Prop := λ x y, 4 * x + 3 * y + 2 = 0

def intersection_point : ℝ × ℝ := (1, 0)

def distance_from_point_to_line : ℝ × ℝ → (ℝ → ℝ → Prop) → ℝ := λ pt line,
  let (px, py) := pt in
  abs (3 * px - 4 * py + 2) / sqrt (9 + 16)

-- Prove the equivalent math problem
theorem proof_problem :
  (∀ x y, line_l x y ↔ 3 * x - 4 * y + 2 = 0) ∧
  line_p₁ point_P = perp_line ∧
  (∃ x y, line1 x y ∧ line2 x y ∧ (x, y) = intersection_point) ∧
  distance_from_point_to_line intersection_point line_l = 1 :=
by
  sorry

end proof_problem_l331_331658


namespace number_of_valid_subsets_l331_331421

open Finset

theorem number_of_valid_subsets : 
  {S : Finset ℕ // ∀ n ∈ S, n ∈ range 1 11 ∧ S.card = 4 ∧
  (∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
   a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
   a > b ∧ a > d ∧
   (b = c + 4 ∨ d = c + 4 ∨ b = d + 4 ∨ d = b + 4)) } = 36 :=
by
  sorry

end number_of_valid_subsets_l331_331421


namespace difference_between_mean_and_median_l331_331129

noncomputable def weighted_mean (w1 w2 w3 w4 s1 s2 s3 s4 : ℝ) : ℝ :=
  (w1 * s1 + w2 * s2 + w3 * s3 + w4 * s4)

theorem difference_between_mean_and_median :
  let w1 := 0.15 in let s1 := 60 in
  let w2 := 0.40 in let s2 := 75 in
  let w3 := 0.20 in let s3 := 85 in
  let w4 := 0.25 in let s4 := 95 in
  let mean_score := weighted_mean w1 w2 w3 w4 s1 s2 s3 s4 in
  let median_score := 75 in
  median_score - mean_score = -5 :=
by
  -- Definitions and conditions
  let w1 := 0.15 
  let s1 := 60 
  let w2 := 0.40 
  let s2 := 75 
  let w3 := 0.20 
  let s3 := 85 
  let w4 := 0.25 
  let s4 := 95 

  -- Calculate mean
  let mean_score := weighted_mean w1 w2 w3 w4 s1 s2 s3 s4

  -- Median score
  let median_score := 75 

  -- Expected result
  have h : median_score - mean_score = -5 := sorry
  exact h

end difference_between_mean_and_median_l331_331129


namespace rationalize_denominator_l331_331773

theorem rationalize_denominator :
  let a := (∛5 : ℝ)
  let b := (∛3 : ℝ)
  let expr := 1 / (a - b)
  let num := a^2 + a*b + b^2
  ∃ A B C D : ℝ,
    expr * num = (∛A + ∛B + ∛C) / D ∧
    A = 25 ∧ B = 15 ∧ C = 9 ∧ D = 2 ∧
    A + B + C + D = 51 :=
by {
  let a := (∛5 : ℝ)
  let b := (∛3 : ℝ)
  let expr := 1 / (a - b)
  let num := a^2 + a*b + b^2

  exists (25 : ℝ)
  exists (15 : ℝ)
  exists (9 : ℝ)
  exists (2 : ℝ)

  split
  { sorry, }
  { split
    { exact rfl, }
    { split
      { exact rfl, }
      { split
        { exact rfl, }
        { split
          { exact rfl, }
          { norm_num }}}}
}

end rationalize_denominator_l331_331773


namespace num_six_digit_even_numbers_l331_331254

theorem num_six_digit_even_numbers : 
  ∀ (s : finset ℕ), s = {1, 2, 3, 4, 5, 6} → 
  ∃ n : ℕ, 
    (∀ d ∈ s, d.to_nat ≥ 0 ∧ d.to_nat < 10) ∧
    (∀ x y : ℕ, x ∈ s → y ∈ s → x ≠ y) ∧ 
    (∀ l : list ℕ, l.length = 6 ∧ l.all (λ x, x ∈ s) → 
      l.nth 5 % 2 = 0 ∧ 
      (list.index_of 1 l ≤ 5 → abs (list.index_of 1 l - list.index_of 5 l) ≠ 1) ∧ 
      (list.index_of 3 l ≤ 5 → abs (list.index_of 3 l - list.index_of 5 l) ≠ 1)) →
    n = 240 := 
by
  sorry

end num_six_digit_even_numbers_l331_331254


namespace pirate_prob_exactly_four_treasure_no_traps_l331_331559

open ProbabilityMeasure

def probability_island_treasure_no_traps : ℚ := 3/10
def probability_island_traps_no_treasure : ℚ := 1/10
def probability_island_both_treasure_traps : ℚ := 1/5
def probability_island_neither_traps_nor_treasure : ℚ := 2/5

theorem pirate_prob_exactly_four_treasure_no_traps :
  ∀ (total_islands : ℕ) (islands_with_treasure_no_traps : ℕ),
    total_islands = 8 →
    islands_with_treasure_no_traps = 4 →
    (choose total_islands islands_with_treasure_no_traps : ℚ) *
    (probability_island_treasure_no_traps ^ islands_with_treasure_no_traps) *
    (probability_island_neither_traps_nor_treasure ^ (total_islands - islands_with_treasure_no_traps)) =
    9072 / 6250000 := sorry

end pirate_prob_exactly_four_treasure_no_traps_l331_331559


namespace find_slope_of_line_l331_331669

-- Define the parabola, point M, and the conditions leading to the slope k.
theorem find_slope_of_line (k : ℝ) :
  let C := {p : ℝ × ℝ | p.2^2 = 4 * p.1}
  let focus : ℝ × ℝ := (1, 0)
  let M : ℝ × ℝ := (-1, 1)
  let line (k : ℝ) (x : ℝ) := k * (x - 1)
  ∃ A B : (ℝ × ℝ), 
    A ∈ C ∧ B ∈ C ∧
    A ≠ B ∧
    A.1 + 1 = B.1 + 1 ∧ 
    A.2 - 1 = B.2 - 1 ∧
    ((A.1 + 1) * (B.1 + 1) + (A.2 - 1) * (B.2 - 1) = 0) -> k = 2 := 
by
  sorry

end find_slope_of_line_l331_331669


namespace vector_colinear_m_l331_331675

theorem vector_colinear_m :
  let a := (-3, 4)
  let b := (m, 2) in
  (2 * a.1 - 3 * b.1, 2 * a.2 - 3 * b.2) = (k * b.1, k * b.2) → 
  m = -3 / 2 :=
by
  let a : ℝ × ℝ := (-3, 4)
  let b : ℝ × ℝ := (m, 2)
  intro h
  sorry

end vector_colinear_m_l331_331675


namespace sequence_50th_term_l331_331364

-- Define the sequence formula
def sequence (n : ℕ) : ℕ := 5 * n - 3

-- Prove the 50th term in the sequence
theorem sequence_50th_term : sequence 50 = 247 := by
  simp [sequence]
  sorry

end sequence_50th_term_l331_331364


namespace tangent_line_equation_l331_331287

noncomputable def deriv (f : ℝ → ℝ) : ℝ → ℝ :=
λ x, Real.deriv f x

def curve (x : ℝ) : ℝ := x + Real.log x

def point_of_tangency := (1 : ℝ, 1 : ℝ)

theorem tangent_line_equation :
  ∃ m b : ℝ, (∀ x : ℝ, 2 * x - 1 = m * x + b) ∧ m = 2 ∧ b = -1 :=
begin
  sorry
end

end tangent_line_equation_l331_331287


namespace right_triangle_area_l331_331359

theorem right_triangle_area (x : ℝ) (h : 3 * x + 4 * x = 10) : 
  (1 / 2) * (3 * x) * (4 * x) = 24 :=
sorry

end right_triangle_area_l331_331359


namespace speed_second_half_l331_331898

theorem speed_second_half (total_time : ℝ) (first_half_speed : ℝ) (total_distance : ℝ) :
    total_time = 12 → first_half_speed = 35 → total_distance = 560 → 
    (280 / (12 - (280 / 35)) = 70) :=
by
  intros ht hf hd
  sorry

end speed_second_half_l331_331898


namespace area_bounded_by_circles_and_x_axis_l331_331938

/--
Circle C has its center at (5, 5) and radius 5 units.
Circle D has its center at (15, 5) and radius 5 units.
Prove that the area of the region bounded by these circles
and the x-axis is 50 - 25 * π square units.
-/
theorem area_bounded_by_circles_and_x_axis :
  let C_center := (5, 5)
  let D_center := (15, 5)
  let radius := 5
  (2 * (radius * radius) * π / 2) + (10 * radius) = 50 - 25 * π :=
sorry

end area_bounded_by_circles_and_x_axis_l331_331938


namespace right_triangle_side_lengths_l331_331590

theorem right_triangle_side_lengths (x : ℝ) :
  (2 * x + 2)^2 + (x + 2)^2 = (x + 4)^2 ∨ (2 * x + 2)^2 + (x + 4)^2 = (x + 2)^2 ↔ (x = 1 ∨ x = 4) :=
by sorry

end right_triangle_side_lengths_l331_331590


namespace female_employees_count_l331_331177

-- Define constants
def E : ℕ  -- Total number of employees
def M : ℕ := (2 / 5) * E  -- Total number of managers
def Male_E : ℕ  -- Total number of male employees
def Female_E : ℕ := E - Male_E  -- Total number of female employees
def Male_M : ℕ := (2 / 5) * Male_E  -- Total number of male managers
def Female_M : ℕ := 200  -- Total number of female managers

-- Given equation relating managers and employees
theorem female_employees_count : Female_E = 500 :=
by
  -- Required proof goes here
  sorry

end female_employees_count_l331_331177


namespace decreasing_interval_and_min_value_l331_331331

open Real

def f (x : ℝ) : ℝ := sin x + sqrt 3 * cos x

theorem decreasing_interval_and_min_value :
  (∀ x y ∈ Icc (π / 6) (π / 2), x < y → f y < f x) ∧
  (Inf (f '' Icc (-π / 2) (π / 2)) = -1) := by
  sorry

end decreasing_interval_and_min_value_l331_331331


namespace f_increasing_solve_a_range_m_l331_331667

-- Definitions
def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)
def g (x : ℝ) (m : ℝ) : ℝ := x^2 + 2 * m * x + 5 / 3

-- 1. Proving that f(x) is an increasing function on ℝ
theorem f_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 :=
sorry

-- 2. Finding the set of all real numbers a that satisfy f(2a - a^2) + f(3) > 0
theorem solve_a : ∀ a : ℝ, f (2 * a - a^2) + f 3 > 0 → -1 < a ∧ a < 3 :=
sorry

-- 3. Finding the range of values for m ensuring ∃ x2 ∈ [-1, 1] such that f(x1) = g(x2) for any x1 ∈ [-1, 1]
theorem range_m (m : ℝ) :
  (∀ x1 ∈ Set.Icc (-1 : ℝ) 1, ∃ x2 ∈ Set.Icc (-1 : ℝ) 1, f x1 = g x2 m) ↔
  m ≤ -3 / 2 ∨ m ≥ 3 / 2 :=
sorry

end f_increasing_solve_a_range_m_l331_331667


namespace least_perimeter_of_triangle_l331_331092

theorem least_perimeter_of_triangle (cosA cosB cosC : ℝ)
  (h₁ : cosA = 13 / 16)
  (h₂ : cosB = 4 / 5)
  (h₃ : cosC = -3 / 5) :
  ∃ a b c : ℕ, a + b + c = 28 ∧ 
  a^2 + b^2 - c^2 = 2 * a * b * cosC ∧ 
  b^2 + c^2 - a^2 = 2 * b * c * cosA ∧ 
  c^2 + a^2 - b^2 = 2 * c * a * cosB :=
sorry

end least_perimeter_of_triangle_l331_331092


namespace minimum_value_of_a_l331_331647

theorem minimum_value_of_a (a b : ℕ) (h₁ : b - a = 2013) 
(h₂ : ∃ x : ℕ, x^2 - a * x + b = 0) : a = 93 :=
sorry

end minimum_value_of_a_l331_331647


namespace intersection_M_N_l331_331673

def M := { y : ℝ | ∃ x : ℝ, y = 2 ^ x ∧ x > 0 }
def N := { x : ℝ | ∃ y : ℝ, y = log (2 * x - x ^ 2) }

theorem intersection_M_N : (∃ y, y ∈ M ∧ y ∈ N) ↔ (∃ x, 1 < x ∧ x < 2) :=
sorry

end intersection_M_N_l331_331673


namespace ef_length_l331_331705

theorem ef_length (FR RG : ℝ) (cos_ERH : ℝ) (h1 : FR = 12) (h2 : RG = 6) (h3 : cos_ERH = 1 / 5) : EF = 30 :=
by
  sorry

end ef_length_l331_331705


namespace xy_square_diff_l331_331055

theorem xy_square_diff (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end xy_square_diff_l331_331055


namespace set_membership_l331_331281

theorem set_membership :
  {m : ℤ | ∃ k : ℤ, 10 = k * (m + 1)} = {-11, -6, -3, -2, 0, 1, 4, 9} :=
by sorry

end set_membership_l331_331281


namespace problem_1_problem_2_problem_3_l331_331397

section
  variables {f : ℝ → ℝ} {n : ℕ} (h_cont_f : ContinuousOn f (Set.Icc 0 1))
  (h_int : ∀ k : ℕ, k < n → ∫ x in 0..1, x^k * f x = 0) {M : ℝ}
  (M_bound : ∀ x ∈ Set.Icc 0 1, abs (f x) ≤ M)

  -- Problem 1: Minimum value of g(t)
  def g (t : ℝ) : ℝ := ∫ x in 0..1, abs (x - t)^n

  theorem problem_1 : ∃ t : ℝ, (∀ s : ℝ, g s ≥ g t) ∧ g t = 2^(-(n+1)) / (n+1) :=
  sorry

  -- Problem 2: Show the equation holds for all real numbers t
  theorem problem_2 (t : ℝ) : ∫ x in 0..1, (x - t)^n * f x = ∫ x in 0..1, x^n * f x :=
  sorry

  -- Problem 3: Show the inequality
  theorem problem_3 : abs (∫ x in 0..1, x^n * f x) ≤ M / (2^n * (n + 1)) :=
  sorry
end

end problem_1_problem_2_problem_3_l331_331397


namespace B_work_days_l331_331893

/-- 
If A can complete a work in 12 days by themselves, and A and B together can complete the same work in 7.2 days, 
then B will complete the work alone in 18 days.
-/
theorem B_work_days (A_work_days : ℝ) (total_work_days : ℝ) : 
  A_work_days = 12 → total_work_days = 7.2 → 1 / 12 + 1 / (18 : ℝ) = 1 / 7.2 :=
by
  intros hAwork hTotal
  rw [←hAwork, ←hTotal]
  norm_num
  different_cast different_for exact sorry
 
end B_work_days_l331_331893


namespace exist_11_consecutive_nat_sum_perfect_cube_l331_331532

theorem exist_11_consecutive_nat_sum_perfect_cube :
  ∃ n : ℕ, let S := ∑ i in range 11, n + i in ∃ k : ℕ, S = k^3 :=
by {
  sorry
}

end exist_11_consecutive_nat_sum_perfect_cube_l331_331532


namespace curves_equations_and_distances_l331_331597

noncomputable def curve_C1_parametric : ℝ × ℝ → Prop :=
λ xy, ∃ φ : ℝ, (xy.1 = 2 * Real.cos φ) ∧ (xy.2 = Real.sin φ)

noncomputable def curve_C2_cartesian : ℝ × ℝ → Prop :=
λ xy, (xy.1) ^ 2 + (xy.2 - 3) ^ 2 = 1

theorem curves_equations_and_distances :
  (∀ xy : ℝ × ℝ, curve_C1_parametric xy ↔ (xy.1 ^ 2 / 4 + xy.2 ^ 2 = 1)) ∧
  (∀ xy : ℝ × ℝ, curve_C2_cartesian xy) ∧
  (∀ x1 y1 x2 y2 : ℝ, curve_C1_parametric (x1, y1) → curve_C2_cartesian (x2, y2) → 
    1 ≤ Real.dist (x1, y1) (x2, y2) ∧ Real.dist (x1, y1) (x2, y2) ≤ 5) :=
by sorry

end curves_equations_and_distances_l331_331597


namespace handshake_couples_l331_331556

/-
  A number of couples met and each person shook hands with everyone else present,
  but not with themselves or their partners. There were 31,000 handshakes altogether.
  Prove that there are 125 couples.
-/

theorem handshake_couples:
  ∃ c : ℕ, (2 * c ^ 2 - 2 * c = 31000) ∧ c = 125 :=
begin
  sorry
end

end handshake_couples_l331_331556


namespace width_of_road_correct_l331_331225

-- Define the given conditions
def sum_of_circumferences (r R : ℝ) : Prop := 2 * Real.pi * r + 2 * Real.pi * R = 88
def radius_relation (r R : ℝ) : Prop := r = (1/3) * R
def width_of_road (R r : ℝ) := R - r

-- State the main theorem
theorem width_of_road_correct (R r : ℝ) (h1 : sum_of_circumferences r R) (h2 : radius_relation r R) :
    width_of_road R r = 22 / Real.pi := by
  sorry

end width_of_road_correct_l331_331225


namespace sum_of_all_three_digit_positive_even_integers_l331_331522

def sum_of_three_digit_even_integers : ℕ :=
  let a := 100
  let l := 998
  let d := 2
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem sum_of_all_three_digit_positive_even_integers :
  sum_of_three_digit_even_integers = 247050 :=
by
  -- proof to be completed
  sorry

end sum_of_all_three_digit_positive_even_integers_l331_331522


namespace negation_of_some_primes_are_even_l331_331489

theorem negation_of_some_primes_are_even : 
  (¬ ∃ p : ℕ, prime p ∧ even p) ↔ (∀ p : ℕ, prime p → odd p) :=
by 
  sorry

end negation_of_some_primes_are_even_l331_331489


namespace evaluate_expressions_l331_331598

theorem evaluate_expressions :
  (3 ^ -3) ≠ 0 → 
  (∀ (x : ℝ), x ≠ 0 → x ^ 0 = 1) → 
  (3 ^ 0 = 1) → 
  (1 ^ 4 = 1) → 
  ((3 ^ -3) ^ 0 + (3 ^ 0) ^ 4 = 2) :=
by
  intros h1 h2 h3 h4
  calc
    (3 ^ -3) ^ 0 + (3 ^ 0) ^ 4
        = 1 + (3 ^ 0) ^ 4 : by rw [h2 (3 ^ -3) h1] 
    ... = 1 + 1       : by rw [h4, h3]
    ... = 2           : by norm_num

end evaluate_expressions_l331_331598


namespace grid_square_division_l331_331390

theorem grid_square_division (m n k : ℕ) (h : m * m = n * k) : ℕ := sorry

end grid_square_division_l331_331390


namespace A_alone_time_l331_331547

theorem A_alone_time (x : ℕ) (h1 : 3 * x / 4  = 12) : x / 3 = 16 := by
  sorry

end A_alone_time_l331_331547


namespace solveForX_l331_331201

theorem solveForX : ∃ (x : ℚ), x + 5/8 = 7/24 + 1/4 ∧ x = -1/12 := by
  sorry

end solveForX_l331_331201


namespace angle_of_inclination_l331_331284
open Real

theorem angle_of_inclination :
  let y := λ x : ℝ, (1 / 2) * x^2 - 2,
      dydx := λ x : ℝ, x in
  dydx 1 = 1 → 
  ∀ θ : ℝ, tan θ = 1 → θ = 45 := by
  sorry

end angle_of_inclination_l331_331284


namespace find_point_A_l331_331752

theorem find_point_A (x : ℝ) (h : x + 7 - 4 = 0) : x = -3 :=
sorry

end find_point_A_l331_331752


namespace rationalize_denominator_l331_331768

theorem rationalize_denominator :
  ∃ A B C D : ℝ, 
    (1 / (real.cbrt 5 - real.cbrt 3) = (real.cbrt A + real.cbrt B + real.cbrt C) / D) ∧ 
    A + B + C + D = 51 :=
  sorry

end rationalize_denominator_l331_331768


namespace difference_in_meaning_of_mozhet_l331_331381

theorem difference_in_meaning_of_mozhet (base height : ℝ) (triangle : Type) (orthocenter : triangle → triangle → Prop) :
  (∀ b h, ∃ area, area = 1 / 2 * b * h) ∧ (∃ t, orthocenter t t → ¬(t ∈ triangle)) :=
sorry

end difference_in_meaning_of_mozhet_l331_331381


namespace no_parallelepiped_possible_l331_331584

theorem no_parallelepiped_possible (x y z : ℕ) (h : x > 1 ∧ y > 1 ∧ z > 1) (volume : x * y * z = 1990) :
  ¬∃ shapes : list (ℕ × ℕ), length shapes = 48 ∧
    (∃ blocks : list (ℕ × ℕ × ℕ × ℕ), ∀ (b in blocks),
       (∃ a b c : ℕ, b = (a, b, c, d)) ∧ valid_shape b shapes) :=
sorry

def valid_shape (block : ℕ × ℕ × ℕ × ℕ) (shapes : list (ℕ × ℕ)) : Prop :=
  match block with
  | (a, b, c, d) => (a, b) ∈ shapes ∧ (c, d) ∈ shapes ∨
                    (b, c) ∈ shapes ∧ (a, d) ∈ shapes ∨
                    (c, d) ∈ shapes ∧ (a, b) ∈ shapes 

end no_parallelepiped_possible_l331_331584


namespace year_when_P_costs_40_paise_more_than_Q_l331_331835

def price_of_P (n : ℕ) : ℝ := 4.20 + 0.40 * n
def price_of_Q (n : ℕ) : ℝ := 6.30 + 0.15 * n

theorem year_when_P_costs_40_paise_more_than_Q :
  ∃ n : ℕ, price_of_P n = price_of_Q n + 0.40 ∧ 2001 + n = 2011 :=
by
  sorry

end year_when_P_costs_40_paise_more_than_Q_l331_331835


namespace solve_for_y_l331_331789

theorem solve_for_y (y : ℝ) : (1 / 8)^(3 * y + 12) = (32)^(3 * y + 7) → y = -71 / 24 := by
  sorry

end solve_for_y_l331_331789


namespace cost_of_gravel_l331_331033

-- Definitions based on conditions
def cost_per_cubic_foot : ℝ := 4
def cubic_yards_to_cubic_feet (y : ℝ) : ℝ := y * 27

-- Main statement to prove
theorem cost_of_gravel (y : ℝ) (cost_per_ft³ : ℝ) (conversion_rate : ℝ) (v := cubic_yards_to_cubic_feet y) 
: v = 8 * conversion_rate → cost_per_ft³ = 4 → conversion_rate = 27 → (v * cost_per_ft³ = 864) :=
by 
  intros h1 h2 h3
  rw [←h3] at h1
  rw [←h1]
  rw [h2]
  sorry

end cost_of_gravel_l331_331033


namespace Peter_initial_candies_eq_50_l331_331127

theorem Peter_initial_candies_eq_50 (x : ℕ) :
  (let day1_candies := (x / 2) - 3 in
   let day2_candies := (day1_candies / 2) - 5 in
   day2_candies = 6) →
  x = 50 :=
begin
  intros h,
  sorry,
end

end Peter_initial_candies_eq_50_l331_331127


namespace Jessie_initial_weight_l331_331392

def lost_first_week : ℕ := 56
def after_first_week : ℕ := 36

theorem Jessie_initial_weight :
  (after_first_week + lost_first_week = 92) :=
by
  sorry

end Jessie_initial_weight_l331_331392


namespace pool_capacity_l331_331702

variables {T : ℕ} {A B C : ℕ → ℕ}

-- Conditions
def valve_rate_A (T : ℕ) : ℕ := T / 180
def valve_rate_B (T : ℕ) := valve_rate_A T + 60
def valve_rate_C (T : ℕ) := valve_rate_A T + 75

def combined_rate (T : ℕ) := valve_rate_A T + valve_rate_B T + valve_rate_C T

-- Theorem to prove
theorem pool_capacity (T : ℕ) (h1 : combined_rate T = T / 40) : T = 16200 :=
by
  sorry

end pool_capacity_l331_331702


namespace g_value_l331_331354

def g (x : ℝ) : ℝ := sorry  -- placeholder for the actual function definition

theorem g_value (h : ∀ x : ℝ, g (3 * x - 4) = 4 * x + 6) : g 2 = 14 :=
by
  -- Using given condition h
  have h_eq : g (3 * 2 - 4) = 4 * 2 + 6 := h 2
  -- Simplify the expressions inside
  simp at h_eq
  -- Assert the desired property
  exact h_eq

end g_value_l331_331354


namespace round_to_nearest_hundred_l331_331679

theorem round_to_nearest_hundred (n : ℕ) (h : n = 1996) : (Nat.round_up n 100 = 2000) :=
sorry

end round_to_nearest_hundred_l331_331679


namespace inequality_problem_l331_331996

variable {a b c d : ℝ}

theorem inequality_problem (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
    a + c > b + d ∧ ad^2 > bc^2 ∧ (1 / bc) < (1 / ad) :=
by
  sorry

end inequality_problem_l331_331996


namespace part_A_part_B_part_C_l331_331000

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f(x * y) = y^2 * f(x) + x^2 * f(y)

theorem part_A : f(0) = 0 := sorry
theorem part_B : f(1) = 0 := sorry
theorem part_C : ∀ x : ℝ, f(x) = f(-x) := sorry

end part_A_part_B_part_C_l331_331000


namespace mark_dice_count_l331_331117

theorem mark_dice_count :
  ∀ (m_total j_total m_percent j_percent needed bought : ℕ),
  m_percent = 60 →
  j_total = 8 →
  j_percent = 75 →
  needed = 14 →
  bought = 2 →
  (j_percent * j_total) / 100 + (m_percent * m_total) / 100 + bought = needed →
  m_total = 10 := 
by 
  intros m_total j_total m_percent j_percent needed bought hm_percent hj_total hj_percent hneeded hbought h_eqn,
  sorry

end mark_dice_count_l331_331117


namespace triangle_angle_B_l331_331387

theorem triangle_angle_B (A B C : ℕ) (h₁ : B + C = 110) (h₂ : A + B + C = 180) (h₃ : A = 70) :
  B = 70 ∨ B = 55 ∨ B = 40 :=
by
  sorry

end triangle_angle_B_l331_331387


namespace can_move_to_upper_left_can_move_to_upper_right_l331_331712

-- Initialize the conditions for the chessboard and the pieces
def Chessboard := { position : Fin 8 × Fin 8 // 
  ∀ (x₁ x₂ : position), 
  ∃ (movement : (Fin 8 × Fin 8) → Prop), 
  movement x₁ → movement x₂ }

structure InitialConfig where
  board : Chessboard
  initial_pieces : Fin 8 × Fin 8 → Prop

-- Statement for moving to the upper left corner
theorem can_move_to_upper_left (initial : InitialConfig) : 
  ∃ (final_config : InitialConfig), 
  final_config.initial_pieces = -- Conditions about pieces being in upper left
  sorry

-- Statement for moving to the upper right corner
theorem can_move_to_upper_right (initial : InitialConfig) :
  ∃ (final_config : InitialConfig), 
  final_config.initial_pieces = -- Conditions about pieces being in upper right
  sorry

end can_move_to_upper_left_can_move_to_upper_right_l331_331712


namespace arithmetic_square_root_of_second_consecutive_l331_331514

theorem arithmetic_square_root_of_second_consecutive (x : ℝ) : 
  real.sqrt (x^2 + 1) = real.sqrt (x^2 + 1) :=
sorry

end arithmetic_square_root_of_second_consecutive_l331_331514


namespace certain_number_is_213_l331_331688

theorem certain_number_is_213 (x : ℝ) (h1 : x * 16 = 3408) (h2 : x * 1.6 = 340.8) : x = 213 :=
sorry

end certain_number_is_213_l331_331688


namespace sale_in_fifth_month_l331_331546

theorem sale_in_fifth_month (sale1 sale2 sale3 sale4 sale6 avg_sale num_months total_sales known_sales_five_months sale5: ℕ) :
  sale1 = 6400 →
  sale2 = 7000 →
  sale3 = 6800 →
  sale4 = 7200 →
  sale6 = 5100 →
  avg_sale = 6500 →
  num_months = 6 →
  total_sales = avg_sale * num_months →
  known_sales_five_months = sale1 + sale2 + sale3 + sale4 + sale6 →
  sale5 = total_sales - known_sales_five_months →
  sale5 = 6500 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end sale_in_fifth_month_l331_331546


namespace digits_base8_sum_l331_331036

open Nat

theorem digits_base8_sum (X Y Z : ℕ) (hX : 0 < X) (hY : 0 < Y) (hZ : 0 < Z) 
  (h_distinct : X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z) (h_base8 : X < 8 ∧ Y < 8 ∧ Z < 8) 
  (h_eq : (8^2 * X + 8 * Y + Z) + (8^2 * Y + 8 * Z + X) + (8^2 * Z + 8 * X + Y) = 8^3 * X + 8^2 * X + 8 * X) : 
  Y + Z = 7 :=
by
  sorry

end digits_base8_sum_l331_331036


namespace bela_wins_if_and_only_if_n_is_odd_l331_331252

theorem bela_wins_if_and_only_if_n_is_odd (n : ℕ) (hn : 6 < n) : 
  (∃ x ∈ set.Icc 0 n, ∀ y ∈ set.range(λ (m : ℕ), x + m * 4 + 2), y ∉ set.Icc 0 n)
  ↔ odd n :=
sorry

end bela_wins_if_and_only_if_n_is_odd_l331_331252


namespace determine_speed_l331_331557

theorem determine_speed :
  ∀ {d : ℝ} {t : ℝ},
  d = 120 →
  t = 4 →
  let total_distance := 2 * d in
  let speed := total_distance / t in
  speed = 60 := by
    intros d t h_d h_t
    simp [h_d, h_t]
    sorry

end determine_speed_l331_331557


namespace polynomial_coefficient_sum_l331_331351

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (1 - 2 * x) ^ 4 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 →
  a₀ + a₁ + a₃ = -39 :=
by
  sorry

end polynomial_coefficient_sum_l331_331351


namespace best_deal_around_the_world_l331_331468

def polina_age : ℕ := 5
def total_people : ℕ := 3

-- Costs for Globus
def globus_per_person_age_above_5 : ℕ := 25400
def globus_discount_percentage : ℕ := 2

-- Costs for Around the World
def around_the_world_cost_age_below_6 : ℕ := 11400
def around_the_world_cost_age_above_6 : ℕ := 23500
def around_the_world_commission_percentage : ℕ := 1

noncomputable def total_cost_globus : ℕ :=
  let initial_cost := total_people * globus_per_person_age_above_5 in
  let discount := initial_cost * globus_discount_percentage / 100 in
  initial_cost - discount

noncomputable def total_cost_around_the_world : ℕ :=
  let initial_cost := around_the_world_cost_age_below_6 + 2 * around_the_world_cost_age_above_6 in
  let commission := initial_cost * around_the_world_commission_percentage / 100 in
  initial_cost + commission

theorem best_deal_around_the_world :
  total_cost_around_the_world < total_cost_globus ∧ total_cost_around_the_world = 58984 := 
  by
    sorry

end best_deal_around_the_world_l331_331468


namespace distance_between_points_l331_331605

theorem distance_between_points : 
  let p1 := (3 : ℝ, 3 : ℝ) in
  let p2 := (-2 : ℝ, -2 : ℝ) in
  dist p1 p2 = 5 * Real.sqrt 2 :=
by
  sorry

end distance_between_points_l331_331605


namespace totalCheeseSlices_l331_331393

-- Define the number of slices of cheese each type of sandwich requires
def slicesPerHamSandwich : Nat := 2
def slicesPerGrilledCheeseSandwich : Nat := 3

-- Define the number of each type of sandwiches Joan makes
def numHamSandwiches : Nat := 10
def numGrilledCheeseSandwiches : Nat := 10

-- Define total cheese slices for each type of sandwich
def totalCheeseForHamSandwiches : Nat := numHamSandwiches * slicesPerHamSandwich := by
  unfold slicesPerHamSandwich numHamSandwiches
  exact 20

def totalCheeseForGrilledCheeseSandwiches : Nat := numGrilledCheeseSandwiches * slicesPerGrilledCheeseSandwich := by
  unfold slicesPerGrilledCheeseSandwich numGrilledCheeseSandwiches
  exact 30

-- Prove the total number of cheese slices used
theorem totalCheeseSlices : totalCheeseForHamSandwiches + totalCheeseForGrilledCheeseSandwiches = 50 := by
  unfold totalCheeseForHamSandwiches totalCheeseForGrilledCheeseSandwiches
  exact (20 + 30)
  sorry

end totalCheeseSlices_l331_331393


namespace daily_evaporation_rate_l331_331901

theorem daily_evaporation_rate
  (initial_water : ℝ)
  (evaporation_rate_percent : ℝ)
  (total_days : ℕ)
  (evaporated_amount: ℝ) 
  (daily_evaporation_correct: evaporated_amount / total_days = 0.4 / 20):
  (initial_water = 10) ∧ (evaporation_rate_percent = 4) ∧ (total_days = 20) →
  evaporated_amount = 0.02 :=
by
  intro h,
  cases h,
  have h1 : evaporated_amount = 0.4,
    calc evaporated_amount
      = evaporation_rate_percent / 100 * initial_water : by sorry
      ... = 0.4 : by sorry,
  have h2 : evaporated_amount / total_days = 0.02,
    calc evaporated_amount / total_days
      = 0.4 / 20 : by sorry
      ... = 0.02 : by sorry,
  exact h2,
sorry

end daily_evaporation_rate_l331_331901


namespace female_avg_combined_is_84_l331_331798

variables (a b c d : ℕ)

/- Conditions -/
def condition1 : Prop := (71 * a + 76 * b) / (a + b) = 74
def condition2 : Prop := (81 * c + 90 * d) / (c + d) = 84
def condition3 : Prop := (71 * a + 81 * c) / (a + c) = 79

/- Question -/
def female_avg_combined := (76 * b + 90 * d) / (b + d)

/- Theorems -/
theorem female_avg_combined_is_84 (h1 : condition1) (h2 : condition2) (h3 : condition3) : female_avg_combined b d = 84 :=
sorry

end female_avg_combined_is_84_l331_331798


namespace monotonic_decreasing_a_ge_one_l331_331690

noncomputable def f (a x : ℝ) : ℝ := x^3 - a * x^2 - x + 6

theorem monotonic_decreasing_a_ge_one
  (a : ℝ)
  (h_decreasing : ∀ x ∈ Ioo (0 : ℝ) 1, (deriv (f a)) x < 0) :
  1 ≤ a :=
sorry

end monotonic_decreasing_a_ge_one_l331_331690


namespace apples_used_l331_331540

theorem apples_used (apples_before : ℕ) (apples_left : ℕ) (apples_used_for_pie : ℕ) 
                    (h1 : apples_before = 19) 
                    (h2 : apples_left = 4) 
                    (h3 : apples_used_for_pie = apples_before - apples_left) : 
  apples_used_for_pie = 15 :=
by
  -- Since we are instructed to leave the proof out, we put sorry here
  sorry

end apples_used_l331_331540


namespace hyperbola_eccentricity_l331_331158

theorem hyperbola_eccentricity {a b c : ℝ} (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) :
  let p := 2 * c in
  let M := ∀ p x y, (x ^ 2) / (a ^ 2) - (y ^ 2) / (b ^ 2) = 1 in
  let N := y ^ 2 = 2 * p * x in
  let F2 := (c, 0) in
  let P := (c, y) in
  let midpoint_PF1 := (0, y) in
  let PF2 := 2 * c in
  let PF1 := 2 * Real.sqrt 2 * c in
  let hyperbola_def := PF1 - PF2 = 2 * a in
  ∀ e, e = c / a → e = Real.sqrt 2 + 1 :=
sorry

end hyperbola_eccentricity_l331_331158


namespace value_range_MA_MB_MC_MD_l331_331371

section Geometry
variable (A B C D M : Type)
variable [HasCoords A] [HasCoords B] [HasCoords C] [HasCoords D] [HasCoords M]

noncomputable def curve_C1 := { p : ℝ × ℝ // p.1^2 + p.2^2 = 4 }
noncomputable def C1_coordinates_A : curve_C1 := (⟨2 * cos (π / 6), 2 * sin (π / 6)⟩)
noncomputable def C1_coordinates_B : curve_C1 := (⟨2 * cos (5 * π / 6), 2 * sin (5 * π / 6)⟩)
noncomputable def C_coordinates_C := (⟨-√3, -1⟩)
noncomputable def D_coordinates_D := (⟨√3, -1⟩)

noncomputable def curve_C2 := { p : ℝ × ℝ // p.1^2 + 4 * p.2^2 = 4 }
def coordinates_M (θ : ℝ) : curve_C2 := (⟨2 * cos θ, sin θ⟩)

theorem value_range_MA_MB_MC_MD (θ : ℝ) : (20 ≤ |coordinates_M θ - C1_coordinates_A|^2 + |coordinates_M θ - C1_coordinates_B|^2 + |coordinates_M θ - C_coordinates_C|^2 + |coordinates_M θ - D_coordinates_D|^2) ∧ 
(|coordinates_M θ - C1_coordinates_A|^2 + |coordinates_M θ - C1_coordinates_B|^2 + |coordinates_M θ - C_coordinates_C|^2 + |coordinates_M θ - D_coordinates_D|^2 ≤ 32) :=
sorry
end Geometry

end value_range_MA_MB_MC_MD_l331_331371


namespace find_point_A_l331_331753

theorem find_point_A (x : ℝ) (h : x + 7 - 4 = 0) : x = -3 :=
sorry

end find_point_A_l331_331753


namespace find_a_l331_331064

theorem find_a (a x : ℝ) (h1 : 2 * (x - 1) - 6 = 0) (h2 : 1 - (3 * a - x) / 3 = 0) (h3 : x = 4) : a = -1 / 3 :=
by
  sorry

end find_a_l331_331064


namespace part1_part2_l331_331014

variables (l : ℝ → ℝ) -- Line l represented by a function from ℝ to ℝ
variables (c slope : ℝ) (A : ℝ)
noncomputable def slope := -1
def point_p := (2, 2)
def area := 12

theorem part1 (h_slope : ∀ x, l x = slope * x + c)
    (h_p : l (point_p.1) = point_p.2) :
    ∃ c, ∀ x, l x = slope * x + 4 :=
sorry

theorem part2 (h_slope : ∀ x, l x = slope * x + c)
    (h_area : (1 / 2) * c^2 = area) :
    ∃ c, ∀ x, l x = slope * x + 2 * real.sqrt 6 ∨ l x = slope * x - 2 * real.sqrt 6 :=
sorry

end part1_part2_l331_331014


namespace best_deal_around_the_world_l331_331467

def polina_age : ℕ := 5
def total_people : ℕ := 3

-- Costs for Globus
def globus_per_person_age_above_5 : ℕ := 25400
def globus_discount_percentage : ℕ := 2

-- Costs for Around the World
def around_the_world_cost_age_below_6 : ℕ := 11400
def around_the_world_cost_age_above_6 : ℕ := 23500
def around_the_world_commission_percentage : ℕ := 1

noncomputable def total_cost_globus : ℕ :=
  let initial_cost := total_people * globus_per_person_age_above_5 in
  let discount := initial_cost * globus_discount_percentage / 100 in
  initial_cost - discount

noncomputable def total_cost_around_the_world : ℕ :=
  let initial_cost := around_the_world_cost_age_below_6 + 2 * around_the_world_cost_age_above_6 in
  let commission := initial_cost * around_the_world_commission_percentage / 100 in
  initial_cost + commission

theorem best_deal_around_the_world :
  total_cost_around_the_world < total_cost_globus ∧ total_cost_around_the_world = 58984 := 
  by
    sorry

end best_deal_around_the_world_l331_331467


namespace base_salary_is_1600_l331_331395

theorem base_salary_is_1600 (B : ℝ) (C : ℝ) (sales : ℝ) (fixed_salary : ℝ) :
  C = 0.04 ∧ sales = 5000 ∧ fixed_salary = 1800 ∧ (B + C * sales = fixed_salary) → B = 1600 :=
by sorry

end base_salary_is_1600_l331_331395


namespace triangle_inequality_l331_331726

theorem triangle_inequality (A B C D E F : Point) (hABC : triangle A B C) 
  (hD : midpoint D B C) (hE : on_line E A C) (hF : on_line F B)
  (hDE_DF : dist D E = dist D F) (hAngle : angle D E F = angle A B C) :
  dist D E ≥ (dist A B + dist A C) / 4 :=
by {sorry}

end triangle_inequality_l331_331726


namespace distance_to_nearest_edge_l331_331908

theorem distance_to_nearest_edge (wall_width picture_width : ℕ) (h1 : wall_width = 19) (h2 : picture_width = 3) (h3 : 2 * x + picture_width = wall_width) :
  x = 8 :=
by
  sorry

end distance_to_nearest_edge_l331_331908


namespace problem_statement_l331_331406

   noncomputable def g (x : ℝ) : ℝ := (3 * x + 4) / (x + 3)

   def T := {y : ℝ | ∃ (x : ℝ), x ≥ 0 ∧ y = g x}

   theorem problem_statement :
     (∃ N, (∀ y ∈ T, y ≤ N) ∧ N = 3 ∧ N ∉ T) ∧
     (∃ n, (∀ y ∈ T, y ≥ n) ∧ n = 4/3 ∧ n ∈ T) :=
   by
     sorry
   
end problem_statement_l331_331406


namespace number_line_problem_l331_331755

theorem number_line_problem (x : ℤ) (h : x + 7 - 4 = 0) : x = -3 :=
by
  -- The proof is omitted as only the statement is required.
  sorry

end number_line_problem_l331_331755


namespace mary_final_weight_l331_331748

def initial_weight : Int := 99
def weight_lost_initially : Int := 12
def weight_gained_back_twice_initial : Int := 2 * weight_lost_initially
def weight_lost_thrice_initial : Int := 3 * weight_lost_initially
def weight_gained_back_half_dozen : Int := 12 / 2

theorem mary_final_weight :
  let final_weight := 
      initial_weight 
      - weight_lost_initially 
      + weight_gained_back_twice_initial 
      - weight_lost_thrice_initial 
      + weight_gained_back_half_dozen
  in final_weight = 78 :=
by
  sorry

end mary_final_weight_l331_331748


namespace solve_for_x_l331_331686

theorem solve_for_x (x t : ℝ)
  (h₁ : t = 9)
  (h₂ : (3 * (x + 5)) / 4 = t + (3 - 3 * x) / 2) :
  x = 3 :=
by
  sorry

end solve_for_x_l331_331686


namespace range_b_over_c_l331_331076

variable {α : Type*}
variables {A B C : α} {a b c S : ℝ}

def acuteTriangle (A : α) : Prop := sorry -- definition of acute angle

def triangleArea (a b c : ℝ) : ℝ := (1/2) * b * c * (real.sin A)

-- conditions
axiom sides_of_triangle (h : acuteTriangle A) : real.is_positive a ∧ real.is_positive b ∧ real.is_positive c
axiom area_of_triangle (h : acuteTriangle A) : S = triangleArea a b c
axiom given_condition (h : acuteTriangle A) [h_area : area_of_triangle h] : 2 * S = a^2 - (b - c)^2

-- theorem to prove
theorem range_b_over_c (h : acuteTriangle A) [h_sides : sides_of_triangle h] [h_area : area_of_triangle h] [h_cond : given_condition h]:
  (3/5 : ℝ) < b / c ∧ b / c < 5/3 :=
sorry

end range_b_over_c_l331_331076


namespace central_cell_value_l331_331380

theorem central_cell_value (numbers : Fin 9 → Nat)
    (is_grid : ∀ n : Nat, n < 9 → ∃ i j : Fin 3, numbers ⟨n, by linarith⟩ = i * 3 + j)
    (adjacent : ∀ n m : Fin 9, numbers n + 1 = numbers m → (abs ((numbers n) / 3 - (numbers m) / 3) + 
                                                           abs ((numbers n) % 3 - (numbers m) % 3) = 1))
    (corner_sum : numbers ⟨0, sorry⟩ + numbers ⟨2, sorry⟩ 
                 + numbers ⟨6, sorry⟩ + numbers ⟨8, sorry⟩ = 18) :
    numbers ⟨4, sorry⟩ = 2 :=
by
  sorry

end central_cell_value_l331_331380


namespace rhombus_area_eq_88_l331_331372

theorem rhombus_area_eq_88 : 
  let A := (0 : ℝ, 5.5 : ℝ)
  let B := (8 : ℝ, 0 : ℝ)
  let C := (0 : ℝ, -5.5 : ℝ)
  let D := (-8 : ℝ, 0 : ℝ)
  let d1 := dist A C
  let d2 := dist B D
  (d1 * d2) / 2 = 88 :=
by
  let A := (0, 5.5)
  let B := (8, 0)
  let C := (0, -5.5)
  let D := (-8, 0)
  let d1 := dist A C
  let d2 := dist B D
  sorry

end rhombus_area_eq_88_l331_331372


namespace ways_to_fill_table_l331_331959

-- Problem statement in Lean
theorem ways_to_fill_table :
  let even_positions := (4.choose 2) * (3 + (3 * 3))
  let odd_positions := (8.choose 4)^2
  even_positions * odd_positions = 441000 :=
by
  sorry

end ways_to_fill_table_l331_331959


namespace invalid_votes_percentage_l331_331082

def total_votes : ℕ := 560000
def valid_votes_A : ℕ := 357000
def percentage_A : ℝ := 0.75
def invalid_percentage (x : ℝ) : Prop := (percentage_A * (1 - x / 100) * total_votes = valid_votes_A)

theorem invalid_votes_percentage : ∃ x : ℝ, invalid_percentage x ∧ x = 15 :=
by 
  use 15
  unfold invalid_percentage
  sorry

end invalid_votes_percentage_l331_331082


namespace fraction_of_students_who_walk_l331_331248

def fraction_by_bus : ℚ := 2 / 5
def fraction_by_car : ℚ := 1 / 5
def fraction_by_scooter : ℚ := 1 / 8
def total_fraction_not_walk := fraction_by_bus + fraction_by_car + fraction_by_scooter

theorem fraction_of_students_who_walk :
  (1 - total_fraction_not_walk) = 11 / 40 :=
by
  sorry

end fraction_of_students_who_walk_l331_331248


namespace largest_real_number_lambda_l331_331639

noncomputable def max_lambda (n : ℕ) (hn : n ≥ 2) (z : ℕ → ℂ) (hz : ∀ k, k < n → |z k| ≤ 1) : ℝ :=
  (n-1 : ℝ) ^ (-(n-1 : ℝ) / n)

theorem largest_real_number_lambda (n : ℕ) (hn : n ≥ 2)
  (z : ℕ → ℂ) (hz : ∀ k, k < n → |(z k)| ≤ 1) :
  ∃ λ, λ = max_lambda n hn z hz ∧
    1 + ∑ k in finset.range n, ∥ ∑ 1 ≤ j_1 < j_2 < ... < j_k ≤ n, (∏ i in range k, z (j_i)) ∥
    ≥ λ * ∑ k in finset.range n, |z k| := 
sorry

end largest_real_number_lambda_l331_331639


namespace percentage_invalid_votes_l331_331085

theorem percentage_invalid_votes 
    (total_votes : ℕ)
    (candidate_A_votes : ℕ)
    (candidate_A_percentage : ℝ)
    (total_valid_percentage : ℝ) :
    total_votes = 560000 ∧
    candidate_A_votes = 357000 ∧
    candidate_A_percentage = 0.75 ∧
    total_valid_percentage = 100 - x ∧
    (0.75 * (total_valid_percentage / 100) * 560000 = 357000) →
    x = 15 :=
by
  sorry

end percentage_invalid_votes_l331_331085


namespace div_neg_cancel_neg_div_example_l331_331940

theorem div_neg_cancel (x y : Int) (h : y ≠ 0) : (-x) / (-y) = x / y := by
  sorry

theorem neg_div_example : (-64 : Int) / (-32) = 2 := by
  apply div_neg_cancel
  norm_num

end div_neg_cancel_neg_div_example_l331_331940


namespace range_of_a_l331_331739

open Set

variable (a x : ℝ)

def U := @univ ℝ
def A := {x : ℝ | abs (x - a) < 1}
def B := {x : ℝ | (x + 1) / (x - 2) ≤ 2}

theorem range_of_a (a : ℝ) (h : A ⊆ compl B) : 3 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l331_331739


namespace train_length_correct_l331_331241

noncomputable def length_of_train (speed_train_kmph : ℕ) (time_to_cross_bridge_sec : ℝ) (length_of_bridge_m : ℝ) : ℝ :=
let speed_train_mps := (speed_train_kmph : ℝ) * (1000 / 3600)
let total_distance := speed_train_mps * time_to_cross_bridge_sec
total_distance - length_of_bridge_m

theorem train_length_correct :
  length_of_train 90 32.99736021118311 660 = 164.9340052795778 :=
by
  have speed_train_mps : ℝ := 90 * (1000 / 3600)
  have total_distance := speed_train_mps * 32.99736021118311
  have length_of_train := total_distance - 660
  exact sorry

end train_length_correct_l331_331241


namespace hyperbola_equation_distance_between_points_l331_331668

namespace HyperbolaProof

-- Define the hyperbola C with given conditions
def hyperbola_C (x y : ℝ) (a b : ℝ) : Prop :=
  (a = 1) ∧ (b = 2) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

-- Define the line intersecting condition
def intersecting_line (x y : ℝ) : Prop :=
  (y = x + 2)

-- Define the distance between points A and B on the hyperbola
def distance_AB (a b : ℝ) : ℝ :=
  (4 * Real.sqrt 14) / 3

-- Theorem to prove the hyperbola equation
theorem hyperbola_equation :
  ∃ a b : ℝ, (2 * a = 2) ∧ (Real.sqrt 5)^2 = a^2 + b^2 ∧ (a = 1) ∧ (b = 2) := 
by
  sorry

-- Theorem to prove the distance between A and B
theorem distance_between_points :
  let h : ℝ := 1 in
  let k : ℝ := 2 in
  ∃ a b x1 x2 y1 y2 : ℝ, 
    hyperbola_C x1 y1 a b ∧ hyperbola_C x2 y2 a b ∧ 
    intersecting_line x1 y1 ∧ intersecting_line x2 y2 ∧ 
    (3*x1^2 - 4*x1 - 8 = 0) ∧ (3*x2^2 - 4*x2 - 8 = 0) ∧
    (|x1 - x2| = distance_AB a b) := 
by
  sorry

end HyperbolaProof

end hyperbola_equation_distance_between_points_l331_331668


namespace projectile_reaches_100_feet_l331_331815

noncomputable def projectile_height (t : ℝ) : ℝ :=
  -16 * t^2 + 80 * t

theorem projectile_reaches_100_feet :
  ∃ t : ℝ, t = 2.5 ∧ projectile_height t = 100 :=
by
  use 2.5
  sorry

end projectile_reaches_100_feet_l331_331815


namespace factorization_correct_l331_331950

theorem factorization_correct (x : ℝ) :
  (16 * x^6 + 36 * x^4 - 9) - (4 * x^6 - 12 * x^4 + 3) = 12 * (x^6 + 4 * x^4 - 1) := by
  sorry

end factorization_correct_l331_331950


namespace minimally_competent_subsets_count_l331_331494

def A : Finset ℕ := Finset.range 11 \{0}

def is_competent (s : Finset ℕ) (n : ℕ) : Prop :=
  n ∈ s ∧ s.card = n

def is_minimally_competent (s : Finset ℕ) : Prop :=
  ∃ n, is_competent s n ∧ ∀ t ⊂ s, ¬ is_competent t n

theorem minimally_competent_subsets_count :
  (Finset.filter is_minimally_competent (Finset.powerset A)).card = 129 :=
sorry

end minimally_competent_subsets_count_l331_331494


namespace a_formula_b_formula_b_less_than_a_l331_331656

-- Definitions based on the problem conditions
def S (n : ℕ) : ℝ := n / (n + 1)

def a (n : ℕ) : ℝ := 1 / (n * (n + 1))

def b : ℕ → ℝ
| 1       := 1
| (n + 1) := let bn := b n in bn / (bn + 2)

-- Proof statements based on the correct answers
theorem a_formula (n : ℕ) (h : n ≥ 1) : a n = S n - S (n - 1) :=
sorry

theorem b_formula (n : ℕ) : b n = 1 / (2^n - 1) :=
sorry

theorem b_less_than_a (n : ℕ) (h : n ≥ 1) : b (n + 1) < a n :=
sorry

end a_formula_b_formula_b_less_than_a_l331_331656


namespace expected_area_convex_hull_correct_l331_331980

def point_placement (x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 10

def convex_hull_area (points : Finset (ℕ × ℤ)) : ℚ := 
  -- Definition of the area calculation goes here. This is a placeholder.
  0  -- Placeholder for the actual calculation

noncomputable def expected_convex_hull_area : ℚ := 
  -- Calculation of the expected area, which is complex and requires integration of the probability.
  sorry  -- Placeholder for the actual expected value

theorem expected_area_convex_hull_correct : 
  expected_convex_hull_area = 1793 / 128 :=
sorry

end expected_area_convex_hull_correct_l331_331980


namespace number_of_ways_to_change_l331_331418

theorem number_of_ways_to_change :
  (∃ n d q : ℕ,
    n > 0 ∧ d > 0 ∧ q > 0 ∧
    n + 2 * d + 5 * q = 400 ∧
    (n + 2 * d + 5 * q).count = 280) :=
sorry

end number_of_ways_to_change_l331_331418


namespace subcommittee_combinations_l331_331221

open Nat

theorem subcommittee_combinations :
  (choose 8 3) * (choose 6 2) = 840 := by
  sorry

end subcommittee_combinations_l331_331221


namespace cos_square_sub_exp_zero_l331_331882

theorem cos_square_sub_exp_zero : 
  (cos (30 * Real.pi / 180))^2 - (2 - Real.pi) ^ 0 = -1 / 4 := by
  sorry

end cos_square_sub_exp_zero_l331_331882


namespace FG_half_AB_l331_331106

structure Triangle (A B C : Type*) :=
(right_angle_at_C : ∠ B C A = 90)
(circumcenter_U : Point)
(points_D_E : ∀ (D : Point on AC) (E : Point on BC), ∠ E U D = 90)
(feet_F_G : ∀ (F : Foot P D to AB) (G : Foot P E to AB), True)

theorem FG_half_AB (A B C U D E F G : Point) (T : Triangle A B C) :
  (F G : Segment) = 1/2 * (A B : Segment) :=
by
  -- Proof skipped
  sorry

end FG_half_AB_l331_331106


namespace mary_final_weight_l331_331744

theorem mary_final_weight : 
  let initial_weight := 99
  let weight_loss1 := 12
  let weight_gain1 := 2 * weight_loss1
  let weight_loss2 := 3 * weight_loss1
  let weight_gain2 := 6
  initial_weight - weight_loss1 + weight_gain1 - weight_loss2 + weight_gain2 = 81 := by 
  sorry

end mary_final_weight_l331_331744


namespace ming_dynasty_wine_problem_l331_331709

theorem ming_dynasty_wine_problem :
  ∃ (x y : ℝ), x + y = 19 ∧ 3 * x + 1/3 * y = 33 :=
by
  use 19 - y, y
  split
  · sorry
  · sorry

end ming_dynasty_wine_problem_l331_331709


namespace third_number_is_seven_l331_331528

-- Define the conditions
def HCF := gcd (gcd 136 144) x = 8
def LCM := lcm (lcm 136 144) x = 2^4 * 3^2 * 17 * 7

-- The proof statement
theorem third_number_is_seven (x : ℕ) (hcf_condition : HCF) (lcm_condition : LCM) : x = 7 :=
by
  sorry

end third_number_is_seven_l331_331528


namespace min_distance_sum_l331_331939

open Real

theorem min_distance_sum {M N P : Point ℝ} :
  (∀ M : Point ℝ, dist M (center C1) = radius C1 ∧
  (∀ N : Point ℝ, dist N (center C2) = radius C2 ∧
  (∀ P : Point ℝ, P.y = -1 → 
  dist P M + dist P N ≥ 5 * sqrt 2 - 4)) :=
begin
  -- the proof goes here
  sorry
end

end min_distance_sum_l331_331939


namespace range_of_a_l331_331650

variable {α : Type*} [OrderedCommSemiring α] (f : α → α)

def is_even (f : α → α) : Prop :=
  ∀ x, f x = f (-x)

def is_mono_increasing_on (f : α → α) (s : Set α) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem range_of_a
  (h_even : is_even f)
  (h_mono_incr : is_mono_increasing_on f (Set.Iio 0))
  (h_ineq : ∀ a, f (2^(a-1)) > f (-(Real.sqrt 2))) :
  1 < a ∧ a < 3/2 := by
  sorry

end range_of_a_l331_331650


namespace smallest_digit_sum_l331_331687

theorem smallest_digit_sum (n : ℕ) (h : n > 0) : (∃ k : ℕ, (3 * n^2 + n + 1).digits.sum = k) ∧ k = 3 :=
by
  sorry

end smallest_digit_sum_l331_331687


namespace sequence_sum_l331_331089

-- Define the arithmetic sequence and conditions
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific values used in the problem
def specific_condition (a : ℕ → ℝ) : Prop :=
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450)

-- The proof goal that needs to be established
theorem sequence_sum (a : ℕ → ℝ) (h1 : arithmetic_seq a) (h2 : specific_condition a) : a 2 + a 8 = 180 :=
by
  sorry

end sequence_sum_l331_331089


namespace total_handshakes_proof_l331_331249

-- Define the groups
def groupA : Finset ℕ := {i | i < 25}
def groupB : Finset ℕ := {i | 25 ≤ i ∧ i < 35}
def groupC : Finset ℕ := {i | 35 ≤ i ∧ i < 40}

-- Calculate the individual group interactions
noncomputable def handshakes_AB := 10 * 25
noncomputable def handshakes_AC := 5 * 25
noncomputable def handshakes_BC := 10 * 5
noncomputable def handshakes_B := (Finset.card groupB) * (Finset.card groupB - 1) / 2
noncomputable def handshakes_C := 0

-- Total handshakes
noncomputable def total_handshakes := handshakes_AB + handshakes_AC + handshakes_BC + handshakes_B + handshakes_C

-- The statement to prove
theorem total_handshakes_proof : total_handshakes = 470 := by
  sorry

end total_handshakes_proof_l331_331249


namespace discriminant_nonnegative_root_greater_than_zero_l331_331618

-- Define the quadratic equation and its discriminant
def quadratic_eq (k x : ℝ) : Prop := x^2 - k*x - k - 1 = 0

-- Prove that the discriminant is always non-negative
theorem discriminant_nonnegative (k : ℝ) : 
    let Δ := k^2 + 4*k + 4 in Δ ≥ 0 :=
by
  let Δ := k^2 + 4*k + 4
  show Δ ≥ 0
  calc
    Δ = (k + 2)^2 : by sorry
    _ ≥ 0 : by sorry

-- Show that if one root is greater than 0, then k > -1
theorem root_greater_than_zero (k : ℝ) (h : ∃ x : ℝ, quadratic_eq k x ∧ x > 0) : k > -1 :=
by
  obtain ⟨x, ⟨hx_eq, hx_pos⟩⟩ := h
  have factor_eq : quadratic_eq k x ↔ (x - (k + 1)) * (x + 1) = 0 := by sorry
  rw [←factor_eq] at hx_eq
  cases (eq_zero_or_eq_zero_of_mul_eq_zero hx_eq) with hx1 hx2
  · have : x = k + 1 := hx1 ..
    have : k + 1 > 0 := hx_pos ..
    show k > -1
    exact lt_of_add_one_gt this
  · have : x = -1 := hx2 ..
    have : -1 > 0 := hx_pos ..
    have : false := by sorry
    contradiction
  sorry

end discriminant_nonnegative_root_greater_than_zero_l331_331618


namespace second_train_length_l331_331190

noncomputable def length_of_second_train (speed1_kmph speed2_kmph : ℝ) (time_seconds : ℝ) (length1_meters : ℝ) : ℝ :=
  let speed1_mps := (speed1_kmph * 1000) / 3600
  let speed2_mps := (speed2_kmph * 1000) / 3600
  let relative_speed_mps := speed1_mps + speed2_mps
  let distance := relative_speed_mps * time_seconds
  distance - length1_meters

theorem second_train_length :
  length_of_second_train 72 18 17.998560115190784 200 = 250 :=
by
  sorry

end second_train_length_l331_331190


namespace mary_final_weight_l331_331751

theorem mary_final_weight :
    let initial_weight := 99
    let initial_loss := 12
    let first_gain := 2 * initial_loss
    let second_loss := 3 * initial_loss
    let final_gain := 6
    let weight_after_first_loss := initial_weight - initial_loss
    let weight_after_first_gain := weight_after_first_loss + first_gain
    let weight_after_second_loss := weight_after_first_gain - second_loss
    let final_weight := weight_after_second_loss + final_gain
    in final_weight = 81 :=
by
    sorry

end mary_final_weight_l331_331751


namespace correct_fraction_order_l331_331203

noncomputable def fraction_ordering : Prop := 
  (16 / 12 < 18 / 13) ∧ (18 / 13 < 21 / 14) ∧ (21 / 14 < 20 / 15)

theorem correct_fraction_order : fraction_ordering := 
by {
  repeat { sorry }
}

end correct_fraction_order_l331_331203


namespace underlined_pair_2013_l331_331724

theorem underlined_pair_2013 :
  ∃ (n : ℕ), (n ≡ 1 [MOD 4] ∨ n ≡ 3 [MOD 4]) ∧
             n = 4025 ∧ (∀ k, k < 2013 → (n_k = 2 * k - 1) → 
                             ((n_k ≡ 1 [MOD 4] ∨ n_k ≡ 3 [MOD 4]) ∧ 3^n_k ≡ n_k [MOD 10]))  :=
begin
  let n := 4025,
  use n,
  split; try { exact or.inr (nat.modeq.mod_eq 4025 3 4 (by norm_num)) },
  split,
  exact nat.mod_eq_of_lt (by norm_num),
  intros k hk hnk,
  split,
  { exact or.inr (nat.modeq.mod_eq (2 * k - 1) 3 4 (by norm_num)) },
  { suffices : (3 ^ (2 * k - 1) % 10) = ((2 * k - 1) % 10),
    simpa,
    sorry }
end

end underlined_pair_2013_l331_331724


namespace largest_n_with_triangle_property_l331_331264

/-- Triangle property: For any subset {a, b, c} with a ≤ b ≤ c, a + b > c -/
def triangle_property (s : Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s → a ≤ b → b ≤ c → a + b > c

/-- Definition of the set {3, 4, ..., n} -/
def consecutive_set (n : ℕ) : Finset ℕ :=
  Finset.range (n + 1) \ Finset.range 3

/-- The problem statement: The largest possible value of n where all eleven-element
 subsets of {3, 4, ..., n} have the triangle property -/
theorem largest_n_with_triangle_property : ∃ n, (∀ s ⊆ consecutive_set n, s.card = 11 → triangle_property s) ∧ n = 321 := sorry

end largest_n_with_triangle_property_l331_331264


namespace triangle_perimeter_l331_331900

-- Define the given sides of the triangle
def side_a := 15
def side_b := 6
def side_c := 12

-- Define the function to calculate the perimeter of the triangle
def perimeter (a b c : ℕ) : ℕ :=
  a + b + c

-- The theorem stating that the perimeter of the given triangle is 33
theorem triangle_perimeter : perimeter side_a side_b side_c = 33 := by
  -- We can include the proof later
  sorry

end triangle_perimeter_l331_331900


namespace compute_f_2023_l331_331305

def f (x : Int) : Int :=
  if x <= 0 then x^2 + 1 else f (x - 2)

theorem compute_f_2023 : f 2023 = 2 := by
  sorry

end compute_f_2023_l331_331305


namespace sum_of_even_numbers_l331_331973

-- Define the sequence of even numbers between 1 and 1001
def even_numbers_sequence (n : ℕ) := 2 * n

-- Conditions
def first_term := 2
def last_term := 1000
def common_difference := 2
def num_terms := 500
def sum_arithmetic_series (n : ℕ) (a l : ℕ) := n * (a + l) / 2

-- Main statement to be proved
theorem sum_of_even_numbers : 
  sum_arithmetic_series num_terms first_term last_term = 250502 := 
by
  sorry

end sum_of_even_numbers_l331_331973


namespace slope_angle_of_line_is_30_degrees_l331_331336

theorem slope_angle_of_line_is_30_degrees :
  let line_eq := ∀ (x y : ℝ), x - √3 * y - 2 = 0
  ∃ α : ℝ, 0 ≤ α ∧ α < 180 ∧ tan α = √3 / 3 ∧ α = 30 :=
by
  sorry

end slope_angle_of_line_is_30_degrees_l331_331336


namespace number_of_strawberries_in_each_basket_l331_331436

variable (x : ℕ) (Lilibeth_picks : 6 * x)
variable (total_strawberries : 4 * 6 * x = 1200)

theorem number_of_strawberries_in_each_basket : x = 50 := by
  sorry

end number_of_strawberries_in_each_basket_l331_331436


namespace distinct_integers_dividing_15_pow_6_l331_331685

theorem distinct_integers_dividing_15_pow_6 :
  ∃ (S : Finset ℕ), S.card = 4 ∧
    (∀ x ∈ S, ∃ (x1 x2 : ℕ), x = 3^x1 * 5^x2 ∧ 0 ≤ x1 ∧ x1 ≤ 6 ∧ 0 ≤ x2 ∧ x2 ≤ 6) ∧
    (∀ (a ∈ S) (b ∈ S), a ≠ b → ¬ (a ∣ b ∨ b ∣ a)) →
  S.card = 1225 := 
sorry

end distinct_integers_dividing_15_pow_6_l331_331685


namespace A_n_correct_B_n_correct_l331_331307

noncomputable def sequences (n : ℕ) : ℕ := sorry

/-- Given a positive integer n, and a sequence (a_1, a_2, ..., a_{2n}),
satisfying the following conditions:
1. For all i in {1, 2, ..., 2n}, a_i is either 1 or -1 (i.e., a_i ∈ {1, -1});
2. For any 1 ≤ k ≤ l ≤ n, we have | (a_{2k-1} + a_{2k} + ... + a_{2l}) | ≤ 2.
-/
variables (a : ℕ → ℤ) (n : ℕ)

axiom valid_sequence : 
  (∀ i, 1 ≤ i ∧ i ≤ 2 * n → a i = 1 ∨ a i = -1) ∧ 
  (∀ k l, 1 ≤ k ∧ k ≤ l ∧ l ≤ n → abs (∑ i in finRange (2 * k - 1, 2 * l), a i) ≤ 2)

/-- Definition of A_n: number of sequences (a_1, a_2, ..., a_{2n}) that 
satisfy a_{2k-1} + a_{2k} = 0 for all 1 ≤ k ≤ n. -/
def A_n : ℕ := sequences n

theorem A_n_correct : 
  (A_n = 2^n) :=
sorry

/-- Definition of B_n: number of sequences (a_1, a_2, ..., a_{2n}) such that
there exists 1 ≤ k ≤ n with a_{2k-1} + a_{2k} ≠ 0. -/
def B_n : ℕ := sequences n

theorem B_n_correct :
  (B_n = 2 * (3^n - 2^n)) :=
sorry

end A_n_correct_B_n_correct_l331_331307


namespace smallest_a_value_l331_331735

theorem smallest_a_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : ∀ x : ℤ, cos (a * x + b) = cos (31 * x)) : a = 31 :=
  sorry

end smallest_a_value_l331_331735


namespace positive_integer_solutions_equation_l331_331961

theorem positive_integer_solutions_equation (x y : ℕ) (positive_x : x > 0) (positive_y : y > 0) :
  x^2 + 6 * x * y - 7 * y^2 = 2009 ↔ (x = 252 ∧ y = 251) ∨ (x = 42 ∧ y = 35) ∨ (x = 42 ∧ y = 1) :=
sorry

end positive_integer_solutions_equation_l331_331961


namespace total_weight_of_10_moles_CaH2_is_420_96_l331_331200

def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_H : ℝ := 1.008
def molecular_weight_CaH2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_H
def moles_CaH2 : ℝ := 10
def total_weight_CaH2 : ℝ := molecular_weight_CaH2 * moles_CaH2

theorem total_weight_of_10_moles_CaH2_is_420_96 :
  total_weight_CaH2 = 420.96 :=
by
  sorry

end total_weight_of_10_moles_CaH2_is_420_96_l331_331200


namespace fruit_groups_l331_331502

theorem fruit_groups : 
  (384 / 16 = 24) ∧ 
  (192 / 345 ≈ 0.5565) ∧ 
  (168 / 28 = 6) ∧ 
  (216 / 24 = 9) :=
by
  sorry

end fruit_groups_l331_331502


namespace Peter_total_distance_l331_331761

theorem Peter_total_distance 
  (total_time : ℝ) 
  (speed1 speed2 fraction1 fraction2 : ℝ) 
  (h_time : total_time = 1.4) 
  (h_speed1 : speed1 = 4) 
  (h_speed2 : speed2 = 5) 
  (h_fraction1 : fraction1 = 2/3) 
  (h_fraction2 : fraction2 = 1/3) 
  (D : ℝ) : 
  (fraction1 * D / speed1 + fraction2 * D / speed2 = total_time) → D = 6 :=
by
  intros h_eq
  sorry

end Peter_total_distance_l331_331761


namespace prob_odd_or_greater_than_4_l331_331869

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_greater_than_4 (n : ℕ) : Prop := n > 4

def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

def prob_event (event : ℕ → Prop) : ℚ :=
  (die_faces.filter event).card / die_faces.card

theorem prob_odd_or_greater_than_4 :
  prob_event (λ n, is_odd n ∨ is_greater_than_4 n) = 2 / 3 :=
by
  sorry

end prob_odd_or_greater_than_4_l331_331869


namespace best_deal_around_the_world_l331_331469

def polina_age : ℕ := 5
def total_people : ℕ := 3

-- Costs for Globus
def globus_per_person_age_above_5 : ℕ := 25400
def globus_discount_percentage : ℕ := 2

-- Costs for Around the World
def around_the_world_cost_age_below_6 : ℕ := 11400
def around_the_world_cost_age_above_6 : ℕ := 23500
def around_the_world_commission_percentage : ℕ := 1

noncomputable def total_cost_globus : ℕ :=
  let initial_cost := total_people * globus_per_person_age_above_5 in
  let discount := initial_cost * globus_discount_percentage / 100 in
  initial_cost - discount

noncomputable def total_cost_around_the_world : ℕ :=
  let initial_cost := around_the_world_cost_age_below_6 + 2 * around_the_world_cost_age_above_6 in
  let commission := initial_cost * around_the_world_commission_percentage / 100 in
  initial_cost + commission

theorem best_deal_around_the_world :
  total_cost_around_the_world < total_cost_globus ∧ total_cost_around_the_world = 58984 := 
  by
    sorry

end best_deal_around_the_world_l331_331469


namespace coaxial_circles_l331_331926

theorem coaxial_circles 
  (A B C D X X' Y Y' Z Z' : Point)
  (l : Line)
  (intersects_AB : Line.intersect l (line_segment A B) = Some X)
  (intersects_CD : Line.intersect l (line_segment C D) = Some X')
  (intersects_AD : Line.intersect l (line_segment A D) = Some Y)
  (intersects_BC : Line.intersect l (line_segment B C) = Some Y')
  (intersects_AC : Line.intersect l (line_segment A C) = Some Z)
  (intersects_BD : Line.intersect l (line_segment B D) = Some Z') :
  coaxial (circle_with_diameter X X') (circle_with_diameter Y Y') (circle_with_diameter Z Z') :=
sorry

end coaxial_circles_l331_331926


namespace relay_team_order_count_l331_331394

theorem relay_team_order_count : 
  let team_members : Finset String := {"Jordan", "A", "B", "C", "D"} in
  let permutations := (team_members.erase "Jordan").permutations.unique in 
  permutations.card = 24 := by
sorry

end relay_team_order_count_l331_331394


namespace identity_x_squared_minus_y_squared_l331_331044

theorem identity_x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : 
  x^2 - y^2 = 80 := by sorry

end identity_x_squared_minus_y_squared_l331_331044


namespace no_hamiltonian_cycle_l331_331844

-- Define the problem constants
def n : ℕ := 2016
def a : ℕ := 2
def b : ℕ := 3

-- Define the circulant graph and the conditions of the Hamiltonian cycle theorem
theorem no_hamiltonian_cycle (s t : ℕ) (h1 : s + t = Int.gcd n (a - b)) :
  ¬ (Int.gcd n (s * a + t * b) = 1) :=
by
  sorry  -- Proof not required as per instructions

end no_hamiltonian_cycle_l331_331844


namespace sum_of_digits_of_x_l331_331906

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem sum_of_digits_of_x :
  ∃ x : ℕ, is_palindrome x ∧ 100 ≤ x ∧ x ≤ 999 ∧ is_palindrome (x + 50) ∧ 1000 ≤ (x + 50) ∧ (x + 50) ≤ 1049 ∧ x.digits 10.sum = 15 :=
by
  sorry

end sum_of_digits_of_x_l331_331906


namespace molly_xanthia_reading_difference_in_minutes_l331_331873

theorem molly_xanthia_reading_difference_in_minutes :
  ∀ (Xanthia_speed Molly_speed pages : ℕ),
  Xanthia_speed = 120 →
  Molly_speed = 60 → 
  pages = 360 →
  ((pages / Molly_speed - pages / Xanthia_speed) * 60 = 180) := by
  intros Xanthia_speed Molly_speed pages hx hm hp
  rw [hx, hm, hp]
  -- Proof steps would go here
  sorry

end molly_xanthia_reading_difference_in_minutes_l331_331873


namespace absolute_inequality_solution_l331_331953

theorem absolute_inequality_solution (x : ℝ) (hx : x > 0) :
  |5 - 2 * x| ≤ 8 ↔ 0 ≤ x ∧ x ≤ 6.5 :=
by sorry

end absolute_inequality_solution_l331_331953


namespace red_region_area_l331_331426

theorem red_region_area :
  let a := 3 in
  let radius := real.sqrt 3 in
  let segments_area := 4 * ((1 / 6) * real.pi * radius ^ 2 - (radius ^ 2 * (real.sqrt 3 / 4))) in
  let total_area := a ^ 2 - segments_area / 4 in 
  (total_area + segments_area) = 2 * real.pi + 3 - 3 * real.sqrt 3 :=
by 
  let a := 3
  let radius := real.sqrt 3
  let segments_area := 4 * ((1 / 6) * real.pi * radius ^ 2 - (radius ^ 2 * (real.sqrt 3 / 4)))
  let total_area := a ^ 2 - segments_area / 4
  show _,
  sorry

end red_region_area_l331_331426


namespace income_to_expenditure_ratio_l331_331159

variable (I E S : ℕ)

def Ratio (a b : ℕ) : ℚ := a / (b : ℚ)

theorem income_to_expenditure_ratio (h1 : I = 14000) (h2 : S = 2000) (h3 : S = I - E) : 
  Ratio I E = 7 / 6 :=
by
  sorry

end income_to_expenditure_ratio_l331_331159


namespace find_f_inverse_sum_l331_331416

def f (x : ℝ) : ℝ := x * abs x

theorem find_f_inverse_sum :
  let f_inv_4 := if (4 : ℝ) ≥ 0 then real.sqrt 4 else -real.sqrt 4 in
  let f_inv_neg_100 := if (-100 : ℝ) ≤ 0 then -real.sqrt 100 else real.sqrt 100 in
  f_inv_4 + f_inv_neg_100 = -8 :=
by
  sorry

end find_f_inverse_sum_l331_331416


namespace projectile_height_l331_331817

theorem projectile_height (t : ℝ) (h : (-16 * t^2 + 80 * t = 100)) : t = 2.5 :=
sorry

end projectile_height_l331_331817


namespace closest_ratio_of_adults_to_children_l331_331148

def total_fees (a c : ℕ) : ℕ := 20 * a + 10 * c
def adults_children_equation (a c : ℕ) : Prop := 2 * a + c = 160

theorem closest_ratio_of_adults_to_children :
  ∃ a c : ℕ, 
    total_fees a c = 1600 ∧
    a ≥ 1 ∧ c ≥ 1 ∧
    adults_children_equation a c ∧
    (∀ a' c' : ℕ, total_fees a' c' = 1600 ∧ 
        a' ≥ 1 ∧ c' ≥ 1 ∧ 
        adults_children_equation a' c' → 
        abs ((a : ℝ) / c - 1) ≤ abs ((a' : ℝ) / c' - 1)) :=
  sorry

end closest_ratio_of_adults_to_children_l331_331148


namespace largest_theta_l331_331519

-- Given conditions translated to Lean statements as necessary.
variable (θ : ℝ)

def cos_conditions (θ : ℝ) : Prop := 
  ∏ k in Finset.range 11, Real.cos (2^k * θ) ≠ 0

def product_condition (θ : ℝ) : Prop :=
  ∏ k in Finset.range 11, (1 + 1 / Real.cos (2^k * θ)) = 1

theorem largest_theta : (θ < Real.pi ∧ cos_conditions θ ∧ product_condition θ) → 
  θ = (2046 * Real.pi / 2047) :=
by
  sorry

end largest_theta_l331_331519


namespace largest_whole_number_l331_331488

theorem largest_whole_number (x : ℕ) (hx : 8 * x < 120) : x ≤ 14 :=
begin
  sorry
end

end largest_whole_number_l331_331488


namespace arithmetic_square_root_of_second_consecutive_l331_331513

theorem arithmetic_square_root_of_second_consecutive (x : ℝ) : 
  real.sqrt (x^2 + 1) = real.sqrt (x^2 + 1) :=
sorry

end arithmetic_square_root_of_second_consecutive_l331_331513


namespace a_2021_2022_2023_product_l331_331317

noncomputable def a_seq : ℕ → ℚ
| 0       := 1 -- dummy value, since sequence defined from a_1
| 1       := (1 : ℚ) / 2
| (n + 2) := 1 / (1 - a_seq (n + 1))

theorem a_2021_2022_2023_product : a_seq 2021 * a_seq 2022 * a_seq 2023 = -1 :=
sorry

end a_2021_2022_2023_product_l331_331317


namespace f_x_gt_0_l331_331061

noncomputable def f : ℝ → ℝ

-- Define the conditions
axiom odd_f : ∀ x, f (-x) = -f x
axiom f_neg_x : ∀ x, x < 0 → f x = cos (3 * x) + sin (2 * x)

-- Define the theorem to be proven
theorem f_x_gt_0 (x : ℝ) (h : x > 0) : f x = -cos (3 * x) + sin (2 * x) :=
sorry

end f_x_gt_0_l331_331061


namespace find_non_zero_real_x_satisfies_equation_l331_331611

theorem find_non_zero_real_x_satisfies_equation :
  ∃! x : ℝ, x ≠ 0 ∧ (9 * x) ^ 18 - (18 * x) ^ 9 = 0 ∧ x = 2 :=
by
  sorry

end find_non_zero_real_x_satisfies_equation_l331_331611


namespace length_FY_l331_331142

/-- Define the length of sides of the hexagon and segment ratios --/
def length_of_side : ℝ := 3
def ratio_AX_to_AB : ℝ := 4
def midpoint_ratio : ℝ := 1 / 2

/-- Define points and their relationships based on the given conditions --/
axiom regular_hexagon (A B C D E F : Type)
  [∀ {A B C D E F : Type}, has_add (A → B → C → D → E → F)]
  (h_hex : AB + BC + CD + DE + EF + FA = 6 * length_of_side)

axiom extended_point (X : Type) : AX = ratio_AX_to_AB * AB

axiom midpoint (DX : Type) (Y : Type) : DY = midpoint_ratio * DX

/-- Prove the length of segment FY --/
theorem length_FY (A B C D E F X Y : Type)
  [regular_hexagon A B C D E F] 
  [extended_point X] 
  [midpoint DX Y] :
  FY = (3 * real.sqrt 3) / 2 :=
sorry

end length_FY_l331_331142


namespace correct_average_l331_331806

theorem correct_average (initial_avg : ℝ) (n : ℕ) (error1 : ℝ) (wrong_num : ℝ) (correct_num : ℝ) :
  initial_avg = 40.2 → n = 10 → error1 = 19 → wrong_num = 13 → correct_num = 31 →
  (initial_avg * n - error1 - wrong_num + correct_num) / n = 40.1 :=
by
  intros
  sorry

end correct_average_l331_331806


namespace minSumDivisibleBy2006Squared_l331_331493

/-- The product of a set of distinct positive integers is divisible by 2006^2. -/
def isDivisibleBy2006Squared (S : Set ℕ) : Prop :=
  ∏ x in S, x % (2006^2) = 0

/-- The minimum sum of a set of distinct positive integers whose product is divisible
    by 2006^2 is 228. -/
theorem minSumDivisibleBy2006Squared (S : Set ℕ) (h : isDivisibleBy2006Squared S) : 
  (∑ x in S, x) ≥ 228 :=
sorry

end minSumDivisibleBy2006Squared_l331_331493


namespace two_digit_numbers_seven_times_sum_of_digits_l331_331684

theorem two_digit_numbers_seven_times_sum_of_digits :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ a b : ℕ, n = 10 * a + b ∧ n = 7 * (a + b) ∧ a < 10 ∧ b < 10}.to_finset.card = 4 :=
by
  sorry

end two_digit_numbers_seven_times_sum_of_digits_l331_331684


namespace problem_l331_331635

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = 2 + (finset.range n).sum a

noncomputable def bn (a : ℕ → ℕ) (n : ℕ) : ℕ :=
1 + (nat.log 2 (a n))^2

noncomputable def seq_term (a : ℕ → ℕ) (n : ℕ) : ℚ :=
1 / ((bn a n) * (bn a (n + 1)))

noncomputable def sum_seq_terms (a : ℕ → ℕ) (n : ℕ) : ℚ :=
(finset.range n).sum (λ i, seq_term a i)

theorem problem (a : ℕ → ℕ) (n : ℕ) (h : sequence a) : a n = 2^n ∧ sum_seq_terms a n < 1/6 :=
begin
  sorry
end

end problem_l331_331635


namespace probability_of_four_sixes_l331_331141

theorem probability_of_four_sixes :
  let n := 10 in
  let k := 4 in
  let p := 1 / 3 in
  let binom (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) in
  let probability := binom n k * p ^ k * (1 - p) ^ (n - k) in
  probability = 13440 / 59049 := 
by
  sorry

end probability_of_four_sixes_l331_331141


namespace inclination_angle_l331_331194

noncomputable def line_eqn : ℝ → ℝ → Prop 
| x, y := x + (sqrt 3) * y - 1 = 0

theorem inclination_angle : (∀ x y, line_eqn x y) → Real.arctan ((-sqrt 3) / 3) = (5 / 6) * Real.pi :=
by
  intro h
  sorry

end inclination_angle_l331_331194


namespace number_of_intersections_l331_331121

theorem number_of_intersections (n : ℕ) (p : ℕ) (h1 : n = 10) (h2 : p = 2) :
  ∃ intersections : ℕ, intersections = 44 :=
begin
  sorry
end

end number_of_intersections_l331_331121


namespace rationalize_denominator_correct_l331_331770

noncomputable def rationalize_denominator_sum : ℕ :=
  let a := real.root (5 : ℝ) 3;
  let b := real.root (3 : ℝ) 3;
  let A := real.root (25 : ℝ) 3;
  let B := real.root (15 : ℝ) 3;
  let C := real.root (9 : ℝ) 3;
  let D := 2;
  (25 + 15 + 9 + 2)

theorem rationalize_denominator_correct :
  rationalize_denominator_sum = 51 :=
  by sorry

end rationalize_denominator_correct_l331_331770


namespace min_value_F_neg_infty_to_0_l331_331013

open Real

def is_odd (h : ℝ → ℝ) := ∀ x : ℝ, h (-x) = - h x

variables {a b : ℝ} {f g : ℝ → ℝ}

-- Define the function F
def F (x : ℝ) : ℝ := a * f x + b * g x + 2

-- The statement of the problem
theorem min_value_F_neg_infty_to_0
  (hf : is_odd f) (hg : is_odd g) (h_max : ∃ c > 0, F c = 5) :
  ∃ d < 0, F d = -1 :=
sorry

end min_value_F_neg_infty_to_0_l331_331013


namespace different_outcomes_count_l331_331297

-- Define the number of students and competitions
def num_students : ℕ := 4
def num_competitions : ℕ := 3

-- Define the proof statement
theorem different_outcomes_count : (num_competitions ^ num_students) = 81 := 
by
  -- Proof will be here
  sorry

end different_outcomes_count_l331_331297


namespace shaded_area_l331_331903

/--
Given a larger square containing a smaller square entirely within it,
where the side length of the smaller square is 5 units
and the side length of the larger square is 10 units,
prove that the area of the shaded region (the area of the larger square minus the area of the smaller square) is 75 square units.
-/
theorem shaded_area :
  let side_length_smaller := 5
  let side_length_larger := 10
  let area_larger := side_length_larger * side_length_larger
  let area_smaller := side_length_smaller * side_length_smaller
  area_larger - area_smaller = 75 := 
by
  let side_length_smaller := 5
  let side_length_larger := 10
  let area_larger := side_length_larger * side_length_larger
  let area_smaller := side_length_smaller * side_length_smaller
  sorry

end shaded_area_l331_331903


namespace inequality_problem_l331_331995

variable {a b c d : ℝ}

theorem inequality_problem (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
    a + c > b + d ∧ ad^2 > bc^2 ∧ (1 / bc) < (1 / ad) :=
by
  sorry

end inequality_problem_l331_331995


namespace trees_same_height_l331_331383

theorem trees_same_height : ∃ t : ℝ, t = 15 ∧ ∀ x : ℝ,
  (0 ≤ t ∧ t ≤ 9 → (height_cedar t x = height_fir t x)) :=
by
  let height_cedar (t x : ℝ) : ℝ := (x / 6) * t
  let height_fir (t x : ℝ) : ℝ := if t - 2 ≥ 0 then (x / 2) * (t - 2) else 0
  sorry

end trees_same_height_l331_331383


namespace number_of_elements_in_complement_intersection_l331_331672

open Set

-- Define sets A and B
def A : Set ℝ := {x | ∃ n : ℕ, x = 3 * n + 1}
def B : Set ℝ := {4, 5, 6, 7, 8}

-- Lean theorem to prove the cardinality
theorem number_of_elements_in_complement_intersection :
  (Finset.card (Finset.filter (λ x, ¬ ∃ n : ℕ, x = 3 * n + 1) {4, 5, 6, 7, 8})) = 3 :=
by
  -- proof goes here
  sorry

end number_of_elements_in_complement_intersection_l331_331672


namespace find_certain_number_l331_331224

theorem find_certain_number (x : ℕ) (certain_number : ℕ)
  (h1 : certain_number * x = 675)
  (h2 : x = 27) : certain_number = 25 :=
by
  -- Proof goes here
  sorry

end find_certain_number_l331_331224


namespace job_completion_days_l331_331039

variable (m r h d : ℕ)

theorem job_completion_days :
  (m + 2 * r) * (h + 1) * (m * h * d / ((m + 2 * r) * (h + 1))) = m * h * d :=
by
  sorry

end job_completion_days_l331_331039


namespace geometric_sequence_problem_l331_331103

theorem geometric_sequence_problem 
  (a : ℕ → ℝ)
  (q : ℝ)
  (h₁ : ∀ n, a (n + 1) = a n * q)
  (h₂ : q = 2)
  (h₃ : (List.range 33).map (λ n, a (n+1)) |>.prod = 2^33) :
  (List.range 11).map (λ k, a (3 * (k + 1))) |>.prod = 2^22 :=
by
  sorry

end geometric_sequence_problem_l331_331103


namespace mary_final_weight_l331_331743

theorem mary_final_weight : 
  let initial_weight := 99
  let weight_loss1 := 12
  let weight_gain1 := 2 * weight_loss1
  let weight_loss2 := 3 * weight_loss1
  let weight_gain2 := 6
  initial_weight - weight_loss1 + weight_gain1 - weight_loss2 + weight_gain2 = 81 := by 
  sorry

end mary_final_weight_l331_331743


namespace unique_solution_implies_relation_l331_331960

open Nat

noncomputable def unique_solution (a b : ℤ) :=
  ∃! (x y z : ℤ), x + y = a - 1 ∧ x * (y + 1) - z^2 = b

theorem unique_solution_implies_relation (a b : ℤ) :
  unique_solution a b → b = (a * a) / 4 := sorry

end unique_solution_implies_relation_l331_331960


namespace total_eggs_collected_l331_331253

-- Define the number of eggs collected by each person based on the conditions given

-- Benjamin collects 6 dozen eggs
def Benjamin := 6

-- Carla collects 3 times the number of eggs that Benjamin collects
def Carla := 3 * Benjamin

-- Trisha collects 4 dozen less than Benjamin
def Trisha := Benjamin - 4

-- David collects twice the number of eggs that Trisha collects,
-- but half the number that Carla collects
def David := 2 * Trisha

-- Emily collects 3/4 the amount of eggs that David collects
def Emily := (3 / 4) * David

-- Emily ends up with 50% more eggs than Trisha
def Emily' := Trisha + (1 / 2) * Trisha

-- Define the total number of eggs collected by all five
def Total := Benjamin + Carla + Trisha + David + Emily

-- Prove that all conditions hold and the total number of eggs is 33 dozen
theorem total_eggs_collected : Total = 33 := by
  have h1 : Benjamin = 6 := by rfl
  have h2 : Carla = 3 * Benjamin := by rfl
  have h3 : Trisha = Benjamin - 4 := by rfl
  have h4 : David = 2 * Trisha := by rfl
  have h5 : Emily = (3 / 4) * David := by rfl
  have h6 : Emily = 3 := by
    rw [h3, h4, h5]
    norm_num
  have h7 : Total = Benjamin + Carla + Trisha + David + Emily := by rfl
  rw [h1, h2, h3, h4, h6, h7]
  norm_num
  sorry

end total_eggs_collected_l331_331253


namespace linear_function_points_relation_l331_331035

theorem linear_function_points_relation :
  ∀ (y1 y2 : ℝ), 
  (y1 = -3 * 2 + 1) ∧ (y2 = -3 * 3 + 1) → y1 > y2 :=
by
  intro y1 y2
  intro h
  cases h
  sorry

end linear_function_points_relation_l331_331035


namespace sum_dist_P_to_BC_l331_331242

-- Definitions from conditions
variable (ABC : Type) [Triangle ABC]
variable (O P : Point)
variable (α τ : Real)

-- Assumptions
axiom angle_BAC (ABC) : ∠ BAC = α
axiom sum_dists (O B C : Point) : dist O B + dist O C = τ

-- Theorem statement
theorem sum_dist_P_to_BC 
    (P B C : Point) 
    (h1 : touches P (ray AB))
    (h2 : touches P (ray AC))
    (h3 : touches P (side BC)) :
    dist P B + dist P C = τ * cot (π - α) / 4 := sorry

end sum_dist_P_to_BC_l331_331242


namespace total_artworks_l331_331432

theorem total_artworks (students art_kits : ℕ)
    (students_per_kit : ℕ → ℕ → ℕ) 
    (artworks_group1 : ℕ → ℕ) 
    (artworks_group2 : ℕ → ℕ)
    (h_students : students = 10)
    (h_art_kits : art_kits = 20)
    (h_students_per_kit : students_per_kit students art_kits = 2)
    (h_group1_size : (students / 2) = 5)
    (h_group2_size : (students / 2) = 5)
    (h_artworks_group1 : (5 * 3) = artworks_group1 5)
    (h_artworks_group2 : (5 * 4) = artworks_group2 5)
    : (artworks_group1 5 + artworks_group2 5) = 35 := 
by 
  rw [h_students, h_art_kits, h_students_per_kit, h_group1_size, h_group2_size, h_artworks_group1, h_artworks_group2]
  sorry

end total_artworks_l331_331432


namespace option1_function_relationship_electricity_usage_when_payment_35_option1_more_economical_range_l331_331741

-- Define the function relationship L(x) for Option 1
def L (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 30 then 2 + 0.5 * x else 0.6 * x - 1

-- Define the function relationship F(x) for Option 2
def F (x : ℝ) : ℝ := 0.58 * x

-- Prove the function relationship for Option 1
theorem option1_function_relationship (x : ℝ) :
  L x = if 0 ≤ x ∧ x ≤ 30 then 2 + 0.5 * x else 0.6 * x - 1 := sorry

-- Prove the electricity usage given the payment amount under Option 1
theorem electricity_usage_when_payment_35 :
  L 60 = 35 := sorry

-- Prove Option 1 is more economical than Option 2 in a given range
theorem option1_more_economical_range (x : ℝ) :
  25 < x ∧ x < 50 → L x < F x := sorry

end option1_function_relationship_electricity_usage_when_payment_35_option1_more_economical_range_l331_331741


namespace sin_theta_square_midpoints_l331_331533

open Real

-- Definitions based on conditions
def isSquare (A B C D : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A ∧
  ∠ A B C = π / 2 ∧ ∠ B C D = π / 2 ∧ ∠ C D A = π / 2 ∧ ∠ D A B = π / 2

def isMidpoint (M B C : Point) : Prop :=
  M.x = (B.x + C.x) / 2 ∧ M.y = (B.y + C.y) / 2

-- The theorem that needs to be proven
theorem sin_theta_square_midpoints (A B C D M N : Point)
  (h_square : isSquare A B C D)
  (h_midpoint1 : isMidpoint M B C)
  (h_midpoint2 : isMidpoint N C D) :
  sin (angle A M N) = 3/5 :=
sorry

end sin_theta_square_midpoints_l331_331533


namespace crayons_selection_l331_331843

-- Definition for the problem constraints
def number_of_ways_to_select_crayons (crayons : ℕ) (select_1 : ℕ) (select_2 : ℕ) : ℕ :=
  (Nat.choose crayons select_1) * (Nat.choose (crayons - select_1) select_2)

-- Problem statement in Lean 4
theorem crayons_selection (h : number_of_ways_to_select_crayons 15 3 4 = 225225) : true :=
  by
    sorry

end crayons_selection_l331_331843


namespace tan_alpha_minus_pi_over_4_l331_331315

theorem tan_alpha_minus_pi_over_4 (α : ℝ) (h : ∀ (x y : ℝ), 2 * x + y + 1 = 0 → α = arctan (-2)) :
  tan (α - π / 4) = 3 :=
by
  sorry

end tan_alpha_minus_pi_over_4_l331_331315


namespace correct_statement_D_l331_331186

-- Conditions expressed as definitions
def candidates_selected_for_analysis : ℕ := 500

def statement_A : Prop := candidates_selected_for_analysis = 500
def statement_B : Prop := "The mathematics scores of the 500 candidates selected are the sample size."
def statement_C : Prop := "The 500 candidates selected are individuals."
def statement_D : Prop := "The mathematics scores of all candidates in the city's high school entrance examination last year constitute the population."

-- Problem statement in Lean
theorem correct_statement_D :
  statement_D := sorry

end correct_statement_D_l331_331186


namespace smallest_n_for_divisibility_l331_331429

noncomputable def common_ratio (a b : ℚ) : ℚ := b / a

theorem smallest_n_for_divisibility (a r : ℚ) (h1 : a = 1/2) (h2 : a * r = 10) :
  ∃ n : ℕ, (a * r^(n-1)) % 100000 = 0 ∧ ∀ m : ℕ, (m < n) → (a * r^(m-1)) % 100000 ≠ 0 :=
by
  use 6
  split
  sorry

end smallest_n_for_divisibility_l331_331429


namespace max_value_expression_l331_331967

theorem max_value_expression (a b : ℝ) (ha: 0 < a) (hb: 0 < b) :
  ∃ M, M = 2 * Real.sqrt 87 ∧
       (∀ a b: ℝ, 0 < a → 0 < b →
       (|4 * a - 10 * b| + |2 * (a - b * Real.sqrt 3) - 5 * (a * Real.sqrt 3 + b)|) / Real.sqrt (a ^ 2 + b ^ 2) ≤ M) :=
sorry

end max_value_expression_l331_331967


namespace sqrt_of_second_number_l331_331511

-- Given condition: the arithmetic square root of a natural number n is x
variable (x : ℕ)
def first_number := x ^ 2
def second_number := first_number + 1

-- The theorem statement we want to prove
theorem sqrt_of_second_number (x : ℕ) : Real.sqrt (x^2 + 1) = Real.sqrt (first_number x + 1) :=
by
  sorry

end sqrt_of_second_number_l331_331511


namespace translate_and_symmetric_l331_331062

theorem translate_and_symmetric (φ : ℝ) :
  ∃ φ : ℝ, φ > 0 ∧ (∀ x : ℝ, sin (2 * x + π / 4 - 2 * φ) = sin (-2 * x - π / 4 - 2 * φ)) ∧ φ = 3 * π / 8 :=
sorry

end translate_and_symmetric_l331_331062


namespace largest_divisor_of_expression_l331_331353

theorem largest_divisor_of_expression (x : ℤ) (h₁ : odd x) :
  ∃ d : ℤ, d = 60 ∧ d ∣ (15 * x + 9) * (15 * x + 15) * (15 * x + 21) :=
sorry

end largest_divisor_of_expression_l331_331353


namespace equilateral_triangle_side_approx_length_l331_331346

-- Definition of perimeter and equal sides condition for equilateral triangle
def equilateral_triangle_side_length (P : ℝ) (s : ℝ) : Prop :=
  3 * s = P

-- Perimeter of the triangle
def perimeter : ℝ := 2.0

-- Length of the side to prove
def side_length_approx : ℝ := 0.67

-- Statement to prove the length of the side of the equilateral triangle given the perimeter is approximately 0.67 meters
theorem equilateral_triangle_side_approx_length : 
  ∀ (s : ℝ), equilateral_triangle_side_length perimeter s → (Real.toRat s).round = (Real.toRat 0.67).round :=
begin
  assume s,
  assume h,
  sorry
end

end equilateral_triangle_side_approx_length_l331_331346


namespace product_of_divisors_pow_l331_331970

theorem product_of_divisors_pow (N : ℕ) (n: ℕ) (h : ∃ (d : List ℕ), d.length = n ∧ ∀ a ∈ d, a ∣ N ∧ (∀ b ∈ d, b ∣ N)) : 
  (∏ i in (Finset.univ : Finset (Fin n)), (d.get i)) = N ^ (n / 2) :=
by
  sorry

end product_of_divisors_pow_l331_331970


namespace sum_first_five_terms_geometric_sequence_condition_sum_b_n_l331_331637

section
variable {α : Type*} [LinearOrderedField α] {a₁ d : α}

def a_n (n : ℕ) : α := a₁ + d * (n - 1)

theorem sum_first_five_terms (h : 5 * a₁ + 10 * d = 55) : ∀ n, a_n n = 7 + 2 * (n - 1) :=
by
    intros n
    sorry
  
theorem geometric_sequence_condition (h : a₁ = 7 ∧ d = 2) : (a₁ + d) * (a₁ + 3 * d - 9) = a₁ + 5 * d + a₁ + 6 * d := 
by
    intros
    sorry

def b_n (n : ℕ) : α := 1 / ((a_n n - 6) * (a_n n - 4))

theorem sum_b_n (h₁ : ∀ n, a_n n = 7 + 2 * (n - 1)) (n : ℕ) : 
  (finset.sum (finset.range n) (λ k, b_n (k + 1))) < (1 / 2 : α) :=
by
    intros
    sorry
end

end sum_first_five_terms_geometric_sequence_condition_sum_b_n_l331_331637


namespace equal_numbers_after_operations_l331_331977

theorem equal_numbers_after_operations (n : ℕ) (h : n ≥ 3) :
  (∀ (a : Fin n → ℤ), ∃ k, ∀ m ≥ k, ∀ i, (let a' := (a i - a (i + 1)) % n 
    in a' = 0)) ↔ ∃ k : ℕ, n = 2^k :=
sorry

end equal_numbers_after_operations_l331_331977


namespace min_value_AB_plus_MN_l331_331012

theorem min_value_AB_plus_MN :
  -- Given the parabola y^2 = 4x with focus F at (1,0)
  let C := { p // p.2^2 = 4 * p.1 }
  let F : C := ⟨(1, 0), by simp [pow_two]⟩
  -- Assume l1 passing through F intersects the parabola at A and B
  (l1 : Line) (A B : C)
  (hAB : l1.contains F ∧ l1.contains A ∧ l1.contains B)
  -- Assume l2 passing through F intersects the parabola at M and N
  (l2 : Line) (M N : C)
  (hMN : l2.contains F ∧ l2.contains M ∧ l2.contains N)
  -- with the condition that the product of slopes of l1 and l2 is -1
  (h_slope : l1.slope * l2.slope = -1) :
  -- we want to prove that |AB| + |MN| has a minimum value of 16
  min_value (dist A B + dist M N) = 16 :=
sorry

end min_value_AB_plus_MN_l331_331012


namespace percent_within_one_std_dev_l331_331538

theorem percent_within_one_std_dev 
    (m d : ℝ) 
    (symmetric_distribution : ∀ x, distribution (-x + m) = distribution (x + m))
    (percent_less_m_add_d : distribution (m + d) = 0.68)
    : distribution (m + d) - distribution (m - d) = 0.68 :=
sorry

end percent_within_one_std_dev_l331_331538


namespace sin_45_is_sqrt2_div_2_l331_331259

noncomputable def sin_45_deg : Real :=
  sin (π / 4)

theorem sin_45_is_sqrt2_div_2 :
  sin_45_deg = Real.sqrt (2) / 2 := by
  -- The proof is omitted
  sorry

end sin_45_is_sqrt2_div_2_l331_331259


namespace number_of_legal_phone_numbers_l331_331531

def is_legal_phone_number (d₁ d₂ d₃ d₄ d₅ d₆ d₇ : ℕ) : Prop :=
  (d₁ = d₄ ∧ d₂ = d₅ ∧ d₃ = d₆) ∨ (d₁ = d₅ ∧ d₂ = d₆ ∧ d₃ = d₇)

theorem number_of_legal_phone_numbers : 
  (∃ count, count = 19990 ∧ 
    (∀ d₁ d₂ d₃ d₄ d₅ d₆ d₇ : ℕ, 
    d₁ ∈ finset.range 10 → d₂ ∈ finset.range 10 →
    d₃ ∈ finset.range 10 → d₄ ∈ finset.range 10 →
    d₅ ∈ finset.range 10 → d₆ ∈ finset.range 10 →
    d₇ ∈ finset.range 10 →
    is_legal_phone_number d₁ d₂ d₃ d₄ d₅ d₆ d₇))
sorry

end number_of_legal_phone_numbers_l331_331531


namespace gcd_1213_1985_eq_1_l331_331606

theorem gcd_1213_1985_eq_1
  (h1: ¬ (1213 % 2 = 0))
  (h2: ¬ (1213 % 3 = 0))
  (h3: ¬ (1213 % 5 = 0))
  (h4: ¬ (1985 % 2 = 0))
  (h5: ¬ (1985 % 3 = 0))
  (h6: ¬ (1985 % 5 = 0)):
  Nat.gcd 1213 1985 = 1 := by
  sorry

end gcd_1213_1985_eq_1_l331_331606


namespace number_of_boys_l331_331245

theorem number_of_boys 
  (n : ℕ) 
  (h1 : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ n → 1 ≤ j ∧ j ≤ n → (i - j + n) % n = 25): 
  n = 52 :=
begin
sorry
end

end number_of_boys_l331_331245


namespace temperature_on_tuesday_l331_331807

variable (T W Th F : ℝ)

theorem temperature_on_tuesday :
  (T + W + Th) / 3 = 45 →
  (W + Th + F) / 3 = 50 →
  F = 53 →
  T = 38 :=
by 
  intros h1 h2 h3
  sorry

end temperature_on_tuesday_l331_331807


namespace final_answer_l331_331007

def f (x : ℝ) : ℝ := sorry

axiom functional_eqn : ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem final_answer : f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : ℝ, f (-x) = f x) ∧ ¬ (∃ ε > 0, ∀ h : ℝ, abs h < ε → f h ≥ f 0) := 
by
  -- omit the proof steps that were provided in the solution
  sorry

end final_answer_l331_331007


namespace hyperbola_asymptote_slope_l331_331322

theorem hyperbola_asymptote_slope (a : ℝ) (h : a < 0) 
  (asymptote_slope : Float) (h_slope : asymptote_slope = 5 * Real.pi / 6) :
  a = -3 :=
by
  sorry

end hyperbola_asymptote_slope_l331_331322


namespace cos_sqr_sub_power_zero_l331_331887

theorem cos_sqr_sub_power_zero :
  (cos (30 * Real.pi / 180))^2 - (2 - Real.pi)^0 = -1/4 :=
by
  sorry

end cos_sqr_sub_power_zero_l331_331887


namespace domain_length_fraction_l331_331286

noncomputable def g (x : ℝ) : ℝ :=
log (log (log (log x) / log 9 / log 1/9) / log 3) / log 1/3

theorem domain_length_fraction (p q : ℕ) (hpq : Nat.coprime p q) :
  g.domain = (1, Real.root 9 9) → (1, Real.root 9 9).length = (183 : ℚ / 2282) →
  p + q = 2465 := by
  sorry

end domain_length_fraction_l331_331286


namespace range_of_a_l331_331987

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2

theorem range_of_a (a : ℝ) :
  (∀ (p q : ℝ), 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1 ∧ p ≠ q → (f a p - f a q) / (p - q) > 1)
  ↔ 3 ≤ a :=
by
  sorry

end range_of_a_l331_331987


namespace area_of_triangle_DEF_l331_331285

theorem area_of_triangle_DEF (DF DE : ℝ) (hDF : DF = 8) (hDE : DE = 8) (h_angle_D : ∠ D E F = 45) (h_right_angle_D : ∠ D E F = 90) :
  let A := 1 / 2 * DF * DE in A = 32 :=
by
  sorry

end area_of_triangle_DEF_l331_331285


namespace magic_numbers_count_less_than_130_l331_331945

theorem magic_numbers_count_less_than_130 :
  {N : ℕ | N < 130 ∧ ∃ (m : ℕ), 10^m % N = 0}.finite.card = 9 :=
sorry

end magic_numbers_count_less_than_130_l331_331945


namespace value_of_a5_l331_331088

variable (a_n : ℕ → ℝ)
variable (a1 a9 a5 : ℝ)

-- Given conditions
axiom a1_plus_a9_eq_10 : a1 + a9 = 10
axiom arithmetic_sequence : ∀ n, a_n n = a1 + (n - 1) * (a_n 2 - a1)

-- Prove that a5 = 5
theorem value_of_a5 : a5 = 5 :=
by
  sorry

end value_of_a5_l331_331088


namespace shortest_tangent_length_to_circle_l331_331172

theorem shortest_tangent_length_to_circle 
  (x y : ℝ) (hP : x - y + 2 * real.sqrt 2 = 0) :
  let C := (0,0)
  let R := 1
  shortest_tangent_length_to_circle x y = real.sqrt 3 :=
sorry

end shortest_tangent_length_to_circle_l331_331172


namespace calculate_water_needed_for_solution_l331_331069

-- Definitions for the conditions
def initial_solution_volume : ℝ := 0.08
def nutrient_concentrate_volume : ℝ := 0.05
def water_volume_in_initial_solution : ℝ := 0.03
def total_solution_needed : ℝ := 0.64

-- Calcuate the fraction of water in the initial solution
def fraction_of_water_in_initial_solution : ℝ :=
  water_volume_in_initial_solution / initial_solution_volume

-- Calculate the total amount of water needed for the experiment
def total_water_needed : ℝ :=
  total_solution_needed * fraction_of_water_in_initial_solution

-- Statement of the proof to be completed
theorem calculate_water_needed_for_solution :
  total_water_needed = 0.24 :=
by
  sorry

end calculate_water_needed_for_solution_l331_331069


namespace n_product_expression_l331_331131

theorem n_product_expression (n : ℕ) : n * (n + 1) * (n + 2) * (n + 3) + 1 = (n * (n + 3) + 1)^2 :=
sorry

end n_product_expression_l331_331131


namespace proof_exists_alpha_l331_331779

def exists_alpha (n : ℕ) (P : ℝ × ℝ) (polygon_vertices : list (ℝ × ℝ)) : Prop :=
  (∃ α > 0, ∀ (d1 d2 : ℝ), 
    d1 ∈ (polygon_vertices.map (λ V, real.sqrt ((V.1 - P.1)^2 + (V.2 - P.2)^2))) ∧
    d2 ∈ (polygon_vertices.map (λ V, real.sqrt ((V.1 - P.1)^2 + (V.2 - P.2)^2))) →
    |d1 - d2| < (1 / n) - (α / n^3))

noncomputable def polygon_inscribed_circle (n : ℕ) : list (ℝ × ℝ) :=
  list.of_fn (λ k, (real.cos (2 * real.pi * k / (2 * n + 1)), real.sin (2 * real.pi * k / (2 * n + 1))))

theorem proof_exists_alpha (n : ℕ) (P : ℝ × ℝ) (hP : real.sqrt (P.1^2 + P.2^2) ≤ 1) :
  exists_alpha n P (polygon_inscribed_circle n) :=
sorry

end proof_exists_alpha_l331_331779


namespace ratio_PQR_XYZ_l331_331385

-- Define the conditions given in the problem
variables {X Y Z L M N P Q R : Type}
variables [add_comm_group X] [add_comm_group Y] [add_comm_group Z]
variables [λ YZ : Y → Z, L YZ (1 : ℝ / 3) = L YZ (1 : ℝ / 3)]
variables [λ XZ : X → Z, M XZ (3 : ℝ / 5) = M XZ (3 : ℝ / 5)]
variables [λ XY : X → Y, N XY (2 : ℝ / 3) = N XY (2 : ℝ / 3)]

-- Define the intersection points conditions
variables {XL YM ZN : X → Y → Z}

-- Define the ratio conditions
variables [is_ratio L YZ 1 3]
variables [is_ratio M XZ 3 2]
variables [is_ratio N XY 2 1]

-- Finally, the main theorem to prove the required ratio of areas
theorem ratio_PQR_XYZ (h1 : is_ratio (length (PQR.path YZ)) (length (XYZ.path YZ)) (3/16)) :
  area_ratio PQR XYZ = 3 / 16 :=
sorry

end ratio_PQR_XYZ_l331_331385


namespace max_value_expression_l331_331966

theorem max_value_expression (a b : ℝ) (ha: 0 < a) (hb: 0 < b) :
  ∃ M, M = 2 * Real.sqrt 87 ∧
       (∀ a b: ℝ, 0 < a → 0 < b →
       (|4 * a - 10 * b| + |2 * (a - b * Real.sqrt 3) - 5 * (a * Real.sqrt 3 + b)|) / Real.sqrt (a ^ 2 + b ^ 2) ≤ M) :=
sorry

end max_value_expression_l331_331966


namespace xy_sum_is_36_l331_331239

theorem xy_sum_is_36 
  (x y : ℕ) 
  (data : List ℕ := [9, 10, 12, 15, x, 17, y, 22, 26]) 
  (h_median : List.median data = 16)
  (h_percentile_75 : List.percentile data 75 = 20) : x + y = 36 := 
by 
  sorry

end xy_sum_is_36_l331_331239


namespace min_area_of_MAPB_l331_331560

noncomputable def point (x y : ℝ) := (x, y)

def is_on_line (P : ℝ × ℝ) (A B : ℝ) (C : ℝ) : Prop := A * P.1 + B * P.2 = C

def center_of_circle : ℝ × ℝ := point 0 4

def radius_of_circle : ℝ := 1

def distance (P1 P2 : ℝ × ℝ) : ℝ := real.sqrt ((P1.1 - P2.1) ^ 2 + (P1.2 - P2.2) ^ 2)

def min_area_quadrilateral (l : ℝ × ℝ → Prop) (P : ℝ × ℝ) (M : ℝ × ℝ) (r : ℝ) : ℝ :=
  let d := (| 3 * (M.1 - P.1) + 4 * (M.2 - P.2) |) / real.sqrt (3 ^ 2 + 4 ^ 2),
      PA_sq := d^2 - r^2,
      PA := real.sqrt PA_sq,
      triangle_area := 1/2 * r * PA
  in 2 * triangle_area

theorem min_area_of_MAPB : 
  ∀ (P : ℝ × ℝ), is_on_line P 3 4 1 → 
    min_area_quadrilateral (is_on_line P 3 4 1) P center_of_circle radius_of_circle = 2 * real.sqrt 2 :=
by 
  intros P hP,
  sorry

end min_area_of_MAPB_l331_331560


namespace mary_avg_speed_round_trip_l331_331119

theorem mary_avg_speed_round_trip :
  let distance_to_school := 1.5 -- in km
  let time_to_school := 45 / 60 -- in hours (converted from minutes)
  let time_back_home := 15 / 60 -- in hours (converted from minutes)
  let total_distance := 2 * distance_to_school
  let total_time := time_to_school + time_back_home
  let avg_speed := total_distance / total_time
  avg_speed = 3 := by
  -- Definitions used directly appear in the conditions.
  -- Each condition used:
  -- Mary lives 1.5 km -> distance_to_school = 1.5
  -- Time to school 45 minutes -> time_to_school = 45 / 60
  -- Time back home 15 minutes -> time_back_home = 15 / 60
  -- Route is same -> total_distance = 2 * distance_to_school, total_time = time_to_school + time_back_home
  -- Proof to show avg_speed = 3
  sorry

end mary_avg_speed_round_trip_l331_331119


namespace sum_of_transformed_numbers_l331_331456

theorem sum_of_transformed_numbers (x y S : ℝ) (h : x + y = S) : 
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 := 
by
  sorry

end sum_of_transformed_numbers_l331_331456


namespace hyperbola_focal_point_k_l331_331974

theorem hyperbola_focal_point_k (k : ℝ) :
  (∃ (c : ℝ), c = 2 ∧ (5 : ℝ) * 2 ^ 2 - k * 0 ^ 2 = 5) →
  k = (5 : ℝ) / 3 :=
by
  sorry

end hyperbola_focal_point_k_l331_331974


namespace b_share_is_1400_l331_331207

-- Definitions of the investments
def investment_b (c : ℝ) : ℝ := (2 / 3) * c
def investment_a (b : ℝ) : ℝ := 3 * b

-- Total investment
def total_investment (a b c : ℝ) : ℝ := a + b + c

-- Calculation of the ratio of B's investment over total investment
def b_share_ratio (b c : ℝ) : ℝ := (investment_b c) / (total_investment (investment_a (investment_b c)) (investment_b c) c)

-- The profit and share calculations
def total_profit : ℝ := 7700
def b_profit_share : ℝ := b_share_ratio (investment_b 1) 1 * total_profit

-- The theorem stating that B's share of the profit is Rs. 1400
theorem b_share_is_1400 (c : ℝ) : b_profit_share = 1400 :=
  sorry

end b_share_is_1400_l331_331207


namespace evaluate_expression_l331_331218

theorem evaluate_expression : 
  908 * 501 - (731 * 1389 - (547 * 236 + 842 * 731 - 495 * 361)) = 5448 := by
  sorry

end evaluate_expression_l331_331218


namespace value_in_set_correct_l331_331176

theorem value_in_set_correct :
  ∃ x : ℝ, x = -1 ∧ (2 * x ≠ x^2 + x) ∧ (2 * x ≠ -4) ∧ (x^2 + x ≠ -4) :=
by
  use -1
  split
  · rfl
  split
  · norm_num
  split
  · norm_num
  norm_num
  sorry

end value_in_set_correct_l331_331176


namespace area_of_blackboard_l331_331202

def side_length : ℝ := 6
def area (side : ℝ) : ℝ := side * side

theorem area_of_blackboard : area side_length = 36 := by
  -- proof
  sorry

end area_of_blackboard_l331_331202


namespace james_chartered_limo_for_six_hours_l331_331718

def ticket_cost : ℕ := 100
def dinner_cost : ℕ := 120
def tip_percentage : ℝ := 0.30
def limo_cost_per_hour : ℕ := 80
def total_cost : ℕ := 836

theorem james_chartered_limo_for_six_hours :
  ∃ hours : ℕ, 
  (2 * ticket_cost) + dinner_cost + (tip_percentage * dinner_cost : ℝ).to_nat + (hours * limo_cost_per_hour) = total_cost ∧
  hours = 6 :=
by
  sorry

end james_chartered_limo_for_six_hours_l331_331718


namespace average_marks_all_students_proof_l331_331210

-- Definitions based on the given conditions
def class1_student_count : ℕ := 35
def class2_student_count : ℕ := 45
def class1_average_marks : ℕ := 40
def class2_average_marks : ℕ := 60

-- Total marks calculations
def class1_total_marks : ℕ := class1_student_count * class1_average_marks
def class2_total_marks : ℕ := class2_student_count * class2_average_marks
def total_marks : ℕ := class1_total_marks + class2_total_marks

-- Total student count
def total_student_count : ℕ := class1_student_count + class2_student_count

-- Average marks of all students
noncomputable def average_marks_all_students : ℚ := total_marks / total_student_count

-- Lean statement to prove
theorem average_marks_all_students_proof
  (h1 : class1_student_count = 35)
  (h2 : class2_student_count = 45)
  (h3 : class1_average_marks = 40)
  (h4 : class2_average_marks = 60) :
  average_marks_all_students = 51.25 := by
  sorry

end average_marks_all_students_proof_l331_331210


namespace min_abs_diff_of_extrema_l331_331431

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin ((π / 2) * x + π / 5)

theorem min_abs_diff_of_extrema :
  (∀ x : ℝ, f(x_1) ≤ f(x) ∧ f(x) ≤ f(x_2)) →
  (∃ x1 x2 : ℝ, f(x1) = (-2) ∧ f(x2) = 2 ∧ abs (x1 - x2) = 2) :=
begin
  sorry
end

end min_abs_diff_of_extrema_l331_331431


namespace number_of_zeros_f_l331_331165

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.log x

theorem number_of_zeros_f : ∃! n : ℕ, n = 2 ∧ ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 2 := by
  sorry

end number_of_zeros_f_l331_331165


namespace polar_eq_curve_C_chord_length_intercepted_l331_331339

-- Definitions for the parametric equations of curve C
def curve_C (α : ℝ) : ℝ × ℝ :=
  (2 + sqrt 5 * cos α, 1 + sqrt 5 * sin α)

-- Definition for the polar equation of line l
def polar_line (ρ θ : ℝ) : Prop :=
  ρ * (sin θ + cos θ) = 1

-- Definition for the polar equation of curve C
def polar_curve (ρ θ : ℝ) : Prop :=
  ρ = 4 * cos θ + 2 * sin θ

-- Distance from the center of the curve to the line
def distance_center_to_line : ℝ :=
  sqrt 2

-- Length of the chord derived from the distance
def chord_length : ℝ :=
  2 * sqrt (5 - 2)

-- Theorem for proving the polar equation of curve C
theorem polar_eq_curve_C (ρ θ : ℝ) :
  (∃ α, (ρ, θ) = (rho, theta) ∧ curve_C α = (2 + sqrt 5 * cos α, 1 + sqrt 5 * sin α)) →
  polar_curve ρ θ :=
sorry

-- Theorem for proving the length of the chord intercepted by line l
theorem chord_length_intercepted (ρ θ : ℝ) :
  polar_line ρ θ →
  chord_length = 2 * sqrt 3 :=
sorry

end polar_eq_curve_C_chord_length_intercepted_l331_331339


namespace probability_heart_and_face_card_club_l331_331510

-- Conditions
def num_cards : ℕ := 52
def num_hearts : ℕ := 13
def num_face_card_clubs : ℕ := 3

-- Define the probabilities
def prob_heart_first : ℚ := num_hearts / num_cards
def prob_face_card_club_given_heart : ℚ := num_face_card_clubs / (num_cards - 1)

-- Proof statement
theorem probability_heart_and_face_card_club :
  prob_heart_first * prob_face_card_club_given_heart = 3 / 204 :=
by
  sorry

end probability_heart_and_face_card_club_l331_331510


namespace largest_k_prime_sequence_l331_331616

noncomputable def sequence_x : ℕ → ℕ
| 0 => 0 -- x_0 is not defined, but for ease, set it to 0.
| (n + 1) => a + n

noncomputable def sequence_y (n : ℕ) : ℕ := 2^(sequence_x n) - 1

theorem largest_k_prime_sequence (a : ℕ) (h_pos : a > 0) :
    ∃ (k : ℕ), (∀ n, n ≤ k → nat.prime (sequence_y n)) → k = 2 := by
  sorry

end largest_k_prime_sequence_l331_331616


namespace solve_for_y_l331_331788

theorem solve_for_y (y : ℝ) : (1 / 8)^(3 * y + 12) = (32)^(3 * y + 7) → y = -71 / 24 := by
  sorry

end solve_for_y_l331_331788


namespace negative_result_in_A_l331_331526

-- Definitions of expressions
def A : ℝ := -sqrt (2^2)
def B : ℝ := (sqrt 2)^2
def C : ℝ := sqrt (2^2)
def D : ℝ := sqrt ((-2)^2)

-- Stating the proof problem
theorem negative_result_in_A : A < 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧ D ≥ 0 :=
by {
  sorry
}

end negative_result_in_A_l331_331526


namespace probability_one_letter_each_brother_l331_331920

/-- Allen and James are brothers.
Each brother's name has 6 letters, totaling 12 letters.
Two cards are picked at random without replacement.
Prove the probability that one card is from Allen's name and one from James's name is 6/11.
--/
theorem probability_one_letter_each_brother (total_cards : ℕ) 
  (allen_letters : ℕ) (james_letters : ℕ) : total_cards = 12 
  ∧ allen_letters = 6 
  ∧ james_letters = 6 
  → (6 : ℚ) / (11 : ℚ) :=
by
  intro h
  sorry

end probability_one_letter_each_brother_l331_331920


namespace wanda_blocks_l331_331193

theorem wanda_blocks (initial_blocks: ℕ) (additional_blocks: ℕ) (total_blocks: ℕ) : 
  initial_blocks = 4 → additional_blocks = 79 → total_blocks = initial_blocks + additional_blocks → total_blocks = 83 :=
by
  intros hi ha ht
  rw [hi, ha] at ht
  exact ht

end wanda_blocks_l331_331193


namespace medians_intersect_at_centroid_l331_331804

variables {A B C D O : Type*}

-- Definition of the geometric points A, B, C, and D
def A : Type := sorry
def B : Type := sorry
def C : Type := sorry
def D : Type := sorry

-- Definition of the centroid of the face ABC
def G := (A + B + C) / 3

-- Definition of the centroid of the tetrahedron
def centroid := (3 * G + D) / 4

-- Definitions of the medians from each vertex to the centroid of the opposite face
def medianA := line_segment A (centroid)
def medianB := line_segment B (centroid)
def medianC := line_segment C (centroid)
def medianD := line_segment D (centroid)

-- The theorem to be proven
theorem medians_intersect_at_centroid :
  ∀ (A B C D : Type), (1 : R) :=
begin
  -- Let's start by defining the variables A, B, C, D and the centroid G
  intros A B C D,
  let G := (A + B + C) / 3,
  let O := (3 * G + D) / 4,
  -- Here we assume that the medians are line segments from vertex to centroid
  have hA : A ≠ O, sorry,
  have hB : B ≠ O, sorry,
  have hC : C ≠ O, sorry,
  have hD : D ≠ O, sorry,
  -- We need to show that O is the centroid and all medians intersect at this point
  -- Also, the point O divides each median in the ratio 3:1
  use h,
  sorry
end

end medians_intersect_at_centroid_l331_331804


namespace find_k_l331_331169

noncomputable def f (x : ℝ) := Real.log x + 2 * x - 8

theorem find_k : ∃ k : ℤ, (3 : ℝ) < Real.log 3 + 6 - 8 ∧ Real.log 4 + 8 - 8 > 0 → k = 3 :=
by
  dsimp [f]
  have h₁ : f 3 < 0 := by sorry
  have h₂ : f 4 > 0 := by sorry
  use 3
  split 
  assumption

end find_k_l331_331169


namespace teairras_pants_count_l331_331458

-- Definitions according to the given conditions
def total_shirts := 5
def plaid_shirts := 3
def purple_pants := 5
def neither_plaid_nor_purple := 21

-- The theorem we need to prove
theorem teairras_pants_count :
  ∃ (pants : ℕ), pants = (neither_plaid_nor_purple - (total_shirts - plaid_shirts)) + purple_pants ∧ pants = 24 :=
by
  sorry

end teairras_pants_count_l331_331458


namespace solution_set_l331_331982

def fractionalPart (x : ℝ) : ℝ := x - x.floor

def validIntervals (k : ℤ) : set ℝ :=
  { x | (k + 1/5 : ℝ) ≤ fractionalPart x ∧ fractionalPart x < (k + 1/3 : ℝ) } ∪
  { x | (k + 2/5 : ℝ) ≤ fractionalPart x ∧ fractionalPart x < (k + 3/5 : ℝ) } ∪
  { x | (k + 2/3 : ℝ) ≤ fractionalPart x ∧ fractionalPart x < (k + 4/5 : ℝ) }

def validSet : set ℝ :=
  ⋃ k : ℤ, validIntervals k

theorem solution_set {x : ℝ} :
  (∃ (k : ℤ), x ∈ validIntervals k) ↔ (∃ (k : ℤ), validSet x) :=
sorry

end solution_set_l331_331982


namespace num_polynomials_is_four_l331_331087

def is_polynomial (expr : String) : Prop :=
  expr = "-7" ∨ expr = "m" ∨ expr = "x^3y^2" ∨ expr = "2x+3y"

theorem num_polynomials_is_four : 
  (List.filter is_polynomial ["-7", "m", "x^3y^2", "1/a", "2x+3y"]).length = 4 :=
by
  sorry

end num_polynomials_is_four_l331_331087


namespace distance_from_plate_to_bottom_edge_l331_331212

theorem distance_from_plate_to_bottom_edge :
  ∀ (W T d : ℕ), W = 73 ∧ T = 20 ∧ (T + d = W) → d = 53 :=
by
  intros W T d
  rintro ⟨hW, hT, h⟩
  rw [hW, hT] at h
  linarith

end distance_from_plate_to_bottom_edge_l331_331212


namespace proper_subsets_count_l331_331490

open Finset

def S : Finset (ℤ × ℤ) := 
  {(0,0), (-1,0), (0,-1), (1,0), (0,1)}

theorem proper_subsets_count : 
  S.card = 5 → 2 ^ S.card - 1 = 31 :=
by intros h; rw h; norm_num

end proper_subsets_count_l331_331490


namespace sum_of_fractions_l331_331278

theorem sum_of_fractions : 
  (∑ n in Finset.range 14, (1 : ℚ) / ((n + 1) * (n + 2))) = 14 / 15 := 
by
  sorry

end sum_of_fractions_l331_331278


namespace fraction_product_l331_331933

theorem fraction_product :
  (8 / 4) * (10 / 5) * (21 / 14) * (16 / 8) * (45 / 15) * (30 / 10) * (49 / 35) * (32 / 16) = 302.4 := by
  sorry

end fraction_product_l331_331933


namespace rectangle_ratio_l331_331368

theorem rectangle_ratio (s : ℕ := 1) (A_smaller A_larger : ℕ := 1) (x y : ℕ)
  (h1 : 4 * A_smaller = A_larger)
  (h2 : A_smaller = s^2)
  (h3 : A_larger = (2*s)^2) 
  (h4 : s + 2 * y = 2)
  (h5 : x + s = 2) :
  x = 1 ∧ y = 0.5 ∧ (x / y) = 2 :=
by
  sorry

end rectangle_ratio_l331_331368


namespace range_of_a_l331_331255

variable {a b c d e : ℝ}

-- Definitions of conditions
def condition1 := a + b = 20
def condition2 := a + c = 200
def condition3 := d + e = 2014
def condition4 := c + e = 2000
def condition5 := a < b ∧ b < c ∧ c < d ∧ d < e

-- Theorem to prove
theorem range_of_a (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) (h5 : condition5) : -793 < a ∧ a < 10 :=
sorry

end range_of_a_l331_331255


namespace sum_of_c_and_d_l331_331736

theorem sum_of_c_and_d (c d : ℝ) :
  (∀ x : ℝ, x ≠ 2 → x ≠ -3 → (x - 2) * (x + 3) = x^2 + c * x + d) →
  c + d = -5 :=
by
  intros h
  sorry

end sum_of_c_and_d_l331_331736


namespace initial_courses_of_bricks_l331_331097

theorem initial_courses_of_bricks (x : ℕ) : 
    400 * x + 2 * 400 - 400 / 2 = 1800 → x = 3 :=
by
  sorry

end initial_courses_of_bricks_l331_331097


namespace real_part_fraction_correct_l331_331109

noncomputable def real_part_of_fraction (r θ : ℝ) (h1 : r ≠ 1) (h2 : r > 0) : ℝ :=
  let z := r * Complex.exp (θ * Complex.I) in
  Complex.re (1 / (1 - z ^ 2))

theorem real_part_fraction_correct (r θ : ℝ) (h1 : r ≠ 1) (h2 : r > 0) :
  real_part_of_fraction r θ h1 h2 = 
  (1 - r^2 * Real.cos (2 * θ)) / (1 - 2 * r^2 * Real.cos (2 * θ) + r^4) :=
sorry

end real_part_fraction_correct_l331_331109


namespace fx_extreme_values_at_pm1_m_range_l331_331019

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x^2 - 3 * x

theorem fx_extreme_values_at_pm1
  (a b : ℝ)
  (h_extreme : ∀ x ∈ ({-1, 1} : set ℝ), deriv (f x a b) x = 0) :
  a = 1 ∧ b = 0 :=
by {
  sorry
}

theorem m_range
  (m : ℝ)
  (h_m : m ≠ -2) :
  (∃ x0 : ℝ, 2 * x0^3 - 3 * x0^2 + m + 3 = 0 ∧ ∀ x1 ≠ x0, 2 * x1^3 - 3 * x1^2 + m + 3 ≠ 0) ↔ -3 < m ∧ m < -2 :=
by {
  sorry
}

end fx_extreme_values_at_pm1_m_range_l331_331019


namespace congruent_triangles_have_equal_perimeters_l331_331695

theorem congruent_triangles_have_equal_perimeters 
  (T1 T2 : Triangle) 
  (h_congruent : congruent T1 T2) 
  (h_perimeter_T1 : perimeter T1 = 5) : 
  perimeter T2 = 5 := 
begin 
  sorry 
end

end congruent_triangles_have_equal_perimeters_l331_331695


namespace find_sum_of_numbers_l331_331168

variables (a b c : ℕ) (h_ratio : a * 7 = b * 5 ∧ b * 9 = c * 7) (h_lcm : Nat.lcm a (Nat.lcm b c) = 6300)

theorem find_sum_of_numbers (h_ratio : a * 7 = b * 5 ∧ b * 9 = c * 7) (h_lcm : Nat.lcm a (Nat.lcm b c) = 6300) :
  a + b + c = 14700 :=
sorry

end find_sum_of_numbers_l331_331168


namespace correct_statements_l331_331574

theorem correct_statements :
  let α_set := { α | ∃ n : ℤ, α = n * π + π / 2 }
  let symmetry_center := ∃ k : ℤ, (k * π + 3 * π / 4, 0)
  let tan_increasing := ∀ x y, (0 < x ∧ x < π / 2) → (0 < y ∧ y < π / 2) → x < y → tan x < tan y
  let shifted_sine := ∀ (f : ℝ → ℝ), f = λ x, sin (2 * x - π / 3) → ∃ g, (g = λ x, sin (2 * x)) ∧ (f = λ x, g (x - π / 6))
  (α_set ≠ { α | ∃ k : ℤ, α = k * π / 2 }) →
  symmetry_center →
  ¬tan_increasing →
  shifted_sine

end correct_statements_l331_331574


namespace simplify_expression_l331_331614

variables {a b c : ℝ}

/-- Simplifying the given expression. -/
theorem simplify_expression :
  (√[(11 : ℕ)]( ((√[(3 : ℕ)](a^4 * b^2 * c))^5 * (√a (a^3 * b^2 * c))^4)^3 ) / (√[(11 : ℕ)]( a^5 ))) = a^3 * b^2 * c :=
by sorry

end simplify_expression_l331_331614


namespace donation_amount_is_correct_l331_331580

def stuffed_animals_barbara : ℕ := 9
def stuffed_animals_trish : ℕ := 2 * stuffed_animals_barbara
def stuffed_animals_sam : ℕ := stuffed_animals_barbara + 5
def stuffed_animals_linda : ℕ := stuffed_animals_sam - 7

def price_per_barbara : ℝ := 2
def price_per_trish : ℝ := 1.5
def price_per_sam : ℝ := 2.5
def price_per_linda : ℝ := 3

def total_amount_collected : ℝ := 
  stuffed_animals_barbara * price_per_barbara +
  stuffed_animals_trish * price_per_trish +
  stuffed_animals_sam * price_per_sam +
  stuffed_animals_linda * price_per_linda

def discount : ℝ := 0.10

def final_amount : ℝ := total_amount_collected * (1 - discount)

theorem donation_amount_is_correct : final_amount = 90.90 := sorry

end donation_amount_is_correct_l331_331580


namespace exterior_angle_BAC_of_square_and_heptagon_l331_331913

theorem exterior_angle_BAC_of_square_and_heptagon :
  let interior_angle_of_polygon (n : ℕ) := 180 * (n - 2) / n,
      angle_BAD := interior_angle_of_polygon 7,
      angle_CAD := 90 in
  360 - angle_BAD - angle_CAD = 990 / 7 :=
by
  -- Problem definitions
  let interior_angle_of_polygon (n : ℕ) := 180 * (n - 2) / n
  let angle_BAD := interior_angle_of_polygon 7
  let angle_CAD := 90
  -- Proof body can be filled here
  sorry

end exterior_angle_BAC_of_square_and_heptagon_l331_331913


namespace moments_of_inertia_relation_l331_331101

-- Define the center of mass and moments of inertia
variables {n : Type*} [fintype n] {m : ℝ} {m_i : n → ℝ}
variables {x_i : n → EuclideanSpace ℝ (fin 3)}
variable {a : EuclideanSpace ℝ (fin 3)} -- vector from X to O

-- Center of mass condition
def center_of_mass (m_i : n → ℝ) (x_i : n → EuclideanSpace ℝ (fin 3)) (m : ℝ) : Prop := 
  ∑ i, m_i i • x_i i = 0

-- Moment of inertia definitions
def moment_of_inertia_O (m_i : n → ℝ) (x_i : n → EuclideanSpace ℝ (fin 3)) : ℝ :=
  ∑ i, m_i i * ∥x_i i∥^2

def moment_of_inertia_X (m_i : n → ℝ) (x_i : n → EuclideanSpace ℝ (fin 3)) (a : EuclideanSpace ℝ (fin 3)) : ℝ :=
  ∑ i, m_i i * ∥x_i i + a∥^2

theorem moments_of_inertia_relation
  (m_i : n → ℝ)
  (x_i : n → EuclideanSpace ℝ (fin 3))
  (a : EuclideanSpace ℝ (fin 3))
  (m : ℝ)
  (h_sum_mi_xi : center_of_mass m_i x_i m) :
  moment_of_inertia_X m_i x_i a = moment_of_inertia_O m_i x_i + m * ∥a∥^2 :=
sorry

end moments_of_inertia_relation_l331_331101


namespace equilateral_triangle_min_rotation_angle_l331_331576

-- Definition of an equilateral triangle with rotational symmetry
def is_equilateral_triangle (T : Type) : Prop :=
  T = triangle /\ (∀ θ, θ ≠ 0 ∧ T.rotated_by θ = T)

-- The theorem to prove the minimum rotation angle of an equilateral triangle
theorem equilateral_triangle_min_rotation_angle (T : Type) (hT : is_equilateral_triangle T) : 
    ∃ θ, θ = 120 :=
by
  sorry

end equilateral_triangle_min_rotation_angle_l331_331576


namespace infinitely_many_primes_l331_331764

-- Define the equation as a predicate
def diophantine_eq (p x y : ℤ) : Prop :=
  x^2 + x + 1 = p * y

-- The main theorem to prove infinitude of primes for the equation
theorem infinitely_many_primes : ∃ᶠ p in Nat.primes, ∃ x y : ℤ, diophantine_eq p x y :=
sorry

end infinitely_many_primes_l331_331764


namespace part_A_part_B_part_C_l331_331003

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f(x * y) = y^2 * f(x) + x^2 * f(y)

theorem part_A : f(0) = 0 := sorry
theorem part_B : f(1) = 0 := sorry
theorem part_C : ∀ x : ℝ, f(x) = f(-x) := sorry

end part_A_part_B_part_C_l331_331003


namespace avg_age_zero_l331_331263

variable {K N : ℕ}
variable {P : ℤ}

-- Conditions
axiom sum_positive_ages : ∀ K (ages : Fin K → ℤ), (∀ i, 0 < ages i ∧ ages i ≤ 100) → Sum (λ i => ages i) = P
axiom sum_negative_ages : ∀ N (ages : Fin N → ℤ), (∀ i, -100 ≤ ages i ∧ ages i < 0) → Sum (λ i => ages i) = -P

theorem avg_age_zero (h1 : K + N = 30) (h2 : sum_positive_ages) (h3 : sum_negative_ages) : P + (-P) = 0 :=
by
  sorry

end avg_age_zero_l331_331263


namespace greatest_natural_number_exists_l331_331288

noncomputable def sum_of_squares (n : ℕ) : ℕ :=
    n * (n + 1) * (2 * n + 1) / 6

noncomputable def squared_sum_from_to (a b : ℕ) : ℕ :=
    sum_of_squares b - sum_of_squares (a - 1)

noncomputable def is_perfect_square (n : ℕ) : Prop :=
    ∃ k, k * k = n

theorem greatest_natural_number_exists :
    ∃ n : ℕ, n = 1921 ∧ n ≤ 2008 ∧ 
    is_perfect_square ((sum_of_squares n) * (squared_sum_from_to (n + 1) (2 * n))) :=
by
  sorry

end greatest_natural_number_exists_l331_331288


namespace common_area_of_rectangle_and_circle_l331_331910

theorem common_area_of_rectangle_and_circle :
  let l := 10
  let w := 2 * Real.sqrt 5
  let r := 3
  ∃ (common_area : ℝ), common_area = 9 * Real.pi :=
by
  let l := 10
  let w := 2 * Real.sqrt 5
  let r := 3
  have common_area := 9 * Real.pi
  use common_area
  sorry

end common_area_of_rectangle_and_circle_l331_331910


namespace find_a_range_of_f_on_interval_l331_331662

-- We need to deal with real numbers and functions
open Real

-- Part (1): Definition of function and proving it is odd implies a = 1/2
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem find_a : ∀ a : ℝ, is_odd_function (λ x, 1 / (2 ^ x + 1) - a) → a = 1 / 2 :=
begin
  sorry
end

-- Part (2): Range of the function on [-1, 3]
def f (x : ℝ) : ℝ := 1 / (2 ^ x + 1) - 1 / 2

theorem range_of_f_on_interval : set.range (λ x, f x) ∩ set.Icc (-1) 3 = set.Icc (-7 / 18) (1 / 6) :=
begin
  sorry
end

end find_a_range_of_f_on_interval_l331_331662


namespace erased_number_is_seven_l331_331238

theorem erased_number_is_seven:
  ∃ (n x : ℕ), (∀ k, k ∈ (range (n + 1)) → k ≠ x) ∧ 
               (n > 1) ∧ 
               (∑ i in (range (n + 1)), i - x = (35 * 17 + 7) * (n - 1) := 
begin
  sorry
end

end erased_number_is_seven_l331_331238


namespace cubic_coefficients_identity_l331_331562

-- Definitions of the constants.
variables {p q r s : ℝ}

-- Definition of the cubic function g(x).
def g (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

-- Given condition: Point (-3, 4) lies on g(x), hence g(-3) = 4.
axiom h₁ : g (-3) = 4

-- The proof statement for 10p - 5q + 3r - 2s = 40.
theorem cubic_coefficients_identity : 10 * p - 5 * q + 3 * r - 2 * s = 40 :=
by
  -- Proof skipped.
  sorry

end cubic_coefficients_identity_l331_331562


namespace diamonds_in_P10_l331_331237

def diamond_seq (n : ℕ) : ℕ :=
  if n = 1 then 4
  else (diamond_seq (n - 1) + 4 * (2 * n - 1))

theorem diamonds_in_P10 : diamond_seq 10 = 400 := 
sorry

end diamonds_in_P10_l331_331237


namespace parabola_intersects_y_axis_vertex_of_parabola_find_k_l331_331713

noncomputable def parabola_eq (b x : ℝ) : ℝ := x^2 + b * x + b - 1

def vertex (b : ℝ) : ℝ × ℝ := (- 1 / 2 * b, - 1 / 4 * b^2 + b - 1)

def line_eq (k x : ℝ) : ℝ := k * x - 1

theorem parabola_intersects_y_axis (b : ℝ) (hb : b ≠ 0) : 
  ∃ y : ℝ, parabola_eq b 0 = y := 
by
  use b - 1
  simp [parabola_eq]

theorem vertex_of_parabola (b : ℝ) : 
  vertex b = (- 1 / 2 * b, - 1 / 4 * b^2 + b - 1) := 
by
  simp [vertex]

theorem find_k (b k : ℝ) (hb : b = 2 * k + 4) (hOAk : 
  ∃ A : ℝ × ℝ, sqrt ((A.1)^2 + (A.2)^2) = sqrt 5 ∧ A = (-2, line_eq k (-2))) :
  k = -1 :=
by
  sorry

end parabola_intersects_y_axis_vertex_of_parabola_find_k_l331_331713


namespace arithmetic_geometric_sequence_l331_331247

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : a 2 * a 5 = (a 4) ^ 2)
  (h4 : d ≠ 0) : d = -1 / 5 :=
by
  sorry

end arithmetic_geometric_sequence_l331_331247


namespace cos_square_sub_exp_zero_l331_331883

theorem cos_square_sub_exp_zero : 
  (cos (30 * Real.pi / 180))^2 - (2 - Real.pi) ^ 0 = -1 / 4 := by
  sorry

end cos_square_sub_exp_zero_l331_331883


namespace range_of_a_l331_331028

theorem range_of_a (a : ℝ) 
  (h : ∀ x y, (a * x^2 - 3 * x + 2 = 0) ∧ (a * y^2 - 3 * y + 2 = 0) → x = y) :
  a = 0 ∨ a ≥ 9/8 :=
sorry

end range_of_a_l331_331028


namespace algebraic_expression_value_l331_331038

theorem algebraic_expression_value (m: ℝ) (h: m^2 + m - 1 = 0) : 2023 - m^2 - m = 2022 := 
by 
  sorry

end algebraic_expression_value_l331_331038


namespace car_speed_first_hour_l331_331175

theorem car_speed_first_hour (x : ℕ) :
  (x + 60) / 2 = 75 → x = 90 :=
by
  -- To complete the proof in Lean, we would need to solve the equation,
  -- reversing the steps provided in the solution. 
  -- But as per instructions, we don't need the proof, hence we put sorry.
  sorry

end car_speed_first_hour_l331_331175


namespace area_of_smaller_rectangle_l331_331133

theorem area_of_smaller_rectangle (L W : ℝ) (hL : 3 * W = 2 * L) (hArea : L * W = 48) :
  let P := L / 2
  let Q := W / 2
  (P * Q = 12) :=
by
  -- Definitions and conditions
  let x := L / 3
  have hW_eq : W = 2 * x := by
    rw [← div_eq_iff (ne_of_gt zero_lt_three), ← mul_div_assoc, mul_comm,
        ← mul_div_assoc, ← mul_div_assoc, ← hL, mul_div_cancel' _ two_ne_zero]
  have hL_calc: L = 3 * x := by
    rw [nat.cast_two, nat.cast_three, hL, ←eq_div_iff (ne_of_gt zero_lt_two), 
        ←eq_div_iff (ne_of_gt zero_lt_two), mul_div_left_comm, mul_div_left_comm, 
        mul_comm]

  have hArea_def : (3 * x) * (2 * x) = 48 := by
    rw [hL_calc, hW_eq]
    -- Conclusion after algebraic manipulations
    sorry
    
  have h_x_value : x = 2 * sqrt 2 := sorry

  have hSmallerLength : P = 3 * sqrt 2 := by
    rw [← div_eq_iff (ne_of_gt zero_lt_three)]
    -- Conclusion after algebraic manipulations
    sorry

  have hSmallerWidth : Q = 2 * sqrt 2 := by
    rw [← div_eq_iff (ne_of_gt zero_lt_two)]
    -- Conclusion after algebraic manipulations
    sorry

  -- The main goal: showing area of the smaller rectangle
  calc (3 * sqrt 2) * (2 * sqrt 2) = 6 * (sqrt 2 * sqrt 2) :=
          by rw [←mul_assoc]
      ... = 6 * 2 := by rw [mul_self_sqrt (le_of_lt (by norm_num)] 
      ... = 12 := by norm_num

end area_of_smaller_rectangle_l331_331133


namespace total_votes_l331_331581

theorem total_votes (Ben_votes Matt_votes total_votes : ℕ)
  (h_ratio : 2 * Matt_votes = 3 * Ben_votes)
  (h_Ben_votes : Ben_votes = 24) :
  total_votes = Ben_votes + Matt_votes :=
sorry

end total_votes_l331_331581


namespace range_of_f_minimal_lambda_l331_331021

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1) / (Real.exp x + 1)

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ∈ set.Ioo (-1 : ℝ) 1 :=
sorry

theorem minimal_lambda :
  ∃ (λ : ℝ) (h : λ = 1 / 2), ∀ x₁ x₂ ∈ set.Icc (Real.log (1 / 2)) (Real.log 2),
  abs ((f x₁ + f x₂) / (x₁ + x₂)) < λ :=
sorry

end range_of_f_minimal_lambda_l331_331021


namespace ratio_p_q_l331_331274

section ProbabilityProof

-- Definitions and constants as per conditions
def N := Nat.factorial 15

def num_ways_A : ℕ := 4 * (Nat.factorial 9 / (Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 3))
def num_ways_B : ℕ := 4 * 3

def p : ℚ := num_ways_A / N
def q : ℚ := num_ways_B / N

-- Theorem: Prove that the ratio p/q is 560
theorem ratio_p_q : p / q = 560 := by
  sorry

end ProbabilityProof

end ratio_p_q_l331_331274


namespace tan_105_eq_neg2_sub_sqrt3_l331_331944

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l331_331944


namespace inequality_proof_l331_331988

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  1 / (1 - x^2) + 1 / (1 - y^2) ≥ 2 / (1 - x * y) :=
sorry

end inequality_proof_l331_331988


namespace remaining_unit_area_l331_331152

theorem remaining_unit_area
    (total_units : ℕ)
    (total_area : ℕ)
    (num_12x6_units : ℕ)
    (length_12x6_unit : ℕ)
    (width_12x6_unit : ℕ)
    (remaining_units_area : ℕ)
    (num_remaining_units : ℕ)
    (remaining_unit_area : ℕ) :
  total_units = 72 →
  total_area = 8640 →
  num_12x6_units = 30 →
  length_12x6_unit = 12 →
  width_12x6_unit = 6 →
  remaining_units_area = total_area - (num_12x6_units * length_12x6_unit * width_12x6_unit) →
  num_remaining_units = total_units - num_12x6_units →
  remaining_unit_area = remaining_units_area / num_remaining_units →
  remaining_unit_area = 154 :=
by
  intros h_total_units h_total_area h_num_12x6_units h_length_12x6_unit h_width_12x6_unit h_remaining_units_area h_num_remaining_units h_remaining_unit_area
  sorry

end remaining_unit_area_l331_331152


namespace chase_blue_jays_count_l331_331205

theorem chase_blue_jays_count:
  ∃ (x : ℝ), 
  let gabrielle_birds := 12 in
  let chase_birds := 2 + x + 5 in
  gabrielle_birds = chase_birds + 0.20 * chase_birds ∧ x = 3 :=
sorry

end chase_blue_jays_count_l331_331205


namespace identity_x_squared_minus_y_squared_l331_331049

theorem identity_x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : 
  x^2 - y^2 = 80 := by sorry

end identity_x_squared_minus_y_squared_l331_331049


namespace sum_difference_even_odd_l331_331932

theorem sum_difference_even_odd : 
    let even_sum := ∑ i in finset.range 100, (i + 1) * 2 in
    let odd_sum := ∑ i in finset.range 100, (i + 1) * 2 - 1 in
    even_sum - odd_sum = 100 :=
by
    sorry

end sum_difference_even_odd_l331_331932


namespace T_2n_l331_331265

-- Given conditions
def a1 := 3
def b1 := 1
def S2 := λ (a1 a2 : ℕ), a1 + a2
def b2 := λ (q b1 : ℕ), q * b1
def eq1 (d q : ℕ) := 3 + 4 * d - 2 * q = 10
def q := 2

-- General terms for sequences a_n and b_n
def a_n := λ n, 2 * n + 1
def b_n := λ n, 2^(n - 1)

-- Definition of sequence c_n
def c_n := λ (n : ℕ), (1 / (2 * (n / 2) * (2 * a1 + (n - 1) * 2))) + 2^(n - 1)

-- Sum of first n terms of c_n
def T_n := λ (n : ℕ), ∑ i in finset.range (n + 1), c_n i

-- Desired result
theorem T_2n (n : ℕ) :
  T_n (2 * n) = 2^(2 * n) - 5 / 8 - (1 / 4) * (1 / (2 * n + 1) + 1 / (2 * n + 2)) := sorry

end T_2n_l331_331265


namespace edge_length_A₁A_l331_331812

/-- Conditions given for part (a) -/
variables (A B C D A₁ B₁ C₁ D₁ : Type)

/-- Edge lengths given -/
variables (CK KD : ℝ)
variables (h1 : CK = 4) (h2 : KD = 1)

/-- Prove the edge length A₁A is 8 -/
theorem edge_length_A₁A (h_perp : A₁ ∈ [⊥ BCC₁B₁]) : A₁A = 8 :=
sorry

end edge_length_A₁A_l331_331812


namespace discount_is_25_percent_l331_331480

noncomputable def discount_percentage (M : ℝ) (C : ℝ) (SP : ℝ) : ℝ :=
  ((M - SP) / M) * 100

theorem discount_is_25_percent (M : ℝ) (C : ℝ) (SP : ℝ) 
  (h1 : C = 0.64 * M) 
  (h2 : SP = C * 1.171875) : 
  discount_percentage M C SP = 25 := 
by 
  sorry

end discount_is_25_percent_l331_331480


namespace range_of_T_n_l331_331304

variable (n : ℕ)

def a_n (n : ℕ) : ℕ := 2 * n

def sequence_term (n : ℕ) : ℝ :=
  if h : n > 0 then
    let a_n_val := a_n n
    4 / (a_n_val * (a_n_val + 2))
  else
    0

def T_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, sequence_term (i + 1)

theorem range_of_T_n : 
  ∃ S, S = {T_n k | n ∈ { k : ℕ | k > 0} } :=
  sorry

end range_of_T_n_l331_331304


namespace correct_inequality_l331_331112

variable (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b)

theorem correct_inequality : (1 / (a * b^2)) < (1 / (a^2 * b)) :=
by
  sorry

end correct_inequality_l331_331112


namespace sequence_below_3_l331_331486

noncomputable def f (x : ℝ) : ℝ := (x+1)^2 / 4 + (1 / 2) * Real.log (Real.abs (x-1))

noncomputable def f_prime (x : ℝ) : ℝ := (x + 1) / 2 + 1 / (2 * (x - 1))

noncomputable def sequence_a (a : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then a
  else f_prime (sequence_a a (n - 1))

theorem sequence_below_3 (a : ℝ) (n : ℕ) (h₁ : a > 3) (h₂ : n ≥ (Real.log(a / 3) / Real.log(4 / 3))) :
  sequence_a a n < 3 :=
sorry

end sequence_below_3_l331_331486


namespace solve_for_y_l331_331795

theorem solve_for_y :
  (∀ y : ℝ, (1 / 8) ^ (3 * y + 12) = 32 ^ (3 * y + 7)) →
  y = -71 / 24 :=
begin
  sorry
end

end solve_for_y_l331_331795


namespace method_comparable_to_euclidean_algorithm_l331_331086

theorem method_comparable_to_euclidean_algorithm :
  ∃ method, method = "Method of Continued Proportionate Reduction" ∧
    (method = "Chinese Remainder Theorem" ∨
     method = "Method of Continued Proportionate Reduction" ∨
     method = "Method of Circular Approximation" ∨
     method = "Qin Jiushao's Algorithm") :=
by 
  use "Method of Continued Proportionate Reduction"
  split
  case left => rfl
  case right => 
    apply Or.inr
    apply Or.inl
    rfl

end method_comparable_to_euclidean_algorithm_l331_331086


namespace final_pie_count_l331_331267

def daily_rate (cooper_ella_tyler_rate : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ := 
  let (cooper_rate, ella_rate, tyler_rate) := cooper_ella_tyler_rate in
  (cooper_rate * 7 / 2, ella_rate * 7 / 2, tyler_rate * 7 / 2) -- Rate considering 5 normal days + 1 double day + 1 half day

def pies_in_two_weeks (daily_rate : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ := 
  let (cooper_daily, ella_daily, tyler_daily) := daily_rate in
  (cooper_daily * 14, ella_daily * 14, tyler_daily * 14)

def spoilage_adjusted (total_pies : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  let (cooper_pies, ella_pies, tyler_pies) := total_pies in
  (cooper_pies - (cooper_pies / 10), ella_pies - (ella_pies * 5 / 100), tyler_pies - (tyler_pies * 15 / 100))

def final_pies (adjusted_pies : ℕ × ℕ × ℕ) (consumed : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  let (cooper_adjusted, ella_adjusted, tyler_adjusted) := adjusted_pies in
  let (cooper_consumed, ella_consumed, tyler_consumed) := consumed in
  (cooper_adjusted - cooper_consumed, ella_adjusted - ella_consumed, tyler_adjusted - tyler_consumed)

theorem final_pie_count : 
  let cooper_ella_tyler_rate := (7, 5, 9)
  let consumed := (12, 7, 5) in
  final_pies 
    (spoilage_adjusted (pies_in_two_weeks (daily_rate cooper_ella_tyler_rate)))
    consumed = (82, 64, 109) :=
by sorry

end final_pie_count_l331_331267


namespace fractions_period_equal_l331_331825

theorem fractions_period_equal {a b c d : ℕ} (hab : Nat.gcd a b = 1) (hcd : Nat.gcd c d = 1)
  (h_periodic_a : ∃ A B, fraction_periodic a b A B) (h_periodic_c : ∃ C D, fraction_periodic c d C D)
  (h_substrings : ∀ n, ∃ m, substring_of_length n B m D) : b = d :=
sorry

/-- A definition that captures the concept of a periodic decimal fraction -/
def fraction_periodic (num denom : ℕ) (A B : list ℕ) : Prop :=
  ∃ m, numerator_of_fraction num denom A B = m

/-- A definition that models the condition that every finite block of digits in one block appears in the other -/
def substring_of_length (n m : ℕ) (B D : list ℕ) : Prop :=
  ∀ (i : ℕ), list.take n (list.drop i B) = list.take m (list.drop i D)

noncomputable def numerator_of_fraction (num denom : ℕ) (A B : list ℕ) : ℕ := sorry

end fractions_period_equal_l331_331825


namespace percentage_invalid_votes_l331_331084

theorem percentage_invalid_votes 
    (total_votes : ℕ)
    (candidate_A_votes : ℕ)
    (candidate_A_percentage : ℝ)
    (total_valid_percentage : ℝ) :
    total_votes = 560000 ∧
    candidate_A_votes = 357000 ∧
    candidate_A_percentage = 0.75 ∧
    total_valid_percentage = 100 - x ∧
    (0.75 * (total_valid_percentage / 100) * 560000 = 357000) →
    x = 15 :=
by
  sorry

end percentage_invalid_votes_l331_331084


namespace range_of_func_l331_331591

noncomputable def func (x : ℝ) : ℝ := 2 * (real.cos x)^2 + 3 * (real.sin x) + 3

theorem range_of_func : 
  set.range (λ x : {x: ℝ // x ≥ real.pi / 6 ∧ x ≤ 2 * real.pi / 3}, func x) = set.Icc 6 (49 / 8) :=
sorry

end range_of_func_l331_331591


namespace expression_simplifies_to_4k6_l331_331780

noncomputable def simplify_expression (k : ℝ) : ℝ :=
  ((1 / (2 * k)) ^ (-2)) * ((-k) ^ 4)

theorem expression_simplifies_to_4k6 (k : ℝ) (hk : k ≠ 0) : 
simplify_expression k = 4 * (k ^ 6) :=
by
  sorry

end expression_simplifies_to_4k6_l331_331780


namespace minimum_cost_for_Dorokhov_family_vacation_l331_331464

theorem minimum_cost_for_Dorokhov_family_vacation :
  let globus_cost := (25400 * 3) * 0.98,
      around_world_cost := (11400 + (23500 * 2)) * 1.01
  in
  min globus_cost around_world_cost = 58984 := by
  let globus_cost := (25400 * 3) * 0.98
  let around_world_cost := (11400 + (23500 * 2)) * 1.01
  sorry

end minimum_cost_for_Dorokhov_family_vacation_l331_331464


namespace inequalities_correct_l331_331998

-- Define the basic conditions
variables {a b c d : ℝ}

-- Conditions given in the problem
axiom h1 : a > b
axiom h2 : b > 0
axiom h3 : 0 > c
axiom h4 : c > d

-- Correct answers to be proven
theorem inequalities_correct (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
  (a + c > b + d) ∧ (a * d^2 > b * c^2) ∧ (1 / (b * c) < 1 / (a * d)) :=
begin
  -- Proof part
  sorry
end

end inequalities_correct_l331_331998


namespace tent_cost_solution_l331_331515

-- We define the prices of the tents and other relevant conditions.
def tent_costs (m n : ℕ) : Prop :=
  2 * m + 4 * n = 5200 ∧ 3 * m + n = 2800

-- Define the condition for the number of tents and constraints.
def optimal_tent_count (x : ℕ) (w : ℕ) : Prop :=
  x + (20 - x) = 20 ∧ x ≤ (20 - x) / 3 ∧ w = 600 * x + 1000 * (20 - x)

-- The main theorem to be proven in Lean.
theorem tent_cost_solution :
  ∃ m n, tent_costs m n ∧ m = 600 ∧ n = 1000 ∧
  ∃ x, optimal_tent_count x 18000 ∧ x = 5 ∧ (20 - x) = 15 :=
by
  sorry

end tent_cost_solution_l331_331515


namespace geometric_sequence_common_ratio_l331_331311

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h_pos : ∀ n, a n > 0) 
  (h_geo : ∀ n, a (n + 1) = a n * q)
  (h_ari : 3 * a 0, (1 / 2) * a 2, 2 * a 1 form_arithmetic_sequence) : 
  q = 3 := 
sorry

end geometric_sequence_common_ratio_l331_331311


namespace vecMA_dotProduct_vecBA_range_l331_331641

-- Define the conditions
def pointM : ℝ × ℝ := (1, 0)

def onEllipse (p : ℝ × ℝ) : Prop := (p.1^2 / 4 + p.2^2 = 1)

def vecMA (A : ℝ × ℝ) := (A.1 - pointM.1, A.2 - pointM.2)
def vecMB (B : ℝ × ℝ) := (B.1 - pointM.1, B.2 - pointM.2)
def vecBA (A B : ℝ × ℝ) := (A.1 - B.1, A.2 - B.2)

def dotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the statement
theorem vecMA_dotProduct_vecBA_range (A B : ℝ × ℝ) (α : ℝ) :
  onEllipse A → onEllipse B → dotProduct (vecMA A) (vecMB B) = 0 → 
  A = (2 * Real.cos α, Real.sin α) → 
  (2/3 ≤ dotProduct (vecMA A) (vecBA A B) ∧ dotProduct (vecMA A) (vecBA A B) ≤ 9) :=
sorry

end vecMA_dotProduct_vecBA_range_l331_331641


namespace range_of_m_l331_331313

variable (m : ℝ)

def delta1 := m^2 - 4
def delta2 := 16 * (m - 2)^2 - 16

def p : Prop := delta1 > 0 ∧ -m < 0 ∧ 1 > 0
def q : Prop := delta2 < 0

theorem range_of_m (h_or : p m ∨ q m) (h_and : ¬ (p m ∧ q m)) : m ∈ Set.Ioo 1 2 ∪ Set.Ici 3 :=
by
    sorry

end range_of_m_l331_331313


namespace election_winner_votes_l331_331182

-- Define the conditions and question in Lean 4
theorem election_winner_votes (V : ℝ) (h1 : V > 0) 
  (h2 : 0.54 * V - 0.46 * V = 288) : 0.54 * V = 1944 :=
by
  sorry

end election_winner_votes_l331_331182


namespace constant_term_in_product_l331_331196

def poly1 (x : ℝ) : ℝ := x ^ 4 + 3 * x ^ 2 + 6
def poly2 (x : ℝ) : ℝ := 2 * x ^ 3 + x ^ 2 + 10

theorem constant_term_in_product : (poly1(0) * poly2(0)) = 60 := by
  sorry

end constant_term_in_product_l331_331196


namespace integral_solve_l331_331582

theorem integral_solve : 
  ∫ (x : ℝ) in 0..2, (x^2 + x - 1) * exp(x / 2) = 2 * (3 * Real.exp 1 - 5) :=
by
  sorry

end integral_solve_l331_331582


namespace cone_lateral_surface_area_l331_331652

noncomputable def lateral_surface_area (r l : ℝ) : ℝ :=
π * r * l

theorem cone_lateral_surface_area :
  lateral_surface_area 3 4 = 12 * π :=
by
  sorry

end cone_lateral_surface_area_l331_331652


namespace breadth_of_rectangular_plot_l331_331162

theorem breadth_of_rectangular_plot
  (b l : ℕ)
  (h1 : l = 3 * b)
  (h2 : l * b = 2028) :
  b = 26 :=
sorry

end breadth_of_rectangular_plot_l331_331162


namespace largest_radius_of_congruent_spheres_in_cube_eq_half_l331_331459

noncomputable def largest_possible_radius : ℝ :=
  let l := 2 -- side length of the cube
  let r := 1 / 2 -- radius of each sphere
  r

theorem largest_radius_of_congruent_spheres_in_cube_eq_half :
  let l := 2 in -- side length of the cube
  let n := 10 in -- number of spheres
  let r := largest_possible_radius in -- radius of each sphere
  l = 2 →
  n = 10 →
  r = 1 / 2 →
  r = largest_possible_radius :=
by
  intros h_l h_n h_r
  sorry

end largest_radius_of_congruent_spheres_in_cube_eq_half_l331_331459


namespace mary_final_weight_l331_331749

theorem mary_final_weight :
    let initial_weight := 99
    let initial_loss := 12
    let first_gain := 2 * initial_loss
    let second_loss := 3 * initial_loss
    let final_gain := 6
    let weight_after_first_loss := initial_weight - initial_loss
    let weight_after_first_gain := weight_after_first_loss + first_gain
    let weight_after_second_loss := weight_after_first_gain - second_loss
    let final_weight := weight_after_second_loss + final_gain
    in final_weight = 81 :=
by
    sorry

end mary_final_weight_l331_331749


namespace new_person_weight_l331_331529

theorem new_person_weight (W : ℝ) (N : ℝ) (avg_increase : ℝ := 2.5) (replaced_weight : ℝ := 35) :
  (W - replaced_weight + N) = (W + (8 * avg_increase)) → N = 55 := sorry

end new_person_weight_l331_331529


namespace identity_x_squared_minus_y_squared_l331_331046

theorem identity_x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : 
  x^2 - y^2 = 80 := by sorry

end identity_x_squared_minus_y_squared_l331_331046


namespace computer_additions_per_hour_l331_331226

def operations_per_second : ℕ := 15000
def additions_per_second : ℕ := operations_per_second / 2
def seconds_per_hour : ℕ := 3600

theorem computer_additions_per_hour : 
  additions_per_second * seconds_per_hour = 27000000 := by
  sorry

end computer_additions_per_hour_l331_331226


namespace chord_length_l331_331229

-- Define the circle using the given equation
def circle (x y : ℝ) := x^2 + y^2 - 4 * y = 0

-- Define the line with slope 60 degrees passing through the origin
def line (x y : ℝ) := y = sqrt 3 * x

-- The proof for the length of the chord formed by the intersection
theorem chord_length :
  ∃ l : ℝ, 
    (∀ x y : ℝ, circle x y → line x y) →
    l = 2 * sqrt 3 :=
by
  -- Proof steps would go here
  sorry

end chord_length_l331_331229


namespace simplify_and_evaluate_expression_l331_331144

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.sqrt 3 + 1) :
  (1 - 1 / m) / ((m ^ 2 - 2 * m + 1) / m) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l331_331144


namespace quadratic_decreasing_interval_l331_331666

theorem quadratic_decreasing_interval (b c : ℝ) (h : ∀ x : ℝ, x ≤ 1 → deriv (λ y, y = x^2 + b * x + c) x ≤ 0) : b = -2 :=
by
  sorry

end quadratic_decreasing_interval_l331_331666


namespace neg_p_l331_331340

theorem neg_p :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by
  sorry

end neg_p_l331_331340


namespace number_of_valid_arrangements_l331_331704

theorem number_of_valid_arrangements : 
  let arranged_digits : List (List ℕ) :=
    [
    [a, b, c],
    [1, d, e],
    [f, g, h]
    ],
    in
    (a = 9) ∧ (e = 1) ∧ 
    ( ∀ (x ∈ [b, d]), x ∈ [6, 7, 8]) ∧ 
    ( ∀ (y ∈ [f, h]), y ∈ [2, 3, 4]) ∧
    (List.nodup (arranged_digits.flatten)) →
    (List.length (all_valid_permutations arranged_digits) = 30) :=
sorry


end number_of_valid_arrangements_l331_331704


namespace find_teacher_age_l331_331805

theorem find_teacher_age (S T : ℕ) (h1 : S / 19 = 20) (h2 : (S + T) / 20 = 21) : T = 40 :=
sorry

end find_teacher_age_l331_331805


namespace inequalities_correct_l331_331997

-- Define the basic conditions
variables {a b c d : ℝ}

-- Conditions given in the problem
axiom h1 : a > b
axiom h2 : b > 0
axiom h3 : 0 > c
axiom h4 : c > d

-- Correct answers to be proven
theorem inequalities_correct (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
  (a + c > b + d) ∧ (a * d^2 > b * c^2) ∧ (1 / (b * c) < 1 / (a * d)) :=
begin
  -- Proof part
  sorry
end

end inequalities_correct_l331_331997


namespace sqrt_expression_domain_l331_331592

theorem sqrt_expression_domain (x : ℝ) : 1 < x ↔ ∃ y, (y = 1 / (real.sqrt (x - 1))) :=
by
  sorry

end sqrt_expression_domain_l331_331592


namespace identify_fake_child_l331_331499

variables (children : Fin 111 → ℝ)
variables (tells_truth : Fin 111 → Bool)
variables (lighter : Fin 111)
variable (teacher_can_identify_fake : Bool)

def is_fake_child (i : Fin 111) : Bool :=
  if tells_truth i then children i != children 0 else children i < children 0

-- The statement we want to prove
theorem identify_fake_child (h1 : ∀ i ≠ lighter, tells_truth i = true)
    (h2 : tells_truth lighter = false)
    (h3 : ∀ i ≠ lighter, children i = children 0)
    (h4 : children lighter < children 0) :
  ∃ i, is_fake_child i = true := 
sorry

end identify_fake_child_l331_331499


namespace choose_numbers_l331_331214

theorem choose_numbers (n : ℕ) (h_even : n % 2 = 0) :
  (∑ a in (Finset.range (n + 1)),
    ∑ b in (Finset.range (n + 1)).filter (λ x, x < a),
    ∑ c in (Finset.range (n + 1)).filter (λ x, x < b),
    ∑ d in (Finset.range (n + 1)).filter (λ x, x < c),
      if a + c = b + d then 1 else 0)
  = n * (n - 2) * (2 * n - 5) / 24 := 
sorry

end choose_numbers_l331_331214


namespace solve_problem_l331_331721

noncomputable def proof_problem (k m n : ℕ) (hk : Real.IsRelPrime m n) : Prop :=
  let acid_content_A := 1.8
  let acid_content_B := 2.4
  let acid_content_transfer_to_A := (k / 100.0) * (m / n)
  let acid_content_transfer_to_B := (k / 100.0) * (1 - m / n)
  let final_acid_A := acid_content_A + acid_content_transfer_to_A
  let final_acid_B := acid_content_B + acid_content_transfer_to_B
  let new_volume_A := 4 + (m / n)
  let new_volume_B := 6 - (m / n)
  let concentration_A := final_acid_A / new_volume_A
  let concentration_B := final_acid_B / new_volume_B
  concentration_A = 0.5 ∧ concentration_B = 0.5 → k + m + n = 85

theorem solve_problem : ∃ k m n : ℕ, k + m + n = 85 ∧ Real.IsRelPrime m n ∧ proof_problem k m n Real.isRelPrime :=
begin
  sorry -- the steps of the proof are skipped here
end

end solve_problem_l331_331721


namespace factor_x12_minus_1_l331_331601

theorem factor_x12_minus_1 :
  ∃ (p : ℕ → polynomial ℝ), (x : ℝ) → polynomial.eval x (polynomial.X^12 - 1) = polynomial.eval x (p 0 * p 1 * p 2 * p 3 * p 4 * p 5) ∧
      ∀ i, p i.degree > 0 ∧ polynomial.coeff i j ∈ ℝ :=
begin
  sorry
end

end factor_x12_minus_1_l331_331601


namespace fare_20km_fare_general_distance_22yuan_distance_fare_l331_331841

/-- Representing the fare calculation based on distance traveled -/
def fare (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 3 then 10
  else if 3 < x ∧ x ≤ 18 then 10 + (x - 3)
  else if 18 < x then 25 + 2 * (x - 18)
  else 0  -- Assuming no charges for non-positive distances to handle edge case

/-- Prove the fare for 20 km travel is 29 yuan -/
theorem fare_20km : fare 20 = 29 := by
  sorry

/-- Prove the fare for x km travel under different conditions -/
theorem fare_general (x : ℝ) (hx : 0 < x) : 
  (x ≤ 3 → fare x = 10) ∧
  (3 < x ∧ x ≤ 18 → fare x = x + 7) ∧
  (18 < x → fare x = 2 * x - 11) := by
  sorry

/-- Prove the distance traveled is 15 km for a fare of 22 yuan -/
theorem distance_22yuan : ∃ d, fare d = 22 ∧ d = 15 := by
  sorry

/-- Prove the distance traveled is based on fare of 10 + x yuan (x > 0) -/
theorem distance_fare (x : ℝ) (hx : 0 < x) : 
  (x ≤ 15 → ∃ d, fare d = 10 + x ∧ d = 3 + x) ∧
  (x > 15 → ∃ d, fare d = 10 + x ∧ d = 18 + ...) := by
  sorry

end fare_20km_fare_general_distance_22yuan_distance_fare_l331_331841


namespace number_line_problem_l331_331754

theorem number_line_problem (x : ℤ) (h : x + 7 - 4 = 0) : x = -3 :=
by
  -- The proof is omitted as only the statement is required.
  sorry

end number_line_problem_l331_331754


namespace identity_x_squared_minus_y_squared_l331_331048

theorem identity_x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : 
  x^2 - y^2 = 80 := by sorry

end identity_x_squared_minus_y_squared_l331_331048


namespace slips_probability_ratio_l331_331782

theorem slips_probability_ratio :
  let n := 60
  let k := 5
  let number_of_slips_per_number := 5
  let total_numbers := 12
  let p := (total_numbers : ℚ) / (nat.choose n k : ℚ)
  let q := (nat.choose total_numbers 2 * nat.choose number_of_slips_per_number 3 * nat.choose number_of_slips_per_number 2 : ℚ) / (nat.choose n k : ℚ)
  q / p = 550 :=
by
  sorry

end slips_probability_ratio_l331_331782


namespace sum_first_five_units_digit_three_primes_eq_135_l331_331256

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 = 3

def first_five_units_digit_three_primes : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_first_five_units_digit_three_primes_eq_135 :
  (∑ n in first_five_units_digit_three_primes, n) = 135 :=
by
  -- The list first_five_units_digit_three_primes contains primes: 3, 13, 23, 43, 53
  sorry

end sum_first_five_units_digit_three_primes_eq_135_l331_331256


namespace nearest_integer_to_power_sum_l331_331864

theorem nearest_integer_to_power_sum :
  let x := (3 + Real.sqrt 5)
  Int.floor ((x ^ 4) + 1 / 2) = 752 :=
by
  sorry

end nearest_integer_to_power_sum_l331_331864


namespace original_mixture_percentage_l331_331870

variables (a w : ℝ)

-- Conditions given
def condition1 : Prop := a / (a + w + 2) = 0.3
def condition2 : Prop := (a + 2) / (a + w + 4) = 0.4

theorem original_mixture_percentage (h1 : condition1 a w) (h2 : condition2 a w) : (a / (a + w)) * 100 = 36 :=
by
sorry

end original_mixture_percentage_l331_331870


namespace cost_price_calculation_l331_331878

-- Define the given conditions
def total_price_incl_tax : ℝ := 616
def tax_rate : ℝ := 0.10
def profit_rate : ℝ := 0.14
def selling_price_before_tax := total_price_incl_tax / (1 + tax_rate)
def cost_price := selling_price_before_tax / (1 + profit_rate)

-- Prove the calculation of the cost price
theorem cost_price_calculation : cost_price = 491.23 :=
by {
  -- The definitions ensure the dependent conditions are correctly set up.
  sorry
}

end cost_price_calculation_l331_331878


namespace trace_ellipse_l331_331544

open Complex

theorem trace_ellipse (z : ℂ) (θ : ℝ) (h₁ : z = 3 * exp (θ * I))
  (h₂ : abs z = 3) : ∃ a b : ℝ, ∀ θ, z + 1/z = a * Real.cos θ + b * (I * Real.sin θ) :=
sorry

end trace_ellipse_l331_331544


namespace rectangle_perimeter_l331_331914

variable {s : ℝ} -- side length of the square

-- Given conditions as premises
axiom square_divided_into_rectangles (P : ℝ) : P = 4 * s → P = 160

-- Conclusion to be proven
theorem rectangle_perimeter (hs : s = 40) : 2 * ((s / 2) + s) = 120 :=
by
  rw [hs]
  norm_num
  sorry

end rectangle_perimeter_l331_331914


namespace part_A_part_B_part_C_l331_331001

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f(x * y) = y^2 * f(x) + x^2 * f(y)

theorem part_A : f(0) = 0 := sorry
theorem part_B : f(1) = 0 := sorry
theorem part_C : ∀ x : ℝ, f(x) = f(-x) := sorry

end part_A_part_B_part_C_l331_331001


namespace area_of_PQRS_l331_331448

noncomputable theory

variables {P Q R S T : Type} [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace S] [MetricSpace T]

def angle_90 (a b c : P) : Prop := ∠ a b c = π / 2

def PQRS_conditions : Prop :=
  angle_90 P Q R ∧
  angle_90 P R S ∧
  dist P R = 25 ∧
  dist R S = 40 ∧
  dist P T = 8

theorem area_of_PQRS
  (h : PQRS_conditions) : 
  area_of_quadrilateral P Q R S = 1000 :=
begin
  sorry
end

end area_of_PQRS_l331_331448


namespace exists_rectangular_parallelepiped_with_equal_surface_area_and_edge_sum_l331_331273

theorem exists_rectangular_parallelepiped_with_equal_surface_area_and_edge_sum :
  ∃ (a b c : ℤ), 2 * (a * b + b * c + c * a) = 4 * (a + b + c) :=
by
  -- Here we prove the existence of such integers a, b, c, which is stated in the theorem
  sorry

end exists_rectangular_parallelepiped_with_equal_surface_area_and_edge_sum_l331_331273


namespace not_prime_sum_l331_331729

theorem not_prime_sum (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_square : ∃ k : ℕ, a^2 - b * c = k^2) : ¬ Nat.Prime (2 * a + b + c) := 
sorry

end not_prime_sum_l331_331729


namespace domain_of_h_l331_331964

open Real

def h (x : ℝ) : ℝ := (x^4 - 16 * x + 3) / (|x - 4| + |x + 2| + x - 3)

theorem domain_of_h : ∀ x : ℝ, (|x - 4| + |x + 2| + x - 3) ≠ 0 :=
by
  intro x
  sorry

end domain_of_h_l331_331964


namespace sqrt_expression_eval_l331_331583

theorem sqrt_expression_eval :
    (Real.sqrt 8 - 2 * Real.sqrt (1 / 2) + (2 - Real.sqrt 3) * (2 + Real.sqrt 3)) = Real.sqrt 2 + 1 := 
by
  sorry

end sqrt_expression_eval_l331_331583


namespace sin_squared_alpha_eq_one_add_sin_squared_beta_l331_331623

variable {α θ β : ℝ}

theorem sin_squared_alpha_eq_one_add_sin_squared_beta
  (h1 : Real.sin α = Real.sin θ + Real.cos θ)
  (h2 : Real.sin β ^ 2 = 2 * Real.sin θ * Real.cos θ) :
  Real.sin α ^ 2 = 1 + Real.sin β ^ 2 := 
sorry

end sin_squared_alpha_eq_one_add_sin_squared_beta_l331_331623


namespace combined_work_time_l331_331450

-- Define the efficiencies
noncomputable def efficiency_K := 1 / 15  -- Krish's efficiency
noncomputable def efficiency_R := efficiency_K / 2  -- Ram's efficiency, verified from the given conditions
noncomputable def efficiency_T := (2 / 3) * efficiency_K  -- Tony's efficiency

-- Prove the combined work rate and time to completion
theorem combined_work_time :
  let combined_efficiency := efficiency_R + efficiency_K + efficiency_T in 
  let total_days := 1 / combined_efficiency in
  total_days = 90 / 13 :=
by
  sorry

end combined_work_time_l331_331450


namespace range_of_m_l331_331649

variable (P Q : Point)
variable (P_x P_y Q_x Q_y : ℝ) (m : ℝ)
variable (l : Line)

def is_slope_eq_PQ (P_x P_y Q_x Q_y : ℝ) :=
  (Q_y - P_y) / (Q_x - P_x) = 1/3

def is_slope_eq_l (m : ℝ) :=
  -1/m = 1/3

def intersects_extension_of_PQ (P_x P_y Q_x Q_y : ℝ) (m : ℝ) : Prop :=
  ∃ l : Line, (x + m * y + m = 0) ∧ 1/3 < -1/m ∧ -1/m < 3/2

theorem range_of_m (P : Point) (Q : Point) 
  (P_x P_y Q_x Q_y : ℝ) (m : ℝ)
  (h1 : P_x = -1) (h2 : P_y = 1) (h3 : Q_x = 2) (h4 : Q_y = 2)
  (h5 : is_slope_eq_PQ P_x P_y Q_x Q_y)
  (h6 : is_slope_eq_l m)
  (h7 : intersects_extension_of_PQ P_x P_y Q_x Q_y m) :
  m ∈ (-3, -2/3) := 
sorry

end range_of_m_l331_331649


namespace Line_NC_Passes_Through_Midpoint_AX_l331_331422

-- Define the problem conditions.
variables {A B C X Y N: Type} [Plane ℝ]

-- Let triangle ABC be an acute-angled triangle with |AB| < |AC|.
variables [Triangle ABC] [AcuteAngle ABC]  
variable [LessThan (|AB|) (|AC|)]

-- Let points X and Y lie on the minor arc BC of the circumcircle of triangle ABC.
variables [OnMinorArc X Y (Circumcircle ABC)]

-- Distances are such that |BX| = |XY| = |YC|.
variables [Eq (|BX|) (|XY|)] [Eq (|XY|) (|YC|)]

-- There exists a point N on the segment AY such that |AN| = |AB| = |NC|.
variables [OnSegment N (A, Y)] [Eq (|AN|) (|AB|)] [Eq (|AN|) (|NC|)]

-- Prove that NC passes through the midpoint of the segment AX.
theorem Line_NC_Passes_Through_Midpoint_AX :
  passes_through_midpoint (line N C) (midpoint (A, X)) :=
sorry

end Line_NC_Passes_Through_Midpoint_AX_l331_331422


namespace republican_vote_percent_l331_331072

theorem republican_vote_percent (V : ℕ) :
  let democrat_percent := 0.60
  let republican_percent := 0.40
  let democrat_vote_A_percent := 0.70
  let total_vote_A_percent := 0.50
  let democrat_votes := democrat_vote_A_percent * democrat_percent * V
  let total_votes := total_vote_A_percent * V
  ∃ (R : ℝ), (republican_percent * R * V + democrat_votes = total_votes) → (R = 0.20) :=
by
  sorry

end republican_vote_percent_l331_331072


namespace sum_of_cubes_eq_formula_l331_331135

theorem sum_of_cubes_eq_formula (n : ℕ) (hn : n > 0) :
    (∑ i in Finset.range (n + 1), i^3) = (n^2 * (n + 1)^2) / 4 := 
sorry

end sum_of_cubes_eq_formula_l331_331135


namespace greatest_three_digit_divisible_by_3_and_6_l331_331198

theorem greatest_three_digit_divisible_by_3_and_6 : ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 3 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ m % 3 = 0 ∧ m % 6 = 0 → m ≤ n :=
begin
  use 996,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  intros m hm1 hm2 hm3 hm4,
  sorry,
end

end greatest_three_digit_divisible_by_3_and_6_l331_331198


namespace third_vs_seventh_game_result_l331_331213

variable {Player : Type} [Fintype Player] [DecidableEq Player]

structure Tournament where
  players : Finset Player
  results : Player → Player → Option (Player × ℕ)
  (h_symmetric : ∀ {p1 p2}, results p1 p2 = results p2 p1)
  (h_diagonal : ∀ {p}, results p p = none)

-- Given conditions
def eight_players : Tournament :=
  { players := {1, 2, 3, 4, 5, 6, 7, 8},
    results := λ p1 p2 => if p1 = p2 then none else if nat.succ (p1 + p2) = p1 then some (p1, 1) else some (p2, 0),
    h_symmetric := by sorry,
    h_diagonal := by sorry }

variables (p2 p3 p7 : Player)

-- Assuming the second-place player’s score equals the combined score of the last four players, which is 6 points
constant second_place_score_combined : ∀ (t : Tournament) (p : Player), 
  t.players.card = 8 ∧ (∀ p1 p2, t.results p1 p2 = some (p1, 1) ∨ t.results p1 p2 = some (p2, 0) ∨ t.results p1 p2 = some (p1, 0.5)) →
  (∃ p2, (∑ p in t.players \ {p2}, t.results p2 p = 6)) 

theorem third_vs_seventh_game_result :
  ∀ (t : Tournament) (p3 p7 : Player), 
  t.players = {1, 2, 3, 4, 5, 6, 7, 8} →
  (∀ p1 p2, t.results p1 p2 = some (p1, 1) ∨ t.results p1 p2 = some (p2, 0) ∨ t.results p1 p2 = some (p1, 0.5)) →
  p2 ≠ p3 ∧ p2 ≠ p7 ∧ p3 ≠ p7 ∧ 
  second_place_score_combined t p2 →
  (t.results p3 p7 = some (p3, 1)) :=
by sorry

end third_vs_seventh_game_result_l331_331213


namespace disk_diameter_l331_331924

theorem disk_diameter (AB : ℝ) (h : ℝ) (r : ℝ) : AB = 16 ∧ h = 2 ∧ ((r^2 = (r - 2)^2 + 8^2)) → (2 * r = 34) :=
by
  intros hab hh hpyth
  sorry

end disk_diameter_l331_331924


namespace range_of_a_l331_331031

-- Given conditions
variable (a : ℝ)

def A (a : ℝ) : set ℝ := {x : ℝ | a * x^2 - 3 * x + 2 = 0}

-- Statement of the proof problem
theorem range_of_a (h : ∀ x1 x2 ∈ A a, x1 = x2) : a ≥ 9 / 8 ∨ a = 0 := 
sorry

end range_of_a_l331_331031


namespace perimeter_of_inner_triangle_smaller_perimeter_of_inner_convex_quadrilateral_smaller_l331_331389

theorem perimeter_of_inner_triangle_smaller {T1 T2 : Triangle} (h : T1 ⊆ T2) :
  perimeter T1 < perimeter T2 := 
sorry

theorem perimeter_of_inner_convex_quadrilateral_smaller {Q1 Q2 : Quadrilateral} 
  (h : Q1 ⊆ Q2) (h_convex : is_convex Q1) :
  perimeter Q1 < perimeter Q2 := 
sorry

end perimeter_of_inner_triangle_smaller_perimeter_of_inner_convex_quadrilateral_smaller_l331_331389


namespace b_investment_l331_331919

theorem b_investment (a_investment : ℝ) (c_investment : ℝ) (total_profit : ℝ) (a_share_profit : ℝ) (b_investment : ℝ) : a_investment = 6300 → c_investment = 10500 → total_profit = 14200 → a_share_profit = 4260 → b_investment = 4220 :=
by
  intro h_a h_c h_total h_a_share
  have h1 : 6300 / (6300 + 4220 + 10500) = 4260 / 14200 := sorry
  have h2 : 6300 * 14200 = 4260 * (6300 + 4220 + 10500) := sorry
  have h3 : b_investment = 4220 := sorry
  exact h3

end b_investment_l331_331919


namespace translation_vector_unique_l331_331823

theorem translation_vector_unique :
  ∃ (m n : ℤ), (4 - 2 * m = 0) ∧ (m^2 - 4 * m + 7 + n = 0) ∧ (m = 2) ∧ (n = -3) :=
by
  use (2, -3)
  simp
  sorry

end translation_vector_unique_l331_331823


namespace minimum_vacation_cost_l331_331461

-- Definitions based on the conditions in the problem.
def Polina_age : ℕ := 5
def parents_age : ℕ := 30 -- arbitrary age above the threshold, assuming adults

def Globus_cost_old : ℕ := 25400
def Globus_discount : ℝ := 0.02

def AroundWorld_cost_young : ℕ := 11400
def AroundWorld_cost_old : ℕ := 23500
def AroundWorld_commission : ℝ := 0.01

def globus_total_cost (num_adults num_children : ℕ) : ℕ :=
  let initial_cost := (num_adults + num_children) * Globus_cost_old
  let discount := Globus_discount * initial_cost
  initial_cost - discount.to_nat

def around_world_total_cost (num_adults num_children : ℕ) : ℕ :=
  let initial_cost := (num_adults * AroundWorld_cost_old) + (num_children * AroundWorld_cost_young)
  let commission := AroundWorld_commission * initial_cost
  initial_cost + commission.to_nat

-- Total costs calculated for the Dorokhov family with specific parameters.
def globus_final_cost : ℕ := globus_total_cost 2 1  -- 2 adults, 1 child
def around_world_final_cost : ℕ := around_world_total_cost 2 1  -- 2 adults, 1 child

theorem minimum_vacation_cost : around_world_final_cost = 58984 := by
  sorry

end minimum_vacation_cost_l331_331461
