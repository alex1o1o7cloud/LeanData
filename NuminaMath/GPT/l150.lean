import Mathlib

namespace tangent_line_parabola_l150_150338

theorem tangent_line_parabola (c : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + c ∧ y^2 = 12 * x → discriminant (y^2 - 4*y + 4*c) = 0) → c = 1 :=
by 
  -- We state the problem
  intros h
  -- Add sorry to indicate the proof is missing
  sorry

end tangent_line_parabola_l150_150338


namespace cloud_height_l150_150692

/--
Given:
- α : ℝ (elevation angle from the top of a tower)
- β : ℝ (depression angle seen in the lake)
- m : ℝ (height of the tower)
Prove:
- The height of the cloud hovering above the observer (h - m) is given by
 2 * m * cos β * sin α / sin (β - α)
-/
theorem cloud_height (α β m : ℝ) :
  (∃ h : ℝ, h - m = 2 * m * Real.cos β * Real.sin α / Real.sin (β - α)) :=
by
  sorry

end cloud_height_l150_150692


namespace nth_inequality_l150_150424

theorem nth_inequality (x : ℝ) (n : ℕ) (h_x_pos : 0 < x) : x + (n^n / x^n) ≥ n + 1 := 
sorry

end nth_inequality_l150_150424


namespace quadratic_common_root_l150_150417

theorem quadratic_common_root (b : ℤ) :
  (∃ x, 2 * x^2 + (3 * b - 1) * x - 3 = 0 ∧ 6 * x^2 - (2 * b - 3) * x - 1 = 0) ↔ b = 2 := 
sorry

end quadratic_common_root_l150_150417


namespace trajectory_of_M_is_ellipse_l150_150721

def circle_eq (x y : ℝ) : Prop := ((x + 3)^2 + y^2 = 100)

def point_B (x y : ℝ) : Prop := (x = 3 ∧ y = 0)

def point_on_circle (P : ℝ × ℝ) : Prop :=
  ∃ x y, P = (x, y) ∧ circle_eq x y

def perpendicular_bisector_intersects_CQ_at_M (B P M : ℝ × ℝ) : Prop :=
  (B.fst = 3 ∧ B.snd = 0) ∧
  point_on_circle P ∧
  ∃ r : ℝ, (P.fst + B.fst) / 2 = M.fst ∧ r = (M.snd - P.snd) / (M.fst - P.fst) ∧ 
  r = -(P.fst - B.fst) / (P.snd - B.snd)

theorem trajectory_of_M_is_ellipse (M : ℝ × ℝ) 
  (hC : ∀ x y, circle_eq x y)
  (hB : point_B 3 0)
  (hP : ∃ P : ℝ × ℝ, point_on_circle P)
  (hM : ∃ B P : ℝ × ℝ, perpendicular_bisector_intersects_CQ_at_M B P M) 
: (M.fst^2 / 25 + M.snd^2 / 16 = 1) := 
sorry

end trajectory_of_M_is_ellipse_l150_150721


namespace arithmetic_problem_l150_150327

theorem arithmetic_problem : 
  (888.88 - 555.55 + 111.11) * 2 = 888.88 := 
sorry

end arithmetic_problem_l150_150327


namespace Petya_wins_l150_150667

theorem Petya_wins
  (n : ℕ)
  (h1 : n > 0)
  (h2 : ∀ d, d > n^2 → ∃ m, (m < n ∧ Nat.Prime m ∧ d - m ≤ n^2) ∨ (m % n = 0 ∧ d - m ≤ n^2) ∨ (m = 1 ∧ d - 1 ≤ n^2)) :
  ∀ d, d > n^2 → ¬(Vasya_winning_strategy n d) → Petya_winning d := by
  sorry

end Petya_wins_l150_150667


namespace g_value_l150_150790

theorem g_value (g : ℝ → ℝ)
  (h0 : g 0 = 0)
  (h_mono : ∀ ⦃x y : ℝ⦄, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y)
  (h_symm : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x)
  (h_prop : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3) :
  g (2 / 5) = 1 / 2 :=
sorry

end g_value_l150_150790


namespace angles_with_same_terminal_side_as_15_degree_l150_150348

def condition1 (β : ℝ) (k : ℤ) : Prop := β = 15 + k * 90
def condition2 (β : ℝ) (k : ℤ) : Prop := β = 15 + k * 180
def condition3 (β : ℝ) (k : ℤ) : Prop := β = 15 + k * 360
def condition4 (β : ℝ) (k : ℤ) : Prop := β = 15 + 2 * k * 360

def has_same_terminal_side_as_15_degree (β : ℝ) : Prop :=
  ∃ k : ℤ, β = 15 + k * 360

theorem angles_with_same_terminal_side_as_15_degree (β : ℝ) :
  (∃ k : ℤ, condition1 β k)  ∨
  (∃ k : ℤ, condition2 β k)  ∨
  (∃ k : ℤ, condition3 β k)  ∨
  (∃ k : ℤ, condition4 β k) →
  has_same_terminal_side_as_15_degree β :=
by
  sorry

end angles_with_same_terminal_side_as_15_degree_l150_150348


namespace math_problem_l150_150691

theorem math_problem :
  (Real.pi - 3.14)^0 + Real.sqrt ((Real.sqrt 2 - 1)^2) = Real.sqrt 2 :=
by
  sorry

end math_problem_l150_150691


namespace base_conversion_sum_l150_150410

noncomputable def A : ℕ := 10

noncomputable def base11_to_nat (x y z : ℕ) : ℕ :=
  x * 11^2 + y * 11^1 + z * 11^0

noncomputable def base12_to_nat (x y z : ℕ) : ℕ :=
  x * 12^2 + y * 12^1 + z * 12^0

theorem base_conversion_sum :
  base11_to_nat 3 7 9 + base12_to_nat 3 9 A = 999 :=
by
  sorry

end base_conversion_sum_l150_150410


namespace delta_value_l150_150437

theorem delta_value : ∃ Δ : ℤ, 5 * (-3) = Δ - 3 ∧ Δ = -12 :=
by {
  use -12,
  split,
  { refl },
  { refl }
}

end delta_value_l150_150437


namespace inequality_one_inequality_two_l150_150161

theorem inequality_one (a b : ℝ) : 
    a^2 + b^2 ≥ (a + b)^2 / 2 := 
by
    sorry

theorem inequality_two (a b : ℝ) : 
    a^2 + b^2 ≥ 2 * (a - b - 1) := 
by
    sorry

end inequality_one_inequality_two_l150_150161


namespace setB_is_PythagoreanTriple_setA_is_not_PythagoreanTriple_setC_is_not_PythagoreanTriple_setD_is_not_PythagoreanTriple_l150_150951

-- Define what it means to be a Pythagorean triple
def isPythagoreanTriple (a b c : Int) : Prop :=
  a^2 + b^2 = c^2

-- Define the given sets
def setA : (Int × Int × Int) := (12, 15, 18)
def setB : (Int × Int × Int) := (3, 4, 5)
def setC : (Rat × Rat × Rat) := (1.5, 2, 2.5)
def setD : (Int × Int × Int) := (6, 9, 15)

-- Proven statements about each set
theorem setB_is_PythagoreanTriple : isPythagoreanTriple 3 4 5 :=
  by
  sorry

theorem setA_is_not_PythagoreanTriple : ¬ isPythagoreanTriple 12 15 18 :=
  by
  sorry

-- Pythagorean triples must consist of positive integers
theorem setC_is_not_PythagoreanTriple : ¬ ∃ (a b c : Int), a^2 + b^2 = c^2 ∧ 
  a = 3/2 ∧ b = 2 ∧ c = 5/2 :=
  by
  sorry

theorem setD_is_not_PythagoreanTriple : ¬ isPythagoreanTriple 6 9 15 :=
  by
  sorry

end setB_is_PythagoreanTriple_setA_is_not_PythagoreanTriple_setC_is_not_PythagoreanTriple_setD_is_not_PythagoreanTriple_l150_150951


namespace retirement_amount_l150_150460

-- Define the principal amount P
def P : ℝ := 750000

-- Define the annual interest rate r
def r : ℝ := 0.08

-- Define the time period in years t
def t : ℝ := 12

-- Define the accumulated amount A
def A : ℝ := P * (1 + r * t)

-- Prove that the accumulated amount A equals 1470000
theorem retirement_amount : A = 1470000 := by
  -- The proof will involve calculating the compound interest
  sorry

end retirement_amount_l150_150460


namespace total_amount_from_grandparents_l150_150614

theorem total_amount_from_grandparents (amount_from_grandpa : ℕ) (multiplier : ℕ) (amount_from_grandma : ℕ) (total_amount : ℕ) 
  (h1 : amount_from_grandpa = 30) 
  (h2 : multiplier = 3) 
  (h3 : amount_from_grandma = multiplier * amount_from_grandpa) 
  (h4 : total_amount = amount_from_grandpa + amount_from_grandma) :
  total_amount = 120 := 
by 
  sorry

end total_amount_from_grandparents_l150_150614


namespace find_numbers_l150_150503

theorem find_numbers (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sqrt (a * b) = Real.sqrt 5) ∧ 
  (2 * a * b / (a + b) = 5 / 3) → 
  (a = 5 ∧ b = 1) ∨ (a = 1 ∧ b = 5) := 
sorry

end find_numbers_l150_150503


namespace pizza_slices_needed_l150_150380

theorem pizza_slices_needed (couple_slices : ℕ) (children : ℕ) (children_slices : ℕ) (pizza_slices : ℕ)
    (hc : couple_slices = 3)
    (hcouple : children = 6)
    (hch : children_slices = 1)
    (hpizza : pizza_slices = 4) : 
    (2 * couple_slices + children * children_slices) / pizza_slices = 3 := 
by
    sorry

end pizza_slices_needed_l150_150380


namespace lines_positional_relationship_l150_150494

-- Defining basic geometric entities and their properties
structure Line :=
  (a b : ℝ)
  (point_on_line : ∃ x, a * x + b = 0)

-- Defining skew lines (two lines that do not intersect and are not parallel)
def skew_lines (l1 l2 : Line) : Prop :=
  ¬(∀ x, l1.a * x + l1.b = l2.a * x + l2.b) ∧ ¬(l1.a = l2.a)

-- Defining intersecting lines
def intersect (l1 l2 : Line) : Prop :=
  ∃ x, l1.a * x + l1.b = l2.a * x + l2.b

-- Main theorem to prove
theorem lines_positional_relationship (l1 l2 k m : Line) 
  (hl1: intersect l1 k) (hl2: intersect l2 k) (hk: skew_lines l1 m) (hm: skew_lines l2 m) :
  (intersect l1 l2) ∨ (skew_lines l1 l2) :=
sorry

end lines_positional_relationship_l150_150494


namespace prove_q_ge_bd_and_p_eq_ac_l150_150053

-- Definitions for the problem
variables (a b c d p q : ℕ)

-- Conditions given in the problem
axiom h1: a * d - b * c = 1
axiom h2: (a : ℚ) / b > (p : ℚ) / q
axiom h3: (p : ℚ) / q > (c : ℚ) / d

-- The theorem to be proved
theorem prove_q_ge_bd_and_p_eq_ac (a b c d p q : ℕ) (h1 : a * d - b * c = 1) 
  (h2 : (a : ℚ) / b > (p : ℚ) / q) (h3 : (p : ℚ) / q > (c : ℚ) / d) :
  q ≥ b + d ∧ (q = b + d → p = a + c) :=
by
  sorry

end prove_q_ge_bd_and_p_eq_ac_l150_150053


namespace translate_line_upwards_by_3_translate_line_right_by_3_l150_150502

theorem translate_line_upwards_by_3 (x : ℝ) :
  let y := 2 * x - 4
  let y' := y + 3
  y' = 2 * x - 1 := 
by
  let y := 2 * x - 4
  let y' := y + 3
  sorry

theorem translate_line_right_by_3 (x : ℝ) :
  let y := 2 * x - 4
  let y_up := y + 3
  let y_right := 2 * (x - 3) - 4
  y_right = 2 * x - 10 :=
by
  let y := 2 * x - 4
  let y_up := y + 3
  let y_right := 2 * (x - 3) - 4
  sorry

end translate_line_upwards_by_3_translate_line_right_by_3_l150_150502


namespace white_mice_count_l150_150030

variable (T W B : ℕ) -- Declare variables T (total), W (white), B (brown)

def W_condition := W = (2 / 3) * T  -- White mice condition
def B_condition := B = 7           -- Brown mice condition
def T_condition := T = W + B       -- Total mice condition

theorem white_mice_count : W = 14 :=
by
  sorry  -- Proof to be filled in

end white_mice_count_l150_150030


namespace person_Y_share_l150_150548

theorem person_Y_share (total_amount : ℝ) (r1 r2 r3 r4 r5 : ℝ) (ratio_Y : ℝ) 
  (h1 : total_amount = 1390) 
  (h2 : r1 = 13) 
  (h3 : r2 = 17)
  (h4 : r3 = 23) 
  (h5 : r4 = 29) 
  (h6 : r5 = 37) 
  (h7 : ratio_Y = 29): 
  (total_amount / (r1 + r2 + r3 + r4 + r5) * ratio_Y) = 338.72 :=
by
  sorry

end person_Y_share_l150_150548


namespace GouguPrinciple_l150_150172

-- Definitions according to conditions
def volumes_not_equal (A B : Type) : Prop := sorry -- p: volumes of A and B are not equal
def cross_sections_not_equal (A B : Type) : Prop := sorry -- q: cross-sectional areas of A and B are not always equal

-- The theorem to be proven
theorem GouguPrinciple (A B : Type) (h1 : volumes_not_equal A B) : cross_sections_not_equal A B :=
sorry

end GouguPrinciple_l150_150172


namespace sandy_age_correct_l150_150524

def is_age_ratio (S M : ℕ) : Prop := S * 9 = M * 7
def is_age_difference (S M : ℕ) : Prop := M = S + 12

theorem sandy_age_correct (S M : ℕ) (h1 : is_age_ratio S M) (h2 : is_age_difference S M) : S = 42 := by
  sorry

end sandy_age_correct_l150_150524


namespace cube_volume_surface_area_l150_150511

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ s : ℝ, s^3 = 8 * x ∧ 6 * s^2 = 2 * x) → x = 0 :=
by
  sorry

end cube_volume_surface_area_l150_150511


namespace S_6_equals_12_l150_150457

noncomputable def S (n : ℕ) : ℝ := sorry -- Definition for the sum of the first n terms

axiom geometric_sequence_with_positive_terms (n : ℕ) : S n > 0

axiom S_3 : S 3 = 3

axiom S_9 : S 9 = 39

theorem S_6_equals_12 : S 6 = 12 := by
  sorry

end S_6_equals_12_l150_150457


namespace one_div_m_plus_one_div_n_l150_150791

theorem one_div_m_plus_one_div_n
  {m n : ℕ} 
  (h1 : Nat.gcd m n = 5) 
  (h2 : Nat.lcm m n = 210)
  (h3 : m + n = 75) :
  (1 : ℚ) / m + (1 : ℚ) / n = 1 / 14 :=
by
  sorry

end one_div_m_plus_one_div_n_l150_150791


namespace smallest_value_3a_plus_1_l150_150023

theorem smallest_value_3a_plus_1 (a : ℚ) (h : 8 * a^2 + 6 * a + 5 = 2) : 3 * a + 1 = -5 / 4 :=
sorry

end smallest_value_3a_plus_1_l150_150023


namespace swimming_speed_in_still_water_l150_150090

theorem swimming_speed_in_still_water 
  (speed_of_water : ℝ) (distance : ℝ) (time : ℝ) (v : ℝ) 
  (h_water_speed : speed_of_water = 2) 
  (h_time_distance : time = 4 ∧ distance = 8) :
  v = 4 :=
by
  sorry

end swimming_speed_in_still_water_l150_150090


namespace solve_inequality_system_l150_150775

theorem solve_inequality_system (x : ℝ) (h1 : 2 * x + 1 < 5) (h2 : 2 - x ≤ 1) : 1 ≤ x ∧ x < 2 :=
by
  sorry

end solve_inequality_system_l150_150775


namespace hyperbola_center_l150_150243

theorem hyperbola_center (x y : ℝ) :
  ∃ h k : ℝ, (∃ a b : ℝ, a = 9/4 ∧ b = 7/2) ∧ (h, k) = (-2, 3) ∧ 
  (4*x + 8)^2 / 81 - (2*y - 6)^2 / 49 = 1 :=
by
  sorry

end hyperbola_center_l150_150243


namespace digit_difference_l150_150489

theorem digit_difference (x y : ℕ) (h : 10 * x + y - (10 * y + x) = 45) : x - y = 5 :=
sorry

end digit_difference_l150_150489


namespace imaginary_part_of_z_is_2_l150_150727

noncomputable def z : ℂ := (3 * Complex.I + 1) / (1 - Complex.I)

theorem imaginary_part_of_z_is_2 : z.im = 2 := 
by 
  -- proof goes here
  sorry

end imaginary_part_of_z_is_2_l150_150727


namespace delta_value_l150_150438

theorem delta_value : ∃ Δ : ℤ, 5 * (-3) = Δ - 3 ∧ Δ = -12 :=
by {
  use -12,
  split,
  { refl },
  { refl }
}

end delta_value_l150_150438


namespace Molly_swam_on_Saturday_l150_150321

variable (total_meters : ℕ) (sunday_meters : ℕ)

def saturday_meters := total_meters - sunday_meters

theorem Molly_swam_on_Saturday : 
  total_meters = 73 ∧ sunday_meters = 28 → saturday_meters total_meters sunday_meters = 45 := by
sorry

end Molly_swam_on_Saturday_l150_150321


namespace minimum_value_of_exp_l150_150279

noncomputable theory

open Locale.Real

def vectors_orthogonal (x y : ℝ) : Prop :=
  let a := (x - 1, 2)
  let b := (4, y)
  a.1 * b.1 + a.2 * b.2 = 0

theorem minimum_value_of_exp (x y : ℝ) :
  vectors_orthogonal x y →
  9^x + 3^y ≥ 6 :=
by
  sorry

end minimum_value_of_exp_l150_150279


namespace berry_difference_l150_150985

/-- Define the initial total number of berries on the bush -/
def total_berries : ℕ := 900

/-- Sergey collects 1 out of every 2 berries he picks -/
def sergey_collection_ratio : ℕ := 2

/-- Dima collects 2 out of every 3 berries he picks -/
def dima_collection_ratio : ℕ := 3

/-- Sergey picks berries twice as fast as Dima -/
def sergey_speed_multiplier : ℕ := 2

/-- Prove that the difference between berries collected in Sergey's and Dima's baskets is 100 -/
theorem berry_difference : 
  let total_picked_sergey := (sergey_speed_multiplier * total_berries) / (sergey_speed_multiplier + 1),
      total_picked_dima := total_berries / (sergey_speed_multiplier + 1),
      sergey_basket := total_picked_sergey / sergey_collection_ratio,
      dima_basket := (2 * total_picked_dima) / dima_collection_ratio
  in sergey_basket - dima_basket = 100 :=
by
  sorry -- proof to be completed

end berry_difference_l150_150985


namespace shadow_stretch_rate_is_5_feet_per_hour_l150_150116

-- Given conditions
def shadow_length_in_inches (hours_past_noon : ℕ) : ℕ := 360
def hours_past_noon : ℕ := 6

-- Convert inches to feet
def inches_to_feet (inches : ℕ) : ℕ := inches / 12

-- Calculate rate of increase of shadow length per hour
def rate_of_shadow_stretch_per_hour : ℕ := inches_to_feet (shadow_length_in_inches hours_past_noon) / hours_past_noon

theorem shadow_stretch_rate_is_5_feet_per_hour :
  rate_of_shadow_stretch_per_hour = 5 := by
  sorry

end shadow_stretch_rate_is_5_feet_per_hour_l150_150116


namespace m_leq_neg3_l150_150451

theorem m_leq_neg3 (m : ℝ) (h : ∀ x ∈ Set.Icc (0 : ℝ) 1, x^2 - 4 * x ≥ m) : m ≤ -3 := 
  sorry

end m_leq_neg3_l150_150451


namespace largest_five_digit_integer_l150_150363

/-- The product of the digits of the integer 98752 is (7 * 6 * 5 * 4 * 3 * 2 * 1), and
    98752 is the largest five-digit integer with this property. -/
theorem largest_five_digit_integer :
  (∃ (n : ℕ), n = 98752 ∧ (∃ (d1 d2 d3 d4 d5 : ℕ),
    n = d1 * 10^4 + d2 * 10^3 + d3 * 10^2 + d4 * 10 + d5 ∧
    (d1 * d2 * d3 * d4 * d5 = 7 * 6 * 5 * 4 * 3 * 2 * 1) ∧
    (∀ (m : ℕ), m ≠ 98752 → m < 100000 ∧ (∃ (e1 e2 e3 e4 e5 : ℕ),
    m = e1 * 10^4 + e2 * 10^3 + e3 * 10^2 + e4 * 10 + e5 →
    (e1 * e2 * e3 * e4 * e5 = 7 * 6 * 5 * 4 * 3 * 2 * 1) → m < 98752)))) :=
  sorry

end largest_five_digit_integer_l150_150363


namespace existence_of_five_regular_polyhedra_l150_150091

def regular_polyhedron (n m : ℕ) : Prop :=
  n ≥ 3 ∧ m ≥ 3 ∧ (2 / m + 2 / n > 1)

theorem existence_of_five_regular_polyhedra :
  ∃ (n m : ℕ), regular_polyhedron n m → 
    (n = 3 ∧ m = 3 ∨ 
     n = 4 ∧ m = 3 ∨ 
     n = 3 ∧ m = 4 ∨ 
     n = 5 ∧ m = 3 ∨ 
     n = 3 ∧ m = 5) :=
by
  sorry

end existence_of_five_regular_polyhedra_l150_150091


namespace problem_proof_l150_150427

theorem problem_proof (x y z : ℝ) 
  (h1 : 1/x + 2/y + 3/z = 0) 
  (h2 : 1/x - 6/y - 5/z = 0) : 
  (x / y + y / z + z / x) = -1 := 
by
  sorry

end problem_proof_l150_150427


namespace min_m_plus_n_l150_150260

theorem min_m_plus_n (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m * n - 2 * m - 3 * n = 20) : 
  m + n = 20 :=
sorry

end min_m_plus_n_l150_150260


namespace range_of_a_iff_l150_150530

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ x : ℝ, |x| + |x - 1| ≤ a → a ≥ 1

theorem range_of_a_iff (a : ℝ) :
  (∃ x : ℝ, |x| + |x - 1| ≤ a) ↔ (a ≥ 1) :=
by sorry

end range_of_a_iff_l150_150530


namespace hindi_speaking_students_l150_150456

theorem hindi_speaking_students 
    (G M T A : ℕ)
    (Total : ℕ)
    (hG : G = 6)
    (hM : M = 6)
    (hT : T = 2)
    (hA : A = 1)
    (hTotal : Total = 22)
    : ∃ H, Total = G + H + M - (T - A) + A ∧ H = 10 := by
  sorry

end hindi_speaking_students_l150_150456


namespace solution_exists_l150_150404

noncomputable def verify_triples (a b c : ℝ) : Prop :=
  a ≠ b ∧ a ≠ 0 ∧ b ≠ 0 ∧ b = -2 * a ∧ c = 4 * a

theorem solution_exists (a b c : ℝ) : verify_triples a b c :=
by
  sorry

end solution_exists_l150_150404


namespace tangent_line_at_point_l150_150704

noncomputable def curve (x : ℝ) : ℝ := Real.exp x + x

theorem tangent_line_at_point :
  (∃ k b : ℝ, (∀ x : ℝ, curve x = k * x + b) ∧ k = 2 ∧ b = 1) :=
by
  sorry

end tangent_line_at_point_l150_150704


namespace abs_diff_max_min_l150_150946

noncomputable def min_and_max_abs_diff (x : ℝ) : ℝ :=
|x - 2| + |x - 3| - |x - 1|

theorem abs_diff_max_min (x : ℝ) (h : 2 ≤ x ∧ x ≤ 3) :
  ∃ (M m : ℝ), M = 0 ∧ m = -1 ∧
    M = max (min_and_max_abs_diff 2) (min_and_max_abs_diff 3) ∧ 
    m = min (min_and_max_abs_diff 2) (min_and_max_abs_diff 3) :=
by
  use [0, -1]
  split
  case inl => sorry
  case inr => sorry

end abs_diff_max_min_l150_150946


namespace sticker_arrangement_l150_150315

theorem sticker_arrangement : 
  ∀ (n : ℕ), n = 35 → 
  (∀ k : ℕ, k = 8 → 
    ∃ m : ℕ, m = 5 ∧ (n + m) % k = 0) := 
by sorry

end sticker_arrangement_l150_150315


namespace range_of_positive_integers_in_list_H_l150_150902

noncomputable def list_H_lower_bound : Int := -15
noncomputable def list_H_length : Nat := 30

theorem range_of_positive_integers_in_list_H :
  ∃(r : Nat), list_H_lower_bound + list_H_length - 1 = 14 ∧ r = 14 - 1 := 
by
  let upper_bound := list_H_lower_bound + Int.ofNat list_H_length - 1
  use (upper_bound - 1).toNat
  sorry

end range_of_positive_integers_in_list_H_l150_150902


namespace tabby_average_speed_l150_150331

noncomputable def overall_average_speed : ℝ := 
  let swimming_speed : ℝ := 1
  let cycling_speed : ℝ := 18
  let running_speed : ℝ := 6
  let time_swimming : ℝ := 2
  let time_cycling : ℝ := 3
  let time_running : ℝ := 2
  let distance_swimming := swimming_speed * time_swimming
  let distance_cycling := cycling_speed * time_cycling
  let distance_running := running_speed * time_running
  let total_distance := distance_swimming + distance_cycling + distance_running
  let total_time := time_swimming + time_cycling + time_running
  total_distance / total_time

theorem tabby_average_speed : overall_average_speed = 9.71 := sorry

end tabby_average_speed_l150_150331


namespace number_of_integers_in_sequence_l150_150920

theorem number_of_integers_in_sequence 
  (a_0 : ℕ) 
  (h_0 : a_0 = 8820) 
  (seq : ℕ → ℕ) 
  (h_seq : ∀ n : ℕ, seq (n + 1) = seq n / 3) :
  ∃ n : ℕ, seq n = 980 ∧ n + 1 = 3 :=
by
  sorry

end number_of_integers_in_sequence_l150_150920


namespace binary_operation_l150_150411

-- Definitions of the binary numbers.
def a : ℕ := 0b10110      -- 10110_2 in base 10
def b : ℕ := 0b10100      -- 10100_2 in base 10
def c : ℕ := 0b10         -- 10_2 in base 10
def result : ℕ := 0b11011100 -- 11011100_2 in base 10

-- The theorem to be proven
theorem binary_operation : (a * b) / c = result := by
  -- Placeholder for the proof
  sorry

end binary_operation_l150_150411


namespace roots_twice_other_p_values_l150_150250

theorem roots_twice_other_p_values (p : ℝ) :
  (∃ (a : ℝ), (a^2 = 9) ∧ (x^2 + p*x + 18 = 0) ∧
  ((x - a)*(x - 2*a) = (0:ℝ))) ↔ (p = 9 ∨ p = -9) :=
sorry

end roots_twice_other_p_values_l150_150250


namespace find_value_of_r_l150_150085

theorem find_value_of_r (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a * r / (1 - r^2) = 8) : r = 2 / 3 :=
by
  sorry

end find_value_of_r_l150_150085


namespace probability_of_red_second_given_red_first_l150_150532

-- Define the conditions as per the problem.
def total_balls := 5
def red_balls := 3
def yellow_balls := 2
def first_draw_red : ℚ := (red_balls : ℚ) / (total_balls : ℚ)
def both_draws_red : ℚ := (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1))

-- Define the probability of drawing a red ball in the second draw given the first was red.
def conditional_probability_red_second_given_first : ℚ :=
  both_draws_red / first_draw_red

-- The main statement to be proved.
theorem probability_of_red_second_given_red_first :
  conditional_probability_red_second_given_first = 1 / 2 :=
by
  sorry

end probability_of_red_second_given_red_first_l150_150532


namespace solve_system_l150_150776

theorem solve_system :
  ∃ (x y : ℤ), 2 * x + y = 4 ∧ x + 2 * y = -1 ∧ x = 3 ∧ y = -2 :=
by
  use [3, -2]
  simp
  ring
  sorry

end solve_system_l150_150776


namespace three_times_two_to_the_n_minus_one_gt_n_squared_plus_three_l150_150170

theorem three_times_two_to_the_n_minus_one_gt_n_squared_plus_three (n : ℕ) (h : n ≥ 4) : 3 * 2^(n-1) > n^2 + 3 := by
  sorry

end three_times_two_to_the_n_minus_one_gt_n_squared_plus_three_l150_150170


namespace equal_sides_length_of_isosceles_right_triangle_l150_150886

noncomputable def isosceles_right_triangle (a c : ℝ) : Prop :=
  c^2 = 2 * a^2 ∧ a^2 + a^2 + c^2 = 725

theorem equal_sides_length_of_isosceles_right_triangle (a c : ℝ) 
  (h : isosceles_right_triangle a c) : 
  a = 13.5 :=
by
  sorry

end equal_sides_length_of_isosceles_right_triangle_l150_150886


namespace george_paint_l150_150304

theorem george_paint colors : fintype colors →  fin (Card colors) = 9 → (Card { x : (fin 9) // x ∈ comb 3 }) = 84 := by sorry

end george_paint_l150_150304


namespace share_of_C_l150_150095

/-- Given the conditions:
  - Total investment is Rs. 120,000.
  - A's investment is Rs. 6,000 more than B's.
  - B's investment is Rs. 8,000 more than C's.
  - Profit distribution ratio among A, B, and C is 4:3:2.
  - Total profit is Rs. 50,000.
Prove that C's share of the profit is Rs. 11,111.11. -/
theorem share_of_C (total_investment : ℝ)
  (A_more_than_B : ℝ)
  (B_more_than_C : ℝ)
  (profit_distribution : ℝ)
  (total_profit : ℝ) :
  total_investment = 120000 →
  A_more_than_B = 6000 →
  B_more_than_C = 8000 →
  profit_distribution = 4 / 9 →
  total_profit = 50000 →
  ∃ (C_share : ℝ), C_share = 11111.11 :=
by
  sorry

end share_of_C_l150_150095


namespace g_of_square_sub_one_l150_150103

variable {R : Type*} [LinearOrderedField R]

def g (x : R) : R := 3

theorem g_of_square_sub_one (x : R) : g ((x - 1)^2) = 3 := 
by sorry

end g_of_square_sub_one_l150_150103


namespace simple_interest_rate_l150_150294

theorem simple_interest_rate (P : ℝ) (T : ℝ) (R : ℝ) (SI : ℝ) (hT : T = 8) 
  (hSI : SI = P / 5) : SI = (P * R * T) / 100 → R = 2.5 :=
by
  intro
  sorry

end simple_interest_rate_l150_150294


namespace delta_value_l150_150436

theorem delta_value : ∃ Δ : ℤ, 5 * (-3) = Δ - 3 ∧ Δ = -12 :=
by {
  use -12,
  split,
  { refl },
  { refl }
}

end delta_value_l150_150436


namespace total_flowers_l150_150189

def tulips : ℕ := 3
def carnations : ℕ := 4

theorem total_flowers : tulips + carnations = 7 := by
  sorry

end total_flowers_l150_150189


namespace votes_for_veggies_l150_150295

theorem votes_for_veggies (T M V : ℕ) (hT : T = 672) (hM : M = 335) (hV : V = T - M) : V = 337 := 
by
  rw [hT, hM] at hV
  simp at hV
  exact hV

end votes_for_veggies_l150_150295


namespace region_to_the_upper_left_of_line_l150_150177

variable (x y : ℝ)

def line_eqn := 3 * x - 2 * y - 6 = 0

def region := 3 * x - 2 * y - 6 < 0

theorem region_to_the_upper_left_of_line :
  ∃ rect_upper_left, (rect_upper_left = region) := 
sorry

end region_to_the_upper_left_of_line_l150_150177


namespace always_non_monotonic_l150_150277

noncomputable def f (a t x : ℝ) : ℝ :=
if x ≤ t then (2*a - 1)*x + 3*a - 4 else x^3 - x

theorem always_non_monotonic (a : ℝ) (t : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → f a t x1 ≤ f a t x2 ∨ f a t x1 ≥ f a t x2) → a ≤ 1 / 2 :=
sorry

end always_non_monotonic_l150_150277


namespace eq_has_infinite_solutions_l150_150993

theorem eq_has_infinite_solutions (b : ℝ) (x : ℝ) :
  5 * (3 * x - b) = 3 * (5 * x + 15) → b = -9 := by
sorry

end eq_has_infinite_solutions_l150_150993


namespace number_of_real_roots_eq_3_eq_m_l150_150120

theorem number_of_real_roots_eq_3_eq_m {x m : ℝ} (h : ∀ x, x^2 - 2 * |x| + 2 = m) : m = 2 :=
sorry

end number_of_real_roots_eq_3_eq_m_l150_150120


namespace total_cost_is_716_mom_has_enough_money_l150_150093

/-- Definition of the price of the table lamp -/
def table_lamp_price : ℕ := 86

/-- Definition of the price of the electric fan -/
def electric_fan_price : ℕ := 185

/-- Definition of the price of the bicycle -/
def bicycle_price : ℕ := 445

/-- The total cost of buying all three items -/
def total_cost : ℕ := table_lamp_price + electric_fan_price + bicycle_price

/-- Mom's money -/
def mom_money : ℕ := 300

/-- Problem 1: Prove that the total cost equals 716 -/
theorem total_cost_is_716 : total_cost = 716 := 
by 
  sorry

/-- Problem 2: Prove that Mom has enough money to buy a table lamp and an electric fan -/
theorem mom_has_enough_money : table_lamp_price + electric_fan_price ≤ mom_money :=
by 
  sorry

end total_cost_is_716_mom_has_enough_money_l150_150093


namespace min_rain_fourth_day_l150_150802

def rain_overflow_problem : Prop :=
    let holding_capacity := 6 * 12 -- in inches
    let drainage_per_day := 3 -- in inches
    let rainfall_day1 := 10 -- in inches
    let rainfall_day2 := 2 * rainfall_day1 -- 20 inches
    let rainfall_day3 := 1.5 * rainfall_day2 -- 30 inches
    let total_rain_three_days := rainfall_day1 + rainfall_day2 + rainfall_day3 -- 60 inches
    let total_drainage_three_days := 3 * drainage_per_day -- 9 inches
    let remaining_capacity := holding_capacity - (total_rain_three_days - total_drainage_three_days) -- 21 inches
    (remaining_capacity = 21)

theorem min_rain_fourth_day : rain_overflow_problem := sorry

end min_rain_fourth_day_l150_150802


namespace root_in_interval_l150_150546

noncomputable def f (x : ℝ) := x^2 + 12 * x - 15

theorem root_in_interval :
  (f 1.1 = -0.59) → (f 1.2 = 0.84) →
  ∃ c, 1.1 < c ∧ c < 1.2 ∧ f c = 0 :=
by
  intros h1 h2
  let h3 := h1
  let h4 := h2
  sorry

end root_in_interval_l150_150546


namespace sum_of_three_numbers_eq_zero_l150_150646

theorem sum_of_three_numbers_eq_zero (a b c : ℝ) (h1 : a ≤ b ∧ b ≤ c) (h2 : (a + b + c) / 3 = a + 20) (h3 : (a + b + c) / 3 = c - 10) (h4 : b = 10) : 
  a + b + c = 0 := 
by 
  sorry

end sum_of_three_numbers_eq_zero_l150_150646


namespace retirement_amount_l150_150461

theorem retirement_amount
  (P : ℝ) (r : ℝ) (t : ℝ)
  (hP : P = 750000)
  (hr : r = 0.08)
  (ht : t = 12) :
  let A := P * (1 + r * t) in
  A = 1470000 :=
by {
  sorry
}

end retirement_amount_l150_150461


namespace parallelogram_area_increase_l150_150553

theorem parallelogram_area_increase (b h : ℕ) :
  let A1 := b * h
  let b' := 2 * b
  let h' := 2 * h
  let A2 := b' * h'
  (A2 - A1) * 100 / A1 = 300 :=
by
  let A1 := b * h
  let b' := 2 * b
  let h' := 2 * h
  let A2 := b' * h'
  sorry

end parallelogram_area_increase_l150_150553


namespace ratio_13_2_l150_150396

def initial_mahogany_trees : ℕ := 50
def initial_narra_trees : ℕ := 30
def total_trees_that_fell : ℕ := 5
def current_total_trees : ℕ := 88

def number_narra_trees_that_fell (N : ℕ) : Prop := N + (N + 1) = total_trees_that_fell
def total_trees_before_typhoon : ℕ := initial_mahogany_trees + initial_narra_trees

def ratio_of_planted_trees_to_narra_fallen (planted : ℕ) (N : ℕ) : Prop := 
  88 - (total_trees_before_typhoon - total_trees_that_fell) = planted ∧ 
  planted / N = 13 / 2

theorem ratio_13_2 : ∃ (planted N : ℕ), 
  number_narra_trees_that_fell N ∧ 
  ratio_of_planted_trees_to_narra_fallen planted N :=
sorry

end ratio_13_2_l150_150396


namespace equation_of_circle_center_0_4_passing_through_3_0_l150_150785

noncomputable def circle_radius (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem equation_of_circle_center_0_4_passing_through_3_0 :
  ∃ (r : ℝ), (r = circle_radius 0 4 3 0) ∧ (r = 5) ∧ ((x y : ℝ) → ((x - 0) ^ 2 + (y - 4) ^ 2 = r ^ 2) ↔ (x ^ 2 + (y - 4) ^ 2 = 25)) :=
by
  sorry

end equation_of_circle_center_0_4_passing_through_3_0_l150_150785


namespace area_of_abs_sum_eq_six_l150_150562

theorem area_of_abs_sum_eq_six : 
  (∃ (R : set (ℝ × ℝ)), (∀ (x y : ℝ), ((|x + y| + |x - y|) ≤ 6 → (x, y) ∈ R)) ∧ area R = 36) :=
sorry

end area_of_abs_sum_eq_six_l150_150562


namespace arrangement_with_A_in_middle_arrangement_with_A_at_end_B_not_at_end_arrangement_with_A_B_adjacent_not_adjacent_to_C_l150_150829

-- Proof Problem 1
theorem arrangement_with_A_in_middle (products : Finset ℕ) (A : ℕ) (hA : A ∈ products) (arrangements : Finset (Fin 5 → ℕ)) :
  5 ∈ products ∧ (∀ a ∈ arrangements, a (Fin.mk 2 sorry) = A) →
  arrangements.card = 24 :=
by sorry

-- Proof Problem 2
theorem arrangement_with_A_at_end_B_not_at_end (products : Finset ℕ) (A B : ℕ) (hA : A ∈ products) (hB : B ∈ products) (arrangements : Finset (Fin 5 → ℕ)) :
  (5 ∈ products ∧ (∀ a ∈ arrangements, (a 0 = A ∨ a 4 = A) ∧ (a 1 ≠ B ∧ a 2 ≠ B ∧ a 3 ≠ B))) →
  arrangements.card = 36 :=
by sorry

-- Proof Problem 3
theorem arrangement_with_A_B_adjacent_not_adjacent_to_C (products : Finset ℕ) (A B C : ℕ) (hA : A ∈ products) (hB : B ∈ products) (hC : C ∈ products) (arrangements : Finset (Fin 5 → ℕ)) :
  (5 ∈ products ∧ (∀ a ∈ arrangements, ((a 0 = A ∧ a 1 = B) ∨ (a 1 = A ∧ a 2 = B) ∨ (a 2 = A ∧ a 3 = B) ∨ (a 3 = A ∧ a 4 = B)) ∧
   (a 0 ≠ A ∧ a 1 ≠ B ∧ a 2 ≠ C))) →
  arrangements.card = 36 :=
by sorry

end arrangement_with_A_in_middle_arrangement_with_A_at_end_B_not_at_end_arrangement_with_A_B_adjacent_not_adjacent_to_C_l150_150829


namespace longest_altitudes_sum_problem_statement_l150_150282

-- We define the sides of the triangle.
def sideA : ℕ := 6
def sideB : ℕ := 8
def sideC : ℕ := 10

-- Here, we state that the triangle formed by these sides is a right triangle.
def isRightTriangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- We assert that the triangle with sides 6, 8, and 10 is a right triangle.
def triangleIsRight : Prop := isRightTriangle sideA sideB sideC

-- We need to find and prove the sum of the lengths of the two longest altitudes.
def sumOfAltitudes (a b c : ℕ) (h : isRightTriangle a b c) : ℕ :=
  a + b

-- Finally, we state the theorem we want to prove.
theorem longest_altitudes_sum {a b c : ℕ} (h : isRightTriangle a b c) : sumOfAltitudes a b c h = 14 := by
  -- skipping the full proof
  sorry

-- Concrete instance for the given problem conditions
theorem problem_statement : longest_altitudes_sum triangleIsRight = 14 := by
  -- skipping the full proof
  sorry

end longest_altitudes_sum_problem_statement_l150_150282


namespace baker_extra_cakes_l150_150376

-- Defining the conditions
def original_cakes : ℕ := 78
def total_cakes : ℕ := 87
def extra_cakes := total_cakes - original_cakes

-- The statement to prove
theorem baker_extra_cakes : extra_cakes = 9 := by
  sorry

end baker_extra_cakes_l150_150376


namespace minimize_expression_l150_150470

theorem minimize_expression (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z) (h_xyz : x * y * z = 2 / 3) :
  x^2 + 6 * x * y + 18 * y^2 + 12 * y * z + 4 * z^2 = 18 :=
sorry

end minimize_expression_l150_150470


namespace smallest_n_not_prime_l150_150709

theorem smallest_n_not_prime : ∃ n, n = 4 ∧ ∀ m : ℕ, m < 4 → Prime (2 * m + 1) ∧ ¬ Prime (2 * 4 + 1) :=
by
  sorry

end smallest_n_not_prime_l150_150709


namespace melanie_more_turnips_l150_150903

theorem melanie_more_turnips (melanie_turnips benny_turnips : ℕ) (h1 : melanie_turnips = 139) (h2 : benny_turnips = 113) :
  melanie_turnips - benny_turnips = 26 := by
  sorry

end melanie_more_turnips_l150_150903


namespace problem1_problem2_l150_150627

-- Definitions
def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

-- Statement 1: If a = 1 and p ∧ q is true, then the range of x is 2 < x < 3
theorem problem1 (x : ℝ) (h : 1 = 1) (hpq : p x 1 ∧ q x) : 2 < x ∧ x < 3 :=
sorry

-- Statement 2: If ¬p is a sufficient but not necessary condition for ¬q, then the range of a is 1 < a ≤ 2
theorem problem2 (a : ℝ) (h1 : 1 < a) (h2 : a ≤ 2) (h3 : ¬ (∃ x, p x a) → ¬ (∃ x, q x)) : 1 < a ∧ a ≤ 2 :=
sorry

end problem1_problem2_l150_150627


namespace fisherman_gets_14_tunas_every_day_l150_150050

-- Define the conditions
def red_snappers_per_day := 8
def cost_per_red_snapper := 3
def cost_per_tuna := 2
def total_earnings_per_day := 52

-- Define the hypothesis
def total_earnings_from_red_snappers := red_snappers_per_day * cost_per_red_snapper  -- $24
def total_earnings_from_tunas := total_earnings_per_day - total_earnings_from_red_snappers -- $28
def number_of_tunas := total_earnings_from_tunas / cost_per_tuna -- 14

-- Lean statement to verify
theorem fisherman_gets_14_tunas_every_day : number_of_tunas = 14 :=
by 
  sorry

end fisherman_gets_14_tunas_every_day_l150_150050


namespace george_paint_l150_150305

theorem george_paint colors : fintype colors →  fin (Card colors) = 9 → (Card { x : (fin 9) // x ∈ comb 3 }) = 84 := by sorry

end george_paint_l150_150305


namespace vertex_of_parabola_l150_150650

theorem vertex_of_parabola (c d : ℝ) (h₁ : ∀ x, -x^2 + c*x + d ≤ 0 ↔ (x ≤ -1 ∨ x ≥ 7)) : 
  ∃ v : ℝ × ℝ, v = (3, 16) :=
by
  sorry

end vertex_of_parabola_l150_150650


namespace area_of_circle_l150_150007

-- Define the given conditions
def pi_approx : ℝ := 3
def radius : ℝ := 0.6

-- Prove that the area is 1.08 given the conditions
theorem area_of_circle : π = pi_approx → radius = 0.6 → 
  (pi_approx * radius^2 = 1.08) :=
by
  intros hπ hr
  sorry

end area_of_circle_l150_150007


namespace units_digit_of_square_ne_2_l150_150947

theorem units_digit_of_square_ne_2 (n : ℕ) : (n * n) % 10 ≠ 2 :=
sorry

end units_digit_of_square_ne_2_l150_150947


namespace concentric_circle_area_ratio_l150_150992

def circle_area (r : ℝ) : ℝ := π * r^2

def ring_area (r_outer r_inner : ℝ) : ℝ := circle_area r_outer - circle_area r_inner

theorem concentric_circle_area_ratio :
  let radii := [1, 3, 5, 7, 9]
  let black_areas := [radii.head!].map circle_area ++ [ring_area radii[2] radii[1], ring_area radii[4] radii[3]]
  let white_areas := [ring_area radii[1] radii[0], ring_area radii[3] radii[2]]
  (black_areas.sum / white_areas.sum = (49 : ℚ) / 32) :=
by
  sorry

end concentric_circle_area_ratio_l150_150992


namespace equal_distribution_l150_150395

theorem equal_distribution (k : ℤ) : ∃ n : ℤ, n = 81 + 95 * k ∧ ∃ b : ℤ, (19 + 6 * n) = 95 * b :=
by
  -- to be proved
  sorry

end equal_distribution_l150_150395


namespace distance_between_trees_l150_150029

theorem distance_between_trees (L : ℕ) (n : ℕ) (hL : L = 150) (hn : n = 11) (h_end_trees : n > 1) : 
  (L / (n - 1)) = 15 :=
by
  -- Replace with the appropriate proof
  sorry

end distance_between_trees_l150_150029


namespace intersect_point_l150_150176

noncomputable def f (x : ℤ) (b : ℤ) : ℤ := 5 * x + b
noncomputable def f_inv (x : ℤ) (b : ℤ) : ℤ := (x - b) / 5

theorem intersect_point (a b : ℤ) (h_intersections : (f (-3) b = a ∧ f a b = -3)) : a = -3 :=
by
  sorry

end intersect_point_l150_150176


namespace fraction_increase_by_50_percent_l150_150593

variable (x y : ℝ)
variable (h1 : 0 < y)

theorem fraction_increase_by_50_percent (h2 : 0.6 * x / 0.4 * y = 1.5 * x / y) : 
  1.5 * (x / y) = 1.5 * (x / y) :=
by
  sorry

end fraction_increase_by_50_percent_l150_150593


namespace girls_with_short_hair_count_l150_150925

-- Definitions based on the problem's conditions
def TotalPeople := 55
def Boys := 30
def FractionLongHair : ℚ := 3 / 5

-- The statement to prove
theorem girls_with_short_hair_count :
  (TotalPeople - Boys) - (TotalPeople - Boys) * FractionLongHair = 10 :=
by
  sorry

end girls_with_short_hair_count_l150_150925


namespace remaining_funds_correct_l150_150696

def david_initial_funds : ℝ := 1800
def emma_initial_funds : ℝ := 2400
def john_initial_funds : ℝ := 1200

def david_spent_percentage : ℝ := 0.60
def emma_spent_percentage : ℝ := 0.75
def john_spent_percentage : ℝ := 0.50

def david_remaining_funds : ℝ := david_initial_funds * (1 - david_spent_percentage)
def emma_spent : ℝ := emma_initial_funds * emma_spent_percentage
def emma_remaining_funds : ℝ := emma_spent - 800
def john_remaining_funds : ℝ := john_initial_funds * (1 - john_spent_percentage)

theorem remaining_funds_correct :
  david_remaining_funds = 720 ∧
  emma_remaining_funds = 1400 ∧
  john_remaining_funds = 600 :=
by
  sorry

end remaining_funds_correct_l150_150696


namespace ratio_ab_l150_150853

theorem ratio_ab (a b : ℚ) (h : b / a = 5 / 13) : (a - b) / (a + b) = 4 / 9 :=
by
  sorry

end ratio_ab_l150_150853


namespace final_hair_length_l150_150630

theorem final_hair_length (x y z : ℕ) (hx : x = 16) (hy : y = 11) (hz : z = 12) : 
  (x - y) + z = 17 :=
by
  sorry

end final_hair_length_l150_150630


namespace determine_jug_capacity_l150_150332

variable (jug_capacity : Nat)
variable (small_jug : Nat)

theorem determine_jug_capacity (h1 : jug_capacity = 5) (h2 : small_jug = 3 ∨ small_jug = 4):
  (∃ overflow_remains : Nat, 
    (overflow_remains = jug_capacity ∧ small_jug = 4) ∨ 
    (¬(overflow_remains = jug_capacity) ∧ small_jug = 3)) :=
by
  sorry

end determine_jug_capacity_l150_150332


namespace discount_rate_l150_150074

theorem discount_rate (marked_price selling_price discount_rate: ℝ) 
  (h₁: marked_price = 80)
  (h₂: selling_price = 68)
  (h₃: discount_rate = ((marked_price - selling_price) / marked_price) * 100) : 
  discount_rate = 15 :=
by
  sorry

end discount_rate_l150_150074


namespace sprint_time_l150_150155

def speed (Mark : Type) : ℝ := 6.0
def distance (Mark : Type) : ℝ := 144.0

theorem sprint_time (Mark : Type) : (distance Mark) / (speed Mark) = 24 := by
  sorry

end sprint_time_l150_150155


namespace num_friends_l150_150156

-- Define the friends
def Mary : Prop := ∃ n : ℕ, n = 6
def Sam : Prop := ∃ n : ℕ, n = 6
def Keith : Prop := ∃ n : ℕ, n = 6
def Alyssa : Prop := ∃ n : ℕ, n = 6

-- Define the set of friends
def friends : set Prop := {Mary, Sam, Keith, Alyssa}

-- Statement to prove
theorem num_friends (h1 : Mary) (h2 : Sam) (h3 : Keith) (h4 : Alyssa) : 
  set.card friends = 4 :=
by sorry

end num_friends_l150_150156


namespace average_speed_calculation_l150_150552

def average_speed (s1 s2 t1 t2 : ℕ) : ℕ :=
  (s1 * t1 + s2 * t2) / (t1 + t2)

theorem average_speed_calculation :
  average_speed 40 60 1 3 = 55 :=
by
  -- skipping the proof
  sorry

end average_speed_calculation_l150_150552


namespace build_wall_in_days_l150_150606

noncomputable def constant_k : ℝ := 20 * 6
def inverse_proportion (m : ℝ) (d : ℝ) : Prop := m * d = constant_k

theorem build_wall_in_days (d : ℝ) : inverse_proportion 30 d → d = 4.0 :=
by
  intros h
  have : d = constant_k / 30,
    sorry
  rw this,
  exact (by norm_num : 120 / 30 = 4.0)

end build_wall_in_days_l150_150606


namespace range_of_m_l150_150714

theorem range_of_m (m : ℝ) :
  (∃ (m : ℝ), (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m * x1 + 1 = 0 ∧ x2^2 + m * x2 + 1 = 0) ∧ 
  (∃ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 ≤ 0)) ↔ (m ≤ 1 ∨ m ≥ 3 ∨ m < -2) :=
by
  sorry

end range_of_m_l150_150714


namespace three_x_squared_y_squared_eq_588_l150_150624

theorem three_x_squared_y_squared_eq_588 (x y : ℤ) 
  (h : y^2 + 3 * x^2 * y^2 = 30 * x^2 + 517) : 
  3 * x^2 * y^2 = 588 :=
sorry

end three_x_squared_y_squared_eq_588_l150_150624


namespace find_mark_age_l150_150067

-- Define Mark and Aaron's ages
variables (M A : ℕ)

-- The conditions
def condition1 : Prop := M - 3 = 3 * (A - 3) + 1
def condition2 : Prop := M + 4 = 2 * (A + 4) + 2

-- The proof statement
theorem find_mark_age (h1 : condition1 M A) (h2 : condition2 M A) : M = 28 :=
by sorry

end find_mark_age_l150_150067


namespace conic_sections_parabolas_l150_150237

theorem conic_sections_parabolas (x y : ℝ) :
  (y^6 - 9*x^6 = 3*y^3 - 1) → 
  ((y^3 = 3*x^3 + 1) ∨ (y^3 = -3*x^3 + 1)) := 
by 
  sorry

end conic_sections_parabolas_l150_150237


namespace solve_equation_l150_150167

theorem solve_equation (x : ℝ) (h : (x^2 - x + 2) / (x - 1) = x + 3) (h1 : x ≠ 1) : 
  x = 5 / 3 :=
sorry

end solve_equation_l150_150167


namespace max_marks_tests_l150_150819

theorem max_marks_tests :
  ∃ (T1 T2 T3 T4 : ℝ),
    0.30 * T1 = 80 + 40 ∧
    0.40 * T2 = 105 + 35 ∧
    0.50 * T3 = 150 + 50 ∧
    0.60 * T4 = 180 + 60 ∧
    T1 = 400 ∧
    T2 = 350 ∧
    T3 = 400 ∧
    T4 = 400 :=
by
    sorry

end max_marks_tests_l150_150819


namespace geometric_sequence_fraction_l150_150265

variable (a_1 : ℝ) (q : ℝ)

theorem geometric_sequence_fraction (h : q = 2) :
  (2 * a_1 + a_1 * q) / (2 * (a_1 * q^2) + a_1 * q^3) = 1 / 4 :=
by sorry

end geometric_sequence_fraction_l150_150265


namespace area_of_quadrilateral_l150_150459

def Quadrilateral (A B C D : Type) :=
  ∃ (ABC_deg : ℝ) (ADC_deg : ℝ) (AD : ℝ) (DC : ℝ) (AB : ℝ) (BC : ℝ),
  (ABC_deg = 90) ∧ (ADC_deg = 90) ∧ (AD = DC) ∧ (AB + BC = 20)

theorem area_of_quadrilateral (A B C D : Type) (h : Quadrilateral A B C D) : 
  ∃ (area : ℝ), area = 100 := 
sorry

end area_of_quadrilateral_l150_150459


namespace john_total_amount_l150_150613

-- Given conditions from a)
def grandpa_amount : ℕ := 30
def grandma_amount : ℕ := 3 * grandpa_amount

-- Problem statement
theorem john_total_amount : grandpa_amount + grandma_amount = 120 :=
by
  sorry

end john_total_amount_l150_150613


namespace min_value_expression_l150_150759

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 48) :
  x^2 + 4 * x * y + 4 * y^2 + 3 * z^2 ≥ 144 :=
sorry

end min_value_expression_l150_150759


namespace ratio_tough_to_good_sales_l150_150907

-- Define the conditions
def tough_week_sales : ℤ := 800
def total_sales : ℤ := 10400
def good_weeks : ℕ := 5
def tough_weeks : ℕ := 3

-- Define the problem in Lean 4:
theorem ratio_tough_to_good_sales : ∃ G : ℤ, (good_weeks * G) + (tough_weeks * tough_week_sales) = total_sales ∧ 
  (tough_week_sales : ℚ) / (G : ℚ) = 1 / 2 :=
sorry

end ratio_tough_to_good_sales_l150_150907


namespace find_sixth_number_l150_150078

theorem find_sixth_number (avg_all : ℝ) (avg_first6 : ℝ) (avg_last6 : ℝ) (total_avg : avg_all = 10.7) (first6_avg: avg_first6 = 10.5) (last6_avg: avg_last6 = 11.4) : 
  let S1 := 6 * avg_first6
  let S2 := 6 * avg_last6
  let total_sum := 11 * avg_all
  let X := total_sum - (S1 - X + S2 - X)
  X = 13.7 :=
by 
  sorry

end find_sixth_number_l150_150078


namespace find_number_l150_150873

theorem find_number (N : ℝ) (h : 0.015 * N = 90) : N = 6000 :=
  sorry

end find_number_l150_150873


namespace Maggie_earnings_l150_150632

theorem Maggie_earnings :
  let family_commission := 7
  let neighbor_commission := 6
  let bonus_fixed := 10
  let bonus_threshold := 10
  let bonus_per_subscription := 1
  let monday_family := 4 + 1 
  let tuesday_neighbors := 2 + 2 * 2
  let wednesday_family := 3 + 1
  let total_family := monday_family + wednesday_family
  let total_neighbors := tuesday_neighbors
  let total_subscriptions := total_family + total_neighbors
  let bonus := if total_subscriptions > bonus_threshold then 
                 bonus_fixed + bonus_per_subscription * (total_subscriptions - bonus_threshold)
               else 0
  let total_earnings := total_family * family_commission + total_neighbors * neighbor_commission + bonus
  total_earnings = 114 := 
by {
  -- Placeholder for the proof. We assume this step will contain a verification of derived calculations.
  sorry
}

end Maggie_earnings_l150_150632


namespace probability_of_two_red_balls_l150_150673

-- Define the total number of balls, number of red balls, and number of white balls
def total_balls := 6
def red_balls := 4
def white_balls := 2
def drawn_balls := 2

-- Define the combination formula
def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The number of ways to choose 2 red balls from 4
def ways_to_choose_red := choose 4 2

-- The number of ways to choose any 2 balls from the total of 6
def ways_to_choose_any := choose 6 2

-- The corresponding probability
def probability := ways_to_choose_red / ways_to_choose_any

-- The theorem we want to prove
theorem probability_of_two_red_balls :
  probability = 2 / 5 :=
by
  sorry

end probability_of_two_red_balls_l150_150673


namespace Q_subset_P_l150_150291

-- Definitions of the sets P and Q
def P : Set ℝ := {x | x ≥ 5}
def Q : Set ℝ := {x | 5 ≤ x ∧ x ≤ 7}

-- Statement to prove the relationship between P and Q
theorem Q_subset_P : Q ⊆ P :=
by
  sorry

end Q_subset_P_l150_150291


namespace intersection_eq_l150_150901

noncomputable def A := {x : ℝ | x^2 - 4*x + 3 < 0 }
noncomputable def B := {x : ℝ | 2*x - 3 > 0 }

theorem intersection_eq : (A ∩ B) = {x : ℝ | (3 / 2) < x ∧ x < 3} := by
  sorry

end intersection_eq_l150_150901


namespace delta_value_l150_150447

-- Define the variables and the hypothesis
variable (Δ : Int)
variable (h : 5 * (-3) = Δ - 3)

-- State the theorem
theorem delta_value : Δ = -12 := by
  sorry

end delta_value_l150_150447


namespace power_of_two_plus_one_div_by_power_of_three_l150_150769

theorem power_of_two_plus_one_div_by_power_of_three (n : ℕ) : 3^(n + 1) ∣ (2^(3^n) + 1) :=
sorry

end power_of_two_plus_one_div_by_power_of_three_l150_150769


namespace temperature_value_l150_150738

theorem temperature_value (k : ℝ) (t : ℝ) (h1 : t = 5 / 9 * (k - 32)) (h2 : k = 221) : t = 105 :=
by
  sorry

end temperature_value_l150_150738


namespace average_book_width_is_3_point_9375_l150_150754

def book_widths : List ℚ := [3, 4, 3/4, 1.5, 7, 2, 5.25, 8]
def number_of_books : ℚ := 8
def total_width : ℚ := List.sum book_widths
def average_width : ℚ := total_width / number_of_books

theorem average_book_width_is_3_point_9375 :
  average_width = 3.9375 := by
  sorry

end average_book_width_is_3_point_9375_l150_150754


namespace quadratic_roots_l150_150344

theorem quadratic_roots (c : ℝ) 
  (h : ∀ x : ℝ, (x^2 - 3*x + c = 0) ↔ (x = (3 + Real.sqrt c) / 2 ∨ x = (3 - Real.sqrt c) / 2)) :
  c = 9 / 5 :=
by
  sorry

end quadratic_roots_l150_150344


namespace smallest_integer_for_perfect_square_l150_150966

theorem smallest_integer_for_perfect_square :
  let y := 2^5 * 3^5 * 4^5 * 5^5 * 6^4 * 7^3 * 8^3 * 9^2
  ∃ z : ℕ, z = 70 ∧ (∃ k : ℕ, y * z = k^2) :=
by
  sorry

end smallest_integer_for_perfect_square_l150_150966


namespace carpet_shaded_area_is_correct_l150_150967

def total_shaded_area (carpet_side_length : ℝ) (large_square_side : ℝ) (small_square_side : ℝ) : ℝ :=
  let large_shaded_area := large_square_side * large_square_side
  let small_shaded_area := small_square_side * small_square_side
  large_shaded_area + 12 * small_shaded_area

theorem carpet_shaded_area_is_correct :
  ∀ (S T : ℝ), 
  12 / S = 4 →
  S / T = 4 →
  total_shaded_area 12 S T = 15.75 :=
by
  intros S T h1 h2
  sorry

end carpet_shaded_area_is_correct_l150_150967


namespace distinct_L_shapes_l150_150543

-- Definitions of conditions
def num_convex_shapes : Nat := 10
def L_shapes_per_convex : Nat := 2
def corner_L_shapes : Nat := 4

-- Total number of distinct "L" shapes
def total_L_shapes : Nat :=
  num_convex_shapes * L_shapes_per_convex + corner_L_shapes

theorem distinct_L_shapes :
  total_L_shapes = 24 :=
by
  -- Proof is omitted
  sorry

end distinct_L_shapes_l150_150543


namespace compute_g_neg_101_l150_150639

variable (g : ℝ → ℝ)

def functional_eqn := ∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x
def g_neg_one := g (-1) = 3
def g_one := g (1) = 1

theorem compute_g_neg_101 (g : ℝ → ℝ)
  (H1 : functional_eqn g)
  (H2 : g_neg_one g)
  (H3 : g_one g) :
  g (-101) = 103 := 
by
  sorry

end compute_g_neg_101_l150_150639


namespace coat_shirt_ratio_l150_150158

variable (P S C k : ℕ)

axiom h1 : P + S = 100
axiom h2 : P + C = 244
axiom h3 : C = k * S
axiom h4 : C = 180

theorem coat_shirt_ratio (P S C k : ℕ) (h1 : P + S = 100) (h2 : P + C = 244) (h3 : C = k * S) (h4 : C = 180) :
  C / S = 5 :=
sorry

end coat_shirt_ratio_l150_150158


namespace intersection_point_on_y_axis_l150_150493

theorem intersection_point_on_y_axis (k : ℝ) :
  ∃ y : ℝ, 2 * 0 + 3 * y - k = 0 ∧ 0 - k * y + 12 = 0 ↔ k = 6 ∨ k = -6 :=
by
  sorry

end intersection_point_on_y_axis_l150_150493


namespace reciprocal_equality_l150_150292

theorem reciprocal_equality (a b : ℝ) (h1 : 1 / a = -8) (h2 : 1 / -b = 8) : a = b :=
sorry

end reciprocal_equality_l150_150292


namespace minimal_range_of_observations_l150_150386

variable {x1 x2 x3 x4 x5 : ℝ}

def arithmetic_mean (x1 x2 x3 x4 x5 : ℝ) : Prop :=
  (x1 + x2 + x3 + x4 + x5) / 5 = 8

def median (x1 x2 x3 x4 x5 : ℝ) : Prop :=
  x3 = 10 ∧ x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 ∧ x4 ≤ x5

theorem minimal_range_of_observations 
  (h_mean : arithmetic_mean x1 x2 x3 x4 x5)
  (h_median : median x1 x2 x3 x4 x5) : 
  ∃ x1 x2 x3 x4 x5 : ℝ, (x1 + x2 + x3 + x4 + x5) = 40 ∧ x3 = 10 ∧ x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 ∧ x4 ≤ x5 ∧ (x5 - x1) = 5 :=
by 
  sorry

end minimal_range_of_observations_l150_150386


namespace min_sticks_to_be_broken_form_square_without_breaks_l150_150060

noncomputable def total_length (n : ℕ) : ℕ := n * (n + 1) / 2

def divisible_by_4 (x : ℕ) : Prop := x % 4 = 0

theorem min_sticks_to_be_broken (n : ℕ) : n = 12 → (¬ divisible_by_4 (total_length n)) ∧ (minimal_breaks n = 2) :=
by
  intro h1
  rw h1
  have h2 : total_length 12 = 78 := by decide
  have h3 : ¬ divisible_by_4 78 := by decide
  exact ⟨h3, sorry⟩

theorem form_square_without_breaks (n : ℕ) : n = 15 → divisible_by_4 (total_length n) ∧ (minimal_breaks n = 0) :=
by
  intro h1
  rw h1
  have h2 : total_length 15 = 120 := by decide
  have h3 : divisible_by_4 120 := by decide
  exact ⟨h3, sorry⟩

end min_sticks_to_be_broken_form_square_without_breaks_l150_150060


namespace cost_price_of_one_toy_l150_150537

theorem cost_price_of_one_toy (C : ℝ) (h : 21 * C = 21000) : C = 1000 :=
by sorry

end cost_price_of_one_toy_l150_150537


namespace smallest_integer_value_of_m_l150_150707

def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

theorem smallest_integer_value_of_m :
  ∀ m : ℤ, (x^2 + 4 * x - m = 0) ∧ has_two_distinct_real_roots 1 4 (-m : ℝ) → m ≥ -3 :=
by
  intro m h
  sorry

end smallest_integer_value_of_m_l150_150707


namespace carmela_gives_each_l150_150100

noncomputable def money_needed_to_give_each (carmela : ℕ) (cousins : ℕ) (cousins_count : ℕ) : ℕ :=
  let total_cousins_money := cousins * cousins_count
  let total_money := carmela + total_cousins_money
  let people_count := 1 + cousins_count
  let equal_share := total_money / people_count
  let total_giveaway := carmela - equal_share
  total_giveaway / cousins_count

theorem carmela_gives_each (carmela : ℕ) (cousins : ℕ) (cousins_count : ℕ) (h_carmela : carmela = 7) (h_cousins : cousins = 2) (h_cousins_count : cousins_count = 4) :
  money_needed_to_give_each carmela cousins cousins_count = 1 :=
by
  rw [h_carmela, h_cousins, h_cousins_count]
  sorry

end carmela_gives_each_l150_150100


namespace parker_total_weight_l150_150044

-- Define the number of initial dumbbells and their weight
def initial_dumbbells := 4
def weight_per_dumbbell := 20

-- Define the number of additional dumbbells
def additional_dumbbells := 2

-- Define the total weight calculation
def total_weight := initial_dumbbells * weight_per_dumbbell + additional_dumbbells * weight_per_dumbbell

-- Prove that the total weight is 120 pounds
theorem parker_total_weight : total_weight = 120 :=
by
  -- proof skipped
  sorry

end parker_total_weight_l150_150044


namespace parabola_vertex_shift_l150_150476

theorem parabola_vertex_shift
  (vertex_initial : ℝ × ℝ)
  (h₀ : vertex_initial = (0, 0))
  (move_left : ℝ)
  (move_up : ℝ)
  (h₁ : move_left = -2)
  (h₂ : move_up = 3):
  (vertex_initial.1 + move_left, vertex_initial.2 + move_up) = (-2, 3) :=
by
  sorry

end parabola_vertex_shift_l150_150476


namespace range_of_a_nonempty_intersection_range_of_a_subset_intersection_l150_150629

-- Define set A
def A : Set ℝ := {x | (x + 1) * (4 - x) ≤ 0}

-- Define set B in terms of variable a
def B (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 2}

-- Statement 1: Proving the range of a when A ∩ B ≠ ∅
theorem range_of_a_nonempty_intersection (a : ℝ) : (A ∩ B a ≠ ∅) → (-1 / 2 ≤ a ∧ a ≤ 2) :=
by
  sorry

-- Statement 2: Proving the range of a when A ∩ B = B
theorem range_of_a_subset_intersection (a : ℝ) : (A ∩ B a = B a) → (a ≥ 2 ∨ a ≤ -3) :=
by
  sorry

end range_of_a_nonempty_intersection_range_of_a_subset_intersection_l150_150629


namespace value_of_expression_l150_150661

theorem value_of_expression : 
  103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 := by
  sorry

end value_of_expression_l150_150661


namespace mixed_sum_proof_l150_150570

def mixed_sum : ℚ :=
  3 + 1/3 + 4 + 1/2 + 5 + 1/5 + 6 + 1/6

def smallest_whole_number_greater_than_mixed_sum : ℤ :=
  Int.ceil (mixed_sum)

theorem mixed_sum_proof :
  smallest_whole_number_greater_than_mixed_sum = 20 := by
  sorry

end mixed_sum_proof_l150_150570


namespace fourth_and_fifth_suppliers_cars_equal_l150_150684

-- Define the conditions
def total_cars : ℕ := 5650000
def cars_supplier_1 : ℕ := 1000000
def cars_supplier_2 : ℕ := cars_supplier_1 + 500000
def cars_supplier_3 : ℕ := cars_supplier_1 + cars_supplier_2
def cars_distributed_first_three : ℕ := cars_supplier_1 + cars_supplier_2 + cars_supplier_3
def cars_remaining : ℕ := total_cars - cars_distributed_first_three

-- Theorem stating the question and answer
theorem fourth_and_fifth_suppliers_cars_equal 
  : (cars_remaining / 2) = 325000 := by
  sorry

end fourth_and_fifth_suppliers_cars_equal_l150_150684


namespace sum_of_squares_expr_l150_150549

theorem sum_of_squares_expr : 
  (23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2) = 288 := 
by
  sorry

end sum_of_squares_expr_l150_150549


namespace probability_interval_l150_150001

variable (P_A P_B q : ℚ)

axiom prob_A : P_A = 5/6
axiom prob_B : P_B = 3/4
axiom prob_A_and_B : q = P_A + P_B - 1

theorem probability_interval :
  7/12 ≤ q ∧ q ≤ 3/4 :=
by
  sorry

end probability_interval_l150_150001


namespace circle_equation_l150_150788

theorem circle_equation (x y : ℝ) :
  let center := (0, 4)
  let point_on_circle := (3, 0)
  (x - center.1)^2 + (y - center.2)^2 = 25 :=
by
  sorry

end circle_equation_l150_150788


namespace normal_price_of_article_l150_150805

theorem normal_price_of_article (P : ℝ) (h : 0.90 * 0.80 * P = 36) : P = 50 :=
by {
  sorry
}

end normal_price_of_article_l150_150805


namespace tessa_owes_30_l150_150867

-- Definitions based on given conditions
def initial_debt : ℕ := 40
def paid_back : ℕ := initial_debt / 2
def remaining_debt_after_payment : ℕ := initial_debt - paid_back
def additional_borrowing : ℕ := 10
def total_debt : ℕ := remaining_debt_after_payment + additional_borrowing

-- Theorem to be proved
theorem tessa_owes_30 : total_debt = 30 :=
by
  sorry

end tessa_owes_30_l150_150867


namespace fold_hexagon_possible_l150_150247

theorem fold_hexagon_possible (a b : ℝ) :
  (∃ x : ℝ, (a - x)^2 + (b - x)^2 = x^2) ↔ (1 / 2 < b / a ∧ b / a < 2) :=
by
  sorry

end fold_hexagon_possible_l150_150247


namespace count_integers_log_condition_l150_150249

theorem count_integers_log_condition :
  (∃! n : ℕ, n = 54 ∧ (∀ x : ℕ, x > 30 ∧ x < 90 ∧ ((x - 30) * (90 - x) < 1000) ↔ (31 <= x ∧ x <= 84))) :=
sorry

end count_integers_log_condition_l150_150249


namespace smallest_possible_QNNN_l150_150196

theorem smallest_possible_QNNN :
  ∃ (Q N : ℕ), (N = 1 ∨ N = 5 ∨ N = 6) ∧ (NN = 10 * N + N) ∧ (Q * 1000 + NN * 10 + N = NN * N) ∧ (Q * 1000 + NN * 10 + N) = 275 :=
sorry

end smallest_possible_QNNN_l150_150196


namespace equation_has_real_roots_l150_150160

theorem equation_has_real_roots (a b : ℝ) (h : ¬ (a = 0 ∧ b = 0)) :
  ∃ x : ℝ, x ≠ 1 ∧ (a^2 / x + b^2 / (x - 1) = 1) :=
by
  sorry

end equation_has_real_roots_l150_150160


namespace expected_value_twelve_sided_die_l150_150824

theorem expected_value_twelve_sided_die : 
  let die_sides := 12 in 
  let outcomes := finset.range (die_sides + 1) in
  (finset.sum outcomes id : ℚ) / die_sides = 6.5 :=
by
  sorry

end expected_value_twelve_sided_die_l150_150824


namespace George_colors_combination_l150_150303

def binom (n k : ℕ) : ℕ := n.choose k

theorem George_colors_combination : binom 9 3 = 84 :=
by {
  exact Nat.choose_eq_factorial_div_factorial (le_refl 3)
}

end George_colors_combination_l150_150303


namespace general_formula_a_S_n_no_arithmetic_sequence_in_b_l150_150719

def sequence_a (a : ℕ → ℚ) :=
  (a 1 = 1 / 4) ∧ (∀ n : ℕ, n > 0 → 3 * a (n + 1) - 2 * a n = 1)

def sequence_b (b : ℕ → ℚ) (a : ℕ → ℚ) :=
  ∀ n : ℕ, n > 0 → b n = a (n + 1) - a n

theorem general_formula_a_S_n (a : ℕ → ℚ) (S : ℕ → ℚ) :
  sequence_a a →
  (∀ n : ℕ, n > 0 → a n = 1 - (3 / 4) * (2 / 3)^(n - 1)) →
  (∀ n : ℕ, n > 0 → S n = (2 / 3)^(n - 2) + n - 9 / 4) →
  True := sorry

theorem no_arithmetic_sequence_in_b (b : ℕ → ℚ) (a : ℕ → ℚ) :
  sequence_b b a →
  (∀ n : ℕ, n > 0 → b n = (1 / 4) * (2 / 3)^(n - 1)) →
  (∀ r s t : ℕ, r < s ∧ s < t → ¬ (b s - b r = b t - b s)) :=
  sorry

end general_formula_a_S_n_no_arithmetic_sequence_in_b_l150_150719


namespace find_insect_stickers_l150_150939

noncomputable def flower_stickers : ℝ := 15
noncomputable def animal_stickers : ℝ := 2 * flower_stickers - 3.5
noncomputable def space_stickers : ℝ := 1.5 * flower_stickers + 5.5
noncomputable def total_stickers : ℝ := 70
noncomputable def insect_stickers : ℝ := total_stickers - (animal_stickers + space_stickers)

theorem find_insect_stickers : insect_stickers = 15.5 := by
  sorry

end find_insect_stickers_l150_150939


namespace short_haired_girls_l150_150929

def total_people : ℕ := 55
def boys : ℕ := 30
def total_girls : ℕ := total_people - boys
def girls_with_long_hair : ℕ := (3 / 5) * total_girls
def girls_with_short_hair : ℕ := total_girls - girls_with_long_hair

theorem short_haired_girls :
  girls_with_short_hair = 10 := sorry

end short_haired_girls_l150_150929


namespace region_area_correct_l150_150568

noncomputable def region_area : ℝ :=
  let region := {p : ℝ × ℝ | |p.1 + p.2| + |p.1 - p.2| ≤ 6}
  let area := (3 - -3) * (3 - -3)
  area

theorem region_area_correct : region_area = 36 :=
by sorry

end region_area_correct_l150_150568


namespace triangle_area_in_circle_l150_150815

theorem triangle_area_in_circle (r : ℝ) (arc1 arc2 arc3 : ℝ) 
  (circumference_eq : arc1 + arc2 + arc3 = 24)
  (radius_eq : 2 * Real.pi * r = 24) : 
  1 / 2 * (r ^ 2) * (Real.sin (105 * Real.pi / 180) + Real.sin (120 * Real.pi / 180) + Real.sin (135 * Real.pi / 180)) = 364.416 / (Real.pi ^ 2) :=
by
  sorry

end triangle_area_in_circle_l150_150815


namespace sphere_surface_area_l150_150798

theorem sphere_surface_area (a : ℝ) (l R : ℝ)
  (h₁ : 6 * l^2 = a)
  (h₂ : l * Real.sqrt 3 = 2 * R) :
  4 * Real.pi * R^2 = (Real.pi / 2) * a :=
sorry

end sphere_surface_area_l150_150798


namespace totalFriendsAreFour_l150_150157

-- Define the friends
def friends := ["Mary", "Sam", "Keith", "Alyssa"]

-- Define the number of friends
def numberOfFriends (f : List String) : ℕ := f.length

-- Claim that the number of friends is 4
theorem totalFriendsAreFour : numberOfFriends friends = 4 :=
by
  -- Skip proof
  sorry

end totalFriendsAreFour_l150_150157


namespace rings_sold_l150_150314

theorem rings_sold (R : ℕ) : 
  ∀ (num_necklaces total_sales necklace_price ring_price : ℕ),
  num_necklaces = 4 →
  total_sales = 80 →
  necklace_price = 12 →
  ring_price = 4 →
  num_necklaces * necklace_price + R * ring_price = total_sales →
  R = 8 := 
by 
  intros num_necklaces total_sales necklace_price ring_price h1 h2 h3 h4 h5
  sorry

end rings_sold_l150_150314


namespace second_derivative_sin_squared_eq_2_cos_2x_l150_150841

noncomputable def y (x : ℝ) : ℝ := (Real.sin x) ^ 2

theorem second_derivative_sin_squared_eq_2_cos_2x : 
  deriv (deriv y) = λ x, 2 * Real.cos (2 * x) := 
sorry

end second_derivative_sin_squared_eq_2_cos_2x_l150_150841


namespace probability_distinct_digits_odd_units_l150_150573

-- Definition of conditions
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999
def allDigitsDistinct (n : ℕ) : Prop := 
  let digits := List.ofFn (λ i => n.digits (10^(i-1)) % 10) in
  digits.1 ≠ digits.2 ∧ digits.1 ≠ digits.3 ∧ digits.1 ≠ digits.4 ∧ digits.2 ≠ digits.3 ∧ digits.2 ≠ digits.4 ∧ digits.3 ≠ digits.4
def unitsDigitOdd (n : ℕ) : Prop := (n % 10) ∈ {1, 3, 5, 7, 9}

-- Main theorem statement
theorem probability_distinct_digits_odd_units : 
  let favorable_outcomes := 2240
  let total_outcomes := 9000
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  isFourDigit n ∧ allDigitsDistinct n ∧ unitsDigitOdd n → probability = 56 / 225 :=
by sorry

end probability_distinct_digits_odd_units_l150_150573


namespace area_isosceles_right_triangle_l150_150420

theorem area_isosceles_right_triangle 
( a : ℝ × ℝ )
( b : ℝ × ℝ )
( h_a : a = (Real.cos (2 / 3 * Real.pi), Real.sin (2 / 3 * Real.pi)) )
( is_isosceles_right_triangle : (a + b).fst * (a - b).fst + (a + b).snd * (a - b).snd = 0 
                                ∧ (a + b).fst * (a + b).fst + (a + b).snd * (a + b).snd 
                                = (a - b).fst * (a - b).fst + (a - b).snd * (a - b).snd ):
  1 / 2 * Real.sqrt ((1 - 1 / 2)^2 + (Real.sqrt 3 / 2 - -1 / 2)^2 )
 * Real.sqrt ((1 - -1 / 2)^2 + (Real.sqrt 3 / 2 - -1 / 2 )^2 ) = 1 :=
by
  sorry

end area_isosceles_right_triangle_l150_150420


namespace total_bottles_remaining_is_14090_l150_150811

-- Define the constants
def total_small_bottles : ℕ := 5000
def total_big_bottles : ℕ := 12000
def small_bottles_sold_percentage : ℕ := 15
def big_bottles_sold_percentage : ℕ := 18

-- Define the remaining bottles
def calc_remaining_bottles (total_bottles sold_percentage : ℕ) : ℕ :=
  total_bottles - (sold_percentage * total_bottles / 100)

-- Define the remaining small and big bottles
def remaining_small_bottles : ℕ := calc_remaining_bottles total_small_bottles small_bottles_sold_percentage
def remaining_big_bottles : ℕ := calc_remaining_bottles total_big_bottles big_bottles_sold_percentage

-- Define the total remaining bottles
def total_remaining_bottles : ℕ := remaining_small_bottles + remaining_big_bottles

-- State the theorem
theorem total_bottles_remaining_is_14090 : total_remaining_bottles = 14090 := by
  sorry

end total_bottles_remaining_is_14090_l150_150811


namespace consecutive_integers_avg_l150_150654

theorem consecutive_integers_avg (n x : ℤ) (h_avg : (2*x + n - 1 : ℝ)/2 = 20.5) (h_10th : x + 9 = 25) :
  n = 10 :=
by
  sorry

end consecutive_integers_avg_l150_150654


namespace expected_value_twelve_sided_die_l150_150823

theorem expected_value_twelve_sided_die : 
  (1 / 12 * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)) = 6.5 := by
  sorry

end expected_value_twelve_sided_die_l150_150823


namespace reece_climbs_15_times_l150_150895

/-
Given:
1. Keaton's ladder height: 30 feet.
2. Keaton climbs: 20 times.
3. Reece's ladder is 4 feet shorter than Keaton's ladder.
4. Total length of ladders climbed by both is 11880 inches.

Prove:
Reece climbed his ladder 15 times.
-/

theorem reece_climbs_15_times :
  let keaton_ladder_feet := 30
  let keaton_climbs := 20
  let reece_ladder_feet := keaton_ladder_feet - 4
  let total_length_inches := 11880
  let feet_to_inches (feet : ℕ) := 12 * feet
  let keaton_ladder_inches := feet_to_inches keaton_ladder_feet
  let reece_ladder_inches := feet_to_inches reece_ladder_feet
  let keaton_total_climbed := keaton_ladder_inches * keaton_climbs
  let reece_total_climbed := total_length_inches - keaton_total_climbed
  let reece_climbs := reece_total_climbed / reece_ladder_inches
  reece_climbs = 15 :=
by
  sorry

end reece_climbs_15_times_l150_150895


namespace ribbon_length_ratio_l150_150644

theorem ribbon_length_ratio (original_length reduced_length : ℕ) (h1 : original_length = 55) (h2 : reduced_length = 35) : 
  (original_length / Nat.gcd original_length reduced_length) = 11 ∧
  (reduced_length / Nat.gcd original_length reduced_length) = 7 := 
  by
    sorry

end ribbon_length_ratio_l150_150644


namespace find_percentage_loss_l150_150542

theorem find_percentage_loss 
  (P : ℝ)
  (initial_marbles remaining_marbles : ℝ)
  (h1 : initial_marbles = 100)
  (h2 : remaining_marbles = 20)
  (h3 : (initial_marbles - initial_marbles * P / 100) / 2 = remaining_marbles) :
  P = 60 :=
by
  sorry

end find_percentage_loss_l150_150542


namespace find_angle_A_determine_triangle_shape_l150_150028

noncomputable def angle_A (A : ℝ) (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = 7 / 2 ∧ m = (Real.cos (A / 2)^2, Real.cos (2 * A)) ∧ 
  n = (4, -1)

theorem find_angle_A : 
  ∃ A : ℝ,  (0 < A ∧ A < Real.pi) ∧ angle_A A (Real.cos (A / 2)^2, Real.cos (2 * A)) (4, -1) 
  := sorry

noncomputable def triangle_shape (a b c : ℝ) (A : ℝ) : Prop :=
  a = Real.sqrt 3 ∧ a^2 = b^2 + c^2 - b * c * Real.cos (A)

theorem determine_triangle_shape :
  ∀ (b c : ℝ), (b * c ≤ 3) → triangle_shape (Real.sqrt 3) b c (Real.pi / 3) →
  (b = Real.sqrt 3 ∧ c = Real.sqrt 3)
  := sorry


end find_angle_A_determine_triangle_shape_l150_150028


namespace angle_between_vectors_l150_150430

noncomputable theory

open Real
open Matrix

variables (a b : EuclideanSpace ℝ (Fin 2))

def vec_a : EuclideanSpace ℝ (Fin 2) := ![0, 1]
def vec_b : EuclideanSpace ℝ (Fin 2) := ![-sqrt 3, 1]

theorem angle_between_vectors :
  a - 2 • b = ![2 * sqrt 3, -1] → b - 2 • a = ![-sqrt 3, -1] →
  real.angle (vec a) (vec b) = π / 3 := 
by
sorry

end angle_between_vectors_l150_150430


namespace delta_value_l150_150444

theorem delta_value (Δ : ℤ) : 5 * (-3) = Δ - 3 → Δ = -12 :=
by
  sorry

end delta_value_l150_150444


namespace inequality_proof_l150_150576

theorem inequality_proof 
  (a b c d : ℝ) (n : ℕ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_n : 9 ≤ n) :
  a^n + b^n + c^n + d^n ≥ a^(n-9)*b^4*c^3*d^2 + b^(n-9)*c^4*d^3*a^2 + c^(n-9)*d^4*a^3*b^2 + d^(n-9)*a^4*b^3*c^2 :=
by
  sorry

end inequality_proof_l150_150576


namespace find_a_and_root_l150_150131

def equation_has_double_root (a x : ℝ) : Prop :=
  a * x^2 + 4 * x - 1 = 0

theorem find_a_and_root (a x : ℝ)
  (h_eqn : equation_has_double_root a x)
  (h_discriminant : 16 + 4 * a = 0) :
  a = -4 ∧ x = 1 / 2 :=
sorry

end find_a_and_root_l150_150131


namespace complex_equation_square_sum_l150_150258

-- Lean 4 statement of the mathematical proof problem
theorem complex_equation_square_sum (a b : ℝ) (i : ℂ) (h : i^2 = -1) 
    (h1 : (a - 2 * i) * i = b - i) : a^2 + b^2 = 5 := by
  sorry

end complex_equation_square_sum_l150_150258


namespace math_problem_l150_150133

theorem math_problem (a b : ℝ) (h1 : 4 + a = 5 - b) (h2 : 5 + b = 8 + a) : 4 - a = 3 :=
by
  sorry

end math_problem_l150_150133


namespace percentage_discount_l150_150400

theorem percentage_discount (P D: ℝ) 
  (sale_price: P * (100 - D) / 100 = 78.2)
  (final_price_increase: 78.2 * 1.25 = P - 5.75):
  D = 24.44 :=
by
  sorry

end percentage_discount_l150_150400


namespace min_sticks_12_to_break_can_form_square_15_l150_150062

-- Problem definition for n = 12
def sticks_12 : List Nat := List.range' 1 12

theorem min_sticks_12_to_break : 
  ... (I realize I need to translate a step better) ..............
  sorry

-- Problem definition for n = 15
def sticks_15 : List Nat := List.range' 1 15

theorem can_form_square_15 : 
  ... (implementing a nice explanation)
  sorry

end min_sticks_12_to_break_can_form_square_15_l150_150062


namespace age_sum_in_5_years_l150_150764

variable (MikeAge MomAge : ℕ)
variable (h1 : MikeAge = MomAge - 30)
variable (h2 : MikeAge + MomAge = 70)

theorem age_sum_in_5_years (h1 : MikeAge = MomAge - 30) (h2 : MikeAge + MomAge = 70) :
  (MikeAge + 5) + (MomAge + 5) = 80 := by
  sorry

end age_sum_in_5_years_l150_150764


namespace roger_initial_money_l150_150483

theorem roger_initial_money (x : ℤ) 
    (h1 : x + 28 - 25 = 19) : 
    x = 16 := 
by 
    sorry

end roger_initial_money_l150_150483


namespace find_value_of_m_l150_150269

-- Definition of the center of the circle
def center := (1 : ℝ, 0 : ℝ)

-- Definition of the line
def line (m : ℝ) : ℝ × ℝ → Prop := λ p, p.1 - m * p.2 + 1 = 0

-- Definition of the circle
def circle : ℝ × ℝ → Prop := λ p, (p.1 - 1) ^ 2 + p.2 ^ 2 = 4

-- Area condition
def area_condition (A B : ℝ × ℝ) : Prop :=
  let C := center in
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2) = 8 / 5

-- Main theorem statement
theorem find_value_of_m (m : ℝ) (A B : ℝ × ℝ) :
  line m A → line m B → circle A → circle B → area_condition A B → m = 2 :=
sorry

end find_value_of_m_l150_150269


namespace base8_to_base10_4532_l150_150002

theorem base8_to_base10_4532 : 
    (4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0) = 2394 := 
by sorry

end base8_to_base10_4532_l150_150002


namespace min_sum_m_n_l150_150263

open Nat

theorem min_sum_m_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : m * n - 2 * m - 3 * n - 20 = 0) : m + n = 20 :=
sorry

end min_sum_m_n_l150_150263


namespace four_number_theorem_l150_150374

theorem four_number_theorem (a b c d : ℕ) (H : a * b = c * d) (Ha : 0 < a) (Hb : 0 < b) (Hc : 0 < c) (Hd : 0 < d) : 
  ∃ (p q r s : ℕ), 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s ∧ a = p * q ∧ b = r * s ∧ c = p * s ∧ d = q * r :=
by
  sorry

end four_number_theorem_l150_150374


namespace lara_additional_miles_needed_l150_150756

theorem lara_additional_miles_needed :
  ∀ (d1 d2 d_total t1 speed1 speed2 avg_speed : ℝ),
    d1 = 20 →
    speed1 = 25 →
    speed2 = 40 →
    avg_speed = 35 →
    t1 = d1 / speed1 →
    d_total = d1 + d2 →
    avg_speed = (d_total) / (t1 + d2 / speed2) →
    d2 = 64 :=
by sorry

end lara_additional_miles_needed_l150_150756


namespace find_y_of_series_eq_92_l150_150703

theorem find_y_of_series_eq_92 (y : ℝ) (h : (∑' n, (2 + 5 * n) * y^n) = 92) (converge : abs y < 1) : y = 18 / 23 :=
sorry

end find_y_of_series_eq_92_l150_150703


namespace profit_percentage_of_revenues_l150_150742

theorem profit_percentage_of_revenues (R P : ℝ)
  (H1 : R > 0)
  (H2 : P > 0)
  (H3 : P * 0.98 = R * 0.098) :
  (P / R) * 100 = 10 := by
  sorry

end profit_percentage_of_revenues_l150_150742


namespace percent_employed_females_l150_150035

theorem percent_employed_females (percent_employed : ℝ) (percent_employed_males : ℝ) :
  percent_employed = 0.64 →
  percent_employed_males = 0.55 →
  (percent_employed - percent_employed_males) / percent_employed * 100 = 14.0625 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end percent_employed_females_l150_150035


namespace estimate_diff_and_prod_l150_150809

variable {x y : ℝ}
variable (hx : x > y) (hy : y > 0)

theorem estimate_diff_and_prod :
  (1.1*x) - (y - 2) = (x - y) + 0.1 * x + 2 ∧ (1.1 * x) * (y - 2) = 1.1 * (x * y) - 2.2 * x :=
by 
  sorry -- Proof details go here

end estimate_diff_and_prod_l150_150809


namespace delta_value_l150_150443

theorem delta_value (Δ : ℤ) : 5 * (-3) = Δ - 3 → Δ = -12 :=
by
  sorry

end delta_value_l150_150443


namespace gcd_is_12_l150_150127

noncomputable def gcd_problem (b : ℤ) : Prop :=
  b % 2027 = 0 → Int.gcd (b^2 + 7*b + 18) (b + 6) = 12

-- Now, let's state the theorem
theorem gcd_is_12 (b : ℤ) : gcd_problem b :=
  sorry

end gcd_is_12_l150_150127


namespace university_committee_l150_150789

open Finset

theorem university_committee (male_count female_count department_count total_committee_count physics_count rcb_count : ℕ)
  (h_male : male_count = 3) (h_female : female_count = 3) (h_department : department_count = 3)
  (h_total : total_committee_count = 7) (h_physics : physics_count = 3) 
  (h_rcb : rcb_count = 4)
  (h_distributed : rcb_count % 2 = 0) :
  (3.choose 2 * 3.choose 1 * 3.choose 1 * 3.choose 1 * 3.choose 1 * 3.choose 1 + 3.choose 3 * 3.choose 1 * 3.choose 1 = 738) :=
by
  rw [h_male, h_female, h_department, h_total, h_physics, h_rcb, h_distributed]
  iterate 2 { rw binomial_eq_choose }
  sorry

end university_committee_l150_150789


namespace solve_for_x_l150_150134

theorem solve_for_x (x : ℕ) (h : (1 / 8) * 2 ^ 36 = 8 ^ x) : x = 11 :=
by
sorry

end solve_for_x_l150_150134


namespace basketball_prob_l150_150359

theorem basketball_prob :
  let P_A := 0.7
  let P_B := 0.6
  P_A * P_B = 0.88 := 
by 
  sorry

end basketball_prob_l150_150359


namespace total_marks_l150_150223

variable (marks_in_music marks_in_maths marks_in_arts marks_in_social_studies : ℕ)

def marks_conditions : Prop :=
  marks_in_maths = marks_in_music - (1/10) * marks_in_music ∧
  marks_in_maths = marks_in_arts - 20 ∧
  marks_in_social_studies = marks_in_music + 10 ∧
  marks_in_music = 70

theorem total_marks 
  (h : marks_conditions marks_in_music marks_in_maths marks_in_arts marks_in_social_studies) :
  marks_in_music + marks_in_maths + marks_in_arts + marks_in_social_studies = 296 :=
by
  sorry

end total_marks_l150_150223


namespace track_meet_girls_short_hair_l150_150932

theorem track_meet_girls_short_hair :
  let total_people := 55
  let boys := 30
  let girls := total_people - boys
  let girls_long_hair := (3 / 5 : ℚ) * girls
  let girls_short_hair := girls - girls_long_hair
  girls_short_hair = 10 :=
by
  let total_people := 55
  let boys := 30
  let girls := total_people - boys
  let girls_long_hair := (3 / 5 : ℚ) * girls
  let girls_short_hair := girls - girls_long_hair
  sorry

end track_meet_girls_short_hair_l150_150932


namespace proof_problem_l150_150852

variable (y θ Q : ℝ)

-- Given condition
def condition : Prop := 5 * (3 * y + 7 * Real.sin θ) = Q

-- Goal to be proved
def goal : Prop := 15 * (9 * y + 21 * Real.sin θ) = 9 * Q

theorem proof_problem (h : condition y θ Q) : goal y θ Q :=
by
  sorry

end proof_problem_l150_150852


namespace problem_statement_l150_150837

noncomputable def smallest_x : ℝ :=
  -8 - (Real.sqrt 292 / 2)

theorem problem_statement (x : ℝ) :
  (15 * x ^ 2 - 40 * x + 18) / (4 * x - 3) + 4 * x = 8 * x - 3 ↔ x = smallest_x :=
by
  sorry

end problem_statement_l150_150837


namespace johns_total_amount_l150_150608

def amount_from_grandpa : ℕ := 30
def multiplier : ℕ := 3
def amount_from_grandma : ℕ := amount_from_grandpa * multiplier
def total_amount : ℕ := amount_from_grandpa + amount_from_grandma

theorem johns_total_amount :
  total_amount = 120 :=
by
  sorry

end johns_total_amount_l150_150608


namespace sum_of_xy_is_1289_l150_150604

-- Define the variables and conditions
def internal_angle1 (x y : ℕ) : ℕ := 5 * x + 3 * y
def internal_angle2 (x y : ℕ) : ℕ := 3 * x + 20
def internal_angle3 (x y : ℕ) : ℕ := 10 * y + 30

-- Definition of the sum of angles of a triangle
def sum_of_angles (x y : ℕ) : ℕ := internal_angle1 x y + internal_angle2 x y + internal_angle3 x y

-- Define the theorem statement
theorem sum_of_xy_is_1289 (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h : sum_of_angles x y = 180) : x + y = 1289 :=
by sorry

end sum_of_xy_is_1289_l150_150604


namespace range_of_m_l150_150716

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x : ℝ, f x = x^2 + 4 * x + 5)
  (h2 : ∀ x : ℝ, f (-2 + x) = f (-2 - x))
  (h3 : ∀ x : ℝ, m ≤ x ∧ x ≤ 0 → 1 ≤ f x ∧ f x ≤ 5)
  : -4 ≤ m ∧ m ≤ -2 :=
  sorry

end range_of_m_l150_150716


namespace digit_pairs_for_divisibility_by_36_l150_150174

theorem digit_pairs_for_divisibility_by_36 (A B : ℕ) :
  (0 ≤ A) ∧ (A ≤ 9) ∧ (0 ≤ B) ∧ (B ≤ 9) ∧
  (∃ k4 k9 : ℕ, (10 * 5 + B = 4 * k4) ∧ (20 + A + B = 9 * k9)) ↔ 
  ((A = 5 ∧ B = 2) ∨ (A = 1 ∧ B = 6)) :=
by sorry

end digit_pairs_for_divisibility_by_36_l150_150174


namespace dishonest_dealer_profit_l150_150810

theorem dishonest_dealer_profit (cost_weight actual_weight : ℝ) (kg_in_g : ℝ) 
  (h1 : cost_weight = 1000) (h2 : actual_weight = 920) (h3 : kg_in_g = 1000) :
  ((cost_weight - actual_weight) / actual_weight) * 100 = 8.7 := by
  sorry

end dishonest_dealer_profit_l150_150810


namespace intersection_point_l150_150008

theorem intersection_point (k : ℚ) :
  (∃ x y : ℚ, x + k * y = 0 ∧ 2 * x + 3 * y + 8 = 0 ∧ x - y - 1 = 0) ↔ (k = -1/2) :=
by sorry

end intersection_point_l150_150008


namespace angle_ne_iff_cos2angle_ne_l150_150740

theorem angle_ne_iff_cos2angle_ne (A B : ℝ) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) :
  (A ≠ B) ↔ (Real.cos (2 * A) ≠ Real.cos (2 * B)) :=
sorry

end angle_ne_iff_cos2angle_ne_l150_150740


namespace minimum_n_minus_m_abs_l150_150126

theorem minimum_n_minus_m_abs (f g : ℝ → ℝ)
  (hf : ∀ x, f x = Real.exp x + 2 * x)
  (hg : ∀ x, g x = 4 * x)
  (m n : ℝ)
  (h_cond : f m = g n) :
  |n - m| = (1 / 2) - (1 / 2) * Real.log 2 := 
sorry

end minimum_n_minus_m_abs_l150_150126


namespace pages_used_l150_150554

variable (n o c : ℕ)

theorem pages_used (h_n : n = 3) (h_o : o = 13) (h_c : c = 8) :
  (n + o) / c = 2 :=
  by
    sorry

end pages_used_l150_150554


namespace apples_per_slice_is_two_l150_150551

def number_of_apples_per_slice (total_apples : ℕ) (total_pies : ℕ) (slices_per_pie : ℕ) : ℕ :=
  total_apples / total_pies / slices_per_pie

theorem apples_per_slice_is_two (total_apples : ℕ) (total_pies : ℕ) (slices_per_pie : ℕ) :
  total_apples = 48 → total_pies = 4 → slices_per_pie = 6 → number_of_apples_per_slice total_apples total_pies slices_per_pie = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end apples_per_slice_is_two_l150_150551


namespace rainfall_on_first_day_l150_150355

theorem rainfall_on_first_day (R1 R2 R3 : ℕ) 
  (hR2 : R2 = 34)
  (hR3 : R3 = R2 - 12)
  (hTotal : R1 + R2 + R3 = 82) : 
  R1 = 26 := by
  sorry

end rainfall_on_first_day_l150_150355


namespace recommended_cooking_time_is_5_minutes_l150_150975

-- Define the conditions
def time_cooked := 45 -- seconds
def time_remaining := 255 -- seconds

-- Define the total cooking time in seconds
def total_time_seconds := time_cooked + time_remaining

-- Define the conversion from seconds to minutes
def to_minutes (seconds : ℕ) : ℕ := seconds / 60

-- The main theorem to prove
theorem recommended_cooking_time_is_5_minutes :
  to_minutes total_time_seconds = 5 :=
by
  sorry

end recommended_cooking_time_is_5_minutes_l150_150975


namespace factorize1_factorize2_factorize3_factorize4_l150_150238

-- 1. Factorize 3x - 12x^3
theorem factorize1 (x : ℝ) : 3 * x - 12 * x^3 = 3 * x * (1 - 2 * x) * (1 + 2 * x) := 
sorry

-- 2. Factorize 9m^2 - 4n^2
theorem factorize2 (m n : ℝ) : 9 * m^2 - 4 * n^2 = (3 * m + 2 * n) * (3 * m - 2 * n) := 
sorry

-- 3. Factorize a^2(x - y) + b^2(y - x)
theorem factorize3 (a b x y : ℝ) : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a + b) * (a - b) := 
sorry

-- 4. Factorize x^2 - 4xy + 4y^2 - 1
theorem factorize4 (x y : ℝ) : x^2 - 4 * x * y + 4 * y^2 - 1 = (x - y + 1) * (x - y - 1) := 
sorry

end factorize1_factorize2_factorize3_factorize4_l150_150238


namespace miles_driven_on_tuesday_l150_150094

-- Define the conditions given in the problem
theorem miles_driven_on_tuesday (T : ℕ) (h_avg : (12 + T + 21) / 3 = 17) :
  T = 18 :=
by
  -- We state what we want to prove, but we leave the proof with sorry
  sorry

end miles_driven_on_tuesday_l150_150094


namespace probability_sequence_l150_150069

def total_cards := 52
def first_card_is_six_of_diamonds := 1 / total_cards
def remaining_cards := total_cards - 1
def second_card_is_queen_of_hearts (first_card_was_six_of_diamonds : Prop) := 1 / remaining_cards
def probability_six_of_diamonds_and_queen_of_hearts : ℝ :=
  first_card_is_six_of_diamonds * second_card_is_queen_of_hearts sorry

theorem probability_sequence : 
  probability_six_of_diamonds_and_queen_of_hearts = 1 / 2652 := sorry

end probability_sequence_l150_150069


namespace hyperbola_eccentricity_b_value_l150_150016

theorem hyperbola_eccentricity_b_value (b : ℝ) (a : ℝ) (e : ℝ) 
  (h1 : a^2 = 1) (h2 : e = 2) 
  (h3 : b > 0) (h4 : b^2 = 4 - 1) : 
  b = Real.sqrt 3 := 
by 
  sorry

end hyperbola_eccentricity_b_value_l150_150016


namespace find_t_of_decreasing_function_l150_150589

theorem find_t_of_decreasing_function 
  (f : ℝ → ℝ)
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (h_A : f 0 = 4)
  (h_B : f 3 = -2)
  (h_solution_set : ∀ x, |f (x + 1) - 1| < 3 ↔ -1 < x ∧ x < 2) :
  (1 : ℝ) = 1 :=
by
  sorry

end find_t_of_decreasing_function_l150_150589


namespace joint_purchases_popular_joint_purchases_unpopular_among_neighbors_l150_150527

-- Define the properties that need to be proven
variables (Q1 Q2 : Prop) (A1 A2 : Prop)

/-- Theorem to prove why joint purchases are popular despite risks -/
theorem joint_purchases_popular : Q1 → A1 :=
begin
  sorry -- proof not provided
end

/-- Theorem to prove why joint purchases are not popular among neighbors for groceries -/
theorem joint_purchases_unpopular_among_neighbors : Q2 → A2 :=
begin
  sorry -- proof not provided
end

end joint_purchases_popular_joint_purchases_unpopular_among_neighbors_l150_150527


namespace simplify_expression_l150_150911

variable (x : ℝ)

theorem simplify_expression : 1 - (2 - (3 - (4 - (5 - x)))) = 3 - x :=
by
  sorry

end simplify_expression_l150_150911


namespace smallest_positive_period_intervals_of_monotonic_decrease_area_of_triangle_abc_l150_150582

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * Real.cos x ^ 2 - Real.sqrt 3 

theorem smallest_positive_period (x : ℝ) : 
  y = f (-3 * x) + 1 → has_period y (π / 3) :=
sorry

theorem intervals_of_monotonic_decrease (x k : ℝ) (h : k ∈ ℤ) : 
  y = f (-3 * x) + 1 → ( (1 / 3) * k * π - π / 36 ≤ x ∧ x ≤ (1 / 3) * k * π + 5 * π / 36 ) :=
sorry

theorem area_of_triangle_abc (A a b c : ℝ) (A_acute : 0 < A ∧ A < π / 2)
  (ha : a = 7) (hbc : b + c = 13) (hbcsum : Real.sin B + Real.sin C = 13 * Real.sqrt 3 / 14)
  (hAf : f (A / 2 - π / 6) = Real.sqrt 3) :
  area_of_triangle ABC = 10 * Real.sqrt 3 :=
sorry

end smallest_positive_period_intervals_of_monotonic_decrease_area_of_triangle_abc_l150_150582


namespace swans_after_10_years_l150_150113

-- Defining the initial conditions
def initial_swans : ℕ := 15

-- Condition that the number of swans doubles every 2 years
def double_every_two_years (n t : ℕ) : ℕ := n * (2 ^ (t / 2))

-- Prove that after 10 years, the number of swans will be 480
theorem swans_after_10_years : double_every_two_years initial_swans 10 = 480 :=
by
  sorry

end swans_after_10_years_l150_150113


namespace sum_of_longest_altitudes_l150_150283

-- Define the sides of the triangle
def a : ℕ := 6
def b : ℕ := 8
def c : ℕ := 10

-- Define the sides are the longest altitudes in the right triangle
def longest_altitude1 : ℕ := a
def longest_altitude2 : ℕ := b

-- Define the main theorem to prove
theorem sum_of_longest_altitudes : longest_altitude1 + longest_altitude2 = 14 := 
by
  -- The proof goes here
  sorry

end sum_of_longest_altitudes_l150_150283


namespace greatest_possible_large_chips_l150_150500

theorem greatest_possible_large_chips (s l : ℕ) (even_prime : ℕ) (h1 : s + l = 100) (h2 : s = l + even_prime) (h3 : even_prime = 2) : l = 49 :=
by
  sorry

end greatest_possible_large_chips_l150_150500


namespace g_at_5_l150_150899

def g (x : ℝ) : ℝ := 3 * x^5 - 15 * x^4 + 30 * x^3 - 45 * x^2 + 24 * x + 50

theorem g_at_5 : g 5 = 2795 :=
by
  sorry

end g_at_5_l150_150899


namespace striped_jerseys_count_l150_150622

-- Define the cost of long-sleeved jerseys
def cost_long_sleeved := 15
-- Define the cost of striped jerseys
def cost_striped := 10
-- Define the number of long-sleeved jerseys bought
def num_long_sleeved := 4
-- Define the total amount spent
def total_spent := 80

-- Define a theorem to prove the number of striped jerseys bought
theorem striped_jerseys_count : ∃ x : ℕ, x * cost_striped = total_spent - num_long_sleeved * cost_long_sleeved ∧ x = 2 := 
by 
-- TODO: The proof steps would go here, but for this exercise, we use 'sorry' to skip the proof.
sorry

end striped_jerseys_count_l150_150622


namespace smallest_n_condition_smallest_n_value_l150_150416

theorem smallest_n_condition :
  ∃ (n : ℕ), n < 1000 ∧ (99999 % n = 0) ∧ (9999 % (n + 7) = 0) ∧ 
  ∀ m, (m < 1000 ∧ (99999 % m = 0) ∧ (9999 % (m + 7) = 0)) → n ≤ m := 
sorry

theorem smallest_n_value :
  ∃ (n : ℕ), n = 266 ∧ n < 1000 ∧ (99999 % n = 0) ∧ (9999 % (n + 7) = 0) := 
sorry

end smallest_n_condition_smallest_n_value_l150_150416


namespace geometric_sequence_first_term_l150_150119

theorem geometric_sequence_first_term (a b c : ℕ) (r : ℕ) (h1 : r = 2) (h2 : b = a * r)
  (h3 : c = b * r) (h4 : 32 = c * r) (h5 : 64 = 32 * r) :
  a = 4 :=
by sorry

end geometric_sequence_first_term_l150_150119


namespace trig_identity_product_l150_150834

theorem trig_identity_product :
  (1 + Real.cos (Real.pi / 12)) * (1 + Real.cos (5 * Real.pi / 12)) * 
  (1 + Real.cos (7 * Real.pi / 12)) * (1 + Real.cos (11 * Real.pi / 12)) = 1 / 16 :=
by
  sorry

end trig_identity_product_l150_150834


namespace zero_point_of_log_a_x_plus_x_minus_m_interval_0_1_l150_150130

theorem zero_point_of_log_a_x_plus_x_minus_m_interval_0_1
  (a m : ℝ) (h₀ : 1 < a) (h₁ : ∃ x : ℝ, 0 < x ∧ x < 1 ∧ log a x + x - m = 0) :
  m < 1 :=
begin
  sorry
end

end zero_point_of_log_a_x_plus_x_minus_m_interval_0_1_l150_150130


namespace parallelogram_area_correct_l150_150990

def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_correct (base height : ℝ) (h_base : base = 30) (h_height : height = 12) : parallelogram_area base height = 360 :=
by
  rw [h_base, h_height]
  simp [parallelogram_area]
  sorry

end parallelogram_area_correct_l150_150990


namespace find_f_1789_l150_150760

def f : ℕ → ℕ := sorry

axiom f_1 : f 1 = 5
axiom f_f_n : ∀ n, f (f n) = 4 * n + 9
axiom f_2n : ∀ n, f (2 * n) = (2 * n) + 1 + 3

theorem find_f_1789 : f 1789 = 3581 :=
by
  sorry

end find_f_1789_l150_150760


namespace part_one_part_two_l150_150583

def f (a x : ℝ) : ℝ := |a - 4 * x| + |2 * a + x|

theorem part_one (x : ℝ) : f 1 x ≥ 3 ↔ x ≤ 0 ∨ x ≥ 2 / 5 := 
sorry

theorem part_two (a x : ℝ) : f a x + f a (-1 / x) ≥ 10 := 
sorry

end part_one_part_two_l150_150583


namespace domain_of_f_range_of_f_monotonic_increasing_interval_of_f_l150_150244

open Real

noncomputable def f (x : ℝ) : ℝ := log (9 - x^2)

theorem domain_of_f : Set.Ioo (-3 : ℝ) 3 = {x : ℝ | -3 < x ∧ x < 3} :=
by
  sorry

theorem range_of_f : ∃ y : ℝ, y ∈ Set.Iic (2 * log 3) :=
by
  sorry

theorem monotonic_increasing_interval_of_f : 
  {x : ℝ | -3 < x} ∩ {x : ℝ | 0 ≥ x} = Set.Ioc (-3 : ℝ) 0 :=
by
  sorry

end domain_of_f_range_of_f_monotonic_increasing_interval_of_f_l150_150244


namespace john_total_amount_l150_150612

-- Given conditions from a)
def grandpa_amount : ℕ := 30
def grandma_amount : ℕ := 3 * grandpa_amount

-- Problem statement
theorem john_total_amount : grandpa_amount + grandma_amount = 120 :=
by
  sorry

end john_total_amount_l150_150612


namespace problem_statement_l150_150254

-- Definitions and conditions
def f (x : ℝ) : ℝ := x

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (2 * a - x) = f x

-- Given the specific condition
def f_symmetric_about_1 : Prop := is_symmetric_about f 1

-- We need to prove that this implies g(x) = 3x - 2
def g (x : ℝ) : ℝ := 3 * x - 2

theorem problem_statement : f_symmetric_about_1 → ∀ x, g x = 3 * x - 2 := 
by
  intro h
  sorry -- Detailed proof is omitted

end problem_statement_l150_150254


namespace find_f_of_2_l150_150529

noncomputable def f (x : ℝ) : ℝ := 
if x < 0 then x^3 + x^2 else 0

theorem find_f_of_2 :
  (∀ x : ℝ, f (-x) = -f x) → (∀ x : ℝ, x < 0 → f x = x^3 + x^2) → f 2 = 4 :=
by
  intros h_odd h_def_neg
  sorry

end find_f_of_2_l150_150529


namespace equation_of_line_l_l150_150792

theorem equation_of_line_l (P : ℝ × ℝ) (hP : P = (1, -1)) (θ₁ θ₂ : ℕ) (hθ₁ : θ₁ = 45) (hθ₂ : θ₂ = θ₁ * 2) (hθ₂_90 : θ₂ = 90) : 
  ∃ l : ℝ → ℝ, (∀ x, l x = l (P.fst)) := 
sorry

end equation_of_line_l_l150_150792


namespace students_who_like_both_apple_pie_and_chocolate_cake_l150_150743

def total_students := 50
def students_who_like_apple_pie := 22
def students_who_like_chocolate_cake := 20
def students_who_like_neither := 10
def students_who_like_only_cookies := 5

theorem students_who_like_both_apple_pie_and_chocolate_cake :
  (students_who_like_apple_pie + students_who_like_chocolate_cake - (total_students - students_who_like_neither - students_who_like_only_cookies)) = 7 := 
by
  sorry

end students_who_like_both_apple_pie_and_chocolate_cake_l150_150743


namespace slices_with_both_toppings_l150_150081

-- Definitions and conditions directly from the problem statement
def total_slices : ℕ := 24
def pepperoni_slices : ℕ := 15
def mushroom_slices : ℕ := 14

-- Theorem proving the number of slices with both toppings
theorem slices_with_both_toppings :
  (∃ n : ℕ, n + (pepperoni_slices - n) + (mushroom_slices - n) = total_slices) → ∃ n : ℕ, n = 5 := 
by 
  sorry

end slices_with_both_toppings_l150_150081


namespace cover_with_L_shapes_l150_150980

def L_shaped (m n : ℕ) : Prop :=
  m > 1 ∧ n > 1 ∧ ∃ k, m * n = 8 * k -- Conditions and tiling pattern coverage.

-- Problem statement as a theorem
theorem cover_with_L_shapes (m n : ℕ) (h1 : m > 1) (h2 : n > 1) : (∃ k, m * n = 8 * k) ↔ L_shaped m n :=
-- Placeholder for the proof
sorry

end cover_with_L_shapes_l150_150980


namespace inequality_proof_l150_150872

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : ¬ (a + d > b + c) := sorry

end inequality_proof_l150_150872


namespace solution_l150_150871

noncomputable def problem (x : ℕ) : Prop :=
  2 ^ 28 = 4 ^ x  -- Simplified form of the condition given

theorem solution : problem 14 :=
by
  sorry

end solution_l150_150871


namespace certain_number_less_32_l150_150959

theorem certain_number_less_32 (x : ℤ) (h : x - 48 = 22) : x - 32 = 38 :=
by
  sorry

end certain_number_less_32_l150_150959


namespace probability_no_physics_and_chemistry_l150_150918

-- Define the probabilities for the conditions
def P_physics : ℚ := 5 / 8
def P_no_physics : ℚ := 1 - P_physics
def P_chemistry_given_no_physics : ℚ := 2 / 3

-- Define the theorem we want to prove
theorem probability_no_physics_and_chemistry :
  P_no_physics * P_chemistry_given_no_physics = 1 / 4 :=
by sorry

end probability_no_physics_and_chemistry_l150_150918


namespace max_fraction_l150_150129

theorem max_fraction (x y : ℝ) (h1 : -6 ≤ x) (h2 : x ≤ -3) (h3 : 3 ≤ y) (h4 : y ≤ 5) :
  (∀ x y, -6 ≤ x → x ≤ -3 → 3 ≤ y → y ≤ 5 → (x - y) / y ≤ -2) :=
by
  sorry

end max_fraction_l150_150129


namespace area_triangle_CMB_eq_105_l150_150205

noncomputable def area_of_triangle (C M B : ℝ × ℝ) : ℝ :=
  0.5 * (M.1 * B.2 - M.2 * B.1)

theorem area_triangle_CMB_eq_105 :
  let C : ℝ × ℝ := (0, 0)
  let M : ℝ × ℝ := (10, 0)
  let B : ℝ × ℝ := (10, 21)
  area_of_triangle C M B = 105 := by
  sorry

end area_triangle_CMB_eq_105_l150_150205


namespace marbles_in_bag_l150_150326

theorem marbles_in_bag (r b : ℕ) : 
  (r - 2) * 10 = (r + b - 2) →
  (r * 6 = (r + b - 3)) →
  ((r - 2) * 8 = (r + b - 4)) →
  r + b = 42 :=
by
  intros h1 h2 h3
  sorry

end marbles_in_bag_l150_150326


namespace number_of_intersection_points_l150_150025

-- Define the standard parabola y = x^2
def parabola (x : ℝ) : ℝ := x^2

-- Define what it means for a line to be tangent to the parabola
def is_tangent (m : ℝ) (c : ℝ) : Prop :=
  ∃ x0 : ℝ, parabola x0 = m * x0 + c ∧ 2 * x0 = m

-- Define what it means for a line to intersect the parabola
def line_intersects_parabola (m : ℝ) (c : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, parabola x1 = m * x1 + c ∧ parabola x2 = m * x2 + c

-- Main theorem statement
theorem number_of_intersection_points :
  (∃ m c : ℝ, is_tangent m c) → (∃ m' c' : ℝ, line_intersects_parabola m' c') →
  ∃ n : ℕ, n = 1 ∨ n = 2 ∨ n = 3 :=
sorry

end number_of_intersection_points_l150_150025


namespace intersection_correct_l150_150733

def M : Set Int := {-1, 1, 3, 5}
def N : Set Int := {-3, 1, 5}

theorem intersection_correct : M ∩ N = {1, 5} := 
by 
    sorry

end intersection_correct_l150_150733


namespace max_height_l150_150818

-- Given definitions
def height_eq (t : ℝ) : ℝ := -16 * t^2 + 64 * t + 10

def max_height_problem : Prop :=
  ∃ t : ℝ, height_eq t = 74 ∧ ∀ t' : ℝ, height_eq t' ≤ height_eq t

-- Statement of the proof
theorem max_height : max_height_problem := sorry

end max_height_l150_150818


namespace work_completion_days_l150_150894

-- Define the work rates
def john_work_rate : ℚ := 1/8
def rose_work_rate : ℚ := 1/16
def dave_work_rate : ℚ := 1/12

-- Define the combined work rate
def combined_work_rate : ℚ := john_work_rate + rose_work_rate + dave_work_rate

-- Define the required number of days to complete the work together
def days_to_complete_work : ℚ := 1 / combined_work_rate

-- Prove that the total number of days required to complete the work is 48/13
theorem work_completion_days : days_to_complete_work = 48 / 13 :=
by 
  -- Here is where the actual proof would be, but it is not needed as per instructions
  sorry

end work_completion_days_l150_150894


namespace simplify_expression_l150_150399

variable (x : ℝ) (hx : x ≠ 0)

theorem simplify_expression : 
  ( (x + 3)^2 + (x + 3) * (x - 3) ) / (2 * x) = x + 3 := by
  sorry

end simplify_expression_l150_150399


namespace number_of_subcommittees_l150_150032

theorem number_of_subcommittees :
  ∃ (k : ℕ), ∀ (num_people num_sub_subcommittees subcommittee_size : ℕ), 
  num_people = 360 → 
  num_sub_subcommittees = 3 → 
  subcommittee_size = 6 → 
  k = (num_people * num_sub_subcommittees) / subcommittee_size :=
sorry

end number_of_subcommittees_l150_150032


namespace solution_set_nonempty_implies_a_range_l150_150498

theorem solution_set_nonempty_implies_a_range (a : ℝ) :
  (∃ x : ℝ, x^2 + a * x + 4 < 0) ↔ (a < -4 ∨ a > 4) :=
by
  sorry

end solution_set_nonempty_implies_a_range_l150_150498


namespace sum_of_x_and_y_l150_150626

theorem sum_of_x_and_y (x y : ℝ) 
  (h1 : (x - 1) ^ 3 + 1997 * (x - 1) = -1)
  (h2 : (y - 1) ^ 3 + 1997 * (y - 1) = 1) : 
  x + y = 2 :=
by
  sorry

end sum_of_x_and_y_l150_150626


namespace roots_square_sum_l150_150955

theorem roots_square_sum (a b : ℝ) 
  (h1 : a^2 - 4 * a + 4 = 0) 
  (h2 : b^2 - 4 * b + 4 = 0) 
  (h3 : a = b) :
  a^2 + b^2 = 8 := 
sorry

end roots_square_sum_l150_150955


namespace parabola_focus_coords_l150_150781

theorem parabola_focus_coords :
  ∀ (x y : ℝ), y^2 = -4 * x → (x, y) = (-1, 0) :=
by
  intros x y h
  sorry

end parabola_focus_coords_l150_150781


namespace y_intercept_l150_150242

theorem y_intercept : ∀ (x y : ℝ), 4 * x + 7 * y = 28 → (0, 4) = (0, y) :=
by
  intros x y h
  sorry

end y_intercept_l150_150242


namespace log_ride_cost_l150_150394

noncomputable def cost_of_log_ride (ferris_wheel : ℕ) (roller_coaster : ℕ) (initial_tickets : ℕ) (additional_tickets : ℕ) : ℕ :=
  let total_needed := initial_tickets + additional_tickets
  let total_known := ferris_wheel + roller_coaster
  total_needed - total_known

theorem log_ride_cost :
  cost_of_log_ride 6 5 2 16 = 7 :=
by
  -- specify the values for ferris_wheel, roller_coaster, initial_tickets, additional_tickets
  let ferris_wheel := 6
  let roller_coaster := 5
  let initial_tickets := 2
  let additional_tickets := 16
  -- calculate the cost of the log ride
  let total_needed := initial_tickets + additional_tickets
  let total_known := ferris_wheel + roller_coaster
  let log_ride := total_needed - total_known
  -- assert that the cost of the log ride is 7
  have : log_ride = 7 := by
    -- use arithmetic to justify the answer
    sorry
  exact this

end log_ride_cost_l150_150394


namespace expression_positive_l150_150164

variable {a b c : ℝ}

theorem expression_positive (h₀ : 0 < a ∧ a < 2) (h₁ : -2 < b ∧ b < 0) : 0 < b + a^2 :=
by
  sorry

end expression_positive_l150_150164


namespace abc_product_l150_150774

/-- Given a b c + a b + b c + a c + a + b + c = 164 -/
theorem abc_product :
  ∃ (a b c : ℕ), a * b * c + a * b + b * c + a * c + a + b + c = 164 ∧ a * b * c = 80 :=
by
  sorry

end abc_product_l150_150774


namespace area_contained_by_graph_l150_150564

theorem area_contained_by_graph (x y : ℝ) :
  (|x + y| + |x - y| ≤ 6) → (36 = 36) := by
  sorry

end area_contained_by_graph_l150_150564


namespace negate_proposition_l150_150732

theorem negate_proposition :
  (¬ (∀ x : ℝ, x > 1 → x^2 + x + 1 > 0)) ↔ (∃ x : ℝ, x > 1 ∧ x^2 + x + 1 ≤ 0) := by
  sorry

end negate_proposition_l150_150732


namespace fx_solution_l150_150289

theorem fx_solution (f : ℝ → ℝ) (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1)
  (h_assumption : f (1 / x) = x / (1 - x)) : f x = 1 / (x - 1) :=
by
  sorry

end fx_solution_l150_150289


namespace trajectory_of_M_l150_150517

-- Define the conditions: P moves on the circle, and Q is fixed
variable (P Q M : ℝ × ℝ)
variable (P_moves_on_circle : P.1^2 + P.2^2 = 1)
variable (Q_fixed : Q = (3, 0))
variable (M_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2))

-- Theorem statement
theorem trajectory_of_M :
  (2 * M.1 - 3)^2 + 4 * M.2^2 = 1 :=
sorry

end trajectory_of_M_l150_150517


namespace problem_a_problem_b_problem_c_l150_150393

noncomputable def probability_without_replacement : ℚ :=
  (6 * 5 * 4) / (21 * 20 * 19)

noncomputable def probability_with_replacement : ℚ :=
  (6 * 6 * 6) / (21 * 21 * 21)

noncomputable def probability_simultaneous_draw : ℚ :=
  (Nat.choose 6 3) / (Nat.choose 21 3)

theorem problem_a : probability_without_replacement = 2 / 133 := by
  sorry

theorem problem_b : probability_with_replacement = 8 / 343 := by
  sorry

theorem problem_c : probability_simultaneous_draw = 2 / 133 := by
  sorry

end problem_a_problem_b_problem_c_l150_150393


namespace area_of_triangle_l150_150268

theorem area_of_triangle {m : ℝ} 
  (h₁ : ∃ A B : ℝ × ℝ, (∃ C : ℝ × ℝ, C = (1, 0) ∧ 
           ((A.1 - 1)^2 + A.2^2 = 4 ∧ 
            (B.1 - 1)^2 + B.2^2 = 4 ∧ 
            (A.1 - m * A.2 + 1 = 0) ∧ 
            (B.1 - m * B.2 + 1 = 0))))
  (h₂ : 2 * 2 * real.sin (real.arcsin (4 / 5)) = 8 / 5) :
  m = 2 := 
sorry

end area_of_triangle_l150_150268


namespace g_of_f_three_l150_150037

def f (x : ℤ) : ℤ := x^3 - 2
def g (x : ℤ) : ℤ := 3*x^2 + 3*x + 2

theorem g_of_f_three : g (f 3) = 1952 := by
  sorry

end g_of_f_three_l150_150037


namespace right_triangle_property_l150_150885

theorem right_triangle_property
  (a b c x : ℝ)
  (h1 : c^2 = a^2 + b^2)
  (h2 : 1/2 * a * b = 1/2 * c * x)
  : 1/x^2 = 1/a^2 + 1/b^2 :=
sorry

end right_triangle_property_l150_150885


namespace squares_characterization_l150_150366

theorem squares_characterization (n : ℕ) (a b : ℤ) (h_cond : n + 1 = a^2 + (a + 1)^2 ∧ n + 1 = b^2 + 2 * (b + 1)^2) :
  ∃ k l : ℤ, 2 * n + 1 = k^2 ∧ 3 * n + 1 = l^2 :=
sorry

end squares_characterization_l150_150366


namespace sum_of_obtuse_angles_l150_150024

theorem sum_of_obtuse_angles (A B : ℝ) (hA1 : A > π / 2) (hA2 : A < π)
  (hB1 : B > π / 2) (hB2 : B < π)
  (hSinA : Real.sin A = Real.sqrt 5 / 5)
  (hSinB : Real.sin B = Real.sqrt 10 / 10) :
  A + B = 7 * π / 4 := 
sorry

end sum_of_obtuse_angles_l150_150024


namespace arithmetic_seq_a1_l150_150306

theorem arithmetic_seq_a1 (a_1 d : ℝ) (h1 : a_1 + 4 * d = 9) (h2 : 2 * (a_1 + 2 * d) = (a_1 + d) + 6) : a_1 = -3 := by
  sorry

end arithmetic_seq_a1_l150_150306


namespace triangle_ratio_l150_150027

theorem triangle_ratio (A B C G H P : Type)
  [inst: Nonempty A] [inst: Nonempty B] [inst: Nonempty C]
  [inst: Nonempty G] [inst: Nonempty H] [inst: Nonempty P]
  (on_line_AB : G ∈ [A, B])
  (on_line_BC : H ∈ [B, C])
  (intersect : (AG ∩ CH).P)
  (AP_PG_eq : ratio (length A P) (length P G) = 5 : ℝ)
  (CP_PH_eq : ratio (length C P) (length P H) = 3 : ℝ) :
  ratio (length B H) (length H C) = 3 / 7 := 
sorry

end triangle_ratio_l150_150027


namespace Z_is_normal_dist_XZ_dist_YZ_dist_X_plus_Z_cov_X_Z_l150_150761

-- Definitions of the given conditions
def X : Distribution := Normal 0 1
def Y : Distribution := uniform [{-1, 1}]
def Z (X Y : Distribution) := X * Y

-- Proof that Z follows a normal distribution
theorem Z_is_normal : Law (Z X Y) = Normal 0 1 := sorry

-- Distribution of the vector (X, Z)
theorem dist_XZ (x z : ℝ) : P(X ≤ x ∧ Z ≤ z) = 
  (Φ (min x z) / 2) + ((Φ x - Φ (-z)) / 2) * (if x ≥ -z then 1 else 0) := sorry

-- Distribution of the vector (Y, Z)
theorem dist_YZ : ∀ y ∈ {-1, 1}, Law (Z X (Y = y)) = Normal 0 1 := sorry

-- Distribution of the random variable X + Z
theorem dist_X_plus_Z (x : ℝ) : P(X + Z ≤ x) = (Φ (x / 2) + if x ≥ 0 then 1 else 0) / 2 := sorry

-- Proving that X and Z are uncorrelated
theorem cov_X_Z : cov X Z = 0 := sorry

end Z_is_normal_dist_XZ_dist_YZ_dist_X_plus_Z_cov_X_Z_l150_150761


namespace range_of_m_l150_150275

theorem range_of_m :
  ∀ m, (∀ x, m ≤ x ∧ x ≤ 4 → (0 ≤ -x^2 + 4*x ∧ -x^2 + 4*x ≤ 4)) ↔ (0 ≤ m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l150_150275


namespace no_real_k_for_distinct_roots_l150_150415

theorem no_real_k_for_distinct_roots (k : ℝ) : ¬ ( -8 * k^2 > 0 ) := 
by
  sorry

end no_real_k_for_distinct_roots_l150_150415


namespace brainiacs_like_neither_l150_150820

variables 
  (total : ℕ) -- Total number of brainiacs.
  (R : ℕ) -- Number of brainiacs who like rebus teasers.
  (M : ℕ) -- Number of brainiacs who like math teasers.
  (both : ℕ) -- Number of brainiacs who like both rebus and math teasers.
  (math_only : ℕ) -- Number of brainiacs who like only math teasers.

-- Given conditions in the problem
def twice_as_many_rebus : Prop := R = 2 * M
def both_teasers : Prop := both = 18
def math_teasers_not_rebus : Prop := math_only = 20
def total_brainiacs : Prop := total = 100

noncomputable def exclusion_inclusion : ℕ := R + M - both

-- Proof statement: The number of brainiacs who like neither rebus nor math teasers totals to 4
theorem brainiacs_like_neither
  (h_total : total_brainiacs total)
  (h_twice : twice_as_many_rebus R M)
  (h_both : both_teasers both)
  (h_math_only : math_teasers_not_rebus math_only)
  (h_M : M = both + math_only) :
  total - exclusion_inclusion R M both = 4 :=
sorry

end brainiacs_like_neither_l150_150820


namespace find_ivans_number_l150_150121

theorem find_ivans_number :
  ∃ (a b c d e f g h i j k l : ℕ),
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    10 ≤ c ∧ c < 100 ∧
    10 ≤ d ∧ d < 100 ∧
    1000 ≤ e ∧ e < 10000 ∧
    (a * 10^10 + b * 10^8 + c * 10^6 + d * 10^4 + e) = 132040530321 := sorry

end find_ivans_number_l150_150121


namespace evaluate_heartsuit_l150_150981

-- Define the given operation
def heartsuit (x y : ℝ) : ℝ := abs (x - y)

-- State the proof problem in Lean
theorem evaluate_heartsuit (a b : ℝ) (h_a : a = 3) (h_b : b = -1) :
  heartsuit (heartsuit a b) (heartsuit (2 * a) (2 * b)) = 4 :=
by
  -- acknowledging that it's correct without providing the solution steps
  sorry

end evaluate_heartsuit_l150_150981


namespace correct_option_C_l150_150808

theorem correct_option_C (a b c : ℝ) : 2 * a^2 * b * c - a^2 * b * c = a^2 * b * c := 
sorry

end correct_option_C_l150_150808


namespace monotonicity_of_f_abs_f_diff_ge_four_abs_diff_l150_150858

noncomputable def f (a x : ℝ) : ℝ := (a + 1) * Real.log x + a * x^2 + 1

theorem monotonicity_of_f {a : ℝ} (x : ℝ) (hx : 0 < x) :
  (f a x) = (f a x) := sorry

theorem abs_f_diff_ge_four_abs_diff {a x1 x2: ℝ} (ha : a ≤ -2) (hx1 : 0 < x1) (hx2 : 0 < x2) :
  |f a x1 - f a x2| ≥ 4 * |x1 - x2| := sorry

end monotonicity_of_f_abs_f_diff_ge_four_abs_diff_l150_150858


namespace total_surfers_l150_150354

theorem total_surfers (num_surfs_santa_monica : ℝ) (ratio_malibu : ℝ) (ratio_santa_monica : ℝ) (ratio_venice : ℝ) (ratio_huntington : ℝ) (ratio_newport : ℝ) :
    num_surfs_santa_monica = 36 ∧ ratio_malibu = 7 ∧ ratio_santa_monica = 4.5 ∧ ratio_venice = 3.5 ∧ ratio_huntington = 2 ∧ ratio_newport = 1.5 →
    (ratio_malibu * (num_surfs_santa_monica / ratio_santa_monica) +
     num_surfs_santa_monica +
     ratio_venice * (num_surfs_santa_monica / ratio_santa_monica) +
     ratio_huntington * (num_surfs_santa_monica / ratio_santa_monica) +
     ratio_newport * (num_surfs_santa_monica / ratio_santa_monica)) = 148 :=
by
  sorry

end total_surfers_l150_150354


namespace min_k_l150_150474

noncomputable 
def f (k : ℕ) (x : ℝ) : ℝ := 
  (Real.sin (k * x / 10)) ^ 4 + (Real.cos (k * x / 10)) ^ 4

theorem min_k (k : ℕ) 
    (h : (∀ a : ℝ, {y | ∃ x : ℝ, a < x ∧ x < a+1 ∧ y = f k x} = 
                  {y | ∃ x : ℝ, y = f k x})) 
    : k ≥ 16 :=
by
  sorry

end min_k_l150_150474


namespace edward_money_proof_l150_150408

def edward_total_money (earned_per_lawn : ℕ) (number_of_lawns : ℕ) (saved_up : ℕ) : ℕ :=
  earned_per_lawn * number_of_lawns + saved_up

theorem edward_money_proof :
  edward_total_money 8 5 7 = 47 :=
by
  sorry

end edward_money_proof_l150_150408


namespace range_of_m_l150_150856

theorem range_of_m (m : ℝ) :
  (3 * 1 - 2 + m) * (3 * 1 - 1 + m) < 0 →
  -2 < m ∧ m < -1 :=
by
  intro h
  sorry

end range_of_m_l150_150856


namespace addition_subtraction_result_l150_150812

theorem addition_subtraction_result :
  27474 + 3699 + 1985 - 2047 = 31111 :=
by {
  sorry
}

end addition_subtraction_result_l150_150812


namespace colin_speed_l150_150401

variable (B T Br C : ℝ)

def Bruce := B = 1
def Tony := T = 2 * B
def Brandon := Br = T / 3
def Colin := C = 6 * Br

theorem colin_speed : Bruce B → Tony B T → Brandon T Br → Colin Br C → C = 4 := by
  sorry

end colin_speed_l150_150401


namespace fraction_of_B_amount_equals_third_of_A_amount_l150_150368

variable (A B : ℝ)
variable (x : ℝ)

theorem fraction_of_B_amount_equals_third_of_A_amount
  (h1 : A + B = 1210)
  (h2 : B = 484)
  (h3 : (1 / 3) * A = x * B) : 
  x = 1 / 2 :=
sorry

end fraction_of_B_amount_equals_third_of_A_amount_l150_150368


namespace simplify_expression_l150_150518

variable {x y z : ℝ} 
variable (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0)

theorem simplify_expression :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (x * y * z)⁻¹ * (x + y + z)⁻¹ :=
sorry

end simplify_expression_l150_150518


namespace decreasing_interval_maximum_on_interval_l150_150019

open Real

-- Definition of the function f: ℝ → ℝ
noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- 1. Prove that f(x) is decreasing on (0, 2)
theorem decreasing_interval :
  ∀ x ∈ Ioo 0 2, deriv f x < 0 :=
by
  sorry

-- 2. Prove that the maximum value of f(x) on [-4, 3] is 0
theorem maximum_on_interval :
  ∃ y ∈ Icc (-4 : ℝ) 3, ∀ z ∈ Icc (-4 : ℝ) 3, f z ≤ f y ∧ f y = 0 :=
by
  sorry

end decreasing_interval_maximum_on_interval_l150_150019


namespace negation_of_proposition_l150_150917

theorem negation_of_proposition :
  ¬ (∃ x : ℝ, x ≤ 0 ∧ x^2 ≥ 0) ↔ ∀ x : ℝ, x ≤ 0 → x^2 < 0 :=
by
  sorry

end negation_of_proposition_l150_150917


namespace sqrt_nested_expression_l150_150528

theorem sqrt_nested_expression : 
  Real.sqrt (32 * Real.sqrt (16 * Real.sqrt (8 * Real.sqrt 4))) = 16 := 
by
  sorry

end sqrt_nested_expression_l150_150528


namespace least_subtracted_divisible_l150_150954

theorem least_subtracted_divisible :
  ∃ k, (5264 - 11) = 17 * k :=
by
  sorry

end least_subtracted_divisible_l150_150954


namespace avg10_students_correct_l150_150592

-- Definitions for the conditions
def avg15_students : ℝ := 70
def num15_students : ℕ := 15
def num10_students : ℕ := 10
def num25_students : ℕ := num15_students + num10_students
def avg25_students : ℝ := 80

-- Total percentage calculation based on conditions
def total_perc25_students := num25_students * avg25_students
def total_perc15_students := num15_students * avg15_students

-- The average percent of the 10 students, based on the conditions and given average for 25 students.
theorem avg10_students_correct : 
  (total_perc25_students - total_perc15_students) / (num10_students : ℝ) = 95 := by
  sorry

end avg10_students_correct_l150_150592


namespace initial_dimes_l150_150635

theorem initial_dimes (x : ℕ) (h1 : x + 7 = 16) : x = 9 := by
  sorry

end initial_dimes_l150_150635


namespace meet_at_starting_point_l150_150207

theorem meet_at_starting_point (track_length : Nat) (speed_A_kmph speed_B_kmph : Nat)
  (h_track_length : track_length = 1500)
  (h_speed_A : speed_A_kmph = 36)
  (h_speed_B : speed_B_kmph = 54) :
  let speed_A_mps := speed_A_kmph * 1000 / 3600
  let speed_B_mps := speed_B_kmph * 1000 / 3600
  let time_A := track_length / speed_A_mps
  let time_B := track_length / speed_B_mps
  let lcm_time := Nat.lcm time_A time_B
  lcm_time = 300 :=
by
  sorry

end meet_at_starting_point_l150_150207


namespace min_sum_m_n_l150_150262

open Nat

theorem min_sum_m_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : m * n - 2 * m - 3 * n - 20 = 0) : m + n = 20 :=
sorry

end min_sum_m_n_l150_150262


namespace time_to_finish_by_p_l150_150203

theorem time_to_finish_by_p (P_rate Q_rate : ℝ) (worked_together_hours remaining_job_rate : ℝ) :
    P_rate = 1/3 ∧ Q_rate = 1/9 ∧ worked_together_hours = 2 ∧ remaining_job_rate = 1 - (worked_together_hours * (P_rate + Q_rate)) → 
    (remaining_job_rate / P_rate) * 60 = 20 := 
by
  sorry

end time_to_finish_by_p_l150_150203


namespace fred_carrots_l150_150325

-- Define the conditions
def sally_carrots : Nat := 6
def total_carrots : Nat := 10

-- Define the problem question and the proof statement
theorem fred_carrots : ∃ fred_carrots : Nat, fred_carrots = total_carrots - sally_carrots := 
by
  sorry

end fred_carrots_l150_150325


namespace gretel_hansel_salary_difference_l150_150585

theorem gretel_hansel_salary_difference :
  let hansel_initial_salary := 30000
  let hansel_raise_percentage := 10
  let gretel_initial_salary := 30000
  let gretel_raise_percentage := 15
  let hansel_new_salary := hansel_initial_salary + (hansel_raise_percentage / 100 * hansel_initial_salary)
  let gretel_new_salary := gretel_initial_salary + (gretel_raise_percentage / 100 * gretel_initial_salary)
  gretel_new_salary - hansel_new_salary = 1500 := sorry

end gretel_hansel_salary_difference_l150_150585


namespace tangents_product_l150_150649

theorem tangents_product (x y : ℝ) 
  (h1 : Real.tan x - Real.tan y = 7) 
  (h2 : 2 * Real.sin (2 * (x - y)) = Real.sin (2 * x) * Real.sin (2 * y)) :
  Real.tan x * Real.tan y = -7/6 := 
sorry

end tangents_product_l150_150649


namespace porter_monthly_earnings_l150_150768

-- Definitions
def regular_daily_rate : ℝ := 8
def days_per_week : ℕ := 5
def overtime_rate : ℝ := 1.5
def tax_deduction_rate : ℝ := 0.10
def insurance_deduction_rate : ℝ := 0.05
def weeks_per_month : ℕ := 4

-- Intermediate Calculations
def regular_weekly_earnings := regular_daily_rate * days_per_week
def extra_day_rate := regular_daily_rate * overtime_rate
def total_weekly_earnings := regular_weekly_earnings + extra_day_rate
def total_monthly_earnings_before_deductions := total_weekly_earnings * weeks_per_month

-- Deductions
def tax_deduction := total_monthly_earnings_before_deductions * tax_deduction_rate
def insurance_deduction := total_monthly_earnings_before_deductions * insurance_deduction_rate
def total_deductions := tax_deduction + insurance_deduction
def total_monthly_earnings_after_deductions := total_monthly_earnings_before_deductions - total_deductions

-- Theorem Statement
theorem porter_monthly_earnings : total_monthly_earnings_after_deductions = 176.80 := by
  sorry

end porter_monthly_earnings_l150_150768


namespace solve_linear_equation_l150_150187

theorem solve_linear_equation : ∀ x : ℝ, 4 * (2 * x - 1) = 1 - 3 * (x + 2) → x = -1 / 11 :=
by
  intro x h
  -- Proof to be filled in
  sorry

end solve_linear_equation_l150_150187


namespace George_colors_combination_l150_150302

def binom (n k : ℕ) : ℕ := n.choose k

theorem George_colors_combination : binom 9 3 = 84 :=
by {
  exact Nat.choose_eq_factorial_div_factorial (le_refl 3)
}

end George_colors_combination_l150_150302


namespace count_divisible_neither_5_nor_7_below_500_l150_150180

def count_divisible_by (n k : ℕ) : ℕ := (n - 1) / k

def count_divisible_by_5_or_7_below (n : ℕ) : ℕ :=
  let count_5 := count_divisible_by n 5
  let count_7 := count_divisible_by n 7
  let count_35 := count_divisible_by n 35
  count_5 + count_7 - count_35

def count_divisible_neither_5_nor_7_below (n : ℕ) : ℕ :=
  n - 1 - count_divisible_by_5_or_7_below n

theorem count_divisible_neither_5_nor_7_below_500 : count_divisible_neither_5_nor_7_below 500 = 343 :=
by
  sorry

end count_divisible_neither_5_nor_7_below_500_l150_150180


namespace fish_population_l150_150879

theorem fish_population (N : ℕ) (h1 : 50 > 0) (h2 : N > 0) 
  (tagged_first_catch : ℕ) (total_first_catch : ℕ)
  (tagged_second_catch : ℕ) (total_second_catch : ℕ)
  (h3 : tagged_first_catch = 50)
  (h4 : total_first_catch = 50)
  (h5 : tagged_second_catch = 2)
  (h6 : total_second_catch = 50)
  (h_percent : (tagged_first_catch : ℝ) / (N : ℝ) = (tagged_second_catch : ℝ) / (total_second_catch : ℝ))
  : N = 1250 := 
  by sorry

end fish_population_l150_150879


namespace harris_spends_146_in_one_year_l150_150281

/-- Conditions: Harris feeds his dog 1 carrot per day. There are 5 carrots in a 1-pound bag. Each bag costs $2.00. There are 365 days in a year. -/
def carrots_per_day := 1
def carrots_per_bag := 5
def cost_per_bag := 2.00
def days_per_year := 365

/-- Prove that Harris will spend $146.00 on carrots in one year -/
theorem harris_spends_146_in_one_year :
  (carrots_per_day * days_per_year / carrots_per_bag) * cost_per_bag = 146.00 :=
by sorry

end harris_spends_146_in_one_year_l150_150281


namespace rachel_wrote_six_pages_l150_150163

theorem rachel_wrote_six_pages
  (write_rate : ℕ)
  (research_time : ℕ)
  (editing_time : ℕ)
  (total_time : ℕ)
  (total_time_in_minutes : ℕ := total_time * 60)
  (actual_time_writing : ℕ := total_time_in_minutes - (research_time + editing_time))
  (pages_written : ℕ := actual_time_writing / write_rate) :
  write_rate = 30 →
  research_time = 45 →
  editing_time = 75 →
  total_time = 5 →
  pages_written = 6 :=
by
  intros h1 h2 h3 h4
  subst h1
  subst h2
  subst h3
  subst h4
  have h5 : total_time_in_minutes = 300 := by sorry
  have h6 : actual_time_writing = 180 := by sorry
  have h7 : pages_written = 6 := by sorry
  exact h7

end rachel_wrote_six_pages_l150_150163


namespace spherical_coordinates_neg_z_l150_150578

theorem spherical_coordinates_neg_z (x y z : ℝ) (h₀ : ρ = 5) (h₁ : θ = 3 * Real.pi / 4) (h₂ : φ = Real.pi / 3)
  (hx : x = ρ * Real.sin φ * Real.cos θ) 
  (hy : y = ρ * Real.sin φ * Real.sin θ) 
  (hz : z = ρ * Real.cos φ) : 
  (ρ, θ, π - φ) = (5, 3 * Real.pi / 4, 2 * Real.pi / 3) :=
by
  sorry

end spherical_coordinates_neg_z_l150_150578


namespace students_shared_cost_l150_150817

theorem students_shared_cost (P n : ℕ) (h_price_range: 100 ≤ P ∧ P ≤ 120)
  (h_div1: P % n = 0) (h_div2: P % (n - 2) = 0) (h_extra_cost: P / n + 1 = P / (n - 2)) : n = 14 := by
  sorry

end students_shared_cost_l150_150817


namespace train_speed_correct_l150_150969

noncomputable def train_speed (length_meters : ℕ) (time_seconds : ℕ) : ℝ :=
  (length_meters : ℝ) / 1000 / (time_seconds / 3600)

theorem train_speed_correct :
  train_speed 2500 50 = 180 := 
by
  -- We leave the proof as sorry, the statement is sufficient
  sorry

end train_speed_correct_l150_150969


namespace combined_weight_l150_150685

-- Definition of conditions
def regular_dinosaur_weight := 800
def five_regular_dinosaurs_weight := 5 * regular_dinosaur_weight
def barney_weight := five_regular_dinosaurs_weight + 1500

-- Statement to prove
theorem combined_weight (h1: five_regular_dinosaurs_weight = 5 * regular_dinosaur_weight)
                        (h2: barney_weight = five_regular_dinosaurs_weight + 1500) : 
        (barney_weight + five_regular_dinosaurs_weight = 9500) :=
by
    sorry

end combined_weight_l150_150685


namespace approx_num_fish_in_pond_l150_150882

noncomputable def numFishInPond (tagged_in_second: ℕ) (total_second: ℕ) (tagged: ℕ) : ℕ :=
  tagged * total_second / tagged_in_second

theorem approx_num_fish_in_pond :
  numFishInPond 2 50 50 = 1250 := by
  sorry

end approx_num_fish_in_pond_l150_150882


namespace length_of_side_divisible_by_4_l150_150378

theorem length_of_side_divisible_by_4 {m n : ℕ} 
  (h : ∀ k : ℕ, (m * k) + (n * k) % 4 = 0 ) : 
  m % 4 = 0 ∨ n % 4 = 0 :=
by
  sorry

end length_of_side_divisible_by_4_l150_150378


namespace extracellular_proof_l150_150197

-- Define the components
def component1 : Set String := {"Na＋", "antibodies", "plasma proteins"}
def component2 : Set String := {"Hemoglobin", "O2", "glucose"}
def component3 : Set String := {"glucose", "CO2", "insulin"}
def component4 : Set String := {"Hormones", "neurotransmitter vesicles", "amino acids"}

-- Define the properties of being a part of the extracellular fluid
def is_extracellular (x : Set String) : Prop :=
  x = component1 ∨ x = component3

-- State the theorem to prove
theorem extracellular_proof : is_extracellular component1 ∧ ¬is_extracellular component2 ∧ is_extracellular component3 ∧ ¬is_extracellular component4 :=
by
  sorry

end extracellular_proof_l150_150197


namespace buns_per_pack_is_eight_l150_150222

-- Declaring the conditions
def burgers_per_guest : ℕ := 3
def total_friends : ℕ := 10
def friends_no_meat : ℕ := 1
def friends_no_bread : ℕ := 1
def packs_of_buns : ℕ := 3

-- Derived values from the conditions
def effective_friends_for_burgers : ℕ := total_friends - friends_no_meat
def effective_friends_for_buns : ℕ := total_friends - friends_no_bread

-- Final computation to prove
def buns_per_pack : ℕ := 24 / packs_of_buns

-- Theorem statement
theorem buns_per_pack_is_eight : buns_per_pack = 8 := by
  -- use sorry as we are not providing the proof steps 
  sorry

end buns_per_pack_is_eight_l150_150222


namespace find_p_4_l150_150379

-- Define the polynomial p(x)
def p (x : ℕ) : ℚ := sorry

-- Given conditions
axiom h1 : p 1 = 1
axiom h2 : p 2 = 1 / 4
axiom h3 : p 3 = 1 / 9
axiom h4 : p 5 = 1 / 25

-- Prove that p(4) = -1/30
theorem find_p_4 : p 4 = -1 / 30 := 
  by sorry

end find_p_4_l150_150379


namespace probability_of_prime_sum_l150_150827

def sum_is_prime (a b : ℕ) : Prop :=
  (a + b = 2) ∨ (a + b = 3) ∨ (a + b = 5) ∨ (a + b = 7) ∨ (a + b = 11) ∨ (a + b = 13)

def is_valid_outcome (faces : ℕ) (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ faces ∧ 1 ≤ b ∧ b ≤ faces

def favorable_outcomes (faces : ℕ) : ℕ :=
  (Finset.Icc 1 faces).sum (λ a => (Finset.Icc 1 faces).filter (sum_is_prime a).card)

theorem probability_of_prime_sum : (8 * 8 : ℚ)⁻¹ * (favorable_outcomes 8) = 23 / 64 := by
  sorry

end probability_of_prime_sum_l150_150827


namespace famous_quote_author_l150_150195

-- conditions
def statement_date := "July 20, 1969"
def mission := "Apollo 11"
def astronauts := ["Neil Armstrong", "Buzz Aldrin", "Michael Collins"]
def first_to_moon := "Neil Armstrong"

-- goal
theorem famous_quote_author : (statement_date = "July 20, 1969") ∧ (mission = "Apollo 11") ∧ (astronauts = ["Neil Armstrong", "Buzz Aldrin", "Michael Collins"]) ∧ (first_to_moon = "Neil Armstrong") → "Neil Armstrong" = "Neil Armstrong" :=
by 
  intros _; 
  exact rfl

end famous_quote_author_l150_150195


namespace evaluate_expression_l150_150839

theorem evaluate_expression :
  (3^1003 + 7^1004)^2 - (3^1003 - 7^1004)^2 = 5.292 * 10^1003 :=
by sorry

end evaluate_expression_l150_150839


namespace prime_dvd_square_l150_150312

theorem prime_dvd_square (p n : ℕ) (hp : Nat.Prime p) (h : p ∣ n^2) : p ∣ n :=
  sorry

end prime_dvd_square_l150_150312


namespace point_within_region_l150_150855

theorem point_within_region (a : ℝ) (h : 2 * a + 2 < 4) : a < 1 := 
sorry

end point_within_region_l150_150855


namespace volume_of_pyramid_l150_150080

theorem volume_of_pyramid (V_cube : ℝ) (h : ℝ) (A : ℝ) (V_pyramid : ℝ) : 
  V_cube = 27 → 
  h = 3 → 
  A = 4.5 → 
  V_pyramid = (1/3) * A * h → 
  V_pyramid = 4.5 := 
by 
  intros V_cube_eq h_eq A_eq V_pyramid_eq 
  sorry

end volume_of_pyramid_l150_150080


namespace train_speed_l150_150388

theorem train_speed (length_m : ℝ) (time_s : ℝ) 
  (h1 : length_m = 120) 
  (h2 : time_s = 3.569962336897346) 
  : (length_m / 1000) / (time_s / 3600) = 121.003 :=
by
  sorry

end train_speed_l150_150388


namespace track_meet_girls_short_hair_l150_150931

theorem track_meet_girls_short_hair :
  let total_people := 55
  let boys := 30
  let girls := total_people - boys
  let girls_long_hair := (3 / 5 : ℚ) * girls
  let girls_short_hair := girls - girls_long_hair
  girls_short_hair = 10 :=
by
  let total_people := 55
  let boys := 30
  let girls := total_people - boys
  let girls_long_hair := (3 / 5 : ℚ) * girls
  let girls_short_hair := girls - girls_long_hair
  sorry

end track_meet_girls_short_hair_l150_150931


namespace lines_perpendicular_l150_150006

theorem lines_perpendicular 
  (a b : ℝ) (θ : ℝ)
  (L1 : ∀ x y : ℝ, x * Real.cos θ + y * Real.sin θ + a = 0)
  (L2 : ∀ x y : ℝ, x * Real.sin θ - y * Real.cos θ + b = 0)
  : ∀ m1 m2 : ℝ, m1 = -(Real.cos θ) / (Real.sin θ) → m2 = (Real.sin θ) / (Real.cos θ) → m1 * m2 = -1 :=
by 
  intros m1 m2 h1 h2
  sorry

end lines_perpendicular_l150_150006


namespace trapezoid_upper_side_length_l150_150448

theorem trapezoid_upper_side_length (area base1 height : ℝ) (h1 : area = 222) (h2 : base1 = 23) (h3 : height = 12) : 
  ∃ base2, base2 = 14 :=
by
  -- The proof will be provided here.
  sorry

end trapezoid_upper_side_length_l150_150448


namespace area_of_abs_sum_eq_six_l150_150561

theorem area_of_abs_sum_eq_six : 
  (∃ (R : set (ℝ × ℝ)), (∀ (x y : ℝ), ((|x + y| + |x - y|) ≤ 6 → (x, y) ∈ R)) ∧ area R = 36) :=
sorry

end area_of_abs_sum_eq_six_l150_150561


namespace chairs_problem_l150_150031

theorem chairs_problem (B G W : ℕ) 
  (h1 : G = 3 * B) 
  (h2 : W = B + G - 13) 
  (h3 : B + G + W = 67) : 
  B = 10 :=
by
  sorry

end chairs_problem_l150_150031


namespace small_pump_fill_time_l150_150680

noncomputable def small_pump_time (large_pump_time combined_time : ℝ) : ℝ :=
  let large_pump_rate := 1 / large_pump_time
  let combined_rate := 1 / combined_time
  let small_pump_rate := combined_rate - large_pump_rate
  1 / small_pump_rate

theorem small_pump_fill_time :
  small_pump_time (1 / 3) 0.2857142857142857 = 2 :=
by
  sorry

end small_pump_fill_time_l150_150680


namespace profit_ratio_l150_150666

theorem profit_ratio (p_investment q_investment : ℝ) (h₁ : p_investment = 50000) (h₂ : q_investment = 66666.67) :
  (1 / q_investment) = (3 / 4 * 1 / p_investment) :=
by
  sorry

end profit_ratio_l150_150666


namespace num_possible_n_l150_150979

theorem num_possible_n (n : ℕ) : (∃ a b c : ℕ, 9 * a + 99 * b + 999 * c = 5000 ∧ n = a + 2 * b + 3 * c) ↔ n ∈ {x | x = a + 2 * b + 3 * c ∧ 0 ≤ 9 * (b + 12 * c) ∧ 9 * (b + 12 * c) ≤ 555} :=
sorry

end num_possible_n_l150_150979


namespace intersection_A_B_subset_A_B_l150_150862

noncomputable def set_A (a : ℝ) : Set ℝ := {x : ℝ | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5}
noncomputable def set_B : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ 22}

theorem intersection_A_B (a : ℝ) (ha : a = 10) : set_A a ∩ set_B = {x : ℝ | 21 ≤ x ∧ x ≤ 22} := by
  sorry

theorem subset_A_B (a : ℝ) : set_A a ⊆ set_B → a ≤ 9 := by
  sorry

end intersection_A_B_subset_A_B_l150_150862


namespace probability_only_one_product_probability_at_least_2_neither_l150_150640

-- Definitions based on given conditions
def prob_A : ℝ := 0.5
def prob_B : ℝ := 0.6
def prob_both_AB : ℝ := prob_A * prob_B
def prob_neither_AB : ℝ := (1 - prob_A) * (1 - prob_B)

-- Question (1)
theorem probability_only_one_product (P A B: Type) [ProbSpace P] (hA: prob_A = 0.5) (hB: prob_B = 0.6) (h_ind_A_B : IndepEvents A B) :
  P (A ∪ B \ (A ∩ B)) = 0.5 := sorry

-- Question (2)
theorem probability_at_least_2_neither (P A B: Type) [ProbSpace P] (hA: prob_A = 0.5) (hB: prob_B = 0.6) (h_ind_A_B : IndepEvents A B) :
  let prob_neither := (1 - prob_A) * (1 - prob_B),
      prob_at_least_2 := 1 - (0.8 ^ 3 + 3 * 0.8 ^ 2 * 0.2) in
  prob_at_least_2 = 0.104 := sorry

end probability_only_one_product_probability_at_least_2_neither_l150_150640


namespace negation_exists_l150_150648

-- Definitions used in the conditions
def prop1 (x : ℝ) : Prop := x^2 ≥ 1
def neg_prop1 : Prop := ∃ x : ℝ, x^2 < 1

-- Statement to be proved
theorem negation_exists (h : ∀ x : ℝ, prop1 x) : neg_prop1 :=
by
  sorry

end negation_exists_l150_150648


namespace Sue_shoe_probability_l150_150328

theorem Sue_shoe_probability :
  let total_shoes := 32
  let black_pairs := 8
  let brown_pairs := 4
  let gray_pairs := 2
  let red_pairs := 2

  -- Total number of shoes
  let num_black_shoes := black_pairs * 2
  let num_brown_shoes := brown_pairs * 2
  let num_gray_shoes := gray_pairs * 2
  let num_red_shoes := red_pairs * 2

  -- Probabilities for each color
  let prob_black := (num_black_shoes / total_shoes) * ((black_pairs) / (total_shoes - 1))
  let prob_brown := (num_brown_shoes / total_shoes) * ((brown_pairs) / (total_shoes - 1))
  let prob_gray := (num_gray_shoes / total_shoes) * ((gray_pairs) / (total_shoes - 1))
  let prob_red := (num_red_shoes / total_shoes) * ((red_pairs) / (total_shoes - 1))
  
  -- Total probability
  let total_probability := prob_black + prob_brown + prob_gray + prob_red

  total_probability = 11 / 62 :=
by
  sorry

end Sue_shoe_probability_l150_150328


namespace quadratic_equation_iff_non_zero_coefficient_l150_150449

theorem quadratic_equation_iff_non_zero_coefficient (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + a * x - 3 = 0 → (a - 2) ≠ 0) ↔ a ≠ 2 :=
by
  sorry

end quadratic_equation_iff_non_zero_coefficient_l150_150449


namespace not_prime_n_quad_plus_n_sq_plus_one_l150_150473

theorem not_prime_n_quad_plus_n_sq_plus_one (n : ℕ) (h : n ≥ 2) : ¬Prime (n^4 + n^2 + 1) :=
by
  sorry

end not_prime_n_quad_plus_n_sq_plus_one_l150_150473


namespace delta_value_l150_150445

-- Define the variables and the hypothesis
variable (Δ : Int)
variable (h : 5 * (-3) = Δ - 3)

-- State the theorem
theorem delta_value : Δ = -12 := by
  sorry

end delta_value_l150_150445


namespace solve_for_x_l150_150487

theorem solve_for_x (x : ℤ) (h : 24 - 6 = 3 + x) : x = 15 :=
by {
  sorry
}

end solve_for_x_l150_150487


namespace range_of_m_l150_150273

noncomputable def f (m x : ℝ) : ℝ := 2 * m * x^2 - 2 * (4 - m) * x + 1
noncomputable def g (m x : ℝ) : ℝ := m * x

theorem range_of_m :
  (∀ x : ℝ, f m x > 0 ∨ g m x > 0) → 0 < m ∧ m < 8 :=
sorry

end range_of_m_l150_150273


namespace circle_radius_increase_l150_150183

theorem circle_radius_increase (r r' : ℝ) (h : π * r'^2 = (25.44 / 100 + 1) * π * r^2) : 
  (r' - r) / r * 100 = 12 :=
by sorry

end circle_radius_increase_l150_150183


namespace smallest_x_consecutive_cubes_l150_150888

theorem smallest_x_consecutive_cubes :
  ∃ (u v w x : ℕ), u < v ∧ v < w ∧ w < x ∧ u + 1 = v ∧ v + 1 = w ∧ w + 1 = x ∧ (u^3 + v^3 + w^3 = x^3) ∧ (x = 6) :=
by {
  sorry
}

end smallest_x_consecutive_cubes_l150_150888


namespace distance_to_moscow_at_4PM_l150_150248

noncomputable def exact_distance_at_4PM (d12: ℝ) (d13: ℝ) (d15: ℝ) : ℝ :=
  d15 - 12

theorem distance_to_moscow_at_4PM  (h12 : 81.5 ≤ 82 ∧ 82 ≤ 82.5)
                                  (h13 : 70.5 ≤ 71 ∧ 71 ≤ 71.5)
                                  (h15 : 45.5 ≤ 46 ∧ 46 ≤ 46.5) :
  exact_distance_at_4PM 82 71 46 = 34 :=
by
  sorry

end distance_to_moscow_at_4PM_l150_150248


namespace transform_equation_l150_150971

theorem transform_equation (x : ℝ) :
  x^2 + 4 * x + 1 = 0 → (x + 2)^2 = 3 :=
by
  intro h
  sorry

end transform_equation_l150_150971


namespace berries_difference_l150_150984

theorem berries_difference (total_berries : ℕ) (dima_rate : ℕ) (sergey_rate : ℕ)
  (sergey_berries_picked : ℕ) (dima_berries_picked : ℕ)
  (dima_basket : ℕ) (sergey_basket : ℕ) :
  total_berries = 900 →
  sergey_rate = 2 * dima_rate →
  sergey_berries_picked = 2 * (total_berries / 3) →
  dima_berries_picked = total_berries / 3 →
  sergey_basket = sergey_berries_picked / 2 →
  dima_basket = (2 * dima_berries_picked) / 3 →
  sergey_basket > dima_basket ∧ sergey_basket - dima_basket = 100 :=
by
  intro h_total h_rate h_sergey_picked h_dima_picked h_sergey_basket h_dima_basket
  sorry

end berries_difference_l150_150984


namespace keith_spent_on_cards_l150_150755

theorem keith_spent_on_cards :
  let digimon_card_cost := 4.45
  let num_digimon_packs := 4
  let baseball_card_cost := 6.06
  let total_spent := num_digimon_packs * digimon_card_cost + baseball_card_cost
  total_spent = 23.86 :=
by
  sorry

end keith_spent_on_cards_l150_150755


namespace unique_fractions_count_l150_150870

theorem unique_fractions_count : 
  (Finset.card (Finset.filter (λ p : ℕ × ℕ, p.1 < p.2 ∧ p.2 ≤ 9) 
    (Finset.product (Finset.range 10).erase 0 (Finset.range 10).erase 0)).image (λ p, p.1 /. p.2)) = 27 :=
by
  sorry

end unique_fractions_count_l150_150870


namespace inequality_lemma_l150_150762

theorem inequality_lemma (x y z : ℝ) (h1 : 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z)
    (h2 : (1 / (x^2 - 1) + 1 / (y^2 - 1) + 1 / (z^2 - 1) = 1)) :
    (1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) ≤ 1) := 
by
  sorry

end inequality_lemma_l150_150762


namespace height_of_new_TV_l150_150475

theorem height_of_new_TV 
  (width1 height1 cost1 : ℝ) 
  (width2 cost2 : ℝ) 
  (cost_diff_per_sq_inch : ℝ) 
  (h1 : width1 = 24) 
  (h2 : height1 = 16) 
  (h3 : cost1 = 672) 
  (h4 : width2 = 48) 
  (h5 : cost2 = 1152) 
  (h6 : cost_diff_per_sq_inch = 1) : 
  ∃ height2 : ℝ, height2 = 32 :=
by
  sorry

end height_of_new_TV_l150_150475


namespace transformed_line_l150_150891

-- Define the original line equation
def original_line (x y : ℝ) : Prop := (x - 2 * y = 2)

-- Define the transformation
def transformation (x y x' y' : ℝ) : Prop :=
  (x' = x) ∧ (y' = 2 * y)

-- Prove that the transformed line equation holds
theorem transformed_line (x y x' y' : ℝ) (h₁ : original_line x y) (h₂ : transformation x y x' y') :
  x' - y' = 2 :=
sorry

end transformed_line_l150_150891


namespace find_length_of_room_l150_150916

noncomputable def cost_of_paving : ℝ := 21375
noncomputable def rate_per_sq_meter : ℝ := 900
noncomputable def width_of_room : ℝ := 4.75

theorem find_length_of_room :
  ∃ l : ℝ, l = (cost_of_paving / rate_per_sq_meter) / width_of_room ∧ l = 5 := by
  sorry

end find_length_of_room_l150_150916


namespace total_marks_l150_150225

variable (A M SS Mu : ℝ)

-- Conditions
def cond1 : Prop := M = A - 20
def cond2 : Prop := SS = Mu + 10
def cond3 : Prop := Mu = 70
def cond4 : Prop := M = (9 / 10) * A

-- Theorem statement
theorem total_marks (A M SS Mu : ℝ) (h1 : cond1 A M)
                                      (h2 : cond2 SS Mu)
                                      (h3 : cond3 Mu)
                                      (h4 : cond4 A M) :
    A + M + SS + Mu = 530 :=
by 
  sorry

end total_marks_l150_150225


namespace factorize_polynomial_1_factorize_polynomial_2_factorize_polynomial_3_l150_150988

theorem factorize_polynomial_1 (x y : ℝ) : 
  12 * x ^ 3 * y - 3 * x * y ^ 2 = 3 * x * y * (4 * x ^ 2 - y) := 
by sorry

theorem factorize_polynomial_2 (x : ℝ) : 
  x - 9 * x ^ 3 = x * (1 + 3 * x) * (1 - 3 * x) :=
by sorry

theorem factorize_polynomial_3 (a b : ℝ) : 
  3 * a ^ 2 - 12 * a * b * (a - b) = 3 * (a - 2 * b) ^ 2 := 
by sorry

end factorize_polynomial_1_factorize_polynomial_2_factorize_polynomial_3_l150_150988


namespace min_m_plus_n_l150_150261

theorem min_m_plus_n (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m * n - 2 * m - 3 * n = 20) : 
  m + n = 20 :=
sorry

end min_m_plus_n_l150_150261


namespace eval_sqrt_4_8_pow_12_l150_150109

theorem eval_sqrt_4_8_pow_12 : ((8 : ℝ)^(1 / 4))^12 = 512 :=
by
  -- This is where the proof steps would go 
  sorry

end eval_sqrt_4_8_pow_12_l150_150109


namespace geom_seq_sum_eq_37_l150_150884

theorem geom_seq_sum_eq_37
  (a r : ℝ)
  (h₃ : a + a*r + a*r^2 = 13)
  (h₇ : a * (1 - r^7) / (1 - r) = 183) :
  a * (1 + r + r^2 + r^3 + r^4) = 37 := 
by
  sorry

end geom_seq_sum_eq_37_l150_150884


namespace first_discount_percentage_l150_150919

theorem first_discount_percentage (D : ℝ) :
  (345 * (1 - D / 100) * 0.75 = 227.70) → (D = 12) :=
by
  intro cond
  sorry

end first_discount_percentage_l150_150919


namespace width_of_rectangle_11_l150_150186

variable (L W : ℕ)

-- The conditions: 
-- 1. The perimeter is 48cm
-- 2. Width is 2 cm shorter than length
def is_rectangle (L W : ℕ) : Prop :=
  2 * L + 2 * W = 48 ∧ W = L - 2

-- The statement we need to prove
theorem width_of_rectangle_11 (L W : ℕ) (h : is_rectangle L W) : W = 11 :=
by
  sorry

end width_of_rectangle_11_l150_150186


namespace value_of_a2_sub_b2_l150_150288

theorem value_of_a2_sub_b2 (a b : ℝ) (h1 : a + b = 6) (h2 : a - b = 2) : a^2 - b^2 = 12 :=
by
  sorry

end value_of_a2_sub_b2_l150_150288


namespace gum_boxes_l150_150536

theorem gum_boxes (c s t g : ℕ) (h1 : c = 2) (h2 : s = 5) (h3 : t = 9) (h4 : c + s + g = t) : g = 2 := by
  sorry

end gum_boxes_l150_150536


namespace complement_of_A_eq_interval_l150_150432

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 1} ∪ {x | x < 0}
def complement_U_A : Set ℝ := {x | 0 ≤ x ∧ x < 1}

theorem complement_of_A_eq_interval : (U \ A) = complement_U_A := by
  sorry

end complement_of_A_eq_interval_l150_150432


namespace john_total_amount_l150_150611

-- Given conditions from a)
def grandpa_amount : ℕ := 30
def grandma_amount : ℕ := 3 * grandpa_amount

-- Problem statement
theorem john_total_amount : grandpa_amount + grandma_amount = 120 :=
by
  sorry

end john_total_amount_l150_150611


namespace regular_polygon_properties_l150_150385

theorem regular_polygon_properties
  (exterior_angle : ℝ := 18) :
  (∃ (n : ℕ), n = 20) ∧ (∃ (interior_angle : ℝ), interior_angle = 162) := 
by
  sorry

end regular_polygon_properties_l150_150385


namespace increasing_function_range_l150_150020

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then
  -x^2 - a*x - 5
else
  a / x

theorem increasing_function_range (a : ℝ) : 
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (-3 ≤ a ∧ a ≤ -2) :=
by
  sorry

end increasing_function_range_l150_150020


namespace nature_of_graph_l150_150236

theorem nature_of_graph :
  ∀ (x y : ℝ), (x^2 - 3 * y) * (x - y + 1) = (y^2 - 3 * x) * (x - y + 1) →
    (y = -x - 3 ∨ y = x ∨ y = x + 1) ∧ ¬( (y = -x - 3) ∧ (y = x) ∧ (y = x + 1) ) :=
by
  intros x y h
  sorry

end nature_of_graph_l150_150236


namespace shirt_original_price_l150_150679

theorem shirt_original_price {P : ℝ} :
  (P * 0.80045740423098913 * 0.8745 = 105) → P = 150 :=
by sorry

end shirt_original_price_l150_150679


namespace rectangular_garden_length_l150_150076

theorem rectangular_garden_length (L P B : ℕ) (h1 : P = 600) (h2 : B = 150) (h3 : P = 2 * (L + B)) : L = 150 :=
by
  sorry

end rectangular_garden_length_l150_150076


namespace quadratic_example_correct_l150_150418

-- Define the quadratic function
def quad_func (x : ℝ) : ℝ := -2 * x^2 + 12 * x - 10

-- Conditions defined
def condition1 := quad_func 1 = 0
def condition2 := quad_func 5 = 0
def condition3 := quad_func 3 = 8

-- Theorem statement combining the conditions
theorem quadratic_example_correct :
  condition1 ∧ condition2 ∧ condition3 :=
by
  -- Proof omitted as per instructions
  sorry

end quadratic_example_correct_l150_150418


namespace service_center_location_l150_150797

def serviceCenterMilepost (x3 x10 : ℕ) (r : ℚ) : ℚ :=
  x3 + r * (x10 - x3)

theorem service_center_location :
  (serviceCenterMilepost 50 170 (2/3) : ℚ) = 130 :=
by
  -- placeholder for the actual proof
  sorry

end service_center_location_l150_150797


namespace int_even_bijection_l150_150252

theorem int_even_bijection :
  ∃ (f : ℤ → ℤ), (∀ n : ℤ, ∃ m : ℤ, f n = m ∧ m % 2 = 0) ∧
                 (∀ m : ℤ, m % 2 = 0 → ∃ n : ℤ, f n = m) := 
sorry

end int_even_bijection_l150_150252


namespace sequence_general_formula_l150_150601

theorem sequence_general_formula (a : ℕ → ℚ) 
  (h1 : a 1 = 1 / 2) 
  (h_rec : ∀ n : ℕ, a (n + 2) = 3 * a (n + 1) / (a (n + 1) + 3)) 
  (n : ℕ) : 
  a (n + 1) = 3 / (n + 6) :=
by
  sorry

end sequence_general_formula_l150_150601


namespace rice_cake_slices_length_l150_150221

noncomputable def slice_length (cake_length : ℝ) (num_cakes : ℕ) (overlap : ℝ) (num_slices : ℕ) : ℝ :=
  let total_original_length := num_cakes * cake_length
  let total_overlap := (num_cakes - 1) * overlap
  let actual_length := total_original_length - total_overlap
  actual_length / num_slices

theorem rice_cake_slices_length : 
  slice_length 2.7 5 0.3 6 = 2.05 :=
by
  sorry

end rice_cake_slices_length_l150_150221


namespace graph_does_not_pass_through_fourth_quadrant_l150_150336

def linear_function (x : ℝ) : ℝ := x + 1

theorem graph_does_not_pass_through_fourth_quadrant : 
  ¬ ∃ x : ℝ, x > 0 ∧ linear_function x < 0 :=
sorry

end graph_does_not_pass_through_fourth_quadrant_l150_150336


namespace find_c_l150_150343

-- Definition of the quadratic roots
def roots_form (c : ℝ) : Prop := 
  ∀ x : ℝ, (x^2 - 3 * x + c = 0) ↔ (x = (3 + real.sqrt c) / 2) ∨ (x = (3 - real.sqrt c) / 2)

-- Statement to prove that c = 9/5 given the roots form condition
theorem find_c (c : ℝ) (h : roots_form c) : c = 9 / 5 :=
sorry

end find_c_l150_150343


namespace minimize_expression_l150_150471

theorem minimize_expression (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z) (h_xyz : x * y * z = 2 / 3) :
  x^2 + 6 * x * y + 18 * y^2 + 12 * y * z + 4 * z^2 = 18 :=
sorry

end minimize_expression_l150_150471


namespace isosceles_triangle_l150_150278

theorem isosceles_triangle
  (α β γ : ℝ)
  (triangle_sum : α + β + γ = Real.pi)
  (second_triangle_angle1 : α + β < Real.pi)
  (second_triangle_angle2 : α + γ < Real.pi) :
  β = γ := 
sorry

end isosceles_triangle_l150_150278


namespace area_of_park_l150_150079

theorem area_of_park (L B : ℝ) (h1 : L / B = 1 / 3) (h2 : 12 * 1000 / 60 * 4 = 2 * (L + B)) : 
  L * B = 30000 :=
by
  sorry

end area_of_park_l150_150079


namespace cleaning_time_is_one_hour_l150_150750

def trees_rows : ℕ := 4
def trees_cols : ℕ := 5
def minutes_per_tree : ℕ := 6
noncomputable def help : ℝ := 1 / 2

noncomputable def total_trees : ℕ := trees_rows * trees_cols
noncomputable def total_minutes_without_help : ℕ := total_trees * minutes_per_tree
noncomputable def total_hours_without_help : ℝ := total_minutes_without_help / 60
noncomputable def total_hours_with_help : ℝ := total_hours_without_help * help

theorem cleaning_time_is_one_hour : total_hours_with_help = 1 :=
by {
  have h1 : total_trees = 20 := rfl,
  have h2 : total_minutes_without_help = 120 := rfl,
  have h3 : total_hours_without_help = 2 := by norm_num,
  have h4 : total_hours_with_help = 1 := by norm_num,
  exact h4,
}

end cleaning_time_is_one_hour_l150_150750


namespace largest_T_l150_150117

theorem largest_T (T : ℝ) (a b c d e : ℝ) 
  (h1: a ≥ 0) (h2: b ≥ 0) (h3: c ≥ 0) (h4: d ≥ 0) (h5: e ≥ 0)
  (h_sum : a + b = c + d + e)
  (h_T : T ≤ (Real.sqrt 30) / (30 + 12 * Real.sqrt 6)) : 
  Real.sqrt (a^2 + b^2 + c^2 + d^2 + e^2) ≥ T * (Real.sqrt a + Real.sqrt b + Real.sqrt c + Real.sqrt d + Real.sqrt e)^2 :=
sorry

end largest_T_l150_150117


namespace product_xyz_42_l150_150722

theorem product_xyz_42 (x y z : ℝ) 
  (h1 : (x - 2)^2 + (y - 3)^2 + (z - 4)^2 = 9)
  (h2 : x + y + z = 12) : x * y * z = 42 :=
by
  sorry

end product_xyz_42_l150_150722


namespace striped_jerseys_count_l150_150621

-- Define the cost of long-sleeved jerseys
def cost_long_sleeved := 15
-- Define the cost of striped jerseys
def cost_striped := 10
-- Define the number of long-sleeved jerseys bought
def num_long_sleeved := 4
-- Define the total amount spent
def total_spent := 80

-- Define a theorem to prove the number of striped jerseys bought
theorem striped_jerseys_count : ∃ x : ℕ, x * cost_striped = total_spent - num_long_sleeved * cost_long_sleeved ∧ x = 2 := 
by 
-- TODO: The proof steps would go here, but for this exercise, we use 'sorry' to skip the proof.
sorry

end striped_jerseys_count_l150_150621


namespace area_parallelogram_l150_150830

theorem area_parallelogram (AE EB : ℝ) (SAEF SCEF SAEC SBEC SABC SABCD : ℝ) (h1 : SAE = 2 * EB)
  (h2 : SCEF = 1) (h3 : SAE == 2 * SCEF / 3) (h4 : SAEC == SAE + SCEF) 
  (h5 : SBEC == 1/2 * SAEC) (h6 : SABC == SAEC + SBEC) (h7 : SABCD == 2 * SABC) :
  SABCD = 5 := sorry

end area_parallelogram_l150_150830


namespace sum_of_squares_l150_150636

theorem sum_of_squares (R r r1 r2 r3 d d1 d2 d3 : ℝ) 
  (h1 : d^2 = R^2 - 2 * R * r)
  (h2 : d1^2 = R^2 + 2 * R * r1)
  (h3 : d^2 + d1^2 + d2^2 + d3^2 = 12 * R^2) :
  d^2 + d1^2 + d2^2 + d3^2 = 12 * R^2 :=
by
  sorry

end sum_of_squares_l150_150636


namespace apples_kilos_first_scenario_l150_150735

noncomputable def cost_per_kilo_oranges : ℝ := 29
noncomputable def cost_per_kilo_apples : ℝ := 29
noncomputable def cost_first_scenario : ℝ := 419
noncomputable def cost_second_scenario : ℝ := 488
noncomputable def kilos_oranges_first_scenario : ℝ := 6
noncomputable def kilos_oranges_second_scenario : ℝ := 5
noncomputable def kilos_apples_second_scenario : ℝ := 7

theorem apples_kilos_first_scenario
  (O A : ℝ) 
  (cost1 cost2 : ℝ) 
  (k_oranges1 k_oranges2 k_apples2 : ℝ) 
  (hO : O = 29) (hA : A = 29) 
  (hCost1 : k_oranges1 * O + x * A = cost1) 
  (hCost2 : k_oranges2 * O + k_apples2 * A = cost2) 
  : x = 8 :=
by
  have hO : O = 29 := sorry
  have hA : A = 29 := sorry
  have h1 : k_oranges1 * O + x * A = cost1 := sorry
  have h2 : k_oranges2 * O + k_apples2 * A = cost2 := sorry
  sorry

end apples_kilos_first_scenario_l150_150735


namespace smallest_b_is_2_plus_sqrt_3_l150_150152

open Real

noncomputable def smallest_b (a b : ℝ) : ℝ :=
  if (2 < a ∧ a < b ∧ (¬(2 + a > b ∧ 2 + b > a ∧ a + b > 2)) ∧
    (¬(1 / b + 1 / a > 2 ∧ 1 / a + 2 > 1 / b ∧ 2 + 1 / b > 1 / a)))
  then b else 0

theorem smallest_b_is_2_plus_sqrt_3 (a b : ℝ) :
  2 < a ∧ a < b ∧ (¬(2 + a > b ∧ 2 + b > a ∧ a + b > 2)) ∧
    (¬(1 / b + 1 / a > 2 ∧ 1 / a + 2 > 1 / b ∧ 2 + 1 / b > 1 / a)) →
  b = 2 + sqrt 3 := sorry

end smallest_b_is_2_plus_sqrt_3_l150_150152


namespace find_xyz_l150_150898

theorem find_xyz
  (a b c x y z : ℂ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : a = (2 * b + 3 * c) / (x - 3))
  (h2 : b = (3 * a + 2 * c) / (y - 3))
  (h3 : c = (2 * a + 2 * b) / (z - 3))
  (h4 : x * y + x * z + y * z = -1)
  (h5 : x + y + z = 1) :
  x * y * z = 1 :=
sorry

end find_xyz_l150_150898


namespace log_equation_solution_l150_150952

theorem log_equation_solution (x : ℝ) (hx_pos : 0 < x) : 
  (Real.log x / Real.log 4) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 4 ↔ x ≠ 1 :=
by
  sorry

end log_equation_solution_l150_150952


namespace number_of_invertibles_mod_15_l150_150697

theorem number_of_invertibles_mod_15 : 
  ∃ (count : ℕ), count = 8 ∧ (set_of (λ a : ℕ, a < 15 ∧ Nat.gcd a 15 = 1)).card = count :=
by {
  sorry
}

end number_of_invertibles_mod_15_l150_150697


namespace common_ratio_geometric_series_l150_150185

theorem common_ratio_geometric_series (a r S : ℝ) (h₁ : S = a / (1 - r))
  (h₂ : r ≠ 1)
  (h₃ : r^4 * S = S / 81) :
  r = 1/3 :=
by 
  sorry

end common_ratio_geometric_series_l150_150185


namespace decimal_to_vulgar_fraction_l150_150212

theorem decimal_to_vulgar_fraction :
  ∃ (n d : ℕ), (0.34 : ℝ) = (n : ℝ) / (d : ℝ) ∧ n = 17 :=
by
  sorry

end decimal_to_vulgar_fraction_l150_150212


namespace find_point_coordinates_l150_150766

theorem find_point_coordinates (P : ℝ × ℝ)
  (h1 : P.1 < 0) -- Point P is in the second quadrant, so x < 0
  (h2 : P.2 > 0) -- Point P is in the second quadrant, so y > 0
  (h3 : abs P.2 = 4) -- distance from P to x-axis is 4
  (h4 : abs P.1 = 5) -- distance from P to y-axis is 5
  : P = (-5, 4) :=
by {
  -- point P is in the second quadrant, so x < 0 and y > 0
  -- |y| = 4 -> y = 4 
  -- |x| = 5 -> x = -5
  sorry
}

end find_point_coordinates_l150_150766


namespace pythagorean_triple_B_l150_150949

def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_B : isPythagoreanTriple 3 4 5 :=
by
  sorry

end pythagorean_triple_B_l150_150949


namespace angle_R_in_triangle_l150_150307

theorem angle_R_in_triangle (P Q R : ℝ) 
  (hP : P = 90)
  (hQ : Q = 4 * R - 10)
  (angle_sum : P + Q + R = 180) 
  : R = 20 := by 
sorry

end angle_R_in_triangle_l150_150307


namespace f_at_2_f_pos_solution_set_l150_150012

variable (a : ℝ)

def f (x : ℝ) : ℝ := x^2 - (3 - a) * x + 2 * (1 - a)

-- Question (I)
theorem f_at_2 : f a 2 = 0 := by sorry

-- Question (II)
theorem f_pos_solution_set :
  (∀ x, (a < -1 → (f a x > 0 ↔ (x < 2 ∨ 1 - a < x))) ∧
       (a = -1 → ¬(f a x > 0)) ∧
       (a > -1 → (f a x > 0 ↔ (1 - a < x ∧ x < 2)))) := 
by sorry

end f_at_2_f_pos_solution_set_l150_150012


namespace find_b_l150_150426

theorem find_b (b : ℤ) (h : ∃ x : ℝ, x^2 + b * x - 35 = 0 ∧ x = 5) : b = 2 :=
sorry

end find_b_l150_150426


namespace find_c_l150_150491

theorem find_c (x c : ℝ) (h₁ : 3 * x + 6 = 0) (h₂ : c * x - 15 = -3) : c = -6 := 
by
  -- sorry is used here as we are not required to provide the proof steps
  sorry

end find_c_l150_150491


namespace total_amount_from_grandparents_l150_150615

theorem total_amount_from_grandparents (amount_from_grandpa : ℕ) (multiplier : ℕ) (amount_from_grandma : ℕ) (total_amount : ℕ) 
  (h1 : amount_from_grandpa = 30) 
  (h2 : multiplier = 3) 
  (h3 : amount_from_grandma = multiplier * amount_from_grandpa) 
  (h4 : total_amount = amount_from_grandpa + amount_from_grandma) :
  total_amount = 120 := 
by 
  sorry

end total_amount_from_grandparents_l150_150615


namespace exists_consecutive_natural_numbers_satisfy_equation_l150_150976

theorem exists_consecutive_natural_numbers_satisfy_equation :
  ∃ (n a b c d: ℕ), a = n ∧ b = n+2 ∧ c = n-1 ∧ d = n+1 ∧ n>0 ∧ a * b - c * d = 11 :=
by
  sorry

end exists_consecutive_natural_numbers_satisfy_equation_l150_150976


namespace chickens_increased_l150_150803

-- Definitions and conditions
def initial_chickens := 45
def chickens_bought_day1 := 18
def chickens_bought_day2 := 12
def total_chickens_bought := chickens_bought_day1 + chickens_bought_day2

-- Proof statement
theorem chickens_increased :
  total_chickens_bought = 30 :=
by
  sorry

end chickens_increased_l150_150803


namespace find_m_l150_150266

-- Definitions of the conditions
def line (m : ℝ) : ℝ × ℝ → Prop := 
  fun p => p.1 - m * p.2 + 1 = 0

def circle (C : ℝ × ℝ) (r : ℝ) : ℝ × ℝ → Prop := 
  fun p => (p.1 - C.1)^2 + (p.2 - C.2)^2 = r^2

def area_triangle (a b c : ℝ × ℝ) : ℝ :=
  0.5 * ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))

-- Hypotheses
variables {m : ℝ}
def points_on_line (m : ℝ) (A B : ℝ × ℝ) : Prop := 
  line m A ∧ line m B

def points_on_circle (A B : ℝ × ℝ) : Prop := 
  circle (1, 0) 2 A ∧ circle (1, 0) 2 B

def area_condition (A B C : ℝ × ℝ) : Prop := 
  area_triangle A B C = 8 / 5

-- Main theorem
theorem find_m (A B : ℝ × ℝ) (C : ℝ × ℝ) :
  points_on_line m A B →
  points_on_circle A B →
  area_condition A B C →
  m = 2 ∨ m = -2 ∨ m = 1 / 2 ∨ m = -1 / 2 :=
sorry

end find_m_l150_150266


namespace integer_values_b_l150_150405

theorem integer_values_b (b : ℤ) : 
  (∃ (x1 x2 : ℤ), x1 + x2 = -b ∧ x1 * x2 = 7 * b) ↔ b = 0 ∨ b = 36 ∨ b = -28 ∨ b = -64 :=
by
  sorry

end integer_values_b_l150_150405


namespace girls_with_short_hair_count_l150_150926

-- Definitions based on the problem's conditions
def TotalPeople := 55
def Boys := 30
def FractionLongHair : ℚ := 3 / 5

-- The statement to prove
theorem girls_with_short_hair_count :
  (TotalPeople - Boys) - (TotalPeople - Boys) * FractionLongHair = 10 :=
by
  sorry

end girls_with_short_hair_count_l150_150926


namespace probability_reach_edge_within_five_hops_l150_150846

-- Define the probability of reaching an edge within n hops from the center
noncomputable def probability_reach_edge_by_hops (n : ℕ) : ℚ :=
if n = 5 then 121 / 128 else 0 -- This is just a placeholder for the real recursive computation.

-- Main theorem to prove
theorem probability_reach_edge_within_five_hops :
  probability_reach_edge_by_hops 5 = 121 / 128 :=
by
  -- Skipping the actual proof here
  sorry

end probability_reach_edge_within_five_hops_l150_150846


namespace eval_sqrt_4_8_pow_12_l150_150110

theorem eval_sqrt_4_8_pow_12 : ((8 : ℝ)^(1 / 4))^12 = 512 :=
by
  -- This is where the proof steps would go 
  sorry

end eval_sqrt_4_8_pow_12_l150_150110


namespace depth_of_channel_l150_150525

noncomputable def trapezium_area (a b h : ℝ) : ℝ :=
1/2 * (a + b) * h

theorem depth_of_channel :
  ∃ h : ℝ, trapezium_area 12 8 h = 700 ∧ h = 70 :=
by
  use 70
  unfold trapezium_area
  sorry

end depth_of_channel_l150_150525


namespace slope_range_of_line_l150_150125

/-- A mathematical proof problem to verify the range of the slope of a line
that passes through a given point (-1, -1) and intersects a circle. -/
theorem slope_range_of_line (
  k : ℝ
) : (∃ x y : ℝ, (y + 1 = k * (x + 1)) ∧ (x - 2) ^ 2 + y ^ 2 = 1) ↔ (0 < k ∧ k < 3 / 4) := 
by
  sorry  

end slope_range_of_line_l150_150125


namespace delta_value_l150_150446

-- Define the variables and the hypothesis
variable (Δ : Int)
variable (h : 5 * (-3) = Δ - 3)

-- State the theorem
theorem delta_value : Δ = -12 := by
  sorry

end delta_value_l150_150446


namespace cubic_inequality_l150_150330

theorem cubic_inequality (x y z : ℝ) :
  x^3 + y^3 + z^3 + 3 * x * y * z ≥ x^2 * (y + z) + y^2 * (z + x) + z^2 * (x + y) :=
sorry

end cubic_inequality_l150_150330


namespace islanders_liars_l150_150099

inductive Person
| A
| B

open Person

def is_liar (p : Person) : Prop :=
  sorry -- placeholder for the actual definition

def makes_statement (p : Person) (statement : Prop) : Prop :=
  sorry -- placeholder for the actual definition

theorem islanders_liars :
  makes_statement A (is_liar A ∧ ¬ is_liar B) →
  is_liar A ∧ is_liar B :=
by
  sorry

end islanders_liars_l150_150099


namespace vendelin_pastels_l150_150159

theorem vendelin_pastels (M V W : ℕ) (h1 : M = 5) (h2 : V < 5) (h3 : W = M + V) (h4 : M + V + W = 7 * V) : W = 7 := 
sorry

end vendelin_pastels_l150_150159


namespace correct_answers_l150_150033

-- Definitions
variable (C W : ℕ)
variable (h1 : C + W = 120)
variable (h2 : 3 * C - W = 180)

-- Goal statement
theorem correct_answers : C = 75 :=
by
  sorry

end correct_answers_l150_150033


namespace coefficient_of_x_squared_in_expansion_l150_150005

theorem coefficient_of_x_squared_in_expansion :
  let general_term (r : ℕ) := (-2)^r * (Nat.choose 7 r) * x^((7 - r) / 2)
  (∃ k, k = -280 ∧ (∃ r, r = 3 ∧ general_term r = k * x^2) ) :=
by
  let general_term := λ (r : ℕ), (-2 : ℤ)^r * (Nat.choose 7 r : ℤ) * (x^((7 - r) / 2) : ℤ)
  exists (k : ℤ),
  have h1 : k = -280,
  exists 3,
  have h2 : 3 = 3,
  general_term 3 = k * (x^2 : ℤ),
  sorry

end coefficient_of_x_squared_in_expansion_l150_150005


namespace value_of_expression_l150_150662

theorem value_of_expression : 
  103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 := by
  sorry

end value_of_expression_l150_150662


namespace good_carrots_l150_150938

-- Definitions
def vanessa_carrots : ℕ := 17
def mother_carrots : ℕ := 14
def bad_carrots : ℕ := 7

-- Proof statement
theorem good_carrots : (vanessa_carrots + mother_carrots) - bad_carrots = 24 := by
  sorry

end good_carrots_l150_150938


namespace greatest_power_of_two_l150_150505

theorem greatest_power_of_two (n : ℕ) (h1 : n = 1004) (h2 : 10^n - 4^(n / 2) = k) : ∃ m : ℕ, 2 ∣ k ∧ m = 1007 :=
by
  sorry

end greatest_power_of_two_l150_150505


namespace find_x_l150_150521

theorem find_x (x y z : ℤ) (h1 : 4 * x + y + z = 80) (h2 : 2 * x - y - z = 40) (h3 : 3 * x + y - z = 20) : x = 20 := by
  sorry

end find_x_l150_150521


namespace combined_weight_of_barney_and_five_dinosaurs_l150_150686

theorem combined_weight_of_barney_and_five_dinosaurs:
  let w := 800
  let combined_weight_five_regular := 5 * w
  let barney_weight := combined_weight_five_regular + 1500
  let combined_weight := barney_weight + combined_weight_five_regular
  in combined_weight = 9500 := by
  sorry

end combined_weight_of_barney_and_five_dinosaurs_l150_150686


namespace total_amount_from_grandparents_l150_150616

theorem total_amount_from_grandparents (amount_from_grandpa : ℕ) (multiplier : ℕ) (amount_from_grandma : ℕ) (total_amount : ℕ) 
  (h1 : amount_from_grandpa = 30) 
  (h2 : multiplier = 3) 
  (h3 : amount_from_grandma = multiplier * amount_from_grandpa) 
  (h4 : total_amount = amount_from_grandpa + amount_from_grandma) :
  total_amount = 120 := 
by 
  sorry

end total_amount_from_grandparents_l150_150616


namespace robin_earns_30_percent_more_than_erica_l150_150923

variable (E R C : ℝ)

theorem robin_earns_30_percent_more_than_erica
  (h1 : C = 1.60 * E)
  (h2 : C = 1.23076923076923077 * R) :
  R = 1.30 * E :=
by
  sorry

end robin_earns_30_percent_more_than_erica_l150_150923


namespace setB_is_PythagoreanTriple_setA_is_not_PythagoreanTriple_setC_is_not_PythagoreanTriple_setD_is_not_PythagoreanTriple_l150_150950

-- Define what it means to be a Pythagorean triple
def isPythagoreanTriple (a b c : Int) : Prop :=
  a^2 + b^2 = c^2

-- Define the given sets
def setA : (Int × Int × Int) := (12, 15, 18)
def setB : (Int × Int × Int) := (3, 4, 5)
def setC : (Rat × Rat × Rat) := (1.5, 2, 2.5)
def setD : (Int × Int × Int) := (6, 9, 15)

-- Proven statements about each set
theorem setB_is_PythagoreanTriple : isPythagoreanTriple 3 4 5 :=
  by
  sorry

theorem setA_is_not_PythagoreanTriple : ¬ isPythagoreanTriple 12 15 18 :=
  by
  sorry

-- Pythagorean triples must consist of positive integers
theorem setC_is_not_PythagoreanTriple : ¬ ∃ (a b c : Int), a^2 + b^2 = c^2 ∧ 
  a = 3/2 ∧ b = 2 ∧ c = 5/2 :=
  by
  sorry

theorem setD_is_not_PythagoreanTriple : ¬ isPythagoreanTriple 6 9 15 :=
  by
  sorry

end setB_is_PythagoreanTriple_setA_is_not_PythagoreanTriple_setC_is_not_PythagoreanTriple_setD_is_not_PythagoreanTriple_l150_150950


namespace third_of_ten_l150_150290

theorem third_of_ten : (1/3 : ℝ) * 10 = 8 / 3 :=
by
  have h : (1/4 : ℝ) * 20 = 4 := by sorry
  sorry

end third_of_ten_l150_150290


namespace sum_of_remainders_l150_150943

theorem sum_of_remainders (a b c d : ℤ) (h1 : a % 53 = 33) (h2 : b % 53 = 25) (h3 : c % 53 = 6) (h4 : d % 53 = 12) : 
  (a + b + c + d) % 53 = 23 :=
by {
  sorry
}

end sum_of_remainders_l150_150943


namespace Moe_has_least_amount_of_money_l150_150689

variables {B C F J M Z : ℕ}

theorem Moe_has_least_amount_of_money
  (h1 : Z > F) (h2 : F > B) (h3 : Z > C) (h4 : B > M) (h5 : C > M) (h6 : Z > J) (h7 : J > M) :
  ∀ x, x ≠ M → x > M :=
by {
  sorry
}

end Moe_has_least_amount_of_money_l150_150689


namespace eval_sqrt4_8_pow12_l150_150108

-- Define the fourth root of 8
def fourthRootOfEight : ℝ := 8 ^ (1 / 4)

-- Define the original expression
def expr := (fourthRootOfEight) ^ 12

-- The theorem to prove
theorem eval_sqrt4_8_pow12: expr = 512 := by
  sorry

end eval_sqrt4_8_pow12_l150_150108


namespace correctly_calculated_value_l150_150953

theorem correctly_calculated_value :
  ∀ (x : ℕ), (x * 15 = 45) → ((x * 5) * 10 = 150) := 
by
  intro x
  intro h
  sorry

end correctly_calculated_value_l150_150953


namespace alcohol_added_amount_l150_150375

theorem alcohol_added_amount :
  ∀ (x : ℝ), (40 * 0.05 + x) = 0.15 * (40 + x + 4.5) -> x = 5.5 :=
by
  intro x
  sorry

end alcohol_added_amount_l150_150375


namespace problem_solution_l150_150340

noncomputable def otimes (a b : ℝ) : ℝ := (a^3) / b

theorem problem_solution :
  (otimes (otimes 2 3) 4) - (otimes 2 (otimes 3 4)) = (32/9) :=
by
  sorry

end problem_solution_l150_150340


namespace expression_value_l150_150603

def α : ℝ := 60
def β : ℝ := 20
def AB : ℝ := 1

noncomputable def γ : ℝ := 180 - (α + β)

noncomputable def AC : ℝ := AB * (Real.sin γ / Real.sin β)
noncomputable def BC : ℝ := (Real.sin α / Real.sin γ) * AB

theorem expression_value : (1 / AC - BC) = 2 := by
  sorry

end expression_value_l150_150603


namespace alex_avg_speed_l150_150391

theorem alex_avg_speed (v : ℝ) : 
  (4.5 * v + 2.5 * 12 + 1.5 * 24 + 8 = 164) → v = 20 := 
by 
  intro h
  sorry

end alex_avg_speed_l150_150391


namespace emily_sold_toys_l150_150701

theorem emily_sold_toys (initial_toys : ℕ) (remaining_toys : ℕ) (sold_toys : ℕ) 
  (h_initial : initial_toys = 7) 
  (h_remaining : remaining_toys = 4) 
  (h_sold : sold_toys = initial_toys - remaining_toys) :
  sold_toys = 3 :=
by sorry

end emily_sold_toys_l150_150701


namespace fruit_mix_apples_count_l150_150213

variable (a o b p : ℕ)

theorem fruit_mix_apples_count :
  a + o + b + p = 240 →
  o = 3 * a →
  b = 2 * o →
  p = 5 * b →
  a = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end fruit_mix_apples_count_l150_150213


namespace least_marbles_l150_150194

theorem least_marbles (n : ℕ) (h1 : n % 7 = 3) (h2 : n % 4 = 2) (h3 : n % 6 = 1) : n = 402 :=
by
  sorry

end least_marbles_l150_150194


namespace sum_x_y_eq_2_l150_150038

open Real

theorem sum_x_y_eq_2 (x y : ℝ) (h : x - 1 = 1 - y) : x + y = 2 :=
by
  sorry

end sum_x_y_eq_2_l150_150038


namespace cube_of_sum_l150_150409

theorem cube_of_sum :
  (100 + 2) ^ 3 = 1061208 :=
by
  sorry

end cube_of_sum_l150_150409


namespace assoc_mul_l150_150807

-- Conditions from the problem
variables (x y z : Type) [Mul x] [Mul y] [Mul z]

theorem assoc_mul (a b c : x) : (a * b) * c = a * (b * c) := by sorry

end assoc_mul_l150_150807


namespace max_value_of_m_l150_150423

theorem max_value_of_m (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (2 / a) + (1 / b) = 1 / 4) : 2 * a + b ≥ 36 :=
by 
  -- Skipping the proof
  sorry

end max_value_of_m_l150_150423


namespace age_of_new_person_l150_150488

theorem age_of_new_person (T A : ℤ) (h : (T / 10 - 3) = (T - 40 + A) / 10) : A = 10 :=
sorry

end age_of_new_person_l150_150488


namespace parking_spots_first_level_l150_150677

theorem parking_spots_first_level (x : ℕ) 
    (h1 : ∃ x, x + (x + 7) + (x + 13) + 14 = 46) : x = 4 :=
by
  sorry

end parking_spots_first_level_l150_150677


namespace case_a_sticks_case_b_square_l150_150056
open Nat 

premise n12 : Nat := 12
premise sticks12_sum : Nat := (n12 * (n12 + 1)) / 2  -- Sum of first 12 natural numbers
premise length_divisibility_4 : ¬ (sticks12_sum % 4 = 0)  -- Check if sum is divisible by 4

-- Need to break at least 2 sticks to form a square
theorem case_a_sticks (h : sticks12_sum = 78) (h2 : length_divisibility_4 = true) : 
  ∃ (k : Nat), k >= 2 := sorry

premise n15 : Nat := 15
premise sticks15_sum : Nat := (n15 * (n15 + 1)) / 2  -- Sum of first 15 natural numbers
premise length_divisibility4_b : sticks15_sum % 4 = 0  -- Check if sum is divisible by 4

-- Possible to form a square without breaking any sticks
theorem case_b_square (h : sticks15_sum = 120) (h2 : length_divisibility4_b = true) : 
  ∃ (k : Nat), k = 0 := sorry

end case_a_sticks_case_b_square_l150_150056


namespace find_correct_speed_l150_150041

variables (d t : ℝ) -- Defining distance and time as real numbers

theorem find_correct_speed
  (h1 : d = 30 * (t + 5 / 60))
  (h2 : d = 50 * (t - 5 / 60)) :
  ∃ r : ℝ, r = 37.5 ∧ d = r * t :=
by 
  -- Skip the proof for now
  sorry

end find_correct_speed_l150_150041


namespace multiples_4_9_l150_150435

theorem multiples_4_9 (T : ℕ) (h1 : T = 201) 
    (A : ℕ) (h2 : A = 50) 
    (B : ℕ) (h3 : B = 22)
    (LCM : ℕ) (h4 : LCM = 36)
    (C : ℕ) (h5 : C = 5) : 
    ∃ (n : ℕ), n = 62 := 
by 
    have multiples_4_or_9_not_both := (A - C) + (B - C)
    show ∃ (n : ℕ), n = 62 from
    ⟨multiples_4_or_9_not_both, sorry⟩

end multiples_4_9_l150_150435


namespace find_distance_MF_l150_150998

-- Define the parabola and point conditions
def parabola (x y : ℝ) := y^2 = 8 * x

-- Define the focus of the parabola
def F : ℝ × ℝ := (2, 0)

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Define the distance squared between two points
def dist_squared (A B : ℝ × ℝ) : ℝ :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2

-- Prove the required statement
theorem find_distance_MF (x y : ℝ) (hM : parabola x y) (h_dist: dist_squared (x, y) O = 3 * (x + 2)) :
  dist_squared (x, y) F = 9 := by
  sorry

end find_distance_MF_l150_150998


namespace grey_area_of_first_grid_is_16_grey_area_of_second_grid_is_15_white_area_of_third_grid_is_5_l150_150371

theorem grey_area_of_first_grid_is_16 (side_length : ℝ := 1) :
  let area_triangle (base height : ℝ) := 0.5 * base * height
  let area_rectangle (length width : ℝ) := length * width
  let grey_area := area_triangle 3 side_length 
                    + area_triangle 4 side_length 
                    + area_rectangle 6 side_length 
                    + area_triangle 2 side_length 
                    + area_triangle 2 side_length 
                    + area_rectangle 2 side_length 
                    + area_triangle 2 side_length 
                    + area_triangle 3 side_length
  grey_area = 16 := by
  sorry

theorem grey_area_of_second_grid_is_15 (side_length : ℝ := 1) :
  let area_triangle (base height : ℝ) := 0.5 * base * height
  let area_rectangle (length width : ℝ) := length * width
  let grey_area := area_triangle 4 side_length 
                    + area_rectangle 2 side_length
                    + area_triangle 6 side_length 
                    + area_rectangle 2 side_length 
                    + area_triangle 2 side_length 
                    + area_triangle 2 side_length 
                    + area_rectangle 4 side_length
  grey_area = 15 := by
  sorry

theorem white_area_of_third_grid_is_5 (total_rectangle_area dark_grey_area : ℝ) (grey_area1 grey_area2 : ℝ) :
    total_rectangle_area = 32 ∧ dark_grey_area = 4 ∧ grey_area1 = 16 ∧ grey_area2 = 15 →
    let total_grey_area_recounted := grey_area1 + grey_area2 - dark_grey_area
    let white_area := total_rectangle_area - total_grey_area_recounted
    white_area = 5 := by
  sorry

end grey_area_of_first_grid_is_16_grey_area_of_second_grid_is_15_white_area_of_third_grid_is_5_l150_150371


namespace put_letters_in_mailboxes_l150_150162

theorem put_letters_in_mailboxes :
  (3:ℕ)^4 = 81 :=
by
  sorry

end put_letters_in_mailboxes_l150_150162


namespace rice_mixture_ratio_l150_150308

theorem rice_mixture_ratio
  (cost_variety1 : ℝ := 5) 
  (cost_variety2 : ℝ := 8.75) 
  (desired_cost_mixture : ℝ := 7.50) 
  (x y : ℝ) :
  5 * x + 8.75 * y = 7.50 * (x + y) → 
  y / x = 2 :=
by
  intro h
  sorry

end rice_mixture_ratio_l150_150308


namespace bounded_roots_l150_150472

open Polynomial

noncomputable def P : ℤ[X] := sorry -- Replace with actual polynomial if necessary

theorem bounded_roots (P : ℤ[X]) (n : ℕ) (hPdeg : P.degree = n) (hdec : 1 ≤ n) :
  ∀ k : ℤ, (P.eval k) ^ 2 = 1 → ∃ (r s : ℕ), r + s ≤ n + 2 := 
by 
  sorry

end bounded_roots_l150_150472


namespace johns_total_amount_l150_150609

def amount_from_grandpa : ℕ := 30
def multiplier : ℕ := 3
def amount_from_grandma : ℕ := amount_from_grandpa * multiplier
def total_amount : ℕ := amount_from_grandpa + amount_from_grandma

theorem johns_total_amount :
  total_amount = 120 :=
by
  sorry

end johns_total_amount_l150_150609


namespace weight_loss_percentage_l150_150198

theorem weight_loss_percentage (W : ℝ) (hW : W > 0) : 
  let new_weight := 0.89 * W
  let final_weight_with_clothes := new_weight * 1.02
  (W - final_weight_with_clothes) / W * 100 = 9.22 := by
  sorry

end weight_loss_percentage_l150_150198


namespace radius_ratio_of_circumscribed_truncated_cone_l150_150799

theorem radius_ratio_of_circumscribed_truncated_cone 
  (R r ρ : ℝ) 
  (h : ℝ) 
  (Vcs Vg : ℝ) 
  (h_eq : h = 2 * ρ)
  (Vcs_eq : Vcs = (π / 3) * h * (R^2 + r^2 + R * r))
  (Vg_eq : Vg = (4 * π * (ρ^3)) / 3)
  (Vcs_Vg_eq : Vcs = 2 * Vg) :
  (R / r) = (3 + Real.sqrt 5) / 2 := 
sorry

end radius_ratio_of_circumscribed_truncated_cone_l150_150799


namespace inequality_positive_reals_l150_150010

theorem inequality_positive_reals (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  (x / (1 + x^2)) + (y / (1 + y^2)) + (z / (1 + z^2)) ≤ (3 * Real.sqrt 3) / 4 := by
  sorry

end inequality_positive_reals_l150_150010


namespace girls_with_short_hair_count_l150_150927

-- Definitions based on the problem's conditions
def TotalPeople := 55
def Boys := 30
def FractionLongHair : ℚ := 3 / 5

-- The statement to prove
theorem girls_with_short_hair_count :
  (TotalPeople - Boys) - (TotalPeople - Boys) * FractionLongHair = 10 :=
by
  sorry

end girls_with_short_hair_count_l150_150927


namespace geometric_sequence_n_is_five_l150_150350

theorem geometric_sequence_n_is_five :
  ∃ n : ℕ, sum_of_geometric_sequence 1 (1/2) n = 31 / 16 ∧ n = 5 :=
sorry

def sum_of_geometric_sequence (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

end geometric_sequence_n_is_five_l150_150350


namespace age_difference_l150_150652

theorem age_difference (d : ℕ) (h1 : 18 + (18 - d) + (18 - 2 * d) + (18 - 3 * d) = 48) : d = 4 :=
sorry

end age_difference_l150_150652


namespace price_reduction_is_not_10_yuan_l150_150960

theorem price_reduction_is_not_10_yuan (current_price original_price : ℝ)
  (CurrentPrice : current_price = 45)
  (Reduction : current_price = 0.9 * original_price)
  (TenPercentReduction : 0.1 * original_price = 10) :
  (original_price - current_price) ≠ 10 := by
  sorry

end price_reduction_is_not_10_yuan_l150_150960


namespace arithmetic_sequence_term_count_l150_150586

theorem arithmetic_sequence_term_count (a1 d an : ℤ) (h₀ : a1 = -6) (h₁ : d = 5) (h₂ : an = 59) :
  ∃ n : ℤ, an = a1 + (n - 1) * d ∧ n = 14 :=
by
  sorry

end arithmetic_sequence_term_count_l150_150586


namespace louie_share_of_pie_l150_150831

def fraction_of_pie_taken_home (total_pie : ℚ) (shares : ℚ) : ℚ :=
  2 * (total_pie / shares)

theorem louie_share_of_pie : fraction_of_pie_taken_home (8 / 9) 4 = 4 / 9 := 
by 
  sorry

end louie_share_of_pie_l150_150831


namespace fencing_required_l150_150200

theorem fencing_required (L W A F : ℝ) (hL : L = 20) (hA : A = 390) (hArea : A = L * W) (hF : F = 2 * W + L) : F = 59 :=
by
  sorry

end fencing_required_l150_150200


namespace Darius_scored_10_points_l150_150004

theorem Darius_scored_10_points
  (D Marius Matt : ℕ)
  (h1 : Marius = D + 3)
  (h2 : Matt = D + 5)
  (h3 : D + Marius + Matt = 38) : 
  D = 10 :=
by
  sorry

end Darius_scored_10_points_l150_150004


namespace correct_region_l150_150669

-- Define the condition for x > 1
def condition_x_gt_1 (x : ℝ) (y : ℝ) : Prop :=
  x > 1 → y^2 > x

-- Define the condition for 0 < x < 1
def condition_0_lt_x_lt_1 (x : ℝ) (y : ℝ) : Prop :=
  0 < x ∧ x < 1 → 0 < y^2 ∧ y^2 < x

-- Formal statement to check the correct region
theorem correct_region (x y : ℝ) : 
  (condition_x_gt_1 x y ∨ condition_0_lt_x_lt_1 x y) →
  y^2 > x ∨ (0 < y^2 ∧ y^2 < x) :=
sorry

end correct_region_l150_150669


namespace ring_toss_total_earnings_l150_150347

noncomputable def daily_earnings : ℕ := 144
noncomputable def number_of_days : ℕ := 22
noncomputable def total_earnings : ℕ := daily_earnings * number_of_days

theorem ring_toss_total_earnings :
  total_earnings = 3168 := by
  sorry

end ring_toss_total_earnings_l150_150347


namespace number_properties_l150_150676

theorem number_properties : 
    ∃ (N : ℕ), 
    35 < N ∧ N < 70 ∧ N % 6 = 3 ∧ N % 8 = 1 ∧ N = 57 :=
by 
  sorry

end number_properties_l150_150676


namespace parabola_line_intersection_constant_line_equation_area_l150_150995

noncomputable def parabola_intersection (y1 y2 : ℝ) (k : ℝ) := (y1 * y2 = -18)

theorem parabola_line_intersection_constant
  (y1 y2 : ℝ) :
  (parabola_intersection y1 y2 (k)) :=
sorry
  
theorem line_equation_area
  (x1 x2 y1 y2 : ℝ)
  (h1 : x2 = 1)
  (h2 : y1 + y2 = 2)
  (h3 : y1 * y2 = -18)
  (h4 : |x1 * y1| = 4)
  (area_AOB : (∃ k: ℝ, 2*k*x1 + 3*k*y1 - 9 = 0 ∨ 2*k*x2 - 3*k*y2 - 9 = 0)) :
  ( area_AOB → (2*x1 + 3*y1 - 9) = 0 ∨ (2*x2 - 3*y2 - 9) = 0) 
  :=
sorry

end parabola_line_intersection_constant_line_equation_area_l150_150995


namespace no_positive_integer_solutions_l150_150555

theorem no_positive_integer_solutions (x y z : ℕ) (h_cond : x^2 + y^2 = 7 * z^2) : 
  x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end no_positive_integer_solutions_l150_150555


namespace find_a9_l150_150600

theorem find_a9 (a_1 a_2 : ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n ≥ 1, a (n + 2) = a (n + 1) + a n)
  (h2 : a 7 = 210)
  (h3 : a 1 = a_1)
  (h4 : a 2 = a_2) : 
  a 9 = 550 := by
  sorry

end find_a9_l150_150600


namespace find_real_parts_l150_150123

theorem find_real_parts (a b : ℝ) (i : ℂ) (hi : i*i = -1) 
(h : a + b*i = (1 - i) * i) : a = 1 ∧ b = -1 :=
sorry

end find_real_parts_l150_150123


namespace gas_cost_per_gallon_l150_150230

def car_mileage : Nat := 450
def car1_mpg : Nat := 50
def car2_mpg : Nat := 10
def car3_mpg : Nat := 15
def monthly_gas_cost : Nat := 56

theorem gas_cost_per_gallon (car_mileage car1_mpg car2_mpg car3_mpg monthly_gas_cost : Nat)
  (h1 : car_mileage = 450) 
  (h2 : car1_mpg = 50) 
  (h3 : car2_mpg = 10) 
  (h4 : car3_mpg = 15) 
  (h5 : monthly_gas_cost = 56) :
  monthly_gas_cost / ((car_mileage / 3) / car1_mpg + 
                      (car_mileage / 3) / car2_mpg + 
                      (car_mileage / 3) / car3_mpg) = 2 := 
by 
  sorry

end gas_cost_per_gallon_l150_150230


namespace dawn_lemonade_price_l150_150832

theorem dawn_lemonade_price (x : ℕ) : 
  (10 * 25) = (8 * x) + 26 → x = 28 :=
by 
  sorry

end dawn_lemonade_price_l150_150832


namespace simplify_expression_l150_150166

noncomputable def y := 
  Real.cos (2 * Real.pi / 15) + 
  Real.cos (4 * Real.pi / 15) + 
  Real.cos (8 * Real.pi / 15) + 
  Real.cos (14 * Real.pi / 15)

theorem simplify_expression : 
  y = (-1 + Real.sqrt 61) / 4 := 
sorry

end simplify_expression_l150_150166


namespace speed_of_second_cyclist_l150_150191

theorem speed_of_second_cyclist (v : ℝ) 
  (circumference : ℝ) 
  (time : ℝ) 
  (speed_first_cyclist : ℝ)
  (meet_time : ℝ)
  (circ_full: circumference = 300) 
  (time_full: time = 20)
  (speed_first: speed_first_cyclist = 7)
  (meet_full: meet_time = time):

  v = 8 := 
by
  sorry

end speed_of_second_cyclist_l150_150191


namespace dinner_cost_l150_150974

variable (total_cost : ℝ)
variable (tax_rate : ℝ)
variable (tip_rate : ℝ)
variable (pre_tax_cost : ℝ)
variable (tip : ℝ)
variable (tax : ℝ)
variable (final_cost : ℝ)

axiom h1 : total_cost = 27.50
axiom h2 : tax_rate = 0.10
axiom h3 : tip_rate = 0.15
axiom h4 : tax = tax_rate * pre_tax_cost
axiom h5 : tip = tip_rate * pre_tax_cost
axiom h6 : final_cost = pre_tax_cost + tax + tip

theorem dinner_cost : pre_tax_cost = 22 := by sorry

end dinner_cost_l150_150974


namespace angle_sum_eq_180_l150_150544

theorem angle_sum_eq_180 (A B C D E F G : ℝ) 
  (h1 : A + B + C + D + E + F = 360) : 
  A + B + C + D + E + F + G = 180 :=
by
  sorry

end angle_sum_eq_180_l150_150544


namespace equation_of_circle_center_0_4_passing_through_3_0_l150_150786

noncomputable def circle_radius (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem equation_of_circle_center_0_4_passing_through_3_0 :
  ∃ (r : ℝ), (r = circle_radius 0 4 3 0) ∧ (r = 5) ∧ ((x y : ℝ) → ((x - 0) ^ 2 + (y - 4) ^ 2 = r ^ 2) ↔ (x ^ 2 + (y - 4) ^ 2 = 25)) :=
by
  sorry

end equation_of_circle_center_0_4_passing_through_3_0_l150_150786


namespace train_speed_l150_150217

theorem train_speed
  (cross_time : ℝ := 5)
  (train_length : ℝ := 111.12)
  (conversion_factor : ℝ := 3.6)
  (speed : ℝ := (train_length / cross_time) * conversion_factor) :
  speed = 80 :=
by
  sorry

end train_speed_l150_150217


namespace find_value_l150_150726

variable (a b : ℝ)

def quadratic_equation_roots : Prop :=
  a^2 - 4 * a - 1 = 0 ∧ b^2 - 4 * b - 1 = 0

def sum_of_roots : Prop :=
  a + b = 4

def product_of_roots : Prop :=
  a * b = -1

theorem find_value (ha : quadratic_equation_roots a b) (hs : sum_of_roots a b) (hp : product_of_roots a b) :
  2 * a^2 + 3 / b + 5 * b = 22 :=
sorry

end find_value_l150_150726


namespace symmetry_x_y_axis_symmetry_line_y_neg1_l150_150782

-- Define point P
structure Point :=
  (x : ℝ)
  (y : ℝ)

def P : Point := { x := 1, y := 2 }

-- Condition for symmetry with respect to x-axis
def symmetric_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

-- Condition for symmetry with respect to the line y = -1
def symmetric_line_y_neg1 (p : Point) : Point :=
  { x := p.x, y := 2 * 1 - p.y - 1 }

-- Theorem statements
theorem symmetry_x_y_axis : symmetric_x P = { x := 1, y := -2 } := sorry
theorem symmetry_line_y_neg1 : symmetric_line_y_neg1 { x := 1, y := -2 } = { x := 1, y := 3 } := sorry

end symmetry_x_y_axis_symmetry_line_y_neg1_l150_150782


namespace product_of_integers_prime_at_most_one_prime_l150_150739

open Nat

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem product_of_integers_prime_at_most_one_prime (a b p : ℤ) (hp : is_prime (Int.natAbs p)) (hprod : a * b = p) :
  (is_prime (Int.natAbs a) ∧ ¬is_prime (Int.natAbs b)) ∨ (¬is_prime (Int.natAbs a) ∧ is_prime (Int.natAbs b)) ∨ ¬is_prime (Int.natAbs a) ∧ ¬is_prime (Int.natAbs b) :=
sorry

end product_of_integers_prime_at_most_one_prime_l150_150739


namespace smallest_number_of_cookies_l150_150674

theorem smallest_number_of_cookies
  (n : ℕ) 
  (hn : 4 * n - 4 = (n^2) / 2) : n = 7 → n^2 = 49 := 
by
  sorry

end smallest_number_of_cookies_l150_150674


namespace smallest_integer_M_exists_l150_150118

theorem smallest_integer_M_exists :
  ∃ (M : ℕ), 
    (M > 0) ∧ 
    (∃ (x y z : ℕ), 
      (x = M ∨ x = M + 1 ∨ x = M + 2) ∧ 
      (y = M ∨ y = M + 1 ∨ y = M + 2) ∧ 
      (z = M ∨ z = M + 1 ∨ z = M + 2) ∧ 
      ((x = M ∨ x = M + 1 ∨ x = M + 2) ∧ x % 8 = 0) ∧ 
      ((y = M ∨ y = M + 1 ∨ y = M + 2) ∧ y % 9 = 0) ∧ 
      ((z = M ∨ z = M + 1 ∨ z = M + 2) ∧ z % 25 = 0) ) ∧ 
    M = 200 := 
by
  sorry

end smallest_integer_M_exists_l150_150118


namespace electricity_consumption_scientific_notation_l150_150757

def electricity_consumption (x : Float) : String := 
  let scientific_notation := "3.64 × 10^4"
  scientific_notation

theorem electricity_consumption_scientific_notation :
  electricity_consumption 36400 = "3.64 × 10^4" :=
by 
  sorry

end electricity_consumption_scientific_notation_l150_150757


namespace rectangle_perimeter_l150_150770

noncomputable def perimeter_rectangle (x y : ℝ) : ℝ := 2 * (x + y)

theorem rectangle_perimeter
  (x y a b : ℝ)
  (H1 : x * y = 2006)
  (H2 : x + y = 2 * a)
  (H3 : x^2 + y^2 = 4 * (a^2 - b^2))
  (b_val : b = Real.sqrt 1003)
  (a_val : a = 2 * Real.sqrt 1003) :
  perimeter_rectangle x y = 8 * Real.sqrt 1003 := by
  sorry

end rectangle_perimeter_l150_150770


namespace track_meet_girls_short_hair_l150_150933

theorem track_meet_girls_short_hair :
  let total_people := 55
  let boys := 30
  let girls := total_people - boys
  let girls_long_hair := (3 / 5 : ℚ) * girls
  let girls_short_hair := girls - girls_long_hair
  girls_short_hair = 10 :=
by
  let total_people := 55
  let boys := 30
  let girls := total_people - boys
  let girls_long_hair := (3 / 5 : ℚ) * girls
  let girls_short_hair := girls - girls_long_hair
  sorry

end track_meet_girls_short_hair_l150_150933


namespace cube_dimension_l150_150509

theorem cube_dimension (x s : ℝ) (hx1 : s^3 = 8 * x) (hx2 : 6 * s^2 = 2 * x) : x = 1728 := 
by {
  sorry
}

end cube_dimension_l150_150509


namespace function_domain_real_l150_150428

theorem function_domain_real (k : ℝ) : 0 ≤ k ∧ k < 4 ↔ (∀ x : ℝ, k * x^2 + k * x + 1 ≠ 0) :=
by
  sorry

end function_domain_real_l150_150428


namespace product_of_ratios_eq_l150_150064

theorem product_of_ratios_eq :
  (∃ x_1 y_1 x_2 y_2 x_3 y_3 : ℝ,
    (x_1^3 - 3 * x_1 * y_1^2 = 2006) ∧
    (y_1^3 - 3 * x_1^2 * y_1 = 2007) ∧
    (x_2^3 - 3 * x_2 * y_2^2 = 2006) ∧
    (y_2^3 - 3 * x_2^2 * y_2 = 2007) ∧
    (x_3^3 - 3 * x_3 * y_3^2 = 2006) ∧
    (y_3^3 - 3 * x_3^2 * y_3 = 2007)) →
    (1 - x_1 / y_1) * (1 - x_2 / y_2) * (1 - x_3 / y_3) = 1 / 1003 :=
by
  sorry

end product_of_ratios_eq_l150_150064


namespace proposition_false_at_4_l150_150965

open Nat

def prop (n : ℕ) : Prop := sorry -- the actual proposition is not specified, so we use sorry

theorem proposition_false_at_4 :
  (∀ k : ℕ, k > 0 → (prop k → prop (k + 1))) →
  ¬ prop 5 →
  ¬ prop 4 :=
by
  intros h_induction h_proposition_false_at_5
  sorry

end proposition_false_at_4_l150_150965


namespace negation_of_divisible_by_2_even_l150_150179

theorem negation_of_divisible_by_2_even :
  (¬ ∀ n : ℤ, (∃ k, n = 2 * k) → (∃ k, n = 2 * k ∧ n % 2 = 0)) ↔
  ∃ n : ℤ, (∃ k, n = 2 * k) ∧ ¬ (n % 2 = 0) :=
by
  sorry

end negation_of_divisible_by_2_even_l150_150179


namespace percentage_of_125_equals_75_l150_150084

theorem percentage_of_125_equals_75 (p : ℝ) (h : p * 125 = 75) : p = 60 / 100 :=
by
  sorry

end percentage_of_125_equals_75_l150_150084


namespace min_sticks_to_break_for_square_12_can_form_square_without_breaking_15_l150_150059

-- Part (a): For n = 12:
theorem min_sticks_to_break_for_square_12 : ∀ (n : ℕ), n = 12 → 
  (∃ (sticks : Finset ℕ), sticks.card = 12 ∧ sticks.sum id = 78 ∧ (¬ (78 % 4 = 0) → 
  ∃ (b : ℕ), b = 2)) := 
by sorry

-- Part (b): For n = 15:
theorem can_form_square_without_breaking_15 : ∀ (n : ℕ), n = 15 → 
  (∃ (sticks : Finset ℕ), sticks.card = 15 ∧ sticks.sum id = 120 ∧ (120 % 4 = 0)) :=
by sorry

end min_sticks_to_break_for_square_12_can_form_square_without_breaking_15_l150_150059


namespace average_rate_decrease_price_reduction_l150_150211

-- Define the initial and final factory prices
def initial_price : ℝ := 200
def final_price : ℝ := 162

-- Define the function representing the average rate of decrease
def average_rate_of_decrease (x : ℝ) : Prop :=
  initial_price * (1 - x) * (1 - x) = final_price

-- Theorem stating the average rate of decrease (proving x = 0.1)
theorem average_rate_decrease : ∃ x : ℝ, average_rate_of_decrease x ∧ x = 0.1 :=
by
  use 0.1
  sorry

-- Define the selling price without reduction, sold without reduction, increase in pieces sold, and profit
def selling_price : ℝ := 200
def sold_without_reduction : ℕ := 20
def increase_pcs_per_5yuan_reduction : ℕ := 10
def profit : ℝ := 1150

-- Define the function representing the price reduction determination
def price_reduction_correct (m : ℝ) : Prop :=
  (38 - m) * (sold_without_reduction + 2 * m / 5) = profit

-- Theorem stating the price reduction (proving m = 15)
theorem price_reduction : ∃ m : ℝ, price_reduction_correct m ∧ m = 15 :=
by
  use 15
  sorry

end average_rate_decrease_price_reduction_l150_150211


namespace zongzi_profit_l150_150779

def initial_cost : ℕ := 10
def initial_price : ℕ := 16
def initial_bags_sold : ℕ := 200
def additional_sales_per_yuan (x : ℕ) : ℕ := 80 * x
def profit_per_bag (x : ℕ) : ℕ := initial_price - x - initial_cost
def number_of_bags_sold (x : ℕ) : ℕ := initial_bags_sold + additional_sales_per_yuan x
def total_profit (profit_per_bag : ℕ) (number_of_bags_sold : ℕ) : ℕ := profit_per_bag * number_of_bags_sold

theorem zongzi_profit (x : ℕ) : 
  total_profit (profit_per_bag x) (number_of_bags_sold x) = 1440 := 
sorry

end zongzi_profit_l150_150779


namespace tom_filled_balloons_l150_150657

theorem tom_filled_balloons :
  ∀ (Tom Luke Anthony : ℕ), 
    (Tom = 3 * Luke) →
    (Luke = Anthony / 4) →
    (Anthony = 44) →
    (Tom = 33) :=
by
  intros Tom Luke Anthony hTom hLuke hAnthony
  sorry

end tom_filled_balloons_l150_150657


namespace total_amount_correct_l150_150453

def num_2won_bills : ℕ := 8
def value_2won_bills : ℕ := 2
def num_1won_bills : ℕ := 2
def value_1won_bills : ℕ := 1

theorem total_amount_correct :
  (num_2won_bills * value_2won_bills) + (num_1won_bills * value_1won_bills) = 18 :=
by
  sorry

end total_amount_correct_l150_150453


namespace intersection_A_B_l150_150874

-- Define set A
def A : Set ℝ := { y | ∃ x : ℝ, y = Real.log x }

-- Define set B
def B : Set ℝ := { x | ∃ y : ℝ, y = Real.sqrt x }

-- Prove that the intersection of sets A and B is [0, +∞)
theorem intersection_A_B : A ∩ B = { x | 0 ≤ x } :=
by
  sorry

end intersection_A_B_l150_150874


namespace second_quadratic_roots_complex_iff_first_roots_real_distinct_l150_150204

theorem second_quadratic_roots_complex_iff_first_roots_real_distinct (q : ℝ) :
  q < 1 → (∀ x : ℂ, (3 - q) * x^2 + 2 * (1 + q) * x + (q^2 - q + 2) ≠ 0) :=
by
  -- Placeholder for the proof
  sorry

end second_quadratic_roots_complex_iff_first_roots_real_distinct_l150_150204


namespace part_a_l150_150011

-- Definition of the function f
def f (x : ℝ) := 2 * (Real.sqrt 3) * (Real.cos x) ^ 2 + 2 * (Real.sin x) * (Real.cos x) - Real.sqrt 3

theorem part_a (x : ℝ) : f x = 2 * Real.cos (2 * x - Real.pi / 6) := 
by
  sorry

end part_a_l150_150011


namespace min_value_of_sum_of_squares_l150_150748

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x - 2 * y - 3 * z = 4) : 
  (x^2 + y^2 + z^2) ≥ 8 / 7 :=
sorry

end min_value_of_sum_of_squares_l150_150748


namespace ratio_sum_of_squares_l150_150497

theorem ratio_sum_of_squares (a b c : ℕ) (h : a = 6 ∧ b = 1 ∧ c = 7 ∧ 72 / 98 = (a * (b.sqrt^2)).sqrt / c) : a + b + c = 14 := by 
  sorry

end ratio_sum_of_squares_l150_150497


namespace martin_travel_time_l150_150317

-- Definitions based on the conditions
def distance : ℕ := 12
def speed : ℕ := 2

-- Statement of the problem to be proven
theorem martin_travel_time : (distance / speed) = 6 := by sorry

end martin_travel_time_l150_150317


namespace time_against_current_l150_150089

-- Define the conditions:
def swimming_speed_still_water : ℝ := 6  -- Speed in still water (km/h)
def current_speed : ℝ := 2  -- Speed of the water current (km/h)
def time_with_current : ℝ := 3.5  -- Time taken to swim with the current (hours)

-- Define effective speeds:
def effective_speed_against_current (swimming_speed_still_water current_speed: ℝ) : ℝ :=
  swimming_speed_still_water - current_speed

def effective_speed_with_current (swimming_speed_still_water current_speed: ℝ) : ℝ :=
  swimming_speed_still_water + current_speed

-- Calculate the distance covered with the current:
def distance_with_current (time_with_current effective_speed_with_current: ℝ) : ℝ :=
  time_with_current * effective_speed_with_current

-- Define the proof goal:
theorem time_against_current (h1 : swimming_speed_still_water = 6) (h2 : current_speed = 2)
  (h3 : time_with_current = 3.5) :
  ∃ (t : ℝ), t = 7 := by
  sorry

end time_against_current_l150_150089


namespace intersection_A_B_l150_150861

-- Conditions
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := { y | ∃ x ∈ A, y = 3 * x - 2 }

-- Question and proof statement
theorem intersection_A_B :
  A ∩ B = {1, 4} := by
  sorry

end intersection_A_B_l150_150861


namespace towels_per_person_l150_150675

-- Define the conditions
def num_rooms : ℕ := 10
def people_per_room : ℕ := 3
def total_towels : ℕ := 60

-- Define the total number of people
def total_people : ℕ := num_rooms * people_per_room

-- Define the proposition to prove
theorem towels_per_person : total_towels / total_people = 2 :=
by sorry

end towels_per_person_l150_150675


namespace order_of_activities_l150_150651

noncomputable def fraction_liking_activity_dodgeball : ℚ := 8 / 24
noncomputable def fraction_liking_activity_barbecue : ℚ := 10 / 30
noncomputable def fraction_liking_activity_archery : ℚ := 9 / 18

theorem order_of_activities :
  (fraction_liking_activity_archery > fraction_liking_activity_dodgeball) ∧
  (fraction_liking_activity_archery > fraction_liking_activity_barbecue) ∧
  (fraction_liking_activity_dodgeball = fraction_liking_activity_barbecue) :=
by
  sorry

end order_of_activities_l150_150651


namespace problem_domains_equal_l150_150972

/-- Proof problem:
    Prove that the domain of the function y = (x - 1)^(-1/2) is equal to the domain of the function y = ln(x - 1).
--/
theorem problem_domains_equal :
  {x : ℝ | x > 1} = {x : ℝ | x > 1} :=
by
  sorry

end problem_domains_equal_l150_150972


namespace debt_calculation_correct_l150_150869

-- Conditions
def initial_debt : ℤ := 40
def repayment : ℤ := initial_debt / 2
def additional_borrowing : ℤ := 10

-- Final Debt Calculation
def remaining_debt : ℤ := initial_debt - repayment
def final_debt : ℤ := remaining_debt + additional_borrowing

-- Proof Statement
theorem debt_calculation_correct : final_debt = 30 := 
by 
  -- Skipping the proof
  sorry

end debt_calculation_correct_l150_150869


namespace quadratic_has_equal_roots_l150_150138

theorem quadratic_has_equal_roots (b : ℝ) (h : ∃ x : ℝ, b*x^2 + 2*b*x + 4 = 0 ∧ b*x^2 + 2*b*x + 4 = 0) :
  b = 4 :=
sorry

end quadratic_has_equal_roots_l150_150138


namespace positive_integers_satisfying_condition_l150_150048

theorem positive_integers_satisfying_condition :
  ∃! n : ℕ, 0 < n ∧ 24 - 6 * n > 12 :=
by
  sorry

end positive_integers_satisfying_condition_l150_150048


namespace jessica_borrowed_amount_l150_150893

def payment_pattern (hour : ℕ) : ℕ :=
  match (hour % 6) with
  | 1 => 2
  | 2 => 4
  | 3 => 6
  | 4 => 8
  | 5 => 10
  | _ => 12

def total_payment (hours_worked : ℕ) : ℕ :=
  (hours_worked / 6) * 42 + (List.sum (List.map payment_pattern (List.range (hours_worked % 6))))

theorem jessica_borrowed_amount :
  total_payment 45 = 306 :=
by
  -- Proof omitted
  sorry

end jessica_borrowed_amount_l150_150893


namespace stadium_seating_and_revenue_l150_150813

   def children := 52
   def adults := 29
   def seniors := 15
   def seats_A := 40
   def seats_B := 30
   def seats_C := 25
   def price_A := 10
   def price_B := 15
   def price_C := 20
   def total_seats := 95

   def revenue_A := seats_A * price_A
   def revenue_B := seats_B * price_B
   def revenue_C := seats_C * price_C
   def total_revenue := revenue_A + revenue_B + revenue_C

   theorem stadium_seating_and_revenue :
     (children <= seats_B + seats_C) ∧
     (adults + seniors <= seats_A + seats_C) ∧
     (children + adults + seniors > total_seats) →
     (revenue_A = 400) ∧
     (revenue_B = 450) ∧
     (revenue_C = 500) ∧
     (total_revenue = 1350) :=
   by
     sorry
   
end stadium_seating_and_revenue_l150_150813


namespace equal_after_operations_l150_150065

theorem equal_after_operations :
  let initial_first_number := 365
  let initial_second_number := 24
  let first_number_after_n_operations := initial_first_number - 19 * 11
  let second_number_after_n_operations := initial_second_number + 12 * 11
  first_number_after_n_operations = second_number_after_n_operations := sorry

end equal_after_operations_l150_150065


namespace oblique_projection_area_correct_l150_150293

variables (a : ℝ)

-- Definition of original area of an equilateral triangle with side length a
def equilateral_triangle_area (a : ℝ) : ℝ := (a * a * real.sqrt 3) / 4

-- Transform the area by the factor of √2/4 in the oblique projection method
def oblique_projection_area (original_area : ℝ) : ℝ := (original_area * real.sqrt 2) / 4

-- Prove that the area in the oblique projection method is √6/16 * a^2
theorem oblique_projection_area_correct (a : ℝ) :
  oblique_projection_area (equilateral_triangle_area a) = (real.sqrt 6 * a^2) / 16 :=
by
  sorry

end oblique_projection_area_correct_l150_150293


namespace sum_of_digits_of_Joey_age_next_time_is_twice_Liam_age_l150_150036

-- Definitions of the conditions
def L := 2
def C := 2 * L^2  -- Chloe's age today based on Liam's age
def J := C + 3    -- Joey's age today

-- The future time when Joey's age is twice Liam's age
def future_time : ℕ := (sorry : ℕ) -- Placeholder for computation of 'n'
lemma compute_n : 2 * (L + future_time) = J + future_time := sorry

-- Joey's age at future time when it is twice Liam's age
def age_at_future_time : ℕ := J + future_time

-- Sum of the two digits of Joey's age at that future time
def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

-- Final statement: sum of the digits of Joey's age at the specified future time
theorem sum_of_digits_of_Joey_age_next_time_is_twice_Liam_age :
  digit_sum age_at_future_time = 9 :=
by
  exact sorry

end sum_of_digits_of_Joey_age_next_time_is_twice_Liam_age_l150_150036


namespace equalize_money_l150_150101

theorem equalize_money (
  Carmela_money : ℕ, Cousin_money : ℕ, num_cousins : ℕ) :
  Carmela_money = 7 → Cousin_money = 2 → num_cousins = 4 →
  ∀ (x : ℕ), Carmela_money - num_cousins * x = Cousin_money + x ∧
  (Carmela_money - num_cousins * x = Cousin_money + x) →
  x = 1 :=
by
  intros hCarmela_money hCousin_money hnum_cousins hx hfinal_eq
  sorry

end equalize_money_l150_150101


namespace short_haired_girls_l150_150928

def total_people : ℕ := 55
def boys : ℕ := 30
def total_girls : ℕ := total_people - boys
def girls_with_long_hair : ℕ := (3 / 5) * total_girls
def girls_with_short_hair : ℕ := total_girls - girls_with_long_hair

theorem short_haired_girls :
  girls_with_short_hair = 10 := sorry

end short_haired_girls_l150_150928


namespace intersection_of_A_and_B_l150_150723

-- Define the sets A and B based on the given conditions
def setA : Set ℝ := {x | x^2 - 2 * x < 0}
def setB : Set ℝ := {x | -1 < x ∧ x < 1}

-- State the theorem to prove the intersection A ∩ B
theorem intersection_of_A_and_B : ((setA ∩ setB) = {x : ℝ | 0 < x ∧ x < 1}) :=
by
  sorry

end intersection_of_A_and_B_l150_150723


namespace swans_in_10_years_l150_150115

def doubling_time := 2
def initial_swans := 15
def periods := 10 / doubling_time

theorem swans_in_10_years : 
  (initial_swans * 2 ^ periods) = 480 := 
by
  sorry

end swans_in_10_years_l150_150115


namespace area_of_region_l150_150345

theorem area_of_region (r : ℝ) (theta_deg : ℝ) (a b c : ℤ) : 
  r = 8 → 
  theta_deg = 45 → 
  (r^2 * theta_deg * Real.pi / 360) - (1/2 * r^2 * Real.sin (theta_deg * Real.pi / 180)) = (a * Real.sqrt b + c * Real.pi) →
  a + b + c = -22 :=
by 
  intros hr htheta Harea 
  sorry

end area_of_region_l150_150345


namespace product_of_all_possible_values_of_x_l150_150285

def conditions (x : ℚ) : Prop := abs (18 / x - 4) = 3

theorem product_of_all_possible_values_of_x:
  ∃ x1 x2 : ℚ, conditions x1 ∧ conditions x2 ∧ ((18 * 18) / (x1 * x2) = 324 / 7) :=
sorry

end product_of_all_possible_values_of_x_l150_150285


namespace problem_part1_problem_part2_l150_150724

-- Definitions and conditions
def a_n : ℕ → ℕ
| 0       := 0
| (n + 1) := 2^(n + 2)  -- Given geometric sequence

def S : ℕ → ℕ := λ n, (Finset.range n).sum (λ i, a_n i)

theorem problem_part1 :
  a_n 1 = 8 ∧ (S 3 + 3 * a_n 4 = S 5) → 
  ∀ n, a_n n = 2^(n + 2) :=
by sorry

-- Additional definitions for b_n, c_n, P_n and Q_n
def b_n (n : ℕ) : ℕ := 2 * n + 5

def c_n (n : ℕ) : ℕ := 1 / (b_n n * b_n (n + 1))

def P_n (n : ℕ) : ℕ := n^2 + 6 * n

def Q_n (n : ℕ) : ℕ := n / (14 * n + 49)

theorem problem_part2 :
  ( ∀ n, b_n n = log 2 (a_n n * a_n (n + 1)) ∧
    ∀ n, c_n n = 1 / (b_n n * b_n (n + 1)) ) →
  ∀ n, P_n n = n^2 + 6 * n ∧ Q_n n = n / (14 * n + 49) :=
by sorry

end problem_part1_problem_part2_l150_150724


namespace rice_mixture_ratio_l150_150747

theorem rice_mixture_ratio (x y : ℝ) (h1 : 7 * x + 8.75 * y = 7.50 * (x + y)) : x / y = 2.5 :=
by
  sorry

end rice_mixture_ratio_l150_150747


namespace pythagorean_triple_B_l150_150948

def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_B : isPythagoreanTriple 3 4 5 :=
by
  sorry

end pythagorean_triple_B_l150_150948


namespace distance_between_consecutive_trees_l150_150670

theorem distance_between_consecutive_trees 
  (yard_length : ℕ) (num_trees : ℕ) (tree_at_each_end : yard_length > 0 ∧ num_trees ≥ 2) 
  (equal_distances : ∀ k, k < num_trees - 1 → (yard_length / (num_trees - 1) : ℝ) = 12) :
  yard_length = 360 → num_trees = 31 → (yard_length / (num_trees - 1) : ℝ) = 12 := 
by
  sorry

end distance_between_consecutive_trees_l150_150670


namespace sqrt_meaningful_condition_l150_150804

theorem sqrt_meaningful_condition (a : ℝ) : 2 - a ≥ 0 → a ≤ 2 := by
  sorry

end sqrt_meaningful_condition_l150_150804


namespace triangle_side_length_l150_150142

theorem triangle_side_length (BC : ℝ) (A : ℝ) (B : ℝ) (AB : ℝ) :
  BC = 2 → A = π / 3 → B = π / 4 → AB = (3 * Real.sqrt 2 + Real.sqrt 6) / 3 :=
by
  sorry

end triangle_side_length_l150_150142


namespace triangle_side_inequality_l150_150218

theorem triangle_side_inequality (y : ℕ) (h : 3 < y^2 ∧ y^2 < 19) : 
  y = 2 ∨ y = 3 ∨ y = 4 :=
sorry

end triangle_side_inequality_l150_150218


namespace geese_in_marsh_l150_150655

theorem geese_in_marsh (D : ℝ) (hD : D = 37.0) (G : ℝ) (hG : G = D + 21) : G = 58.0 := 
by 
  sorry

end geese_in_marsh_l150_150655


namespace line_of_intersecting_circles_l150_150658

theorem line_of_intersecting_circles
  (A B : ℝ × ℝ)
  (hAB1 : A.1^2 + A.2^2 + 4 * A.1 - 4 * A.2 = 0)
  (hAB2 : B.1^2 + B.2^2 + 4 * B.1 - 4 * B.2 = 0)
  (hAB3 : A.1^2 + A.2^2 + 2 * A.1 - 12 = 0)
  (hAB4 : B.1^2 + B.2^2 + 2 * B.1 - 12 = 0) :
  ∃ (a b c : ℝ), a * A.1 + b * A.2 + c = 0 ∧ a * B.1 + b * B.2 + c = 0 ∧
                  a = 1 ∧ b = -2 ∧ c = 6 :=
sorry

end line_of_intersecting_circles_l150_150658


namespace mean_proportion_of_3_and_4_l150_150794

theorem mean_proportion_of_3_and_4 : ∃ x : ℝ, 3 / x = x / 4 ∧ (x = 2 * Real.sqrt 3 ∨ x = - (2 * Real.sqrt 3)) :=
by
  sorry

end mean_proportion_of_3_and_4_l150_150794


namespace find_tangent_line_l150_150413

def curve := fun x : ℝ => x^3 + 2 * x + 1
def tangent_point := 1
def tangent_line (x y : ℝ) := 5 * x - y - 1 = 0

theorem find_tangent_line :
  tangent_line tangent_point (curve tangent_point) :=
by
  sorry

end find_tangent_line_l150_150413


namespace line_through_point_parallel_l150_150335

theorem line_through_point_parallel (x y : ℝ) : 
  (∃ c : ℝ, x - 2 * y + c = 0 ∧ ∃ p : ℝ × ℝ, p = (1, 0) ∧ x - 2 * p.2 + c = 0) → (x - 2 * y - 1 = 0) :=
by
  sorry

end line_through_point_parallel_l150_150335


namespace division_proof_l150_150071

theorem division_proof :
  ((2 * 4 * 6) / (1 + 3 + 5 + 7) - (1 * 3 * 5) / (2 + 4 + 6)) / (1 / 2) = 3.5 :=
by
  -- definitions based on conditions
  let numerator1 := 2 * 4 * 6
  let denominator1 := 1 + 3 + 5 + 7
  let numerator2 := 1 * 3 * 5
  let denominator2 := 2 + 4 + 6
  -- the statement of the theorem
  sorry

end division_proof_l150_150071


namespace problem_a_eq_2_problem_a_real_pos_problem_a_real_zero_problem_a_real_neg_l150_150731

theorem problem_a_eq_2 (x : ℝ) : (12 * x^2 - 2 * x > 4) ↔ (x < -1 / 2 ∨ x > 2 / 3) := sorry

theorem problem_a_real_pos (a x : ℝ) (h : a > 0) : (12 * x^2 - a * x > a^2) ↔ (x < -a / 4 ∨ x > a / 3) := sorry

theorem problem_a_real_zero (x : ℝ) : (12 * x^2 > 0) ↔ (x ≠ 0) := sorry

theorem problem_a_real_neg (a x : ℝ) (h : a < 0) : (12 * x^2 - a * x > a^2) ↔ (x < a / 3 ∨ x > -a / 4) := sorry

end problem_a_eq_2_problem_a_real_pos_problem_a_real_zero_problem_a_real_neg_l150_150731


namespace roots_quadratic_identity_l150_150590

theorem roots_quadratic_identity 
  (a b c r s : ℝ)
  (h_root1 : a * r^2 + b * r + c = 0)
  (h_root2 : a * s^2 + b * s + c = 0)
  (h_distinct_roots : r ≠ s)
  : (1 / r^2) + (1 / s^2) = (b^2 - 2 * a * c) / c^2 := 
by
  sorry

end roots_quadratic_identity_l150_150590


namespace case_a_sticks_case_b_square_l150_150057
open Nat 

premise n12 : Nat := 12
premise sticks12_sum : Nat := (n12 * (n12 + 1)) / 2  -- Sum of first 12 natural numbers
premise length_divisibility_4 : ¬ (sticks12_sum % 4 = 0)  -- Check if sum is divisible by 4

-- Need to break at least 2 sticks to form a square
theorem case_a_sticks (h : sticks12_sum = 78) (h2 : length_divisibility_4 = true) : 
  ∃ (k : Nat), k >= 2 := sorry

premise n15 : Nat := 15
premise sticks15_sum : Nat := (n15 * (n15 + 1)) / 2  -- Sum of first 15 natural numbers
premise length_divisibility4_b : sticks15_sum % 4 = 0  -- Check if sum is divisible by 4

-- Possible to form a square without breaking any sticks
theorem case_b_square (h : sticks15_sum = 120) (h2 : length_divisibility4_b = true) : 
  ∃ (k : Nat), k = 0 := sorry

end case_a_sticks_case_b_square_l150_150057


namespace probability_origin_not_in_convex_hull_l150_150046

open ProbabilityTheory
open Set

noncomputable def S1 : Set ℂ := {z : ℂ | complex.abs z = 1}

theorem probability_origin_not_in_convex_hull :
  ∀ (points : Fin 7 → ℂ) (h_points : ∀ i, points i ∈ S1),
  ∃ p : ℝ, p = 57/64 := by
    sorry

end probability_origin_not_in_convex_hull_l150_150046


namespace f_eq_g_l150_150311

noncomputable def f : ℕ → ℕ := sorry
noncomputable def g : ℕ → ℕ := sorry

variable (f_onto : ∀ m : ℕ, ∃ n : ℕ, f n = m)
variable (g_one_one : ∀ m n : ℕ, g m = g n → m = n)
variable (f_ge_g : ∀ n : ℕ, f n ≥ g n)

theorem f_eq_g : f = g :=
sorry

end f_eq_g_l150_150311


namespace jonathan_typing_time_l150_150146

theorem jonathan_typing_time 
(J : ℕ) 
(h_combined_rate : (1 / (J : ℝ)) + (1 / 30) + (1 / 24) = 1 / 10) : 
  J = 40 :=
by {
  sorry
}

end jonathan_typing_time_l150_150146


namespace fish_population_l150_150880

theorem fish_population (N : ℕ) (h1 : 50 > 0) (h2 : N > 0) 
  (tagged_first_catch : ℕ) (total_first_catch : ℕ)
  (tagged_second_catch : ℕ) (total_second_catch : ℕ)
  (h3 : tagged_first_catch = 50)
  (h4 : total_first_catch = 50)
  (h5 : tagged_second_catch = 2)
  (h6 : total_second_catch = 50)
  (h_percent : (tagged_first_catch : ℝ) / (N : ℝ) = (tagged_second_catch : ℝ) / (total_second_catch : ℝ))
  : N = 1250 := 
  by sorry

end fish_population_l150_150880


namespace xy_squared_value_l150_150591

variable {x y : ℝ}

theorem xy_squared_value :
  (y + 6 = (x - 3)^2) ∧ (x + 6 = (y - 3)^2) ∧ (x ≠ y) → (x^2 + y^2 = 25) := 
by
  sorry

end xy_squared_value_l150_150591


namespace line_passes_through_circle_center_l150_150875

theorem line_passes_through_circle_center (a : ℝ) :
  (∃ x y : ℝ, (x^2 + y^2 + 2*x - 4*y = 0) ∧ (3*x + y + a = 0)) → a = 1 :=
by
  sorry

end line_passes_through_circle_center_l150_150875


namespace longest_altitudes_sum_l150_150284

theorem longest_altitudes_sum (a b c : ℕ) (h : a = 6 ∧ b = 8 ∧ c = 10) : 
  let triangle = {a, b, c} in (a + b = 14) :=
by
  sorry  -- Proof goes here

end longest_altitudes_sum_l150_150284


namespace bar_graph_proportion_correct_l150_150983

def white : ℚ := 1/2
def black : ℚ := 1/4
def gray : ℚ := 1/8
def light_gray : ℚ := 1/16

theorem bar_graph_proportion_correct :
  (white = 1 / 2) ∧
  (black = white / 2) ∧
  (gray = black / 2) ∧
  (light_gray = gray / 2) →
  (white = 1 / 2) ∧
  (black = 1 / 4) ∧
  (gray = 1 / 8) ∧
  (light_gray = 1 / 16) :=
by
  intros
  sorry

end bar_graph_proportion_correct_l150_150983


namespace exponential_comparison_l150_150124

theorem exponential_comparison
  (a : ℕ := 3^55)
  (b : ℕ := 4^44)
  (c : ℕ := 5^33) :
  c < a ∧ a < b :=
by
  sorry

end exponential_comparison_l150_150124


namespace count_multiples_4_or_9_but_not_both_l150_150434

theorem count_multiples_4_or_9_but_not_both (n : ℕ) (h : n = 200) :
  let count_multiples (k : ℕ) := (n / k)
  count_multiples 4 + count_multiples 9 - 2 * count_multiples 36 = 62 :=
by
  sorry

end count_multiples_4_or_9_but_not_both_l150_150434


namespace qiantang_tide_facts_l150_150596

noncomputable def f (A ω φ : ℝ) (x : ℝ) := A * Real.sin (ω * x + φ)

noncomputable def f_prime (A ω φ : ℝ) (x : ℝ) := A * ω * Real.cos (ω * x + φ)

theorem qiantang_tide_facts (A ω φ : ℝ) (hA : 0 < A) (hω : 0 < ω) (hφ : |φ| < Real.pi / 3)
  (h1 : f A ω φ (2 * Real.pi) = f_prime A ω φ (2 * Real.pi))
  (h2 : ∀ x, f_prime A ω φ x ≥ -4) :
  (f A ω φ (Real.pi / 3) = Real.sqrt 6 + Real.sqrt 2) ∧
  (Real.Even.fun (λ x, f_prime A ω φ (x - Real.pi / 4))) := 
  sorry

end qiantang_tide_facts_l150_150596


namespace yi_wins_probability_l150_150753

open Finset

theorem yi_wins_probability :
  let pJia := 2/3,
      pYi := 1/3 in
  ( ∑ k in range 3, nat.choose 5 k * (pYi ^ k) * (pJia ^ (5 - k)) )
  + ( ∑ k in range 1, nat.choose 5 (k + 3) * (pYi ^ (k + 3)) * (pJia ^ (5 - (k + 3))) )
  = 17/81 :=
by
  sorry

end yi_wins_probability_l150_150753


namespace tessa_debt_l150_150865

theorem tessa_debt :
  let initial_debt : ℤ := 40 in
  let repayment : ℤ := initial_debt / 2 in
  let debt_after_repayment : ℤ := initial_debt - repayment in
  let additional_debt : ℤ := 10 in
  debt_after_repayment + additional_debt = 30 :=
by
  -- The proof goes here.
  sorry

end tessa_debt_l150_150865


namespace total_coins_l150_150631

def piles_of_quarters : Nat := 5
def piles_of_dimes : Nat := 5
def coins_per_pile : Nat := 3

theorem total_coins :
  (piles_of_quarters * coins_per_pile) + (piles_of_dimes * coins_per_pile) = 30 := by
  sorry

end total_coins_l150_150631


namespace sum_of_xyz_l150_150286

noncomputable def log_base (b a : ℝ) := Real.log a / Real.log b

theorem sum_of_xyz :
  ∃ x y z : ℝ,
  log_base 3 (log_base 4 (log_base 5 x)) = 0 ∧
  log_base 4 (log_base 5 (log_base 3 y)) = 0 ∧
  log_base 5 (log_base 3 (log_base 4 z)) = 0 ∧
  x + y + z = 932 :=
by
  sorry

end sum_of_xyz_l150_150286


namespace range_of_f_l150_150896

noncomputable def f (x : ℝ) : ℝ :=
  (Real.arccos x)^4 + (Real.arcsin x)^4

theorem range_of_f :
  ∀ y, (∃ x, x ∈ Set.Icc (-1:ℝ) 1 ∧ f x = y) ↔ y ∈ Set.Icc 0 (Real.pi^4 / 8) :=
sorry

end range_of_f_l150_150896


namespace outer_boundary_diameter_l150_150210

def width_jogging_path : ℝ := 10
def width_vegetable_garden : ℝ := 12
def diameter_pond : ℝ := 20

theorem outer_boundary_diameter :
  2 * (diameter_pond / 2 + width_vegetable_garden + width_jogging_path) = 64 := by
  sorry

end outer_boundary_diameter_l150_150210


namespace andrew_age_l150_150828

variables (a g : ℕ)

theorem andrew_age : 
  (g = 16 * a) ∧ (g - a = 60) → a = 4 := by
  sorry

end andrew_age_l150_150828


namespace min_distance_A_D_l150_150481

theorem min_distance_A_D (A B C E D : Type) 
  (d_AB d_BC d_CE d_ED : ℝ) 
  (h1 : d_AB = 12) 
  (h2 : d_BC = 7) 
  (h3 : d_CE = 2) 
  (h4 : d_ED = 5) : 
  ∃ d_AD : ℝ, d_AD = 2 := 
by
  sorry

end min_distance_A_D_l150_150481


namespace angle_between_sides_of_triangle_l150_150863

noncomputable def right_triangle_side_lengths1 : Nat × Nat × Nat := (15, 36, 39)
noncomputable def right_triangle_side_lengths2 : Nat × Nat × Nat := (40, 42, 58)

-- Assuming both triangles are right triangles
def is_right_triangle (a b c : Nat) : Prop := a^2 + b^2 = c^2

theorem angle_between_sides_of_triangle
  (h1 : is_right_triangle 15 36 39)
  (h2 : is_right_triangle 40 42 58) : 
  ∃ (θ : ℝ), θ = 90 :=
by
  sorry

end angle_between_sides_of_triangle_l150_150863


namespace find_n_between_50_and_150_l150_150991

theorem find_n_between_50_and_150 :
  ∃ (n : ℤ), 50 ≤ n ∧ n ≤ 150 ∧
  n % 7 = 0 ∧ 
  n % 9 = 3 ∧ 
  n % 6 = 3 ∧ 
  n % 4 = 1 ∧
  n = 105 :=
by
  sorry

end find_n_between_50_and_150_l150_150991


namespace number_of_outfits_l150_150634

-- Definitions based on conditions a)
def trousers : ℕ := 5
def shirts : ℕ := 7
def jackets : ℕ := 3
def specific_trousers : ℕ := 2
def specific_jackets : ℕ := 2

-- Lean 4 theorem statement to prove the number of outfits
theorem number_of_outfits (trousers shirts jackets specific_trousers specific_jackets : ℕ) :
  (3 * jackets + specific_trousers * specific_jackets) * shirts = 91 :=
by
  sorry

end number_of_outfits_l150_150634


namespace weight_of_NH4I_H2O_l150_150693

noncomputable def total_weight (moles_NH4I : ℕ) (molar_mass_NH4I : ℝ) 
                             (moles_H2O : ℕ) (molar_mass_H2O : ℝ) : ℝ :=
  (moles_NH4I * molar_mass_NH4I) + (moles_H2O * molar_mass_H2O)

theorem weight_of_NH4I_H2O :
  total_weight 15 144.95 7 18.02 = 2300.39 :=
by
  sorry

end weight_of_NH4I_H2O_l150_150693


namespace sum_of_digits_of_n_l150_150149

theorem sum_of_digits_of_n :
  ∃ n : ℕ,
    n > 2000 ∧
    n + 135 % 75 = 15 ∧
    n + 75 % 135 = 45 ∧
    (n = 2025 ∧ (2 + 0 + 2 + 5 = 9)) :=
by
  sorry

end sum_of_digits_of_n_l150_150149


namespace measure_of_U_is_120_l150_150887

variable {α β γ δ ε ζ : ℝ}
variable (h1 : α = γ) (h2 : α = ζ) (h3 : β + δ = 180) (h4 : ε + ζ = 180)

noncomputable def measure_of_U : ℝ :=
  let total_sum := 720
  have sum_of_angles : α + β + γ + δ + ζ + ε = total_sum := by
    sorry
  have subs_suppl_G_R : β + δ = 180 := h3
  have subs_suppl_E_U : ε + ζ = 180 := h4
  have congruent_F_I_U : α = γ ∧ α = ζ := ⟨h1, h2⟩
  let α : ℝ := sorry
  α

theorem measure_of_U_is_120 : measure_of_U h1 h2 h3 h4 = 120 :=
  sorry

end measure_of_U_is_120_l150_150887


namespace arithmetic_sequence_general_formula_bn_sequence_sum_l150_150141

/-- 
  In an arithmetic sequence {a_n}, a_2 = 5 and a_6 = 21. 
  Prove the general formula for the nth term a_n and the sum of the first n terms S_n. 
-/
theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : a 2 = 5) (h2 : a 6 = 21) : 
  (∀ n, a n = 4 * n - 3) ∧ (∀ n, S n = n * (2 * n - 1)) := 
sorry

/--
  Given b_n = 2 / (S_n + 5 * n), prove the sum of the first n terms T_n for the sequence {b_n}.
-/
theorem bn_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (b : ℕ → ℚ) (T : ℕ → ℚ) 
  (h1 : a 2 = 5) (h2 : a 6 = 21) 
  (ha : ∀ n, a n = 4 * n - 3) (hS : ∀ n, S n = n * (2 * n - 1)) 
  (hb : ∀ n, b n = 2 / (S n + 5 * n)) : 
  (∀ n, T n = 3 / 4 - 1 / (2 * (n + 1)) - 1 / (2 * (n + 2))) :=
sorry

end arithmetic_sequence_general_formula_bn_sequence_sum_l150_150141


namespace central_angle_of_sector_l150_150678

theorem central_angle_of_sector (R θ l : ℝ) (h1 : 2 * R + l = π * R) : θ = π - 2 := 
by
  sorry

end central_angle_of_sector_l150_150678


namespace square_with_12_sticks_square_with_15_sticks_l150_150055

-- Definitions for problem conditions
def sum_of_first_n_natural_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def can_form_square (total_length : ℕ) : Prop :=
  total_length % 4 = 0

-- Given n = 12, check if breaking 2 sticks is required to form a square
theorem square_with_12_sticks : (n = 12) → ¬ can_form_square (sum_of_first_n_natural_numbers 12) → true :=
by
  intros
  sorry

-- Given n = 15, check if it is possible to form a square without breaking any sticks
theorem square_with_15_sticks : (n = 15) → can_form_square (sum_of_first_n_natural_numbers 15) → true :=
by
  intros
  sorry

end square_with_12_sticks_square_with_15_sticks_l150_150055


namespace infinitely_many_positive_integers_l150_150989

theorem infinitely_many_positive_integers (k : ℕ) (m := 13 * k + 1) (h : m ≠ 8191) :
  8191 = 2 ^ 13 - 1 → ∃ (m : ℕ), ∀ k : ℕ, (13 * k + 1) ≠ 8191 ∧ ∃ (t : ℕ), (2 ^ (13 * k) - 1) = 8191 * m * t := by
  intros
  sorry

end infinitely_many_positive_integers_l150_150989


namespace cost_of_concessions_l150_150842

theorem cost_of_concessions (total_cost : ℕ) (adult_ticket_cost : ℕ) (child_ticket_cost : ℕ) (num_adults : ℕ) (num_children : ℕ) :
  total_cost = 76 →
  adult_ticket_cost = 10 →
  child_ticket_cost = 7 →
  num_adults = 5 →
  num_children = 2 →
  total_cost - (num_adults * adult_ticket_cost + num_children * child_ticket_cost) = 12 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end cost_of_concessions_l150_150842


namespace quotient_remainder_difference_l150_150088

theorem quotient_remainder_difference :
  ∀ (N Q Q' R : ℕ), 
    N = 75 →
    N = 5 * Q →
    N = 34 * Q' + R →
    Q > R →
    Q - R = 8 :=
by
  intros N Q Q' R hN hDiv5 hDiv34 hGt
  sorry

end quotient_remainder_difference_l150_150088


namespace cos_D_zero_l150_150026

noncomputable def area_of_triangle (a b: ℝ) (sinD: ℝ) : ℝ := 1 / 2 * a * b * sinD

theorem cos_D_zero (DE DF : ℝ) (D : ℝ) (h1 : area_of_triangle DE DF (Real.sin D) = 98) (h2 : Real.sqrt (DE * DF) = 14) : Real.cos D = 0 :=
  by
  sorry

end cos_D_zero_l150_150026


namespace monday_to_sunday_ratio_l150_150477

-- Define the number of pints Alice bought on Sunday
def sunday_pints : ℕ := 4

-- Define the number of pints Alice bought on Monday as a multiple of Sunday
def monday_pints (k : ℕ) : ℕ := 4 * k

-- Define the number of pints Alice bought on Tuesday
def tuesday_pints (k : ℕ) : ℚ := (4 * k) / 3

-- Define the number of pints Alice returned on Wednesday
def wednesday_return (k : ℕ) : ℚ := (2 * k) / 3

-- Define the total number of pints Alice had on Wednesday before returning the expired ones
def total_pre_return (k : ℕ) : ℚ := 18 + (2 * k) / 3

-- Define the total number of pints purchased from Sunday to Tuesday
def total_pints (k : ℕ) : ℚ := 4 + 4 * k + (4 * k) / 3

-- The statement to be proven
theorem monday_to_sunday_ratio : ∃ k : ℕ, 
  (4 * k + (4 * k) / 3 + 4 = 18 + (2 * k) / 3) ∧
  (4 * k) / 4 = 3 :=
by 
  sorry

end monday_to_sunday_ratio_l150_150477


namespace incorrect_simplification_l150_150073

theorem incorrect_simplification :
  (-(1 + 1/2) ≠ 1 + 1/2) := 
by sorry

end incorrect_simplification_l150_150073


namespace cookie_radius_and_area_l150_150003

def boundary_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8 = 2 * x + 4 * y

theorem cookie_radius_and_area :
  (∃ r : ℝ, r = Real.sqrt 13) ∧ (∃ A : ℝ, A = 13 * Real.pi) :=
by
  sorry

end cookie_radius_and_area_l150_150003


namespace roots_eq_202_l150_150625

theorem roots_eq_202 (p q : ℝ) 
  (h1 : ∀ x : ℝ, ((x + p) * (x + q) * (x + 10) = 0 ↔ (x = -p ∨ x = -q ∨ x = -10)) ∧ 
       ∀ x : ℝ, ((x + 5) ^ 2 = 0 ↔ x = -5)) 
  (h2 : ∀ x : ℝ, ((x + 2 * p) * (x + 4) * (x + 8) = 0 ↔ (x = -2 * p ∨ x = -4 ∨ x = -8)) ∧ 
       ∀ x : ℝ, ((x + q) * (x + 10) = 0 ↔ (x = -q ∨ x = -10))) 
  (hpq : p = q) (neq_5 : q ≠ 5) (p_2 : p = 2):
  100 * p + q = 202 := sorry

end roots_eq_202_l150_150625


namespace pens_bought_is_17_l150_150783

def number_of_pens_bought (C S : ℝ) (bought_pens : ℝ) : Prop :=
  (bought_pens * C = 12 * S) ∧ (0.4 = (S - C) / C)

theorem pens_bought_is_17 (C S : ℝ) (bought_pens : ℝ) 
  (h1 : bought_pens * C = 12 * S)
  (h2 : 0.4 = (S - C) / C) :
  bought_pens = 17 :=
sorry

end pens_bought_is_17_l150_150783


namespace andrew_eggs_count_l150_150545

def cost_of_toast (num_toasts : ℕ) : ℕ :=
  num_toasts * 1

def cost_of_eggs (num_eggs : ℕ) : ℕ :=
  num_eggs * 3

def total_cost (num_toasts : ℕ) (num_eggs : ℕ) : ℕ :=
  cost_of_toast num_toasts + cost_of_eggs num_eggs

theorem andrew_eggs_count (E : ℕ) (H1 : total_cost 2 2 = 8)
                       (H2 : total_cost 1 E + 8 = 15) : E = 2 := by
  sorry

end andrew_eggs_count_l150_150545


namespace no_solutions_iff_a_positive_and_discriminant_non_positive_l150_150452

theorem no_solutions_iff_a_positive_and_discriminant_non_positive (a b c : ℝ) (h1 : a ≠ 0) :
  (∀ x : ℝ, ¬ (a * x^2 + b * x + c < 0)) ↔ (a > 0 ∧ (b^2 - 4 * a * c) ≤ 0) :=
  sorry

end no_solutions_iff_a_positive_and_discriminant_non_positive_l150_150452


namespace students_like_all_three_l150_150712

variables (N : ℕ) (r : ℚ) (j : ℚ) (o : ℕ) (n : ℕ)

-- Number of students in the class
def num_students := N = 40

-- Fraction of students who like Rock
def fraction_rock := r = 1/4

-- Fraction of students who like Jazz
def fraction_jazz := j = 1/5

-- Number of students who like other genres
def num_other_genres := o = 8

-- Number of students who do not like any of the three genres
def num_no_genres := n = 6

---- Proof theorem
theorem students_like_all_three
  (h1 : num_students N)
  (h2 : fraction_rock r)
  (h3 : fraction_jazz j)
  (h4 : num_other_genres o)
  (h5 : num_no_genres n) :
  ∃ z : ℕ, z = 2 := 
sorry

end students_like_all_three_l150_150712


namespace ζn_converges_in_prob_to_c_ξn_ζn_pair_converges_in_distr_expected_val_converges_ξn_mult_ζn_converges_in_distr_l150_150668

open MeasureTheory ProbabilityTheory TopologicalSpace

variables {α : Type*} {β : Type*} {γ : Type*} 
          {F : Type*} {P : Type*} {G : Type*}

noncomputable theory

-- Assuming the distributions and convergence as given conditions.
variables (ζn : ℕ → α) (c : α)
variables (ξn : ℕ → β) (ξ : β)

-- Condition: ζn converges in distribution to c.
axiom ζn_converges_to_c : ∀ (ε : ℝ), ∃ (N : ℕ), ∀ (n ≥ N), 
  |ζn n - c| ≤ ε

-- Condition: ξn converges in distribution to ξ.
axiom ξn_converges_to_ξ: ∀ (ε : ℝ), ∃ (N : ℕ), ∀ (n ≥ N), 
  |ξn n - ξ| ≤ ε

-- Statement: ζn converges in probability to c.
theorem ζn_converges_in_prob_to_c : ℕ → ℝ

-- Statement: ξn, ζn pair converges in distribution to (ξ, c).
theorem ξn_ζn_pair_converges_in_distr : ∀ (f : α × β → ℝ) (hf : continuous f ∧ ∃ C, ∀ x, ∥f x∥ ≤ C), 
  lintegral (ξn × ζn) f = lintegral (ξ × c) f

-- Statement: E f(ξn, ζn) converges to E f(ξ, c) for any continuous bounded function f.
theorem expected_val_converges : ∀ (f : α × β → ℝ) (hf : continuous f ∧ ∃ C, ∀ x, ∥f x∥ ≤ C), 
  (∫ x in measure_space α, f (ξn x, ζn x)) = (∫ x in measure_space α, f (ξ x, c))

-- Statement: ξn · ζn converges in distribution to c · ξ.
theorem ξn_mult_ζn_converges_in_distr : 
  (ξn × ζn) = (ξ × c)

sorry -- Proofs are omitted as per instructions.

end ζn_converges_in_prob_to_c_ξn_ζn_pair_converges_in_distr_expected_val_converges_ξn_mult_ζn_converges_in_distr_l150_150668


namespace daniel_stickers_l150_150098

def stickers_data 
    (total_stickers : Nat)
    (fred_extra : Nat)
    (andrew_kept : Nat) : Prop :=
  total_stickers = 750 ∧ fred_extra = 120 ∧ andrew_kept = 130

theorem daniel_stickers (D : Nat) :
  stickers_data 750 120 130 → D + (D + 120) = 750 - 130 → D = 250 :=
by
  intros h_data h_eq
  sorry

end daniel_stickers_l150_150098


namespace remainder_when_divided_by_63_l150_150072

theorem remainder_when_divided_by_63 (x : ℤ) (h1 : ∃ q : ℤ, x = 63 * q + r ∧ 0 ≤ r ∧ r < 63) (h2 : ∃ k : ℤ, x = 9 * k + 2) :
  ∃ r : ℤ, 0 ≤ r ∧ r < 63 ∧ r = 7 :=
by
  sorry

end remainder_when_divided_by_63_l150_150072


namespace original_number_divisible_by_3_l150_150464

theorem original_number_divisible_by_3:
  ∃ (a b c d e f g h : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h) ∧
  (b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h) ∧
  (c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h) ∧
  (d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h) ∧
  (e ≠ f ∧ e ≠ g ∧ e ≠ h) ∧
  (f ≠ g ∧ f ≠ h) ∧
  (g ≠ h) ∧ 
  (a + b + c + b + d + e + f + e + g + d + h) % 3 = 0 :=
sorry

end original_number_divisible_by_3_l150_150464


namespace total_marks_l150_150226

variable (A M SS Mu : ℝ)

-- Conditions
def cond1 : Prop := M = A - 20
def cond2 : Prop := SS = Mu + 10
def cond3 : Prop := Mu = 70
def cond4 : Prop := M = (9 / 10) * A

-- Theorem statement
theorem total_marks (A M SS Mu : ℝ) (h1 : cond1 A M)
                                      (h2 : cond2 SS Mu)
                                      (h3 : cond3 Mu)
                                      (h4 : cond4 A M) :
    A + M + SS + Mu = 530 :=
by 
  sorry

end total_marks_l150_150226


namespace invitees_count_l150_150638

theorem invitees_count 
  (packages : ℕ) 
  (weight_per_package : ℕ) 
  (weight_per_burger : ℕ) 
  (total_people : ℕ)
  (H1 : packages = 4)
  (H2 : weight_per_package = 5)
  (H3 : weight_per_burger = 2)
  (H4 : total_people + 1 = (packages * weight_per_package) / weight_per_burger) :
  total_people = 9 := 
by
  sorry

end invitees_count_l150_150638


namespace express_as_scientific_notation_l150_150455

-- Definitions
def billion : ℝ := 10^9
def amount : ℝ := 850 * billion

-- Statement
theorem express_as_scientific_notation : amount = 8.5 * 10^11 :=
by
  sorry

end express_as_scientific_notation_l150_150455


namespace range_of_a_l150_150556

theorem range_of_a (a : ℝ) : (∃ x₀ : ℝ, x₀^2 + (a-1)*x₀ + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by
  sorry

end range_of_a_l150_150556


namespace smallest_n_for_terminating_decimal_l150_150941

-- Theorem follows the tuple of (question, conditions, correct answer)
theorem smallest_n_for_terminating_decimal (n : ℕ) (h : ∃ k : ℕ, n + 75 = 2^k ∨ n + 75 = 5^k ∨ n + 75 = (2^k * 5^k)) :
  n = 50 :=
by
  sorry -- Proof is omitted

end smallest_n_for_terminating_decimal_l150_150941


namespace find_side_c_and_area_S_find_sinA_plus_cosB_l150_150878

-- Definitions for the conditions given
structure Triangle :=
  (a b c : ℝ)
  (angleA angleB angleC : ℝ)

noncomputable def givenTriangle : Triangle :=
  { a := 2, b := 4, c := 2 * Real.sqrt 3, angleA := 30, angleB := 90, angleC := 60 }

-- Prove the length of side c and the area S
theorem find_side_c_and_area_S (t : Triangle) (h : t = givenTriangle) :
  t.c = 2 * Real.sqrt 3 ∧ (1 / 2) * t.a * t.b * Real.sin (t.angleC * Real.pi / 180) = 2 * Real.sqrt 3 :=
by
  sorry

-- Prove the value of sin A + cos B
theorem find_sinA_plus_cosB (t : Triangle) (h : t = givenTriangle) :
  Real.sin (t.angleA * Real.pi / 180) + Real.cos (t.angleB * Real.pi / 180) = 1 / 2 :=
by
  sorry

end find_side_c_and_area_S_find_sinA_plus_cosB_l150_150878


namespace urn_gold_coins_percent_l150_150097

theorem urn_gold_coins_percent (perc_beads : ℝ) (perc_silver_coins : ℝ) (perc_gold_coins : ℝ) :
  perc_beads = 0.2 →
  perc_silver_coins = 0.4 →
  perc_gold_coins = 0.48 :=
by
  intros h1 h2
  sorry

end urn_gold_coins_percent_l150_150097


namespace math_problem_l150_150751

theorem math_problem :
  let result := 83 - 29
  let final_sum := result + 58
  let rounded := if final_sum % 10 < 5 then final_sum - final_sum % 10 else final_sum + (10 - final_sum % 10)
  rounded = 110 := by
  sorry

end math_problem_l150_150751


namespace expected_value_of_twelve_sided_die_l150_150825

theorem expected_value_of_twelve_sided_die : 
  let face_values := finset.range (12 + 1) \ finset.singleton 0 in
  (finset.sum face_values (λ x, x) : ℝ) / 12 = 6.5 :=
by
  sorry

end expected_value_of_twelve_sided_die_l150_150825


namespace max_value_x_div_y_l150_150151

variables {x y a b : ℝ}

theorem max_value_x_div_y (h1 : x ≥ y) (h2 : y > 0) (h3 : 0 ≤ a) (h4 : a ≤ x) (h5 : 0 ≤ b) (h6 : b ≤ y) 
  (h7 : (x - a)^2 + (y - b)^2 = x^2 + b^2) (h8 : x^2 + b^2 = y^2 + a^2) :
  x / y ≤ (2 * Real.sqrt 3) / 3 :=
sorry

end max_value_x_div_y_l150_150151


namespace find_m_range_a_l150_150859

noncomputable def f (x m : ℝ) : ℝ :=
  m - |x - 3|

theorem find_m (m : ℝ) (h : ∀ x, 2 < f x m ↔ 2 < x ∧ x < 4) : m = 3 :=
  sorry

theorem range_a (a : ℝ) (h : ∀ x, |x - a| ≥ f x 3) : a ≤ 0 ∨ 6 ≤ a :=
  sorry

end find_m_range_a_l150_150859


namespace gain_percentage_is_15_l150_150086

-- Initial conditions
def CP_A : ℤ := 100
def CP_B : ℤ := 200
def CP_C : ℤ := 300
def SP_A : ℤ := 110
def SP_B : ℤ := 250
def SP_C : ℤ := 330

-- Definitions for total values
def Total_CP : ℤ := CP_A + CP_B + CP_C
def Total_SP : ℤ := SP_A + SP_B + SP_C
def Overall_gain : ℤ := Total_SP - Total_CP
def Gain_percentage : ℚ := (Overall_gain * 100) / Total_CP

-- Theorem to prove the overall gain percentage
theorem gain_percentage_is_15 :
  Gain_percentage = 15 := 
by
  -- Proof placeholder
  sorry

end gain_percentage_is_15_l150_150086


namespace add_alcohol_solve_l150_150082

variable (x : ℝ)

def initial_solution_volume : ℝ := 6
def initial_alcohol_fraction : ℝ := 0.20
def desired_alcohol_fraction : ℝ := 0.50

def initial_alcohol_content : ℝ := initial_alcohol_fraction * initial_solution_volume
def total_solution_volume_after_addition : ℝ := initial_solution_volume + x
def total_alcohol_content_after_addition : ℝ := initial_alcohol_content + x

theorem add_alcohol_solve (x : ℝ) :
  (initial_alcohol_content + x) / (initial_solution_volume + x) = desired_alcohol_fraction →
  x = 3.6 :=
by
  sorry

end add_alcohol_solve_l150_150082


namespace decimal_representation_prime_has_zeros_l150_150717

theorem decimal_representation_prime_has_zeros (p : ℕ) [Fact (Nat.Prime p)] : 
  ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, 10^2002 ∣ p^n * 10^k :=
sorry

end decimal_representation_prime_has_zeros_l150_150717


namespace complement_union_in_universe_l150_150188

variable (U : Set ℕ := {1, 2, 3, 4, 5})
variable (M : Set ℕ := {1, 3})
variable (N : Set ℕ := {1, 2})

theorem complement_union_in_universe :
  (U \ (M ∪ N)) = {4, 5} :=
by
  sorry

end complement_union_in_universe_l150_150188


namespace part_a_part_b_l150_150370

-- the conditions
variables (r R x : ℝ) (h_rltR : r < R)
variables (h_x : x = (R - r) / 2)
variables (h1 : 0 < x)
variables (h12_circles : ∀ i : ℕ, i ∈ Finset.range 12 → ∃ c_i : ℝ × ℝ, True)  -- Informal way to note 12 circles of radius x are placed

-- prove each part
theorem part_a (r R : ℝ) (h_rltR : r < R) : x = (R - r) / 2 :=
sorry

theorem part_b (r R : ℝ) (h_rltR : r < R) (h_x : x = (R - r) / 2) :
  (R / r) = (4 + Real.sqrt 6 - Real.sqrt 2) / (4 - Real.sqrt 6 + Real.sqrt 2) :=
sorry

end part_a_part_b_l150_150370


namespace area_contained_by_graph_l150_150563

theorem area_contained_by_graph (x y : ℝ) :
  (|x + y| + |x - y| ≤ 6) → (36 = 36) := by
  sorry

end area_contained_by_graph_l150_150563


namespace score_standard_deviation_l150_150009

theorem score_standard_deviation (mean std_dev : ℝ)
  (h1 : mean = 76)
  (h2 : mean - 2 * std_dev = 60) :
  100 = mean + 3 * std_dev :=
by
  -- Insert proof here
  sorry

end score_standard_deviation_l150_150009


namespace find_m_l150_150575

-- Define the vectors a and b and the condition for parallelicity
def a : ℝ × ℝ := (2, 1)
def b (m : ℝ) : ℝ × ℝ := (m, 2)
def parallel (u v : ℝ × ℝ) := u.1 * v.2 = u.2 * v.1

-- State the theorem with the given conditions and required proof goal
theorem find_m (m : ℝ) (h : parallel a (b m)) : m = 4 :=
by sorry  -- skipping proof

end find_m_l150_150575


namespace product_of_solutions_abs_eq_l150_150414

theorem product_of_solutions_abs_eq (x1 x2 : ℝ) (h1 : |2 * x1 - 1| + 4 = 24) (h2 : |2 * x2 - 1| + 4 = 24) : x1 * x2 = -99.75 := 
sorry

end product_of_solutions_abs_eq_l150_150414


namespace no_such_integers_and_function_l150_150407

theorem no_such_integers_and_function (f : ℝ → ℝ) (m n : ℤ) (h1 : ∀ x, f (f x) = 2 * f x - x - 2) (h2 : (m : ℝ) ≤ (n : ℝ) ∧ f m = n) : False :=
sorry

end no_such_integers_and_function_l150_150407


namespace find_y_l150_150139

theorem find_y (x y : ℝ) (h1 : x = 4 * y) (h2 : (1 / 2) * x = 1) : y = 1 / 2 :=
by
  sorry

end find_y_l150_150139


namespace eq_exponents_l150_150572

theorem eq_exponents (m n : ℤ) : ((5 + 3 * Real.sqrt 2) ^ m = (3 + 5 * Real.sqrt 2) ^ n) → (m = 0 ∧ n = 0) :=
by
  sorry

end eq_exponents_l150_150572


namespace function_C_is_odd_and_decreasing_l150_150519

-- Conditions
def f (x : ℝ) : ℝ := -x^3 - x

-- Odd function condition
def is_odd (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

-- Strictly decreasing condition
def is_strictly_decreasing (f : ℝ → ℝ) : Prop :=
∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2

-- The theorem we want to prove
theorem function_C_is_odd_and_decreasing : 
  is_odd f ∧ is_strictly_decreasing f :=
by
  sorry

end function_C_is_odd_and_decreasing_l150_150519


namespace shorter_side_of_rectangular_room_l150_150540

theorem shorter_side_of_rectangular_room 
  (a b : ℕ) 
  (h1 : 2 * a + 2 * b = 52) 
  (h2 : a * b = 168) : 
  min a b = 12 := 
  sorry

end shorter_side_of_rectangular_room_l150_150540


namespace find_k_l150_150402

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 1 / x + 5
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := 2 * x^2 - k

theorem find_k (k : ℝ) : 
  (f 3 - g 3 k = 6) → k = -23/3 := 
by
  sorry

end find_k_l150_150402


namespace combinations_9_choose_3_l150_150296

theorem combinations_9_choose_3 : (nat.choose 9 3) = 84 :=
by
  sorry

end combinations_9_choose_3_l150_150296


namespace eval_sqrt4_8_pow12_l150_150106

-- Define the fourth root of 8
def fourthRootOfEight : ℝ := 8 ^ (1 / 4)

-- Define the original expression
def expr := (fourthRootOfEight) ^ 12

-- The theorem to prove
theorem eval_sqrt4_8_pow12: expr = 512 := by
  sorry

end eval_sqrt4_8_pow12_l150_150106


namespace eval_sqrt4_8_pow12_l150_150107

-- Define the fourth root of 8
def fourthRootOfEight : ℝ := 8 ^ (1 / 4)

-- Define the original expression
def expr := (fourthRootOfEight) ^ 12

-- The theorem to prove
theorem eval_sqrt4_8_pow12: expr = 512 := by
  sorry

end eval_sqrt4_8_pow12_l150_150107


namespace algebraic_expression_value_l150_150850

variables (m n x y : ℤ)

def condition1 := m - n = 100
def condition2 := x + y = -1

theorem algebraic_expression_value :
  condition1 m n → condition2 x y → (n + x) - (m - y) = -101 :=
by
  intro h1 h2
  sorry

end algebraic_expression_value_l150_150850


namespace associates_more_than_two_years_l150_150458

-- Definitions based on the given conditions
def total_associates := 100
def second_year_associates_percent := 25
def not_first_year_associates_percent := 75

-- The theorem to prove
theorem associates_more_than_two_years :
  not_first_year_associates_percent - second_year_associates_percent = 50 :=
by
  -- The proof is omitted
  sorry

end associates_more_than_two_years_l150_150458


namespace doughnut_cost_l150_150906

theorem doughnut_cost:
  ∃ (D C : ℝ), 
    3 * D + 4 * C = 4.91 ∧ 
    5 * D + 6 * C = 7.59 ∧ 
    D = 0.45 :=
by
  sorry

end doughnut_cost_l150_150906


namespace men_build_wall_l150_150605

theorem men_build_wall (k : ℕ) (h1 : 20 * 6 = k) : ∃ d : ℝ, (30 * d = k) ∧ d = 4.0 := by
  sorry

end men_build_wall_l150_150605


namespace simplify_expression_l150_150637

theorem simplify_expression : 
  (1 / (1 / (1 / 3)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3 + 1 / (1 / 3)^4)) = 1 / 120 := 
by 
  sorry

end simplify_expression_l150_150637


namespace min_sticks_to_break_for_square_12_can_form_square_without_breaking_15_l150_150058

-- Part (a): For n = 12:
theorem min_sticks_to_break_for_square_12 : ∀ (n : ℕ), n = 12 → 
  (∃ (sticks : Finset ℕ), sticks.card = 12 ∧ sticks.sum id = 78 ∧ (¬ (78 % 4 = 0) → 
  ∃ (b : ℕ), b = 2)) := 
by sorry

-- Part (b): For n = 15:
theorem can_form_square_without_breaking_15 : ∀ (n : ℕ), n = 15 → 
  (∃ (sticks : Finset ℕ), sticks.card = 15 ∧ sticks.sum id = 120 ∧ (120 % 4 = 0)) :=
by sorry

end min_sticks_to_break_for_square_12_can_form_square_without_breaking_15_l150_150058


namespace arithmetic_sequence_ratio_l150_150997

open Nat

noncomputable def S (n : ℕ) : ℝ := n^2
noncomputable def T (n : ℕ) : ℝ := n * (2 * n + 3)

theorem arithmetic_sequence_ratio 
  (h : ∀ n : ℕ, (2 * n + 3) * S n = n * T n) : 
  (S 5 - S 4) / (T 6 - T 5) = 9 / 25 := by
  sorry

end arithmetic_sequence_ratio_l150_150997


namespace no_obtuse_triangle_probability_l150_150710

-- Define the problem in Lean 4

noncomputable def probability_no_obtuse_triangles (points : Fin 4 → ℝ) : ℝ :=
  -- Assuming the points are uniformly distributed and simplifying the problem calculation as suggested
  (3 / 8) ^ 6

-- Statement of the problem
theorem no_obtuse_triangle_probability : ∃ (points : Fin 4 → ℝ), probability_no_obtuse_triangles points = (3 / 8) ^ 6 := 
by 
  sorry -- Proof is omitted

end no_obtuse_triangle_probability_l150_150710


namespace no_integer_solutions_3a2_eq_b2_plus_1_l150_150560

theorem no_integer_solutions_3a2_eq_b2_plus_1 : 
  ¬ ∃ a b : ℤ, 3 * a^2 = b^2 + 1 :=
by
  intro h
  obtain ⟨a, b, hab⟩ := h
  sorry

end no_integer_solutions_3a2_eq_b2_plus_1_l150_150560


namespace train_speed_is_60_kmph_l150_150683

noncomputable def train_length : ℝ := 110
noncomputable def time_to_pass_man : ℝ := 5.999520038396929
noncomputable def man_speed_kmph : ℝ := 6

theorem train_speed_is_60_kmph :
  let man_speed_mps := man_speed_kmph * (1000 / 3600)
  let relative_speed := train_length / time_to_pass_man
  let train_speed_mps := relative_speed - man_speed_mps
  let train_speed_kmph := train_speed_mps * (3600 / 1000)
  train_speed_kmph = 60 :=
by
  sorry

end train_speed_is_60_kmph_l150_150683


namespace hawksbill_to_green_turtle_ratio_l150_150229

theorem hawksbill_to_green_turtle_ratio (total_turtles : ℕ) (green_turtles : ℕ) (hawksbill_turtles : ℕ) (h1 : green_turtles = 800) (h2 : total_turtles = 3200) (h3 : hawksbill_turtles = total_turtles - green_turtles) :
  hawksbill_turtles / green_turtles = 3 :=
by {
  sorry
}

end hawksbill_to_green_turtle_ratio_l150_150229


namespace subsets_with_mean_six_l150_150132

open Finset

def originalSet : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12}

def targetSumRemoved : ℕ := 19
def targetMeanRemaining : ℕ := 6

theorem subsets_with_mean_six :
  (originalSet.filter (λ s, s.card = 3 ∧ s.sum = targetSumRemoved)).card = 4 := sorry

end subsets_with_mean_six_l150_150132


namespace second_largest_between_28_and_31_l150_150940

theorem second_largest_between_28_and_31 : 
  ∃ (n : ℕ), n > 28 ∧ n ≤ 31 ∧ (∀ m, (m > 28 ∧ m ≤ 31 ∧ m < 31) ->  m ≤ 30) :=
sorry

end second_largest_between_28_and_31_l150_150940


namespace joe_paint_usage_l150_150145

noncomputable def paint_used_after_four_weeks : ℝ := 
  let total_paint := 480
  let first_week_paint := (1/5) * total_paint
  let second_week_paint := (1/6) * (total_paint - first_week_paint)
  let third_week_paint := (1/7) * (total_paint - first_week_paint - second_week_paint)
  let fourth_week_paint := (2/9) * (total_paint - first_week_paint - second_week_paint - third_week_paint)
  first_week_paint + second_week_paint + third_week_paint + fourth_week_paint

theorem joe_paint_usage :
  abs (paint_used_after_four_weeks - 266.66) < 0.01 :=
sorry

end joe_paint_usage_l150_150145


namespace alpha_sufficient_but_not_necessary_condition_of_beta_l150_150039
open Classical

variable (x : ℝ)
def α := x = -1
def β := x ≤ 0

theorem alpha_sufficient_but_not_necessary_condition_of_beta :
  (α x → β x) ∧ ¬(β x → α x) :=
by
  sorry

end alpha_sufficient_but_not_necessary_condition_of_beta_l150_150039


namespace sufficient_but_not_necessary_condition_l150_150577

variable (a : ℝ)

theorem sufficient_but_not_necessary_condition :
  (a > 2 → a^2 > 2 * a)
  ∧ (¬(a^2 > 2 * a → a > 2)) := by
  sorry

end sufficient_but_not_necessary_condition_l150_150577


namespace range_of_a_l150_150018

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + a * x + 4 < 0 → false) → (-4 ≤ a ∧ a ≤ 4) :=
by 
  sorry

end range_of_a_l150_150018


namespace striped_jerseys_count_l150_150619

noncomputable def totalSpent : ℕ := 80
noncomputable def longSleevedJerseyCost : ℕ := 15
noncomputable def stripedJerseyCost : ℕ := 10
noncomputable def numberOfLongSleevedJerseys : ℕ := 4

theorem striped_jerseys_count :
  (totalSpent - numberOfLongSleevedJerseys * longSleevedJerseyCost) / stripedJerseyCost = 2 := by
  sorry

end striped_jerseys_count_l150_150619


namespace sqrt_calc_l150_150398

theorem sqrt_calc : Real.sqrt (Real.sqrt (0.00032 ^ (1 / 5))) = 0.669 := by
  sorry

end sqrt_calc_l150_150398


namespace range_of_a_l150_150876

variable {a : ℝ}

theorem range_of_a (h : ¬ ∃ x > 0, x^2 + a * x + 1 < 0) : a ≥ -2 := 
by
  sorry

end range_of_a_l150_150876


namespace max_s_value_l150_150466

theorem max_s_value (r s : ℕ) (hr : r ≥ s) (hs : s ≥ 3)
  (h : ((r - 2) * 180 / r : ℚ) / ((s - 2) * 180 / s) = 60 / 59) :
  s = 117 :=
by
  sorry

end max_s_value_l150_150466


namespace sum_of_three_numbers_l150_150356

theorem sum_of_three_numbers :
  ∀ (a b c : ℕ), 
  a ≤ b ∧ b ≤ c → b = 10 →
  (a + b + c) / 3 = a + 20 →
  (a + b + c) / 3 = c - 30 →
  a + b + c = 60 :=
by
  sorry

end sum_of_three_numbers_l150_150356


namespace integer_part_not_perfect_square_l150_150045

noncomputable def expr (n : ℕ) : ℝ :=
  2 * Real.sqrt (n + 1) / (Real.sqrt (n + 1) - Real.sqrt n)

theorem integer_part_not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, k^2 = ⌊expr n⌋ :=
  sorry

end integer_part_not_perfect_square_l150_150045


namespace students_count_l150_150752

theorem students_count (S : ℕ) (num_adults : ℕ) (cost_student cost_adult total_cost : ℕ)
  (h1 : num_adults = 4)
  (h2 : cost_student = 5)
  (h3 : cost_adult = 6)
  (h4 : total_cost = 199) :
  5 * S + 4 * 6 = 199 → S = 35 := by
  sorry

end students_count_l150_150752


namespace fred_carrots_l150_150324

-- Define the conditions
def sally_carrots : Nat := 6
def total_carrots : Nat := 10

-- Define the problem question and the proof statement
theorem fred_carrots : ∃ fred_carrots : Nat, fred_carrots = total_carrots - sally_carrots := 
by
  sorry

end fred_carrots_l150_150324


namespace puppies_per_cage_l150_150539

/-
Theorem: If a pet store had 56 puppies, sold 24 of them, and placed the remaining puppies into 8 cages, then each cage contains 4 puppies.
-/

theorem puppies_per_cage
  (initial_puppies : ℕ)
  (sold_puppies : ℕ)
  (cages : ℕ)
  (remaining_puppies : ℕ)
  (puppies_per_cage : ℕ) :
  initial_puppies = 56 →
  sold_puppies = 24 →
  cages = 8 →
  remaining_puppies = initial_puppies - sold_puppies →
  puppies_per_cage = remaining_puppies / cages →
  puppies_per_cage = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end puppies_per_cage_l150_150539


namespace older_brother_catches_up_l150_150220

theorem older_brother_catches_up (D : ℝ) (t : ℝ) :
  let vy := D / 25
  let vo := D / 15
  let time := 20
  15 * time = 25 * (time - 8) → (15 * time = 25 * (time - 8) → t = 20)
:= by
  sorry

end older_brother_catches_up_l150_150220


namespace avg_of_multiples_l150_150504

theorem avg_of_multiples (n : ℝ) (h : (n + 2 * n + 3 * n + 4 * n + 5 * n + 6 * n + 7 * n + 8 * n + 9 * n + 10 * n) / 10 = 60.5) : n = 11 :=
by
  sorry

end avg_of_multiples_l150_150504


namespace intersection_A_B_l150_150574

-- Definitions of sets A and B based on the given conditions
def A : Set ℕ := {4, 5, 6, 7}
def B : Set ℕ := {x | 3 ≤ x ∧ x < 6}

-- The theorem stating the proof problem
theorem intersection_A_B : A ∩ B = {4, 5} :=
by
  sorry

end intersection_A_B_l150_150574


namespace fourth_root_cubed_eq_729_l150_150450

theorem fourth_root_cubed_eq_729 (x : ℝ) (hx : (x^(1/4))^3 = 729) : x = 6561 :=
  sorry

end fourth_root_cubed_eq_729_l150_150450


namespace solve_triple_l150_150571

theorem solve_triple (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c + a * b + c = a^3) : 
  (b = a - 1 ∧ c = a) ∨ (b = 1 ∧ c = a * (a - 1)) :=
by 
  sorry

end solve_triple_l150_150571


namespace area_of_enclosing_square_is_100_l150_150362

noncomputable def radius : ℝ := 5

noncomputable def diameter_of_circle (r : ℝ) : ℝ := 2 * r

noncomputable def side_length_of_square (d : ℝ) : ℝ := d

noncomputable def area_of_square (s : ℝ) : ℝ := s * s

theorem area_of_enclosing_square_is_100 :
  area_of_square (side_length_of_square (diameter_of_circle radius)) = 100 :=
by
  sorry

end area_of_enclosing_square_is_100_l150_150362


namespace prove_f_of_pi_div_4_eq_0_l150_150274

noncomputable
def tan_function (ω : ℝ) (x : ℝ) : ℝ := Real.tan (ω * x)

theorem prove_f_of_pi_div_4_eq_0 
  (ω : ℝ) (hω : ω > 0)
  (h_period : ∀ x : ℝ, tan_function ω (x + π / (4 * ω)) = tan_function ω x) :
  tan_function ω (π / 4) = 0 :=
by
  -- This is where the proof would go.
  sorry

end prove_f_of_pi_div_4_eq_0_l150_150274


namespace area_of_abs_inequality_l150_150565

theorem area_of_abs_inequality :
  (setOf (λ (p : ℝ×ℝ), |p.1 + p.2| + |p.1 - p.2| ≤ 6)).measure = 36 :=
sorry

end area_of_abs_inequality_l150_150565


namespace vector_subtraction_scalar_mul_l150_150690

theorem vector_subtraction_scalar_mul :
  let v₁ := (3, -8) 
  let scalar := -5 
  let v₂ := (4, 6)
  v₁.1 - scalar * v₂.1 = 23 ∧ v₁.2 - scalar * v₂.2 = 22 := by
    sorry

end vector_subtraction_scalar_mul_l150_150690


namespace recommended_cups_water_l150_150936

variable (currentIntake : ℕ)
variable (increasePercentage : ℕ)

def recommendedIntake : ℕ := 
  currentIntake + (increasePercentage * currentIntake) / 100

theorem recommended_cups_water (h1 : currentIntake = 15) 
                               (h2 : increasePercentage = 40) : 
  recommendedIntake currentIntake increasePercentage = 21 := 
by 
  rw [h1, h2]
  have h3 : (40 * 15) / 100 = 6 := by norm_num
  rw [h3]
  norm_num
  sorry

end recommended_cups_water_l150_150936


namespace intersection_eq_l150_150628

-- Define sets A and B
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {0, 1, 2}

-- The theorem to be proved
theorem intersection_eq : A ∩ B = {2} := 
by sorry

end intersection_eq_l150_150628


namespace reena_loan_l150_150956

/-- 
  Problem setup:
  Reena took a loan of $1200 at simple interest for a period equal to the rate of interest years. 
  She paid $192 as interest at the end of the loan period.
  We aim to prove that the rate of interest is 4%. 
-/
theorem reena_loan (P : ℝ) (SI : ℝ) (R : ℝ) (N : ℝ) 
  (hP : P = 1200) 
  (hSI : SI = 192) 
  (hN : N = R) 
  (hSI_formula : SI = P * R * N / 100) : 
  R = 4 := 
by 
  sorry

end reena_loan_l150_150956


namespace number_of_crayons_given_to_friends_l150_150480

def totalCrayonsLostOrGivenAway := 229
def crayonsLost := 16
def crayonsGivenToFriends := totalCrayonsLostOrGivenAway - crayonsLost

theorem number_of_crayons_given_to_friends :
  crayonsGivenToFriends = 213 :=
by
  sorry

end number_of_crayons_given_to_friends_l150_150480


namespace cost_of_child_ticket_l150_150381

theorem cost_of_child_ticket
  (total_seats : ℕ)
  (adult_ticket_price : ℕ)
  (num_children : ℕ)
  (total_revenue : ℕ)
  (H1 : total_seats = 250)
  (H2 : adult_ticket_price = 6)
  (H3 : num_children = 188)
  (H4 : total_revenue = 1124) :
  let num_adults := total_seats - num_children
  let revenue_from_adults := num_adults * adult_ticket_price
  let cost_of_child_ticket := (total_revenue - revenue_from_adults) / num_children
  cost_of_child_ticket = 4 :=
by
  sorry

end cost_of_child_ticket_l150_150381


namespace find_k_no_xy_term_l150_150181

theorem find_k_no_xy_term (k : ℝ) :
  (¬ ∃ x y : ℝ, (-x^2 - 3 * k * x * y - 3 * y^2 + 9 * x * y - 8) = (- x^2 - 3 * y^2 - 8)) → k = 3 :=
by
  sorry

end find_k_no_xy_term_l150_150181


namespace total_cost_l150_150910

/-- Sam initially has s yellow balloons.
He gives away a of these balloons to Fred.
Mary has m yellow balloons.
Each balloon costs c dollars.
Determine the total cost for the remaining balloons that Sam and Mary jointly have.
Given: s = 6.0, a = 5.0, m = 7.0, c = 9.0 dollars.
Expected result: the total cost is 72.0 dollars.
-/
theorem total_cost (s a m c : ℝ) (h_s : s = 6.0) (h_a : a = 5.0) (h_m : m = 7.0) (h_c : c = 9.0) :
  (s - a + m) * c = 72.0 := 
by
  rw [h_s, h_a, h_m, h_c]
  -- At this stage, the proof would involve showing the expression is 72.0, but since no proof is required:
  sorry

end total_cost_l150_150910


namespace vertical_distance_from_top_to_bottom_l150_150534

-- Conditions
def ring_thickness : ℕ := 2
def largest_ring_diameter : ℕ := 18
def smallest_ring_diameter : ℕ := 4

-- Additional definitions based on the problem context
def count_rings : ℕ := (largest_ring_diameter - smallest_ring_diameter) / ring_thickness + 1
def inner_diameters_sum : ℕ := count_rings * (largest_ring_diameter - ring_thickness + smallest_ring_diameter) / 2
def vertical_distance : ℕ := inner_diameters_sum + 2 * ring_thickness

-- The problem statement to prove
theorem vertical_distance_from_top_to_bottom :
  vertical_distance = 76 := by
  sorry

end vertical_distance_from_top_to_bottom_l150_150534


namespace range_of_m_l150_150734

def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * x + 2 ≥ m

def proposition_q (m : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → -(7 - 3*m)^x > -(7 - 3*m)^y

theorem range_of_m (m : ℝ) :
  (proposition_p m ∧ ¬ proposition_q m) ∨ (¬ proposition_p m ∧ proposition_q m) ↔ (1 < m ∧ m < 2) :=
sorry

end range_of_m_l150_150734


namespace other_coin_value_l150_150914

-- Condition definitions
def total_coins : ℕ := 36
def dime_count : ℕ := 26
def total_value_dollars : ℝ := 3.10
def dime_value : ℝ := 0.10

-- Derived definitions
def total_dimes_value : ℝ := dime_count * dime_value
def remaining_value : ℝ := total_value_dollars - total_dimes_value
def other_coin_count : ℕ := total_coins - dime_count

-- Proof statement
theorem other_coin_value : (remaining_value / other_coin_count) = 0.05 := by
  sorry

end other_coin_value_l150_150914


namespace common_area_of_equilateral_triangles_in_unit_square_l150_150372

theorem common_area_of_equilateral_triangles_in_unit_square
  (unit_square_side_length : ℝ)
  (triangle_side_length : ℝ)
  (common_area : ℝ)
  (h_unit_square : unit_square_side_length = 1)
  (h_triangle_side : triangle_side_length = 1) :
  common_area = -1 :=
by
  sorry

end common_area_of_equilateral_triangles_in_unit_square_l150_150372


namespace square_with_12_sticks_square_with_15_sticks_l150_150054

-- Definitions for problem conditions
def sum_of_first_n_natural_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def can_form_square (total_length : ℕ) : Prop :=
  total_length % 4 = 0

-- Given n = 12, check if breaking 2 sticks is required to form a square
theorem square_with_12_sticks : (n = 12) → ¬ can_form_square (sum_of_first_n_natural_numbers 12) → true :=
by
  intros
  sorry

-- Given n = 15, check if it is possible to form a square without breaking any sticks
theorem square_with_15_sticks : (n = 15) → can_form_square (sum_of_first_n_natural_numbers 15) → true :=
by
  intros
  sorry

end square_with_12_sticks_square_with_15_sticks_l150_150054


namespace number_halfway_l150_150052

theorem number_halfway (a b : ℚ) (h1 : a = 1/12) (h2 : b = 1/10) : (a + b) / 2 = 11 / 120 := by
  sorry

end number_halfway_l150_150052


namespace confidence_of_independence_test_l150_150357

-- Define the observed value of K^2
def K2_obs : ℝ := 5

-- Define the critical value(s) of K^2 for different confidence levels
def K2_critical_0_05 : ℝ := 3.841
def K2_critical_0_01 : ℝ := 6.635

-- Define the confidence levels corresponding to the critical values
def P_K2_ge_3_841 : ℝ := 0.05
def P_K2_ge_6_635 : ℝ := 0.01

-- Define the statement to be proved: there is 95% confidence that "X and Y are related".
theorem confidence_of_independence_test
  (K2_obs K2_critical_0_05 P_K2_ge_3_841 : ℝ)
  (hK2_obs_gt_critical : K2_obs > K2_critical_0_05)
  (hP : P_K2_ge_3_841 = 0.05) :
  1 - P_K2_ge_3_841 = 0.95 :=
by
  -- The proof is omitted
  sorry

end confidence_of_independence_test_l150_150357


namespace exists_root_interval_l150_150429

noncomputable def f (x : ℝ) : ℝ := (1 / x) - Real.log x / Real.log 2

theorem exists_root_interval :
  (∃ x ∈ Ioo 1 2, f x = 0) :=
begin
  -- Formalize the conditions needed to apply the Intermediate Value Theorem.
  have h_continuous : ContinuousOn f (Ioo 1 2),
  { apply ContinuousOn.sub,
    { apply ContinuousOn.continuous_on_inv,
      intros x hx, simp at hx, linarith, },
    { apply ContinuousOn.comp,
      { exact continuous_log.continuous_on,
        intros x hx, simp at hx, linarith, },
      { exact continuous_const.continuous_on } } },
  have h_decreasing : ∀ x y ∈ Ioo 1 2, x < y → f x > f y,
  { intros x x_in y y_in xy,
    -- Proof of the function being decreasing
    sorry
  },
  have h_sign_change : f 1 > 0 ∧ f 2 < 0,
  { split; simp [f], linarith },
  -- Apply the Intermediate Value Theorem
  exact intermediate_value_Ioo h_continuous (1 : ℝ) (2 : ℝ) zero_lt_one zero_lt_two ((-1 : ℝ).lt_add_one) h_sign_change,
end

end exists_root_interval_l150_150429


namespace solve_for_a_l150_150397

theorem solve_for_a (a : ℤ) : -2 - a = 0 → a = -2 :=
by
  sorry

end solve_for_a_l150_150397


namespace simplify_expression_l150_150486

variable (y : ℝ)

theorem simplify_expression :
  y * (4 * y^2 - 3) - 6 * (y^2 - 3 * y + 8) = 4 * y^3 - 6 * y^2 + 15 * y - 48 :=
by
  sorry

end simplify_expression_l150_150486


namespace g_g_g_3_l150_150040

def g (n : ℕ) : ℕ :=
if n < 5 then n^2 + 2*n + 1 else 4*n - 3

theorem g_g_g_3 : g (g (g 3)) = 241 := by
  sorry

end g_g_g_3_l150_150040


namespace barney_and_regular_dinosaurs_combined_weight_l150_150687

theorem barney_and_regular_dinosaurs_combined_weight :
  (∀ (barney five_dinos_weight) (reg_dino_weight : ℕ),
    (barney = five_dinos_weight + 1500) →
    (five_dinos_weight = 5 * reg_dino_weight) →
    (reg_dino_weight = 800) →
    barney + five_dinos_weight = 9500) :=
by 
  intros barney five_dinos_weight reg_dino_weight h1 h2 h3
  have h4 : five_dinos_weight = 5 * 800 := by rw [h3, mul_comm]
  rw h4 at h2
  have h5 : five_dinos_weight = 4000 := by linarith
  rw h5 at h1
  have h6 : barney = 4000 + 1500 := h1
  have h7 : barney = 5500 := by linarith
  rw [h5, h7]
  linarith

end barney_and_regular_dinosaurs_combined_weight_l150_150687


namespace percentage_increase_in_items_sold_l150_150961

-- Definitions
variables (P N M : ℝ)
-- Given conditions:
-- The new price of an item
def new_price := P * 0.90
-- The relationship between incomes
def income_increase := (P * 0.90) * M = P * N * 1.125

-- The problem statement
theorem percentage_increase_in_items_sold (h : income_increase P N M) :
  M = N * 1.25 :=
sorry

end percentage_increase_in_items_sold_l150_150961


namespace a5_value_l150_150883

-- Definitions
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n, a (n + 1) = q * a n

def positive_terms (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0

def product_condition (a : ℕ → ℝ) : Prop :=
  ∀ n, a n * a (n + 1) = 2^(2 * n + 1)

-- Theorem statement
theorem a5_value (a : ℕ → ℝ) (h_geo : geometric_sequence a) (h_pos : positive_terms a) (h_prod : product_condition a) : a 5 = 32 :=
sorry

end a5_value_l150_150883


namespace person_speed_l150_150199

noncomputable def distance_meters : ℝ := 1080
noncomputable def time_minutes : ℝ := 14
noncomputable def distance_kilometers : ℝ := distance_meters / 1000
noncomputable def time_hours : ℝ := time_minutes / 60
noncomputable def speed_km_per_hour : ℝ := distance_kilometers / time_hours

theorem person_speed :
  abs (speed_km_per_hour - 4.63) < 0.01 :=
by
  -- conditions extracted
  let distance_in_km := distance_meters / 1000
  let time_in_hours := time_minutes / 60
  let speed := distance_in_km / time_in_hours
  -- We expect speed to be approximately 4.63
  sorry 

end person_speed_l150_150199


namespace delta_value_l150_150442

theorem delta_value (Δ : ℤ) : 5 * (-3) = Δ - 3 → Δ = -12 :=
by
  sorry

end delta_value_l150_150442


namespace first_player_always_wins_l150_150070

theorem first_player_always_wins :
  ∃ A B : ℤ, A ≠ 0 ∧ B ≠ 0 ∧
  (A = 1998 ∧ B = -2 * 1998) ∧
  (∀ a b c : ℤ, (a = A ∨ a = B ∨ a = 1998) ∧ 
                (b = A ∨ b = B ∨ b = 1998) ∧ 
                (c = A ∨ c = B ∨ c = 1998) ∧ 
                a ≠ b ∧ b ≠ c ∧ a ≠ c → 
                ∃ x1 x2 : ℚ, x1 ≠ x2 ∧ 
                (a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0)) :=
by
  sorry

end first_player_always_wins_l150_150070


namespace find_abc_l150_150958

theorem find_abc (a b c : ℝ) 
  (h1 : 2 * b = a + c)  -- a, b, c form an arithmetic sequence
  (h2 : a + b + c = 12) -- The sum of a, b, and c is 12
  (h3 : (b + 2)^2 = (a + 2) * (c + 5)) -- a+2, b+2, and c+5 form a geometric sequence
: (a = 1 ∧ b = 4 ∧ c = 7) ∨ (a = 10 ∧ b = 4 ∧ c = -2) :=
sorry

end find_abc_l150_150958


namespace distribute_5_graduates_l150_150358

theorem distribute_5_graduates :
  (∃ G : Finset (Finset ℕ), G.card = 3 ∧ ∀ g ∈ G, g.card ≤ 2 ∧ g.card ≥ 1 ∧ Finset.univ.card (Finset ⋃₀ G) = 5) → 
  fintype.card {g | g.card = 5} /  A 5 3 = 90 :=
begin
  sorry
end

end distribute_5_graduates_l150_150358


namespace work_completion_time_l150_150367

noncomputable def work_done (hours : ℕ) (a_rate : ℚ) (b_rate : ℚ) : ℚ :=
  if hours % 2 = 0 then (hours / 2) * (a_rate + b_rate)
  else ((hours - 1) / 2) * (a_rate + b_rate) + a_rate

theorem work_completion_time :
  let a_rate := 1/4
  let b_rate := 1/12
  (∃ t, work_done t a_rate b_rate = 1) → t = 6 := 
by
  intro h
  sorry

end work_completion_time_l150_150367


namespace expected_value_of_twelve_sided_die_l150_150822

theorem expected_value_of_twelve_sided_die : 
  let outcomes := (finset.range 13).filter (λ n, n ≠ 0) in
  (finset.sum outcomes (λ n, (n:ℝ)) / 12 = 6.5) :=
by
  let outcomes := (finset.range 13).filter (λ n, n ≠ 0)
  have h1 : ∑ n in outcomes, (n : ℝ) = 78, sorry
  have h2 : (78 / 12) = 6.5, sorry
  exact h2

end expected_value_of_twelve_sided_die_l150_150822


namespace circle_equation_l150_150787

theorem circle_equation (x y : ℝ) :
  let center := (0, 4)
  let point_on_circle := (3, 0)
  (x - center.1)^2 + (y - center.2)^2 = 25 :=
by
  sorry

end circle_equation_l150_150787


namespace q_value_at_2_l150_150232

-- Define the function q and the fact that (2, 3) is on its graph
def q : ℝ → ℝ := sorry

-- Condition: (2, 3) is on the graph of q(x)
axiom q_at_2 : q 2 = 3

-- Theorem: The value of q(2) is 3
theorem q_value_at_2 : q 2 = 3 := 
by 
  apply q_at_2

end q_value_at_2_l150_150232


namespace quotient_is_eight_l150_150364

theorem quotient_is_eight (d v r q : ℕ) (h₁ : d = 141) (h₂ : v = 17) (h₃ : r = 5) (h₄ : d = v * q + r) : q = 8 :=
by
  sorry

end quotient_is_eight_l150_150364


namespace n_decomposable_form_l150_150897

theorem n_decomposable_form (n : ℕ) (a : ℕ) (h₁ : a > 2) (h₂ : ∃ k, 1 < k ∧ n = 2^k) :
  (∀ d : ℕ, d ∣ n ∧ d ≠ n → (a^n - 2^n) % (a^d + 2^d) = 0) → ∃ k, 1 < k ∧ n = 2^k :=
by {
  sorry
}

end n_decomposable_form_l150_150897


namespace question_eq_answer_l150_150588

theorem question_eq_answer (n : ℝ) (h : 0.25 * 0.1 * n = 15) :
  0.1 * 0.25 * n = 15 :=
by
  sorry

end question_eq_answer_l150_150588


namespace min_value_four_x_plus_one_over_x_l150_150994

theorem min_value_four_x_plus_one_over_x (x : ℝ) (hx : x > 0) : 4*x + 1/x ≥ 4 := by
  sorry

end min_value_four_x_plus_one_over_x_l150_150994


namespace number_of_true_propositions_l150_150496

-- Define the original proposition
def prop (x: Real) : Prop := x^2 > 1 → x > 1

-- Define converse, inverse, contrapositive
def converse (x: Real) : Prop := x > 1 → x^2 > 1
def inverse (x: Real) : Prop := x^2 ≤ 1 → x ≤ 1
def contrapositive (x: Real) : Prop := x ≤ 1 → x^2 ≤ 1

-- Define the proposition we want to prove: the number of true propositions among them
theorem number_of_true_propositions :
  (converse 2 = True) ∧ (inverse 2 = True) ∧ (contrapositive 2 = False) → 2 = 2 :=
by sorry

end number_of_true_propositions_l150_150496


namespace draw_points_value_l150_150140

theorem draw_points_value
  (D : ℕ) -- Let D be the number of points for a draw
  (victory_points : ℕ := 3) -- points for a victory
  (defeat_points : ℕ := 0) -- points for a defeat
  (total_matches : ℕ := 20) -- total matches
  (points_after_5_games : ℕ := 8) -- points scored in the first 5 games
  (minimum_wins_remaining : ℕ := 9) -- at least 9 matches should be won in the remaining matches
  (target_points : ℕ := 40) : -- target points by the end of the tournament
  D = 1 := 
by 
  sorry


end draw_points_value_l150_150140


namespace total_flowers_l150_150800

theorem total_flowers (R T L : ℕ) 
  (hR : R = 58)
  (hT : R = T + 15)
  (hL : R = L - 25) :
  R + T + L = 184 :=
by 
  sorry

end total_flowers_l150_150800


namespace non_neg_reals_inequality_l150_150013

theorem non_neg_reals_inequality (a b c : ℝ)
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c)
  (h₃ : a + b + c ≤ 3) :
  (a / (1 + a^2) + b / (1 + b^2) + c / (1 + c^2) ≤ 3/2) ∧
  (3/2 ≤ (1 / (1 + a) + 1 / (1 + b) + 1 / (1 + c))) :=
by
  sorry

end non_neg_reals_inequality_l150_150013


namespace rectangle_difference_l150_150784

theorem rectangle_difference (L B D : ℝ)
  (h1 : L - B = D)
  (h2 : 2 * (L + B) = 186)
  (h3 : L * B = 2030) :
  D = 23 :=
by
  sorry

end rectangle_difference_l150_150784


namespace cube_volume_surface_area_eq_1728_l150_150515

theorem cube_volume_surface_area_eq_1728 (x : ℝ) (h1 : (side : ℝ) (v : ℝ) hvolume : v = 8 * x ∧ v = side^3) (h2: (side : ℝ) (a : ℝ) hsurface : a = 2 * x ∧ a = 6 * side^2) : x = 1728 :=
by
  sorry

end cube_volume_surface_area_eq_1728_l150_150515


namespace num_sets_B_l150_150431

open Set

theorem num_sets_B (A B : Set ℕ) (hA : A = {1, 2}) (h_union : A ∪ B = {1, 2, 3}) : ∃ n, n = 4 :=
by
  sorry

end num_sets_B_l150_150431


namespace modular_inverse_calculation_l150_150412

theorem modular_inverse_calculation : 
  (3 * (49 : ℤ) + 12 * (40 : ℤ)) % 65 = 42 := 
by
  sorry

end modular_inverse_calculation_l150_150412


namespace probability_f_ge1_l150_150728

noncomputable def f (x: ℝ) : ℝ := 3*x^2 - x - 1

def domain : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

def valid_intervals : Set ℝ := { x | -1 ≤ x ∧ x ≤ -2/3 } ∪ { x | 1 ≤ x ∧ x ≤ 2 }

def interval_length (a b : ℝ) : ℝ := b - a

theorem probability_f_ge1 : 
  (interval_length (-2/3) (-1) + interval_length 1 2) / interval_length (-1) 2 = 4 / 9 := 
by
  sorry

end probability_f_ge1_l150_150728


namespace negation_of_proposition_l150_150373

variables (a b : ℕ)

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def both_even (a b : ℕ) : Prop := is_even a ∧ is_even b

def sum_even (a b : ℕ) : Prop := is_even (a + b)

theorem negation_of_proposition : ¬ (both_even a b → sum_even a b) ↔ ¬both_even a b ∨ ¬sum_even a b :=
by sorry

end negation_of_proposition_l150_150373


namespace jimmy_income_l150_150165

variable (J : ℝ)

def rebecca_income : ℝ := 15000
def income_increase : ℝ := 3000
def rebecca_income_after_increase : ℝ := rebecca_income + income_increase
def combined_income : ℝ := 2 * rebecca_income_after_increase

theorem jimmy_income (h : rebecca_income_after_increase + J = combined_income) : 
  J = 18000 := by
  sorry

end jimmy_income_l150_150165


namespace combinations_9_choose_3_l150_150297

theorem combinations_9_choose_3 : (nat.choose 9 3) = 84 :=
by
  sorry

end combinations_9_choose_3_l150_150297


namespace trisha_total_distance_l150_150042

theorem trisha_total_distance :
  let d1 := 0.1111111111111111
  let d2 := 0.1111111111111111
  let d3 := 0.6666666666666666
  d1 + d2 + d3 = 0.8888888888888888 := 
by
  sorry

end trisha_total_distance_l150_150042


namespace swans_in_10_years_l150_150114

def doubling_time := 2
def initial_swans := 15
def periods := 10 / doubling_time

theorem swans_in_10_years : 
  (initial_swans * 2 ^ periods) = 480 := 
by
  sorry

end swans_in_10_years_l150_150114


namespace sarah_min_width_l150_150484

noncomputable def minWidth (S : Type) [LinearOrder S] (w : S) : Prop :=
  ∃ w, w ≥ 0 ∧ w * (w + 20) ≥ 150 ∧ ∀ w', (w' ≥ 0 ∧ w' * (w' + 20) ≥ 150) → w ≤ w'

theorem sarah_min_width : minWidth ℝ 10 :=
by {
  sorry -- proof goes here
}

end sarah_min_width_l150_150484


namespace abs_ineq_solution_l150_150843

theorem abs_ineq_solution (x : ℝ) : (2 ≤ |x - 5| ∧ |x - 5| ≤ 4) ↔ (1 ≤ x ∧ x ≤ 3) ∨ (7 ≤ x ∧ x ≤ 9) :=
by
  sorry

end abs_ineq_solution_l150_150843


namespace k_range_m_range_l150_150849

noncomputable def f (x : ℝ) : ℝ := 1 - (2 / (2^x + 1))

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem k_range (k : ℝ) : (∃ x : ℝ, g x = (2^x + 1) * f x + k) → k < 1 :=
by
  sorry

theorem m_range (m : ℝ) : (∀ x1 : ℝ, 0 < x1 ∧ x1 < 1 → 
                        ∃ x2 : ℝ, -Real.pi / 4 ≤ x2 ∧ x2 ≤ Real.pi / 6 ∧ f x1 - m * 2^x1 > g x2) 
                       → m ≤ 7 / 6 :=
by
  sorry

end k_range_m_range_l150_150849


namespace sum_of_geometric_sequence_eq_31_over_16_l150_150351

theorem sum_of_geometric_sequence_eq_31_over_16 (n : ℕ) :
  let a := 1
  let r := (1 / 2 : ℝ)
  let S_n := 2 - 2 * r^n
  (S_n = (31 / 16 : ℝ)) ↔ (n = 5) := by
{
  sorry
}

end sum_of_geometric_sequence_eq_31_over_16_l150_150351


namespace number_of_cages_l150_150216

-- Definitions based on the conditions
def parrots_per_cage := 2
def parakeets_per_cage := 6
def total_birds := 72

-- Goal: Prove the number of cages
theorem number_of_cages : 
  (parrots_per_cage + parakeets_per_cage) * x = total_birds → x = 9 :=
by
  sorry

end number_of_cages_l150_150216


namespace sufficient_condition_for_inequality_l150_150854

theorem sufficient_condition_for_inequality (a b : ℝ) (h_nonzero : a * b ≠ 0) : (a < b ∧ b < 0) → (1 / a ^ 2 > 1 / b ^ 2) :=
by
  intro h
  sorry

end sufficient_condition_for_inequality_l150_150854


namespace problem1_proof_problem2_proof_l150_150000

noncomputable def problem1 : Real :=
  Real.sqrt 2 * Real.sqrt 3 + Real.sqrt 24

theorem problem1_proof : problem1 = 3 * Real.sqrt 6 :=
  sorry

noncomputable def problem2 : Real :=
  (3 * Real.sqrt 2 - Real.sqrt 12) * (Real.sqrt 18 + 2 * Real.sqrt 3)

theorem problem2_proof : problem2 = 6 :=
  sorry

end problem1_proof_problem2_proof_l150_150000


namespace find_b_l150_150921

theorem find_b (a b c : ℝ) (h1 : a + b + c = 120) (h2 : a + 5 = b - 5) (h3 : b - 5 = c^2) : b = 61.25 :=
by {
  sorry
}

end find_b_l150_150921


namespace hyperbola_intersect_circle_l150_150341

noncomputable def point (x y : ℝ) : ℝ × ℝ := (x, y)

theorem hyperbola_intersect_circle 
  (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hA : A = (4, 1 / 4))
  (hB : B = (-5, -1 / 5))
  (h_hypA : A.1 * A.2 = 1)
  (h_hypB : B.1 * B.2 = 1) :
  ∃ X Y : ℝ × ℝ, 
    (X.1 * X.2 = 1) ∧ (Y.1 * Y.2 = 1) ∧ 
    dist X Y = real.sqrt (401 / 5) :=
begin
  sorry
end

end hyperbola_intersect_circle_l150_150341


namespace avg_rate_of_change_l150_150047

def f (x : ℝ) := 2 * x + 1

theorem avg_rate_of_change : (f 5 - f 1) / (5 - 1) = 2 := by
  sorry

end avg_rate_of_change_l150_150047


namespace cube_x_value_l150_150513

noncomputable def cube_side_len (x : ℝ) : ℝ := (8 * x) ^ (1 / 3)

lemma cube_volume (x : ℝ) : (cube_side_len x) ^ 3 = 8 * x :=
  by sorry

lemma cube_surface_area (x : ℝ) : 6 * (cube_side_len x) ^ 2 = 2 * x :=
  by sorry

theorem cube_x_value (x : ℝ) (hV : (cube_side_len x) ^ 3 = 8 * x) (hS : 6 * (cube_side_len x) ^ 2 = 2 * x) : x = sqrt 3 / 72 :=
  by sorry

end cube_x_value_l150_150513


namespace initial_rope_length_l150_150310

variable (R₀ R₁ R₂ R₃ : ℕ)
variable (h_cut1 : 2 * R₀ = R₁) -- Josh cuts the original rope in half
variable (h_cut2 : 2 * R₁ = R₂) -- He cuts one of the halves in half again
variable (h_cut3 : 5 * R₂ = R₃) -- He cuts one of the resulting pieces into fifths
variable (h_held_piece : R₃ = 5) -- The piece Josh is holding is 5 feet long

theorem initial_rope_length:
  R₀ = 100 :=
by
  sorry

end initial_rope_length_l150_150310


namespace margin_in_terms_of_ratio_l150_150137

variable (S m : ℝ)

theorem margin_in_terms_of_ratio (h1 : M = (1/m) * S) (h2 : C = S - M) : M = (1/m) * S :=
sorry

end margin_in_terms_of_ratio_l150_150137


namespace range_of_m_l150_150857

theorem range_of_m (α β m : ℝ)
  (h1 : 0 < α ∧ α < 1)
  (h2 : 1 < β ∧ β < 2)
  (h3 : ∀ x, x^2 - m * x + 1 = 0 ↔ (x = α ∨ x = β)) :
  2 < m ∧ m < 5 / 2 :=
sorry

end range_of_m_l150_150857


namespace tangent_line_to_parabola_l150_150337

theorem tangent_line_to_parabola (c : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + c → y^2 = 12 * x) → c = 1 :=
by
  sorry

end tangent_line_to_parabola_l150_150337


namespace smallest_positive_m_l150_150806

theorem smallest_positive_m (m : ℕ) :
  (∃ (r s : ℤ), 18 * r * s = 252 ∧ m = 18 * (r + s) ∧ r ≠ s) ∧ m > 0 →
  m = 162 := 
sorry

end smallest_positive_m_l150_150806


namespace naomi_stickers_l150_150942

theorem naomi_stickers :
  ∃ S : ℕ, S > 1 ∧
    (S % 5 = 2) ∧
    (S % 9 = 2) ∧
    (S % 11 = 2) ∧
    S = 497 :=
by
  sorry

end naomi_stickers_l150_150942


namespace approx_num_fish_in_pond_l150_150881

noncomputable def numFishInPond (tagged_in_second: ℕ) (total_second: ℕ) (tagged: ℕ) : ℕ :=
  tagged * total_second / tagged_in_second

theorem approx_num_fish_in_pond :
  numFishInPond 2 50 50 = 1250 := by
  sorry

end approx_num_fish_in_pond_l150_150881


namespace calculate_salary_l150_150964

-- Define the constants and variables
def food_percentage : ℝ := 0.35
def rent_percentage : ℝ := 0.25
def clothes_percentage : ℝ := 0.20
def transportation_percentage : ℝ := 0.10
def recreational_percentage : ℝ := 0.15
def emergency_fund : ℝ := 3000
def total_percentage : ℝ := food_percentage + rent_percentage + clothes_percentage + transportation_percentage + recreational_percentage

-- Define the salary
def salary (S : ℝ) : Prop :=
  (total_percentage - 1) * S = emergency_fund

-- The theorem stating the salary is 60000
theorem calculate_salary : ∃ S : ℝ, salary S ∧ S = 60000 :=
by
  use 60000
  unfold salary total_percentage
  sorry

end calculate_salary_l150_150964


namespace julieta_total_spent_l150_150147

def original_price_backpack : ℕ := 50
def original_price_ring_binder : ℕ := 20
def quantity_ring_binders : ℕ := 3
def price_increase_backpack : ℕ := 5
def price_decrease_ring_binder : ℕ := 2

def total_spent (original_price_backpack original_price_ring_binder quantity_ring_binders price_increase_backpack price_decrease_ring_binder : ℕ) : ℕ :=
  let new_price_backpack := original_price_backpack + price_increase_backpack
  let new_price_ring_binder := original_price_ring_binder - price_decrease_ring_binder
  new_price_backpack + (new_price_ring_binder * quantity_ring_binders)

theorem julieta_total_spent :
  total_spent original_price_backpack original_price_ring_binder quantity_ring_binders price_increase_backpack price_decrease_ring_binder = 109 :=
by 
  -- Proof steps are omitted intentionally
  sorry

end julieta_total_spent_l150_150147


namespace average_mark_of_excluded_students_l150_150334

theorem average_mark_of_excluded_students (N A E A_R A_E : ℝ) 
  (hN : N = 25) 
  (hA : A = 80) 
  (hE : E = 5) 
  (hAR : A_R = 90) 
  (h_eq : N * A - E * A_E = (N - E) * A_R) : 
  A_E = 40 := 
by 
  sorry

end average_mark_of_excluded_students_l150_150334


namespace delta_value_l150_150440

theorem delta_value (Delta : ℤ) (h : 5 * (-3) = Delta - 3) : Delta = -12 := 
by 
  sorry

end delta_value_l150_150440


namespace joe_sold_50_cookies_l150_150462

theorem joe_sold_50_cookies :
  ∀ (x : ℝ), (1.20 = 1 + 0.20 * 1) → (60 = 1.20 * x) → x = 50 :=
by
  intros x h1 h2
  sorry

end joe_sold_50_cookies_l150_150462


namespace select_3_products_select_exactly_1_defective_select_at_least_1_defective_l150_150741

noncomputable def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

namespace ProductInspection

def total_products : Nat := 100
def qualified_products : Nat := 98
def defective_products : Nat := 2

-- Proof Problem 1
theorem select_3_products (h : combination total_products 3 = 161700) : True := by
  trivial

-- Proof Problem 2
theorem select_exactly_1_defective (h : combination defective_products 1 * combination qualified_products 2 = 9506) : True := by
  trivial

-- Proof Problem 3
theorem select_at_least_1_defective (h : combination total_products 3 - combination qualified_products 3 = 9604) : True := by
  trivial

end ProductInspection

end select_3_products_select_exactly_1_defective_select_at_least_1_defective_l150_150741


namespace fuel_for_empty_plane_per_mile_l150_150977

theorem fuel_for_empty_plane_per_mile :
  let F := 106000 / 400 - (35 * 3 + 70 * 2)
  F = 20 := 
by
  sorry

end fuel_for_empty_plane_per_mile_l150_150977


namespace striped_jerseys_count_l150_150620

noncomputable def totalSpent : ℕ := 80
noncomputable def longSleevedJerseyCost : ℕ := 15
noncomputable def stripedJerseyCost : ℕ := 10
noncomputable def numberOfLongSleevedJerseys : ℕ := 4

theorem striped_jerseys_count :
  (totalSpent - numberOfLongSleevedJerseys * longSleevedJerseyCost) / stripedJerseyCost = 2 := by
  sorry

end striped_jerseys_count_l150_150620


namespace coefficient_x3_in_expansion_l150_150104

theorem coefficient_x3_in_expansion : 
  (∃ (r : ℕ), 5 - r / 2 = 3 ∧ 2 * Nat.choose 5 r = 10) :=
by 
  sorry

end coefficient_x3_in_expansion_l150_150104


namespace votes_for_winning_candidate_l150_150501

-- Define the variables and conditions
variable (V : ℝ) -- Total number of votes
variable (W : ℝ) -- Votes for the winner

-- Condition 1: The winner received 75% of the votes
axiom winner_votes: W = 0.75 * V

-- Condition 2: The winner won by 500 votes
axiom win_by_500: W - 0.25 * V = 500

-- The statement we want to prove
theorem votes_for_winning_candidate : W = 750 :=
by sorry

end votes_for_winning_candidate_l150_150501


namespace cards_difference_l150_150533

theorem cards_difference
  (H : ℕ)
  (F : ℕ)
  (B : ℕ)
  (hH : H = 200)
  (hF : F = 4 * H)
  (hTotal : B + F + H = 1750) :
  F - B = 50 :=
by
  sorry

end cards_difference_l150_150533


namespace age_of_other_man_l150_150641

variables (A M : ℝ)

theorem age_of_other_man 
  (avg_age_of_men : ℝ)
  (replaced_man_age : ℝ)
  (avg_age_of_women : ℝ)
  (total_age_6_men : 6 * avg_age_of_men = 6 * (avg_age_of_men + 3) - replaced_man_age - M + 2 * avg_age_of_women) :
  M = 44 :=
by
  sorry

end age_of_other_man_l150_150641


namespace find_phi_l150_150729

theorem find_phi 
  (f : ℝ → ℝ)
  (phi : ℝ)
  (y : ℝ → ℝ)
  (h1 : ∀ x, f x = Real.sin (2 * x + Real.pi / 6))
  (h2 : 0 < phi ∧ phi < Real.pi / 2)
  (h3 : ∀ x, y x = f (x - phi) ∧ y x = y (-x)) :
  phi = Real.pi / 3 :=
by
  sorry

end find_phi_l150_150729


namespace factorize_m_square_minus_4m_l150_150559

theorem factorize_m_square_minus_4m (m : ℝ) : m^2 - 4 * m = m * (m - 4) :=
by
  sorry

end factorize_m_square_minus_4m_l150_150559


namespace expected_value_of_twelve_sided_die_l150_150826

theorem expected_value_of_twelve_sided_die : ∑ k in finset.range 13, k / 12 = 6.5 := 
sorry

end expected_value_of_twelve_sided_die_l150_150826


namespace exists_three_cycle_l150_150968

variable {α : Type}

def tournament (P : α → α → Prop) : Prop :=
  (∃ (participants : List α), participants.length ≥ 3) ∧
  (∀ x y, x ≠ y → P x y ∨ P y x) ∧
  (∀ x, ∃ y, P x y)

theorem exists_three_cycle {α : Type} (P : α → α → Prop) :
  tournament P → ∃ A B C, P A B ∧ P B C ∧ P C A :=
by
  sorry

end exists_three_cycle_l150_150968


namespace max_selection_no_five_times_l150_150847

theorem max_selection_no_five_times (S : Finset ℕ) (hS : S = Finset.Icc 1 2014) :
  ∃ n, n = 1665 ∧ 
  ∀ (a b : ℕ), a ∈ S → b ∈ S → (a = 5 * b ∨ b = 5 * a) → false :=
sorry

end max_selection_no_five_times_l150_150847


namespace find_integer_n_l150_150705

theorem find_integer_n : ∃ (n : ℤ), (-90 ≤ n ∧ n ≤ 90) ∧ (Real.sin (n * Real.pi / 180) = Real.cos (456 * Real.pi / 180)) ∧ n = -6 := 
by
  sorry

end find_integer_n_l150_150705


namespace min_sticks_to_be_broken_form_square_without_breaks_l150_150061

noncomputable def total_length (n : ℕ) : ℕ := n * (n + 1) / 2

def divisible_by_4 (x : ℕ) : Prop := x % 4 = 0

theorem min_sticks_to_be_broken (n : ℕ) : n = 12 → (¬ divisible_by_4 (total_length n)) ∧ (minimal_breaks n = 2) :=
by
  intro h1
  rw h1
  have h2 : total_length 12 = 78 := by decide
  have h3 : ¬ divisible_by_4 78 := by decide
  exact ⟨h3, sorry⟩

theorem form_square_without_breaks (n : ℕ) : n = 15 → divisible_by_4 (total_length n) ∧ (minimal_breaks n = 0) :=
by
  intro h1
  rw h1
  have h2 : total_length 15 = 120 := by decide
  have h3 : divisible_by_4 120 := by decide
  exact ⟨h3, sorry⟩

end min_sticks_to_be_broken_form_square_without_breaks_l150_150061


namespace fraction_of_x_by_110_l150_150671

theorem fraction_of_x_by_110 (x : ℝ) (f : ℝ) (h1 : 0.6 * x = f * x + 110) (h2 : x = 412.5) : f = 1 / 3 :=
by 
  sorry

end fraction_of_x_by_110_l150_150671


namespace smallest_positive_m_condition_l150_150508

theorem smallest_positive_m_condition
  (p q : ℤ) (m : ℤ) (h_prod : p * q = 42) (h_diff : |p - q| ≤ 10) 
  (h_roots : 15 * (p + q) = m) : m = 195 :=
sorry

end smallest_positive_m_condition_l150_150508


namespace number_of_coprimes_to_15_l150_150698

open Nat

theorem number_of_coprimes_to_15 : (Finset.filter (λ a, gcd a 15 = 1) (Finset.range 15)).card = 8 := by
  sorry

end number_of_coprimes_to_15_l150_150698


namespace shortest_distance_parabola_to_line_l150_150257

open Real

theorem shortest_distance_parabola_to_line :
  ∃ (d : ℝ), 
    (∀ (P : ℝ × ℝ), (P.1 = (P.2^2) / 8) → 
      ((2 * P.1 - P.2 - 4) / sqrt 5 ≥ d)) ∧ 
    (d = 3 * sqrt 5 / 5) :=
sorry

end shortest_distance_parabola_to_line_l150_150257


namespace minimum_value_of_sum_l150_150419

variable (x y : ℝ)

theorem minimum_value_of_sum (hx : x > 0) (hy : y > 0) : ∃ x y, x > 0 ∧ y > 0 ∧ (x + 2 * y) = 9 :=
sorry

end minimum_value_of_sum_l150_150419


namespace ratio_length_to_breadth_l150_150643

-- Definitions of the given conditions
def length_landscape : ℕ := 120
def area_playground : ℕ := 1200
def ratio_playground_to_landscape : ℕ := 3

-- Property that the area of the playground is 1/3 of the area of the landscape
def total_area_landscape (area_playground : ℕ) (ratio_playground_to_landscape : ℕ) : ℕ :=
  area_playground * ratio_playground_to_landscape

-- Calculation that breadth of the landscape
def breadth_landscape (length_landscape total_area_landscape : ℕ) : ℕ :=
  total_area_landscape / length_landscape

-- The proof statement for the ratio of length to breadth
theorem ratio_length_to_breadth (length_landscape area_playground : ℕ) (ratio_playground_to_landscape : ℕ)
  (h1 : length_landscape = 120)
  (h2 : area_playground = 1200)
  (h3 : ratio_playground_to_landscape = 3)
  (h4 : total_area_landscape area_playground ratio_playground_to_landscape = 3600)
  (h5 : breadth_landscape length_landscape (total_area_landscape area_playground ratio_playground_to_landscape) = 30) :
  length_landscape / breadth_landscape length_landscape (total_area_landscape area_playground ratio_playground_to_landscape) = 4 :=
by
  sorry


end ratio_length_to_breadth_l150_150643


namespace divisibility_by_2k_l150_150900

-- Define the sequence according to the given conditions
def seq (a : ℕ → ℤ) : Prop :=
  a 0 = 0 ∧ a 1 = 1 ∧ ∀ n, 2 ≤ n → a n = 2 * a (n - 1) + a (n - 2)

-- The theorem to be proved
theorem divisibility_by_2k (a : ℕ → ℤ) (k : ℕ) (n : ℕ)
  (h : seq a) :
  2^k ∣ a n ↔ 2^k ∣ n :=
sorry

end divisibility_by_2k_l150_150900


namespace correct_choice_l150_150227

theorem correct_choice (a : ℝ) : -(-a)^2 * a^4 = -a^6 := 
sorry

end correct_choice_l150_150227


namespace smallest_n_inequality_l150_150569

theorem smallest_n_inequality :
  ∃ n : ℕ, (∀ x y z : ℝ, (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4)) ∧
    ∀ m : ℕ, m < n → ¬ (∀ x y z : ℝ, (x^2 + y^2 + z^2)^2 ≤ m * (x^4 + y^4 + z^4)) :=
by
  sorry

end smallest_n_inequality_l150_150569


namespace union_complement_set_l150_150021

theorem union_complement_set (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4, 5}) 
  (hA : A = {1, 2, 3, 5}) (hB : B = {2, 4}) :
  (U \ A) ∪ B = {0, 2, 4} :=
by
  rw [Set.diff_eq, hU, hA, hB]
  simp
  sorry

end union_complement_set_l150_150021


namespace incorrect_option_C_l150_150467

def line (α : Type*) := α → Prop
def plane (α : Type*) := α → Prop

variables {α : Type*} (m n : line α) (a b : plane α)

def parallel (m n : line α) : Prop := ∀ x, m x → n x
def perpendicular (m n : line α) : Prop := ∃ x, m x ∧ n x

def lies_in (m : line α) (a : plane α) : Prop := ∀ x, m x → a x

theorem incorrect_option_C (h : lies_in m a) : ¬ (parallel m n ∧ lies_in m a → parallel n a) :=
sorry

end incorrect_option_C_l150_150467


namespace rectangle_area_k_l150_150795

noncomputable def rectangle_k (d : ℝ) (length width : ℝ) (ratio : length / width = 5 / 2) 
  (diagonal : d = Real.sqrt (length^2 + width^2)) : ℝ := 
  (length * width / d^2)

theorem rectangle_area_k {d length width : ℝ} 
  (h_ratio : length / width = 5 / 2)
  (h_diagonal : d = Real.sqrt (length^2 + width^2)) :
  rectangle_k d length width h_ratio h_diagonal = 10 / 29 :=
by
  sorry

end rectangle_area_k_l150_150795


namespace game_show_prizes_count_l150_150214

theorem game_show_prizes_count:
  let digits := [1, 1, 1, 1, 3, 3, 3, 3]
  let is_valid_prize (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9999
  let is_three_digit_or_more (n : ℕ) : Prop := 100 ≤ n
  ∃ (A B C : ℕ), 
    is_valid_prize A ∧ is_valid_prize B ∧ is_valid_prize C ∧
    is_three_digit_or_more C ∧
    (A + B + C = digits.sum) ∧
    (A + B + C = 1260) := sorry

end game_show_prizes_count_l150_150214


namespace find_value_of_expr_l150_150259

variables (a b : ℝ)

def condition1 : Prop := a^2 + a * b = -2
def condition2 : Prop := b^2 - 3 * a * b = -3

theorem find_value_of_expr (h1 : condition1 a b) (h2 : condition2 a b) : a^2 + 4 * a * b - b^2 = 1 :=
sorry

end find_value_of_expr_l150_150259


namespace debt_calculation_correct_l150_150868

-- Conditions
def initial_debt : ℤ := 40
def repayment : ℤ := initial_debt / 2
def additional_borrowing : ℤ := 10

-- Final Debt Calculation
def remaining_debt : ℤ := initial_debt - repayment
def final_debt : ℤ := remaining_debt + additional_borrowing

-- Proof Statement
theorem debt_calculation_correct : final_debt = 30 := 
by 
  -- Skipping the proof
  sorry

end debt_calculation_correct_l150_150868


namespace simplify_expression_l150_150485

theorem simplify_expression : 9 * (12 / 7) * ((-35) / 36) = -15 := by
  sorry

end simplify_expression_l150_150485


namespace one_fourth_more_than_x_equals_twenty_percent_less_than_80_l150_150066

theorem one_fourth_more_than_x_equals_twenty_percent_less_than_80 :
  ∃ n : ℝ, (80 - 0.30 * 80 = 56) ∧ (5 / 4 * n = 56) ∧ (n = 45) :=
by
  sorry

end one_fourth_more_than_x_equals_twenty_percent_less_than_80_l150_150066


namespace value_of_each_gift_card_l150_150765

theorem value_of_each_gift_card (students total_thank_you_cards with_gift_cards total_value : ℕ) 
  (h1 : students = 50)
  (h2 : total_thank_you_cards = 30 * students / 100)
  (h3 : with_gift_cards = total_thank_you_cards / 3)
  (h4 : total_value = 50) :
  total_value / with_gift_cards = 10 := by
  sorry

end value_of_each_gift_card_l150_150765


namespace valid_parameterizations_l150_150339

theorem valid_parameterizations (y x : ℝ) (t : ℝ) :
  let A := (⟨0, 4⟩ : ℝ × ℝ) + t • (⟨3, 1⟩ : ℝ × ℝ)
  let B := (⟨-4/3, 0⟩ : ℝ × ℝ) + t • (⟨-1, -3⟩ : ℝ × ℝ)
  let C := (⟨1, 7⟩ : ℝ × ℝ) + t • (⟨9, 3⟩ : ℝ × ℝ)
  let D := (⟨2, 10⟩ : ℝ × ℝ) + t • (⟨1/3, 1⟩ : ℝ × ℝ)
  let E := (⟨-4, -8⟩ : ℝ × ℝ) + t • (⟨1/9, 1/3⟩ : ℝ × ℝ)
  (B = (x, y) ∧ D = (x, y) ∧ E = (x, y)) ↔ y = 3 * x + 4 :=
sorry

end valid_parameterizations_l150_150339


namespace cube_dimension_l150_150510

theorem cube_dimension (x s : ℝ) (hx1 : s^3 = 8 * x) (hx2 : 6 * s^2 = 2 * x) : x = 1728 := 
by {
  sorry
}

end cube_dimension_l150_150510


namespace avg_class_l150_150597

-- Problem definitions
def total_students : ℕ := 40
def num_students_95 : ℕ := 8
def num_students_0 : ℕ := 5
def num_students_70 : ℕ := 10
def avg_remaining_students : ℝ := 50

-- Assuming we have these marks
def marks_95 : ℝ := 95
def marks_0 : ℝ := 0
def marks_70 : ℝ := 70

-- We need to prove that the total average is 57.75 given the above conditions
theorem avg_class (h1 : total_students = 40)
                  (h2 : num_students_95 = 8)
                  (h3 : num_students_0 = 5)
                  (h4 : num_students_70 = 10)
                  (h5 : avg_remaining_students = 50)
                  (h6 : marks_95 = 95)
                  (h7 : marks_0 = 0)
                  (h8 : marks_70 = 70) :
                  (8 * 95 + 5 * 0 + 10 * 70 + 50 * (40 - (8 + 5 + 10))) / 40 = 57.75 :=
by sorry

end avg_class_l150_150597


namespace find_other_number_l150_150595

theorem find_other_number (n : ℕ) (h_lcm : Nat.lcm 12 n = 60) (h_hcf : Nat.gcd 12 n = 3) : n = 15 := by
  sorry

end find_other_number_l150_150595


namespace find_q_minus_p_l150_150150

theorem find_q_minus_p (p q : ℕ) (h1 : 0 < p) (h2 : 0 < q) 
  (h3 : 6 * q < 11 * p) (h4 : 9 * p < 5 * q) (h_min : ∀ r : ℕ, r > 0 → (6:ℚ)/11 < (p:ℚ)/r → (p:ℚ)/r < (5:ℚ)/9 → q ≤ r) :
  q - p = 9 :=
sorry

end find_q_minus_p_l150_150150


namespace radius_of_circle_l150_150171

/-- Given the equation of a circle x^2 + y^2 - 8 = 2x + 4y,
    we need to prove that the radius of the circle is sqrt 13. -/
theorem radius_of_circle : 
    ∃ r : ℝ, (∀ x y : ℝ, x^2 + y^2 - 8 = 2*x + 4*y → r = Real.sqrt 13) :=
by
    sorry

end radius_of_circle_l150_150171


namespace find_integer_solutions_l150_150801

theorem find_integer_solutions (n : ℕ) (h1 : ∃ b : ℤ, 8 * n - 7 = b^2) (h2 : ∃ a : ℤ, 18 * n - 35 = a^2) : 
  n = 2 ∨ n = 22 := 
sorry

end find_integer_solutions_l150_150801


namespace quadratic_real_solution_l150_150708

theorem quadratic_real_solution (m : ℝ) (i : ℂ) (h_i : i * i = -1)
  (h_quad : ∃ z : ℝ, z^2 + (i * z) + m = 0) : m = 0 :=
sorry

end quadratic_real_solution_l150_150708


namespace Adam_ate_more_than_Bill_l150_150656

-- Definitions
def Sierra_ate : ℕ := 12
def Bill_ate : ℕ := Sierra_ate / 2
def total_pies_eaten : ℕ := 27
def Sierra_and_Bill_ate : ℕ := Sierra_ate + Bill_ate
def Adam_ate : ℕ := total_pies_eaten - Sierra_and_Bill_ate
def Adam_more_than_Bill : ℕ := Adam_ate - Bill_ate

-- Statement to prove
theorem Adam_ate_more_than_Bill :
  Adam_more_than_Bill = 3 :=
by
  sorry

end Adam_ate_more_than_Bill_l150_150656


namespace radian_measure_of_minute_hand_rotation_l150_150664

theorem radian_measure_of_minute_hand_rotation :
  ∀ (t : ℝ), (t = 10) → (2 * π / 60 * t = -π/3) := by
  sorry

end radian_measure_of_minute_hand_rotation_l150_150664


namespace tricycle_wheels_l150_150353

theorem tricycle_wheels (T : ℕ) 
  (h1 : 3 * 2 = 6) 
  (h2 : 7 * 1 = 7) 
  (h3 : 6 + 7 + 4 * T = 25) : T = 3 :=
sorry

end tricycle_wheels_l150_150353


namespace find_first_number_l150_150780

theorem find_first_number (y x : ℤ) (h1 : (y + 76 + x) / 3 = 5) (h2 : x = -63) : y = 2 :=
by
  -- To be filled in with the proof steps
  sorry

end find_first_number_l150_150780


namespace one_fourth_of_7point2_is_9div5_l150_150240

theorem one_fourth_of_7point2_is_9div5 : (7.2 / 4 : ℚ) = 9 / 5 := 
by sorry

end one_fourth_of_7point2_is_9div5_l150_150240


namespace total_chapters_read_l150_150904

def books_read : ℕ := 12
def chapters_per_book : ℕ := 32

theorem total_chapters_read : books_read * chapters_per_book = 384 :=
by
  sorry

end total_chapters_read_l150_150904


namespace problem_integer_condition_l150_150329

theorem problem_integer_condition (a : ℤ) (h1 : 0 ≤ a ∧ a ≤ 14)
  (h2 : (235935623 * 74^0 + 2 * 74^1 + 6 * 74^2 + 5 * 74^3 + 3 * 74^4 + 9 * 74^5 + 
         5 * 74^6 + 3 * 74^7 + 2 * 74^8 - a) % 15 = 0) : a = 0 :=
by
  sorry

end problem_integer_condition_l150_150329


namespace average_GPA_of_class_l150_150175

theorem average_GPA_of_class (n : ℕ) (h1 : n > 0) 
  (GPA1 : ℝ := 60) (GPA2 : ℝ := 66) 
  (students_ratio1 : ℝ := 1 / 3) (students_ratio2 : ℝ := 2 / 3) :
  let total_students := (students_ratio1 * n + students_ratio2 * n)
  let total_GPA := (students_ratio1 * n * GPA1 + students_ratio2 * n * GPA2)
  let average_GPA := total_GPA / total_students
  average_GPA = 64 := by
    sorry

end average_GPA_of_class_l150_150175


namespace stone_breadth_l150_150215

theorem stone_breadth 
  (hall_length_m : ℕ) (hall_breadth_m : ℕ)
  (stone_length_dm : ℕ) (num_stones : ℕ)
  (hall_area_dm2 : ℕ) (stone_area_dm2 : ℕ) 
  (hall_length_dm hall_breadth_dm : ℕ) (b : ℕ) :
  hall_length_m = 36 → hall_breadth_m = 15 →
  stone_length_dm = 8 → num_stones = 1350 →
  hall_length_dm = hall_length_m * 10 → hall_breadth_dm = hall_breadth_m * 10 →
  hall_area_dm2 = hall_length_dm * hall_breadth_dm →
  stone_area_dm2 = stone_length_dm * b →
  hall_area_dm2 = num_stones * stone_area_dm2 →
  b = 5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  -- Proof would go here
  sorry

end stone_breadth_l150_150215


namespace multiply_large_numbers_l150_150075

theorem multiply_large_numbers :
  72519 * 9999 = 724817481 :=
by
  sorry

end multiply_large_numbers_l150_150075


namespace ratio_of_speeds_l150_150987

-- Conditions
def total_distance_Eddy : ℕ := 200 + 240 + 300
def total_distance_Freddy : ℕ := 180 + 420
def total_time_Eddy : ℕ := 5
def total_time_Freddy : ℕ := 6

-- Average speeds
def avg_speed_Eddy (d t : ℕ) : ℚ := d / t
def avg_speed_Freddy (d t : ℕ) : ℚ := d / t

-- Ratio of average speeds
def ratio_speeds (s1 s2 : ℚ) : ℚ := s1 / s2

theorem ratio_of_speeds : 
  ratio_speeds (avg_speed_Eddy total_distance_Eddy total_time_Eddy) 
               (avg_speed_Freddy total_distance_Freddy total_time_Freddy) 
  = 37 / 25 := by
  -- Proof omitted
  sorry

end ratio_of_speeds_l150_150987


namespace subset_A_implies_a_subset_B_implies_range_a_l150_150253

variable (a : ℝ)

def A : Set ℝ := {x | x^2 - 2*x - 8 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a^2 - 12 = 0}

theorem subset_A_implies_a (h : A ⊆ B a) : a = -2 := 
sorry

theorem subset_B_implies_range_a (h : B a ⊆ A) : a >= 4 ∨ a < -4 ∨ a = -2 := 
sorry

end subset_A_implies_a_subset_B_implies_range_a_l150_150253


namespace abs_difference_extrema_l150_150944

theorem abs_difference_extrema (x : ℝ) (h : 2 ≤ x ∧ x < 3) :
  max (|x-2| + |x-3| - |x-1|) = 0 ∧ min (|x-2| + |x-3| - |x-1|) = -1 :=
by
  sorry

end abs_difference_extrema_l150_150944


namespace gain_per_year_is_200_l150_150665

noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem gain_per_year_is_200 :
  let borrowed_principal := 5000
  let borrowing_rate := 4
  let borrowing_time := 2
  let lent_principal := 5000
  let lending_rate := 8
  let lending_time := 2

  let interest_paid := simple_interest borrowed_principal borrowing_rate borrowing_time
  let interest_earned := simple_interest lent_principal lending_rate lending_time

  let total_gain := interest_earned - interest_paid
  let gain_per_year := total_gain / 2

  gain_per_year = 200 := by
  sorry

end gain_per_year_is_200_l150_150665


namespace number_of_subsets_with_exactly_one_isolated_element_l150_150623

def is_isolated_element (A : Finset ℤ) (k : ℤ) : Prop :=
  k ∈ A ∧ k - 1 ∉ A ∧ k + 1 ∉ A

def has_exactly_one_isolated_element (A : Finset ℤ) (B : Finset ℤ) : Prop :=
  ∃ k ∈ B, is_isolated_element B k ∧ ∀ m ∈ B, m ≠ k → ¬ is_isolated_element B m

theorem number_of_subsets_with_exactly_one_isolated_element :
  let A := ({1, 2, 3, 4, 5} : Finset ℤ) in
  (A.powerset.filter (has_exactly_one_isolated_element A)).card = 13 :=
by
  sorry

end number_of_subsets_with_exactly_one_isolated_element_l150_150623


namespace sales_tax_amount_l150_150309

variable (T : ℝ := 25) -- Total amount spent
variable (y : ℝ := 19.7) -- Cost of tax-free items
variable (r : ℝ := 0.06) -- Tax rate

theorem sales_tax_amount : 
  ∃ t : ℝ, t = 0.3 ∧ (T - y) * r = t :=
by 
  sorry

end sales_tax_amount_l150_150309


namespace loss_percent_l150_150973

theorem loss_percent (cost_price selling_price loss_percent : ℝ) 
  (h_cost_price : cost_price = 600)
  (h_selling_price : selling_price = 550)
  (h_loss_percent : loss_percent = 8.33) : 
  (loss_percent = ((cost_price - selling_price) / cost_price) * 100) := 
by
  rw [h_cost_price, h_selling_price]
  sorry

end loss_percent_l150_150973


namespace sophie_hours_needed_l150_150777

-- Sophie needs 206 hours to finish the analysis of all bones.
theorem sophie_hours_needed (num_bones : ℕ) (time_per_bone : ℕ) (total_hours : ℕ) (h1 : num_bones = 206) (h2 : time_per_bone = 1) : 
  total_hours = num_bones * time_per_bone :=
by
  rw [h1, h2]
  norm_num
  sorry

end sophie_hours_needed_l150_150777


namespace cricketer_stats_l150_150962

theorem cricketer_stats :
  let total_runs := 225
  let total_balls := 120
  let boundaries := 4 * 15
  let sixes := 6 * 8
  let twos := 2 * 3
  let singles := 1 * 10
  let perc_boundaries := (boundaries / total_runs.toFloat) * 100
  let perc_sixes := (sixes / total_runs.toFloat) * 100
  let perc_twos := (twos / total_runs.toFloat) * 100
  let perc_singles := (singles / total_runs.toFloat) * 100
  let strike_rate := (total_runs.toFloat / total_balls.toFloat) * 100
  perc_boundaries = 26.67 ∧
  perc_sixes = 21.33 ∧
  perc_twos = 2.67 ∧
  perc_singles = 4.44 ∧
  strike_rate = 187.5 :=
by
  sorry

end cricketer_stats_l150_150962


namespace scientific_notation_100000_l150_150322

theorem scientific_notation_100000 : ∃ a n, (1 ≤ a) ∧ (a < 10) ∧ (100000 = a * 10 ^ n) :=
by
  use 1, 5
  repeat { split }
  repeat { sorry }

end scientific_notation_100000_l150_150322


namespace tan_alpha_value_l150_150425

open Real

theorem tan_alpha_value
  (α : ℝ)
  (h₀ : 0 < α)
  (h₁ : α < π / 2)
  (h₂ : cos (2 * α) = (2 * sqrt 5 / 5) * sin (α + π / 4)) :
  tan α = 1 / 3 :=
sorry

end tan_alpha_value_l150_150425


namespace fedora_cleaning_time_l150_150239

-- Definitions based on given conditions
def cleaning_time_per_section (total_time sections_cleaned : ℕ) : ℕ :=
  total_time / sections_cleaned

def remaining_sections (total_sections cleaned_sections : ℕ) : ℕ :=
  total_sections - cleaned_sections

def total_cleaning_time (remaining_sections time_per_section : ℕ) : ℕ :=
  remaining_sections * time_per_section

-- Theorem statement
theorem fedora_cleaning_time 
  (total_time : ℕ) 
  (sections_cleaned : ℕ)
  (additional_time : ℕ)
  (additional_sections : ℕ)
  (cleaned_sections : ℕ)
  (total_sections : ℕ)
  (h1 : total_time = 33)
  (h2 : sections_cleaned = 3)
  (h3 : additional_time = 165)
  (h4 : additional_sections = 15)
  (h5 : cleaned_sections = 3)
  (h6 : total_sections = 18)
  (h7 : cleaning_time_per_section total_time sections_cleaned = 11)
  (h8 : remaining_sections total_sections cleaned_sections = additional_sections)
  : total_cleaning_time additional_sections (cleaning_time_per_section total_time sections_cleaned) = additional_time := sorry

end fedora_cleaning_time_l150_150239


namespace remainder_8347_div_9_l150_150365
-- Import all necessary Mathlib modules

-- Define the problem and conditions
theorem remainder_8347_div_9 : (8347 % 9) = 4 :=
by
  -- To ensure the code builds successfully and contains a placeholder for the proof
  sorry

end remainder_8347_div_9_l150_150365


namespace six_digit_phone_number_count_l150_150996

def six_digit_to_seven_digit_count (six_digit : ℕ) (h : 100000 ≤ six_digit ∧ six_digit < 1000000) : ℕ :=
  let num_positions := 7
  let num_digits := 10
  num_positions * num_digits

theorem six_digit_phone_number_count (six_digit : ℕ) (h : 100000 ≤ six_digit ∧ six_digit < 1000000) :
  six_digit_to_seven_digit_count six_digit h = 70 := by
  -- Proof goes here
  sorry

end six_digit_phone_number_count_l150_150996


namespace moe_pie_share_l150_150231

theorem moe_pie_share
  (leftover_pie : ℚ)
  (num_people : ℕ)
  (H_leftover : leftover_pie = 5 / 8)
  (H_people : num_people = 4) :
  (leftover_pie / num_people = 5 / 32) :=
by
  sorry

end moe_pie_share_l150_150231


namespace line_dividing_circle_maximizes_area_l150_150014

noncomputable theory

-- Define the circular region and the point P
def circular_region (x y : ℝ) : Prop := x^2 + y^2 ≤ 4
def point_P : ℝ × ℝ := (1, 1)

-- Define the desired result
def optimal_line_eq (x y : ℝ) : Prop := x + y - 2 = 0

-- Formal statement of the problem
theorem line_dividing_circle_maximizes_area : 
  ∃ (x y : ℝ), (circular_region x y ∧ (x, y) = point_P) → optimal_line_eq x y :=
by sorry

end line_dividing_circle_maximizes_area_l150_150014


namespace daniel_total_spent_l150_150695

/-
Daniel buys various items with given prices, receives a 10% coupon discount,
a store credit of $1.50, a 5% student discount, and faces a 6.5% sales tax.
Prove that the total amount he spends is $8.23.
-/
def total_spent (prices : List ℝ) (coupon_discount store_credit student_discount sales_tax : ℝ) : ℝ :=
  let initial_total := prices.sum
  let after_coupon := initial_total * (1 - coupon_discount)
  let after_student := after_coupon * (1 - student_discount)
  let after_store_credit := after_student - store_credit
  let final_total := after_store_credit * (1 + sales_tax)
  final_total

theorem daniel_total_spent :
  total_spent 
    [0.85, 0.50, 1.25, 3.75, 2.99, 1.45] -- prices of items
    0.10 -- 10% coupon discount
    1.50 -- $1.50 store credit
    0.05 -- 5% student discount
    0.065 -- 6.5% sales tax
  = 8.23 :=
by
  sorry

end daniel_total_spent_l150_150695


namespace max_and_min_values_g_l150_150945

noncomputable def f (x : ℝ) : ℝ := abs (x - 2) + abs (x - 3)
noncomputable def g (x : ℝ) : ℝ := abs (x - 2) + abs (x - 3) - abs (x - 1)

theorem max_and_min_values_g :
  (∀ x, (2 ≤ x ∧ x ≤ 3) → f x = 1) →
  (∃ a b, (∀ x, (2 ≤ x ∧ x ≤ 3) → a ≤ g x ∧ g x ≤ b) ∧ a = -1 ∧ b = 0) :=
by
  intros H
  use [-1, 0]
  split
  sorry
  sorry

end max_and_min_values_g_l150_150945


namespace smallest_x_for_perfect_cube_l150_150406

theorem smallest_x_for_perfect_cube (M : ℤ) :
  ∃ x : ℕ, 1680 * x = M^3 ∧ ∀ y : ℕ, 1680 * y = M^3 → 44100 ≤ y := 
sorry

end smallest_x_for_perfect_cube_l150_150406


namespace math_problem_l150_150276

theorem math_problem
  (a b c x1 x2 : ℝ)
  (h1 : a > 0)
  (h2 : a^2 = 4 * b)
  (h3 : |x1 - x2| = 4)
  (h4 : x1 < x2) :
  (a^2 - b^2 ≤ 4) ∧ (a^2 + 1 / b ≥ 4) ∧ (c = 4) :=
by
  sorry

end math_problem_l150_150276


namespace area_of_abs_inequality_l150_150566

theorem area_of_abs_inequality :
  (setOf (λ (p : ℝ×ℝ), |p.1 + p.2| + |p.1 - p.2| ≤ 6)).measure = 36 :=
sorry

end area_of_abs_inequality_l150_150566


namespace painting_time_l150_150772

noncomputable def work_rate (t : ℕ) : ℚ := 1 / t

theorem painting_time (shawn_time karen_time alex_time total_work_rate : ℚ)
  (h_shawn : shawn_time = 18)
  (h_karen : karen_time = 12)
  (h_alex : alex_time = 15) :
  total_work_rate = 1 / (shawn_time + karen_time + alex_time) :=
by
  sorry

end painting_time_l150_150772


namespace volume_pyramid_PABCD_is_384_l150_150771

noncomputable def volume_of_pyramid : ℝ :=
  let AB := 12
  let BC := 6
  let PA := Real.sqrt (20^2 - 12^2)
  let base_area := AB * BC
  (1 / 3) * base_area * PA

theorem volume_pyramid_PABCD_is_384 :
  volume_of_pyramid = 384 := 
by
  sorry

end volume_pyramid_PABCD_is_384_l150_150771


namespace final_result_is_102_l150_150387

-- Definitions and conditions from the problem
def chosen_number : ℕ := 120
def multiplied_result : ℕ := 2 * chosen_number
def final_result : ℕ := multiplied_result - 138

-- The proof statement
theorem final_result_is_102 : final_result = 102 := 
by 
sorry

end final_result_is_102_l150_150387


namespace find_AX_l150_150746

theorem find_AX (AB AC BC : ℝ) (CX_bisects_ACB : Prop) (h1 : AB = 50) (h2 : AC = 28) (h3 : BC = 56) : AX = 50 / 3 :=
by
  -- Proof can be added here
  sorry

end find_AX_l150_150746


namespace pieces_equality_l150_150105

-- Define the pieces of chocolate and their areas.
def piece1_area : ℝ := 6 -- Area of triangle EBC
def piece2_area : ℝ := 6 -- Area of triangle AEC
def piece3_area : ℝ := 6 -- Area of polygon AHGFD
def piece4_area : ℝ := 6 -- Area of polygon CFGH

-- State the problem: proving the equality of the areas.
theorem pieces_equality : piece1_area = piece2_area ∧ piece2_area = piece3_area ∧ piece3_area = piece4_area :=
by
  sorry

end pieces_equality_l150_150105


namespace martin_travel_time_l150_150316

-- Definitions based on the conditions
def distance : ℕ := 12
def speed : ℕ := 2

-- Statement of the problem to be proven
theorem martin_travel_time : (distance / speed) = 6 := by sorry

end martin_travel_time_l150_150316


namespace find_unit_prices_l150_150096

-- Define the prices of brush and chess set
variables (x y : ℝ)

-- Condition 1: Buying 5 brushes and 12 chess sets costs 315 yuan
def condition1 : Prop := 5 * x + 12 * y = 315

-- Condition 2: Buying 8 brushes and 6 chess sets costs 240 yuan
def condition2 : Prop := 8 * x + 6 * y = 240

-- Prove that the unit price of each brush is 15 yuan and each chess set is 20 yuan
theorem find_unit_prices (hx : condition1 x y) (hy : condition2 x y) :
  x = 15 ∧ y = 20 := 
sorry

end find_unit_prices_l150_150096


namespace vectors_perpendicular_l150_150022

def vec (a b : ℝ) := (a, b)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

@[simp]
def a := vec (-1) 2
@[simp]
def b := vec 1 3

theorem vectors_perpendicular :
  dot_product a (vector_sub a b) = 0 := by
  sorry

end vectors_perpendicular_l150_150022


namespace cube_volume_surface_area_eq_1728_l150_150516

theorem cube_volume_surface_area_eq_1728 (x : ℝ) (h1 : (side : ℝ) (v : ℝ) hvolume : v = 8 * x ∧ v = side^3) (h2: (side : ℝ) (a : ℝ) hsurface : a = 2 * x ∧ a = 6 * side^2) : x = 1728 :=
by
  sorry

end cube_volume_surface_area_eq_1728_l150_150516


namespace exists_isosceles_triangle_containing_l150_150255

variables {A B C X Y Z : Type} [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]

noncomputable def triangle (a b c : A) := a + b + c

def is_triangle (a b c : A) := a + b > c ∧ b + c > a ∧ c + a > b

def isosceles_triangle (a b c : A) := (a = b ∨ b = c ∨ c = a) ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem exists_isosceles_triangle_containing
  (a b c : A)
  (h1 : a < 1)
  (h2 : b < 1)
  (h3 : c < 1)
  (h_ABC : is_triangle a b c)
  : ∃ (x y z : A), isosceles_triangle x y z ∧ is_triangle x y z ∧ a < x ∧ b < y ∧ c < z ∧ x < 1 ∧ y < 1 ∧ z < 1 :=
sorry

end exists_isosceles_triangle_containing_l150_150255


namespace correct_graph_for_race_l150_150682

-- Define the conditions for the race.
def tortoise_constant_speed (d t : ℝ) := 
  ∃ k : ℝ, k > 0 ∧ d = k * t

def hare_behavior (d t t_nap t_end d_nap : ℝ) :=
  ∃ k1 k2 : ℝ, k1 > 0 ∧ k2 > 0 ∧ t_nap > 0 ∧ t_end > t_nap ∧
  (d = k1 * t ∨ (t_nap < t ∧ t < t_end ∧ d = d_nap) ∨ (t_end ≥ t ∧ d = d_nap + k2 * (t - t_end)))

-- Define the competition outcome.
def tortoise_wins (d_tortoise d_hare : ℝ) :=
  d_tortoise > d_hare

-- Proof that the graph which describes the race is Option (B).
theorem correct_graph_for_race :
  ∃ d_t d_h t t_nap t_end d_nap, 
    tortoise_constant_speed d_t t ∧ hare_behavior d_h t t_nap t_end d_nap ∧ tortoise_wins d_t d_h → "Option B" = "correct" :=
sorry -- Proof omitted.

end correct_graph_for_race_l150_150682


namespace sequence_contains_2017_l150_150392

theorem sequence_contains_2017 (a1 d : ℕ) (hpos : d > 0)
  (k n m l : ℕ) 
  (hk : 25 = a1 + k * d)
  (hn : 41 = a1 + n * d)
  (hm : 65 = a1 + m * d)
  (h2017 : 2017 = a1 + l * d) : l > 0 :=
sorry

end sequence_contains_2017_l150_150392


namespace unpacked_boxes_l150_150251

-- Definitions of boxes per case
def boxesPerCaseLemonChalet : Nat := 12
def boxesPerCaseThinMints : Nat := 15
def boxesPerCaseSamoas : Nat := 10
def boxesPerCaseTrefoils : Nat := 18

-- Definitions of boxes sold by Deborah
def boxesSoldLemonChalet : Nat := 31
def boxesSoldThinMints : Nat := 26
def boxesSoldSamoas : Nat := 17
def boxesSoldTrefoils : Nat := 44

-- The theorem stating the number of boxes that will not be packed to a case
theorem unpacked_boxes :
  boxesSoldLemonChalet % boxesPerCaseLemonChalet = 7 ∧
  boxesSoldThinMints % boxesPerCaseThinMints = 11 ∧
  boxesSoldSamoas % boxesPerCaseSamoas = 7 ∧
  boxesSoldTrefoils % boxesPerCaseTrefoils = 8 := 
by
  sorry

end unpacked_boxes_l150_150251


namespace least_positive_integer_divisible_by_5_to_15_l150_150246

def is_divisible_by_all (n : ℕ) (l : List ℕ) : Prop :=
  ∀ m ∈ l, m ∣ n

theorem least_positive_integer_divisible_by_5_to_15 :
  ∃ n : ℕ, n > 0 ∧ is_divisible_by_all n [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] ∧
  ∀ m : ℕ, m > 0 ∧ is_divisible_by_all m [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] → n ≤ m ∧ n = 360360 :=
by
  sorry

end least_positive_integer_divisible_by_5_to_15_l150_150246


namespace dmitry_black_socks_l150_150986

theorem dmitry_black_socks (b : ℕ) : 
  let blue_socks := 14
  let initial_black_socks := 24
  let white_socks := 10
  let initial_total_socks := blue_socks + initial_black_socks + white_socks
  let new_total_socks := initial_total_socks + b
  let new_black_socks := initial_black_socks + b
  (new_black_socks : ℚ) = (3 / 5) * new_total_socks → b = 12 :=
by
  intros
  let initial_total_socks := 14 + 24 + 10
  let new_total_socks := initial_total_socks + b
  let new_black_socks := 24 + b
  suffices : (new_black_socks : ℚ) = (3 / 5) * new_total_socks → b = 12
  sorry

end dmitry_black_socks_l150_150986


namespace simplified_evaluated_expression_l150_150773

noncomputable def a : ℚ := 1 / 3
noncomputable def b : ℚ := 1 / 2
noncomputable def c : ℚ := 1

def expression (a b c : ℚ) : ℚ := a^2 + 2 * b - c

theorem simplified_evaluated_expression :
  expression a b c = 1 / 9 :=
by
  sorry

end simplified_evaluated_expression_l150_150773


namespace range_m_l150_150720

noncomputable def circle_c (x y : ℝ) : Prop := (x - 4) ^ 2 + (y - 3) ^ 2 = 4

def point_A (m : ℝ) : ℝ × ℝ := (-m, 0)
def point_B (m : ℝ) : ℝ × ℝ := (m, 0)

theorem range_m (m : ℝ) (P : ℝ × ℝ) :
  circle_c P.1 P.2 ∧ m > 0 ∧ (∃ (a b : ℝ), P = (a, b) ∧ (a + m) * (a - m) + b ^ 2 = 0) → m ∈ Set.Icc 3 7 :=
sorry

end range_m_l150_150720


namespace cube_edge_length_l150_150653

theorem cube_edge_length {e : ℝ} (h : 12 * e = 108) : e = 9 :=
by sorry

end cube_edge_length_l150_150653


namespace unique_parallelogram_l150_150479

theorem unique_parallelogram :
  ∃! (A B D C : ℤ × ℤ), 
  A = (0, 0) ∧ 
  (B.2 = B.1) ∧ 
  (D.2 = 2 * D.1) ∧ 
  (C.2 = 3 * C.1) ∧ 
  (A.1 = 0 ∧ A.2 = 0) ∧ 
  (B.1 > 0 ∧ B.2 > 0) ∧ 
  (D.1 > 0 ∧ D.2 > 0) ∧ 
  (C.1 > 0 ∧ C.2 > 0) ∧ 
  (B.1 - A.1, B.2 - A.2) + (D.1 - A.1, D.2 - A.2) = (C.1 - A.1, C.2 - A.2) ∧
  (abs ((B.1 * C.2 + C.1 * D.2 + D.1 * A.2 + A.1 * B.2) - (A.1 * C.2 + B.1 * D.2 + C.1 * B.2 + D.1 * A.2)) / 2) = 2000000 
  := by sorry

end unique_parallelogram_l150_150479


namespace two_less_than_six_times_l150_150349

theorem two_less_than_six_times {x : ℤ} (h : x + (x - 1) = 33) : 6 * x - 2 = 100 :=
by
  sorry

end two_less_than_six_times_l150_150349


namespace eval_sqrt_4_8_pow_12_l150_150111

theorem eval_sqrt_4_8_pow_12 : ((8 : ℝ)^(1 / 4))^12 = 512 :=
by
  -- This is where the proof steps would go 
  sorry

end eval_sqrt_4_8_pow_12_l150_150111


namespace max_notebooks_no_more_than_11_l150_150168

noncomputable def maxNotebooks (money : ℕ) (cost_single : ℕ) (cost_pack4 : ℕ) (cost_pack7 : ℕ) (max_pack7 : ℕ) : ℕ :=
if money >= cost_pack7 then
  if (money - cost_pack7) >= cost_pack4 then 7 + 4
  else if (money - cost_pack7) >= cost_single then 7 + 1
  else 7
else if money >= cost_pack4 then
  if (money - cost_pack4) >= cost_pack4 then 4 + 4
  else if (money - cost_pack4) >= cost_single then 4 + 1
  else 4
else
  money / cost_single

theorem max_notebooks_no_more_than_11 :
  maxNotebooks 15 2 6 9 1 = 11 :=
by
  sorry

end max_notebooks_no_more_than_11_l150_150168


namespace min_1x1_tiles_l150_150361

/-- To cover a 23x23 grid using 1x1, 2x2, and 3x3 tiles (without gaps or overlaps),
the minimum number of 1x1 tiles required is 1. -/
theorem min_1x1_tiles (a b c : ℕ) (h : a + 2 * b + 3 * c = 23 * 23) : 
  a ≥ 1 :=
sorry

end min_1x1_tiles_l150_150361


namespace delta_value_l150_150439

theorem delta_value (Delta : ℤ) (h : 5 * (-3) = Delta - 3) : Delta = -12 := 
by 
  sorry

end delta_value_l150_150439


namespace black_white_ratio_l150_150702

theorem black_white_ratio :
  let original_black := 18
  let original_white := 39
  let replaced_black := original_black + 13
  let inner_border_black := (9^2 - 7^2)
  let outer_border_white := (11^2 - 9^2)
  let total_black := replaced_black + inner_border_black
  let total_white := original_white + outer_border_white
  let ratio_black_white := total_black / total_white
  ratio_black_white = 63 / 79 :=
sorry

end black_white_ratio_l150_150702


namespace polygon_with_equal_angle_sums_is_quadrilateral_l150_150877

theorem polygon_with_equal_angle_sums_is_quadrilateral 
    (n : ℕ)
    (h1 : (n - 2) * 180 = 360)
    (h2 : 360 = 360) :
  n = 4 := 
sorry

end polygon_with_equal_angle_sums_is_quadrilateral_l150_150877


namespace remainder_of_5_pow_2023_mod_6_l150_150192

theorem remainder_of_5_pow_2023_mod_6 : 5^2023 % 6 = 5 := 
by sorry

end remainder_of_5_pow_2023_mod_6_l150_150192


namespace men_count_in_first_group_is_20_l150_150083

noncomputable def men_needed_to_build_fountain (work1 : ℝ) (days1 : ℕ) (length1 : ℝ) (workers2 : ℕ) (days2 : ℕ) (length2 : ℝ) (work_per_man_per_day2 : ℝ) : ℕ :=
  let work_per_day2 := length2 / days2
  let work_per_man_per_day2 := work_per_day2 / workers2
  let total_work1 := length1 / days1
  Nat.floor (total_work1 / work_per_man_per_day2)

theorem men_count_in_first_group_is_20 :
  men_needed_to_build_fountain 56 6 56 35 3 49 (49 / (35 * 3)) = 20 :=
by
  sorry

end men_count_in_first_group_is_20_l150_150083


namespace length_of_living_room_l150_150178

theorem length_of_living_room (width area : ℝ) (h_width : width = 14) (h_area : area = 215.6) :
  ∃ length : ℝ, length = 15.4 ∧ area = length * width :=
by
  sorry

end length_of_living_room_l150_150178


namespace clarence_to_matthew_ratio_l150_150234

theorem clarence_to_matthew_ratio (D C M : ℝ) (h1 : D = 6.06) (h2 : D = 1 / 2 * C) (h3 : D + C + M = 20.20) : C / M = 6 := 
by 
  sorry

end clarence_to_matthew_ratio_l150_150234


namespace meaningful_fraction_l150_150068

theorem meaningful_fraction (x : ℝ) : (∃ y, y = (1 / (x - 2))) ↔ x ≠ 2 :=
by
  sorry

end meaningful_fraction_l150_150068


namespace choose_9_3_eq_84_l150_150300

theorem choose_9_3_eq_84 : Nat.choose 9 3 = 84 :=
by
  sorry

end choose_9_3_eq_84_l150_150300


namespace smallest_consecutive_cube_x_l150_150889

theorem smallest_consecutive_cube_x:
  ∃ n: ℤ, 
  let u := n-1, v := n, w := n+1, x := n+2 in 
  u^3 + v^3 + w^3 = x^3 → x = 6 :=
  sorry

end smallest_consecutive_cube_x_l150_150889


namespace short_haired_girls_l150_150930

def total_people : ℕ := 55
def boys : ℕ := 30
def total_girls : ℕ := total_people - boys
def girls_with_long_hair : ℕ := (3 / 5) * total_girls
def girls_with_short_hair : ℕ := total_girls - girls_with_long_hair

theorem short_haired_girls :
  girls_with_short_hair = 10 := sorry

end short_haired_girls_l150_150930


namespace helmet_price_for_given_profit_helmet_price_for_max_profit_l150_150963

section helmet_sales

-- Define the conditions
variable (original_price : ℝ := 80) (initial_sales : ℝ := 200) (cost_price : ℝ := 50) 
variable (price_reduction_unit : ℝ := 1) (additional_sales_per_reduction : ℝ := 10)
variable (minimum_price_reduction : ℝ := 10)

-- Profits
def profit (x : ℝ) : ℝ :=
  (original_price - x - cost_price) * (initial_sales + additional_sales_per_reduction * x)

-- Prove the selling price when profit is 5250 yuan
theorem helmet_price_for_given_profit (GDP : profit 15 = 5250) : (original_price - 15) = 65 :=
by
  sorry

-- Prove the price for maximum profit
theorem helmet_price_for_max_profit : 
  ∃ x, x = 10 ∧ (original_price - x = 70) ∧ (profit x = 6000) :=
by 
  sorry

end helmet_sales

end helmet_price_for_given_profit_helmet_price_for_max_profit_l150_150963


namespace y_intercept_of_line_b_is_minus_8_l150_150763

/-- Define a line in slope-intercept form y = mx + c --/
structure Line :=
  (m : ℝ)   -- slope
  (c : ℝ)   -- y-intercept

/-- Define a point in 2D Cartesian coordinate system --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Define conditions for the problem --/
def line_b_parallel_to (l: Line) (p: Point) : Prop :=
  l.m = 2 ∧ p.x = 3 ∧ p.y = -2

/-- Define the target statement to prove --/
theorem y_intercept_of_line_b_is_minus_8 :
  ∀ (b: Line) (p: Point), line_b_parallel_to b p → b.c = -8 := by
  -- proof goes here
  sorry

end y_intercept_of_line_b_is_minus_8_l150_150763


namespace mark_initial_kept_percentage_l150_150154

-- Defining the conditions
def initial_friends : Nat := 100
def remaining_friends : Nat := 70
def percentage_contacted (P : ℝ) := 100 - P
def percentage_responded : ℝ := 0.5

-- Theorem statement: Mark initially kept 40% of his friends
theorem mark_initial_kept_percentage (P : ℝ) : 
  (P / 100 * initial_friends) + (percentage_contacted P / 100 * initial_friends * percentage_responded) = remaining_friends → 
  P = 40 := by
  sorry

end mark_initial_kept_percentage_l150_150154


namespace dan_job_time_l150_150403

theorem dan_job_time
  (Annie_time : ℝ) (Dan_work_time : ℝ) (Annie_work_remain : ℝ) (total_work : ℝ)
  (Annie_time_cond : Annie_time = 9)
  (Dan_work_time_cond : Dan_work_time = 8)
  (Annie_work_remain_cond : Annie_work_remain = 3.0000000000000004)
  (total_work_cond : total_work = 1) :
  ∃ (Dan_time : ℝ), Dan_time = 12 := by
  sorry

end dan_job_time_l150_150403


namespace range_of_a_l150_150136

theorem range_of_a (a : ℝ) :
  (abs (15 - 3 * a) / 5 ≤ 3) → (0 ≤ a ∧ a ≤ 10) :=
by
  intro h
  sorry

end range_of_a_l150_150136


namespace avg_height_students_l150_150333

theorem avg_height_students 
  (x : ℕ)  -- number of students in the first group
  (avg_height_first_group : ℕ)  -- average height of the first group
  (avg_height_second_group : ℕ)  -- average height of the second group
  (avg_height_combined_group : ℕ)  -- average height of the combined group
  (h1 : avg_height_first_group = 20)
  (h2 : avg_height_second_group = 20)
  (h3 : avg_height_combined_group = 20)
  (h4 : 20*x + 20*11 = 20*31) :
  x = 20 := 
  by {
    sorry
  }

end avg_height_students_l150_150333


namespace planes_parallel_if_any_line_parallel_l150_150767

axiom Plane : Type
axiom Line : Type
axiom contains : Plane → Line → Prop
axiom parallel : Plane → Plane → Prop
axiom parallel_lines : Line → Plane → Prop

theorem planes_parallel_if_any_line_parallel (α β : Plane)
  (h₁ : ∀ l, contains α l → parallel_lines l β) :
  parallel α β :=
sorry

end planes_parallel_if_any_line_parallel_l150_150767


namespace matrix_determinant_equiv_l150_150264

variable {x y z w : ℝ}

theorem matrix_determinant_equiv (h : x * w - y * z = 7) :
    (x + 2 * z) * w - (y + 2 * w) * z = 7 :=
by
    sorry

end matrix_determinant_equiv_l150_150264


namespace find_detergent_volume_l150_150184

variable (B D W : ℕ)
variable (B' D' W': ℕ)
variable (water_volume: unit)
variable (detergent_volume: unit)

def original_ratio (B D W : ℕ) : Prop := B = 2 * W / 100 ∧ D = 40 * W / 100

def altered_ratio (B' D' W' B D W : ℕ) : Prop :=
  B' = 3 * B ∧ D' = D / 2 ∧ W' = W ∧ W' = 300

theorem find_detergent_volume {B D W B' D' W'} (h₀ : original_ratio B D W) (h₁ : altered_ratio B' D' W' B D W) :
  D' = 120 :=
sorry

end find_detergent_volume_l150_150184


namespace polynomial_root_recip_squares_l150_150102

theorem polynomial_root_recip_squares (a b c : ℝ) 
  (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 11) (h3 : a * b * c = 6):
  1 / a^2 + 1 / b^2 + 1 / c^2 = 49 / 36 :=
sorry

end polynomial_root_recip_squares_l150_150102


namespace find_angle_A_find_range_expression_l150_150143

-- Define the variables and conditions in a way consistent with Lean's syntax
variables {α β γ : Type}
variables (a b c : ℝ) (A B C : ℝ)

-- The mathematical conditions translated to Lean
def triangle_condition (a b c A B C : ℝ) : Prop := (b + c) / a = Real.cos B + Real.cos C

-- Statement for Proof 1: Prove that A = π/2 given the conditions
theorem find_angle_A (h : triangle_condition a b c A B C) : A = Real.pi / 2 :=
sorry

-- Statement for Proof 2: Prove the range of the given expression under the given conditions
theorem find_range_expression (h : triangle_condition a b c A B C) (hA : A = Real.pi / 2) :
  ∃ (l u : ℝ), l = Real.sqrt 3 + 2 ∧ u = Real.sqrt 3 + 3 ∧ (2 * Real.cos (B / 2) ^ 2 + 2 * Real.sqrt 3 * Real.cos (C / 2) ^ 2) ∈ Set.Ioc l u :=
sorry

end find_angle_A_find_range_expression_l150_150143


namespace red_red_pairs_l150_150599

theorem red_red_pairs (green_shirts red_shirts total_students total_pairs green_green_pairs : ℕ)
    (hg1 : green_shirts = 64)
    (hr1 : red_shirts = 68)
    (htotal : total_students = 132)
    (htotal_pairs : total_pairs = 66)
    (hgreen_green_pairs : green_green_pairs = 28) :
    (total_students = green_shirts + red_shirts) ∧
    (green_green_pairs ≤ total_pairs) ∧
    (∃ red_red_pairs, red_red_pairs = 30) :=
by
  sorry

end red_red_pairs_l150_150599


namespace probability_of_5A_level_spot_probability_of_selecting_b_and_e_l150_150838

-- Proof problem 1
theorem probability_of_5A_level_spot :
  let num_5A_spots := 4
  let num_4A_spots := 6
  let total_spots := num_5A_spots + num_4A_spots
  (num_5A_spots / total_spots) = (2 / 5) :=
by
  let num_5A_spots := 4
  let num_4A_spots := 6
  let total_spots := num_5A_spots + num_4A_spots
  show (num_5A_spots / total_spots) = (2 / 5)
  sorry

-- Proof problem 2
theorem probability_of_selecting_b_and_e :
  let selected_spot := 'a'
  let additional_spots := {'b', 'c', 'd', 'e'}
  let total_combinations := 12
  let favorable_outcomes := 2
  (favorable_outcomes / total_combinations) = (1 / 6) :=
by
  let selected_spot := 'a'
  let additional_spots := {'b', 'c', 'd', 'e'}
  let total_combinations := 12
  let favorable_outcomes := 2
  show (favorable_outcomes / total_combinations) = (1 / 6)
  sorry

end probability_of_5A_level_spot_probability_of_selecting_b_and_e_l150_150838


namespace incorrect_value_at_x5_l150_150190

theorem incorrect_value_at_x5 
  (f : ℕ → ℕ) 
  (provided_values : List ℕ) 
  (h_f : ∀ x, f x = 2 * x ^ 2 + 3 * x + 5)
  (h_provided_values : provided_values = [10, 18, 29, 44, 63, 84, 111, 140]) : 
  ¬ (f 5 = provided_values.get! 4) := 
by
  sorry

end incorrect_value_at_x5_l150_150190


namespace sum_of_squares_of_two_numbers_l150_150495

theorem sum_of_squares_of_two_numbers (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) :
  x^2 + y^2 = 289 := 
  sorry

end sum_of_squares_of_two_numbers_l150_150495


namespace john_allowance_calculation_l150_150617

theorem john_allowance_calculation (initial_money final_money game_cost allowance: ℕ) 
(h_initial: initial_money = 5) 
(h_game_cost: game_cost = 2) 
(h_final: final_money = 29) 
(h_allowance: final_money = initial_money - game_cost + allowance) : 
  allowance = 26 :=
by
  sorry

end john_allowance_calculation_l150_150617


namespace solve_for_x_l150_150077

theorem solve_for_x (x y z : ℤ) (h1 : x + y + z = 14) (h2 : x - y - z = 60) (h3 : x + z = 2 * y) : x = 37 := by
  sorry

end solve_for_x_l150_150077


namespace min_value_expression_l150_150469

theorem min_value_expression (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_cond : x * y * z = 2 / 3) :
  x^2 + 6 * x * y + 18 * y^2 + 12 * y * z + 4 * z^2 ≥ 18 :=
by
  sorry

end min_value_expression_l150_150469


namespace weighted_average_AC_l150_150924

theorem weighted_average_AC (avgA avgB avgC wA wB wC total_weight: ℝ)
  (h_avgA : avgA = 7.3)
  (h_avgB : avgB = 7.6) 
  (h_avgC : avgC = 7.2)
  (h_wA : wA = 3)
  (h_wB : wB = 4)
  (h_wC : wC = 1)
  (h_total_weight : total_weight = 5) :
  ((avgA * wA + avgC * wC) / total_weight) = 5.82 :=
by
  sorry

end weighted_average_AC_l150_150924


namespace least_divisible_by_first_five_primes_l150_150506

-- Conditions
def prime1 := 2
def prime2 := 3
def prime3 := 5
def prime4 := 7
def prime5 := 11

-- The least positive whole number divisible by these five primes
def least_number := 2310

theorem least_divisible_by_first_five_primes :
  ∃ n, (n = prime1 * prime2 * prime3 * prime4 * prime5) ∧ (∀ m, m > 0 → (m % prime1 = 0 ∧ m % prime2 = 0 ∧ m % prime3 = 0 ∧ m % prime4 = 0 ∧ m % prime5 = 0 → m ≥ n)) :=
by {
  use least_number,
  sorry -- Proof needs to be filled in
}

end least_divisible_by_first_five_primes_l150_150506


namespace cone_volume_increase_l150_150522

theorem cone_volume_increase (r h : ℝ) (k : ℝ) :
  let V := (1/3) * π * r^2 * h
  let h' := 2.60 * h
  let r' := r * (1 + k / 100)
  let V' := (1/3) * π * (r')^2 * h'
  let percentage_increase := ((V' / V) - 1) * 100
  percentage_increase = ((1 + k / 100)^2 * 2.60 - 1) * 100 :=
by
  sorry

end cone_volume_increase_l150_150522


namespace cos_x_is_necessary_but_not_sufficient_for_sin_x_zero_l150_150049

-- Defining the conditions
def cos_x_eq_one (x : ℝ) : Prop := Real.cos x = 1
def sin_x_eq_zero (x : ℝ) : Prop := Real.sin x = 0

-- Main theorem statement
theorem cos_x_is_necessary_but_not_sufficient_for_sin_x_zero (x : ℝ) : 
  (∀ x, cos_x_eq_one x → sin_x_eq_zero x) ∧ (∃ x, sin_x_eq_zero x ∧ ¬ cos_x_eq_one x) :=
by 
  sorry

end cos_x_is_necessary_but_not_sufficient_for_sin_x_zero_l150_150049


namespace pow_mod_cycle_remainder_5_pow_2023_l150_150193

theorem pow_mod_cycle (n : ℕ) : (5^n % 6 = if n % 2 = 1 then 5 else 1) := 
by sorry

theorem remainder_5_pow_2023 : 5^2023 % 6 = 5 :=
by
  have cycle_properties : ∀ n, 5^n % 6 = (if n % 2 = 1 then 5 else 1) := pow_mod_cycle
  calc
    5^2023 % 6 = if 2023 % 2 = 1 then 5 else 1 := cycle_properties 2023
             ... = 5                   := by norm_num

end pow_mod_cycle_remainder_5_pow_2023_l150_150193


namespace intersection_M_N_l150_150737

-- Define set M
def M : Set ℝ := {x : ℝ | ∃ t : ℝ, x = 2^(-t) }

-- Define set N
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.sin x }

-- Theorem stating the intersection of M and N
theorem intersection_M_N :
  (M ∩ N) = {y : ℝ | 0 < y ∧ y ≤ 1} :=
by
  sorry

end intersection_M_N_l150_150737


namespace stratified_sampling_elderly_count_l150_150814

-- Definitions of conditions
def elderly := 30
def middleAged := 90
def young := 60
def totalPeople := elderly + middleAged + young
def sampleSize := 36
def samplingFraction := sampleSize / totalPeople
def expectedElderlySample := elderly * samplingFraction

-- The theorem we want to prove
theorem stratified_sampling_elderly_count : expectedElderlySample = 6 := 
by 
  -- Proof is omitted
  sorry

end stratified_sampling_elderly_count_l150_150814


namespace suff_and_nec_eq_triangle_l150_150957

noncomputable def triangle (A B C: ℝ) (a b c : ℝ) : Prop :=
(B + C = 2 * A) ∧ (b + c = 2 * a)

theorem suff_and_nec_eq_triangle (A B C a b c : ℝ) (h : triangle A B C a b c) :
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c :=
sorry

end suff_and_nec_eq_triangle_l150_150957


namespace value_after_addition_l150_150382

theorem value_after_addition (x : ℕ) (h : x / 9 = 8) : x + 11 = 83 :=
by
  sorry

end value_after_addition_l150_150382


namespace simplify_expression_l150_150912

variable (x : ℝ)

theorem simplify_expression : 1 - (2 - (3 - (4 - (5 - x)))) = 3 - x :=
by
  sorry

end simplify_expression_l150_150912


namespace find_m_n_sum_l150_150937

noncomputable def point (x : ℝ) (y : ℝ) := (x, y)

def center_line (P : ℝ × ℝ) : Prop := P.1 - P.2 - 2 = 0

def on_circle (C : ℝ × ℝ) (P : ℝ × ℝ) (r : ℝ) : Prop := 
  (P.1 - C.1)^2 + (P.2 - C.2)^2 = r^2

def circles_intersect (A B C D : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  on_circle A C r₁ ∧ on_circle A D r₂ ∧ on_circle B C r₁ ∧ on_circle B D r₂

theorem find_m_n_sum 
  (A : ℝ × ℝ) (m n : ℝ)
  (C D : ℝ × ℝ)
  (r₁ r₂ : ℝ)
  (H1 : A = point 1 3)
  (H2 : circles_intersect A (point m n) C D r₁ r₂)
  (H3 : center_line C ∧ center_line D) :
  m + n = 4 :=
sorry

end find_m_n_sum_l150_150937


namespace taller_tree_height_l150_150352

/-- The top of one tree is 20 feet higher than the top of another tree.
    The heights of the two trees are in the ratio 2:3.
    The shorter tree is 40 feet tall.
    Show that the height of the taller tree is 60 feet. -/
theorem taller_tree_height 
  (shorter_tree_height : ℕ) 
  (height_difference : ℕ)
  (height_ratio_num : ℕ)
  (height_ratio_denom : ℕ)
  (H1 : shorter_tree_height = 40)
  (H2 : height_difference = 20)
  (H3 : height_ratio_num = 2)
  (H4 : height_ratio_denom = 3)
  : ∃ taller_tree_height : ℕ, taller_tree_height = 60 :=
by
  sorry

end taller_tree_height_l150_150352


namespace matchsticks_distribution_l150_150890

open Nat

theorem matchsticks_distribution
  (length_sticks : ℕ)
  (width_sticks : ℕ)
  (length_condition : length_sticks = 60)
  (width_condition : width_sticks = 10)
  (total_sticks : ℕ)
  (total_sticks_condition : total_sticks = 60 * 11 + 10 * 61)
  (children_count : ℕ)
  (children_condition : children_count > 100)
  (division_condition : total_sticks % children_count = 0) :
  children_count = 127 := by
  sorry

end matchsticks_distribution_l150_150890


namespace dig_eq_conditions_l150_150840

theorem dig_eq_conditions (n k : ℕ) 
  (h1 : 10^(k-1) ≤ n^n ∧ n^n < 10^k)
  (h2 : 10^(n-1) ≤ k^k ∧ k^k < 10^n) :
  (n = 1 ∧ k = 1) ∨ (n = 8 ∧ k = 8) ∨ (n = 9 ∧ k = 9) :=
by
  sorry

end dig_eq_conditions_l150_150840


namespace total_bill_is_correct_l150_150909

-- Define conditions as constant values
def cost_per_scoop : ℕ := 2
def pierre_scoops : ℕ := 3
def mom_scoops : ℕ := 4

-- Define the total bill calculation
def total_bill := (pierre_scoops * cost_per_scoop) + (mom_scoops * cost_per_scoop)

-- State the theorem that the total bill equals 14
theorem total_bill_is_correct : total_bill = 14 := by
  sorry

end total_bill_is_correct_l150_150909


namespace a_81_eq_640_l150_150718

noncomputable def sequence_a (n : ℕ) : ℕ :=
if n = 0 then 0 -- auxiliary value since sequence begins from n=1
else if n = 1 then 1
else (2 * n - 1) ^ 2 - (2 * n - 3) ^ 2

theorem a_81_eq_640 : sequence_a 81 = 640 :=
by
  sorry

end a_81_eq_640_l150_150718


namespace polynomial_value_l150_150660

theorem polynomial_value : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 :=
by sorry

end polynomial_value_l150_150660


namespace delta_value_l150_150441

theorem delta_value (Delta : ℤ) (h : 5 * (-3) = Delta - 3) : Delta = -12 := 
by 
  sorry

end delta_value_l150_150441


namespace sequence_general_term_l150_150860

open Nat

def sequence_a (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  a 2 = 3 ∧
  (∀ n : ℕ, 0 < n → a (n + 2) ≤ a n + 3 * 2^n) ∧
  (∀ n : ℕ, 0 < n → a (n + 1) ≥ 2 * a n + 1)

theorem sequence_general_term (a : ℕ → ℕ) (h : sequence_a a) :
  ∀ n : ℕ, 0 < n → a n = 2^n - 1 :=
by
  sorry

end sequence_general_term_l150_150860


namespace geometric_series_sum_l150_150581

theorem geometric_series_sum (a : ℝ) (q : ℝ) (a₁ : ℝ) 
  (h1 : a₁ = 1)
  (h2 : q = a - (3/2))
  (h3 : |q| < 1)
  (h4 : a = a₁ / (1 - q)) :
  a = 2 :=
sorry

end geometric_series_sum_l150_150581


namespace arithmetic_geom_seq_l150_150034

noncomputable def geom_seq (a q : ℝ) : ℕ → ℝ 
| 0     => a
| (n+1) => q * (geom_seq a q n)

theorem arithmetic_geom_seq
  (a q : ℝ)
  (h_arith : 2 * geom_seq a q 1 = 1 + (geom_seq a q 2 - 1))
  (h_q : q = 2) :
  (geom_seq a q 2 + geom_seq a q 3) / (geom_seq a q 4 + geom_seq a q 5) = 1 / 4 :=
by
  sorry

end arithmetic_geom_seq_l150_150034


namespace at_least_50_singers_l150_150602

def youth_summer_village (total people_not_working people_with_families max_subset : ℕ) : Prop :=
  total = 100 ∧ 
  people_not_working = 50 ∧ 
  people_with_families = 25 ∧ 
  max_subset = 50

theorem at_least_50_singers (S : ℕ) (h : youth_summer_village 100 50 25 50) : S ≥ 50 :=
by
  obtain ⟨h1, h2, h3, h4⟩ := h
  sorry

end at_least_50_singers_l150_150602


namespace find_A_l150_150836

def spadesuit (A B : ℝ) : ℝ := 4 * A + 3 * B - 2

theorem find_A (A : ℝ) : spadesuit A 7 = 40 ↔ A = 21 / 4 :=
by
  sorry

end find_A_l150_150836


namespace range_of_a_l150_150730

noncomputable def f (x : ℝ) := -Real.exp x - x
noncomputable def g (a x : ℝ) := a * x + Real.cos x

theorem range_of_a :
  (∀ x : ℝ, ∃ y : ℝ, (g a y - g a y) / (y - y) * ((f x - f x) / (x - x)) = -1) →
  (0 ≤ a ∧ a ≤ 1) :=
by 
  sorry

end range_of_a_l150_150730


namespace percentage_seeds_germinated_l150_150844

theorem percentage_seeds_germinated :
  let S1 := 300
  let S2 := 200
  let S3 := 150
  let S4 := 250
  let S5 := 100
  let G1 := 0.20
  let G2 := 0.35
  let G3 := 0.45
  let G4 := 0.25
  let G5 := 0.60
  (G1 * S1 + G2 * S2 + G3 * S3 + G4 * S4 + G5 * S5) / (S1 + S2 + S3 + S4 + S5) * 100 = 32 := 
by
  sorry

end percentage_seeds_germinated_l150_150844


namespace option_a_is_odd_l150_150287

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem option_a_is_odd (a b : ℤ) (ha : is_odd a) (hb : is_odd b) : is_odd (a + 2 * b + 1) :=
by sorry

end option_a_is_odd_l150_150287


namespace grid_diagonal_segments_l150_150580

theorem grid_diagonal_segments (m n : ℕ) (hm : m = 100) (hn : n = 101) :
    let d := m + n - gcd m n
    d = 200 := by
  sorry

end grid_diagonal_segments_l150_150580


namespace initial_population_l150_150523

theorem initial_population (P : ℝ) : 
  (P * 1.2 * 0.8 = 9600) → P = 10000 :=
by
  sorry

end initial_population_l150_150523


namespace unit_digit_3_pow_2023_l150_150905

def unit_digit_pattern (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0

theorem unit_digit_3_pow_2023 : unit_digit_pattern 2023 = 7 :=
by sorry

end unit_digit_3_pow_2023_l150_150905


namespace min_value_expression_l150_150468

theorem min_value_expression (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_cond : x * y * z = 2 / 3) :
  x^2 + 6 * x * y + 18 * y^2 + 12 * y * z + 4 * z^2 ≥ 18 :=
by
  sorry

end min_value_expression_l150_150468


namespace annual_interest_rate_l150_150538

theorem annual_interest_rate (r : ℝ): 
  (1000 * r * 4.861111111111111 + 1400 * r * 4.861111111111111 = 350) → 
  r = 0.03 :=
sorry

end annual_interest_rate_l150_150538


namespace harris_carrot_expense_l150_150280

theorem harris_carrot_expense
  (carrots_per_day : ℕ)
  (days_per_year : ℕ)
  (carrots_per_bag : ℕ)
  (cost_per_bag : ℝ)
  (total_expense : ℝ) :
  carrots_per_day = 1 →
  days_per_year = 365 →
  carrots_per_bag = 5 →
  cost_per_bag = 2 →
  total_expense = 146 :=
by
  intros h1 h2 h3 h4
  sorry

end harris_carrot_expense_l150_150280


namespace cube_x_value_l150_150514

noncomputable def cube_side_len (x : ℝ) : ℝ := (8 * x) ^ (1 / 3)

lemma cube_volume (x : ℝ) : (cube_side_len x) ^ 3 = 8 * x :=
  by sorry

lemma cube_surface_area (x : ℝ) : 6 * (cube_side_len x) ^ 2 = 2 * x :=
  by sorry

theorem cube_x_value (x : ℝ) (hV : (cube_side_len x) ^ 3 = 8 * x) (hS : 6 * (cube_side_len x) ^ 2 = 2 * x) : x = sqrt 3 / 72 :=
  by sorry

end cube_x_value_l150_150514


namespace jack_cleaning_time_is_one_hour_l150_150749

def jackGrove : ℕ × ℕ := (4, 5)
def timeToCleanEachTree : ℕ := 6
def timeReductionFactor : ℕ := 2
def totalCleaningTimeWithHelpMin : ℕ :=
  (jackGrove.fst * jackGrove.snd) * (timeToCleanEachTree / timeReductionFactor)
def totalCleaningTimeWithHelpHours : ℕ :=
  totalCleaningTimeWithHelpMin / 60

theorem jack_cleaning_time_is_one_hour :
  totalCleaningTimeWithHelpHours = 1 := by
  sorry

end jack_cleaning_time_is_one_hour_l150_150749


namespace no_solution_exists_l150_150051

theorem no_solution_exists :
  ¬ ∃ (n : ℤ), 50 ≤ n ∧ n ≤ 150 ∧ n % 8 = 0 ∧ n % 10 = 6 ∧ n % 7 = 6 := 
by
  sorry

end no_solution_exists_l150_150051


namespace time_to_get_to_lawrence_house_l150_150318

def distance : ℝ := 12
def speed : ℝ := 2

theorem time_to_get_to_lawrence_house : (distance / speed) = 6 :=
by
  sorry

end time_to_get_to_lawrence_house_l150_150318


namespace evening_to_morning_ratio_l150_150323

-- Definitions based on conditions
def morning_miles : ℕ := 2
def total_miles : ℕ := 12
def evening_miles : ℕ := total_miles - morning_miles

-- Lean statement to prove the ratio
theorem evening_to_morning_ratio : evening_miles / morning_miles = 5 := by
  -- we simply state the final ratio we want to prove
  sorry

end evening_to_morning_ratio_l150_150323


namespace squirrel_rainy_days_l150_150647

theorem squirrel_rainy_days (s r : ℕ) (h1 : 20 * s + 12 * r = 112) (h2 : s + r = 8) : r = 6 :=
by {
  -- sorry to skip the proof
  sorry
}

end squirrel_rainy_days_l150_150647


namespace evaluate_expression_l150_150913

variable (a b : ℤ)

-- Define the main expression
def main_expression (a b : ℤ) : ℤ :=
  (a - b)^2 + (a + 3 * b) * (a - 3 * b) - a * (a - 2 * b)

theorem evaluate_expression : main_expression (-1) 2 = -31 := by
  -- substituting the value and solving it in the proof block
  sorry

end evaluate_expression_l150_150913


namespace expression_in_multiply_form_l150_150694

def a : ℕ := 3 ^ 1005
def b : ℕ := 7 ^ 1006
def m : ℕ := 114337548

theorem expression_in_multiply_form : 
  (a + b)^2 - (a - b)^2 = m * 10 ^ 1006 :=
by
  sorry

end expression_in_multiply_form_l150_150694


namespace largest_common_element_l150_150228

theorem largest_common_element (S1 S2 : ℕ → ℕ) (a_max : ℕ) :
  (∀ n, S1 n = 2 + 5 * n → ∃ k, S2 k = 3 + 8 * k ∧ S1 n = S2 k) →
  (147 < a_max) →
  ∀ m, (m < a_max → (∀ n, S1 n = 2 + 5 * n → ∃ k, S2 k = 3 + 8 * k ∧ S1 n = S2 k) → 147 = 27 + 40 * 3) :=
sorry

end largest_common_element_l150_150228


namespace polynomial_value_l150_150659

theorem polynomial_value : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 :=
by sorry

end polynomial_value_l150_150659


namespace exists_zero_in_interval_l150_150173
open Set Real Function

noncomputable def f (x : ℝ) := Real.exp x * Real.log 2 + x^2 - 6*x - 1

theorem exists_zero_in_interval : 
  ∃ c ∈ Ioo 3 4, f c = 0 :=
begin
  -- The proof goes here
  sorry
end

end exists_zero_in_interval_l150_150173


namespace area_of_triangle_condition_l150_150270

theorem area_of_triangle_condition (m : ℝ) (x y : ℝ) :
  (∀ (A B : ℝ × ℝ), (∀ x y, (x - m * y + 1 = 0 → (x - 1)^2 + y^2 = 4)) ∧ 
  (∃ A B : ℝ × ℝ, (x - m * y + 1 = 0 ∧ (x - 1)^2 + y^2 = 4) → (1 / 2) * 2 * 2 * sin (angle A (1, 0) B) = 8 / 5)) →
  m = 2 :=
begin
  sorry
end

end area_of_triangle_condition_l150_150270


namespace problem_l150_150758

def f (x : ℤ) := 3 * x + 2

theorem problem : f (f (f 3)) = 107 := by
  sorry

end problem_l150_150758


namespace smallest_n_Sn_pos_l150_150725

theorem smallest_n_Sn_pos {a : ℕ → ℤ} (S : ℕ → ℤ) 
  (h1 : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1))
  (h2 : ∀ n, (n ≠ 5 → S n > S 5))
  (h3 : |a 5| > |a 6|) :
  ∃ n : ℕ, S n > 0 ∧ ∀ m < n, S m ≤ 0 :=
by 
  -- Actual proof steps would go here.
  sorry

end smallest_n_Sn_pos_l150_150725


namespace remainder_product_mod_5_l150_150346

theorem remainder_product_mod_5 (a b c : ℕ) (h_a : a % 5 = 2) (h_b : b % 5 = 3) (h_c : c % 5 = 4) :
  (a * b * c) % 5 = 4 := 
by
  sorry

end remainder_product_mod_5_l150_150346


namespace systematic_sampling_missiles_l150_150482

theorem systematic_sampling_missiles (S : Set ℕ) (hS : S = {n | 1 ≤ n ∧ n ≤ 50}) :
  (∃ seq : Fin 5 → ℕ, (∀ i : Fin 4, seq (Fin.succ i) - seq i = 10) ∧ seq 0 = 3)
  → (∃ seq : Fin 5 → ℕ, seq = ![3, 13, 23, 33, 43]) :=
by
  sorry

end systematic_sampling_missiles_l150_150482


namespace tessa_owes_30_l150_150866

-- Definitions based on given conditions
def initial_debt : ℕ := 40
def paid_back : ℕ := initial_debt / 2
def remaining_debt_after_payment : ℕ := initial_debt - paid_back
def additional_borrowing : ℕ := 10
def total_debt : ℕ := remaining_debt_after_payment + additional_borrowing

-- Theorem to be proved
theorem tessa_owes_30 : total_debt = 30 :=
by
  sorry

end tessa_owes_30_l150_150866


namespace day_of_100th_day_of_2005_l150_150778

-- Define the days of the week
inductive Weekday
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
deriving DecidableEq, Repr

open Weekday

-- Define a function to add days to a given weekday
def add_days (d: Weekday) (n: ℕ) : Weekday :=
  match d with
  | Sunday => [Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday].get? (n % 7) |>.getD Sunday
  | Monday => [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday].get? (n % 7) |>.getD Monday
  | Tuesday => [Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday, Monday].get? (n % 7) |>.getD Tuesday
  | Wednesday => [Wednesday, Thursday, Friday, Saturday, Sunday, Monday, Tuesday].get? (n % 7) |>.getD Wednesday
  | Thursday => [Thursday, Friday, Saturday, Sunday, Monday, Tuesday, Wednesday].get? (n % 7) |>.getD Thursday
  | Friday => [Friday, Saturday, Sunday, Monday, Tuesday, Wednesday, Thursday].get? (n % 7) |>.getD Friday
  | Saturday => [Saturday, Sunday, Monday, Tuesday, Wednesday, Thursday, Friday].get? (n % 7) |>.getD Saturday

-- State the theorem
theorem day_of_100th_day_of_2005 :
  add_days Tuesday 55 = Monday :=
by sorry

end day_of_100th_day_of_2005_l150_150778


namespace supplementary_angle_ratio_l150_150360

theorem supplementary_angle_ratio (x : ℝ) (hx : 4 * x + x = 180) : x = 36 :=
by sorry

end supplementary_angle_ratio_l150_150360


namespace find_F_l150_150490

variable {R : Type*} [NontriviallyNormedField R]

noncomputable def F (x : R) : R := -Real.cos (Real.sin (Real.sin (Real.sin x)))

theorem find_F :
  ∃ (F : R → R)
    (h_diff : Differentiable ℝ F)
    (h_f0 : F 0 = -1)
    (h_deriv : ∀ x, deriv F x = Real.sin (Real.sin (Real.sin (Real.sin x))) * 
                                  Real.cos (Real.sin (Real.sin x)) * 
                                  Real.cos (Real.sin x) * 
                                  Real.cos x),
       F = -Real.cos (Real.sin (Real.sin (Real.sin x))) :=
begin
    use F,
    sorry
end

end find_F_l150_150490


namespace compare_neg_fractions_l150_150978

theorem compare_neg_fractions : - (3 / 5 : ℚ) < - (1 / 5 : ℚ) :=
by
  sorry

end compare_neg_fractions_l150_150978


namespace tim_sarah_age_ratio_l150_150706

theorem tim_sarah_age_ratio :
  ∀ (x : ℕ), ∃ (t s : ℕ),
    t = 23 ∧ s = 11 ∧
    (23 + x) * 2 = (11 + x) * 3 → x = 13 :=
by
  sorry

end tim_sarah_age_ratio_l150_150706


namespace find_two_digit_number_l150_150389

def digit_eq_square_of_units (n x : ℤ) : Prop :=
  10 * (x - 3) + x = n ∧ n = x * x

def units_digit_3_larger_than_tens (x : ℤ) : Prop :=
  x - 3 >= 1 ∧ x - 3 < 10 ∧ x >= 3 ∧ x < 10

theorem find_two_digit_number (n x : ℤ) (h1 : digit_eq_square_of_units n x)
  (h2 : units_digit_3_larger_than_tens x) : n = 25 ∨ n = 36 :=
by sorry

end find_two_digit_number_l150_150389


namespace smallest_y_for_square_l150_150699

theorem smallest_y_for_square (y M : ℕ) (h1 : 2310 * y = M^2) (h2 : 2310 = 2 * 3 * 5 * 7 * 11) : y = 2310 :=
by sorry

end smallest_y_for_square_l150_150699


namespace box_dimensions_l150_150833

theorem box_dimensions {a b c : ℕ} (h1 : a + c = 17) (h2 : a + b = 13) (h3 : 2 * (b + c) = 40) :
  a = 5 ∧ b = 8 ∧ c = 12 :=
by {
  sorry
}

end box_dimensions_l150_150833


namespace part1_l150_150848

variables (a c : ℝ × ℝ)
variables (a_parallel_c : ∃ k : ℝ, c = (k * a.1, k * a.2))
variables (a_value : a = (1,2))
variables (c_magnitude : (c.1 ^ 2 + c.2 ^ 2) = (3 * Real.sqrt 5) ^ 2)

theorem part1: c = (3, 6) ∨ c = (-3, -6) :=
by
  sorry

end part1_l150_150848


namespace length_of_bridge_l150_150821

noncomputable def speed_in_m_per_s (v_kmh : ℕ) : ℝ :=
  v_kmh * (1000 / 3600)

noncomputable def total_distance (v : ℝ) (t : ℝ) : ℝ :=
  v * t

theorem length_of_bridge (L_train : ℝ) (v_train_kmh : ℕ) (t : ℝ) (L_bridge : ℝ) :
  L_train = 288 →
  v_train_kmh = 29 →
  t = 48.29 →
  L_bridge = total_distance (speed_in_m_per_s v_train_kmh) t - L_train →
  L_bridge = 100.89 := by
  sorry

end length_of_bridge_l150_150821


namespace p_squared_plus_13_mod_n_eq_2_l150_150594

theorem p_squared_plus_13_mod_n_eq_2 (p : ℕ) (prime_p : Prime p) (h : p > 3) (n : ℕ) :
  (∃ (k : ℕ), p ^ 2 + 13 = k * n + 2) → n = 2 :=
by
  sorry

end p_squared_plus_13_mod_n_eq_2_l150_150594


namespace value_in_parentheses_l150_150745

theorem value_in_parentheses (x : ℝ) (h : x / Real.sqrt 18 = Real.sqrt 2) : x = 6 :=
sorry

end value_in_parentheses_l150_150745


namespace b_spends_85_percent_l150_150208

-- Definitions based on the given conditions
def combined_salary (a_salary b_salary : ℤ) : Prop := a_salary + b_salary = 3000
def a_salary : ℤ := 2250
def a_spending_ratio : ℝ := 0.95
def a_savings : ℝ := a_salary - a_salary * a_spending_ratio
def b_savings : ℝ := a_savings

-- The goal is to prove that B spends 85% of his salary
theorem b_spends_85_percent (b_salary : ℤ) (b_spending_ratio : ℝ) :
  combined_salary a_salary b_salary →
  b_spending_ratio * b_salary = 0.85 * b_salary :=
  sorry

end b_spends_85_percent_l150_150208


namespace johns_total_amount_l150_150610

def amount_from_grandpa : ℕ := 30
def multiplier : ℕ := 3
def amount_from_grandma : ℕ := amount_from_grandpa * multiplier
def total_amount : ℕ := amount_from_grandpa + amount_from_grandma

theorem johns_total_amount :
  total_amount = 120 :=
by
  sorry

end johns_total_amount_l150_150610


namespace unique_function_satisfying_conditions_l150_150245

noncomputable def f : (ℝ → ℝ) := sorry

axiom condition1 : f 1 = 1
axiom condition2 : ∀ x y : ℝ, f (x * y + f x) = x * f y + f x

theorem unique_function_satisfying_conditions : ∀ x : ℝ, f x = x := sorry

end unique_function_satisfying_conditions_l150_150245


namespace total_marks_l150_150224

variable (marks_in_music marks_in_maths marks_in_arts marks_in_social_studies : ℕ)

def marks_conditions : Prop :=
  marks_in_maths = marks_in_music - (1/10) * marks_in_music ∧
  marks_in_maths = marks_in_arts - 20 ∧
  marks_in_social_studies = marks_in_music + 10 ∧
  marks_in_music = 70

theorem total_marks 
  (h : marks_conditions marks_in_music marks_in_maths marks_in_arts marks_in_social_studies) :
  marks_in_music + marks_in_maths + marks_in_arts + marks_in_social_studies = 296 :=
by
  sorry

end total_marks_l150_150224


namespace parcel_cost_guangzhou_shanghai_l150_150209

theorem parcel_cost_guangzhou_shanghai (x y : ℕ) :
  (x + 2 * y = 10 ∧ x + 3 * (y + 3) + 2 = 23) →
  (x = 6 ∧ y = 2 ∧ (6 + 4 * 2 = 14)) := by
  sorry

end parcel_cost_guangzhou_shanghai_l150_150209


namespace cos_double_angle_l150_150422

theorem cos_double_angle {α : ℝ} (h1 : 0 < α ∧ α < 2 * Real.pi ∧ α > 3 * Real.pi / 2) 
  (h2 : Real.sin α + Real.cos α = Real.sqrt 3 / 3) : 
  Real.cos (2 * α) = Real.sqrt 5 / 3 := 
by
  sorry

end cos_double_angle_l150_150422


namespace pencil_distribution_l150_150206

theorem pencil_distribution (C C' : ℕ) (pencils : ℕ) (remaining : ℕ) (less_per_class : ℕ) 
  (original_classes : C = 4) 
  (total_pencils : pencils = 172) 
  (remaining_pencils : remaining = 7) 
  (less_pencils : less_per_class = 28)
  (actual_classes : C' > C) 
  (distribution_mistake : (pencils - remaining) / C' + less_per_class = pencils / C) :
  C' = 11 := 
sorry

end pencil_distribution_l150_150206


namespace student_test_ratio_l150_150541

theorem student_test_ratio :
  ∀ (total_questions correct_responses : ℕ),
  total_questions = 100 →
  correct_responses = 93 →
  (total_questions - correct_responses) / correct_responses = 7 / 93 :=
by
  intros total_questions correct_responses h_total_questions h_correct_responses
  sorry

end student_test_ratio_l150_150541


namespace exists_m_area_triangle_ABC_l150_150267

theorem exists_m_area_triangle_ABC :
  ∃ m : ℝ, 
    m = 2 ∧ 
    (∃ A B : ℝ × ℝ, 
      ∃ C : ℝ × ℝ, 
        C = (1, 0) ∧ 
        (A ≠ B) ∧
        ((A.fst - 1)^2 + A.snd^2 = 4) ∧
        ((B.fst - 1)^2 + B.snd^2 = 4) ∧
        ((A.fst - m * A.snd + 1 = 0) ∧ 
         (B.fst - m * B.snd + 1 = 0)) ∧ 
        (1 / 2 * 2 * 2 * Real.sin (angle A C B) = 8 / 5)) :=
sorry

end exists_m_area_triangle_ABC_l150_150267


namespace problem_statement_l150_150715

theorem problem_statement (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end problem_statement_l150_150715


namespace find_f_half_l150_150713

theorem find_f_half (f : ℝ → ℝ) (h : ∀ x, f (2 * x / (x + 1)) = x^2 - 1) : f (1 / 2) = -8 / 9 :=
by
  sorry

end find_f_half_l150_150713


namespace exists_root_between_1_1_and_1_2_l150_150547

def f (x : ℝ) : ℝ := x^2 + 12 * x - 15

theorem exists_root_between_1_1_and_1_2 :
  ∃ x, 1.1 < x ∧ x < 1.2 ∧ f x = 0 :=
by
  have pf1 : f 1.1 = -0.59 := by norm_num1
  have pf2 : f 1.2 = 0.84 := by norm_num1
  apply exists_between_of_sign_change pf1 pf2
  sorry

end exists_root_between_1_1_and_1_2_l150_150547


namespace negation_of_universal_proposition_l150_150736

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 1) ↔ (∃ x : ℝ, x^2 ≤ 1) :=
by
  sorry

end negation_of_universal_proposition_l150_150736


namespace fixed_point_l150_150492

theorem fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) : (1, 4) ∈ {p : ℝ × ℝ | ∃ x, p = (x, a^(x-1) + 3)} :=
by
  sorry

end fixed_point_l150_150492


namespace find_m_l150_150271

open Real

def circle_center : Point := (1, 0)
def radius : ℝ := 2

def line (m : ℝ) : set Point := {p | p.1 - m * p.2 + 1 = 0}

def circle : set Point := {p | (p.1 - 1)^2 + p.2^2 = radius^2}

def area_ABC (A B C : Point) : ℝ :=
  (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem find_m (m : ℝ) (A B : Point) (hA : A ∈ line m) (hB : B ∈ line m)
  (hA_circle : A ∈ circle) (hB_circle : B ∈ circle) :
  (A = (1 - sqrt 5 / 2,  sqrt 5 / 2 ∨ (1 + sqrt 5 / 2, -sqrt 5 / 2))
  (B = (1 + sqrt 5 / 2, sqrt 5 / 2) ∨ (1 - sqrt 5 / 2,  -sqrt 5 / 2))  →
  area_ABC A B circle_center = 8 / 5 →
  (m = 2 ∨ m = -2 ∨ m = 1/2 ∨ m = -1/2) :=
sorry

end find_m_l150_150271


namespace min_sticks_12_to_break_can_form_square_15_l150_150063

-- Problem definition for n = 12
def sticks_12 : List Nat := List.range' 1 12

theorem min_sticks_12_to_break : 
  ... (I realize I need to translate a step better) ..............
  sorry

-- Problem definition for n = 15
def sticks_15 : List Nat := List.range' 1 15

theorem can_form_square_15 : 
  ... (implementing a nice explanation)
  sorry

end min_sticks_12_to_break_can_form_square_15_l150_150063


namespace maximum_value_of_func_l150_150128

noncomputable def func (x : ℝ) : ℝ := 4 * x - 2 + 1 / (4 * x - 5)

theorem maximum_value_of_func (x : ℝ) (h : x < 5 / 4) : ∃ y, y = 1 ∧ ∀ z, z = func x → z ≤ y :=
sorry

end maximum_value_of_func_l150_150128


namespace probability_tile_from_ANGLE_l150_150558

def letters_in_ALGEBRA : List Char := ['A', 'L', 'G', 'E', 'B', 'R', 'A']
def letters_in_ANGLE : List Char := ['A', 'N', 'G', 'L', 'E']
def count_matching_letters (letters: List Char) (target: List Char) : Nat :=
  letters.foldr (fun l acc => if l ∈ target then acc + 1 else acc) 0

theorem probability_tile_from_ANGLE :
  (count_matching_letters letters_in_ALGEBRA letters_in_ANGLE : ℚ) / (letters_in_ALGEBRA.length : ℚ) = 5 / 7 :=
by
  sorry

end probability_tile_from_ANGLE_l150_150558


namespace tens_digit_less_than_5_probability_l150_150535

theorem tens_digit_less_than_5_probability 
  (n : ℕ) 
  (hn : 10000 ≤ n ∧ n ≤ 99999)
  (h_even : ∃ k, n % 10 = 2 * k ∧ k < 5) :
  (∃ p, 0 ≤ p ∧ p ≤ 1 ∧ p = 1 / 2) :=
by
  sorry

end tens_digit_less_than_5_probability_l150_150535


namespace inverse_of_matrix_C_l150_150851

-- Define the given matrix C
def C : Matrix (Fin 3) (Fin 3) ℚ := ![
  ![1, 2, 1],
  ![3, -5, 3],
  ![2, 7, -1]
]

-- Define the claimed inverse of the matrix C
def C_inv : Matrix (Fin 3) (Fin 3) ℚ := (1 / 33 : ℚ) • ![
  ![-16,  9,  11],
  ![  9, -3,   0],
  ![ 31, -3, -11]
]

-- Statement to prove that C_inv is the inverse of C
theorem inverse_of_matrix_C : C * C_inv = 1 ∧ C_inv * C = 1 := by
  sorry

end inverse_of_matrix_C_l150_150851


namespace additional_hours_to_travel_l150_150970

theorem additional_hours_to_travel (distance1 time1 rate distance2 : ℝ)
  (H1 : distance1 = 360)
  (H2 : time1 = 3)
  (H3 : rate = distance1 / time1)
  (H4 : distance2 = 240)
  :
  distance2 / rate = 2 := 
sorry

end additional_hours_to_travel_l150_150970


namespace compute_n_binom_l150_150579

-- Definitions based on conditions
def n : ℕ := sorry  -- Assume n is a positive integer defined elsewhere
def k : ℕ := 4

-- The binomial coefficient definition
def binom (n k : ℕ) : ℕ :=
  if h₁ : k ≤ n then
    (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))
  else 0

-- The theorem to prove
theorem compute_n_binom : n * binom k 3 = 4 * n :=
by
  sorry

end compute_n_binom_l150_150579


namespace two_buttons_diff_size_color_l150_150557

variables (box : Type) 
variable [Finite box]
variables (Big Small White Black : box → Prop)

axiom big_ex : ∃ x, Big x
axiom small_ex : ∃ x, Small x
axiom white_ex : ∃ x, White x
axiom black_ex : ∃ x, Black x
axiom size : ∀ x, Big x ∨ Small x
axiom color : ∀ x, White x ∨ Black x

theorem two_buttons_diff_size_color : 
  ∃ x y, x ≠ y ∧ (Big x ∧ Small y ∨ Small x ∧ Big y) ∧ (White x ∧ Black y ∨ Black x ∧ White y) := 
by
  sorry

end two_buttons_diff_size_color_l150_150557


namespace find_general_students_l150_150934

-- Define the conditions and the question
structure Halls :=
  (general : ℕ)
  (biology : ℕ)
  (math : ℕ)
  (total : ℕ)

def conditions_met (h : Halls) : Prop :=
  h.biology = 2 * h.general ∧
  h.math = (3 / 5 : ℚ) * (h.general + h.biology) ∧
  h.total = h.general + h.biology + h.math ∧
  h.total = 144

-- The proof problem statement
theorem find_general_students (h : Halls) (h_cond : conditions_met h) : h.general = 30 :=
sorry

end find_general_students_l150_150934


namespace major_axis_length_l150_150383

theorem major_axis_length {r : ℝ} (h_r : r = 1) (h_major : ∃ (minor_axis : ℝ), minor_axis = 2 * r ∧ 1.5 * minor_axis = major_axis) : major_axis = 3 :=
by
  sorry

end major_axis_length_l150_150383


namespace value_of_x_l150_150135

theorem value_of_x : (∃ x : ℝ, (1 / 8) * 2 ^ 36 = 8 ^ x) → x = 11 := by
  intro h
  rcases h with ⟨x, hx⟩
  have h1 : 1 / 8 = 2 ^ (-3) := by norm_num
  rw [h1, ←pow_add] at hx
  norm_num at hx
  have h2 : 8 = 2 ^ 3 := by norm_num
  rw [h2, pow_mul] at hx
  norm_num at hx
  exact hx.symm

end value_of_x_l150_150135


namespace daisy_count_per_bouquet_l150_150816

-- Define the conditions
def roses_per_bouquet := 12
def total_bouquets := 20
def rose_bouquets := 10
def daisy_bouquets := total_bouquets - rose_bouquets
def total_flowers_sold := 190
def total_roses_sold := rose_bouquets * roses_per_bouquet
def total_daisies_sold := total_flowers_sold - total_roses_sold

-- Define the problem: prove that the number of daisies per bouquet is 7
theorem daisy_count_per_bouquet : total_daisies_sold / daisy_bouquets = 7 := by
  sorry

end daisy_count_per_bouquet_l150_150816


namespace no_function_satisfies_condition_l150_150700

theorem no_function_satisfies_condition :
  ¬ ∃ (f: ℕ → ℕ), ∀ (n: ℕ), f (f n) = n + 2017 :=
by
  -- Proof details are omitted
  sorry

end no_function_satisfies_condition_l150_150700


namespace average_length_of_strings_l150_150633

theorem average_length_of_strings {l1 l2 l3 : ℝ} (h1 : l1 = 2) (h2 : l2 = 6) (h3 : l3 = 9) : 
  (l1 + l2 + l3) / 3 = 17 / 3 :=
by
  sorry

end average_length_of_strings_l150_150633


namespace population_increase_l150_150711

theorem population_increase (i j : ℝ) : 
  ∀ (m : ℝ), m * (1 + i / 100) * (1 + j / 100) = m * (1 + (i + j + i * j / 100) / 100) := 
by
  intro m
  sorry

end population_increase_l150_150711


namespace complete_square_add_term_l150_150390

theorem complete_square_add_term (x : ℝ) :
  ∃ (c : ℝ), (c = 4 * x ^ 4 ∨ c = 4 * x ∨ c = -4 * x ∨ c = -1 ∨ c = -4 * x ^2) ∧
  (4 * x ^ 2 + 1 + c) * (4 * x ^ 2 + 1 + c) = (2 * x + 1) * (2 * x + 1) :=
sorry

end complete_square_add_term_l150_150390


namespace least_positive_divisible_l150_150507

/-- The first five different prime numbers are given as conditions: -/
def prime1 := 2
def prime2 := 3
def prime3 := 5
def prime4 := 7
def prime5 := 11

/-- The least positive whole number divisible by the first five primes is 2310. -/
theorem least_positive_divisible :
  ∃ n : ℕ, n > 0 ∧ (n % prime1 = 0) ∧ (n % prime2 = 0) ∧ (n % prime3 = 0) ∧ (n % prime4 = 0) ∧ (n % prime5 = 0) ∧ n = 2310 :=
sorry

end least_positive_divisible_l150_150507


namespace train_distance_900_l150_150369

theorem train_distance_900 (x t : ℝ) (H1 : x = 50 * t) (H2 : x - 100 = 40 * t) : 
  x + (x - 100) = 900 :=
by
  sorry

end train_distance_900_l150_150369


namespace one_fourth_of_7point2_is_9div5_l150_150241

theorem one_fourth_of_7point2_is_9div5 : (7.2 / 4 : ℚ) = 9 / 5 := 
by sorry

end one_fourth_of_7point2_is_9div5_l150_150241


namespace num_of_poly_sci_majors_l150_150908

-- Define the total number of applicants
def total_applicants : ℕ := 40

-- Define the number of applicants with GPA > 3.0
def gpa_higher_than_3_point_0 : ℕ := 20

-- Define the number of applicants who did not major in political science and had GPA ≤ 3.0
def non_poly_sci_and_low_gpa : ℕ := 10

-- Define the number of political science majors with GPA > 3.0
def poly_sci_with_high_gpa : ℕ := 5

-- Prove the number of political science majors
theorem num_of_poly_sci_majors : ∀ (P : ℕ),
  P = poly_sci_with_high_gpa + 
      (total_applicants - non_poly_sci_and_low_gpa - 
       (gpa_higher_than_3_point_0 - poly_sci_with_high_gpa)) → 
  P = 20 :=
by
  intros P h
  sorry

end num_of_poly_sci_majors_l150_150908


namespace minimum_sum_of_dimensions_of_box_l150_150915

theorem minimum_sum_of_dimensions_of_box (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_vol : a * b * c = 2310) :
  a + b + c ≥ 52 :=
sorry

end minimum_sum_of_dimensions_of_box_l150_150915


namespace consecutive_negative_product_sum_l150_150182

theorem consecutive_negative_product_sum (n : ℤ) (h : n * (n + 1) = 2850) : n + (n + 1) = -107 :=
sorry

end consecutive_negative_product_sum_l150_150182


namespace arithmetic_mean_is_one_l150_150233

theorem arithmetic_mean_is_one (x a : ℝ) (hx : x ≠ 0) (hx2a : x^2 ≠ a) :
  (1 / 2 * ((x^2 + a) / x^2 + (x^2 - a) / x^2) = 1) :=
by
  sorry

end arithmetic_mean_is_one_l150_150233


namespace probability_of_three_even_numbers_l150_150688

theorem probability_of_three_even_numbers (n : ℕ) (k : ℕ) (p_even : ℚ) (p_odd : ℚ) (comb : ℕ → ℕ → ℕ) 
    (h_n : n = 5) (h_k : k = 3) (h_p_even : p_even = 1/2) (h_p_odd : p_odd = 1/2) 
    (h_comb : comb 5 3 = 10) :
    comb n k * (p_even ^ k) * (p_odd ^ (n - k)) = 5 / 16 :=
by sorry

end probability_of_three_even_numbers_l150_150688


namespace rickshaw_distance_l150_150087

theorem rickshaw_distance :
  ∃ (distance : ℝ), 
  (13.5 + (distance - 1) * (2.50 / (1 / 3))) = 103.5 ∧ distance = 13 :=
by
  sorry

end rickshaw_distance_l150_150087


namespace seokjin_higher_than_jungkook_l150_150463

variable (Jungkook_yoojeong_seokjin_stairs : ℕ)

def jungkook_stair := 19
def yoojeong_stair := jungkook_stair + 8
def seokjin_stair := yoojeong_stair - 5

theorem seokjin_higher_than_jungkook : seokjin_stair - jungkook_stair = 3 :=
by sorry

end seokjin_higher_than_jungkook_l150_150463


namespace prob_A_is_15_16_prob_B_is_3_4_prob_C_is_5_9_prob_exactly_two_good_ratings_is_77_576_l150_150478

-- Define the probability of success for student A, B, and C on a single jump
def p_A1 := 3 / 4
def p_B1 := 1 / 2
def p_C1 := 1 / 3

-- Calculate the total probability of excellence for A, B, and C
def P_A := p_A1 + (1 - p_A1) * p_A1
def P_B := p_B1 + (1 - p_B1) * p_B1
def P_C := p_C1 + (1 - p_C1) * p_C1

-- Statement to prove probabilities
theorem prob_A_is_15_16 : P_A = 15 / 16 := sorry
theorem prob_B_is_3_4 : P_B = 3 / 4 := sorry
theorem prob_C_is_5_9 : P_C = 5 / 9 := sorry

-- Definition for P(Good_Ratings) - exactly two students get a good rating
def P_Good_Ratings := 
  P_A * (1 - P_B) * (1 - P_C) + 
  (1 - P_A) * P_B * (1 - P_C) + 
  (1 - P_A) * (1 - P_B) * P_C

-- Statement to prove the given condition about good ratings
theorem prob_exactly_two_good_ratings_is_77_576 : P_Good_Ratings = 77 / 576 := sorry

end prob_A_is_15_16_prob_B_is_3_4_prob_C_is_5_9_prob_exactly_two_good_ratings_is_77_576_l150_150478


namespace recommended_water_intake_l150_150935

theorem recommended_water_intake (current_intake : ℕ) (increase_percentage : ℚ) (recommended_intake : ℕ) : 
  current_intake = 15 → increase_percentage = 0.40 → recommended_intake = 21 :=
by
  intros h1 h2
  sorry

end recommended_water_intake_l150_150935


namespace region_area_correct_l150_150567

noncomputable def region_area : ℝ :=
  let region := {p : ℝ × ℝ | |p.1 + p.2| + |p.1 - p.2| ≤ 6}
  let area := (3 - -3) * (3 - -3)
  area

theorem region_area_correct : region_area = 36 :=
by sorry

end region_area_correct_l150_150567


namespace overall_percentage_change_is_113_point_4_l150_150092

-- Define the conditions
def total_customers_survey_1 := 100
def male_percentage_survey_1 := 60
def respondents_survey_1 := 10
def male_respondents_survey_1 := 5

def total_customers_survey_2 := 80
def male_percentage_survey_2 := 70
def respondents_survey_2 := 16
def male_respondents_survey_2 := 12

def total_customers_survey_3 := 70
def male_percentage_survey_3 := 40
def respondents_survey_3 := 21
def male_respondents_survey_3 := 13

def total_customers_survey_4 := 90
def male_percentage_survey_4 := 50
def respondents_survey_4 := 27
def male_respondents_survey_4 := 8

-- Define the calculated response rates
def original_male_response_rate := (male_respondents_survey_1.toFloat / (total_customers_survey_1 * male_percentage_survey_1 / 100).toFloat) * 100
def final_male_response_rate := (male_respondents_survey_4.toFloat / (total_customers_survey_4 * male_percentage_survey_4 / 100).toFloat) * 100

-- Calculate the percentage change in response rate
def percentage_change := ((final_male_response_rate - original_male_response_rate) / original_male_response_rate) * 100

-- The target theorem 
theorem overall_percentage_change_is_113_point_4 : percentage_change = 113.4 := sorry

end overall_percentage_change_is_113_point_4_l150_150092


namespace people_who_came_to_game_l150_150793

def total_seats : Nat := 92
def people_with_banners : Nat := 38
def empty_seats : Nat := 45

theorem people_who_came_to_game : (total_seats - empty_seats = 47) :=
by 
  sorry

end people_who_came_to_game_l150_150793


namespace number_of_solutions_l150_150587

theorem number_of_solutions :
  ∃ (solutions : Finset (ℝ × ℝ)), 
  (∀ (x y : ℝ), (x, y) ∈ solutions ↔ (x + 2 * y = 2 ∧ abs (abs x - 2 * abs y) = 1)) ∧ 
  solutions.card = 2 :=
by
  sorry

end number_of_solutions_l150_150587


namespace george_choices_l150_150298

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- State the theorem to prove the number of ways to choose 3 out of 9 colors is 84
theorem george_choices : binomial 9 3 = 84 := by
  sorry

end george_choices_l150_150298


namespace lost_card_number_l150_150320

theorem lost_card_number (p : ℕ) (c : ℕ) (h : 0 ≤ c ∧ c ≤ 9)
  (sum_remaining_cards : 10 * p + 45 - (p + c) = 2012) : p + c = 223 := by
  sorry

end lost_card_number_l150_150320


namespace jackie_eligible_for_free_shipping_l150_150144

def shampoo_cost : ℝ := 2 * 12.50
def conditioner_cost : ℝ := 3 * 15.00
def face_cream_cost : ℝ := 20.00  -- Considering the buy-one-get-one-free deal

def subtotal : ℝ := shampoo_cost + conditioner_cost + face_cream_cost
def discount : ℝ := 0.10 * subtotal
def total_after_discount : ℝ := subtotal - discount

theorem jackie_eligible_for_free_shipping : total_after_discount >= 75 := by
  sorry

end jackie_eligible_for_free_shipping_l150_150144


namespace rounding_estimation_correct_l150_150598

theorem rounding_estimation_correct (a b d : ℕ)
  (ha : a > 0) (hb : b > 0) (hd : d > 0)
  (a_round : ℕ) (b_round : ℕ) (d_round : ℕ)
  (h_round_a : a_round ≥ a) (h_round_b : b_round ≤ b) (h_round_d : d_round ≤ d) :
  (Real.sqrt (a_round / b_round) - Real.sqrt d_round) > (Real.sqrt (a / b) - Real.sqrt d) :=
by
  sorry

end rounding_estimation_correct_l150_150598


namespace cube_volume_surface_area_l150_150512

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ s : ℝ, s^3 = 8 * x ∧ 6 * s^2 = 2 * x) → x = 0 :=
by
  sorry

end cube_volume_surface_area_l150_150512


namespace george_choices_l150_150299

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- State the theorem to prove the number of ways to choose 3 out of 9 colors is 84
theorem george_choices : binomial 9 3 = 84 := by
  sorry

end george_choices_l150_150299


namespace g_five_eq_one_l150_150235

variable (g : ℝ → ℝ)
variable (h : ∀ x y : ℝ, g (x - y) = g x * g y)
variable (h_ne_zero : ∀ x : ℝ, g x ≠ 0)

theorem g_five_eq_one : g 5 = 1 :=
by
  sorry

end g_five_eq_one_l150_150235


namespace num_people_present_l150_150520

-- Given conditions
def associatePencilCount (A : ℕ) : ℕ := 2 * A
def assistantPencilCount (B : ℕ) : ℕ := B
def associateChartCount (A : ℕ) : ℕ := A
def assistantChartCount (B : ℕ) : ℕ := 2 * B

def totalPencils (A B : ℕ) : ℕ := associatePencilCount A + assistantPencilCount B
def totalCharts (A B : ℕ) : ℕ := associateChartCount A + assistantChartCount B

-- Prove the total number of people present
theorem num_people_present (A B : ℕ) (h1 : totalPencils A B = 11) (h2 : totalCharts A B = 16) : A + B = 9 :=
by
  sorry

end num_people_present_l150_150520


namespace distinct_x_sum_l150_150313

theorem distinct_x_sum (x y z : ℂ) 
(h1 : x + y * z = 9) 
(h2 : y + x * z = 12) 
(h3 : z + x * y = 12) : 
(x = 1 ∨ x = 3) ∧ (¬(x = 1 ∧ x = 3) → x ≠ 1 ∧ x ≠ 3) ∧ (1 + 3 = 4) :=
by
  sorry

end distinct_x_sum_l150_150313


namespace sum_of_distinct_integers_l150_150148

theorem sum_of_distinct_integers 
  (a b c d e : ℤ)
  (h1 : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 60)
  (h2 : (7 - a) ≠ (7 - b) ∧ (7 - a) ≠ (7 - c) ∧ (7 - a) ≠ (7 - d) ∧ (7 - a) ≠ (7 - e))
  (h3 : (7 - b) ≠ (7 - c) ∧ (7 - b) ≠ (7 - d) ∧ (7 - b) ≠ (7 - e))
  (h4 : (7 - c) ≠ (7 - d) ∧ (7 - c) ≠ (7 - e))
  (h5 : (7 - d) ≠ (7 - e)) : 
  a + b + c + d + e = 24 := 
sorry

end sum_of_distinct_integers_l150_150148


namespace range_of_a_l150_150272

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧ 
  (∀ x, |x^3 - a * x^2| = x → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) →
  a > 2 :=
by
  -- The proof is to be provided here.
  sorry

end range_of_a_l150_150272


namespace problem_equiv_l150_150845

def dollar (a b : ℝ) : ℝ := (a - b) ^ 2

theorem problem_equiv (x y : ℝ) : dollar ((2 * x + y) ^ 2) ((x - 2 * y) ^ 2) = (3 * x ^ 2 + 8 * x * y - 3 * y ^ 2) ^ 2 := by
  sorry

end problem_equiv_l150_150845


namespace find_quadruple_l150_150015

/-- Problem Statement:
Given distinct positive integers a, b, c, and d such that a + b = c * d and a * b = c + d,
find the quadruple (a, b, c, d) that meets these conditions.
-/

theorem find_quadruple :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
            0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
            (a + b = c * d) ∧ (a * b = c + d) ∧
            ((a, b, c, d) = (1, 5, 3, 2) ∨ (a, b, c, d) = (1, 5, 2, 3) ∨
             (a, b, c, d) = (5, 1, 3, 2) ∨ (a, b, c, d) = (5, 1, 2, 3) ∨
             (a, b, c, d) = (2, 3, 1, 5) ∨ (a, b, c, d) = (3, 2, 1, 5) ∨
             (a, b, c, d) = (2, 3, 5, 1) ∨ (a, b, c, d) = (3, 2, 5, 1)) :=
sorry

end find_quadruple_l150_150015


namespace best_value_l150_150681

variables {cS qS cM qL cL : ℝ}
variables (medium_cost : cM = 1.4 * cS) (medium_quantity : qM = 0.7 * qL)
variables (large_quantity : qL = 1.5 * qS) (large_cost : cL = 1.2 * cM)

theorem best_value :
  let small_value := cS / qS
  let medium_value := cM / (0.7 * qL)
  let large_value := cL / qL
  small_value < large_value ∧ large_value < medium_value :=
sorry

end best_value_l150_150681


namespace log_base_half_iff_l150_150122

theorem log_base_half_iff (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (log (1/2) a > log (1/2) b) ↔ (a < b) :=
sorry

end log_base_half_iff_l150_150122


namespace highest_score_l150_150642

theorem highest_score (H L : ℕ) (avg total46 total44 runs46 runs44 : ℕ)
  (h1 : H - L = 150)
  (h2 : avg = 61)
  (h3 : total46 = 46)
  (h4 : runs46 = avg * total46)
  (h5 : runs46 = 2806)
  (h6 : total44 = 44)
  (h7 : runs44 = 58 * total44)
  (h8 : runs44 = 2552)
  (h9 : runs46 - runs44 = H + L) :
  H = 202 := by
  sorry

end highest_score_l150_150642


namespace gcd_polynomials_l150_150999

theorem gcd_polynomials (b : ℤ) (h: ∃ k : ℤ, b = 2 * k * 953) :
  Int.gcd (3 * b^2 + 17 * b + 23) (b + 19) = 34 :=
sorry

end gcd_polynomials_l150_150999


namespace perfect_square_trinomial_l150_150017

-- Define the conditions
theorem perfect_square_trinomial (k : ℤ) : 
  ∃ (a b : ℤ), (a^2 = 1 ∧ b^2 = 16 ∧ (x^2 + k * x * y + 16 * y^2 = (a * x + b * y)^2)) ↔ (k = 8 ∨ k = -8) :=
by
  sorry

end perfect_square_trinomial_l150_150017


namespace swans_after_10_years_l150_150112

-- Defining the initial conditions
def initial_swans : ℕ := 15

-- Condition that the number of swans doubles every 2 years
def double_every_two_years (n t : ℕ) : ℕ := n * (2 ^ (t / 2))

-- Prove that after 10 years, the number of swans will be 480
theorem swans_after_10_years : double_every_two_years initial_swans 10 = 480 :=
by
  sorry

end swans_after_10_years_l150_150112


namespace max_area_of_triangle_l150_150835

theorem max_area_of_triangle :
  ∀ (O O' : EuclideanSpace ℝ (Fin 2)) (M : EuclideanSpace ℝ (Fin 2)),
  dist O O' = 2014 →
  dist O M = 1 ∨ dist O' M = 1 →
  ∃ (A : ℝ), A = 1007 :=
by
  intros O O' M h₁ h₂
  sorry

end max_area_of_triangle_l150_150835


namespace not_possible_to_fill_6x6_with_1x4_l150_150892

theorem not_possible_to_fill_6x6_with_1x4 :
  ¬ (∃ (a b : ℕ), a + 4 * b = 6 ∧ 4 * a + b = 6) :=
by
  -- Assuming a and b represent the number of 1x4 rectangles aligned horizontally and vertically respectively
  sorry

end not_possible_to_fill_6x6_with_1x4_l150_150892


namespace abc_equality_l150_150169

noncomputable def abc_value (a b c : ℝ) : ℝ := (11 + Real.sqrt 117) / 2

theorem abc_equality (a b c : ℝ) (h1 : a + 1/b = 5) (h2 : b + 1/c = 2) (h3 : (c + 1/a)^2 = 4) :
  a * b * c = abc_value a b c := 
sorry

end abc_equality_l150_150169


namespace find_K_l150_150256

def satisfies_conditions (K m n h : ℕ) : Prop :=
  K ∣ (m^h - 1) ∧ K ∣ (n ^ ((m^h - 1) / K) + 1)

def odd (n : ℕ) : Prop := n % 2 = 1

theorem find_K (r : ℕ) (h : ℕ := 2^r) :
    ∀ K : ℕ, (∃ (m : ℕ), odd m ∧ m > 1 ∧ ∃ (n : ℕ), satisfies_conditions K m n h) ↔
    (∃ s t : ℕ, K = 2^(r + s) * t ∧ 2 ∣ t) := sorry

end find_K_l150_150256


namespace choose_9_3_eq_84_l150_150301

theorem choose_9_3_eq_84 : Nat.choose 9 3 = 84 :=
by
  sorry

end choose_9_3_eq_84_l150_150301


namespace quadratic_roots_l150_150342

theorem quadratic_roots (c : ℝ) : 
  (∀ x : ℝ, (x^2 - 3*x + c = 0) ↔ (x = (3 + real.sqrt c) / 2 ∨ x = (3 - real.sqrt c) / 2)) → 
  c = 9 / 5 :=
by
  sorry

end quadratic_roots_l150_150342


namespace joint_purchases_popular_joint_purchases_unpopular_among_neighbors_l150_150526

section JointPurchases

/-- Given that joint purchases allow significant cost savings, reduced overhead costs,
improved quality assessment, and community trust, prove that joint purchases 
are popular in many countries despite the risks. -/
theorem joint_purchases_popular
    (cost_savings : Prop)
    (reduced_overhead_costs : Prop)
    (improved_quality_assessment : Prop)
    (community_trust : Prop)
    : Prop :=
    cost_savings ∧ reduced_overhead_costs ∧ improved_quality_assessment ∧ community_trust

/-- Given that high transaction costs, organizational difficulties,
convenience of proximity to stores, and potential disputes are challenges for neighbors,
prove that joint purchases of groceries and household goods are unpopular among neighbors. -/
theorem joint_purchases_unpopular_among_neighbors
    (high_transaction_costs : Prop)
    (organizational_difficulties : Prop)
    (convenience_proximity : Prop)
    (potential_disputes : Prop)
    : Prop :=
    high_transaction_costs ∧ organizational_difficulties ∧ convenience_proximity ∧ potential_disputes

end JointPurchases

end joint_purchases_popular_joint_purchases_unpopular_among_neighbors_l150_150526


namespace time_to_get_to_lawrence_house_l150_150319

def distance : ℝ := 12
def speed : ℝ := 2

theorem time_to_get_to_lawrence_house : (distance / speed) = 6 :=
by
  sorry

end time_to_get_to_lawrence_house_l150_150319


namespace expression_value_l150_150663

theorem expression_value (x : ℤ) (h : x = -2) : x ^ 2 + 6 * x - 8 = -16 := 
by 
  rw [h]
  sorry

end expression_value_l150_150663


namespace find_a_find_cos_2C_l150_150454

noncomputable def triangle_side_a (A B : Real) (b : Real) (cosA : Real) : Real := 
  3

theorem find_a (A : Real) (B : Real) (b : Real) (cosA : Real) 
  (h₁ : b = 3 * Real.sqrt 2) 
  (h₂ : cosA = Real.sqrt 6 / 3) 
  (h₃ : B = A + Real.pi / 2) : 
  triangle_side_a A B b cosA = 3 := by
  sorry

noncomputable def cos_2C (A B C a b : Real) (cosA sinC : Real) : Real :=
  7 / 9

theorem find_cos_2C (A : Real) (B : Real) (C : Real) (a : Real) (b : Real) (cosA : Real) (sinC: Real)
  (h₁ : b = 3 * Real.sqrt 2) 
  (h₂ : cosA = Real.sqrt 6 / 3)
  (h₃ : B = A + Real.pi /2)
  (h₄ : a = 3)
  (h₅ : sinC = 1 / 3) :
  cos_2C A B C a b cosA sinC = 7 / 9 := by
  sorry

end find_a_find_cos_2C_l150_150454


namespace hypotenuse_length_l150_150645

theorem hypotenuse_length
  (x : ℝ) 
  (h_leg_relation : 3 * x - 3 > 0) -- to ensure the legs are positive
  (hypotenuse : ℝ)
  (area_eq : 1 / 2 * x * (3 * x - 3) = 84)
  (pythagorean : hypotenuse^2 = x^2 + (3 * x - 3)^2) :
  hypotenuse = Real.sqrt 505 :=
by 
  sorry

end hypotenuse_length_l150_150645


namespace quadratic_has_one_real_root_l150_150982

theorem quadratic_has_one_real_root (k : ℝ) : 
  (∃ (x : ℝ), -2 * x^2 + 8 * x + k = 0 ∧ ∀ y, -2 * y^2 + 8 * y + k = 0 → y = x) ↔ k = -8 := 
by
  sorry

end quadratic_has_one_real_root_l150_150982


namespace total_weight_of_dumbbells_l150_150043

theorem total_weight_of_dumbbells : 
  let initial_dumbbells := 4
  let additional_dumbbells := 2
  let weight_per_dumbbell := 20 in
  (initial_dumbbells + additional_dumbbells) * weight_per_dumbbell = 120 := 
by
  -- conditions and definitions
  let initial_dumbbells := 4
  let additional_dumbbells := 2
  let weight_per_dumbbell := 20
  -- calculation
  calc
  (initial_dumbbells + additional_dumbbells) * weight_per_dumbbell 
  = (4 + 2) * 20 : by rw [initial_dumbbells, additional_dumbbells, weight_per_dumbbell]
  ... = 6 * 20 : by norm_num
  ... = 120 : by norm_num

end total_weight_of_dumbbells_l150_150043


namespace total_students_l150_150531

theorem total_students (N : ℕ)
    (h1 : (15 * 75) + (10 * 90) = N * 81) :
    N = 25 :=
by
  sorry

end total_students_l150_150531


namespace find_point_P_l150_150796

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def isEquidistant (p1 p2 : Point3D) (q : Point3D) : Prop :=
  (q.x - p1.x)^2 + (q.y - p1.y)^2 + (q.z - p1.z)^2 = (q.x - p2.x)^2 + (q.y - p2.y)^2 + (q.z - p2.z)^2

theorem find_point_P (P : Point3D) :
  (∀ (Q : Point3D), isEquidistant ⟨2, 3, -4⟩ P Q → (8 * Q.x - 6 * Q.y + 18 * Q.z = 70)) →
  P = ⟨6, 0, 5⟩ :=
by 
  sorry

end find_point_P_l150_150796


namespace corrected_mean_is_124_931_l150_150202

/-
Given:
- original_mean : Real = 125.6
- num_observations : Nat = 100
- incorrect_obs1 : Real = 95.3
- incorrect_obs2 : Real = -15.9
- correct_obs1 : Real = 48.2
- correct_obs2 : Real = -35.7

Prove:
- new_mean == 124.931
-/

noncomputable def original_mean : ℝ := 125.6
def num_observations : ℕ := 100
noncomputable def incorrect_obs1 : ℝ := 95.3
noncomputable def incorrect_obs2 : ℝ := -15.9
noncomputable def correct_obs1 : ℝ := 48.2
noncomputable def correct_obs2 : ℝ := -35.7

noncomputable def incorrect_total_sum : ℝ := original_mean * num_observations
noncomputable def sum_incorrect_obs : ℝ := incorrect_obs1 + incorrect_obs2
noncomputable def sum_correct_obs : ℝ := correct_obs1 + correct_obs2
noncomputable def corrected_total_sum : ℝ := incorrect_total_sum - sum_incorrect_obs + sum_correct_obs
noncomputable def new_mean : ℝ := corrected_total_sum / num_observations

theorem corrected_mean_is_124_931 : new_mean = 124.931 := sorry

end corrected_mean_is_124_931_l150_150202


namespace product_of_two_numbers_is_320_l150_150922

theorem product_of_two_numbers_is_320 (x y : ℕ) (h1 : x + y = 36) (h2 : x - y = 4) (h3 : x = 5 * (y / 4)) : x * y = 320 :=
by {
  sorry
}

end product_of_two_numbers_is_320_l150_150922


namespace find_x1_l150_150421

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4) 
  (h2 : x4 ≤ x3) 
  (h3 : x3 ≤ x2) 
  (h4 : x2 ≤ x1) 
  (h5 : x1 ≤ 1) 
  (condition : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 5) : 
  x1 = 4 / 5 := 
sorry

end find_x1_l150_150421


namespace two_p_in_S_l150_150465

def is_in_S (a b : ℤ) : Prop :=
  ∃ k : ℤ, k = a^2 + 5 * b^2 ∧ Int.gcd a b = 1

def S : Set ℤ := { x | ∃ a b : ℤ, is_in_S a b ∧ a^2 + 5 * b^2 = x }

theorem two_p_in_S (k p n : ℤ) (hp1 : p = 4 * n + 3) (hp2 : Nat.Prime (Int.natAbs p))
  (hk : 0 < k) (hkp : k * p ∈ S) : 2 * p ∈ S := 
sorry

end two_p_in_S_l150_150465


namespace relay_team_orderings_l150_150618

theorem relay_team_orderings (Jordan Mike Friend1 Friend2 Friend3 : Type) :
  ∃ n : ℕ, n = 12 :=
by
  -- Define the team members
  let team : List Type := [Jordan, Mike, Friend1, Friend2, Friend3]
  
  -- Define the number of ways to choose the 4th and 5th runners
  let ways_choose_45 := 2
  
  -- Define the number of ways to order the first 3 runners
  let ways_order_123 := Nat.factorial 3
  
  -- Calculate the total number of ways
  let total_ways := ways_choose_45 * ways_order_123
  
  -- The total ways should be 12
  use total_ways
  have h : total_ways = 12
  sorry
  exact h

end relay_team_orderings_l150_150618


namespace interval_of_monotonic_increase_parallel_vectors_tan_x_perpendicular_vectors_smallest_positive_x_l150_150433

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := 2 * (a x).1 * (b x).1 + 2 * (a x).2 * (b x).2 - 1

theorem interval_of_monotonic_increase (x : ℝ) :
  ∃ k : ℤ, k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 := sorry

theorem parallel_vectors_tan_x (x : ℝ) (h₁ : Real.sin x * Real.cos x - Real.sqrt 3 * Real.cos x * Real.cos x = 0) (h₂ : Real.cos x ≠ 0) :
  Real.tan x = Real.sqrt 3 := sorry

theorem perpendicular_vectors_smallest_positive_x (x : ℝ) (h₁ : Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x * Real.cos x = 0) (h₂ : Real.cos x ≠ 0) :
 x = 5 * Real.pi / 6 := sorry

end interval_of_monotonic_increase_parallel_vectors_tan_x_perpendicular_vectors_smallest_positive_x_l150_150433


namespace initial_amount_l150_150201

theorem initial_amount (x : ℝ) (h : 0.015 * x = 750) : x = 50000 :=
by
  sorry

end initial_amount_l150_150201


namespace interior_edges_sum_l150_150384

theorem interior_edges_sum (frame_width area outer_length : ℝ) (h1 : frame_width = 2) (h2 : area = 30)
  (h3 : outer_length = 7) : 
  2 * (outer_length - 2 * frame_width) + 2 * ((area / outer_length - 4)) = 7 := 
by
  sorry

end interior_edges_sum_l150_150384


namespace apples_per_slice_l150_150550

theorem apples_per_slice 
  (dozens_apples : ℕ)
  (apples_per_dozen : ℕ)
  (number_of_pies : ℕ)
  (pieces_per_pie : ℕ) :
  dozens_apples = 4 →
  apples_per_dozen = 12 →
  number_of_pies = 4 →
  pieces_per_pie = 6 →
  (dozens_apples * apples_per_dozen) / (number_of_pies * pieces_per_pie) = 2 :=
by
  intros h_dozen h_per_dozen h_pies h_pieces
  rw [h_dozen, h_per_dozen, h_pies, h_pieces]
  norm_num
  sorry

end apples_per_slice_l150_150550


namespace total_marks_more_than_physics_l150_150499

variable (P C M : ℕ)

theorem total_marks_more_than_physics :
  (P + C + M > P) ∧ ((C + M) / 2 = 75) → (P + C + M) - P = 150 := by
  intros h
  sorry

end total_marks_more_than_physics_l150_150499


namespace problem_I_problem_II_l150_150584

def f (x : ℝ) : ℝ := abs (x - 1)

theorem problem_I (x : ℝ) : f (2 * x) + f (x + 4) ≥ 8 ↔ x ≤ -10 / 3 ∨ x ≥ 2 := by
  sorry

variable {a b : ℝ}
theorem problem_II (ha : abs a < 1) (hb : abs b < 1) (h_neq : a ≠ 0) : 
  (abs (a * b - 1) / abs a) > abs ((b / a) - 1) :=
by
  sorry

end problem_I_problem_II_l150_150584


namespace tessa_debt_l150_150864

theorem tessa_debt :
  let initial_debt : ℤ := 40 in
  let repayment : ℤ := initial_debt / 2 in
  let debt_after_repayment : ℤ := initial_debt - repayment in
  let additional_debt : ℤ := 10 in
  debt_after_repayment + additional_debt = 30 :=
by
  -- The proof goes here.
  sorry

end tessa_debt_l150_150864


namespace horizontal_length_of_monitor_l150_150153

def monitor_diagonal := 32
def aspect_ratio_horizontal := 16
def aspect_ratio_height := 9

theorem horizontal_length_of_monitor :
  ∃ (horizontal_length : ℝ), horizontal_length = 512 / Real.sqrt 337 := by
  sorry

end horizontal_length_of_monitor_l150_150153


namespace countSumPairs_correct_l150_150744

def countSumPairs (n : ℕ) : ℕ :=
  n / 2

theorem countSumPairs_correct (n : ℕ) : countSumPairs n = n / 2 := by
  sorry

end countSumPairs_correct_l150_150744


namespace work_completion_time_l150_150219

def workRateB : ℚ := 1 / 18
def workRateA : ℚ := 2 * workRateB
def combinedWorkRate : ℚ := workRateA + workRateB
def days : ℚ := 1 / combinedWorkRate

theorem work_completion_time (h1 : workRateA = 2 * workRateB) (h2 : workRateB = 1 / 18) : days = 6 :=
by
  -- h1: workRateA = 2 * workRateB
  -- h2: workRateB = 1 / 18
  sorry

end work_completion_time_l150_150219


namespace votes_cast_l150_150377

theorem votes_cast (V : ℝ) (h1 : ∃ Vc, Vc = 0.25 * V) (h2 : ∃ Vr, Vr = 0.25 * V + 4000) : V = 8000 :=
sorry

end votes_cast_l150_150377


namespace all_girls_select_same_color_probability_l150_150672

theorem all_girls_select_same_color_probability :
  let white_marbles := 10
  let black_marbles := 10
  let red_marbles := 10
  let girls := 15
  ∀ (total_marbles : ℕ), total_marbles = white_marbles + black_marbles + red_marbles →
  (white_marbles < girls ∧ black_marbles < girls ∧ red_marbles < girls) →
  0 = 0 :=
by
  intros
  sorry

end all_girls_select_same_color_probability_l150_150672


namespace joe_two_kinds_of_fruit_l150_150607

-- Definitions based on the conditions
def meals := ["breakfast", "lunch", "snack", "dinner"] -- 4 meals
def fruits := ["apple", "orange", "banana"] -- 3 kinds of fruits

-- Probability that Joe consumes the same fruit for all meals
noncomputable def prob_same_fruit := (1 / 3) ^ 4

-- Probability that Joe eats at least two different kinds of fruits
noncomputable def prob_at_least_two_kinds := 1 - 3 * prob_same_fruit

theorem joe_two_kinds_of_fruit :
  prob_at_least_two_kinds = 26 / 27 :=
by
  -- Proof omitted for this theorem
  sorry

end joe_two_kinds_of_fruit_l150_150607
