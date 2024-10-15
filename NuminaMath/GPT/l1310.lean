import Mathlib

namespace NUMINAMATH_GPT_abs_distance_equation_1_abs_distance_equation_2_l1310_131023

theorem abs_distance_equation_1 (x : ℚ) : |x - (3 : ℚ)| = 5 ↔ x = 8 ∨ x = -2 := 
sorry

theorem abs_distance_equation_2 (x : ℚ) : |x - (3 : ℚ)| = |x + (1 : ℚ)| ↔ x = 1 :=
sorry

end NUMINAMATH_GPT_abs_distance_equation_1_abs_distance_equation_2_l1310_131023


namespace NUMINAMATH_GPT_find_a_of_odd_function_l1310_131011

theorem find_a_of_odd_function (a : ℝ) (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_neg : ∀ x, x < 0 → f x = x^2 + a * x)
  (h_pos_value : f 2 = 6) : a = 5 := by
  sorry

end NUMINAMATH_GPT_find_a_of_odd_function_l1310_131011


namespace NUMINAMATH_GPT_who_drank_most_l1310_131055

theorem who_drank_most (eunji yujeong yuna : ℝ) 
    (h1 : eunji = 0.5) 
    (h2 : yujeong = 7 / 10) 
    (h3 : yuna = 6 / 10) :
    max (max eunji yujeong) yuna = yujeong :=
by {
    sorry
}

end NUMINAMATH_GPT_who_drank_most_l1310_131055


namespace NUMINAMATH_GPT_boys_without_calculators_l1310_131053

theorem boys_without_calculators :
    ∀ (total_boys students_with_calculators girls_with_calculators : ℕ),
    total_boys = 16 →
    students_with_calculators = 22 →
    girls_with_calculators = 13 →
    total_boys - (students_with_calculators - girls_with_calculators) = 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_boys_without_calculators_l1310_131053


namespace NUMINAMATH_GPT_calculate_expr_l1310_131062

variable (x y : ℝ)
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)

theorem calculate_expr : ((x^3 * y^2)^2 * (x / y^3)) = x^7 * y :=
by sorry

end NUMINAMATH_GPT_calculate_expr_l1310_131062


namespace NUMINAMATH_GPT_cafeteria_total_cost_l1310_131050

-- Definitions based on conditions
def cost_per_coffee := 4
def cost_per_cake := 7
def cost_per_ice_cream := 3
def mell_coffee := 2 
def mell_cake := 1 
def friends_coffee := 2 
def friends_cake := 1 
def friends_ice_cream := 1 
def num_friends := 2
def total_coffee := mell_coffee + num_friends * friends_coffee
def total_cake := mell_cake + num_friends * friends_cake
def total_ice_cream := num_friends * friends_ice_cream

-- Total cost
def total_cost := total_coffee * cost_per_coffee + total_cake * cost_per_cake + total_ice_cream * cost_per_ice_cream

-- Theorem statement
theorem cafeteria_total_cost : total_cost = 51 := by
  sorry

end NUMINAMATH_GPT_cafeteria_total_cost_l1310_131050


namespace NUMINAMATH_GPT_triathlon_minimum_speeds_l1310_131068

theorem triathlon_minimum_speeds (x : ℝ) (T : ℝ := 80) (total_time : ℝ := (800 / x + 20000 / (7.5 * x) + 4000 / (3 * x))) :
  total_time ≤ T → x ≥ 60 ∧ 3 * x = 180 ∧ 7.5 * x = 450 :=
by
  sorry

end NUMINAMATH_GPT_triathlon_minimum_speeds_l1310_131068


namespace NUMINAMATH_GPT_at_most_one_cube_l1310_131069

theorem at_most_one_cube (a : ℕ → ℕ) (h₁ : ∀ n, a (n + 1) = a n ^ 2 + 2018) :
  ∃! n, ∃ m : ℕ, a n = m ^ 3 := sorry

end NUMINAMATH_GPT_at_most_one_cube_l1310_131069


namespace NUMINAMATH_GPT_find_number_l1310_131093

theorem find_number (x : ℤ) (h : 72516 * x = 724797420) : x = 10001 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1310_131093


namespace NUMINAMATH_GPT_AM_GM_Ineq_l1310_131039

theorem AM_GM_Ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
by
  sorry

end NUMINAMATH_GPT_AM_GM_Ineq_l1310_131039


namespace NUMINAMATH_GPT_expression_equals_two_l1310_131004

noncomputable def math_expression : ℝ :=
  27^(1/3) + Real.log 4 + 2 * Real.log 5 - Real.exp (Real.log 3)

theorem expression_equals_two : math_expression = 2 := by
  sorry

end NUMINAMATH_GPT_expression_equals_two_l1310_131004


namespace NUMINAMATH_GPT_ab_value_l1310_131029

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end NUMINAMATH_GPT_ab_value_l1310_131029


namespace NUMINAMATH_GPT_sequence_term_number_l1310_131040

theorem sequence_term_number (n : ℕ) : (n ≥ 1) → (n + 3 = 17 ∧ n + 1 = 15) → n = 14 := 
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_sequence_term_number_l1310_131040


namespace NUMINAMATH_GPT_find_n_eq_6_l1310_131002

theorem find_n_eq_6 (n : ℕ) (p : ℕ) (prime_p : Nat.Prime p) : 2^n + n^2 + 25 = p^3 → n = 6 := by
  sorry

end NUMINAMATH_GPT_find_n_eq_6_l1310_131002


namespace NUMINAMATH_GPT_number_of_rectangles_in_5x5_grid_l1310_131042

-- Number of ways to choose k elements from a set of n elements
def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Given conditions
def points_in_each_direction : ℕ := 5
def number_of_rectangles : ℕ :=
  binomial points_in_each_direction 2 * binomial points_in_each_direction 2

-- Lean statement to prove the problem
theorem number_of_rectangles_in_5x5_grid :
  number_of_rectangles = 100 :=
by
  -- begin Lean proof
  sorry

end NUMINAMATH_GPT_number_of_rectangles_in_5x5_grid_l1310_131042


namespace NUMINAMATH_GPT_number_of_girls_in_basketball_club_l1310_131061

-- Define the number of members in the basketball club
def total_members : ℕ := 30

-- Define the number of members who attended the practice session
def attended : ℕ := 18

-- Define the unknowns: number of boys (B) and number of girls (G)
variables (B G : ℕ)

-- Define the conditions provided in the problem
def condition1 : Prop := B + G = total_members
def condition2 : Prop := B + (1 / 3) * G = attended

-- Define the theorem to prove
theorem number_of_girls_in_basketball_club (B G : ℕ) (h1 : condition1 B G) (h2 : condition2 B G) : G = 18 :=
sorry

end NUMINAMATH_GPT_number_of_girls_in_basketball_club_l1310_131061


namespace NUMINAMATH_GPT_compute_expression_l1310_131078

theorem compute_expression :
  3 * 3^4 - 9^60 / 9^57 = -486 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l1310_131078


namespace NUMINAMATH_GPT_relationship_between_a_and_b_l1310_131047

noncomputable section
open Classical

theorem relationship_between_a_and_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b :=
sorry

end NUMINAMATH_GPT_relationship_between_a_and_b_l1310_131047


namespace NUMINAMATH_GPT_uniquely_identify_figure_l1310_131046

structure Figure where
  is_curve : Bool
  has_axis_of_symmetry : Bool
  has_center_of_symmetry : Bool

def Circle : Figure := { is_curve := true, has_axis_of_symmetry := true, has_center_of_symmetry := true }
def Ellipse : Figure := { is_curve := true, has_axis_of_symmetry := true, has_center_of_symmetry := false }
def Triangle : Figure := { is_curve := false, has_axis_of_symmetry := false, has_center_of_symmetry := false }
def Square : Figure := { is_curve := false, has_axis_of_symmetry := true, has_center_of_symmetry := true }
def Rectangle : Figure := { is_curve := false, has_axis_of_symmetry := true, has_center_of_symmetry := true }
def Parallelogram : Figure := { is_curve := false, has_axis_of_symmetry := false, has_center_of_symmetry := true }
def Trapezoid : Figure := { is_curve := false, has_axis_of_symmetry := false, has_center_of_symmetry := false }

theorem uniquely_identify_figure (figures : List Figure) (q1 q2 q3 : Figure → Bool) :
  ∀ (f : Figure), ∃! (f' : Figure), 
    q1 f' = q1 f ∧ q2 f' = q2 f ∧ q3 f' = q3 f :=
by
  sorry

end NUMINAMATH_GPT_uniquely_identify_figure_l1310_131046


namespace NUMINAMATH_GPT_probability_ace_spades_then_king_spades_l1310_131080

theorem probability_ace_spades_then_king_spades :
  ∃ (p : ℚ), (p = 1/52 * 1/51) := sorry

end NUMINAMATH_GPT_probability_ace_spades_then_king_spades_l1310_131080


namespace NUMINAMATH_GPT_combined_avg_score_l1310_131074

theorem combined_avg_score (x : ℕ) : 
  let avgA := 65
  let avgB := 90 
  let avgC := 77 
  let ratioA := 4 
  let ratioB := 6 
  let ratioC := 5 
  let total_students := 15 * x 
  let total_score := (ratioA * avgA + ratioB * avgB + ratioC * avgC) * x
  (total_score / total_students) = 79 := 
by
  sorry

end NUMINAMATH_GPT_combined_avg_score_l1310_131074


namespace NUMINAMATH_GPT_single_reduction_equivalent_l1310_131096

theorem single_reduction_equivalent (P : ℝ) (P_pos : 0 < P) : 
  (P - (P - 0.30 * P)) / P = 0.70 := 
by
  -- Let's denote the original price by P, 
  -- apply first 25% and then 60% reduction 
  -- and show that it's equivalent to a single 70% reduction
  sorry

end NUMINAMATH_GPT_single_reduction_equivalent_l1310_131096


namespace NUMINAMATH_GPT_total_baseball_cards_is_100_l1310_131038

-- Define the initial number of baseball cards Mike has
def initial_baseball_cards : ℕ := 87

-- Define the number of baseball cards Sam gave to Mike
def given_baseball_cards : ℕ := 13

-- Define the total number of baseball cards Mike has now
def total_baseball_cards : ℕ := initial_baseball_cards + given_baseball_cards

-- State the theorem that the total number of baseball cards is 100
theorem total_baseball_cards_is_100 : total_baseball_cards = 100 := by
  sorry

end NUMINAMATH_GPT_total_baseball_cards_is_100_l1310_131038


namespace NUMINAMATH_GPT_total_players_is_60_l1310_131070

-- Define the conditions
def Cricket_players : ℕ := 25
def Hockey_players : ℕ := 20
def Football_players : ℕ := 30
def Softball_players : ℕ := 18

def Cricket_and_Hockey : ℕ := 5
def Cricket_and_Football : ℕ := 8
def Cricket_and_Softball : ℕ := 3
def Hockey_and_Football : ℕ := 4
def Hockey_and_Softball : ℕ := 6
def Football_and_Softball : ℕ := 9

def Cricket_Hockey_and_Football_not_Softball : ℕ := 2

-- Define total unique players present on the ground
def total_unique_players : ℕ :=
  Cricket_players + Hockey_players + Football_players + Softball_players -
  (Cricket_and_Hockey + Cricket_and_Football + Cricket_and_Softball +
   Hockey_and_Football + Hockey_and_Softball + Football_and_Softball) +
  Cricket_Hockey_and_Football_not_Softball

-- Statement
theorem total_players_is_60:
  total_unique_players = 60 :=
by
  sorry

end NUMINAMATH_GPT_total_players_is_60_l1310_131070


namespace NUMINAMATH_GPT_foldable_positions_are_7_l1310_131000

-- Define the initial polygon with 6 congruent squares forming a cross shape
def initial_polygon : Prop :=
  -- placeholder definition, in practice, this would be a more detailed geometrical model
  sorry

-- Define the positions where an additional square can be attached (11 positions in total)
def position (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 11

-- Define the resulting polygon when an additional square is attached at position n
def resulting_polygon (n : ℕ) : Prop :=
  position n ∧ initial_polygon

-- Define the condition that a polygon can be folded into a cube with one face missing
def can_fold_to_cube_with_missing_face (p : Prop) : Prop := sorry

-- The theorem that needs to be proved
theorem foldable_positions_are_7 : 
  ∃ (positions : Finset ℕ), 
    positions.card = 7 ∧ 
    ∀ n ∈ positions, can_fold_to_cube_with_missing_face (resulting_polygon n) :=
  sorry

end NUMINAMATH_GPT_foldable_positions_are_7_l1310_131000


namespace NUMINAMATH_GPT_trapezoid_area_l1310_131083

-- Define the properties of the isosceles trapezoid
structure IsoscelesTrapezoid where
  leg : ℝ
  diagonal : ℝ
  longer_base : ℝ
  is_isosceles : True
  legs_equal : True

-- Provide the specific conditions of the problem
def trapezoid : IsoscelesTrapezoid := {
  leg := 40,
  diagonal := 50,
  longer_base := 60,
  is_isosceles := True.intro,
  legs_equal := True.intro
}

-- State the main theorem to translate the proof problem into Lean
theorem trapezoid_area (T : IsoscelesTrapezoid) : T = trapezoid →
  (∃ A : ℝ, A = (15000 - 2000 * Real.sqrt 11) / 9) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_trapezoid_area_l1310_131083


namespace NUMINAMATH_GPT_cmp_c_b_a_l1310_131034

noncomputable def a : ℝ := 17 / 18
noncomputable def b : ℝ := Real.cos (1 / 3)
noncomputable def c : ℝ := 3 * Real.sin (1 / 3)

theorem cmp_c_b_a:
  c > b ∧ b > a := by
  sorry

end NUMINAMATH_GPT_cmp_c_b_a_l1310_131034


namespace NUMINAMATH_GPT_find_prime_pair_l1310_131082

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem find_prime_pair (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : p > q) (h_prime : is_prime (p^5 - q^5)) : (p, q) = (3, 2) := 
  sorry

end NUMINAMATH_GPT_find_prime_pair_l1310_131082


namespace NUMINAMATH_GPT_find_a_l1310_131094

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then
  x * (x + 1)
else
  -((-x) * ((-x) + 1))

theorem find_a (a : ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x) (h_pos : ∀ x : ℝ, x >= 0 → f x = x * (x + 1)) (h_a: f a = -2) : a = -1 :=
sorry

end NUMINAMATH_GPT_find_a_l1310_131094


namespace NUMINAMATH_GPT_coffee_price_increase_l1310_131084

variable (C : ℝ) -- cost per pound of green tea and coffee in June
variable (P_green_tea_july : ℝ := 0.1) -- price of green tea per pound in July
variable (mixture_cost : ℝ := 3.15) -- cost of mixture of equal quantities of green tea and coffee for 3 lbs
variable (green_tea_cost_per_lb_july : ℝ := 0.1) -- cost per pound of green tea in July
variable (green_tea_weight : ℝ := 1.5) -- weight of green tea in the mixture in lbs
variable (coffee_weight : ℝ := 1.5) -- weight of coffee in the mixture in lbs
variable (coffee_cost_per_lb_july : ℝ := 2.0) -- cost per pound of coffee in July

theorem coffee_price_increase :
  C = 1 → mixture_cost = 3.15 →
  P_green_tea_july * C = green_tea_cost_per_lb_july →
  green_tea_weight * green_tea_cost_per_lb_july + coffee_weight * coffee_cost_per_lb_july = mixture_cost →
  (coffee_cost_per_lb_july - C) / C * 100 = 100 :=
by
  intros
  sorry

end NUMINAMATH_GPT_coffee_price_increase_l1310_131084


namespace NUMINAMATH_GPT_positive_number_solution_exists_l1310_131030

theorem positive_number_solution_exists (x : ℝ) (h₁ : 0 < x) (h₂ : (2 / 3) * x = (64 / 216) * (1 / x)) : x = 2 / 3 :=
by sorry

end NUMINAMATH_GPT_positive_number_solution_exists_l1310_131030


namespace NUMINAMATH_GPT_abcd_inequality_l1310_131021

theorem abcd_inequality (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
(h_eq : (a^2 / (1 + a^2)) + (b^2 / (1 + b^2)) + (c^2 / (1 + c^2)) + (d^2 / (1 + d^2)) = 1) :
  a * b * c * d ≤ 1 / 9 :=
sorry

end NUMINAMATH_GPT_abcd_inequality_l1310_131021


namespace NUMINAMATH_GPT_number_of_males_is_one_part_l1310_131001

-- Define the total population
def population : ℕ := 480

-- Define the number of divided parts
def parts : ℕ := 3

-- Define the population part represented by one square.
def part_population (total_population : ℕ) (n_parts : ℕ) : ℕ :=
  total_population / n_parts

-- The Lean statement for the problem
theorem number_of_males_is_one_part : part_population population parts = 160 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_number_of_males_is_one_part_l1310_131001


namespace NUMINAMATH_GPT_water_bottles_needed_l1310_131043

theorem water_bottles_needed : 
  let number_of_people := 4
  let hours_to_destination := 8
  let hours_to_return := 8
  let hours_total := hours_to_destination + hours_to_return
  let bottles_per_person_per_hour := 1 / 2
  let total_bottles_per_hour := number_of_people * bottles_per_person_per_hour
  let total_bottles := total_bottles_per_hour * hours_total
  total_bottles = 32 :=
by
  sorry

end NUMINAMATH_GPT_water_bottles_needed_l1310_131043


namespace NUMINAMATH_GPT_reggie_father_money_l1310_131051

theorem reggie_father_money :
  let books := 5
  let cost_per_book := 2
  let amount_left := 38
  books * cost_per_book + amount_left = 48 :=
by
  sorry

end NUMINAMATH_GPT_reggie_father_money_l1310_131051


namespace NUMINAMATH_GPT_modulus_of_z_l1310_131005

open Complex

theorem modulus_of_z (z : ℂ) (hz : (1 + I) * z = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_modulus_of_z_l1310_131005


namespace NUMINAMATH_GPT_cannot_determine_b_l1310_131060

theorem cannot_determine_b 
  (a b c d : ℝ) 
  (h_avg : (a + b + c + d) / 4 = 12.345) 
  (h_ineq : a > b ∧ b > c ∧ c > d) : 
  ¬((b = 12.345) ∨ (b > 12.345) ∨ (b < 12.345)) :=
sorry

end NUMINAMATH_GPT_cannot_determine_b_l1310_131060


namespace NUMINAMATH_GPT_difference_of_squares_l1310_131091

theorem difference_of_squares : (540^2 - 460^2 = 80000) :=
by
  have a := 540
  have b := 460
  have identity := (a + b) * (a - b)
  sorry

end NUMINAMATH_GPT_difference_of_squares_l1310_131091


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_has_max_value_l1310_131066

noncomputable section
open Classical

-- Defining an arithmetic sequence with first term a1 and common difference d
def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + d * (n - 1)

-- Defining the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

-- The main statement to prove: Sn has a maximum value given conditions a1 > 0 and d < 0
theorem sum_arithmetic_sequence_has_max_value (a1 d : ℝ) (h1 : a1 > 0) (h2 : d < 0) :
  ∃ M, ∀ n, sum_arithmetic_sequence a1 d n ≤ M :=
by
  sorry

end NUMINAMATH_GPT_sum_arithmetic_sequence_has_max_value_l1310_131066


namespace NUMINAMATH_GPT_doubles_tournament_handshakes_l1310_131056

theorem doubles_tournament_handshakes :
  let num_teams := 3
  let players_per_team := 2
  let total_players := num_teams * players_per_team
  let handshakes_per_player := total_players - 2
  let total_handshakes := total_players * handshakes_per_player / 2
  total_handshakes = 12 :=
by
  sorry

end NUMINAMATH_GPT_doubles_tournament_handshakes_l1310_131056


namespace NUMINAMATH_GPT_unique_solution_in_z3_l1310_131081

theorem unique_solution_in_z3 (x y z : ℤ) (h : x^3 + 2 * y^3 = 4 * z^3) : 
  x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end NUMINAMATH_GPT_unique_solution_in_z3_l1310_131081


namespace NUMINAMATH_GPT_point_distance_5_5_l1310_131079

-- Define the distance function in the context of the problem
def distance_from_origin (x : ℝ) : ℝ := abs x

-- Formalize the proposition
theorem point_distance_5_5 (x : ℝ) : distance_from_origin x = 5.5 → (x = -5.5 ∨ x = 5.5) :=
by
  intro h
  simp [distance_from_origin] at h
  sorry

end NUMINAMATH_GPT_point_distance_5_5_l1310_131079


namespace NUMINAMATH_GPT_well_filled_ways_1_5_l1310_131028

-- Define a structure for representing the conditions of the figure filled with integers
structure WellFilledFigure where
  top_circle : ℕ
  shaded_circle_possibilities : Finset ℕ
  sub_diagram_possibilities : ℕ

-- Define an example of this structure corresponding to our problem
def figure1_5 : WellFilledFigure :=
  { top_circle := 5,
    shaded_circle_possibilities := {1, 2, 3, 4},
    sub_diagram_possibilities := 2 }

-- Define the theorem statement
theorem well_filled_ways_1_5 (f : WellFilledFigure) : (f.top_circle = 5) → 
  (f.shaded_circle_possibilities.card = 4) → 
  (f.sub_diagram_possibilities = 2) → 
  (4 * 2 = 8) := by
  sorry

end NUMINAMATH_GPT_well_filled_ways_1_5_l1310_131028


namespace NUMINAMATH_GPT_sides_of_length_five_l1310_131085

theorem sides_of_length_five (GH HI : ℝ) (L : ℝ) (total_perimeter : ℝ) :
  GH = 7 → HI = 5 → total_perimeter = 38 → (∃ n m : ℕ, n + m = 6 ∧ n * 7 + m * 5 = 38 ∧ m = 2) := by
  intros hGH hHI hPerimeter
  sorry

end NUMINAMATH_GPT_sides_of_length_five_l1310_131085


namespace NUMINAMATH_GPT_non_congruent_triangles_l1310_131026

-- Definition of points and isosceles property
variable (A B C P Q R : Type)
variable [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited Q] [Inhabited R]

-- Conditions of the problem
def is_isosceles (A B C : Type) : Prop := (A = B) ∧ (A = C)
def is_midpoint (P Q R : Type) (A B C : Type) : Prop := sorry -- precise formal definition omitted for brevity

-- Theorem stating the final result
theorem non_congruent_triangles (A B C P Q R : Type)
  (h_iso : is_isosceles A B C)
  (h_midpoints : is_midpoint P Q R A B C) :
  ∃ (n : ℕ), n = 4 := 
  by 
    -- proof abbreviated
    sorry

end NUMINAMATH_GPT_non_congruent_triangles_l1310_131026


namespace NUMINAMATH_GPT_evan_books_two_years_ago_l1310_131052

theorem evan_books_two_years_ago (B B2 : ℕ) 
  (h1 : 860 = 5 * B + 60) 
  (h2 : B2 = B + 40) : 
  B2 = 200 := 
by 
  sorry

end NUMINAMATH_GPT_evan_books_two_years_ago_l1310_131052


namespace NUMINAMATH_GPT_find_m_l1310_131088

theorem find_m {m : ℝ} (a b : ℝ × ℝ) (H : a = (3, m) ∧ b = (2, -1)) (H_dot : a.1 * b.1 + a.2 * b.2 = 0) : m = 6 := 
by
  sorry

end NUMINAMATH_GPT_find_m_l1310_131088


namespace NUMINAMATH_GPT_total_number_of_legs_l1310_131092

def kangaroos : ℕ := 23
def goats : ℕ := 3 * kangaroos
def legs_of_kangaroo : ℕ := 2
def legs_of_goat : ℕ := 4

theorem total_number_of_legs : 
  (kangaroos * legs_of_kangaroo + goats * legs_of_goat) = 322 := by
  sorry

end NUMINAMATH_GPT_total_number_of_legs_l1310_131092


namespace NUMINAMATH_GPT_part_a_exists_part_b_impossible_l1310_131072

def gridSize : Nat := 7 * 14
def cellCount (x y : Nat) : Nat := 4 * x + 3 * y
def x_equals_y_condition (x y : Nat) : Prop := x = y
def x_greater_y_condition (x y : Nat) : Prop := x > y

theorem part_a_exists (x y : Nat) (h : cellCount x y = gridSize) : ∃ (x y : Nat), x_equals_y_condition x y ∧ cellCount x y = gridSize :=
by
  sorry

theorem part_b_impossible (x y : Nat) (h : cellCount x y = gridSize) : ¬ ∃ (x y : Nat), x_greater_y_condition x y ∧ cellCount x y = gridSize :=
by
  sorry


end NUMINAMATH_GPT_part_a_exists_part_b_impossible_l1310_131072


namespace NUMINAMATH_GPT_beth_longer_distance_by_5_miles_l1310_131003

noncomputable def average_speed_john : ℝ := 40
noncomputable def time_john_hours : ℝ := 30 / 60
noncomputable def distance_john : ℝ := average_speed_john * time_john_hours

noncomputable def average_speed_beth : ℝ := 30
noncomputable def time_beth_hours : ℝ := (30 + 20) / 60
noncomputable def distance_beth : ℝ := average_speed_beth * time_beth_hours

theorem beth_longer_distance_by_5_miles : distance_beth - distance_john = 5 := by 
  sorry

end NUMINAMATH_GPT_beth_longer_distance_by_5_miles_l1310_131003


namespace NUMINAMATH_GPT_income_ratio_l1310_131044

theorem income_ratio (I1 I2 E1 E2 : ℕ)
  (hI1 : I1 = 3500)
  (hE_ratio : (E1:ℚ) / E2 = 3 / 2)
  (hSavings : ∀ (x y : ℕ), x - E1 = 1400 ∧ y - E2 = 1400 → x = I1 ∧ y = I2) :
  I1 / I2 = 5 / 4 :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_income_ratio_l1310_131044


namespace NUMINAMATH_GPT_expression_X_l1310_131013

variable {a b X : ℝ}

theorem expression_X (h1 : a / b = 4 / 3) (h2 : (3 * a + 2 * b) / X = 3) : X = 2 * b := 
sorry

end NUMINAMATH_GPT_expression_X_l1310_131013


namespace NUMINAMATH_GPT_scarves_per_box_l1310_131099

theorem scarves_per_box (S M : ℕ) (h1 : S = M) (h2 : 6 * (S + M) = 60) : S = 5 :=
by
  sorry

end NUMINAMATH_GPT_scarves_per_box_l1310_131099


namespace NUMINAMATH_GPT_sequence_sum_formula_l1310_131054

theorem sequence_sum_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_S : ∀ n, S n = (1 / 6) * (a n ^ 2 + 3 * a n - 4)) : 
  ∀ n, S n = (3 / 2) * n ^ 2 + (5 / 2) * n :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_formula_l1310_131054


namespace NUMINAMATH_GPT_find_a_of_inequality_solution_set_l1310_131009

theorem find_a_of_inequality_solution_set
  (a : ℝ)
  (h : ∀ x : ℝ, |a * x + 2| < 6 ↔ -1 < x ∧ x < 2) :
  a = -4 :=
sorry

end NUMINAMATH_GPT_find_a_of_inequality_solution_set_l1310_131009


namespace NUMINAMATH_GPT_compatibility_condition_l1310_131019

theorem compatibility_condition (a b c d x : ℝ) 
  (h1 : a * x + b = 0) (h2 : c * x + d = 0) : a * d - b * c = 0 :=
sorry

end NUMINAMATH_GPT_compatibility_condition_l1310_131019


namespace NUMINAMATH_GPT_games_against_other_division_l1310_131010

theorem games_against_other_division
  (N M : ℕ) (h1 : N > 2 * M) (h2 : M > 5)
  (total_games : N * 4 + 5 * M = 82) :
  5 * M = 30 :=
by
  sorry

end NUMINAMATH_GPT_games_against_other_division_l1310_131010


namespace NUMINAMATH_GPT_tycho_jogging_schedule_count_l1310_131071

-- Definition of the conditions
def non_consecutive_shot_schedule (days : Finset ℕ) : Prop :=
  ∀ day ∈ days, ∀ next_day ∈ days, day < next_day → next_day - day > 1

-- Definition stating there are exactly seven valid schedules
theorem tycho_jogging_schedule_count :
  ∃ (S : Finset (Finset ℕ)), (∀ s ∈ S, s.card = 3 ∧ non_consecutive_shot_schedule s) ∧ S.card = 7 := 
sorry

end NUMINAMATH_GPT_tycho_jogging_schedule_count_l1310_131071


namespace NUMINAMATH_GPT_product_of_four_consecutive_integers_is_not_square_l1310_131048

theorem product_of_four_consecutive_integers_is_not_square (n : ℤ) : 
  ¬ ∃ k : ℤ, k * k = (n-1)*n*(n+1)*(n+2) :=
sorry

end NUMINAMATH_GPT_product_of_four_consecutive_integers_is_not_square_l1310_131048


namespace NUMINAMATH_GPT_remainder_of_square_l1310_131097

theorem remainder_of_square (n : ℤ) (h : n % 5 = 3) : n^2 % 5 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_of_square_l1310_131097


namespace NUMINAMATH_GPT_Riverdale_High_students_l1310_131059

theorem Riverdale_High_students
  (f j : ℕ)
  (h1 : (3 / 7) * f + (3 / 4) * j = 234)
  (h2 : f + j = 420) :
  f = 64 ∧ j = 356 := by
  sorry

end NUMINAMATH_GPT_Riverdale_High_students_l1310_131059


namespace NUMINAMATH_GPT_books_count_l1310_131077

theorem books_count (books_per_box : ℕ) (boxes : ℕ) (total_books : ℕ) 
  (h1 : books_per_box = 3)
  (h2 : boxes = 8)
  (h3 : total_books = books_per_box * boxes) : 
  total_books = 24 := 
by 
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_books_count_l1310_131077


namespace NUMINAMATH_GPT_find_larger_number_l1310_131025

theorem find_larger_number (x y : ℤ) (h1 : 5 * y = 6 * x) (h2 : y - x = 12) : y = 72 :=
sorry

end NUMINAMATH_GPT_find_larger_number_l1310_131025


namespace NUMINAMATH_GPT_car_distance_problem_l1310_131057

-- A definition for the initial conditions.
def initial_conditions (D : ℝ) (S : ℝ) (T : ℝ) : Prop :=
  T = 6 ∧ S = 50 ∧ (3/2 * T = 9)

-- The statement corresponding to the given problem.
theorem car_distance_problem (D : ℝ) (S : ℝ) (T : ℝ) :
  initial_conditions D S T → D = 450 :=
by
  -- leave the proof as an exercise.
  sorry

end NUMINAMATH_GPT_car_distance_problem_l1310_131057


namespace NUMINAMATH_GPT_sum_second_largest_and_smallest_l1310_131063

theorem sum_second_largest_and_smallest :
  let numbers := [10, 11, 12, 13, 14]
  ∃ second_largest second_smallest, (List.nthLe numbers 3 sorry = second_largest ∧ List.nthLe numbers 1 sorry = second_smallest ∧ second_largest + second_smallest = 24) :=
sorry

end NUMINAMATH_GPT_sum_second_largest_and_smallest_l1310_131063


namespace NUMINAMATH_GPT_alyssa_total_games_l1310_131008

def calc_total_games (games_this_year games_last_year games_next_year : ℕ) : ℕ :=
  games_this_year + games_last_year + games_next_year

theorem alyssa_total_games :
  calc_total_games 11 13 15 = 39 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_alyssa_total_games_l1310_131008


namespace NUMINAMATH_GPT_power_modulus_difference_l1310_131015

theorem power_modulus_difference (m : ℤ) :
  (51 % 6 = 3) → (9 % 6 = 3) → ((51 : ℤ)^1723 - (9 : ℤ)^1723) % 6 = 0 :=
by 
  intros h1 h2
  sorry

end NUMINAMATH_GPT_power_modulus_difference_l1310_131015


namespace NUMINAMATH_GPT_cube_mod_35_divisors_l1310_131012

theorem cube_mod_35_divisors (a : ℤ) : (35 ∣ a^3 - 1) ↔
  (∃ k : ℤ, a = 35 * k + 1) ∨ 
  (∃ k : ℤ, a = 35 * k + 11) ∨ 
  (∃ k : ℤ, a = 35 * k + 16) :=
by sorry

end NUMINAMATH_GPT_cube_mod_35_divisors_l1310_131012


namespace NUMINAMATH_GPT_incorrect_option_l1310_131064

-- Definitions and conditions from the problem
def p (x : ℝ) : Prop := (x - 2) * Real.sqrt (x^2 - 3*x + 2) ≥ 0
def q (k : ℝ) : Prop := ∀ x : ℝ, k * x^2 - k * x - 1 < 0

-- The Lean 4 statement to verify the problem
theorem incorrect_option :
  (¬ ∃ x, p x) ∧ (∃ k, q k) ∧
  (∀ k, -4 < k ∧ k ≤ 0 → q k) →
  (∃ x, ¬p x) :=
  by
  sorry

end NUMINAMATH_GPT_incorrect_option_l1310_131064


namespace NUMINAMATH_GPT_find_x_for_equation_l1310_131087

theorem find_x_for_equation : ∃ x : ℝ, (1 / 2) + ((2 / 3) * x + 4) - (8 / 16) = 4.25 ↔ x = 0.375 := 
by
  sorry

end NUMINAMATH_GPT_find_x_for_equation_l1310_131087


namespace NUMINAMATH_GPT_benny_eggs_l1310_131073

theorem benny_eggs (dozen_count : ℕ) (eggs_per_dozen : ℕ) (total_eggs : ℕ) 
  (h1 : dozen_count = 7) 
  (h2 : eggs_per_dozen = 12) 
  (h3 : total_eggs = dozen_count * eggs_per_dozen) : 
  total_eggs = 84 := 
by 
  sorry

end NUMINAMATH_GPT_benny_eggs_l1310_131073


namespace NUMINAMATH_GPT_depth_of_box_l1310_131022

theorem depth_of_box (length width depth : ℕ) (side_length : ℕ)
  (h_length : length = 30)
  (h_width : width = 48)
  (h_side_length : Nat.gcd length width = side_length)
  (h_cubes : side_length ^ 3 = 216)
  (h_volume : 80 * (side_length ^ 3) = length * width * depth) :
  depth = 12 :=
by
  sorry

end NUMINAMATH_GPT_depth_of_box_l1310_131022


namespace NUMINAMATH_GPT_no_integer_solution_exists_l1310_131076

theorem no_integer_solution_exists :
  ¬ ∃ m n : ℤ, m^3 = 3 * n^2 + 3 * n + 7 := by
  sorry

end NUMINAMATH_GPT_no_integer_solution_exists_l1310_131076


namespace NUMINAMATH_GPT_inequality_lt_l1310_131036

theorem inequality_lt (x y : ℝ) (h1 : x > y) (h2 : y > 0) (n k : ℕ) (h3 : n > k) :
  (x^k - y^k) ^ n < (x^n - y^n) ^ k := 
  sorry

end NUMINAMATH_GPT_inequality_lt_l1310_131036


namespace NUMINAMATH_GPT_total_salmon_l1310_131095

def male_salmon : Nat := 712261
def female_salmon : Nat := 259378

theorem total_salmon :
  male_salmon + female_salmon = 971639 := by
  sorry

end NUMINAMATH_GPT_total_salmon_l1310_131095


namespace NUMINAMATH_GPT_Frank_worked_days_l1310_131031

theorem Frank_worked_days
  (h_per_day : ℕ) (total_hours : ℕ) (d : ℕ) 
  (h_day_def : h_per_day = 8) 
  (total_hours_def : total_hours = 32) 
  (d_def : d = total_hours / h_per_day) : 
  d = 4 :=
by 
  rw [total_hours_def, h_day_def] at d_def
  exact d_def

end NUMINAMATH_GPT_Frank_worked_days_l1310_131031


namespace NUMINAMATH_GPT_ratio_of_discount_l1310_131035

theorem ratio_of_discount (price_pair1 price_pair2 : ℕ) (total_paid : ℕ) (discount_percent : ℕ) (h1 : price_pair1 = 40)
    (h2 : price_pair2 = 60) (h3 : total_paid = 60) (h4 : discount_percent = 50) :
    (price_pair1 * discount_percent / 100) / (price_pair1 + (price_pair2 - price_pair1 * discount_percent / 100)) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_discount_l1310_131035


namespace NUMINAMATH_GPT_exact_sunny_days_probability_l1310_131007

noncomputable def choose (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

def rain_prob : ℚ := 3 / 4
def sun_prob : ℚ := 1 / 4
def days : ℕ := 5

theorem exact_sunny_days_probability : (choose days 2 * (sun_prob^2 * rain_prob^3) = 135 / 512) :=
by
  sorry

end NUMINAMATH_GPT_exact_sunny_days_probability_l1310_131007


namespace NUMINAMATH_GPT_cost_backpack_is_100_l1310_131032

-- Definitions based on the conditions
def cost_wallet : ℕ := 50
def cost_sneakers_per_pair : ℕ := 100
def num_sneakers_pairs : ℕ := 2
def cost_jeans_per_pair : ℕ := 50
def num_jeans_pairs : ℕ := 2
def total_spent : ℕ := 450

-- The problem statement
theorem cost_backpack_is_100 (x : ℕ) 
  (leonard_total : ℕ := cost_wallet + num_sneakers_pairs * cost_sneakers_per_pair) 
  (michael_non_backpack_total : ℕ := num_jeans_pairs * cost_jeans_per_pair) :
  total_spent = leonard_total + michael_non_backpack_total + x → x = 100 := 
by
  unfold cost_wallet cost_sneakers_per_pair num_sneakers_pairs total_spent cost_jeans_per_pair num_jeans_pairs
  intro h
  sorry

end NUMINAMATH_GPT_cost_backpack_is_100_l1310_131032


namespace NUMINAMATH_GPT_mass_percentage_of_Cl_in_bleach_l1310_131017

-- Definitions based on conditions
def Na_molar_mass : Float := 22.99
def Cl_molar_mass : Float := 35.45
def O_molar_mass : Float := 16.00

def NaClO_molar_mass : Float := Na_molar_mass + Cl_molar_mass + O_molar_mass

def mass_NaClO (mass_na: Float) (mass_cl: Float) (mass_o: Float) : Float :=
  mass_na + mass_cl + mass_o

def mass_of_NaClO : Float := 100.0

def mass_of_Cl_in_NaClO (mass_of_NaClO: Float) : Float :=
  (Cl_molar_mass / NaClO_molar_mass) * mass_of_NaClO

-- Statement to prove
theorem mass_percentage_of_Cl_in_bleach :
  let mass_Cl := mass_of_Cl_in_NaClO mass_of_NaClO
  (mass_Cl / mass_of_NaClO) * 100 = 47.61 :=
by 
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_mass_percentage_of_Cl_in_bleach_l1310_131017


namespace NUMINAMATH_GPT_side_length_of_square_l1310_131086

theorem side_length_of_square :
  ∃ n : ℝ, n^2 = 9/16 ∧ n = 3/4 :=
sorry

end NUMINAMATH_GPT_side_length_of_square_l1310_131086


namespace NUMINAMATH_GPT_cubic_expansion_solution_l1310_131018

theorem cubic_expansion_solution (x y : ℕ) (h_x : x = 27) (h_y : y = 9) : 
  x^3 + 3 * x^2 * y + 3 * x * y^2 + y^3 = 46656 :=
by
  sorry

end NUMINAMATH_GPT_cubic_expansion_solution_l1310_131018


namespace NUMINAMATH_GPT_revenue_after_decrease_l1310_131041

theorem revenue_after_decrease (original_revenue : ℝ) (percentage_decrease : ℝ) (final_revenue : ℝ) 
  (h1 : original_revenue = 69.0) 
  (h2 : percentage_decrease = 24.637681159420293) 
  (h3 : final_revenue = original_revenue - (original_revenue * (percentage_decrease / 100))) 
  : final_revenue = 52.0 :=
by
  sorry

end NUMINAMATH_GPT_revenue_after_decrease_l1310_131041


namespace NUMINAMATH_GPT_jackson_sandwiches_l1310_131049

theorem jackson_sandwiches (weeks : ℕ) (missed_wednesdays : ℕ) (missed_fridays : ℕ)
    (h_weeks : weeks = 36) (h_missed_wednesdays : missed_wednesdays = 1) (h_missed_fridays : missed_fridays = 2) :
    let total_days := weeks * 2
    let missed_days := missed_wednesdays + missed_fridays
    total_days - missed_days = 69 :=
by
    sorry

end NUMINAMATH_GPT_jackson_sandwiches_l1310_131049


namespace NUMINAMATH_GPT_quadratic_inequalities_l1310_131024

variable (c x₁ y₁ y₂ y₃ : ℝ)
noncomputable def quadratic_function := -x₁^2 + 2*x₁ + c

theorem quadratic_inequalities
  (h_c : c < 0)
  (h_y₁ : quadratic_function c x₁ > 0)
  (h_y₂ : y₂ = quadratic_function c (x₁ - 2))
  (h_y₃ : y₃ = quadratic_function c (x₁ + 2)) :
  y₂ < 0 ∧ y₃ < 0 :=
by sorry

end NUMINAMATH_GPT_quadratic_inequalities_l1310_131024


namespace NUMINAMATH_GPT_third_place_books_max_l1310_131045

theorem third_place_books_max (x y z : ℕ) (hx : 100 ∣ x) (hxpos : 0 < x) (hy : 100 ∣ y) (hz : 100 ∣ z)
  (h_sum : 2 * x + 100 + x + 100 + x + y + z ≤ 10000)
  (h_first_eq : 2 * x + 100 = x + 100 + x)
  (h_second_eq : x + 100 = y + z) 
  : x ≤ 1900 := sorry

end NUMINAMATH_GPT_third_place_books_max_l1310_131045


namespace NUMINAMATH_GPT_smallest_6_digit_divisible_by_111_l1310_131014

theorem smallest_6_digit_divisible_by_111 :
  ∃ x : ℕ, 100000 ≤ x ∧ x ≤ 999999 ∧ x % 111 = 0 ∧ x = 100011 :=
  by
    sorry

end NUMINAMATH_GPT_smallest_6_digit_divisible_by_111_l1310_131014


namespace NUMINAMATH_GPT_win_sector_area_l1310_131006

theorem win_sector_area (r : ℝ) (h1 : r = 8) (h2 : (1 / 4) = 1 / 4) : 
  ∃ (area : ℝ), area = 16 * Real.pi := 
by
  existsi (16 * Real.pi); exact sorry

end NUMINAMATH_GPT_win_sector_area_l1310_131006


namespace NUMINAMATH_GPT_green_ball_removal_l1310_131033

variable (total_balls : ℕ)
variable (initial_green_balls : ℕ)
variable (initial_yellow_balls : ℕ)
variable (desired_green_percentage : ℚ)
variable (removals : ℕ)

theorem green_ball_removal :
  initial_green_balls = 420 → 
  total_balls = 600 → 
  desired_green_percentage = 3 / 5 →
  (420 - removals) / (600 - removals) = desired_green_percentage → 
  removals = 150 :=
sorry

end NUMINAMATH_GPT_green_ball_removal_l1310_131033


namespace NUMINAMATH_GPT_yards_dyed_green_calc_l1310_131020

-- Given conditions: total yards dyed and yards dyed pink
def total_yards_dyed : ℕ := 111421
def yards_dyed_pink : ℕ := 49500

-- Goal: Prove the number of yards dyed green
theorem yards_dyed_green_calc : total_yards_dyed - yards_dyed_pink = 61921 :=
by 
-- sorry means that the proof is skipped.
sorry

end NUMINAMATH_GPT_yards_dyed_green_calc_l1310_131020


namespace NUMINAMATH_GPT_equivalent_expression_l1310_131058

noncomputable def problem_statement (α β γ δ p q : ℝ) :=
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = -2 * (p^2 - q^2) + 4

theorem equivalent_expression
  (α β γ δ p q : ℝ)
  (h1 : ∀ x, x^2 + p * x + 2 = 0 → (x = α ∨ x = β))
  (h2 : ∀ x, x^2 + q * x + 2 = 0 → (x = γ ∨ x = δ)) :
  problem_statement α β γ δ p q :=
by sorry

end NUMINAMATH_GPT_equivalent_expression_l1310_131058


namespace NUMINAMATH_GPT_solve_for_x_l1310_131065

theorem solve_for_x (x : ℚ) : (x + 4) / (x - 3) = (x - 2) / (x + 2) -> x = -2 / 11 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1310_131065


namespace NUMINAMATH_GPT_isosceles_triangle_sides_l1310_131016

theorem isosceles_triangle_sides (P : ℝ) (a b c : ℝ) (h₀ : P = 26) (h₁ : a = 11) (h₂ : a = b ∨ a = c)
  (h₃ : a + b + c = P) : 
  (b = 11 ∧ c = 4) ∨ (b = 7.5 ∧ c = 7.5) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_sides_l1310_131016


namespace NUMINAMATH_GPT_total_cost_proof_l1310_131098

-- Define the conditions
def length_grass_field : ℝ := 75
def width_grass_field : ℝ := 55
def width_path : ℝ := 2.5
def area_path : ℝ := 6750
def cost_per_sq_m : ℝ := 10

-- Calculate the outer dimensions
def outer_length : ℝ := length_grass_field + 2 * width_path
def outer_width : ℝ := width_grass_field + 2 * width_path

-- Calculate the area of the entire field including the path
def area_entire_field : ℝ := outer_length * outer_width

-- Calculate the area of the grass field without the path
def area_grass_field : ℝ := length_grass_field * width_grass_field

-- Calculate the area of the path
def area_calculated_path : ℝ := area_entire_field - area_grass_field

-- Calculate the total cost of constructing the path
noncomputable def total_cost : ℝ := area_calculated_path * cost_per_sq_m

-- The theorem to prove
theorem total_cost_proof :
  area_calculated_path = area_path ∧ total_cost = 6750 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_proof_l1310_131098


namespace NUMINAMATH_GPT_general_term_of_an_l1310_131090

theorem general_term_of_an (a : ℕ → ℕ) (h1 : a 1 = 1)
    (h_rec : ∀ n : ℕ, a (n + 1) = 2 * a n + 1) :
    ∀ n : ℕ, a n = 2^n - 1 :=
sorry

end NUMINAMATH_GPT_general_term_of_an_l1310_131090


namespace NUMINAMATH_GPT_largest_polygon_is_E_l1310_131037

def area (num_unit_squares num_right_triangles num_half_squares: ℕ) : ℚ :=
  num_unit_squares + num_right_triangles * 0.5 + num_half_squares * 0.25

def polygon_A_area := area 3 2 0
def polygon_B_area := area 4 1 0
def polygon_C_area := area 2 4 2
def polygon_D_area := area 5 0 0
def polygon_E_area := area 3 3 4

theorem largest_polygon_is_E :
  polygon_E_area > polygon_A_area ∧ 
  polygon_E_area > polygon_B_area ∧ 
  polygon_E_area > polygon_C_area ∧ 
  polygon_E_area > polygon_D_area :=
by
  sorry

end NUMINAMATH_GPT_largest_polygon_is_E_l1310_131037


namespace NUMINAMATH_GPT_complex_powers_sum_zero_l1310_131067

theorem complex_powers_sum_zero (i : ℂ) (h : i^2 = -1) : i^2023 + i^2024 + i^2025 + i^2026 = 0 :=
by
  sorry

end NUMINAMATH_GPT_complex_powers_sum_zero_l1310_131067


namespace NUMINAMATH_GPT_area_of_region_l1310_131027

theorem area_of_region :
  (∃ (x y: ℝ), x^2 + y^2 = 5 * |x - y| + 2 * |x + y|) → 
  (∃ (A : ℝ), A = 14.5 * Real.pi) :=
sorry

end NUMINAMATH_GPT_area_of_region_l1310_131027


namespace NUMINAMATH_GPT_eval_expr_at_2_l1310_131075

def expr (x : ℝ) : ℝ := (3 * x + 4)^2

theorem eval_expr_at_2 : expr 2 = 100 :=
by sorry

end NUMINAMATH_GPT_eval_expr_at_2_l1310_131075


namespace NUMINAMATH_GPT_smallest_common_multiple_of_8_and_6_l1310_131089

theorem smallest_common_multiple_of_8_and_6 : ∃ n : ℕ, n > 0 ∧ (8 ∣ n) ∧ (6 ∣ n) ∧ ∀ m : ℕ, (m > 0 ∧ (8 ∣ m) ∧ (6 ∣ m)) → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_common_multiple_of_8_and_6_l1310_131089
