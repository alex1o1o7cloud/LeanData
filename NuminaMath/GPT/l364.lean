import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Fraction
import Mathlib.Algebra.Geometry
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Ring.Basic
import Mathlib.AlgebraicGeometry
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Variance
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Combination
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Angle
import Mathlib.Data.Real.Basic
import Mathlib.Init.Data.Nat.Basic
import Mathlib.Logic.Basic
import Mathlib.NumberTheory.Basic
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Trigonometry.Basic

namespace probability_of_A_east_of_B_and_C_l364_364933

-- Define the facts about the triangle and the problem conditions
def triangle_ABC : Type := 
  {A B C : Point} 
  (angle_A_40 : angle A B C = 40)

-- Define the probability calculation
def probability_A_east_given_angle_40 
  (t : triangle_ABC) : ℚ :=
  7 / 18

-- The theorem statement
theorem probability_of_A_east_of_B_and_C 
  (t : triangle_ABC) : 
  probability_A_east_given_angle_40 t = 7 / 18 := 
  sorry

end probability_of_A_east_of_B_and_C_l364_364933


namespace shift_f_to_g_l364_364037

-- Definitions of the given functions
def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)
def g (x : ℝ) : ℝ := Real.sin (2 * x + 2 * Real.pi / 3)

-- Proposition to be proved
theorem shift_f_to_g :
  ∀ x : ℝ, g (x - Real.pi / 4) = f x := 
sorry

end shift_f_to_g_l364_364037


namespace num_real_numbers_l364_364430

theorem num_real_numbers (x : ℝ) : 
  (∃ n : ℕ, n ≤ 11 ∧ x = (123 - n^2)^2) ↔ 
  (∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ (x = (123 - s^2)^2) ∧ 
  (∃! x, ∃ k : ℕ, 0 ≤ k ∧ k ≤ 11 ∧ x = (123 - k^2)^2)) :=
by
  sorry

end num_real_numbers_l364_364430


namespace part_a_part_b_l364_364035

-- Definition of the function f and the condition it satisfies
variable (f : ℕ → ℕ)
variable (k n : ℕ)

theorem part_a (h1 : ∀ k n : ℕ, (k * f n) ≤ f (k * n) ∧ f (k * n) ≤ (k * f n) + k - 1)
  (a b : ℕ) :
  f a + f b ≤ f (a + b) ∧ f (a + b) ≤ f a + f b + 1 :=
by
  exact sorry  -- Proof to be supplied

theorem part_b (h1 : ∀ k n : ℕ, (k * f n) ≤ f (k * n) ∧ f (k * n) ≤ (k * f n) + k - 1)
  (h2 : ∀ n : ℕ, f (2007 * n) ≤ 2007 * f n + 200) :
  ∃ c : ℕ, f (2007 * c) = 2007 * f c :=
by
  exact sorry  -- Proof to be supplied

end part_a_part_b_l364_364035


namespace complex_expression_result_l364_364859

theorem complex_expression_result (z : ℂ) (hz : z = 1 - 2 * complex.I) : 
  z * conj z + z = 6 - 2 * complex.I := 
by 
  sorry

end complex_expression_result_l364_364859


namespace part_one_part_two_l364_364868

def f (x a : ℝ) : ℝ :=
  x^2 + a * (abs x) + x 

theorem part_one (x1 x2 a : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) :
  (1 / 2) * (f x1 a + f x2 a) ≥ f ((x1 + x2) / 2) a :=
sorry

theorem part_two (a : ℝ) (ha : 0 ≤ a) (x1 x2 : ℝ) :
  (1 / 2) * (f x1 a + f x2 a) ≥ f ((x1 + x2) / 2) a :=
sorry

end part_one_part_two_l364_364868


namespace symmetric_function_value_l364_364550

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2 * x * Real.exp (-x) else 0 -- we only define for x ≤ 1 as per the problem

theorem symmetric_function_value :
  (∀ x : ℝ, f (1 + x) = -f (1 - x)) →
  f (2 + 3 * Real.log 2) = 48 * Real.log 2 :=
by
  intro h
  sorry

end symmetric_function_value_l364_364550


namespace distinct_real_x_l364_364418

theorem distinct_real_x (x : ℝ) :
  (∃! x, ∃ (s : ℝ), s ∈ set.Icc 0 11 ∧ s = real.sqrt (123 - real.sqrt x)) =
  12 := sorry

end distinct_real_x_l364_364418


namespace matrix_power_B_l364_364014

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 1, 0],
  ![0, 0, 1],
  ![1, 0, 0]
]

theorem matrix_power_B :
  B ^ 150 = 1 :=
by sorry

end matrix_power_B_l364_364014


namespace greatest_odd_integer_x_l364_364653

theorem greatest_odd_integer_x (x : ℕ) (h1 : x % 2 = 1) (h2 : x^4 / x^2 < 50) : x ≤ 7 :=
sorry

end greatest_odd_integer_x_l364_364653


namespace number_of_real_solutions_l364_364424

theorem number_of_real_solutions : 
  set.countable {x : ℝ | ∃ n : ℕ, (sqrt (123 - sqrt x) = n)} = 12 :=
by
  sorry

end number_of_real_solutions_l364_364424


namespace composite_dice_product_probability_l364_364470

theorem composite_dice_product_probability :
  let outcomes := 6 ^ 4
  let non_composite_ways := 13
  let composite_probability := 1 - non_composite_ways / outcomes
  composite_probability = 1283 / 1296 :=
by
  sorry

end composite_dice_product_probability_l364_364470


namespace apple_count_l364_364187

-- Define constants for the problem
constant bags: ℕ → Prop
constant total_apples: ℕ
constant apples_in_bag1: ℕ := 12
constant apples_in_bag2: ℕ := 6

-- Define the conditions
axiom condition1 (n: ℕ) : bags n → (n = apples_in_bag1 ∨ n = apples_in_bag2)
axiom condition2 : 70 ≤ total_apples ∧ total_apples ≤ 80
axiom condition3 : total_apples % 6 = 0

-- State the proof problem
theorem apple_count :
  total_apples = 72 ∨ total_apples = 78 :=
sorry

end apple_count_l364_364187


namespace discount_percentage_is_10_l364_364061

def hamburger_meat : ℝ := 5.00
def box_of_crackers : ℝ := 3.50
def bags_of_frozen_vegetables_count : ℕ := 4
def price_per_bag_of_frozen_vegetables : ℝ := 2.00
def pack_of_cheese : ℝ := 3.50
def total_bill_with_discount : ℝ := 18.00
def total_cost_without_discount : ℝ := 
  hamburger_meat + box_of_crackers + (bags_of_frozen_vegetables_count * price_per_bag_of_frozen_vegetables) + pack_of_cheese
def discount_amount : ℝ := total_cost_without_discount - total_bill_with_discount
def percentage_of_discount : ℝ := (discount_amount / total_cost_without_discount) * 100

theorem discount_percentage_is_10 : percentage_of_discount = 10 := 
by sorry

end discount_percentage_is_10_l364_364061


namespace largest_prime_factor_of_3328_l364_364217

theorem largest_prime_factor_of_3328 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 3328 ∧ (∀ q : ℕ, Nat.Prime q ∧ q ∣ 3328 → q ≤ p) :=
by
  have h : 3328 = 2^8 * 13 := by norm_num
  use 13
  split
  · exact Nat.prime_13
  split
  · rw h
    exact dvd_mul_right _ _
  · intros q hq1 hq2
    rw h at hq2
    cases Nat.dvd_mul.mp hq2 with hq2 hq2
    · exact Nat.le_of_dvd (Nat.pos_pow_of_pos 8 zero_lt_two) hq2
    · exact Nat.le_of_eq (EuclideanDomain.gcd_eq_right hq1 hq2).symm

end largest_prime_factor_of_3328_l364_364217


namespace quantities_at_higher_profit_l364_364731

def sugar_total := 1600
def rice_total := 1200
def flour_total := 800

def sugar_profit_lower := 0.08
def sugar_profit_higher := 0.12
def sugar_profit_overall := 0.11

def rice_profit_lower := 0.10
def rice_profit_higher := 0.15
def rice_profit_overall := 0.13

def flour_profit_lower := 0.12
def flour_profit_higher := 0.18
def flour_profit_overall := 0.16

theorem quantities_at_higher_profit (S R F : ℕ) :
  sugar_profit_lower * (sugar_total - S) + sugar_profit_higher * S = sugar_profit_overall * sugar_total →
  rice_profit_lower * (rice_total - R) + rice_profit_higher * R = rice_profit_overall * rice_total →
  flour_profit_lower * (flour_total - F) + flour_profit_higher * F = flour_profit_overall * flour_total →
  S = 1200 ∧ R = 720 ∧ F = 533 :=
by
  intros h_s h_r h_f
  split
  -- Prove S = 1200
  sorry
  split
  -- Prove R = 720
  sorry
  -- Prove F = 533
  sorry

end quantities_at_higher_profit_l364_364731


namespace log_function_value_l364_364392

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x - 2)

theorem log_function_value 
  (a : ℝ) (h : f a 11 = 2) : f 3 5 = 1 := 
  sorry

end log_function_value_l364_364392


namespace brianne_savings_ratio_l364_364313

theorem brianne_savings_ratio
  (r : ℝ)
  (H1 : 10 * r^4 = 160) :
  r = 2 :=
by 
  sorry

end brianne_savings_ratio_l364_364313


namespace min_value_2x_plus_y_l364_364902

theorem min_value_2x_plus_y (x y : ℝ) (h1 : |y| ≤ 2 - x) (h2 : x ≥ -1) : 
  ∃ (x y : ℝ), |y| ≤ 2 - x ∧ x ≥ -1 ∧ (∀ y : ℝ, |y| ≤ 2 - x → x ≥ -1 → 2 * x + y ≥ -5) ∧ (2 * x + y = -5) :=
by
  sorry

end min_value_2x_plus_y_l364_364902


namespace count_distinct_x_l364_364435

theorem count_distinct_x :
  { x : ℝ | ∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ s = floor (sqrt (123 - sqrt x)) } = 12 :=
sorry

end count_distinct_x_l364_364435


namespace minimum_n_l364_364555

noncomputable def sequence (a : ℕ → ℝ) :=
  a 1 = 1 ∧ ∀ n ≥ 2, a n = 10 * a (n-1) - 1

theorem minimum_n (a : ℕ → ℝ) (h : sequence a) :
  ∃ n, n = 102 ∧ a n > 10^100 :=
sorry

end minimum_n_l364_364555


namespace find_angle_C_l364_364912

theorem find_angle_C
  (a b c : ℝ)
  (h : a^2 + b^2 - c^2 = √3 * a * b) :
  ∠C = real.arccos (√3 / 2) :=
by
  sorry

end find_angle_C_l364_364912


namespace equal_segments_of_sides_l364_364195

variables {O A : Point} {B1 B2 C1 C2 : Point}
-- Define the necessary geometric properties and conditions
def angle_bisector (O A B1 B2 C1 C2 : Point) : Prop :=
  -- Conditions:
  -- Two circles passing through O and A (Definition of the circles),
  -- Intersection points on sides of the angle such that A lies on the angle bisector.
  -- Angle Bisector condition ensuring symmetry needed for the congruence proof.
  sorry

theorem equal_segments_of_sides (h : angle_bisector O A B1 B2 C1 C2) : 
  dist B1 C1 = dist B2 C2 :=
sorry

end equal_segments_of_sides_l364_364195


namespace tiffany_max_points_l364_364192

section
  variables
  (initial_money : ℕ := 3)
  (cost_per_game : ℕ := 1)
  (rings_per_game : ℕ := 5)
  (points_red_bucket : ℕ := 2)
  (points_green_bucket : ℕ := 3)
  (points_miss : ℕ := 0)
  (games_played : ℕ := 2)
  (red_buckets : ℕ := 4)
  (green_buckets : ℕ := 5)
  (additional_games : ℕ := initial_money - games_played)
  (points_per_game_from_green_buckets : ℕ := rings_per_game * points_green_bucket)
  (total_points : ℕ := (red_buckets * points_red_bucket) + (green_buckets * points_green_bucket) + (additional_games * points_per_game_from_green_buckets))

  theorem tiffany_max_points : total_points = 38 := 
  sorry
end

end tiffany_max_points_l364_364192


namespace abc_inequality_l364_364843

variable (a b c : ℝ)

noncomputable def cond := (1/a + 1/b + 1/c >= a + b + c) 

theorem abc_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : cond a b c) : 
  a + b + c >= 3 * a * b * c := 
sorry

end abc_inequality_l364_364843


namespace find_incorrect_conclusion_l364_364683

-- Define the conditions as given in the problem
variables {R : Type*} [linear_ordered_field R]
variables (a b c t : R)
variable (h_nonzero : a ≠ 0)
variable (h1 : a - b + c = 0)
variable (h2 : a + b + c = 4)
variable (h3 : c = 3)

-- Define the main hypothesis that must be checked
def incorrect_conclusion : Prop :=
  let p := 3 + (-x ^ 2 + 2 * x + 3) in
  ∀ (x : R), 2 * x + p > x^2

-- State the proof problem
theorem find_incorrect_conclusion : (¬ incorrect_conclusion) :=
sorry

end find_incorrect_conclusion_l364_364683


namespace permutations_count_l364_364753

def satisfies_conditions (a : List ℕ) : Prop :=
  a[0] < a[1] ∧ a[1] > a[2] ∧ a[2] < a[3] ∧ a[3] > a[4]

theorem permutations_count : 
  list.filter satisfies_conditions (list.permutations [1,2,3,4,5]).length = 16 :=
sorry

end permutations_count_l364_364753


namespace x_minus_25_is_perfect_square_l364_364394

theorem x_minus_25_is_perfect_square (n : ℕ) : 
  let x := 10^(2*n + 4) + 10^(n + 3) + 50 in
  ∃ k : ℕ, x - 25 = k^2 := by
  sorry

end x_minus_25_is_perfect_square_l364_364394


namespace nine_point_circle_l364_364984

variable {A B C : Type}
variable [EuclideanPlane A B C]

-- Definitions as given in the conditions
def circumcenter (ABC : Triangle A B C) : Point A B C := sorry
def orthocenter (ABC : Triangle A B C) : Point A B C := sorry
def nine_point_circle_center (ABC : Triangle A B C) : Point A B C := midpoint (circumcenter ABC) (orthocenter ABC)

-- Given nine points: midpoints of sides, feet of the altitudes, midpoints from orthocenter to vertices
def midpoints_of_sides (ABC : Triangle A B C) : List (Point A B C) := sorry
def feet_of_altitudes (ABC : Triangle A B C) : List (Point A B C) := sorry
def midpoints_orthocenter_vertices (ABC : Triangle A B C) : List (Point A B C) := sorry

-- The proof problem statement
theorem nine_point_circle (ABC : Triangle A B C) :
  ∀ p ∈ (midpoints_of_sides ABC ++ feet_of_altitudes ABC ++ midpoints_orthocenter_vertices ABC), 
  distance p (nine_point_circle_center ABC) = (circumradius ABC) / 2 := 
sorry

end nine_point_circle_l364_364984


namespace sin_expansion_constants_l364_364645

theorem sin_expansion_constants :
  ∀ (θ : ℝ), 
    let b1 := 5 / 8,
        b2 := 0,
        b3 := -5 / 16,
        b4 := 0,
        b5 := 1 / 16 in
    (sin θ)^5 = b1 * sin θ + b2 * sin (2 * θ) + b3 * sin (3 * θ) + b4 * sin (4 * θ) + b5 * sin (5 * θ) 
    ∧ (b1^2 + b2^2 + b3^2 + b4^2 + b5^2 = 63 / 128) :=
by
  intro θ
  let b1 := 5 / 8
  let b2 := 0
  let b3 := -5 / 16
  let b4 := 0
  let b5 := 1 / 16
  have h1 : (sin θ)^5 = b1 * sin θ + b2 * sin (2 * θ) + b3 * sin (3 * θ) + b4 * sin (4 * θ) + b5 * sin (5 * θ) := sorry
  have h2 : b1^2 + b2^2 + b3^2 + b4^2 + b5^2 = 63 / 128 := sorry
  exact And.intro h1 h2

end sin_expansion_constants_l364_364645


namespace domain_of_sqrt_x_minus_2_l364_364334

theorem domain_of_sqrt_x_minus_2 :
  {x : ℝ | ∃ y : ℝ, y = real.sqrt (x - 2)} = {x : ℝ | x ≥ 2} :=
sorry

end domain_of_sqrt_x_minus_2_l364_364334


namespace intersecting_chords_second_length_l364_364255

theorem intersecting_chords_second_length (a b : ℕ) (k : ℕ) 
  (h_a : a = 12) (h_b : b = 18) (h_ratio : k ^ 2 = (a * b) / 24) 
  (x y : ℕ) (h_x : x = 3 * k) (h_y : y = 8 * k) :
  x + y = 33 :=
by
  sorry

end intersecting_chords_second_length_l364_364255


namespace max_number_of_different_ages_l364_364594

theorem max_number_of_different_ages
  (a : ℤ) (s : ℤ)
  (h1 : a = 31)
  (h2 : s = 5) :
  ∃ n : ℕ, n = (36 - 26 + 1) :=
by sorry

end max_number_of_different_ages_l364_364594


namespace probability_of_composite_l364_364491

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ m < n ∧ 1 < k ∧ k < n ∧ m * k = n

def dice_outcomes (faces : ℕ) (rolls : ℕ) : ℕ :=
  faces ^ rolls

def non_composite_product_ways : ℕ :=
  1 + (3 * 4)  -- one way for all 1s, plus combinations of (1,1,1,{2,3,5})

def total_outcomes : ℕ :=
  dice_outcomes 6 4  -- 6^4 total possible outcomes

def probability_composite : ℚ :=
  1 - (non_composite_product_ways / total_outcomes)

theorem probability_of_composite:
  probability_composite = 1283 / 1296 := 
by
  sorry

end probability_of_composite_l364_364491


namespace min_pieces_to_get_special_l364_364246

theorem min_pieces_to_get_special
  (P K : ℕ → ℕ × ℕ)
  (special : ℕ × ℕ)
  (h_8x8 : ∀ i, P i ∈ { (x, y) : ℕ × ℕ | x < 8 ∧ y < 8 })
  (h_3x3 : ∀ x y, (∀ i, P i ∈ {(x, y) | x <= x + 2 ∧ y <= y + 2} → card P ≤ 1))
  (h_6x6 : ∀ x y, (∀ i, P i ∈ {(x, y) | x <= x + 5 ∧ y <= y + 5} → card P ≥ 2))
  : (∃ S : finset (ℕ × ℕ), card S = 5 ∧ special ∈ S) := sorry

end min_pieces_to_get_special_l364_364246


namespace dice_product_composite_probability_l364_364480

theorem dice_product_composite_probability :
  (let total_outcomes := (6 : ℕ)^4,
       non_composite_outcomes := 1,
       composite_probability := 1 - (non_composite_outcomes / total_outcomes) in
   composite_probability = 1295 / 1296) :=
by sorry

end dice_product_composite_probability_l364_364480


namespace cone_height_l364_364725

noncomputable def height_of_cone (r : ℝ) (n : ℕ) : ℝ :=
  let sector_circumference := (2 * Real.pi * r) / n
  let cone_base_radius := sector_circumference / (2 * Real.pi)
  Real.sqrt (r^2 - cone_base_radius^2)

theorem cone_height
  (r_original : ℝ)
  (n : ℕ)
  (h : r_original = 10)
  (hc : n = 4) :
  height_of_cone r_original n = 5 * Real.sqrt 3 := by
  sorry

end cone_height_l364_364725


namespace kolya_speed_increase_factor_l364_364114

theorem kolya_speed_increase_factor
  (N : ℕ)
  (x : ℕ) 
  (kolya_solves : ℕ)
  (seryozha_solves : ℕ)
  (seryozha_remaining : ℕ)
  (kolya_remaining : ℕ)
  (total_remaining_time : ℕ)
  (current_time_used : ℕ)
  (kolya_current_rate : ℕ)
  (kolya_required_rate : ℕ):
  (seryozha_remaining = x / 2) →
  (kolya_solves = seryozha_remaining / 3) →
  (N = x + seryozha_remaining) →
  (N = 3 * x / 2) →
  (kolya_remaining = N - kolya_solves) →
  (kolya_remaining = 4 * x / 3) →
  (total_remaining_time = current_time_used / 2) →
  (kolya_current_rate = kolya_solves / current_time_used) →
  (kolya_required_rate = kolya_remaining / total_remaining_time) →
  (required_factor = kolya_required_rate / kolya_current_rate) →
  required_factor = 16 := 
begin
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10,
  sorry
end

end kolya_speed_increase_factor_l364_364114


namespace least_amount_of_money_l364_364757

variable (Money : Type) [LinearOrder Money]
variable (Anne Bo Coe Dan El : Money)

-- Conditions from the problem
axiom anne_less_than_bo : Anne < Bo
axiom dan_less_than_bo : Dan < Bo
axiom coe_less_than_anne : Coe < Anne
axiom coe_less_than_el : Coe < El
axiom coe_less_than_dan : Coe < Dan
axiom dan_less_than_anne : Dan < Anne

theorem least_amount_of_money : (∀ x, x = Anne ∨ x = Bo ∨ x = Coe ∨ x = Dan ∨ x = El → Coe < x) :=
by
  sorry

end least_amount_of_money_l364_364757


namespace sum_odd_minus_even_l364_364678

theorem sum_odd_minus_even :
  (∑ i in Finset.range 1013, (2 * i + 1) - ∑ i in Finset.range 1012, (2 * (i + 1))) = 1013 := 
by
  sorry

end sum_odd_minus_even_l364_364678


namespace rice_purchase_new_weight_l364_364560

theorem rice_purchase_new_weight (x : ℝ) (M : ℝ) :
  let new_price := 0.75 * x in
  let original_weight := 30 in
  M = original_weight * x →
  ∃ Q : ℝ, M = Q * new_price ∧ Q = 40 :=
by
  intros
  use 40
  sorry

end rice_purchase_new_weight_l364_364560


namespace janice_remaining_hours_l364_364005

def homework_time : ℕ := 30
def clean_room_time : ℕ := homework_time / 2
def walk_dog_time : ℕ := homework_time + 5
def trash_time : ℕ := homework_time / 6
def total_task_time : ℕ := homework_time + clean_room_time + walk_dog_time + trash_time
def remaining_minutes : ℕ := 35

theorem janice_remaining_hours : (remaining_minutes : ℚ) / 60 = (7 / 12 : ℚ) :=
by
  sorry

end janice_remaining_hours_l364_364005


namespace range_of_b_l364_364383

def f (a x : ℝ) := (a^x - a^(-x)) / (a - 1)
def g (b x : ℝ) := Real.log x - b*x + 1

-- The proof problem
theorem range_of_b (a : ℝ) (b : ℝ) :
  (a > 0) ∧ (a ≠ 1) → 
  ((∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → a * f a x ≤ 2 * b * (a + 1)) 
  ∨ (∃ x, g b x = 0))
  ∧ ¬((∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → a * f a x ≤ 2 * b * (a + 1)) 
  ∧ (∃ x, g b x = 0)) → 
  b ∈ Set.Ioo (-∞ : ℝ) (1 / 2) ∪ Set.Ioo (1 : ℝ) ∞ :=
sorry

end range_of_b_l364_364383


namespace segment_parallel_to_x_axis_l364_364050

theorem segment_parallel_to_x_axis 
  (f : ℤ → ℤ) 
  (hf : ∀ n, ∃ m, f n = m) 
  (a b : ℤ) 
  (h_dist : ∃ d : ℤ, d * d = (b - a) * (b - a) + (f b - f a) * (f b - f a)) : 
  f a = f b :=
sorry

end segment_parallel_to_x_axis_l364_364050


namespace composite_dice_product_probability_l364_364471

theorem composite_dice_product_probability :
  let outcomes := 6 ^ 4
  let non_composite_ways := 13
  let composite_probability := 1 - non_composite_ways / outcomes
  composite_probability = 1283 / 1296 :=
by
  sorry

end composite_dice_product_probability_l364_364471


namespace largest_prime_factor_3328_l364_364219

theorem largest_prime_factor_3328 : ∃ p : ℕ, p.prime ∧ p ∣ 3328 ∧ ∀ q : ℕ, q.prime ∧ q ∣ 3328 → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_13, },  -- 13 is prime
  split,
  { exact dvd.intro (2^8) rfl, },  -- 13 divides 3328
  { 
    intros q hq_prime hq_dvd, 
    sorry,  -- it remains to show q ≤ 13 for all prime q dividing 3328
  },
end

end largest_prime_factor_3328_l364_364219


namespace find_other_number_l364_364081

theorem find_other_number (HCF LCM one_number other_number : ℤ)
  (hHCF : HCF = 12)
  (hLCM : LCM = 396)
  (hone_number : one_number = 48)
  (hrelation : HCF * LCM = one_number * other_number) :
  other_number = 99 :=
by
  sorry

end find_other_number_l364_364081


namespace true_proposition_in_problem_l364_364621

theorem true_proposition_in_problem :
  let F1 F2 : Type := ℝ
  let M : Type := ℝ
  let | F1 F2 | := 6
  let | M F1 | + | M F2 | := 6
  let A B C : ℝ
  let ∠ B := 60
  let ∠ A := (2 * ∠ B - ∠ C)
  let x : ℕ
  let x := 0
  let x_ge_0 := x ≥ 0
  let a b c : ℝ
  let l1 := λ x y, a * x + 3 * y - 1 = 0
  let l2 := λ x y, x + b * y + 1 = 0
  let a_div_b_equal_neg3 := a / b = -3
  ∀ (v1 v2 v3 : ℝ), linear_independent ℝ [v1, v2, v3] → linear_independent ℝ [v1 + v2, v1 - v2, v3] :=
by
  intros
  exact sorry

end true_proposition_in_problem_l364_364621


namespace Johann_oranges_l364_364538

-- Define the given conditions
def initial_oranges := 60
def eaten_oranges := 10
def half_remaining_oranges := (initial_oranges - eaten_oranges) / 2
def returned_oranges := 5

-- Define the statement to prove
theorem Johann_oranges :
  initial_oranges - eaten_oranges - half_remaining_oranges + returned_oranges = 30 := by
  sorry

end Johann_oranges_l364_364538


namespace max_sum_permutations_eval_max_permutations_result_l364_364547

theorem max_sum_permutations {s : Finset (Fin 5)} (xs : ∀ i, i ∈ s → ℕ) : 
  let e := xs 0 * xs 1 + xs 1 * xs 2 + xs 2 * xs 3 + xs 3 * xs 4 + xs 4 * xs 0 
  in e ≤ 48 ∧ 
  (∃ (p : Perm (Fin 5)) (h : ∀ i ∈ s, p i ∈ s), (xs i ∈ s → xs i * xs (i + 1) ≤ 48)) :=
by
  sorry

theorem eval_max_permutations_result : 
  let M := 48
  let N := 10
  in M + N = 58 :=
by
  sorry

end max_sum_permutations_eval_max_permutations_result_l364_364547


namespace usable_parking_lot_percentage_l364_364004

theorem usable_parking_lot_percentage
  (length width : ℝ) (area_per_car : ℝ) (number_of_cars : ℝ)
  (h_len : length = 400)
  (h_wid : width = 500)
  (h_area_car : area_per_car = 10)
  (h_cars : number_of_cars = 16000) :
  ((number_of_cars * area_per_car) / (length * width) * 100) = 80 := 
by
  -- Proof omitted
  sorry

end usable_parking_lot_percentage_l364_364004


namespace total_cost_of_pets_is_130_l364_364200
noncomputable theory

def cost_of_pets :=
  let parakeet_cost := 10
  let puppy_cost := 3 * parakeet_cost
  let kitten_cost := 2 * parakeet_cost
  let total_cost := 2 * puppy_cost + 2 * kitten_cost + 3 * parakeet_cost
  total_cost

theorem total_cost_of_pets_is_130 : cost_of_pets = 130 :=
by
  -- Showing that cost_of_pets indeed evaluates to 130
  let parakeet_cost := 10
  let puppy_cost := 3 * parakeet_cost
  let kitten_cost := 2 * parakeet_cost
  let total_cost := 2 * puppy_cost + 2 * kitten_cost + 3 * parakeet_cost
  show total_cost = 130 from sorry

end total_cost_of_pets_is_130_l364_364200


namespace number_of_gcd_values_l364_364688

theorem number_of_gcd_values (a b : ℤ) (h : a * b = 180) : 
  {d : ℤ | d = Int.gcd a b}.finite.toFinset.card = 8 := 
sorry

end number_of_gcd_values_l364_364688


namespace julie_reimburses_sarah_l364_364578

theorem julie_reimburses_sarah : 
  ∀ (dollars_in_cents lollipops total_cost_sarah share_fraction),
    dollars_in_cents = 100 →
    lollipops = 12 →
    total_cost_sarah = 300 →
    share_fraction = 1 / 4 →
    ((total_cost_sarah / lollipops) * (lollipops * share_fraction)) = 75 :=
by
  intros dollars_in_cents lollipops total_cost_sarah share_fraction h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end julie_reimburses_sarah_l364_364578


namespace roots_polynomial_sum_cubes_l364_364960

theorem roots_polynomial_sum_cubes (u v w : ℂ) (h : (∀ x, (x = u ∨ x = v ∨ x = w) → 5 * x ^ 3 + 500 * x + 1005 = 0)) :
  (u + v) ^ 3 + (v + w) ^ 3 + (w + u) ^ 3 = 603 := sorry

end roots_polynomial_sum_cubes_l364_364960


namespace z_squared_z_cubed_z_n_l364_364379

namespace ComplexProof

def z (θ : ℝ) : ℂ := Complex.ofReal (Real.cos θ) + Complex.I * Complex.ofReal (Real.sin θ)

theorem z_squared (θ : ℝ) : z θ ^ 2 = Complex.ofReal (Real.cos (2 * θ)) + Complex.I * Complex.ofReal (Real.sin (2 * θ)) :=
by sorry

theorem z_cubed (θ : ℝ) : z θ ^ 3 = Complex.ofReal (Real.cos (3 * θ)) + Complex.I * Complex.ofReal (Real.sin (3 * θ)) :=
by sorry

theorem z_n (θ : ℝ) (n : ℕ) : z θ ^ n = Complex.ofReal (Real.cos (n * θ)) + Complex.I * Complex.ofReal (Real.sin (n * θ)) :=
by sorry

end ComplexProof

end z_squared_z_cubed_z_n_l364_364379


namespace total_apples_l364_364161

theorem total_apples (n : ℕ) (h1 : ∀ k : ℕ, 6 * k = n ∨ 12 * k = n) (h2 : 70 ≤ n) (h3 : n ≤ 80) : 
  n = 72 ∨ n = 78 := 
sorry

end total_apples_l364_364161


namespace factorial_divisibility_l364_364897

theorem factorial_divisibility
  (n p : ℕ)
  (h1 : p > 0)
  (h2 : n ≤ p + 1) :
  (factorial (p^2)) ∣ (factorial p)^(p + 1) :=
sorry

end factorial_divisibility_l364_364897


namespace valid_number_of_apples_l364_364140

theorem valid_number_of_apples (n : ℕ) : 
  (∃ k m : ℕ, n = 12 * k + 6 * m) ∧ (70 ≤ n ∧ n ≤ 80) ↔ (n = 72 ∨ n = 78) :=
by
  sorry

end valid_number_of_apples_l364_364140


namespace gcd_lcm_product_180_l364_364694

theorem gcd_lcm_product_180 :
  ∃ (a b : ℕ), (gcd a b) * (lcm a b) = 180 ∧ 
  let possible_gcd_values := 
    {d | ∃ a b : ℕ, gcd a b = d ∧ (gcd a b) * (lcm a b) = 180} in
  possible_gcd_values.card = 7 :=
by
  sorry

end gcd_lcm_product_180_l364_364694


namespace apples_total_l364_364172

theorem apples_total (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apples_total_l364_364172


namespace complex_magnitude_ratio_eq_sqrt133_div_7_l364_364839

noncomputable def complex_magnitude_ratio (z1 z2 : ℂ) (h1 : |z1| = 2) (h2 : |z2| = 3) (theta : ℝ) (h_theta : theta = real.pi / 3) : ℝ :=
  |z1 + z2| / |z1 - z2|

theorem complex_magnitude_ratio_eq_sqrt133_div_7 
  (z1 z2 : ℂ) 
  (h1 : |z1| = 2) 
  (h2 : |z2| = 3) 
  (theta : ℝ) 
  (h_theta : theta = real.pi / 3) 
: complex_magnitude_ratio z1 z2 h1 h2 theta h_theta = real.sqrt 133 / 7 := 
sorry

end complex_magnitude_ratio_eq_sqrt133_div_7_l364_364839


namespace probability_at_least_three_heads_l364_364080

theorem probability_at_least_three_heads :
  let outcomes := Finset.powerset (Finset.range 5)
  let favorable := outcomes.filter (λ s, s.card ≥ 3)
  (favorable.card : ℚ) / outcomes.card = 1 / 2 :=
by
  sorry

end probability_at_least_three_heads_l364_364080


namespace correct_option_D_l364_364823

variable {Line : Type} [IsLinearSpace Line]
variable {Plane : Type} [IsPlanarSpace Plane]

variable (m n : Line)
variable (α β γ : Plane)

-- Assumptions
def is_parallel (x y : Plane) : Prop := sorry -- Definition of parallelism (planes)
def line_intersect_plane (l : Line) (p : Plane) : Set Point := sorry -- Intersection of line and plane
def plane_intersect_plane (p q : Plane) : Set Line := sorry -- Intersection of planes
def subset_line_of_plane (l : Line) (p : Plane) : Prop := sorry -- Line is subset of plane
def distinct_lines (l1 l2 : Line) : Prop := l1 ≠ l2
def distinct_planes (p1 p2 p3 : Plane) : Prop := p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3

-- Problem Statement
theorem correct_option_D 
  (h1 : distinct_lines m n) 
  (h2 : distinct_planes α β γ)
  (h3 : is_parallel α β)
  (h4 : plane_intersect_plane γ α = {m})
  (h5 : plane_intersect_plane γ β = {n}) : 
  is_parallel m n := 
sorry

end correct_option_D_l364_364823


namespace pyramid_properties_l364_364614

-- Define the edge length of the cube
variable (a : ℝ)

-- Define the volume of the cube
def volume_cube (a : ℝ) : ℝ := a^3

-- Define the volume of one pyramid
def volume_pyramid (a : ℝ) : ℝ := (volume_cube a) / 6

-- Define the surface area of one pyramid
def surface_area_pyramid (a : ℝ) : ℝ := a^2 * (1 + Real.sqrt 2)

-- The theorem to be proven
theorem pyramid_properties (a : ℝ) :
  (volume_pyramid a = a^3 / 6) ∧ (surface_area_pyramid a = a^2 * (1 + Real.sqrt 2)) :=
by
  sorry

end pyramid_properties_l364_364614


namespace christine_savings_l364_364323

def commission_rate : ℝ := 0.12
def total_sales : ℝ := 24000
def personal_needs_percentage : ℝ := 0.60
def savings_percentage : ℝ := 1 - personal_needs_percentage

noncomputable def commission_earned : ℝ := total_sales * commission_rate
noncomputable def amount_saved : ℝ := commission_earned * savings_percentage

theorem christine_savings :
  amount_saved = 1152 :=
by
  sorry

end christine_savings_l364_364323


namespace percentage_of_female_officers_on_duty_l364_364570

theorem percentage_of_female_officers_on_duty
    (on_duty : ℕ) (half_on_duty_female : on_duty / 2 = 100)
    (total_female_officers : ℕ)
    (total_female_officers_value : total_female_officers = 1000)
    : (100 / total_female_officers : ℝ) * 100 = 10 :=
by sorry

end percentage_of_female_officers_on_duty_l364_364570


namespace idiom_describes_random_event_l364_364228

-- Define the idioms as propositions.
def FishingForMoonInWater : Prop := ∀ (x : Type), x -> False
def CastlesInTheAir : Prop := ∀ (y : Type), y -> False
def WaitingByStumpForHare : Prop := ∃ (z : Type), True
def CatchingTurtleInJar : Prop := ∀ (w : Type), w -> False

-- Define the main theorem to state that WaitingByStumpForHare describes a random event.
theorem idiom_describes_random_event : WaitingByStumpForHare :=
  sorry

end idiom_describes_random_event_l364_364228


namespace greatest_three_digit_multiple_23_l364_364656

theorem greatest_three_digit_multiple_23 : 
  ∃ n : ℕ, n < 1000 ∧ n % 23 = 0 ∧ (∀ m : ℕ, m < 1000 ∧ m % 23 = 0 → m ≤ n) ∧ n = 989 :=
sorry

end greatest_three_digit_multiple_23_l364_364656


namespace probability_composite_l364_364495

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def non_composite_outcomes : ℕ := 1 + 12

theorem probability_composite :
  let total_outcomes := 6^4 in
  let non_composite := non_composite_outcomes in
  let composite_probability := (total_outcomes - non_composite) / total_outcomes.toRat in
  composite_probability = 1283 / 1296 := by
  let total_outcomes := 6^4
  let non_composite := 1 + 12
  let composite_probability := (total_outcomes - non_composite).toRat / total_outcomes.toRat
  sorry

end probability_composite_l364_364495


namespace greatest_three_digit_multiple_of_23_l364_364666

theorem greatest_three_digit_multiple_of_23 : ∃ n : ℕ, n % 23 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℕ, m % 23 = 0 ∧ 100 ≤ m ∧ m ≤ 999 → m ≤ n := 
by
  use 989
  split
  · -- 989 is a multiple of 23
    exact (by norm_num : 989 % 23 = 0)
  · split
    · -- 989 is at least 100
      exact (by norm_num : 100 ≤ 989)
    · split
      · -- 989 is at most 999
        exact (by norm_num : 989 ≤ 999)
      · -- 989 is the greatest such number within the range
        sorry

end greatest_three_digit_multiple_of_23_l364_364666


namespace center_of_image_circle_l364_364611

def point := ℝ × ℝ

def reflect_about_y_eq_neg_x (p : point) : point :=
  let (a, b) := p
  (-b, -a)

theorem center_of_image_circle :
  reflect_about_y_eq_neg_x (8, -3) = (3, -8) :=
by
  sorry

end center_of_image_circle_l364_364611


namespace swimming_pool_visits_l364_364791

theorem swimming_pool_visits (n : ℕ) :
  (∀ (x y : ℕ), 0 ≤ x ∧ x ≤ 450 ∧ 0 ≤ y ∧ y ≤ 450 ∧ (6 * 75 = 450) ) ∧
  (∀ (days_visits : ℕ → ℕ), (∀ d, days_visits d ≤ 6) ∧ (∀ d, days_visits d ≥ 0) ∧ (∑ d in finset.range 100, days_visits d = 450) ) ∧
  (∑ d in finset.range 100, (ite (days_visits d ≥ 5) 1 0)) = n
  → (n = 90 ∨ n = 25) :=
by
  sorry

end swimming_pool_visits_l364_364791


namespace find_angle_EAD_l364_364553

noncomputable theory

open EuclideanGeometry

def points_collinear (pts : List (EuclideanGeometry.Point ℝ)) : Prop :=
  ∃ line : EuclideanGeometry.Line ℝ, ∀ p ∈ pts, p ∈ line

def equal_segments (points : List (EuclideanGeometry.Point ℝ)) : Prop :=
  match points with
  | []        => true
  | [_]       => true
  | p :: ps   =>
      let q := ps.head
      let len := EuclideanGeometry.distance p q
      List.all (List.pairs ps) (λ (a, b) => EuclideanGeometry.distance a b = len)

theorem find_angle_EAD (A B C D E F G : EuclideanGeometry.Point ℝ)
  (h1 : EuclideanGeometry.distance A B = EuclideanGeometry.distance B C)
  (h2 : EuclideanGeometry.distance B C = EuclideanGeometry.distance C D)
  (h3 : EuclideanGeometry.distance C D = EuclideanGeometry.distance D E)
  (h4 : EuclideanGeometry.distance D E = EuclideanGeometry.distance E F)
  (h5 : EuclideanGeometry.distance E F = EuclideanGeometry.distance F G)
  (h6 : EuclideanGeometry.distance F G = EuclideanGeometry.distance G A)
  (h7 : points_collinear [A, B, F, D])
  (h8 : points_collinear [A, G, C, E]) :
  ∠ E A D = π / 3 :=
sorry

end find_angle_EAD_l364_364553


namespace rectangle_cut_dimensions_l364_364739

-- Define the original dimensions of the rectangle as constants.
def original_length : ℕ := 12
def original_height : ℕ := 6

-- Define the dimensions of the new rectangle after slicing parallel to the longer side.
def new_length := original_length / 2
def new_height := original_height

-- The theorem statement.
theorem rectangle_cut_dimensions :
  new_length = 6 ∧ new_height = 6 :=
by
  sorry

end rectangle_cut_dimensions_l364_364739


namespace original_weight_of_potatoes_l364_364721

theorem original_weight_of_potatoes (W : ℝ) (h : W / (W / 2) = 36) : W = 648 := by
  sorry

end original_weight_of_potatoes_l364_364721


namespace find_a_l364_364094

noncomputable def f : ℝ → ℝ
| x => if x < 2 then 3^(x-2) else log 3 (x^2 - 1)

theorem find_a (a : ℝ) : f a = 1 ↔ a = 2 :=
by
  sorry

end find_a_l364_364094


namespace density_of_cone_in_mercury_l364_364282

variable {h : ℝ} -- height of the cone
variable {ρ : ℝ} -- density of the cone
variable {ρ_m : ℝ} -- density of the mercury
variable {k : ℝ} -- proportion factor

-- Archimedes' principle applied to the cone floating in mercury
theorem density_of_cone_in_mercury (stable_eq: ∀ (V V_sub: ℝ), (ρ * V) = (ρ_m * V_sub))
(h_sub: h / k = (k - 1) / k) :
  ρ = ρ_m * ((k - 1)^3 / k^3) :=
by
  sorry

end density_of_cone_in_mercury_l364_364282


namespace intersection_circle_radius_squared_l364_364633

noncomputable def parabola1 (x : ℝ) : ℝ := (x - 1) ^ 2
noncomputable def parabola2 (y : ℝ) : ℝ := (y + 1) ^ 2 + 2

theorem intersection_circle_radius_squared : 
  let x y := (∃ x y, y = parabola1 x ∧ x - 2 = parabola2 y)
  ((x - 1/2)^2 + (y + 1)^2 = 1 / 4) :=
by sorry

end intersection_circle_radius_squared_l364_364633


namespace terminal_side_in_first_quadrant_l364_364903

theorem terminal_side_in_first_quadrant (α : ℝ) (k : ℤ) (h : k * 360 + 270 < α ∧ α < k * 360 + 360) :
  ∃ n : ℤ, n * 360 - 360 < -α ∧ -α < n * 360 - 270 :=
begin
  sorry
end

end terminal_side_in_first_quadrant_l364_364903


namespace move_line_up_l364_364520

theorem move_line_up (a : ℝ) : (∀ x : ℝ, y = -2 * x + a ↔ y = -2 * x + 5) :=
by assume x
   have h₁ : y = -2 * x + a := rfl
   have h₂ : a = 5 := sorry
   rw h₂ at h₁
   exact ⟨λ h, h₁, λ h, h₁⟩

end move_line_up_l364_364520


namespace rational_numbers_inequality_l364_364330

theorem rational_numbers_inequality (f : ℚ → ℤ) :
  ∃ a b : ℚ, (f(a) + f(b)) / 2 ≤ f((a + b) / 2) :=
sorry

end rational_numbers_inequality_l364_364330


namespace possible_apple_counts_l364_364127

theorem possible_apple_counts (n : ℕ) (h₁ : 70 ≤ n) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end possible_apple_counts_l364_364127


namespace A_inter_B_eq_A_union_C_U_B_eq_l364_364967

section
  -- Define the universal set U
  def U : Set ℝ := { x | x^2 - (5 / 2) * x + 1 ≥ 0 }

  -- Define set A
  def A : Set ℝ := { x | |x - 1| > 1 }

  -- Define set B
  def B : Set ℝ := { x | (x + 1) / (x - 2) ≥ 0 }

  -- Define the complement of B in U
  def C_U_B : Set ℝ := U \ B

  -- Theorem for A ∩ B
  theorem A_inter_B_eq : A ∩ B = { x | x ≤ -1 ∨ x > 2 } := sorry

  -- Theorem for A ∪ (C_U_B)
  theorem A_union_C_U_B_eq : A ∪ C_U_B = U := sorry
end

end A_inter_B_eq_A_union_C_U_B_eq_l364_364967


namespace number_of_real_solutions_l364_364420

theorem number_of_real_solutions : 
  set.countable {x : ℝ | ∃ n : ℕ, (sqrt (123 - sqrt x) = n)} = 12 :=
by
  sorry

end number_of_real_solutions_l364_364420


namespace perpendicular_lines_and_planes_l364_364846

variables {m n : ℝ^3} {α β : ℝ^3}

-- Definitions of the conditions
def parallel (x y : ℝ^3) : Prop := ∃ k : ℝ, x = k • y
def perpendicular (x y : ℝ^3) : Prop := dot_product x y = 0

-- Problem statement in Lean
theorem perpendicular_lines_and_planes
(m_parallel_n : parallel m n)
(alpha_perpendicular_beta : perpendicular α β)
(m_perpendicular_alpha : perpendicular m α) :
perpendicular n α :=
sorry

end perpendicular_lines_and_planes_l364_364846


namespace tiffany_max_points_l364_364190

theorem tiffany_max_points : 
  let initial_money := 3
  let cost_per_game := 1
  let points_red_bucket := 2
  let points_green_bucket := 3
  let rings_per_game := 5
  let games_played := 2
  let red_buckets_first_two_games := 4
  let green_buckets_first_two_games := 5
  let remaining_money := initial_money - games_played * cost_per_game
  let remaining_games := remaining_money / cost_per_game
  let points_first_two_games := red_buckets_first_two_games * points_red_bucket + green_buckets_first_two_games * points_green_bucket
  let max_points_third_game := rings_per_game * points_green_bucket
  points_first_two_games + max_points_third_game = 38 := 
by
  sorry

end tiffany_max_points_l364_364190


namespace apples_total_l364_364170

theorem apples_total (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apples_total_l364_364170


namespace abs_a_plus_b_plus_c_l364_364021

open Complex

theorem abs_a_plus_b_plus_c (a b c : ℂ) (h1 : |a| = 1) (h2 : |b| = 1) (h3 : |c| = 1) 
(h4 : a^3 / (b * c) + b^3 / (a * c) + c^3 / (a * b) = -3) :
|a + b + c| = 1 :=
sorry

end abs_a_plus_b_plus_c_l364_364021


namespace two_teachers_place_A_probability_l364_364720

-- Given 3 teachers, each being assigned randomly to one of two places A or B
def probability_two_teachers_place_A : ℚ :=
  let total_assignments := 2^3
  let ways_two_teachers_A := Nat.choose 3 2
  ways_two_teachers_A / total_assignments

-- Proof statement
theorem two_teachers_place_A_probability : probability_two_teachers_place_A = 3 / 8 := by
  sorry

end two_teachers_place_A_probability_l364_364720


namespace triangle_lines_perpendicular_l364_364525

def triangle_ABC (A B C : Type) [MetricSpace A] :=
  ∃ (a b c : A), True

def midpoint (A B C_1 C : Type) [MetricSpace A B C_1 C] :=
  ∃ (c1 : C_1), True

def perpendicular_foot (B C B' C' : Type) [MetricSpace B C B' C'] :=
  ∃ (b' c' : B' C'), True

def lines_perpendicular (B_1 C' C_1 B' : Type) [MetricSpace B_1 C' C_1 B'] :=
  ∃ (b1 c' c1 b' : B_1 C' C_1 B'), True

-- Given that the angle at A is 150 degrees
def angle_A_150 (A B C : Type) [MetricSpace A] :=
  ∃ (a1 b c: A), True

-- Prove lines B_1C' and C_1B' are perpendicular to each other
theorem triangle_lines_perpendicular (A B C C_1 B_1 B' C' : Type) 
  [MetricSpace A B C] [MetricSpace C_1 B_1] [MetricSpace B' C'] :
  angle_A_150 A B C →
  midpoint A B C_1 C →
  midpoint A C B_1 B →
  perpendicular_foot B C B' C' →
  lines_perpendicular B_1 C' C_1 B' :=
sorry

end triangle_lines_perpendicular_l364_364525


namespace trajectory_parabola_minimum_area_quadrilateral_l364_364860

-- Define the ellipse and points F1 and F2
def ellipse : set (ℝ × ℝ) :=
  { p | ∃ x y, p = (x, y) ∧ (x^2 / 8) + (y^2 / 4) = 1 }

def F1 : (ℝ × ℝ) := (-2, 0)
def F2 : (ℝ × ℝ) := (2, 0)

-- Define the line l1 through F1, perpendicular to the x-axis
def l1 : set (ℝ × ℝ) :=
  { p | ∃ y, p = (-2, y)}

-- Define the line l2 perpendicular to l1 at point P
def P : (ℝ × ℝ) := (-2, 0)  -- Assume P is on the y-axis for simplicity

def l2 : set (ℝ × ℝ) :=
  { p | ∃ x, p = (x, 0)}

-- Define the perpendicular bisector of segment PF2 intersecting l2 at M
def M_line : set (ℝ × ℝ) :=
  { p | ∃ x, p = (x, x / 2) }

def M : (ℝ × ℝ) := (1, 0.5)

-- Proof that the trajectory of M is a parabola
theorem trajectory_parabola :
  ∀ M ∈ M_line →  (M.2)^2 = 8 * M.1 :=
by sorry

-- Define the area of quadrilateral ABCD
def area (A B C D : (ℝ × ℝ)) : ℝ :=
  0.5 * (abs (A.1 * B.2 + B.1 * C.2 + C.1 * D.2 + D.1 * A.2 - (A.2 * B.1 + B.2 * C.1 + C.2 * D.1 + D.2 * A.1)))

-- Define the points A, B, C, D on the ellipse through F2
def A : (ℝ × ℝ) := (2, sqrt(4 - (4 / 8)))
def C : (ℝ × ℝ) := (2, -sqrt(4 - (4 / 8)))
def B : (ℝ × ℝ) := (2, sqrt(4 - (4 / 4)))
def D : (ℝ × ℝ) := (2, -sqrt(4 - (4 / 4)))

-- Proof that the minimum area of quadrilateral ABCD is 64/9
theorem minimum_area_quadrilateral :
  area A B C D = 64 / 9 :=
by sorry

end trajectory_parabola_minimum_area_quadrilateral_l364_364860


namespace apple_bags_l364_364176

theorem apple_bags (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apple_bags_l364_364176


namespace train_speed_l364_364296

theorem train_speed 
(length_of_train : ℕ) 
(time_to_cross_pole : ℕ) 
(h_length : length_of_train = 135) 
(h_time : time_to_cross_pole = 9) : 
  (length_of_train / time_to_cross_pole) * 3.6 = 54 :=
by 
  sorry

end train_speed_l364_364296


namespace lift_final_position_l364_364272

def vertical_displacements := [6, -7, 5, -6]

def total_displacement (displacements : List Int) : Int :=
  displacements.sum

def final_position (displacements : List Int) : String × Int :=
if total_displacement displacements < 0 then ("below", Int.natAbs (total_displacement displacements))
else if total_displacement displacements > 0 then ("above", Int.natAbs (total_displacement displacements))
else ("initial", 0)

theorem lift_final_position :
  final_position vertical_displacements = ("below", 2) :=
by
  simp [vertical_displacements, total_displacement, final_position]
  sorry

end lift_final_position_l364_364272


namespace sequence_general_formula_l364_364908

theorem sequence_general_formula (a : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n + 2 * n) :
  ∀ n, a n = n^2 - n + 1 :=
by sorry

end sequence_general_formula_l364_364908


namespace find_a_b_l364_364869

noncomputable def f (x : ℝ) : ℝ := 1 + x - x^2 / 2 + x^3 / 3 - x^4 / 4 + ... + x^2017 / 2017
noncomputable def g (x : ℝ) : ℝ := 1 - x + x^2 / 2 - x^3 / 3 + x^4 / 4 - ... - x^2017 / 2017
noncomputable def F (x : ℝ) : ℝ := f (x + 4) * g (x - 4)
def is_zero_interval (a b : ℤ) : Prop := ∀ x, x ∈ [a - 1, a] ∪ [b - 1, b] → F x = 0

theorem find_a_b (a b : ℤ) (h₁ : a < b) (h₂ : is_zero_interval a b) :
  a + b = 2 :=
sorry  -- proof is not required

end find_a_b_l364_364869


namespace relationship_xyz_l364_364853

noncomputable def x := sorry -- define x such that x^3 = 3
noncomputable def y := sorry -- define y such that y^6 = 7
noncomputable def z := (7 : ℝ)^(1/7)

theorem relationship_xyz : z < y ∧ y < x :=
by
  have hx : x^3 = 3 := sorry
  have hy : y^6 = 7 := sorry
  have hz : z^6 = 7^(6/7) := by sorry
  sorry -- the proof of z < y < x.

end relationship_xyz_l364_364853


namespace triangle_reflection_AB_length_l364_364524

/-
In triangle ABC, point P is the midpoint of AC, point Q is the midpoint of AB. 
Given AP = 9, PC = 18, BQ = 15, and after reflecting triangle ABC over line PQ 
resulting in triangle A'B'C', the length of segment AB in the reflected triangle 
(A'B'C') is equal to 30.
-/
theorem triangle_reflection_AB_length 
  (A B C P Q : Point) 
  (hPQ_midpointQ : Q = midpoint A B)
  (hP_midpointP : P = midpoint A C)
  (hAP : dist A P = 9)
  (hPC : dist P C = 18)
  (hBQ : dist B Q = 15) :
  dist A B = 30 :=
sorry

end triangle_reflection_AB_length_l364_364524


namespace two_angle_BAC_plus_angle_EYF_eq_180_l364_364955

open classical

noncomputable def orthocenter (A B C : Point) : Point := sorry
noncomputable def midpoint (X Y : Point) : Point := sorry
noncomputable def perpendicular (L1 L2 : Line) : Prop := sorry
noncomputable def intersect (L1 : Line) (seg : Segment) : Point := sorry
noncomputable def angle (P Q R : Point) : Real := sorry
noncomputable def angle_sum (α β γ : Real) : Prop := α + β + γ = 180

variables (A B C H E F P Q Y : Point)

axiom H_is_orthocenter : H = orthocenter A B C
axiom E_is_midpoint_AB : E = midpoint A B
axiom F_is_midpoint_AC : F = midpoint A C
axiom P_is_intersection : (∃ L : Line, perpendicular L (Line.mk E H) ∧ P = intersect L (Segment.mk A C))
axiom Q_is_intersection : (∃ L : Line, perpendicular L (Line.mk F H) ∧ Q = intersect L (Segment.mk A B))
axiom Y_is_midpoint_PQ : Y = midpoint P Q

theorem two_angle_BAC_plus_angle_EYF_eq_180 : 2 * angle B A C + angle E Y F = 180 :=
by sorry

end two_angle_BAC_plus_angle_EYF_eq_180_l364_364955


namespace center_of_image_circle_l364_364612

def point := ℝ × ℝ

def reflect_about_y_eq_neg_x (p : point) : point :=
  let (a, b) := p
  (-b, -a)

theorem center_of_image_circle :
  reflect_about_y_eq_neg_x (8, -3) = (3, -8) :=
by
  sorry

end center_of_image_circle_l364_364612


namespace expected_value_correct_l364_364071

-- Define the probability distribution of the user's score in the first round
noncomputable def first_round_prob (X : ℕ) : ℚ :=
  if X = 3 then 1 / 4
  else if X = 2 then 1 / 2
  else if X = 1 then 1 / 4
  else 0

-- Define the conditional probability of the user's score in the second round given the first round score
noncomputable def second_round_prob (X Y : ℕ) : ℚ :=
  if X = 3 then
    if Y = 2 then 1 / 5
    else if Y = 1 then 4 / 5
    else 0
  else
    if Y = 2 then 1 / 3
    else if Y = 1 then 2 / 3
    else 0

-- Define the total score probability
noncomputable def total_score_prob (X Y : ℕ) : ℚ :=
  first_round_prob X * second_round_prob X Y

-- Compute the expected value of the user's total score
noncomputable def expected_value : ℚ :=
  (5 * (total_score_prob 3 2) +
   4 * (total_score_prob 3 1 + total_score_prob 2 2) +
   3 * (total_score_prob 2 1 + total_score_prob 1 2) +
   2 * (total_score_prob 1 1))

-- The theorem to be proven
theorem expected_value_correct : expected_value = 3.3 := 
by sorry

end expected_value_correct_l364_364071


namespace jason_gave_9_cards_l364_364528

theorem jason_gave_9_cards (initial_cards current_cards : ℕ) (h1 : initial_cards = 13) (h2 : current_cards = 4) : initial_cards - current_cards = 9 := by
  rw [h1, h2]
  norm_num
  sorry

end jason_gave_9_cards_l364_364528


namespace total_apples_l364_364157

theorem total_apples (n : ℕ) (h1 : ∀ k : ℕ, 6 * k = n ∨ 12 * k = n) (h2 : 70 ≤ n) (h3 : n ≤ 80) : 
  n = 72 ∨ n = 78 := 
sorry

end total_apples_l364_364157


namespace smallest_integer_modulus_l364_364786

theorem smallest_integer_modulus :
  ∃ n : ℕ, 0 < n ∧ (7 ^ n ≡ n ^ 4 [MOD 3]) ∧
  ∀ m : ℕ, 0 < m ∧ (7 ^ m ≡ m ^ 4 [MOD 3]) → n ≤ m :=
by
  sorry

end smallest_integer_modulus_l364_364786


namespace common_difference_of_variance_is_half_l364_364361

variable {a : ℕ → ℝ} {d : ℝ}

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def mean (s : Fin 7 → ℝ) : ℝ :=
  (1 / 7) * (Finset.sum (Finset.univ : Finset (Fin 7)) s)

def variance (s : Fin 7 → ℝ) : ℝ :=
  mean (λ i : Fin 7, (s i - mean s)^2)

theorem common_difference_of_variance_is_half :
  arithmetic_seq a d →
  variance (λ i, a i.succ) = 1 →
  d = 1/2 ∨ d = -1/2 :=
by
  sorry

end common_difference_of_variance_is_half_l364_364361


namespace sample_may_be_simple_random_l364_364265

-- Define the context of the problem
def class_has_students (boys girls : ℕ) : Prop := boys = 20 ∧ girls = 30

def sample_has_students (sample_boys sample_girls : ℕ) : Prop := sample_boys = 4 ∧ sample_girls = 6

-- State the proposition to be proved
theorem sample_may_be_simple_random (boys girls sample_boys sample_girls : ℕ) 
  (H_class : class_has_students boys girls) 
  (H_sample : sample_has_students sample_boys sample_girls) : 
  ∃ (method : string), method = "simple random sample" :=
sorry

end sample_may_be_simple_random_l364_364265


namespace center_of_image_circle_l364_364613

def point := ℝ × ℝ

def reflect_about_y_eq_neg_x (p : point) : point :=
  let (a, b) := p
  (-b, -a)

theorem center_of_image_circle :
  reflect_about_y_eq_neg_x (8, -3) = (3, -8) :=
by
  sorry

end center_of_image_circle_l364_364613


namespace elephant_weight_equivalence_l364_364516

variable (y : ℝ)
variable (porter_weight : ℝ := 120)
variable (blocks_1 : ℝ := 20)
variable (blocks_2 : ℝ := 21)
variable (porters_1 : ℝ := 3)
variable (porters_2 : ℝ := 1)

theorem elephant_weight_equivalence :
  (y - porters_1 * porter_weight) / blocks_1 = (y - porters_2 * porter_weight) / blocks_2 := 
sorry

end elephant_weight_equivalence_l364_364516


namespace erasers_left_in_the_box_l364_364644

-- Conditions expressed as definitions
def E0 : ℕ := 320
def E1 : ℕ := E0 - 67
def E2 : ℕ := E1 - 126
def E3 : ℕ := E2 + 30

-- Proof problem statement
theorem erasers_left_in_the_box : E3 = 157 := 
by sorry

end erasers_left_in_the_box_l364_364644


namespace reciprocal_solution_l364_364886

theorem reciprocal_solution {x : ℝ} (h : x * -9 = 1) : x = -1/9 :=
sorry

end reciprocal_solution_l364_364886


namespace janice_weekly_earnings_l364_364941

def weekday_rate : ℕ → ℕ 
| hours := if hours ≤ 40 then hours * 10 else (40 * 10) + ((hours - 40) * 15)

def weekend_rate : ℕ → ℕ 
| hours := if hours ≤ 40 then hours * 12 else (40 * 12) + ((hours - 40) * 18)

def holiday_rate : ℕ → ℕ 
| hours := hours * 36

def cases_bonus_penalty (cases : ℕ) : ℕ :=
if cases >= 20 then 50 else if cases <= 15 then -30 else 0

def weekly_earnings := weekday_rate 30 + weekend_rate 25 + holiday_rate 5 + cases_bonus_penalty 17

theorem janice_weekly_earnings : weekly_earnings = 870 := by 
sorry

end janice_weekly_earnings_l364_364941


namespace product_of_y_values_l364_364335

theorem product_of_y_values (y : ℝ) (h : abs (2 * y * 3) + 5 = 47) :
  ∃ y1 y2, (abs (2 * y1 * 3) + 5 = 47) ∧ (abs (2 * y2 * 3) + 5 = 47) ∧ y1 * y2 = -49 :=
by 
  sorry

end product_of_y_values_l364_364335


namespace find_polynomials_l364_364802

theorem find_polynomials (P : ℤ[X]) (a m : ℕ) (h_monic : P.monic) (h_nonconst : degree(P) > 1) 
  (h_exists : ∃ a m, ∀ n : ℕ, n ≡ a [MOD m] → 
    (2022 * ((n + 1)^(n + 1) - n^n) / (P.eval n)) % 1 = 0) :
  (P = X^2 + X + 1) ∨ (P = (X^2 + X + 1)^2) :=
sorry

end find_polynomials_l364_364802


namespace dice_product_composite_probability_l364_364478

theorem dice_product_composite_probability :
  (let total_outcomes := (6 : ℕ)^4,
       non_composite_outcomes := 1,
       composite_probability := 1 - (non_composite_outcomes / total_outcomes) in
   composite_probability = 1295 / 1296) :=
by sorry

end dice_product_composite_probability_l364_364478


namespace laundry_time_l364_364123

theorem laundry_time (n : ℕ) (wash_time dry_time : ℕ) (loads : ℕ) : (loads = 8) → (wash_time = 45) → (dry_time = 60) → (n = 14) → 
  (loads * (wash_time + dry_time)) / 60 = n := 
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end laundry_time_l364_364123


namespace initial_money_l364_364737

theorem initial_money (M : ℝ) 
  (h_clothes : M / 3)
  (h_food : (2 * M / 3) / 5) 
  (h_travel : (4 * 2 * M / (3 * 5)) / 4)
  (h_remaining : (3 * 8 * M / (4 * 15)) = 600) 
: M = 1500 :=
sorry

end initial_money_l364_364737


namespace total_cost_of_pets_is_130_l364_364199
noncomputable theory

def cost_of_pets :=
  let parakeet_cost := 10
  let puppy_cost := 3 * parakeet_cost
  let kitten_cost := 2 * parakeet_cost
  let total_cost := 2 * puppy_cost + 2 * kitten_cost + 3 * parakeet_cost
  total_cost

theorem total_cost_of_pets_is_130 : cost_of_pets = 130 :=
by
  -- Showing that cost_of_pets indeed evaluates to 130
  let parakeet_cost := 10
  let puppy_cost := 3 * parakeet_cost
  let kitten_cost := 2 * parakeet_cost
  let total_cost := 2 * puppy_cost + 2 * kitten_cost + 3 * parakeet_cost
  show total_cost = 130 from sorry

end total_cost_of_pets_is_130_l364_364199


namespace symmetrical_circle_equation_l364_364093

theorem symmetrical_circle_equation :
  ∀ (x y : ℝ), (x^2 + y^2 - 2 * x - 1 = 0) ∧ (2 * x - y + 1 = 0) →
  ((x + 7/5)^2 + (y - 6/5)^2 = 2) :=
sorry

end symmetrical_circle_equation_l364_364093


namespace count_distinct_x_l364_364438

theorem count_distinct_x :
  { x : ℝ | ∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ s = floor (sqrt (123 - sqrt x)) } = 12 :=
sorry

end count_distinct_x_l364_364438


namespace abs_neg_two_l364_364084

theorem abs_neg_two : abs (-2) = 2 :=
by
  sorry

end abs_neg_two_l364_364084


namespace find_b_for_continuity_l364_364557

def f_continuous_at_3 (b : ℚ) : Prop :=
∀ x : ℚ, f x = if x ≤ 3 then 3 * x^2 + 5 else b * x + 6

theorem find_b_for_continuity (b : ℚ) : b = 26 / 3 :=
by
  sorry

end find_b_for_continuity_l364_364557


namespace circle_reflection_l364_364600

theorem circle_reflection (x y : ℝ) (hx : x = 8) (hy : y = -3) : 
  let reflected_x := -y, reflected_y := -x in
  reflected_x = 3 ∧ reflected_y = -8 :=
by
  sorry

end circle_reflection_l364_364600


namespace expression_non_negative_l364_364581

theorem expression_non_negative (α : ℝ) (h1 : α ≠ (π / 2) + k * π) (h2 : α ≠ k * π) (h3 : α ≠ π + k * π) : 
  (sin α + tan α) / (cos α + cot α) ≥ 0 := sorry

end expression_non_negative_l364_364581


namespace smallest_b_l364_364812

theorem smallest_b (b : ℕ) : (∀ x : ℤ, ¬ prime (x^4 + b^2)) ∧ (b % 2 = 0) → b ≥ 8 ∧ (∀ x : ℤ, ¬ prime (x^4 + 64)) :=
begin
  sorry
end

end smallest_b_l364_364812


namespace short_video_length_l364_364946

theorem short_video_length 
  (videos_per_day : ℕ) 
  (short_videos_factor : ℕ) 
  (weekly_total_minutes : ℕ) 
  (days_in_week : ℕ) 
  (total_videos : videos_per_day = 3)
  (one_video_longer : short_videos_factor = 6)
  (total_weekly_minutes : weekly_total_minutes = 112)
  (days_a_week : days_in_week = 7) :
  ∃ x : ℕ, (videos_per_day * (short_videos_factor + 2)) * days_in_week = weekly_total_minutes ∧ 
            x = 2 := 
by 
  sorry 

end short_video_length_l364_364946


namespace HappyValleyKennel_l364_364082

theorem HappyValleyKennel :
  ∃ ways : ℕ, ways = 2 * (Nat.factorial 3) * (Nat.factorial 4) * (Nat.factorial 6) ∧ ways = 69120 :=
begin
  let chickens := 3,
  let dogs := 4,
  let cats := 6,
  let total := 13,
  let cat_center_arrangements := 2,
  let chicken_arrangements := Nat.factorial chickens,
  let dog_arrangements := Nat.factorial dogs,
  let cat_arrangements := Nat.factorial cats,
  let ways := cat_center_arrangements * chicken_arrangements * dog_arrangements * cat_arrangements,
  use ways,
  split,
  { refl },
  { norm_num }
end

end HappyValleyKennel_l364_364082


namespace probability_one_pair_same_color_l364_364401

theorem probability_one_pair_same_color :
  let n := 8
  let r := 4
  let total_combinations := Nat.choose n r
  let choose_colors := Nat.choose 4 3
  let choose_pair_color := Nat.choose 3 1
  let favorable_ways := choose_colors * choose_pair_color * 2 * 2
  let probability := (favorable_ways : ℚ) / (total_combinations : ℚ)
  probability = 24 / 35 :=
by
  sorry

end probability_one_pair_same_color_l364_364401


namespace const_term_binom_expansion_l364_364086

theorem const_term_binom_expansion (r x : ℝ) (h : 8 = 3 * r) :
  binomial_coeff 8 2 = 28 :=
by sorry

noncomputable def binomial_coeff (n k : ℕ) : ℚ :=
if k > n then 0 else (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

end const_term_binom_expansion_l364_364086


namespace Shyne_total_plants_l364_364066

theorem Shyne_total_plants :
  let eggplants_per_packet := 14 in
  let sunflowers_per_packet := 10 in
  let packets_of_eggplants := 4 in
  let packets_of_sunflowers := 6 in
  let total_plants := (eggplants_per_packet * packets_of_eggplants) + (sunflowers_per_packet * packets_of_sunflowers) in
  total_plants = 116 := 
by
  -- The proof would go here
  sorry

end Shyne_total_plants_l364_364066


namespace O2_tangent_O_l364_364243

-- Definitions of points and circles
structure Point := (x : ℝ) (y : ℝ)
structure Circle := (center : Point) (radius : ℝ)

-- Two distinct points on a given circle O
axiom A : Point
axiom B : Point
axiom O : Circle

-- Midpoint of A and B
def midpoint (A B : Point) : Point :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

noncomputable def P : Point := midpoint A B

-- Circle O1 tangent to AB at P and tangent to circle O
axiom O1 : Circle
axiom tangent_AB_O1 : P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2
axiom tangent_O_O1 : ∀ P : Point, dist P O1.center = O1.radius → dist P O.center = O.radius - O1.radius

-- Tangent line ℓ through A intersecting circle O at C
axiom ℓ : Point → Point → list Point
noncomputable def C : Point := List.head' (ℓ A O.center).tail

-- Q is the midpoint of BC
noncomputable def Q : Point := midpoint B C

-- Circle O2 tangent to BC at Q and tangent to line segment AC
axiom O2 : Circle
axiom tangent_BC_O2 : ∃ P : Point, dist P O2.center = O2.radius ∧ P = Q
axiom tangent_AC_O2 : ∃ P : Point, dist P O2.center = O2.radius → ∃ P : Point, dist P O.center = O.radius - O2.radius

-- Main theorem to prove
theorem O2_tangent_O : ∃ P : Point, dist P O2.center = O2.radius ∧ dist P O.center = O.radius - O2.radius :=
sorry

end O2_tangent_O_l364_364243


namespace average_sales_proof_l364_364772

noncomputable def adjusted_sales_may := 110 / 0.9
def total_sales (sales : List ℝ) (adjusted_may : ℝ) : ℝ :=
  sales.foldr (λ x acc => x + acc) 0 - 110 + adjusted_may

def average_sales (total_sales : ℝ) (months : ℝ) : ℝ :=
  total_sales / months

theorem average_sales_proof :
  let sales := [120, 80, 50, 130, 110, 90]
  let adjusted_may := adjusted_sales_may
  let total := total_sales sales adjusted_may
  let avg := average_sales total 6
  avg = 98.70 :=
by
  sorry

end average_sales_proof_l364_364772


namespace distinct_real_x_l364_364417

theorem distinct_real_x (x : ℝ) :
  (∃! x, ∃ (s : ℝ), s ∈ set.Icc 0 11 ∧ s = real.sqrt (123 - real.sqrt x)) =
  12 := sorry

end distinct_real_x_l364_364417


namespace tetrakis_hexahedronColoring_l364_364586

noncomputable def distinguishable_colorings (n_colors: ℕ) (n_faces: ℕ) (group_order: ℕ) : ℕ :=
  Nat.desc_factorial n_colors (n_faces - 1) / group_order

theorem tetrakis_hexahedronColoring :
  distinguishable_colorings 12 16 24 = 479001600 :=
by
  sorry

end tetrakis_hexahedronColoring_l364_364586


namespace area_of_trapezoid_l364_364928

/--
In trapezoid \(ABCD\) with bases \(AD \parallel BC\), the diagonals intersect at point \(E\).
Given the areas \(S(\triangle ADE) = 12\) and \(S(\triangle BCE) = 3\),
the area of the trapezoid \(ABCD\) is 27.
-/
theorem area_of_trapezoid (A B C D E : Type) 
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
  (area_ADE area_BCE : ℝ) (h_area_ADE : area_ADE = 12) (h_area_BCE : area_BCE = 3)
  (h_parallel : AD ∥ BC) (h_diag_intersection : intersect A C B D = E) :
  area_trapezoid ABCD = 27 := 
sorry

end area_of_trapezoid_l364_364928


namespace distance_between_house_and_school_l364_364744

variable (T D : ℝ)

axiom cond1 : 9 * (T + 20 / 60) = D
axiom cond2 : 12 * (T - 20 / 60) = D
axiom cond3 : 15 * (T - 40 / 60) = D

theorem distance_between_house_and_school : D = 24 := 
by
  sorry

end distance_between_house_and_school_l364_364744


namespace admission_plans_eq_150_l364_364789

noncomputable def num_admission_plans : ℕ :=
  let students := {A, B, C, D, E}
  let universities := {PekingU, SJTU, ZJU}
  -- Define university constraints ensuring at least one student per university
  ∑ (admissions : students → universities) in {f | (∀ u ∈ universities, ∃ s ∈ students, f s = u) ∧ f A ≠ PekingU}, 1

theorem admission_plans_eq_150 :
  num_admission_plans = 150 :=
sorry

end admission_plans_eq_150_l364_364789


namespace M_less_equal_fraction_M_M_greater_equal_fraction_M_M_less_equal_sum_M_l364_364985

noncomputable def M : ℕ → ℕ → ℕ → ℝ := sorry

theorem M_less_equal_fraction_M (n k h : ℕ) : 
  M n k h ≤ (n / h) * M (n-1) (k-1) (h-1) :=
sorry

theorem M_greater_equal_fraction_M (n k h : ℕ) : 
  M n k h ≥ (n / (n - h)) * M (n-1) k k :=
sorry

theorem M_less_equal_sum_M (n k h : ℕ) : 
  M n k h ≤ M (n-1) (k-1) (h-1) + M (n-1) k h :=
sorry

end M_less_equal_fraction_M_M_greater_equal_fraction_M_M_less_equal_sum_M_l364_364985


namespace probability_composite_product_l364_364466

theorem probability_composite_product :
  let dice_faces := 6
  let rolls := 4
  let total_outcomes := dice_faces ^ rolls
  let non_composite_cases := 13
  let non_composite_probability := non_composite_cases / total_outcomes
  let composite_probability := 1 - non_composite_probability
  composite_probability = 1283 / 1296 := by
  sorry

end probability_composite_product_l364_364466


namespace winner_C_l364_364515

noncomputable def votes_A : ℕ := 4500
noncomputable def votes_B : ℕ := 7000
noncomputable def votes_C : ℕ := 12000
noncomputable def votes_D : ℕ := 8500
noncomputable def votes_E : ℕ := 3500

noncomputable def total_votes : ℕ := votes_A + votes_B + votes_C + votes_D + votes_E

noncomputable def percentage (votes : ℕ) : ℚ :=
   (votes : ℚ) / (total_votes : ℚ) * 100

noncomputable def percentage_A : ℚ := percentage votes_A
noncomputable def percentage_B : ℚ := percentage votes_B
noncomputable def percentage_C : ℚ := percentage votes_C
noncomputable def percentage_D : ℚ := percentage votes_D
noncomputable def percentage_E : ℚ := percentage votes_E

theorem winner_C : (percentage_C = 33.803) := 
sorry

end winner_C_l364_364515


namespace trigonometric_identity_l364_364997

theorem trigonometric_identity :
  let sin_30 := 1 / 2,
      sin_60 := Real.sqrt 3 / 2,
      cos_30 := Real.sqrt 3 / 2,
      cos_60 := 1 / 2,
      sin_45 := Real.sqrt 2 / 2,
      cos_45 := Real.sqrt 2 / 2 in
  (sin_30 + sin_60) / (cos_30 + cos_60) = Real.tan (Real.pi / 4) := 
  by
    sorry

end trigonometric_identity_l364_364997


namespace solution_l364_364787

def is_rational (x : ℝ) : Prop := ∃ a b : ℚ, b ≠ 0 ∧ x = a / b

def expr1 := sqrt (4 * real.pi^2)
def expr2 := real.cbrt 1.728
def expr3 := real.rpow 0.0032 (1 / 5)
def expr4 := real.cbrt (-8) * sqrt ((0.25)⁻¹)

theorem solution : ¬ is_rational expr1 ∧ is_rational expr2 ∧ is_rational expr3 ∧ is_rational expr4 := by
  sorry

end solution_l364_364787


namespace sum_of_S_p_equals_204875_l364_364347

-- Define the arithmetic progression sum function
def arithmetic_progression_sum (p n : ℕ) : ℕ :=
  let a_n := p + (n - 1) * 3 * p
  in n * (p + a_n) / 2

-- Define S_p as the sum of the first 50 terms of the arithmetic progression
def S_p (p : ℕ) : ℕ := arithmetic_progression_sum p 50

-- Prove the main theorem
theorem sum_of_S_p_equals_204875 :
  ∑ p in Finset.range 10, S_p (p + 1) = 204875 :=
by
  -- Directly use the given conditions and results
  sorry

end sum_of_S_p_equals_204875_l364_364347


namespace value_of_x_plus_y_l364_364400

theorem value_of_x_plus_y 
  (x y : ℝ)
  (h1 : |x| = 3)
  (h2 : |y| = 2)
  (h3 : x > y) :
  x + y = 5 ∨ x + y = 1 := 
  sorry

end value_of_x_plus_y_l364_364400


namespace total_hours_charged_l364_364053

theorem total_hours_charged (K P M : ℕ) 
  (h₁ : P = 2 * K)
  (h₂ : P = (1 / 3 : ℚ) * (K + 80))
  (h₃ : M = K + 80) : K + P + M = 144 :=
by {
    sorry
}

end total_hours_charged_l364_364053


namespace parabola_properties_l364_364376

-- Define the parabola with given conditions
def parabola (p x y : ℝ) := y^2 = 2 * p * x ∧ p > 0

-- Define the focus of the parabola
structure focus (p : ℝ) :=
(val : ℝ)
(cond : val = p/2)

-- Define the conditions
axiom point_on_parabola (p b : ℝ) : ∃ p, parabola p 2 b
axiom distance_to_focus (p b : ℝ) (F : focus p) : |2 - F.val| = 4

-- Define the intersection and line equation
axiom line_intersection (t m x1 y1 x2 y2 : ℝ) : 
  y1^2 = 8 * x1 ∧
  y2^2 = 8 * x2 ∧
  x1 = t * y1 + m ∧
  x2 = t * y2 + m ∧
  m > 0

-- Prove fixed point for line passing through origin
theorem parabola_properties (t m x1 y1 x2 y2 : ℝ) (F : focus 4) :
  point_on_parabola 4 y1 ∧
  distance_to_focus 4 y1 F ∧
  line_intersection t m x1 y1 x2 y2 →
  (parabola 4 2 y1 ∧ x = t * y + m ∧ m = 8 → x = 8) :=
sorry

end parabola_properties_l364_364376


namespace small_font_words_per_page_l364_364945

theorem small_font_words_per_page (total_words : ℕ) (large_font_words_per_page : ℕ)
  (total_pages : ℕ) (large_font_pages : ℕ) (small_font_words_per_page : ℕ) :
  total_words = 48000 →
  large_font_words_per_page = 1800 →
  total_pages = 21 →
  large_font_pages = 4 →
  small_font_words_per_page = (total_words - large_font_words_per_page * large_font_pages) / (total_pages - large_font_pages) →
  small_font_words_per_page = 2400 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  simp at h5
  exact h5

end small_font_words_per_page_l364_364945


namespace math_problem_l364_364352

noncomputable def op (a b : ℝ) (x y : ℝ) : ℝ := a * x - b * y

theorem math_problem (x y : ℝ) :
  (sqrt (2 * x + 1)) = 3 →
  (∛ (-3 * x + y + 1)) = -2 →
  (x = 4 ∧ y = 3) →
  (∀ m n : ℝ, (4 * m - 3 * n = 1 ∧ 4 * m - 6 * n = -2 → -m < 0 ∧ -n < 0)) →
  (∀ k m n : ℝ, (m ≠ n) → (4 * k * m - 3 * n = 4 * k * n - 3 * m) → k = -3/4) →
  (∀ m : ℝ, (abs (m - 5) - abs (m + 2)) = (abs (5 - m) - abs (2 + m)) →
    ((4 * m - 6) * ((12 * m) + 24) >= 0)) →
  true :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end math_problem_l364_364352


namespace probability_composite_l364_364498

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def non_composite_outcomes : ℕ := 1 + 12

theorem probability_composite :
  let total_outcomes := 6^4 in
  let non_composite := non_composite_outcomes in
  let composite_probability := (total_outcomes - non_composite) / total_outcomes.toRat in
  composite_probability = 1283 / 1296 := by
  let total_outcomes := 6^4
  let non_composite := 1 + 12
  let composite_probability := (total_outcomes - non_composite).toRat / total_outcomes.toRat
  sorry

end probability_composite_l364_364498


namespace number_of_tiles_needed_l364_364283

-- Definitions based on the given conditions
def room_length : ℝ := 15
def room_width : ℝ := 20
def tile_length : ℝ := 1 / 4
def tile_width : ℝ := 3 / 4

-- Mathematical statement to prove
theorem number_of_tiles_needed : room_length * room_width / (tile_length * tile_width) = 1600 := by
  -- Proof script (the details of the proof are skipped here using sorry)
  sorry

end number_of_tiles_needed_l364_364283


namespace cuboid_height_l364_364646

-- Given conditions
def volume_cuboid : ℝ := 1380 -- cubic meters
def base_area_cuboid : ℝ := 115 -- square meters

-- Prove that the height of the cuboid is 12 meters
theorem cuboid_height : volume_cuboid / base_area_cuboid = 12 := by
  sorry

end cuboid_height_l364_364646


namespace julie_hours_per_week_during_school_year_l364_364540

def julie's_hourly_wage (total_earnings : ℕ) (hours_worked : ℕ) : ℕ :=
  total_earnings / hours_worked

def weekly_earnings (total_earnings : ℕ) (duration_in_weeks : ℕ) : ℕ :=
  total_earnings / duration_in_weeks

theorem julie_hours_per_week_during_school_year
  (hours_summer_per_week : ℕ)
  (weeks_summer : ℕ)
  (total_summer_earnings : ℕ)
  (weeks_school_year : ℕ)
  (total_school_year_earnings : ℕ)
  (h1 : hours_summer_per_week = 60)
  (h2 : weeks_summer = 10)
  (h3 : total_summer_earnings = 7500)
  (h4 : weeks_school_year = 50)
  (h5 : total_school_year_earnings = 7500) :
  let hourly_wage := julie's_hourly_wage total_summer_earnings (hours_summer_per_week * weeks_summer)
      required_weekly_earnings := weekly_earnings total_school_year_earnings weeks_school_year
  in required_weekly_earnings / hourly_wage = 12 := by
  sorry

end julie_hours_per_week_during_school_year_l364_364540


namespace percentage_increase_in_take_home_pay_l364_364010

namespace JohnPayRaise

-- Conditions from part a)
variable (gross_pay_before : ℝ := 60)
variable (gross_pay_after : ℝ := 70)
variable (tax_rate_before : ℝ := 0.15)
variable (tax_rate_after : ℝ := 0.18)

-- Derived values based on conditions
def take_home_pay_before := gross_pay_before - (tax_rate_before * gross_pay_before)
def take_home_pay_after := gross_pay_after - (tax_rate_after * gross_pay_after)

-- Statement to prove percentage increase
theorem percentage_increase_in_take_home_pay :
  let percentage_increase := ((take_home_pay_after - take_home_pay_before) / take_home_pay_before) * 100 in
  percentage_increase = 12.55 :=
by
  sorry

end JohnPayRaise

end percentage_increase_in_take_home_pay_l364_364010


namespace profit_increase_example_l364_364635

noncomputable def percent_increase (initial_profit final_profit : ℝ) : ℝ :=
  ((final_profit - initial_profit) / initial_profit) * 100

theorem profit_increase_example (P : ℝ) :
  let April_Profit := 1.30 * P
  let May_Profit := 1.30 * P - 0.20 * (1.30 * P)
  let June_Profit := May_Profit + 0.50 * May_Profit
  percent_increase P June_Profit = 56 := 
by
  have April_Profit_eq : April_Profit = 1.30 * P := rfl
  have May_Profit_eq : May_Profit = 1.04 * P := rfl
  have June_Profit_eq : June_Profit = 1.56 * P := rfl
  rw [April_Profit_eq, May_Profit_eq, June_Profit_eq]
  unfold percent_increase
  simp
  norm_num
  sorry

end profit_increase_example_l364_364635


namespace v_19_in_terms_of_b_l364_364332

noncomputable def sequence (b : ℝ) : ℕ → ℝ
| 0       := b
| (n + 1) := -1 / (sequence b n + 2)

theorem v_19_in_terms_of_b (b : ℝ) (h : b > 0) : sequence b 18 = - (b + 2) / (b + 3) := by
  sorry

end v_19_in_terms_of_b_l364_364332


namespace jake_earnings_per_hour_l364_364527

-- Definitions for conditions
def initialDebt : ℕ := 100
def payment : ℕ := 40
def hoursWorked : ℕ := 4
def remainingDebt : ℕ := initialDebt - payment

-- Theorem stating Jake's earnings per hour
theorem jake_earnings_per_hour : remainingDebt / hoursWorked = 15 := by
  sorry

end jake_earnings_per_hour_l364_364527


namespace sam_birthday_l364_364573

-- Assuming days of the week are represented as natural numbers starting from 0 (Monday)
inductive Day : Type
| Monday : Day
| Tuesday : Day
| Wednesday : Day
| Thursday : Day
| Friday : Day
| Saturday : Day
| Sunday : Day

open Day

-- Define the function that calculates the day of the week after a certain number of days
def days_after (n : ℕ) (start_day : Day) : Day :=
  match (enum start_day + n) % 7 with
  | 0 => Monday
  | 1 => Tuesday
  | 2 => Wednesday
  | 3 => Thursday
  | 4 => Friday
  | 5 => Saturday
  | 6 => Sunday
  | _ => sorry -- this case is actually unreachable

-- The statement to be proven
theorem sam_birthday (n : ℕ) (start_day : Day) (h : start_day = Monday) (h1 : n = 45) :
  days_after n start_day = Thursday :=
by
  rw [h, h1]
  unfold days_after
  decide

end sam_birthday_l364_364573


namespace min_distance_curve_to_line_l364_364981

noncomputable def curve : ℝ → ℝ := λ x, x^2 - Real.log x
noncomputable def line : ℝ → ℝ := λ x, x + 2

theorem min_distance_curve_to_line :
  ∀ P : ℝ × ℝ, P.snd = curve P.fst → 
  (∃ d : ℝ, d = Real.sqrt 2 ∧
  ∀ Q : ℝ × ℝ, Q.snd = line Q.fst → Real.dist P Q ≥ d) :=
by
  sorry

end min_distance_curve_to_line_l364_364981


namespace omega_value_l364_364372

theorem omega_value (ω : ℕ) (h : ω > 0) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = Real.sin (ω * x + Real.pi / 4)) 
  (h2 : ∀ x y, (Real.pi / 6 < x ∧ x < Real.pi / 3) → (Real.pi / 6 < y ∧ y < Real.pi / 3) → x < y → f y < f x) :
    ω = 2 ∨ ω = 3 := 
sorry

end omega_value_l364_364372


namespace num_real_numbers_l364_364427

theorem num_real_numbers (x : ℝ) : 
  (∃ n : ℕ, n ≤ 11 ∧ x = (123 - n^2)^2) ↔ 
  (∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ (x = (123 - s^2)^2) ∧ 
  (∃! x, ∃ k : ℕ, 0 ≤ k ∧ k ≤ 11 ∧ x = (123 - k^2)^2)) :=
by
  sorry

end num_real_numbers_l364_364427


namespace first_formula_second_formula_l364_364248

variable {R : Type*} [CommRing R] (a b : R) (n : ℕ)

theorem first_formula : a^(n + 1) - b^(n + 1) = (a - b) * (Finset.sum (Finset.range (n + 1)) (λ k, a ^ (n - k) * b ^ k)) :=
begin
  sorry
end

theorem second_formula : a^(2 * n + 1) + b^(2 * n + 1) = (a + b) * (Finset.sum (Finset.range (2 * n + 1)) (λ k, (-1)^k * a ^ (2 * n - k) * b ^ k)) :=
begin
  sorry
end

end first_formula_second_formula_l364_364248


namespace count_distinct_x_l364_364436

theorem count_distinct_x :
  { x : ℝ | ∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ s = floor (sqrt (123 - sqrt x)) } = 12 :=
sorry

end count_distinct_x_l364_364436


namespace max_indistinguishable_digits_for_reciprocal_factorials_l364_364788

theorem max_indistinguishable_digits_for_reciprocal_factorials : 
  let Dima_factorial_lower := 80
  let Dima_factorial_upper := 99
  ∃ N : ℕ, 
    (∀ k ∈ Finset.range (Dima_factorial_upper - Dima_factorial_lower + 1), 
      let k_fact := (Dima_factorial_lower + k)!
      ∀ l ∈ Finset.range (Dima_factorial_upper - Dima_factorial_lower + 1), k ≠ l →
        ∃ f : ℕ → ℕ → ℕ → string, f k_fact l 0 ≠ f l.fact k 0) ∧ 
    N = 155 :=
begin
  sorry
end

end max_indistinguishable_digits_for_reciprocal_factorials_l364_364788


namespace vertex_A_east_probability_l364_364938

theorem vertex_A_east_probability (A B C : Type) (angle_A : ℝ) 
  (h : angle_A = 40) : 
  probability_A_east(A, B, C) = 7 / 18 := by
  sorry

end vertex_A_east_probability_l364_364938


namespace num_real_satisfying_x_l364_364443

theorem num_real_satisfying_x : 
  (∃ (x : ℝ), ∃ (s : ℕ) (hs : s ∈ set.Ico 0 (floor (sqrt 123) + 1)), 
   s = int.of_nat (floor (sqrt (123 - sqrt x)))) →
   12 := sorry

end num_real_satisfying_x_l364_364443


namespace total_milkshakes_made_l364_364311

def Augustus : ℕ := 3
def Luna : ℕ := 7
def Neptune : ℕ := 5

def hours_worked : ℕ := 12
def hours_when_Neptune_joined : ℕ := 3
def break_interval_1 : ℕ := 3
def break_interval_2 : ℕ := 7

def break_consumption_1 : ℕ := 6
def break_consumption_2 : ℕ := 8
def break_consumption_3 : ℕ := 4

def breaks (hours : ℕ) (intervals : List ℕ) : ℕ :=
  intervals.map (λ n => hours / n).sum

theorem total_milkshakes_made : 
  (Augustus * 3 + Luna * 3 + (Augustus + Luna + Neptune) * 9 - (breaks 12 [3, 7] * (break_consumption_1 + break_consumption_2 + break_consumption_3)) = 93) :=
by
  sorry

end total_milkshakes_made_l364_364311


namespace minimum_value_of_sum_of_squares_l364_364459

variable {x y : ℝ}

theorem minimum_value_of_sum_of_squares (h : x^2 + 2*x*y - y^2 = 7) : 
  x^2 + y^2 ≥ 7 * Real.sqrt 2 / 2 := by 
    sorry

end minimum_value_of_sum_of_squares_l364_364459


namespace apple_count_l364_364185

-- Define constants for the problem
constant bags: ℕ → Prop
constant total_apples: ℕ
constant apples_in_bag1: ℕ := 12
constant apples_in_bag2: ℕ := 6

-- Define the conditions
axiom condition1 (n: ℕ) : bags n → (n = apples_in_bag1 ∨ n = apples_in_bag2)
axiom condition2 : 70 ≤ total_apples ∧ total_apples ≤ 80
axiom condition3 : total_apples % 6 = 0

-- State the proof problem
theorem apple_count :
  total_apples = 72 ∨ total_apples = 78 :=
sorry

end apple_count_l364_364185


namespace num_real_numbers_l364_364428

theorem num_real_numbers (x : ℝ) : 
  (∃ n : ℕ, n ≤ 11 ∧ x = (123 - n^2)^2) ↔ 
  (∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ (x = (123 - s^2)^2) ∧ 
  (∃! x, ∃ k : ℕ, 0 ≤ k ∧ k ≤ 11 ∧ x = (123 - k^2)^2)) :=
by
  sorry

end num_real_numbers_l364_364428


namespace distinct_real_x_l364_364413

theorem distinct_real_x (x : ℝ) :
  (∃! x, ∃ (s : ℝ), s ∈ set.Icc 0 11 ∧ s = real.sqrt (123 - real.sqrt x)) =
  12 := sorry

end distinct_real_x_l364_364413


namespace possible_apple_counts_l364_364126

theorem possible_apple_counts (n : ℕ) (h₁ : 70 ≤ n) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end possible_apple_counts_l364_364126


namespace num_real_numbers_l364_364429

theorem num_real_numbers (x : ℝ) : 
  (∃ n : ℕ, n ≤ 11 ∧ x = (123 - n^2)^2) ↔ 
  (∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ (x = (123 - s^2)^2) ∧ 
  (∃! x, ∃ k : ℕ, 0 ≤ k ∧ k ≤ 11 ∧ x = (123 - k^2)^2)) :=
by
  sorry

end num_real_numbers_l364_364429


namespace sum_of_drawn_numbers_is_26_l364_364339

theorem sum_of_drawn_numbers_is_26 :
  ∃ A B : ℕ, A > 1 ∧ A ≤ 50 ∧ B ≤ 50 ∧ A ≠ B ∧ Prime B ∧
           (150 * B + A = k^2) ∧ 1 ≤ B ∧ (B > 1 → A > 1 ∧ B = 2) ∧ A + B = 26 :=
by
  sorry

end sum_of_drawn_numbers_is_26_l364_364339


namespace no_real_solutions_l364_364458

theorem no_real_solutions (a b : ℝ) (hb : b > 2) : ¬ ∃ x : ℝ, sqrt (b - cos (a + x)) = x :=
by
  sorry

end no_real_solutions_l364_364458


namespace cross_product_correct_l364_364807

def vec1 : ℝ × ℝ × ℝ := (4, 2, -1)
def vec2 : ℝ × ℝ × ℝ := (3, -3, 6)

theorem cross_product_correct : 
  cross_product vec1 vec2 = (9, -27, -18) :=
sorry

end cross_product_correct_l364_364807


namespace find_f_2016_l364_364854

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain (x : ℝ) : f x ∈ ℝ
axiom f_condition (x y : ℝ) : 4 * f x * f y = f (x + y) + f (x - y)
axiom f_given : f 1 = 1 / 4

theorem find_f_2016 : f 2016 = 1 / 2 :=
by
  sorry

end find_f_2016_l364_364854


namespace apple_bags_l364_364178

theorem apple_bags (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apple_bags_l364_364178


namespace possible_apple_counts_l364_364130

theorem possible_apple_counts (n : ℕ) (h₁ : 70 ≤ n) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end possible_apple_counts_l364_364130


namespace value_of_x_plus_y_l364_364399

theorem value_of_x_plus_y 
  (x y : ℝ)
  (h1 : |x| = 3)
  (h2 : |y| = 2)
  (h3 : x > y) :
  x + y = 5 ∨ x + y = 1 := 
  sorry

end value_of_x_plus_y_l364_364399


namespace pow_mod_eleven_l364_364675

theorem pow_mod_eleven : 
  ∀ (n : ℕ), (n ≡ 5 ^ 1 [MOD 11] → n ≡ 5 [MOD 11]) ∧ 
             (n ≡ 5 ^ 2 [MOD 11] → n ≡ 3 [MOD 11]) ∧ 
             (n ≡ 5 ^ 3 [MOD 11] → n ≡ 4 [MOD 11]) ∧ 
             (n ≡ 5 ^ 4 [MOD 11] → n ≡ 9 [MOD 11]) ∧ 
             (n ≡ 5 ^ 5 [MOD 11] → n ≡ 1 [MOD 11]) →
  5 ^ 1233 ≡ 4 [MOD 11] :=
by
  intro n h
  sorry

end pow_mod_eleven_l364_364675


namespace min_f_case1_min_f_case2_min_f_l364_364556

open Real

-- Definitions
def f (a b m n : ℕ+) (x : ℝ) : ℝ :=
  (sin x) ^ (m : ℕ) / (a : ℝ) + (b : ℝ) / (sin x) ^ (n : ℕ)

-- Proving the minimum value of f(x)
theorem min_f_case1 (a b m n : ℕ+) (h : (↑a * ↑b * ↑n : ℝ) ≥ ↑m) :
  (∃ x ∈ Ioo 0 π, f a b m n x = 1 / a + b) :=
sorry

theorem min_f_case2 (a b m n : ℕ+) (h : (↑a * ↑b * ↑n : ℝ) < ↑m) :
  (∃ x ∈ Ioo 0 π, f a b m n x = (m + n) * (1 / (n * a : ℝ)) ^ n * (b / m : ℝ) ^ m) :=
sorry

-- The general minima theorem
theorem min_f (a b m n : ℕ+) :
  ( if (↑a * ↑b * ↑n : ℝ) ≥ ↑m then 
      ∃ x ∈ Ioo 0 π, f a b m n x = (1 / a : ℝ) + b 
    else 
      ∃ x ∈ Ioo 0 π, f a b m n x = (m + n) * (1 / (n * a : ℝ)) ^ n * (b / m : ℝ) ^ m ) :=
by
  by_cases (h : (↑a * ↑b * ↑n : ℝ) ≥ ↑m)
  · exact min_f_case1 a b m n h
  · exact min_f_case2 a b m n (not_le.mp h)

end min_f_case1_min_f_case2_min_f_l364_364556


namespace composite_p_squared_plus_36_l364_364704

theorem composite_p_squared_plus_36 (p : ℕ) (h_prime : Prime p) : 
  ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ (k * m = p^2 + 36) :=
by {
  sorry
}

end composite_p_squared_plus_36_l364_364704


namespace fourth_number_is_57_l364_364736

noncomputable def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def sum_list (l : List ℕ) : ℕ :=
  l.foldr (.+.) 0

theorem fourth_number_is_57 : 
  ∃ (N : ℕ), N < 100 ∧ 177 + N = 4 * (33 + digit_sum N) ∧ N = 57 :=
by {
  sorry
}

end fourth_number_is_57_l364_364736


namespace shareCoins_l364_364920

theorem shareCoins (a b c d e d : ℝ)
  (h1 : b = a - d)
  (h2 : ((a-2*d) + b = a + (a+d) + (a+2*d)))
  (h3 : (a-2*d) + b + a + (a+d) + (a+2*d) = 5) :
  b = 7 / 6 :=
by
  sorry

end shareCoins_l364_364920


namespace center_of_image_circle_l364_364610

def point := ℝ × ℝ

def reflect_about_y_eq_neg_x (p : point) : point :=
  let (a, b) := p
  (-b, -a)

theorem center_of_image_circle :
  reflect_about_y_eq_neg_x (8, -3) = (3, -8) :=
by
  sorry

end center_of_image_circle_l364_364610


namespace find_cost_price_l364_364274

-- Defining the conditions
def selling_price : ℝ := 70
def profit_rate : ℝ := 40 / 100

-- The cost price (CP) we wish to prove
def cost_price : ℝ := 50

-- The equation representing the relationship between SP, CP, and profit_rate
def selling_price_eq (CP : ℝ) : Prop :=
  selling_price = CP * (1 + profit_rate)

-- Main theorem to prove the cost price
theorem find_cost_price : selling_price_eq cost_price :=
  sorry

end find_cost_price_l364_364274


namespace dad_caught_more_l364_364771

theorem dad_caught_more {trouts_caleb : ℕ} (h₁ : trouts_caleb = 2) 
    (h₂ : ∃ trouts_dad : ℕ, trouts_dad = 3 * trouts_caleb) : 
    ∃ more_trouts : ℕ, more_trouts = 4 := by
  sorry

end dad_caught_more_l364_364771


namespace sum_of_solutions_l364_364089

theorem sum_of_solutions (x : ℝ) :
  (∀ x : ℝ, x^3 + x^2 - 6*x - 20 = 4*x + 24) →
  let polynomial := (x^3 + x^2 - 10*x - 44);
  (polynomial = 0) →
  let a := 1;
  let b := 1;
  -b/a = -1 :=
sorry

end sum_of_solutions_l364_364089


namespace lim_G_to_zero_l364_364816

noncomputable def G (r : ℝ) : ℝ :=
Inf { |r - (m^2 + 2 * n^2).sqrt | | m n : ℤ }

theorem lim_G_to_zero : ∀ ε > 0, ∃ R > 0, ∀ r > R, G r < ε :=
by
  sorry

end lim_G_to_zero_l364_364816


namespace rational_a_sqrt2_l364_364075

noncomputable theory

open_locale classical

variables (M : set ℝ) (hM : M ≠ ∅) (hM_card : M.card = 2003)

theorem rational_a_sqrt2 (h : ∀ x y ∈ M, x ≠ y → x^2 + y * real.sqrt 2 ∈ ℚ) :
  ∀ a ∈ M, a * real.sqrt 2 ∈ ℚ :=
begin
  sorry
end

end rational_a_sqrt2_l364_364075


namespace greatest_three_digit_multiple_23_l364_364655

theorem greatest_three_digit_multiple_23 : 
  ∃ n : ℕ, n < 1000 ∧ n % 23 = 0 ∧ (∀ m : ℕ, m < 1000 ∧ m % 23 = 0 → m ≤ n) ∧ n = 989 :=
sorry

end greatest_three_digit_multiple_23_l364_364655


namespace incenter_of_triangle_l364_364852

theorem incenter_of_triangle 
  (O O' : Type)
  [circle O] [circle O']
  (tangency_point : Point) 
  (arbitrary_point : Point)
  (chord_1 chord_2 : Line)
  (EF AO' : Line)
  (intersection_point : Point) 
  (touching_points_1 touching_points_2 : Point)
  (on_circle : arbitrary_point ∈ O)
  (touch_1 : chord_1 ∩ O' = touching_points_1)
  (touch_2 : chord_2 ∩ O' = touching_points_2)
  (EF_intersects_AO' : EF ∩ AO' = some intersection_point)
  : is_incenter_triangle intersection_point := 
sorry

end incenter_of_triangle_l364_364852


namespace bert_average_words_in_crossword_l364_364766

theorem bert_average_words_in_crossword :
  (10 * 35 + 4 * 65) / (10 + 4) = 43.57 :=
by
  sorry

end bert_average_words_in_crossword_l364_364766


namespace distinct_gcd_count_l364_364699

theorem distinct_gcd_count :
  ∃ (a b : ℕ), (gcd a b * Nat.lcm a b = 180) ∧
  (∀ (d : ℕ), d = gcd a b → 
    d ∈ {1, 2, 3, 5, 6, 10, 15, 30} ∧ 
    (∀ d' ∈ {1, 2, 3, 5, 6, 10, 15, 30}, d' ≠ d → 
      ∃ a' b', gcd a' b' * Nat.lcm a' b' = 180 ∧ gcd a' b' = d')) := sorry

end distinct_gcd_count_l364_364699


namespace number_of_real_solutions_l364_364423

theorem number_of_real_solutions : 
  set.countable {x : ℝ | ∃ n : ℕ, (sqrt (123 - sqrt x) = n)} = 12 :=
by
  sorry

end number_of_real_solutions_l364_364423


namespace forest_coverage_2009_min_annual_growth_rate_l364_364194

variables (a : ℝ)

-- Conditions
def initially_forest_coverage (a : ℝ) := a
def annual_natural_growth_rate := 0.02

-- Questions reformulated:
-- Part 1: Prove the forest coverage at the end of 2009
theorem forest_coverage_2009 : (∃ a : ℝ, (y : ℝ) = a * (1 + 0.02)^5 ∧ y = 1.104 * a) :=
by sorry

-- Part 2: Prove the minimum annual average growth rate by 2014
theorem min_annual_growth_rate : (∀ p : ℝ, (a : ℝ) * (1 + p)^10 ≥ 2 * a → p ≥ 0.072) :=
by sorry

end forest_coverage_2009_min_annual_growth_rate_l364_364194


namespace divisible_by_5_last_digit_l364_364783

theorem divisible_by_5_last_digit (d : ℕ) (h : d < 10) :
  (∃ k : ℕ, 41830 + d = 5 * k) ↔ (d = 0 ∨ d = 5) :=
by
  split
  {
    intro h_div,
    rcases h_div with ⟨k, hk⟩,
    have last_digit_div_by_5 : (41830 + d) % 5 = 0,
    {
      rw hk,
      exact Nat.mod_mul_left_div_self k 5,
    },
    show d = 0 ∨ d = 5,
    { sorry },
  },
  {
    rintro (rfl|rfl),
    {
      use 8366,
      norm_num,
    },
    {
      use 8367,
      norm_num,
    },
  }

end divisible_by_5_last_digit_l364_364783


namespace gilled_mushrooms_count_l364_364262

def mushrooms_problem (G S : ℕ) : Prop :=
  (S = 9 * G) ∧ (G + S = 30) → (G = 3)

-- The theorem statement corresponding to the problem
theorem gilled_mushrooms_count (G S : ℕ) : mushrooms_problem G S :=
by {
  sorry
}

end gilled_mushrooms_count_l364_364262


namespace combined_selling_price_l364_364735

theorem combined_selling_price (C_c : ℕ) (C_s : ℕ) (C_m : ℕ) (L_c L_s L_m : ℕ)
  (hc : C_c = 1600)
  (hs : C_s = 12000)
  (hm : C_m = 45000)
  (hlc : L_c = 15)
  (hls : L_s = 10)
  (hlm : L_m = 5) :
  85 * C_c / 100 + 90 * C_s / 100 + 95 * C_m / 100 = 54910 := by
  sorry

end combined_selling_price_l364_364735


namespace num_real_satisfying_x_l364_364441

theorem num_real_satisfying_x : 
  (∃ (x : ℝ), ∃ (s : ℕ) (hs : s ∈ set.Ico 0 (floor (sqrt 123) + 1)), 
   s = int.of_nat (floor (sqrt (123 - sqrt x)))) →
   12 := sorry

end num_real_satisfying_x_l364_364441


namespace odd_n_gt_one_not_divides_m_pow_n_minus_one_plus_one_l364_364025

theorem odd_n_gt_one_not_divides_m_pow_n_minus_one_plus_one
    (n : ℕ) (hn_odd: Odd n) (hn_gt_one : n > 1)
    (m : ℤ) :
    ¬ n ∣ (m ^ (n - 1) + 1) := 
sorry

end odd_n_gt_one_not_divides_m_pow_n_minus_one_plus_one_l364_364025


namespace domain_of_function_l364_364619

theorem domain_of_function :
  { x : ℝ | x - 2 ≥ 0 ∧ 4 - x > 0 } = set.Ico 2 4 :=
by
  sorry

end domain_of_function_l364_364619


namespace series_sum_is_correct_l364_364775

noncomputable def infinite_series_value : ℝ :=
  (∑' n : ℕ, if 2 ≤ n then (n^4 - 2*n^3 + 3*n^2 + 5*n + 20 : ℝ) / (3^n * (n^4 + 4*n^2 + 4)) else 0)

theorem series_sum_is_correct :
  infinite_series_value = 1 / 18 :=
begin
  -- Proof steps would go here
  sorry
end

end series_sum_is_correct_l364_364775


namespace apples_total_l364_364168

theorem apples_total (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apples_total_l364_364168


namespace new_stamps_ratio_l364_364628

theorem new_stamps_ratio (x : ℕ) (h1 : 7 * x = P) (h2 : 4 * x = Q)
  (h3 : P - 8 = 8 + (Q + 8)) : (P - 8) / gcd (P - 8) (Q + 8) = 6 ∧ (Q + 8) / gcd (P - 8) (Q + 8) = 5 :=
by
  sorry

end new_stamps_ratio_l364_364628


namespace coefficient_x_pow_7_l364_364207

theorem coefficient_x_pow_7 :
  ∃ c : Int, (∀ k : Nat, (x : Real) → coeff (expand_binom (x-2) 10) k = c → k = 7 ∧ c = -960) :=
by
  sorry

end coefficient_x_pow_7_l364_364207


namespace simplify_division_l364_364582

theorem simplify_division (a b c d : ℕ) (h1 : a = 27) (h2 : b = 10^12) (h3 : c = 9) (h4 : d = 10^4) :
  ((a * b) / (c * d) = 300000000) :=
by {
  sorry
}

end simplify_division_l364_364582


namespace cartesian_equation_of_curve_min_perimeter_and_coords_l364_364517

theorem cartesian_equation_of_curve (ρ θ x y : ℝ) (h : ρ^2 = 3 / (1 + 2 * (sin θ)^2)) 
    (hx : ρ * cos θ = x) (hy : ρ * sin θ = y) :
    x^2 / 3 + y^2 = 1 := sorry

theorem min_perimeter_and_coords (θ : ℝ) (P R Q : ℝ × ℝ) (hR : R = (2 * sqrt 2, π / 4)) 
    (hP : P = (sqrt 3 * cos θ, sin θ))
    (perimeter : ℝ) (h_perimeter : perimeter = 2 * (4 - sqrt 3 * cos θ - sin θ)) :
  ∃ θ_min, θ_min = π / 6 ∧ P = (3 / 2, 1 / 2) ∧ perimeter = 4 := sorry

end cartesian_equation_of_curve_min_perimeter_and_coords_l364_364517


namespace sequence_periodic_l364_364111

-- Define the sequence
def sequence (c : ℕ) : ℕ → ℕ
| 0 := 1
| (n + 1) := nat.divisors_count (sequence c n) + c

-- Auxiliary definition for periodicity
def periodic_after (k : ℕ) (f : ℕ → ℕ) := ∃ m, ∀ n ≥ m, f (n + k) = f n

-- Main theorem statement
theorem sequence_periodic (c : ℕ) (hc : c > 0) :
  ∃ k : ℕ, periodic_after k (sequence c) :=
sorry

end sequence_periodic_l364_364111


namespace owls_joined_l364_364258

theorem owls_joined (initial_owls : ℕ) (total_owls : ℕ) (join_owls : ℕ) 
  (h_initial : initial_owls = 3) (h_total : total_owls = 5) : join_owls = 2 :=
by {
  -- Sorry is used to skip the proof
  sorry
}

end owls_joined_l364_364258


namespace partition_possible_l364_364031

theorem partition_possible (n : ℕ) (h_even : n % 2 = 0) 
  (h_positive : 0 < n) (h_partition: ∃ (A B : Finset ℕ), 
    A ∪ B = Finset.range (n^2 + 1) ∧ 
    A ∩ B = ∅ ∧ 
    |A| = |B| ∧
    (∑ i in A, i) * 19 = (∑ i in B, i) * 20 ) :
  (∃ k : ℕ, n = 78 * k) ∨ 
  (∃ l : ℕ, n = 78 * l + 18) ∨ 
  (∃ l : ℕ, n = 78 * l + 60) :=
sorry

end partition_possible_l364_364031


namespace apple_count_l364_364184

-- Define constants for the problem
constant bags: ℕ → Prop
constant total_apples: ℕ
constant apples_in_bag1: ℕ := 12
constant apples_in_bag2: ℕ := 6

-- Define the conditions
axiom condition1 (n: ℕ) : bags n → (n = apples_in_bag1 ∨ n = apples_in_bag2)
axiom condition2 : 70 ≤ total_apples ∧ total_apples ≤ 80
axiom condition3 : total_apples % 6 = 0

-- State the proof problem
theorem apple_count :
  total_apples = 72 ∨ total_apples = 78 :=
sorry

end apple_count_l364_364184


namespace petya_vasya_game_l364_364054

noncomputable def smallest_k := 84

theorem petya_vasya_game :
  ∀ (k : ℕ),
    (∃ (marks : Finset (Fin 169)), marks.card = k ∧ 
    ∀ (rectangle : Finset (Fin 169)), rectangle.card = 6 → 
    ∃! (position : Finset (Fin 169)), 
    rectangle ⊆ marks ⊆ (Finset.range 169)) ↔ k ≥ smallest_k :=
by sorry

end petya_vasya_game_l364_364054


namespace circle_reflection_l364_364599

theorem circle_reflection (x y : ℝ) (hx : x = 8) (hy : y = -3) : 
  let reflected_x := -y, reflected_y := -x in
  reflected_x = 3 ∧ reflected_y = -8 :=
by
  sorry

end circle_reflection_l364_364599


namespace rectangle_sides_perpendicular_to_cone_axis_l364_364115

theorem rectangle_sides_perpendicular_to_cone_axis {S : Point} {A B C D : Point} (h : OnConeLateralSurface S {A, B, C, D} ∧ Rectangle {A, B, C, D}) : 
  ∃ l l', (Parallel l l') ∧ Perpendicular l (ConeAxis S) ∧ Perpendicular l' (ConeAxis S) := 
sorry

end rectangle_sides_perpendicular_to_cone_axis_l364_364115


namespace apple_bags_l364_364152

theorem apple_bags (n : ℕ) (h₁ : n ≥ 70) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 := 
sorry

end apple_bags_l364_364152


namespace range_of_m_l364_364038

noncomputable def f (m x : ℝ) : ℝ := real.sqrt 3 * real.sin (real.pi * x / m)

theorem range_of_m (m x₀ : ℝ) (k : ℤ) (h1 : f m x₀ = real.sqrt 3 ∨ f m x₀ = -real.sqrt 3)
  (h2 : x₀ = (2 * ↑k + 1) / 2 * m)
  (h3 : x₀^2 + (f m x₀)^2 < m^2) : m < -2 ∨ 2 < m :=
begin 
  sorry
end

end range_of_m_l364_364038


namespace valid_number_of_apples_l364_364134

theorem valid_number_of_apples (n : ℕ) : 
  (∃ k m : ℕ, n = 12 * k + 6 * m) ∧ (70 ≤ n ∧ n ≤ 80) ↔ (n = 72 ∨ n = 78) :=
by
  sorry

end valid_number_of_apples_l364_364134


namespace median_of_consecutive_sum_3125_l364_364113

theorem median_of_consecutive_sum_3125 : 
  ∃ median : ℤ, (25 * median = 5^5) ∧ median = 125 := 
by
  use 125
  split
  {
    calc 
      25 * 125 = 3125 : by norm_num,
      3125 = 5^5 : by norm_num
  }
  {
    exact rfl
  }

end median_of_consecutive_sum_3125_l364_364113


namespace num_real_x_l364_364410

theorem num_real_x (a b : ℝ) (h1 : a = 123) (h2 : b = 11) :
  ∃ n : ℕ, n = 12 ∧
  ∀ k : ℕ, k ≤ b → ∃ x : ℝ, x = (a - k^2)^2 :=
by
  sorry

end num_real_x_l364_364410


namespace expression_evaluation_l364_364317

theorem expression_evaluation :
  (real.cbrt (-8)) * ((-1)^(2023 : ℤ)) - 6 / 2 + (1 / 2)^0 = 0 :=
by
  sorry

end expression_evaluation_l364_364317


namespace triangle_area_l364_364738

theorem triangle_area (d : ℝ) (h : d = 8 * Real.sqrt 10) (ang : ∀ {α β γ : ℝ}, α = 45 ∨ β = 45 ∨ γ = 45) :
  ∃ A : ℝ, A = 160 :=
by
  sorry

end triangle_area_l364_364738


namespace geometric_sequence_a3a7_l364_364830

namespace geometric_seq

open_locale classical

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

variables {a : ℕ → ℝ} (h : is_geometric_sequence a) (h5 : a 5 = 4)

theorem geometric_sequence_a3a7 :
  a 3 * a 7 = 16 :=
begin
  obtain ⟨r, hr⟩ := h,
  have h5_def : a 5 = r^4 * a 1 := by {
    iterate 4 { rw hr },
    simp,
  },
  have h3_def : a 3 = r^2 * a 1 := by {
    iterate 2 { rw hr },
    simp,
  },
  have h7_def : a 7 = r^6 * a 1 := by {
    iterate 6 { rw hr },
    simp,
  },
  calc
    a 3 * a 7
        = (r^2 * a 1) * (r^6 * a 1) : by rw [h3_def, h7_def]
    ... = r^8 * (a 1)^2          : by ring
    ... = (r^4 * a 1)^2         : by { rw ←pow_mul, congr, ring }
    ... = (a 5)^2                : by rw [h5_def, h5],
  have : (a 5 <insert actual proof here> _, sorry,
end

end geometric_seq

end geometric_sequence_a3a7_l364_364830


namespace problem_part1_problem_part2_l364_364337

noncomputable def f (x : ℝ) : ℝ := |x - 1| - |x + 3|

theorem problem_part1 (x : ℝ) :
  f(x) > 2 ↔ x < -2 :=
sorry

theorem problem_part2 (k : ℝ) :
  (∀ x ∈ set.Icc (-3 : ℝ) (-1), f(x) ≤ k * x + 1) ↔ k ≤ -1 :=
sorry

end problem_part1_problem_part2_l364_364337


namespace sum_factors_of_30_l364_364677

theorem sum_factors_of_30 : (1 + 2 + 3 + 5 + 6 + 10 + 15 + 30) = 72 :=
by
  sorry

end sum_factors_of_30_l364_364677


namespace last_person_perform_32_last_person_perform_39_l364_364587

theorem last_person_perform_32 (n : ℕ) (h : n = 32) : last_person_to_perform n = 32 :=
by {
  sorry
}

theorem last_person_perform_39 (n : ℕ) (h : n = 39) : last_person_to_perform n = 12 :=
by {
  sorry
}

end last_person_perform_32_last_person_perform_39_l364_364587


namespace num_real_satisfying_x_l364_364442

theorem num_real_satisfying_x : 
  (∃ (x : ℝ), ∃ (s : ℕ) (hs : s ∈ set.Ico 0 (floor (sqrt 123) + 1)), 
   s = int.of_nat (floor (sqrt (123 - sqrt x)))) →
   12 := sorry

end num_real_satisfying_x_l364_364442


namespace factorial_divisibility_l364_364895

theorem factorial_divisibility {p : ℕ} (hp : 1 < p) : (p^2)! % (p!)^(p+1) = 0 :=
by
  sorry

end factorial_divisibility_l364_364895


namespace line_intersects_y_axis_at_point_l364_364734

theorem line_intersects_y_axis_at_point :
  let x1 := 3
  let y1 := 20
  let x2 := -7
  let y2 := 2

  -- line equation from 2 points: y - y1 = m * (x - x1)
  -- slope m = (y2 - y1) / (x2 - x1)
  -- y-intercept when x = 0:
  
  (0, 14.6) ∈ { p : ℝ × ℝ | ∃ m b, p.2 = m * p.1 + b ∧ 
    m = (y2 - y1) / (x2 - x1) ∧ 
    b = y1 - m * x1 }
  :=
  sorry

end line_intersects_y_axis_at_point_l364_364734


namespace inequality_for_real_numbers_l364_364965

theorem inequality_for_real_numbers (x : ℕ → ℝ) (n : ℕ) (h1 : n ≥ 2) (h2 : ∀ i, 1 ≤ i ∧ i < n → x i > 1) (h3 : ∀ i, 1 ≤ i ∧ i < n → |x i - x (i+1)| < 1) :
  (list.sum (list.map (λ i, x i / x (i+1)) (list.range (n-1)) ++ [x (n-1) / x 0])) < 2 * n - 1 :=
by
  sorry

end inequality_for_real_numbers_l364_364965


namespace correct_time_fraction_is_11_over_15_l364_364278

/-!
  We define the conditions where a digital clock displays the time incorrectly:
  - The clock mistakenly displays a 5 whenever it should display a 2.
  - The hours range from 1 to 12.
  - The minutes range from 00 to 59.
-/

def correct_hours_fraction : ℚ :=
  11 / 12  -- 11 correct hours out of 12

def correct_minutes_fraction : ℚ :=
  11 / 15  -- 44 correct minutes out of 60, simplified to 11/15

def correct_time_fraction : ℚ :=
  correct_hours_fraction * correct_minutes_fraction

theorem correct_time_fraction_is_11_over_15 :
  correct_time_fraction = 11 / 15 :=
by
  -- this is the statement we're proving, proof is omitted
  sorry

end correct_time_fraction_is_11_over_15_l364_364278


namespace circle_reflection_l364_364601

theorem circle_reflection (x y : ℝ) (hx : x = 8) (hy : y = -3) : 
  let reflected_x := -y, reflected_y := -x in
  reflected_x = 3 ∧ reflected_y = -8 :=
by
  sorry

end circle_reflection_l364_364601


namespace pyramid_volume_regular_hexagon_isosceles_rt_triangle_l364_364740

theorem pyramid_volume_regular_hexagon_isosceles_rt_triangle 
  (ABCDEF : Type) 
  (P : Type) 
  (PA DA : Type) 
  (h_base : regular_hexagon ABCDEF)
  (h_isosceles_rt : is_isosceles_right_triangle P A D) 
  (PA_eq_10 : PA.length = 10) 
  (DA_eq_10 : DA.length = 10) : 
  volume (pyramid P ABCDEF) = 125 * real.sqrt 6 :=
by sorry

end pyramid_volume_regular_hexagon_isosceles_rt_triangle_l364_364740


namespace total_apples_l364_364164

theorem total_apples (n : ℕ) (h1 : ∀ k : ℕ, 6 * k = n ∨ 12 * k = n) (h2 : 70 ≤ n) (h3 : n ≤ 80) : 
  n = 72 ∨ n = 78 := 
sorry

end total_apples_l364_364164


namespace probability_log_in_interval_l364_364384

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 8}

lemma interval_length (a b : ℝ) : MeasurableSet (Set.Icc a b) → ℝ :=
λ (h : MeasurableSet (Set.Icc a b)), b - a

theorem probability_log_in_interval :
  (interval_length A (IsMeasurableSet.interval.left 2 4)) / 
  (interval_length B (IsMeasurableSet.interval.left 1 8)) = 2 / 7 :=
sorry

end probability_log_in_interval_l364_364384


namespace prove_angle_sum_l364_364836

open Real

theorem prove_angle_sum (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : cos α / sin β + cos β / sin α = 2) : 
  α + β = π / 2 := 
sorry

end prove_angle_sum_l364_364836


namespace counterfeit_coin_3_coins_counterfeit_coin_4_coins_counterfeit_coin_9_coins_l364_364708

/-- Given a balance scale and a set of 3 coins where exactly one coin is counterfeit,
prove that 2 weighings are sufficient to identify the counterfeit coin. -/
theorem counterfeit_coin_3_coins : ∃ (n : ℕ), n = 2 ∧ ∀ (coins : Fin 3 → ℤ), (∃! c, coins c ≠ 0) → n weighings_sufficient coins :=
by sorry

/-- Given a balance scale and a set of 4 coins where exactly one coin is counterfeit,
prove that 2 weighings are sufficient to identify the counterfeit coin. -/
theorem counterfeit_coin_4_coins : ∃ (n : ℕ), n = 2 ∧ ∀ (coins : Fin 4 → ℤ), (∃! c, coins c ≠ 0) → n weighings_sufficient coins :=
by sorry

/-- Given a balance scale and a set of 9 coins where exactly one coin is counterfeit,
prove that 3 weighings are sufficient to identify the counterfeit coin. -/
theorem counterfeit_coin_9_coins : ∃ (n : ℕ), n = 3 ∧ ∀ (coins : Fin 9 → ℤ), (∃! c, coins c ≠ 0) → n weighings_sufficient coins :=
by sorry

end counterfeit_coin_3_coins_counterfeit_coin_4_coins_counterfeit_coin_9_coins_l364_364708


namespace number_of_adults_in_family_l364_364730

def regular_ticket_cost := 9
def child_ticket_discount := 2
def child_ticket_cost := regular_ticket_cost - child_ticket_discount
def num_children := 3
def payment_amount := 2 * 20
def change_received := 1
def total_ticket_cost := payment_amount - change_received
def children_total_cost := num_children * child_ticket_cost
def adult_ticket_cost := regular_ticket_cost

theorem number_of_adults_in_family 
  (regular_ticket_cost = 9) 
  (child_ticket_cost = regular_ticket_cost - 2) 
  (num_children = 3) 
  (payment_amount = 2 * 20) 
  (change_received = 1) 
  (total_ticket_cost = payment_amount - change_received) 
  (children_total_cost = num_children * child_ticket_cost)
  (adult_ticket_cost = regular_ticket_cost) :
  (total_ticket_cost - children_total_cost) / adult_ticket_cost = 2 :=
by sorry

end number_of_adults_in_family_l364_364730


namespace apple_count_l364_364186

-- Define constants for the problem
constant bags: ℕ → Prop
constant total_apples: ℕ
constant apples_in_bag1: ℕ := 12
constant apples_in_bag2: ℕ := 6

-- Define the conditions
axiom condition1 (n: ℕ) : bags n → (n = apples_in_bag1 ∨ n = apples_in_bag2)
axiom condition2 : 70 ≤ total_apples ∧ total_apples ≤ 80
axiom condition3 : total_apples % 6 = 0

-- State the proof problem
theorem apple_count :
  total_apples = 72 ∨ total_apples = 78 :=
sorry

end apple_count_l364_364186


namespace sum_log_terms_l364_364510

-- Define the conditions for the geometric sequence 
def geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∀ n, a n > 0 ∧ ∃ r, ∀ m, a (m + 1) = a m * r

-- Define the sequence log2(a_n)
def log_sequence (a : ℕ → ℝ) := λ n, Real.logBase 2 (a n)

theorem sum_log_terms 
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a3a5 : a 3 * a 5 = 4) :
  (Finset.range 7).sum (λ n => log_sequence a n) = 7 := 
sorry

end sum_log_terms_l364_364510


namespace polynomial_irreducible_l364_364964

def is_irreducible_ZX (p q : ℕ) (n : ℕ) (P : Polynomial ℤ) : Prop :=
  irreducible P

theorem polynomial_irreducible (p q : ℕ) (h1 : Prime p) (h2 : Prime q) (h3 : p ≠ q) (n : ℕ) (h4 : n ≥ 5) :
  is_irreducible_ZX p q n (Polynomial.Coeff ℤ X ^ n + Polynomial.Coeff ℤ (p + q) * X ^ (n - 2) - Polynomial.Coeff ℤ p * Polynomial.Coeff ℤ q) :=
sorry

end polynomial_irreducible_l364_364964


namespace how_many_imply_l364_364779

variables (p q r : Prop)

def implies_statement_1 : Prop := (¬ p ∧ ¬ r ∧ q) → ((p ∧ q) → ¬ r) 
def implies_statement_2 : Prop := (p ∧ ¬ r ∧ ¬ q) → ((p ∧ q) → ¬ r) 
def implies_statement_3 : Prop := (¬ p ∧ r ∧ q) → ((p ∧ q) → ¬ r) 
def implies_statement_4 : Prop := (p ∧ r ∧ ¬ q) → ((p ∧ q) → ¬ r)

theorem how_many_imply : 4 = (if implies_statement_1 p q r ∧ implies_statement_2 p q r ∧ implies_statement_3 p q r ∧ implies_statement_4 p q r then 4 else 0) :=
sorry

end how_many_imply_l364_364779


namespace vector_addition_proof_l364_364340

-- Definitions as conditions for the problem
def u : ℝ^3 := ⟨3, -2, 5⟩
def v : ℝ^3 := ⟨-1, 4, -3⟩

-- The theorem to be proven based on conditions
theorem vector_addition_proof : u + 2 • v = ⟨1, 6, -1⟩ := 
by {
  sorry
}

end vector_addition_proof_l364_364340


namespace coeff_x7_in_expansion_l364_364213

-- Each definition in Lean 4 statement reflects the conditions of the problem.
def binomial_coefficient (n k : ℕ) : ℕ := nat.choose n k

-- The condition for expansion using Binomial Theorem
def binomial_expansion_term (n k : ℕ) (a x : ℤ) : ℤ :=
  binomial_coefficient n k * a ^ (n - k) * x ^ k

-- Prove that the coefficient of x^7 in the expansion of (x - 2)^{10} is -960
theorem coeff_x7_in_expansion : 
  binomial_coefficient 10 3 * (-2) ^ 3 = -960 := 
sorry

end coeff_x7_in_expansion_l364_364213


namespace greatest_three_digit_multiple_of_23_l364_364662

def is_three_digit (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

def is_multiple_of_23 (n : ℕ) : Prop :=
  n % 23 = 0

theorem greatest_three_digit_multiple_of_23 :
  ∀ n, is_three_digit n ∧ is_multiple_of_23 n → n ≤ 989 :=
by
  sorry

end greatest_three_digit_multiple_of_23_l364_364662


namespace probability_of_A_east_of_B_and_C_l364_364935

-- Define the facts about the triangle and the problem conditions
def triangle_ABC : Type := 
  {A B C : Point} 
  (angle_A_40 : angle A B C = 40)

-- Define the probability calculation
def probability_A_east_given_angle_40 
  (t : triangle_ABC) : ℚ :=
  7 / 18

-- The theorem statement
theorem probability_of_A_east_of_B_and_C 
  (t : triangle_ABC) : 
  probability_A_east_given_angle_40 t = 7 / 18 := 
  sorry

end probability_of_A_east_of_B_and_C_l364_364935


namespace parabola_b_value_l364_364103

noncomputable def find_b (a h : ℝ) (h_ne_zero : h ≠ 0) : ℝ :=
  let c := -2 * h
  a * h^2 + 3 * h = 0 → b = 6

theorem parabola_b_value (a h : ℝ) (h_ne_zero : h ≠ 0) (vertex_eq : ∀ x : ℝ, y = a * (x - h)^2 + h)
  (y_intercept_eq : y = -2 * h) : 
  find_b a h h_ne_zero = 6 :=
  by
    sorry

end parabola_b_value_l364_364103


namespace largest_multiple_l364_364670

theorem largest_multiple (n : ℤ) (h8 : 8 ∣ n) (h : -n > -80) : n = 72 :=
by 
  sorry

end largest_multiple_l364_364670


namespace math_problem_proof_l364_364206

open Nat

noncomputable def a := 3
noncomputable def bc := 44
noncomputable def def := 149

-- Conditions derived from the problem
def is_even (n : ℕ) := n % 2 = 0
def is_odd (n : ℕ) := n % 2 = 1
def is_perfect_square (n : ℕ) := ∃ k : ℕ, k * k = n

-- Problem translated into Lean 4 statement
theorem math_problem_proof :
  is_odd a ∧ is_even bc ∧ is_odd def ∧
  (a * bc * def) % 100 = 68 ∧
  is_perfect_square (a + bc + def) ∧
  (is_perfect_square (100 * (def / 100) + bc) ∧ is_perfect_square (def % 100)) :=
by
  sorry

end math_problem_proof_l364_364206


namespace f_at_neg_pi_over_6_f_range_l364_364862

noncomputable def f : ℝ → ℝ := λ x, Real.cos (2 * x) + 2 * Real.sin x

theorem f_at_neg_pi_over_6 : f (-Real.pi / 6) = -1 / 2 := by
  sorry

theorem f_range : Set.Icc (-3 : ℝ) (3 / 2) = Set.range f := by
  sorry

end f_at_neg_pi_over_6_f_range_l364_364862


namespace valid_number_of_apples_l364_364133

theorem valid_number_of_apples (n : ℕ) : 
  (∃ k m : ℕ, n = 12 * k + 6 * m) ∧ (70 ≤ n ∧ n ≤ 80) ↔ (n = 72 ∨ n = 78) :=
by
  sorry

end valid_number_of_apples_l364_364133


namespace suitcase_lock_settings_l364_364745

theorem suitcase_lock_settings : 
  (∃ dials : ℕ → ℕ, 
     (∀ i < 4, dials i ∈ (Finset.range 10)) ∧ 
     (∀ i < 3, dials i ≠ dials (i+1))
  ) →
  (Finset.card 
      (Finset.filter 
         (λ f, ∀ i < 3, f i ≠ f (i+1))
         (Finset.univ : Finset (ℕ → ℕ))) = 7290) :=
by
  sorry

end suitcase_lock_settings_l364_364745


namespace num_ordered_pairs_of_squares_difference_96_l364_364403

theorem num_ordered_pairs_of_squares_difference_96 :
    ∃ (n : ℕ), n = 4 ∧ ∀ m n : ℕ, 0 < m ∧ 0 < n ∧ m ≥ n ∧ (m * m - n * n) = 96 ↔ 
    (m, n) = (25, 23) ∨ (m, n) = (14, 10) ∨ (m, n) = (11, 5) ∨ (m, n) = (10, 2) :=
begin
  sorry
end

end num_ordered_pairs_of_squares_difference_96_l364_364403


namespace number_of_girls_in_school_l364_364239

theorem number_of_girls_in_school : 
  ∀ (B G : ℕ), 
  (B + G = 632) → 
  (12 * B + 11 * G = 11.75 * 632) → 
  G = 156 :=
by
  intros B G h1 h2
  sorry

end number_of_girls_in_school_l364_364239


namespace find_f_value_mono_increasing_interval_symmetry_axis_eq_l364_364385

def f (ω : ℝ) (x : ℝ) := (1/2) * Real.cos (2 * ω * x) + Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x)

theorem find_f_value (ω : ℝ) (hx : ω = 1) : f ω (2 * Real.pi / 3) = -1 := sorry

theorem mono_increasing_interval (k : ℤ) : 
  ∀ x, k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 → 
  ∃ y, ∀ z, f 1 z = f 1 y → (k * Real.pi - Real.pi / 3 ≤ z ∧ z ≤ k * Real.pi + Real.pi / 6) := sorry

theorem symmetry_axis_eq (k : ℤ) : ∃ x, x = (k/2) * Real.pi + Real.pi / 6 := sorry

end find_f_value_mono_increasing_interval_symmetry_axis_eq_l364_364385


namespace point_outside_circle_l364_364377

theorem point_outside_circle
  (radius : ℝ) (distance : ℝ) (h_radius : radius = 8) (h_distance : distance = 10) :
  distance > radius :=
by sorry

end point_outside_circle_l364_364377


namespace number_of_real_solutions_l364_364425

theorem number_of_real_solutions : 
  set.countable {x : ℝ | ∃ n : ℕ, (sqrt (123 - sqrt x) = n)} = 12 :=
by
  sorry

end number_of_real_solutions_l364_364425


namespace number_of_intersections_l364_364784

/-- 
  Define the two curves as provided in the problem:
  curve1 is defined by the equation 3x² + 2y² = 6,
  curve2 is defined by the equation x² - 2y² = 1.
  We aim to prove that there are exactly 4 distinct intersection points.
--/
def curve1 (x y : ℝ) : Prop := 3 * x^2 + 2 * y^2 = 6

def curve2 (x y : ℝ) : Prop := x^2 - 2 * y^2 = 1

theorem number_of_intersections : ∃ (points : Finset (ℝ × ℝ)), (∀ p ∈ points, curve1 p.1 p.2 ∧ curve2 p.1 p.2) ∧ points.card = 4 :=
sorry

end number_of_intersections_l364_364784


namespace probability_A_east_lemma_l364_364932

noncomputable def probability_A_east {α β γ : ℕ} (hα : α = 40) (hβγ : β + γ = 180 - α) : ℚ :=
  140 / 360

theorem probability_A_east_lemma {α β γ : ℕ} 
  (hα : α = 40)
  (hβγ : β + γ = 180 - α) :
  probability_A_east hα hβγ = 7 / 18 :=
by
  unfold probability_A_east
  rw [hα]
  norm_num
  sorry

end probability_A_east_lemma_l364_364932


namespace problem_statement_l364_364519

noncomputable theory

structure Point where
  x : ℝ
  y : ℝ

def line_l (t : ℝ) : Point := 
  { x := 1 + (3 / 5) * t, y := 1 + (4 / 5) * t }

def curve_C (x y : ℝ) : Prop := 
  (x^2 / 2) + y^2 = 1

def point_P : Point :=
  { x := 1, y := 1 }

def distance (P M : Point) : ℝ := 
  real.sqrt ((P.x - M.x)^2 + (P.y - M.y)^2)

theorem problem_statement (t t1 t2 : ℝ) :
  curve_C (point_P.x) (point_P.y) ∧
  line_l t = { x := 1 + (3 / 5) * t, y := 1 + (4 / 5) * t } ∧ 
  (curve_C (line_l t1).x (line_l t1).y ∧ curve_C (line_l t2).x (line_l t2).y ∧ 
    let M : Point := { x := (line_l t1).x / 2 + (line_l t2).x / 2, y := (line_l t1).y / 2 + (line_l t2).y / 2 }
    in distance point_P M = 55 / 41) := 
sorry

end problem_statement_l364_364519


namespace circle_reflection_l364_364602

variable (x₀ y₀ : ℝ)

def reflect_over_line_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem circle_reflection 
  (h₁ : x₀ = 8)
  (h₂ : y₀ = -3) :
  reflect_over_line_y_eq_neg_x (x₀, y₀) = (3, -8) := by
  sorry

end circle_reflection_l364_364602


namespace fraction_of_selected_films_in_color_l364_364234

variables (x y : ℕ)

theorem fraction_of_selected_films_in_color (B C : ℕ) (e : ℚ)
  (h1 : B = 20 * x)
  (h2 : C = 6 * y)
  (h3 : e = (6 * y : ℚ) / (((y / 5 : ℚ) + 6 * y))) :
  e = 30 / 31 :=
by {
  sorry
}

end fraction_of_selected_films_in_color_l364_364234


namespace black_white_sums_equal_l364_364567

theorem black_white_sums_equal (n : ℕ) :
  let grid := (fin (2*n) → fin (2*n) → ℕ) -- 2n x 2n grid
  let cell_numbering := λ (r : fin (2 * n)) (c : fin (2 * n)), 1 + r.val * 2 * n + c.val
  let black_cells : fin (2*n) → fin (2*n) → Prop := sorry -- user defines half cells black in each row and column
  let white_cells : fin (2*n) → fin (2*n) → Prop := sorry -- user defines remaining cells white
  ∀ f : fin (2*n) → fin (2*n) → ℕ, -- cell (i,j) numbering function
    (∀ i j, f i j = cell_numbering i j) ∧ (∀ i, (∀ j, black_cells i j ↔ ¬white_cells i j) 
    ∧ (finset.card (finset.univ.filter (black_cells i)) = n) ∧ (finset.card (finset.univ.filter (white_cells i)) = n)) →
  (finset.sum (finset.univ.filter (λ ⟨i, j⟩, black_cells i j)) (λ ⟨i, j⟩, f i j) = 
   finset.sum (finset.univ.filter (λ ⟨i, j⟩, white_cells i j)) (λ ⟨i, j⟩, f i j)) :=
sorry

end black_white_sums_equal_l364_364567


namespace simplify_trig_l364_364994

open Real

theorem simplify_trig : 
  (sin (30 * pi / 180) + sin (60 * pi / 180)) / (cos (30 * pi / 180) + cos (60 * pi / 180)) = tan (45 * pi / 180) :=
by
  sorry

end simplify_trig_l364_364994


namespace dice_product_composite_probability_l364_364479

theorem dice_product_composite_probability :
  (let total_outcomes := (6 : ℕ)^4,
       non_composite_outcomes := 1,
       composite_probability := 1 - (non_composite_outcomes / total_outcomes) in
   composite_probability = 1295 / 1296) :=
by sorry

end dice_product_composite_probability_l364_364479


namespace train_speed_l364_364290

theorem train_speed (time_seconds : ℕ) (length_meters : ℕ) (speed_kmph : ℕ)
  (h1 : time_seconds = 9) (h2 : length_meters = 135) : speed_kmph = 54 :=
sorry

end train_speed_l364_364290


namespace three_digit_numbers_not_multiple_of_5_or_7_l364_364454

theorem three_digit_numbers_not_multiple_of_5_or_7 : 
  (∑ n in (finset.range 900).filter (λ n, n + 100), (¬ (n + 100) % 5 = 0) ∧ ¬ (n + 100) % 7 = 0).card = 618 :=
sorry

end three_digit_numbers_not_multiple_of_5_or_7_l364_364454


namespace hyperbola_params_l364_364961

-- Define the primary conditions
def F1 : (ℝ × ℝ) := (2, -1)
def F2 : (ℝ × ℝ) := (2, 3)
def d : ℝ := 2

-- Define any needed calculations or parameters
def c : ℝ := dist F1 F2 / 2

noncomputable def a : ℝ := d / 2
noncomputable def h : ℝ := (F1.1 + F2.1) / 2
noncomputable def k : ℝ := (F1.2 + F2.2) / 2
noncomputable def b : ℝ := sqrt (c^2 - a^2)

-- Prove the target equation
theorem hyperbola_params : h + k + a + b = 4 + sqrt 3 :=
by
  sorry

end hyperbola_params_l364_364961


namespace apple_bags_l364_364150

theorem apple_bags (n : ℕ) (h₁ : n ≥ 70) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 := 
sorry

end apple_bags_l364_364150


namespace probability_of_odd_l364_364222

def odds := {3, 1, 7, 9}
def total_sections := 6

theorem probability_of_odd :
  (set.size odds) / total_sections = 2 / 3 :=
by
  sorry

end probability_of_odd_l364_364222


namespace monotonic_increasing_interval_l364_364099

def f (x : ℝ) : ℝ := (27 / 2) * x^2 + 1 / x

theorem monotonic_increasing_interval :
  ∀ x : ℝ, x > 1/3 → ((27 * x^3 - 1) / x^2) > 0 :=
by
  assume x,
  assume hx : x > 1/3,
  have derivative := (27 * x^3 - 1) / x^2,
  sorry

end monotonic_increasing_interval_l364_364099


namespace eggs_left_on_shelf_l364_364041

-- Define the conditions as variables in the Lean statement
variables (x y z : ℝ)

-- Define the final theorem statement
theorem eggs_left_on_shelf (hx : 0 ≤ x) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) :
  x * (1 - y) - z = (x - y * x) - z :=
by
  sorry

end eggs_left_on_shelf_l364_364041


namespace age_difference_l364_364913

theorem age_difference (B_age : ℕ) (A_age : ℕ) (X : ℕ) : 
  B_age = 42 → 
  A_age = B_age + 12 → 
  A_age + 10 = 2 * (B_age - X) → 
  X = 10 :=
by
  intros hB_age hA_age hEquation 
  -- define variables based on conditions
  have hB : B_age = 42 := hB_age
  have hA : A_age = B_age + 12 := hA_age
  have hEq : A_age + 10 = 2 * (B_age - X) := hEquation
  -- expected result
  sorry

end age_difference_l364_364913


namespace largest_reservoir_is_D_l364_364124

variables (a : ℝ) 
def final_amount_A : ℝ := a * (1 + 0.1) * (1 - 0.05)
def final_amount_B : ℝ := a * (1 + 0.09) * (1 - 0.04)
def final_amount_C : ℝ := a * (1 + 0.08) * (1 - 0.03)
def final_amount_D : ℝ := a * (1 + 0.07) * (1 - 0.02)

theorem largest_reservoir_is_D
  (hA : final_amount_A a = a * 1.045)
  (hB : final_amount_B a = a * 1.0464)
  (hC : final_amount_C a = a * 1.0476)
  (hD : final_amount_D a = a * 1.0486) :
  final_amount_D a > final_amount_A a ∧ 
  final_amount_D a > final_amount_B a ∧ 
  final_amount_D a > final_amount_C a :=
by sorry

end largest_reservoir_is_D_l364_364124


namespace total_apples_l364_364163

theorem total_apples (n : ℕ) (h1 : ∀ k : ℕ, 6 * k = n ∨ 12 * k = n) (h2 : 70 ≤ n) (h3 : n ≤ 80) : 
  n = 72 ∨ n = 78 := 
sorry

end total_apples_l364_364163


namespace coeff_x7_expansion_l364_364211

theorem coeff_x7_expansion : 
  let expansion := (∑ k in Finset.range 11, (Nat.choose 10 k) * x^(10 - k) * (-2)^k)
  ∃ coeff : ℤ, 
  (coeff * x^7 ∈ expansion) ∧ coeff = -960 :=
begin
  let expansion := (∑ k in Finset.range 11, (Nat.choose 10 k) * x^(10 - k) * (-2)^k),
  use -960,
  split,
  { sorry, },
  { reflexivity, }
end

end coeff_x7_expansion_l364_364211


namespace percentage_decrease_l364_364636

theorem percentage_decrease (S0 S2 : ℝ) (d : ℝ) (h1 : S0 = 3000) (h2 : S2 = 3135) :
  let S1 := S0 * 1.1 in
  S2 = S1 * (1 - d / 100) → d = 5 :=
by
  intro h3
  have h1' : S0 = 3000 := h1
  have h2' : S2 = 3135 := h2
  let S1 := S0 * 1.1
  have hS1 : S1 = 3300 := by rw [h1]; norm_num
  rw [hS1] at h3
  simp at h3
  field_simp at h3
  norm_num at h3
  exact h3

end percentage_decrease_l364_364636


namespace fraction_done_by_B_l364_364462

theorem fraction_done_by_B {A B : ℝ} (h : A = (2/5) * B) : (B / (A + B)) = (5/7) :=
by
  sorry

end fraction_done_by_B_l364_364462


namespace trivia_team_points_l364_364299

theorem trivia_team_points 
    (total_members : ℕ) 
    (members_absent : ℕ) 
    (total_points : ℕ) 
    (members_present : ℕ := total_members - members_absent) 
    (points_per_member : ℕ := total_points / members_present) 
    (h1 : total_members = 7) 
    (h2 : members_absent = 2) 
    (h3 : total_points = 20) : 
    points_per_member = 4 :=
by
    sorry

end trivia_team_points_l364_364299


namespace regular_tetrahedron_vertices_edges_l364_364630

theorem regular_tetrahedron_vertices_edges :
  ∃ (v e : ℕ), v = 4 ∧ e = 6 ∧ is_regular_tetrahedron v e :=
sorry

end regular_tetrahedron_vertices_edges_l364_364630


namespace cone_volume_correct_l364_364108

def cone_volume (R : ℝ) (H : ℝ) : ℝ := (1 / 3) * Mathlib.pi * R^2 * H

noncomputable def cone_height (R : ℝ) : ℝ := 
  (2 * Mathlib.pi * R) / (Mathlib.pi^2 - 1)

theorem cone_volume_correct (R : ℝ) (H : ℝ) :
  let S1 := R * H in
  let S2 := Mathlib.pi * R^2 in
  let S3 := Mathlib.pi * R * (Mathlib.sqrt (H^2 + R^2)) in
  S3 = S1 + S2 →
  cone_volume R (cone_height R) = (2 * Mathlib.pi^2 * R^3) / (3 * (Mathlib.pi^2 - 1)) :=
by
  sorry

end cone_volume_correct_l364_364108


namespace trigonometric_identity_l364_364996

theorem trigonometric_identity :
  let sin_30 := 1 / 2,
      sin_60 := Real.sqrt 3 / 2,
      cos_30 := Real.sqrt 3 / 2,
      cos_60 := 1 / 2,
      sin_45 := Real.sqrt 2 / 2,
      cos_45 := Real.sqrt 2 / 2 in
  (sin_30 + sin_60) / (cos_30 + cos_60) = Real.tan (Real.pi / 4) := 
  by
    sorry

end trigonometric_identity_l364_364996


namespace technician_percent_drive_from_center_l364_364288

-- Define the problem conditions:
def distance_to_center (D : ℝ) := D
def round_trip_distance (D : ℝ) := 2 * D
def distance_driven (D : ℝ) := 0.55 * (2 * D)

-- Define the question:
def percent_drive_from_center (D : ℝ) := ((distance_driven D - distance_to_center D) / distance_to_center D) * 100

-- Prove the correct answer:
theorem technician_percent_drive_from_center : ∀ (D : ℝ), percent_drive_from_center D = 10 :=
by
  intro D
  unfold percent_drive_from_center distance_driven distance_to_center round_trip_distance
  sorry

end technician_percent_drive_from_center_l364_364288


namespace partition_count_l364_364790

theorem partition_count (n : ℕ) :
  (∃ A1 A2 A3 : set ℕ, A1 ∪ A2 ∪ A3 = {i | 1 ≤ i ∧ i ≤ n} ∧
     (∀ x ∈ A1, ∀ y ∈ A1, x < y → (even x ↔ ¬even y)) ∧
     (∀ x ∈ A2, ∀ y ∈ A2, x < y → (even x ↔ ¬even y)) ∧
     (∀ x ∈ A3, ∀ y ∈ A3, x < y → (even x ↔ ¬even y)) ∧
     (A1 ≠ ∅ ∧ A2 ≠ ∅ ∧ A3 ≠ ∅ →
       (∃ s ∈ A1 ∪ A2 ∪ A3, even s ∧ (∀ t ∈ A1 ∪ A2 ∪ A3, s ≤ t → ¬even t))
     )) →
  ({A1 A2 A3 : set ℕ // A1 ∪ A2 ∪ A3 = {i | 1 ≤ i ∧ i ≤ n} ∧
     (∀ x ∈ A1, ∀ y ∈ A1, x < y → (even x ↔ ¬even y)) ∧
     (∀ x ∈ A2, ∀ y ∈ A2, x < y → (even x ↔ ¬even y)) ∧
     (∀ x ∈ A3, ∀ y ∈ A3, x < y → (even x ↔ ¬even y)) ∧
     (A1 ≠ ∅ ∧ A2 ≠ ∅ ∧ A3 ≠ ∅ →
       (∃ s ∈ A1 ∪ A2 ∪ A3, even s ∧ (∀ t ∈ A1 ∪ A2 ∪ A3, s ≤ t → ¬even t))
     }) // count = 2^(n-1)) sorry.

end partition_count_l364_364790


namespace incorrect_statement_l364_364231

def P_K2_ge_6_635 : Prop := P (K^2 ≥ 6.635) ≈ 0.01
def regression_line (x : ℝ) : ℝ := 0.5 * x - 85

theorem incorrect_statement :
  ¬(∀ x : ℝ, regression_line 200 = 15) :=
by
  sorry

end incorrect_statement_l364_364231


namespace find_point_of_intersection_l364_364638
noncomputable def point_of_intersection_curve_line : Prop :=
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^y = y^x ∧ y = x ∧ x = Real.exp 1 ∧ y = Real.exp 1

theorem find_point_of_intersection : point_of_intersection_curve_line :=
sorry

end find_point_of_intersection_l364_364638


namespace sum_of_integers_satisfying_inequality_l364_364813

theorem sum_of_integers_satisfying_inequality : 
  (∑ n in Finset.filter (λ n : ℕ, 1.5 * n - 6.3 < 4.5) (Finset.range 100), n) = 21 := 
by
  sorry

end sum_of_integers_satisfying_inequality_l364_364813


namespace polar_equation_of_line_l364_364104

-- Define the point A(5, 0)
def point_A := (5 : ℝ, 0 : ℝ)

-- Define the line alpha = π/4 and its perpendicular condition
def angle_alpha := (Real.pi / 4 : ℝ)
def perpendicular_line : ℝ → ℝ := λ x, -x + 5

-- Define the polar equation to be proved
theorem polar_equation_of_line :
  (λ (ρ θ : ℝ), ρ * Real.sin (angle_alpha + θ) = (5 * Real.sqrt 2) / 2) :=
sorry

end polar_equation_of_line_l364_364104


namespace problem_statement_l364_364237

-- Definitions for the operations as given in the problem
def at_op (a b : ℕ) : ℝ := (a^b : ℝ) / 2 + Real.log2 (Nat.factorial b)
def hash_op (a b c : ℕ) : ℝ := ((at_op a b)^c) / 4 + Real.log2 (Nat.factorial c)

-- The theorem that needs to be proved
theorem problem_statement : hash_op 3 2 4 = 232.765625 + Real.log2 (1.5) := 
by
  sorry

end problem_statement_l364_364237


namespace sides_and_diagonals_l364_364378

def number_of_sides_of_polygon (n : ℕ) :=
  180 * (n - 2) = 360 + (1 / 4 : ℤ) * 360

def number_of_diagonals_of_polygon (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem sides_and_diagonals : 
  (∃ n : ℕ, number_of_sides_of_polygon n ∧ n = 12) ∧ number_of_diagonals_of_polygon 12 = 54 :=
by {
  -- Proof will be filled in later
  sorry
}

end sides_and_diagonals_l364_364378


namespace paths_from_A_to_C_l364_364260

theorem paths_from_A_to_C : 
  let num_paths_A_to_Blue := 2 * 2,
      num_paths_Blue_to_Green := (3 * 2) * 2,
      num_paths_Green_to_C := 2 * 2,
      total_paths := num_paths_A_to_Blue * num_paths_Blue_to_Green * num_paths_Green_to_C
  in total_paths = 192 :=
by
  unfold let num_paths_A_to_Blue,
  unfold let num_paths_Blue_to_Green,
  unfold let num_paths_Green_to_C,
  unfold let total_paths,
  exact rfl

end paths_from_A_to_C_l364_364260


namespace circle_reflection_l364_364603

variable (x₀ y₀ : ℝ)

def reflect_over_line_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem circle_reflection 
  (h₁ : x₀ = 8)
  (h₂ : y₀ = -3) :
  reflect_over_line_y_eq_neg_x (x₀, y₀) = (3, -8) := by
  sorry

end circle_reflection_l364_364603


namespace gcd_values_count_l364_364691

theorem gcd_values_count :
  ∃ a b : ℕ, (gcd a b) * (nat.lcm a b) = 180 ∧
    set.card { gcd a b | ∃ a b, a * b = 180 } = 8 :=
by
  -- Problem statement as provided by conditions and question
  -- Definitions and notations are provided correctly and fully, proof is omitted
  sorry

end gcd_values_count_l364_364691


namespace total_apples_l364_364159

theorem total_apples (n : ℕ) (h1 : ∀ k : ℕ, 6 * k = n ∨ 12 * k = n) (h2 : 70 ≤ n) (h3 : n ≤ 80) : 
  n = 72 ∨ n = 78 := 
sorry

end total_apples_l364_364159


namespace total_apples_l364_364158

theorem total_apples (n : ℕ) (h1 : ∀ k : ℕ, 6 * k = n ∨ 12 * k = n) (h2 : 70 ≤ n) (h3 : n ≤ 80) : 
  n = 72 ∨ n = 78 := 
sorry

end total_apples_l364_364158


namespace Hank_sold_bikes_on_Sunday_l364_364046

theorem Hank_sold_bikes_on_Sunday :
  ∃ S : ℕ, let net_increase_fri := 15 - 10 in
           let net_increase_sat := 8 - 12 in
           let net_increase_sun := 11 - S in
           net_increase_fri + net_increase_sat + net_increase_sun = 3 ∧ S = 10 :=
begin
  sorry
end

end Hank_sold_bikes_on_Sunday_l364_364046


namespace sum_seq_c_l364_364833

def seq_an : ℕ → ℕ
| 1 := 1
| (n + 2) := 2 * seq_an (n + 1) + 1

def seq_bn (n : ℕ) := Real.log2 (seq_an n + 1)

def seq_c (n : ℕ) := 1 / (seq_bn n * seq_bn (n + 2))

lemma seq_geo (n : ℕ) : seq_an (n + 1) + 1 = 2 * (seq_an n + 1) :=
by {
  induction n with k hk,
  { simp [seq_an], },
  { simp [seq_an, -add_comm, two_mul, hk, add_assoc] },
}

theorem sum_seq_c (n : ℕ) : 
  (∑ i in Finset.range n, seq_c i) = (3 * n^2 + 5 * n) / (4 * n^2 + 12 * n + 8) :=
by { sorry }

end sum_seq_c_l364_364833


namespace max_area_OAMB_l364_364518

theorem max_area_OAMB : 
  ∃ (M : ℝ × ℝ), 
  (M.1 ^ 2 / 4 + M.2 ^ 2 / 12 = 1) ∧ 
  M.1 ≥ 0 ∧ M.2 ≥ 0 ∧ 
  M.1 ≤ 2 ∧ M.2 ≤ 2 * Real.sqrt 3 ∧ 
  let A := (2 : ℝ, 0) in
  let B := (0 : ℝ, 2 * Real.sqrt 3) in
  let O := (0 : ℝ, 0) in
  (∃ θ, M = (2 * Real.cos θ, 2 * Real.sqrt 3 * Real.sin θ) ∧ θ > 0 ∧ θ < Real.pi / 2) ∧
  (∃ S, 
    S = (1 / 2) * (O.1 * (M.2 - O.2) + M.1 * (B.2 - M.2)) +
        (1 / 2) * (M.2 * (B.1 - M.1) + O.2 * (M.1 - B.1)) ∧
    S = 2 * Real.sqrt 6)  :=
sorry

end max_area_OAMB_l364_364518


namespace ABCD_is_parallelogram_l364_364543

variables {A B C D E F M N : Type*}
variables [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
variables [HasSub A] [HasSub B] [HasSub C] [HasSub D]
variables [HasSmulℚ A] [HasSmulℚ B] [HasSmulℚ C] [HasSmulℚ D]
variables [HasAdd A] [HasAdd B] [HasAdd C] [HasAdd D]

-- Definitions of midpoints, intersection points, and parallelogram conditions
variables (E : A) (F : B) (M : C) (N : D)
variables (BC DE BF AC : Set (A × A)) (MENF : Set (A × A))

-- Conditions
variables (is_midpoint_E : ∀ x : A, x ∈ BC → E = (1/2) • bc x)
variables (is_midpoint_F : ∀ x : B, x ∈ AD → F = (1/2) • ad x)
variables (is_intersection_M : M ∈ (AC ∩ BF))
variables (is_intersection_N : N ∈ (AC ∩ DE))
variables (is_parallelogram_MENF : MENF ∈ parallelogram_properties)

-- Parallelogram properties
axioms (parallelogram_properties : ∀ x y z w : Type*, x ∈ parallelogram_properties → y ∈ parallelogram_properties →

theorem ABCD_is_parallelogram 
  (ABCD : Set (A × A)) 
  (hABCD_quadrilateral : ABCD ∈ quadrilateral_properties)
  (h1 : is_midpoint_E)
  (h2 : is_midpoint_F)
  (h3 : is_intersection_M)
  (h4 : is_intersection_N)
  (h5 : is_parallelogram_MENF)
: ABCD ∈ parallelogram_properties := 
sorry

end ABCD_is_parallelogram_l364_364543


namespace find_a_range_l364_364921

variable {a : ℝ}

def circle_C (x y : ℝ) := (x - a)^2 + (y - a + 2)^2 = 1
def circle_D (x y : ℝ) := x^2 + (y - 1)^2 = 4
def center_distance := real.sqrt (a^2 + (a - 3)^2)
def exist_point_on_circles_intersection := ∃ (x y : ℝ), circle_C x y ∧ circle_D x y
def circles_intersect_or_tangent := 1 ≤ center_distance ∧ center_distance ≤ 3
def range_a_valid := 0 ≤ a ∧ a ≤ 3

theorem find_a_range : 
  (exist_point_on_circles_intersection ↔ circles_intersect_or_tangent) → range_a_valid :=
begin
  assume h,
  sorry
end

end find_a_range_l364_364921


namespace composite_dice_product_probability_l364_364472

theorem composite_dice_product_probability :
  let outcomes := 6 ^ 4
  let non_composite_ways := 13
  let composite_probability := 1 - non_composite_ways / outcomes
  composite_probability = 1283 / 1296 :=
by
  sorry

end composite_dice_product_probability_l364_364472


namespace unique_surjective_sum_free_function_l364_364950

open Set

definition sum_free (A : Set ℕ) : Prop :=
  ∀ x y ∈ A, x + y ∉ A

noncomputable def is_surjective (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, f m = n

theorem unique_surjective_sum_free_function :
  ∀ (f : ℕ → ℕ), is_surjective f → 
  (∀ (A : Set ℕ), sum_free A → sum_free (Set.image f A)) →
  ∀ x : ℕ, f x = x := 
by
  sorry

end unique_surjective_sum_free_function_l364_364950


namespace combined_ratio_is_1_l364_364201

-- Conditions
variables (V1 V2 M1 W1 M2 W2 : ℝ)
variables (x : ℝ)
variables (ratio_volumes ratio_milk_water_v1 ratio_milk_water_v2 : ℝ)

-- Given conditions as hypotheses
-- Condition: V1 / V2 = 3 / 5
-- Hypothesis 1: The volume ratio of the first and second vessels
def volume_ratio : Prop :=
  V1 / V2 = 3 / 5

-- Condition: M1 / W1 = 1 / 2 in first vessel
-- Hypothesis 2: The milk to water ratio in the first vessel
def milk_water_ratio_v1 : Prop :=
  M1 / W1 = 1 / 2

-- Condition: M2 / W2 = 3 / 2 in the second vessel
-- Hypothesis 3: The milk to water ratio in the second vessel
def milk_water_ratio_v2 : Prop :=
  M2 / W2 = 3 / 2

-- Definition: Total volumes of milk and water in the larger vessel
def total_milk_water_ratio : Prop :=
  (M1 + M2) / (W1 + W2) = 1 / 1

-- Main theorem: Given the ratios, the ratio of milk to water in the larger vessel is 1:1
theorem combined_ratio_is_1 :
  (volume_ratio V1 V2) →
  (milk_water_ratio_v1 M1 W1) →
  (milk_water_ratio_v2 M2 W2) →
  total_milk_water_ratio M1 W1 M2 W2 :=
by
  -- Proof omitted
  sorry

end combined_ratio_is_1_l364_364201


namespace floor_of_minus_3_point_7_l364_364767

theorem floor_of_minus_3_point_7 : Real.floor (-3.7) = -4 := by
  sorry

end floor_of_minus_3_point_7_l364_364767


namespace total_cost_is_130_l364_364198

-- Defining the number of each type of pet
def n_puppies : ℕ := 2
def n_kittens : ℕ := 2
def n_parakeets : ℕ := 3

-- Defining the cost of one parakeet
def c_parakeet : ℕ := 10

-- Defining the cost of one puppy and one kitten based on the conditions
def c_puppy : ℕ := 3 * c_parakeet
def c_kitten : ℕ := 2 * c_parakeet

-- Defining the total cost of all pets
def total_cost : ℕ :=
  (n_puppies * c_puppy) + (n_kittens * c_kitten) + (n_parakeets * c_parakeet)

-- Lean theorem stating that the total cost is 130 dollars
theorem total_cost_is_130 : total_cost = 130 := by
  -- The proof will be filled in here.
  sorry

end total_cost_is_130_l364_364198


namespace axis_of_symmetry_interval_of_decrease_range_of_m_l364_364351

open Real

def a (x : ℝ) : ℝ × ℝ := (sin x, cos x)
def b (x : ℝ) : ℝ × ℝ := (sin x, sin x)
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem axis_of_symmetry :
  (∃ k : ℤ, ∀ x : ℝ, f x = (1/2) + (√2 / 2) * sin (2 * x - π / 4) → 
    x = k * (π / 2) + 3 * π / 8) :=
sorry

theorem interval_of_decrease :
  (∀ k : ℤ, ∀ x : ℝ, f x = (1/2) + (√2 / 2) * sin (2 * x - π / 4) → 
    (k * π - π / 8 <= x ∧ x <= k * π + 3 * π / 8)) :=
sorry

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Icc (π / 6) (π / 3), f x - m < 2) → m > (√3 - 5) / 4 :=
sorry

end axis_of_symmetry_interval_of_decrease_range_of_m_l364_364351


namespace simplify_expression_l364_364583

theorem simplify_expression (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m / n - n / m) / (1 / m - 1 / n) = -(m + n) :=
by sorry

end simplify_expression_l364_364583


namespace intersect_at_single_point_l364_364358

-- Given data and geometric constructions for the problem
variables {A B C M N L K : Point}
variable acuteABC : Triangle A B C → Prop -- A property indicating that triangle ABC is acute-angled
variable equalRectangles : Rectangle A B M N → Rectangle L B C K → Prop -- A property indicating the rectangles are equal and outwardly constructed
variable AB_equals_KC : A B = K C -- Property indicating AB = KC

-- Statement of the problem
theorem intersect_at_single_point
  (acute_ABC : acuteABC (Triangle.mk A B C))
  (rectangles_eq : equalRectangles (Rectangle.mk A B M N) (Rectangle.mk L B C K))
  (eq_sides : AB_equals_KC) :
  ∃ X : Point, collinear A L X ∧ collinear N K X ∧ collinear M C X :=
sorry

end intersect_at_single_point_l364_364358


namespace larger_angle_of_nonagon_l364_364117

theorem larger_angle_of_nonagon : 
  ∀ (n : ℕ) (x : ℝ), 
  n = 9 → 
  (∃ a b : ℕ, a + b = n ∧ a * x + b * (3 * x) = 180 * (n - 2)) → 
  3 * (180 * (n - 2) / 15) = 252 :=
by
  sorry

end larger_angle_of_nonagon_l364_364117


namespace bees_count_l364_364971

-- Definitions of the conditions
def day1_bees (x : ℕ) := x  -- Number of bees on the first day
def day2_bees (x : ℕ) := 3 * day1_bees x  -- Number of bees on the second day is 3 times that on the first day

theorem bees_count (x : ℕ) (h : day2_bees x = 432) : day1_bees x = 144 :=
by
  dsimp [day1_bees, day2_bees] at h
  have h1 : 3 * x = 432 := h
  sorry

end bees_count_l364_364971


namespace number_of_real_solutions_l364_364421

theorem number_of_real_solutions : 
  set.countable {x : ℝ | ∃ n : ℕ, (sqrt (123 - sqrt x) = n)} = 12 :=
by
  sorry

end number_of_real_solutions_l364_364421


namespace total_revenue_generated_l364_364286

-- Definitions for conditions
def small_bottles_in_storage := 6000
def big_bottles_in_storage := 14000
def medium_bottles_in_storage := 9000

def price_per_small_bottle := 2
def price_per_big_bottle := 4
def price_per_medium_bottle := 3

def percentage_small_bottles_sold := 0.20
def percentage_big_bottles_sold := 0.23
def percentage_medium_bottles_sold := 0.15

-- Define the number of bottles sold for each size
def small_bottles_sold := percentage_small_bottles_sold * small_bottles_in_storage
def big_bottles_sold := percentage_big_bottles_sold * big_bottles_in_storage
def medium_bottles_sold := percentage_medium_bottles_sold * medium_bottles_in_storage

-- Define the revenue from each size
def revenue_from_small_bottles := small_bottles_sold * price_per_small_bottle
def revenue_from_big_bottles := big_bottles_sold * price_per_big_bottle
def revenue_from_medium_bottles := medium_bottles_sold * price_per_medium_bottle

-- Define the total revenue
def total_revenue := revenue_from_small_bottles + revenue_from_big_bottles + revenue_from_medium_bottles

-- The main theorem to prove
theorem total_revenue_generated : total_revenue = 19330 := by
  sorry

end total_revenue_generated_l364_364286


namespace find_incorrect_conclusion_l364_364682

-- Define the conditions as given in the problem
variables {R : Type*} [linear_ordered_field R]
variables (a b c t : R)
variable (h_nonzero : a ≠ 0)
variable (h1 : a - b + c = 0)
variable (h2 : a + b + c = 4)
variable (h3 : c = 3)

-- Define the main hypothesis that must be checked
def incorrect_conclusion : Prop :=
  let p := 3 + (-x ^ 2 + 2 * x + 3) in
  ∀ (x : R), 2 * x + p > x^2

-- State the proof problem
theorem find_incorrect_conclusion : (¬ incorrect_conclusion) :=
sorry

end find_incorrect_conclusion_l364_364682


namespace LeahsCoinsValueIs68_l364_364948

def LeahsCoinsWorthInCents (p n d : Nat) : Nat :=
  p * 1 + n * 5 + d * 10

theorem LeahsCoinsValueIs68 {p n d : Nat} (h1 : p + n + d = 17) (h2 : n + 2 = p) :
  LeahsCoinsWorthInCents p n d = 68 := by
  sorry

end LeahsCoinsValueIs68_l364_364948


namespace multiples_count_l364_364879

theorem multiples_count (count_5 count_7 count_35 count_total : ℕ) :
  count_5 = 600 →
  count_7 = 428 →
  count_35 = 85 →
  count_total = count_5 + count_7 - count_35 →
  count_total = 943 :=
by
  sorry

end multiples_count_l364_364879


namespace simplify_vector_expression_l364_364999

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (A B C M O : V)

-- Define the vectors
def AB := B - A
def MB := B - M
def BO := O - B
def BC := C - B
def OM := M - O
def AC := C - A

theorem simplify_vector_expression :
  ((AB + MB) + (BO + BC) + OM) = AC := by
sorry

end simplify_vector_expression_l364_364999


namespace calculate_ratio_of_areas_l364_364623

noncomputable def ratio_pond_to_field : ℚ := 1 / 8

theorem calculate_ratio_of_areas (W L A_field A_pond : ℚ) 
  (h1 : L = 2 * W) 
  (h2 : L = 32) 
  (h3 : A_pond = 8 * 8) 
  (h4 : A_field = L * W) : 
  A_pond / A_field = ratio_pond_to_field := 
by 
  have W_calc : W = 16 := by linarith,
  have A_field_calc : A_field = 512 := by linarith,
  have A_pond_calc : A_pond = 64 := by norm_num,
  simp [A_field_calc, A_pond_calc, ratio_pond_to_field],
  sorry

end calculate_ratio_of_areas_l364_364623


namespace square_side_length_l364_364592

theorem square_side_length (A : ℝ) (side_length : ℝ) (width : ℝ) (length : ℝ) (h1 : width = 3) (h2 : length = 3) (h3 : A = width * length) (h4 : A = side_length * side_length) : side_length = 3 :=
by
  solve_by_elim
  sorry

end square_side_length_l364_364592


namespace propositions_correctness_l364_364381

theorem propositions_correctness:
  (¬ (∀ L₁ L₂: Line, (L₁ ∥ L₂) → (∃ P: Plane, L₁ ∥ P ∧ L₂ ∈ P))) ∧
  (¬ (∀ L: Line, ∀ P: Plane, (L ∥ P) → (∀ p₁ p₂: Point, p₁ ∈ P → p₂ ∈ P → (p₁p₂ ∥ L)))) ∧
  (∀ a b: Line, ∃ P: Point, (skew a b) → (∃ Q: Plane, a ∥ Q ∧ b ∥ Q ∧ P ∉ Q)) ∧
  (∀ a b: Line, skew a b → (∃ P: Plane, a ⊂ P ∧ b ∥ P)) :=
by
  -- Placeholder for proof
  sorry

end propositions_correctness_l364_364381


namespace probability_of_N10_mod_7_eq_1_l364_364756

theorem probability_of_N10_mod_7_eq_1 :
  let N_set := {n : ℕ | 1 ≤ n ∧ n ≤ 2023}
  let favorable_n := {n : ℕ | n ∈ N_set ∧ (n % 7 = 1 ∨ n % 7 = 2 ∨ n % 7 = 6)}
  let probability := (favorable_n.card : ℚ) / (N_set.card : ℚ)
  in probability = 3 / 7 :=
by
  sorry

end probability_of_N10_mod_7_eq_1_l364_364756


namespace transformation_power_of_two_l364_364976

theorem transformation_power_of_two (n : ℕ) (h : n ≥ 3) :
  ∃ s : ℕ, 2 ^ s ≥ n :=
by sorry

end transformation_power_of_two_l364_364976


namespace general_formula_a_sum_first_n_terms_c_l364_364368

variable {ℕ : Type} [Nonzero ℚ] 

-- Define the conditions
def a_sequence (n : ℕ) : ℚ := 2 * n - 1
def b_sequence (n : ℕ) : ℚ := 3 ^ (n - 1)
def c_sequence (n : ℕ) : ℚ := a_sequence (2 * n - 1) + b_sequence (2 * n - 1)

-- Prove the general formula for {a_n}
theorem general_formula_a :
  ∀ n : ℕ, a_sequence n = 2 * n - 1 :=
sorry

-- Prove the sum of the first n terms of {c_n}
theorem sum_first_n_terms_c :
  ∀ n : ℕ, (Finset.range n).sum (λ i, c_sequence (i + 1)) = n^2 + (9^n - 1) / 8 :=
sorry

end general_formula_a_sum_first_n_terms_c_l364_364368


namespace probability_three_at_marked_l364_364259

def sudoku_matrix := Matrix (Fin 9) (Fin 9) (Fin 9.succ)

def valid_sudoku (m : sudoku_matrix) : Prop :=
  ∀ i, (∀ j, (m i j).val ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (∀ i₁ i₂ j, i₁ ≠ i₂ → m i₁ j ≠ m i₂ j) ∧
  (∀ i j₁ j₂, j₁ ≠ j₂ → m i j₁ ≠ m i j₂) ∧
  (∀ k l n₁ n₂, (n₁ ≠ n₂) →
    m ⟨3 * (k / 3) + (n₁ / 3), _⟩ ⟨3 * (l / 3) + (n₁ % 3), _⟩ ≠
    m ⟨3 * (k / 3) + (n₂ / 3), _⟩ ⟨3 * (l / 3) + (n₂ % 3), _⟩)

noncomputable def probability_digit_at (m : sudoku_matrix) (d : Fin 9.succ) (i j : Fin 9) : ℚ :=
  if valid_sudoku m then (1 : ℚ) / 9 else 0

theorem probability_three_at_marked (m : sudoku_matrix) (marked : Fin 9 × Fin 9) :
  valid_sudoku m →
  probability_digit_at m (Fin.succ ⟨2, by norm_num⟩) marked.1 marked.2 = (1 : ℚ) / 9 :=
sorry

end probability_three_at_marked_l364_364259


namespace select_100_out_of_200_divisible_by_100_l364_364058

theorem select_100_out_of_200_divisible_by_100 (s : Finset ℤ) (h : s.card = 200) :
  ∃ (t : Finset ℤ), t.card = 100 ∧ (∑ x in t, x) % 100 = 0 :=
sorry

end select_100_out_of_200_divisible_by_100_l364_364058


namespace fortieth_number_l364_364750

/-- Prove that the 40th integer in the list of integers between 1 and 100 (inclusive) 
    that cannot be written as the product of two consecutive positive integers is 46. -/
theorem fortieth_number (L : List ℕ) (hL : ∀ n ∈ L, 1 ≤ n ∧ n ≤ 100)
  (h_no_consecutive_product : ∀ n ∈ L, ¬ (∃ k : ℕ, n = k * (k + 1)))
  (h_length : L.length = 61)
  : L.nth 39 = some 46 := by
  sorry

end fortieth_number_l364_364750


namespace coeff_x7_in_expansion_l364_364214

-- Each definition in Lean 4 statement reflects the conditions of the problem.
def binomial_coefficient (n k : ℕ) : ℕ := nat.choose n k

-- The condition for expansion using Binomial Theorem
def binomial_expansion_term (n k : ℕ) (a x : ℤ) : ℤ :=
  binomial_coefficient n k * a ^ (n - k) * x ^ k

-- Prove that the coefficient of x^7 in the expansion of (x - 2)^{10} is -960
theorem coeff_x7_in_expansion : 
  binomial_coefficient 10 3 * (-2) ^ 3 = -960 := 
sorry

end coeff_x7_in_expansion_l364_364214


namespace pythagorean_triple_l364_364907

theorem pythagorean_triple (m n : ℕ) (hn : 0 < n) (hm : 0 < m) (h : n < m) :
  let x := m^2 - n^2,
      y := 2 * m * n,
      z := m^2 + n^2
  in x^2 + y^2 = z^2 :=
by
  sorry

end pythagorean_triple_l364_364907


namespace exists_twelve_distinct_x_l364_364450

theorem exists_twelve_distinct_x :
  ∃ S : Set ℝ, (S.card = 12) ∧ (∀ x ∈ S, ∃ k : ℤ, 0 ≤ k ∧ k ≤ 11 ∧ (sqrt (123 - sqrt x) = k)) :=
by
  sorry

end exists_twelve_distinct_x_l364_364450


namespace expansion_third_and_constant_term_l364_364342

theorem expansion_third_and_constant_term :
  let T (k : ℕ) := (Nat.choose 5 k) * (x^3)^(5-k) * (2 / (3 * x^2))^k
  in (T 2, T 3) = (Nat.choose 5 2 * 4 / 9, Nat.choose 5 3 * (2/3)^3) :=
by
  let T (k : ℕ) := (Nat.choose 5 k) * (x^3)^(5-k) * (2 / (3 * x^2))^k
  have h1 : T 2 = Nat.choose 5 2 * 4 / 9 := sorry
  have h2 : T 3 = Nat.choose 5 3 * (2/3)^3 := sorry
  exact ⟨h1, h2⟩

end expansion_third_and_constant_term_l364_364342


namespace calculate_flat_tax_l364_364105

open Real

def price_per_sq_ft (property: String) : Real :=
  if property = "Condo" then 98
  else if property = "BarnHouse" then 84
  else if property = "DetachedHouse" then 102
  else if property = "Townhouse" then 96
  else if property = "Garage" then 60
  else if property = "PoolArea" then 50
  else 0

def area_in_sq_ft (property: String) : Real :=
  if property = "Condo" then 2400
  else if property = "BarnHouse" then 1200
  else if property = "DetachedHouse" then 3500
  else if property = "Townhouse" then 2750
  else if property = "Garage" then 480
  else if property = "PoolArea" then 600
  else 0

def total_value : Real :=
  (price_per_sq_ft "Condo" * area_in_sq_ft "Condo") +
  (price_per_sq_ft "BarnHouse" * area_in_sq_ft "BarnHouse") +
  (price_per_sq_ft "DetachedHouse" * area_in_sq_ft "DetachedHouse") +
  (price_per_sq_ft "Townhouse" * area_in_sq_ft "Townhouse") +
  (price_per_sq_ft "Garage" * area_in_sq_ft "Garage") +
  (price_per_sq_ft "PoolArea" * area_in_sq_ft "PoolArea")

def tax_rate : Real := 0.0125

theorem calculate_flat_tax : total_value * tax_rate = 12697.50 := by
  sorry

end calculate_flat_tax_l364_364105


namespace find_x_l364_364026

theorem find_x (a b x : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < x)
    (h4 : (4 * a)^(4 * b) = a^b * x^(2 * b)) : 
    x = 16 * a^(3 / 2) :=
by 
  sorry

end find_x_l364_364026


namespace melissa_total_cost_l364_364970

-- Definitions based on conditions
def daily_rental_rate : ℝ := 15
def mileage_rate : ℝ := 0.10
def number_of_days : ℕ := 3
def number_of_miles : ℕ := 300

-- Theorem statement to prove the total cost
theorem melissa_total_cost : daily_rental_rate * number_of_days + mileage_rate * number_of_miles = 75 := 
by 
  sorry

end melissa_total_cost_l364_364970


namespace distance_walked_is_4_point_6_l364_364579

-- Define the number of blocks Sarah walked in each direction
def blocks_west : ℕ := 8
def blocks_south : ℕ := 15

-- Define the length of each block in miles
def block_length : ℚ := 1 / 5

-- Calculate the total number of blocks
def total_blocks : ℕ := blocks_west + blocks_south

-- Calculate the total distance walked in miles
def total_distance_walked : ℚ := total_blocks * block_length

-- Statement to prove the total distance walked is 4.6 miles
theorem distance_walked_is_4_point_6 : total_distance_walked = 4.6 := sorry

end distance_walked_is_4_point_6_l364_364579


namespace dice_product_composite_probability_l364_364485

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

-- This function calculates the probability of an event occurring by counting the favorable and total outcomes.
def probability (favorable total : ℕ) : ℚ :=
  favorable / total

noncomputable def probability_of_composite_product : ℚ :=
  probability 1283 1296

theorem dice_product_composite_probability : probability_of_composite_product = 1283 / 1296 := sorry

end dice_product_composite_probability_l364_364485


namespace triangle_values_correct_l364_364506

noncomputable def find_triangle_values (a b c S : ℝ) (A B C : ℝ) : Prop :=
  a = 2 ∧
  B = 60 ∧
  3 * b * (Real.sin A) = 2 * c * (Real.sin B) ∧
  c = 3 ∧
  b = Real.sqrt 7 ∧
  S = 1 / 2 * a * c * (Real.sin B) ∧
  S = 3 * Real.sqrt 3 / 2

theorem triangle_values_correct :
  ∃ (a b c S : ℝ) (A B C : ℝ), find_triangle_values a b c S A B C :=
by
  let a := 2
  let B := 60
  let c := 3
  let b := Real.sqrt 7
  let S := 3 * Real.sqrt 3 / 2
  let A := sorry
  let C := sorry
  use [a, b, c, S, A, B, C]
  have h1 : 3 * b * (Real.sin A) = 2 * c * (Real.sin B) := sorry
  split 
  . rfl
  split 
  . rfl
  split 
  . exact h1
  split 
  . rfl
  split 
  . rfl
  split 
  . calc
    S = 1 / 2 * a * c * (Real.sin B) : _
    ... = 3 * Real.sqrt 3 / 2 : _
    sorry

end triangle_values_correct_l364_364506


namespace reflection_of_point_l364_364607

def reflect_across_y_neg_x (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P
  (y, x)

theorem reflection_of_point
  (P : ℝ × ℝ)
  (h : P = (8, -3)) :
  reflect_across_y_neg_x P = (3, -8) :=
by
  rw [h]
  sorry

end reflection_of_point_l364_364607


namespace solve_for_q_l364_364327

theorem solve_for_q (q : ℝ) (p : ℝ) (h : p = 15 * q^2 - 5) : p = 40 → q = Real.sqrt 3 :=
by
  sorry

end solve_for_q_l364_364327


namespace tan_alpha_l364_364821

-- Given condition
def cond : Prop := tan (α + (π / 4)) = 1 / 7

-- Proposition to be proved, using the given condition
theorem tan_alpha : cond → tan (α) = -3 / 4 := by
  intro h
  sorry

end tan_alpha_l364_364821


namespace jeremy_total_earnings_l364_364006

theorem jeremy_total_earnings :
  let steven_rate : ℚ := 12 / 3
  let mark_rate : ℚ := 10 / 4
  let steven_rooms : ℚ := 8 / 3
  let mark_rooms : ℚ := 9 / 4
  let steven_payment : ℚ := steven_rate * steven_rooms
  let mark_payment : ℚ := mark_rate * mark_rooms
  steven_payment + mark_payment = 391 / 24 :=
by
  let steven_rate : ℚ := 12 / 3
  let mark_rate : ℚ := 10 / 4
  let steven_rooms : ℚ := 8 / 3
  let mark_rooms : ℚ := 9 / 4
  let steven_payment : ℚ := steven_rate * steven_rooms
  let mark_payment : ℚ := mark_rate * mark_rooms
  sorry

end jeremy_total_earnings_l364_364006


namespace female_students_count_l364_364257

-- Define the constants and conditions
def total_visitors : ℕ := 1260
def men_fraction : ℚ := 7/18
def women_students_fraction : ℚ := 6/11

-- Define the number of men and women using the given fractions
def number_of_men : ℕ := (men_fraction * total_visitors).to_nat
def number_of_women : ℕ := total_visitors - number_of_men

-- Define the number of female students
def number_of_female_students : ℕ := (women_students_fraction * number_of_women).to_nat

-- The theorem to prove
theorem female_students_count 
: number_of_female_students = 420 := by sorry

end female_students_count_l364_364257


namespace divisibility_1989_l364_364952

theorem divisibility_1989 (n : ℕ) (h1 : n ≥ 3) :
  1989 ∣ n^(n^(n^n)) - n^(n^n) :=
sorry

end divisibility_1989_l364_364952


namespace flowers_per_day_l364_364564

-- Definitions for conditions
def total_flowers := 360
def days := 6

-- Proof that the number of flowers Miriam can take care of in one day is 60
theorem flowers_per_day : total_flowers / days = 60 := by
  sorry

end flowers_per_day_l364_364564


namespace number_of_hot_dogs_served_during_lunch_today_l364_364281

def number_of_hot_dogs_served_total_today : ℕ := 11
def number_of_hot_dogs_served_during_dinner_today : ℕ := 2

theorem number_of_hot_dogs_served_during_lunch_today :
  ∃ l : ℕ, l = number_of_hot_dogs_served_total_today - number_of_hot_dogs_served_during_dinner_today :=
begin
  use 9,
  {
    sorry
  }
end

end number_of_hot_dogs_served_during_lunch_today_l364_364281


namespace train_speed_kph_l364_364292

-- Definitions based on conditions
def time_seconds : ℕ := 9
def length_meters : ℕ := 135
def conversion_factor : ℕ := 36 -- 3.6 represented as an integer by multiplying both sides by 10

-- The proof statement
theorem train_speed_kph : (length_meters * conversion_factor / 10 / time_seconds = 54) :=
by
  sorry

end train_speed_kph_l364_364292


namespace factorial_divisibility_l364_364896

theorem factorial_divisibility
  (n p : ℕ)
  (h1 : p > 0)
  (h2 : n ≤ p + 1) :
  (factorial (p^2)) ∣ (factorial p)^(p + 1) :=
sorry

end factorial_divisibility_l364_364896


namespace ball_hits_ground_l364_364091

theorem ball_hits_ground (t : ℝ) (y : ℝ) : 
  (y = -8 * t^2 - 12 * t + 72) → 
  (y = 0) → 
  t = 3 := 
by
  sorry

end ball_hits_ground_l364_364091


namespace expression_evaluation_l364_364770

theorem expression_evaluation : 
  (let expr := (((3 + 2)⁻¹ + 2)⁻¹ + 2)⁻¹ + 2
  in expr = 65 / 27) :=
by
  sorry

end expression_evaluation_l364_364770


namespace apple_count_l364_364144

/--
Given the total number of apples in several bags, where each bag contains either 12 or 6 apples,
and knowing that the total number of apples is between 70 and 80 inclusive,
prove that the total number of apples can only be 72 or 78.
-/
theorem apple_count (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : ∃ k : ℕ, n = 6 * k) : n = 72 ∨ n = 78 :=
by {
  sorry
}

end apple_count_l364_364144


namespace distance_between_two_girls_after_12_hours_l364_364238

theorem distance_between_two_girls_after_12_hours :
  let speed1 := 7 -- speed of the first girl (km/hr)
  let speed2 := 3 -- speed of the second girl (km/hr)
  let time := 12 -- time (hours)
  let distance1 := speed1 * time -- distance traveled by the first girl
  let distance2 := speed2 * time -- distance traveled by the second girl
  distance1 + distance2 = 120 := -- total distance
by
  -- Here, we would provide the proof, but we put sorry to skip it
  sorry

end distance_between_two_girls_after_12_hours_l364_364238


namespace largest_num_blocks_l364_364216

-- Define the volume of the box
def volume_box (l₁ w₁ h₁ : ℕ) : ℕ :=
  l₁ * w₁ * h₁

-- Define the volume of the block
def volume_block (l₂ w₂ h₂ : ℕ) : ℕ :=
  l₂ * w₂ * h₂

-- Define the function to calculate maximum blocks
def max_blocks (V_box V_block : ℕ) : ℕ :=
  V_box / V_block

theorem largest_num_blocks :
  max_blocks (volume_box 5 4 6) (volume_block 3 3 2) = 6 :=
by
  sorry

end largest_num_blocks_l364_364216


namespace probability_composite_product_l364_364467

theorem probability_composite_product :
  let dice_faces := 6
  let rolls := 4
  let total_outcomes := dice_faces ^ rolls
  let non_composite_cases := 13
  let non_composite_probability := non_composite_cases / total_outcomes
  let composite_probability := 1 - non_composite_probability
  composite_probability = 1283 / 1296 := by
  sorry

end probability_composite_product_l364_364467


namespace eliza_is_18_l364_364763

-- Define the relevant ages
def aunt_ellen_age : ℕ := 48
def dina_age : ℕ := aunt_ellen_age / 2
def eliza_age : ℕ := dina_age - 6

-- Theorem to prove Eliza's age is 18
theorem eliza_is_18 : eliza_age = 18 := by
  sorry

end eliza_is_18_l364_364763


namespace m_minus_n_is_square_l364_364712

theorem m_minus_n_is_square (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 2001 * m ^ 2 + m = 2002 * n ^ 2 + n) : ∃ k : ℕ, m - n = k ^ 2 :=
sorry

end m_minus_n_is_square_l364_364712


namespace no_solution_for_given_arithmetic_progression_l364_364329

noncomputable def no_valid_y_for_arithmetic_sequence (y : ℤ) : Prop :=
  let a1 := y - 3
  let a2 := 3 * y + 1
  let a3 := 5 * y - 7
  ¬ (2 * y + 4 = 2 * y - 8)

theorem no_solution_for_given_arithmetic_progression : ∀ y : ℤ, no_valid_y_for_arithmetic_sequence y := 
by {
  intro y,
  let a1 := y - 3,
  let a2 := 3 * y + 1,
  let a3 := 5 * y - 7,
  have h1 : a2 - a1 = 2 * y + 4 := by simp [a1, a2],
  have h2 : a3 - a2 = 2 * y - 8 := by simp [a2, a3],
  suffices : 2 * y + 4 ≠ 2 * y - 8,
  contradiction,
  intro h,
  have : 4 = -8 := by linarith [h],
  linarith,
}

end no_solution_for_given_arithmetic_progression_l364_364329


namespace Lisa_needs_additional_marbles_l364_364969

theorem Lisa_needs_additional_marbles :
  ∀ (n m : ℕ),
    (∃ f : ℕ → ℕ, (∀ i j, i ≠ j → f i ≠ f j) ∧ (∀ i, 1 ≤ f i) ∧ (∑ i in Finset.range n, f i) = m)
    → (∑ i in Finset.range 12, i + 1) = 78
    → m = 50
    → n = 12
    → 78 - m = 28 :=
by
  intros n m h_un_friends sum_old sum_curr n_old
  sorry

end Lisa_needs_additional_marbles_l364_364969


namespace laundry_time_l364_364122

theorem laundry_time (n : ℕ) (wash_time dry_time : ℕ) (loads : ℕ) : (loads = 8) → (wash_time = 45) → (dry_time = 60) → (n = 14) → 
  (loads * (wash_time + dry_time)) / 60 = n := 
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end laundry_time_l364_364122


namespace people_attend_both_reunions_l364_364713

theorem people_attend_both_reunions (N D H x : ℕ) 
  (hN : N = 50)
  (hD : D = 50)
  (hH : H = 60)
  (h_total : N = D + H - x) : 
  x = 60 :=
by
  sorry

end people_attend_both_reunions_l364_364713


namespace solution_set_for_log_inequality_l364_364023

noncomputable def f : ℝ → ℝ := sorry

def isEven (f : ℝ → ℝ) := ∀ x, f (-x) = f x

def isIncreasingOnNonNeg (f : ℝ → ℝ) := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

def f_positive_at_third : Prop := f (1 / 3) > 0

theorem solution_set_for_log_inequality
  (hf_even : isEven f)
  (hf_increasing : isIncreasingOnNonNeg f)
  (hf_positive : f_positive_at_third) :
  {x : ℝ | f (Real.log x / Real.log (1/8)) > 0} = {x : ℝ | 0 < x ∧ x < 1/2} ∪ {x : ℝ | 2 < x} := sorry

end solution_set_for_log_inequality_l364_364023


namespace A_inter_B_l364_364874

open Set

noncomputable def A : Set ℤ := { x | x^2 - 4 * x ≤ 0 }

noncomputable def B : Set ℝ := { y | ∃ x : ℤ, x ∈ A ∧ y = Real.log (x.toReal + 1) / Real.log 2}

theorem A_inter_B :
  A ∩ B = {0, 1, 2} :=
sorry

end A_inter_B_l364_364874


namespace dice_product_composite_probability_l364_364484

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

-- This function calculates the probability of an event occurring by counting the favorable and total outcomes.
def probability (favorable total : ℕ) : ℚ :=
  favorable / total

noncomputable def probability_of_composite_product : ℚ :=
  probability 1283 1296

theorem dice_product_composite_probability : probability_of_composite_product = 1283 / 1296 := sorry

end dice_product_composite_probability_l364_364484


namespace train_speed_l364_364289

theorem train_speed (time_seconds : ℕ) (length_meters : ℕ) (speed_kmph : ℕ)
  (h1 : time_seconds = 9) (h2 : length_meters = 135) : speed_kmph = 54 :=
sorry

end train_speed_l364_364289


namespace quad_sum_of_squares_eq_diameter_squared_l364_364279

theorem quad_sum_of_squares_eq_diameter_squared
  (A B C D O T : Point)
  (R : ℝ)
  (h_cyclic : IsCyclicQuadrilateral A B C D) 
  (h_perpendicular : ∠(A, T, C) = 90 ∧ ∠(B, T, D) = 90)
  (h_circle : CircumscribedCircle A B C D = Some ⟨O, R⟩) 
  (h_intersection : Line(A, C).Intersect(Line(B, D)) = T):
  (dist A B) ^ 2 + (dist C D) ^ 2 = 4 * R ^ 2 :=
sorry

end quad_sum_of_squares_eq_diameter_squared_l364_364279


namespace product_of_fractions_is_3_div_80_l364_364223

def product_fractions (a b c d e f : ℚ) : ℚ := (a / b) * (c / d) * (e / f)

theorem product_of_fractions_is_3_div_80 
  (h₁ : product_fractions 3 8 2 5 1 4 = 3 / 80) : True :=
by
  sorry

end product_of_fractions_is_3_div_80_l364_364223


namespace probability_conditional_l364_364373

-- Definitions based on conditions
variables A B : Prop              -- Events A and B
variable P : Prop → ℝ             -- Probability function P

-- Conditions
axiom P_A : P A = 9 / 10
axiom P_AB : P (A ∧ B) = 1 / 2

-- Statement to prove
theorem probability_conditional :
  P (A ∧ B) / P A = 5 / 9 :=
by
  rw [P_AB, P_A]
  norm_num

end probability_conditional_l364_364373


namespace laundry_time_l364_364120

theorem laundry_time (n : ℕ) (wash_time dry_time total_loads : ℕ) (h1 : n = 8) 
  (h2 : wash_time = 45) (h3 : dry_time = 60) (h4 : total_loads = 8) : 
  (n * (wash_time + dry_time)) / 60 = 14 :=
by {
  rw [h1, h2, h3, h4],
  sorry
}

end laundry_time_l364_364120


namespace apple_bags_l364_364173

theorem apple_bags (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apple_bags_l364_364173


namespace median_is_80_84_l364_364780

def student_scores : List (ℕ × ℕ) :=
  [(90, 94), (10),
   (85, 89), (20),
   (80, 84), (30),
   (75, 79), (25),
   (70, 74), (10),
   (65, 69), (6)]

def median_interval (scores : List (ℕ × ℕ)) : (ℕ × ℕ) :=
  let median_pos := (101 + 1) / 2
  let rec find_interval (pos : ℕ) (lst : List (ℕ × ℕ)) : (ℕ × ℕ) :=
    match lst with
    | [] => (0, 0)
    | (low, high, count) :: rest =>
      if pos <= count then (low, high)
      else find_interval (pos - count) rest
  find_interval median_pos scores

theorem median_is_80_84 : median_interval student_scores = (80, 84) :=
  sorry

end median_is_80_84_l364_364780


namespace books_remainder_l364_364917

theorem books_remainder (total_books piles_books new_piles_books remainder : ℕ)
  (h_total_books : total_books = 1452)
  (h_piles_books : piles_books = 42)
  (h_new_piles_books : new_piles_books = 43)
  : remainder = total_books % new_piles_books := 
by
  rw [h_total_books, h_new_piles_books],
  exact nat.mod_eq_of_lt 33 (by norm_num),
  sorry

end books_remainder_l364_364917


namespace num_common_tangents_l364_364877

def circle1 : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
def circle2 : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 4 * p.1 + 2 * p.2 + 4 = 0}

theorem num_common_tangents (C1 C2 : set (ℝ × ℝ)) (hC1 : C1 = circle1) (hC2 : C2 = circle2) : 
∃ n : ℕ, n = 2 ∧ ∀ p : ℝ × ℝ, (∃ t1 : set (ℝ × ℝ), tangent_to_circle t1 C1 ∧ tangent_to_circle t1 C2 ∧ t1 = p) → n = 2 :=
by sorry

end num_common_tangents_l364_364877


namespace complex_modulus_one_l364_364858

noncomputable def z : ℂ := sorry

theorem complex_modulus_one (z : ℂ) (hz : (1 + z) / (1 - z) = complex.I) : complex.abs z = 1 := 
by {
  sorry
}

end complex_modulus_one_l364_364858


namespace distance_to_first_sign_l364_364044

-- Definitions based on conditions
def total_distance : ℕ := 1000
def after_second_sign : ℕ := 275
def between_signs : ℕ := 375

-- Problem statement
theorem distance_to_first_sign 
  (D : ℕ := total_distance) 
  (a : ℕ := after_second_sign) 
  (d : ℕ := between_signs) : 
  (D - a - d = 350) :=
by
  sorry

end distance_to_first_sign_l364_364044


namespace probability_identical_digits_l364_364673

def two_digit_numbers : Finset ℤ := Finset.Icc 10 99

def numbers_with_identical_digits : Finset ℤ := 
  Finset.filter (λ n, (10 * (n / 11) = n)) (Finset.Icc 10 99)

theorem probability_identical_digits : 
  (numbers_with_identical_digits.card : ℝ) / (two_digit_numbers.card : ℝ) = 0.1 :=
by
  sorry

end probability_identical_digits_l364_364673


namespace percentage_increase_in_take_home_pay_l364_364009

namespace JohnPayRaise

-- Conditions from part a)
variable (gross_pay_before : ℝ := 60)
variable (gross_pay_after : ℝ := 70)
variable (tax_rate_before : ℝ := 0.15)
variable (tax_rate_after : ℝ := 0.18)

-- Derived values based on conditions
def take_home_pay_before := gross_pay_before - (tax_rate_before * gross_pay_before)
def take_home_pay_after := gross_pay_after - (tax_rate_after * gross_pay_after)

-- Statement to prove percentage increase
theorem percentage_increase_in_take_home_pay :
  let percentage_increase := ((take_home_pay_after - take_home_pay_before) / take_home_pay_before) * 100 in
  percentage_increase = 12.55 :=
by
  sorry

end JohnPayRaise

end percentage_increase_in_take_home_pay_l364_364009


namespace simplify_trig_l364_364993

open Real

theorem simplify_trig : 
  (sin (30 * pi / 180) + sin (60 * pi / 180)) / (cos (30 * pi / 180) + cos (60 * pi / 180)) = tan (45 * pi / 180) :=
by
  sorry

end simplify_trig_l364_364993


namespace dice_product_composite_probability_l364_364483

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

-- This function calculates the probability of an event occurring by counting the favorable and total outcomes.
def probability (favorable total : ℕ) : ℚ :=
  favorable / total

noncomputable def probability_of_composite_product : ℚ :=
  probability 1283 1296

theorem dice_product_composite_probability : probability_of_composite_product = 1283 / 1296 := sorry

end dice_product_composite_probability_l364_364483


namespace ratio_saturday_friday_l364_364975

variable (S : ℕ)
variable (soldOnFriday : ℕ := 30)
variable (soldOnSunday : ℕ := S - 15)
variable (totalSold : ℕ := 135)

theorem ratio_saturday_friday (h1 : soldOnFriday = 30)
                              (h2 : totalSold = 135)
                              (h3 : soldOnSunday = S - 15)
                              (h4 : soldOnFriday + S + soldOnSunday = totalSold) :
  (S / soldOnFriday) = 2 :=
by
  -- Prove the theorem here...
  sorry

end ratio_saturday_friday_l364_364975


namespace range_of_a_l364_364095

def is_decreasing_on_interval (f : ℝ → ℝ) (interval : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ interval → x₂ ∈ interval → x₁ ≤ x₂ → f x₁ ≥ f x₂

theorem range_of_a (a : ℝ) :
  let f : ℝ → ℝ := λ x, x^2 + 2*(a - 5)*x - 6
  (is_decreasing_on_interval f {x : ℝ | x ≤ -5}) → a ≤ 10 := by
  sorry

end range_of_a_l364_364095


namespace number_of_solutions_l364_364101

noncomputable def f (x : ℝ) : ℝ := x^(-2) - (1/2)^x

theorem number_of_solutions : 
  ∃ x1 x2 x3 : ℝ,
     f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 :=
sorry

end number_of_solutions_l364_364101


namespace expansion_n_equals_8_l364_364954

theorem expansion_n_equals_8 (n : ℕ) (h1 : n ≥ 3) 
  (h2 : let a2 := Nat.choose n 2 in let a3 := -Nat.choose n 3 in a3 + 2 * a2 = 0) : n = 8 :=
by
  sorry

end expansion_n_equals_8_l364_364954


namespace number_of_real_solutions_l364_364419

theorem number_of_real_solutions : 
  set.countable {x : ℝ | ∃ n : ℕ, (sqrt (123 - sqrt x) = n)} = 12 :=
by
  sorry

end number_of_real_solutions_l364_364419


namespace max_b_div_a_plus_c_l364_364842

-- Given positive numbers a, b, c
-- equation: b^2 + 2(a + c)b - ac = 0
-- Prove: ∀ a b c : ℝ (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : b^2 + 2*(a + c)*b - a*c = 0),
--         b/(a + c) ≤ (Real.sqrt 5 - 2)/2

theorem max_b_div_a_plus_c (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq : b^2 + 2 * (a + c) * b - a * c = 0) :
  b / (a + c) ≤ (Real.sqrt 5 - 2) / 2 :=
sorry

end max_b_div_a_plus_c_l364_364842


namespace vector_product_magnitude_l364_364782

def vector (α : Type*) := α × α

def mag (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def vec_prod (a b : ℝ × ℝ) : ℝ :=
  mag a * mag b * real.sin (real.atan2 b.2 b.1 - real.atan2 a.2 a.1)

def u : ℝ × ℝ := (2, 0)
def v : ℝ × ℝ := (1, real.sqrt 3)
def u_plus_v : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)

theorem vector_product_magnitude :
  vec_prod u u_plus_v = 2 * real.sqrt 3 :=
by sorry

end vector_product_magnitude_l364_364782


namespace complex_number_in_first_quadrant_l364_364522

-- Define the complex number multiplication condition.
def complex_expression := (Complex.i * (1 - Complex.i))

-- Define the coordinates of the complex number.
def coordinates (z : ℂ) : ℝ × ℝ := (z.re, z.im)

-- Define the first quadrant condition.
def is_first_quadrant (x y : ℝ) : Prop := (x > 0) ∧ (y > 0)

-- The statement to be proven.
theorem complex_number_in_first_quadrant : is_first_quadrant (coordinates complex_expression).fst (coordinates complex_expression).snd :=
by {
  sorry -- Proof omitted as per instructions.
}

end complex_number_in_first_quadrant_l364_364522


namespace fraction_not_equal_3_over_7_l364_364755

theorem fraction_not_equal_3_over_7 :
  ¬ (∃ (x: ℚ), x = 13 / 28 ∧ x = 3 / 7) :=
by
  intro h
  cases h with x hx
  cases hx with h1 h2
  have : 13 * 7 = 3 * 28 := by rw [h1, h2]
  norm_num at this
  sorry

end fraction_not_equal_3_over_7_l364_364755


namespace gcd_of_gx_and_x_l364_364847

theorem gcd_of_gx_and_x (x : ℤ) (hx : x % 11739 = 0) :
  Int.gcd ((3 * x + 4) * (5 * x + 3) * (11 * x + 5) * (x + 11)) x = 3 :=
sorry

end gcd_of_gx_and_x_l364_364847


namespace sequence_term_geometric_l364_364835

theorem sequence_term_geometric :
  ∀ (a : ℕ → ℕ), 
    a 1 = 1 →
    (∀ n, n ≥ 2 → (a n) / (a (n - 1)) = 2^(n-1)) →
    a 101 = 2^5050 :=
by
  sorry

end sequence_term_geometric_l364_364835


namespace probability_composite_l364_364494

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def non_composite_outcomes : ℕ := 1 + 12

theorem probability_composite :
  let total_outcomes := 6^4 in
  let non_composite := non_composite_outcomes in
  let composite_probability := (total_outcomes - non_composite) / total_outcomes.toRat in
  composite_probability = 1283 / 1296 := by
  let total_outcomes := 6^4
  let non_composite := 1 + 12
  let composite_probability := (total_outcomes - non_composite).toRat / total_outcomes.toRat
  sorry

end probability_composite_l364_364494


namespace mult_3_smallest_f_2017_l364_364778

def f : ℕ → ℤ
| 0       := 0
| (n+1) := if (n + 1) % 2 = 0 then -f (n / 2) else f n + 1

theorem mult_3 (n : ℕ) : (3 ∣ n ↔ 3 ∣ f n) :=
by sorry

noncomputable def smallest_n := Inf {n : ℕ | f n = 2017}

theorem smallest_f_2017 : f smallest_n = 2017 :=
by sorry

end mult_3_smallest_f_2017_l364_364778


namespace num_real_numbers_l364_364426

theorem num_real_numbers (x : ℝ) : 
  (∃ n : ℕ, n ≤ 11 ∧ x = (123 - n^2)^2) ↔ 
  (∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ (x = (123 - s^2)^2) ∧ 
  (∃! x, ∃ k : ℕ, 0 ≤ k ∧ k ≤ 11 ∧ x = (123 - k^2)^2)) :=
by
  sorry

end num_real_numbers_l364_364426


namespace amount_B_l364_364710

noncomputable def A : ℝ := sorry -- Definition of A
noncomputable def B : ℝ := sorry -- Definition of B

-- Conditions
def condition1 : Prop := A + B = 100
def condition2 : Prop := (3 / 10) * A = (1 / 5) * B

-- Statement to prove
theorem amount_B : condition1 ∧ condition2 → B = 60 :=
by
  intros
  sorry

end amount_B_l364_364710


namespace distinct_real_x_l364_364416

theorem distinct_real_x (x : ℝ) :
  (∃! x, ∃ (s : ℝ), s ∈ set.Icc 0 11 ∧ s = real.sqrt (123 - real.sqrt x)) =
  12 := sorry

end distinct_real_x_l364_364416


namespace first_sample_blood_cells_l364_364749

theorem first_sample_blood_cells (T S2 S1 : ℕ) (h1 : T = 7341) (h2 : S2 = 3120) : S1 = T - S2 := by
  rw [h1, h2]
  rfl  -- since 7341 - 3120 = 4221

end first_sample_blood_cells_l364_364749


namespace perpendicular_BF_AE_l364_364244

variable {α : Type*} [EuclideanGeometry α]

-- Given triangle ABC with ∠ABC = 90°
variables {A B C D E F : α}
variables (h_triangle : IsRightTriangle A B C)
variables (hD_on_AC : LiesOn D (line_through A C))
variables (hE_on_BC : LiesOn E (line_through B C))
variables (hBD_perp_AC : Perpendicular B D (line_through A C))
variables (hDE_perp_BC : Perpendicular D E (line_through B C))
variables (h_circumcircle : CirclePassingThrough C D E)
variables (hF_on_AE : LiesOn F (line_through A E))
variables (hF_on_circle : LiesOn F (circumcircle [C, D, E]))

-- Goal: Prove that BF ⊥ AE
theorem perpendicular_BF_AE :
  Perpendicular B F (line_through A E) :=
sorry

end perpendicular_BF_AE_l364_364244


namespace oranges_per_box_l364_364751

theorem oranges_per_box
  (total_oranges : ℕ)
  (boxes : ℕ)
  (h1 : total_oranges = 35)
  (h2 : boxes = 7) :
  total_oranges / boxes = 5 := by
  sorry

end oranges_per_box_l364_364751


namespace find_fraction_value_l364_364549

variable (a b : ℝ)
variable (h1 : b > a)
variable (h2 : a > 0)
variable (h3 : a / b + b / a = 4)

theorem find_fraction_value (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a / b + b / a = 4) : (a + b) / (a - b) = Real.sqrt 3 := by
  sorry

end find_fraction_value_l364_364549


namespace dice_product_composite_probability_l364_364481

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

-- This function calculates the probability of an event occurring by counting the favorable and total outcomes.
def probability (favorable total : ℕ) : ℚ :=
  favorable / total

noncomputable def probability_of_composite_product : ℚ :=
  probability 1283 1296

theorem dice_product_composite_probability : probability_of_composite_product = 1283 / 1296 := sorry

end dice_product_composite_probability_l364_364481


namespace equal_tetrahedra_volumes_l364_364855

noncomputable def points_on_planes (A B C D A' B' C' D' : ℝ³) (P Q : set ℝ³) : Prop :=
  (A ∈ P) ∧ (B ∈ P) ∧ (C ∈ P) ∧ (D ∈ P) ∧
  (A' ∈ Q) ∧ (B' ∈ Q) ∧ (C' ∈ Q) ∧ (D' ∈ Q) ∧
  (line_through A A' ∣∣ line_through B B') ∧
  (line_through B B' ∣∣ line_through C C') ∧
  (line_through C C' ∣∣ line_through D D') ∧
  (line_through A A' ∣∣ line_through D D') ∧
  (∀ X Y, {X, Y} ⊆ {A, B, C, D, A', B', C', D'} → X ≠ Y) ∧
  (∀ X Y Z, {X, Y, Z} ⊆ {A, B, C, D, A', B', C', D'} → ¬ collinear ℝ³ ({X, Y, Z}))

theorem equal_tetrahedra_volumes
  (A B C D A' B' C' D' : ℝ³) (P Q : set ℝ³)
  (h : points_on_planes A B C D A' B' C' D' P Q) :
  volume_of_tetrahedron A B C D' = volume_of_tetrahedron A' B' C' D :=
sorry

end equal_tetrahedra_volumes_l364_364855


namespace num_real_satisfying_x_l364_364444

theorem num_real_satisfying_x : 
  (∃ (x : ℝ), ∃ (s : ℕ) (hs : s ∈ set.Ico 0 (floor (sqrt 123) + 1)), 
   s = int.of_nat (floor (sqrt (123 - sqrt x)))) →
   12 := sorry

end num_real_satisfying_x_l364_364444


namespace eccentricity_of_given_ellipse_l364_364559

-- Definitions for the conditions
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) : set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ (x^2) / (a^2) + (y^2) / (b^2) = 1}

def isosceles_right_triangle (A B C : (ℝ × ℝ)) : Prop :=
  ∃ (x1 y1 x2 y2 x3 y3 h1 h2 h3 : ℝ), A = (x1, y1) ∧ B = (x2, y2) ∧ C = (x3, y3) ∧
  (x1 - x2)^2 + (y1 - y2)^2 = (x1 - x3)^2 + (y1 - y3)^2 ∧
  ((x1 - x2)^2 + (y1 - y2)^2) = ((x1 - x3)^2 + (y1 - y3)^2) + ((x2 - x3)^2 + (y2 - y3)^2)

def eccentricity (F1 F2 : (ℝ × ℝ)) : ℝ :=
  let c := dist F1 F2 / 2 in
  c / (sqrt (c^2 + (dist (0, 0) F1)^2))

noncomputable def eccentricity_of_isosceles_right_triangle_ellipse
  (a b : ℝ) (hp : a > b ∧ b > 0)
  (F1 F2 P : (ℝ × ℝ))
  (hF1F2 : P ∈ ellipse a b hp)
  (hTriangle : isosceles_right_triangle F1 F2 P) : ℝ :=
eccentricity F1 F2

-- Statement to prove the eccentricity of the ellipse is sqrt(2) - 1.
theorem eccentricity_of_given_ellipse (a b : ℝ) (hp : a > b ∧ b > 0)
  (F1 F2 P : (ℝ × ℝ))
  (hF1F2 : P ∈ ellipse a b hp)
  (hTriangle : isosceles_right_triangle F1 F2 P) :
  eccentricity_of_isosceles_right_triangle_ellipse a b hp F1 F2 P hF1F2 hTriangle = sqrt 2 - 1 :=
sorry

end eccentricity_of_given_ellipse_l364_364559


namespace mass_percentage_bromine_in_AlBr3_l364_364809

noncomputable def molar_mass_Al : ℝ := 26.98
noncomputable def molar_mass_Br : ℝ := 79.90
noncomputable def molar_mass_AlBr3 : ℝ := molar_mass_Al + 3 * molar_mass_Br

theorem mass_percentage_bromine_in_AlBr3 :
  (3 * molar_mass_Br / molar_mass_AlBr3) * 100 ≈ 89.89 :=
by
  have h_sum : molar_mass_AlBr3 = molar_mass_Al + 3 * molar_mass_Br := rfl
  have h_bromine := (3 * molar_mass_Br / molar_mass_AlBr3) * 100
  sorry

end mass_percentage_bromine_in_AlBr3_l364_364809


namespace repeating_decimal_to_fraction_l364_364799

theorem repeating_decimal_to_fraction :
  ∀ (r q : ℚ), r = 6 / 10 ∧ q = 13 / 99 → 0.613.repeat recurring (1, 2) = 362 / 495 :=
by
  sorry

end repeating_decimal_to_fraction_l364_364799


namespace tan_p1_theta_rational_divisible_by_p_l364_364953

theorem tan_p1_theta_rational_divisible_by_p
  (p : ℕ) (hp_prime : Nat.Prime p) (hp_mod4 : p % 4 = 3)
  (θ : ℝ) (r : ℚ) (rational_tanθ : tan θ = r) :
  ∃ (u v : ℤ), (v > 0) ∧ (Int.gcd u v = 1) ∧ (u % p = 0) ∧ (tan ((p + 1) * θ) = (u : ℝ) / (v : ℝ)) :=
by
  sorry

end tan_p1_theta_rational_divisible_by_p_l364_364953


namespace range_of_a_l364_364845

variable (x a : ℝ)

def p : Prop := x^2 - 2 * x - 3 < 0
def q : Prop := x^2 - a * x - 2 * a^2 < 0

theorem range_of_a (h : ∀ x, p x → q x) (hsuff : ∃ x, q x ∧ ¬ p x) : -1/2 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l364_364845


namespace reflection_of_point_l364_364609

def reflect_across_y_neg_x (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P
  (y, x)

theorem reflection_of_point
  (P : ℝ × ℝ)
  (h : P = (8, -3)) :
  reflect_across_y_neg_x P = (3, -8) :=
by
  rw [h]
  sorry

end reflection_of_point_l364_364609


namespace quadrilateral_KLNM_area_l364_364056

theorem quadrilateral_KLNM_area (A B C M N K L : Type*)
    [line_segment A C M] [line_segment A C N] [line_segment A B K] [line_segment A B L]
    (h1: AM : MN : NC = 1 : 3 : 1)
    (h2: AK = KL = LB)
    (area_ABC_eq_1: area (triangle A B C) = 1) :
    area (quadrilateral K L N M) = 7 / 15 :=
sorry

end quadrilateral_KLNM_area_l364_364056


namespace num_two_digit_numbers_l364_364861

theorem num_two_digit_numbers (digits : Finset ℕ) (h : digits = {1, 3, 5, 8}) 
    (no_repeat : ∀ x ∈ digits, ∀ y ∈ digits, x ≠ y → set.pairwiseₓ digits (≠)) :
    (digits.card = 4) → 
    (∑ x in digits, (digits.erase x).card) = 12 := 
by
  intros h_card
  rw finset.card_eq_sum_ones
  have h_erase : ∀ x ∈ digits, (digits.erase x).card = 3,
  { intros x hx,
    rw [finset.card_erase_of_mem hx, h_card],
    exact nat.pred_succ 3, },
  simp [h, h_erase, finset.card_eq_sum_ones]
  sorry

end num_two_digit_numbers_l364_364861


namespace sum_of_real_factors_l364_364949

theorem sum_of_real_factors (P : ℝ → ℝ) (k_vals : List ℝ) :
  P = λ x, k * x^3 + 2 * k^2 * x^2 + k^3 → 
  (∀ k ∈ k_vals, ∃ k, P 2 = 0) → 
  k_vals.sum = -8 :=
by
  sorry

end sum_of_real_factors_l364_364949


namespace find_coordinates_of_point_P_l364_364904

theorem find_coordinates_of_point_P : 
  ∃ (y₀ : ℝ), 
    (y₀ = 2 ∨ y₀ = 8) ∧ 
    ∀ (P : ℝ × ℝ × ℝ), P = (0, y₀, 0) → (dist P (2, 5, -6) = 7) :=
by {
  let P := (0, y₀, 0),
  have h : dist P (2, 5, -6) = 7,
  
  sorry
}

end find_coordinates_of_point_P_l364_364904


namespace compare_growth_rates_l364_364365

noncomputable def f (x : ℝ) := x^2
noncomputable def g (x : ℝ) := 2^x
noncomputable def h (x : ℝ) := Real.logb 2 x

theorem compare_growth_rates (x : ℝ) (hx : x > 4) : g x > f x ∧ f x > h x :=
by
  sorry

end compare_growth_rates_l364_364365


namespace OE_perp_CD_l364_364759

noncomputable def midpoint (A B : V) : V :=
  (A + B) / 2

noncomputable def centroid (A C D : V) : V :=
  (A + C + D) / 3

theorem OE_perp_CD (A B C O D E : V) 
  (hO : is_circumcenter O A B C) 
  (hD : D = midpoint A B) 
  (hE : E = centroid A C D) 
  (hAB_eq_AC : ∥A - B∥ = ∥A - C∥) :
  is_perpendicular (E - O) (D - C) :=
sorry

end OE_perp_CD_l364_364759


namespace apple_count_l364_364183

-- Define constants for the problem
constant bags: ℕ → Prop
constant total_apples: ℕ
constant apples_in_bag1: ℕ := 12
constant apples_in_bag2: ℕ := 6

-- Define the conditions
axiom condition1 (n: ℕ) : bags n → (n = apples_in_bag1 ∨ n = apples_in_bag2)
axiom condition2 : 70 ≤ total_apples ∧ total_apples ≤ 80
axiom condition3 : total_apples % 6 = 0

-- State the proof problem
theorem apple_count :
  total_apples = 72 ∨ total_apples = 78 :=
sorry

end apple_count_l364_364183


namespace gcd_lcm_product_180_l364_364696

theorem gcd_lcm_product_180 :
  ∃ (a b : ℕ), (gcd a b) * (lcm a b) = 180 ∧ 
  let possible_gcd_values := 
    {d | ∃ a b : ℕ, gcd a b = d ∧ (gcd a b) * (lcm a b) = 180} in
  possible_gcd_values.card = 7 :=
by
  sorry

end gcd_lcm_product_180_l364_364696


namespace largest_prime_factor_3328_l364_364220

theorem largest_prime_factor_3328 : ∃ p : ℕ, p.prime ∧ p ∣ 3328 ∧ ∀ q : ℕ, q.prime ∧ q ∣ 3328 → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_13, },  -- 13 is prime
  split,
  { exact dvd.intro (2^8) rfl, },  -- 13 divides 3328
  { 
    intros q hq_prime hq_dvd, 
    sorry,  -- it remains to show q ≤ 13 for all prime q dividing 3328
  },
end

end largest_prime_factor_3328_l364_364220


namespace num_real_x_l364_364409

theorem num_real_x (a b : ℝ) (h1 : a = 123) (h2 : b = 11) :
  ∃ n : ℕ, n = 12 ∧
  ∀ k : ℕ, k ≤ b → ∃ x : ℝ, x = (a - k^2)^2 :=
by
  sorry

end num_real_x_l364_364409


namespace largest_prime_factor_of_3328_l364_364218

theorem largest_prime_factor_of_3328 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 3328 ∧ (∀ q : ℕ, Nat.Prime q ∧ q ∣ 3328 → q ≤ p) :=
by
  have h : 3328 = 2^8 * 13 := by norm_num
  use 13
  split
  · exact Nat.prime_13
  split
  · rw h
    exact dvd_mul_right _ _
  · intros q hq1 hq2
    rw h at hq2
    cases Nat.dvd_mul.mp hq2 with hq2 hq2
    · exact Nat.le_of_dvd (Nat.pos_pow_of_pos 8 zero_lt_two) hq2
    · exact Nat.le_of_eq (EuclideanDomain.gcd_eq_right hq1 hq2).symm

end largest_prime_factor_of_3328_l364_364218


namespace minimum_unused_area_for_given_shapes_l364_364733

def remaining_area (side_length : ℕ) (total_area used_area : ℕ) : ℕ :=
  total_area - used_area

theorem minimum_unused_area_for_given_shapes : (remaining_area 5 (5 * 5) (2 * 2 + 1 * 3 + 2 * 1) = 16) :=
by
  -- We skip the proof here, as instructed.
  sorry

end minimum_unused_area_for_given_shapes_l364_364733


namespace average_value_continuous_l364_364370

noncomputable def average_value (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (1 / (b - a)) * ∫ x in a..b, f x

theorem average_value_continuous (f : ℝ → ℝ) (a b : ℝ) (h : ContinuousOn f (Set.Icc a b)) :
  (average_value f a b) = (1 / (b - a)) * (∫ x in a..b, f x) :=
by
  sorry

end average_value_continuous_l364_364370


namespace Liz_needs_more_money_l364_364043

theorem Liz_needs_more_money (P : ℝ) (h1 : P = 30000 + 2500) (h2 : 0.80 * P = 26000) : 30000 - (0.80 * P) = 4000 :=
by
  sorry

end Liz_needs_more_money_l364_364043


namespace extremum_condition_monotonic_intervals_b_range_l364_364850

def f (x : ℝ) (a : ℝ) := a * Real.log (1 + x) + x^2 - 10 * x

def extremum_at (x : ℝ) (a : ℝ) : Prop := 
  let f' := (a / (1 + x)) + 2 * x - 10
  f' = 0

def increasing_in_interval (a : ℝ) (interval : Set ℝ) : Prop := 
  ∀ x y : ℝ, x ∈ interval → y ∈ interval → x < y → f x a < f y a

def decreasing_in_interval (a : ℝ) (interval : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ interval → y ∈ interval → x < y → f x a > f y a

def intersects_at_three_points (a b : ℝ) : Prop :=
  32 * Real.log 2 - 21 < b ∧ b < 16 * Real.log 2 - 9

theorem extremum_condition (a : ℝ) :
  extremum_at 3 a → a = 16 :=
sorry

theorem monotonic_intervals (a : ℝ) :
  a = 16 →
  increasing_in_interval a {x | -1 < x ∧ x < 1} ∧
  decreasing_in_interval a {x | 1 < x ∧ x < 3} ∧
  increasing_in_interval a {x | 3 < x} :=
sorry

theorem b_range (a b : ℝ) :
  a = 16 →
  intersects_at_three_points a b →
  32 * Real.log 2 - 21 < b ∧ b < 16 * Real.log 2 - 9 :=
sorry

end extremum_condition_monotonic_intervals_b_range_l364_364850


namespace unit_place_3_pow_34_l364_364240

theorem unit_place_3_pow_34 : Nat.mod (3^34) 10 = 9 :=
by
  sorry

end unit_place_3_pow_34_l364_364240


namespace vertex_A_east_probability_l364_364937

theorem vertex_A_east_probability (A B C : Type) (angle_A : ℝ) 
  (h : angle_A = 40) : 
  probability_A_east(A, B, C) = 7 / 18 := by
  sorry

end vertex_A_east_probability_l364_364937


namespace value_of_a_plus_b_l364_364505

theorem value_of_a_plus_b (a b : ℝ) (h : set_of (λ x, (x - a) * (x - b) < 0) = set.Ioo (-1 : ℝ) 2) :
  a + b = 1 :=
by
  sorry

end value_of_a_plus_b_l364_364505


namespace false_props_l364_364382

-- Definitions for conditions
def prop1 :=
  ∀ (a b c d : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ (a * d = b * c) → 
  (a / b = b / c ∧ b / c = c / d)

def prop2 :=
  ∀ (a : ℕ), (∃ k : ℕ, a = 2 * k) → (a % 2 = 0)

def prop3 :=
  ∀ (A : ℝ), (A > 30) → (Real.sin (A * Real.pi / 180) > 1 / 2)

-- Theorem statement
theorem false_props : (¬ prop1) ∧ (¬ prop3) :=
by sorry

end false_props_l364_364382


namespace distinct_real_x_l364_364415

theorem distinct_real_x (x : ℝ) :
  (∃! x, ∃ (s : ℝ), s ∈ set.Icc 0 11 ∧ s = real.sqrt (123 - real.sqrt x)) =
  12 := sorry

end distinct_real_x_l364_364415


namespace infinite_intersection_iff_edge_l364_364962

-- Definitions and conditions
variables {V : Type} [fintype V] (G : simple_graph V)
variables (n : ℕ) (vertices : fin n → V)
variables (H : fin n → set ℕ)

-- Prove the statement, i.e., that we can construct the sets H_1, ..., H_n
theorem infinite_intersection_iff_edge :
  (∃ (H : fin n → set ℕ), 
    ∀ i j : fin n, 
      (H i ∩ H j).infinite ↔ G.adj (vertices i) (vertices j)) :=
sorry

end infinite_intersection_iff_edge_l364_364962


namespace max_abs_z_l364_364024

-- Define the complex unit i
def i : ℂ := complex.I

-- Define the conditions
def z (x y : ℝ) : ℂ := x + y * i
def center : ℂ := 3 + 4 * i

-- Define the property of z
def property (x y : ℝ) : Prop :=
  complex.abs (z x y - center) = 1

-- The theorem to prove that the maximum value of |z| is 6 given the property
theorem max_abs_z (x y : ℝ) (h : property x y) : complex.abs (z x y) ≤ 6 := sorry

end max_abs_z_l364_364024


namespace cross_product_result_l364_364820

variables (v w : ℝ^3)
def cross_product : ℝ^3 → ℝ^3 → ℝ^3 := fun v w => 
  ⟨v.y * w.z - v.z * w.y, v.z * w.x - v.x * w.z, v.x * w.y - v.y * w.x⟩

#check prod.fst
#check prod.swd

def v_w_cross : ℝ^3 := ⟨3, -1, 2⟩

theorem cross_product_result (v w : ℝ^3) (hv : cross_product v w = v_w_cross) :
  cross_product (3 • (v + w)) (2 • (v - w)) = 
  ⟨-36, 12, -24⟩ := 
sorry

end cross_product_result_l364_364820


namespace convert_line_eq_to_slope_intercept_l364_364273

theorem convert_line_eq_to_slope_intercept :
  ∀ x y : ℝ, (⟨2, -1⟩: ℝ × ℝ) • (⟨x, y⟩ - ⟨1, -1⟩) = 0 → y = 2 * x - 3 :=
by
  sorry

end convert_line_eq_to_slope_intercept_l364_364273


namespace apple_bags_l364_364177

theorem apple_bags (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apple_bags_l364_364177


namespace percentage_z_of_x_l364_364371

noncomputable def x (y : ℝ) := 1.20 * y
noncomputable def z (x y : ℝ) := 0.85 * (x + y)
noncomputable def percentage (z x : ℝ) := (z / x) * 100

theorem percentage_z_of_x (y : ℝ) : percentage (z (x y) y) (x y) = 155.83 := by
  have x_eq : x y = 1.20 * y := rfl
  have sum_eq : x y + y = 2.20 * y := by linarith [x_eq]
  have z_eq : z (x y) y = 1.87 * y := by simp [x_eq, sum_eq, z]
  have percentage_eq : percentage (1.87 * y) (1.20 * y) = 155.83 := by
    rw [percentage, z_eq, x_eq]
    have h : (1.87 * y) / (1.20 * y) = 1.87 / 1.20 := by field_simp [ne_of_gt (show 0 < y from by linarith)]
    rw [h]
    norm_num
  exact percentage_eq
  
-- use sorry here instead of proving the final step directly

end percentage_z_of_x_l364_364371


namespace math_problem_proof_l364_364393

-- Definition of the first part of the problem to find the values of m and n.
def find_values_of_m_n (m n x : ℝ) : Prop :=
  (∀ x, (0 ≤ x ∧ x ≤ 4) ↔ abs (x - m) ≤ n)

-- Definition for the minimum value problem.
def min_value_of_a_b (a b m n : ℝ) : Prop :=
  (a > 0 ∧ b > 0 ∧ a + b = (m / a) + (n / b)) → (a + b ≥ 2 * real.sqrt (m * n))

-- The complete statement combining both.
theorem math_problem_proof :
  (find_values_of_m_n 2 2 x) →
  (∀ a b, min_value_of_a_b a b 2 2) :=
begin
  sorry
end

end math_problem_proof_l364_364393


namespace break_even_performances_l364_364074

def totalCost (x : ℕ) : ℕ := 81000 + 7000 * x
def totalRevenue (x : ℕ) : ℕ := 16000 * x

theorem break_even_performances : ∃ x : ℕ, totalCost x = totalRevenue x ∧ x = 9 := 
by
  sorry

end break_even_performances_l364_364074


namespace sum_last_two_digits_pow_mod_eq_zero_l364_364676

/-
Given condition: 
Sum of the last two digits of \( 9^{25} + 11^{25} \)
-/
theorem sum_last_two_digits_pow_mod_eq_zero : 
  let a := 9
  let b := 11
  let n := 25 
  (a ^ n + b ^ n) % 100 = 0 :=
by
  sorry

end sum_last_two_digits_pow_mod_eq_zero_l364_364676


namespace count_non_confusing_numbers_counts_to_88060_l364_364572

-- Definition of valid digits that look the same, or have a unique inverse, when reversed.
def valid_digits (d : Char) : Prop :=
  d = '0' ∨ d = '1' ∨ d = '6' ∨ d = '8' ∨ d = '9'

-- Checking if reversing a number in given digit range causes confusion.
def not_confusing (n : Nat) : Prop := 
 if 10000 ≤ n ∧ n ≤ 99999 then
   let s := n.toString
   valid_digits s[0] ∧ valid_digits s[1] ∧ valid_digits s[2] ∧ 
   valid_digits s[3] ∧ valid_digits s[4] ∧ 
   if s[0] = '6' then s[4] = '9' else if s[0] = '9' then s[4] = '6' else s[0] = s[4] ∨
   if s[1] = '6' then s[3] = '9' else if s[1] = '9' then s[3] = '6' else s[1] = s[3]
 else false

-- The main theorem statement 
theorem count_non_confusing_numbers_counts_to_88060 : 
  (90000 - (List.range' 10000 90000).countp (λ n => not_confusing n) = 88060) := 
by 
  -- The proof is supposed to be here
  sorry

end count_non_confusing_numbers_counts_to_88060_l364_364572


namespace circle_symmetry_proof_l364_364092

def symm_circle_eq : Prop := ∀ (x y : ℝ),
  (circle_eq : (x + 2)^2 + y^2 = 2016) →
  (line_eq : x - y + 1 = 0) →
  (symm_circle_eq : (x + 1)^2 + (y + 1)^2 = 2016)

theorem circle_symmetry_proof (x y : ℝ) 
  (circle_eq : (x + 2)^2 + y^2 = 2016)
  (line_eq : x - y + 1 = 0) : 
  (x + 1)^2 + (y + 1)^2 = 2016 :=
sorry

end circle_symmetry_proof_l364_364092


namespace apple_count_l364_364147

/--
Given the total number of apples in several bags, where each bag contains either 12 or 6 apples,
and knowing that the total number of apples is between 70 and 80 inclusive,
prove that the total number of apples can only be 72 or 78.
-/
theorem apple_count (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : ∃ k : ℕ, n = 6 * k) : n = 72 ∨ n = 78 :=
by {
  sorry
}

end apple_count_l364_364147


namespace magnitude_z_l364_364036

theorem magnitude_z (z : ℂ) (h : (1 + complex.i) * z = 2 * complex.i) : complex.abs z = real.sqrt 2 := 
sorry

end magnitude_z_l364_364036


namespace gcd_values_count_l364_364690

theorem gcd_values_count :
  ∃ a b : ℕ, (gcd a b) * (nat.lcm a b) = 180 ∧
    set.card { gcd a b | ∃ a b, a * b = 180 } = 8 :=
by
  -- Problem statement as provided by conditions and question
  -- Definitions and notations are provided correctly and fully, proof is omitted
  sorry

end gcd_values_count_l364_364690


namespace dot_product_formula_not_perpendicular_parallel_k_values_l364_364876

variables {α β : ℝ} (k : ℝ) (h : k > 0)
def vector_a := (Real.cos α, Real.sin α)
def vector_b := (Real.cos β, Real.sin β)

def mag (v : ℝ × ℝ) := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def dot_product (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_formula : mag (k • vector_a + vector_b) = Real.sqrt 3 * mag (vector_a - k • vector_b) → (dot_product vector_a vector_b = (k ^ 2 + 1) / (4 * k)) := 
by
  intro h
  sorry

theorem not_perpendicular : mag (k • vector_a + vector_b) = Real.sqrt 3 * mag (vector_a - k • vector_b) → ¬ (dot_product vector_a vector_b = 0) := 
by
  intro h
  sorry

theorem parallel_k_values : mag (k • vector_a + vector_b) = Real.sqrt 3 * mag (vector_a - k • vector_b) → 
  ((dot_product vector_a vector_b = 1) → (k = 2 + Real.sqrt 3 ∨ k = 2 - Real.sqrt 3)) :=
by
  intro h
  sorry

end dot_product_formula_not_perpendicular_parallel_k_values_l364_364876


namespace greatest_three_digit_multiple_of_23_is_991_l364_364661

theorem greatest_three_digit_multiple_of_23_is_991 :
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 23 = 0) ∧ ∀ m : ℤ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 23 = 0) → m ≤ n :=
begin
  use 991,
  -- proof steps go here
  sorry
end

end greatest_three_digit_multiple_of_23_is_991_l364_364661


namespace possible_apple_counts_l364_364128

theorem possible_apple_counts (n : ℕ) (h₁ : 70 ≤ n) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end possible_apple_counts_l364_364128


namespace probability_at_least_three_heads_l364_364079

theorem probability_at_least_three_heads :
  let outcomes := Finset.powerset (Finset.range 5)
  let favorable := outcomes.filter (λ s, s.card ≥ 3)
  (favorable.card : ℚ) / outcomes.card = 1 / 2 :=
by
  sorry

end probability_at_least_three_heads_l364_364079


namespace apple_bags_l364_364174

theorem apple_bags (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apple_bags_l364_364174


namespace count_distinct_x_l364_364439

theorem count_distinct_x :
  { x : ℝ | ∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ s = floor (sqrt (123 - sqrt x)) } = 12 :=
sorry

end count_distinct_x_l364_364439


namespace probability_composite_product_l364_364468

theorem probability_composite_product :
  let dice_faces := 6
  let rolls := 4
  let total_outcomes := dice_faces ^ rolls
  let non_composite_cases := 13
  let non_composite_probability := non_composite_cases / total_outcomes
  let composite_probability := 1 - non_composite_probability
  composite_probability = 1283 / 1296 := by
  sorry

end probability_composite_product_l364_364468


namespace max_boys_for_99_apples_min_apples_given_99_among_10_boys_l364_364973

-- Maximum number of boys that can receive apples such that each boy receives a different number of apples summing to at most 99
theorem max_boys_for_99_apples : ∃ n : ℕ, (∑ i in (finset.range (n + 1)), i) ≤ 99 ∧ ∀ m > n, (∑ i in (finset.range (m + 1)), i) > 99 :=
begin
    -- Placeholder for the proof. This statement encapsulates the essence of finding the maximum number of boys.
    sorry
end

-- Maximum number of apples received by the boy who got the fewest when there are 10 boys
theorem min_apples_given_99_among_10_boys : 
    ∃ m : ℕ, ∀ (apples : finset ℕ), apples.card = 10 → (∀ x y ∈ apples, x ≠ y) →
    (finset.sum apples id = 99) → (min' apples (by finish)) = m ∧ m = 5 :=
begin
    -- Placeholder for the proof. This statement encapsulates the essence of finding the fewest apples received.
    sorry
end

end max_boys_for_99_apples_min_apples_given_99_among_10_boys_l364_364973


namespace sequence_general_formula_l364_364873

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then 3
  else if n = 1 then 8
  else sequence (n - 1) * 2 + sequence (n - 2) * 2

theorem sequence_general_formula :
  ∀ n : ℕ, n > 0 →
  sequence n = 
  (2 + Real.sqrt 3) / (2 * Real.sqrt 3) * (1 + Real.sqrt 3)^n +
  (Real.sqrt 3 - 2) / (2 * Real.sqrt 3) * (1 - Real.sqrt 3)^n :=
sorry

end sequence_general_formula_l364_364873


namespace number_of_combinations_6x6_no_two_same_row_column_l364_364285

theorem number_of_combinations_6x6_no_two_same_row_column : 
  let grid_size := 6
  let blocks_to_select := 4
  let choose := λ n k => n.choose k
  let factorial := λ n => n.factorial
  choose grid_size blocks_to_select * choose grid_size blocks_to_select * factorial blocks_to_select = 5400 :=
by
  let grid_size := 6
  let blocks_to_select := 4
  let choose := λ n k => n.choose k
  let factorial := λ n => n.factorial
  sorry

end number_of_combinations_6x6_no_two_same_row_column_l364_364285


namespace find_A_unique_triangle_and_height_l364_364000

-- Define the necessary variables and conditions
variables {A B C : ℝ} {a b c h : ℝ}
variables (π : ℝ) (sqrt2 sqrt3 : ℝ) (sin cos : ℝ → ℝ)
variables (sinC : ℝ) (f : ∀ {x : ℝ}, x > 0 → x < π → sin x = sin x.cosπ)

-- Conditions and assumptions
variables (H1 : sinC = sin A * cos B + sqrt2 / 2 * sin (A + C))
variables (H2 : a > 0) (H3 : b > 0) (H4 : c > 0)

-- Conditions for uniqueness
variables (H5 : a = 2) (H6 : b = sqrt2) (H7 : sinC = sqrt3 / 2)
variables (H8 : a = sqrt5) (H9 : b = sqrt2)

-- Prove angle A
theorem find_A : A = π / 4 :=
by sorry

-- Prove uniqueness and height h under Condition 2
theorem unique_triangle_and_height : (a = sqrt5 ∧ b = sqrt2) → (a^2 - b^2 - c^2 + sqrt2 * b * c = 0) → 
                                       c = 3 ∧ h = (3 * sqrt5) / 5 :=
by sorry

end find_A_unique_triangle_and_height_l364_364000


namespace apple_bags_l364_364175

theorem apple_bags (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apple_bags_l364_364175


namespace wire_length_around_square_field_l364_364236

theorem wire_length_around_square_field (area : ℝ) (times : ℕ) (wire_length : ℝ) 
    (h1 : area = 69696) (h2 : times = 15) : wire_length = 15840 :=
by
  sorry

end wire_length_around_square_field_l364_364236


namespace apple_count_l364_364148

/--
Given the total number of apples in several bags, where each bag contains either 12 or 6 apples,
and knowing that the total number of apples is between 70 and 80 inclusive,
prove that the total number of apples can only be 72 or 78.
-/
theorem apple_count (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : ∃ k : ℕ, n = 6 * k) : n = 72 ∨ n = 78 :=
by {
  sorry
}

end apple_count_l364_364148


namespace Vasya_password_combinations_l364_364204

theorem Vasya_password_combinations :
  let digits := {d | d ∈ {0, 1, 3, 4, 5, 6, 7, 8, 9}}
  let non_adjacent (x y : ℕ) := x ≠ y
  let A_choices := {d ∈ digits | d ≠ 0}
  let B_choices := digits
  let C_choices (B : ℕ) := {d ∈ digits | non_adjacent d B}
  A_choices.card * B_choices.card * C_choices.card = 576 :=
by
  sorry

end Vasya_password_combinations_l364_364204


namespace max_area_curves_intersection_l364_364589

open Real

def C₁ (x : ℝ) : ℝ := x^3 - x
def C₂ (x a : ℝ) : ℝ := (x - a)^3 - (x - a)

theorem max_area_curves_intersection (a : ℝ) (h : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ C₁ x₁ = C₂ x₁ a ∧ C₁ x₂ = C₂ x₂ a) :
  ∃ A_max : ℝ, A_max = 3 / 4 :=
by
  -- TODO: Provide the proof here
  sorry

end max_area_curves_intersection_l364_364589


namespace simplify_sin_cos_expr_cos_pi_six_alpha_expr_l364_364254

open Real

-- Problem (1)
theorem simplify_sin_cos_expr (x : ℝ) :
  (sin x ^ 2 / (sin x - cos x)) - ((sin x + cos x) / (tan x ^ 2 - 1)) - sin x = cos x :=
sorry

-- Problem (2)
theorem cos_pi_six_alpha_expr (α : ℝ) (h : cos (π / 6 - α) = sqrt 3 / 3) :
  cos (5 * π / 6 + α) + cos (4 * π / 3 + α) ^ 2 = (2 - sqrt 3) / 3 :=
sorry

end simplify_sin_cos_expr_cos_pi_six_alpha_expr_l364_364254


namespace count_distinct_x_l364_364434

theorem count_distinct_x :
  { x : ℝ | ∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ s = floor (sqrt (123 - sqrt x)) } = 12 :=
sorry

end count_distinct_x_l364_364434


namespace apples_total_l364_364171

theorem apples_total (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apples_total_l364_364171


namespace line_circle_intersection_l364_364870

-- Define the line l and circle C
section
variables {t m : ℝ}

def line_l (m : ℝ) : set (ℝ × ℝ) := { p : ℝ × ℝ | ∃ t : ℝ, (p.1 = 3 * t) ∧ (p.2 = 4 * t + m) }

def circle_C : set (ℝ × ℝ) := { p : ℝ × ℝ | (p.1 - 1) ^ 2 + p.2 ^ 2 = 1 }

-- Prove the statement about the possible values of m
theorem line_circle_intersection (m : ℝ) : (∃! p : ℝ × ℝ, p ∈ line_l m ∧ p ∈ circle_C) ↔ m = 1/3 ∨ m = -3 :=
by
  sorry
end

end line_circle_intersection_l364_364870


namespace greatest_three_digit_multiple_of_23_l364_364665

def is_three_digit (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

def is_multiple_of_23 (n : ℕ) : Prop :=
  n % 23 = 0

theorem greatest_three_digit_multiple_of_23 :
  ∀ n, is_three_digit n ∧ is_multiple_of_23 n → n ≤ 989 :=
by
  sorry

end greatest_three_digit_multiple_of_23_l364_364665


namespace arithmetic_mean_of_primes_l364_364803

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 → m ∣ n → m = n)

def arithmetic_mean (nums : List ℕ) : ℕ :=
  nums.sum / nums.length

theorem arithmetic_mean_of_primes (nums : List ℕ) (primes : List ℕ) : 
    nums = [33, 35, 37, 39, 41] ∧ primes = nums.filter is_prime →
    arithmetic_mean primes = 39 :=
by
  intro h
  let primes := [37, 41]
  sorry

end arithmetic_mean_of_primes_l364_364803


namespace area_G_l364_364956

-- Given: P is the midpoint of BC
-- Given: G'_1, G'_2, and G'_3 are the centroids of triangles PBC, PCA, and PAB respectively
-- Given: The area of triangle ABC is 24
-- Prove: The area of triangle G'_1 G'_2 G'_3 is 6

variables {A B C P G'_1 G'_2 G'_3 : Point}
variables (area_ABC : area △ABC = 24)
variables (midpoint_P : isMidpoint P B C)
variables (centroid_G'_1 : isCentroid G'_1 P B C)
variables (centroid_G'_2 : isCentroid G'_2 P C A)
variables (centroid_G'_3 : isCentroid G'_3 P A B)

theorem area_G'_1_G'_2_G'_3 : area △G'_1 G'_2 G'_3 = 6 :=
sorry

end area_G_l364_364956


namespace cost_of_carpeting_l364_364087

noncomputable def cost_per_meter_in_paise (cost : ℝ) (length_in_meters : ℝ) : ℝ :=
  cost * 100 / length_in_meters

theorem cost_of_carpeting (room_length room_breadth carpet_width_m cost_total : ℝ) (h1 : room_length = 15) 
  (h2 : room_breadth = 6) (h3 : carpet_width_m = 0.75) (h4 : cost_total = 36) :
  cost_per_meter_in_paise cost_total (room_length * room_breadth / carpet_width_m) = 30 :=
by
  sorry

end cost_of_carpeting_l364_364087


namespace pascal_triangle_ratios_l364_364507
open Nat

theorem pascal_triangle_ratios :
  ∃ n r : ℕ, 
  (choose n r) * 4 = (choose n (r + 1)) * 3 ∧ 
  (choose n (r + 1)) * 3 = (choose n (r + 2)) * 4 ∧ 
  n = 34 :=
by
  sorry

end pascal_triangle_ratios_l364_364507


namespace circle_reflection_l364_364604

variable (x₀ y₀ : ℝ)

def reflect_over_line_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem circle_reflection 
  (h₁ : x₀ = 8)
  (h₂ : y₀ = -3) :
  reflect_over_line_y_eq_neg_x (x₀, y₀) = (3, -8) := by
  sorry

end circle_reflection_l364_364604


namespace triangle_angle_A_triangle_area_l364_364398

noncomputable def sin_squared_half_sum (B C : ℝ) : ℝ := 
  sin (B + C) / 2 * sin (B + C) / 2

theorem triangle_angle_A 
  (a b c A B C : ℝ)
  (h1 : 4 * sin_squared_half_sum B C - cos (2 * A) = 7 / 2)
  (h2 : 0 < A ∧ A < π)
  : A = π / 3 := sorry

theorem triangle_area 
  (a b c A B C : ℝ)
  (hA : A = π / 3)
  (hcosB : cos B = 3 / 5)
  (ha : a = sqrt 3)
  (hb : b = (sqrt 3 * sin B) / sin 60)
  (hC : C = π - (A + B))
  : (1 / 2) * a * b * sin C = (8 * sqrt 3 + 18) / 25 := sorry

end triangle_angle_A_triangle_area_l364_364398


namespace valid_number_of_apples_l364_364139

theorem valid_number_of_apples (n : ℕ) : 
  (∃ k m : ℕ, n = 12 * k + 6 * m) ∧ (70 ≤ n ∧ n ≤ 80) ↔ (n = 72 ∨ n = 78) :=
by
  sorry

end valid_number_of_apples_l364_364139


namespace numberOfValidCombinations_l364_364529

def isValidCombination (comb : List ℕ) : Prop :=
  comb.length = 6 ∧
  (∀ i, i < 5 → ((comb.nth! i) % 2 = 1) → ((comb.nth! (i + 1)) % 2 = 0))

def allValidCombinations : List (List ℕ) :=
  List.filter isValidCombination (List.permutations [1, 2, 3, 4, 5, 6].map (λ p, p.firstN 6))

theorem numberOfValidCombinations :
  allValidCombinations.length = 1458 :=
  sorry

end numberOfValidCombinations_l364_364529


namespace percentage_alcohol_final_l364_364565

-- Let's define the given conditions
variable (A B totalVolume : ℝ)
variable (percentAlcoholA percentAlcoholB : ℝ)
variable (approxA : ℝ)

-- Assume the conditions
axiom condition1 : percentAlcoholA = 0.20
axiom condition2 : percentAlcoholB = 0.50
axiom condition3 : totalVolume = 15
axiom condition4 : approxA = 10
axiom condition5 : A = approxA
axiom condition6 : B = totalVolume - A

-- The proof statement
theorem percentage_alcohol_final : 
  (0.20 * A + 0.50 * B) / 15 * 100 = 30 :=
by 
  -- Introduce enough structure for Lean to handle the problem.
  sorry

end percentage_alcohol_final_l364_364565


namespace bus_trip_length_l364_364723

theorem bus_trip_length (v T : ℝ) 
    (h1 : 2 * v + (T - 2 * v) * (3 / (2 * v)) + 1 = T / v + 5)
    (h2 : 2 + 30 / v + (T - (2 * v + 30)) * (3 / (2 * v)) + 1 = T / v + 4) : 
    T = 180 :=
    sorry

end bus_trip_length_l364_364723


namespace range_of_m_l364_364966

theorem range_of_m 
  (m : ℝ)
  (hM : -4 ≤ m ∧ m ≤ 4)
  (ellipse : ∀ (x y : ℝ), x^2 / 16 + y^2 / 12 = 1 → y = 0) :
  1 ≤ m ∧ m ≤ 4 := sorry

end range_of_m_l364_364966


namespace apple_count_l364_364141

/--
Given the total number of apples in several bags, where each bag contains either 12 or 6 apples,
and knowing that the total number of apples is between 70 and 80 inclusive,
prove that the total number of apples can only be 72 or 78.
-/
theorem apple_count (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : ∃ k : ℕ, n = 6 * k) : n = 72 ∨ n = 78 :=
by {
  sorry
}

end apple_count_l364_364141


namespace kerry_age_l364_364541

theorem kerry_age (candles_per_box : ℕ) (cost_per_box : ℝ) (total_cost : ℝ) (num_cakes : ℕ) 
    (boxes : ℕ) (total_candles : ℕ) (candles_per_cake : ℝ) (age : ℕ) : 
  (candles_per_box = 22) → 
  (cost_per_box = 4.5) → 
  (total_cost = 27) → 
  (num_cakes = 5) → 
  (boxes = total_cost / cost_per_box) → 
  (total_candles = boxes * candles_per_box) → 
  (candles_per_cake = total_candles / num_cakes) → 
  (age = candles_per_cake.to_nat) → 
  (age = 26) := 
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end kerry_age_l364_364541


namespace maximize_projections_distance_l364_364977

-- Definitions: Triangle ABC and Circumcircle
variables {A B C M : Point}

-- Assume M is on the circumcircle of triangle ABC
def circumcircle (A B C M : Point) : Prop := sorry -- Define the predicate for M being on the circumcircle of ABC

-- Projections P and Q of M on AC and BC
def projection (M : Point) (line_segment : LineSegment) : Point := sorry -- Define the projection of M onto a given line segment

-- Main theorem statement
theorem maximize_projections_distance (h_circ : circumcircle A B C M) :
  ∃ M, ∀ (P Q : Point), 
  P = projection M (line_segment A C) ∧ Q = projection M (line_segment B C) →
  (distance P Q = diameter (circumcircle A B C M)) :=
sorry

end maximize_projections_distance_l364_364977


namespace tiffany_max_points_l364_364191

section
  variables
  (initial_money : ℕ := 3)
  (cost_per_game : ℕ := 1)
  (rings_per_game : ℕ := 5)
  (points_red_bucket : ℕ := 2)
  (points_green_bucket : ℕ := 3)
  (points_miss : ℕ := 0)
  (games_played : ℕ := 2)
  (red_buckets : ℕ := 4)
  (green_buckets : ℕ := 5)
  (additional_games : ℕ := initial_money - games_played)
  (points_per_game_from_green_buckets : ℕ := rings_per_game * points_green_bucket)
  (total_points : ℕ := (red_buckets * points_red_bucket) + (green_buckets * points_green_bucket) + (additional_games * points_per_game_from_green_buckets))

  theorem tiffany_max_points : total_points = 38 := 
  sorry
end

end tiffany_max_points_l364_364191


namespace find_expression_for_x_l364_364849

variable (x : ℝ) (hx : x^3 + (1 / x^3) = -52)

theorem find_expression_for_x : x + (1 / x) = -4 :=
by sorry

end find_expression_for_x_l364_364849


namespace sum_abs_val_lt_4_l364_364640

theorem sum_abs_val_lt_4 : 
  (Finset.sum (Finset.filter (λ x : ℤ, abs x < 4) (Finset.Icc (-3) 3))) = 0 :=
by
  sorry

end sum_abs_val_lt_4_l364_364640


namespace rope_length_increased_l364_364741

noncomputable def additional_area := 933.4285714285714
noncomputable def original_length := 12.0
noncomputable def pi_approx := Real.pi

def new_length : Real := Real.sqrt ((additional_area / pi_approx) + original_length^2)

theorem rope_length_increased :
  abs (new_length - 21.0) < 0.1 :=
by
  sorry

end rope_length_increased_l364_364741


namespace apples_total_l364_364165

theorem apples_total (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apples_total_l364_364165


namespace polynomial_expansion_proof_l364_364338

variable (z : ℤ)

-- Define the polynomials p and q
noncomputable def p (z : ℤ) : ℤ := 3 * z^2 - 4 * z + 1
noncomputable def q (z : ℤ) : ℤ := 2 * z^3 + 3 * z^2 - 5 * z + 2

-- Define the expanded polynomial
noncomputable def expanded (z : ℤ) : ℤ :=
  6 * z^5 + z^4 - 25 * z^3 + 29 * z^2 - 13 * z + 2

-- The goal is to prove the equivalence of (p * q) == expanded 
theorem polynomial_expansion_proof :
  (p z) * (q z) = expanded z :=
by
  sorry

end polynomial_expansion_proof_l364_364338


namespace geom_seq_prop_l364_364915

-- Definitions from the conditions
def geom_seq (a : ℕ → ℝ) := ∀ (n : ℕ), (a (n + 1)) / (a n) = (a 1) / (a 0) ∧ a n > 0

def condition (a : ℕ → ℝ) :=
  (1 / (a 2 * a 4)) + (2 / (a 4 ^ 2)) + (1 / (a 4 * a 6)) = 81

-- The statement to prove
theorem geom_seq_prop (a : ℕ → ℝ) (hgeom : geom_seq a) (hcond : condition a) :
  (1 / (a 3) + 1 / (a 5)) = 9 :=
sorry

end geom_seq_prop_l364_364915


namespace seven_digit_palindromes_l364_364315

theorem seven_digit_palindromes : 
  let a_choices := 9 in
  let b_choices := 5 in
  let c_choices := 10 in
  let d_choices := 10 in
  a_choices * b_choices * c_choices * d_choices = 4500 :=
by 
  let a_choices := 9
  let b_choices := 5
  let c_choices := 10
  let d_choices := 10
  show a_choices * b_choices * c_choices * d_choices = 4500
  sorry

end seven_digit_palindromes_l364_364315


namespace initial_average_mark_of_class_l364_364596

theorem initial_average_mark_of_class
  (avg_excluded : ℝ) (n_excluded : ℕ) (avg_remaining : ℝ)
  (n_total : ℕ) : 
  avg_excluded = 70 → 
  n_excluded = 5 → 
  avg_remaining = 90 → 
  n_total = 10 → 
  (10 * (10 / n_total + avg_excluded - avg_remaining) / 10) = 80 :=
by 
  intros 
  sorry

end initial_average_mark_of_class_l364_364596


namespace find_area_of_rectangle_l364_364924

-- Define the geometric setup and conditions
variables (ABCD : Type) [rectangle ABCD] (AD BC CE : Type)
variables (E : ABCD) (F : CE)
variables (area_triangle_BDF : ℝ) (area_rect_ABCD : ℝ)

-- Define the conditions
axiom midpoint_AD_E : midpoint AD E
axiom midpoint_CE_F : midpoint CE F
axiom area_triangle_BDF_eq_12 : area_triangle_BDF = 12

-- Define the area calculation goal
def area_of_rectangle (ABCD : Type) [rectangle ABCD] (area_rect_ABCD : ℝ) : Prop :=
  area_rect_ABCD = 96

-- State the theorem
theorem find_area_of_rectangle
  (ABCD : Type) [rectangle ABCD] (AD : Type) (BC : Type) (CE : Type)
  (E : ABCD) [midpoint AD E] (F : CE) [midpoint CE F]
  (area_triangle_BDF : ℝ) (area_triangle_BDF_eq_12 : area_triangle_BDF = 12):
  area_of_rectangle ABCD 96 :=
sorry

end find_area_of_rectangle_l364_364924


namespace minimum_value_a_plus_2b_l364_364822

theorem minimum_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 2) + 1 / (b + 2) = 1 / 3) : a + 2 * b ≥ 3 + 6 * Real.sqrt 2 := 
by 
  skip -- proof is not required, so we add skip to avoid unproven theorem issues

end minimum_value_a_plus_2b_l364_364822


namespace probability_of_composite_l364_364487

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ m < n ∧ 1 < k ∧ k < n ∧ m * k = n

def dice_outcomes (faces : ℕ) (rolls : ℕ) : ℕ :=
  faces ^ rolls

def non_composite_product_ways : ℕ :=
  1 + (3 * 4)  -- one way for all 1s, plus combinations of (1,1,1,{2,3,5})

def total_outcomes : ℕ :=
  dice_outcomes 6 4  -- 6^4 total possible outcomes

def probability_composite : ℚ :=
  1 - (non_composite_product_ways / total_outcomes)

theorem probability_of_composite:
  probability_composite = 1283 / 1296 := 
by
  sorry

end probability_of_composite_l364_364487


namespace possible_apple_counts_l364_364132

theorem possible_apple_counts (n : ℕ) (h₁ : 70 ≤ n) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end possible_apple_counts_l364_364132


namespace problem_solution_l364_364029

theorem problem_solution (m : ℕ) (h : m = 16^2023) : (m / 8) + 2 = 2^8089 + 2 := by
  have m_exp : m = 2^8092 := by
    rw [h, pow_mul]
    norm_num
  rw [m_exp, nat.div_add_self_iff_eq_mul_self]
  norm_num
  ring
  sorry

end problem_solution_l364_364029


namespace train_length_proper_l364_364747

noncomputable def train_length (speed distance_time pass_time : ℝ) : ℝ :=
  speed * pass_time

axiom speed_of_train : ∀ (distance_time : ℝ), 
  (10 * 1000 / (15 * 60)) = 11.11

theorem train_length_proper :
  train_length 11.11 900 10 = 111.1 := by
  sorry

end train_length_proper_l364_364747


namespace equivalent_proof_problem_l364_364367

theorem equivalent_proof_problem (x : ℤ) (h : (x + 2) * (x - 2) = 1221) :
    (x = 35 ∨ x = -35) ∧ ((x + 1) * (x - 1) = 1224) :=
sorry

end equivalent_proof_problem_l364_364367


namespace probability_A_east_lemma_l364_364930

noncomputable def probability_A_east {α β γ : ℕ} (hα : α = 40) (hβγ : β + γ = 180 - α) : ℚ :=
  140 / 360

theorem probability_A_east_lemma {α β γ : ℕ} 
  (hα : α = 40)
  (hβγ : β + γ = 180 - α) :
  probability_A_east hα hβγ = 7 / 18 :=
by
  unfold probability_A_east
  rw [hα]
  norm_num
  sorry

end probability_A_east_lemma_l364_364930


namespace cauchy_schwarz_inequality_l364_364963

theorem cauchy_schwarz_inequality (n : ℕ) (a : Fin n → ℝ) 
  (h : ∀ i, 0 < a i) : 
  (∑ i, a i) * (∑ i, (1 / a i)) ≥ (n : ℝ)^2 := sorry

end cauchy_schwarz_inequality_l364_364963


namespace problem_inequality_l364_364866

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x

theorem problem_inequality (a x : ℝ) (h : a ∈ Set.Iic (-1/Real.exp 2)) :
  f a x ≥ 2 * a * x - x * Real.exp (a * x - 1) := 
sorry

end problem_inequality_l364_364866


namespace is_positive_integer_iff_l364_364885

theorem is_positive_integer_iff (p : ℕ) : 
  (p > 0 → ∃ k : ℕ, (4 * p + 17 = k * (3 * p - 7))) ↔ (3 ≤ p ∧ p ≤ 40) := 
sorry

end is_positive_integer_iff_l364_364885


namespace monotonic_decreasing_interval_l364_364625

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x
noncomputable def f' (x : ℝ) : ℝ := (x - 2) * Real.exp x

theorem monotonic_decreasing_interval : 
  ∀ x : ℝ, x < 2 → f' x < 0 :=
by
  intro x hx
  sorry

end monotonic_decreasing_interval_l364_364625


namespace projects_count_minimize_time_l364_364252

-- Define the conditions as given in the problem
def total_projects := 15
def energy_transfer_condition (x y : ℕ) : Prop := x = 2 * y - 3

-- Define question 1 as a proof problem
theorem projects_count (x y : ℕ) (h1 : x + y = total_projects) (h2 : energy_transfer_condition x y) :
  x = 9 ∧ y = 6 :=
by
  sorry

-- Define conditions for question 2
def average_time (energy_transfer_time leaping_gate_time : ℕ) (m n total_time : ℕ) : Prop :=
  total_time = 6 * m + 8 * n

-- Define additional conditions needed for Question 2 regarding time
theorem minimize_time (m n total_time : ℕ)
  (h1 : m + n = 10)
  (h2 : 10 - m > n)
  (h3 : average_time 6 8 m n total_time)
  (h4 : m = 6) :
  total_time = 68 :=
by
  sorry

end projects_count_minimize_time_l364_364252


namespace sum_S_2013_l364_364834

noncomputable def sequence (n : ℕ) : ℤ :=
  if n = 1 then 1
  else if n > 1 then
    let rec aux : ℕ → ℤ
        | 1 => 1
        | k + 1 => (-1)^k * (aux k + 1)
    in aux n
  else 0

def S (n : ℕ) : ℤ := ∑ i in Finset.range n + 1, sequence i

theorem sum_S_2013 : S 2013 = -1005 := 
  sorry

end sum_S_2013_l364_364834


namespace percentage_increase_in_combined_cost_l364_364542

theorem percentage_increase_in_combined_cost
  (last_year_bicycle : ℝ)
  (last_year_helmet : ℝ)
  (last_year_water_bottle : ℝ)
  (bicycle_increase : ℝ)
  (helmet_increase : ℝ)
  (water_bottle_increase : ℝ)
  (new_year_bicycle : ℝ)
  (new_year_helmet : ℝ)
  (new_year_water_bottle : ℝ)
  (original_total : ℝ)
  (new_total : ℝ)
  (increase : ℝ)
  (percentage_increase : ℝ)
  (h_bicycle : last_year_bicycle = 200)
  (h_helmet : last_year_helmet = 50)
  (h_water_bottle : last_year_water_bottle = 15)
  (h_bicycle_inc : bicycle_increase = 0.06)
  (h_helmet_inc : helmet_increase = 0.12)
  (h_water_bottle_inc : water_bottle_increase = 0.08)
  (h_new_bicycle : new_year_bicycle = last_year_bicycle + (last_year_bicycle * bicycle_increase))
  (h_new_helmet : new_year_helmet = last_year_helmet + (last_year_helmet * helmet_increase))
  (h_new_water_bottle : new_year_water_bottle = last_year_water_bottle + (last_year_water_bottle * water_bottle_increase))
  (h_original_total : original_total = last_year_bicycle + last_year_helmet + last_year_water_bottle)
  (h_new_total : new_total = new_year_bicycle + new_year_helmet + new_year_water_bottle)
  (h_increase : increase = new_total - original_total)
  (h_percentage_increase : percentage_increase = (increase / original_total) * 100) :
  percentage_increase ≈ 7.25 := sorry

end percentage_increase_in_combined_cost_l364_364542


namespace faye_age_l364_364042

variable (C D E F : ℕ)

def condition1 : Prop := D = E - 2
def condition2 : Prop := E = C + 5
def condition3 : Prop := F = C + 4
def condition4 : Prop := D = 15

theorem faye_age (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : F = 16 :=
by
  sorry

end faye_age_l364_364042


namespace average_first_last_conditioned_l364_364102

theorem average_first_last_conditioned {α : Type*} [linear_order α] [add_comm_group α] [has_scalar ℚ α] [module ℚ α]
    (S : fin 5 → α) (largest : α) (smallest : α) (median : α)
    (hL : largest = 11)
    (hS : smallest = -3)
    (hM : median = 5)
    (rule_1 : ∃ (i : fin 5), i ∈ {2, 3, 4} ∧ S i = largest)
    (rule_2 : ∃ (i : fin 5), i ∈ {0, 1, 2} ∧ S i = smallest)
    (rule_3 : ¬ (S 2 = median)) :
    (S 0 + S 4) / 2 = 6.5 :=
by
  sorry

end average_first_last_conditioned_l364_364102


namespace travel_time_change_l364_364620

def drive_time_at_speed (distance_rate_time : ℝ × ℝ × ℝ) (new_speed : ℝ) : ℝ :=
  let (distance, rate, original_time) := distance_rate_time
  original_time * rate / new_speed

theorem travel_time_change (original_time : ℝ) (old_speed : ℝ) (new_speed : ℝ) :
  old_speed = 80 ∧ original_time = 3 ∧ new_speed = 50 →
  drive_time_at_speed (old_speed * original_time, old_speed, original_time) new_speed = 4.8 :=
by
  intros h
  cases h
  unfold drive_time_at_speed
  sorry

end travel_time_change_l364_364620


namespace fresh_grapes_weight_l364_364818

theorem fresh_grapes_weight :
  ∀ (F : ℝ), (∀ (water_content_fresh : ℝ) (water_content_dried : ℝ) (weight_dried : ℝ),
    water_content_fresh = 0.90 → water_content_dried = 0.20 → weight_dried = 3.125 →
    (F * 0.10 = 0.80 * weight_dried) → F = 78.125) := 
by
  intros F
  intros water_content_fresh water_content_dried weight_dried
  intros h1 h2 h3 h4
  sorry

end fresh_grapes_weight_l364_364818


namespace total_apples_l364_364160

theorem total_apples (n : ℕ) (h1 : ∀ k : ℕ, 6 * k = n ∨ 12 * k = n) (h2 : 70 ≤ n) (h3 : n ≤ 80) : 
  n = 72 ∨ n = 78 := 
sorry

end total_apples_l364_364160


namespace julie_reimbursement_l364_364576

-- Conditions
def num_lollipops : ℕ := 12
def total_cost_dollars : ℝ := 3
def shared_fraction : ℝ := 1 / 4

-- Conversion factor
def dollars_to_cents : ℝ := 100

-- Question to prove
theorem julie_reimbursement : 
  let cost_per_lollipop := total_cost_dollars / num_lollipops in
  let num_shared_lollipops := (shared_fraction * num_lollipops : ℝ) in
  let total_cost_shared_lollipops := cost_per_lollipop * num_shared_lollipops in
  let reimbursement_cents := total_cost_shared_lollipops * dollars_to_cents in
  reimbursement_cents = 75 :=
by
  sorry

end julie_reimbursement_l364_364576


namespace find_a₁_l364_364641

noncomputable def S_3 (a₁ q : ℝ) : ℝ := a₁ + a₁ * q + a₁ * q^2

theorem find_a₁ (S₃_eq : S_3 a₁ q = a₁ + 3 * (a₁ * q)) (a₄_eq : a₁ * q^3 = 8) : a₁ = 1 :=
by
  -- proof skipped
  sorry

end find_a₁_l364_364641


namespace necessary_but_not_sufficient_condition_l364_364460

noncomputable def necessary_but_not_sufficient (x : ℝ) : Prop :=
  (3 - x >= 0 → |x - 1| ≤ 2) ∧ ¬(3 - x >= 0 ↔ |x - 1| ≤ 2)

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  necessary_but_not_sufficient x :=
sorry

end necessary_but_not_sufficient_condition_l364_364460


namespace find_angle_OPQ_l364_364028

variables {A B C O P Q : Type}
variables [EuclideanGeometry A B C O P Q]

def midpoint (M X Y : Type) [EuclideanGeometry X Y M] : Prop :=
  dist X M = dist M Y

variables (O_center : circle_center A B C O)
variables (P_midpoint_AO : midpoint P A O)
variables (Q_midpoint_BC : midpoint Q B C)
variables (x : Real.Angle.Degree)

def angle_CBA_eq_4x : angle_deg C B A = 4 * x := by sorry
def angle_ACB_eq_6x : angle_deg A C B = 6 * x := by sorry

theorem find_angle_OPQ :
  ∃ x : Real.Angle.Degree, (angle_deg O P Q = x) ∧ x = 12 := by
  use 12
  sorry

end find_angle_OPQ_l364_364028


namespace trigonometric_inequality_l364_364825

open Real

theorem trigonometric_inequality 
  (x y z : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < y) 
  (h3 : y < z) 
  (h4 : z < π / 2) : 
  π / 2 + 2 * sin x * cos y + 2 * sin y * cos z > sin (2 * x) + sin (2 * y) + sin (2 * z) :=
  sorry

end trigonometric_inequality_l364_364825


namespace decrease_in_radius_l364_364792

theorem decrease_in_radius
  (dist_summer : ℝ)
  (dist_winter : ℝ)
  (radius_summer : ℝ) 
  (mile_to_inch : ℝ)
  (π : ℝ) 
  (δr : ℝ) :
  dist_summer = 560 →
  dist_winter = 570 →
  radius_summer = 20 →
  mile_to_inch = 63360 →
  π = Real.pi →
  δr = 0.33 :=
sorry

end decrease_in_radius_l364_364792


namespace emeralds_count_l364_364732

-- Definitions
def jeweler_problem (D R E : ℕ) : Prop :=
  R = D + 15 ∧
  R = 13 + 8 ∧
  D = 2 + 4 ∧
  E = 5 + 7

-- The statement to prove
theorem emeralds_count : ∃ E : ℕ, jeweler_problem 6 21 E ∧ E = 12 :=
by {
  use 12,
  unfold jeweler_problem,
  split,
  {split, norm_num, split, norm_num, split, norm_num, refl},
  refl
}

end emeralds_count_l364_364732


namespace minute_hand_angle_backward_l364_364302

theorem minute_hand_angle_backward (backward_minutes : ℝ) (h : backward_minutes = 10) :
  (backward_minutes / 60) * (2 * Real.pi) = Real.pi / 3 := by
  sorry

end minute_hand_angle_backward_l364_364302


namespace tuition_fee_payment_l364_364312

theorem tuition_fee_payment (T E S R : ℝ) (hT : T = 90) (hE : E = 15) (hS : S = 0.30) (hR : R = 18) :
  let scholarship_amount := S * T,
      total_after_scholarship := T - scholarship_amount,
      amount_paid_so_far := total_after_scholarship - R,
      M := amount_paid_so_far / E in
  M = 3 :=
by
  sorry

end tuition_fee_payment_l364_364312


namespace range_C_8_x_l364_364548

def greatestIntLessOrEqual (x : ℝ) : ℤ := ⌊x⌋

def C_n^x (n : ℕ) (x : ℝ) : ℝ := 
  (1:ℝ) * (n:ℝ) *
  ∏ i in Finset.range (greatestIntLessOrEqual x).toNat_pred, 
      (n - i : ℝ) /
  (x * ∏ i in Finset.range (greatestIntLessOrEqual x).toNat_pred,
      (x - (i : ℝ)))

theorem range_C_8_x : 
  ∀ (x : ℝ), (x ∈ Ico (3 / 2) 3) →
  (C_n^x 8 x ∈ Set.Ioo 4 (16 / 3) ∪ Set.Ioo (28 / 3) 28) :=
sorry

end range_C_8_x_l364_364548


namespace trapezoid_height_l364_364502

theorem trapezoid_height (a b : ℝ) (A : ℝ) (h : ℝ) : a = 5 → b = 9 → A = 56 → A = (1 / 2) * (a + b) * h → h = 8 :=
by 
  intros ha hb hA eqn
  sorry

end trapezoid_height_l364_364502


namespace fill_tanker_in_10_hours_l364_364271

-- Definitions for the given conditions
def rateA : ℝ := 1 / 30
def rateB : ℝ := 1 / 15

-- Define combined rate
def combined_rate : ℝ := rateA + rateB

-- Define the time to fill the tanker
def time_to_fill_tanker : ℝ := 1 / combined_rate

-- Problem statement: Prove that the time to fill the tanker using both pipes is 10 hours
theorem fill_tanker_in_10_hours (h1 : rateA = 1 / 30) (h2 : rateB = 1 / 15) (h3 : combined_rate = rateA + rateB) 
(h4 : time_to_fill_tanker = 1 / combined_rate) : time_to_fill_tanker = 10 :=
by {
  sorry
}

end fill_tanker_in_10_hours_l364_364271


namespace number_of_true_propositions_l364_364106

theorem number_of_true_propositions (m : ℝ) (hm : m > 0) : 
  let orig := ∀ (m : ℝ), m > 0 → ∃ x : ℝ, x^2 + x - m = 0,
      conv := ∀ (m : ℝ), (∃ x : ℝ, x^2 + x - m = 0) → m > 0,
      inv := ∀ (m : ℝ), m ≤ 0 → ¬ ∃ x : ℝ, x^2 + x - m = 0,
      contra := ∀ (m : ℝ), ¬ (∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0 in
  (if orig then 1 else 0) + (if conv then 1 else 0) + (if inv then 1 else 0) + (if contra then 1 else 0) = 2 :=
begin
  sorry
end

end number_of_true_propositions_l364_364106


namespace inequality_proof_l364_364032

theorem inequality_proof (n : ℕ) (a : ℝ) (h₀ : n > 1) (h₁ : 0 < a) (h₂ : a < 1) : 
  1 + a < (1 + a / n) ^ n ∧ (1 + a / n) ^ n < (1 + a / (n + 1)) ^ (n + 1) := 
sorry

end inequality_proof_l364_364032


namespace sin_double_angle_of_tan_l364_364353

theorem sin_double_angle_of_tan (α : ℝ) (hα1 : Real.tan α = 2) (hα2 : 0 < α ∧ α < Real.pi / 2) : Real.sin (2 * α) = 4 / 5 := by
  sorry

end sin_double_angle_of_tan_l364_364353


namespace exists_circular_chain_of_four_l364_364649

-- Let A and B be the two teams, each with a set of players.
variable {A B : Type}
-- Assume there exists a relation "beats" that determines match outcomes.
variable (beats : A → B → Prop)

-- Each player in both teams has at least one win and one loss against the opposite team.
axiom each_has_win_and_loss (a : A) : ∃ b1 b2 : B, beats a b1 ∧ ¬beats a b2 ∧ b1 ≠ b2
axiom each_has_win_and_loss' (b : B) : ∃ a1 a2 : A, beats a1 b ∧ ¬beats a2 b ∧ a1 ≠ a2

-- Main theorem: Exist four players forming a circular chain of victories.
theorem exists_circular_chain_of_four :
  ∃ (a1 a2 : A) (b1 b2 : B), beats a1 b1 ∧ ¬beats a1 b2 ∧ beats a2 b2 ∧ ¬beats a2 b1 ∧ b1 ≠ b2 ∧ a1 ≠ a2 :=
sorry

end exists_circular_chain_of_four_l364_364649


namespace probability_of_composite_l364_364489

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ m < n ∧ 1 < k ∧ k < n ∧ m * k = n

def dice_outcomes (faces : ℕ) (rolls : ℕ) : ℕ :=
  faces ^ rolls

def non_composite_product_ways : ℕ :=
  1 + (3 * 4)  -- one way for all 1s, plus combinations of (1,1,1,{2,3,5})

def total_outcomes : ℕ :=
  dice_outcomes 6 4  -- 6^4 total possible outcomes

def probability_composite : ℚ :=
  1 - (non_composite_product_ways / total_outcomes)

theorem probability_of_composite:
  probability_composite = 1283 / 1296 := 
by
  sorry

end probability_of_composite_l364_364489


namespace range_of_a_l364_364096

theorem range_of_a (a : ℝ) : (∀ x ∈ set.Ici (2:ℝ), deriv (λ x, x^2 + 2*a*x + 1) x ≥ 0) ↔ a ∈ set.Ici (-2) :=
by
  sorry

end range_of_a_l364_364096


namespace exists_twelve_distinct_x_l364_364453

theorem exists_twelve_distinct_x :
  ∃ S : Set ℝ, (S.card = 12) ∧ (∀ x ∈ S, ∃ k : ℤ, 0 ≤ k ∧ k ≤ 11 ∧ (sqrt (123 - sqrt x) = k)) :=
by
  sorry

end exists_twelve_distinct_x_l364_364453


namespace train_speed_kph_l364_364294

-- Definitions based on conditions
def time_seconds : ℕ := 9
def length_meters : ℕ := 135
def conversion_factor : ℕ := 36 -- 3.6 represented as an integer by multiplying both sides by 10

-- The proof statement
theorem train_speed_kph : (length_meters * conversion_factor / 10 / time_seconds = 54) :=
by
  sorry

end train_speed_kph_l364_364294


namespace fraction_order_correct_l364_364705

theorem fraction_order_correct : 
  let f1 := (-16 : ℚ) / 12
  let f2 := (-18 : ℚ) / 14
  let f3 := (-20 : ℚ) / 15
  (f1 = f3 ∧ f3 < f2) := 
by
  dsimp [f1, f2, f3]
  sorry

end fraction_order_correct_l364_364705


namespace sum_diagonal_arithmetic_geometric_sequence_l364_364718

theorem sum_diagonal_arithmetic_geometric_sequence (n : ℕ) (q : ℝ) 
  (a : ℕ × ℕ → ℝ)
  (h1 : n ≥ 4)
  (h2 : ∀ i j, a (i, j) =  a (1, j) + (i - 1) * (a (1, j) - a (1, 1)))
  (h3 : ∀ i j, a (i, j) = a (1, j) * q ^ (i - 1))
  (h4 : a (2, 4) = 1)
  (h5 : a (4, 2) = 1 / 8)
  (h6 : a (4, 3) = 3 / 16) :
  (∑ k in Finset.range n, a (k + 1, k + 1)) = 
    (2 : ℝ) - (1 / 2 ^ (n - 1)) - (n / 2 ^ n) :=
begin
  sorry
end

end sum_diagonal_arithmetic_geometric_sequence_l364_364718


namespace jake_later_than_austin_by_20_seconds_l364_364764

theorem jake_later_than_austin_by_20_seconds :
  (9 * 30) / 3 - 60 = 20 :=
by
  sorry

end jake_later_than_austin_by_20_seconds_l364_364764


namespace greatest_three_digit_multiple_of_23_is_991_l364_364658

theorem greatest_three_digit_multiple_of_23_is_991 :
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 23 = 0) ∧ ∀ m : ℤ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 23 = 0) → m ≤ n :=
begin
  use 991,
  -- proof steps go here
  sorry
end

end greatest_three_digit_multiple_of_23_is_991_l364_364658


namespace greatest_three_digit_multiple_of_23_is_991_l364_364659

theorem greatest_three_digit_multiple_of_23_is_991 :
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 23 = 0) ∧ ∀ m : ℤ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 23 = 0) → m ≤ n :=
begin
  use 991,
  -- proof steps go here
  sorry
end

end greatest_three_digit_multiple_of_23_is_991_l364_364659


namespace wanda_walks_distance_l364_364205

theorem wanda_walks_distance :
  let daily_distance := 0.5 * 2 * 2, -- miles walked in a day due to school trips
      weekly_distance := daily_distance * 5, -- miles walked in a week (5 days)
      monthly_distance := weekly_distance * 4 in -- miles walked in 4 weeks
  monthly_distance = 40 :=
by
  let daily_distance := 0.5 * 2 * 2
  let weekly_distance := daily_distance * 5
  let monthly_distance := weekly_distance * 4
  have : monthly_distance = 40 := by sorry
  exact this

end wanda_walks_distance_l364_364205


namespace solution_set_f_lt_g_l364_364863

noncomputable def f : ℝ → ℝ := sorry -- Assume f exists according to the given conditions

lemma f_at_one : f 1 = -2 := sorry

lemma f_derivative_neg (x : ℝ) : (deriv f x) < 0 := sorry

def g (x : ℝ) : ℝ := x - 3

lemma g_at_one : g 1 = -2 := sorry

theorem solution_set_f_lt_g :
  {x : ℝ | f x < g x} = {x : ℝ | 1 < x} :=
sorry

end solution_set_f_lt_g_l364_364863


namespace max_ab_value_l364_364882

theorem max_ab_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h_perpendicular : (2 * a - 1) * b = -1) : ab <= 1 / 8 := by
  sorry

end max_ab_value_l364_364882


namespace ken_gave_manny_10_pencils_l364_364012

theorem ken_gave_manny_10_pencils (M : ℕ) 
  (ken_pencils : ℕ := 50)
  (ken_kept : ℕ := 20)
  (ken_distributed : ℕ := ken_pencils - ken_kept)
  (nilo_pencils : ℕ := M + 10)
  (distribution_eq : M + nilo_pencils = ken_distributed) : 
  M = 10 :=
by
  sorry

end ken_gave_manny_10_pencils_l364_364012


namespace pattern_forth_operation_l364_364511

theorem pattern_forth_operation (x y : ℕ) (h1 : 2 + 3 = 8)
  (h2 : 3 + 7 = 27)
  (h3 : 4 + 5 = 32)
  (h5 : 6 + 7 = 72)
  (h6 : 7 + 8 = 98)
  (hx : x = 5)
  (hy : y = 8)
  : (x * y + x * (y - x) = 55) :=
by {
  -- We provide this as an assumption derived from the pattern
  have rule : ∀ a b, a + b = a * b + a * (b - a) := sorry,
  -- Apply the rule to x = 5 and y = 8
  specialize rule 5 8,
  rw [← hx, ← hy] at rule,
  exact rule,
}

end pattern_forth_operation_l364_364511


namespace number_of_ways_to_form_right_triangle_l364_364049

open Finset

-- We need a function that calculates the number of ways to form a right-angle triangle.
def count_right_triangles (n : Nat) : Nat :=
  (choose n 2) * n * 2

-- There are 58 points on each line.
def num_points_on_each_line : Nat := 58

-- The total number of ways to form right triangles with the given conditions.
def total_ways_to_form_right_triangles : Nat :=
  count_right_triangles num_points_on_each_line

-- Theorem statement
theorem number_of_ways_to_form_right_triangle (h : num_points_on_each_line = 58) :
    total_ways_to_form_right_triangles = 6724 := by
  sorry

end number_of_ways_to_form_right_triangle_l364_364049


namespace abs_pi_sub_abs_pi_sub_ten_eq_ten_sub_2pi_l364_364774

theorem abs_pi_sub_abs_pi_sub_ten_eq_ten_sub_2pi (h : Real.pi < 10) :
  |Real.pi - |Real.pi - 10|| = 10 - 2 * Real.pi := 
sorry

end abs_pi_sub_abs_pi_sub_ten_eq_ten_sub_2pi_l364_364774


namespace Christine_savings_l364_364320

theorem Christine_savings 
  (commission_rate: ℝ) 
  (total_sales: ℝ) 
  (personal_needs_percentage: ℝ) 
  (savings: ℝ) 
  (h1: commission_rate = 0.12) 
  (h2: total_sales = 24000) 
  (h3: personal_needs_percentage = 0.60) 
  (h4: savings = total_sales * commission_rate * (1 - personal_needs_percentage)) : 
  savings = 1152 := by 
  sorry

end Christine_savings_l364_364320


namespace count_distinct_x_l364_364437

theorem count_distinct_x :
  { x : ℝ | ∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ s = floor (sqrt (123 - sqrt x)) } = 12 :=
sorry

end count_distinct_x_l364_364437


namespace euler_line_of_triangle_is_perpendicular_bisector_l364_364563

open Real

-- Define the vertices B and C
def B : (ℝ × ℝ) := (-1, 0)
def C : (ℝ × ℝ) := (0, 2)

-- Triangle ABC with AB = AC
def is_isosceles (A B C : (ℝ × ℝ)) : Prop := dist A B = dist A C

-- Midpoint of segment BC
def midpoint (P Q : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Slope function
def slope (A B : (ℝ × ℝ)) : ℝ :=
  (B.2 - A.2) / (B.1 - A.1)

-- Perpendicular bisector line equation through a point with a given slope
def perp_bisector_eq (P : (ℝ × ℝ)) (m : ℝ) : (ℝ × ℝ) → Prop :=
  λ Q, Q.2 - P.2 = -1 / m * (Q.1 - P.1)

-- The Euler line equation
def euler_line_eq (A B C : (ℝ × ℝ)) : (ℝ × ℝ) → Prop :=
  λ P, 2 * P.1 + 4 * P.2 - 3 = 0

theorem euler_line_of_triangle_is_perpendicular_bisector :
  ∃ A : (ℝ × ℝ), is_isosceles A B C ∧ euler_line_eq A B C = perp_bisector_eq (midpoint B C) (slope B C) :=
sorry

end euler_line_of_triangle_is_perpendicular_bisector_l364_364563


namespace problem_solution_l364_364887

theorem problem_solution 
  (x y : ℝ) 
  (h : x^2 - 2*x*y + 2*y^2 = 1) : 
  |x| ≤ sqrt 2 ∧ x^2 + 2*y^2 > 1 / 2 := 
by
  sorry

end problem_solution_l364_364887


namespace factorial_divisibility_l364_364898

theorem factorial_divisibility
  (n p : ℕ)
  (h1 : p > 0)
  (h2 : n ≤ p + 1) :
  (factorial (p^2)) ∣ (factorial p)^(p + 1) :=
sorry

end factorial_divisibility_l364_364898


namespace teacher_sampling_l364_364402

theorem teacher_sampling (total_teachers senior_teachers intermediate_teachers junior_teachers sampled_teachers : ℕ) 
                         (h1 : total_teachers = 150) 
                         (h2 : senior_teachers = 15) 
                         (h3 : intermediate_teachers = 90) 
                         (h4 : junior_teachers = total_teachers - senior_teachers - intermediate_teachers) 
                         (h5 : sampled_teachers = 30) : 
                         (senior_sampled intermediate_sampled junior_sampled : ℕ) 
                         (h6 : senior_sampled = (senior_teachers * sampled_teachers) / total_teachers)
                         (h7 : intermediate_sampled = (intermediate_teachers * sampled_teachers) / total_teachers)
                         (h8 : junior_sampled = sampled_teachers - senior_sampled - intermediate_sampled) :
                         (senior_sampled = 3) ∧ (intermediate_sampled = 18) ∧ (junior_sampled = 9) :=
by
  sorry

end teacher_sampling_l364_364402


namespace gcd_values_count_l364_364689

theorem gcd_values_count :
  ∃ a b : ℕ, (gcd a b) * (nat.lcm a b) = 180 ∧
    set.card { gcd a b | ∃ a b, a * b = 180 } = 8 :=
by
  -- Problem statement as provided by conditions and question
  -- Definitions and notations are provided correctly and fully, proof is omitted
  sorry

end gcd_values_count_l364_364689


namespace vertex_in_first_quadrant_l364_364632

theorem vertex_in_first_quadrant (a : ℝ) (h : a > 1) : 
  let x_vertex := (a + 1) / 2
  let y_vertex := (a + 3)^2 / 4
  x_vertex > 0 ∧ y_vertex > 0 := 
by
  sorry

end vertex_in_first_quadrant_l364_364632


namespace value_of_f_neg2011_l364_364867

def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x - 2

theorem value_of_f_neg2011 (a b : ℝ) (h : f 2011 a b = 10) : f (-2011) a b = -14 := by
  sorry

end value_of_f_neg2011_l364_364867


namespace find_base_side_length_l364_364622

-- Regular triangular pyramid properties and derived values
variables
  (a l h : ℝ) -- side length of the base, slant height, and height of the pyramid
  (V : ℝ) -- volume of the pyramid

-- Given conditions
def inclined_to_base_plane_at_angle (angle : ℝ) := angle = 45
def volume_of_pyramid (V : ℝ) := V = 18

-- Prove the side length of the base
theorem find_base_side_length
  (h_eq : h = a * Real.sqrt 3 / 3)
  (volume_eq : V = 1 / 3 * (a * a * Real.sqrt 3 / 4) * h)
  (volume_given : V = 18) :
  a = 6 := by
  sorry

end find_base_side_length_l364_364622


namespace laundry_time_l364_364121

theorem laundry_time (n : ℕ) (wash_time dry_time total_loads : ℕ) (h1 : n = 8) 
  (h2 : wash_time = 45) (h3 : dry_time = 60) (h4 : total_loads = 8) : 
  (n * (wash_time + dry_time)) / 60 = 14 :=
by {
  rw [h1, h2, h3, h4],
  sorry
}

end laundry_time_l364_364121


namespace probability_of_A_east_of_B_and_C_l364_364934

-- Define the facts about the triangle and the problem conditions
def triangle_ABC : Type := 
  {A B C : Point} 
  (angle_A_40 : angle A B C = 40)

-- Define the probability calculation
def probability_A_east_given_angle_40 
  (t : triangle_ABC) : ℚ :=
  7 / 18

-- The theorem statement
theorem probability_of_A_east_of_B_and_C 
  (t : triangle_ABC) : 
  probability_A_east_given_angle_40 t = 7 / 18 := 
  sorry

end probability_of_A_east_of_B_and_C_l364_364934


namespace valid_number_of_apples_l364_364135

theorem valid_number_of_apples (n : ℕ) : 
  (∃ k m : ℕ, n = 12 * k + 6 * m) ∧ (70 ≤ n ∧ n ≤ 80) ↔ (n = 72 ∨ n = 78) :=
by
  sorry

end valid_number_of_apples_l364_364135


namespace sequence_divisibility_l364_364951

noncomputable def a_seq (a : ℕ) : ℕ → ℕ
| 0     := a
| (n+1) := a^(a_seq n)

theorem sequence_divisibility (a : ℕ) (h : a ≥ 2) (n : ℕ) : 
  (a_seq a (n+1) - a_seq a n) ∣ (a_seq a (n+2) - a_seq a (n+1)) :=
sorry

end sequence_divisibility_l364_364951


namespace complete_square_transform_l364_364107

theorem complete_square_transform :
  ∀ x : ℝ, x^2 - 4 * x - 6 = 0 → (x - 2)^2 = 10 :=
by
  intros x h
  sorry

end complete_square_transform_l364_364107


namespace distinct_gcd_count_l364_364697

theorem distinct_gcd_count :
  ∃ (a b : ℕ), (gcd a b * Nat.lcm a b = 180) ∧
  (∀ (d : ℕ), d = gcd a b → 
    d ∈ {1, 2, 3, 5, 6, 10, 15, 30} ∧ 
    (∀ d' ∈ {1, 2, 3, 5, 6, 10, 15, 30}, d' ≠ d → 
      ∃ a' b', gcd a' b' * Nat.lcm a' b' = 180 ∧ gcd a' b' = d')) := sorry

end distinct_gcd_count_l364_364697


namespace valid_number_of_apples_l364_364138

theorem valid_number_of_apples (n : ℕ) : 
  (∃ k m : ℕ, n = 12 * k + 6 * m) ∧ (70 ≤ n ∧ n ≤ 80) ↔ (n = 72 ∨ n = 78) :=
by
  sorry

end valid_number_of_apples_l364_364138


namespace min_club_members_l364_364629

theorem min_club_members : ∃ N : ℕ, N < 80 ∧ (N - 5) % 56 = 0 ∧ (N - 7) % 9 = 0 ∧ N = 61 :=
by
  use 61
  split
  · exact Nat.lt_of_le_of_lt (Nat.zero_lt_succ Nat.zero_lt_one) (by norm_num)
  split
  · norm_num
  split
  · norm_num
  sorry

end min_club_members_l364_364629


namespace ethan_not_next_to_dianne_l364_364118

def beverly := "Beverly"
def dianne := "Dianne"
def ethan := "Ethan"
def jamaal := "Jamaal"

def total_arrangements (people : List String) : Nat :=
  people.permutations.length

def arrangements_adjacent (people : List String) (a b : String) : Nat :=
  (people.permutations.filter (λ perm, 
    (perm.indexOf? a).withNone 
      (λ idx_a, 
        (perm.indexOf? b).withNone 
          (λ idx_b, 
            (idx_a + 1 == idx_b) || (idx_b + 1 == idx_a))))
  ).length

theorem ethan_not_next_to_dianne :
  ∀ (people : List String), 
  people = [beverly, dianne, ethan, jamaal] →
  let total_ways := total_arrangements people in
  let adjacent_ways := arrangements_adjacent people ethan dianne in
  total_ways - adjacent_ways = 12 := 
by 
  intro people h_eq
  let total_ways := total_arrangements people
  let adjacent_ways := arrangements_adjacent people ethan dianne
  have total_eq : total_ways = 24 := sorry
  have adj_eq : adjacent_ways = 12 := sorry
  rw [total_eq, adj_eq]
  exact rfl

end ethan_not_next_to_dianne_l364_364118


namespace Shyne_total_plants_l364_364069

theorem Shyne_total_plants :
  ∀ (eggplants_per_packet sunflowers_per_packet eggplant_packets sunflower_packets : ℕ),
  eggplants_per_packet = 14 →
  sunflowers_per_packet = 10 →
  eggplant_packets = 4 →
  sunflower_packets = 6 →
  (eggplants_per_packet * eggplant_packets + sunflowers_per_packet * sunflower_packets = 116) :=
by
  intros eggplants_per_packet sunflowers_per_packet eggplant_packets sunflower_packets
  intro h1 h2 h3 h4
  rw [h1, h3, h2, h4]
  norm_num
  sorry

end Shyne_total_plants_l364_364069


namespace max_writers_and_editors_l364_364256

theorem max_writers_and_editors (total_people writers editors x : ℕ) (h_total_people : total_people = 100)
(h_writers : writers = 40) (h_editors : editors > 38) (h_both : 2 * x + (writers + editors - x) = total_people) :
x ≤ 21 := sorry

end max_writers_and_editors_l364_364256


namespace number_of_classes_l364_364742

theorem number_of_classes (max_val : ℕ) (min_val : ℕ) (class_interval : ℕ) (range : ℕ) (num_classes : ℕ) :
  max_val = 169 → min_val = 143 → class_interval = 3 → range = max_val - min_val → num_classes = (range + 2) / class_interval + 1 :=
sorry

end number_of_classes_l364_364742


namespace minimum_red_chips_l364_364722

variable (w b r : ℕ)

axiom C1 : b ≥ (1 / 3 : ℚ) * w
axiom C2 : b ≤ (1 / 4 : ℚ) * r
axiom C3 : w + b ≥ 75

theorem minimum_red_chips : r = 76 := by sorry

end minimum_red_chips_l364_364722


namespace ten_pow_m_minus_n_l364_364308

theorem ten_pow_m_minus_n (m n : ℝ) (h₁ : 10^m = 12) (h₂ : 10^n = 3) : 10^(m - n) = 4 :=
sorry

end ten_pow_m_minus_n_l364_364308


namespace parallelogram_height_l364_364088

/-- The cost of leveling a field in the form of a parallelogram is Rs. 50 per 10 sq. meter, 
    with the base being 54 m and a certain perpendicular distance from the other side. 
    The total cost is Rs. 6480. What is the perpendicular distance from the other side 
    of the parallelogram? -/
theorem parallelogram_height
  (cost_per_10_sq_meter : ℝ)
  (base_length : ℝ)
  (total_cost : ℝ)
  (height : ℝ)
  (h1 : cost_per_10_sq_meter = 50)
  (h2 : base_length = 54)
  (h3 : total_cost = 6480)
  (area : ℝ)
  (h4 : area = (total_cost / cost_per_10_sq_meter) * 10)
  (h5 : area = base_length * height) :
  height = 24 :=
by { sorry }

end parallelogram_height_l364_364088


namespace log_base_5_3_proof_l364_364455

noncomputable def log_base_5_45_eq (a : ℝ) : Prop :=
  log 5 45 = a

noncomputable def log_base_5_3_eq (a : ℝ) : ℝ :=
  (a - 1) / 2

theorem log_base_5_3_proof (a : ℝ) (h : log_base_5_45_eq a) : 
  log 5 3 = log_base_5_3_eq a :=
by sorry

end log_base_5_3_proof_l364_364455


namespace volume_tetrahedron_l364_364523

variables (A B C D : Type*)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] 

-- Distance function needs to be defined in the context of a metric space.
def distAB : ℝ := 3
def distCD : ℝ := 2
def distance_between_lines : ℝ := 2
def angle_between_lines : ℝ := real.pi / 6

theorem volume_tetrahedron (V : ℝ) : 
  (∃ (V : ℝ), 
    V = (1 / 3) * distAB * distCD * sin(angle_between_lines) * distance_between_lines) → 
  V = 1 / 2 :=
by
  sorry

end volume_tetrahedron_l364_364523


namespace number_of_gcd_values_l364_364685

theorem number_of_gcd_values (a b : ℤ) (h : a * b = 180) : 
  {d : ℤ | d = Int.gcd a b}.finite.toFinset.card = 8 := 
sorry

end number_of_gcd_values_l364_364685


namespace simplify_and_evaluate_expression_l364_364070

variables (a : ℝ)

theorem simplify_and_evaluate_expression (h : a = -real.sqrt 2) :
  (a - 3) / a * 6 / (a^2 - 6*a + 9) - (2*a + 6) / (a^2 - 9) = real.sqrt 2 :=
by 
  sorry

end simplify_and_evaluate_expression_l364_364070


namespace min_value_f_l364_364386

-- Define the function f
def f (ω x : ℝ) := sin (ω * x) - 2 * sqrt 3 * (sin (ω * x / 2))^2 + sqrt 3

-- Problem statement with conditions and required proof
theorem min_value_f : 
  ∀ (ω : ℝ), ω > 0 → (∀ x1 x2 : ℝ, x1 ≠ x2 ∧ f ω x1 = 0 ∧ f ω x2 = 0 ∧ abs (x2 - x1) = π / 2) →
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 → f ω x ≥ -sqrt 3) :=
begin
  sorry
end

end min_value_f_l364_364386


namespace arc_lengths_proof_l364_364090

variables (a b c : ℝ)
variables (AP BQ PQ : ℝ)
variables (right_angle_ABC : ∠BAC = π / 2)

-- Given conditions
def BC := a
def CA := b
def AB := c

-- Pythagorean theorem condition
axiom pythagorean_theorem : a^2 + b^2 = c^2

-- Derived lengths from the arcs
def AP := c - a
def BQ := c - b

theorem arc_lengths_proof
  (PQ_squared : PQ^2 = (a + b - c)^2) :
  (1 / 2) * PQ^2 = AP * BQ :=
sorry

end arc_lengths_proof_l364_364090


namespace minimize_x_plus_y_on_circle_l364_364366

theorem minimize_x_plus_y_on_circle (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 1) : x + y ≥ 2 :=
by
  sorry

end minimize_x_plus_y_on_circle_l364_364366


namespace overall_profit_l364_364539

theorem overall_profit
    (p_g p_m : ℕ)
    (loss_percent_g profit_percent_m : ℚ)
    (selling_price_g selling_price_m : ℕ)
    (overall_profit : ℤ) :
    p_g = 15000 →
    p_m = 8000 →
    loss_percent_g = 4/100 →
    profit_percent_m = 10/100 →
    selling_price_g = p_g - (loss_percent_g * p_g).toNat →
    selling_price_m = p_m + (profit_percent_m * p_m).toNat →
    overall_profit = (selling_price_g + selling_price_m) - (p_g + p_m) →
    overall_profit = 200 :=
by
  intros
  sorry

end overall_profit_l364_364539


namespace middle_car_person_l364_364968

universe u

variables {α : Type u} [DecidableEq α]

/-- Hypothetical individuals --/
inductive Person
| Maren : Person
| Aaron : Person
| Sharon : Person
| Darren : Person
| Karen : Person

open Person

/-- Hypothetical seat positions --/
def seats := list Person

/-- Given conditions --/
def valid_seating (seating : seats) : Prop :=
(seating.length = 5) ∧
(seating.ilast = Maren) ∧
(list.index_of Aaron seating = list.index_of Sharon seating + 1) ∧
(list.index_of Darren seating = list.index_of Karen seating + 1) ∧
(abs (list.index_of Aaron seating - list.lastIndexOf Maren seating) ≥ 2)

/-- The person sitting in the middle car --/
def whoIsMiddleCar (seating : seats) : Person :=
seating.nth_le 2 (by simp [seating.length_eq_five])

theorem middle_car_person (seating : seats)
  (h : valid_seating seating) :
  whoIsMiddleCar seating = Sharon :=
sorry

end middle_car_person_l364_364968


namespace instantaneous_acceleration_at_3_l364_364002

def v (t : ℝ) : ℝ := t^2 + 3

theorem instantaneous_acceleration_at_3 :
  deriv v 3 = 6 :=
by
  sorry

end instantaneous_acceleration_at_3_l364_364002


namespace probability_sum_multiple_of_3_eq_one_third_probability_sum_prime_eq_five_twelfths_probability_second_greater_than_first_eq_five_twelfths_l364_364987

noncomputable def probability_sum_is_multiple_of_3 : ℝ :=
  let total_events := 36
  let favorable_events := 12
  favorable_events / total_events

noncomputable def probability_sum_is_prime : ℝ :=
  let total_events := 36
  let favorable_events := 15
  favorable_events / total_events

noncomputable def probability_second_greater_than_first : ℝ :=
  let total_events := 36
  let favorable_events := 15
  favorable_events / total_events

theorem probability_sum_multiple_of_3_eq_one_third :
  probability_sum_is_multiple_of_3 = 1 / 3 :=
by sorry

theorem probability_sum_prime_eq_five_twelfths :
  probability_sum_is_prime = 5 / 12 :=
by sorry

theorem probability_second_greater_than_first_eq_five_twelfths :
  probability_second_greater_than_first = 5 / 12 :=
by sorry

end probability_sum_multiple_of_3_eq_one_third_probability_sum_prime_eq_five_twelfths_probability_second_greater_than_first_eq_five_twelfths_l364_364987


namespace projects_count_minimize_time_l364_364253

-- Define the conditions as given in the problem
def total_projects := 15
def energy_transfer_condition (x y : ℕ) : Prop := x = 2 * y - 3

-- Define question 1 as a proof problem
theorem projects_count (x y : ℕ) (h1 : x + y = total_projects) (h2 : energy_transfer_condition x y) :
  x = 9 ∧ y = 6 :=
by
  sorry

-- Define conditions for question 2
def average_time (energy_transfer_time leaping_gate_time : ℕ) (m n total_time : ℕ) : Prop :=
  total_time = 6 * m + 8 * n

-- Define additional conditions needed for Question 2 regarding time
theorem minimize_time (m n total_time : ℕ)
  (h1 : m + n = 10)
  (h2 : 10 - m > n)
  (h3 : average_time 6 8 m n total_time)
  (h4 : m = 6) :
  total_time = 68 :=
by
  sorry

end projects_count_minimize_time_l364_364253


namespace apples_total_l364_364167

theorem apples_total (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apples_total_l364_364167


namespace factorial_divisibility_l364_364894

theorem factorial_divisibility {p : ℕ} (hp : 1 < p) : (p^2)! % (p!)^(p+1) = 0 :=
by
  sorry

end factorial_divisibility_l364_364894


namespace probability_composite_l364_364496

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def non_composite_outcomes : ℕ := 1 + 12

theorem probability_composite :
  let total_outcomes := 6^4 in
  let non_composite := non_composite_outcomes in
  let composite_probability := (total_outcomes - non_composite) / total_outcomes.toRat in
  composite_probability = 1283 / 1296 := by
  let total_outcomes := 6^4
  let non_composite := 1 + 12
  let composite_probability := (total_outcomes - non_composite).toRat / total_outcomes.toRat
  sorry

end probability_composite_l364_364496


namespace coefficient_x_pow_7_l364_364208

theorem coefficient_x_pow_7 :
  ∃ c : Int, (∀ k : Nat, (x : Real) → coeff (expand_binom (x-2) 10) k = c → k = 7 ∧ c = -960) :=
by
  sorry

end coefficient_x_pow_7_l364_364208


namespace range_of_CM_dot_CN_l364_364910

noncomputable def points_on_line (M N : ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ M = (a, 2 - a) ∧ N = (b, 2 - b)

def distance_between_points (M N : ℝ × ℝ) : ℝ :=
  real.sqrt ((M.1 - N.1) ^ 2 + (M.2 - N.2) ^ 2)

def CM_dot_CN (M N : ℝ × ℝ) : ℝ := 
  (M.1, M.2) • (N.1, N.2)

theorem range_of_CM_dot_CN :
  ∀ (M N : ℝ × ℝ),
  points_on_line M N →
  distance_between_points M N = real.sqrt 2 →
  3 / 2 ≤ CM_dot_CN M N ∧ CM_dot_CN M N ≤ 2 :=
by
  sorry

end range_of_CM_dot_CN_l364_364910


namespace wizard_viable_combinations_l364_364748

def wizard_combination_problem : Prop :=
  let total_combinations := 4 * 6
  let incompatible_combinations := 3
  let viable_combinations := total_combinations - incompatible_combinations
  viable_combinations = 21

theorem wizard_viable_combinations : wizard_combination_problem :=
by
  sorry

end wizard_viable_combinations_l364_364748


namespace equivalence_of_functions_l364_364230

noncomputable def f1 (x : ℝ) : ℝ := (sqrt x)^2 / x
noncomputable def f2 (x : ℝ) : ℝ := x / (sqrt x)^2

theorem equivalence_of_functions : ∀ (x : ℝ), x > 0 → f1 x = f2 x :=
by
  sorry

end equivalence_of_functions_l364_364230


namespace alberto_bikes_more_l364_364926

-- Definitions of given speeds
def alberto_speed : ℝ := 15
def bjorn_speed : ℝ := 11.25

-- The time duration considered
def time_hours : ℝ := 5

-- Calculate the distances each traveled
def alberto_distance : ℝ := alberto_speed * time_hours
def bjorn_distance : ℝ := bjorn_speed * time_hours

-- Calculate the difference in distances
def distance_difference : ℝ := alberto_distance - bjorn_distance

-- The theorem to be proved
theorem alberto_bikes_more : distance_difference = 18.75 := by
    sorry

end alberto_bikes_more_l364_364926


namespace find_amplitude_l364_364765

theorem find_amplitude
  (a b : ℝ)
  (c1 : ∀ x, x ≠ 0 → a * (1 / real.sin(b * x)) ≥ 3)  -- Condition for minimum positive value
  (c2 : ∀ x, a * (1 / real.sin(x)) = a * (1 / real.sin(x + 2 * π / b))) -- Condition for period
  (h : ∀ y, y ≠ 0 → a * (1 / real.sin(b * y)) = 3): -- Given minimum positive value
  a = 3 := 
sorry

end find_amplitude_l364_364765


namespace number_of_days_worked_l364_364729

theorem number_of_days_worked (total_toys_per_week : ℕ) (toys_per_day : ℕ) (h₁ : total_toys_per_week = 6000) (h₂ : toys_per_day = 1500) : (total_toys_per_week / toys_per_day) = 4 :=
by
  sorry

end number_of_days_worked_l364_364729


namespace pq_sum_identity_l364_364395

def q (x : ℝ) := (8 / 3) * (x - 1) * (x - 3)

-- Given conditions
def q_condition (x : ℝ) := q 4 = 8
def p (x : ℝ) := 3 * x
def p_condition (x : ℝ) := p 5 = 15

-- Prove p(x) + q(x) == (8 / 3) * x^2 - (29 / 3) * x + 8
theorem pq_sum_identity (x : ℝ) (h_q : q_condition x) (h_p : p_condition x) :
  p(x) + q(x) = (8 / 3) * x^2 - (29 / 3) * x + 8 :=
sorry

end pq_sum_identity_l364_364395


namespace distinct_gcd_count_l364_364698

theorem distinct_gcd_count :
  ∃ (a b : ℕ), (gcd a b * Nat.lcm a b = 180) ∧
  (∀ (d : ℕ), d = gcd a b → 
    d ∈ {1, 2, 3, 5, 6, 10, 15, 30} ∧ 
    (∀ d' ∈ {1, 2, 3, 5, 6, 10, 15, 30}, d' ≠ d → 
      ∃ a' b', gcd a' b' * Nat.lcm a' b' = 180 ∧ gcd a' b' = d')) := sorry

end distinct_gcd_count_l364_364698


namespace sector_area_correct_l364_364591

-- Define the conditions
def radius : ℝ := 2
def central_angle_degrees : ℝ := 120

-- Convert the central angle from degrees to radians
def central_angle_radians : ℝ := central_angle_degrees * (Real.pi / 180)

-- Define the formula for the area of a sector
def sector_area (r theta : ℝ) : ℝ := (1/2) * theta * r^2

-- Define the theorem and its statement
theorem sector_area_correct :
  sector_area radius central_angle_radians = (4 * Real.pi) / 3 := 
sorry

end sector_area_correct_l364_364591


namespace opposite_of_neg3_l364_364221

theorem opposite_of_neg3 : ∃ x : ℝ, -3 + x = 0 ∧ x = 3 := 
by
  use 3
  split
  . simp
  . refl

end opposite_of_neg3_l364_364221


namespace min_value_expression_l364_364831

theorem min_value_expression (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : 4 * a + b = 1) :
  (1 / a) + (4 / b) = 16 := sorry

end min_value_expression_l364_364831


namespace distinct_real_x_l364_364414

theorem distinct_real_x (x : ℝ) :
  (∃! x, ∃ (s : ℝ), s ∈ set.Icc 0 11 ∧ s = real.sqrt (123 - real.sqrt x)) =
  12 := sorry

end distinct_real_x_l364_364414


namespace coeff_x7_expansion_l364_364210

theorem coeff_x7_expansion : 
  let expansion := (∑ k in Finset.range 11, (Nat.choose 10 k) * x^(10 - k) * (-2)^k)
  ∃ coeff : ℤ, 
  (coeff * x^7 ∈ expansion) ∧ coeff = -960 :=
begin
  let expansion := (∑ k in Finset.range 11, (Nat.choose 10 k) * x^(10 - k) * (-2)^k),
  use -960,
  split,
  { sorry, },
  { reflexivity, }
end

end coeff_x7_expansion_l364_364210


namespace alex_carla_weight_l364_364304

-- We define real numbers representing the weights of Alex, Ben, Carla, and Derek.
variables (a b c d : ℝ)

-- We introduce the given conditions as hypotheses.
hypothesis h1 : a + b = 280
hypothesis h2 : b + c = 235
hypothesis h3 : c + d = 260
hypothesis h4 : a + d = 295

-- The theorem to prove that the sum of Alex's and Carla's weights is 235 pounds.
theorem alex_carla_weight : a + c = 235 :=
by {
    sorry -- Proof is omitted, we only provide the statement.
}

end alex_carla_weight_l364_364304


namespace image_of_1_2_is_0_neg3_l364_364906

def image_under_mapping (f : ℕ × ℕ → ℕ × ℕ) (p : ℕ × ℕ) : ℕ × ℕ :=
  f p

def mapping (p : ℕ × ℕ) : ℕ × ℕ :=
  (2 * p.1 - p.2, p.1 - 2 * p.2)

theorem image_of_1_2_is_0_neg3 : image_under_mapping mapping (1, 2) = (0, -3) :=
  sorry

end image_of_1_2_is_0_neg3_l364_364906


namespace apple_count_l364_364188

-- Define constants for the problem
constant bags: ℕ → Prop
constant total_apples: ℕ
constant apples_in_bag1: ℕ := 12
constant apples_in_bag2: ℕ := 6

-- Define the conditions
axiom condition1 (n: ℕ) : bags n → (n = apples_in_bag1 ∨ n = apples_in_bag2)
axiom condition2 : 70 ≤ total_apples ∧ total_apples ≤ 80
axiom condition3 : total_apples % 6 = 0

-- State the proof problem
theorem apple_count :
  total_apples = 72 ∨ total_apples = 78 :=
sorry

end apple_count_l364_364188


namespace apples_total_l364_364169

theorem apples_total (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apples_total_l364_364169


namespace percentage_shaded_region_l364_364978

-- Definitions representing conditions
structure Circle (α : Type) :=
(center : α)
(radius : ℝ)

structure Diameter (α : Type) :=
(point1 : α)
(point2 : α)
(center : α)
(is_diameter : point1 ≠ point2 ∧ (point1 - center = center - point2))

-- Let's say R and S are half of a sector QOT which is a quarter of the circle.
structure EqualAreas (α : Type) :=
(area_R : ℝ)
(area_S : ℝ)
(total_area : ℝ)
(equal_areas : area_R = area_S)
(sector_fraction : total_area = 12.5 / 100 * total_area)

noncomputable def shaded_region_percentage (c : Circle ℝ) (d : Diameter ℝ) (e : EqualAreas ℝ) : ℝ :=
  (e.total_area * 12.5) / 100

theorem percentage_shaded_region (c : Circle ℝ) (d : Diameter ℝ) (e : EqualAreas ℝ) :
  shaded_region_percentage c d e = 12.5 := by
  sorry

end percentage_shaded_region_l364_364978


namespace length_of_CE_l364_364508

theorem length_of_CE (A B C E F : Point) (h_concyclic: Concyclic A B C F)
    (h_angle_bisector: IsAngleBisector (∠ABC) CE)
    (h_FB : dist F B = 2)
    (h_EF : dist E F = 1) :
    dist C E = 3 := by
  sorry

end length_of_CE_l364_364508


namespace Christine_savings_l364_364321

theorem Christine_savings 
  (commission_rate: ℝ) 
  (total_sales: ℝ) 
  (personal_needs_percentage: ℝ) 
  (savings: ℝ) 
  (h1: commission_rate = 0.12) 
  (h2: total_sales = 24000) 
  (h3: personal_needs_percentage = 0.60) 
  (h4: savings = total_sales * commission_rate * (1 - personal_needs_percentage)) : 
  savings = 1152 := by 
  sorry

end Christine_savings_l364_364321


namespace number_of_projects_min_total_time_l364_364250

noncomputable def energy_transfer_projects (x y : ℕ) : Prop :=
  x + y = 15 ∧ x = 2 * y - 3

theorem number_of_projects (x y : ℕ) (h : energy_transfer_projects x y) :
  x = 9 ∧ y = 6 :=
sorry

noncomputable def minimize_time (m : ℕ) : Prop :=
  m + 10 - m = 10 ∧ 10 - m > m / 2 ∧ -2 * m + 80 = 68

theorem min_total_time (m : ℕ) (h : minimize_time m) :
  m = 6 ∧ 10 - m = 4 :=
sorry

end number_of_projects_min_total_time_l364_364250


namespace minimum_value_of_f_l364_364344

def f (x : ℝ) : ℝ := x + 4 / x

theorem minimum_value_of_f : ∀ x : ℝ, x > 0 → (f x ≥ 4) ∧ (∀ y : ℝ, y > 0 → f y < f x → false) :=
by
  sorry

end minimum_value_of_f_l364_364344


namespace complex_powers_l364_364797

theorem complex_powers (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) :
  i^(23 : ℕ) + i^(58 : ℕ) = -1 - i :=
by sorry

end complex_powers_l364_364797


namespace train_speed_l364_364291

theorem train_speed (time_seconds : ℕ) (length_meters : ℕ) (speed_kmph : ℕ)
  (h1 : time_seconds = 9) (h2 : length_meters = 135) : speed_kmph = 54 :=
sorry

end train_speed_l364_364291


namespace EX_parallel_AP_l364_364301

open EuclideanGeometry

theorem EX_parallel_AP
  (ABC : Triangle)
  (H : Point) (P : Point) (E : Point) (Q : Point) (R : Point) (X : Point)
  (H_orthocenter : is_orthocenter H ABC)
  (P_on_circumcircle : on_circumcircle P ABC) (P_not_vertex : P ≠ ABC.A ∧ P ≠ ABC.B ∧ P ≠ ABC.C)
  (BE_altitude : is_altitude BE ABC)
  (Q_def : Q = line_through_parallel A (line_through B P) ∩ line_through_parallel B (line_through A P))
  (R_def : R = line_through_parallel A (line_through C P) ∩ line_through_parallel C (line_through A P))
  (X_def : X = line_intersection HR (line_through A Q))
  : is_parallel EX AP :=
  sorry

end EX_parallel_AP_l364_364301


namespace B_pow_2040_eq_I_l364_364013

open Matrix

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  !![sqrt (3/2), 0, -1/sqrt 2;
     0, 1, 0;
     1/sqrt 2, 0, sqrt (3/2)]

theorem B_pow_2040_eq_I : B ^ 2040 = (1 : Matrix (Fin 3) (Fin 3) ℝ) :=
  sorry

end B_pow_2040_eq_I_l364_364013


namespace inclination_angle_l364_364457

theorem inclination_angle (α : ℝ) (h : tan α = -2) : α = π - arctan 2 :=
sorry

end inclination_angle_l364_364457


namespace translate_parabola_property_l364_364871

def translate (f : ℝ → ℝ) (a : ℝ) := λ x, f (x - a)

theorem translate_parabola_property :
  ∀ x : ℝ, x < 2 → (translate (λ x, -x^2 + 1) 2) x > (translate (λ x, -x^2 + 1) 2) (x - 1) :=
by
  sorry

end translate_parabola_property_l364_364871


namespace common_chord_length_l364_364616

theorem common_chord_length : 
  let C1 := {p : ℝ × ℝ | p.1^2 + p.2^2 + 4 * p.1 - 4 * p.2 = 0}
  let C2 := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * p.1 - 12 = 0}
  ( ∃ common_chord, 
    (∀ p ∈ common_chord, p ∈ C1 ∧ p ∈ C2) ∧
    (∃ center radius, (center, radius) ∈ centers_and_radii C1) ∧
    (∃ center radius, (center, radius) ∈ centers_and_radii C2) ∧
    chord_length center radius common_chord = 4 * real.sqrt 2 ) := 
sorry

end common_chord_length_l364_364616


namespace length_GH_l364_364672

-- Definitions of the segments lengths and parallel condition
def segments_parallel (A B C D E F G H : ℝ) : Prop :=
  A + B = C + D ∧ E + F = G + H ∧ A ∥ C ∧ B ∥ D ∧ E ∥ F ∧ G ∥ H

-- Given conditions
def AB_parallel_CD_EF_GH : Prop := 
  segments_parallel 200 200 100 100 66.67 66.67 40 40

-- The main theorem statement
theorem length_GH (AB CD : ℝ) 
  (h_parallel: AB_parallel_CD_EF_GH) : 
  GH = 40 := 
sorry

end length_GH_l364_364672


namespace christine_savings_l364_364322

def commission_rate : ℝ := 0.12
def total_sales : ℝ := 24000
def personal_needs_percentage : ℝ := 0.60
def savings_percentage : ℝ := 1 - personal_needs_percentage

noncomputable def commission_earned : ℝ := total_sales * commission_rate
noncomputable def amount_saved : ℝ := commission_earned * savings_percentage

theorem christine_savings :
  amount_saved = 1152 :=
by
  sorry

end christine_savings_l364_364322


namespace cosine_distinctness_l364_364827

-- Definitions for angles in a convex pentagon
variables (α β γ δ ε : ℝ)

-- Angle sum condition for a convex pentagon
def sum_of_angles_pentagon (α β γ δ ε : ℝ) : Prop :=
  α + β + γ + δ + ε = 3 * Real.pi

-- Condition for having not four distinct sine values
def not_four_distinct_sines (α β γ δ ε : ℝ) : Prop :=
  ¬(Set.card (Set.ofList [Real.sin α, Real.sin β, Real.sin γ, Real.sin δ, Real.sin ε]) = 4)

-- The main theorem statement
theorem cosine_distinctness (h_angles : sum_of_angles_pentagon α β γ δ ε)
  (h_sines : not_four_distinct_sines α β γ δ ε) : ¬ (Function.injective (λ θ, Real.cos θ) [α, β, γ, δ, ε]) :=
sorry

end cosine_distinctness_l364_364827


namespace median_of_triangle_l364_364639

theorem median_of_triangle (a b c : ℝ) (h₁ : a = 11) (h₂ : b = 12) (h₃ : c = 13) :
  let m := (1/2) * (real.sqrt (2*b^2 + 2*c^2 - a^2)) in m = 19 / 2 :=
by
  sorry

end median_of_triangle_l364_364639


namespace condition_necessary_but_not_sufficient_l364_364354
-- Import the entirety of the necessary library

-- Define the conditions and state the theorem
theorem condition_necessary_but_not_sufficient (m : ℝ) (p : m > -3) (q : (m > 1)): ∀ m, (m > -3) → ¬(m > 1) → p ∧ q :=
by
  sorry

end condition_necessary_but_not_sufficient_l364_364354


namespace triangle_perimeter_range_l364_364939

theorem triangle_perimeter_range {A B C a b c : ℝ} 
  (a_eq : a = 1)
  (height_eq : ∃ h, h = tan A ∧ h = a * sin A / 2): 
  ∃ A : ℝ, 
    0 < A ∧ A < π / 2 ∧ 
    ∃ b c : ℝ, 
      b^2 + c^2 = 3 ∧ 
      bc = 1 / cos A ∧ 
      (1 + b + c) > √5 + 1 :=
by 
  sorry

end triangle_perimeter_range_l364_364939


namespace complex_problem_l364_364888

open Complex

theorem complex_problem (z : ℂ) (h : z + z⁻¹ = 2 * √2) : z^2021 + z^(-2021) = 0 :=
by
  sorry

end complex_problem_l364_364888


namespace terms_added_induction_l364_364202

theorem terms_added_induction (n : ℕ) (h₁ : n > 1) : 
  let sum_terms (n : ℕ) := ∑ i in range (2^n), 1 / (i + 1) in
  ∀ (k : ℕ), 1 < k → 
  (sum_terms (k + 1) - sum_terms k).natAbs = 2^k := 
by
  intros
  sorry

end terms_added_induction_l364_364202


namespace sin_double_angle_l364_364881

theorem sin_double_angle (θ : ℝ) (h : sin θ = 1 / 4) : sin (2 * θ) = sqrt 15 / 8 :=
by sorry

end sin_double_angle_l364_364881


namespace expression_evaluates_to_one_l364_364232

noncomputable def a := Real.sqrt 2 + 0.8
noncomputable def b := Real.sqrt 2 - 0.2

theorem expression_evaluates_to_one : 
  ( (2 - b) / (b - 1) + 2 * (a - 1) / (a - 2) ) / ( b * (a - 1) / (b - 1) + a * (2 - b) / (a - 2) ) = 1 :=
by
  sorry

end expression_evaluates_to_one_l364_364232


namespace min_value_expr_l364_364810

theorem min_value_expr : ∃ x : ℝ, (15 - x) * (9 - x) * (15 + x) * (9 + x) = -5184 :=
by
  sorry

end min_value_expr_l364_364810


namespace circumcenter_ADE_on_circumcircle_ABC_l364_364513

namespace GeometryProblem

variable {P Q R S T U V : Type}
variables [metric_space P] -- Assuming P is a metric space for distance and circumcircle

-- Given triangle ABC
variables {A B C O M : P}

-- Conditions
-- Circumcentre (O) and centroid (M) of triangle ABC
variable {circumcenter : P}
variable {centroid : P}
-- OM and AM are perpendicular
variable (is_perpendicular : ∃ (OM AM : P), OM ⟂ AM)
-- AM intersects the circumcircle of ABC again at A'
variable {A' : P}
variable (A'_intersection : ∃ (circumcircle : set P), A' ∈ circumcircle)
-- BA' and AC intersect at D, CA' and AB intersect at E
variable {D E : P}
variable (D_intersection : ∃ (BA' AC : set P), D ∈ (BA' ∩ AC))
variable (E_intersection : ∃ (CA' AB : set P), E ∈ (CA' ∩ AB))

-- Prove that the circumcentre of triangle ADE lies on the circumcircle of triangle ABC
theorem circumcenter_ADE_on_circumcircle_ABC :
  ∃ circumcenter_ADE : P, circumcenter_ADE ∈ (circumcircle : set P) :=
sorry

end GeometryProblem

end circumcenter_ADE_on_circumcircle_ABC_l364_364513


namespace min_selections_l364_364580

def is_representable (selected : Finset ℕ) (n : ℕ) : Prop :=
  n ∈ selected ∨ ∃ a b ∈ selected, a + b = n

def covers_range (selected : Finset ℕ) : Prop :=
  ∀ n, 1 ≤ n ∧ n ≤ 20 → is_representable selected n

theorem min_selections (selected : Finset ℕ) :
  covers_range selected ∧ selected ⊆ Finset.range 11 → selected.card = 6 :=
sorry

end min_selections_l364_364580


namespace intersection_distance_proof_l364_364615

noncomputable def distance_between_intersections : ℝ :=
  let c : ℝ×ℝ → Prop := λ p, (p.1 + 3)^2 + (p.2 - 4)^2 = 50
  let l : ℝ×ℝ → Prop := λ p, p.2 = 2 * p.1 - 5
  let dist (p₁ p₂ : ℝ×ℝ) : ℝ := real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2)
  let sol1 : ℝ×ℝ := (2, -1)
  let sol2 : ℝ×ℝ := (4, 3)
  if c sol1 ∧ l sol1 ∧ c sol2 ∧ l sol2 then
    dist sol1 sol2
  else
    0

theorem intersection_distance_proof :
  distance_between_intersections = 2 * real.sqrt 5 := by
  sorry

end intersection_distance_proof_l364_364615


namespace probability_correct_l364_364891

noncomputable def probability_two_queens_or_at_least_one_jack : ℚ :=
  let total_cards := 52
  let queens := 3
  let jacks := 1
  let prob_two_queens := (queens * (queens - 1)) / (total_cards * (total_cards - 1))
  let prob_one_jack := jacks / total_cards * (total_cards - jacks) / (total_cards - 1) + (total_cards - jacks) / total_cards * jacks / (total_cards - 1)
  prob_two_queens + prob_one_jack

theorem probability_correct : probability_two_queens_or_at_least_one_jack = 9 / 221 := by
  sorry

end probability_correct_l364_364891


namespace plum_difference_l364_364988

theorem plum_difference : 
  let Sharon_plums := 7 in
  let Allan_plums := 10 in
  Allan_plums - Sharon_plums = 3 := 
by
  sorry

end plum_difference_l364_364988


namespace packages_ratio_l364_364728

theorem packages_ratio (packages_yesterday packages_today : ℕ)
  (h1 : packages_yesterday = 80)
  (h2 : packages_today + packages_yesterday = 240) :
  (packages_today / packages_yesterday) = 2 :=
by
  sorry

end packages_ratio_l364_364728


namespace place_circle_without_overlap_l364_364716

noncomputable def large_square_side_length : ℝ := 15
noncomputable def small_square_side_length : ℝ := 1
noncomputable def number_of_small_squares : ℕ := 20
noncomputable def circle_radius : ℝ := 1

theorem place_circle_without_overlap :
  ∃ (point : ℝ × ℝ), 
    (∀ (i : ℕ), 
      (i < number_of_small_squares) → 
      ¬ (ball point circle_radius ∩ (small_square_region i) ≠ ∅)) ∧
  point_within_large_square point :=
begin
  sorry,
end

end place_circle_without_overlap_l364_364716


namespace distance_from_starting_point_return_to_starting_point_fare_for_last_passenger_general_fare_formula_l364_364590

-- Define the driving distances
def distances : List ℤ := [5, -3, 6, -7, 6, -2, -5, -4, 6, -8]

-- Part 1: Proving distance from starting point
theorem distance_from_starting_point : List.sum distances = -6 := 
by
  sorry

-- Part 2: Proving the return to the starting point after the 7th passenger
theorem return_to_starting_point : List.sum (distances.take 7) = 0 :=
by
  sorry

-- Definitions for fare calculation
def starting_fare : ℕ := 8
def additional_fare_per_km : ℤ := 1.5

-- Part 3: Fare for the last passenger
theorem fare_for_last_passenger : 
  let total_distance := distances.nth_le 9 (by decide)
  let exceeding_distance := total_distance - 3
  starting_fare + exceeding_distance * additional_fare_per_km = 15.5 :=
by
  sorry

-- General fare formula for a distance x > 3 km
theorem general_fare_formula (x : ℕ) (hx : x > 3) :
  starting_fare + (x - 3) * additional_fare_per_km = 1.5 * x + 3.5 :=
by
  sorry


end distance_from_starting_point_return_to_starting_point_fare_for_last_passenger_general_fare_formula_l364_364590


namespace num_real_satisfying_x_l364_364445

theorem num_real_satisfying_x : 
  (∃ (x : ℝ), ∃ (s : ℕ) (hs : s ∈ set.Ico 0 (floor (sqrt 123) + 1)), 
   s = int.of_nat (floor (sqrt (123 - sqrt x)))) →
   12 := sorry

end num_real_satisfying_x_l364_364445


namespace contractor_daily_wage_l364_364268

theorem contractor_daily_wage (total_days : ℕ) (absent_days : ℕ) (fine_per_absent_day total_amount : ℝ) (daily_wage : ℝ)
  (h_total_days : total_days = 30)
  (h_absent_days : absent_days = 8)
  (h_fine : fine_per_absent_day = 7.50)
  (h_total_amount : total_amount = 490) 
  (h_work_days : total_days - absent_days = 22)
  (h_total_fined : fine_per_absent_day * absent_days = 60)
  (h_total_earned : 22 * daily_wage - 60 = 490) :
  daily_wage = 25 := 
by 
  sorry

end contractor_daily_wage_l364_364268


namespace find_c_plus_d_l364_364627

theorem find_c_plus_d (c d : ℝ) (h1 : 2 * c = 6) (h2 : c^2 - d = 4) : c + d = 8 := by
  sorry

end find_c_plus_d_l364_364627


namespace collinear_M_O_N_l364_364758

-- Definitions of points and midpoints within a quadrilateral circumscribed about a circle
def isCircumscribedAbout (ABCD : Quadrilateral) (O : Point) (r : ℝ) : Prop :=
  ∀(P ∈ vertices ABCD), distance P O = r

def isMidpoint (M : Point) (A C : Point) : Prop :=
  2 • M = A + C

-- Proving collinearity of points M, O, and N
theorem collinear_M_O_N 
  (A B C D M N O : Point)
  (r : ℝ)
  (h1 : isCircumscribedAbout ⟨A, B, C, D⟩ O r)
  (h2 : isMidpoint M A C)
  (h3 : isMidpoint N B D)
  : Collinear M O N :=
begin
  sorry
end

end collinear_M_O_N_l364_364758


namespace cost_of_replacing_headlights_l364_364319

theorem cost_of_replacing_headlights :
  let asking_price := 5200
  let maintenance_cost := asking_price / 10
  let earnings_from_first_offer := asking_price - maintenance_cost
  let h := 180
  let tire_cost := 3 * h
  let total_cost_to_replace := h + tire_cost
  let earnings_from_second_offer := asking_price - total_cost_to_replace
  let difference_in_earnings := earnings_from_first_offer - earnings_from_second_offer
  difference_in_earnings = 200 :=
by
  let asking_price := 5200
  let maintenance_cost := asking_price / 10
  let earnings_from_first_offer := asking_price - maintenance_cost
  let h := 180
  let tire_cost := 3 * h
  let total_cost_to_replace := h + tire_cost
  let earnings_from_second_offer := asking_price - total_cost_to_replace
  let difference_in_earnings := earnings_from_first_offer - earnings_from_second_offer
  show difference_in_earnings = 200, from sorry

end cost_of_replacing_headlights_l364_364319


namespace constant_term_binomial_expansion_l364_364617

-- Define the general term of the binomial expansion
noncomputable def general_term (r : ℕ) : ℚ :=
  Nat.choose 6 r * (-1) ^ r * (x : ℚ) ^ ((3 / 2) * r - 6)

-- Statement of the problem
theorem constant_term_binomial_expansion : 
  (∃ k : ℚ, k * (x : ℚ) ^ 0 = (general_term 4)) → k = 15 :=
by
  sorry

end constant_term_binomial_expansion_l364_364617


namespace num_real_x_l364_364411

theorem num_real_x (a b : ℝ) (h1 : a = 123) (h2 : b = 11) :
  ∃ n : ℕ, n = 12 ∧
  ∀ k : ℕ, k ≤ b → ∃ x : ℝ, x = (a - k^2)^2 :=
by
  sorry

end num_real_x_l364_364411


namespace monotonicity_g_expression_g_min_value_l364_364387

noncomputable def f (a x : ℝ) := a * x^2 - 2 * x + 1

def isMonotonic (f : ℝ → ℝ) (a : ℝ) :=
  if a > 0 then
    ∀ x y, (x < y ∧ y < 1/a ∨ 1/a < y ∧ y < x) → f x > f y
  else if a < 0 then
    ∀ x y, (x < y ∧ y < 1/a ∨ 1/a < y ∧ y < x) → f x < f y
  else
    False

def M (a : ℝ) :=
  if 1/3 ≤ a ∧ a ≤ 1/2 then 
    f a 1
  else if 1/2 < a ∧ a ≤ 1 then 
    f a 3
  else 
    0

def N (a : ℝ) :=
  1 - 1/a

def g (a : ℝ) := 
  M a - N a

theorem monotonicity (a : ℝ) : 
  a ≠ 0 → isMonotonic (f a) a := 
sorry

theorem g_expression (a : ℝ) (h : 1/3 ≤ a ∧ a ≤ 1) :
  g a = if 1/3 ≤ a ∧ a ≤ 1/2 then a - 2 + 1/a
        else if 1/2 < a ∧ a ≤ 1 then 9*a - 6 + 1/a
        else 0 :=
sorry

theorem g_min_value (a : ℝ) (h1 : 1/3 ≤ a) (h2 : a ≤ 1) :
  g a ≥ 1/2 :=
sorry

end monotonicity_g_expression_g_min_value_l364_364387


namespace defective_chip_ratio_l364_364267

theorem defective_chip_ratio (defective_chips total_chips : ℕ)
  (h1 : defective_chips = 15)
  (h2 : total_chips = 60000) :
  defective_chips / total_chips = 1 / 4000 :=
by
  sorry

end defective_chip_ratio_l364_364267


namespace simplify_and_evaluate_l364_364584

noncomputable def x := Real.tan (Real.pi / 4) + Real.cos (Real.pi / 6)

theorem simplify_and_evaluate :
  ((x / (x ^ 2 - 1)) * ((x - 1) / x - 2)) = - (2 * Real.sqrt 3) / 3 := 
sorry

end simplify_and_evaluate_l364_364584


namespace total_shaded_area_proof_l364_364065

-- Define the conditions in Lean
def diameter : ℝ := 3
def radius : ℝ := diameter / 2
def length_of_pattern : ℝ := 10
def number_of_full_diameters : ℤ := Int.floor (length_of_pattern / diameter)
def area_of_full_circle : ℝ := real.pi * radius ^ 2
def total_shaded_area : ℝ := number_of_full_diameters * area_of_full_circle * 2

-- Prove the total shaded area
theorem total_shaded_area_proof : total_shaded_area = 6.75 * real.pi := by
  sorry

end total_shaded_area_proof_l364_364065


namespace sum_greater_than_four_l364_364983

theorem sum_greater_than_four (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hprod : x * y > x + y) : x + y > 4 :=
by
  sorry

end sum_greater_than_four_l364_364983


namespace eq1_solutions_eq2_solutions_l364_364072

theorem eq1_solutions (x : ℝ) : x^2 - 6 * x + 3 = 0 ↔ (x = 3 + Real.sqrt 6) ∨ (x = 3 - Real.sqrt 6) :=
by {
  sorry
}

theorem eq2_solutions (x : ℝ) : x * (x - 2) = x - 2 ↔ (x = 2) ∨ (x = 1) :=
by {
  sorry
}

end eq1_solutions_eq2_solutions_l364_364072


namespace integer_values_t_l364_364349

-- Define the function f and its properties as conditions
variable (f : ℝ → ℝ)

-- Define the functional equation and specific value conditions
axiom functional_eq : ∀ x y : ℝ, f(x + y) = f(x) + f(y) + x * y + 1
axiom specific_val : f(-2) = -2

-- Define the theorem to be proven
theorem integer_values_t (t: ℤ) : f(t) = t ↔ t = 1 ∨ t = -2 :=
by
  sorry -- Skip the proof

end integer_values_t_l364_364349


namespace find_a_l364_364872

noncomputable def real_variable := ℝ

open MeasureTheory

-- Given conditions: Normal distribution parameters (mean and variance)
def X := MeasureTheory.ProbabilityTheory.Normal 2 (5^2)

-- The problem to solve
theorem find_a :
  (∃ (a : real_variable), P(X ≤ 0) = P(X ≥ a - 2)) ↔ a = 6 :=
by
  sorry

end find_a_l364_364872


namespace f_additive_f_positive_lt_x_zero_f_at_one_f_odd_f_inequality_l364_364829
open Real

noncomputable def f : ℝ → ℝ := sorry

theorem f_additive (a b : ℝ) : f (a + b) = f a + f b := sorry
theorem f_positive_lt_x_zero (x : ℝ) (h_pos : 0 < x) : f x < 0 := sorry
theorem f_at_one : f 1 = 1 := sorry

-- Prove that f is an odd function
theorem f_odd (x : ℝ) : f (-x) = -f x :=
  sorry

-- Solve the inequality: f((log2 x)^2 - log2 (x^2)) > 3
theorem f_inequality (x : ℝ) (h_pos : 0 < x) : (f ((log x / log 2)^2 - (log x^2 / log 2))) > 3 ↔ 1 / 2 < x ∧ x < 8 :=
  sorry

end f_additive_f_positive_lt_x_zero_f_at_one_f_odd_f_inequality_l364_364829


namespace max_books_borrowed_by_one_student_l364_364509

def total_students := 20
def students_no_books := 3
def students_one_book := 9
def students_two_books := 4
def total_books := 40
def unknown_students := total_students - (students_no_books + students_one_book + students_two_books)
def books_per_student_on_average := 2

theorem max_books_borrowed_by_one_student :
  let books_borrowed_by_unknown_students := total_books - (students_no_books * 0 + students_one_book * 1 + students_two_books * 2) in
  let books_borrowed_by_remaining_3_students := 3 * 3 in
  books_borrowed_by_unknown_students - books_borrowed_by_remaining_3_students ≤ 14 :=
by
  sorry

end max_books_borrowed_by_one_student_l364_364509


namespace apple_bags_l364_364149

theorem apple_bags (n : ℕ) (h₁ : n ≥ 70) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 := 
sorry

end apple_bags_l364_364149


namespace circle_reflection_l364_364605

variable (x₀ y₀ : ℝ)

def reflect_over_line_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem circle_reflection 
  (h₁ : x₀ = 8)
  (h₂ : y₀ = -3) :
  reflect_over_line_y_eq_neg_x (x₀, y₀) = (3, -8) := by
  sorry

end circle_reflection_l364_364605


namespace tory_sells_grandmother_l364_364348

theorem tory_sells_grandmother (G : ℕ)
    (total_goal : ℕ) (sold_to_uncle : ℕ) (sold_to_neighbor : ℕ) (remaining_to_sell : ℕ)
    (h_goal : total_goal = 50) (h_sold_to_uncle : sold_to_uncle = 7)
    (h_sold_to_neighbor : sold_to_neighbor = 5) (h_remaining_to_sell : remaining_to_sell = 26) :
    (G + sold_to_uncle + sold_to_neighbor + remaining_to_sell = total_goal) → G = 12 :=
by
    intros h
    -- Proof goes here
    sorry

end tory_sells_grandmother_l364_364348


namespace not_even_not_odd_neither_even_nor_odd_l364_364526

noncomputable def f (x : ℝ) : ℝ := ⌊x⌋ + 1 / 2

theorem not_even (x : ℝ) : f (-x) ≠ f x := sorry
theorem not_odd (x : ℝ) : f (0) ≠ 0 ∨ f (-x) ≠ -f x := sorry

theorem neither_even_nor_odd : ∀ x : ℝ, f (-x) ≠ f x ∧ (f (0) ≠ 0 ∨ f (-x) ≠ -f x) :=
by
  intros x
  exact ⟨not_even x, not_odd x⟩

end not_even_not_odd_neither_even_nor_odd_l364_364526


namespace Shyne_total_plants_l364_364068

theorem Shyne_total_plants :
  ∀ (eggplants_per_packet sunflowers_per_packet eggplant_packets sunflower_packets : ℕ),
  eggplants_per_packet = 14 →
  sunflowers_per_packet = 10 →
  eggplant_packets = 4 →
  sunflower_packets = 6 →
  (eggplants_per_packet * eggplant_packets + sunflowers_per_packet * sunflower_packets = 116) :=
by
  intros eggplants_per_packet sunflowers_per_packet eggplant_packets sunflower_packets
  intro h1 h2 h3 h4
  rw [h1, h3, h2, h4]
  norm_num
  sorry

end Shyne_total_plants_l364_364068


namespace necessary_but_not_sufficient_condition_l364_364375

theorem necessary_but_not_sufficient_condition (m : ℝ) :
  (m + 3) * (2m + 1) < 0 ∧ -(2m - 1) > m + 2 ∧ m + 2 > 0 ↔ (-2 < m ∧ m < -1/3) :=
sorry

end necessary_but_not_sufficient_condition_l364_364375


namespace smallest_palindrome_base3_base5_gt7_l364_364324

noncomputable def is_palindrome {α : Type*} [DecidableEq α] (l : List α) : Prop :=
  l.reverse = l

def to_digits_base (b n : ℕ) : List ℕ :=
  Nat.digits b n

theorem smallest_palindrome_base3_base5_gt7 : ∃ n : ℕ, n > 7 ∧
  is_palindrome (to_digits_base 3 n) ∧
  is_palindrome (to_digits_base 5 n) ∧
  ∀ m : ℕ, (m > 7 ∧
            is_palindrome (to_digits_base 3 m) ∧
            is_palindrome (to_digits_base 5 m)) → n ≤ m :=
exists.intro 26 (by
  sorry -- Proof steps will be elaborated in a complete proof.
)

end smallest_palindrome_base3_base5_gt7_l364_364324


namespace pie_ratio_is_17_13_15_l364_364588

def steve_bakes_pies (apple_pies_monday_friday : ℕ)
                     (blueberry_pies_monday_friday : ℕ)
                     (apple_pies_wednesday : ℕ)
                     (blueberry_pies_wednesday : ℕ)
                     (cherry_pies_tuesday : ℕ)
                     (blueberry_pies_tuesday : ℕ)
                     (cherry_pies_thursday : ℕ)
                     (blueberry_pies_thursday : ℕ)
                     (apple_pies_saturday : ℕ)
                     (cherry_pies_saturday : ℕ)
                     (blueberry_pies_saturday : ℕ)
                     (apple_pies_sunday : ℕ)
                     (cherry_pies_sunday : ℕ)
                     (blueberry_pies_sunday : ℕ) : Prop :=
  let total_apple_pies := 2 * apple_pies_monday_friday + apple_pies_wednesday + apple_pies_saturday + apple_pies_sunday in
  let total_cherry_pies := cherry_pies_tuesday + cherry_pies_thursday + cherry_pies_saturday + cherry_pies_sunday in
  let total_blueberry_pies := 2 * blueberry_pies_monday_friday + blueberry_pies_wednesday + blueberry_pies_tuesday + blueberry_pies_thursday + blueberry_pies_saturday + blueberry_pies_sunday in
  let gcd := Nat.gcd (Nat.gcd total_apple_pies total_cherry_pies) total_blueberry_pies in
  (total_apple_pies / gcd = 17) ∧
  (total_cherry_pies / gcd = 13) ∧
  (total_blueberry_pies / gcd = 15)

theorem pie_ratio_is_17_13_15 :
  steve_bakes_pies 16 10 
                   20 12 
                   14 8 
                   18 10 
                   10 8 6 
                   6 12 4 :=
by
  sorry

end pie_ratio_is_17_13_15_l364_364588


namespace total_apples_l364_364162

theorem total_apples (n : ℕ) (h1 : ∀ k : ℕ, 6 * k = n ∨ 12 * k = n) (h2 : 70 ≤ n) (h3 : n ≤ 80) : 
  n = 72 ∨ n = 78 := 
sorry

end total_apples_l364_364162


namespace johann_oranges_problem_l364_364531

theorem johann_oranges_problem :
  ∀ (initial_oranges johann_ate half_stolen carson_returned : ℕ),
  initial_oranges = 60 →
  johann_ate = 10 →
  half_stolen = (initial_oranges - johann_ate) / 2 →
  carson_returned = 5 →
  initial_oranges - johann_ate - half_stolen + carson_returned = 30 :=
begin
  intros initial_oranges johann_ate half_stolen carson_returned,
  sorry
end

end johann_oranges_problem_l364_364531


namespace train_length_is_400_meters_l364_364298

noncomputable def speed_kmh_to_ms (speed_kmh : ℕ) : ℝ :=
  (speed_kmh * 1000) / 3600

noncomputable def length_of_train (speed_kmh : ℕ) (time_s : ℕ) : ℝ :=
  speed_kmh_to_ms(speed_kmh) * time_s

theorem train_length_is_400_meters :
  length_of_train 180 8 = 400 :=
by
  sorry

end train_length_is_400_meters_l364_364298


namespace gcd_lcm_product_180_l364_364693

theorem gcd_lcm_product_180 :
  ∃ (a b : ℕ), (gcd a b) * (lcm a b) = 180 ∧ 
  let possible_gcd_values := 
    {d | ∃ a b : ℕ, gcd a b = d ∧ (gcd a b) * (lcm a b) = 180} in
  possible_gcd_values.card = 7 :=
by
  sorry

end gcd_lcm_product_180_l364_364693


namespace num_real_x_l364_364407

theorem num_real_x (a b : ℝ) (h1 : a = 123) (h2 : b = 11) :
  ∃ n : ℕ, n = 12 ∧
  ∀ k : ℕ, k ≤ b → ∃ x : ℝ, x = (a - k^2)^2 :=
by
  sorry

end num_real_x_l364_364407


namespace scientific_notation_correct_l364_364568

-- Define the given number
def given_number : ℕ := 138000

-- Define the scientific notation expression
def scientific_notation : ℝ := 1.38 * 10^5

-- The proof goal: Prove that 138,000 expressed in scientific notation is 1.38 * 10^5
theorem scientific_notation_correct : (given_number : ℝ) = scientific_notation := by
  -- Sorry is used to skip the proof
  sorry

end scientific_notation_correct_l364_364568


namespace tiffany_max_points_l364_364189

theorem tiffany_max_points : 
  let initial_money := 3
  let cost_per_game := 1
  let points_red_bucket := 2
  let points_green_bucket := 3
  let rings_per_game := 5
  let games_played := 2
  let red_buckets_first_two_games := 4
  let green_buckets_first_two_games := 5
  let remaining_money := initial_money - games_played * cost_per_game
  let remaining_games := remaining_money / cost_per_game
  let points_first_two_games := red_buckets_first_two_games * points_red_bucket + green_buckets_first_two_games * points_green_bucket
  let max_points_third_game := rings_per_game * points_green_bucket
  points_first_two_games + max_points_third_game = 38 := 
by
  sorry

end tiffany_max_points_l364_364189


namespace problem_statement_l364_364019

def B : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def count_functions (f : ℕ → ℕ) : ℕ :=
  if ∃ c ∈ B, ∀ x ∈ B, f(f(x)) = c then
    1
  else
    0

def M : ℕ := ∑ f in (B → B), count_functions f

theorem problem_statement :
  (M % 500 = 76) :=
sorry

end problem_statement_l364_364019


namespace train_speed_kph_l364_364293

-- Definitions based on conditions
def time_seconds : ℕ := 9
def length_meters : ℕ := 135
def conversion_factor : ℕ := 36 -- 3.6 represented as an integer by multiplying both sides by 10

-- The proof statement
theorem train_speed_kph : (length_meters * conversion_factor / 10 / time_seconds = 54) :=
by
  sorry

end train_speed_kph_l364_364293


namespace find_z_l364_364624

theorem find_z (z : ℚ) : (7 + 11 + 23) / 3 = (15 + z) / 2 → z = 37 / 3 :=
by
  sorry

end find_z_l364_364624


namespace combination_10_4_l364_364919

theorem combination_10_4 : nat.choose 10 4 = 210 := 
by {
  sorry
}

end combination_10_4_l364_364919


namespace tan_sum_sin_cos_conditions_l364_364456

theorem tan_sum_sin_cos_conditions {x y : ℝ} 
  (h1 : Real.sin x + Real.sin y = 1 / 2) 
  (h2 : Real.cos x + Real.cos y = Real.sqrt 3 / 2) :
  Real.tan x + Real.tan y = -Real.sqrt 3 := 
sorry

end tan_sum_sin_cos_conditions_l364_364456


namespace santana_brothers_l364_364574

theorem santana_brothers (b : ℕ) (x : ℕ) (h1 : x + b = 7) (h2 : 3 + 8 = x + 1 + 2 + 7) : x = 1 :=
by
  -- Providing the necessary definitions and conditions
  let brothers := 7 -- Santana has 7 brothers
  let march_birthday := 3 -- 3 brothers have birthdays in March
  let november_birthday := 1 -- 1 brother has a birthday in November
  let december_birthday := 2 -- 2 brothers have birthdays in December
  let total_presents_first_half := 3 -- Total presents in the first half of the year is 3 (March)
  let x := x -- Number of brothers with birthdays in October to be proved
  let total_presents_second_half := x + 1 + 2 + 7 -- Total presents in the second half of the year
  have h3 : total_presents_first_half + 8 = total_presents_second_half := h2 -- Condition equation
  
  -- Start solving the proof
  sorry

end santana_brothers_l364_364574


namespace pounds_of_beef_l364_364940

theorem pounds_of_beef (meals_price : ℝ) (total_sales : ℝ) (meat_per_meal : ℝ) (relationship : ℝ) (total_meat_used : ℝ) (beef_pounds : ℝ) :
  (total_sales = 400) → (meals_price = 20) → (meat_per_meal = 1.5) → (relationship = 0.5) → (20 * meals_price = total_sales) → (total_meat_used = 30) →
  (beef_pounds + beef_pounds * relationship = total_meat_used) → beef_pounds = 20 :=
by
  intros
  sorry

end pounds_of_beef_l364_364940


namespace apple_bags_l364_364180

theorem apple_bags (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apple_bags_l364_364180


namespace find_b_l364_364800

noncomputable def b := (1 : ℝ) / (2 : ℝ)^(3.6)

theorem find_b (b : ℝ) : log b 64 = (-5) / 3 := by
  sorry

end find_b_l364_364800


namespace set_difference_is_single_element_l364_364333

-- Define the sets M and N based on the given conditions
def M : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2002}
def N : Set ℕ := {y | 2 ≤ y ∧ y ≤ 2003}

-- State the theorem that we need to prove
theorem set_difference_is_single_element : (N \ M) = {2003} :=
sorry

end set_difference_is_single_element_l364_364333


namespace total_number_of_crickets_l364_364709

def initial_crickets : ℝ := 7.0
def additional_crickets : ℝ := 11.0
def total_crickets : ℝ := 18.0

theorem total_number_of_crickets :
  initial_crickets + additional_crickets = total_crickets :=
by
  sorry

end total_number_of_crickets_l364_364709


namespace new_member_money_l364_364595

variable (T M : ℝ)
variable (H1 : T / 7 = 20)
variable (H2 : (T + M) / 8 = 14)

theorem new_member_money : M = 756 :=
by
  sorry

end new_member_money_l364_364595


namespace sum_first_nine_terms_arithmetic_sequence_l364_364521

theorem sum_first_nine_terms_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : ∀ n, a (n + 1) - a n = (a 2 - a 1))
  (h1 : a 1 + a 4 + a 7 = 45)
  (h2 : a 3 + a 6 + a 9 = 27) :
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) = 108 := 
sorry

end sum_first_nine_terms_arithmetic_sequence_l364_364521


namespace correct_option_C_l364_364397

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the complements of sets A and B in U
def complA : Set ℕ := {2, 4}
def complB : Set ℕ := {3, 4}

-- Define sets A and B using the complements
def A : Set ℕ := U \ complA
def B : Set ℕ := U \ complB

-- Mathematical proof problem statement
theorem correct_option_C : 3 ∈ A ∧ 3 ∉ B := by
  sorry

end correct_option_C_l364_364397


namespace count_distinct_x_l364_364433

theorem count_distinct_x :
  { x : ℝ | ∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ s = floor (sqrt (123 - sqrt x)) } = 12 :=
sorry

end count_distinct_x_l364_364433


namespace find_primes_l364_364345

theorem find_primes (p : ℕ) (x y : ℕ) (hx : x > 0) (hy : y > 0) (hp : Nat.Prime p) : 
  (x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ (p = 2 ∨ p = 3 ∨ p = 7) := sorry

end find_primes_l364_364345


namespace max_value_of_expression_l364_364022

theorem max_value_of_expression (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 1) (h5 : a * b * c = 1 / 27) :
    a + real.sqrt(a^2 * b) + real.cbrt(a * b * c) ≤ 101 / 96 :=
by
  sorry

end max_value_of_expression_l364_364022


namespace average_length_of_strings_l364_364051

theorem average_length_of_strings :
  let lengths := [1, 3, 5] in (lengths.sum / lengths.length) = 3 := by
  sorry

end average_length_of_strings_l364_364051


namespace midpoint_of_AC_and_CD_l364_364760

theorem midpoint_of_AC_and_CD
  (ABCD : Type) [quadrilateral ABCD]
  (area_ratio : real)
  (AM_AC_eq_CN_CD : ∀ M N : ABCD, AM / AC = CN / CD)
  (B_collinear_MN : ∀ B M N : ABCD, collinear B M N)
  (area_ABD_BCD_ABC : area(∠ABD)/area(∠BCD) = 3/4)
  :  midpoint M AC ∧ midpoint N CD :=
by
  sorry

end midpoint_of_AC_and_CD_l364_364760


namespace elevator_stop_time_l364_364561

def time_to_reach_top (stories time_per_story : Nat) : Nat := stories * time_per_story

def total_time_with_stops (stories time_per_story stop_time : Nat) : Nat :=
  stories * time_per_story + (stories - 1) * stop_time

theorem elevator_stop_time (stories : Nat) (lola_time_per_story elevator_time_per_story total_elevator_time_to_top stop_time_per_floor : Nat)
  (lola_total_time : Nat) (is_slower : Bool)
  (h_lola: lola_total_time = time_to_reach_top stories lola_time_per_story)
  (h_slower: total_elevator_time_to_top = if is_slower then lola_total_time else 220)
  (h_no_stops: time_to_reach_top stories elevator_time_per_story + (stories - 1) * stop_time_per_floor = total_elevator_time_to_top) :
  stop_time_per_floor = 3 := 
  sorry

end elevator_stop_time_l364_364561


namespace apples_total_l364_364166

theorem apples_total (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apples_total_l364_364166


namespace distinct_gcd_count_l364_364700

theorem distinct_gcd_count :
  ∃ (a b : ℕ), (gcd a b * Nat.lcm a b = 180) ∧
  (∀ (d : ℕ), d = gcd a b → 
    d ∈ {1, 2, 3, 5, 6, 10, 15, 30} ∧ 
    (∀ d' ∈ {1, 2, 3, 5, 6, 10, 15, 30}, d' ≠ d → 
      ∃ a' b', gcd a' b' * Nat.lcm a' b' = 180 ∧ gcd a' b' = d')) := sorry

end distinct_gcd_count_l364_364700


namespace Johann_oranges_l364_364536

-- Define the given conditions
def initial_oranges := 60
def eaten_oranges := 10
def half_remaining_oranges := (initial_oranges - eaten_oranges) / 2
def returned_oranges := 5

-- Define the statement to prove
theorem Johann_oranges :
  initial_oranges - eaten_oranges - half_remaining_oranges + returned_oranges = 30 := by
  sorry

end Johann_oranges_l364_364536


namespace johann_oranges_problem_l364_364532

theorem johann_oranges_problem :
  ∀ (initial_oranges johann_ate half_stolen carson_returned : ℕ),
  initial_oranges = 60 →
  johann_ate = 10 →
  half_stolen = (initial_oranges - johann_ate) / 2 →
  carson_returned = 5 →
  initial_oranges - johann_ate - half_stolen + carson_returned = 30 :=
begin
  intros initial_oranges johann_ate half_stolen carson_returned,
  sorry
end

end johann_oranges_problem_l364_364532


namespace particle_speed_l364_364277

def particle_position (t : ℝ) : ℝ × ℝ := (3 * t + 5, 5 * t - 9)

theorem particle_speed : 
  let pos_difference := (particle_position (1) - particle_position (0))
  ∥pos_difference∥ = Real.sqrt 34 :=
by
  admit -- sorry to skip proof

end particle_speed_l364_364277


namespace coeff_of_x_in_expansion_l364_364909

theorem coeff_of_x_in_expansion (x n : ℝ) (h_cond : (∑ k in (finset.range (n.to_nat + 1)), abs (binom (n.to_nat) k * ((sqrt x)^(n.to_nat - k) * (-3 / x)^k))) = 1024) 
: (coeff_x (binom (n.to_nat) 1 * ((sqrt x)^(n.to_nat - 1) * (-3 / x)^1)) = -15) := 
begin
  sorry
end

end coeff_of_x_in_expansion_l364_364909


namespace additional_weekly_rate_l364_364085

theorem additional_weekly_rate (rate_first_week : ℝ) (total_days_cost : ℝ) (days_first_week : ℕ) (total_days : ℕ) (cost_total : ℝ) (cost_first_week : ℝ) (days_after_first_week : ℕ) : 
  (rate_first_week * days_first_week = cost_first_week) → 
  (total_days = days_first_week + days_after_first_week) → 
  (cost_total = cost_first_week + (days_after_first_week * (rate_first_week * 7 / days_first_week))) →
  (rate_first_week = 18) →
  (cost_total = 350) →
  total_days = 23 → 
  (days_first_week = 7) → 
  cost_first_week = 126 →
  (days_after_first_week = 16) →
  rate_first_week * 7 / days_first_week * days_after_first_week = 14 := 
by 
  sorry

end additional_weekly_rate_l364_364085


namespace locus_of_M_circle_l364_364648

-- Define terms
def Sphere : Type := sorry
def Point : Type := sorry

variables (α β ω : Sphere) (A B : Point)
           (touches : ∀ (s₁ s₂ : Sphere) (p : Point), Prop)
           (M N K : Point)
           (chosen_on : Point → Sphere → Prop)
           (intersect : Point → Sphere → Point)

-- Conditions
axiom touches_condition_α : touches α ω A
axiom touches_condition_β : touches β ω B
axiom chosen_on_condition : chosen_on M α
axiom intersect_condition_N : intersect M ω = N
axiom intersect_condition_K : intersect N β = K

-- Locus discovery
theorem locus_of_M_circle :
  ∀ M, chosen_on M α →
        let N := intersect M ω in
        let K := intersect N β in
        is_tangent (MK M K) β → locus_of_M_is_circle M :=
sorry

end locus_of_M_circle_l364_364648


namespace cost_of_bobs_sandwiches_l364_364306

theorem cost_of_bobs_sandwiches :
  ∀ (andy_total bob_fruit_drink : ℝ),
    andy_total = 5 → bob_fruit_drink = 2 → andy_total = 5 →
    (andy_total - bob_fruit_drink) = 3 :=
by
  intros andy_total bob_fruit_drink h1 h2 h3
  have h4 : andy_total = 5 := h1
  have h5 : bob_fruit_drink = 2 := h2
  calc
    (andy_total - bob_fruit_drink) = 5 - 2 : by rw [h4, h5]
                           ... = 3 : by norm_num

end cost_of_bobs_sandwiches_l364_364306


namespace square_of_1008_l364_364325

theorem square_of_1008 : 1008^2 = 1016064 := 
by sorry

end square_of_1008_l364_364325


namespace composite_dice_product_probability_l364_364474

theorem composite_dice_product_probability :
  let outcomes := 6 ^ 4
  let non_composite_ways := 13
  let composite_probability := 1 - non_composite_ways / outcomes
  composite_probability = 1283 / 1296 :=
by
  sorry

end composite_dice_product_probability_l364_364474


namespace composite_dice_product_probability_l364_364473

theorem composite_dice_product_probability :
  let outcomes := 6 ^ 4
  let non_composite_ways := 13
  let composite_probability := 1 - non_composite_ways / outcomes
  composite_probability = 1283 / 1296 :=
by
  sorry

end composite_dice_product_probability_l364_364473


namespace gcd_lcm_product_180_l364_364695

theorem gcd_lcm_product_180 :
  ∃ (a b : ℕ), (gcd a b) * (lcm a b) = 180 ∧ 
  let possible_gcd_values := 
    {d | ∃ a b : ℕ, gcd a b = d ∧ (gcd a b) * (lcm a b) = 180} in
  possible_gcd_values.card = 7 :=
by
  sorry

end gcd_lcm_product_180_l364_364695


namespace smallest_n_for_g_n_eq_4_l364_364551

def g (n : ℕ) : ℕ :=
  -- Define the function to compute the number of distinct pairs (a, b)
  -- such that a^2 + b^2 + ab = n
  (Finset.univ.product Finset.univ).filter (λ (ab : ℕ × ℕ), (ab.1)^2 + (ab.2)^2 + ab.1 * ab.2 = n).card

theorem smallest_n_for_g_n_eq_4 : ∃ n : ℕ, g(n) = 4 ∧ ∀ m : ℕ, (g(m) = 4 → m ≥ n) :=
by sorry

end smallest_n_for_g_n_eq_4_l364_364551


namespace statement_A_incorrect_statement_B_correct_statement_C_incorrect_statement_D_incorrect_l364_364707

-- Statement A
theorem statement_A_incorrect (a b c d : ℝ) (ha : a < b) (hc : c < d) : ¬ (a * c < b * d) := by
  sorry

-- Statement B
theorem statement_B_correct (a b : ℝ) (ha : -2 < a ∧ a < 3) (hb : 1 < b ∧ b < 2) : -1 < a / b ∧ a / b < 3 := by
  sorry

-- Statement C
theorem statement_C_incorrect (m : ℝ) : ¬ (∀ x > 0, x / 2 + 2 / x ≥ m) ∧ (m ≤ 1) := by
  sorry

-- Statement D
theorem statement_D_incorrect : ∃ x : ℝ, (x^2 + 2) + 1 / (x^2 + 2) ≠ 2 := by
  sorry

end statement_A_incorrect_statement_B_correct_statement_C_incorrect_statement_D_incorrect_l364_364707


namespace apple_bags_l364_364154

theorem apple_bags (n : ℕ) (h₁ : n ≥ 70) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 := 
sorry

end apple_bags_l364_364154


namespace ashley_guests_count_l364_364761

variable (number_of_guests bottles servings_per_bottle glasses_per_guest guests_per_bottle : Nat)

-- Conditions
def condition1 : glasses_per_guest = 2 := sorry
def condition2 : servings_per_bottle = 6 := sorry
def condition3 : bottles = 40 := sorry

-- Derived
def guests_per_bottle : Nat := servings_per_bottle / glasses_per_guest

theorem ashley_guests_count : 
  guests_per_bottle * bottles = 120 := by
  sorry

end ashley_guests_count_l364_364761


namespace equal_areas_of_six_triangles_l364_364929

-- Define triangle and centroid
structure Triangle :=
  (A B C : Point)
  [dec_eq : decidable_eq Point]

-- Define point P as centroid
def is_centroid (P : Point) (T : Triangle) : Prop :=
  ∀ M1 M2 M3 : Point, (median T T.A T.B M1 P ∧ median T T.B T.C M2 P ∧ median T T.C T.A M3 P) →
  (∃ k : ℝ, 0 < k ∧ k < 1 ∧ segments_ratio_eq T P k)

-- Median definition
def median (T : Triangle) (A B C: Point) (P : Point) : Prop :=
  ∃ D : Point, midpoint D B P ∧ line_eq A D C

-- Midpoint definition
def midpoint (D B P : Point) : Prop :=
  dist D B = dist D P

-- Segment ratio equality definition
def segments_ratio_eq (T : Triangle) (P : Point) (k : ℝ) : Prop :=
  ∀ part_area : ℝ, part_area = (area T.T.A P T.T.B) / 6 ∧ 
                     part_area = (area T.T.A P T.T.C) / 6 ∧ 
                     part_area = (area T.T.B P T.T.C) / 6 ∧ 
                     part_area = (area T.B P T.T.C) / 6 ∧ 
                     part_area = (area T.B P T.T.A) / 6

-- Area definition
def area (A B C : Point) : ℝ := 
sorry -- To be defined or use existing libraries

-- The mathematical proof problem statement
theorem equal_areas_of_six_triangles (T : Triangle) (P : Point) (H : is_centroid P T) : 
  area T.T.A P T.T.B = (area T) / 6 ∧ 
  area T.T.A P T.T.C = (area T) / 6 ∧ 
  area T.T.B P T.T.C = (area T) / 6 ∧ 
  area T.B P T.T.C = (area T) / 6 ∧ 
  area T.B P T.T.A = (area T) / 6 ∧ 
  sorry -- For proof

end equal_areas_of_six_triangles_l364_364929


namespace min_value_expression_l364_364811

theorem min_value_expression :
  ∀ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ (5 * Real.sqrt 6) / 3 :=
by
  sorry

end min_value_expression_l364_364811


namespace pyramid_volume_with_inscribed_sphere_l364_364003

theorem pyramid_volume_with_inscribed_sphere (r S_0 : ℝ) (S : ℝ) (n : ℕ) (Si : ℕ → ℝ)    (base : ℝ) :
  let V := (1 / 3) * r * (S_0 + ∑ i in finset.range n, Si i) in
  V = (1 / 3) * S * r :=
sorry

end pyramid_volume_with_inscribed_sphere_l364_364003


namespace apple_count_l364_364181

-- Define constants for the problem
constant bags: ℕ → Prop
constant total_apples: ℕ
constant apples_in_bag1: ℕ := 12
constant apples_in_bag2: ℕ := 6

-- Define the conditions
axiom condition1 (n: ℕ) : bags n → (n = apples_in_bag1 ∨ n = apples_in_bag2)
axiom condition2 : 70 ≤ total_apples ∧ total_apples ≤ 80
axiom condition3 : total_apples % 6 = 0

-- State the proof problem
theorem apple_count :
  total_apples = 72 ∨ total_apples = 78 :=
sorry

end apple_count_l364_364181


namespace impossible_to_half_boys_sit_with_girls_l364_364643

theorem impossible_to_half_boys_sit_with_girls:
  ∀ (g b : ℕ), 
  (g + b = 30) → 
  (∃ k, g = 2 * k) →
  (∀ (d : ℕ), 2 * d = g) →
  ¬ ∃ m, (b = 2 * m) ∧ (∀ (d : ℕ), 2 * d = b) :=
by
  sorry

end impossible_to_half_boys_sit_with_girls_l364_364643


namespace power_mod_11_l364_364674

theorem power_mod_11 (n : ℕ) : (3 ^ 2040) % 11 = 1 := 
by
  -- Definitions and calculations from conditions identified in a)
  have h1 : 3 % 11 = 3 := by norm_num
  have h2 : (3 ^ 2) % 11 = 9 := by norm_num
  have h3 : (3 ^ 3) % 11 = 5 := by norm_num
  have h4 : (3 ^ 4) % 11 = 4 := by norm_num
  have h5 : (3 ^ 5) % 11 = 1 := by norm_num
  -- Using the property derived from h5
  have key : (3 ^ 2040) % 11 = (3 ^ (5 * 408)) % 11 := by norm_num
  rw [pow_mul, h5, one_pow]
  exact h5

end power_mod_11_l364_364674


namespace num_real_numbers_l364_364432

theorem num_real_numbers (x : ℝ) : 
  (∃ n : ℕ, n ≤ 11 ∧ x = (123 - n^2)^2) ↔ 
  (∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ (x = (123 - s^2)^2) ∧ 
  (∃! x, ∃ k : ℕ, 0 ≤ k ∧ k ≤ 11 ∧ x = (123 - k^2)^2)) :=
by
  sorry

end num_real_numbers_l364_364432


namespace solve_system_l364_364060

theorem solve_system :
  ∃ x y : ℝ, 2 * x + y = 7 ∧ 4 * x + 5 * y = 11 ∧ x = 4 ∧ y = -1 :=
by {
  use [4, -1],
  split,
  -- 2 * x + y = 7
  { norm_num },
  split,
  -- 4 * x + 5 * y = 11
  { norm_num },
  -- x=4 and y=-1
  split; refl
}

end solve_system_l364_364060


namespace find_u_plus_v_l364_364880

theorem find_u_plus_v (u v : ℚ) (h1 : 3 * u - 7 * v = 17) (h2 : 5 * u + 3 * v = 1) : 
  u + v = - 6 / 11 :=
  sorry

end find_u_plus_v_l364_364880


namespace range_of_r_for_symmetric_points_l364_364922

def point_symmetric_wrt_line (P : ℝ × ℝ) (L : ℝ × ℝ) : ℝ × ℝ :=
  (2 * L.1 - P.1, 2 * L.2 - P.2)

def circle (centre : ℝ × ℝ) (radius : ℝ) : set (ℝ × ℝ) :=
  { P | (P.1 - centre.1) ^ 2 + (P.2 - centre.2) ^ 2 = radius ^ 2 }

theorem range_of_r_for_symmetric_points (r : ℝ) (P Q : ℝ × ℝ)
(h1 : P ∈ circle (0, 1) r)
(h2 : Q = point_symmetric_wrt_line P (P.2, P.1))
(h3 : Q ∈ circle (2, 1) 1) :
  ∃ r, r ∈ set.Icc (Real.sqrt 2 - 1) (Real.sqrt 2 + 1) :=
sorry

end range_of_r_for_symmetric_points_l364_364922


namespace sum_of_two_lowest_scores_l364_364585

theorem sum_of_two_lowest_scores
  (scores : List ℕ)
  (h_length : scores.length = 6)
  (h_mean : (scores.sum : ℚ) / 6 = 85)
  (h_median: List.median scores = 88)
  (h_mode: List.mode scores = 90) :
  (scores.sorted.headI + scores.sorted.tail.headI = 154) :=
sorry

end sum_of_two_lowest_scores_l364_364585


namespace cos_squared_sum_gte_three_fourths_l364_364247

-- The following statement defines the mathematical problem
theorem cos_squared_sum_gte_three_fourths
  (α β γ : ℝ)
  (h : α + β + γ = 180) :
  (Real.cos α)^2 + (Real.cos β)^2 + (Real.cos γ)^2 ≥ 3 / 4 := by
  sorry

end cos_squared_sum_gte_three_fourths_l364_364247


namespace possible_apple_counts_l364_364129

theorem possible_apple_counts (n : ℕ) (h₁ : 70 ≤ n) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end possible_apple_counts_l364_364129


namespace dice_product_composite_probability_l364_364475

theorem dice_product_composite_probability :
  (let total_outcomes := (6 : ℕ)^4,
       non_composite_outcomes := 1,
       composite_probability := 1 - (non_composite_outcomes / total_outcomes) in
   composite_probability = 1295 / 1296) :=
by sorry

end dice_product_composite_probability_l364_364475


namespace circumcenter_distances_equal_l364_364196

-- Definitions from the conditions
variable (O1 O2 A B C D O3 : Type)
variable [Circle O1] [Circle O2]
variable (isMeet : MeetAt O1 O2 A B)
variable (isAngleBisector : AngleBisector O1 A O2 C D)
variable (isCircumcenter : Circumcenter O3 C B D)

-- The theorem to show equal distances from circumcenter of triangle CBD to O1 and O2
theorem circumcenter_distances_equal :
  distance O3 O1 = distance O3 O2 :=
by
  sorry

end circumcenter_distances_equal_l364_364196


namespace madeline_utilities_l364_364562

variables
  (rent : ℕ)
  (groceries : ℕ)
  (medical_expenses : ℕ)
  (emergency_savings : ℕ)
  (hourly_wage : ℕ)
  (hours_worked : ℕ)

def total_expenses_without_utilities : ℕ :=
  rent + groceries + medical_expenses + emergency_savings

def total_income : ℕ :=
  hourly_wage * hours_worked

def amount_for_utilities (rent groceries medical_expenses emergency_savings hourly_wage hours_worked : ℕ) : ℕ :=
  total_income - total_expenses_without_utilities

theorem madeline_utilities 
  (h1 : rent = 1200)
  (h2 : groceries = 400)
  (h3 : medical_expenses = 200)
  (h4 : emergency_savings = 200)
  (h5 : hourly_wage = 15)
  (h6 : hours_worked = 138) :
  amount_for_utilities rent groceries medical_expenses emergency_savings hourly_wage hours_worked = 70 :=
by
  sorry

end madeline_utilities_l364_364562


namespace andy_incorrect_l364_364307

theorem andy_incorrect (a b c d : ℕ) (h1 : a + b = c + d) (h2 : a + d = b + c + 6) (h3 : c = 8) : a = 14 :=
by
  sorry

end andy_incorrect_l364_364307


namespace trigonometric_identity_l364_364998

theorem trigonometric_identity :
  let sin_30 := 1 / 2,
      sin_60 := Real.sqrt 3 / 2,
      cos_30 := Real.sqrt 3 / 2,
      cos_60 := 1 / 2,
      sin_45 := Real.sqrt 2 / 2,
      cos_45 := Real.sqrt 2 / 2 in
  (sin_30 + sin_60) / (cos_30 + cos_60) = Real.tan (Real.pi / 4) := 
  by
    sorry

end trigonometric_identity_l364_364998


namespace least_number_to_subtract_l364_364679

theorem least_number_to_subtract (n : ℕ) : (n = 5) → (5000 - n) % 37 = 0 :=
by sorry

end least_number_to_subtract_l364_364679


namespace coronavirus_diameter_in_scientific_notation_l364_364047

noncomputable def nanometers_to_millimeters (n : ℝ) : ℝ := n * 1e-6

theorem coronavirus_diameter_in_scientific_notation :
  nanometers_to_millimeters 80 = 8 * 10^(-5) :=
by
  sorry

end coronavirus_diameter_in_scientific_notation_l364_364047


namespace arithmetic_mean_prime_numbers_l364_364805

-- Define the list of numbers
def num_list := [33, 35, 37, 39, 41]

-- Define a predicate to check for prime numbers
def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

-- Filter the prime numbers from the list
def prime_numbers (l : List ℕ) : List ℕ := l.filter is_prime

-- Compute the arithmetic mean of a list of numbers
def arithmetic_mean (l : List ℕ) : ℕ := l.sum / l.length

-- Target theorem
theorem arithmetic_mean_prime_numbers :
  arithmetic_mean (prime_numbers num_list) = 39 := by
  sorry

end arithmetic_mean_prime_numbers_l364_364805


namespace least_time_for_at_least_six_horses_to_meet_l364_364642

theorem least_time_for_at_least_six_horses_to_meet (S : ℕ) (times : Fin 12 → ℕ) 
  (h_times : times = ![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) 
  (h_S : S = 27720) :
  ∃ (T : ℕ), (T = 60) ∧ nat.sum_digits T = 6 :=
by
  sorry

end least_time_for_at_least_six_horses_to_meet_l364_364642


namespace find_triangle_side_length_l364_364309

theorem find_triangle_side_length (side_length : ℕ)
  (height : ℕ)
  (common_base : ℕ)
  (area_difference : ℕ)
  (x y : ℕ) :
  side_length = 10 →
  common_base = 5 →
  2 * (area_difference / height) = x - y →
  area_difference = 10 →
  x - y = 4 →
  x = 4 :=
by
  intros hside hbase hdifference harea htriangle
  rw hdifference at htriangle
  rw htriangle
  exact rfl
  sorry

end find_triangle_side_length_l364_364309


namespace train_speed_l364_364297

theorem train_speed 
(length_of_train : ℕ) 
(time_to_cross_pole : ℕ) 
(h_length : length_of_train = 135) 
(h_time : time_to_cross_pole = 9) : 
  (length_of_train / time_to_cross_pole) * 3.6 = 54 :=
by 
  sorry

end train_speed_l364_364297


namespace soccer_team_substitutions_modulo_l364_364743

theorem soccer_team_substitutions_modulo :
  let total_players : ℕ := 22
  let starters : ℕ := 11
  let substitutes : ℕ := 11
  let max_substitutions : ℕ := 4
  let no_substitutions : ℕ := 1
  let one_substitution : ℕ := 11 * 11
  let two_substitutions : ℕ := (Nat.choose 11 2) * 2! * 11 * 10
  let three_substitutions : ℕ := (Nat.choose 11 3) * 3! * 11 * 10 * 9
  let four_substitutions : ℕ := (Nat.choose 11 4) * 4! * 11 * 10 * 9 * 8
  let total_ways : ℕ := no_substitutions + one_substitution + two_substitutions + three_substitutions + four_substitutions
  total_ways % 1000 = 122 :=
by 
  sorry

end soccer_team_substitutions_modulo_l364_364743


namespace range_of_a_l364_364841

def f (x: ℝ) : ℝ := x + 1
def g (x : ℝ) (a : ℝ) : ℝ := 2^(abs (x + 2)) + a

theorem range_of_a (a : ℝ) : 
  (∀ (x1 : ℝ), 3 ≤ x1 ∧ x1 ≤ 4 → ∃ (x2 : ℝ), -3 ≤ x2 ∧ x2 ≤ 1 ∧ f x1 ≥ g x2 a) → 
  a ≤ 3 :=
by
  sorry

end range_of_a_l364_364841


namespace ratio_of_area_of_CDGE_to_ABC_l364_364001

/-- Given a triangle ABC with medians AD and BE meeting at the centroid G,
    and D and E being midpoints of sides BC and AC respectively, 
    prove the ratio of the area of quadrilateral CDGE to the area of triangle ABC is 1/3. -/
theorem ratio_of_area_of_CDGE_to_ABC (A B C D E G : Point) (h1 : IsMedian A D)
  (h2 : IsMedian B E) (h3 : IsCentroid G A D B E) (h4 : Midpoint D B C) (h5 : Midpoint E A C) :
  area (Quadrilateral C D G E) / area (Triangle A B C) = 1 / 3 :=
  sorry

end ratio_of_area_of_CDGE_to_ABC_l364_364001


namespace part_a_part_b_part_c_l364_364027

open Finset

-- Define N, S1, and S2
variables {N : ℕ} (S1 S2 : Finset ℕ)

-- Define the condition of a "good division"
def is_good_division (S1 S2 : Finset ℕ) : Prop :=
  (S1 ∩ S2 = ∅) ∧ (S1 ∪ S2 = range (N + 1)) ∧ (S1.card ≠ 0) ∧ (S2.card ≠ 0) ∧ 
  (∑ x in S1, x = ∏ x in S2, x)

-- Part (a)
theorem part_a : ∃ (S1 S2 : Finset ℕ), is_good_division 7 {2, 4, 5, 7} {1, 3, 6} :=
by { sorry }

-- Part (b)
theorem part_b : ∃ N ≥ 1, (∃ S1 S2 : Finset ℕ, is_good_division N S1 S2) ∧ (∃ S1' S2' : Finset ℕ, is_good_division N S1' S2' ∧ (S1 ≠ S1' ∨ S2 ≠ S2')) :=
by { sorry }

-- Part (c)
theorem part_c (N : ℕ) (hN : N ≥ 5) : ∃ S1 S2 : Finset ℕ, is_good_division N S1 S2 :=
by { sorry }

end part_a_part_b_part_c_l364_364027


namespace number_of_gcd_values_l364_364687

theorem number_of_gcd_values (a b : ℤ) (h : a * b = 180) : 
  {d : ℤ | d = Int.gcd a b}.finite.toFinset.card = 8 := 
sorry

end number_of_gcd_values_l364_364687


namespace existence_of_ys_l364_364650

open Complex

theorem existence_of_ys (n : ℕ) (h : n ≥ 2) (x : Fin (n + 1) → ℝ) :
    (∃ y : Fin (n + 1) → ℝ, let z := λ k, x k + Complex.I * y k 
    in z 0 ^ 2 = (Finset.univ.erase 0).sum (λ k, (z k) ^ 2)) ↔
    x 0 ^ 2 ≤ (Finset.univ.erase 0).sum (λ k, x k ^ 2) := 
sorry

end existence_of_ys_l364_364650


namespace factorial_divisibility_l364_364893

theorem factorial_divisibility {p : ℕ} (hp : 1 < p) : (p^2)! % (p!)^(p+1) = 0 :=
by
  sorry

end factorial_divisibility_l364_364893


namespace nature_of_roots_real_distinct_l364_364331

theorem nature_of_roots_real_distinct (d : ℝ) 
  (h : 3 * d = 36) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
  (3 : ℝ) * x₁^2 - (4 : ℝ) * x₁ * real.sqrt 3 + d = 0 ∧ 
  (3 : ℝ) * x₂^2 - (4 : ℝ) * x₂ * real.sqrt 3 + d = 0 :=
by
  sorry

end nature_of_roots_real_distinct_l364_364331


namespace product_of_primes_l364_364224

theorem product_of_primes : 5 * 7 * 997 = 34895 :=
by
  sorry

end product_of_primes_l364_364224


namespace value_of_expression_l364_364702

theorem value_of_expression (x : ℝ) (h : x^2 + 3*x + 5 = 7) : 3*x^2 + 9*x - 2 = 4 :=
by
  -- The proof will be filled here; it's currently skipped using 'sorry'
  sorry

end value_of_expression_l364_364702


namespace binary_111011_is_59_l364_364781

def binary_to_decimal (b : List ℕ) : ℕ :=
  b.reverse.foldl (λ acc bit, 2 * acc + bit) 0

theorem binary_111011_is_59 :
  binary_to_decimal [1, 1, 1, 0, 1, 1] = 59 := by
  sorry

end binary_111011_is_59_l364_364781


namespace rectangle_area_diagonal_ratio_l364_364110

theorem rectangle_area_diagonal_ratio (d : ℝ) (x : ℝ) (h_ratio : 5 * x ≥ 0 ∧ 2 * x ≥ 0)
  (h_diagonal : d^2 = (5 * x)^2 + (2 * x)^2) :
  ∃ k : ℝ, (5 * x) * (2 * x) = k * d^2 ∧ k = 10 / 29 :=
by
  sorry

end rectangle_area_diagonal_ratio_l364_364110


namespace minimum_a_l364_364840

noncomputable def f (x : ℝ) := x - Real.exp (x - Real.exp 1)

theorem minimum_a (a : ℝ) (x1 x2 : ℝ) (hx : x2 - x1 ≥ Real.exp 1)
  (hy : Real.exp x1 = 1 + Real.log (x2 - a)) : a ≥ Real.exp 1 - 1 :=
by
  sorry

end minimum_a_l364_364840


namespace ant_paths_difference_l364_364305

-- Define the problem parameters
def grid_size : ℕ := 10

def N (k : ℕ) : ℕ := sorry  -- Placeholder for the exact definition of N

-- Statement of the proof problem
theorem ant_paths_difference : N(6) - N(5) = 3456 :=
by
  sorry

end ant_paths_difference_l364_364305


namespace converse_opposite_l364_364706

theorem converse_opposite (x y : ℝ) : (x + y = 0) → (y = -x) :=
by
  sorry

end converse_opposite_l364_364706


namespace find_x_value_l364_364890

noncomputable def solve_some_number (x : ℝ) : Prop :=
  let expr := (x - (8 / 7) * 5 + 10)
  expr = 13.285714285714286

theorem find_x_value : ∃ x : ℝ, solve_some_number x ∧ x = 9 := by
  sorry

end find_x_value_l364_364890


namespace appears_in_yk_three_times_iff_l364_364034

noncomputable def appear_in_sequence_x (n : ℕ) (a : ℝ) : ℕ :=
  ⌈(n + 0.5) / a⌉.to_nat - 1

noncomputable def appear_in_sequence_y (n : ℕ) (a : ℝ) : ℕ :=
  ((⌈(n + 0.5) * (1 / a + 2)⌉.to_nat - 1) : ℕ)

theorem appears_in_yk_three_times_iff (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a * b = a - b) (n : ℕ) :
  (∃ k : ℕ, (appear_in_sequence_x n a) = k) ↔
  (∃ k : ℕ, (appear_in_sequence_y n a) = k ∧ k ≥ 3) :=
sorry

end appears_in_yk_three_times_iff_l364_364034


namespace exp_lt_one_plus_x_l364_364225

theorem exp_lt_one_plus_x {x : ℝ} (h : x ≠ 0) : exp x > 1 + x :=
by
  sorry

end exp_lt_one_plus_x_l364_364225


namespace annual_return_percentage_l364_364310

theorem annual_return_percentage (P_purchase P_profit : ℕ) (P_purchase_eq : P_purchase = 5000)
  (P_profit_eq : P_profit = 400) : 
  ((P_profit / P_purchase.to_rat) * 100) = 8 := 
by 
  -- Given conditions
  have h1 : P_purchase = 5000 := P_purchase_eq,
  have h2 : P_profit = 400 := P_profit_eq,
  -- Perform the division
  have h3 : (P_profit.to_rat / P_purchase.to_rat = 0.08),
  { rw [h1, h2],
    norm_cast,
    norm_num,
  },
  -- Convert to percentage
  exact (by linarith : 0.08 * 100 = 8)

sorry -- this line should be replaced by actual proof steps

end annual_return_percentage_l364_364310


namespace replace_basket_after_n_people_l364_364052

theorem replace_basket_after_n_people (n placards_per_person total_placards : ℕ) (h : placards_per_person * n < total_placards) : n = 411 :=
by
  have np := 2 * 411
  have tp := 823
  -- np < tp should hold
  have h : 2 * 411 < 823
  sorry

end replace_basket_after_n_people_l364_364052


namespace option_d_not_equal_four_thirds_l364_364229

theorem option_d_not_equal_four_thirds :
  1 + (2 / 7) ≠ 4 / 3 :=
by
  sorry

end option_d_not_equal_four_thirds_l364_364229


namespace distinct_real_x_l364_364412

theorem distinct_real_x (x : ℝ) :
  (∃! x, ∃ (s : ℝ), s ∈ set.Icc 0 11 ∧ s = real.sqrt (123 - real.sqrt x)) =
  12 := sorry

end distinct_real_x_l364_364412


namespace books_price_arrangement_l364_364817

theorem books_price_arrangement (c : ℝ) (prices : Fin 40 → ℝ)
  (h₁ : ∀ i : Fin 39, prices i.succ = prices i + 3)
  (h₂ : prices ⟨39, by norm_num⟩ = prices ⟨19, by norm_num⟩ + prices ⟨20, by norm_num⟩) :
  prices 20 = prices 19 + 3 := 
sorry

end books_price_arrangement_l364_364817


namespace greatest_three_digit_multiple_of_23_l364_364663

def is_three_digit (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

def is_multiple_of_23 (n : ℕ) : Prop :=
  n % 23 = 0

theorem greatest_three_digit_multiple_of_23 :
  ∀ n, is_three_digit n ∧ is_multiple_of_23 n → n ≤ 989 :=
by
  sorry

end greatest_three_digit_multiple_of_23_l364_364663


namespace octagon_diagonal_intersections_l364_364752

theorem octagon_diagonal_intersections :
  let n := 8 in
  let diagonals := n * (n - 3) / 2 in
  let intersections := Nat.choose n 4 in
  intersections = 70 :=
by
  let n := 8
  have diagonals := n * (n - 3) / 2
  have intersections := Nat.choose n 4
  show intersections = 70
  sorry

end octagon_diagonal_intersections_l364_364752


namespace apple_bags_l364_364179

theorem apple_bags (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apple_bags_l364_364179


namespace probability_of_composite_l364_364488

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ m < n ∧ 1 < k ∧ k < n ∧ m * k = n

def dice_outcomes (faces : ℕ) (rolls : ℕ) : ℕ :=
  faces ^ rolls

def non_composite_product_ways : ℕ :=
  1 + (3 * 4)  -- one way for all 1s, plus combinations of (1,1,1,{2,3,5})

def total_outcomes : ℕ :=
  dice_outcomes 6 4  -- 6^4 total possible outcomes

def probability_composite : ℚ :=
  1 - (non_composite_product_ways / total_outcomes)

theorem probability_of_composite:
  probability_composite = 1283 / 1296 := 
by
  sorry

end probability_of_composite_l364_364488


namespace bacteria_population_at_15_l364_364007

noncomputable def bacteria_population (t : ℕ) : ℕ := 
  20 * 2 ^ (t / 3)

theorem bacteria_population_at_15 : bacteria_population 15 = 640 := by
  sorry

end bacteria_population_at_15_l364_364007


namespace solve_equation_l364_364814

theorem solve_equation (x : ℝ) : (2*x - 1)^2 = 81 ↔ (x = 5 ∨ x = -4) :=
by
  sorry

end solve_equation_l364_364814


namespace greatest_three_digit_multiple_23_l364_364657

theorem greatest_three_digit_multiple_23 : 
  ∃ n : ℕ, n < 1000 ∧ n % 23 = 0 ∧ (∀ m : ℕ, m < 1000 ∧ m % 23 = 0 → m ≤ n) ∧ n = 989 :=
sorry

end greatest_three_digit_multiple_23_l364_364657


namespace total_skateboarding_distance_l364_364011

def skateboarded_to_park : ℕ := 16
def skateboarded_back_home : ℕ := 9

theorem total_skateboarding_distance : 
  skateboarded_to_park + skateboarded_back_home = 25 := by 
  sorry

end total_skateboarding_distance_l364_364011


namespace max_b_in_box_l364_364116

theorem max_b_in_box (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b ≤ 12 := 
by
  sorry

end max_b_in_box_l364_364116


namespace probability_composite_product_l364_364465

theorem probability_composite_product :
  let dice_faces := 6
  let rolls := 4
  let total_outcomes := dice_faces ^ rolls
  let non_composite_cases := 13
  let non_composite_probability := non_composite_cases / total_outcomes
  let composite_probability := 1 - non_composite_probability
  composite_probability = 1283 / 1296 := by
  sorry

end probability_composite_product_l364_364465


namespace solution_set_inequality_range_of_m_l364_364389

def f (x : ℝ) (m : ℝ) : ℝ := m - |x - 1| - |x + 1|

-- Problem 1
theorem solution_set_inequality (x : ℝ) : 
  (f x 5 > 2) ↔ (-3 / 2 < x ∧ x < 3 / 2) :=
sorry

-- Problem 2
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = (x^2 + 2 * x + 3) ∧ y = f x m) ↔ (m ≥ 4) :=
sorry

end solution_set_inequality_range_of_m_l364_364389


namespace johann_oranges_l364_364534

/-
  Johann had 60 oranges. He decided to eat 10.
  Once he ate them, half were stolen by Carson.
  Carson returned exactly 5. 
  How many oranges does Johann have now?
-/
theorem johann_oranges (initial_oranges : Nat) (eaten_oranges : Nat) (carson_returned : Nat) : 
  initial_oranges = 60 → eaten_oranges = 10 → carson_returned = 5 → 
  (initial_oranges - eaten_oranges) / 2 + carson_returned = 30 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end johann_oranges_l364_364534


namespace number_of_zeros_of_f_is_two_l364_364631

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then x^2 + 2 * x - 3 else -2 + Real.log x

theorem number_of_zeros_of_f_is_two : {x : ℝ // f x = 0}.toFinset.card = 2 := by
  sorry

end number_of_zeros_of_f_is_two_l364_364631


namespace max_value_of_expression_l364_364500

theorem max_value_of_expression 
  (x y : ℝ) 
  (h : 2 * x^2 - 6 * x + y^2 = 0) : 
  x^2 + y^2 + 2 * x ≤ 15 := sorry

end max_value_of_expression_l364_364500


namespace apple_count_l364_364182

-- Define constants for the problem
constant bags: ℕ → Prop
constant total_apples: ℕ
constant apples_in_bag1: ℕ := 12
constant apples_in_bag2: ℕ := 6

-- Define the conditions
axiom condition1 (n: ℕ) : bags n → (n = apples_in_bag1 ∨ n = apples_in_bag2)
axiom condition2 : 70 ≤ total_apples ∧ total_apples ≤ 80
axiom condition3 : total_apples % 6 = 0

-- State the proof problem
theorem apple_count :
  total_apples = 72 ∨ total_apples = 78 :=
sorry

end apple_count_l364_364182


namespace eval_difference_of_squares_l364_364798

theorem eval_difference_of_squares :
  (81^2 - 49^2 = 4160) :=
by
  -- Since the exact mathematical content is established in a formal context, 
  -- we omit the detailed proof steps.
  sorry

end eval_difference_of_squares_l364_364798


namespace number_of_gcd_values_l364_364686

theorem number_of_gcd_values (a b : ℤ) (h : a * b = 180) : 
  {d : ℤ | d = Int.gcd a b}.finite.toFinset.card = 8 := 
sorry

end number_of_gcd_values_l364_364686


namespace total_cost_is_130_l364_364197

-- Defining the number of each type of pet
def n_puppies : ℕ := 2
def n_kittens : ℕ := 2
def n_parakeets : ℕ := 3

-- Defining the cost of one parakeet
def c_parakeet : ℕ := 10

-- Defining the cost of one puppy and one kitten based on the conditions
def c_puppy : ℕ := 3 * c_parakeet
def c_kitten : ℕ := 2 * c_parakeet

-- Defining the total cost of all pets
def total_cost : ℕ :=
  (n_puppies * c_puppy) + (n_kittens * c_kitten) + (n_parakeets * c_parakeet)

-- Lean theorem stating that the total cost is 130 dollars
theorem total_cost_is_130 : total_cost = 130 := by
  -- The proof will be filled in here.
  sorry

end total_cost_is_130_l364_364197


namespace round_to_nearest_hundredth_l364_364203

theorem round_to_nearest_hundredth : real :=
  let x := 2.3449
  let rounded := 2.34
  (round (x * 100) / 100 = rounded) :=
begin
  sorry
end

end round_to_nearest_hundredth_l364_364203


namespace exists_twelve_distinct_x_l364_364452

theorem exists_twelve_distinct_x :
  ∃ S : Set ℝ, (S.card = 12) ∧ (∀ x ∈ S, ∃ k : ℤ, 0 ≤ k ∧ k ≤ 11 ∧ (sqrt (123 - sqrt x) = k)) :=
by
  sorry

end exists_twelve_distinct_x_l364_364452


namespace vertex_A_east_probability_l364_364936

theorem vertex_A_east_probability (A B C : Type) (angle_A : ℝ) 
  (h : angle_A = 40) : 
  probability_A_east(A, B, C) = 7 / 18 := by
  sorry

end vertex_A_east_probability_l364_364936


namespace loan_principal_and_repayment_amount_l364_364326

theorem loan_principal_and_repayment_amount (P R : ℝ) (r : ℝ) (years : ℕ) (total_interest : ℝ)
    (h1: r = 0.12)
    (h2: years = 3)
    (h3: total_interest = 5400)
    (h4: total_interest / years = R)
    (h5: R = P * r) :
    P = 15000 ∧ R = 1800 :=
sorry

end loan_principal_and_repayment_amount_l364_364326


namespace julie_reimbursement_l364_364575

-- Conditions
def num_lollipops : ℕ := 12
def total_cost_dollars : ℝ := 3
def shared_fraction : ℝ := 1 / 4

-- Conversion factor
def dollars_to_cents : ℝ := 100

-- Question to prove
theorem julie_reimbursement : 
  let cost_per_lollipop := total_cost_dollars / num_lollipops in
  let num_shared_lollipops := (shared_fraction * num_lollipops : ℝ) in
  let total_cost_shared_lollipops := cost_per_lollipop * num_shared_lollipops in
  let reimbursement_cents := total_cost_shared_lollipops * dollars_to_cents in
  reimbursement_cents = 75 :=
by
  sorry

end julie_reimbursement_l364_364575


namespace probability_of_3_tails_in_8_flips_l364_364350

open ProbabilityTheory

/-- The probability of getting exactly 3 tails out of 8 flips of an unfair coin, where the probability of tails is 4/5 and the probability of heads is 1/5, is 3584/390625. -/
theorem probability_of_3_tails_in_8_flips :
  let p_heads := 1 / 5
  let p_tails := 4 / 5
  let n_trials := 8
  let k_successes := 3
  let binomial_coefficient := Nat.choose n_trials k_successes
  let probability := binomial_coefficient * (p_tails ^ k_successes) * (p_heads ^ (n_trials - k_successes))
  probability = (3584 : ℚ) / 390625 := 
by 
  sorry

end probability_of_3_tails_in_8_flips_l364_364350


namespace apple_bags_l364_364156

theorem apple_bags (n : ℕ) (h₁ : n ≥ 70) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 := 
sorry

end apple_bags_l364_364156


namespace probability_of_absolute_difference_gt_one_over_three_l364_364986

def roll (d : ℕ) : set ℝ :=
  if d = 1 ∨ d = 2 then set.Icc 0 (1/2)
  else if d = 3 ∨ d = 4 then set.Icc (1/2) 1
  else if d = 5 then {0}
  else if d = 6 then {1}
  else ∅

def prob_event (p : ℝ) (d1 d2 : ℕ) (e : set (ℝ × ℝ)) : ℝ :=
  ∫⁻ x in roll d1 ∫⁻ y in roll d2, (e (x, y)) ∂(measure_theory.measure_space.volume)

theorem probability_of_absolute_difference_gt_one_over_three :
  ∀ (d1 d2 : ℕ), d1 ∈ finset.range 6 → d2 ∈ finset.range 6 →
  prob_event 1 (d1 + 1) (d2 + 1) (fun pxy => |fst pxy - snd pxy| > (1/3)) = 2 / 9 :=
begin
  sorry
end

end probability_of_absolute_difference_gt_one_over_three_l364_364986


namespace power_of_10_eq_4_l364_364889

theorem power_of_10_eq_4
  (x : ℕ)
  (h : (10^x * 3.456789)^11 = t)
  (digits_right_of_decimal : (if t >= 1 then t else 1 / t).natAbs.digits = 22) :
  x = 4 :=
by
  sorry

end power_of_10_eq_4_l364_364889


namespace detect_and_correct_error_minimum_undetectable_errors_l364_364241

-- Define the function to compute the extended n+1 by n+1 matrix
def extended_matrix (n : ℕ) (M : matrix (fin n) (fin n) ℤ) : matrix (fin (n+1)) (fin (n+1)) ℤ :=
  λ i j, if h : ↑i < n then 
            if h' : ↑j < n then M i j 
            else (finset.univ.sum (λ k, M i k)) % 2
         else if h' : ↑j < n then (finset.univ.sum (λ k, M k j)) % 2
               else (finset.univ.sum (λ i, finset.univ.sum (λ j, M i j))) % 2

-- Part (a) Proof: 
theorem detect_and_correct_error (n : ℕ) (M : matrix (fin n) (fin n) ℤ) (T : matrix (fin (n+1)) (fin (n+1)) ℤ) :
  (∀ i j, T i j = extended_matrix n M i j) →
  (∃ i j, T i j ≠ (extended_matrix n M i j)) →
  ∃ i j, (T.update i (T i).update j (T i j + 1) = extended_matrix n M) :=
by
  sorry

-- Part (b) Minimum number of undetectable errors:
theorem minimum_undetectable_errors (n : ℕ) (M : matrix (fin n) (fin n) ℤ) (T : matrix (fin (n+1)) (fin (n+1)) ℤ) :
  (∀ i j, T i j = extended_matrix n M i j) →
  ∀ (E : finset (fin (n+1) × fin (n+1))), E.card < 4 →
  (∀ i j, (E.sum (λ p, if T p.1 p.2 ≠ extended_matrix n M p.1 p.2 then 1 else 0))) = 0 :=
by
  sorry

end detect_and_correct_error_minimum_undetectable_errors_l364_364241


namespace sin_of_angle_terminal_point_l364_364832

-- Definition of the conditions: point on the terminal side of angle α
def point_on_terminal_side (x y : ℝ) (α : ℝ) : Prop := 
  x = 1 ∧ y = -2 ∧ α = real.atan2 y x

-- The problem statement rewritten in Lean 4
theorem sin_of_angle_terminal_point (α : ℝ) :
  point_on_terminal_side 1 (-2) α →
  real.sin α = - (2 * real.sqrt 5) / 5 :=
by
  sorry

end sin_of_angle_terminal_point_l364_364832


namespace simplify_trig_expression_l364_364992

theorem simplify_trig_expression : 
  (sin (Real.pi / 6) + sin (Real.pi / 3)) / (cos (Real.pi / 6) + cos (Real.pi / 3)) = 1 := by
  sorry

end simplify_trig_expression_l364_364992


namespace range_of_m_l364_364504

theorem range_of_m (m : ℝ) : (∀ x, 0 ≤ x ∧ x ≤ m → -6 ≤ x^2 - 4 * x - 2 ∧ x^2 - 4 * x - 2 ≤ -2) → 2 ≤ m ∧ m ≤ 4 :=
by
  sorry

end range_of_m_l364_364504


namespace possible_apple_counts_l364_364125

theorem possible_apple_counts (n : ℕ) (h₁ : 70 ≤ n) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end possible_apple_counts_l364_364125


namespace apple_count_l364_364146

/--
Given the total number of apples in several bags, where each bag contains either 12 or 6 apples,
and knowing that the total number of apples is between 70 and 80 inclusive,
prove that the total number of apples can only be 72 or 78.
-/
theorem apple_count (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : ∃ k : ℕ, n = 6 * k) : n = 72 ∨ n = 78 :=
by {
  sorry
}

end apple_count_l364_364146


namespace find_speed_of_stream_l364_364711

variable (b s : ℝ)

-- Equation derived from downstream condition
def downstream_equation := b + s = 24

-- Equation derived from upstream condition
def upstream_equation := b - s = 10

theorem find_speed_of_stream
  (b s : ℝ)
  (h1 : downstream_equation b s)
  (h2 : upstream_equation b s) :
  s = 7 := by
  -- placeholder for the proof
  sorry

end find_speed_of_stream_l364_364711


namespace greatest_three_digit_multiple_of_23_is_991_l364_364660

theorem greatest_three_digit_multiple_of_23_is_991 :
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 23 = 0) ∧ ∀ m : ℤ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 23 = 0) → m ≤ n :=
begin
  use 991,
  -- proof steps go here
  sorry
end

end greatest_three_digit_multiple_of_23_is_991_l364_364660


namespace children_neither_blue_nor_red_is_20_l364_364762

-- Definitions
def num_children : ℕ := 45
def num_adults : ℕ := num_children / 3
def num_adults_blue : ℕ := num_adults / 3
def num_adults_red : ℕ := 4
def num_adults_other_colors : ℕ := num_adults - num_adults_blue - num_adults_red
def num_children_red : ℕ := 15
def num_remaining_children : ℕ := num_children - num_children_red
def num_children_other_colors : ℕ := num_remaining_children / 2
def num_children_blue : ℕ := 2 * num_adults_blue
def num_children_neither_blue_nor_red : ℕ := num_children - num_children_red - num_children_blue

-- Theorem statement
theorem children_neither_blue_nor_red_is_20 : num_children_neither_blue_nor_red = 20 :=
  by
  sorry

end children_neither_blue_nor_red_is_20_l364_364762


namespace quadrilateral_is_isosceles_trapezium_l364_364249

noncomputable def fifth_roots_of_unity_exclude_one : List ℂ :=
  [exp (2 * Real.pi * Complex.I / 5), exp (4 * Real.pi * Complex.I / 5), exp (6 * Real.pi * Complex.I / 5), exp (8 * Real.pi * Complex.I / 5)]

theorem quadrilateral_is_isosceles_trapezium
    (z1 z2 z3 z4 : ℂ)
    (h1 : z1 ^ 4 + z1 ^ 3 + z1 ^ 2 + z1 + 1 = 0)
    (h2 : z2 ^ 4 + z2 ^ 3 + z2 ^ 2 + z2 + 1 = 0)
    (h3 : z3 ^ 4 + z3 ^ 3 + z3 ^ 2 + z3 + 1 = 0)
    (h4 : z4 ^ 4 + z4 ^ 3 + z4 ^ 2 + z4 + 1 = 0)
    (h_dist : List.pairwise (fun (x y : ℂ) => x ≠ y) [z1, z2, z3, z4])
    (h_rotations : List.mem z1 fifth_roots_of_unity_exclude_one ∧
                   List.mem z2 fifth_roots_of_unity_exclude_one ∧
                   List.mem z3 fifth_roots_of_unity_exclude_one ∧
                   List.mem z4 fifth_roots_of_unity_exclude_one) :
  is_isosceles_trapezium z1 z2 z3 z4 := sorry

end quadrilateral_is_isosceles_trapezium_l364_364249


namespace area_of_figure_eq_two_l364_364593

theorem area_of_figure_eq_two :
  ∫ x in (1 / Real.exp 1)..(Real.exp 1), 1 / x = 2 :=
by sorry

end area_of_figure_eq_two_l364_364593


namespace julie_reimburses_sarah_l364_364577

theorem julie_reimburses_sarah : 
  ∀ (dollars_in_cents lollipops total_cost_sarah share_fraction),
    dollars_in_cents = 100 →
    lollipops = 12 →
    total_cost_sarah = 300 →
    share_fraction = 1 / 4 →
    ((total_cost_sarah / lollipops) * (lollipops * share_fraction)) = 75 :=
by
  intros dollars_in_cents lollipops total_cost_sarah share_fraction h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end julie_reimburses_sarah_l364_364577


namespace greatest_three_digit_multiple_of_23_l364_364667

theorem greatest_three_digit_multiple_of_23 : ∃ n : ℕ, n % 23 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℕ, m % 23 = 0 ∧ 100 ≤ m ∧ m ≤ 999 → m ≤ n := 
by
  use 989
  split
  · -- 989 is a multiple of 23
    exact (by norm_num : 989 % 23 = 0)
  · split
    · -- 989 is at least 100
      exact (by norm_num : 100 ≤ 989)
    · split
      · -- 989 is at most 999
        exact (by norm_num : 989 ≤ 999)
      · -- 989 is the greatest such number within the range
        sorry

end greatest_three_digit_multiple_of_23_l364_364667


namespace KT_tangent_to_Γ_l364_364544

open EuclideanGeometry

noncomputable def setup (R S T J A K : Point) (Ω Γ : Circle) (ℓ : Line) : Prop :=
  R ≠ S ∧
  ¬Diameter Ω R S ∧
  Tangent ℓ Ω R ∧
  Midpoint S R T ∧
  (∃ J, ArcContains (short_arc Ω R S) J ∧
     Intersects (Circumcircle J S T) ℓ ∧
     CloserTo R A A (Circumcircle J S T) ℓ) ∧
  OnCircle K Ω ∧
  OnLine A J K

theorem KT_tangent_to_Γ :
  ∀ (R S T J A K : Point) (Ω Γ : Circle) (ℓ : Line),
    setup R S T J A K Ω Γ ℓ →
    Tangent (LineThrough K T) Γ T :=
by
  intros R S T J A K Ω Γ ℓ h
  cases h
  cases h_right_right
  cases h_left
  cases h_right
  sorry

end KT_tangent_to_Γ_l364_364544


namespace Shyne_total_plants_l364_364067

theorem Shyne_total_plants :
  let eggplants_per_packet := 14 in
  let sunflowers_per_packet := 10 in
  let packets_of_eggplants := 4 in
  let packets_of_sunflowers := 6 in
  let total_plants := (eggplants_per_packet * packets_of_eggplants) + (sunflowers_per_packet * packets_of_sunflowers) in
  total_plants = 116 := 
by
  -- The proof would go here
  sorry

end Shyne_total_plants_l364_364067


namespace remainder_when_sum_divided_mod7_l364_364958

theorem remainder_when_sum_divided_mod7 (a b c : ℕ)
  (h1 : a < 7) (h2 : b < 7) (h3 : c < 7)
  (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0)
  (h7 : a * b * c % 7 = 2)
  (h8 : 3 * c % 7 = 1)
  (h9 : 4 * b % 7 = (2 + b) % 7) :
  (a + b + c) % 7 = 3 := by
  sorry

end remainder_when_sum_divided_mod7_l364_364958


namespace convex_polygon_obtuse_adjacent_angles_l364_364989

theorem convex_polygon_obtuse_adjacent_angles (n : ℕ) (hn : n ≥ 7) : 
  ∃ i, (i < n) ∧ interior_angle (P i) > 90 ∧ interior_angle (P (i+1) % n) > 90 :=
by
  sorry

end convex_polygon_obtuse_adjacent_angles_l364_364989


namespace problem_solution_equiv_l364_364773

noncomputable def triangle_area_tangency {O1 O2 O3 : Type} [metric_space O1] [metric_space O2] [metric_space O3]
  (d12 : ℝ) (d23 : ℝ) (d13 : ℝ) (h1 : dist O1 O2 = 3)
  (h2 : dist O2 O3 = 5)
  (h3 : dist O1 O3 = 4) : ℝ :=
  let A := 3 in
  let B := 4 in
  let C := 5 in
  let area := 0.5 * A * B in
  area / 1.0  -- substitute with the semi-perimeter if more complex 

-- Main theorem statement
theorem problem_solution_equiv : triangle_area_tangency 3 5 4 = 6 / 5 := sorry

end problem_solution_equiv_l364_364773


namespace seashells_given_l364_364944

theorem seashells_given (initial left given : ℕ) (h1 : initial = 8) (h2 : left = 2) (h3 : given = initial - left) : given = 6 := by
  sorry

end seashells_given_l364_364944


namespace coins_configuration_l364_364982

theorem coins_configuration (n : ℕ) : 
  (∃ m : ℕ, n = 4 * m) ↔ 
  (∃ moves : list (ℕ × ℕ), 
    ∀ i ∈ finset.range n, 
      let Ai := (i + 1) in 
      let A_next := (n + 1 - (i + 1)) in 
      (sum_in_moves : (ℕ × ℕ) × (ℕ) → int :=
        λ ((a, b), _), 
        if a = b 
        then 0 
        else if (a + 1) % n = b % n ∨ (a + n - 1) % n = b % n
        then 2 
        else 0 
      ∧ Ai - A_next = total_change) :=
sorry

end coins_configuration_l364_364982


namespace arithmetic_sequence_problem_l364_364360

theorem arithmetic_sequence_problem 
  (a : ℕ → ℚ) (d : ℚ) (h_d : d ≠ 0) 
  (h1 : a 1 + a 4 = 14)
  (h2 : a 2 ^ 2 = a 1 * a 7) :
  (∀ n : ℕ, a n = 4 * n - 3) ∧ 
  (∀ n : ℕ, (∑ i in finset.range n, a (i + 1)) = 2 * n^2 - n) ∧ 
  (∀ k, 
    let S : ℕ → ℚ := λ n, 2 * n^2 - n,
        b : ℕ → ℚ := λ n, S n / (n + k) in 
        is_arith_seq b → 
        let T : ℕ → ℚ := λ n, nat.rec_on n 0 (λ n T_n, 
                      T_n + 1 / (b n * b (n + 1))) in 
        ∀ n : ℕ, T n ≤ 1/2
  ) :=
sorry

end arithmetic_sequence_problem_l364_364360


namespace product_of_0_25_and_0_75_is_0_1875_l364_364769

noncomputable def product_of_decimals : ℝ := 0.25 * 0.75

theorem product_of_0_25_and_0_75_is_0_1875 :
  product_of_decimals = 0.1875 :=
by
  sorry

end product_of_0_25_and_0_75_is_0_1875_l364_364769


namespace percentage_increase_l364_364008

theorem percentage_increase
  (black_and_white_cost color_cost : ℕ)
  (h_bw : black_and_white_cost = 160)
  (h_color : color_cost = 240) :
  ((color_cost - black_and_white_cost) * 100) / black_and_white_cost = 50 :=
by
  sorry

end percentage_increase_l364_364008


namespace gilled_mushrooms_count_l364_364264

theorem gilled_mushrooms_count : 
  ∀ (total_mushrooms gilled_mushrooms_ratio spotted_mushrooms_ratio : ℕ),
  (total_mushrooms = 30) →
  (gilled_mushrooms_ratio = 1) →
  (spotted_mushrooms_ratio = 9) →
  total_mushrooms / (gilled_mushrooms_ratio + spotted_mushrooms_ratio) = 3 :=
by
  intros total_mushrooms gilled_mushrooms_ratio spotted_mushrooms_ratio
  assume h_total h_gilled h_spotted
  rw [h_total, h_gilled, h_spotted]
  norm_num
  sorry

end gilled_mushrooms_count_l364_364264


namespace find_ellipse_and_coordinates_P_l364_364362

noncomputable def ellipse_passes_through_points : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  (∀ (x y : ℝ), (x, y) = (-real.sqrt 2, (2 * real.sqrt 6) / 3) ∨
                  (x, y) = (real.sqrt 3, real.sqrt 2) → 
                  x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def line_intersects_at_P : Prop :=
  ∃ (P : ℝ × ℝ), P.2 = 0 ∧ 
  (∀ (A B : ℝ × ℝ), A ≠ B ∧ 
  ∃ C : ℝ × ℝ, C.2 = 0 ∧ 
                |A.1 - B.1| = 2 * real.sqrt 3 * |P.1 - C.1|)

theorem find_ellipse_and_coordinates_P : 
  ellipse_passes_through_points ∧ line_intersects_at_P → 
  (∃ a b : ℝ, a = real.sqrt 6 ∧ b = 2 ∧ 
   ∃ P : ℝ × ℝ, P = (real.sqrt 2, 0) ∨ P = (-real.sqrt 2, 0)) :=
sorry

end find_ellipse_and_coordinates_P_l364_364362


namespace incorrect_option_D_l364_364680

-- Conditions
variables {a b c t : ℝ}
axiom h1 : a ≠ 0
axiom h2 : a + b + c = 4
axiom h3 : c = 3
axiom h4 : a - b + c = 0

-- Prove that option D is incorrect with the given conditions.
theorem incorrect_option_D : ¬ ∀ p : ℝ, ( ∀ x : ℝ, -x^2 + 2*x + 3 < 2*x + p ) → p ≥ 3 :=
begin
  sorry
end

end incorrect_option_D_l364_364680


namespace minimum_flower_cost_l364_364062

def vertical_strip_width : ℝ := 3
def horizontal_strip_height : ℝ := 2
def bed_width : ℝ := 11
def bed_height : ℝ := 6

def easter_lily_cost : ℝ := 3
def dahlia_cost : ℝ := 2.5
def canna_cost : ℝ := 2

def vertical_strip_area : ℝ := vertical_strip_width * bed_height
def horizontal_strip_area : ℝ := horizontal_strip_height * bed_width
def overlap_area : ℝ := vertical_strip_width * horizontal_strip_height
def remaining_area : ℝ := (bed_width * bed_height) - vertical_strip_area - (horizontal_strip_area - overlap_area)

def easter_lily_area : ℝ := horizontal_strip_area - overlap_area
def dahlia_area : ℝ := vertical_strip_area
def canna_area : ℝ := remaining_area

def easter_lily_total_cost : ℝ := easter_lily_area * easter_lily_cost
def dahlia_total_cost : ℝ := dahlia_area * dahlia_cost
def canna_total_cost : ℝ := canna_area * canna_cost

def total_cost : ℝ := easter_lily_total_cost + dahlia_total_cost + canna_total_cost

theorem minimum_flower_cost : total_cost = 157 := by
  sorry

end minimum_flower_cost_l364_364062


namespace derivative_of_parametric_eq_l364_364715

-- Define x as a function of t
def x (t : ℝ) : ℝ := (1 + (Real.cos t)^2)^2

-- Define y as a function of t
def y (t : ℝ) : ℝ := (Real.cos t) / ((Real.sin t)^2)

-- The main theorem stating the condition and the expected derivative result
theorem derivative_of_parametric_eq (t : ℝ) (ht : Real.sin t ≠ 0) :
  deriv y t / deriv x t = 1 / (4 * (Real.cos t) * (Real.sin t)^4) :=
by
  sorry

end derivative_of_parametric_eq_l364_364715


namespace complex_product_l364_364364

noncomputable def cos_15_deg : ℂ := (Real.cos (Real.pi / 12) : ℂ)
noncomputable def sin_15_deg : ℂ := (Real.sin (Real.pi / 12) : ℂ)

noncomputable def z1 : ℂ := 1 + complex.I
noncomputable def z2 : ℂ := (Real.sqrt 2) * (cos_15_deg + complex.I * sin_15_deg)

theorem complex_product :
  z1 * z2 = 1 + (Real.sqrt 3) * complex.I := 
sorry

end complex_product_l364_364364


namespace four_digit_number_count_l364_364819

theorem four_digit_number_count :
  ∃ (count : ℕ), (count = 18) ∧ 
    (∀ (a b c d : ℕ), 
      a ∈ {1} ∧ b ∈ {2, 3, 4} ∧ c ∈ {2, 3, 4} ∧ d ∈ {2, 3, 4} ∧ 
      (a = 1) ∧ (b = c ∨ b = d ∨ c = d) ∧ 
      b ≠ d → True
    ) := sorry

end four_digit_number_count_l364_364819


namespace sum_arithmetic_sequence_ge_four_l364_364837

theorem sum_arithmetic_sequence_ge_four
  (a_n : ℕ → ℚ) -- arithmetic sequence
  (S : ℕ → ℚ) -- sum of the first n terms of the sequence
  (h_arith_seq : ∀ n, S n = (n * a_n 1) + (n * (n - 1) / 2) * (a_n 2 - a_n 1))
  (p q : ℕ)
  (hpq_ne : p ≠ q)
  (h_sp : S p = p / q)
  (h_sq : S q = q / p) :
  S (p + q) ≥ 4 :=
by
  sorry

end sum_arithmetic_sequence_ge_four_l364_364837


namespace mid_segment_length_of_trapezoid_l364_364927

theorem mid_segment_length_of_trapezoid
  (KN LM KL MN : ℝ)
  (hKN : KN = 25)
  (hLM : LM = 15)
  (hKL : KL = 6)
  (hMN : MN = 8) :
  let PQ := (KN + LM) / 2 
  in PQ = 20 :=
by
  sorry

end mid_segment_length_of_trapezoid_l364_364927


namespace monotonic_increasing_intervals_l364_364098

noncomputable def y (x : ℝ) : ℝ := x^2 * Real.exp x

theorem monotonic_increasing_intervals :
  { x : ℝ | ∀ x, y' x > 0 } = set.Ioo (-∞ : ℝ) (-2) ∪ set.Ioo 0 (∞ : ℝ) :=
by
  sorry

end monotonic_increasing_intervals_l364_364098


namespace probability_at_least_three_heads_l364_364077

def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem probability_at_least_three_heads :
  (∑ k in {3, 4, 5}, binomial 5 k) = 16 → (16 / 32 = 1 / 2) :=
by
  sorry

end probability_at_least_three_heads_l364_364077


namespace num_real_x_l364_364408

theorem num_real_x (a b : ℝ) (h1 : a = 123) (h2 : b = 11) :
  ∃ n : ℕ, n = 12 ∧
  ∀ k : ℕ, k ≤ b → ∃ x : ℝ, x = (a - k^2)^2 :=
by
  sorry

end num_real_x_l364_364408


namespace landmark_postcards_probability_l364_364045

theorem landmark_postcards_probability :
  let total_postcards := 12
  let landmark_postcards := 4
  let total_arrangements := Nat.factorial total_postcards
  let favorable_arrangements := Nat.factorial (total_postcards - landmark_postcards + 1) * Nat.factorial landmark_postcards
  favorable_arrangements / total_arrangements = (1:ℝ) / 55 :=
by
  sorry

end landmark_postcards_probability_l364_364045


namespace y_square_range_l364_364461

theorem y_square_range (y : ℝ) (h : (y + 16) ^ (1/3) - (y - 16) ^ (1/3) = 4) : 
  230 ≤ y^2 ∧ y^2 < 240 :=
sorry

end y_square_range_l364_364461


namespace lambda_range_l364_364826

noncomputable def vector_a : ℝ × ℝ := (3, 2)
noncomputable def vector_b : ℝ × ℝ := (2, -1)

theorem lambda_range (λ : ℝ) :
  let a := vector_a
  let b := vector_b
  let l1 := (λ * a.1 + b.1, λ * a.2 + b.2)
  let l2 := (a.1 + λ * b.1, a.2 + λ * b.2)
  let dot_product := l1.1 * l2.1 + l1.2 * l2.2
  let acute_angle := dot_product > 0
  let collinear_condition := (l1.1 * l2.2 ≠ l1.2 * l2.1)
  in
  (acute_angle ∧ collinear_condition) → 
  (λ > (-9 + real.sqrt 65) / 4 ∨ λ < (-9 - real.sqrt 65) / 4) ∧ λ ≠ 1 :=
by
  sorry

end lambda_range_l364_364826


namespace infinite_primes_4k1_l364_364059

theorem infinite_primes_4k1 : ∀ (P : List ℕ), (∀ (p : ℕ), p ∈ P → Nat.Prime p ∧ ∃ k, p = 4 * k + 1) → 
  ∃ q, Nat.Prime q ∧ ∃ k, q = 4 * k + 1 ∧ q ∉ P :=
sorry

end infinite_primes_4k1_l364_364059


namespace max_seq_value_l364_364838

def is_arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∀ n m, a (n + m) = a n + a m

variables (a : ℕ → ℤ)
variables (S : ℕ → ℤ)

axiom distinct_terms (h : is_arithmetic_seq a) : ∀ n m, n ≠ m → a n ≠ a m
axiom condition_1 : ∀ n, a (2 * n) = 2 * a n - 3
axiom condition_2 : a 6 * a 6 = a 1 * a 21
axiom sum_of_first_n_terms : ∀ n, S n = n * (n + 4)

noncomputable def seq (n : ℕ) : ℤ := S n / 2^(n - 1)

theorem max_seq_value : 
  (∀ n, seq n >= seq (n - 1) ∧ seq n >= seq (n + 1)) → 
  (∃ n, seq n = 6) :=
sorry

end max_seq_value_l364_364838


namespace valid_number_of_apples_l364_364137

theorem valid_number_of_apples (n : ℕ) : 
  (∃ k m : ℕ, n = 12 * k + 6 * m) ∧ (70 ≤ n ∧ n ≤ 80) ↔ (n = 72 ∨ n = 78) :=
by
  sorry

end valid_number_of_apples_l364_364137


namespace starWars_earnings_calculation_l364_364083

def lionKing_cost : ℝ := 10
def lionKing_earnings : ℝ := 200
def starWars_cost : ℝ := 25
def lionKing_profit (earnings cost : ℝ) : ℝ := earnings - cost
def starWars_profit (lk_profit : ℝ) : ℝ := 2 * lk_profit
def total_starWars_earnings (profit cost : ℝ) : ℝ := profit + cost

theorem starWars_earnings_calculation :
  let lk_profit := lionKing_profit lionKing_earnings lionKing_cost in
  total_starWars_earnings (starWars_profit lk_profit) starWars_cost = 405 :=
by
  sorry

end starWars_earnings_calculation_l364_364083


namespace total_blue_marbles_correct_l364_364942

def total_blue_marbles (j t e : ℕ) : ℕ :=
  j + t + e

theorem total_blue_marbles_correct :
  total_blue_marbles 44 24 36 = 104 :=
by
  sorry

end total_blue_marbles_correct_l364_364942


namespace composite_dice_product_probability_l364_364469

theorem composite_dice_product_probability :
  let outcomes := 6 ^ 4
  let non_composite_ways := 13
  let composite_probability := 1 - non_composite_ways / outcomes
  composite_probability = 1283 / 1296 :=
by
  sorry

end composite_dice_product_probability_l364_364469


namespace tree_height_l364_364857

theorem tree_height (shadow_tree shadow_pole : ℝ) 
  (height_pole : ℝ) (h_ratio : height_pole / shadow_pole = (height_pole / shadow_pole)) : 
  (height_pole / shadow_pole) * shadow_tree = 15 :=
by
  -- Alias to ease readability
  let height_pole := 1.5
  let shadow_pole := 3
  let shadow_tree := 30
  have ratio := height_pole / shadow_pole
  have height_tree := ratio * shadow_tree
  have : height_tree = 15 := sorry
  exact this

#eval tree_height 30 3 1.5 sorry

end tree_height_l364_364857


namespace algebra_power_9_l364_364883

theorem algebra_power_9 (a b : ℝ) (h1 : b = (sqrt (3 - a)) + (sqrt (a - 3)) + 2) (h2 : 3 - a ≥ 0) (h3 : a - 3 ≥ 0) : a^b = 9 :=
by sorry

end algebra_power_9_l364_364883


namespace greatest_three_digit_multiple_of_23_l364_364669

theorem greatest_three_digit_multiple_of_23 : ∃ n : ℕ, n % 23 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℕ, m % 23 = 0 ∧ 100 ≤ m ∧ m ≤ 999 → m ≤ n := 
by
  use 989
  split
  · -- 989 is a multiple of 23
    exact (by norm_num : 989 % 23 = 0)
  · split
    · -- 989 is at least 100
      exact (by norm_num : 100 ≤ 989)
    · split
      · -- 989 is at most 999
        exact (by norm_num : 989 ≤ 999)
      · -- 989 is the greatest such number within the range
        sorry

end greatest_three_digit_multiple_of_23_l364_364669


namespace wind_horizontal_displacement_max_horizontal_distance_calm_distance_wind_in_direction_distance_wind_opposite_direction_l364_364300

-- Define given conditions
def height : ℝ := 15
def wind_speed : ℝ := 16
def gravity : ℝ := 9.81

-- a) Horizontal displacement due to the wind
theorem wind_horizontal_displacement (h : height = 15) (c : wind_speed = 16) (g : gravity = 9.81) :
  let t := 2 * real.sqrt (2 * h / g) in
  c * t = 56 :=
by sorry

-- b) Maximum horizontal distance in calm weather
theorem max_horizontal_distance_calm (h : height = 15) (g : gravity = 9.81) :
  let v_0 := real.sqrt (2 * h * g) in
  v_0 * (2 * v_0 / g) = 60 :=
by sorry

-- c) Distance with wind in the direction of the pipe's tilt
theorem distance_wind_in_direction (h : height = 15) (c : wind_speed = 16) (g : gravity = 9.81) :
  let v_0 := real.sqrt (2 * h * g) in
  v_0 * (2 * v_0 / g) + c * (2 * v_0 / g) = 116 :=
by sorry

-- d) Distance with wind in the opposite direction of the pipe's tilt
theorem distance_wind_opposite_direction (h : height = 15) (c : wind_speed = 16) (g : gravity = 9.81) :
  let v_0 := real.sqrt (2 * h * g) in
  v_0 * (2 * v_0 / g) - c * (2 * v_0 / g) = 4 :=
by sorry

end wind_horizontal_displacement_max_horizontal_distance_calm_distance_wind_in_direction_distance_wind_opposite_direction_l364_364300


namespace apple_bags_l364_364151

theorem apple_bags (n : ℕ) (h₁ : n ≥ 70) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 := 
sorry

end apple_bags_l364_364151


namespace sum_solutions_bound_l364_364020

theorem sum_solutions_bound :
  let S := ∑ x in {x : ℝ | x > 0 ∧ x ^ (3 ^ (real.sqrt 3)) = (real.sqrt 3) ^ (3 ^ x)} in
  2 ≤ S ∧ S < 6 :=
by {
  sorry
}

end sum_solutions_bound_l364_364020


namespace exists_twelve_distinct_x_l364_364447

theorem exists_twelve_distinct_x :
  ∃ S : Set ℝ, (S.card = 12) ∧ (∀ x ∈ S, ∃ k : ℤ, 0 ≤ k ∧ k ≤ 11 ∧ (sqrt (123 - sqrt x) = k)) :=
by
  sorry

end exists_twelve_distinct_x_l364_364447


namespace gilled_mushrooms_count_l364_364261

def mushrooms_problem (G S : ℕ) : Prop :=
  (S = 9 * G) ∧ (G + S = 30) → (G = 3)

-- The theorem statement corresponding to the problem
theorem gilled_mushrooms_count (G S : ℕ) : mushrooms_problem G S :=
by {
  sorry
}

end gilled_mushrooms_count_l364_364261


namespace arithmetic_mean_of_primes_l364_364804

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 → m ∣ n → m = n)

def arithmetic_mean (nums : List ℕ) : ℕ :=
  nums.sum / nums.length

theorem arithmetic_mean_of_primes (nums : List ℕ) (primes : List ℕ) : 
    nums = [33, 35, 37, 39, 41] ∧ primes = nums.filter is_prime →
    arithmetic_mean primes = 39 :=
by
  intro h
  let primes := [37, 41]
  sorry

end arithmetic_mean_of_primes_l364_364804


namespace fewest_printers_l364_364266

theorem fewest_printers (x y : ℕ) (h1 : 375 * x = 150 * y) : x + y = 7 :=
  sorry

end fewest_printers_l364_364266


namespace ratio_of_areas_l364_364017

theorem ratio_of_areas {A B C D E F G H : ℝ} (h1 : 0 < A) (h2 : 0 < B) 
    (h3 : 0 < C) (h4 : 0 < D) (h5 : 0 < E) (h6 : 0 < F) 
    (h7 : 0 < G) (h8 : 0 < H) (ABCD : square A B C D)
    (EFGH : centers_of_right_angle_isosceles_triangles E F G H ABCD)
    (side_length_ABCD : side_length ABCD = 4) : 
    area_ratio E F G H A B C D = (3 + 2 * real.sqrt 2) / 4 :=
sorry

end ratio_of_areas_l364_364017


namespace exists_different_colors_l364_364794

-- Considering each real number greater than 1 is colored either red or blue
-- and both colors are used.
axiom is_colored (r : ℝ) : r > 1 → (colors : Set ℝ)
axiom both_colors_used : ∃ r₁ r₂, r₁ > 1 ∧ r₂ > 1 ∧ r₁ ∈ colors ∧ r₂ ∉ colors

theorem exists_different_colors (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  (a + 1 / b ∈ colors ∧ b + 1 / a ∉ colors) ∨ (a + 1 / b ∉ colors ∧ b + 1 / a ∈ colors) :=
by
  sorry

end exists_different_colors_l364_364794


namespace number_of_smaller_cubes_with_color_pairs_l364_364269

-- Define the number of smaller cubes having both faces with specific color pairs in a given colored cube.
theorem number_of_smaller_cubes_with_color_pairs :
  ∀ (n : ℕ) (color_faces : Fin 6 → String),
  (n = 5) → (color_faces = !["red", "blue", "green", "yellow", "orange", "purple"]) →
  let total_cubes := n ^ 3 in
  total_cubes = 125 →
  let red_blue_edge_cubes := n - 2 in
  let blue_green_edge_cubes := n - 2 in
  let green_yellow_edge_cubes := n - 2 in
  red_blue_edge_cubes = 3 ∧ blue_green_edge_cubes = 3 ∧ green_yellow_edge_cubes = 3 :=
by
  intro n color_faces hn hcolor_faces htotal_cubes
  let total_cubes := n ^ 3
  have : total_cubes = 125 := htotal_cubes
  let red_blue_edge_cubes := n - 2
  let blue_green_edge_cubes := n - 2
  let green_yellow_edge_cubes := n - 2
  have h1 : red_blue_edge_cubes = 3 := by linarith
  have h2 : blue_green_edge_cubes = 3 := by linarith
  have h3 : green_yellow_edge_cubes = 3 := by linarith
  exact ⟨h1, h2, h3⟩

end number_of_smaller_cubes_with_color_pairs_l364_364269


namespace possible_apple_counts_l364_364131

theorem possible_apple_counts (n : ℕ) (h₁ : 70 ≤ n) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end possible_apple_counts_l364_364131


namespace tetrahedron_circumsphere_radius_l364_364746

theorem tetrahedron_circumsphere_radius :
  ∃ (r : ℝ), 
    (∀ (A B C P : ℝ × ℝ × ℝ),
      (dist A B = 5) ∧
      (dist A C = 5) ∧
      (dist A P = 5) ∧
      (dist B C = 5) ∧
      (dist B P = 5) ∧
      (dist C P = 6) →
      r = (20 * Real.sqrt 39) / 39) :=
sorry

end tetrahedron_circumsphere_radius_l364_364746


namespace number_of_real_solutions_l364_364422

theorem number_of_real_solutions : 
  set.countable {x : ℝ | ∃ n : ℕ, (sqrt (123 - sqrt x) = n)} = 12 :=
by
  sorry

end number_of_real_solutions_l364_364422


namespace dice_product_composite_probability_l364_364476

theorem dice_product_composite_probability :
  (let total_outcomes := (6 : ℕ)^4,
       non_composite_outcomes := 1,
       composite_probability := 1 - (non_composite_outcomes / total_outcomes) in
   composite_probability = 1295 / 1296) :=
by sorry

end dice_product_composite_probability_l364_364476


namespace fraction_meaningfulness_l364_364226

def fraction_is_meaningful (x : ℝ) : Prop :=
  x ≠ 3 / 2

theorem fraction_meaningfulness (x : ℝ) : 
  (2 * x - 3) ≠ 0 ↔ fraction_is_meaningful x :=
by
  sorry

end fraction_meaningfulness_l364_364226


namespace minimum_value_xy_l364_364856

theorem minimum_value_xy (x y : ℝ) (h : (x + Real.sqrt (x^2 + 1)) * (y + Real.sqrt (y^2 + 1)) ≥ 1) : x + y ≥ 0 :=
sorry

end minimum_value_xy_l364_364856


namespace reflection_of_point_l364_364606

def reflect_across_y_neg_x (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P
  (y, x)

theorem reflection_of_point
  (P : ℝ × ℝ)
  (h : P = (8, -3)) :
  reflect_across_y_neg_x P = (3, -8) :=
by
  rw [h]
  sorry

end reflection_of_point_l364_364606


namespace number_of_registration_methods_l364_364193

theorem number_of_registration_methods :
  let students := {A, B, C, D}
  let interest_groups := {painting, chess, basketball}
  ∃ f : students → interest_groups, 
    (∀ g ∈ interest_groups, ∃ s ∈ students, f s = g) ∧ 
    (f '' students).card = 36 :=
sorry

end number_of_registration_methods_l364_364193


namespace shaded_fraction_of_equilateral_triangle_l364_364914

theorem shaded_fraction_of_equilateral_triangle 
  (ABC : Type) [triangle ABC] 
  (equilateral : is_equilateral ABC)
  (divided_into_9 : is_divided_into_9_smaller_equilateral_triangles ABC) 
  (side_length_1 : ∀ t ∈ smaller_triangles ABC, side_length t = 1) 
  (shaded_area : area_of_shaded_region ABC = 1 / 2 * 2 * (sqrt 3 / 4))
  : shaded_fraction ABC = 2 / 9 := sorry

end shaded_fraction_of_equilateral_triangle_l364_364914


namespace Catriona_goldfish_count_l364_364318

theorem Catriona_goldfish_count (G : ℕ) (A : ℕ) (U : ℕ) 
    (h1 : A = G + 4) 
    (h2 : U = 2 * A) 
    (h3 : G + A + U = 44) : G = 8 :=
by
  -- Proof goes here
  sorry

end Catriona_goldfish_count_l364_364318


namespace find_a5_div_a7_l364_364925

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- The sequence {aₙ} is a positive geometric sequence.
axiom geo_seq (n : ℕ) : a (n + 1) = a n * q
axiom pos_seq (n : ℕ) : 0 < a n

-- Given conditions
axiom a2a8_eq_6 : a 2 * a 8 = 6
axiom a4_plus_a6_eq_5 : a 4 + a 6 = 5
axiom decreasing_seq (n : ℕ) : a (n + 1) < a n

theorem find_a5_div_a7 : a 5 / a 7 = 3 / 2 := 
sorry

end find_a5_div_a7_l364_364925


namespace lcm_triples_count_l364_364404

theorem lcm_triples_count :
  let P (x y z : ℕ) := Nat.lcm x y = 120 ∧ Nat.lcm x z = 450 ∧ Nat.lcm y z = 600
  in (Finset.univ.filter (λ xyz : ℕ × ℕ × ℕ, P xyz.1 xyz.2.1 xyz.2.2)).card = 3 :=
by
  sorry

end lcm_triples_count_l364_364404


namespace converse_true_inverse_true_contrapositive_false_sufficiency_necessity_l364_364057

-- Define the original proposition with conditions
def prop : Prop := ∀ (m n : ℝ), m ≤ 0 ∨ n ≤ 0 → m + n ≤ 0

-- Identify converse, inverse, and contrapositive
def converse : Prop := ∀ (m n : ℝ), m + n ≤ 0 → m ≤ 0 ∨ n ≤ 0
def inverse : Prop := ∀ (m n : ℝ), m > 0 ∧ n > 0 → m + n > 0
def contrapositive : Prop := ∀ (m n : ℝ), m + n > 0 → m > 0 ∧ n > 0

-- Identifying the conditions of sufficiency and necessity
def necessary_but_not_sufficient (p q : Prop) : Prop := 
  (¬p → ¬q) ∧ (q → p) ∧ ¬(p → q)

-- Prove or provide the statements
theorem converse_true : converse := sorry
theorem inverse_true : inverse := sorry
theorem contrapositive_false : ¬contrapositive := sorry
theorem sufficiency_necessity : necessary_but_not_sufficient 
  (∀ (m n : ℝ), m ≤ 0 ∨ n ≤ 0) 
  (∀ (m n : ℝ), m + n ≤ 0) := sorry

end converse_true_inverse_true_contrapositive_false_sufficiency_necessity_l364_364057


namespace probability_composite_l364_364497

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def non_composite_outcomes : ℕ := 1 + 12

theorem probability_composite :
  let total_outcomes := 6^4 in
  let non_composite := non_composite_outcomes in
  let composite_probability := (total_outcomes - non_composite) / total_outcomes.toRat in
  composite_probability = 1283 / 1296 := by
  let total_outcomes := 6^4
  let non_composite := 1 + 12
  let composite_probability := (total_outcomes - non_composite).toRat / total_outcomes.toRat
  sorry

end probability_composite_l364_364497


namespace min_k_to_fit_ratio_condition_l364_364754

theorem min_k_to_fit_ratio_condition :
  ∃ k, ∀ (s : Finset ℕ), (s ⊆ Finset.range 100) → s.card = k →
    ∃ (a b : ℕ), a ∈ s ∧ b ∈ s ∧ a ≠ b ∧ (1/2 : ℝ) ≤ b.toReal / a ∧ b.toReal / a ≤ 2 :=
begin
  let k := 7,
  use k,
  intros s hs hs_card,
  sorry,
end

end min_k_to_fit_ratio_condition_l364_364754


namespace monotonicity_and_extrema_l364_364865

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x + 3) + x^2

theorem monotonicity_and_extrema :
  (∀ x, -3 / 2 < x ∧ x < -1 → f x < f (x + 0.0001)) ∧
  (∀ x, -1 < x ∧ x < -1 / 2 → f x > f (x + 0.0001)) ∧
  (∀ x, -1 / 2 < x ∧ x < (Real.exp 2 - 3) / 2 → f x < f (x + 0.0001)) ∧
  ∀ x, x ∈ Set.Icc (-1 : ℝ) ((Real.exp 2 - 3) / 2) →
     (f (x) ≥ Real.log 2 + 1 / 4 → x = -1 / 2) ∧
     (f (x) ≤ 2 + (Real.exp 2 - 3)^2 / 4 → x = (Real.exp 2 - 3) / 2) :=
sorry

end monotonicity_and_extrema_l364_364865


namespace complex_magnitude_addition_l364_364796

theorem complex_magnitude_addition :
  (Complex.abs (3 / 4 - 3 * Complex.I) + 5 / 12) = (9 * Real.sqrt 17 + 5) / 12 := 
  sorry

end complex_magnitude_addition_l364_364796


namespace angle_covering_l364_364016

theorem angle_covering (n : ℕ) (hn : n > 0) (points : fin n → ℝ × ℝ) :
  ∃ (angles : fin n → set (ℝ × ℝ)), (∀ i : fin n, ∃ θ : ℝ, angles i = {p | ∠ (points i) p < θ + π} ∧ θ = 2 * π / n) ∧ (∀ p : ℝ × ℝ, ∃ i : fin n, p ∈ angles i) := by
  sorry

end angle_covering_l364_364016


namespace y_simplified_y_at_three_l364_364328

variable (x : ℝ)

def y (x : ℝ) := Real.sqrt (x^2 - 4*x + 4) + Real.sqrt (x^2 + 4*x + 4)

theorem y_simplified (x : ℝ) : y x = |x-2| + |x+2| :=
by
  unfold y
  sorry

theorem y_at_three : y 3 = 6 :=
by
  have simplification : y 3 = |3-2| + |3+2| := by
    exact y_simplified 3
  rw [simplification]
  norm_num

end y_simplified_y_at_three_l364_364328


namespace number_of_non_empty_subsets_of_P_l364_364040

noncomputable def P : Set ℝ := {x | ∃ (t : ℝ), x > 0 ∧ ∫ u in 0..t, (3*u^2 - 10*u + 6) = 0}

theorem number_of_non_empty_subsets_of_P : ∃ n : ℕ, n = 3 ∧ (n = 2^(P.to_finset.card) - 1) :=
by
  sorry

end number_of_non_empty_subsets_of_P_l364_364040


namespace distance_between_parallel_lines_l364_364618

-- Defining the equations of the two parallel lines
def l1 (x y : ℝ) : Prop := x - y + 1 = 0
def l2 (x y : ℝ) : Prop := x - y + 3 = 0

-- Definition and statement of the problem
theorem distance_between_parallel_lines :
  let A := 1
  let B := -1
  let C1 := 1
  let C2 := 3 in
  (abs (C2 - C1)) / (real.sqrt (A^2 + B^2)) = real.sqrt 2 :=
by sorry

end distance_between_parallel_lines_l364_364618


namespace max_profit_l364_364727

-- Definitions based on given conditions
def P (x : ℝ) : ℝ := (x + 2) / 4
def cost (P : ℝ) : ℝ := 6 * (P + 1 / P)
def price (P : ℝ) : ℝ := 4 + 20 / P
def profit (x : ℝ) : ℝ := (price (P x)) * (P x) - x - cost (P x)

-- The main theorem to be proven
theorem max_profit (a : ℝ) (a_positive : 0 < a) :
  (0 ≤ x → x ≤ a → profit x ≤ profit 2) ∧
  (a < 2 → ∃ x, x = a ∧ profit x = 19 - 24 / (a + 2) - 3 / 2 * a) :=
by
  sorry

end max_profit_l364_364727


namespace fraction_left_handed_non_throwers_l364_364048

theorem fraction_left_handed_non_throwers 
    (total_players : ℕ) (throwers : ℕ) (total_right_handed : ℕ) (all_throwers_right_handed : true)
    (non_throwers : ℕ := total_players - throwers)
    (right_handed_non_throwers : ℕ := total_right_handed - throwers)
    (left_handed_non_throwers : ℕ := non_throwers - right_handed_non_throwers) :
    total_players = 61 →
    throwers = 37 →
    total_right_handed = 53 →
    fraction_left_handed := (left_handed_non_throwers : ℚ) / non_throwers =
    1 / 3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end fraction_left_handed_non_throwers_l364_364048


namespace reflection_of_point_l364_364608

def reflect_across_y_neg_x (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P
  (y, x)

theorem reflection_of_point
  (P : ℝ × ℝ)
  (h : P = (8, -3)) :
  reflect_across_y_neg_x P = (3, -8) :=
by
  rw [h]
  sorry

end reflection_of_point_l364_364608


namespace coeff_x7_expansion_l364_364212

theorem coeff_x7_expansion : 
  let expansion := (∑ k in Finset.range 11, (Nat.choose 10 k) * x^(10 - k) * (-2)^k)
  ∃ coeff : ℤ, 
  (coeff * x^7 ∈ expansion) ∧ coeff = -960 :=
begin
  let expansion := (∑ k in Finset.range 11, (Nat.choose 10 k) * x^(10 - k) * (-2)^k),
  use -960,
  split,
  { sorry, },
  { reflexivity, }
end

end coeff_x7_expansion_l364_364212


namespace incorrect_option_D_l364_364681

-- Conditions
variables {a b c t : ℝ}
axiom h1 : a ≠ 0
axiom h2 : a + b + c = 4
axiom h3 : c = 3
axiom h4 : a - b + c = 0

-- Prove that option D is incorrect with the given conditions.
theorem incorrect_option_D : ¬ ∀ p : ℝ, ( ∀ x : ℝ, -x^2 + 2*x + 3 < 2*x + p ) → p ≥ 3 :=
begin
  sorry
end

end incorrect_option_D_l364_364681


namespace quadratic_difference_sum_l364_364768

theorem quadratic_difference_sum :
  let a := 2
  let b := -10
  let c := 3
  let Δ := b * b - 4 * a * c
  let root1 := (10 + Real.sqrt Δ) / (2 * a)
  let root2 := (10 - Real.sqrt Δ) / (2 * a)
  let diff := root1 - root2
  let m := 19  -- from the difference calculation
  let n := 1   -- from the simplified form
  m + n = 20 :=
by
  -- Placeholders for calculation and proof steps.
  sorry

end quadratic_difference_sum_l364_364768


namespace initial_population_l364_364719

theorem initial_population (P : ℝ) (h : 0.72 * P = 3168) : P = 4400 :=
sorry

end initial_population_l364_364719


namespace length_of_de_l364_364597

-- Definitions based on the given conditions
def base_ab : ℝ := 16
def height_above_ab (h : ℝ) : ℝ := h
def area_abc (h : ℝ) : ℝ := 1/2 * base_ab * h
def area_shaded (h : ℝ) : ℝ := 0.16 * area_abc h
def height_below_ab (h : ℝ) : ℝ := 0.4 * h
def height_to_de (h : ℝ) : ℝ := height_above_ab h - height_below_ab h
def length_de (h : ℝ) : ℝ := (height_to_de h / h) * base_ab

-- Statement to be proved
theorem length_of_de (h : ℝ) (h_pos : 0 < h) :
  length_de h = 9.6 :=
sorry

end length_of_de_l364_364597


namespace coeff_x7_in_expansion_l364_364215

-- Each definition in Lean 4 statement reflects the conditions of the problem.
def binomial_coefficient (n k : ℕ) : ℕ := nat.choose n k

-- The condition for expansion using Binomial Theorem
def binomial_expansion_term (n k : ℕ) (a x : ℤ) : ℤ :=
  binomial_coefficient n k * a ^ (n - k) * x ^ k

-- Prove that the coefficient of x^7 in the expansion of (x - 2)^{10} is -960
theorem coeff_x7_in_expansion : 
  binomial_coefficient 10 3 * (-2) ^ 3 = -960 := 
sorry

end coeff_x7_in_expansion_l364_364215


namespace Johann_oranges_l364_364537

-- Define the given conditions
def initial_oranges := 60
def eaten_oranges := 10
def half_remaining_oranges := (initial_oranges - eaten_oranges) / 2
def returned_oranges := 5

-- Define the statement to prove
theorem Johann_oranges :
  initial_oranges - eaten_oranges - half_remaining_oranges + returned_oranges = 30 := by
  sorry

end Johann_oranges_l364_364537


namespace find_probability_l364_364039

variable (p : ℚ) [fact (0 < p)] [fact (p < 1)]
variable (ξ : ℕ → Prop) (η : ℕ → Prop)
variable (B_2_p : ∀ n : ℕ, ξ n ↔ (n ≤ 2 ∧ ∃ k : ℕ, (nat.choose 2 k) * p^k * (1-p)^(2-k) = 1))
variable (B_4_p : ∀ n : ℕ, η n ↔ (n ≤ 4 ∧ ∃ k : ℕ, (nat.choose 4 k) * p^k * (1-p)^(4-k) = 1))
variable (P : (ℕ → Prop) → ℚ)
variable (P_xi_ge_1 : P (λ n, 1 ≤ n ∧ ξ n) = 5/9)
variable (P_eta_ge_2 : P (λ n, 2 ≤ n ∧ η n) = 11/27)

theorem find_probability :
  P (λ n, 2 ≤ n ∧ η n) = 11/27 := sorry

end find_probability_l364_364039


namespace jenny_correct_number_l364_364943

theorem jenny_correct_number (x : ℤ) (h : x - 26 = -14) : x + 26 = 38 :=
by
  sorry

end jenny_correct_number_l364_364943


namespace find_sum_of_money_l364_364233

theorem find_sum_of_money (P : ℝ) (H1 : P * 0.18 * 2 - P * 0.12 * 2 = 840) : P = 7000 :=
by
  sorry

end find_sum_of_money_l364_364233


namespace sum_of_arithmetic_progressions_l364_364815

def arithmetic_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_of_arithmetic_progressions : (∑ p in Finset.range 11, arithmetic_sum p (2 * p) 50) = 137500 :=
by
  sorry

end sum_of_arithmetic_progressions_l364_364815


namespace greatest_three_digit_multiple_of_23_l364_364668

theorem greatest_three_digit_multiple_of_23 : ∃ n : ℕ, n % 23 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℕ, m % 23 = 0 ∧ 100 ≤ m ∧ m ≤ 999 → m ≤ n := 
by
  use 989
  split
  · -- 989 is a multiple of 23
    exact (by norm_num : 989 % 23 = 0)
  · split
    · -- 989 is at least 100
      exact (by norm_num : 100 ≤ 989)
    · split
      · -- 989 is at most 999
        exact (by norm_num : 989 ≤ 999)
      · -- 989 is the greatest such number within the range
        sorry

end greatest_three_digit_multiple_of_23_l364_364668


namespace apple_bags_l364_364153

theorem apple_bags (n : ℕ) (h₁ : n ≥ 70) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 := 
sorry

end apple_bags_l364_364153


namespace total_area_covered_is_60_l364_364647

-- Declare the dimensions of the strips
def length_strip : ℕ := 12
def width_strip : ℕ := 2
def num_strips : ℕ := 3

-- Define the total area covered without overlaps
def total_area_no_overlap := num_strips * (length_strip * width_strip)

-- Define the area of overlap for each pair of strips
def overlap_area_per_pair := width_strip * width_strip

-- Define the total overlap area given 3 pairs
def total_overlap_area := 3 * overlap_area_per_pair

-- Define the actual total covered area
def total_covered_area := total_area_no_overlap - total_overlap_area

-- Prove that the total covered area is 60 square units
theorem total_area_covered_is_60 : total_covered_area = 60 := by 
  sorry

end total_area_covered_is_60_l364_364647


namespace coefficient_x_pow_7_l364_364209

theorem coefficient_x_pow_7 :
  ∃ c : Int, (∀ k : Nat, (x : Real) → coeff (expand_binom (x-2) 10) k = c → k = 7 ∧ c = -960) :=
by
  sorry

end coefficient_x_pow_7_l364_364209


namespace even_sum_exactly_one_even_l364_364652

theorem even_sum_exactly_one_even (a b c : ℕ) (h : (a + b + c) % 2 = 0) :
  (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0) :=
by
  sorry

end even_sum_exactly_one_even_l364_364652


namespace standard_equation_of_ellipse_standard_equation_of_circle_product_length_range_l364_364923

open Real

noncomputable def eccentricity (a b : ℝ) : ℝ := sqrt ((a ^ 2 - b ^ 2) / a ^ 2)

def ellipse_eq (a b : ℝ) : Prop := a > b ∧ b > 0 ∧ eccentricity a b = sqrt 3 / 2

def chord_length (a b : ℝ) : Prop := ((a ^ 2 + b ^ 2 = 5) → sqrt(5))

theorem standard_equation_of_ellipse (a b : ℝ) (h₁ : ellipse_eq a b) (h₂ : chord_length a b) : 
  (a = 2) ∧ (b = 1) → (∀ x y : ℝ, (x ^ 2 / 4 + y ^ 2 = 1)) :=
sorry

def tangent_to_circle (a b : ℝ) (x y : ℝ) : Prop := 
  ((x = 3) ∧ (y = 2)) ∧ (a * x + b * y - 2 = 0) 

theorem standard_equation_of_circle (x y : ℝ) (m : ℝ) (h₃ : tangent_to_circle 1 2 x y) : (∀ x y : ℝ, ((x - 3)^2 + (y - 2)^2 = 5)) :=
sorry

def range_of_product (k : ℝ) : Prop := 
  0 ≤ k ^ 2 ∧ k ^ 2 < (1 / 5)

noncomputable def lengths_product (k : ℝ) : ℝ := 8 * sqrt (1 - 25 * k ^ 4) / (1 + 4 * k ^ 2) ^ 2

theorem product_length_range (k : ℝ) (h₄ : range_of_product k) : ∀ t : ℝ, 0 < lengths_product k ∧ lengths_product k ≤ 8 :=
sorry

end standard_equation_of_ellipse_standard_equation_of_circle_product_length_range_l364_364923


namespace num_real_satisfying_x_l364_364440

theorem num_real_satisfying_x : 
  (∃ (x : ℝ), ∃ (s : ℕ) (hs : s ∈ set.Ico 0 (floor (sqrt 123) + 1)), 
   s = int.of_nat (floor (sqrt (123 - sqrt x)))) →
   12 := sorry

end num_real_satisfying_x_l364_364440


namespace factorial_div_l364_364899

theorem factorial_div (p : ℕ) : (p^2)! ∣ (p!)^(p+1) :=
by
  sorry

end factorial_div_l364_364899


namespace find_k_l364_364357

theorem find_k (k α β : ℝ)
  (h1 : (∀ x : ℝ, x^2 - (k-1) * x - 3*k - 2 = 0 → x = α ∨ x = β))
  (h2 : α^2 + β^2 = 17) :
  k = 2 :=
sorry

end find_k_l364_364357


namespace good_fortune_probability_l364_364569

noncomputable def probability_of_good_fortune : Real :=
  let α_range := Icc 0 Real.pi
  let β_range := Icc 0 (2 * Real.pi)
  let region_G := α_range.prod β_range
  let f (α β : Real) := (0 ≤ β ∧ β < (Real.pi/2 - α/2)) ∧ (0 ≤ α ∧ α < (Real.pi/2 - β/2))
  let region_F := { p : Real × Real | f p.fst p.snd }
  let area_region_G := 2 * Real.pi^2
  let area_region_F := area_region_G / 12
  let P_F := area_region_F / area_region_G
  1 - P_F

theorem good_fortune_probability : probability_of_good_fortune = 11 / 12 :=
sorry

end good_fortune_probability_l364_364569


namespace probability_composite_l364_364493

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def non_composite_outcomes : ℕ := 1 + 12

theorem probability_composite :
  let total_outcomes := 6^4 in
  let non_composite := non_composite_outcomes in
  let composite_probability := (total_outcomes - non_composite) / total_outcomes.toRat in
  composite_probability = 1283 / 1296 := by
  let total_outcomes := 6^4
  let non_composite := 1 + 12
  let composite_probability := (total_outcomes - non_composite).toRat / total_outcomes.toRat
  sorry

end probability_composite_l364_364493


namespace unit_cubes_fill_box_l364_364651

theorem unit_cubes_fill_box (p : ℕ) (hp : Nat.Prime p) :
  let length := p
  let width := 2 * p
  let height := 3 * p
  length * width * height = 6 * p^3 :=
by
  -- Proof here
  sorry

end unit_cubes_fill_box_l364_364651


namespace intersections_form_trapezoid_l364_364777

noncomputable def pointE : ℝ × ℝ := (0, 0)
noncomputable def pointF : ℝ × ℝ := (0, 5)
noncomputable def pointG : ℝ × ℝ := (8, 5)
noncomputable def pointH : ℝ × ℝ := (8, 0)

-- Lines equations:
noncomputable def line_E_45 (x : ℝ) : ℝ := x
noncomputable def line_E_75 (x : ℝ) : ℝ := 3.732 * x
noncomputable def line_F_neg45 (x : ℝ) : ℝ := 5 - x
noncomputable def line_F_neg75 (x : ℝ) : ℝ := 5 - 3.732 * x

-- Intersection points:
noncomputable def intersection_45_neg45 : ℝ × ℝ := (2.5, 2.5)
noncomputable def intersection_75_neg75 : ℝ × ℝ := (0.67, 2.5)

theorem intersections_form_trapezoid :
  let I1 := intersection_45_neg45
  let I2 := intersection_75_neg75
  let shape_vertices := {pointE, pointF, I1, I2}
  shape vertices = Trapezoid := sorry

end intersections_form_trapezoid_l364_364777


namespace minimum_discount_l364_364276

variable (C P : ℝ) (r x : ℝ)

def microwave_conditions := 
  C = 1000 ∧ 
  P = 1500 ∧ 
  r = 0.02 ∧ 
  P * (x / 10) ≥ C * (1 + r)

theorem minimum_discount : ∃ x, microwave_conditions C P r x ∧ x ≥ 6.8 :=
by 
  sorry

end minimum_discount_l364_364276


namespace dice_product_composite_probability_l364_364486

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

-- This function calculates the probability of an event occurring by counting the favorable and total outcomes.
def probability (favorable total : ℕ) : ℚ :=
  favorable / total

noncomputable def probability_of_composite_product : ℚ :=
  probability 1283 1296

theorem dice_product_composite_probability : probability_of_composite_product = 1283 / 1296 := sorry

end dice_product_composite_probability_l364_364486


namespace least_possible_value_of_squares_l364_364634

theorem least_possible_value_of_squares (a b x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h1 : 15 * a + 16 * b = x^2) (h2 : 16 * a - 15 * b = y^2) : 
  ∃ (x : ℕ) (y : ℕ), min (x^2) (y^2) = 231361 := 
sorry

end least_possible_value_of_squares_l364_364634


namespace sufficient_but_not_necessary_condition_for_odd_function_l364_364905

def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f (-x) = -f x

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := 2^x - (k^2 - 3) * 2^(-x)

theorem sufficient_but_not_necessary_condition_for_odd_function (k : ℝ) :
  (is_odd_function (f k)) ↔ (k^2 - 3 = 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_odd_function_l364_364905


namespace max_Xs_without_lines_of_3_l364_364776

theorem max_Xs_without_lines_of_3 (grid : Matrix (Fin 5) (Fin 5) Bool) :
  (∀ i, ∑ j, if grid i j then 1 else 0 ≤ 2) ∧
  (∀ j, ∑ i, if grid i j then 1 else 0 ≤ 2) ∧
  (∀ diag, ∑ k, if grid k (diag - k) then 1 else 0) ≤ 2 ∧
  (∀ diag, ∑ k, if grid k (diag - k - 5) then 1 else 0) ≤ 2 →
  ∑ i j, if grid i j then 1 else 0 ≤ 11 :=
sorry

end max_Xs_without_lines_of_3_l364_364776


namespace integral_curve_exists_l364_364343

noncomputable theory

def integral_curve (y : ℝ → ℝ) : Prop :=
  ∀ x, y'' x = x + 1

def passes_through (y : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  y p.1 = p.2

def tangent_at (y : ℝ → ℝ) (y_tangent : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  y p.1 = p.2 ∧ y' p.1 = y_tangent' p.1

theorem integral_curve_exists :
  ∃ y : ℝ → ℝ,
    integral_curve y ∧
    passes_through y (1, 1) ∧
    tangent_at y (λ x, (1/2) * x + (1/2)) (1, 1) ∧
    ∀ x, y x = x^3 / 6 + x^2 / 2 - x + 4 / 3 :=
by
  sorry

end integral_curve_exists_l364_364343


namespace probability_of_composite_l364_364492

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ m < n ∧ 1 < k ∧ k < n ∧ m * k = n

def dice_outcomes (faces : ℕ) (rolls : ℕ) : ℕ :=
  faces ^ rolls

def non_composite_product_ways : ℕ :=
  1 + (3 * 4)  -- one way for all 1s, plus combinations of (1,1,1,{2,3,5})

def total_outcomes : ℕ :=
  dice_outcomes 6 4  -- 6^4 total possible outcomes

def probability_composite : ℚ :=
  1 - (non_composite_product_ways / total_outcomes)

theorem probability_of_composite:
  probability_composite = 1283 / 1296 := 
by
  sorry

end probability_of_composite_l364_364492


namespace chord_length_by_circle_l364_364356

noncomputable def line_distance (a b c : ℝ) : ℝ := 
  abs c / (real.sqrt (a^2 + b^2))

theorem chord_length_by_circle (r d : ℝ) (h : r² = (d/2)² + (sqrt 13 / 13)²) : 
  (line_distance 2 (-3) (-1) = sqrt 13 / 13) :=
by
  rw [line_distance]
  sorry

end chord_length_by_circle_l364_364356


namespace exists_twelve_distinct_x_l364_364449

theorem exists_twelve_distinct_x :
  ∃ S : Set ℝ, (S.card = 12) ∧ (∀ x ∈ S, ∃ k : ℤ, 0 ≤ k ∧ k ≤ 11 ∧ (sqrt (123 - sqrt x) = k)) :=
by
  sorry

end exists_twelve_distinct_x_l364_364449


namespace num_supermarkets_in_US_l364_364119

theorem num_supermarkets_in_US (U C : ℕ) (h1 : U + C = 420) (h2 : U = C + 56) : U = 238 :=
by
  sorry

end num_supermarkets_in_US_l364_364119


namespace arithmetic_sum_problem_l364_364359

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a 1 + a n)) / 2

theorem arithmetic_sum_problem (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h_arith_seq : arithmetic_sequence a)
  (h_S_def : ∀ n : ℕ, S n = sum_of_first_n_terms a n)
  (h_S13 : S 13 = 52) : a 4 + a 8 + a 9 = 12 :=
sorry

end arithmetic_sum_problem_l364_364359


namespace arithmetic_mean_prime_numbers_l364_364806

-- Define the list of numbers
def num_list := [33, 35, 37, 39, 41]

-- Define a predicate to check for prime numbers
def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

-- Filter the prime numbers from the list
def prime_numbers (l : List ℕ) : List ℕ := l.filter is_prime

-- Compute the arithmetic mean of a list of numbers
def arithmetic_mean (l : List ℕ) : ℕ := l.sum / l.length

-- Target theorem
theorem arithmetic_mean_prime_numbers :
  arithmetic_mean (prime_numbers num_list) = 39 := by
  sorry

end arithmetic_mean_prime_numbers_l364_364806


namespace final_student_score_l364_364972

/-- Mrs. Jackson teaches a class of 20 students. After grading 19 of the tests, the average 
  score was 75. When the final student's test was graded, the average score of all 20 students 
  increased to 78. Prove that the final student's test score was 135. -/
theorem final_student_score
  (n : ℕ) 
  (k1 : ℕ) 
  (k2 : ℕ)
  (average_19 : ℤ) 
  (average_20 : ℤ) 
  (h_n : n = 20)
  (h_k1 : k1 = 19)
  (h_average_19 : average_19 = 75)
  (h_average_20 : average_20 = 78)
  : ∃ x : ℤ, x = 135 := 
by 
  have h1 : k1 * average_19 = 1425 := by symm;
    exact (calc
      k1 * average_19 = 19 * 75 := by 
        rw [h_k1, h_average_19])
  have h2 : n * average_20 = 1560 := by symm;
    exact (calc
      n * average_20 = 20 * 78 := by 
        rw [h_n, h_average_20])
  have h3 := h2 - h1
  use 135
  calc
    135 = 1560 - 1425 := by norm_num
    ... = x := by assumption

end final_student_score_l364_364972


namespace AB_eq_BC_l364_364242

structure Triangle (α : Type*) [Real] :=
(A B C : α)

def is_median {α : Type*} [Real] (A B C D : α) : Prop :=
is_midpoint (B, C) D

def is_midpoint {α : Type*} [Real] (p1 p2 : α × α) (D : α) : Prop :=
(p1.1 + p2.1) / 2 = D ∧ (p1.2 + p2.2) / 2 = D

def angle_eq_30 {α : Type*} [Real] (A C D : α) (B C E : α) : Prop :=
(angle A C D) = 30 ∧ (angle B C E) = 30

theorem AB_eq_BC {α : Type*} [Real] (ABC : Triangle α) (D E : α) (A B C : α):
  (is_median A B C D) → (is_median B C A E) → angle_eq_30 A C D B C E → 
  AB = BC := 
sorry

end AB_eq_BC_l364_364242


namespace part_a_l364_364245

theorem part_a (p : ℕ) (hp : Nat.Prime p) (odd_p : p % 2 = 1) :
  ∃ (m : ℕ) (x1 x2 x3 x4 : ℤ), m < p ∧ (x1^2 + x2^2 + x3^2 + x4^2 = m * p) :=
sorry

end part_a_l364_364245


namespace a_is_3_times_faster_than_b_l364_364270

-- Define the rates of A and B in terms of their capacities (work per day)
def rateA : ℝ := 1 / 20
def rateB : ℝ := 1 / 15 - rateA

-- Define the condition we're trying to prove: A is 3 times faster than B
theorem a_is_3_times_faster_than_b : rateA / rateB = 3 := 
by
  sorry

end a_is_3_times_faster_than_b_l364_364270


namespace johann_oranges_problem_l364_364530

theorem johann_oranges_problem :
  ∀ (initial_oranges johann_ate half_stolen carson_returned : ℕ),
  initial_oranges = 60 →
  johann_ate = 10 →
  half_stolen = (initial_oranges - johann_ate) / 2 →
  carson_returned = 5 →
  initial_oranges - johann_ate - half_stolen + carson_returned = 30 :=
begin
  intros initial_oranges johann_ate half_stolen carson_returned,
  sorry
end

end johann_oranges_problem_l364_364530


namespace arc_intersects_l364_364957

variables {P1 P2 P3 : set (ℝ × ℝ)}
variables {P : set (ℝ × ℝ)}

-- Assume P1, P2, P3 are arcs with two common endpoints and no other intersections.
axiom arcs_disjoint (h12 : P1 ∩ P2 = ∅) (h23 : P2 ∩ P3 = ∅) (h13 : P1 ∩ P3 = ∅) : 
  ∀ x, x ∈ P1 → x ∈ P3 → ∃ y, y ∈ P2

-- Assume the complement of P1, P2, and P3 consists of three regions with specific boundaries.
axiom regions_bounded (R1 R2 R3 : set (ℝ × ℝ))
  (hR1 : R1 = (ℝ × ℝ) \ (P1 ∪ P2))
  (hR2 : R2 = (ℝ × ℝ) \ (P2 ∪ P3))
  (hR3 : R3 = (ℝ × ℝ) \ (P1 ∪ P3)) : 
  (ℝ × ℝ) \ (P1 ∪ P2 ∪ P3) = R1 ∪ R2 ∪ R3

-- Assume P is an arc between a point in the interior of P1 and a point in the interior of P3,
-- and it lies within the region including the interior of P2.
axiom arc_interior (int_P1 int_P2 int_P3 int_P : set (ℝ × ℝ))
  (hP1_int : int_P1 ⊂ P1) (hP2_int : int_P2 ⊂ P2) (hP3_int : int_P3 ⊂ P3) (hP_int : int_P ⊂ P) 
  (hP : ∀ x, x ∈ int_P → int_P1 ∩ int_P3 = ∅) : 
  int_P ⊂ (ℝ × ℝ) \ (P1 ∪ P3) → int_P2 ∩ int_P ≠ ∅

theorem arc_intersects {P1 P2 P3 P : set (ℝ × ℝ)}
  (R1 R2 R3 : set (ℝ × ℝ))
  (int_P1 int_P2 int_P3 int_P : set (ℝ × ℝ))
  (h12 : P1 ∩ P2 = ∅) (h23 : P2 ∩ P3 = ∅) (h13 : P1 ∩ P3 = ∅)
  (hR1 : R1 = (ℝ × ℝ) \ (P1 ∪ P2)) (hR2 : R2 = (ℝ × ℝ) \ (P2 ∪ P3)) (hR3 : R3 = (ℝ × ℝ) \ (P1 ∪ P3))
  (hP1_int : int_P1 ⊂ P1) (hP2_int : int_P2 ⊂ P2) (hP3_int : int_P3 ⊂ P3) (hP_int : int_P ⊂ P)
  (hP : ∀ x, x ∈ int_P → int_P1 ∩ int_P3 = ∅)
  (hP_region : int_P ⊂ (ℝ × ℝ) \ (P1 ∪ P3)) :
  int_P2 ∩ int_P ≠ ∅ := sorry

end arc_intersects_l364_364957


namespace apple_count_l364_364142

/--
Given the total number of apples in several bags, where each bag contains either 12 or 6 apples,
and knowing that the total number of apples is between 70 and 80 inclusive,
prove that the total number of apples can only be 72 or 78.
-/
theorem apple_count (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : ∃ k : ℕ, n = 6 * k) : n = 72 ∨ n = 78 :=
by {
  sorry
}

end apple_count_l364_364142


namespace dice_product_composite_probability_l364_364477

theorem dice_product_composite_probability :
  (let total_outcomes := (6 : ℕ)^4,
       non_composite_outcomes := 1,
       composite_probability := 1 - (non_composite_outcomes / total_outcomes) in
   composite_probability = 1295 / 1296) :=
by sorry

end dice_product_composite_probability_l364_364477


namespace max_lambda_leq_64_div_27_l364_364033

theorem max_lambda_leq_64_div_27 (a b c : ℝ) (ha : 0 < a ∧ a ≤ 1) (hb : 0 < b ∧ b ≤ 1) (hc : 0 < c ∧ c ≤ 1) :
  (1:ℝ) + (64:ℝ) / (27:ℝ) * (1 - a) * (1 - b) * (1 - c) ≤ Real.sqrt 3 / Real.sqrt (a + b + c) := 
sorry

end max_lambda_leq_64_div_27_l364_364033


namespace unique_intersection_iff_k_l364_364396

noncomputable section

variable {a : ℝ}
variable {k : ℝ}

def P := { p : ℝ × ℝ | p.snd = k }
def Q := { q : ℝ × ℝ | ∃ x : ℝ, q.snd = a^x + 1 ∧ a > 0 ∧ a ≠ 1 }

theorem unique_intersection_iff_k:
  (∃! p : ℝ × ℝ, p ∈ P ∧ p ∈ Q) ↔ k ≤ 1 :=
by sorry

end unique_intersection_iff_k_l364_364396


namespace cot_squared_sum_inequality_l364_364545

variable (α β γ : ℝ) (R p : ℝ)
variable (isTriangle : α + β + γ = π)
variable (perimeterCondition : α + β + γ = 2 * p)

theorem cot_squared_sum_inequality
  (h1: α + β + γ = π)
  (h2: 2 * p > 0)
  (h3: R > 0) :
  (Real.cot α)^2 + (Real.cot β)^2 + (Real.cot γ)^2 ≥ 3 * (9 * (R^2 / p^2) - 1) ∧
  (α = β ∧ β = γ) :=
sorry

end cot_squared_sum_inequality_l364_364545


namespace minAreaTriangle_l364_364055

-- Define the parametric form of the point P on the ellipse
def pointP (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : ℝ × ℝ :=
  (4 * Real.cos θ, 3 * Real.sin θ)

-- Define the equation of the line AB passing through point P and tangent to the circle
def lineAB (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : ℝ → ℝ :=
  λ x, (9 - 4 * x * Real.cos θ) / (3 * Real.sin θ)

-- Define the x-intercept of the line AB
def xIntercept (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : ℝ :=
  9 / (4 * Real.cos θ)

-- Define the y-intercept of the line AB
def yIntercept (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : ℝ :=
  3 / (Real.sin θ)

-- Define the area of the triangle MON
def areaMON (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : ℝ :=
  (1 / 2) * (xIntercept θ hθ) * (yIntercept θ hθ)

-- Main theorem stating the minimum area of the triangle MON
theorem minAreaTriangle : (∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ areaMON θ sorry = 27 / 4) :=
  sorry

end minAreaTriangle_l364_364055


namespace decreasing_condition_l364_364884

-- Define the function f(x)
def f (x : ℝ) (b : ℝ) : ℝ := - (1 / 2) * x^2 + b * Real.log (x + 2)

-- The statement of the problem in Lean 4
theorem decreasing_condition (b : ℝ) : 
  (∀ x : ℝ, -1 < x → Deriv (λ x, f x b) x ≤ 0) ↔ b ≤ 1 := sorry

end decreasing_condition_l364_364884


namespace apple_bags_l364_364155

theorem apple_bags (n : ℕ) (h₁ : n ≥ 70) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 := 
sorry

end apple_bags_l364_364155


namespace evaluate_expression_l364_364346

theorem evaluate_expression : (2 + 1 / (2 + 1 / (2 + 2)) = 22 / 9) := by
    sorry

end evaluate_expression_l364_364346


namespace add_mul_of_3_l364_364076

theorem add_mul_of_3 (a b : ℤ) (ha : ∃ m : ℤ, a = 6 * m) (hb : ∃ n : ℤ, b = 9 * n) : ∃ k : ℤ, a + b = 3 * k :=
by
  sorry

end add_mul_of_3_l364_364076


namespace polynomial_of_degree_2_l364_364546

noncomputable def polynomialSeq (n : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → 
    ∃ (f_k f_k1 f_k2 : Polynomial ℝ),
      f_k ≠ Polynomial.C 0 ∧ (f_k * f_k1 = f_k1.comp f_k2)

theorem polynomial_of_degree_2 (n : ℕ) (h : n ≥ 3) :
  polynomialSeq n → 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → 
    ∃ f : Polynomial ℝ, f = Polynomial.X ^ 2 :=
sorry

end polynomial_of_degree_2_l364_364546


namespace incorrect_statement_option_three_l364_364703

theorem incorrect_statement_option_three (a b m : ℝ) :
  a < b → ¬ (am^2 < bm^2) ↔ m = 0 := 
sorry

end incorrect_statement_option_three_l364_364703


namespace extreme_points_l364_364503

noncomputable def f (x : ℝ) : ℝ :=
  x + 4 / x

theorem extreme_points (P : ℝ × ℝ) :
  (P = (2, f 2) ∨ P = (-2, f (-2))) ↔ 
  ∃ x : ℝ, x ≠ 0 ∧ (P = (x, f x)) ∧ 
    (∀ ε > 0, f (x - ε) < f x ∧ f x > f (x + ε) ∨ f (x - ε) > f x ∧ f x < f (x + ε)) := 
sorry

end extreme_points_l364_364503


namespace profit_increase_percentage_l364_364726

-- Definitions based on problem conditions
variables (a b : ℝ)

-- Condition variables
def cost_per_piece (a : ℝ) : ℝ := 0.75 * a
def profit_per_piece_sep (a : ℝ) : ℝ := 0.25 * a
def profit_per_piece_oct (a : ℝ) : ℝ := 0.15 * a
def pieces_sold_sep (b : ℝ) : ℝ := b
def pieces_sold_oct (b : ℝ) : ℝ := 1.8 * b

-- Proof Statement
theorem profit_increase_percentage:
  let profit_sep := profit_per_piece_sep a * pieces_sold_sep b in
  let profit_oct := profit_per_piece_oct a * pieces_sold_oct b in
  profit_oct = 1.62 * profit_sep :=
by {
  -- Definitions for total profits
  have h1: profit_sep = 0.25 * a * b, by unfold profit_per_piece_sep pieces_sold_sep,
  have h2: profit_oct = 0.15 * a * 1.8 * b, by unfold profit_per_piece_oct pieces_sold_oct,
  -- Calculate the profit increase percentage
  have h3: profit_oct = 0.27 * a * b, by linarith,
  have h4: 1.62 * profit_sep = 1.62 * (0.25 * a * b), by rw h1,
  rw h3 at h4,
  linarith,
}

end profit_increase_percentage_l364_364726


namespace scatter_plot_linear_l364_364499

theorem scatter_plot_linear (points : List (ℝ × ℝ)) 
  (h : ∃ (a b : ℝ), ∀ (x y : ℝ), (x, y) ∈ points → y = a * x + b) :
  (sum_of_squared_residuals points = 0) ∧ (correlation_coefficient points = 1) :=
begin
  sorry
end

end scatter_plot_linear_l364_364499


namespace sum_coefficients_nonzero_power_y_l364_364701

theorem sum_coefficients_nonzero_power_y :
  let f := (2 * x + 3 * y + 4) * (5 * x + 7 * y + 6) 
  (sum (f.get_coeffs \(m\subseteq \mathbf N\rightarrow \mathbf ℤ)) (some m ∈ Some (\ _ y \!= \ 0 )))    ---none of the terms contains 0-) = --- sum of terms with nonzero exponent is 96 :
  sorry

end sum_coefficients_nonzero_power_y_l364_364701


namespace roots_quadratic_l364_364369

theorem roots_quadratic (a b : ℝ) 
  (h1: a^2 + 3 * a - 2010 = 0) 
  (h2: b^2 + 3 * b - 2010 = 0)
  (h_roots: a + b = -3 ∧ a * b = -2010):
  a^2 - a - 4 * b = 2022 :=
by
  sorry

end roots_quadratic_l364_364369


namespace johann_oranges_l364_364533

/-
  Johann had 60 oranges. He decided to eat 10.
  Once he ate them, half were stolen by Carson.
  Carson returned exactly 5. 
  How many oranges does Johann have now?
-/
theorem johann_oranges (initial_oranges : Nat) (eaten_oranges : Nat) (carson_returned : Nat) : 
  initial_oranges = 60 → eaten_oranges = 10 → carson_returned = 5 → 
  (initial_oranges - eaten_oranges) / 2 + carson_returned = 30 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end johann_oranges_l364_364533


namespace exists_2k_integers_not_divisible_by_n_l364_364030

theorem exists_2k_integers_not_divisible_by_n (n k : ℕ) (hn : 1 < n) (hk : 1 < k) (hnk : n < 2^k) :
  ∃ (a : Fin 2k → ℕ), (∀ i, ¬ (n ∣ a i)) ∧ (∀ s : Finset (Fin 2k), ¬ (↑s.card = ↑k) → n ∣ s.sum (λ i, a i)) :=
sorry

end exists_2k_integers_not_divisible_by_n_l364_364030


namespace prob_both_excellent_prob_distribution_expected_value_l364_364974

-- Define the conditions as constants/parameters
def number_of_students : ℕ := 100
def frequency_distribution : (ℕ → ℝ) := 
  λ x, if x = 0 then 0.1 else if x = 1 then 0.1 else if x = 2 then 0.3 else if x = 3 then 0.3 else if x = 4 then 0.2 else 0

def good_score_range := (80, 90)
def excellent_score_range := (90, 100)

def sampling_ratio := 1 / 10
def total_selected_students := 5
def discussion_selected_students := 2

-- Part 1 proof statement
theorem prob_both_excellent : 
  let num_good := 30 * sampling_ratio in
  let num_excellent := 20 * sampling_ratio in
  (num_good = 3) ∧ (num_excellent = 2) →
  (C(2, 2) / C(5, 2)) = 1 / 10 :=
sorry

-- Part 2 proof statement
theorem prob_distribution_expected_value :
  let excellent_probability := 1 / 5 in
  (X : ℕ) → (probability_mass_function :
    (X = 0 → C(3, 0) * (1/5)^0 * (4/5)^3 = 27/125) ∧
    (X = 1 → C(3, 1) * (1/5)^1 * (4/5)^2 = 54/125) ∧
    (X = 2 → C(3, 2) * (1/5)^2 * (4/5)^1 = 36/125) ∧
    (X = 3 → C(3, 3) * (1/5)^3 * (4/5)^0 = 8/125)) →
  (expected_value : ∑ x in Finset.range 4, x * probability_mass_function x = 3 / 5) :=
sorry

end prob_both_excellent_prob_distribution_expected_value_l364_364974


namespace simplify_trig_expression_l364_364991

theorem simplify_trig_expression : 
  (sin (Real.pi / 6) + sin (Real.pi / 3)) / (cos (Real.pi / 6) + cos (Real.pi / 3)) = 1 := by
  sorry

end simplify_trig_expression_l364_364991


namespace inequality_1_inequality_2_inequality_3_inequality_4_l364_364073

-- Definition for the first problem
theorem inequality_1 (x : ℝ) : |2 * x - 1| < 15 ↔ (-7 < x ∧ x < 8) := by
  sorry
  
-- Definition for the second problem
theorem inequality_2 (x : ℝ) : x^2 + 6 * x - 16 < 0 ↔ (-8 < x ∧ x < 2) := by
  sorry

-- Definition for the third problem
theorem inequality_3 (x : ℝ) : |2 * x + 1| > 13 ↔ (x < -7 ∨ x > 6) := by
  sorry

-- Definition for the fourth problem
theorem inequality_4 (x : ℝ) : x^2 - 2 * x > 0 ↔ (x < 0 ∨ x > 2) := by
  sorry

end inequality_1_inequality_2_inequality_3_inequality_4_l364_364073


namespace num_four_digits_divisible_by_5_l364_364878

theorem num_four_digits_divisible_by_5 : 
  ∃ n : ℕ, (∀ x : ℕ, 1000 ≤ x ∧ x ≤ 9999 ∧ x % 5 = 0 ↔ (x = 1000 + (n - 1) * 5)) ∧ n = 1800 :=
by
  let a := 1000
  let l := 9995
  let d := 5
  have a_le_l : a ≤ l := by norm_num
  have h_seq : ∀ x, 1000 ≤ x ∧ x ≤ 9999 ∧ x % 5 = 0 ↔ (∃ n, n ≥ 1 ∧ x = a + (n - 1) * d) :=
    by sorry
  have n : ℕ := ((l - a) / d) + 1
  use n
  split
  { exact h_seq }
  { have h_n : n = 1800 := by sorry
    exact h_n }

end num_four_digits_divisible_by_5_l364_364878


namespace cone_coverage_inequality_l364_364112

theorem cone_coverage_inequality (n : ℕ) (ϕ : Fin n → ℝ)
  (h_cone_cover : ∑ i in Finset.univ, (Real.sin (ϕ i / 4)) ^ 2 ≥ 1) :
  ∑ i in Finset.univ, (ϕ i) ^ 2 ≥ 16 :=
by
  sorry

end cone_coverage_inequality_l364_364112


namespace num_real_x_l364_364405

theorem num_real_x (a b : ℝ) (h1 : a = 123) (h2 : b = 11) :
  ∃ n : ℕ, n = 12 ∧
  ∀ k : ℕ, k ≤ b → ∃ x : ℝ, x = (a - k^2)^2 :=
by
  sorry

end num_real_x_l364_364405


namespace cos_double_angle_in_fourth_quadrant_l364_364374

variable {θ : ℝ}
variable {x₀ y₀ : ℝ}
variable (h₀ : θ ∈ Icc (3 * π / 2) (2 * π))
variable (h₁ : x₀ = Real.cos θ)
variable (h₂ : y₀ = Real.sin θ)
variable (h₃ : x₀ + y₀ = -1 / 3)

theorem cos_double_angle_in_fourth_quadrant :
  Real.cos (2 * θ) = -√17 / 9 :=
sorry

end cos_double_angle_in_fourth_quadrant_l364_364374


namespace circle_reflection_l364_364598

theorem circle_reflection (x y : ℝ) (hx : x = 8) (hy : y = -3) : 
  let reflected_x := -y, reflected_y := -x in
  reflected_x = 3 ∧ reflected_y = -8 :=
by
  sorry

end circle_reflection_l364_364598


namespace sufficient_and_necessary_condition_l364_364287

theorem sufficient_and_necessary_condition (a b : ℝ) :
  (a < b) ↔ (∀ x ∈ set.Ioc 0 1, a + x < b) := sorry

end sufficient_and_necessary_condition_l364_364287


namespace area_of_triangle_ABC_l364_364558

noncomputable def sqrt4 (x : ℝ) : ℝ := real.sqrt (real.sqrt x)

-- Define the points O, A, B, and C in the 3D coordinate system
def O : ℝ × ℝ × ℝ := (0, 0, 0)
def A : ℝ × ℝ × ℝ := (sqrt4 81, 0, 0)
def B : ℝ × ℝ × ℝ := (0, sqrt4 81, 0)
def C : ℝ × ℝ × ℝ := (0, 0, sqrt4 81)

-- Define the distance OA, which is √[4]{81}
def OA : ℝ := sqrt4 81

-- Define the angle BAC as 45 degrees in radians
def angle_BAC : ℝ := real.pi / 4

-- The correct answer for the area of triangle ABC
def correct_area : ℝ := (9 * real.sqrt 2) / 4

-- Now state the theorem
theorem area_of_triangle_ABC : 
  ∃ (A B C : ℝ × ℝ × ℝ), 
    (A = (sqrt4 81, 0, 0)) ∧
    (B = (0, sqrt4 81, 0)) ∧
    (C = (0, 0, sqrt4 81)) ∧
    (∠ B A C = angle_BAC) ∧
    (area_of_triangle A B C = correct_area) :=
sorry

end area_of_triangle_ABC_l364_364558


namespace find_trajectoryQ_intersection_with_line_and_pointT_l364_364363

-- Define the circle E and its properties
def circleE (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16
def pointF := (1, 0)

-- Define the trajectory Γ of point Q
noncomputable def trajectoryQ (x y : ℝ) : Prop :=
  (x^2) / 4 + (y^2) / 3 = 1

-- Problem (1) Statement
theorem find_trajectoryQ :
  ∀ P : ℝ × ℝ, circleE P.1 P.2 →
  ∀ Q : ℝ × ℝ, ∃ Q : ℝ × ℝ, (trajectoryQ Q.1 Q.2) :=
sorry

-- Problem (2) Statement
theorem intersection_with_line_and_pointT (k : ℝ) :
  ∃ T : ℝ × ℝ, T = (4, 0) ∧ ∀ R S : ℝ × ℝ, trajectoryQ R.1 R.2 → trajectoryQ S.1 S.2 →
  ((∃ x1 y1 x2 y2 : ℝ, R = (x1, y1) ∧ S = (x2, y2) ∧ 
   y1 = k * (x1 - 1) ∧ y2 = k * (x2 - 1)) →
   ∃⦃O : ℝ × ℝ⦄ (OTS O T S : ℝ) (OTR O T R : ℝ),
   OTS = OTR) :=
sorry

end find_trajectoryQ_intersection_with_line_and_pointT_l364_364363


namespace sum_of_two_digit_divisors_l364_364959

theorem sum_of_two_digit_divisors (d : ℕ) (h₁ : d > 0) (h₂ : 221 % d = 5) : 
  ∑ x in {dd : ℕ | dd ∈ (finset.Icc 10 99) ∧ (221 - 5) % dd = 0}, x = 216 := 
by sorry

end sum_of_two_digit_divisors_l364_364959


namespace gcd_values_count_l364_364692

theorem gcd_values_count :
  ∃ a b : ℕ, (gcd a b) * (nat.lcm a b) = 180 ∧
    set.card { gcd a b | ∃ a b, a * b = 180 } = 8 :=
by
  -- Problem statement as provided by conditions and question
  -- Definitions and notations are provided correctly and fully, proof is omitted
  sorry

end gcd_values_count_l364_364692


namespace apple_count_l364_364143

/--
Given the total number of apples in several bags, where each bag contains either 12 or 6 apples,
and knowing that the total number of apples is between 70 and 80 inclusive,
prove that the total number of apples can only be 72 or 78.
-/
theorem apple_count (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : ∃ k : ℕ, n = 6 * k) : n = 72 ∨ n = 78 :=
by {
  sorry
}

end apple_count_l364_364143


namespace simplify_trig_l364_364995

open Real

theorem simplify_trig : 
  (sin (30 * pi / 180) + sin (60 * pi / 180)) / (cos (30 * pi / 180) + cos (60 * pi / 180)) = tan (45 * pi / 180) :=
by
  sorry

end simplify_trig_l364_364995


namespace angle_opposite_side_AB_l364_364911

theorem angle_opposite_side_AB (a b c : ℝ) (h : (a + b + c) * (a + b - c) = a * b) : 
  ∠ (a + b + c) = 120 :=
begin
  sorry
end

end angle_opposite_side_AB_l364_364911


namespace ratio_is_one_eighth_l364_364916

def point (x y : ℝ) := (x, y)

def large_square_area : ℝ := 10 * 10

def shaded_vertices : List (ℝ × ℝ) :=
  [point 0 0, point 10 5, point 5 10, point 5 5]

def area_of_quadrilateral (vertices : List (ℝ × ℝ)) : ℝ :=
  (1 / 2) * 
  abs (
    vertices[0].1 * vertices[1].2 + 
    vertices[1].1 * vertices[2].2 + 
    vertices[2].1 * vertices[3].2 + 
    vertices[3].1 * vertices[0].2 -
    (vertices[0].2 * vertices[1].1 + 
    vertices[1].2 * vertices[2].1 + 
    vertices[2].2 * vertices[3].1 + 
    vertices[3].2 * vertices[0].1)
  )

def shaded_area : ℝ :=
  area_of_quadrilateral shaded_vertices

def ratio_of_areas : ℝ :=
  shaded_area / large_square_area

theorem ratio_is_one_eighth :
  ratio_of_areas = 1 / 8 := by
  sorry

end ratio_is_one_eighth_l364_364916


namespace find_real_polynomials_l364_364341

theorem find_real_polynomials (f : ℝ → ℝ) (h : ∀ x y : ℝ, 2 * y * f(x + y) + (x - y) * (f(x) + f(y)) ≥ 0) :
  ∃ c : ℝ, c ≥ 0 ∧ ∀ x : ℝ, f(x) = c * x :=
sorry

end find_real_polynomials_l364_364341


namespace cos_angle_between_vectors_l364_364714

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vector3D (P Q : Point3D) : Point3D :=
  { x := Q.x - P.x, y := Q.y - P.y, z := Q.z - P.z }

def dotProduct (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def magnitude (v : Point3D) : ℝ :=
  Real.sqrt (v.x * v.x + v.y * v.y + v.z * v.z)

def cosTheta (P Q R : Point3D) : ℝ :=
  dotProduct (vector3D P Q) (vector3D P R) / (magnitude (vector3D P Q) * magnitude (vector3D P R))

theorem cos_angle_between_vectors :
  let A := Point3D.mk (-4) 3 0
  let B := Point3D.mk 0 1 3
  let C := Point3D.mk (-2) 4 (-2)
  cosTheta A B C = 0 :=
by
  let A := Point3D.mk (-4) 3 0
  let B := Point3D.mk 0 1 3
  let C := Point3D.mk (-2) 4 (-2)
  sorry

end cos_angle_between_vectors_l364_364714


namespace math_problem_l364_364314

theorem math_problem : 
  27 * ((8/3 : ℚ) - (13/4 : ℚ)) / ((3/2 : ℚ) + (11/5 : ℚ)) = -4 - (43/74 : ℚ) :=
by
  sorry

end math_problem_l364_364314


namespace min_vertical_distance_l364_364097

-- Definitions of the given functions
def abs_function (x : ℝ) : ℝ := abs x
def quad_function (x : ℝ) : ℝ := x^2 - 5 * x + 4

-- The statement that we need to prove
theorem min_vertical_distance := 
  ∃ x : ℝ, abs_function x = quad_function x :=
sorry

end min_vertical_distance_l364_364097


namespace sum_of_interior_edges_l364_364280

-- Conditions
def width_of_frame_piece : ℝ := 1.5
def one_interior_edge : ℝ := 4.5
def total_frame_area : ℝ := 27

-- Statement of the problem as a theorem in Lean
theorem sum_of_interior_edges : 
  (∃ y : ℝ, (width_of_frame_piece * 2 + one_interior_edge) * (width_of_frame_piece * 2 + y) 
    - one_interior_edge * y = total_frame_area) →
  (4 * (one_interior_edge + y) = 12) :=
sorry

end sum_of_interior_edges_l364_364280


namespace annual_income_A_l364_364626

variable (A B C : ℝ)
variable (monthly_income_C : C = 17000)
variable (monthly_income_B : B = C + 0.12 * C)
variable (ratio_A_to_B : A / B = 5 / 2)

theorem annual_income_A (A B C : ℝ) 
    (hC : C = 17000) 
    (hB : B = C + 0.12 * C) 
    (hR : A / B = 5 / 2) : 
    A * 12 = 571200 :=
by
  sorry

end annual_income_A_l364_364626


namespace probability_A_east_lemma_l364_364931

noncomputable def probability_A_east {α β γ : ℕ} (hα : α = 40) (hβγ : β + γ = 180 - α) : ℚ :=
  140 / 360

theorem probability_A_east_lemma {α β γ : ℕ} 
  (hα : α = 40)
  (hβγ : β + γ = 180 - α) :
  probability_A_east hα hβγ = 7 / 18 :=
by
  unfold probability_A_east
  rw [hα]
  norm_num
  sorry

end probability_A_east_lemma_l364_364931


namespace car_stopping_distance_l364_364227

theorem car_stopping_distance
  (a : ℕ := 36) (d : ℤ := -9) (n : ℕ := 4):
  (a = 36) →
  (d = -9) →
  (n = 4) →
  let l := a + (n - 1) * d in
  let S := (n * (a + l)) / 2 in
  S = 90 := 
by
  intros
  sorry

end car_stopping_distance_l364_364227


namespace monotone_increasing_intervals_exists_x0_implies_p_l364_364864

noncomputable def f (x : ℝ) := 6 * Real.log x + x ^ 2 - 8 * x
noncomputable def g (x : ℝ) (p : ℝ) := p / x + x ^ 2

theorem monotone_increasing_intervals :
  (∀ x, (0 < x ∧ x ≤ 1) → ∃ ε > 0, ∀ y, x < y → f y > f x) ∧
  (∀ x, (3 ≤ x) → ∃ ε > 0, ∀ y, x < y → f y > f x) := by
  sorry

theorem exists_x0_implies_p :
  (∃ x0, 1 ≤ x0 ∧ x0 ≤ Real.exp 1 ∧ f x0 > g x0 p) → p < -8 := by
  sorry

end monotone_increasing_intervals_exists_x0_implies_p_l364_364864


namespace minimizer_point_l364_364801

-- Define points in the plane for the triangle
variables {Point : Type} [metric_space Point] (A B C : Point)

-- Define positive real numbers a, b, c
variables (a b c : ℝ)
noncomputable def MA (A M : Point) : ℝ := dist A M
noncomputable def MB (B M : Point) : ℝ := dist B M
noncomputable def MC (C M : Point) : ℝ := dist C M

-- Declare the points A, B, C form a triangle
-- We are not using this condition directly, as the metric_space will suffice  

-- Prove the minimal point M is A when a >= b + c
theorem minimizer_point (h : a ≥ b + c) : 
  ∃ M : Point, M = A ∧ (∀ X : Point, a * (MA A X) + b * (MB B X) + c * (MC C X) ≥ a * (MA A A) + b * (MB B A) + c * (MC C A)) :=
begin
  sorry
end

end minimizer_point_l364_364801


namespace initial_apples_count_l364_364637

variable (initial_apples : ℕ)
variable (used_apples : ℕ := 2)
variable (bought_apples : ℕ := 23)
variable (final_apples : ℕ := 38)

theorem initial_apples_count :
  initial_apples - used_apples + bought_apples = final_apples ↔ initial_apples = 17 := by
  sorry

end initial_apples_count_l364_364637


namespace exists_twelve_distinct_x_l364_364451

theorem exists_twelve_distinct_x :
  ∃ S : Set ℝ, (S.card = 12) ∧ (∀ x ∈ S, ∃ k : ℤ, 0 ≤ k ∧ k ≤ 11 ∧ (sqrt (123 - sqrt x) = k)) :=
by
  sorry

end exists_twelve_distinct_x_l364_364451


namespace length_of_railway_bridge_l364_364275

theorem length_of_railway_bridge
    (time_cross_bridge : ℕ := 20)
    (time_cross_man : ℕ := 8)
    (length_train : ℕ := 120)
    (speed_train : ℕ := 15) :
    let L := 180 in
    speed_train * time_cross_bridge = length_train + L :=
by
  sorry

end length_of_railway_bridge_l364_364275


namespace valid_number_of_apples_l364_364136

theorem valid_number_of_apples (n : ℕ) : 
  (∃ k m : ℕ, n = 12 * k + 6 * m) ∧ (70 ≤ n ∧ n ≤ 80) ↔ (n = 72 ∨ n = 78) :=
by
  sorry

end valid_number_of_apples_l364_364136


namespace num_real_numbers_l364_364431

theorem num_real_numbers (x : ℝ) : 
  (∃ n : ℕ, n ≤ 11 ∧ x = (123 - n^2)^2) ↔ 
  (∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ (x = (123 - s^2)^2) ∧ 
  (∃! x, ∃ k : ℕ, 0 ≤ k ∧ k ≤ 11 ∧ x = (123 - k^2)^2)) :=
by
  sorry

end num_real_numbers_l364_364431


namespace greatest_three_digit_multiple_of_23_l364_364664

def is_three_digit (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

def is_multiple_of_23 (n : ℕ) : Prop :=
  n % 23 = 0

theorem greatest_three_digit_multiple_of_23 :
  ∀ n, is_three_digit n ∧ is_multiple_of_23 n → n ≤ 989 :=
by
  sorry

end greatest_three_digit_multiple_of_23_l364_364664


namespace range_of_f_l364_364336

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ x + 1

theorem range_of_f : set.range (f) = set.Icc (3 / 2) 3 :=
by
  sorry

end range_of_f_l364_364336


namespace Joseph_has_122_socks_l364_364947

def JosephSocks : Nat := 
  let red_pairs := 9 / 2
  let white_pairs := red_pairs + 2
  let green_pairs := 2 * red_pairs
  let blue_pairs := 3 * green_pairs
  let black_pairs := blue_pairs - 5
  (red_pairs + white_pairs + green_pairs + blue_pairs + black_pairs) * 2

theorem Joseph_has_122_socks : JosephSocks = 122 := 
  by
  sorry

end Joseph_has_122_socks_l364_364947


namespace simplify_trig_expression_l364_364990

theorem simplify_trig_expression : 
  (sin (Real.pi / 6) + sin (Real.pi / 3)) / (cos (Real.pi / 6) + cos (Real.pi / 3)) = 1 := by
  sorry

end simplify_trig_expression_l364_364990


namespace part_a_part_b_l364_364015

variables {A B C A_1 A_2 A_3 B_1 B_2 B_3 C_1 C_2 C_3 : Type}

-- Define the conditions
def square_on_side (P Q R S : Type) : Prop := 
  sorry -- Definition of the condition for a square constructed on a side of a triangle

def lines_passthrough_points (line_1 line_2 point : Type) : Prop :=
  sorry -- Definition of the condition for lines passing through points

-- Define the problem statements
theorem part_a (H1 : square_on_side A A_1 A_2 A_3)
  (H2 : square_on_side B B_1 B_2 B_3)
  (H3 : square_on_side C C_1 C_2 C_3)
  (H4 : lines_passthrough_points A_1 A_2 B)
  (H5 : lines_passthrough_points B_1 B_2 C)
  (H6 : lines_passthrough_points C_1 C_2 A)
  (H7 : lines_passthrough_points A_2 A_3 C)
  (H8 : lines_passthrough_points B_2 B_3 A)
  (H9 : lines_passthrough_points C_2 C_3 B) :
  ∃ P : Type, intersects_at_one_point AA_2 B_1B_2 C_1C_3 P :=
sorry

theorem part_b (H1 : square_on_side A A_1 A_2 A_3)
  (H2 : square_on_side B B_1 B_2 B_3)
  (H3 : square_on_side C C_1 C_2 C_3)
  (H4 : lines_passthrough_points A_1 A_2 B)
  (H5 : lines_passthrough_points B_1 B_2 C)
  (H6 : lines_passthrough_points C_1 C_2 A)
  (H7 : lines_passthrough_points A_2 A_3 C)
  (H8 : lines_passthrough_points B_2 B_3 A)
  (H9 : lines_passthrough_points C_2 C_3 B) :
  ∃ P : Type, intersects_at_one_point AA_2 BB_2 CC_2 P :=
sorry

end part_a_part_b_l364_364015


namespace apple_count_l364_364145

/--
Given the total number of apples in several bags, where each bag contains either 12 or 6 apples,
and knowing that the total number of apples is between 70 and 80 inclusive,
prove that the total number of apples can only be 72 or 78.
-/
theorem apple_count (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : ∃ k : ℕ, n = 6 * k) : n = 72 ∨ n = 78 :=
by {
  sorry
}

end apple_count_l364_364145


namespace factorial_div_l364_364901

theorem factorial_div (p : ℕ) : (p^2)! ∣ (p!)^(p+1) :=
by
  sorry

end factorial_div_l364_364901


namespace problem_I_a_le_0_problem_I_0_lt_a_le_1_problem_I_a_gt_1_problem_II_min_positive_k_l364_364390

-- Define the function f
def f (a x : ℝ) : ℝ := x * Real.exp x - a * x

-- Problem (I)
theorem problem_I_a_le_0 (a x : ℝ) (h₀ : a ≤ 0) (h₁ : f a x > 0) : x > 0 :=
by
  sorry

theorem problem_I_0_lt_a_le_1 (a x : ℝ) (h₀ : 0 < a ∧ a ≤ 1) (h₁ : f a x > 0) : x > 0 ∨ x < Real.log a :=
by
  sorry

theorem problem_I_a_gt_1 (a x : ℝ) (h₀ : a > 1) (h₁ : f a x > 0) : x > Real.log a ∨ x < 0 :=
by
  sorry

-- Problem (II)
theorem problem_II_min_positive_k : ∃ k : ℕ, k = 1 ∧ ∀ x : ℝ, f 2 x + k > 0 :=
by
  use 1
  sorry

end problem_I_a_le_0_problem_I_0_lt_a_le_1_problem_I_a_gt_1_problem_II_min_positive_k_l364_364390


namespace probability_at_least_three_heads_l364_364078

def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem probability_at_least_three_heads :
  (∑ k in {3, 4, 5}, binomial 5 k) = 16 → (16 / 32 = 1 / 2) :=
by
  sorry

end probability_at_least_three_heads_l364_364078


namespace num_real_x_l364_364406

theorem num_real_x (a b : ℝ) (h1 : a = 123) (h2 : b = 11) :
  ∃ n : ℕ, n = 12 ∧
  ∀ k : ℕ, k ≤ b → ∃ x : ℝ, x = (a - k^2)^2 :=
by
  sorry

end num_real_x_l364_364406


namespace greatest_three_digit_multiple_23_l364_364654

theorem greatest_three_digit_multiple_23 : 
  ∃ n : ℕ, n < 1000 ∧ n % 23 = 0 ∧ (∀ m : ℕ, m < 1000 ∧ m % 23 = 0 → m ≤ n) ∧ n = 989 :=
sorry

end greatest_three_digit_multiple_23_l364_364654


namespace sum_perimeter_area_l364_364795

-- Define the points of the parallelogram
def P1 := (2, 7)
def P2 := (7, 2)
def P3 := (7, 7)
def P4 := (2, 2)

-- Define function to calculate distance between two points
def distance (A B : ℕ × ℕ) : ℕ :=
  (abs (A.1 - B.1)) + (abs (A.2 - B.2))

-- Define function to calculate perimeter of the parallelogram
def perimeter : ℕ :=
  distance P1 P3 + distance P2 P4 + distance P1 P2 + distance P3 P4

-- Define the area of the parallelogram, which is taken as a square of the side length
def area : ℕ :=
  (distance P1 P3) * (distance P1 P3)

-- Prove the sum of the perimeter and area is 45
theorem sum_perimeter_area : perimeter + area = 45 :=
by
  sorry

end sum_perimeter_area_l364_364795


namespace num_real_satisfying_x_l364_364446

theorem num_real_satisfying_x : 
  (∃ (x : ℝ), ∃ (s : ℕ) (hs : s ∈ set.Ico 0 (floor (sqrt 123) + 1)), 
   s = int.of_nat (floor (sqrt (123 - sqrt x)))) →
   12 := sorry

end num_real_satisfying_x_l364_364446


namespace factorial_div_l364_364900

theorem factorial_div (p : ℕ) : (p^2)! ∣ (p!)^(p+1) :=
by
  sorry

end factorial_div_l364_364900


namespace fraction_sum_l364_364316

theorem fraction_sum : (3 / 4 : ℚ) + (6 / 9 : ℚ) = 17 / 12 := 
by 
  -- Sorry placeholder to indicate proof is not provided.
  sorry

end fraction_sum_l364_364316


namespace fraction_of_juniors_l364_364512

theorem fraction_of_juniors (J S : ℕ) (h1 : J > 0) (h2 : S > 0) (h : 1 / 2 * J = 2 / 3 * S) : J / (J + S) = 4 / 7 :=
by
  sorry

end fraction_of_juniors_l364_364512


namespace amount_paid_is_200_l364_364566

-- Definitions of the costs and change received
def cost_of_pants := 140
def cost_of_shirt := 43
def cost_of_tie := 15
def change_received := 2

-- Total cost calculation
def total_cost := cost_of_pants + cost_of_shirt + cost_of_tie

-- Lean proof statement
theorem amount_paid_is_200 : total_cost + change_received = 200 := by
  -- Definitions ensure the total cost and change received are used directly from conditions
  sorry

end amount_paid_is_200_l364_364566


namespace max_S_value_achieve_max_S_maximum_S_value_and_achieving_it_l364_364848

theorem max_S_value (x y S : ℝ) (hx : x > 0) (hy : y > 0)
  (hS : S = min x (y + 1/x) (1/y)) : S ≤ Real.sqrt 2 :=
sorry

theorem achieve_max_S (S : ℝ) (hx : x = Real.sqrt 2) (hy : y = Real.sqrt 2 / 2) :
  S = Real.sqrt 2 →
  S = min x (y + 1/x) (1/y) :=
sorry

-- Lean statement to state the overall problem
theorem maximum_S_value_and_achieving_it (x y S : ℝ) :
  (∀x > 0, y > 0, S = min x (y + 1/x) (1/y) → S ≤ Real.sqrt 2) ∧
  (x = Real.sqrt 2 ∧ y = Real.sqrt 2 / 2 → S = Real.sqrt 2 ∧ S = min x (y + 1/x) (1/y)) :=
⟨max_S_value, achieve_max_S⟩

end max_S_value_achieve_max_S_maximum_S_value_and_achieving_it_l364_364848


namespace sector_area_l364_364501

-- Define the conditions as hypotheses
axiom central_angle_eq : central_angle = 2 -- central angle is 2 radians
axiom arc_length_eq : arc_length = 4 -- arc length is 4 cm

-- Define the conjecture to be proven
theorem sector_area :
  (central_angle = 2) ∧ (arc_length = 4) → (area_of_sector = 4) :=
sorry

end sector_area_l364_364501


namespace smallest_r_for_B_in_C_l364_364552

def A : Set ℝ := {t | 0 < t ∧ t < 2 * Real.pi}

def B : Set (ℝ × ℝ) := 
  {p | ∃ t ∈ A, p.1 = Real.sin t ∧ p.2 = 2 * Real.sin t * Real.cos t}

def C (r : ℝ) : Set (ℝ × ℝ) := 
  {p | p.1^2 + p.2^2 ≤ r^2 ∧ r > 0}

theorem smallest_r_for_B_in_C : ∃ r, (B ⊆ C r ∧ ∀ r', r' < r → ¬ (B ⊆ C r')) :=
  sorry

end smallest_r_for_B_in_C_l364_364552


namespace negation_of_p_l364_364844

def p : Prop := ∀ x : ℝ, x ≥ 0 → x^2 - x ≥ 0

theorem negation_of_p : ¬ p ↔ ∃ x : ℝ, x ≥ 0 ∧ x^2 - x < 0 :=
by
  sorry

end negation_of_p_l364_364844


namespace range_of_H_l364_364785

def H (x : ℝ) : ℝ := |3 * x + 1| - |x - 2|

theorem range_of_H : set.range H = set.univ :=
sorry

end range_of_H_l364_364785


namespace probability_composite_product_l364_364463

theorem probability_composite_product :
  let dice_faces := 6
  let rolls := 4
  let total_outcomes := dice_faces ^ rolls
  let non_composite_cases := 13
  let non_composite_probability := non_composite_cases / total_outcomes
  let composite_probability := 1 - non_composite_probability
  composite_probability = 1283 / 1296 := by
  sorry

end probability_composite_product_l364_364463


namespace number_of_subsets_l364_364875

noncomputable def Q : Set ℕ := { x | 2 * x^2 - 5 * x ≤ 0 }
def P (Q : Set ℕ) := { P | P ⊆ Q }
def card_subsets (Q : Set ℕ) : ℕ := 2 ^ Q.to_finset.card

theorem number_of_subsets :
  card_subsets Q = 8 :=
by
  sorry

end number_of_subsets_l364_364875


namespace plane_eq_of_point_and_parallel_l364_364808

theorem plane_eq_of_point_and_parallel (A B C D : ℤ) 
  (h1 : A = 3) (h2 : B = -2) (h3 : C = 4) 
  (point : ℝ × ℝ × ℝ) (hpoint : point = (2, -3, 5))
  (h4 : 3 * (2 : ℝ) - 2 * (-3 : ℝ) + 4 * (5 : ℝ) + (D : ℝ) = 0)
  (hD : D = -32)
  (hGCD : Int.gcd (Int.natAbs 3) (Int.gcd (Int.natAbs (-2)) (Int.gcd (Int.natAbs 4) (Int.natAbs (-32)))) = 1) : 
  3 * (x : ℝ) - 2 * (y : ℝ) + 4 * (z : ℝ) - 32 = 0 :=
sorry

end plane_eq_of_point_and_parallel_l364_364808


namespace angle_OHC_eq_90_l364_364514

variables {A B C H O X Y P Q : Type} [EuclideanGeometry A B C H O X Y P Q]

-- Given conditions
variables [AcuteAngledTriangle ABC]
variables [IsOrthocenter H ABC]
variables [IsCircumcenter O ABC]
variables [PerpBisectorIntersection CH AC X]
variables [PerpBisectorIntersection CH BC Y]
variables [LineIntersection XO AB P]
variables [LineIntersection YO AB Q]
variables [lengthXP_YQ_eq_AB_XY : XP + YQ = AB + XY]

-- Prove angle OHC equals 90 degrees
theorem angle_OHC_eq_90 (h : AcuteAngledTriangle ABC) 
                        (h1 : IsOrthocenter H ABC) 
                        (h2 : IsCircumcenter O ABC) 
                        (h3 : PerpBisectorIntersection CH AC X) 
                        (h4 : PerpBisectorIntersection CH BC Y) 
                        (h5 : LineIntersection XO AB P) 
                        (h6 : LineIntersection YO AB Q) 
                        (h7 : XP + YQ = AB + XY) :
  ∠ O H C = 90 :=
sorry

end angle_OHC_eq_90_l364_364514


namespace ratio_of_side_lengths_l364_364109

theorem ratio_of_side_lengths 
  (area_ratio : ℚ)
  (h : area_ratio = 50 / 98) :
  ∃ p q r : ℕ, (p > 0) ∧ (q > 0) ∧ (r > 0) ∧ 
               (√area_ratio = (p * √q) / r) ∧ (p + q + r = 13) :=
sorry

end ratio_of_side_lengths_l364_364109


namespace find_percentage_l364_364892

theorem find_percentage (x p : ℝ) (h1 : x = 840) (h2 : 0.25 * x + 15 = p / 100 * 1500) : p = 15 := 
by
  sorry

end find_percentage_l364_364892


namespace johann_oranges_l364_364535

/-
  Johann had 60 oranges. He decided to eat 10.
  Once he ate them, half were stolen by Carson.
  Carson returned exactly 5. 
  How many oranges does Johann have now?
-/
theorem johann_oranges (initial_oranges : Nat) (eaten_oranges : Nat) (carson_returned : Nat) : 
  initial_oranges = 60 → eaten_oranges = 10 → carson_returned = 5 → 
  (initial_oranges - eaten_oranges) / 2 + carson_returned = 30 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end johann_oranges_l364_364535


namespace number_of_projects_min_total_time_l364_364251

noncomputable def energy_transfer_projects (x y : ℕ) : Prop :=
  x + y = 15 ∧ x = 2 * y - 3

theorem number_of_projects (x y : ℕ) (h : energy_transfer_projects x y) :
  x = 9 ∧ y = 6 :=
sorry

noncomputable def minimize_time (m : ℕ) : Prop :=
  m + 10 - m = 10 ∧ 10 - m > m / 2 ∧ -2 * m + 80 = 68

theorem min_total_time (m : ℕ) (h : minimize_time m) :
  m = 6 ∧ 10 - m = 4 :=
sorry

end number_of_projects_min_total_time_l364_364251


namespace rectangular_solid_surface_area_l364_364793

theorem rectangular_solid_surface_area (a b c : ℕ) (h1 : a.prime) (h2 : b.prime) (h3 : c.prime) (h4 : a * b * c = 1155) : 
  2 * (a * b + b * c + c * a) = 142 :=
by
  sorry

end rectangular_solid_surface_area_l364_364793


namespace alice_bob_meet_after_five_turns_l364_364724

def alice_and_bob_meet (circle_points: ℕ) (start_point: ℕ) (alice_move: ℕ) (bob_move: ℕ) : ℕ :=
  let effective_bob_move := circle_points - bob_move
  let relative_move := (effective_bob_move + alice_move) % circle_points
  let num_turns := (circle_points / relative_move) -- This division assumes exactly k turns fit, adjust otherwise
  num_turns

theorem alice_bob_meet_after_five_turns :
  alice_and_bob_meet 15 15 4 8 = 5 :=
by
  unfold alice_and_bob_meet
  have effective_move := (15 - 8) % 15
  have relative_move := (4 + effective_move) % 15
  have num_turns := 15 / relative_move
  calc
    relative_move = 3 : by sorry
    num_turns = 5 : by sorry

end alice_bob_meet_after_five_turns_l364_364724


namespace range_of_product_of_slopes_l364_364571

noncomputable def P_in_first_quadrant_and_on_hyperbola (x y : ℝ) : Prop :=
  (x > 0) ∧ (y > 0) ∧ (x^2 / 4 - y^2 = 1)

theorem range_of_product_of_slopes (x y : ℝ) (h : P_in_first_quadrant_and_on_hyperbola x y) :
  let k1 := y / (x + 2)
      k2 := y / x
      k3 := y / (x - 2)
  in 0 < k1 * k2 * k3 ∧ k1 * k2 * k3 < 1 / 8 :=
by
  sorry

end range_of_product_of_slopes_l364_364571


namespace part_I_solution_set_part_II_range_of_b_l364_364391

def f (x : ℝ) (a : ℝ) : ℝ := abs (x + real.sqrt a) - abs (x - real.sqrt (1 - a))

-- Part (I)
theorem part_I_solution_set (x : ℝ) : f x 1 ≥ 1/2 ↔ x ≥ -0.25 := 
by sorry

-- Part (II)
theorem part_II_range_of_b (a b : ℝ) (h_a : 0 ≤ a ∧ a ≤ 1) (h_b : ∀ x, ¬ f x a ≥ b) : b > real.sqrt (1 - a) + real.sqrt a := 
by sorry


end part_I_solution_set_part_II_range_of_b_l364_364391


namespace elements_of_set_A_l364_364851

theorem elements_of_set_A (A : Set ℝ) (h₁ : ∀ a : ℝ, a ∈ A → (1 + a) / (1 - a) ∈ A)
(h₂ : -3 ∈ A) : A = {-3, -1/2, 1/3, 2} := by
  sorry

end elements_of_set_A_l364_364851


namespace B_finishes_race_in_25_seconds_l364_364918

def time_for_race (distance time speed : ℝ) : ℝ := distance / speed

def speed (distance time : ℝ) : ℝ := distance / time

theorem B_finishes_race_in_25_seconds :
  ∀ (A_time B_distance race_distance : ℝ),
  A_time = 20 ∧ race_distance = 110 ∧ race_distance - B_distance = 22 → 
  time_for_race race_distance (speed B_distance A_time) = 25 :=
by
  intros A_time B_distance race_distance h
  obtain ⟨hA_time, hrace_distance, hBeats⟩ := h
  rw [hA_time, hrace_distance, hBeats]
  unfold time_for_race speed
  field_simp
  norm_num
  sorry

end B_finishes_race_in_25_seconds_l364_364918


namespace probability_composite_product_l364_364464

theorem probability_composite_product :
  let dice_faces := 6
  let rolls := 4
  let total_outcomes := dice_faces ^ rolls
  let non_composite_cases := 13
  let non_composite_probability := non_composite_cases / total_outcomes
  let composite_probability := 1 - non_composite_probability
  composite_probability = 1283 / 1296 := by
  sorry

end probability_composite_product_l364_364464


namespace selection_ways_l364_364064

noncomputable def choose (n k : ℕ) : ℕ := nat.choose n k

theorem selection_ways : 
  let teachers := 4
  let students := 5
  let total_ways := (choose teachers 1) * (choose students 2) + (choose teachers 2) * (choose students 1)
  total_ways = 70 :=
by
  sorry

end selection_ways_l364_364064


namespace whisky_replacement_quantity_l364_364235

variable (V x : ℝ) -- Total volume V and quantity replaced x
variable (initialAlc finalAlc replaceAlc : ℝ)

axiom initial_alcohol_percent : initialAlc = 0.40
axiom replace_alcohol_percent : replaceAlc = 0.19
axiom final_alcohol_percent : finalAlc = 0.26

def quantity_of_whisky_replaced (V : ℝ) : ℝ :=
  let alc_initial := initialAlc * V
  let alc_removed := initialAlc * x
  let alc_added := replaceAlc * x
  let alc_final := finalAlc * V
  (alc_initial - alc_removed + alc_added = alc_final) → (x = (2/3) * V)

theorem whisky_replacement_quantity (V : ℝ) :
  quantity_of_whisky_replaced V :=
by
  apply sorry

end whisky_replacement_quantity_l364_364235


namespace find_f_value_l364_364388

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^5 - b * x^3 + c * x - 3

theorem find_f_value (a b c : ℝ) (h : f a b c (-3) = 7) : f a b c 3 = -13 :=
by
  sorry

end find_f_value_l364_364388


namespace path_of_D_l364_364979

theorem path_of_D (A C : Point) (λ : ℝ) (b : Line) :
  (B : Point) → B ∈ b →
  let D := λ (B : Point), ∃ d : Line, (D ∈ d ∧ d ∥ b) ∧ (D = point_on_ray C (B- A) (λ * distance A B)) in
  ∀ B, D(B) := λ B, 
    by
      sorry

end path_of_D_l364_364979


namespace min_value_x_y_l364_364824

theorem min_value_x_y {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 4 / y + 1 / x = 4) : x + y ≥ 2 :=
sorry

end min_value_x_y_l364_364824


namespace gilled_mushrooms_count_l364_364263

theorem gilled_mushrooms_count : 
  ∀ (total_mushrooms gilled_mushrooms_ratio spotted_mushrooms_ratio : ℕ),
  (total_mushrooms = 30) →
  (gilled_mushrooms_ratio = 1) →
  (spotted_mushrooms_ratio = 9) →
  total_mushrooms / (gilled_mushrooms_ratio + spotted_mushrooms_ratio) = 3 :=
by
  intros total_mushrooms gilled_mushrooms_ratio spotted_mushrooms_ratio
  assume h_total h_gilled h_spotted
  rw [h_total, h_gilled, h_spotted]
  norm_num
  sorry

end gilled_mushrooms_count_l364_364263


namespace sum_of_roots_l364_364828

variables {a b c : ℝ}

-- Conditions
-- The polynomial with roots a, b, c
def poly (x : ℝ) : ℝ := 24 * x^3 - 36 * x^2 + 14 * x - 1

-- The roots are in (0, 1)
def in_interval (x : ℝ) : Prop := 0 < x ∧ x < 1

-- All roots are distinct
def distinct (a b c : ℝ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a

-- Main Theorem
theorem sum_of_roots :
  (∀ x, poly x = 0 → x = a ∨ x = b ∨ x = c) →
  in_interval a →
  in_interval b →
  in_interval c →
  distinct a b c →
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 3 / 2) :=
by
  intros
  sorry

end sum_of_roots_l364_364828


namespace proof_equation_a_is_first_order_linear_proof_equation_b_is_not_first_order_linear_proof_equation_c_is_not_first_order_linear_l364_364380

-- Definitions based on the conditions provided.
def equation_a (x y y' : ℝ) : Prop := y' + (2 * y) / (x + 1) = (x + 1) ^ 3
def equation_b (x y y' y'' : ℝ) : Prop := y'' + 2 * x * y = 0
def equation_c (x y y' : ℝ) : Prop := y' + x * y ^ 2 = (x - 3) ^ 2

-- The first-order linear equation predicate.
def is_first_order_linear (eq : (ℝ → ℝ → ℝ → Prop)) : Prop :=
  ∀ (x y y' : ℝ), eq x y y' → y' = y' ∧ y = y ∧ ∃ p q, eq x y y = y' + p * y = q

-- The proof problems.
theorem proof_equation_a_is_first_order_linear :
  is_first_order_linear (λ x y y' => equation_a x y y') :=
by 
  sorry

theorem proof_equation_b_is_not_first_order_linear :
  ¬ is_first_order_linear (λ x y y' => equation_b x y y' 0) := 
by 
  sorry

theorem proof_equation_c_is_not_first_order_linear :
  ¬ is_first_order_linear (λ x y y' => equation_c x y y') := 
by 
  sorry

end proof_equation_a_is_first_order_linear_proof_equation_b_is_not_first_order_linear_proof_equation_c_is_not_first_order_linear_l364_364380


namespace sqrt_205_between_14_and_15_l364_364717

theorem sqrt_205_between_14_and_15 : 14 < Real.sqrt 205 ∧ Real.sqrt 205 < 15 := 
by
  sorry

end sqrt_205_between_14_and_15_l364_364717


namespace original_number_is_fraction_l364_364980

theorem original_number_is_fraction (x : ℚ) (h : 1 + (1 / x) = 9 / 4) : x = 4 / 5 :=
by
  sorry

end original_number_is_fraction_l364_364980


namespace non_pos_int_solutions_inequality_l364_364100

theorem non_pos_int_solutions_inequality : 
  (∃ n ∈ {-1, 0} : Set ℤ, 2 * n - 3 ≤ 5 * n) ∧
  (∀ n ∈ ℤ, 2 * n - 3 ≤ 5 * n → n ∈ {-1, 0}) :=
  by sorry

end non_pos_int_solutions_inequality_l364_364100


namespace students_A_and_C_spoke_the_truth_l364_364284

-- Define the classes
inductive Class
| one | two | three | four

-- Define the two winning classes
constant winning_classes : set Class

-- The statements made by students A, B, C, and D
def A_statement := winning_classes = {Class.two, Class.three, Class.four}
def B_statement := ¬ Class.two ∈ winning_classes ∧ Class.three ∈ winning_classes
def C_statement := Class.one ∈ winning_classes ↔ Class.four ∉ winning_classes
def D_statement := B_statement

-- Number of truthful statements
constant truthful_statements : fin 4 → Prop

-- Define the proof problem
theorem students_A_and_C_spoke_the_truth
  (h1 : |winning_classes| = 2)
  (h2 : ∃ p, set.count (λ x, x ∈ winning_classes) = p → p = 2)
  (h3 : truthful_statements ⅓ → A_statement ∧ C_statement):  
  truthful_statements (Fin.mk 0 (by decide)) ∧ truthful_statements (Fin.mk 2 (by decide)) :=
sorry

end students_A_and_C_spoke_the_truth_l364_364284


namespace max_value_of_u_is_zero_l364_364355

def max_value_of_u (x y : ℝ) : ℝ :=
  Real.logb (1/2) (8 * x * y + 4 * y^2 + 1)

theorem max_value_of_u_is_zero (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 2 * y = 1 / 2) : 
  max_value_of_u x y ≤ 0 :=
sorry

end max_value_of_u_is_zero_l364_364355


namespace probability_of_composite_l364_364490

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ m < n ∧ 1 < k ∧ k < n ∧ m * k = n

def dice_outcomes (faces : ℕ) (rolls : ℕ) : ℕ :=
  faces ^ rolls

def non_composite_product_ways : ℕ :=
  1 + (3 * 4)  -- one way for all 1s, plus combinations of (1,1,1,{2,3,5})

def total_outcomes : ℕ :=
  dice_outcomes 6 4  -- 6^4 total possible outcomes

def probability_composite : ℚ :=
  1 - (non_composite_product_ways / total_outcomes)

theorem probability_of_composite:
  probability_composite = 1283 / 1296 := 
by
  sorry

end probability_of_composite_l364_364490


namespace isosceles_trapezoid_circle_tangent_radius_l364_364018

theorem isosceles_trapezoid_circle_tangent_radius :
  let AB := 8
  let BC := 7
  let DA := 7
  let CD := 6
  let r := (24 * Math.sqrt 3 - 24) / 2
  let circle_A_radius := 4
  let circle_B_radius := 4
  let circle_C_radius := 3
  let circle_D_radius := 3
  in -24 + 24 + 3 + 2 = 53 := 
begin
  sorry
end

end isosceles_trapezoid_circle_tangent_radius_l364_364018


namespace least_possible_value_l364_364671

theorem least_possible_value (x y : ℝ) : (x + y - 1)^2 + (x * y)^2 ≥ 0 :=
by 
  sorry

end least_possible_value_l364_364671


namespace dice_product_composite_probability_l364_364482

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

-- This function calculates the probability of an event occurring by counting the favorable and total outcomes.
def probability (favorable total : ℕ) : ℚ :=
  favorable / total

noncomputable def probability_of_composite_product : ℚ :=
  probability 1283 1296

theorem dice_product_composite_probability : probability_of_composite_product = 1283 / 1296 := sorry

end dice_product_composite_probability_l364_364482


namespace find_actual_price_of_good_l364_364303

theorem find_actual_price_of_good 
  (final_price : ℝ) 
  (discounts : List ℝ) 
  (h_discounts : discounts = [0.30, 0.20, 0.15, 0.10, 0.05])
  (h_final_price : final_price = 10000) : 
  ∃ P : ℝ, P ≈ 24570.65 ∧ 
         final_price = P * (1 - 0.30) * (1 - 0.20) * (1 - 0.15) * (1 - 0.10) * (1 - 0.05) :=
by {
  -- proof goes here
  sorry
}

end find_actual_price_of_good_l364_364303


namespace determine_values_l364_364554

variable {n : ℕ} (a r x : Fin n → ℝ)

theorem determine_values (h : ∀ (x : Fin n → ℝ),
  ∑ k, r k * (x k - a k) ≤ (∑ k, (x k)^2)^(1/2) - (∑ k, (a k)^2)^(1/2)) :
  ∀ i, r i = a i / (∑ k, (a k)^2)^(1/2) :=
by sorry

end determine_values_l364_364554


namespace train_speed_l364_364295

theorem train_speed 
(length_of_train : ℕ) 
(time_to_cross_pole : ℕ) 
(h_length : length_of_train = 135) 
(h_time : time_to_cross_pole = 9) : 
  (length_of_train / time_to_cross_pole) * 3.6 = 54 :=
by 
  sorry

end train_speed_l364_364295


namespace exists_twelve_distinct_x_l364_364448

theorem exists_twelve_distinct_x :
  ∃ S : Set ℝ, (S.card = 12) ∧ (∀ x ∈ S, ∃ k : ℤ, 0 ≤ k ∧ k ≤ 11 ∧ (sqrt (123 - sqrt x) = k)) :=
by
  sorry

end exists_twelve_distinct_x_l364_364448


namespace find_y_l364_364684

theorem find_y 
  (x y : ℕ) 
  (h1 : x % y = 9) 
  (h2 : x / y = 96) 
  (h3 : (x % y: ℝ) / y = 0.12) 
  : y = 75 := 
  by 
    sorry

end find_y_l364_364684


namespace scott_invests_l364_364063

theorem scott_invests (x r : ℝ) (h1 : 2520 = x + 1260) (h2 : 2520 * 0.08 = x * r) : r = 0.16 :=
by
  -- Proof goes here
  sorry

end scott_invests_l364_364063
