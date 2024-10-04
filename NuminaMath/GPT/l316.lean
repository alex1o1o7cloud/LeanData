import Mathlib
import Mathlib.Algebra.AddGroup
import Mathlib.Algebra.Basic
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Log
import Mathlib.Algebra.Logarithm
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Algebra.Rational
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Calculus.ParametricIntegrals
import Mathlib.Analysis.Covering.Covers
import Mathlib.Analysis.Normed.Field.Basic
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Polynomial.Identities
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Probability.ProbabilityMassFunction
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Data.Set.Function
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Init.Data.List.Basic
import Mathlib.Logic.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMeasure
import Mathlib.Tactic
import Mathlib.Tactics.Basic
import Real

namespace swimming_pool_area_l316_316870

open Nat

-- Define the width (w) and length (l) with given conditions
def width (w : ℕ) : Prop :=
  exists (l : ℕ), l = 2 * w + 40 ∧ 2 * w + 2 * l = 800

-- Define the area of the swimming pool
def pool_area (w l : ℕ) : ℕ :=
  w * l

theorem swimming_pool_area : 
  ∃ (w l : ℕ), width w ∧ width l -> pool_area w l = 33600 :=
by
  sorry

end swimming_pool_area_l316_316870


namespace train_length_correct_l316_316567

noncomputable def length_of_first_train (speed1 speed2 : ℝ) (time : ℝ) (length2 : ℝ) : ℝ :=
  let speed1_ms := speed1 * 1000 / 3600
  let speed2_ms := speed2 * 1000 / 3600
  let relative_speed := speed1_ms - speed2_ms
  let total_distance := relative_speed * time
  total_distance - length2

theorem train_length_correct :
  length_of_first_train 72 36 69.99440044796417 300 = 399.9440044796417 :=
by
  sorry

end train_length_correct_l316_316567


namespace sales_profit_at_13_yuan_sales_price_increase_when_profit_360_l316_316547

variables (x y : ℕ) (purchase_price selling_price initial_sales_volume price_increase sales_volume profit : ℕ)

-- Define the conditions
def purchase_price := 8
def initial_selling_price := 10
def initial_sales_volume := 100
def sales_volume (x : ℕ) := initial_sales_volume - 10 * x
def price_increase (x : ℕ) := x
def profit (x : ℕ) := (price_increase x + (initial_selling_price - purchase_price)) * (sales_volume x)

-- Prove the daily sales profit when the selling price is 13 yuan (x = 3)
theorem sales_profit_at_13_yuan : profit 3 = 350 := by
  sorry

-- Prove the sales price increase when the profit is 360 yuan
theorem sales_price_increase_when_profit_360 : profit x = 360 → x = 4 := by
  sorry

end sales_profit_at_13_yuan_sales_price_increase_when_profit_360_l316_316547


namespace minjoo_siwoo_piggy_bank_days_eq_l316_316228

theorem minjoo_siwoo_piggy_bank_days_eq :
  let minjoo_initial := 12000
      minjoo_daily := 300
      siwoo_initial := 4000
      siwoo_daily := 500
      d := 40 in
  minjoo_initial + minjoo_daily * d = siwoo_initial + siwoo_daily * d :=
by
  let minjoo_initial := 12000
  let minjoo_daily := 300
  let siwoo_initial := 4000
  let siwoo_daily := 500
  let d := 40
  sorry

end minjoo_siwoo_piggy_bank_days_eq_l316_316228


namespace max_percentage_both_l316_316929

theorem max_percentage_both (P_WI : ℝ) (P_SN : ℝ) (h1 : P_WI = 0.45) (h2 : P_SN = 0.70) : 
  ∃ P_BOTH : ℝ, P_BOTH = 0.45 :=
by
  use 0.45
  sorry

end max_percentage_both_l316_316929


namespace mode_and_median_of_shoe_sizes_l316_316881

def shoe_data : List (ℝ × Nat) := [(25, 1), (25.5, 1), (26, 2), (26.5, 4), (27, 2)]

def mode (data : List (ℝ × Nat)) : ℝ :=
  data.map (Prod.snd).maxIdx.give (data.lookup (· = ·).get? 0.1)

def median (data : List (ℝ × Nat)) : ℝ :=
  let sorted_data := data.flatMap (λ ⟨size, count⟩ => List.replicate count size) |>.qsort (· ≤ ·) 
  let n := sorted_data.length
  if n % 2 = 0 then (sorted_data.get? (n / 2 - 1) + sorted_data.get? (n / 2)).get! / 2 else sorted_data.get! (n / 2)

theorem mode_and_median_of_shoe_sizes :
  mode shoe_data = 26.5 ∧ median shoe_data = 26.5 :=
by
  -- Proving that the mode of the shoe sizes is 26.5
  have h_mode : mode shoe_data = 26.5 := sorry
  -- Proving that the median of the shoe sizes is 26.5
  have h_median : median shoe_data = 26.5 := sorry
  exact ⟨h_mode, h_median⟩

end mode_and_median_of_shoe_sizes_l316_316881


namespace vector_subset_sum_length_exceeds_one_l316_316470

open Real

theorem vector_subset_sum_length_exceeds_one (n : ℕ) (a : Fin n → ℝ × ℝ)
  (h_sum_lengths : (Finset.univ.sum (λ i, (euclideanNorm (a i)))) = 4) :
  ∃ (S : Finset (Fin n)), (euclideanNorm (Finset.sum S a)) > 1 :=
by
  sorry

end vector_subset_sum_length_exceeds_one_l316_316470


namespace union_eq_interval_l316_316410

def A := { x : ℝ | 1 < x ∧ x < 4 }
def B := { x : ℝ | (x - 3) * (x + 1) ≤ 0 }

theorem union_eq_interval : (A ∪ B) = { x : ℝ | -1 ≤ x ∧ x < 4 } :=
by
  sorry

end union_eq_interval_l316_316410


namespace a_100_value_l316_316310

open Nat

-- Define the sequence a_n with the given conditions
def a : ℕ → ℕ
| 0       := 0  -- a_0 is not used, but defined for totality
| 1       := 2
| (n + 2) := a (n + 1) + 2 * (n + 1)

-- State the theorem we want to prove
theorem a_100_value : a 100 = 9902 := sorry

end a_100_value_l316_316310


namespace smaller_square_properties_l316_316335

noncomputable def side_length (area : ℝ) := Real.sqrt area

theorem smaller_square_properties (area_large_square : ℝ) (h : area_large_square = 144) :
  let s := side_length area_large_square in
  let d := s * Real.sqrt 2 in
  let side_small := d / 2 in
  (side_small = 6 * Real.sqrt 2) ∧ (side_small^2 = 72) :=
by
  sorry

end smaller_square_properties_l316_316335


namespace prove_a₈_l316_316830

noncomputable def first_term (a : ℕ → ℝ) : Prop := a 1 = 3
noncomputable def arithmetic_b (a b : ℕ → ℝ) : Prop := ∀ n, b n = a (n + 1) - a n
noncomputable def b_conditions (b : ℕ → ℝ) : Prop := b 3 = -2 ∧ b 10 = 12

theorem prove_a₈ (a b : ℕ → ℝ) (h1 : first_term a) (h2 : arithmetic_b a b) (h3 : b_conditions b) :
  a 8 = 3 :=
sorry

end prove_a₈_l316_316830


namespace cost_per_metre_is_6_97_l316_316869

-- Define the conditions provided in the problem statement
def length (b : ℝ) : ℝ := b + 20
def total_fencing_cost : ℝ := 5300
def given_length : ℝ := 200

-- Definition of the perimeter of a rectangle
def perimeter (length : ℝ) (breadth : ℝ) : ℝ := 2 * length + 2 * breadth

-- Proof problem stating that the fencing cost per metre is approximately Rs. 6.97
theorem cost_per_metre_is_6_97 (b : ℝ) (h1 : length b = given_length)
    (h2 : perimeter given_length b = 760) : total_fencing_cost / 760 = 6.97 := 
begin
  sorry
end

end cost_per_metre_is_6_97_l316_316869


namespace birth_year_1849_l316_316557

theorem birth_year_1849 (x : ℕ) (h1 : 1850 ≤ x^2 - 2 * x + 1) (h2 : x^2 - 2 * x + 1 < 1900) (h3 : x^2 - x + 1 = x) : x = 44 ↔ x^2 - 2 * x + 1 = 1849 := 
sorry

end birth_year_1849_l316_316557


namespace lawyer_rate_correct_l316_316771

def base_fine : ℕ := 50
def additional_penalty_fn (speed_over : ℕ) : ℕ := speed_over * 2
def speed_limit : ℕ := 30
def speed : ℕ := 75
def school_zone_multiplier : ℕ := 2
def court_costs : ℕ := 300
def total_fine_paid : ℕ := 820
def lawyer_hours : ℕ := 3

def fine_over_speed_limit (speed limit : ℕ) : ℕ :=
  fine + additional_penalty_fn (speed - limit)

def fine_in_school_zone (fine : ℕ) (multiplier : ℕ) : ℕ :=
  fine * multiplier

def total_fine
  (base_fine : ℕ) (additional_penalty : ℕ) (school_zone_multiplier : ℕ) (court_costs : ℕ) : ℕ :=
    (base_fine + additional_penalty) * school_zone_multiplier + court_costs

def lawyer_rate (hours : ℕ) (total_fine_paid total_fine: ℕ) : ℕ :=
  (total_fine_paid - total_fine) / hours

theorem lawyer_rate_correct :
  lawyer_rate lawyer_hours total_fine_paid
    (total_fine base_fine (additional_penalty_fn (speed - speed_limit)) school_zone_multiplier court_costs) = 80 :=
sorry

end lawyer_rate_correct_l316_316771


namespace characterization_of_a_l316_316271

variable (a b c d e : ℝ)

theorem characterization_of_a
  (h1 : be ≠ 0)
  (h2 : a / b < c / b - d / e) :
  ∃ x : ℝ, a = x :=
by
  revert a
  existsi a
  sorry

end characterization_of_a_l316_316271


namespace probability_sum_divisible_by_3_l316_316810

open Finset

-- Define the set of numbers
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define a predicated to check if the sum of a pair is divisible by 3
def sum_divisible_by_3 (a b : ℕ) : Prop := (a + b) % 3 = 0

-- Define the set of pairs (a, b) from S such that a < b
def pairs := (S.product S).filter (λab, ab.1 < ab.2)

-- Define the set of pairs (a, b) such that the sum of a and b is divisible by 3
def valid_pairs := pairs.filter (λp, sum_divisible_by_3 p.1 p.2)

-- Calculate the probability
theorem probability_sum_divisible_by_3 : 
  (card valid_pairs : ℚ) / (card pairs : ℚ) = 1 / 3 :=
by
  sorry

end probability_sum_divisible_by_3_l316_316810


namespace net_change_over_week_l316_316788

-- Definitions of initial quantities on Day 1
def baking_powder_day1 : ℝ := 4
def flour_day1 : ℝ := 12
def sugar_day1 : ℝ := 10
def chocolate_chips_day1 : ℝ := 6

-- Definitions of final quantities on Day 7
def baking_powder_day7 : ℝ := 2.5
def flour_day7 : ℝ := 7
def sugar_day7 : ℝ := 6.5
def chocolate_chips_day7 : ℝ := 3.7

-- Definitions of changes in quantities
def change_baking_powder : ℝ := baking_powder_day1 - baking_powder_day7
def change_flour : ℝ := flour_day1 - flour_day7
def change_sugar : ℝ := sugar_day1 - sugar_day7
def change_chocolate_chips : ℝ := chocolate_chips_day1 - chocolate_chips_day7

-- Statement to prove
theorem net_change_over_week : change_baking_powder + change_flour + change_sugar + change_chocolate_chips = 12.3 :=
by
  -- (Proof omitted)
  sorry

end net_change_over_week_l316_316788


namespace apples_remaining_l316_316797

-- Define the initial condition of the number of apples on the tree
def initial_apples : ℕ := 7

-- Define the number of apples picked by Rachel
def picked_apples : ℕ := 4

-- Proof goal: the number of apples remaining on the tree is 3
theorem apples_remaining : (initial_apples - picked_apples = 3) :=
sorry

end apples_remaining_l316_316797


namespace roque_commute_time_l316_316373

theorem roque_commute_time :
  let walk_time := 2
  let bike_time := 1
  let walks_per_week := 3
  let bike_rides_per_week := 2
  let total_walk_time := 2 * walks_per_week * walk_time
  let total_bike_time := 2 * bike_rides_per_week * bike_time
  total_walk_time + total_bike_time = 16 :=
by sorry

end roque_commute_time_l316_316373


namespace area_AMB_eq_l316_316786

open Real

variables {A B C D M : Point}
variables (α S1 S2 S : ℝ)

-- Definitions for orthogonality and segment conditions
variable (is_between : IsBetween A C D)
variable (perpendicular_AM_MD : Perpendicular (LineThrough A M) (LineThrough M D))
variable (perpendicular_CM_MB : Perpendicular (LineThrough C M) (LineThrough M B))

-- Definitions for given areas and angle
variable (angle_CMD : angle CAD = α)
variable (area_AMD : area (Triangle A M D) = S1)
variable (area_CMB : area (Triangle C M B) = S2)

-- The main statement to be proved
theorem area_AMB_eq :
  S = (1 / 2) * (S1 + S2 + sqrt ((S1 + S2)^2 - 4 * S1 * S2 * sin(α)^2)) :=
sorry

end area_AMB_eq_l316_316786


namespace max_balls_drawn_l316_316040

-- Conditions:
variable (items : Type) 
variable (Petya : items → Prop)
variable (yellow round edible : items → Prop)
variable (sun ball tomato banana : items)

variable (Ht : ∀ x, tomato x → round x ∧ ¬yellow x) -- All tomatoes are round and red
variable (Hb : ∀ x, banana x → yellow x ∧ ¬round x) -- All bananas are yellow and not round
variable (Hba : ∀ x, ball x → round x) -- All balls are round

variable (yellow_count : ∑ x in items, yellow x = 15) -- Exactly 15 yellow items
variable (round_count : ∑ x in items, round x = 18) -- Exactly 18 round items
variable (edible_count : ∑ x in items, edible x = 13) -- Exactly 13 edible items

-- Proving the maximum number of balls
theorem max_balls_drawn : ∑ x in items, ball x ≤ 18 :=
by sorry

end max_balls_drawn_l316_316040


namespace simplify_expression_1_simplify_expression_2_l316_316432

-- Problem (I)
theorem simplify_expression_1 : 
  (0 < 20 ∧ 20 < 45) → 
  (cos (20 * pi / 180) > 0 ∧ sin (20 * pi / 180) - cos (20 * pi / 180) < 0) → 
  (sqrt (1 - 2 * sin (20 * pi / 180) * cos (20 * pi / 180)) / (sin (20 * pi / 180) - cos (20 * pi / 180)) = -1) :=
by
  sorry

-- Problem (II)
theorem simplify_expression_2 (α : Real) : 
  (90 < α ∧ α < 180) → 
  (cos α < 0 ∧ sin α > 0) → 
  (cos α * sqrt ((1 - sin α) / (1 + sin α)) + sin α * sqrt ((1 - cos α) / (1 + cos α)) = sin α - cos α) :=
by
  sorry

end simplify_expression_1_simplify_expression_2_l316_316432


namespace standard_deviation_of_sample_is_correct_l316_316055

def sample_masses : List ℤ := [125, 124, 121, 123, 127]

def mean (lst : List ℤ) : Float :=
  (lst.sum : Float) / lst.length

def variance (lst : List ℤ) : Float :=
  let mv := mean lst
  (lst.foldl (λ acc x => acc + (x - mv)^2) 0 : Float) / (lst.length - 1)

def standard_deviation (lst : List ℤ) : Float :=
  Float.sqrt (variance lst)

theorem standard_deviation_of_sample_is_correct :
  standard_deviation sample_masses ≈ 2.24 := 
sorry

end standard_deviation_of_sample_is_correct_l316_316055


namespace compute_M_l316_316591

theorem compute_M : 
  let M := 150^2 + 148^2 - 146^2 - 144^2 + 142^2 + ...
          + 6^2 + 4^2 - 2^2 in M = 22800 :=
by
  -- Condition 1: Defining the sequence of sums of squares
  sorry

end compute_M_l316_316591


namespace people_receiving_roses_l316_316053

-- Defining the conditions.
def initial_roses : Nat := 40
def stolen_roses : Nat := 4
def roses_per_person : Nat := 4

-- Stating the theorem.
theorem people_receiving_roses : 
  (initial_roses - stolen_roses) / roses_per_person = 9 :=
by sorry

end people_receiving_roses_l316_316053


namespace bacteria_growth_time_l316_316825

theorem bacteria_growth_time :
  (∃ (t : ℕ), t * 5 = 20) :=
begin
  sorry
end

end bacteria_growth_time_l316_316825


namespace original_sticker_price_l316_316413

theorem original_sticker_price (x : ℝ) (h1 : store_x_price = 0.80 * x - 120) (h2 : store_y_price = 0.70 * x) (h3 : store_x_price + 25 = store_y_price) : x = 950 :=
by
  have eq1 : 0.80 * x - 120 + 25 = 0.70 * x := by rw [h1, h2, h3]
  rw [_root_.add_sub_cancel] at eq1
  rw [mul_sub] at eq1
  simp only [mul_one] at eq1
  let x_val := 95 / 0.10
  rw [←eq1]
  rw [eq_of_sub_eq_sub_eq_cancel] at x_val
  exact x_val

#lint

end original_sticker_price_l316_316413


namespace weight_of_one_bowling_ball_l316_316058

def weight_canoe := 36 -- The weight of one canoe in pounds.

def weight_of_n_canoes (n : ℕ) : ℕ := n * weight_canoe

def equivalent_weight_bowling_balls (canoe_count bowling_ball_count : ℕ) : Prop :=
  weight_of_n_canoes canoe_count = weight_of_n_bowling_balls bowling_ball_count

def weight_of_n_bowling_balls (n : ℕ) : ℕ := 24 * n
-- Given that one canoe weighs 36 pounds and six bowling balls weigh the same as four canoes
theorem weight_of_one_bowling_ball :
  equivalent_weight_bowling_balls 4 6 :=
by 
  have h : 4 * weight_canoe = 144 := by simp [weight_canoe]
  have h' : 6 * 24 = 144 := by simp
  rw [weight_of_n_canoes, weight_of_n_bowling_balls, h, h']
  sorry

end weight_of_one_bowling_ball_l316_316058


namespace least_possible_value_of_n_l316_316552

def radios_condition (n : ℕ) : Prop :=
  n > 3 ∧
  10 * n - 130 - 300 / n = 0

theorem least_possible_value_of_n : ∃ (n : ℕ), n = 15 ∧ radios_condition n := by
  use 15
  dsimp [radios_condition]
  split
  { exact 15 > 3 }
  { apply (10 * 15 - 130 - 300 / 15 = 0)
    sorry }

end least_possible_value_of_n_l316_316552


namespace nathaniel_wins_probability_l316_316004

def fair_six_sided_die : Type := {n : ℕ // n ≥ 1 ∧ n ≤ 6}

def probability_nathaniel_wins : ℚ :=
  have fair_die : fair_six_sided_die := sorry,
  have nathaniel_first : Prop := sorry,
  have win_condition (sum : ℕ) : Prop := sum % 7 = 0,

  if nathaniel_first ∧ ∀ sum. win_condition sum
  then 5 / 11
  else 0

theorem nathaniel_wins_probability :
  probability_nathaniel_wins = 5 / 11 :=
sorry

end nathaniel_wins_probability_l316_316004


namespace value_of_Y_is_669_l316_316319

theorem value_of_Y_is_669 :
  let A := 3009 / 3
  let B := A / 3
  let Y := A - B
  Y = 669 :=
by
  sorry

end value_of_Y_is_669_l316_316319


namespace amithab_january_expense_l316_316574

theorem amithab_january_expense :
  let avg_first_half := 4200
  let july_expense := 1500
  let avg_second_half := 4250
  let total_first_half_months := 6
  let total_second_half_months := 6
  let total_first_half_expense := total_first_half_months * avg_first_half
  let total_second_half_expense := total_second_half_months * avg_second_half
  let J := 1800
  total_second_half_expense - total_first_half_expense = 300 -> J - july_expense = 300 :=
begin
  sorry
end

end amithab_january_expense_l316_316574


namespace proof_problem_l316_316753

variables {a b c d : ℝ} (h1 : a ≠ -2) (h2 : b ≠ -2) (h3 : c ≠ -2) (h4 : d ≠ -2)
variable (ω : ℂ) (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
variable (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / ω)

theorem proof_problem : 
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 2 :=
sorry

end proof_problem_l316_316753


namespace max_distance_MN_l316_316465

-- Define the conditions
def circle_M (a : ℝ) : set (ℝ × ℝ) :=
  { p | let x := p.1, y := p.2 in x^2 + y^2 - 2 * a * x - 2 * a * y + 2 * a^2 - 2 = 0 }

def circle_O : set (ℝ × ℝ) :=
  { p | let x := p.1, y := p.2 in x^2 + y^2 = 18 }

def center_M (a : ℝ) : ℝ × ℝ := (a, a)

def point_N : ℝ × ℝ := (1, 2)

-- Prove that the maximum distance between the center of circle M and point N is √13
theorem max_distance_MN (a : ℝ) :
  (∀ p, p ∈ circle_M a ↔ p ∈ circle_O) → (∃ a : ℝ, 2 ≤ |a| ∧ |a| ≤ 4) →
  ∃ a, ∀ b, 2 ≤ |b| ∧ |b| ≤ 4 → dist (center_M a) point_N ≤ dist (center_M b) point_N :=
sorry

end max_distance_MN_l316_316465


namespace coefficient_of_x_in_binomial_expansion_l316_316098

theorem coefficient_of_x_in_binomial_expansion :
  (∃ n : ℕ, 2^n = 256 ∧ n = 8) → 
  (let T_r := λ r : ℕ, (Nat.choose 8 r) * (-3) ^ r * (x : ℝ) ^ r in 
  T_r 1 = -24) :=
by {
  sorry
}

end coefficient_of_x_in_binomial_expansion_l316_316098


namespace mod_congruence_l316_316695

theorem mod_congruence (x : ℤ) (h : 5 * x + 8 ≡ 3 [MOD 14]) : 3 * x + 10 ≡ 7 [MOD 14] :=
by sorry

end mod_congruence_l316_316695


namespace prod_gcd_lcm_eq_864_l316_316252

theorem prod_gcd_lcm_eq_864 : 
  let a := 24
  let b := 36
  let gcd_ab := Nat.gcd a b
  let lcm_ab := Nat.lcm a b
  gcd_ab * lcm_ab = 864 :=
by
  sorry

end prod_gcd_lcm_eq_864_l316_316252


namespace enlarge_garden_area_l316_316563

noncomputable def rectangular_length : ℝ := 40
noncomputable def rectangular_width : ℝ := 20
noncomputable def rectangular_area (l w : ℝ) : ℝ := l * w
noncomputable def perimeter (l w : ℝ) : ℝ := 2 * (l + w)
noncomputable def radius (P : ℝ) : ℝ := P / (2 * Real.pi)
noncomputable def circular_area (r : ℝ) : ℝ := Real.pi * r^2
noncomputable def area_increase (circular_area rectangular_area : ℝ) : ℝ := circular_area - rectangular_area

theorem enlarge_garden_area :
  area_increase (circular_area (radius (perimeter rectangular_length rectangular_width))) 
                (rectangular_area rectangular_length rectangular_width) 
  ≈ 345.92 :=
by
  sorry

end enlarge_garden_area_l316_316563


namespace verify_p_q_l316_316757

open Matrix

def N : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![-5, 2]]

def p : ℤ := 5
def q : ℤ := -26

theorem verify_p_q :
  N * N = p • N + q • (1 : Matrix (Fin 2) (Fin 2) ℤ) :=
by
  -- Skipping the proof
  sorry

end verify_p_q_l316_316757


namespace remainder_when_concat_numbers_1_to_54_div_55_l316_316388

def concat_numbers (n : ℕ) : ℕ :=
  let digits x := x.digits 10
  (List.range n).bind digits |> List.reverse |> List.foldl (λ acc x => acc * 10 + x) 0

theorem remainder_when_concat_numbers_1_to_54_div_55 :
  let M := concat_numbers 55
  M % 55 = 44 :=
by
  sorry

end remainder_when_concat_numbers_1_to_54_div_55_l316_316388


namespace obtain_2010_via_trig_functions_l316_316225

theorem obtain_2010_via_trig_functions:
  ∃ (f : ℝ → ℝ), 
    (f = λ x, (cot ∘ arctan ∘ sin) (arctan x)) ∨
    (f = λ x, (sin ∘ arctan) x) ∨
    (f = λ x, (cos ∘ arctan) x) →
    f 1 = 2010 :=
sorry

end obtain_2010_via_trig_functions_l316_316225


namespace find_sum_l316_316880

-- Defining the conditions of the problem
variables (P r t : ℝ) 
theorem find_sum 
  (h1 : (P * r * t) / 100 = 88) 
  (h2 : (P * r * t) / (100 + (r * t)) = 80) 
  : P = 880 := 
sorry

end find_sum_l316_316880


namespace find_k_and_general_term_sum_first_n_terms_l316_316674

noncomputable def sequence_sum (k : ℕ) (n : ℕ) : ℕ := k * 2^n - k
def a_2 : ℕ := 4
def a_n (n : ℕ) : ℕ := 2^n
def T_n (n : ℕ) : ℕ := (n-1) * 2^(n+1) + 2

theorem find_k_and_general_term :
  (∃ k : ℕ, 
    sequence_sum k 2 = 4 ∧ 
    ∀ n ≥ 2, a_n n = sequence_sum k n - sequence_sum k (n-1)
  ) ∧ ∀ n, a_n n = 2^n := sorry

theorem sum_first_n_terms (n : ℕ) :
  T_n n = 2 + (∑ i in finset.range n, (i+1) * 2^(i+1)) := sorry

end find_k_and_general_term_sum_first_n_terms_l316_316674


namespace probability_computation_l316_316496

def probability_area_less_than_circumference : ℚ :=
  let outcomes := [(x, y) | x ← [1, 2, 3, 4, 5, 6, 7, 8], y ← [1, 2, 3, 4, 5, 6, 7, 8]]
  let valid := [(x, y) | (x, y) ∈ outcomes, x + y < 4]
  ((valid.length : ℚ) / (outcomes.length : ℚ)) * (1 : ℚ)

theorem probability_computation :
  probability_area_less_than_circumference = 3 / 64 := 
  sorry

end probability_computation_l316_316496


namespace find_A_inv_and_M_l316_316627

variable (A : Matrix (Fin 2) (Fin 2) ℚ) (B : Matrix (Fin 2) (Fin 2) ℚ)

def A_def : A = ![![2, 0], ![-1, 1]] := by rfl
def B_def : B = ![![2, 4], ![3, 5]] := by rfl

theorem find_A_inv_and_M (A_inv M : Matrix (Fin 2) (Fin 2) ℚ) :
  A_inv = ![![1/2, 0], ![1/2, 1]] ∧ M = A_inv ⬝ B := by
  have hA : A = ![![2, 0], ![-1, 1]] := by exact A_def
  have hB : B = ![![2, 4], ![3, 5]] := by exact B_def
  sorry

end find_A_inv_and_M_l316_316627


namespace solve_system_eq_l316_316062

theorem solve_system_eq (x y z : ℝ) :
    (x^2 - y^2 + z = 64 / (x * y)) ∧
    (y^2 - z^2 + x = 64 / (y * z)) ∧
    (z^2 - x^2 + y = 64 / (x * z)) ↔ 
    (x = 4 ∧ y = 4 ∧ z = 4) ∨ 
    (x = -4 ∧ y = -4 ∧ z = 4) ∨ 
    (x = -4 ∧ y = 4 ∧ z = -4) ∨ 
    (x = 4 ∧ y = -4 ∧ z = -4) := by
  sorry

end solve_system_eq_l316_316062


namespace parabola_intersection_length_l316_316658

theorem parabola_intersection_length
  (parabola_eqn : ∀ x y : ℝ, y^2 = 8 * x)
  (focus : (ℝ × ℝ) := (2, 0))
  (A : (ℝ × ℝ) := (1, 2 * Real.sqrt 2))
  (l : ℝ → ℝ := fun x ↔ x - 2)
  (intersects : ∀ F : (ℝ × ℝ), F = focus → ∃ B : ℝ × ℝ, B ∈ parabola_eqn ∧ B ∈ l)
  (distance_AB : ∀ A B : (ℝ × ℝ), A = (1, 2 * Real.sqrt 2) ∧ exists_B B, |AB| = (Real.sqrt((B.1 - A.1)^2 + (B.2 - A.2)^2)) = 9) :
  True :=
begin
  sorry
end

end parabola_intersection_length_l316_316658


namespace sahil_selling_price_l316_316806

variable (purchasePrice : ℤ) (repairCosts : ℤ) (transportationCharges : ℤ) (profitPercentage : ℤ)

def totalCost := purchasePrice + repairCosts + transportationCharges
def profit := (profitPercentage * totalCost) / 100
def sellingPrice := totalCost + profit

theorem sahil_selling_price:
  purchasePrice = 9000 → 
  repairCosts = 5000 → 
  transportationCharges = 1000 → 
  profitPercentage = 50 → 
  sellingPrice = 22500 :=
by
  intros h1 h2 h3 h4
  unfold totalCost profit sellingPrice
  rw [h1, h2, h3, h4]
  sorry

end sahil_selling_price_l316_316806


namespace product_in_terms_of_sum_squares_and_ratio_l316_316942

variables (a r T P : ℝ) (n : ℕ)

def geometric_progression (a r : ℝ) (n : ℕ) : set ℝ :=
  {x | ∃ k : ℕ, (k < n) ∧ (x = a * r^k)}

def product_of_terms (a r : ℝ) (n : ℕ) : ℝ :=
  a ^ n * r ^ (n * (n - 1) / 2)

def sum_of_squares_of_terms (a r : ℝ) (n : ℕ) : ℝ :=
  a ^ 2 * (∑ k in finset.range n, r^(2 * k))

theorem product_in_terms_of_sum_squares_and_ratio
    (ha : a ≠ 0) (hr : r ≠ 1) (P : ℝ) (T : ℝ)
    (Hprod : P = product_of_terms a r n)
    (Hsum : T = sum_of_squares_of_terms a r n) :
  P = T ^ (n / 2) * ((1 - r ^ 2) / (1 - r ^ (2 * n))) ^ (n / 2) * r ^ (n * (n - 1) / 2) := 
by sorry

end product_in_terms_of_sum_squares_and_ratio_l316_316942


namespace altitude_length_l316_316333

variable (a b c : ℝ)
-- Let's state our conditions as hypotheses
-- Triangle ABC is a right triangle with AB as the hypotenuse.
-- The radius of the inscribed circle is 5/24.

open Real

def radius (a b c : ℝ) : ℝ := (a * b) / (a + b + c)

theorem altitude_length (h₁ : c = sqrt (a^2 + b^2))
  (h₂ : radius a b c = 5 / 24) :
  sqrt (c * (5 / 24)) - 2 * (5 / 24) = sqrt (5 * c / 24) - 5 / 12 :=
by {
  sorry
}

end altitude_length_l316_316333


namespace b_plus_c_div_a_l316_316488

-- Establish the setting
parameters {a b c d e : ℚ}

-- Given conditions
def polynomial := λ x : ℚ, a * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e
def root_cond_1 := polynomial 4 = 0
def root_cond_2 := polynomial (-3) = 0
def root_cond_3 := polynomial 0 = 0
def a_nonzero : Prop := a ≠ 0

-- The theorem to be proved
theorem b_plus_c_div_a : root_cond_1 ∧ root_cond_2 ∧ root_cond_3 ∧ a_nonzero → (b + c) = -13 * a := by
  intros
  sorry

end b_plus_c_div_a_l316_316488


namespace necessary_conditions_for_propositions_l316_316965

variable {x y a b c : ℝ}

theorem necessary_conditions_for_propositions :
  (∀ x y : ℝ, ((x > 10) → (x > 5))) ∧
  (∀ a b c : ℝ, (c ≠ 0 → ((ac = bc) → (a = b)))) ∧
  (∀ x y : ℝ, ((2x + 1 = 2y + 1) → (x = y))) :=
by
  sorry

end necessary_conditions_for_propositions_l316_316965


namespace team_B_eligible_l316_316127

-- Define the conditions
def max_allowed_height : ℝ := 168
def average_height_team_A : ℝ := 166
def median_height_team_B : ℝ := 167
def tallest_sailor_in_team_C : ℝ := 169
def mode_height_team_D : ℝ := 167

-- Define the proof statement
theorem team_B_eligible : 
  (∃ (heights_B : list ℝ), heights_B.length > 0 ∧ median heights_B = median_height_team_B) →
  (∀ h ∈ heights_B, h ≤ max_allowed_height) ∨ (∃ (S : finset ℝ), S.card ≥ heights_B.length / 2 ∧ ∀ h ∈ S, h ≤ max_allowed_height) :=
sorry

end team_B_eligible_l316_316127


namespace find_base_of_triangle_l316_316201

-- Given data
def perimeter : ℝ := 20 -- The perimeter of the triangle
def tangent_segment : ℝ := 2.4 -- The segment of the tangent to the inscribed circle contained between the sides

-- Define the problem and expected result
theorem find_base_of_triangle (a b c : ℝ) (P : a + b + c = perimeter)
  (tangent_parallel_base : ℝ := tangent_segment):
  a = 4 ∨ a = 6 :=
sorry

end find_base_of_triangle_l316_316201


namespace students_dont_eat_lunch_l316_316544

theorem students_dont_eat_lunch
  (total_students : ℕ)
  (students_in_cafeteria : ℕ)
  (students_bring_lunch : ℕ)
  (students_no_lunch : ℕ)
  (h1 : total_students = 60)
  (h2 : students_in_cafeteria = 10)
  (h3 : students_bring_lunch = 3 * students_in_cafeteria)
  (h4 : students_no_lunch = total_students - (students_in_cafeteria + students_bring_lunch)) :
  students_no_lunch = 20 :=
by
  sorry

end students_dont_eat_lunch_l316_316544


namespace harmonic_series_inequality_additional_terms_l316_316792

theorem harmonic_series_inequality_additional_terms (n : ℕ) (h : n > 0):
  (∑ i in finset.range (2^n - 1 + 1), (1 : ℝ) / i.succ) > (n : ℝ) / 2 →
  (∑ i in finset.range (2^(n+1) - 1 + 1), (1 : ℝ) / i.succ).card - (∑ i in finset.range (2^n - 1 + 1), (1 : ℝ) / i.succ).card = (2^n) :=
by sorry

end harmonic_series_inequality_additional_terms_l316_316792


namespace intersection_of_sets_l316_316748

-- Define sets A and B as given in the conditions
def A : Set ℝ := { x | -2 < x ∧ x < 2 }

def B : Set ℝ := {0, 1, 2}

-- Define the proposition to be proved
theorem intersection_of_sets : A ∩ B = {0, 1} :=
by
  sorry

end intersection_of_sets_l316_316748


namespace subsequences_property_l316_316610

noncomputable def valid_subsequences (n : ℕ) : list (list ℕ) :=
[(list.range (n + 1)).tail,
 (1 :: (list.range' 3 (n - 2)).tail ++ [2])]

theorem subsequences_property (n : ℕ) (a : list ℕ) (h₁ : a ⊆ (list.range (n + 1)).tail)
  (h₂ : ∀ i (h : i < n), 2 * (list.take (i + 1) a).sum % (i + 2) = 0) :
  a ∈ valid_subsequences n := by
  sorry

end subsequences_property_l316_316610


namespace largest_M_for_bernardo_wins_l316_316715

def game_step_bernardo (n : ℕ) : ℕ := 3 * n
def game_step_silvia (n : ℕ) : ℕ := n + 30

noncomputable def bernardo_wins (M : ℕ) : Prop :=
  M < 500 ∧ (game_step_bernardo M < 1500) ∧ (game_step_silvia (game_step_bernardo M) < 1500) ∧
  (game_step_bernardo (game_step_silvia (game_step_bernardo M)) < 1500) ∧
  (game_step_silvia (game_step_bernardo (game_step_silvia (game_step_bernardo M))) < 1500) ∧
  (game_step_bernardo (game_step_silvia (game_step_bernardo (game_step_silvia (game_step_bernardo M)))) < 1500) 

theorem largest_M_for_bernardo_wins : (∃ M : ℕ, (bernardo_wins M)) ∧ (40 = max {M | (bernardo_wins M)}) :=
sorry

end largest_M_for_bernardo_wins_l316_316715


namespace m_n_not_both_odd_l316_316762

open Nat

theorem m_n_not_both_odd (m n : ℕ) (h : (1 / m : ℚ) + (1 / n) = 1 / 2020) : ¬ (odd m ∧ odd n) :=
sorry

end m_n_not_both_odd_l316_316762


namespace exists_root_interval_l316_316068

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) - 2 / x

theorem exists_root_interval :
  ∃ x ∈ Ioo 1 2, f x = 0 :=
sorry

end exists_root_interval_l316_316068


namespace no_lunch_students_l316_316542

variable (total_students : ℕ) (cafeteria_eaters : ℕ) (lunch_bringers : ℕ)

theorem no_lunch_students : 
  total_students = 60 →
  cafeteria_eaters = 10 →
  lunch_bringers = 3 * cafeteria_eaters →
  total_students - (cafeteria_eaters + lunch_bringers) = 20 :=
by
  sorry

end no_lunch_students_l316_316542


namespace find_x_l316_316632

theorem find_x (x y z : ℝ) (h1 : y * z ≠ 0) (h2 : ({2 * x, 3 * z, x * y} : set ℝ) = {y, 2 * x * x, 3 * x * z}) : x = 1 :=
by
  sorry

end find_x_l316_316632


namespace rectangle_area_l316_316936

noncomputable def radius : ℝ := 7
noncomputable def width : ℝ := 2 * radius
noncomputable def length : ℝ := 3 * width
noncomputable def area (l w : ℝ) : ℝ := l * w

theorem rectangle_area : area length width = 588 := sorry

end rectangle_area_l316_316936


namespace principal_amount_invested_l316_316369

theorem principal_amount_invested :
  ∃ P : ℝ, 
    let r1 := 0.05 in
    let r2 := 0.04 in
    let r3 := 0.06 in
    let r4 := 0.07 in
    let A := 1120 in
    A = P * (1 + r1) * (1 + r2) * (1 + r3) * (1 + r4) ∧
    P ≈ 903.92 :=
begin
  sorry
end

end principal_amount_invested_l316_316369


namespace max_balls_possible_l316_316019

structure Conditions :=
  (yellow_objects : ℕ)
  (round_objects : ℕ)
  (edible_objects : ℕ)
  (all_objects : set string)
  (is_round : string → Prop)
  (is_yellow : string → Prop)
  (is_edible : string → Prop)
  (is_red : string → Prop)
  (p_types : list string)

namespace Problem
def PetyaConditions : Conditions :=
  {
    yellow_objects := 15,
    round_objects := 18,
    edible_objects := 13,
    all_objects := {"sun", "ball", "tomato", "banana"},
    is_round := λ x, x = "tomato" ∨ x = "ball",
    is_yellow := λ x, x = "banana" ∨ x = "ball",
    is_edible := λ x, x = "banana" ∨ x = "tomato",
    is_red := λ x, x = "tomato",
    p_types := ["sun", "ball", "tomato", "banana"]
  }

theorem max_balls_possible (cond : Conditions)
  (h1 : cond.yellow_objects = 15)
  (h2 : cond.round_objects = 18)
  (h3 : cond.edible_objects = 13)
  (h4 : ∀ x, x ∈ cond.all_objects → (cond.is_round x → ¬cond.is_yellow x → cond.is_edible x → ¬cond.is_red x))
  : ∃ n, n = 18 :=
by {
  sorry
}

end Problem

end max_balls_possible_l316_316019


namespace find_C_perimeter_range_l316_316339

-- Given conditions:
variables (A B C : ℝ) -- Angles
variables (a b c : ℝ) -- Sides opposite to angles A, B, C respectively
variables (ma mb : ℝ) -- Components of vector m
variables (na nb : ℝ) -- Components of vector n

axiom acute_triangle (h_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2) -- Acute triangle
axiom sides_opposite (ha : a = 3 * (sin A)) (hb : b = 3 * (sin B)) (hc : c = 3 / 2) -- Given sides opposite the angles
axiom vector_m (h_m : ma = (√3) * a ∧ mb = c)
axiom vector_n (h_n : na = sin A ∧ nb = cos C)
axiom vector_relation (h_vector : ma = 3 * na ∧ mb = 3 * nb)

-- Proof Statements:
theorem find_C : C = π / 3 := by 
  sorry

theorem perimeter_range : 
  (π / 6) < A ∧ A < (π / 2) →
  (sqrt 3 + 3) / 2 < a + b + c ∧ a + b + c ≤ 9 / 2 := by 
  sorry

end find_C_perimeter_range_l316_316339


namespace not_always_within_700_meters_always_within_800_meters_l316_316477

-- Define the width of the river as a constant
def river_width : ℝ := 1000
-- Define a predicate to express that a point is within x meters from the shore
def within_distance_from_shore (distance : ℝ) : Prop := ∀ point : ℝ, point ≤ distance

-- Problem (a): Prove it is not always possible to swim within 700 meters of each shore
theorem not_always_within_700_meters :
  ¬ (∀ point1 point2 : ℝ, within_distance_from_shore 700 point1 → within_distance_from_shore 700 point2 → point2 - point1 ≤ river_width) :=
sorry

-- Problem (b): Prove it is always possible to swim within 800 meters of each shore
theorem always_within_800_meters :
  ∀ point1 point2 : ℝ, within_distance_from_shore 800 point1 → within_distance_from_shore 800 point2 → point2 - point1 ≤ river_width :=
sorry

end not_always_within_700_meters_always_within_800_meters_l316_316477


namespace regression_sum_squares_l316_316701

theorem regression_sum_squares :
  ∀ (y : ℕ → ℝ) (y_hat : ℕ → ℝ) (y_bar : ℝ),
  (∑ i in finset.range 10, (y i - y_hat i)^2 = 120.55) →
  (0.95 = 1 - (∑ i in finset.range 10, (y i - y_hat i)^2) / 
                (∑ i in finset.range 10, (y i - y_bar)^2)) →
  (∑ i in finset.range 10, (y i - y_bar)^2 = 2411) :=
by
  intros
  sorry

end regression_sum_squares_l316_316701


namespace four_digit_numbers_div_by_5_with_34_end_l316_316679

theorem four_digit_numbers_div_by_5_with_34_end : 
  ∃ (count : ℕ), count = 90 ∧
  ∀ (n : ℕ), (1000 ≤ n ∧ n < 10000) →
  (n % 100 = 34) →
  ((10 ∣ n) ∨ (5 ∣ n)) →
  (count = 90) :=
sorry

end four_digit_numbers_div_by_5_with_34_end_l316_316679


namespace segments_ratio_l316_316924

variable (α : Real) (A B C M N : Point)
variable (triangle_ABC : Triangle A B C)
variable (BC : Real)
variable (AC : Real)
variable (ratio_BC_AC : BC = 3 * AC)
variable (angleACB : Angle B C A)
variable (angleACB_α : angleACB = α)
variable (trisect_rays_CM_CN : Trisects (angleACB, CM, CN))
variable (segment_CM_within_triangle : Segment C M)
variable (segment_CN_within_triangle : Segment C N)

theorem segments_ratio :
  (length (segment_CM_within_triangle)) / (length (segment_CN_within_triangle)) =
  (2 * cos (α / 3) + 3) / (6 * cos (α / 3) + 1) := sorry

end segments_ratio_l316_316924


namespace max_balls_l316_316018

theorem max_balls (total_yellow total_round total_edible : ℕ) 
  (suns balls tomatoes bananas : ℕ) :
  (total_yellow = 15) →
  (total_round = 18) →
  (total_edible = 13) →
  (tomatoes + balls ≤ total_round) →
  (tomatoes + bananas ≤ total_edible) →
  (suns + balls + tomatoes + bananas = total_yellow + total_round + total_edible) →
  (∀ b, b ∈ {balls, tomatoes, bananas, suns} → b ≥ 0) →
  (tomatoes ≤ total_round) →
  balls = 18 :=
by 
  sorry

end max_balls_l316_316018


namespace seven_digits_sum_example_l316_316508

-- Define a structure to represent the problem
structure SevenDigitsSum where
  a b c : ℕ
  unique_digits : (a.digits ∪ b.digits ∪ c.digits).card = 7
  digit_counts : a.digits.count = 4 ∧ b.digits.count = 2 ∧ c.digits.count = 1
  sum_eq : a + b + c = 2015

-- State the proof problem in Lean 4
theorem seven_digits_sum_example : SevenDigitsSum :=
  { a := 1987,
    b := 25,
    c := 3,
    unique_digits := sorry, -- proof to show digits are unique 
    digit_counts := sorry, -- proof to show counts of digits
    sum_eq := sorry } -- proof to show the sum is 2015

end seven_digits_sum_example_l316_316508


namespace max_balls_drawn_l316_316042

-- Conditions:
variable (items : Type) 
variable (Petya : items → Prop)
variable (yellow round edible : items → Prop)
variable (sun ball tomato banana : items)

variable (Ht : ∀ x, tomato x → round x ∧ ¬yellow x) -- All tomatoes are round and red
variable (Hb : ∀ x, banana x → yellow x ∧ ¬round x) -- All bananas are yellow and not round
variable (Hba : ∀ x, ball x → round x) -- All balls are round

variable (yellow_count : ∑ x in items, yellow x = 15) -- Exactly 15 yellow items
variable (round_count : ∑ x in items, round x = 18) -- Exactly 18 round items
variable (edible_count : ∑ x in items, edible x = 13) -- Exactly 13 edible items

-- Proving the maximum number of balls
theorem max_balls_drawn : ∑ x in items, ball x ≤ 18 :=
by sorry

end max_balls_drawn_l316_316042


namespace circle_square_properties_l316_316935

-- Define the conditions
def radius : ℝ := 6
def diameter : ℝ := 2 * radius
def side_length_of_square : ℝ := diameter
def area_of_square : ℝ := side_length_of_square ^ 2
def circumference_of_circle : ℝ := 2 * Real.pi * radius

-- The proof problem statement
theorem circle_square_properties :
  area_of_square = 144 ∧ circumference_of_circle = 12 * Real.pi := by
  sorry

end circle_square_properties_l316_316935


namespace ashton_pencils_left_l316_316972

theorem ashton_pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (pencils_given : ℕ) :
  boxes = 2 → pencils_per_box = 14 → pencils_given = 6 → (boxes * pencils_per_box) - pencils_given = 22 :=
by
  intros boxes_eq pencils_per_box_eq pencils_given_eq
  rw [boxes_eq, pencils_per_box_eq, pencils_given_eq]
  norm_num
  sorry

end ashton_pencils_left_l316_316972


namespace locus_of_R_eq_two_rotated_squares_l316_316409

/-- Definition of a square -/
structure Square (α : Type*) :=
(A B C D : α)

/-- Definition of an equilateral triangle -/
structure EquilateralTriangle (α : Type*) :=
(P Q R : α)

/-- The locus of points R as P moves along the perimeter of square ABCD and PQR is an equilateral triangle,
    is the union of two squares obtained by rotating ABCD by ±60° around Q. -/
theorem locus_of_R_eq_two_rotated_squares
  {α : Type*}
  (ABCD A1B1C1D1 A2B2C2D2 : Square α)
  (P Q R1 R2 : α)
  (hP : P ∈ [ABCD.A, ABCD.B, ABCD.C, ABCD.D])
  (hQ : Q ∉ [ABCD.A, ABCD.B, ABCD.C, ABCD.D])
  (h1 : EquilateralTriangle α (P, Q, R1))
  (h2 : EquilateralTriangle α (P, Q, R2))
  (h_rotate1 : A1B1C1D1.rotated 60 Q ABCD)
  (h_rotate2 : A2B2C2D2.rotated (-60) Q ABCD) :
  locus_of_R = A1B1C1D1 ∪ A2B2C2D2 :=
sorry

end locus_of_R_eq_two_rotated_squares_l316_316409


namespace smallest_sphere_radius_in_pyramid_l316_316925

noncomputable def radius_of_sphere (α : ℝ) (AB : ℝ) (area_BMC : ℝ) (area_ABC : ℝ) (BP_CP : ℝ) (volume_pyramid : ℝ) : ℝ :=
  if α = π/6 ∧ AB = 1 ∧ area_BMC = 2 * area_ABC ∧ BP_CP = sqrt 7 ∧ volume_pyramid = 3/4 then
    1/2
  else
    0 -- This case should never happen given the conditions

theorem smallest_sphere_radius_in_pyramid : 
  radius_of_sphere (π / 6) 1 (2 * (1/4 * 1 * α)) (1/4 * 1 * α) (sqrt 7) (3/4) = 1/2 :=
by
  rw radius_of_sphere
  split_ifs
  · -- The only case
    refl
  ·
    sorry -- This should never be reached

end smallest_sphere_radius_in_pyramid_l316_316925


namespace non_empty_subsets_satisfying_conditions_l316_316691

def count_subsets_20 : ℕ :=
  let n := 20 in
  let valid_ks := [1, 2, 3, 4, 5, 6, 7, 8] in
  valid_ks.map (λ k, Nat.choose (n - k + 1) k).sum

theorem non_empty_subsets_satisfying_conditions :
  count_subsets_20 = 2744 :=
by
  sorry

end non_empty_subsets_satisfying_conditions_l316_316691


namespace triangle_area_eq_l316_316641

noncomputable def areaOfTriangle (a b c A B C: ℝ): ℝ :=
1 / 2 * a * c * (Real.sin A)

theorem triangle_area_eq
  (a b c A B C : ℝ)
  (h1 : a = 2)
  (h2 : A = Real.pi / 3)
  (h3 : Real.sqrt 3 / 2 - Real.sin (B - C) = Real.sin (2 * B)) :
  areaOfTriangle a b c A B C = Real.sqrt 3 ∨ areaOfTriangle a b c A B C = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end triangle_area_eq_l316_316641


namespace distance_between_lines_equation_of_line3_l316_316307

-- Given conditions
def line1 (x y : ℝ) : Prop := x + 3*y + 18 = 0
def line2 (x y : ℝ) : Prop := 2*x + 3*y - 8 = 0
def mid_point (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Proof goal 1: Distance between line1 and line2
theorem distance_between_lines : ∀ x y : ℝ, 
  ∀ (line1 x y) (line2 x y),
  (abs ((-4) - 18) / sqrt (1^2 + 3^2)) = 2 * (sqrt 13) := 
sorry

-- Proof goal 2: Equation of line3
def line3 (x y : ℝ) : Prop := 2*x + 3*y + 5 = 0

theorem equation_of_line3 : 
  ∀ x y : ℝ,
  let A := (x, y) in
  let B := (x, y) in
  ∀ midpoint := mid_point A B,
  ∀ (H_A : line1 x y) (H_B : line2 x y),
  2* (midpoint.1) + 3* (midpoint.2) + 5 = 0 := 
sorry

end distance_between_lines_equation_of_line3_l316_316307


namespace probability_of_forming_triangle_l316_316483

noncomputable def can_form_triangle (a b c : ℕ) : Prop :=
a + b > c ∧ a + c > b ∧ b + c > a

noncomputable def count_valid_combinations : ℕ :=
[(2, 3, 4), (2, 3, 5), (2, 3, 7), (2, 4, 5), (2, 4, 7), (2, 5, 7),
 (3, 4, 5), (3, 4, 7), (3, 5, 7), (4, 5, 7)].countp (λ (x : ℕ × ℕ × ℕ), can_form_triangle x.1 x.2.1 x.2.2)

theorem probability_of_forming_triangle : (5 : ℝ) / 10 = 1 / 2 :=
by {
  have h : count_valid_combinations = 5,
  { refl },
  have total_combinations : ℕ := 10,
  simp [h, total_combinations],
  norm_num,
  sorry
}

end probability_of_forming_triangle_l316_316483


namespace nathaniel_wins_probability_l316_316003

/-- 
  Nathaniel and Obediah play a game where they take turns rolling a fair six-sided die 
  and keep a running tally. A player wins if the tally is a multiple of 7.
  If Nathaniel goes first, the probability that he wins is 5/11.
-/
theorem nathaniel_wins_probability :
  ∀ (die : ℕ → ℕ) (tally : ℕ → ℕ)
  (turn : ℕ → ℕ) (current_player : ℕ)
  (win_condition : ℕ → Prop),
  (∀ i, die i ∈ {1, 2, 3, 4, 5, 6}) →
  (∀ i, tally (i + 1) = tally i + die (i % 6)) →
  (win_condition n ↔ tally n % 7 = 0) →
  current_player 0 = 0 →  -- Nathaniel starts
  (turn i = if i % 2 = 0 then 0 else 1) →
  P(current_player wins) = 5/11 :=
by
  sorry

end nathaniel_wins_probability_l316_316003


namespace rounding_example_l316_316914

theorem rounding_example :
  let A := 65.179
  let B := 65.1777
  let C := 65.174999
  let D := 65.185
  let E := 65.18444
  C.round (2) ≠ 65.18 := 
by
  let A := 65.179
  let B := 65.1777
  let C := 65.174999
  let D := 65.185
  let E := 65.18444
  sorry

end rounding_example_l316_316914


namespace arithmetic_square_root_of_3_is_sqrt3_l316_316070

-- Given the condition that the arithmetic (or principal) square root is the non-negative square root
variable (a : ℝ)
variable (ha : 0 ≤ sqrt a)
variable (a_pos : 3 > 0)

theorem arithmetic_square_root_of_3_is_sqrt3 : sqrt 3 = real.sqrt 3 := by
  sorry

end arithmetic_square_root_of_3_is_sqrt3_l316_316070


namespace sequence_a_p_cubed_mod_p_l316_316634

theorem sequence_a_p_cubed_mod_p (p : ℕ) (hp : Nat.Prime p) (hp_gt : p > 3) :
  let a : ℕ → ℕ := λ n, if n < p then n else a (n - 1) + a (n - p)
  a (p ^ 3) % p = p - 1 :=
by
  sorry

end sequence_a_p_cubed_mod_p_l316_316634


namespace findChickens_l316_316414

def numCows : ℕ := 9

def numGoats (C : ℕ) : ℕ := 4 * numCows

def numChickens (G : ℕ) : ℕ := G / 2

def numDucks (C : ℕ) : ℕ := (3 * C) / 2

def sumAnimals (G C D : ℕ) : Bool := G + C + D ≤ 100

def conditionDivisible (D C : ℕ) : Bool := (D - 2 * C) % 3 == 0

theorem findChickens : ∃ C, 
  let G := numGoats C in 
  let D := numDucks C in
  C = numChickens G ∧
  sumAnimals G C D = true ∧ 
  conditionDivisible D C = true ∧
  C = 18 := 
by 
  sorry

end findChickens_l316_316414


namespace inequality_with_incenter_condition_l316_316793

theorem inequality_with_incenter_condition
  (A B C O : Point) (h_inside : PointInsideTriangle O A B C)
  (a b c : ℝ) (h_sides : TriangleSides A B C a b c) :
  let p := (a + b + c) / 2 in
  OA * Real.cos(A / 2) + OB * Real.cos(B / 2) + OC * Real.cos(C / 2) ≥ p ↔ PointIsIncenter O A B C :=
by
  sorry

end inequality_with_incenter_condition_l316_316793


namespace team_B_at_least_half_can_serve_l316_316123

-- Define the height limit condition
def height_limit (h : ℕ) : Prop := h ≤ 168

-- Define the team conditions
def team_A_avg_height : Prop := (160 + 169 + 169) / 3 = 166

def team_B_median_height (B : List ℕ) : Prop :=
  B.length % 2 = 1 ∧ B.perm ([167] ++ (B.eraseNth (B.length / 2))) ∧ B.nth (B.length / 2) = some 167

def team_C_tallest_height (C : List ℕ) : Prop :=
  ∀ (h : ℕ), h ∈ C → h ≤ 169

def team_D_mode_height (D : List ℕ) : Prop :=
  ∃ k, ∀ (h : ℕ), h ≠ 167 ∨ D.count 167 ≥ D.count h

-- Declare the main theorem to be proven
theorem team_B_at_least_half_can_serve (B : List ℕ) :
  (∀ h, h ∈ B → height_limit h) ↔ team_B_median_height B := sorry

end team_B_at_least_half_can_serve_l316_316123


namespace dot_product_NO_NM_equals_neg4_l316_316390

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 3

-- Define points N and O
def N : (ℝ × ℝ) := (0, 1)
def O : (ℝ × ℝ) := (0, 0)

-- Specify the condition for M and the given distance
def M_condition (M : ℝ × ℝ) : Prop :=
  ∃ (x : ℝ), 0 < x ∧ x < 2 ∧ M = (x, f x)

def OM_distance (M : ℝ × ℝ) : Prop :=
  let (Mx, My) := M in
  real.sqrt (Mx^2 + My^2) = 3 * real.sqrt 3

-- Define the vectors and the dot product
def vector_NO : (ℝ × ℝ) := (N.1 - O.1, N.2 - O.2)
def vector_NM (M : ℝ × ℝ) : (ℝ × ℝ) := (M.1 - N.1, M.2 - N.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- The proof statement
theorem dot_product_NO_NM_equals_neg4 (M : ℝ × ℝ) (hM_cond : M_condition M) (h_OM_dist : OM_distance M) :
  dot_product vector_NO (vector_NM M) = -4 := sorry

end dot_product_NO_NM_equals_neg4_l316_316390


namespace largest_possible_m_l316_316758

theorem largest_possible_m (n m : ℕ) (h : n > 1) :
  (∀ (commissions : Finset (Finset ℕ)), 
     (∀ K ∈ commissions, K.card = m) ∧ 
     (∀ x ∈ (⋃ K ∈ commissions, K), 
      (count_memberships x commissions).card = 2) ∧ 
     (∀ K1 K2 ∈ commissions, K1 ≠ K2 → K1 ∩ K2 = ∅))
  → m ≤ 2 * n - 1 :=
sorry

/-- Helper function to count the number of commissions a member is part of -/
def count_memberships (member : ℕ) (commissions : Finset (Finset ℕ)) : Finset (Finset ℕ) :=
  commissions.filter (λ K, member ∈ K)

end largest_possible_m_l316_316758


namespace conjugate_of_z_is_1_minus_2i_l316_316662

-- Define the complex number z
def z : ℂ := (3 + complex.I) / (1 - complex.I)

-- Define the complex conjugate of z
def z_conj : ℂ := complex.conj z

-- Theorem stating that the complex conjugate of z is 1 - 2i
theorem conjugate_of_z_is_1_minus_2i : z_conj = 1 - 2 * complex.I := by
  sorry

end conjugate_of_z_is_1_minus_2i_l316_316662


namespace football_count_white_patches_count_l316_316491

theorem football_count (x : ℕ) (footballs : ℕ) (students : ℕ) (h1 : students - 9 = footballs + 9) (h2 : students = 2 * footballs + 9) : footballs = 27 :=
sorry

theorem white_patches_count (white_patches : ℕ) (h : 2 * 12 * 5 = 6 * white_patches) : white_patches = 20 :=
sorry

end football_count_white_patches_count_l316_316491


namespace longest_line_segment_squared_is_162_l316_316540

noncomputable def diameter : ℝ := 18
def sector_count : ℕ := 4
def angle_in_degrees : ℝ := 360 / sector_count
def radius : ℝ := diameter / 2
noncomputable def longest_line_segment : ℝ := 2 * radius * Real.sin (angle_in_degrees / 2 * Real.pi / 180)
def l_squared := longest_line_segment ^ 2

theorem longest_line_segment_squared_is_162 :
  l_squared = 162 :=
sorry

end longest_line_segment_squared_is_162_l316_316540


namespace selection_of_books_l316_316352

-- Define the problem context and the proof statement
theorem selection_of_books (n k : ℕ) (h_n : n = 10) (h_k : k = 5) : nat.choose n k = 252 := by
  -- Given: n = 10, k = 5
  -- Prove: (10 choose 5) = 252
  rw [h_n, h_k]
  norm_num
  sorry

end selection_of_books_l316_316352


namespace second_player_wins_l316_316486

theorem second_player_wins 
  (pile1 : ℕ) (pile2 : ℕ) (pile3 : ℕ)
  (h1 : pile1 = 10) (h2 : pile2 = 15) (h3 : pile3 = 20) :
  (pile1 - 1) + (pile2 - 1) + (pile3 - 1) % 2 = 0 :=
by
  sorry

end second_player_wins_l316_316486


namespace max_balls_l316_316016

theorem max_balls (total_yellow total_round total_edible : ℕ) 
  (suns balls tomatoes bananas : ℕ) :
  (total_yellow = 15) →
  (total_round = 18) →
  (total_edible = 13) →
  (tomatoes + balls ≤ total_round) →
  (tomatoes + bananas ≤ total_edible) →
  (suns + balls + tomatoes + bananas = total_yellow + total_round + total_edible) →
  (∀ b, b ∈ {balls, tomatoes, bananas, suns} → b ≥ 0) →
  (tomatoes ≤ total_round) →
  balls = 18 :=
by 
  sorry

end max_balls_l316_316016


namespace max_norm_value_l316_316395

variables {V : Type*} [inner_product_space ℝ V]

def norm_sq (v : V) : ℝ := ⟪v, v⟫

theorem max_norm_value
  (a b c : V)
  (ha : norm a = 2)
  (hb : norm b = 3)
  (hc : norm c = 4)
  (hab : ⟪a, b⟫ = 0)
  (hbc : ⟪b, c⟫ = 0)
  (hac : ⟪a, c⟫ = 0) :
  norm_sq (a - 3 • b) + norm_sq (b - 3 • c) + norm_sq (c - 3 • a) = 290 :=
sorry

end max_norm_value_l316_316395


namespace students_average_vegetables_l316_316492

variable (points_needed : ℕ) (points_per_vegetable : ℕ) (students : ℕ) (school_days : ℕ) (school_weeks : ℕ)

def average_vegetables_per_student_per_week (points_needed points_per_vegetable students school_days school_weeks : ℕ) : ℕ :=
  let total_vegetables := points_needed / points_per_vegetable
  let vegetables_per_student := total_vegetables / students
  vegetables_per_student / school_weeks

theorem students_average_vegetables 
  (h1 : points_needed = 200) 
  (h2 : points_per_vegetable = 2) 
  (h3 : students = 25) 
  (h4 : school_days = 10) 
  (h5 : school_weeks = 2) : 
  average_vegetables_per_student_per_week points_needed points_per_vegetable students school_days school_weeks = 2 :=
by
  sorry

end students_average_vegetables_l316_316492


namespace line_through_points_l316_316078

theorem line_through_points (x₁ y₁ x₂ y₂ : ℝ) (h₁ : (x₁, y₁) = (2, -1)) (h₂ : (x₂, y₂) = (-1, 6)) :
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  m + b = 4 / 3 :=
by {
  rw [h₁, h₂],
  change m = (-7) / 3 ∧ b = 11 / 3,
  split,
  {
    apply congr_arg,
    norm_num,
  },
  {
    apply congr_arg,
    norm_num,
  }
}

end line_through_points_l316_316078


namespace grace_have_30_pastries_l316_316588

theorem grace_have_30_pastries (F : ℕ) :
  (2 * (F + 8) + F + (F + 13) = 97) → (F + 13 = 30) :=
by
  sorry

end grace_have_30_pastries_l316_316588


namespace max_balls_count_l316_316027

-- Definitions
def Object := Type
def sun : Object := sorry
def ball : Object := sorry
def tomato : Object := sorry
def banana : Object := sorry

def is_yellow : Object → Prop := sorry
def is_round : Object → Prop := sorry
def is_edible : Object → Prop := sorry

axiom yellow_items_count : ∃ (Y : set Object), Y.card = 15 ∧ ∀ y ∈ Y, is_yellow y
axiom round_items_count : ∃ (R : set Object), R.card = 18 ∧ ∀ r ∈ R, is_round r
axiom edible_items_count : ∃ (E : set Object), E.card = 13 ∧ ∀ e ∈ E, is_edible e

-- Problem conditions
axiom tomato_is_round_and_red : ∀ t, t = tomato → is_round t ∧ ¬is_yellow t
axiom ball_is_round : ∀ b, b = ball → is_round b
axiom banana_is_yellow_and_not_round : ∀ b, b = banana → is_yellow b ∧ ¬is_round b

-- Target proposition
theorem max_balls_count : 
  ∀ (sun_count ball_count tomato_count banana_count : ℕ),
    is_round sun → ¬is_yellow sun →
    is_round ball → is_edible ball → ¬is_yellow ball → 
    is_yellow tomato → is_round tomato → is_edible tomato →
    is_yellow banana → ¬is_round banana → is_edible banana →
    sun_count + ball_count + tomato_count + banana_count = 46 →
    ball_count ≤ 18 := sorry

end max_balls_count_l316_316027


namespace fold_point_area_sum_l316_316278

noncomputable def fold_point_area (AB AC : ℝ) (angle_B : ℝ) : ℝ :=
  let BC := Real.sqrt (AB ^ 2 + AC ^ 2)
  -- Assuming the fold point area calculation as per the problem's solution
  let q := 270
  let r := 324
  let s := 3
  q * Real.pi - r * Real.sqrt s

theorem fold_point_area_sum (AB AC : ℝ) (angle_B : ℝ) (hAB : AB = 36) (hAC : AC = 72) (hangle_B : angle_B = π / 2) :
  let S := fold_point_area AB AC angle_B
  ∃ q r s : ℕ, S = q * Real.pi - r * Real.sqrt s ∧ q + r + s = 597 :=
by
  sorry

end fold_point_area_sum_l316_316278


namespace complex_product_l316_316215

def complex_mult (a b : ℂ) : ℂ := (a * b)

theorem complex_product : complex_mult (1 + Complex.i) (2 - Complex.i) = 3 + Complex.i :=
by
  -- proof steps would go here
  sorry

end complex_product_l316_316215


namespace isosceles_triangle_perimeter_l316_316341

theorem isosceles_triangle_perimeter (a b : ℕ) (c : ℕ) 
  (h1 : a = 5) (h2 : b = 5) (h3 : c = 2) 
  (isosceles : a = b ∨ b = c ∨ c = a) 
  (triangle_inequality1 : a + b > c)
  (triangle_inequality2 : a + c > b)
  (triangle_inequality3 : b + c > a) : 
  a + b + c = 12 :=
  sorry

end isosceles_triangle_perimeter_l316_316341


namespace solution_polyhedron_volume_l316_316722

def polyhedron_volume_problem (A E F : Type) (B C D : Type) (G : Type) : Prop :=
  (∀ (a e f : A) (h_leg : ∀ x: A, isosceles_right_triangle x ∧ legs_length x = sqrt(2)),
     ∀ (b c d : B) (h_rect : ∀ x: B, rectangle x ∧ length x = 2 ∧ width x = 1),
     ∀ g : G (h_eq_tri: ∀ x: G, equilateral_triangle x ∧ side_length x = 2 * sqrt(2)),
     volume (polyhedron_formed_by_folding [a, e, f, b, c, d, g]) = 2)

theorem solution_polyhedron_volume : ∃ A E F B C D G, polyhedron_volume_problem A E F B C D G :=
by
  unfold polyhedron_volume_problem
  sorry

end solution_polyhedron_volume_l316_316722


namespace part1_part2_min_value_l316_316638

variable {R : Type} [LinearOrderedField R]

def quadratic_function (m x : R) : R := x^2 - m * x + m - 1

theorem part1 (h : quadratic_function m 0 = quadratic_function m 2) : m = 2 :=
by
  sorry

theorem part2_min_value (m : R) :
  ∃ min_val, min_val = if m ≤ ⟨-4⟩ then ⟨3 * m + 3⟩
                       else if -4 < m ∧ m < ⟨4⟩ then ⟨-m^2 / 4 + m - 1⟩
                       else ⟨3 - m⟩ :=
by
  sorry

end part1_part2_min_value_l316_316638


namespace total_boxes_packed_l316_316532

-- Definitions of the conditions
def initial_boxes : ℕ := 400
def food_value_per_box : ℕ := 80
def supplies_value_per_box : ℕ := 165
def donor_multiplier : ℕ := 4

-- Total value of one box
def total_value_per_box : ℕ := food_value_per_box + supplies_value_per_box

-- Initial spending
def initial_spending : ℕ := initial_boxes * total_value_per_box

-- Donation amount
def donation_amount : ℕ := donor_multiplier * initial_spending

-- Number of additional boxes packed with the donation
def additional_boxes : ℕ := donation_amount / total_value_per_box

-- Total number of boxes packed
def total_boxes : ℕ := initial_boxes + additional_boxes

-- Statement to be proven
theorem total_boxes_packed : total_boxes = 2000 := by
  -- Proof for this theorem goes here...
  -- The proof is omitted in this statement as requested.
  sorry

end total_boxes_packed_l316_316532


namespace ocean_depth_of_conical_hill_l316_316939

noncomputable def depth_of_ocean (h : ℝ) (volume_ratio : ℝ) : ℝ :=
  let submerged_height := h * real.cbrt (volume_ratio)
  in h - submerged_height

theorem ocean_depth_of_conical_hill :
  depth_of_ocean 5000 (4/5) = 347 :=
by
  sorry

end ocean_depth_of_conical_hill_l316_316939


namespace max_probability_sum_eleven_l316_316159

theorem max_probability_sum_eleven :
  let lst := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  let remove_one (l : List ℤ) (x : ℤ) := l.erase x in
  let pairs_summing_to_eleven (l : List ℤ) := (l.product l).filter (fun (p : ℤ × ℤ) => p.1 ≠ p.2 ∧ p.1 + p.2 = 11) in
  let remaining_pairs_after_removal := fun x => pairs_summing_to_eleven (remove_one lst x) in
  list.sum (list.map remaining_pairs_after_removal lst) ≤ list.sum (pairs_summing_to_eleven (remove_one lst 6)) :=
sorry

end max_probability_sum_eleven_l316_316159


namespace grasshoppers_positions_swap_l316_316107

theorem grasshoppers_positions_swap :
  ∃ (A B C: ℤ), A = -1 ∧ B = 0 ∧ C = 1 ∧
  (∀ m n p : ℤ, (A, B, C) = (m, n, p) → n = 0 → 
  (m^2 - n^2 + p^2 = 0) → A = 1 ∧ B = 0 ∧ C = -1) :=
begin
  -- Adding assumptions
  let x₁ := -1 : ℤ,
  let x₂ := 0 : ℤ,
  let x₃ := 1 : ℤ,
  existsi x₁, existsi x₂, existsi x₃,
  split, refl,
  split, refl,
  split, refl,
  intros m n p hperm hnze hyp,
  sorry -- the detailed proof will go here
end

end grasshoppers_positions_swap_l316_316107


namespace coeff_of_x_in_expansion_l316_316446

theorem coeff_of_x_in_expansion :
  let f := λ x : ℤ, (x^3 + 1) * (sqrt x - 2 / x)^5
  let binom_exp := λ r : ℕ, binomial 5 r * (-2)^r * x^((5 - 3*r)/2)
  (∑ r in (finset.range 6), binomial 5 r * (-2)^r * x^((5 - 3*r)/2)) = -90 :=
begin
  sorry
end

end coeff_of_x_in_expansion_l316_316446


namespace most_cost_effective_years_l316_316447

noncomputable def total_cost (x : ℕ) : ℝ := 100000 + 15000 * x + 1000 + 2000 * ((x * (x - 1)) / 2)

noncomputable def average_annual_cost (x : ℕ) : ℝ := total_cost x / x

theorem most_cost_effective_years : ∃ (x : ℕ), x = 10 ∧
  (∀ y : ℕ, y ≠ 10 → average_annual_cost x ≤ average_annual_cost y) :=
by
  sorry

end most_cost_effective_years_l316_316447


namespace prod_gcd_lcm_eq_864_l316_316251

theorem prod_gcd_lcm_eq_864 : 
  let a := 24
  let b := 36
  let gcd_ab := Nat.gcd a b
  let lcm_ab := Nat.lcm a b
  gcd_ab * lcm_ab = 864 :=
by
  sorry

end prod_gcd_lcm_eq_864_l316_316251


namespace finite_S_k_iff_k_power_of_2_l316_316618

def S_k_finite (k : ℕ) : Prop :=
  ∃ (n a b : ℕ), (n ≠ 0 ∧ n % 2 = 1) ∧ (a + b = k) ∧ (Nat.gcd a b = 1) ∧ (n ∣ (a^n + b^n))

theorem finite_S_k_iff_k_power_of_2 (k : ℕ) (h : k > 1) : 
  (∀ n a b, n ≠ 0 → n % 2 = 1 → a + b = k → Nat.gcd a b = 1 → n ∣ (a^n + b^n) → false) ↔ 
  ∃ m : ℕ, k = 2^m :=
by
  sorry

end finite_S_k_iff_k_power_of_2_l316_316618


namespace nathaniel_wins_probability_l316_316006

def fair_six_sided_die : Type := {n : ℕ // n ≥ 1 ∧ n ≤ 6}

def probability_nathaniel_wins : ℚ :=
  have fair_die : fair_six_sided_die := sorry,
  have nathaniel_first : Prop := sorry,
  have win_condition (sum : ℕ) : Prop := sum % 7 = 0,

  if nathaniel_first ∧ ∀ sum. win_condition sum
  then 5 / 11
  else 0

theorem nathaniel_wins_probability :
  probability_nathaniel_wins = 5 / 11 :=
sorry

end nathaniel_wins_probability_l316_316006


namespace value_of_m_squared_plus_reciprocal_squared_l316_316698

theorem value_of_m_squared_plus_reciprocal_squared 
  (m : ℝ) 
  (h : m + 1/m = 10) :
  m^2 + 1/m^2 + 4 = 102 :=
by {
  sorry
}

end value_of_m_squared_plus_reciprocal_squared_l316_316698


namespace find_x_given_y_l316_316888

variable (x y : ℝ)

theorem find_x_given_y :
  (0 < x) → (0 < y) → 
  (∃ k : ℝ, (3 * x^2 * y = k)) → 
  (y = 18 → x = 3) → 
  (y = 2400) → 
  x = 9 * Real.sqrt 6 / 85 :=
by
  -- Proof goes here
  sorry

end find_x_given_y_l316_316888


namespace find_g_neg6_l316_316834

-- Define the function g with conditions as hypotheses
variables {g : ℤ → ℤ}

-- Condition 1
axiom g_condition_1 : g 1 - 1 > 0

-- Condition 2
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x

-- Condition 3
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The goal is to find g(-6)
theorem find_g_neg6 : g (-6) = 723 :=
by {
  sorry
}

end find_g_neg6_l316_316834


namespace power_series_expansion_ln_l316_316238

theorem power_series_expansion_ln (x : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = ∑' n : ℕ, (λ n, if n = 0 then ln(2) else (-1)^(n+1) * (2^n + 1) / (2^n * n) * x^n) n) ∧ 
                (-1 < x ∧ x ≤ 1)) := 
  sorry

end power_series_expansion_ln_l316_316238


namespace hundredth_valid_ternary_term_l316_316971

def is_valid_ternary (n : ℕ) : Prop :=
  ∀ m, (n.bitwise m = 1) → (m = 1 ∨ m = 0)

def nth_valid_ternary (n : ℕ) : ℕ :=
  let binary_rep := Nat.binary_digits n in
  binary_rep.foldr (λ b acc, acc * 3 + b) 0

theorem hundredth_valid_ternary_term :
  nth_valid_ternary 100 = 981 :=
by sorry

end hundredth_valid_ternary_term_l316_316971


namespace example_problem_l316_316328

theorem example_problem (f : ℝ → ℝ)
    (h_diff : Differentiable ℝ f)
    (h_eq : ∀ x, f x = x^2 + 2 * (f' 1) * x + 3) :
    f 0 = f 4 :=
by
    sorry

end example_problem_l316_316328


namespace find_g_neg_six_l316_316864

theorem find_g_neg_six (g : ℤ → ℤ)
  (h1 : g 1 - 1 > 0)
  (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
  (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 :=
sorry

end find_g_neg_six_l316_316864


namespace selection_of_books_l316_316351

-- Define the problem context and the proof statement
theorem selection_of_books (n k : ℕ) (h_n : n = 10) (h_k : k = 5) : nat.choose n k = 252 := by
  -- Given: n = 10, k = 5
  -- Prove: (10 choose 5) = 252
  rw [h_n, h_k]
  norm_num
  sorry

end selection_of_books_l316_316351


namespace num_integers_abs_x_lt_5pi_l316_316682

theorem num_integers_abs_x_lt_5pi : 
    (finset.card {x : ℤ | abs x < 5 * real.pi} = 31) := 
    sorry

end num_integers_abs_x_lt_5pi_l316_316682


namespace sum_of_two_integers_l316_316075

noncomputable def sum_of_integers (a b : ℕ) : ℕ :=
a + b

theorem sum_of_two_integers (a b : ℕ) (h1 : a - b = 14) (h2 : a * b = 120) : sum_of_integers a b = 26 := 
by
  sorry

end sum_of_two_integers_l316_316075


namespace area_of_triangle_l316_316906

-- Define the lines as functions
def line1 : ℝ → ℝ := fun x => 3 * x - 4
def line2 : ℝ → ℝ := fun x => -2 * x + 16

-- Define the vertices of the triangle formed by lines and y-axis
def vertex1 : ℝ × ℝ := (0, -4)
def vertex2 : ℝ × ℝ := (0, 16)
def vertex3 : ℝ × ℝ := (4, 8)

-- Define the proof statement
theorem area_of_triangle : 
  let A := vertex1 
  let B := vertex2 
  let C := vertex3 
  -- Compute the area of the triangle using the determinant formula
  let area := (1 / 2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))
  area = 40 := 
by
  sorry

end area_of_triangle_l316_316906


namespace inequality_of_fractions_l316_316293

theorem inequality_of_fractions
  (f : ℝ → ℝ)
  (h : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 ≠ x2 → x2 * f(x1) - x1 * f(x2) < 0) :
  (f (2^(0.2))) / (2^(0.2)) > (f (Real.log 3 / Real.log Real.pi)) / (Real.log 3 / Real.log Real.pi) ∧
  (f (Real.log 3 / Real.log Real.pi)) / (Real.log 3 / Real.log Real.pi) > (f (Real.sin (Real.pi / 6))) / (Real.sin (Real.pi / 6)) :=
by
  sorry

end inequality_of_fractions_l316_316293


namespace abc_sum_16_l316_316694

theorem abc_sum_16 (a b c : ℕ) (h1 : a ≥ 4) (h2 : b ≥ 4) (h3 : c ≥ 4) (h4 : a ≠ b ∨ b ≠ c ∨ a ≠ c)
  (h5 : 4 * a * b * c = (a + 3) * (b + 3) * (c + 3)) : a + b + c = 16 :=
by
  sorry

end abc_sum_16_l316_316694


namespace midpoint_x_sum_l316_316885

variable {p q r s : ℝ}

theorem midpoint_x_sum (h : p + q + r + s = 20) :
  ((p + q) / 2 + (q + r) / 2 + (r + s) / 2 + (s + p) / 2) = 20 :=
by
  sorry

end midpoint_x_sum_l316_316885


namespace students_play_neither_l316_316919

-- Defining the problem parameters
def total_students : ℕ := 36
def football_players : ℕ := 26
def tennis_players : ℕ := 20
def both_players : ℕ := 17

-- Statement to be proved
theorem students_play_neither : (total_students - (football_players + tennis_players - both_players)) = 7 :=
by show total_students - (football_players + tennis_players - both_players) = 7; sorry

end students_play_neither_l316_316919


namespace part1_part2_min_value_l316_316639

variable {R : Type} [LinearOrderedField R]

def quadratic_function (m x : R) : R := x^2 - m * x + m - 1

theorem part1 (h : quadratic_function m 0 = quadratic_function m 2) : m = 2 :=
by
  sorry

theorem part2_min_value (m : R) :
  ∃ min_val, min_val = if m ≤ ⟨-4⟩ then ⟨3 * m + 3⟩
                       else if -4 < m ∧ m < ⟨4⟩ then ⟨-m^2 / 4 + m - 1⟩
                       else ⟨3 - m⟩ :=
by
  sorry

end part1_part2_min_value_l316_316639


namespace sally_cards_final_count_l316_316807

def initial_cards : ℕ := 27
def cards_from_Dan : ℕ := 41
def cards_bought : ℕ := 20
def cards_traded : ℕ := 15
def cards_lost : ℕ := 7

def final_cards (initial : ℕ) (from_Dan : ℕ) (bought : ℕ) (traded : ℕ) (lost : ℕ) : ℕ :=
  initial + from_Dan + bought - traded - lost

theorem sally_cards_final_count :
  final_cards initial_cards cards_from_Dan cards_bought cards_traded cards_lost = 66 := by
  sorry

end sally_cards_final_count_l316_316807


namespace monotonically_increasing_interval_l316_316871

-- Define the function f(x) as a piecewise function
def f (x : ℝ) : ℝ :=
  if x ≥ -2 then x + 2 else -x - 2

-- The proof statement
theorem monotonically_increasing_interval :
  ∀ x : ℝ, x ∈ Ici (-2) → monotone_on f (Ici (-2)) :=
by sorry

end monotonically_increasing_interval_l316_316871


namespace find_a11_times_a55_l316_316829

noncomputable theory

def arithmetic_sequence (a b c d e : ℤ) : Prop :=
  b - a = d - c ∧ c - b = e - d

def geometric_sequence (a b c d e : ℤ) : Prop :=
  b * b = a * c ∧ c * c = b * d ∧ d * d = c * e

variables {a : ℕ → ℕ → ℤ}

axioms (h_arith : ∀ i, 1 ≤ i ∧ i ≤ 5 → arithmetic_sequence (a i 1) (a i 2) (a i 3) (a i 4) (a i 5))
        (h_geom : ∀ j, 1 ≤ j ∧ j ≤ 5 → geometric_sequence (a 1 j) (a 2 j) (a 3 j) (a 4 j) (a 5 j))
        (h_same_ratio : ∀ j₁ j₂, 1 ≤ j₁ ∧ j₁ ≤ 5 → 1 ≤ j₂ ∧ j₂ ≤ 5 → 
                        (a 2 j₁) / (a 1 j₁) = (a 2 j₂) / (a 1 j₂))
        (h_a24 : a 2 4 = 4)
        (h_a41 : a 4 1 = -2)
        (h_a43 : a 4 3 = 10)

theorem find_a11_times_a55 : a 1 1 * a 5 5 = -11 :=
by sorry

end find_a11_times_a55_l316_316829


namespace trees_to_plant_l316_316481

variable (CurrentShortTrees : Nat) (TotalShortTreesAfterPlanting : Nat)

theorem trees_to_plant : CurrentShortTrees = 41 → TotalShortTreesAfterPlanting = 98 → TotalShortTreesAfterPlanting - CurrentShortTrees = 57 :=
by
  intros h1 h2
  rw [h1, h2]
  simp
  exact rfl

end trees_to_plant_l316_316481


namespace probability_consecutive_groups_l316_316932

theorem probability_consecutive_groups :
  let green_chips := 4
      blue_chips := 3
      red_chips := 5
      total_chips := green_chips + blue_chips + red_chips in
  (total_chips = 12) →
  let arrangements := Nat.factorial 3 * Nat.factorial green_chips * Nat.factorial blue_chips * Nat.factorial red_chips in
  let total_arrangements := Nat.factorial total_chips in
  (↑arrangements / ↑total_arrangements = (1 : ℚ) / 4620) :=
begin
  intros,
  sorry
end

end probability_consecutive_groups_l316_316932


namespace maximum_M_value_l316_316408

noncomputable def max_value_of_M : ℝ :=
  Real.sqrt 2 + 1 

theorem maximum_M_value {x y z : ℝ} (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) (hz1 : z ≤ 1) :
  Real.sqrt (abs (x - y)) + Real.sqrt (abs (y - z)) + Real.sqrt (abs (z - x)) ≤ max_value_of_M :=
by
  sorry

end maximum_M_value_l316_316408


namespace landmark_distance_l316_316958

theorem landmark_distance (d : ℝ) : 
  (d >= 7 → d < 7) ∨ (d <= 8 → d > 8) ∨ (d <= 10 → d > 10) → d > 10 :=
by
  sorry

end landmark_distance_l316_316958


namespace select_people_l316_316809

theorem select_people (boys girls : ℕ) (h_boys : boys = 4) (h_girls : girls = 3) :
  (∑ b in finset.range(4 + 1), choose boys b * choose girls (3 - b)) = 30 :=
by
  -- Given the number of boys and girls
  have h_b : boys = 4 := h_boys
  have h_g : girls = 3 := h_girls

  -- Reasoning about the sum of scenarios where 3 people are chosen
  -- ensuring at least one boy and one girl
  sorry

end select_people_l316_316809


namespace interest_rate_proof_l316_316468

theorem interest_rate_proof : 
    let P₁ := 5250 -- Principal for simple interest
    let R₁ := 4 -- Rate for simple interest
    let T₁ := 2 -- Time for simple interest
    let P₂ := 4000 -- Principal for compound interest
    let T₂ := 2 -- Time for compound interest
    let SI₁ := (P₁ * R₁ * T₁) / 100
    let CI₁ := 2 * SI₁
in CI₁ = P₂ * ((1 + (10 : ℝ)/100)^T₂ - 1) :=
by aka sorry

end interest_rate_proof_l316_316468


namespace largest_root_of_quadratic_l316_316231

theorem largest_root_of_quadratic :
  ∀ (x : ℝ), x^2 - 9*x - 22 = 0 → x ≤ 11 :=
by
  sorry

end largest_root_of_quadratic_l316_316231


namespace cubics_sum_div_abc_eq_three_l316_316398

theorem cubics_sum_div_abc_eq_three {a b c : ℝ} (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : a + b + c = 0) :
  (a^3 + b^3 + c^3) / (a * b * c) = 3 :=
by
  sorry

end cubics_sum_div_abc_eq_three_l316_316398


namespace binomial_expansion_terms_l316_316462

theorem binomial_expansion_terms (x y : ℝ) : 
  (x - y) ^ 10 = (11 : ℕ) :=
by
  sorry

end binomial_expansion_terms_l316_316462


namespace carrots_eaten_after_dinner_l316_316732

def carrots_eaten_before_dinner : ℕ := 22
def total_carrots_eaten : ℕ := 37

theorem carrots_eaten_after_dinner : total_carrots_eaten - carrots_eaten_before_dinner = 15 := by
  sorry

end carrots_eaten_after_dinner_l316_316732


namespace contest_B_third_place_4_competitions_l316_316346

/-- Given conditions:
1. There are three contestants: A, B, and C.
2. Scores for the first three places in each knowledge competition are \(a\), \(b\), and \(c\) where \(a > b > c\) and \(a, b, c ∈ ℕ^*\).
3. The final score of A is 26 points.
4. The final scores of both B and C are 11 points.
5. Contestant B won first place in one of the competitions.
Prove that Contestant B won third place in four competitions.
-/
theorem contest_B_third_place_4_competitions
  (a b c : ℕ)
  (ha : a > b)
  (hb : b > c)
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (hc_pos : 0 < c)
  (hA_score : a + a + a + a + b + c = 26)
  (hB_score : a + c + c + c + c + b = 11)
  (hC_score : b + b + b + b + c + c = 11) :
  ∃ n1 n3 : ℕ,
    n1 = 1 ∧ n3 = 4 ∧
    ∃ k m l p1 p2 : ℕ,
      n1 * a + k * a + l * a + m * a + p1 * a + p2 * a + p1 * b + k * b + p2 * b + n3 * c = 11 := sorry

end contest_B_third_place_4_competitions_l316_316346


namespace coefficient_x2_is_160_l316_316886

variable (n : ℕ)

/-- Condition that the sum of the coefficients of all terms in (2x + 1/(3x))^n is 729 -/
def sum_of_coeffs_eq_729 : Prop := (7 / 3 : ℚ) ^ n = 729

/-- General term of the expansion -/
def general_term (r : ℕ) : ℚ := binomial n r * 2^(n - r) * (1/3)^r

/-- Finding the coefficient of x^k in the expansion -/
def coefficient_of_x (k : ℕ) (n : ℕ) : ℚ := 
  ∑ r in Finset.range (n+1), if n - 2*r = k then general_term n r else 0

/-- The theorem stating that the coefficient of x^2 in (2x + 1/(3x))^n is 160 given the condition -/
theorem coefficient_x2_is_160 (h : sum_of_coeffs_eq_729 n) : coefficient_of_x 2 n = 160 := 
  sorry   -- Proof omitted

end coefficient_x2_is_160_l316_316886


namespace team_B_elibility_l316_316116

-- Define conditions as hypotheses
variables (avg_height_A : ℕ)
variables (median_height_B : ℕ)
variables (tallest_height_C : ℕ)
variables (mode_height_D : ℕ)
variables (max_height_allowed : ℕ)

-- Basic height statistics given in the problem
def team_A_statistics := avg_height_A = 166
def team_B_statistics := median_height_B = 167
def team_C_statistics := tallest_height_C = 169
def team_D_statistics := mode_height_D = 167

-- Height constraint condition
def height_constraint := max_height_allowed = 168

-- Mathematical equivalent proof problem: Prove that at least half of Team B sailors can serve
theorem team_B_elibility : height_constraint → team_B_statistics → (∀ (n : ℕ), median_height_B ≤ max_height_allowed) :=
by
  intros constraint_B median_B
  sorry

end team_B_elibility_l316_316116


namespace max_balls_count_l316_316026

-- Definitions
def Object := Type
def sun : Object := sorry
def ball : Object := sorry
def tomato : Object := sorry
def banana : Object := sorry

def is_yellow : Object → Prop := sorry
def is_round : Object → Prop := sorry
def is_edible : Object → Prop := sorry

axiom yellow_items_count : ∃ (Y : set Object), Y.card = 15 ∧ ∀ y ∈ Y, is_yellow y
axiom round_items_count : ∃ (R : set Object), R.card = 18 ∧ ∀ r ∈ R, is_round r
axiom edible_items_count : ∃ (E : set Object), E.card = 13 ∧ ∀ e ∈ E, is_edible e

-- Problem conditions
axiom tomato_is_round_and_red : ∀ t, t = tomato → is_round t ∧ ¬is_yellow t
axiom ball_is_round : ∀ b, b = ball → is_round b
axiom banana_is_yellow_and_not_round : ∀ b, b = banana → is_yellow b ∧ ¬is_round b

-- Target proposition
theorem max_balls_count : 
  ∀ (sun_count ball_count tomato_count banana_count : ℕ),
    is_round sun → ¬is_yellow sun →
    is_round ball → is_edible ball → ¬is_yellow ball → 
    is_yellow tomato → is_round tomato → is_edible tomato →
    is_yellow banana → ¬is_round banana → is_edible banana →
    sun_count + ball_count + tomato_count + banana_count = 46 →
    ball_count ≤ 18 := sorry

end max_balls_count_l316_316026


namespace total_distance_across_country_l316_316570

theorem total_distance_across_country (d_monday d_tuesday d_remaining total : ℕ) (h_monday : d_monday = 907) (h_tuesday : d_tuesday = 582) (h_remaining : d_remaining = 6716) (h_total : total = 8205) : 
  d_monday + d_tuesday + d_remaining = total :=
by
  rw [h_monday, h_tuesday, h_remaining, h_total]
  sorry

end total_distance_across_country_l316_316570


namespace solution_set_l316_316097

-- Define the two conditions as hypotheses
variables (x : ℝ)

def condition1 : Prop := x + 6 ≤ 8
def condition2 : Prop := x - 7 < 2 * (x - 3)

-- The statement to prove
theorem solution_set (h1 : condition1 x) (h2 : condition2 x) : -1 < x ∧ x ≤ 2 :=
by
  sorry

end solution_set_l316_316097


namespace rectangle_perimeter_l316_316822

theorem rectangle_perimeter 
  (a b : ℝ)
  (h_area : 3 * a^2 - 3 * a * b + 6 * a)
  (h_side1 : 3 * a) :
  2 * (h_side1 + (h_area / h_side1)) = 8 * a - 2 * b + 4 :=
by
  sorry

end rectangle_perimeter_l316_316822


namespace perimeter_triangle_AEC_l316_316565

noncomputable def solve_perimeter_problem (A B C D E C' : ℝ × ℝ)
    (AD BC AB : ℝ)
    (side_length : ℝ)
    (fold_condition1 : C' = (2, 4 / 3))
    (fold_condition2 : E = (3/2, 3/2))
    (fold_condition3 : AD = 2)
    (fold_condition4 : B = (0, 0))
    (fold_condition5 : A = (0, 2))
    (fold_condition6 : C = (2, 0))
    (fold_condition7 : D = (2, 2)) : ℝ :=
  let AE := real.sqrt (((3/2) - 0) ^ 2 + ((3/2) - 2) ^ 2)
  let EC' := real.sqrt (((2 - (3/2)) ^ 2) + ((4/3 - (3/2)) ^ 2))
  let AC' := 2
  AE + EC' + AC'

theorem perimeter_triangle_AEC'_is_correct (A B C D E C' : ℝ × ℝ)
    (AD : ℝ)
    (side_length : ℝ)
    (fold_condition1 : C' = (2, 4 / 3))
    (fold_condition2 : E = (3/2, 3/2))
    (fold_condition3 : AD = 2)
    (fold_condition4 : B = (0, 0))
    (fold_condition5 : A = (0, 2))
    (fold_condition6 : C = (2, 0))
    (fold_condition7 : D = (2, 2))
    (EC'_length : EC' = real.sqrt (10 / 36))
    (AE_length : AE = real.sqrt (10 / 4)) :
  solve_perimeter_problem A B C D E C' AD ((C.1 - B.1) * (B.2 - C.2))
  (side_length = 2) fold_condition1 fold_condition2 fold_condition3
  fold_condition4 fold_condition5 fold_condition6 fold_condition7 = (2 * real.sqrt(10) / 3 + 2) := sorry

end perimeter_triangle_AEC_l316_316565


namespace amithab_january_expense_l316_316573

theorem amithab_january_expense :
  let avg_first_half := 4200
  let july_expense := 1500
  let avg_second_half := 4250
  let total_first_half_months := 6
  let total_second_half_months := 6
  let total_first_half_expense := total_first_half_months * avg_first_half
  let total_second_half_expense := total_second_half_months * avg_second_half
  let J := 1800
  total_second_half_expense - total_first_half_expense = 300 -> J - july_expense = 300 :=
begin
  sorry
end

end amithab_january_expense_l316_316573


namespace fractional_lawn_remains_mowed_l316_316411

-- Defining the conditions as constants
constant Mary_mow_time : ℕ := 3
constant Tom_mow_time : ℕ := 6
constant Tom_work_time : ℕ := 3

-- Formulating the main theorem
theorem fractional_lawn_remains_mowed :
  let Tom_mowing_rate := 1 / Tom_mow_time
  let Tom_mowed := Tom_mowing_rate * Tom_work_time
  let remaining := 1 - Tom_mowed
  remaining = (1 / 2) :=
by
  sorry

end fractional_lawn_remains_mowed_l316_316411


namespace find_slope_of_l_l316_316593

noncomputable def parabola (x y : ℝ) := y ^ 2 = 4 * x

-- Definition of the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Definition of the point M
def M : ℝ × ℝ := (-1, 2)

-- Check if two vectors are perpendicular
def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Proof problem statement
theorem find_slope_of_l (x1 x2 y1 y2 k : ℝ)
  (h1 : parabola x1 y1)
  (h2 : parabola x2 y2)
  (h3 : is_perpendicular (x1 + 1, y1 - 2) (x2 + 1, y2 - 2))
  (eq1 : y1 = k * (x1 - 1))
  (eq2 : y2 = k * (x2 - 1)) :
  k = 1 := by
  sorry

end find_slope_of_l_l316_316593


namespace sum_of_squares_difference_l316_316987

theorem sum_of_squares_difference : 
  let sequence_sum := (∑ k in Finset.range 50, (2 * (50 - k))^2 - (2 * (50 - k) - 1)^2)
  in sequence_sum = 5050 :=
by
  sorry

end sum_of_squares_difference_l316_316987


namespace numbers_partitioned_into_pairs_l316_316608

noncomputable def can_be_partitioned (n : ℕ) : Prop :=
∃ pairs : List (ℕ × ℕ), 
  (∀ (p : ℕ × ℕ) ∈ pairs, p.1 + p.2 ∈ Finset.range (2 * n + 1)) ∧
  pairs.length = n ∧
  ∀ i j, i ≠ j → (i ∈ pairs → j ∈ pairs → i.fst ≠ j.fst ∧ i.snd ≠ j.snd) ∧ 
  (finset.univ = finset.univ.attach.map (λ x, x.1.1) ∪ finset.univ.attach.map (λ x, x.1.2)) ∧
  is_square (pairs.map (λ (p : ℕ × ℕ), p.1 + p.2)).prod

theorem numbers_partitioned_into_pairs (n : ℕ) (h : n > 1) : can_be_partitioned n := 
sorry

end numbers_partitioned_into_pairs_l316_316608


namespace min_a_for_inequality_l316_316619

theorem min_a_for_inequality :
  (∀ (x : ℝ), |x + a| - |x + 1| ≤ 2 * a) → a ≥ 1/3 :=
sorry

end min_a_for_inequality_l316_316619


namespace max_balls_drawn_l316_316032

structure DrawingConditions where
  yellow_items : ℕ
  round_items : ℕ
  edible_items : ℕ
  tomato_properties : Prop
  ball_properties : Prop
  banana_properties : Prop

axiom petya_conditions : DrawingConditions → Prop 

theorem max_balls_drawn (cond : DrawingConditions)
  (h1 : cond.yellow_items = 15)
  (h2 : cond.round_items = 18)
  (h3 : cond.edible_items = 13)
  (h4 : cond.tomato_properties)
  (h5 : cond.ball_properties)
  (h6 : cond.banana_properties) :
  ∀ balls : ℕ, balls ≤ 18 :=
by
  sorry

end max_balls_drawn_l316_316032


namespace find_april_decrease_l316_316708

noncomputable def stock_price_increase (initial_price : ℝ) (percent_inc : ℝ) : ℝ :=
  initial_price * (1 + percent_inc / 100)

noncomputable def stock_price_decrease (initial_price : ℝ) (percent_dec : ℝ) : ℝ :=
  initial_price * (1 - percent_dec / 100)

theorem find_april_decrease : 
  ∀ (S₀ : ℝ) (x : ℝ),
  let S₁ := stock_price_increase S₀ 25,
      S₂ := stock_price_decrease S₁ 15,
      S₃ := stock_price_increase S₂ 20,
      S₄ := stock_price_decrease S₃ x in
  S₀ = 100 → 
  S₄ = S₀ → 
  x = 22 :=
begin
  intros,
  sorry
end

end find_april_decrease_l316_316708


namespace students_in_favor_of_both_issues_is_117_l316_316583

open Set

variable (U : Type) [Fintype U]
variable (A B : Set U)
variable (total_students : Finset U)
variable (students_in_favor_of_first_issue students_in_favor_of_second_issue : Finset U)
variable (students_against_both_issues : Finset U)

noncomputable def number_of_students_in_favor_of_both_issues : ℕ :=
  students_in_favor_of_first_issue.card +
  students_in_favor_of_second_issue.card - 
  (total_students.card - students_against_both_issues.card)

theorem students_in_favor_of_both_issues_is_117
  (h₁ : total_students.card = 215)
  (h₂ : students_in_favor_of_first_issue.card = 160)
  (h₃ : students_in_favor_of_second_issue.card = 132)
  (h₄ : students_against_both_issues.card = 40) :
  number_of_students_in_favor_of_both_issues total_students students_in_favor_of_first_issue students_in_favor_of_second_issue students_against_both_issues = 117 := by
  sorry

end students_in_favor_of_both_issues_is_117_l316_316583


namespace probability_at_least_one_boy_and_girl_l316_316821

theorem probability_at_least_one_boy_and_girl :
  let total_members := 24
  let total_boys := 12
  let total_girls := 12
  let committee_size := 5
  let total_ways_form_committee := nat.choose total_members committee_size
  let ways_all_boys := nat.choose total_boys committee_size
  let ways_all_girls := nat.choose total_girls committee_size
  let ways_all_boys_or_all_girls := 2 * ways_all_boys
  let ways_with_at_least_one_boy_and_girl := total_ways_form_committee - ways_all_boys_or_all_girls
  let probability := (ways_with_at_least_one_boy_and_girl.to_rat / total_ways_form_committee.to_rat)
  in probability = (455 / 472 : ℚ) :=
sorry

end probability_at_least_one_boy_and_girl_l316_316821


namespace positive_difference_between_greatest_and_least_perimeters_l316_316175

def dimensions := (length width : ℝ)

def is_congruent (dims1 dims2 : dimensions) : Prop :=
  dims1.length * dims1.width = dims2.length * dims2.width

def possible_perimeters (dims : dimensions) : Set ℝ :=
  { p | ∃ L W : ℝ, (L * W = dims.length * dims.width) ∧
                    (L ≠ dims.length ∨ W ≠ dims.width) ∧
                    (L * W = dims.length * dims.width / 8) ∧
                    (p = 2 * (L + W)) }

theorem positive_difference_between_greatest_and_least_perimeters :
  ∃ (P : dimensions), 
    let perimeters := possible_perimeters P in
    Sup perimeters - Inf perimeters = 11.5 :=
begin
  -- ToDo: Proof to be completed
  sorry
end

end positive_difference_between_greatest_and_least_perimeters_l316_316175


namespace angle_inclination_of_fa_l316_316949

noncomputable def parabola (x : ℝ) : set (ℝ × ℝ) := { p | p.2^2 = 3 * p.1 }

def focus : ℝ × ℝ := (3/4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def angle_of_inclination (p1 p2 : ℝ × ℝ) : ℝ := Real.atan ((p2.2 - p1.2) / (p2.1 - p1.1))

theorem angle_inclination_of_fa (A : ℝ × ℝ) (hA : parabola A) (hD : distance A focus = 3) :
  angle_of_inclination focus A = π / 3 ∨ angle_of_inclination focus A = 2 * π / 3 :=
by
  sorry

end angle_inclination_of_fa_l316_316949


namespace team_B_eligible_l316_316128

-- Define the conditions
def max_allowed_height : ℝ := 168
def average_height_team_A : ℝ := 166
def median_height_team_B : ℝ := 167
def tallest_sailor_in_team_C : ℝ := 169
def mode_height_team_D : ℝ := 167

-- Define the proof statement
theorem team_B_eligible : 
  (∃ (heights_B : list ℝ), heights_B.length > 0 ∧ median heights_B = median_height_team_B) →
  (∀ h ∈ heights_B, h ≤ max_allowed_height) ∨ (∃ (S : finset ℝ), S.card ≥ heights_B.length / 2 ∧ ∀ h ∈ S, h ≤ max_allowed_height) :=
sorry

end team_B_eligible_l316_316128


namespace sum_powers_ω_equals_ω_l316_316993

-- Defining the complex exponential ω = e^(2 * π * i / 17)
def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 17)

-- Statement of the theorem
theorem sum_powers_ω_equals_ω : 
  (Finset.range 16).sum (λ k, Complex.exp (2 * Real.pi * Complex.I * (k + 1) / 17)) = ω :=
sorry

end sum_powers_ω_equals_ω_l316_316993


namespace square_folding_l316_316063

-- Defining the proof problem
theorem square_folding (PT : ℝ) (k : ℝ) (m : ℝ) (side_length : ℝ) (diagonal_length : ℝ) (fold_eq : diagonal_length = 2 * (PT + PT)) :
  side_length = 2 ∧ PT = (diagonal_length / 2) / sqrt 2 ∧ k = 4 ∧ m = 1.5 → (k + m = 5.5) :=
by {
  intros,
  sorry
}

end square_folding_l316_316063


namespace group9_40_41_right_angled_l316_316960

theorem group9_40_41_right_angled :
  ¬ (∃ a b c : ℝ, a = 3 ∧ b = 4 ∧ c = 7 ∧ a^2 + b^2 = c^2) ∧
  ¬ (∃ a b c : ℝ, a = 1/3 ∧ b = 1/4 ∧ c = 1/5 ∧ a^2 + b^2 = c^2) ∧
  ¬ (∃ a b c : ℝ, a = 4 ∧ b = 6 ∧ c = 8 ∧ a^2 + b^2 = c^2) ∧
  (∃ a b c : ℝ, a = 9 ∧ b = 40 ∧ c = 41 ∧ a^2 + b^2 = c^2) :=
by
  sorry

end group9_40_41_right_angled_l316_316960


namespace count_integers_in_interval_l316_316688

theorem count_integers_in_interval : 
  set.countable {x : ℤ | abs x < (5 * Real.pi)} = 31 :=
sorry

end count_integers_in_interval_l316_316688


namespace max_silver_coins_l316_316160

theorem max_silver_coins (n : ℕ) : (n < 150) ∧ (n % 15 = 3) → n = 138 :=
by
  sorry

end max_silver_coins_l316_316160


namespace spring_length_at_9kg_l316_316157

theorem spring_length_at_9kg :
  (∃ (k b : ℝ), (∀ x : ℝ, y = k * x + b) ∧ 
                 (y = 10 ∧ x = 0) ∧ 
                 (y = 10.5 ∧ x = 1)) → 
  (∀ x : ℝ, x = 9 → y = 14.5) :=
sorry

end spring_length_at_9kg_l316_316157


namespace figure_reconstruction_l316_316229

-- Definitions of provided figures
inductive Figure
| Figure1 : Figure
| Figure2 : Figure
| Figure3 : Figure

-- Definitions of parts after cutting
structure Parts where
  part1 : Figure
  part2 : Figure

-- Function to check if parts can form a given figure
def canForm (parts : Parts) (target : Figure) : Prop :=
  -- assume there exists a way to verify that parts can form the figure
  sorry

-- Given the initial conditions
axiom initial_conditions : ∃ (parts : Parts), canForm parts Figure2 ∧ canForm parts Figure3

-- The main theorem statement
theorem figure_reconstruction : ∃ (parts : Parts), canForm parts Figure2 ∧ canForm parts Figure3 :=
begin
  exact initial_conditions,
end

end figure_reconstruction_l316_316229


namespace exists_constant_not_geometric_l316_316088

-- Definitions for constant and geometric sequences
def is_constant_sequence (seq : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ n : ℕ, seq n = c

def is_geometric_sequence (seq : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, seq (n + 1) = r * seq n

-- The negation problem statement
theorem exists_constant_not_geometric :
  ∃ seq : ℕ → ℝ, is_constant_sequence seq ∧ ¬is_geometric_sequence seq :=
sorry

end exists_constant_not_geometric_l316_316088


namespace trigonometric_bounds_l316_316422

theorem trigonometric_bounds (x : ℝ) : 
  -4 ≤ cos (2 * x) + 3 * sin x ∧ cos (2 * x) + 3 * sin x ≤ 17 / 8 :=
sorry

end trigonometric_bounds_l316_316422


namespace equal_playing_time_l316_316815

-- Given conditions
def total_minutes : Nat := 120
def number_of_children : Nat := 6
def children_playing_at_a_time : Nat := 2

-- Proof problem statement
theorem equal_playing_time :
  (children_playing_at_a_time * total_minutes) / number_of_children = 40 :=
by
  sorry

end equal_playing_time_l316_316815


namespace minimum_cards_to_draw_to_ensure_2_of_each_suit_l316_316940

noncomputable def min_cards_to_draw {total_cards : ℕ} {suit_count : ℕ} {cards_per_suit : ℕ} {joker_count : ℕ}
  (h_total : total_cards = 54)
  (h_suits : suit_count = 4)
  (h_cards_per_suit : cards_per_suit = 13)
  (h_jokers : joker_count = 2) : ℕ :=
  43

theorem minimum_cards_to_draw_to_ensure_2_of_each_suit 
  (total_cards suit_count cards_per_suit joker_count : ℕ)
  (h_total : total_cards = 54) 
  (h_suits : suit_count = 4)
  (h_cards_per_suit : cards_per_suit = 13)
  (h_jokers : joker_count = 2) : 
  min_cards_to_draw h_total h_suits h_cards_per_suit h_jokers = 43 :=
  by
  sorry

end minimum_cards_to_draw_to_ensure_2_of_each_suit_l316_316940


namespace light_bulb_no_explosion_configurations_l316_316501

-- Define the lattice points with coordinates (±1, ±1, ±1) which are √3 units away from the origin.
def lattice_points : List (ℤ × ℤ × ℤ) :=
  [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), 
   (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]

-- Define a predicate to check if two points are at most 2 units apart.
def at_most_two_units_apart (p1 p2 : ℤ × ℤ × ℤ) : Prop :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  (x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2 ≤ 4

-- Define a predicate to check if a given configuration (list of on/off states) causes an explosion.
def causes_explosion (configuration : List Bool) : Prop :=
  lattice_points.enum.any (λ (i1, p1) =>
    lattice_points.enum.any (λ (i2, p2) =>
      i1 ≠ i2 ∧ configuration.get i1 = true ∧ configuration.get i2 = true ∧ at_most_two_units_apart p1 p2))

-- Define the main theorem to state the desired result.
theorem light_bulb_no_explosion_configurations :
  ∃ n, n = 23 ∧ ∀ configuration : List Bool,
    configuration.length = 8 →
    (¬ causes_explosion configuration) →
    (number_of_such_configurations configuration = n) :=
by sorry

end light_bulb_no_explosion_configurations_l316_316501


namespace composite_divides_factorial_l316_316406

open BigOperators

def largestPrimeLessThan (k : ℕ) : ℕ := 
  if k ≤ 2 then 2 else (Nat.filter Nat.prime (List.range (k-1))).maximum

def isComposite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 2 ≤ a ∧ a ≤ b ∧ n = a * b

theorem composite_divides_factorial 
  (k n : ℕ) 
  (hk : k ≥ 14) 
  (P_k := largestPrimeLessThan k) 
  (hP₁ : P_k < k) 
  (hP₂ : P_k ≥ 3 * k / 4) 
  (hcomp : isComposite n) 
  (hn : n > 2 * P_k) : 
  n ∣ ( (n - k)! ) :=
sorry

end composite_divides_factorial_l316_316406


namespace sum_of_black_angles_eq_180_mountain_valley_difference_l316_316514
-- Part (a)

theorem sum_of_black_angles_eq_180 
  (angles : Finset ℝ) 
  (coloring : ℝ → Prop) 
  (h1 : ∀ x ∈ angles, coloring x ∨ ¬coloring x) 
  (h2 : Finset.card ({x ∈ angles | coloring x}) = Finset.card ({x ∈ angles | ¬coloring x})) : 
  ∑ x in ({x ∈ angles | ¬coloring x} : Finset ℝ), x = 180 := 
sorry

-- Part (b)

theorem mountain_valley_difference
  (n m : ℕ)
  (mountains valleys : Finset ℕ)
  (h1 : Finset.card mountains = n)
  (h2 : Finset.card valleys = m)
  (h3 : n + m > 0):
  (n - m).abs = 2 :=
sorry

end sum_of_black_angles_eq_180_mountain_valley_difference_l316_316514


namespace james_carrot_sticks_l316_316733

def carrots_eaten_after_dinner (total_carrots : ℕ) (carrots_before_dinner : ℕ) : ℕ :=
  total_carrots - carrots_before_dinner

theorem james_carrot_sticks : carrots_eaten_after_dinner 37 22 = 15 := by
  sorry

end james_carrot_sticks_l316_316733


namespace roden_total_fish_l316_316424

def total_goldfish : Nat :=
  15 + 10 + 3 + 4

def total_blue_fish : Nat :=
  7 + 12 + 7 + 8

def total_green_fish : Nat :=
  5 + 9 + 6

def total_purple_fish : Nat :=
  2

def total_red_fish : Nat :=
  1

def total_fish : Nat :=
  total_goldfish + total_blue_fish + total_green_fish + total_purple_fish + total_red_fish

theorem roden_total_fish : total_fish = 89 :=
by
  unfold total_fish total_goldfish total_blue_fish total_green_fish total_purple_fish total_red_fish
  sorry

end roden_total_fish_l316_316424


namespace birdhouse_flown_distance_l316_316473

-- Definition of the given conditions.
def car_distance : ℕ := 200
def lawn_chair_distance : ℕ := 2 * car_distance
def birdhouse_distance : ℕ := 3 * lawn_chair_distance

-- Statement of the proof problem.
theorem birdhouse_flown_distance : birdhouse_distance = 1200 := by
  sorry

end birdhouse_flown_distance_l316_316473


namespace max_balls_count_l316_316030

-- Definitions
def Object := Type
def sun : Object := sorry
def ball : Object := sorry
def tomato : Object := sorry
def banana : Object := sorry

def is_yellow : Object → Prop := sorry
def is_round : Object → Prop := sorry
def is_edible : Object → Prop := sorry

axiom yellow_items_count : ∃ (Y : set Object), Y.card = 15 ∧ ∀ y ∈ Y, is_yellow y
axiom round_items_count : ∃ (R : set Object), R.card = 18 ∧ ∀ r ∈ R, is_round r
axiom edible_items_count : ∃ (E : set Object), E.card = 13 ∧ ∀ e ∈ E, is_edible e

-- Problem conditions
axiom tomato_is_round_and_red : ∀ t, t = tomato → is_round t ∧ ¬is_yellow t
axiom ball_is_round : ∀ b, b = ball → is_round b
axiom banana_is_yellow_and_not_round : ∀ b, b = banana → is_yellow b ∧ ¬is_round b

-- Target proposition
theorem max_balls_count : 
  ∀ (sun_count ball_count tomato_count banana_count : ℕ),
    is_round sun → ¬is_yellow sun →
    is_round ball → is_edible ball → ¬is_yellow ball → 
    is_yellow tomato → is_round tomato → is_edible tomato →
    is_yellow banana → ¬is_round banana → is_edible banana →
    sun_count + ball_count + tomato_count + banana_count = 46 →
    ball_count ≤ 18 := sorry

end max_balls_count_l316_316030


namespace solve_complex_magnitude_l316_316657

noncomputable def complex_magnitude_problem
  (z1 z2 : ℂ) (hz1 : complex.abs z1 = 2) (hz2 : complex.abs z2 = 3) (angle_60 : ℝ := 60) (cos_60 : ℝ := (1/2)) : Prop :=
  complex.abs ((z1 + z2) / (z1 - z2)) = real.sqrt (19) / real.sqrt (7)

theorem solve_complex_magnitude : complex_magnitude_problem z1 z2 := 
  sorry

end solve_complex_magnitude_l316_316657


namespace profit_in_percentage_is_75_l316_316525

def cost_price : ℝ := 32
def selling_price : ℝ := 56

def profit_percentage (cp sp : ℝ) : ℝ :=
  ((sp - cp) / cp) * 100

theorem profit_in_percentage_is_75 :
  profit_percentage cost_price selling_price = 75 :=
by
  sorry

end profit_in_percentage_is_75_l316_316525


namespace range_h_x_equiv_range_k_equiv_l316_316669

open Real

noncomputable def f (x : ℝ) := 3 - 2 * log 2 x
noncomputable def g (x : ℝ) := log 2 x
noncomputable def h (x : ℝ) := (4 - 2 * log 2 x) * log 2 x

-- Prove that the range of h for x in [1, 4] is [0, 2]
theorem range_h_x_equiv {x : ℝ} (hx : 1 ≤ x ∧ x ≤ 4) : 
  (0 ≤ h x ∧ h x ≤ 2) :=
  sorry

-- Prove the range of k under f(x^2) * f(sqrt x) > k * g(x) for x in [1, 4]
theorem range_k_equiv {x : ℝ} (hx : 1 ≤ x ∧ x ≤ 4) (k: ℝ) :
  (f (x^2) * f (sqrt x) > k * g x → k ∈ Iio (-3)) :=
  sorry

end range_h_x_equiv_range_k_equiv_l316_316669


namespace find_second_candy_cost_l316_316545

theorem find_second_candy_cost :
  ∃ (x : ℝ), 
    (15 * 8 + 30 * x = 45 * 6) ∧
    x = 5 := by
  sorry

end find_second_candy_cost_l316_316545


namespace angle_DAB_fixed_l316_316705

-- Define the given conditions and prove that the angle is fixed at 45 degrees.

variable {A B C D E F : Type} 
variables [Between C A B] [Between C B D] [Between C A F]

theorem angle_DAB_fixed :
  ∀ (CA CB AB : ℝ) (angle_CDE angle_CAF : ℝ),
  CA ≠ CB →
  AB > BC →
  square CBDE →
  equilateral CAF →
  angle_CDE = 90 →
  angle_CAF = 60 →
  ∠ D A B = 45 :=
by
  intros CA CB AB angle_CDE angle_CAF hCAneqCB hABgtBC hSquareCBDE hEquilateralCAF hAngleCDE hAngleCAF
  -- The detailed steps are omitted as this is only the statement.
  sorry

end angle_DAB_fixed_l316_316705


namespace integer_sided_triangle_with_60_degree_angle_exists_l316_316579

theorem integer_sided_triangle_with_60_degree_angle_exists 
  (m n t : ℤ) : 
  ∃ (x y z : ℤ), (x = (m^2 - n^2) * t) ∧ 
                  (y = m * (m - 2 * n) * t) ∧ 
                  (z = (m^2 - m * n + n^2) * t) := by
  sorry

end integer_sided_triangle_with_60_degree_angle_exists_l316_316579


namespace part_a_part_b_l316_316518

-- Definition of part (a)
def flip_one_checker_6x6_possible : Prop :=
  ∃ i j : ℕ, i < 6 ∧ j < 6 ∧ (
    ∀ i' j', (i' = i ∨ j' = j) → (i' ≠ i ∨ j' ≠ j) → 
    (i', j') results in a flip (flipping twice returns to original state))

-- Definition of part (b)
def all_checkers_white_from_initial_half_black (m n : ℕ) (initial_black : ℕ) : Prop :=
  (m = 5 ∧ n = 6 ∧ initial_black = (m * n) / 2) →
  ¬ (∀ i j, 0 < i ∧ i ≤ m ∧ 0 < j ∧ j ≤ n → checker (i, j) = white)

-- Theorems
theorem part_a : flip_one_checker_6x6_possible := sorry

theorem part_b : all_checkers_white_from_initial_half_black 5 6 15 := sorry

end part_a_part_b_l316_316518


namespace each_person_received_5_l316_316189

theorem each_person_received_5 (S n : ℕ) (hn₁ : n > 5) (hn₂ : 5 * S = 2 * n * (n - 5)) (hn₃ : 4 * S = n * (n + 4)) :
  S / (n + 4) = 5 :=
by
  sorry

end each_person_received_5_l316_316189


namespace total_marbles_l316_316336

theorem total_marbles (y b g : ℝ) (h1 : y = 1.4 * b) (h2 : g = 1.75 * y) :
  b + y + g = 3.4643 * y :=
sorry

end total_marbles_l316_316336


namespace ratio_of_areas_l316_316140

-- Define the right triangle ABC with A at (0,0), B at (1,0), and C at (0,1)
def point := (ℝ × ℝ)
def A : point := (0, 0)
def B : point := (1, 0)
def C : point := (0, 1)

-- Define the midpoint M of A and B
def M : point := (1 / 2, 0)

-- Define the speeds of the particles
def speed_A_to_B := 1 -- Assume speed v translated to relative speed units
def speed_M_to_A := 2 -- Assume speed 2v translated to relative speed units

-- The problem statement in Lean 4
theorem ratio_of_areas : 
  let R_area := (1 / 4) * (1 / 2)
  let ABC_area := (1 / 2)
  R_area / ABC_area = 1 / 4 :=
by 
  sorry

end ratio_of_areas_l316_316140


namespace less_than_half_l316_316904

theorem less_than_half (a b c : ℝ) (h₁ : a = 43.2) (h₂ : b = 0.5) (h₃ : c = 42.7) : a - b = c := by
  sorry

end less_than_half_l316_316904


namespace pigeonhole_fir_trees_l316_316438

theorem pigeonhole_fir_trees (n_trees needles_upper_bound : ℕ) 
  (h_trees : n_trees = 710000) 
  (h_needles : needles_upper_bound = 100000) : 
  ∃ k, (k ≥ 7 ∧ ∃ t : ℕ → ℕ, (∀ i, t i ≤ needles_upper_bound) ∧ (finset.card (finset.image t (finset.range n_trees)) < n_trees) ∧ (finset.card (finset.image t (finset.range k)) = n_trees)) :=
by {
  sorry
}

end pigeonhole_fir_trees_l316_316438


namespace brick_width_proof_l316_316184

theorem brick_width_proof (
  length_courtyard : ℕ,
  width_courtyard : ℕ,
  length_brick : ℕ,
  total_bricks : ℕ,
  total_area : ℕ,
  area_eq : total_area = 2500 * 1800,
  total_area_bricks : total_area = (length_brick * width_brick) * total_bricks,
  brick_dimensions : length_brick = 20,
  num_bricks : total_bricks = 22500
) : width_brick = 10 := by
  sorry

end brick_width_proof_l316_316184


namespace zoo_structure_l316_316206

theorem zoo_structure (P : ℕ) (h1 : ∃ (snakes monkeys elephants zebras : ℕ),
  snakes = 3 * P ∧
  monkeys = 6 * P ∧
  elephants = (P + snakes) / 2 ∧
  zebras = elephants - 3 ∧
  monkeys - zebras = 35) : P = 8 :=
sorry

end zoo_structure_l316_316206


namespace rationalize_denominator_l316_316803

theorem rationalize_denominator (a : ℝ) (n : ℕ) (h : a = 7) (hn : n = 3) :
  7 / real.cbrt (7 ^ 3) = 1 := 
by
  sorry

end rationalize_denominator_l316_316803


namespace maximize_net_profit_l316_316957

section ShenyangMetro

variable (t : ℝ) (p : ℝ → ℝ) (Q : ℝ → ℝ)

-- Definition of passenger capacity function p
def passenger_capacity : ℝ → ℝ :=
  λ t, if 10 ≤ t ∧ t ≤ 20 then 1300 else 1300 - 10 * (10 - t)^2

-- Definition of net profit function Q
def net_profit_per_minute : ℝ → ℝ :=
  λ t, (6 * passenger_capacity t - 3960) / t - 350

theorem maximize_net_profit : 2 ≤ t ∧ t ≤ 20 → net_profit_per_minute t = 130 :=
  by intros h; sorry

end ShenyangMetro

end maximize_net_profit_l316_316957


namespace correct_statements_count_is_one_l316_316397

-- Define the conditions of the problem
variable (a : Line) (α : Plane)

-- Formalize each of the three statements as conditions
def statement1 : Prop := ∃! (β : Plane), a ⊆ β ∧ β ⊥ α
def statement2 : Prop := ∃ (b : Line), b ⊆ α ∧ b ⊥ a
def statement3 : Prop := ∀ (P Q : Point), (P ∈ a ∧ Q ∈ a ∧ dist_point_plane P α = 1 ∧ dist_point_plane Q α = 1) → a ∥ α

-- Define the problem to check if there is exactly one correct statement
def number_of_correct_statements : Nat :=
  (if statement1 then 1 else 0) + (if statement2 then 1 else 0) + (if statement3 then 1 else 0)

-- Prove that the number of correct statements is 1
theorem correct_statements_count_is_one (h1 : ¬ statement1) (h2 : statement2) (h3 : ¬ statement3) :
  number_of_correct_statements a α = 1 := by
  sorry

end correct_statements_count_is_one_l316_316397


namespace minimum_time_for_cars_tunnel_passage_l316_316440

theorem minimum_time_for_cars_tunnel_passage :
  ∀ x > 0, x ≤ 25,
  let y : ℝ :=
    if x ≤ 12 then 3480 / x
    else 5 * x + 2880 / x + 10
  in
  ∃ (y_min x_min : ℝ),
  y_min = 250 ∧ x_min = 24 ∧
  (∀ (x' > 0) (x' ≤ 25), 
    let y' : ℝ :=
      if x' ≤ 12 then 3480 / x'
      else 5 * x' + 2880 / x' + 10
    in y' ≥ y_min) := 
sorry

end minimum_time_for_cars_tunnel_passage_l316_316440


namespace probability_total_greater_than_7_l316_316920

open ProbabilityTheory

-- Define a throwing of two dice and compute the probability of getting a total > 7
def num_faces : ℕ := 6
def total_outcomes : ℕ := num_faces * num_faces

def favorable_outcomes : ℕ :=
  have roll_results : List (ℕ × ℕ) := [(3, 5), (3, 6),
                                        (4, 4), (4, 5), (4, 6),
                                        (5, 3), (5, 4), (5, 5), (5, 6),
                                        (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)]
  roll_results.length

theorem probability_total_greater_than_7 :
  (favorable_outcomes / total_outcomes : ℚ) = 7 / 18 := by
  -- Here you will provide the proof steps
  sorry

end probability_total_greater_than_7_l316_316920


namespace jack_emails_morning_l316_316730

-- Definitions from conditions
def emails_evening : ℕ := 7
def additional_emails_morning : ℕ := 2
def emails_morning : ℕ := emails_evening + additional_emails_morning

-- The proof problem
theorem jack_emails_morning : emails_morning = 9 := by
  -- proof goes here
  sorry

end jack_emails_morning_l316_316730


namespace sum_of_solutions_eq_neg4_l316_316433

theorem sum_of_solutions_eq_neg4 :
  ∑ x in {x : ℝ | 4^(x^2 + 6 * x + 9) = 16^(x + 3)}, x = -4 := by
  sorry

end sum_of_solutions_eq_neg4_l316_316433


namespace find_k_proof_l316_316944

noncomputable def find_k : ℕ := 43

theorem find_k_proof : 
  ∀ (k : ℕ), 
  (∃ k, (k, 23) ∈ set_of (λ (p : ℕ × ℤ), line_through_point (1, -5) p ∥ line_with_eq (4*x + 6*y = 30))) →
  k = find_k := 
by
  intro k,
  simp,
  intro h_parallel_lines,
  -- detailed proof skipped; assuming it is correct
  sorry

end find_k_proof_l316_316944


namespace largest_pillar_radius_correct_l316_316549

def crate_length := 12
def crate_width := 8
def crate_height := 6

def largest_pillar_radius (crate_length crate_width crate_height : ℝ) : ℝ :=
  crate_width / 2

theorem largest_pillar_radius_correct :
  largest_pillar_radius crate_length crate_width crate_height = 4 :=
by
  unfold largest_pillar_radius
  simp
  /- split and rearrange to keep the logic contained -/
  sorry

end largest_pillar_radius_correct_l316_316549


namespace part1_part2_l316_316276

-- Define the hypotheses
def complex_z (a : ℝ) : ℂ := (a^2 - 4) + (a + 2) * complex.I

-- Proof for part (Ⅰ): If z is pure imaginary
theorem part1 (a : ℝ) (h : complex_z a = (0 : ℝ) + (a + 2) * complex.I) : a = 2 :=
by sorry

-- Proof for part (Ⅱ): If z lies on the line x + 2y + 1 = 0
theorem part2 (a : ℝ) (h : (a^2 - 4) + 2 * (a + 2) + 1 = 0) : a = -1 :=
by sorry

end part1_part2_l316_316276


namespace nathaniel_wins_probability_l316_316002

/-- 
  Nathaniel and Obediah play a game where they take turns rolling a fair six-sided die 
  and keep a running tally. A player wins if the tally is a multiple of 7.
  If Nathaniel goes first, the probability that he wins is 5/11.
-/
theorem nathaniel_wins_probability :
  ∀ (die : ℕ → ℕ) (tally : ℕ → ℕ)
  (turn : ℕ → ℕ) (current_player : ℕ)
  (win_condition : ℕ → Prop),
  (∀ i, die i ∈ {1, 2, 3, 4, 5, 6}) →
  (∀ i, tally (i + 1) = tally i + die (i % 6)) →
  (win_condition n ↔ tally n % 7 = 0) →
  current_player 0 = 0 →  -- Nathaniel starts
  (turn i = if i % 2 = 0 then 0 else 1) →
  P(current_player wins) = 5/11 :=
by
  sorry

end nathaniel_wins_probability_l316_316002


namespace matrix_determinant_l316_316222

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![[3, 1, -2],
    [8, 5, -4],
    [3, 3, 6]]

theorem matrix_determinant : det A = 48 := by 
  sorry

end matrix_determinant_l316_316222


namespace correct_statement_c_l316_316961

variable (ξ : Type) [DiscreteRandomVariable ξ] (E_ξ : ℝ) (D_ξ : ℝ)

-- An example predicate to capture the condition of expectation reflecting average level
def expectation_reflects_average : Prop :=
  ∀ (x : ξ) (p : ℝ), E_ξ = ∑ i, (x_i * p_i)

theorem correct_statement_c
  (hE : ∀ (x : ξ) (p : ℝ), E_ξ = ∑ i, (x_i * p_i))
  (hD : ∀ (x : ξ) (p : ℝ), D_ξ = ∑ i, ((x_i - E_ξ)^2 * p_i)) :
  expectation_reflects_average ξ E_ξ :=
by
  sorry

end correct_statement_c_l316_316961


namespace tan_beta_formula_l316_316629

theorem tan_beta_formula (α β : ℝ) 
  (h1 : Real.tan α = -2/3)
  (h2 : Real.tan (α + β) = 1/2) :
  Real.tan β = 7/4 :=
sorry

end tan_beta_formula_l316_316629


namespace sequence_properties_l316_316640

namespace SequenceProblem

-- Definitions from conditions in the problem
def a (n : ℕ) : ℚ :=
  if n = 1 then 1 else (n * a n) / (2 * (n - 1))

def b (n : ℕ) : ℚ := a n / n

-- The required theorem stating the proof
theorem sequence_properties :
  b 1 = 1 ∧ b 2 = 1/2 ∧ b 3 = 1/4 ∧
  (∀ n : ℕ, n ≥ 1 → b (n + 1) = 1 / 2 * b n) ∧
  ∀ n : ℕ, n ≥ 1 → a n = n / 2 ^ (n - 1) := by 
  sorry

end SequenceProblem

end sequence_properties_l316_316640


namespace one_over_a5_eq_30_l316_316320

noncomputable def S : ℕ → ℝ
| n => n / (n + 1)

noncomputable def a (n : ℕ) := if n = 0 then S 0 else S n - S (n - 1)

theorem one_over_a5_eq_30 :
  (1 / a 5) = 30 :=
by
  sorry

end one_over_a5_eq_30_l316_316320


namespace original_quantity_of_ghee_l316_316921

theorem original_quantity_of_ghee
  (Q : ℝ) 
  (H1 : (0.5 * Q) = (0.3 * (Q + 20))) : 
  Q = 30 := 
by
  -- proof goes here
  sorry

end original_quantity_of_ghee_l316_316921


namespace exists_number_divisible_by_5_pow_1000_with_no_zeros_l316_316794

theorem exists_number_divisible_by_5_pow_1000_with_no_zeros :
  ∃ n : ℕ, (5 ^ 1000 ∣ n) ∧ (∀ d ∈ n.digits 10, d ≠ 0) := 
sorry

end exists_number_divisible_by_5_pow_1000_with_no_zeros_l316_316794


namespace smallest_k_for_multiple_of_150_l316_316260

-- Definition for the sum of the first k squares.
def sum_of_squares (k : ℕ) : ℕ := k * (k + 1) * (2 * k + 1) / 6

-- Predicate for sum of squares being a multiple of 150
def multiple_of_150 (n : ℕ) : Prop := 150 ∣ n

-- Main statement: Prove that 100 is the smallest positive integer such that the sum
-- of the first k squares is a multiple of 150.
theorem smallest_k_for_multiple_of_150 : ∀ k : ℕ, 0 < k → multiple_of_150 (sum_of_squares k) → k ≥ 100 :=
begin
  sorry
end

end smallest_k_for_multiple_of_150_l316_316260


namespace nancy_crayons_l316_316780

theorem nancy_crayons (packs : Nat) (crayons_per_pack : Nat) (total_crayons : Nat) 
  (h1 : packs = 41) (h2 : crayons_per_pack = 15) (h3 : total_crayons = packs * crayons_per_pack) : 
  total_crayons = 615 := by
  sorry

end nancy_crayons_l316_316780


namespace sum_midpoint_x_coords_l316_316883

theorem sum_midpoint_x_coords (a b c d : ℝ) (h : a + b + c + d = 20) :
  ((a + b) / 2 + (b + c) / 2 + (c + d) / 2 + (d + a) / 2) = 20 :=
by 
  calc
    ((a + b) / 2 + (b + c) / 2 + (c + d) / 2 + (d + a) / 2)
      = (a + b + b + c + c + d + d + a) / 2 : by sorry
    ... = (2 * (a + b + c + d)) / 2         : by sorry
    ... = a + b + c + d                     : by sorry
    ... = 20                                : by exact h

end sum_midpoint_x_coords_l316_316883


namespace bottles_remaining_in_storage_l316_316564

-- Definitions based on conditions
def initial_small_bottles : ℕ := 5000
def initial_big_bottles : ℕ := 12000
def percentage_small_sold : ℚ := 0.15
def percentage_big_sold : ℚ := 0.18

-- Required proof statement
theorem bottles_remaining_in_storage :
  (initial_small_bottles - (percentage_small_sold * initial_small_bottles).toNat)
  + (initial_big_bottles - (percentage_big_sold * initial_big_bottles).toNat) = 14090 :=
by
  sorry

end bottles_remaining_in_storage_l316_316564


namespace total_boxes_packed_l316_316531

-- Definitions of the conditions
def initial_boxes : ℕ := 400
def food_value_per_box : ℕ := 80
def supplies_value_per_box : ℕ := 165
def donor_multiplier : ℕ := 4

-- Total value of one box
def total_value_per_box : ℕ := food_value_per_box + supplies_value_per_box

-- Initial spending
def initial_spending : ℕ := initial_boxes * total_value_per_box

-- Donation amount
def donation_amount : ℕ := donor_multiplier * initial_spending

-- Number of additional boxes packed with the donation
def additional_boxes : ℕ := donation_amount / total_value_per_box

-- Total number of boxes packed
def total_boxes : ℕ := initial_boxes + additional_boxes

-- Statement to be proven
theorem total_boxes_packed : total_boxes = 2000 := by
  -- Proof for this theorem goes here...
  -- The proof is omitted in this statement as requested.
  sorry

end total_boxes_packed_l316_316531


namespace decreasing_function_on_negative_real_l316_316208

theorem decreasing_function_on_negative_real :
  ∀ x : ℝ, x < 0 → ∃ f : ℝ → ℝ,
    (f = (λ x, 1 / (x - 1)) ∨ f = (λ x, 1 - x^2) ∨ f = (λ x, x^2 + x) ∨ f = (λ x, 1 / (x + 1))) ∧
    (f = (λ x, 1 / (x - 1)) → ∀ x : ℝ, x < 0 → deriv f x < 0) ∧
    (f = (λ x, 1 - x^2) → ∀ x : ℝ, x < 0 → deriv f x > 0) ∧
    (f = (λ x, x^2 + x) → ¬ (∀ x : ℝ, x < 0 → deriv f x < 0)) ∧
    (f = (λ x, 1 / (x + 1)) → ¬ (∀ x : ℝ, x < 0 → deriv f x < 0)) :=
by
  intro x hx
  use λ x, 1 / (x - 1)
  split;
  { left,
    split; intros,
    { sorry }, -- Proof that derivative of y = 1 / (x - 1) < 0 on (-∞, 0)
    { sorry }, -- Proof that derivative of y = 1 - x^2 > 0 on (-∞, 0)
    { sorry }, -- Proof that derivative of y = x^2 + x does not < 0 on (-∞, 0)
    { sorry }  -- Proof that derivative of y = 1 / (x + 1) does not < 0 on (-∞, 0)
  }

end decreasing_function_on_negative_real_l316_316208


namespace team_B_elibility_l316_316115

-- Define conditions as hypotheses
variables (avg_height_A : ℕ)
variables (median_height_B : ℕ)
variables (tallest_height_C : ℕ)
variables (mode_height_D : ℕ)
variables (max_height_allowed : ℕ)

-- Basic height statistics given in the problem
def team_A_statistics := avg_height_A = 166
def team_B_statistics := median_height_B = 167
def team_C_statistics := tallest_height_C = 169
def team_D_statistics := mode_height_D = 167

-- Height constraint condition
def height_constraint := max_height_allowed = 168

-- Mathematical equivalent proof problem: Prove that at least half of Team B sailors can serve
theorem team_B_elibility : height_constraint → team_B_statistics → (∀ (n : ℕ), median_height_B ≤ max_height_allowed) :=
by
  intros constraint_B median_B
  sorry

end team_B_elibility_l316_316115


namespace total_boxes_correct_l316_316538

noncomputable def initial_boxes : ℕ := 400
noncomputable def cost_per_box : ℕ := 80 + 165
noncomputable def initial_spent : ℕ := initial_boxes * cost_per_box
noncomputable def donor_amount : ℕ := 4 * initial_spent
noncomputable def additional_boxes : ℕ := donor_amount / cost_per_box
noncomputable def total_boxes : ℕ := initial_boxes + additional_boxes

theorem total_boxes_correct : total_boxes = 2000 := by
  sorry

end total_boxes_correct_l316_316538


namespace length_of_goods_train_l316_316163

-- Define the given data
def speed_kmph := 72
def platform_length_m := 250
def crossing_time_s := 36

-- Convert speed from kmph to m/s
def speed_mps := speed_kmph * (5 / 18)

-- Define the total distance covered while crossing the platform
def distance_covered_m := speed_mps * crossing_time_s

-- Define the length of the train
def train_length_m := distance_covered_m - platform_length_m

-- The theorem to be proven
theorem length_of_goods_train : train_length_m = 470 := by
  sorry

end length_of_goods_train_l316_316163


namespace professors_chair_selection_l316_316487

/-- The total number of ways Professors Alpha, Beta, and Gamma can choose their chairs 
such that each is between two students and at least two chairs apart from each other 
is 36. --/
theorem professors_chair_selection : 
  ∃ (total_ways : ℕ), 
    total_ways = 36 ∧ 
    ∀ (chairs : fin 13 → Prop), 
      (∀ i, chairs i → i > 3 ∧ i < 10) →
      (∀ (i j : ℕ), chairs i → chairs j → i ≠ j → abs (i - j) ≥ 3) →
      total_ways = 
      (∑ a b c in {4,5,6,7,8,9,10}, (if (abs (a - b) ≥ 3 ∧ abs (a - c) ≥ 3 ∧ abs (b - c) ≥ 3) then 1 else 0) * 6) 
:= sorry

end professors_chair_selection_l316_316487


namespace find_a_l316_316306

-- Define the conditions
def line_eq : ℝ → ℝ → Prop := λ x y, x - y + 1 = 0

def circle_eq (a : ℝ) : ℝ → ℝ → Prop := λ x y, x^2 + y^2 + 2x - 4y + a = 0

def points_on_circle (a : ℝ) (x y : ℝ) : Prop := circle_eq a x y

-- Define the perpendicular condition
def perpendicular (C A B : ℝ × ℝ) : Prop := 
  let (x_C, y_C) := C in
  let (x_A, y_A) := A in
  let (x_B, y_B) := B in
  (y_A - y_C) * (y_B - y_C) + (x_A - x_C) * (x_B - x_C) = 0

-- Define the intersection condition
def intersects_at (x y a : ℝ) : Prop :=
  ∃ A B : (ℝ × ℝ), points_on_circle a A.1 A.2 ∧ points_on_circle a B.1 B.2 ∧
  line_eq A.1 A.2 ∧ line_eq B.1 B.2

-- The main proof statement
theorem find_a (a : ℝ) : 
  (∃ C : (ℝ × ℝ), C = (-1, 2) ∧ intersects_at 1 (-2) a ∧ perpendicular C = true) → a = 1 :=
by 
  sorry

end find_a_l316_316306


namespace tangency_pedal_triangle_similarity_perimeter_half_l316_316420

-- Define a triangle and its incircle points of tangency
variables {A B C A1 B1 C1 : Type} [triangle ABC]
[hincircle : incircle (triangle A B C) A1 B1 C1]

-- Define the original triangle and its pedal triangle
def original_triangle := triangle A B C
def original_pedal_triangle := pedal_triangle (triangle A B C)

-- Define the triangle formed by the points of tangency
def tangency_triangle := triangle A1 B1 C1
def tangency_pedal_triangle := pedal_triangle (triangle A1 B1 C1)

-- Theorem: Prove the similarity and perimeter condition
theorem tangency_pedal_triangle_similarity_perimeter_half :
  similar (tangency_pedal_triangle) (original_triangle) ∧
  perimeter (tangency_pedal_triangle) = (perimeter (original_pedal_triangle)) / 2 :=
by
  sorry

end tangency_pedal_triangle_similarity_perimeter_half_l316_316420


namespace find_g_minus_6_l316_316846

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Conditions given in the problem
axiom cond1 : g 1 - 1 > 0
axiom cond2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom cond3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The proof we want to make (assertion)
theorem find_g_minus_6 : g (-6) = 723 := 
sorry

end find_g_minus_6_l316_316846


namespace linear_substitution_correct_l316_316621

theorem linear_substitution_correct (x y : ℝ) 
  (h1 : y = x - 1) 
  (h2 : x + 2 * y = 7) : 
  x + 2 * x - 2 = 7 := 
by
  sorry

end linear_substitution_correct_l316_316621


namespace simplify_expression_l316_316057

variable (x : ℝ)

theorem simplify_expression :
  (3 * x - 2) * (5 * x ^ 12 - 3 * x ^ 11 + 2 * x ^ 9 - x ^ 6) =
  15 * x ^ 13 - 19 * x ^ 12 - 6 * x ^ 11 + 6 * x ^ 10 - 4 * x ^ 9 - 3 * x ^ 7 + 2 * x ^ 6 :=
by
  sorry

end simplify_expression_l316_316057


namespace julian_needs_more_legos_l316_316739

theorem julian_needs_more_legos :
  ∀ (julian_has legos_per_model num_models : ℕ),
    julian_has = 400 →
    legos_per_model = 375 →
    num_models = 4 →
    (legos_per_model * num_models - julian_has) = 1100 :=
by
  intros julian_has legos_per_model num_models h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end julian_needs_more_legos_l316_316739


namespace box_volume_l316_316513

theorem box_volume (length width square_side : ℝ) (h_length : length = 48) (h_width : width = 36) (h_square_side : square_side = 8) : 
  let new_length := length - 2 * square_side in
  let new_width := width - 2 * square_side in
  let height := square_side in
  new_length * new_width * height = 5120 :=
by
  -- Definitions from the given conditions
  have h1 : new_length = 48 - 2 * 8 := by rw [h_length, h_square_side]
  have h2 : new_width = 36 - 2 * 8 := by rw [h_width, h_square_side]
  have h3 : height = 8 := by rw [h_square_side]

  -- Direct calculation based on translated steps
  rw [h1, h2, h3]
  simp
  norm_num
  sorry

end box_volume_l316_316513


namespace polar_coords_of_point_l316_316227

theorem polar_coords_of_point : 
  ∃ r θ, r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ (r, θ) = (4, 5 * Real.pi / 3)
  ∧ r = Real.sqrt (2^2 + (-2 * Real.sqrt 3)^2) 
  ∧ θ = Real.arctan (Real.abs (-2 * Real.sqrt 3 / 2)) := 
by {
  use 4,
  use 5 * Real.pi / 3,
  split,
  { exact zero_lt_four },
  split,
  { apply Real.le_of_lt, apply mul_pos, exact zero_lt_five, exact Real.pi_pos },
  split,
  { apply Real.lt_of_le_of_lt, norm_num, apply mul_lt_mul_of_pos_left, 
    { apply Real.pi_pos },
    { norm_num } },
  split,
  { refl },
  split,
  { norm_num, norm_cast, rw [Real.sqrt_eq_rpow, Real.sqrt_eq_rpow, Real.rpow_add], norm_num },
  { rw [Real.arctan_eq_inr], norm_num, ring, norm_cast, rw [Real.sqrt_mul] },
}

end polar_coords_of_point_l316_316227


namespace Problem_l316_316676

def f (x : ℕ) : ℕ := x ^ 2 + 1
def g (x : ℕ) : ℕ := 2 * x - 1

theorem Problem : f (g (3 + 1)) = 50 := by
  sorry

end Problem_l316_316676


namespace proof_problem_l316_316383

theorem proof_problem (k m : ℕ) (hk_pos : 0 < k) (hm_pos : 0 < m) (hkm : k > m)
  (hdiv : (k * m * (k ^ 2 - m ^ 2)) ∣ (k ^ 3 - m ^ 3)) :
  (k - m) ^ 3 > 3 * k * m :=
sorry

end proof_problem_l316_316383


namespace complex_roots_eqn_l316_316066

open Complex

theorem complex_roots_eqn (a b c d e k : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) 
  (hk1 : a * k^3 + b * k^2 + c * k + d = e)
  (hk2 : b * k^3 + c * k^2 + d * k + e = a) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I := 
sorry

end complex_roots_eqn_l316_316066


namespace num_integers_abs_x_lt_5pi_l316_316684

theorem num_integers_abs_x_lt_5pi : 
    (finset.card {x : ℤ | abs x < 5 * real.pi} = 31) := 
    sorry

end num_integers_abs_x_lt_5pi_l316_316684


namespace total_boxes_correct_l316_316539

noncomputable def initial_boxes : ℕ := 400
noncomputable def cost_per_box : ℕ := 80 + 165
noncomputable def initial_spent : ℕ := initial_boxes * cost_per_box
noncomputable def donor_amount : ℕ := 4 * initial_spent
noncomputable def additional_boxes : ℕ := donor_amount / cost_per_box
noncomputable def total_boxes : ℕ := initial_boxes + additional_boxes

theorem total_boxes_correct : total_boxes = 2000 := by
  sorry

end total_boxes_correct_l316_316539


namespace ashton_pencils_left_l316_316976

def pencils_left (boxes : Nat) (pencils_per_box : Nat) (pencils_given : Nat) : Nat :=
  boxes * pencils_per_box - pencils_given

theorem ashton_pencils_left :
  pencils_left 2 14 6 = 22 :=
by
  sorry

end ashton_pencils_left_l316_316976


namespace max_balls_drawn_l316_316033

structure DrawingConditions where
  yellow_items : ℕ
  round_items : ℕ
  edible_items : ℕ
  tomato_properties : Prop
  ball_properties : Prop
  banana_properties : Prop

axiom petya_conditions : DrawingConditions → Prop 

theorem max_balls_drawn (cond : DrawingConditions)
  (h1 : cond.yellow_items = 15)
  (h2 : cond.round_items = 18)
  (h3 : cond.edible_items = 13)
  (h4 : cond.tomato_properties)
  (h5 : cond.ball_properties)
  (h6 : cond.banana_properties) :
  ∀ balls : ℕ, balls ≤ 18 :=
by
  sorry

end max_balls_drawn_l316_316033


namespace vasya_arrangement_count_l316_316141
open Nat

/-
Define the problem conditions.
1. Define the numbers 1 to 6.
2. Define the connection constraints where the number in the higher square must be greater.
3. Define the structure based on these constraints.
-/

def sequence : List Nat := [1, 2, 3, 4, 5, 6]

def is_valid_permutation (perm : List Nat) : Prop :=
  (perm[0] < perm[1]) ∧ (perm[1] < perm[2]) ∧ (perm[1] < perm[3]) ∧ (perm[2] < perm[5]) ∧ (perm[4] < perm[5])

/- 
The main theorem stating the question and the answer.
-/
theorem vasya_arrangement_count : 
  (∃ l : List Nat, l.permutations.count is_valid_permutation = 12) := 
sorry

end vasya_arrangement_count_l316_316141


namespace smallest_sum_of_prime_set_l316_316196

open Nat

def single_digit_primes := [2, 3, 5, 7]
def starts_with_3_primes := [31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89]

def is_valid_prime_set (set : List Nat) : Prop :=
  set.length = 4 ∧
  (∀ p ∈ set, Prime p) ∧
  (∀ d ∈ List.iota 9 + 1, ∃ p ∈ set, d ∈ p.digits 10) ∧
  (∃ p ∈ set, p.digits 10 ≠ [] ∧ p.digits 10.head = 3) ∧
  (∃ p ∈ set, p ∈ single_digit_primes)

def smallest_prime_sum (set : List Nat) : Nat :=
  if is_valid_prime_set set then set.sum else 0

theorem smallest_sum_of_prime_set : ∃ set : List Nat, is_valid_prime_set set ∧ set.sum = 449 :=
sorry

end smallest_sum_of_prime_set_l316_316196


namespace quadratic_sum_l316_316092

theorem quadratic_sum (a b c : ℝ) (h : ∀ x : ℝ, 5 * x^2 - 30 * x - 45 = a * (x + b)^2 + c) :
  a + b + c = -88 := by
  sorry

end quadratic_sum_l316_316092


namespace train_length_is_300_l316_316165

noncomputable def speed_kmph : Float := 90
noncomputable def speed_mps : Float := (speed_kmph * 1000) / 3600
noncomputable def time_sec : Float := 12
noncomputable def length_of_train : Float := speed_mps * time_sec

theorem train_length_is_300 : length_of_train = 300 := by
  sorry

end train_length_is_300_l316_316165


namespace not_transformable_to_l316_316051

def quadratic (a b c : ℝ) : (ℝ → ℝ) := λ x => a * x^2 + b * x + c

def transform1 (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x => x^2 * f(1 + 1/x)

def transform2 (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x => (x - 1)^2 * f(1 / (x - 1))

theorem not_transformable_to :
  ¬ ∃ f : ℝ → ℝ, ∃ g : ℝ → ℝ,
  (f = quadratic 1 4 3) ∧ (g = quadratic 1 10 9) ∧
  (
    ∃ n : ℕ, 
    (n = 0 → f = g) ∨ 
    (∃ (fs : fin n → (ℝ → ℝ)), 
    (fs 0 = f) ∧ (fs (fin.last n) = g) ∧ (∀ i : fin n, fs (i+1) = transform1 (fs i) ∨ fs (i+1) = transform2 (fs i))
  )
) := sorry

end not_transformable_to_l316_316051


namespace triangle_DEF_acute_l316_316519

theorem triangle_DEF_acute (ABC : Type*) [triangle ABC]
  (incircle : incircle (ABC : Type*)) (D E F : ABC) :
  touches (incircle : ABC) (BC : ABC) D ∧ touches (incircle : ABC) (CA : ABC) E ∧ touches (incircle : ABC) (AB : ABC) F →
  acute_triangle DEF :=
begin
  sorry
end

end triangle_DEF_acute_l316_316519


namespace a_seq_limit_l316_316381

noncomputable def a_seq (n : ℕ) : ℝ := sorry -- the sequence is not explicitly given

axiom a_seq_pos : ∀ n, a_seq n > 0
axiom a_seq_unbounded : ∀ M : ℝ, ∃ n : ℕ, a_seq n > M
axiom a_seq_increasing : ∀ n, a_seq n < a_seq (n + 1)
axiom a_seq_mean_property : ∀ n : ℕ,
  ∃ k : ℕ, (a_seq n + a_seq (n + 1) + a_seq (n + 2) + a_seq (n + 3)) / 4 = a_seq k

theorem a_seq_limit : ∃ L : ℝ, (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, (a_seq (n + 1) / a_seq n - L).abs < ε) ∧ L = 1 + Real.sqrt 2 := sorry

end a_seq_limit_l316_316381


namespace gcd_lcm_product_l316_316254

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 24) (h₂ : b = 36) :
  Nat.gcd a b * Nat.lcm a b = 864 :=
by
  rw [h₁, h₂]
  unfold Nat.gcd Nat.lcm
  sorry

end gcd_lcm_product_l316_316254


namespace distinct_terms_expansion_l316_316985

theorem distinct_terms_expansion {a b c x y z w t : Type} [decidable_eq a] [decidable_eq b] [decidable_eq c] [decidable_eq x] [decidable_eq y] [decidable_eq z] [decidable_eq w] [decidable_eq t] :
  fintype.card (finset.product {a, b, c} {x, y, z, w, t}) = 15 :=
by sorry

end distinct_terms_expansion_l316_316985


namespace AMDN_is_parallelogram_l316_316143

noncomputable def are_parallel {α : Type*} [euclidean_space α] (l1 l2 : affine_subspace α (point α)) : Prop :=
sorry

noncomputable def is_parallelogram {α : Type*} [euclidean_space α] (A M D N : point α) : Prop :=
are_parallel (line_through A M) (line_through D N) ∧ are_parallel (line_through A D) (line_through M N)

theorem AMDN_is_parallelogram
  {α : Type*} [euclidean_space α]
  (Γ₁ Γ₂ : affine_subspace α (point α))
  (P Q A B C D E F : point α)
  (M N : point α)
  (h₁ : P ∈ Γ₁ ∧ P ∈ Γ₂)
  (h₂ : Q ∈ Γ₁ ∧ Q ∈ Γ₂)
  (h₃ : A ∈ Γ₁) (h₄ : B ∈ Γ₁) (h₅ : C ∈ Γ₁)
  (h₆ : D ∈ Γ₂) (h₇ : E ∈ Γ₂) (h₈ : F ∈ Γ₂)
  (h9 : line_through A E = line_through B D ∩ P)
  (h10 : line_through A F = line_through C D ∩ Q)
  (h11 : M = line_through A B ∩ line_through D E)
  (h12 : N = line_through A C ∩ line_through D F) :
  is_parallelogram A M D N :=
sorry

end AMDN_is_parallelogram_l316_316143


namespace convex_quadrilateral_area_l316_316434

-- Defining the geometrical setup.
structure Square :=
  (E F G H Q : Point)
  (side_length : ℝ)
  (eq_dist : ℝ)
  (fq_dist : ℝ)
  (is_square : E.distance F = side_length ∧ F.distance G = side_length ∧ G.distance H = side_length ∧ H.distance E = side_length)
  (inside_square : Q ∈ Interior (ConvexHull ℝ (Set.insert E (Set.insert F (Set.insert G (Set.singleton H))))))
  (eq_condition : E.distance Q = eq_dist)
  (fq_condition : F.distance Q = fq_dist)

-- Theorem stating the main proof problem
theorem convex_quadrilateral_area (EFGH : Square)
  (side_length_eq : EFGH.side_length = 40)
  (eq_dist_eq : EFGH.eq_dist = 16)
  (fq_dist_eq : EFGH.fq_dist = 34)
  : ∃ quad : Quadrilateral, quad.area = 200 := sorry

end convex_quadrilateral_area_l316_316434


namespace find_g_neg6_l316_316833

-- Define the function g with conditions as hypotheses
variables {g : ℤ → ℤ}

-- Condition 1
axiom g_condition_1 : g 1 - 1 > 0

-- Condition 2
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x

-- Condition 3
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The goal is to find g(-6)
theorem find_g_neg6 : g (-6) = 723 :=
by {
  sorry
}

end find_g_neg6_l316_316833


namespace total_weight_correct_l316_316412

def weight_male_clothes : ℝ := 2.6
def weight_female_clothes : ℝ := 5.98
def total_weight_clothes : ℝ := weight_male_clothes + weight_female_clothes

theorem total_weight_correct : total_weight_clothes = 8.58 := by
  sorry

end total_weight_correct_l316_316412


namespace common_internal_tangent_length_l316_316445

-- Definitions based on given conditions
def center_distance : ℝ := 50
def radius_small : ℝ := 7
def radius_large : ℝ := 10

-- Target theorem
theorem common_internal_tangent_length :
  let AB := center_distance
  let BE := radius_small + radius_large 
  let AE := Real.sqrt (AB^2 - BE^2)
  AE = Real.sqrt 2211 :=
by
  sorry

end common_internal_tangent_length_l316_316445


namespace gcd_lcm_product_24_36_l316_316257

-- Definitions for gcd, lcm, and product for given numbers, skipping proof with sorry
theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  -- Sorry used to skip proof
  sorry

end gcd_lcm_product_24_36_l316_316257


namespace area_of_square_with_diagonal_l316_316905

theorem area_of_square_with_diagonal (d : ℝ) (hd : d = 16) : ∃ (area : ℝ), area = 128 :=
by
  -- Given the diagonal length of a square
  let a := d / (real.sqrt 2)
  -- The area is calculated as the side length squared.
  let area := a^2
  have h1 : a = 8 * real.sqrt 2 := by sorry
  have h2 : area = (8 * real.sqrt 2)^2 := by sorry
  have h3 : area = 128 := by sorry
  use area
  exact h3

end area_of_square_with_diagonal_l316_316905


namespace prob_all_fail_prob_at_least_one_pass_l316_316654

variable (A B C : Prop)
variable [Independent A B] [Independent A C] [Independent B C]
variable (pA pB pC : ℝ)
variable (hA : pA = 1/2) (hB : pB = 1/2) (hC : pC = 1/2)

namespace ProbabilityProblem

-- Probability that all three fail
theorem prob_all_fail (h : P(A) = pA ∧ P(B) = pB ∧ P(C) = pC) : P(¬A ∧ ¬B ∧ ¬C) = 1 / 8 :=
by sorry

-- Probability that at least one person passes
theorem prob_at_least_one_pass (h : P(A) = pA ∧ P(B) = pB ∧ P(C) = pC) : 1 - P(¬A ∧ ¬B ∧ ¬C) = 7 / 8 :=
by sorry

end ProbabilityProblem

end prob_all_fail_prob_at_least_one_pass_l316_316654


namespace range_of_a_l316_316666

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then 2 * a * x^2 - x - 1 / 4
  else Real.log a x - 1

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y ∨ f a y ≤ f a x) ↔ (1 / 8 ≤ a ∧ a ≤ 1 / 4) := 
sorry

end range_of_a_l316_316666


namespace correct_inequality_l316_316911

theorem correct_inequality (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 0) :
  a^2 > ab ∧ ab > a :=
sorry

end correct_inequality_l316_316911


namespace cyclist_wait_time_after_passing_l316_316205

-- Defining speeds and conditions
def walker_speed_mph : ℝ := 4 -- Walker speed in miles per hour
def cyclist_speed_mph : ℝ := 20 -- Cyclist speed in miles per hour
def time_waiting_min : ℝ := 20 -- Time cyclist waits for the walker in minutes

-- Unit conversions: speeds in miles per minute
def walker_speed_mpm : ℝ := walker_speed_mph / 60
def cyclist_speed_mpm : ℝ := cyclist_speed_mph / 60

-- Distance walker covers in the waiting period
def distance_covered_by_walker : ℝ := walker_speed_mpm * time_waiting_min

-- Prove t (time cyclist travels before stopping) is 4 minutes
theorem cyclist_wait_time_after_passing : 
  ∃ t : ℝ, distance_covered_by_walker = t * cyclist_speed_mpm ∧ t = 4 := by
  sorry

end cyclist_wait_time_after_passing_l316_316205


namespace place_numbers_l316_316791

theorem place_numbers (a b c d : ℕ) (hab : Nat.gcd a b = 1) (hac : Nat.gcd a c = 1) 
  (had : Nat.gcd a d = 1) (hbc : Nat.gcd b c = 1) (hbd : Nat.gcd b d = 1) 
  (hcd : Nat.gcd c d = 1) :
  ∃ (bc ad ab cd abcd : ℕ), 
    bc = b * c ∧ ad = a * d ∧ ab = a * b ∧ cd = c * d ∧ abcd = a * b * c * d ∧
    Nat.gcd bc abcd > 1 ∧ Nat.gcd ad abcd > 1 ∧ Nat.gcd ab abcd > 1 ∧ 
    Nat.gcd cd abcd > 1 ∧
    Nat.gcd ab cd = 1 ∧ Nat.gcd ab ad = 1 ∧ Nat.gcd ab bc = 1 ∧ 
    Nat.gcd cd ad = 1 ∧ Nat.gcd cd bc = 1 ∧ Nat.gcd ad bc = 1 :=
by
  sorry

end place_numbers_l316_316791


namespace will_buttons_l316_316379

/-
Variables and Conditions
-/

def Mari := 8
def Kendra := 5 * Mari + 4
def Sue := (1/2 : ℝ) * Kendra
def Combined_buttons := Kendra + Sue
def Will := 2 * Combined_buttons

/-
Goal: Will's buttons Proof Statement
-/

theorem will_buttons : Will = 132 :=
by
  sorry

end will_buttons_l316_316379


namespace number_of_unique_three_digit_integers_l316_316299

theorem number_of_unique_three_digit_integers : 
  let digits := {1, 3, 5, 7} in
  (∃ (s1 s2 s3 : ℕ), 
     s1 ∈ digits ∧ 
     s2 ∈ digits ∧ 
     s3 ∈ digits ∧ 
     s1 ≠ s2 ∧ 
     s1 ≠ s3 ∧ 
     s2 ≠ s3) →
  finset.card (finset.pi_finset digits {1, 2, 3} (λ _, digits)) = 24 := 
by 
  let digits := {1, 3, 5, 7}
  sorry

end number_of_unique_three_digit_integers_l316_316299


namespace cost_of_siding_l316_316427

noncomputable def wall_width : ℕ := 10
noncomputable def wall_height : ℕ := 7
noncomputable def roof_width : ℕ := 10
noncomputable def roof_height : ℕ := 6
noncomputable def siding_section_width : ℕ := 10
noncomputable def siding_section_height : ℕ := 15
noncomputable def siding_section_cost : ℕ := 35

theorem cost_of_siding :
  let wall_area := wall_width * wall_height in
  let roof_area := roof_width * roof_height * 2 in
  let total_area := wall_area + roof_area in
  let section_area := siding_section_width * siding_section_height in
  let sections_needed := (total_area + section_area - 1) / section_area in
  let total_cost := sections_needed * siding_section_cost in
  total_cost = 70 :=
by
  sorry

end cost_of_siding_l316_316427


namespace max_balls_drawn_l316_316038

-- Conditions:
variable (items : Type) 
variable (Petya : items → Prop)
variable (yellow round edible : items → Prop)
variable (sun ball tomato banana : items)

variable (Ht : ∀ x, tomato x → round x ∧ ¬yellow x) -- All tomatoes are round and red
variable (Hb : ∀ x, banana x → yellow x ∧ ¬round x) -- All bananas are yellow and not round
variable (Hba : ∀ x, ball x → round x) -- All balls are round

variable (yellow_count : ∑ x in items, yellow x = 15) -- Exactly 15 yellow items
variable (round_count : ∑ x in items, round x = 18) -- Exactly 18 round items
variable (edible_count : ∑ x in items, edible x = 13) -- Exactly 13 edible items

-- Proving the maximum number of balls
theorem max_balls_drawn : ∑ x in items, ball x ≤ 18 :=
by sorry

end max_balls_drawn_l316_316038


namespace tanP_tanQ_equals_4_l316_316875

open Real

-- Definitions based on the problem's conditions
variables (P Q R : Point) (H M : Point)
variable [nonempty M]
variables (dP dQ dR dH dM : ℝ)
variables (θP θQ : ℝ)

-- H is the orthocenter and divides QM into HM = 3 and HQ = 9
def orthocenter_divides_altitude : Prop :=
  (H = (orthocenter P Q R)) ∧
  (distance H M = 3) ∧
  (distance H Q = 9)

-- Objective: prove that tan P * tan Q = 4
theorem tanP_tanQ_equals_4 (h : orthocenter_divides_altitude P Q R H M) :
  tan θP * tan θQ = 4 :=
sorry

end tanP_tanQ_equals_4_l316_316875


namespace find_g_neg6_l316_316832

-- Define the function g with conditions as hypotheses
variables {g : ℤ → ℤ}

-- Condition 1
axiom g_condition_1 : g 1 - 1 > 0

-- Condition 2
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x

-- Condition 3
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The goal is to find g(-6)
theorem find_g_neg6 : g (-6) = 723 :=
by {
  sorry
}

end find_g_neg6_l316_316832


namespace team_B_eligible_l316_316124

-- Define the conditions
def max_allowed_height : ℝ := 168
def average_height_team_A : ℝ := 166
def median_height_team_B : ℝ := 167
def tallest_sailor_in_team_C : ℝ := 169
def mode_height_team_D : ℝ := 167

-- Define the proof statement
theorem team_B_eligible : 
  (∃ (heights_B : list ℝ), heights_B.length > 0 ∧ median heights_B = median_height_team_B) →
  (∀ h ∈ heights_B, h ≤ max_allowed_height) ∨ (∃ (S : finset ℝ), S.card ≥ heights_B.length / 2 ∧ ∀ h ∈ S, h ≤ max_allowed_height) :=
sorry

end team_B_eligible_l316_316124


namespace find_g_neg_six_l316_316865

theorem find_g_neg_six (g : ℤ → ℤ)
  (h1 : g 1 - 1 > 0)
  (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
  (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 :=
sorry

end find_g_neg_six_l316_316865


namespace correct_ordering_l316_316555

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f (-x) = f x
axiom monotonicity (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : x1 ≠ x2) : (x1 - x2) * (f x1 - f x2) > 0

theorem correct_ordering : f 1 < f (-2) ∧ f (-2) < f 3 :=
by sorry

end correct_ordering_l316_316555


namespace at_least_half_team_B_can_serve_on_submarine_l316_316133

theorem at_least_half_team_B_can_serve_on_submarine
    (max_height : ℕ)
    (team_A_avg_height : ℕ)
    (team_B_median_height : ℕ)
    (team_C_tallest_height : ℕ)
    (team_D_mode_height : ℕ)
    (h1 : max_height = 168)
    (h2 : team_A_avg_height = 166)
    (h3 : team_B_median_height = 167)
    (h4 : team_C_tallest_height = 169)
    (h5 : team_D_mode_height = 167) :
  ∀ (height : ℕ), height ≤ max_height → ∃ (b_sailors : ℕ → Prop) (H : ∃ n, b_sailors n),
  (∃ (n_half : ℕ), (∀ h ≤ team_B_median_height, b_sailors h) ∧ (2 * n_half ≤ n)) :=
sorry

end at_least_half_team_B_can_serve_on_submarine_l316_316133


namespace num_solutions_l316_316461

def is_positive_integer (n : ℤ) : Prop := n > 0

def valid_solution (x y : ℤ) : Prop :=
  2 * x + 3 * y = 317 ∧ is_positive_integer x ∧ is_positive_integer y

theorem num_solutions : fintype { p : ℤ × ℤ // valid_solution p.1 p.2 }.card = 53 :=
sorry

end num_solutions_l316_316461


namespace fraction_division_correct_l316_316903

theorem fraction_division_correct :
  (2 / 5) / 3 = 2 / 15 :=
by sorry

end fraction_division_correct_l316_316903


namespace correct_options_l316_316665

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3) + 1

def option_2 : Prop := ∃ x : ℝ, f (x) = 0 ∧ x = Real.pi / 3
def option_3 : Prop := ∀ T > 0, (∀ x : ℝ, f (x) = f (x + T)) → T = Real.pi
def option_5 : Prop := ∀ x : ℝ, f (x - Real.pi / 6) = f (-(x - Real.pi / 6))

theorem correct_options :
  option_2 ∧ option_3 ∧ option_5 :=
by
  sorry

end correct_options_l316_316665


namespace necessary_condition_equiv_l316_316967

variable {X Y A B C P Q : Prop}
variable {x y : ℝ}
variable {a b c : ℝ}

-- Definition of necessary condition
def necessary_condition (p q : Prop) : Prop :=
  q → p

-- Statement of the problem
theorem necessary_condition_equiv : 
  (necessary_condition (x > 5) (x > 10)) ∧ 
  (∀ (a b c : ℝ), c ≠ 0 → necessary_condition (a * c = b * c) (a = b)) ∧ 
  (∀ (x y : ℝ), necessary_condition (2 * x + 1 = 2 * y + 1) (x = y)) :=
  by
    split; sorry
    split; sorry
    split; sorry

end necessary_condition_equiv_l316_316967


namespace number_of_terms_is_10_l316_316580

noncomputable def arith_seq_number_of_terms (a : ℕ) (n : ℕ) (d : ℕ) : Prop :=
  (n % 2 = 0) ∧ ((n-1)*d = 16) ∧ (n * (2*a + (n-2)*d) = 56) ∧ (n * (2*a + n*d) = 76)

theorem number_of_terms_is_10 (a d n : ℕ) (h : arith_seq_number_of_terms a n d) : n = 10 := by
  sorry

end number_of_terms_is_10_l316_316580


namespace spider_paths_l316_316185

theorem spider_paths : 
  let total_steps := 11
  let upward_steps := 5
  let rightward_steps := 6
  binomial total_steps upward_steps = 462 :=
by
  let total_steps := 11
  let upward_steps := 5
  let rightward_steps := 6
  sorry

end spider_paths_l316_316185


namespace derivative_of_periodic_is_periodic_l316_316500

-- Definition of a periodic function
def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f(x + T) = f(x)

-- Theorem statement: The derivative of a periodic function is periodic with the same period
theorem derivative_of_periodic_is_periodic {f : ℝ → ℝ} {T : ℝ} 
  (hf : is_periodic f T) (h_diff : differentiable ℝ f) : 
  is_periodic (deriv f) T :=
by
  sorry

end derivative_of_periodic_is_periodic_l316_316500


namespace symmetric_point_origin_l316_316074

theorem symmetric_point_origin (x y : Int) (hx : x = -(-4)) (hy : y = -(3)) :
    (x, y) = (4, -3) := by
  sorry

end symmetric_point_origin_l316_316074


namespace intersection_A_compl_B_subset_E_B_l316_316647

namespace MathProof

-- Definitions
def A := {x : ℝ | (x + 3) * (x - 6) ≥ 0}
def B := {x : ℝ | (x + 2) / (x - 14) < 0}
def compl_R_B := {x : ℝ | x ≤ -2 ∨ x ≥ 14}
def E (a : ℝ) := {x : ℝ | 2 * a < x ∧ x < a + 1}

-- Theorem for intersection of A and complement of B
theorem intersection_A_compl_B : A ∩ compl_R_B = {x : ℝ | x ≤ -3 ∨ x ≥ 14} :=
by
  sorry

-- Theorem for subset relationship to determine range of a
theorem subset_E_B (a : ℝ) : (E a ⊆ B) → a ≥ -1 :=
by
  sorry

end MathProof

end intersection_A_compl_B_subset_E_B_l316_316647


namespace composite_divides_factorial_of_difference_l316_316404

noncomputable def P_k (k : ℕ) [Fact (k ≥ 14)] : ℕ := 
  Nat.findGreatestPrime (k - 1)

theorem composite_divides_factorial_of_difference
  (k n : ℕ) 
  [Fact (k ≥ 14)]
  (hk : ∃ p : ℕ, P_k k = p ∧ p < k)
  (hprime : P_k k ≥ 3 * k / 4)
  (h_composite : ∃ a b : ℕ, 2 ≤ a ∧ a ≤ b ∧ n = a * b)
  (h_n_gt : n > 2 * P_k k) 
  : n ∣ (n - k)! := 
by 
  sorry

end composite_divides_factorial_of_difference_l316_316404


namespace range_of_m_l316_316453

-- Define the function and its properties
variable {f : ℝ → ℝ}
variable (increasing : ∀ x1 x2 : ℝ, x1 > x2 → f x1 > f x2)

theorem range_of_m (h: ∀ m : ℝ, f (2 * m) > f (-m + 9)) : 
  ∀ m : ℝ, m > 3 ↔ f (2 * m) > f (-m + 9) :=
by
  intros
  sorry

end range_of_m_l316_316453


namespace gcd_values_count_l316_316506

theorem gcd_values_count (a b : ℕ) (h : a * b = 300) : 
  set_finite {d : ℕ | ∃ p q r s t u : ℕ, a = 2^p * 3^q * 5^r ∧ b = 2^s * 3^t * 5^u ∧ p + s = 2 ∧ q + t = 1 ∧ r + u = 2 ∧ d = Integer.gcd a b } ∧ 
  set_card {d : ℕ | ∃ p q r s t u : ℕ, a = 2^p * 3^q * 5^r ∧ b = 2^s * 3^t * 5^u ∧ p + s = 2 ∧ q + t = 1 ∧ r + u = 2 ∧ d = Integer.gcd a b } = 8 := 
by
  sorry

end gcd_values_count_l316_316506


namespace max_quotient_l316_316700

theorem max_quotient (a b : ℕ) (h₁ : 100 ≤ a) (h₂ : a ≤ 300) (h₃ : 1200 ≤ b) (h₄ : b ≤ 2400) :
  b / a ≤ 24 :=
sorry

end max_quotient_l316_316700


namespace part1_part2_l316_316653

-- Given conditions for the problem
variable (n : ℕ) (hn : 4 ≤ n) (M : Finset ℕ := Finset.range (n + 1))

-- Part 1: Given n = 7 and a specific ordering condition, find the number of sequences T
theorem part1 : n = 7 → (∃ (a : Fin 8 → ℕ), a 2 < a 3 ∧ a 3 < a 4 ∧ a 4 < a 5 ∧ a 5 < a 6 ∧ ∀ i, a i ∈ M ∧ Function.Injective a) → Finset.card ({a : Fin 8 → ℕ | a 2 < a 3 ∧ a 3 < a 4 ∧ a 4 < a 5 ∧ a 5 < a 6 ∧ (∀ i, a i ∈ M) ∧ Function.Injective a}) = 42 :=
by intros; sorry

-- Part 2: Given unique term a_k such that a_k > a_(k+1), find the number of sequences T
theorem part2 : (∃ k ∈ Finset.filter (λ k, k < n) M, ∀ a : Fin n.succ → ℕ, (a k > a (k + 1) ∧ ∀ i, a i ∈ M ∧ Function.Injective a)) → Finset.card ({T : Fin n.succ → ℕ | ∃ k, k < n ∧ T k > T (k + 1) ∧ (∀ i, T i ∈ M) ∧ Function.Injective T}) = 2^n - n - 1 :=
by intros; sorry

end part1_part2_l316_316653


namespace max_balls_l316_316017

theorem max_balls (total_yellow total_round total_edible : ℕ) 
  (suns balls tomatoes bananas : ℕ) :
  (total_yellow = 15) →
  (total_round = 18) →
  (total_edible = 13) →
  (tomatoes + balls ≤ total_round) →
  (tomatoes + bananas ≤ total_edible) →
  (suns + balls + tomatoes + bananas = total_yellow + total_round + total_edible) →
  (∀ b, b ∈ {balls, tomatoes, bananas, suns} → b ≥ 0) →
  (tomatoes ≤ total_round) →
  balls = 18 :=
by 
  sorry

end max_balls_l316_316017


namespace cone_surface_area_is_4pi_l316_316952

-- Definitions of given conditions
def sector_central_angle : Real := 120
def sector_area : Real := 3 * Real.pi

-- Function to compute the surface area of the cone
def cone_surface_area (r l : Real) : Real :=
  Real.pi * r * l + Real.pi * r^2

-- The theorem to prove
theorem cone_surface_area_is_4pi (R l r : Real) (h1 : 1 / 3 * Real.pi * R^2 = 3 * Real.pi)
  (h2 : l = R) (h3 : Real.pi * r * l = 3 * Real.pi) :
  cone_surface_area r l = 4 * Real.pi :=
by
  sorry

end cone_surface_area_is_4pi_l316_316952


namespace team_B_elibility_l316_316118

-- Define conditions as hypotheses
variables (avg_height_A : ℕ)
variables (median_height_B : ℕ)
variables (tallest_height_C : ℕ)
variables (mode_height_D : ℕ)
variables (max_height_allowed : ℕ)

-- Basic height statistics given in the problem
def team_A_statistics := avg_height_A = 166
def team_B_statistics := median_height_B = 167
def team_C_statistics := tallest_height_C = 169
def team_D_statistics := mode_height_D = 167

-- Height constraint condition
def height_constraint := max_height_allowed = 168

-- Mathematical equivalent proof problem: Prove that at least half of Team B sailors can serve
theorem team_B_elibility : height_constraint → team_B_statistics → (∀ (n : ℕ), median_height_B ≤ max_height_allowed) :=
by
  intros constraint_B median_B
  sorry

end team_B_elibility_l316_316118


namespace man_walking_speed_l316_316945

noncomputable def walkingSpeed_km_per_hr : ℝ :=
  let distance_meters := 1750
  let time_minutes := 15
  let speed_m_per_min := distance_meters / time_minutes
  let speed_m_per_hr := speed_m_per_min * 60
  speed_m_per_hr / 1000

theorem man_walking_speed :
  walkingSpeed_km_per_hr = 7.002 :=
by
  let distance_meters := 1750
  let time_minutes := 15
  let speed_m_per_min := distance_meters / time_minutes
  let speed_m_per_hr := speed_m_per_min * 60
  let result := speed_m_per_hr / 1000
  show result = 7.002
  sorry

end man_walking_speed_l316_316945


namespace extreme_points_count_sum_of_extreme_points_l316_316302

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + a * (x^2 - x)

-- Statement for Part (1):
theorem extreme_points_count 
  (a : ℝ) : 
  (∃ x_1 x_2 : ℝ, x_1 ≠ x_2 ∧ f'(x_1, a) = 0 ∧ f'(x_2, a) = 0 ∧ f''(x_1, a) * f''(x_2, a) < 0) ↔ a > 8 :=
sorry

-- Statement for Part (2):
theorem sum_of_extreme_points 
  (a : ℝ) (h : a > 8)
  (x_1 x_2 : ℝ) 
  (hx1x2 : x_1 ≠ x_2) 
  (hx1 : f'(x_1, a) = 0) 
  (hx2 : f'(x_2, a) = 0) :
  f(x_1, a) + f(x_2, a) < -3 - 4 * Real.log 2 :=
sorry

end extreme_points_count_sum_of_extreme_points_l316_316302


namespace sum_of_odd_capacities_S4_l316_316392

-- Definitions given in the conditions
def Sn (n : ℕ) : Set ℕ := { x | x ∈ Finset.range (n + 1) ∧ x ≠ 0 }

def capacity (X : Set ℕ) : ℕ :=
  if X = ∅ then 0 else X.prod id

def is_odd_subset (X : Set ℕ) : Prop :=
  (capacity X) % 2 = 1

def sum_of_odd_capacities (n : ℕ) : ℕ :=
  (Finset.powerset (Sn n)).filter is_odd_subset
                           .sum (λ X, capacity (X : Set ℕ))

-- The proof problem
theorem sum_of_odd_capacities_S4 : sum_of_odd_capacities 4 = 7 := sorry

end sum_of_odd_capacities_S4_l316_316392


namespace number_of_digits_l316_316314

theorem number_of_digits (
  X : ℕ
) : (nat.digits 10 (50^8 * 8^X * 11^2 * 10^4)).length = 18 := 
sorry

end number_of_digits_l316_316314


namespace roots_greater_than_one_l316_316267

def quadratic_roots_greater_than_one (a : ℝ) : Prop :=
  ∀ x : ℝ, (1 + a) * x^2 - 3 * a * x + 4 * a = 0 → x > 1

theorem roots_greater_than_one (a : ℝ) :
  -16/7 < a ∧ a < -1 → quadratic_roots_greater_than_one a :=
sorry

end roots_greater_than_one_l316_316267


namespace complex_modulus_is_five_l316_316927

def complex_modulus_condition (z : ℂ) : Prop :=
  z + 2 * conj(z) = 9 + 4 * complex.I

theorem complex_modulus_is_five (z : ℂ) (h : complex_modulus_condition z) : complex.norm z = 5 := 
  sorry

end complex_modulus_is_five_l316_316927


namespace find_common_difference_l316_316284

variable {a : ℕ → ℝ} (d : ℝ) (a₁ : ℝ)

-- defining the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (a₁ : ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a n = a₁ + n * d

-- condition for the sum of even indexed terms
def sum_even_terms (a : ℕ → ℝ) : ℝ := a 2 + a 4 + a 6 + a 8 + a 10

-- condition for the sum of odd indexed terms
def sum_odd_terms (a : ℕ → ℝ) : ℝ := a 1 + a 3 + a 5 + a 7 + a 9

-- main theorem to prove
theorem find_common_difference
  (a : ℕ → ℝ) (a₁ : ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a a₁ d)
  (h_even_sum : sum_even_terms a = 30)
  (h_odd_sum : sum_odd_terms a = 25) :
  d = 1 := by
  sorry

end find_common_difference_l316_316284


namespace exists_product_of_any_five_greater_than_remaining_six_l316_316478

theorem exists_product_of_any_five_greater_than_remaining_six :
  ∃ (a : Fin 11 → ℤ), (∀ s t : Finset (Fin 11), s.card = 5 → t.card = 6 → s ∪ t = Finset.univ → (∏ i in s, a i) > (∏ i in t, a i)) :=
by
  sorry

end exists_product_of_any_five_greater_than_remaining_six_l316_316478


namespace problem1_l316_316171

variable (α : ℝ)

theorem problem1 (h : Real.tan α = -3/4) : 
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / 
  (Real.cos (11 * π / 2 - α) * Real.sin (9 * π / 2 + α)) = -3/4 := 
sorry

end problem1_l316_316171


namespace tiles_with_no_gaps_l316_316560

-- Define the condition that the tiling consists of regular octagons
def regular_octagon_internal_angle := 135

-- Define the other regular polygons
def regular_triangle_internal_angle := 60
def regular_square_internal_angle := 90
def regular_pentagon_internal_angle := 108
def regular_hexagon_internal_angle := 120

-- The proposition to be proved: A flat surface without gaps
-- can be achieved using regular squares and regular octagons.
theorem tiles_with_no_gaps :
  ∃ (m n : ℕ), regular_octagon_internal_angle * m + regular_square_internal_angle * n = 360 :=
sorry

end tiles_with_no_gaps_l316_316560


namespace triangle_area_CO_B_l316_316049

-- Define the conditions as given in the problem
structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩
def Q : Point := ⟨0, 15⟩

variable (p : ℝ)
def C : Point := ⟨0, p⟩
def B : Point := ⟨15, 0⟩

-- Prove the area of triangle COB is 15p / 2
theorem triangle_area_CO_B :
  p ≥ 0 → p ≤ 15 → 
  let base := 15
  let height := p
  let area := (1 / 2) * base * height
  area = (15 * p) / 2 := 
by
  intros hp0 hp15
  let base := 15
  let height := p
  let area := (1 / 2) * base * height
  have : area = (15 * p) / 2 := sorry
  exact this

end triangle_area_CO_B_l316_316049


namespace convex_polygon_point_selection_l316_316646

-- Given a convex polygon P with n vertices and no concurrent diagonals
variables {n : ℕ} (P : polygon)

-- Condition: n is a positive integer greater than or equal to 5
axiom h_n : n ≥ 5

-- Condition: P is a convex polygon with vertices A_1, A_2, ..., A_n
axiom h_convex : convex P

-- Condition: No diagonals of P are concurrent
axiom h_no_concurrent_diags : no_concurrent_diagonals P

-- The theorem to prove
theorem convex_polygon_point_selection :
  ∃ (points : set (point P)) , 
    (∀ q in points, ∃ i j k l, 1 ≤ i ∧ i < j ∧ j < k ∧ k < l ∧ l ≤ n ∧ q ∈ interior (polygon.quad P i j k l) ∧ 
    ∀ q1 q2 ∈ points, q1 ≠ q2 → segment q1 q2 ∩ ⋃ d ∈ P.diagonals, d ≠ ∅) :=
sorry

end convex_polygon_point_selection_l316_316646


namespace gcd_lcm_product_l316_316255

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 24) (h₂ : b = 36) :
  Nat.gcd a b * Nat.lcm a b = 864 :=
by
  rw [h₁, h₂]
  unfold Nat.gcd Nat.lcm
  sorry

end gcd_lcm_product_l316_316255


namespace purely_imaginary_z_value_l316_316325

theorem purely_imaginary_z_value (a : ℝ) (h : (a^2 - a - 2) = 0 ∧ (a + 1) ≠ 0) : a = 2 :=
sorry

end purely_imaginary_z_value_l316_316325


namespace plane_vectors_angle_l316_316296

open Real
open_locale big_operators

variables (a b : ℝ × ℝ)
variables (t : ℝ)

noncomputable def vec_length (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def vec_dot (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem plane_vectors_angle :
  (vec_length a = 1) →
  (vec_length b = 2) →
  (∀ t : ℝ, vec_length (b + t • a) ≥ vec_length (b - a)) →
  (real.angle (2 • a - b) b = (2 * real.pi) / 3) :=
begin
  intros h_a_len h_b_len h_condition,
  -- skipping proof steps
  sorry,
end

end plane_vectors_angle_l316_316296


namespace number_of_lion_cubs_l316_316197

theorem number_of_lion_cubs 
    (initial_animal_count final_animal_count : ℕ)
    (gorillas_sent hippo_adopted rhinos_taken new_animals : ℕ)
    (lion_cubs meerkats : ℕ) :
    initial_animal_count = 68 ∧ 
    gorillas_sent = 6 ∧ 
    hippo_adopted = 1 ∧ 
    rhinos_taken = 3 ∧ 
    final_animal_count = 90 ∧ 
    meerkats = 2 * lion_cubs ∧
    new_animals = lion_cubs + meerkats ∧
    final_animal_count = initial_animal_count - gorillas_sent + hippo_adopted + rhinos_taken + new_animals
    → lion_cubs = 8 := sorry

end number_of_lion_cubs_l316_316197


namespace solve_for_x_l316_316696

theorem solve_for_x (k x y : ℝ) (h1 : 9 * 3^(k * x) = 7^(y + 12)) (h2 : y = -12) : x = (-2) / k :=
by
  sorry

end solve_for_x_l316_316696


namespace max_value_ab_cd_l316_316268

theorem max_value_ab_cd : 
    ∃ a b c d : ℤ, 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    (a ∈ {-1, -2, -3, -4, -5}) ∧ 
    (b ∈ {-1, -2, -3, -4, -5}) ∧ 
    (c ∈ {-1, -2, -3, -4, -5}) ∧ 
    (d ∈ {-1, -2, -3, -4, -5}) ∧ 
    (a^b + c^d = 10 / 9) :=
begin 
  sorry
end

end max_value_ab_cd_l316_316268


namespace probability_y_ge_x_div_2_l316_316289

theorem probability_y_ge_x_div_2 : 
  let s := {(x, y) | x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6} ∧ x + y = 7} in
  let favorable := {(x, y) | (x, y) ∈ s ∧ y ≥ x / 2} in
  (favorable.to_finset.card : ℚ) / (s.to_finset.card : ℚ) = 2 / 3 :=
by
  sorry

end probability_y_ge_x_div_2_l316_316289


namespace find_g_neg_6_l316_316860

noncomputable def g : ℤ → ℤ := sorry

-- Conditions from the problem
axiom g_condition_1 : g 1 - 1 > 0
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The statement we need to prove
theorem find_g_neg_6 : g (-6) = 723 :=
by { sorry }

end find_g_neg_6_l316_316860


namespace slope_range_of_line_intersecting_circle_l316_316502

theorem slope_range_of_line_intersecting_circle
  (k : ℝ)
  (P : ℝ × ℝ) 
  (curve : ℝ → ℝ)
  (hP : P = (-√3, -1))
  (hcurve : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ 0 ≤ curve x ∧ curve x = √(1 - x^2))
  : ∃ k, (k = (√3 - 1) / 2 ∨ k = √3) ∧ (k ≥ (√3 - 1) / 2 ∧ k ≤ √3)
:= sorry

end slope_range_of_line_intersecting_circle_l316_316502


namespace intuitive_diagram_area_l316_316280

theorem intuitive_diagram_area (
    side_length : ℝ,
    (h_side_length : side_length = 2),
    oblique_projection_ratio : ℝ,
    (h_oblique_projection_ratio : oblique_projection_ratio = 2 * Real.sqrt 2)
  ) :
  let S_original := (Real.sqrt 3 / 4) * side_length^2 in
  let S_intuitive := S_original / oblique_projection_ratio in
  S_intuitive = Real.sqrt 6 / 4 := by
{
  sorry
}

end intuitive_diagram_area_l316_316280


namespace team_B_eligible_l316_316126

-- Define the conditions
def max_allowed_height : ℝ := 168
def average_height_team_A : ℝ := 166
def median_height_team_B : ℝ := 167
def tallest_sailor_in_team_C : ℝ := 169
def mode_height_team_D : ℝ := 167

-- Define the proof statement
theorem team_B_eligible : 
  (∃ (heights_B : list ℝ), heights_B.length > 0 ∧ median heights_B = median_height_team_B) →
  (∀ h ∈ heights_B, h ≤ max_allowed_height) ∨ (∃ (S : finset ℝ), S.card ≥ heights_B.length / 2 ∧ ∀ h ∈ S, h ≤ max_allowed_height) :=
sorry

end team_B_eligible_l316_316126


namespace krishan_money_l316_316093

theorem krishan_money
  (R G K : ℕ)
  (h1 : R / G = 7 / 17)
  (h2 : G / K = 7 / 17)
  (h3 : R = 637) : 
  K = 3774 := 
by
  sorry

end krishan_money_l316_316093


namespace correlation_coefficient_correctness_condition_l316_316154

theorem correlation_coefficient_correctness_condition
  (r : ℝ)
  (h1 : r > 0 → strong_linear_correlation(r))
  (h2 : r < 0 → unrelated_variables(r))
  (h3 : |r| → 1 → stronger_linear_correlation(r))
  (h4 : r → smaller_weak_linear_correlation(r))
  : (the closer |r| is to 1, the stronger the linear correlation between the two variables)
:= sorry

end correlation_coefficient_correctness_condition_l316_316154


namespace nathaniel_wins_probability_is_5_over_11_l316_316009

open ProbabilityTheory

noncomputable def nathaniel_wins_probability : ℝ :=
  if ∃ n : ℕ, (∑ k in finset.range (n + 1), k % 7) = 0 then
    5 / 11
  else
    sorry

theorem nathaniel_wins_probability_is_5_over_11 :
  nathaniel_wins_probability = 5 / 11 :=
sorry

end nathaniel_wins_probability_is_5_over_11_l316_316009


namespace van_distance_l316_316203

theorem van_distance
  (D : ℝ)  -- distance the van needs to cover
  (S : ℝ)  -- original speed
  (h1 : D = S * 5)  -- the van takes 5 hours to cover the distance D
  (h2 : D = 62 * 7.5)  -- the van should maintain a speed of 62 kph to cover the same distance in 7.5 hours
  : D = 465 :=         -- prove that the distance D is 465 kilometers
by
  sorry

end van_distance_l316_316203


namespace part1_part2_l316_316668

open Real

noncomputable def f (x : ℝ) : ℝ :=
  2 * x - (x + 1) * log x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
  x * log x - a * x^2 - 1

/- First part: Prove that for all x \in (1, +\infty), f(x) < 2 -/
theorem part1 (x : ℝ) (hx : 1 < x) : f x < 2 := sorry

/- Second part: Prove that if g(x) = 0 has two roots x₁ and x₂, then 
   (log x₁ + log x₂) / 2 > 1 + 2 / sqrt (x₁ * x₂) -/
theorem part2 (a x₁ x₂ : ℝ) (hx₁ : g x₁ a = 0) (hx₂ : g x₂ a = 0) : 
  (log x₁ + log x₂) / 2 > 1 + 2 / sqrt (x₁ * x₂) := sorry

end part1_part2_l316_316668


namespace max_balls_drawn_l316_316034

structure DrawingConditions where
  yellow_items : ℕ
  round_items : ℕ
  edible_items : ℕ
  tomato_properties : Prop
  ball_properties : Prop
  banana_properties : Prop

axiom petya_conditions : DrawingConditions → Prop 

theorem max_balls_drawn (cond : DrawingConditions)
  (h1 : cond.yellow_items = 15)
  (h2 : cond.round_items = 18)
  (h3 : cond.edible_items = 13)
  (h4 : cond.tomato_properties)
  (h5 : cond.ball_properties)
  (h6 : cond.banana_properties) :
  ∀ balls : ℕ, balls ≤ 18 :=
by
  sorry

end max_balls_drawn_l316_316034


namespace inequality_example_equality_case_l316_316169

theorem inequality_example (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 2) :
  (1 / (1 + a * b)) + (1 / (1 + b * c)) + (1 / (1 + c * a)) ≥ 27 / 13 :=
by
  -- Proof skipped
  sorry

theorem equality_case (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 2) :
  (1 / (1 + a * b)) + (1 / (1 + b * c)) + (1 / (1 + c * a)) = 27 / 13 ↔ a = b ∧ b = c ∧ c = 2 / 3 :=
by
  -- Proof skipped
  sorry

end inequality_example_equality_case_l316_316169


namespace divide_pencils_l316_316479

theorem divide_pencils (students pencils : ℕ) (h_students : students = 2) (h_pencils : pencils = 18) :
  pencils / students = 9 :=
by
  rw [h_students, h_pencils]
  norm_num
  sorry

end divide_pencils_l316_316479


namespace smallest_sum_of_factors_of_12_factorial_l316_316082

theorem smallest_sum_of_factors_of_12_factorial :
  ∃ (x y z w : Nat), x * y * z * w = Nat.factorial 12 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ x + y + z + w = 147 :=
by
  sorry

end smallest_sum_of_factors_of_12_factorial_l316_316082


namespace find_expenditure_in_January_l316_316571

-- Definitions for given conditions
def avg_exp_Jan_to_Jun : ℝ := 4200
def amt_in_July : ℝ := 1500
def avg_exp_Feb_to_Jul : ℝ := 4250
def months_Jan_to_Jun : ℝ := 6
def months_Feb_to_Jul : ℝ := 6
def total_exp_Jan_to_Jun := avg_exp_Jan_to_Jun * months_Jan_to_Jun
def total_exp_Feb_to_Jul := avg_exp_Feb_to_Jul * months_Feb_to_Jul

-- Define the amount spent in January
def amt_in_Jan : ℝ := 1200

-- Lean statement to prove the amount spent in January
theorem find_expenditure_in_January :
  (total_exp_Feb_to_Jul - total_exp_Jan_to_Jun = amt_in_July - amt_in_Jan) →
  amt_in_Jan = 1200 :=
by
  intro h
  rw [← h]
  sorry

end find_expenditure_in_January_l316_316571


namespace g_neg_six_eq_neg_twenty_l316_316849

theorem g_neg_six_eq_neg_twenty (g : ℤ → ℤ)
    (h1 : g 1 - 1 > 0)
    (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
    (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 := 
sorry

end g_neg_six_eq_neg_twenty_l316_316849


namespace car_travel_distance_l316_316527

noncomputable def distance_traveled (diameter : ℝ) (revolutions : ℝ) : ℝ :=
  let pi := Real.pi
  let circumference := pi * diameter
  circumference * revolutions / 12 / 5280

theorem car_travel_distance
  (diameter : ℝ)
  (revolutions : ℝ)
  (h_diameter : diameter = 13)
  (h_revolutions : revolutions = 775.5724667489372) :
  distance_traveled diameter revolutions = 0.5 :=
by
  simp [distance_traveled, h_diameter, h_revolutions, Real.pi]
  sorry

end car_travel_distance_l316_316527


namespace total_prime_factors_l316_316262

theorem total_prime_factors : 
  (let expr := (4:ℕ)^14 * (37:ℕ)^7 * (2:ℕ)^13 * (19:ℕ)^3 in
  ∑ p in ([(4, (4:ℕ)), (37, (37:ℕ)), (2, (2:ℕ)), (19, (19:ℕ))]).map (λ x, match x with | (k, n) => n * k.factors.length end),
  p) = 51 := 
begin
  sorry
end

end total_prime_factors_l316_316262


namespace ratios_of_PQR_and_XYZ_l316_316139

-- Define triangle sides
def sides_PQR : ℕ × ℕ × ℕ := (7, 24, 25)
def sides_XYZ : ℕ × ℕ × ℕ := (9, 40, 41)

-- Perimeter calculation functions
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- Area calculation functions for right triangles
def area (a b : ℕ) : ℕ := (a * b) / 2

-- Required proof statement
theorem ratios_of_PQR_and_XYZ :
  let (a₁, b₁, c₁) := sides_PQR
  let (a₂, b₂, c₂) := sides_XYZ
  area a₁ b₁ * 15 = 7 * area a₂ b₂ ∧ perimeter a₁ b₁ c₁ * 45 = 28 * perimeter a₂ b₂ c₂ :=
sorry

end ratios_of_PQR_and_XYZ_l316_316139


namespace prod_gcd_lcm_eq_864_l316_316250

theorem prod_gcd_lcm_eq_864 : 
  let a := 24
  let b := 36
  let gcd_ab := Nat.gcd a b
  let lcm_ab := Nat.lcm a b
  gcd_ab * lcm_ab = 864 :=
by
  sorry

end prod_gcd_lcm_eq_864_l316_316250


namespace directrix_of_parabola_l316_316827

theorem directrix_of_parabola (a : ℝ) (h : a = 1) :
  ∃ y : ℝ, 4 * y + 1 = 0 ∧ directrix y
  (parabola y) :=
sorry

end directrix_of_parabola_l316_316827


namespace smallest_angle_half_largest_l316_316467

open Real

-- Statement of the problem
theorem smallest_angle_half_largest (a b c : ℝ) (α β γ : ℝ)
  (h_sides : a = 4 ∧ b = 5 ∧ c = 6)
  (h_angles : α < β ∧ β < γ)
  (h_cos_alpha : cos α = (b^2 + c^2 - a^2) / (2 * b * c))
  (h_cos_gamma : cos γ = (a^2 + b^2 - c^2) / (2 * a * b)) :
  2 * α = γ := 
sorry

end smallest_angle_half_largest_l316_316467


namespace abs_neg_five_l316_316441

theorem abs_neg_five : abs (-5) = 5 := 
by 
  sorry

end abs_neg_five_l316_316441


namespace nat_divisible_by_five_l316_316897

theorem nat_divisible_by_five (a b : ℕ) (h : 5 ∣ (a * b)) : (5 ∣ a) ∨ (5 ∣ b) :=
by
  have h₀ : ¬ ((5 ∣ a) ∨ (5 ∣ b)) → ¬ (5 ∣ (a * b)) := sorry
  -- Proof by contradiction steps go here
  sorry

end nat_divisible_by_five_l316_316897


namespace determinant_of_A_l316_316224

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![3, 1, -2], ![8, 5, -4], ![3, 3, 6]]

theorem determinant_of_A : A.det = 48 := 
by
  sorry

end determinant_of_A_l316_316224


namespace gcm_9_15_lt_150_l316_316149

theorem gcm_9_15_lt_150 (h_lcm : Nat.lcm 9 15 = 45) : 
  ∃ k : ℕ, 45 * k < 150 ∧ 135 = 45 * 3 :=
by
  -- We know that the least common multiple of 9 and 15 is 45.
  have h_lcm_def : Nat.lcm 9 15 = 45 := h_lcm

  -- Find the largest integer k such that 45k < 150.
  let k := 3

  -- We provide the necessary proof steps to justify the calculations.
  have h_ineq : 45 * k < 150 :=
    by
    calc
      45 * k = 45 * 3 : by rfl
      ... = 135 : by rfl
      ... < 150 : by norm_num
  
  -- We conclude the proof.
  existsi k
  exact ⟨h_ineq, rfl⟩

end gcm_9_15_lt_150_l316_316149


namespace at_least_half_team_B_can_serve_on_submarine_l316_316130

theorem at_least_half_team_B_can_serve_on_submarine
    (max_height : ℕ)
    (team_A_avg_height : ℕ)
    (team_B_median_height : ℕ)
    (team_C_tallest_height : ℕ)
    (team_D_mode_height : ℕ)
    (h1 : max_height = 168)
    (h2 : team_A_avg_height = 166)
    (h3 : team_B_median_height = 167)
    (h4 : team_C_tallest_height = 169)
    (h5 : team_D_mode_height = 167) :
  ∀ (height : ℕ), height ≤ max_height → ∃ (b_sailors : ℕ → Prop) (H : ∃ n, b_sailors n),
  (∃ (n_half : ℕ), (∀ h ≤ team_B_median_height, b_sailors h) ∧ (2 * n_half ≤ n)) :=
sorry

end at_least_half_team_B_can_serve_on_submarine_l316_316130


namespace product_of_divisors_sum_l316_316464

theorem product_of_divisors_sum :
  ∃ (a b c : ℕ), (a ∣ 11^3) ∧ (b ∣ 11^3) ∧ (c ∣ 11^3) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a * b * c = 11^3) ∧ (a + b + c = 133) :=
sorry

end product_of_divisors_sum_l316_316464


namespace sale_price_sarees_at_discounts_l316_316094

noncomputable def final_sale_price (original_price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (λ price discount, price * (1 - discount / 100)) original_price

theorem sale_price_sarees_at_discounts (original_price : ℝ) (d1 d2 d3 : ℝ) (final_price : ℝ) :
  original_price = 400 → 
  d1 = 12 → 
  d2 = 5 → 
  d3 = 7 → 
  final_price = 311 →
  final_sale_price original_price [d1, d2, d3] = final_price := by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end sale_price_sarees_at_discounts_l316_316094


namespace rationalize_denominator_7_over_cube_root_343_l316_316798

theorem rationalize_denominator_7_over_cube_root_343 :
  (7 / Real.cbrt 343) = 1 :=
by {
  have h : Real.cbrt 343 = 7 := rfl,
  rw [h],
  norm_num,
  rw [div_self],
  norm_num,
  sorry
}

end rationalize_denominator_7_over_cube_root_343_l316_316798


namespace rectangle_area_l316_316937

theorem rectangle_area (radius length width : ℝ) (h1 : radius = 7)
  (h2 : width = 2 * radius) (h3 : length = 3 * width) :
  length * width = 588 :=
by
  have h4 : width = 14 := by rw [h1, h2]; norm_num
  have h5 : length = 42 := by rw [h3, h4]; norm_num
  rw [h4, h5]
  norm_num 
  sorry

end rectangle_area_l316_316937


namespace operations_correctness_l316_316507

theorem operations_correctness (a b : ℝ) : 
  ((-ab)^2 ≠ -a^2 * b^2)
  ∧ (a^3 * a^2 ≠ a^6)
  ∧ ((a^3)^4 ≠ a^7)
  ∧ (b^2 + b^2 = 2 * b^2) :=
by
  sorry

end operations_correctness_l316_316507


namespace neg_one_exponent_gt_next_l316_316631

theorem neg_one_exponent_gt_next (n : ℤ) (h : n ∈ {-1, 0, 1, 2, 3}) (h1 : (-1)^n > (-1)^(n+1)) : n = 0 ∨ n = 2 := 
by
  sorry

end neg_one_exponent_gt_next_l316_316631


namespace angle_in_second_quadrant_l316_316324

variables {α : ℝ} -- declaring α as a real number variable

def in_fourth_quadrant (α : ℝ) : Prop :=
  (sin α > 0) ∧ (tan α < 0)

theorem angle_in_second_quadrant (α : ℝ) (h : in_fourth_quadrant α) : α > π / 2 ∧ α < π :=
sorry

end angle_in_second_quadrant_l316_316324


namespace tangent_line_at_e_l316_316652

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x + real.log (-x) else -(-x + real.log x)

theorem tangent_line_at_e :
  f e = e - 1 →
  ∀ (x : ℝ), f x = -f (-x) →
  ∀ (x < 0), f x = x + real.log (-x) →
  ∃ m b : ℝ, (∀ x : ℝ, y = m * x + b) ∧ m = 1 - 1 / e :=
sorry

end tangent_line_at_e_l316_316652


namespace curve_symmetric_implies_r_eq_neg_s_l316_316455

theorem curve_symmetric_implies_r_eq_neg_s 
  (p q r s : ℝ) 
  (hp : p ≠ 0) 
  (hq : q ≠ 0) 
  (hr : r ≠ 0) 
  (hs : s ≠ 0) 
  (h_symmetry : ∀ (x y : ℝ), y = (px + q) / (rx + s) ↔ -x = (p(-y) + q) / (r(-y) + s)) : 
  r = -s := 
sorry

end curve_symmetric_implies_r_eq_neg_s_l316_316455


namespace inequality_solution_set_l316_316096

theorem inequality_solution_set (x : ℝ) : (x - 1 < 7) ∧ (3 * x + 1 ≥ -2) ↔ -1 ≤ x ∧ x < 8 :=
by
  sorry

end inequality_solution_set_l316_316096


namespace box_marbles_l316_316498

variable {A B : Type}
variable [FinType A] [FinType B]
variable (a b : ℕ) (total_marbles : a + b = 24)
variable (x y : ℕ) (black_prob : (x / a) * (y / b) = 28 / 45)

theorem box_marbles :
  ∃ m n : ℕ, gcd m n = 1 ∧ (prob_white : (m / a) * (n / b) = 2 / 135) ∧ (m + n = 137) :=
by sorry

end box_marbles_l316_316498


namespace linear_function_exists_l316_316746

theorem linear_function_exists
  (f : ℝ → ℝ)
  (h_cont : continuous f)
  (h_strict_inc : ∀ x y, x < y → f x < f y)
  (h_property : ∀ x y, f⁻¹ ((f x + f y) / 2) * (f x + f y) = (x + y) * f ((x + y) / 2)) :
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b :=
sorry

end linear_function_exists_l316_316746


namespace proof_arithmetic_sequence_l316_316286

variable {a_n : ℕ → ℤ}

def is_arithmetic_sequence (a_n : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

def sum_of_first_n_terms (a_n : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in Finset.range n, a_n i

theorem proof_arithmetic_sequence (a_n : ℕ → ℤ) (S_n : ℕ → ℤ)
  (h_arith : is_arithmetic_sequence a_n)
  (h_sum : ∀n, S_n n = sum_of_first_n_terms a_n n)
  (h_a4 : a_n 4 = 10)
  (h_S6_S3 : S_n 6 = S_n 3 + 39) :
  a_n 1 = 1 ∧ (∀ n : ℕ, a_n n = 3 * n - 2) :=
sorry

end proof_arithmetic_sequence_l316_316286


namespace multiplication_result_l316_316060

theorem multiplication_result :
  121 * 54 = 6534 := by
  sorry

end multiplication_result_l316_316060


namespace area_of_triangle_BEF_is_correct_l316_316085

noncomputable def area_triangle_BEF : ℝ :=
  let length_AB := 5
  let width_AD := 3
  let diag_AC := Real.sqrt (length_AB^2 + width_AD^2) in
  let segment_EF := diag_AC / 3 in
  let height_B := (15 / diag_AC) in
  (1 / 2) * segment_EF * height_B

theorem area_of_triangle_BEF_is_correct : area_triangle_BEF = 5 / 2 :=
sorry

end area_of_triangle_BEF_is_correct_l316_316085


namespace convex_octagon_min_obtuse_l316_316710

-- Define a type for a polygon (here specifically an octagon)
structure Polygon (n : ℕ) :=
(vertices : ℕ)
(convex : Prop)

-- Define that an octagon is a specific polygon with 8 vertices
def octagon : Polygon 8 :=
{ vertices := 8,
  convex := sorry }

-- Define the predicate for convex polygons
def is_convex (poly : Polygon 8) : Prop := poly.convex

-- Defining the statement that a convex octagon has at least 5 obtuse interior angles
theorem convex_octagon_min_obtuse (poly : Polygon 8) (h : is_convex poly) : ∃ (n : ℕ), n = 5 :=
sorry

end convex_octagon_min_obtuse_l316_316710


namespace general_term_arithmetic_seq_l316_316285

theorem general_term_arithmetic_seq (a : ℤ) (n : ℕ) :
  let a1 := a - 1
      a2 := a + 1
      a3 := 2 * a + 3
  in
  (a1 + a3) / 2 = a2 → ∀ n, a_n = -1 + (n - 1) * 2 := 
by
  sorry

end general_term_arithmetic_seq_l316_316285


namespace line_AB_bisects_CD_l316_316393

-- Circles γ₁ and γ₂ intersect at points A and B, and have a common tangent intersecting at points C and D
variable (Γ₁ Γ₂ : Circle) (A B C D : Point)
hypothesis (h1 : A ∈ Γ₁ ∧ A ∈ Γ₂)
hypothesis (h2 : B ∈ Γ₁ ∧ B ∈ Γ₂)
hypothesis (h3 : tangent Γ₁ Γ₂ C ∧ tangent Γ₁ Γ₂ D)

theorem line_AB_bisects_CD (P : Point) (hP : P = intersect (line_through A B) (segment C D)) : 
  midpoint P C D :=
sorry

end line_AB_bisects_CD_l316_316393


namespace simplified_expr_evaluation_l316_316814

theorem simplified_expr_evaluation (x : ℤ) 
    (h1 : x ∈ ({0, 2, 5, 6} : set ℤ)) 
    (h2 : x ≠ 0 ∧ x ≠ 2) : 
    (E(x) = -1/5) :=
by
  let E := λ x, (2 / (x^2 - 2*x)) - ((x - 6) / ((x - 2)^2) / ((x - 6) / (x - 2)))
  have h3 : E 5 = -1 / 5
  sorry

end simplified_expr_evaluation_l316_316814


namespace integral_p_equals_one_l316_316419

noncomputable def p (α : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 0 else α * Real.exp (-α * x)

theorem integral_p_equals_one {α : ℝ} (h : 0 < α) : 
  ∫ x in -∞..∞, p α x = 1 := by
  sorry

end integral_p_equals_one_l316_316419


namespace stratified_sampling_l316_316505

theorem stratified_sampling
  (population : Type)
  (strata : population → Prop)
  (proportion : population → ℝ)
  (total_samples : ℝ)
  (samples_from_each_stratum : Π (s : population), ℝ) :
  (∀ s, samples_from_each_stratum s = proportion s * total_samples) →
  (∃ (sample_method: population → Prop), sample_method = λ s, stratified_sampling) := 
by
  sorry

end stratified_sampling_l316_316505


namespace sum_of_series_equals_neg_one_l316_316997

noncomputable def omega : Complex := Complex.exp (2 * π * Complex.I / 17)

theorem sum_of_series_equals_neg_one :
  (∑ k in Finset.range 16, omega ^ (k + 1)) = -1 :=
by
  sorry

end sum_of_series_equals_neg_one_l316_316997


namespace sum_first_n_terms_l316_316650

noncomputable def a_n : ℕ → ℝ := λ n, 2 * n - 3
noncomputable def b_n : ℕ → ℝ := λ n, 2 * 3^(n-1)

noncomputable def S_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a_n (i + 1) + b_n (i + 1)

theorem sum_first_n_terms (n : ℕ) :
  S_n n = n^2 - 2*n + 3^n - 1 :=
sorry

end sum_first_n_terms_l316_316650


namespace increasing_intervals_find_constants_l316_316300

noncomputable def f (x a b : ℝ) : ℝ := a * (2 * (Real.cos (x / 2))^2 + Real.sin x) + b

theorem increasing_intervals (k : ℤ) :
  ∀ x : ℝ, -1 = -1 → 2 * k * Real.pi + Real.pi / 4 < x ∧ x < 2 * k * Real.pi + 5 * Real.pi / 4 → 
  let a := -1 in StrictMono (f x a 0) := sorry

theorem find_constants (a b : ℝ) :
  (∀ x ∈ Icc (0 : ℝ) Real.pi, 5 ≤ f x a b ∧ f x a b ≤ 8) → 
  (a = 3 * Real.sqrt 2 - 3 ∧ b = 5) ∨ (a = 3 - 3 * Real.sqrt 2 ∧ b = 8) := sorry

end increasing_intervals_find_constants_l316_316300


namespace smallest_n_meets_condition_l316_316615

namespace MathProblem

def numDivisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).card

def meetsCondition (n : ℕ) : Prop :=
  numDivisors (n * (n - 1)) = n + 1

theorem smallest_n_meets_condition :
  ∃ n : ℕ, n > 0 ∧ meetsCondition n ∧ (∀ m : ℕ, m > 0 ∧ meetsCondition m → n ≤ m) :=
by
  use 8
  sorry

end MathProblem

end smallest_n_meets_condition_l316_316615


namespace pentagon_diagonal_length_l316_316435

noncomputable def diagonal_length_square (A B C D X: ℝ × ℝ) (h₁ : (A - B).sqrAbs = 4)
  (h₂ : (A - X).sqrAbs = 2) (h₃ : (B - X).sqrAbs = 2) : ℝ :=
  sqrt 10

theorem pentagon_diagonal_length (A B C D X: ℝ × ℝ) 
  (h₁ : (A - B).sqrAbs = 4) (h₂ : (A - X).sqrAbs = 2) (h₃ : (B - X).sqrAbs = 2) 
  : diagonal_length_square A B C D X h₁ h₂ h₃ = sqrt 10 :=
sorry

end pentagon_diagonal_length_l316_316435


namespace chord_length_is_2sqrt3_midpoint_polar_equation_l316_316355

-- Conditions
def line_parametric (t : ℝ) : ℝ × ℝ :=
  (1/2 * t, (sqrt 3) / 2 * t)

def circle_polar (θ : ℝ) : ℝ :=
  4 * sin θ

-- Proof problem for (Ⅰ)
theorem chord_length_is_2sqrt3 :
  let l : ℝ → ℝ × ℝ := line_parametric
  let C : ℝ → ℝ := circle_polar
  ∃ (t1 t2 : ℝ),
  (l t1).1^2 + ((l t1).2 - 2)^2 = 4 ∧
  (l t2).1^2 + ((l t2).2 - 2)^2 = 4 ∧
  t1 ≠ t2 ∧
  real.dist (l t1) (l t2) = 2 * sqrt 3 := sorry

-- Proof problem for (Ⅱ)
theorem midpoint_polar_equation :
  let midpoint_polar (ρ₀ θ₀ : ℝ) : ℝ × ℝ :=
    (2 * ρ₀, θ₀)
  ∃ ρ₀ θ₀,
  (circle_polar θ₀ = ρ₀) →
  midpoint_polar ρ₀ θ₀ = (2 * sin θ₀, θ₀) := sorry

end chord_length_is_2sqrt3_midpoint_polar_equation_l316_316355


namespace max_value_of_expr_l316_316741

theorem max_value_of_expr (A M C : ℕ) (h : A + M + C = 12) : 
  A * M * C + A * M + M * C + C * A ≤ 112 :=
sorry

end max_value_of_expr_l316_316741


namespace B_I_C_concyclic_l316_316747

-- Define the triangle ABC
variable (A B C : Point)

-- Define the incenter I
variable (I : Point)
-- I is the intersection of angle bisectors of triangle ABC
def is_incenter (I : Point) : Prop :=
  is_angle_bisector A I B ∧ is_angle_bisector B I C ∧ is_angle_bisector C I A

-- Define the point S as intersection of perpendicular bisector of BC with line AI
variable (S : Point)
def is_perpendicular_bisector (S : Point) : Prop :=
  S ∈ perpendicular_bisector B C ∧ S ∈ line A I

-- Proof goal: B, I, and C are concyclic with center S
theorem B_I_C_concyclic (h₁ : is_incenter I) (h₂ : is_perpendicular_bisector S) : 
  ∃ r, circle S r B ∧ circle S r I ∧ circle S r C :=
by
  sorry

end B_I_C_concyclic_l316_316747


namespace solution_to_sin_cos_determinant_l316_316469

theorem solution_to_sin_cos_determinant (k : ℤ) :
  (∃ x : ℝ, 
    (matrix.det ![![sin x, 1], ![1, (4:ℝ) * cos x]] = 0) ∧ 
    (x = (π / 12) + k * π ∨ x = (5 * π / 12) + k * π)) :=
sorry

end solution_to_sin_cos_determinant_l316_316469


namespace functional_equation_solutions_l316_316394

theorem functional_equation_solutions (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x * f y) + f (x + y) = f (x * y)) : 
  (∀ x : ℝ, f x = 0) ∨
  (∀ x : ℝ, f x = x - 1) ∨
  (∀ x : ℝ, f x = 1 - x) :=
sorry

end functional_equation_solutions_l316_316394


namespace triangle_area_AEC_36_l316_316721

theorem triangle_area_AEC_36
  (B : MyPoint) (A C E : MyPoint)
  (h1 : ∠BAC = 90)
  (h2 : dist A C = dist C B)
  (h3 : is_perpendicular E A B)
  (h4 : dist A B = 24)
  (h5 : dist B C = 18)
  (h6 : dist C E = 6) :
  area A E C = 36 := sorry

end triangle_area_AEC_36_l316_316721


namespace nathaniel_wins_probability_l316_316001

/-- 
  Nathaniel and Obediah play a game where they take turns rolling a fair six-sided die 
  and keep a running tally. A player wins if the tally is a multiple of 7.
  If Nathaniel goes first, the probability that he wins is 5/11.
-/
theorem nathaniel_wins_probability :
  ∀ (die : ℕ → ℕ) (tally : ℕ → ℕ)
  (turn : ℕ → ℕ) (current_player : ℕ)
  (win_condition : ℕ → Prop),
  (∀ i, die i ∈ {1, 2, 3, 4, 5, 6}) →
  (∀ i, tally (i + 1) = tally i + die (i % 6)) →
  (win_condition n ↔ tally n % 7 = 0) →
  current_player 0 = 0 →  -- Nathaniel starts
  (turn i = if i % 2 = 0 then 0 else 1) →
  P(current_player wins) = 5/11 :=
by
  sorry

end nathaniel_wins_probability_l316_316001


namespace rectangular_coordinate_equation_ordinary_equation_of_curve_minimum_distance_point_minimum_distance_value_l316_316064

noncomputable theory
open Classical

section Problem
variable {θ φ : ℝ}

-- Definitions
def polar_line (ρ θ : ℝ) : Prop := ρ * (cos θ + 2 * sin θ) = 10
def param_eqns (φ : ℝ) : ℝ × ℝ := (3 * cos φ, 2 * sin φ)
def rect_line_eqn (x y : ℝ) : Prop := x + 2 * y = 10
def ordinary_curve_eqn (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 4) = 1

-- Assertions
theorem rectangular_coordinate_equation (ρ θ : ℝ) :
  (polar_line ρ θ) → (rect_line_eqn (ρ * cos θ) (ρ * sin θ)) := sorry

theorem ordinary_equation_of_curve :
  ∀ (φ : ℝ), ordinary_curve_eqn (3 * cos φ) (2 * sin φ) := sorry

theorem minimum_distance_point (φ0 : ℝ) :
  (cos φ0 = 3 / 5) → (sin φ0 = 4 / 5) → param_eqns φ0 = (9 / 5, 8 / 5) := sorry

theorem minimum_distance_value (φ0 : ℝ) :
  (cos φ0 = 3 / 5) → (sin φ0 = 4 / 5) → let M := param_eqns φ0
  in (λ x y, x + 2 * y - 10) (M.1) (M.2) = 0 → dist (M.1, M.2) {p | rect_line_eqn p.1 p.2} = sqrt 5 := sorry

end Problem

end rectangular_coordinate_equation_ordinary_equation_of_curve_minimum_distance_point_minimum_distance_value_l316_316064


namespace circle_symmetry_l316_316077

theorem circle_symmetry {a : ℝ} (h : a ≠ 0) :
  ∀ {x y : ℝ}, (x^2 + y^2 + 2*a*x - 2*a*y = 0) → (x + y = 0) :=
sorry

end circle_symmetry_l316_316077


namespace populations_equal_l316_316168

-- Define the initial populations and the rates of change
def villageX_initial_population : ℕ := 68000
def villageX_decrease_rate : ℕ := 1200
def villageY_initial_population : ℕ := 42000
def villageY_increase_rate : ℕ := 800

-- Define the years after which the populations will be equal
noncomputable def years_to_equal_population : ℕ := 13

-- Theorem stating the populations will be equal after the specified number of years
theorem populations_equal :
  villageX_initial_population - villageX_decrease_rate * years_to_equal_population =
  villageY_initial_population + villageY_increase_rate * years_to_equal_population :=
by
  -- Leaving the proof as a simple sorry for now
  sorry

-- Run this code to ensure it builds without errors
#eval populations_equal

end populations_equal_l316_316168


namespace cruzs_marbles_l316_316110

theorem cruzs_marbles (Atticus Jensen Cruz : ℕ) 
  (h1 : 3 * (Atticus + Jensen + Cruz) = 60) 
  (h2 : Atticus = Jensen / 2) 
  (h3 : Atticus = 4) : 
  Cruz = 8 := 
sorry

end cruzs_marbles_l316_316110


namespace exist_real_x_y_for_b_iff_b_in_0_1_l316_316590

theorem exist_real_x_y_for_b_iff_b_in_0_1 (b : ℝ) :
  (∃ x y : ℝ, sqrt (x * y) = b^(2 * b) ∧ log b (x^(log b y)) + log b (y^(log b x)) = 5 * b^5) ↔ (0 ≤ b ∧ b ≤ 1) :=
sorry

end exist_real_x_y_for_b_iff_b_in_0_1_l316_316590


namespace find_seq_count_l316_316187

-- Define the sequence with given conditions
def seq_satisfies_conditions (a : ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ k ≥ 2, k ≤ n → let u_k := if k = 2 then 0 else (Finset.card (Finset.filter (λ i, a i < a (i + 1)) (Finset.range (k - 1)))) in
    a k ≤ u_k + 1) ∧
  (∀ i j k, 1 ≤ i → i < j → j < k → k ≤ n → ¬ (a j < a i ∧ a i < a k))

def f (n : ℕ) : ℕ :=
  (3^(n-1) + 1) / 2

-- Theorem statement
theorem find_seq_count (n : ℕ) (a : ℕ → ℕ) :
  seq_satisfies_conditions a n → ∃ (f_n : ℕ), f_n = (3^(n-1) + 1) / 2 :=
by
  sorry -- Proof goes here

end find_seq_count_l316_316187


namespace buses_in_parking_lot_l316_316104

def initial_buses : ℕ := 7
def additional_buses : ℕ := 6
def total_buses : ℕ := initial_buses + additional_buses

theorem buses_in_parking_lot : total_buses = 13 := by
  sorry

end buses_in_parking_lot_l316_316104


namespace magnitude_sum_l316_316292

-- Lean 4 definitions
variables {V : Type*} [inner_product_space ℝ V] {a b : V}
variable (ha : ∥a∥ = 2)
variable (hb : ∥b∥ = 5)
variable (hab : ⟪a, b⟫ = -3)

theorem magnitude_sum (ha : ∥a∥ = 2) (hb : ∥b∥ = 5) (hab : ⟪a, b⟫ = -3) : 
  ∥a + b∥ = real.sqrt 23 := 
sorry

end magnitude_sum_l316_316292


namespace sum_numerator_denominator_of_repeating_decimal_equals_146_l316_316912

noncomputable def sum_of_fraction_components (x : ℚ) : ℕ :=
  let num := x.num;
  let denom := x.denom;
  num.nat_abs + denom

theorem sum_numerator_denominator_of_repeating_decimal_equals_146 :
  sum_of_fraction_components (0.474747474747 : ℚ) = 146 :=
by
  sorry

end sum_numerator_denominator_of_repeating_decimal_equals_146_l316_316912


namespace original_price_of_cycle_l316_316193

/--
A man bought a cycle for some amount and sold it at a loss of 20%.
The selling price of the cycle is Rs. 1280.
What was the original price of the cycle?
-/
theorem original_price_of_cycle
    (loss_percent : ℝ)
    (selling_price : ℝ)
    (original_price : ℝ)
    (h_loss_percent : loss_percent = 0.20)
    (h_selling_price : selling_price = 1280)
    (h_selling_eqn : selling_price = (1 - loss_percent) * original_price) :
    original_price = 1600 :=
sorry

end original_price_of_cycle_l316_316193


namespace right_cylinder_surface_areas_l316_316950

noncomputable def lateral_surface_area (r h : ℝ) := 2 * π * r * h
noncomputable def base_area (r : ℝ) := π * r ^ 2
noncomputable def total_surface_area (r h : ℝ) := lateral_surface_area r h + 2 * base_area r

theorem right_cylinder_surface_areas :
  (lateral_surface_area 3 8 = 48 * π) ∧ (total_surface_area 3 8 = 66 * π) :=
by
  split
  -- Proof for lateral_surface_area 3 8 = 48 * π
  · sorry
  -- Proof for total_surface_area 3 8 = 66 * π
  · sorry

end right_cylinder_surface_areas_l316_316950


namespace tangent_line_equation_l316_316079

theorem tangent_line_equation :
  ∀ (x : ℝ) (y : ℝ), y = 4 * x - x^3 → 
  (x = -1) → (y = -3) →
  (∀ (m : ℝ), m = 4 - 3 * (-1)^2) →
  ∃ (line_eq : ℝ → ℝ), (∀ x, line_eq x = x - 2) :=
by
  sorry

end tangent_line_equation_l316_316079


namespace train_speed_l316_316953

theorem train_speed (length : ℝ) (time : ℝ) (length_eq : length = 375.03) (time_eq : time = 5) :
  let speed_kmph := (length / 1000) / (time / 3600)
  speed_kmph = 270.02 :=
by
  sorry

end train_speed_l316_316953


namespace g_neg_six_eq_neg_twenty_l316_316851

theorem g_neg_six_eq_neg_twenty (g : ℤ → ℤ)
    (h1 : g 1 - 1 > 0)
    (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
    (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 := 
sorry

end g_neg_six_eq_neg_twenty_l316_316851


namespace linear_system_substitution_correct_l316_316624

theorem linear_system_substitution_correct (x y : ℝ)
  (h1 : y = x - 1)
  (h2 : x + 2 * y = 7) :
  x + 2 * x - 2 = 7 :=
by
  sorry

end linear_system_substitution_correct_l316_316624


namespace acute_angle_LO_CA_l316_316331

noncomputable def triangle_CAT : ℝ := sorry
noncomputable def angle_C : ℝ := 36
noncomputable def angle_A : ℝ := 56
noncomputable def CA : ℝ := 12
noncomputable def CX : ℝ := 2
noncomputable def ZC : ℝ := 2
noncomputable def midpoint (a b : ℝ) : ℝ := (a + b) / 2
noncomputable def L := midpoint (CA : ℝ) 0
noncomputable def O := midpoint (CX : ℝ) (ZC : ℝ)
noncomputable def angle_LO_CA : ℝ := 88 -- The acute angle formed by lines LO and CA

theorem acute_angle_LO_CA 
  (h_triangle : triangle_CAT = ℝ)
  (h_angle_C : angle_C = 36)
  (h_angle_A : angle_A = 56)
  (h_CA : CA = 12)
  (h_CX : CX = 2)
  (h_ZC : ZC = 2)
  (h_midpoint_L : L = 6)
  (h_midpoint_O : O = (CX + ZC) / 2) :
  angle_LO_CA = 88 := 
sorry

end acute_angle_LO_CA_l316_316331


namespace find_100m_plus_n_l316_316100

def chosen_integers : fin 15 → ℕ := sorry -- 15 integers chosen from 0 to 999
def units_digit_sum (f : fin 15 → ℕ) : ℕ := (finset.univ.sum (λ i, f i % 10)) % 10
def last_three_digits_sum (f : fin 15 → ℕ) : ℕ := (finset.univ.sum f) % 1000

theorem find_100m_plus_n :
  let P := 1 / 100 in
  100 * 1 + 100 = 200 := 
by
  sorry

end find_100m_plus_n_l316_316100


namespace bear_fur_color_is_white_l316_316943

noncomputable def north_pole_hunter := 
  ∃ (bear_position : ℝ × ℝ) 
    (hunter_position : ℝ × ℝ)
    (walk_east_then_north : ℝ × ℝ), 
    bear_position = (0, 0) ∧
    hunter_position = (0, -100) ∧
    (walk_east_then_north = (100, -100)) ∧
    (hunter_position.2 + 100 = bear_position.2) ∧ 
    (bear_position.1 - 100 = walk_east_then_north.1)   

theorem bear_fur_color_is_white : 
  north_pole_hunter → 
  ∃ (bear : string), bear = "polar bear" ∧ 
                      (polar bear's fur color is white)
:=
by {
  sorry
}

end bear_fur_color_is_white_l316_316943


namespace find_fake_coin_in_two_weighings_l316_316787

theorem find_fake_coin_in_two_weighings (coins : Fin 8 → ℝ) (h : ∃ i : Fin 8, (∀ j ≠ i, coins i < coins j)) : 
  ∃! i : Fin 8, ∀ j ≠ i, coins i < coins j :=
by
  sorry

end find_fake_coin_in_two_weighings_l316_316787


namespace angle_between_vectors_l316_316295

open Real EuclideanGeometry

variables {a b : EuclideanSpace ℝ (Fin 2)}

-- Define the conditions
def cond1 : Prop := ‖a‖ = 1
def cond2 : Prop := ‖b‖ = 2
def cond3 : Prop := ∀ t : ℝ, ‖b + t • a‖ ≥ ‖b - a‖

-- Define the goal (to prove the angle is 2π/3)
def angle_goal : Prop :=
  let angle := arccos ((2 • a - b) ⬝ b / (‖2 • a - b‖ * ‖b‖))
  angle = 2 * π / 3

-- The final theorem statement
theorem angle_between_vectors (h1 : cond1) (h2 : cond2) (h3 : cond3) : angle_goal :=
  sorry

end angle_between_vectors_l316_316295


namespace bee_directions_when_12_feet_apart_l316_316497

-- Define the position functions for both bees
def position_A (n : ℕ) : ℝ × ℝ × ℝ :=
  if n % 2 = 0 then (n, n / 2, 0) else (n / 2 + 2, 0, 0)

def position_B (n : ℕ) : ℝ × ℝ × ℝ :=
  let k  := n / 3 in
  let rem := n % 3 in
  if rem = 0 then (-k, -2 * k, k)
  else if rem = 1 then (-(k + 1), -2 * k, k)
  else (-(k + 1), -2 * (k + 1), k + 1)

-- Distance function between two points in 3D space
def dist (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

-- Proof that the bees are 12 feet apart
theorem bee_directions_when_12_feet_apart :
  ∃ (n : ℕ), dist (position_A n) (position_B n) = 12 ∧
  (position_A (n + 1) = (position_A n).1 + 2, (position_A n).2) ∧
  (position_B (n + 1) = (position_B n).1, (position_B n).2, (position_B n).3 + 1) :=
sorry

end bee_directions_when_12_feet_apart_l316_316497


namespace combination_10_5_l316_316348

theorem combination_10_5 :
  (Nat.choose 10 5) = 2520 :=
by
  sorry

end combination_10_5_l316_316348


namespace birdhouse_flown_distance_l316_316472

-- Definition of the given conditions.
def car_distance : ℕ := 200
def lawn_chair_distance : ℕ := 2 * car_distance
def birdhouse_distance : ℕ := 3 * lawn_chair_distance

-- Statement of the proof problem.
theorem birdhouse_flown_distance : birdhouse_distance = 1200 := by
  sorry

end birdhouse_flown_distance_l316_316472


namespace total_boxes_packed_l316_316536

section
variable (initial_boxes : ℕ) (cost_per_box : ℕ) (donation_multiplier : ℕ)
variable (donor_donation : ℕ) (additional_boxes : ℕ) (total_boxes : ℕ)

-- Given conditions
def initial_boxes := 400
def cost_per_box := 80 + 165  -- 245
def donation_multiplier := 4

def initial_expenditure : ℕ := initial_boxes * cost_per_box
def donor_donation : ℕ := initial_expenditure * donation_multiplier
def additional_boxes : ℕ := donor_donation / cost_per_box
def total_boxes : ℕ := initial_boxes + additional_boxes

-- Proof statement
theorem total_boxes_packed : total_boxes = 2000 :=
by
  unfold initial_boxes cost_per_box donation_multiplier initial_expenditure donor_donation additional_boxes total_boxes
  simp
  sorry  -- Since the proof is not required
end

end total_boxes_packed_l316_316536


namespace team_B_elibility_l316_316114

-- Define conditions as hypotheses
variables (avg_height_A : ℕ)
variables (median_height_B : ℕ)
variables (tallest_height_C : ℕ)
variables (mode_height_D : ℕ)
variables (max_height_allowed : ℕ)

-- Basic height statistics given in the problem
def team_A_statistics := avg_height_A = 166
def team_B_statistics := median_height_B = 167
def team_C_statistics := tallest_height_C = 169
def team_D_statistics := mode_height_D = 167

-- Height constraint condition
def height_constraint := max_height_allowed = 168

-- Mathematical equivalent proof problem: Prove that at least half of Team B sailors can serve
theorem team_B_elibility : height_constraint → team_B_statistics → (∀ (n : ℕ), median_height_B ≤ max_height_allowed) :=
by
  intros constraint_B median_B
  sorry

end team_B_elibility_l316_316114


namespace chairs_per_row_l316_316101

/-- There are 10 rows of chairs, with the first row for awardees, the second and third rows for
    administrators and teachers, the last two rows for parents, and the remaining five rows for students.
    Given that 4/5 of the student seats are occupied, and there are 15 vacant seats among the students,
    proves that the number of chairs per row is 15. --/
theorem chairs_per_row (x : ℕ) (h1 : 10 = 1 + 1 + 1 + 5 + 2)
  (h2 : 4 / 5 * (5 * x) + 1 / 5 * (5 * x) = 5 * x)
  (h3 : 1 / 5 * (5 * x) = 15) : x = 15 :=
sorry

end chairs_per_row_l316_316101


namespace g_neg_six_eq_neg_twenty_l316_316854

theorem g_neg_six_eq_neg_twenty (g : ℤ → ℤ)
    (h1 : g 1 - 1 > 0)
    (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
    (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 := 
sorry

end g_neg_six_eq_neg_twenty_l316_316854


namespace gcd_subtraction_method_gcd_euclidean_algorithm_l316_316248

theorem gcd_subtraction_method (a b : ℕ) (h₁ : a = 72) (h₂ : b = 168) : Int.gcd a b = 24 := by
  sorry

theorem gcd_euclidean_algorithm (a b : ℕ) (h₁ : a = 98) (h₂ : b = 280) : Int.gcd a b = 14 := by
  sorry

end gcd_subtraction_method_gcd_euclidean_algorithm_l316_316248


namespace find_g_neg_six_l316_316861

theorem find_g_neg_six (g : ℤ → ℤ)
  (h1 : g 1 - 1 > 0)
  (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
  (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 :=
sorry

end find_g_neg_six_l316_316861


namespace polygon_area_is_14_l316_316554

def vertices : List (ℕ × ℕ) :=
  [(1, 2), (2, 2), (3, 3), (3, 4), (4, 5), (5, 5), (6, 5), (6, 4), (5, 3),
   (4, 3), (4, 2), (3, 1), (2, 1), (1, 1)]

noncomputable def area_of_polygon (vs : List (ℕ × ℕ)) : ℝ := sorry

theorem polygon_area_is_14 :
  area_of_polygon vertices = 14 := sorry

end polygon_area_is_14_l316_316554


namespace wire_length_calc_l316_316173

-- Define the conditions
def volume_cm3 : ℝ := 66
def diameter_mm : ℝ := 1

-- Convert the conditions to SI units
def volume_m3 := volume_cm3 * (1 / 1000000) -- converting cm³ to m³
def radius_m := (diameter_mm / 1000) / 2 -- converting mm to m and then finding the radius

-- Define the mathematical problem
theorem wire_length_calc : 
  let V := volume_m3 in
  let r := radius_m in
  let π := Real.pi in
  let h := V / (π * (r * r)) in
  abs (h - 84.029) < 0.001 :=
by
  sorry

end wire_length_calc_l316_316173


namespace sum_of_squares_of_roots_l316_316213

theorem sum_of_squares_of_roots (a b c : ℝ) :
  (Polynomial.roots (3 * X ^ 3 - 6 * X ^ 2 + 9 * X + 18)).toFinset = {a, b, c} →
  a^2 + b^2 + c^2 = -2 := 
by
  sorry

end sum_of_squares_of_roots_l316_316213


namespace find_g_minus_6_l316_316845

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Conditions given in the problem
axiom cond1 : g 1 - 1 > 0
axiom cond2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom cond3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The proof we want to make (assertion)
theorem find_g_minus_6 : g (-6) = 723 := 
sorry

end find_g_minus_6_l316_316845


namespace birdhouse_flight_distance_l316_316475

variable (car_distance : ℕ)
variable (lawn_chair_distance : ℕ)
variable (birdhouse_distance : ℕ)

def problem_condition1 := car_distance = 200
def problem_condition2 := lawn_chair_distance = 2 * car_distance
def problem_condition3 := birdhouse_distance = 3 * lawn_chair_distance

theorem birdhouse_flight_distance
  (h1 : car_distance = 200)
  (h2 : lawn_chair_distance = 2 * car_distance)
  (h3 : birdhouse_distance = 3 * lawn_chair_distance) :
  birdhouse_distance = 1200 := by
  sorry

end birdhouse_flight_distance_l316_316475


namespace isosceles_triangle_perimeter_l316_316342

theorem isosceles_triangle_perimeter (a b : ℕ) (c : ℕ) 
  (h1 : a = 5) (h2 : b = 5) (h3 : c = 2) 
  (isosceles : a = b ∨ b = c ∨ c = a) 
  (triangle_inequality1 : a + b > c)
  (triangle_inequality2 : a + c > b)
  (triangle_inequality3 : b + c > a) : 
  a + b + c = 12 :=
  sorry

end isosceles_triangle_perimeter_l316_316342


namespace find_g_neg_6_l316_316857

noncomputable def g : ℤ → ℤ := sorry

-- Conditions from the problem
axiom g_condition_1 : g 1 - 1 > 0
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The statement we need to prove
theorem find_g_neg_6 : g (-6) = 723 :=
by { sorry }

end find_g_neg_6_l316_316857


namespace count_integers_in_interval_l316_316689

theorem count_integers_in_interval : 
  set.countable {x : ℤ | abs x < (5 * Real.pi)} = 31 :=
sorry

end count_integers_in_interval_l316_316689


namespace waiter_tables_l316_316204

theorem waiter_tables (total_customers : ℕ) (left_customers : ℕ) (people_per_table : ℕ) (remaining_customers : ℕ) (tables : ℕ) :
  total_customers = 62 →
  left_customers = 17 →
  people_per_table = 9 →
  remaining_customers = total_customers - left_customers →
  tables = remaining_customers / people_per_table →
  tables = 5 := by
  sorry

end waiter_tables_l316_316204


namespace projectile_first_reaches_70_feet_l316_316562

theorem projectile_first_reaches_70_feet :
  ∃ t : ℝ , (t > 0) ∧ (-16 * t^2 + 80 * t = 70) ∧ (∀ t' : ℝ, (t' > 0) ∧ (-16 * t'^2 + 80 * t' = 70) → t ≤ t') :=
sorry

end projectile_first_reaches_70_feet_l316_316562


namespace koolaid_percentage_is_correct_l316_316735

noncomputable def percentage_koolaid_powder (k w : ℕ) : ℚ :=
  let w' := 0.75 * w
  let w'' := 5 * w'
  let m := k + w''
  (k / m) * 100

theorem koolaid_percentage_is_correct :
  percentage_koolaid_powder 3 20 ≈ 3.846 := by
  sorry

end koolaid_percentage_is_correct_l316_316735


namespace selection_of_books_l316_316350

-- Define the problem context and the proof statement
theorem selection_of_books (n k : ℕ) (h_n : n = 10) (h_k : k = 5) : nat.choose n k = 252 := by
  -- Given: n = 10, k = 5
  -- Prove: (10 choose 5) = 252
  rw [h_n, h_k]
  norm_num
  sorry

end selection_of_books_l316_316350


namespace at_least_half_team_B_can_serve_l316_316134

theorem at_least_half_team_B_can_serve (height_limit : ℕ)
    (avg_A : ℕ) (median_B : ℕ) (tallest_C : ℕ) (mode_D : ℕ)
    (H1 : height_limit = 168)
    (H2 : avg_A = 166)
    (H3 : median_B = 167)
    (H4 : tallest_C = 169)
    (H5 : mode_D = 167) :
    ∃ (eligible_B : ℕ → Prop), (∀ n, eligible_B n → n ≤ height_limit) ∧ (∃ k, k ≤ median_B ∧ eligible_B k ∧ k ≥ 0) :=
by
  sorry

end at_least_half_team_B_can_serve_l316_316134


namespace num_real_roots_of_eq_l316_316874

theorem num_real_roots_of_eq (x : ℝ) (h : x * |x| - 3 * |x| - 4 = 0) : 
  ∃! x : ℝ, x * |x| - 3 * |x| - 4 = 0 :=
sorry

end num_real_roots_of_eq_l316_316874


namespace expected_score_of_losing_team_is_three_half_l316_316778

def expected_losing_score : ℚ :=
  let probability_B_0 := 1 / 10 in
  let probability_B_1 := 3 / 10 in
  let probability_B_2 := 6 / 10 in
  let expected_score := (probability_B_0 * 0) + (probability_B_1 * 1) + (probability_B_2 * 2) in
  expected_score

theorem expected_score_of_losing_team_is_three_half :
  expected_losing_score = 3 / 2 :=
by
  -- Proof goes here
  sorry

end expected_score_of_losing_team_is_three_half_l316_316778


namespace ashton_pencils_left_l316_316974

theorem ashton_pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (pencils_given : ℕ) :
  boxes = 2 → pencils_per_box = 14 → pencils_given = 6 → (boxes * pencils_per_box) - pencils_given = 22 :=
by
  intros boxes_eq pencils_per_box_eq pencils_given_eq
  rw [boxes_eq, pencils_per_box_eq, pencils_given_eq]
  norm_num
  sorry

end ashton_pencils_left_l316_316974


namespace part1_part2_l316_316667

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + a
def g (x : ℝ) : ℝ := abs (2 * x - 1)

theorem part1 (x : ℝ) : f x 2 ≤ 6 ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

theorem part2 (a : ℝ) : 2 ≤ a ↔ ∀ (x : ℝ), f x a + g x ≥ 3 := by
  sorry

end part1_part2_l316_316667


namespace midpoint_AB_equals_midpoint_PQ_l316_316766

variable {ABC : Type} [Triangle ABC] -- Define a triangle type

variable {a b c : ℝ} -- side lengths of the triangle
variable (AB BC AC : LineSegment ABC) -- the sides of the triangle
variable (P Q : Point ABC) -- Points on the triangle

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

axiom excircle_touches_AC (P : Point ABC) (p : ℝ) :
  distance P (AC.A) = p - c

axiom excircle_touches_BC (Q : Point ABC) (p : ℝ) :
  distance Q (BC.B) = p - b

theorem midpoint_AB_equals_midpoint_PQ 
  (A_mid_B : midpoint) (P_mid_Q : midpoint)
  (h1 : distance P (AC.A) = semiperimeter a b c - c)
  (h2 : distance Q (BC.B) = semiperimeter a b c - b) :
  A_mid_B = P_mid_Q :=
sorry

end midpoint_AB_equals_midpoint_PQ_l316_316766


namespace max_balls_possible_l316_316023

structure Conditions :=
  (yellow_objects : ℕ)
  (round_objects : ℕ)
  (edible_objects : ℕ)
  (all_objects : set string)
  (is_round : string → Prop)
  (is_yellow : string → Prop)
  (is_edible : string → Prop)
  (is_red : string → Prop)
  (p_types : list string)

namespace Problem
def PetyaConditions : Conditions :=
  {
    yellow_objects := 15,
    round_objects := 18,
    edible_objects := 13,
    all_objects := {"sun", "ball", "tomato", "banana"},
    is_round := λ x, x = "tomato" ∨ x = "ball",
    is_yellow := λ x, x = "banana" ∨ x = "ball",
    is_edible := λ x, x = "banana" ∨ x = "tomato",
    is_red := λ x, x = "tomato",
    p_types := ["sun", "ball", "tomato", "banana"]
  }

theorem max_balls_possible (cond : Conditions)
  (h1 : cond.yellow_objects = 15)
  (h2 : cond.round_objects = 18)
  (h3 : cond.edible_objects = 13)
  (h4 : ∀ x, x ∈ cond.all_objects → (cond.is_round x → ¬cond.is_yellow x → cond.is_edible x → ¬cond.is_red x))
  : ∃ n, n = 18 :=
by {
  sorry
}

end Problem

end max_balls_possible_l316_316023


namespace find_u_l316_316742

-- Defining the fixed unit circle C in the Cartesian plane
def C : Metric.Sphere (0 : ℝ × ℝ) 1 := sorry

-- Defining a convex polygon P each of whose sides is tangent to C
structure ConvexPolygon :=
(sides_tangent_to_C : ∀ L, L ∈ sides → ∃ p ∈ L, p ∈ C)

-- Definitions given in the problem
def N (P : ConvexPolygon) (h k : ℝ) : ℝ := sorry
def H (P : ConvexPolygon) : Set (ℝ × ℝ) := {p | ∃ h k, N P (p.1) (p.2) ≥ 1}
def F (P : ConvexPolygon) : ℝ := sorry

-- Main theorem statement
theorem find_u : ∀ (P : ConvexPolygon), 
  ((1 / F(P)) * (∫∫ dx dy in H(P), N(P, x, y))) < (8 / 3) :=
sorry

end find_u_l316_316742


namespace sum_of_series_equals_neg_one_l316_316996

noncomputable def omega : Complex := Complex.exp (2 * π * Complex.I / 17)

theorem sum_of_series_equals_neg_one :
  (∑ k in Finset.range 16, omega ^ (k + 1)) = -1 :=
by
  sorry

end sum_of_series_equals_neg_one_l316_316996


namespace total_boxes_packed_l316_316535

section
variable (initial_boxes : ℕ) (cost_per_box : ℕ) (donation_multiplier : ℕ)
variable (donor_donation : ℕ) (additional_boxes : ℕ) (total_boxes : ℕ)

-- Given conditions
def initial_boxes := 400
def cost_per_box := 80 + 165  -- 245
def donation_multiplier := 4

def initial_expenditure : ℕ := initial_boxes * cost_per_box
def donor_donation : ℕ := initial_expenditure * donation_multiplier
def additional_boxes : ℕ := donor_donation / cost_per_box
def total_boxes : ℕ := initial_boxes + additional_boxes

-- Proof statement
theorem total_boxes_packed : total_boxes = 2000 :=
by
  unfold initial_boxes cost_per_box donation_multiplier initial_expenditure donor_donation additional_boxes total_boxes
  simp
  sorry  -- Since the proof is not required
end

end total_boxes_packed_l316_316535


namespace tower_heights_651_l316_316243

theorem tower_heights_651 :
  let bricks_count := 50
  let brick_dims := (5, 12, 18)
  let orientations := {brick_dims.1, brick_dims.2, brick_dims.3}
  let smallest_height := bricks_count * brick_dims.1
  let increment_12 := brick_dims.2 - brick_dims.1
  let increment_18 := brick_dims.3 - brick_dims.1
  let total_bricks := Finset.range ((bricks_count + 1) * increment_12 + 1) ∪
                       Finset.range ((bricks_count + 1) * increment_18 + 1)
  let total_range := Finset.range (increment_12 * bricks_count + 1) ∪ 
                     Finset.range (increment_18 * bricks_count + 1)
  let bounded_total := Finset.image (λ x => smallest_height + x * increment_12 + (bricks_count - x) * increment_18) total_range
  let heights := bounded_total.filter(λ h, smallest_height <= h ∧ h <= smallest_height + bricks_count * increment_18)
  1 + (bricks_count * increment_18) - (smallest_height) = 651 := by
  sorry

end tower_heights_651_l316_316243


namespace starbursts_count_l316_316789

theorem starbursts_count (h_ratio : 13 / 8 = 143 / S) (h_mm : 143) : S = 88 := sorry

end starbursts_count_l316_316789


namespace negation_of_proposition_l316_316089

variable (f : ℕ+ → ℕ)

theorem negation_of_proposition :
  (¬ ∀ n : ℕ+, f n ≤ n) ↔ (∃ n : ℕ+, f n > n) :=
by sorry

end negation_of_proposition_l316_316089


namespace large_hole_in_paper_l316_316499

theorem large_hole_in_paper (p : Type) [Inhabited p] (n : ℕ) :
  (∃ (c : p), large_enough_hole p n) :=
sorry

end large_hole_in_paper_l316_316499


namespace length_DE_leq_semiperimeter_l316_316361

open Real

variables {A B C D E : Point}
variables (tri : Triangle A B C)
variables (h1 : ∠ A D B = 90)
variables (h2 : ∠ B E C = 90)
noncomputable def semiperimeter (T : Triangle A B C) :=
  (T.a + T.b + T.c) / 2

theorem length_DE_leq_semiperimeter (tri : Triangle A B C) (h1 : ∠ A D B = 90) (h2 : ∠ B E C = 90) : 
  (length (D E) ≤ semiperimeter tri) :=
by
  sorry

end length_DE_leq_semiperimeter_l316_316361


namespace max_balls_possible_l316_316024

structure Conditions :=
  (yellow_objects : ℕ)
  (round_objects : ℕ)
  (edible_objects : ℕ)
  (all_objects : set string)
  (is_round : string → Prop)
  (is_yellow : string → Prop)
  (is_edible : string → Prop)
  (is_red : string → Prop)
  (p_types : list string)

namespace Problem
def PetyaConditions : Conditions :=
  {
    yellow_objects := 15,
    round_objects := 18,
    edible_objects := 13,
    all_objects := {"sun", "ball", "tomato", "banana"},
    is_round := λ x, x = "tomato" ∨ x = "ball",
    is_yellow := λ x, x = "banana" ∨ x = "ball",
    is_edible := λ x, x = "banana" ∨ x = "tomato",
    is_red := λ x, x = "tomato",
    p_types := ["sun", "ball", "tomato", "banana"]
  }

theorem max_balls_possible (cond : Conditions)
  (h1 : cond.yellow_objects = 15)
  (h2 : cond.round_objects = 18)
  (h3 : cond.edible_objects = 13)
  (h4 : ∀ x, x ∈ cond.all_objects → (cond.is_round x → ¬cond.is_yellow x → cond.is_edible x → ¬cond.is_red x))
  : ∃ n, n = 18 :=
by {
  sorry
}

end Problem

end max_balls_possible_l316_316024


namespace find_f_at_1_3_l316_316290

noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ := Real.sin (ω * x + ϕ)

theorem find_f_at_1_3
  (ω ϕ : ℝ)
  (h1 : x1 = 1 ∨ x3 = 3)
  (h2 : ∀ x, f (Real.sin (ω * x + ϕ)))
  (h3 : ∀ x, x = (Range (-1)) (ω > 0))
  (h_der : has_deriv_at f (ω : ∀ x, ϕ) ((3 / 2) (f' < 0)))
  : f (1 / 3) ω ϕ = 1 / 2 :=
sorry

end find_f_at_1_3_l316_316290


namespace max_total_cut_length_l316_316174

theorem max_total_cut_length :
  let side_length := 30
  let num_pieces := 225
  let area_per_piece := (side_length ^ 2) / num_pieces
  let outer_perimeter := 4 * side_length
  let max_perimeter_per_piece := 10
  (num_pieces * max_perimeter_per_piece - outer_perimeter) / 2 = 1065 :=
by
  sorry

end max_total_cut_length_l316_316174


namespace cows_count_l316_316167

theorem cows_count (D C : ℕ) (h_legs : 2 * D + 4 * C = 2 * (D + C) + 24) : C = 12 := 
  sorry

end cows_count_l316_316167


namespace area_of_pentagon_l316_316356

noncomputable def is_convex_pentagon
  (A B C D E : ℝ × ℝ)
  (angle_A angle_B : ℝ) (len_EA len_AB len_BC len_CD len_DE : ℝ) : Prop :=
  angle_A = 120 ∧ angle_B = 120 ∧
  len_EA = 2 ∧ len_AB = 2 ∧ len_BC = 2 ∧
  len_CD = 4 ∧ len_DE = 4 -- Definitions using problem conditions
  
theorem area_of_pentagon {A B C D E : ℝ × ℝ}
  (h : is_convex_pentagon A B C D E 120 120 2 2 2 4 4) :
  let area := 7 * real.sqrt 3 in
  area = 7 * real.sqrt 3 :=
sorry

end area_of_pentagon_l316_316356


namespace max_balls_drawn_l316_316035

structure DrawingConditions where
  yellow_items : ℕ
  round_items : ℕ
  edible_items : ℕ
  tomato_properties : Prop
  ball_properties : Prop
  banana_properties : Prop

axiom petya_conditions : DrawingConditions → Prop 

theorem max_balls_drawn (cond : DrawingConditions)
  (h1 : cond.yellow_items = 15)
  (h2 : cond.round_items = 18)
  (h3 : cond.edible_items = 13)
  (h4 : cond.tomato_properties)
  (h5 : cond.ball_properties)
  (h6 : cond.banana_properties) :
  ∀ balls : ℕ, balls ≤ 18 :=
by
  sorry

end max_balls_drawn_l316_316035


namespace quadruple_solutions_l316_316609

theorem quadruple_solutions {a b c d Q : ℝ} :
  (a + b * c * d = Q ∧ b + c * d * a = Q ∧ c + d * a * b = Q ∧ d + a * b * c = Q) →
  (∃ (x : ℝ), (a = x ∧ b = x ∧ c = x ∧ d = x) ∨
              (a ≠ 0 ∧ b ≠ 0 ∧ a = b ∧ c = 1 / a ∧ d = 1 / a) ∨
              ((a = 1 ∧ b = 1 ∧ c = 1 ∧ d = x) ∨ (a = -1 ∧ b = -1 ∧ c = -1 ∧ d = x))) :=
begin
  intro h,
  sorry -- Proof omitted
end

end quadruple_solutions_l316_316609


namespace roberta_shopping_l316_316423

theorem roberta_shopping :
  ∀ (B : ℝ), 
  (B + 45.0 + B / 4.0 = 80.0) → 
  (B = 28.0) → 
  (45.0 - B = 17.0) :=
by
  intro B
  intro h1
  intro h2
  rw [h2] at h1
  have h3 : 45.0 - 28.0 = 17.0 := by norm_num
  exact h3
  sorry

end roberta_shopping_l316_316423


namespace linear_substitution_correct_l316_316622

theorem linear_substitution_correct (x y : ℝ) 
  (h1 : y = x - 1) 
  (h2 : x + 2 * y = 7) : 
  x + 2 * x - 2 = 7 := 
by
  sorry

end linear_substitution_correct_l316_316622


namespace equation_of_circle_exactly_one_externally_tangent_circle_l316_316277

-- Define the circle and the line condition
noncomputable def circle_center_on_line (a b : ℝ) : Prop :=
  a - b + 10 = 0

-- Define the distance function from a point to a line
noncomputable def distance_from_point_to_line (x y : ℝ) : ℝ :=
  abs 10 / real.sqrt(1 + 1)

-- Define the condition for an externally tangent circle
noncomputable def is_circles_externally_tangent (r : ℝ) : Prop :=
  r + 5 = distance_from_point_to_line 0 0

-- Move on to the proof statements
theorem equation_of_circle (a b : ℝ) :
  circle_center_on_line a b →
  ((-5 - a)^2 + (0 - b)^2 = 25) →
  ((x + 10)^2 + y^2 = 25 ∨ (x + 5)^2 + (y - 5)^2 = 25) := sorry

theorem exactly_one_externally_tangent_circle :
  ∃ r : ℝ, r > 0 ∧ is_circles_externally_tangent r → r = 5 * real.sqrt(2) - 5 := sorry

end equation_of_circle_exactly_one_externally_tangent_circle_l316_316277


namespace plants_in_second_garden_l316_316790

-- Define the conditions
def num_plants_first_garden : ℕ := 20
def pct_tomato_first_garden : ℝ := 0.10
def pct_tomato_second_garden : ℝ := 1 / 3
def pct_tomato_total : ℝ := 0.20

-- Define that there are P plants in the second garden
def num_plants_second_garden (P : ℕ) : ℕ := P

-- Total number of plants
def total_num_plants (P : ℕ) : ℕ := num_plants_first_garden + num_plants_second_garden P

-- Number of tomato plants in the first garden
def num_tomato_first_garden : ℕ := (pct_tomato_first_garden * num_plants_first_garden).toNat

-- Number of tomato plants in the second garden
def num_tomato_second_garden (P : ℕ) : ℕ := (pct_tomato_second_garden * P).toNat

-- Total number of tomato plants
def total_num_tomato (P : ℕ) : ℕ := num_tomato_first_garden + num_tomato_second_garden P

-- 20% of the total number of plants
def twenty_pct_total_num_plants (P : ℕ) : ℕ := (pct_tomato_total * total_num_plants P).toNat

theorem plants_in_second_garden : ∃ P : ℕ, total_num_tomato P = twenty_pct_total_num_plants P ∧ P = 15 :=
by
  sorry

end plants_in_second_garden_l316_316790


namespace sam_initial_investment_is_6000_l316_316426

variables (P : ℝ)
noncomputable def final_amount (P : ℝ) : ℝ :=
  P * (1 + 0.10 / 2) ^ (2 * 1)

theorem sam_initial_investment_is_6000 :
  final_amount 6000 = 6615 :=
by
  unfold final_amount
  sorry

end sam_initial_investment_is_6000_l316_316426


namespace symmetry_center_sum_l316_316767

def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem symmetry_center_sum :
  (f (-2015) + f (-2014) + f (-2013) + ⋯ + f 2014) = 4031 := by
  sorry

end symmetry_center_sum_l316_316767


namespace find_k_value_l316_316718

noncomputable def InverseProportionalityFunction
  (OA OC : ℝ) (f : ℝ → ℝ)
  (E F : ℝ × ℝ) (area_diff : ℝ) : Prop :=
  let rectangle := (OA, OC)
  ∧ let O := (0, 0)
  ∧ let A := (OA, 0)
  ∧ let B := (OA, OC)
  ∧ let C := (0, OC)
  ∧ f(6) = 3
  ∧ E = (6, f(6))
  ∧ let x_F := OC / f ^ -1
  ∧ F = (x_F, OC)
  ∧ let area_OEF := 0.5 * (E.1 * F.2 - F.1 * E.2)
  ∧ let area_BFE := 0.5 * ((B.1 - F.1) * (B.2 - F.2) - (B.1 - E.1) * (B.2 - E.2))
  ∧ area_diff = |area_OEF - area_BFE|

theorem find_k_value :
  InverseProportionalityFunction 6 5 (fun x => 18 / x) (6, 3) (15 / 2) (5 + 11 / 30) :=
by
  -- The proof should be provided here.
  sorry

end find_k_value_l316_316718


namespace distinct_values_of_P_l316_316602

noncomputable def P (n : ℤ) : ℂ :=
  let i := Complex.I in
  i^n * Real.cos (n * Real.pi / 2) + i^(-n) * Real.sin (n * Real.pi / 2)

theorem distinct_values_of_P : (Set.toFinset (SetOf (fun n : ℤ => P n))).card = 3 := by
  sorry

end distinct_values_of_P_l316_316602


namespace percentage_runs_by_running_l316_316916

theorem percentage_runs_by_running
  (total_runs : ℕ)
  (boundaries : ℕ)
  (sixes : ℕ)
  (runs_per_boundary : ℕ)
  (runs_per_six : ℕ)
  (eq_total_runs : total_runs = 120)
  (eq_boundaries : boundaries = 3)
  (eq_sixes : sixes = 8)
  (eq_runs_per_boundary : runs_per_boundary = 4)
  (eq_runs_per_six : runs_per_six = 6) :
  ((total_runs - (boundaries * runs_per_boundary + sixes * runs_per_six)) / total_runs * 100) = 50 :=
by
  sorry

end percentage_runs_by_running_l316_316916


namespace compound_interest_years_l316_316054

-- Define the parameters
def principal : ℝ := 7500
def future_value : ℝ := 8112
def annual_rate : ℝ := 0.04
def compounding_periods : ℕ := 1

-- Define the proof statement
theorem compound_interest_years :
  ∃ t : ℕ, future_value = principal * (1 + annual_rate / compounding_periods) ^ t ∧ t = 2 :=
by
  sorry

end compound_interest_years_l316_316054


namespace pencils_left_l316_316980

def total_pencils (boxes : ℕ) (pencils_per_box : ℕ) : ℕ :=
  boxes * pencils_per_box

def remaining_pencils (initial_pencils : ℕ) (pencils_given : ℕ) : ℕ :=
  initial_pencils - pencils_given

theorem pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (pencils_given : ℕ)
  (h_boxes : boxes = 2) (h_pencils_per_box : pencils_per_box = 14) (h_pencils_given : pencils_given = 6) :
  remaining_pencils (total_pencils boxes pencils_per_box) pencils_given = 22 :=
by
  rw [h_boxes, h_pencils_per_box, h_pencils_given]
  norm_num
  sorry

end pencils_left_l316_316980


namespace max_balls_l316_316014

theorem max_balls (total_yellow total_round total_edible : ℕ) 
  (suns balls tomatoes bananas : ℕ) :
  (total_yellow = 15) →
  (total_round = 18) →
  (total_edible = 13) →
  (tomatoes + balls ≤ total_round) →
  (tomatoes + bananas ≤ total_edible) →
  (suns + balls + tomatoes + bananas = total_yellow + total_round + total_edible) →
  (∀ b, b ∈ {balls, tomatoes, bananas, suns} → b ≥ 0) →
  (tomatoes ≤ total_round) →
  balls = 18 :=
by 
  sorry

end max_balls_l316_316014


namespace fort_worth_zoo_l316_316439

def two_legged_birds (heads legs : ℕ) : Prop :=
  ∃ x y : ℕ, x + y = heads ∧ 2 * x + 6 * y = legs ∧ x = 180

theorem fort_worth_zoo :
  two_legged_birds 250 780 :=
begin
  sorry
end

end fort_worth_zoo_l316_316439


namespace triangle_median_bc_l316_316706

-- Define the problem context and the required proof
theorem triangle_median_bc 
  (A B C D : Type)
  [has_sub A] [has_sub B] [has_sub C] 
  [has_add A] [has_add B] [has_add C] 
  [has_mul A] [has_mul B] [has_mul C] 
  [has_div A] [has_div B] [has_div C] 
  [has_pow A ℕ] [has_pow B ℕ] [has_pow C ℕ] 
  [has_sqrt A] [has_sqrt B] [has_sqrt C]
  (AB AC AD BC : A) 
  (H1 : AB = 4) 
  (H2 : AC = 7)
  (H3 : AD = 7 / 2) :
  BC = 9 := 
sorry

end triangle_median_bc_l316_316706


namespace max_balls_drawn_l316_316047

-- Defining the concepts of objects Petya can draw
inductive Object
| Sun
| Ball
| Tomato
| Banana

-- Defining properties for objects
def isYellow (o : Object) : Bool :=
  match o with
  | Object.Banana => true
  | _ => false

def isRound (o : Object) : Bool :=
  match o with
  | Object.Ball   => true
  | Object.Tomato => true
  | _             => false

def isEdible (o : Object) : Bool :=
  match o with
  | Object.Tomato => true
  | Object.Banana => true
  | _             => false

def countObjects (p : Object -> Bool) (os : List Object) : Nat :=
  os.countp p

theorem max_balls_drawn (os : List Object) :
  countObjects isYellow os = 15 →
  countObjects isRound os = 18 →
  countObjects isEdible os = 13 →
  countObjects (λ o => o = Object.Ball) os = 18 :=
by
  intros hy hr he
  -- Proof will go here
  sorry

end max_balls_drawn_l316_316047


namespace small_cubes_one_face_painted_red_l316_316550

-- Definitions
def is_red_painted (cube : ℕ) : Bool := true -- representing the condition that the cube is painted red
def side_length (cube : ℕ) : ℕ := 4 -- side length of the original cube is 4 cm
def smaller_cube_side_length : ℕ := 1 -- smaller cube side length is 1 cm

-- Theorem Statement
theorem small_cubes_one_face_painted_red :
  ∀ (large_cube : ℕ), (side_length large_cube = 4) ∧ is_red_painted large_cube → 
  (∃ (number_of_cubes : ℕ), number_of_cubes = 24) :=
by
  sorry

end small_cubes_one_face_painted_red_l316_316550


namespace functional_equation_solution_l316_316261

theorem functional_equation_solution :
  ∃ Q : Polynomial ℝ, Q.degree = 2015 ∧ ∀ x : ℝ, P(x) = Q(1/2 - x) + 1/2 ∧ 
  Q (1/2 - x) = -Q(x - 1/2) := 
sorry

end functional_equation_solution_l316_316261


namespace compare_negatives_l316_316589

theorem compare_negatives : -3 < -2 :=
by {
  -- Placeholder for proof
  sorry
}

end compare_negatives_l316_316589


namespace remainder_of_P_div_by_D_is_333_l316_316910

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 8 * x^4 - 18 * x^3 + 27 * x^2 - 14 * x - 30

-- Define the divisor D(x) and simplify it, but this is not necessary for the theorem statement.
-- def D (x : ℝ) : ℝ := 4 * x - 12  

-- Prove the remainder is 333 when x = 3
theorem remainder_of_P_div_by_D_is_333 : P 3 = 333 := by
  sorry

end remainder_of_P_div_by_D_is_333_l316_316910


namespace unit_cube_same_color_distance_1_4_unit_cube_same_color_not_distance_1_5_l316_316902

theorem unit_cube_same_color_distance_1_4 (color : ℝ × ℝ × ℝ → ℕ) (h_color : ∀ x, color x ∈ {1, 2, 3}) :
  ∃ x y : ℝ × ℝ × ℝ, x ≠ y ∧ dist x y ≥ 1.4 ∧ color x = color y := 
sorry

theorem unit_cube_same_color_not_distance_1_5 (color : ℝ × ℝ × ℝ → ℕ) (h_color : ∀ x, color x ∈ {1, 2, 3}) :
  ¬ ∀ x y : ℝ × ℝ × ℝ, x ≠ y → (color x = color y → dist x y ≥ 1.5) :=
sorry

end unit_cube_same_color_distance_1_4_unit_cube_same_color_not_distance_1_5_l316_316902


namespace normal_parts_outside_range_proof_l316_316353

noncomputable def normal_parts_outside_range (μ σ : ℝ) : ℕ :=
  let percent_outside_range := 0.27 / 100
  let total_parts := 1000
  Float.ceil (total_parts * percent_outside_range.toFloat)

theorem normal_parts_outside_range_proof (μ σ : ℝ) :
  normal_parts_outside_range μ σ = 3 :=
by
  sorry

end normal_parts_outside_range_proof_l316_316353


namespace at_least_half_team_B_can_serve_l316_316138

theorem at_least_half_team_B_can_serve (height_limit : ℕ)
    (avg_A : ℕ) (median_B : ℕ) (tallest_C : ℕ) (mode_D : ℕ)
    (H1 : height_limit = 168)
    (H2 : avg_A = 166)
    (H3 : median_B = 167)
    (H4 : tallest_C = 169)
    (H5 : mode_D = 167) :
    ∃ (eligible_B : ℕ → Prop), (∀ n, eligible_B n → n ≤ height_limit) ∧ (∃ k, k ≤ median_B ∧ eligible_B k ∧ k ≥ 0) :=
by
  sorry

end at_least_half_team_B_can_serve_l316_316138


namespace max_balls_l316_316013

theorem max_balls (total_yellow total_round total_edible : ℕ) 
  (suns balls tomatoes bananas : ℕ) :
  (total_yellow = 15) →
  (total_round = 18) →
  (total_edible = 13) →
  (tomatoes + balls ≤ total_round) →
  (tomatoes + bananas ≤ total_edible) →
  (suns + balls + tomatoes + bananas = total_yellow + total_round + total_edible) →
  (∀ b, b ∈ {balls, tomatoes, bananas, suns} → b ≥ 0) →
  (tomatoes ≤ total_round) →
  balls = 18 :=
by 
  sorry

end max_balls_l316_316013


namespace meeting_people_count_l316_316981

theorem meeting_people_count (k : ℕ) (hk : ∃ k, 12 * k = 36) : 
  let N := 12 * k in
  let d := 3 * k + 6 in
  let n := (9 * k^2 + 30 * k + 30) / (12 * k - 7) in
  N = 36 :=
by
  intros
  have h1 : N = 36 := sorry
  exact h1

end meeting_people_count_l316_316981


namespace m_n_not_both_odd_l316_316763

open Nat

theorem m_n_not_both_odd (m n : ℕ) (h : (1 / m : ℚ) + (1 / n) = 1 / 2020) : ¬ (odd m ∧ odd n) :=
sorry

end m_n_not_both_odd_l316_316763


namespace stanley_run_walk_difference_l316_316817

theorem stanley_run_walk_difference :
  ∀ (ran walked : ℝ), ran = 0.4 → walked = 0.2 → ran - walked = 0.2 :=
by
  intros ran walked h_ran h_walk
  rw [h_ran, h_walk]
  norm_num

end stanley_run_walk_difference_l316_316817


namespace calculate_f2016_l316_316399

-- Definitions according to the conditions
def g (n : ℕ) : ℕ :=
  if n % 2 = 1 then n else g (n / 2)

def f (k : ℕ) : ℕ :=
  (List.range (2^k)).map g |> List.sum

-- Assertion we need to prove
theorem calculate_f2016 : f 2016 = (4^2015 * 4 / 3 - 1 / 3) := sorry

end calculate_f2016_l316_316399


namespace find_g_neg6_l316_316835

-- Define the function g with conditions as hypotheses
variables {g : ℤ → ℤ}

-- Condition 1
axiom g_condition_1 : g 1 - 1 > 0

-- Condition 2
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x

-- Condition 3
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The goal is to find g(-6)
theorem find_g_neg6 : g (-6) = 723 :=
by {
  sorry
}

end find_g_neg6_l316_316835


namespace trip_cost_equation_minimum_trip_cost_l316_316202

theorem trip_cost_equation (x : ℝ) (h₁ : 50 ≤ x) (h₂ : x ≤ 100) : 
  let y := (24600 / x) + ((40 * x) / 7) in true := 
by sorry

theorem minimum_trip_cost : ∃ x y, 
  50 ≤ x ∧ x ≤ 100 ∧ y = (24600 / x) + ((40 * x) / 7) ∧ 
  x = 66 ∧ y = 750 :=
by sorry

end trip_cost_equation_minimum_trip_cost_l316_316202


namespace integer_solutions_of_abs_lt_5pi_l316_316685

theorem integer_solutions_of_abs_lt_5pi : 
  let x := 5 * Real.pi in
  ∃ n : ℕ, (∀m : ℤ, abs m < x ↔ m ∈ (Icc (-(n : ℤ)) n)) ∧ n = 15 :=
by
  sorry

end integer_solutions_of_abs_lt_5pi_l316_316685


namespace intersection_of_M_and_P_l316_316311

theorem intersection_of_M_and_P :
  let M : Set ℝ := {y : ℝ | y > 0}
  let P : Set ℝ := [1, +∞)
  M ∩ P = [1, +∞) := 
by {
  sorry
}

end intersection_of_M_and_P_l316_316311


namespace omega_range_l316_316630

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem omega_range (ω : ℝ) (h1 : ω > 0)
  (h2 : ∀ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 4), f ω x ≥ -2) :
  0 < ω ∧ ω ≤ 3 / 2 :=
by
  sorry

end omega_range_l316_316630


namespace volume_of_cone_l316_316599

theorem volume_of_cone (d slant_height : ℝ) (V : ℝ) : 
  d = 12 ∧ slant_height = 10 → V = 96 * Real.pi :=
by
  intros h,
  sorry

end volume_of_cone_l316_316599


namespace prove_cartesian_eq_l316_316719

-- Given the parametric equations
def parametric_line (t : ℝ) := 
  (x : ℝ) ∃ y : ℝ, x = -2 + (real.sqrt 2 / 2) * t ∧ y = -2 - (real.sqrt 2 / 2) * t

-- Given the polar coordinate equation
axiom polar_eq (ρ θ : ℝ) : ρ * real.sin θ ^ 2 = 4 * real.cos θ

noncomputable def cartesian_eq (x y : ℝ) : Prop :=
  y^2 = 4 * x

noncomputable def general_eq_line (x y : ℝ) : Prop :=
  x + y + 4 = 0

-- Minimum distance from a point on curve to line
def minimum_distance (x y : ℝ) : ℝ :=
  abs ((x + y + 4) / real.sqrt 2)

def on_curve (x y : ℝ) : Prop :=
  y^2 = 4 * x

def main_goal : Prop :=
  ∃ (x y : ℝ), (on_curve x y) ∧ (minimum_distance x y = (3 * real.sqrt 2) / 2)

theorem prove_cartesian_eq : (∀ x y t : ℝ, parametric_line t) → (∀ ρ θ : ℝ, polar_eq ρ θ → cartesian_eq x y) → main_goal :=
by
  intros,
  sorry

end prove_cartesian_eq_l316_316719


namespace hyperbola_eccentricity_l316_316760

theorem hyperbola_eccentricity (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 0 < e) 
  (he : e = (2 * c^2) / (c^2 - 2 * a^2)) (hf : c = a * sqrt 1) : 
  e = sqrt 3 + 1 :=
  sorry

end hyperbola_eccentricity_l316_316760


namespace greatest_possible_sum_of_squares_l316_316484

theorem greatest_possible_sum_of_squares (x y : ℤ) (h : x^2 + y^2 = 36) : x + y ≤ 8 :=
by sorry

end greatest_possible_sum_of_squares_l316_316484


namespace number_of_moles_NaNO3_formed_l316_316614

theorem number_of_moles_NaNO3_formed 
  (moles_NH4NO3 : ℝ) (moles_NaOH : ℝ) 
  (reaction : ℝ → ℝ → ℝ → Prop) :
  moles_NH4NO3 = 2 → 
  moles_NaOH = 2 → 
  (∀ a b c, reaction a b c → a = b → b = c) → 
  reaction 2 2 2 →
  2 = 2 := 
by 
  intros h1 h2 h3 h4 
  exact h1

end number_of_moles_NaNO3_formed_l316_316614


namespace num_real_roots_l316_316460

theorem num_real_roots (f : ℝ → ℝ)
  (h_eq : ∀ x, f x = 2 * x ^ 3 - 6 * x ^ 2 + 7)
  (h_interval : ∀ x, 0 < x ∧ x < 2 → f x < 0 ∧ f (2 - x) > 0) : 
  ∃! x, 0 < x ∧ x < 2 ∧ f x = 0 :=
sorry

end num_real_roots_l316_316460


namespace trapezoid_angles_equal_l316_316080

theorem trapezoid_angles_equal (A B C D P Q M : Type)
    [trapezoid A B C D]
    (h1 : extension_intersects AD BC P)
    (h2 : diagonals_intersect AC BD Q)
    (h3 : M_on_base_with_proportion BC M AM MD) :
    ∠(P, M, B) = ∠(Q, M, B) := 
sorry

end trapezoid_angles_equal_l316_316080


namespace intersect_single_point_l316_316329

theorem intersect_single_point (m : ℝ) : 
  (∃ x : ℝ, (m - 3) * x^2 - 4 * x + 2 = 0) ∧ ∀ x₁ x₂ : ℝ, 
  (m - 3) * x₁^2 - 4 * x₁ + 2 = 0 → (m - 3) * x₂^2 - 4 * x₂ + 2 = 0 → x₁ = x₂ ↔ m = 3 ∨ m = 5 := 
sorry

end intersect_single_point_l316_316329


namespace at_least_half_team_B_can_serve_l316_316137

theorem at_least_half_team_B_can_serve (height_limit : ℕ)
    (avg_A : ℕ) (median_B : ℕ) (tallest_C : ℕ) (mode_D : ℕ)
    (H1 : height_limit = 168)
    (H2 : avg_A = 166)
    (H3 : median_B = 167)
    (H4 : tallest_C = 169)
    (H5 : mode_D = 167) :
    ∃ (eligible_B : ℕ → Prop), (∀ n, eligible_B n → n ≤ height_limit) ∧ (∃ k, k ≤ median_B ∧ eligible_B k ∧ k ≥ 0) :=
by
  sorry

end at_least_half_team_B_can_serve_l316_316137


namespace impossible_to_turn_off_all_lamps_l316_316592

-- Define the 8x8 chessboard and the initial configuration
def chessboard : Type := Fin 8 → Fin 8 → Int

-- Initial configuration of the chessboard
def initial_configuration : chessboard := 
  fun i j => if i = 0 ∧ j = 3 then -1 else 1

-- Define operations that invert rows, columns, and diagonals
def invert_row (board : chessboard) (r : Fin 8) : chessboard :=
  fun i j => if i = r then -board i j else board i j

def invert_column (board : chessboard) (c : Fin 8) : chessboard :=
  fun i j => if j = c then -board i j else board i j

def invert_diagonal (board : chessboard) (d : Int) (parallel : Bool) : chessboard :=
  fun i j => if parallel then
              if i + j = d then -board i j else board i j
            else
              if i - j = d then -board i j else board i j

-- The goal to prove: It is impossible to turn all lamps off
theorem impossible_to_turn_off_all_lamps : 
  ¬(∃ final_board : chessboard, (∀ i j, final_board i j = -1) ∧ 
  (∃ inversions : List (Fin 8 × Fin 8 × Bool), 
    (initial_configuration.iterate inversions (fun b (r, c, is_diag) =>
      if is_diag
      then invert_diagonal b (r - c) (r = c)
      else invert_row (invert_column b c) r) = final_board))) :=
by
  sorry

end impossible_to_turn_off_all_lamps_l316_316592


namespace expand_expression_l316_316237

variable (x y : ℝ)

theorem expand_expression :
  ((6 * x + 8 - 3 * y) * (4 * x - 5 * y)) = 
  (24 * x^2 - 42 * x * y + 32 * x - 40 * y + 15 * y^2) :=
by
  sorry

end expand_expression_l316_316237


namespace average_score_correct_l316_316707

-- Define the conditions as constants
variables {T : ℕ} -- Total number of students
constant male_percentage : ℝ := 0.45
constant avg_male_score : ℝ := 72
constant avg_female_score : ℝ := 74
constant avg_class_score : ℝ := 73.1

-- The proof problem statement
theorem average_score_correct : 
  (male_percentage * T * avg_male_score + (1 - male_percentage) * T * avg_female_score) / T = avg_class_score :=
by
  sorry

end average_score_correct_l316_316707


namespace roque_commute_time_l316_316374

theorem roque_commute_time :
  let walk_time := 2
  let bike_time := 1
  let walks_per_week := 3
  let bike_rides_per_week := 2
  let total_walk_time := 2 * walks_per_week * walk_time
  let total_bike_time := 2 * bike_rides_per_week * bike_time
  total_walk_time + total_bike_time = 16 :=
by sorry

end roque_commute_time_l316_316374


namespace supplements_of_congruent_angles_are_congruent_l316_316239

theorem supplements_of_congruent_angles_are_congruent (a b : ℝ) (h1 : a + \degree(180) = b + \degree(180)) : a = b :=
sorry

end supplements_of_congruent_angles_are_congruent_l316_316239


namespace max_balls_possible_l316_316020

structure Conditions :=
  (yellow_objects : ℕ)
  (round_objects : ℕ)
  (edible_objects : ℕ)
  (all_objects : set string)
  (is_round : string → Prop)
  (is_yellow : string → Prop)
  (is_edible : string → Prop)
  (is_red : string → Prop)
  (p_types : list string)

namespace Problem
def PetyaConditions : Conditions :=
  {
    yellow_objects := 15,
    round_objects := 18,
    edible_objects := 13,
    all_objects := {"sun", "ball", "tomato", "banana"},
    is_round := λ x, x = "tomato" ∨ x = "ball",
    is_yellow := λ x, x = "banana" ∨ x = "ball",
    is_edible := λ x, x = "banana" ∨ x = "tomato",
    is_red := λ x, x = "tomato",
    p_types := ["sun", "ball", "tomato", "banana"]
  }

theorem max_balls_possible (cond : Conditions)
  (h1 : cond.yellow_objects = 15)
  (h2 : cond.round_objects = 18)
  (h3 : cond.edible_objects = 13)
  (h4 : ∀ x, x ∈ cond.all_objects → (cond.is_round x → ¬cond.is_yellow x → cond.is_edible x → ¬cond.is_red x))
  : ∃ n, n = 18 :=
by {
  sorry
}

end Problem

end max_balls_possible_l316_316020


namespace postal_rate_correct_l316_316458

-- Define the conditions as Lean 4 definitions
def postal_rate (W : ℝ) : ℝ := max 20 (10 * ⌈W⌉)

-- The Lean theorem statement
theorem postal_rate_correct (W : ℝ) : 
    postal_rate W = max 20 (10 * ⌈W⌉) :=
by
  sorry

end postal_rate_correct_l316_316458


namespace total_boxes_packed_l316_316534

section
variable (initial_boxes : ℕ) (cost_per_box : ℕ) (donation_multiplier : ℕ)
variable (donor_donation : ℕ) (additional_boxes : ℕ) (total_boxes : ℕ)

-- Given conditions
def initial_boxes := 400
def cost_per_box := 80 + 165  -- 245
def donation_multiplier := 4

def initial_expenditure : ℕ := initial_boxes * cost_per_box
def donor_donation : ℕ := initial_expenditure * donation_multiplier
def additional_boxes : ℕ := donor_donation / cost_per_box
def total_boxes : ℕ := initial_boxes + additional_boxes

-- Proof statement
theorem total_boxes_packed : total_boxes = 2000 :=
by
  unfold initial_boxes cost_per_box donation_multiplier initial_expenditure donor_donation additional_boxes total_boxes
  simp
  sorry  -- Since the proof is not required
end

end total_boxes_packed_l316_316534


namespace find_percentage_l316_316529

theorem find_percentage (x : ℝ) (percentage : ℝ) : 
  (percentage / 100) * x = 0.20 * 552.50 → x = 170 → percentage = 65 :=
begin
  intros h1 h2,
  sorry
end

end find_percentage_l316_316529


namespace probability_two_cities_less_than_8000_l316_316471

-- Define the city names
inductive City
| Bangkok | CapeTown | Honolulu | London | NewYork
deriving DecidableEq, Inhabited

-- Define the distance between cities
def distance : City → City → ℕ
| City.Bangkok, City.CapeTown  => 6300
| City.Bangkok, City.Honolulu  => 6609
| City.Bangkok, City.London    => 5944
| City.Bangkok, City.NewYork   => 8650
| City.CapeTown, City.Bangkok  => 6300
| City.CapeTown, City.Honolulu => 11535
| City.CapeTown, City.London   => 5989
| City.CapeTown, City.NewYork  => 7800
| City.Honolulu, City.Bangkok  => 6609
| City.Honolulu, City.CapeTown => 11535
| City.Honolulu, City.London   => 7240
| City.Honolulu, City.NewYork  => 4980
| City.London, City.Bangkok    => 5944
| City.London, City.CapeTown   => 5989
| City.London, City.Honolulu   => 7240
| City.London, City.NewYork    => 3470
| City.NewYork, City.Bangkok   => 8650
| City.NewYork, City.CapeTown  => 7800
| City.NewYork, City.Honolulu  => 4980
| City.NewYork, City.London    => 3470
| _, _                         => 0

-- Prove the probability
theorem probability_two_cities_less_than_8000 :
  let pairs := [(City.Bangkok, City.CapeTown), (City.Bangkok, City.Honolulu), (City.Bangkok, City.London), (City.CapeTown, City.London), (City.CapeTown, City.NewYork), (City.Honolulu, City.London), (City.Honolulu, City.NewYork), (City.London, City.NewYork)]
  (pairs.length : ℚ) / 10 = 4 / 5 :=
sorry

end probability_two_cities_less_than_8000_l316_316471


namespace combination_10_5_l316_316349

theorem combination_10_5 :
  (Nat.choose 10 5) = 2520 :=
by
  sorry

end combination_10_5_l316_316349


namespace proof_smallest_integer_a_l316_316259

noncomputable def proof_system_of_equations : Prop :=
  ∀ (a x y : ℝ),
    ( (y / (a - real.sqrt x - 1) = 4) ∧
      (y = (real.sqrt x + 5) / (real.sqrt x + 1)) → 
      (∀ a, ∃ x y, y / (a - real.sqrt x - 1) = 4 ∧ y = (real.sqrt x + 5) / (real.sqrt x + 1) → a = 3)
    )

theorem proof_smallest_integer_a : proof_system_of_equations :=
by 
  sorry

end proof_smallest_integer_a_l316_316259


namespace max_balls_drawn_l316_316037

-- Conditions:
variable (items : Type) 
variable (Petya : items → Prop)
variable (yellow round edible : items → Prop)
variable (sun ball tomato banana : items)

variable (Ht : ∀ x, tomato x → round x ∧ ¬yellow x) -- All tomatoes are round and red
variable (Hb : ∀ x, banana x → yellow x ∧ ¬round x) -- All bananas are yellow and not round
variable (Hba : ∀ x, ball x → round x) -- All balls are round

variable (yellow_count : ∑ x in items, yellow x = 15) -- Exactly 15 yellow items
variable (round_count : ∑ x in items, round x = 18) -- Exactly 18 round items
variable (edible_count : ∑ x in items, edible x = 13) -- Exactly 13 edible items

-- Proving the maximum number of balls
theorem max_balls_drawn : ∑ x in items, ball x ≤ 18 :=
by sorry

end max_balls_drawn_l316_316037


namespace magnitude_of_2a_minus_b_l316_316312

variables {V : Type*} [inner_product_space ℝ V] (a b : V)
open Real

-- Conditions
axiom hab : ∥a∥ = 2
axiom hbb : ∥b∥ = 1
axiom hab_dot : ⟪a, b⟫ = 1

-- Statement to prove
theorem magnitude_of_2a_minus_b : ∥2 • a - b∥ = sqrt 13 :=
by
  -- Placeholder for the actual proof
  sorry

end magnitude_of_2a_minus_b_l316_316312


namespace bug_total_distance_l316_316176

theorem bug_total_distance : 
  let start := 4
  let mid1 := -3
  let mid2 := 6
  let final := 2
  abs(mid1 - start) + abs(mid2 - mid1) + abs(final - mid2) = 20 :=
by
  sorry

end bug_total_distance_l316_316176


namespace optimal_location_D_l316_316941

noncomputable def distance_to_minimize_cost := sorry

theorem optimal_location_D:
  ∃ D : ℝ, D = 15 ∧ D ∈ set.Icc 0 100 :=
begin
  use 15,
  split,
  { refl },
  { split; linarith },
  sorry
end

end optimal_location_D_l316_316941


namespace total_income_correct_average_daily_income_correct_l316_316177

-- Define the income for each day with conditions
def daily_income : ℕ → ℝ
| 1 := 200 * 1.1       -- Rainy
| 2 := 150
| 3 := 750
| 4 := 400
| 5 := 500 * 0.95      -- Cloudy
| 6 := 300 * 1.1       -- Rainy - Weekend
| 7 := 650
| 8 := 350 * 0.95      -- Cloudy
| 9 := 600 * 1.2       -- Sunny - Peak hours
| 10 := 450
| 11 := 530
| 12 := 480 * 0.95     -- Cloudy - Weekend
| _ := 0

-- Calculate the total income for 12 days
def total_income : ℝ := (List.sum (List.map daily_income [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))

-- Calculate the average daily income
def average_daily_income : ℝ := total_income / 12

-- Proof objectives
theorem total_income_correct : total_income = 4963.50 := 
sorry

theorem average_daily_income_correct : average_daily_income = 413.625 := 
sorry

end total_income_correct_average_daily_income_correct_l316_316177


namespace team_B_elibility_l316_316117

-- Define conditions as hypotheses
variables (avg_height_A : ℕ)
variables (median_height_B : ℕ)
variables (tallest_height_C : ℕ)
variables (mode_height_D : ℕ)
variables (max_height_allowed : ℕ)

-- Basic height statistics given in the problem
def team_A_statistics := avg_height_A = 166
def team_B_statistics := median_height_B = 167
def team_C_statistics := tallest_height_C = 169
def team_D_statistics := mode_height_D = 167

-- Height constraint condition
def height_constraint := max_height_allowed = 168

-- Mathematical equivalent proof problem: Prove that at least half of Team B sailors can serve
theorem team_B_elibility : height_constraint → team_B_statistics → (∀ (n : ℕ), median_height_B ≤ max_height_allowed) :=
by
  intros constraint_B median_B
  sorry

end team_B_elibility_l316_316117


namespace sum_of_integers_l316_316108

theorem sum_of_integers (p q r : ℕ) (h : 2 * real.sqrt (real.cbrt 7 - real.cbrt 3) = real.cbrt p + real.cbrt q - real.cbrt r) : 
  p + q + r = 63 := 
  sorry

end sum_of_integers_l316_316108


namespace number_of_valid_subsets_l316_316317

def original_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

def total_sum : ℕ := 78

def remaining_sum (x : ℕ) : ℕ := (12 - x) * 7

theorem number_of_valid_subsets : (Finset.card (Finset.filter (λ x, ∃ S ∈ original_set.powerset, S.sum = total_sum - remaining_sum x) (Finset.range 13)) = 4) :=
sorry

end number_of_valid_subsets_l316_316317


namespace students_average_vegetables_l316_316493

variable (points_needed : ℕ) (points_per_vegetable : ℕ) (students : ℕ) (school_days : ℕ) (school_weeks : ℕ)

def average_vegetables_per_student_per_week (points_needed points_per_vegetable students school_days school_weeks : ℕ) : ℕ :=
  let total_vegetables := points_needed / points_per_vegetable
  let vegetables_per_student := total_vegetables / students
  vegetables_per_student / school_weeks

theorem students_average_vegetables 
  (h1 : points_needed = 200) 
  (h2 : points_per_vegetable = 2) 
  (h3 : students = 25) 
  (h4 : school_days = 10) 
  (h5 : school_weeks = 2) : 
  average_vegetables_per_student_per_week points_needed points_per_vegetable students school_days school_weeks = 2 :=
by
  sorry

end students_average_vegetables_l316_316493


namespace root_of_quadratic_l316_316968

theorem root_of_quadratic :
  (∀ x : ℝ, 2 * x^2 + 3 * x - 65 = 0 → x = 5 ∨ x = -6.5) :=
sorry

end root_of_quadratic_l316_316968


namespace divisor_greater_2016_l316_316711

theorem divisor_greater_2016 (d : ℕ) (h : 2016 / d = 0) : d > 2016 :=
sorry

end divisor_greater_2016_l316_316711


namespace richard_twice_scott_l316_316180

theorem richard_twice_scott :
  let d := 14 in   -- David's current age
  let r := d + 6 in  -- Richard's current age
  let s := d - 8 in  -- Scott's current age
  ∃ x : ℕ, (r + x = 2 * (s + x)) ∧ x = 8 :=
by
  sorry

end richard_twice_scott_l316_316180


namespace ratio_of_square_sides_l316_316200

theorem ratio_of_square_sides
  (a b : ℝ) 
  (h1 : ∃ square1 : ℝ, square1 = 2 * a)
  (h2 : ∃ square2 : ℝ, square2 = 2 * b)
  (h3 : a ^ 2 - 4 * a * b - 5 * b ^ 2 = 0) :
  2 * a / 2 * b = 5 :=
by
  sorry

end ratio_of_square_sides_l316_316200


namespace find_k_and_max_ck_largest_ck_for_k0_largest_ck_for_k2_l316_316166

theorem find_k_and_max_ck:
  (∀ (x y z : ℝ), (x > 0) → (y > 0) → (z > 0) →
    ∃ (c_k : ℝ), c_k > 0 ∧ (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ c_k * (x + y + z)^k) →
  (∀ (k : ℝ), 0 ≤ k ∧ k ≤ 2) :=
by
  sorry

theorem largest_ck_for_k0:
  (∀ (x y z : ℝ), (x > 0) → (y > 0) → (z > 0) →
    (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ 1) := 
by
  sorry

theorem largest_ck_for_k2:
  (∀ (x y z : ℝ), (x > 0) → (y > 0) → (z > 0) →
    (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ (8/9) * (x + y + z)^2) :=
by
  sorry

end find_k_and_max_ck_largest_ck_for_k0_largest_ck_for_k2_l316_316166


namespace max_balls_possible_l316_316022

structure Conditions :=
  (yellow_objects : ℕ)
  (round_objects : ℕ)
  (edible_objects : ℕ)
  (all_objects : set string)
  (is_round : string → Prop)
  (is_yellow : string → Prop)
  (is_edible : string → Prop)
  (is_red : string → Prop)
  (p_types : list string)

namespace Problem
def PetyaConditions : Conditions :=
  {
    yellow_objects := 15,
    round_objects := 18,
    edible_objects := 13,
    all_objects := {"sun", "ball", "tomato", "banana"},
    is_round := λ x, x = "tomato" ∨ x = "ball",
    is_yellow := λ x, x = "banana" ∨ x = "ball",
    is_edible := λ x, x = "banana" ∨ x = "tomato",
    is_red := λ x, x = "tomato",
    p_types := ["sun", "ball", "tomato", "banana"]
  }

theorem max_balls_possible (cond : Conditions)
  (h1 : cond.yellow_objects = 15)
  (h2 : cond.round_objects = 18)
  (h3 : cond.edible_objects = 13)
  (h4 : ∀ x, x ∈ cond.all_objects → (cond.is_round x → ¬cond.is_yellow x → cond.is_edible x → ¬cond.is_red x))
  : ∃ n, n = 18 :=
by {
  sorry
}

end Problem

end max_balls_possible_l316_316022


namespace solve_system_of_equations_l316_316663

theorem solve_system_of_equations (x y : ℚ) 
  (h1 : 3 * x - 2 * y = 8) 
  (h2 : x + 3 * y = 9) : 
  x = 42 / 11 ∧ y = 19 / 11 :=
by {
  sorry
}

end solve_system_of_equations_l316_316663


namespace team_B_at_least_half_can_serve_l316_316122

-- Define the height limit condition
def height_limit (h : ℕ) : Prop := h ≤ 168

-- Define the team conditions
def team_A_avg_height : Prop := (160 + 169 + 169) / 3 = 166

def team_B_median_height (B : List ℕ) : Prop :=
  B.length % 2 = 1 ∧ B.perm ([167] ++ (B.eraseNth (B.length / 2))) ∧ B.nth (B.length / 2) = some 167

def team_C_tallest_height (C : List ℕ) : Prop :=
  ∀ (h : ℕ), h ∈ C → h ≤ 169

def team_D_mode_height (D : List ℕ) : Prop :=
  ∃ k, ∀ (h : ℕ), h ≠ 167 ∨ D.count 167 ≥ D.count h

-- Declare the main theorem to be proven
theorem team_B_at_least_half_can_serve (B : List ℕ) :
  (∀ h, h ∈ B → height_limit h) ↔ team_B_median_height B := sorry

end team_B_at_least_half_can_serve_l316_316122


namespace lateral_surface_area_cone_correct_l316_316551

-- Define the given parameters
def r_cyl : ℝ := 3 / 4
def h_cyl : ℝ := 3 / 2
noncomputable def r : ℝ := (3 / 4) * (Real.sqrt 2 - 1)

-- Calculate the slant height of the cone using Pythagorean theorem
noncomputable def slant_height_cone : ℝ :=
  Real.sqrt ((h_cyl + r)^2 + (2 * r_cyl)^2)

-- Define the lateral surface area of the cone
noncomputable def lateral_surface_area_cone : ℝ :=
  Real.pi * r_cyl * slant_height_cone

-- State the theorem to prove
theorem lateral_surface_area_cone_correct :
  lateral_surface_area_cone = (9 * Real.pi * (Real.sqrt (7 + 2 * Real.sqrt 21)) / 16) :=
  sorry

end lateral_surface_area_cone_correct_l316_316551


namespace quadratic_roots_l316_316635

-- Define the condition for the quadratic equation
def quadratic_eq (x m : ℝ) : Prop := x^2 - 4*x + m + 2 = 0

-- Define the discriminant condition
def discriminant_pos (m : ℝ) : Prop := (4^2 - 4 * (m + 2)) > 0

-- Define the condition range for m
def m_range (m : ℝ) : Prop := m < 2

-- Define the condition for m as a positive integer
def m_positive_integer (m : ℕ) : Prop := m = 1

-- The main theorem stating the problem
theorem quadratic_roots : 
  (∀ (m : ℝ), discriminant_pos m → m_range m) ∧ 
  (∀ m : ℕ, m_positive_integer m → (∃ x1 x2 : ℝ, quadratic_eq x1 m ∧ quadratic_eq x2 m ∧ x1 = 1 ∧ x2 = 3)) := 
by 
  sorry

end quadratic_roots_l316_316635


namespace product_of_elements_in_M_and_N_l316_316400

def M := {z : ℂ | z * complex.I = 1}
def N := {z : ℂ | z + complex.I = 1}

theorem product_of_elements_in_M_and_N :
  ∀ z1 (h1 : z1 ∈ M), ∀ z2 (h2 : z2 ∈ N), z1 * z2 = -1 - complex.I :=
by
  sorry

end product_of_elements_in_M_and_N_l316_316400


namespace circumscribed_sphere_position_l316_316795

noncomputable def orthocentric_tetrahedron (A1 A2 A3 A4 M O : Type) :=
  -- A tetrahedron where orthocenter M bisects altitudes from A1, A2, A3, A4 to faces.
sorry

theorem circumscribed_sphere_position
  {A1 A2 A3 A4 : Type} 
  {M O : Type}
  (h_orthocentric : orthocentric_tetrahedron A1 A2 A3 A4 M O)
  (h_condition_1 : ∀ (A : Type), A ∈ {A1, A2, A3, A4} → between_foot_midpoint M A)
  (h_condition_2 : ∃ (A : Type), A ∈ {A1, A2, A3, A4} ∧ M bisects_altitude A)
  (h_condition_3 : ∃ (A : Type), A ∈ {A1, A2, A3, A4} ∧ outside_foot_midpoint M A):
  -- Conclusion based on the conditions
  if h_condition_1 then inside_tetrahedron O
  else if h_condition_2 then on_face_tetrahedron O
  else if h_condition_3 then outside_tetrahedron O
  else false :=
begin
  sorry,
end

end circumscribed_sphere_position_l316_316795


namespace max_balls_count_l316_316029

-- Definitions
def Object := Type
def sun : Object := sorry
def ball : Object := sorry
def tomato : Object := sorry
def banana : Object := sorry

def is_yellow : Object → Prop := sorry
def is_round : Object → Prop := sorry
def is_edible : Object → Prop := sorry

axiom yellow_items_count : ∃ (Y : set Object), Y.card = 15 ∧ ∀ y ∈ Y, is_yellow y
axiom round_items_count : ∃ (R : set Object), R.card = 18 ∧ ∀ r ∈ R, is_round r
axiom edible_items_count : ∃ (E : set Object), E.card = 13 ∧ ∀ e ∈ E, is_edible e

-- Problem conditions
axiom tomato_is_round_and_red : ∀ t, t = tomato → is_round t ∧ ¬is_yellow t
axiom ball_is_round : ∀ b, b = ball → is_round b
axiom banana_is_yellow_and_not_round : ∀ b, b = banana → is_yellow b ∧ ¬is_round b

-- Target proposition
theorem max_balls_count : 
  ∀ (sun_count ball_count tomato_count banana_count : ℕ),
    is_round sun → ¬is_yellow sun →
    is_round ball → is_edible ball → ¬is_yellow ball → 
    is_yellow tomato → is_round tomato → is_edible tomato →
    is_yellow banana → ¬is_round banana → is_edible banana →
    sun_count + ball_count + tomato_count + banana_count = 46 →
    ball_count ≤ 18 := sorry

end max_balls_count_l316_316029


namespace condition_not_right_triangle_l316_316152

theorem condition_not_right_triangle 
  (AB BC AC : ℕ) (angleA angleB angleC : ℕ)
  (h_A : AB = 3 ∧ BC = 4 ∧ AC = 5)
  (h_B : AB / BC = 3 / 4 ∧ BC / AC = 4 / 5 ∧ AB + BC > AC ∧ AB + AC > BC ∧ BC + AC > AB)
  (h_C : angleA / angleB = 3 / 4 ∧ angleB / angleC = 4 / 5 ∧ angleA + angleB + angleC = 180)
  (h_D : angleA = 40 ∧ angleB = 50 ∧ angleA + angleB + angleC = 180) :
  angleA = 45 ∧ angleB = 60 ∧ angleC = 75 ∧ (¬ (angleA = 90 ∨ angleB = 90 ∨ angleC = 90)) :=
sorry

end condition_not_right_triangle_l316_316152


namespace segment_less_equal_ST_l316_316144

theorem segment_less_equal_ST {P : Type} [convex_polygon P] (S T A B C : P)
  (h1 : divides_perimeter S T P (1 / 2))
  (h2 : moves_along_boundary A B C P)
  : ∃ t : ℝ, segment_length_le ST P (segment_length_le A B t ∨ segment_length_le B C t ∨ segment_length_le C A t)
:= sorry

end segment_less_equal_ST_l316_316144


namespace factorize_problem1_factorize_problem2_l316_316606

-- Problem 1: Prove that 6p^3q - 10p^2 == 2p^2 * (3pq - 5)
theorem factorize_problem1 (p q : ℝ) : 
    6 * p^3 * q - 10 * p^2 = 2 * p^2 * (3 * p * q - 5) := 
by 
    sorry

-- Problem 2: Prove that a^4 - 8a^2 + 16 == (a-2)^2 * (a+2)^2
theorem factorize_problem2 (a : ℝ) : 
    a^4 - 8 * a^2 + 16 = (a - 2)^2 * (a + 2)^2 := 
by 
    sorry

end factorize_problem1_factorize_problem2_l316_316606


namespace probability_two_out_of_three_win_l316_316917

theorem probability_two_out_of_three_win :
  let p_A := 1 / 5
  let p_B := 3 / 8
  let p_C := 2 / 7 in
  let scenarios := [
    (p_A) * (p_B) * ((1 - p_C)),
    (p_A) * (p_C) * ((1 - p_B)),
    (p_B) * (p_C) * ((1 - p_A))
  ] in
  (scenarios.sum = 49 / 280) :=
  by
    sorry

end probability_two_out_of_three_win_l316_316917


namespace max_balls_drawn_l316_316039

-- Conditions:
variable (items : Type) 
variable (Petya : items → Prop)
variable (yellow round edible : items → Prop)
variable (sun ball tomato banana : items)

variable (Ht : ∀ x, tomato x → round x ∧ ¬yellow x) -- All tomatoes are round and red
variable (Hb : ∀ x, banana x → yellow x ∧ ¬round x) -- All bananas are yellow and not round
variable (Hba : ∀ x, ball x → round x) -- All balls are round

variable (yellow_count : ∑ x in items, yellow x = 15) -- Exactly 15 yellow items
variable (round_count : ∑ x in items, round x = 18) -- Exactly 18 round items
variable (edible_count : ∑ x in items, edible x = 13) -- Exactly 13 edible items

-- Proving the maximum number of balls
theorem max_balls_drawn : ∑ x in items, ball x ≤ 18 :=
by sorry

end max_balls_drawn_l316_316039


namespace nancy_initial_files_l316_316000

theorem nancy_initial_files (deleted_files : ℕ) (files_per_folder : ℕ) (num_folders : ℕ) :
  deleted_files = 31 → files_per_folder = 6 → num_folders = 2 → 
  initial_files (deleted_files + num_folders * files_per_folder) = 43 :=
by
  sorry

end nancy_initial_files_l316_316000


namespace mn_not_both_odd_l316_316764

theorem mn_not_both_odd (m n : ℕ) (h : (1 / (m : ℝ) + 1 / (n : ℝ) = 1 / 2020)) :
  ¬ (odd m ∧ odd n) :=
sorry

end mn_not_both_odd_l316_316764


namespace investment_ratio_l316_316569

-- Define the investments
def A_investment (x : ℝ) : ℝ := 3 * x
def B_investment (x : ℝ) : ℝ := x
def C_investment (y : ℝ) : ℝ := y

-- Define the total profit and B's share of the profit
def total_profit : ℝ := 4400
def B_share : ℝ := 800

-- Define the ratio condition B's share based on investments
def B_share_cond (x y : ℝ) : Prop := (B_investment x / (A_investment x + B_investment x + C_investment y)) * total_profit = B_share

-- Define what we need to prove
theorem investment_ratio (x y : ℝ) (h : B_share_cond x y) : x / y = 2 / 3 :=
by 
  sorry

end investment_ratio_l316_316569


namespace find_g_minus_6_l316_316847

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Conditions given in the problem
axiom cond1 : g 1 - 1 > 0
axiom cond2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom cond3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The proof we want to make (assertion)
theorem find_g_minus_6 : g (-6) = 723 := 
sorry

end find_g_minus_6_l316_316847


namespace distance_to_cut_pyramid_l316_316476

theorem distance_to_cut_pyramid (V A V1 : ℝ) (h1 : V > 0) (h2 : A > 0) :
  ∃ d : ℝ, d = (3 / A) * (V - (V^2 * (V - V1))^(1 / 3)) :=
by
  sorry

end distance_to_cut_pyramid_l316_316476


namespace matt_sales_l316_316515

noncomputable def total_sales (n : ℕ) : ℕ :=
  n + 1

theorem matt_sales (n : ℕ) (h1 : 1300 + n * 250 = 400 * (n + 1)): total_sales n = 7 :=
begin
  sorry
end

end matt_sales_l316_316515


namespace birdhouse_flight_distance_l316_316474

variable (car_distance : ℕ)
variable (lawn_chair_distance : ℕ)
variable (birdhouse_distance : ℕ)

def problem_condition1 := car_distance = 200
def problem_condition2 := lawn_chair_distance = 2 * car_distance
def problem_condition3 := birdhouse_distance = 3 * lawn_chair_distance

theorem birdhouse_flight_distance
  (h1 : car_distance = 200)
  (h2 : lawn_chair_distance = 2 * car_distance)
  (h3 : birdhouse_distance = 3 * lawn_chair_distance) :
  birdhouse_distance = 1200 := by
  sorry

end birdhouse_flight_distance_l316_316474


namespace pucks_cannot_return_after_25_hits_l316_316490

-- Define a type for Orientation, which can be either clockwise or counterclockwise
inductive Orientation
| clockwise
| counterclockwise

-- Define a type for pucks
inductive Puck
| A
| B
| C

-- Define a function that describes the change in orientation for a given hit count
def orientation_after_hits (initial : Orientation) (hits : Nat) : Orientation :=
  if hits % 2 = 0 then initial
  else match initial with
       | Orientation.clockwise       => Orientation.counterclockwise
       | Orientation.counterclockwise => Orientation.clockwise

-- Theorem: proving that pucks cannot return to their original positions after 25 hits
theorem pucks_cannot_return_after_25_hits (initial : Orientation) : 
  orientation_after_hits initial 25 ≠ initial :=
by
  sorry

end pucks_cannot_return_after_25_hits_l316_316490


namespace darks_washing_time_l316_316959

theorem darks_washing_time (x : ℕ) :
  (72 + x + 45) + (50 + 65 + 54) = 344 → x = 58 :=
by
  sorry

end darks_washing_time_l316_316959


namespace g_minus_6_eq_neg_20_l316_316838

noncomputable def g : ℤ → ℤ := sorry

axiom condition1 : g 1 - 1 > 0
axiom condition2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom condition3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

theorem g_minus_6_eq_neg_20 : g (-6) = -20 := sorry

end g_minus_6_eq_neg_20_l316_316838


namespace two_hundredth_term_of_non_square_sequence_l316_316147

def is_square_free (n : ℕ) : Prop := 
  ∀ k : ℕ, k > 0 → k * k ≠ n

def nth_non_square (n : ℕ) : ℕ :=
  (n : ℕ) + (Nat.floor (Real.sqrt n))

theorem two_hundredth_term_of_non_square_sequence : nth_non_square 200 = 214 :=
by
  sorry

end two_hundredth_term_of_non_square_sequence_l316_316147


namespace median_mode_difference_l316_316081

def monthly_incomes : list ℕ := [45000, 18000, 10000, 5500, 5500, 5500, 5000, 5000, 5000, 5000, 5000, 5000, 3400, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 2500]

def mode (lst : list ℕ) : ℕ := 3000 -- Based on the problem data

def median (lst : list ℕ) : ℕ := 3400 -- Based on the problem data

theorem median_mode_difference : median monthly_incomes - mode monthly_incomes = 400 :=
by
  sorry

end median_mode_difference_l316_316081


namespace arithmetic_sequence_difference_l316_316281

def a₁ : ℕ := 3

def S (n : ℕ) : ℕ :=
  2 * a n + (3 / 2 : ℚ) * ((-1) ^ n - 1)

theorem arithmetic_sequence_difference (p q : ℕ) (hpq : 1 < p ∧ p < q) (h₁ : a₁ = 3) 
  (h₂ : ∀ n, S n = 2 * a n + (3 / 2 : ℚ) * ((-1) ^ n - 1)) 
  (harithmetic : 2 * a p = a q + a₁) : q - p = 1 :=
sorry

end arithmetic_sequence_difference_l316_316281


namespace reflection_across_x_axis_l316_316155

theorem reflection_across_x_axis (x y : ℝ) : (x, -y) = (-2, 3) ↔ (x, y) = (-2, -3) :=
by sorry

end reflection_across_x_axis_l316_316155


namespace jerry_claim_percentage_l316_316375

theorem jerry_claim_percentage
  (salary_years : ℕ)
  (annual_salary : ℕ)
  (medical_bills : ℕ)
  (punitive_multiplier : ℕ)
  (received_amount : ℕ)
  (total_claim : ℕ)
  (percentage_claim : ℕ) :
  salary_years = 30 →
  annual_salary = 50000 →
  medical_bills = 200000 →
  punitive_multiplier = 3 →
  received_amount = 5440000 →
  total_claim = (annual_salary * salary_years) + medical_bills + (punitive_multiplier * ((annual_salary * salary_years) + medical_bills)) →
  percentage_claim = (received_amount * 100) / total_claim →
  percentage_claim = 80 :=
by
  sorry

end jerry_claim_percentage_l316_316375


namespace greatest_common_factor_36_45_l316_316907

theorem greatest_common_factor_36_45 : 
  ∃ g, g = (gcd 36 45) ∧ g = 9 :=
by {
  sorry
}

end greatest_common_factor_36_45_l316_316907


namespace a_and_b_together_work_days_l316_316511

-- Definitions for the conditions:
def a_work_rate : ℚ := 1 / 9
def b_work_rate : ℚ := 1 / 18

-- The theorem statement:
theorem a_and_b_together_work_days : (a_work_rate + b_work_rate)⁻¹ = 6 := by
  sorry

end a_and_b_together_work_days_l316_316511


namespace least_positive_difference_l316_316811

open Nat

def geometric_seq (a r : ℕ) (n : ℕ) : ℕ := a * r ^ n
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + d * n

-- Definition of sequence C
def seq_C (n : ℕ) : ℕ := geometric_seq 3 3 n

-- Definition of sequence D
def seq_D (n : ℕ) : ℕ := arithmetic_seq 15 15 n

-- The problem statement
theorem least_positive_difference : ∃ m n, seq_C m ≤ 450 ∧ seq_D n ≤ 450 ∧ (∀ k l, seq_C k ≤ 450 → seq_D l ≤ 450 → (abs (seq_C m - seq_D n) ≤ abs (seq_C k - seq_D l))) ∧ abs (seq_C m - seq_D n) = 3 :=
by
  sorry

end least_positive_difference_l316_316811


namespace at_least_half_team_B_can_serve_on_submarine_l316_316129

theorem at_least_half_team_B_can_serve_on_submarine
    (max_height : ℕ)
    (team_A_avg_height : ℕ)
    (team_B_median_height : ℕ)
    (team_C_tallest_height : ℕ)
    (team_D_mode_height : ℕ)
    (h1 : max_height = 168)
    (h2 : team_A_avg_height = 166)
    (h3 : team_B_median_height = 167)
    (h4 : team_C_tallest_height = 169)
    (h5 : team_D_mode_height = 167) :
  ∀ (height : ℕ), height ≤ max_height → ∃ (b_sailors : ℕ → Prop) (H : ∃ n, b_sailors n),
  (∃ (n_half : ℕ), (∀ h ≤ team_B_median_height, b_sailors h) ∧ (2 * n_half ≤ n)) :=
sorry

end at_least_half_team_B_can_serve_on_submarine_l316_316129


namespace probability_of_all_red_is_correct_l316_316934

noncomputable def probability_of_all_red_drawn : ℚ :=
  let total_ways := (Nat.choose 10 5)   -- Total ways to choose 5 balls from 10
  let red_ways := (Nat.choose 5 5)      -- Ways to choose all 5 red balls
  red_ways / total_ways

theorem probability_of_all_red_is_correct :
  probability_of_all_red_drawn = 1 / 252 := by
  sorry

end probability_of_all_red_is_correct_l316_316934


namespace find_smallest_n_l316_316598

theorem find_smallest_n :
  ∃ (n a b : ℕ), n = 1843 ∧ n = a^3 + b^3 ∧ ∀ p : ℕ, p.prime → p ∣ n → p > 18 :=
by
  sorry

end find_smallest_n_l316_316598


namespace sum_of_three_integers_l316_316485

def three_positive_integers (x y z : ℕ) : Prop :=
  x + y = 2003 ∧ y - z = 1000

theorem sum_of_three_integers (x y z : ℕ) (h1 : x + y = 2003) (h2 : y - z = 1000) (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) : 
  x + y + z = 2004 := 
by 
  sorry

end sum_of_three_integers_l316_316485


namespace log_prime_factor_inequality_l316_316796

open Real

noncomputable def num_prime_factors (n : ℕ) : ℕ := sorry 

theorem log_prime_factor_inequality (n : ℕ) (h : 0 < n) : 
  log n ≥ num_prime_factors n * log 2 := 
sorry

end log_prime_factor_inequality_l316_316796


namespace projection_of_point_onto_xOy_plane_l316_316364

def point := (ℝ × ℝ × ℝ)

def projection_onto_xOy_plane (P : point) : point :=
  let (x, y, z) := P
  (x, y, 0)

theorem projection_of_point_onto_xOy_plane : 
  projection_onto_xOy_plane (2, 3, 4) = (2, 3, 0) :=
by
  -- proof steps would go here
  sorry

end projection_of_point_onto_xOy_plane_l316_316364


namespace sum_of_digits_Joey_age_twice_Max_next_l316_316737

noncomputable def Joey_is_two_years_older (C : ℕ) : ℕ := C + 2

noncomputable def Max_age_today := 2

noncomputable def Eight_multiples_of_Max (C : ℕ) := 
  ∃ n : ℕ, C = 24 + n

noncomputable def Next_Joey_age_twice_Max (C J M n : ℕ): Prop := J + n = 2 * (M + n)

theorem sum_of_digits_Joey_age_twice_Max_next (C J M n : ℕ) 
  (h1: J = Joey_is_two_years_older C)
  (h2: M = Max_age_today)
  (h3: Eight_multiples_of_Max C)
  (h4: Next_Joey_age_twice_Max C J M n) 
  : ∃ s, s = 7 :=
sorry

end sum_of_digits_Joey_age_twice_Max_next_l316_316737


namespace maximum_value_of_y_l316_316893

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.sin (x - π / 3)
noncomputable def y (x : ℝ) : ℝ := f x + g x

theorem maximum_value_of_y : ∃ x : ℝ, y x = sqrt 3 := by
  sorry

end maximum_value_of_y_l316_316893


namespace knights_probability_l316_316105

theorem knights_probability :
  let knights : Nat := 30
  let chosen : Nat := 4
  let probability (n k : Nat) := 1 - (((n - k + 1) * (n - k - 1) * (n - k - 3) * (n - k - 5)) / 
                                      ((n - 0) * (n - 1) * (n - 2) * (n - 3)))
  probability knights chosen = (389 / 437) := sorry

end knights_probability_l316_316105


namespace sum_powers_ω_equals_ω_l316_316991

-- Defining the complex exponential ω = e^(2 * π * i / 17)
def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 17)

-- Statement of the theorem
theorem sum_powers_ω_equals_ω : 
  (Finset.range 16).sum (λ k, Complex.exp (2 * Real.pi * Complex.I * (k + 1) / 17)) = ω :=
sorry

end sum_powers_ω_equals_ω_l316_316991


namespace hexagram_shell_arrangements_l316_316376

/--
John places twelve different sea shells at the vertices of a regular six-pointed star (hexagram).
How many distinct ways can he place the shells, considering arrangements that differ by rotations or reflections as equivalent?
-/
theorem hexagram_shell_arrangements :
  (Nat.factorial 12) / 12 = 39916800 :=
by
  sorry

end hexagram_shell_arrangements_l316_316376


namespace boxes_of_nerds_l316_316983

def totalCandies (kitKatBars hersheyKisses lollipops babyRuths reeseCups nerds : Nat) : Nat := 
  kitKatBars + hersheyKisses + lollipops + babyRuths + reeseCups + nerds

def adjustForGivenLollipops (total lollipopsGiven : Nat) : Nat :=
  total - lollipopsGiven

theorem boxes_of_nerds :
  ∀ (kitKatBars hersheyKisses lollipops babyRuths reeseCups lollipopsGiven totalAfterGiving nerds : Nat),
  kitKatBars = 5 →
  hersheyKisses = 3 * kitKatBars →
  lollipops = 11 →
  babyRuths = 10 →
  reeseCups = babyRuths / 2 →
  lollipopsGiven = 5 →
  totalAfterGiving = 49 →
  totalCandies kitKatBars hersheyKisses lollipops babyRuths reeseCups 0 - lollipopsGiven + nerds = totalAfterGiving →
  nerds = 8 :=
by
  intros
  sorry

end boxes_of_nerds_l316_316983


namespace correct_operation_is_C_l316_316153

theorem correct_operation_is_C :
    (∀ x : ℝ, x = \sqrt{(-13)^2} → x ≠ -13) ∧ 
    (∀ y : ℝ, y = 3 * (\sqrt{2}) - 2 * (\sqrt{2}) → y ≠ 1) ∧ 
    (∀ z : ℝ, z = \sqrt{10} / \sqrt{5} → z = \sqrt{2}) ∧ 
    (∀ w : ℝ, w = (2 * (\sqrt{5}))^2 → w ≠ 10) :=
by
  sorry

end correct_operation_is_C_l316_316153


namespace product_calculation_l316_316986

theorem product_calculation :
  (∏ k in Finset.range 5 \+\ Finset.range 2, (k^4 - 1) / (k^4 + 1)) = (4 : ℚ) / 1115 := 
by
  sorry

end product_calculation_l316_316986


namespace probability_at_least_one_white_ball_l316_316437

noncomputable def total_combinations : ℕ := (Nat.choose 5 3)
noncomputable def no_white_combinations : ℕ := (Nat.choose 3 3)
noncomputable def prob_no_white_balls : ℚ := no_white_combinations / total_combinations
noncomputable def prob_at_least_one_white_ball : ℚ := 1 - prob_no_white_balls

theorem probability_at_least_one_white_ball :
  prob_at_least_one_white_ball = 9 / 10 :=
by
  have h : total_combinations = 10 := by sorry
  have h1 : no_white_combinations = 1 := by sorry
  have h2 : prob_no_white_balls = 1 / 10 := by sorry
  have h3 : prob_at_least_one_white_ball = 1 - prob_no_white_balls := by sorry
  norm_num [prob_no_white_balls, prob_at_least_one_white_ball, h, h1, h2, h3]

end probability_at_least_one_white_ball_l316_316437


namespace constant_s_l316_316891

def parabola (x : ℝ) : ℝ := 2 * x ^ 2

def distance_squared (a b : ℝ × ℝ) : ℝ :=
  (a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2

theorem constant_s (x1 x2 : ℝ) (c : ℝ) (h1 : x1 + x2 = 1 / 2) (h2 : x1 * x2 = -1 / 2) :
  let P := (x1, parabola x1),
      Q := (x2, parabola x2),
      D := (1, c),
      PD2 := distance_squared P D,
      QD2 := distance_squared Q D,
      s := 1 / PD2 + 1 / QD2
  in s = 8 := sorry

end constant_s_l316_316891


namespace concentrate_to_water_ratio_l316_316207

theorem concentrate_to_water_ratio :
  ∀ (c w : ℕ), (∀ c, w = 3 * c) → (35 * 3 = 105) → (1 / 3 = (1 : ℝ) / (3 : ℝ)) :=
by
  intros c w h1 h2
  sorry

end concentrate_to_water_ratio_l316_316207


namespace squares_with_center_25_60_l316_316011

theorem squares_with_center_25_60 :
  let center_x := 25
  let center_y := 60
  let non_neg_int_coords (x : ℤ) (y : ℤ) := x ≥ 0 ∧ y ≥ 0
  let is_center (x : ℤ) (y : ℤ) := x = center_x ∧ y = center_y
  let num_squares := 650
  ∃ n : ℤ, (n = num_squares) ∧ ∀ (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℤ), 
    non_neg_int_coords x₁ y₁ ∧ non_neg_int_coords x₂ y₂ ∧ 
    non_neg_int_coords x₃ y₃ ∧ non_neg_int_coords x₄ y₄ ∧ 
    is_center ((x₁ + x₂ + x₃ + x₄) / 4) ((y₁ + y₂ + y₃ + y₄) / 4) → 
    ∃ (k : ℤ), n = 650 :=
sorry

end squares_with_center_25_60_l316_316011


namespace pencils_left_l316_316979

def total_pencils (boxes : ℕ) (pencils_per_box : ℕ) : ℕ :=
  boxes * pencils_per_box

def remaining_pencils (initial_pencils : ℕ) (pencils_given : ℕ) : ℕ :=
  initial_pencils - pencils_given

theorem pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (pencils_given : ℕ)
  (h_boxes : boxes = 2) (h_pencils_per_box : pencils_per_box = 14) (h_pencils_given : pencils_given = 6) :
  remaining_pencils (total_pencils boxes pencils_per_box) pencils_given = 22 :=
by
  rw [h_boxes, h_pencils_per_box, h_pencils_given]
  norm_num
  sorry

end pencils_left_l316_316979


namespace find_x_l316_316928

theorem find_x (x : ℝ) (h : 45 * x = 0.60 * 900) : x = 12 :=
by
  sorry

end find_x_l316_316928


namespace expected_value_max_of_four_element_subset_l316_316382

open Nat

noncomputable def expected_value_max_element (S : finset ℕ) : ℚ := 
  ∑ i in S.filter (λ x, x = S.max ℕ), i.to_rat / S.card.to_rat

theorem expected_value_max_of_four_element_subset  :
  let S := (finset.range 9).powerset.filter (λ s, s.card = 4) in
  let expected_value := (S.sum (λ s, ↑(s.max ℕ) * s.card.to_rat)).sum / S.card.to_rat in
  m, n : ℕ,
  m = 36 ∧ n = 5 ∧ nat.coprime m n ∧ expected_value = (m / n : ℚ) → 
  m + n = 41 := by sorry

end expected_value_max_of_four_element_subset_l316_316382


namespace james_carrot_sticks_l316_316734

def carrots_eaten_after_dinner (total_carrots : ℕ) (carrots_before_dinner : ℕ) : ℕ :=
  total_carrots - carrots_before_dinner

theorem james_carrot_sticks : carrots_eaten_after_dinner 37 22 = 15 := by
  sorry

end james_carrot_sticks_l316_316734


namespace lineup_count_l316_316220

theorem lineup_count (P: Finset ℕ) (hP_card : P.card = 15) (Bob Yogi Zoey : ℕ) 
(hBob : Bob ∈ P) (hYogi : Yogi ∈ P) (hZoey : Zoey ∈ P) 
(hB : ∀ l, Yogi ∈ l -> Bob ∉ l ∧ Zoey ∉ l)
(hY : ∀ l, Bob ∈ l -> Yogi ∉ l ∧ Zoey ∉ l)
: (P.erase Bob).choose 4 ∪ (P.erase Yogi).choose 4 ∪ (P.erase Zoey).choose 4 ∪ (P.erase Bob.erase Yogi.erase Zoey).choose 5).card = 2277 :=
sorry

end lineup_count_l316_316220


namespace total_boxes_correct_l316_316537

noncomputable def initial_boxes : ℕ := 400
noncomputable def cost_per_box : ℕ := 80 + 165
noncomputable def initial_spent : ℕ := initial_boxes * cost_per_box
noncomputable def donor_amount : ℕ := 4 * initial_spent
noncomputable def additional_boxes : ℕ := donor_amount / cost_per_box
noncomputable def total_boxes : ℕ := initial_boxes + additional_boxes

theorem total_boxes_correct : total_boxes = 2000 := by
  sorry

end total_boxes_correct_l316_316537


namespace area_of_trapezium_l316_316065

variables (x : ℝ) (h : x > 0)

def shorter_base := 2 * x
def altitude := 2 * x
def longer_base := 6 * x

theorem area_of_trapezium (hx : x > 0) :
  (1 / 2) * (shorter_base x + longer_base x) * altitude x = 8 * x^2 := 
sorry

end area_of_trapezium_l316_316065


namespace af_over_cd_is_025_l316_316655

theorem af_over_cd_is_025
  (a b c d e f X : ℝ)
  (h1 : a * b * c = X)
  (h2 : b * c * d = X)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 0.25 := by
  sorry

end af_over_cd_is_025_l316_316655


namespace value_of_a4_l316_316158

theorem value_of_a4 (a1 a2 a3 a4 a5 : ℕ) 
  (h1: 695 = a1 + a2 * 2! + a3 * 3! + a4 * 4! + a5 * 5!)
  (h2: 0 ≤ a1 ∧ a1 ≤ 1)
  (h3: 0 ≤ a2 ∧ a2 ≤ 2)
  (h4: 0 ≤ a3 ∧ a3 ≤ 3)
  (h5: 0 ≤ a4 ∧ a4 ≤ 4)
  (h6: 0 ≤ a5 ∧ a5 ≤ 5) : 
  a4 = 3 :=
sorry

end value_of_a4_l316_316158


namespace tara_questions_wrong_l316_316820

variable {ℕ : Type} (t u w : ℕ)

theorem tara_questions_wrong (v : ℕ) (hv : v = 3) 
  (h1 : t + u = v + w) (h2 : t + w = u + v + 6) : t = 9 :=
by 
  sorry

end tara_questions_wrong_l316_316820


namespace find_length_EX_l316_316756

-- Given definitions and assumptions
variables {A B C H D E F X : Point}
variables (h_dist_AH_CircA : dist A H = dist A F) (h_dist_BH_CircB : dist B H = dist B E)
variables (h_orthocenter_AHC : orthocenter A B C H) (h_triangle_ABC : triangle A B C)
variables (h_FE_pairwise_foot : foot F A C B ∧ foot E B A C) (h_line_DF : line D F)
variables (h_circle_intersect_DF_circA : inter DF (circumcircle {A, H, F}) X)

-- To prove
theorem find_length_EX : dist E X = 190 / 49 :=
sorry

end find_length_EX_l316_316756


namespace ordering_of_a_b_c_l316_316651

noncomputable def a := 2^(-1/3)
noncomputable def b := Real.log 1 / 3 / Real.log 2
noncomputable def c := Real.log π / Real.log 3

theorem ordering_of_a_b_c : c > a ∧ a > b := 
by
  sorry

end ordering_of_a_b_c_l316_316651


namespace intersection_A_B_l316_316751

def A : Set ℝ := {x | abs x < 2}
def B : Set ℕ := {0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} :=
  sorry

end intersection_A_B_l316_316751


namespace age_ratios_l316_316922

variable (A B : ℕ)

-- Given conditions
theorem age_ratios :
  (A / B = 2 / 1) → (A - 4 = B + 4) → ((A + 4) / (B - 4) = 5 / 1) :=
by
  intro h1 h2
  sorry

end age_ratios_l316_316922


namespace exists_diagonals_angle_le_three_deg_l316_316813

theorem exists_diagonals_angle_le_three_deg (P : Polygon) (hP : P.sides = 12) (hConvex : P.convex) : 
  ∃ d1 d2 : Diagonal, angle_between d1 d2 ≤ 3 :=
sorry

end exists_diagonals_angle_le_three_deg_l316_316813


namespace geometric_sequence_frac_l316_316362

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (h_geometric : ∃ q > 0, ∀ n, a (n+1) = a n * q)
variable (h_decreasing : ∀ n, a (n+1) < a n)
variable (h1 : a 2 * a 8 = 6)
variable (h2 : a 4 + a 6 = 5)

theorem geometric_sequence_frac (h_geometric : ∃ q > 0, ∀ n, a (n+1) = a n * q)
                                (h_decreasing : ∀ n, a (n+1) < a n)
                                (h1 : a 2 * a 8 = 6)
                                (h2 : a 4 + a 6 = 5) :
                                a 3 / a 7 = 9 / 4 :=
by sorry

end geometric_sequence_frac_l316_316362


namespace proof_l_squared_l316_316181

noncomputable def longest_line_segment (diameter : ℝ) (sectors : ℕ) : ℝ :=
  let R := diameter / 2
  let theta := (2 * Real.pi) / sectors
  2 * R * (Real.sin (theta / 2))

theorem proof_l_squared :
  let diameter := 18
  let sectors := 4
  let l := longest_line_segment diameter sectors
  l^2 = 162 := by
  let diameter := 18
  let sectors := 4
  let l := longest_line_segment diameter sectors
  have h : l^2 = 162 := sorry
  exact h

end proof_l_squared_l316_316181


namespace team_B_at_least_half_can_serve_l316_316121

-- Define the height limit condition
def height_limit (h : ℕ) : Prop := h ≤ 168

-- Define the team conditions
def team_A_avg_height : Prop := (160 + 169 + 169) / 3 = 166

def team_B_median_height (B : List ℕ) : Prop :=
  B.length % 2 = 1 ∧ B.perm ([167] ++ (B.eraseNth (B.length / 2))) ∧ B.nth (B.length / 2) = some 167

def team_C_tallest_height (C : List ℕ) : Prop :=
  ∀ (h : ℕ), h ∈ C → h ≤ 169

def team_D_mode_height (D : List ℕ) : Prop :=
  ∃ k, ∀ (h : ℕ), h ≠ 167 ∨ D.count 167 ≥ D.count h

-- Declare the main theorem to be proven
theorem team_B_at_least_half_can_serve (B : List ℕ) :
  (∀ h, h ∈ B → height_limit h) ↔ team_B_median_height B := sorry

end team_B_at_least_half_can_serve_l316_316121


namespace pascal_family_min_children_l316_316720

-- We define the conditions b >= 3 and g >= 2
def b_condition (b : ℕ) : Prop := b >= 3
def g_condition (g : ℕ) : Prop := g >= 2

-- We state that the smallest number of children given these conditions is 5
theorem pascal_family_min_children (b g : ℕ) (hb : b_condition b) (hg : g_condition g) : b + g = 5 :=
sorry

end pascal_family_min_children_l316_316720


namespace mojmir_can_form_right_triangles_l316_316775

theorem mojmir_can_form_right_triangles
  (T : Type)
  (triangle : T)
  (is_30_60_90_triangle : triangle.angles = {30, 60, 90}):
∃ (formed_triangles : list T), (∀ t ∈ formed_triangles, t.angles = {30, 60, 90}) ∧
  (formed_triangles.length = 1 ∨ formed_triangles.length = 3 ∨ formed_triangles.length = 4 ∨ formed_triangles.length = 9) ∧
  (∀ t ∈ formed_triangles, t.is_right_triangle) := sorry

end mojmir_can_form_right_triangles_l316_316775


namespace intervals_of_monotonicity_range_of_k_l316_316303

noncomputable def f (x : ℝ) (k : ℝ) := Real.log (x - 1) - k * (x - 1) + 1

theorem intervals_of_monotonicity (k : ℝ) :
  (∀ x, 1 < x → k ≤ 0 → 0 < deriv (λ x, f x k) x) ∧
  (∀ x, 1 < x ∧ 1 < x ∧ x < (k + 1) / k → 0 < deriv (λ x, f x k) x) ∧
  (∀ x : ℝ, (k > 0 ∧ (x > (k + 1) / k) → deriv (λ x, f x k) x < 0)) :=
sorry

theorem range_of_k (k : ℝ) :
  (∀ x, 1 < x → f x k ≤ 0 → k ≥ 1) :=
sorry

end intervals_of_monotonicity_range_of_k_l316_316303


namespace probability_odd_four_digit_number_l316_316826

theorem probability_odd_four_digit_number :
  let digits := {2, 3, 5, 7}
  let is_odd (n : ℕ) := n % 2 = 1
  let four_digit_numbers := finset.permutations digits
  let odd_numbers := four_digit_numbers.filter (λ n, is_odd (n % 10))
  let probability_odd := (odd_numbers.card : ℚ) / (four_digit_numbers.card : ℚ)
  in probability_odd = 3 / 4 :=
by
  sorry

end probability_odd_four_digit_number_l316_316826


namespace price_restoration_percentage_l316_316463

noncomputable def original_price := 100 -- Assumption for simplicity

def first_reduction (P : ℝ) := P * 0.65
def second_reduction (P : ℝ) := P * 0.9
def third_reduction (P : ℝ) := P * 0.95

def final_price := 
  let P₁ := first_reduction original_price
  let P₂ := second_reduction P₁
  let P₃ := third_reduction P₂
  P₃

theorem price_restoration_percentage :
  let percentage_increase := ((original_price - final_price) / final_price) * 100
  percentage_increase ≈ 79.9 :=
by
  sorry

end price_restoration_percentage_l316_316463


namespace rectangle_tileable_iff_divisible_l316_316418

def divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

def tileable_with_0b_tiles (m n b : ℕ) : Prop :=
  ∃ t : ℕ, t * (2 * b) = m * n  -- This comes from the total area divided by the area of one tile

theorem rectangle_tileable_iff_divisible (m n b : ℕ) :
  tileable_with_0b_tiles m n b ↔ divisible_by (2 * b) m ∨ divisible_by (2 * b) n := 
sorry

end rectangle_tileable_iff_divisible_l316_316418


namespace jamie_minimum_4th_quarter_score_l316_316892

-- Define the conditions for Jamie's scores and the average requirement
def qualifying_score := 85
def first_quarter_score := 80
def second_quarter_score := 85
def third_quarter_score := 78

-- The function to determine the required score in the 4th quarter
def minimum_score_for_quarter (N : ℕ) := first_quarter_score + second_quarter_score + third_quarter_score + N ≥ 4 * qualifying_score

-- The main statement to be proved
theorem jamie_minimum_4th_quarter_score (N : ℕ) : minimum_score_for_quarter N ↔ N ≥ 97 :=
by
  sorry

end jamie_minimum_4th_quarter_score_l316_316892


namespace lobster_distribution_l316_316086

theorem lobster_distribution :
  let HarborA := 50
  let HarborB := 70.5
  let HarborC := (2 / 3) * HarborB
  let HarborD := HarborA - 0.15 * HarborA
  let Sum := HarborA + HarborB + HarborC + HarborD
  let HooperBay := 3 * Sum
  let Total := HooperBay + Sum
  Total = 840 := by
  sorry

end lobster_distribution_l316_316086


namespace plane_vectors_angle_l316_316297

open Real
open_locale big_operators

variables (a b : ℝ × ℝ)
variables (t : ℝ)

noncomputable def vec_length (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def vec_dot (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem plane_vectors_angle :
  (vec_length a = 1) →
  (vec_length b = 2) →
  (∀ t : ℝ, vec_length (b + t • a) ≥ vec_length (b - a)) →
  (real.angle (2 • a - b) b = (2 * real.pi) / 3) :=
begin
  intros h_a_len h_b_len h_condition,
  -- skipping proof steps
  sorry,
end

end plane_vectors_angle_l316_316297


namespace Alissa_presents_equal_9_l316_316234

def Ethan_presents : ℝ := 31.0
def difference : ℝ := 22.0
def Alissa_presents := Ethan_presents - difference

theorem Alissa_presents_equal_9 : Alissa_presents = 9.0 := 
by sorry

end Alissa_presents_equal_9_l316_316234


namespace probability_two_girls_from_five_ticket_holders_l316_316575

theorem probability_two_girls_from_five_ticket_holders (total_students : ℕ) (total_girls : ℕ) (tickets_drawn : ℕ) 
  (required_girls : ℕ) (rounded_probability_answer : ℚ) (htotal_students : total_students = 25) 
  (htotal_girls : total_girls = 10) (htickets_drawn : tickets_drawn = 5)
  (hrequired_girls : required_girls = 2) (hrounded_probability_answer : rounded_probability_answer = 0.385) : 
  ∃ p : ℚ, p = (Nat.choose 15 3 * Nat.choose 10 2) / Nat.choose 25 5 ∧ p ≈ rounded_probability_answer := 
by
  sorry

end probability_two_girls_from_five_ticket_holders_l316_316575


namespace g_minus_6_eq_neg_20_l316_316842

noncomputable def g : ℤ → ℤ := sorry

axiom condition1 : g 1 - 1 > 0
axiom condition2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom condition3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

theorem g_minus_6_eq_neg_20 : g (-6) = -20 := sorry

end g_minus_6_eq_neg_20_l316_316842


namespace alloy_problem_l316_316578

theorem alloy_problem (x y : ℝ) 
  (h1 : x + y = 1000) 
  (h2 : 0.25 * x + 0.50 * y = 450) 
  (hx : 0 ≤ x) 
  (hy : 0 ≤ y) :
  x = 200 ∧ y = 800 := 
sorry

end alloy_problem_l316_316578


namespace nathaniel_wins_probability_l316_316005

def fair_six_sided_die : Type := {n : ℕ // n ≥ 1 ∧ n ≤ 6}

def probability_nathaniel_wins : ℚ :=
  have fair_die : fair_six_sided_die := sorry,
  have nathaniel_first : Prop := sorry,
  have win_condition (sum : ℕ) : Prop := sum % 7 = 0,

  if nathaniel_first ∧ ∀ sum. win_condition sum
  then 5 / 11
  else 0

theorem nathaniel_wins_probability :
  probability_nathaniel_wins = 5 / 11 :=
sorry

end nathaniel_wins_probability_l316_316005


namespace total_profit_l316_316161

-- We start by defining the necessary parameters in Lean.
variable (P_A P_B P_C P Share_A Share_B Share_C : ℝ)

-- Given conditions:
def conditions : Prop :=
  P_A = 5000 ∧ P_B = 8000 ∧ P_C = 9000 ∧ Share_C = 36000

-- Prove total profit P.
theorem total_profit (h : conditions P_A P_B P_C P Share_A Share_B Share_C) : P = 97777.78 :=
by
  rcases h with ⟨hP_A, hP_B, hP_C, hShare_C⟩
  -- Definitions of investments and their ratios
  let ratio_A := P_A / 1000
  let ratio_B := P_B / 1000
  let ratio_C := P_C / 1000
  -- Total parts in the ratio
  let total_parts := ratio_A + ratio_B + ratio_C
  -- Given C's share as per condition
  let C_ratio_parts := ratio_C
  -- Find total profit P
  have : 9 / P = ratio_C / total_parts * 36000 / P :=
    sorry -- Proof is omitted, but logically ensuring we set the proper equality and proportions
  exact sorry

end total_profit_l316_316161


namespace average_age_increase_l316_316442

theorem average_age_increase (n : ℕ) (m : ℕ) (a b : ℝ) (h1 : n = 19) (h2 : m = 20) (h3 : a = 20) (h4 : b = 40) :
  ((n * a + b) / (n + 1)) - a = 1 :=
by
  -- Proof omitted
  sorry

end average_age_increase_l316_316442


namespace qed_product_l316_316697

def Q : Complex := 7 + 3 * Complex.i
def E : Complex := 2 * Complex.i
def D : Complex := 7 - 3 * Complex.i

theorem qed_product : Q * E * D = 116 * Complex.i := by
  sorry

end qed_product_l316_316697


namespace quotient_remainder_difference_l316_316948

theorem quotient_remainder_difference (N Q P R k : ℕ) (h1 : N = 75) (h2 : N = 5 * Q) (h3 : N = 34 * P + R) (h4 : Q = R + k) (h5 : k > 0) :
  Q - R = 8 :=
sorry

end quotient_remainder_difference_l316_316948


namespace cost_per_pack_is_correct_l316_316930

def total_amount_spent : ℝ := 120
def num_packs_bought : ℕ := 6
def expected_cost_per_pack : ℝ := 20

theorem cost_per_pack_is_correct :
  total_amount_spent / num_packs_bought = expected_cost_per_pack :=
  by 
    -- here would be the proof
    sorry

end cost_per_pack_is_correct_l316_316930


namespace find_g_neg_6_l316_316858

noncomputable def g : ℤ → ℤ := sorry

-- Conditions from the problem
axiom g_condition_1 : g 1 - 1 > 0
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The statement we need to prove
theorem find_g_neg_6 : g (-6) = 723 :=
by { sorry }

end find_g_neg_6_l316_316858


namespace g_neg_six_eq_neg_twenty_l316_316852

theorem g_neg_six_eq_neg_twenty (g : ℤ → ℤ)
    (h1 : g 1 - 1 > 0)
    (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
    (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 := 
sorry

end g_neg_six_eq_neg_twenty_l316_316852


namespace acute_triangle_solutions_l316_316338

theorem acute_triangle_solutions :
  ∀ (A B C : ℝ) (a b c : ℝ),
  (0 < A ∧ A < π / 2) ∧ (0 < B ∧ B < π / 2) ∧ (0 < C ∧ C < π / 2) ∧
  A + B + C = π ∧ c = 2 ∧ A = π / 3 →
  a * sin C = sqrt 3 ∧ (1 + sqrt 3 < a + b ∧ a + b < 4 + 2 * sqrt 3)
:=
  by
    intros A B C a b c h
    sorry

end acute_triangle_solutions_l316_316338


namespace probability_correct_l316_316776

noncomputable def probability_more_sons_than_daughters_or_more_daughters_than_sons : ℚ :=
  95 / 128

def gender_distribution : finset (fin 2) := finset.univ

def num_children : ℕ := 8

def num_twin_pairs : ℚ := 0.1

def num_combinations_without_twins := 2 ^ num_children

def ways_to_have_equal_genders_without_twins := nat.choose num_children (num_children / 2)

def complementary_prob_without_twins := num_combinations_without_twins - ways_to_have_equal_genders_without_twins

def effective_num_children_with_twins := num_children - 2

def num_combinations_with_twins := 2 ^ effective_num_children_with_twins

def ways_to_have_equal_genders_with_twins := nat.choose effective_num_children_with_twins (effective_num_children_with_twins / 2)

def complementary_prob_with_twins := num_combinations_with_twins - ways_to_have_equal_genders_with_twins

def scale_comp_with_twins := num_twin_pairs * complementary_prob_with_twins

def final_complementary_prob := complementary_prob_without_twins + scale_comp_with_twins

def final_prob := final_complementary_prob / num_combinations_without_twins

theorem probability_correct :
  final_prob = probability_more_sons_than_daughters_or_more_daughters_than_sons :=
by
  sorry

end probability_correct_l316_316776


namespace painted_polygon_exists_l316_316095

theorem painted_polygon_exists (P : Type) (is_convex : convex_polygon P) 
  (diagonals : set (diagonal P)) 
  (no_three_intersect : ∀ (d1 d2 d3 : diagonal P), 
    d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 → ¬(intersect_three d1 d2 d3)) 
  (painted_borders : ∀ (e : edge P ∨ diagonal P), painted_one_side e) : 
  ∃ (Q : polygon P), is_sub_polygon Q P ∧ completely_painted_outside Q :=
sorry

end painted_polygon_exists_l316_316095


namespace point_outside_circle_l316_316703

theorem point_outside_circle 
  (a b : ℝ) 
  (h1 : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ ax + by = 1) : a^2 + b^2 > 1 :=
by
  sorry

end point_outside_circle_l316_316703


namespace volume_of_piece_containing_W_l316_316954

theorem volume_of_piece_containing_W (unit_cube : set ℝ^3)
    (diagonal_cuts : unit_cube → finset (set ℝ^3))
    (parallel_cuts : unit_cube → finset (set ℝ^3)) :
    (volume_of_piece_containing vertex_W unit_cube diagonal_cuts parallel_cuts = 1/81) :=
sorry

end volume_of_piece_containing_W_l316_316954


namespace number_of_pyramids_with_vertex_of_cube_l316_316459

theorem number_of_pyramids_with_vertex_of_cube :
  let C := fun (n k : ℕ) => (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))),
      num_pyramids := C 8 4 - 12 in
  num_pyramids = C 8 4 - 12 :=
by
  sorry

end number_of_pyramids_with_vertex_of_cube_l316_316459


namespace at_least_half_team_B_can_serve_on_submarine_l316_316132

theorem at_least_half_team_B_can_serve_on_submarine
    (max_height : ℕ)
    (team_A_avg_height : ℕ)
    (team_B_median_height : ℕ)
    (team_C_tallest_height : ℕ)
    (team_D_mode_height : ℕ)
    (h1 : max_height = 168)
    (h2 : team_A_avg_height = 166)
    (h3 : team_B_median_height = 167)
    (h4 : team_C_tallest_height = 169)
    (h5 : team_D_mode_height = 167) :
  ∀ (height : ℕ), height ≤ max_height → ∃ (b_sailors : ℕ → Prop) (H : ∃ n, b_sailors n),
  (∃ (n_half : ℕ), (∀ h ≤ team_B_median_height, b_sailors h) ∧ (2 * n_half ≤ n)) :=
sorry

end at_least_half_team_B_can_serve_on_submarine_l316_316132


namespace find_initial_apples_l316_316873

theorem find_initial_apples (A : ℤ)
  (h1 : 6 * ((A / 8) + 8 - 30) = 12) :
  A = 192 :=
sorry

end find_initial_apples_l316_316873


namespace calc_a_calc_b_calc_c_calc_d_l316_316322

-- Prove 1: Calculation of a
theorem calc_a (x k a : ℝ) (hx : x^2 - 8*x + 26 = (x + k)^2 + a) : a = 10 := sorry

-- Prove 2: Calculation of b
theorem calc_b (a b : ℝ) (ha : sin (a * (real.pi / 180)) = cos (b * (real.pi / 180))) (hb : 270 < b ∧ b < 360) : b = 280 := sorry

-- Prove 3: Calculation of c
theorem calc_c (b c : ℝ) (hb : b = c * (1 - 0.30)) : c = b / 0.70 := sorry

-- Prove 4: Calculation of d
theorem calc_d (c d : ℝ) (h : d = 2 * (3 * c / 10)) : d = (3 * c) / 5 := sorry

end calc_a_calc_b_calc_c_calc_d_l316_316322


namespace total_probability_sum_is_788_l316_316183

def rolls_are_indistinguishable (rolls : List String) : Prop := 
  rolls.perm ["nut", "nut", "nut", "nut", "cheese", "cheese", "cheese", "cheese", "fruit", "fruit", "fruit", "fruit"]

noncomputable def probability_each_guest_gets_one_roll_of_each_type 
  (rolls : List String) 
  (distribution : List (List String))
  (h_perms : ∀ (d : List (List String)), d.perm distribution → ∀ guest, distributionPermutationCorrect rolls guest) : ℚ := 
-- The function body that calculates probability will be developed later.
sorry

theorem total_probability_sum_is_788 : 
  ∀ (rolls : List String) (distribution : List (List String)), 
    rolls_are_indistinguishable rolls → 
    (probability_each_guest_gets_one_roll_of_each_type rolls distribution _ = 18/770) → 
    18 + 770 = 788 :=
by
  intros
  simp [probability_each_guest_gets_one_roll_of_each_type, rolls_are_indistinguishable]
  sorry

end total_probability_sum_is_788_l316_316183


namespace Q_zero_has_unique_solution_l316_316265

noncomputable def Q (x : ℝ) : ℝ :=
  sin x + sin (x + real.pi / 6) - sin (2 * x) - sin (2 * x + real.pi / 3) + sin (3 * x) + sin (3 * x + real.pi / 2)

theorem Q_zero_has_unique_solution : ∃! x ∈ set.Ico 0 (2 * real.pi), Q x = 0 :=
sorry

end Q_zero_has_unique_solution_l316_316265


namespace internal_parallelogram_area_l316_316713

noncomputable def area {α : Type*} [linear_ordered_field α] {A B C D K L M N : α → α} : α := sorry 

theorem internal_parallelogram_area (α : Type*) [linear_ordered_field α]
  (A B C D K L M N : α → α)
  (midpoint_K : K = (A + B) / 2)
  (midpoint_L : L = (B + C) / 2)
  (midpoint_M : M = (C + D) / 2)
  (midpoint_N : N = (D + A) / 2) :
  area A B C D K L M N = (1 / 5) * area A B C D :=
sorry


end internal_parallelogram_area_l316_316713


namespace ashton_pencils_left_l316_316977

def pencils_left (boxes : Nat) (pencils_per_box : Nat) (pencils_given : Nat) : Nat :=
  boxes * pencils_per_box - pencils_given

theorem ashton_pencils_left :
  pencils_left 2 14 6 = 22 :=
by
  sorry

end ashton_pencils_left_l316_316977


namespace frog_escapes_l316_316712

def P : ℕ → ℚ
| 0     := 0
| 12    := 1
| N + 1 := if 1 ≤ N + 1 ∧ N + 1 ≤ 11 then
              ((N + 2) / 13) * P (N) + (1 - (N + 2) / 13) * P (N + 2)
           else if N + 1 = 12 then
              1
           else
              0
| _     := 0

theorem frog_escapes (P2_correct : P 2 = 2261 / 3721) : P 2 = 2261 / 3721 :=
by sorry

end frog_escapes_l316_316712


namespace shift_transform_l316_316113

theorem shift_transform :
  ∀ (x : ℝ), (4 * Real.sin x * Real.cos x) = (Real.sin (2 * (x + (π / 6))) - sqrt 3 * Real.cos (2 * (x + (π / 6)))) :=
by
  sorry

end shift_transform_l316_316113


namespace find_expenditure_in_January_l316_316572

-- Definitions for given conditions
def avg_exp_Jan_to_Jun : ℝ := 4200
def amt_in_July : ℝ := 1500
def avg_exp_Feb_to_Jul : ℝ := 4250
def months_Jan_to_Jun : ℝ := 6
def months_Feb_to_Jul : ℝ := 6
def total_exp_Jan_to_Jun := avg_exp_Jan_to_Jun * months_Jan_to_Jun
def total_exp_Feb_to_Jul := avg_exp_Feb_to_Jul * months_Feb_to_Jul

-- Define the amount spent in January
def amt_in_Jan : ℝ := 1200

-- Lean statement to prove the amount spent in January
theorem find_expenditure_in_January :
  (total_exp_Feb_to_Jul - total_exp_Jan_to_Jun = amt_in_July - amt_in_Jan) →
  amt_in_Jan = 1200 :=
by
  intro h
  rw [← h]
  sorry

end find_expenditure_in_January_l316_316572


namespace ball_reaches_30_feet_at_171_seconds_l316_316454

noncomputable def ball_height_at_t (t : ℝ) : ℝ := 60 - 9 * t - 5 * t^2

theorem ball_reaches_30_feet_at_171_seconds :
  ∃ t : ℝ, t ≈ 1.71 ∧ ball_height_at_t t = 30 :=
by
  have h_eq : ball_height_at_t t = 60 - 9 * t - 5 * t^2 := rfl
  sorry

end ball_reaches_30_feet_at_171_seconds_l316_316454


namespace smallest_n_gt_T1989_2_l316_316677

noncomputable def T (w : ℕ) : ℕ → ℕ
| 1       := w
| (n + 1) := w ^ T n

theorem smallest_n_gt_T1989_2 :
  ∃ n : ℕ, T 3 n > T 2 1989 ∧ ∀ m < n, T 3 m ≤ T 2 1989 :=
  sorry

end smallest_n_gt_T1989_2_l316_316677


namespace c_sub_a_eq_60_l316_316918

theorem c_sub_a_eq_60 (a b c : ℝ) 
  (h1 : (a + b) / 2 = 30) 
  (h2 : (b + c) / 2 = 60) : 
  c - a = 60 := 
by 
  sorry

end c_sub_a_eq_60_l316_316918


namespace solve_nested_sqrt_l316_316816

theorem solve_nested_sqrt (x : ℝ) (h : sqrt (2 + sqrt (3 + sqrt x)) = root 4 (2 + sqrt x)) : x = 0 :=
sorry

end solve_nested_sqrt_l316_316816


namespace triangle_centroid_angle_relation_l316_316755

theorem triangle_centroid_angle_relation
  (A B C G R S : Point)
  (h_centroid : is_centroid G A B C)
  (hR_on_GB : on_ray R G B)
  (hS_on_GC : on_ray S G C)
  (h_angles : ∠ A B S = ∠ A C R ∧ ∠ A B S = 180 - ∠ B G C) : 
  ∠ R A S + ∠ B A C = ∠ B G C :=
sorry

end triangle_centroid_angle_relation_l316_316755


namespace area_of_second_side_l316_316069

theorem area_of_second_side 
  (L W H : ℝ) 
  (h1 : L * H = 120) 
  (h2 : L * W = 60) 
  (h3 : L * W * H = 720) : 
  W * H = 72 :=
sorry

end area_of_second_side_l316_316069


namespace rationalize_denominator_7_over_cube_root_343_l316_316800

theorem rationalize_denominator_7_over_cube_root_343 :
  (7 / Real.cbrt 343) = 1 :=
by {
  have h : Real.cbrt 343 = 7 := rfl,
  rw [h],
  norm_num,
  rw [div_self],
  norm_num,
  sorry
}

end rationalize_denominator_7_over_cube_root_343_l316_316800


namespace avg_growth_rate_total_profit_l316_316233

-- Definitions for conditions
def sales_january : ℕ := 150
def sales_march : ℕ := 216
def cost_price : ℕ := 2300
def selling_price : ℕ := 2800

-- Theorem for the monthly average growth rate of sales
theorem avg_growth_rate (x : ℝ) :
  (sales_january : ℝ) * (1 + x)^2 = (sales_march : ℝ) ↔ x = 0.2 :=
by sorry

-- Theorem for the total profit
theorem total_profit :
  let profit_per_bicycle := selling_price - cost_price in
  let sales_february := sales_january * (1 + 0.2) in 
  let total_sales := sales_january + sales_february + sales_march in
  profit_per_bicycle * total_sales = 273000 :=
by sorry

end avg_growth_rate_total_profit_l316_316233


namespace find_percentage_l316_316528

theorem find_percentage (P : ℕ) (h1 : 0.20 * 650 = 130) (h2 : P * 800 / 100 = 320) : P = 40 := 
by { 
  sorry 
}

end find_percentage_l316_316528


namespace max_distinct_substrings_l316_316428

def is_valid_letter (c : Char) : Prop :=
  c = 'A' ∨ c = 'T' ∨ c = 'C' ∨ c = 'G'

def is_valid_string (s : String) : Prop :=
  s.length = 66 ∧ ∀ c ∈ s, is_valid_letter c

theorem max_distinct_substrings (s : String) (h : is_valid_string s) : 
  ∃ (n : ℕ), n = 2100 :=
by
  sorry

end max_distinct_substrings_l316_316428


namespace g_neg_six_eq_neg_twenty_l316_316853

theorem g_neg_six_eq_neg_twenty (g : ℤ → ℤ)
    (h1 : g 1 - 1 > 0)
    (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
    (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 := 
sorry

end g_neg_six_eq_neg_twenty_l316_316853


namespace jeans_sold_l316_316182

theorem jeans_sold (J : ℕ)
    (shirts_sold : ℕ := 20)
    (shirt_cost : ℕ := 10)
    (jeans_cost : ℕ := 2 * shirt_cost)
    (total_revenue : ℕ := shirts_sold * shirt_cost + J * jeans_cost)
    (total_earnings : ℕ := 400) :
    total_revenue = total_earnings → J = 10 := 
by {
    intros h,
    sorry
}

end jeans_sold_l316_316182


namespace area_ratio_l316_316365

section TriangleProblem

variables (A B C D E F : Type) [AddCommGroup A] [Module ℝ A]
variables (a b c d e f : A)
variables {AD BD AF CF : ℝ} (h_AB : 130 = 130) (h_AC : 130 = 130) (h_AD : AD = 45)
variables (h_CF : CF = 85) (h_BD : BD = 85) (h_AF : AF = 45)

def point_on_line_segment (p1 p2 : A) (t : ℝ) : A := (1 - t) • p1 + t • p2

noncomputable def D := point_on_line_segment a b (45 / 130)
noncomputable def F := point_on_line_segment a c (45 / 130)

noncomputable def E := sorry -- Intersection point of lines DF and BC

theorem area_ratio (h_intersection : D + F = (45 / 130) • b + (85 / 130) • c) :
  sorry -- This needs to be filled with the specifics of the intersection point
  (let area_CFD := sorry; let area_BDE := sorry in area_CFD / area_BDE = 17 / 9) := sorry

end TriangleProblem

end area_ratio_l316_316365


namespace max_balls_drawn_l316_316041

-- Conditions:
variable (items : Type) 
variable (Petya : items → Prop)
variable (yellow round edible : items → Prop)
variable (sun ball tomato banana : items)

variable (Ht : ∀ x, tomato x → round x ∧ ¬yellow x) -- All tomatoes are round and red
variable (Hb : ∀ x, banana x → yellow x ∧ ¬round x) -- All bananas are yellow and not round
variable (Hba : ∀ x, ball x → round x) -- All balls are round

variable (yellow_count : ∑ x in items, yellow x = 15) -- Exactly 15 yellow items
variable (round_count : ∑ x in items, round x = 18) -- Exactly 18 round items
variable (edible_count : ∑ x in items, edible x = 13) -- Exactly 13 edible items

-- Proving the maximum number of balls
theorem max_balls_drawn : ∑ x in items, ball x ≤ 18 :=
by sorry

end max_balls_drawn_l316_316041


namespace average_speed_l316_316186

def segment_distance_and_speed : List (ℕ × ℕ) :=
  [(45, 60), (75, 50), (105, 80), (55, 40)]

def total_distance (segments : List (ℕ × ℕ)) : ℕ :=
  segments.foldl (λ acc seg => acc + seg.fst) 0

def total_time (segments : List (ℕ × ℕ)) : ℝ :=
  segments.foldl (λ acc seg => acc + (seg.fst : ℝ) / (seg.snd : ℝ)) 0

theorem average_speed :
  let segments := segment_distance_and_speed
  let distance := total_distance segments
  let time := total_time segments
  distance = 280 ∧ time = 4.9375 → (distance / time ≈ 56.72) :=
by
  sorry

end average_speed_l316_316186


namespace right_triangle_area_l316_316951

theorem right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) 
  (h_angle_sum : a = 45) (h_other_angle : b = 45) (h_right_angle : c = 90)
  (h_altitude : ∃ height : ℝ, height = 4) :
  ∃ area : ℝ, area = 8 := 
by
  sorry

end right_triangle_area_l316_316951


namespace pairwise_disjoint_chords_exists_l316_316275

theorem pairwise_disjoint_chords_exists (n : ℕ) (h : n = 2^499) :
  (∃ (chords : finset (fin (2 * n) × fin (2 * n))), chords.card = 100 ∧ 
  ∀ (a b ∈ chords), a ≠ b → prod.fst a + prod.snd a = prod.fst b + prod.snd b) :=
sorry

end pairwise_disjoint_chords_exists_l316_316275


namespace problem_statement_l316_316754

-- Given a cubic polynomial with roots p, q, r
def cubic_poly (x : ℂ) := x^3 - 3 * x - 2

-- Definitions of the roots
variables {p q r : ℂ}

-- Conditions based on Vieta's formulas
def sum_roots : Prop := p + q + r = 0
def sum_of_products_of_roots : Prop := p * q + q * r + r * p = -3
def product_of_roots : Prop := p * q * r = 2

-- The mathematical statement to prove
theorem problem_statement (h1 : sum_roots) (h2 : sum_of_products_of_roots) (h3 : product_of_roots) :
  p * (q - r)^2 + q * (r - p)^2 + r * (p - q)^2 = -18 := by
  sorry

end problem_statement_l316_316754


namespace sum_of_series_equals_neg_one_l316_316995

noncomputable def omega : Complex := Complex.exp (2 * π * Complex.I / 17)

theorem sum_of_series_equals_neg_one :
  (∑ k in Finset.range 16, omega ^ (k + 1)) = -1 :=
by
  sorry

end sum_of_series_equals_neg_one_l316_316995


namespace total_boxes_packed_l316_316533

-- Definitions of the conditions
def initial_boxes : ℕ := 400
def food_value_per_box : ℕ := 80
def supplies_value_per_box : ℕ := 165
def donor_multiplier : ℕ := 4

-- Total value of one box
def total_value_per_box : ℕ := food_value_per_box + supplies_value_per_box

-- Initial spending
def initial_spending : ℕ := initial_boxes * total_value_per_box

-- Donation amount
def donation_amount : ℕ := donor_multiplier * initial_spending

-- Number of additional boxes packed with the donation
def additional_boxes : ℕ := donation_amount / total_value_per_box

-- Total number of boxes packed
def total_boxes : ℕ := initial_boxes + additional_boxes

-- Statement to be proven
theorem total_boxes_packed : total_boxes = 2000 := by
  -- Proof for this theorem goes here...
  -- The proof is omitted in this statement as requested.
  sorry

end total_boxes_packed_l316_316533


namespace cindy_gave_25_pens_l316_316915

theorem cindy_gave_25_pens (initial_pens mike_gave pens_given_sharon final_pens : ℕ) (h1 : initial_pens = 5) (h2 : mike_gave = 20) (h3 : pens_given_sharon = 19) (h4 : final_pens = 31) :
  final_pens = initial_pens + mike_gave - pens_given_sharon + 25 :=
by 
  -- Insert the proof here later
  sorry

end cindy_gave_25_pens_l316_316915


namespace part1_part2_part3_l316_316282

open Real

-- Define the set M with the given property
def M (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∃ x0 ∈ D, ∀ x ∈ D, f (x0 + 1) = f x0 + f 1

-- Problem Part 1: f(x) = 1/x ∉ M
theorem part1 (f : ℝ → ℝ) (D : Set ℝ) (hD : D = (-∞, 0) ∪ (0, ∞)) :
  f = (λ x, 1 / x) → ¬ M f D :=
by
  intros hf hM
  simp [hf, M] at hM
  sorry

-- Problem Part 2: Conditions on k and b for f(x) = k * 2^x + b ∈ M
theorem part2 (k b : ℝ) (f : ℝ → ℝ) (D : Set ℝ) (hD : D = univ) :
  f = (λ x, k * 2 ^ x + b) → M f D ↔ (k = 0 ∧ b = 0) ∨ 2 * k + b > 0 :=
by
  intros hf
  simp [hf, M]
  sorry

-- Problem Part 3: Range for a in f(x) = log (a / (x^2 + 2)) ∈ M
theorem part3 (a : ℝ) (f : ℝ → ℝ) (D : Set ℝ) (hD : D = univ) :
  f = (λ x, log (a / (x^2 + 2))) → M f D ↔ (3 / 2 ≤ a ∧ a ≤ 6 ∧ a ≠ 3) :=
by
  intros hf
  simp [hf, M]
  sorry

end part1_part2_part3_l316_316282


namespace probability_vowel_probability_consonant_probability_ch_l316_316269

def word := "дифференцициал"
def total_letters := 12
def num_vowels := 5
def num_consonants := 7
def num_letter_ch := 0

theorem probability_vowel : (num_vowels : ℚ) / total_letters = 5 / 12 := by
  sorry

theorem probability_consonant : (num_consonants : ℚ) / total_letters = 7 / 12 := by
  sorry

theorem probability_ch : (num_letter_ch : ℚ) / total_letters = 0 := by
  sorry

end probability_vowel_probability_consonant_probability_ch_l316_316269


namespace radius_of_inscribed_semicircle_in_right_triangle_l316_316724

-- Definitions representing the conditions
def right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

def triangle_area (a b : ℝ) : ℝ := 1 / 2 * a * b

def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

def inradius (area s : ℝ) : ℝ := area / s

-- Problem statement
theorem radius_of_inscribed_semicircle_in_right_triangle :
  ∀ (PR PQ : ℝ),
  PR = 15 →
  PQ = 8 →
  right_triangle PR PQ (Real.sqrt (PR^2 + PQ^2)) →
  inradius (triangle_area PR PQ) (semiperimeter PR PQ (Real.sqrt (PR^2 + PQ^2))) = 3 :=
by
  intros PR PQ hPR hPQ hRightTriangle
  sorry

end radius_of_inscribed_semicircle_in_right_triangle_l316_316724


namespace cotangent_difference_triangle_l316_316368

-- We will define the problem in terms of a triangle with medians and trigonometric properties.

theorem cotangent_difference_triangle (X Y Z M : Type) 
  (Q : Prop) (u v : ℝ)
  (h_median: ∀ (A : Type), A = YZ) -- It represents that YM = MZ
  (h_angle: ∀ (B : Type), B = 60)  -- It represents that angle made by median with YZ is 60 degrees
  (h_u_eq_v: u = v) 
  (cot_Y : ℝ) (cot_Z : ℝ)
  (h_cot_Y : cot_Y = -((v) / (u + v)))
  (h_cot_Z : cot_Z = ((2*u + v) / (u + v))) :
  |cot_Y - cot_Z| = 5 / 2 := 
    by {
        sorry -- skips the actual proof
    }

end cotangent_difference_triangle_l316_316368


namespace gcd_lcm_product_24_36_l316_316256

-- Definitions for gcd, lcm, and product for given numbers, skipping proof with sorry
theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  -- Sorry used to skip proof
  sorry

end gcd_lcm_product_24_36_l316_316256


namespace find_g_neg_6_l316_316856

noncomputable def g : ℤ → ℤ := sorry

-- Conditions from the problem
axiom g_condition_1 : g 1 - 1 > 0
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The statement we need to prove
theorem find_g_neg_6 : g (-6) = 723 :=
by { sorry }

end find_g_neg_6_l316_316856


namespace tan_alpha_cos2a_div_sin2a_plus1_l316_316274

theorem tan_alpha (α : ℝ) (hα1 : α ∈ Ioc (π/2) π) (hα2 : cos α = -3/5) : 
  tan α = -4/3 := 
sorry

theorem cos2a_div_sin2a_plus1 (α : ℝ) (hα1 : α ∈ Ioc (π/2) π) (hα2 : cos α = -3/5) :
  (cos (2 * α)) / (sin (2 * α) + 1) = -7 := 
sorry

end tan_alpha_cos2a_div_sin2a_plus1_l316_316274


namespace rationalize_denominator_l316_316802

theorem rationalize_denominator (a : ℝ) (n : ℕ) (h : a = 7) (hn : n = 3) :
  7 / real.cbrt (7 ^ 3) = 1 := 
by
  sorry

end rationalize_denominator_l316_316802


namespace set_equality_x_plus_y_l316_316142

theorem set_equality_x_plus_y (x y : ℝ) (A B : Set ℝ) (hA : A = {0, |x|, y}) (hB : B = {x, x * y, Real.sqrt (x - y)}) (h : A = B) : x + y = -2 :=
by
  sorry

end set_equality_x_plus_y_l316_316142


namespace greatest_number_of_donated_oranges_greatest_number_of_donated_cookies_l316_316714

-- Define the given conditions
def totalOranges : ℕ := 81
def totalCookies : ℕ := 65
def numberOfChildren : ℕ := 7

-- Define the floor division for children
def orangesPerChild : ℕ := totalOranges / numberOfChildren
def cookiesPerChild : ℕ := totalCookies / numberOfChildren

-- Calculate leftover (donated) quantities
def orangesLeftover : ℕ := totalOranges % numberOfChildren
def cookiesLeftover : ℕ := totalCookies % numberOfChildren

-- Statements to prove
theorem greatest_number_of_donated_oranges : orangesLeftover = 4 := by {
    sorry
}

theorem greatest_number_of_donated_cookies : cookiesLeftover = 2 := by {
    sorry
}

end greatest_number_of_donated_oranges_greatest_number_of_donated_cookies_l316_316714


namespace angle_between_vectors_l316_316294

open Real EuclideanGeometry

variables {a b : EuclideanSpace ℝ (Fin 2)}

-- Define the conditions
def cond1 : Prop := ‖a‖ = 1
def cond2 : Prop := ‖b‖ = 2
def cond3 : Prop := ∀ t : ℝ, ‖b + t • a‖ ≥ ‖b - a‖

-- Define the goal (to prove the angle is 2π/3)
def angle_goal : Prop :=
  let angle := arccos ((2 • a - b) ⬝ b / (‖2 • a - b‖ * ‖b‖))
  angle = 2 * π / 3

-- The final theorem statement
theorem angle_between_vectors (h1 : cond1) (h2 : cond2) (h3 : cond3) : angle_goal :=
  sorry

end angle_between_vectors_l316_316294


namespace max_balls_count_l316_316025

-- Definitions
def Object := Type
def sun : Object := sorry
def ball : Object := sorry
def tomato : Object := sorry
def banana : Object := sorry

def is_yellow : Object → Prop := sorry
def is_round : Object → Prop := sorry
def is_edible : Object → Prop := sorry

axiom yellow_items_count : ∃ (Y : set Object), Y.card = 15 ∧ ∀ y ∈ Y, is_yellow y
axiom round_items_count : ∃ (R : set Object), R.card = 18 ∧ ∀ r ∈ R, is_round r
axiom edible_items_count : ∃ (E : set Object), E.card = 13 ∧ ∀ e ∈ E, is_edible e

-- Problem conditions
axiom tomato_is_round_and_red : ∀ t, t = tomato → is_round t ∧ ¬is_yellow t
axiom ball_is_round : ∀ b, b = ball → is_round b
axiom banana_is_yellow_and_not_round : ∀ b, b = banana → is_yellow b ∧ ¬is_round b

-- Target proposition
theorem max_balls_count : 
  ∀ (sun_count ball_count tomato_count banana_count : ℕ),
    is_round sun → ¬is_yellow sun →
    is_round ball → is_edible ball → ¬is_yellow ball → 
    is_yellow tomato → is_round tomato → is_edible tomato →
    is_yellow banana → ¬is_round banana → is_edible banana →
    sun_count + ball_count + tomato_count + banana_count = 46 →
    ball_count ≤ 18 := sorry

end max_balls_count_l316_316025


namespace find_omega_l316_316702

open Real Trigonometric

theorem find_omega (ω : ℝ) (hω₀ : ω > 0) :
  (∀ x1 x2, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ π / 3 → sin (ω * x1) < sin (ω * x2)) ∧
  (∀ x1 x2, π / 3 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ π / 2 → sin (ω * x1) > sin (ω * x2)) →
  ω = 3 / 2 :=
by
  sorry

end find_omega_l316_316702


namespace arithmetic_seq_inequality_l316_316396

theorem arithmetic_seq_inequality (a1 a2 a3 : ℝ) (d : ℝ) (h_seq : a2 = a1 + d) (h_seq2 : a3 = a2 + d) (h_pos : 0 < a1) (h_ineq : a1 < a2) :
  a2 > real.sqrt (a1 * a3) :=
by
  sorry

end arithmetic_seq_inequality_l316_316396


namespace inequality_proof_l316_316279

theorem inequality_proof (n : ℕ) (n_ge_two : n ≥ 2) 
  (a : Fin n → ℕ) (ai_strict_increasing : ∀ i j, i < j → a i < a j) 
  (sum_reciprocal_le_one : ∑ i in Finset.range n, (1 / a i) ≤ 1)
  (x : ℝ):
  (∑ i in Finset.range n, (1 / (a i ^ 2 + x ^ 2))) ^ 2 ≤ 
  (1 / 2) * (1 / (a 0 * (a 0 - 1) + x ^ 2)) := 
  sorry

end inequality_proof_l316_316279


namespace total_vegetables_l316_316955

-- Define the initial conditions
def potatoes : Nat := 560
def cucumbers : Nat := potatoes - 132
def tomatoes : Nat := 3 * cucumbers
def peppers : Nat := tomatoes / 2
def carrots : Nat := cucumbers + tomatoes

-- State the theorem to prove the total number of vegetables
theorem total_vegetables :
  560 + (560 - 132) + (3 * (560 - 132)) + ((3 * (560 - 132)) / 2) + ((560 - 132) + (3 * (560 - 132))) = 4626 := by
  sorry

end total_vegetables_l316_316955


namespace nancy_crayons_l316_316779

theorem nancy_crayons (packs : Nat) (crayons_per_pack : Nat) (total_crayons : Nat) 
  (h1 : packs = 41) (h2 : crayons_per_pack = 15) (h3 : total_crayons = packs * crayons_per_pack) : 
  total_crayons = 615 := by
  sorry

end nancy_crayons_l316_316779


namespace negative_reciprocal_of_opposite_abs_neg_three_l316_316090

-- Definitions for opposite number and reciprocal
def opposite_number (x : ℝ) : ℝ := -x
def reciprocal (x : ℝ) : ℝ := 1 / x
def negative_reciprocal (x : ℝ) : ℝ := - reciprocal x

-- Given condition
def abs_neg_three : ℝ := abs (-3)

-- Theorem to prove
theorem negative_reciprocal_of_opposite_abs_neg_three : negative_reciprocal (opposite_number abs_neg_three) = - 1 / 3 := 
sorry

end negative_reciprocal_of_opposite_abs_neg_three_l316_316090


namespace nathaniel_wins_probability_is_5_over_11_l316_316008

open ProbabilityTheory

noncomputable def nathaniel_wins_probability : ℝ :=
  if ∃ n : ℕ, (∑ k in finset.range (n + 1), k % 7) = 0 then
    5 / 11
  else
    sorry

theorem nathaniel_wins_probability_is_5_over_11 :
  nathaniel_wins_probability = 5 / 11 :=
sorry

end nathaniel_wins_probability_is_5_over_11_l316_316008


namespace richmond_tickets_l316_316067

theorem richmond_tickets (total_tickets : ℕ) (second_half_tickets : ℕ) (first_half_tickets : ℕ) :
  total_tickets = 9570 →
  second_half_tickets = 5703 →
  first_half_tickets = total_tickets - second_half_tickets →
  first_half_tickets = 3867 := by
  sorry

end richmond_tickets_l316_316067


namespace union_sets_example_l316_316391

theorem union_sets_example :
  let M := {4, -3}
  let N := {0, -3}
  M ∪ N = {0, -3, 4} :=
by
  sorry

end union_sets_example_l316_316391


namespace supplements_of_congruent_angles_are_congruent_l316_316242

-- Define the concept of supplementary angles
def is_supplementary (α β : ℝ) : Prop := α + β = 180

-- Statement of the problem
theorem supplements_of_congruent_angles_are_congruent :
  ∀ {α β γ δ : ℝ},
  is_supplementary α β →
  is_supplementary γ δ →
  β = δ →
  α = γ :=
by
  intros α β γ δ h1 h2 h3
  sorry

end supplements_of_congruent_angles_are_congruent_l316_316242


namespace find_g_neg6_l316_316836

-- Define the function g with conditions as hypotheses
variables {g : ℤ → ℤ}

-- Condition 1
axiom g_condition_1 : g 1 - 1 > 0

-- Condition 2
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x

-- Condition 3
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The goal is to find g(-6)
theorem find_g_neg6 : g (-6) = 723 :=
by {
  sorry
}

end find_g_neg6_l316_316836


namespace solution_l316_316170

-- Definition of the problem equation
def problem_equation (x : ℝ) : ℝ := 5.46 * cot x - tan x - 2 * tan (2 * x) - 4 * tan (4 * x) + 8

-- Definitions of cotangent and tangent for the problem domain
noncomputable def cot (x : ℝ) : ℝ := 1 / tan x

-- Key identity used: cot(x) - tan(x) = 2 * cot(2x)
lemma cot_tan_identity (x : ℝ) : (cot x - tan x) = 2 * cot (2 * x) := sorry

-- Key identity used: cot(2x) = (cot^2(x) - 1) / (2 * cot(x))
lemma double_angle_identity (x : ℝ) : cot (2 * x) = (cot(x)^2 - 1) / (2 * cot x) := sorry

-- Conclusion: x = (π / 32) * (4k + 3) for k in ℤ
theorem solution (x : ℝ) : (∃ k : ℤ, x = (π / 32) * (4 * k + 3)) ↔ problem_equation x = 0 := sorry

end solution_l316_316170


namespace remaining_money_l316_316195

theorem remaining_money (m : ℝ) (c f t r : ℝ)
  (h_initial : m = 1500)
  (h_clothes : c = (1 / 3) * m)
  (h_food : f = (1 / 5) * (m - c))
  (h_travel : t = (1 / 4) * (m - c - f))
  (h_remaining : r = m - c - f - t) :
  r = 600 := 
by
  sorry

end remaining_money_l316_316195


namespace acute_triangle_l316_316337

theorem acute_triangle (a b c : ℝ) (A B C : ℝ) (h1 : sqrt 3 * a = 2 * c * sin A) 
                      (h2 : c = sqrt 7) 
                      (h3 : (1 / 2) * a * b * sin C = (3 * sqrt 3) / 2)
                      (acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2) :
  a + b = 5 :=
sorry

end acute_triangle_l316_316337


namespace perfect_squares_count_l316_316680

theorem perfect_squares_count (n : ℕ) (h : n = 5^6) : ∃ k : ℕ, k = 4 :=
by
  have h_fact : n = 5^6 := h
  -- Definition of perfect square factors of 15625
  let perfect_squares := {d : ℕ | d ∣ n ∧ ∃ a : ℕ, d = 5^a ∧ even a}
  -- Counting number of perfect square factors
  have count : finset.card perfect_squares = 4 := sorry
  use 4
  exact count

end perfect_squares_count_l316_316680


namespace expected_digits_fair_icosahedral_die_l316_316740

theorem expected_digits_fair_icosahedral_die : 
  let n := 20 in 
  let p1 := 9 / 20 in 
  let p2 := 11 / 20 in 
  let E := p1 * 1 + p2 * 2 in 
  E = 1.55 := 
by 
  let n := 20
  let num_one_digit := 9
  let num_two_digit := 11
  let p1 := num_one_digit / n
  let p2 := num_two_digit / n
  let E := p1 * 1 + p2 * 2
  show E = 1.55 from sorry

end expected_digits_fair_icosahedral_die_l316_316740


namespace correct_propositions_l316_316963

-- Definitions based on the conditions given
def non_parallel_vectors (u v : Vector) : Prop := ¬ (∃ a, u = a • v)

-- 1. Any pair of non-parallel vectors in a plane can serve as a basis
def condition_1 (u v : Vector) : Prop := 
  ∃ u v, non_parallel_vectors u v ∧ (∀ w, ∃ a b, a • u + b • v = w)

-- 2. There are infinitely many pairs of non-parallel vectors that can serve as a basis
def condition_2 : Prop := 
  ∀ plane : Set Vector, ∃ pairs : Set (Vector × Vector), 
  (∀ u v, (u, v) ∈ pairs → non_parallel_vectors u v ∧ (∀ w, ∃ a b, a • u + b • v = w)) ∧ 
  size pairs = ∞

-- 3. Basis vectors can be perpendicular to each other
def condition_3 (u v : Vector) : Prop := 
  non_parallel_vectors u v ∧ u ⬝ v = 0

-- 4. Any non-zero vector in a plane can be represented as a linear combination of three non-parallel vectors
def condition_4 (u v w : Vector) : Prop := 
  non_parallel_vectors u v ∧ non_parallel_vectors u w ∧ non_parallel_vectors v w ∧ 
  ∀ x ≠ 0, ∃ a b c, a • u + b • v + c • w = x ∧ 
  ∀ a' b' c', a' • u + b' • v + c' • w = x → a = a' ∧ b = b' ∧ c = c'

-- Prove that:
theorem correct_propositions : 
  (condition_2 ∧ condition_3) :=
by 
  sorry

end correct_propositions_l316_316963


namespace problem_solution_l316_316366

def triangle_with_parallel_vectors (a b c A B: ℝ) (R: ℝ) (angle_in_range : 0 < B ∧ B < 2 * Real.pi / 3) 
  (angle_eq : ∀ A, A = Real.pi / 3) : Prop :=
  a = 2 ∧ 
  (let m := (a, Real.sqrt 3 * b); n := (Real.cos A, Real.sin B) in
   ((a = 0 ∨ Real.sin B = 0) ∧ (Real.sqrt 3 * b = 0 ∨ Real.cos A = 0)) ∨
    (a * (Real.sin B) - (Real.sqrt 3) * b * (Real.cos A) = 0)) ∧ 
  2 * R = 4 * Real.sqrt 3 / 3 ∧ 
  ∀ (b c B : ℝ), b + c > 2 ∧ b + c ≤ 4

theorem problem_solution 
  (a b c B: ℝ) 
  (A : ℝ := Real.pi / 3) 
  (R: ℝ := 2 * a / Real.sin A)
  (angle_in_range : 0 < B ∧ B < 2 * Real.pi / 3) 
  (angle_eq : ∀ A, A = Real.pi / 3) 
  : 
  triangle_with_parallel_vectors 2 b c A B R angle_in_range angle_eq := 
  sorry

end problem_solution_l316_316366


namespace third_grade_is_40_l316_316556

noncomputable def total_students : ℕ := 2000
noncomputable def total_sample : ℕ := 100
noncomputable def first_grade_sample : ℕ := 30
noncomputable def second_grade_sample : ℕ := 30
noncomputable def third_grade_sample : ℕ := total_sample - first_grade_sample - second_grade_sample

theorem third_grade_is_40 : third_grade_sample = 40 := by
  rw [third_grade_sample, total_sample, first_grade_sample, second_grade_sample]
  norm_num
  done

end third_grade_is_40_l316_316556


namespace monica_subjects_l316_316774

theorem monica_subjects :
  ∃ M : ℕ, (let Marius := M + 4 in let Millie := M + 7 in M + Marius + Millie = 41) ∧ M = 10 :=
by
  use 10 -- Monica takes 10 subjects
  split
  { unfold Marius Millie
    norm_num }
  { norm_num }

end monica_subjects_l316_316774


namespace min_value_l316_316548

variable (a b : ℝ) (z : ℂ) (abs : ℂ → ℝ)

-- Define the complex number and the new operation
def z_def (a b : ℝ) : ℂ := complex.mk a b
def op (z1 z2 : ℂ) : ℝ := (abs z1 + abs z2) / 2

-- Given conditions
axiom abs_z : abs (z_def a b) = real.sqrt (a^2 + b^2)
axiom abs_conjugate_z : abs (complex.conj (z_def a b)) = abs (z_def a b)
axiom sum_condition : a + b = 3

-- Theorem to prove
theorem min_value : (z_def a b * complex.conj (z_def a b)).re = (3*real.sqrt 2) / 2 := sorry

end min_value_l316_316548


namespace average_marks_of_first_class_l316_316824

theorem average_marks_of_first_class 
  (A : ℝ)
  (h1 : ∃ x, x = A * 26) : 
  let B := 60 in
  let total_students := 76 in
  let combined_average := 53.1578947368421 in
  A = 40 :=
by
  let total_marks_first_class := A * 26
  let total_marks_second_class := B * 50
  let total_marks := total_marks_first_class + total_marks_second_class
  let combined_marks := combined_average * 76
  have h : total_marks = combined_marks := by sorry
  have hA : A * 26 = combined_marks - B * 50 := by sorry
  have hA_div := hA / 26
  have h_final : A = 40 := by sorry
  exact h_final

end average_marks_of_first_class_l316_316824


namespace problem1_problem2_problem3_problem4_l316_316219

theorem problem1 : 7 + (-14) - (-9) - (+12) = -10 := by
  sorry

theorem problem2 : (-5) * 6 + (-125) / (-5) = -5 := by
  sorry

theorem problem3 : -10 + 8 / (-2)^2 - 3 * (-4) = 4 := by
  sorry

theorem problem4 : -1^2022 - 2 * (-3)^3 / (-1/2) = -109 := by
  sorry

end problem1_problem2_problem3_problem4_l316_316219


namespace splittable_point_range_l316_316330

theorem splittable_point_range (a : ℝ) (f : ℝ → ℝ) (h_f_def : ∀ x : ℝ, f x = log 5 (a / (2^x + 1)))
  (splittable : ∃ x₀ : ℝ, f (x₀ + 1) = f x₀ + f 1) : a ∈ set.Ioo (3 / 2) 3 :=
by
  sorry

end splittable_point_range_l316_316330


namespace g_minus_6_eq_neg_20_l316_316839

noncomputable def g : ℤ → ℤ := sorry

axiom condition1 : g 1 - 1 > 0
axiom condition2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom condition3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

theorem g_minus_6_eq_neg_20 : g (-6) = -20 := sorry

end g_minus_6_eq_neg_20_l316_316839


namespace cos_theta_correct_lines_intersection_exists_l316_316191

open Real

noncomputable def direction_vec1 := (4 : ℝ, -1 : ℝ)
noncomputable def direction_vec2 := (-2 : ℝ, 5 : ℝ)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def norm (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1^2 + v.2^2)

def cos_theta (v1 v2 : ℝ × ℝ) : ℝ :=
  abs (dot_product v1 v2 / (norm v1 * norm v2))

theorem cos_theta_correct : cos_theta direction_vec1 direction_vec2 = 13 / sqrt 493 :=
by
  sorry

def line1 (t : ℝ) : ℝ × ℝ :=
  (2 + 4 * t, 3 - t)

def line2 (u : ℝ) : ℝ × ℝ :=
  (5 - 2 * u, 6 + 5 * u)

theorem lines_intersection_exists : ∃ t u : ℝ, line1 t = line2 u :=
by
  sorry

end cos_theta_correct_lines_intersection_exists_l316_316191


namespace cheenu_difference_in_time_l316_316151

-- Definitions of conditions
def time_per_mile_young : ℕ := 240 / 20
def time_per_mile_old : ℕ := 300 / 12

-- Theorem stating the problem and expected proof
theorem cheenu_difference_in_time : (time_per_mile_old - time_per_mile_young) = 13 :=
by
  -- Definitions of the expected time per mile computations
  have young_time : time_per_mile_young = 12 := by sorry
  have old_time : time_per_mile_old = 25 := by sorry
  -- Computation using the definitions and solving for the difference
  show (time_per_mile_old - time_per_mile_young) = 13 from
    calc
      time_per_mile_old - time_per_mile_young
      = 25 - 12 : by rw [old_time, young_time]
      ... = 13 : by norm_num

end cheenu_difference_in_time_l316_316151


namespace probability_color_change_l316_316566

noncomputable def cycle_duration : ℕ := 90
noncomputable def green_duration : ℕ := 45
noncomputable def yellow_duration : ℕ := 5
noncomputable def red_duration : ℕ := 40
noncomputable def observation_interval : ℕ := 5

theorem probability_color_change :
  let probability := (3 * observation_interval) / cycle_duration
  in probability = 1 / 6 := 
by
  -- Given durations
  -- Total cycle: 90 seconds
  -- Green: 45 seconds
  -- Yellow: 5 seconds
  -- Red: 40 seconds
  -- Observation interval: 5 seconds
  -- Therefore, the probability calculation follows.
  -- Proof omitted
  sorry

end probability_color_change_l316_316566


namespace prime_divides_l316_316385

theorem prime_divides
  (p : ℕ) (hp : p > 2 ∧ Nat.prime p)
  (m : ℕ) (hm : m > 1)
  (n : ℕ) (hn : n > 0)
  (h_prime : Nat.prime ((m^(p * n) - 1) / (m^n - 1))) :
  p * n ∣ ((p - 1)^n + 1) :=
by
  sorry

end prime_divides_l316_316385


namespace trigonometric_identity_solution_l316_316509

theorem trigonometric_identity_solution (z : ℝ) (k : ℤ) :
  (cos z ≠ 0) →
  (sin z ≠ 0) →
  (5.38 * (1 / (cos z)^4) = (160 / 9) - (2 * ((cos (2*z) / sin (2*z)) * (cos z / sin z) + 1) / (sin z)^2)) →
  ∃ k ∈ ℤ, z = (π / 6) * (3 * k ± 1) :=
by
  sorry

end trigonometric_identity_solution_l316_316509


namespace no_one_lost_money_l316_316156

structure Participant :=
(name : String)
(owes : Participant → ℕ)

def banker : Participant := { name := "banker", owes := λ p, if p = butcher then 5 else 0 }
def butcher : Participant := { name := "butcher", owes := λ p, if p = farmer then 5 else 0 }
def farmer : Participant := { name := "farmer", owes := λ p, if p = merchant then 5 else 0 }
def merchant : Participant := { name := "merchant", owes := λ p, if p = laundress then 5 else 0 }
def laundress : Participant := { name := "laundress", owes := λ p, if p = banker then 5 else 0 }

theorem no_one_lost_money :
  (banker.owes butcher) = 0 ∧ (butcher.owes farmer) = 0 ∧ (farmer.owes merchant) = 0 ∧ (merchant.owes laundress) = 0 ∧ (laundress.owes banker) = 0 :=
by {
    sorry
}

end no_one_lost_money_l316_316156


namespace find_x_l316_316354

/-!
# Problem Statement
Given that the segment with endpoints (-8, 0) and (32, 0) is the diameter of a circle,
and the point (x, 20) lies on the circle, prove that x = 12.
-/

def point_on_circle (x y : ℝ) (center_x center_y radius : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = radius^2

theorem find_x : 
  let center_x := (32 + (-8)) / 2
  let center_y := (0 + 0) / 2
  let radius := (32 - (-8)) / 2
  ∃ x : ℝ, point_on_circle x 20 center_x center_y radius → x = 12 :=
by
  sorry

end find_x_l316_316354


namespace roots_of_unity_quadratic_count_roots_of_unity_quadratic_l316_316586

theorem roots_of_unity_quadratic (a : ℤ) (z : ℂ) (n : ℕ) (hz : z^n = 1) (ha : z^2 + a * z - 1 = 0) : 
  ∃ (a : ℤ), z = 1 ∨ z = -1 := 
sorry

theorem count_roots_of_unity_quadratic : 
  {z : ℂ | ∃ (n : ℕ) (a : ℤ), z^n = 1 ∧ z^2 + a * z - 1 = 0}.to_finset.card = 2 := 
sorry

end roots_of_unity_quadratic_count_roots_of_unity_quadratic_l316_316586


namespace necessary_condition_equiv_l316_316966

variable {X Y A B C P Q : Prop}
variable {x y : ℝ}
variable {a b c : ℝ}

-- Definition of necessary condition
def necessary_condition (p q : Prop) : Prop :=
  q → p

-- Statement of the problem
theorem necessary_condition_equiv : 
  (necessary_condition (x > 5) (x > 10)) ∧ 
  (∀ (a b c : ℝ), c ≠ 0 → necessary_condition (a * c = b * c) (a = b)) ∧ 
  (∀ (x y : ℝ), necessary_condition (2 * x + 1 = 2 * y + 1) (x = y)) :=
  by
    split; sorry
    split; sorry
    split; sorry

end necessary_condition_equiv_l316_316966


namespace radius_of_semi_circle_l316_316466

noncomputable def given_perimeter := 179.95574287564276
noncomputable def pi_approx := 3.141592653589793

theorem radius_of_semi_circle (P : ℝ) (h : P = given_perimeter) : 
  let r := P / (pi_approx + 2) in r ≈ 35 :=
by
  intro P h
  let r := P / (pi_approx + 2)
  have : r = 35 := sorry
  exact this

end radius_of_semi_circle_l316_316466


namespace distance_EC_l316_316894

-- Define the points and given distances as conditions
structure Points :=
  (A B C D E : Type)

-- Distances between points
variables {Points : Type}
variables (dAB dBC dCD dDE dEA dEC : ℝ)
variables [Nonempty Points]

-- Specify conditions: distances in kilometers
def distances_given (dAB dBC dCD dDE dEA : ℝ) : Prop :=
  dAB = 30 ∧ dBC = 80 ∧ dCD = 236 ∧ dDE = 86 ∧ dEA = 40

-- Main theorem: prove that the distance from E to C is 63.4 km
theorem distance_EC (h : distances_given 30 80 236 86 40) : dEC = 63.4 :=
sorry

end distance_EC_l316_316894


namespace min_value_of_x_l316_316288

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : 3 * log 3 x - 2 ≥ log 3 27 + log 3 x) : 
  x ≥ real.exp (5 / 2 * real.log (3 : ℝ)) :=
begin
  sorry
end

end min_value_of_x_l316_316288


namespace like_more_than_half_total_by_3_l316_316956

variables (x : ℕ) -- the number of students who "dislike" photography
variables (like neutral dislike total : ℕ)

-- Conditions
def neutral := dislike + 12
def total := 9 * x
def like := 5 * x
def dislike := x

-- The \\5 students "like", 1 "dislike", and 3 "neutral" ensures the ratio
theorem like_more_than_half_total_by_3 (h : neutral = dislike + 12) 
                                       (h1 : like = 5 * (total/9))
                                       (h2 : total = 9 * dislike):
    like > (total / 2) + 3 :=
sorry

end like_more_than_half_total_by_3_l316_316956


namespace sum_of_diagonals_l316_316387

-- Definitions of the given lengths
def AB := 5
def CD := 5
def BC := 12
def DE := 12
def AE := 18

-- Variables for the diagonal lengths
variables (AC BD CE : ℚ)

-- The Lean 4 theorem statement
theorem sum_of_diagonals (hAC : AC = 723 / 44) (hBD : BD = 44 / 3) (hCE : CE = 351 / 22) :
  AC + BD + CE = 6211 / 132 :=
by
  sorry

end sum_of_diagonals_l316_316387


namespace rationalize_denominator_l316_316801

theorem rationalize_denominator (a : ℝ) (n : ℕ) (h : a = 7) (hn : n = 3) :
  7 / real.cbrt (7 ^ 3) = 1 := 
by
  sorry

end rationalize_denominator_l316_316801


namespace team_B_at_least_half_can_serve_l316_316119

-- Define the height limit condition
def height_limit (h : ℕ) : Prop := h ≤ 168

-- Define the team conditions
def team_A_avg_height : Prop := (160 + 169 + 169) / 3 = 166

def team_B_median_height (B : List ℕ) : Prop :=
  B.length % 2 = 1 ∧ B.perm ([167] ++ (B.eraseNth (B.length / 2))) ∧ B.nth (B.length / 2) = some 167

def team_C_tallest_height (C : List ℕ) : Prop :=
  ∀ (h : ℕ), h ∈ C → h ≤ 169

def team_D_mode_height (D : List ℕ) : Prop :=
  ∃ k, ∀ (h : ℕ), h ≠ 167 ∨ D.count 167 ≥ D.count h

-- Declare the main theorem to be proven
theorem team_B_at_least_half_can_serve (B : List ℕ) :
  (∀ h, h ∈ B → height_limit h) ↔ team_B_median_height B := sorry

end team_B_at_least_half_can_serve_l316_316119


namespace factor_polynomial_l316_316605

def A (x : ℝ) : ℝ := x^2 + 5 * x + 3
def B (x : ℝ) : ℝ := x^2 + 9 * x + 20
def C (x : ℝ) : ℝ := x^2 + 7 * x - 8

theorem factor_polynomial (x : ℝ) :
  (A x) * (B x) + (C x) = (x^2 + 7 * x + 8) * (x^2 + 7 * x + 14) :=
by
  sorry

end factor_polynomial_l316_316605


namespace problem_inequality_l316_316264

theorem problem_inequality 
  (n : ℕ)
  (a b : Fin n → ℝ)
  (h1 : (Finset.univ.sum (λ i, (a i)^2)) = 1) 
  (h2 : (Finset.univ.sum (λ i, (b i)^2)) = 1) : 
  (abs (Finset.univ.sum (λ i, (a i) * (b i)))) ≤ 1 :=
  sorry

end problem_inequality_l316_316264


namespace trajectory_eq_l316_316291

-- Definitions for points A and B and their placements on the x and y axes respectively
def A (a : ℝ) : ℝ × ℝ := (a, 0)
def B (b : ℝ) : ℝ × ℝ := (0, b)
def O : ℝ × ℝ := (0, 0)

-- Definition of distance |AB| and its condition
def dist_AB (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)
def dist_condition (a b : ℝ) : Prop := dist_AB a b = 3

-- Definition for the point P
def P (a b : ℝ) : ℝ × ℝ := (2/3 * a, 1/3 * b)

-- The goal: find a relationship involving x and y coordinates of P
theorem trajectory_eq (a b x y : ℝ) (h1 : dist_condition a b) (h2 : P a b = (x, y)) : 
    x^2 / 4 + y^2 = 1 := sorry

end trajectory_eq_l316_316291


namespace no_lunch_students_l316_316541

variable (total_students : ℕ) (cafeteria_eaters : ℕ) (lunch_bringers : ℕ)

theorem no_lunch_students : 
  total_students = 60 →
  cafeteria_eaters = 10 →
  lunch_bringers = 3 * cafeteria_eaters →
  total_students - (cafeteria_eaters + lunch_bringers) = 20 :=
by
  sorry

end no_lunch_students_l316_316541


namespace perfect_square_sum_pile_l316_316384

theorem perfect_square_sum_pile (n : ℕ) (h : n ≥ 100) :
  ∃ (A B : Finset ℕ), (n ≤ m ∧ m ≤ 2 * n) → (∃ a b ∈ m, ∃ k : ℕ, a + b = k^2) :=
begin
  sorry
end

end perfect_square_sum_pile_l316_316384


namespace xn_plus_inv_xn_is_integer_l316_316401

theorem xn_plus_inv_xn_is_integer (x : ℝ) (hx : x ≠ 0) (k : ℤ) (h : x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m :=
by sorry

end xn_plus_inv_xn_is_integer_l316_316401


namespace grid_problem_l316_316872

theorem grid_problem 
    (A B : ℕ)
    (H1 : 1 ≠ A)
    (H2 : 1 ≠ B)
    (H3 : 2 ≠ A)
    (H4 : 2 ≠ B)
    (H5 : 3 ≠ A)
    (H6 : 3 ≠ B)
    (H7 : A = 2)
    (H8 : B = 1)
    :
    A * B = 2 :=
by
  sorry

end grid_problem_l316_316872


namespace grasshoppers_positions_swap_l316_316106

theorem grasshoppers_positions_swap :
  ∃ (A B C: ℤ), A = -1 ∧ B = 0 ∧ C = 1 ∧
  (∀ m n p : ℤ, (A, B, C) = (m, n, p) → n = 0 → 
  (m^2 - n^2 + p^2 = 0) → A = 1 ∧ B = 0 ∧ C = -1) :=
begin
  -- Adding assumptions
  let x₁ := -1 : ℤ,
  let x₂ := 0 : ℤ,
  let x₃ := 1 : ℤ,
  existsi x₁, existsi x₂, existsi x₃,
  split, refl,
  split, refl,
  split, refl,
  intros m n p hperm hnze hyp,
  sorry -- the detailed proof will go here
end

end grasshoppers_positions_swap_l316_316106


namespace sequence_an_form_l316_316309

-- Definitions based on the given conditions
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ := (n : ℝ)^2 * a n
def a_1 : ℝ := 1

-- The conjecture we need to prove
theorem sequence_an_form (a : ℕ → ℝ) (h₁ : ∀ n ≥ 2, sum_first_n_terms a n = (n : ℝ)^2 * a n)
  (h₂ : a 1 = a_1) :
  ∀ n ≥ 2, a n = 2 / (n * (n + 1)) :=
by
  sorry

end sequence_an_form_l316_316309


namespace unique_three_digit_numbers_eq_180_three_digit_numbers_div_by_5_eq_55_three_digit_numbers_with_odd_eq_90_l316_316482

variable (digits : Finset ℕ) (three_digit_numbers : Finset ℕ)

axiom valid_digits : digits = {0, 1, 2, 3, 4, 5, 6}
axiom valid_three_digit_numbers : three_digit_numbers = {x | ∃ a b c, a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ x = 100 * a + 10 * b + c ∧ a ≠ 0 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c}

-- (1) Prove the number of unique three-digit numbers is 180
theorem unique_three_digit_numbers_eq_180 : three_digit_numbers.card = 180 := by
 sorry

-- (2) Prove the number of unique three-digit numbers that are divisible by $5$ is 55
theorem three_digit_numbers_div_by_5_eq_55
  (div_by_5 : Finset ℕ := {x ∈ three_digit_numbers | x % 5 = 0}) :
  div_by_5.card = 55 := by sorry

-- (3) Prove the number of three-digit numbers that contain at least one odd digit is 90
theorem three_digit_numbers_with_odd_eq_90
  (odd_digits : Finset ℕ := {1, 3, 5})
  (three_digit_numbers_with_odd : Finset ℕ 
    := {x ∈ three_digit_numbers | ∃ a b c, x = 100 * a + 10 * b + c ∧ 
        (a ∈ odd_digits ∨ b ∈ odd_digits ∨ c ∈ odd_digits) }) :
  three_digit_numbers_with_odd.card = 90 := by sorry

end unique_three_digit_numbers_eq_180_three_digit_numbers_div_by_5_eq_55_three_digit_numbers_with_odd_eq_90_l316_316482


namespace three_planes_divide_space_l316_316489

-- Define the conditions given in the problem
def planes_parallel_pairwise (P1 P2 P3 : Plane) : Prop :=
  P1.parallel P2 ∧ P2.parallel P3 ∧ P1.parallel P3

def planes_intersect_pairwise (P1 P2 P3 : Plane) (I1 I2 I3 : Line) : Prop :=
  P1.intersects P2 I1 ∧ P2.intersects P3 I2 ∧ P1.intersects P3 I3 ∧ 
  (∃ p : Point, I1.contains p ∧ I2.contains p ∧ I3.contains p)

-- Define the statement to be proven
theorem three_planes_divide_space (P1 P2 P3 : Plane) (I1 I2 I3 : Line) :
  ((planes_parallel_pairwise P1 P2 P3) → (n = 4)) ∧
  ((planes_intersect_pairwise P1 P2 P3 I1 I2 I3) → (n = 8)) ∧
  (4 ≤ n ∧ n ≤ 8) :=
begin
  sorry
end

end three_planes_divide_space_l316_316489


namespace always_at_least_one_negative_l316_316785

theorem always_at_least_one_negative (x y z : ℤ) (h : x = 1 ∧ y = 1 ∧ z = -1) : 
  ∀ (f : ℤ × ℤ → ℤ × ℤ → ℤ × ℤ), 
  (∀ a b c, b ≠ c → 
    (let (a', b') := f (a, b) (c, 0) in (a', b') = (2 * a + c, 2 * b - c))) →
  ∀ (a b c : ℤ), a = x → b = y → c = z → a < 0 ∨ b < 0 ∨ c < 0 :=
begin
  sorry
end

end always_at_least_one_negative_l316_316785


namespace find_missing_score_and_median_l316_316989

theorem find_missing_score_and_median :
  ∃ (B : ℕ), 
  let scores := [86, B, 82, 88] in 
  (85 = (86 + B + 82 + 88) / 4) ∧ 
  (B ∈ [82, 84, 86, 88] ∧ 85 = (list.nth_le (list.merge_sort (≤) scores) 1 sorry + list.nth_le (list.merge_sort (≤) scores) 2 sorry) / 2) := 
by
  use 84
  sorry

end find_missing_score_and_median_l316_316989


namespace part_a_part_b_l316_316818

open Nat

noncomputable def f : ℕ → ℕ
| 1       := 1
| (n + 1) := if n = f (f n - n + 1) then f n + 2 else f n + 1

theorem part_a (n : ℕ) : f (f n - n + 1) = n ∨ f (f n - n + 1) = n + 1 := 
sorry

noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2

theorem part_b (n : ℕ) : f n = Real.toInt (Real.floor (phi * n)) := 
sorry

end part_a_part_b_l316_316818


namespace partial_fraction_sum_l316_316877

theorem partial_fraction_sum :
  ∃ P Q R : ℚ, 
    P * ((-1 : ℚ) * (-2 : ℚ)) + Q * ((-3 : ℚ) * (-2 : ℚ)) + R * ((-3 : ℚ) * (1 : ℚ))
    = 14 ∧ 
    R * (1 : ℚ) * (3 : ℚ) + Q * ((-4 : ℚ) * (-3 : ℚ)) + P * ((3 : ℚ) * (1 : ℚ)) 
      = 12 ∧ 
    P + Q + R = 115 / 30 := by
  sorry

end partial_fraction_sum_l316_316877


namespace midpoint_sum_of_coordinates_l316_316050

theorem midpoint_sum_of_coordinates
  (M : ℝ × ℝ) (C : ℝ × ℝ) (D : ℝ × ℝ)
  (hmx : (C.1 + D.1) / 2 = M.1)
  (hmy : (C.2 + D.2) / 2 = M.2)
  (hM : M = (3, 5))
  (hC : C = (5, 3)) :
  D.1 + D.2 = 8 :=
by
  sorry

end midpoint_sum_of_coordinates_l316_316050


namespace crayons_total_l316_316898

theorem crayons_total (Wanda Dina Jacob: ℕ) (hW: Wanda = 62) (hD: Dina = 28) (hJ: Jacob = Dina - 2) :
  Wanda + Dina + Jacob = 116 :=
by
  sorry

end crayons_total_l316_316898


namespace product_of_first_two_numbers_l316_316102

theorem product_of_first_two_numbers (A B C : ℕ) (h_coprime: Nat.gcd A B = 1 ∧ Nat.gcd B C = 1 ∧ Nat.gcd A C = 1)
  (h_product: B * C = 1073) (h_sum: A + B + C = 85) : A * B = 703 :=
sorry

end product_of_first_two_numbers_l316_316102


namespace complex_numbers_product_l316_316326

open Complex

theorem complex_numbers_product 
  (z1 z2 : ℂ) 
  (hz1 : ∥z1∥ = 2) 
  (hz2 : ∥z2∥ = 3) 
  (heq : 3 * z1 - 2 * z2 = (3/2) - I) : 
  z1 * z2 = - (30 / 13) + (72 / 13) * I :=
by
  sorry

end complex_numbers_product_l316_316326


namespace find_AD_l316_316727

noncomputable def AD_length (AB AC BD_ratio CD_ratio BC : ℝ) (h_corr: AB = 13 ∧ AC = 20 ∧ BD_ratio = 3/7 ∧ BC = 21 ∧ 40*h^2 = 4681 ∧ h = real.sqrt 117.025) : ℝ :=
10.82

theorem find_AD : 
  ∃ (h : ℝ), AD_length 13 20 (3/7) (7/3) 21 h ∧ h ≈ 10.82 :=
begin
  sorry,
end

end find_AD_l316_316727


namespace matrix_determinant_l316_316221

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![[3, 1, -2],
    [8, 5, -4],
    [3, 3, 6]]

theorem matrix_determinant : det A = 48 := by 
  sorry

end matrix_determinant_l316_316221


namespace max_area_quadrilateral_l316_316656

noncomputable def ellipse_equation : (ℝ → ℝ → Prop) :=
  λ x y, x^2 / 8 + y^2 / 4 = 1

theorem max_area_quadrilateral {A B : ℝ × ℝ} {F1 F2 : ℝ × ℝ} 
  (hA : ellipse_equation A.1 A.2) (hB : ellipse_equation B.1 B.2) 
  (hCenter: A.1 + B.1 = 0 ∧ A.2 + B.2 = 0)
  (F1_eq : F1 = (2, 2)) (F2_eq : F2 = (-2, -2)) : -- Assuming F1 (2,2) and F2 (-2,-2)
  2 * (real.sqrt 8) ≤ 8 :=
by
  sorry

end max_area_quadrilateral_l316_316656


namespace required_large_loans_l316_316112

-- We start by introducing the concepts of the number of small, medium, and large loans
def small_loans : Type := ℕ
def medium_loans : Type := ℕ
def large_loans : Type := ℕ

-- Definition of the conditions as two scenarios
def Scenario1 (m s b : ℕ) : Prop := (m = 9 ∧ s = 6 ∧ b = 1)
def Scenario2 (m s b : ℕ) : Prop := (m = 3 ∧ s = 2 ∧ b = 3)

-- Definition of the problem
theorem required_large_loans (m s b : ℕ) (H1 : Scenario1 m s b) (H2 : Scenario2 m s b) :
  b = 4 :=
sorry

end required_large_loans_l316_316112


namespace area_of_EFCD_is_270_l316_316726

-- Definitions of the lengths and altitude
def AB : ℝ := 10
def CD : ℝ := 26
def altitude : ℝ := 15

-- Position of points E and F
def ratio1 : ℝ := 1 / 4
def ratio2 : ℝ := 3 / 4

-- Length of EF
def EF : ℝ := ratio1 * AB + ratio2 * CD

-- Effective altitude for quadrilateral EFCD
def efcd_altitude : ℝ := ratio2 * altitude

-- Area calculation
def area := (1 / 2) * (EF + CD) * efcd_altitude

-- Given conditions and the problem statement
theorem area_of_EFCD_is_270 : area = 270 := by 
  sorry

end area_of_EFCD_is_270_l316_316726


namespace perp_line_eq_l316_316450

theorem perp_line_eq (p : Point) (l : Line) :
  p = (2, 1) ∧ l = {a := 1, b := 3, c := 4} → ∃ l' : Line, l'.a = 3 ∧ l'.b = -1 ∧ l'.c = -5 :=
by
  sorry

end perp_line_eq_l316_316450


namespace cos_B_is_1_over_12_l316_316283

noncomputable def triangleABC (α β γ A B C : Type) [AddCommGroup α] [Module ℝ α] :
  Type :=
{ centroid : α // ∃ G: α,
  (2 * real.sin A • vector (G - A) +
  real.sqrt 3 * real.sin B • vector (G - B) +
  3 * real.sin C • vector (G - C) = 0) }

theorem cos_B_is_1_over_12
  (α β γ A B C : Type) [AddCommGroup α] [Module ℝ α]
  (tr : triangleABC α β γ A B C) :
  real.cos B = 1 / 12 :=
sorry

end cos_B_is_1_over_12_l316_316283


namespace inverse_of_matrix_A_is_zero_matrix_l316_316249

noncomputable def matrix_A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![5, 15], ![-3, -9]]

theorem inverse_of_matrix_A_is_zero_matrix :
  (matrix_A.det = 0) →
  (∀ (B : Matrix (Fin 2) (Fin 2) ℝ), matrix_A ⬝ B = 1 → B = 0) :=
by
  intro h
  sorry

end inverse_of_matrix_A_is_zero_matrix_l316_316249


namespace abs_difference_equality_l316_316218

theorem abs_difference_equality : (abs (3 - Real.sqrt 2) - abs (Real.sqrt 2 - 2) = 1) :=
  by
    -- Define our conditions as hypotheses
    have h1 : 3 > Real.sqrt 2 := sorry
    have h2 : Real.sqrt 2 < 2 := sorry
    -- The proof itself is skipped in this step
    sorry

end abs_difference_equality_l316_316218


namespace average_score_is_7_standard_deviation_is_2_l316_316521

def scores : List ℝ := [7, 8, 7, 9, 5, 4, 9, 10, 7, 4]

theorem average_score_is_7 : 
  (scores.sum / scores.length : ℝ) = 7 := 
by
  sorry

theorem standard_deviation_is_2 :
  real.sqrt ((scores.map (fun x => (x - (scores.sum / scores.length : ℝ)) ^ 2)).sum / scores.length) = 2 := 
by
  sorry

end average_score_is_7_standard_deviation_is_2_l316_316521


namespace calc1_calc2_l316_316520

variable (a b : ℝ) 

theorem calc1 : (-b)^2 * (-b)^3 * (-b)^5 = b^10 :=
by sorry

theorem calc2 : (2 * a * b^2)^3 = 8 * a^3 * b^6 :=
by sorry

end calc1_calc2_l316_316520


namespace solve_equation_l316_316059

theorem solve_equation (x : ℝ) : x * (x + 5)^3 * (5 - x) = 0 ↔ x = 0 ∨ x = -5 ∨ x = 5 := by
  sorry

end solve_equation_l316_316059


namespace max_balls_drawn_l316_316048

-- Defining the concepts of objects Petya can draw
inductive Object
| Sun
| Ball
| Tomato
| Banana

-- Defining properties for objects
def isYellow (o : Object) : Bool :=
  match o with
  | Object.Banana => true
  | _ => false

def isRound (o : Object) : Bool :=
  match o with
  | Object.Ball   => true
  | Object.Tomato => true
  | _             => false

def isEdible (o : Object) : Bool :=
  match o with
  | Object.Tomato => true
  | Object.Banana => true
  | _             => false

def countObjects (p : Object -> Bool) (os : List Object) : Nat :=
  os.countp p

theorem max_balls_drawn (os : List Object) :
  countObjects isYellow os = 15 →
  countObjects isRound os = 18 →
  countObjects isEdible os = 13 →
  countObjects (λ o => o = Object.Ball) os = 18 :=
by
  intros hy hr he
  -- Proof will go here
  sorry

end max_balls_drawn_l316_316048


namespace midpoint_x_sum_l316_316884

variable {p q r s : ℝ}

theorem midpoint_x_sum (h : p + q + r + s = 20) :
  ((p + q) / 2 + (q + r) / 2 + (r + s) / 2 + (s + p) / 2) = 20 :=
by
  sorry

end midpoint_x_sum_l316_316884


namespace final_sequence_number_l316_316436

theorem final_sequence_number :
  let start := 2^3 * 3^3 * 5^3 * 29 in
  let step1 := start / 3 in
  let step2 := step1 * 4 in
  let step3 := step2 / 3 in
  let step4 := step3 * 4 in
  let step5 := step4 / 3 in
  let step6 := step5 * 4 in
  let step7 := step6 / 3 in
  let step8 := step7 * 4 in
  let step9 := step8 / 3 in
  let step10 := step9 * 4 in
  step10 = 2^13 * 5^3 * 29 := 
by 
  sorry

end final_sequence_number_l316_316436


namespace find_g_minus_6_l316_316844

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Conditions given in the problem
axiom cond1 : g 1 - 1 > 0
axiom cond2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom cond3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The proof we want to make (assertion)
theorem find_g_minus_6 : g (-6) = 723 := 
sorry

end find_g_minus_6_l316_316844


namespace pyramid_volume_l316_316443

open Real

variables (α β S : ℝ) (h : S > 0) (ha : α > 0) (hα : α < π / 2) (hb : β > 0) (hβ : β < π / 2)

theorem pyramid_volume (α β S : ℝ) (hS : S > 0) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  ∃ V, V = (S * tan β * sqrt(S * sin α)) / 6 := 
begin
  sorry
end

end pyramid_volume_l316_316443


namespace find_f_and_extrema_l316_316301

noncomputable def f (a b c : ℝ) (x : ℝ) := -x^3 + a*x^2 + b*x + c
noncomputable def g (a b c : ℝ) (x : ℝ) := f a b c x - a*x^2 + 3

theorem find_f_and_extrema {a b c : ℝ} :
  (∀ x : ℝ, ∃ y : ℝ, tangent_at (f a b c) 1 (-3 * x + 1)) ∧
  (∀ x : ℝ, g a b c x = -(g a b c (-x))) →
  (f a b c = -x^3 - 2*x^2 + 4*x - 3) ∧
  (f a b c (-2) = -11) ∧
  (f a b c (2/3) = -41/27) :=
by {
  sorry
}

end find_f_and_extrema_l316_316301


namespace sequence_2009_l316_316725

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 7
  else (sequence (n - 1) * sequence (n - 2)) % 10

theorem sequence_2009 : sequence 2009 = 2 := by
  -- proof is omitted
  sorry

end sequence_2009_l316_316725


namespace commuting_hours_l316_316371

theorem commuting_hours (walk_hours_per_trip bike_hours_per_trip : ℕ) 
  (walk_trips_per_week bike_trips_per_week : ℕ) 
  (walk_hours_per_trip = 2) 
  (bike_hours_per_trip = 1)
  (walk_trips_per_week = 3) 
  (bike_trips_per_week = 2) : 
  (2 * (walk_hours_per_trip * walk_trips_per_week) + 2 * (bike_hours_per_trip * bike_trips_per_week)) = 16 := 
  by
  sorry

end commuting_hours_l316_316371


namespace sum_of_three_numbers_l316_316456

-- Definitions for the conditions
def mean_condition_1 (x y z : ℤ) := (x + y + z) / 3 = x + 20
def mean_condition_2 (x y z : ℤ) := (x + y + z) / 3 = z - 18
def median_condition (y : ℤ) := y = 9

-- The Lean 4 statement to prove the sum of x, y, and z is 21
theorem sum_of_three_numbers (x y z : ℤ) 
  (h1 : mean_condition_1 x y z) 
  (h2 : mean_condition_2 x y z) 
  (h3 : median_condition y) : 
  x + y + z = 21 := 
  by 
    sorry

end sum_of_three_numbers_l316_316456


namespace largest_even_integer_sum_eq_650_l316_316887

theorem largest_even_integer_sum_eq_650 : 
  ∃ n : ℕ, (n % 2 = 0) ∧ (2 + 4 + ... + 50 = 2 * (25 * (25 + 1) / 2)) ∧ (6 * n - 30 = 650) ∧ (n = 114) := by
begin
  -- Definitions
  let sum_even_25 := 2 * (25 * (25 + 1) / 2),

  -- Prove the sum of first 25 positive even integers is 650
  have h1 : sum_even_25 = 650, -- eventually the calculation and proof steps here
  sorry,

  -- Define n and check the conditions
  existsi 114,
  split,
  {
    refl,
  },
  {
    repeat { sorry },
  }

end largest_even_integer_sum_eq_650_l316_316887


namespace congruent_side_length_of_isosceles_triangle_l316_316969

noncomputable def equilateral_triangle_side := 2
def isosceles_triangle_base := equilateral_triangle_side
noncomputable def equilateral_triangle_area := (Real.sqrt 3)
def isosceles_triangle_area := equilateral_triangle_area / 3
noncomputable def isosceles_triangle_height := (Real.sqrt 3) / 3

theorem congruent_side_length_of_isosceles_triangle
  (a : ℝ) (h_eq : a = equilateral_triangle_side)
  (b : ℝ) (h_base : b = isosceles_triangle_base)
  (A_eq : ℝ) (h_area_eq : A_eq = equilateral_triangle_area)
  (a_iso : ℝ) (h_area_iso : a_iso = isosceles_triangle_area)
  (h : ℝ) (h_height : h = isosceles_triangle_height) :
  let c := Real.sqrt (1 + (Real.sqrt 3 / 3)^2)
  in c = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end congruent_side_length_of_isosceles_triangle_l316_316969


namespace proof_problem_l316_316386

open ProbabilityTheory
open Set

variables {Ω : Type*} [ProbabilitySpace Ω]
variables (A B : Set Ω)

-- conditions
theorem proof_problem (h_exclusive : Disjoint A B) (h_PA : 0 < P[A]) (h_PB : 0 < P[B]) : 
  P[A ∪ B] = P[A] + P[B] :=
sorry

end proof_problem_l316_316386


namespace negation_of_universal_proposition_l316_316671

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > 0 → log x ≥ 2 * (x - 1) / (x + 1)))
  ↔ (∃ x : ℝ, x > 0 ∧ log x < 2 * (x - 1) / (x + 1)) :=
by
  sorry

end negation_of_universal_proposition_l316_316671


namespace max_balls_drawn_l316_316036

structure DrawingConditions where
  yellow_items : ℕ
  round_items : ℕ
  edible_items : ℕ
  tomato_properties : Prop
  ball_properties : Prop
  banana_properties : Prop

axiom petya_conditions : DrawingConditions → Prop 

theorem max_balls_drawn (cond : DrawingConditions)
  (h1 : cond.yellow_items = 15)
  (h2 : cond.round_items = 18)
  (h3 : cond.edible_items = 13)
  (h4 : cond.tomato_properties)
  (h5 : cond.ball_properties)
  (h6 : cond.banana_properties) :
  ∀ balls : ℕ, balls ≤ 18 :=
by
  sorry

end max_balls_drawn_l316_316036


namespace length_of_AX_l316_316359

theorem length_of_AX (AB BC AC : ℝ) (h1 : AB = 60) (h2 : BC = 40) (h3 : AC = 20) 
  (h4 : ∃ X, X ∈ Icc 0 AB ∧ ∠AXC = ∠BXC) : 
  ∃ AX : ℝ, AX = 20 := 
by
  sorry

end length_of_AX_l316_316359


namespace integer_solutions_of_abs_lt_5pi_l316_316687

theorem integer_solutions_of_abs_lt_5pi : 
  let x := 5 * Real.pi in
  ∃ n : ℕ, (∀m : ℤ, abs m < x ↔ m ∈ (Icc (-(n : ℤ)) n)) ∧ n = 15 :=
by
  sorry

end integer_solutions_of_abs_lt_5pi_l316_316687


namespace isosceles_triangle_perimeter_l316_316343

def is_isosceles_triangle (a b c : ℝ) : Prop := (a = b ∨ b = c ∨ a = c) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 2) (h2 : b = 5) :
  ∃ c, is_isosceles_triangle a b c ∧ a + b + c = 12 :=
by {
  use 5,
  split,
  simp [is_isosceles_triangle, h1, h2],
  split,
  linarith,
  split,
  linarith,
  linarith,
  ring,
}

end isosceles_triangle_perimeter_l316_316343


namespace polynomial_coeff_sum_l316_316318

noncomputable def polynomial_expansion (x : ℝ) : ℝ :=
  (2 * x - 1) ^ 2016

theorem polynomial_coeff_sum :
  let p := polynomial_expansion
  (a_0 : ℝ) (a_1 a_2 ... a_{2015} a_{2016} : ℝ) in
  (p 0 = a_0 + a_1 * 0 + a_2 * 0^2 + ... + a_{2016} * 0^2016) →
  (p (1/2) = a_0 + a_1 * (1 / 2) + a_2 * (1 / 2)^2 + ... + a_{2016} * (1 / 2)^2016) →
  a_0 = 1 →
  (a_1 / 2 + a_2 / 2^2 + ... + a_{2016} / 2^2016) = -1 :=
by
  intros
  sorry

end polynomial_coeff_sum_l316_316318


namespace g_minus_6_eq_neg_20_l316_316837

noncomputable def g : ℤ → ℤ := sorry

axiom condition1 : g 1 - 1 > 0
axiom condition2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom condition3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

theorem g_minus_6_eq_neg_20 : g (-6) = -20 := sorry

end g_minus_6_eq_neg_20_l316_316837


namespace max_balls_drawn_l316_316044

-- Defining the concepts of objects Petya can draw
inductive Object
| Sun
| Ball
| Tomato
| Banana

-- Defining properties for objects
def isYellow (o : Object) : Bool :=
  match o with
  | Object.Banana => true
  | _ => false

def isRound (o : Object) : Bool :=
  match o with
  | Object.Ball   => true
  | Object.Tomato => true
  | _             => false

def isEdible (o : Object) : Bool :=
  match o with
  | Object.Tomato => true
  | Object.Banana => true
  | _             => false

def countObjects (p : Object -> Bool) (os : List Object) : Nat :=
  os.countp p

theorem max_balls_drawn (os : List Object) :
  countObjects isYellow os = 15 →
  countObjects isRound os = 18 →
  countObjects isEdible os = 13 →
  countObjects (λ o => o = Object.Ball) os = 18 :=
by
  intros hy hr he
  -- Proof will go here
  sorry

end max_balls_drawn_l316_316044


namespace merchant_profit_percentage_l316_316164

-- Given
def initial_cost_price : ℝ := 100
def marked_price : ℝ := initial_cost_price + 0.50 * initial_cost_price
def discount_percentage : ℝ := 0.20
def discount : ℝ := discount_percentage * marked_price
def selling_price : ℝ := marked_price - discount

-- Prove
theorem merchant_profit_percentage :
  ((selling_price - initial_cost_price) / initial_cost_price) * 100 = 20 :=
by
  sorry

end merchant_profit_percentage_l316_316164


namespace compute_last_two_digits_l316_316743

noncomputable def f (x : ℝ) : ℝ := sorry

theorem compute_last_two_digits :
  (f(2^2^2020) % 100) = 7
  (h1 : ∀ x y : ℝ, x > 0 → y > 0 → f(x) * f(y) = f(x * y) + f(x / y))
  (h2 : f(2) = 3) :
  (f(2^(2^2020)) % 100) = 7 :=
begin
  sorry
end

end compute_last_two_digits_l316_316743


namespace sum_of_powers_eq_one_l316_316998

noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 17)

theorem sum_of_powers_eq_one : (∑ k in Finset.range 1 17, ω ^ k) = 1 :=
by {
  have h : ω ^ 17 = 1 := by {
    rw [ω, Complex.exp_nat_mul, Complex.mul_div_cancel' (2 * Real.pi * Complex.I) (show (17 : ℂ) ≠ 0, by norm_cast; norm_num)],
    rw Complex.exp_cycle,
  },
  sorry
}

end sum_of_powers_eq_one_l316_316998


namespace solveInequalityRegion_l316_316759

noncomputable def greatestIntegerLessThan (x : ℝ) : ℤ :=
  Int.floor x

theorem solveInequalityRegion :
  ∀ (x y : ℝ), abs x < 1 → abs y < 1 → x * y ≠ 0 → (greatestIntegerLessThan (x + y) ≤ 
  greatestIntegerLessThan x + greatestIntegerLessThan y) :=
by
  intros x y h1 h2 h3
  sorry

end solveInequalityRegion_l316_316759


namespace simplify_expression_at_neg4_l316_316430

theorem simplify_expression_at_neg4 {x : ℝ} (h : x = -4) :
  (x^2 - 2 * x) / (x - 3) / ((1 / (x + 3)) + (1 / (x - 3))) = 3 :=
by {
  have expr := (x^2 - 2 * x) / (x - 3) / ((1 / (x + 3)) + (1 / (x - 3))),
  rw h at expr,
  sorry
}

end simplify_expression_at_neg4_l316_316430


namespace gcd_lcm_product_l316_316253

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 24) (h₂ : b = 36) :
  Nat.gcd a b * Nat.lcm a b = 864 :=
by
  rw [h₁, h₂]
  unfold Nat.gcd Nat.lcm
  sorry

end gcd_lcm_product_l316_316253


namespace min_value_option_C_l316_316576

noncomputable def option_A (x : ℝ) : ℝ := x + 4 / x
noncomputable def option_B (x : ℝ) : ℝ := Real.sin x + 4 / Real.sin x
noncomputable def option_C (x : ℝ) : ℝ := Real.exp x + 4 * Real.exp (-x)
noncomputable def option_D (x : ℝ) : ℝ := Real.log x / Real.log 3 + 4 * Real.log 3 / Real.log x

theorem min_value_option_C : ∃ x : ℝ, option_C x = 4 :=
by
  use 0
  -- Proof goes here.
  sorry

end min_value_option_C_l316_316576


namespace focal_length_of_hyperbola_l316_316452

-- Define the equation of the hyperbola
def hyperbola_equation (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m^2 + 12) - y^2 / (4 - m^2) = 1

-- Prove that the focal length is 8
theorem focal_length_of_hyperbola (m : ℝ) (h : hyperbola_equation m) : 2 * real.sqrt ((m^2 + 12) + (4 - m^2)) = 8 := 
by 
  sorry

end focal_length_of_hyperbola_l316_316452


namespace determinant_of_A_l316_316223

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![3, 1, -2], ![8, 5, -4], ![3, 3, 6]]

theorem determinant_of_A : A.det = 48 := 
by
  sorry

end determinant_of_A_l316_316223


namespace minimize_t_at_mean_l316_316644

variable {Q : Type*} [LinearOrder Q]

def dist2 (Q Q₁ : Q) : ℝ := (Q - Q₁) * (Q - Q₁)

noncomputable def t (Q : Q) (Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ : Q) : ℝ :=
  dist2 Q Q₁ + dist2 Q Q₂ + dist2 Q Q₃ + dist2 Q Q₄ + dist2 Q Q₅ + dist2 Q Q₆ + dist2 Q Q₇ + dist2 Q Q₈ + dist2 Q Q₉

theorem minimize_t_at_mean (Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ : Q) :
  ∃ Q : Q, (∀ Q' : Q, t Q Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ ≤ t Q' Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉) ∧
           Q = (Q₁ + Q₂ + Q₃ + Q₄ + Q₅ + Q₆ + Q₇ + Q₈ + Q₉) / 9 :=
sorry

end minimize_t_at_mean_l316_316644


namespace g_is_odd_l316_316232

def g (x : ℝ) : ℝ := log (x + sqrt (1 - x^2))

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  sorry

end g_is_odd_l316_316232


namespace equation_of_curve_C_max_area_of_quadrilateral_point_N_on_fixed_line_l316_316661

section ProblemData

variable (O : Set (ℝ × ℝ)) (A : ℝ × ℝ) (B : ℝ × ℝ) (M : ℝ × ℝ) (C : Set (ℝ × ℝ))
          (T : ℝ × ℝ) (E : ℝ × ℝ) (F : ℝ × ℝ) (G : ℝ × ℝ) (H : ℝ × ℝ) (P : ℝ × ℝ)
          (Q : ℝ × ℝ) (N : ℝ × ℝ)

def CircleO := {p : ℝ × ℝ | p.1^2 + p.2^2 = 16}
def PointA := (6, 0)
def MidPointM := M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2
def LocusC := ∃ B ∈ CircleO, MidPointM
def PointT := (2, 0)
def IntersectsCurveC (l : Set (ℝ × ℝ)) := ∃ E ∈ C, ∃ F ∈ C, E ≠ F ∧ ∀ p ∈ l, p ∈ C → p = E ∨ p = F
def PerpendicularLine (m l : Set (ℝ × ℝ)) := ∃ b ∈ ℝ, m = {p : ℝ × ℝ | p.1 + b = T.1} ∧ T ∈ l
def MaxAreaQuadrilateral := ∃ S, S = 7
def FixedLine := ∃ n ∈ ℝ, N = (-1, n)

end ProblemData

theorem equation_of_curve_C :
  LocusC C → ∀ p ∈ C, (p.1 - 3)^2 + p.2^2 = 4 :=
sorry

theorem max_area_of_quadrilateral :
  ∀ l m : Set (ℝ × ℝ), l ≠ {p : ℝ × ℝ | p.2 = 0} ∧ IntersectsCurveC l → PerpendicularLine m l →
  MaxAreaQuadrilateral :=
sorry

theorem point_N_on_fixed_line :
  ∀ k n : Set (ℝ × ℝ), IntersectsCurveC k ∧ IntersectsCurveC n →
  P ∈ C ∧ Q ∈ C → k = {p : ℝ × ℝ | (P.1 - E.1) * (p.2 - E.2) = (p.1 - E.1) * (P.2 - E.2)} →
  n = {p : ℝ × ℝ | (Q.1 - F.1) * (p.2 - F.2) = (p.1 - F.1) * (Q.2 - F.2)} → N ∈ FixedLine :=
sorry

end equation_of_curve_C_max_area_of_quadrilateral_point_N_on_fixed_line_l316_316661


namespace hair_growth_l316_316370

-- Define the length of Isabella's hair initially and the growth
def initial_length : ℕ := 18
def growth : ℕ := 4

-- Define the final length of the hair after growth
def final_length (initial_length : ℕ) (growth : ℕ) : ℕ := initial_length + growth

-- State the theorem that the final length is 22 inches
theorem hair_growth : final_length initial_length growth = 22 := 
by
  sorry

end hair_growth_l316_316370


namespace Tom_water_intake_daily_l316_316604

theorem Tom_water_intake_daily (cans_per_day : ℕ) (oz_per_can : ℕ) (fluid_per_week : ℕ) (days_per_week : ℕ)
  (h1 : cans_per_day = 5) 
  (h2 : oz_per_can = 12) 
  (h3 : fluid_per_week = 868) 
  (h4 : days_per_week = 7) : 
  ((fluid_per_week - (cans_per_day * oz_per_can * days_per_week)) / days_per_week) = 64 := 
sorry

end Tom_water_intake_daily_l316_316604


namespace curve_properties_l316_316363

-- Definitions
def slope_angle := (3 * Real.pi) / 4
def point_P := (2 : ℝ, 6 : ℝ)
def polar_eq_C (θ : ℝ) : ℝ := 20 * Real.sin ((Real.pi / 4) - θ / 2) * Real.cos ((Real.pi / 4) - θ / 2)

-- Theorem statement
theorem curve_properties :
  (∀ (x y : ℝ), (∃ t : ℝ, (x = 2 - (Real.sqrt 2) / 2 * t) ∧ (y = 6 + (Real.sqrt 2) / 2 * t)) → (x^2 + y^2 - 10 * x = 0)) ∧
  (∃ t1 t2 : ℝ, (point_P : ℝ × ℝ) = (x, y) → (abs t1 + abs t2 = 9 * Real.sqrt 2)) :=
begin
  sorry
end

end curve_properties_l316_316363


namespace mn_not_both_odd_l316_316765

theorem mn_not_both_odd (m n : ℕ) (h : (1 / (m : ℝ) + 1 / (n : ℝ) = 1 / 2020)) :
  ¬ (odd m ∧ odd n) :=
sorry

end mn_not_both_odd_l316_316765


namespace solve_problem_l316_316648

theorem solve_problem (y : ℤ) (h : 3^y + 3^y + 3^y + 3^y = 243) : (y + 2) * (y - 2) = 5 := 
by 
  sorry

end solve_problem_l316_316648


namespace avg_speed_round_trip_l316_316561

-- Definitions for the conditions
def speed_P_to_Q : ℝ := 80
def distance (D : ℝ) : ℝ := D
def speed_increase_percentage : ℝ := 0.1
def speed_Q_to_P : ℝ := speed_P_to_Q * (1 + speed_increase_percentage)

-- Average speed calculation function
noncomputable def average_speed (D : ℝ) : ℝ := 
  let total_distance := 2 * D
  let time_P_to_Q := D / speed_P_to_Q
  let time_Q_to_P := D / speed_Q_to_P
  let total_time := time_P_to_Q + time_Q_to_P
  total_distance / total_time

-- Theorem: Average speed for the round trip is 83.81 km/hr
theorem avg_speed_round_trip (D : ℝ) : average_speed D = 83.81 := 
by 
  -- Dummy proof placeholder
  sorry

end avg_speed_round_trip_l316_316561


namespace at_least_half_team_B_can_serve_l316_316136

theorem at_least_half_team_B_can_serve (height_limit : ℕ)
    (avg_A : ℕ) (median_B : ℕ) (tallest_C : ℕ) (mode_D : ℕ)
    (H1 : height_limit = 168)
    (H2 : avg_A = 166)
    (H3 : median_B = 167)
    (H4 : tallest_C = 169)
    (H5 : mode_D = 167) :
    ∃ (eligible_B : ℕ → Prop), (∀ n, eligible_B n → n ≤ height_limit) ∧ (∃ k, k ≤ median_B ∧ eligible_B k ∧ k ≥ 0) :=
by
  sorry

end at_least_half_team_B_can_serve_l316_316136


namespace weather_hours_correct_l316_316012

-- Define the hours of weather conditions for each day as given in the conditions
def Thursday := (rain: 6, overcast: 6, sunshine: 0, thunderstorm: 0)
def Friday := (rain: 2, overcast: 4, sunshine: 6, thunderstorm: 0)
def Saturday := (rain: 0, overcast: 4, sunshine: 6, thunderstorm: 2)

-- Calculate the total hours for each weather condition over the three days
def totalHours (weather: (rain: Nat, overcast: Nat, sunshine: Nat, thunderstorm: Nat)) : 
  (rain: Nat, overcast: Nat, sunshine: Nat, thunderstorm: Nat) :=
  (rain := weather.1.rain + weather.2.rain + weather.3.rain, 
   overcast := weather.1.overcast + weather.2.overcast + weather.3.overcast,
   sunshine := weather.1.sunshine + weather.2.sunshine + weather.3.sunshine,
   thunderstorm := weather.1.thunderstorm + weather.2.thunderstorm + weather.3.thunderstorm)

-- The proof problem: Prove the total hours for each weather condition as given the correct answer
theorem weather_hours_correct :
  totalHours (Thursday, Friday, Saturday) = (8, 14, 12, 2) :=
by
  sorry

end weather_hours_correct_l316_316012


namespace at_least_half_team_B_can_serve_l316_316135

theorem at_least_half_team_B_can_serve (height_limit : ℕ)
    (avg_A : ℕ) (median_B : ℕ) (tallest_C : ℕ) (mode_D : ℕ)
    (H1 : height_limit = 168)
    (H2 : avg_A = 166)
    (H3 : median_B = 167)
    (H4 : tallest_C = 169)
    (H5 : mode_D = 167) :
    ∃ (eligible_B : ℕ → Prop), (∀ n, eligible_B n → n ≤ height_limit) ∧ (∃ k, k ≤ median_B ∧ eligible_B k ∧ k ≥ 0) :=
by
  sorry

end at_least_half_team_B_can_serve_l316_316135


namespace sum_powers_ω_equals_ω_l316_316990

-- Defining the complex exponential ω = e^(2 * π * i / 17)
def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 17)

-- Statement of the theorem
theorem sum_powers_ω_equals_ω : 
  (Finset.range 16).sum (λ k, Complex.exp (2 * Real.pi * Complex.I * (k + 1) / 17)) = ω :=
sorry

end sum_powers_ω_equals_ω_l316_316990


namespace team_B_eligible_l316_316125

-- Define the conditions
def max_allowed_height : ℝ := 168
def average_height_team_A : ℝ := 166
def median_height_team_B : ℝ := 167
def tallest_sailor_in_team_C : ℝ := 169
def mode_height_team_D : ℝ := 167

-- Define the proof statement
theorem team_B_eligible : 
  (∃ (heights_B : list ℝ), heights_B.length > 0 ∧ median heights_B = median_height_team_B) →
  (∀ h ∈ heights_B, h ≤ max_allowed_height) ∨ (∃ (S : finset ℝ), S.card ≥ heights_B.length / 2 ∧ ∀ h ∈ S, h ≤ max_allowed_height) :=
sorry

end team_B_eligible_l316_316125


namespace sum_powers_ω_equals_ω_l316_316992

-- Defining the complex exponential ω = e^(2 * π * i / 17)
def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 17)

-- Statement of the theorem
theorem sum_powers_ω_equals_ω : 
  (Finset.range 16).sum (λ k, Complex.exp (2 * Real.pi * Complex.I * (k + 1) / 17)) = ω :=
sorry

end sum_powers_ω_equals_ω_l316_316992


namespace cruzs_marbles_l316_316111

theorem cruzs_marbles (Atticus Jensen Cruz : ℕ) 
  (h1 : 3 * (Atticus + Jensen + Cruz) = 60) 
  (h2 : Atticus = Jensen / 2) 
  (h3 : Atticus = 4) : 
  Cruz = 8 := 
sorry

end cruzs_marbles_l316_316111


namespace problem_equivalent_l316_316287

noncomputable def f : ℝ → ℝ := sorry

-- Given conditions:
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f(x) = f(-x)
def periodic_condition (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f(x + 2) = -f(x)
def interval_condition (f : ℝ → ℝ) : Prop := ∀ x : ℝ, (2 ≤ x ∧ x ≤ 3) → f(x) = x

-- Statement to prove:
theorem problem_equivalent :
  even_function f ∧ periodic_condition f ∧ interval_condition f → f(1.5) = 2.5 :=
by
  sorry

end problem_equivalent_l316_316287


namespace not_possible_values_l316_316091

theorem not_possible_values (p h e : ℕ) (H1 : 5 * p - 6 * h = 1240)
  (H2 : p - h = e) (H3 : 6 * h > 0) : 
  (finset.range 249).card = 248 :=
by
  have : h = 5 * e - 1240, by
    calc
      h = 5 * e - 1240 : sorry
  -- Additional reasoning can be provided here if needed.
  trivial

end not_possible_values_l316_316091


namespace ashton_pencils_left_l316_316973

theorem ashton_pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (pencils_given : ℕ) :
  boxes = 2 → pencils_per_box = 14 → pencils_given = 6 → (boxes * pencils_per_box) - pencils_given = 22 :=
by
  intros boxes_eq pencils_per_box_eq pencils_given_eq
  rw [boxes_eq, pencils_per_box_eq, pencils_given_eq]
  norm_num
  sorry

end ashton_pencils_left_l316_316973


namespace length_segment_MN_l316_316670

-- Define the basic setup and conditions of the problem.
variables {P F M N : Type} [coord_space ℝ P] [coord_space ℝ F] [coord_space ℝ M] [coord_space ℝ N]

def parabola (x y : ℝ) := y^2 = 8 * x
def focus := (2, 0 : ℝ)
def directrix (x : ℝ) := x = -2

-- Assert the conditions about the points and the vector relationships.
variables (p: P) (f: F) (m: M) (n: N)
variables (hx1 : ∃ (x1 y1 : ℝ), (parabola x1 y1) ∧ p.directrix (-2))
variables (hx2 : ∃ (x2 y2 : ℝ), (parabola x2 y2) ∧ (directrix l = -2) ∧ (segment (PF) intersects segment (M) and (N)))
variables (vector_relation : (vector PF = 3 * vector MF))

-- Assert the question to be proved
theorem length_segment_MN : length_segment MN = 32 / 3 := 
sorry

end length_segment_MN_l316_316670


namespace Bruce_paid_l316_316585

noncomputable def total_paid : ℝ :=
  let grapes_price := 9 * 70 * (1 - 0.10)
  let mangoes_price := 7 * 55 * (1 - 0.05)
  let oranges_price := 5 * 45 * (1 - 0.15)
  let apples_price := 3 * 80 * (1 - 0.20)
  grapes_price + mangoes_price + oranges_price + apples_price

theorem Bruce_paid (h : total_paid = 1316.25) : true :=
by
  -- This is where the proof would be
  sorry

end Bruce_paid_l316_316585


namespace infinite_geometric_progression_pairs_l316_316596

/-- Statement: Determine the number of pairs (a, b) of real numbers such that 12, a, b, ab form a geometric progression. --/
theorem infinite_geometric_progression_pairs :
  ∃ f : ℝ → ℝ × ℝ, function.surjective f :=
sorry

end infinite_geometric_progression_pairs_l316_316596


namespace trig_expression_value_l316_316628

theorem trig_expression_value (x : ℝ) (h : Real.tan x = 1/2) :
  (2 * Real.sin x + 3 * Real.cos x) / (Real.cos x - Real.sin x) = 8 :=
by
  sorry

end trig_expression_value_l316_316628


namespace car_travel_distance_l316_316178

-- The rate of the car traveling
def rate_of_travel (distance : ℝ) (time : ℝ) : ℝ := distance / time

-- Convert hours to minutes
def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

-- The distance covered
def distance_covered (rate : ℝ) (time : ℝ) : ℝ := rate * time

-- Main theorem statement to prove
theorem car_travel_distance : distance_covered (rate_of_travel 3 4) (hours_to_minutes 2) = 90 := sorry

end car_travel_distance_l316_316178


namespace find_numbers_l316_316263

theorem find_numbers (x y : ℕ) (h1 : x / y = 3) (h2 : (x^2 + y^2) / (x + y) = 5) : 
  x = 6 ∧ y = 2 := 
by
  sorry

end find_numbers_l316_316263


namespace fraction_A_approx_l316_316332

-- Define the conditions
def fraction_B := 1/4
def fraction_C := 1/2
def num_D := 30
def total_students_approx := 600

-- Define the fraction of A's
noncomputable def fraction_A (T : ℝ) : ℝ := 
  1 - fraction_B - fraction_C - (num_D / T)

-- Lean Statement: Verify fraction of A's is approximately 1/5
theorem fraction_A_approx : fraction_A total_students_approx ≈ 1/5 := 
by 
  -- Converting decimals to fractions and specifying approximations
  have h : fraction_A 600 = 1 - 1/4 - 1/2 - 30/600 := rfl
  have h₁ : 30 / 600 = 1 / 20 := by norm_num
  have h₂ : 1 - 1/4 - 1/2 - 1/20 = 1/5 := by norm_num
  rw [h, h₁, h₂]
  exact eq_approx_of_approx total_students_approx 600 sorry

end fraction_A_approx_l316_316332


namespace sum_of_integers_ending_in_6_l316_316216

theorem sum_of_integers_ending_in_6 :
  let seq := list.range' 55 31 |>.filter (λ n, n % 10 == 6),
      S := seq.sum
  in S = 6030 :=
by
  sorry

end sum_of_integers_ending_in_6_l316_316216


namespace page_added_twice_l316_316876

theorem page_added_twice (n k : ℕ) (h1 : (n * (n + 1)) / 2 + k = 1986) : k = 33 :=
sorry

end page_added_twice_l316_316876


namespace train_arrival_problem_shooting_problem_l316_316577

-- Define trials and outcome types
inductive OutcomeTrain : Type
| onTime
| notOnTime

inductive OutcomeShooting : Type
| hitTarget
| missTarget

-- Scenario 1: Train Arrival Problem
def train_arrival_trials_refers_to (n : Nat) : Prop := 
  ∃ trials : List OutcomeTrain, trials.length = 3

-- Scenario 2: Shooting Problem
def shooting_trials_refers_to (n : Nat) : Prop :=
  ∃ trials : List OutcomeShooting, trials.length = 2

theorem train_arrival_problem : train_arrival_trials_refers_to 3 :=
by
  sorry

theorem shooting_problem : shooting_trials_refers_to 2 :=
by
  sorry

end train_arrival_problem_shooting_problem_l316_316577


namespace nonconstant_polynomials_are_quadratics_l316_316380
-- Import necessary lean library

-- Define the statement in Lean 4
theorem nonconstant_polynomials_are_quadratics (n : ℕ) (hn : 3 ≤ n) (f : ℕ → polynomial ℝ)
  (h : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → f k * f (k + 1) = f (k + 1).comp (f (k + 2)))
  (h1 : f (n + 1) = f 1) (h2 : f (n + 2) = f 2) : 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → f k = polynomial.X ^ 2) :=
sorry

end nonconstant_polynomials_are_quadratics_l316_316380


namespace john_change_proof_l316_316738

def quarter_value : ℕ := 25
def dime_value : ℕ := 10
def nickel_value : ℕ := 5

def cost_of_candy_bar : ℕ := 131
def quarters_paid : ℕ := 4
def dimes_paid : ℕ := 3
def nickels_paid : ℕ := 1

def total_payment : ℕ := (quarters_paid * quarter_value) + (dimes_paid * dime_value) + (nickels_paid * nickel_value)
def change_received : ℕ := total_payment - cost_of_candy_bar

theorem john_change_proof : change_received = 4 :=
by
  -- Proof will be provided here
  sorry

end john_change_proof_l316_316738


namespace surface_area_small_prism_l316_316582

-- Definitions and conditions
variables (a b c : ℝ)

def small_cuboid_surface_area (a b c : ℝ) : ℝ :=
  2 * a * b + 2 * a * c + 2 * b * c

def large_cuboid_surface_area (a b c : ℝ) : ℝ :=
  2 * (3 * b) * (3 * b) + 2 * (3 * b) * (4 * c) + 2 * (4 * c) * (3 * b)

-- Conditions
def conditions : Prop :=
  (3 * b = 2 * a) ∧ (a = 3 * c) ∧ (large_cuboid_surface_area a b c = 360)

-- Desired result
def result : Prop :=
  small_cuboid_surface_area a b c = 88

-- The theorem
theorem surface_area_small_prism (a b c : ℝ) (h : conditions a b c) : result a b c :=
by
  sorry

end surface_area_small_prism_l316_316582


namespace sushi_cost_l316_316209

variable (x : ℕ)

theorem sushi_cost (h1 : 9 * x = 180) : x + (9 * x) = 200 :=
by 
  sorry

end sushi_cost_l316_316209


namespace unique_triad_l316_316421

theorem unique_triad (x y z : ℕ) 
  (h_distinct: x ≠ y ∧ y ≠ z ∧ z ≠ x) 
  (h_gcd: Nat.gcd (Nat.gcd x y) z = 1)
  (h_div_properties: (z ∣ x + y) ∧ (x ∣ y + z) ∧ (y ∣ z + x)) :
  (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨
  (x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) :=
sorry

end unique_triad_l316_316421


namespace mrs_jackson_decorations_l316_316777

theorem mrs_jackson_decorations (boxes decorations_in_each_box decorations_used : Nat) 
  (h1 : boxes = 4) 
  (h2 : decorations_in_each_box = 15) 
  (h3 : decorations_used = 35) :
  boxes * decorations_in_each_box - decorations_used = 25 := 
  by
  sorry

end mrs_jackson_decorations_l316_316777


namespace equidistant_points_quadrants_l316_316305

theorem equidistant_points_quadrants (x y : ℝ)
  (h_line : 4 * x + 7 * y = 28)
  (h_equidistant : abs x = abs y) :
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end equidistant_points_quadrants_l316_316305


namespace snacks_in_3h40m_l316_316010

def minutes_in_hours (hours : ℕ) : ℕ := hours * 60

def snacks_in_time (total_minutes : ℕ) (snack_interval : ℕ) : ℕ := total_minutes / snack_interval

theorem snacks_in_3h40m : snacks_in_time (minutes_in_hours 3 + 40) 20 = 11 :=
by
  sorry

end snacks_in_3h40m_l316_316010


namespace figure_50_squares_l316_316244

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 7
  | 2 => 19
  | 3 => 37
  | _ => 3*n^2 + 3*n + 1

theorem figure_50_squares : 
  sequence 50 = 7651 :=
by
  -- exact calculation based on the quadratic formula derived
  sorry

end figure_50_squares_l316_316244


namespace repeating_decimal_division_l316_316146

-- Define x and y as the repeating decimals.
noncomputable def x : ℚ := 84 / 99
noncomputable def y : ℚ := 21 / 99

-- Proof statement of the equivalence.
theorem repeating_decimal_division : (x / y) = 4 := by
  sorry

end repeating_decimal_division_l316_316146


namespace find_g_neg_six_l316_316866

theorem find_g_neg_six (g : ℤ → ℤ)
  (h1 : g 1 - 1 > 0)
  (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
  (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 :=
sorry

end find_g_neg_six_l316_316866


namespace intersection_A_B_l316_316750

def A : Set ℝ := {x | abs x < 2}
def B : Set ℕ := {0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} :=
  sorry

end intersection_A_B_l316_316750


namespace arrangement_impossible_l316_316729

theorem arrangement_impossible : 
  ¬∃ (f : Fin 7 → Fin 8) (x : ℕ), 
  (∀ a b c, a ∈ {0, 1, 2} → b ∈ {3, 4, 5} → c ∈ {6, 7, 8} → 
    (f a + f b + f c = x ∧ f 0 + f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 = 28)) :=
by
  sorry

end arrangement_impossible_l316_316729


namespace route_down_distance_l316_316190

theorem route_down_distance :
  ∀ (rate_up rate_down time_up time_down distance_up distance_down : ℝ),
    -- Conditions
    rate_down = 1.5 * rate_up →
    time_up = time_down →
    rate_up = 6 →
    time_up = 2 →
    distance_up = rate_up * time_up →
    distance_down = rate_down * time_down →
    -- Question: Prove the correct answer
    distance_down = 18 :=
by
  intros rate_up rate_down time_up time_down distance_up distance_down h1 h2 h3 h4 h5 h6
  sorry

end route_down_distance_l316_316190


namespace smallest_n_for_unity_roots_l316_316503

theorem smallest_n_for_unity_roots :
  ∃ n : ℕ, n > 0 ∧ (∀ z : ℂ, (z^6 - z^3 + 1 = 0) → ∃ k : ℤ, z = exp(2 * real.pi * complex.I * k / n)) ∧
  (∀ m : ℕ, m > 0 ∧ (∀ z : ℂ, (z^6 - z^3 + 1 = 0) → ∃ k : ℤ, z = exp(2 * real.pi * complex.I * k / m)) → m ≥ n) :=
begin
  use 9,
  sorry -- The proof goes here
end

end smallest_n_for_unity_roots_l316_316503


namespace P_subset_M_l316_316389

def M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}

def P : Set ℕ := {y | ∃ x, x ∈ M ∧ y = x^2}

theorem P_subset_M : P ⊂ M :=
by
  sorry

end P_subset_M_l316_316389


namespace linear_system_substitution_correct_l316_316623

theorem linear_system_substitution_correct (x y : ℝ)
  (h1 : y = x - 1)
  (h2 : x + 2 * y = 7) :
  x + 2 * x - 2 = 7 :=
by
  sorry

end linear_system_substitution_correct_l316_316623


namespace smallest_triangle_perimeter_l316_316226

theorem smallest_triangle_perimeter :
  ∀ (A B C D : Point) (AC CD : ℕ),
  AC % 2 = 0 ∧ CD % 2 = 0 ∧ (dist B D)^2 = 36 ∧ 
  (dist A B) = (dist A C) ∧ 
  D ∈ line_segment A C ∧ 
  B ∈ line _ (90 : ℕ) (with_origin D) :=
  (dist A B) + (dist B C) + (dist A C) = 24 :=
sorry

end smallest_triangle_perimeter_l316_316226


namespace parabola_coefficients_sum_l316_316073

theorem parabola_coefficients_sum :
  ∃ a b c : ℝ, 
  (∀ y : ℝ, (7 = -(6 ^ 2) * a + b * 6 + c)) ∧
  (5 = a * (-4) ^ 2 + b * (-4) + c) ∧
  (a + b + c = -42) := 
sorry

end parabola_coefficients_sum_l316_316073


namespace range_of_a_l316_316323

open Real

-- Define the equation of the curve
def curve (a : ℝ) : ℝ × ℝ → Prop :=
  λ p, p.1^2 + p.2^2 + 2*a*p.1 - 4*a*p.2 + 5*a^2 - 4 = 0

-- Define the condition for a point to be in the second quadrant
def in_second_quadrant : ℝ × ℝ → Prop :=
  λ p, p.1 < 0 ∧ p.2 > 0

-- The main theorem statement
theorem range_of_a (a : ℝ) :
  (∀ p : ℝ × ℝ, curve a p → in_second_quadrant p) → a > 2 :=
sorry

end range_of_a_l316_316323


namespace problem1_problem2_l316_316273

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem problem1 (x : ℝ) : f x ≤ x + 2 ↔ 0 ≤ x ∧ x ≤ 2 := by
  sorry

theorem problem2 (x : ℝ) (a : ℝ) (h : a ≠ 0) : 
  (f x ≥ (|a + 1| - |2a - 1|) / |a|) ↔ (x ≤ -3/2 ∨ x ≥ 3/2) := by
  sorry

end problem1_problem2_l316_316273


namespace max_shadow_distance_l316_316270

-- Definitions based on conditions
def initial_speed : ℝ := 5
def gravity : ℝ := 10
def time_to_hit : ℝ := 1
def vertical_drop : ℝ := -1

-- Known values from calculations
def sin_alpha : ℝ := 4 / 5
def cos_alpha : ℝ := 3 / 5
def horizontal_acceleration : ℝ := 6

-- Theorem statement
theorem max_shadow_distance : 
  let V := initial_speed
  let g := gravity
  let τ := time_to_hit
  let y_drop := vertical_drop
  let sinα := sin_alpha
  let cosα := cos_alpha
  let a := horizontal_acceleration
in (V^2 * cosα^2) / (2 * a) = 0.75 :=
by
  sorry  -- The proof would be placed here

end max_shadow_distance_l316_316270


namespace solve_for_x_l316_316072

theorem solve_for_x (x : ℝ) :
    (1 / 3 * ((x + 8) + (7 * x + 3) + (3 * x + 9)) = 5 * x - 10) → x = 12.5 :=
by
  intro h
  sorry

end solve_for_x_l316_316072


namespace value_of_expression_l316_316626

theorem value_of_expression (a b : ℝ) (h1 : 10^a = 2) (h2 : 100^b = 7) : 10^(2 * a - 2 * b) = 4 / 7 := 
by
  sorry

end value_of_expression_l316_316626


namespace isosceles_triangle_perimeter_l316_316340

theorem isosceles_triangle_perimeter (a b : ℕ) (c : ℕ) 
  (h1 : a = 5) (h2 : b = 5) (h3 : c = 2) 
  (isosceles : a = b ∨ b = c ∨ c = a) 
  (triangle_inequality1 : a + b > c)
  (triangle_inequality2 : a + c > b)
  (triangle_inequality3 : b + c > a) : 
  a + b + c = 12 :=
  sorry

end isosceles_triangle_perimeter_l316_316340


namespace num_distinct_sums_of_four_distinct_elements_l316_316678

-- Define the set
def S : Set ℕ := {2, 5, 8, 11, 14, 17, 20}

-- Define the condition for sums of four distinct elements
def is_sum_of_four_distinct_elements (n : ℕ) : Prop :=
  ∃ a b c d ∈ S, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + b + c + d = n

-- State the problem
theorem num_distinct_sums_of_four_distinct_elements : 
  (Finset.filter is_sum_of_four_distinct_elements (Finset.Icc 26 62)).card = 12 :=
sorry

end num_distinct_sums_of_four_distinct_elements_l316_316678


namespace largest_tile_side_length_l316_316504

theorem largest_tile_side_length (w l : ℕ) (hw : w = 120) (hl : l = 96) : 
  ∃ s, s = Nat.gcd w l ∧ s = 24 :=
by
  sorry

end largest_tile_side_length_l316_316504


namespace find_g_neg6_l316_316831

-- Define the function g with conditions as hypotheses
variables {g : ℤ → ℤ}

-- Condition 1
axiom g_condition_1 : g 1 - 1 > 0

-- Condition 2
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x

-- Condition 3
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The goal is to find g(-6)
theorem find_g_neg6 : g (-6) = 723 :=
by {
  sorry
}

end find_g_neg6_l316_316831


namespace simplify_polynomial_l316_316056

def P (x : ℝ) : ℝ := 3*x^3 + 4*x^2 - 5*x + 8
def Q (x : ℝ) : ℝ := 2*x^3 + x^2 + 3*x - 15

theorem simplify_polynomial (x : ℝ) : P x - Q x = x^3 + 3*x^2 - 8*x + 23 := 
by 
  -- proof goes here
  sorry

end simplify_polynomial_l316_316056


namespace surface_areas_l316_316188

def lower_base_radius : ℝ := 8
def upper_base_radius : ℝ := 4
def height : ℝ := 5
def slant_height : ℝ := Real.sqrt (height^2 + (lower_base_radius - upper_base_radius)^2)
def lateral_surface_area : ℝ := Real.pi * (lower_base_radius + upper_base_radius) * slant_height
def base1_area : ℝ := Real.pi * lower_base_radius^2
def base2_area : ℝ := Real.pi * upper_base_radius^2
def total_surface_area : ℝ := lateral_surface_area + base1_area + base2_area

theorem surface_areas :
  lateral_surface_area = 12 * Real.pi * Real.sqrt 41 ∧
  total_surface_area = (80 + 12 * Real.sqrt 41) * Real.pi :=
by
  sorry

end surface_areas_l316_316188


namespace additional_track_length_l316_316449

theorem additional_track_length (e : ℝ) (g₁ g₂ : ℝ) (h₁ : g₁ > 0) (h₂ : g₂ > 0) (h_cond : g₁ > g₂) : 
  let initial_length := e / g₁ in
  let desired_length := e / g₂ in
  desired_length - initial_length = 33333 :=
by
  sorry

end additional_track_length_l316_316449


namespace prob_set_S_l316_316559

noncomputable def prob_of_digit (d : ℕ) : ℝ :=
  if d = 0 then 0 else (Real.log (d + 1) / Real.log 10) - (Real.log d / Real.log 10)

theorem prob_set_S :
  let S := {d : ℕ | d ∈ {4, 5, 6, 7, 8}} in
  let P3 := prob_of_digit 3 in
  let PS := ∑ d in S, prob_of_digit d in
  PS = 3 * P3 :=
sorry

end prob_set_S_l316_316559


namespace second_person_days_l316_316804

theorem second_person_days (x : ℕ) (h1 : ∀ y : ℝ, y = 24 → 1 / y = 1 / 24)
  (h2 : ∀ z : ℝ, z = 15 → 1 / z = 1 / 15) :
  (1 / 24 + 1 / x = 1 / 15) → x = 40 :=
by
  intro h
  have h3 : 15 * (x + 24) = 24 * x := sorry
  have h4 : 15 * x + 360 = 24 * x := sorry
  have h5 : 360 = 24 * x - 15 * x := sorry
  have h6 : 360 = 9 * x := sorry
  have h7 : x = 360 / 9 := sorry
  have h8 : x = 40 := sorry
  exact h8

end second_person_days_l316_316804


namespace place_numbers_in_table_l316_316889

theorem place_numbers_in_table (nums : Fin 100 → ℝ) (h_distinct : Function.Injective nums) :
  ∃ (table : Fin 10 → Fin 10 → ℝ),
    (∀ i j, table i j = nums ⟨10 * i + j, sorry⟩) ∧
    (∀ i j k l, (i, j) ≠ (k, l) → (i = k ∧ (j = l + 1 ∨ j = l - 1) ∨ j = l ∧ (i = k + 1 ∨ i = k - 1)) →
      |table i j - table k l| ≠ 1) := sorry  -- Proof omitted

end place_numbers_in_table_l316_316889


namespace triangle_sides_l316_316210

theorem triangle_sides (ABC : Type) (K N M : ABC → ABC → Type)
  (incircle_touches_sides : ∀ A B C : ABC, touches_circle (K A C) (N A B) (M B C))
  (bisects_segment : ∀ A B C : ABC, bisects_segment (K A C))
  (angle_M : ∀ {A B C : ABC}, ∠ (M B C) = 75)
  (product_sides : ∀ {A B C : ABC}, (side_length (K A C)) * (side_length (N A B)) * (side_length (M B C)) = 9 + 6 * sqrt 3) :
  sides_of_triangle (ABC) = (2 * (2 + sqrt 3), 2 * (2 + sqrt 3), 2 * sqrt 3 * (2 + sqrt 3)) :=
sorry

end triangle_sides_l316_316210


namespace manny_original_marbles_l316_316933

/-- 
Let total marbles be 120, and the marbles are divided between Mario, Manny, and Mike in the ratio 4:5:6. 
Let x be the number of marbles Manny is left with after giving some marbles to his brother.
Prove that Manny originally had 40 marbles. 
-/
theorem manny_original_marbles (total_marbles : ℕ) (ratio_mario ratio_manny ratio_mike : ℕ)
    (present_marbles : ℕ) (total_parts : ℕ)
    (h_marbles : total_marbles = 120) 
    (h_ratio : ratio_mario = 4 ∧ ratio_manny = 5 ∧ ratio_mike = 6) 
    (h_total_parts : total_parts = ratio_mario + ratio_manny + ratio_mike)
    (h_manny_parts : total_marbles/total_parts * ratio_manny = 40) : 
  present_marbles = 40 := 
sorry

end manny_original_marbles_l316_316933


namespace find_m_l316_316704

theorem find_m (m : ℝ) (h1 : |m - 3| = 4) (h2 : m - 7 ≠ 0) : m = -1 :=
sorry

end find_m_l316_316704


namespace standard_deviation_is_two_l316_316052

def weights : List ℝ := [125, 124, 121, 123, 127]

noncomputable def mean (l : List ℝ) : ℝ :=
  (l.sum / l.length)

noncomputable def variance (l : List ℝ) : ℝ :=
  ((l.map (λ x => (x - mean l)^2)).sum / l.length)

noncomputable def standard_deviation (l : List ℝ) : ℝ :=
  Real.sqrt (variance l)

theorem standard_deviation_is_two : standard_deviation weights = 2 := 
by
  sorry

end standard_deviation_is_two_l316_316052


namespace airlines_routes_l316_316890

open Function

theorem airlines_routes
  (n_regions m_regions : ℕ)
  (h_n_regions : n_regions = 18)
  (h_m_regions : m_regions = 10)
  (A B : Fin n_regions → Fin n_regions → Bool)
  (h_flight : ∀ r1 r2 : Fin n_regions, r1 ≠ r2 → (A r1 r2 = true ∨ B r1 r2 = true) ∧ ¬(A r1 r2 = true ∧ B r1 r2 = true)) :
  ∃ (routes_A routes_B : List (List (Fin n_regions))),
    (∀ route ∈ routes_A, 2 ∣ route.length) ∧
    (∀ route ∈ routes_B, 2 ∣ route.length) ∧
    routes_A ≠ [] ∧
    routes_B ≠ [] :=
sorry

end airlines_routes_l316_316890


namespace product_is_zero_l316_316235

theorem product_is_zero (n : ℤ) (h : n = 3) :
  (n - 3) * (n - 2) * (n - 1) * n * (n + 1) * (n + 4) = 0 := 
by
  sorry

end product_is_zero_l316_316235


namespace proof_problem_l316_316728

noncomputable theory
open Real

variables 
  (A B C D : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  [normed_vector_space ℝ A] [normed_vector_space ℝ B] [normed_vector_space ℝ C] [normed_vector_space ℝ D]
  (sin_half_angle_ABC: sin (∠ B A C / 2) = (↑3 ^ (1/2)) / 3)
  (AB : ℝ) (BC : ℝ) (BD : ℝ) (AD_ratio : ℝ) (DC_ratio : ℝ)

-- Conditions
def conditions := (AB = 2) ∧
                  (AD_ratio = 2) ∧
                  (DC_ratio = 1) ∧
                  (BD = (4 * ↑3 ^ (1/2)) / 3)

-- Question 1: Find the length of BC
def length_BC (conditions : Prop) : Prop :=
  conditions → (BC = 3)

-- Question 2: Find the area of triangle DBC
def area_triangle_DBC (BC : ℝ) (conditions : Prop) : Prop :=
  conditions → (BC = 3) → (1 / 2 * BD * BC * sin (∠ B D C) = (2 * (↑2 ^ (1/2))) / 3)

-- Theorem statement combining both questions
theorem proof_problem (conditions : Prop) : conditions → 
  (length_BC conditions) →
  (area_triangle_DBC BC conditions) :=
by sorry

end proof_problem_l316_316728


namespace AI_tangent_to_circumcircle_l316_316212

open Set
open Equiv

section Geometry

variables (A B C I M P: Point)
variable [acute_angle_triangle : acute_angle_triangle A B C]
variable [AB_lt_AC : length A B < length A C]
variable [incenter I A B C : incenter I A B C]
variable [midpoint M B C : midpoint M B C]
variable [perpendicular_intersection E A I : drop_perpendicular E A I]
variable [tangent_intersection P M incircle_intersection_A M incircle_intersection_E : tangent_intersection P M incircle_intersection_A M incircle_intersection_E]

theorem AI_tangent_to_circumcircle :
  tangent AI (circumcircle (triangle M I P)) :=
sorry

end Geometry

end AI_tangent_to_circumcircle_l316_316212


namespace pascal_28_25_eq_2925_l316_316148

-- Define the Pascal's triangle nth-row function
def pascal (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the theorem to prove that the 25th element in the 28 element row is 2925
theorem pascal_28_25_eq_2925 :
  pascal 27 24 = 2925 :=
by
  sorry

end pascal_28_25_eq_2925_l316_316148


namespace int_satisfy_property_l316_316594

theorem int_satisfy_property (n : ℤ) : 
  (∀ (a : ℕ → ℤ), ∑ i in finset.range n, a i ∉ n ∣ 0 → 
    ∃ (b : ℕ → ℤ), (∑ i in finset.range n, (i + 1) * (b i)) ∣ n) ↔ 
  ((∃ (k : ℕ), n = 2^k) ∨ (∃ (m : ℕ), n = 2 * m + 1)) := 
by sorry

end int_satisfy_property_l316_316594


namespace percentage_increase_is_200_l316_316868

noncomputable def total_cost : ℝ := 300
noncomputable def rate_per_sq_m : ℝ := 5
noncomputable def length : ℝ := 13.416407864998739
noncomputable def area : ℝ := total_cost / rate_per_sq_m
noncomputable def breadth : ℝ := area / length
noncomputable def percentage_increase : ℝ := (length - breadth) / breadth * 100

theorem percentage_increase_is_200 :
  percentage_increase = 200 :=
by
  sorry

end percentage_increase_is_200_l316_316868


namespace distinct_parenthesizations_of_3_3_3_3_l316_316603

theorem distinct_parenthesizations_of_3_3_3_3 : 
  ∃ (v1 v2 v3 v4 v5 : ℕ), 
    v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v1 ≠ v5 ∧ 
    v2 ≠ v3 ∧ v2 ≠ v4 ∧ v2 ≠ v5 ∧ 
    v3 ≠ v4 ∧ v3 ≠ v5 ∧ 
    v4 ≠ v5 ∧ 
    v1 = 3 ^ (3 ^ (3 ^ 3)) ∧ 
    v2 = 3 ^ ((3 ^ 3) ^ 3) ∧ 
    v3 = (3 ^ 3) ^ (3 ^ 3) ∧ 
    v4 = ((3 ^ 3) ^ 3) ^ 3 ∧ 
    v5 = 3 ^ (27 ^ 27) :=
  sorry

end distinct_parenthesizations_of_3_3_3_3_l316_316603


namespace common_chord_line_l316_316611

variable {r θ : ℝ}

-- Definitions based on the given conditions
def circle1 (ρ : ℝ) : Prop := ρ = r
def circle2 (ρ : ℝ) : Prop := ρ = -2 * r * (Real.sin (θ + π/4))

theorem common_chord_line : 
  ∀ (ρ : ℝ), (circle1 ρ) → (circle2 ρ) → r > 0 → (√2 * ρ * (Real.sin θ + Real.cos θ) = -r) :=
by
  intros ρ h1 h2 hr_gt_zero
  sorry

end common_chord_line_l316_316611


namespace area_triangle_ABC_l316_316367

-- Given conditions for the problem
variables {α : Type*} [linear_ordered_field α] (a b c : α) (A C : α)

-- Hypotheses and statements for part (1)
lemma measure_angle_A
  (h : a * (cos C) + c * (cos A) = 2 * b * (cos A)) (h_angles: 0 < A ∧ A < π) :
  A = π / 3 :=
sorry

-- Hypotheses and statements for part (2)
noncomputable def area_of_triangle
  (A C : α) (a c : α) (hA : A = π / 3) (hAC : a = sqrt 3) (hc : c = 2) :
  α :=
(1/2) * c * 1 * (sin (π / 3))

theorem area_triangle_ABC
  (a b c : α) (A C : α)
  (hAC : a = sqrt (3 : α)) (HC : c = 2) (h_a : A = π / 3) :
  area_of_triangle A C a c h_a hAC HC = sqrt (3 : α) / 2 :=
begin
  unfold area_of_triangle,
  rw [hAC, HC, h_a],
  sorry
end

end area_triangle_ABC_l316_316367


namespace nancy_crayons_l316_316782

theorem nancy_crayons (p c t : ℕ) (h1 : p = 41) (h2 : c = 15) (h3 : t = p * c) : t = 615 :=
by
  sorry

end nancy_crayons_l316_316782


namespace concurrency_of_lines_l316_316783

noncomputable def radius (center : Point) (circle : Circle) : Real := sorry
noncomputable def distance_to_side (point : Point) (side : Line) : Real := sorry

variable (A B C Oa Ob Oc : Point)
variable (ra rb rc : ℝ)
variable (triangle : Triangle)
variable (func_side_BC func_side_CA func_side_AB : Point → ℝ)
variable (circle_tangent_equiv_1 : ∀ X, tangent_to_same_side_BC -> func_side_BC X = ra)
variable (circle_tangent_equiv_2 : ∀ X, tangent_to_same_side_CA -> func_side_CA X = rb)
variable (circle_tangent_equiv_3 : ∀ X, tangent_to_same_side_AB -> func_side_AB X = rc)
variable (prop_relation_1 : distance_to_side Ob (side BC) / rb = distance_to_side Oa (side CA) / ra)
variable (prop_relation_2 : distance_to_side Oc (side AB) / rc = distance_to_side Ob (side BC) / rb)
variable (prop_relation_3 : distance_to_side Oa (side CA) / ra = distance_to_side Oc (side AB) / rc)

theorem concurrency_of_lines :
  let d_a (X : Point) := distance_to_side X (side BC)
  let d_b (X : Point) := distance_to_side X (side CA)
  let d_c (X : Point) := distance_to_side X (side AB) in
  d_a Ob / rb * d_b Oc / rc * d_c Oa / ra = 1 →
  ∃ P : Point, collinear P A Oa ∧ collinear P B Ob ∧ collinear P C Oc :=
begin
  sorry
end

end concurrency_of_lines_l316_316783


namespace catherine_last_three_digits_l316_316988

/-- Catherine starts listing numbers with the first digit 2.
    Find the three-digit number formed by the digits positioned at 1498th, 1499th, and 1500th places.
 -/
theorem catherine_last_three_digits : 
  let digits := list.to_string (list.join (list.map to_string (list.filter (λ n, n / 10^(nat.log10 n) = 2) (list.range (10^5)))))
  in (digits.get_or_else ⟨1497, ' '⟩, digits.get_or_else ⟨1498, ' '⟩, digits.get_or_else ⟨1499, ' '⟩) = ('2', '2', '9') := sorry

end catherine_last_three_digits_l316_316988


namespace ravi_overall_profit_l316_316805

theorem ravi_overall_profit
  (refrigerator_original : ℝ) (phone_original : ℝ) (washing_machine_original : ℝ)
  (discount_refrigerator : ℝ) (discount_phone : ℝ)
  (loss_refrigerator : ℝ) (profit_phone : ℝ) (profit_washing_machine : ℝ) 
  (tax_washing_machine : ℝ)
  (refrigerator_original = 15000) (phone_original = 8000) (washing_machine_original = 10000)
  (discount_refrigerator = 0.05) (discount_phone = 0.07)
  (loss_refrigerator = 0.06) (profit_phone = 0.12) (profit_washing_machine = 0.08) 
  (tax_washing_machine = 0.03):
  let refrigerator_purchase := refrigerator_original * (1 - discount_refrigerator),
      phone_purchase := phone_original * (1 - discount_phone),
      washing_machine_purchase := washing_machine_original,
      refrigerator_selling := refrigerator_purchase * (1 - loss_refrigerator),
      phone_selling := phone_purchase * (1 + profit_phone),
      washing_machine_selling := washing_machine_purchase * (1 + profit_washing_machine),
      washing_machine_selling_with_tax := washing_machine_selling * (1 + tax_washing_machine),
      total_purchase := refrigerator_purchase + phone_purchase + washing_machine_purchase,
      total_selling := refrigerator_selling + phone_selling + washing_machine_selling_with_tax,
      overall_profit := total_selling - total_purchase,
      overall_profit_percentage := (overall_profit / total_purchase) * 100
  in overall_profit_percentage = 3.66 :=
sorry

end ravi_overall_profit_l316_316805


namespace problem_statement_l316_316247

noncomputable def n : ℕ := 4

def a_i x y : ℝ := sorry
def b_i x y : ℝ := sorry
def c_i x y : ℝ := sorry

def condition (x y : ℝ) : Prop :=
  (x, y) coordinates a vertex of a regular octagon

theorem problem_statement (x y : ℝ) (h : condition x y) :
  ∑ i in finset.range n, |(a_i x y) * x + (b_i x y) * y + (c_i x y)| = 10 :=
sorry

end problem_statement_l316_316247


namespace jeremy_money_ratio_l316_316736

theorem jeremy_money_ratio :
  let cost_computer := 3000
  let cost_accessories := 0.10 * cost_computer
  let money_left := 2700
  let total_spent := cost_computer + cost_accessories
  let money_before_purchase := total_spent + money_left
  (money_before_purchase / cost_computer) = 2 := by
  sorry

end jeremy_money_ratio_l316_316736


namespace problem_statement_l316_316923

-- Define propositions 甲, 乙, and 丙
variables (P Q R : Prop)

-- Conditions based on the problem statement
def condition1 : Q → P := sorry  -- 乙 implies 甲 (Q implies P)
def condition2 : R → Q := sorry  -- 丙 implies 乙 (R implies Q)
def condition3 : ¬ (Q → R) := sorry  -- 乙 does not imply 丙 (not (Q implies R))

-- Prove that 丙 is a sufficient but not necessary condition for 甲
theorem problem_statement : (R → P) ∧ (¬ (P → R)) :=
begin
  sorry,  -- Proof will be provided here
end

end problem_statement_l316_316923


namespace rhombus_OA_OC_constant_trajectory_of_C_l316_316357

namespace RhombusProof

variables (O A B C D : Point)
variables (a b c d : ℝ)
variables (M : ℝ → ℝ × ℝ)

def semicircle (x₁ x₂ r : ℝ) (x y : ℝ) : Prop := (x - x₁)^2 + y^2 = r^2 ∧ x₂ ≤ x ∧ x ≤ x₁

theorem rhombus_OA_OC_constant
  (hABCD : rhombus O A B C D)
  (h_side_length : side_length O A B C D = 4)
  (h_point_lengths : |O - B| = 6 ∧ |O - D| = 6) :
  |O - A| * |O - C| = 20 := sorry

theorem trajectory_of_C 
  (hA_on_semicircle : ∀ t, 2 ≤ t ∧ t ≤ 4 → (A = M t))
  (hM_semi : ∀ x y, semicircle 2 4 2 x y) :
  ∀ y, -5 ≤ y ∧ y ≤ 5 → C = (5, y) := sorry

end RhombusProof

end rhombus_OA_OC_constant_trajectory_of_C_l316_316357


namespace sum_of_a_n_sum_of_b_n_lambda_range_l316_316099

noncomputable def a (n : ℕ) : ℕ := if n = 1 then 1 else 2^(n-1)
noncomputable def b (n : ℕ) : ℚ := if n = 1 then 1 else 2 / (n * (n + 1))

def S (n : ℕ) : ℕ := a 1 + (n > 1).pred * sum (Finset.range n ) (λ i, 2^i)

def T (n : ℕ) : ℚ := match n with
| 0       => 0
| m + 1  => (Finset.range (m + 1)).sum (λ i, b (i + 1))

theorem sum_of_a_n (n : ℕ) (hn : 1 < n) :
    ∀ k ≥ 2, a k = 2^(k-1) :=
by
  sorry

theorem sum_of_b_n (n : ℕ) :
    b n = 2 / (n * (n + 1)) :=
by
  sorry

theorem lambda_range :
    ∀ (n : ℕ), n ≥ 1 → (S n + 1 > λ / b n) → λ < 4/3 :=
by
  sorry

end sum_of_a_n_sum_of_b_n_lambda_range_l316_316099


namespace find_g_neg_six_l316_316863

theorem find_g_neg_six (g : ℤ → ℤ)
  (h1 : g 1 - 1 > 0)
  (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
  (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 :=
sorry

end find_g_neg_six_l316_316863


namespace mean_of_other_four_l316_316823

theorem mean_of_other_four (a b c d e : ℕ) (h_mean : (a + b + c + d + e + 90) / 6 = 75)
  (h_max : max a (max b (max c (max d (max e 90)))) = 90)
  (h_twice : b = 2 * a) :
  (a + c + d + e) / 4 = 60 :=
by
  sorry

end mean_of_other_four_l316_316823


namespace transformation_correctness_l316_316913

theorem transformation_correctness :
  (∀ x : ℝ, 3 * x = -4 → x = -4 / 3) ∧
  (∀ x : ℝ, 5 = 2 - x → x = -3) ∧
  (∀ x : ℝ, (x - 1) / 6 - (2 * x + 3) / 8 = 1 → 4 * (x - 1) - 3 * (2 * x + 3) = 24) ∧
  (∀ x : ℝ, 3 * x - (2 - 4 * x) = 5 → 3 * x + 4 * x - 2 = 5) :=
by
  -- Prove the given conditions
  sorry

end transformation_correctness_l316_316913


namespace simplify_polynomial_l316_316431

theorem simplify_polynomial (y : ℝ) :
  (3 * y - 2) * (5 * y ^ 12 + 3 * y ^ 11 + y ^ 10 + 2 * y ^ 9) =
  15 * y ^ 13 - y ^ 12 - 3 * y ^ 11 + 4 * y ^ 10 - 4 * y ^ 9 := 
by
  sorry

end simplify_polynomial_l316_316431


namespace limit_nb_n_l316_316230

noncomputable def P (x : ℝ) : ℝ := x - (x^2 / 4)

def b_n (n : ℕ) : ℝ := if n > 0 then 
  let rec iterate_P (x : ℝ) (k : ℕ) : ℝ :=
    if k = 0 then x else P (iterate_P x (k - 1))
  iterate_P (25 / n) n
else 0

theorem limit_nb_n : filter.tendsto (λ n : ℕ, (n : ℝ) * b_n n) filter.at_top (filter.tendsto_const_nhds (50 / 27)) :=
sorry

end limit_nb_n_l316_316230


namespace car_travel_distance_l316_316179

-- The rate of the car traveling
def rate_of_travel (distance : ℝ) (time : ℝ) : ℝ := distance / time

-- Convert hours to minutes
def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

-- The distance covered
def distance_covered (rate : ℝ) (time : ℝ) : ℝ := rate * time

-- Main theorem statement to prove
theorem car_travel_distance : distance_covered (rate_of_travel 3 4) (hours_to_minutes 2) = 90 := sorry

end car_travel_distance_l316_316179


namespace find_f6_l316_316642

def odd_function_on_reals (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = -f(x)

def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x + 2) = -f(x)

theorem find_f6 (f : ℝ → ℝ) (h1 : odd_function_on_reals f) (h2 : periodic_function f) : 
  f(6) = 0 :=
sorry

end find_f6_l316_316642


namespace value_of_m_l316_316637

theorem value_of_m (m : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = x^2 - m * x + m - 1) 
  (h_eq : f 0 = f 2) : m = 2 :=
sorry

end value_of_m_l316_316637


namespace geometric_sequence_n_terms_l316_316334

/-- In a geometric sequence with the first term a₁ and common ratio q,
the number of terms n for which the nth term aₙ has a given value -/
theorem geometric_sequence_n_terms (a₁ aₙ q : ℚ) (n : ℕ)
  (h1 : a₁ = 9/8)
  (h2 : aₙ = 1/3)
  (h3 : q = 2/3)
  (h_seq : aₙ = a₁ * q^(n-1)) :
  n = 4 := sorry

end geometric_sequence_n_terms_l316_316334


namespace num_positive_integers_log_b_1024_l316_316316

theorem num_positive_integers_log_b_1024 :
  ∃ (n : ℕ), n = 4 ∧ ∀ b : ℕ, (∃ k : ℕ, k ∣ 10 ∧ b = 2^k) → ∃ m : ℕ, log b 1024 = m ∧ 0 < m :=
by
  sorry

end num_positive_integers_log_b_1024_l316_316316


namespace count_of_four_digit_numbers_l316_316681

def four_digit_numbers_satisfy_conditions : Prop :=
  ∃ N : ℕ, 4000 ≤ N ∧ N < 6000 ∧
           (N % 10 = 0) ∧
           (∃ a b c d : ℕ, 
             N = 1000 * a + 100 * b + 10 * c + d ∧ 
             (a = 4 ∨ a = 5) ∧
             3 ≤ b ∧ b ≤ c ∧ c ≤ 7 
           )

theorem count_of_four_digit_numbers : (finset.range 10000).filter four_digit_numbers_satisfy_conditions).card = 30 :=
sorry

end count_of_four_digit_numbers_l316_316681


namespace intersection_of_sets_l316_316749

-- Define sets A and B as given in the conditions
def A : Set ℝ := { x | -2 < x ∧ x < 2 }

def B : Set ℝ := {0, 1, 2}

-- Define the proposition to be proved
theorem intersection_of_sets : A ∩ B = {0, 1} :=
by
  sorry

end intersection_of_sets_l316_316749


namespace tetrahedron_surface_area_ratio_tetrahedron_volume_ratio_l316_316444

noncomputable def edge_length_ratio : ℝ := 1/3

def surface_area (a : ℝ) : ℝ := 4 * sqrt 3 / 4 * a^2

def volume (a : ℝ) : ℝ := a^3 / (6 * sqrt 2)

theorem tetrahedron_surface_area_ratio (a : ℝ) :
  let a2 := a * edge_length_ratio in
  let S1 := surface_area a in
  let S2 := surface_area a2 in
  S1 / S2 = 9 := sorry

theorem tetrahedron_volume_ratio (a : ℝ) :
  let a2 := a * edge_length_ratio in
  let V1 := volume a in
  let V2 := volume a2 in
  V1 / V2 = 27 := sorry

end tetrahedron_surface_area_ratio_tetrahedron_volume_ratio_l316_316444


namespace student_arrangement_l316_316716

theorem student_arrangement (students : Fin 6 → Char) :
  (∃ r1 r2 b1 b2 bk yl : Fin 6,
    r1 ≠ r2 ∧ b1 ≠ b2 ∧ bk ≠ yl ∧ 
    students r1 = 'R' ∧ students r2 = 'R' ∧
    students b1 = 'B' ∧ students b2 = 'B' ∧
    students bk = 'K' ∧ students yl = 'Y' ∧ 
    (∀ i j : Fin 6, (students i = students j ∧ i ≠ j) → (|i.val - j.val| ≠ 1)) ∧ 
    (|bk.val - yl.val| = 1)) →
  (set.powerset (Fin 6)).card = 96 :=
by sorry

end student_arrangement_l316_316716


namespace count_integers_in_interval_l316_316690

theorem count_integers_in_interval : 
  set.countable {x : ℤ | abs x < (5 * Real.pi)} = 31 :=
sorry

end count_integers_in_interval_l316_316690


namespace number_of_special_integers_l316_316315

open Nat Set

-- Definitions aligned with the conditions
def is_sum_of_k_even_integers (M k m : ℕ) : Prop :=
  M = k * (2 * m + k - 1)

def has_exactly_three_k_factors (M : ℕ) : Prop :=
  ∃ (s : Finset ℕ), s.card = 3 ∧ ∀ k ∈ s, k ≥ 1 ∧ k * (k - 1) < M

-- Mathematically equivalent proof problem
theorem number_of_special_integers : 
  Finset.card {M ∈ (Finset.range 500) | has_exactly_three_k_factors M} = 10 :=
by
  sorry

end number_of_special_integers_l316_316315


namespace distinct_flags_count_l316_316553

theorem distinct_flags_count :
  let colors := {red, white, blue, green, yellow}
  let stripes := (s1, s2, s3) : (colors * colors * colors)
  (s1 ≠ s2) ∧ (s2 ≠ s3) ∧ (s1 ≠ s3) →
  (5 * 4 * 3) = 60 :=
by
  sorry

end distinct_flags_count_l316_316553


namespace locus_of_M_l316_316308

theorem locus_of_M 
  (p α β x y : ℝ)
  (h_parabola : (y^2 = 2*p*x))
  (M_locus : 2*p*x^2 - β*x*y + α*y^2 - 2*p*α*x = 0) :
  ((β^2 = 8*p*α) ∨ (β^2 < 8*p*α) ∨ (β^2 > 8*p*α)) → 
  (if β^2 = 8*p*α then ∃ k : ℝ, M_locus = k*y^2 = 8*p*x else 
  if β^2 < 8*p*α then ∃ a b : ℝ, M_locus = a*x^2 + b*y^2 = 1 else 
  if β^2 > 8*p*α then ∃ h k : ℝ, M_locus = h*x^2 - k*y^2 = 1) :=
begin
  sorry
end

end locus_of_M_l316_316308


namespace derivative_at_one_l316_316272

section

variable {f : ℝ → ℝ}

-- Define the condition
def limit_condition (f : ℝ → ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ Δx, abs Δx < δ → abs ((f (1 + Δx) - f (1 - Δx)) / Δx + 6) < ε

-- State the main theorem
theorem derivative_at_one (h : limit_condition f) : deriv f 1 = -3 :=
by
  sorry

end

end derivative_at_one_l316_316272


namespace ball_hits_ground_at_correct_time_l316_316523

def initial_velocity : ℝ := 7
def initial_height : ℝ := 10

-- The height function as given by the condition
def height_function (t : ℝ) : ℝ := -4.9 * t^2 + initial_velocity * t + initial_height

-- Statement
theorem ball_hits_ground_at_correct_time :
  ∃ t : ℝ, height_function t = 0 ∧ t = 2313 / 1000 :=
by
  sorry

end ball_hits_ground_at_correct_time_l316_316523


namespace cosine_range_l316_316597

theorem cosine_range {x : ℝ} (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) (h3 : Real.cos x ≤ 1 / 2) : 
  x ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 3) :=
by
  sorry

end cosine_range_l316_316597


namespace integer_solutions_of_abs_lt_5pi_l316_316686

theorem integer_solutions_of_abs_lt_5pi : 
  let x := 5 * Real.pi in
  ∃ n : ℕ, (∀m : ℤ, abs m < x ↔ m ∈ (Icc (-(n : ℤ)) n)) ∧ n = 15 :=
by
  sorry

end integer_solutions_of_abs_lt_5pi_l316_316686


namespace crayons_total_l316_316899

theorem crayons_total (Wanda Dina Jacob: ℕ) (hW: Wanda = 62) (hD: Dina = 28) (hJ: Jacob = Dina - 2) :
  Wanda + Dina + Jacob = 116 :=
by
  sorry

end crayons_total_l316_316899


namespace max_balls_l316_316015

theorem max_balls (total_yellow total_round total_edible : ℕ) 
  (suns balls tomatoes bananas : ℕ) :
  (total_yellow = 15) →
  (total_round = 18) →
  (total_edible = 13) →
  (tomatoes + balls ≤ total_round) →
  (tomatoes + bananas ≤ total_edible) →
  (suns + balls + tomatoes + bananas = total_yellow + total_round + total_edible) →
  (∀ b, b ∈ {balls, tomatoes, bananas, suns} → b ≥ 0) →
  (tomatoes ≤ total_round) →
  balls = 18 :=
by 
  sorry

end max_balls_l316_316015


namespace find_r_l316_316245

open Real

noncomputable def cond (r : ℝ) : Prop :=
  log 49 (3 * r - 2) = -1 / 2

theorem find_r (r : ℝ) (h : cond r) : r = 5 / 7 :=
  sorry

end find_r_l316_316245


namespace rationalize_denominator_7_over_cube_root_343_l316_316799

theorem rationalize_denominator_7_over_cube_root_343 :
  (7 / Real.cbrt 343) = 1 :=
by {
  have h : Real.cbrt 343 = 7 := rfl,
  rw [h],
  norm_num,
  rw [div_self],
  norm_num,
  sorry
}

end rationalize_denominator_7_over_cube_root_343_l316_316799


namespace new_average_amount_l316_316071

theorem new_average_amount (A : ℝ) (H : A = 14) (new_amount : ℝ) (H1 : new_amount = 56) : 
  ((7 * A + new_amount) / 8) = 19.25 :=
by
  rw [H, H1]
  norm_num

end new_average_amount_l316_316071


namespace mean_of_combined_sets_l316_316087

theorem mean_of_combined_sets (mean_set1 mean_set2 : ℝ) (n1 n2 : ℕ) 
  (h1 : mean_set1 = 15) (h2 : mean_set2 = 20) (h3 : n1 = 5) (h4 : n2 = 8) :
  (n1 * mean_set1 + n2 * mean_set2) / (n1 + n2) = 235 / 13 :=
by
  sorry

end mean_of_combined_sets_l316_316087


namespace boxer_weight_on_fight_day_l316_316526

theorem boxer_weight_on_fight_day:
  ∀ (initial_weight months_to_fight weight_loss_per_month : ℕ),
  initial_weight = 97 →
  months_to_fight = 4 →
  weight_loss_per_month = 3 →
  initial_weight - (weight_loss_per_month * months_to_fight) = 85 :=
by
  intros initial_weight months_to_fight weight_loss_per_month h1 h2 h3
  calc
    initial_weight - (weight_loss_per_month * months_to_fight)
      = 97 - (3 * 4) : by rw [h1, h2, h3]
      ... = 85 : by norm_num

end boxer_weight_on_fight_day_l316_316526


namespace complex_square_eq_l316_316819

theorem complex_square_eq (a b : ℤ) (ha : a > 0) (hb : b > 0) (h : (a + b * Complex.I) ^ 2 = 7 + 24 * Complex.I) : 
  a + b * Complex.I = 4 + 3 * Complex.I :=
by
  sorry

end complex_square_eq_l316_316819


namespace sets_of_consecutive_integers_sum_21_l316_316693

theorem sets_of_consecutive_integers_sum_21: 
  (finset.filter 
    (λ 𝑠 : finset ℕ, finset.card 𝑠 ≥ 2 ∧ finset.sum 𝑠 id = 21) 
    (finset.powerset (finset.range 22))).card = 1 := 
by sorry

end sets_of_consecutive_integers_sum_21_l316_316693


namespace value_of_m_l316_316636

theorem value_of_m (m : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = x^2 - m * x + m - 1) 
  (h_eq : f 0 = f 2) : m = 2 :=
sorry

end value_of_m_l316_316636


namespace expected_value_fair_dodecahedral_die_l316_316214

-- Lean 4 statement
theorem expected_value_fair_dodecahedral_die : 
  let faces := {n : ℕ | 1 ≤ n ∧ n ≤ 12}
  let probability := 1 / 12
  (∑ i in faces, probability * i) = 6.5 :=
by
  sorry

end expected_value_fair_dodecahedral_die_l316_316214


namespace nn_gt_n1n1_l316_316429

theorem nn_gt_n1n1 (n : ℕ) (h : n > 1) : n^n > (n + 1)^(n - 1) := 
sorry

end nn_gt_n1n1_l316_316429


namespace masha_and_bear_eat_porridge_l316_316773

theorem masha_and_bear_eat_porridge:
  (∀ t b, (t = 12) ∧ (b = t / 2) → (∃ r, r = 1 / t + 1 / b) ∧ (∃ T, T = 6 / r) ∧ T = 24) :=
by
  intros t b h
  obtain ⟨ht, hb⟩ := h
  rw [hb, ht] at *
  use (1 / t + 1 / (t / 2))
  simp
  use (6 / (1 / 12 + 1 / 6))
  simp
  norm_num
  sorry

end masha_and_bear_eat_porridge_l316_316773


namespace perimeter_of_triangle_l316_316192

def trianglePerimeter : ℝ :=
  (2 + Real.sqrt 3 + (Real.sqrt 3) / 3 + Real.sqrt (5 + 2 * Real.sqrt 3))

theorem perimeter_of_triangle :
  (∃ (m : ℝ), ∃ (k : ℝ), ∃ (h k' : ℝ),
    m * 1 = -1/Real.sqrt 3 ∧
    k = 1 + Real.sqrt 3 ∧
    h = (1, 1 + Real.sqrt 3) ∧
    k' = (1, -1/Real.sqrt 3) ∧
    trianglePerimeter = 2 + Real.sqrt 3 + (Real.sqrt 3) / 3 + Real.sqrt (5 + 2 * Real.sqrt 3)
  ) :=
  sorry

end perimeter_of_triangle_l316_316192


namespace area_correct_l316_316083

noncomputable def area_of_30_60_90_triangle (hypotenuse : ℝ) (angle : ℝ) : ℝ :=
if hypotenuse = 10 ∧ angle = 30 then 25 * Real.sqrt 3 / 2 else 0

theorem area_correct {hypotenuse angle : ℝ} (h1 : hypotenuse = 10) (h2 : angle = 30) :
  area_of_30_60_90_triangle hypotenuse angle = 25 * Real.sqrt 3 / 2 :=
by
  sorry

end area_correct_l316_316083


namespace women_non_french_percentage_l316_316931

theorem women_non_french_percentage
  (total_employees men_percentage men_french_percentage total_french_percentage : ℝ)
  (h1 : total_employees = 100)
  (h2 : men_percentage = 0.65)
  (h3 : men_french_percentage = 0.60)
  (h4 : total_french_percentage = 0.40)
  : (34 / 35) * 100 ≈ 97.14 := by
  sorry

end women_non_french_percentage_l316_316931


namespace amount_C_l316_316425

theorem amount_C (A_amt B_amt C_amt : ℚ)
  (h1 : A_amt + B_amt + C_amt = 527)
  (h2 : A_amt = (2 / 3) * B_amt)
  (h3 : B_amt = (1 / 4) * C_amt) :
  C_amt = 372 :=
sorry

end amount_C_l316_316425


namespace carrots_eaten_after_dinner_l316_316731

def carrots_eaten_before_dinner : ℕ := 22
def total_carrots_eaten : ℕ := 37

theorem carrots_eaten_after_dinner : total_carrots_eaten - carrots_eaten_before_dinner = 15 := by
  sorry

end carrots_eaten_after_dinner_l316_316731


namespace angle_A_area_of_triangle_l316_316761

theorem angle_A (a b c : ℝ) (A B C : ℝ) 
  (h₁ : b * (sin B - sin C) + (c - a) * (sin A + sin C) = 0)
  (h₂ : A + B + C = π) :
  A = π / 3 :=
by sorry

theorem area_of_triangle (a c : ℝ) (A B C : ℝ) 
  (h₁ : sin C = (1 + sqrt 3) / 2 * sin B)
  (h₂ : A + B + C = π)
  (h₃ : a = sqrt 3)
  (h₄ : A = π / 3) :
  ∃ b : ℝ, 
  ∃ area : ℝ, 
  b = sqrt 2 ∧ area = (3 + sqrt 3) / 4 :=
by sorry

end angle_A_area_of_triangle_l316_316761


namespace find_f_2015_plus_f_2016_l316_316659

def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom functional_equation (x : ℝ) : f (3/2 - x) = f x
axiom value_at_minus2 : f (-2) = -3

theorem find_f_2015_plus_f_2016 : f 2015 + f 2016 = 3 := 
by {
  sorry
}

end find_f_2015_plus_f_2016_l316_316659


namespace b_days_work_alone_l316_316512

theorem b_days_work_alone 
  (W_b : ℝ)  -- Work done by B in one day
  (W_a : ℝ)  -- Work done by A in one day
  (D_b : ℝ)  -- Number of days for B to complete the work alone
  (h1 : W_a = 2 * W_b)  -- A is twice as good a workman as B
  (h2 : 7 * (W_a + W_b) = D_b * W_b)  -- A and B took 7 days together to do the work
  : D_b = 21 :=
sorry

end b_days_work_alone_l316_316512


namespace exist_interesting_u_v_l316_316625

/-- Definition of an interesting triple (a, b, c) of positive integers -/
def is_interesting_triple (a b c : ℕ) : Prop :=
  c^2 + 1 ∣ (a^2 + 1) * (b^2 + 1) ∧ ¬ (c^2 + 1 ∣ a^2 + 1) ∧ ¬ (c^2 + 1 ∣ b^2 + 1)

/-- The main theorem statement -/
theorem exist_interesting_u_v {a b c : ℕ} (h : is_interesting_triple a b c) :
  ∃ u v : ℕ, u > 0 ∧ v > 0 ∧ is_interesting_triple u v c ∧ u * v < c^3 :=
begin
  sorry
end

end exist_interesting_u_v_l316_316625


namespace max_balls_possible_l316_316021

structure Conditions :=
  (yellow_objects : ℕ)
  (round_objects : ℕ)
  (edible_objects : ℕ)
  (all_objects : set string)
  (is_round : string → Prop)
  (is_yellow : string → Prop)
  (is_edible : string → Prop)
  (is_red : string → Prop)
  (p_types : list string)

namespace Problem
def PetyaConditions : Conditions :=
  {
    yellow_objects := 15,
    round_objects := 18,
    edible_objects := 13,
    all_objects := {"sun", "ball", "tomato", "banana"},
    is_round := λ x, x = "tomato" ∨ x = "ball",
    is_yellow := λ x, x = "banana" ∨ x = "ball",
    is_edible := λ x, x = "banana" ∨ x = "tomato",
    is_red := λ x, x = "tomato",
    p_types := ["sun", "ball", "tomato", "banana"]
  }

theorem max_balls_possible (cond : Conditions)
  (h1 : cond.yellow_objects = 15)
  (h2 : cond.round_objects = 18)
  (h3 : cond.edible_objects = 13)
  (h4 : ∀ x, x ∈ cond.all_objects → (cond.is_round x → ¬cond.is_yellow x → cond.is_edible x → ¬cond.is_red x))
  : ∃ n, n = 18 :=
by {
  sorry
}

end Problem

end max_balls_possible_l316_316021


namespace maximum_triangle_area_l316_316358

-- Define Points A, B, and C
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (12, 0)
def C : ℝ × ℝ := (20, 0)

-- Define Line Definitions
def l_A (θ : ℝ) : ℝ → ℝ :=
  λ x, x * tan θ

def l_B (θ : ℝ) : ℝ := B.1

def l_C (θ : ℝ) : ℝ → ℝ :=
  λ x, (x - C.1) * tan (-θ) + C.2

-- The main hypothesis: the lines rotate at the same rate θ
-- We need to determine the intersection points and maximum area of triangle formed
noncomputable def max_area_triangle : ℝ :=
  let X := (B.1, l_C θ B.1)
  let Y := let x_intercept := (C.2 - tan θ * C.1) / (tan θ - tan (-θ)) in (x_intercept, tan θ * x_intercept)
  let Z := (B.1, tan θ * B.1) in
  let area_ABC := 1 /2 * abs((X.1 * Y.2 + Y.1 * Z.2 + Z.1 * X.2) - (X.2 * Y.1 + Y.2 * Z.1 + Z.2 * X.1)) in
  sorry

-- The statement to prove
theorem maximum_triangle_area :
  max_area_triangle = 104 :=
sorry

end maximum_triangle_area_l316_316358


namespace projection_of_a_onto_b_is_correct_l316_316313

/-- Define vectors a and b -/
def vec_a : ℝ × ℝ := (1, 1)
def vec_b : ℝ × ℝ := (0, -2)

/-- Define the dot product of two vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

/-- Define the magnitude of a vector -/
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

/-- Define the projection of a vector onto another vector -/
def projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (dot_product a b) / (magnitude b ^ 2)
  (scalar * b.1, scalar * b.2)

/-- Theorem stating the projection of vec_a onto vec_b is (0, 1) -/
theorem projection_of_a_onto_b_is_correct :
  projection vec_a vec_b = (0, 1) :=
sorry

end projection_of_a_onto_b_is_correct_l316_316313


namespace grasshopper_max_reach_points_l316_316896

theorem grasshopper_max_reach_points
  (α : ℝ) (α_eq : α = 36 * Real.pi / 180)
  (L : ℕ)
  (jump_constant : ∀ (n : ℕ), L = L) :
  ∃ (N : ℕ), N ≤ 10 :=
by 
  sorry

end grasshopper_max_reach_points_l316_316896


namespace bus_riders_count_l316_316103

-- Definitions of the problem conditions
variables (TotalStudents WalkingStudents BusRiders BikeRiders RemainingStudents : ℕ)
variables (p : Prop)

-- Setting up the conditions
def total_students_all : TotalStudents = 92 := by sorry
def walking_students : WalkingStudents = 27 := by sorry
def remaining_students : RemainingStudents = TotalStudents - BusRiders := by sorry
def bike_riders : BikeRiders = (5 / 8) * RemainingStudents := by sorry
def total_students : TotalStudents = BusRiders + BikeRiders + WalkingStudents := by sorry

-- Statement to prove
theorem bus_riders_count : (BusRiders = 20) :=
by
  -- Importing required values and conditions
  have h1 : TotalStudents = 92 := total_students_all,
  have h2 : WalkingStudents = 27 := walking_students,
  have h3 : RemainingStudents = TotalStudents - BusRiders := remaining_students,
  have h4 : BikeRiders = (5 / 8) * RemainingStudents := bike_riders,
  have h5 : TotalStudents = BusRiders + BikeRiders + WalkingStudents := total_students,
  -- Skipping the proof steps
  sorry

end bus_riders_count_l316_316103


namespace max_balls_drawn_l316_316045

-- Defining the concepts of objects Petya can draw
inductive Object
| Sun
| Ball
| Tomato
| Banana

-- Defining properties for objects
def isYellow (o : Object) : Bool :=
  match o with
  | Object.Banana => true
  | _ => false

def isRound (o : Object) : Bool :=
  match o with
  | Object.Ball   => true
  | Object.Tomato => true
  | _             => false

def isEdible (o : Object) : Bool :=
  match o with
  | Object.Tomato => true
  | Object.Banana => true
  | _             => false

def countObjects (p : Object -> Bool) (os : List Object) : Nat :=
  os.countp p

theorem max_balls_drawn (os : List Object) :
  countObjects isYellow os = 15 →
  countObjects isRound os = 18 →
  countObjects isEdible os = 13 →
  countObjects (λ o => o = Object.Ball) os = 18 :=
by
  intros hy hr he
  -- Proof will go here
  sorry

end max_balls_drawn_l316_316045


namespace odd_number_adjacent_product_diff_l316_316211

variable (x : ℤ)

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem odd_number_adjacent_product_diff (h : is_odd x)
  (adjacent_diff : x * (x + 2) - x * (x - 2) = 44) : x = 11 :=
by
  sorry

end odd_number_adjacent_product_diff_l316_316211


namespace percent_increase_in_perimeter_l316_316970

theorem percent_increase_in_perimeter :
  let side_length_first : ℝ := 3
  let scaling_factor : ℝ := 1.25
  let fourth_triangle_side_length := scaling_factor^3 * side_length_first
  let first_triangle_perimeter := 3 * side_length_first
  let fourth_triangle_perimeter := 3 * fourth_triangle_side_length
  let percent_increase := ((fourth_triangle_perimeter - first_triangle_perimeter) / first_triangle_perimeter) * 100
  percent_increase = 95.3 :=
by
  -- Definitions based on the conditions
  let side_length_first := 3 : ℝ
  let scaling_factor := 1.25 : ℝ
  let fourth_triangle_side_length := scaling_factor^3 * side_length_first
  let first_triangle_perimeter := 3 * side_length_first
  let fourth_triangle_perimeter := 3 * fourth_triangle_side_length
  let percent_increase := ((fourth_triangle_perimeter - first_triangle_perimeter) / first_triangle_perimeter) * 100
  -- Prove the statement
  sorry

end percent_increase_in_perimeter_l316_316970


namespace range_of_lambda_l316_316673

theorem range_of_lambda (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (λ : ℝ) :
  (∀ n : ℕ, n > 0 → S_n n = 2 * a_n n - 2^(n + 1)) →
  (∀ n : ℕ, n > 0 → (-1)^n * λ < S_n n / S_n (n + 1)) →
  -1/4 < λ ∧ λ < 1/3 :=
by
  intros h_sum h_ineq
  sorry

end range_of_lambda_l316_316673


namespace parallelogram_area_l316_316926

theorem parallelogram_area (AB BD BE BF: ℝ) (angle_A : ℝ):
  AB = 2 ∧ angle_A = π / 4 ∧ 
  ∃ E F, (∠ AEB = π / 2 ∧ ∠ CFD = π / 2) ∧ (BF = 3 / 2 * BE) → 
  let AD := 2 in let BK := (AB * real.sin(angle_A)).sqrt 
  in (AD * BK).sqrt = 3 :=
begin
  sorry
end

end parallelogram_area_l316_316926


namespace necessary_conditions_for_propositions_l316_316964

variable {x y a b c : ℝ}

theorem necessary_conditions_for_propositions :
  (∀ x y : ℝ, ((x > 10) → (x > 5))) ∧
  (∀ a b c : ℝ, (c ≠ 0 → ((ac = bc) → (a = b)))) ∧
  (∀ x y : ℝ, ((2x + 1 = 2y + 1) → (x = y))) :=
by
  sorry

end necessary_conditions_for_propositions_l316_316964


namespace students_dont_eat_lunch_l316_316543

theorem students_dont_eat_lunch
  (total_students : ℕ)
  (students_in_cafeteria : ℕ)
  (students_bring_lunch : ℕ)
  (students_no_lunch : ℕ)
  (h1 : total_students = 60)
  (h2 : students_in_cafeteria = 10)
  (h3 : students_bring_lunch = 3 * students_in_cafeteria)
  (h4 : students_no_lunch = total_students - (students_in_cafeteria + students_bring_lunch)) :
  students_no_lunch = 20 :=
by
  sorry

end students_dont_eat_lunch_l316_316543


namespace find_g_neg_6_l316_316855

noncomputable def g : ℤ → ℤ := sorry

-- Conditions from the problem
axiom g_condition_1 : g 1 - 1 > 0
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The statement we need to prove
theorem find_g_neg_6 : g (-6) = 723 :=
by { sorry }

end find_g_neg_6_l316_316855


namespace acute_angled_triangle_exists_l316_316643

theorem acute_angled_triangle_exists (a b c d e : ℝ) 
  (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : d ≤ e)
  (h5 : ∀ x y z : ℝ, x ∈ {a, b, c, d, e} ∧ y ∈ {a, b, c, d, e} ∧ z ∈ {a, b, c, d, e} ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z → x + y > z) :
  ∃ x y z : ℝ, x ∈ {a, b, c, d, e} ∧ y ∈ {a, b, c, d, e} ∧ z ∈ {a, b, c, d, e} ∧ x ≤ y ∧ y ≤ z ∧ x^2 + y^2 > z^2 :=
by
  sorry

end acute_angled_triangle_exists_l316_316643


namespace roots_are_real_and_positive_l316_316745

noncomputable def polynomial := (x : ℝ) → x * (x - 2) * (3 * x - 7) - 2

def positive_real (x : ℝ) := x > 0

theorem roots_are_real_and_positive 
  (r s t : ℝ)
  (hroots : polynomial r = 0 ∧ polynomial s = 0 ∧ polynomial t = 0)
  (hdistinct : r ≠ s ∧ s ≠ t ∧ t ≠ r)
  (hpositive : positive_real r ∧ positive_real s ∧ positive_real t) :
  (r + s + t = 13 / 3) ∧ 
  (r * s + s * t + t * r = 14 / 3) ∧ 
  (r * s * t = 2 / 3) ∧ 
  (arctan r + arctan s + arctan t = 3 * Real.pi / 4) :=
sorry

end roots_are_real_and_positive_l316_316745


namespace time_to_drink_whole_bottle_l316_316770

theorem time_to_drink_whole_bottle :
  ∀ (total_volume first_hour_sip_volume second_hour_sip_volume third_hour_sip_volume sip_interval minutes_in_an_hour break_time : ℕ),
    total_volume = 2000 →
    first_hour_sip_volume = 40 →
    second_hour_sip_volume = 50 →
    third_hour_sip_volume = 60 →
    sip_interval = 5 →
    minutes_in_an_hour = 60 →
    break_time = 30 →
    let sips_per_hour := minutes_in_an_hour / sip_interval in
    let first_hour_drink := sips_per_hour * first_hour_sip_volume in
    let second_hour_drink := sips_per_hour * second_hour_sip_volume in
    let third_hour_drink := sips_per_hour * third_hour_sip_volume in
    let total_drink_in_three_hours := first_hour_drink + second_hour_drink + third_hour_drink in
    let remaining_drink := total_volume - total_drink_in_three_hours in
    let additional_sips := (remaining_drink + third_hour_sip_volume - 1) / third_hour_sip_volume in
    let additional_time := additional_sips * sip_interval in
    let total_drink_time := (3 * minutes_in_an_hour) + additional_time in
    (total_drink_time + break_time) = 230 :=
by {
  intros total_volume first_hour_sip_volume second_hour_sip_volume third_hour_sip_volume sip_interval minutes_in_an_hour break_time,
  assume h1 h2 h3 h4 h5 h6 h7,
  let sips_per_hour := minutes_in_an_hour / sip_interval,
  let first_hour_drink := sips_per_hour * first_hour_sip_volume,
  let second_hour_drink := sips_per_hour * second_hour_sip_volume,
  let third_hour_drink := sips_per_hour * third_hour_sip_volume,
  let total_drink_in_three_hours := first_hour_drink + second_hour_drink + third_hour_drink,
  let remaining_drink := total_volume - total_drink_in_three_hours,
  let additional_sips := (remaining_drink + third_hour_sip_volume - 1) / third_hour_sip_volume,
  let additional_time := additional_sips * sip_interval,
  let total_drink_time := (3 * minutes_in_an_hour) + additional_time,
  have : (total_drink_time + break_time) = 230 := sorry,
  exact this
}

end time_to_drink_whole_bottle_l316_316770


namespace required_homework_assignments_l316_316946

noncomputable def assignmentsPerPoint (n : ℕ) : ℕ :=
  Nat.ceil ((n / 5 : ℚ) + 1)

theorem required_homework_assignments : 
  (∑ n in List.range 1 21, assignmentsPerPoint n) = 40 := by
  sorry

end required_homework_assignments_l316_316946


namespace inequality_holds_for_all_x_y_l316_316812

theorem inequality_holds_for_all_x_y (x y : ℝ) : 
  x^2 + y^2 + 1 ≥ x + y + x * y := 
by sorry

end inequality_holds_for_all_x_y_l316_316812


namespace find_b8_l316_316752

def sequence (b : ℕ → ℕ) : Prop :=
  b 1 = 2 ∧ (∀ m n : ℕ, 0 < m → 0 < n → b (m + n) = b m + b n + m ^ n)

theorem find_b8 (b : ℕ → ℕ) (h : sequence b) : b 8 = 284 := 
sorry

end find_b8_l316_316752


namespace find_a_l316_316076

noncomputable def eq_center_of_circle (a : ℝ) : Prop :=
  let cx := 1
  let cy := 4
  let dist_to_line := abs (a * cx + cy - 1) / sqrt (a^2 + 1)
  dist_to_line = 1

theorem find_a : ∃ (a : ℝ), eq_center_of_circle a ∧ a = -4/3 := by
  sorry

end find_a_l316_316076


namespace quadratic_root_difference_l316_316595

noncomputable def roots_diff (a b c : ℝ) : ℝ :=
  let Δ := b^2 - 4 * a * c in
  (-b + real.sqrt Δ) / (2 * a) - (-b - real.sqrt Δ) / (2 * a)

theorem quadratic_root_difference : 
  let a := 5 + 3 * real.sqrt 2
  let b := 3 + real.sqrt 2
  let c := -1
  roots_diff a b c = 2 - real.sqrt 2 :=
sorry

end quadratic_root_difference_l316_316595


namespace odd_integers_in_range_l316_316692

theorem odd_integers_in_range (x : ℤ) : 
  4 = (set.count {x | (41 / 8 : ℝ) < x ∧ x < (57 / 4 : ℝ) ∧ (x % 2 = 1)}).to_nat :=
by
  sorry

end odd_integers_in_range_l316_316692


namespace total_crayons_l316_316900

theorem total_crayons (Wanda Dina Jacob : Nat) (hWanda : Wanda = 62) (hDina : Dina = 28) (hJacob : Jacob = Dina - 2) :
  Wanda + Dina + Jacob = 116 :=
by
  -- We first use the given conditions to substitute the values
  rw [hWanda, hDina, hJacob]
  -- Simplify the expression to verify the result is as expected
  rw [Nat.succ_sub, Nat.sub_self, Nat.add_comm, Nat.add_assoc]
  norm_num
  sorry

end total_crayons_l316_316900


namespace manager_salary_l316_316516

theorem manager_salary (avg_salary_50 : ℕ) (num_employees : ℕ) (increment_new_avg : ℕ)
  (new_avg_salary : ℕ) (total_old_salary : ℕ) (total_new_salary : ℕ) (M : ℕ) :
  avg_salary_50 = 2000 →
  num_employees = 50 →
  increment_new_avg = 250 →
  new_avg_salary = avg_salary_50 + increment_new_avg →
  total_old_salary = num_employees * avg_salary_50 →
  total_new_salary = (num_employees + 1) * new_avg_salary →
  M = total_new_salary - total_old_salary →
  M = 14750 :=
by {
  sorry
}

end manager_salary_l316_316516


namespace shaded_area_l316_316878

theorem shaded_area (D : ℝ) (k : ℝ) (π : ℝ) (hD : D = 1) (hπ : π = real.pi) :
  (∃ k : ℝ, 0 < k ∧ ∃ A : ℝ, A = (1/8) * k * π ∧ A = (121/8) * π) :=
by
  use 121
  sorry

end shaded_area_l316_316878


namespace max_balls_drawn_l316_316046

-- Defining the concepts of objects Petya can draw
inductive Object
| Sun
| Ball
| Tomato
| Banana

-- Defining properties for objects
def isYellow (o : Object) : Bool :=
  match o with
  | Object.Banana => true
  | _ => false

def isRound (o : Object) : Bool :=
  match o with
  | Object.Ball   => true
  | Object.Tomato => true
  | _             => false

def isEdible (o : Object) : Bool :=
  match o with
  | Object.Tomato => true
  | Object.Banana => true
  | _             => false

def countObjects (p : Object -> Bool) (os : List Object) : Nat :=
  os.countp p

theorem max_balls_drawn (os : List Object) :
  countObjects isYellow os = 15 →
  countObjects isRound os = 18 →
  countObjects isEdible os = 13 →
  countObjects (λ o => o = Object.Ball) os = 18 :=
by
  intros hy hr he
  -- Proof will go here
  sorry

end max_balls_drawn_l316_316046


namespace number_of_boys_l316_316784

theorem number_of_boys
  (total_children happy_children sad_children: ℕ)
  (number_girls happy_boys sad_girls: ℕ) :
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  number_girls = 42 →
  happy_boys = 6 →
  sad_girls = 4 →
  ∃ (B : ℕ), B + number_girls = total_children ∧ B = 18 :=
by {
  intros h1 h2 h3 h4 h5 h6,
  use 18,
  split,
  {
    rw [h4, h1],
    exact (by norm_num : 18 + 42 = 60),
  },
  {
    exact (by norm_num : 18 = 18),
  },
  sorry
}

end number_of_boys_l316_316784


namespace g_neg_six_eq_neg_twenty_l316_316850

theorem g_neg_six_eq_neg_twenty (g : ℤ → ℤ)
    (h1 : g 1 - 1 > 0)
    (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
    (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 := 
sorry

end g_neg_six_eq_neg_twenty_l316_316850


namespace inequality_range_of_a_l316_316084

theorem inequality_range_of_a (a : ℝ) :
  (∀ x y : ℝ, (1 ≤ x ∧ x ≤ 2) ∧ (1 ≤ y ∧ y ≤ 3) → 2 * x^2 - a * x * y + y^2 ≥ 0) →
  a ≤ 2 * Real.sqrt 2 :=
by
  intros h
  sorry

end inequality_range_of_a_l316_316084


namespace isosceles_triangle_perimeter_l316_316344

def is_isosceles_triangle (a b c : ℝ) : Prop := (a = b ∨ b = c ∨ a = c) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 2) (h2 : b = 5) :
  ∃ c, is_isosceles_triangle a b c ∧ a + b + c = 12 :=
by {
  use 5,
  split,
  simp [is_isosceles_triangle, h1, h2],
  split,
  linarith,
  split,
  linarith,
  linarith,
  ring,
}

end isosceles_triangle_perimeter_l316_316344


namespace length_of_MN_l316_316645

-- Define points M and N with their coordinates
def M : ℝ × ℝ := (1, 1)
def N : ℝ × ℝ := (4, 5)

-- Distance formula
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Statement of the theorem
theorem length_of_MN : distance M N = 5 := by
  sorry

end length_of_MN_l316_316645


namespace gcd_lcm_45_75_l316_316145

theorem gcd_lcm_45_75 : gcd 45 75 = 15 ∧ lcm 45 75 = 1125 :=
by sorry

end gcd_lcm_45_75_l316_316145


namespace supplements_of_congruent_angles_are_congruent_l316_316241

-- Define the concept of supplementary angles
def is_supplementary (α β : ℝ) : Prop := α + β = 180

-- Statement of the problem
theorem supplements_of_congruent_angles_are_congruent :
  ∀ {α β γ δ : ℝ},
  is_supplementary α β →
  is_supplementary γ δ →
  β = δ →
  α = γ :=
by
  intros α β γ δ h1 h2 h3
  sorry

end supplements_of_congruent_angles_are_congruent_l316_316241


namespace equal_number_of_boys_and_girls_l316_316480

theorem equal_number_of_boys_and_girls (total_individuals boys girls : ℕ) (h_total: total_individuals = 20) (h_boys: boys = 10) (h_girls: girls = 10) :
  let number_of_ways := Nat.choose 20 10 in
  number_of_ways = 184756 :=
by
  intros
  rw [h_total, h_boys, h_girls]
  show Nat.choose 20 10 = 184756
  sorry

end equal_number_of_boys_and_girls_l316_316480


namespace angle_between_plane_and_base_l316_316828

-- Definitions based on conditions
def regular_hexagonal_pyramid (G: Point) (vertices: Fin 6 → Point) : Prop :=
(∀ i, ∠(G, vertices i, vertices ((i+1) % 6)) = 45)

def plane_intersects_parallel_segments (S: Plane) (base_edge: Line) (vertices: Fin 6 → Point) : Prop :=
(∃ BP AQ : Line, 
  BP ∥ AQ ∧ 
  BP ∥ base_edge ∧ 
  AQ ∥ base_edge ∧
  OnPlane S BP ∧ 
  OnPlane S AQ)

-- Theorem statement combining condition and conclusion
theorem angle_between_plane_and_base (G: Point) (vertices: Fin 6 → Point) (S: Plane) (base_edge: Line) :
  regular_hexagonal_pyramid G vertices →
  plane_intersects_parallel_segments S base_edge vertices →
  ∠(S, base (G, vertices)) = arctan (1 / 2) :=
by
  sorry

end angle_between_plane_and_base_l316_316828


namespace eq_of_abs_inequality_solution_max_of_sqrt_expression_l316_316304

-- Part 1: Prove that a = 3 given the inequality condition
theorem eq_of_abs_inequality_solution (a : ℝ) (h : ∀ x : ℝ, |a * x - 2| < 6 ↔ -4 / 3 < x ∧ x < 8 / 3) : a = 3 := 
sorry

-- Part 2: Prove the maximum value of the given expression
theorem max_of_sqrt_expression (a b t : ℝ) (h1 : a = 3) (h2 : b = 1) : 
  ∀ t : ℝ, 
  (sqrt (-a * t + 12) + sqrt (3 * b * t) ≤ 2 * sqrt 6) ∧ 
  (sqrt (-a * t + 12) + sqrt (3 * b * t) = 2 * sqrt 6 ↔ t = 2) :=
sorry

end eq_of_abs_inequality_solution_max_of_sqrt_expression_l316_316304


namespace find_g_minus_6_l316_316848

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Conditions given in the problem
axiom cond1 : g 1 - 1 > 0
axiom cond2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom cond3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The proof we want to make (assertion)
theorem find_g_minus_6 : g (-6) = 723 := 
sorry

end find_g_minus_6_l316_316848


namespace recommendation_plans_l316_316360

theorem recommendation_plans (students universities : ℕ) (max_students : universities -> ℕ) 
  (h_students : students = 4) (h_universities : universities = 3) (h_max_students : ∀ (u : ℕ), u < universities → max_students u = 2) :
  let plans := (4.choose 2 * (3.choose 3).permute) + (3.choose 2 * 4.choose 2) in
  plans = 54 :=
by
  sorry

end recommendation_plans_l316_316360


namespace team_B_at_least_half_can_serve_l316_316120

-- Define the height limit condition
def height_limit (h : ℕ) : Prop := h ≤ 168

-- Define the team conditions
def team_A_avg_height : Prop := (160 + 169 + 169) / 3 = 166

def team_B_median_height (B : List ℕ) : Prop :=
  B.length % 2 = 1 ∧ B.perm ([167] ++ (B.eraseNth (B.length / 2))) ∧ B.nth (B.length / 2) = some 167

def team_C_tallest_height (C : List ℕ) : Prop :=
  ∀ (h : ℕ), h ∈ C → h ≤ 169

def team_D_mode_height (D : List ℕ) : Prop :=
  ∃ k, ∀ (h : ℕ), h ≠ 167 ∨ D.count 167 ≥ D.count h

-- Declare the main theorem to be proven
theorem team_B_at_least_half_can_serve (B : List ℕ) :
  (∀ h, h ∈ B → height_limit h) ↔ team_B_median_height B := sorry

end team_B_at_least_half_can_serve_l316_316120


namespace smallest_q_l316_316808

theorem smallest_q (p q : ℕ) (h : ∑ i in range 7, if i < 6 then i + 1 else 0 = 21) (mean : (21 + 5 * p + 7 * q) / (6 + p + q) = 5.3) : q = 9 :=
sorry

end smallest_q_l316_316808


namespace initial_speed_is_100_l316_316194

-- Given the conditions as definitions
variables (D T : ℝ) -- Total distance and total time
variable (S : ℝ) -- Initial speed

-- The equations derived from the conditions in the problem
def speed_for_first_part (D T : ℝ) : ℝ := (2 * D) / T
def remaining_distance (D : ℝ) : ℝ := D / 3
def remaining_speed : ℝ := 25
def time_for_second_part (D : ℝ) : ℝ := (D / 3) / 25
def total_time_condition (T : ℝ) (time_for_second_part D : ℝ) : Prop := (T / 3) + time_for_second_part D = T

-- The initial speed is 100 kmph
theorem initial_speed_is_100 (D T : ℝ) (S : ℝ) :
  speed_for_first_part D T = S ↔
  time_for_second_part D = D / 75 ∧
  total_time_condition T (time_for_second_part D) →
  S = 100 :=
by 
  intro h1 h2,
  sorry

end initial_speed_is_100_l316_316194


namespace find_area_of_triangle_abe_l316_316717

open Real

-- Definitions to set up the problem
def parallelogram (A B C D : Type*) [has_add A] [has_mul B] : Prop :=
sorry -- Add formal definition of parallelogram

def angle_bisector (A B C : Type*) [has_distrib_mul_action A B C] (angle : ℝ) : Prop :=
sorry -- Add formal definition of angle bisector

noncomputable def triangle_area (A B C : Type*) [has_add A] [has_mul B] : ℝ :=
sorry -- Add formula for triangle area

def sin_120 : ℝ := sqrt 3 / 2

-- Statement of the problem as a theorem
theorem find_area_of_triangle_abe
  (A B C D E : Type*) [has_add A] [has_mul B]
  (h_para : parallelogram A B C D)
  (h_angle_bad : ∠BAD = 60)
  (h_ab_length : AB = 3)
  (h_angle_bisector_A : angle_bisector A E C 60) :
  triangle_area A B E = (9 * sqrt 3) / 4 :=
by
  sorry

end find_area_of_triangle_abe_l316_316717


namespace div_by_3_iff_n_form_l316_316607

theorem div_by_3_iff_n_form (n : ℕ) : (3 ∣ (n * 2^n + 1)) ↔ (∃ k : ℕ, n = 6 * k + 1 ∨ n = 6 * k + 2) :=
by
  sorry

end div_by_3_iff_n_form_l316_316607


namespace g_minus_6_eq_neg_20_l316_316841

noncomputable def g : ℤ → ℤ := sorry

axiom condition1 : g 1 - 1 > 0
axiom condition2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom condition3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

theorem g_minus_6_eq_neg_20 : g (-6) = -20 := sorry

end g_minus_6_eq_neg_20_l316_316841


namespace largest_angle_in_triangle_l316_316867

theorem largest_angle_in_triangle (x : ℝ) (h1 : 40 + 60 + x = 180) (h2 : max 40 60 ≤ x) : x = 80 :=
by
  -- Proof skipped
  sorry

end largest_angle_in_triangle_l316_316867


namespace expression_evaluation_l316_316217

theorem expression_evaluation :
  (3.14 - Real.pi : ℝ) + (-1 / 2 : ℝ) ^ (-2 : ℤ) + Real.abs (1 - Real.sqrt 8) - 4 * Real.cos (Real.pi / 4) = 4 := 
by
  -- This lets Lean know that we'll provide a proof later
  sorry

end expression_evaluation_l316_316217


namespace prob_8th_roll_last_l316_316699

-- Define the conditions as functions or constants
def prob_diff_rolls : ℚ := 5/6
def prob_same_roll : ℚ := 1/6

-- Define the theorem stating the probability of the 8th roll being the last roll
theorem prob_8th_roll_last : (1 : ℚ) * prob_diff_rolls^6 * prob_same_roll = 15625 / 279936 := 
sorry

end prob_8th_roll_last_l316_316699


namespace geometric_sequence_ninth_term_l316_316451

-- Given conditions
variables (a r : ℝ)
axiom fifth_term_condition : a * r^4 = 80
axiom seventh_term_condition : a * r^6 = 320

-- Goal: Prove that the ninth term is 1280
theorem geometric_sequence_ninth_term : a * r^8 = 1280 :=
by
  sorry

end geometric_sequence_ninth_term_l316_316451


namespace discriminant_positive_l316_316620

theorem discriminant_positive
  (a b c : ℝ)
  (h : (a + b + c) * c < 0) : b^2 - 4 * a * c > 0 :=
sorry

end discriminant_positive_l316_316620


namespace p_true_when_a_eq_1_p_and_q_implies_a_in_range_l316_316417

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + a * x

def p (a : ℝ) : Prop :=
 ∀ x, (3 * x^2 + 2 * a * x + a) ≥ 0

def q (a : ℝ) : Prop :=
(a + 2) * (a - 2) < 0

theorem p_true_when_a_eq_1 : p 1 :=
by
  -- Proof needed here
  sorry

theorem p_and_q_implies_a_in_range {a : ℝ} : p a ∧ q a → a ∈ set.Ico 0 2 :=
by
  -- Proof needed here
  sorry

end p_true_when_a_eq_1_p_and_q_implies_a_in_range_l316_316417


namespace sum_of_powers_eq_one_l316_316999

noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 17)

theorem sum_of_powers_eq_one : (∑ k in Finset.range 1 17, ω ^ k) = 1 :=
by {
  have h : ω ^ 17 = 1 := by {
    rw [ω, Complex.exp_nat_mul, Complex.mul_div_cancel' (2 * Real.pi * Complex.I) (show (17 : ℂ) ≠ 0, by norm_cast; norm_num)],
    rw Complex.exp_cycle,
  },
  sorry
}

end sum_of_powers_eq_one_l316_316999


namespace cruzs_marbles_l316_316109

theorem cruzs_marbles (Atticus Jensen Cruz : ℕ) 
  (h1 : 3 * (Atticus + Jensen + Cruz) = 60) 
  (h2 : Atticus = Jensen / 2) 
  (h3 : Atticus = 4) : 
  Cruz = 8 := 
sorry

end cruzs_marbles_l316_316109


namespace volume_of_material_removed_l316_316522

-- Definitions corresponding to the conditions
def cube_side : ℝ := 2
def cut_depth : ℝ := cube_side / 2
def remaining_square_side : ℝ := 1

-- Derived dimensions of each cut-out piece
def cut_length : ℝ := (cube_side - remaining_square_side) / 2
def cut_width : ℝ := (cube_side - remaining_square_side) / 2

-- Volume of one cut-out piece
def cut_volume : ℝ := cut_length * cut_width * cut_depth

-- Total volume of material removed
def total_removed_volume : ℝ := 8 * cut_volume

-- Theorem to prove the total volume removed is 2 cubic units
theorem volume_of_material_removed : total_removed_volume = 2 :=
by
  -- Proof not provided, replace with sorry
  sorry

end volume_of_material_removed_l316_316522


namespace eventually_constant_sequence_a_floor_l316_316675

noncomputable def sequence_a (n : ℕ) : ℝ := sorry
noncomputable def sequence_b (n : ℕ) : ℝ := sorry

axiom base_conditions : 
  (sequence_a 1 = 1) ∧
  (sequence_b 1 = 2) ∧
  (∀ n, sequence_a (n + 1) * sequence_b n = 1 + sequence_a n + sequence_a n * sequence_b n) ∧
  (∀ n, sequence_b (n + 1) * sequence_a n = 1 + sequence_b n + sequence_a n * sequence_b n)

theorem eventually_constant_sequence_a_floor:
  (∃ N, ∀ n ≥ N, 4 < sequence_a n ∧ sequence_a n < 5) →
  (∃ N, ∀ n ≥ N, Int.floor (sequence_a n) = 4) :=
sorry

end eventually_constant_sequence_a_floor_l316_316675


namespace locus_of_points_is_ray_l316_316416

open EuclideanGeometry

/-- Description of the geometric problem -/
def perpendicular_intersection (A B C M : Point) : Prop :=
  ∃ perpendicular_to_CA : Line, ∃ perpendicular_to_CB : Line, 
    perpendicular_to_CA ∩ CA = {A} ∧ 
    perpendicular_to_CB ∩ CB = {B} ∧ 
    perpendicular_to_CA ∩ perpendicular_to_CB = {M} ∧ 
    parallel AB CA ∧ 
    parallel AB CB

/-- Theorem stating the geometric locus of points M forms a ray -/
theorem locus_of_points_is_ray (C : Point) (A B : Point) :
  perpendicular_intersection A B C M → is_ray_starting_at M C :=
sorry

end locus_of_points_is_ray_l316_316416


namespace max_consecutive_integers_sum_45_l316_316909

theorem max_consecutive_integers_sum_45 :
  ∀ (n a : ℤ), (∑ i in Finset.range n, (a + i)) = 45 → 1 ≤ n ∧ n ∣ 90 → n = 90 := 
sorry

end max_consecutive_integers_sum_45_l316_316909


namespace num_integers_abs_x_lt_5pi_l316_316683

theorem num_integers_abs_x_lt_5pi : 
    (finset.card {x : ℤ | abs x < 5 * real.pi} = 31) := 
    sorry

end num_integers_abs_x_lt_5pi_l316_316683


namespace largest_multiple_of_45_with_9_and_0_digits_l316_316612

-- Define the problem conditions in Lean
def is_valid_m (m : ℕ) : Prop :=
  (m % 45 = 0) ∧ (∀ (d ∈ m.digits 10), d = 9 ∨ d = 0)

-- Define correctness statement to be proved
theorem largest_multiple_of_45_with_9_and_0_digits :
  (∃ (m : ℕ), is_valid_m m ∧ m = 99990) ∧ (99990 / 45 = 2222) :=
by sorry

end largest_multiple_of_45_with_9_and_0_digits_l316_316612


namespace julian_height_l316_316378

theorem julian_height (tree_height shadow1 : ℝ) (statue_height shadow2 julian_shadow : ℝ) 
  (h_tree : tree_height = 50) (h_shadow1 : shadow1 = 25)
  (h_statue_height : statue_height = 30) (h_shadow2 : shadow2 = 10) 
  (h_julian_shadow : julian_shadow = 20) :
  (julian_shadow * (tree_height / shadow1)) = 40 :=
by
  rw [h_tree, h_shadow1, h_julian_shadow]
  norm_num
  -- The proof steps would go here
  sorry

end julian_height_l316_316378


namespace average_vegetables_per_week_l316_316495

theorem average_vegetables_per_week (P Vp S W : ℕ) (h1 : P = 200) (h2 : Vp = 2) (h3 : S = 25) (h4 : W = 2) :
  (P / Vp) / S / W = 2 :=
by
  sorry

end average_vegetables_per_week_l316_316495


namespace cosine_value_l316_316649

theorem cosine_value (α : ℝ) (h1 : sin (α + π / 3) + sin α = -4 * sqrt 3 / 5) 
                              (h2 : -π / 2 < α ∧ α < 0) :
  cos (α + 2 * π / 3) = 4 / 5 :=
by sorry

end cosine_value_l316_316649


namespace simplify_fraction_expression_l316_316984

theorem simplify_fraction_expression : 
  (18 / 42 - 3 / 8 - 1 / 12 : ℚ) = -5 / 168 :=
by
  sorry

end simplify_fraction_expression_l316_316984


namespace crow_eating_time_l316_316162

theorem crow_eating_time (n : ℕ) (h : ∀ t : ℕ, t = (n / 5) → t = 4) : (4 + (4 / 5) = 4.8) :=
by
  sorry

end crow_eating_time_l316_316162


namespace inequality_condition_l316_316617

theorem inequality_condition (m : ℝ) :
  (∀ x : ℝ, (x^2 - 8 * x + 20) / (m * x^2 + 2 * (m + 1) * x + 9 * m + 4) < 0) ↔ m ∈ Iio (-1/2) :=
by sorry

end inequality_condition_l316_316617


namespace power_of_2_l316_316246

theorem power_of_2 (n : ℕ) (h1 : n ≥ 1) (h2 : ∃ m : ℕ, m ≥ 1 ∧ (2^n - 1) ∣ (m^2 + 9)) : ∃ k : ℕ, n = 2^k := 
sorry

end power_of_2_l316_316246


namespace isosceles_trapezoid_point_conditions_l316_316581

noncomputable def isosceles_trapezoid_point (a b h : ℝ) : ℝ :=
  if h^2 > a*b then (h + sqrt(h^2 - a*b)) / 2
  else if h^2 = a*b then h / 2
  else sorry -- No real solution case

theorem isosceles_trapezoid_point_conditions (a b h : ℝ) :
  (h^2 > a * b → ∃ P, P = (h + sqrt(h^2 - a * b)) / 2 ∨ P = (h - sqrt(h^2 - a * b)) / 2) ∧
  (h^2 = a * b → ∃! P, P = h / 2) ∧
  (h^2 < a * b → ¬ ∃ P, true) :=
by
  -- Insert detailed proof here
  sorry

end isosceles_trapezoid_point_conditions_l316_316581


namespace g_minus_6_eq_neg_20_l316_316840

noncomputable def g : ℤ → ℤ := sorry

axiom condition1 : g 1 - 1 > 0
axiom condition2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom condition3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

theorem g_minus_6_eq_neg_20 : g (-6) = -20 := sorry

end g_minus_6_eq_neg_20_l316_316840


namespace minimum_value_of_quad_func_l316_316672

def quad_func (x : ℝ) : ℝ :=
  2 * x^2 - 8 * x + 15

theorem minimum_value_of_quad_func :
  (∀ x : ℝ, quad_func 2 ≤ quad_func x) ∧ (quad_func 2 = 7) :=
by
  -- sorry to skip proof
  sorry

end minimum_value_of_quad_func_l316_316672


namespace least_positive_integer_satisfying_congruences_l316_316908

theorem least_positive_integer_satisfying_congruences :
  ∃ b : ℕ, b > 0 ∧
    (b % 6 = 5) ∧
    (b % 7 = 6) ∧
    (b % 8 = 7) ∧
    (b % 9 = 8) ∧
    ∀ n : ℕ, (n > 0 → (n % 6 = 5) ∧ (n % 7 = 6) ∧ (n % 8 = 7) ∧ (n % 9 = 8) → n ≥ b) ∧
    b = 503 :=
by
  sorry

end least_positive_integer_satisfying_congruences_l316_316908


namespace choose_best_route_l316_316558

theorem choose_best_route
  (stoplights_RouteA : ℕ = 3)
  (traffic_RouteA : String = "moderate")
  (weather_RouteA : String = "light rain")
  (roadCond_RouteA : String = "good")
  (baseTime_RouteA : ℕ = 10)
  (extraTime_RouteA_per_light : ℕ = 3)
  
  (stoplights_RouteB : ℕ = 4)
  (traffic_RouteB : String = "high due to sporting event")
  (weather_RouteB : String = "clear")
  (roadCond_RouteB : String = "pothole")
  (baseTime_RouteB : ℕ = 12)
  (extraTime_RouteB_per_light : ℕ = 2)
  
  (stoplights_RouteC : ℕ = 2)
  (traffic_RouteC : String = "low")
  (weather_RouteC : String = "clear")
  (roadCond_RouteC : String = "good but under construction")
  (baseTime_RouteC : ℕ = 11)
  (extraTime_RouteC_per_light : ℕ = 4)
  
  (stoplights_RouteD : ℕ = 0)
  (traffic_RouteD : String = "medium")
  (weather_RouteD : String = "fog")
  (baseTime_RouteD : ℕ = 14) :
  
  (min (baseTime_RouteA + stoplights_RouteA * extraTime_RouteA_per_light)
       (min (baseTime_RouteB + stoplights_RouteB * extraTime_RouteB_per_light)
            (min (baseTime_RouteC + stoplights_RouteC * extraTime_RouteC_per_light)
                 baseTime_RouteD))) = baseTime_RouteD :=
by
  sorry

end choose_best_route_l316_316558


namespace line_intersects_circle_l316_316298

theorem line_intersects_circle :
  ∀ (l : ℝ → ℝ) (P : ℝ × ℝ),
  P = (3, 0) →
  (∀ (x y : ℝ), x^2 + y^2 - 4 * x = 0 ↔ (x - 2) ^ 2 + y ^ 2 = 4) →
  ∃ (x y : ℝ), (x, y) ∈ set_of (λ p : ℝ × ℝ, l p.1 = p.2) ∧
  ((x - 2) ^ 2 + y ^ 2 = 4) :=
by
  intros l P hP hcirc
  sorry

end line_intersects_circle_l316_316298


namespace solve_system_equations_l316_316061

noncomputable def system_equations : Prop :=
  ∃ x y : ℝ,
    (8 * x^2 - 26 * x * y + 15 * y^2 + 116 * x - 150 * y + 360 = 0) ∧
    (8 * x^2 + 18 * x * y - 18 * y^2 + 60 * x + 45 * y + 108 = 0) ∧
    ((x = 0 ∧ y = 4) ∨ (x = -7.5 ∧ y = 1) ∨ (x = -4.5 ∧ y = 0))

theorem solve_system_equations : system_equations := 
by
  sorry

end solve_system_equations_l316_316061


namespace find_original_number_l316_316568

-- Definitions based on the conditions of the problem
def tens_digit (x : ℕ) := 2 * x
def original_number (x : ℕ) := 10 * (tens_digit x) + x
def reversed_number (x : ℕ) := 10 * x + (tens_digit x)

-- Proof statement
theorem find_original_number (x : ℕ) (h1 : original_number x - reversed_number x = 27) : original_number x = 63 := by
  sorry

end find_original_number_l316_316568


namespace gcd_lcm_sum_correct_l316_316150

def gcd_lcm_sum : ℕ :=
  let gcd_40_60 := Nat.gcd 40 60
  let lcm_20_15 := Nat.lcm 20 15
  gcd_40_60 + 2 * lcm_20_15

theorem gcd_lcm_sum_correct : gcd_lcm_sum = 140 := by
  -- Definitions based on conditions
  let gcd_40_60 := Nat.gcd 40 60
  let lcm_20_15 := Nat.lcm 20 15
  
  -- sorry to skip the proof
  sorry

end gcd_lcm_sum_correct_l316_316150


namespace total_planks_needed_l316_316266

theorem total_planks_needed (large_planks small_planks : ℕ) (h1 : large_planks = 37) (h2 : small_planks = 42) : large_planks + small_planks = 79 :=
by
  sorry

end total_planks_needed_l316_316266


namespace find_a_for_perpendicular_tangents_l316_316172

theorem find_a_for_perpendicular_tangents (a : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 + 4*y = 0 → 
  x^2 + y^2 + 2*(a-1)*x + 2*y + a^2 = 0 → 
  let (m, n) := (x, y) in 
  (n + 2) / m * (n + 1) / (m - (1 - a)) = -1) → 
  a = -2 :=
sorry

end find_a_for_perpendicular_tangents_l316_316172


namespace circle_formed_by_PO_equals_3_l316_316402

variable (P : ℝ × ℝ) (O : ℝ × ℝ) (r : ℝ)
variable (h_O_fixed : True)
variable (h_PO_constant : dist P O = 3)

theorem circle_formed_by_PO_equals_3 : 
  {P | ∃ (x y : ℝ), dist (x, y) O = 3} = {P | (dist P O = r) ∧ (r = 3)} :=
by
  sorry

end circle_formed_by_PO_equals_3_l316_316402


namespace commuting_hours_l316_316372

theorem commuting_hours (walk_hours_per_trip bike_hours_per_trip : ℕ) 
  (walk_trips_per_week bike_trips_per_week : ℕ) 
  (walk_hours_per_trip = 2) 
  (bike_hours_per_trip = 1)
  (walk_trips_per_week = 3) 
  (bike_trips_per_week = 2) : 
  (2 * (walk_hours_per_trip * walk_trips_per_week) + 2 * (bike_hours_per_trip * bike_trips_per_week)) = 16 := 
  by
  sorry

end commuting_hours_l316_316372


namespace a_beats_b_by_one_round_in_4_round_race_a_beats_b_by_T_a_minus_T_b_l316_316510

noncomputable def T_a : ℝ := 7.5
noncomputable def T_b : ℝ := 10
noncomputable def rounds_a (n : ℕ) : ℝ := n * T_a
noncomputable def rounds_b (n : ℕ) : ℝ := n * T_b

theorem a_beats_b_by_one_round_in_4_round_race :
  rounds_a 4 = rounds_b 3 := by
  sorry

theorem a_beats_b_by_T_a_minus_T_b :
  T_b - T_a = 2.5 := by
  sorry

end a_beats_b_by_one_round_in_4_round_race_a_beats_b_by_T_a_minus_T_b_l316_316510


namespace min_workers_for_profit_l316_316938

theorem min_workers_for_profit (cm : ℕ) (w : ℕ) (p : ℕ) (s : ℕ) (t : ℕ) : 
  cm = 800 → 
  w = 20 → 
  p = 6 → 
  s = 450 → -- Handling $4.50 as cents
  t = 9 → 
  13 ≤ ((800 * 100 + 180 * n) / 100) 
    where n : ℕ := (800 + 180* n)/ 243 + 1 :=
  sorry

end min_workers_for_profit_l316_316938


namespace nathaniel_wins_probability_is_5_over_11_l316_316007

open ProbabilityTheory

noncomputable def nathaniel_wins_probability : ℝ :=
  if ∃ n : ℕ, (∑ k in finset.range (n + 1), k % 7) = 0 then
    5 / 11
  else
    sorry

theorem nathaniel_wins_probability_is_5_over_11 :
  nathaniel_wins_probability = 5 / 11 :=
sorry

end nathaniel_wins_probability_is_5_over_11_l316_316007


namespace sum_of_series_equals_neg_one_l316_316994

noncomputable def omega : Complex := Complex.exp (2 * π * Complex.I / 17)

theorem sum_of_series_equals_neg_one :
  (∑ k in Finset.range 16, omega ^ (k + 1)) = -1 :=
by
  sorry

end sum_of_series_equals_neg_one_l316_316994


namespace batsman_new_average_l316_316524

-- Define the given conditions
variables (A : ℝ) -- Average before the 12th inning
variables (n : ℕ) (new_score : ℝ) (average_increase : ℝ)
variables (total_runs_11 : ℝ) (new_average : ℝ)

-- Conditions of the problem
def conditions : Prop :=
  n = 11 ∧
  new_score = 60 ∧
  average_increase = 4 ∧
  A = (total_runs_11 / n) ∧
  new_average = A + 4

-- The final statement to prove
theorem batsman_new_average (A : ℝ) (total_runs_11 : ℝ) (new_average : ℝ) :
  conditions A total_runs_11 ∧
  12 * new_average = total_runs_11 + 60 →
  new_average = 16 :=
by
  sorry

-- Make all variables constant to prevent mutation
noncomputable def average_before : ℝ := 12
noncomputable def total_runs_before : ℝ := 12 * 11
noncomputable def final_new_average : ℝ := 16

end batsman_new_average_l316_316524


namespace find_g_neg_6_l316_316859

noncomputable def g : ℤ → ℤ := sorry

-- Conditions from the problem
axiom g_condition_1 : g 1 - 1 > 0
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The statement we need to prove
theorem find_g_neg_6 : g (-6) = 723 :=
by { sorry }

end find_g_neg_6_l316_316859


namespace find_g_minus_6_l316_316843

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Conditions given in the problem
axiom cond1 : g 1 - 1 > 0
axiom cond2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom cond3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The proof we want to make (assertion)
theorem find_g_minus_6 : g (-6) = 723 := 
sorry

end find_g_minus_6_l316_316843


namespace composite_divides_factorial_l316_316407

open BigOperators

def largestPrimeLessThan (k : ℕ) : ℕ := 
  if k ≤ 2 then 2 else (Nat.filter Nat.prime (List.range (k-1))).maximum

def isComposite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 2 ≤ a ∧ a ≤ b ∧ n = a * b

theorem composite_divides_factorial 
  (k n : ℕ) 
  (hk : k ≥ 14) 
  (P_k := largestPrimeLessThan k) 
  (hP₁ : P_k < k) 
  (hP₂ : P_k ≥ 3 * k / 4) 
  (hcomp : isComposite n) 
  (hn : n > 2 * P_k) : 
  n ∣ ( (n - k)! ) :=
sorry

end composite_divides_factorial_l316_316407


namespace lily_break_duration_l316_316769

/--
Lily types 15 words per minute and takes a break every 10 minutes.
Given she takes 19 minutes to type 255 words, prove that the length of her break is 2 minutes.
-/
theorem lily_break_duration : 
  (typing_speed : ℕ) (typing_interval : ℕ) (total_time : ℕ) (total_words : ℕ)
  (h_typing_speed : typing_speed = 15)
  (h_typing_interval : typing_interval = 10)
  (h_total_time : total_time = 19)
  (h_total_words : total_words = 255) :
  ∃ (break_time : ℕ), break_time = 2 := 
  sorry

end lily_break_duration_l316_316769


namespace percentage_problem_l316_316415

theorem percentage_problem (P : ℝ) (h : (P / 100) * 180 - (1 / 3) * (P / 100) * 180 = 42) : P = 35 := 
by
  -- Proof goes here
  sorry

end percentage_problem_l316_316415


namespace polar_intersects_l316_316723

noncomputable def polar_intersection : Prop :=
  ∀ ρ θ : ℝ, (ρ * real.sin θ = 2 ∧ ρ = 4 * real.sin θ) →
  ∃ x y : ℝ, (y = 2 ∧ x^2 + (y - 2)^2 = 4)

theorem polar_intersects : polar_intersection :=
by 
  sorry

end polar_intersects_l316_316723


namespace cubes_fractional_parts_sum_l316_316744

noncomputable def fractional_part (x : ℚ) : ℚ := x - x.floor

theorem cubes_fractional_parts_sum (n : ℕ) (h : n ≥ 2) :
  ∃ (x : Fin (n + 1) → ℚ), (∀ i, x i ∉ ℤ) ∧ 
  (fractional_part ((x 0)^3) + fractional_part ((x 1)^3) + ... + fractional_part ((x n)^3) = fractional_part ((x (n+1))^3)) :=
sorry

end cubes_fractional_parts_sum_l316_316744


namespace distance_covered_l316_316517

-- Definitions
def speed : ℕ := 150  -- Speed in km/h
def time : ℕ := 8  -- Time in hours

-- Proof statement
theorem distance_covered : speed * time = 1200 := 
by
  sorry

end distance_covered_l316_316517


namespace interest_time_period_l316_316530

-- Define the constants given in the problem
def principal : ℝ := 4000
def rate1 : ℝ := 0.18
def rate2 : ℝ := 0.12
def interest_difference : ℝ := 480

-- Define the time period T
def time_period : ℝ := 2

-- Define a proof statement
theorem interest_time_period :
  (principal * rate1 * time_period) - (principal * rate2 * time_period) = interest_difference :=
by {
  -- We skip the proof since it's not required by the problem statement
  sorry
}

end interest_time_period_l316_316530


namespace intersection_with_complement_l316_316768

-- Define the universal set U, sets A and B
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 4}

-- Define the complement of B with respect to U
def complement_B : Set ℕ := { x | x ∈ U ∧ x ∉ B }

-- The equivalent proof problem in Lean 4
theorem intersection_with_complement :
  A ∩ complement_B = {0, 2} :=
by
  sorry

end intersection_with_complement_l316_316768


namespace sphere_fitting_cones_l316_316895

/-- Given two congruent right circular cones each with base radius 4 and height 10,
    with axes of symmetry intersecting at right angles at a point inside the cones,
    a distance 4 from the base of each cone, we prove that the maximum possible value of r²,
    where r is the radius of a sphere fitting within both cones,
    is 𝑚/𝑛 with 𝑚 and 𝑛 being coprime integers and 𝑚 + 𝑛 = 557. -/
theorem sphere_fitting_cones :
  ∃ (m n : ℕ), Nat.coprime m n ∧ (4 * ((sqrt 116) - 4) / (sqrt 116)) ^ 2 = m / n ∧ m + n = 557 := by
  sorry

end sphere_fitting_cones_l316_316895


namespace value_of_m_l316_316327

def quadratic_has_one_zero (m : ℝ) : Prop :=
  let Δ := 4 - 12 * m in
  Δ = 0

theorem value_of_m (m : ℝ) :
  quadratic_has_one_zero m → (m = 0 ∨ m = 1 / 3) :=
by
  intro h
  sorry

end value_of_m_l316_316327


namespace sum_tan_roots_l316_316616

open Real

theorem sum_tan_roots (h : 0 ≤ x ∧ x < 2 * π) :
  let tan_x := tan x
  let a := 1
  let b := -7
  let c := 2
  sum_of_roots (λ y, y^2 - 7 * y + 2) x 0 2π = 3 * π :=
sorry

end sum_tan_roots_l316_316616


namespace min_keys_each_director_l316_316546

theorem min_keys_each_director (directors locks : ℕ) (has_key : ℕ → ℕ → Prop) :
  directors = 5 →
  locks = 10 →
  (∀ M : set ℕ, M.card ≥ 3 → (∀ l : ℕ, l < locks → ∃ d ∈ M, has_key d l)) →
  (∀ m : set ℕ, m.card ≤ 2 → ∃ l < locks, ∀ d ∈ m, ¬ has_key d l) →
  ∃ n, n = 6 :=
by
  sorry

end min_keys_each_director_l316_316546


namespace derivative_of_x_ln_x_l316_316448

noncomputable
def x_ln_x (x : ℝ) : ℝ := x * Real.log x

theorem derivative_of_x_ln_x (x : ℝ) (hx : x > 0) :
  deriv (x_ln_x) x = 1 + Real.log x :=
by
  -- Proof body, with necessary assumptions and justifications
  sorry

end derivative_of_x_ln_x_l316_316448


namespace min_max_games_l316_316709

-- Define the conditions of the problem
variables (n : ℕ) (p : fin n → ℕ)
hypothesis h1 : ∀ i j, i ≠ j → p i ≠ p j

-- Define the theorem to be proven in Lean 4
theorem min_max_games (h2 : ∀ i, i < n → 1 ≤ p i) : 
  (∑ i in finset.range n, p i) = (n - 1) * n / 2 := sorry

end min_max_games_l316_316709


namespace ashton_pencils_left_l316_316975

def pencils_left (boxes : Nat) (pencils_per_box : Nat) (pencils_given : Nat) : Nat :=
  boxes * pencils_per_box - pencils_given

theorem ashton_pencils_left :
  pencils_left 2 14 6 = 22 :=
by
  sorry

end ashton_pencils_left_l316_316975


namespace no_third_number_for_lcm_l316_316613

theorem no_third_number_for_lcm (a : ℕ) : ¬ (Nat.lcm (Nat.lcm 23 46) a = 83) :=
sorry

end no_third_number_for_lcm_l316_316613


namespace monotonic_intervals_log_inequality_l316_316664

def f (x : ℝ) : ℝ := x / exp x

theorem monotonic_intervals :
  (∀ x, f' x > 0 ↔ x ∈ Ioo (-∞) 1) ∧ (∀ x, f' x < 0 ↔ x ∈ Ioo 1 ∞) :=
sorry

theorem log_inequality (x : ℝ) (hx : 0 < x) :
  log x > (1 / (exp x) - 2 / (e * x)) :=
sorry

end monotonic_intervals_log_inequality_l316_316664


namespace max_balls_drawn_l316_316031

structure DrawingConditions where
  yellow_items : ℕ
  round_items : ℕ
  edible_items : ℕ
  tomato_properties : Prop
  ball_properties : Prop
  banana_properties : Prop

axiom petya_conditions : DrawingConditions → Prop 

theorem max_balls_drawn (cond : DrawingConditions)
  (h1 : cond.yellow_items = 15)
  (h2 : cond.round_items = 18)
  (h3 : cond.edible_items = 13)
  (h4 : cond.tomato_properties)
  (h5 : cond.ball_properties)
  (h6 : cond.banana_properties) :
  ∀ balls : ℕ, balls ≤ 18 :=
by
  sorry

end max_balls_drawn_l316_316031


namespace combination_10_5_l316_316347

theorem combination_10_5 :
  (Nat.choose 10 5) = 2520 :=
by
  sorry

end combination_10_5_l316_316347


namespace number_of_lion_cubs_l316_316198

theorem number_of_lion_cubs 
    (initial_animal_count final_animal_count : ℕ)
    (gorillas_sent hippo_adopted rhinos_taken new_animals : ℕ)
    (lion_cubs meerkats : ℕ) :
    initial_animal_count = 68 ∧ 
    gorillas_sent = 6 ∧ 
    hippo_adopted = 1 ∧ 
    rhinos_taken = 3 ∧ 
    final_animal_count = 90 ∧ 
    meerkats = 2 * lion_cubs ∧
    new_animals = lion_cubs + meerkats ∧
    final_animal_count = initial_animal_count - gorillas_sent + hippo_adopted + rhinos_taken + new_animals
    → lion_cubs = 8 := sorry

end number_of_lion_cubs_l316_316198


namespace supplements_of_congruent_angles_are_congruent_l316_316240

theorem supplements_of_congruent_angles_are_congruent (a b : ℝ) (h1 : a + \degree(180) = b + \degree(180)) : a = b :=
sorry

end supplements_of_congruent_angles_are_congruent_l316_316240


namespace volume_new_parallelepiped_l316_316660

variable {V : Type*} [inner_product_space ℝ V]

-- Given vectors a, b, and c
variables (a b c : V)

-- Given condition
axiom volume_given : abs (inner_product_space.dot a (cross_product b c)) = 4

-- Theorem to prove
theorem volume_new_parallelepiped :
  abs (inner_product_space.dot (a + 2 • b) (cross_product (b + c) (c + 5 • a))) = 44 :=
by 
  sorry

end volume_new_parallelepiped_l316_316660


namespace at_least_half_team_B_can_serve_on_submarine_l316_316131

theorem at_least_half_team_B_can_serve_on_submarine
    (max_height : ℕ)
    (team_A_avg_height : ℕ)
    (team_B_median_height : ℕ)
    (team_C_tallest_height : ℕ)
    (team_D_mode_height : ℕ)
    (h1 : max_height = 168)
    (h2 : team_A_avg_height = 166)
    (h3 : team_B_median_height = 167)
    (h4 : team_C_tallest_height = 169)
    (h5 : team_D_mode_height = 167) :
  ∀ (height : ℕ), height ≤ max_height → ∃ (b_sailors : ℕ → Prop) (H : ∃ n, b_sailors n),
  (∃ (n_half : ℕ), (∀ h ≤ team_B_median_height, b_sailors h) ∧ (2 * n_half ≤ n)) :=
sorry

end at_least_half_team_B_can_serve_on_submarine_l316_316131


namespace g_neither_even_nor_odd_l316_316600

def g (x : ℝ) : ℝ := 2^x - 3 * x + 1

theorem g_neither_even_nor_odd : 
  ¬ (∀ x, g (-x) = g x) ∧ ¬ (∀ x, g (-x) = -g x) :=
by
  sorry

end g_neither_even_nor_odd_l316_316600


namespace total_earrings_l316_316584

def BellaEarrings : ℕ := 10
def MonicaEarrings : ℕ := BellaEarrings / 0.25  -- Given Bella's earrings are 25% of Monica's earrings
def RachelEarrings : ℕ := MonicaEarrings / 2  -- Given Monica has twice as many earrings as Rachel

theorem total_earrings (B M R : ℕ) 
  (hB : B = BellaEarrings) 
  (hM : M = BellaEarrings / 0.25)
  (hR : R = M / 2) : 
  B + M + R = 70 := 
by 
  sorry

end total_earrings_l316_316584


namespace find_set_Mk_l316_316403

-- Define f(x) using the given condition
def periodic_function (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

def I_k (k : ℤ) : set ℝ := {x | (2 * k - 1:ℝ) < x ∧ x ≤ (2 * k + 1:ℝ)}

def f (k : ℤ) (x : ℝ) : ℝ := (x - 2 * k) ^ 2

-- Condition: The function f(x) is periodic with period 2
axiom periodic_f : periodic_function (f 0)

-- Problem statement
theorem find_set_Mk (k : ℤ) (hk : k > 0) :
  (I_k k).nonempty ∧
  (∀ a : ℝ, (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 ∈ I_k k ∧ x2 ∈ I_k k ∧ (f k x1 = a * x1 ∧ f k x2 = a * x2)) ↔ 
           a ∈ (set.Ioo (-8 * k : ℝ) 0 ∪ set.Ioo (0 : ℝ) ∞)) :=
sorry

end find_set_Mk_l316_316403


namespace shadow_boundary_equation_l316_316199
-- Import necessary libraries

-- Define the conditions
def sphere_center := (0 : ℝ, 0, 2)
def sphere_radius := 2
def light_source := (0 : ℝ, -2, 3)
def shadow_boundary_eq (x : ℝ) : ℝ := x^2 / 6 - 2

-- Statement of the proof problem
theorem shadow_boundary_equation (x y : ℝ) :
  (y = shadow_boundary_eq x) ↔
  -- Conditions
  (sphere_center = (0, 0, 2)) ∧
  (sphere_radius = 2) ∧
  (light_source = (0, -2, 3)) :=
begin
  sorry
end

end shadow_boundary_equation_l316_316199


namespace find_g_neg_six_l316_316862

theorem find_g_neg_six (g : ℤ → ℤ)
  (h1 : g 1 - 1 > 0)
  (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
  (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 :=
sorry

end find_g_neg_six_l316_316862


namespace total_students_in_class_l316_316982

variable (num_chinese num_math num_both num_neither total_students : ℕ)

axiom h1 : num_chinese = 15
axiom h2 : num_math = 18
axiom h3 : num_both = 8
axiom h4 : num_neither = 20

theorem total_students_in_class : total_students = num_chinese + num_math - num_both + num_neither := by
  rw [h1, h2, h3, h4]
  exact rfl

#check total_students_in_class

end total_students_in_class_l316_316982


namespace factorization_correct_l316_316962

theorem factorization_correct: 
  (a : ℝ) → a^2 - 9 = (a - 3) * (a + 3) :=
by
  intro a
  sorry

end factorization_correct_l316_316962


namespace symmetric_scanning_codes_9x9_l316_316947

theorem symmetric_scanning_codes_9x9 :
  ∃ (f : ℕ → ℕ → bool), 
    (∀ i j, (i < 9 ∧ j < 9) → (f i j = f (8 - i) (8 - j))) ∧ -- symmetry under rotations
    (∀ i j, (i < 9 ∧ j < 9) → (f i j = f j i)) ∧ -- symmetry under reflections
    (∃ i j, (i < 9 ∧ j < 9) ∧ f i j) ∧ -- at least one square is true (black)
    (∃ i j, (i < 9 ∧ j < 9) ∧ ¬ f i j) ∧ -- at least one square is false (white)
    (finset.card (finset.univ.image (λ (f : ℕ → ℕ → bool), f)) = 8190) := 
sorry

end symmetric_scanning_codes_9x9_l316_316947


namespace isosceles_triangle_perimeter_l316_316345

def is_isosceles_triangle (a b c : ℝ) : Prop := (a = b ∨ b = c ∨ a = c) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 2) (h2 : b = 5) :
  ∃ c, is_isosceles_triangle a b c ∧ a + b + c = 12 :=
by {
  use 5,
  split,
  simp [is_isosceles_triangle, h1, h2],
  split,
  linarith,
  split,
  linarith,
  linarith,
  ring,
}

end isosceles_triangle_perimeter_l316_316345


namespace child_l316_316457

noncomputable def C (G : ℝ) := 60 - 46
noncomputable def G := 130 - 60
noncomputable def ratio := (C G) / G

theorem child's_weight_to_grandmother's_weight_is_1_5 :
  ratio = 1 / 5 :=
by
  sorry

end child_l316_316457


namespace gcd_lcm_product_24_36_l316_316258

-- Definitions for gcd, lcm, and product for given numbers, skipping proof with sorry
theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  -- Sorry used to skip proof
  sorry

end gcd_lcm_product_24_36_l316_316258


namespace f_shape_open_top_positions_l316_316633

def f_shape_foldable_positions : ℕ := 2

theorem f_shape_open_top_positions
  (initial_configuration : list (ℕ × ℕ))
  (additional_square_position : ℕ × ℕ) :
  (initial_configuration = [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (4, 1)]) →
  ∃ positions : list (ℕ × ℕ), |positions| = f_shape_foldable_positions ∧
  ∀ pos ∈ positions, foldable_with_open_top (initial_configuration ++ [pos]) :=
sorry

end f_shape_open_top_positions_l316_316633


namespace nancy_crayons_l316_316781

theorem nancy_crayons (p c t : ℕ) (h1 : p = 41) (h2 : c = 15) (h3 : t = p * c) : t = 615 :=
by
  sorry

end nancy_crayons_l316_316781


namespace reporters_not_covering_politics_l316_316236

-- Definitions of basic quantities
variables (R P : ℝ) (percentage_local : ℝ) (percentage_no_local : ℝ)

-- Conditions from the problem
def conditions : Prop :=
  R = 100 ∧
  percentage_local = 10 ∧
  percentage_no_local = 30 ∧
  percentage_local = 0.7 * P

-- Theorem statement for the problem
theorem reporters_not_covering_politics (h : conditions R P percentage_local percentage_no_local) :
  100 - P = 85.71 :=
by sorry

end reporters_not_covering_politics_l316_316236


namespace find_t_l316_316321

variable {g V0 k V S t : ℝ}

-- Conditions in the problem
def condition1 : Prop := V = g * t + V0
def condition2 : Prop := S = (1 / 2) * g * t^2 + V0 * t + k * t^3

-- Proposition stating the question with correct answer
theorem find_t (h1 : condition1) (h2 : condition2) : 
  t = (2 * S * (V - V0)) / (V^2 - V0^2 + 2 * k * (V - V0)^2) :=
sorry

end find_t_l316_316321


namespace find_minimal_quadratic_function_l316_316879

theorem find_minimal_quadratic_function :
  ∃ (a b c : ℤ), (∀ x : ℝ, (f(x) = a * x^2 + b * x + c)) ∧
                 (∀ (x y : ℝ), f(f(x)) = 0 → f(f(y)) = 0 → x ≠ y) ∧
                 (f(f(x)) = 0 → ∃ d : ℤ, ∀ i j : ℕ, x_i = d + i * k ∧ y_j = d + j * k) ∧
                 (a = 1) ∧
                 (sum_coeff(a, b, c) is minimized) ∧
                 (f(x) = x^2 + 22x + 105) :=
begin
  sorry
end

end find_minimal_quadratic_function_l316_316879


namespace composite_divides_factorial_of_difference_l316_316405

noncomputable def P_k (k : ℕ) [Fact (k ≥ 14)] : ℕ := 
  Nat.findGreatestPrime (k - 1)

theorem composite_divides_factorial_of_difference
  (k n : ℕ) 
  [Fact (k ≥ 14)]
  (hk : ∃ p : ℕ, P_k k = p ∧ p < k)
  (hprime : P_k k ≥ 3 * k / 4)
  (h_composite : ∃ a b : ℕ, 2 ≤ a ∧ a ≤ b ∧ n = a * b)
  (h_n_gt : n > 2 * P_k k) 
  : n ∣ (n - k)! := 
by 
  sorry

end composite_divides_factorial_of_difference_l316_316405


namespace multiply_98_102_l316_316587

theorem multiply_98_102 : 98 * 102 = 9996 :=
by sorry

end multiply_98_102_l316_316587


namespace pencils_left_l316_316978

def total_pencils (boxes : ℕ) (pencils_per_box : ℕ) : ℕ :=
  boxes * pencils_per_box

def remaining_pencils (initial_pencils : ℕ) (pencils_given : ℕ) : ℕ :=
  initial_pencils - pencils_given

theorem pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (pencils_given : ℕ)
  (h_boxes : boxes = 2) (h_pencils_per_box : pencils_per_box = 14) (h_pencils_given : pencils_given = 6) :
  remaining_pencils (total_pencils boxes pencils_per_box) pencils_given = 22 :=
by
  rw [h_boxes, h_pencils_per_box, h_pencils_given]
  norm_num
  sorry

end pencils_left_l316_316978


namespace more_flour_than_sugar_l316_316772

variable (total_flour : ℕ) (total_sugar : ℕ)
variable (flour_added : ℕ)

def additional_flour_needed (total_flour flour_added : ℕ) : ℕ :=
  total_flour - flour_added

theorem more_flour_than_sugar :
  additional_flour_needed 10 7 - 2 = 1 :=
by
  sorry

end more_flour_than_sugar_l316_316772


namespace total_crayons_l316_316901

theorem total_crayons (Wanda Dina Jacob : Nat) (hWanda : Wanda = 62) (hDina : Dina = 28) (hJacob : Jacob = Dina - 2) :
  Wanda + Dina + Jacob = 116 :=
by
  -- We first use the given conditions to substitute the values
  rw [hWanda, hDina, hJacob]
  -- Simplify the expression to verify the result is as expected
  rw [Nat.succ_sub, Nat.sub_self, Nat.add_comm, Nat.add_assoc]
  norm_num
  sorry

end total_crayons_l316_316901


namespace probability_two_shoes_same_color_opposite_foot_l316_316377

noncomputable def john_shoes : ℕ := 32
noncomputable def black_shoes : ℕ := 16
noncomputable def brown_shoes : ℕ := 10
noncomputable def white_shoes : ℕ := 6

def probability_same_color_opposite_foot : ℚ :=
  (black_shoes / john_shoes) * (8 / (john_shoes - 1)) +
  (brown_shoes / john_shoes) * (5 / (john_shoes - 1)) +
  (white_shoes / john_shoes) * (3 / (john_shoes - 1))

theorem probability_two_shoes_same_color_opposite_foot :
  probability_same_color_opposite_foot = 49 / 248 := by
  sorry

end probability_two_shoes_same_color_opposite_foot_l316_316377


namespace non_integer_x_and_y_impossible_l316_316601

theorem non_integer_x_and_y_impossible 
  (x y : ℚ) (m n : ℤ) 
  (h1 : 5 * x + 7 * y = m)
  (h2 : 7 * x + 10 * y = n) : 
  ∃ (x y : ℤ), 5 * x + 7 * y = m ∧ 7 * x + 10 * y = n := 
sorry

end non_integer_x_and_y_impossible_l316_316601


namespace average_vegetables_per_week_l316_316494

theorem average_vegetables_per_week (P Vp S W : ℕ) (h1 : P = 200) (h2 : Vp = 2) (h3 : S = 25) (h4 : W = 2) :
  (P / Vp) / S / W = 2 :=
by
  sorry

end average_vegetables_per_week_l316_316494


namespace max_balls_count_l316_316028

-- Definitions
def Object := Type
def sun : Object := sorry
def ball : Object := sorry
def tomato : Object := sorry
def banana : Object := sorry

def is_yellow : Object → Prop := sorry
def is_round : Object → Prop := sorry
def is_edible : Object → Prop := sorry

axiom yellow_items_count : ∃ (Y : set Object), Y.card = 15 ∧ ∀ y ∈ Y, is_yellow y
axiom round_items_count : ∃ (R : set Object), R.card = 18 ∧ ∀ r ∈ R, is_round r
axiom edible_items_count : ∃ (E : set Object), E.card = 13 ∧ ∀ e ∈ E, is_edible e

-- Problem conditions
axiom tomato_is_round_and_red : ∀ t, t = tomato → is_round t ∧ ¬is_yellow t
axiom ball_is_round : ∀ b, b = ball → is_round b
axiom banana_is_yellow_and_not_round : ∀ b, b = banana → is_yellow b ∧ ¬is_round b

-- Target proposition
theorem max_balls_count : 
  ∀ (sun_count ball_count tomato_count banana_count : ℕ),
    is_round sun → ¬is_yellow sun →
    is_round ball → is_edible ball → ¬is_yellow ball → 
    is_yellow tomato → is_round tomato → is_edible tomato →
    is_yellow banana → ¬is_round banana → is_edible banana →
    sun_count + ball_count + tomato_count + banana_count = 46 →
    ball_count ≤ 18 := sorry

end max_balls_count_l316_316028


namespace max_balls_drawn_l316_316043

-- Defining the concepts of objects Petya can draw
inductive Object
| Sun
| Ball
| Tomato
| Banana

-- Defining properties for objects
def isYellow (o : Object) : Bool :=
  match o with
  | Object.Banana => true
  | _ => false

def isRound (o : Object) : Bool :=
  match o with
  | Object.Ball   => true
  | Object.Tomato => true
  | _             => false

def isEdible (o : Object) : Bool :=
  match o with
  | Object.Tomato => true
  | Object.Banana => true
  | _             => false

def countObjects (p : Object -> Bool) (os : List Object) : Nat :=
  os.countp p

theorem max_balls_drawn (os : List Object) :
  countObjects isYellow os = 15 →
  countObjects isRound os = 18 →
  countObjects isEdible os = 13 →
  countObjects (λ o => o = Object.Ball) os = 18 :=
by
  intros hy hr he
  -- Proof will go here
  sorry

end max_balls_drawn_l316_316043


namespace sum_midpoint_x_coords_l316_316882

theorem sum_midpoint_x_coords (a b c d : ℝ) (h : a + b + c + d = 20) :
  ((a + b) / 2 + (b + c) / 2 + (c + d) / 2 + (d + a) / 2) = 20 :=
by 
  calc
    ((a + b) / 2 + (b + c) / 2 + (c + d) / 2 + (d + a) / 2)
      = (a + b + b + c + c + d + d + a) / 2 : by sorry
    ... = (2 * (a + b + c + d)) / 2         : by sorry
    ... = a + b + c + d                     : by sorry
    ... = 20                                : by exact h

end sum_midpoint_x_coords_l316_316882
