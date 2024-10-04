import Mathlib
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Parity
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Sqrt
import Mathlib.Data.Perm.Basic
import Mathlib.Data.Probability
import Mathlib.Data.Real.Basic
import Mathlib.Exponents
import Mathlib.Functions
import Mathlib.Init.Data.Nat.Basic
import Mathlib.Order.Continuity
import Mathlib.Probability.Basic
import Mathlib.Probability.Distribution
import Mathlib.Probability.Expectation
import Mathlib.Probability.Identity
import Mathlib.Probability.Independent
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Pigeonhole
import Mathlib.Tactic.Ring
import data.real.basic

namespace number_of_three_digit_integers_with_odd_factors_l186_186419

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l186_186419


namespace initial_bananas_per_child_l186_186965

theorem initial_bananas_per_child : 
  ∀ (B n m x : ℕ), 
  n = 740 → 
  m = 370 → 
  (B = n * x) → 
  (B = (n - m) * (x + 2)) → 
  x = 2 := 
by
  intros B n m x h1 h2 h3 h4
  sorry

end initial_bananas_per_child_l186_186965


namespace length_of_DE_parallel_to_base_l186_186990

-- Define the lengths and the variables
def base_AB : ℝ := 20
def area_ratio : ℝ := 0.81
def scale_factor : ℝ := real.sqrt area_ratio

-- length_DE : The length of DE we need to prove
theorem length_of_DE_parallel_to_base :
  ∃ (length_DE : ℝ), length_DE = scale_factor * base_AB :=
begin
  use scale_factor * base_AB,
  have h_scale_factor : scale_factor = 0.9, 
  { rw [real.sqrt_eq_rpow, real.rpow_nat_cast], exact real.sqrt_eq_rpow, norm_cast, linarith },
  norm_num,
  sorry
end

end length_of_DE_parallel_to_base_l186_186990


namespace three_digit_integers_with_odd_number_of_factors_l186_186165

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l186_186165


namespace num_chickens_is_one_l186_186724

-- Define the number of dogs and the number of total legs
def num_dogs := 2
def total_legs := 10

-- Define the number of legs per dog and per chicken
def legs_per_dog := 4
def legs_per_chicken := 2

-- Define the number of chickens
def num_chickens := (total_legs - num_dogs * legs_per_dog) / legs_per_chicken

-- Prove that the number of chickens is 1
theorem num_chickens_is_one : num_chickens = 1 := by
  -- This is the proof placeholder
  sorry

end num_chickens_is_one_l186_186724


namespace red_bushes_in_middle_probability_l186_186694

theorem red_bushes_in_middle_probability :
  let total_arrangements := (4.factorial / (2.factorial * 2.factorial))
  let favorable_arrangements := 1
  (favorable_arrangements.to_rat / total_arrangements.to_rat) = (1 / 6) := 
by
  sorry

end red_bushes_in_middle_probability_l186_186694


namespace number_of_three_digit_squares_l186_186433

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l186_186433


namespace find_x_of_geometric_sequence_l186_186941

noncomputable def fractional (x : ℝ) : ℝ := x - floor x

theorem find_x_of_geometric_sequence (x : ℝ) (hx₀ : x ≠ 0)
  (hgeom : fractional x ≠ 0 ∧ fractional x < 1 ∧ 
           (floor x : ℝ) ≠ 0 ∧ 
           (floor x : ℝ) / fractional x = x / (floor x : ℝ)) :
  x = Real.sqrt 5 / 2 :=
sorry

end find_x_of_geometric_sequence_l186_186941


namespace reflection_curve_eq_l186_186950

theorem reflection_curve_eq (a b c : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, let y := ax^2 + bx + c in
   let C1 := ax^2 - bx + c in
   let C2 := -C1 in
   y = -ax^2 + bx - c) :=
sorry

end reflection_curve_eq_l186_186950


namespace correct_propositions_count_l186_186052

-- Definitions of the four conclusions in the problem
def conclusion1 := "In regression analysis, the coefficient of determination \( R^2 \) can be used to judge the model's fitting effect. The larger the \( R^2 \), the better the model's fitting effect."
def conclusion2 := "In regression analysis, the sum of squared residuals can be used to judge the model's fitting effect. The larger the sum of squared residuals, the better the model's fitting effect."
def conclusion3 := "In regression analysis, the correlation coefficient \( r \) can be used to judge the model's fitting effect. The larger the \( r \), the better the model's fitting effect."
def conclusion4 := "In regression analysis, the residual plot can be used to judge the model's fitting effect. If the residual points are evenly distributed in a horizontal band, it indicates that the model is appropriate. The narrower the band, the higher the precision of the model's fitting."

-- Counting the correct conclusions
def correct_conclusions := 2

-- Statement: Prove that the number of correct propositions is equal to 2
theorem correct_propositions_count : 
  (true ∧ ¬true ∧ true ∧ ¬true) = (true, false, true, false) := by sorry

end correct_propositions_count_l186_186052


namespace number_of_three_digit_integers_with_odd_number_of_factors_l186_186333

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l186_186333


namespace difference_in_earnings_l186_186734

theorem difference_in_earnings :
  let cost_TOP := 8
  let cost_ABC := 23
  let num_TOP_sold := 13
  let num_ABC_sold := 4
  let earnings_TOP := cost_TOP * num_TOP_sold
  let earnings_ABC := cost_ABC * num_ABC_sold
  earnings_TOP - earnings_ABC = 12 :=
by
  let cost_TOP := 8
  let cost_ABC := 23
  let num_TOP_sold := 13
  let num_ABC_sold := 4
  let earnings_TOP := cost_TOP * num_TOP_sold
  let earnings_ABC := cost_ABC * num_ABC_sold
  show earnings_TOP - earnings_ABC = 12 from sorry

end difference_in_earnings_l186_186734


namespace num_three_digit_integers_with_odd_factors_l186_186367

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l186_186367


namespace three_digit_integers_with_odd_factors_count_l186_186286

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l186_186286


namespace initial_number_divisible_by_97_l186_186658

theorem initial_number_divisible_by_97 (N : ℕ) (k : ℤ) (h1 : N - 5 = 97 * k) : N = 102 :=
begin
  sorry
end

end initial_number_divisible_by_97_l186_186658


namespace three_digit_oddfactors_count_l186_186137

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l186_186137


namespace Linda_greatest_possible_average_speed_l186_186725

-- Definition of the conditions
def initial_odometer := 12321
def is_palindrome (n : Nat) : Bool :=
  let s := toString n
  s == s.reverse

def is_speed_limit (speed : Nat) : Bool :=
  speed <= 75

def driving_time := 3 -- hours
def max_possible_distance := 75 * driving_time

-- Lean statement of the proof problem
theorem Linda_greatest_possible_average_speed :
  ∃ avg_speed : ℚ, 
    (∀ d : ℕ, initial_odometer < d ∧ d ≤ initial_odometer + max_possible_distance →
      is_palindrome d → avg_speed = d / driving_time) ∧ 
    avg_speed = 200 / 3 :=
sorry

end Linda_greatest_possible_average_speed_l186_186725


namespace three_digit_odds_factors_count_l186_186351

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l186_186351


namespace three_digit_integers_with_odd_number_of_factors_count_l186_186279

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l186_186279


namespace three_digit_integers_with_odd_factors_l186_186218

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l186_186218


namespace three_digit_integers_with_odd_number_of_factors_count_l186_186273

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l186_186273


namespace optionC_is_correct_l186_186924

def KalobsWindowLength : ℕ := 50
def KalobsWindowWidth : ℕ := 80
def KalobsWindowArea : ℕ := KalobsWindowLength * KalobsWindowWidth

def DoubleKalobsWindowArea : ℕ := 2 * KalobsWindowArea

def optionC_Length : ℕ := 50
def optionC_Width : ℕ := 160
def optionC_Area : ℕ := optionC_Length * optionC_Width

theorem optionC_is_correct : optionC_Area = DoubleKalobsWindowArea := by
  sorry

end optionC_is_correct_l186_186924


namespace num_three_digit_integers_with_odd_factors_l186_186374

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l186_186374


namespace daria_weeks_needed_l186_186758

-- Defining the parameters and conditions
def initial_amount : ℕ := 20
def weekly_savings : ℕ := 10
def cost_of_vacuum_cleaner : ℕ := 120

-- Defining the total money Daria needs to add to her initial amount
def additional_amount_needed : ℕ := cost_of_vacuum_cleaner - initial_amount

-- Defining the number of weeks needed to save the additional amount, given weekly savings
def weeks_needed : ℕ := additional_amount_needed / weekly_savings

-- The theorem stating that Daria needs exactly 10 weeks to cover the expense of the vacuum cleaner
theorem daria_weeks_needed : weeks_needed = 10 := by
  sorry

end daria_weeks_needed_l186_186758


namespace number_of_three_digit_integers_with_odd_factors_l186_186402

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l186_186402


namespace area_of_ABCD_l186_186566

variable (A B C D M N O : Type)
variable [ConvexQuadrilateral A B C D] 
variable [Midpoint M B C] 
variable [Midpoint N A D]
variable [Intersection O (Segment M N) (Segment A C)]
variable (MO_eq_ON : Distance O M = Distance O N)
variable (area_ABC : TriangleArea A B C = 2017)

theorem area_of_ABCD : AreaOfQuadrilateral A B C D = 4034 := by sorry

end area_of_ABCD_l186_186566


namespace symmetric_point_coords_l186_186515

theorem symmetric_point_coords
  (a : ℝ)
  (A : ℝ × ℝ)
  (Ax Ay : ℝ)
  (A = (Ax, Ay))
  (eqn : Ay = a * (Ax + 2)^2) :
  (Ax = 1 ∧ Ay = 4) → (Ax - -2) = (3 : ℝ) ∧ (-2 - 3) = -5 ∧ Ay = 4 :=
by {
  intro h,
  cases h with hAx hAy,
  split,
  {
    rw hAx,
    norm_num,
  },
  split,
  {
    norm_num,
  },
  {
    exact hAy,
  }
}

end symmetric_point_coords_l186_186515


namespace average_grains_per_teaspoon_l186_186637

noncomputable def grains_per_teaspoon (grains_per_cup : ℕ) (tablespoons_per_half_cup : ℝ) (teaspoons_per_tablespoon : ℝ) : ℝ :=
  (grains_per_cup / (2 * tablespoons_per_half_cup)) / teaspoons_per_tablespoon

theorem average_grains_per_teaspoon :
  let
    basmati := grains_per_teaspoon 490 8 3,
    jasmine := grains_per_teaspoon 470 7.5 3.5,
    arborio := grains_per_teaspoon 450 9 2.5
  in
  abs ((basmati + jasmine + arborio) / 3 - 9.7333) < 0.0001 :=
by
  sorry

end average_grains_per_teaspoon_l186_186637


namespace weight_of_each_bar_l186_186532

theorem weight_of_each_bar 
  (num_bars : ℕ) 
  (cost_per_pound : ℝ) 
  (total_cost : ℝ) 
  (total_weight : ℝ) 
  (weight_per_bar : ℝ)
  (h1 : num_bars = 20)
  (h2 : cost_per_pound = 0.5)
  (h3 : total_cost = 15)
  (h4 : total_weight = total_cost / cost_per_pound)
  (h5 : weight_per_bar = total_weight / num_bars)
  : weight_per_bar = 1.5 := 
by
  sorry

end weight_of_each_bar_l186_186532


namespace range_of_angle_of_inclination_l186_186771

noncomputable def angle_of_inclination (theta : ℝ) (h1 : -1 ≤ Real.sin theta) (h2 : Real.sin theta ≤ 1) : ℝ :=
  if h : Real.sin theta = 0 
  then Real.pi / 2 
  else Real.arctan (1 / Real.sin theta)

theorem range_of_angle_of_inclination (theta : ℝ) 
  (h1 : -1 ≤ Real.sin theta) 
  (h2 : Real.sin theta ≤ 1) 
  : ∃ alpha : ℝ, (alpha = angle_of_inclination theta h1 h2) ∧ (alpha ∈ set.Icc (Real.pi / 4) (3 * Real.pi / 4)) :=
sorry

end range_of_angle_of_inclination_l186_186771


namespace solve_sqrt_equation_l186_186958

theorem solve_sqrt_equation :
  ∀ x : ℝ, (sqrt (2 * x + 3) = x) ↔ (x = 3) := by
  intro x
  split
  {
    -- Assume sqrt (2 * x + 3) = x and prove x = 3.
    sorry
  }
  {
    -- Assume x = 3 and prove sqrt (2 * x + 3) = x.
    sorry
  }

end solve_sqrt_equation_l186_186958


namespace three_digit_oddfactors_count_is_22_l186_186095

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l186_186095


namespace more_consistent_l186_186501

open Finset

/-- Define the score variance for participant A and scores for participant B -/
def variance (s : Finset ℝ) : ℝ :=
  let n := (card s).toReal
  let mean := (s.sum id) / n
  (s.sum (λ x => (x - mean) ^ 2)) / n

def scores_A := 0.3
def scores_B := {8, 9, 9, 9, 10}

theorem more_consistent :
  variance scores_B > scores_A := by
  unfold variance
  simp [scores_B, scores_A]
  sorry

end more_consistent_l186_186501


namespace power_function_increasing_implies_m_gt_1_l186_186883

theorem power_function_increasing_implies_m_gt_1 (m : ℝ) (f : ℝ → ℝ) 
  (h : ∀ x ∈ set.Ioi (0:ℝ), deriv f x > 0) : m > 1 :=
begin
  assume (h : ∀ x ∈ set.Ioi (0:ℝ), deriv (λ x, x^(m-1)) x > 0),
  sorry
end

end power_function_increasing_implies_m_gt_1_l186_186883


namespace find_b_perpendicular_lines_l186_186612

theorem find_b_perpendicular_lines :
  ∀ (b : ℝ), (∀ x y : ℝ, 2 * x - 3 * y + 6 = 0 ∧ b * x - 3 * y + 6 = 0 →
      (2 / 3) * (b / 3) = -1) → b = -9 / 2 :=
sorry

end find_b_perpendicular_lines_l186_186612


namespace three_digit_integers_with_odd_factors_l186_186214

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l186_186214


namespace sum_first_15_nat_eq_120_l186_186051

-- Define a function to sum the first n natural numbers
def sum_natural_numbers (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Define the theorem to show that the sum of the first 15 natural numbers equals 120
theorem sum_first_15_nat_eq_120 : sum_natural_numbers 15 = 120 := 
  by
    sorry

end sum_first_15_nat_eq_120_l186_186051


namespace eccentricity_of_ellipse_l186_186543

open Real

def ellipse_eq (a b x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def foci_dist_eq (a c : ℝ) : Prop :=
  2 * c / (2 * a) = sqrt 6 / 2

noncomputable def eccentricity (c a : ℝ) : ℝ :=
  c / a

theorem eccentricity_of_ellipse (a b x y c : ℝ)
  (h1 : ellipse_eq a b x y)
  (h2 : foci_dist_eq a c) :
  eccentricity c a = sqrt 6 / 3 :=
sorry

end eccentricity_of_ellipse_l186_186543


namespace smallest_possible_ten_digit_number_l186_186964

theorem smallest_possible_ten_digit_number :
  let numbers := [415, 43, 7, 8, 74, 3]
  let sorted_numbers := [3, 7, 8, 43, 74, 415]
  let concatenated_number := list.foldr (λ x acc, x * 10^(int.log10 acc + 1).to_nat + acc) 0 sorted_numbers
  concatenated_number = 3415437478 :=
by {
  let numbers := [415, 43, 7, 8, 74, 3],
  let sorted_numbers := [3, 7, 8, 43, 74, 415],
  have h : (list.foldr (λ x acc, x * 10^(int.log10 acc + 1).to_nat + acc) 0 sorted_numbers) = 3415437478,
    sorry,
  exact h,
}

end smallest_possible_ten_digit_number_l186_186964


namespace functional_equation_l186_186611

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f(x + y) = f(x) + f(y))
  (h2 : ∀ x : ℝ, x > 0 → f(x) < 0) :
  (∀ x : ℝ, f(-x) = -f(x)) ∧ (∀ x1 x2 : ℝ, x1 < x2 → f(x2) < f(x1)) :=
begin
  sorry
end

end functional_equation_l186_186611


namespace line_intersects_y_axis_l186_186735

theorem line_intersects_y_axis :
  let p1 := (2 : ℝ, 8 : ℝ)
  let p2 := (4 : ℝ, 12 : ℝ)
  let m := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b := p1.2 - m * p1.1
  line_equation : ∀ (x : ℝ), (m * x + b)
  (m = 2) → (b = 4) →
  line_equation 0 = 4 :=
by
  intros
  sorry

end line_intersects_y_axis_l186_186735


namespace general_term_formula_possible_values_b_3_min_value_k_l186_186032

-- Definitions and Conditions
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, d ≠ 0 ∧ ∀ n : ℕ, a (n+1) = a n + d

def geometric_seq (a : ℕ → ℤ) : Prop :=
  ∀ m n p : ℕ, a n * a n = a m * a p

def a_n (n : ℕ) : ℤ := n + 3

def b_n (b : ℕ → ℤ) : Prop :=
  b 1 = 0 ∧ ∀ n ≥ 2, |b n - b (n - 1)| = 2^(a_n n)

theorem general_term_formula :
  ∀ (a : ℕ → ℤ) (d : ℤ),
  arithmetic_seq a ∧ a 2 = 5 ∧ (a 1 * a 1 = (a 2 - d) * (a 2 + 4 * d)) → 
  (a = a_n) :=
sorry

theorem possible_values_b_3 :
  ∀ (b : ℕ → ℤ),
  b_n b →
  b 3 ∈ {-96, -32, 32, 96} :=
sorry

theorem min_value_k :
  ∀ (b : ℕ → ℤ) (k : ℕ),
  b_n b ∧ b k = 2116 → 
  k = 7 :=
sorry

end general_term_formula_possible_values_b_3_min_value_k_l186_186032


namespace part_one_part_two_l186_186644

-- Probability of hitting the target for shooters A and B
def p_A : ℚ := 2 / 3
def p_B : ℚ := 3 / 4

-- Probability that A misses the target at least once in three consecutive shots
def prob_A1 : ℚ := 1 - p_A^3

-- Probability of hitting the target exactly twice for A in three shots
def prob_A2 : ℚ := (3.choose 2) * p_A^2 * (1 - p_A)

-- Probability of hitting the target exactly once for B in three shots
def prob_B2 : ℚ := (3.choose 1) * p_B * (1 - p_B)^2

-- Combined probability of both events
def prob_A2B2 : ℚ := prob_A2 * prob_B2

theorem part_one : prob_A1 = 19 / 27 := 
sorry

theorem part_two : prob_A2B2 = 1 / 16 := 
sorry

end part_one_part_two_l186_186644


namespace three_digit_integers_with_odd_factors_l186_186221

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l186_186221


namespace three_digit_integers_with_odd_factors_count_l186_186295

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l186_186295


namespace three_digit_odds_factors_count_l186_186349

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l186_186349


namespace magazine_cost_l186_186957

theorem magazine_cost :
  ∃ (x : ℝ), 
    let total_books := 9 + 1,
    let cost_per_book := 15,
    let total_cost_books := total_books * cost_per_book,
    let total_spent := 170,
    let num_magazines := 10,
    let total_cost_magazines := num_magazines * x 
    in total_cost_books + total_cost_magazines = total_spent ∧ x = 2 :=
sorry

end magazine_cost_l186_186957


namespace part_a_part_b_l186_186525

-- Definitions based on conditions
def vertices : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
def distance (i j : ℕ) : ℕ := min ((abs (j - i)) % 13) (13 - ((abs (j - i)) % 13))

-- Proof problem for part (a)
theorem part_a : 
  ∃ (v₁ v₂ v₃ v₄ : ℕ), 
    v₁ ∈ vertices ∧ v₂ ∈ vertices ∧ v₃ ∈ vertices ∧ v₄ ∈ vertices ∧
    v₁ ≠ v₂ ∧ v₁ ≠ v₃ ∧ v₁ ≠ v₄ ∧ v₂ ≠ v₃ ∧ v₂ ≠ v₄ ∧ v₃ ≠ v₄ ∧ 
    list.nodup ([distance v₁ v₂, distance v₁ v₃, distance v₁ v₄, distance v₂ v₃, distance v₂ v₄, distance v₃ v₄] : List ℕ) := 
sorry

-- Proof problem for part (b)
theorem part_b : 
  ¬ ∃ (v₁ v₂ v₃ v₄ v₅ : ℕ), 
    v₁ ∈ vertices ∧ v₂ ∈ vertices ∧ v₃ ∈ vertices ∧ v₄ ∈ vertices ∧ v₅ ∈ vertices ∧
    v₁ ≠ v₂ ∧ v₁ ≠ v₃ ∧ v₁ ≠ v₄ ∧ v₁ ≠ v₅ ∧
    v₂ ≠ v₃ ∧ v₂ ≠ v₄ ∧ v₂ ≠ v₅ ∧
    v₃ ≠ v₄ ∧ v₃ ≠ v₅ ∧
    v₄ ≠ v₅ ∧ 
    list.nodup ([distance v₁ v₂, distance v₁ v₃, distance v₁ v₄, distance v₁ v₅, distance v₂ v₃, distance v₂ v₄, distance v₂ v₅, distance v₃ v₄, distance v₃ v₅, distance v₄ v₅] : List ℕ) := 
sorry

-- Add noncomputable to avoid computation when it is necessary
noncomputable section

end part_a_part_b_l186_186525


namespace three_digit_oddfactors_count_is_22_l186_186093

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l186_186093


namespace greatest_integer_less_or_equal_l186_186653

theorem greatest_integer_less_or_equal :
  let x := (2^50 + 5^50) / (2^47 + 5^47)
  in floor x = 124 := by
  sorry

end greatest_integer_less_or_equal_l186_186653


namespace number_of_three_digit_squares_l186_186440

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l186_186440


namespace three_digit_integers_with_odd_factors_l186_186211

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l186_186211


namespace platform_length_proof_l186_186666

noncomputable def length_of_platform (train_length : ℕ) (pole_time : ℕ) (platform_time : ℕ) : ℕ :=
  let speed := train_length / pole_time in
  let total_distance := speed * platform_time in
  total_distance - train_length

theorem platform_length_proof :
  length_of_platform 900 18 39 = 1050 :=
by
  rfl

end platform_length_proof_l186_186666


namespace true_propositions_l186_186053

/-- Propositions Concerning Sampling, Correlation, Chi-Squared Statistic, and Normal Distribution. -/
def proposition1 : Prop :=
  ∀ (s : ℕ → ℝ), (∀ (k : ℕ), s k = k * 30) → ∀ (k : ℕ), s k = k * 30

def proposition2 : Prop :=
  ∀ (r : ℝ), (r ∈ [-1, 1]) → (|r| → Prop) → r ≠ ±1 

def proposition3 : Prop :=
  ∀ (X Y : Type) (k2 : ℝ), (k2 < 0) → X ≠ Y

def proposition4 : Prop :=
  ∀ (X : ℝ → ℝ), 
  (∀ (x : ℝ), X x = exp (-x^2 / 2) / sqrt (2 * π)) → 
  ∀ (p : ℝ), (p = exp (-1 / 2) / sqrt (2 * π)) → (P (abs X < 1) = 2 * P (X < 1) - 1)

-- The final theorem stating which propositions are true:
theorem true_propositions :  proposition1 ∧ proposition4 :=
by {
  sorry -- to be proved
}

end true_propositions_l186_186053


namespace range_of_a_l186_186067

-- Define the sets A, B, and C
def set_A (x : ℝ) : Prop := -3 < x ∧ x ≤ 2
def set_B (x : ℝ) : Prop := -1 < x ∧ x < 3
def set_A_int_B (x : ℝ) : Prop := -1 < x ∧ x ≤ 2
def set_C (x : ℝ) (a : ℝ) : Prop := a < x ∧ x < a + 1

-- The target theorem to prove
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, set_C x a → set_A_int_B x) → 
  (-1 ≤ a ∧ a ≤ 1) :=
by 
  sorry

end range_of_a_l186_186067


namespace solve_for_k_l186_186481

theorem solve_for_k (x : ℝ) (k : ℝ) (h₁ : 2 * x - 1 = 3) (h₂ : 3 * x + k = 0) : k = -6 :=
by
  sorry

end solve_for_k_l186_186481


namespace find_point_B_l186_186034

structure Point where
  x : ℝ
  y : ℝ

def vec_scalar_mult (c : ℝ) (v : Point) : Point :=
  ⟨c * v.x, c * v.y⟩

def vec_add (p : Point) (v : Point) : Point :=
  ⟨p.x + v.x, p.y + v.y⟩

theorem find_point_B :
  let A := Point.mk 1 (-3)
  let a := Point.mk 3 4
  let B := vec_add A (vec_scalar_mult 2 a)
  B = Point.mk 7 5 :=
by {
  sorry
}

end find_point_B_l186_186034


namespace hyperbola_eccentricity_range_l186_186846

theorem hyperbola_eccentricity_range 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h_intersect : ∀ F : ℝ × ℝ, 
                  ∃ P : ℝ × ℝ, 
                    ( ∃ x y : ℝ, 
                        ((x / a) ^ 2 - (y / b) ^ 2 = 1) ∧
                        P = (x, y) ) ∧
                    ( ∃ θ : ℝ, 
                        θ = 60 ∧ (P.1 - F.1) = (P.2 - F.2) * tan 60 
                        )
                      ) : 
  let e := (a^2 + b^2) / a^2 in
  2 ≤ e :=
by 
  sorry

end hyperbola_eccentricity_range_l186_186846


namespace medians_perpendicular_l186_186962

   theorem medians_perpendicular (DP EQ DE : ℝ) (hDP : DP = 27) (hEQ : EQ = 36)
     (hMediansPerpendicular : ∀ G : Point, is_centroid G ⟶ (is_median DP ∧ is_median EQ ⟶ DP ⊥ EQ)) :
     DE = 30 :=
   by
     sorry
   
end medians_perpendicular_l186_186962


namespace exists_solution_in_interval_l186_186911

-- Define the function f
def f (x : ℝ) : ℝ := 2^x + 2 * x - 6

-- Theorem statement
theorem exists_solution_in_interval :
  f 1 < 0 ∧ f 2 > 0 → ∃ x ∈ set.Ioo 1 2, f x = 0 :=
by
  intro h,
  use sorry

end exists_solution_in_interval_l186_186911


namespace number_of_three_digit_squares_l186_186423

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l186_186423


namespace distinct_prime_factors_of_M_l186_186766

noncomputable def is_prime (n : Nat) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def count_prime_factors (n : Nat) : Nat :=
  (filter is_prime (List.range (n + 1))).count (λ p, p ∣ n)

theorem distinct_prime_factors_of_M :
  ∃ M : ℕ, (log 2 (log 3 (log 5 (log 7 (log 11 M)))) = 8) ∧ (count_prime_factors M = 1) :=
sorry

end distinct_prime_factors_of_M_l186_186766


namespace num_three_digit_integers_with_odd_factors_l186_186379

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l186_186379


namespace number_of_solutions_l186_186769

noncomputable def num_solutions_cos_eq : ℝ :=
  let f := λ x : ℝ, 3 * (Real.cos x)^3 - 7 * (Real.cos x)^2 + 3 * (Real.cos x)
  let solutions := {x : ℝ | 0 <= x ∧ x <= 2 * Real.pi ∧ f x = 0}
  solutions.to_finset.card

theorem number_of_solutions : num_solutions_cos_eq = 4 := by
  sorry

end number_of_solutions_l186_186769


namespace num_three_digit_integers_with_odd_factors_l186_186365

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l186_186365


namespace three_digit_integers_odd_factors_count_l186_186445

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l186_186445


namespace inclination_angle_of_line_l186_186613

theorem inclination_angle_of_line :
  ∃ θ : ℝ, ∃ m : ℝ, (x y : ℝ), x - y - 2 = 0 → m = 1 ∧ tan θ = m ∧ θ = arctan(m) :=
sorry

end inclination_angle_of_line_l186_186613


namespace three_digit_integers_with_odd_number_of_factors_count_l186_186274

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l186_186274


namespace M_inter_N_l186_186853

namespace ProofProblem

def M : Set ℝ := { x | 3 * x - x^2 > 0 }
def N : Set ℝ := { x | x^2 - 4 * x + 3 > 0 }

theorem M_inter_N : M ∩ N = { x | 0 < x ∧ x < 1 } :=
sorry

end ProofProblem

end M_inter_N_l186_186853


namespace vector_addition_find_lambda_l186_186070

theorem vector_addition (a b : ℝ × ℝ) (ha : a = (2, 1)) (hb : b = (1, 1)) :
  a + b = (3, 2) :=
by
  rw [ha, hb]
  simp

theorem find_lambda (a b c : ℝ × ℝ) (λ : ℝ) (ha : a = (2, 1)) (hb : b = (1, 1)) (hc : c = (5, 2))
  (h_parallel : ∃ (k : ℝ), a = k • (λ • b + c)) : λ = 1 :=
sorry

end vector_addition_find_lambda_l186_186070


namespace max_digits_construct_valid_six_digit_sequence_l186_186985

-- Define the conditions for the digits of the integer
def valid_digit_sequence (a : List ℕ) : Prop :=
  (∀ i j, i ≠ j → List.get a i ≠ List.get a j) ∧ -- All digits are distinct
  (∀ i, List.get a i + List.get a (i+1) + List.get a (i+2) % 5 = 0) -- Sum of any three consecutive digits is divisible by 5

-- The theorem statements
theorem max_digits (n : ℕ) (a : List ℕ) (h1 : valid_digit_sequence a) : n ≤ 6 :=
sorry

theorem construct_valid_six_digit_sequence (d : ℕ) :
  ∃ (a : List ℕ), valid_digit_sequence a ∧ List.length a = 6 ∧ List.get a 0 = d :=
sorry

end max_digits_construct_valid_six_digit_sequence_l186_186985


namespace exists_infinite_commuting_functions_l186_186927

variable {ℝ : Type} [CommRing ℝ]

def bijective_function (f : ℝ → ℝ) : Prop :=
  ∃ (g : ℝ → ℝ), (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

def infinite_commuting_functions (f : ℝ → ℝ) : Prop :=
  ∃ (g : ℝ → ℝ), ∀ x, f (g x) = g (f x)

theorem exists_infinite_commuting_functions (f : ℝ → ℝ) (hf : bijective_function f) : ∃ g, infinite_commuting_functions f :=
sorry

end exists_infinite_commuting_functions_l186_186927


namespace T_property_exists_d_l186_186928

-- Let n be a positive integer
variable {n : ℕ} (hn : 0 < n)

def in_T (x : (Fin n) → ℝ) :=
  ∃ σ : (Fin n) → Fin n, (∀ i : Fin (n - 1), x (σ i) - x (σ (i+1)) ≥ 1)

theorem T_property_exists_d :
  ∃ d : ℝ, ∀ (a : Fin n → ℝ), ∃ (b c : Fin n → ℝ), 
    in_T hn b ∧ in_T hn c ∧ (∀ i : Fin n, a i = (1 / 2) * (b i + c i) ∧ |a i - b i| ≤ d ∧ |a i - c i| ≤ d) :=
begin
  sorry
end

end T_property_exists_d_l186_186928


namespace mode_of_student_scores_l186_186892

-- We define the problem: the list of scores
def student_scores : List ℕ := [70, 80, 100, 60, 80, 70, 90, 50, 80, 70, 80, 70, 90, 80, 90, 80, 70, 90, 60, 80]

-- Prove that the mode of student_scores is 80
theorem mode_of_student_scores : List.mode student_scores = 80 := 
by
  sorry

end mode_of_student_scores_l186_186892


namespace evaporation_period_length_l186_186696

theorem evaporation_period_length
  (initial_water : ℕ) (daily_evaporation : ℝ) (evaporated_percentage : ℝ) : 
  evaporated_percentage * (initial_water : ℝ) / 100 / daily_evaporation = 22 :=
by
  -- Conditions of the problem
  let initial_water := 12
  let daily_evaporation := 0.03
  let evaporated_percentage := 5.5
  -- Sorry proof placeholder
  sorry

end evaporation_period_length_l186_186696


namespace smallest_sum_of_digits_l186_186654

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (3 * n^2 + n + 1).digits.sum

theorem smallest_sum_of_digits : ∀ n : ℕ, sum_of_digits n ≥ 3 :=
begin
  sorry
end

end smallest_sum_of_digits_l186_186654


namespace three_digit_oddfactors_count_is_22_l186_186097

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l186_186097


namespace number_of_three_digit_squares_l186_186432

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l186_186432


namespace contractor_daily_wage_l186_186689

theorem contractor_daily_wage :
  let x := 25 in
  let total_days := 30 in
  let absent_days := 10 in
  let fine_per_absent_day := 7.50 in
  let total_earned := 425 in
  let worked_days := total_days - absent_days in
  (worked_days * x - absent_days * fine_per_absent_day = total_earned) → x = 25 :=
by
  intros
  sorry

end contractor_daily_wage_l186_186689


namespace tangent_line_to_curve_at_point_l186_186002

theorem tangent_line_to_curve_at_point :
  ∀ (x y : ℝ), y = 1 / x ∧ (x, y) = (1, 1) → x + y - 2 = 0 :=
by
  intros x y
  rintro ⟨hx, h⟩
  have h₁ : y = 1 / x := hx
  have h₂ : (x = 1 ∧ y = 1) := by injection h with h₃ h₄
  rw [h₃, h₄]
  simp

end tangent_line_to_curve_at_point_l186_186002


namespace knight_cannot_traverse_all_squares_exactly_once_l186_186899

def knight_move (start finish: ℕ × ℕ) : Prop :=
  let (r_start, c_start) := start
  let (r_finish, c_finish) := finish
  (abs (r_start - r_finish) = 2 ∧ abs (c_start - c_finish) = 1) ∨ (abs (r_start - r_finish) = 1 ∧ abs (c_start - c_finish) = 2)

def knight_on_board (pos: ℕ × ℕ) : Prop :=
  let (row, col) := pos
  1 ≤ row ∧ row ≤ 9 ∧ 1 ≤ col ∧ col ≤ 9

theorem knight_cannot_traverse_all_squares_exactly_once :
  ¬ ∃ (path : List (ℕ × ℕ)), 
    (List.length path = 81 ∧
     path.head = some (3, 4) ∧
     ∀ i, i < 80 → knight_on_board (path.nth_le i (by linarith)) ∧ knight_move (path.nth_le i (by linarith)) (path.nth_le (i+1) (by linarith))) :=
by
  sorry

end knight_cannot_traverse_all_squares_exactly_once_l186_186899


namespace three_digit_integers_with_odd_factors_l186_186143

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l186_186143


namespace sum_of_zeros_l186_186475

noncomputable def f : ℝ → ℝ := sorry

theorem sum_of_zeros (h_symm : ∀ x, f (1 - x) = f (3 + x))
  (h_zeros : ∃ a b c, f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ (a ≠ b ∧ b ≠ c ∧ c ≠ a)) :
  (∃ a b c, a + b + c = 6) :=
begin
  sorry
end

end sum_of_zeros_l186_186475


namespace RichardsNumbersUnique_l186_186575

noncomputable def RichardNumbersSum := 73591 + 46280
noncomputable def RichardNumbersDifference := 73591 - 46280

theorem RichardsNumbersUnique (a b : ℕ) 
  (h1 : (a < 100000 ∧ 9999 < a) ∧ ∀ d ∈ (a.digits 10), d % 2 = 1)
  (h2 : (b < 100000 ∧ 9999 < b) ∧ ∀ e ∈ (b.digits 10), e % 2 = 0) 
  (sum_start : (a + b).digits 10.reverse.take 2 = [1, 1]) 
  (sum_end : (a + b) % 10 = 1)
  (diff_start : (a - b).digits 10.head 1 = 2)
  (diff_end : (a - b).digits 10.drop (4) = [1, 1]) :
  (a = 73591 ∧ b = 46280) :=
by 
  sorry

end RichardsNumbersUnique_l186_186575


namespace num_three_digit_integers_with_odd_factors_l186_186380

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l186_186380


namespace num_three_digit_ints_with_odd_factors_l186_186248

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l186_186248


namespace three_digit_integers_with_odd_number_of_factors_l186_186180

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l186_186180


namespace author_earnings_correct_l186_186731

def paperback_earnings (copies : ℕ) (price : ℝ) (percentage : ℝ) : ℝ :=
  (copies * price) * percentage / 100

def hardcover_earnings (copies : ℕ) (price : ℝ) (percentage : ℝ) : ℝ :=
  (copies * price) * percentage / 100

def ebook_earnings (copies : ℕ) (price : ℝ) (percentage : ℝ) : ℝ :=
  (copies * price) * percentage / 100

def audiobook_earnings (copies : ℕ) (price : ℝ) (percentage : ℝ) : ℝ :=
  (copies * price) * percentage / 100

def author_total_earnings (pb_earnings hc_earnings eb_earnings ab_earnings : ℝ) : ℝ :=
  pb_earnings + hc_earnings + eb_earnings + ab_earnings

theorem author_earnings_correct :
  let pb_copies := 32000
      pb_price := 0.20
      pb_percentage := 6
      hc_copies := 15000
      hc_price := 0.40
      hc_percentage := 12
      eb_copies := 10000
      eb_price := 0.15
      eb_percentage := 8
      ab_copies := 5000
      ab_price := 0.50
      ab_percentage := 10
  in author_total_earnings
    (paperback_earnings pb_copies pb_price pb_percentage)
    (hardcover_earnings hc_copies hc_price hc_percentage)
    (ebook_earnings eb_copies eb_price eb_percentage)
    (audiobook_earnings ab_copies ab_price ab_percentage) = 1474 :=
by
  sorry

end author_earnings_correct_l186_186731


namespace probability_of_wave_number_l186_186707

def is_wave_number (n : Nat) : Prop :=
  let d := [n / 10000 % 10, n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  d[1] > d[0] ∧ d[1] > d[2] ∧ d[3] > d[2] ∧ d[3] > d[4]

def five_digit_non_repeating_numbers : List Nat :=
  [1, 2, 3, 4, 5].permutations.map (λ l => l.foldl (λ acc d => acc * 10 + d) 0)

def count_wave_numbers : Nat :=
  (five_digit_non_repeating_numbers.filter is_wave_number).length

theorem probability_of_wave_number : count_wave_numbers = 8 ∧ 5! = 120 → (count_wave_numbers : ℚ) / 120 = 1 / 15 :=
by
  intros
  rw [Nat.factorial, Nat.cast_mul, Nat.cast_bit0, Nat.cast_one, Nat.cast_mul, Nat.cast_bit0, Nat.cast_bit0, Nat.cast_one]
  sorry

end probability_of_wave_number_l186_186707


namespace three_digit_integers_odd_factors_count_l186_186453

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l186_186453


namespace tan_ratio_identity_l186_186464

/-- Given trigonometric identities and equation conditions, we need to prove the result -/
theorem tan_ratio_identity (x y : ℝ) 
  (h1 : (sin x) / (cos y) - (sin y) / (cos x) = 2)
  (h2 : (cos x) / (sin y) - (cos y) / (sin x) = 3) :
  (tan x) / (tan y) - (tan y) / (tan x) = -1 :=
sorry

end tan_ratio_identity_l186_186464


namespace three_digit_integers_with_odd_number_of_factors_count_l186_186263

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l186_186263


namespace max_points_on_poly_graph_l186_186821

theorem max_points_on_poly_graph (P : Polynomial ℤ) (h_deg : P.degree = 20):
  ∃ (S : Finset (ℤ × ℤ)), (∀ p ∈ S, 0 ≤ p.snd ∧ p.snd ≤ 10) ∧ S.card ≤ 20 ∧ 
  ∀ S' : Finset (ℤ × ℤ), (∀ p ∈ S', 0 ≤ p.snd ∧ p.snd ≤ 10) → S'.card ≤ 20 :=
by
  sorry

end max_points_on_poly_graph_l186_186821


namespace count_valid_pairs_l186_186868

theorem count_valid_pairs : 
  ∃ n : ℕ, n = 5 ∧ 
  ∀ (i j : ℕ), 0 ≤ i ∧ i < j ∧ j ≤ 40 →
  (5^j - 2^i) % 1729 = 0 →
  i = 0 ∧ j = 36 ∨ 
  i = 1 ∧ j = 37 ∨ 
  i = 2 ∧ j = 38 ∨ 
  i = 3 ∧ j = 39 ∨ 
  i = 4 ∧ j = 40 :=
by
  sorry

end count_valid_pairs_l186_186868


namespace problem_proof_l186_186551

noncomputable def quadratic_roots (a b c : ℝ) : ℝ × ℝ :=
(((-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a)), 
 ((-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)))

theorem problem_proof :
  let p := (quadratic_roots 1 13 (-4)).1,
      q := (quadratic_roots 1 13 (-4)).2 in
  (p + 4) * (q + 4) = -40 :=
by
  let p := (quadratic_roots 1 13 (-4)).1
  let q := (quadratic_roots 1 13 (-4)).2
  have h : (x : ℝ) -> (x - 6) * (2 * x + 10) = x^2 - 15 * x + 56 := sorry
  have hpq : (x^2 + 13 * x - 4 = 0) ∧
             (p = (-13 + Real.sqrt 185) / 2) ∧
             (q = (-13 - Real.sqrt 185) / 2) := sorry
  have key : (∀ p q, (p + 4) * (q + 4) = -40) := sorry
  sorry

end problem_proof_l186_186551


namespace total_cost_of_breakfast_l186_186803

-- Definitions based on conditions
def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3
def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2
def kiera_muffins : ℕ := 2
def kiera_fruit_cup : ℕ := 1

-- The proof statement
theorem total_cost_of_breakfast : 
  muffin_cost * francis_muffins + 
  fruit_cup_cost * francis_fruit_cups + 
  muffin_cost * kiera_muffins + 
  fruit_cup_cost * kiera_fruit_cup = 17 := 
  by sorry

end total_cost_of_breakfast_l186_186803


namespace point_coordinates_l186_186020

/-- Given the vector from point A to point B, if point A is the origin, then point B will have coordinates determined by the vector. -/
theorem point_coordinates (A B: ℝ × ℝ) (v: ℝ × ℝ) 
  (h: A = (0, 0)) (h_v: v = (-2, 4)) (h_ab: B = (A.1 + v.1, A.2 + v.2)): 
  B = (-2, 4) :=
by
  sorry

end point_coordinates_l186_186020


namespace find_circle_center_l186_186064

-- Define the conditions
variables (k a b : ℝ) (p q : ℝ)
hypothesis parabola_eq : ∀ x, ∃ y, y = k * x^2
hypothesis circle_eq : ∀ x y, x^2 - 2 * p * x + y^2 - 2 * q * y = 0
hypothesis roots_eq : ∀ x, x^3 + a * x + b = 0

-- Define the target statement
theorem find_circle_center:
  p = -k^2 / 2 * b →
  q = 1 / (2 * k) - k * a / 2 →
  (p, q) = (-b / 2, (1 - a) / 2) :=
by
  intros hp hq
  rw [hp, hq]
  simp
  exact sorry

end find_circle_center_l186_186064


namespace sum_distances_greater_l186_186601

noncomputable def dist (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

noncomputable def distance_to_line (P A B : ℝ × ℝ) : ℝ :=
  (abs ((B.2 - A.2) * P.1 - (B.1 - A.1) * P.2 + B.1 * A.2 - B.2 * A.1)) / (dist A B)

variables {A B C D M : ℝ × ℝ}

theorem sum_distances_greater (
  h1 : dist C A = dist A B,
  h2 : dist D B = dist B A,
  h3 : A ≠ B,
  h4 : M ≠ A,
  h5 : M ≠ B,
  h6 : M ≠ C,
  h7 : M ≠ D
) : 
  (distance_to_line M A D + distance_to_line M B C) >
  (distance_to_line M A B) :=
sorry

end sum_distances_greater_l186_186601


namespace zeros_in_decimal_representation_l186_186863

theorem zeros_in_decimal_representation : 
  let n := 2^7 * 5^3 in
  ∀ (a b : ℕ), (1 / n * (5^4) = a / (10^b)) → b = 7 → a = 625 → 4 = 7 - 3 :=
by
  intros n a b h₁ h₂ h₃
  cases b
  { exfalso,
    sorry }
  cases b
  { exfalso,
    sorry }
  cases b
  { exfalso,
    sorry }
  cases b
  { exfalso,
    sorry }
  cases b
  { exfalso,
    sorry }
  cases b
  { exfalso,
    sorry }
  cases b
  { exfalso,
    sorry }
  cases b
  { exfalso,
    sorry }
  cases b
  { exfalso,
    sorry }
  cases b
  { exfalso,
    sorry }
  cases b
  {  refl },
  sorry


end zeros_in_decimal_representation_l186_186863


namespace num_three_digit_ints_with_odd_factors_l186_186243

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l186_186243


namespace trapezium_area_l186_186784

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 13) :
  (1 / 2) * (a + b) * h = 247 :=
by
  sorry

end trapezium_area_l186_186784


namespace three_digit_oddfactors_count_l186_186135

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l186_186135


namespace Daria_vacuum_cleaner_problem_l186_186759

theorem Daria_vacuum_cleaner_problem (initial_savings weekly_savings target_savings weeks_needed : ℕ)
  (h1 : initial_savings = 20)
  (h2 : weekly_savings = 10)
  (h3 : target_savings = 120)
  (h4 : weeks_needed = (target_savings - initial_savings) / weekly_savings) : 
  weeks_needed = 10 :=
by
  sorry

end Daria_vacuum_cleaner_problem_l186_186759


namespace minimum_pencils_in_one_box_l186_186585

theorem minimum_pencils_in_one_box (total_pencils : ℕ) (num_boxes : ℕ) (max_capacity : ℕ) 
    (h1 : total_pencils = 74) (h2 : num_boxes = 13) (h3 : max_capacity = 6) :
    ∃ (min_pencils : ℕ), min_pencils = 2 ∧ min_pencils <= max_capacity ∧ 
      (total_pencils ≤ num_boxes * max_capacity) ∧ 
      (∀ k, k ∈ finset.range(num_boxes) → (total_pencils - 12 * max_capacity = 2) → 
      (min_pencils = 2)) :=
by
  sorry

end minimum_pencils_in_one_box_l186_186585


namespace simplify_expression_l186_186981

theorem simplify_expression (k : ℂ) : 
  ((1 / (3 * k)) ^ (-3) * ((-2) * k) ^ (4)) = 432 * (k ^ 7) := 
by sorry

end simplify_expression_l186_186981


namespace overall_profit_percentage_is_30_l186_186714

noncomputable def overall_profit_percentage (n_A n_B : ℕ) (price_A price_B profit_A profit_B : ℝ) : ℝ :=
  (n_A * profit_A + n_B * profit_B) / (n_A * price_A + n_B * price_B) * 100

theorem overall_profit_percentage_is_30 :
  overall_profit_percentage 5 10 850 950 225 300 = 30 :=
by
  sorry

end overall_profit_percentage_is_30_l186_186714


namespace three_digit_integers_with_odd_number_of_factors_count_l186_186278

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l186_186278


namespace number_of_three_digit_integers_with_odd_number_of_factors_l186_186339

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l186_186339


namespace log_400_cannot_be_computed_l186_186815

theorem log_400_cannot_be_computed :
  let log_8 : ℝ := 0.9031
  let log_9 : ℝ := 0.9542
  let log_7 : ℝ := 0.8451
  (∀ (log_2 log_3 log_5 : ℝ), log_2 = 1 / 3 * log_8 → log_3 = 1 / 2 * log_9 → log_5 = 1 → 
    (∀ (log_val : ℝ), 
      (log_val = log_21 → log_21 = log_3 + log_7 → log_val = (1 / 2) * log_9 + log_7)
      ∧ (log_val = log_9_over_8 → log_9_over_8 = log_9 - log_8)
      ∧ (log_val = log_126 → log_126 = log_2 + log_7 + log_9 → log_val = (1 / 3) * log_8 + log_7 + log_9)
      ∧ (log_val = log_0_875 → log_0_875 = log_7 - log_8)
      ∧ (log_val = log_400 → log_400 = log_8 + 1 + log_5) 
      → False))
:= 
sorry

end log_400_cannot_be_computed_l186_186815


namespace complex_number_solution_l186_186814

theorem complex_number_solution (a b : ℝ) (i : ℂ) (h₀ : Complex.I = i)
  (h₁ : (a - 2* (i^3)) / (b + i) = i) : a + b = 1 :=
by 
  sorry

end complex_number_solution_l186_186814


namespace theta_value_pure_imaginary_l186_186468

theorem theta_value_pure_imaginary (θ : ℝ) (k : ℤ) :
  (sin (2 * θ) - 1 + (real.sqrt 2 + 1) * I = (real.sqrt 2 + 1) * I) → θ = k * real.pi + real.pi / 4 :=
by
  intro h
  sorry

end theta_value_pure_imaginary_l186_186468


namespace product_of_four_is_perfect_square_l186_186627

theorem product_of_four_is_perfect_square (S : Finset ℕ) (hS_card : S.card = 48)
  (hS_product : ∃ (p : Finset ℕ), p.card = 10 ∧ ∀ (n ∈ S), ∀ (q ∈ p), Prime q ∧ q ∣ n) :
  ∃ (T : Finset ℕ), T.card = 4 ∧ ∃ k : ℕ, (k * k = T.prod id) :=
by {
  sorry
}

end product_of_four_is_perfect_square_l186_186627


namespace trains_cross_time_same_direction_l186_186646

/--
Two trains of equal length, running with the speeds of 60 and 40 kmph, take 11 seconds to cross each other 
when running in the opposite direction. Prove they take 55 seconds to cross each other when running in 
the same direction.
-/
theorem trains_cross_time_same_direction
  (L : ℝ) -- Length of each train in meters
  (v1 v2 : ℝ) -- Speeds of the trains in km/h
  (t_opposite : ℝ) -- Time to cross in opposite directions in seconds
  (h1 : v1 = 60) (h2 : v2 = 40) (h3 : t_opposite = 11) :
  let
    relative_speed_opposite := (v1 + v2) * (5/18) -- converting km/h to m/s
    distance_opposite := relative_speed_opposite * t_opposite
    L := distance_opposite / 2
    relative_speed_same := (v1 - v2) * (5/18) -- converting km/h to m/s 
    t_same := (2 * L) / relative_speed_same
  in
  t_same = 55 := by
  sorry

end trains_cross_time_same_direction_l186_186646


namespace quadratic_min_value_max_l186_186066

theorem quadratic_min_value_max (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : b^2 - 4 * a * c ≥ 0) :
    (min (min ((b + c) / a) ((c + a) / b)) ((a + b) / c)) ≤ (5 / 4) :=
sorry

end quadratic_min_value_max_l186_186066


namespace three_digit_integers_with_odd_number_of_factors_l186_186174

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l186_186174


namespace three_digit_integers_with_odd_factors_l186_186213

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l186_186213


namespace three_digit_integers_with_odd_factors_l186_186227

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l186_186227


namespace nonnegative_integers_count_l186_186767

-- Define the range of coefficients
def coeffs : List Int := [-2, -1, 0, 1, 2]

-- Define the polynomial expression
def polynomial (a : Fin 8 → Int) : Int :=
  (List.range 8).sum (function (i : Nat) => (a ⟨i, sorry⟩) * (5 ^ i))

-- Define the main problem statement
theorem nonnegative_integers_count : 
  {n : Int // ∃ (a : Fin 8 → Int), (∀ i, a i ∈ coeffs) ∧ polynomial a = n }.count (λ n, n ≥ 0) = 156253 := sorry

end nonnegative_integers_count_l186_186767


namespace three_digit_integers_with_odd_factors_l186_186240

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l186_186240


namespace repair_cost_l186_186577

variable (R : ℝ)

theorem repair_cost (purchase_price transportation_charges profit_rate selling_price : ℝ) (h1 : purchase_price = 12000) (h2 : transportation_charges = 1000) (h3 : profit_rate = 0.5) (h4 : selling_price = 27000) :
  R = 5000 :=
by
  have total_cost := purchase_price + R + transportation_charges
  have selling_price_eq := 1.5 * total_cost
  have sp_eq_27000 := selling_price = 27000
  sorry

end repair_cost_l186_186577


namespace breakfast_cost_l186_186807

def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3

def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2

def kiera_muffins : ℕ := 2
def kiera_fruit_cup : ℕ := 1

theorem breakfast_cost :
  muffin_cost * francis_muffins + fruit_cup_cost * francis_fruit_cups
  + muffin_cost * kiera_muffins + fruit_cup_cost * kiera_fruit_cup = 17 :=
by
  -- skipping proof
  sorry

end breakfast_cost_l186_186807


namespace three_digit_oddfactors_count_is_22_l186_186094

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l186_186094


namespace three_digit_oddfactors_count_is_22_l186_186098

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l186_186098


namespace num_three_digit_ints_with_odd_factors_l186_186255

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l186_186255


namespace three_digit_integers_with_odd_factors_l186_186396

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l186_186396


namespace three_digit_integers_with_odd_factors_l186_186142

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l186_186142


namespace minimum_value_of_f_l186_186609

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt ((x + 2)^2 + 4^2)) + (Real.sqrt ((x + 1)^2 + 3^2))

theorem minimum_value_of_f : ∃ x : ℝ, f x = 5 * Real.sqrt 2 ∧ ∀ y : ℝ, f y ≥ f x :=
by
  use -3
  sorry

end minimum_value_of_f_l186_186609


namespace tangent_line_parallel_x_axis_coordinates_l186_186482

theorem tangent_line_parallel_x_axis_coordinates :
  (∃ P : ℝ × ℝ, P = (1, -2) ∨ P = (-1, 2)) ↔
  (∃ x y : ℝ, y = x^3 - 3 * x ∧ ∃ y', y' = 3 * x^2 - 3 ∧ y' = 0) :=
by
  sorry

end tangent_line_parallel_x_axis_coordinates_l186_186482


namespace triangle_area_l186_186513

theorem triangle_area (BC : ℝ) (h : ℝ) (right_angle : ∠ACB = π / 2) (BC_length : BC = 12) (height_length : h = 15) :
  ∃ (area : ℝ), area = 90 := 
by
  use (1 / 2 * BC * h)
  rw [BC_length, height_length]
  norm_num
  sorry

end triangle_area_l186_186513


namespace number_of_three_digit_squares_l186_186422

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l186_186422


namespace Daria_vacuum_cleaner_problem_l186_186761

theorem Daria_vacuum_cleaner_problem (initial_savings weekly_savings target_savings weeks_needed : ℕ)
  (h1 : initial_savings = 20)
  (h2 : weekly_savings = 10)
  (h3 : target_savings = 120)
  (h4 : weeks_needed = (target_savings - initial_savings) / weekly_savings) : 
  weeks_needed = 10 :=
by
  sorry

end Daria_vacuum_cleaner_problem_l186_186761


namespace equation_of_perpendicular_line_l186_186788

theorem equation_of_perpendicular_line :
  ∃ c : ℝ, (∀ x y : ℝ, (2 * x + y + c = 0 ↔ (x = 1 ∧ y = 1))) → (c = -3) := 
by
  sorry

end equation_of_perpendicular_line_l186_186788


namespace three_digit_oddfactors_count_l186_186125

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l186_186125


namespace trapezium_area_l186_186780

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 13) :
  (1 / 2) * (a + b) * h = 247 :=
by
  rw [ha, hb, hh]
  norm_num
  sorry

end trapezium_area_l186_186780


namespace find_smallest_positive_omega_l186_186059

theorem find_smallest_positive_omega :
  ∃ ω > 0, (∀ x : ℝ, sin (ω * (x - π / 3) + π / 3) = -sin (ω * x + π / 3)) ∧ ω = 3 :=
begin
  sorry
end

end find_smallest_positive_omega_l186_186059


namespace starting_number_selection_l186_186914

theorem starting_number_selection (n : ℕ) (a : ℕ → ℕ) 
  (h_sum : (∑ i in range n, a i) = 2 * n - 1) :
  ∃ s : fin n → fin n, 
    (∀ k : ℕ, k < n → (∑ i in range k, a (s i)) ≤ 2 * k - 1) :=
sorry

end starting_number_selection_l186_186914


namespace january_1_on_friday_l186_186890

theorem january_1_on_friday (hM : ∃ d. (∀ i ∈ Finset.range 5, (d + 7 * i) < 32 → day_of_january (d + 7 * i) = Monday))
                            (hT : ∃ d. (∀ i ∈ Finset.range 5, (d + 7 * i) < 32 → day_of_january (d + 7 * i) = Thursday)) : 
    day_of_january 1 = Friday := 
sorry

end january_1_on_friday_l186_186890


namespace num_three_digit_integers_with_odd_factors_l186_186378

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l186_186378


namespace sum_of_consecutive_squares_is_not_a_square_example_of_11_consecutive_squares_l186_186975

theorem sum_of_consecutive_squares_is_not_a_square (m : ℕ) (h : m = 3 ∨ m = 4 ∨ m = 5 ∨ m = 6) :
  ¬ ∃ n k : ℤ, ∑ i in range m, (n + i)^2 = k^2 := by
  sorry

theorem example_of_11_consecutive_squares :
  ∃ (n : ℕ) (L : list ℕ), L = range 11 ∧ (∑ i in L.map (λ x, (x + n)^2)) = 77^2 := by
  use 18
  use [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  apply and.intro
  {
    rfl
  }
  {
    calc
      ∑ i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map (λ x, (x + 18)^2)
      = 18^2 + 19^2 + 20^2 + 21^2 + 22^2 + 23^2 + 24^2 + 25^2 + 26^2 + 27^2 + 28^2 : by simp
      ... = 77^2 : by norm_num
  }

end sum_of_consecutive_squares_is_not_a_square_example_of_11_consecutive_squares_l186_186975


namespace inscribed_quadrilateral_exists_l186_186753

theorem inscribed_quadrilateral_exists (a b c d : ℝ) (h1: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  ∃ (p q : ℝ),
    p = Real.sqrt ((a * c + b * d) * (a * d + b * c) / (a * b + c * d)) ∧
    q = Real.sqrt ((a * b + c * d) * (a * d + b * c) / (a * c + b * d)) ∧
    a * c + b * d = p * q :=
by
  sorry

end inscribed_quadrilateral_exists_l186_186753


namespace three_digit_integers_with_odd_factors_count_l186_186283

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l186_186283


namespace three_digit_odds_factors_count_l186_186361

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l186_186361


namespace distance_between_A_and_B_is_5_l186_186847

-- Define point A
def A : ℝ × ℝ := (-2, 3)

-- Define point B
def B : ℝ × ℝ := (1, -1)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- State the main theorem we want to prove
theorem distance_between_A_and_B_is_5 : distance A B = 5 :=
by
  -- The proof is omitted
  sorry

end distance_between_A_and_B_is_5_l186_186847


namespace complex_modulus_problem_l186_186038

theorem complex_modulus_problem (x y : ℝ) (h : (1 + complex.i) * x = 1 + y * complex.i) :
  complex.abs (x + y * complex.i) = real.sqrt 2 :=
sorry

end complex_modulus_problem_l186_186038


namespace collinear_M_N_P_l186_186942

-- Defining the geometric configuration
variable (A B C I D E P M N : Point)
variable (incircle_touch_BC : Point) (incircle_touch_AC : Point)
variable (incenter : Point)
variable (midpoint_BC : Point) (midpoint_AB : Point)
variable (incircle : Circumcircle)
variable (segment_AI : Line)
variable (segment_DE : Line)

-- Defining conditions
def triangle_ABC := Triangle A B C
def incenter_condition := Incenter I triangle_ABC
def touch_BC := incircle_touch_BC = D
def touch_AC := incircle_touch_AC = E
def intersection_condition := Intersects segment_AI segment_DE P
def midpoint_condition_BC := Midpoint B C M
def midpoint_condition_AB := Midpoint A B N

-- The statement to prove
theorem collinear_M_N_P 
  (triangle_ABC : Triangle A B C)
  (incenter_condition : Incenter I triangle_ABC)
  (touch_BC : incircle_touch_BC = D)
  (touch_AC : incircle_touch_AC = E)
  (intersection_condition : Intersects segment_AI segment_DE P)
  (midpoint_condition_BC : Midpoint B C M)
  (midpoint_condition_AB : Midpoint A B N) :
  Collinear M N P :=
sorry

end collinear_M_N_P_l186_186942


namespace minimum_distance_l186_186931

open_locale big_operators

noncomputable def parabola : set (ℝ × ℝ) := { p | (p.snd)^2 = 8 * (p.fst) }

def focus : ℝ × ℝ := (2, 3)
def F : ℝ × ℝ := (2, 0) -- Considering F is the focus located at (2,0)

def distance (x y : ℝ × ℝ) : ℝ :=
  real.sqrt ((x.fst - y.fst)^2 + (x.snd - y.snd)^2)

def D (P : ℝ × ℝ) : set (ℝ × ℝ) := {D | D.fst = -2}

theorem minimum_distance (P : parabola) : ∃ P : ℝ × ℝ, P ∈ parabola ∧ 
  distance P focus + distance P F = 4 ∧ P = (9 / 8, 3) :=
begin
  use (9/8, 3),
  simp,
  split,
  { exact sorry }, -- Parabola condition
  split,
  { exact sorry }, -- Minimum value condition
  { exact sorry }, -- Coordinate condition
end

end minimum_distance_l186_186931


namespace tetrahedron_angle_sum_l186_186072

open Real
open Finset

-- Definitions for the problem setup
variables {O A B C P : Point}
variables {α β γ : ℝ}

-- Mutual perpendicularity condition
def mutually_perpendicular (O A B C : Point) : Prop := 
  perpendicular (O, A, B) ∧ perpendicular (O, B, C) ∧ perpendicular (O, A, C)

-- Point P in the triangle ABC condition
def point_in_triangle (P A B C : Point) : Prop := 
  ∃ (u v w : ℝ), u + v + w = 1 ∧ (0 ≤ u ∧ 0 ≤ v ∧ 0 ≤ w) ∧ u • A + v • B + w • C = P

-- Angles α, β, γ between OP and planes OAB, OBC, OCA respectively
def angle_between_line_and_plane (OP OAB : ℝ) : Prop := 
  -- Placeholder for the actual definition
  true

-- The main theorem statement
theorem tetrahedron_angle_sum 
  (h1 : mutually_perpendicular O A B C)
  (h2 : point_in_triangle P A B C)
  (hα : angle_between_line_and_plane (dist O P) (dist P A))
  (hβ : angle_between_line_and_plane (dist O P) (dist P B))
  (hγ : angle_between_line_and_plane (dist O P) (dist P C)) :
  ∃ (α β γ : ℝ), 
    0 < α ∧ α < π / 2 ∧ 
    0 < β ∧ β < π / 2 ∧ 
    0 < γ ∧ γ < π / 2 ∧
    (π / 2 < α + β + γ ∧ α + β + γ ≤ 3 * arcsin (sqrt 3 / 3)) := 
sorry

end tetrahedron_angle_sum_l186_186072


namespace three_digit_integers_with_odd_factors_l186_186148

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l186_186148


namespace ang_B_is_45_degrees_l186_186518

theorem ang_B_is_45_degrees (A B C D : Type) [IsoscelesTriangle ABC AB AC]
  (h1 : AB = AC)
  (h2 : ∃ D ∈ segment AC, IsAngleBisector (angle BCA) (ray CD))
  (h3 : CD = CB) :
  angle B = 45 :=
  sorry

end ang_B_is_45_degrees_l186_186518


namespace three_digit_integers_with_odd_factors_l186_186238

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l186_186238


namespace num_odd_factors_of_three_digit_integers_eq_22_l186_186113

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l186_186113


namespace number_of_three_digit_integers_with_odd_factors_l186_186406

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l186_186406


namespace beaver_hid_90_carrots_l186_186799

-- Defining the number of burrows and carrot condition homomorphic to the problem
def beaver_carrots (x : ℕ) := 5 * x
def rabbit_carrots (y : ℕ) := 7 * y

-- Stating the main theorem based on conditions derived from the problem
theorem beaver_hid_90_carrots (x y : ℕ) (h1 : beaver_carrots x = rabbit_carrots y) (h2 : y = x - 5) : 
  beaver_carrots x = 90 := 
by 
  sorry

end beaver_hid_90_carrots_l186_186799


namespace three_digit_integers_with_odd_factors_count_l186_186285

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l186_186285


namespace find_t_l186_186825

variables {m n : ℝ}
variables (t : ℝ)
variables (mv nv : ℝ)
variables (dot_m_m dot_m_n dot_n_n : ℝ)
variables (cos_theta : ℝ)

-- Define the basic assumptions
axiom non_zero_vectors : m ≠ 0 ∧ n ≠ 0
axiom magnitude_condition : mv = 2 * nv
axiom cos_condition : cos_theta = 1 / 3
axiom perpendicular_condition : dot_m_n = (mv * nv * cos_theta) ∧ (t * dot_m_n + dot_m_m = 0)

-- Utilize the conditions and prove the target
theorem find_t : t = -6 :=
sorry

end find_t_l186_186825


namespace product_mod_25_l186_186772

def remainder_when_divided_by_25 (n : ℕ) : ℕ := n % 25

theorem product_mod_25 (a b c d : ℕ) 
  (h1 : a = 1523) (h2 : b = 1857) (h3 : c = 1919) (h4 : d = 2012) :
  remainder_when_divided_by_25 (a * b * c * d) = 8 :=
by
  sorry

end product_mod_25_l186_186772


namespace three_digit_integers_with_odd_factors_l186_186312

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l186_186312


namespace sum_of_zeros_l186_186477

-- Defining the conditions and the result
theorem sum_of_zeros (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = f (3 + x)) (a b c : ℝ)
  (h1 : f a = 0) (h2 : f b = 0) (h3 : f c = 0) : 
  a + b + c = 3 := 
by 
  sorry

end sum_of_zeros_l186_186477


namespace relationship_t_s_l186_186508

variable {a b : ℝ}

theorem relationship_t_s (a b : ℝ) (t : ℝ) (s : ℝ) (ht : t = a + 2 * b) (hs : s = a + b^2 + 1) :
  t ≤ s := 
sorry

end relationship_t_s_l186_186508


namespace incorrect_props_l186_186728

theorem incorrect_props :
  (Subsets A = {0, 1}) = 3 → 
  (∀ a b m, am^2 < bm^2 → a < b) → 
  (∀ p q, (p ∨ q) → (p ∧ q)) → 
  (¬ ∀ x ∈ ℝ, x^2 - 3x - 2 ≥ 0) →
  [false, true, false, false] := by
sorry

end incorrect_props_l186_186728


namespace num_three_digit_integers_with_odd_factors_l186_186369

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l186_186369


namespace three_digit_oddfactors_count_is_22_l186_186101

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l186_186101


namespace largest_prime_divisor_36_squared_plus_81_squared_l186_186003

-- Definitions of the key components in the problem
def a := 36
def b := 81
def expr := a^2 + b^2
def largest_prime_divisor (n : ℕ) : ℕ := sorry -- Assume this function can compute the largest prime divisor

-- Theorem stating the problem
theorem largest_prime_divisor_36_squared_plus_81_squared : largest_prime_divisor (36^2 + 81^2) = 53 := 
  sorry

end largest_prime_divisor_36_squared_plus_81_squared_l186_186003


namespace number_of_three_digit_squares_l186_186437

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l186_186437


namespace problem_l186_186466

variable (x y : ℝ)
variable (h1 : 3^x = 36)
variable (h2 : 4^y = 36)

theorem problem (h1 : 3^x = 36) (h2 : 4^y = 36) : (2 / x) + (1 / y) = 1 :=
sorry

end problem_l186_186466


namespace percentage_of_B_students_is_correct_l186_186489

def scores : List ℕ := [88, 73, 55, 95, 76, 91, 86, 73, 76, 64, 85, 79, 72, 81, 89, 77]

def grading_scale (score : ℕ) : Char :=
  if 95 ≤ score ∧ score ≤ 100 then 'A'
  else if 87 ≤ score ∧ score ≤ 94 then 'B'
  else if 78 ≤ score ∧ score ≤ 86 then 'C'
  else if 70 ≤ score ∧ score ≤ 77 then 'D'
  else 'F'
  
def count_grades_in_range (scores : List ℕ) (low high : ℕ) : ℕ :=
  scores.count (λ score => low ≤ score ∧ score ≤ high)

def percentage_of_students_with_B (scores : List ℕ) : ℚ :=
  (count_grades_in_range scores 87 94).toRat / scores.length.toRat * 100

theorem percentage_of_B_students_is_correct :
  percentage_of_students_with_B scores = 12.5 :=
by
  sorry

end percentage_of_B_students_is_correct_l186_186489


namespace total_cost_of_breakfast_l186_186804

-- Definitions based on conditions
def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3
def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2
def kiera_muffins : ℕ := 2
def kiera_fruit_cup : ℕ := 1

-- The proof statement
theorem total_cost_of_breakfast : 
  muffin_cost * francis_muffins + 
  fruit_cup_cost * francis_fruit_cups + 
  muffin_cost * kiera_muffins + 
  fruit_cup_cost * kiera_fruit_cup = 17 := 
  by sorry

end total_cost_of_breakfast_l186_186804


namespace three_digit_integers_with_odd_number_of_factors_l186_186179

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l186_186179


namespace even_four_digit_numbers_count_l186_186755

-- Math problem translation: Prove that 156 is the number of valid four-digit even numbers
noncomputable def count_even_numbers : ℕ := 156

theorem even_four_digit_numbers_count :
  -- Define the conditions as sets or sets of rules
  let digits := {0, 1, 2, 3, 4, 5}
  ∃ count, count = count_even_numbers ∧ count = 156 :=
by
  sorry

end even_four_digit_numbers_count_l186_186755


namespace three_digit_integers_with_odd_factors_l186_186161

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l186_186161


namespace three_digit_perfect_squares_count_l186_186197

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l186_186197


namespace daria_weeks_needed_l186_186757

-- Defining the parameters and conditions
def initial_amount : ℕ := 20
def weekly_savings : ℕ := 10
def cost_of_vacuum_cleaner : ℕ := 120

-- Defining the total money Daria needs to add to her initial amount
def additional_amount_needed : ℕ := cost_of_vacuum_cleaner - initial_amount

-- Defining the number of weeks needed to save the additional amount, given weekly savings
def weeks_needed : ℕ := additional_amount_needed / weekly_savings

-- The theorem stating that Daria needs exactly 10 weeks to cover the expense of the vacuum cleaner
theorem daria_weeks_needed : weeks_needed = 10 := by
  sorry

end daria_weeks_needed_l186_186757


namespace three_digit_oddfactors_count_is_22_l186_186100

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l186_186100


namespace tan_alpha_sin_cos_alpha_l186_186047

-- First proof problem: Prove tan(α) = -12/5
theorem tan_alpha {a : ℝ} (h : a < 0) : 
  let x := 5 * a,
      y := -12 * a,
      tan_alpha := y / x in
  tan_alpha = -12 / 5 := by
  sorry

-- Second proof problem: Prove sin(α) + cos(α) = 7/13
theorem sin_cos_alpha {a : ℝ} (h : a < 0) :
  let x := 5 * a,
      y := -12 * a,
      r := -13 * a,
      sin_alpha := y / r,
      cos_alpha := x / r in
  sin_alpha + cos_alpha = 7 / 13 := by
  sorry

end tan_alpha_sin_cos_alpha_l186_186047


namespace point_coordinates_l186_186019

/-- Given the vector from point A to point B, if point A is the origin, then point B will have coordinates determined by the vector. -/
theorem point_coordinates (A B: ℝ × ℝ) (v: ℝ × ℝ) 
  (h: A = (0, 0)) (h_v: v = (-2, 4)) (h_ab: B = (A.1 + v.1, A.2 + v.2)): 
  B = (-2, 4) :=
by
  sorry

end point_coordinates_l186_186019


namespace problem1_problem2_l186_186859

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, -Real.sqrt 3 * Real.cos x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)

noncomputable def f (x : ℝ) : ℝ :=
  let dotProduct := (a x).1 * (b x).1 + (a x).2 * (b x).2
  dotProduct + 1

-- Problem 1: Prove the monotonically increasing interval for f(x)
theorem problem1 (k : ℤ) : ∀ x : ℝ, (x ∈ Icc (k * Real.pi + Real.pi / 3) (5 * Real.pi / 6 + k * Real.pi)) → increasing_on f (Icc (k * Real.pi + Real.pi / 3) (5 * Real.pi / 6 + k * Real.pi)) := sorry

-- Problem 2: Given θ ∈ (π/3, 2π/3) and f(θ) = 5/6, find sin(2θ)
theorem problem2 (θ : ℝ) (h1 : θ ∈ Ioo (Real.pi / 3) (2 * Real.pi / 3)) (h2 : f θ = 5 / 6) : Real.sin (2 * θ) = (2 * Real.sqrt 3 - Real.sqrt 5) / 6 := sorry

end problem1_problem2_l186_186859


namespace three_digit_integers_with_odd_factors_l186_186146

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l186_186146


namespace fraction_of_n_is_80_l186_186877

-- Definitions from conditions
def n := (5 / 6) * 240

-- The theorem we want to prove
theorem fraction_of_n_is_80 : (2 / 5) * n = 80 :=
by
  -- This is just a placeholder to complete the statement, 
  -- actual proof logic is not included based on the prompt instructions
  sorry

end fraction_of_n_is_80_l186_186877


namespace three_digit_integers_with_odd_number_of_factors_l186_186169

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l186_186169


namespace three_digit_oddfactors_count_is_22_l186_186090

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l186_186090


namespace number_of_three_digit_integers_with_odd_number_of_factors_l186_186330

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l186_186330


namespace three_digit_integers_with_odd_factors_l186_186160

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l186_186160


namespace total_students_l186_186718

theorem total_students (rank_right rank_left : ℕ) (h1 : rank_right = 16) (h2 : rank_left = 6) : rank_right + rank_left - 1 = 21 := by
  sorry

end total_students_l186_186718


namespace describe_set_T_l186_186748

-- Define the conditions for the set of points T
def satisfies_conditions (x y : ℝ) : Prop :=
  (x + 3 = 4 ∧ y < 7) ∨ (y - 3 = 4 ∧ x < 1)

-- Define the set T based on the conditions
def set_T := {p : ℝ × ℝ | satisfies_conditions p.1 p.2}

-- Statement to prove the geometric description of the set T
theorem describe_set_T :
  (∃ x y, satisfies_conditions x y) → ∃ p1 p2,
  (p1 = (1, t) ∧ t < 7 → satisfies_conditions 1 t) ∧
  (p2 = (t, 7) ∧ t < 1 → satisfies_conditions t 7) ∧
  (p1 ≠ p2) :=
sorry

end describe_set_T_l186_186748


namespace sin_double_angle_l186_186039

theorem sin_double_angle (α : ℝ) (h : Real.tan (Real.pi + α) = 2) : Real.sin (2 * α) = 4 / 5 := 
by 
  sorry

end sin_double_angle_l186_186039


namespace parabola_focus_vertex_l186_186933

theorem parabola_focus_vertex
  (V F A : ℝ)
  (AF : ℝ := 20)
  (AV : ℝ := 21)
  (FV_solutions : Set ℝ := {FV | ∃ (d1 d2: ℝ), d1 + d2 = FV ∧ (3 * d1^2 - 40 * d1 + 41 = 0) ∧ (3 * d2^2 - 40 * d2 + 41 = 0)}):
  (∀ FV ∈ FV_solutions, FV = 40 / 3) :=
begin
  sorry
end

end parabola_focus_vertex_l186_186933


namespace count_valid_3x3_grids_l186_186681

theorem count_valid_3x3_grids : 
  (∃ (grid : list (list ℕ)),
    grid.length = 3 ∧
    (∀ row ∈ grid, row.length = 3 ∧ ∀ n ∈ row, n ∈ [1, 2, 3]) ∧
    (∀ col ∈ (list.range 3), list.nodup (list.map (λ row, row.nth_le col (by simp)) grid)) ∧
    (∀ row ∈ grid, list.nodup row)
  ) →
  (finset.card (finset.filter 
    (λ grid : list (list ℕ),
      grid.length = 3 ∧
      (∀ row ∈ grid, row.length = 3 ∧ ∀ n ∈ row, n ∈ [1, 2, 3]) ∧
      (∀ col ∈ (list.range 3), list.nodup (list.map (λ row, row.nth_le col (by simp)) grid)) ∧
      (∀ row ∈ grid, list.nodup row)
    ) (finset.univ : finset (list (list ℕ))))) = 12 :=
by
  sorry

end count_valid_3x3_grids_l186_186681


namespace find_angle_C_l186_186895

variable (a b : ℝ) (S : ℝ) (C : ℝ)

-- Given conditions
def acute_triangle (a b S : ℝ) (C : ℝ) : Prop :=
  a = 3 ∧ b = 4 ∧ S = 3 * Real.sqrt 3 ∧ 0 < C ∧ C < π / 2

-- Theorem statement that angle C = π / 3 under above conditions
theorem find_angle_C (h : acute_triangle a b S C) : C = π / 3 :=
  sorry

end find_angle_C_l186_186895


namespace total_rainfall_2004_l186_186487

theorem total_rainfall_2004 (avg_2003 : ℝ) (increment : ℝ) (months : ℕ) (total_2004 : ℝ) 
  (h1 : avg_2003 = 41.5) 
  (h2 : increment = 2) 
  (h3 : months = 12) 
  (h4 : total_2004 = avg_2003 + increment * months) :
  total_2004 = 522 :=
by 
  sorry

end total_rainfall_2004_l186_186487


namespace distinct_five_digit_numbers_with_product_18_l186_186739

theorem distinct_five_digit_numbers_with_product_18 : 
  (∃ (digits : Fin 5 → ℕ), (∀ i, digits i ∈ (Finset.range 10)) ∧ (digits 0 * digits 1 * digits 2 * digits 3 * digits 4 = 18)) → 
  (Finset.univ.filter (λ (n : Fin 100000 → ℕ), (∀ i, n i ∈ (Finset.range 10)) ∧ 
    (n 0 * n 1 * n 2 * n 3 * n 4 = 18))).card = 60 :=
by
  sorry

end distinct_five_digit_numbers_with_product_18_l186_186739


namespace dealership_cars_count_l186_186888

-- Define the total number of cars as C
variable (C : ℝ)

-- Conditions
def sixty_percent_hybrids := 0.60 * C
def hybrids_with_full_headlights := 216

-- 40% of hybrids have one headlight, so 60% of hybrids have full headlights
def sixty_percent_of_hybrids := 0.60 * sixty_percent_hybrids

theorem dealership_cars_count :
  sixty_percent_of_hybrids = hybrids_with_full_headlights → 
  C = 600 :=
by
  intro H
  have : sixty_percent_of_hybrids = 0.36 * C := by
    unfold sixty_percent_of_hybrids
    unfold sixty_percent_hybrids
    ring
  rw [this] at H
  have eq_216 : 0.36 * C = 216 := by
    exact H
  have sol_C : C = 600 := by
    field_simp at eq_216
    linarith
  exact sol_C

end dealership_cars_count_l186_186888


namespace cost_of_tax_free_items_l186_186923

-- Definitions based on the conditions.
def total_spending : ℝ := 20
def sales_tax_percentage : ℝ := 0.30
def tax_rate : ℝ := 0.06

-- Derived calculations for intermediate variables for clarity
def taxable_items_cost : ℝ := total_spending * (1 - sales_tax_percentage)
def sales_tax_paid : ℝ := taxable_items_cost * tax_rate
def tax_free_items_cost : ℝ := total_spending - taxable_items_cost

-- Lean 4 statement for the problem
theorem cost_of_tax_free_items :
  tax_free_items_cost = 6 := by
    -- The proof would go here, but we are skipping it.
    sorry

end cost_of_tax_free_items_l186_186923


namespace three_digit_integers_with_odd_factors_l186_186237

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l186_186237


namespace Daria_vacuum_cleaner_problem_l186_186760

theorem Daria_vacuum_cleaner_problem (initial_savings weekly_savings target_savings weeks_needed : ℕ)
  (h1 : initial_savings = 20)
  (h2 : weekly_savings = 10)
  (h3 : target_savings = 120)
  (h4 : weeks_needed = (target_savings - initial_savings) / weekly_savings) : 
  weeks_needed = 10 :=
by
  sorry

end Daria_vacuum_cleaner_problem_l186_186760


namespace activity_order_l186_186733

-- Define the fractions for each activity
def dodgeball_popularity := 13 / 40
def nature_walk_popularity := 8 / 25
def painting_popularity := 9 / 20

-- Define a function that finds the common denominator and converts fractions
noncomputable def fractions_to_common_denominator (f1 f2 f3 : ℚ) : (ℚ × ℚ × ℚ) :=
  let common_denominator := 200
  (f1 * (common_denominator / f1.den),
   f2 * (common_denominator / f2.den),
   f3 * (common_denominator / f3.den))

-- Proof statement: Verify that the order of popularities is as specified
theorem activity_order :
  let (d, n, p) := fractions_to_common_denominator dodgeball_popularity nature_walk_popularity painting_popularity in
  p > d ∧ d > n := 
by
  -- Ensuring that the theorem structure is correct.
  sorry

end activity_order_l186_186733


namespace three_digit_integers_odd_factors_count_l186_186457

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l186_186457


namespace three_digit_perfect_squares_count_l186_186200

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l186_186200


namespace cistern_width_l186_186687

theorem cistern_width (l h w total_surface_area : ℝ) (hl : l = 6) (hh : h = 1.25) (ht : total_surface_area = 49) :
  6 * w + 2 * 6 * 1.25 + 2 * w * 1.25 = 49 → w = 4 :=
by
  intros h_eq
  simp_all only [hl, hh, ht]
  rw [mul_add, add_assoc, ←mul_assoc, mul_comm 6]
  simp at h_eq
  rw [add_assoc, add_comm, add_sub_cancel, div_eq_inv_mul]
  simp at h_eq
  rw [h_eq]
  exact le_antisymm (le_refl 4) (le_refl 4)
  sorry

end cistern_width_l186_186687


namespace three_digit_odds_factors_count_l186_186345

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l186_186345


namespace flint_pouches_l186_186797

theorem flint_pouches (n : ℕ) (h : n = 60) : (∀ k ∈ {2, 3, 4, 5}, n % k = 0) → n = 60 :=
by 
  intros k hk
  simp at hk
  fin_cases hk
  case Decidable.isTrue 2 {exact dec_trivial}
  case Decidable.isTrue 3 {exact dec_trivial}
  case Decidable.isTrue 4 {exact dec_trivial}
  case Decidable.isTrue 5 {exact dec_trivial}
  sorry

end flint_pouches_l186_186797


namespace num_three_digit_ints_with_odd_factors_l186_186251

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l186_186251


namespace tan_diff_l186_186871

-- Declaration of the given condition
def cond (α : ℝ) : Prop := 2 * Real.tan α = 3 * Real.tan (Real.pi / 8)

-- The main property to prove
theorem tan_diff (α : ℝ) (h : cond α) : 
  Real.tan (α - Real.pi / 8) = (5 * Real.sqrt 2 + 1) / 49 :=
sorry

end tan_diff_l186_186871


namespace yard_length_l186_186638

theorem yard_length (number_of_trees distance_between_trees : ℕ) (h1 : number_of_trees = 14) (h2 : distance_between_trees = 21) : 
  (number_of_trees - 1) * distance_between_trees = 273 := by
  have num_spaces := number_of_trees - 1
  have h3 : num_spaces = 13 := by norm_num [h1]
  calc
    num_spaces * distance_between_trees = 13 * 21 : by rw [h3, h2]
                         ... = 273 : by norm_num
  sorry

end yard_length_l186_186638


namespace three_digit_integers_with_odd_factors_l186_186150

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l186_186150


namespace area_of_figure_M_l186_186512

-- Defining the condition predicates
def cond1 (y : ℝ) : Prop := |y| + |4 - y| ≤ 4
def cond2 (x y : ℝ) : Prop := (y^2 + x - 4y + 1)/(2y + x - 7) ≤ 0

-- Define the set of points that satisfy both conditions
def figure_M : set (ℝ × ℝ) := {p : ℝ × ℝ | cond1 p.2 ∧ cond2 p.1 p.2}

-- Proving that the area of figure M is 8
theorem area_of_figure_M : area figure_M = 8 :=
sorry

end area_of_figure_M_l186_186512


namespace prob_at_least_one_even_l186_186684

theorem prob_at_least_one_even :
  let events := [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)] in
  let success_events := [(1,2), (2,1), (2,2), (2,3), (3,2)] in
  (success_events.length / events.length : ℚ) = 5 / 9 :=
by
  sorry

end prob_at_least_one_even_l186_186684


namespace solve_system_l186_186592

theorem solve_system (x y : ℝ) :
  (sqrt (5 * x * (x + 2)) + sqrt (5 * y * (x + 2)) = 2 * sqrt (2 * (x + y) * (x + 2))) ∧
  (x * y = 9) →
  (x, y) = (-9, -1) ∨ (x, y) = (-2, -9/2) ∨ (x, y) = (1, 9) ∨ (x, y) = (9, 1) :=
by
  sorry

end solve_system_l186_186592


namespace Yankees_to_Mets_ratio_l186_186495

theorem Yankees_to_Mets_ratio (Y M B : ℕ) 
  (h1 : M = 88) 
  (h2 : Y + M + B = 330) 
  (h3 : 4 * B = 5 * M) :
  Y = 3 * M / 2 :=
by 
  have hM : M = 88 := h1
  have hB : B = (5 * M / 4) := by
    rw h1 at h3
    exact Nat.div_eq_of_eq_mul_sub h3 rfl
  have hY : Y = 330 - M - B := sub_eq_iff_eq_add.mpr (eq_comm.mp h2)
  rw [h1, hB] at hY
  have hB_eq : B = 110 := by
    calc B = (5 * 88 / 4) : by { rw h1 }
       ... = 110 : by norm_num
  rw [hM, hB_eq] at hY
  exact int.coe_nat_div 3 88

end Yankees_to_Mets_ratio_l186_186495


namespace num_three_digit_integers_with_odd_factors_l186_186371

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l186_186371


namespace three_digit_integers_with_odd_factors_l186_186383

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l186_186383


namespace triangle_medians_rational_l186_186630

theorem triangle_medians_rational
  (a b c : ℕ)
  (a_eq : a = 8)
  (b_eq : b = 9)
  (c_eq : c = 11) :
  let s_a := Real.sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4)
  let s_b := Real.sqrt ((2 * c^2 + 2 * a^2 - b^2) / 4)
  let s_c := Real.sqrt ((2 * a^2 + 2 * b^2 - c^2) / 4)
  Rational (s_b) ∧ Rational (s_c) :=
by
  sorry

end triangle_medians_rational_l186_186630


namespace projection_unique_same_result_l186_186858

def a : Vector ℝ 3 := ![-1, 4, 2]
def b : Vector ℝ 3 := ![3, -1, 5]
def v : Vector ℝ 3 := sorry -- unspecified direction vector for projection

theorem projection_unique_same_result (p : Vector ℝ 3) (h_proj_a : proj v a = p) (h_proj_b : proj v b = p) : 
  p = ![11/25, 11/5, 77/25] := 
begin
  sorry
end

end projection_unique_same_result_l186_186858


namespace num_people_visited_iceland_l186_186497

noncomputable def total := 100
noncomputable def N := 43  -- Number of people who visited Norway
noncomputable def B := 61  -- Number of people who visited both Iceland and Norway
noncomputable def Neither := 63  -- Number of people who visited neither country
noncomputable def I : ℕ := 55  -- Number of people who visited Iceland (need to prove)

-- Lean statement to prove
theorem num_people_visited_iceland : I = total - Neither + B - N := by
  sorry

end num_people_visited_iceland_l186_186497


namespace number_of_three_digit_integers_with_odd_factors_l186_186414

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l186_186414


namespace three_digit_integers_with_odd_factors_l186_186401

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l186_186401


namespace three_digit_perfect_squares_count_l186_186188

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l186_186188


namespace percentage_less_l186_186874

theorem percentage_less (x y : ℝ) (h : x = 1.3 * y) : 
  ∃ p : ℝ, p ≈ 23.08 ∧ p = 100 * ((x - y) / x) := 
by 
  sorry

end percentage_less_l186_186874


namespace num_odd_factors_of_three_digit_integers_eq_22_l186_186116

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l186_186116


namespace train_speed_calculation_train_speed_correct_l186_186668

def train_length : ℝ := 210
def crossing_time : ℝ := 13
def train_speed : ℝ := 16.15

theorem train_speed_calculation (d t s : ℝ) (h1 : d = train_length) (h2 : t = crossing_time) :
  s = d / t :=
sorry

theorem train_speed_correct : train_speed = train_length / crossing_time :=
by 
  have : train_length = 210 := rfl
  have : crossing_time = 13 := rfl
  rw [this, this]
  exact calc
    train_speed = 210 / 13 : rfl
               ... = 16.15 : rfl
               ... = train_length / crossing_time : by rw [←this, ←this]

end train_speed_calculation_train_speed_correct_l186_186668


namespace moles_of_naoh_needed_l186_186791

-- Define the chemical reaction
def balanced_eqn (nh4no3 naoh nano3 nh4oh : ℕ) : Prop :=
  nh4no3 = naoh ∧ nh4no3 = nano3

-- Theorem stating the moles of NaOH required to form 2 moles of NaNO3 from 2 moles of NH4NO3
theorem moles_of_naoh_needed (nh4no3 naoh nano3 nh4oh : ℕ) (h_balanced_eqn : balanced_eqn nh4no3 naoh nano3 nh4oh) 
  (h_nano3: nano3 = 2) (h_nh4no3: nh4no3 = 2) : naoh = 2 :=
by
  unfold balanced_eqn at h_balanced_eqn
  sorry

end moles_of_naoh_needed_l186_186791


namespace breakfast_cost_l186_186808

def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3

def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2

def kiera_muffins : ℕ := 2
def kiera_fruit_cup : ℕ := 1

theorem breakfast_cost :
  muffin_cost * francis_muffins + fruit_cup_cost * francis_fruit_cups
  + muffin_cost * kiera_muffins + fruit_cup_cost * kiera_fruit_cup = 17 :=
by
  -- skipping proof
  sorry

end breakfast_cost_l186_186808


namespace santino_fruit_total_l186_186581

-- Definitions of the conditions
def numPapayaTrees : ℕ := 2
def numMangoTrees : ℕ := 3
def papayasPerTree : ℕ := 10
def mangosPerTree : ℕ := 20
def totalFruits (pTrees : ℕ) (pPerTree : ℕ) (mTrees : ℕ) (mPerTree : ℕ) : ℕ :=
  (pTrees * pPerTree) + (mTrees * mPerTree)

-- Theorem that states the total number of fruits is 80 given the conditions
theorem santino_fruit_total : totalFruits numPapayaTrees papayasPerTree numMangoTrees mangosPerTree = 80 := 
  sorry

end santino_fruit_total_l186_186581


namespace three_digit_integers_with_odd_factors_l186_186390

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l186_186390


namespace part_a_part_d_l186_186608

-- Definition of quaternion multiplication and properties

axiom i_square : i * i = -1
axiom j_square : j * j = -1
axiom k_square : k * k = -1
axiom i_zero : i ^ 0 = 1
axiom j_zero : j ^ 0 = 1
axiom k_zero : k ^ 0 = 1
axiom ij_k : i * j = k
axiom ji_neg_k : j * i = -k
axiom jk_i : j * k = i
axiom kj_neg_i : k * j = -i
axiom ki_j : k * i = j
axiom ik_neg_j : i * k = -j

-- Representation of quaternions
structure quaternion :=
(a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)

-- Definition of α and β
def alpha : quaternion := ⟨a₁, b₁, c₁, d₁⟩
def beta : quaternion := ⟨a₂, b₂, c₂, d₂⟩

-- Quaternion multiplication
noncomputable def quaternion_mul (α β : quaternion) : quaternion :=
⟨ α.a * β.a - α.b * β.b - α.c * β.c - α.d * β.d,
  α.a * β.b + α.b * β.a + α.c * β.d - α.d * β.c,
  α.a * β.c + α.c * β.a + α.d * β.b - α.b * β.d,
  α.a * β.d + α.d * β.a + α.b * β.c - α.c * β.b ⟩

-- Problem statements in Lean
theorem part_a : i * j * k = -1 := 
by apply sorry

theorem part_d (alpha_beta: quaternion := quaternion_mul (⟨1, 1, 1, 1⟩) beta) : 
  alpha = ⟨1, 1, 1, 1⟩ ∧ alpha_beta = ⟨4, 0, 0, 0⟩ → beta = ⟨1, -1, -1, -1⟩ := 
by apply sorry

end part_a_part_d_l186_186608


namespace maximize_triangle_area_l186_186822

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  (y^2 / 4) + (x^2 / 2) = 1

theorem maximize_triangle_area
  (b : ℝ) (hb: 0 < b ∧ b < 2)
  (ecc_ellipse : ℝ := 1 / real.sqrt 2)
  (ecc_hyperbola : ℝ := real.sqrt 2)
  (line1 line2 : ℝ → ℝ)
  (line1_eq : ∀ x, line1 x = real.sqrt 2 * x + 2)
  (line2_eq : ∀ x, line2 x = real.sqrt 2 * x - 2)
  (P : ℝ × ℝ)
  (hP : P.1 = 1 ∧ P.2 > 0)
  (hP_on_ellipse : ellipse_equation P.1 P.2) :
  ∃ (m : ℝ),
    (∀ x, (line1 x - P.2) * P.1 = m * (x - P.1)) ∨
    (∀ x, (line2 x - P.2) * P.1 = m * (x - P.1)) :=
sorry

end maximize_triangle_area_l186_186822


namespace plain_pancakes_l186_186762

/-- Define the given conditions -/
def total_pancakes : ℕ := 67
def blueberry_pancakes : ℕ := 20
def banana_pancakes : ℕ := 24

/-- Define a theorem stating the number of plain pancakes given the conditions -/
theorem plain_pancakes : total_pancakes - (blueberry_pancakes + banana_pancakes) = 23 := by
  -- Here we will provide a proof
  sorry

end plain_pancakes_l186_186762


namespace hourly_rate_for_carriage_l186_186922

theorem hourly_rate_for_carriage
  (d : ℕ) (s : ℕ) (f : ℕ) (c : ℕ)
  (h_d : d = 20)
  (h_s : s = 10)
  (h_f : f = 20)
  (h_c : c = 80) :
  (c - f) / (d / s) = 30 := by
  sorry

end hourly_rate_for_carriage_l186_186922


namespace three_digit_integers_with_odd_factors_l186_186388

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l186_186388


namespace num_three_digit_ints_with_odd_factors_l186_186250

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l186_186250


namespace trigonometric_identity_l186_186817

theorem trigonometric_identity 
  (x : ℝ) 
  (h : Real.sin (x + (Real.pi / 6)) = 1 / 4) : 
  Real.sin (5 * Real.pi / 6 - x) + Real.cos (Real.pi / 3 - x) ^ 2 = 5 / 16 := 
sorry

end trigonometric_identity_l186_186817


namespace three_digit_integers_with_odd_factors_l186_186219

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l186_186219


namespace num_odd_factors_of_three_digit_integers_eq_22_l186_186107

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l186_186107


namespace num_odd_factors_of_three_digit_integers_eq_22_l186_186110

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l186_186110


namespace net_amount_received_l186_186598

variable (sale_amount : ℝ)
variable (brokerage_rate : ℝ)

theorem net_amount_received (h1 : sale_amount = 108.25) (h2 : brokerage_rate = 1 / 4 / 100) :
  sale_amount - (sale_amount * brokerage_rate).round(2) = 107.98 :=
by
  sorry

end net_amount_received_l186_186598


namespace three_digit_integers_odd_factors_count_l186_186448

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l186_186448


namespace multiples_of_10_5_l186_186675

theorem multiples_of_10_5 (n : ℤ) (h1 : ∀ k : ℤ, k % 10 = 0 → k % 5 = 0) (h2 : n % 10 = 0) : n % 5 = 0 := 
by
  sorry

end multiples_of_10_5_l186_186675


namespace equation_of_line_MN_l186_186832

noncomputable def parabola_focus_1 : (ℝ × ℝ) := (2, real.sqrt 3)

def directrix_1 (x : ℝ) : Prop := x = 0

noncomputable def parabola_focus_2 : (ℝ × ℝ) := (2, real.sqrt 3)

def directrix_2 (x y : ℝ) : Prop := x - real.sqrt 3 * y = 0

def intersects (M N : ℝ × ℝ) : Prop :=
  let (x1, y1) := M
  let (x2, y2) := N
  (parabola_focus_1, directrix_1 x1, parabola_focus_2, directrix_2 x2 y2) 

theorem equation_of_line_MN (M N : ℝ × ℝ) :
  intersects M N → 
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c = 0 ∧ (a = real.sqrt 3) ∧ (b = -1) → 
  equation line MN is real.sqrt 3 * x - y = 0 := 
sorry

end equation_of_line_MN_l186_186832


namespace number_of_three_digit_squares_l186_186429

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l186_186429


namespace number_of_three_digit_squares_l186_186434

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l186_186434


namespace three_digit_integers_with_odd_factors_l186_186311

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l186_186311


namespace three_digit_odds_factors_count_l186_186343

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l186_186343


namespace pascal_sum_diff_correct_l186_186749

noncomputable def pascal_sum_diff : ℝ :=
  let n := 2005
  let a_i (i : ℕ) := Nat.choose (n - 1) i
  let b_i (i : ℕ) := Nat.choose n i
  let c_i (i : ℕ) := Nat.choose (n + 1) i
  (∑ i in Finset.range (2005 + 1), (b_i i : ℝ) / (c_i i) : ℝ)
  - (∑ i in Finset.range (2004 + 1), (a_i i : ℝ) / (b_i i) : ℝ)

theorem pascal_sum_diff_correct : pascal_sum_diff = 0.5 := by
  sorry

end pascal_sum_diff_correct_l186_186749


namespace num_odd_factors_of_three_digit_integers_eq_22_l186_186117

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l186_186117


namespace zero_point_in_interval_l186_186635

noncomputable def f (x : ℝ) : ℝ := 3^x - 4

theorem zero_point_in_interval : ∃ x ∈ set.Ioo 1 2, f x = 0 :=
by sorry

end zero_point_in_interval_l186_186635


namespace attraction_ticket_cost_l186_186511

theorem attraction_ticket_cost
  (cost_park_entry : ℕ)
  (cost_attraction_parent : ℕ)
  (total_paid : ℕ)
  (num_children : ℕ)
  (num_parents : ℕ)
  (num_grandmother : ℕ)
  (x : ℕ)
  (h_costs : cost_park_entry = 5)
  (h_attraction_parent : cost_attraction_parent = 4)
  (h_family : num_children = 4 ∧ num_parents = 2 ∧ num_grandmother = 1)
  (h_total_paid : total_paid = 55)
  (h_equation : (num_children + num_parents + num_grandmother) * cost_park_entry + (num_parents + num_grandmother) * cost_attraction_parent + num_children * x = total_paid) :
  x = 2 := by
  sorry

end attraction_ticket_cost_l186_186511


namespace arcsin_add_arccos_eq_pi_div_two_l186_186648

open Real

theorem arcsin_add_arccos_eq_pi_div_two (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  arcsin x + arccos x = (π / 2) :=
sorry

end arcsin_add_arccos_eq_pi_div_two_l186_186648


namespace trapezium_area_l186_186783

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 13) :
  (1 / 2) * (a + b) * h = 247 :=
by
  sorry

end trapezium_area_l186_186783


namespace three_digit_integers_with_odd_number_of_factors_count_l186_186269

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l186_186269


namespace three_digit_integers_with_odd_factors_l186_186152

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l186_186152


namespace three_digit_oddfactors_count_l186_186124

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l186_186124


namespace three_digit_integers_with_odd_number_of_factors_count_l186_186280

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l186_186280


namespace three_digit_integers_odd_factors_count_l186_186451

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l186_186451


namespace power_identity_l186_186986

theorem power_identity (a b : ℕ) (R S : ℕ) (hR : R = 2^a) (hS : S = 5^b) : 
    20^(a * b) = R^(2 * b) * S^a := 
by 
    -- Insert the proof here
    sorry

end power_identity_l186_186986


namespace three_digit_integers_with_odd_factors_l186_186231

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l186_186231


namespace faster_train_speed_l186_186647

/-- Mathematical Problem:
Two trains, each 100m long, moving in opposite directions, cross each other in 12 seconds. If one is moving twice as fast as the other, what is the speed of the faster train?
--/
theorem faster_train_speed (v : ℝ) :
  let train_length := 100,
      crossing_time := 12,
      total_distance := train_length + train_length,
      relative_speed := 3 * v in
  (relative_speed = total_distance / crossing_time) →
  (2 * v = 11.1) :=
begin
  sorry
end

end faster_train_speed_l186_186647


namespace number_of_cuboids_painted_l186_186463

-- Define the problem conditions
def painted_faces (total_faces : ℕ) (faces_per_cuboid : ℕ) : ℕ :=
  total_faces / faces_per_cuboid

-- Define the theorem to prove
theorem number_of_cuboids_painted (total_faces : ℕ) (faces_per_cuboid : ℕ) :
  total_faces = 48 → faces_per_cuboid = 6 → painted_faces total_faces faces_per_cuboid = 8 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end number_of_cuboids_painted_l186_186463


namespace three_digit_perfect_squares_count_l186_186182

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l186_186182


namespace petya_sum_invariant_l186_186966

theorem petya_sum_invariant (x y z : ℕ) :
  ∃ t : ℕ, (∀ k : ℕ, k < t → 
    (∃ i j : ℕ, i ≠ j ∧ i < 3 ∧ j < 3 ∧
     ((x + y + z = k * 3 + a) ∧ (x*y*z = k * 2 + b ∧ 
      ∃ a b c : ℕ, 
      (a = x - k ∨ a = y - k ∨ a = z - k) ∧
      (b = x * y ∧ c = z ∨
       b = y * z ∧ c = x ∨ 
       b = x * z ∧ c = y)))) ∧
    (∃ i j : ℕ, i ≠ j ∧ i < 3 ∧ j < 3 ∧
     x * y * z = k ∧
     ∃ p : ℕ, p = x * y * z))) :=
sorry

end petya_sum_invariant_l186_186966


namespace solve_for_m_l186_186472

-- Define the conditions as hypotheses
def hyperbola_equation (x y : Real) (m : Real) : Prop :=
  (x^2)/(m+9) + (y^2)/9 = 1

def eccentricity (e : Real) (a b : Real) : Prop :=
  e = 2 ∧ e^2 = 1 + (b^2)/(a^2)

-- Prove that m = -36 given the conditions
theorem solve_for_m (m : Real) (h : hyperbola_equation x y m) (h_ecc : eccentricity 2 3 (Real.sqrt (-(m+9)))) :
  m = -36 :=
sorry

end solve_for_m_l186_186472


namespace angle_ABO_eq_3pi_over_7_l186_186649

noncomputable theory

open_locale classical

variables (A B C D O E F : Type) [IsCircle O]
variables (on_circle_B : IsVertexOnCircle B O)
variables (on_circle_C : IsVertexOnCircle C O)
variables (on_circle_D : IsVertexOnCircle D O)
variables (intersects_F : IntersectsAt A B F)
variables (intersects_E : IntersectsAt A D E)
variables (angle_BAD_right : angle BAD = π / 2)
variables (chord_EF_eq_FB : chord EF = chord FB)
variables (chords_eq_BC_CD_ED : ∀ {P Q}, (P = C ∧ Q = D) ∨ (P = B ∧ Q = C) ∨ (P = D ∧ Q = E) → chord P Q = chord C D)

theorem angle_ABO_eq_3pi_over_7 :
  angle ABO = 3 * π / 7 :=
sorry

end angle_ABO_eq_3pi_over_7_l186_186649


namespace num_three_digit_integers_with_odd_factors_l186_186364

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l186_186364


namespace min_table_sum_l186_186557

-- Define the sequence x_k
def x_seq (k : ℕ) (h_k : k ≤ 100) : ℕ := sorry

-- Define the table entry
def table_entry (i k : ℕ) (h_i : i ≤ 80) (h_k : k ≤ 80) : ℝ :=
  Real.log (x_seq k sorry) (x_seq i sorry / 16)

-- Define the sum of table entries
def table_sum : ℝ :=
  ∑ i in Finset.range 80, ∑ k in Finset.range 80, table_entry i k sorry sorry

-- The theorem stating the minimum sum
theorem min_table_sum (h : ∀ k, x_seq k sorry > 1) : table_sum = -19200 :=
by
  sorry

end min_table_sum_l186_186557


namespace fixed_point_l186_186997

-- Given the function f(x) = a^(x-1) + 4
def f (a : ℝ) (x : ℝ) := a^(x - 1) + 4

-- The coordinates of point P through which the graph of the function f always passes are (1, 5)
theorem fixed_point (a : ℝ) : f a 1 = 5 :=
by {
  -- The precise steps of the proof are omitted as per instructions
  sorry
}

end fixed_point_l186_186997


namespace area_of_shaded_region_l186_186802

theorem area_of_shaded_region (perimeter_A perimeter_D : ℕ) (sum_sides_BC diff_sides_BC : ℕ)
  (h1 : perimeter_A = 12) (h2 : perimeter_D = 60) (h3 : sum_sides_BC = 15) (h4 : diff_sides_BC = 3) :
  let side_A := perimeter_A / 4,
      side_D := perimeter_D / 4,
      side_B := (sum_sides_BC + diff_sides_BC) / 2,
      side_C := (sum_sides_BC - diff_sides_BC) / 2
  in
  (1 / 2 * side_B * side_B + 1 / 2 * side_C * side_C) = 58.5 :=
by
  -- Here goes the proof, which is not required according to the instructions
  -- so we replace it with sorry.
  sorry

end area_of_shaded_region_l186_186802


namespace fraction_difference_739_999_l186_186540

theorem fraction_difference_739_999 :
  let F : ℚ := 739 / 999 in
  999 - 739 = 260 :=
by { rw [nat.sub_self], refl; sorry }

end fraction_difference_739_999_l186_186540


namespace number_of_three_digit_squares_l186_186428

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l186_186428


namespace number_of_three_digit_integers_with_odd_factors_l186_186404

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l186_186404


namespace find_number_l186_186470

open Real

def equation_satisfied (x : ℝ) (num : ℝ) : Prop :=
  (27 / 4) * x - 18 = 3 * x + num

theorem find_number : ∃ num : ℝ, equation_satisfied 12 num ∧ num = 27 :=
by
  use 27
  split
  · sorry
  · rfl

end find_number_l186_186470


namespace B_time_to_finish_race_l186_186490

theorem B_time_to_finish_race (t : ℝ) 
  (race_distance : ℝ := 130)
  (A_time : ℝ := 36)
  (A_beats_B_by : ℝ := 26)
  (A_speed : ℝ := race_distance / A_time) 
  (B_distance_when_A_finishes : ℝ := race_distance - A_beats_B_by) 
  (B_speed := B_distance_when_A_finishes / t) :
  B_speed * (t - A_time) = A_beats_B_by → t = 48 := 
by
  intros h
  sorry

end B_time_to_finish_race_l186_186490


namespace most_economical_cost_l186_186616

noncomputable def paving_cost : ℝ → ℝ → ℝ → ℝ := λ length width pillar_diameter =>
  let total_area := length * width
  let pillar_radius := pillar_diameter / 2
  let pillar_area := π * (pillar_radius ^ 2)
  let effective_area := total_area - pillar_area
  let ceramic_cost := effective_area * 700
  ceramics_cost

theorem most_economical_cost : paving_cost 5.5 4 0.5 = 15260 := by
  sorry

end most_economical_cost_l186_186616


namespace number_of_squares_in_figure_50_l186_186774

-- Define the given data for the initial figures.
def squares_in_figures : ℕ → ℕ
| 0 := 1
| 1 := 6
| 2 := 15
| 3 := 28
| n := 2 * n^2 + 3 * n + 1  -- Continuation of the pattern (quadratic sequence)

theorem number_of_squares_in_figure_50 :
  squares_in_figures 50 = 5151 :=
by
  have h0 : squares_in_figures 0 = 1 := by rfl
  have h1 : squares_in_figures 1 = 6 := by rfl
  have h2 : squares_in_figures 2 = 15 := by rfl
  have h3 : squares_in_figures 3 = 28 := by rfl
  have h50 : squares_in_figures 50 = 2 * 50^2 + 3 * 50 + 1 := by simp
  have h_eq : 2 * 50^2 + 3 * 50 + 1 = 5151 := by norm_num
  rw h50
  exact h_eq

end number_of_squares_in_figure_50_l186_186774


namespace percentage_increase_first_year_l186_186016

theorem percentage_increase_first_year (P : ℝ) (X : ℝ) 
  (h1 : P * (1 + X / 100) * 0.75 * 1.15 = P * 1.035) : 
  X = 20 :=
by
  sorry

end percentage_increase_first_year_l186_186016


namespace f_diff_bounds_f_eq_1025_l186_186610

def f : ℕ → ℝ := sorry

axiom f_base_1 : f 1 = 1
axiom f_base_2 : f 2 = 2
axiom f_recursive (n : ℕ) : f (n + 2) = f (n + 2 - (f (n + 1))) + f (n + 1 - (f n))

theorem f_diff_bounds (n : ℕ) : 0 ≤ f (n + 1) - f n ∧ f (n + 1) - f n ≤ 1 := sorry

theorem f_eq_1025 (n : ℕ) : f n = 1025 := sorry

end f_diff_bounds_f_eq_1025_l186_186610


namespace three_digit_odds_factors_count_l186_186354

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l186_186354


namespace cone_radius_l186_186836

theorem cone_radius (r l : ℝ)
  (h1 : 6 * Real.pi = Real.pi * r^2 + Real.pi * r * l)
  (h2 : 2 * Real.pi * r = Real.pi * l) :
  r = Real.sqrt 2 :=
by
  sorry

end cone_radius_l186_186836


namespace three_digit_integers_with_odd_factors_l186_186208

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l186_186208


namespace eggs_from_gertrude_l186_186919

-- Definitions of given conditions
def eggs_from_blanche := 3
def eggs_from_nancy := 2
def eggs_from_martha := 2
def dropped_eggs := 2
def eggs_left := 9

-- Theorem to prove
theorem eggs_from_gertrude : 
  let total_eggs := eggs_left + dropped_eggs in
  let eggs_from_others := eggs_from_blanche + eggs_from_nancy + eggs_from_martha in
  total_eggs - eggs_from_others = 4 :=
by {
  sorry
}

end eggs_from_gertrude_l186_186919


namespace three_digit_integers_with_odd_factors_count_l186_186290

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l186_186290


namespace three_digit_oddfactors_count_is_22_l186_186085

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l186_186085


namespace conic_section_properties_l186_186850

theorem conic_section_properties
  (C : set (ℝ × ℝ))
  (A : ℝ × ℝ)
  (F1 F2 : ℝ × ℝ)
  (ρ θ : ℝ)
  (line : set (ℝ × ℝ))
  (M N : ℝ × ℝ)
  (foci : C = {p : ℝ × ℝ | 3 * p.1^2 + 4 * p.2^2 = 12})
  (polar_eq : ρ^2 = 12 / (3 + sin θ^2))
  (A_coords : A = (0, -sqrt 3))
  (F1_coords : F1 = (-1, 0))
  (F2_coords : F2 = (1, 0))
  (line_eq : ∀ p ∈ line, p.snd = sqrt 3 * (p.fst + 1)) :
  (∀ p ∈ C, 3 * p.1^2 + 4 * p.2^2 = 12) ∧
  (∀ p ∈ line, p.snd = sqrt 3 * (p.fst + 1)) ∧
  |(F1_coords.1 - M.1)^2 + (F1_coords.2 - M.2)^2| * |(F1_coords.1 - N.1)^2 + (F1_coords.2 - N.2)^2| =
  12 / 5 :=
sorry

end conic_section_properties_l186_186850


namespace tim_math_score_l186_186642

theorem tim_math_score : (List.range 11).map (λ n => 2 * (n + 1)).sum = 132 :=
by
  sorry

end tim_math_score_l186_186642


namespace original_game_no_chance_chief_reclaim_false_chief_knew_expulsions_true_maximum_expulsions_six_natives_cannot_lose_second_game_l186_186706

noncomputable def original_no_chance (n : ℕ) (trades : ℕ) : Prop :=
  (n = 30) → (trades > 15) → (∃ (a b : ℕ), a ≠ b ∧ a = b)

noncomputable def chief_cannot_reclaim (remaining : ℕ) (coins : ℕ) : Prop :=
  coins = 270 → (remaining = 30) → (∀ x : ℕ, remaining - x = unique_coins → redistributed_coins ≤ 270) → False

noncomputable def chief_knew_expulsions (expelled : ℕ) : Prop :=
  (expelled = 6) → True

noncomputable def max_expulsions (total : ℕ) (remaining : ℕ) (coins : ℕ) : Prop :=
  remaining = total - 6 ∧ coins = 270

noncomputable def merchant_lost_first (lost_first : Prop) (second_game_result : Prop) : Prop :=
  lost_first → (second_game_result = False)

-- Statement
theorem original_game_no_chance : original_no_chance 30 435 := sorry

theorem chief_reclaim_false : chief_cannot_reclaim 30 270 := sorry

theorem chief_knew_expulsions_true : chief_knew_expulsions 6 := sorry

theorem maximum_expulsions_six : max_expulsions 30 24 270 := sorry

theorem natives_cannot_lose_second_game : merchant_lost_first True False := sorry

end original_game_no_chance_chief_reclaim_false_chief_knew_expulsions_true_maximum_expulsions_six_natives_cannot_lose_second_game_l186_186706


namespace amount_lent_by_A_to_B_l186_186698

theorem amount_lent_by_A_to_B
  (P : ℝ)
  (H1 : P * 0.115 * 3 - P * 0.10 * 3 = 1125) :
  P = 25000 :=
by
  sorry

end amount_lent_by_A_to_B_l186_186698


namespace other_cube_side_length_l186_186998

theorem other_cube_side_length (s_1 s_2 : ℝ) (h1 : s_1 = 1) (h2 : 6 * s_2^2 / 6 = 36) : s_2 = 6 :=
by
  sorry

end other_cube_side_length_l186_186998


namespace painting_count_l186_186682

def is_valid_painting (grid : ℕ → ℕ → bool) : Prop :=
  ∀ i j, grid i j = tt →
    (i = 0 ∨ grid (i-1) j ≠ ff) ∧
    (i = 2 ∨ grid (i+1) j ≠ ff) ∧
    (j = 0 ∨ grid i (j-1) ≠ ff) ∧
    (j = 2 ∨ grid i (j+1) ≠ ff)

theorem painting_count :
  ∃ (count : ℕ), count = 4 ∧
    (∀ (grid : ℕ → ℕ → bool), is_valid_painting grid) :=
sorry

end painting_count_l186_186682


namespace trajectory_of_M_max_value_of_S_l186_186901

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
  {p | p.fst^2 / a^2 + p.snd^2 / b^2 = 1}

def S (m : ℝ) : ℝ :=
  let t := 3 * m^2 + 4 in
  6 * real.sqrt (1 - 1 / (3 * t))

theorem trajectory_of_M :
  let F1 := (-1, 0)
  let F2 := (1, 0)
  (∀ (M : ℝ × ℝ),
    midpoint F1 M = F2)
  →
  (midpoint F1 F2) ∈ ellipse 2 (real.sqrt 3)
  → (∀ p : ℝ × ℝ, p ∈ ellipse 2 (real.sqrt 3) → p.snd ≠ 0) :=
by
  intros F1 F2 M midpoint_cond ellipse_cond
  sorry

theorem max_value_of_S :
  ∃ m : ℝ, 
  (S m = real.sqrt 33) :=
by
  exists 2 / real.sqrt 3
  sorry

end trajectory_of_M_max_value_of_S_l186_186901


namespace solution_set_of_inequality_l186_186939

-- Let f be an even function defined on ℝ, and f' be its derivative
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def f_derivative (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop := ∀ x, f' x = (deriv f x)

-- Given conditions
variables {f f' : ℝ → ℝ}
hypothesis even_f : even_function f
hypothesis der_f : f_derivative f f'
hypothesis f_at_2 : f 2 = 0
hypothesis condition : ∀ x > 0, x * f' x > f x

-- To prove
theorem solution_set_of_inequality (x : ℝ) : x * f x < 0 ↔ (x < -2 ∨ (0 < x ∧ x < 2)) :=
sorry

end solution_set_of_inequality_l186_186939


namespace polynomial_divisible_by_24_l186_186973

theorem polynomial_divisible_by_24 (n : ℤ) : 24 ∣ (n^4 + 6 * n^3 + 11 * n^2 + 6 * n) :=
sorry

end polynomial_divisible_by_24_l186_186973


namespace alternating_sequence_property_l186_186553

theorem alternating_sequence_property (x : ℝ) (n : ℕ) (h : x ≠ 0) : 
  let y := if n % 2 = 0 then (x + 1) ^ (-2)^n else (x + 1) ^ (-2)^n
  in y = (x + 1) ^ (-2)^n := 
sorry

end alternating_sequence_property_l186_186553


namespace three_digit_integers_with_odd_factors_l186_186209

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l186_186209


namespace incorrect_statement_b_l186_186014

-- Let a, b be real numbers
variables {a b : ℝ}

-- The floor function and fractional part definitions
def floor (x : ℝ) := int.to_nat (int.floor x)
def frac (x : ℝ) := x - (floor x)

-- Condition 1: Real number a and b, and a ≠ 0
axiom a_non_zero : a ≠ 0

-- Condition 2: Real number b and b ≠ 0
axiom b_non_zero : b ≠ 0

-- Condition 3: The main equation a = b * floor(a/b) - b * frac(a/b)
axiom main_eq : a = b * (floor (a / b)) - b * (frac (a / b))

-- The goal is to prove that if a is a non-zero integer, b is not necessarily an integer (thus proving the statement is incorrect)
theorem incorrect_statement_b : int.to_nat (int.floor a) ≠ 0 → (∃ (a_int : ℤ), a = a_int) → ¬ (∃ (b_int : ℤ), b = b_int) :=
by
  sorry

end incorrect_statement_b_l186_186014


namespace calculate_S9_l186_186827

open Function

-- Given definitions and properties of the arithmetic sequence
variables {a : ℕ → ℝ} -- defines an arithmetic sequence

-- Conditions
axiom a2_a5_a8_sum : a 2 + a 5 + a 8 = 12

-- Desired sum of the first 9 terms of the sequence
noncomputable def S (n : ℕ) : ℝ := n * (a 1 + a n) / 2

theorem calculate_S9 : S 9 = 36 :=
by
  -- Introducing the middle term property axiom
  have property_of_arithmetic_sequence : ∀ n : ℕ, a n + a (n + 3) + a (n + 6) = 3 * a (n + 3),
  sorry

  -- By using the provided condition we need to deduce that S 9 equals 36
  have a5_value : a 5 = 4,
  sorry

  have sum_formula : S 9 = 9 * a 5,
  sorry

  show S 9 = 36,
  sorry

end calculate_S9_l186_186827


namespace tangent_line_lemma_l186_186837

def f (x n : ℝ) := Real.log x + n * x

noncomputable def f' (x n : ℝ) := 1 / x + n

noncomputable def tangent_at_x0 (x0 n : ℝ) := 2 * x0 - 1

theorem tangent_line_lemma (n : ℝ) :
  (∃ x0 : ℝ, f' x0 n = 2) → 
  (f 1 1 + f' 1 1 = 3) :=
by
  sorry

end tangent_line_lemma_l186_186837


namespace three_digit_integers_with_odd_number_of_factors_count_l186_186266

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l186_186266


namespace three_digit_integers_with_odd_factors_l186_186159

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l186_186159


namespace range_of_m_l186_186037

variable (p q : Prop)
variable (m : ℝ)
variable (hp : (∀ x y : ℝ, (x^2 / (2 * m) + y^2 / (1 - m) = 1) → (0 < m ∧ m < 1/3)))
variable (hq : (m^2 - 15 * m < 0))

theorem range_of_m (h_not_p_and_q : ¬ (p ∧ q)) (h_p_or_q : p ∨ q) :
  (1/3 ≤ m ∧ m < 15) :=
sorry

end range_of_m_l186_186037


namespace snake_length_difference_l186_186526

theorem snake_length_difference :
  ∀ (jake_len penny_len : ℕ), 
    jake_len > penny_len →
    jake_len + penny_len = 70 →
    jake_len = 41 →
    jake_len - penny_len = 12 :=
by
  intros jake_len penny_len h1 h2 h3
  sorry

end snake_length_difference_l186_186526


namespace derivative_of_y_l186_186787

noncomputable def y (x : ℝ) : ℝ := 
  (2 * x - 1) / 4 * sqrt (2 + x - x^2) + 9 / 8 * Real.arcsin ((2 * x - 1) / 3)

theorem derivative_of_y (x : ℝ) : 
  deriv y x = sqrt (2 + x - x^2) :=
by
  sorry

end derivative_of_y_l186_186787


namespace three_digit_integers_with_odd_factors_l186_186321

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l186_186321


namespace three_digit_integers_with_odd_number_of_factors_l186_186168

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l186_186168


namespace roots_of_quadratic_l186_186048

theorem roots_of_quadratic (x1 x2 : ℝ) (h : ∀ x, x^2 - 3 * x - 2 = 0 → x = x1 ∨ x = x2) :
  x1 * x2 + x1 + x2 = 1 :=
sorry

end roots_of_quadratic_l186_186048


namespace three_digit_integers_odd_factors_count_l186_186460

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l186_186460


namespace exists_uv_among_101_reals_l186_186976

theorem exists_uv_among_101_reals 
  (S : Set ℝ) (hS : S.card = 101) : 
  ∃ (u v : ℝ), u ∈ S ∧ v ∈ S ∧ u ≠ v ∧ 100 * |u - v| * |1 - u * v| ≤ (1 + u^2) * (1 + v^2) :=
by
  sorry

end exists_uv_among_101_reals_l186_186976


namespace number_of_three_digit_squares_l186_186438

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l186_186438


namespace quadrilateral_max_area_l186_186504

theorem quadrilateral_max_area 
  (A B C D : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] 
  (h1: ∠A D C = (90:ℝ) + ∠B A C)
  (h2: dist A B = 17) (h3: dist B C = 17) (h4: dist C D = 16) :
  area ABCD ≤ 529 / 2 :=
begin
  sorry
end

end quadrilateral_max_area_l186_186504


namespace three_digit_integers_with_odd_factors_count_l186_186291

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l186_186291


namespace max_subset_size_l186_186925

open Set

theorem max_subset_size {A : Set ℕ} (h : ∀ x y ∈ A, x ≠ y → x * y ∉ A) :
  ∃ (A : Set ℕ), A ⊆ finset.range 1000001 ∧
                 (∀ x y ∈ A, x ≠ y → x * y ∉ A) ∧
                 A.card = 999001 :=
sorry

end max_subset_size_l186_186925


namespace divisible_by_42_l186_186972

theorem divisible_by_42 (a : ℤ) : ∃ k : ℤ, a^7 - a = 42 * k := 
sorry

end divisible_by_42_l186_186972


namespace bottle_caps_bought_l186_186661

theorem bottle_caps_bought (original : ℕ) (total : ℕ) (h1 : original = 2) (h2 : total = 43) : 
  total - original = 41 := 
by
  rw [h1, h2]
  simp
  sorry

end bottle_caps_bought_l186_186661


namespace three_digit_integers_with_odd_factors_count_l186_186298

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l186_186298


namespace three_digit_integers_with_odd_number_of_factors_l186_186164

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l186_186164


namespace maximum_acute_triangles_from_four_points_l186_186823

-- Define a point in a plane
structure Point (α : Type) := (x : α) (y : α)

-- Definition of an acute triangle is intrinsic to the problem
def is_acute_triangle {α : Type} [LinearOrderedField α] (A B C : Point α) : Prop :=
  sorry -- Assume implementation for determining if a triangle is acute angles based

def maximum_number_acute_triangles {α : Type} [LinearOrderedField α] (A B C D : Point α) : ℕ :=
  sorry -- Assume implementation for verifying maximum number of acute triangles from four points

theorem maximum_acute_triangles_from_four_points {α : Type} [LinearOrderedField α] (A B C D : Point α) :
  maximum_number_acute_triangles A B C D = 4 :=
  sorry

end maximum_acute_triangles_from_four_points_l186_186823


namespace value_of_leftover_coins_l186_186710

theorem value_of_leftover_coins : 
  let quarters_per_roll := 30
      dimes_per_roll := 40
      sally_quarters := 101
      sally_dimes := 173
      ben_quarters := 150
      ben_dimes := 195 in
      let total_quarters := sally_quarters + ben_quarters
      total_dimes := sally_dimes + ben_dimes in
      let leftover_quarters := total_quarters % quarters_per_roll
      leftover_dimes := total_dimes % dimes_per_roll in
      let value_leftover_quarters := leftover_quarters * 0.25
      value_leftover_dimes := leftover_dimes * 0.10 in
      value_leftover_quarters + value_leftover_dimes = 3.55 :=
by
  sorry

end value_of_leftover_coins_l186_186710


namespace num_odd_factors_of_three_digit_integers_eq_22_l186_186121

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l186_186121


namespace three_digit_integers_odd_factors_count_l186_186452

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l186_186452


namespace problem_statement_l186_186031

open LinearAlgebra

variables {R : Type*} [Field R]
variables (U : R^3 → R^3)
variables (a b : R) (v w : R^3)
variables (u : R^3 → R^3 → R^3)

-- Given conditions
axiom linearity : ∀ (a b : R) (v w : R^3), U (a • v + b • w) = a • U v + b • U w
axiom cross_product : ∀ (v w : R^3), U (u v w) = u (U v) (U w)
axiom U_v1 : U (⟨9, 3, 6⟩ : R^3) = ⟨3, 9, -2⟩
axiom U_v2 : U (⟨3, -6, 9⟩ : R^3) = ⟨-2, 3, 9⟩

-- Problem statement to prove
theorem problem_statement : U (⟨6, 12, 15⟩ : R^3) = ⟨-22/7, 66/7, 186/7⟩ :=
sorry

end problem_statement_l186_186031


namespace slope_angle_of_line_l_distance_between_A_and_B_l186_186849

-- Define the parameterized equation of line l
def line_l (t : ℝ) : (ℝ × ℝ) :=
  (1 / 2 * t, (√2 / 2) + (√3 / 2) * t)

-- Define the polar equation of curve C
def curve_C (θ : ℝ) : ℝ :=
  2 * Real.cos (θ - π / 4)

-- Define the corresponding Cartesian equation of line l
def cartesian_line_l (x : ℝ) : ℝ :=
  √3 * x + √2 / 2

-- Define the Cartesian equation of the curve
def cartesian_curve_C (x y : ℝ) : Prop :=
  (x - √2 / 2) ^ 2 + (y - √2 / 2) ^ 2 = 1

-- Statement (1): Prove the slope angle of line l is 60 degrees (π / 3)
theorem slope_angle_of_line_l : 
  ∃ θ : ℝ, θ = π / 3 ∧ ∀ t : ℝ, let (x, y) := line_l t in y = tan θ * x + (√2 / 2) :=
sorry

-- Statement (2): Prove that the distance |AB| between the intersection points A and B is √10 / 2
theorem distance_between_A_and_B :
  ∃ A B : (ℝ × ℝ), 
    cartesian_curve_C A.1 A.2 ∧ 
    cartesian_curve_C B.1 B.2 ∧ 
    cartesian_line_l A.1 = A.2 ∧ 
    cartesian_line_l B.1 = B.2 ∧ 
    (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) = sqrt 10 / 2 :=
sorry

end slope_angle_of_line_l_distance_between_A_and_B_l186_186849


namespace three_digit_integers_with_odd_factors_l186_186384

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l186_186384


namespace num_three_digit_ints_with_odd_factors_l186_186244

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l186_186244


namespace three_digit_perfect_squares_count_l186_186195

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l186_186195


namespace three_digit_integers_with_odd_factors_l186_186224

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l186_186224


namespace three_digit_integers_with_odd_factors_count_l186_186289

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l186_186289


namespace num_three_digit_integers_with_odd_factors_l186_186377

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l186_186377


namespace trapezium_area_l186_186781

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 13) :
  (1 / 2) * (a + b) * h = 247 :=
by
  rw [ha, hb, hh]
  norm_num
  sorry

end trapezium_area_l186_186781


namespace three_digit_integers_with_odd_factors_l186_186210

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l186_186210


namespace three_digit_oddfactors_count_is_22_l186_186088

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l186_186088


namespace three_digit_perfect_squares_count_l186_186198

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l186_186198


namespace number_of_three_digit_integers_with_odd_number_of_factors_l186_186322

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l186_186322


namespace three_digit_perfect_squares_count_l186_186193

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l186_186193


namespace three_digit_odds_factors_count_l186_186350

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l186_186350


namespace three_digit_integers_with_odd_factors_l186_186206

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l186_186206


namespace three_digit_oddfactors_count_is_22_l186_186096

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l186_186096


namespace arccos_cos_solution_l186_186589

theorem arccos_cos_solution (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ (Real.pi / 2)) (h₂ : Real.arccos (Real.cos x) = 2 * x) : 
    x = 0 :=
by 
  sorry

end arccos_cos_solution_l186_186589


namespace three_digit_oddfactors_count_l186_186129

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l186_186129


namespace percent_of_d_is_e_l186_186946

variable (a b c d e : ℝ)
variable (h1 : d = 0.40 * a)
variable (h2 : d = 0.35 * b)
variable (h3 : e = 0.50 * b)
variable (h4 : e = 0.20 * c)
variable (h5 : c = 0.30 * a)
variable (h6 : c = 0.25 * b)

theorem percent_of_d_is_e : (e / d) * 100 = 15 :=
by sorry

end percent_of_d_is_e_l186_186946


namespace perfect_squares_count_l186_186081

theorem perfect_squares_count :
  let lower_bound := 3^5 + 1
  let upper_bound := 3^{10} + 1
  (number_of_perfect_squares lower_bound upper_bound) = 228 := by 
  sorry

/-- Function to count the number of perfect squares in the given range [lower_bound, upper_bound] -/
def number_of_perfect_squares (lower_bound upper_bound : ℕ) : ℕ :=
  let lower := Nat.sqrt lower_bound
  let upper := Nat.sqrt upper_bound
  if lower * lower < lower_bound then
    Nat.succ (upper - lower)
  else
    upper - lower + 1

end perfect_squares_count_l186_186081


namespace number_of_three_digit_squares_l186_186426

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l186_186426


namespace maximize_rectangle_area_l186_186522

theorem maximize_rectangle_area (a H : ℝ) : 
  ∃ (y : ℝ) (x : ℝ), 
  (∀ y, x = ((H - y) * a) / H) → 
  (∃ S : ℝ, S = x * y) → 
  (S.max (H / 2) → 
  x = (1 / 2) * a ∧ y = (1 / 2) * H) :=
by
  sorry

end maximize_rectangle_area_l186_186522


namespace lottery_tickets_equal_chance_l186_186626

-- Definition: Each lottery ticket has a 0.1% (0.001) chance of winning.
-- Definition: Each lottery ticket's outcome is independent of the others.

theorem lottery_tickets_equal_chance :
  let p := 0.001 in
  ∀ (n : ℕ) (tickets : Fin n → Prop),
    (∀ i, Prob (tickets i) = p) →
    (∀ i j, i ≠ j → Indep (tickets i) (tickets j)) →
    ∀ i, Prob (tickets i) = p :=
by
  intros p n tickets hprob hindep i
  exact hprob i
  sorry

end lottery_tickets_equal_chance_l186_186626


namespace graph_edge_labeling_l186_186930

open Classical

noncomputable section

variables (G : Type) [Graph G] [ConnectedGraph G]

def edge_labeling_possible (G : Graph) (k : ℕ) : Prop :=
  ∃ (label : G.edge → ℕ),
    (∀ e, 1 ≤ label e ∧ label e ≤ k) ∧
    (∀ v, Finset.card (incident_edges G v) ≥ 2 → 
          Int.gcd (label <$> incident_edges G v) = 1)

theorem graph_edge_labeling (G : Graph) (k : ℕ) (hG : connected G) (hk : G.edge_count = k) :
  edge_labeling_possible G k :=
sorry

end graph_edge_labeling_l186_186930


namespace three_digit_integers_with_odd_factors_l186_186314

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l186_186314


namespace number_of_three_digit_integers_with_odd_number_of_factors_l186_186327

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l186_186327


namespace count_zeros_in_decimal_l186_186864

theorem count_zeros_in_decimal (n d : ℕ) (h₀ : n = 1) (h₁ : d = 2^7 * 5^3) :
  ∃ k, (\(n / d) = (k / 10^7) ∧ (0 < k < 10^7) ∧ (natLength (natToDigits 10 k) = 3)) ∧ natLength (takeWhile (ArrEq 0) (natToDigits 10 ((n / d))) - 1) = 5 := 
sorry

end count_zeros_in_decimal_l186_186864


namespace john_overall_profit_l186_186671

-- Definitions
def Cost_grinder : ℝ := 15000
def Cost_mobile : ℝ := 10000
def Loss_grinder : ℝ := 4 / 100
def Profit_mobile : ℝ := 10 / 100

-- Selling prices calculated from the cost prices and respective profit/loss percentages
def Selling_price_grinder := Cost_grinder - (Loss_grinder * Cost_grinder)
def Selling_price_mobile := Cost_mobile + (Profit_mobile * Cost_mobile)

-- Total selling price and total cost price
def Total_selling_price := Selling_price_grinder + Selling_price_mobile
def Total_cost_price := Cost_grinder + Cost_mobile

-- Overall profit calculation
def Overall_profit := Total_selling_price - Total_cost_price

-- The theorem to be proven
theorem john_overall_profit : Overall_profit = 400 := by
  -- Code to provide the proof here
  sorry

end john_overall_profit_l186_186671


namespace simplify_sqrt_expression_l186_186588

variable (x : ℝ)

theorem simplify_sqrt_expression (h : x ≠ 0) :
  sqrt (1 + ( (x^6 - x^3 - 2) / (3 * x^3) ) ^ 2) = 
  (sqrt (x^12 - 2 * x^9 + 6 * x^6 - 2 * x^3 + 4)) / (3 * x^3) :=
by
  sorry

end simplify_sqrt_expression_l186_186588


namespace probability_sum_of_diagonals_odd_l186_186623

-- Define the condition that check the probability of sum of diagonals being odd is 1/126
def probability_odd_diagonals : ℚ := 1 / 126

theorem probability_sum_of_diagonals_odd :
  ∀ grid : Fin 3 → Fin 3 → ℕ,
  (∀ i j, grid i j ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ)) →
  (∀ i j i' j', grid i j ≠ grid i' j' ∨ (i = i' ∧ j = j')) →
  (∃ rows, ∃ (condition : ∀ row, row ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ))) →
    (∃ (condition : ∀ cell, cell ∈ (Fin 3 × Fin 3).1)) →
  (∃ sums_odd: ∀ diagonal, diagonal ∈ {sum (Finset.fin_range 3) (λ i, grid i i), sum (Finset.fin_range 3) (λ i, grid (Fin.mk i) (Fin.mk (2 - i)))} ) →
  probability_odd_diagonals = 1 / 126 :=
by
  sorry

end probability_sum_of_diagonals_odd_l186_186623


namespace three_digit_integers_with_odd_factors_l186_186306

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l186_186306


namespace S_21_equals_4641_l186_186856

-- Define the first element of the nth set
def first_element_of_set (n : ℕ) : ℕ :=
  1 + (n * (n - 1)) / 2

-- Define the last element of the nth set
def last_element_of_set (n : ℕ) : ℕ :=
  (first_element_of_set n) + n - 1

-- Define the sum of the nth set
def S (n : ℕ) : ℕ :=
  n * ((first_element_of_set n) + (last_element_of_set n)) / 2

-- The goal statement we want to prove
theorem S_21_equals_4641 : S 21 = 4641 := by
  sorry

end S_21_equals_4641_l186_186856


namespace number_of_three_digit_integers_with_odd_factors_l186_186412

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l186_186412


namespace three_digit_integers_with_odd_factors_l186_186318

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l186_186318


namespace three_digit_integers_with_odd_factors_count_l186_186284

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l186_186284


namespace three_digit_integers_with_odd_factors_l186_186316

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l186_186316


namespace total_people_in_school_l186_186500

def number_of_girls := 315
def number_of_boys := 309
def number_of_teachers := 772
def total_number_of_people := number_of_girls + number_of_boys + number_of_teachers

theorem total_people_in_school :
  total_number_of_people = 1396 :=
by sorry

end total_people_in_school_l186_186500


namespace total_surface_area_of_tower_l186_186988

theorem total_surface_area_of_tower :
  let volumes := [1, 8, 27, 64, 125, 216, 343, 512, 729, 1000]
  let side_lengths := List.map (λ v, Int.cbrt v) volumes
  let surface_areas := List.map (λ s, 6 * s * s) side_lengths
  let adjusted_surface_areas := [
    surface_areas[0], -- 1st cube: No adjustment needed
    surface_areas[1] - 4, -- 2nd cube: One face not visible upwards
    surface_areas[2] - 9, -- 3rd cube: One face not visible downwards
    surface_areas[3] - 16, -- 4th cube: One face not visible upwards
    surface_areas[4] - 25, -- 5th cube: One face not visible downwards
    surface_areas[5] - 36, -- 6th cube: One face not visible upwards
    surface_areas[6] - 49, -- 7th cube: One face not visible downwards
    surface_areas[7] - 64, -- 8th cube: One face not visible upwards
    surface_areas[8] - 81, -- 9th cube: One face not visible downwards
    surface_areas[9] -- 10th cube: Fully visible
  ]
  List.sum adjusted_surface_areas = 2026 := by 
  sorry

end total_surface_area_of_tower_l186_186988


namespace three_digit_integers_with_odd_factors_l186_186220

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l186_186220


namespace number_of_three_digit_squares_l186_186436

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l186_186436


namespace three_digit_integers_odd_factors_count_l186_186447

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l186_186447


namespace mean_of_set_is_16_6_l186_186999

theorem mean_of_set_is_16_6 (m : ℝ) (h : m + 7 = 16) :
  (9 + 11 + 16 + 20 + 27) / 5 = 16.6 :=
by
  -- Proof steps would go here, but we use sorry to skip the proof.
  sorry

end mean_of_set_is_16_6_l186_186999


namespace triangles_xyz_l186_186752

theorem triangles_xyz (A B C D P Q R : Type) 
    (u v w x : ℝ)
    (angle_ADB angle_BDC angle_CDA : ℝ)
    (h1 : angle_ADB = 120) 
    (h2 : angle_BDC = 120) 
    (h3 : angle_CDA = 120) :
    x = u + v + w :=
sorry

end triangles_xyz_l186_186752


namespace three_digit_integers_with_odd_number_of_factors_l186_186181

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l186_186181


namespace three_digit_integers_with_odd_number_of_factors_l186_186167

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l186_186167


namespace three_digit_oddfactors_count_is_22_l186_186083

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l186_186083


namespace sum_not_unit_fraction_l186_186537

theorem sum_not_unit_fraction 
  (p : Fin 42 → ℕ) (hp : ∀ i j, i ≠ j → p i ≠ p j) (hp_prime : ∀ i, Prime (p i)) : 
  ¬ ∃ n : ℕ, (∑ j in Finset.univ, 1 / ((p j) ^ 2 + 1 : ℝ)) = 1 / (n ^ 2 : ℝ) := 
sorry

end sum_not_unit_fraction_l186_186537


namespace find_complex_number_l186_186024

-- Define the complex number z and the condition
variable (z : ℂ)
variable (h : (conj z) / (1 + I) = 1 - 2 * I)

-- State the theorem
theorem find_complex_number (hz : h) : z = 3 + I := 
sorry

end find_complex_number_l186_186024


namespace three_digit_perfect_squares_count_l186_186187

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l186_186187


namespace pointA_when_B_origin_pointB_when_A_origin_l186_186018

def vectorAB : ℝ × ℝ := (-2, 4)

-- Prove that when point B is the origin, the coordinates of point A are (2, -4)
theorem pointA_when_B_origin : vectorAB = (-2, 4) → (0, 0) - (-2, 4) = (2, -4) :=
by
  sorry

-- Prove that when point A is the origin, the coordinates of point B are (-2, 4)
theorem pointB_when_A_origin : vectorAB = (-2, 4) → (0, 0) + (-2, 4) = (-2, 4) :=
by
  sorry

end pointA_when_B_origin_pointB_when_A_origin_l186_186018


namespace jake_total_cost_proof_l186_186528

structure Brand (cost : ℕ) (washes : ℕ)

def calculate_bottles_needed (weeks : ℕ) (washes_per_bottle : ℕ) : ℕ :=
  (weeks + washes_per_bottle - 1) / washes_per_bottle

def apply_discount (cost : ℕ) (bottles : ℕ) : ℕ :=
  if bottles >= 5 then (90 * cost * bottles) / 100 else cost * bottles

def total_cost (brandA brandB brandC : Brand) (weight1 weight2 weight3 : ℕ) : ℕ :=
  let total_a := apply_discount brandA.cost (calculate_bottles_needed weight1 brandA.washes)
  let total_b := apply_discount brandB.cost (calculate_bottles_needed weight2 brandB.washes)
  let total_c := brandC.cost * (calculate_bottles_needed weight3 brandC.washes)
  total_a + total_b + total_c

theorem jake_total_cost_proof :
  let brandA : Brand := { cost := 4, washes := 4 }
  let brandB : Brand := { cost := 6, washes := 6 }
  let brandC : Brand := { cost := 8, washes := 9 }
  total_cost brandA brandB brandC 8 7 5 = 26 :=
by {
  sorry
}

end jake_total_cost_proof_l186_186528


namespace three_digit_integers_with_odd_number_of_factors_count_l186_186276

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l186_186276


namespace three_digit_integers_with_odd_number_of_factors_count_l186_186268

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l186_186268


namespace three_digit_oddfactors_count_is_22_l186_186084

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l186_186084


namespace plant_height_at_2_years_l186_186709

-- Define the height function h, where h(n) is the height of the plant at the end of year n
def height (h : ℕ → ℝ) (quadruples : ∀ n, h (n + 1) = 4 * h n) (h_at_4 : h 4 = 256) : Prop :=
  h 2 = 16

-- The problem statement
theorem plant_height_at_2_years :
  ∃ h : ℕ → ℝ, (∀ n, h (n + 1) = 4 * h n) ∧ h 4 = 256 ∧ height h (λ n, by sorry) := by
    sorry

end plant_height_at_2_years_l186_186709


namespace jackpot_probability_l186_186618

-- Definitions based on conditions
def MegaBallProbability : ℚ := 1 / 30
def WinnerBallsCombination : ℕ := Nat.binomial 50 5
def WinnerBallsProbability : ℚ := 1 / WinnerBallsCombination
def BonusBallProbability : ℚ := 1 / 15

-- The statement to prove
theorem jackpot_probability : 
  MegaBallProbability * WinnerBallsProbability * BonusBallProbability = 1 / 954594900 := 
by
  unfold MegaBallProbability WinnerBallsProbability BonusBallProbability WinnerBallsCombination
  have h1 : WinnerBallsCombination = 2118760 := by
    sorry -- This should compute Nat.binomial 50 5 = 2118760
  rw [h1]
  norm_num
  sorry

end jackpot_probability_l186_186618


namespace three_digit_integers_with_odd_factors_l186_186145

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l186_186145


namespace power_of_fraction_fraction_power_result_l186_186737

theorem power_of_fraction (a b : ℝ) (n : ℕ) : 
  (b ≠ 0) → ((a / b)^n = a^n / b^n) :=
begin
  sorry
end

noncomputable def calculate_fraction_power : ℝ := (5/3) ^ 3

theorem fraction_power_result : calculate_fraction_power = 125 / 27 :=
begin
  sorry
end

end power_of_fraction_fraction_power_result_l186_186737


namespace find_principal_amount_l186_186701

theorem find_principal_amount
  (P R T SI : ℝ) 
  (rate_condition : R = 12)
  (time_condition : T = 20)
  (interest_condition : SI = 2100) :
  SI = (P * R * T) / 100 → P = 875 :=
by
  sorry

end find_principal_amount_l186_186701


namespace three_digit_integers_with_odd_factors_l186_186319

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l186_186319


namespace eccentricity_of_hyperbola_l186_186845

noncomputable def hyperbola_eccentricity (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
  (dot_product_cond : a * (sqrt (a^2 + b^2)) - b^2 = 0) : ℝ :=
  let c := sqrt (a^2 + b^2) in
  c / a

theorem eccentricity_of_hyperbola (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
  (dot_product_cond : a * (sqrt (a^2 + b^2)) - b^2 = 0) :
  hyperbola_eccentricity a b a_pos b_pos dot_product_cond = (1 + sqrt 5) / 2 :=
sorry

end eccentricity_of_hyperbola_l186_186845


namespace minimize_y_at_l186_186558

variable (a b c d : ℝ)

def y (x : ℝ) : ℝ :=
  (x - a)^2 + (x - b)^2 + c * (x - d)^2

theorem minimize_y_at :. ∃ x, x = (a + b + c * d) / (2 + c) ∧ ∀ x' : ℝ, y a b c d x ≤ y a b c d x' :=
sorry

end minimize_y_at_l186_186558


namespace num_odd_factors_of_three_digit_integers_eq_22_l186_186119

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l186_186119


namespace frame_price_increase_l186_186665

-- Definitions based on conditions
variable (P : ℝ) -- price of the initial frame
variable (budget : ℝ := 60) -- Yvette's budget
variable (remaining : ℝ := 6) -- money left after buying the smaller frame
variable (smaller_frame_factor : ℝ := 3/4) -- ratio of smaller frame price to initial frame price

-- We define the price Yvette paid for the smaller frame 
def smaller_frame_price := smaller_frame_factor * P 

-- We know from the conditions that the remaining money after buying the smaller frame is $6
def equation := smaller_frame_price = budget - remaining

-- Main theorem to prove
theorem frame_price_increase (h : equation) : 
  let P_value := (budget - remaining) / smaller_frame_factor in
  let increase := P_value - budget in
  let percentage_increase := (increase / budget) * 100 in
  percentage_increase = 20 := by
  sorry

end frame_price_increase_l186_186665


namespace boys_at_reunion_l186_186685

theorem boys_at_reunion (n : ℕ) (H : n * (n - 1) / 2 = 45) : n = 10 :=
by sorry

end boys_at_reunion_l186_186685


namespace sqrt_inequality_l186_186572

theorem sqrt_inequality (a : ℝ) (h : a ≥ 2) : 
  sqrt (a + 1) - sqrt a < sqrt (a - 1) - sqrt (a - 2) := 
by
  sorry

end sqrt_inequality_l186_186572


namespace three_digit_integers_with_odd_factors_l186_186151

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l186_186151


namespace three_digit_integers_with_odd_factors_l186_186309

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l186_186309


namespace parabola_distance_condition_l186_186028

noncomputable def parabola_equation {p : ℝ} (h_p_gt_zero : p > 0) (h_AB : abs ((parabola : ℝ) - (focus_line : ℝ)) = 8) :
  y^2 = 6*x := by
  sorry

def parabola (p : ℝ) :  Prop :=
  y^2 = 2 * p * x

def focus_of_parabola (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

def line_through_focus (p : ℝ) : ℝ :=
  fun (x : ℝ) y -> y = sqrt(3) * (x - p / 2)

def intersection_points (p : ℝ) : Set (ℝ × ℝ) :=
  {A B : ℝ × ℝ // A ≠ B ∧ A ∈ parabola(p)}

theorem parabola_distance_condition (p : ℝ) (h_p_gt_zero : p > 0)
  (h_inter : ∀ A B ∈ intersection_points(p), abs (A - B) = 8) :
  y^2 = 6*x := sorry

end parabola_distance_condition_l186_186028


namespace medians_square_sum_l186_186656

theorem medians_square_sum (a b c : ℝ) (ha : a = 13) (hb : b = 13) (hc : c = 10) :
  let m_a := (1 / 2 * (2 * b^2 + 2 * c^2 - a^2))^(1/2)
  let m_b := (1 / 2 * (2 * c^2 + 2 * a^2 - b^2))^(1/2)
  let m_c := (1 / 2 * (2 * a^2 + 2 * b^2 - c^2))^(1/2)
  m_a^2 + m_b^2 + m_c^2 = 432 :=
by
  sorry

end medians_square_sum_l186_186656


namespace solve_for_x_l186_186467

theorem solve_for_x (x : ℝ) (h : 9 / x^2 = x / 81) : x = 9 := 
  sorry

end solve_for_x_l186_186467


namespace three_digit_integers_with_odd_factors_l186_186229

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l186_186229


namespace circle_equation_l186_186063

theorem circle_equation (x y : ℝ) (h1 : y^2 = 4 * x) (h2 : (x - 1)^2 + y^2 = 4) :
    x^2 + y^2 - 2 * x - 3 = 0 :=
sorry

end circle_equation_l186_186063


namespace contractor_daily_wage_l186_186688

theorem contractor_daily_wage :
  let x := 25 in
  let total_days := 30 in
  let absent_days := 10 in
  let fine_per_absent_day := 7.50 in
  let total_earned := 425 in
  let worked_days := total_days - absent_days in
  (worked_days * x - absent_days * fine_per_absent_day = total_earned) → x = 25 :=
by
  intros
  sorry

end contractor_daily_wage_l186_186688


namespace find_point_A_l186_186699

noncomputable def point_A_coordinates (a : ℝ) : ℝ × ℝ :=
  (a, -Real.log a / Real.log 2 / 3)

theorem find_point_A :
  ∃ a : ℝ, 0 < a ∧ a < Real.exp 1 ∧
    (2 * a * Real.log 2  =  Real.log 2 + a * Real.log 2) ∧
    point_A_coordinates (Real.cbrt 4 / 2) = (Real.cbrt 4 / 2, -Real.log 2 / 3) :=
begin
  sorry
end

end find_point_A_l186_186699


namespace fraction_equation_solution_l186_186983

theorem fraction_equation_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) :
  (3 / (x - 3) = 4 / (x - 4)) → x = 0 :=
by
  sorry

end fraction_equation_solution_l186_186983


namespace fraction_of_n_is_80_l186_186878

-- Definitions from conditions
def n := (5 / 6) * 240

-- The theorem we want to prove
theorem fraction_of_n_is_80 : (2 / 5) * n = 80 :=
by
  -- This is just a placeholder to complete the statement, 
  -- actual proof logic is not included based on the prompt instructions
  sorry

end fraction_of_n_is_80_l186_186878


namespace race_length_l186_186889

theorem race_length (A_time : ℕ) (diff_distance diff_time : ℕ) (A_time_eq : A_time = 380)
  (diff_distance_eq : diff_distance = 50) (diff_time_eq : diff_time = 20) :
  let B_speed := diff_distance / diff_time
  let B_time := A_time + diff_time
  let race_length := B_speed * B_time
  race_length = 1000 := 
by
  sorry

end race_length_l186_186889


namespace visible_during_metaphase_l186_186563

-- Define the structures which could be present in a plant cell during mitosis.
inductive Structure
| Chromosomes
| Spindle
| CellWall
| MetaphasePlate
| CellMembrane
| Nucleus
| Nucleolus

open Structure

-- Define what structures are visible during metaphase.
def visibleStructures (phase : String) : Set Structure :=
  if phase = "metaphase" then
    {Chromosomes, Spindle, CellWall}
  else
    ∅

-- The proof statement
theorem visible_during_metaphase :
  visibleStructures "metaphase" = {Chromosomes, Spindle, CellWall} :=
by
  sorry

end visible_during_metaphase_l186_186563


namespace intersecting_lines_in_triangle_l186_186521

theorem intersecting_lines_in_triangle
  {K I A R E : Type*}
  [Lined K I A]
  [KA_length : KA < KI]
  [FeetPerpendicular R E K angle_bisector]
  : ∃ (M : Type*), ∃ (line_IE line_RA perp_KR : Line), 
      intersects line_IE M ∧ intersects line_RA M ∧ intersects perp_KR M :=
sorry

end intersecting_lines_in_triangle_l186_186521


namespace part_a_part_b_l186_186673

variable {n : ℕ} 
variable {x : Fin n → ℤ}
hypothesis h1 : ∀ i, x i = 1 ∨ x i = -1 
hypothesis h2 : ((Finset.univ.image ℕ λ i, x i * x (i + 1) % n)).sum = 0 

theorem part_a : 
  (∀ k : ℕ, k ∈ Finset.range n \ {0} → (Finset.univ.image ℕ λ i, x i * x (i + k) % n)).sum = 0 → 
  ∃ m : ℕ, n = m ^ 2 :=
by
  sorry

theorem part_b :
  n = 16 → 
  (∀ k : ℕ, k ∈ Finset.range n \ {0} → (Finset.univ.image ℕ λ i, x i * x (i + k) % n)).sum = 0 →
  ¬ (∃ x : Fin 16 → ℤ, ∀ i, x i = 1 ∨ x i = -1 ∧ (Finset.univ.image ℕ λ i, x i * x (i + 1) % n)).sum = 0 :=
by
  sorry

end part_a_part_b_l186_186673


namespace mean_reading_days_last_week_l186_186499

theorem mean_reading_days_last_week :
  let num_students := [2, 3, 5, 4, 7, 10, 3]
  let num_days := [0, 1, 2, 3, 4, 5, 6]
  let total_days := list.sum (list.map (λ p : ℕ × ℕ, p.1 * p.2) (list.zip num_students num_days))
  let total_students := list.sum num_students
  let mean := (total_days : ℚ) / total_students
  mean = 356 / 100 :=
by
  sorry

end mean_reading_days_last_week_l186_186499


namespace albert_more_than_joshua_l186_186534

def joshua_rocks : ℝ := 80
def jose_rocks : ℝ := joshua_rocks - 14
def albert_rocks : ℝ := (2.5 * jose_rocks) - 4
def clara_rocks : ℝ := jose_rocks / 2
def maria_rocks : ℝ := (1.5 * clara_rocks) + 2.5

theorem albert_more_than_joshua : albert_rocks - joshua_rocks = 81 :=
by
  sorry

end albert_more_than_joshua_l186_186534


namespace solution_set_of_inequality_l186_186938

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry

theorem solution_set_of_inequality :
  (∀ x : ℝ, f(x) + f'(x) > 2) →
  (f(0) = 2021) →
  { x : ℝ | f(x) > 2 + 2019 / real.exp x } = { x : ℝ | 0 < x } :=
by
  sorry

end solution_set_of_inequality_l186_186938


namespace cosine_of_smallest_angle_l186_186519

theorem cosine_of_smallest_angle (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) : 
  let cos_A := (b^2 + c^2 - a^2) / (2 * b * c) in
  cos_A = 7 / 8 :=
by
  rw [h1, h2, h3],
  have : (3^2 + 4^2 - 2^2 : ℝ) = 21 := rfl,
  have : (2 * 3 * 4 : ℝ) = 24 := rfl,
  rw [this, this],
  simp,
  sorry

end cosine_of_smallest_angle_l186_186519


namespace clock_angle_at_5_15_l186_186077

def degrees_in_circle := 360
def hours_on_clock := 12
def hour_angle := degrees_in_circle / hours_on_clock  -- Each hour mark in degrees

def minutes_in_hour := 60
def minute_percentage := 15 / minutes_in_hour    -- 15 minutes as a fraction of an hour
def minute_angle := minute_percentage * degrees_in_circle  -- Angle of the minute hand at 15 minutes

def degrees_per_minute := hour_angle / minutes_in_hour    -- Hour hand movement per minute
def hour_position := 5 * hour_angle    -- Hour hand position at 5:00
def hour_additional := degrees_per_minute * 15    -- Additional movement by the hour hand in 15 minutes
def hour_angle_at_time := hour_position + hour_additional    -- Total hour hand position at 5:15

-- Calculate the smaller angle between the two hands
def smaller_angle := hour_angle_at_time - minute_angle

theorem clock_angle_at_5_15 : smaller_angle = 67.5 := by
  sorry

end clock_angle_at_5_15_l186_186077


namespace triangle_pur_area_l186_186908

/-- In triangle PQR, the medians PS and QT have lengths 25 and 36, respectively, and PQ = 30.
Extend QT to intersect the circumcircle of PQR at U. The area of triangle PUR is 2 * sqrt(151). -/
theorem triangle_pur_area (P Q R S T U : Point)
  (hPQR : Triangle P Q R)
  (hPS : Median P S)
  (hQT : Median Q T)
  (hPQ : dist P Q = 30)
  (hPS_length : dist P S = 25)
  (hQT_length : dist Q T = 36)
  (h_circumcircle_intersection : ∃ U, Extends T U (circumcircle P Q R)) :
  ∃ (k m : ℕ), k = 2 ∧ m = 151 ∧ area P U R = k * real.sqrt m :=
by sorry

end triangle_pur_area_l186_186908


namespace range_of_x_l186_186561

variable (x : ℝ)

def p := x^2 - 4 * x + 3 < 0
def q := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

theorem range_of_x : ¬ (p x ∧ q x) ∧ (p x ∨ q x) → (1 < x ∧ x ≤ 2) ∨ x = 3 :=
by 
  sorry

end range_of_x_l186_186561


namespace reflection_curve_eq_l186_186951

theorem reflection_curve_eq (a b c : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, let y := ax^2 + bx + c in
   let C1 := ax^2 - bx + c in
   let C2 := -C1 in
   y = -ax^2 + bx - c) :=
sorry

end reflection_curve_eq_l186_186951


namespace number_of_three_digit_squares_l186_186425

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l186_186425


namespace expected_value_Y_in_steady_state_l186_186617

-- Define the stationary functions X and Y
variables (X Y : ℝ → ℝ)

-- Define the expectations of X and Y
noncomputable def m_x : ℝ := 5
noncomputable def m_y : ℝ := 3 * m_x

-- Define the differential equation
axiom diff_eq : ∀ t, deriv Y t + 2 * Y t = 5 * deriv X t + 6 * X t

-- Define the stationarity conditions
axiom stationary_X : ∀ t, ∫ (τ : ℝ), deriv X τ = 0
axiom stationary_Y : ∀ t, ∫ (τ : ℝ), deriv Y τ = 0

-- The proof statement to be proved
theorem expected_value_Y_in_steady_state : m_y = 15 :=
  sorry

end expected_value_Y_in_steady_state_l186_186617


namespace max_value_expression_l186_186015

theorem max_value_expression (a : ℝ) (h : a^2 ≤ 4) : 
    ∃ t_max, (∀ t, 
        t ≤ (7 * real.sqrt((4 * a)^2 + 4) - 2 * a^2 - 2) / (real.sqrt(4 + 16 * a^2) + 6)) → t ≤ t_max :=
sorry

end max_value_expression_l186_186015


namespace cricketer_bowling_average_l186_186691

variable (A : ℝ) (R : ℝ := A * 85 + 26) (W : ℝ := 90)
variable (old_avg_reduction : ℝ := 0.4)
variable (initial_wickets : ℝ := 85)
variable (wickets_in_last_match : ℝ := 5)
variable (runs_in_last_match : ℝ := 26)

def average_before_last_match (A : ℝ) : Prop :=
  A - old_avg_reduction = R / W

theorem cricketer_bowling_average :
    average_before_last_match A →
    A = 12.4 :=
by
  sorry

end cricketer_bowling_average_l186_186691


namespace imaginary_part_of_conjugate_l186_186790

noncomputable def Z : ℂ := complex.I + 1
noncomputable def Z_conjugate := conj Z

theorem imaginary_part_of_conjugate : (Z_conjugate.im) = -1 := by
  sorry

end imaginary_part_of_conjugate_l186_186790


namespace jason_cutting_grass_time_l186_186915

-- Conditions
def time_to_cut_one_lawn : ℕ := 30 -- in minutes
def lawns_cut_each_day : ℕ := 8
def days : ℕ := 2
def minutes_in_an_hour : ℕ := 60

-- Proof that the number of hours Jason spends cutting grass over the weekend is 8
theorem jason_cutting_grass_time:
  ((lawns_cut_each_day * days) * time_to_cut_one_lawn) / minutes_in_an_hour = 8 :=
by
  sorry

end jason_cutting_grass_time_l186_186915


namespace chord_intersection_eq_l186_186824

theorem chord_intersection_eq (x y : ℝ) (r : ℝ) : 
  (x + 1)^2 + y^2 = r^2 → 
  (x - 4)^2 + (y - 1)^2 = 4 → 
  (x = 4) → 
  (y = 1) → 
  (r^2 = 26) → (5 * x + y - 19 = 0) :=
by
  sorry

end chord_intersection_eq_l186_186824


namespace numberOfElementsInMIntersectionN_l186_186876

def M (x y : ℝ) : Prop := Real.tan (π * y) + Real.sin (π * x) ^ 2 = 0
def N (x y : ℝ) : Prop := x ^ 2 + y ^ 2 ≤ 2

theorem numberOfElementsInMIntersectionN : 
  ∃ S : Finset (ℝ × ℝ), (∀ p ∈ S, M p.1 p.2 ∧ N p.1 p.2) ∧ Finset.card S = 9 :=
sorry

end numberOfElementsInMIntersectionN_l186_186876


namespace three_digit_odds_factors_count_l186_186348

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l186_186348


namespace number_of_three_digit_integers_with_odd_factors_l186_186405

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l186_186405


namespace three_digit_integers_with_odd_factors_l186_186387

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l186_186387


namespace three_digit_integers_with_odd_factors_count_l186_186287

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l186_186287


namespace sin_tan_sqrt_l186_186677

theorem sin_tan_sqrt (sin cos : ℝ → ℝ)
  (h1 : sin 40 = sin(40 : ℝ))
  (h2 : cos 10 = cos(10 : ℝ))
  (h3 : sin 80 = cos 10)
  (h4 : tan 10 = sin 10 / cos 10)
  (h5 : sin (a - b) = sin a * cos b - cos a * sin b)
  (h6 : sin (2 * θ) = 2 * sin θ * cos θ) :
  sin 40 * (tan 10 - sqrt 3) = -1 := 
sorry

end sin_tan_sqrt_l186_186677


namespace num_odd_factors_of_three_digit_integers_eq_22_l186_186111

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l186_186111


namespace finite_unpainted_blocks_l186_186708

theorem finite_unpainted_blocks 
  (m n r : ℕ) : 
  m * n * r = 2 * (m - 2) * (n - 2) * (r - 2) →
  set.finite { (m, n, r) : ℕ × ℕ × ℕ | m * n * r = 2 * (m - 2) * (n - 2) * (r - 2) } :=
begin
  sorry,
end

lemma test_finite_unpainted_blocks : finite_unpainted_blocks.

end finite_unpainted_blocks_l186_186708


namespace symmetric_difference_sets_l186_186929

theorem symmetric_difference_sets {n : ℕ} (h : 0 < n) (S : Finset (Finset ℕ)) (hS : S.card = 2^n + 1) :
  ∃ A B : Finset (Finset ℕ), A.card ≥ 1 ∧ B.card ≥ 1 ∧ A ∪ B = S ∧ 
  (Finset.card (A.bind (λ x, B.image (λ y, x.symm_diff y))) ≥ 2^n) :=
sorry

end symmetric_difference_sets_l186_186929


namespace statement_II_is_true_l186_186711

-- Defining the statements according to the problem conditions
def statement_I (digit : ℕ) : Prop := digit = 5
def statement_II (digit : ℕ) : Prop := digit ≠ 6
def statement_III (digit : ℕ) : Prop := digit = 7
def statement_IV (digit : ℕ) : Prop := digit ≠ 8

-- Given conditions
axiom h1 : statement_I digit ∨ statement_II digit ∨ statement_III digit ∨ statement_IV digit
axiom h2 : ¬(statement_I digit ∧ statement_III digit)

-- Proving that statement_II must be true
theorem statement_II_is_true (digit : ℕ) : statement_II digit :=
begin
 sorry
end

end statement_II_is_true_l186_186711


namespace trigonometric_ratio_l186_186909

theorem trigonometric_ratio (a b c A B C : ℝ)
  (hΔ : A + B + C = π)
  (hA : A = π / 3)
  (ha : a = sqrt 3)
  (law_of_sines : a / sin A = b / sin B ∧ b / sin B = c / sin C) :
  (b + c) / (sin B + sin C) = 2 :=
sorry

end trigonometric_ratio_l186_186909


namespace three_digit_integers_with_odd_factors_l186_186157

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l186_186157


namespace number_of_three_digit_squares_l186_186427

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l186_186427


namespace total_fencing_cost_l186_186786

def side1 : ℕ := 34
def side2 : ℕ := 28
def side3 : ℕ := 45
def side4 : ℕ := 50
def side5 : ℕ := 55

def cost1_per_meter : ℕ := 2
def cost2_per_meter : ℕ := 2
def cost3_per_meter : ℕ := 3
def cost4_per_meter : ℕ := 3
def cost5_per_meter : ℕ := 4

def total_cost : ℕ :=
  side1 * cost1_per_meter +
  side2 * cost2_per_meter +
  side3 * cost3_per_meter +
  side4 * cost4_per_meter +
  side5 * cost5_per_meter

theorem total_fencing_cost : total_cost = 629 := by
  sorry

end total_fencing_cost_l186_186786


namespace product_of_common_roots_eq_17_l186_186607

-- Definitions for the polynomial equations and roots
def poly1 (x : ℝ) : ℝ := x^3 - 5 * x + 20
def poly2 (x : ℝ) (C : ℝ) : ℝ := x^3 + C * x^2 + 80

-- Conditions as hypothesized
variables (u v w t C : ℝ)
axiom h1 : poly1 u = 0
axiom h2 : poly1 v = 0
axiom h3 : poly1 w = 0
axiom h4 : poly2 u C = 0
axiom h5 : poly2 v C = 0
axiom h6 : poly2 t C = 0

-- Conditions from Vieta's formulas
axiom h7 : u + v + w = 0
axiom h8 : u * v * w = -20
axiom h9 : u * v + u * t + v * t = 0
axiom h10 : u * v * t = -80

-- Goal: prove that the product of the common roots in simplified form gives a + b + c = 17
theorem product_of_common_roots_eq_17 : 
  let uv := u * v in
  let a := 10 in
  let b := 3 in
  let c := 4 in
  a * (uv^(1/b.toReal)) = 10 * (1600^(1/3.toReal) / 4^(1/4.toReal)) → 
  a + b + c = 17 :=
by {
  sorry
}

end product_of_common_roots_eq_17_l186_186607


namespace part_I_part_II_part_III_l186_186029

-- Given conditions
def sequence (a : ℕ → ℝ) : Prop := ∀ n, a n > 0

def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = S n / 2 * (a n + 1 / a n)

-- Part I
theorem part_I (S : ℕ → ℝ) (a : ℕ → ℝ) (S_property : sum_first_n_terms S a) :
  ∀ n, S n^2 - S (n-1)^2 = 1 := sorry

-- Part II
theorem part_II (a : ℕ → ℝ) : 
  ∀ n ≥ 1, a n = sqrt n - sqrt (n - 1) := sorry

-- Part III
theorem part_III (b : ℕ → ℝ) :
  ∑ i in finset.range 100, b i = 10 := sorry

end part_I_part_II_part_III_l186_186029


namespace arithmetic_sequence_l186_186510

variable (a : ℕ → ℕ)
variable (h : a 1 + 3 * a 8 + a 15 = 120)

theorem arithmetic_sequence (h : a 1 + 3 * a 8 + a 15 = 120) : a 2 + a 14 = 48 :=
sorry

end arithmetic_sequence_l186_186510


namespace simplify_vector_expression_l186_186587

variable (P M N : Type) [AddCommGroup P] [AddCommGroup M] [AddCommGroup N]
variable (PM PN MN NP NM : P)

theorem simplify_vector_expression (PM PN MN NP NM : P) 
  (h1 : PM - PN + MN = PM + (-PN) + MN)
  (h2 : -PN = NP)
  (h3 : PM + NP + MN = NM + MN)
  (h4 : NM + MN = (0 : P)) :
  PM - PN + MN = 0 := 
by {
  rw h1,
  rw h2,
  rw h3,
  rw h4,
  exact rfl,
}

end simplify_vector_expression_l186_186587


namespace polar_coordinates_and_triangle_perimeter_l186_186906

open Real

/-- Given the parametric equation of line l and a circle C,
    we need to prove specific properties about their intersections
    in a polar coordinate system. -/
theorem polar_coordinates_and_triangle_perimeter :
  (∀ t : ℝ, (λ p : ℝ × ℝ, p = (t, 1 + t) ∧ line_m_parallel_to_line_l_through_origin := p.2 = p.1)) →
  (∀ ϕ : ℝ, (λ c : ℝ × ℝ, c = (1 + cos ϕ, 2 + sin ϕ))) →
  (the_polar_equation_of_line_m_equal θ = π / 4) →
  (the_polar_equation_of_circle_C_equal ρ^2 - 2ρcosθ - 4ρsinθ + 4 = 0) →
  (points_A_and_B_and_perimeter_ABC ρ1 ρ2 :=
    ((ρ1 = (3 * sqrt 2 + sqrt (16 - (3 * sqrt 2)^2)) / 2) ∧
     (ρ2 = (3 * sqrt 2 - sqrt (16 - (3 * sqrt 2)^2)) / 2)) ↔
    perimeter_ABC = 2 + sqrt 2) :=
begin
  sorry -- Proof to be completed
end

end polar_coordinates_and_triangle_perimeter_l186_186906


namespace number_of_three_digit_integers_with_odd_number_of_factors_l186_186328

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l186_186328


namespace num_odd_factors_of_three_digit_integers_eq_22_l186_186120

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l186_186120


namespace num_three_digit_integers_with_odd_factors_l186_186366

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l186_186366


namespace WXYZ_area_l186_186900

-- Define the structure of the rectangle and its trisection points
structure Rectangle (P Q R S M N T U : Type) :=
  (is_Rectangle : true) -- To represent PQRS is a rectangle
  (trisect_PS : true) -- M and N divide PS into three equal parts
  (trisect_RQ : true) -- T and U divide RQ into three equal parts
  (PR_eq_3 : true) -- PR = 3
  (PN_eq_3 : true) -- PN = 3

-- Define the midpoints W, X, Y, and Z
structure Midpoints (W X Y Z : Type) :=
  (is_midpoint_MT : true) -- W is the midpoint of MT
  (is_midpoint_TU : true) -- X is the midpoint of TU
  (is_midpoint_NU : true) -- Y is the midpoint of NU
  (is_midpoint_MN : true) -- Z is the midpoint of MN

-- Prove the area of WXYZ is 1, given the conditions
theorem WXYZ_area (P Q R S M N T U W X Y Z : Type) 
  (rect : Rectangle P Q R S M N T U)
  (mid : Midpoints W X Y Z) :
  true :=
begin
  sorry
end

end WXYZ_area_l186_186900


namespace three_digit_integers_with_odd_factors_l186_186389

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l186_186389


namespace excellent_pairs_l186_186000

theorem excellent_pairs (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) :
  (∀ a b : ℕ, Nat.coprime a b ∧ a > 0 ∧ b > 0 ∧ a ∣ x^3 + y^3 ∧ b ∣ x^3 + y^3 → a + b - 1 ∣ x^3 + y^3) →
  (∃ k : ℕ, (x = 2^k ∧ y = 2^k) ∨ (x = 2 * 3^k ∧ y = 3^k) ∨ (x = 3^k ∧ y = 2 * 3^k)) :=
by
  sorry

end excellent_pairs_l186_186000


namespace three_digit_perfect_squares_count_l186_186183

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l186_186183


namespace three_digit_perfect_squares_count_l186_186201

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l186_186201


namespace unique_solution_count_l186_186603

noncomputable def rectangle_solution_count (a b : ℝ) (h : a < b) : ℕ := 
  if (a < b) then 
    let x := a / 2;
        y := b / 2 in
    if (x < b ∧ y < b ∧ 2 * (x + y) = a + b ∧ x * y = ab / 2) then 1 else 0
  else 
    0

theorem unique_solution_count (a b : ℝ) (h : a < b) : 
  rectangle_solution_count a b h = 1 := 
sorry

end unique_solution_count_l186_186603


namespace sally_additional_savings_l186_186578

-- Definitions based on conditions
def initial_savings : ℕ := 28
def parking_cost : ℕ := 10
def entrance_fee : ℕ := 55
def meal_pass_cost : ℕ := 25
def souvenirs_cost : ℕ := 40
def hotel_cost : ℕ := 80
def distance_one_way : ℕ := 165
def car_mpg : ℕ := 30
def gas_price : ℕ := 3

-- Proposition to be proved
theorem sally_additional_savings :
  let total_travel_distance := 2 * distance_one_way in
  let gallons_needed := total_travel_distance / car_mpg in
  let gas_cost := gallons_needed * gas_price in
  let total_trip_cost := parking_cost + entrance_fee + meal_pass_cost + souvenirs_cost + hotel_cost + gas_cost in
  total_trip_cost - initial_savings = 215 :=
by
  sorry

end sally_additional_savings_l186_186578


namespace find_z_to_8_l186_186040

noncomputable def complex_number_z (z : ℂ) : Prop :=
  z + z⁻¹ = 2 * Complex.cos (Real.pi / 4)

theorem find_z_to_8 (z : ℂ) (h : complex_number_z z) : (z ^ 8 + (z ^ 8)⁻¹ = 2) :=
by
  sorry

end find_z_to_8_l186_186040


namespace product_of_roots_eq_l186_186770

theorem product_of_roots_eq : 
  let p1 := (3 * x^4 + 2 * x^3 - 9 * x^2 + 15 : ℝ[x])
  let p2 := (4 * x^2 - 20 * x + 25 : ℝ[x])
  let resulting_poly := p1 * p2
  ∃ c : ℝ, resulting_poly.leading_coeff * resulting_poly.constant_term = c ∧
          c = 375 / 12 :=
by 
  sorry

end product_of_roots_eq_l186_186770


namespace teacher_li_sheets_l186_186594

theorem teacher_li_sheets (x : ℕ)
    (h1 : ∀ (n : ℕ), n = 24 → (x / 24) = ((x / 32) + 2)) :
    x = 192 := by
  sorry

end teacher_li_sheets_l186_186594


namespace three_digit_perfect_squares_count_l186_186199

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l186_186199


namespace three_digit_integers_with_odd_number_of_factors_l186_186176

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l186_186176


namespace twins_ages_sum_equals_20_l186_186535

def sum_of_ages (A K : ℕ) := 2 * A + K

theorem twins_ages_sum_equals_20 (A K : ℕ) (h1 : A = A) (h2 : A * A * K = 256) : 
  sum_of_ages A K = 20 :=
by
  sorry

end twins_ages_sum_equals_20_l186_186535


namespace three_digit_oddfactors_count_l186_186131

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l186_186131


namespace three_digit_integers_with_odd_factors_l186_186216

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l186_186216


namespace number_of_ways_ab_together_l186_186795

theorem number_of_ways_ab_together (A B C D E : Type) : 
  let people := [A, B, C, D, E],
      ab_entity := [(A, B), (B, A)]
  in (∃ ways : ℕ, ways = 4! * 2!) →
     48 = 4! * 2! := by
  sorry

end number_of_ways_ab_together_l186_186795


namespace solve_for_a_l186_186830

theorem solve_for_a (b c m a : ℝ) (h : m = (b * c^2 * a^2) / (c - a)) : 
  a = (-m + real.sqrt (m^2 + 4 * b * m * c^3)) / (2 * b * c^2) ∨ 
  a = (-m - real.sqrt (m^2 + 4 * b * m * c^3)) / (2 * b * c^2) := 
by 
  sorry

end solve_for_a_l186_186830


namespace number_of_three_digit_integers_with_odd_number_of_factors_l186_186329

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l186_186329


namespace num_three_digit_integers_with_odd_factors_l186_186363

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l186_186363


namespace three_digit_integers_with_odd_factors_count_l186_186301

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l186_186301


namespace number_of_three_digit_integers_with_odd_factors_l186_186420

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l186_186420


namespace angle_between_planes_ABD_CBD_l186_186516

variables (A B C D : Type) [Point A B C D] (plane_ABC : Plane A B C) (plane_ABD : Plane A B D) (plane_CBD : Plane C B D)
variable (alpha : Real)

-- In the pyramid \(A B C D\), the angle \( \alpha \) is \( \angle A B C \)
-- and the orthogonal projection of \( D \) onto the plane \(A B C\) is \( B \).
axiom angle_ABC_alpha : angle A B C = alpha
axiom ortho_proj_D_on_ABC_is_B : orthogonalProjection D plane_ABC = B

-- Prove that the angle between the planes \( A B D \) and \( C B D \) is \( \alpha \).
theorem angle_between_planes_ABD_CBD :
  angle_between_planes plane_ABD plane_CBD = alpha := by
  sorry

end angle_between_planes_ABD_CBD_l186_186516


namespace three_digit_integers_with_odd_factors_l186_186305

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l186_186305


namespace non_congruent_right_triangles_unique_l186_186001

theorem non_congruent_right_triangles_unique :
  ∃! (a: ℝ) (b: ℝ) (c: ℝ), a > 0 ∧ b = 2 * a ∧ c = a * Real.sqrt 5 ∧
  (3 * a + a * Real.sqrt 5 - a^2 = a * Real.sqrt 5) :=
by
  sorry

end non_congruent_right_triangles_unique_l186_186001


namespace total_amount_spent_correct_l186_186690

-- Definitions based on conditions
def price_of_food_before_tax_and_tip : ℝ := 140
def sales_tax_rate : ℝ := 0.10
def tip_rate : ℝ := 0.20

-- Definitions of intermediate steps
def sales_tax : ℝ := sales_tax_rate * price_of_food_before_tax_and_tip
def total_before_tip : ℝ := price_of_food_before_tax_and_tip + sales_tax
def tip : ℝ := tip_rate * total_before_tip
def total_amount_spent : ℝ := total_before_tip + tip

-- Theorem statement to be proved
theorem total_amount_spent_correct : total_amount_spent = 184.80 :=
by
  sorry -- Proof is skipped

end total_amount_spent_correct_l186_186690


namespace three_digit_integers_with_odd_number_of_factors_l186_186171

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l186_186171


namespace input_value_l186_186632

def largest_integer_not_exceeding (x : ℝ) : ℤ :=
  floor x

def z (y : ℤ) : ℤ :=
  2 ^ y - y

theorem input_value (x : ℝ) (hx : largest_integer_not_exceeding x = 5) :
  z (largest_integer_not_exceeding x) = 27 ↔ x = 5.5 :=
by
  sorry

end input_value_l186_186632


namespace three_digit_oddfactors_count_is_22_l186_186087

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l186_186087


namespace circle_line_intersection_points_l186_186833

noncomputable def radius : ℝ := 6
noncomputable def distance : ℝ := 5

theorem circle_line_intersection_points :
  radius > distance -> number_of_intersection_points = 2 := 
by
  sorry

end circle_line_intersection_points_l186_186833


namespace three_digit_integers_with_odd_factors_l186_186144

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l186_186144


namespace train_crossing_bridge_l186_186869

theorem train_crossing_bridge :
  ∀ (length_train length_bridge : ℕ) (speed_kmph : ℕ),
  length_train = 100 →
  length_bridge = 120 →
  speed_kmph = 36 →
  (length_train + length_bridge) / (speed_kmph * 1000 / 3600) = 22 :=
by
  intros length_train length_bridge speed_kmph hlt hlb hsk
  rw [hlt, hlb, hsk]
  norm_num
  -- Here, norm_num is used to simplify the arithmetic involving constants.
  sorry

end train_crossing_bridge_l186_186869


namespace three_digit_integers_with_odd_number_of_factors_count_l186_186270

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l186_186270


namespace santino_fruit_total_l186_186582

-- Definitions of the conditions
def numPapayaTrees : ℕ := 2
def numMangoTrees : ℕ := 3
def papayasPerTree : ℕ := 10
def mangosPerTree : ℕ := 20
def totalFruits (pTrees : ℕ) (pPerTree : ℕ) (mTrees : ℕ) (mPerTree : ℕ) : ℕ :=
  (pTrees * pPerTree) + (mTrees * mPerTree)

-- Theorem that states the total number of fruits is 80 given the conditions
theorem santino_fruit_total : totalFruits numPapayaTrees papayasPerTree numMangoTrees mangosPerTree = 80 := 
  sorry

end santino_fruit_total_l186_186582


namespace three_digit_integers_odd_factors_count_l186_186455

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l186_186455


namespace three_digit_integers_with_odd_factors_l186_186149

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l186_186149


namespace three_digit_integers_with_odd_factors_l186_186308

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l186_186308


namespace distance_from_center_l186_186652

-- Define the circle equation as a predicate
def isCircle (x y : ℝ) : Prop :=
  x^2 + y^2 = 2 * x - 4 * y + 8

-- Define the center of the circle
def circleCenter : ℝ × ℝ := (1, -2)

-- Define the point in question
def point : ℝ × ℝ := (-3, 4)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the proof problem
theorem distance_from_center :
  ∀ (x y : ℝ), isCircle x y → distance circleCenter point = 2 * Real.sqrt 13 :=
by
  sorry

end distance_from_center_l186_186652


namespace dart_lands_in_center_square_l186_186715

theorem dart_lands_in_center_square (s : ℝ) (h : 0 < s) :
  let center_square_area := (s / 2) ^ 2
  let triangle_area := 1 / 2 * (s / 2) ^ 2
  let total_triangle_area := 4 * triangle_area
  let total_board_area := center_square_area + total_triangle_area
  let probability := center_square_area / total_board_area
  probability = 1 / 3 :=
by
  sorry

end dart_lands_in_center_square_l186_186715


namespace three_digit_integers_with_odd_number_of_factors_l186_186177

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l186_186177


namespace three_digit_integers_with_odd_factors_l186_186204

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l186_186204


namespace three_digit_oddfactors_count_l186_186123

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l186_186123


namespace trapezium_area_l186_186785

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 13) :
  (1 / 2) * (a + b) * h = 247 :=
by
  sorry

end trapezium_area_l186_186785


namespace inverse_proposition_l186_186664

theorem inverse_proposition (L₁ L₂ : Line) (a₁ a₂ : Angle) :
  -- Condition: If L₁ is parallel to L₂, then alternate interior angles are equal
  (L₁ ∥ L₂ → a₁ = a₂) →
  -- Proposition to prove: If alternate interior angles are equal, then L₁ is parallel to L₂
  (a₁ = a₂ → L₁ ∥ L₂) :=
by
  sorry

end inverse_proposition_l186_186664


namespace cory_fruit_orders_l186_186754

-- Defining the conditions
def num_apples : ℕ := 4
def num_oranges : ℕ := 3
def num_bananas : ℕ := 2
def total_fruits := num_apples + num_oranges + num_bananas

-- Mathematical problem
theorem cory_fruit_orders : total_fruits = 9 ∧ 
                            fact total_fruits / (fact num_apples * fact num_oranges * fact num_bananas) = 1260 := by
  sorry

end cory_fruit_orders_l186_186754


namespace car_speed_l186_186667

theorem car_speed (v : ℝ) (h1 : 1 / 900 * 3600 = 4) (h2 : 1 / v * 3600 = 6) : v = 600 :=
by
  sorry

end car_speed_l186_186667


namespace three_digit_odds_factors_count_l186_186344

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l186_186344


namespace three_digit_integers_odd_factors_count_l186_186450

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l186_186450


namespace probability_of_drawing_white_ball_l186_186503

def total_balls (red white : ℕ) : ℕ := red + white

def number_of_white_balls : ℕ := 2

def number_of_red_balls : ℕ := 3

def probability_of_white_ball (white total : ℕ) : ℚ := white / total

-- Theorem statement
theorem probability_of_drawing_white_ball :
  probability_of_white_ball number_of_white_balls (total_balls number_of_red_balls number_of_white_balls) = 2 / 5 :=
sorry

end probability_of_drawing_white_ball_l186_186503


namespace number_of_three_digit_integers_with_odd_factors_l186_186410

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l186_186410


namespace sum_of_digits_of_a_l186_186669

-- Define a as 10^10 - 47
def a : ℕ := (10 ^ 10) - 47

-- Function to compute the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- The theorem to prove that the sum of all the digits of a is 81
theorem sum_of_digits_of_a : sum_of_digits a = 81 := by
  sorry

end sum_of_digits_of_a_l186_186669


namespace max_regions_divided_by_circles_l186_186524

-- Define the function P representing the maximum number of regions created by n circles
def P (n : ℕ) : ℕ := n * (n - 1) + 2

-- The proof statement for the maximum number of regions created by n circles
theorem max_regions_divided_by_circles (n : ℕ) : P(n) = n * (n - 1) + 2 := 
by
  sorry

end max_regions_divided_by_circles_l186_186524


namespace custom_operation_example_l186_186624

def custom_operation (a b : ℝ) : ℝ := a - (a / b)

theorem custom_operation_example : custom_operation 8 4 = 6 :=
by
  sorry

end custom_operation_example_l186_186624


namespace number_of_three_digit_integers_with_odd_number_of_factors_l186_186325

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l186_186325


namespace three_digit_integers_with_odd_factors_l186_186400

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l186_186400


namespace a4_eq_pm6_l186_186041

variable {a : ℕ → ℝ}
axiom geo_seq : ∀ n : ℕ, a (n+1) = a(0) * (a(1) / a(0)) ^ n
axiom a2a6_eq_36 : a 2 * a 6 = 36

theorem a4_eq_pm6 : a 4 = 6 ∨ a 4 = -6 :=
by
  sorry

end a4_eq_pm6_l186_186041


namespace three_digit_integers_with_odd_factors_l186_186235

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l186_186235


namespace f_increasing_intervals_l186_186843

def f (x : ℝ) : ℝ := sin x + cos (x + π / 6)

theorem f_increasing_intervals :
  ∃ k ∈ ℤ, ∀ x ∈ Icc (2 * k * π - 5 * π / 6) (2 * k * π + π / 6), f' x > 0 :=
sorry

end f_increasing_intervals_l186_186843


namespace number_of_three_digit_integers_with_odd_number_of_factors_l186_186334

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l186_186334


namespace population_net_increase_l186_186893

-- Definitions for birth and death rate, and the number of seconds in a day
def birth_rate : ℕ := 10
def death_rate : ℕ := 2
def seconds_in_day : ℕ := 86400

-- Calculate the population net increase in one day
theorem population_net_increase (birth_rate death_rate seconds_in_day : ℕ) :
  (seconds_in_day / 2) * birth_rate - (seconds_in_day / 2) * death_rate = 345600 :=
by
  sorry

end population_net_increase_l186_186893


namespace number_of_three_digit_integers_with_odd_number_of_factors_l186_186326

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l186_186326


namespace clock_angle_at_5_15_l186_186074

/--
At 5:15 p.m., the hour hand is at 157.5 degrees, and the minute hand is at 90 degrees.
We need to prove that the smaller angle between the hour and minute hands is 67.5 degrees.
-/
theorem clock_angle_at_5_15 :
  let hour_degrees := 150.0 + 7.5,
      minute_degrees := 15.0 * 6.0 in
  abs (hour_degrees - minute_degrees) = 67.5 :=
by
  let hour_degrees := 150.0 + 7.5
  let minute_degrees := 15.0 * 6.0
  have : abs (hour_degrees - minute_degrees) = 67.5 := sorry
  exact this

end clock_angle_at_5_15_l186_186074


namespace three_digit_perfect_squares_count_l186_186186

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l186_186186


namespace three_digit_integers_with_odd_number_of_factors_count_l186_186275

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l186_186275


namespace number_and_sum_of_f3_is_zero_l186_186546

theorem number_and_sum_of_f3_is_zero 
  {f : ℝ → ℝ} 
  (h : ∀ x y : ℝ, f (x * f y + 2 * x) = 2 * x * y + f x) : 
  let n := 2 in 
  let s := f 3 / real.sqrt 2 + -f 3 / real.sqrt 2 in 
  n * s = 0 := 
by
  sorry

end number_and_sum_of_f3_is_zero_l186_186546


namespace find_line_m_eq_l186_186773

-- Given conditions as definitions
def line_ell (x y : ℝ) : Prop := 5 * x - y = 0
def P : ℝ × ℝ := (-1, 4)
def P'' : ℝ × ℝ := (4, 1)
def intersects_at_origin (L1 L2 : ℝ → ℝ → Prop) : Prop := L1 0 0 ∧ L2 0 0

-- Theorem statement to prove 
theorem find_line_m_eq (
  h1 : ∀ x y, line_ell x y ↔ 5 * x - y = 0,
  h2 : intersects_at_origin line_ell (λ x y, x * a + y * b = 0),
  h3 : P = (-1, 4),
  h4 : P'' = (4, 1)
  ) : exists a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (λ x y, 2 * x - 3 * y = 0) x y ∧ (a * x + b * y = 0) x y := sorry

end find_line_m_eq_l186_186773


namespace T_10_value_l186_186062

noncomputable def a (n : ℕ) : ℝ := 11 - 2 * n

def T (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), |a i|

theorem T_10_value : T 10 = 50 :=
  by sorry

end T_10_value_l186_186062


namespace cube_skew_lines_l186_186693

theorem cube_skew_lines (cube : Prop) (diagonal : Prop) (edges : Prop) :
  ( ∃ n : ℕ, n = 6 ) :=
by
  sorry

end cube_skew_lines_l186_186693


namespace num_three_digit_integers_with_odd_factors_l186_186362

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l186_186362


namespace points_on_line_l186_186569

theorem points_on_line (n : ℕ) (P : Fin n → ℝ × ℝ)
  (h : ∀ (i j : Fin n), i ≠ j → ∃ k : Fin n, k ≠ i ∧ k ≠ j ∧ collinear (P i) (P j) (P k)) :
  ∃ a b : ℝ, ∀ i : Fin n, (P i).2 = a * (P i).1 + b :=
by sorry

end points_on_line_l186_186569


namespace trajectory_of_center_l186_186044

-- Define the fixed circle B with center at (-sqrt(5), 0) and radius 6
def circle_B := { x : ℝ | ∃ y : ℝ, (x + sqrt 5)^2 + y^2 = 36 }

-- Define the point A(sqrt(5), 0)
def point_A := (sqrt 5, 0)

-- Define the trajectory E as an ellipse
def trajectory_E := { (x, y) : ℝ × ℝ | (x^2 / 9) + (y^2 / 4) = 1 }

-- Define the range of values for x + 2y
noncomputable def range_x_plus_2y := { x : ℝ | -5 ≤ x ∧ x ≤ 5 }

theorem trajectory_of_center
    (P : ℝ × ℝ)
    (h1 : P ∈ circle_B)
    (h2 : ∃ y : ℝ, P.1^2 / 9 + P.2^2 / 4 = 1)
    (h3 : P = point_A ∨ P = (-sqrt 5, 0)) :
    P ∈ trajectory_E ∧ (P.1 + 2 * P.2 ∈ range_x_plus_2y) :=
by
  sorry

end trajectory_of_center_l186_186044


namespace number_of_subsets_sum_to_2008_l186_186792

def set_A : Finset ℕ := Finset.range 64  -- This gives the set {0, 1, ..., 63}

def sum_eq_2008 (S : Finset ℕ) : Prop := S.sum id = 2008

theorem number_of_subsets_sum_to_2008 :
  (set_A.filter sum_eq_2008).card = 6 :=
sorry

end number_of_subsets_sum_to_2008_l186_186792


namespace daria_weeks_needed_l186_186756

-- Defining the parameters and conditions
def initial_amount : ℕ := 20
def weekly_savings : ℕ := 10
def cost_of_vacuum_cleaner : ℕ := 120

-- Defining the total money Daria needs to add to her initial amount
def additional_amount_needed : ℕ := cost_of_vacuum_cleaner - initial_amount

-- Defining the number of weeks needed to save the additional amount, given weekly savings
def weeks_needed : ℕ := additional_amount_needed / weekly_savings

-- The theorem stating that Daria needs exactly 10 weeks to cover the expense of the vacuum cleaner
theorem daria_weeks_needed : weeks_needed = 10 := by
  sorry

end daria_weeks_needed_l186_186756


namespace number_of_three_digit_integers_with_odd_factors_l186_186415

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l186_186415


namespace three_digit_integers_with_odd_factors_l186_186307

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l186_186307


namespace cone_surface_area_is_correct_l186_186712

noncomputable def cone_surface_area (central_angle_degrees : ℝ) (sector_area : ℝ) : ℝ :=
  if central_angle_degrees = 120 ∧ sector_area = 3 * Real.pi then 4 * Real.pi else 0

theorem cone_surface_area_is_correct :
  cone_surface_area 120 (3 * Real.pi) = 4 * Real.pi :=
by
  -- proof would go here
  sorry

end cone_surface_area_is_correct_l186_186712


namespace three_digit_integers_with_odd_factors_count_l186_186300

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l186_186300


namespace f_properties_l186_186936

noncomputable def f : ℕ → ℕ := sorry

theorem f_properties (f : ℕ → ℕ) :
  (∀ x y : ℕ, x > 0 → y > 0 → f (x * y) = f x + f y) →
  (f 10 = 16) →
  (f 40 = 24) →
  (f 3 = 5) →
  (f 800 = 44) :=
by
  intros h1 h2 h3 h4
  sorry

end f_properties_l186_186936


namespace three_digit_integers_with_odd_factors_count_l186_186288

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l186_186288


namespace three_digit_integers_with_odd_factors_l186_186386

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l186_186386


namespace Tony_midpoint_age_l186_186968

-- Definitions based on the conditions
noncomputable def Tony_hours_per_day : ℝ := 3
noncomputable def Tony_daily_rate (age : ℝ) := 2.25 * age
noncomputable def total_days : ℝ := 60
noncomputable def Tony_starting_age : ℝ := 10
noncomputable def Tony_age_at_midpoint : ℝ := 11
noncomputable def Tony_age_at_end : ℝ := 12
noncomputable def Tony_total_earnings : ℝ := 1125

-- Lean 4 statement to prove
theorem Tony_midpoint_age :
  let y := 30 in
  2.25 * Tony_starting_age * y + 2.25 * Tony_age_at_midpoint * (30 - y) + 2.25 * Tony_age_at_end * 30 = Tony_total_earnings → 
  Tony_age_at_midpoint = 11 := sorry

end Tony_midpoint_age_l186_186968


namespace glue_pieces_equivalence_l186_186971

variable (initial_pieces : ℕ) (minutes_for_2_piece_gluing : ℕ)

def pieces_gluing_time_2_piece : Prop :=
  initial_pieces = 121 ∧ minutes_for_2_piece_gluing = 120

def pieces_gluing_time_3_piece (initial_pieces minutes_for_3_piece_gluing : ℕ) : Prop :=
  initial_pieces = 121 ∧ minutes_for_3_piece_gluing = 60

theorem glue_pieces_equivalence :
  ∀ (initial_pieces minutes_for_2_piece_gluing : ℕ), pieces_gluing_time_2_piece initial_pieces minutes_for_2_piece_gluing →
  pieces_gluing_time_3_piece initial_pieces 60 :=
by
  intros initial_pieces minutes_for_2_piece_gluing h
  rw [pieces_gluing_time_2_piece, pieces_gluing_time_3_piece] at h
  cases h
  exact sorry

end glue_pieces_equivalence_l186_186971


namespace competition_participation_l186_186702

theorem competition_participation (n : ℕ) (k : ℕ) (p : ℕ) (first_part : ℕ) (solve : ℕ → set ℕ) :
  n = 18 ∧ p = 28 ∧ k = 7 ∧ (∀ x y, x ≠ y → (∃ pp1 pp2, pp1 ∈ solve x ∧ pp2 ∈ solve x ∧ pp1 ≠ pp2 ∧ pp1 ∈ solve y ∧ pp2 ∈ solve y)) →
  (∃ x, (solve x ∩ set.range (λ i, i < first_part) = ∅) ∨ (set.card (solve x ∩ set.range (λ i, i < first_part)) ≥ 4)) :=
sorry

end competition_participation_l186_186702


namespace solve_equation_l186_186779

theorem solve_equation : ∃ x : ℝ, x^2 + 6 * x + 6 * x * real.sqrt (x + 5) = 24 ∧ x = (-17 + real.sqrt 277) / 2 :=
by
  sorry

end solve_equation_l186_186779


namespace three_digit_perfect_squares_count_l186_186196

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l186_186196


namespace cruise_ship_sunglasses_l186_186692

theorem cruise_ship_sunglasses :
  let total_passengers := 2500 in
  let adults := (1 / 2) * total_passengers in
  let children := (3 / 10) * total_passengers in
  let number_of_men := adults / 2 in
  let number_of_women := adults / 2 in
  let women_wearing_sunglasses := 0.15 * number_of_women in
  let men_wearing_sunglasses := 0.12 * number_of_men in
  let children_wearing_sunglasses := 0.1 * children in
  let total_wearing_sunglasses := women_wearing_sunglasses + men_wearing_sunglasses + children_wearing_sunglasses in
  total_wearing_sunglasses ≈ 244 :=
  by
  let total_passengers := 2500
  let adults := (1 / 2) * total_passengers
  let children := (3 / 10) * total_passengers
  let number_of_men := adults / 2
  let number_of_women := adults / 2
  let women_wearing_sunglasses := 0.15 * number_of_women
  let men_wearing_sunglasses := 0.12 * number_of_men
  let children_wearing_sunglasses := 0.1 * children
  let total_wearing_sunglasses := women_wearing_sunglasses + men_wearing_sunglasses + children_wearing_sunglasses
  linarith

end cruise_ship_sunglasses_l186_186692


namespace number_of_incorrect_conclusions_is_1_l186_186952

def f (x : ℝ) : ℝ := sorry

-- Conditions
axiom domain_of_f : ∀ x : ℝ, f x ∈ ℝ
axiom odd_f_x_minus_1 : ∀ x : ℝ, f (-x-1) = -f (x-1)
axiom even_f_x_plus_1 : ∀ x : ℝ, f (-x+1) = f (x+1)
axiom when_x_in_neg1_1 : ∀ x : ℝ, -1 < x ∧ x < 1 → f x = -x^2 + 1

-- Prove that the number of incorrect conclusions is 1
theorem number_of_incorrect_conclusions_is_1 :
  (f (7 / 2) = -3 / 4 ∧ 
   (∀ x : ℝ, (f (x+7) = -f x)) ∧
   (∀ x : ℝ, 6 < x ∧ x < 8 →  f x > f (x+1)) ∧
   (∀ x : ℝ, f (x) = f (x+8))) → 
  (num_of_incorrect_conclusions = 1) :=
by
  sorry

end number_of_incorrect_conclusions_is_1_l186_186952


namespace spring_mass_relationship_l186_186897

theorem spring_mass_relationship (x y : ℕ) (h1 : y = 18 + 2 * x) : 
  y = 32 → x = 7 :=
by
  sorry

end spring_mass_relationship_l186_186897


namespace three_digit_integers_with_odd_number_of_factors_l186_186173

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l186_186173


namespace constant_term_expansion_l186_186599

theorem constant_term_expansion : 
  ∃ r : ℕ, (9 - 3 * r / 2 = 0) ∧ 
  ∀ (x : ℝ) (hx : x ≠ 0), (2 * x - 1 / Real.sqrt x) ^ 9 = 672 := 
by sorry

end constant_term_expansion_l186_186599


namespace projection_y_ge_one_l186_186049

variable {n : ℕ} (h1 : 0 < n)
variable (A : Fin (2 * n) → ℝ × ℝ)
variable (h2 : ∀ i, (A i).2 ≥ 0)
variable (h3 : ∀ i, (A i).1^2 + (A i).2^2 = 1)
variable (h4 : (Finset.univ.sum (λ i, (A i).1)).toNat % 2 = 1)

noncomputable def v : ℝ × ℝ :=
(∑ i : Fin (2 * n), (A i).1, ∑ i : Fin (2 * n), (A i).2)

theorem projection_y_ge_one : (v A).2 ≥ 1 :=
sorry

end projection_y_ge_one_l186_186049


namespace sum_of_divisors_85_l186_186655

theorem sum_of_divisors_85 :
  let p1 := 5
  let p2 := 17
  let n := p1 * p2
  n = 85 -> (1 + p1) * (1 + p2) = 108 := 
by
  assume h1 : 85 = 5 * 17
  sorry

end sum_of_divisors_85_l186_186655


namespace total_number_of_doves_l186_186813

-- Definition of the problem conditions
def initial_doves : ℕ := 20
def eggs_per_dove : ℕ := 3
def hatch_rate : ℚ := 3/4

-- The Lean 4 statement that expresses the proof problem
theorem total_number_of_doves (initial_doves eggs_per_dove : ℕ) (hatch_rate : ℚ) : 
  initial_doves + (hatch_rate * (initial_doves * eggs_per_dove)) = 65 := 
by {
  -- Defining intermediate quantities
  let total_eggs := initial_doves * eggs_per_dove,
  let hatched_eggs := hatch_rate * total_eggs,

  -- Proving the final number of doves
  have : initial_doves + hatched_eggs = 65,
    calc
      initial_doves + hatched_eggs
          = 20 + (3/4 * 60) :   by sorry
      ... = 20 + 45           : by sorry
      ... = 65                : by sorry,
  exact this
}

end total_number_of_doves_l186_186813


namespace desc_order_XYZ_l186_186989

def X : ℝ := 0.6 * 0.5 + 0.4
def Y : ℝ := 0.6 * 0.5 / 0.4
def Z : ℝ := 0.6 * 0.5 * 0.4

theorem desc_order_XYZ : Y > X ∧ X > Z :=
by
  have hX : X = 0.7, from by norm_num ; have hY : Y = 0.75, from by norm_num ; have hZ : Z = 0.12, from by norm_num ;
  sorry

end desc_order_XYZ_l186_186989


namespace value_of_a_l186_186507

theorem value_of_a (a : ℝ) (x y : ℝ) : 
  (x + a^2 * y + 6 = 0 ∧ (a - 2) * x + 3 * a * y + 2 * a = 0) ↔ a = -1 :=
by
  sorry

end value_of_a_l186_186507


namespace simplify_expression_l186_186980

theorem simplify_expression (k : ℂ) : 
  ((1 / (3 * k)) ^ (-3) * ((-2) * k) ^ (4)) = 432 * (k ^ 7) := 
by sorry

end simplify_expression_l186_186980


namespace solve_toenail_problem_l186_186861

def toenail_problem (b_toenails r_toenails_already r_toenails_more : ℕ) : Prop :=
  (b_toenails = 20) ∧
  (r_toenails_already = 40) ∧
  (r_toenails_more = 20) →
  (r_toenails_already + r_toenails_more = 60)

theorem solve_toenail_problem : toenail_problem 20 40 20 :=
by {
  sorry
}

end solve_toenail_problem_l186_186861


namespace sarah_total_profit_l186_186583

theorem sarah_total_profit :
  ∀ (regular_price hot_price cost_per_cup : ℝ)
    (hot_days regular_days : ℕ),
    hot_price = 1.6351744186046513 →
    cost_per_cup = 0.75 →
    hot_days = 3 →
    regular_days = 7 →
    32 * hot_days * hot_price + 
    32 * regular_days * regular_price - 
    320 * cost_per_cup = 210.2265116279069 → 
    regular_price = 1.308139534883721 → 
    regular_price + 0.25 * regular_price = hot_price → 
    true :=
begin
  sorry
end

end sarah_total_profit_l186_186583


namespace clock_angle_at_5_15_l186_186075

/--
At 5:15 p.m., the hour hand is at 157.5 degrees, and the minute hand is at 90 degrees.
We need to prove that the smaller angle between the hour and minute hands is 67.5 degrees.
-/
theorem clock_angle_at_5_15 :
  let hour_degrees := 150.0 + 7.5,
      minute_degrees := 15.0 * 6.0 in
  abs (hour_degrees - minute_degrees) = 67.5 :=
by
  let hour_degrees := 150.0 + 7.5
  let minute_degrees := 15.0 * 6.0
  have : abs (hour_degrees - minute_degrees) = 67.5 := sorry
  exact this

end clock_angle_at_5_15_l186_186075


namespace three_digit_integers_with_odd_factors_l186_186241

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l186_186241


namespace three_digit_integers_with_odd_factors_l186_186153

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l186_186153


namespace roy_older_than_julia_l186_186576

variable {R J K x : ℝ}

theorem roy_older_than_julia (h1 : R = J + x)
                            (h2 : R = K + x / 2)
                            (h3 : R + 2 = 2 * (J + 2))
                            (h4 : (R + 2) * (K + 2) = 192) :
                            x = 2 :=
by
  sorry

end roy_older_than_julia_l186_186576


namespace three_digit_integers_with_odd_factors_l186_186207

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l186_186207


namespace sum_of_zeros_l186_186474

noncomputable def f : ℝ → ℝ := sorry

theorem sum_of_zeros (h_symm : ∀ x, f (1 - x) = f (3 + x))
  (h_zeros : ∃ a b c, f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ (a ≠ b ∧ b ≠ c ∧ c ≠ a)) :
  (∃ a b c, a + b + c = 6) :=
begin
  sorry
end

end sum_of_zeros_l186_186474


namespace sqrt_4_of_63504000_l186_186747

theorem sqrt_4_of_63504000 :
  (4 : ℝ).sqrt_root (63504000 : ℝ) = 2 * (square_root (2 : ℝ)) * (square_root (3 : ℝ)) * (4 : ℝ).sqrt_root (11 : ℝ) * 10 ^ (3 / 4 : ℝ) := 
  sorry

end sqrt_4_of_63504000_l186_186747


namespace triangle_vector_sum_zero_l186_186885

theorem triangle_vector_sum_zero
  (A B C : ℝ → ℝ → Prop) :
  (let AB := λ i : ℝ, B i - A i,
       BC := λ i : ℝ, C i - B i,
       CA := λ i : ℝ, A i - C i in
  AB + BC + CA = λ i : ℝ, 0) :=
by
  sorry

end triangle_vector_sum_zero_l186_186885


namespace orthographic_projection_rhombus_not_rhombus_l186_186625

theorem orthographic_projection_rhombus_not_rhombus
    (P : Type) [Planar P]
    (rhombus : P) -- assume we have a rhombus in a plane
    (is_rhombus : is_rhombus rhombus) -- assume the proof that it is a rhombus
    (orthographic_projection : P → P) -- assume the function of orthographic projection
    (lines_x_unchanged : ∀ (l : Line) (orthographic_projection l = l) (parallel_x : parallel_to_x l), true) -- condition that lines parallel to x-axis are unchanged
    (lines_y_halved : ∀ (l : Line) (parallel_y : parallel_to_y l), length (orthographic_projection l) = (1/2) * length l) -- condition that the length parallel to y-axis is halved
    : ¬ is_rhombus (orthographic_projection rhombus) :=
by
  sorry

end orthographic_projection_rhombus_not_rhombus_l186_186625


namespace forgotten_angles_sum_l186_186969

theorem forgotten_angles_sum (n : ℕ) (h : (n-2) * 180 = 3240 + x) : x = 180 :=
by {
  sorry
}

end forgotten_angles_sum_l186_186969


namespace number_of_three_digit_integers_with_odd_number_of_factors_l186_186335

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l186_186335


namespace problem_l186_186834

noncomputable theory

-- Definitions based on conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_mean (a3 a2 a4 : ℝ) : Prop :=
  a3 ^ 2 = a2 * (a4 + 1)

-- Problem definitions
def a₁ : ℝ := 2
def a (n : ℕ) : ℝ := 2 * n

def b (n : ℕ) : ℝ := 2 / ((n + 3) * (a n + 2))

def sum_first_n (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum b

-- Lean 4 Theorem
theorem problem (H1 : arithmetic_sequence a) 
  (H2 : a 1 = a₁)
  (H3 : is_geometric_mean (a 3) (a 2) (a 4)) :
  (∀ n: ℕ, a n = 2 * n) ∧ 
  (∀ n: ℕ, sum_first_n b n = (5/12) - ((2*n + 5) / (2 * (n + 2) * (n + 3)))) :=
by
  sorry

end problem_l186_186834


namespace three_digit_oddfactors_count_l186_186132

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l186_186132


namespace number_of_three_digit_squares_l186_186435

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l186_186435


namespace find_point_M_l186_186036

variable {x1 y1 x2 y2 : ℝ}
variable {λ : ℝ}

noncomputable def point_M (x1 y1 x2 y2 λ : ℝ) : ℝ × ℝ :=
 ((1 - λ) * x1 + λ * x2, (1 - λ) * y1 + λ * y2)

theorem find_point_M (hx1 hy1 hx2 hy2 hλ : ℝ) (hλ_nonneg : 0 ≤ hλ) :
  ∃ (x0 y0 : ℝ), point_M hx1 hy1 hx2 hy2 hλ = (x0, y0) ∧
    x0 = (1 - hλ) * hx1 + hλ * hx2 ∧
    y0 = (1 - hλ) * hy1 + hλ * hy2 :=
  sorry

end find_point_M_l186_186036


namespace pens_sold_day_one_l186_186564

theorem pens_sold_day_one :
  ∃ (P : ℕ), (13 * 48 = P + 12 * 44) ∧ P = 96 :=
by
  use 96
  split
  · exact Nat.mul_pos 13 48
  · exact rfl
  sorry

end pens_sold_day_one_l186_186564


namespace three_digit_integers_with_odd_factors_l186_186315

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l186_186315


namespace isosceles_triangle_perimeter_l186_186484

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 2) (h2 : b = 4) (h3 : a ≠ b) (h4 : a + b > b) (h5 : a + b > a) 
: ∃ p : ℝ, p = 10 :=
by
  -- Using the given conditions to determine the perimeter
  sorry

end isosceles_triangle_perimeter_l186_186484


namespace radio_loss_percentage_l186_186600

theorem radio_loss_percentage :
  ∀ (cost_price selling_price : ℝ), 
    cost_price = 1500 → 
    selling_price = 1290 → 
    ((cost_price - selling_price) / cost_price) * 100 = 14 :=
by
  intros cost_price selling_price h_cp h_sp
  sorry

end radio_loss_percentage_l186_186600


namespace three_digit_integers_with_odd_factors_l186_186147

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l186_186147


namespace f_at_neg_one_l186_186801

theorem f_at_neg_one :
    ∃ (p q r : ℝ), ∀ (g f : ℝ → ℝ),
      g = (λ x, x^3 + p*x^2 - 5*x + 15) →
      (∀ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a → 
       g a = 0 ∧ g b = 0 ∧ g c = 0 → f a = 0 ∧ f b = 0 ∧ f c = 0) →
      f = (λ x, x^4 + x^3 + q*x^2 + 50*x + r) →
      f (-1) = 78 :=
begin
    sorry
end

end f_at_neg_one_l186_186801


namespace number_of_three_digit_integers_with_odd_number_of_factors_l186_186336

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l186_186336


namespace pentagon_proof_l186_186559

theorem pentagon_proof
  (P A B C D E : Point)  -- Points defining the pentagon and intersection point
  (h_len : distance A B = distance B C ∧ distance B C = distance C D ∧ distance C D = distance D E ∧ distance D E = distance E A) -- All sides of equal length
  (h_angle_bcd : angle B C D = 90)
  (h_angle_cde : angle C D E = 90)
  (h_a_outside : ¬ inside_quadrilateral A B C D E) -- A is outside the quadrilateral
  (h_p_intersection : P = line_intersection (line A C) (line B D)) : -- P is intersection of lines AC and BD
  distance A P = distance P D := 
sorry

end pentagon_proof_l186_186559


namespace avg_nested_l186_186987

def avg (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem avg_nested {x y z : ℕ} :
  avg (avg 2 3 1) (avg 4 1 0) 5 = 26 / 9 :=
by
  sorry

end avg_nested_l186_186987


namespace num_odd_factors_of_three_digit_integers_eq_22_l186_186106

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l186_186106


namespace three_digit_integers_with_odd_factors_l186_186217

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l186_186217


namespace sum_of_zeros_l186_186476

-- Defining the conditions and the result
theorem sum_of_zeros (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = f (3 + x)) (a b c : ℝ)
  (h1 : f a = 0) (h2 : f b = 0) (h3 : f c = 0) : 
  a + b + c = 3 := 
by 
  sorry

end sum_of_zeros_l186_186476


namespace problem_statement_l186_186679

-- Definitions
def div_remainder (a b : ℕ) : ℕ × ℕ :=
  (a / b, a % b)

-- Conditions and question as Lean structures
def condition := ∀ (a b k : ℕ), k ≠ 0 → div_remainder (a * k) (b * k) = (a / b, (a % b) * k)
def question := div_remainder 4900 600 = div_remainder 49 6

-- Theorem stating the problem's conclusion
theorem problem_statement (cond : condition) : ¬question :=
by
  sorry

end problem_statement_l186_186679


namespace geometric_sequence_a3_a5_l186_186905

-- Define the geometric sequence condition using a function
def is_geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

-- Define the given conditions
variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (h1 : is_geometric_seq a)
variable (h2 : a 1 > 0)
variable (h3 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25)

-- The main goal is to prove: a 3 + a 5 = 5
theorem geometric_sequence_a3_a5 : a 3 + a 5 = 5 :=
by
  simp [is_geometric_seq] at h1
  obtain ⟨q, ⟨hq_pos, hq⟩⟩ := h1
  sorry

end geometric_sequence_a3_a5_l186_186905


namespace intersection_points_line_circle_l186_186622

theorem intersection_points_line_circle :
  ∀ t α : ℝ,
  let line := (λ t : ℝ, (2 + t, -1 - t)),
      circle := (λ α : ℝ, (3 * Real.cos α, 3 * Real.sin α)),
      std_line := (λ x y : ℝ, x + y - 1 = 0),
      std_circle := (λ x y : ℝ, x^2 + y^2 = 9),
      d := (1 / Real.sqrt 2)
  in std_line (line t).1 (line t).2 ∧ std_circle (circle α).1 (circle α).2 → d < 3
     → ∃! (x y : ℝ), std_line x y ∧ std_circle x y :=
by
  sorry

end intersection_points_line_circle_l186_186622


namespace find_foci_l186_186750

def hyperbolaFoci : Prop :=
  let eq := ∀ x y, 2 * x^2 - 3 * y^2 + 8 * x - 12 * y - 23 = 0
  ∃ foci : ℝ × ℝ, foci = (-2 - Real.sqrt (5 / 6), -2) ∨ foci = (-2 + Real.sqrt (5 / 6), -2)

theorem find_foci : hyperbolaFoci :=
by
  sorry

end find_foci_l186_186750


namespace number_of_three_digit_integers_with_odd_factors_l186_186411

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l186_186411


namespace power_function_odd_l186_186045

theorem power_function_odd (f : ℝ → ℝ) (α : ℝ) 
  (h1 : ∀ x, f(x) = x^α) 
  (h2 : f (real.sqrt 3 / 3) = real.sqrt 3) : 
  α = -1 ∧ ∀ x, f (-x) = -f(x) := 
by 
  sorry

end power_function_odd_l186_186045


namespace three_digit_oddfactors_count_l186_186126

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l186_186126


namespace num_odd_factors_of_three_digit_integers_eq_22_l186_186105

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l186_186105


namespace perpendicular_lines_a_value_l186_186033

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∃ x y : ℝ, ax + y + 1 = 0) ∧ (∃ x y : ℝ, x + y + 2 = 0) ∧ (∃ x y : ℝ, (y = -ax)) → a = -1 := by
  sorry

end perpendicular_lines_a_value_l186_186033


namespace total_votes_4500_l186_186502

theorem total_votes_4500 (V : ℝ) 
  (h : 0.60 * V - 0.40 * V = 900) : V = 4500 :=
by
  sorry

end total_votes_4500_l186_186502


namespace problem_statement_l186_186660

theorem problem_statement :
  (3.14159.to_nearest_tenth = 3.1) ∧
  ((3.14 * 10^3).accurate_to_nearest_ten) ∧
  ((30 * 1000).accurate_to_nearest_thousand = 30000) ∧
  (precision(3.10) ≠ precision(3.1)) →
  (3.14 * 10^3).accurate_to_nearest_ten := 
sorry

end problem_statement_l186_186660


namespace num_three_digit_integers_with_odd_factors_l186_186370

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l186_186370


namespace num_three_digit_ints_with_odd_factors_l186_186242

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l186_186242


namespace part_I_part_II_l186_186058

-- Define the functions f and g
def f (k : ℝ) (x : ℝ) := k * (x - 1) * Real.exp x + x^2
def g (k : ℝ) (x : ℝ) := x^2 + (k + 2) * x

-- Define the derivative f' for k = 1/2
def f' (x : ℝ) := (1 / 2) * x * Real.exp x + 2 * x

-- Define h corresponding to the equation f'(x) = g(x)
def h (x : ℝ) := Real.exp x - 2 * x - 1

-- Problem I: Prove that h has exactly one zero for x > 0 when k = 1/2
theorem part_I : ∀ x : ℝ, x > 0 → (h x = 0) ↔ (x = Classical.some (Exists.intro x (h x = 0))) :=
sorry

-- Problem II: Prove that the minimum value of f on [k, 1] for k ≤ -1 is 1
theorem part_II (k : ℝ) (h : k ≤ -1) : ∃ m : ℝ, (∀ x ∈ Set.Icc k 1, f k x ≥ m) ∧ m = 1 :=
sorry

end part_I_part_II_l186_186058


namespace congruent_quadrilaterals_l186_186602

def quadrilateral := {A B C D E F G H I J K L : Type*} 
  -- Define the vertices and additional points as types

namespace geometry

variables {V : Type*} [inner_product_space ℝ V] [metric_space V] [normed_group V]

/- Define perpendicular diagonals property -/
def diagonals_perpendicular 
  (A B C D : V) : Prop := 
  inner_product (C - A) (D - B) = 0

/- Define squares erected on sides property -/
def squares_externally_erected (A B C D E F G H I J K L : V) : Prop := 
  (is_square A B E F) ∧ 
  (is_square B C G H) ∧ 
  (is_square C D I J) ∧ 
  (is_square D A K L)

/- Define intersections forming points -/
def intersect_lines (X Y Z W : V) (P: V) : Prop := 
  ∃ t, X + t * (Y - X) = P ∧ ∃ s, Z + s * (W - Z) = P

/- Define quadrilateral congruence property -/
def quadrilaterals_congruent (P1 Q1 R1 S1 P2 Q2 R2 S2 : V) : Prop := 
  congruent_quadrangle P1 Q1 R1 S1 P2 Q2 R2 S2

/- Formal statement of the mathematical problem -/
theorem congruent_quadrilaterals
  (A B C D E F G H I J K L P1 Q1 R1 S1 P2 Q2 R2 S2 : V)
  (h_diag_perp : diagonals_perpendicular A B C D)
  (h_squares : squares_externally_erected A B C D E F G H I J K L)
  (h_intersections_1 : intersect_lines C L F D P1 ∧ intersect_lines F D A H Q1 ∧ intersect_lines A H B J R1 ∧ intersect_lines B J C L S1)
  (h_intersections_2 : intersect_lines A I B K P2 ∧ intersect_lines B K C E Q2 ∧ intersect_lines C E D G R2 ∧ intersect_lines D G A I S2) :
  quadrilaterals_congruent P1 Q1 R1 S1 P2 Q2 R2 S2 :=
sorry

end geometry

end congruent_quadrilaterals_l186_186602


namespace solution_proof_l186_186496

-- Define a geometric setting based on the given conditions
variables {A B C D M O : Point}

-- Assumptions corresponding to the problem conditions
axiom mutually_perpendicular_diameters : Perpendicular (Line A B) (Line C D)
axiom M_on_arc_AC : OnArc M A C
axiom sum_of_distances : dist M A + dist M C = a

-- Definition that encapsulates the required relationship
def problem_statement : Prop :=
  dist M B + dist M D = a * (1 + Real.sqrt 2)

-- The statement of our problem in Lean
theorem solution_proof : problem_statement :=
  by
    -- Placeholder for the proof
    sorry

end solution_proof_l186_186496


namespace three_digit_integers_with_odd_number_of_factors_count_l186_186271

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l186_186271


namespace three_digit_odds_factors_count_l186_186342

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l186_186342


namespace cross_section_area_l186_186996

-- Define initial conditions related to the edge length and midpoints
def edge_length := 2
def midpoint (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (0.5 * (a.1 + b.1), 0.5 * (a.2 + b.2), 0.5 * (a.3 + b.3))

-- Define points for the cube vertices
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (0, 0, edge_length)
def C : ℝ × ℝ × ℝ := (0, edge_length, 0)
def D : ℝ × ℝ × ℝ := (0, edge_length, edge_length)
def E : ℝ × ℝ × ℝ := (edge_length, 0, 0)
def F : ℝ × ℝ × ℝ := (edge_length, 0, edge_length)
def G : ℝ × ℝ × ℝ := (edge_length, edge_length, 0)
def H : ℝ × ℝ × ℝ := (edge_length, edge_length, edge_length)

-- Define midpoints used in the plane cut
def Y := midpoint A C
def T := midpoint A E
def V := midpoint E G
def W := midpoint C G
def U := midpoint F H  -- Additional midpoint
def X := midpoint B D  -- Additional midpoint

-- Main theorem statement to prove the area
theorem cross_section_area : area_of_cross_section Y T V W U X = 3 * Real.sqrt 3 :=
by
  -- Proof steps go here
  sorry

end cross_section_area_l186_186996


namespace three_digit_odds_factors_count_l186_186356

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l186_186356


namespace solve_problem_l186_186506

-- Definitions for curves C1 and C2
def C1_polar_equation (θ : ℝ) : ℝ := ρ * (Real.cos θ + Real.sin θ) = 4

def C2_parametric_equation (θ : ℝ) : (ℝ × ℝ) :=
  (2 + 3 * Real.cos θ, 1 + 3 * Real.sin θ)

-- Conditions in Cartesian form
def C1_cartesian (x y : ℝ) : Prop := x + y = 4

def C2_cartesian (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 1) ^ 2 = 9

-- Intersection points A and B
def point_A : ℝ × ℝ := ((3 + Real.sqrt 17) / 2, (5 - Real.sqrt 17) / 2)
def point_B : ℝ × ℝ := ((3 - Real.sqrt 17) / 2, (5 + Real.sqrt 17) / 2)

-- Distance between A and B
def distance_AB (A B : ℝ × ℝ) : ℝ := Real.sqrt 34

-- Maximum area of ΔPAB
def max_area_triangle (A B P : ℝ × ℝ) : ℝ :=
  (3 * Real.sqrt 34 + Real.sqrt 17) / 2

theorem solve_problem :
  (C1_cartesian x y ∧ C2_cartesian x y ∧
    point_A = ((3 + Real.sqrt 17) / 2, (5 - Real.sqrt 17) / 2) ∧
    point_B = ((3 - Real.sqrt 17) / 2, (5 + Real.sqrt 17) / 2) ∧
    distance_AB point_A point_B = Real.sqrt 34) →
  max_area_triangle point_A point_B (2 + 3 * Real.cos θ, 1 + 3 * Real.sin θ) =
    (3 * Real.sqrt 34 + Real.sqrt 17) / 2 :=
sorry

end solve_problem_l186_186506


namespace sum_geometric_arithmetic_progression_l186_186794

theorem sum_geometric_arithmetic_progression :
  ∃ (a b r d : ℝ), a = 1 * r ∧ b = 1 * r^2 ∧ b = a + d ∧ 16 = b + d ∧ (a + b = 12.64) :=
by
  sorry

end sum_geometric_arithmetic_progression_l186_186794


namespace original_game_no_chance_chief_cannot_reclaim_chief_knows_expulsions_max_tribesmen_expelled_natives_could_lose_second_game_l186_186703

theorem original_game_no_chance:
  ∀ (n : ℕ), n = 30 → (∀ a b : ℕ, a ≠ b → a = b → False) → False := sorry

theorem chief_cannot_reclaim:
  ∀ (distributed_coins total_people : ℕ), distributed_coins = 270 → total_people < 30 → 
  (∃ x : ℕ, (x = total_people) ∧ (finset.sum (finset.range x) (λ i, i+1)) ≠ distributed_coins) := sorry

theorem chief_knows_expulsions:
  ∀ (expelled remaining total : ℕ), total = 30 → (remaining = total - expelled) → ∃ n, n = expelled := sorry

theorem max_tribesmen_expelled:
  ∀ (distributed_coins : ℕ), distributed_coins = 270 → 
  ∃ max_expul : ℕ, max_expul ≤ 6 ∧ 
  (∃ unique_coins : finset ℕ, unique_coins.card = (30 - max_expul) ∧ 
  finset.sum unique_coins (λ i, i+1) = distributed_coins) := sorry

theorem natives_could_lose_second_game:
  ∀ (merchant_lost_first_game natives_won_first_game : Prop), 
  merchant_lost_first_game → natives_won_first_game → natives_won_first_game →
  (merchant_lost_first_game → ¬ (natives_won_first_game → True)) := sorry

end original_game_no_chance_chief_cannot_reclaim_chief_knows_expulsions_max_tribesmen_expelled_natives_could_lose_second_game_l186_186703


namespace three_digit_integers_with_odd_factors_l186_186320

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l186_186320


namespace num_true_propositions_is_3_l186_186730

-- Define each proposition corresponding to the similarity of geometric figures.
def proposition_1 : Prop := ∀ (T1 T2 : Triangle), isIsosceles T1 ∧ isIsosceles T2 → Similar T1 T2
def proposition_2 : Prop := ∀ (T1 T2 : Triangle), isEquilateral T1 ∧ isEquilateral T2 → Similar T1 T2
def proposition_3 : Prop := ∀ (R1 R2 : Rectangle), Similar R1 R2
def proposition_4 : Prop := ∀ (Rh1 Rh2 : Rhombus), Similar Rh1 Rh2
def proposition_5 : Prop := ∀ (Sq1 Sq2 : Square), Similar Sq1 Sq2
def proposition_6 : Prop := ∀ (P1 P2 : RegularPolygon), P1.sides = P2.sides → Similar P1 P2

-- Define the proof problem to show the number of true propositions.
theorem num_true_propositions_is_3 :
  (¬ proposition_1) ∧
  proposition_2 ∧
  (¬ proposition_3) ∧
  (¬ proposition_4) ∧
  proposition_5 ∧
  proposition_6 →
  3 = (if proposition_1 then 1 else 0) +
      (if proposition_2 then 1 else 0) +
      (if proposition_3 then 1 else 0) +
      (if proposition_4 then 1 else 0) +
      (if proposition_5 then 1 else 0) +
      (if proposition_6 then 1 else 0) := 
by 
  sorry

end num_true_propositions_is_3_l186_186730


namespace no_pre_period_decimal_representation_l186_186570

theorem no_pre_period_decimal_representation (m : ℕ) (h : Nat.gcd m 10 = 1) : ¬∃ k : ℕ, ∃ a : ℕ, 0 < a ∧ 10^a < m ∧ (10^a - 1) % m = k ∧ k ≠ 0 :=
sorry

end no_pre_period_decimal_representation_l186_186570


namespace problem_statement_l186_186021

def f : ℝ → ℝ
| x := if x ≤ 0 then f (x + 1) else 2 * x

theorem problem_statement : f (-4/3) + f (4/3) = 4 :=
by
  sorry

end problem_statement_l186_186021


namespace three_digit_oddfactors_count_is_22_l186_186089

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l186_186089


namespace number_of_three_digit_integers_with_odd_number_of_factors_l186_186337

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l186_186337


namespace three_digit_integers_with_odd_number_of_factors_count_l186_186264

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l186_186264


namespace ellipse_major_axis_length_l186_186732

theorem ellipse_major_axis_length :
  ∀ (F1 F2 : ℝ × ℝ) (C : ℝ × ℝ) (a : ℝ),
  (F1 = (3, 1 + sqrt 3)) →
  (F2 = (3, 1 - sqrt 3)) →
  (C = (3, 1)) →
  (a = 3) →
  (∃ R : ℝ, ellipse_tangent_to_lines C a (0, 0) (4, 0) ∧
             ellipse_has_foci C a F1 F2 ↔ R = 6) :=
by sorry

end ellipse_major_axis_length_l186_186732


namespace f_strictly_increasing_g_minus_f_solution_in_interval_l186_186056

def f (x : ℝ) : ℝ := Real.log (2^(x : ℝ) - 1) / Real.log 2
def g (x : ℝ) : ℝ := Real.log (2^(x : ℝ) + 1) / Real.log 2

theorem f_strictly_increasing (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x < y) : f x < f y := 
sorry

theorem g_minus_f_solution_in_interval (m : ℝ) :
  ∃ x ∈ Set.Icc 1 2, g x - f x = m →
  (Real.log 5 / 3 / Real.log 2 ≤ m) ∧ (m ≤ Real.log 3 / Real.log 2) :=
sorry

end f_strictly_increasing_g_minus_f_solution_in_interval_l186_186056


namespace ron_spending_increase_l186_186480

variable (P Q : ℝ) -- initial price and quantity
variable (X : ℝ)   -- intended percentage increase in spending

theorem ron_spending_increase :
  (1 + X / 100) * P * Q = 1.25 * P * (0.92 * Q) →
  X = 15 := 
by
  sorry

end ron_spending_increase_l186_186480


namespace increasing_range_of_a_l186_186881

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 1 / x

theorem increasing_range_of_a (a : ℝ) : (∀ x > (1/2), (3 * x^2 + a - 1 / x^2) ≥ 0) ↔ a ≥ (13 / 4) :=
by sorry

end increasing_range_of_a_l186_186881


namespace sufficient_but_not_necessary_condition_l186_186828

theorem sufficient_but_not_necessary_condition (a : ℝ) (h₁ : a > 2) : a ≥ 1 ∧ ¬(∀ (a : ℝ), a ≥ 1 → a > 2) := 
by
  sorry

end sufficient_but_not_necessary_condition_l186_186828


namespace number_of_three_digit_integers_with_odd_number_of_factors_l186_186324

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l186_186324


namespace three_digit_integers_with_odd_number_of_factors_l186_186170

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l186_186170


namespace red_pens_count_l186_186492

-- Define the conditions given in the problem
def num_green_pens : ℕ := 5
def num_black_pens : ℕ := 6
def prob_not_red_nor_green : ℝ := 1 / 3

-- Define the proof goal
theorem red_pens_count (R : ℕ) (total_pens : ℕ) :
  total_pens = num_green_pens + num_black_pens + R →
  prob_not_red_nor_green = num_black_pens / total_pens →
  R = 7 := by
  sorry

end red_pens_count_l186_186492


namespace Alex_has_more_than_200_marbles_on_Monday_of_next_week_l186_186727

theorem Alex_has_more_than_200_marbles_on_Monday_of_next_week :
  ∃ k : ℕ, k > 0 ∧ 3 * 2^k > 200 ∧ k % 7 = 1 := by
  sorry

end Alex_has_more_than_200_marbles_on_Monday_of_next_week_l186_186727


namespace count_zeros_in_decimal_l186_186865

theorem count_zeros_in_decimal (n d : ℕ) (h₀ : n = 1) (h₁ : d = 2^7 * 5^3) :
  ∃ k, (\(n / d) = (k / 10^7) ∧ (0 < k < 10^7) ∧ (natLength (natToDigits 10 k) = 3)) ∧ natLength (takeWhile (ArrEq 0) (natToDigits 10 ((n / d))) - 1) = 5 := 
sorry

end count_zeros_in_decimal_l186_186865


namespace three_digit_perfect_squares_count_l186_186194

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l186_186194


namespace three_digit_integers_with_odd_factors_l186_186156

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l186_186156


namespace three_digit_integers_with_odd_factors_count_l186_186292

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l186_186292


namespace three_digit_integers_with_odd_factors_l186_186313

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l186_186313


namespace three_digit_integers_with_odd_factors_l186_186212

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l186_186212


namespace number_of_three_digit_integers_with_odd_factors_l186_186403

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l186_186403


namespace three_digit_integers_with_odd_number_of_factors_l186_186175

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l186_186175


namespace sequence_properties_l186_186030

open Nat

def sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, n > 0 → S n = (finset.range n).sum (λ i, a (i + 1))) ∧
  (a 1 = 1) ∧
  (∀ n : ℕ, n > 0 → a (n + 1) = 2 * S n + 1)

theorem sequence_properties
  (a S : ℕ → ℕ)
  (h : sequence a S) :
  a 2 = 3 ∧ ∀ n : ℕ, n > 0 → a n = 3^(n - 1) :=
by
  sorry

end sequence_properties_l186_186030


namespace polygon_area_l186_186768

open Real

structure Point where
  x : ℝ
  y : ℝ

def shoelace_formula (vertices : List Point) : ℝ :=
  (0.5 * | (List.sum (List.zipWith (λ (i : Point) (j : Point), i.x * j.y - i.y * j.x)
  vertices (List.tail vertices ++ [List.head vertices])))).abs

def vertices : List Point :=
  [⟨0, 1⟩, ⟨3, 4⟩, ⟨7, 1⟩, ⟨3, 7⟩]

theorem polygon_area : shoelace_formula vertices = 10.5 :=
  sorry

end polygon_area_l186_186768


namespace jake_fewer_peaches_undetermined_l186_186527

theorem jake_fewer_peaches_undetermined 
    (steven_peaches : ℕ) 
    (steven_apples : ℕ) 
    (jake_fewer_peaches : steven_peaches > jake_peaches) 
    (jake_more_apples : jake_apples = steven_apples + 3) 
    (steven_peaches_val : steven_peaches = 9) 
    (steven_apples_val : steven_apples = 8) : 
    ∃ n : ℕ, jake_peaches = n ∧ ¬(∃ m : ℕ, steven_peaches - jake_peaches = m) := 
sorry

end jake_fewer_peaches_undetermined_l186_186527


namespace cyclic_quadrilateral_if_b_squared_plus_c_squared_minus_a_squared_eq_3R_squared_l186_186542

-- Definitions of the points and conditions 
variable {A B C H O' N D : Type}
variable (triangle_ABC : Triangle A B C)
variable (H_is_orthocenter : Orthocenter H triangle_ABC)
variable (O'_is_circumcenter_BHC : Circumcenter O' (Triangle B H C))
variable (N_is_midpoint_AO' : Midpoint N A O')
variable (D_is_reflection_of_N_across_BC : Reflection D N B C)
variable (a : Length BC)
variable (b : Length CA)
variable (c : Length AB)
variable (R : Length (Circumradius (Triangle A B C)))

-- Question statement 
theorem cyclic_quadrilateral_if_b_squared_plus_c_squared_minus_a_squared_eq_3R_squared :
  are_concyclic_quad (Quadrilateral A B D C) ↔ b^2 + c^2 - a^2 = 3 * R^2 := 
sorry

end cyclic_quadrilateral_if_b_squared_plus_c_squared_minus_a_squared_eq_3R_squared_l186_186542


namespace three_digit_oddfactors_count_is_22_l186_186099

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l186_186099


namespace num_three_digit_ints_with_odd_factors_l186_186259

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l186_186259


namespace inverse_proposition_l186_186663

theorem inverse_proposition (L₁ L₂ : Line) (a₁ a₂ : Angle) :
  -- Condition: If L₁ is parallel to L₂, then alternate interior angles are equal
  (L₁ ∥ L₂ → a₁ = a₂) →
  -- Proposition to prove: If alternate interior angles are equal, then L₁ is parallel to L₂
  (a₁ = a₂ → L₁ ∥ L₂) :=
by
  sorry

end inverse_proposition_l186_186663


namespace find_principal_l186_186670

noncomputable def principal_amount (P : ℝ) (r : ℝ) : Prop :=
  (800 = (P * r * 2) / 100) ∧ (820 = P * (1 + r / 100)^2 - P)

theorem find_principal (P : ℝ) (r : ℝ) (h : principal_amount P r) : P = 8000 :=
by
  sorry

end find_principal_l186_186670


namespace johnny_total_earnings_l186_186533

def first_job_pay_rate : ℝ := 3.25
def first_job_hours : ℝ := 8
def overtime_threshold : ℝ := 6
def overtime_multiplier : ℝ := 1.5
def second_job_pay_rate : ℝ := 4.50
def second_job_hours : ℝ := 5
def total_earnings (regular_rate : ℝ) (hours : ℝ) (overtime_threshold : ℝ) (overtime_multiplier : ℝ) (second_rate : ℝ) (second_hours : ℝ) : ℝ :=
  let regular_hours := min hours overtime_threshold
  let overtime_hours := max 0 (hours - overtime_threshold)
  let regular_earnings := regular_hours * regular_rate
  let overtime_earnings := overtime_hours * (regular_rate * overtime_multiplier)
  let first_job_total := regular_earnings + overtime_earnings
  let second_job_total := second_hours * second_rate
  first_job_total + second_job_total

theorem johnny_total_earnings : 
  total_earnings first_job_pay_rate first_job_hours overtime_threshold overtime_multiplier second_job_pay_rate second_job_hours = 58.25 := 
  sorry

end johnny_total_earnings_l186_186533


namespace three_digit_integers_with_odd_factors_l186_186225

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l186_186225


namespace num_three_digit_ints_with_odd_factors_l186_186260

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l186_186260


namespace min_value_of_a2_b2_l186_186473

noncomputable def f (x a b : ℝ) := Real.exp x + a * x + b

theorem min_value_of_a2_b2 {a b : ℝ} (h : ∃ t ∈ Set.Icc (1 : ℝ) (3 : ℝ), f t a b = 0) :
  a^2 + b^2 ≥ (Real.exp 1)^2 / 2 :=
by
  sorry

end min_value_of_a2_b2_l186_186473


namespace gain_percentage_equal_25_l186_186716

variable (P Q : ℝ)

theorem gain_percentage_equal_25 (h₁ : 80 * P / 60 * P = 2/3) (h₂ : 120 * Q / 90 * Q = 4/3) :
  GainPercentage P 80 60 = (1:ℝ):
  GainPercentage Q 120 90 = (1:ℝ) :=
by
  sorry

end gain_percentage_equal_25_l186_186716


namespace solution_set_of_inequality_l186_186009

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 - x + 2 ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
by
  sorry

end solution_set_of_inequality_l186_186009


namespace cos_angle_a_b_l186_186069

-- Given vectors a and b
def a : ℝ × ℝ := (2, -4)
def b : ℝ × ℝ := (-3, -4)

-- Definition of dot product for 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Definition of magnitude of a 2D vector
def magnitude (v : ℝ × ℝ) : ℝ := 
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- The cosine of the angle between vectors a and b
def cos_theta : ℝ :=
  dot_product a b / (magnitude a * magnitude b)

-- Theorem: Cosine of the angle between vectors a and b is sqrt(5)/5
theorem cos_angle_a_b : cos_theta = real.sqrt 5 / 5 :=
by
  sorry

end cos_angle_a_b_l186_186069


namespace banker_l186_186597

-- Define the given conditions
def BD : ℝ := 2150
def R : ℝ := 12
def n : ℝ := 6

-- Define the formula for Banker's Gain
def BG (BD R n : ℝ) : ℝ := (BD * R * n) / (100 + (R * n))

-- Prove that the Banker's Gain is 900
theorem banker's_gain : BG BD R n = 900 := by
  unfold BG
  simp [BD, R, n]
  sorry

end banker_l186_186597


namespace three_digit_oddfactors_count_l186_186134

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l186_186134


namespace number_of_three_digit_squares_l186_186441

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l186_186441


namespace cell_remains_uncut_in_grid_l186_186485

theorem cell_remains_uncut_in_grid :
  ∃ cell : ℕ × ℕ, ∀ placement : list (ℕ × ℕ), 
    (∀ rect : ℕ × ℕ, rect ∈ placement → 
      (rect.fst + 5 < 11 ∨ rect.fst - 5 > 0)) ∧ 
    (placement.length > (11 * 11 / 6)) → 
    cell ∉ placement :=
sorry

end cell_remains_uncut_in_grid_l186_186485


namespace percentage_increase_l186_186523

theorem percentage_increase (x : ℝ) : 
  (1 + x / 100)^2 = 1.1025 → x = 5.024 := 
sorry

end percentage_increase_l186_186523


namespace simplify_expression_l186_186979

theorem simplify_expression (k : ℂ) : 
  ((1 / (3 * k)) ^ (-3) * ((-2) * k) ^ (4)) = 432 * (k ^ 7) := 
by sorry

end simplify_expression_l186_186979


namespace breakfast_cost_l186_186806

def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3

def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2

def kiera_muffins : ℕ := 2
def kiera_fruit_cup : ℕ := 1

theorem breakfast_cost :
  muffin_cost * francis_muffins + fruit_cup_cost * francis_fruit_cups
  + muffin_cost * kiera_muffins + fruit_cup_cost * kiera_fruit_cup = 17 :=
by
  -- skipping proof
  sorry

end breakfast_cost_l186_186806


namespace part_I_part_II_l186_186842

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + 2 * real.sqrt 3 * cos x ^ 2 - real.sqrt 3

theorem part_I :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π :=
sorry

theorem part_II {x : ℝ} (h : -π / 3 ≤ x ∧ x ≤ π / 12) :
  f x ≥ -real.sqrt 3 :=
sorry

end part_I_part_II_l186_186842


namespace inequality_proof_l186_186560

variables {a b c d e f : ℝ}

theorem inequality_proof (h1 : ∀ x : ℝ, a * x^2 + b * x + c ≥ |d * x^2 + e * x + f|) :
  4 * a * c - b^2 ≥ |4 * d * f - e^2| :=
by
  intro x
  sorry

end inequality_proof_l186_186560


namespace correct_number_to_multiply_l186_186695

theorem correct_number_to_multiply 
  (num_to_multiply : ℕ) (mistaken_number : ℕ) (difference : ℕ) :
    num_to_multiply = 138 →
    mistaken_number = 34 →
    difference = 1242 →
    let x := 43 in
    num_to_multiply * x = num_to_multiply * mistaken_number + difference :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end correct_number_to_multiply_l186_186695


namespace min_value_of_sum_inverses_l186_186465

theorem min_value_of_sum_inverses (x y : ℝ) (hx : 2^x + 2^y = 5) : 2⁻¹^x + 2⁻¹^y ≥ 4/5 :=
sorry

end min_value_of_sum_inverses_l186_186465


namespace intersection_subset_l186_186852

def set_A : Set ℝ := {x | -4 < x ∧ x < 2}
def set_B : Set ℝ := {x | x > 1 ∨ x < -5}
def set_C (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < m}

theorem intersection_subset (m : ℝ) :
  (set_A ∩ set_B) ⊆ set_C m ↔ m = 2 :=
by
  sorry

end intersection_subset_l186_186852


namespace calculate_p_q_r_s_sum_l186_186947

structure Point where
  x : ℚ
  y : ℚ

def A : Point := ⟨0, 0⟩
def B : Point := ⟨2, 3⟩
def C : Point := ⟨4, 4⟩
def D : Point := ⟨5, 0⟩

def area_of_quadrilateral (A B C D : Point) : ℚ :=
  (1 / 2) * (abs ((A.x * B.y - B.x * A.y) + (B.x * C.y - C.x * B.y) + (C.x * D.y - D.x * C.y) + (D.x * A.y - A.x * D.y)))

def intersection_point (A C D : Point) : Point :=
  let slope_CD := (D.y - C.y) / (D.x - C.x)
  let intercept_CD := C.y - slope_CD * C.x
  let x_intersect := (-A.y + intercept_CD) / slope_CD
  ⟨⟨x_intersect, -4 * x_intersect + 20⟩⟩

theorem calculate_p_q_r_s_sum : 
  let inter_pt := intersection_point A C D
  let p := inter_pt.x.num
  let q := inter_pt.x.den
  let r := inter_pt.y.num
  let s := inter_pt.y.den
  p + q + r + s = 25 := by
  -- Using sorry to skip the proof
  sorry

end calculate_p_q_r_s_sum_l186_186947


namespace distance_from_point_to_line_example_l186_186994

def point := ℝ × ℝ
def line := ℝ × ℝ × ℝ

def distance_point_to_line (p : point) (l : line) : ℝ :=
  let (x₀, y₀) := p
  let (A, B, C) := l
  (abs (A * x₀ + B * y₀ + C)) / (sqrt (A^2 + B^2))

theorem distance_from_point_to_line_example :
  ∀ (P : point) (L : line), P = (1, -2) → L = (3, -4, -1) → distance_point_to_line P L = 2 :=
by
  intros P L hP hL
  rw [hP, hL]
  sorry

end distance_from_point_to_line_example_l186_186994


namespace three_digit_integers_with_odd_factors_count_l186_186297

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l186_186297


namespace smaller_angle_at_6_30_l186_186866
-- Import the Mathlib library

-- Define the conditions as a structure
structure ClockAngleConditions where
  hours_on_clock : ℕ
  degrees_per_hour : ℕ
  minute_hand_position : ℕ
  hour_hand_position : ℕ

-- Initialize the conditions for 6:30
def conditions : ClockAngleConditions := {
  hours_on_clock := 12,
  degrees_per_hour := 30,
  minute_hand_position := 180,
  hour_hand_position := 195
}

-- Define the theorem to be proven
theorem smaller_angle_at_6_30 (c : ClockAngleConditions) : 
  c.hour_hand_position - c.minute_hand_position = 15 :=
by
  -- Skip the proof
  sorry

end smaller_angle_at_6_30_l186_186866


namespace number_of_three_digit_integers_with_odd_factors_l186_186417

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l186_186417


namespace three_digit_integers_with_odd_factors_l186_186228

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l186_186228


namespace cannot_place_blocks_l186_186023

def block_dim : List ℕ := [3, 3, 1]
def box_dim : List ℕ := [7, 9, 11]
def num_blocks : ℕ := 77

theorem cannot_place_blocks :
  ¬ ∃ (arrangement : List (ℕ × ℕ × ℕ)),
    (∀ b ∈ arrangement, b ∈ (permutations block_dim)) ∧
    size arrangement = num_blocks ∧
    arrangement_fits_in_box arrangement box_dim :=
by
  sorry

end cannot_place_blocks_l186_186023


namespace equation_solution_l186_186778

theorem equation_solution (x : ℝ) (h : x ≠ 1 ∧ x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5) :
  (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
   1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 4) ↔ 
  (x = 3 + 2 * Real.sqrt 5 ∨ x = 3 - 2 * Real.sqrt 5) := 
sorry

end equation_solution_l186_186778


namespace num_odd_factors_of_three_digit_integers_eq_22_l186_186118

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l186_186118


namespace three_digit_integers_with_odd_number_of_factors_l186_186172

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l186_186172


namespace three_digit_integers_with_odd_factors_l186_186310

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l186_186310


namespace three_digit_integers_with_odd_factors_l186_186399

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l186_186399


namespace number_of_three_digit_squares_l186_186424

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l186_186424


namespace f_is_odd_f_value_x_l186_186840

-- Define the function f
def f (x : ℝ) : ℝ := (2^x + 1) / (2^x - 1)

-- Statement (I): Define the domain of the function
def domain_f : Set ℝ := { x | x ≠ 0 }

-- Statement (II): Prove the function is odd
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f (x) := 
by
  sorry

-- Statement (III): Given f(x) = -5/3, find the value of x
theorem f_value_x (x : ℝ) : f (x) = -5 / 3 → x = -2 := 
by
  sorry

end f_is_odd_f_value_x_l186_186840


namespace remainder_x3_plus_3x2_l186_186007

noncomputable def remainder_of_polynomial_division (f g : Polynomial ℝ) : Polynomial ℝ :=
  Polynomial.modByMonic f (Polynomial.leadingCoeff g • Polynomial.monic g)

theorem remainder_x3_plus_3x2 (x : ℝ) :
  remainder_of_polynomial_division (Polynomial.X ^ 3 + 3 * Polynomial.X ^ 2) 
                                   (Polynomial.X ^ 2 + 4 * Polynomial.X + 2) 
  = - Polynomial.X ^ 2 - 2 * Polynomial.X :=
sorry

end remainder_x3_plus_3x2_l186_186007


namespace three_digit_integers_with_odd_factors_l186_186397

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l186_186397


namespace bisector_sum_squared_l186_186471

variable (l l' : ℝ)
variable (a b c R S : ℝ)
variable (A B C : Point) -- Points A, B, C that make up the triangle
variable (triangle : Triangle A B C)
variable [IsBisector l (Angle C)]
variable [IsExteriorBisector l' (Angle C)]
variable [Circumcircle triangle = Circle (radius R)]
variable [Area triangle = S]

theorem bisector_sum_squared :
  l^2 + l'^2 = (64 * R^2 * S^2) / (a^2 - b^2)^2 :=
  sorry

end bisector_sum_squared_l186_186471


namespace three_digit_perfect_squares_count_l186_186185

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l186_186185


namespace possible_timetables_proof_l186_186488

-- Define the problem conditions
def compulsory_subjects := ["Chinese", "Mathematics", "English"]
def elective_subjects := ["Physics", "Chemistry", "Biology", "History", "Geography", "Politics"]
def compulsory_elective_subjects := ["Physics", "Chemistry"]
def total_classes := 8
def morning_classes := 4
def afternoon_classes := 4
def self_study_sessions := ["SelfStudy1", "SelfStudy2"]

-- Define constraints
def constraints (timetable : List String) : Prop :=
  -- Constraint 3: Fourth class in the morning and the first class in the afternoon are not adjacent
  timetable.nth 3 != timetable.nth 4 ∧
  -- Constraint 4: Last two classes are self-study
  timetable.nth 6 = "SelfStudy1" ∧
  timetable.nth 7 = "SelfStudy2" ∧
  -- Constraint 5: Mathematics cannot be scheduled for the first class in the afternoon
  timetable.nth 4 != "Mathematics" ∧
  -- Constraint 6: Chinese and English cannot be adjacent
  ∀ i, i < total_classes - 1 → (timetable.nth i = "Chinese" → timetable.nth (i+1) != "English") ∧
       (timetable.nth i = "English" → timetable.nth (i+1) != "Chinese")

-- Define the problem statement
def possible_timetables : Nat :=
  4 * (108 + 336)

theorem possible_timetables_proof : possible_timetables = 1776 :=
  sorry

end possible_timetables_proof_l186_186488


namespace calculate_expression_l186_186744

theorem calculate_expression : 5 * 12 + 6 * 11 - 2 * 15 + 7 * 9 = 159 := by
  sorry

end calculate_expression_l186_186744


namespace num_three_digit_integers_with_odd_factors_l186_186381

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l186_186381


namespace three_digit_integers_with_odd_number_of_factors_count_l186_186272

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l186_186272


namespace smallest_n_not_dividing_square_l186_186008

theorem smallest_n_not_dividing_square (n : ℕ) (hn : 1000 ≤ n ∧ n < 10000) 
  (S_n := n * (n + 1) / 2) (P_n := (nat.fact n) ^ 2) :
  (∃ n : ℕ, (1000 ≤ n ∧ n < 10000) ∧ (P_n % S_n ≠ 0) ∧ 
  (∀ m : ℕ, (1000 ≤ m ∧ m < 10000) → (P_m % (m * (m + 1) / 2) ≠ 0) → m ≥ n)) :=
begin
  sorry
end

end smallest_n_not_dividing_square_l186_186008


namespace sum_a_16_to_20_l186_186046

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
axiom S_def : ∀ n, S n = a 0 * (1 - (a 1 / a 0) ^ n) / (1 - (a 1 / a 0))
axiom S_5_eq_2 : S 5 = 2
axiom S_10_eq_6 : S 10 = 6

-- Theorem to prove
theorem sum_a_16_to_20 : a 16 + a 17 + a 18 + a 19 + a 20 = 16 :=
by
  sorry

end sum_a_16_to_20_l186_186046


namespace three_digit_integers_with_odd_factors_l186_186394

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l186_186394


namespace domain_of_sqrt_3x_plus_2_plus_1_div_x_minus_2_l186_186604

def domain_of_function : Set ℝ := { x : ℝ | 3 * x + 2 ≥ 0 ∧ x ≠ 2 }

theorem domain_of_sqrt_3x_plus_2_plus_1_div_x_minus_2 :
  ∀ x : ℝ, (3 * x + 2 ≥ 0 ∧ x ≠ 2) ↔ (x ∈ domain_of_function) :=
by
  intro x
  split
  { intro h
    exact h
  }
  { intro h
    exact h
  }

end domain_of_sqrt_3x_plus_2_plus_1_div_x_minus_2_l186_186604


namespace probability_of_three_primes_from_30_l186_186641

noncomputable def primes_up_to_30 : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_of_three_primes_from_30 :
  ((primes_up_to_30.card.choose 3) / ((Finset.range 31).card.choose 3)) = (6 / 203) :=
by
  sorry

end probability_of_three_primes_from_30_l186_186641


namespace sequence_1953rd_digit_is_6_l186_186993

def sequence_digit(n : ℕ) : ℕ :=
  let blocks := concat (list.range 1 10) (λ n, repeat n n)
  let total_block := flatten (repeat blocks 100)
  total_block[n - 1] -- index from 0

theorem sequence_1953rd_digit_is_6 : sequence_digit 1953 = 6 :=
  sorry

end sequence_1953rd_digit_is_6_l186_186993


namespace three_digit_integers_with_odd_factors_count_l186_186282

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l186_186282


namespace number_of_three_digit_integers_with_odd_number_of_factors_l186_186332

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l186_186332


namespace dividend_rate_of_stock_is_5_l186_186686

-- Definitions based on the conditions
def dividend_yield : ℝ := 10 / 100
def market_value : ℝ := 50

-- The proof problem
theorem dividend_rate_of_stock_is_5 :
  (dividend_yield * market_value) = 5 :=
by
  sorry

end dividend_rate_of_stock_is_5_l186_186686


namespace turtles_order_l186_186796

-- Define variables for each turtle as real numbers representing their positions
variables (O P S E R : ℝ)

-- Define the conditions given in the problem
def condition1 := S = O - 10
def condition2 := S = R + 25
def condition3 := R = E - 5
def condition4 := E = P - 25

-- Define the order of arrival
def order_of_arrival (O P S E R : ℝ) := 
     O = 0 ∧ 
     P = -5 ∧
     S = -10 ∧
     E = -30 ∧
     R = -35

-- Theorem to show the given conditions imply the order of arrival
theorem turtles_order (h1 : condition1 S O)
                     (h2 : condition2 S R)
                     (h3 : condition3 R E)
                     (h4 : condition4 E P) :
  order_of_arrival O P S E R :=
by sorry

end turtles_order_l186_186796


namespace John_traded_in_car_money_back_l186_186662

-- First define the conditions provided in the problem.
def UberEarnings : ℝ := 30000
def CarCost : ℝ := 18000
def UberProfit : ℝ := 18000

-- We need to prove that John got $6000 back when trading in the car.
theorem John_traded_in_car_money_back : 
  UberEarnings - UberProfit = CarCost - 6000 := 
by
  -- provide the detailed steps inside the proof block if needed
  sorry

end John_traded_in_car_money_back_l186_186662


namespace ratio_of_radii_of_pyramid_l186_186614

variables {α β γ : Real}
variables (S : Real) (cos_alpha cos_beta cos_gamma : Real)
variables (r r' : Real)

-- Condition: lateral face areas equality and angle definitions
-- s is the area of the base
def s : Real := S * (cos_alpha + cos_beta + cos_gamma)

-- Equating the volumes in terms of inscribed and escribed radii
def volume_eq {r r' : Real} (S : Real) (s : Real) : Prop :=
  (3 * S + s) * r = (3 * S - s) * r'

theorem ratio_of_radii_of_pyramid (h : s = S * (cos_alpha + cos_beta + cos_gamma)) :
  (r : Real) \/
  (r' : Real) \/ 
  α β γ : Real \/
  (S : Real) \/ 
= (3 - cos_alpha - cos_beta - cos_gamma) / (3 + cos_alpha + cos_beta + cos_gamma)


end ratio_of_radii_of_pyramid_l186_186614


namespace trig_function_properties_l186_186574

/-- 
  For the function y = 2 * sin(2 * x + π/4) + 1:
  1. The graph is not symmetric about x = π/4.
  2. The graph can be obtained by changing the abscissa of all points on the graph of y = 2 * sin(x + π/4) + 1 to 1/2 of the original.
  3. The graph is not symmetric about (3π/8, 0).
  4. The range of the function is [-1, 3].
  Verify that the correct statements among these are (2) and (4).
-/
theorem trig_function_properties :
  let f := λ x : ℝ, 2 * Real.sin (2 * x + Real.pi / 4) + 1 in
  ¬ (∀ x, f x = f (π/4 - x)) ∧ 
  (∀ x, (2 * Real.sin (x + Real.pi / 4) + 1) = f (1/2 * x)) ∧ 
  ¬ (f (3*Real.pi/8) = 0) ∧
  (∀ y, y ∈ range f ↔ (-1 ≤ y ∧ y ≤ 3)) →
  (true /\ false /\ false /\ true) := 
begin
  intro h,
  sorry
end

end trig_function_properties_l186_186574


namespace curve_C2_eq_l186_186949

def curve_C (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def reflect_y_axis (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-x)
def reflect_x_axis (f : ℝ → ℝ) (x : ℝ) : ℝ := - (f x)

theorem curve_C2_eq (a b c : ℝ) (h : a ≠ 0) :
  ∀ x : ℝ, reflect_x_axis (reflect_y_axis (curve_C a b c)) x = -a * x^2 + b * x - c := by
  sorry

end curve_C2_eq_l186_186949


namespace three_digit_integers_with_odd_factors_l186_186234

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l186_186234


namespace three_digit_oddfactors_count_l186_186138

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l186_186138


namespace mass_percentage_O_in_KBrO3_l186_186838

theorem mass_percentage_O_in_KBrO3 : 
  mass_percentage (KBrO3) (O) = 28.74 :=
by
  -- Definitions for atomic masses
  let m_K := 39.10 -- g/mol
  let m_Br := 79.90 -- g/mol
  let m_O := 16.00 -- g/mol
  -- Definition for molar mass of KBrO3
  let m_KBrO3 := m_K + m_Br + 3 * m_O
  -- Definition for mass of oxygen in KBrO3
  let mass_O_in_KBrO3 := 3 * m_O
  -- Calculation of mass percentage of oxygen in KBrO3
  let mass_percentage := (mass_O_in_KBrO3 / m_KBrO3) * 100
  have : mass_percentage = 28.74 := by
    calc mass_percentage
      = (3 * m_O / m_KBrO3) * 100 : by rfl
      = (3 * 16.00 / 167.00) * 100 : by norm_num
      = 28.74 : by norm_num
  exact this

end mass_percentage_O_in_KBrO3_l186_186838


namespace three_digit_integers_with_odd_factors_l186_186233

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l186_186233


namespace value_of_k_l186_186932

open Nat

theorem value_of_k (k : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS : ∀ n : ℕ, 0 < n → S n = k * (n : ℝ) ^ 2 + (n : ℝ))
  (h_a : ∀ n : ℕ, 1 < n → a n = S n - S (n-1))
  (h_geom : ∀ m : ℕ, 0 < m → (a m) ≠ 0 → (a (2*m))^2 = a m * a (4*m)) :
  k = 0 ∨ k = 1 :=
sorry

end value_of_k_l186_186932


namespace boy_girl_swaps_l186_186894

theorem boy_girl_swaps (n : ℕ) (h : n = 8) :
  let students := 2 * n in
  let initial := (list.range students).map (λ i, if i % 2 = 0 then "B" else "G") in
  let final := (list.range students).map (λ i, if i < n then "G" else "B") in
  ∑ k in (finset.range n), (n - k) = 36 :=
by
  sorry

end boy_girl_swaps_l186_186894


namespace three_digit_integers_with_odd_factors_l186_186205

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l186_186205


namespace three_digit_integers_with_odd_factors_l186_186154

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l186_186154


namespace three_digit_integers_with_odd_number_of_factors_l186_186178

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l186_186178


namespace three_digit_integers_with_odd_factors_l186_186239

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l186_186239


namespace geom_seq_value_l186_186026

theorem geom_seq_value (a : ℕ → ℝ) (r : ℝ)
    (h0 : a 8 = a 4 * r^4)
    (h1 : a 4 + a 8 = -2)
    (h2 : a 6 = a 4 * r^2)
    (h3 : a 2 = a 4 * r⁻²)
    (h4 : a 10 = a 4 * r^6) :
    a 6 * (a 2 + 2 * a 6 + a 10) = 2 := by
  sorry

end geom_seq_value_l186_186026


namespace compute_dot_product_l186_186934

noncomputable theory

open_locale vector_space

variables (u v w : ℝ^3)

-- Conditions
-- u is a unit vector
axiom u_unit : ∥u∥ = 1
-- v is a unit vector
axiom v_unit : ∥v∥ = 1
-- 2(u × v) + u = 3w
axiom w_eq : 2 * (u × v) + u = 3 * w
-- 3(w × u) = 2v
axiom w_cross : 3 * (w × u) = 2 * v

-- To Prove
theorem compute_dot_product : u ⋅ (v × w) = 1 :=
sorry

end compute_dot_product_l186_186934


namespace exists_cycle_length_not_divisible_by_three_l186_186493

variable (V : Type) [Fintype V] [DecidableEq V]

structure Graph (V : Type) :=
(edges : V → V → Prop)
(symm : ∀ {x y : V}, edges x y → edges y x)
(degree : ∀ v : V, Fintype.card { w : V // edges v w} = 3)

theorem exists_cycle_length_not_divisible_by_three (G : Graph V) : 
  ∃ (cycle : List V), 
  (∀ v ∈ cycle, G.edges v (cycle.head!)) ∧ 
  (cycle.length ∉ {n : ℕ | ∃ m : ℕ, n = 3 * m}) :=
by
  sorry

end exists_cycle_length_not_divisible_by_three_l186_186493


namespace log_domain_l186_186605

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1) / Real.log 2

theorem log_domain :
  ∀ x : ℝ, (∃ y : ℝ, f y = Real.log (x - 1) / Real.log 2) ↔ x ∈ Set.Ioi 1 :=
by {
  sorry
}

end log_domain_l186_186605


namespace number_of_extreme_points_zero_l186_186621

def f (x a : ℝ) : ℝ := x^3 + 3*x^2 + 4*x - a

theorem number_of_extreme_points_zero (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ∀ x, f x1 a = f x a → x = x1 ∨ x = x2) → False := 
by
  sorry

end number_of_extreme_points_zero_l186_186621


namespace max_ab_l186_186509

open Real

-- Define the conditions given in the problem
variables {a b : ℝ}
def circle_eq : (ℝ → ℝ → Prop) :=
  λ x y, (x - a) ^ 2 + (y - b) ^ 2 = 1

def line_eq : (ℝ → ℝ → Prop) :=
  λ x y, x + 2 * y - 1 = 0

-- Define the chord length condition
def chord_length (a b : ℝ) : Prop :=
  |a + 2 * b - 1| / sqrt 5 < 1

-- The main statement to prove
theorem max_ab : 
  (∃ a b : ℝ, circle_eq a b ∩ line_eq a b ∧ chord_length (a + 2 * b - 1) ∧ a > 0 ∧ b > 0) → 
  max (a * b) = 1 / 2 := 
begin
  sorry
end

end max_ab_l186_186509


namespace number_of_special_four_digit_integers_l186_186078

theorem number_of_special_four_digit_integers : 
  ∃ n : ℕ, n = 81 ∧ (∀ x, x ∈ [2, 3, 5] -> x ∈ (list.range 10000).filter (λ d, ∀ i < 4, (d / 10^i) % 10 = 2 ∨ (d / 10^i) % 10 = 3 ∨ (d / 10^i) % 10 = 5)) :=
by
  sorry

end number_of_special_four_digit_integers_l186_186078


namespace three_digit_integers_odd_factors_count_l186_186458

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l186_186458


namespace breakfast_cost_l186_186809

theorem breakfast_cost :
  ∀ (muffin_cost fruit_cup_cost : ℕ) (francis_muffins francis_fruit_cups kiera_muffins kiera_fruit_cups : ℕ),
  muffin_cost = 2 ∧ fruit_cup_cost = 3 ∧ francis_muffins = 2 ∧ francis_fruit_cups = 2 ∧ kiera_muffins = 2 ∧ kiera_fruit_cups = 1
  → (francis_muffins * muffin_cost + francis_fruit_cups * fruit_cup_cost + kiera_muffins * muffin_cost + kiera_fruit_cups * fruit_cup_cost = 17) :=
by
  intros muffin_cost fruit_cup_cost francis_muffins francis_fruit_cups kiera_muffins kiera_fruit_cups
  intro cond
  cases cond with muffin_cost_eq rest
  cases rest with fruit_cup_cost_eq rest
  cases rest with francis_muffins_eq rest
  cases rest with francis_fruit_cups_eq rest
  cases rest with kiera_muffins_eq kiera_fruit_cups_eq

  rw [muffin_cost_eq, fruit_cup_cost_eq, francis_muffins_eq, francis_fruit_cups_eq, kiera_muffins_eq, kiera_fruit_cups_eq]
  norm_num
  sorry

end breakfast_cost_l186_186809


namespace three_digit_integers_with_odd_factors_l186_186230

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l186_186230


namespace dot_product_self_eq_two_l186_186068

noncomputable def a : ℝ × ℝ × ℝ := (Real.sin (Real.pi / 15), Real.cos (Real.pi / 15), -1)

theorem dot_product_self_eq_two : 
  let a := (Real.sin (Real.pi / 15), Real.cos (Real.pi / 15), -1)
  in (a.1 * a.1 + a.2 * a.2 + a.3 * a.3) = 2 :=
by
  sorry

end dot_product_self_eq_two_l186_186068


namespace ratio_of_areas_l186_186910

theorem ratio_of_areas (R r : ℝ) (A B C A1 B1 C1 : ℝ) 
  (h₁ : is_circumradius R A B C)
  (h₂ : is_inradius r A B C)
  (h₃ : is_angle_bisector_intersects_circumcircle A1 B1 C1 A B C)
  : area_ratio (triangle_area A B C) (triangle_area A1 B1 C1) = 2 * r / R :=
sorry

end ratio_of_areas_l186_186910


namespace three_digit_integers_with_odd_factors_l186_186215

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l186_186215


namespace polynomial_not_factorable_l186_186954

open Polynomial

theorem polynomial_not_factorable {n : ℕ} (a : Finₓ n → ℤ) (h_distinct : Function.Injective a) :
  ¬∃ (g h : ℤ[X]), 
    degree g > 0 ∧ degree h > 0 ∧ (∃ c : ℤ, c * g * h = 
    ∏ i in Finset.finRange n, (X - C (a i)) - 1) := 
by
  sorry

end polynomial_not_factorable_l186_186954


namespace Fedya_can_construct_altitudes_l186_186674

noncomputable def can_construct_altitudes (triangle : Type) (is_acute_angled : Prop) (is_non_equilateral : Prop)
  (has_compass : Prop) (has_straightedge : Prop)
  (draw_line : (Point → Point → Line)) (draw_circle : (Point → Point → Circle))
  (mark_points : (Set Point → Prop)) (erase_points : (Set Point → Prop)) : Prop :=
  ∀ (Fedya_turns_first : Prop) (initially_no_points_marked : Prop), 
    (can_construct_all_altitudes : Prop)

theorem Fedya_can_construct_altitudes
  (triangle : Type) (is_acute_angled : Prop) (is_non_equilateral : Prop)
  (has_compass : Prop) (has_straightedge : Prop)
  (draw_line : (Point → Point → Line)) (draw_circle : (Point → Point → Circle))
  (mark_points : (Set Point → Prop)) (erase_points : (Set Point → Prop))
  (Fedya_turns_first : Prop) (initially_no_points_marked : Prop) :
  can_construct_altitudes triangle is_acute_angled is_non_equilateral has_compass has_straightedge draw_line draw_circle mark_points erase_points Fedya_turns_first initially_no_points_marked :=
sorry

end Fedya_can_construct_altitudes_l186_186674


namespace three_digit_integers_with_odd_number_of_factors_count_l186_186267

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l186_186267


namespace find_p_if_parabola_axis_tangent_to_circle_l186_186848

theorem find_p_if_parabola_axis_tangent_to_circle :
  ∀ (p : ℝ), 0 < p →
    (∃ (C : ℝ × ℝ) (r : ℝ), 
      (C = (2, 0)) ∧ (r = 3) ∧ (dist (C.1 + p / 2, C.2) (C.1, C.2) = r) 
    ) → p = 2 :=
by
  intro p hp h
  rcases h with ⟨C, r, hC, hr, h_dist⟩ 
  have h_eq : C = (2, 0) := hC
  have hr_eq : r = 3 := hr
  rw [h_eq, hr_eq] at h_dist
  sorry

end find_p_if_parabola_axis_tangent_to_circle_l186_186848


namespace problem_solution_l186_186713

-- Define the conditions
variables (M m : ℝ) -- masses of the slope and the block
variable (g : ℝ) -- gravitational acceleration
variable (smooth_horizontal_surface : Type) -- type representing a smooth horizontal surface
variable (block_starts_from_rest : Prop) -- proposition representing the block starts from rest
variable (friction_between_block_and_slope : Prop) -- proposition representing friction between block and slope

-- Define the proposition to prove
def friction_force_does_negative_work_on_slope :=
  (friction_between_block_and_slope → negative_work (force_of_slope_on_block))

-- State the theorem
theorem problem_solution
  (h_smooth_surface : smooth_horizontal_surface)
  (h_block_rest : block_starts_from_rest)
  (h_friction : friction_between_block_and_slope)
  : friction_force_does_negative_work_on_slope :=
sorry

end problem_solution_l186_186713


namespace option_A_correct_l186_186765

variables {R : Type*} [linear_ordered_field R]

def directed_distance (a b c x y : R) : R :=
  (a * x + b * y + c) / (Real.sqrt (a * a + b * b))

theorem option_A_correct (a b c x1 y1 x2 y2 : R) (h1 : a^2 + b^2 ≠ 0) 
    (d1 d2 : R) (h2 : directed_distance a b c x1 y1 = d1)
    (h3 : directed_distance a b c x2 y2 = d2)
    (h4 : d1 = 1) (h5 : d2 = 1) : 
    (a * x1 + b * y1 + c = a * x2 + b * y2 + c) :=
sorry

end option_A_correct_l186_186765


namespace walt_total_interest_l186_186967

noncomputable def total_investment : ℝ := 12000
noncomputable def investment_at_7_percent : ℝ := 5500
noncomputable def investment_at_9_percent : ℝ := total_investment - investment_at_7_percent
noncomputable def rate_7_percent : ℝ := 0.07
noncomputable def rate_9_percent : ℝ := 0.09

theorem walt_total_interest :
  let interest_7 : ℝ := investment_at_7_percent * rate_7_percent
  let interest_9 : ℝ := investment_at_9_percent * rate_9_percent
  interest_7 + interest_9 = 970 := by
  sorry

end walt_total_interest_l186_186967


namespace minimum_value_of_quadratic_l186_186751

theorem minimum_value_of_quadratic (p q : ℝ) (hp : 0 < p) (hq : 0 < q) : 
  ∃ x : ℝ, x = - (p + q) / 2 ∧ ∀ y : ℝ, (y^2 + p*y + q*y) ≥ ((- (p + q) / 2)^2 + p*(- (p + q) / 2) + q*(- (p + q) / 2)) := by
  sorry

end minimum_value_of_quadratic_l186_186751


namespace three_digit_integers_with_odd_factors_l186_186382

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l186_186382


namespace speed_of_man_rowing_upstream_l186_186700

theorem speed_of_man_rowing_upstream (Vm Vdownstream Vupstream : ℝ) (hVm : Vm = 40) (hVdownstream : Vdownstream = 45) : Vupstream = 35 :=
by
  sorry

end speed_of_man_rowing_upstream_l186_186700


namespace sum_eighth_row_interior_numbers_l186_186913

-- Define the sum of the interior numbers in the nth row of Pascal's Triangle.
def sum_interior_numbers (n : ℕ) : ℕ := 2^(n-1) - 2

-- Problem statement: Prove the sum of the interior numbers of Pascal's Triangle in the eighth row is 126,
-- given the sums for the fifth and sixth rows.
theorem sum_eighth_row_interior_numbers :
  sum_interior_numbers 5 = 14 →
  sum_interior_numbers 6 = 30 →
  sum_interior_numbers 8 = 126 :=
by
  sorry

end sum_eighth_row_interior_numbers_l186_186913


namespace num_three_digit_ints_with_odd_factors_l186_186261

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l186_186261


namespace three_digit_oddfactors_count_is_22_l186_186091

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l186_186091


namespace math_problem_l186_186743

theorem math_problem : 2 + 5 * 4 - 6 + 3 = 19 := by
  sorry

end math_problem_l186_186743


namespace cos_angle_through_point_l186_186483

theorem cos_angle_through_point (a : ℝ) (h : a < 0) :
  let P : ℝ × ℝ := (a, 2 * a)
  let x := P.1
  let y := P.2
  let r := -real.sqrt 5 * a
  cos (real.arctan2 y x) = -real.sqrt 5 / 5 :=
by {
  -- Definitions based on conditions
  sorry
}

end cos_angle_through_point_l186_186483


namespace number_of_three_digit_integers_with_odd_factors_l186_186407

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l186_186407


namespace cube_and_sphere_identical_projections_l186_186505

-- Definitions of shapes and their orthographic projections
inductive Shape
| cone
| cube
| cylinder
| sphere
| regular_tetrahedron

noncomputable def identical_projections (s : Shape) : Prop :=
match s with
| Shape.cone => false
| Shape.cube => true
| Shape.cylinder => false
| Shape.sphere => true
| Shape.regular_tetrahedron => false
end

-- Theorem statement: Cube and Sphere have identical orthographic projections
theorem cube_and_sphere_identical_projections :
  identical_projections Shape.cube ∧ identical_projections Shape.sphere :=
by
  sorry

end cube_and_sphere_identical_projections_l186_186505


namespace three_digit_integers_with_odd_factors_l186_186223

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l186_186223


namespace total_cost_of_breakfast_l186_186805

-- Definitions based on conditions
def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3
def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2
def kiera_muffins : ℕ := 2
def kiera_fruit_cup : ℕ := 1

-- The proof statement
theorem total_cost_of_breakfast : 
  muffin_cost * francis_muffins + 
  fruit_cup_cost * francis_fruit_cups + 
  muffin_cost * kiera_muffins + 
  fruit_cup_cost * kiera_fruit_cup = 17 := 
  by sorry

end total_cost_of_breakfast_l186_186805


namespace find_k_l186_186777

variable (α : ℝ)

theorem find_k (h : ((Real.tan α + Real.cot α)^2 + (Real.sin α + Real.cos α)^2 
                     = k + Real.sec α ^ 2 + Real.csc α ^ 2)) : 
  k = 7 :=
sorry

end find_k_l186_186777


namespace three_digit_perfect_squares_count_l186_186190

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l186_186190


namespace simplify_sqrt_expression_l186_186982

theorem simplify_sqrt_expression (α : ℝ) : sqrt (1 - sin α ^ 2) = abs (cos α) := by sorry

end simplify_sqrt_expression_l186_186982


namespace three_digit_integers_with_odd_number_of_factors_l186_186163

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l186_186163


namespace root_of_sqrt_2x_plus_3_eq_x_l186_186960

theorem root_of_sqrt_2x_plus_3_eq_x :
  ∀ x : ℝ, (sqrt (2 * x + 3) = x) → x = 3 := by
  intro x
  intro h
  sorry

end root_of_sqrt_2x_plus_3_eq_x_l186_186960


namespace residue_S_mod_4020_l186_186554

def S : ℤ := (List.sum (List.range 4021).map (λ n, if n % 2 = 0 then -n else n))

theorem residue_S_mod_4020 : S % 4020 = 2010 :=
by sorry

end residue_S_mod_4020_l186_186554


namespace zeros_in_decimal_representation_l186_186862

theorem zeros_in_decimal_representation : 
  let n := 2^7 * 5^3 in
  ∀ (a b : ℕ), (1 / n * (5^4) = a / (10^b)) → b = 7 → a = 625 → 4 = 7 - 3 :=
by
  intros n a b h₁ h₂ h₃
  cases b
  { exfalso,
    sorry }
  cases b
  { exfalso,
    sorry }
  cases b
  { exfalso,
    sorry }
  cases b
  { exfalso,
    sorry }
  cases b
  { exfalso,
    sorry }
  cases b
  { exfalso,
    sorry }
  cases b
  { exfalso,
    sorry }
  cases b
  { exfalso,
    sorry }
  cases b
  { exfalso,
    sorry }
  cases b
  { exfalso,
    sorry }
  cases b
  {  refl },
  sorry


end zeros_in_decimal_representation_l186_186862


namespace three_digit_integers_with_odd_factors_count_l186_186296

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l186_186296


namespace find_e_l186_186937

theorem find_e (b e : ℝ) (f g : ℝ → ℝ)
    (h1 : ∀ x, f x = 5 * x + b)
    (h2 : ∀ x, g x = b * x + 3)
    (h3 : ∀ x, f (g x) = 15 * x + e) : e = 18 :=
by
  sorry

end find_e_l186_186937


namespace three_digit_integers_with_odd_factors_count_l186_186299

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l186_186299


namespace matrix_multiplication_l186_186544

variables (N : Matrix (Fin 2) (Fin 2) ℝ)
variables (v1 v2 v3 v4 : ℝ)

theorem matrix_multiplication :
  (N ⬝ ⟨![7, 2]⟩) = ⟨![21, -4]⟩ :=
by
  have h1 : (N ⬝ ⟨![3, -2]⟩) = ⟨![5, 0]⟩ := sorry,
  have h2 : (N ⬝ ⟨![ -4, 6] ⟩) = ⟨![ -2, -2]⟩ := sorry,
  sorry

end matrix_multiplication_l186_186544


namespace sum_of_coordinates_eq_69_l186_186882

theorem sum_of_coordinates_eq_69 {f k : ℝ → ℝ} (h₁ : f 4 = 8) (h₂ : ∀ x, k x = (f x)^2 + 1) : 4 + k 4 = 69 :=
by
  sorry

end sum_of_coordinates_eq_69_l186_186882


namespace average_songs_in_remaining_sets_l186_186628

-- Conditions
def total_songs : ℕ := 50
def songs_first_set : ℕ := 8
def songs_second_set : ℕ := 12
def songs_encores : ℕ := 4
def remaining_sets : ℕ := 3

-- Theorem statement
theorem average_songs_in_remaining_sets :
  let total_played := songs_first_set + songs_second_set + songs_encores in
  let remaining_songs := total_songs - total_played in
  let average_songs_per_set := (remaining_songs : ℝ) / remaining_sets in
  average_songs_per_set = 8.67 :=
by
  sorry

end average_songs_in_remaining_sets_l186_186628


namespace smallest_value_geometric_seq_l186_186545

theorem smallest_value_geometric_seq (s : ℝ) :
  let b₁ := 2
  let b₂ := b₁ * s
  let b₃ := b₂ * s in
  3 * b₂ + 4 * b₃ ≥ -9 / 8 :=
by
  sorry

end smallest_value_geometric_seq_l186_186545


namespace max_diff_leq_6_l186_186055

def f (x : ℝ) : ℝ := if x ≥ 0 then 1 else -2

theorem max_diff_leq_6 (x1 x2 : ℝ) 
  (h1 : x1 + (x1 - 1) * f (x1 + 1) ≤ 5) 
  (h2 : x2 + (x2 - 1) * f (x2 + 1) ≤ 5) :
  x1 - x2 ≤ 6 :=
sorry

end max_diff_leq_6_l186_186055


namespace three_digit_perfect_squares_count_l186_186184

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l186_186184


namespace unique_orthocenter_exists_l186_186857

noncomputable theory

open EuclideanGeometry

variable {A B C X : Point}

theorem unique_orthocenter_exists (hABC : ¬Collinear A B C) :
    ∃! X, dist X A ^ 2 + dist X B ^ 2 + dist A B ^ 2 =
         dist X B ^ 2 + dist X C ^ 2 + dist B C ^ 2 ∧
         dist X C ^ 2 + dist X A ^ 2 + dist C A ^ 2 :=
begin
  sorry
end

end unique_orthocenter_exists_l186_186857


namespace part1_part2_l186_186839

noncomputable def f (ω x : ℝ) : ℝ := 4 * ((Real.sin (ω * x - Real.pi / 4)) * (Real.cos (ω * x)))

noncomputable def g (α : ℝ) : ℝ := 2 * (Real.sin (α - Real.pi / 6)) - Real.sqrt 2

theorem part1 (ω : ℝ) (x : ℝ) (hω : 0 < ω ∧ ω < 2) (hx : f ω (Real.pi / 4) = Real.sqrt 2) : 
  ∃ T > 0, ∀ x, f ω (x + T) = f ω x :=
sorry

theorem part2 (α : ℝ) (hα: 0 < α ∧ α < Real.pi / 2) (h : g α = 4 / 3 - Real.sqrt 2) : 
  Real.cos α = (Real.sqrt 15 - 2) / 6 :=
sorry

end part1_part2_l186_186839


namespace percentage_of_part_whole_l186_186680

theorem percentage_of_part_whole (part whole : ℝ) (h_part : part = 75) (h_whole : whole = 125) : 
  (part / whole) * 100 = 60 :=
by
  rw [h_part, h_whole]
  -- Simplification steps would follow, but we substitute in the placeholders
  sorry

end percentage_of_part_whole_l186_186680


namespace radovan_start_reading_page_l186_186573

theorem radovan_start_reading_page :
  ∃ n : ℕ, (finset.range 15).sum (λ i, n + i) = (finset.range 12).sum (λ i, (n + 15) + i) ∧ (n + 27 = 74) :=
begin
  have y_pag_sum : ∀ n : ℕ, (finset.range 15).sum (λ i, n + i) = 15 * n + 105,
  { intro n, simp [finset.sum_range_succ, nat.succ_eq_add_one, add_right_comm, nat.add_comm] },
  have t_pag_sum : ∀ n : ℕ, (finset.range 12).sum (λ i, (n + 15) + i) = 12 * n + 246,
  { intro n, simp [finset.sum_range_succ, nat.succ_eq_add_one, add_assoc, nat.add_comm, add_right_comm, nat.add_comm] },
  use 47,
  split,
  { rw [y_pag_sum, t_pag_sum], exact nat.add_sub_cancel' 15 43 },
  refl
end

end radovan_start_reading_page_l186_186573


namespace isosceles_triangle_circumradii_equal_l186_186568

theorem isosceles_triangle_circumradii_equal {A B C D : Type} [EuclideanGeometry A B C D]
  (h : is_isosceles_triangle A B C) (H1 : B ∈ segment A C) :
  circumradius (triangle A B D) = circumradius (triangle B C D) :=
begin
  sorry
end

end isosceles_triangle_circumradii_equal_l186_186568


namespace train_length_correct_l186_186721

/-
Given:
1. Speed of train = 40 km/hr
2. Speed of man = 5 km/hr (opposite direction)
3. Time taken to pass the man = 8.799296056315494 seconds

Prove:
The length of the train is 110.041200704443675 meters
-/

def speed_of_train : ℝ := 40
def speed_of_man : ℝ := 5
def time_to_pass : ℝ := 8.799296056315494

def relative_speed_kmph : ℝ := speed_of_train + speed_of_man
def relative_speed_mps : ℝ := relative_speed_kmph * (5 / 18)
def length_of_train : ℝ := relative_speed_mps * time_to_pass

theorem train_length_correct :
  length_of_train = 110.041200704443675 := by
  sorry

end train_length_correct_l186_186721


namespace number_of_three_digit_integers_with_odd_number_of_factors_l186_186338

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l186_186338


namespace student_marks_obtained_l186_186717

-- Definition of total_marks and percentage needed to pass
def total_marks : ℝ := 300
def pass_percentage : ℝ := 0.33
def fail_difference : ℝ := 40

-- Calculate the passing marks
def passing_marks := pass_percentage * total_marks

-- Proof statement
theorem student_marks_obtained : 
  ∃ (M : ℝ), M = passing_marks - fail_difference ∧ M = 59 :=
by
  let M := passing_marks - fail_difference
  use M
  split
  {
    exact rfl
  }
  {
    calc
    M = 99 - 40 : by norm_num [passing_marks, fail_difference]
    ...= 59      : by norm_num
  }

end student_marks_obtained_l186_186717


namespace subtraction_of_absolute_value_l186_186742

theorem subtraction_of_absolute_value : 2 - (abs (-3)) = -1 := by
  sorry

end subtraction_of_absolute_value_l186_186742


namespace jason_cutting_grass_time_l186_186916

-- Conditions
def time_to_cut_one_lawn : ℕ := 30 -- in minutes
def lawns_cut_each_day : ℕ := 8
def days : ℕ := 2
def minutes_in_an_hour : ℕ := 60

-- Proof that the number of hours Jason spends cutting grass over the weekend is 8
theorem jason_cutting_grass_time:
  ((lawns_cut_each_day * days) * time_to_cut_one_lawn) / minutes_in_an_hour = 8 :=
by
  sorry

end jason_cutting_grass_time_l186_186916


namespace math_problem_l186_186875

/-- Given that \( y^2 = 4y - \sqrt{x-3} - 4 \), we are to prove that \( x + 2y = 7 \). --/
theorem math_problem (x y : ℝ) 
  (h : y^2 = 4y - real.sqrt (x-3) - 4) : 
  x + 2 * y = 7 := 
sorry

end math_problem_l186_186875


namespace three_digit_integers_with_odd_factors_l186_186222

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l186_186222


namespace rectangle_perimeter_l186_186615

theorem rectangle_perimeter (b : ℕ) (h1 : 3 * b * b = 192) : 2 * ((3 * b) + b) = 64 := 
by
  sorry

end rectangle_perimeter_l186_186615


namespace three_digit_integers_with_odd_number_of_factors_count_l186_186277

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l186_186277


namespace three_digit_integers_odd_factors_count_l186_186443

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l186_186443


namespace num_three_digit_ints_with_odd_factors_l186_186256

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l186_186256


namespace sum_series_a_eq_one_sum_series_b_eq_half_sum_series_c_eq_third_l186_186571

noncomputable def sum_series_a : ℝ :=
∑' n, (1 / (n * (n + 1)))

noncomputable def sum_series_b : ℝ :=
∑' n, (1 / ((n + 1) * (n + 2)))

noncomputable def sum_series_c : ℝ :=
∑' n, (1 / ((n + 2) * (n + 3)))

theorem sum_series_a_eq_one : sum_series_a = 1 := sorry

theorem sum_series_b_eq_half : sum_series_b = 1 / 2 := sorry

theorem sum_series_c_eq_third : sum_series_c = 1 / 3 := sorry

end sum_series_a_eq_one_sum_series_b_eq_half_sum_series_c_eq_third_l186_186571


namespace circles_concurrent_iff_collinear_l186_186978

-- Definitions of the vertices A, B, C and points D, E, F on the respective sides of the triangle.
variables {A B C D E F : Type}

-- Definition of collinearity
def collinear (D E F : Point) : Prop :=
  ∃ l : Line, ∀ p ∈ {D, E, F}, p ∈ l

-- Definition of concurrency of circles
def concurrent (ω₁ ω₂ ω₃ : Circle) : Prop :=
  ∃ M : Point, M ∈ ω₁ ∧ M ∈ ω₂ ∧ M ∈ ω₃

-- Problem statement in Lean
theorem circles_concurrent_iff_collinear
  (triangle_ABC : Triangle)
  (D_on_BC : D ∈ segment B C)
  (E_on_CA : E ∈ segment C A)
  (F_on_AB : F ∈ segment A B)
  (ω₁ := circumcircle A E F)
  (ω₂ := circumcircle B F D)
  (ω₃ := circumcircle C D E) :
  (concurrent ω₁ ω₂ ω₃ ↔ collinear D E F) :=
sorry

end circles_concurrent_iff_collinear_l186_186978


namespace total_number_of_doves_l186_186812

-- Definition of the problem conditions
def initial_doves : ℕ := 20
def eggs_per_dove : ℕ := 3
def hatch_rate : ℚ := 3/4

-- The Lean 4 statement that expresses the proof problem
theorem total_number_of_doves (initial_doves eggs_per_dove : ℕ) (hatch_rate : ℚ) : 
  initial_doves + (hatch_rate * (initial_doves * eggs_per_dove)) = 65 := 
by {
  -- Defining intermediate quantities
  let total_eggs := initial_doves * eggs_per_dove,
  let hatched_eggs := hatch_rate * total_eggs,

  -- Proving the final number of doves
  have : initial_doves + hatched_eggs = 65,
    calc
      initial_doves + hatched_eggs
          = 20 + (3/4 * 60) :   by sorry
      ... = 20 + 45           : by sorry
      ... = 65                : by sorry,
  exact this
}

end total_number_of_doves_l186_186812


namespace three_digit_integers_with_odd_number_of_factors_count_l186_186281

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l186_186281


namespace max_area_of_section_of_cone_is_2_l186_186478

-- Definitions for conditions
def radius_of_sector : ℝ := 2
def central_angle : ℝ := (5 * Real.pi) / 3

-- Mathematical proof statement
theorem max_area_of_section_of_cone_is_2 :
  ∀ (r : ℝ) (θ : ℝ), r = 2 → θ = (5 * Real.pi) / 3 → 
  let l := 2 in
  let a_max := (2 * sqrt 2) in
  let s := (a_max / 2) * sqrt (4 - (a_max ^ 2 / 4)) in
  s = 2 := 
by
  intros r θ hr hθ l a_max s
  sorry

end max_area_of_section_of_cone_is_2_l186_186478


namespace decreasing_interval_of_f_l186_186873

theorem decreasing_interval_of_f (m : ℝ) (f : ℝ → ℝ) (hdef : f = λ x, (m-2)x^2 + mx + 4) (h_even : ∀ x : ℝ, f (-x) = f x) (hm : m = 0) :
  ∃ I : set ℝ, I = set.Ici 0 ∧ ∀ x ∈ I, monotone_decreasing_on (λ x, (m-2)x^2 + mx + 4) x :=
by
  sorry

end decreasing_interval_of_f_l186_186873


namespace three_digit_integers_with_odd_factors_l186_186202

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l186_186202


namespace prove_equation_l186_186593

variables (α β : ℝ)
hypothesis (h1 : 0 < α ∧ α < π / 2)
hypothesis (h2 : 0 < β ∧ β < π / 2)
hypothesis (h3 : Real.tan α = (1 + Real.sin β) / Real.cos β)

theorem prove_equation : 2 * α - β = π / 2 :=
by 
  sorry

end prove_equation_l186_186593


namespace five_lattice_points_l186_186022

theorem five_lattice_points (p : Fin 5 → ℤ × ℤ) :
  ∃ i j : Fin 5, i ≠ j ∧ ((p i).1 + (p j).1) % 2 = 0 ∧ ((p i).2 + (p j).2) % 2 = 0 := by
  sorry

end five_lattice_points_l186_186022


namespace three_digit_odds_factors_count_l186_186359

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l186_186359


namespace santino_total_fruits_l186_186579

theorem santino_total_fruits :
  ∃ (papaya_trees mango_trees papayas_per_tree mangos_per_tree total_fruits : ℕ),
    papaya_trees = 2 ∧
    papayas_per_tree = 10 ∧
    mango_trees = 3 ∧
    mangos_per_tree = 20 ∧
    total_fruits = (papaya_trees * papayas_per_tree) + (mango_trees * mangos_per_tree) ∧
    total_fruits = 80 :=
by
  -- Definitions
  let papaya_trees := 2
  let papayas_per_tree := 10
  let mango_trees := 3
  let mangos_per_tree := 20
  let total_fruits := (papaya_trees * papayas_per_tree) + (mango_trees * mangos_per_tree)

  -- Goal
  have : papaya_trees = 2 := rfl
  have : papayas_per_tree = 10 := rfl
  have : mango_trees = 3 := rfl
  have : mangos_per_tree = 20 := rfl
  have : total_fruits = 80 :=
    by
      calc
        total_fruits
          = (papaya_trees * papayas_per_tree) + (mango_trees * mangos_per_tree) : rfl
          ... = 20 + 60 : by simp [papaya_trees, papayas_per_tree, mango_trees, mangos_per_tree]
          ... = 80 : rfl
  exact ⟨papaya_trees, mango_trees, papayas_per_tree, mangos_per_tree, total_fruits, rfl, rfl, rfl, rfl, this⟩


end santino_total_fruits_l186_186579


namespace three_digit_oddfactors_count_l186_186122

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l186_186122


namespace problem_1_problem_2_l186_186935

variable (a : ℝ)
def p : Prop := ∃ (x : ℝ), x^2 - 2 * x + a^2 + 3 * a - 3 < 0
def r : Prop := ∀ (x : ℝ), 1 - a ≤ x ∧ x ≤ 1 + a

theorem problem_1 (h : ¬¬p) : -4 < a ∧ a < 1 :=
sorry

theorem problem_2 (h : p → r) : 5 ≤ a :=
sorry

end problem_1_problem_2_l186_186935


namespace last_digit_of_product_l186_186650

-- Define the concept of non-trivial primes
def is_non_trivial_prime (n : ℕ) : Prop :=
  nat.prime n ∧ n > 10 ∧ (n % 10 ≠ 5) ∧ (n % 2 = 1)

-- Define the set of possible last digits of non-trivial primes
def last_digit_of (n : ℕ) : ℕ := n % 10

-- The main theorem to show that the product of two distinct non-trivial primes 
-- results in a last digit that is one of {1, 3, 7, 9}.
theorem last_digit_of_product (p q : ℕ) (hp : is_non_trivial_prime p) (hq : is_non_trivial_prime q) (hneq : p ≠ q) :
  last_digit_of (p * q) ∈ {1, 3, 7, 9} :=
sorry

end last_digit_of_product_l186_186650


namespace three_digit_odds_factors_count_l186_186357

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l186_186357


namespace infinite_real_numbers_with_finite_sequence_values_l186_186548

noncomputable def g (x : ℝ) : ℝ := 3 * x - x^2

theorem infinite_real_numbers_with_finite_sequence_values :
  ∃ (S : set ℝ), set.infinite S ∧ ∀ x0 ∈ S, 
  ∃ B : set ℝ, B.finite ∧ ∀ n : ℕ, g^[n] x0 ∈ B :=
sorry

end infinite_real_numbers_with_finite_sequence_values_l186_186548


namespace three_digit_odds_factors_count_l186_186355

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l186_186355


namespace three_digit_oddfactors_count_l186_186140

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l186_186140


namespace correct_time_l186_186887

-- Define the observed times on the clocks
def time1 := 14 * 60 + 54  -- 14:54 in minutes
def time2 := 14 * 60 + 57  -- 14:57 in minutes
def time3 := 15 * 60 + 2   -- 15:02 in minutes
def time4 := 15 * 60 + 3   -- 15:03 in minutes

-- Define the inaccuracies of the clocks
def inaccuracy1 := 2  -- First clock off by 2 minutes
def inaccuracy2 := 3  -- Second clock off by 3 minutes
def inaccuracy3 := -4  -- Third clock off by 4 minutes
def inaccuracy4 := -5  -- Fourth clock off by 5 minutes

-- State that given these conditions, the correct time is 14:58
theorem correct_time : ∃ (T : Int), 
  (time1 + inaccuracy1 = T) ∧
  (time2 + inaccuracy2 = T) ∧
  (time3 + inaccuracy3 = T) ∧
  (time4 + inaccuracy4 = T) ∧
  (T = 14 * 60 + 58) :=
by
  sorry

end correct_time_l186_186887


namespace num_three_digit_integers_with_odd_factors_l186_186376

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l186_186376


namespace number_of_three_digit_squares_l186_186430

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l186_186430


namespace pointA_when_B_origin_pointB_when_A_origin_l186_186017

def vectorAB : ℝ × ℝ := (-2, 4)

-- Prove that when point B is the origin, the coordinates of point A are (2, -4)
theorem pointA_when_B_origin : vectorAB = (-2, 4) → (0, 0) - (-2, 4) = (2, -4) :=
by
  sorry

-- Prove that when point A is the origin, the coordinates of point B are (-2, 4)
theorem pointB_when_A_origin : vectorAB = (-2, 4) → (0, 0) + (-2, 4) = (-2, 4) :=
by
  sorry

end pointA_when_B_origin_pointB_when_A_origin_l186_186017


namespace three_digit_integers_odd_factors_count_l186_186459

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l186_186459


namespace arithmetic_geometric_sequence_fraction_l186_186835

theorem arithmetic_geometric_sequence_fraction 
  (a1 a2 b1 b2 b3 : ℝ)
  (h1 : a1 + a2 = 10)
  (h2 : 1 * b3 = 9)
  (h3 : b2 ^ 2 = 9) : 
  b2 / (a1 + a2) = 3 / 10 := 
by 
  sorry

end arithmetic_geometric_sequence_fraction_l186_186835


namespace am_gm_example_l186_186945

variable {x y z : ℝ}

theorem am_gm_example (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  x / y + y / z + z / x + y / x + z / y + x / z ≥ 6 :=
sorry

end am_gm_example_l186_186945


namespace regular_triangular_pyramid_volume_l186_186011

theorem regular_triangular_pyramid_volume (a γ : ℝ) : 
  ∃ V, V = (a^3 * Real.sin (γ / 2)^2) / (12 * Real.sqrt (1 - (Real.sin (γ / 2))^2)) := 
sorry

end regular_triangular_pyramid_volume_l186_186011


namespace not_translatable_l186_186726

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := - (1/2) * x^2 + x + 1

-- Define the proposed untranslatable parabola
def incorrect_translation (x : ℝ) : ℝ := - x^2 + x + 1

-- Define a proposition for the translated form invariance
def is_translation (f g : ℝ → ℝ) : Prop :=
  ∃ (h : ℝ → ℝ), ∀ x, g x = f (h x)

-- Formal statement of the problem
theorem not_translatable :
  ¬ is_translation original_parabola incorrect_translation :=
sorry

end not_translatable_l186_186726


namespace num_three_digit_ints_with_odd_factors_l186_186254

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l186_186254


namespace trajectory_classification_l186_186035

-- Define the points F1 and F2
def F1 : Point := {-5, 0}
def F2 : Point := {5, 0}

-- Define the moving point P
def P (x y : ℝ) : Point := {x, y}

-- Define the function that checks the condition for the sum of distances
def sum_of_distances (P : Point) (F1 F2 : Point) : ℝ :=
  dist P F1 + dist P F2

-- Define the 'ellipse, line segment or does not exist' problem statement
theorem trajectory_classification (a : ℝ) (ha : 0 < a):
  ∀ (P : Point),
  (sum_of_distances P F1 F2 = 2 * a) →
  (if a = 10 then ∃ (P : Point), true ∧ segment P F1 F2 ∧ (∑ P))
  else if a > 10 then ∃ (P : Point), true ∧ ellipse P F1 F2 ∧ (∑ P)
  else false := sorry

end trajectory_classification_l186_186035


namespace breakfast_cost_l186_186810

theorem breakfast_cost :
  ∀ (muffin_cost fruit_cup_cost : ℕ) (francis_muffins francis_fruit_cups kiera_muffins kiera_fruit_cups : ℕ),
  muffin_cost = 2 ∧ fruit_cup_cost = 3 ∧ francis_muffins = 2 ∧ francis_fruit_cups = 2 ∧ kiera_muffins = 2 ∧ kiera_fruit_cups = 1
  → (francis_muffins * muffin_cost + francis_fruit_cups * fruit_cup_cost + kiera_muffins * muffin_cost + kiera_fruit_cups * fruit_cup_cost = 17) :=
by
  intros muffin_cost fruit_cup_cost francis_muffins francis_fruit_cups kiera_muffins kiera_fruit_cups
  intro cond
  cases cond with muffin_cost_eq rest
  cases rest with fruit_cup_cost_eq rest
  cases rest with francis_muffins_eq rest
  cases rest with francis_fruit_cups_eq rest
  cases rest with kiera_muffins_eq kiera_fruit_cups_eq

  rw [muffin_cost_eq, fruit_cup_cost_eq, francis_muffins_eq, francis_fruit_cups_eq, kiera_muffins_eq, kiera_fruit_cups_eq]
  norm_num
  sorry

end breakfast_cost_l186_186810


namespace sqrt_product_simplification_l186_186738

variable (p : ℝ)

theorem sqrt_product_simplification : 
  sqrt (45 * p) * sqrt (15 * p) * (sqrt (10 * p^3))^(1/3) = 150 * (5)^(1/3) * p := 
by 
  sorry

end sqrt_product_simplification_l186_186738


namespace trigonometric_identity_l186_186676

variable (α : ℝ)

theorem trigonometric_identity :
  (1 + Real.cos (4 * α - 2 * Real.pi) + Real.cos (4 * α - Real.pi / 2)) /
  (1 + Real.cos (4 * α + Real.pi) + Real.cos (4 * α + 3 * Real.pi / 2)) = 
  Real.cot (2 * α) := 
sorry

end trigonometric_identity_l186_186676


namespace num_odd_factors_of_three_digit_integers_eq_22_l186_186108

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l186_186108


namespace T_0_2006_correct_T_1_2006_correct_T_2_2006_correct_l186_186514

def T (r n : ℕ) : ℕ :=
  sorry -- Define the function T_r(n) according to the problem's condition

-- Specific cases given in the problem statement
noncomputable def T_0_2006 : ℕ := T 0 2006
noncomputable def T_1_2006 : ℕ := T 1 2006
noncomputable def T_2_2006 : ℕ := T 2 2006

-- Theorems stating the result
theorem T_0_2006_correct : T_0_2006 = 1764 := sorry
theorem T_1_2006_correct : T_1_2006 = 122 := sorry
theorem T_2_2006_correct : T_2_2006 = 121 := sorry

end T_0_2006_correct_T_1_2006_correct_T_2_2006_correct_l186_186514


namespace find_value_of_m_and_n_l186_186061

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^3 + 3*x^2 + m * x
noncomputable def g (x : ℝ) (n : ℝ) : ℝ := Real.log (x + 1) + n * x

theorem find_value_of_m_and_n (m n : ℝ) (h₀ : n > 0) 
  (h₁ : f (-1) m = -1) 
  (h₂ : ∀ x : ℝ, f x m = g x n → x = 0) :
  m + n = 5 := 
by 
  sorry

end find_value_of_m_and_n_l186_186061


namespace original_game_no_chance_chief_cannot_reclaim_chief_knows_expulsions_max_tribesmen_expelled_natives_could_lose_second_game_l186_186704

theorem original_game_no_chance:
  ∀ (n : ℕ), n = 30 → (∀ a b : ℕ, a ≠ b → a = b → False) → False := sorry

theorem chief_cannot_reclaim:
  ∀ (distributed_coins total_people : ℕ), distributed_coins = 270 → total_people < 30 → 
  (∃ x : ℕ, (x = total_people) ∧ (finset.sum (finset.range x) (λ i, i+1)) ≠ distributed_coins) := sorry

theorem chief_knows_expulsions:
  ∀ (expelled remaining total : ℕ), total = 30 → (remaining = total - expelled) → ∃ n, n = expelled := sorry

theorem max_tribesmen_expelled:
  ∀ (distributed_coins : ℕ), distributed_coins = 270 → 
  ∃ max_expul : ℕ, max_expul ≤ 6 ∧ 
  (∃ unique_coins : finset ℕ, unique_coins.card = (30 - max_expul) ∧ 
  finset.sum unique_coins (λ i, i+1) = distributed_coins) := sorry

theorem natives_could_lose_second_game:
  ∀ (merchant_lost_first_game natives_won_first_game : Prop), 
  merchant_lost_first_game → natives_won_first_game → natives_won_first_game →
  (merchant_lost_first_game → ¬ (natives_won_first_game → True)) := sorry

end original_game_no_chance_chief_cannot_reclaim_chief_knows_expulsions_max_tribesmen_expelled_natives_could_lose_second_game_l186_186704


namespace sum_of_zeros_of_odd_function_l186_186043

noncomputable def f : ℝ → ℝ := sorry

theorem sum_of_zeros_of_odd_function 
  (h_odd : ∀ x, f (-x) = - f x)
  (h_zeros : (Set.range (f')).finite ∧ (Set.range (f')) = {x ∈ Set.univ | f x = 0})
  (h_len : (Set.range (f')).to_finset.card = 2017) :
  ∑ x in (Set.range (f')).to_finset, x = 0 :=
sorry

end sum_of_zeros_of_odd_function_l186_186043


namespace num_three_digit_integers_with_odd_factors_l186_186375

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l186_186375


namespace minimum_positive_period_and_zero_points_max_min_values_in_interval_l186_186057

noncomputable def f (x : ℝ) := 4 * sin x * cos (x - π / 3) - sqrt 3

theorem minimum_positive_period_and_zero_points :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ 
  (∀ k : ℤ, f (π / 6 + k * π / 2) = 0) :=
sorry

theorem max_min_values_in_interval :
  (∀ x, π / 24 ≤ x ∧ x ≤ 3 * π / 4 → -sqrt 2 ≤ f x ∧ f x ≤ 2) ∧ 
  (f (π / 24) = -sqrt 2 ∧ f (5 * π / 12) = 2) :=
sorry

end minimum_positive_period_and_zero_points_max_min_values_in_interval_l186_186057


namespace arithmetic_sequence_properties_l186_186896

/-- In an arithmetic sequence {a_n}, let S_n represent the sum of the first n terms, 
and it is given that S_6 < S_7 and S_7 > S_8. 
Prove that the correct statements among the given options are: 
1. The common difference d < 0 
2. S_9 < S_6 
3. S_7 is definitively the maximum value among all sums S_n. -/
theorem arithmetic_sequence_properties 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h_arith_seq : ∀ n, S (n + 1) = S n + a (n + 1))
  (h_S6_lt_S7 : S 6 < S 7)
  (h_S7_gt_S8 : S 7 > S 8) :
  (a 7 > 0 ∧ a 8 < 0 ∧ ∃ d, ∀ n, a (n + 1) = a n + d ∧ d < 0 ∧ S 9 < S 6 ∧ ∀ n, S n ≤ S 7) :=
by
  -- Proof omitted
  sorry

end arithmetic_sequence_properties_l186_186896


namespace three_digit_integers_with_odd_factors_l186_186391

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l186_186391


namespace plot_area_in_acres_l186_186498

-- Definitions for the problem conditions
def bottom_base : ℝ := 20
def top_base : ℝ := 25
def height : ℝ := 15
def cm_to_mile : ℝ := 1 -- given scale 1 cm = 1 mile
def sq_mile_to_acre : ℝ := 640 -- given 1 square mile = 640 acres

-- Trapezoid area formula in cm²
def trapezoid_area_cm := (bottom_base + top_base) * height * (1 / 2)

-- Convert area from cm² to square miles
def trapezoid_area_miles := trapezoid_area_cm * (cm_to_mile ^ 2)

-- Convert area from square miles to acres
def trapezoid_area_acres := trapezoid_area_miles * sq_mile_to_acre

-- Theorem statement asserting the area of the plot in acres
theorem plot_area_in_acres : trapezoid_area_acres = 216000 := by
  -- Proof details would go here
  sorry

end plot_area_in_acres_l186_186498


namespace number_of_three_digit_squares_l186_186439

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l186_186439


namespace max_value_of_sequence_l186_186820

theorem max_value_of_sequence : 
  ∃ n : ℕ, n > 0 ∧ ∀ m : ℕ, m > 0 → (∃ (a : ℝ), a = (m / (m^2 + 6 : ℝ)) ∧ a ≤ (n / (n^2 + 6 : ℝ))) :=
sorry

end max_value_of_sequence_l186_186820


namespace find_tan_A_l186_186520

-- Define the sides of triangle ABC with given conditions
variables {A B C : Type} [LinearOrderedField A]
def triangle_ABC (AB AC BC : A) : Prop :=
  AB^2 = AC^2 + BC^2  -- Pythagorean theorem for right triangle

-- Define the specific sides AB and AC given in the problem
def AB : A := Real.sqrt 17
def AC : A := 4

-- Define the length BC using Pythagorean theorem
noncomputable def BC : A := Real.sqrt (AB^2 - AC^2)

-- State the problem of finding tangent of angle A
noncomputable def tan_A : A := BC / AC

-- The main theorem that needs to be proved
theorem find_tan_A : tan_A = 1 / 4 :=
by
  have h : triangle_ABC AB AC BC, from
    by sorry,  -- Assume triangle_ABC as valid right triangle through given AB and AC
  sorry -- Proof ends here for Lean equivalent statement

end find_tan_A_l186_186520


namespace area_of_triangle_ADE_l186_186645

-- Define the sides and properties of the triangles
def AB : ℝ := 8
def BC : ℝ := 6
def BD : ℝ := 8

-- Define the property that AE bisects the angle CAD
def bisects (A C D E : Type) : Prop := ∃ (AE : Type), AE = CA ∧ AE = DA / 2

theorem area_of_triangle_ADE :
  ∃ A B C D E : Type, 
    ∃ AB BD BC : ℝ, 
      AB = 8 ∧ BC = 6 ∧ BD = 8 ∧ 
      is_right ∠B ∧ bisects A C D E ∧ 
      area (triangle ADE) = 20 := 
by
  -- proof
  sorry

end area_of_triangle_ADE_l186_186645


namespace continuous_distribution_function_of_Y_l186_186678

noncomputable def problem (X : ℕ → ℝ) (α : ℝ) : Prop :=
  0 < α ∧ α < 1 ∧ 
  (∀ n m : ℕ, n ≠ m → Independent (X n) (X m)) ∧
  (∃ pdf : ℝ → ℝ, ∀ n, pdf = pdf ∧ pdf ≠ const 0 ∧ 
    let S := ∑ n, (α^n) * (X n)
    (λ x, Pr({y : ℝ | y = S }) < ⊤)) ∧
  (Continuous (λ x, Pr( {y : ℝ | y ≤ x} )))

-- The theorem we need to prove:
theorem continuous_distribution_function_of_Y (X : ℕ → ℝ) (α : ℝ) :
  problem X α → Continuous (λ x, Pr( { y : ℝ | y = ∑ i, (α^i) * X i})) :=
sorry

end continuous_distribution_function_of_Y_l186_186678


namespace even_number_of_scientists_with_odd_handshakes_l186_186565

theorem even_number_of_scientists_with_odd_handshakes (n : ℕ) (a : Fin n → ℕ) :
  (∃ k : ℕ, (Finset.univ.filter (λ i, a i % 2 = 1)).card = 2 * k) :=
sorry

end even_number_of_scientists_with_odd_handshakes_l186_186565


namespace three_digit_integers_with_odd_factors_count_l186_186293

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l186_186293


namespace three_digit_integers_with_odd_factors_l186_186304

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l186_186304


namespace area_of_smaller_circle_l186_186643

-- Conditions
variables (P A A' B B' : Point)
variable (r : ℝ)
variable {R : ℝ} -- radius of the larger circle
variable {PA PA' AB A'B' : ℝ}
variable {k : ℝ}

-- Assuming necessary geometric conditions
axiom circles_tangent : ∃ (c1 c2 : Circle), c1.radius = r ∧ c2.radius = R ∧ externally_tangent c1 c2
axiom tangents_PA_PAP : PA = 5 ∧ PA' = 5
axiom tangents_AB_A'B' : AB = 3 ∧ A'B' = 3
axiom similar_triangles : ∃ k, R = k * r ∧ 2 * r + r = 4

-- Statement to prove
theorem area_of_smaller_circle : (π * (r^2) = (16/9) * π) :=
by
  -- Proof is skipped
  sorry

end area_of_smaller_circle_l186_186643


namespace three_digit_odds_factors_count_l186_186352

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l186_186352


namespace team_E_not_played_against_team_B_l186_186723

-- Define the teams
inductive Team
| A | B | C | D | E | F
deriving DecidableEq

open Team

-- Define the matches played by each team
def matches_played : Team → Nat
| A => 5
| B => 4
| C => 3
| D => 2
| E => 1
| F => 0

-- Define the pairwise matches function
def paired : Team → Team → Prop
| A, B => true
| A, C => true
| A, D => true
| A, E => true
| A, F => true
| B, C => true
| B, D => true
| B, F  => true
| _, _ => false

-- Define the theorem based on the conditions and question
theorem team_E_not_played_against_team_B :
  ¬ paired E B :=
by
  sorry

end team_E_not_played_against_team_B_l186_186723


namespace find_f_ln2_l186_186829

variable (f : ℝ → ℝ)

-- Condition: f is an odd function
axiom odd_fn : ∀ x : ℝ, f (-x) = -f x

-- Condition: f(x) = e^(-x) - 2 for x < 0
axiom def_fn : ∀ x : ℝ, x < 0 → f x = Real.exp (-x) - 2

-- Problem: Find f(ln 2)
theorem find_f_ln2 : f (Real.log 2) = 0 := by
  sorry

end find_f_ln2_l186_186829


namespace possible_placement_diff_4_impossible_placement_diff_3_l186_186586

-- Definitions for the 4 x 4 grid and placements
structure Board :=
  (numbers : Fin 4 × Fin 4 → Fin 17)
  (valid_placement : ∀ (i j : Fin 4) (di dj : Fin 2), 
    (di, dj) ≠ (0, 0) ∧ i.val + di.val < 4 ∧ j.val + dj.val < 4 →
    (numbers (i, j) - numbers (⟨i.val + di.val, _⟩, ⟨j.val + dj.val, _⟩)).natAbs ≤ 4)

-- Theorem for the possibility of placing with difference at most 4
theorem possible_placement_diff_4 : 
  ∃ (b : Board), ∀ (i j : Fin 4) (di dj : Fin 2), 
    (di, dj) ≠ (0, 0) ∧ i.val + di.val < 4 ∧ j.val + dj.val < 4 →
    (b.numbers (i, j) - b.numbers (⟨i.val + di.val, _⟩, ⟨j.val + dj.val, _⟩)).natAbs ≤ 4 := 
sorry

-- Theorem for the impossibility of placing with difference at most 3
theorem impossible_placement_diff_3 : 
  ¬ ∃ (b : Board), ∀ (i j : Fin 4) (di dj : Fin 2), 
    (di, dj) ≠ (0, 0) ∧ i.val + di.val < 4 ∧ j.val + dj.val < 4 →
    (b.numbers (i, j) - b.numbers (⟨i.val + di.val, _⟩, ⟨j.val + dj.val, _⟩)).natAbs ≤ 3 := 
sorry

end possible_placement_diff_4_impossible_placement_diff_3_l186_186586


namespace radius_of_inscribed_semicircle_l186_186898

theorem radius_of_inscribed_semicircle (PQ PR: ℝ) (angleQ: angle Q) (is_isosceles_right_triangle: triangle PQR): 
  PQ = 10 → PR = 10 → angleQ = π/2 → 
  r = 10 - 5 * sqrt 2 := 
sorry

end radius_of_inscribed_semicircle_l186_186898


namespace three_digit_integers_with_odd_factors_l186_186226

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l186_186226


namespace polynomial_root_divisibility_l186_186953

noncomputable def p (x : ℤ) (a b c : ℤ) : ℤ := x^3 + a * x^2 + b * x + c

theorem polynomial_root_divisibility (a b c : ℤ) (h : ∃ u v : ℤ, p 0 a b c = (u * v * u * v)) :
  2 * (p (-1) a b c) ∣ (p 1 a b c + p (-1) a b c - 2 * (1 + p 0 a b c)) :=
sorry

end polynomial_root_divisibility_l186_186953


namespace three_digit_integers_with_odd_factors_l186_186155

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l186_186155


namespace scientific_notation_of_million_l186_186620

theorem scientific_notation_of_million (x : ℝ) (h : x = 2600000) : x = 2.6 * 10^6 := by
  sorry

end scientific_notation_of_million_l186_186620


namespace length_of_chord_l186_186479

-- Definitions from conditions
def parabola (x : ℝ) : Set (ℝ × ℝ) := { p | p.2 * p.2 = 8 * p.1 }
def line (k : ℝ) : Set (ℝ × ℝ) := { p | p.2 = k * p.1 - 2 }
def midpoint_x_is_2 (A B : ℝ × ℝ) : Prop := (A.1 + B.1) / 2 = 2

-- Definition to calculate distance between two points
def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Main theorem statement
theorem length_of_chord
  (k : ℝ)
  (A B : ℝ × ℝ)
  (hA : A ∈ parabola A.1 ∩ line k)
  (hB : B ∈ parabola B.1 ∩ line k)
  (hMid : midpoint_x_is_2 A B) :
  distance A B = 2 * Real.sqrt 15 :=
sorry

end length_of_chord_l186_186479


namespace three_digit_oddfactors_count_l186_186128

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l186_186128


namespace number_of_three_digit_integers_with_odd_number_of_factors_l186_186331

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l186_186331


namespace tetrahedron_point_locus_proof_l186_186912

-- Define the tetrahedron with specified edge lengths.
structure Tetrahedron :=
  (edge_lengths : (fin 6) → ℝ)
  -- Constrain five edges to have length 2 and one edge to have length 1
  (edges_valid : (∃ i j, i ≠ j ∧ edge_lengths i = 1 ∧ ∀ k ≠ i, edge_lengths k = 2))

-- Define the point P and distances to the faces.
structure PointDistancesToFaces (P : ℝ × ℝ × ℝ) (tet : Tetrahedron) :=
  (a b c d : ℝ)

noncomputable def minimal_maximal_sum_loci (P : ℝ × ℝ × ℝ) (tet : Tetrahedron) 
  (dists : PointDistancesToFaces P tet) : set (ℝ × ℝ × ℝ) :=
  { p | (∀ a b c d, dists.a + dists.b + dists.c + dists.d = minimal_sum → p ∈ edge_with_1_length) ∧
        (∀ a b c d, dists.a + dists.b + dists.c + dists.d = maximal_sum → p ∈ opposite_edge)}

theorem tetrahedron_point_locus_proof (P : ℝ × ℝ × ℝ) (tet : Tetrahedron) 
  (dists : PointDistancesToFaces P tet) : 
  minimal_maximal_sum_loci P tet dists.p :=
sorry

end tetrahedron_point_locus_proof_l186_186912


namespace expected_volunteers_by_2022_l186_186907

noncomputable def initial_volunteers : ℕ := 1200
noncomputable def increase_2021 : ℚ := 0.15
noncomputable def increase_2022 : ℚ := 0.30

theorem expected_volunteers_by_2022 :
  (initial_volunteers * (1 + increase_2021) * (1 + increase_2022)) = 1794 := 
by
  sorry

end expected_volunteers_by_2022_l186_186907


namespace point_in_second_quadrant_iff_l186_186902

theorem point_in_second_quadrant_iff (a : ℝ) : (a - 2 < 0) ↔ (a < 2) :=
by
  sorry

end point_in_second_quadrant_iff_l186_186902


namespace three_digit_oddfactors_count_l186_186141

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l186_186141


namespace number_of_three_digit_integers_with_odd_factors_l186_186408

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l186_186408


namespace ratio_brothers_sisters_boys_ratio_brothers_sisters_girls_l186_186530

variables (x y k t : ℕ)

theorem ratio_brothers_sisters_boys (h1 : (x+1) / y = k) (h2 : x / (y+1) = t) :
  (x / (y+1)) = t := 
by simp [h2]

theorem ratio_brothers_sisters_girls (h1 : (x+1) / y = k) (h2 : x / (y+1) = t) :
  ((x+1) / y) = k := 
by simp [h1]

#check ratio_brothers_sisters_boys    -- Just for verification
#check ratio_brothers_sisters_girls   -- Just for verification

end ratio_brothers_sisters_boys_ratio_brothers_sisters_girls_l186_186530


namespace three_digit_integers_odd_factors_count_l186_186442

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l186_186442


namespace number_of_possible_M_l186_186872

theorem number_of_possible_M :
  (M : Set ℕ) → {2, 3} ⊂ M ∧ M ⊂ {1, 2, 3, 4, 5} → ∃ S, S = {M | ∀ M, {2, 3} ⊂ M ∧ M ⊂ {1, 2, 3, 4, 5}} ∧ S.card = 6 :=
by
  sorry

end number_of_possible_M_l186_186872


namespace santino_total_fruits_l186_186580

theorem santino_total_fruits :
  ∃ (papaya_trees mango_trees papayas_per_tree mangos_per_tree total_fruits : ℕ),
    papaya_trees = 2 ∧
    papayas_per_tree = 10 ∧
    mango_trees = 3 ∧
    mangos_per_tree = 20 ∧
    total_fruits = (papaya_trees * papayas_per_tree) + (mango_trees * mangos_per_tree) ∧
    total_fruits = 80 :=
by
  -- Definitions
  let papaya_trees := 2
  let papayas_per_tree := 10
  let mango_trees := 3
  let mangos_per_tree := 20
  let total_fruits := (papaya_trees * papayas_per_tree) + (mango_trees * mangos_per_tree)

  -- Goal
  have : papaya_trees = 2 := rfl
  have : papayas_per_tree = 10 := rfl
  have : mango_trees = 3 := rfl
  have : mangos_per_tree = 20 := rfl
  have : total_fruits = 80 :=
    by
      calc
        total_fruits
          = (papaya_trees * papayas_per_tree) + (mango_trees * mangos_per_tree) : rfl
          ... = 20 + 60 : by simp [papaya_trees, papayas_per_tree, mango_trees, mangos_per_tree]
          ... = 80 : rfl
  exact ⟨papaya_trees, mango_trees, papayas_per_tree, mangos_per_tree, total_fruits, rfl, rfl, rfl, rfl, this⟩


end santino_total_fruits_l186_186580


namespace log_expression_identity_l186_186880

theorem log_expression_identity :
  let l₁ := log 5 5
  let l₂ := log 4 9
  let l₃ := log 3 2
  l₁ * l₂ * l₃ = 1 → l₁ = 1 :=
by
  intros l₁ l₂ l₃ h
  unfold l₁
  exact sorry

end log_expression_identity_l186_186880


namespace condition_A_necessary_but_not_sufficient_l186_186891

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), a i

noncomputable def is_increasing_sequence (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S (n + 1) > S n

theorem condition_A_necessary_but_not_sufficient
  (a : ℕ → ℝ) (q : ℝ) (hq : q > 0)
  (geo_seq : is_geometric_sequence a q) :
  ∃ a1 : ℝ, a 0 = a1 ∧ ∀ n : ℕ, sum_of_first_n_terms a n > sum_of_first_n_terms a (n - 1) :=
sorry

end condition_A_necessary_but_not_sufficient_l186_186891


namespace num_odd_factors_of_three_digit_integers_eq_22_l186_186114

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l186_186114


namespace no_zeros_sin_log_l186_186462

open Real

theorem no_zeros_sin_log (x : ℝ) (h1 : 1 < x) (h2 : x < exp 1) : ¬ (sin (log x) = 0) :=
sorry

end no_zeros_sin_log_l186_186462


namespace three_digit_oddfactors_count_l186_186130

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l186_186130


namespace three_digit_perfect_squares_count_l186_186191

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l186_186191


namespace soccer_teams_participation_l186_186494

theorem soccer_teams_participation (total_games : ℕ) (teams_play : ℕ → ℕ) (x : ℕ) :
  (total_games = 20) → (teams_play x = x * (x - 1)) → x = 5 :=
by
  sorry

end soccer_teams_participation_l186_186494


namespace common_ratio_geometric_sequence_l186_186633

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

def S (a : ℕ → ℝ) : ℕ → ℝ
| 0     := a 0
| (n+1) := a 0 * ((1 - (q^(n + 2))) / (1 - q))  -- sum of first n terms of a geometric series with common ratio q

theorem common_ratio_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h_seq : is_geometric_sequence a q)
  (h : S a 2 + 4 * S a 1 + a 0 = 0) : q = -2 ∨ q = -3 :=
sorry

end common_ratio_geometric_sequence_l186_186633


namespace leading_coefficient_g_l186_186065

def g (x : ℝ) : ℝ := sorry  -- Placeholder definition for g

axiom g_property : ∀ x : ℝ, g(x + 1) - g(x) = 8 * x + 2

theorem leading_coefficient_g : 
  ∃ c : ℝ, ∀ x : ℝ, (g(x) = 4 * x^2 + x + c) :=
begin
  sorry
end

end leading_coefficient_g_l186_186065


namespace gold_coins_equality_l186_186798

theorem gold_coins_equality (pouches : List ℕ) 
  (h_pouches_length : pouches.length = 9)
  (h_pouches_sum : pouches.sum = 60)
  : (∃ s_2 : List (List ℕ), s_2.length = 2 ∧ ∀ l ∈ s_2, l.sum = 30) ∧
    (∃ s_3 : List (List ℕ), s_3.length = 3 ∧ ∀ l ∈ s_3, l.sum = 20) ∧
    (∃ s_4 : List (List ℕ), s_4.length = 4 ∧ ∀ l ∈ s_4, l.sum = 15) ∧
    (∃ s_5 : List (List ℕ), s_5.length = 5 ∧ ∀ l ∈ s_5, l.sum = 12) :=
sorry

end gold_coins_equality_l186_186798


namespace three_digit_perfect_squares_count_l186_186189

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l186_186189


namespace cosine_dihedral_angle_l186_186629

/-- A regular triangular prism with a lateral edge length of 2
and an equilateral triangle base with a side length of 1 has a cross-section passing
through side AB which divides the volume into two equal parts. This theorem proves
that the cosine of the dihedral angle between the cross-section and the base is 2 / sqrt 15. -/
theorem cosine_dihedral_angle (S A B C D E : Type) [regular_triangular_prism S A B C 2 1] 
  (cross_section : divides_volume S A B 2 = (1 / 2) * volume S A B C) :
  cos (dihedral_angle (cross_section_plane S A B) (base_plane A B C)) = 2 / sqrt 15 := 
sorry

end cosine_dihedral_angle_l186_186629


namespace three_digit_perfect_squares_count_l186_186192

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l186_186192


namespace num_three_digit_ints_with_odd_factors_l186_186245

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l186_186245


namespace max_min_values_of_f_l186_186844

-- Define the function f(x) and the conditions about its coefficients
def f (x : ℝ) (p q : ℝ) : ℝ := x^3 - p * x^2 - q * x

def intersects_x_axis_at_1 (p q : ℝ) : Prop :=
  f 1 p q = 0

-- Define the maximum and minimum values on the interval [-1, 1]
theorem max_min_values_of_f (p q : ℝ) 
  (h1 : f 1 p q = 0) :
  (p = 2) ∧ (q = -1) ∧ (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x 2 (-1) ≤ f (1/3) 2 (-1)) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f (-1) 2 (-1) ≤ f x 2 (-1)) :=
sorry

end max_min_values_of_f_l186_186844


namespace does_not_necessarily_hold_l186_186469

theorem does_not_necessarily_hold (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : ac < 0) : 
  ¬(∀ b, cb^2 < ab^2 ) := 
sorry

end does_not_necessarily_hold_l186_186469


namespace num_three_digit_ints_with_odd_factors_l186_186257

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l186_186257


namespace exists_number_with_multiple_irreducible_factorizations_l186_186550

def is_irreducible_in_Vn (n m : ℕ) : Prop :=
  n > 2 ∧ ∀ (p q : ℕ), (p ∈ {1 + k * n | k : ℕ} ∧ q ∈ {1 + k * n | k : ℕ}) → p * q ≠ m

def exists_multiple_factorizations (n : ℕ) (Vn : ℕ → Prop) : Prop :=
  ∃ r ∈ Vn n, ∃ (f1 f2 : list ℕ),
    (∀ m ∈ f1, Vn n m) ∧ (∀ m ∈ f2, Vn n m) ∧
    (∀ m ∈ f1, is_irreducible_in_Vn n m) ∧ 
    (∀ m ∈ f2, is_irreducible_in_Vn n m) ∧ 
    f1.prod = r ∧ f2.prod = r ∧
    f1 ≠ f2

theorem exists_number_with_multiple_irreducible_factorizations (n : ℕ) (h : n > 2) :
  exists_multiple_factorizations n (λ n m, ∃ k : ℕ, m = 1 + k * n) :=
sorry

end exists_number_with_multiple_irreducible_factorizations_l186_186550


namespace three_digit_oddfactors_count_is_22_l186_186086

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l186_186086


namespace min_vector_magnitude_l186_186816

variables (t : ℝ)
def a : ℝ × ℝ × ℝ := (1 - t, 2 * t - 1, 0)
def b : ℝ × ℝ × ℝ := (2, t, t)

noncomputable def vector_sub : ℝ × ℝ × ℝ := (b.1 - a.1, b.2 - a.2, b.3 - a.3)

def vector_magnitude (v: ℝ × ℝ × ℝ) : ℝ := 
  real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

theorem min_vector_magnitude : ∀ t : ℝ, vector_magnitude (vector_sub t) = sqrt 2 :=
by
  sorry

end min_vector_magnitude_l186_186816


namespace three_digit_integers_with_odd_factors_l186_186302

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l186_186302


namespace scott_runs_80_meters_when_chris_runs_100_l186_186977

theorem scott_runs_80_meters_when_chris_runs_100 :
  (Scott_distance Chris_distance : ℝ) (h_ratio : Scott_distance / Chris_distance = 4 / 5) 
  (h_chris : Chris_distance = 100) : Scott_distance = 80 :=
by sorry

end scott_runs_80_meters_when_chris_runs_100_l186_186977


namespace calculate_difference_l186_186549

def g (x : ℝ) : ℝ := 3 * x^2 + 4 * x + 5

theorem calculate_difference (x h : ℝ) : g (x + h) - g x = h * (6 * x + 3 * h + 4) :=
by
  sorry

end calculate_difference_l186_186549


namespace joe_initial_average_score_l186_186531

theorem joe_initial_average_score (A : ℕ) 
  (h1 : ∑ s in finset.fin_range 3, s = 165) 
  (h2 : ∑ (s : ℕ) in ({35} : finset ℕ), s + ∑ s in finset.fin_range 3, s = 4 * A) :
  A = 50 :=
by
  sorry

end joe_initial_average_score_l186_186531


namespace three_digit_integers_with_odd_factors_l186_186158

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l186_186158


namespace range_of_x_that_satisfies_f_l186_186562

def f (x: ℝ) : ℝ := 
  if x ≤ 0 then x + 1 else 2 ^ x

theorem range_of_x_that_satisfies_f (x: ℝ) : f(x) + f(x - 1/2) > 1 ↔ x > -1/4 :=
  sorry

end range_of_x_that_satisfies_f_l186_186562


namespace green_ball_removal_l186_186491

theorem green_ball_removal :
  let initial_balls := 600
  let initial_green_balls := 0.7 * initial_balls
  let initial_yellow_balls := initial_balls - initial_green_balls
  let target_percentage := 0.6
  let x := 150
  remove_green x initial_green_balls initial_balls target_percentage :=
  (initial_green_balls - x) = target_percentage * (initial_balls - x) :=
by
  let initial_balls := 600
  let initial_green_balls := 0.7 * initial_balls
  let initial_yellow_balls := initial_balls - initial_green_balls
  let target_percentage := 0.6
  let x := 150
  have h: (420 - 150) = 0.6 * (600 - 150) := by sorry
  exact h

end green_ball_removal_l186_186491


namespace next_shipment_correct_l186_186683

variable (first_shipment second_shipment couscous_per_dish total_dishes next_shipment : ℕ)

-- Define initial conditions
def first_shipment := 7
def second_shipment := 13
def couscous_per_dish := 5
def total_dishes := 13

-- Define total usage and initial shipments
def total_usage : ℕ := total_dishes * couscous_per_dish
def initial_shipments : ℕ := first_shipment + second_shipment

-- Prove that the next day's shipment was 45 pounds
theorem next_shipment_correct : next_shipment = total_usage - initial_shipments := by
  sorry

end next_shipment_correct_l186_186683


namespace three_digit_integers_odd_factors_count_l186_186456

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l186_186456


namespace interior_box_surface_area_l186_186984

-- Given conditions
def original_length : ℕ := 40
def original_width : ℕ := 60
def corner_side : ℕ := 8

-- Calculate the initial area
def area_original : ℕ := original_length * original_width

-- Calculate the area of one corner
def area_corner : ℕ := corner_side * corner_side

-- Calculate the total area removed by four corners
def total_area_removed : ℕ := 4 * area_corner

-- Theorem to state the final area remaining
theorem interior_box_surface_area : 
  area_original - total_area_removed = 2144 :=
by
  -- Place the proof here
  sorry

end interior_box_surface_area_l186_186984


namespace three_digit_oddfactors_count_is_22_l186_186092

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l186_186092


namespace cot_identity_l186_186013

theorem cot_identity (x : ℝ) : 
  (sin x ≠ 0 ∧ sin (x / 2) ≠ 0 → cot (x / 2) - cot (2 * x) = (sin (3 * x / 2)) / (sin (x / 2) * sin (2 * x))) :=
by
  intros h
  sorry

end cot_identity_l186_186013


namespace f_recursive_l186_186860

noncomputable def f (n : ℕ) : ℚ :=
  ∑ i in finset.range (n+1), 1 / (n + 1 + i : ℚ)

theorem f_recursive (n : ℕ) :
  f (n+1) - f n = (1 / (2*n + 1 : ℚ)) + (1 / (2*n + 2)) - (1 / (n+1)) :=
by
  sorry

end f_recursive_l186_186860


namespace equal_gifts_possible_l186_186584

theorem equal_gifts_possible :
  ∃ (f : Fin 7 → Fin 7 → Bool),
  (∀ i, ∑ j, if f i j then 1 else 0 = Fin.val i) ∧ -- Pay close attention to different number of gifts given
  (∀ i, ∑ j, if f j i then 1 else 0 = 3)         -- Ensure everyone receives 3 gifts
  ∧ (∀ i, f i i = false) :=                      -- No self-gifting
sorry

end equal_gifts_possible_l186_186584


namespace three_digit_integers_odd_factors_count_l186_186449

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l186_186449


namespace num_odd_factors_of_three_digit_integers_eq_22_l186_186104

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l186_186104


namespace root_of_sqrt_2x_plus_3_eq_x_l186_186961

theorem root_of_sqrt_2x_plus_3_eq_x :
  ∀ x : ℝ, (sqrt (2 * x + 3) = x) → x = 3 := by
  intro x
  intro h
  sorry

end root_of_sqrt_2x_plus_3_eq_x_l186_186961


namespace smallest_circle_equation_l186_186831

theorem smallest_circle_equation :
  ∃ (x y : ℝ), (y^2 = 4 * x) ∧ (x - 1)^2 + y^2 = 1 ∧ ((x - 1)^2 + y^2 = 1) = (x^2 + y^2 = 1) := 
sorry

end smallest_circle_equation_l186_186831


namespace solve_equation_l186_186590

theorem solve_equation : ∀ x y : ℤ, x^2 + y^2 = 3 * x * y → x = 0 ∧ y = 0 := by
  intros x y h
  sorry

end solve_equation_l186_186590


namespace a_14_pow_14_l186_186944

noncomputable def a : ℕ → ℤ 
| 1 := 11^11 
| 2 := 12^12 
| 3 := 13^13 
| (n + 1) := if h₁ : 4 ≤ n + 1 then abs (a n - a (n - 1)) + abs (a (n - 1) - a (n - 2)) else a (n + 1)

theorem a_14_pow_14 : a (14 ^ 14) = 1 := 
by 
  sorry

end a_14_pow_14_l186_186944


namespace remainder_is_neg7_l186_186005

def polynomial : Polynomial ℤ := 5 * Polynomial.X ^ 5 - 12 * Polynomial.X ^ 4 + 3 * Polynomial.X ^ 3 - 7 * Polynomial.X + 15
def divisor : Polynomial ℤ := 3 * Polynomial.X - 6

theorem remainder_is_neg7 : polynomial.eval 2 = -7 :=
by sorry

end remainder_is_neg7_l186_186005


namespace fill_boxes_l186_186775

theorem fill_boxes (a b c d e f g : ℤ) 
  (h1 : a + (-1) + 2 = 4)
  (h2 : 2 + 1 + b = 3)
  (h3 : c + (-4) + (-3) = -2)
  (h4 : b - 5 - 4 = -9)
  (h5 : f = d - 3)
  (h6 : g = d + 3)
  (h7 : -8 = 4 + 3 - 9 - 2 + (d - 3) + (d + 3)) : 
  a = 3 ∧ b = 0 ∧ c = 5 ∧ d = -2 ∧ e = -9 ∧ f = -5 ∧ g = 1 :=
by {
  sorry
}

end fill_boxes_l186_186775


namespace sum_of_super_cool_rectangle_areas_l186_186764

def isSuperCoolRectangle (a b : ℕ) : Prop :=
  a * b = 6 * (a + b)

theorem sum_of_super_cool_rectangle_areas :
  let areas := { z | ∃ a b : ℕ, isSuperCoolRectangle a b ∧ z = a * b }
  ∑' z in areas, z = 606 :=
by
  sorry

end sum_of_super_cool_rectangle_areas_l186_186764


namespace distance_B_amusement_park_l186_186634

variable (d_A d_B v_A v_B t_A t_B : ℝ)

axiom h1 : v_A = 3
axiom h2 : v_B = 4
axiom h3 : d_B = d_A + 2
axiom h4 : t_A + t_B = 4
axiom h5 : t_A = d_A / v_A
axiom h6 : t_B = d_B / v_B

theorem distance_B_amusement_park:
  d_A / 3 + (d_A + 2) / 4 = 4 → d_B = 8 :=
by
  sorry

end distance_B_amusement_park_l186_186634


namespace length_of_A_l186_186538

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem length_of_A'B' (A B C A' B' : ℝ × ℝ) (hc : C = (3, 7)) (hA : A = (0, 10)) (hB : B = (0, 15)) 
  (hA' : A'.1 = A'.2) (hB' : B'.1 = B'.2) (hAC : C ∈ set.line_of_two_points A A')
  (hBC : C ∈ set.line_of_two_points B B') : distance A' B' = 10 * real.sqrt 2 / 11 := sorry

end length_of_A_l186_186538


namespace three_digit_integers_with_odd_number_of_factors_l186_186162

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l186_186162


namespace girls_neighbors_l186_186536

theorem girls_neighbors (n k : ℕ) (gt_cond : n > k) (ge_cond : k ≥ 1) (students : Fin (2 * n + 1) → Bool)
  (girls_count : (Fin (2 * n + 1)).count (λ i => students i = true) = n + 1) :
    ∃ i : Fin (2 * n + 1), (Finset.range (2 * k)).count (λ j => students (i + j + 1) = true) ≥ k :=
sorry

end girls_neighbors_l186_186536


namespace number_of_three_digit_integers_with_odd_factors_l186_186409

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l186_186409


namespace three_digit_integers_with_odd_number_of_factors_count_l186_186265

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l186_186265


namespace num_three_digit_integers_with_odd_factors_l186_186368

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l186_186368


namespace angle_of_sum_cis_arithmetic_sequence_l186_186741

noncomputable def sum_cis_arithmetic_sequence : ℝ :=
  let angles := list.range' 60 10 9 in
  list.sum (angles.map (λ x => complex.exp (real.pi * x / 180)))

theorem angle_of_sum_cis_arithmetic_sequence :
  ∃ r, r > 0 ∧ ∃ θ, θ = 100 ∧ sum_cis_arithmetic_sequence = r * (complex.exp (real.pi * θ / 180)) := 
sorry

end angle_of_sum_cis_arithmetic_sequence_l186_186741


namespace salami_pizza_fraction_l186_186879

theorem salami_pizza_fraction 
    (d_pizza : ℝ) 
    (n_salami_diameter : ℕ) 
    (n_salami_total : ℕ) 
    (h1 : d_pizza = 16)
    (h2 : n_salami_diameter = 8) 
    (h3 : n_salami_total = 32) 
    : 
    (32 * (Real.pi * (d_pizza / (2 * n_salami_diameter / 2)) ^ 2)) / (Real.pi * (d_pizza / 2) ^ 2) = 1 / 2 := 
by 
  sorry

end salami_pizza_fraction_l186_186879


namespace erasers_left_l186_186636

/-- 
There are initially 250 erasers in a box. Doris takes 75 erasers, Mark takes 40 
erasers, and Ellie takes 30 erasers out of the box. Prove that 105 erasers are 
left in the box.
-/
theorem erasers_left (initial_erasers : ℕ) (doris_takes : ℕ) (mark_takes : ℕ) (ellie_takes : ℕ)
  (h_initial : initial_erasers = 250)
  (h_doris : doris_takes = 75)
  (h_mark : mark_takes = 40)
  (h_ellie : ellie_takes = 30) :
  initial_erasers - doris_takes - mark_takes - ellie_takes = 105 :=
  by 
  sorry

end erasers_left_l186_186636


namespace circumcircles_common_intersection_l186_186539

open EuclideanGeometry

noncomputable def common_intersection_point (ABC : Triangle) (P : Point) (A1 B1 C1 A2 B2 C2 : Point) : Prop :=
  let A := ABC.V₁
  let B := ABC.V₂
  let C := ABC.V₃
  isAcuteABC : ∠ A B C < 90 ∧ ∠ B C A < 90 ∧ ∠ C A B < 90,
  P_interior : ∃ p : inTriangle P ABC,
  not_on_altitudeA : ¬ lies_on_altitude P A ABC,
  not_on_altitudeB : ¬ lies_on_altitude P B ABC,
  not_on_altitudeC : ¬ lies_on_altitude P C ABC,
  A1_foot : A1 = foot_of_perpendicular A BC,
  B1_foot : B1 = foot_of_perpendicular B AC,
  C1_foot : C1 = foot_of_perpendicular C AB,
  AP_intersect : ∃ k, circle (C.center) (C.radius) = k ∧ isOnRay A P ∧ A2 ∈ k,
  BP_intersect : ∃ k, circle (C.center) (C.radius) = k ∧ isOnRay B P ∧ B2 ∈ k,
  CP_intersect : ∃ k, circle (C.center) (C.radius) = k ∧ isOnRay C P ∧ C2 ∈ k
in
  let k_a := circumcircle A A1 A2
  let k_b := circumcircle B B1 B2
  let k_c := circumcircle C C1 C2
  ∃ common_point, common_point ∈ k_a ∧ common_point ∈ k_b ∧ common_point ∈ k_c

-- To prove this result
theorem circumcircles_common_intersection {ABC : Triangle} {P A1 B1 C1 A2 B2 C2 : Point}
  (isAcuteABC : ∠ABC < 90 ∧ ∠BCA < 90 ∧ ∠CAB < 90)
  (P_interior : ∃ p : inTriangle P ABC)
  (not_on_altitudeA : ¬lies_on_altitude P A ABC)
  (not_on_altitudeB : ¬lies_on_altitude P B ABC)
  (not_on_altitudeC : ¬lies_on_altitude P C ABC)
  (A1_foot : A1 = foot_of_perpendicular A BC)
  (B1_foot : B1 = foot_of_perpendicular B AC)
  (C1_foot : C1 = foot_of_perpendicular C AB)
  (AP_intersect : ∃ k, circle (C.center) (C.radius) = k ∧ isOnRay A P ∧ A2 ∈ k)
  (BP_intersect : ∃ k, circle (C.center) (C.radius) = k ∧ isOnRay B P ∧ B2 ∈ k)
  (CP_intersect : ∃ k, circle (C.center) (C.radius) = k ∧ isOnRay C P ∧ C2 ∈ k)
: ∃ K, K ∈ circumcircle A A1 A2 ∧ K ∈ circumcircle B B1 B2 ∧ K ∈ circumcircle C C1 C2 :=
sorry

end circumcircles_common_intersection_l186_186539


namespace bridge_length_correct_l186_186720

noncomputable def length_of_bridge (train_length : ℝ) (train_speed_kmph : ℝ) (time_to_cross_bridge : ℝ) : ℝ :=
(let train_speed_mps := train_speed_kmph * (1000 / 3600) in
 let total_distance := train_speed_mps * time_to_cross_bridge in
 total_distance - train_length)

theorem bridge_length_correct :
  length_of_bridge 165 36 82.49340052795776 = 659.9340052795776 :=
by
  simp [length_of_bridge]
  sorry

end bridge_length_correct_l186_186720


namespace number_of_sodas_bought_l186_186657

theorem number_of_sodas_bought
  (sandwich_cost : ℝ)
  (num_sandwiches : ℝ)
  (soda_cost : ℝ)
  (total_cost : ℝ)
  (h1 : sandwich_cost = 3.49)
  (h2 : num_sandwiches = 2)
  (h3 : soda_cost = 0.87)
  (h4 : total_cost = 10.46) :
  (total_cost - num_sandwiches * sandwich_cost) / soda_cost = 4 := 
sorry

end number_of_sodas_bought_l186_186657


namespace remainder_when_divided_by_r_minus_1_l186_186006

def f (r : Int) : Int := r^14 - r + 5

theorem remainder_when_divided_by_r_minus_1 : f 1 = 5 := by
  sorry

end remainder_when_divided_by_r_minus_1_l186_186006


namespace three_digit_integers_with_odd_number_of_factors_count_l186_186262

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l186_186262


namespace count_solutions_l186_186080

def satisfies_equation (x : ℤ) : Prop :=
  (x^2 - x - 2)^(x + 3) = 1

theorem count_solutions :
  ∃ n, n = 4 ∧ ∃ S : Finset ℤ, S.card = n ∧ ∀ x, x ∈ S ↔ satisfies_equation x :=
by
  sorry

end count_solutions_l186_186080


namespace tan_neg_5_pi_over_6_l186_186776

-- Definitions and statements of the problem
def angle_in_radians : ℝ := -5 * Real.pi / 6
def angle_in_degrees := -150
def tangent_period := 180
def tan_30_degrees := 1 / Real.sqrt 3

-- Statement part
theorem tan_neg_5_pi_over_6 :
    Real.tan angle_in_radians = tan_30_degrees :=
sorry

end tan_neg_5_pi_over_6_l186_186776


namespace num_odd_factors_of_three_digit_integers_eq_22_l186_186102

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l186_186102


namespace complex_part_reciprocal_l186_186012

theorem complex_part_reciprocal (a : ℝ) (h : (a ≠ 0)) : 
  (let z := (2 + complex.i * a) * complex.i in
   z.re = -(z.im)) → a = 2 :=
by
  sorry

end complex_part_reciprocal_l186_186012


namespace num_odd_factors_of_three_digit_integers_eq_22_l186_186103

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l186_186103


namespace solution_mn_l186_186818

theorem solution_mn (m n : ℤ) (h1 : |m| = 4) (h2 : |n| = 5) (h3 : n < 0) : m + n = -1 ∨ m + n = -9 := 
by
  sorry

end solution_mn_l186_186818


namespace three_digit_integers_with_odd_number_of_factors_l186_186166

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l186_186166


namespace jack_initial_yen_l186_186921

theorem jack_initial_yen 
  (pounds yen_per_pound euros pounds_per_euro total_yen : ℕ)
  (h₁ : pounds = 42)
  (h₂ : euros = 11)
  (h₃ : pounds_per_euro = 2)
  (h₄ : yen_per_pound = 100)
  (h₅ : total_yen = 9400) : 
  ∃ initial_yen : ℕ, initial_yen = 3000 :=
by
  sorry

end jack_initial_yen_l186_186921


namespace emma_investment_l186_186659

-- Define the basic problem parameters
def P : ℝ := 2500
def r : ℝ := 0.04
def n : ℕ := 21
def expected_amount : ℝ := 6101.50

-- Define the compound interest formula result
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- State the theorem
theorem emma_investment : 
  compound_interest P r n = expected_amount := 
  sorry

end emma_investment_l186_186659


namespace M_intersection_N_l186_186955

-- Definition of sets M and N
def M : Set ℝ := { x | x^2 + 2 * x - 8 < 0 }
def N : Set ℝ := { y | ∃ x : ℝ, y = 2^x }

-- Goal: Prove that M ∩ N = (0, 2)
theorem M_intersection_N :
  M ∩ N = { y | 0 < y ∧ y < 2 } :=
sorry

end M_intersection_N_l186_186955


namespace num_three_digit_ints_with_odd_factors_l186_186249

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l186_186249


namespace charlie_dana_rest_days_l186_186745

-- Define the cyclic schedules for Charlie and Dana
def charlie_schedule (n : ℕ) : Bool :=
  (n % 6 = 4) || (n % 6 = 5)

def dana_schedule (n : ℕ) : Bool :=
  (n % 7 = 5) || (n % 7 = 6)

-- Define the main theorem to prove
theorem charlie_dana_rest_days : 
  (∑ n in Finset.range 1000, if charlie_schedule n && dana_schedule n then 1 else 0) = 92 :=
by
  sorry

end charlie_dana_rest_days_l186_186745


namespace number_of_three_digit_integers_with_odd_number_of_factors_l186_186323

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l186_186323


namespace range_of_t_l186_186042

-- Define the function f with the given conditions
def f : ℝ → ℝ := 
λ x, if x ≤ 0 then x^3 else x^3

-- Condition: f(x-1) = -f(-x+1) for all real x
axiom f_fun_eq (x : ℝ) : f(x - 1) = -f(-x + 1)

-- Proof statement: For any t, if for all x in [t, t+2], f(x+t) ≥ 2√2 f(x), then t ≥ √2
theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, t ≤ x ∧ x ≤ t + 2 → f(x + t) ≥ 2 * real.sqrt 2 * f(x)) → t ≥ real.sqrt 2 :=
by
  sorry

end range_of_t_l186_186042


namespace sequence_nonzero_l186_186851

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧
  a 2 = 2 ∧
  ∀ n : ℕ, 
    (a n * a (n + 1)) % 2 = 0 → a (n + 2) = 5 * a (n + 1) - 3 * a n ∧
    (a n * a (n + 1)) % 2 = 1 → a (n + 2) = a (n + 1) - a n

theorem sequence_nonzero (a : ℕ → ℤ) (h : sequence a) : ∀ n : ℕ, 0 < n → a n ≠ 0 :=
sorry

end sequence_nonzero_l186_186851


namespace javier_six_attractions_order_count_l186_186529

theorem javier_six_attractions_order_count : 
  ∀ (n : ℕ), n = 6 → nat.factorial n = 720 :=
begin
  intros n h,
  rw h,
  exact rfl,
end

end javier_six_attractions_order_count_l186_186529


namespace count_f_t_mod_5_eq_0_l186_186547

def f (x : ℤ) : ℤ := x^3 + 2 * x^2 + 3 * x + 1

def T : Finset ℤ := (Finset.range 31).map ⟨(λ x, x : ℕ → ℤ), Int.coe_nat_injective⟩

theorem count_f_t_mod_5_eq_0 : (T.filter (λ t, f t % 5 = 0)).card = 6 :=
by
  sorry

end count_f_t_mod_5_eq_0_l186_186547


namespace problem1_problem2_l186_186054

noncomputable def f (a : ℝ) (h_pos : 0 < a) (h_neq_one : a ≠ 1) : ℝ → ℝ :=
  λ x, - (Real.sqrt a) / (a^x + Real.sqrt a)

theorem problem1 (a : ℝ) (h_pos : 0 < a) (h_neq_one : a ≠ 1) :
  ∀ x, f a h_pos h_neq_one x + f a h_pos h_neq_one (1-x) = -1 :=
sorry

theorem problem2 (a : ℝ) (h_pos : 0 < a) (h_neq_one : a ≠ 1) :
  f a h_pos h_neq_one (-2) + f a h_pos h_neq_one (-1) +
  f a h_pos h_neq_one 0 + f a h_pos h_neq_one 1 + 
  f a h_pos h_neq_one 2 + f a h_pos h_neq_one 3 = -3 :=
sorry

end problem1_problem2_l186_186054


namespace jason_grass_cutting_time_l186_186917

def total_minutes (hours : ℕ) : ℕ := hours * 60
def minutes_per_yard : ℕ := 30
def total_yards_per_weekend : ℕ := 8 * 2
def total_minutes_per_weekend : ℕ := minutes_per_yard * total_yards_per_weekend
def convert_minutes_to_hours (minutes : ℕ) : ℕ := minutes / 60

theorem jason_grass_cutting_time : 
  convert_minutes_to_hours total_minutes_per_weekend = 8 := by
  sorry

end jason_grass_cutting_time_l186_186917


namespace chickens_and_rabbits_l186_186722

theorem chickens_and_rabbits (total_animals : ℕ) (total_legs : ℕ) (chickens : ℕ) (rabbits : ℕ) 
    (h1 : total_animals = 40) 
    (h2 : total_legs = 108) 
    (h3 : total_animals = chickens + rabbits) 
    (h4 : total_legs = 2 * chickens + 4 * rabbits) : 
    chickens = 26 ∧ rabbits = 14 :=
by
  sorry

end chickens_and_rabbits_l186_186722


namespace find_m_of_split_number_l186_186800

noncomputable def split_number := λ m : ℕ, 3 + (m * (m - 1)) / 2

theorem find_m_of_split_number (m : ℕ) (h1 : m > 1) (h2 : split_number m = 2017) : m = 45 :=
sorry

end find_m_of_split_number_l186_186800


namespace christinas_total_driving_time_l186_186746

theorem christinas_total_driving_time :
  let total_distance := 210
  let speed_limit1 := 30
  let speed_limit2 := 40
  let speed_limit3 := 50
  let speed_limit4 := 60
  let friend_distance2 := 120
  let christina_distance3 := 50
  let remaining_distance4 := (total_distance - friend_distance2 - christina_distance3) / 2
  let time_first_segment := 40 / speed_limit1 * 60
  let time_fourth_segment := 20 / speed_limit4 * 60
  in
  time_first_segment + time_fourth_segment = 100 := 
by sorry

end christinas_total_driving_time_l186_186746


namespace inequality_am_gm_l186_186555

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    a^3 / (b * c) + b^3 / (c * a) + c^3 / (a * b) ≥ a + b + c :=
by {
    sorry
}

end inequality_am_gm_l186_186555


namespace three_digit_integers_odd_factors_count_l186_186446

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l186_186446


namespace num_three_digit_integers_with_odd_factors_l186_186372

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l186_186372


namespace solve_equation_l186_186631

theorem solve_equation (x : ℝ) (hx : x ≠ 1) : (x / (x - 1) - 1 = 1) → (x = 2) :=
by
  sorry

end solve_equation_l186_186631


namespace find_a_plus_b_l186_186556

theorem find_a_plus_b (a b : ℝ) (f g : ℝ → ℝ) (h1 : f = λ x, a * x + b)
    (h2 : g = λ x, 3 * x - 6) (h3 : ∀ x, g (f x) = 4 * x + 3) : a + b = 13 / 3 := 
begin
  sorry
end

end find_a_plus_b_l186_186556


namespace find_general_formula_find_Tn_l186_186904

-- Definitions based on given conditions
def a_sequence (n : ℕ) : ℕ := 2 * n - 1
def b_sequence (n : ℕ) : ℚ := 4 / (a_sequence n * a_sequence (n + 1))
def sum_sequence (n : ℕ) : ℚ := (range n).sum b_sequence

-- Statement for the first part
theorem find_general_formula (n : ℕ) (a3_eq_5 : a_sequence 3 = 5) (S4_eq_16 : sum_sequence 4 = 16) :
  a_sequence n = 2 * n - 1 :=
sorry

-- Statement for the second part
theorem find_Tn (n : ℕ) (a3_eq_5 : a_sequence 3 = 5) (S4_eq_16 : sum_sequence 4 = 16) :
  sum_sequence n = 4 * n / (2 * n + 1) :=
sorry

end find_general_formula_find_Tn_l186_186904


namespace three_digit_oddfactors_count_l186_186133

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l186_186133


namespace three_digit_odds_factors_count_l186_186346

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l186_186346


namespace time_to_cross_bridge_l186_186870

-- Definitions
def length_train1 : ℝ := 120
def length_train2 : ℝ := 150
def speed_train1_kmph : ℝ := 36
def speed_train1_mps : ℝ := speed_train1_kmph * (1000 / 3600)
def speed_train2_kmph : ℝ := 45
def speed_train2_mps : ℝ := speed_train2_kmph * (1000 / 3600)
def length_bridge : ℝ := 300

-- The relative speed is the sum of the individual speeds as the trains are moving towards each other
def relative_speed : ℝ := speed_train1_mps + speed_train2_mps

-- The total distance to be covered includes both trains' lengths and the bridge length
def total_distance : ℝ := length_train1 + length_train2 + length_bridge

-- Calculating the time taken using distance/speed
def time_taken : ℝ := total_distance / relative_speed

-- Statement to be proved
theorem time_to_cross_bridge : time_taken = 25.33 :=
by
  sorry

end time_to_cross_bridge_l186_186870


namespace greatest_number_of_schools_l186_186595
noncomputable theory

def canDistribute (total : Nat) (minPerSchool maxPerSchool : Nat) (nSchools : Nat) : Prop :=
  total % nSchools = 0 ∧ minPerSchool ≤ total / nSchools ∧ total / nSchools ≤ maxPerSchool

theorem greatest_number_of_schools (n : Nat) :
  canDistribute 48 4 8 n ∧
  canDistribute 32 2 4 n ∧
  canDistribute 60 3 7 n ∧
  canDistribute 20 1 3 n →
  n = 4 :=
by
  sorry

end greatest_number_of_schools_l186_186595


namespace number_of_three_digit_squares_l186_186431

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l186_186431


namespace system_solution_and_range_l186_186071

theorem system_solution_and_range (a x y : ℝ) (h1 : 2 * x + y = 5 * a) (h2 : x - 3 * y = -a + 7) :
  (x = 2 * a + 1 ∧ y = a - 2) ∧ (-1/2 ≤ a ∧ a < 2 → 2 * a + 1 ≥ 0 ∧ a - 2 < 0) :=
by
  sorry

end system_solution_and_range_l186_186071


namespace three_digit_odds_factors_count_l186_186360

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l186_186360


namespace positive_quadratic_if_and_only_if_l186_186552

variable (a : ℝ)
def p (x : ℝ) : ℝ := a * x^2 + 2 * x + 1

theorem positive_quadratic_if_and_only_if (h : ∀ x : ℝ, p a x > 0) : a > 1 := sorry

end positive_quadratic_if_and_only_if_l186_186552


namespace fibonacci_sum_eq_3_l186_186541

def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

noncomputable def fib_sum : ℕ → ℝ
| 0     := (fib 0) / 2^0
| (n+1) := fib (n+1) / 2^(n+1) + fib_sum n

theorem fibonacci_sum_eq_3 :
  (∑' n : ℕ, (fib n / 2^n: ℝ)) = 3 :=
  sorry

end fibonacci_sum_eq_3_l186_186541


namespace arithmetic_sequence_problem_l186_186903

theorem arithmetic_sequence_problem (a : ℕ → ℤ) (d : ℤ) 
    (h1 : a 2 + a 6 + a 10 = 15)
    (h2 : a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12 = 99) :
    (∀ n, a n = 3 * n - 13) ∧ (∑ i in finset.range 20, a (i + 1) = 370) := 
by
  sorry

end arithmetic_sequence_problem_l186_186903


namespace three_digit_odds_factors_count_l186_186358

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l186_186358


namespace no_integer_n_perfect_cube_in_range_l186_186079

theorem no_integer_n_perfect_cube_in_range :
  ¬ ∃ n : ℤ, (4 ≤ n ∧ n ≤ 11) ∧ ∃ k : ℤ, (n^2 + 3 * n + 2 = k^3) :=
begin
  sorry
end

end no_integer_n_perfect_cube_in_range_l186_186079


namespace three_digit_oddfactors_count_l186_186136

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l186_186136


namespace ray_bounces_equilateral_triangle_l186_186943

theorem ray_bounces_equilateral_triangle (n : ℕ) : 
  (∃ (A B C : Type) [equilateral_triangle A B C] 
     (ray : A → Prop),
     (ray_reflections ray n ABC) ∧ 
     ray_returns_to_A ray n ∧ 
     (∀ k, ray_does_not_land_on ray k [B, C])) → 
  (n ≡ 1 [MOD 6] ∨ n ≡ 5 [MOD 6]) ∧ n ≠ 5 ∧ n ≠ 17 := 
sorry

end ray_bounces_equilateral_triangle_l186_186943


namespace projection_locus_l186_186819

noncomputable def parabola (p : ℝ) (hp : 0 < p) : set (ℝ × ℝ) :=
{ (x, y) | y^2 = 2 * p * x }

def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
v1.1 * v2.1 + v1.2 * v2.2 = 0

def locus (p : ℝ) (hp : 0 < p) : set (ℝ × ℝ) :=
{ (x, y) | x^2 - 2*p*x + y^2 = 0 ∧ 0 < x }

theorem projection_locus (p : ℝ) (hp : 0 < p) :
  ∀ (A B : ℝ × ℝ),
  (A ∈ parabola p hp) →
  (B ∈ parabola p hp) →
  is_perpendicular A B →
  let M := (x : ℝ × ℝ) in 
  locus p hp M :=
sorry

end projection_locus_l186_186819


namespace min_value_phi_phi_le_one_l186_186060

def h (a x : ℝ) : ℝ := -a * Real.log x
def phi (a : ℝ) : ℝ := 2 * a * (1 - Real.log (2 * a))

theorem min_value_phi (a : ℝ) (ha : a > 0) : 
  ∃ x, (x > 0) ∧ (∀ y, (y > 0) → (h a y) ≥ (h a x)) ∧ (phi a = h a x) := sorry

theorem phi_le_one (a : ℝ) (ha : 0 < a) : phi a ≤ 1 := sorry

end min_value_phi_phi_le_one_l186_186060


namespace sum_distances_l186_186050

noncomputable def point := (ℝ, ℝ)

def curve (P : point) : Prop := ∃ x, P = (x, sqrt (-x^2 + 10*x - 9))

def A : point := (1, 0)

def on_line (x : ℝ) : Prop := x = -1/3

def distance_point (P Q : point) : ℝ := sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def distance_line (P : point) : ℝ := abs (3 * P.1 + 1) / 3

def conditions (B C : point) : Prop :=
  curve B ∧ curve C ∧
  distance_line B = distance_point B A ∧
  distance_line C = distance_point C A

theorem sum_distances (B C : point) (h : conditions B C) :
  distance_point B A + distance_point C A = 8 :=
sorry

end sum_distances_l186_186050


namespace three_digit_integers_with_odd_factors_l186_186398

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l186_186398


namespace focal_length_of_hyperbola_l186_186992

theorem focal_length_of_hyperbola :
  ∀ (ρ θ : ℝ), (5 * ρ^2 * cos (2 * θ) + ρ^2 - 24 = 0) → focal_length ρ θ = 2 * sqrt 10 :=
by
  sorry

end focal_length_of_hyperbola_l186_186992


namespace number_of_three_digit_integers_with_odd_factors_l186_186413

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l186_186413


namespace trapezium_area_l186_186782

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 13) :
  (1 / 2) * (a + b) * h = 247 :=
by
  rw [ha, hb, hh]
  norm_num
  sorry

end trapezium_area_l186_186782


namespace star_example_l186_186763

def star (x y : ℝ) : ℝ := 2 * x * y - 3 * x + y

theorem star_example : (star 6 4) - (star 4 6) = -8 := by
  sorry

end star_example_l186_186763


namespace three_digit_integers_with_odd_factors_l186_186392

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l186_186392


namespace num_three_digit_ints_with_odd_factors_l186_186252

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l186_186252


namespace original_game_no_chance_chief_reclaim_false_chief_knew_expulsions_true_maximum_expulsions_six_natives_cannot_lose_second_game_l186_186705

noncomputable def original_no_chance (n : ℕ) (trades : ℕ) : Prop :=
  (n = 30) → (trades > 15) → (∃ (a b : ℕ), a ≠ b ∧ a = b)

noncomputable def chief_cannot_reclaim (remaining : ℕ) (coins : ℕ) : Prop :=
  coins = 270 → (remaining = 30) → (∀ x : ℕ, remaining - x = unique_coins → redistributed_coins ≤ 270) → False

noncomputable def chief_knew_expulsions (expelled : ℕ) : Prop :=
  (expelled = 6) → True

noncomputable def max_expulsions (total : ℕ) (remaining : ℕ) (coins : ℕ) : Prop :=
  remaining = total - 6 ∧ coins = 270

noncomputable def merchant_lost_first (lost_first : Prop) (second_game_result : Prop) : Prop :=
  lost_first → (second_game_result = False)

-- Statement
theorem original_game_no_chance : original_no_chance 30 435 := sorry

theorem chief_reclaim_false : chief_cannot_reclaim 30 270 := sorry

theorem chief_knew_expulsions_true : chief_knew_expulsions 6 := sorry

theorem maximum_expulsions_six : max_expulsions 30 24 270 := sorry

theorem natives_cannot_lose_second_game : merchant_lost_first True False := sorry

end original_game_no_chance_chief_reclaim_false_chief_knew_expulsions_true_maximum_expulsions_six_natives_cannot_lose_second_game_l186_186705


namespace intersection_nonempty_implies_t_lt_1_l186_186854

def M (x : ℝ) := x ≤ 1
def P (t : ℝ) (x : ℝ) := x > t

theorem intersection_nonempty_implies_t_lt_1 {t : ℝ} (h : ∃ x, M x ∧ P t x) : t < 1 :=
by
  sorry

end intersection_nonempty_implies_t_lt_1_l186_186854


namespace three_digit_odds_factors_count_l186_186353

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l186_186353


namespace nonneg_int_solutions_eq_binom_l186_186974

-- Definitions for the conditions
variables (k n : ℕ)

-- The statement to be proven
theorem nonneg_int_solutions_eq_binom (h_k_pos : k > 0) (h_n_pos : n > 0) :
  (∃ (x : Fin k → ℕ), (∑ i, x i) = n) ↔ Nat.choose (n+k-1) (k-1) := 
sorry

end nonneg_int_solutions_eq_binom_l186_186974


namespace solve_for_x_l186_186010

theorem solve_for_x (x : ℝ) (h1 : 4 * x + 6 ≥ 0) (h2 : 8 * x + 12 ≥ 0) :
  (sqrt (4 * x + 6) / sqrt (8 * x + 12) = sqrt 2 / 2) ↔ (x ≥ -3 / 2) :=
sorry

end solve_for_x_l186_186010


namespace total_apples_picked_is_108_l186_186736

noncomputable def apples_picked_total : ℕ :=
  let benny_trees := 4
  let benny_apples_per_tree := 2
  let benny_apples_total := benny_trees * benny_apples_per_tree
  let dan_trees := 5
  let dan_apples_per_tree := 9
  let dan_apples_total := dan_trees * dan_apples_per_tree
  let sarah_apples_total := (dan_apples_total / 2).ceil
  let total_benny_dan := benny_apples_total + dan_apples_total
  let lisa_apples_total := (total_benny_dan * 3 / 5).ceil
  benny_apples_total + dan_apples_total + sarah_apples_total + lisa_apples_total

theorem total_apples_picked_is_108 : apples_picked_total = 108 := 
by 
  -- Here would be the proof steps, which we are skipping as per instructions
  sorry

end total_apples_picked_is_108_l186_186736


namespace arithmetic_sequence_num_terms_l186_186740

theorem arithmetic_sequence_num_terms :
  ∀ (a d l : ℕ), 
    a = 3 → 
    d = 6 → 
    l = 69 → 
    ∃ n : ℕ, 
      n = ((l - a) / d) + 1 ∧ 
      n = 12 :=
by
  intros a d l ha hd hl
  use ((l - a) / d) + 1
  split
  { rw [ha, hd]
    norm_num }
  { rw [ha, hd, hl]
    norm_num }

end arithmetic_sequence_num_terms_l186_186740


namespace three_digit_integers_with_odd_factors_count_l186_186294

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l186_186294


namespace three_digit_integers_with_odd_factors_l186_186393

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l186_186393


namespace find_range_of_a_l186_186841

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp (2 * x) - a * x

theorem find_range_of_a (a : ℝ) :
  (∀ x > 0, f x a > a * x^2 + 1) → a ≤ 2 :=
by
  sorry

end find_range_of_a_l186_186841


namespace convex_cyclic_quadrilaterals_count_l186_186867

/-- A convex cyclic quadrilateral is a cyclic quadrilateral where the sides are in cyclic order such 
    that the sides satisfy the triangle inequality for any three consecutive sides. -/
structure ConvexCyclicQuadrilateral (a b c d : ℕ) : Prop :=
(perimeter: a + b + c + d = 40)
(min_side : min a (min b (min c d)) ≥ 5)

theorem convex_cyclic_quadrilaterals_count : 
  {q : Σ a b c d : ℕ, ConvexCyclicQuadrilateral a b c d // true}.toSet.card = 442 :=
by sorry

end convex_cyclic_quadrilaterals_count_l186_186867


namespace cauchy_functional_eq_l186_186789

theorem cauchy_functional_eq (f : ℚ → ℝ) (h : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ a : ℝ, ∀ q : ℚ, f q = a * q :=
sorry

end cauchy_functional_eq_l186_186789


namespace three_digit_integers_with_odd_factors_l186_186203

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l186_186203


namespace part_a_part_b_l186_186926

variable (G : Type) [group G]
variable (H : subgroup G)
variable (m n : ℕ)
variable [fintype G]
variable [fintype H]
variable (e : G) -- the neutral element of G
variable (H_proper : H < (⊤ : subgroup G))
variable (m_card : fintype.card G = m)
variable (n_card : fintype.card H = n)
variable (H_inter_conj : ∀ x ∈ (G : set G) \ H, (H.map (conjugate x)).to_set ∩ H.to_set = {e})

theorem part_a (x y : G) :
  (H.map (conjugate x)).to_set = (H.map (conjugate y)).to_set ↔ x⁻¹ * y ∈ H :=
sorry

theorem part_b :
  (∑ x in finset.univ.image (λ x : G, (H.map (conjugate x)).to_set), fintype.card ((H.map (conjugate x)).to_set)) = 1 + m * (n - 1) / n :=
sorry

end part_a_part_b_l186_186926


namespace heath_average_carrots_per_hour_l186_186073

theorem heath_average_carrots_per_hour 
  (rows1 rows2 : ℕ)
  (plants_per_row1 plants_per_row2 : ℕ)
  (hours1 hours2 : ℕ)
  (h1 : rows1 = 200)
  (h2 : rows2 = 200)
  (h3 : plants_per_row1 = 275)
  (h4 : plants_per_row2 = 325)
  (h5 : hours1 = 15)
  (h6 : hours2 = 25) :
  ((rows1 * plants_per_row1 + rows2 * plants_per_row2) / (hours1 + hours2) = 3000) :=
  by
  sorry

end heath_average_carrots_per_hour_l186_186073


namespace jason_grass_cutting_time_l186_186918

def total_minutes (hours : ℕ) : ℕ := hours * 60
def minutes_per_yard : ℕ := 30
def total_yards_per_weekend : ℕ := 8 * 2
def total_minutes_per_weekend : ℕ := minutes_per_yard * total_yards_per_weekend
def convert_minutes_to_hours (minutes : ℕ) : ℕ := minutes / 60

theorem jason_grass_cutting_time : 
  convert_minutes_to_hours total_minutes_per_weekend = 8 := by
  sorry

end jason_grass_cutting_time_l186_186918


namespace three_digit_odds_factors_count_l186_186347

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l186_186347


namespace solve_sqrt_equation_l186_186959

theorem solve_sqrt_equation :
  ∀ x : ℝ, (sqrt (2 * x + 3) = x) ↔ (x = 3) := by
  intro x
  split
  {
    -- Assume sqrt (2 * x + 3) = x and prove x = 3.
    sorry
  }
  {
    -- Assume x = 3 and prove sqrt (2 * x + 3) = x.
    sorry
  }

end solve_sqrt_equation_l186_186959


namespace length_of_DF_l186_186963

theorem length_of_DF (D E F P Q G : Point) 
  (h1 : isMedian D P ∧ isMedian E Q)
  (h2 : rightAngle (DP) (EQ))
  (h3 : divides G (DP) (3/5) (2/5))
  (h4 : divides G (EQ) (3/5) (2/5))
  (hDP : length DP = 27)
  (hEQ : length EQ = 30) : 
  length DF = 40.32 := 
sorry

end length_of_DF_l186_186963


namespace three_digit_integers_with_odd_factors_l186_186236

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l186_186236


namespace min_distance_in_triangle_l186_186486

theorem min_distance_in_triangle :
  ∀ (A B C D E : Type) 
    [has_ANGLE A B C] [has_DIST A B C]
    (angleBAC : angle B A C = 60)
    (AB : dist A B = 8)
    (AC : dist A C = 5)
    (AD : dist A D = 3) (D_on_AB : between A D B)
    (AE : dist A E = 2) (E_on_AC : between A E C),
  let BD := dist B D,
      CE := dist C E,
      BE := Math.sqrt (8^2 + 2^2 - 2 * 8 * 2 * Math.cos (60)),
      DE := Math.sqrt ((8 - 3)^2 + (5 - 2)^2),
      CD := 2
   in BE + DE + CD = Math.sqrt(52) + Math.sqrt(34) + 2 :=
sorry

end min_distance_in_triangle_l186_186486


namespace three_digit_oddfactors_count_l186_186139

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l186_186139


namespace three_digit_oddfactors_count_is_22_l186_186082

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l186_186082


namespace stockings_total_cost_l186_186970

noncomputable def total_cost (n : ℕ) (price_per_stocking : ℝ) (discount : ℝ) (monogram_cost_per_stocking : ℝ) : ℝ :=
  let total_stockings := 5 * n
  let initial_cost := total_stockings * price_per_stocking
  let discounted_price := initial_cost * (1 - discount)
  let monogramming_cost := total_stockings * monogram_cost_per_stocking
  discounted_price + monogramming_cost

theorem stockings_total_cost : 
  total_cost 9 20 0.1 5 = 1035 :=
by
  unfold total_cost
  rfl

end stockings_total_cost_l186_186970


namespace num_three_digit_ints_with_odd_factors_l186_186258

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l186_186258


namespace multiplication_table_odd_fraction_l186_186886

def is_odd (n : ℕ) : Prop := n % 2 = 1

def fraction_of_odd_products_to_nearest_hundredth : ℝ :=
  let factors := finset.range 16 
  let products := (finset.product factors factors).image (λ p, p.1 * p.2)
  let odd_products := (products.filter is_odd).card.to_nat
  let total_products := (finset.product factors factors).card.to_nat
  (odd_products : ℝ) / (total_products : ℝ)

theorem multiplication_table_odd_fraction : fraction_of_odd_products_to_nearest_hundredth = 0.25 :=
by
  sorry

end multiplication_table_odd_fraction_l186_186886


namespace total_flowers_purchased_l186_186920

-- Define the conditions
def sets : ℕ := 3
def pieces_per_set : ℕ := 90

-- State the proof problem
theorem total_flowers_purchased : sets * pieces_per_set = 270 :=
by
  sorry

end total_flowers_purchased_l186_186920


namespace number_of_three_digit_integers_with_odd_factors_l186_186418

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l186_186418


namespace suitcase_lock_settings_l186_186719

theorem suitcase_lock_settings : 
  (∃ settings : Finset (Finset (Fin 10)), settings.card = 4 ∧ settings.pairwise Disjoint) →
  ∃ n : ℕ, n = 10 * 9 * 8 * 7 ∧ n = 5040 :=
by
  sorry

end suitcase_lock_settings_l186_186719


namespace team_final_position_east_of_A_total_fuel_consumed_l186_186697

def final_position (distances : List Int) : Int :=
  distances.sum

def total_fuel_consumption (distances : List Int) (fuel_rate : Float) : Float :=
  let total_distance := distances.map Int.natAbs |>.sum + (distances.sum.abs)
  total_distance * fuel_rate

theorem team_final_position_east_of_A :
  final_position [+12, -6, +4, -2, -8, +13, -2] = 11 := by
  sorry

theorem total_fuel_consumed :
  total_fuel_consumption [+12, -6, +4, -2, -8, +13, -2] 0.2 = 11.6 := by
  sorry

end team_final_position_east_of_A_total_fuel_consumed_l186_186697


namespace three_digit_integers_with_odd_factors_l186_186317

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l186_186317


namespace diameter_of_circle_l186_186651

theorem diameter_of_circle (A : ℝ) (hA : A = 400 * Real.pi) : 
  (∃ d : ℝ, d = 1600) :=
by
  have hπ : Real.pi ≠ 0 := Real.pi_ne_zero
  -- Given A = π * r^2, solve for r
  let r := (A / Real.pi).sqrt
  have hr : r = 20 := by
    calc
      r = (400 * Real.pi / Real.pi).sqrt : by rw [hA]
      _ = 400.sqrt                     : by rw [Real.div_self hπ]
      _ = 20                           : by norm_num [Real.sqrt_eq_iff_mul_self_eq, ← pow_two, sq]

  -- Diameter is four times the square of the radius
  let d := 4 * r^2
  have hd : d = 1600 := by
    calc
      d = 4 * (20 : ℝ) ^ 2 : by rw [hr]
      _ = 4 * 400         : by norm_num
      _ = 1600            : by norm_num

  exact ⟨d, hd⟩

end diameter_of_circle_l186_186651


namespace clock_angle_at_5_15_l186_186076

def degrees_in_circle := 360
def hours_on_clock := 12
def hour_angle := degrees_in_circle / hours_on_clock  -- Each hour mark in degrees

def minutes_in_hour := 60
def minute_percentage := 15 / minutes_in_hour    -- 15 minutes as a fraction of an hour
def minute_angle := minute_percentage * degrees_in_circle  -- Angle of the minute hand at 15 minutes

def degrees_per_minute := hour_angle / minutes_in_hour    -- Hour hand movement per minute
def hour_position := 5 * hour_angle    -- Hour hand position at 5:00
def hour_additional := degrees_per_minute * 15    -- Additional movement by the hour hand in 15 minutes
def hour_angle_at_time := hour_position + hour_additional    -- Total hour hand position at 5:15

-- Calculate the smaller angle between the two hands
def smaller_angle := hour_angle_at_time - minute_angle

theorem clock_angle_at_5_15 : smaller_angle = 67.5 := by
  sorry

end clock_angle_at_5_15_l186_186076


namespace number_of_three_digit_integers_with_odd_factors_l186_186421

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l186_186421


namespace num_odd_factors_of_three_digit_integers_eq_22_l186_186112

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l186_186112


namespace remainder_sum_modulo_eleven_l186_186004

theorem remainder_sum_modulo_eleven :
  (88132 + 88133 + 88134 + 88135 + 88136 + 88137 + 88138 + 88139 + 88140 + 88141) % 11 = 1 :=
by
  sorry

end remainder_sum_modulo_eleven_l186_186004


namespace breakfast_cost_l186_186811

theorem breakfast_cost :
  ∀ (muffin_cost fruit_cup_cost : ℕ) (francis_muffins francis_fruit_cups kiera_muffins kiera_fruit_cups : ℕ),
  muffin_cost = 2 ∧ fruit_cup_cost = 3 ∧ francis_muffins = 2 ∧ francis_fruit_cups = 2 ∧ kiera_muffins = 2 ∧ kiera_fruit_cups = 1
  → (francis_muffins * muffin_cost + francis_fruit_cups * fruit_cup_cost + kiera_muffins * muffin_cost + kiera_fruit_cups * fruit_cup_cost = 17) :=
by
  intros muffin_cost fruit_cup_cost francis_muffins francis_fruit_cups kiera_muffins kiera_fruit_cups
  intro cond
  cases cond with muffin_cost_eq rest
  cases rest with fruit_cup_cost_eq rest
  cases rest with francis_muffins_eq rest
  cases rest with francis_fruit_cups_eq rest
  cases rest with kiera_muffins_eq kiera_fruit_cups_eq

  rw [muffin_cost_eq, fruit_cup_cost_eq, francis_muffins_eq, francis_fruit_cups_eq, kiera_muffins_eq, kiera_fruit_cups_eq]
  norm_num
  sorry

end breakfast_cost_l186_186811


namespace num_three_digit_ints_with_odd_factors_l186_186253

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l186_186253


namespace three_digit_integers_odd_factors_count_l186_186461

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l186_186461


namespace number_of_three_digit_integers_with_odd_number_of_factors_l186_186341

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l186_186341


namespace three_digit_integers_with_odd_factors_l186_186395

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l186_186395


namespace reconstruct_triangle_from_feet_of_altitudes_l186_186640

-- Define that the points A1, B1, and C1 are the feet of the altitudes from vertices A, B, and C of an acute-angled triangle ABC.
variables (A B C A1 B1 C1 : Type) [Nonempty A] [Nonempty B] [Nonempty C]
variables [Nonempty A1] [Nonempty B1] [Nonempty C1]

-- Define that the given triangle is acute-angled and that A1, B1, and C1 are the feet of the altitudes.
def is_acute_angled (ABC_flag : Type) : Prop := 
  -- logical statement ensuring triangle ABC is acute-angled
  sorry

def are_feet_of_altitudes (A1 B1 C1 ABC_flag : Type) : Prop := 
  -- logical statement ensuring A1, B1, and C1 are the feet of the altitudes from A, B, and C respectively
  sorry

theorem reconstruct_triangle_from_feet_of_altitudes :
  ∀ (A B C A1 B1 C1 : Type),
  (is_acute_angled (A × B × C)) →
  (are_feet_of_altitudes A1 B1 C1 (A × B × C)) →
  ∃ (reconstruct_ABC : Type), 
  -- reconstruction using compass and straightedge.
  sorry

end reconstruct_triangle_from_feet_of_altitudes_l186_186640


namespace negation_of_proposition_l186_186619

theorem negation_of_proposition :
  ¬ (∃ x : ℝ, x < 1) ↔ ∀ x : ℝ, x ≥ 1 :=
by sorry

end negation_of_proposition_l186_186619


namespace complement_union_intersection_l186_186956

variable U : Set α
variable A : Set α
variable B : Set α
variable C : Set α

variable a b c d e : α
variable hU : U = {a, b, c, d, e}
variable hA : A = {a, c, d}
variable hB : B = {a, b}
variable hC : C = {d, e}

theorem complement_union_intersection :
  (U \ A) ∪ (B ∩ C) = {b, e} :=
by sorry

end complement_union_intersection_l186_186956


namespace coefficient_x3_in_expansion_l186_186991

theorem coefficient_x3_in_expansion : 
  (coeff ((1 : ℝ) + 2 * X)^6 3) = 160 := 
sorry

end coefficient_x3_in_expansion_l186_186991


namespace Paco_ate_28_salty_cookies_l186_186567

-- Definitions of conditions
def sweet_cookies_eaten : ℕ := 15
def salty_cookies_eaten_more_than_sweet : ℕ := 13

-- Let S be the number of salty cookies Paco ate
def salty_cookies_eaten (S : ℕ) := S = sweet_cookies_eaten + salty_cookies_eaten_more_than_sweet

-- Proof statement (to be solved)
theorem Paco_ate_28_salty_cookies : ∃ S, salty_cookies_eaten S ∧ S = 28 :=
begin
  use 28,
  split,
  { refl },
  { refl }
end

end Paco_ate_28_salty_cookies_l186_186567


namespace three_digit_integers_with_odd_factors_l186_186303

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l186_186303


namespace set_union_intersection_example_l186_186855

open Set

theorem set_union_intersection_example :
  let A := {1, 3, 4, 5}
  let B := {2, 4, 6}
  let C := {0, 1, 2, 3, 4}
  (A ∪ B) ∩ C = ({1, 2, 3, 4} : Set ℕ) :=
by
  sorry

end set_union_intersection_example_l186_186855


namespace problem_l186_186025

-- Definitions and conditions as captured above
def omega : ℂ := -1/2 + (complex.I * (real.sqrt 3) / 2)

-- Statement of the mathematical proof problem
theorem problem (hω : omega = -1/2 + (complex.I * (real.sqrt 3) / 2)) : 
  1 + omega + omega^2 + omega^3 = 1 := 
by 
  sorry

end problem_l186_186025


namespace solve_integro_differential_equation_l186_186591

noncomputable def ϕ : ℝ → ℝ := λ x, x * Real.exp x - Real.exp x + 1

theorem solve_integro_differential_equation :
  (ϕ'' x + ∫ t in 0..x, Real.exp (2 * (x - t)) * (fun t => (deriv ϕ) t) t = Real.exp (2 * x)) ∧
  (ϕ 0 = 0) ∧
  ((deriv ϕ) 0 = 0) :=
by
  -- sorries will be used to skip the proof steps
  sorry

end solve_integro_differential_equation_l186_186591


namespace three_digit_integers_with_odd_factors_l186_186232

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l186_186232


namespace num_odd_factors_of_three_digit_integers_eq_22_l186_186109

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l186_186109


namespace three_digit_integers_odd_factors_count_l186_186444

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l186_186444


namespace number_of_three_digit_integers_with_odd_factors_l186_186416

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l186_186416


namespace num_three_digit_ints_with_odd_factors_l186_186246

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l186_186246


namespace length_of_AB_l186_186027

-- Definition and condition for the parabola
def parabola (x y : ℝ) : Prop := y^2 = 3 * x

-- Definition and condition for the focus
def focus : ℝ × ℝ := (3 / 4, 0)

-- Definition for the line equation passing through focus at 30 degree angle
def line_eq (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * (x - 3 / 4)

-- Statement of the problem to prove
theorem length_of_AB : 
  ∃ A B : ℝ × ℝ, 
    (let (xA, yA) := A, (xB, yB) := B in 
      parabola xA yA ∧ parabola xB yB ∧ 
      line_eq xA yA ∧ line_eq xB yB ∧ 
      (abs (xA - xB) + 3 / 2 = 12)) :=
sorry

end length_of_AB_l186_186027


namespace max_distance_to_line_is_3_l186_186517

noncomputable def center_of_circle_in_polar_coordinates :=
  let x_center := - (Real.sqrt 2 / 2)
  let y_center := - (Real.sqrt 2 / 2)
  (Real.sqrt (x_center ^ 2 + y_center ^ 2), Real.pi * 5 / 4)

theorem max_distance_to_line_is_3 (r : ℝ) :
  let d := 1 + Real.sqrt 2 / 2 in
  d + r = 3 → r = 2 - Real.sqrt 2 / 2 :=
by
  intros h
  sorry

end max_distance_to_line_is_3_l186_186517


namespace plane_distance_l186_186995

theorem plane_distance (n : ℕ) : n % 45 = 0 ∧ (n / 10) % 100 = 39 ∧ n <= 5000 → n = 1395 := 
by
  sorry

end plane_distance_l186_186995


namespace three_digit_integers_with_odd_factors_l186_186385

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l186_186385


namespace arithmetic_sequence_find_c_l186_186884

-- Define the triangle sides and the given equation
variable {a b c A B C : ℝ}
variable h1 : b * (cos (A / 2))^2 + a * (cos (B / 2))^2 = (3 / 2) * c

-- Problem (I) statement
theorem arithmetic_sequence (h1: b * (cos (A / 2))^2 + a * (cos (B / 2))^2 = (3 / 2) * c) : a + b = 2 * c :=
sorry

-- Additional conditions for Problem (II)
variable h2 : C = π / 3
variable h3 : (1 / 2) * a * b * sin C = 2 * sqrt 3

-- Problem (II) statement
theorem find_c (h1: b * (cos (A / 2))^2 + a * (cos (B / 2))^2 = (3 / 2) * c)
               (h2: C = π / 3)
               (h3: (1 / 2) * a * b * sin C = 2 * sqrt 3)
               : c = 2 * sqrt 2 :=
sorry

end arithmetic_sequence_find_c_l186_186884


namespace hyperbola_focus_distance_l186_186826

-- Define the hyperbola and foci conditions
def is_on_hyperbola (x y: ℝ) : Prop := (x^2 / 64) - (y^2 / 36) = 1

def focus_distance : ℝ := real.sqrt 100 

-- Lean 4 statement to prove the necessary property
theorem hyperbola_focus_distance (x y : ℝ) (h : is_on_hyperbola x y) (PF1_dist PF2_dist : ℝ) 
  (hF1 : PF1_dist = 17) : PF2_dist = 33 :=
sorry

end hyperbola_focus_distance_l186_186826


namespace total_number_of_birds_l186_186639

def geese : ℕ := 58
def ducks : ℕ := 37
def swans : ℕ := 42

theorem total_number_of_birds : geese + ducks + swans = 137 := by
  sorry

end total_number_of_birds_l186_186639


namespace average_height_correct_l186_186596

noncomputable def initially_calculated_average_height 
  (num_students : ℕ) (incorrect_height correct_height : ℕ) 
  (actual_average : ℝ) 
  (A : ℝ) : Prop :=
  let incorrect_sum := num_students * A
  let height_difference := incorrect_height - correct_height
  let actual_sum := num_students * actual_average
  incorrect_sum = actual_sum + height_difference

theorem average_height_correct 
  (num_students : ℕ) (incorrect_height correct_height : ℕ) 
  (actual_average : ℝ) :
  initially_calculated_average_height num_students incorrect_height correct_height actual_average 175 :=
by {
  sorry
}

end average_height_correct_l186_186596


namespace correct_propositions_count_l186_186729

-- Definitions of propositions
def proposition_1 : Prop :=
  ∀ (A B : Set α), Complementary A B → MutuallyExclusive A B

def proposition_2 : Prop :=
  ∀ (A B : Set α) [ProbabilitySpace Ω], P(A ∪ B) = P(A) + P(B)

def proposition_3 : Prop :=
  ∀ (A B C : Set α) [ProbabilitySpace Ω], MutuallyExclusive A B C →
    P(A) + P(B) + P(C) = 1

def proposition_4 : Prop :=
  ∀ (A B : Set α) [ProbabilitySpace Ω], P(A) + P(B) = 1 → Complementary A B

-- The goal is to prove that only one of these propositions is correct
theorem correct_propositions_count : 
  (NumberOfTrue [proposition_1, proposition_2, proposition_3, proposition_4] = 1) :=
begin
  sorry
end

end correct_propositions_count_l186_186729


namespace domain_of_sqrt_div_l186_186606

theorem domain_of_sqrt_div (x : ℝ) :
  (sqrt (1 - x^2) ≥ 0) ∧ (x ≠ 0) ↔ (x ∈ set.Icc (-1 : ℝ) 1) \ {0} :=
by
  sorry

end domain_of_sqrt_div_l186_186606


namespace number_of_three_digit_integers_with_odd_number_of_factors_l186_186340

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l186_186340


namespace maximum_value_condition_l186_186940

open Real

theorem maximum_value_condition {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h1 : x + y = 16) (h2 : x = 2 * y) :
  (1 / x + 1 / y) = 9 / 32 :=
by
  sorry

end maximum_value_condition_l186_186940


namespace three_digit_oddfactors_count_l186_186127

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l186_186127


namespace radius_of_unique_circle_l186_186793

noncomputable def circle_radius (z : ℂ) (h k : ℝ) : ℝ :=
  if z = 2 then 1/4 else 0  -- function that determines the circle

def unique_circle_radius : Prop :=
  let x1 := 2
  let y1 := 0
  
  let x2 := 3 / 2
  let y2 := Real.sqrt 11 / 2

  let h := 7 / 4 -- x-coordinate of the circle's center
  let k := 0    -- y-coordinate of the circle's center

  let r := 1 / 4 -- Radius of the circle
  
  -- equation of the circle passing through (x1, y1) and (x2, y2) should satisfy
  -- the radius of the resulting circle is r

  (x1 - h)^2 + y1^2 = r^2 ∧ (x2 - h)^2 + y2^2 = r^2

theorem radius_of_unique_circle :
  unique_circle_radius :=
sorry

end radius_of_unique_circle_l186_186793


namespace num_three_digit_integers_with_odd_factors_l186_186373

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l186_186373


namespace three_digit_integers_odd_factors_count_l186_186454

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l186_186454


namespace num_odd_factors_of_three_digit_integers_eq_22_l186_186115

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l186_186115


namespace curve_C2_eq_l186_186948

def curve_C (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def reflect_y_axis (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-x)
def reflect_x_axis (f : ℝ → ℝ) (x : ℝ) : ℝ := - (f x)

theorem curve_C2_eq (a b c : ℝ) (h : a ≠ 0) :
  ∀ x : ℝ, reflect_x_axis (reflect_y_axis (curve_C a b c)) x = -a * x^2 + b * x - c := by
  sorry

end curve_C2_eq_l186_186948


namespace num_three_digit_ints_with_odd_factors_l186_186247

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l186_186247


namespace median_of_consecutive_integers_is_100_l186_186672

-- Define the set of consecutive integers and conditions
def consecutive_integers (a : ℤ) (n : ℕ) : list ℤ := list.range n |>.map (λ i, a + i)

-- Define the sum condition
def sum_condition (a : ℤ) (n : ℕ) : Prop := n > 0 ∧ 2 * a + (n - 1) = 200

-- Define the concept of median for our specific problem
def median (a : ℤ) (n : ℕ) : ℤ := a + (n - 1) / 2

-- The theorem to prove 
theorem median_of_consecutive_integers_is_100 (a : ℤ) (n : ℕ) 
 (h_sum_condition : sum_condition a n) 
 : median a n = 100 :=
sorry

end median_of_consecutive_integers_is_100_l186_186672
