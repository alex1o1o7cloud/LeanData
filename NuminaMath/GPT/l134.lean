import Mathlib

namespace couscous_dishes_l134_134215

def dishes (a b c d : ℕ) : ℕ := (a + b + c) / d

theorem couscous_dishes :
  dishes 7 13 45 5 = 13 :=
by
  unfold dishes
  sorry

end couscous_dishes_l134_134215


namespace jane_paid_cashier_l134_134999

-- Define the conditions in Lean
def skirts_bought : ℕ := 2
def price_per_skirt : ℕ := 13
def blouses_bought : ℕ := 3
def price_per_blouse : ℕ := 6
def change_received : ℤ := 56

-- Calculate the total cost in Lean
def cost_of_skirts : ℕ := skirts_bought * price_per_skirt
def cost_of_blouses : ℕ := blouses_bought * price_per_blouse
def total_cost : ℕ := cost_of_skirts + cost_of_blouses
def amount_paid : ℤ := total_cost + change_received

-- Lean statement to prove the question
theorem jane_paid_cashier :
  amount_paid = 100 :=
by
  sorry

end jane_paid_cashier_l134_134999


namespace tangent_line_at_one_minimum_a_range_of_a_l134_134375

-- Definitions for the given functions
def g (a x : ℝ) := a * x^2 - (a + 2) * x
noncomputable def h (x : ℝ) := Real.log x
noncomputable def f (a x : ℝ) := g a x + h x

-- Part (1): Prove the tangent line equation at x = 1 for a = 1
theorem tangent_line_at_one (x y : ℝ) (h_x : x = 1) (h_a : 1 = (1 : ℝ)) :
  x + y + 1 = 0 := by
  sorry

-- Part (2): Prove the minimum value of a given certain conditions
theorem minimum_a (a : ℝ) (h_a_pos : 0 < a) (h_x : 1 ≤ x ∧ x ≤ Real.exp 1)
  (h_fmin : ∀ x, f a x ≥ -2) : 
  a = 1 := by
  sorry

-- Part (3): Prove the range of values for a given a condition
theorem range_of_a (a x₁ x₂ : ℝ) (h_x : 0 < x₁ ∧ x₁ < x₂) 
  (h_f : ∀ x₁ x₂, (f a x₁ - f a x₂) / (x₁ - x₂) > -2) :
  0 ≤ a ∧ a ≤ 8 := by
  sorry

end tangent_line_at_one_minimum_a_range_of_a_l134_134375


namespace second_number_in_first_set_l134_134754

theorem second_number_in_first_set :
  ∃ (x : ℝ), (20 + x + 60) / 3 = (10 + 80 + 15) / 3 + 5 ∧ x = 40 :=
by
  use 40
  sorry

end second_number_in_first_set_l134_134754


namespace factorize_1_factorize_2_l134_134841

-- Proof problem 1: Prove x² - 6x + 9 = (x - 3)²
theorem factorize_1 (x : ℝ) : x^2 - 6 * x + 9 = (x - 3)^2 :=
by sorry

-- Proof problem 2: Prove x²(y - 2) - 4(y - 2) = (y - 2)(x + 2)(x - 2)
theorem factorize_2 (x y : ℝ) : x^2 * (y - 2) - 4 * (y - 2) = (y - 2) * (x + 2) * (x - 2) :=
by sorry

end factorize_1_factorize_2_l134_134841


namespace isabella_initial_hair_length_l134_134997

theorem isabella_initial_hair_length
  (final_length : ℕ)
  (growth_over_year : ℕ)
  (initial_length : ℕ)
  (h_final : final_length = 24)
  (h_growth : growth_over_year = 6)
  (h_initial : initial_length = 18) :
  initial_length + growth_over_year = final_length := 
by 
  sorry

end isabella_initial_hair_length_l134_134997


namespace probability_both_numbers_are_prime_l134_134627

open Nat

def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

/-- Define the number of ways to choose two elements from a set of size n -/
def num_ways_to_choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_both_numbers_are_prime :
  let total_pairs := num_ways_to_choose_2 30
  let prime_pairs := num_ways_to_choose_2 primes_up_to_30.length
  (prime_pairs.to_rat / total_pairs.to_rat) = 5 / 48 := by
  sorry

end probability_both_numbers_are_prime_l134_134627


namespace proposition_B_proposition_C_l134_134983

variable (a b c d : ℝ)

-- Proposition B: If |a| > |b|, then a² > b²
theorem proposition_B (h : |a| > |b|) : a^2 > b^2 :=
sorry

-- Proposition C: If (a - b)c² > 0, then a > b
theorem proposition_C (h : (a - b) * c^2 > 0) : a > b :=
sorry

end proposition_B_proposition_C_l134_134983


namespace Emily_candies_l134_134732

theorem Emily_candies (jennifer_candies emily_candies bob_candies : ℕ) 
    (h1: jennifer_candies = 2 * emily_candies)
    (h2: jennifer_candies = 3 * bob_candies)
    (h3: bob_candies = 4) : emily_candies = 6 :=
by
  -- Proof to be provided
  sorry

end Emily_candies_l134_134732


namespace find_a10_l134_134389

variable {q : ℝ}
variable {a : ℕ → ℝ}

-- Sequence conditions
axiom geo_seq (n : ℕ) : a (n + 1) = a n * q
axiom positive_ratio : 0 < q
axiom condition_1 : a 2 = 1
axiom condition_2 : a 4 * a 8 = 2 * (a 5) ^ 2

theorem find_a10 : a 10 = 16 := by
  sorry

end find_a10_l134_134389


namespace exists_f_ff_eq_square_l134_134066

open Nat

theorem exists_f_ff_eq_square : ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n ^ 2 :=
by
  -- proof to be provided
  sorry

end exists_f_ff_eq_square_l134_134066


namespace expression_value_l134_134489

theorem expression_value : (15 + 7)^2 - (15^2 + 7^2) = 210 := by
  sorry

end expression_value_l134_134489


namespace sale_in_third_month_l134_134332

def average_sale (s1 s2 s3 s4 s5 s6 : ℕ) : ℕ :=
  (s1 + s2 + s3 + s4 + s5 + s6) / 6

theorem sale_in_third_month
  (S1 S2 S3 S4 S5 S6 : ℕ)
  (h1 : S1 = 6535)
  (h2 : S2 = 6927)
  (h4 : S4 = 7230)
  (h5 : S5 = 6562)
  (h6 : S6 = 4891)
  (havg : average_sale S1 S2 S3 S4 S5 S6 = 6500) :
  S3 = 6855 := 
sorry

end sale_in_third_month_l134_134332


namespace quadratic_b_value_l134_134206

theorem quadratic_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + b * x - 12 < 0 ↔ x < 3 ∨ x > 7) → b = 10 :=
by 
  sorry

end quadratic_b_value_l134_134206


namespace mean_equality_l134_134601

theorem mean_equality (y : ℝ) :
  ((3 + 7 + 11 + 15) / 4 = (10 + 14 + y) / 3) → y = 3 :=
by
  sorry

end mean_equality_l134_134601


namespace sqrt27_add_sqrt3_sub_sqrt12_eq_2sqrt3_sqrt24_inverse_add_abs_sqrt6_sub_3_add_half_inverse_sub_2016_pow0_eq_4_min_11sqrt6_div_12_sqrt3_add_sqrt2_squared_sub_sqrt3_sub_sqrt2_squared_eq_4sqrt6_l134_134487

-- Proof 1: 
theorem sqrt27_add_sqrt3_sub_sqrt12_eq_2sqrt3 :
  Real.sqrt 27 + Real.sqrt 3 - Real.sqrt 12 = 2 * Real.sqrt 3 :=
by
  sorry

-- Proof 2:
theorem sqrt24_inverse_add_abs_sqrt6_sub_3_add_half_inverse_sub_2016_pow0_eq_4_min_11sqrt6_div_12 :
  1 / Real.sqrt 24 + abs (Real.sqrt 6 - 3) + (1 / 2)⁻¹ - 2016 ^ 0 = 4 - 11 * Real.sqrt 6 / 12 :=
by
  sorry

-- Proof 3:
theorem sqrt3_add_sqrt2_squared_sub_sqrt3_sub_sqrt2_squared_eq_4sqrt6 :
  (Real.sqrt 3 + Real.sqrt 2) ^ 2 - (Real.sqrt 3 - Real.sqrt 2) ^ 2 = 4 * Real.sqrt 6 :=
by
  sorry

end sqrt27_add_sqrt3_sub_sqrt12_eq_2sqrt3_sqrt24_inverse_add_abs_sqrt6_sub_3_add_half_inverse_sub_2016_pow0_eq_4_min_11sqrt6_div_12_sqrt3_add_sqrt2_squared_sub_sqrt3_sub_sqrt2_squared_eq_4sqrt6_l134_134487


namespace trigonometric_identity_l134_134970

theorem trigonometric_identity (α : ℝ) (h : Real.tan (Real.pi + α) = 2) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) / (Real.sin (Real.pi + α) - Real.cos (Real.pi - α)) = 3 :=
by
  sorry

end trigonometric_identity_l134_134970


namespace cos_half_pi_plus_alpha_l134_134032

open Real

noncomputable def alpha : ℝ := sorry

theorem cos_half_pi_plus_alpha :
  let a := (1 / 3, tan alpha)
  let b := (cos alpha, 1)
  ((1 / 3) / (cos alpha) = (tan alpha) / 1) →
  cos (pi / 2 + alpha) = -1 / 3 :=
by
  intros
  sorry

end cos_half_pi_plus_alpha_l134_134032


namespace no_natural_has_2021_trailing_zeros_l134_134830

-- Define the function f(n) which computes the number of trailing zeros in n!
def trailing_zeros (n : ℕ) : ℕ :=
  let rec aux (k : ℕ) (acc : ℕ) : ℕ :=
    if k > n then acc
    else aux (k * 5) (acc + n / k)
  aux 5 0

-- Prove that there does not exist a natural number n such that the number of trailing zeros in n! is exactly 2021
theorem no_natural_has_2021_trailing_zeros :
  ¬ ∃ n : ℕ, trailing_zeros n = 2021 :=
by {
  intro h,
  sorry
}

end no_natural_has_2021_trailing_zeros_l134_134830


namespace grazing_months_of_B_l134_134329

variable (A_cows A_months C_cows C_months D_cows D_months A_rent total_rent : ℕ)
variable (B_cows x : ℕ)

theorem grazing_months_of_B
  (hA_cows : A_cows = 24)
  (hA_months : A_months = 3)
  (hC_cows : C_cows = 35)
  (hC_months : C_months = 4)
  (hD_cows : D_cows = 21)
  (hD_months : D_months = 3)
  (hA_rent : A_rent = 1440)
  (htotal_rent : total_rent = 6500)
  (hB_cows : B_cows = 10) :
  x = 5 := 
sorry

end grazing_months_of_B_l134_134329


namespace highest_average_speed_interval_l134_134917

theorem highest_average_speed_interval
  (d : ℕ → ℕ)
  (h0 : d 0 = 45)        -- Distance from 0 to 30 minutes
  (h1 : d 1 = 135)       -- Distance from 30 to 60 minutes
  (h2 : d 2 = 255)       -- Distance from 60 to 90 minutes
  (h3 : d 3 = 325) :     -- Distance from 90 to 120 minutes
  (1 / 2) * ((d 2 - d 1 : ℕ) : ℝ) > 
  max ((1 / 2) * ((d 1 - d 0 : ℕ) : ℝ)) 
      (max ((1 / 2) * ((d 3 - d 2 : ℕ) : ℝ))
          ((1 / 2) * ((d 3 - d 1 : ℕ) : ℝ))) :=
by
  sorry

end highest_average_speed_interval_l134_134917


namespace quadratic_rewriting_l134_134890

theorem quadratic_rewriting (b n : ℝ) (h₁ : 0 < n)
  (h₂ : ∀ x : ℝ, x^2 + b*x + 72 = (x + n)^2 + 20) :
  b = 4 * Real.sqrt 13 :=
by
  sorry

end quadratic_rewriting_l134_134890


namespace trapezoid_area_l134_134544

/-- Given that the area of the outer square is 36 square units and the area of the inner square is 
4 square units, the area of one of the four congruent trapezoids formed between the squares is 8 
square units. -/
theorem trapezoid_area (outer_square_area inner_square_area : ℕ) 
  (h_outer : outer_square_area = 36)
  (h_inner : inner_square_area = 4) : 
  (outer_square_area - inner_square_area) / 4 = 8 :=
by sorry

end trapezoid_area_l134_134544


namespace part_I_solution_set_part_II_min_value_l134_134264

-- Define the function f
def f (x a : ℝ) := 2*|x + 1| - |x - a|

-- Part I: Prove the solution set of f(x) ≥ 0 when a = 2
theorem part_I_solution_set (x : ℝ) :
  f x 2 ≥ 0 ↔ x ≤ -4 ∨ x ≥ 0 :=
sorry

-- Define the function g
def g (x a : ℝ) := f x a + 3*|x - a|

-- Part II: Prove the minimum value of m + n given t = 4 when a = 1
theorem part_II_min_value (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x, g x 1 ≥ 4) → (2/m + 1/(2*n) = 4) → m + n = 9/8 :=
sorry

end part_I_solution_set_part_II_min_value_l134_134264


namespace solve_for_y_l134_134590

theorem solve_for_y {y : ℕ} (h : (1000 : ℝ) = (10 : ℝ)^3) : (1000 : ℝ)^4 = (10 : ℝ)^y ↔ y = 12 :=
by
  sorry

end solve_for_y_l134_134590


namespace total_weight_of_remaining_macaroons_l134_134406

def total_weight_remaining_macaroons (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (bags : ℕ) (bags_eaten : ℕ) : ℕ :=
  let macaroons_per_bag := total_macaroons / bags
  let remaining_macaroons := total_macaroons - macaroons_per_bag * bags_eaten
  remaining_macaroons * weight_per_macaroon

theorem total_weight_of_remaining_macaroons
  (total_macaroons : ℕ)
  (weight_per_macaroon : ℕ)
  (bags : ℕ)
  (bags_eaten : ℕ)
  (h1 : total_macaroons = 12)
  (h2 : weight_per_macaroon = 5)
  (h3 : bags = 4)
  (h4 : bags_eaten = 1)
  : total_weight_remaining_macaroons total_macaroons weight_per_macaroon bags bags_eaten = 45 := by
  sorry

end total_weight_of_remaining_macaroons_l134_134406


namespace f_of_pi_over_6_l134_134705

noncomputable def f (ω : ℝ) (ϕ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + ϕ)

theorem f_of_pi_over_6 (ω ϕ : ℝ) (h₀ : ω > 0) (h₁ : -Real.pi / 2 ≤ ϕ) (h₂ : ϕ < Real.pi / 2) 
  (transformed : ∀ x, f ω ϕ (x/2 - Real.pi/6) = Real.sin x) :
  f ω ϕ (Real.pi / 6) = Real.sqrt 2 / 2 :=
by
  sorry

end f_of_pi_over_6_l134_134705


namespace num_divisible_by_7_in_range_l134_134099

theorem num_divisible_by_7_in_range (n : ℤ) (h : 1 ≤ n ∧ n ≤ 2015)
    : (∃ k, 1 ≤ k ∧ k ≤ 335 ∧ 3 ^ (6 * k) + (6 * k) ^ 3 ≡ 0 [MOD 7]) :=
sorry

end num_divisible_by_7_in_range_l134_134099


namespace max_value_quadratic_l134_134022

noncomputable def quadratic (x : ℝ) : ℝ := -3 * (x - 2)^2 - 3

theorem max_value_quadratic : ∀ x : ℝ, quadratic x ≤ -3 ∧ (∀ y : ℝ, quadratic y = -3 → (∀ z : ℝ, quadratic z ≤ quadratic y)) :=
by
  sorry

end max_value_quadratic_l134_134022


namespace part1_part2_l134_134382

theorem part1 (a : ℝ) (h1 : ∀ x y, y = a * x + 1 → 3 * x^2 - y^2 = 1) (h2 : ∃ x1 y1 x2 y2 : ℝ, y1 = a * x1 + 1 ∧ y2 = a * x2 + 1 ∧ 3 * x1 * x1 - y1 * y1 = 1 ∧ 3 * x2 * x2 - y2 * y2 = 1 ∧ x1 * x2 + (a * x1 + 1) * (a * x2 + 1) = 0) : a = 1 ∨ a = -1 := sorry

theorem part2 (h : ∀ x y, y = a * x + 1 → 3 * x^2 - y^2 = 1) (a : ℝ) (h2 : ∃ x1 y1 x2 y2 : ℝ, y1 = a * x1 + 1 ∧ y2 = a * x2 + 1 ∧ 3 * x1 * x1 - y1 * y1 = 1 ∧ 3 * x2 * x2 - y2 * y2 = 1 ∧ (y1 + y2) / 2 = (1 / 2) * (x1 + x2) / 2 ∧ (y1 - y2) / (x1 - x2) = -2) : false := sorry

end part1_part2_l134_134382


namespace ratio_lateral_surface_area_to_surface_area_l134_134727

theorem ratio_lateral_surface_area_to_surface_area (r : ℝ) (h : ℝ) (V_sphere V_cone A_cone A_sphere : ℝ)
    (h_eq : h = r)
    (V_sphere_eq : V_sphere = (4 / 3) * Real.pi * r^3)
    (V_cone_eq : V_cone = (1 / 3) * Real.pi * (2 * r)^2 * h)
    (V_eq : V_sphere = V_cone)
    (A_cone_eq : A_cone = 2 * Real.sqrt 5 * Real.pi * r^2)
    (A_sphere_eq : A_sphere = 4 * Real.pi * r^2) :
    A_cone / A_sphere = Real.sqrt 5 / 2 := by
  sorry

end ratio_lateral_surface_area_to_surface_area_l134_134727


namespace acrobat_eq_two_lambs_l134_134640

variables (ACROBAT DOG BARREL SPOOL LAMB : ℝ)

axiom acrobat_dog_eq_two_barrels : ACROBAT + DOG = 2 * BARREL
axiom dog_eq_two_spools : DOG = 2 * SPOOL
axiom lamb_spool_eq_barrel : LAMB + SPOOL = BARREL

theorem acrobat_eq_two_lambs : ACROBAT = 2 * LAMB :=
by
  sorry

end acrobat_eq_two_lambs_l134_134640


namespace veranda_width_l134_134196

def room_length : ℕ := 17
def room_width : ℕ := 12
def veranda_area : ℤ := 132

theorem veranda_width :
  ∃ (w : ℝ), (17 + 2 * w) * (12 + 2 * w) - 17 * 12 = 132 ∧ w = 2 :=
by
  use 2
  sorry

end veranda_width_l134_134196


namespace tom_finishes_in_four_hours_l134_134743

noncomputable def maryMowingRate := 1 / 3
noncomputable def tomMowingRate := 1 / 6
noncomputable def timeMaryMows := 1
noncomputable def remainingLawn := 1 - (timeMaryMows * maryMowingRate)

theorem tom_finishes_in_four_hours :
  remainingLawn / tomMowingRate = 4 :=
by sorry

end tom_finishes_in_four_hours_l134_134743


namespace balloon_count_l134_134853

-- Conditions
def Fred_balloons : ℕ := 5
def Sam_balloons : ℕ := 6
def Mary_balloons : ℕ := 7
def total_balloons : ℕ := 18

-- Proof statement
theorem balloon_count :
  Fred_balloons + Sam_balloons + Mary_balloons = total_balloons :=
by
  exact Nat.add_assoc 5 6 7 ▸ rfl

end balloon_count_l134_134853


namespace snow_globes_in_box_l134_134496

theorem snow_globes_in_box (S : ℕ) 
  (h1 : ∀ (box_decorations : ℕ), box_decorations = 4 + 1 + S)
  (h2 : ∀ (num_boxes : ℕ), num_boxes = 12)
  (h3 : ∀ (total_decorations : ℕ), total_decorations = 120) :
  S = 5 :=
by
  sorry

end snow_globes_in_box_l134_134496


namespace initial_amount_l134_134797

theorem initial_amount (P R : ℝ) (h1 : 956 = P * (1 + (3 * R) / 100)) (h2 : 1052 = P * (1 + (3 * (R + 4)) / 100)) : P = 800 := 
by
  -- We would provide the proof steps here normally
  sorry

end initial_amount_l134_134797


namespace maximal_s_value_l134_134771

noncomputable def max_tiles_sum (a b c : ℕ) : ℕ := a + c

theorem maximal_s_value :
  ∃ s : ℕ, 
    ∃ a b c : ℕ, 
      4 * a + 4 * c + 5 * b = 3986000 ∧ 
      s = max_tiles_sum a b c ∧ 
      s = 996500 := 
    sorry

end maximal_s_value_l134_134771


namespace ratio_is_three_l134_134599

-- Define the conditions
def area_of_garden : ℕ := 588
def width_of_garden : ℕ := 14
def length_of_garden : ℕ := area_of_garden / width_of_garden

-- Define the ratio
def ratio_length_to_width := length_of_garden / width_of_garden

-- The proof statement
theorem ratio_is_three : ratio_length_to_width = 3 := 
by sorry

end ratio_is_three_l134_134599


namespace fourth_square_state_l134_134566

inductive Shape
| Circle
| Triangle
| LineSegment
| Square

inductive Position
| TopLeft
| TopRight
| BottomLeft
| BottomRight

structure SquareState where
  circle : Position
  triangle : Position
  line_segment_parallel_to : Bool -- True = Top & Bottom; False = Left & Right
  square : Position

def move_counterclockwise : Position → Position
| Position.TopLeft => Position.BottomLeft
| Position.BottomLeft => Position.BottomRight
| Position.BottomRight => Position.TopRight
| Position.TopRight => Position.TopLeft

def update_square_states (s1 s2 s3 : SquareState) : Prop :=
  move_counterclockwise s1.circle = s2.circle ∧
  move_counterclockwise s2.circle = s3.circle ∧
  move_counterclockwise s1.triangle = s2.triangle ∧
  move_counterclockwise s2.triangle = s3.triangle ∧
  s1.line_segment_parallel_to = !s2.line_segment_parallel_to ∧
  s2.line_segment_parallel_to = !s3.line_segment_parallel_to ∧
  move_counterclockwise s1.square = s2.square ∧
  move_counterclockwise s2.square = s3.square

theorem fourth_square_state (s1 s2 s3 s4 : SquareState) (h : update_square_states s1 s2 s3) :
  s4.circle = move_counterclockwise s3.circle ∧
  s4.triangle = move_counterclockwise s3.triangle ∧
  s4.line_segment_parallel_to = !s3.line_segment_parallel_to ∧
  s4.square = move_counterclockwise s3.square :=
sorry

end fourth_square_state_l134_134566


namespace gcd_six_digit_repeat_l134_134121

theorem gcd_six_digit_repeat (n : ℕ) (h1 : 100 ≤ n) (h2 : n ≤ 999) : 
  ∀ m : ℕ, m = 1001 * n → (gcd m 1001 = 1001) :=
by
  sorry

end gcd_six_digit_repeat_l134_134121


namespace does_not_represent_right_triangle_l134_134281

/-- In triangle ABC, the sides opposite to angles A, B, and C are a, b, and c respectively. Given:
  - a:b:c = 6:8:10
  - ∠A:∠B:∠C = 1:1:3
  - a^2 + c^2 = b^2
  - ∠A + ∠B = ∠C

Prove that the condition ∠A:∠B:∠C = 1:1:3 does not represent a right triangle ABC. -/
theorem does_not_represent_right_triangle
  (a b c : ℝ) (A B C : ℝ)
  (h1 : a / b = 6 / 8 ∧ b / c = 8 / 10)
  (h2 : A / B = 1 / 1 ∧ B / C = 1 / 3)
  (h3 : a^2 + c^2 = b^2)
  (h4 : A + B = C) :
  ¬ (B = 90) :=
sorry

end does_not_represent_right_triangle_l134_134281


namespace Vasya_mushrooms_l134_134770

def isThreeDigit (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

def digitsSum (n : ℕ) : ℕ := (n / 100) + ((n % 100) / 10) + (n % 10)

theorem Vasya_mushrooms :
  ∃ n : ℕ, isThreeDigit n ∧ digitsSum n = 14 ∧ n = 950 := 
by
  sorry

end Vasya_mushrooms_l134_134770


namespace K1K2_eq_one_over_four_l134_134947

theorem K1K2_eq_one_over_four
  (K1 : ℝ) (hK1 : K1 ≠ 0)
  (K2 : ℝ)
  (x1 y1 x2 y2 : ℝ)
  (hx1y1 : x1^2 - 4 * y1^2 = 4)
  (hx2y2 : x2^2 - 4 * y2^2 = 4)
  (hx0 : x0 = (x1 + x2) / 2)
  (hy0 : y0 = (y1 + y2) / 2)
  (K1_eq : K1 = (y1 - y2) / (x1 - x2))
  (K2_eq : K2 = y0 / x0) :
  K1 * K2 = 1 / 4 :=
sorry

end K1K2_eq_one_over_four_l134_134947


namespace no_integer_roots_l134_134189

-- Define a predicate for checking if a number is odd
def is_odd (a : ℤ) : Prop := a % 2 = 1

-- Define the polynomial with integer coefficients
def P (a : list ℤ) (x : ℤ) : ℤ := 
  (a.zipWithIndex.map (λ (ai, i), ai * x ^ i)).sum

-- The main theorem stating the polynomial does not have integer roots
theorem no_integer_roots (a : list ℤ) (h0 : is_odd (P a 0)) (h1 : is_odd (P a 1)) :
  ∀ r : ℤ, P a r ≠ 0 := 
sorry

end no_integer_roots_l134_134189


namespace rebus_solution_l134_134001

theorem rebus_solution :
  ∃ (A B C : ℕ), A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    (A*100 + B*10 + A) + (A*100 + B*10 + C) + (A*100 + C*10 + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by {
  sorry
}

end rebus_solution_l134_134001


namespace cylinder_base_area_l134_134798

-- Definitions: Adding variables and hypotheses based on the problem statement.
variable (A_c A_r : ℝ) -- Base areas of the cylinder and the rectangular prism
variable (h1 : 8 * A_c = 6 * A_r) -- Condition from the rise in water levels
variable (h2 : A_c + A_r = 98) -- Sum of the base areas
variable (h3 : A_c / A_r = 3 / 4) -- Ratio of the base areas

-- Statement: The goal is to prove that the base area of the cylinder is 42.
theorem cylinder_base_area : A_c = 42 :=
by
  sorry

end cylinder_base_area_l134_134798


namespace addition_problem_base6_l134_134848

theorem addition_problem_base6 (X Y : ℕ) (h1 : Y + 3 = X) (h2 : X + 2 = 7) : X + Y = 7 :=
by
  sorry

end addition_problem_base6_l134_134848


namespace trains_cross_time_l134_134642

theorem trains_cross_time (length1 length2 : ℕ) (time1 time2 : ℕ) 
  (speed1 speed2 relative_speed total_length : ℚ) 
  (h1 : length1 = 120) (h2 : length2 = 150) 
  (h3 : time1 = 10) (h4 : time2 = 15) 
  (h5 : speed1 = length1 / time1) (h6 : speed2 = length2 / time2) 
  (h7 : relative_speed = speed1 - speed2) 
  (h8 : total_length = length1 + length2) : 
  (total_length / relative_speed = 135) := 
by sorry

end trains_cross_time_l134_134642


namespace initial_average_weight_l134_134443

theorem initial_average_weight
  (A : ℝ)
  (h : 30 * 27.4 - 10 = 29 * A) : 
  A = 28 := 
by
  sorry

end initial_average_weight_l134_134443


namespace juggling_contest_l134_134094

theorem juggling_contest (B : ℕ) (rot_baseball : ℕ := 80)
    (rot_per_apple : ℕ := 101) (num_apples : ℕ := 4)
    (winner_rotations : ℕ := 404) :
    (num_apples * rot_per_apple = winner_rotations) :=
by
  sorry

end juggling_contest_l134_134094


namespace system1_solution_system2_solution_l134_134072

theorem system1_solution (x y : ℤ) (h1 : x - y = 2) (h2 : x + 1 = 2 * (y - 1)) :
  x = 7 ∧ y = 5 :=
sorry

theorem system2_solution (x y : ℤ) (h1 : 2 * x + 3 * y = 1) (h2 : (y - 1) * 3 = (x - 2) * 4) :
  x = 1 ∧ y = -1 / 3 :=
sorry

end system1_solution_system2_solution_l134_134072


namespace solve_for_t_l134_134034

theorem solve_for_t (p t : ℝ) (h1 : 5 = p * 3^t) (h2 : 45 = p * 9^t) : t = 2 :=
by
  sorry

end solve_for_t_l134_134034


namespace arithmetic_sequence_sum_l134_134278

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Definition of the sum of the first n terms
def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

-- Problem statement in Lean 4
theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (S : ℕ → ℤ)
  (h1 : arithmetic_sequence a)
  (h2 : sum_first_n_terms a S)
  (h3 : S 9 = a 4 + a 5 + a 6 + 66) :
  a 2 + a 8 = 22 := by
  sorry

end arithmetic_sequence_sum_l134_134278


namespace find_value_l134_134296

variable (N : ℝ)

def condition : Prop := (1 / 4) * (1 / 3) * (2 / 5) * N = 16

theorem find_value (h : condition N) : (1 / 3) * (2 / 5) * N = 64 :=
sorry

end find_value_l134_134296


namespace f_bounds_l134_134607

-- Define the function f with the given properties
def f : ℝ → ℝ :=
sorry 

-- Specify the conditions on f
axiom f_0 : f 0 = 0
axiom f_1 : f 1 = 1
axiom f_ratio (x y z : ℝ) (h1 : 0 ≤ x) (h2 : x < y) (h3 : y < z) (h4 : z ≤ 1) 
  (h5 : z - y = y - x) : 1/2 ≤ (f z - f y) / (f y - f x) ∧ (f z - f y) / (f y - f x) ≤ 2

-- State the theorem to be proven
theorem f_bounds : 1 / 7 ≤ f (1 / 3) ∧ f (1 / 3) ≤ 4 / 7 :=
sorry

end f_bounds_l134_134607


namespace find_x_l134_134562

theorem find_x (x : ℚ) (h : (3 + 1 / (2 + 1 / (3 + 3 / (4 + x)))) = 225 / 68) : 
  x = -50 / 19 := 
sorry

end find_x_l134_134562


namespace int_modulo_l134_134317

theorem int_modulo (n : ℤ) (h1 : 0 ≤ n) (h2 : n < 17) (h3 : 38574 ≡ n [ZMOD 17]) : n = 1 :=
by
  sorry

end int_modulo_l134_134317


namespace find_quantities_of_raib_ornaments_and_pendants_l134_134351

theorem find_quantities_of_raib_ornaments_and_pendants (x y : ℕ)
  (h1 : x + y = 90)
  (h2 : 40 * x + 25 * y = 2850) :
  x = 40 ∧ y = 50 :=
sorry

end find_quantities_of_raib_ornaments_and_pendants_l134_134351


namespace complement_of_angle_l134_134528

theorem complement_of_angle (A : ℝ) (hA : A = 35) : 180 - A = 145 := by
  sorry

end complement_of_angle_l134_134528


namespace one_fourths_in_five_eighths_l134_134526

theorem one_fourths_in_five_eighths : (5/8 : ℚ) / (1/4) = (5/2 : ℚ) := 
by
  -- Placeholder for the proof
  sorry

end one_fourths_in_five_eighths_l134_134526


namespace problem_statement_l134_134647

theorem problem_statement :
  102^3 + 3 * 102^2 + 3 * 102 + 1 = 1092727 :=
  by sorry

end problem_statement_l134_134647


namespace hiring_manager_acceptance_l134_134078

theorem hiring_manager_acceptance 
    (average_age : ℤ) (std_dev : ℤ) (num_ages : ℤ)
    (applicant_ages_are_int : ∀ (x : ℤ), x ≥ (average_age - std_dev) ∧ x ≤ (average_age + std_dev)) :
    (∃ k : ℤ, (average_age + k * std_dev) - (average_age - k * std_dev) + 1 = num_ages) → k = 1 :=
by 
  intros h
  sorry

end hiring_manager_acceptance_l134_134078


namespace coffee_shop_distance_l134_134216

theorem coffee_shop_distance (resort_distance mall_distance : ℝ) 
  (coffee_dist : ℝ)
  (h_resort_distance : resort_distance = 400) 
  (h_mall_distance : mall_distance = 700)
  (h_equidistant : ∀ S, (S - resort_distance) ^ 2 + resort_distance ^ 2 = S ^ 2 ∧ 
  (mall_distance - S) ^ 2 + resort_distance ^ 2 = S ^ 2 → coffee_dist = S):
  coffee_dist = 464 := 
sorry

end coffee_shop_distance_l134_134216


namespace find_f_neg_9_over_2_l134_134257

noncomputable def f : ℝ → ℝ
| x => if 0 ≤ x ∧ x ≤ 1 then 2^x else sorry

theorem find_f_neg_9_over_2
  (hf_even : ∀ x : ℝ, f (-x) = f x)
  (hf_periodic : ∀ x : ℝ, f (x + 2) = f x)
  (hf_definition : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → f x = 2^x) :
  f (-9 / 2) = Real.sqrt 2 := by
  sorry

end find_f_neg_9_over_2_l134_134257


namespace geometric_sequence_k_eq_6_l134_134994

theorem geometric_sequence_k_eq_6 
  (a : ℕ → ℝ) (q : ℝ) (k : ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a n = a 1 * q ^ (n - 1))
  (h3 : q ≠ 1)
  (h4 : q ≠ -1)
  (h5 : a k = a 2 * a 5) :
  k = 6 :=
sorry

end geometric_sequence_k_eq_6_l134_134994


namespace circle_center_sum_l134_134248

theorem circle_center_sum (h k : ℝ) :
  (∀ x y : ℝ, (x - h) ^ 2 + (y - k) ^ 2 = x ^ 2 + y ^ 2 - 6 * x - 8 * y + 38) → h + k = 7 :=
by sorry

end circle_center_sum_l134_134248


namespace smallest_integer_solution_l134_134710

theorem smallest_integer_solution (y : ℤ) (h : 7 - 3 * y < 25) : y ≥ -5 :=
by {
  sorry
}

end smallest_integer_solution_l134_134710


namespace pow_sub_nat_ge_seven_l134_134558

open Nat

theorem pow_sub_nat_ge_seven
  (m n : ℕ) 
  (h1 : m > 1)
  (h2 : 2^(2 * m + 1) - n^2 ≥ 0) : 
  2^(2 * m + 1) - n^2 ≥ 7 :=
sorry

end pow_sub_nat_ge_seven_l134_134558


namespace find_line_through_midpoint_of_hyperbola_l134_134946

theorem find_line_through_midpoint_of_hyperbola
  (x1 y1 x2 y2 : ℝ)
  (P : ℝ × ℝ := (4, 1))
  (A : ℝ × ℝ := (x1, y1))
  (B : ℝ × ℝ := (x2, y2))
  (H_midpoint : P = ((x1 + x2) / 2, (y1 + y2) / 2))
  (H_hyperbola_A : (x1^2 / 4 - y1^2 = 1))
  (H_hyperbola_B : (x2^2 / 4 - y2^2 = 1)) :
  ∃ m b : ℝ, (m = 1) ∧ (b = 3) ∧ (∀ x y : ℝ, y = m * x + b → x - y - 3 = 0) := by
  sorry

end find_line_through_midpoint_of_hyperbola_l134_134946


namespace range_of_x0_l134_134697

noncomputable def point_on_circle_and_line (x0 : ℝ) (y0 : ℝ) : Prop :=
(x0^2 + y0^2 = 1) ∧ (3 * x0 + 2 * y0 = 4)

theorem range_of_x0 
  (x0 : ℝ) (y0 : ℝ) 
  (h1 : 3 * x0 + 2 * y0 = 4)
  (h2 : ∃ A B : ℝ × ℝ, (A.1^2 + A.2^2 = 1) ∧ (B.1^2 + B.2^2 = 1) ∧ (A ≠ B) ∧ (A + B = (x0, y0))) :
  0 < x0 ∧ x0 < 24 / 13 :=
sorry

end range_of_x0_l134_134697


namespace calc_ratio_of_d_to_s_l134_134881

theorem calc_ratio_of_d_to_s {n s d : ℝ} (h_n_eq_24 : n = 24)
    (h_tiles_area_64_pct : (576 * s^2) = 0.64 * (n * s + d)^2) : 
    d / s = 6 / 25 :=
by
  sorry

end calc_ratio_of_d_to_s_l134_134881


namespace sqrt_expression_meaningful_l134_134725

theorem sqrt_expression_meaningful (x : ℝ) : (2 * x - 4 ≥ 0) ↔ (x ≥ 2) :=
by
  -- Proof will be skipped
  sorry

end sqrt_expression_meaningful_l134_134725


namespace find_original_cost_price_l134_134229

variable (C : ℝ)

-- Conditions
def first_discount (C : ℝ) : ℝ := 0.95 * C
def second_discount (C : ℝ) : ℝ := 0.9215 * C
def loss_price (C : ℝ) : ℝ := 0.90 * C
def gain_price_before_tax (C : ℝ) : ℝ := 1.08 * C
def gain_price_after_tax (C : ℝ) : ℝ := 1.20 * C

-- Prove that original cost price is 1800
theorem find_original_cost_price 
  (h1 : first_discount C = loss_price C)
  (h2 : gain_price_after_tax C - loss_price C = 540) : 
  C = 1800 := 
sorry

end find_original_cost_price_l134_134229


namespace interval_of_monotonic_increase_l134_134083

noncomputable def f (x : ℝ) : ℝ := Real.logb (1/2) (6 + x - x^2)

theorem interval_of_monotonic_increase :
  {x : ℝ | -2 < x ∧ x < 3} → x ∈ Set.Ioc (1/2) 3 :=
by
  sorry

end interval_of_monotonic_increase_l134_134083


namespace farmer_animals_l134_134112

theorem farmer_animals : 
  ∃ g s : ℕ, 
    35 * g + 40 * s = 2000 ∧ 
    g = 2 * s ∧ 
    (0 < g ∧ 0 < s) ∧ 
    g = 36 ∧ s = 18 := 
by 
  sorry

end farmer_animals_l134_134112


namespace number_of_subsets_l134_134308

theorem number_of_subsets (x y : Type) :  ∃ s : Finset (Finset Type), s.card = 4 := 
sorry

end number_of_subsets_l134_134308


namespace adam_books_l134_134118

theorem adam_books (before_books total_shelves books_per_shelf after_books leftover_books bought_books : ℕ)
  (h_before: before_books = 56)
  (h_shelves: total_shelves = 4)
  (h_books_per_shelf: books_per_shelf = 20)
  (h_leftover: leftover_books = 2)
  (h_after: after_books = (total_shelves * books_per_shelf) + leftover_books)
  (h_difference: bought_books = after_books - before_books) :
  bought_books = 26 :=
by
  sorry

end adam_books_l134_134118


namespace problem_statement_l134_134015

theorem problem_statement (x : ℝ) (h : x^2 + 3 * x - 1 = 0) : x^3 + 5 * x^2 + 5 * x + 18 = 20 :=
by
  sorry

end problem_statement_l134_134015


namespace find_m_l134_134980

variable {m : ℝ}

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (m : ℝ) : ℝ × ℝ := (m, -1)
def vector_diff (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_m (hm: dot_product vector_a (vector_diff vector_a (vector_b m)) = 0) : m = 3 :=
  by
  sorry

end find_m_l134_134980


namespace apples_number_l134_134609

def num_apples (A O B : ℕ) : Prop :=
  A = O + 27 ∧ O = B + 11 ∧ A + O + B = 301 → A = 122

theorem apples_number (A O B : ℕ) : num_apples A O B := by
  sorry

end apples_number_l134_134609


namespace greatest_divisor_four_consecutive_squared_l134_134646

theorem greatest_divisor_four_consecutive_squared :
  ∀ (n: ℕ), ∃ m: ℕ, (∀ (n: ℕ), m ∣ (n * (n + 1) * (n + 2) * (n + 3))^2) ∧ m = 144 := 
sorry

end greatest_divisor_four_consecutive_squared_l134_134646


namespace cos_difference_simplification_l134_134424

theorem cos_difference_simplification :
  let x := Real.cos (20 * Real.pi / 180)
  let y := Real.cos (40 * Real.pi / 180)
  (y = 2 * x^2 - 1) →
  (x = 1 - 2 * y^2) →
  x - y = 1 / 2 :=
by
  intros x y h1 h2
  sorry

end cos_difference_simplification_l134_134424


namespace smallest_possible_value_l134_134270

theorem smallest_possible_value (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (⌊(a + b + c) / d⌋ + ⌊(a + b + d) / c⌋ + ⌊(a + c + d) / b⌋ + ⌊(b + c + d) / a⌋) ≥ 8 :=
sorry

end smallest_possible_value_l134_134270


namespace closed_polygon_inequality_l134_134039

noncomputable def length_eq (A B C D : ℝ × ℝ × ℝ) (l : ℝ) : Prop :=
  dist A B = l ∧ dist B C = l ∧ dist C D = l ∧ dist D A = l

theorem closed_polygon_inequality 
  (A B C D P : ℝ × ℝ × ℝ) (l : ℝ)
  (hABCD : length_eq A B C D l) :
  dist P A < dist P B + dist P C + dist P D :=
sorry

end closed_polygon_inequality_l134_134039


namespace min_value_is_2_sqrt_2_l134_134692

noncomputable def min_value (a b : ℝ) : ℝ :=
  a^2 + b^2 / (a - b)

theorem min_value_is_2_sqrt_2 (a b : ℝ) (h1 : a > b) (h2 : a > 0) (h3 : a * b = 1) : 
  min_value a b = 2 * Real.sqrt 2 := 
sorry

end min_value_is_2_sqrt_2_l134_134692


namespace intersection_A_B_l134_134978

-- Conditions
def A : Set ℝ := {1, 2, 0.5}
def B : Set ℝ := {y | ∃ x, x ∈ A ∧ y = x^2}

-- Theorem statement
theorem intersection_A_B :
  A ∩ B = {1} :=
sorry

end intersection_A_B_l134_134978


namespace range_of_a1_l134_134993

noncomputable def geometric_sequence_cond (a_1 : ℝ) (q : ℝ) : Prop :=
  ∃ (S_n : ℕ → ℝ), (S_n = λ n, a_1 * (1 - q^n) / (1 - q)) ∧ (tendsto S_n at_top (𝓝 (1 / a_1)))

theorem range_of_a1 {a_1 q : ℝ} (h1 : a_1 > 1) (h2 : abs q < 1)
  (h3 : geometric_sequence_cond a_1 q) : 1 < a_1 ∧ a_1 < sqrt 2 :=
by sorry

end range_of_a1_l134_134993


namespace part_a_part_b_l134_134030

noncomputable def sequence (n : ℕ) : ℝ := Real.cos (Real.pi * 10 ^ n / 180)

theorem part_a : (sequence 100) > 0 := 
by {
  sorry
}

theorem part_b : abs (sequence 100) < 0.18 :=
by {
  sorry
}

end part_a_part_b_l134_134030


namespace total_toys_l134_134816

theorem total_toys (K A L : ℕ) (h1 : A = K + 30) (h2 : L = 2 * K) (h3 : K + A = 160) : 
    K + A + L = 290 :=
by
  sorry

end total_toys_l134_134816


namespace seats_not_occupied_l134_134837

theorem seats_not_occupied (seats_per_row : ℕ) (rows : ℕ) (fraction_allowed : ℚ) (total_seats : ℕ) (allowed_seats_per_row : ℕ) (allowed_total : ℕ) (unoccupied_seats : ℕ) :
  seats_per_row = 8 →
  rows = 12 →
  fraction_allowed = 3 / 4 →
  total_seats = seats_per_row * rows →
  allowed_seats_per_row = seats_per_row * fraction_allowed →
  allowed_total = allowed_seats_per_row * rows →
  unoccupied_seats = total_seats - allowed_total →
  unoccupied_seats = 24 :=
by sorry

end seats_not_occupied_l134_134837


namespace no_intersection_of_lines_l134_134799

theorem no_intersection_of_lines :
  ¬ ∃ (s v : ℝ) (x y : ℝ),
    (x = 1 - 2 * s ∧ y = 4 + 6 * s) ∧
    (x = 3 - v ∧ y = 10 + 3 * v) :=
by {
  sorry
}

end no_intersection_of_lines_l134_134799


namespace green_notebook_cost_l134_134063

def total_cost : ℕ := 45
def black_cost : ℕ := 15
def pink_cost : ℕ := 10
def num_green_notebooks : ℕ := 2

theorem green_notebook_cost :
  (total_cost - (black_cost + pink_cost)) / num_green_notebooks = 10 :=
by
  sorry

end green_notebook_cost_l134_134063


namespace parabola_vertex_coordinates_l134_134446

theorem parabola_vertex_coordinates :
  (∃ x : ℝ, (λ x, x^2 - 2) = (0, -2)) :=
sorry

end parabola_vertex_coordinates_l134_134446


namespace polynomial_division_result_q_neg1_r_1_sum_l134_134186

noncomputable def f (x : ℝ) : ℝ := 3 * x^4 + 5 * x^3 - 4 * x^2 + 2 * x + 1
noncomputable def d (x : ℝ) : ℝ := x^2 + 2 * x - 3
noncomputable def q (x : ℝ) : ℝ := 3 * x^2 + x
noncomputable def r (x : ℝ) : ℝ := 7 * x + 4

theorem polynomial_division_result : f (-1) = q (-1) * d (-1) + r (-1)
  ∧ f 1 = q 1 * d 1 + r 1 :=
by sorry

theorem q_neg1_r_1_sum : (q (-1) + r 1) = 13 :=
by sorry

end polynomial_division_result_q_neg1_r_1_sum_l134_134186


namespace find_m_value_l134_134522

theorem find_m_value : 
  ∀ (u v : ℝ), 
    (3 * u^2 + 4 * u + 5 = 0) ∧ 
    (3 * v^2 + 4 * v + 5 = 0) ∧ 
    (u + v = -4/3) ∧ 
    (u * v = 5/3) → 
    ∃ m n : ℝ, 
      (x^2 + m * x + n = 0) ∧ 
      ((u^2 + 1) + (v^2 + 1) = -m) ∧ 
      (m = -4/9) :=
by {
  -- Insert proof here
  sorry
}

end find_m_value_l134_134522


namespace total_remaining_books_l134_134100

-- Define the initial conditions as constants
def total_books_crazy_silly_school : ℕ := 14
def read_books_crazy_silly_school : ℕ := 8
def total_books_mystical_adventures : ℕ := 10
def read_books_mystical_adventures : ℕ := 5
def total_books_sci_fi_universe : ℕ := 18
def read_books_sci_fi_universe : ℕ := 12

-- Define the remaining books calculation
def remaining_books_crazy_silly_school : ℕ :=
  total_books_crazy_silly_school - read_books_crazy_silly_school

def remaining_books_mystical_adventures : ℕ :=
  total_books_mystical_adventures - read_books_mystical_adventures

def remaining_books_sci_fi_universe : ℕ :=
  total_books_sci_fi_universe - read_books_sci_fi_universe

-- Define the proof statement
theorem total_remaining_books : 
  remaining_books_crazy_silly_school + remaining_books_mystical_adventures + remaining_books_sci_fi_universe = 17 := by
  sorry

end total_remaining_books_l134_134100


namespace ellipse_slope_ratio_l134_134515

theorem ellipse_slope_ratio (a b x1 y1 x2 y2 c k1 k2 : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : c = a / 2) (h4 : a = 2) (h5 : c = 1) (h6 : b = Real.sqrt 3) 
  (h7 : 3 * x1 ^ 2 + 4 * y1 ^ 2 = 12 * c ^ 2) 
  (h8 : 3 * x2 ^ 2 + 4 * y2 ^ 2 = 12 * c ^ 2) 
  (h9 : x1 = y1 - c) (h10 : x2 = y2 - c)
  (h11 : y1^2 = 9 / 4)
  (h12 : y1 = -3 / 2 ∨ y1 = 3 / 2) 
  (h13 : k1 = -3 / 2) 
  (h14 : k2 = -1 / 2) :
  k1 / k2 = 3 := 
  sorry

end ellipse_slope_ratio_l134_134515


namespace determine_k_and_solution_l134_134680

theorem determine_k_and_solution :
  ∃ (k : ℚ), (5 * k * x^2 + 30 * x + 10 = 0 → k = 9/2) ∧
    (∃ (x : ℚ), (5 * (9/2) * x^2 + 30 * x + 10 = 0) ∧ x = -2/3) := by
  sorry

end determine_k_and_solution_l134_134680


namespace fish_remaining_l134_134903

def initial_fish : ℝ := 47.0
def given_away_fish : ℝ := 22.5

theorem fish_remaining : initial_fish - given_away_fish = 24.5 :=
by
  sorry

end fish_remaining_l134_134903


namespace vacation_cost_proof_l134_134537

noncomputable def vacation_cost (C : ℝ) :=
  C / 5 - C / 8 = 120

theorem vacation_cost_proof {C : ℝ} (h : vacation_cost C) : C = 1600 :=
by
  sorry

end vacation_cost_proof_l134_134537


namespace parameterization_of_line_l134_134306

theorem parameterization_of_line (t : ℝ) (g : ℝ → ℝ) 
  (h : ∀ t, (g t - 10) / 2 = t ) :
  g t = 5 * t + 10 := by
  sorry

end parameterization_of_line_l134_134306


namespace geom_seq_min_value_l134_134017

theorem geom_seq_min_value (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ n, a n = a 1 * q ^ (n - 1))
  (h_condition : a 7 = a 6 + 2 * a 5)
  (h_mult : ∃ m n, m ≠ n ∧ a m * a n = 16 * (a 1) ^ 2) :
  ∃ (m n : ℕ), m ≠ n ∧ m + n = 6 ∧ (1 / m : ℝ) + (4 / n : ℝ) = 3 / 2 :=
by
  sorry

end geom_seq_min_value_l134_134017


namespace sqrt_expression_meaningful_l134_134723

theorem sqrt_expression_meaningful (x : ℝ) : (2 * x - 4 ≥ 0) ↔ (x ≥ 2) :=
by
  -- Proof will be skipped
  sorry

end sqrt_expression_meaningful_l134_134723


namespace delta_value_l134_134140

noncomputable def delta : ℝ :=
  Real.arccos (
    (Finset.range 3600).sum (fun k => Real.sin ((2539 + k) * Real.pi / 180)) ^ Real.cos (2520 * Real.pi / 180) +
    (Finset.range 3599).sum (fun k => Real.cos ((2521 + k) * Real.pi / 180)) +
    Real.cos (6120 * Real.pi / 180)
  )

theorem delta_value : delta = 71 :=
by
  sorry

end delta_value_l134_134140


namespace find_solutions_in_positive_integers_l134_134683

theorem find_solutions_in_positive_integers :
  ∃ a b c x y z : ℕ,
  a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧
  a + b + c = x * y * z ∧ x + y + z = a * b * c ∧
  ((a = 3 ∧ b = 2 ∧ c = 1 ∧ x = 3 ∧ y = 2 ∧ z = 1) ∨
   (a = 5 ∧ b = 2 ∧ c = 1 ∧ x = 3 ∧ y = 3 ∧ z = 1) ∨
   (a = 3 ∧ b = 3 ∧ c = 1 ∧ x = 5 ∧ y = 2 ∧ z = 1)) :=
sorry

end find_solutions_in_positive_integers_l134_134683


namespace length_of_field_l134_134598

def width : ℝ := 13.5

def length (w : ℝ) : ℝ := 2 * w - 3

theorem length_of_field : length width = 24 :=
by
  -- full proof goes here
  sorry

end length_of_field_l134_134598


namespace qatar_location_is_accurate_l134_134076

def qatar_geo_location :=
  "The most accurate representation of Qatar's geographical location is latitude 25 degrees North, longitude 51 degrees East."

theorem qatar_location_is_accurate :
  qatar_geo_location = "The most accurate representation of Qatar's geographical location is latitude 25 degrees North, longitude 51 degrees East." :=
sorry

end qatar_location_is_accurate_l134_134076


namespace total_grapes_l134_134814

theorem total_grapes (r a n : ℕ) (h1 : r = 25) (h2 : a = r + 2) (h3 : n = a + 4) : r + a + n = 83 := by
  sorry

end total_grapes_l134_134814


namespace marbles_problem_l134_134276

theorem marbles_problem :
  let red_marbles := 20
  let green_marbles := 3 * red_marbles
  let yellow_marbles := 0.20 * green_marbles
  let total_marbles := green_marbles + 3 * green_marbles
  total_marbles - (red_marbles + green_marbles + yellow_marbles) = 148 := by
  sorry

end marbles_problem_l134_134276


namespace find_A_l134_134645

theorem find_A (A B : ℕ) (h : 632 - (100 * A + 10 * B) = 41) : A = 5 :=
by 
  sorry

end find_A_l134_134645


namespace sqrt_meaningful_l134_134720

theorem sqrt_meaningful (x : ℝ) : (2 * x - 4 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end sqrt_meaningful_l134_134720


namespace max_friendly_groups_19_max_friendly_groups_20_l134_134941

def friendly_group {Team : Type} (beat : Team → Team → Prop) (A B C : Team) : Prop :=
  beat A B ∧ beat B C ∧ beat C A

def max_friendly_groups_19_teams : ℕ := 285
def max_friendly_groups_20_teams : ℕ := 330

theorem max_friendly_groups_19 {Team : Type} (n : ℕ) (h : n = 19) (beat : Team → Team → Prop) :
  ∃ (G : ℕ), G = max_friendly_groups_19_teams := sorry

theorem max_friendly_groups_20 {Team : Type} (n : ℕ) (h : n = 20) (beat : Team → Team → Prop) :
  ∃ (G : ℕ), G = max_friendly_groups_20_teams := sorry

end max_friendly_groups_19_max_friendly_groups_20_l134_134941


namespace flag_arrangement_modulo_1000_l134_134313

theorem flag_arrangement_modulo_1000 :
  let red_flags := 8
  let white_flags := 8
  let black_flags := 1
  let total_flags := red_flags + white_flags + black_flags
  let number_of_gaps := total_flags + 1
  let valid_arrangements := (Nat.choose number_of_gaps white_flags) * (number_of_gaps - 2)
  valid_arrangements % 1000 = 315 :=
by
  sorry

end flag_arrangement_modulo_1000_l134_134313


namespace combination_lock_l134_134220

theorem combination_lock :
  (∃ (n_1 n_2 n_3 : ℕ), 
    n_1 ≥ 0 ∧ n_1 ≤ 39 ∧
    n_2 ≥ 0 ∧ n_2 ≤ 39 ∧
    n_3 ≥ 0 ∧ n_3 ≤ 39 ∧ 
    n_1 % 4 = n_3 % 4 ∧ 
    n_2 % 4 = (n_1 + 2) % 4) →
  ∃ (count : ℕ), count = 4000 :=
by
  sorry

end combination_lock_l134_134220


namespace csc_neg_45_eq_neg_sqrt_2_l134_134500

noncomputable def csc (θ : Real) : Real := 1 / Real.sin θ

theorem csc_neg_45_eq_neg_sqrt_2 :
  csc (-Real.pi / 4) = -Real.sqrt 2 := by
  sorry

end csc_neg_45_eq_neg_sqrt_2_l134_134500


namespace linear_function_not_passing_through_third_quadrant_l134_134370

theorem linear_function_not_passing_through_third_quadrant
  (m : ℝ)
  (h : 4 + 4 * m < 0) : 
  ∀ x y : ℝ, (y = m * x - m) → ¬ (x < 0 ∧ y < 0) :=
by
  sorry

end linear_function_not_passing_through_third_quadrant_l134_134370


namespace graduation_graduates_l134_134540

theorem graduation_graduates :
  ∃ G : ℕ, (∀ (chairs_for_parents chairs_for_teachers chairs_for_admins : ℕ),
    chairs_for_parents = 2 * G ∧
    chairs_for_teachers = 20 ∧
    chairs_for_admins = 10 ∧
    G + chairs_for_parents + chairs_for_teachers + chairs_for_admins = 180) ↔ G = 50 :=
by
  sorry

end graduation_graduates_l134_134540


namespace trigonometric_equation_solution_l134_134783

theorem trigonometric_equation_solution (x : ℝ) (k : ℤ) :
  5.14 * (Real.sin (3 * x)) + Real.sin (5 * x) = 2 * (Real.cos (2 * x)) ^ 2 - 2 * (Real.sin (3 * x)) ^ 2 →
  (∃ k : ℤ, x = (π / 2) * (2 * k + 1)) ∨ (∃ k : ℤ, x = (π / 18) * (4 * k + 1)) :=
  by
  intro h
  sorry

end trigonometric_equation_solution_l134_134783


namespace twelfth_even_multiple_of_5_l134_134926

theorem twelfth_even_multiple_of_5 : 
  ∃ n : ℕ, n > 0 ∧ (n % 2 = 0) ∧ (n % 5 = 0) ∧ ∀ m, (m > 0 ∧ (m % 2 = 0) ∧ (m % 5 = 0) ∧ m < n) → (m = 10 * (fin (n / 10) - 1)) := 
sorry

end twelfth_even_multiple_of_5_l134_134926


namespace second_train_cross_time_l134_134623

noncomputable def time_to_cross_second_train : ℝ :=
  let length := 120
  let t1 := 10
  let t_cross := 13.333333333333334
  let v1 := length / t1
  let v_combined := 240 / t_cross
  let v2 := v_combined - v1
  length / v2

theorem second_train_cross_time :
  let t2 := time_to_cross_second_train
  t2 = 20 :=
by
  sorry

end second_train_cross_time_l134_134623


namespace primes_ge_3_are_4k_pm1_infinitely_many_primes_4k_minus1_l134_134106

-- Part 1: Prove that every prime number >= 3 is of the form 4k-1 or 4k+1
theorem primes_ge_3_are_4k_pm1 (p : ℕ) (hp_prime: Nat.Prime p) (hp_ge_3: p ≥ 3) : 
  ∃ k : ℕ, p = 4 * k + 1 ∨ p = 4 * k - 1 :=
by
  sorry

-- Part 2: Prove that there are infinitely many primes of the form 4k-1
theorem infinitely_many_primes_4k_minus1 : 
  ∀ (n : ℕ), ∃ (p : ℕ), Nat.Prime p ∧ p = 4 * k - 1 ∧ p > n :=
by
  sorry

end primes_ge_3_are_4k_pm1_infinitely_many_primes_4k_minus1_l134_134106


namespace isosceles_triangle_perimeter_l134_134991

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 2) (h2 : b = 4) (isosceles : (a = b) ∨ (a = 2) ∨ (b = 2)) :
  (a = 2 ∧ b = 4 → 10) :=
begin
  -- assuming isosceles triangle means either two sides are equal or a = 2 or b = 2 which fits the isosceles definition in the context of provided lengths.
  sorry
end

end isosceles_triangle_perimeter_l134_134991


namespace math_problem_l134_134593

-- Definitions based on conditions
def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg4 (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

-- Main theorem statement
theorem math_problem :
  avg4 (avg4 2 2 0 2) (avg2 3 1) 0 3 = 13 / 8 :=
by
  sorry

end math_problem_l134_134593


namespace interior_diagonal_length_l134_134200

theorem interior_diagonal_length (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 26)
  (h2 : 4 * (a + b + c) = 28) : 
  (a^2 + b^2 + c^2) = 23 :=
by
  sorry

end interior_diagonal_length_l134_134200


namespace bus_tour_total_sales_l134_134223

noncomputable def total_sales (total_tickets sold_senior_tickets : Nat) (cost_senior_ticket cost_regular_ticket : Nat) : Nat :=
  let sold_regular_tickets := total_tickets - sold_senior_tickets
  let sales_senior := sold_senior_tickets * cost_senior_ticket
  let sales_regular := sold_regular_tickets * cost_regular_ticket
  sales_senior + sales_regular

theorem bus_tour_total_sales :
  total_sales 65 24 10 15 = 855 := by
    sorry

end bus_tour_total_sales_l134_134223


namespace number_of_possible_n_values_l134_134729

noncomputable def possible_n_values : Finset Nat := 
  { n | (0 < n ∧ n < 5 ∧ (2 * n + 10 + n + 15 > 3 * n + 5) ∧ (2 * n + 10 + 3 * n + 5 > n + 15) ∧ (n + 15 + 3 * n + 5 > 2 * n + 10)) }.toFinset

theorem number_of_possible_n_values : possible_n_values.card = 4 := 
  by
  sorry

end number_of_possible_n_values_l134_134729


namespace initial_passengers_l134_134953

theorem initial_passengers (P : ℝ) :
  (1/2 * (2/3 * P + 280) + 12 = 242) → P = 270 :=
by
  sorry

end initial_passengers_l134_134953


namespace sandy_initial_cost_l134_134749

theorem sandy_initial_cost 
  (repairs_cost : ℝ)
  (selling_price : ℝ)
  (gain_percent : ℝ)
  (h1 : repairs_cost = 200)
  (h2 : selling_price = 1400)
  (h3 : gain_percent = 40) :
  ∃ P : ℝ, P = 800 :=
by
  -- Proof steps would go here
  sorry

end sandy_initial_cost_l134_134749


namespace domain_of_f_l134_134596

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - x)

theorem domain_of_f :
  {x : ℝ | f x = Real.log (x^2 - x)} = {x : ℝ | x < 0 ∨ x > 1} :=
sorry

end domain_of_f_l134_134596


namespace initial_amount_is_800_l134_134795

variables (P R : ℝ)

theorem initial_amount_is_800
  (h1 : 956 = P * (1 + 3 * R / 100))
  (h2 : 1052 = P * (1 + 3 * (R + 4) / 100)) :
  P = 800 :=
sorry

end initial_amount_is_800_l134_134795


namespace cid_earnings_l134_134347

variable (x : ℕ)
variable (oil_change_price repair_price car_wash_price : ℕ)
variable (cars_repaired cars_washed total_earnings : ℕ)

theorem cid_earnings :
  (oil_change_price = 20) →
  (repair_price = 30) →
  (car_wash_price = 5) →
  (cars_repaired = 10) →
  (cars_washed = 15) →
  (total_earnings = 475) →
  (oil_change_price * x + repair_price * cars_repaired + car_wash_price * cars_washed = total_earnings) →
  x = 5 := by sorry

end cid_earnings_l134_134347


namespace total_cost_one_pizza_and_three_burgers_l134_134989

def burger_cost : ℕ := 9
def pizza_cost : ℕ := burger_cost * 2
def total_cost : ℕ := pizza_cost + (burger_cost * 3)

theorem total_cost_one_pizza_and_three_burgers :
  total_cost = 45 :=
by
  rw [total_cost, pizza_cost, burger_cost]
  norm_num

end total_cost_one_pizza_and_three_burgers_l134_134989


namespace free_space_on_new_drive_l134_134586

theorem free_space_on_new_drive
  (initial_free : ℝ) (initial_used : ℝ) (delete_size : ℝ) (new_files_size : ℝ) (new_drive_size : ℝ) :
  initial_free = 2.4 → initial_used = 12.6 → delete_size = 4.6 → new_files_size = 2 → new_drive_size = 20 →
  (new_drive_size - ((initial_used - delete_size) + new_files_size)) = 10 :=
by simp; sorry

end free_space_on_new_drive_l134_134586


namespace percentage_of_paycheck_went_to_taxes_l134_134043

-- Definitions
def original_paycheck : ℝ := 125
def savings : ℝ := 20
def spend_percentage : ℝ := 0.80
def save_percentage : ℝ := 0.20

-- Statement that needs to be proved
theorem percentage_of_paycheck_went_to_taxes (T : ℝ) :
  (0.20 * (1 - T / 100) * original_paycheck = savings) → T = 20 := 
by
  sorry

end percentage_of_paycheck_went_to_taxes_l134_134043


namespace george_total_socks_l134_134249

-- Define the initial number of socks George had
def initial_socks : ℝ := 28.0

-- Define the number of socks he bought
def bought_socks : ℝ := 36.0

-- Define the number of socks his Dad gave him
def given_socks : ℝ := 4.0

-- Define the number of total socks
def total_socks : ℝ := initial_socks + bought_socks + given_socks

-- State the theorem we want to prove
theorem george_total_socks : total_socks = 68.0 :=
by
  sorry

end george_total_socks_l134_134249


namespace shyam_weight_increase_l134_134616

theorem shyam_weight_increase (total_weight_after_increase : ℝ) (ram_initial_weight_ratio : ℝ) 
    (shyam_initial_weight_ratio : ℝ) (ram_increase_percent : ℝ) (total_increase_percent : ℝ) 
    (ram_total_weight_ratio : ram_initial_weight_ratio = 6) (shyam_initial_total_weight_ratio : shyam_initial_weight_ratio = 5) 
    (total_weight_after_increase_eq : total_weight_after_increase = 82.8) 
    (ram_increase_percent_eq : ram_increase_percent = 0.10) 
    (total_increase_percent_eq : total_increase_percent = 0.15) : 
  shyam_increase_percent = (21 : ℝ) :=
sorry

end shyam_weight_increase_l134_134616


namespace angelina_speed_from_grocery_to_gym_l134_134655

theorem angelina_speed_from_grocery_to_gym
    (v : ℝ)
    (hv : v > 0)
    (home_to_grocery_distance : ℝ := 150)
    (grocery_to_gym_distance : ℝ := 200)
    (time_difference : ℝ := 10)
    (time_home_to_grocery : ℝ := home_to_grocery_distance / v)
    (time_grocery_to_gym : ℝ := grocery_to_gym_distance / (2 * v))
    (h_time_diff : time_home_to_grocery - time_grocery_to_gym = time_difference) :
    2 * v = 10 := by
  sorry

end angelina_speed_from_grocery_to_gym_l134_134655


namespace price_of_pants_l134_134613

-- Given conditions
variables (P B : ℝ)
axiom h1 : P + B = 70.93
axiom h2 : P = B - 2.93

-- Statement to prove
theorem price_of_pants : P = 34.00 :=
by
  sorry

end price_of_pants_l134_134613


namespace simplify_expression_l134_134587

theorem simplify_expression (p : ℤ) : 
  ((7 * p + 3) - 3 * p * 2) * 4 + (5 - 2 / 2) * (8 * p - 12) = 36 * p - 36 :=
by
  sorry

end simplify_expression_l134_134587


namespace intersection_A_B_l134_134149

noncomputable def A : Set ℝ := {x | 2 * x^2 - 3 * x - 2 ≤ 0}
noncomputable def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := by
  sorry

end intersection_A_B_l134_134149


namespace probability_three_draws_one_white_one_red_probability_two_draws_one_white_one_red_one_other_l134_134923

-- Conditions
constant red_balls : ℕ := 3
constant white_balls : ℕ := 2

def total_balls : ℕ := red_balls + white_balls
def num_draws : ℕ := 3
def event_A_probability : ℝ := (red_balls / total_balls) * (white_balls / (total_balls - 1)) + (white_balls / total_balls) * (red_balls / (total_balls - 1))

-- Problem Statement
theorem probability_three_draws_one_white_one_red :
  let P_A := event_A_probability in
  let P_3_3 := P_A ^ num_draws in
  P_3_3 = 0.216 := by
  sorry

theorem probability_two_draws_one_white_one_red_one_other :
  let P_A := event_A_probability in
  let P_3_2 := (3.choose 2) * (P_A ^ 2) * ((1 - P_A) ^ 1) in
  P_3_2 = 0.432 := by
  sorry

end probability_three_draws_one_white_one_red_probability_two_draws_one_white_one_red_one_other_l134_134923


namespace difference_in_spending_l134_134828

-- Condition: original prices and discounts
def original_price_candy_bar : ℝ := 6
def discount_candy_bar : ℝ := 0.25
def original_price_chocolate : ℝ := 3
def discount_chocolate : ℝ := 0.10

-- The theorem to prove
theorem difference_in_spending : 
  (original_price_candy_bar * (1 - discount_candy_bar) - original_price_chocolate * (1 - discount_chocolate)) = 1.80 :=
by
  sorry

end difference_in_spending_l134_134828


namespace complex_expression_evaluation_l134_134047

-- Defining the imaginary unit
def i : ℂ := Complex.I

-- Defining the complex number z
def z : ℂ := 1 - i

-- Stating the theorem to prove
theorem complex_expression_evaluation : z^2 + (2 / z) = 1 - i := by
  sorry

end complex_expression_evaluation_l134_134047


namespace trajectory_of_midpoint_l134_134860

theorem trajectory_of_midpoint (M : ℝ × ℝ) (P : ℝ × ℝ) (N : ℝ × ℝ) :
  (P.1^2 + P.2^2 = 1) ∧
  (P.1 = M.1 ∧ P.2 = 2 * M.2) ∧ 
  (N.1 = P.1 ∧ N.2 = 0) ∧ 
  (M.1 = (P.1 + N.1) / 2 ∧ M.2 = (P.2 + N.2) / 2)
  → M.1^2 + 4 * M.2^2 = 1 := 
by
  sorry

end trajectory_of_midpoint_l134_134860


namespace domain_of_f_l134_134942

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f y = x}

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1)

theorem domain_of_f : domain f = {x | x > 1} := sorry

end domain_of_f_l134_134942


namespace smallest_n_l134_134142

theorem smallest_n (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x ∣ y^3) (h2 : y ∣ z^3) (h3 : z ∣ x^3)
  (h4 : x * y * z ∣ (x + y + z)^n) : n = 13 :=
sorry

end smallest_n_l134_134142


namespace average_goods_per_hour_l134_134485

-- Define the conditions
def morning_goods : ℕ := 64
def morning_hours : ℕ := 4
def afternoon_rate : ℕ := 23
def afternoon_hours : ℕ := 3

-- Define the target statement to be proven
theorem average_goods_per_hour : (morning_goods + afternoon_rate * afternoon_hours) / (morning_hours + afternoon_hours) = 19 := by
  -- Add proof steps here
  sorry

end average_goods_per_hour_l134_134485


namespace sec_240_eq_neg2_l134_134365

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

theorem sec_240_eq_neg2 : sec 240 = -2 := by
  -- Proof goes here
  sorry

end sec_240_eq_neg2_l134_134365


namespace bob_corn_stalks_per_row_l134_134820

noncomputable def corn_stalks_per_row
  (rows : ℕ)
  (bushels : ℕ)
  (stalks_per_bushel : ℕ) :
  ℕ :=
  (bushels * stalks_per_bushel) / rows

theorem bob_corn_stalks_per_row
  (rows : ℕ)
  (bushels : ℕ)
  (stalks_per_bushel : ℕ) :
  rows = 5 → bushels = 50 → stalks_per_bushel = 8 → corn_stalks_per_row rows bushels stalks_per_bushel = 80 :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  unfold corn_stalks_per_row
  rfl

end bob_corn_stalks_per_row_l134_134820


namespace find_a_l134_134517

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = Real.log (-a * x)) (h2 : ∀ x : ℝ, f (-x) = -f x) :
  a = 1 :=
by
  sorry

end find_a_l134_134517


namespace more_orange_pages_read_l134_134353

-- Define the conditions
def purple_pages_per_book : Nat := 230
def orange_pages_per_book : Nat := 510
def purple_books_read : Nat := 5
def orange_books_read : Nat := 4

-- Calculate the total pages read from purple and orange books respectively
def total_purple_pages_read : Nat := purple_pages_per_book * purple_books_read
def total_orange_pages_read : Nat := orange_pages_per_book * orange_books_read

-- State the theorem to be proved
theorem more_orange_pages_read : total_orange_pages_read - total_purple_pages_read = 890 :=
by
  -- This is where the proof steps would go, but we'll leave it as sorry to indicate the proof is not provided
  sorry

end more_orange_pages_read_l134_134353


namespace marcy_total_people_served_l134_134413

noncomputable def total_people_served_lip_gloss
  (tubs_lip_gloss : ℕ) (tubes_per_tub_lip_gloss : ℕ) (people_per_tube_lip_gloss : ℕ) : ℕ :=
  tubs_lip_gloss * tubes_per_tub_lip_gloss * people_per_tube_lip_gloss

noncomputable def total_people_served_mascara
  (tubs_mascara : ℕ) (tubes_per_tub_mascara : ℕ) (people_per_tube_mascara : ℕ) : ℕ :=
  tubs_mascara * tubes_per_tub_mascara * people_per_tube_mascara

theorem marcy_total_people_served :
  ∀ (tubs_lip_gloss tubs_mascara : ℕ) 
    (tubes_per_tub_lip_gloss tubes_per_tub_mascara 
     people_per_tube_lip_gloss people_per_tube_mascara : ℕ),
    tubs_lip_gloss = 6 → 
    tubes_per_tub_lip_gloss = 2 → 
    people_per_tube_lip_gloss = 3 → 
    tubs_mascara = 4 → 
    tubes_per_tub_mascara = 3 → 
    people_per_tube_mascara = 5 → 
    total_people_served_lip_gloss tubs_lip_gloss 
                                 tubes_per_tub_lip_gloss 
                                 people_per_tube_lip_gloss = 36 :=
by
  intros tubs_lip_gloss tubs_mascara 
         tubes_per_tub_lip_gloss tubes_per_tub_mascara 
         people_per_tube_lip_gloss people_per_tube_mascara
         h_tubs_lip_gloss h_tubes_per_tub_lip_gloss h_people_per_tube_lip_gloss
         h_tubs_mascara h_tubes_per_tub_mascara h_people_per_tube_mascara
  rw [h_tubs_lip_gloss, h_tubes_per_tub_lip_gloss, h_people_per_tube_lip_gloss]
  exact rfl


end marcy_total_people_served_l134_134413


namespace scientific_notation_correct_l134_134436

-- Define the input number
def input_number : ℕ := 858000000

-- Define the expected scientific notation result
def scientific_notation (n : ℕ) : ℝ := 8.58 * 10^8

-- The theorem states that the input number in scientific notation is indeed 8.58 * 10^8
theorem scientific_notation_correct :
  scientific_notation input_number = 8.58 * 10^8 :=
sorry

end scientific_notation_correct_l134_134436


namespace ab_cd_eq_one_l134_134832

theorem ab_cd_eq_one (a b c d : ℕ) (p : ℕ) 
  (h_div_a : a % p = 0)
  (h_div_b : b % p = 0)
  (h_div_c : c % p = 0)
  (h_div_d : d % p = 0)
  (h_div_ab_cd : (a * b - c * d) % p = 0) : 
  (a * b - c * d) = 1 :=
sorry

end ab_cd_eq_one_l134_134832


namespace cone_height_90_deg_is_36_8_l134_134664

noncomputable def cone_height_volume (V : ℝ) (θ : ℝ) : ℝ :=
  if θ = π / 2 then
    let r := (3 * V / π)^(1/3) in r
  else
    0  -- Not valid if the angle isn't 90 degrees

theorem cone_height_90_deg_is_36_8 :
  cone_height_volume (16384 * π) (π / 2) = 36.8 :=
by
  sorry

end cone_height_90_deg_is_36_8_l134_134664


namespace cookie_distribution_probability_l134_134660

theorem cookie_distribution_probability :
  let total_cookies := 12
  let types := 3
  let each_type := 4
  let children := 4
  let cookies_per_child := 3
  let p := 72
  let q := 1925
  let probability := ⟨p, q⟩ 

  (4 * 4 * 4 / (total_cookies choose cookies_per_child)) *
  (3 * 3 * 3 / ((total_cookies - cookies_per_child) choose cookies_per_child)) *
  (2 * 2 * 2 / ((total_cookies - 2 * cookies_per_child) choose cookies_per_child)) *
  1 = probability ∧ Nat.gcd p q = 1 ∧
  p + q = 1997 :=
by
  sorry

end cookie_distribution_probability_l134_134660


namespace nh3_oxidation_mass_l134_134231

theorem nh3_oxidation_mass
  (initial_volume : ℚ)
  (initial_cl2_percentage : ℚ)
  (initial_n2_percentage : ℚ)
  (escaped_volume : ℚ)
  (escaped_cl2_percentage : ℚ)
  (escaped_n2_percentage : ℚ)
  (molar_volume : ℚ)
  (cl2_molar_mass : ℚ)
  (nh3_molar_mass : ℚ) :
  initial_volume = 1.12 →
  initial_cl2_percentage = 0.9 →
  initial_n2_percentage = 0.1 →
  escaped_volume = 0.672 →
  escaped_cl2_percentage = 0.5 →
  escaped_n2_percentage = 0.5 →
  molar_volume = 22.4 →
  cl2_molar_mass = 71 →
  nh3_molar_mass = 17 →
  ∃ (mass_nh3_oxidized : ℚ),
    mass_nh3_oxidized = 0.34 := 
by {
  sorry
}

end nh3_oxidation_mass_l134_134231


namespace max_S_at_n_four_l134_134977

-- Define the sequence sum S_n
def S (n : ℕ) : ℤ := -(n^2 : ℤ) + (8 * n : ℤ)

-- Prove that S_n attains its maximum value at n = 4
theorem max_S_at_n_four : ∀ n : ℕ, S n ≤ S 4 :=
by
  sorry

end max_S_at_n_four_l134_134977


namespace toy_discount_price_l134_134767

theorem toy_discount_price (original_price : ℝ) (discount_rate : ℝ) (price_after_first_discount : ℝ) (price_after_second_discount : ℝ) : 
  original_price = 200 → 
  discount_rate = 0.1 →
  price_after_first_discount = original_price * (1 - discount_rate) →
  price_after_second_discount = price_after_first_discount * (1 - discount_rate) →
  price_after_second_discount = 162 :=
by
  intros h1 h2 h3 h4
  sorry

end toy_discount_price_l134_134767


namespace sample_size_ratio_l134_134331

theorem sample_size_ratio (n : ℕ) (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ)
                          (total_ratio : ℕ) (B_in_sample : ℕ)
                          (h_ratio : ratio_A = 1 ∧ ratio_B = 3 ∧ ratio_C = 5)
                          (h_total : total_ratio = ratio_A + ratio_B + ratio_C)
                          (h_B_sample : B_in_sample = 27)
                          (h_sampling_ratio_B : ratio_B / total_ratio = 1 / 3) :
                          n = 81 :=
by sorry

end sample_size_ratio_l134_134331


namespace statement_T_true_for_given_values_l134_134894

/-- Statement T: If the sum of the digits of a whole number m is divisible by 9, 
    then m is divisible by 9.
    The given values to check are 45, 54, 81, 63, and none of these. --/

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem statement_T_true_for_given_values :
  ∀ (m : ℕ), (m = 45 ∨ m = 54 ∨ m = 81 ∨ m = 63) →
    (is_divisible_by_9 (sum_of_digits m) → is_divisible_by_9 m) :=
by
  intros m H
  cases H
  case inl H1 => sorry
  case inr H2 =>
    cases H2
    case inl H1 => sorry
    case inr H2 =>
      cases H2
      case inl H1 => sorry
      case inr H2 => sorry

end statement_T_true_for_given_values_l134_134894


namespace find_D_plus_E_plus_F_l134_134916

noncomputable def g (x : ℝ) (D E F : ℝ) : ℝ := (x^2) / (D * x^2 + E * x + F)

theorem find_D_plus_E_plus_F (D E F : ℤ) 
  (h1 : ∀ x : ℝ, x > 3 → g x D E F > 0.3)
  (h2 : ∀ x : ℝ, ¬(D * x^2 + E * x + F = 0 ↔ (x = -3 ∨ x = 2))) :
  D + E + F = -8 :=
sorry

end find_D_plus_E_plus_F_l134_134916


namespace inequality_solution_non_negative_integer_solutions_l134_134073

theorem inequality_solution (x : ℝ) :
  (x - 2) / 2 ≤ (7 - x) / 3 → x ≤ 4 :=
by
  sorry

theorem non_negative_integer_solutions :
  { n : ℤ | n ≥ 0 ∧ n ≤ 4 } = {0, 1, 2, 3, 4} :=
by
  sorry

end inequality_solution_non_negative_integer_solutions_l134_134073


namespace find_n_l134_134011

noncomputable def arctan_sum_eq_pi_over_2 (n : ℕ) : Prop :=
  Real.arctan (1 / 3) + Real.arctan (1 / 4) + Real.arctan (1 / 7) + Real.arctan (1 / n) = Real.pi / 2

theorem find_n (h : ∃ n, arctan_sum_eq_pi_over_2 n) : ∃ n, n = 54 := by
  obtain ⟨n, hn⟩ := h
  have H : 1 / 3 + 1 / 4 + 1 / 7 < 1 := by sorry
  sorry

end find_n_l134_134011


namespace difference_of_squares_l134_134966

theorem difference_of_squares : 73^2 - 47^2 = 3120 :=
by sorry

end difference_of_squares_l134_134966


namespace cannot_obtain_fraction_3_5_l134_134734

theorem cannot_obtain_fraction_3_5 (n k : ℕ) :
  ¬ ∃ (a b : ℕ), (a = 5 + k ∧ b = 8 + k ∨ (∃ m : ℕ, a = m * 5 ∧ b = m * 8)) ∧ (a = 3 ∧ b = 5) :=
by
  sorry

end cannot_obtain_fraction_3_5_l134_134734


namespace arc_length_semicubical_parabola_correct_l134_134344

noncomputable def arc_length_semicubical_parabola : ℝ :=
∫ x in 0..9, sqrt(1 + (3 / 2 * x ^ (1 / 2)) ^ 2)

theorem arc_length_semicubical_parabola_correct :
  arc_length_semicubical_parabola = 28.552 :=
sorry

end arc_length_semicubical_parabola_correct_l134_134344


namespace measure_of_angle_A_proof_range_of_values_of_b_plus_c_over_a_proof_l134_134996

noncomputable def measure_of_angle_a (a b c : ℝ) (S : ℝ) (h_c : c = 2) (h_S : b * Real.cos (A / 2) = S) : Prop :=
  A = Real.pi / 3

theorem measure_of_angle_A_proof (a b c : ℝ) (S : ℝ) (h_c : c = 2) (h_S : b * Real.cos (A / 2) = S) : measure_of_angle_a a b c S h_c h_S :=
sorry

noncomputable def range_of_values_of_b_plus_c_over_a (a b c : ℝ) (A : ℝ) (h_A : A = Real.pi / 3) (h_c : c = 2) : Set ℝ :=
  {x : ℝ | 1 < x ∧ x ≤ 2}

theorem range_of_values_of_b_plus_c_over_a_proof (a b c : ℝ) (A : ℝ) (h_A : A = Real.pi / 3) (h_c : c = 2) : 
  ∃ x, x ∈ range_of_values_of_b_plus_c_over_a a b c A h_A h_c :=
sorry

end measure_of_angle_A_proof_range_of_values_of_b_plus_c_over_a_proof_l134_134996


namespace cost_of_450_candies_l134_134480

theorem cost_of_450_candies :
  let cost_per_box := 8
  let candies_per_box := 30
  let num_candies := 450
  cost_per_box * (num_candies / candies_per_box) = 120 := 
by 
  sorry

end cost_of_450_candies_l134_134480


namespace rebus_solution_l134_134006

theorem rebus_solution :
  ∃ (A B C : ℕ), 
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l134_134006


namespace greatest_n_leq_inequality_l134_134319

theorem greatest_n_leq_inequality : ∃ n : ℤ, (n^2 - 13 * n + 36 ≤ 0) ∧ ∀ m : ℤ, (m^2 - 13 * m + 36 ≤ 0) → m ≤ n := 
by
  existsi (9 : ℤ)
  split
  {
    -- Validate that 9 satisfies the inequality
    sorry
  }
  {
    -- Show for any m, if m satisfies the inequality, it must be less than or equals to 9
    intro m
    intro hm
    -- prove m <= 9
    sorry
  }

end greatest_n_leq_inequality_l134_134319


namespace pyramid_base_side_length_l134_134195

theorem pyramid_base_side_length (A : ℝ) (h : ℝ) (s : ℝ)
  (hA : A = 200)
  (hh : h = 40)
  (hface : A = (1 / 2) * s * h) : 
  s = 10 :=
by
  sorry

end pyramid_base_side_length_l134_134195


namespace det_A_is_neg9_l134_134960

noncomputable def A : Matrix (Fin 2) (Fin 2) ℤ := ![![-7, 5], ![6, -3]]

theorem det_A_is_neg9 : Matrix.det A = -9 := 
by 
  sorry

end det_A_is_neg9_l134_134960


namespace candy_left_l134_134887

variable (x : ℕ)

theorem candy_left (x : ℕ) : x - (18 + 7) = x - 25 :=
by sorry

end candy_left_l134_134887


namespace opposite_event_of_hitting_target_at_least_once_is_both_shots_miss_l134_134669

/-- A person is shooting at a target, firing twice in succession. 
    The opposite event of "hitting the target at least once" is "both shots miss". -/
theorem opposite_event_of_hitting_target_at_least_once_is_both_shots_miss :
  ∀ (A B : Prop) (hits_target_at_least_once both_shots_miss : Prop), 
    (hits_target_at_least_once → (A ∨ B)) → (both_shots_miss ↔ ¬hits_target_at_least_once) ∧ 
    (¬(A ∧ B) → both_shots_miss) :=
by
  sorry

end opposite_event_of_hitting_target_at_least_once_is_both_shots_miss_l134_134669


namespace polar_coordinates_of_2_neg2_l134_134679

noncomputable def rect_to_polar_coord (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let theta := if y < 0 
                then 2 * Real.pi - Real.arctan (x / (-y)) 
                else Real.arctan (y / x)
  (r, theta)

theorem polar_coordinates_of_2_neg2 :
  rect_to_polar_coord 2 (-2) = (2 * Real.sqrt 2, 7 * Real.pi / 4) :=
by 
  sorry

end polar_coordinates_of_2_neg2_l134_134679


namespace cos_36_is_correct_l134_134131

noncomputable def cos_36_eq : Prop :=
  let b := Real.cos (Real.pi * 36 / 180)
  let a := Real.cos (Real.pi * 72 / 180)
  (a = 2 * b^2 - 1) ∧ (b = (1 + Real.sqrt 5) / 4)

theorem cos_36_is_correct : cos_36_eq :=
by sorry

end cos_36_is_correct_l134_134131


namespace ratio_of_areas_of_circles_l134_134985

theorem ratio_of_areas_of_circles 
  (R_A R_B : ℝ) 
  (h : (π / 2 * R_A) = (π / 3 * R_B)) : 
  (π * R_A ^ 2) / (π * R_B ^ 2) = (4 : ℚ) / 9 := 
sorry

end ratio_of_areas_of_circles_l134_134985


namespace repeating_decimal_sum_correct_l134_134840

noncomputable def repeating_decimal_sum : ℚ :=
  let x := 2 / 3
  let y := 2 / 9
  let z := 4 / 9
  x + y - z

theorem repeating_decimal_sum_correct :
  repeating_decimal_sum = 4 / 9 :=
by
  sorry

end repeating_decimal_sum_correct_l134_134840


namespace apples_number_l134_134610

def num_apples (A O B : ℕ) : Prop :=
  A = O + 27 ∧ O = B + 11 ∧ A + O + B = 301 → A = 122

theorem apples_number (A O B : ℕ) : num_apples A O B := by
  sorry

end apples_number_l134_134610


namespace sum_minimum_nine_l134_134378

noncomputable def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
a + n * d

theorem sum_minimum_nine (a_1 a_8 a_13 S_n : ℤ) (d : ℤ) :
  a_1 = -26 ∧ a_8 + a_13 = 5 → 
  (∀ n : ℤ, S_n = (3 / 2) * n^2 - (55 / 2) * n) → (∃ n : ℕ, n = 9 ∧ ∀ m : ℕ, S_n n ≤ S_n m) :=
begin
  sorry
end

end sum_minimum_nine_l134_134378


namespace prove_expression_l134_134604

def otimes (a b : ℚ) : ℚ := a^2 / b

theorem prove_expression : ((otimes (otimes 1 2) 3) - (otimes 1 (otimes 2 3))) = -2/3 :=
by 
  sorry

end prove_expression_l134_134604


namespace find_line_equation_l134_134502

-- Define the point (2, -1) which the line passes through
def point : ℝ × ℝ := (2, -1)

-- Define the line perpendicular to 2x - 3y = 1
def perpendicular_line (x y : ℝ) : Prop := 2 * x - 3 * y - 1 = 0

-- The equation of the line we are supposed to find
def equation_of_line (x y : ℝ) : Prop := 3 * x + 2 * y - 4 = 0

-- Proof problem: prove the equation satisfies given the conditions
theorem find_line_equation :
  (equation_of_line point.1 point.2) ∧ 
  (∃ (a b c : ℝ), ∀ (x y : ℝ), perpendicular_line x y → equation_of_line x y) := sorry

end find_line_equation_l134_134502


namespace permutations_count_5040_four_digit_numbers_count_4356_even_four_digit_numbers_count_1792_l134_134512

open Finset

def digits : Finset ℤ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

noncomputable def permutations_no_repetition : ℤ :=
  (digits.card.factorial) / ((digits.card - 4).factorial)

noncomputable def four_digit_numbers_no_repetition : ℤ :=
  9 * ((digits.card - 1).factorial / ((digits.card - 1 - 3).factorial))

noncomputable def even_four_digit_numbers_gt_3000_no_repetition : ℤ :=
  784 + 1008

theorem permutations_count_5040 : permutations_no_repetition = 5040 := by
  sorry

theorem four_digit_numbers_count_4356 : four_digit_numbers_no_repetition = 4356 := by
  sorry

theorem even_four_digit_numbers_count_1792 : even_four_digit_numbers_gt_3000_no_repetition = 1792 := by
  sorry

end permutations_count_5040_four_digit_numbers_count_4356_even_four_digit_numbers_count_1792_l134_134512


namespace expected_expenditure_l134_134752

-- Define the parameters and conditions
def b : ℝ := 0.8
def a : ℝ := 2
def e_condition (e : ℝ) : Prop := |e| < 0.5
def revenue : ℝ := 10

-- Define the expenditure function based on the conditions
def expenditure (x e : ℝ) : ℝ := b * x + a + e

-- The expected expenditure should not exceed 10.5
theorem expected_expenditure (e : ℝ) (h : e_condition e) : expenditure revenue e ≤ 10.5 :=
sorry

end expected_expenditure_l134_134752


namespace points_on_line_l134_134467

theorem points_on_line (x y : ℝ) (h : x + y = 0) : y = -x :=
by
  sorry

end points_on_line_l134_134467


namespace find_y_l134_134874

theorem find_y (y: ℕ)
  (h1: ∃ (k : ℕ), y = 9 * k)
  (h2: y^2 > 225)
  (h3: y < 30)
: y = 18 ∨ y = 27 := 
sorry

end find_y_l134_134874


namespace min_value_a2b3c_l134_134693

theorem min_value_a2b3c {m : ℝ} (hm : m > 0)
  (hineq : ∀ x : ℝ, |x + 1| + |2 * x - 1| ≥ m)
  {a b c : ℝ} (habc : a^2 + 2 * b^2 + 3 * c^2 = m) :
  a + 2 * b + 3 * c ≥ -3 :=
sorry

end min_value_a2b3c_l134_134693


namespace ab_value_l134_134315

theorem ab_value (a b : ℝ) (h1 : a + b = 8) (h2 : a^3 + b^3 = 172) : ab = 85 / 6 := 
by
  sorry

end ab_value_l134_134315


namespace increasing_function_in_interval_l134_134956

noncomputable def y₁ (x : ℝ) : ℝ := abs (x + 1)
noncomputable def y₂ (x : ℝ) : ℝ := 3 - x
noncomputable def y₃ (x : ℝ) : ℝ := 1 / x
noncomputable def y₄ (x : ℝ) : ℝ := -x^2 + 4

theorem increasing_function_in_interval : ∀ x, (0 < x ∧ x < 1) → 
  y₁ x > y₁ (x - 0.1) ∧ y₂ x < y₂ (x - 0.1) ∧ y₃ x < y₃ (x - 0.1) ∧ y₄ x < y₄ (x - 0.1) :=
by {
  sorry
}

end increasing_function_in_interval_l134_134956


namespace fruit_problem_l134_134221

theorem fruit_problem
  (A B C : ℕ)
  (hA : A = 4) 
  (hB : B = 6) 
  (hC : C = 12) :
  ∃ x : ℕ, 1 = x / 2 := 
by
  sorry

end fruit_problem_l134_134221


namespace problem_solution_l134_134895

noncomputable def greatest_integer_not_exceeding (z : ℝ) : ℤ := Int.floor z

theorem problem_solution (x : ℝ) (y : ℝ) 
  (h1 : y = 4 * greatest_integer_not_exceeding x + 4)
  (h2 : y = 5 * greatest_integer_not_exceeding (x - 3) + 7)
  (h3 : x > 3 ∧ ¬ ∃ (n : ℤ), x = ↑n) :
  64 < x + y ∧ x + y < 65 :=
by
  sorry

end problem_solution_l134_134895


namespace either_p_or_q_false_suff_not_p_true_l134_134899

theorem either_p_or_q_false_suff_not_p_true (p q : Prop) : (p ∨ q = false) → (¬p = true) :=
by
  sorry

end either_p_or_q_false_suff_not_p_true_l134_134899


namespace common_point_geometric_progression_passing_l134_134954

theorem common_point_geometric_progression_passing
  (a b c : ℝ) (r : ℝ) (h_b : b = a * r) (h_c : c = a * r^2) :
  ∃ x y : ℝ, (∀ a ≠ 0, a * x + (a * r) * y = a * r^2) → (x = 0 ∧ y = 1) :=
by
  sorry

end common_point_geometric_progression_passing_l134_134954


namespace exp_calculation_l134_134127

theorem exp_calculation : 0.125^8 * (-8)^7 = -0.125 :=
by
  -- conditions used directly in proof
  have h1 : 0.125 = 1 / 8 := sorry
  have h2 : (-1)^7 = -1 := sorry
  -- the problem statement
  sorry

end exp_calculation_l134_134127


namespace parabola_tangent_line_l134_134893

noncomputable def gcd (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

theorem parabola_tangent_line (a b c : ℕ) (h1 : a^2 + (104 / 5) * a * b - 4 * b * c = 0)
  (h2 : b^2 - 5 * a^2 + 4 * a * c = 0) (hgcd : gcd a b c = 1) :
  a + b + c = 17 := by
  sorry

end parabola_tangent_line_l134_134893


namespace sum_three_times_integers_15_to_25_l134_134345

noncomputable def sumArithmeticSequence (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem sum_three_times_integers_15_to_25 :
  let a := 15
  let d := 1
  let n := 25 - 15 + 1
  3 * sumArithmeticSequence a d n = 660 := by
  -- This part can be filled in with the actual proof
  sorry

end sum_three_times_integers_15_to_25_l134_134345


namespace exponent_calculation_l134_134271

theorem exponent_calculation (a m n : ℝ) (h1 : a^m = 3) (h2 : a^n = 2) : 
  a^(2 * m - 3 * n) = 9 / 8 := 
by
  sorry

end exponent_calculation_l134_134271


namespace height_of_spherical_caps_l134_134514

theorem height_of_spherical_caps
  (r q : ℝ)
  (m₁ m₂ m₃ m₄ : ℝ)
  (h1 : m₂ = m₁ * q)
  (h2 : m₃ = m₁ * q^2)
  (h3 : m₄ = m₁ * q^3)
  (h4 : m₁ + m₂ + m₃ + m₄ = 2 * r) :
  m₁ = 2 * r * (q - 1) / (q^4 - 1) := 
sorry

end height_of_spherical_caps_l134_134514


namespace probability_both_truth_l134_134716

variable (P_A : ℝ) (P_B : ℝ)

theorem probability_both_truth (hA : P_A = 0.55) (hB : P_B = 0.60) :
  P_A * P_B = 0.33 :=
by
  sorry

end probability_both_truth_l134_134716


namespace reduced_price_per_dozen_l134_134469

theorem reduced_price_per_dozen 
  (P : ℝ) -- original price per apple
  (R : ℝ) -- reduced price per apple
  (A : ℝ) -- number of apples originally bought for Rs. 30
  (H1 : R = 0.7 * P) 
  (H2 : A * P = (A + 54) * R) :
  30 / (A + 54) * 12 = 2 :=
by
  sorry

end reduced_price_per_dozen_l134_134469


namespace age_problem_l134_134784

variables (a b c : ℕ)

theorem age_problem (h₁ : a = b + 2) (h₂ : b = 2 * c) (h₃ : a + b + c = 27) : b = 10 :=
by {
  -- Interactive proof steps can go here.
  sorry
}

end age_problem_l134_134784


namespace P_2n_expression_l134_134739

noncomputable def a (n : ℕ) : ℕ :=
  2 * n + 1

noncomputable def S (n : ℕ) : ℕ :=
  n * (n + 2)

noncomputable def b (n : ℕ) : ℕ :=
  2 ^ (n - 1)

noncomputable def T (n : ℕ) : ℕ :=
  2 * b n - 1

noncomputable def c (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 / S n else a n * b n
  
noncomputable def P (n : ℕ) : ℕ :=
  if n % 2 = 0 then (12 * n - 1) * 2^(2 * n + 1) / 9 + 2 / 9 + 2 * n / (2 * n + 1) else 0

theorem P_2n_expression (n : ℕ) : 
  P (2 * n) = (12 * n - 1) * 2^(2 * n + 1) / 9 + 2 / 9 + 2 * n / (2 * n + 1) :=
sorry

end P_2n_expression_l134_134739


namespace andrew_paid_1428_l134_134342

-- Define the constants for the problem
def rate_per_kg_grapes : ℕ := 98
def kg_grapes : ℕ := 11

def rate_per_kg_mangoes : ℕ := 50
def kg_mangoes : ℕ := 7

-- Calculate the cost of grapes and mangoes
def cost_grapes := rate_per_kg_grapes * kg_grapes
def cost_mangoes := rate_per_kg_mangoes * kg_mangoes

-- Calculate the total amount paid
def total_amount_paid := cost_grapes + cost_mangoes

-- State the proof problem
theorem andrew_paid_1428 :
  total_amount_paid = 1428 :=
by
  -- Add the proof to verify the calculations
  sorry

end andrew_paid_1428_l134_134342


namespace seeds_total_l134_134113

theorem seeds_total (wednesday_seeds thursday_seeds : ℕ) (h_wed : wednesday_seeds = 20) (h_thu : thursday_seeds = 2) : (wednesday_seeds + thursday_seeds) = 22 := by
  sorry

end seeds_total_l134_134113


namespace minimum_value_of_expression_l134_134493

theorem minimum_value_of_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + a * b + a * c + b * c = 4) : 
  2 * a + b + c ≥ 4 := 
by 
  sorry

end minimum_value_of_expression_l134_134493


namespace beth_longer_distance_by_5_miles_l134_134548

noncomputable def average_speed_john : ℝ := 40
noncomputable def time_john_hours : ℝ := 30 / 60
noncomputable def distance_john : ℝ := average_speed_john * time_john_hours

noncomputable def average_speed_beth : ℝ := 30
noncomputable def time_beth_hours : ℝ := (30 + 20) / 60
noncomputable def distance_beth : ℝ := average_speed_beth * time_beth_hours

theorem beth_longer_distance_by_5_miles : distance_beth - distance_john = 5 := by 
  sorry

end beth_longer_distance_by_5_miles_l134_134548


namespace angle_B_triangle_perimeter_l134_134885

variable {A B C a b c : Real}

-- Definitions and conditions for part 1
def sides_relation (a b c : ℝ) (A : ℝ) : Prop :=
  2 * c = a + 2 * b * Real.cos A

-- Definitions and conditions for part 2
def triangle_area (a b c : ℝ) (B : ℝ) : Prop :=
  (1 / 2) * a * c * Real.sin B = Real.sqrt 3

def side_b_value (b : ℝ) : Prop :=
  b = Real.sqrt 13

-- Theorem statement for part 1 
theorem angle_B (a b c A : ℝ) (h1: sides_relation a b c A) : B = Real.pi / 3 :=
sorry

-- Theorem statement for part 2 
theorem triangle_perimeter (a b c B : ℝ) (h1 : triangle_area a b c B) (h2 : side_b_value b) (h3 : B = Real.pi / 3) : a + b + c = 5 + Real.sqrt 13 :=
sorry

end angle_B_triangle_perimeter_l134_134885


namespace monochromatic_triangle_probability_l134_134241

open Classical

noncomputable def hexagon_edges : ℕ := 15  -- Total number of edges in K_6

-- Probability a single triangle is not monochromatic
noncomputable def prob_not_monochromatic : ℝ := 3/4

-- Probability that at least one triangle in K_6 is monochromatic
noncomputable def prob_monochromatic_triangle : ℝ := 1 - (prob_not_monochromatic)^20

theorem monochromatic_triangle_probability : 
  prob_monochromatic_triangle ≈ 0.99683 := 
by 
  -- The use of approximation here is abstract; in practice, you would detail the proof steps.
  sorry

end monochromatic_triangle_probability_l134_134241


namespace quadratic_has_single_solution_l134_134348

theorem quadratic_has_single_solution (k : ℚ) : 
  (∀ x : ℚ, 3 * x^2 - 7 * x + k = 0 → x = 7 / 6) ↔ k = 49 / 12 := 
by
  sorry

end quadratic_has_single_solution_l134_134348


namespace largest_of_five_consecutive_integers_with_product_15120_is_9_l134_134505

theorem largest_of_five_consecutive_integers_with_product_15120_is_9 :
  ∃ (a b c d e : ℤ), a * b * c * d * e = 15120 ∧ a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧ e = 9 :=
sorry

end largest_of_five_consecutive_integers_with_product_15120_is_9_l134_134505


namespace minimum_y_value_inequality_proof_l134_134144
-- Import necessary Lean library

-- Define a > 0, b > 0, and a + b = 1
variables {a b : ℝ}
variables (h_a_pos : a > 0) (h_b_pos : b > 0) (h_sum : a + b = 1)

-- Statement for Part (I): Prove the minimum value of y is 25/4
theorem minimum_y_value :
  (a + 1/a) * (b + 1/b) = 25/4 :=
sorry

-- Statement for Part (II): Prove the inequality
theorem inequality_proof :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 :=
sorry

end minimum_y_value_inequality_proof_l134_134144


namespace no_intersection_range_k_l134_134868

def problem_statement (k : ℝ) : Prop :=
  ∀ (x : ℝ),
    ¬(x > 1 ∧ x + 1 = k * x + 2) ∧ ¬(x < 1 ∧ -x - 1 = k * x + 2) ∧ 
    (x = 1 → (x + 1 ≠ k * x + 2 ∧ -x - 1 ≠ k * x + 2))

theorem no_intersection_range_k :
  ∀ (k : ℝ), problem_statement k ↔ -4 ≤ k ∧ k < -1 :=
sorry

end no_intersection_range_k_l134_134868


namespace smallest_four_digit_int_equiv_8_mod_9_l134_134777

theorem smallest_four_digit_int_equiv_8_mod_9 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 9 = 8 ∧ n = 1007 := 
by
  sorry

end smallest_four_digit_int_equiv_8_mod_9_l134_134777


namespace expression_undefined_at_12_l134_134851

theorem expression_undefined_at_12 :
  ¬ ∃ x : ℝ, x = 12 ∧ (x^2 - 24 * x + 144 = 0) →
  (∃ y : ℝ, y = (3 * x^3 + 5) / (x^2 - 24 * x + 144)) :=
by
  sorry

end expression_undefined_at_12_l134_134851


namespace abs_difference_of_squares_l134_134773

theorem abs_difference_of_squares : abs ((102: ℤ) ^ 2 - (98: ℤ) ^ 2) = 800 := by
  sorry

end abs_difference_of_squares_l134_134773


namespace probability_two_primes_is_1_over_29_l134_134631

open Finset

noncomputable def primes_upto_30 : Finset ℕ := filter Nat.Prime (range 31)

def total_pairs : ℕ := (range 31).card.choose 2

def prime_pairs : ℕ := (primes_upto_30).card.choose 2

theorem probability_two_primes_is_1_over_29 :
  prime_pairs.to_rat / total_pairs.to_rat = (1 : ℚ) / 29 := sorry

end probability_two_primes_is_1_over_29_l134_134631


namespace library_books_difference_l134_134918

theorem library_books_difference (total_books : ℕ) (borrowed_percentage : ℕ) 
  (initial_books : total_books = 400) 
  (percentage_borrowed : borrowed_percentage = 30) :
  (total_books - (borrowed_percentage * total_books / 100)) = 280 :=
by
  sorry

end library_books_difference_l134_134918


namespace work_completed_in_8_days_l134_134943

theorem work_completed_in_8_days 
  (A_complete : ℕ → Prop)
  (B_complete : ℕ → Prop)
  (C_complete : ℕ → Prop)
  (A_can_complete_in_10_days : A_complete 10)
  (B_can_complete_in_20_days : B_complete 20)
  (C_can_complete_in_30_days : C_complete 30)
  (A_leaves_5_days_before_completion : ∀ x : ℕ, x ≥ 5 → A_complete (x - 5))
  (C_leaves_3_days_before_completion : ∀ x : ℕ, x ≥ 3 → C_complete (x - 3)) :
  ∃ x : ℕ, x = 8 := sorry

end work_completed_in_8_days_l134_134943


namespace infection_probability_l134_134175

theorem infection_probability
  (malaria_percent : ℝ)
  (zika_percent : ℝ)
  (vaccine_reduction : ℝ)
  (prob_random_infection : ℝ)
  (P : ℝ) :
  malaria_percent = 0.40 →
  zika_percent = 0.20 →
  vaccine_reduction = 0.50 →
  prob_random_infection = 0.15 →
  0.15 = (0.40 * 0.50 * P) + (0.20 * P) →
  P = 0.375 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end infection_probability_l134_134175


namespace at_least_one_negative_l134_134909

-- Defining the circle partition and the properties given in the problem.
def circle_partition (a : Fin 7 → ℤ) : Prop :=
  ∃ (l1 l2 l3 : Finset (Fin 7)),
    l1.card = 4 ∧ l2.card = 4 ∧ l3.card = 4 ∧
    (∀ i ∈ l1, ∀ j ∉ l1, a i + a j = 0) ∧
    (∀ i ∈ l2, ∀ j ∉ l2, a i + a j = 0) ∧
    (∀ i ∈ l3, ∀ j ∉ l3, a i + a j = 0) ∧
    ∃ i, a i = 0

-- The main theorem to prove.
theorem at_least_one_negative : 
  ∀ (a : Fin 7 → ℤ), 
  circle_partition a → 
  ∃ i, a i < 0 :=
by
  sorry

end at_least_one_negative_l134_134909


namespace length_of_third_side_l134_134391

-- Definitions for sides and perimeter condition
variables (a b : ℕ) (h1 : a = 3) (h2 : b = 10) (p : ℕ) (h3 : p % 6 = 0)
variable (c : ℕ)

-- Definition for the triangle inequality
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Statement to prove the length of the third side
theorem length_of_third_side (h4 : triangle_inequality a b c)
  (h5 : p = a + b + c) : c = 11 :=
sorry

end length_of_third_side_l134_134391


namespace nancy_packs_of_crayons_l134_134745

def total_crayons : ℕ := 615
def crayons_per_pack : ℕ := 15

theorem nancy_packs_of_crayons : total_crayons / crayons_per_pack = 41 :=
by
  sorry

end nancy_packs_of_crayons_l134_134745


namespace complex_fraction_l134_134896

open Complex

theorem complex_fraction
  (a b : ℂ)
  (h : (a + b) / (a - b) - (a - b) / (a + b) = 2) :
  (a^4 + b^4) / (a^4 - b^4) - (a^4 - b^4) / (a^4 + b^4) = ((a^2 + b^2) - 3) / 3 := 
by
  sorry

end complex_fraction_l134_134896


namespace inequality_proof_l134_134701

theorem inequality_proof (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  (x^4 / (y-1)^2) + (y^4 / (z-1)^2) + (z^4 / (x-1)^2) ≥ 48 := 
by
  sorry -- The actual proof is omitted

end inequality_proof_l134_134701


namespace final_price_percentage_l134_134951

theorem final_price_percentage (P : ℝ) (h₀ : P > 0)
  (h₁ : ∃ P₁, P₁ = 0.80 * P)
  (h₂ : ∃ P₂, P₁ = 0.80 * P ∧ P₂ = P₁ - 0.10 * P₁) :
  P₂ = 0.72 * P :=
by
  sorry

end final_price_percentage_l134_134951


namespace distinct_four_digit_odd_numbers_l134_134708

-- Define the conditions as Lean definitions
def is_odd_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def valid_first_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 7 ∨ d = 9

-- The proposition we want to prove
theorem distinct_four_digit_odd_numbers (n : ℕ) :
  (∀ d, d ∈ [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] → is_odd_digit d) →
  valid_first_digit (n / 1000 % 10) →
  1000 ≤ n ∧ n < 10000 →
  n = 500 :=
sorry

end distinct_four_digit_odd_numbers_l134_134708


namespace evaluate_series_l134_134242

-- Define the series S
noncomputable def S : ℝ := ∑' n : ℕ, (n + 1) / (3 ^ (n + 1))

-- Lean statement to show the evaluated series
theorem evaluate_series : (3:ℝ)^S = (3:ℝ)^(3 / 4) :=
by
  -- The proof is omitted
  sorry

end evaluate_series_l134_134242


namespace range_of_f_l134_134136

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x+1) + 3

theorem range_of_f : Set.range f = Set.Ici 2 := 
by 
  sorry

end range_of_f_l134_134136


namespace omega_range_l134_134185

theorem omega_range (ω : ℝ) (a b : ℝ) (hω_pos : ω > 0) (h_range : π ≤ a ∧ a < b ∧ b ≤ 2 * π)
  (h_sin : Real.sin (ω * a) + Real.sin (ω * b) = 2) :
  ω ∈ Set.Icc (9 / 4 : ℝ) (5 / 2) ∪ Set.Ici (13 / 4) :=
by
  sorry

end omega_range_l134_134185


namespace sara_spent_on_rented_movie_l134_134294

def total_spent_on_movies : ℝ := 36.78
def spent_on_tickets : ℝ := 2 * 10.62
def spent_on_bought_movie : ℝ := 13.95

theorem sara_spent_on_rented_movie : 
  (total_spent_on_movies - spent_on_tickets - spent_on_bought_movie = 1.59) := 
by sorry

end sara_spent_on_rented_movie_l134_134294


namespace factor_expression_l134_134233

theorem factor_expression :
  let expr := (20 * x^3 + 100 * x - 10) - (-5 * x^3 + 5 * x - 10)
  expr = 5 * x * (5 * x^2 + 19) :=
by {
  let term1 := 20 * x^3 + 100 * x - 10,
  let term2 := -5 * x^3 + 5 * x - 10,
  have h : expr = term1 - term2,
  sorry
}

end factor_expression_l134_134233


namespace complex_number_properties_l134_134251

open Complex

-- Defining the imaginary unit
def i : ℂ := Complex.I

-- Given conditions in Lean: \( z \) satisfies \( z(2+i) = i^{10} \)
def satisfies_condition (z : ℂ) : Prop :=
  z * (2 + i) = i^10

-- Theorem stating the required proofs
theorem complex_number_properties (z : ℂ) (hc : satisfies_condition z) :
  Complex.abs z = Real.sqrt 5 / 5 ∧ 
  (z.re < 0 ∧ z.im > 0) := by
  -- Placeholders for the proof steps
  sorry

end complex_number_properties_l134_134251


namespace brokerage_percentage_correct_l134_134600

noncomputable def brokerage_percentage (market_value : ℝ) (income : ℝ) (investment : ℝ) (nominal_rate : ℝ) : ℝ :=
  let face_value := (income * 100) / nominal_rate
  let market_price := (face_value * market_value) / 100
  let brokerage_amount := investment - market_price
  (brokerage_amount / investment) * 100

theorem brokerage_percentage_correct :
  brokerage_percentage 110.86111111111111 756 8000 10.5 = 0.225 :=
by
  sorry

end brokerage_percentage_correct_l134_134600


namespace scientific_notation_correct_l134_134437

-- Define the input number
def input_number : ℕ := 858000000

-- Define the expected scientific notation result
def scientific_notation (n : ℕ) : ℝ := 8.58 * 10^8

-- The theorem states that the input number in scientific notation is indeed 8.58 * 10^8
theorem scientific_notation_correct :
  scientific_notation input_number = 8.58 * 10^8 :=
sorry

end scientific_notation_correct_l134_134437


namespace three_layers_coverage_l134_134460

/--
Three table runners have a combined area of 208 square inches. 
By overlapping the runners to cover 80% of a table of area 175 square inches, 
the area that is covered by exactly two layers of runner is 24 square inches. 
Prove that the area of the table that is covered with three layers of runner is 22 square inches.
--/
theorem three_layers_coverage :
  ∀ (A T two_layers total_table_coverage : ℝ),
  A = 208 ∧ total_table_coverage = 0.8 * 175 ∧ two_layers = 24 →
  A = (total_table_coverage - two_layers - T) + 2 * two_layers + 3 * T →
  T = 22 :=
by
  intros A T two_layers total_table_coverage h1 h2
  sorry

end three_layers_coverage_l134_134460


namespace circles_are_separate_l134_134161

def circle_center (a b r : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

theorem circles_are_separate :
  circle_center 0 0 1 x y → 
  circle_center 3 (-4) 3 x' y' →
  dist (0, 0) (3, -4) > 1 + 3 :=
by
  intro h₁ h₂
  sorry

end circles_are_separate_l134_134161


namespace cannot_obtain_fraction_l134_134735

noncomputable def fraction (a b : ℕ) : ℚ := a / b

theorem cannot_obtain_fraction (k n : ℕ) :
  let f_start := fraction 5 8 in
  let f_target := fraction 3 5 in
  ∀ (a b : ℕ), 
    (a = 5 + k ∧ b = 8 + k) ∨ 
    (a = n * 5 ∧ b = n * 8) →
  fraction a b ≠ f_target :=
by
  let f_start := fraction 5 8
  let f_target := fraction 3 5
  assume a b h
  cases h with h1 h2
  -- Add your proof here
  · sorry
  · sorry

end cannot_obtain_fraction_l134_134735


namespace ordered_pair_solution_l134_134141

theorem ordered_pair_solution :
  ∃ x y : ℚ, 7 * x - 50 * y = 3 ∧ 3 * y - x = 5 ∧ x = -259 / 29 ∧ y = -38 / 29 :=
by sorry

end ordered_pair_solution_l134_134141


namespace math_proof_equiv_l134_134081

def A := 5
def B := 3
def C := 2
def D := 0
def E := 0
def F := 1
def G := 0

theorem math_proof_equiv : (A * 1000 + B * 100 + C * 10 + D) + (E * 100 + F * 10 + G) = 5300 :=
by
  sorry

end math_proof_equiv_l134_134081


namespace perimeter_of_new_rectangle_l134_134654

-- Definitions based on conditions
def side_of_square : ℕ := 8
def length_of_rectangle : ℕ := 8
def breadth_of_rectangle : ℕ := 4

-- Perimeter calculation
def perimeter (length breadth : ℕ) : ℕ := 2 * (length + breadth)

-- Formal statement of the problem
theorem perimeter_of_new_rectangle :
  perimeter (side_of_square + length_of_rectangle) side_of_square = 48 :=
  by sorry

end perimeter_of_new_rectangle_l134_134654


namespace slope_range_l134_134677

theorem slope_range (a b : ℝ) (h₁ : a ≠ -2) (h₂ : a ≠ 2) 
  (h₃ : a^2 / 4 + b^2 / 3 = 1) (h₄ : -2 ≤ b / (a - 2) ∧ b / (a - 2) ≤ -1) :
  (3 / 8 ≤ b / (a + 2) ∧ b / (a + 2) ≤ 3 / 4) :=
sorry

end slope_range_l134_134677


namespace bob_distance_when_meet_l134_134103

-- Definitions of the variables and conditions
def distance_XY : ℝ := 40
def yolanda_rate : ℝ := 2  -- Yolanda's walking rate in miles per hour
def bob_rate : ℝ := 4      -- Bob's walking rate in miles per hour
def yolanda_start_time : ℝ := 1 -- Yolanda starts 1 hour earlier 

-- Prove that Bob has walked 25.33 miles when he meets Yolanda
theorem bob_distance_when_meet : 
  ∃ t : ℝ, 2 * (t + yolanda_start_time) + 4 * t = distance_XY ∧ (4 * t = 25.33) := 
by
  sorry

end bob_distance_when_meet_l134_134103


namespace find_num_students_l134_134104

variables (N T : ℕ)
variables (h1 : T = N * 80)
variables (h2 : 5 * 20 = 100)
variables (h3 : (T - 100) / (N - 5) = 90)

theorem find_num_students (h1 : T = N * 80) (h3 : (T - 100) / (N - 5) = 90) : N = 35 :=
sorry

end find_num_students_l134_134104


namespace neither_probability_l134_134272

-- Definitions of the probabilities P(A), P(B), and P(A ∩ B)
def P_A : ℝ := 0.63
def P_B : ℝ := 0.49
def P_A_and_B : ℝ := 0.32

-- Definition stating the probability of neither event
theorem neither_probability :
  (1 - (P_A + P_B - P_A_and_B)) = 0.20 := 
sorry

end neither_probability_l134_134272


namespace Durakavalyanie_last_lesson_class_1C_l134_134477

theorem Durakavalyanie_last_lesson_class_1C :
  ∃ (class_lesson : String × Nat → String), 
  class_lesson ("1B", 1) = "Kurashenie" ∧
  (∃ (k m n : Nat), class_lesson ("1A", k) = "Durakavalyanie" ∧ class_lesson ("1B", m) = "Durakavalyanie" ∧ m > k) ∧
  class_lesson ("1A", 2) ≠ "Nizvedenie" ∧
  class_lesson ("1C", 3) = "Durakavalyanie" :=
sorry

end Durakavalyanie_last_lesson_class_1C_l134_134477


namespace john_vegetables_l134_134889

theorem john_vegetables (beef_used vege_used : ℕ) :
  beef_used = 4 - 1 →
  vege_used = 2 * beef_used →
  vege_used = 6 :=
by
  intros h_beef_used h_vege_used
  unfold beef_used vege_used
  exact sorry

end john_vegetables_l134_134889


namespace rectangle_diagonal_floor_eq_169_l134_134542

-- Definitions of points and properties
structure Rectangle (α : Type*) :=
(P Q R S : α)
(PQ : ℝ) (PS : ℝ)
(PQ_eq : PQ = 120)
(T_mid_PR : Prop)
(S_perpendicular_PQ : Prop)

-- Prove the desired property using the conditions
theorem rectangle_diagonal_floor_eq_169 {α : Type*} (rect : Rectangle α)
  (h : rect.PQ = 120)
  (ht : rect.T_mid_PR)
  (hs : rect.S_perpendicular_PQ) : 
  ⌊rect.PQ * Real.sqrt 2⌋ = 169 :=
sorry

end rectangle_diagonal_floor_eq_169_l134_134542


namespace A_oplus_B_eq_l134_134687

def set_diff (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}
def symm_diff (M N : Set ℝ) : Set ℝ := set_diff M N ∪ set_diff N M
def A : Set ℝ := {y | ∃ x:ℝ, y = 3^x}
def B : Set ℝ := {y | ∃ x:ℝ, y = -(x-1)^2 + 2}

theorem A_oplus_B_eq : symm_diff A B = {y | y ≤ 0} ∪ {y | y > 2} := by {
  sorry
}

end A_oplus_B_eq_l134_134687


namespace price_after_two_reductions_l134_134805

-- Define the two reductions as given in the conditions
def first_day_reduction (P : ℝ) : ℝ := P * 0.88
def second_day_reduction (P : ℝ) : ℝ := first_day_reduction P * 0.9

-- Main theorem: Price on the second day is 79.2% of the original price
theorem price_after_two_reductions (P : ℝ) : second_day_reduction P = 0.792 * P :=
by
  sorry

end price_after_two_reductions_l134_134805


namespace total_assignments_for_28_points_l134_134569

-- Definitions based on conditions
def assignments_needed (points : ℕ) : ℕ :=
  (points / 7 + 1) * (points % 7) + (points / 7) * (7 - points % 7)

-- The theorem statement, which asserts the answer to the given problem
theorem total_assignments_for_28_points : assignments_needed 28 = 70 :=
by
  -- proof will go here
  sorry

end total_assignments_for_28_points_l134_134569


namespace expression_value_l134_134227

theorem expression_value :
  (6^2 - 3^2)^4 = 531441 := by
  -- Proof steps were omitted
  sorry

end expression_value_l134_134227


namespace lcm_25_35_50_l134_134969

theorem lcm_25_35_50 : Nat.lcm (Nat.lcm 25 35) 50 = 350 := by
  sorry

end lcm_25_35_50_l134_134969


namespace sum_of_three_squares_not_divisible_by_3_l134_134578

theorem sum_of_three_squares_not_divisible_by_3
    (N : ℕ) (n : ℕ) (a b c : ℤ) 
    (h1 : N = 9^n * (a^2 + b^2 + c^2))
    (h2 : ∃ (a1 b1 c1 : ℤ), a = 3 * a1 ∧ b = 3 * b1 ∧ c = 3 * c1) :
    ∃ (k m n : ℤ), N = k^2 + m^2 + n^2 ∧ (¬ (3 ∣ k ∧ 3 ∣ m ∧ 3 ∣ n)) :=
sorry

end sum_of_three_squares_not_divisible_by_3_l134_134578


namespace aira_rubber_bands_l134_134580

theorem aira_rubber_bands (total_bands : ℕ) (bands_each : ℕ) (samantha_extra : ℕ) (aira_fewer : ℕ)
  (h1 : total_bands = 18) 
  (h2 : bands_each = 6) 
  (h3 : samantha_extra = 5) 
  (h4 : aira_fewer = 1) : 
  ∃ x : ℕ, x + (x + samantha_extra) + (x + aira_fewer) = total_bands ∧ x = 4 :=
by
  sorry

end aira_rubber_bands_l134_134580


namespace sum_of_coefficients_l134_134304

noncomputable def simplify (x : ℝ) : ℝ := 
  (x^3 + 11 * x^2 + 38 * x + 40) / (x + 3)

theorem sum_of_coefficients : 
  (∀ x : ℝ, (x ≠ -3) → (simplify x = x^2 + 8 * x + 14)) ∧
  (1 + 8 + 14 + -3 = 20) :=
by      
  sorry

end sum_of_coefficients_l134_134304


namespace distinct_natural_primes_l134_134842

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem distinct_natural_primes :
  ∃ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 5 ∧
  is_prime (a * b + c * d) ∧
  is_prime (a * c + b * d) ∧
  is_prime (a * d + b * c) := by
  sorry

end distinct_natural_primes_l134_134842


namespace fraction_square_equality_l134_134579

theorem fraction_square_equality (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) 
    (h : a / b + c / d = 1) : 
    (a / b)^2 + c / d = (c / d)^2 + a / b :=
by
  sorry

end fraction_square_equality_l134_134579


namespace total_combined_grapes_l134_134812

theorem total_combined_grapes :
  ∀ (r a y : ℕ), (r = 25) → (a = r + 2) → (y = a + 4) → (r + a + y = 83) :=
by
  intros r a y hr ha hy
  rw [hr, ha, hy]
  sorry

end total_combined_grapes_l134_134812


namespace a100_pos_a100_abs_lt_018_l134_134024

noncomputable def a (n : ℕ) : ℝ := real.cos (10^n * real.pi / 180)

theorem a100_pos : (a 100 > 0) :=
by
  sorry

theorem a100_abs_lt_018 : (|a 100| < 0.18) :=
by
  sorry

end a100_pos_a100_abs_lt_018_l134_134024


namespace mary_saves_in_five_months_l134_134056

def washing_earnings : ℕ := 20
def walking_earnings : ℕ := 40
def monthly_earnings : ℕ := washing_earnings + walking_earnings
def savings_rate : ℕ := 2
def monthly_savings : ℕ := monthly_earnings / savings_rate
def total_savings_target : ℕ := 150

theorem mary_saves_in_five_months :
  total_savings_target / monthly_savings = 5 :=
by
  sorry

end mary_saves_in_five_months_l134_134056


namespace probability_two_primes_l134_134628

theorem probability_two_primes (S : Finset ℕ) (S = {1, 2, ..., 30}) 
  (primes : Finset ℕ) (primes = {p ∈ S | Prime p}) :
  (primes.card = 10) →
  (S.card = 30) →
  (nat.choose 2) (primes.card) / (nat.choose 2) (S.card) = 1 / 10 :=
begin
  sorry
end

end probability_two_primes_l134_134628


namespace Thabo_owns_25_hardcover_nonfiction_books_l134_134193

variable (H P F : ℕ)

-- Conditions
def condition1 := P = H + 20
def condition2 := F = 2 * P
def condition3 := H + P + F = 160

-- Goal
theorem Thabo_owns_25_hardcover_nonfiction_books (H P F : ℕ) (h1 : condition1 H P) (h2 : condition2 P F) (h3 : condition3 H P F) : H = 25 :=
by
  sorry

end Thabo_owns_25_hardcover_nonfiction_books_l134_134193


namespace initial_percentage_water_is_80_l134_134330

noncomputable def initial_kola_solution := 340
noncomputable def added_sugar := 3.2
noncomputable def added_water := 10
noncomputable def added_kola := 6.8
noncomputable def final_percentage_sugar := 14.111111111111112
noncomputable def percentage_kola := 6

theorem initial_percentage_water_is_80 :
  ∃ (W : ℝ), W = 80 :=
by
  sorry

end initial_percentage_water_is_80_l134_134330


namespace box_length_is_10_l134_134666

theorem box_length_is_10
  (width height vol_cube num_cubes : ℕ)
  (h₀ : width = 13)
  (h₁ : height = 5)
  (h₂ : vol_cube = 5)
  (h₃ : num_cubes = 130) :
  (num_cubes * vol_cube) / (width * height) = 10 :=
by
  -- Proof steps will be filled here.
  sorry

end box_length_is_10_l134_134666


namespace value_of_expression_l134_134334

theorem value_of_expression (p q r s : ℝ) (h : -27 * p + 9 * q - 3 * r + s = -7) : 
  4 * p - 2 * q + r - s = 7 :=
by
  sorry

end value_of_expression_l134_134334


namespace marias_profit_l134_134055

theorem marias_profit 
  (initial_loaves : ℕ)
  (morning_price : ℝ)
  (afternoon_discount : ℝ)
  (late_afternoon_price : ℝ)
  (cost_per_loaf : ℝ)
  (loaves_sold_morning : ℕ)
  (loaves_sold_afternoon : ℕ)
  (loaves_remaining : ℕ)
  (revenue_morning : ℝ)
  (revenue_afternoon : ℝ)
  (revenue_late_afternoon : ℝ)
  (total_revenue : ℝ)
  (total_cost : ℝ)
  (profit : ℝ) :
  initial_loaves = 60 →
  morning_price = 3.0 →
  afternoon_discount = 0.75 →
  late_afternoon_price = 1.50 →
  cost_per_loaf = 1.0 →
  loaves_sold_morning = initial_loaves / 3 →
  loaves_sold_afternoon = (initial_loaves - loaves_sold_morning) / 2 →
  loaves_remaining = initial_loaves - loaves_sold_morning - loaves_sold_afternoon →
  revenue_morning = loaves_sold_morning * morning_price →
  revenue_afternoon = loaves_sold_afternoon * (afternoon_discount * morning_price) →
  revenue_late_afternoon = loaves_remaining * late_afternoon_price →
  total_revenue = revenue_morning + revenue_afternoon + revenue_late_afternoon →
  total_cost = initial_loaves * cost_per_loaf →
  profit = total_revenue - total_cost →
  profit = 75 := sorry

end marias_profit_l134_134055


namespace hexagonal_prism_sum_maximum_l134_134339

noncomputable def hexagonal_prism_max_sum (h_u h_v h_w h_x h_y h_z : ℕ) (u v w x y z : ℝ) : ℝ :=
  u + v + w + x + y + z

def max_sum_possible (h_u h_v h_w h_x h_y h_z : ℕ) : ℝ :=
  if h_u = 4 ∧ h_v = 7 ∧ h_w = 10 ∨
     h_u = 4 ∧ h_x = 7 ∧ h_y = 10 ∨
     h_u = 4 ∧ h_y = 7 ∧ h_z = 10 ∨
     h_v = 4 ∧ h_x = 7 ∧ h_w = 10 ∨
     h_v = 4 ∧ h_y = 7 ∧ h_z = 10 ∨
     h_w = 4 ∧ h_x = 7 ∧ h_z = 10
  then 78
  else 0

theorem hexagonal_prism_sum_maximum (h_u h_v h_w h_x h_y h_z : ℕ) :
  max_sum_possible h_u h_v h_w h_x h_y h_z = 78 → ∃ (u v w x y z : ℝ), hexagonal_prism_max_sum h_u h_v h_w h_x h_y h_z u v w x y z = 78 := 
by 
  sorry

end hexagonal_prism_sum_maximum_l134_134339


namespace sqrt_meaningful_l134_134721

theorem sqrt_meaningful (x : ℝ) : (2 * x - 4 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end sqrt_meaningful_l134_134721


namespace number_division_reduction_l134_134786

theorem number_division_reduction (x : ℕ) (h : x / 3 = x - 48) : x = 72 := 
sorry

end number_division_reduction_l134_134786


namespace probability_digit_three_in_repeating_block_l134_134295

theorem probability_digit_three_in_repeating_block :
  let repeating_block := "615384" in
  let num_threes := repeating_block.to_list.filter (λ digit => digit = '3').length in
  let block_length := repeating_block.to_list.length in
  num_threes / block_length = (1 : ℚ) / 6 :=
by
  sorry

end probability_digit_three_in_repeating_block_l134_134295


namespace carrots_as_potatoes_l134_134902

variable (G O C P : ℕ)

theorem carrots_as_potatoes :
  G = 8 →
  G = (1 / 3 : ℚ) * O →
  O = 2 * C →
  P = 2 →
  (C / P : ℚ) = 6 :=
by intros hG1 hG2 hO hP; sorry

end carrots_as_potatoes_l134_134902


namespace M_plus_2N_equals_330_l134_134395

theorem M_plus_2N_equals_330 (M N : ℕ) :
  (4 : ℚ) / 7 = M / 63 ∧ (4 : ℚ) / 7 = 84 / N → M + 2 * N = 330 := by
  sorry

end M_plus_2N_equals_330_l134_134395


namespace solve_for_t_l134_134372

variable (f : ℝ → ℝ)
variable (x t : ℝ)

-- Conditions
def cond1 : Prop := ∀ x, f ((1 / 2) * x - 1) = 2 * x + 3
def cond2 : Prop := f t = 4

-- Theorem statement
theorem solve_for_t (h1 : cond1 f) (h2 : cond2 f t) : t = -3 / 4 := by
  sorry

end solve_for_t_l134_134372


namespace coordinates_of_C_l134_134402

noncomputable def point := (ℚ × ℚ)

def A : point := (2, 8)
def B : point := (6, 14)
def M : point := (4, 11)
def L : point := (6, 6)
def C : point := (14, 2)

-- midpoint formula definition
def is_midpoint (M A B : point) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Main statement to prove
theorem coordinates_of_C (hM : is_midpoint M A B) : C = (14, 2) :=
  sorry

end coordinates_of_C_l134_134402


namespace arc_length_is_correct_l134_134261

-- Define the radius and central angle as given
def radius := 16
def central_angle := 2

-- Define the arc length calculation
def arc_length (r : ℕ) (α : ℕ) := α * r

-- The theorem stating the mathematically equivalent proof problem
theorem arc_length_is_correct : arc_length radius central_angle = 32 :=
by sorry

end arc_length_is_correct_l134_134261


namespace megan_dials_fatima_correctly_l134_134564

noncomputable def count_permutations : ℕ := (Finset.univ : Finset (Equiv.Perm (Fin 3))).card
noncomputable def total_numbers : ℕ := 4 * count_permutations

theorem megan_dials_fatima_correctly :
  (1 : ℚ) / (total_numbers : ℚ) = 1 / 24 :=
by
  sorry

end megan_dials_fatima_correctly_l134_134564


namespace molecular_weight_N2O3_l134_134244

variable (atomic_weight_N : ℝ) (atomic_weight_O : ℝ)
variable (n_N_atoms : ℝ) (n_O_atoms : ℝ)
variable (expected_molecular_weight : ℝ)

theorem molecular_weight_N2O3 :
  atomic_weight_N = 14.01 →
  atomic_weight_O = 16.00 →
  n_N_atoms = 2 →
  n_O_atoms = 3 →
  expected_molecular_weight = 76.02 →
  (n_N_atoms * atomic_weight_N + n_O_atoms * atomic_weight_O = expected_molecular_weight) :=
by
  intros
  sorry

end molecular_weight_N2O3_l134_134244


namespace red_marbles_eq_14_l134_134550

theorem red_marbles_eq_14 (total_marbles : ℕ) (yellow_marbles : ℕ) (R : ℕ) (B : ℕ)
  (h1 : total_marbles = 85)
  (h2 : yellow_marbles = 29)
  (h3 : B = 3 * R)
  (h4 : (total_marbles - yellow_marbles) = R + B) :
  R = 14 :=
by
  sorry

end red_marbles_eq_14_l134_134550


namespace min_k_plus_p_is_19199_l134_134681

noncomputable def find_min_k_plus_p : ℕ :=
  let D := 1007
  let domain_len := 1 / D
  let min_k : ℕ := 19  -- Minimum k value for which domain length condition holds, found via problem conditions
  let p_for_k (k : ℕ) : ℕ := (D * (k^2 - 1)) / k
  let k_plus_p (k : ℕ) : ℕ := k + p_for_k k
  k_plus_p min_k

theorem min_k_plus_p_is_19199 : find_min_k_plus_p = 19199 :=
  sorry

end min_k_plus_p_is_19199_l134_134681


namespace Maria_high_school_students_l134_134414

theorem Maria_high_school_students (M J : ℕ) (h1 : M = 4 * J) (h2 : M + J = 3600) : M = 2880 :=
sorry

end Maria_high_school_students_l134_134414


namespace intersection_of_M_and_N_l134_134160

def set_M : Set ℝ := {x : ℝ | x^2 - x ≥ 0}
def set_N : Set ℝ := {x : ℝ | x < 2}

theorem intersection_of_M_and_N :
  set_M ∩ set_N = {x : ℝ | x ≤ 0 ∨ (1 ≤ x ∧ x < 2)} :=
by
  sorry

end intersection_of_M_and_N_l134_134160


namespace red_button_probability_l134_134282

-- Definitions of the initial state
def initial_red_buttons : ℕ := 8
def initial_blue_buttons : ℕ := 12
def total_buttons := initial_red_buttons + initial_blue_buttons

-- Condition of removal and remaining buttons
def removed_buttons := total_buttons - (5 / 8 : ℚ) * total_buttons

-- Equal number of red and blue buttons removed
def removed_red_buttons := removed_buttons / 2
def removed_blue_buttons := removed_buttons / 2

-- State after removal
def remaining_red_buttons := initial_red_buttons - removed_red_buttons
def remaining_blue_buttons := initial_blue_buttons - removed_blue_buttons

-- Jars after removal
def jar_X := remaining_red_buttons + remaining_blue_buttons
def jar_Y := removed_red_buttons + removed_blue_buttons

-- Probability calculations
def probability_red_X : ℚ := remaining_red_buttons / jar_X
def probability_red_Y : ℚ := removed_red_buttons / jar_Y

-- Final probability
def final_probability : ℚ := probability_red_X * probability_red_Y

theorem red_button_probability :
  final_probability = 4 / 25 := 
  sorry

end red_button_probability_l134_134282


namespace xyz_squared_l134_134373

theorem xyz_squared (x y z p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0)
  (hxy : x + y = p) (hyz : y + z = q) (hzx : z + x = r) :
  x^2 + y^2 + z^2 = (p^2 + q^2 + r^2 - p * q - q * r - r * p) / 2 :=
by
  sorry

end xyz_squared_l134_134373


namespace part1_part2_part3_l134_134376

open Real

noncomputable def g (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x
noncomputable def h (x : ℝ) : ℝ := log x
noncomputable def f (a x : ℝ) : ℝ := g a x + h x

theorem part1 (a : ℝ) (h_a : a = 1) : 
  let g := g a
  let g' := 2 * x - 3
  ∀ (x y : ℝ), y = g 1 → (x + y + 1 = 0) → (g 1 = g' 1 := by sorry

theorem part2 (a : ℝ) (h_positive : 0 < a) (h_fmin : ∀ x, 1 ≤ x ∧ x ≤ exp 1 → -2 ≤ f a x) : 
  a = 1 := by sorry

theorem part3 (a : ℝ) (h_ineq : ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → 
  (f a x1 - f a x2) / (x1 - x2) > -2) : 
  0 ≤ a ∧ a ≤ 8 := by sorry

end part1_part2_part3_l134_134376


namespace range_of_a_l134_134385

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, (a+2)/(x+1) = 1 ∧ x ≤ 0) ↔ (a ≤ -1 ∧ a ≠ -2) :=
by
  sorry

end range_of_a_l134_134385


namespace vertex_of_parabola_y_eq_x2_minus_2_l134_134445

theorem vertex_of_parabola_y_eq_x2_minus_2 :
  vertex (λ x : ℝ, x^2 - 2) = (0, -2) := 
sorry

end vertex_of_parabola_y_eq_x2_minus_2_l134_134445


namespace regular_polygon_interior_angle_160_l134_134240

theorem regular_polygon_interior_angle_160 (n : ℕ) (h : 160 * n = 180 * (n - 2)) : n = 18 :=
by {
  sorry
}

end regular_polygon_interior_angle_160_l134_134240


namespace find_cost_price_l134_134788

/-- Statement: Given Mohit sold an article for $18000 and 
if he offered a discount of 10% on the selling price, he would have earned a profit of 8%, 
prove that the cost price (CP) of the article is $15000. -/

def discounted_price (sp : ℝ) := sp - (0.10 * sp)
def profit_price (cp : ℝ) := cp * 1.08

theorem find_cost_price (sp : ℝ) (discount: sp = 18000) (profit_discount: profit_price (discounted_price sp) = discounted_price sp):
    ∃ (cp : ℝ), cp = 15000 :=
by
    sorry

end find_cost_price_l134_134788


namespace calculate_expression_l134_134823

theorem calculate_expression :
  (-0.25) ^ 2014 * (-4) ^ 2015 = -4 :=
by
  sorry

end calculate_expression_l134_134823


namespace ring_width_l134_134937

noncomputable def innerCircumference : ℝ := 352 / 7
noncomputable def outerCircumference : ℝ := 528 / 7

noncomputable def radius (C : ℝ) : ℝ := C / (2 * Real.pi)

theorem ring_width :
  let r_inner := radius innerCircumference
  let r_outer := radius outerCircumference
  r_outer - r_inner = 4 :=
by
  -- Definitions for inner and outer radius
  let r_inner := radius innerCircumference
  let r_outer := radius outerCircumference
  -- Proof goes here
  sorry

end ring_width_l134_134937


namespace problem_statement_l134_134098

def product_of_first_n (n : ℕ) : ℕ := List.prod (List.range' 1 n)

def sum_of_first_n (n : ℕ) : ℕ := List.sum (List.range' 1 n)

theorem problem_statement : 
  let numerator := product_of_first_n 9  -- product of numbers 1 through 8
  let denominator := sum_of_first_n 9  -- sum of numbers 1 through 8
  numerator / denominator = 1120 :=
by {
  sorry
}

end problem_statement_l134_134098


namespace total_coins_last_month_l134_134818

theorem total_coins_last_month (m s : ℝ) : 
  (100 = 1.25 * m) ∧ (100 = 0.80 * s) → m + s = 205 :=
by sorry

end total_coins_last_month_l134_134818


namespace analogical_reasoning_correctness_l134_134650

theorem analogical_reasoning_correctness 
  (a b c : ℝ)
  (va vb vc : ℝ) :
  (a + b) * c = (a * c + b * c) ↔ 
  (va + vb) * vc = (va * vc + vb * vc) := 
sorry

end analogical_reasoning_correctness_l134_134650


namespace tony_average_time_l134_134574

-- Definitions based on the conditions
def distance_to_store : ℕ := 4 -- in miles
def walking_speed : ℕ := 2 -- in MPH
def running_speed : ℕ := 10 -- in MPH

-- Conditions
def time_walking : ℕ := (distance_to_store / walking_speed) * 60 -- in minutes
def time_running : ℕ := (distance_to_store / running_speed) * 60 -- in minutes

def total_time : ℕ := time_walking + 2 * time_running -- Total time spent in minutes
def number_of_days : ℕ := 3 -- Number of days

def average_time : ℕ := total_time / number_of_days -- Average time in minutes

-- Statement to prove
theorem tony_average_time : average_time = 56 := by 
  sorry

end tony_average_time_l134_134574


namespace odd_base_divisibility_by_2_base_divisibility_by_m_l134_134656

-- Part (a)
theorem odd_base_divisibility_by_2 (q : ℕ) :
  (∀ a : ℕ, (a * q) % 2 = 0 ↔ a % 2 = 0) → q % 2 = 1 := 
sorry

-- Part (b)
theorem base_divisibility_by_m (q m : ℕ) (h1 : m > 1) :
  (∀ a : ℕ, (a * q) % m = 0 ↔ a % m = 0) → ∃ k : ℕ, q = 1 + m * k ∧ k ≥ 1 :=
sorry

end odd_base_divisibility_by_2_base_divisibility_by_m_l134_134656


namespace proof_of_problem_l134_134152

noncomputable def f : ℝ → ℝ := sorry  -- define f as a function in ℝ to ℝ

theorem proof_of_problem 
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_f1 : f 1 = 1)
  (h_periodic : ∀ x : ℝ, f (x + 6) = f x + f 3) :
  f 2015 + f 2016 = -1 := 
sorry

end proof_of_problem_l134_134152


namespace rectangle_area_increase_l134_134987

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  let l_new := 1.3 * l
  let w_new := 1.2 * w
  let A_new := l_new * w_new
  let A := l * w
  let increase := A_new - A
  let percent_increase := (increase / A) * 100
  percent_increase = 56 := sorry

end rectangle_area_increase_l134_134987


namespace white_surface_area_fraction_l134_134827

theorem white_surface_area_fraction
    (total_cubes : ℕ)
    (white_cubes : ℕ)
    (red_cubes : ℕ)
    (edge_length : ℕ)
    (white_exposed_area : ℕ)
    (total_surface_area : ℕ)
    (fraction : ℚ)
    (h1 : total_cubes = 64)
    (h2 : white_cubes = 14)
    (h3 : red_cubes = 50)
    (h4 : edge_length = 4)
    (h5 : white_exposed_area = 6)
    (h6 : total_surface_area = 96)
    (h7 : fraction = 1 / 16)
    (h8 : white_cubes + red_cubes = total_cubes)
    (h9 : 6 * (edge_length * edge_length) = total_surface_area)
    (h10 : white_exposed_area / total_surface_area = fraction) :
    fraction = 1 / 16 := by
    sorry

end white_surface_area_fraction_l134_134827


namespace iterated_kernels_l134_134968

noncomputable def K (x t : ℝ) : ℝ := 
  if 0 ≤ x ∧ x < t then 
    x + t 
  else if t < x ∧ x ≤ 1 then 
    x - t 
  else 
    0

noncomputable def K1 (x t : ℝ) : ℝ := K x t

noncomputable def K2 (x t : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < t then 
    (-2 / 3) * x^3 + t^3 - x^2 * t + 2 * x * t^2 - x * t + (x - t) / 2 + 1 / 3
  else if t < x ∧ x ≤ 1 then 
    (-2 / 3) * x^3 - t^3 + x^2 * t + 2 * x * t^2 - x * t + (x - t) / 2 + 1 / 3
  else
    0

theorem iterated_kernels (x t : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1) :
  K1 x t = K x t ∧
  K2 x t = 
  if 0 ≤ x ∧ x < t then 
    (-2 / 3) * x^3 + t^3 - x^2 * t + 2 * x * t^2 - x * t + (x - t) / 2 + 1 / 3
  else if t < x ∧ x ≤ 1 then 
    (-2 / 3) * x^3 - t^3 + x^2 * t + 2 * x * t^2 - x * t + (x - t) / 2 + 1 / 3
  else
    0 := by
  sorry

end iterated_kernels_l134_134968


namespace a4_b4_c4_double_square_l134_134101

theorem a4_b4_c4_double_square (a b c : ℤ) (h : a = b + c) : 
  a^4 + b^4 + c^4 = 2 * ((a^2 - b * c)^2) :=
by {
  sorry -- proof is not provided as per instructions
}

end a4_b4_c4_double_square_l134_134101


namespace sum_as_common_fraction_l134_134363

/-- The sum of 0.2 + 0.04 + 0.006 + 0.0008 + 0.00010 as a common fraction -/
theorem sum_as_common_fraction : (0.2 + 0.04 + 0.006 + 0.0008 + 0.00010) = (12345 / 160000) := by
  sorry

end sum_as_common_fraction_l134_134363


namespace Robert_more_than_Claire_l134_134054

variable (Lisa Claire Robert : ℕ)

theorem Robert_more_than_Claire (h1 : Lisa = 3 * Claire) (h2 : Claire = 10) (h3 : Robert > Claire) :
  Robert > 10 :=
by
  rw [h2] at h3
  assumption

end Robert_more_than_Claire_l134_134054


namespace triangle_inequality_l134_134901

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥ 3 :=
sorry

end triangle_inequality_l134_134901


namespace mark_has_3_tanks_l134_134415

-- Define conditions
def pregnant_fish_per_tank : ℕ := 4
def young_per_fish : ℕ := 20
def total_young : ℕ := 240

-- Theorem statement that Mark has 3 tanks
theorem mark_has_3_tanks : (total_young / (pregnant_fish_per_tank * young_per_fish)) = 3 :=
by
  sorry

end mark_has_3_tanks_l134_134415


namespace aira_rubber_bands_l134_134582

variable (S A J : ℕ)

-- Conditions
def conditions (S A J : ℕ) : Prop :=
  S = A + 5 ∧ A = J - 1 ∧ S + A + J = 18

-- Proof problem
theorem aira_rubber_bands (S A J : ℕ) (h : conditions S A J) : A = 4 :=
by
  -- introduce the conditions
  obtain ⟨h₁, h₂, h₃⟩ := h
  -- use sorry to skip the proof
  sorry

end aira_rubber_bands_l134_134582


namespace max_frac_sum_l134_134695

theorem max_frac_sum {n : ℕ} (h_n : n > 1) :
  ∀ (a b c d : ℕ), (a + c ≤ n) ∧ (b > 0) ∧ (d > 0) ∧
  (a * d + b * c < b * d) → 
  ↑a / ↑b + ↑c / ↑d ≤ (1 - 1 / ( ⌊(2*n : ℝ)/3 + 1/6⌋₊ + 1) * ( ⌊(2*n : ℝ)/3 + 1/6⌋₊ * (n - ⌊(2*n : ℝ)/3 + 1/6⌋₊) + 1)) :=
by sorry

end max_frac_sum_l134_134695


namespace sum_of_squares_edges_l134_134444

-- Define Points
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define given conditions (4 vertices each on two parallel planes)
def A1 : Point := { x := 0, y := 0, z := 0 }
def A2 : Point := { x := 1, y := 0, z := 0 }
def A3 : Point := { x := 1, y := 1, z := 0 }
def A4 : Point := { x := 0, y := 1, z := 0 }

def B1 : Point := { x := 0, y := 0, z := 1 }
def B2 : Point := { x := 1, y := 0, z := 1 }
def B3 : Point := { x := 1, y := 1, z := 1 }
def B4 : Point := { x := 0, y := 1, z := 1 }

-- Function to calculate distance squared between two points
def dist_sq (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2 + (p1.z - p2.z) ^ 2

-- The Theorem to be proven
theorem sum_of_squares_edges : dist_sq A1 B2 + dist_sq A2 B3 + dist_sq A3 B4 + dist_sq A4 B1 = 8 := by
  sorry

end sum_of_squares_edges_l134_134444


namespace cd_e_value_l134_134016

theorem cd_e_value (a b c d e f : ℝ) 
  (h1 : a * b * c = 195) (h2 : b * c * d = 65) 
  (h3 : d * e * f = 250) (h4 : (a * f) / (c * d) = 0.75) :
  c * d * e = 1000 := 
by
  sorry

end cd_e_value_l134_134016


namespace vectors_parallel_iff_l134_134162

-- Define the vectors a and b as given in the conditions
def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, m + 1)

-- Define what it means for two vectors to be parallel
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

-- The statement that we need to prove
theorem vectors_parallel_iff (m : ℝ) : parallel a (b m) ↔ m = 1 := by
  sorry

end vectors_parallel_iff_l134_134162


namespace max_value_of_function_l134_134150

theorem max_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1.5) : 
  ∃ m, ∀ y, y = 4 * x * (3 - 2 * x) → m = 9 / 2 :=
sorry

end max_value_of_function_l134_134150


namespace sufficient_but_not_necessary_condition_l134_134855

variable (a b x y : ℝ)

theorem sufficient_but_not_necessary_condition (ha : a > 0) (hb : b > 0) :
  ((x > a ∧ y > b) → (x + y > a + b ∧ x * y > a * b)) ∧
  ¬((x + y > a + b ∧ x * y > a * b) → (x > a ∧ y > b)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l134_134855


namespace problem1_problem2_l134_134074

section
variables (x a : ℝ)

-- Problem 1: Prove \(2^{3x-1} < 2 \implies x < \frac{2}{3}\)
theorem problem1 : (2:ℝ)^(3*x-1) < 2 → x < (2:ℝ)/3 :=
by sorry

-- Problem 2: Prove \(a^{3x^2+3x-1} < a^{3x^2+3} \implies (a > 1 \implies x < \frac{4}{3}) \land (0 < a < 1 \implies x > \frac{4}{3})\) given \(a > 0\) and \(a \neq 1\)
theorem problem2 (h0 : a > 0) (h1 : a ≠ 1) :
  a^(3*x^2 + 3*x - 1) < a^(3*x^2 + 3) →
  ((1 < a → x < (4:ℝ)/3) ∧ (0 < a ∧ a < 1 → x > (4:ℝ)/3)) :=
by sorry
end

end problem1_problem2_l134_134074


namespace total_pay_is_correct_l134_134948

-- Define the constants and conditions
def regular_rate := 3  -- $ per hour
def regular_hours := 40  -- hours
def overtime_multiplier := 2  -- overtime pay is twice the regular rate
def overtime_hours := 8  -- hours

-- Calculate regular and overtime pay
def regular_pay := regular_rate * regular_hours
def overtime_rate := regular_rate * overtime_multiplier
def overtime_pay := overtime_rate * overtime_hours

-- Calculate total pay
def total_pay := regular_pay + overtime_pay

-- Prove that the total pay is $168
theorem total_pay_is_correct : total_pay = 168 := by
  -- The proof goes here
  sorry

end total_pay_is_correct_l134_134948


namespace toll_constant_l134_134612

theorem toll_constant (t : ℝ) (x : ℝ) (constant : ℝ) : 
  (t = 1.50 + 0.50 * (x - constant)) → 
  (x = 18 / 2) → 
  (t = 5) → 
  constant = 2 :=
by
  intros h1 h2 h3
  sorry

end toll_constant_l134_134612


namespace quadratic_solution_value_l134_134608

open Real

theorem quadratic_solution_value (a b : ℝ) (h1 : 2 + b = -a) (h2 : 2 * b = -6) :
  (2 * a + b)^2023 = -1 :=
sorry

end quadratic_solution_value_l134_134608


namespace day_50_of_year_N_minus_1_l134_134177

-- Definitions for the problem conditions
def day_of_week (n : ℕ) : ℕ := n % 7

-- Given that the 250th day of year N is a Friday
axiom day_250_of_year_N_is_friday : day_of_week 250 = 5

-- Given that the 150th day of year N+1 is a Friday
axiom day_150_of_year_N_plus_1_is_friday : day_of_week 150 = 5

-- Calculate the day of the week for the 50th day of year N-1
theorem day_50_of_year_N_minus_1 :
  day_of_week 50 = 4 :=
  sorry

end day_50_of_year_N_minus_1_l134_134177


namespace initial_mean_corrected_l134_134451

theorem initial_mean_corrected
  (M : ℝ)
  (h : 30 * M + 10 = 30 * 140.33333333333334) :
  M = 140 :=
by
  sorry

end initial_mean_corrected_l134_134451


namespace fence_perimeter_l134_134765

theorem fence_perimeter 
  (N : ℕ) (w : ℝ) (g : ℝ) 
  (square_posts : N = 36) 
  (post_width : w = 0.5) 
  (gap_length : g = 8) :
  4 * ((N / 4 - 1) * g + (N / 4) * w) = 274 :=
by
  sorry

end fence_perimeter_l134_134765


namespace domain_of_function_l134_134449

noncomputable def domain : Set ℝ := {x | x ≥ 1/2 ∧ x ≠ 1}

theorem domain_of_function : ∀ (x : ℝ), (2 * x - 1 ≥ 0) ∧ (x ^ 2 + x - 2 ≠ 0) ↔ (x ∈ domain) :=
by 
  sorry

end domain_of_function_l134_134449


namespace scientific_notation_of_858_million_l134_134439

theorem scientific_notation_of_858_million :
  858000000 = 8.58 * 10 ^ 8 :=
sorry

end scientific_notation_of_858_million_l134_134439


namespace Liam_homework_assignments_l134_134115

theorem Liam_homework_assignments : 
  let assignments_needed (points : ℕ) : ℕ := match points with
    | 0     => 0
    | n+1 =>
        if n+1 <= 4 then 1
        else (4 + (((n+1) - 1)/4 - 1))

  30 <= 4 + 8 + 12 + 16 + 20 + 24 + 28 + 16 → ((λ points => List.sum (List.map assignments_needed (List.range points))) 30) = 128 :=
by
  sorry

end Liam_homework_assignments_l134_134115


namespace sum_of_digits_largest_n_is_13_l134_134288

-- Define the necessary conditions
def single_digit_primes : List ℕ := [2, 3, 5, 7]

def is_valid_prime_combination (d e : ℕ) : Prop := 
  d ∈ single_digit_primes ∧ 
  e ∈ single_digit_primes ∧ 
  d < e ∧ 
  Prime (d^2 + e^2)

def product_three_primes (d e : ℕ) : ℕ := d * e * (d^2 + e^2)

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

noncomputable def largest_n : ℕ := 
  (single_digit_primes.product single_digit_primes).filter (λ p, is_valid_prime_combination p.1 p.2)
  |>.map (λ p, product_three_primes p.1 p.2)
  |>.maximum.get_or_else 0

theorem sum_of_digits_largest_n_is_13 : sum_of_digits largest_n = 13 := by
  sorry

end sum_of_digits_largest_n_is_13_l134_134288


namespace green_notebook_cost_l134_134064

def total_cost : ℕ := 45
def black_cost : ℕ := 15
def pink_cost : ℕ := 10
def num_green_notebooks : ℕ := 2

theorem green_notebook_cost :
  (total_cost - (black_cost + pink_cost)) / num_green_notebooks = 10 :=
by
  sorry

end green_notebook_cost_l134_134064


namespace rebus_solution_l134_134005

theorem rebus_solution :
  ∃ (A B C : ℕ), 
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l134_134005


namespace pow_mod_eq_l134_134925

theorem pow_mod_eq : (17 ^ 2001) % 23 = 11 := 
by {
  sorry
}

end pow_mod_eq_l134_134925


namespace isabella_initial_hair_length_l134_134998

theorem isabella_initial_hair_length
  (final_length : ℕ)
  (growth_over_year : ℕ)
  (initial_length : ℕ)
  (h_final : final_length = 24)
  (h_growth : growth_over_year = 6)
  (h_initial : initial_length = 18) :
  initial_length + growth_over_year = final_length := 
by 
  sorry

end isabella_initial_hair_length_l134_134998


namespace quadratic_root_l134_134018

/-- If one root of the quadratic equation x^2 - 2x + n = 0 is 3, then n is -3. -/
theorem quadratic_root (n : ℝ) (h : (3 : ℝ)^2 - 2 * 3 + n = 0) : n = -3 :=
sorry

end quadratic_root_l134_134018


namespace sum_of_decimals_as_fraction_l134_134361

theorem sum_of_decimals_as_fraction :
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00010 = 2469 / 10000 :=
by
  sorry

end sum_of_decimals_as_fraction_l134_134361


namespace tony_average_time_l134_134573

-- Definitions based on the conditions
def distance_to_store : ℕ := 4 -- in miles
def walking_speed : ℕ := 2 -- in MPH
def running_speed : ℕ := 10 -- in MPH

-- Conditions
def time_walking : ℕ := (distance_to_store / walking_speed) * 60 -- in minutes
def time_running : ℕ := (distance_to_store / running_speed) * 60 -- in minutes

def total_time : ℕ := time_walking + 2 * time_running -- Total time spent in minutes
def number_of_days : ℕ := 3 -- Number of days

def average_time : ℕ := total_time / number_of_days -- Average time in minutes

-- Statement to prove
theorem tony_average_time : average_time = 56 := by 
  sorry

end tony_average_time_l134_134573


namespace monotone_intervals_range_of_t_for_three_roots_l134_134519

def f (t x : ℝ) : ℝ := x^3 - 2 * x^2 + x + t

def f_prime (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 1

-- 1. Monotonic intervals
theorem monotone_intervals (t : ℝ) :
  (∀ x, f_prime x > 0 → x < 1/3 ∨ x > 1) ∧
  (∀ x, f_prime x < 0 → 1/3 < x ∧ x < 1) :=
sorry

-- 2. Range of t for three real roots
theorem range_of_t_for_three_roots (t : ℝ) :
  (∃ a b : ℝ, f t a = 0 ∧ f t b = 0 ∧ a ≠ b ∧
   a = 1/3 ∧ b = 1 ∧
   -4/27 + t > 0 ∧ t < 0) :=
sorry

end monotone_intervals_range_of_t_for_three_roots_l134_134519


namespace weighted_average_remaining_two_l134_134300

theorem weighted_average_remaining_two (avg_10 : ℝ) (avg_2 : ℝ) (avg_3 : ℝ) (avg_3_next : ℝ) :
  avg_10 = 4.25 ∧ avg_2 = 3.4 ∧ avg_3 = 3.85 ∧ avg_3_next = 4.7 →
  (42.5 - (2 * 3.4 + 3 * 3.85 + 3 * 4.7)) / 2 = 5.025 :=
by
  intros
  sorry

end weighted_average_remaining_two_l134_134300


namespace geometric_sequence_sum_l134_134305

variable {a : ℕ → ℕ}

-- Defining the geometric sequence and the conditions
def is_geometric_sequence (a : ℕ → ℕ) (q : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def condition1 (a : ℕ → ℕ) : Prop :=
  a 1 = 3

def condition2 (a : ℕ → ℕ) : Prop :=
  a 1 + a 3 + a 5 = 21

-- The main theorem
theorem geometric_sequence_sum (a : ℕ → ℕ) (q : ℕ) 
  (h1 : condition1 a) (h2: condition2 a) (hq : is_geometric_sequence a q) : 
  a 3 + a 5 + a 7 = 42 := 
sorry

end geometric_sequence_sum_l134_134305


namespace count_japanese_stamps_l134_134568

theorem count_japanese_stamps (total_stamps : ℕ) (perc_chinese perc_us : ℕ) (h1 : total_stamps = 100) 
  (h2 : perc_chinese = 35) (h3 : perc_us = 20): 
  total_stamps - ((perc_chinese * total_stamps / 100) + (perc_us * total_stamps / 100)) = 45 :=
by
  sorry

end count_japanese_stamps_l134_134568


namespace percent_of_N_in_M_l134_134760

theorem percent_of_N_in_M (N M : ℝ) (hM : M ≠ 0) : (N / M) * 100 = 100 * N / M :=
by
  sorry

end percent_of_N_in_M_l134_134760


namespace range_of_a_l134_134384

theorem range_of_a {a : ℝ} (h : ∃ x : ℝ, (a+2)/(x+1) = 1 ∧ x ≤ 0) :
  a ≤ -1 ∧ a ≠ -2 := 
sorry

end range_of_a_l134_134384


namespace bear_population_l134_134490

theorem bear_population (black_bears white_bears brown_bears total_bears : ℕ) 
(h1 : black_bears = 60)
(h2 : white_bears = black_bears / 2)
(h3 : brown_bears = black_bears + 40) :
total_bears = black_bears + white_bears + brown_bears :=
sorry

end bear_population_l134_134490


namespace intersecting_diagonals_of_parallelogram_l134_134297

theorem intersecting_diagonals_of_parallelogram (A C : ℝ × ℝ) (hA : A = (2, -3)) (hC : C = (14, 9)) :
    ∃ M : ℝ × ℝ, M = (8, 3) ∧ M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) :=
by {
  sorry
}

end intersecting_diagonals_of_parallelogram_l134_134297


namespace exists_colored_triangle_l134_134900

structure Point := (x : ℝ) (y : ℝ)
inductive Color
| red
| blue

def collinear (a b c : Point) : Prop :=
  (b.y - a.y) * (c.x - a.x) = (c.y - a.y) * (b.x - a.x)
  
def same_color_triangle_exists (S : Finset Point) (color : Point → Color) : Prop :=
  ∃ (A B C : Point), A ∈ S ∧ B ∈ S ∧ C ∈ S ∧
                    (color A = color B ∧ color B = color C) ∧
                    ¬ collinear A B C ∧
                    (∃ (X Y Z : Point), 
                      ((X ∈ S ∧ color X ≠ color A ∧ (X ≠ A ∧ X ≠ B ∧ X ≠ C)) ∧ 
                       (Y ∈ S ∧ color Y ≠ color A ∧ (Y ≠ A ∧ Y ≠ B ∧ Y ≠ C)) ∧
                       (Z ∈ S ∧ color Z ≠ color A ∧ (Z ≠ A ∧ Z ≠ B ∧ Z ≠ C)) → 
                       False))

theorem exists_colored_triangle 
  (S : Finset Point) (h1 : 5 ≤ S.card) (color : Point → Color) 
  (h2 : ∀ (A B C : Point), A ∈ S → B ∈ S → C ∈ S → (color A = color B ∧ color B = color C) → ¬ collinear A B C) 
  : same_color_triangle_exists S color :=
sorry

end exists_colored_triangle_l134_134900


namespace couscous_dishes_l134_134214

def dishes (a b c d : ℕ) : ℕ := (a + b + c) / d

theorem couscous_dishes :
  dishes 7 13 45 5 = 13 :=
by
  unfold dishes
  sorry

end couscous_dishes_l134_134214


namespace find_m_l134_134091

noncomputable def a_seq (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1 : ℝ) * d

noncomputable def S_n (a d : ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a + (n - 1 : ℝ) * d)

theorem find_m (a d : ℝ) (m : ℕ) 
  (h1 : a_seq a d (m-1) + a_seq a d (m+1) - a = 0)
  (h2 : S_n a d (2*m - 1) = 38) : 
  m = 10 := 
sorry

end find_m_l134_134091


namespace aira_rubber_bands_l134_134583

variable (S A J : ℕ)

-- Conditions
def conditions (S A J : ℕ) : Prop :=
  S = A + 5 ∧ A = J - 1 ∧ S + A + J = 18

-- Proof problem
theorem aira_rubber_bands (S A J : ℕ) (h : conditions S A J) : A = 4 :=
by
  -- introduce the conditions
  obtain ⟨h₁, h₂, h₃⟩ := h
  -- use sorry to skip the proof
  sorry

end aira_rubber_bands_l134_134583


namespace fill_buckets_lcm_l134_134399

theorem fill_buckets_lcm :
  (∀ (A B C : ℕ), (2 / 3 : ℚ) * A = 90 ∧ (1 / 2 : ℚ) * B = 120 ∧ (3 / 4 : ℚ) * C = 150 → lcm A (lcm B C) = 1200) :=
by
  sorry

end fill_buckets_lcm_l134_134399


namespace prob_primes_1_to_30_l134_134624

-- Define the set of integers from 1 to 30
def set_1_to_30 : Set ℕ := { n | 1 ≤ n ∧ n ≤ 30 }

-- Define the set of prime numbers from 1 to 30
def primes_1_to_30 : Set ℕ := { n | n ∈ set_1_to_30 ∧ Nat.Prime n }

-- Define the number of ways to choose 2 items from a set of size n
def choose (n k : ℕ) := n * (n - 1) / 2

-- The goal is to prove that the probability of choosing 2 prime numbers from the set is 1/10
theorem prob_primes_1_to_30 : (choose (Set.card primes_1_to_30) 2) / (choose (Set.card set_1_to_30) 2) = 1 / 10 :=
by
  sorry

end prob_primes_1_to_30_l134_134624


namespace solve_equation_l134_134432

theorem solve_equation 
  (x : ℚ)
  (h : (x^2 + 3*x + 4)/(x + 5) = x + 6) :
  x = -13/4 := 
by
  sorry

end solve_equation_l134_134432


namespace chickens_pigs_legs_l134_134880

variable (x : ℕ)

-- Define the conditions
def sum_chickens_pigs (x : ℕ) : Prop := x + (70 - x) = 70
def total_legs (x : ℕ) : Prop := 2 * x + 4 * (70 - x) = 196

-- Main theorem to prove the given mathematical statement
theorem chickens_pigs_legs (x : ℕ) (h1 : sum_chickens_pigs x) (h2 : total_legs x) : (2 * x + 4 * (70 - x) = 196) :=
by sorry

end chickens_pigs_legs_l134_134880


namespace pet_store_initial_gerbils_l134_134803

-- Define sold gerbils
def sold_gerbils : ℕ := 69

-- Define left gerbils
def left_gerbils : ℕ := 16

-- Define the initial number of gerbils
def initial_gerbils : ℕ := sold_gerbils + left_gerbils

-- State the theorem to be proved
theorem pet_store_initial_gerbils : initial_gerbils = 85 := by
  -- This is where the proof would go
  sorry

end pet_store_initial_gerbils_l134_134803


namespace find_y_l134_134755

theorem find_y (y : ℤ) (h : (15 + 24 + y) / 3 = 23) : y = 30 :=
by
  sorry

end find_y_l134_134755


namespace CarrieSpent_l134_134346

variable (CostPerShirt NumberOfShirts : ℝ)

def TotalCost (CostPerShirt NumberOfShirts : ℝ) : ℝ :=
  CostPerShirt * NumberOfShirts

theorem CarrieSpent {CostPerShirt NumberOfShirts : ℝ} 
  (h1 : CostPerShirt = 9.95) 
  (h2 : NumberOfShirts = 20) : 
  TotalCost CostPerShirt NumberOfShirts = 199.00 :=
by
  sorry

end CarrieSpent_l134_134346


namespace product_neg_six_l134_134197

theorem product_neg_six (m b : ℝ)
  (h1 : m = 2)
  (h2 : b = -3) : m * b < -3 := by
-- Proof skipped
sorry

end product_neg_six_l134_134197


namespace proof_of_A_inter_complement_B_l134_134974

variable (U : Set Nat) 
variable (A B : Set Nat)

theorem proof_of_A_inter_complement_B :
    (U = {1, 2, 3, 4}) →
    (B = {1, 2}) →
    (compl (A ∪ B) = {4}) →
    (A ∩ compl B = {3}) :=
by
  intros hU hB hCompl
  sorry

end proof_of_A_inter_complement_B_l134_134974


namespace find_side_c_l134_134280

theorem find_side_c (a C S : ℝ) (ha : a = 3) (hC : C = 120) (hS : S = (15 * Real.sqrt 3) / 4) : 
  ∃ (c : ℝ), c = 7 :=
by
  sorry

end find_side_c_l134_134280


namespace scheduling_arrangements_l134_134067

-- We want to express this as a problem to prove the number of scheduling arrangements.

theorem scheduling_arrangements (n : ℕ) (h : n = 6) :
  (Nat.choose 6 1) * (Nat.choose 5 1) * (Nat.choose 4 2) = 180 := by
  sorry

end scheduling_arrangements_l134_134067


namespace trigonometric_identity_l134_134213

theorem trigonometric_identity : 
  (Real.sin (42 * Real.pi / 180) * Real.cos (18 * Real.pi / 180) - 
   Real.cos (138 * Real.pi / 180) * Real.cos (72 * Real.pi / 180)) = 
  (Real.sqrt 3 / 2) :=
by
  sorry

end trigonometric_identity_l134_134213


namespace arithmetic_seq_general_term_geometric_seq_general_term_sequence_sum_l134_134441

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℕ := 2^n

def seq_sum (n : ℕ) (seq : ℕ → ℕ) : ℕ :=
  (Finset.range n).sum seq

noncomputable def T_n (n : ℕ) : ℕ :=
  seq_sum n (λ i => (a_n (i + 1) + 1) * b_n (i + 1))

theorem arithmetic_seq_general_term (n : ℕ) : a_n n = 2 * n - 1 := by
  sorry

theorem geometric_seq_general_term (n : ℕ) : b_n n = 2^n := by
  sorry

theorem sequence_sum (n : ℕ) : T_n n = (n - 1) * 2^(n+2) + 4 := by
  sorry

end arithmetic_seq_general_term_geometric_seq_general_term_sequence_sum_l134_134441


namespace find_m_of_ellipse_l134_134381

theorem find_m_of_ellipse (m : ℝ) (h₀ : m > 0) 
  (h₁ : ∃ (x y : ℝ), x^2 / 25 + y^2 / m^2 = 1) 
  (h₂ : ∀ c, (c = 4) → (∃ a b : ℝ, a = 5 ∧ b = m ∧ 25 = m^2 + 16)) :
  m = 3 :=
by
  sorry

end find_m_of_ellipse_l134_134381


namespace bobby_paid_for_shoes_l134_134958

theorem bobby_paid_for_shoes :
  let mold_cost := 250
  let hourly_labor_rate := 75
  let hours_worked := 8
  let discount_rate := 0.80
  let materials_cost := 150
  let tax_rate := 0.10

  let labor_cost := hourly_labor_rate * hours_worked
  let discounted_labor_cost := discount_rate * labor_cost
  let total_cost_before_tax := mold_cost + discounted_labor_cost + materials_cost
  let tax := total_cost_before_tax * tax_rate
  let total_cost_with_tax := total_cost_before_tax + tax

  total_cost_with_tax = 968 :=
by
  sorry

end bobby_paid_for_shoes_l134_134958


namespace one_fourth_of_8_point_4_is_21_over_10_l134_134245

theorem one_fourth_of_8_point_4_is_21_over_10 : (8.4 / 4 : ℚ) = 21 / 10 := 
by
  sorry

end one_fourth_of_8_point_4_is_21_over_10_l134_134245


namespace length_of_second_parallel_side_l134_134009

-- Define the given conditions
def parallel_side1 : ℝ := 20
def distance : ℝ := 14
def area : ℝ := 266

-- Define the theorem to prove the length of the second parallel side
theorem length_of_second_parallel_side (x : ℝ) 
  (h : area = (1 / 2) * (parallel_side1 + x) * distance) : 
  x = 18 :=
sorry

end length_of_second_parallel_side_l134_134009


namespace distribution_of_K_l134_134478

theorem distribution_of_K (x y z : ℕ) 
  (h_total : x + y + z = 370)
  (h_diff : y + z - x = 50)
  (h_prop : x * z = y^2) :
  x = 160 ∧ y = 120 ∧ z = 90 := by
  sorry

end distribution_of_K_l134_134478


namespace probability_multiple_of_12_and_even_l134_134538

open Set

-- Definitions based on conditions
def chosen_set : Set ℕ := {4, 6, 8, 9}

def pairs (s : Set ℕ) : Set (ℕ × ℕ) := 
  { p | p.1 ∈ s ∧ p.2 ∈ s ∧ p.1 ≠ p.2 }

def is_multiple_of_12 (n : ℕ) : Prop := n % 12 = 0

def has_even_number (p : ℕ × ℕ) : Prop := 
  p.1 % 2 = 0 ∨ p.2 % 2 = 0

-- Target theorem to prove
theorem probability_multiple_of_12_and_even : 
  let valid_pairs := { p ∈ pairs chosen_set | is_multiple_of_12 (p.1 * p.2) ∧ has_even_number p } in
  (valid_pairs.card : ℚ) / (pairs chosen_set).card = 2 / 3 :=
by
  sorry

end probability_multiple_of_12_and_even_l134_134538


namespace determine_parabola_equation_l134_134532

-- Define the conditions
def focus_on_line (focus : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, focus = (k - 2, k / 2 - 1)

-- Define the result equations
def is_standard_equation (eq : ℝ → ℝ → Prop) : Prop :=
  (∀ x y : ℝ, eq x y → x^2 = 4 * y) ∨ (∀ x y : ℝ, eq x y → y^2 = -8 * x)

-- Define the theorem stating that given the condition,
-- the standard equation is one of the two forms
theorem determine_parabola_equation (focus : ℝ × ℝ) (H : focus_on_line focus) :
  ∃ eq : ℝ → ℝ → Prop, is_standard_equation eq :=
sorry

end determine_parabola_equation_l134_134532


namespace probability_two_primes_l134_134629

theorem probability_two_primes (S : Finset ℕ) (S = {1, 2, ..., 30}) 
  (primes : Finset ℕ) (primes = {p ∈ S | Prime p}) :
  (primes.card = 10) →
  (S.card = 30) →
  (nat.choose 2) (primes.card) / (nat.choose 2) (S.card) = 1 / 10 :=
begin
  sorry
end

end probability_two_primes_l134_134629


namespace cage_cost_correct_l134_134733

def cost_of_cat_toy : Real := 10.22
def total_cost_of_purchases : Real := 21.95
def cost_of_cage : Real := total_cost_of_purchases - cost_of_cat_toy

theorem cage_cost_correct : cost_of_cage = 11.73 := by
  sorry

end cage_cost_correct_l134_134733


namespace monotonically_increasing_interval_l134_134602

open Real

/-- The monotonically increasing interval of the function y = (cos x + sin x) * cos (x - π / 2)
    is [kπ - π / 8, kπ + 3π / 8] for k ∈ ℤ. -/
theorem monotonically_increasing_interval (k : ℤ) :
  ∀ x : ℝ, (cos x + sin x) * cos (x - π / 2) = y →
  (k * π - π / 8) ≤ x ∧ x ≤ (k * π + 3 * π / 8) := 
sorry

end monotonically_increasing_interval_l134_134602


namespace brooke_added_balloons_l134_134230

-- Definitions stemming from the conditions
def initial_balloons_brooke : Nat := 12
def added_balloons_brooke (x : Nat) : Nat := x
def initial_balloons_tracy : Nat := 6
def added_balloons_tracy : Nat := 24
def total_balloons_tracy : Nat := initial_balloons_tracy + added_balloons_tracy
def final_balloons_tracy : Nat := total_balloons_tracy / 2
def total_balloons (x : Nat) : Nat := initial_balloons_brooke + added_balloons_brooke x + final_balloons_tracy

-- Mathematical proof problem
theorem brooke_added_balloons (x : Nat) :
  total_balloons x = 35 → x = 8 := by
  sorry

end brooke_added_balloons_l134_134230


namespace gain_percent_is_87_point_5_l134_134530

noncomputable def gain_percent (C S : ℝ) : ℝ :=
  ((S - C) / C) * 100

theorem gain_percent_is_87_point_5 {C S : ℝ} (h : 75 * C = 40 * S) :
  gain_percent C S = 87.5 :=
by
  sorry

end gain_percent_is_87_point_5_l134_134530


namespace binomial_fermat_l134_134065

theorem binomial_fermat (p : ℕ) (a b : ℤ) (hp : p.Prime) : 
  ((a + b)^p - a^p - b^p) % p = 0 := by
  sorry

end binomial_fermat_l134_134065


namespace necessary_and_sufficient_condition_l134_134864

theorem necessary_and_sufficient_condition (a b : ℝ) (h : a * b ≠ 0) : 
  a - b = 1 ↔ a^3 - b^3 - a * b - a^2 - b^2 = 0 := by
  sorry

end necessary_and_sufficient_condition_l134_134864


namespace range_of_a_l134_134386

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, (a+2)/(x+1) = 1 ∧ x ≤ 0) ↔ (a ≤ -1 ∧ a ≠ -2) :=
by
  sorry

end range_of_a_l134_134386


namespace find_alpha_plus_beta_l134_134521

variable (α β : ℝ)

def condition_1 : Prop := α^3 - 3*α^2 + 5*α = 1
def condition_2 : Prop := β^3 - 3*β^2 + 5*β = 5

theorem find_alpha_plus_beta (h1 : condition_1 α) (h2 : condition_2 β) : α + β = 2 := 
  sorry

end find_alpha_plus_beta_l134_134521


namespace sample_size_is_59_l134_134945

def totalStudents : Nat := 295
def samplingRatio : Nat := 5

theorem sample_size_is_59 : totalStudents / samplingRatio = 59 := 
by
  sorry

end sample_size_is_59_l134_134945


namespace best_chart_for_temperature_changes_l134_134621

def Pie_chart := "Represent the percentage of parts in the whole."
def Line_chart := "Represent changes over time."
def Bar_chart := "Show the specific number of each item."

theorem best_chart_for_temperature_changes : 
  "The best statistical chart to use for understanding temperature changes throughout a day" = Line_chart :=
by
  sorry

end best_chart_for_temperature_changes_l134_134621


namespace a_n_is_perfect_square_l134_134976

def sequence_c (n : ℕ) : ℤ :=
  if n = 0 then 1
  else if n = 1 then 0
  else if n = 2 then 2005
  else -3 * sequence_c (n - 2) - 4 * sequence_c (n - 3) + 2008

def sequence_a (n : ℕ) :=
  if n < 2 then 0
  else 5 * (sequence_c (n + 2) - sequence_c n) * (502 - sequence_c (n - 1) - sequence_c (n - 2)) + (4 ^ n) * 2004 * 501

theorem a_n_is_perfect_square (n : ℕ) (h : n > 2) : ∃ k : ℤ, sequence_a n = k^2 :=
sorry

end a_n_is_perfect_square_l134_134976


namespace largest_first_term_geometric_progression_l134_134225

noncomputable def geometric_progression_exists (d : ℝ) : Prop :=
  ∃ (a : ℝ), a = 5 ∧ (a + d + 3) / a = (a + 2 * d + 15) / (a + d + 3)

theorem largest_first_term_geometric_progression : ∀ (d : ℝ), 
  d^2 + 6 * d - 36 = 0 → 
  ∃ (a : ℝ), a = 5 ∧ geometric_progression_exists d ∧ a = 5 ∧ 
  ∀ (a' : ℝ), geometric_progression_exists d → a' ≤ a :=
by intros d h; sorry

end largest_first_term_geometric_progression_l134_134225


namespace total_height_of_pipes_l134_134506

theorem total_height_of_pipes 
  (diameter : ℝ) (radius : ℝ) (total_pipes : ℕ) (first_row_pipes : ℕ) (second_row_pipes : ℕ) 
  (h : ℝ) 
  (h_diam : diameter = 10)
  (h_radius : radius = 5)
  (h_total_pipes : total_pipes = 5)
  (h_first_row : first_row_pipes = 2)
  (h_second_row : second_row_pipes = 3) :
  h = 10 + 5 * Real.sqrt 3 := 
sorry

end total_height_of_pipes_l134_134506


namespace find_a_l134_134691

/-- 
Given sets A and B defined by specific quadratic equations, 
if A ∪ B = A, then a ∈ (-∞, 0).
-/
theorem find_a :
  ∀ (a : ℝ),
    (A = {x : ℝ | x^2 - 3 * x + 2 = 0}) →
    (B = {x : ℝ | x^2 - 2 * a * x + a^2 - a = 0}) →
    (A ∪ B = A) →
    a < 0 :=
by
  sorry

end find_a_l134_134691


namespace log_product_identity_l134_134822

theorem log_product_identity :
    (Real.log 9 / Real.log 8) * (Real.log 32 / Real.log 9) = 5 / 3 := 
by 
  sorry

end log_product_identity_l134_134822


namespace kristine_travel_distance_l134_134409

theorem kristine_travel_distance :
  ∃ T : ℝ, T + T / 2 + T / 6 = 500 ∧ T = 300 := by
  sorry

end kristine_travel_distance_l134_134409


namespace rebus_solution_l134_134003

theorem rebus_solution :
  ∃ (A B C : ℕ), A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    (A*100 + B*10 + A) + (A*100 + B*10 + C) + (A*100 + C*10 + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by {
  sorry
}

end rebus_solution_l134_134003


namespace isosceles_triangle_perimeter_l134_134990

noncomputable theory

def is_isosceles_triangle (a b c : ℝ) : Prop := 
  a = b ∨ b = c ∨ a = c

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem isosceles_triangle_perimeter (a b : ℝ) (h_iso : is_isosceles_triangle a b 4) (h_valid : is_valid_triangle a b 4) :
  perimeter a b 4 = 10 :=
  sorry

end isosceles_triangle_perimeter_l134_134990


namespace macaroon_weight_l134_134549

theorem macaroon_weight (bakes : ℕ) (packs : ℕ) (bags_after_eat : ℕ) (remaining_weight : ℕ) (macaroons_per_bag : ℕ) (weight_per_bag : ℕ)
  (H1 : bakes = 12) 
  (H2 : packs = 4)
  (H3 : bags_after_eat = 3)
  (H4 : remaining_weight = 45)
  (H5 : macaroons_per_bag = bakes / packs) 
  (H6 : weight_per_bag = remaining_weight / bags_after_eat) :
  ∀ (weight_per_macaroon : ℕ), weight_per_macaroon = weight_per_bag / macaroons_per_bag → weight_per_macaroon = 5 :=
by
  sorry -- Proof will come here, not required as per instructions

end macaroon_weight_l134_134549


namespace sqrt_expression_meaningful_l134_134717

theorem sqrt_expression_meaningful {x : ℝ} : (2 * x - 4) ≥ 0 → x ≥ 2 :=
by
  intro h
  sorry

end sqrt_expression_meaningful_l134_134717


namespace triangle_A1B1C1_sides_l134_134854

theorem triangle_A1B1C1_sides
  (a b c x y z R : ℝ) 
  (h_positive_a : a > 0)
  (h_positive_b : b > 0)
  (h_positive_c : c > 0)
  (h_positive_x : x > 0)
  (h_positive_y : y > 0)
  (h_positive_z : z > 0)
  (h_positive_R : R > 0) :
  (↑a * ↑y / (2 * ↑R), ↑b * ↑z / (2 * ↑R), ↑c * ↑x / (2 * ↑R)) = (↑c * ↑x / (2 * ↑R), ↑a * ↑y / (2 * ↑R), ↑b * ↑z / (2 * ↑R)) :=
by sorry

end triangle_A1B1C1_sides_l134_134854


namespace max_days_proof_l134_134758

-- Define a graph with n vertices and bidirectional edges
structure Graph (V : Type) :=
  (adj : V → V → Prop)
  (sym : ∀ {u v : V}, adj u v → adj v u)

-- Define the problem conditions
def airport_problem (V : Type) (n : ℕ) [finite V] [fintype V] (G : Graph V) : Prop :=
  n = fintype.card V ∧
  n ≥ 3 ∧
  (∃ D, ∀ v : V, D v = card (finset.filter (G.adj v) (finset.univ : finset V))) ∧
  (∀ (t : ℕ), t < n - 3 → ∃ (v : V), D v = max (λ v : V, D v))

-- Define the maximum number of days for each n
def max_days (n : ℕ) : ℕ :=
  if n = 3 then 1 else n - 3

-- Lean theorem stating the equivalence of condition and answer
theorem max_days_proof (V : Type) (n : ℕ) [finite V] [fintype V] (G : Graph V)
  (cond : airport_problem V n G) : 
  cond → max_days n = (if n = 3 then 1 else n - 3) :=
sorry

end max_days_proof_l134_134758


namespace power_identity_l134_134858

theorem power_identity (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 3) : x^(3 * a + 2 * b) = 72 := 
  sorry

end power_identity_l134_134858


namespace solution_set_of_inequality_l134_134847

theorem solution_set_of_inequality :
  ∀ x : ℝ, (x-50)*(60-x) > 0 ↔ 50 < x ∧ x < 60 :=
by
  sorry

end solution_set_of_inequality_l134_134847


namespace distinct_colorings_l134_134662

def sections : ℕ := 6
def red_count : ℕ := 3
def blue_count : ℕ := 1
def green_count : ℕ := 1
def yellow_count : ℕ := 1

def permutations_without_rotation : ℕ := Nat.factorial sections / 
  (Nat.factorial red_count * Nat.factorial blue_count * Nat.factorial green_count * Nat.factorial yellow_count)

def rotational_symmetry : ℕ := permutations_without_rotation / sections

theorem distinct_colorings (rotational_symmetry) : rotational_symmetry = 20 :=
  sorry

end distinct_colorings_l134_134662


namespace find_somu_age_l134_134474

noncomputable def somu_age (S F : ℕ) : Prop :=
  S = (1/3 : ℝ) * F ∧ S - 6 = (1/5 : ℝ) * (F - 6)

theorem find_somu_age {S F : ℕ} (h : somu_age S F) : S = 12 :=
by sorry

end find_somu_age_l134_134474


namespace sum_of_coefficients_l134_134379

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def polynomial (a : ℝ) (x : ℝ) : ℝ :=
  (2 + a * x) * (1 + x)^5

def x2_coefficient_condition (a : ℝ) : Prop :=
  2 * binomial_coefficient 5 2 + a * binomial_coefficient 5 1 = 15

theorem sum_of_coefficients (a : ℝ) (h : x2_coefficient_condition a) : 
  polynomial a 1 = 64 := 
sorry

end sum_of_coefficients_l134_134379


namespace row_seat_notation_l134_134089

-- Define that the notation (4, 5) corresponds to "Row 4, Seat 5"
def notation_row_seat := (4, 5)

-- Prove that "Row 5, Seat 4" should be denoted as (5, 4)
theorem row_seat_notation : (5, 4) = (5, 4) :=
by sorry

end row_seat_notation_l134_134089


namespace probability_two_primes_is_1_over_29_l134_134630

open Finset

noncomputable def primes_upto_30 : Finset ℕ := filter Nat.Prime (range 31)

def total_pairs : ℕ := (range 31).card.choose 2

def prime_pairs : ℕ := (primes_upto_30).card.choose 2

theorem probability_two_primes_is_1_over_29 :
  prime_pairs.to_rat / total_pairs.to_rat = (1 : ℚ) / 29 := sorry

end probability_two_primes_is_1_over_29_l134_134630


namespace find_b_c_l134_134861

-- Definitions and the problem statement
theorem find_b_c (b c : ℝ) (x1 x2 : ℝ) (h1 : x1 = 1) (h2 : x2 = -2) 
  (h_eq : ∀ x, x^2 - b * x + c = (x - x1) * (x - x2)) :
  b = -1 ∧ c = -2 :=
by
  sorry

end find_b_c_l134_134861


namespace smallest_number_condition_l134_134936

theorem smallest_number_condition :
  ∃ n, 
  (n > 0) ∧ 
  (∀ k, k < n → (n - 3) % 12 = 0 ∧ (n - 3) % 16 = 0 ∧ (n - 3) % 18 = 0 ∧ (n - 3) % 21 = 0 ∧ (n - 3) % 28 = 0 → k = 0) ∧
  (n - 3) % 12 = 0 ∧
  (n - 3) % 16 = 0 ∧
  (n - 3) % 18 = 0 ∧
  (n - 3) % 21 = 0 ∧
  (n - 3) % 28 = 0 ∧
  n = 1011 :=
sorry

end smallest_number_condition_l134_134936


namespace translation_correct_l134_134266

-- Define the points in the Cartesian coordinate system
structure Point where
  x : ℤ
  y : ℤ

-- Given points A and B
def A : Point := { x := -1, y := 0 }
def B : Point := { x := 1, y := 2 }

-- Translated point A' (A₁)
def A₁ : Point := { x := 2, y := -1 }

-- Define the translation applied to a point
def translate (p : Point) (v : Point) : Point :=
  { x := p.x + v.x, y := p.y + v.y }

-- Calculate the translation vector from A to A'
def translationVector : Point :=
  { x := A₁.x - A.x, y := A₁.y - A.y }

-- Define the expected point B' (B₁)
def B₁ : Point := { x := 4, y := 1 }

-- Theorem statement
theorem translation_correct :
  translate B translationVector = B₁ :=
by
  -- proof goes here
  sorry

end translation_correct_l134_134266


namespace evaluate_expression_l134_134357

theorem evaluate_expression :
  ((Int.ceil ((21 : ℚ) / 5 - Int.ceil ((35 : ℚ) / 23))) : ℚ) /
  (Int.ceil ((35 : ℚ) / 5 + Int.ceil ((5 * 23 : ℚ) / 35))) = 3 / 11 := by
  sorry

end evaluate_expression_l134_134357


namespace sales_this_month_l134_134905

-- Define the given conditions
def price_large := 60
def price_small := 30
def num_large_last_month := 8
def num_small_last_month := 4

-- Define the computation of total sales for last month
def sales_last_month : ℕ :=
  price_large * num_large_last_month + price_small * num_small_last_month

-- State the theorem to prove the sales this month
theorem sales_this_month : sales_last_month * 2 = 1200 :=
by
  -- Proof will follow, for now we use sorry as a placeholder
  sorry

end sales_this_month_l134_134905


namespace second_bag_roger_is_3_l134_134748

def total_candy_sandra := 2 * 6
def total_candy_roger := total_candy_sandra + 2
def first_bag_roger := 11
def second_bag_roger := total_candy_roger - first_bag_roger

theorem second_bag_roger_is_3 : second_bag_roger = 3 :=
by
  sorry

end second_bag_roger_is_3_l134_134748


namespace green_notebook_cost_each_l134_134062

-- Definitions for conditions:
def num_notebooks := 4
def num_green_notebooks := 2
def num_black_notebooks := 1
def num_pink_notebooks := 1
def total_cost := 45
def black_notebook_cost := 15
def pink_notebook_cost := 10

-- Define the problem statement:
theorem green_notebook_cost_each : 
  (2 * g + black_notebook_cost + pink_notebook_cost = total_cost) → 
  g = 10 := 
by 
  intros h
  sorry

end green_notebook_cost_each_l134_134062


namespace ratio_siblings_l134_134179

theorem ratio_siblings (M J C : ℕ) 
  (hM : M = 60)
  (hJ : J = 4 * M - 60)
  (hJ_C : J = C + 135) :
  (C : ℚ) / M = 3 / 4 :=
by
  sorry

end ratio_siblings_l134_134179


namespace digit_B_in_4B52B_divisible_by_9_l134_134303

theorem digit_B_in_4B52B_divisible_by_9 (B : ℕ) (h : (2 * B + 11) % 9 = 0) : B = 8 :=
by {
  sorry
}

end digit_B_in_4B52B_divisible_by_9_l134_134303


namespace smallest_four_digit_equiv_8_mod_9_l134_134776

theorem smallest_four_digit_equiv_8_mod_9 :
  ∃ n : ℕ, n % 9 = 8 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 9 = 8 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m :=
sorry

end smallest_four_digit_equiv_8_mod_9_l134_134776


namespace profit_per_meter_is_15_l134_134952

def sellingPrice (meters : ℕ) : ℕ := 
    if meters = 85 then 8500 else 0

def costPricePerMeter : ℕ := 85

def totalCostPrice (meters : ℕ) : ℕ := 
    meters * costPricePerMeter

def totalProfit (meters : ℕ) (sellingPrice : ℕ) (costPrice : ℕ) : ℕ := 
    sellingPrice - costPrice

def profitPerMeter (profit : ℕ) (meters : ℕ) : ℕ := 
    profit / meters

theorem profit_per_meter_is_15 : profitPerMeter (totalProfit 85 (sellingPrice 85) (totalCostPrice 85)) 85 = 15 := 
by sorry

end profit_per_meter_is_15_l134_134952


namespace passing_marks_l134_134930

theorem passing_marks (T P : ℝ) (h1 : 0.30 * T = P - 30) (h2 : 0.45 * T = P + 15) : P = 120 := 
by
  sorry

end passing_marks_l134_134930


namespace sum_of_decimals_as_fraction_l134_134359

theorem sum_of_decimals_as_fraction :
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00010 = 2469 / 10000 :=
by
  sorry

end sum_of_decimals_as_fraction_l134_134359


namespace geom_series_common_ratio_l134_134454

theorem geom_series_common_ratio (a r S : ℝ) (hS : S = a / (1 - r)) (hNewS : (ar^3) / (1 - r) = S / 27) : r = 1 / 3 :=
by
  sorry

end geom_series_common_ratio_l134_134454


namespace days_elapsed_l134_134125

theorem days_elapsed
  (initial_amount : ℕ)
  (daily_spending : ℕ)
  (total_savings : ℕ)
  (doubling_factor : ℕ)
  (additional_amount : ℕ)
  :
  initial_amount = 50 →
  daily_spending = 15 →
  doubling_factor = 2 →
  additional_amount = 10 →
  2 * (initial_amount - daily_spending) * total_savings + additional_amount = 500 →
  total_savings = 7 :=
by
  intros h_initial h_spending h_doubling h_additional h_total
  sorry

end days_elapsed_l134_134125


namespace martingale_l134_134052

variables {n : ℕ} (η : Fin n → ℝ) (f : (nat → ℝ) → (nat → ℝ) → ℝ)

def is_martingale (ξ : Fin n → ℝ) (ℱ : Fin n → measurable_space ℝ) : Prop :=
  ∀⦃k⦄, 1 ≤ k → k < n → 
  measurable_space.sub_measurable_space (ℱ (k : Fin n)) (ℱ (k + 1 : Fin n)) →
  forall (y : ℝ), ∫ (λ ω, ξ k), P = ∫ (λ ω, conditional_expectation (ξ (k+1)) (ℱ (k + 1))).to_fun ω, P

def ξ : Fin n → ℝ
| 0 => η 0
| k+1 => ∑ i in Finset.range (k + 1), f (η 0 .. η i) (η (i + 1))

theorem martingale : is_martingale ξ _ :=
sorry

end martingale_l134_134052


namespace remaining_macaroons_weight_l134_134407

-- Problem conditions
variables (macaroons_per_bake : ℕ) (weight_per_macaroon : ℕ) (bags : ℕ) (macaroons_eaten : ℕ)

-- Definitions from problem conditions
def macaroons_per_bake := 12
def weight_per_macaroon := 5
def bags := 4
def macaroons_per_bag := macaroons_per_bake / bags
def macaroons_eaten := macaroons_per_bag

-- Lean theorem
theorem remaining_macaroons_weight : (macaroons_per_bake - macaroons_eaten) * weight_per_macaroon = 45 :=
by
  have h1 : macaroons_per_bag = 12 / 4 := rfl
  have h2 : macaroons_per_bag = 3 := by norm_num [h1]
  have h3 : macaroons_eaten = 3 := h2
  have h4 : macaroons_per_bake - macaroons_eaten = 12 - 3 := rfl
  have h5 : macaroons_per_bake - macaroons_eaten = 9 := by norm_num [h4]
  have h6 : (macaroons_per_bake - macaroons_eaten) * weight_per_macaroon = 9 * 5 := by rw [h5]
  calc
    (macaroons_per_bake - macaroons_eaten) * weight_per_macaroon = 9 * 5 : by rw [h6]
    ... = 45 : by norm_num

end remaining_macaroons_weight_l134_134407


namespace range_of_k_in_first_quadrant_l134_134698

theorem range_of_k_in_first_quadrant (k : ℝ) (h₁ : k ≠ -1) :
  (∃ x y : ℝ, y = k * x - 1 ∧ x + y - 1 = 0 ∧ x > 0 ∧ y > 0) ↔ 1 < k := by sorry

end range_of_k_in_first_quadrant_l134_134698


namespace find_t_l134_134169

theorem find_t (s t : ℝ) (h1 : 15 * s + 7 * t = 236) (h2 : t = 2 * s + 1) : t = 16.793 :=
by
  sorry

end find_t_l134_134169


namespace number_of_distinguishable_arrangements_l134_134982

-- Define the conditions
def num_blue_tiles : Nat := 1
def num_red_tiles : Nat := 2
def num_green_tiles : Nat := 3
def num_yellow_tiles : Nat := 2
def total_tiles : Nat := num_blue_tiles + num_red_tiles + num_green_tiles + num_yellow_tiles

-- The goal is to prove the number of distinguishable arrangements
theorem number_of_distinguishable_arrangements : 
  (Nat.factorial total_tiles) / ((Nat.factorial num_green_tiles) * 
                                (Nat.factorial num_red_tiles) * 
                                (Nat.factorial num_yellow_tiles) * 
                                (Nat.factorial num_blue_tiles)) = 1680 := by
  sorry

end number_of_distinguishable_arrangements_l134_134982


namespace last_two_digits_7_pow_2018_l134_134316

theorem last_two_digits_7_pow_2018 : 
  (7 ^ 2018) % 100 = 49 := 
sorry

end last_two_digits_7_pow_2018_l134_134316


namespace total_savings_over_12_weeks_l134_134584

-- Define the weekly savings and durations for each period
def weekly_savings_period_1 : ℕ := 5
def duration_period_1 : ℕ := 4

def weekly_savings_period_2 : ℕ := 10
def duration_period_2 : ℕ := 4

def weekly_savings_period_3 : ℕ := 20
def duration_period_3 : ℕ := 4

-- Define the total savings calculation for each period
def total_savings_period_1 : ℕ := weekly_savings_period_1 * duration_period_1
def total_savings_period_2 : ℕ := weekly_savings_period_2 * duration_period_2
def total_savings_period_3 : ℕ := weekly_savings_period_3 * duration_period_3

-- Prove that the total savings over 12 weeks equals $140.00
theorem total_savings_over_12_weeks : total_savings_period_1 + total_savings_period_2 + total_savings_period_3 = 140 := 
by 
  sorry

end total_savings_over_12_weeks_l134_134584


namespace domain_of_f_l134_134759

def domain_f := {x : ℝ | 2 * x - 3 > 0}

theorem domain_of_f : ∀ x : ℝ, x ∈ domain_f ↔ x > 3 / 2 := 
by
  intro x
  simp [domain_f]
  sorry

end domain_of_f_l134_134759


namespace ben_savings_l134_134126

theorem ben_savings:
  ∃ x : ℕ, (50 - 15) * x * 2 + 10 = 500 ∧ x = 7 :=
by
  -- Definitions based on conditions
  let daily_savings := 50 - 15
  have h1 : daily_savings = 35 := by norm_num
  let total_savings := daily_savings * x
  let doubled_savings := 2 * total_savings
  let final_savings := doubled_savings + 10

  -- Existence of x such that (50 - 15) * x * 2 + 10 = 500 and x = 7 
  use 7
  split
  { -- Show that the equation holds
    show final_savings = 500,
    calc
      final_savings = (daily_savings * 7 * 2) + 10 : by sorry
                   ... = 500 : by norm_num
  }
  { -- Show that x = 7
    refl
  }
  sorry

end ben_savings_l134_134126


namespace find_x_l134_134258

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 80) : x = 26 :=
by 
  sorry

end find_x_l134_134258


namespace smallest_four_digit_int_equiv_8_mod_9_l134_134778

theorem smallest_four_digit_int_equiv_8_mod_9 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 9 = 8 ∧ n = 1007 := 
by
  sorry

end smallest_four_digit_int_equiv_8_mod_9_l134_134778


namespace sarah_marry_age_l134_134239

/-- Sarah is 9 years old. -/
def Sarah_age : ℕ := 9

/-- Sarah's name has 5 letters. -/
def Sarah_name_length : ℕ := 5

/-- The game's rule is to add the number of letters in the player's name 
    to twice the player's age. -/
def game_rule (name_length age : ℕ) : ℕ :=
  name_length + 2 * age

/-- Prove that Sarah will get married at the age of 23. -/
theorem sarah_marry_age : game_rule Sarah_name_length Sarah_age = 23 := 
  sorry

end sarah_marry_age_l134_134239


namespace smallest_y_l134_134774

theorem smallest_y (y : ℤ) :
  (∃ k : ℤ, y^2 + 3*y + 7 = k*(y-2)) ↔ y = -15 :=
sorry

end smallest_y_l134_134774


namespace relationship_between_a_and_b_l134_134097

def ellipse_touching_hyperbola (a b : ℝ) :=
  ∀ x y : ℝ, ( (x / a) ^ 2 + (y / b) ^ 2 = 1 ∧ y = 1 / x → False )

  theorem relationship_between_a_and_b (a b : ℝ) :
  ellipse_touching_hyperbola a b →
  a * b = 2 :=
by
  sorry

end relationship_between_a_and_b_l134_134097


namespace carpet_rate_l134_134843

theorem carpet_rate (length breadth cost area: ℝ) (h₁ : length = 13) (h₂ : breadth = 9) (h₃ : cost = 1872) (h₄ : area = length * breadth) :
  cost / area = 16 := by
  sorry

end carpet_rate_l134_134843


namespace percentage_cats_less_dogs_l134_134400

theorem percentage_cats_less_dogs (C D F : ℕ) (h1 : C < D) (h2 : F = 2 * D) (h3 : C + D + F = 304) (h4 : F = 160) :
  ((D - C : ℕ) * 100 / D : ℕ) = 20 := 
sorry

end percentage_cats_less_dogs_l134_134400


namespace volume_is_six_l134_134040

-- Define the polygons and their properties
def right_triangle (a b c : ℝ) := (a^2 + b^2 = c^2 ∧ a > 0 ∧ b > 0 ∧ c > 0)
def rectangle (l w : ℝ) := (l > 0 ∧ w > 0)
def equilateral_triangle (s : ℝ) := (s > 0)

-- The given polygons
def A := right_triangle 1 2 (Real.sqrt 5)
def E := right_triangle 1 2 (Real.sqrt 5)
def F := right_triangle 1 2 (Real.sqrt 5)
def B := rectangle 1 2
def C := rectangle 2 3
def D := rectangle 1 3
def G := equilateral_triangle (Real.sqrt 5)

-- The volume of the polyhedron
-- Assume the largest rectangle C forms the base and a reasonable height
def volume_of_polyhedron : ℝ := 6

theorem volume_is_six : 
  (right_triangle 1 2 (Real.sqrt 5)) → 
  (rectangle 1 2) → 
  (rectangle 2 3) → 
  (rectangle 1 3) → 
  (equilateral_triangle (Real.sqrt 5)) → 
  volume_of_polyhedron = 6 := 
by 
  sorry

end volume_is_six_l134_134040


namespace arithmetic_progression_common_difference_and_first_terms_l134_134341

def sum (n : ℕ) : ℕ := 5 * n ^ 2
def Sn (a1 d n : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_progression_common_difference_and_first_terms:
  ∀ n : ℕ, Sn 5 10 n = sum n :=
by
  sorry

end arithmetic_progression_common_difference_and_first_terms_l134_134341


namespace probability_of_B_not_losing_is_70_l134_134479

-- Define the probabilities as given in the conditions
def prob_A_winning : ℝ := 0.30
def prob_draw : ℝ := 0.50

-- Define the probability of B not losing
def prob_B_not_losing : ℝ := 0.50 + (1 - prob_A_winning - prob_draw)

-- State the theorem
theorem probability_of_B_not_losing_is_70 :
  prob_B_not_losing = 0.70 := by
  sorry -- Proof to be filled in

end probability_of_B_not_losing_is_70_l134_134479


namespace find_a4_l134_134156

noncomputable def S : ℕ → ℤ
| 0 => 0
| 1 => -1
| n+1 => 3 * S n + 2^(n+1) - 3

def a : ℕ → ℤ
| 0 => 0
| 1 => -1
| n+1 => 3 * a n + 2^n

theorem find_a4 (h1 : ∀ n ≥ 2, S n = 3 * S (n - 1) + 2^n - 3) (h2 : a 1 = -1) : a 4 = 11 :=
by
  sorry

end find_a4_l134_134156


namespace range_of_t_l134_134388

noncomputable def f (t : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then x^2 + 2 * t * x + t^2 else x + 1 / x + t

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f t 0 ≤ f t x) ↔ (0 ≤ t ∧ t ≤ 2) :=
by sorry

end range_of_t_l134_134388


namespace number_of_balls_condition_l134_134217

theorem number_of_balls_condition (X : ℕ) (h1 : 25 - 20 = X - 25) : X = 30 :=
by
  sorry

end number_of_balls_condition_l134_134217


namespace general_term_formula_smallest_m_l134_134191

-- Define the arithmetic sequence and its sum condition
def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
def S_n (a : ℕ → ℝ) (n : ℕ) := n * a ((n - 1) / 2)

axiom S7_eq_7 : S_n a 7 = 7
axiom S15_eq_75 : S_n a 15 = 75

-- Derive the general term formula of the sequence
theorem general_term_formula : ∃ d, ∃ a_4 : ℝ, d = 1 ∧ a_4 = 1 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
begin
  sorry,
end

-- Define the sequence b_n
def b (n : ℕ) : ℝ := 2 * (n - 3) + 5

-- Define the sum T_n
def T_n (n : ℕ) : ℝ :=
  (∑ k in finset.range n, 1 / (b k * b (k + 1)))

-- Prove the smallest positive integer m such that T_n < m / 4
theorem smallest_m (n : ℕ) : ∃ m : ℕ, m = 2 ∧ ∀ n, T_n n < m / 4 :=
begin
  sorry,
end

end general_term_formula_smallest_m_l134_134191


namespace part1_part2_l134_134262

noncomputable def f (x m : ℝ) := |x + 1| + |m - x|

theorem part1 (x : ℝ) : (f x 3) ≥ 6 ↔ (x ≤ -2 ∨ x ≥ 4) :=
by sorry

theorem part2 (m : ℝ) : (∀ x, f x m ≥ 8) ↔ (m ≥ 7 ∨ m ≤ -9) :=
by sorry

end part1_part2_l134_134262


namespace no_real_solutions_l134_134166

theorem no_real_solutions :
  ¬ ∃ x : ℝ, (x - 3 * x + 8)^2 + 4 = -2 * |x| :=
by
  sorry

end no_real_solutions_l134_134166


namespace circle_symmetry_l134_134915

theorem circle_symmetry (a : ℝ) : 
  (∀ x y : ℝ, (x^2 + y^2 - a*x + 2*y + 1 = 0 ↔ x^2 + y^2 = 1) ↔ a = 2) :=
sorry

end circle_symmetry_l134_134915


namespace exists_triang_and_square_le_50_l134_134321

def is_triang_num (n : ℕ) : Prop := ∃ m : ℕ, n = m * (m + 1) / 2
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem exists_triang_and_square_le_50 : ∃ n : ℕ, n ≤ 50 ∧ is_triang_num n ∧ is_perfect_square n :=
by
  sorry

end exists_triang_and_square_le_50_l134_134321


namespace bear_population_l134_134491

theorem bear_population (black_bears white_bears brown_bears total_bears : ℕ) 
(h1 : black_bears = 60)
(h2 : white_bears = black_bears / 2)
(h3 : brown_bears = black_bears + 40) :
total_bears = black_bears + white_bears + brown_bears :=
sorry

end bear_population_l134_134491


namespace coffee_pods_per_box_l134_134068

theorem coffee_pods_per_box (d k : ℕ) (c e : ℝ) (h1 : d = 40) (h2 : k = 3) (h3 : c = 8) (h4 : e = 32) :
  ∃ b : ℕ, b = 30 :=
by
  sorry

end coffee_pods_per_box_l134_134068


namespace range_of_a_l134_134263

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then (1/2 : ℝ) * x - 1 else 1 / x

theorem range_of_a (a : ℝ) : f a > a ↔ a < -1 :=
sorry

end range_of_a_l134_134263


namespace parabola_vertex_l134_134302

theorem parabola_vertex (x y : ℝ) : 
  (∀ x y, y^2 - 8*y + 4*x = 12 → (x, y) = (7, 4)) :=
by
  intros x y h
  sorry

end parabola_vertex_l134_134302


namespace vec_subtraction_l134_134871

-- Definitions
def a : ℝ × ℝ := (1, -2)
def b (m : ℝ) : ℝ × ℝ := (m, 4)

-- Condition: a is parallel to b
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

-- Main theorem
theorem vec_subtraction (m : ℝ) (h : are_parallel a (b m)) :
  2 • a - b m = (4, -8) :=
sorry

end vec_subtraction_l134_134871


namespace tony_average_time_l134_134575

-- Definitions for the conditions
def speed_walk : ℝ := 2  -- speed in miles per hour when Tony walks
def speed_run : ℝ := 10  -- speed in miles per hour when Tony runs
def distance_to_store : ℝ := 4  -- distance to the store in miles
def days : List String := ["Sunday", "Tuesday", "Thursday"]  -- days Tony goes to the store

-- Definition of times taken on each day
def time_sunday := distance_to_store / speed_walk  -- time in hours to get to the store on Sunday
def time_tuesday := distance_to_store / speed_run  -- time in hours to get to the store on Tuesday
def time_thursday := distance_to_store / speed_run -- time in hours to get to the store on Thursday

-- Converting times to minutes
def time_sunday_minutes := time_sunday * 60
def time_tuesday_minutes := time_tuesday * 60
def time_thursday_minutes := time_thursday * 60

-- Definition of average time
def average_time_minutes : ℝ :=
  (time_sunday_minutes + time_tuesday_minutes + time_thursday_minutes) / days.length

-- The theorem to prove
theorem tony_average_time : average_time_minutes = 56 := by
  sorry

end tony_average_time_l134_134575


namespace unique_solution_for_system_l134_134511

theorem unique_solution_for_system (a : ℝ) :
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 4 * y = 0 ∧ x + a * y + a * z - a = 0 →
    (a = 2 ∨ a = -2)) :=
by
  intros x y z h
  sorry

end unique_solution_for_system_l134_134511


namespace inequality_holds_l134_134371

theorem inequality_holds (x : ℝ) (m : ℝ) :
  (∀ x : ℝ, (x^2 - m * x - 2) / (x^2 - 3 * x + 4) > -1) ↔ (-7 < m ∧ m < 1) :=
by
  sorry

end inequality_holds_l134_134371


namespace carol_seq_last_three_digits_l134_134959

/-- Carol starts to make a list, in increasing order, of the positive integers that have 
    a first digit of 2. She writes 2, 20, 21, 22, ...
    Prove that the three-digit number formed by the 1198th, 1199th, 
    and 1200th digits she wrote is 218. -/
theorem carol_seq_last_three_digits : 
  (digits_1198th_1199th_1200th = 218) :=
by
  sorry

end carol_seq_last_three_digits_l134_134959


namespace luncheon_cost_l134_134757

section LuncheonCosts

variables (s c p : ℝ)

/- Conditions -/
def eq1 : Prop := 2 * s + 5 * c + 2 * p = 6.25
def eq2 : Prop := 5 * s + 8 * c + 3 * p = 12.10

/- Goal -/
theorem luncheon_cost : eq1 s c p → eq2 s c p → s + c + p = 1.55 :=
by
  intro h1 h2
  sorry

end LuncheonCosts

end luncheon_cost_l134_134757


namespace larry_jogs_first_week_days_l134_134553

-- Defining the constants and conditions
def daily_jogging_time := 30 -- Larry jogs for 30 minutes each day
def total_jogging_time_in_hours := 4 -- Total jogging time in two weeks in hours
def total_jogging_time_in_minutes := total_jogging_time_in_hours * 60 -- Convert hours to minutes
def jogging_days_in_second_week := 5 -- Larry jogs 5 days in the second week
def daily_jogging_time_in_week2 := jogging_days_in_second_week * daily_jogging_time -- Total jogging time in minutes in the second week

-- Theorem statement
theorem larry_jogs_first_week_days : 
  (total_jogging_time_in_minutes - daily_jogging_time_in_week2) / daily_jogging_time = 3 :=
by
  -- Definitions and conditions used above should directly appear from the problem statement
  sorry

end larry_jogs_first_week_days_l134_134553


namespace trapezium_distance_parallel_sides_l134_134008

theorem trapezium_distance_parallel_sides
  (l1 l2 area : ℝ) (h : ℝ)
  (h_area : area = (1 / 2) * (l1 + l2) * h)
  (hl1 : l1 = 30)
  (hl2 : l2 = 12)
  (h_area_val : area = 336) :
  h = 16 :=
by
  sorry

end trapezium_distance_parallel_sides_l134_134008


namespace Carol_cleaning_time_l134_134119

theorem Carol_cleaning_time 
(Alice_time : ℕ) 
(Bob_time : ℕ) 
(Carol_time : ℕ) 
(h1 : Alice_time = 40) 
(h2 : Bob_time = 3 * Alice_time / 4) 
(h3 : Carol_time = 2 * Bob_time) :
  Carol_time = 60 := 
sorry

end Carol_cleaning_time_l134_134119


namespace maximum_height_when_isosceles_l134_134651

variable (c : ℝ) (c1 c2 : ℝ)

def right_angled_triangle (c1 c2 c : ℝ) : Prop :=
  c1 * c1 + c2 * c2 = c * c

def isosceles_right_triangle (c1 c2 : ℝ) : Prop :=
  c1 = c2

noncomputable def height_relative_to_hypotenuse (c : ℝ) : ℝ :=
  c / 2

theorem maximum_height_when_isosceles 
  (c1 c2 c : ℝ) 
  (h_right : right_angled_triangle c1 c2 c) 
  (h_iso : isosceles_right_triangle c1 c2) :
  height_relative_to_hypotenuse c = c / 2 :=
  sorry

end maximum_height_when_isosceles_l134_134651


namespace min_value_fraction_l134_134377

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2 * a + b = 1) : 
  ∃x : ℝ, (x = (1/a + 2/b)) ∧ (∀y : ℝ, (y = (1/a + 2/b)) → y ≥ 8) :=
by
  sorry

end min_value_fraction_l134_134377


namespace sqrt_expression_meaningful_l134_134719

theorem sqrt_expression_meaningful {x : ℝ} : (2 * x - 4) ≥ 0 → x ≥ 2 :=
by
  intro h
  sorry

end sqrt_expression_meaningful_l134_134719


namespace other_solution_l134_134153

theorem other_solution (x : ℚ) (h : 30*x^2 + 13 = 47*x - 2) (hx : x = 3/5) : x = 5/6 ∨ x = 3/5 := by
  sorry

end other_solution_l134_134153


namespace stone_145_is_5_l134_134498

theorem stone_145_is_5 :
  ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 15) → (145 % 28) = 5 → n = 5 :=
by
  intros n h h145
  sorry

end stone_145_is_5_l134_134498


namespace smallest_of_three_consecutive_odd_numbers_l134_134199

theorem smallest_of_three_consecutive_odd_numbers (x : ℤ) 
(h_sum : x + (x+2) + (x+4) = 69) : x = 21 :=
by
  sorry

end smallest_of_three_consecutive_odd_numbers_l134_134199


namespace div_trans_l134_134529

variable {a b c : ℝ}

theorem div_trans :
  a / b = 3 → b / c = 5 / 2 → c / a = 2 / 15 :=
  by
  intro h1 h2
  sorry

end div_trans_l134_134529


namespace line_always_passes_through_fixed_point_l134_134084

theorem line_always_passes_through_fixed_point :
  ∀ (m : ℝ), ∃ (x y : ℝ), (y = m * x + 2 * m + 1) ∧ (x = -2) ∧ (y = 1) :=
by
  sorry

end line_always_passes_through_fixed_point_l134_134084


namespace value_of_a_l134_134536

theorem value_of_a (a : ℤ) (h1 : 2 * a + 6 + (3 - a) = 0) : a = -9 :=
sorry

end value_of_a_l134_134536


namespace new_average_weight_l134_134079

-- Statement only
theorem new_average_weight (avg_weight_29: ℝ) (weight_new_student: ℝ) (total_students: ℕ) 
  (h1: avg_weight_29 = 28) (h2: weight_new_student = 22) (h3: total_students = 29) : 
  (avg_weight_29 * total_students + weight_new_student) / (total_students + 1) = 27.8 :=
by
  -- declare local variables for simpler proof
  let total_weight := avg_weight_29 * total_students
  let new_total_weight := total_weight + weight_new_student
  let new_total_students := total_students + 1
  have t_weight : total_weight = 812 := by sorry
  have new_t_weight : new_total_weight = 834 := by sorry
  have n_total_students : new_total_students = 30 := by sorry
  exact sorry

end new_average_weight_l134_134079


namespace solve_equation_l134_134429

theorem solve_equation (x : ℚ) (h : x ≠ -5) : 
  (x^2 + 3*x + 4) / (x + 5) = x + 6 ↔ x = -13 / 4 := by
  sorry

end solve_equation_l134_134429


namespace find_constants_l134_134829

theorem find_constants : 
  ∃ (a b : ℝ), a • (⟨1, 4⟩ : ℝ × ℝ) + b • (⟨3, -2⟩ : ℝ × ℝ) = (⟨5, 6⟩ : ℝ × ℝ) ∧ a = 2 ∧ b = 1 :=
by 
  sorry

end find_constants_l134_134829


namespace optimal_play_probability_Reimu_l134_134422

noncomputable def probability_Reimu_wins : ℚ :=
  5 / 16

theorem optimal_play_probability_Reimu :
  probability_Reimu_wins = 5 / 16 := 
by
  sorry

end optimal_play_probability_Reimu_l134_134422


namespace zero_of_f_a_neg_sqrt2_range_of_a_no_zero_l134_134869

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 2 then 2^x + a else a - x

theorem zero_of_f_a_neg_sqrt2 : 
  ∀ x, f x (- Real.sqrt 2) = 0 ↔ x = 1/2 :=
by
  sorry

theorem range_of_a_no_zero :
  ∀ a, (¬∃ x, f x a = 0) ↔ a ∈ Set.Iic (-4) ∪ Set.Ico 0 2 :=
by
  sorry

end zero_of_f_a_neg_sqrt2_range_of_a_no_zero_l134_134869


namespace triangle_side_length_l134_134178

theorem triangle_side_length (A B : ℝ) (b : ℝ) (a : ℝ) 
  (hA : A = 60) (hB : B = 45) (hb : b = 2) 
  (h : a = b * (Real.sin A) / (Real.sin B)) :
  a = Real.sqrt 6 := by
  sorry

end triangle_side_length_l134_134178


namespace dice_probabilities_relationship_l134_134649

theorem dice_probabilities_relationship :
  let p1 := 5 / 18
  let p2 := 11 / 18
  let p3 := 1 / 2
  p1 < p3 ∧ p3 < p2
:= by
  sorry

end dice_probabilities_relationship_l134_134649


namespace total_accidents_l134_134555

theorem total_accidents :
  let accidentsA := (75 / 100) * 2500
  let accidentsB := (50 / 80) * 1600
  let accidentsC := (90 / 200) * 1900
  accidentsA + accidentsB + accidentsC = 3730 :=
by
  let accidentsA := (75 / 100) * 2500
  let accidentsB := (50 / 80) * 1600
  let accidentsC := (90 / 200) * 1900
  sorry

end total_accidents_l134_134555


namespace range_of_m_l134_134863

theorem range_of_m (a b m : ℝ) (h₀ : a > 0) (h₁ : b > 1) (h₂ : a + b = 2) (h₃ : ∀ m, (4/a + 1/(b-1)) > m^2 + 8*m) : -9 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l134_134863


namespace probability_odd_product_l134_134682

theorem probability_odd_product :
  let box1 := [1, 2, 3, 4]
  let box2 := [1, 2, 3, 4]
  let total_outcomes := 4 * 4
  let favorable_outcomes := [(1,1), (1,3), (3,1), (3,3)]
  let num_favorable := favorable_outcomes.length
  (num_favorable / total_outcomes : ℚ) = 1 / 4 := 
by
  sorry

end probability_odd_product_l134_134682


namespace find_abc_l134_134865

variables {a b c : ℕ}

theorem find_abc (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : abc ∣ ((a * b - 1) * (b * c - 1) * (c * a - 1))) : a = 2 ∧ b = 3 ∧ c = 5 :=
by {
    sorry
}

end find_abc_l134_134865


namespace problem_part_1_problem_part_2_l134_134971
open Set Real

noncomputable def A (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a^2 - 2}
noncomputable def B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem problem_part_1 : A 3 ∪ B = {x | 1 < x ∧ x ≤ 7} := 
  by
  sorry

theorem problem_part_2 : (∀ a : ℝ, A a ∪ B = B → 2 < a ∧ a < sqrt 7) :=
  by 
  sorry

end problem_part_1_problem_part_2_l134_134971


namespace students_present_l134_134455

theorem students_present (absent_students male_students female_student_diff : ℕ) 
  (h1 : absent_students = 18) 
  (h2 : male_students = 848) 
  (h3 : female_student_diff = 49) : 
  (male_students + (male_students - female_student_diff) - absent_students = 1629) := 

by 
  sorry

end students_present_l134_134455


namespace asian_games_discount_equation_l134_134831

variable (a : ℝ)

theorem asian_games_discount_equation :
  168 * (1 - a / 100)^2 = 128 :=
sorry

end asian_games_discount_equation_l134_134831


namespace plums_for_20_oranges_l134_134397

noncomputable def oranges_to_pears (oranges : ℕ) : ℕ :=
  (oranges / 5) * 3

noncomputable def pears_to_plums (pears : ℕ) : ℕ :=
  (pears / 4) * 6

theorem plums_for_20_oranges :
  oranges_to_pears 20 = 12 ∧ pears_to_plums 12 = 18 :=
by
  sorry

end plums_for_20_oranges_l134_134397


namespace chord_length_condition_l134_134707

theorem chord_length_condition (c : ℝ) (h : c > 0) :
  (∃ (x1 x2 : ℝ), 
    x1 ≠ x2 ∧ 
    dist (x1, x1^2) (x2, x2^2) = 2 ∧ 
    ∃ k : ℝ, x1 * k + c = x1^2 ∧ x2 * k + c = x2^2 ) 
    ↔ c > 0 :=
sorry

end chord_length_condition_l134_134707


namespace grapes_total_sum_l134_134807

theorem grapes_total_sum (R A N : ℕ) 
  (h1 : A = R + 2) 
  (h2 : N = A + 4) 
  (h3 : R = 25) : 
  R + A + N = 83 := by
  sorry

end grapes_total_sum_l134_134807


namespace aira_rubber_bands_l134_134581

theorem aira_rubber_bands (total_bands : ℕ) (bands_each : ℕ) (samantha_extra : ℕ) (aira_fewer : ℕ)
  (h1 : total_bands = 18) 
  (h2 : bands_each = 6) 
  (h3 : samantha_extra = 5) 
  (h4 : aira_fewer = 1) : 
  ∃ x : ℕ, x + (x + samantha_extra) + (x + aira_fewer) = total_bands ∧ x = 4 :=
by
  sorry

end aira_rubber_bands_l134_134581


namespace contingency_table_proof_l134_134533

noncomputable def probability_of_mistake (K_squared : ℝ) : ℝ :=
if K_squared > 3.841 then 0.05 else 1.0 -- placeholder definition to be refined

theorem contingency_table_proof :
  probability_of_mistake 4.013 ≤ 0.05 :=
by sorry

end contingency_table_proof_l134_134533


namespace find_number_l134_134872

theorem find_number (n : ℕ) (h : (1 / 2 : ℝ) * n + 5 = 13) : n = 16 := 
by
  sorry

end find_number_l134_134872


namespace max_value_of_expression_l134_134044

theorem max_value_of_expression (a b c : ℝ) (ha : 0 ≤ a) (ha2 : a ≤ 2) (hb : 0 ≤ b) (hb2 : b ≤ 2) (hc : 0 ≤ c) (hc2 : c ≤ 2) :
  2 * Real.sqrt (abc / 8) + Real.sqrt ((2 - a) * (2 - b) * (2 - c)) ≤ 2 :=
by
  sorry

end max_value_of_expression_l134_134044


namespace infinite_series_sum_l134_134050

noncomputable def inf_series (a b : ℝ) : ℝ :=
  ∑' (n : ℕ), if n = 1 then 1 / (b * a)
  else if n % 2 = 0 then 1 / ((↑(n - 1) * a - b) * (↑n * a - b))
  else 1 / ((↑(n - 1) * a + b) * (↑n * a - b))

theorem infinite_series_sum (a b : ℝ) 
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) :
  inf_series a b = 1 / (a * b) :=
sorry

end infinite_series_sum_l134_134050


namespace homework_checked_on_friday_given_not_checked_until_thursday_l134_134924

open ProbabilityTheory

variables {Ω : Type} {P : ProbabilitySpace Ω}
variables (S : Event Ω) (A : Event Ω) (B : Event Ω)
variables [Fact (Probability S = 1 / 2)]
variables [Fact (Probability (Sᶜ ∩ B) = 1 / 10)]
variables [Fact (Probability A = 3 / 5)]
variables [Fact (A = S ∪ B)]
variables [Fact (Aᶜ = Sᶜ ∩ Aᶜ)]

theorem homework_checked_on_friday_given_not_checked_until_thursday :
  condProb B A = 1 / 6 := sorry

end homework_checked_on_friday_given_not_checked_until_thursday_l134_134924


namespace height_of_Linda_room_l134_134053

theorem height_of_Linda_room (w l: ℝ) (h a1 a2 a3 paint_area: ℝ) 
  (hw: w = 20) (hl: l = 20) 
  (d1_h: a1 = 3) (d1_w: a2 = 7) 
  (d2_h: a3 = 4) (d2_w: a4 = 6) 
  (d3_h: a5 = 5) (d3_w: a6 = 7) 
  (total_paint_area: paint_area = 560):
  h = 6 := 
by
  sorry

end height_of_Linda_room_l134_134053


namespace age_problem_l134_134507

open Classical

noncomputable def sum_cubes_ages (r j m : ℕ) : ℕ :=
  r^3 + j^3 + m^3

theorem age_problem (r j m : ℕ) (h1 : 5 * r + 2 * j = 3 * m)
    (h2 : 3 * m^2 + 2 * j^2 = 5 * r^2) (h3 : Nat.gcd r (Nat.gcd j m) = 1) :
    sum_cubes_ages r j m = 3 := by
  sorry

end age_problem_l134_134507


namespace solve_equation_l134_134366

theorem solve_equation (x : ℝ) (hx : x ≠ 0) : 
  x^2 + 36 / x^2 = 13 ↔ (x = 2 ∨ x = -2 ∨ x = 3 ∨ x = -3) := by
  sorry

end solve_equation_l134_134366


namespace two_digit_multiple_condition_l134_134648

theorem two_digit_multiple_condition :
  ∃ x : ℕ, 10 ≤ x ∧ x < 100 ∧ ∃ k : ℤ, x = 30 * k + 2 :=
by
  sorry

end two_digit_multiple_condition_l134_134648


namespace increase_corrosion_with_more_active_metal_rivets_l134_134772

-- Definitions representing conditions
def corrosion_inhibitor (P : Type) : Prop := true
def more_active_metal_rivets (P : Type) : Prop := true
def less_active_metal_rivets (P : Type) : Prop := true
def painted_parts (P : Type) : Prop := true

-- Main theorem statement
theorem increase_corrosion_with_more_active_metal_rivets (P : Type) 
  (h1 : corrosion_inhibitor P)
  (h2 : more_active_metal_rivets P)
  (h3 : less_active_metal_rivets P)
  (h4 : painted_parts P) : 
  more_active_metal_rivets P :=
by {
  -- proof goes here
  sorry
}

end increase_corrosion_with_more_active_metal_rivets_l134_134772


namespace negative_solution_range_l134_134535

theorem negative_solution_range (m x : ℝ) (h : (2 * x + m) / (x - 1) = 1) (hx : x < 0) : m > -1 :=
  sorry

end negative_solution_range_l134_134535


namespace probability_of_pink_gumball_l134_134404

theorem probability_of_pink_gumball (P_B P_P : ℝ)
    (h1 : P_B ^ 2 = 25 / 49)
    (h2 : P_B + P_P = 1) :
    P_P = 2 / 7 := 
    sorry

end probability_of_pink_gumball_l134_134404


namespace factor_expr_l134_134232

variable (x : ℝ)

def expr : ℝ := (20 * x ^ 3 + 100 * x - 10) - (-5 * x ^ 3 + 5 * x - 10)

theorem factor_expr :
  expr x = 5 * x * (5 * x ^ 2 + 19) :=
by
  sorry

end factor_expr_l134_134232


namespace probability_prime_and_cube_is_correct_l134_134622

-- Conditions based on the problem
def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_cube (n : ℕ) : Prop :=
  n = 1 ∨ n = 8

def possible_outcomes := 8 * 8
def successful_outcomes := 4 * 2

noncomputable def probability_of_prime_and_cube :=
  (successful_outcomes : ℝ) / (possible_outcomes : ℝ)

theorem probability_prime_and_cube_is_correct :
  probability_of_prime_and_cube = 1 / 8 :=
by
  sorry

end probability_prime_and_cube_is_correct_l134_134622


namespace parallel_lines_m_eq_one_l134_134273

theorem parallel_lines_m_eq_one (m : ℝ) :
  (∀ x y : ℝ, 2 * x + m * y + 8 = 0 ∧ (m + 1) * x + y + (m - 2) = 0 → m = 1) :=
by
  intro x y h
  let L1_slope := -2 / m
  let L2_slope := -(m + 1)
  have h_slope : L1_slope = L2_slope := sorry
  have m_positive : m = 1 := sorry
  exact m_positive

end parallel_lines_m_eq_one_l134_134273


namespace sarah_marriage_age_l134_134238

theorem sarah_marriage_age : 
  let name_length := 5 in
  let current_age := 9 in
  let twice_age := 2 * current_age in
  name_length + twice_age = 23 :=
by
  let name_length := 5
  let current_age := 9
  let twice_age := 2 * current_age
  show name_length + twice_age = 23
  sorry

end sarah_marriage_age_l134_134238


namespace smallest_n_for_triangle_area_l134_134825

theorem smallest_n_for_triangle_area :
  ∃ n : ℕ, 10 * n^4 - 8 * n^3 - 52 * n^2 + 32 * n - 24 > 10000 ∧ ∀ m : ℕ, 
  (m < n → ¬ (10 * m^4 - 8 * m^3 - 52 * m^2 + 32 * m - 24 > 10000)) :=
sorry

end smallest_n_for_triangle_area_l134_134825


namespace y1_increasing_on_0_1_l134_134955

noncomputable def y1 (x : ℝ) : ℝ := |x|
noncomputable def y2 (x : ℝ) : ℝ := 3 - x
noncomputable def y3 (x : ℝ) : ℝ := 1 / x
noncomputable def y4 (x : ℝ) : ℝ := -x^2 + 4

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem y1_increasing_on_0_1 :
  is_increasing_on y1 0 1 ∧
  ¬ is_increasing_on y2 0 1 ∧
  ¬ is_increasing_on y3 0 1 ∧
  ¬ is_increasing_on y4 0 1 :=
by
  sorry

end y1_increasing_on_0_1_l134_134955


namespace probability_two_primes_from_1_to_30_l134_134634

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem probability_two_primes_from_1_to_30 : 
  (primes_up_to_30.length.choose 2).to_rational / (30.choose 2).to_rational = 5/29 :=
by
  -- proof steps here
  sorry

end probability_two_primes_from_1_to_30_l134_134634


namespace relation_P_Q_l134_134914

def P : Set ℝ := {x | x ≠ 0}
def Q : Set ℝ := {x | x > 0}
def complement_P : Set ℝ := {0}

theorem relation_P_Q : Q ∩ complement_P = ∅ := 
by sorry

end relation_P_Q_l134_134914


namespace total_weight_of_lifts_l134_134541

theorem total_weight_of_lifts 
  (F S : ℕ)
  (h1 : F = 400)
  (h2 : 2 * F = S + 300) :
  F + S = 900 :=
by
  sorry

end total_weight_of_lifts_l134_134541


namespace greatest_cars_with_ac_not_racing_stripes_l134_134473

-- Definitions
def total_cars : ℕ := 100
def cars_without_ac : ℕ := 47
def cars_with_ac : ℕ := total_cars - cars_without_ac
def at_least_racing_stripes : ℕ := 53

-- Prove that the greatest number of cars that could have air conditioning but not racing stripes is 53
theorem greatest_cars_with_ac_not_racing_stripes :
  ∃ maximum_cars_with_ac_not_racing_stripes, 
    maximum_cars_with_ac_not_racing_stripes = cars_with_ac - 0 ∧
    maximum_cars_with_ac_not_racing_stripes = 53 := 
by
  sorry

end greatest_cars_with_ac_not_racing_stripes_l134_134473


namespace find_a1_l134_134147

theorem find_a1 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_rec : ∀ n ≥ 2, a n + 2 * S n * S (n - 1) = 0)
  (h_S5 : S 5 = 1 / 11) : 
  a 1 = 1 / 3 := 
sorry

end find_a1_l134_134147


namespace total_animals_counted_l134_134551

theorem total_animals_counted :
  let antelopes := 80
  let rabbits := antelopes + 34
  let total_rabbits_antelopes := rabbits + antelopes
  let hyenas := total_rabbits_antelopes - 42
  let wild_dogs := hyenas + 50
  let leopards := rabbits / 2
  (antelopes + rabbits + hyenas + wild_dogs + leopards) = 605 :=
by
  let antelopes := 80
  let rabbits := antelopes + 34
  let total_rabbits_antelopes := rabbits + antelopes
  let hyenas := total_rabbits_antelopes - 42
  let wild_dogs := hyenas + 50
  let leopards := rabbits / 2
  show (antelopes + rabbits + hyenas + wild_dogs + leopards) = 605
  sorry

end total_animals_counted_l134_134551


namespace find_inradius_of_scalene_triangle_l134_134287

noncomputable def side_a := 32
noncomputable def side_b := 40
noncomputable def side_c := 24
noncomputable def ic := 18
noncomputable def expected_inradius := 2 * Real.sqrt 17

theorem find_inradius_of_scalene_triangle (a b c : ℝ) (h : a = side_a) (h1 : b = side_b) (h2 : c = side_c) (ic_length : ℝ) (h3: ic_length = ic) : (Real.sqrt (ic_length ^ 2 - (b - ((a + b - c) / 2)) ^ 2)) = expected_inradius :=
by
  sorry

end find_inradius_of_scalene_triangle_l134_134287


namespace probability_two_primes_from_1_to_30_l134_134635

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem probability_two_primes_from_1_to_30 : 
  (primes_up_to_30.length.choose 2).to_rational / (30.choose 2).to_rational = 5/29 :=
by
  -- proof steps here
  sorry

end probability_two_primes_from_1_to_30_l134_134635


namespace find_height_of_tank_A_l134_134192

noncomputable def height_of_tank_A (C_A C_B h_B ratio V_ratio : ℝ) : ℝ :=
  let r_A := C_A / (2 * Real.pi)
  let r_B := C_B / (2 * Real.pi)
  let V_A := Real.pi * (r_A ^ 2) * ratio
  let V_B := Real.pi * (r_B ^ 2) * h_B
  (V_ratio * V_B) / (Real.pi * (r_A ^ 2))

theorem find_height_of_tank_A :
  height_of_tank_A 8 10 8 10 0.8000000000000001 = 10 :=
by
  sorry

end find_height_of_tank_A_l134_134192


namespace middle_aged_selection_l134_134486

def total_teachers := 80 + 160 + 240
def sample_size := 60
def middle_aged_proportion := 160 / total_teachers
def middle_aged_sample := middle_aged_proportion * sample_size

theorem middle_aged_selection : middle_aged_sample = 20 :=
  sorry

end middle_aged_selection_l134_134486


namespace one_fourth_of_8_point_4_is_21_over_10_l134_134247

theorem one_fourth_of_8_point_4_is_21_over_10 : (8.4 / 4 : ℚ) = 21 / 10 := 
by
  sorry

end one_fourth_of_8_point_4_is_21_over_10_l134_134247


namespace total_grapes_l134_134813

theorem total_grapes (r a n : ℕ) (h1 : r = 25) (h2 : a = r + 2) (h3 : n = a + 4) : r + a + n = 83 := by
  sorry

end total_grapes_l134_134813


namespace max_chord_length_of_parabola_l134_134392

-- Definitions based on the problem conditions
def parabola (x y : ℝ) : Prop := x^2 = 8 * y
def y_midpoint_condition (y1 y2 : ℝ) : Prop := (y1 + y2) / 2 = 4

-- The theorem to prove that the maximum length of the chord AB is 12
theorem max_chord_length_of_parabola (x1 y1 x2 y2 : ℝ) 
  (h1 : parabola x1 y1) 
  (h2 : parabola x2 y2) 
  (h_mid : y_midpoint_condition y1 y2) : 
  abs ((y1 + y2) + 2 * 2) = 12 :=
sorry

end max_chord_length_of_parabola_l134_134392


namespace negation_of_universal_proposition_l134_134086

open Real

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → (x+1) * exp x > 1) ↔ ∃ x : ℝ, x > 0 ∧ (x+1) * exp x ≤ 1 :=
by sorry

end negation_of_universal_proposition_l134_134086


namespace anticipated_margin_l134_134661

noncomputable def anticipated_profit_margin (original_purchase_price : ℝ) (decrease_percentage : ℝ) (profit_margin_increase : ℝ) (selling_price : ℝ) : ℝ :=
original_purchase_price * (1 + profit_margin_increase / 100)

theorem anticipated_margin (x : ℝ) (original_purchase_price_decrease : ℝ := 0.064) (profit_margin_increase : ℝ := 8) (selling_price : ℝ) :
  selling_price = original_purchase_price * (1 + x / 100) ∧ selling_price = (1 - original_purchase_price_decrease) * (1 + (x + profit_margin_increase) / 100) →
  true :=
by
  sorry

end anticipated_margin_l134_134661


namespace angle_trig_identity_l134_134036

theorem angle_trig_identity
  (A B C : ℝ)
  (h_sum : A + B + C = Real.pi) :
  Real.cos (A / 2) ^ 2 = Real.cos (B / 2) ^ 2 + Real.cos (C / 2) ^ 2 - 
                       2 * Real.cos (B / 2) * Real.cos (C / 2) * Real.sin (A / 2) :=
by
  sorry

end angle_trig_identity_l134_134036


namespace min_value_fraction_sum_l134_134973

theorem min_value_fraction_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → (4 / (x + 2) + 1 / (y + 1)) ≥ 9 / 4) :=
by
  sorry

end min_value_fraction_sum_l134_134973


namespace fish_left_in_sea_l134_134328

-- Definitions based on conditions
def total_fish_westward : Nat := 1800
def total_fish_eastward : Nat := 3200
def total_fish_north : Nat := 500

def caught_fraction_westward : Rat := 3/4
def caught_fraction_eastward : Rat := 2/5

-- Theorem statement
theorem fish_left_in_sea : 
  let fish_left_westward := total_fish_westward - (caught_fraction_westward * total_fish_westward).nat
  let fish_left_eastward := total_fish_eastward - (caught_fraction_eastward * total_fish_eastward).nat
  let fish_left_north := total_fish_north
  fish_left_westward + fish_left_eastward + fish_left_north = 2870 := 
by
  -- Placeholder for proof
  sorry

end fish_left_in_sea_l134_134328


namespace x_plus_y_value_l134_134210

def sum_evens_40_to_60 : ℕ :=
  (40 + 42 + 44 + 46 + 48 + 50 + 52 + 54 + 56 + 58 + 60)

def num_evens_40_to_60 : ℕ := 11

theorem x_plus_y_value : sum_evens_40_to_60 + num_evens_40_to_60 = 561 := by
  sorry

end x_plus_y_value_l134_134210


namespace ratio_dislikes_to_likes_l134_134659

theorem ratio_dislikes_to_likes 
  (D : ℕ) 
  (h1 : D + 1000 = 2600) 
  (h2 : 3000 > 0) : 
  D / 3000 = 8 / 15 :=
by sorry

end ratio_dislikes_to_likes_l134_134659


namespace sum_mod_nine_l134_134012

def a : ℕ := 1234
def b : ℕ := 1235
def c : ℕ := 1236
def d : ℕ := 1237
def e : ℕ := 1238
def modulus : ℕ := 9

theorem sum_mod_nine : (a + b + c + d + e) % modulus = 6 :=
by
  sorry

end sum_mod_nine_l134_134012


namespace finite_set_cardinality_l134_134898

-- Define the main theorem statement
theorem finite_set_cardinality (m : ℕ) (A : Finset ℤ) (B : ℕ → Finset ℤ)
  (hm : m ≥ 2)
  (hB : ∀ k : ℕ, k ∈ Finset.range m.succ → (B k).sum id = m^k) :
  A.card ≥ m / 2 := 
sorry

end finite_set_cardinality_l134_134898


namespace problem_statement_l134_134368

variables {R : Type*} [LinearOrderedField R]

-- Definitions of f and its derivatives
variable (f : R → R)
variable (f' : R → R) 
variable (f'' : R → R)

-- Conditions given in the math problem
axiom decreasing_f : ∀ x1 x2 : R, x1 < x2 → f x1 > f x2
axiom derivative_condition : ∀ x : R, f'' x ≠ 0 → f x / f'' x < 1 - x

-- Lean 4 statement for the proof problem
theorem problem_statement (decreasing_f : ∀ x1 x2 : R, x1 < x2 → f x1 > f x2)
    (derivative_condition : ∀ x : R, f'' x ≠ 0 → f x / f'' x < 1 - x) :
    ∀ x : R, f x > 0 :=
by
  sorry

end problem_statement_l134_134368


namespace a100_pos_a100_abs_lt_018_l134_134023

noncomputable def a (n : ℕ) : ℝ := real.cos (10^n * real.pi / 180)

theorem a100_pos : (a 100 > 0) :=
by
  sorry

theorem a100_abs_lt_018 : (|a 100| < 0.18) :=
by
  sorry

end a100_pos_a100_abs_lt_018_l134_134023


namespace hall_area_l134_134080

theorem hall_area (L : ℝ) (B : ℝ) (A : ℝ) (h1 : B = (2/3) * L) (h2 : L = 60) (h3 : A = L * B) : A = 2400 := 
by 
sorry

end hall_area_l134_134080


namespace total_savings_eighteen_l134_134143

theorem total_savings_eighteen :
  let fox_price := 15
  let pony_price := 18
  let discount_rate_sum := 50
  let fox_quantity := 3
  let pony_quantity := 2
  let pony_discount_rate := 50
  let total_price_without_discount := (fox_quantity * fox_price) + (pony_quantity * pony_price)
  let discounted_pony_price := (pony_price * (1 - (pony_discount_rate / 100)))
  let total_price_with_discount := (fox_quantity * fox_price) + (pony_quantity * discounted_pony_price)
  let total_savings := total_price_without_discount - total_price_with_discount
  total_savings = 18 :=
by sorry

end total_savings_eighteen_l134_134143


namespace problem_1_max_value_problem_2_good_sets_count_l134_134172

noncomputable def goodSetMaxValue : ℤ :=
  2012

noncomputable def goodSetCount : ℤ :=
  1006

theorem problem_1_max_value {M : Set ℤ} (hM : ∀ x, x ∈ M ↔ |x| ≤ 2014) :
  ∀ a b c : ℤ, (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (1 / a + 1 / b = 2 / c) →
  (a + c = 2 * b) →
  a ∈ M ∧ b ∈ M ∧ c ∈ M →
  ∃ P : Set ℤ, P = {a, b, c} ∧ a ∈ P ∧ b ∈ P ∧ c ∈ P ∧
  goodSetMaxValue = 2012 :=
sorry

theorem problem_2_good_sets_count {M : Set ℤ} (hM : ∀ x, x ∈ M ↔ |x| ≤ 2014) :
  ∀ a b c : ℤ, (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (1 / a + 1 / b = 2 / c) →
  (a + c = 2 * b) →
  a ∈ M ∧ b ∈ M ∧ c ∈ M →
  ∃ P : Set ℤ, P = {a, b, c} ∧ a ∈ P ∧ b ∈ P ∧ c ∈ P ∧
  goodSetCount = 1006 :=
sorry

end problem_1_max_value_problem_2_good_sets_count_l134_134172


namespace find_initial_period_l134_134310

theorem find_initial_period (P : ℝ) (T : ℝ) 
  (h1 : 1680 = (P * 4 * T) / 100)
  (h2 : 1680 = (P * 5 * 4) / 100) 
  : T = 5 := 
by 
  sorry

end find_initial_period_l134_134310


namespace initial_noodles_l134_134134

variable (d w e r : ℕ)

-- Conditions
def gave_to_william (w : ℕ) := w = 15
def gave_to_emily (e : ℕ) := e = 20
def remaining_noodles (r : ℕ) := r = 40

-- The statement to be proven
theorem initial_noodles (h1 : gave_to_william w) (h2 : gave_to_emily e) (h3 : remaining_noodles r) : d = w + e + r := by
  -- Proof will be filled in later.
  sorry

end initial_noodles_l134_134134


namespace ellipse_properties_l134_134148

theorem ellipse_properties :
  let C : set (ℝ × ℝ) := {p | (p.1^2) / 4 + (p.2^2) / 3 = 1}
  (A : ℝ × ℝ) (f1 f2 : ℝ × ℝ),
  A = (1, 3 / 2) ∧ f1 = (-1, 0) ∧ f2 = (1, 0) →
  (C A := A ∈ C) ∧ 
  (∀ (E F : ℝ × ℝ), 
    E ∈ C → F ∈ C →
    (∃ k : ℝ, 
      E.2 - A.2 = k * (E.1 - A.1) ∧
      F.2 - A.2 = (-1 / k) * (F.1 - A.1)) →
      (let slope_EF := (F.2 - E.2) / (F.1 - E.1) in
        slope_EF = 1 / 2))
  :=
by
  sorry

end ellipse_properties_l134_134148


namespace digit_making_527B_divisible_by_9_l134_134204

theorem digit_making_527B_divisible_by_9 (B : ℕ) : 14 + B ≡ 0 [MOD 9] → B = 4 :=
by
  intro h
  -- sorry is used in place of the actual proof.
  sorry

end digit_making_527B_divisible_by_9_l134_134204


namespace sqrt_product_simplify_l134_134821

theorem sqrt_product_simplify (x : ℝ) (hx : 0 ≤ x):
  Real.sqrt (48*x) * Real.sqrt (3*x) * Real.sqrt (50*x) = 60 * x * Real.sqrt x := 
by
  sorry

end sqrt_product_simplify_l134_134821


namespace average_percentage_reduction_l134_134111

theorem average_percentage_reduction (x : ℝ) (hx : 0 < x ∧ x < 1)
  (initial_price final_price : ℝ)
  (h_initial : initial_price = 25)
  (h_final : final_price = 16)
  (h_reduction : final_price = initial_price * (1-x)^2) :
  x = 0.2 :=
by {
  --". Convert fraction \( = x / y \)", proof is omitted
  sorry
}

end average_percentage_reduction_l134_134111


namespace avg_speed_is_40_l134_134208

noncomputable def average_speed (x : ℝ) : ℝ :=
  let time1 := x / 40
  let time2 := 2 * x / 20
  let total_time := time1 + time2
  let total_distance := 5 * x
  total_distance / total_time

theorem avg_speed_is_40 (x : ℝ) (hx : x > 0) :
  average_speed x = 40 := by
  sorry

end avg_speed_is_40_l134_134208


namespace tony_average_time_to_store_l134_134572

-- Definitions and conditions
def speed_walking := 2  -- MPH
def speed_running := 10  -- MPH
def distance_to_store := 4  -- miles
def days_walking := 1  -- Sunday
def days_running := 2  -- Tuesday, Thursday
def total_days := days_walking + days_running

-- Proof statement
theorem tony_average_time_to_store :
  let time_walking := (distance_to_store / speed_walking) * 60
  let time_running := (distance_to_store / speed_running) * 60
  let total_time := time_walking * days_walking + time_running * days_running
  let average_time := total_time / total_days
  average_time = 56 :=
by
  sorry  -- Proof to be completed

end tony_average_time_to_store_l134_134572


namespace probability_of_prime_pairs_l134_134638

def primes_in_range := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def num_pairs (n : Nat) : Nat := (n * (n - 1)) / 2

theorem probability_of_prime_pairs : (num_pairs 10 : ℚ) / (num_pairs 30) = (1 : ℚ) / 10 := by
  sorry

end probability_of_prime_pairs_l134_134638


namespace yokohama_entrance_exam_solution_l134_134737

noncomputable def volume_of_solid (a : ℝ) (f g : ℝ → ℝ) :=
  ∫ x in 0..1, π * ((g x) ^ 2)dx +
  ∫ x in 1..exp (1 / 3), π * (((g x) ^ 2) - ((f x) ^ 2))dx

theorem yokohama_entrance_exam_solution :
  ∀ a : ℝ, (a = 1 / (3 * exp(1))) →
  (∀ x, (f x) = ln x / x) →
  (∀ x, (g x) = a * x^2) →
  volume_of_solid a f g = π * (1 + 100 * exp (1 / 3) - 72 * exp (2 / 3)) / (36 * exp (2 / 3)) :=
by
  intros a ha hf hg
  rw [hf, hg]
  sorry

end yokohama_entrance_exam_solution_l134_134737


namespace mark_height_feet_l134_134292

theorem mark_height_feet
  (mark_height_inches : ℕ)
  (mike_height_feet : ℕ)
  (mike_height_inches : ℕ)
  (mike_taller_than_mark : ℕ)
  (foot_in_inches : ℕ)
  (mark_height_eq : mark_height_inches = 3)
  (mike_height_eq : mike_height_feet * foot_in_inches + mike_height_inches = 73)
  (mike_taller_eq : mike_height_feet * foot_in_inches + mike_height_inches = mark_height_inches + mike_taller_than_mark)
  (foot_in_inches_eq : foot_in_inches = 12) :
  mark_height_inches = 63 ∧ mark_height_inches / foot_in_inches = 5 := by
sorry

end mark_height_feet_l134_134292


namespace B_correct_A_inter_B_correct_l134_134031

def A := {x : ℝ | 1 < x ∧ x < 8}
def B := {x : ℝ | x^2 - 5 * x - 14 ≥ 0}

theorem B_correct : B = {x : ℝ | x ≤ -2 ∨ x ≥ 7} := 
sorry

theorem A_inter_B_correct : A ∩ B = {x : ℝ | 7 ≤ x ∧ x < 8} :=
sorry

end B_correct_A_inter_B_correct_l134_134031


namespace sum_of_factors_1656_l134_134728

theorem sum_of_factors_1656 : ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 1656 ∧ a + b = 110 := by
  sorry

end sum_of_factors_1656_l134_134728


namespace solve_abs_quadratic_l134_134070

theorem solve_abs_quadratic :
  ∀ x : ℝ, abs (x^2 - 4 * x + 4) = 3 - x ↔ (x = (3 + Real.sqrt 5) / 2 ∨ x = (3 - Real.sqrt 5) / 2) :=
by
  sorry

end solve_abs_quadratic_l134_134070


namespace median_of_circumscribed_trapezoid_l134_134228

theorem median_of_circumscribed_trapezoid (a b c d : ℝ) (h1 : a + b + c + d = 12) (h2 : a + b = c + d) : (a + b) / 2 = 3 :=
by
  sorry

end median_of_circumscribed_trapezoid_l134_134228


namespace number_of_men_in_club_l134_134219

variables (M W : ℕ)

theorem number_of_men_in_club 
  (h1 : M + W = 30) 
  (h2 : (1 / 3 : ℝ) * W + M = 18) : 
  M = 12 := 
sorry

end number_of_men_in_club_l134_134219


namespace total_cats_handled_last_year_l134_134183

theorem total_cats_handled_last_year (num_adult_cats : ℕ) (two_thirds_female : ℕ) (seventy_five_percent_litters : ℕ) 
                                     (kittens_per_litter : ℕ) (adopted_returned : ℕ) :
  num_adult_cats = 120 →
  two_thirds_female = (2 * num_adult_cats) / 3 →
  seventy_five_percent_litters = (3 * two_thirds_female) / 4 →
  kittens_per_litter = 3 →
  adopted_returned = 15 →
  num_adult_cats + seventy_five_percent_litters * kittens_per_litter + adopted_returned = 315 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end total_cats_handled_last_year_l134_134183


namespace range_of_a_l134_134703

theorem range_of_a (a : ℝ) (h : 0 < a ∧ a < 2) (h_ineq : (sin (1 - a) + 5 * (1 - a)) + (sin (1 - a^2) + 5 * (1 - a^2)) < 0) : 1 < a ∧ a < real.sqrt 2 :=
sorry

end range_of_a_l134_134703


namespace man_speed_was_5_kmph_l134_134333

theorem man_speed_was_5_kmph (time_in_minutes : ℕ) (distance_in_km : ℝ)
  (h_time : time_in_minutes = 30)
  (h_distance : distance_in_km = 2.5) :
  (distance_in_km / (time_in_minutes / 60 : ℝ) = 5) :=
by
  sorry

end man_speed_was_5_kmph_l134_134333


namespace find_k_l134_134259

theorem find_k (k : ℕ) (hk : k > 0) (h_coeff : 15 * k^4 < 120) : k = 1 := 
by 
  sorry

end find_k_l134_134259


namespace remaining_money_after_shopping_l134_134644

theorem remaining_money_after_shopping (initial_money : ℝ) (percentage_spent : ℝ) (final_amount : ℝ) :
  initial_money = 1200 → percentage_spent = 0.30 → final_amount = initial_money - (percentage_spent * initial_money) → final_amount = 840 :=
by
  intros h_initial h_percentage h_final
  sorry

end remaining_money_after_shopping_l134_134644


namespace sum_of_roots_quadratic_specific_sum_of_roots_l134_134466

theorem sum_of_roots_quadratic:
  ∀ a b c : ℚ, a ≠ 0 → 
  ∀ x1 x2 : ℚ, (a * x1^2 + b * x1 + c = 0) ∧ 
               (a * x2^2 + b * x2 + c = 0) → 
               x1 + x2 = -b / a := 
by
  sorry

theorem specific_sum_of_roots:
  ∀ x1 x2 : ℚ, (12 * x1^2 + 19 * x1 - 21 = 0) ∧ 
               (12 * x2^2 + 19 * x2 - 21 = 0) → 
               x1 + x2 = -19 / 12 := 
by
  sorry

end sum_of_roots_quadratic_specific_sum_of_roots_l134_134466


namespace expression_eq_neg_one_l134_134236

theorem expression_eq_neg_one (a b y : ℝ) (h1 : a ≠ 0) (h2 : b ≠ a) (h3 : y ≠ a) (h4 : y ≠ -a) :
  ( ( (a + b) / (a + y) + y / (a - y) ) / ( (y + b) / (a + y) - a / (a - y) ) = -1 ) ↔ ( y = a - b ) := 
sorry

end expression_eq_neg_one_l134_134236


namespace bananas_to_pears_l134_134123

theorem bananas_to_pears:
  (∀ b a o p : ℕ, 
    6 * b = 4 * a → 
    5 * a = 3 * o → 
    4 * o = 7 * p → 
    36 * b = 28 * p) :=
by
  intros b a o p h1 h2 h3
  -- We need to prove 36 * b = 28 * p under the given conditions
  sorry

end bananas_to_pears_l134_134123


namespace y_give_z_start_l134_134476

variables (Vx Vy Vz T : ℝ)
variables (D : ℝ)

-- Conditions
def condition1 : Prop := Vx * T = Vy * T + 100
def condition2 : Prop := Vx * T = Vz * T + 200
def condition3 : Prop := T > 0

theorem y_give_z_start (h1 : condition1 Vx Vy T) (h2 : condition2 Vx Vz T) (h3 : condition3 T) : (Vy - Vz) * T = 200 := 
by
  sorry

end y_give_z_start_l134_134476


namespace garden_roller_diameter_l134_134448

theorem garden_roller_diameter
  (l : ℝ) (A : ℝ) (r : ℕ) (pi : ℝ)
  (h_l : l = 2)
  (h_A : A = 44)
  (h_r : r = 5)
  (h_pi : pi = 22 / 7) :
  ∃ d : ℝ, d = 1.4 :=
by {
  sorry
}

end garden_roller_diameter_l134_134448


namespace smallest_n_integer_price_l134_134226

theorem smallest_n_integer_price (p : ℚ) (h : ∃ x : ℕ, p = x ∧ 1.06 * p = n) : n = 53 :=
sorry

end smallest_n_integer_price_l134_134226


namespace g_at_100_l134_134597

-- Defining that g is a function from positive real numbers to real numbers
def g : ℝ → ℝ := sorry

-- The given conditions
axiom functional_equation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x * g y - y * g x = g (x / y)

axiom g_one : g 1 = 1

-- The theorem to prove
theorem g_at_100 : g 100 = 50 :=
by
  sorry

end g_at_100_l134_134597


namespace value_of_b_minus_d_squared_l134_134472

theorem value_of_b_minus_d_squared
  (a b c d : ℤ)
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 9) :
  (b - d) ^ 2 = 4 :=
sorry

end value_of_b_minus_d_squared_l134_134472


namespace meaningful_domain_l134_134453

def is_meaningful (x : ℝ) : Prop :=
  (x - 1) ≠ 0

theorem meaningful_domain (x : ℝ) : is_meaningful x ↔ (x ≠ 1) :=
  sorry

end meaningful_domain_l134_134453


namespace difference_of_squares_650_550_l134_134927

theorem difference_of_squares_650_550 : 650^2 - 550^2 = 120000 :=
by sorry

end difference_of_squares_650_550_l134_134927


namespace initial_distance_l134_134839

theorem initial_distance (speed_enrique speed_jamal : ℝ) (hours : ℝ) 
  (h_enrique : speed_enrique = 16) 
  (h_jamal : speed_jamal = 23) 
  (h_time : hours = 8) 
  (h_difference : speed_jamal = speed_enrique + 7) : 
  (speed_enrique * hours + speed_jamal * hours = 312) :=
by 
  sorry

end initial_distance_l134_134839


namespace solve_m_correct_l134_134867

noncomputable def solve_for_m (Q t h : ℝ) : ℝ :=
  if h >= 0 ∧ Q > 0 ∧ t > 0 then
    (Real.log (t / Q)) / (Real.log (1 + Real.sqrt h))
  else
    0 -- Define default output for invalid inputs

theorem solve_m_correct (Q t h : ℝ) (m : ℝ) :
  Q = t / (1 + Real.sqrt h)^m → m = (Real.log (t / Q)) / (Real.log (1 + Real.sqrt h)) :=
by
  intros h1
  rw [h1]
  sorry

end solve_m_correct_l134_134867


namespace product_of_solutions_l134_134845

theorem product_of_solutions :
  (∃ x y : ℝ, (|x^2 - 6 * x| + 5 = 41) ∧ (|y^2 - 6 * y| + 5 = 41) ∧ x ≠ y ∧ x * y = -36) :=
by
  sorry

end product_of_solutions_l134_134845


namespace exponent_problem_l134_134396

theorem exponent_problem (a : ℝ) (m n : ℕ) (h1 : a ^ m = 3) (h2 : a ^ n = 2) : a ^ (m - 2 * n) = 3 / 4 := by
  sorry

end exponent_problem_l134_134396


namespace value_of_expression_l134_134984

theorem value_of_expression (m n : ℝ) (h : m + n = 3) :
  2 * m^2 + 4 * m * n + 2 * n^2 - 6 = 12 :=
by
  sorry

end value_of_expression_l134_134984


namespace unique_spicy_pair_l134_134130

def is_spicy (n : ℕ) : Prop :=
  let A := (n / 100) % 10
  let B := (n / 10) % 10
  let C := n % 10
  n = A^3 + B^3 + C^3

theorem unique_spicy_pair : ∃! n : ℕ, is_spicy n ∧ is_spicy (n + 1) ∧ 100 ≤ n ∧ n < 1000 ∧ n = 370 := 
sorry

end unique_spicy_pair_l134_134130


namespace volume_between_spheres_l134_134459

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem volume_between_spheres :
  volume_of_sphere 10 - volume_of_sphere 4 = (3744 / 3) * Real.pi := by
  sorry

end volume_between_spheres_l134_134459


namespace area_of_square_with_diagonal_two_l134_134877

theorem area_of_square_with_diagonal_two {a d : ℝ} (h : d = 2) (h' : d = a * Real.sqrt 2) : a^2 = 2 := 
by
  sorry

end area_of_square_with_diagonal_two_l134_134877


namespace probability_of_two_prime_numbers_l134_134636

open Finset

noncomputable def primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_of_two_prime_numbers :
  (primes.card.choose 2 / ((range 1 31).card.choose 2) : ℚ) = 1 / 9.67 := sorry

end probability_of_two_prime_numbers_l134_134636


namespace transform_1_to_811_impossible_l134_134482

theorem transform_1_to_811_impossible :
  ∀ (seq_operations : List (ℕ → ℕ)),
    (∀ n, n ∈ seq_operations → (∃ m, n = λ x, (2 * x) ∘ (permute_digits m))) →
    (permute_digits: ℕ → ℕ → ℕ) → -- The function that permutes the digits, given a number and a permutation function
    ∀ n : ℕ, ¬ (1 = 811) := -- Proving that it is impossible for the transformations to result in 811 starting from 1.

begin
  -- Variables definition and initial assumptions
  intro seq_operations,
  intro valid_operations,
  intro permute_digits,
  intro n,

  sorry
end

end transform_1_to_811_impossible_l134_134482


namespace phoenix_equal_roots_implies_a_eq_c_l134_134495

-- Define the "phoenix" equation property
def is_phoenix (a b c : ℝ) : Prop := a + b + c = 0

-- Define the property that a quadratic equation has equal real roots
def has_equal_real_roots (a b c : ℝ) : Prop := b^2 - 4 * a * c = 0

theorem phoenix_equal_roots_implies_a_eq_c (a b c : ℝ) (h₀ : a ≠ 0) 
  (h₁ : is_phoenix a b c) (h₂ : has_equal_real_roots a b c) : a = c :=
sorry

end phoenix_equal_roots_implies_a_eq_c_l134_134495


namespace total_handshakes_l134_134124

-- Definitions and conditions
def num_dwarves := 25
def num_elves := 18

def handshakes_among_dwarves : ℕ := num_dwarves * (num_dwarves - 1) / 2
def handshakes_between_dwarves_and_elves : ℕ := num_elves * num_dwarves

-- Total number of handshakes
theorem total_handshakes : handshakes_among_dwarves + handshakes_between_dwarves_and_elves = 750 := by 
  sorry

end total_handshakes_l134_134124


namespace curve_equation_with_params_l134_134545

theorem curve_equation_with_params (a m x y : ℝ) (ha : a > 0) (hm : m ≠ 0) :
    (y^2) = m * (x^2 - a^2) ↔ mx^2 - y^2 = ma^2 := by
  sorry

end curve_equation_with_params_l134_134545


namespace least_alpha_condition_l134_134286

variables {a b α : ℝ}

theorem least_alpha_condition (a_gt_1 : a > 1) (b_gt_0 : b > 0) : 
  ∀ x, (x ≥ α) → (a + b) ^ x ≥ a ^ x + b ↔ α = 1 :=
by
  sorry

end least_alpha_condition_l134_134286


namespace m_le_n_l134_134696

def polygon : Type := sorry  -- A placeholder definition for polygon.

variables (M : polygon) -- The polygon \( M \)
def max_non_overlapping_circles (M : polygon) : ℕ := sorry -- The maximum number of non-overlapping circles with diameter 1 inside \( M \).
def min_covering_circles (M : polygon) : ℕ := sorry -- The minimum number of circles with radius 1 required to cover \( M \).

theorem m_le_n (M : polygon) : min_covering_circles M ≤ max_non_overlapping_circles M :=
sorry

end m_le_n_l134_134696


namespace percentage_less_than_y_is_70_percent_less_than_z_l134_134173

variable {x y z : ℝ}

theorem percentage_less_than (h1 : x = 1.20 * y) (h2 : x = 0.36 * z) : y = 0.3 * z :=
by
  sorry

theorem y_is_70_percent_less_than_z (h : y = 0.3 * z) : (1 - y / z) * 100 = 70 :=
by
  sorry

end percentage_less_than_y_is_70_percent_less_than_z_l134_134173


namespace combined_depths_underwater_l134_134806

theorem combined_depths_underwater :
  let Ron_height := 12
  let Dean_height := Ron_height - 11
  let Sam_height := Dean_height + 2
  let Ron_depth := Ron_height / 2
  let Sam_depth := Sam_height
  let Dean_depth := Dean_height + 3
  Ron_depth + Sam_depth + Dean_depth = 13 :=
by
  let Ron_height := 12
  let Dean_height := Ron_height - 11
  let Sam_height := Dean_height + 2
  let Ron_depth := Ron_height / 2
  let Sam_depth := Sam_height
  let Dean_depth := Dean_height + 3
  show Ron_depth + Sam_depth + Dean_depth = 13
  sorry

end combined_depths_underwater_l134_134806


namespace compare_y1_y2_l134_134252

noncomputable def quadratic (x : ℝ) : ℝ := -x^2 + 2

theorem compare_y1_y2 :
  let y1 := quadratic 1
  let y2 := quadratic 3
  y1 > y2 :=
by
  let y1 := quadratic 1
  let y2 := quadratic 3
  sorry

end compare_y1_y2_l134_134252


namespace angle_coterminal_l134_134849

theorem angle_coterminal (k : ℤ) : 
  ∃ α : ℝ, α = 30 + k * 360 :=
sorry

end angle_coterminal_l134_134849


namespace david_marks_in_physics_l134_134963

theorem david_marks_in_physics
  (marks_english : ℤ)
  (marks_math : ℤ)
  (marks_chemistry : ℤ)
  (marks_biology : ℤ)
  (average_marks : ℚ)
  (number_of_subjects : ℤ)
  (h_english : marks_english = 96)
  (h_math : marks_math = 98)
  (h_chemistry : marks_chemistry = 100)
  (h_biology : marks_biology = 98)
  (h_average : average_marks = 98.2)
  (h_subjects : number_of_subjects = 5) : 
  ∃ (marks_physics : ℤ), marks_physics = 99 := 
by {
  sorry
}

end david_marks_in_physics_l134_134963


namespace min_length_intersection_l134_134524

theorem min_length_intersection
  (m n : ℝ)
  (hM0 : 0 ≤ m)
  (hM1 : m + 3/4 ≤ 1)
  (hN0 : n - 1/3 ≥ 0)
  (hN1 : n ≤ 1) :
  ∃ x, 0 ≤ x ∧ x ≤ 1 ∧
  x = ((m + 3/4) + (n - 1/3)) - 1 :=
sorry

end min_length_intersection_l134_134524


namespace multiples_sum_squared_l134_134411

theorem multiples_sum_squared :
  let a := 4
  let b := 4
  ((a + b)^2) = 64 :=
by
  sorry

end multiples_sum_squared_l134_134411


namespace sin_identity_l134_134033

theorem sin_identity (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 4) :
  Real.sin (2 * α + π / 6) = 7 / 8 := 
by
  sorry

end sin_identity_l134_134033


namespace isosceles_triangle_perimeter_l134_134992

theorem isosceles_triangle_perimeter (a b c : ℕ) (h_iso : a = b ∨ b = c ∨ c = a)
  (h_triangle_ineq1 : a + b > c) (h_triangle_ineq2 : b + c > a) (h_triangle_ineq3 : c + a > b)
  (h_sides : (a = 2 ∧ b = 2 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 2)) :
  a + b + c = 10 :=
by
  sorry

end isosceles_triangle_perimeter_l134_134992


namespace noah_sales_value_l134_134907

def last_month_large_sales : ℕ := 8
def last_month_small_sales : ℕ := 4
def price_large : ℕ := 60
def price_small : ℕ := 30

def this_month_large_sales : ℕ := 2 * last_month_large_sales
def this_month_small_sales : ℕ := 2 * last_month_small_sales

def this_month_large_sales_value : ℕ := this_month_large_sales * price_large
def this_month_small_sales_value : ℕ := this_month_small_sales * price_small

def this_month_total_sales : ℕ := this_month_large_sales_value + this_month_small_sales_value

theorem noah_sales_value :
  this_month_total_sales = 1200 :=
by
  sorry

end noah_sales_value_l134_134907


namespace problem1_line_equation_problem2_circle_equation_l134_134325

-- Problem 1: Equation of a specific line
def line_intersection (x y : ℝ) : Prop := 
  2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0

def line_perpendicular (x y : ℝ) : Prop :=
  6 * x - 8 * y + 3 = 0

noncomputable def find_line (x y : ℝ) : Prop :=
  ∃ (l : ℝ), (8 * x + 6 * y + l = 0) ∧ 
  line_intersection x y ∧ line_perpendicular x y

theorem problem1_line_equation : ∃ (x y : ℝ), find_line x y :=
sorry

-- Problem 2: Equation of a specific circle
def point_A (x y : ℝ) : Prop := 
  x = 5 ∧ y = 2

def point_B (x y : ℝ) : Prop := 
  x = 3 ∧ y = -2

def center_on_line (x y : ℝ) : Prop :=
  2 * x - y = 3

noncomputable def find_circle (x y r : ℝ) : Prop :=
  ((x - 2)^2 + (y - 1)^2 = r) ∧
  ∃ x1 y1 x2 y2, point_A x1 y1 ∧ point_B x2 y2 ∧ center_on_line x y ∧ ((x1 - x)^2 + (y1 - y)^2 = r)

theorem problem2_circle_equation : ∃ (x y r : ℝ), find_circle x y 10 :=
sorry

end problem1_line_equation_problem2_circle_equation_l134_134325


namespace amount_paid_l134_134298

theorem amount_paid (lemonade_price_per_cup sandwich_price_per_item change_received : ℝ) 
    (num_lemonades num_sandwiches : ℕ)
    (h1 : lemonade_price_per_cup = 2) 
    (h2 : sandwich_price_per_item = 2.50) 
    (h3 : change_received = 11) 
    (h4 : num_lemonades = 2) 
    (h5 : num_sandwiches = 2) : 
    (lemonade_price_per_cup * num_lemonades + sandwich_price_per_item * num_sandwiches + change_received = 20) :=
by
  sorry

end amount_paid_l134_134298


namespace cheaper_to_buy_more_cheaper_2_values_l134_134981

def cost_function (n : ℕ) : ℕ :=
  if (1 ≤ n ∧ n ≤ 30) then 15 * n - 20
  else if (31 ≤ n ∧ n ≤ 55) then 14 * n
  else if (56 ≤ n) then 13 * n + 10
  else 0  -- Assuming 0 for n < 1 as it shouldn't happen in this context

theorem cheaper_to_buy_more_cheaper_2_values : 
  ∃ n1 n2 : ℕ, n1 < n2 ∧ cost_function (n1 + 1) < cost_function n1 ∧ cost_function (n2 + 1) < cost_function n2 ∧
  ∀ n : ℕ, (cost_function (n + 1) < cost_function n → n = n1 ∨ n = n2) := 
sorry

end cheaper_to_buy_more_cheaper_2_values_l134_134981


namespace area_of_feasible_region_l134_134667

theorem area_of_feasible_region :
  (∃ k m : ℝ, (∀ x y : ℝ,
    (kx - y + 1 ≥ 0 ∧ kx - my ≤ 0 ∧ y ≥ 0) ↔
    (x - y + 1 ≥ 0 ∧ x + y ≤ 0 ∧ y ≥ 0)) ∧
    k = 1 ∧ m = -1) →
  ∃ a : ℝ, a = 1 / 4 :=
by sorry

end area_of_feasible_region_l134_134667


namespace rebus_solution_l134_134004

theorem rebus_solution :
  ∃ (A B C : ℕ), 
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l134_134004


namespace price_of_pants_l134_134614

-- Given conditions
variables (P B : ℝ)
axiom h1 : P + B = 70.93
axiom h2 : P = B - 2.93

-- Statement to prove
theorem price_of_pants : P = 34.00 :=
by
  sorry

end price_of_pants_l134_134614


namespace recipe_sugar_amount_l134_134416

-- Definitions from A)
def cups_of_salt : ℕ := 9
def additional_cups_of_sugar (sugar salt : ℕ) : Prop := sugar = salt + 2

-- Statement to prove
theorem recipe_sugar_amount (salt : ℕ) (h : salt = cups_of_salt) : ∃ sugar : ℕ, additional_cups_of_sugar sugar salt ∧ sugar = 11 :=
by
  sorry

end recipe_sugar_amount_l134_134416


namespace reciprocal_check_C_l134_134929

theorem reciprocal_check_C : 0.1 * 10 = 1 := 
by 
  sorry

end reciprocal_check_C_l134_134929


namespace victoria_initial_money_l134_134643

-- Definitions based on conditions
def cost_rice := 2 * 20
def cost_flour := 3 * 25
def cost_soda := 150
def total_spent := cost_rice + cost_flour + cost_soda
def remaining_balance := 235

-- Theorem to prove
theorem victoria_initial_money (initial_money : ℕ) :
  initial_money = total_spent + remaining_balance :=
by
  sorry

end victoria_initial_money_l134_134643


namespace sqrt_expression_meaningful_l134_134718

theorem sqrt_expression_meaningful {x : ℝ} : (2 * x - 4) ≥ 0 → x ≥ 2 :=
by
  intro h
  sorry

end sqrt_expression_meaningful_l134_134718


namespace ratio_d_e_l134_134702

theorem ratio_d_e (a b c d e f : ℝ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : e / f = 1 / 6)
  (h5 : a * b * c / (d * e * f) = 0.25) :
  d / e = 1 / 4 :=
sorry

end ratio_d_e_l134_134702


namespace class_students_l134_134492

theorem class_students (A B : ℕ) 
  (h1 : A + B = 85) 
  (h2 : (3 * A) / 8 + (3 * B) / 5 = 42) : 
  A = 40 ∧ B = 45 :=
by
  sorry

end class_students_l134_134492


namespace find_other_number_l134_134481

theorem find_other_number
  (a b : ℕ)  -- Define the numbers as natural numbers
  (h1 : a = 300)             -- Condition stating the certain number is 300
  (h2 : a = 150 * b)         -- Condition stating the ratio is 150:1
  : b = 2 :=                 -- Goal stating the other number should be 2
  by
    sorry                    -- Placeholder for the proof steps

end find_other_number_l134_134481


namespace undecided_voters_percentage_l134_134188

theorem undecided_voters_percentage
  (biff_percent : ℝ)
  (total_people : ℤ)
  (marty_votes : ℤ)
  (undecided_percent : ℝ) :
  biff_percent = 0.45 →
  total_people = 200 →
  marty_votes = 94 →
  undecided_percent = ((total_people - (marty_votes + (biff_percent * total_people))) / total_people) * 100 →
  undecided_percent = 8 :=
by 
  intros h1 h2 h3 h4
  sorry

end undecided_voters_percentage_l134_134188


namespace exact_sunny_days_probability_l134_134988

noncomputable def choose (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

def rain_prob : ℚ := 3 / 4
def sun_prob : ℚ := 1 / 4
def days : ℕ := 5

theorem exact_sunny_days_probability : (choose days 2 * (sun_prob^2 * rain_prob^3) = 135 / 512) :=
by
  sorry

end exact_sunny_days_probability_l134_134988


namespace william_max_riding_time_l134_134781

theorem william_max_riding_time (x : ℝ) :
  (2 * x + 2 * 1.5 + 2 * (1 / 2 * x) = 21) → (x = 6) :=
by
  sorry

end william_max_riding_time_l134_134781


namespace total_birds_remaining_l134_134921

theorem total_birds_remaining (grey_birds_in_cage : ℕ) (white_birds_next_to_cage : ℕ) :
  (grey_birds_in_cage = 40) →
  (white_birds_next_to_cage = grey_birds_in_cage + 6) →
  (1/2 * grey_birds_in_cage = 20) →
  (1/2 * grey_birds_in_cage + white_birds_next_to_cage = 66) :=
by 
  intros h_grey_birds h_white_birds h_grey_birds_freed
  sorry

end total_birds_remaining_l134_134921


namespace area_of_triangle_l134_134440

theorem area_of_triangle 
  (h : ∀ x y : ℝ, (x / 5 + y / 2 = 1) → ((x = 5 ∧ y = 0) ∨ (x = 0 ∧ y = 2))) : 
  ∃ t : ℝ, t = 1 / 2 * 2 * 5 := 
sorry

end area_of_triangle_l134_134440


namespace existence_of_inf_polynomials_l134_134750

noncomputable def P_xy_defined (P : ℝ→ℝ) (x y z : ℝ) :=
  P x ^ 2 + P y ^ 2 + P z ^ 2 + 2 * P x * P y * P z = 1

theorem existence_of_inf_polynomials (x y z : ℝ) (P : ℕ → ℝ → ℝ) :
  (x^2 + y^2 + z^2 + 2 * x * y * z = 1) →
  (∀ n, P (n+1) = P n ∘ P n) →
  P_xy_defined (P 0) x y z →
  ∀ n, P_xy_defined (P n) x y z :=
by
  intros h1 h2 h3
  sorry

end existence_of_inf_polynomials_l134_134750


namespace cone_cross_section_area_l134_134082

theorem cone_cross_section_area (h α β : ℝ) (h_α_nonneg : 0 ≤ α) (h_β_gt : β > π / 2 - α) :
  ∃ S : ℝ,
    S = (h^2 * Real.sqrt (-Real.cos (α + β) * Real.cos (α - β))) / (Real.cos α * Real.sin β ^ 2) :=
sorry

end cone_cross_section_area_l134_134082


namespace Vasya_mushrooms_l134_134769

def isThreeDigit (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

def digitsSum (n : ℕ) : ℕ := (n / 100) + ((n % 100) / 10) + (n % 10)

theorem Vasya_mushrooms :
  ∃ n : ℕ, isThreeDigit n ∧ digitsSum n = 14 ∧ n = 950 := 
by
  sorry

end Vasya_mushrooms_l134_134769


namespace green_notebook_cost_each_l134_134061

-- Definitions for conditions:
def num_notebooks := 4
def num_green_notebooks := 2
def num_black_notebooks := 1
def num_pink_notebooks := 1
def total_cost := 45
def black_notebook_cost := 15
def pink_notebook_cost := 10

-- Define the problem statement:
theorem green_notebook_cost_each : 
  (2 * g + black_notebook_cost + pink_notebook_cost = total_cost) → 
  g = 10 := 
by 
  intros h
  sorry

end green_notebook_cost_each_l134_134061


namespace john_savings_percentage_l134_134891

theorem john_savings_percentage :
  ∀ (savings discounted_price total_price original_price : ℝ),
  savings = 4.5 →
  total_price = 49.5 →
  total_price = discounted_price * 1.10 →
  original_price = discounted_price + savings →
  (savings / original_price) * 100 = 9 := by
  intros
  sorry

end john_savings_percentage_l134_134891


namespace plates_arrangement_l134_134801

theorem plates_arrangement :
  let B := 5
  let R := 2
  let G := 2
  let O := 1
  let total_plates := B + R + G + O
  let total_arrangements := Nat.factorial total_plates / (Nat.factorial B * Nat.factorial R * Nat.factorial G * Nat.factorial O)
  let circular_arrangements := total_arrangements / total_plates
  let green_adjacent_arrangements := Nat.factorial (total_plates - 1) / (Nat.factorial B * Nat.factorial R * (Nat.factorial (G - 1)) * Nat.factorial O)
  let circular_green_adjacent_arrangements := green_adjacent_arrangements / (total_plates - 1)
  let non_adjacent_green_arrangements := circular_arrangements - circular_green_adjacent_arrangements
  non_adjacent_green_arrangements = 588 :=
by
  let B := 5
  let R := 2
  let G := 2
  let O := 1
  let total_plates := B + R + G + O
  let total_arrangements := Nat.factorial total_plates / (Nat.factorial B * Nat.factorial R * Nat.factorial G * Nat.factorial O)
  let circular_arrangements := total_arrangements / total_plates
  let green_adjacent_arrangements := Nat.factorial (total_plates - 1) / (Nat.factorial B * Nat.factorial R * (Nat.factorial (G - 1)) * Nat.factorial O)
  let circular_green_adjacent_arrangements := green_adjacent_arrangements / (total_plates - 1)
  let non_adjacent_green_arrangements := circular_arrangements - circular_green_adjacent_arrangements
  sorry

end plates_arrangement_l134_134801


namespace choir_members_number_l134_134085

theorem choir_members_number
  (n : ℕ)
  (h1 : n % 12 = 10)
  (h2 : n % 14 = 12)
  (h3 : 300 ≤ n ∧ n ≤ 400) :
  n = 346 :=
sorry

end choir_members_number_l134_134085


namespace piece_length_is_111_l134_134369

-- Define the conditions
axiom condition1 : ∃ (x : ℤ), 9 * x ≤ 1000
axiom condition2 : ∃ (x : ℤ), 9 * x ≤ 1100

-- State the problem: Prove that the length of each piece is 111 centimeters
theorem piece_length_is_111 (x : ℤ) (h1 : 9 * x ≤ 1000) (h2 : 9 * x ≤ 1100) : x = 111 :=
by sorry

end piece_length_is_111_l134_134369


namespace probability_point_outside_circle_l134_134075

/-- Let P be a point with coordinates (m, n) determined by rolling a fair 6-sided die twice.
Prove that the probability that P falls outside the circle x^2 + y^2 = 25 is 7/12. -/
theorem probability_point_outside_circle :
  ∃ (p : ℚ), p = 7/12 ∧
  ∀ (m n : ℕ), (1 ≤ m ∧ m ≤ 6) → (1 ≤ n ∧ n ≤ 6) → 
  (m^2 + n^2 > 25 → p = (7 : ℚ) / 12) :=
sorry

end probability_point_outside_circle_l134_134075


namespace hyperbola_standard_equation_l134_134390

noncomputable def c (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem hyperbola_standard_equation
  (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
  (focus_distance_condition : ∃ (F1 F2 : ℝ), |F1 - F2| = 2 * (c a b))
  (circle_intersects_asymptote : ∃ (x y : ℝ), (x, y) = (1, 2) ∧ y = (b/a) * x + 2): 
  (a = 1) ∧ (b = 2) → (x^2 - (y^2 / 4) = 1) := 
sorry

end hyperbola_standard_equation_l134_134390


namespace distance_from_axis_gt_l134_134699

theorem distance_from_axis_gt 
  (a b x1 x2 y1 y2 : ℝ) (h₁ : a > 0) 
  (h₂ : y1 = a * x1^2 - 2 * a * x1 + b) 
  (h₃ : y2 = a * x2^2 - 2 * a * x2 + b) 
  (h₄ : y1 > y2) : 
  |x1 - 1| > |x2 - 1| := 
sorry

end distance_from_axis_gt_l134_134699


namespace min_value_problem_l134_134010

theorem min_value_problem (a b c d : ℝ) (h : a + 2*b + 3*c + 4*d = 12) : 
    a^2 + b^2 + c^2 + d^2 >= 24 / 5 := 
by
  sorry

end min_value_problem_l134_134010


namespace relationship_between_number_and_value_l134_134483

theorem relationship_between_number_and_value (n v : ℝ) (h1 : n = 7) (h2 : n - 4 = 21 * v) : v = 1 / 7 :=
  sorry

end relationship_between_number_and_value_l134_134483


namespace find_k_l134_134014

theorem find_k (k : ℝ) (A B : ℝ → ℝ)
  (hA : ∀ x, A x = 2 * x^2 + k * x - 6 * x)
  (hB : ∀ x, B x = -x^2 + k * x - 1)
  (hIndependent : ∀ x, ∃ C : ℝ, A x + 2 * B x = C) :
  k = 2 :=
by 
  sorry

end find_k_l134_134014


namespace sum_of_coefficients_zero_l134_134133

theorem sum_of_coefficients_zero (A B C D E F : ℝ) :
  (∀ x : ℝ,
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
      A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
by
  intro h
  -- Proof omitted
  sorry

end sum_of_coefficients_zero_l134_134133


namespace polynomial_evaluation_l134_134866

theorem polynomial_evaluation (n : ℕ) (p : ℕ → ℝ) 
  (h_poly : ∀ k, k ≤ n → p k = 1 / (Nat.choose (n + 1) k)) :
  p (n + 1) = if n % 2 = 0 then 1 else 0 :=
by
  sorry

end polynomial_evaluation_l134_134866


namespace area_of_triangle_l134_134157

theorem area_of_triangle (A B C : ℝ) (a c : ℝ) (d B_value: ℝ) (h1 : A + B + C = 180) 
                         (h2 : A = B - d) (h3 : C = B + d) (h4 : a = 4) (h5 : c = 3)
                         (h6 : B = 60) :
  (1 / 2) * a * c * Real.sin (B * Real.pi / 180) = 3 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l134_134157


namespace cos_value_l134_134250

theorem cos_value (α : ℝ) (h : Real.sin (π / 5 - α) = 1 / 3) : 
  Real.cos (2 * α + 3 * π / 5) = -7 / 9 := by
  sorry

end cos_value_l134_134250


namespace gcd_n4_plus_27_n_plus_3_l134_134686

theorem gcd_n4_plus_27_n_plus_3 (n : ℕ) (h_pos : n > 9) : 
  gcd (n^4 + 27) (n + 3) = if n % 3 = 0 then 3 else 1 := 
by
  sorry

end gcd_n4_plus_27_n_plus_3_l134_134686


namespace gate_paid_more_l134_134120

def pre_booked_economy_cost : Nat := 10 * 140
def pre_booked_business_cost : Nat := 10 * 170
def total_pre_booked_cost : Nat := pre_booked_economy_cost + pre_booked_business_cost

def gate_economy_cost : Nat := 8 * 190
def gate_business_cost : Nat := 12 * 210
def gate_first_class_cost : Nat := 10 * 300
def total_gate_cost : Nat := gate_economy_cost + gate_business_cost + gate_first_class_cost

theorem gate_paid_more {gate_paid_more_cost : Nat} :
  total_gate_cost - total_pre_booked_cost = 3940 :=
by
  sorry

end gate_paid_more_l134_134120


namespace trig_inequality_l134_134088

theorem trig_inequality : Real.tan 1 > Real.sin 1 ∧ Real.sin 1 > Real.cos 1 := by
  sorry

end trig_inequality_l134_134088


namespace problem1_problem2_problem3_l134_134151

-- Definitions of arithmetic and geometric sequences
def arithmetic (a_n : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a_n n = a_n 0 + n * d
def geometric (b_n : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, b_n n = b_n 0 * q ^ n
def E (m p r : ℕ) := m < p ∧ p < r
def common_difference_greater_than_one (m p r : ℕ) := (p - m = r - p) ∧ (p - m > 1)

-- Problem (1)
theorem problem1 (a_n b_n : ℕ → ℝ) (d q : ℝ) (h₁: arithmetic a_n d) (h₂: geometric b_n q) (hne: q ≠ 1 ∧ q ≠ -1)
  (h: a_n 0 + b_n 1 = a_n 1 + b_n 2 ∧ a_n 1 + b_n 2 = a_n 2 + b_n 0) :
  q = -1/2 :=
sorry

-- Problem (2)
theorem problem2 (a_n b_n : ℕ → ℝ) (d q : ℝ) (m p r : ℕ) (h₁: arithmetic a_n d) (h₂: geometric b_n q) (hne: q ≠ 1 ∧ q ≠ -1)
  (hE: E m p r) (hDiff: common_difference_greater_than_one m p r)
  (h: a_n m + b_n p = a_n p + b_n r ∧ a_n p + b_n r = a_n r + b_n m) :
  q = - (1/2)^(1/3) :=
sorry

-- Problem (3)
theorem problem3 (a_n b_n : ℕ → ℝ) (m p r : ℕ) (hE: E m p r)
  (hG: ∀ n : ℕ, b_n n = (-1/2)^((n:ℕ)-1)) (h: a_n m + b_n m = 0 ∧ a_n p + b_n p = 0 ∧ a_n r + b_n r = 0) :
  ∃ (E : ℕ × ℕ × ℕ) (a : ℕ → ℝ), (E = ⟨1, 3, 4⟩ ∧ ∀ n : ℕ, a n = 3/8 * n - 11/8) :=
sorry

end problem1_problem2_problem3_l134_134151


namespace pradeep_passing_percentage_l134_134746

-- Define the constants based on the conditions
def totalMarks : ℕ := 550
def marksObtained : ℕ := 200
def marksFailedBy : ℕ := 20

-- Calculate the passing marks
def passingMarks : ℕ := marksObtained + marksFailedBy

-- Define the percentage calculation as a noncomputable function
noncomputable def requiredPercentageToPass : ℚ := (passingMarks / totalMarks) * 100

-- The theorem to prove
theorem pradeep_passing_percentage :
  requiredPercentageToPass = 40 := 
sorry

end pradeep_passing_percentage_l134_134746


namespace part_a_part_b_l134_134029

noncomputable def sequence (n : ℕ) : ℝ := Real.cos (Real.pi * 10 ^ n / 180)

theorem part_a : (sequence 100) > 0 := 
by {
  sorry
}

theorem part_b : abs (sequence 100) < 0.18 :=
by {
  sorry
}

end part_a_part_b_l134_134029


namespace reflection_angle_sum_l134_134285

open EuclideanGeometry

theorem reflection_angle_sum (ABC : Triangle) (J : Point) (K : Point) (E F : Point) :
  is_excenter_J A ABC ->
  reflection J (line_through B C) = K ->
  (E ∈ line_through B J) ∧ (F ∈ line_through C J) ->
  (\<angle EAB = 90) ∧ (\<angle CAF = 90) ->
  (\<angle FKE + \<angle FJE = 180) :=
  sorry

end reflection_angle_sum_l134_134285


namespace a_100_positive_a_100_abs_lt_018_l134_134025

-- Define the sequence based on the given conditions
def a_n (n : ℕ) : ℝ := real.cos (real.pi / 180 * (10^n) : ℝ)

-- Translate mathematical claims to Lean theorems
theorem a_100_positive : a_n 100 > 0 :=
sorry

theorem a_100_abs_lt_018 : |a_n 100| < 0.18 :=
sorry

end a_100_positive_a_100_abs_lt_018_l134_134025


namespace regression_passes_through_none_l134_134309

theorem regression_passes_through_none (b a x y : ℝ) (h₀ : (0, 0) ≠ (0*b + a, 0))
                                     (h₁ : (x, 0) ≠ (x*b + a, 0))
                                     (h₂ : (x, y) ≠ (x*b + a, y)) : 
                                     ¬ ((0, 0) = (0*b + a, 0) ∨ (x, 0) = (x*b + a, 0) ∨ (x, y) = (x*b + a, y)) :=
by sorry

end regression_passes_through_none_l134_134309


namespace animal_shelter_kittens_count_l134_134618

def num_puppies : ℕ := 32
def num_kittens_more : ℕ := 14

theorem animal_shelter_kittens_count : 
  ∃ k : ℕ, k = (2 * num_puppies) + num_kittens_more := 
sorry

end animal_shelter_kittens_count_l134_134618


namespace tony_average_time_l134_134576

-- Definitions for the conditions
def speed_walk : ℝ := 2  -- speed in miles per hour when Tony walks
def speed_run : ℝ := 10  -- speed in miles per hour when Tony runs
def distance_to_store : ℝ := 4  -- distance to the store in miles
def days : List String := ["Sunday", "Tuesday", "Thursday"]  -- days Tony goes to the store

-- Definition of times taken on each day
def time_sunday := distance_to_store / speed_walk  -- time in hours to get to the store on Sunday
def time_tuesday := distance_to_store / speed_run  -- time in hours to get to the store on Tuesday
def time_thursday := distance_to_store / speed_run -- time in hours to get to the store on Thursday

-- Converting times to minutes
def time_sunday_minutes := time_sunday * 60
def time_tuesday_minutes := time_tuesday * 60
def time_thursday_minutes := time_thursday * 60

-- Definition of average time
def average_time_minutes : ℝ :=
  (time_sunday_minutes + time_tuesday_minutes + time_thursday_minutes) / days.length

-- The theorem to prove
theorem tony_average_time : average_time_minutes = 56 := by
  sorry

end tony_average_time_l134_134576


namespace roots_opposite_sign_eq_magnitude_l134_134349

theorem roots_opposite_sign_eq_magnitude (c d e n : ℝ) (h : ((n+2) * (x^2 + c*x + d)) = (n-2) * (2*x - e)) :
  n = (-4 - 2 * c) / (c - 2) :=
by
  sorry

end roots_opposite_sign_eq_magnitude_l134_134349


namespace difference_in_pages_l134_134355

def purple_pages_per_book : ℕ := 230
def orange_pages_per_book : ℕ := 510
def purple_books_read : ℕ := 5
def orange_books_read : ℕ := 4

theorem difference_in_pages : 
  orange_books_read * orange_pages_per_book - purple_books_read * purple_pages_per_book = 890 :=
by
  sorry

end difference_in_pages_l134_134355


namespace same_terminal_side_angle_exists_l134_134077

theorem same_terminal_side_angle_exists :
  ∃ k : ℤ, -5 * π / 8 + 2 * k * π = 11 * π / 8 := 
by
  sorry

end same_terminal_side_angle_exists_l134_134077


namespace find_b_l134_134611

noncomputable def geom_seq_term (a b c : ℝ) : Prop :=
∃ r : ℝ, r > 0 ∧ b = a * r ∧ c = b * r

theorem find_b (b : ℝ) (h_geom : geom_seq_term 160 b (108 / 64)) (h_pos : b > 0) :
  b = 15 * Real.sqrt 6 :=
by
  sorry

end find_b_l134_134611


namespace f_even_function_l134_134450

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_even_function : ∀ x : ℝ, f x = f (-x) :=
by
  intro x
  show f x = f (-x)
  sorry

end f_even_function_l134_134450


namespace constant_term_in_expansion_l134_134301

-- Define the binomial expansion general term
def binomial_general_term (x : ℤ) (r : ℕ) : ℤ :=
  (-2)^r * 3^(5 - r) * (Nat.choose 5 r) * x^(10 - 5 * r)

-- Define the condition for the specific r that makes the exponent of x zero
def condition (r : ℕ) : Prop :=
  10 - 5 * r = 0

-- Define the constant term calculation
def const_term : ℤ :=
  4 * 27 * (Nat.choose 5 2)

-- Theorem statement
theorem constant_term_in_expansion : const_term = 1080 :=
by 
  -- The proof is omitted
  sorry

end constant_term_in_expansion_l134_134301


namespace fib_mod_13_multiples_count_l134_134235

noncomputable def fib_mod (n : ℕ) : ℕ := Nat.fib n % 13

def is_multiple_of_13 (n : ℕ) : Prop := fib_mod n = 0

def count_fib_multiples_of_13 (upper_bound : ℕ) : ℕ :=
  Nat.length (List.filter is_multiple_of_13 (List.range (upper_bound + 1)))

theorem fib_mod_13_multiples_count :
  count_fib_multiples_of_13 100 = 15 :=
sorry

end fib_mod_13_multiples_count_l134_134235


namespace problem_condition_l134_134256

variable {f : ℝ → ℝ}

theorem problem_condition (h_diff : Differentiable ℝ f) (h_ineq : ∀ x : ℝ, f x < iteratedDeriv 2 f x) : 
  e^2019 * f (-2019) < f 0 ∧ f 2019 > e^2019 * f 0 :=
by
  sorry

end problem_condition_l134_134256


namespace solution_set_of_absolute_value_inequality_l134_134763

theorem solution_set_of_absolute_value_inequality {x : ℝ} : 
  (|2 * x - 3| > 1) ↔ (x < 1 ∨ x > 2) := 
sorry

end solution_set_of_absolute_value_inequality_l134_134763


namespace carrots_cost_l134_134563

/-
Define the problem conditions and parameters.
-/
def num_third_grade_classes := 5
def students_per_third_grade_class := 30
def num_fourth_grade_classes := 4
def students_per_fourth_grade_class := 28
def num_fifth_grade_classes := 4
def students_per_fifth_grade_class := 27

def cost_per_hamburger : ℝ := 2.10
def cost_per_cookie : ℝ := 0.20
def total_lunch_cost : ℝ := 1036

/-
Calculate the total number of students.
-/
def total_students : ℕ :=
  (num_third_grade_classes * students_per_third_grade_class) +
  (num_fourth_grade_classes * students_per_fourth_grade_class) +
  (num_fifth_grade_classes * students_per_fifth_grade_class)

/-
Calculate the cost of hamburgers and cookies.
-/
def hamburgers_cost : ℝ := total_students * cost_per_hamburger
def cookies_cost : ℝ := total_students * cost_per_cookie
def total_hamburgers_and_cookies_cost : ℝ := hamburgers_cost + cookies_cost

/-
State the proof problem: How much do the carrots cost?
-/
theorem carrots_cost : total_lunch_cost - total_hamburgers_and_cookies_cost = 185 :=
by
  -- Proof is omitted
  sorry

end carrots_cost_l134_134563


namespace combined_age_of_Jane_and_John_in_future_l134_134552

def Justin_age : ℕ := 26
def Jessica_age_when_Justin_born : ℕ := 6
def James_older_than_Jessica : ℕ := 7
def Julia_younger_than_Justin : ℕ := 8
def Jane_older_than_James : ℕ := 25
def John_older_than_Jane : ℕ := 3
def years_later : ℕ := 12

theorem combined_age_of_Jane_and_John_in_future :
  let Jessica_age := Justin_age + Jessica_age_when_Justin_born
  let James_age := Jessica_age + James_older_than_Jessica
  let Julia_age := Justin_age - Julia_younger_than_Justin
  let Jane_age := James_age + Jane_older_than_James
  let John_age := Jane_age + John_older_than_Jane
  let Jane_age_after_years := Jane_age + years_later
  let John_age_after_years := John_age + years_later
  Jane_age_after_years + John_age_after_years = 155 :=
by
  sorry

end combined_age_of_Jane_and_John_in_future_l134_134552


namespace free_space_on_new_drive_l134_134585

theorem free_space_on_new_drive
  (initial_free : ℝ) (initial_used : ℝ) (delete_size : ℝ) (new_files_size : ℝ) (new_drive_size : ℝ) :
  initial_free = 2.4 → initial_used = 12.6 → delete_size = 4.6 → new_files_size = 2 → new_drive_size = 20 →
  (new_drive_size - ((initial_used - delete_size) + new_files_size)) = 10 :=
by simp; sorry

end free_space_on_new_drive_l134_134585


namespace chlorine_discount_l134_134800

theorem chlorine_discount
  (cost_chlorine : ℕ)
  (cost_soap : ℕ)
  (num_chlorine : ℕ)
  (num_soap : ℕ)
  (discount_soap : ℤ)
  (total_savings : ℤ)
  (price_chlorine : ℤ)
  (price_soap_after_discount : ℤ)
  (total_price_before_discount : ℤ)
  (total_price_after_discount : ℤ)
  (goal_discount : ℤ) :
  cost_chlorine = 10 →
  cost_soap = 16 →
  num_chlorine = 3 →
  num_soap = 5 →
  discount_soap = 25 →
  total_savings = 26 →
  price_soap_after_discount = (1 - (discount_soap / 100)) * 16 →
  total_price_before_discount = (num_chlorine * cost_chlorine) + (num_soap * cost_soap) →
  total_price_after_discount = (num_chlorine * ((100 - goal_discount) / 100) * cost_chlorine) + (num_soap * 12) →
  total_price_before_discount - total_price_after_discount = total_savings →
  goal_discount = 20 :=
by
  intros
  sorry

end chlorine_discount_l134_134800


namespace power_identity_l134_134859

theorem power_identity (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 3) : x^(3 * a + 2 * b) = 72 := 
  sorry

end power_identity_l134_134859


namespace money_spent_on_paintbrushes_l134_134462

-- Define the conditions
def total_spent : ℝ := 90.00
def cost_canvases : ℝ := 40.00
def cost_paints : ℝ := cost_canvases / 2
def cost_easel : ℝ := 15.00

-- Define the problem
theorem money_spent_on_paintbrushes : total_spent - (cost_canvases + cost_paints + cost_easel) = 15.00 :=
by sorry

end money_spent_on_paintbrushes_l134_134462


namespace min_points_to_guarantee_win_l134_134277

theorem min_points_to_guarantee_win (P Q R S: ℕ) (bonus: ℕ) :
    (P = 6 ∨ P = 4 ∨ P = 2) ∧ (Q = 6 ∨ Q = 4 ∨ Q = 2) ∧ 
    (R = 6 ∨ R = 4 ∨ R = 2) ∧ (S = 6 ∨ S = 4 ∨ S = 2) →
    (bonus = 3 ↔ ((P = 6 ∧ Q = 4 ∧ R = 2) ∨ (P = 6 ∧ Q = 2 ∧ R = 4) ∨ 
                   (P = 4 ∧ Q = 6 ∧ R = 2) ∨ (P = 4 ∧ Q = 2 ∧ R = 6) ∨ 
                   (P = 2 ∧ Q = 6 ∧ R = 4) ∨ (P = 2 ∧ Q = 4 ∧ R = 6))) →
    (P + Q + R + S + bonus ≥ 24) :=
by sorry

end min_points_to_guarantee_win_l134_134277


namespace probability_of_odd_number_l134_134911

theorem probability_of_odd_number (total_outcomes : ℕ) (odd_outcomes : ℕ) (h1 : total_outcomes = 6) (h2 : odd_outcomes = 3) : (odd_outcomes / total_outcomes : ℚ) = 1 / 2 :=
by
  sorry 

end probability_of_odd_number_l134_134911


namespace three_digit_division_l134_134497

theorem three_digit_division (abc : ℕ) (a b c : ℕ) (h1 : 100 ≤ abc ∧ abc < 1000) (h2 : abc = 100 * a + 10 * b + c) (h3 : a ≠ 0) :
  (1001 * abc) / 7 / 11 / 13 = abc :=
by
  sorry

end three_digit_division_l134_134497


namespace find_retail_price_l134_134207

-- Define the wholesale price
def wholesale_price : ℝ := 90

-- Define the profit as 20% of the wholesale price
def profit (w : ℝ) : ℝ := 0.2 * w

-- Define the selling price as the wholesale price plus the profit
def selling_price (w p : ℝ) : ℝ := w + p

-- Define the selling price as 90% of the retail price t
def discount_selling_price (t : ℝ) : ℝ := 0.9 * t

-- Prove that the retail price t is 120 given the conditions
theorem find_retail_price :
  ∃ t : ℝ, wholesale_price + (profit wholesale_price) = discount_selling_price t → t = 120 :=
by
  sorry

end find_retail_price_l134_134207


namespace school_competition_students_l134_134603

theorem school_competition_students (n : ℤ)
  (h1 : 100 < n) 
  (h2 : n < 200) 
  (h3 : n % 4 = 2) 
  (h4 : n % 5 = 2) 
  (h5 : n % 6 = 2) :
  n = 122 ∨ n = 182 :=
sorry

end school_competition_students_l134_134603


namespace shaded_figure_perimeter_l134_134464

theorem shaded_figure_perimeter (a b : ℝ) (area_overlap : ℝ) (side_length : ℝ) (side_length_overlap : ℝ):
    a = 5 → b = 5 → area_overlap = 4 → side_length_overlap * side_length_overlap = area_overlap →
    side_length_overlap = 2 →
    ((4 * a) + (4 * b) - (4 * side_length_overlap)) = 32 :=
by
  intros
  sorry

end shaded_figure_perimeter_l134_134464


namespace last_digit_of_expression_l134_134713

-- Conditions
def a : ℤ := 25
def b : ℤ := -3

-- Statement to be proved
theorem last_digit_of_expression :
  (a ^ 1999 + b ^ 2002) % 10 = 4 :=
by
  -- proof would go here
  sorry

end last_digit_of_expression_l134_134713


namespace fg_square_diff_l134_134516

open Real

noncomputable def f (x: ℝ) : ℝ := sorry
noncomputable def g (x: ℝ) : ℝ := sorry

axiom h1 (x: ℝ) (hx : -π / 2 < x ∧ x < π / 2) : f x + g x = sqrt ((1 + cos (2 * x)) / (1 - sin x))
axiom h2 : ∀ x, f (-x) = -f x
axiom h3 : ∀ x, g (-x) = g x

theorem fg_square_diff (x : ℝ) (hx : -π / 2 < x ∧ x < π / 2) : (f x)^2 - (g x)^2 = -2 * cos x := 
sorry

end fg_square_diff_l134_134516


namespace x_less_than_y_by_35_percent_l134_134274

noncomputable def percentage_difference (x y : ℝ) : ℝ :=
  ((y / x) - 1) * 100

theorem x_less_than_y_by_35_percent (x y : ℝ) (h : y = 1.5384615384615385 * x) :
  percentage_difference x y = 53.846153846153854 :=
by
  sorry

end x_less_than_y_by_35_percent_l134_134274


namespace hemisphere_surface_area_l134_134789

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (h1: 0 < π) (h2: A = 3) (h3: S = 4 * π * r^2):
  ∃ t, t = 9 :=
by
  sorry

end hemisphere_surface_area_l134_134789


namespace rounding_sum_eq_one_third_probability_l134_134299

noncomputable def rounding_sum_probability : ℝ :=
  (λ (total : ℝ) => 
    let round := (λ (x : ℝ) => if x < 0.5 then 0 else if x < 1.5 then 1 else if x < 2.5 then 2 else 3)
    let interval := (λ (start : ℝ) (end_ : ℝ) => end_ - start)
    let sum_conditions := [((0.5,1.5), 3), ((1.5,2.5), 2)]
    let total_length := 3

    let valid_intervals := sum_conditions.map (λ p => interval (p.fst.fst) (p.fst.snd))
    let total_valid_interval := List.sum valid_intervals
    total_valid_interval / total_length
  ) 3

theorem rounding_sum_eq_one_third_probability : rounding_sum_probability = 2 / 3 := by sorry

end rounding_sum_eq_one_third_probability_l134_134299


namespace sandy_correct_sums_l134_134211

-- Definitions based on the conditions
variables (c i : ℕ)

-- Conditions as Lean statements
axiom h1 : 3 * c - 2 * i = 65
axiom h2 : c + i = 30

-- Proof goal
theorem sandy_correct_sums : c = 25 := 
by
  sorry

end sandy_correct_sums_l134_134211


namespace number_of_piles_l134_134041

-- Defining the number of walnuts in total
def total_walnuts : Nat := 55

-- Defining the number of walnuts in the first pile
def first_pile_walnuts : Nat := 7

-- Defining the number of walnuts in each of the rest of the piles
def other_pile_walnuts : Nat := 12

-- The proposition we want to prove
theorem number_of_piles (n : Nat) :
  (n > 1) →
  (other_pile_walnuts * (n - 1) + first_pile_walnuts = total_walnuts) → n = 5 :=
sorry

end number_of_piles_l134_134041


namespace smallest_four_digit_equiv_8_mod_9_l134_134775

theorem smallest_four_digit_equiv_8_mod_9 :
  ∃ n : ℕ, n % 9 = 8 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 9 = 8 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m :=
sorry

end smallest_four_digit_equiv_8_mod_9_l134_134775


namespace smallest_positive_integer_divisible_by_10_13_14_l134_134013

theorem smallest_positive_integer_divisible_by_10_13_14 : ∃ n : ℕ, n > 0 ∧ (10 ∣ n) ∧ (13 ∣ n) ∧ (14 ∣ n) ∧ n = 910 :=
by {
  sorry
}

end smallest_positive_integer_divisible_by_10_13_14_l134_134013


namespace unoccupied_seats_l134_134836

theorem unoccupied_seats 
    (seats_per_row : ℕ) 
    (rows : ℕ) 
    (seatable_fraction : ℚ) 
    (total_seats := seats_per_row * rows) 
    (seatable_seats_per_row := (seatable_fraction * seats_per_row)) 
    (seatable_seats := seatable_seats_per_row * rows) 
    (unoccupied_seats := total_seats - seatable_seats) {
  seats_per_row = 8, 
  rows = 12, 
  seatable_fraction = 3/4 
  : unoccupied_seats = 24 :=
by
  sorry

end unoccupied_seats_l134_134836


namespace grapes_total_sum_l134_134809

theorem grapes_total_sum (R A N : ℕ) 
  (h1 : A = R + 2) 
  (h2 : N = A + 4) 
  (h3 : R = 25) : 
  R + A + N = 83 := by
  sorry

end grapes_total_sum_l134_134809


namespace more_regular_than_diet_l134_134114

-- Define the conditions
def num_regular_soda : Nat := 67
def num_diet_soda : Nat := 9

-- State the theorem
theorem more_regular_than_diet :
  num_regular_soda - num_diet_soda = 58 :=
by
  sorry

end more_regular_than_diet_l134_134114


namespace a_older_than_b_l134_134222

theorem a_older_than_b (A B : ℕ) (h1 : B = 36) (h2 : A + 10 = 2 * (B - 10)) : A - B = 6 :=
  sorry

end a_older_than_b_l134_134222


namespace oak_taller_than_shortest_l134_134744

noncomputable def pine_tree_height : ℚ := 14 + 1 / 2
noncomputable def elm_tree_height : ℚ := 13 + 1 / 3
noncomputable def oak_tree_height : ℚ := 19 + 1 / 2

theorem oak_taller_than_shortest : 
  oak_tree_height - elm_tree_height = 6 + 1 / 6 := 
  sorry

end oak_taller_than_shortest_l134_134744


namespace savings_with_discount_l134_134184

theorem savings_with_discount :
  let original_price := 3.00
  let discount_rate := 0.30
  let discounted_price := original_price * (1 - discount_rate)
  let number_of_notebooks := 7
  let total_cost_without_discount := number_of_notebooks * original_price
  let total_cost_with_discount := number_of_notebooks * discounted_price
  total_cost_without_discount - total_cost_with_discount = 6.30 :=
by
  sorry

end savings_with_discount_l134_134184


namespace smallest_solution_l134_134503

noncomputable def equation (x : ℝ) : Prop :=
  (1 / (x - 1)) + (1 / (x - 5)) = 4 / (x - 4)

theorem smallest_solution : 
  ∃ x : ℝ, equation x ∧ x ≠ 1 ∧ x ≠ 5 ∧ x ≠ 4 ∧ x = (5 - Real.sqrt 33) / 2 :=
by
  sorry

end smallest_solution_l134_134503


namespace parabola_opens_downwards_iff_l134_134021

theorem parabola_opens_downwards_iff (a : ℝ) : (∀ x : ℝ, (a - 1) * x^2 + 2 * x ≤ 0) ↔ a < 1 := 
sorry

end parabola_opens_downwards_iff_l134_134021


namespace sqrt_expression_meaningful_l134_134724

theorem sqrt_expression_meaningful (x : ℝ) : (2 * x - 4 ≥ 0) ↔ (x ≥ 2) :=
by
  -- Proof will be skipped
  sorry

end sqrt_expression_meaningful_l134_134724


namespace find_quotient_from_conditions_l134_134595

variable (x y : ℕ)
variable (k : ℕ)

theorem find_quotient_from_conditions :
  y - x = 1360 ∧ y = 1614 ∧ y % x = 15 → y / x = 6 :=
by
  intro h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end find_quotient_from_conditions_l134_134595


namespace yulia_max_candies_l134_134565

def maxCandies (totalCandies : ℕ) (horizontalCandies : ℕ) (verticalCandies : ℕ) (diagonalCandies : ℕ) : ℕ :=
  totalCandies - min (2 * horizontalCandies + 3 * diagonalCandies) (3 * diagonalCandies + 2 * verticalCandies)

-- Constants
def totalCandies : ℕ := 30
def horizontalMoveCandies : ℕ := 2
def verticalMoveCandies : ℕ := 2
def diagonalMoveCandies : ℕ := 3
def path1_horizontalMoves : ℕ := 5
def path1_diagonalMoves : ℕ := 2
def path2_verticalMoves : ℕ := 1
def path2_diagonalMoves : ℕ := 5

theorem yulia_max_candies :
  maxCandies totalCandies (path1_horizontalMoves + path2_verticalMoves) 0 (path1_diagonalMoves + path2_diagonalMoves) = 14 :=
by
  sorry

end yulia_max_candies_l134_134565


namespace sample_mean_experimental_group_median_and_significance_l134_134672

namespace OzoneExperiment

def control_group : List ℝ := 
  [15.2, 18.8, 20.2, 21.3, 22.5, 23.2, 25.8, 26.5, 27.5, 30.1,
   32.6, 34.3, 34.8, 35.6, 35.6, 35.8, 36.2, 37.3, 40.5, 43.2]

def experimental_group : List ℝ := 
  [7.8, 9.2, 11.4, 12.4, 13.2, 15.5, 16.5, 18.0, 18.8, 19.2, 
   19.8, 20.2, 21.6, 22.8, 23.6, 23.9, 25.1, 28.2, 32.3, 36.5]

def combined : List ℝ :=
  control_group ++ experimental_group

-- Sample mean calculation
theorem sample_mean_experimental_group
  (ex_group_sum : ∑ x in experimental_group, x = 396 ) :
  (∑ x in experimental_group, x) / 20 = 19.8 :=
begin
  have divisor := 20,
  calc (∑ x in experimental_group, x) / divisor
      = 396 / divisor : by rw ex_group_sum
  ... = 19.8 : by norm_num
end

-- Median calculation and significance
theorem median_and_significance
  (sorted_combined := combined.sort (≤))
  (median_calculation : (sorted_combined[19] + sorted_combined[20]) / 2 = 23.4)
  (a b c d : ℕ) (h_table : a = 6 ∧ b = 14 ∧ c = 14 ∧ d = 6)
  (h_ksquare : (40 * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d)) = 6.4)
  (h_critical_value : 6.4 > 3.841) : 
  m = 23.4 ∧ (6.4 > 3.841) :=
begin
  sorry
end

end OzoneExperiment

end sample_mean_experimental_group_median_and_significance_l134_134672


namespace animal_shelter_kittens_count_l134_134617

def num_puppies : ℕ := 32
def num_kittens_more : ℕ := 14

theorem animal_shelter_kittens_count : 
  ∃ k : ℕ, k = (2 * num_puppies) + num_kittens_more := 
sorry

end animal_shelter_kittens_count_l134_134617


namespace painting_price_decrease_l134_134761

theorem painting_price_decrease (P : ℝ) (h1 : 1.10 * P - 0.935 * P = x * 1.10 * P) :
  x = 0.15 := by
  sorry

end painting_price_decrease_l134_134761


namespace race_problem_equivalent_l134_134336

noncomputable def race_track_distance (D_paved D_dirt D_muddy : ℝ) : Prop :=
  let v1 := 100 -- speed on paved section in km/h
  let v2 := 70  -- speed on dirt section in km/h
  let v3 := 15  -- speed on muddy section in km/h
  let initial_distance := 0.5 -- initial distance in km (since 500 meters is 0.5 km)
  
  -- Time to cover paved section
  let t_white_paved := D_paved / v1
  let t_red_paved := (D_paved - initial_distance) / v1

  -- Times to cover dirt section
  let t_white_dirt := D_dirt / v2
  let t_red_dirt := D_dirt / v2 -- same time since both start at the same time on dirt

  -- Times to cover muddy section
  let t_white_muddy := D_muddy / v3
  let t_red_muddy := D_muddy / v3 -- same time since both start at the same time on mud

  -- Distances between cars on dirt and muddy sections
  ((t_white_paved - t_red_paved) * v2 = initial_distance) ∧ 
  ((t_white_paved - t_red_paved) * v3 = initial_distance)

-- Prove the distance between the cars when both are on the dirt and muddy sections is 500 meters
theorem race_problem_equivalent (D_paved D_dirt D_muddy : ℝ) : race_track_distance D_paved D_dirt D_muddy :=
by
  -- Insert proof here, for now we use sorry
  sorry

end race_problem_equivalent_l134_134336


namespace rebus_solution_l134_134007

theorem rebus_solution :
  ∃ (A B C : ℕ), 
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l134_134007


namespace no_solution_exists_l134_134501

theorem no_solution_exists (f : ℝ → ℝ) :
  ¬ (∀ x y : ℝ, f (f x + 2 * y) = 3 * x + f (f (f y) - x)) :=
sorry

end no_solution_exists_l134_134501


namespace part_I_part_II_part_III_l134_134260

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 2) - (1 / (2^x + 1))

theorem part_I :
  ∃ a : ℝ, ∀ x : ℝ, f x = a - (1 / (2^x + 1)) → a = (1 / 2) :=
by sorry

theorem part_II :
  ∀ y : ℝ, y = f x → (-1 / 2) < y ∧ y < (1 / 2) :=
by sorry

theorem part_III :
  ∀ m n : ℝ, m + n ≠ 0 → (f m + f n) / (m^3 + n^3) > f 0 :=
by sorry

end part_I_part_II_part_III_l134_134260


namespace prob_A_wins_correct_l134_134465

noncomputable def prob_A_wins : ℚ :=
  let outcomes : ℕ := 3^3
  let win_one_draw_two : ℕ := 3
  let win_two_other : ℕ := 6
  let win_all : ℕ := 1
  let total_wins : ℕ := win_one_draw_two + win_two_other + win_all
  total_wins / outcomes

theorem prob_A_wins_correct :
  prob_A_wins = 10/27 :=
by
  sorry

end prob_A_wins_correct_l134_134465


namespace acute_angle_parallel_vectors_l134_134045

theorem acute_angle_parallel_vectors (x : ℝ) (a b : ℝ × ℝ)
    (h₁ : a = (Real.sin x, 1))
    (h₂ : b = (1 / 2, Real.cos x))
    (h₃ : ∃ k : ℝ, a = k • b ∧ k ≠ 0) :
    x = Real.pi / 4 :=
by
  sorry

end acute_angle_parallel_vectors_l134_134045


namespace tan_triple_angle_l134_134875

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_triple_angle_l134_134875


namespace number_of_technicians_l134_134442

-- Define the problem statements
variables (T R : ℕ)

-- Conditions based on the problem description
def condition1 : Prop := T + R = 42
def condition2 : Prop := 3 * T + R = 56

-- The main goal to prove
theorem number_of_technicians (h1 : condition1 T R) (h2 : condition2 T R) : T = 7 :=
by
  sorry -- Proof is omitted as per instructions

end number_of_technicians_l134_134442


namespace smallest_solution_eq_l134_134504

theorem smallest_solution_eq (x : ℝ) (hneq1 : x ≠ 1) (hneq5 : x ≠ 5) (hneq4 : x ≠ 4) :
  (∃ x : ℝ, (1 / (x - 1)) + (1 / (x - 5)) = (4 / (x - 4)) ∧
            (∀ y : ℝ, (1 / (y - 1)) + (1 / (y - 5)) = (4 / (y - 4)) → x ≤ y → y = x) ∧
            x = (5 - Real.sqrt 33) / 2) := 
begin
  sorry
end

end smallest_solution_eq_l134_134504


namespace moneySpentOnPaintbrushes_l134_134461

def totalExpenditure := 90
def costOfCanvases := 40
def costOfPaints := costOfCanvases / 2
def costOfEasel := 15
def costOfOthers := costOfCanvases + costOfPaints + costOfEasel

theorem moneySpentOnPaintbrushes : totalExpenditure - costOfOthers = 15 := by
  sorry

end moneySpentOnPaintbrushes_l134_134461


namespace intersection_A_B_l134_134393

def setA : Set ℝ := { x | x^2 - 2*x < 3 }
def setB : Set ℝ := { x | x ≤ 2 }
def setC : Set ℝ := { x | -1 < x ∧ x ≤ 2 }

theorem intersection_A_B :
  (setA ∩ setB) = setC :=
by
  sorry

end intersection_A_B_l134_134393


namespace total_weight_of_beef_l134_134087

-- Define the conditions
def packages_weight := 4
def first_butcher_packages := 10
def second_butcher_packages := 7
def third_butcher_packages := 8

-- Define the total weight calculation
def total_weight := (first_butcher_packages * packages_weight) +
                    (second_butcher_packages * packages_weight) +
                    (third_butcher_packages * packages_weight)

-- The statement to prove
theorem total_weight_of_beef : total_weight = 100 := by
  -- proof goes here
  sorry

end total_weight_of_beef_l134_134087


namespace rebus_solution_l134_134000

theorem rebus_solution :
  ∃ (A B C : ℕ), A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    (A*100 + B*10 + A) + (A*100 + B*10 + C) + (A*100 + C*10 + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by {
  sorry
}

end rebus_solution_l134_134000


namespace csc_neg_45_eq_neg_sqrt2_l134_134499

-- Define the question in Lean given the conditions and prove the answer.
theorem csc_neg_45_eq_neg_sqrt2 : Real.csc (-π/4) = -Real.sqrt 2 :=
by
  -- Sorry placeholder since proof is not required.
  sorry

end csc_neg_45_eq_neg_sqrt2_l134_134499


namespace sum_of_decimals_as_fraction_l134_134360

theorem sum_of_decimals_as_fraction :
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00010 = 2469 / 10000 :=
by
  sorry

end sum_of_decimals_as_fraction_l134_134360


namespace find_a_b_l134_134158

theorem find_a_b
  (f : ℝ → ℝ) (a b : ℝ) (h_a_ne_zero : a ≠ 0) (h_f : ∀ x, f x = x^3 + 3 * x^2 + 1)
  (h_eq : ∀ x, f x - f a = (x - b) * (x - a)^2) :
  a = -2 ∧ b = 1 :=
by
  sorry

end find_a_b_l134_134158


namespace solution_set_of_inequality_l134_134762

theorem solution_set_of_inequality :
  {x : ℝ | (x - 1) / (x^2 - x - 6) ≥ 0} = {x : ℝ | (-2 < x ∧ x ≤ 1) ∨ (3 < x)} := 
sorry

end solution_set_of_inequality_l134_134762


namespace range_of_k_l134_134704

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, f (-x^2 + 3 * x) + f (x - 2 * k) ≤ 0) ↔ k ≥ 2 :=
by
  sorry

end range_of_k_l134_134704


namespace systematic_sampling_eighth_group_l134_134116

theorem systematic_sampling_eighth_group
  (total_employees : ℕ)
  (target_sample : ℕ)
  (third_group_value : ℕ)
  (group_count : ℕ)
  (common_difference : ℕ)
  (eighth_group_value : ℕ) :
  total_employees = 840 →
  target_sample = 42 →
  third_group_value = 44 →
  group_count = total_employees / target_sample →
  common_difference = group_count →
  eighth_group_value = third_group_value + (8 - 3) * common_difference →
  eighth_group_value = 144 :=
sorry

end systematic_sampling_eighth_group_l134_134116


namespace probability_at_least_75_cents_l134_134753

theorem probability_at_least_75_cents (p n d q c50 : Prop) 
  (Hp : p = tt ∨ p = ff)
  (Hn : n = tt ∨ n = ff)
  (Hd : d = tt ∨ d = ff)
  (Hq : q = tt ∨ q = ff)
  (Hc50 : c50 = tt ∨ c50 = ff) :
  (1 / 2 : ℝ) = 
  ((if c50 = tt then (if q = tt then 1 else 0) else 0) + 
  (if c50 = tt then 2^3 else 0)) / 2^5 :=
by sorry

end probability_at_least_75_cents_l134_134753


namespace vanya_scores_not_100_l134_134768

-- Definitions for initial conditions
def score_r (M : ℕ) := M - 14
def score_p (M : ℕ) := M - 9
def score_m (M : ℕ) := M

-- Define the maximum score constraint
def max_score := 100

-- Main statement to be proved
theorem vanya_scores_not_100 (M : ℕ) 
  (hr : score_r M ≤ max_score) 
  (hp : score_p M ≤ max_score) 
  (hm : score_m M ≤ max_score) : 
  ¬(score_r M = max_score ∧ (score_p M = max_score ∨ score_m M = max_score)) ∧
  ¬(score_r M = max_score ∧ score_p M = max_score ∧ score_m M = max_score) :=
sorry

end vanya_scores_not_100_l134_134768


namespace negation_proof_l134_134307

theorem negation_proof :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end negation_proof_l134_134307


namespace prime_quadruples_unique_l134_134657

noncomputable def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → (m = 1 ∨ m = n)

theorem prime_quadruples_unique (p q r n : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) (hn : n > 0)
  (h_eq : p^2 = q^2 + r^n) :
  (p, q, r, n) = (3, 2, 5, 1) ∨ (p, q, r, n) = (5, 3, 2, 4) :=
by
  sorry

end prime_quadruples_unique_l134_134657


namespace sum_absolute_b_eq_fraction_l134_134688

def P (x : ℚ) : ℚ :=
  1 - (2 / 5) * x + (1 / 8) * x^2 + (1 / 10) * x^3

noncomputable def Q (x : ℚ) : ℚ :=
  P(x) * P(x^4) * P(x^6) * P(x^8)

noncomputable def b : List ℚ :=
  (Polynomial.coeff (Q (Polynomial.C 1))).coeffs

noncomputable def abs_sum_b : ℚ :=
  b.sum (fun coeff => abs coeff)

theorem sum_absolute_b_eq_fraction :
  abs_sum_b = ((43 : ℚ) / 40)^4 :=
by
  sorry

end sum_absolute_b_eq_fraction_l134_134688


namespace simplify_expression_l134_134069

theorem simplify_expression :
  let a := (1/2)^2
  let b := (1/2)^3
  let c := (1/2)^4
  let d := (1/2)^5
  1 / (1/a + 1/b + 1/c + 1/d) = 1/60 :=
by
  sorry

end simplify_expression_l134_134069


namespace integral_one_over_x_l134_134137

theorem integral_one_over_x:
  ∫ x in (1 : ℝ)..(Real.exp 1), 1 / x = 1 := 
by 
  sorry

end integral_one_over_x_l134_134137


namespace problem1_l134_134747

theorem problem1 (x : ℝ) (hx : x > 0) : (x + 1/x = 2) ↔ (x = 1) :=
by
  sorry

end problem1_l134_134747


namespace find_M_base7_l134_134559

theorem find_M_base7 :
  ∃ M : ℕ, M = 48 ∧ (M^2).digits 7 = [6, 6] ∧ (∃ (m : ℕ), 49 ≤ m^2 ∧ m^2 < 343 ∧ M = m - 1) :=
sorry

end find_M_base7_l134_134559


namespace total_books_in_school_l134_134882

theorem total_books_in_school (tables_A tables_B tables_C : ℕ)
  (books_per_table_A books_per_table_B books_per_table_C : ℕ → ℕ)
  (hA : tables_A = 750)
  (hB : tables_B = 500)
  (hC : tables_C = 850)
  (h_books_per_table_A : ∀ n, books_per_table_A n = 3 * n / 5)
  (h_books_per_table_B : ∀ n, books_per_table_B n = 2 * n / 5)
  (h_books_per_table_C : ∀ n, books_per_table_C n = n / 3) :
  books_per_table_A tables_A + books_per_table_B tables_B + books_per_table_C tables_C = 933 :=
by sorry

end total_books_in_school_l134_134882


namespace minimum_value_of_expression_l134_134412

noncomputable def minimum_value_expression (x y z : ℝ) : ℝ :=
  1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z))

theorem minimum_value_of_expression : ∀ (x y z : ℝ), -1 < x ∧ x < 0 ∧ -1 < y ∧ y < 0 ∧ -1 < z ∧ z < 0 → 
  minimum_value_expression x y z ≥ 2 := 
by
  intro x y z h
  sorry

end minimum_value_of_expression_l134_134412


namespace AM_GM_HY_order_l134_134046

noncomputable def AM (a b c : ℝ) : ℝ := (a + b + c) / 3
noncomputable def GM (a b c : ℝ) : ℝ := (a * b * c)^(1/3)
noncomputable def HY (a b c : ℝ) : ℝ := 2 * a * b * c / (a * b + b * c + c * a)

theorem AM_GM_HY_order (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  AM a b c > GM a b c ∧ GM a b c > HY a b c := by
  sorry

end AM_GM_HY_order_l134_134046


namespace range_of_a_for_f_ge_a_l134_134051

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem range_of_a_for_f_ge_a :
  (∀ x : ℝ, (-1 ≤ x → f x a ≥ a)) ↔ (-3 ≤ a ∧ a ≤ 1) :=
  sorry

end range_of_a_for_f_ge_a_l134_134051


namespace sum_of_center_coordinates_eq_neg2_l134_134605

theorem sum_of_center_coordinates_eq_neg2 
  (x1 y1 x2 y2 : ℤ)
  (h1 : x1 = 7)
  (h2 : y1 = -8)
  (h3 : x2 = -5)
  (h4 : y2 = 2) 
  : (x1 + x2) / 2 + (y1 + y2) / 2 = -2 :=
by
  -- Insert proof here
  sorry

end sum_of_center_coordinates_eq_neg2_l134_134605


namespace john_uses_six_pounds_of_vegetables_l134_134888

-- Define the given conditions:
def pounds_of_beef_bought : ℕ := 4
def pounds_beef_used_in_soup := pounds_of_beef_bought - 1
def pounds_of_vegetables_used := 2 * pounds_beef_used_in_soup

-- Statement to prove:
theorem john_uses_six_pounds_of_vegetables : pounds_of_vegetables_used = 6 :=
by
  sorry

end john_uses_six_pounds_of_vegetables_l134_134888


namespace tangent_line_equation_l134_134146

noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x - 5

def point_A : ℝ × ℝ := (1, -2)

theorem tangent_line_equation :
  ∀ x y : ℝ, (y = 4 * x - 6) ↔ (fderiv ℝ f (point_A.1) x = 4) ∧ (y = f (point_A.1) + 4 * (x - point_A.1)) := by
  sorry

end tangent_line_equation_l134_134146


namespace cone_volume_and_surface_area_l134_134090

noncomputable def cone_volume (slant_height height : ℝ) : ℝ := 
  1 / 3 * Real.pi * (Real.sqrt (slant_height^2 - height^2))^2 * height

noncomputable def cone_surface_area (slant_height height : ℝ) : ℝ :=
  Real.pi * (Real.sqrt (slant_height^2 - height^2)) * (Real.sqrt (slant_height^2 - height^2) + slant_height)

theorem cone_volume_and_surface_area :
  (cone_volume 15 9 = 432 * Real.pi) ∧ (cone_surface_area 15 9 = 324 * Real.pi) :=
by
  sorry

end cone_volume_and_surface_area_l134_134090


namespace final_position_3000_l134_134949

def initial_position : ℤ × ℤ := (0, 0)
def moves_up_first_minute (pos : ℤ × ℤ) : ℤ × ℤ := (pos.1, pos.2 + 1)

def next_position (n : ℕ) (pos : ℤ × ℤ) : ℤ × ℤ :=
  if n % 4 = 0 then (pos.1 + n, pos.2)
  else if n % 4 = 1 then (pos.1, pos.2 + n)
  else if n % 4 = 2 then (pos.1 - n, pos.2)
  else (pos.1, pos.2 - n)

def final_position (minutes : ℕ) : ℤ × ℤ := sorry

theorem final_position_3000 : final_position 3000 = (0, 27) :=
by {
  -- logic to compute final_position
  sorry -- proof exists here
}

end final_position_3000_l134_134949


namespace middle_number_is_11_l134_134311

theorem middle_number_is_11 (a b c : ℕ) (h1 : a + b = 18) (h2 : a + c = 22) (h3 : b + c = 26) (h4 : c - a = 10) :
  b = 11 :=
by
  sorry

end middle_number_is_11_l134_134311


namespace find_number_l134_134367

theorem find_number : 
  (15^2 * 9^2) / x = 51.193820224719104 → x = 356 :=
by
  sorry

end find_number_l134_134367


namespace find_n_l134_134844

theorem find_n (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) (h3 : ∃ k : ℤ, 721 = n + 360 * k): n = 1 :=
sorry

end find_n_l134_134844


namespace initial_amount_is_800_l134_134794

variables (P R : ℝ)

theorem initial_amount_is_800
  (h1 : 956 = P * (1 + 3 * R / 100))
  (h2 : 1052 = P * (1 + 3 * (R + 4) / 100)) :
  P = 800 :=
sorry

end initial_amount_is_800_l134_134794


namespace cinema_cost_comparison_l134_134093

theorem cinema_cost_comparison (x : ℕ) (hx : x = 1000) :
  let cost_A := if x ≤ 100 then 30 * x else 24 * x + 600
  let cost_B := 27 * x
  cost_A < cost_B :=
by
  sorry

end cinema_cost_comparison_l134_134093


namespace domain_of_g_l134_134826

theorem domain_of_g (t : ℝ) : (t - 1)^2 + (t + 1)^2 + t ≠ 0 :=
  by
  sorry

end domain_of_g_l134_134826


namespace base6_addition_sum_l134_134873

theorem base6_addition_sum 
  (P Q R : ℕ) 
  (h1 : P ≠ Q) 
  (h2 : Q ≠ R) 
  (h3 : P ≠ R) 
  (h4 : P < 6) 
  (h5 : Q < 6) 
  (h6 : R < 6) 
  (h7 : 2*R % 6 = P) 
  (h8 : 2*Q % 6 = R)
  : P + Q + R = 7 := 
  sorry

end base6_addition_sum_l134_134873


namespace intersection_point_polar_coords_l134_134995

open Real

def curve_C1 (x y : ℝ) : Prop :=
  x^2 + y^2 = 2

def curve_C2 (t x y : ℝ) : Prop :=
  (x = 2 - t) ∧ (y = t)

theorem intersection_point_polar_coords :
  ∃ (ρ θ : ℝ), (ρ = sqrt 2) ∧ (θ = π / 4) ∧
  ∃ (x y t : ℝ), curve_C2 t x y ∧ curve_C1 x y ∧
  (ρ = sqrt (x^2 + y^2)) ∧ (tan θ = y / x) :=
by
  sorry

end intersection_point_polar_coords_l134_134995


namespace circle_radius_on_sphere_l134_134418

theorem circle_radius_on_sphere
  (sphere_radius : ℝ)
  (circle1_radius : ℝ)
  (circle2_radius : ℝ)
  (circle3_radius : ℝ)
  (all_circle_touch_each_other : Prop)
  (smaller_circle_touches_all : Prop)
  (smaller_circle_radius : ℝ) :
  sphere_radius = 2 →
  circle1_radius = 1 →
  circle2_radius = 1 →
  circle3_radius = 1 →
  all_circle_touch_each_other →
  smaller_circle_touches_all →
  smaller_circle_radius = 1 - Real.sqrt (2 / 3) :=
by
  intros h_sphere_radius h_circle1_radius h_circle2_radius h_circle3_radius h_all_circle_touch h_smaller_circle_touch
  sorry

end circle_radius_on_sphere_l134_134418


namespace rachel_picked_total_apples_l134_134910

-- Define the conditions
def num_trees : ℕ := 4
def apples_per_tree_picked : ℕ := 7
def apples_remaining : ℕ := 29

-- Define the total apples picked
def total_apples_picked : ℕ := num_trees * apples_per_tree_picked

-- Formal statement of the goal
theorem rachel_picked_total_apples : total_apples_picked = 28 := 
by
  sorry

end rachel_picked_total_apples_l134_134910


namespace hcf_36_84_l134_134209

def highestCommonFactor (a b : ℕ) : ℕ := Nat.gcd a b

theorem hcf_36_84 : highestCommonFactor 36 84 = 12 := by
  sorry

end hcf_36_84_l134_134209


namespace waiter_tip_amount_l134_134343

theorem waiter_tip_amount (n n_no_tip E : ℕ) (h_n : n = 10) (h_no_tip : n_no_tip = 5) (h_E : E = 15) :
  (E / (n - n_no_tip) = 3) :=
by
  -- Proof goes here (we are only writing the statement with sorry)
  sorry

end waiter_tip_amount_l134_134343


namespace solve_inequality_system_l134_134190

theorem solve_inequality_system :
  (∀ x : ℝ, (1 - 3 * (x - 1) < 8 - x) ∧ ((x - 3) / 2 + 2 ≥ x)) →
  ∃ (integers : Set ℤ), integers = {x : ℤ | -2 < (x : ℝ) ∧ (x : ℝ) ≤ 1} ∧ integers = {-1, 0, 1} :=
by
  sorry

end solve_inequality_system_l134_134190


namespace prism_faces_l134_134335

-- Define the conditions of the problem
def prism (E : ℕ) : Prop :=
  ∃ (L : ℕ), 3 * L = E

-- Define the main proof statement
theorem prism_faces (E : ℕ) (hE : prism E) : E = 27 → 2 + E / 3 = 11 :=
by
  sorry -- Proof is not required

end prism_faces_l134_134335


namespace math_problem_equivalent_l134_134824

-- Given that the problem requires four distinct integers a, b, c, d which are less than 12 and invertible modulo 12.
def coprime_with_12 (x : ℕ) : Prop := Nat.gcd x 12 = 1

theorem math_problem_equivalent 
  (a b c d : ℕ) (ha : coprime_with_12 a) (hb : coprime_with_12 b) 
  (hc : coprime_with_12 c) (hd : coprime_with_12 d) 
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c)
  (hbd : b ≠ d) (hcd : c ≠ d) :
  ((a * b * c * d) + (a * b * c) + (a * b * d) + (a * c * d) + (b * c * d)) * Nat.gcd (a * b * c * d) 12 = 1 :=
sorry

end math_problem_equivalent_l134_134824


namespace express_in_scientific_notation_l134_134620

theorem express_in_scientific_notation :
  (10.58 * 10^9) = 1.058 * 10^10 :=
by
  sorry

end express_in_scientific_notation_l134_134620


namespace vampire_needs_7_gallons_per_week_l134_134117

-- Define conditions given in the problem
def pints_per_person : ℕ := 2
def people_per_day : ℕ := 4
def days_per_week : ℕ := 7
def pints_per_gallon : ℕ := 8

-- Prove the vampire needs 7 gallons of blood per week to survive
theorem vampire_needs_7_gallons_per_week :
  (pints_per_person * people_per_day * days_per_week) / pints_per_gallon = 7 := 
by 
  sorry

end vampire_needs_7_gallons_per_week_l134_134117


namespace Li_age_is_12_l134_134322

-- Given conditions:
def Zhang_twice_Li (Li: ℕ) : ℕ := 2 * Li
def Jung_older_Zhang (Zhang: ℕ) : ℕ := Zhang + 2
def Jung_age := 26

-- Proof problem:
theorem Li_age_is_12 : ∃ Li: ℕ, Jung_older_Zhang (Zhang_twice_Li Li) = Jung_age ∧ Li = 12 :=
by
  sorry

end Li_age_is_12_l134_134322


namespace longer_side_is_40_l134_134690

-- Given the conditions
variable (small_rect_width : ℝ) (small_rect_length : ℝ)
variable (num_rects : ℕ)

-- Conditions 
axiom rect_width_is_10 : small_rect_width = 10
axiom length_is_twice_width : small_rect_length = 2 * small_rect_width
axiom four_rectangles : num_rects = 4

-- Prove length of the longer side of the large rectangle
theorem longer_side_is_40 :
  small_rect_width = 10 → small_rect_length = 2 * small_rect_width → num_rects = 4 →
  (2 * small_rect_length) = 40 := sorry

end longer_side_is_40_l134_134690


namespace total_birds_remaining_l134_134922

-- Definitions from conditions
def initial_grey_birds : ℕ := 40
def additional_white_birds : ℕ := 6
def white_birds (grey_birds: ℕ) : ℕ := grey_birds + additional_white_birds
def remaining_grey_birds (grey_birds: ℕ) : ℕ := grey_birds / 2

-- Proof problem
theorem total_birds_remaining : 
  let grey_birds := initial_grey_birds;
  let white_birds_next_to_cage := white_birds(grey_birds);
  let grey_birds_remaining := remaining_grey_birds(grey_birds);
  (grey_birds_remaining + white_birds_next_to_cage) = 66 :=
by {
  sorry
}

end total_birds_remaining_l134_134922


namespace complete_job_days_l134_134108

-- Variables and Conditions
variables (days_5_8 : ℕ) (days_1 : ℕ)

-- Assume that completing 5/8 of the job takes 10 days
def five_eighths_job_days := 10

-- Find days to complete one job at the same pace. 
-- This is the final statement we need to prove
theorem complete_job_days
  (h : 5 * days_1 = 8 * days_5_8) :
  days_1 = 16 := by
  -- Proof is omitted.
  sorry

end complete_job_days_l134_134108


namespace diana_statues_painted_l134_134470

theorem diana_statues_painted :
  let paint_remaining := (1 : ℚ) / 2
  let paint_per_statue := (1 : ℚ) / 4
  (paint_remaining / paint_per_statue) = 2 :=
by
  sorry

end diana_statues_painted_l134_134470


namespace airplane_speed_l134_134668

noncomputable def distance : ℝ := 378.6   -- Distance in km
noncomputable def time : ℝ := 693.5       -- Time in seconds

noncomputable def altitude : ℝ := 10      -- Altitude in km
noncomputable def earth_radius : ℝ := 6370 -- Earth's radius in km

noncomputable def speed : ℝ := distance / time * 3600  -- Speed in km/h
noncomputable def adjusted_speed : ℝ := speed * (earth_radius + altitude) / earth_radius

noncomputable def min_distance : ℝ := 378.6 - 0.03     -- Minimum possible distance in km
noncomputable def max_distance : ℝ := 378.6 + 0.03     -- Maximum possible distance in km
noncomputable def min_time : ℝ := 693.5 - 1.5          -- Minimum possible time in s
noncomputable def max_time : ℝ := 693.5 + 1.5          -- Maximum possible time in s

noncomputable def max_speed : ℝ := max_distance / min_time * 3600 -- Max speed with uncertainty
noncomputable def min_speed : ℝ := min_distance / max_time * 3600 -- Min speed with uncertainty

theorem airplane_speed :
  1960 < adjusted_speed ∧ adjusted_speed < 1970 :=
by
  sorry

end airplane_speed_l134_134668


namespace proof_f_f_2008_eq_2008_l134_134234

-- Define the function f
axiom f : ℝ → ℝ

-- The conditions given in the problem
axiom odd_f : ∀ x, f (-x) = -f x
axiom periodic_f : ∀ x, f (x + 6) = f x
axiom f_at_4 : f 4 = -2008

-- The goal to prove
theorem proof_f_f_2008_eq_2008 : f (f 2008) = 2008 :=
by
  sorry

end proof_f_f_2008_eq_2008_l134_134234


namespace prob_primes_1_to_30_l134_134625

-- Define the set of integers from 1 to 30
def set_1_to_30 : Set ℕ := { n | 1 ≤ n ∧ n ≤ 30 }

-- Define the set of prime numbers from 1 to 30
def primes_1_to_30 : Set ℕ := { n | n ∈ set_1_to_30 ∧ Nat.Prime n }

-- Define the number of ways to choose 2 items from a set of size n
def choose (n k : ℕ) := n * (n - 1) / 2

-- The goal is to prove that the probability of choosing 2 prime numbers from the set is 1/10
theorem prob_primes_1_to_30 : (choose (Set.card primes_1_to_30) 2) / (choose (Set.card set_1_to_30) 2) = 1 / 10 :=
by
  sorry

end prob_primes_1_to_30_l134_134625


namespace find_m_l134_134155

variables (x m : ℝ)

def equation (x m : ℝ) : Prop := 3 * x - 2 * m = 4

theorem find_m (h1 : equation 6 m) : m = 7 :=
by
  sorry

end find_m_l134_134155


namespace find_m_direct_proportion_l134_134037

theorem find_m_direct_proportion (m : ℝ) (h1 : m + 2 ≠ 0) (h2 : |m| - 1 = 1) : m = 2 :=
sorry

end find_m_direct_proportion_l134_134037


namespace a_100_positive_a_100_abs_lt_018_l134_134026

-- Define the sequence based on the given conditions
def a_n (n : ℕ) : ℝ := real.cos (real.pi / 180 * (10^n) : ℝ)

-- Translate mathematical claims to Lean theorems
theorem a_100_positive : a_n 100 > 0 :=
sorry

theorem a_100_abs_lt_018 : |a_n 100| < 0.18 :=
sorry

end a_100_positive_a_100_abs_lt_018_l134_134026


namespace solve_for_x_l134_134589

theorem solve_for_x (x : ℕ) (h : 5 * (2 ^ x) = 320) : x = 6 :=
by
  sorry

end solve_for_x_l134_134589


namespace find_side_length_of_largest_square_l134_134944

theorem find_side_length_of_largest_square (A : ℝ) (hA : A = 810) :
  ∃ a : ℝ, (5 / 8) * a ^ 2 = A ∧ a = 36 := by
  sorry

end find_side_length_of_largest_square_l134_134944


namespace power_function_constant_l134_134020

theorem power_function_constant (k α : ℝ)
  (h : (1 / 2 : ℝ) ^ α * k = (Real.sqrt 2 / 2)) : k + α = 3 / 2 := by
  sorry

end power_function_constant_l134_134020


namespace num_emails_received_after_second_deletion_l134_134730

-- Define the initial conditions and final question
variable (initialEmails : ℕ)    -- Initial number of emails
variable (deletedEmails1 : ℕ)   -- First batch of deleted emails
variable (receivedEmails1 : ℕ)  -- First batch of received emails
variable (deletedEmails2 : ℕ)   -- Second batch of deleted emails
variable (receivedEmails2 : ℕ)  -- Second batch of received emails
variable (receivedEmails3 : ℕ)  -- Third batch of received emails
variable (finalEmails : ℕ)      -- Final number of emails in the inbox

-- Conditions based on the problem description
axiom initialEmails_def : initialEmails = 0
axiom deletedEmails1_def : deletedEmails1 = 50
axiom receivedEmails1_def : receivedEmails1 = 15
axiom deletedEmails2_def : deletedEmails2 = 20
axiom receivedEmails3_def : receivedEmails3 = 10
axiom finalEmails_def : finalEmails = 30

-- Question: Prove that the number of emails received after the second deletion is 5
theorem num_emails_received_after_second_deletion : receivedEmails2 = 5 :=
by
  sorry

end num_emails_received_after_second_deletion_l134_134730


namespace function_no_real_zeros_l134_134712

variable (a b c : ℝ)

-- Conditions: a, b, c form a geometric sequence and ac > 0
def geometric_sequence (a b c : ℝ) : Prop := b^2 = a * c
def positive_product (a c : ℝ) : Prop := a * c > 0

theorem function_no_real_zeros (h_geom : geometric_sequence a b c) (h_pos : positive_product a c) :
  ∀ x : ℝ, a * x^2 + b * x + c ≠ 0 := 
by
  sorry

end function_no_real_zeros_l134_134712


namespace jame_weeks_tearing_cards_l134_134731

def cards_tears_per_time : ℕ := 30
def cards_per_deck : ℕ := 55
def tears_per_week : ℕ := 3
def decks_bought : ℕ := 18

theorem jame_weeks_tearing_cards :
  (cards_tears_per_time * tears_per_week * decks_bought * cards_per_deck) / (cards_tears_per_time * tears_per_week) = 11 := by
  sorry

end jame_weeks_tearing_cards_l134_134731


namespace kayla_score_fourth_level_l134_134736

theorem kayla_score_fourth_level 
  (score1 score2 score3 score5 score6 : ℕ) 
  (h1 : score1 = 2) 
  (h2 : score2 = 3) 
  (h3 : score3 = 5) 
  (h5 : score5 = 12) 
  (h6 : score6 = 17)
  (h_diff : ∀ n : ℕ, score2 - score1 + n = score3 - score2 + n + 1 ∧ score3 - score2 + n + 2 = score5 - score3 + n + 3 ∧ score5 - score3 + n + 4 = score6 - score5 + n + 5) :
  ∃ score4 : ℕ, score4 = 8 :=
by
  sorry

end kayla_score_fourth_level_l134_134736


namespace Jenny_recycling_l134_134180

theorem Jenny_recycling:
  let bottle_weight := 6
  let can_weight := 2
  let glass_jar_weight := 8
  let max_weight := 100
  let num_cans := 20
  let bottle_value := 10
  let can_value := 3
  let glass_jar_value := 12
  let total_money := (num_cans * can_value) + (7 * glass_jar_value) + (0 * bottle_value)
  total_money = 144 ∧ num_cans = 20 ∧ glass_jars = 7 ∧ bottles = 0 := by sorry

end Jenny_recycling_l134_134180


namespace f_diff_l134_134856

def f (n : ℕ) : ℚ :=
  (Finset.range (3 * n)).sum (λ k => (1 : ℚ) / (k + 1))

theorem f_diff (n : ℕ) : f (n + 1) - f n = (1 / (3 * n) + 1 / (3 * n + 1) + 1 / (3 * n + 2)) :=
by
  sorry

end f_diff_l134_134856


namespace positive_integers_satisfying_inequality_l134_134135

-- Define the assertion that there are exactly 5 positive integers x satisfying the given inequality
theorem positive_integers_satisfying_inequality :
  (∃! x : ℕ, 4 < x ∧ x < 10 ∧ (10 * x)^4 > x^8 ∧ x^8 > 2^16) :=
sorry

end positive_integers_satisfying_inequality_l134_134135


namespace correct_statements_l134_134780

-- Define the universal set U as ℤ (integers)
noncomputable def U : Set ℤ := Set.univ

-- Conditions
def is_subset_of_int : Prop := {0} ⊆ (Set.univ : Set ℤ)

def counterexample_subsets (A B : Set ℤ) : Prop :=
  (A = {1, 2} ∧ B = {1, 2, 3}) ∧ (B ∩ (U \ A) ≠ ∅)

def negation_correct_1 : Prop :=
  ¬(∀ x : ℤ, x^2 > 0) ↔ ∃ x : ℤ, x^2 ≤ 0

def negation_correct_2 : Prop :=
  ¬(∀ x : ℤ, x^2 > 0) ↔ ¬(∀ x : ℤ, x^2 < 0)

-- The theorem to prove the equivalence of correct statements
theorem correct_statements :
  (is_subset_of_int ∧
   ∀ A B : Set ℤ, A ⊆ U → B ⊆ U → (A ⊆ B → counterexample_subsets A B) ∧
   negation_correct_1 ∧
   ¬negation_correct_2) ↔
  (true) :=
by 
  sorry

end correct_statements_l134_134780


namespace range_of_distance_l134_134986

noncomputable def A (α : ℝ) : ℝ × ℝ × ℝ := (3 * Real.cos α, 3 * Real.sin α, 1)
noncomputable def B (β : ℝ) : ℝ × ℝ × ℝ := (2 * Real.cos β, 2 * Real.sin β, 1)

theorem range_of_distance (α β : ℝ) :
  1 ≤ Real.sqrt ((3 * Real.cos α - 2 * Real.cos β)^2 + (3 * Real.sin α - 2 * Real.sin β)^2) ∧
  Real.sqrt ((3 * Real.cos α - 2 * Real.cos β)^2 + (3 * Real.sin α - 2 * Real.sin β)^2) ≤ 5 :=
by
  sorry

end range_of_distance_l134_134986


namespace duration_of_resulting_video_l134_134675

theorem duration_of_resulting_video 
    (vasya_walk_time : ℕ) (petya_walk_time : ℕ) 
    (sync_meet_point : ℕ) :
    vasya_walk_time = 8 → petya_walk_time = 5 → sync_meet_point = sync_meet_point → 
    (vasya_walk_time - sync_meet_point + petya_walk_time) = 5 :=
by
  intros
  sorry

end duration_of_resulting_video_l134_134675


namespace family_ate_doughnuts_l134_134110

variable (box_initial : ℕ) (box_left : ℕ) (dozen : ℕ)

-- Define the initial and remaining conditions
def dozen_value : ℕ := 12
def box_initial_value : ℕ := 2 * dozen_value
def doughnuts_left_value : ℕ := 16

theorem family_ate_doughnuts (h1 : box_initial = box_initial_value) (h2 : box_left = doughnuts_left_value) :
  box_initial - box_left = 8 := by
  -- h1 says the box initially contains 2 dozen, which is 24.
  -- h2 says that there are 16 doughnuts left.
  sorry

end family_ate_doughnuts_l134_134110


namespace friends_boat_crossing_impossible_l134_134685

theorem friends_boat_crossing_impossible : 
  ∀ (friends : Finset ℕ) (boat_capacity : ℕ), friends.card = 5 → boat_capacity ≥ 5 → 
  ¬ (∀ group : Finset ℕ, group ⊆ friends → group ≠ ∅ → group.card ≤ boat_capacity → 
  ∃ crossing : ℕ, (crossing = group.card ∧ group ⊆ friends)) :=
by
  intro friends boat_capacity friends_card boat_capacity_cond goal
  sorry

end friends_boat_crossing_impossible_l134_134685


namespace no_int_solutions_l134_134410

open Nat

theorem no_int_solutions (p1 p2 α n : ℕ)
  (hp1_prime : p1.Prime)
  (hp2_prime : p2.Prime)
  (hp1_odd : p1 % 2 = 1)
  (hp2_odd : p2 % 2 = 1)
  (hα_pos : 0 < α)
  (hn_pos : 0 < n)
  (hα_gt1 : 1 < α)
  (hn_gt1 : 1 < n) :
  ¬(let lhs := ((p2 - 1) / 2) ^ p1 + ((p2 + 1) / 2) ^ p1
    lhs = α ^ n) :=
sorry

end no_int_solutions_l134_134410


namespace pet_center_final_count_l134_134092

def initial_dogs : Nat := 36
def initial_cats : Nat := 29
def adopted_dogs : Nat := 20
def collected_cats : Nat := 12
def final_pets : Nat := 57

theorem pet_center_final_count :
  (initial_dogs - adopted_dogs) + (initial_cats + collected_cats) = final_pets := 
by
  sorry

end pet_center_final_count_l134_134092


namespace sales_this_month_l134_134904

-- Define the given conditions
def price_large := 60
def price_small := 30
def num_large_last_month := 8
def num_small_last_month := 4

-- Define the computation of total sales for last month
def sales_last_month : ℕ :=
  price_large * num_large_last_month + price_small * num_small_last_month

-- State the theorem to prove the sales this month
theorem sales_this_month : sales_last_month * 2 = 1200 :=
by
  -- Proof will follow, for now we use sorry as a placeholder
  sorry

end sales_this_month_l134_134904


namespace find_x_l134_134139

theorem find_x (x : ℝ) (h : x + 2.75 + 0.158 = 2.911) : x = 0.003 :=
sorry

end find_x_l134_134139


namespace years_passed_l134_134109

def initial_ages : List ℕ := [19, 34, 37, 42, 48]

def new_ages (x : ℕ) : List ℕ :=
  initial_ages.map (λ age => age + x)

-- Hypothesis: The new ages fit the following stem-and-leaf plot structure
def valid_stem_and_leaf (ages : List ℕ) : Bool :=
  ages = [25, 31, 34, 37, 43, 48]

theorem years_passed : ∃ x : ℕ, valid_stem_and_leaf (new_ages x) := by
  sorry

end years_passed_l134_134109


namespace perimeter_of_square_field_l134_134766

-- Given conditions
def num_posts : ℕ := 36
def post_width_inch : ℝ := 6
def gap_length_feet : ℝ := 8

-- Derived conditions
def posts_per_side : ℕ := num_posts / 4
def gaps_per_side : ℕ := posts_per_side - 1
def total_gap_length_per_side : ℝ := gaps_per_side * gap_length_feet
def post_width_feet : ℝ := post_width_inch / 12
def total_post_width_per_side : ℝ := posts_per_side * post_width_feet
def side_length : ℝ := total_gap_length_per_side + total_post_width_per_side

-- Goal: The perimeter of the square field
theorem perimeter_of_square_field : 4 * side_length = 242 := by
  sorry

end perimeter_of_square_field_l134_134766


namespace sum_of_disk_areas_l134_134425

-- Definitions corresponding to the problem conditions
def radius_of_large_circle : ℝ := 2
def number_of_disks : ℕ := 16

-- The lean statement for the problem
theorem sum_of_disk_areas (r : ℝ)
  (h1 : ∀ i j : Fin number_of_disks, i ≠ j → ¬∃ x : Fin number_of_disks, (x = i ∧ x = j))
  (h2 : ∀ i : Fin number_of_disks, ∃! p : ℝ × ℝ, (p.1^2 + p.2^2 = radius_of_large_circle^2))
  (h3 : ∀ i, ∃! p : ℝ × ℝ, (p.1^2 + p.2^2 = r^2)) :
  (16 * (Real.pi * r^2) = Real.pi * (112 - 64 * Real.sqrt 3))
:= by
  sorry

end sum_of_disk_areas_l134_134425


namespace circle_radius_tangent_to_circumcircles_l134_134284

noncomputable def circumradius (a b c : ℝ) : ℝ :=
  (a * b * c) / (4 * (Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))))

theorem circle_radius_tangent_to_circumcircles (AB BC CA : ℝ) (H : Point) 
  (h_AB : AB = 13) (h_BC : BC = 14) (h_CA : CA = 15) : 
  (radius : ℝ) = 65 / 16 :=
by
  sorry

end circle_radius_tangent_to_circumcircles_l134_134284


namespace chess_piece_problem_l134_134458

theorem chess_piece_problem
  (a b c : ℕ)
  (h1 : b = b * 2 - a)
  (h2 : c = c * 2)
  (h3 : a = a * 2 - b)
  (h4 : c = c * 2 - a + b)
  (h5 : a * 2 = 16)
  (h6 : b * 2 = 16)
  (h7 : c * 2 = 16) : 
  a = 26 ∧ b = 14 ∧ c = 8 := 
sorry

end chess_piece_problem_l134_134458


namespace jan_total_skips_l134_134403

def jan_initial_speed : ℕ := 70
def jan_training_factor : ℕ := 2
def jan_skipping_time : ℕ := 5

theorem jan_total_skips :
  (jan_initial_speed * jan_training_factor) * jan_skipping_time = 700 := by
  sorry

end jan_total_skips_l134_134403


namespace range_of_m_l134_134531

open Real

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 4 * cos x + sin x ^ 2 + m - 4 = 0) ↔ 0 ≤ m ∧ m ≤ 8 :=
sorry

end range_of_m_l134_134531


namespace probability_of_two_prime_numbers_l134_134637

open Finset

noncomputable def primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_of_two_prime_numbers :
  (primes.card.choose 2 / ((range 1 31).card.choose 2) : ℚ) = 1 / 9.67 := sorry

end probability_of_two_prime_numbers_l134_134637


namespace binary_to_octal_of_101101110_l134_134961

def binaryToDecimal (n : Nat) : Nat :=
  List.foldl (fun acc b => acc * 2 + b) 0 (Nat.digits 2 n)

def decimalToOctal (n : Nat) : Nat :=
  List.foldl (fun acc b => acc * 10 + b) 0 (Nat.digits 8 n)

theorem binary_to_octal_of_101101110 :
  decimalToOctal (binaryToDecimal 0b101101110) = 556 :=
by sorry

end binary_to_octal_of_101101110_l134_134961


namespace divisor_is_twelve_l134_134939

theorem divisor_is_twelve (d : ℕ) (h : 64 = 5 * d + 4) : d = 12 := 
sorry

end divisor_is_twelve_l134_134939


namespace range_of_m_l134_134167

theorem range_of_m (x m : ℝ) (h1 : -1 ≤ x ∧ x ≤ 1) (h2 : |x - m| ≤ 2) : -1 ≤ m ∧ m ≤ 1 :=
sorry

end range_of_m_l134_134167


namespace plants_producing_flowers_l134_134163

noncomputable def germinate_percent_daisy : ℝ := 0.60
noncomputable def germinate_percent_sunflower : ℝ := 0.80
noncomputable def produce_flowers_percent : ℝ := 0.80
noncomputable def daisy_seeds_planted : ℕ := 25
noncomputable def sunflower_seeds_planted : ℕ := 25

theorem plants_producing_flowers : 
  let daisy_plants_germinated := germinate_percent_daisy * daisy_seeds_planted,
      sunflower_plants_germinated := germinate_percent_sunflower * sunflower_seeds_planted,
      total_plants_germinated := daisy_plants_germinated + sunflower_plants_germinated,
      plants_that_produce_flowers := produce_flowers_percent * total_plants_germinated
  in plants_that_produce_flowers = 28 :=
by
  sorry

end plants_producing_flowers_l134_134163


namespace value_of_x_l134_134168

theorem value_of_x (x : ℝ) : 3 - 5 + 7 = 6 - x → x = 1 :=
by
  intro h
  sorry

end value_of_x_l134_134168


namespace jonah_profit_l134_134850

def cost_per_pineapple (quantity : ℕ) : ℝ :=
  if quantity > 50 then 1.60 else if quantity > 40 then 1.80 else 2.00

def total_cost (quantity : ℕ) : ℝ :=
  cost_per_pineapple quantity * quantity

def bundle_revenue (bundles : ℕ) : ℝ :=
  bundles * 20

def single_ring_revenue (rings : ℕ) : ℝ :=
  rings * 4

def total_revenue (bundles : ℕ) (rings : ℕ) : ℝ :=
  bundle_revenue bundles + single_ring_revenue rings

noncomputable def profit (quantity bundles rings : ℕ) : ℝ :=
  total_revenue bundles rings - total_cost quantity

theorem jonah_profit : profit 60 35 150 = 1204 := by
  sorry

end jonah_profit_l134_134850


namespace watermelon_seeds_l134_134203

theorem watermelon_seeds (n_slices : ℕ) (total_seeds : ℕ) (B W : ℕ) 
  (h1: n_slices = 40) 
  (h2: B = W) 
  (h3 : n_slices * B + n_slices * W = total_seeds)
  (h4 : total_seeds = 1600) : B = 20 :=
by {
  sorry
}

end watermelon_seeds_l134_134203


namespace min_value_of_2gx_sq_minus_fx_l134_134979

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
noncomputable def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c

theorem min_value_of_2gx_sq_minus_fx (a b c : ℝ) (h_a_nonzero : a ≠ 0)
  (h_min_fx : ∃ x : ℝ, 2 * (f a b x)^2 - g a c x = 7 / 2) :
  ∃ x : ℝ, 2 * (g a c x)^2 - f a b x = -15 / 4 :=
sorry

end min_value_of_2gx_sq_minus_fx_l134_134979


namespace pyramid_height_l134_134338

noncomputable def height_pyramid (perimeter_base : ℝ) (distance_apex_vertex : ℝ) : ℝ :=
  let side_length := perimeter_base / 4
  let half_diagonal := (side_length * Real.sqrt 2) / 2
  Real.sqrt (distance_apex_vertex ^ 2 - half_diagonal ^ 2)

theorem pyramid_height
  (perimeter_base: ℝ)
  (h_perimeter : perimeter_base = 32)
  (distance_apex_vertex: ℝ)
  (h_distance : distance_apex_vertex = 10) :
  height_pyramid perimeter_base distance_apex_vertex = 2 * Real.sqrt 17 :=
by
  sorry

end pyramid_height_l134_134338


namespace age_of_cat_l134_134202

variables (cat_age rabbit_age dog_age : ℕ)

-- Conditions
def condition1 : Prop := rabbit_age = cat_age / 2
def condition2 : Prop := dog_age = 3 * rabbit_age
def condition3 : Prop := dog_age = 12

-- Question
def question (cat_age : ℕ) : Prop := cat_age = 8

theorem age_of_cat (h1 : condition1 cat_age rabbit_age) (h2 : condition2 rabbit_age dog_age) (h3 : condition3 dog_age) : question cat_age :=
by
  sorry

end age_of_cat_l134_134202


namespace problem1_solution_problem2_solution_l134_134071

-- Problem 1: Prove that x = 1 given 6x - 7 = 4x - 5
theorem problem1_solution (x : ℝ) (h : 6 * x - 7 = 4 * x - 5) : x = 1 := by
  sorry


-- Problem 2: Prove that x = -1 given (3x - 1) / 4 - 1 = (5x - 7) / 6
theorem problem2_solution (x : ℝ) (h : (3 * x - 1) / 4 - 1 = (5 * x - 7) / 6) : x = -1 := by
  sorry

end problem1_solution_problem2_solution_l134_134071


namespace quadrilateral_area_ratio_l134_134421

noncomputable def area_of_octagon (a : ℝ) : ℝ := 2 * a^2 * (1 + Real.sqrt 2)

noncomputable def area_of_square (s : ℝ) : ℝ := s^2

theorem quadrilateral_area_ratio (a : ℝ) (s : ℝ)
    (h1 : s = a * Real.sqrt (2 + Real.sqrt 2))
    : (area_of_square s) / (area_of_octagon a) = Real.sqrt 2 / 2 :=
by
  sorry

end quadrilateral_area_ratio_l134_134421


namespace rectangle_length_width_l134_134804

theorem rectangle_length_width (x y : ℝ) (h1 : 2 * (x + y) = 26) (h2 : x * y = 42) : 
  (x = 7 ∧ y = 6) ∨ (x = 6 ∧ y = 7) :=
by
  sorry

end rectangle_length_width_l134_134804


namespace total_notebooks_l134_134764

theorem total_notebooks (num_boxes : ℕ) (parts_per_box : ℕ) (notebooks_per_part : ℕ) (h1 : num_boxes = 22)
  (h2 : parts_per_box = 6) (h3 : notebooks_per_part = 5) : 
  num_boxes * parts_per_box * notebooks_per_part = 660 := 
by
  sorry

end total_notebooks_l134_134764


namespace text_messages_December_l134_134283

-- Definitions of the number of text messages sent each month
def text_messages_November := 1
def text_messages_January := 4
def text_messages_February := 8
def doubling_pattern (a b : ℕ) : Prop := b = 2 * a

-- Prove that Jared sent 2 text messages in December
theorem text_messages_December : ∃ x : ℕ, 
  doubling_pattern text_messages_November x ∧ 
  doubling_pattern x text_messages_January ∧ 
  doubling_pattern text_messages_January text_messages_February ∧ 
  x = 2 :=
by
  sorry

end text_messages_December_l134_134283


namespace number_of_clients_l134_134653

-- Definitions from the problem
def cars : ℕ := 18
def selections_per_client : ℕ := 3
def selections_per_car : ℕ := 3

-- Theorem statement: Prove that the number of clients is 18
theorem number_of_clients (total_cars : ℕ) (cars_selected_by_each_client : ℕ) (each_car_selected : ℕ)
  (h_cars : total_cars = cars)
  (h_select_each : cars_selected_by_each_client = selections_per_client)
  (h_selected_car : each_car_selected = selections_per_car) :
  (total_cars * each_car_selected) / cars_selected_by_each_client = 18 :=
by
  rw [h_cars, h_select_each, h_selected_car]
  sorry

end number_of_clients_l134_134653


namespace polynomial_value_at_one_l134_134714

theorem polynomial_value_at_one
  (a b c : ℝ)
  (h1 : -a - b - c + 1 = 6)
  : a + b + c + 1 = -4 :=
by {
  sorry
}

end polynomial_value_at_one_l134_134714


namespace part1_part2_part3_l134_134520

theorem part1 (k : ℝ) (h₀ : k ≠ 0) (h : ∀ x : ℝ, k * x^2 - 2 * x + 6 * k < 0 ↔ x < -3 ∨ x > -2) : k = -2/5 :=
sorry

theorem part2 (k : ℝ) (h₀ : k ≠ 0) (h : ∀ x : ℝ, k * x^2 - 2 * x + 6 * k < 0) : k < -Real.sqrt 6 / 6 :=
sorry

theorem part3 (k : ℝ) (h₀ : k ≠ 0) (h : ∀ x : ℝ, ¬ (k * x^2 - 2 * x + 6 * k < 0)) : k ≥ Real.sqrt 6 / 6 :=
sorry

end part1_part2_part3_l134_134520


namespace inequality_holds_for_all_x_l134_134237

theorem inequality_holds_for_all_x (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x ^ 2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Icc (-2 : ℝ) 2 :=
sorry

end inequality_holds_for_all_x_l134_134237


namespace rhombus_diagonal_l134_134641

theorem rhombus_diagonal (a b : ℝ) (area_triangle : ℝ) (d1 d2 : ℝ)
  (h1 : 2 * area_triangle = a * b)
  (h2 : area_triangle = 75)
  (h3 : a = 20) :
  b = 15 :=
by
  sorry

end rhombus_diagonal_l134_134641


namespace goldfish_cost_discrete_points_l134_134267

def goldfish_cost (n : ℕ) : ℝ :=
  0.25 * n + 5

theorem goldfish_cost_discrete_points :
  ∀ n : ℕ, 5 ≤ n ∧ n ≤ 20 → ∃ k : ℕ, goldfish_cost n = goldfish_cost k ∧ 5 ≤ k ∧ k ≤ 20 :=
by sorry

end goldfish_cost_discrete_points_l134_134267


namespace dave_initial_boxes_l134_134962

def pieces_per_box : ℕ := 3
def boxes_given_away : ℕ := 5
def pieces_left : ℕ := 21
def total_pieces_given_away := boxes_given_away * pieces_per_box
def total_pieces_initially := total_pieces_given_away + pieces_left

theorem dave_initial_boxes : total_pieces_initially / pieces_per_box = 12 := by
  sorry

end dave_initial_boxes_l134_134962


namespace final_price_correct_l134_134352

-- Definitions that follow the given conditions
def initial_price : ℝ := 150
def increase_percentage_year1 : ℝ := 1.5
def decrease_percentage_year2 : ℝ := 0.3

-- Compute intermediate values
noncomputable def price_end_year1 : ℝ := initial_price + (increase_percentage_year1 * initial_price)
noncomputable def price_end_year2 : ℝ := price_end_year1 - (decrease_percentage_year2 * price_end_year1)

-- The final theorem stating the price at the end of the second year
theorem final_price_correct : price_end_year2 = 262.5 := by
  sorry

end final_price_correct_l134_134352


namespace problem_1_problem_2_l134_134972

variable (a : ℕ → ℝ)

variables (h1 : ∀ n, 0 < a n) (h2 : ∀ n, a (n + 1) + 1 / a n < 2)

-- Prove that: (1) a_{n+2} < a_{n+1} < 2 for n ∈ ℕ*
theorem problem_1 (n : ℕ) : a (n + 2) < a (n + 1) ∧ a (n + 1) < 2 := 
sorry

-- Prove that: (2) a_n > 1 for n ∈ ℕ*
theorem problem_2 (n : ℕ) : 1 < a n := 
sorry

end problem_1_problem_2_l134_134972


namespace number_of_teams_l134_134457

theorem number_of_teams (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
by
  sorry

end number_of_teams_l134_134457


namespace factor_expression_l134_134138

theorem factor_expression (x : ℝ) : 
  4 * x * (x - 5) + 6 * (x - 5) = (4 * x + 6) * (x - 5) :=
by 
  sorry

end factor_expression_l134_134138


namespace min_negative_numbers_l134_134570

theorem min_negative_numbers (a b c d : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h5 : a + b + c < d) (h6 : a + b + d < c) (h7 : a + c + d < b) (h8 : b + c + d < a) :
  3 ≤ (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) + (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0) := 
sorry

end min_negative_numbers_l134_134570


namespace discarded_number_l134_134756

theorem discarded_number (S S_48 : ℝ) (h1 : S = 1000) (h2 : S_48 = 900) (h3 : ∃ x : ℝ, S - S_48 = 45 + x): 
  ∃ x : ℝ, x = 55 :=
by {
  -- Using the conditions provided to derive the theorem.
  sorry 
}

end discarded_number_l134_134756


namespace trig_expression_identity_l134_134254

theorem trig_expression_identity (a : ℝ) (h : 2 * Real.sin a = 3 * Real.cos a) : 
  (4 * Real.sin a + Real.cos a) / (5 * Real.sin a - 2 * Real.cos a) = 14 / 11 :=
by
  sorry

end trig_expression_identity_l134_134254


namespace expression_not_defined_at_12_l134_134852

theorem expression_not_defined_at_12 : 
  ¬ ∃ x, x^2 - 24 * x + 144 = 0 ∧ (3 * x^3 + 5) / (x^2 - 24 * x + 144) = 0 :=
by
  intro h
  cases h with x hx
  have hx2 : x^2 - 24 * x + 144 = 0 := hx.1
  have denom_zero : x^2 - 24 * x + 144 = 0 := by sorry
  subst denom_zero
  sorry

end expression_not_defined_at_12_l134_134852


namespace arcsin_eq_solution_domain_l134_134591

open Real

theorem arcsin_eq_solution_domain (x : ℝ) (hx1 : abs (x * sqrt 5 / 3) ≤ 1)
  (hx2 : abs (x * sqrt 5 / 6) ≤ 1)
  (hx3 : abs (7 * x * sqrt 5 / 18) ≤ 1) :
  arcsin (x * sqrt 5 / 3) + arcsin (x * sqrt 5 / 6) = arcsin (7 * x * sqrt 5 / 18) ↔ 
  x = 0 ∨ x = 8 / 7 ∨ x = -8 / 7 := sorry

end arcsin_eq_solution_domain_l134_134591


namespace part1_union_part1_intersection_complement_part2_necessary_sufficient_condition_l134_134561

-- Definitions of the sets and conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -4 < x ∧ x < 1}
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 2}

-- Part 1
theorem part1_union (a : ℝ) (ha : a = 1) : 
  A ∪ B a = { x | -4 < x ∧ x ≤ 3 } :=
sorry

theorem part1_intersection_complement (a : ℝ) (ha : a = 1) : 
  A ∩ (U \ B a) = { x | -4 < x ∧ x < 0 } :=
sorry

-- Part 2
theorem part2_necessary_sufficient_condition (a : ℝ) : 
  (∀ x, x ∈ B a ↔ x ∈ A) ↔ (-3 < a ∧ a < -1) :=
sorry

end part1_union_part1_intersection_complement_part2_necessary_sufficient_condition_l134_134561


namespace probability_xi_12_l134_134176

noncomputable def P_xi_eq_12 : ℝ :=
  let p_red := 3 / 8
  let p_white := 5 / 8
  let n := 12
  let k := 10
  let x := 11
  let y := 9
  nat.choose x y * (p_red ^ y) * (p_white ^ (x - y)) * p_red

theorem probability_xi_12 :
  let p_red := 3 / 8
  let p_white := 5 / 8
  let n := 12
  let k := 10
  let x := 11
  let y := 9
  let P_xi := nat.choose x y * (p_red ^ y) * (p_white ^ (x - y)) * p_red
  P_xi = C_{11}^{9} \cdot \left(\dfrac{3}{8}\right)^{9} \cdot \left(\dfrac{5}{8}\right)^{2} \cdot \dfrac{3}{8} := by
  sorry

end probability_xi_12_l134_134176


namespace g_neg6_eq_neg1_l134_134738

def f : ℝ → ℝ := fun x => 4 * x - 6
def g : ℝ → ℝ := fun x => 2 * x^2 + 7 * x - 1

theorem g_neg6_eq_neg1 : g (-6) = -1 := by
  sorry

end g_neg6_eq_neg1_l134_134738


namespace triangle_BD_length_l134_134279

theorem triangle_BD_length 
  (A B C D : Type) 
  (hAC : AC = 8) 
  (hBC : BC = 8) 
  (hAD : AD = 6) 
  (hCD : CD = 5) : BD = 6 :=
  sorry

end triangle_BD_length_l134_134279


namespace tony_average_time_to_store_l134_134571

-- Definitions and conditions
def speed_walking := 2  -- MPH
def speed_running := 10  -- MPH
def distance_to_store := 4  -- miles
def days_walking := 1  -- Sunday
def days_running := 2  -- Tuesday, Thursday
def total_days := days_walking + days_running

-- Proof statement
theorem tony_average_time_to_store :
  let time_walking := (distance_to_store / speed_walking) * 60
  let time_running := (distance_to_store / speed_running) * 60
  let total_time := time_walking * days_walking + time_running * days_running
  let average_time := total_time / total_days
  average_time = 56 :=
by
  sorry  -- Proof to be completed

end tony_average_time_to_store_l134_134571


namespace calculate_expression_l134_134129

theorem calculate_expression :
  4 + ((-2)^2) * 2 + (-36) / 4 = 3 := by
  sorry

end calculate_expression_l134_134129


namespace arc_length_condition_l134_134678

open Real

noncomputable def hyperbola_eq (a b x y: ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

theorem arc_length_condition (a b r: ℝ) (h1: hyperbola_eq a b 2 1) (h2: r > 0)
  (h3: ∃ x y, x^2 + y^2 = r^2 ∧ hyperbola_eq a b x y) :
  r > 2 * sqrt 2 :=
sorry

end arc_length_condition_l134_134678


namespace unoccupied_seats_l134_134835

theorem unoccupied_seats 
    (seats_per_row : ℕ) 
    (rows : ℕ) 
    (seatable_fraction : ℚ) 
    (total_seats := seats_per_row * rows) 
    (seatable_seats_per_row := (seatable_fraction * seats_per_row)) 
    (seatable_seats := seatable_seats_per_row * rows) 
    (unoccupied_seats := total_seats - seatable_seats) {
  seats_per_row = 8, 
  rows = 12, 
  seatable_fraction = 3/4 
  : unoccupied_seats = 24 :=
by
  sorry

end unoccupied_seats_l134_134835


namespace mixture_ratio_l134_134095

variables (p q V W : ℝ)

-- Condition summaries:
-- - First jar has volume V, ratio of alcohol to water is p:1.
-- - Second jar has volume W, ratio of alcohol to water is q:2.

theorem mixture_ratio (hp : p > 0) (hq : q > 0) (hV : V > 0) (hW : W > 0) : 
  (p * V * (p + 2) + q * W * (p + 1)) / ((p + 1) * (q + 2) * (V + 2 * W)) =
  (p * V) / (p + 1) + (q * W) / (q + 2) :=
sorry

end mixture_ratio_l134_134095


namespace solve_equation_l134_134434

theorem solve_equation 
  (x : ℚ)
  (h : (x^2 + 3*x + 4)/(x + 5) = x + 6) :
  x = -13/4 := 
by
  sorry

end solve_equation_l134_134434


namespace sector_area_proof_l134_134475

/-- Define the radius and arc length as given -/
def radius : ℝ := 4
def arc_length : ℝ := 3.5

/-- Define the formula for the area of a sector -/
def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2

def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

def area_of_sector (l : ℝ) (circ : ℝ) (circle_area : ℝ) : ℝ := (l / circ) * circle_area

#eval let r := 4
        let l := 3.5
        let circ := circumference r
        let circle_area := area_of_circle r
        area_of_sector l circ circle_area

/-- Lean statement to prove that the area of the sector is 7 cm^2 -/
theorem sector_area_proof : area_of_sector arc_length (circumference radius) (area_of_circle radius) = 7 :=
by
    sorry

end sector_area_proof_l134_134475


namespace find_m_intersection_points_l134_134523

theorem find_m (m : ℝ) (hp : 2^2 + 2 * m + m^2 - 3 = 4) (h_pos : m > 0) : m = 1 := 
by
  sorry

theorem intersection_points (m : ℝ) (hp : 2^2 + 2 * m + m^2 - 3 = 4) (h_pos : m > 0) 
  (hm : m = 1) : ∃ x1 x2 : ℝ, (x^2 + x - 2 = 0) ∧ x1 ≠ x2 :=
by
  sorry

end find_m_intersection_points_l134_134523


namespace aunt_gave_each_20_l134_134547

theorem aunt_gave_each_20
  (jade_initial : ℕ)
  (julia_initial : ℕ)
  (total_after_aunt : ℕ)
  (equal_amount_from_aunt : ℕ)
  (h1 : jade_initial = 38)
  (h2 : julia_initial = jade_initial / 2)
  (h3 : total_after_aunt = 97)
  (h4 : jade_initial + julia_initial + 2 * equal_amount_from_aunt = total_after_aunt) :
  equal_amount_from_aunt = 20 := 
sorry

end aunt_gave_each_20_l134_134547


namespace complex_real_imag_eq_l134_134170

theorem complex_real_imag_eq (b : ℝ) (h : (2 + b) / 5 = (2 * b - 1) / 5) : b = 3 :=
  sorry

end complex_real_imag_eq_l134_134170


namespace rent_percentage_l134_134933

variable (E : ℝ)

def rent_last_year (E : ℝ) : ℝ := 0.20 * E 
def earnings_this_year (E : ℝ) : ℝ := 1.15 * E
def rent_this_year (E : ℝ) : ℝ := 0.25 * (earnings_this_year E)

-- Prove that the rent this year is 143.75% of the rent last year
theorem rent_percentage : (rent_this_year E) = 1.4375 * (rent_last_year E) :=
by
  sorry

end rent_percentage_l134_134933


namespace candy_last_days_l134_134938

variable (candy_from_neighbors candy_from_sister candy_per_day : ℕ)

theorem candy_last_days
  (h_candy_from_neighbors : candy_from_neighbors = 66)
  (h_candy_from_sister : candy_from_sister = 15)
  (h_candy_per_day : candy_per_day = 9) :
  let total_candy := candy_from_neighbors + candy_from_sister  
  (total_candy / candy_per_day) = 9 := by
  sorry

end candy_last_days_l134_134938


namespace find_irrational_satisfying_conditions_l134_134967

-- Define a real number x which is irrational
def is_irrational (x : ℝ) : Prop := ¬∃ (q : ℚ), (x : ℝ) = q

-- Define that x satisfies the given conditions
def rational_conditions (x : ℝ) : Prop :=
  (∃ (r1 : ℚ), x^3 - 17 * x = r1) ∧ (∃ (r2 : ℚ), x^2 + 4 * x = r2)

-- The main theorem statement
theorem find_irrational_satisfying_conditions (x : ℝ) 
  (hx_irr : is_irrational x) 
  (hx_cond : rational_conditions x) : x = -2 + Real.sqrt 5 ∨ x = -2 - Real.sqrt 5 :=
by
  sorry

end find_irrational_satisfying_conditions_l134_134967


namespace aunt_gave_each_20_l134_134546

theorem aunt_gave_each_20
  (jade_initial : ℕ)
  (julia_initial : ℕ)
  (total_after_aunt : ℕ)
  (equal_amount_from_aunt : ℕ)
  (h1 : jade_initial = 38)
  (h2 : julia_initial = jade_initial / 2)
  (h3 : total_after_aunt = 97)
  (h4 : jade_initial + julia_initial + 2 * equal_amount_from_aunt = total_after_aunt) :
  equal_amount_from_aunt = 20 := 
sorry

end aunt_gave_each_20_l134_134546


namespace initial_amount_l134_134796

theorem initial_amount (P R : ℝ) (h1 : 956 = P * (1 + (3 * R) / 100)) (h2 : 1052 = P * (1 + (3 * (R + 4)) / 100)) : P = 800 := 
by
  -- We would provide the proof steps here normally
  sorry

end initial_amount_l134_134796


namespace equal_area_split_l134_134619

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def circle1 : Circle := { center := (10, 90), radius := 4 }
def circle2 : Circle := { center := (15, 80), radius := 4 }
def circle3 : Circle := { center := (20, 85), radius := 4 }

theorem equal_area_split :
  ∃ m : ℝ, ∀ x y : ℝ, m * (x - 15) = y - 80 ∧ m = 0 ∧   
    ∀ circle : Circle, circle ∈ [circle1, circle2, circle3] →
      ∃ k : ℝ, k * (x - circle.center.1) + y - circle.center.2 = 0 :=
sorry

end equal_area_split_l134_134619


namespace total_chickens_on_farm_l134_134665

noncomputable def total_chickens (H R : ℕ) : ℕ := H + R

theorem total_chickens_on_farm (H R : ℕ) (h1 : H = 9 * R - 5) (h2 : H = 67) : total_chickens H R = 75 := 
by
  sorry

end total_chickens_on_farm_l134_134665


namespace probability_win_more_than_5000_l134_134934

def boxes : Finset ℕ := {5, 500, 5000}
def keys : Finset (Finset ℕ) := { {5}, {500}, {5000} }

noncomputable def probability_correct_key (box : ℕ) : ℚ :=
  if box = 5000 then 1 / 3 else if box = 500 then 1 / 2 else 1

theorem probability_win_more_than_5000 :
    (probability_correct_key 5000) * (probability_correct_key 500) = 1 / 6 :=
by
  -- Proof is omitted
  sorry

end probability_win_more_than_5000_l134_134934


namespace painted_faces_of_large_cube_l134_134059

theorem painted_faces_of_large_cube (n : ℕ) (unpainted_cubes : ℕ) :
  n = 9 ∧ unpainted_cubes = 343 → (painted_faces : ℕ) = 3 :=
by
  intros h
  let ⟨h_n, h_unpainted⟩ := h
  sorry

end painted_faces_of_large_cube_l134_134059


namespace solve_equation1_solve_equation2_l134_134751

theorem solve_equation1 :
  ∀ x : ℝ, ((x-1) * (x-1) = 3 * (x-1)) ↔ (x = 1 ∨ x = 4) :=
by
  intro x
  sorry

theorem solve_equation2 :
  ∀ x : ℝ, (x^2 - 4 * x + 1 = 0) ↔ (x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) :=
by
  intro x
  sorry

end solve_equation1_solve_equation2_l134_134751


namespace count_japanese_stamps_l134_134567

theorem count_japanese_stamps (total_stamps : ℕ) (perc_chinese perc_us : ℕ) (h1 : total_stamps = 100) 
  (h2 : perc_chinese = 35) (h3 : perc_us = 20): 
  total_stamps - ((perc_chinese * total_stamps / 100) + (perc_us * total_stamps / 100)) = 45 :=
by
  sorry

end count_japanese_stamps_l134_134567


namespace part_a_part_b_l134_134028

noncomputable def sequence_a (n : ℕ) : ℝ :=
  Real.cos (10^n * Real.pi / 180)

theorem part_a (h : 100 > 2) : sequence_a 100 > 0 := by
  sorry

theorem part_b : |sequence_a 100| < 0.18 := by
  sorry

end part_a_part_b_l134_134028


namespace min_value_of_m_l134_134534

def ellipse (x y : ℝ) := (y^2 / 16) + (x^2 / 9) = 1
def line (x y m : ℝ) := y = x + m
def shortest_distance (d : ℝ) := d = Real.sqrt 2

theorem min_value_of_m :
  ∃ (m : ℝ), (∀ (x y : ℝ), ellipse x y → ∃ d, shortest_distance d ∧ line x y m) 
  ∧ ∀ m', m' < m → ¬(∃ (x y : ℝ), ellipse x y ∧ ∃ d, shortest_distance d ∧ line x y m') :=
sorry

end min_value_of_m_l134_134534


namespace probability_two_even_dice_l134_134674

open ProbabilityTheory

theorem probability_two_even_dice : 
  let p_even := 1 / 2,
      p_odd := 1 / 2,
      number_of_ways := Nat.choose 4 2,
      probability_per_way := p_even ^ 2 * p_odd ^ 2,
      total_probability := (number_of_ways : ℚ) * probability_per_way in
  total_probability = 3 / 8 :=
by
  let p_even := 1 / 2
  let p_odd := 1 / 2
  let number_of_ways := Nat.choose 4 2
  let probability_per_way := p_even ^ 2 * p_odd ^ 2
  let total_probability := (number_of_ways : ℚ) * probability_per_way
  have probability_calculation : total_probability = 3 / 8 := by
    sorry
  exact probability_calculation

end probability_two_even_dice_l134_134674


namespace probability_of_F_l134_134218

-- Definitions for the probabilities of regions D, E, and the total probability
def P_D : ℚ := 3 / 8
def P_E : ℚ := 1 / 4
def total_probability : ℚ := 1

-- The hypothesis
lemma total_probability_eq_one : P_D + P_E + (1 - P_D - P_E) = total_probability :=
by
  simp [P_D, P_E, total_probability]

-- The goal is to prove this statement
theorem probability_of_F : 1 - P_D - P_E = 3 / 8 :=
by
  -- Using the total_probability_eq_one hypothesis
  have h := total_probability_eq_one
  -- This is a structured approach where verification using hypothesis and simplification can be done
  sorry

end probability_of_F_l134_134218


namespace f_sqrt_2_l134_134560

noncomputable def f : ℝ → ℝ :=
sorry

axiom domain_f : ∀ x, 0 < x → 0 < f x
axiom add_property : ∀ x y, f (x * y) = f x + f y
axiom f_at_8 : f 8 = 6

theorem f_sqrt_2 : f (Real.sqrt 2) = 1 :=
by
  have sqrt2pos : 0 < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  sorry

end f_sqrt_2_l134_134560


namespace scientific_notation_of_0_0000000005_l134_134293

theorem scientific_notation_of_0_0000000005 : 0.0000000005 = 5 * 10^(-10) :=
by {
  sorry
}

end scientific_notation_of_0_0000000005_l134_134293


namespace find_number_l134_134326

theorem find_number (number : ℝ) (h : 0.001 * number = 0.24) : number = 240 :=
sorry

end find_number_l134_134326


namespace scientific_notation_of_858_million_l134_134438

theorem scientific_notation_of_858_million :
  858000000 = 8.58 * 10 ^ 8 :=
sorry

end scientific_notation_of_858_million_l134_134438


namespace find_value_l134_134694

theorem find_value (x y : ℝ) (h : x - 2 * y = 1) : 3 - 4 * y + 2 * x = 5 := sorry

end find_value_l134_134694


namespace work_completion_time_l134_134931

theorem work_completion_time 
    (A B : ℝ) 
    (h1 : A = 2 * B) 
    (h2 : (A + B) * 18 = 1) : 
    1 / A = 27 := 
by 
    sorry

end work_completion_time_l134_134931


namespace parabola_vertex_l134_134447

theorem parabola_vertex :
  (∃ h k, ∀ x, (x^2 - 2 = ((x - h) ^ 2) + k) ∧ (h = 0) ∧ (k = -2)) :=
by
  sorry

end parabola_vertex_l134_134447


namespace greatest_integer_solution_l134_134320

theorem greatest_integer_solution (n : ℤ) (h : n^2 - 13 * n + 36 ≤ 0) : n ≤ 9 :=
by
  sorry

end greatest_integer_solution_l134_134320


namespace probability_both_numbers_are_prime_l134_134626

open Nat

def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

/-- Define the number of ways to choose two elements from a set of size n -/
def num_ways_to_choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_both_numbers_are_prime :
  let total_pairs := num_ways_to_choose_2 30
  let prime_pairs := num_ways_to_choose_2 primes_up_to_30.length
  (prime_pairs.to_rat / total_pairs.to_rat) = 5 / 48 := by
  sorry

end probability_both_numbers_are_prime_l134_134626


namespace work_days_l134_134324

theorem work_days (p_can : ℕ → ℝ) (q_can : ℕ → ℝ) (together_can: ℕ → ℝ) :
  (together_can 6 = 1) ∧ (q_can 10 = 1) → (1 / (p_can x) + 1 / (q_can 10) = 1 / (together_can 6)) → (x = 15) :=
by
  sorry

end work_days_l134_134324


namespace graph_inverse_prop_function_quadrants_l134_134509

theorem graph_inverse_prop_function_quadrants :
  ∀ x : ℝ, x ≠ 0 → (x > 0 ∧ y = 4 / x → y > 0) ∨ (x < 0 ∧ y = 4 / x → y < 0) := 
sorry

end graph_inverse_prop_function_quadrants_l134_134509


namespace sum_as_common_fraction_l134_134362

/-- The sum of 0.2 + 0.04 + 0.006 + 0.0008 + 0.00010 as a common fraction -/
theorem sum_as_common_fraction : (0.2 + 0.04 + 0.006 + 0.0008 + 0.00010) = (12345 / 160000) := by
  sorry

end sum_as_common_fraction_l134_134362


namespace car_speed_to_keep_window_dry_l134_134658

theorem car_speed_to_keep_window_dry :
  ∀ (v : ℝ) (alpha : ℝ), v = 2 ∧ α = 60 * Real.pi / 180 → 
  (∃ u : ℝ, u = 2 / Real.sqrt 3) :=
by
  intros v alpha h
  cases h
  use 2 / Real.sqrt 3
  sorry

end car_speed_to_keep_window_dry_l134_134658


namespace square_of_1005_l134_134132

theorem square_of_1005 : (1005 : ℕ)^2 = 1010025 := 
  sorry

end square_of_1005_l134_134132


namespace max_positive_integer_value_of_n_l134_134594

-- Define the arithmetic sequence with common difference d and first term a₁.
variable {d a₁ : ℝ}

-- The quadratic inequality condition which provides the solution set [0,9].
def inequality_condition (d a₁ : ℝ) : Prop :=
  ∀ (x : ℝ), (0 ≤ x ∧ x ≤ 9) → d * x^2 + 2 * a₁ * x ≥ 0

-- Maximum integer n such that the sum of the first n terms of the sequence is maximum.
noncomputable def max_n (d a₁ : ℝ) : ℕ :=
  if d < 0 then 5 else 0

-- Statement to be proved.
theorem max_positive_integer_value_of_n (d a₁ : ℝ) 
  (h : inequality_condition d a₁) : max_n d a₁ = 5 :=
sorry

end max_positive_integer_value_of_n_l134_134594


namespace probability_first_four_hearts_and_fifth_king_l134_134671

theorem probability_first_four_hearts_and_fifth_king :
  let total_cards := 52
  let hearts := 13
  let kings := 4
  let prob_first_heart := (hearts : ℚ) / total_cards
  let prob_second_heart := (hearts - 1 : ℚ) / (total_cards - 1)
  let prob_third_heart := (hearts - 2 : ℚ) / (total_cards - 2)
  let prob_fourth_heart := (hearts - 3 : ℚ) / (total_cards - 3)
  let prob_fifth_king := (kings : ℚ) / (total_cards - 4)
  prob_first_heart * prob_second_heart * prob_third_heart * prob_fourth_heart * prob_fifth_king = 286 / 124900 :=
by
  -- Definitions
  let total_cards := 52
  let hearts := 13
  let kings := 4
  
  -- Probabilities
  let prob_first_heart := (hearts : ℚ) / total_cards
  let prob_second_heart := (hearts - 1 : ℚ) / (total_cards - 1)
  let prob_third_heart := (hearts - 2 : ℚ) / (total_cards - 2)
  let prob_fourth_heart := (hearts - 3 : ℚ) / (total_cards - 3)
  let prob_fifth_king := (kings : ℚ) / (total_cards - 4)
  
  -- Equality
  have h : prob_first_heart * prob_second_heart * prob_third_heart * prob_fourth_heart * prob_fifth_king = 
    (13 / 52) * (12 / 51) * (11 / 50) * (10 / 49) * (1 / 12),
  by sorry
  rw h,
  calc (13 / 52) * (12 / 51) * (11 / 50) * (10 / 49) * (1 / 12) = 286 / 124900 : sorry -- Skip actual multiplication steps

end probability_first_four_hearts_and_fifth_king_l134_134671


namespace total_grapes_l134_134815

theorem total_grapes (r a n : ℕ) (h1 : r = 25) (h2 : a = r + 2) (h3 : n = a + 4) : r + a + n = 83 := by
  sorry

end total_grapes_l134_134815


namespace required_weekly_hours_approx_27_l134_134268

noncomputable def planned_hours_per_week : ℝ := 25
noncomputable def planned_weeks : ℝ := 15
noncomputable def total_amount : ℝ := 4500
noncomputable def sick_weeks : ℝ := 3
noncomputable def increased_wage_weeks : ℝ := 5
noncomputable def wage_increase_factor : ℝ := 1.5 -- 50%

-- Normal hourly wage
noncomputable def normal_hourly_wage : ℝ := total_amount / (planned_hours_per_week * planned_weeks)

-- Increased hourly wage
noncomputable def increased_hourly_wage : ℝ := normal_hourly_wage * wage_increase_factor

-- Earnings in the last 5 weeks at increased wage
noncomputable def earnings_in_last_5_weeks : ℝ := increased_hourly_wage * planned_hours_per_week * increased_wage_weeks

-- Amount needed before the wage increase
noncomputable def amount_needed_before_wage_increase : ℝ := total_amount - earnings_in_last_5_weeks

-- We have 7 weeks before the wage increase
noncomputable def weeks_before_increase : ℝ := planned_weeks - sick_weeks - increased_wage_weeks

-- New required weekly hours before wage increase
noncomputable def required_weekly_hours : ℝ := amount_needed_before_wage_increase / (normal_hourly_wage * weeks_before_increase)

theorem required_weekly_hours_approx_27 :
  abs (required_weekly_hours - 27) < 1 :=
sorry

end required_weekly_hours_approx_27_l134_134268


namespace seats_not_occupied_l134_134834

def seats_per_row : ℕ := 8
def total_rows : ℕ := 12
def seat_utilization_ratio : ℚ := 3 / 4

theorem seats_not_occupied : 
  (seats_per_row * total_rows) - (seats_per_row * seat_utilization_ratio * total_rows) = 24 := 
by
  sorry

end seats_not_occupied_l134_134834


namespace min_people_wearing_both_l134_134401

theorem min_people_wearing_both (n : ℕ) (h_lcm : n % 24 = 0) 
  (h_gloves : 3 * n % 8 = 0) (h_hats : 5 * n % 6 = 0) :
  ∃ x, x = 5 := 
by
  let gloves := 3 * n / 8
  let hats := 5 * n / 6
  let both := gloves + hats - n
  have h1 : both = 5 := sorry
  exact ⟨both, h1⟩

end min_people_wearing_both_l134_134401


namespace typesetter_times_l134_134171

theorem typesetter_times (α β γ : ℝ) (h1 : 1 / β - 1 / α = 10)
                                        (h2 : 1 / β - 1 / γ = 6)
                                        (h3 : 9 * (α + β) = 10 * (β + γ)) :
    α = 1 / 20 ∧ β = 1 / 30 ∧ γ = 1 / 24 :=
by {
  sorry
}

end typesetter_times_l134_134171


namespace same_solution_eq_l134_134265

theorem same_solution_eq (a b : ℤ) (x y : ℤ) 
  (h₁ : 4 * x + 3 * y = 11)
  (h₂ : a * x + b * y = -2)
  (h₃ : 3 * x - 5 * y = 1)
  (h₄ : b * x - a * y = 6) :
  (a + b) ^ 2023 = 0 := by
  sorry

end same_solution_eq_l134_134265


namespace area_of_inscribed_octagon_l134_134096

-- Define the given conditions and required proof
theorem area_of_inscribed_octagon (r : ℝ) (h : π * r^2 = 400 * π) :
  let A := r^2 * (1 + Real.sqrt 2)
  A = 20^2 * (1 + Real.sqrt 2) :=
by 
  sorry

end area_of_inscribed_octagon_l134_134096


namespace intervals_of_monotonicity_interval_max_min_l134_134387

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x - 2

theorem intervals_of_monotonicity :
  (∀ (x : ℝ), x < -1 → deriv f x < 0) ∧ 
  (∀ (x : ℝ), -1 < x ∧ x < 3 → deriv f x > 0) ∧ 
  (∀ (x : ℝ), x > 3 → deriv f x < 0) := 
sorry

theorem interval_max_min :
  f 2 = 20 → f (-1) = -7 := 
sorry

end intervals_of_monotonicity_interval_max_min_l134_134387


namespace problem_solution_l134_134253

theorem problem_solution (x y z : ℝ) (h1 : 2 * x - y - 2 * z - 6 = 0) (h2 : x^2 + y^2 + z^2 ≤ 4) :
  2 * x + y + z = 2 / 3 := 
by 
  sorry

end problem_solution_l134_134253


namespace duration_of_each_turn_l134_134323

-- Definitions based on conditions
def Wa := 1 / 4
def Wb := 1 / 12

-- Define the duration of each turn as T
def T : ℝ := 1 -- This is the correct answer we proved

-- Given conditions
def total_work_done := 6 * Wa + 6 * Wb

-- Lean statement to prove 
theorem duration_of_each_turn : T = 1 := by
  -- According to conditions, the total work done by a and b should equal the whole work
  have h1 : 3 * Wa + 3 * Wb = 1 := by sorry
  -- Let's conclude that T = 1
  sorry

end duration_of_each_turn_l134_134323


namespace Loris_needs_more_books_l134_134741

noncomputable def books_needed (Loris Darryl Lamont : ℕ) :=
  (Lamont - Loris)

theorem Loris_needs_more_books
  (darryl_books: ℕ)
  (lamont_books: ℕ)
  (loris_books_total: ℕ)
  (total_books: ℕ)
  (h1: lamont_books = 2 * darryl_books)
  (h2: darryl_books = 20)
  (h3: loris_books_total + darryl_books + lamont_books = total_books)
  (h4: total_books = 97) :
  books_needed loris_books_total darryl_books lamont_books = 3 :=
sorry

end Loris_needs_more_books_l134_134741


namespace total_combined_grapes_l134_134811

theorem total_combined_grapes :
  ∀ (r a y : ℕ), (r = 25) → (a = r + 2) → (y = a + 4) → (r + a + y = 83) :=
by
  intros r a y hr ha hy
  rw [hr, ha, hy]
  sorry

end total_combined_grapes_l134_134811


namespace solve_equation_l134_134431

theorem solve_equation (x : ℚ) (h : x ≠ -5) : 
  (x^2 + 3*x + 4) / (x + 5) = x + 6 ↔ x = -13 / 4 := by
  sorry

end solve_equation_l134_134431


namespace age_difference_l134_134122

theorem age_difference (A B n : ℕ) (h1 : A = B + n) (h2 : A - 1 = 3 * (B - 1)) (h3 : A = B^2) : n = 2 :=
by
  sorry

end age_difference_l134_134122


namespace nailcutter_sound_count_l134_134398

-- Definitions based on conditions
def nails_per_person : ℕ := 20
def number_of_customers : ℕ := 3
def sound_per_nail : ℕ := 1

-- The statement to prove 
theorem nailcutter_sound_count :
  (nails_per_person * number_of_customers * sound_per_nail) = 60 := by
  sorry

end nailcutter_sound_count_l134_134398


namespace next_two_series_numbers_l134_134269

theorem next_two_series_numbers :
  ∀ (a : ℕ → ℤ), a 1 = 2 → a 2 = 3 →
    (∀ n, 3 ≤ n → a n = a (n - 1) + a (n - 2) - 5) →
    a 7 = -26 ∧ a 8 = -45 :=
by
  intros a h1 h2 h3
  sorry

end next_two_series_numbers_l134_134269


namespace students_on_couch_per_room_l134_134950

def total_students : ℕ := 30
def total_rooms : ℕ := 6
def students_per_bed : ℕ := 2
def beds_per_room : ℕ := 2
def students_in_beds_per_room : ℕ := beds_per_room * students_per_bed

theorem students_on_couch_per_room :
  (total_students / total_rooms) - students_in_beds_per_room = 1 := by
  sorry

end students_on_couch_per_room_l134_134950


namespace hortense_flower_production_l134_134164

-- Define the initial conditions
def daisy_seeds : ℕ := 25
def sunflower_seeds : ℕ := 25
def daisy_germination_rate : ℚ := 0.60
def sunflower_germination_rate : ℚ := 0.80
def flower_production_rate : ℚ := 0.80

-- Prove the number of plants that produce flowers
theorem hortense_flower_production :
  (daisy_germination_rate * daisy_seeds + sunflower_germination_rate * sunflower_seeds) * flower_production_rate = 28 :=
by sorry

end hortense_flower_production_l134_134164


namespace bank_balance_after_five_years_l134_134802

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem bank_balance_after_five_years :
  let P0 := 5600
  let r1 := 0.03
  let r2 := 0.035
  let r3 := 0.04
  let r4 := 0.045
  let r5 := 0.05
  let D := 2000
  let A1 := compoundInterest P0 r1 1 1
  let A2 := compoundInterest A1 r2 1 1
  let A3 := compoundInterest (A2 + D) r3 1 1
  let A4 := compoundInterest A3 r4 1 1
  let A5 := compoundInterest A4 r5 1 1
  A5 = 9094.2 := by
  sorry

end bank_balance_after_five_years_l134_134802


namespace sum_first_five_terms_eq_ninety_three_l134_134255

variable (a : ℕ → ℕ)

-- Definitions
def geometric_sequence (a : ℕ → ℕ) : Prop :=
∀ n m : ℕ, a (n + m) = a n * a m

variables (a1 : ℕ) (a2 : ℕ) (a4 : ℕ)
variables (S : ℕ → ℕ)

-- Conditions
axiom a1_value : a1 = 3
axiom a2a4_value : a2 * a4 = 144

-- Question: Prove S_5 = 93
theorem sum_first_five_terms_eq_ninety_three
    (h1 : geometric_sequence a)
    (h2 : a 1 = a1)
    (h3 : a 2 = a2)
    (h4 : a 4 = a4)
    (Sn_def : S 5 = (a1 * (1 - (2:ℕ)^5)) / (1 - 2)) :
  S 5 = 93 :=
sorry

end sum_first_five_terms_eq_ninety_three_l134_134255


namespace total_combined_grapes_l134_134810

theorem total_combined_grapes :
  ∀ (r a y : ℕ), (r = 25) → (a = r + 2) → (y = a + 4) → (r + a + y = 83) :=
by
  intros r a y hr ha hy
  rw [hr, ha, hy]
  sorry

end total_combined_grapes_l134_134810


namespace area_of_parallelogram_l134_134684

theorem area_of_parallelogram (base height : ℝ) (h_base : base = 12) (h_height : height = 8) :
  base * height = 96 := by
  sorry

end area_of_parallelogram_l134_134684


namespace probability_same_color_given_first_red_l134_134275

-- Definitions of events
def event_A (draw1 : ℕ) : Prop := draw1 = 1 -- Event A: the first ball drawn is red (drawing 1 means the first ball is red)

def event_B (draw1 draw2 : ℕ) : Prop := -- Event B: the two balls drawn are of the same color
  (draw1 = 1 ∧ draw2 = 1) ∨ (draw1 = 2 ∧ draw2 = 2)

-- Given probabilities
def P_A : ℚ := 2 / 5
def P_AB : ℚ := (2 / 5) * (1 / 4)

-- The conditional probability P(B|A)
def P_B_given_A : ℚ := P_AB / P_A

theorem probability_same_color_given_first_red : P_B_given_A = 1 / 4 := 
by 
  unfold P_B_given_A P_A P_AB
  sorry

end probability_same_color_given_first_red_l134_134275


namespace other_root_of_quadratic_l134_134494

theorem other_root_of_quadratic (a b k : ℝ) (h : 1^2 - (a+b) * 1 + ab * (1 - k) = 0) : 
  ∃ r : ℝ, r = a + b - 1 := 
sorry

end other_root_of_quadratic_l134_134494


namespace volume_of_fifth_section_l134_134883

theorem volume_of_fifth_section
  (a : ℕ → ℚ)
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)  -- Arithmetic sequence constraint
  (h_sum_top_four : a 0 + a 1 + a 2 + a 3 = 3)  -- Sum of the top four sections
  (h_sum_bottom_three : a 6 + a 7 + a 8 = 4)  -- Sum of the bottom three sections
  : a 4 = 67 / 66 := sorry

end volume_of_fifth_section_l134_134883


namespace difference_in_pages_l134_134356

def purple_pages_per_book : ℕ := 230
def orange_pages_per_book : ℕ := 510
def purple_books_read : ℕ := 5
def orange_books_read : ℕ := 4

theorem difference_in_pages : 
  orange_books_read * orange_pages_per_book - purple_books_read * purple_pages_per_book = 890 :=
by
  sorry

end difference_in_pages_l134_134356


namespace isosceles_triangle_triangle_area_l134_134174

noncomputable def area_of_Δ (a b c : ℝ) (cosA : ℝ) : ℝ :=
  1/2 * b * c * (Real.sqrt (1 - cosA^2))

theorem isosceles_triangle {a b c : ℝ} (h : b * Real.cos c = a * Real.cos B^2 + b * Real.cos A * Real.cos B) :
  B = c :=
sorry

theorem triangle_area {a b c : ℝ} (cosA : ℝ) (cosA_eq : cosA = 7/8) (perimeter : a + b + c = 5) 
  (b_eq_c : b = c) :
  area_of_Δ a b c cosA = Real.sqrt 15 / 4 :=
sorry

end isosceles_triangle_triangle_area_l134_134174


namespace probability_exactly_one_first_class_l134_134463

-- Define the probabilities
def prob_first_class_first_intern : ℚ := 2 / 3
def prob_first_class_second_intern : ℚ := 3 / 4
def prob_not_first_class_first_intern : ℚ := 1 - prob_first_class_first_intern
def prob_not_first_class_second_intern : ℚ := 1 - prob_first_class_second_intern

-- Define the event A, which is the event that exactly one of the two parts is of first-class quality
def prob_event_A : ℚ :=
  (prob_first_class_first_intern * prob_not_first_class_second_intern) +
  (prob_not_first_class_first_intern * prob_first_class_second_intern)

theorem probability_exactly_one_first_class (h1 : prob_first_class_first_intern = 2 / 3) 
    (h2 : prob_first_class_second_intern = 3 / 4) 
    (h3 : prob_event_A = 
          (prob_first_class_first_intern * (1 - prob_first_class_second_intern)) + 
          ((1 - prob_first_class_first_intern) * prob_first_class_second_intern)) : 
  prob_event_A = 5 / 12 := 
  sorry

end probability_exactly_one_first_class_l134_134463


namespace flooring_sq_ft_per_box_l134_134314

/-- The problem statement converted into a Lean theorem -/
theorem flooring_sq_ft_per_box
  (living_room_length : ℕ)
  (living_room_width : ℕ)
  (flooring_installed : ℕ)
  (additional_boxes : ℕ)
  (correct_answer : ℕ) 
  (h1 : living_room_length = 16)
  (h2 : living_room_width = 20)
  (h3 : flooring_installed = 250)
  (h4 : additional_boxes = 7)
  (h5 : correct_answer = 10) :
  
  (living_room_length * living_room_width - flooring_installed) / additional_boxes = correct_answer :=
by 
  sorry

end flooring_sq_ft_per_box_l134_134314


namespace inequality_selection_l134_134791

theorem inequality_selection (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) 
  (h₃ : ∀ x : ℝ, |x + a| + |x - b| + c ≥ 4) : 
  a + b + c = 4 ∧ (∀ x, |x + a| + |x - b| + c = 4 → x = (a - b)/2) ∧ (a = 8 / 7 ∧ b = 18 / 7 ∧ c = 2 / 7) :=
by
  sorry

end inequality_selection_l134_134791


namespace sum_first_32_terms_bn_l134_134154

noncomputable def a_n (n : ℕ) : ℝ := 3 * n + 1

noncomputable def b_n (n : ℕ) : ℝ :=
  1 / ((a_n n) * Real.sqrt (a_n (n + 1)) + (a_n (n + 1)) * Real.sqrt (a_n n))

noncomputable def sum_bn (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) b_n

theorem sum_first_32_terms_bn : sum_bn 32 = 2 / 15 := 
sorry

end sum_first_32_terms_bn_l134_134154


namespace sufficient_but_not_necessary_condition_l134_134913

theorem sufficient_but_not_necessary_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x > 0 ∧ y > 0 → (x / y + y / x ≥ 2)) ∧ ¬((x / y + y / x ≥ 2) → (x > 0 ∧ y > 0)) :=
sorry

end sufficient_but_not_necessary_condition_l134_134913


namespace smallest_n_for_candy_distribution_l134_134350

theorem smallest_n_for_candy_distribution : ∃ (n : ℕ), (∀ (a : ℕ), ∃ (x : ℕ), (x * (x + 1)) / 2 % n = a % n) ∧ n = 2 :=
sorry

end smallest_n_for_candy_distribution_l134_134350


namespace meal_combinations_l134_134782

def number_of_menu_items : ℕ := 15

theorem meal_combinations (different_orderings : ∀ Yann Camille : ℕ, Yann ≠ Camille → Yann ≤ number_of_menu_items ∧ Camille ≤ number_of_menu_items) : 
  (number_of_menu_items * (number_of_menu_items - 1)) = 210 :=
by sorry

end meal_combinations_l134_134782


namespace ordered_triples_count_l134_134165

open Real

theorem ordered_triples_count :
  ∃ (S : Finset (ℝ × ℝ × ℝ)),
    (∀ (a b c : ℝ), (a, b, c) ∈ S ↔ (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ab = c ∧ bc = a + b ∧ ca = b)) ∧
    S.card = 2 := 
sorry

end ordered_triples_count_l134_134165


namespace f_2012_eq_3_l134_134145

noncomputable def f (a b α β : ℝ) (x : ℝ) : ℝ := a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem f_2012_eq_3 
  (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) 
  (h : f a b α β 2011 = 5) : 
  f a b α β 2012 = 3 :=
by
  sorry

end f_2012_eq_3_l134_134145


namespace minimum_value_exists_l134_134019

theorem minimum_value_exists (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_condition : x + 4 * y = 2) : 
  ∃ z : ℝ, z = (x + 40 * y + 4) / (3 * x * y) ∧ z ≥ 18 :=
by
  sorry

end minimum_value_exists_l134_134019


namespace cube_face_sum_l134_134912

theorem cube_face_sum (a b c d e f : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0) :
  (a * b * c + a * e * c + a * b * f + a * e * f + d * b * c + d * e * c + d * b * f + d * e * f = 1287) →
  (a + d + b + e + c + f = 33) :=
by
  sorry

end cube_face_sum_l134_134912


namespace yeast_population_at_1_20_pm_l134_134423

def yeast_population (initial : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  initial * rate^time

theorem yeast_population_at_1_20_pm : 
  yeast_population 50 3 4 = 4050 :=
by
  -- Proof goes here
  sorry

end yeast_population_at_1_20_pm_l134_134423


namespace alan_glasses_drank_l134_134417

-- Definition for the rate of drinking water
def glass_per_minutes := 1 / 20

-- Definition for the total time in minutes
def total_minutes := 5 * 60

-- Theorem stating the number of glasses Alan will drink in the given time
theorem alan_glasses_drank : (glass_per_minutes * total_minutes) = 15 :=
by 
  sorry

end alan_glasses_drank_l134_134417


namespace evaluate_expression_l134_134358

theorem evaluate_expression (x : ℝ) : 
  (36 + 12 * x) ^ 2 - (12^2 * x^2 + 36^2) = 864 * x :=
by
  sorry

end evaluate_expression_l134_134358


namespace find_M_l134_134527

theorem find_M : 995 + 997 + 999 + 1001 + 1003 = 5100 - 104 :=
by 
  sorry

end find_M_l134_134527


namespace more_girls_than_boys_l134_134879

theorem more_girls_than_boys (total students : ℕ) (girls boys : ℕ) (h1 : total = 41) (h2 : girls = 22) (h3 : girls + boys = total) : (girls - boys) = 3 :=
by
  sorry

end more_girls_than_boys_l134_134879


namespace determine_a_l134_134870

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 + (Real.exp x - a)^2

theorem determine_a (a x₀ : ℝ)
  (h₀ : f x₀ a ≤ 1/2) : a = 1/2 :=
sorry

end determine_a_l134_134870


namespace find_n_l134_134035

theorem find_n (n : ℕ) (h1 : 0 < n) (h2 : n < 11) (h3 : (18888 - n) % 11 = 0) : n = 1 :=
by
  sorry

end find_n_l134_134035


namespace grapes_total_sum_l134_134808

theorem grapes_total_sum (R A N : ℕ) 
  (h1 : A = R + 2) 
  (h2 : N = A + 4) 
  (h3 : R = 25) : 
  R + A + N = 83 := by
  sorry

end grapes_total_sum_l134_134808


namespace hexagons_formed_square_z_l134_134957

theorem hexagons_formed_square_z (a b s z : ℕ) (hexagons_congruent : a = 9 ∧ b = 16 ∧ s = 12 ∧ z = 4): 
(z = 4) := by
  sorry

end hexagons_formed_square_z_l134_134957


namespace solve_equation_l134_134428

theorem solve_equation (x : ℚ) :
  (x^2 + 3 * x + 4) / (x + 5) = x + 6 ↔ x = -13 / 4 :=
by
  sorry

end solve_equation_l134_134428


namespace gcd_9011_2147_l134_134205

theorem gcd_9011_2147 : Int.gcd 9011 2147 = 1 := sorry

end gcd_9011_2147_l134_134205


namespace tan_sum_identity_l134_134518

theorem tan_sum_identity (x : ℝ) (h : Real.tan (x + Real.pi / 4) = 2) : Real.tan x = 1 / 3 := 
by 
  sorry

end tan_sum_identity_l134_134518


namespace mrs_choi_profit_percentage_l134_134058

theorem mrs_choi_profit_percentage :
  ∀ (original_price selling_price : ℝ) (broker_percentage : ℝ),
    original_price = 80000 →
    selling_price = 100000 →
    broker_percentage = 0.05 →
    (selling_price - (broker_percentage * original_price) - original_price) / original_price * 100 = 20 :=
by
  intros original_price selling_price broker_percentage h1 h2 h3
  sorry

end mrs_choi_profit_percentage_l134_134058


namespace values_of_a2_add_b2_l134_134042

theorem values_of_a2_add_b2 (a b : ℝ) (h1 : a^3 - 3 * a * b^2 = 11) (h2 : b^3 - 3 * a^2 * b = 2) : a^2 + b^2 = 5 := 
by
  sorry

end values_of_a2_add_b2_l134_134042


namespace total_children_l134_134312

-- Definitions for the conditions in the problem
def boys : ℕ := 19
def girls : ℕ := 41

-- Theorem stating the total number of children is 60
theorem total_children : boys + girls = 60 :=
by
  -- calculation done to show steps, but not necessary for the final statement
  sorry

end total_children_l134_134312


namespace andrea_sod_rectangles_l134_134817

def section_1_length : ℕ := 35
def section_1_width : ℕ := 42
def section_2_length : ℕ := 55
def section_2_width : ℕ := 86
def section_3_length : ℕ := 20
def section_3_width : ℕ := 50
def section_4_length : ℕ := 48
def section_4_width : ℕ := 66

def sod_length : ℕ := 3
def sod_width : ℕ := 4

def area (length width : ℕ) : ℕ := length * width
def sod_area : ℕ := area sod_length sod_width

def rectangles_needed (section_length section_width sod_area : ℕ) : ℕ :=
  (area section_length section_width + sod_area - 1) / sod_area

def total_rectangles_needed : ℕ :=
  rectangles_needed section_1_length section_1_width sod_area +
  rectangles_needed section_2_length section_2_width sod_area +
  rectangles_needed section_3_length section_3_width sod_area +
  rectangles_needed section_4_length section_4_width sod_area

theorem andrea_sod_rectangles : total_rectangles_needed = 866 := by
  sorry

end andrea_sod_rectangles_l134_134817


namespace resulting_solid_vertices_l134_134484

theorem resulting_solid_vertices (s1 s2 : ℕ) (orig_vertices removed_cubes : ℕ) :
  s1 = 5 → s2 = 2 → orig_vertices = 8 → removed_cubes = 8 → 
  (orig_vertices - removed_cubes + removed_cubes * (4 * 3 - 3)) = 40 := by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end resulting_solid_vertices_l134_134484


namespace concurrency_iff_concyclity_l134_134048

variables {A B C D E F I : Point ℝ}

-- Definitions of conditions
def is_cyclic (A B C D : Point ℝ) : Prop := 
  ∃ (Γ : Circle ℝ), A ∈ Γ ∧ B ∈ Γ ∧ C ∈ Γ ∧ D ∈ Γ

def lines_concurrent (A B C D E F I : Point ℝ) : Prop :=
  ∃ I : Point ℝ, lies_on_line I A B ∧ lies_on_line I C D ∧ lies_on_line I E F

-- Problem to prove
theorem concurrency_iff_concyclity
  (h₁ : is_cyclic A B C D)
  (h₂ : is_cyclic C D E F)
  (h₃ : ¬(parallel (line_through A B) (line_through C D)))
  (h₄ : ¬(parallel (line_through A B) (line_through E F)))
  (h₅ : ¬(parallel (line_through C D) (line_through E F))) :
  (lines_concurrent A B C D E F) ↔ (is_cyclic A B E F) :=
sorry

end concurrency_iff_concyclity_l134_134048


namespace original_cost_price_l134_134652

theorem original_cost_price (selling_price_friend : ℝ) (gain_percent : ℝ) (loss_percent : ℝ) 
  (final_selling_price : ℝ) : 
  final_selling_price = 54000 → gain_percent = 0.2 → loss_percent = 0.1 → 
  selling_price_friend = (1 - loss_percent) * x → final_selling_price = (1 + gain_percent) * selling_price_friend → 
  x = 50000 :=
by 
  sorry

end original_cost_price_l134_134652


namespace factorial_expression_l134_134488

open Nat

theorem factorial_expression : ((sqrt (5! * 4!)) ^ 2 + 3!) = 2886 := by
  sorry

end factorial_expression_l134_134488


namespace ordered_pairs_count_l134_134508

theorem ordered_pairs_count :
  (∃ (a b : ℝ), (∃ (x y : ℤ),
    a * (x : ℝ) + b * (y : ℝ) = 1 ∧
    (x : ℝ)^2 + (y : ℝ)^2 = 65)) →
  ∃ (n : ℕ), n = 128 :=
by
  sorry

end ordered_pairs_count_l134_134508


namespace haley_marbles_l134_134038

theorem haley_marbles (boys marbles_per_boy : ℕ) (h1: boys = 5) (h2: marbles_per_boy = 7) : boys * marbles_per_boy = 35 := 
by 
  sorry

end haley_marbles_l134_134038


namespace MattSkipsRopesTimesPerSecond_l134_134057

theorem MattSkipsRopesTimesPerSecond:
  ∀ (minutes_jumped : ℕ) (total_skips : ℕ), 
  minutes_jumped = 10 → 
  total_skips = 1800 → 
  (total_skips / (minutes_jumped * 60)) = 3 :=
by
  intros minutes_jumped total_skips h_jumped h_skips
  sorry

end MattSkipsRopesTimesPerSecond_l134_134057


namespace neg_pow_eq_pow_four_l134_134468

variable (a : ℝ)

theorem neg_pow_eq_pow_four (a : ℝ) : (-a)^4 = a^4 :=
sorry

end neg_pow_eq_pow_four_l134_134468


namespace rectangular_prism_sum_l134_134405

-- Definitions based on conditions
def edges := 12
def corners := 8
def faces := 6

-- Lean statement to prove question == answer given conditions.
theorem rectangular_prism_sum : edges + corners + faces = 26 := by
  sorry

end rectangular_prism_sum_l134_134405


namespace intersection_point_exists_l134_134435

def equation_1 (x y : ℝ) : Prop := 3 * x^2 - 12 * y^2 = 48
def line_eq (x y : ℝ) : Prop := y = - (1 / 3) * x + 5

theorem intersection_point_exists :
  ∃ (x y : ℝ), equation_1 x y ∧ line_eq x y ∧ x = 75 / 8 ∧ y = 15 / 8 :=
sorry

end intersection_point_exists_l134_134435


namespace min_marbles_to_draw_l134_134793

theorem min_marbles_to_draw (reds greens blues yellows oranges purples : ℕ)
  (h_reds : reds = 35)
  (h_greens : greens = 25)
  (h_blues : blues = 24)
  (h_yellows : yellows = 18)
  (h_oranges : oranges = 15)
  (h_purples : purples = 12)
  : ∃ n : ℕ, n = 103 ∧ (∀ r g b y o p : ℕ, 
       r ≤ reds ∧ g ≤ greens ∧ b ≤ blues ∧ y ≤ yellows ∧ o ≤ oranges ∧ p ≤ purples ∧ 
       r < 20 ∧ g < 20 ∧ b < 20 ∧ y < 20 ∧ o < 20 ∧ p < 20 → r + g + b + y + o + p < n) ∧
      (∀ r g b y o p : ℕ, 
       r ≤ reds ∧ g ≤ greens ∧ b ≤ blues ∧ y ≤ yellows ∧ o ≤ oranges ∧ p ≤ purples ∧ 
       r + g + b + y + o + p = n → r = 20 ∨ g = 20 ∨ b = 20 ∨ y = 20 ∨ o = 20 ∨ p = 20) :=
sorry

end min_marbles_to_draw_l134_134793


namespace total_cost_of_backpack_and_pencil_case_l134_134224

-- Definitions based on the given conditions
def pencil_case_price : ℕ := 8
def backpack_price : ℕ := 5 * pencil_case_price

-- Statement of the proof problem
theorem total_cost_of_backpack_and_pencil_case : 
  pencil_case_price + backpack_price = 48 :=
by
  -- Skip the proof
  sorry

end total_cost_of_backpack_and_pencil_case_l134_134224


namespace solve_equation_l134_134426

theorem solve_equation (x : ℚ) :
  (x^2 + 3 * x + 4) / (x + 5) = x + 6 ↔ x = -13 / 4 :=
by
  sorry

end solve_equation_l134_134426


namespace max_sum_of_positives_l134_134878

theorem max_sum_of_positives (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y + 1 / x + 1 / y = 5) : x + y ≤ 4 :=
sorry

end max_sum_of_positives_l134_134878


namespace problem1_problem2_problem3_l134_134182

-- Problem Conditions
def inductive_reasoning (s: Sort _) (g: Sort _) : Prop := 
  ∀ (x: s → g), true 

def probabilistic_conclusion : Prop :=
  ∀ (x : Prop), true

def analogical_reasoning (a: Sort _) : Prop := 
  ∀ (x: a), true 

-- The Statements to be Proved
theorem problem1 : ¬ inductive_reasoning Prop Prop = true := 
sorry

theorem problem2 : probabilistic_conclusion = true :=
sorry 

theorem problem3 : ¬ analogical_reasoning Prop = true :=
sorry 

end problem1_problem2_problem3_l134_134182


namespace fish_left_in_sea_l134_134327

theorem fish_left_in_sea : 
  let westward_initial := 1800
  let eastward_initial := 3200
  let north_initial := 500
  let eastward_caught := (2 / 5) * eastward_initial
  let westward_caught := (3 / 4) * westward_initial
  let eastward_left := eastward_initial - eastward_caught
  let westward_left := westward_initial - westward_caught
  let north_left := north_initial
  eastward_left + westward_left + north_left = 2870 := 
by 
  sorry

end fish_left_in_sea_l134_134327


namespace probability_of_two_primes_is_correct_l134_134633

open Finset

noncomputable def probability_two_primes : ℚ :=
  let primes := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  let total_ways := (finset.range 30).card.choose 2
  let prime_ways := primes.card.choose 2
  prime_ways / total_ways

theorem probability_of_two_primes_is_correct :
  probability_two_primes = 15 / 145 :=
by
  -- Proof omitted
  sorry

end probability_of_two_primes_is_correct_l134_134633


namespace poly_sum_of_squares_iff_nonneg_l134_134740

open Polynomial

variable {R : Type*} [Ring R] [OrderedRing R]

theorem poly_sum_of_squares_iff_nonneg (A : Polynomial ℝ) :
  (∃ P Q : Polynomial ℝ, A = P^2 + Q^2) ↔ ∀ x : ℝ, 0 ≤ A.eval x := sorry

end poly_sum_of_squares_iff_nonneg_l134_134740


namespace triangle_obtuse_of_cos_relation_l134_134700

theorem triangle_obtuse_of_cos_relation
  (a b c : ℝ)
  (A B C : ℝ)
  (hTriangle : A + B + C = Real.pi)
  (hSides : a^2 = b^2 + c^2 - 2*b*c*Real.cos A)
  (hSides' : b^2 = a^2 + c^2 - 2*a*c*Real.cos B)
  (hSides'' : c^2 = a^2 + b^2 - 2*a*b*Real.cos C)
  (hRelation : a * Real.cos C = b + 2/3 * c) :
 ∃ (A' : ℝ), A' = A ∧ A > (Real.pi / 2) := 
sorry

end triangle_obtuse_of_cos_relation_l134_134700


namespace probability_of_less_than_20_l134_134787

variable (total_people : ℕ) (people_over_30 : ℕ)
variable (people_under_20 : ℕ) (probability_under_20 : ℝ)

noncomputable def group_size := total_people = 150
noncomputable def over_30 := people_over_30 = 90
noncomputable def under_20 := people_under_20 = total_people - people_over_30

theorem probability_of_less_than_20
  (total_people_eq : total_people = 150)
  (people_over_30_eq : people_over_30 = 90)
  (people_under_20_eq : people_under_20 = 60)
  (under_20_eq : 60 = total_people - people_over_30) :
  probability_under_20 = people_under_20 / total_people := by
  sorry

end probability_of_less_than_20_l134_134787


namespace number_of_polynomials_is_seven_l134_134884

-- Definitions of what constitutes a polynomial
def is_polynomial (expr : String) : Bool :=
  match expr with
  | "3/4*x^2" => true
  | "3ab" => true
  | "x+5" => true
  | "y/5x" => false
  | "-1" => true
  | "y/3" => true
  | "a^2-b^2" => true
  | "a" => true
  | _ => false

-- Given set of algebraic expressions
def expressions : List String := 
  ["3/4*x^2", "3ab", "x+5", "y/5x", "-1", "y/3", "a^2-b^2", "a"]

-- Count the number of polynomials in the given expressions
def count_polynomials (exprs : List String) : Nat :=
  exprs.foldr (fun expr count => if is_polynomial expr then count + 1 else count) 0

theorem number_of_polynomials_is_seven : count_polynomials expressions = 7 :=
  by
    sorry

end number_of_polynomials_is_seven_l134_134884


namespace nonnegative_integer_count_l134_134709

def balanced_quaternary_nonnegative_count : Nat :=
  let base := 4
  let max_index := 6
  let valid_digits := [-1, 0, 1]
  let max_sum := (base ^ (max_index + 1) - 1) / (base - 1)
  max_sum + 1

theorem nonnegative_integer_count : balanced_quaternary_nonnegative_count = 5462 := by
  sorry

end nonnegative_integer_count_l134_134709


namespace reimbursement_correct_l134_134556

-- Define the days and miles driven each day
def miles_monday : ℕ := 18
def miles_tuesday : ℕ := 26
def miles_wednesday : ℕ := 20
def miles_thursday : ℕ := 20
def miles_friday : ℕ := 16

-- Define the mileage rate
def mileage_rate : ℝ := 0.36

-- Define the total miles driven
def total_miles_driven : ℕ := miles_monday + miles_tuesday + miles_wednesday + miles_thursday + miles_friday

-- Define the total reimbursement
def reimbursement : ℝ := total_miles_driven * mileage_rate

-- Prove that the reimbursement is $36
theorem reimbursement_correct : reimbursement = 36 := by
  sorry

end reimbursement_correct_l134_134556


namespace find_number_l134_134715

theorem find_number (some_number : ℤ) : 45 - (28 - (some_number - (15 - 19))) = 58 ↔ some_number = 37 := 
by 
  sorry

end find_number_l134_134715


namespace greatest_four_digit_n_l134_134318

theorem greatest_four_digit_n :
  ∃ (n : ℕ), (1000 ≤ n ∧ n ≤ 9999) ∧ (∃ m : ℕ, n + 1 = m^2) ∧ ¬(n! % (n * (n + 1) / 2) = 0) ∧ n = 9999 :=
by sorry

end greatest_four_digit_n_l134_134318


namespace no_real_solutions_l134_134964

theorem no_real_solutions : ∀ x : ℝ, ¬(3 * x - 2 * x + 8) ^ 2 = -|x| - 4 :=
by
  intro x
  sorry

end no_real_solutions_l134_134964


namespace distinct_necklace_arrangements_l134_134592

open_locale big_operators

/-- The number of distinct necklace arrangements with 6 red, 1 white, and 8 yellow balls, 
    considering rotational and reflectional symmetries, is 1519. -/
theorem distinct_necklace_arrangements :
  let n := 15 in
  let r := 6 in
  let w := 1 in
  let y := 8 in
  (n = r + w + y) →
  ∑ k in finset.range ((r + y)! / (r! * y!)), 2 • 1 = 1519 :=
by
  intros n r w y h_n
  have h1: 14! / (6! * 8!) = 3003 := sorry
  have h2: 3003 / 2 = 1501.5 := sorry
  have h3: (3003 - 35) / 2 + 35 = 1519 := sorry
  exact h3

end distinct_necklace_arrangements_l134_134592


namespace total_accidents_all_three_highways_l134_134554

def highway_conditions : Type :=
  (accident_rate : ℕ, per_million : ℕ, total_traffic : ℕ)

def highway_a : highway_conditions := (75, 100, 2500)
def highway_b : highway_conditions := (50, 80, 1600)
def highway_c : highway_conditions := (90, 200, 1900)

def total_accidents (hc : highway_conditions) : ℕ :=
  hc.accident_rate * hc.total_traffic / hc.per_million

theorem total_accidents_all_three_highways :
  total_accidents highway_a +
  total_accidents highway_b +
  total_accidents highway_c = 3730 := by
  sorry

end total_accidents_all_three_highways_l134_134554


namespace simplify_expression_l134_134588

theorem simplify_expression (x y z : ℝ) : - (x - (y - z)) = -x + y - z := by
  sorry

end simplify_expression_l134_134588


namespace derivative_of_m_l134_134876

noncomputable def m (x : ℝ) : ℝ := (2 : ℝ)^x / (1 + x)

theorem derivative_of_m (x : ℝ) : 
  deriv m x = (2^x * (1 + x) * Real.log 2 - 2^x) / (1 + x)^2 :=
by
  sorry

end derivative_of_m_l134_134876


namespace ducks_killed_is_20_l134_134243

variable (x : ℕ)

def killed_ducks_per_year (x : ℕ) : Prop :=
  let initial_flock := 100
  let annual_births := 30
  let years := 5
  let additional_flock := 150
  let final_flock := 300
  initial_flock + years * (annual_births - x) + additional_flock = final_flock

theorem ducks_killed_is_20 : killed_ducks_per_year 20 :=
by
  sorry

end ducks_killed_is_20_l134_134243


namespace a_must_be_negative_l134_134711

theorem a_must_be_negative (a b : ℝ) (h1 : b > 0) (h2 : a / b < -2 / 3) : a < 0 :=
sorry

end a_must_be_negative_l134_134711


namespace parabola_focus_value_of_a_l134_134107

theorem parabola_focus_value_of_a :
  (∀ a : ℝ, (∃ y : ℝ, y = a * (0^2) ∧ (0, y) = (0, 3 / 8)) → a = 2 / 3) := by
sorry

end parabola_focus_value_of_a_l134_134107


namespace ratio_third_to_second_is_one_l134_134965

variable (x y : ℕ)

-- The second throw skips 2 more times than the first throw
def second_throw := x + 2
-- The third throw skips y times
def third_throw := y
-- The fourth throw skips 3 fewer times than the third throw
def fourth_throw := y - 3
-- The fifth throw skips 1 more time than the fourth throw
def fifth_throw := (y - 3) + 1

-- The fifth throw skipped 8 times
axiom fifth_throw_condition : fifth_throw y = 8
-- The total number of skips between all throws is 33
axiom total_skips_condition : x + second_throw x + y + fourth_throw y + fifth_throw y = 33

-- Prove the ratio of skips in third throw to the second throw is 1:1
theorem ratio_third_to_second_is_one : (third_throw y) / (second_throw x) = 1 := sorry

end ratio_third_to_second_is_one_l134_134965


namespace probability_of_karnataka_student_l134_134908

-- Defining the conditions

-- Number of students from each region
def total_students : ℕ := 10
def maharashtra_students : ℕ := 4
def karnataka_students : ℕ := 3
def goa_students : ℕ := 3

-- Number of students to be selected
def students_to_select : ℕ := 4

-- Total ways to choose 4 students out of 10
def C_total : ℕ := Nat.choose total_students students_to_select

-- Ways to select 4 students from the 7 students not from Karnataka
def non_karnataka_students : ℕ := maharashtra_students + goa_students
def C_non_karnataka : ℕ := Nat.choose non_karnataka_students students_to_select

-- Probability calculations
def P_no_karnataka : ℚ := C_non_karnataka / C_total
def P_at_least_one_karnataka : ℚ := 1 - P_no_karnataka

-- The statement to be proved
theorem probability_of_karnataka_student :
  P_at_least_one_karnataka = 5 / 6 :=
sorry

end probability_of_karnataka_student_l134_134908


namespace part_a_part_b_l134_134027

noncomputable def sequence_a (n : ℕ) : ℝ :=
  Real.cos (10^n * Real.pi / 180)

theorem part_a (h : 100 > 2) : sequence_a 100 > 0 := by
  sorry

theorem part_b : |sequence_a 100| < 0.18 := by
  sorry

end part_a_part_b_l134_134027


namespace probability_of_two_primes_is_correct_l134_134632

open Finset

noncomputable def probability_two_primes : ℚ :=
  let primes := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  let total_ways := (finset.range 30).card.choose 2
  let prime_ways := primes.card.choose 2
  prime_ways / total_ways

theorem probability_of_two_primes_is_correct :
  probability_two_primes = 15 / 145 :=
by
  -- Proof omitted
  sorry

end probability_of_two_primes_is_correct_l134_134632


namespace seats_not_occupied_l134_134833

def seats_per_row : ℕ := 8
def total_rows : ℕ := 12
def seat_utilization_ratio : ℚ := 3 / 4

theorem seats_not_occupied : 
  (seats_per_row * total_rows) - (seats_per_row * seat_utilization_ratio * total_rows) = 24 := 
by
  sorry

end seats_not_occupied_l134_134833


namespace remainder_127_14_l134_134928

theorem remainder_127_14 : ∃ r : ℤ, r = 127 - (14 * 9) ∧ r = 1 := by
  sorry

end remainder_127_14_l134_134928


namespace find_c_l134_134510

-- Define the necessary conditions for the circle equation and the radius
variable (c : ℝ)

-- The given conditions
def circle_eq := ∀ (x y : ℝ), x^2 + 8*x + y^2 - 6*y + c = 0
def radius_five := (∀ (h k r : ℝ), r = 5 → ∃ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2)

theorem find_c (h k r : ℝ) (r_eq : r = 5) : c = 0 :=
by {
  sorry
}

end find_c_l134_134510


namespace electronics_weight_l134_134105

-- Define the initial conditions and the solution we want to prove.
theorem electronics_weight (B C E : ℕ) (k : ℕ) 
  (h1 : B = 7 * k) 
  (h2 : C = 4 * k) 
  (h3 : E = 3 * k) 
  (h4 : (B : ℚ) / (C - 8 : ℚ) = 2 * (B : ℚ) / (C : ℚ)) :
  E = 12 := 
sorry

end electronics_weight_l134_134105


namespace part1_part2_l134_134187

section
variable (x a : ℝ)

def p (a x : ℝ) : Prop :=
  x^2 - 4*a*x + 3*a^2 < 0 ∧ a > 0

def q (x : ℝ) : Prop :=
  (x - 3) / (x - 2) ≤ 0

theorem part1 (h1 : p 1 x ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

theorem part2 (h2 : ∀ x, ¬p a x → ¬q x) : 1 < a ∧ a ≤ 2 := by
  sorry

end

end part1_part2_l134_134187


namespace curve_crosses_itself_at_point_l134_134673

theorem curve_crosses_itself_at_point :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ t₁^2 - 4 = t₂^2 - 4 ∧ t₁^3 - 6 * t₁ + 4 = t₂^3 - 6 * t₂ + 4 ∧ t₁^2 - 4 = 2 ∧ t₁^3 - 6 * t₁ + 4 = 4 :=
by 
  sorry

end curve_crosses_itself_at_point_l134_134673


namespace parallel_lines_if_and_only_if_perpendicular_lines_if_and_only_if_l134_134525

variable (m x y : ℝ)

def l1_eq : Prop := (3 - m) * x + 2 * m * y + 1 = 0
def l2_eq : Prop := 2 * m * x + 2 * y + m = 0

theorem parallel_lines_if_and_only_if : l1_eq m x y → l2_eq m x y → (m = -3/2) :=
by sorry

theorem perpendicular_lines_if_and_only_if : l1_eq m x y → l2_eq m x y → (m = 0 ∨ m = 5) :=
by sorry

end parallel_lines_if_and_only_if_perpendicular_lines_if_and_only_if_l134_134525


namespace rebus_solution_l134_134002

theorem rebus_solution :
  ∃ (A B C : ℕ), A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    (A*100 + B*10 + A) + (A*100 + B*10 + C) + (A*100 + C*10 + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by {
  sorry
}

end rebus_solution_l134_134002


namespace train_passing_platform_time_l134_134102

theorem train_passing_platform_time
  (L_train : ℝ) (L_plat : ℝ) (time_to_cross_tree : ℝ) (time_to_pass_platform : ℝ)
  (H1 : L_train = 2400) 
  (H2 : L_plat = 800)
  (H3 : time_to_cross_tree = 60) :
  time_to_pass_platform = 80 :=
by
  -- add proof here
  sorry

end train_passing_platform_time_l134_134102


namespace specific_gravity_cylinder_l134_134340

noncomputable def specific_gravity_of_cylinder (r m : ℝ) : ℝ :=
  (1 / 3) - (Real.sqrt 3 / (4 * Real.pi))

theorem specific_gravity_cylinder
  (r m : ℝ) 
  (cylinder_floats : r > 0 ∧ m > 0)
  (submersion_depth : r / 2 = r / 2) :
  specific_gravity_of_cylinder r m = 0.1955 :=
sorry

end specific_gravity_cylinder_l134_134340


namespace point_on_ellipse_l134_134212

noncomputable def ellipse_condition (P F1 F2 : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  let d1 := ((x - F1.1)^2 + (y - F1.2)^2).sqrt
  let d2 := ((x - F2.1)^2 + (y - F2.2)^2).sqrt
  x^2 + 4 * y^2 = 16 ∧ d1 = 7

theorem point_on_ellipse (P F1 F2 : ℝ × ℝ)
  (h : ellipse_condition P F1 F2) : 
  let x := P.1
  let y := P.2
  let d2 := ((x - F2.1)^2 + (y - F2.2)^2).sqrt
  d2 = 1 :=
sorry

end point_on_ellipse_l134_134212


namespace honors_students_count_l134_134201

variable {total_students : ℕ}
variable {total_girls total_boys : ℕ}
variable {honors_girls honors_boys : ℕ}

axiom class_size_constraint : total_students < 30
axiom prob_girls_honors : (honors_girls : ℝ) / total_girls = 3 / 13
axiom prob_boys_honors : (honors_boys : ℝ) / total_boys = 4 / 11
axiom total_students_eq : total_students = total_girls + total_boys
axiom honors_girls_value : honors_girls = 3
axiom honors_boys_value : honors_boys = 4

theorem honors_students_count : 
  honors_girls + honors_boys = 7 :=
by
  sorry

end honors_students_count_l134_134201


namespace seats_not_occupied_l134_134838

theorem seats_not_occupied (seats_per_row : ℕ) (rows : ℕ) (fraction_allowed : ℚ) (total_seats : ℕ) (allowed_seats_per_row : ℕ) (allowed_total : ℕ) (unoccupied_seats : ℕ) :
  seats_per_row = 8 →
  rows = 12 →
  fraction_allowed = 3 / 4 →
  total_seats = seats_per_row * rows →
  allowed_seats_per_row = seats_per_row * fraction_allowed →
  allowed_total = allowed_seats_per_row * rows →
  unoccupied_seats = total_seats - allowed_total →
  unoccupied_seats = 24 :=
by sorry

end seats_not_occupied_l134_134838


namespace f_relationship_l134_134380

noncomputable def f (x : ℝ) : ℝ := sorry -- definition of f needs to be filled in later

-- Conditions given in the problem
variable (h_diff : Differentiable ℝ f)
variable (h_gt : ∀ x: ℝ, deriv f x > f x)
variable (a : ℝ) (h_pos : a > 0)

theorem f_relationship (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_gt : ∀ x: ℝ, deriv f x > f x) (a : ℝ) (h_pos : a > 0) :
  f a > Real.exp a * f 0 :=
sorry

end f_relationship_l134_134380


namespace find_garden_perimeter_l134_134194

noncomputable def garden_perimeter (a : ℝ) (P : ℝ) : Prop :=
  a = 2 * P + 14.25 ∧ a = 90.25

theorem find_garden_perimeter :
  ∃ P : ℝ, garden_perimeter 90.25 P ∧ P = 38 :=
by
  sorry

end find_garden_perimeter_l134_134194


namespace exponent_property_l134_134919

theorem exponent_property : (-2)^2004 + 3 * (-2)^2003 = -2^2003 :=
by 
  sorry

end exponent_property_l134_134919


namespace side_length_square_eq_6_l134_134337

theorem side_length_square_eq_6
  (width length : ℝ)
  (h_width : width = 2)
  (h_length : length = 18) :
  (∃ s : ℝ, s^2 = width * length) ∧ (∀ s : ℝ, s^2 = width * length → s = 6) :=
by
  sorry

end side_length_square_eq_6_l134_134337


namespace find_k_and_shifted_function_l134_134543

noncomputable def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x + 1

theorem find_k_and_shifted_function (k : ℝ) (h : k ≠ 0) (h1 : linear_function k 1 = 3) :
  k = 2 ∧ linear_function 2 x + 2 = 2 * x + 3 :=
by
  sorry

end find_k_and_shifted_function_l134_134543


namespace ratio_w_y_l134_134606

theorem ratio_w_y (w x y z : ℚ) 
  (h1 : w / x = 5 / 4) 
  (h2 : y / z = 4 / 3)
  (h3 : z / x = 1 / 8) : 
  w / y = 15 / 2 := 
by
  sorry

end ratio_w_y_l134_134606


namespace problem_statement_l134_134159

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x
noncomputable def F (x : ℝ) : ℝ := f x - g x
noncomputable def m (x x₀ : ℝ) : ℝ := if x ≤ x₀ then f x else g x

-- Statement of the theorem
theorem problem_statement (x₀ x₁ x₂ n : ℝ) (hx₀ : x₀ ∈ Set.Ioo 1 2)
  (hF_root : F x₀ = 0)
  (hm_roots : m x₁ x₀ = n ∧ m x₂ x₀ = n ∧ 1 < x₁ ∧ x₁ < x₀ ∧ x₀ < x₂) :
  x₁ + x₂ > 2 * x₀ :=
sorry

end problem_statement_l134_134159


namespace sqrt_meaningful_l134_134722

theorem sqrt_meaningful (x : ℝ) : (2 * x - 4 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end sqrt_meaningful_l134_134722


namespace temperature_at_6_km_l134_134886

-- Define the initial conditions
def groundTemperature : ℝ := 25
def temperatureDropPerKilometer : ℝ := 5

-- Define the question which is the temperature at a height of 6 kilometers
def temperatureAtHeight (height : ℝ) : ℝ :=
  groundTemperature - temperatureDropPerKilometer * height

-- Prove that the temperature at 6 kilometers is -5 degrees Celsius
theorem temperature_at_6_km : temperatureAtHeight 6 = -5 := by
  -- Use expected proof  
  simp [temperatureAtHeight, groundTemperature, temperatureDropPerKilometer]
  sorry

end temperature_at_6_km_l134_134886


namespace sum_as_common_fraction_l134_134364

/-- The sum of 0.2 + 0.04 + 0.006 + 0.0008 + 0.00010 as a common fraction -/
theorem sum_as_common_fraction : (0.2 + 0.04 + 0.006 + 0.0008 + 0.00010) = (12345 / 160000) := by
  sorry

end sum_as_common_fraction_l134_134364


namespace minimum_value_l134_134289

theorem minimum_value (p q r s t u v w : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
    (ht : 0 < t) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) (h₁ : p * q * r * s = 16) (h₂ : t * u * v * w = 25) :
    (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 40 := 
sorry

end minimum_value_l134_134289


namespace number_of_bicycles_l134_134456

theorem number_of_bicycles (B T : ℕ) (h1 : T = 14) (h2 : 2 * B + 3 * T = 90) : B = 24 := by
  sorry

end number_of_bicycles_l134_134456


namespace solve_equation_l134_134427

theorem solve_equation (x : ℚ) :
  (x^2 + 3 * x + 4) / (x + 5) = x + 6 ↔ x = -13 / 4 :=
by
  sorry

end solve_equation_l134_134427


namespace area_of_rectangle_l134_134935

theorem area_of_rectangle (S R L B A : ℝ)
  (h1 : L = (2 / 5) * R)
  (h2 : R = S)
  (h3 : S^2 = 1600)
  (h4 : B = 10)
  (h5 : A = L * B) : 
  A = 160 := 
sorry

end area_of_rectangle_l134_134935


namespace solution_set_l134_134706

theorem solution_set (x : ℝ) : 
  1 < |x + 2| ∧ |x + 2| < 5 ↔ 
  (-7 < x ∧ x < -3) ∨ (-1 < x ∧ x < 3) := 
by 
  sorry

end solution_set_l134_134706


namespace total_games_for_18_players_l134_134792

-- Define the number of players
def num_players : ℕ := 18

-- Define the function to calculate total number of games
def total_games (n : ℕ) : ℕ := n * (n - 1) * 2

-- Theorem statement asserting the total number of games for 18 players
theorem total_games_for_18_players : total_games num_players = 612 :=
by
  -- proof goes here
  sorry

end total_games_for_18_players_l134_134792


namespace more_orange_pages_read_l134_134354

-- Define the conditions
def purple_pages_per_book : Nat := 230
def orange_pages_per_book : Nat := 510
def purple_books_read : Nat := 5
def orange_books_read : Nat := 4

-- Calculate the total pages read from purple and orange books respectively
def total_purple_pages_read : Nat := purple_pages_per_book * purple_books_read
def total_orange_pages_read : Nat := orange_pages_per_book * orange_books_read

-- State the theorem to be proved
theorem more_orange_pages_read : total_orange_pages_read - total_purple_pages_read = 890 :=
by
  -- This is where the proof steps would go, but we'll leave it as sorry to indicate the proof is not provided
  sorry

end more_orange_pages_read_l134_134354


namespace mark_peters_pond_depth_l134_134742

theorem mark_peters_pond_depth :
  let mark_depth := 19
  let peter_depth := 5
  let three_times_peter_depth := 3 * peter_depth
  mark_depth - three_times_peter_depth = 4 :=
by
  sorry

end mark_peters_pond_depth_l134_134742


namespace golu_distance_after_turning_left_l134_134394

theorem golu_distance_after_turning_left :
  ∀ (a c b : ℝ), a = 8 → c = 10 → (c ^ 2 = a ^ 2 + b ^ 2) → b = 6 :=
by
  intros a c b ha hc hpyth
  rw [ha, hc] at hpyth
  sorry

end golu_distance_after_turning_left_l134_134394


namespace power_modulo_l134_134290

theorem power_modulo {a : ℤ} : a^561 ≡ a [ZMOD 561] :=
sorry

end power_modulo_l134_134290


namespace polynomial_remainder_l134_134846

theorem polynomial_remainder (x : ℂ) :
  (x ^ 2030 + 1) % (x ^ 6 - x ^ 4 + x ^ 2 - 1) = x ^ 2 - 1 :=
by
  sorry

end polynomial_remainder_l134_134846


namespace trigonometric_comparison_l134_134897

noncomputable def a : ℝ := 2 * Real.sin (13 * Real.pi / 180) * Real.cos (13 * Real.pi / 180)
noncomputable def b : ℝ := 2 * Real.tan (76 * Real.pi / 180) / (1 + Real.tan (76 * Real.pi / 180)^2)
noncomputable def c : ℝ := Real.sqrt ((1 - Real.cos (50 * Real.pi / 180)) / 2)

theorem trigonometric_comparison : b > a ∧ a > c := by
  sorry

end trigonometric_comparison_l134_134897


namespace noah_sales_value_l134_134906

def last_month_large_sales : ℕ := 8
def last_month_small_sales : ℕ := 4
def price_large : ℕ := 60
def price_small : ℕ := 30

def this_month_large_sales : ℕ := 2 * last_month_large_sales
def this_month_small_sales : ℕ := 2 * last_month_small_sales

def this_month_large_sales_value : ℕ := this_month_large_sales * price_large
def this_month_small_sales_value : ℕ := this_month_small_sales * price_small

def this_month_total_sales : ℕ := this_month_large_sales_value + this_month_small_sales_value

theorem noah_sales_value :
  this_month_total_sales = 1200 :=
by
  sorry

end noah_sales_value_l134_134906


namespace solve_equation_l134_134430

theorem solve_equation (x : ℚ) (h : x ≠ -5) : 
  (x^2 + 3*x + 4) / (x + 5) = x + 6 ↔ x = -13 / 4 := by
  sorry

end solve_equation_l134_134430


namespace no_solution_for_ab_ba_l134_134940

theorem no_solution_for_ab_ba (a b x : ℕ)
  (ab ba : ℕ)
  (h_ab : ab = 10 * a + b)
  (h_ba : ba = 10 * b + a) :
  (ab^x - 2 = ba^x - 7) → false :=
by
  sorry

end no_solution_for_ab_ba_l134_134940


namespace students_in_both_band_and_chorus_l134_134920

-- Definitions of conditions
def total_students := 250
def band_students := 90
def chorus_students := 120
def band_or_chorus_students := 180

-- Theorem statement to prove the number of students in both band and chorus
theorem students_in_both_band_and_chorus : 
  (band_students + chorus_students - band_or_chorus_students) = 30 := 
by sorry

end students_in_both_band_and_chorus_l134_134920


namespace solve_inequality_min_value_F_l134_134857

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) - abs (x + 1)
def m := 3    -- Arbitrary constant, m + n = 7 implies n = 4
def n := 4

-- First statement: Solve the inequality f(x) ≥ (m + n)x
theorem solve_inequality (x : ℝ) : f x ≥ (m + n) * x ↔ x ≤ 0 := by
  sorry

noncomputable def F (x y : ℝ) : ℝ := max (abs (x^2 - 4 * y + m)) (abs (y^2 - 2 * x + n))

-- Second statement: Find the minimum value of F
theorem min_value_F (x y : ℝ) : (F x y) ≥ 1 ∧ (∃ x y, (F x y) = 1) := by
  sorry

end solve_inequality_min_value_F_l134_134857


namespace one_fourth_of_8_point_4_is_21_over_10_l134_134246

theorem one_fourth_of_8_point_4_is_21_over_10 : (8.4 / 4 : ℚ) = 21 / 10 := 
by
  sorry

end one_fourth_of_8_point_4_is_21_over_10_l134_134246


namespace limit_of_sequence_l134_134676

open Real Filter

theorem limit_of_sequence :
  tendsto (λ n : ℕ, ( (2 * n - 1) / (2 * n + 1) ) ^ (n + 1)) at_top (𝓝 (1 / exp 1)) :=
sorry

end limit_of_sequence_l134_134676


namespace problem_statement_l134_134513

-- Define that the function f is even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define that the function f satisfies f(x) = f(2 - x)
def satisfies_symmetry (f : ℝ → ℝ) : Prop := ∀ x, f x = f (2 - x)

-- Define that the function f is decreasing on a given interval
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Define that the function f is increasing on a given interval
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Given hypotheses and the theorem to prove. We use two statements for clarity.
theorem problem_statement (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_symmetry : satisfies_symmetry f) 
  (h_decreasing_1_2 : is_decreasing_on f 1 2) : 
  is_increasing_on f (-2) (-1) ∧ is_decreasing_on f 3 4 := 
by 
  sorry

end problem_statement_l134_134513


namespace expression_equals_5_l134_134128

def expression_value : ℤ := 8 + 15 / 3 - 2^3

theorem expression_equals_5 : expression_value = 5 :=
by
  sorry

end expression_equals_5_l134_134128


namespace sum_of_last_two_digits_l134_134779

theorem sum_of_last_two_digits (a b : ℕ) (ha: a = 6) (hb: b = 10) :
  ((a^15 + b^15) % 100) = 0 :=
by
  -- ha, hb represent conditions given.
  sorry

end sum_of_last_two_digits_l134_134779


namespace arrangement_two_girls_next_to_each_other_l134_134689

theorem arrangement_two_girls_next_to_each_other :
  let boys := 4
  let girls := 3
  in (∃ arrangements, number_of_arrangements_exactly_two_girls_next_to_each_other boys girls arrangements ∧ arrangements = 2880) :=
by
  sorry

end arrangement_two_girls_next_to_each_other_l134_134689


namespace matrices_commute_l134_134049

noncomputable def S : Finset ℕ :=
  {0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196}

theorem matrices_commute (n : ℕ) :
  ∀ (A : Matrix (Fin 2) (Fin 2) ℕ) (B : Matrix (Fin 2) (Fin 2) ℕ),
  (∀ a b c d e f g h : ℕ, a ∈ S → b ∈ S → c ∈ S → d ∈ S → e ∈ S → f ∈ S → g ∈ S → h ∈ S →
    A = ![![a, b], ![c, d]] → B = ![![e, f], ![g, h]] → n > 50432 → A.mul B = B.mul A) :=
sorry

end matrices_commute_l134_134049


namespace garden_ratio_l134_134670

-- Define the given conditions
def garden_length : ℕ := 100
def garden_perimeter : ℕ := 300

-- Problem statement: Prove the ratio of the length to the width is 2:1
theorem garden_ratio : 
  ∃ (W L : ℕ), 
    L = garden_length ∧ 
    2 * L + 2 * W = garden_perimeter ∧ 
    L / W = 2 :=
by 
  sorry

end garden_ratio_l134_134670


namespace bus_stop_time_l134_134471

/-- 
  We are given:
  speed_ns: speed of bus without stoppages (32 km/hr)
  speed_ws: speed of bus including stoppages (16 km/hr)
  
  We need to prove the bus stops for t = 30 minutes each hour.
-/
theorem bus_stop_time
  (speed_ns speed_ws: ℕ)
  (h_ns: speed_ns = 32)
  (h_ws: speed_ws = 16):
  ∃ t: ℕ, t = 30 := 
sorry

end bus_stop_time_l134_134471


namespace original_price_of_sarees_l134_134198

theorem original_price_of_sarees
  (P : ℝ)
  (h_sale_price : 0.80 * P * 0.85 = 306) :
  P = 450 :=
sorry

end original_price_of_sarees_l134_134198


namespace julia_age_correct_l134_134892

def julia_age_proof : Prop :=
  ∃ (j : ℚ) (m : ℚ), m = 15 * j ∧ m - j = 40 ∧ j = 20 / 7

theorem julia_age_correct : julia_age_proof :=
by
  sorry

end julia_age_correct_l134_134892


namespace temperature_range_l134_134060

-- Conditions: highest temperature and lowest temperature
def highest_temp : ℝ := 5
def lowest_temp : ℝ := -2
variable (t : ℝ) -- given temperature on February 1, 2018

-- Proof problem statement
theorem temperature_range : lowest_temp ≤ t ∧ t ≤ highest_temp :=
sorry

end temperature_range_l134_134060


namespace even_odd_set_equivalence_sum_measures_even_equal_odd_sum_measures_odd_sets_l134_134557

open Finset

-- Define X_n as a Finset of natural numbers {1, 2, ..., n}
noncomputable def X_n (n : ℕ) (h : n ≥ 3) : Finset ℕ := (range n).map (nat.cast ∘ (λ x, x + 1))

-- Measure function of subset of X_n
def measure (X : Finset ℕ) : ℕ :=
  X.sum id

-- Even and be sets in X_n
def is_even (X : Finset ℕ) : Prop :=
  measure X % 2 = 0

def is_odd (X : Finset ℕ) : Prop :=
  ¬(is_even X)

-- Part (a): The number of even sets equals the number of odd sets
theorem even_odd_set_equivalence (n : ℕ) (h : n ≥ 3) :
  (univ.filter is_even).card = (univ.filter is_odd).card := sorry

-- Part (b): The sum of the measures of the even sets equals the sum of the measures of the odd sets
theorem sum_measures_even_equal_odd (n : ℕ) (h : n ≥ 3) :
  (univ.filter is_even).sum measure = (univ.filter is_odd).sum measure := sorry

-- Part (c): The sum of the measures of the odd sets is (n+1 choose 2) * 2^(n-2)
theorem sum_measures_odd_sets (n : ℕ) (h : n ≥ 3) :
  (univ.filter is_odd).sum measure = nat.choose (n + 1) 2 * 2^(n - 2) := sorry

end even_odd_set_equivalence_sum_measures_even_equal_odd_sum_measures_odd_sets_l134_134557


namespace cone_height_l134_134663

theorem cone_height (V : ℝ) (h : ℝ) (r : ℝ) (vertex_angle : ℝ) 
  (H1 : V = 16384 * Real.pi)
  (H2 : vertex_angle = 90) 
  (H3 : V = (1 / 3) * Real.pi * r^2 * h)
  (H4 : h = r) : 
  h = 36.6 :=
by
  sorry

end cone_height_l134_134663


namespace remaining_macaroons_weight_l134_134408

theorem remaining_macaroons_weight (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (total_bags : ℕ) :
  (total_macaroons = 12) → 
  (weight_per_macaroon = 5) → 
  (total_bags = 4) → 
  let macaroons_per_bag := total_macaroons / total_bags in
  let weight_per_bag := macaroons_per_bag * weight_per_macaroon in
  let weight_eaten_by_steve := weight_per_bag in
  let total_weight := total_macaroons * weight_per_macaroon in
  let remaining_weight := total_weight - weight_eaten_by_steve in
  remaining_weight = 45 :=
by {
  sorry
}

end remaining_macaroons_weight_l134_134408


namespace total_population_milburg_l134_134615

def num_children : ℕ := 2987
def num_adults : ℕ := 2269

theorem total_population_milburg : num_children + num_adults = 5256 := by
  sorry

end total_population_milburg_l134_134615


namespace min_sum_of_factors_l134_134452

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 1800) : 
  a + b + c = 64 :=
sorry

end min_sum_of_factors_l134_134452


namespace fill_up_minivans_l134_134539

theorem fill_up_minivans (service_cost : ℝ) (fuel_cost_per_liter : ℝ) (total_cost : ℝ)
  (mini_van_liters : ℝ) (truck_percent_bigger : ℝ) (num_trucks : ℕ) (num_minivans : ℕ) :
  service_cost = 2.3 ∧ fuel_cost_per_liter = 0.7 ∧ total_cost = 396 ∧
  mini_van_liters = 65 ∧ truck_percent_bigger = 1.2 ∧ num_trucks = 2 →
  num_minivans = 4 :=
by
  sorry

end fill_up_minivans_l134_134539


namespace find_prices_min_cost_l134_134419

-- Definitions based on conditions
def price_difference (x y : ℕ) : Prop := x - y = 50
def total_cost (x y : ℕ) : Prop := 2 * x + 3 * y = 250
def cost_function (a : ℕ) : ℕ := 50 * a + 6000
def min_items (a : ℕ) : Prop := a ≥ 80
def total_items : ℕ := 200

-- Lean 4 statements for the proof problem
theorem find_prices (x y : ℕ) (h1 : price_difference x y) (h2 : total_cost x y) :
  (x = 80) ∧ (y = 30) :=
sorry

theorem min_cost (a : ℕ) (h1 : min_items a) :
  cost_function a ≥ 10000 :=
sorry

#check find_prices
#check min_cost

end find_prices_min_cost_l134_134419


namespace number_of_pencils_selling_price_equals_loss_l134_134577

theorem number_of_pencils_selling_price_equals_loss :
  ∀ (S C L : ℝ) (N : ℕ),
  C = 1.3333333333333333 * S →
  L = C - S →
  (S / 60) * N = L →
  N = 20 :=
by
  intros S C L N hC hL hN
  sorry

end number_of_pencils_selling_price_equals_loss_l134_134577


namespace solve_equation_l134_134433

theorem solve_equation 
  (x : ℚ)
  (h : (x^2 + 3*x + 4)/(x + 5) = x + 6) :
  x = -13/4 := 
by
  sorry

end solve_equation_l134_134433


namespace binom_identity_l134_134420

-- Definition: Combinatorial coefficient (binomial coefficient)
def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem binom_identity (n k : ℕ) (h : k ≤ n) :
  binom (n + 1) k = binom n k + binom n (k - 1) := by
  sorry

end binom_identity_l134_134420


namespace intersection_eq_l134_134291

open Set

def S : Set ℝ := { y | ∃ x : ℝ, y = 3^x }
def T : Set ℝ := { y | ∃ x : ℝ, y = x^2 + 1 }

theorem intersection_eq :
  S ∩ T = T := by
  sorry

end intersection_eq_l134_134291


namespace range_of_a_l134_134383

theorem range_of_a {a : ℝ} (h : ∃ x : ℝ, (a+2)/(x+1) = 1 ∧ x ≤ 0) :
  a ≤ -1 ∧ a ≠ -2 := 
sorry

end range_of_a_l134_134383


namespace jim_net_paycheck_l134_134181

-- Let’s state the problem conditions:
def biweekly_gross_pay : ℝ := 1120
def retirement_percentage : ℝ := 0.25
def tax_deduction : ℝ := 100

-- Define the amount deduction for the retirement account
def retirement_deduction (gross : ℝ) (percentage : ℝ) : ℝ := gross * percentage

-- Define the remaining paycheck after all deductions
def net_paycheck (gross : ℝ) (retirement : ℝ) (tax : ℝ) : ℝ :=
  gross - retirement - tax

-- The theorem to prove:
theorem jim_net_paycheck :
  net_paycheck biweekly_gross_pay (retirement_deduction biweekly_gross_pay retirement_percentage) tax_deduction = 740 :=
by
  sorry

end jim_net_paycheck_l134_134181


namespace cara_total_debt_l134_134932

def simple_interest (P R T : ℝ) : ℝ := P * R * T

theorem cara_total_debt :
  let P := 54
  let R := 0.05
  let T := 1
  let I := simple_interest P R T
  let total := P + I
  total = 56.7 :=
by
  sorry

end cara_total_debt_l134_134932


namespace range_of_m_l134_134862

-- Define the points and hyperbola condition
section ProofProblem

variables (m y₁ y₂ : ℝ)

-- Given conditions
def point_A_hyperbola : Prop := y₁ = -3 - m
def point_B_hyperbola : Prop := y₂ = (3 + m) / 2
def y1_greater_than_y2 : Prop := y₁ > y₂

-- The theorem to prove
theorem range_of_m (h1 : point_A_hyperbola m y₁) (h2 : point_B_hyperbola m y₂) (h3 : y1_greater_than_y2 y₁ y₂) : m < -3 :=
by { sorry }

end ProofProblem

end range_of_m_l134_134862


namespace ratio_men_to_women_l134_134790

theorem ratio_men_to_women (M W : ℕ) (h1 : W = M + 4) (h2 : M + W = 18) : M = 7 ∧ W = 11 :=
by
  sorry

end ratio_men_to_women_l134_134790


namespace range_of_a_l134_134374

variable {x a : ℝ}

def p (x : ℝ) := 2*x^2 - 3*x + 1 ≤ 0
def q (x : ℝ) (a : ℝ) := (x - a) * (x - a - 1) ≤ 0

theorem range_of_a (h : ¬ p x → ¬ q x a) : 0 ≤ a ∧ a ≤ 1/2 := by
  sorry

end range_of_a_l134_134374


namespace line_through_points_l134_134726

theorem line_through_points (a b : ℝ) (h₁ : 1 = a * 3 + b) (h₂ : 13 = a * 7 + b) : a - b = 11 := 
  sorry

end line_through_points_l134_134726


namespace inequality_bound_l134_134975

theorem inequality_bound (a : ℝ) (h : ∃ x : ℝ, 0 < x ∧ e^x * (x^2 - x + 1) * (a * x + 3 * a - 1) < 1) : a < 2 / 3 :=
by
  sorry

end inequality_bound_l134_134975


namespace probability_of_prime_pairs_l134_134639

def primes_in_range := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def num_pairs (n : Nat) : Nat := (n * (n - 1)) / 2

theorem probability_of_prime_pairs : (num_pairs 10 : ℚ) / (num_pairs 30) = (1 : ℚ) / 10 := by
  sorry

end probability_of_prime_pairs_l134_134639


namespace grace_apples_after_6_weeks_l134_134819

def apples_per_day_bella : ℕ := 6

def days_per_week : ℕ := 7

def fraction_apples_bella_consumes : ℚ := 1/3

def weeks : ℕ := 6

theorem grace_apples_after_6_weeks :
  let apples_per_week_bella := apples_per_day_bella * days_per_week
  let apples_per_week_grace := apples_per_week_bella / fraction_apples_bella_consumes
  let remaining_apples_week := apples_per_week_grace - apples_per_week_bella
  let total_apples := remaining_apples_week * weeks
  total_apples = 504 := by
  sorry

end grace_apples_after_6_weeks_l134_134819


namespace sum_of_digits_l134_134785

theorem sum_of_digits (a b : ℕ) (h1 : 10 * a + b + 10 * b + a = 202) (h2 : a < 10) (h3 : b < 10) :
  a + b = 12 :=
sorry

end sum_of_digits_l134_134785
