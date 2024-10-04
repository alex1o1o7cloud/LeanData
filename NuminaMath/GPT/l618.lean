import Mathlib

namespace largest_subset_size_l618_618977

-- Definition of the subset condition
def valid_subset (s : Set ℕ) : Prop :=
  ∀ (a ∈ s) (b ∈ s), a ≠ 4 * b

-- Setting the range
def range_set : Set ℕ := { n | 1 ≤ n ∧ n ≤ 150 }

-- The main theorem statement
theorem largest_subset_size : ∃ (s : Set ℕ), s ⊆ range_set ∧ valid_subset (s) ∧ #(s) = 150 :=
by
  sorry

end largest_subset_size_l618_618977


namespace intersection_equals_l618_618658

def A : Set ℝ := {x | x < 1}

def B : Set ℝ := {x | x^2 + x ≤ 6}

theorem intersection_equals : A ∩ B = {x : ℝ | -3 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_equals_l618_618658


namespace largest_initial_number_l618_618350

theorem largest_initial_number :
  ∃ n : ℕ, (∀ k : ℕ, (n % k ≠ 0 → k ∈ {2, 2, 2, 2, 3}) ∧ (n + 11 = 200)) ∧ (n = 189) :=
begin
  sorry -- Proof not required per instruction
end

end largest_initial_number_l618_618350


namespace hyperbola_eccentricity_l618_618867

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
    (h3 : ∀ x, y = 2 * x) 
    (h4 : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1) :
    (∃ e, e = sqrt 5) :=
by
  sorry

end hyperbola_eccentricity_l618_618867


namespace area_of_square_EFGH_is_144_l618_618557

-- Definitions
def largeSquareSide : ℝ := 6
def semicircleRadius : ℝ := largeSquareSide / 2

-- Condition that each side of the inner square is tangent to the semicircles
def isTangentToSemicircles (sideLength : ℝ) : Prop :=
  sideLength = largeSquareSide + 2 * semicircleRadius

-- Prove that the area of the inner square is 144
theorem area_of_square_EFGH_is_144 :
  ∃ sideLength : ℝ, isTangentToSemicircles sideLength ∧ sideLength^2 = 144 :=
by
  use 12
  split
  · -- Proof that the side length is equal to the large square side length plus two semicircle radii
    show 12 = largeSquareSide + 2 * semicircleRadius
    -- Specific imports would be needed for actual calculation, ignoring here
    sorry
  · -- Proof that the area of the square is 144
    show 12^2 = 144
    norm_num

end area_of_square_EFGH_is_144_l618_618557


namespace sale_in_third_month_l618_618523

theorem sale_in_third_month (s_1 s_2 s_4 s_5 s_6 : ℝ) (avg_sale : ℝ) (h1 : s_1 = 6435) (h2 : s_2 = 6927) (h4 : s_4 = 7230) (h5 : s_5 = 6562) (h6 : s_6 = 6191) (h_avg : avg_sale = 6700) :
  ∃ s_3 : ℝ, s_1 + s_2 + s_3 + s_4 + s_5 + s_6 = 6 * avg_sale ∧ s_3 = 6855 :=
by 
  sorry

end sale_in_third_month_l618_618523


namespace percent_decrease_is_20_l618_618080

/-- Define the original price and sale price as constants. -/
def P_original : ℕ := 100
def P_sale : ℕ := 80

/-- Define the formula for percent decrease. -/
def percent_decrease (P_original P_sale : ℕ) : ℕ :=
  ((P_original - P_sale) * 100) / P_original

/-- Prove that the percent decrease is 20%. -/
theorem percent_decrease_is_20 : percent_decrease P_original P_sale = 20 :=
by
  sorry

end percent_decrease_is_20_l618_618080


namespace option_a_equals_half_option_c_equals_half_l618_618549

theorem option_a_equals_half : 
  ( ∃ x : ℝ, x = (√2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180)) ∧ x = 1 / 2 ) := 
sorry

theorem option_c_equals_half : 
  ( ∃ y : ℝ, y = (Real.tan (22.5 * Real.pi / 180) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)) ∧ y = 1 / 2 ) := 
sorry

end option_a_equals_half_option_c_equals_half_l618_618549


namespace who_made_a_mistake_l618_618033

-- Definitions of the conditions
def at_least_four_blue_pencils (B : ℕ) : Prop := B ≥ 4
def at_least_five_green_pencils (G : ℕ) : Prop := G ≥ 5
def at_least_three_blue_and_four_green_pencils (B G : ℕ) : Prop := B ≥ 3 ∧ G ≥ 4
def at_least_four_blue_and_four_green_pencils (B G : ℕ) : Prop := B ≥ 4 ∧ G ≥ 4

-- The main theorem stating who made a mistake
theorem who_made_a_mistake (B G : ℕ) 
  (hv : at_least_four_blue_pencils B)
  (hk : at_least_five_green_pencils G)
  (hp : at_least_three_blue_and_four_green_pencils B G)
  (hm : at_least_four_blue_and_four_green_pencils B G) 
  (h_truth : (hv ∧ hk ∧ hp ∧ hm) ∨ (¬hv ∧ hk ∧ hp ∧ hm) ∨ (hv ∧ ¬hk ∧ hp ∧ hm) ∨ (hv ∧ hk ∧ ¬hp ∧ hm) ∨ (hv ∧ hk ∧ hp ∧ ¬hm))
  (h_truthful: ∑ b in [hv, hk, hp, hm], (if b then 1 else 0) = 3) : 
  hk = false := 
sorry

end who_made_a_mistake_l618_618033


namespace value_of_x_in_interval_l618_618454

theorem value_of_x_in_interval :
  (let x := 1 / Real.logb 2 3 + 1
  in x > 1 ∧ x < 2) := by
  sorry

end value_of_x_in_interval_l618_618454


namespace total_weight_l618_618913

-- Definitions for molar masses of elements
def molar_mass_N : ℝ := 14.01
def molar_mass_H : ℝ := 1.01
def molar_mass_Br : ℝ := 79.90
def molar_mass_Mg : ℝ := 24.31
def molar_mass_Cl : ℝ := 35.45

-- Definitions for the moles of compounds
def moles_NH4Br : ℝ := 3.72
def moles_MgCl2 : ℝ := 2.45

-- Theorem to be proved
theorem total_weight :
  let molar_mass_NH4Br := molar_mass_N + 4 * molar_mass_H + molar_mass_Br
      molar_mass_MgCl2 := molar_mass_Mg + 2 * molar_mass_Cl
      weight_NH4Br := moles_NH4Br * molar_mass_NH4Br
      weight_MgCl2 := moles_MgCl2 * molar_mass_MgCl2
  in weight_NH4Br + weight_MgCl2 = 597.64 := by
  sorry

end total_weight_l618_618913


namespace more_trees_in_ahmeds_orchard_l618_618132

-- Given conditions
def ahmed_orange_trees : ℕ := 8
def hassan_apple_trees : ℕ := 1
def hassan_orange_trees : ℕ := 2
def ahmed_apple_trees : ℕ := 4 * hassan_apple_trees
def ahmed_total_trees : ℕ := ahmed_orange_trees + ahmed_apple_trees
def hassan_total_trees : ℕ := hassan_apple_trees + hassan_orange_trees

-- Statement to be proven
theorem more_trees_in_ahmeds_orchard : ahmed_total_trees - hassan_total_trees = 9 :=
by
  sorry

end more_trees_in_ahmeds_orchard_l618_618132


namespace number_of_correct_statements_l618_618689

def f (x : ℝ) : ℝ := logBase (1 / 3) (2 - x) - logBase 3 (x + 4)

def st1_domain (x : ℝ) : Prop := -4 < x ∧ x < 2
def st2_even (x : ℝ) : Prop := f(x - 1) = f(-(x - 1))
def st3_decreasing (x : ℝ) : Prop := ∀ a b, -1 ≤ a ∧ a < b ∧ b < 2 → f a > f b
def st4_range (y : ℝ) : Prop := ∃ x, f x = y ∧ y ≤ -2

theorem number_of_correct_statements : 
  (∃ h1, ¬ st1_domain ∧ ∃ h2, st2_even ∧ ∃ h3, ¬ st3_decreasing ∧ ∃ h4, st4_range) → 
  ∑ n in {1, 2, 3, 4}, if st1_domain n ∨ st2_even n ∨ st3_decreasing n ∨ st4_range n then 1 else 0 = 1 := by
  sorry

end number_of_correct_statements_l618_618689


namespace smallest_positive_solution_l618_618187

noncomputable def tan_eq (x : ℝ) : Prop :=
  tan (3 * x) - sin (2 * x) = cos (2 * x)

theorem smallest_positive_solution (x : ℝ) :
  tan_eq x ↔ x = (real.pi / 4) := by
sorry

end smallest_positive_solution_l618_618187


namespace circles_intersect_at_O_l618_618799

variable {α : Type*}

structure Point (α : Type*) := 
  (x : ℝ)
  (y : ℝ)

structure Triangle (α : Type*) :=
  (A B C : Point α)

structure Circle (α : Type*) :=
  (center : Point α)
  (radius : ℝ)

def midpoint (P Q : Point α) : Point α :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

-- Let ABC be a triangle
variables (ABC : Triangle α)

-- Define midpoints
def D : Point α := midpoint ABC.A ABC.B
def E : Point α := midpoint ABC.B ABC.C
def F : Point α := midpoint ABC.C ABC.A

-- Define circles k1, k2, k3
def k1 : Circle α := { center := ABC.A, radius := (ABC.A.x - D.x)^2 + (ABC.A.y - D.y)^2 }
def k2 : Circle α := { center := ABC.B, radius := (ABC.B.x - E.x)^2 + (ABC.B.y - E.y)^2 }
def k3 : Circle α := { center := ABC.C, radius := (ABC.C.x - F.x)^2 + (ABC.C.y - F.y)^2 }

-- Assume the existence of the circumcenter O of triangle ABC
axiom circumcenter (ABC : Triangle α) : Point α

-- Prove that circumcenter O lies on circles k1, k2, and k3
theorem circles_intersect_at_O :
  let O := circumcenter ABC in
  (O.x - k1.center.x)^2 + (O.y - k1.center.y)^2 = k1.radius ∧
  (O.x - k2.center.x)^2 + (O.y - k2.center.y)^2 = k2.radius ∧
  (O.x - k3.center.x)^2 + (O.y - k3.center.y)^2 = k3.radius := 
sorry

end circles_intersect_at_O_l618_618799


namespace largest_initial_number_l618_618317

theorem largest_initial_number (n : ℕ) (h : (∃ a b c d e : ℕ, n ≠ 0 ∧ n + a + b + c + d + e = 200 
                                              ∧ n % a ≠ 0 ∧ n % b ≠ 0 ∧ n % c ≠ 0 ∧ n % d ≠ 0 ∧ n % e ≠ 0)) 
: n ≤ 189 :=
sorry

end largest_initial_number_l618_618317


namespace find_k_l618_618256

variable (k : ℝ)
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

def linear_combination1 : ℝ × ℝ := (k * (a.1) + b.1, k * (a.2) + b.2)
def linear_combination2 : ℝ × ℝ := (a.1 - 3 * b.1, a.2 - 3 * b.2)

def collinear (v w : ℝ × ℝ) : Prop := ∃ λ : ℝ, v = (λ * w.1, λ * w.2)

theorem find_k :
  collinear (linear_combination1 k) linear_combination2 ↔ k = -1/3 :=
by 
  sorry

end find_k_l618_618256


namespace part1_part2_l618_618696

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * real.exp x - real.cos x - x

-- Part 1: Prove f(x) >= 0 for all x when a = 1
theorem part1 (x : ℝ) : f 1 x ≥ 0 := by
  sorry

-- Part 2: Find the range of a such that f(x) has two extreme points in (0, π)
theorem part2 (a : ℝ) : (∃ x1 x2 ∈ set.Ioo 0 real.pi, 
  deriv (f a) x1 = 0 ∧ deriv (f a) x2 = 0 ∧ x1 ≠ x2) ↔ (0 < a ∧ a < exp (-real.pi)) := by
  sorry

end part1_part2_l618_618696


namespace least_possible_sum_24_l618_618374

noncomputable def leastSum (m n : ℕ) 
  (h1: m > 0)
  (h2: n > 0)
  (h3: Nat.gcd (m + n) 231 = 1)
  (h4: m^m % n^n = 0)
  (h5: ¬ (m % n = 0))
  : ℕ :=
  m + n

theorem least_possible_sum_24 : ∃ (m n : ℕ), 
  m > 0 ∧ 
  n > 0 ∧ 
  Nat.gcd (m + n) 231 = 1 ∧ 
  m^m % n^n = 0 ∧ 
  ¬(m % n = 0) ∧ 
  leastSum m n m_pos n_pos gcd_cond mult_cond not_mult_cond = 24 :=
begin
  sorry
end

end least_possible_sum_24_l618_618374


namespace ratio_MO_OA_l618_618823

variables {A B C D M N O : Type*}
variables [parallelogram ABCD] [midpoint M BC] [midpoint N CD]
variables [line_segment AM] [line_segment BN]
variable [intersection O AM BN]

theorem ratio_MO_OA (O : Point) : MO / OA = 1 / 4 := sorry

end ratio_MO_OA_l618_618823


namespace green_cards_count_l618_618458

theorem green_cards_count (total_cards red_frac black_frac : ℕ → ℝ)
  (h_total : total_cards = 120)
  (h_red_frac : red_frac = 2 / 5)
  (h_black_frac : black_frac = 5 / 9)
  (h_red_cards : ∀ total red_frac, red_frac * total = 48)
  (h_black_cards : ∀ remainder black_frac, black_frac * remainder = 40) :
  total_cards - (48 + 40) = 32 :=
by
  rw [h_total, h_red_frac, h_black_frac, ←h_red_cards, ←h_black_cards]
  real.smul
  refl 
  sorry

end green_cards_count_l618_618458


namespace more_trees_in_ahmeds_orchard_l618_618131

-- Given conditions
def ahmed_orange_trees : ℕ := 8
def hassan_apple_trees : ℕ := 1
def hassan_orange_trees : ℕ := 2
def ahmed_apple_trees : ℕ := 4 * hassan_apple_trees
def ahmed_total_trees : ℕ := ahmed_orange_trees + ahmed_apple_trees
def hassan_total_trees : ℕ := hassan_apple_trees + hassan_orange_trees

-- Statement to be proven
theorem more_trees_in_ahmeds_orchard : ahmed_total_trees - hassan_total_trees = 9 :=
by
  sorry

end more_trees_in_ahmeds_orchard_l618_618131


namespace max_initial_number_l618_618342

theorem max_initial_number (n : ℕ) : 
  (∃ (a b c d e : ℕ), 
    200 = n + a + b + c + d + e ∧ 
    ¬ (n % a = 0) ∧ 
    ¬ ((n + a) % b = 0) ∧ 
    ¬ ((n + a + b) % c = 0) ∧ 
    ¬ ((n + a + b + c) % d = 0) ∧ 
    ¬ ((n + a + b + c + d) % e = 0)) → 
  n ≤ 189 := 
sorry

end max_initial_number_l618_618342


namespace Gage_total_cubes_l618_618258

-- Define the conditions
def Grady.red_cubes : ℕ := 20
def Grady.blue_cubes : ℕ := 15
def Grady.red_to_Gage_ratio : ℚ := 2 / 5
def Grady.blue_to_Gage_ratio : ℚ := 1 / 3
def Gage.initial_red_cubes : ℕ := 10
def Gage.initial_blue_cubes : ℕ := 12

-- Define the theorem to be proved
theorem Gage_total_cubes : 
  let received_red_cubes := Grady.red_to_Gage_ratio * Grady.red_cubes,
      received_blue_cubes := Grady.blue_to_Gage_ratio * Grady.blue_cubes,
      total_red_cubes := Gage.initial_red_cubes + received_red_cubes,
      total_blue_cubes := Gage.initial_blue_cubes + received_blue_cubes
  in total_red_cubes + total_blue_cubes = 35 :=
by
  sorry

end Gage_total_cubes_l618_618258


namespace find_constant_value_in_formula_to_double_investment_l618_618275

-- Let's define all conditions first
def initial_investment : ℝ := 5000
def interest_rate : ℝ := 0.08  -- 8 percent interest
def time_years : ℝ := 18
def future_value : ℝ := 20000

-- Define the Rule of 72 formula for the number of years to double
def years_to_double (r : ℝ) : ℝ := 72 / r

-- Define the doubling period for the given interest rate
def doubling_period : ℝ := years_to_double interest_rate

-- Define the mathematical proof problem
theorem find_constant_value_in_formula_to_double_investment :
  doubling_period = 9 :=
sorry

end find_constant_value_in_formula_to_double_investment_l618_618275


namespace a_profit_or_loss_situation_l618_618093

theorem a_profit_or_loss_situation :
  let price_bought := 10000
  let profit_percentage := 0.10
  let loss_percentage := 0.10
  let resale_percentage := 0.90

  let price_sold_to_B := price_bought * (1 + profit_percentage)
  let price_bought_back_from_B := price_sold_to_B * (1 - loss_percentage)
  let price_sold_to_C := price_bought_back_from_B * resale_percentage

  let initial_profit := price_bought * profit_percentage
  let final_loss := price_bought_back_from_B - price_sold_to_C
  let overall_profit := initial_profit - final_loss

  overall_profit = 10 := by
  -- sorry is used to skip the proof
  simp [price_bought, profit_percentage, loss_percentage, resale_percentage, price_sold_to_B, price_bought_back_from_B, price_sold_to_C, initial_profit, final_loss, overall_profit]
  sorry

end a_profit_or_loss_situation_l618_618093


namespace kolya_mistaken_l618_618007

-- Definitions relating to the conditions
def at_least_four_blue_pencils (blue_pencils : ℕ) : Prop := blue_pencils >= 4
def at_least_five_green_pencils (green_pencils : ℕ) : Prop := green_pencils >= 5
def at_least_three_blue_pencils_and_four_green_pencils (blue_pencils green_pencils : ℕ) : Prop := blue_pencils >= 3 ∧ green_pencils >= 4
def at_least_four_blue_and_four_green_pencils (blue_pencils green_pencils : ℕ) : Prop := blue_pencils >= 4 ∧ green_pencils >= 4

-- Speaking truth conditions
variables (blue_pencils green_pencils : ℕ)
def vasya_true : Prop := at_least_four_blue_pencils blue_pencils
def kolya_true : Prop := at_least_five_green_pencils green_pencils
def petya_true : Prop := at_least_three_blue_pencils_and_four_green_pencils blue_pencils green_pencils
def misha_true : Prop := at_least_four_blue_and_four_green_pencils blue_pencils green_pencils

-- Given known information: three are true, one is false
def known_information : Prop := (vasya_true blue_pencils green_pencils ∧ petya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬kolya_true blue_pencils green_pencils)
                            ∨ (vasya_true blue_pencils green_pencils ∧ petya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ ¬misha_true blue_pencils green_pencils)
                            ∨ (vasya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬petya_true blue_pencils green_pencils)
                            ∨ (petya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬vasya_true blue_pencils green_pencils)

-- Theorem to be proved
theorem kolya_mistaken : known_information blue_pencils green_pencils → ¬kolya_true blue_pencils green_pencils :=
sorry

end kolya_mistaken_l618_618007


namespace equal_cost_at_150_miles_l618_618415

def cost_Safety (m : ℝ) := 41.95 + 0.29 * m
def cost_City (m : ℝ) := 38.95 + 0.31 * m
def cost_Metro (m : ℝ) := 44.95 + 0.27 * m

theorem equal_cost_at_150_miles (m : ℝ) :
  cost_Safety m = cost_City m ∧ cost_Safety m = cost_Metro m → m = 150 :=
by
  sorry

end equal_cost_at_150_miles_l618_618415


namespace part_I_part_II_l618_618213

-- Definition of the sequence and its sum
variable {a : ℕ → ℝ} {S : ℕ → ℝ}

-- Given condition
axiom a_condition : ∀ n : ℕ, 3 * a n = 2 * S n + n

-- Initial condition
axiom a1_condition : a 1 + 1 / 2 = 3 / 2

-- Proof problems
theorem part_I :
  ∃ r b : ℝ, ∀ n : ℕ, a (n + 1) + 1 / 2 = r * (a n + 1 / 2) :=
sorry

theorem part_II :
  ∀ n : ℕ, S n = (3^(n + 1) - 3) / 4 - n / 2 →
  T n = (3^(n + 2) - 9) / 8 - (n^2 + 4 * n) / 4 :=
sorry

end part_I_part_II_l618_618213


namespace fg_of_2_l618_618271

def f (x : ℤ) : ℤ := 3 * x + 2
def g (x : ℤ) : ℤ := (x + 1)^2

theorem fg_of_2 : f (g 2) = 29 := by
  sorry

end fg_of_2_l618_618271


namespace isabella_canadian_dollars_sum_l618_618759

def sum_of_digits (n : Nat) : Nat :=
  (n % 10) + ((n / 10) % 10)

theorem isabella_canadian_dollars_sum (d : Nat) (H: 10 * d = 7 * d + 280) : sum_of_digits d = 12 :=
by
  sorry

end isabella_canadian_dollars_sum_l618_618759


namespace orange_marbles_l618_618463

-- Definitions based on the given conditions
def total_marbles : ℕ := 24
def blue_marbles : ℕ := total_marbles / 2
def red_marbles : ℕ := 6

-- The statement to prove: the number of orange marbles is 6
theorem orange_marbles : (total_marbles - (blue_marbles + red_marbles)) = 6 := 
  by 
  sorry

end orange_marbles_l618_618463


namespace meaningful_expression_l618_618872

theorem meaningful_expression (x : ℝ) : 
  (x ≥ -1 ∧ x ≠ 1) → ∃ y : ℝ, y = (sqrt (x + 1) / (x - 1)) :=
by
  assume h
  cases h with hx1 hx2
  use (sqrt (x + 1) / (x - 1))
  sorry

end meaningful_expression_l618_618872


namespace f_is_even_f_symmetric_about_pi_f_l618_618844

noncomputable theory

def f (x : ℝ) : ℝ := cos x + (cos (2 * x) / 2) + (cos (4 * x) / 4)

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

theorem f_symmetric_about_pi : ∀ x : ℝ, f (2 * π - x) = f x := by
  sorry

theorem f'_less_than_3 : ∀ x : ℝ, (derivative f x) < 3 := by
  sorry

end f_is_even_f_symmetric_about_pi_f_l618_618844


namespace cookie_radius_l618_618420

theorem cookie_radius (x y : ℝ) :
  (x^2 + y^2 + 2 * x - 4 * y = 8) → ∃ r : ℝ, r = √13 :=
by
  intro h
  use real.sqrt 13
  sorry

end cookie_radius_l618_618420


namespace general_term_formula_l618_618877

-- Define the sequence {a_n} and the sum S_n
def a : ℕ → ℕ
| 0       := 1
| (n + 1) := 2 * (finset.range (n + 1)).sum a + 1

-- Define the sum S_n
def S : ℕ → ℕ
| n := (finset.range n).sum a

-- Prove the general term formula
theorem general_term_formula (n : ℕ) : a n = 3^n := sorry

end general_term_formula_l618_618877


namespace largest_subset_size_l618_618970

def largest_subset_with_property (s : Set ℤ) : Prop :=
  ∀ (a b : ℤ), a ∈ s → b ∈ s → (a ≠ 4 * b ∧ b ≠ 4 * a)

theorem largest_subset_size : ∃ s : Set ℤ, (∀ x : ℤ, x ∈ s → 1 ≤ x ∧ x ≤ 150) ∧ largest_subset_with_property s ∧ s.card = 120 :=
by
  sorry

end largest_subset_size_l618_618970


namespace largest_initial_number_l618_618313

theorem largest_initial_number (n : ℕ) (h : (∃ a b c d e : ℕ, n ≠ 0 ∧ n + a + b + c + d + e = 200 
                                              ∧ n % a ≠ 0 ∧ n % b ≠ 0 ∧ n % c ≠ 0 ∧ n % d ≠ 0 ∧ n % e ≠ 0)) 
: n ≤ 189 :=
sorry

end largest_initial_number_l618_618313


namespace david_moore_total_time_l618_618544

-- Given conditions
def david_work_rate := 1 / 12
def days_david_worked_alone := 6
def remaining_work_days_together := 3
def total_work := 1

-- Definition of total time taken for both to complete the job
def combined_total_time := 6

-- Proof problem statement in Lean
theorem david_moore_total_time :
  let d_work_done_alone := days_david_worked_alone * david_work_rate
  let remaining_work := total_work - d_work_done_alone
  let combined_work_rate := remaining_work / remaining_work_days_together
  let moore_work_rate := combined_work_rate - david_work_rate
  let new_combined_work_rate := david_work_rate + moore_work_rate
  total_work / new_combined_work_rate = combined_total_time := by
    sorry

end david_moore_total_time_l618_618544


namespace largest_initial_number_l618_618352

theorem largest_initial_number :
  ∃ n : ℕ, (∀ k : ℕ, (n % k ≠ 0 → k ∈ {2, 2, 2, 2, 3}) ∧ (n + 11 = 200)) ∧ (n = 189) :=
begin
  sorry -- Proof not required per instruction
end

end largest_initial_number_l618_618352


namespace arc_length_ln_correct_l618_618147

noncomputable def arc_length_ln (a b : ℝ) (h₁ : sqrt 3 ≤ a) (h₂ : b ≤ sqrt 8): ℝ :=
  let f (x : ℝ) := Real.log 7 - Real.log x
  let f' (x : ℝ) := -1 / x
  ∫ x in a..b, Real.sqrt (1 + (f' x)^2)

theorem arc_length_ln_correct : arc_length_ln (sqrt 3) (sqrt 8) (by norm_num [Real.sqrt_le_sqrt]) (by norm_num [Real.sqrt_le_sqrt]) = 1 + (1 / 2) * Real.log (3 / 2) :=
  sorry

end arc_length_ln_correct_l618_618147


namespace problem_1_min_max_problem_2_monotonic_range_problem_3_min_value_l618_618936

section math_problems

def f (a x : ℝ) : ℝ := x^2 + 2*a*x + 2

theorem problem_1_min_max (a : ℝ) (h : a = -1) :
  ∃ min max : ℝ, (∀ x ∈ Icc (-5) 5, f a x ≥ min) ∧ (∃ x ∈ Icc (-5) 5, f a x = min) ∧
  (∀ x ∈ Icc (-5) 5, f a x ≤ max) ∧ (∃ x ∈ Icc (-5) 5, f a x = max) ∧
  min = 1 ∧ max = 37 :=
sorry

theorem problem_2_monotonic_range :
  {a : ℝ | ∀ x y ∈ Icc (-5) 5, x ≤ y → f a x ≤ f a y} = {a | a ≤ -5 ∨ a ≥ 5} :=
sorry

theorem problem_3_min_value (a : ℝ) :
  (∀ x ∈ Icc (-5) 5, f a x ≥ g a) ∧ (∃ x ∈ Icc (-5) 5, f a x = g a) ∧
  ((a ≥ 5 → g a = 27 - 10 * a) ∧ (5 ≥ a ∧ a ≥ -5 → g a = 2 - a^2) ∧ (a ≤ -5 → g a = 27 + 10 * a)) :=
sorry

-- helper function for problem 3
def g (a : ℝ) : ℝ :=
  if a ≥ 5 then 27 - 10 * a
  else if 5 ≥ a ∧ a ≥ -5 then 2 - a^2
  else 27 + 10 * a

end math_problems

end problem_1_min_max_problem_2_monotonic_range_problem_3_min_value_l618_618936


namespace sin_double_angle_l618_618684

theorem sin_double_angle (α : ℝ) (h : real.tan α = -2) : 
  real.sin (2 * α) = -4 / 5 :=
sorry

end sin_double_angle_l618_618684


namespace quadratic_inequality_solution_l618_618249

theorem quadratic_inequality_solution:
  (∃ p : ℝ, ∀ x : ℝ, x^2 + p * x - 6 < 0 ↔ -3 < x ∧ x < 2) → ∃ p : ℝ, p = 1 :=
by
  intro h
  sorry

end quadratic_inequality_solution_l618_618249


namespace min_value_l618_618234

theorem min_value (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) (h_sum : x1 + x2 = 1) :
  ∃ m, (∀ x1 x2, x1 > 0 ∧ x2 > 0 ∧ x1 + x2 = 1 → (3 * x1 / x2 + 1 / (x1 * x2)) ≥ m) ∧ m = 6 :=
by
  sorry

end min_value_l618_618234


namespace book_price_l618_618394

theorem book_price (x : ℕ) : 
  9 * x ≤ 1100 ∧ 13 * x ≤ 1500 → x = 123 :=
sorry

end book_price_l618_618394


namespace calculate_rate_l618_618855

-- Definitions corresponding to the conditions in the problem
def bankers_gain (td : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  td * rate * time

-- Given values according to the problem
def BG : ℝ := 7.8
def TD : ℝ := 65
def Time : ℝ := 1
def expected_rate_percentage : ℝ := 12

-- The mathematical proof problem statement in Lean 4
theorem calculate_rate : (BG = bankers_gain TD (expected_rate_percentage / 100) Time) :=
sorry

end calculate_rate_l618_618855


namespace range_of_m_l618_618728

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + m * x + 2 * m - 3 ≥ 0) ↔ 2 ≤ m ∧ m ≤ 6 := 
by
  sorry

end range_of_m_l618_618728


namespace soccer_team_games_l618_618958

theorem soccer_team_games :
  ∃ G : ℕ, G % 2 = 0 ∧ 
           45 / 100 * 36 = 16 ∧ 
           ∀ R, R = G - 36 → (16 + 75 / 100 * R) = 62 / 100 * G ∧
           G = 84 :=
sorry

end soccer_team_games_l618_618958


namespace necessary_not_sufficient_condition_l618_618661

theorem necessary_not_sufficient_condition (m : ℝ) 
  (h : 2 < m ∧ m < 6) :
  (∃ (x y : ℝ), (x^2 / (m - 2) + y^2 / (6 - m) = 1)) ∧ (∀ m', 2 < m' ∧ m' < 6 → ∃ (x' y' : ℝ), (x'^2 / (m' - 2) + y'^2 / (6 - m') = 1) ∧ m' ≠ 4) :=
by
  sorry

end necessary_not_sufficient_condition_l618_618661


namespace find_p8_l618_618797

noncomputable def p (x : ℝ) : ℝ := sorry -- p is a monic polynomial of degree 7

def monic_degree_7 (p : ℝ → ℝ) : Prop := sorry -- p is monic polynomial of degree 7
def satisfies_conditions (p : ℝ → ℝ) : Prop :=
  p 1 = 2 ∧ p 2 = 3 ∧ p 3 = 4 ∧ p 4 = 5 ∧ p 5 = 6 ∧ p 6 = 7 ∧ p 7 = 8

theorem find_p8 (h_monic : monic_degree_7 p) (h_conditions : satisfies_conditions p) : p 8 = 5049 :=
by
  sorry

end find_p8_l618_618797


namespace find_n_l618_618726

-- Conditions
variables (n x : ℚ)
-- Provided conditions
def condition1 := n * (x - 3) = 15
def condition2 := x = 12

-- The proof goal
theorem find_n : n = 5 / 3 :=
by
  rw [condition2] at condition1
  sorry

end find_n_l618_618726


namespace circle_line_intersection_l618_618752

theorem circle_line_intersection (a : ℝ) (h : a ≠ 0) :
  (∀ t : ℝ, let x := t + 2, y := 2 * t + 3 in (x^2 + (y - a)^2 = a^2 → 2 * x - y - 1 = 0)) ∧
  (∀ t : ℝ, ∃ p : ℝ × ℝ, let x := p.1, y := p.2 in
    x = t + 2 ∧ y = 2 * t + 3 ∧ x^2 + y^2 = 2 * a * y →
    a ∈ (Set.Ioc (-∞) (1 - Real.sqrt 5) / 4 ] ∪[ (1 + Real.sqrt 5) / 4, ∞)) :=
sorry

end circle_line_intersection_l618_618752


namespace multiplication_problem_l618_618903

variables (y : ℝ)

theorem multiplication_problem (y : ℝ) : (-24 * y^3) * (5 * y^2) * (1 / (2*y)^3) = -15 * y^2 :=
begin
  sorry
end

end multiplication_problem_l618_618903


namespace area_of_triangle_LNP_l618_618531

theorem area_of_triangle_LNP (L M N O P Q : Type)
  [hexagon : regular_hexagon LMNOPQ]
  (side_length : length_side LMNOPQ = 4) :
  area_of_triangle LNP = 8 * sqrt 3 :=
sorry

end area_of_triangle_LNP_l618_618531


namespace abc_inequality_l618_618192

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) : 
    (ab / (a^5 + ab + b^5)) + (bc / (b^5 + bc + c^5)) + (ca / (c^5 + ca + a^5)) ≤ 1 := 
sorry

end abc_inequality_l618_618192


namespace muffin_count_l618_618164

theorem muffin_count (doughnuts cookies muffins : ℕ) (h1 : doughnuts = 50) (h2 : cookies = (3 * doughnuts) / 5) (h3 : muffins = (1 * doughnuts) / 5) : muffins = 10 :=
by sorry

end muffin_count_l618_618164


namespace min_value_reciprocal_sum_l618_618231

theorem min_value_reciprocal_sum (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (h_sum : x + y = 1) : 
  ∃ z, z = 4 ∧ (∀ x y, x > 0 ∧ y > 0 ∧ x + y = 1 -> z ≤ (1/x + 1/y)) :=
sorry

end min_value_reciprocal_sum_l618_618231


namespace river_current_speed_l618_618527

def motorboat_speed_still_water : ℝ := 20
def distance_between_points : ℝ := 60
def total_trip_time : ℝ := 6.25

theorem river_current_speed : ∃ v_T : ℝ, v_T = 4 ∧ 
  (distance_between_points / (motorboat_speed_still_water + v_T)) + 
  (distance_between_points / (motorboat_speed_still_water - v_T)) = total_trip_time := 
sorry

end river_current_speed_l618_618527


namespace general_term_arithmetic_sum_first_n_terms_l618_618668

variable {n : ℕ} (a : ℕ → ℤ) (S : ℕ → ℤ) (b : ℕ → ℤ) (T : ℕ → ℤ)
variable {d a1 : ℤ}

-- Definitions based on the given conditions:
def arithmetic_sequence := ∀ n, a n = a1 + (n - 1) * d
def sum_arithmetic_sequence := ∀ n, S n = n * (a 1 + a n) / 2
def geometric_sequence := ∀ n, b n = 2 ^ a n
def sum_geometric_sequence := ∀ n, T n = b 1 * (1 - 4 ^ n) / (1 - 4)

-- Specific conditions given in the problem:
axiom (h_a3 : a 3 = 5)
axiom (h_S5 : S 5 = 3 * S 3 - 2)

-- Proving the general term of the arithmetic sequence:
theorem general_term_arithmetic :
  (arithmetic_sequence a) →
  (∀ n, S n = n * (a 1 + a n) / 2) →
  (a 3 = 5) →
  (S 5 = 3 * S 3 - 2) →
  ∀ n, a n = 2 * n - 1 :=
by sorry

-- Proving the sum of the first n terms of the sequence {b_n}:
theorem sum_first_n_terms :
  (geometric_sequence b) →
  (∀ n, T n = b 1 * (1 - 4 ^ n) / (1 - 4)) →
  (∀ n, a n = 2 * n - 1) →
  ∀ n, T n = (2 / 3) * (4 ^ n - 1) :=
by sorry

end general_term_arithmetic_sum_first_n_terms_l618_618668


namespace skater_speeds_l618_618835

-- Definitions
def constant_speeds (V1 V2 : ℝ) : Prop := V1 > 0 ∧ V2 > 0
def meeting_time (L V1 V2 : ℝ) : ℝ := L / (V1 + V2)
def overtaking_time (L V1 V2 : ℝ) : ℝ := L / (abs (V1 - V2))
def frequency_relation (L V1 V2 : ℝ) : Prop := (overtaking_time L V1 V2) / (meeting_time L V1 V2) = 4

-- Theorem
theorem skater_speeds (L V1 V2 : ℝ) (h1 : constant_speeds V1 V2) 
    (h2 : frequency_relation L V1 V2) 
    (h3 : V1 = 6 ∨ V2 = 6) : V1 = 6 ∧ V2 = 3.6 ∨ V1 = 10 ∧ V2 = 6 :=
by
  sorry

end skater_speeds_l618_618835


namespace area_ratio_mn_l618_618233

theorem area_ratio_mn (ABCD : ℝ) (AB GD : ℝ) (BE CF : ℝ) (G : ℝ) :
  (BE = (1 / 3) * AB) →
  (CF = (1 / 3) * AB) →
  (ABCD = 1) →
  let area_ratio := ((AB GD) / ABCD)
  in ∃ (m n : ℕ), (area_ratio = (9 / 14)) ∧ (m + n = 23) :=
by
  intros
  let area_ratio := ((AB GD) / ABCD)
  use 9, 14
  split
  · sorry
  · rfl

end area_ratio_mn_l618_618233


namespace calories_for_breakfast_l618_618056

theorem calories_for_breakfast :
  let cake_calories := 110
  let chips_calories := 310
  let coke_calories := 215
  let lunch_calories := 780
  let daily_limit := 2500
  let remaining_calories := 525
  let total_dinner_snacks := cake_calories + chips_calories + coke_calories
  let total_lunch_dinner := total_dinner_snacks + lunch_calories
  let total_consumed := daily_limit - remaining_calories
  total_consumed - total_lunch_dinner = 560 := by
  sorry

end calories_for_breakfast_l618_618056


namespace angle_ACB_eq_90_degrees_l618_618999

theorem angle_ACB_eq_90_degrees
  (A B C D : Point)
  (h1 : grid_of_equilateral_triangles A B C)
  (h2 : extend_segment B C C D BC_eq_CD : BC = CD)
  (h3 : equal_segments AD AB : AD = AB)
  (h4 : is_median AC_of_ABD : is_median A C B D) :
  angle A C B = 90 :=
  sorry

end angle_ACB_eq_90_degrees_l618_618999


namespace total_journey_distance_l618_618492

-- Define the given constants and conditions
def journey_time : ℝ := 20 -- total time in hours
def speed1 : ℝ := 10 -- speed for the first half in km/hr
def speed2 : ℝ := 15 -- speed for the second half in km/hr

-- Define the time equations
def time_first_half (D : ℝ) : ℝ := D / 2 / speed1
def time_second_half (D : ℝ) : ℝ := D / 2 / speed2

-- The Lean statement we want to prove
theorem total_journey_distance (D : ℝ) :
  time_first_half D + time_second_half D = journey_time → D = 240 :=
by
  sorry

end total_journey_distance_l618_618492


namespace solve_for_y_l618_618848

theorem solve_for_y (y : ℝ) : 
  (\left(\frac{1}{8}\right)^{3 * y + 6} = (64)^{3 * y - 2}) → (y = -\frac{2}{9}) :=
by
  sorry

end solve_for_y_l618_618848


namespace fraction_of_work_completed_l618_618084

-- Definitions
def work_rate_x : ℚ := 1 / 14
def work_rate_y : ℚ := 1 / 20
def work_rate_z : ℚ := 1 / 25

-- Given the combined work rate and time
def combined_work_rate : ℚ := work_rate_x + work_rate_y + work_rate_z
def time_worked : ℚ := 5

-- The fraction of work completed
def fraction_work_completed : ℚ := combined_work_rate * time_worked

-- Statement to prove
theorem fraction_of_work_completed : fraction_work_completed = 113 / 140 := by
  sorry

end fraction_of_work_completed_l618_618084


namespace range_of_a_l618_618692

noncomputable def f : ℝ → ℝ :=
  λ x, if (1/2 < x ∧ x ≤ 1) then x^3 / (x + 1)
       else if (0 ≤ x ∧ x ≤ 1/2) then -1/6 * x + 1/12
       else 0

noncomputable def g (a : ℝ) (h : a > 0) : ℝ → ℝ :=
  λ x, a * sin (π / 6 * x) - a + 1

theorem range_of_a {a : ℝ} (h : a > 0) :
  (∃ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 ∧ f x1 = g a h x2) ↔ (1/2 ≤ a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l618_618692


namespace vector_b_and_k_l618_618237

noncomputable def vect_a : ℝ × ℝ × ℝ := (2, -1, 2)

theorem vector_b_and_k (λ k : ℝ) (vect_b : ℝ × ℝ × ℝ) 
  (h1 : vect_b = (λ * vect_a.1, λ * vect_a.2, λ * vect_a.3)) 
  (h2 : vect_a.1 * vect_b.1 + vect_a.2 * vect_b.2 + vect_a.3 * vect_b.3 = 18)
  (h3 : (k * vect_a.1 + vect_b.1, k * vect_a.2 + vect_b.2, k * vect_a.3 + vect_b.3) 
        .1 * (k * vect_a.1 - vect_b.1) + 
        (k * vect_a.2 + vect_b.2) * (k * vect_a.2 - vect_b.2) +
        (k * vect_a.3 + vect_b.3) * (k * vect_a.3 - vect_b.3) = 0) : 
  vect_b = (4, -2, 4) ∧ (k = 2 ∨ k = -2) := 
by
  sorry

end vector_b_and_k_l618_618237


namespace number_of_valid_pairs_l618_618109

-- Definitions based on conditions
def is_valid_pair (a b : ℕ) : Prop :=
  b > a ∧ ab = 2 * (a - 4) * (b - 4)

-- The main proof statement
theorem number_of_valid_pairs : 
  ∃ (count : ℕ), count = 3 ∧ ∀ (a b : ℕ), is_valid_pair a b → [ (a, b) = (9, 40) ∨ (a, b) = (10, 24) ∨ (a, b) = (12, 16)].length = count :=
sorry

end number_of_valid_pairs_l618_618109


namespace problem1_l618_618506

theorem problem1 {a b c : ℝ} (h : a + b + c = 2) : a^2 + b^2 + c^2 + 2 * a * b * c < 2 :=
sorry

end problem1_l618_618506


namespace gcd_1113_1897_l618_618634

theorem gcd_1113_1897 : Int.gcd 1113 1897 = 7 := by
  sorry

end gcd_1113_1897_l618_618634


namespace lasagna_package_weight_l618_618045

theorem lasagna_package_weight 
  (beef : ℕ) 
  (noodles_needed_per_beef : ℕ) 
  (current_noodles : ℕ) 
  (packages_needed : ℕ) 
  (noodles_per_package : ℕ) 
  (H1 : beef = 10)
  (H2 : noodles_needed_per_beef = 2)
  (H3 : current_noodles = 4)
  (H4 : packages_needed = 8)
  (H5 : noodles_per_package = (2 * beef - current_noodles) / packages_needed) :
  noodles_per_package = 2 := 
by
  sorry

end lasagna_package_weight_l618_618045


namespace max_rooks_no_attack_l618_618908

/-- What is the maximum number of rooks that can be placed on an 8x8 chessboard so that none of them can attack each other? --/
theorem max_rooks_no_attack : 
  ∀ (board_size : ℕ) (rook_moves : (ℕ -> ℕ -> Prop)),
    board_size = 8 →
    (∀ x y, rook_moves x y ↔ x = y ∨ x = y + 8 ∨ x + 8 = y) →
    ∃ (max_rooks : ℕ),
      (∀ r : fin board_size, r < max_rooks) →
      (∀ r₁ r₂, r₁ < max_rooks → r₂ < max_rooks → r₁ ≠ r₂ → ¬ (rook_moves r₁ r₂)) →
      max_rooks = 8 :=
begin
  sorry
end

end max_rooks_no_attack_l618_618908


namespace incorrect_quadrants_l618_618071

def inverse_proportion_function (x : ℝ) : ℝ := 6 / x

theorem incorrect_quadrants :
  ¬ (∀ x : ℝ, (x > 0 → inverse_proportion_function x < 0) ∧ (x < 0 → inverse_proportion_function x > 0)) :=
by
  sorry

end incorrect_quadrants_l618_618071


namespace shortest_distance_parabola_line_l618_618773

/--
Let A be a point on the parabola y = x^2 - 4x + 7,
and let B be a point on the line y = 2x - 3.
Prove that the shortest possible distance between points A and B is 1 / sqrt(5).
-/
theorem shortest_distance_parabola_line :
  let d (a : ℝ) := (abs (a^2 - 6 * a + 10)) / (sqrt (5)) in
  ∃ a : ℝ, d a = 1 / (sqrt (5)) :=
by
  sorry

end shortest_distance_parabola_line_l618_618773


namespace percentage_not_second_year_l618_618292

theorem percentage_not_second_year (T : ℕ) 
  (third_year_students : ℕ := 0.30 * T)
  (fraction_not_third_but_second : ℚ := 1 / 7) 
  (not_third_year_students := (1 - 0.30) * T) : 
  real.of_rat(fraction_not_third_but_second * 0.70 * T) = real.of_rat(0.10 * T) → 
  (100 - (fraction_not_third_but_second * 100 * (1 - 0.30)) = 90) :=
sorry

end percentage_not_second_year_l618_618292


namespace pairwise_sum_product_l618_618820

theorem pairwise_sum_product :
  ∃ (a b c d e : ℕ), (pairwise_sums a b c d e = {5, 8, 9, 13, 14, 14, 15, 17, 18, 23}) ∧ (a * b * c * d * e = 4752) :=
begin
  sorry
end

def pairwise_sums (a b c d e : ℕ) : Multiset ℕ :=
  [a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e]

end pairwise_sum_product_l618_618820


namespace find_a_b_find_inverse_A_l618_618780

variables {a b : ℝ}

-- Condition 1: a > 0, b > 0
def positive_a : Prop := a > 0
def positive_b : Prop := b > 0

-- Condition 2: Given matrix A and its transformation
def matrix_A : Matrix (Fin 2) (Fin 2) ℝ := ![![a, 0], ![0, b]]
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

-- Proof problem 1: Prove a = 2, b = sqrt(3)
theorem find_a_b (h1 : positive_a) (h2 : positive_b) (h3: ∀ x y, circle_eq x y → ellipse_eq (a * x) (b * y)) :
  a = 2 ∧ b = Real.sqrt 3 :=
sorry

-- Matrix inversion for A
def inverse_A : Matrix (Fin 2) (Fin 2) ℝ := ![![1/(2 : ℝ), 0], ![0, 1/(Real.sqrt 3)]]

-- Proof problem 2: 
theorem find_inverse_A : matrix_A = ![![2, 0], ![0, Real.sqrt 3]] →
  (∃ (invA : Matrix (Fin 2) (Fin 2) ℝ), matrix_A ⬝ invA = (1 : Matrix (Fin 2) (Fin 2) ℝ)) :=
begin
  intro hA,
  use inverse_A,
  simp [matrix_A, inverse_A, Matrix.mul, Matrix.one, hA],
  sorry,
end

end find_a_b_find_inverse_A_l618_618780


namespace find_angle_and_area_l618_618214

theorem find_angle_and_area
  (A B C : ℝ)
  (h_acute : 0 < A ∧ A < π/2)
  (h_acute2 : 0 < B ∧ B < π/2)
  (h_acute3 : 0 < C ∧ C < π/2)
  (cosA : ℝ := Real.cos A)
  (sinA : ℝ := Real.sin A)
  (cosB : ℝ := Real.cos B)
  (sinB : ℝ := Real.sin B)
  (cosC : ℝ := Real.cos C)
  (sinC : ℝ := Real.sin C)
  (p : ℝ × ℝ := (cosA + sinA, 2 - 2*sinA))
  (q : ℝ × ℝ := (cosA - sinA, 1 + sinA))
  (h_perp : p.1 * q.1 + p.2 * q.2 = 0)
  (AC : ℝ := 2)
  (h_sin_eq : sinA^2 + sinB^2 = sinC^2) :
  (A = π/4) ∧ (let BC := AC * Real.tan π/4 in (1 / 2) * AC * BC = 2) :=
by
  sorry

end find_angle_and_area_l618_618214


namespace fraction_position_1991_1949_l618_618503

theorem fraction_position_1991_1949 :
  ∃ (row position : ℕ), 
    ∀ (i j : ℕ), 
      (∃ k : ℕ, k = i + j - 1 ∧ k = 3939) ∧
      (∃ p : ℕ, p = j ∧ p = 1949) → 
      row = 3939 ∧ position = 1949 := 
sorry

end fraction_position_1991_1949_l618_618503


namespace gage_cube_count_l618_618260

-- Define the given conditions as constants
def Grady_red : Nat := 20
def Grady_blue : Nat := 15
def Gage_orig_red : Nat := 10
def Gage_orig_blue : Nat := 12

-- Calculate the cubes Gage received
def Gage_received_red : Nat := Grady_red * (2 / 5 : ℚ)    -- 8
def Gage_received_blue : Nat := Grady_blue * (1 / 3 : ℚ)  -- 5

-- Total cubes Gage has
def Gage_total_red : Nat := Gage_orig_red + Gage_received_red -- 18
def Gage_total_blue : Nat := Gage_orig_blue + Gage_received_blue -- 17

-- Finally
def Gage_total_cubes : Nat := Gage_total_red + Gage_total_blue -- 35

theorem gage_cube_count :
  Gage_total_cubes = 35 :=
by sorry

end gage_cube_count_l618_618260


namespace totalNumberOfGamesInSeason_l618_618853

section HighSchoolTenBasketball

def numberOfTeams : ℕ := 10
def gamesPerTeamAgainstNonConferenceOpponents : ℕ := 5

def withinConferenceGames : ℕ := (number_of_teams.choose 2) * 2
def outsideConferenceGames : ℕ := numberOfTeams * gamesPerTeamAgainstNonConferenceOpponents
def totalGames : ℕ := withinConferenceGames + outsideConferenceGames

theorem totalNumberOfGamesInSeason : totalGames = 140 := 
by
  sorry

end HighSchoolTenBasketball

end totalNumberOfGamesInSeason_l618_618853


namespace number_of_divisors_125n5_l618_618651

theorem number_of_divisors_125n5 (n : ℕ) (hn : n > 0)
  (h150 : ∀ m : ℕ, m = 150 * n ^ 4 → (∃ d : ℕ, d * (d + 1) = 150)) :
  ∃ d : ℕ, d = 125 * n ^ 5 ∧ ((13 + 1) * (5 + 1) * (5 + 1) = 504) :=
by
  sorry

end number_of_divisors_125n5_l618_618651


namespace surface_area_difference_l618_618102

def side_length_cube (volume: ℕ) := volume^(1/3 : ℚ)
def surface_area_cube (side_length: ℚ) := 6 * side_length^2

theorem surface_area_difference :
  ∀ (volume_large_cube volume_small_cube : ℕ) (num_small_cubes : ℕ),
  volume_large_cube = 64 →
  volume_small_cube = 1 →
  num_small_cubes = 64 →
  let side_length_large_cube := side_length_cube volume_large_cube in
  let side_length_small_cube := side_length_cube volume_small_cube in
  let surface_area_large_cube := surface_area_cube side_length_large_cube in
  let surface_area_small_cube := surface_area_cube side_length_small_cube in
  let total_surface_area_small_cubes := num_small_cubes * surface_area_small_cube in
  total_surface_area_small_cubes - surface_area_large_cube = 288 :=
by {
  intros volume_large_cube volume_small_cube num_small_cubes hvl hvs hns,
  let side_length_large_cube := side_length_cube volume_large_cube,
  let side_length_small_cube := side_length_cube volume_small_cube,
  let surface_area_large_cube := surface_area_cube side_length_large_cube,
  let surface_area_small_cube := surface_area_cube side_length_small_cube,
  let total_surface_area_small_cubes := num_small_cubes * surface_area_small_cube,
  sorry
}

end surface_area_difference_l618_618102


namespace yellow_tint_percentage_correct_l618_618104

-- Define initial mixture components
def initial_mixture_volume : ℝ := 40
def initial_red_tint : ℝ := 0.20 * initial_mixture_volume
def initial_yellow_tint : ℝ := 0.25 * initial_mixture_volume
def initial_water : ℝ := 0.55 * initial_mixture_volume

-- Additional quantities added
def additional_yellow_tint : ℝ := 8
def additional_water : ℝ := 2

-- Water evaporation percentage
def evaporation_percentage : ℝ := 0.05

-- Calculate new quantities
def new_yellow_tint := initial_yellow_tint + additional_yellow_tint
def new_water_before_evaporation := initial_water + additional_water
def evaporated_water := evaporation_percentage * new_water_before_evaporation
def new_water := new_water_before_evaporation - evaporated_water

-- Total volume of the new mixture
def total_new_mixture_volume := initial_red_tint + new_yellow_tint + new_water

-- Calculate the percentage of yellow tint
def yellow_tint_percentage := (new_yellow_tint / total_new_mixture_volume) * 100

-- The goal is to show that the yellow tint percentage is approximately 37%
theorem yellow_tint_percentage_correct : abs (yellow_tint_percentage - 37) < 1 :=
by
  sorry

end yellow_tint_percentage_correct_l618_618104


namespace alpha_beta_roots_l618_618257

variable (α β : ℝ)

theorem alpha_beta_roots (h1 : α^2 - 7 * α + 3 = 0) (h2 : β^2 - 7 * β + 3 = 0) (h3 : α > β) :
  α^2 + 7 * β = 46 :=
sorry

end alpha_beta_roots_l618_618257


namespace least_n_b_n_multiple_121_l618_618794

open Nat

def b : ℕ → ℕ
| 20 := 20
| (n+1) := if h : n+1 > 20 then 200 * b n - (n+1) else b 20

theorem least_n_b_n_multiple_121 : 
  ∃ n, n > 20 ∧ b n % 121 = 0 ∧ (∀ m, 20 < m < n → b m % 121 ≠ 0) :=
sorry

end least_n_b_n_multiple_121_l618_618794


namespace min_num_plays_to_obtain_subsets_l618_618359

noncomputable def num_plays_required (n : ℕ) : ℕ := 3 * n - 6

theorem min_num_plays_to_obtain_subsets (n : ℕ) (h : 2 ≤ n)
  (initial_F : Finset (Finset (Fin n))) 
  (h_init : ∀a ∈ initial_F, a.card = 1):
  ∃ F : Finset (Finset (Fin n)), 
    (∀ A B ∈ initial_F, disjoint A B → (A ∪ B) ∈ initial_F) ∧
    (∀ T : Finset (Fin n), T.card = n - 1 → T ∈ F) ∧
    F.card ≤ num_plays_required n :=
begin
  sorry
end

end min_num_plays_to_obtain_subsets_l618_618359


namespace min_links_for_weights_l618_618040

def min_links_to_break (n : ℕ) : ℕ :=
  if n = 60 then 3 else sorry

theorem min_links_for_weights (n : ℕ) (h1 : n = 60) :
  min_links_to_break n = 3 :=
by
  rw [h1]
  trivial

end min_links_for_weights_l618_618040


namespace total_games_played_l618_618493

-- Definitions
def won_percent_first_30_games (won_first_30_games : ℕ) : Prop := 
  won_first_30_games = 0.40 * 30

def won_percent_remaining_games (won_remaining_games total_remaining_games : ℕ) : Prop := 
  won_remaining_games = 0.80 * total_remaining_games

def won_percent_total_games (total_won_games total_games : ℕ) : Prop := 
  total_won_games = 0.60 * total_games

def total_games (total_first_games total_remaining_games total : ℕ) : Prop := 
  total = total_first_games + total_remaining_games

-- Problem Statement
theorem total_games_played (won_first_30_games won_remaining_games total_remaining_games total total_won_games : ℕ) 
  (h1 : won_percent_first_30_games won_first_30_games)
  (h2 : won_percent_remaining_games won_remaining_games total_remaining_games)
  (h3 : won_percent_total_games total_won_games total)
  (h4 : total_games 30 total_remaining_games total) :
  total = 60 :=
by sorry

end total_games_played_l618_618493


namespace probability_distance_less_6000_is_half_l618_618856

open List

def distances : List (Nat × Nat) := [
  (6200, 1),  -- Bangkok to Cape Town
  (7100, 0.), -- Bangkok to Honolulu
  (5800, 1),  -- Bangkok to London
  (12000, 0), -- Cape Town to Honolulu
  (6100, 0),  -- Cape Town to London
  (7500, 0)   -- Honolulu to London
]

/-- The target theorem -/
theorem probability_distance_less_6000_is_half : 
  let total_pairs := 6
  let favorable_pairs := 3
  total_pairs ≠ 0 ∧ 
  (favorable_pairs / total_pairs = 1 / 2) := 
by sorry

end probability_distance_less_6000_is_half_l618_618856


namespace largest_subset_no_member_is_4_times_another_l618_618983

-- Define the predicate that characterizes the subset
def valid_subset (S : Set ℕ) : Prop :=
  ∀ (x ∈ S) (y ∈ S), x ≠ 4 * y ∧ y ≠ 4 * x

-- Define the set of integers from 1 to 150
def range_1_to_150 : Set ℕ := {n | 1 ≤ n ∧ n ≤ 150}

-- State the theorem
theorem largest_subset_no_member_is_4_times_another :
  ∃ S ⊆ range_1_to_150, valid_subset S ∧ S.card = 140 := sorry

end largest_subset_no_member_is_4_times_another_l618_618983


namespace sabrina_sequence_expected_terms_l618_618414

noncomputable def expected_terms_in_sequence : ℕ :=
  10

theorem sabrina_sequence_expected_terms :
  ∀ (sequence : List ℕ), (∀ (n : ℕ), n ∈ sequence → n ∈ {1, 2, 3, 4}) →
  (∀ i, i < sequence.length - 1 →
    (sequence[i] + sequence[i + 1] ≠ 5)) →
  (∀ n, n ∈ {1, 2, 3, 4} → n ∈ sequence) →
  ∃ (terms : ℕ), terms = expected_terms_in_sequence :=
by
  sorry

end sabrina_sequence_expected_terms_l618_618414


namespace range_of_m_l618_618693

def f (x : ℝ) (m : ℝ) : ℝ := real.sqrt 3 * real.sin (real.pi * x / m)

def condition (x₀ m : ℝ) : Prop :=
  ∃ x₀, (∃ f' : ℝ → ℝ, ∀ x, deriv (f x m) x₀ = 0) ∧ (x₀^2 + (f x₀ m)^2 < m^2)

theorem range_of_m (m : ℝ) : condition x₀ m → m ∈ set.Ioo (-∞) (-2) ∪ set.Ioo 2 ∞ :=
begin
  sorry,
end

end range_of_m_l618_618693


namespace nabla_value_l618_618440

def nabla (a b c d : ℕ) : ℕ := a * c + b * d

theorem nabla_value : nabla 3 1 4 2 = 14 :=
by
  sorry

end nabla_value_l618_618440


namespace probability_at_least_2_out_of_3_l618_618994

/-- Define probability of a single accurate weather forecast -/
def accuracy_rate : ℝ := 0.8

/-- Define probability of a single inaccurate weather forecast -/
def inaccuracy_rate : ℝ := 1 - accuracy_rate

/-- Define the binomial coefficient for given n and k -/
def binom (n k : ℕ) : ℕ := nat.choose n k

/-- Calculate the probability of having exactly k accurate forecasts out of n days -/
def exact_k_out_of_n (n k : ℕ) (accuracy_rate inaccuracy_rate : ℝ) : ℝ :=
  (binom n k : ℝ) * (accuracy_rate^k) * (inaccuracy_rate^(n-k))

/-- Calculate the probability of having at least k accurate forecasts out of n days -/
noncomputable def at_least_k_out_of_n (n k : ℕ) (accuracy_rate inaccuracy_rate : ℝ) : ℝ :=
  ∑ i in finset.range(n + 1), if i ≥ k then exact_k_out_of_n n i accuracy_rate inaccuracy_rate else 0

/-- Proof statement: The probability of having at least 2 accurate forecasts out of 3 days
    with an accuracy rate of 0.8 is 0.896 -/
theorem probability_at_least_2_out_of_3 : at_least_k_out_of_n 3 2 accuracy_rate inaccuracy_rate = 0.896 :=
by
  sorry

end probability_at_least_2_out_of_3_l618_618994


namespace f_tan_squared_l618_618189

theorem f_tan_squared (t : ℝ) (x : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) (h0_le_t : 0 ≤ t) (h_t_le_pi_div_2 : t ≤ π / 2)
  (hf : ∀ x, x ≠ 0 → x ≠ 1 → f (x / (x - 1)) = 1 / x)
  : f (tan t ^ 2) = csc t ^ 5 * cos (2 * t) :=
sorry

end f_tan_squared_l618_618189


namespace exponent_division_l618_618578

theorem exponent_division (a : ℕ) (m n : ℕ) (h1 : 19 = a) (h2 : 11 = m) (h3 : 8 = n) : a^(m - n) = 6859 := by
  sorry

end exponent_division_l618_618578


namespace coeff_third_term_binom_expansion_l618_618423

theorem coeff_third_term_binom_expansion : 
  let T := Nat.choose 3 2 * 2^(3-2) * (x : ℕ)^2
  (T = 6) :=
by
  let T := (Nat.choose 3 2 * 2^(3-2) : ℕ)
  show T * (6 : ℕ) sorry

end coeff_third_term_binom_expansion_l618_618423


namespace largest_subset_size_l618_618972

def largest_subset_with_property (s : Set ℤ) : Prop :=
  ∀ (a b : ℤ), a ∈ s → b ∈ s → (a ≠ 4 * b ∧ b ≠ 4 * a)

theorem largest_subset_size : ∃ s : Set ℤ, (∀ x : ℤ, x ∈ s → 1 ≤ x ∧ x ≤ 150) ∧ largest_subset_with_property s ∧ s.card = 120 :=
by
  sorry

end largest_subset_size_l618_618972


namespace bob_distance_when_met_l618_618083

theorem bob_distance_when_met (d : ℝ) (rate_y : ℝ) (rate_b : ℝ) (dist : ℝ) (start_diff : ℝ) (t : ℝ)
  (hy : d = rate_y * t + rate_y * start_diff) (hb : d = dist) (rel : rate_y + rate_b = hb / t) :
  rate_b * t = 8 := sorry

end bob_distance_when_met_l618_618083


namespace max_x_satisfying_eq_l618_618063

theorem max_x_satisfying_eq (x : ℝ) (h : (sqrt (5 * x) = 3 * x)) : x ≤ 5 / 9 :=
by {
  sorry
}

end max_x_satisfying_eq_l618_618063


namespace rhombus_perimeter_is_20_l618_618747

theorem rhombus_perimeter_is_20 (y : ℝ) (h_root : y^2 - 7*y + 10 = 0) 
    (h_rhombus : ∀ (A B C D : euclidean_affine_plane), rhombus A B C D ∧ dist A B = y ∧ (diagonal_lengths A B C D).fst = 6) : 
    4 * y = 20 :=
by 
  -- proof steps will be inserted here
  sorry

end rhombus_perimeter_is_20_l618_618747


namespace growth_rate_inequality_l618_618946

theorem growth_rate_inequality (a b x : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_x_pos : x > 0) :
  x ≤ (a + b) / 2 :=
sorry

end growth_rate_inequality_l618_618946


namespace problem_solution_A_problem_solution_C_l618_618553

noncomputable def expr_A : ℝ :=
  (Real.sqrt 2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180))

noncomputable def expr_C : ℝ :=
  Real.tan (22.5 * Real.pi / 180) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)

theorem problem_solution_A :
  expr_A = 1 / 2 :=
by
  sorry

theorem problem_solution_C :
  expr_C = 1 / 2 :=
by
  sorry

end problem_solution_A_problem_solution_C_l618_618553


namespace ship_meetings_l618_618814

/-- 
On an east-west shipping lane are ten ships sailing individually. The first five from the west are sailing eastwards while the other five ships are sailing westwards. They sail at the same constant speed at all times. Whenever two ships meet, each turns around and sails in the opposite direction. 

When all ships have returned to port, how many meetings of two ships have taken place? 

Proof: The total number of meetings is 25.
-/
theorem ship_meetings (east_ships west_ships : ℕ) (h_east : east_ships = 5) (h_west : west_ships = 5) : 
  east_ships * west_ships = 25 :=
by
  rw [h_east, h_west]
  exact Mul.mul 5 5
  exact eq.refl 25

end ship_meetings_l618_618814


namespace at_most_one_real_root_l618_618280

-- Definitions based on the conditions
variable {α : Type*} {β : Type*} [linear_ordered_field α] [linear_ordered_field β]
variable (f : α → β)

-- Condition: f has an inverse function
def has_inverse (f : α → β) := ∃ g : β → α, ∀ x, g (f x) = x

-- The constant m
variable (m : β)

theorem at_most_one_real_root (h_inv : has_inverse f) : 
  ∃! x : α, f x = m ∨ ∀ x : α, f x ≠ m :=
sorry

end at_most_one_real_root_l618_618280


namespace product_of_possible_P_values_l618_618141

theorem product_of_possible_P_values : 
  ∀ {P : ℤ} (A B : ℤ) (t : ℤ), 
    (A = B + P) →
    (A - 6 = B + P - 6) →
    (B + 2 = B + 2) →
    (|((B + P - 6) - (B + 2))| = 4) →
    (P = 12 ∨ P = 4) →
    (P = 48 := 12 * 4) := sorry

end product_of_possible_P_values_l618_618141


namespace matrix_exponent_is_310_l618_618268

noncomputable def matrixC (b : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![ ![1, 3, b], ![0, 1, 5], ![1, 0, 1] ]

noncomputable def targetMatrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![ ![1, 33, 3080], ![1, 1, 65], ![1, 0, 1] ]

theorem matrix_exponent_is_310 {b m : ℝ} 
  (h : matrix.pow (matrixC b) m = targetMatrix) : b + m = 310 := 
  sorry

end matrix_exponent_is_310_l618_618268


namespace total_age_l618_618923

-- Define the ages of a, b, and c based on the conditions given
variables (a b c : ℕ)

-- Condition 1: a is two years older than b
def age_condition1 := a = b + 2

-- Condition 2: b is twice as old as c
def age_condition2 := b = 2 * c

-- Condition 3: b is 12 years old
def age_condition3 := b = 12

-- Prove that the total of the ages of a, b, and c is 32 years
theorem total_age : age_condition1 → age_condition2 → age_condition3 → a + b + c = 32 :=
by
  intros h1 h2 h3 
  -- Proof would go here
  sorry

end total_age_l618_618923


namespace monotonicity_of_f_l618_618431

noncomputable def f (a x : ℝ) : ℝ := log a (abs (x + 1))

theorem monotonicity_of_f (a : ℝ) (h : 1 < a) :
  (∀ x y ∈ set.Ioo (-1 : ℝ) 0, x < y → f a x < f a y) →
  (∀ x y ∈ set.Iio (-1 : ℝ), x < y → f a y < f a x) ∧
  (∃ c : ℝ, c ∈ set.Iio (-1 : ℝ) ∧ f a c = 0 ∧ ∀ x ∈ set.Iio c, f a x > 0 ∧ ∀ x ∈ set.Iio (-1 : ℝ), f a x < 0) :=
sorry

end monotonicity_of_f_l618_618431


namespace goods_train_passing_time_l618_618103

/-- Definition of train parameters. --/
structure Train :=
  (speed_kmph : ℝ) (length_m : ℝ)

/-- Setup the given conditions. --/
def man_train : Train := { speed_kmph := 30, length_m := 0 }
def goods_train : Train := { speed_kmph := 82, length_m := 280 }

/-- Definition of relative speed when trains are moving in opposite direction. --/
def relative_speed_kmph (t1 t2 : Train) : ℝ :=
  t1.speed_kmph + t2.speed_kmph

/-- Conversion function from kmph to m/s. --/
def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

/-- Definition of time taken for goods train to pass the man. --/
def time_to_pass (t1 t2 : Train) : ℝ :=
  let relative_speed_mps := kmph_to_mps (relative_speed_kmph t1 t2)
  t2.length_m / relative_speed_mps

/-- Theorem stating the result. --/
theorem goods_train_passing_time : time_to_pass man_train goods_train ≈ 8.993 := by
  sorry

end goods_train_passing_time_l618_618103


namespace handshakes_count_l618_618464

/-- 
There are 25 goblins and 18 elves at the Annual Forest Gathering. 
The elves do not shake hands with each other but shake hands with every goblin. 
The goblins shake hands with each other and with all elves. 
Each pair of creatures shakes hands at most once.
Prove that the total number of handshakes at the gathering is 750.
-/
theorem handshakes_count
  (num_goblins : ℕ)
  (num_elves : ℕ)
  (shake_hands_goblins : nat.choose 25 2)
  (shake_hands_elves_goblins : 18 * 25)
  (total_handshakes : shake_hands_goblins + shake_hands_elves_goblins = 750) :
  total_handshakes = 750 :=
sorry

end handshakes_count_l618_618464


namespace exponent_division_l618_618577

theorem exponent_division (a : ℕ) (m n : ℕ) (h1 : 19 = a) (h2 : 11 = m) (h3 : 8 = n) : a^(m - n) = 6859 := by
  sorry

end exponent_division_l618_618577


namespace inequality_l618_618446

-- Let's define necessary real numbers and natural numbers
variable {x : ℕ → ℝ}

-- Condition given in the problem
def condition (m n : ℕ) : Prop := |x (m + n) - x m - x n| ≤ 1

-- The theorem we aim to prove
theorem inequality (m n : ℕ) (h : ∀ (m n : ℕ), condition m n) : 
  |(x m / m) - (x n / n)| < (1 / m) + (1 / n) :=
by
  -- Proof is omitted
  sorry

end inequality_l618_618446


namespace find_p9_l618_618788

noncomputable def p : polynomial ℝ :=
sorry

lemma p_conditions (x : ℝ) :
  (p.eval 1 = 1) ∧
  (p.eval 2 = 2) ∧
  (p.eval 3 = 3) ∧
  (p.eval 4 = 4) ∧
  (p.eval 5 = 5) ∧
  (p.eval 6 = 6) ∧
  (p.eval 7 = 7) ∧
  (p.eval 8 = 8) :=
sorry

lemma p_degree (x : ℝ) :
  p.degree = 8 :=
sorry

lemma p_is_monic :
  p.monic :=
sorry

theorem find_p9 : 
  p.eval 9 = 40329 :=
sorry

end find_p9_l618_618788


namespace first_year_with_sum_of_digits_10_after_2200_l618_618090

/-- Prove that the first year after 2200 in which the sum of the digits equals 10 is 2224. -/
theorem first_year_with_sum_of_digits_10_after_2200 :
  ∃ y, y > 2200 ∧ (List.sum (y.digits 10) = 10) ∧ 
       ∀ z, (2200 < z ∧ z < y) → (List.sum (z.digits 10) ≠ 10) :=
sorry

end first_year_with_sum_of_digits_10_after_2200_l618_618090


namespace who_made_a_mistake_l618_618039

-- Definitions of the conditions
def at_least_four_blue_pencils (B : ℕ) : Prop := B ≥ 4
def at_least_five_green_pencils (G : ℕ) : Prop := G ≥ 5
def at_least_three_blue_and_four_green_pencils (B G : ℕ) : Prop := B ≥ 3 ∧ G ≥ 4
def at_least_four_blue_and_four_green_pencils (B G : ℕ) : Prop := B ≥ 4 ∧ G ≥ 4

-- The main theorem stating who made a mistake
theorem who_made_a_mistake (B G : ℕ) 
  (hv : at_least_four_blue_pencils B)
  (hk : at_least_five_green_pencils G)
  (hp : at_least_three_blue_and_four_green_pencils B G)
  (hm : at_least_four_blue_and_four_green_pencils B G) 
  (h_truth : (hv ∧ hk ∧ hp ∧ hm) ∨ (¬hv ∧ hk ∧ hp ∧ hm) ∨ (hv ∧ ¬hk ∧ hp ∧ hm) ∨ (hv ∧ hk ∧ ¬hp ∧ hm) ∨ (hv ∧ hk ∧ hp ∧ ¬hm))
  (h_truthful: ∑ b in [hv, hk, hp, hm], (if b then 1 else 0) = 3) : 
  hk = false := 
sorry

end who_made_a_mistake_l618_618039


namespace andy_max_cookies_l618_618890

-- Definitions for the problem conditions
def total_cookies := 36
def bella_eats (andy_cookies : ℕ) := 2 * andy_cookies
def charlie_eats (andy_cookies : ℕ) := andy_cookies
def consumed_cookies (andy_cookies : ℕ) := andy_cookies + bella_eats andy_cookies + charlie_eats andy_cookies

-- The statement to prove
theorem andy_max_cookies : ∃ (a : ℕ), consumed_cookies a = total_cookies ∧ a = 9 :=
by
  sorry

end andy_max_cookies_l618_618890


namespace gcd_lcm_of_240_360_l618_618632

theorem gcd_lcm_of_240_360 :
  ∃ (gcd lcm : ℕ), (gcd = Nat.gcd 240 360) ∧ (lcm = Nat.lcm 240 360) ∧ (gcd = 120) ∧ (lcm = 720) :=
by
  use Nat.gcd 240 360,
  use Nat.lcm 240 360,
  split,
  { dsimp,
    sorry },
  { dsimp,
    sorry }

end gcd_lcm_of_240_360_l618_618632


namespace carol_weight_l618_618860

variable (a c : ℝ)

-- Conditions based on the problem statement
def combined_weight : Prop := a + c = 280
def weight_difference : Prop := c - a = c / 3

theorem carol_weight (h1 : combined_weight a c) (h2 : weight_difference a c) : c = 168 :=
by
  -- Proof goes here
  sorry

end carol_weight_l618_618860


namespace tangent_line_eq_min_value_f_max_value_F_l618_618244

noncomputable theory

-- Definition of function f(x)
def f (x : ℝ) (a : ℝ) := x + a * Real.log x

-- (I) Tangent line problem 
theorem tangent_line_eq (a : ℝ) (h_a : a = 1) : ∀ x, 2 * x - f x 1 - 1 = 0 :=
by sorry

-- (II) Minimum value problem
theorem min_value_f (a : ℝ) (h1 : 1 ≤ e)
  (h2 : ∀ x, 1 ≤ x → x ≤ e → f (x) (a) = 1 ∨ f (x) (a) = -a + a * Real.log (-a) ∨ f (x) (a) = e + a) :
  true := 
by sorry

-- (III) Maximum value problem
def F (x : ℝ) := f x 2 / (x ^ 2)

theorem max_value_F (M : ℝ) (h_max : ∀ x, 1 < x ∧ x < 2 → F x < (3/2)) :
  true :=
by sorry

end tangent_line_eq_min_value_f_max_value_F_l618_618244


namespace part_a_exists_a_part_b_num_good_pairs_l618_618057

-- Define the conditions of a good pair
def good_pair (a p : ℕ) : Prop :=
  a^3 + p^3 % (a^2 - p^2) = 0 ∧ a > p

-- Part (a): there exists a natural number a such that (a, 11) is a good pair
theorem part_a_exists_a : ∃ a : ℕ, good_pair a 11 :=
sorry

-- Prime numbers less than 16
def primes_less_than_16 : List ℕ := [2, 3, 5, 7, 11, 13]

-- Part (b): number of good pairs (a, p) where p is a prime number less than 16
theorem part_b_num_good_pairs : (primes_less_than_16.filter (λ p, ∃ a : ℕ, good_pair a p)).length = 18 :=
sorry

end part_a_exists_a_part_b_num_good_pairs_l618_618057


namespace percentage_increase_from_1200_to_1680_is_40_l618_618105

theorem percentage_increase_from_1200_to_1680_is_40 :
  let initial_value := 1200
  let final_value := 1680
  let percentage_increase := ((final_value - initial_value) / initial_value) * 100
  percentage_increase = 40 := by
  let initial_value := 1200
  let final_value := 1680
  let percentage_increase := ((final_value - initial_value) / initial_value) * 100
  sorry

end percentage_increase_from_1200_to_1680_is_40_l618_618105


namespace count_neighboring_subsets_l618_618086

-- Definitions for the problem conditions
def neighboring_set (S : Set ℕ) : Prop :=
  S.card = 4 ∧ ∀ x ∈ S, (x - 1 ∈ S ∨ x + 1 ∈ S)

-- Problem statement in Lean 4
theorem count_neighboring_subsets (n : ℕ) (hn : n ≥ 2) :
  ∃ k, k = (Finset.range (n - 1)).choose 2.length ∧ 
  ∀ S : Set ℕ, neighboring_set S → S ⊆ (Finset.range n).to_Set := sorry

end count_neighboring_subsets_l618_618086


namespace planes_through_three_points_l618_618218

theorem planes_through_three_points (A B C : Point) : 
  (collinear A B C → ∃ P : Plane, ∀ Q : Plane, Q = P) ∧ 
  (¬ collinear A B C → ∃ P : Plane, ∀ Q : Plane, Q = P ∨ (∃ R : Line, A ∈ R ∧ B ∈ R ∧ C ∈ R ∧ R ⊂ Q)) := sorry

end planes_through_three_points_l618_618218


namespace marco_20_cent_coins_l618_618396

-- Define the variables and their conditions
variables (x y z : ℕ)

-- First condition: Marco has 15 coins in total
def total_coins := x + y + z = 15

-- Second condition: 59 - 3x - 2y = 28
def distinct_values := 59 - 3 * x - 2 * y = 28

-- To prove: Marco has 4 20-cent coins
theorem marco_20_cent_coins : total_coins x y z → distinct_values x y → z = 4 :=
by assumption

end marco_20_cent_coins_l618_618396


namespace sale_in_fifth_month_l618_618524

-- Define the sales for the first four months and the required sale for the sixth month
def sale_month1 : ℕ := 5124
def sale_month2 : ℕ := 5366
def sale_month3 : ℕ := 5808
def sale_month4 : ℕ := 5399
def sale_month6 : ℕ := 4579

-- Define the target average sale and number of months
def target_average_sale : ℕ := 5400
def number_of_months : ℕ := 6

-- Define the total sales calculation using the provided information
def total_sales : ℕ := target_average_sale * number_of_months
def total_sales_first_four_months : ℕ := sale_month1 + sale_month2 + sale_month3 + sale_month4

-- Prove the sale in the fifth month
theorem sale_in_fifth_month : 
  sale_month1 + sale_month2 + sale_month3 + sale_month4 + (total_sales - 
  (total_sales_first_four_months + sale_month6)) + sale_month6 = total_sales :=
by
  sorry

end sale_in_fifth_month_l618_618524


namespace conditional_probability_l618_618947

def event (Ω : Type) := set Ω
variable {Ω : Type} [probability_space Ω]

variables (A B : event Ω)

-- Conditions
axiom P_A : prob A = 1/2
axiom P_AB : prob (A ∩ B) = 1/5

-- Theorem statement
theorem conditional_probability :
  prob (A ∩ B) / prob A = 2/5 := by
  -- Import could include tools for probability, but the goal here is to show a simple probability result
  suffices P_A_ne_zero : prob A ≠ 0 by sorry

  -- Calculate the conditional probability using the provided axioms
  sorry

end conditional_probability_l618_618947


namespace belongs_A_14_l618_618807

noncomputable def a_n (n : ℕ) : ℤ := 2 * n - 1

def A (k : ℕ) : set ℤ := {a | (k * (k - 1) / 2 + 1 ≤ n ∧ n ≤ k * (k + 1) / 2) ∧ a = a_n (n + (k - 1) * k / 2)}

theorem belongs_A_14 :
  199 ∈ A 14 :=
sorry

end belongs_A_14_l618_618807


namespace inequality_proof_l618_618408

theorem inequality_proof (a b c : ℝ) : 
  1 < (a / (Real.sqrt (a^2 + b^2)) + b / (Real.sqrt (b^2 + c^2)) + 
  c / (Real.sqrt (c^2 + a^2))) ∧ 
  (a / (Real.sqrt (a^2 + b^2)) + b / (Real.sqrt (b^2 + c^2)) + 
  c / (Real.sqrt (c^2 + a^2))) ≤ (3 * Real.sqrt 2 / 2) :=
by
  sorry

end inequality_proof_l618_618408


namespace dogs_sold_l618_618140

theorem dogs_sold (cats_sold : ℕ) (h1 : cats_sold = 16) (ratio : ℕ × ℕ) (h2 : ratio = (2, 1)) : ∃ dogs_sold : ℕ, dogs_sold = 8 := by
  sorry

end dogs_sold_l618_618140


namespace exponent_division_l618_618573

theorem exponent_division : (19 ^ 11) / (19 ^ 8) = 6859 :=
by
  -- Here we assume the properties of powers and arithmetic operations
  sorry

end exponent_division_l618_618573


namespace cylinder_radius_in_cone_l618_618532

theorem cylinder_radius_in_cone (d h r : ℝ) (h_d : d = 20) (h_h : h = 24) (h_cylinder : 2 * r = r):
  r = 60 / 11 :=
by
  sorry

end cylinder_radius_in_cone_l618_618532


namespace cylinder_volume_l618_618098

def radius : ℝ := 1
def height : ℝ := 2

theorem cylinder_volume : π * radius^2 * height = 2 * π := by
  sorry

end cylinder_volume_l618_618098


namespace inequality_proof_l618_618407

theorem inequality_proof
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a^3 + b^3 + c^3 = 3) :
  (1 / (a^4 + 3) + 1 / (b^4 + 3) + 1 / (c^4 + 3) >= 3 / 4) :=
by
  sorry

end inequality_proof_l618_618407


namespace greatest_integer_expression_l618_618148

theorem greatest_integer_expression :
  let a := (5:ℕ)
  let b := (2:ℕ)
  let expr := (a ^ 98 + b ^ 104) / (a ^ 95 + b ^ 101)
  ⌊expr⌋ = 96 :=
by
  let a := (5:ℕ)
  let b := (2:ℕ)
  let expr := (a ^ 98 + b ^ 104) / (a ^ 95 + b ^ 101)
  have h : expr < 97 := sorry
  have h2 : 96 ≤ expr := sorry
  exact Nat.floor_eq_iff.mpr ⟨h2, h⟩

end greatest_integer_expression_l618_618148


namespace find_number_l618_618938

theorem find_number (x : ℝ) (h : -200 * x = 1600) : x = -8 :=
by sorry

end find_number_l618_618938


namespace expected_length_palindromic_substring_l618_618529

-- Define a random binary string of length 1000
def random_binary_string (n : ℕ) : Type := 
  vector ℕ n

-- Define a function to check if a substring is a palindrome
def is_palindrome {n : ℕ} (s : vector ℕ n) : Prop := 
  ∀ i, i < n / 2 → s[i] = s[n - i - 1]

-- Given condition
noncomputable def expected_length_of_longest_palindrome (n : ℕ) : ℝ := 23.120

-- Proof statement in Lean
theorem expected_length_palindromic_substring (s : vector ℕ 1000) (H : random_binary_string 1000) : 
  ∃ L, L ≈ 23.120 ∧ is_expected_length_of_longest_palindrome s L
:=
by 
  sorry

end expected_length_palindromic_substring_l618_618529


namespace find_f3_value_l618_618781

noncomputable def f (x : ℚ) : ℚ := (x^2 + 2*x + 1) / (4*x - 5)

theorem find_f3_value : f 3 = 16 / 7 :=
by sorry

end find_f3_value_l618_618781


namespace evens_minus_odds_equal_40_l618_618079

-- Define the sum of even integers from 2 to 80
def sum_evens : ℕ := (List.range' 2 40).sum

-- Define the sum of odd integers from 1 to 79
def sum_odds : ℕ := (List.range' 1 40).sum

-- Define the main theorem to prove
theorem evens_minus_odds_equal_40 : sum_evens - sum_odds = 40 := by
  -- Proof will go here
  sorry

end evens_minus_odds_equal_40_l618_618079


namespace Kolya_made_the_mistake_l618_618000

def pencils_in_box (blue green : ℕ) : Prop :=
  (blue ≥ 4 ∨ blue < 4) ∧ (green ≥ 4 ∨ green < 4)

def boys_statements (blue green : ℕ) : Prop :=
  (Vasya : blue ≥ 4) ∧
  (Kolya : green ≥ 5) ∧
  (Petya : blue ≥ 3 ∧ green ≥ 4) ∧
  (Misha : blue ≥ 4 ∧ green ≥ 4)

theorem Kolya_made_the_mistake:
  ∀ {blue green : ℕ},
  pencils_in_box blue green →
  boys_statements blue green →
  ∃ (Vasya_truth Petya_truth Misha_truth : Prop),
  Vasya_truth ∧ Petya_truth ∧ Misha_truth ∧ ¬ Kolya_truth :=
begin
  sorry
end

end Kolya_made_the_mistake_l618_618000


namespace star_perimeter_diff_zero_l618_618363

-- Define the equiangular convex hexagon with given perimeter
structure EquiangularHexagon :=
  (A B C D E F : ℝ)
  (perimeter_eq : A + B + C + D + E + F = 1)
  (angles_eq : ∀ (x y : ℝ), x ∈ {A, B, C, D, E, F} ∧ y ∈ {A, B, C, D, E, F} → x = y → x + y = 120)

-- The main theorem we are trying to prove
theorem star_perimeter_diff_zero (hex : EquiangularHexagon) : 
  let s := (hex.A + hex.B + hex.C + hex.D + hex.E + hex.F) * (2 / sqrt 3) in
  (s - s = 0) :=
by
  sorry

end star_perimeter_diff_zero_l618_618363


namespace bags_of_soil_needed_l618_618124

theorem bags_of_soil_needed
  (length width height : ℕ)
  (beds : ℕ)
  (volume_per_bag : ℕ)
  (h_length : length = 8)
  (h_width : width = 4)
  (h_height : height = 1)
  (h_beds : beds = 2)
  (h_volume_per_bag : volume_per_bag = 4) :
  (length * width * height * beds) / volume_per_bag = 16 :=
by
  sorry

end bags_of_soil_needed_l618_618124


namespace largest_subset_size_l618_618989

theorem largest_subset_size :
  ∃ S : Set ℕ, S ⊆ {i | 1 ≤ i ∧ i ≤ 150} ∧ 
  (∀ (a b : ℕ), a ∈ S → b ∈ S → ¬ (a = 4 * b ∨ b = 4 * a)) ∧ 
  S.card = 141 :=
sorry

end largest_subset_size_l618_618989


namespace common_sales_days_l618_618092

def july_days := Finset.range 31 -- days in July, considering days start from 0 to 30, will be used as 1 to 31

def bookstore_sales : Finset Nat := { d ∈ july_days | (d + 1) % 7 = 0 }
def shoe_store_sales : Finset Nat := { d ∈ july_days | (d - 1) % 6 = 0 }

theorem common_sales_days :
  bookstore_sales ∩ shoe_store_sales = { 13 } := sorry

end common_sales_days_l618_618092


namespace fixed_circle_for_O_A_l618_618358

-- Definitions of the elements involved.
def circle (ω : Circle) (BC : Chord) (A : VarPoint) (H : Point) (D E : Point) (O_A : Circumcenter ADE) : Prop :=
  ∃ fixed_circle : Circle, ∀ (A : VarPoint), O_A ∈ fixed_circle

-- Statement of the problem.
theorem fixed_circle_for_O_A
  (ω : Circle)
  (BC: Chord)
  (A : VarPoint)
  (H : Orthocenter)
  (D E : Point)
  (O_A : Circumcenter ADE)
  (cond1 : A ∈ major arc BC of ω)
  (cond2 : H = Orthocenter of △ABC)
  (cond3 : D ∈ AB)
  (cond4 : E ∈ AC)
  (cond5 : H is midpoint of DE)
  (cond6 : O_A is circumcenter of △ADE) :
  circle ω BC A H D E O_A := 
sorry

end fixed_circle_for_O_A_l618_618358


namespace max_initial_number_l618_618339

theorem max_initial_number (n : ℕ) : 
  (∃ (a b c d e : ℕ), 
    200 = n + a + b + c + d + e ∧ 
    ¬ (n % a = 0) ∧ 
    ¬ ((n + a) % b = 0) ∧ 
    ¬ ((n + a + b) % c = 0) ∧ 
    ¬ ((n + a + b + c) % d = 0) ∧ 
    ¬ ((n + a + b + c + d) % e = 0)) → 
  n ≤ 189 := 
sorry

end max_initial_number_l618_618339


namespace solve_for_x_l618_618930

theorem solve_for_x : ∀ (x : ℝ), (3 / 5) * x^2 = 126.15 → x = 14.5 :=
by
  intro x
  assume h : (3 / 5) * x^2 = 126.15
  -- Proof goes here
  sorry

end solve_for_x_l618_618930


namespace length_of_bridge_l618_618538

-- Define the problem conditions
def length_train : ℝ := 110 -- Length of the train in meters
def speed_kmph : ℝ := 60 -- Speed of the train in kmph

-- Convert speed from kmph to m/s
noncomputable def speed_mps : ℝ := speed_kmph * 1000 / 3600

-- Define the time taken to cross the bridge
def time_seconds : ℝ := 16.7986561075114

-- Define the total distance covered
noncomputable def total_distance : ℝ := speed_mps * time_seconds

-- Prove the length of the bridge
theorem length_of_bridge : total_distance - length_train = 170 := 
by
  -- Proof will be here
  sorry

end length_of_bridge_l618_618538


namespace largest_initial_number_l618_618314

theorem largest_initial_number (n : ℕ) (h : (∃ a b c d e : ℕ, n ≠ 0 ∧ n + a + b + c + d + e = 200 
                                              ∧ n % a ≠ 0 ∧ n % b ≠ 0 ∧ n % c ≠ 0 ∧ n % d ≠ 0 ∧ n % e ≠ 0)) 
: n ≤ 189 :=
sorry

end largest_initial_number_l618_618314


namespace sum_of_arithmetic_sequence_l618_618215

noncomputable def arithmetic_sequence_sum (a_1 d : ℝ) (n : ℕ) : ℝ :=
n * a_1 + (n * (n - 1) / 2) * d

theorem sum_of_arithmetic_sequence (a_1 d : ℝ) (p q : ℕ) (h₁ : p ≠ q) (h₂ : arithmetic_sequence_sum a_1 d p = q) (h₃ : arithmetic_sequence_sum a_1 d q = p) : 
arithmetic_sequence_sum a_1 d (p + q) = - (p + q) := sorry

end sum_of_arithmetic_sequence_l618_618215


namespace who_made_mistake_l618_618016

-- Defining conditions for the colored pencils
def has_at_least_four_blue_pencils (b : Nat) : Prop := b >= 4
def has_at_least_five_green_pencils (g : Nat) : Prop := g >= 5
def has_at_least_three_blue_four_green_pencils (b g : Nat) : Prop := b >= 3 ∧ g >= 4
def has_at_least_four_blue_four_green_pencils (b g : Nat) : Prop := b >= 4 ∧ g >= 4

-- Statement of the problem
theorem who_made_mistake (b g : Nat) (vasya kolya petya misha : Prop) :
  has_at_least_four_blue_pencils b →
  has_at_least_five_green_pencils g →
  has_at_least_three_blue_four_green_pencils b g →
  has_at_least_four_blue_four_green_pencils b g →
  (∃ T : Set Prop, {vasya, kolya, petya, misha}.Erase T = {vasya, kolya, petya, misha} ∧
    T.Card = 3) →
  (kolya ↔ ¬ g >= 5) := 
sorry

end who_made_mistake_l618_618016


namespace exponent_calculation_l618_618564

-- Define the necessary exponents and base
def base : ℕ := 19
def exp1 : ℕ := 11
def exp2 : ℕ := 8

-- Given condition
lemma power_property (a : ℕ) (m n : ℕ) : a^m / a^n = a^(m - n) := by sorry

-- Proof that 19^11 / 19^8 = 6859
theorem exponent_calculation : base^exp1 / base^exp2 = 6859 := by
  have : base^exp1 / base^exp2 = base^(exp1 - exp2) := power_property base exp1 exp2
  have : base^(exp1 - exp2) = base^3 := by rw [nat.sub_eq_iff_eq_add.mpr rfl]
  have : base^3 = 6859 := by -- This would be an arithmetic computation
    rfl -- Placeholder for the actual arithmetic; ideally, you'd verify this step.
  sorry

end exponent_calculation_l618_618564


namespace factorization_l618_618619

theorem factorization (a : ℝ) : 2 * a^2 - 2 * a + 1/2 = 2 * (a - 1/2)^2 :=
by
  sorry

end factorization_l618_618619


namespace digits_in_2_pow_120_l618_618180

theorem digits_in_2_pow_120 {a b : ℕ} (h : 10^a ≤ 2^200 ∧ 2^200 < 10^b) (ha : a = 60) (hb : b = 61) : 
  ∃ n : ℕ, 10^(n-1) ≤ 2^120 ∧ 2^120 < 10^n ∧ n = 37 :=
by {
  sorry
}

end digits_in_2_pow_120_l618_618180


namespace geometric_configuration_l618_618468
open Real

variables {r R p : ℝ}

/-- Geometric configuration constraints -/
def geom_constraints (r R p : ℝ) : Prop :=
  (p^2 / (4 * (p + 1))) < (r / R) ∧ (r / R) < (p^2 / (2 * (p + 1)))

/-- Length of segment BC -/
def length_BC (r R p : ℝ) : ℝ :=
  (p / (p + 1)) * sqrt (4 * (p + 1) * R * r - p^2 * R^2)

theorem geometric_configuration (r R p : ℝ) (h1 : geom_constraints r R p) :
  (BC : ℝ) = length_BC r R p :=
sorry

end geometric_configuration_l618_618468


namespace gcd_of_numbers_l618_618062

theorem gcd_of_numbers :
  let a := 125^2 + 235^2 + 349^2
  let b := 124^2 + 234^2 + 350^2
  gcd a b = 1 := by
  sorry

end gcd_of_numbers_l618_618062


namespace boundary_function_area_of_metal_piece_elbow_pipe_6_segments_elbow_pipe_8_segments_l618_618926

-- Define Part (a) conditions and proofs
variables (c d r i : ℝ)

-- Part (a) - Boundary function
theorem boundary_function :
  ∀ (c d r i : ℝ), 
  (c ≥ d) → 
  (r > 0) → 
  z = (c + d) / 2 - (c - d) / 2 * Real.cos(i / r) :=
sorry

-- Part (a) - Area of the metal piece
theorem area_of_metal_piece :
  ∀ (c d r : ℝ), 
  (c ≥ d) → 
  (r > 0) → 
  area = π * r * (c + d) :=
sorry

-- Define Part (b) conditions and proofs
variables (r e : ℝ)

-- Part (b) - Surface area for 6-segment elbow pipe
theorem elbow_pipe_6_segments :
  ∀ (r e : ℝ), 
  (r > 0) → 
  (e > 0) → 
  surface_area = 10.1 * r * e :=
sorry

-- Part (b) - Surface area for 8-segment elbow pipe
theorem elbow_pipe_8_segments :
  ∀ (r e : ℝ), 
  (r > 0) → 
  (e > 0) → 
  surface_area = 10.0 * r * e :=
sorry

end boundary_function_area_of_metal_piece_elbow_pipe_6_segments_elbow_pipe_8_segments_l618_618926


namespace triangle_inequality_l618_618993

theorem triangle_inequality (x : ℕ) (hx : x > 0) :
  (x ≥ 34) ↔ (x + (10 + x) > 24) ∧ (x + 24 > 10 + x) ∧ ((10 + x) + 24 > x) := by
  sorry

end triangle_inequality_l618_618993


namespace cost_of_adult_ticket_l618_618466

theorem cost_of_adult_ticket
  (A : ℝ) -- Cost of an adult ticket in dollars
  (x y : ℝ) -- Number of children tickets and number of adult tickets respectively
  (hx : x = 90) -- Condition: number of children tickets sold
  (hSum : x + y = 130) -- Condition: total number of tickets sold
  (hTotal : 4 * x + A * y = 840) -- Condition: total receipts from all tickets
  : A = 12 := 
by
  -- Proof is skipped as per instruction
  sorry

end cost_of_adult_ticket_l618_618466


namespace circumference_proof_l618_618127

-- Assume points A, B, and D are in a 2D plane and C is the midpoint of AB
noncomputable def C : ℝ × ℝ := (4, 0) -- midpoint of A(−4, 0) and B(4, 0)
noncomputable def D : ℝ × ℝ := (4, 2.2) -- perpendicular from C

-- Lengths are given as conditions
def AB := 8.0 -- length of AB
def CD := 2.2 -- length of CD

-- The radius of the circumcircle
noncomputable def r : ℝ := 4.736 

-- The circumference of the circle is to be proven
def circumference : ℝ := 2 * Real.pi * r

theorem circumference_proof : circumference = 29.759 :=
by
  sorry

end circumference_proof_l618_618127


namespace points_on_hyperbola_order_l618_618682

theorem points_on_hyperbola_order (k a b c : ℝ) (hk : k > 0)
  (h₁ : a = k / -2)
  (h₂ : b = k / 2)
  (h₃ : c = k / 3) :
  a < c ∧ c < b := 
sorry

end points_on_hyperbola_order_l618_618682


namespace find_a_and_t_l618_618224

-- Definitions from conditions
def condition1 := sqrt (2 + 2 / 3) = 2 * sqrt (2 / 3)
def condition2 := sqrt (3 + 3 / 8) = 3 * sqrt (3 / 8)
def condition3 := sqrt (4 + 4 / 15) = 4 * sqrt (4 / 15)
def general_condition (n m : ℝ) := sqrt (n + n / m) = n * sqrt (n / m)

-- Defining the main statement
theorem find_a_and_t (a t : ℝ) (h : sqrt (6 + a / t) = 6 * sqrt (a / t)) : t + a = 41 :=
by 
  sorry

end find_a_and_t_l618_618224


namespace exponent_division_l618_618574

theorem exponent_division (a : ℕ) (m n : ℕ) (h1 : 19 = a) (h2 : 11 = m) (h3 : 8 = n) : a^(m - n) = 6859 := by
  sorry

end exponent_division_l618_618574


namespace exists_line_intersecting_segment_l618_618495

theorem exists_line_intersecting_segment (ABCD : square) (M : convex_polygon) 
  (area_M : area M > 1/2) (length_ABCD : side_length ABCD = 1) :
  ∃ l : line, is_parallel l ABCD.side ∧ (∃ seg : segment, intersects l seg ∧ length seg > 1/2) :=
sorry

end exists_line_intersecting_segment_l618_618495


namespace find_roots_of_polynomial_l618_618631

theorem find_roots_of_polynomial :
  ∀ x : ℝ, (3 * x ^ 4 - x ^ 3 - 8 * x ^ 2 - x + 3 = 0) →
    (x = 2 ∨ x = 1/3 ∨ x = -1) :=
by
  intros x h
  sorry

end find_roots_of_polynomial_l618_618631


namespace equation_of_ellipse_value_of_k_l618_618216

noncomputable def a := 2
noncomputable def e := (Real.sqrt 2) / 2
noncomputable def b := Real.sqrt (a^2 - e^2 * a^2)
def ellipse_eq (x y : ℝ) : Prop := (x^2) / (a^2) + (y^2) / (b^2) = 1
def line_eq (k x : ℝ) : ℝ := k * (x - 1)

theorem equation_of_ellipse :
  (a > 0) → (b > 0) → (a > b) →
  e = (Real.sqrt 2) / 2 → 
  ∀ x y, ellipse_eq x y = ((x^2) / 4 + (y^2) / 2 = 1) :=
by
  intros ha hb hab he 
  have h_eq : ellipse_eq = λ x y : ℝ, (x^2) / 4 + (y^2) / 2 = 1 := sorry
  -- Detailed proof omitted
  exact h_eq

theorem value_of_k :
  ∀ k,
    let d := (abs k) / (Real.sqrt (1 + k^2)) in
    let area := (abs k) * (Real.sqrt (4 + 6 * k^2)) / (1 + 2 * k^2) in
    area = (Real.sqrt 10) / 3 → k = 1 ∨ k = -1 :=
by
  intros k d area h_area
  -- Detailed proof omitted
  have key_lemma : 
    (abs k) * (Real.sqrt (4 + 6 * k^2)) / (1 + 2 * k^2) = (Real.sqrt 10) / 3 ↔ k = 1 ∨ k = -1 := sorry
  exact key_lemma.mpr h_area

end equation_of_ellipse_value_of_k_l618_618216


namespace total_sum_first_week_is_approximately_413_33_l618_618512

-- Define the ratios in the first week
def share_A : ℝ := 1
def share_B : ℝ := 0.75
def share_C : ℝ := 0.60
def share_D : ℝ := 0.45
def share_E : ℝ := 0.30

-- Define the changes in the ratios over weeks
def decrement_B_C_D : ℝ := 0.05
def increment_E : ℝ := 0.15

-- Define the shares after 5 weeks
def share_B_5th_week := share_B - 4 * decrement_B_C_D
def share_C_5th_week := share_C - 4 * decrement_B_C_D
def share_D_5th_week := share_D - 4 * decrement_B_C_D
def share_E_5th_week := share_E + 4 * increment_E

-- Given E's share in the 5th week
def E_share_5th_week_money : ℝ := 120

-- Calculate the value of one part in the ratio
def one_part_value := E_share_5th_week_money / share_E_5th_week

-- Total parts in the first week
def total_parts_first_week := share_A + share_B + share_C + share_D + share_E

-- Total sum of money in the first week
def total_sum_first_week := total_parts_first_week * one_part_value

theorem total_sum_first_week_is_approximately_413_33 :
  abs (total_sum_first_week - 413.33) < 1 :=
by
  sorry

end total_sum_first_week_is_approximately_413_33_l618_618512


namespace exponent_division_l618_618571

theorem exponent_division : (19 ^ 11) / (19 ^ 8) = 6859 :=
by
  -- Here we assume the properties of powers and arithmetic operations
  sorry

end exponent_division_l618_618571


namespace gage_cube_count_l618_618261

-- Define the given conditions as constants
def Grady_red : Nat := 20
def Grady_blue : Nat := 15
def Gage_orig_red : Nat := 10
def Gage_orig_blue : Nat := 12

-- Calculate the cubes Gage received
def Gage_received_red : Nat := Grady_red * (2 / 5 : ℚ)    -- 8
def Gage_received_blue : Nat := Grady_blue * (1 / 3 : ℚ)  -- 5

-- Total cubes Gage has
def Gage_total_red : Nat := Gage_orig_red + Gage_received_red -- 18
def Gage_total_blue : Nat := Gage_orig_blue + Gage_received_blue -- 17

-- Finally
def Gage_total_cubes : Nat := Gage_total_red + Gage_total_blue -- 35

theorem gage_cube_count :
  Gage_total_cubes = 35 :=
by sorry

end gage_cube_count_l618_618261


namespace trigonometric_identity_l618_618586

noncomputable def trigonometric_identity_proof : Prop :=
  let cos_30 := Real.sqrt 3 / 2;
  let sin_60 := Real.sqrt 3 / 2;
  let sin_30 := 1 / 2;
  let cos_60 := 1 / 2;
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 1

theorem trigonometric_identity : trigonometric_identity_proof :=
  sorry

end trigonometric_identity_l618_618586


namespace simple_interest_correct_l618_618909
variable (P R T SI : ℝ)
#check 780,  -- check for correctness of literals
#check 4.166666666666667 / 100,   -- representation correctness
#check 780 * (4.166666666666667 / 100) * 4,  -- intermediate computation
noncomputable def principal : ℝ := 780
noncomputable def rate : ℝ := 4.166666666666667 / 100
noncomputable def time : ℝ := 4
noncomputable def simple_interest (P R T : ℝ) : ℝ := P * R * T

theorem simple_interest_correct : 
  simple_interest principal rate time = 130 := 
by
  unfold simple_interest principal rate time
  -- Use norm_num to calculate the product directly
  norm_num

end simple_interest_correct_l618_618909


namespace length_segment_MN_l618_618436

open Real

noncomputable def line (x : ℝ) : ℝ := x + 2

def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 5

theorem length_segment_MN :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
    on_circle x₁ y₁ →
    on_circle x₂ y₂ →
    (line x₁ = y₁ ∧ line x₂ = y₂) →
    dist (x₁, y₁) (x₂, y₂) = 2 * sqrt 3 :=
by
  sorry

end length_segment_MN_l618_618436


namespace max_initial_number_l618_618330

noncomputable def verify_addition (n x : ℕ) : Prop := 
  ∀ (a : ℕ), (a ∣ n) → (a ≠ 1) → (n + a = x) → False

theorem max_initial_number :
  ∃ (n : ℕ), 
  (∀ (a1 a2 a3 a4 a5 : ℕ), 
    verify_addition n a1 ∧ verify_addition (n + a1) a2 ∧
    verify_addition (n + a1 + a2) a3 ∧ verify_addition (n + a1 + a2 + a3) a4 ∧
    verify_addition (n + a1 + a2 + a3 + a4) a5 ∧
    (n + a1 + a2 + a3 + a4 + a5 = 200)) ∧
  (∀ m : ℕ, 
    (∃ (a1 a2 a3 a4 a5 : ℕ), 
      verify_addition m a1 ∧ verify_addition (m + a1) a2 ∧
      verify_addition (m + a1 + a2) a3 ∧ verify_addition (m + a1 + a2 + a3) a4 ∧
      verify_addition (m + a1 + a2 + a3 + a4) a5 ∧
      (m + a1 + a2 + a3 + a4 + a5 = 200)) →
    m ≤ 189)
: ∃ n, n = 189 := by
  sorry

end max_initial_number_l618_618330


namespace geometric_sum_equals_fraction_l618_618878

theorem geometric_sum_equals_fraction (n : ℕ) (a r : ℝ) 
  (h_a : a = 1) (h_r : r = 1 / 2) 
  (h_sum : a * (1 - r^n) / (1 - r) = 511 / 512) : 
  n = 9 := 
by 
  sorry

end geometric_sum_equals_fraction_l618_618878


namespace largest_subset_size_l618_618990

theorem largest_subset_size :
  ∃ S : Set ℕ, S ⊆ {i | 1 ≤ i ∧ i ≤ 150} ∧ 
  (∀ (a b : ℕ), a ∈ S → b ∈ S → ¬ (a = 4 * b ∨ b = 4 * a)) ∧ 
  S.card = 141 :=
sorry

end largest_subset_size_l618_618990


namespace probability_at_least_five_stay_l618_618725

-- Definitions based on conditions
def all_people : ℕ := 6
def sure_to_stay : ℕ := 3
def unsure_to_stay_probability : ℚ := 2 / 5

-- Statement for the probability computation
theorem probability_at_least_five_stay :
  (Probability (event_at_least_five all_people sure_to_stay unsure_to_stay_probability) = 44 / 125) :=
sorry

end probability_at_least_five_stay_l618_618725


namespace total_time_to_row_l618_618929

theorem total_time_to_row (boat_speed_in_still_water : ℝ) (stream_speed : ℝ) (distance : ℝ) :
  boat_speed_in_still_water = 9 → stream_speed = 1.5 → distance = 105 → 
  (distance / (boat_speed_in_still_water + stream_speed)) + (distance / (boat_speed_in_still_water - stream_speed)) = 24 :=
by
  intro h_boat_speed h_stream_speed h_distance
  rw [h_boat_speed, h_stream_speed, h_distance]
  sorry

end total_time_to_row_l618_618929


namespace certain_event_l618_618997

def event_A : Prop := "It will rain after thunder"
def event_B : Prop := "Tomorrow will be sunny"
def event_C : Prop := "1 hour equals 60 minutes"
def event_D : Prop := "There will be a rainbow after the rain"

theorem certain_event : event_C = "1 hour equals 60 minutes" := 
by
  sorry

end certain_event_l618_618997


namespace sum_of_arithmetic_sequence_l618_618294

noncomputable def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h_seq : arithmetic_sequence a)
  (h_a5 : a 5 = 3)
  (h_a7 : a 7 = 7) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
  sorry

end sum_of_arithmetic_sequence_l618_618294


namespace Kolya_mistake_l618_618023

def boys := ["Vasya", "Kolya", "Petya", "Misha"]

constant num_blue_pencils : ℕ
constant num_green_pencils : ℕ

axiom Vasya_statement : num_blue_pencils >= 4
axiom Kolya_statement : num_green_pencils >= 5
axiom Petya_statement : num_blue_pencils >= 3 ∧ num_green_pencils >= 4
axiom Misha_statement : num_blue_pencils >= 4 ∧ num_green_pencils >= 4

axiom three_truths_one_mistake : 
  (Vasya_statement ∨ ¬Vasya_statement) ∧
  (Kolya_statement ∨ ¬Kolya_statement) ∧
  (Petya_statement ∨ ¬Petya_statement) ∧
  (Misha_statement ∨ ¬Misha_statement) ∧
  ((Vasya_statement ? true : 1) + 
   (Kolya_statement ? true : 1) + 
   (Petya_statement ? true : 1) +
   (Misha_statement ? true : 1) == 3)

theorem Kolya_mistake : ¬Kolya_statement :=
by
  sorry

end Kolya_mistake_l618_618023


namespace all_numbers_turn_into_power_of_two_l618_618671

def valid_transformation (n X : ℕ) :=
  ∃ (sequence : list ℕ → list ℕ) (steps : ℕ),
  ∀ S, S = list.range' 1 n →
  (∀ k, k < steps → ∃ x y ∈ S, (sequence ((x + y)::(abs (x - y))::(S.erase x).erase y) = sequence (S.erase x).erase y)) →
  (sequence (list.range' 1 n)).all (λ z, z = X)

theorem all_numbers_turn_into_power_of_two {n : ℕ} (h : 3 ≤ n) :
  ∀ X, (valid_transformation n X ↔ (∃ k, X = 2^k ∧ 2^k ≥ n)) :=
begin
  sorry
end

end all_numbers_turn_into_power_of_two_l618_618671


namespace middle_number_is_4_l618_618465

-- Sum conditions for the numbers
variables {a b c d e : ℕ}
variables (h1 : a + b = 4)
variables (h2 : a + c = 5)
variables (h3 : b + c = 7)
variables (h4 : a + d = 7)
variables (h5 : d + e = 13)

-- Final statement to prove
theorem middle_number_is_4 : 
  (a + b = 4) → 
  (a + c = 5) →
  (b + c = 7) →
  (a + d = 7) →
  (d + e = 13) →
  (∀ (lst : List ℕ), 
    lst = [a, b, c, d, e] → 
    lst.sort.nthLe 2 sorry = 4) := 
by
  sorry

end middle_number_is_4_l618_618465


namespace sum_of_digits_is_4_l618_618766

-- Define the initial sequence and the erasure conditions
def initial_sequence : List ℕ := [1, 2, 3, 4, 5, 6].repeatN 2000

-- Define the function to erase every nth element from a list
def erase_every_nth (n : ℕ) (l : List ℕ) : List ℕ := l.enum.filter (λ ⟨idx, _⟩ => (idx + 1) % n ≠ 0).map Prod.snd

-- Define the final sequence after the three rounds of erasures
def final_sequence : List ℕ :=
  let after_first_erase := erase_every_nth 4 initial_sequence
  let after_second_erase := erase_every_nth 5 after_first_erase
  erase_every_nth 6 after_second_erase

-- Find the sum of the three digits at the positions 2019, 2020, 2021
def sum_of_digits : ℕ := 
  let n := 20000 -- to ensure the sequence is long enough
  let positions := [2019, 2020, 2021].map (λ x => (x - 1) % final_sequence.length)
  positions.map (λ pos => final_sequence.nth pos).sum

theorem sum_of_digits_is_4 : sum_of_digits = 4 := by
  -- elaborate proof steps go here
  sorry

end sum_of_digits_is_4_l618_618766


namespace alpha_pow_n_fractional_sum_integer_l618_618777

open BigOperators

theorem alpha_pow_n_fractional_sum_integer 
  (α : ℝ) (hα : α + (1 / α) ∈ ℤ) : 
  ∀ n : ℕ, α^n + (1 / α^n) ∈ ℤ :=
sorry

end alpha_pow_n_fractional_sum_integer_l618_618777


namespace determine_y_l618_618384

noncomputable def y : ℝ := sqrt (4 + y)

theorem determine_y : y = (1 + Real.sqrt 17) / 2 :=
sorry

end determine_y_l618_618384


namespace exponent_calculation_l618_618568

-- Define the necessary exponents and base
def base : ℕ := 19
def exp1 : ℕ := 11
def exp2 : ℕ := 8

-- Given condition
lemma power_property (a : ℕ) (m n : ℕ) : a^m / a^n = a^(m - n) := by sorry

-- Proof that 19^11 / 19^8 = 6859
theorem exponent_calculation : base^exp1 / base^exp2 = 6859 := by
  have : base^exp1 / base^exp2 = base^(exp1 - exp2) := power_property base exp1 exp2
  have : base^(exp1 - exp2) = base^3 := by rw [nat.sub_eq_iff_eq_add.mpr rfl]
  have : base^3 = 6859 := by -- This would be an arithmetic computation
    rfl -- Placeholder for the actual arithmetic; ideally, you'd verify this step.
  sorry

end exponent_calculation_l618_618568


namespace largest_n_divisible_by_n_plus_10_l618_618635

theorem largest_n_divisible_by_n_plus_10 :
  ∃ n : ℕ, (n^3 + 100) % (n + 10) = 0 ∧ ∀ m : ℕ, ((m^3 + 100) % (m + 10) = 0 → m ≤ n) ∧ n = 890 := 
sorry

end largest_n_divisible_by_n_plus_10_l618_618635


namespace unpainted_cubes_count_l618_618507

noncomputable def total_unit_cubes : ℕ := 64
noncomputable def cube_side_length : ℕ := 4
noncomputable def painted_strip_width : ℕ := 2
noncomputable def painted_faces : ℕ := 6

theorem unpainted_cubes_count : 
  let painted_cubes := 24 in
  total_unit_cubes - painted_cubes = 40
:= by
  sorry

end unpainted_cubes_count_l618_618507


namespace binary_addition_l618_618125

theorem binary_addition :
  (0b1101 : Nat) + 0b101 + 0b1110 + 0b111 + 0b1010 = 0b10101 := by
  sorry

end binary_addition_l618_618125


namespace eq_sqrt_l618_618912

-- Defining the problem statement
theorem eq_sqrt (x : ℝ) (h : sqrt (x + 1) = 3 * x - 1) : x = 7 / 9 :=
sorry -- Proof not included

end eq_sqrt_l618_618912


namespace problem1_problem2_l618_618703

open Nat

noncomputable def a_n : ℕ → ℝ
| 0       => 0
| (n + 1) => 1 / (2 * (n + 1) + 1)

theorem problem1 : ∀ n : ℕ, n ≠ 0 → (1 / a_n n.succ) = 2 * n + 3 := by
  sorry

theorem problem2 (n : ℕ) (hn : n ≠ 0) : (∑ i in Finset.range n, a_n (i + 1) * a_n (i + 2)) < 1 / 6 := by
  sorry

end problem1_problem2_l618_618703


namespace binary_operations_unique_l618_618622

def binary_operation (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → (f a (f b c) = (f a b) * c)
  ∧ ∀ a : ℝ, a > 0 → a ≥ 1 → f a a ≥ 1

theorem binary_operations_unique (f : ℝ → ℝ → ℝ) (h : binary_operation f) :
  (∀ a b, f a b = a * b) ∨ (∀ a b, f a b = a / b) :=
sorry

end binary_operations_unique_l618_618622


namespace quadratic_eq_two_distinct_real_roots_l618_618448

theorem quadratic_eq_two_distinct_real_roots :
    ∃ x y : ℝ, x ≠ y ∧ (x^2 + x - 1 = 0) ∧ (y^2 + y - 1 = 0) :=
by
    sorry

end quadratic_eq_two_distinct_real_roots_l618_618448


namespace largest_initial_number_l618_618312

theorem largest_initial_number (n : ℕ) (h : (∃ a b c d e : ℕ, n ≠ 0 ∧ n + a + b + c + d + e = 200 
                                              ∧ n % a ≠ 0 ∧ n % b ≠ 0 ∧ n % c ≠ 0 ∧ n % d ≠ 0 ∧ n % e ≠ 0)) 
: n ≤ 189 :=
sorry

end largest_initial_number_l618_618312


namespace round_999_9951_l618_618413

-- Defining the conditions for rounding to the nearest hundredth
def digit_in_hundredths (x : ℝ) : ℕ :=
  let x_str := x.toString
  let dec_index := x_str.indexOf '.'
  x_str[dec_index + 2].digitVal

def digit_in_thousandths (x : ℝ) : ℕ :=
  let x_str := x.toString
  let dec_index := x_str.indexOf '.'
  x_str[dec_index + 3].digitVal

def round_to_nearest_hundredth (x : ℝ) : ℝ :=
  let hundredth_digit := digit_in_hundredths x
  let thousandth_digit := digit_in_thousandths x
  if thousandth_digit > 5 ∨ (thousandth_digit = 5 ∧ hundredth_digit % 2 = 1) then
    let mul_rounded_up := (x * 100).ceil / 100
    mul_rounded_up
  else
    let mul_rounded_down := (x * 100).floor / 100
    mul_rounded_down

theorem round_999_9951 : round_to_nearest_hundredth 999.9951 = 1000.00 :=
by {
  sorry
}

end round_999_9951_l618_618413


namespace quadratic_complete_square_l618_618443

theorem quadratic_complete_square :
  ∃ a b c : ℝ, (∀ x : ℝ, 4 * x^2 - 40 * x + 100 = a * (x + b)^2 + c) ∧ a + b + c = -1 :=
sorry

end quadratic_complete_square_l618_618443


namespace frank_reading_days_l618_618195

-- Define the parameters
def pages_weekdays : ℚ := 5.7
def pages_weekends : ℚ := 9.5
def total_pages : ℚ := 576
def pages_per_week : ℚ := (pages_weekdays * 5) + (pages_weekends * 2)

-- Define the property to be proved
theorem frank_reading_days : 
  (total_pages / pages_per_week).floor * 7 + 
  (total_pages - (total_pages / pages_per_week).floor * pages_per_week) / pages_weekdays 
  = 85 := 
  by
    sorry

end frank_reading_days_l618_618195


namespace line_intersects_yz_plane_at_l618_618637

-- Define the two points in 3D space
def P1 : ℝ × ℝ × ℝ := (3, 5, 1)
def P2 : ℝ × ℝ × ℝ := (5, 3, 6)

-- Define the direction vector as the difference between P2 and P1
def direction_vector (P1 P2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (P2.1 - P1.1, P2.2 - P1.2, P2.3 - P1.3)

-- Parameterize the line passing through P1 with the calculated direction vector
def parametric_line (t : ℝ) (P1 direction : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (P1.1 + t * direction.1, P1.2 + t * direction.2, P1.3 + t * direction.3)

-- Define the assertion that the line intersects the yz-plane at a specific point (0, 8, -5/2)
theorem line_intersects_yz_plane_at :
  ∃ (t : ℝ), parametric_line t P1 (direction_vector P1 P2) = (0, 8, -5/2) :=
sorry  -- Proof is to be completed.

end line_intersects_yz_plane_at_l618_618637


namespace travis_should_be_paid_l618_618894

theorem travis_should_be_paid :
  let (total_bowls, lost_bowls, broken_bowls) := (638, 12, 15) in
  let (fee_per_safe_bowl, fee_for_lost_bowl, fixed_fee) := (3, 4, 100) in
  (total_bowls - lost_bowls - broken_bowls) * fee_per_safe_bowl + fixed_fee - (lost_bowls + broken_bowls) * fee_for_lost_bowl = 1825 :=
by
  -- sorry as placeholder for the proof
  sorry

end travis_should_be_paid_l618_618894


namespace domain_of_sqrt_l618_618162

noncomputable def domain_of_function (x : ℝ) : Prop :=
  ∃ k : ℤ, ∃ n : ℕ, (x ≥ 2 * k.π - (2 * n.π / 3) ∧ x ≤ 2 * k.π + (2 * n.π / 3))

theorem domain_of_sqrt (x : ℝ) : 
  (∃ k : ℤ, 2 * k.π - (2 * 3.f.inv * π / 3) ≤ x ∧ x ≤ 2 * k.π + (2 * 3.f.inv * π / 3)) → 
  domain_of_function x :=
begin
  sorry
end

end domain_of_sqrt_l618_618162


namespace quiz_points_minus_homework_points_l618_618403

theorem quiz_points_minus_homework_points
  (total_points : ℕ)
  (quiz_points : ℕ)
  (test_points : ℕ)
  (homework_points : ℕ)
  (h1 : total_points = 265)
  (h2 : test_points = 4 * quiz_points)
  (h3 : homework_points = 40)
  (h4 : homework_points + quiz_points + test_points = total_points) :
  quiz_points - homework_points = 5 :=
by sorry

end quiz_points_minus_homework_points_l618_618403


namespace coded_CDE_equals_174_l618_618097

def encode_base6_to_decimal : String → ℕ
| "A" := 0
| "B" := 1
| "C" := 2
| "D" := 3
| "E" := 4
| "F" := 5
| _ := 0  -- Default case, although it shouldn't be reached

def decode_base6 (s : String) : ℕ :=
  match s.to_list.map encode_base6_to_decimal with
  | [c, d, e] := c * 36 + d * 6 + e  -- 36 = 6^2, 6 = 6^1, 1 = 6^0
  | _ := 0  -- This assumes the input is always a well-formed 3 character string

theorem coded_CDE_equals_174 (C D E : ℕ) (hC : C = 2) (hD : D = 3) (hE : E = 4):
  decode_base6 "CDE" = 174 :=
by
  have eq_C: encode_base6_to_decimal "C" = C := by rw [←hC]; refl
  have eq_D: encode_base6_to_decimal "D" = D := by rw [←hD]; refl
  have eq_E: encode_base6_to_decimal "E" = E := by rw [←hE]; refl
  simp only [decode_base6, eq_C, eq_D, eq_E, nat.mul_succ, nat.succ_eq_add_one, nat.add_mul_mod_self, nat.add_assoc]
  sorry  -- Completing the proof would be skipped per instructions

end coded_CDE_equals_174_l618_618097


namespace num_solutions_f_fx_eq_7_l618_618596

def f (x : ℝ) : ℝ :=
if x ≥ -2 then x^2 - 3 else x + 4

theorem num_solutions_f_fx_eq_7 : 
  (set_of (λ x : ℝ, f (f x) = 7)).finite.count = 4 := 
by 
  sorry

end num_solutions_f_fx_eq_7_l618_618596


namespace find_fourth_number_l618_618091

theorem find_fourth_number (x : ℝ) (h : 3 + 33 + 333 + x = 369.63) : x = 0.63 :=
sorry

end find_fourth_number_l618_618091


namespace kolya_mistaken_l618_618010

-- Definitions relating to the conditions
def at_least_four_blue_pencils (blue_pencils : ℕ) : Prop := blue_pencils >= 4
def at_least_five_green_pencils (green_pencils : ℕ) : Prop := green_pencils >= 5
def at_least_three_blue_pencils_and_four_green_pencils (blue_pencils green_pencils : ℕ) : Prop := blue_pencils >= 3 ∧ green_pencils >= 4
def at_least_four_blue_and_four_green_pencils (blue_pencils green_pencils : ℕ) : Prop := blue_pencils >= 4 ∧ green_pencils >= 4

-- Speaking truth conditions
variables (blue_pencils green_pencils : ℕ)
def vasya_true : Prop := at_least_four_blue_pencils blue_pencils
def kolya_true : Prop := at_least_five_green_pencils green_pencils
def petya_true : Prop := at_least_three_blue_pencils_and_four_green_pencils blue_pencils green_pencils
def misha_true : Prop := at_least_four_blue_and_four_green_pencils blue_pencils green_pencils

-- Given known information: three are true, one is false
def known_information : Prop := (vasya_true blue_pencils green_pencils ∧ petya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬kolya_true blue_pencils green_pencils)
                            ∨ (vasya_true blue_pencils green_pencils ∧ petya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ ¬misha_true blue_pencils green_pencils)
                            ∨ (vasya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬petya_true blue_pencils green_pencils)
                            ∨ (petya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬vasya_true blue_pencils green_pencils)

-- Theorem to be proved
theorem kolya_mistaken : known_information blue_pencils green_pencils → ¬kolya_true blue_pencils green_pencils :=
sorry

end kolya_mistaken_l618_618010


namespace least_upper_bound_neg_expression_l618_618179

noncomputable def least_upper_bound : ℝ :=
  - (9 / 2)

theorem least_upper_bound_neg_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  ∃ M, M = least_upper_bound ∧
  ∀ x, (∀ a b, 0 < a → 0 < b → a + b = 1 → x ≤ - (1 / (2 * a)) - (2 / b)) ↔ x ≤ M :=
sorry

end least_upper_bound_neg_expression_l618_618179


namespace exponent_division_l618_618570

theorem exponent_division : (19 ^ 11) / (19 ^ 8) = 6859 :=
by
  -- Here we assume the properties of powers and arithmetic operations
  sorry

end exponent_division_l618_618570


namespace slope_of_line_l618_618910

def point1 : (ℤ × ℤ) := (-4, 6)
def point2 : (ℤ × ℤ) := (3, -4)

def slope_formula (p1 p2 : (ℤ × ℤ)) : ℚ :=
  (p2.snd - p1.snd) / (p2.fst - p1.fst : ℚ)

theorem slope_of_line : slope_formula point1 point2 = -10 / 7 := by
  sorry

end slope_of_line_l618_618910


namespace dimes_paid_l618_618262

theorem dimes_paid (cost_in_dollars : ℕ) (dollars_to_dimes : ℕ) (h_cost : cost_in_dollars = 5) (h_conversion : dollars_to_dimes = 10) : cost_in_dollars * dollars_to_dimes = 50 :=
by
  rw [h_cost, h_conversion]
  norm_num

end dimes_paid_l618_618262


namespace intersection_point_coordinates_l618_618302

variables {A B C G H Q : Type}
variables {AG GC AH HB : ℝ}
variables {Qx Qy Qz : ℝ} 

-- Vector algebra setup
def is_linear_combination (a b c : ℝ) (x y z : ℝ) : Prop :=
  a + b + c = 1

def linear_combination (a b c : ℝ) (vA vB vC : A × B × C) : A × B × C :=
  (a * vA.1 + b * vB.1 + c * vC.1,
   a * vA.2 + b * vB.2 + c * vC.2,
   a * vA.3 + b * vB.3 + c * vC.3)

-- Given conditions
axiom AG_GC_ratio : AG / GC = 3 / 2
axiom AH_HB_ratio : AH / HB = 2 / 3

-- Intersection point coordinates
axiom Q_definition : Q = linear_combination (Qx, Qy, Qz) (A, B, C)

-- Goal
theorem intersection_point_coordinates :
  Qx = 3 / 14 ∧ Qy = 1 / 7 ∧ Qz = 9 / 14 :=
sorry

end intersection_point_coordinates_l618_618302


namespace measure_of_angle_F_l618_618756

theorem measure_of_angle_F (angle_D angle_E angle_F : ℝ) (h1 : angle_D = 80)
  (h2 : angle_E = 4 * angle_F + 10)
  (h3 : angle_D + angle_E + angle_F = 180) : angle_F = 18 := 
by
  sorry

end measure_of_angle_F_l618_618756


namespace solution_set_l618_618681

variable {f : ℝ → ℝ}
variable (h1 : ∀ x, x < 0 → x * deriv f x - 2 * f x > 0)
variable (h2 : ∀ x, x < 0 → f x ≠ 0)

theorem solution_set (h3 : ∀ x, -2024 < x ∧ x < -2023 → f (x + 2023) - (x + 2023)^2 * f (-1) < 0) :
    {x : ℝ | f (x + 2023) - (x + 2023)^2 * f (-1) < 0} = {x : ℝ | -2024 < x ∧ x < -2023} :=
by
  sorry

end solution_set_l618_618681


namespace intersection_A_B_l618_618673

open Set Nat

def A : Set ℕ := { x | x ≤ 2 }
def B : Set ℕ := { x | -2 < x ∧ x ≤ 3 }

theorem intersection_A_B : A ∩ B = {0, 1, 2} := 
by 
  sorry

end intersection_A_B_l618_618673


namespace horner_operations_l618_618070

-- Define the polynomial f(x)
def f (x : ℤ) : ℤ := 9 * x^6 + 12 * x^5 + 7 * x^4 + 54 * x^3 + 34 * x^2 + 9 * x + 1

-- Statement asserting the number of operations required by Horner's method
theorem horner_operations :
  (number_of_mul_operations f = 6) ∧ (number_of_add_operations f = 6) := 
sorry

end horner_operations_l618_618070


namespace determine_a_for_continuity_at_3_l618_618803

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 3 then 2*x^3 - x - 2 else a*x + 7

theorem determine_a_for_continuity_at_3 (a : ℝ) (h_cont : continuous_at (f a) 3) : a = 14 := by
sorry

end determine_a_for_continuity_at_3_l618_618803


namespace determine_A_value_l618_618851

noncomputable def solve_for_A (A B C : ℝ) : Prop :=
  (A = 1/16) ↔ 
  (∀ x : ℝ, (1 / ((x + 5) * (x - 3) * (x + 3))) = (A / (x + 5)) + (B / (x - 3)) + (C / (x + 3)))

theorem determine_A_value :
  solve_for_A (1/16) B C :=
by
  sorry

end determine_A_value_l618_618851


namespace EquationD_is_quadratic_l618_618483

variable (x a b c y : ℝ)

-- Definition of equations
def EquationA := x - (1 / x) + 2 = 0
def EquationB := x^2 + 2 * x + y = 0
def EquationC := a * x^2 + b * x + c = 0
def EquationD := x^2 - x + 1 = 0

-- Statement that Equation D is a quadratic equation
theorem EquationD_is_quadratic : EquationD := 
sorry

end EquationD_is_quadratic_l618_618483


namespace infinite_pairs_exist_l618_618827

theorem infinite_pairs_exist : 
  ∃^∞ (a b : ℤ), ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ * x₂ = 1 ∧ x₁^2012 = a * x₁ + b ∧ x₂^2012 = a * x₂ + b :=
sorry

end infinite_pairs_exist_l618_618827


namespace largest_subset_size_l618_618974

-- Definition of the subset condition
def valid_subset (s : Set ℕ) : Prop :=
  ∀ (a ∈ s) (b ∈ s), a ≠ 4 * b

-- Setting the range
def range_set : Set ℕ := { n | 1 ≤ n ∧ n ≤ 150 }

-- The main theorem statement
theorem largest_subset_size : ∃ (s : Set ℕ), s ⊆ range_set ∧ valid_subset (s) ∧ #(s) = 150 :=
by
  sorry

end largest_subset_size_l618_618974


namespace tangent_line_to_parabola_parallel_l618_618429

theorem tangent_line_to_parabola_parallel (m : ℝ) :
  ∀ (x y : ℝ), (y = x^2) → (2*x - y + m = 0 → m = -1) :=
by
  sorry

end tangent_line_to_parabola_parallel_l618_618429


namespace xiaohui_median_determines_top_5_l618_618510

theorem xiaohui_median_determines_top_5 (scores : Fin 9 → ℝ) (distinct_scores : ∀ i j : Fin 9, i ≠ j → scores i ≠ scores j) (xiaohui_score : ℝ):
  to_enter_top_5 (scores, distinct_scores, xiaohui_score) = median scores := 
sorry

def to_enter_top_5 (scores : Fin 9 → ℝ) (distinct_scores : ∀ i j : Fin 9, i ≠ j → scores i ≠ scores j) (xiaohui_score : ℝ) : ℝ :=
if xiaohui_score > median scores then median scores else median scores

noncomputable def median (scores : Fin 9 → ℝ) : ℝ :=
(sorry : ℝ)

end xiaohui_median_determines_top_5_l618_618510


namespace sum_seq_15_l618_618873

def sequence (n : ℕ) : ℤ := (-1)^(n-1) * (4*n - 3)

def sum_seq (n : ℕ) : ℤ :=
  (Finset.range n).sum (λ k => sequence (k + 1))

theorem sum_seq_15 : sum_seq 15 = 29 := by
  sorry

end sum_seq_15_l618_618873


namespace hyperbola_eccentricity_l618_618864

-- Definitions extracted from the problem conditions
def hyperbola_eq (x y a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

def asymptote_eq (a b : ℝ) : Prop := b / a = 2

-- The statement to prove the eccentricity e equals √5
theorem hyperbola_eccentricity (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : asymptote_eq a b) : 
  let c := real.sqrt (a^2 + b^2) in 
  let e := c / a in 
  e = real.sqrt 5 := 
by
  sorry

end hyperbola_eccentricity_l618_618864


namespace fraction_to_decimal_l618_618598

theorem fraction_to_decimal : (3 : ℚ) / 60 = 0.05 := 
by sorry

end fraction_to_decimal_l618_618598


namespace largest_initial_number_l618_618349

theorem largest_initial_number :
  ∃ n : ℕ, (∀ k : ℕ, (n % k ≠ 0 → k ∈ {2, 2, 2, 2, 3}) ∧ (n + 11 = 200)) ∧ (n = 189) :=
begin
  sorry -- Proof not required per instruction
end

end largest_initial_number_l618_618349


namespace number_of_green_cards_l618_618456

theorem number_of_green_cards :
  ∀ (total_cards red_fraction black_fraction : ℕ),
    total_cards = 120 →
    red_fraction = 2 / 5 →
    black_fraction = 5 / 9 →
    let red_cards := (red_fraction * total_cards) in
    let non_red_cards := (total_cards - red_cards) in
    let black_cards := (black_fraction * non_red_cards) in
    let green_cards := (non_red_cards - black_cards) in
    green_cards = 32 := by
  sorry

end number_of_green_cards_l618_618456


namespace largest_initial_number_l618_618307

theorem largest_initial_number : ∃ n : ℕ, (n + 5 ∑ k : ℕ, k ≠ 0 ∧ ¬ (n % k = 0)) = 200 ∧ n = 189 :=
begin
  sorry
end

end largest_initial_number_l618_618307


namespace factorization_l618_618618

theorem factorization (a : ℝ) : 2 * a^2 - 2 * a + 1/2 = 2 * (a - 1/2)^2 :=
by
  sorry

end factorization_l618_618618


namespace exists_lambdas_orthocenter_l618_618502

theorem exists_lambdas_orthocenter
    {A1 A2 A3 A4 : Type}
    (orthocenter : orthocenter A1 A2 A3 = A4)
    (not_right_triangle : ¬ is_right_triangle A1 A2 A3) :
    ∃ (λ1 λ2 λ3 λ4 : ℝ),
    (∀ i j : ℕ, i ≠ j → (distance A1 A2)^2 = (λ1 + λ2)) ∧
    (∑ i in [1, 2, 3, 4], (1 / λi)) = 0 :=
sorry

end exists_lambdas_orthocenter_l618_618502


namespace kolya_mistaken_l618_618005

-- Definitions relating to the conditions
def at_least_four_blue_pencils (blue_pencils : ℕ) : Prop := blue_pencils >= 4
def at_least_five_green_pencils (green_pencils : ℕ) : Prop := green_pencils >= 5
def at_least_three_blue_pencils_and_four_green_pencils (blue_pencils green_pencils : ℕ) : Prop := blue_pencils >= 3 ∧ green_pencils >= 4
def at_least_four_blue_and_four_green_pencils (blue_pencils green_pencils : ℕ) : Prop := blue_pencils >= 4 ∧ green_pencils >= 4

-- Speaking truth conditions
variables (blue_pencils green_pencils : ℕ)
def vasya_true : Prop := at_least_four_blue_pencils blue_pencils
def kolya_true : Prop := at_least_five_green_pencils green_pencils
def petya_true : Prop := at_least_three_blue_pencils_and_four_green_pencils blue_pencils green_pencils
def misha_true : Prop := at_least_four_blue_and_four_green_pencils blue_pencils green_pencils

-- Given known information: three are true, one is false
def known_information : Prop := (vasya_true blue_pencils green_pencils ∧ petya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬kolya_true blue_pencils green_pencils)
                            ∨ (vasya_true blue_pencils green_pencils ∧ petya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ ¬misha_true blue_pencils green_pencils)
                            ∨ (vasya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬petya_true blue_pencils green_pencils)
                            ∨ (petya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬vasya_true blue_pencils green_pencils)

-- Theorem to be proved
theorem kolya_mistaken : known_information blue_pencils green_pencils → ¬kolya_true blue_pencils green_pencils :=
sorry

end kolya_mistaken_l618_618005


namespace AB_value_l618_618303

noncomputable def AB_in_right_triangle (A B C : Type) 
  [InnerProductSpace ℝ (EuclideanSpace ℝ (Fin 2))] 
  (A B C : Fin 2 → ℝ) 
  (hA : ∠ A B C = 90)
  (BC : distance B C = 20) 
  (h_tan_sin : tan (angle A B C) = 4 * sin (angle B A C)) : ℝ :=
√ ((20^2) - (5^2) )

theorem AB_value (A B C : Type) 
  [InnerProductSpace ℝ (EuclideanSpace ℝ (Fin 2))] 
  (A B C : Fin 2 → ℝ)
  (hA : ∠ A B C = 90)
  (BC : distance B C = 20) 
  (h_tan_sin : tan (angle A B C) = 4 * sin (angle B A C)) : 
  AB_in_right_triangle A B C hA BC h_tan_sin = 5 * √15 := by
sorry

end AB_value_l618_618303


namespace trapezium_area_l618_618077

/--
The problem states:
Find the area of a trapezium whose parallel sides are 28 cm and 20 cm long, and the distance between them is 21 cm.
We will prove that the area of the trapezium is 504 square centimeters.
-/

def top_side := 28 -- cm
def bottom_side := 20 -- cm
def distance_between_sides := 21 -- cm

theorem trapezium_area :
  1/2 * (top_side + bottom_side) * distance_between_sides = 504 :=
by
sorrry

end trapezium_area_l618_618077


namespace find_a_divides_poly_l618_618193

theorem find_a_divides_poly (a : ℤ) :
  (∀ p : polynomial ℤ, (x^2 - x + a) * p = x^12 + x + 72) → 
  a ∣ 72 ∧ a ∣ 74 ∧ (a + 1) ∣ 70 → 
  a = 2 :=
by
  -- Proof omitted
  sorry

end find_a_divides_poly_l618_618193


namespace number_of_cows_brought_l618_618511

/--
A certain number of cows and 10 goats are brought for Rs. 1500. 
If the average price of a goat is Rs. 70, and the average price of a cow is Rs. 400, 
then the number of cows brought is 2.
-/
theorem number_of_cows_brought : 
  ∃ c : ℕ, ∃ g : ℕ, g = 10 ∧ (70 * g + 400 * c = 1500) ∧ c = 2 :=
sorry

end number_of_cows_brought_l618_618511


namespace intersection_probability_l618_618615

-- Define the points and the condition of being evenly spaced around a circle
def points : Set ℕ := {i | 0 ≤ i ∧ i < 2023}

-- Define the event in terms of the problem conditions
theorem intersection_probability (A B C D E F G H : points) :
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (D ≠ E) ∧ (E ≠ F) ∧ (F ≠ G) ∧ (G ≠ H) ∧ (H ≠ A) →
  ∀ (x1 x2 x3 x4 : points), x1 ∈ {A, B, C, D} → x2 ∈ {A, B, C, D} → x3 ∈ {E, F, G, H} → x4 ∈ {E, F, G, H} →
  (x1 ≠ x2) ∧ (x3 ≠ x4) →
  (∀ A B C D, chord_intersects A B C D) →
  (∀ E F G H, chord_intersects E F G H) →
  intersection_probability = (1 : ℚ) / 36 :=
sorry

end intersection_probability_l618_618615


namespace unique_m_power_function_increasing_l618_618869

theorem unique_m_power_function_increasing : 
  ∃! (m : ℝ), (∀ x : ℝ, x > 0 → (m^2 - m - 5) * x^(m-1) > 0) ∧ (m^2 - m - 5 = 1) ∧ (m - 1 > 0) :=
by
  sorry

end unique_m_power_function_increasing_l618_618869


namespace alex_has_9_twenty_cent_coins_l618_618545

def alex_coin_problem (x y : ℕ) : Prop :=
  x + y = 14 ∧ (27 - x = 22) → y = 9

theorem alex_has_9_twenty_cent_coins : ∃ x y : ℕ, alex_coin_problem x y :=
by {
  existsi 5,
  existsi 9,
  unfold alex_coin_problem,
  split,
  {
    -- Proof that x + y = 14
    exact rfl,
  },
  {
    -- Proof that 27 - x = 22
    exact rfl,
  }
}

end alex_has_9_twenty_cent_coins_l618_618545


namespace parabola_range_m_l618_618252

noncomputable def parabola (m : ℝ) (x : ℝ) : ℝ := x^2 - (4*m + 1)*x + (2*m - 1)

theorem parabola_range_m (m : ℝ) :
  (∀ x : ℝ, parabola m x = 0 → (1 < x ∧ x < 2) ∨ (x < 1 ∨ x > 2)) ∧
  parabola m 0 < -1/2 →
  1/6 < m ∧ m < 1/4 :=
by
  sorry

end parabola_range_m_l618_618252


namespace constant_term_in_expansion_l618_618235

theorem constant_term_in_expansion (n : ℕ) (h : (x - (2 / x)) ^ n) (sum_coeffs : 2^n = 64) : (constant_term (x - (2 / x)) ^ n) = -160 :=
by
  sorry

end constant_term_in_expansion_l618_618235


namespace fourth_power_sqrt_expr_l618_618868

theorem fourth_power_sqrt_expr : (sqrt (1 + sqrt (1 + sqrt (1)))) ^ 4 = 3 + 2 * sqrt 2 :=
by
  sorry

end fourth_power_sqrt_expr_l618_618868


namespace hyperbola_eccentricity_l618_618865

-- Definitions extracted from the problem conditions
def hyperbola_eq (x y a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

def asymptote_eq (a b : ℝ) : Prop := b / a = 2

-- The statement to prove the eccentricity e equals √5
theorem hyperbola_eccentricity (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : asymptote_eq a b) : 
  let c := real.sqrt (a^2 + b^2) in 
  let e := c / a in 
  e = real.sqrt 5 := 
by
  sorry

end hyperbola_eccentricity_l618_618865


namespace trajectory_of_P_is_line_segment_l618_618804

noncomputable def F1 : (ℝ × ℝ) := (0, -3)
noncomputable def F2 : (ℝ × ℝ) := (0, 3)
def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem trajectory_of_P_is_line_segment :
  (∀ P : ℝ × ℝ, distance P F1 + distance P F2 = 6) →
  (∀ P : ℝ × ℝ, P ∈ segment ℝ F1 F2)
:= sorry

end trajectory_of_P_is_line_segment_l618_618804


namespace largest_subset_size_l618_618985

theorem largest_subset_size :
  ∃ S : Set ℕ, S ⊆ {i | 1 ≤ i ∧ i ≤ 150} ∧ 
  (∀ (a b : ℕ), a ∈ S → b ∈ S → ¬ (a = 4 * b ∨ b = 4 * a)) ∧ 
  S.card = 141 :=
sorry

end largest_subset_size_l618_618985


namespace inequality_abc_l618_618824

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  (1 / (1 + a + b)) + (1 / (1 + b + c)) + (1 / (1 + c + a)) ≤ 1 :=
by
  sorry

end inequality_abc_l618_618824


namespace max_distinct_rectangles_l618_618353

theorem max_distinct_rectangles (n : ℕ) (h : n = 8) :
  ∃ N, N = 12 ∧ ∀ rect ∈ {rect | let ⟨w, h⟩ := rect in w ≤ n ∧ h ≤ n}, N = Cardinal.mk {rect | let ⟨w, h⟩ := rect in w ≠ h} :=
sorry

end max_distinct_rectangles_l618_618353


namespace cylinder_radius_in_cone_l618_618533

theorem cylinder_radius_in_cone (d h r : ℝ) (h_d : d = 20) (h_h : h = 24) (h_cylinder : 2 * r = r):
  r = 60 / 11 :=
by
  sorry

end cylinder_radius_in_cone_l618_618533


namespace functional_eq_solution_l618_618623

theorem functional_eq_solution (k : ℝ) (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (2 * x - 2 * y) + x = f (3 * x) - f (2 * y) + k * y) →
  (f = λ x, x ∧ k = 0) :=
by
  sorry

end functional_eq_solution_l618_618623


namespace maximum_possible_value_of_k_l618_618636

theorem maximum_possible_value_of_k :
  ∀ (k : ℕ), 
    (∃ (x : ℕ → ℝ), 
      (∀ i j : ℕ, 1 ≤ i ∧ i ≤ k ∧ 1 ≤ j ∧ j ≤ k → x i > 1 ∧ x i ≠ x j ∧ x i ^ ⌊x j⌋ = x j ^ ⌊x i⌋)) 
      → k ≤ 4 :=
by
  sorry

end maximum_possible_value_of_k_l618_618636


namespace graph_squares_count_l618_618433

theorem graph_squares_count :
  let x_intercept := 45
  let y_intercept := 5
  let total_squares := x_intercept * y_intercept
  let diagonal_squares := x_intercept + y_intercept - 1
  let non_diagonal_squares := total_squares - diagonal_squares
  non_diagonal_squares / 2 = 88 :=
by
  let x_intercept := 45
  let y_intercept := 5
  let total_squares := x_intercept * y_intercept
  let diagonal_squares := x_intercept + y_intercept - 1
  let non_diagonal_squares := total_squares - diagonal_squares
  have h : (non_diagonal_squares / 2 = 88) := sorry
  exact h

end graph_squares_count_l618_618433


namespace eleven_power_2023_mod_50_l618_618066

theorem eleven_power_2023_mod_50 :
  11^2023 % 50 = 31 :=
by
  sorry

end eleven_power_2023_mod_50_l618_618066


namespace quadratic_no_real_roots_probability_l618_618441

theorem quadratic_no_real_roots_probability :
  (1 : ℝ) - 1 / 4 - 0 = 3 / 4 :=
by
  sorry

end quadratic_no_real_roots_probability_l618_618441


namespace largest_subset_size_l618_618968

def largest_subset_with_property (s : Set ℤ) : Prop :=
  ∀ (a b : ℤ), a ∈ s → b ∈ s → (a ≠ 4 * b ∧ b ≠ 4 * a)

theorem largest_subset_size : ∃ s : Set ℤ, (∀ x : ℤ, x ∈ s → 1 ≤ x ∧ x ≤ 150) ∧ largest_subset_with_property s ∧ s.card = 120 :=
by
  sorry

end largest_subset_size_l618_618968


namespace kolya_mistaken_l618_618009

-- Definitions relating to the conditions
def at_least_four_blue_pencils (blue_pencils : ℕ) : Prop := blue_pencils >= 4
def at_least_five_green_pencils (green_pencils : ℕ) : Prop := green_pencils >= 5
def at_least_three_blue_pencils_and_four_green_pencils (blue_pencils green_pencils : ℕ) : Prop := blue_pencils >= 3 ∧ green_pencils >= 4
def at_least_four_blue_and_four_green_pencils (blue_pencils green_pencils : ℕ) : Prop := blue_pencils >= 4 ∧ green_pencils >= 4

-- Speaking truth conditions
variables (blue_pencils green_pencils : ℕ)
def vasya_true : Prop := at_least_four_blue_pencils blue_pencils
def kolya_true : Prop := at_least_five_green_pencils green_pencils
def petya_true : Prop := at_least_three_blue_pencils_and_four_green_pencils blue_pencils green_pencils
def misha_true : Prop := at_least_four_blue_and_four_green_pencils blue_pencils green_pencils

-- Given known information: three are true, one is false
def known_information : Prop := (vasya_true blue_pencils green_pencils ∧ petya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬kolya_true blue_pencils green_pencils)
                            ∨ (vasya_true blue_pencils green_pencils ∧ petya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ ¬misha_true blue_pencils green_pencils)
                            ∨ (vasya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬petya_true blue_pencils green_pencils)
                            ∨ (petya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬vasya_true blue_pencils green_pencils)

-- Theorem to be proved
theorem kolya_mistaken : known_information blue_pencils green_pencils → ¬kolya_true blue_pencils green_pencils :=
sorry

end kolya_mistaken_l618_618009


namespace max_value_of_XYZ_XY_YZ_ZX_l618_618365

theorem max_value_of_XYZ_XY_YZ_ZX (X Y Z : ℕ) (h : X + Y + Z = 15) : 
  X * Y * Z + X * Y + Y * Z + Z * X ≤ 200 := 
sorry

end max_value_of_XYZ_XY_YZ_ZX_l618_618365


namespace sum_of_faces_of_larger_cube_l618_618846

-- Definitions based on conditions
def standard_die_sums_to_seven (a b : ℕ) : Prop := a + b = 7

def cube_face_numbers (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6}

def sixty_four_cubes_form_larger_cube: Prop := 
  ∀ cube : ℕ, cube = 64

-- The math proof problem
theorem sum_of_faces_of_larger_cube (sum_on_faces : ℕ) 
  (h1 : ∀ a b, standard_die_sums_to_seven a b)
  (h2 : ∀ n, cube_face_numbers n)
  (h3 : sixty_four_cubes_form_larger_cube) :
  144 ≤ sum_on_faces ∧ sum_on_faces ≤ 528 :=
sorry

end sum_of_faces_of_larger_cube_l618_618846


namespace range_of_f_l618_618445

-- Define the function y = (sin 2x - 3) / (sin x + cos x - 2)
noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x) - 3) / (Real.sin x + Real.cos x - 2)

-- Define the range of the function using the inequality derived
theorem range_of_f : set.range f = set.Icc (2 - Real.sqrt 2) (2 + Real.sqrt 2) := sorry

end range_of_f_l618_618445


namespace max_value_of_expression_l618_618646

theorem max_value_of_expression (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) : 
  (∃ x : ℝ, x = 3 → 
    ∀ A : ℝ, A = (a^4 + b^4 + c^4) / ((a + b + c)^4 - 80 * (a * b * c)^(4/3)) → 
      A ≤ x) :=
by
  sorry

end max_value_of_expression_l618_618646


namespace part1_part2_part3_l618_618367

def harmonic_set (A : Set ℝ) : Prop :=
  ∑ t in A, t = ∏ t in A, t

theorem part1 (E F : Set ℝ) (hE : E = {1, 2}) (hF : F = {-1, 0, 1}) :
  ¬ harmonic_set E ∧ harmonic_set F :=
  by sorry

theorem part2 (x y : ℝ) (hxy : harmonic_set {x, y}) :
  ¬ ∃ z : ℝ, harmonic_set {x, y, z} :=
  by sorry

theorem part3 (M : Set ℕ) (hM : harmonic_set (M : Set ℝ)) :
  M = {1, 2, 3} :=
  by sorry

end part1_part2_part3_l618_618367


namespace sum_of_squares_of_roots_l618_618151

theorem sum_of_squares_of_roots :
  (∀ y : ℝ, y^3 - 8*y^2 + 9*y - 2 = 0 → y ≥ 0) →
  (∃ r s t : ℝ, (r^3 - 8*r^2 + 9*r - 2 = 0) ∧ (s^3 - 8*s^2 + 9*s - 2 = 0) ∧ (t^3 - 8*t^2 + 9*t - 2 = 0) ∧
          r + s + t = 8 ∧ r * s + s * t + t * r = 9) →
  r^2 + s^2 + t^2 = 46 :=
by transition
by_paths
by tactic
show ex.

Sorry: Sign-of- root {.no.root,y} good.tags -lean.error

end sum_of_squares_of_roots_l618_618151


namespace pairwise_sum_product_l618_618821

theorem pairwise_sum_product :
  ∃ (a b c d e : ℕ), (pairwise_sums a b c d e = {5, 8, 9, 13, 14, 14, 15, 17, 18, 23}) ∧ (a * b * c * d * e = 4752) :=
begin
  sorry
end

def pairwise_sums (a b c d e : ℕ) : Multiset ℕ :=
  [a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e]

end pairwise_sum_product_l618_618821


namespace cos_arccos_minus_arctan_eq_l618_618620

noncomputable def cos_arccos_minus_arctan: Real :=
  Real.cos (Real.arccos (4 / 5) - Real.arctan (1 / 2))

theorem cos_arccos_minus_arctan_eq : cos_arccos_minus_arctan = (11 * Real.sqrt 5) / 25 := by
  sorry

end cos_arccos_minus_arctan_eq_l618_618620


namespace minimum_positive_phi_odd_function_l618_618160

noncomputable def determinant : ℝ × ℝ × ℝ × ℝ → ℝ :=
λ ⟨a1, a2, a3, a4⟩, a1 * a4 - a2 * a3

noncomputable def f (x : ℝ) : ℝ :=
  determinant (⟨Real.sin x, Real.cos x, 1, Real.sqrt 3⟩)

theorem minimum_positive_phi_odd_function :
  ∃ (ϕ : ℝ), ϕ > 0 ∧ (∀ x : ℝ, f (x + ϕ) = -f (-x)) ∧ ϕ = (5 / 6) * Real.pi :=
sorry

end minimum_positive_phi_odd_function_l618_618160


namespace eggs_per_chicken_per_day_l618_618166

theorem eggs_per_chicken_per_day (E c d : ℕ) (hE : E = 36) (hc : c = 4) (hd : d = 3) :
  (E / d) / c = 3 := by
  sorry

end eggs_per_chicken_per_day_l618_618166


namespace triangle_area_correct_l618_618779

-- Define the vectors a, b, and c as given in the problem
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (6, 2)
def c : ℝ × ℝ := (1, -1)

-- Define the function to calculate the area of the triangle with the given vertices
def triangle_area (u v w : ℝ × ℝ) : ℝ :=
  0.5 * abs ((v.1 - u.1) * (w.2 - u.2) - (w.1 - u.1) * (v.2 - u.2))

-- State the proof problem
theorem triangle_area_correct : triangle_area c (a.1 + c.1, a.2 + c.2) (b.1 + c.1, b.2 + c.2) = 8.5 :=
by
  -- Proof can go here
  sorry

end triangle_area_correct_l618_618779


namespace contrapositive_statement_l618_618678

variable {a : ℝ}
variable {f : ℝ → ℝ}

theorem contrapositive_statement
  (h : ∀ x, x ≥ 0 → f(x) ≥ 0 → a ≥ 1 / 2) :
  (∃ x, x ≥ 0 ∧ f(x) < 0) → a < 1 / 2 :=
by
  sorry

end contrapositive_statement_l618_618678


namespace min_even_integers_proof_l618_618889

-- Definitions of conditions
def sum_of_three_integers (x y z : ℤ) : Prop :=
  x + y + z = 33

def sum_of_two_additional_integers (x y z a b : ℤ) : Prop :=
  x + y + z + a + b = 52 ∧ a ∈ {8, 9, 10} ∧ b ∈ {8, 9, 10}

def sum_of_all_integers (x y z a b m : ℤ) : Prop :=
  x + y + z + a + b + m = 67 ∧ m ∈ {13, 14, 15}

noncomputable def min_even_integers (x y z a b m : ℤ) : ℤ :=
  if is_even x + is_even y + is_even z + is_even a + is_even b + is_even m then 1 else 0

-- Theorem statement
theorem min_even_integers_proof (x y z a b m : ℤ) :
  sum_of_three_integers x y z →
  sum_of_two_additional_integers x y z a b →
  sum_of_all_integers x y z a b m →
  min_even_integers x y z a b m = 1 :=
sorry

end min_even_integers_proof_l618_618889


namespace complex_abs_eq_one_l618_618798

/--
If \( r \) is a real number such that \( |r| < 3 \) and \( z \) is a complex number such that \( z + \frac{1}{z} = r \), then \( |z| = 1 \).
-/
theorem complex_abs_eq_one (r : ℝ) (z : ℂ) (h_r : |r| < 3) (h_z : z + z⁻¹ = r) : |z| = 1 := 
sorry

end complex_abs_eq_one_l618_618798


namespace energy_increase_vertex_to_center_l618_618194

noncomputable def energy_increase (e_initial : ℝ) : ℝ := 15 * Real.sqrt 2 - 10

theorem energy_increase_vertex_to_center 
  (q : ℝ) (s : ℝ) : 
  let e_initial := 20 in 
  let e_final := 15 * Real.sqrt 2 + 10 in 
  energy_increase e_initial = e_final - e_initial := 
sorry

end energy_increase_vertex_to_center_l618_618194


namespace cost_of_one_book_l618_618392

theorem cost_of_one_book (x : ℝ) : 
  (9 * x = 11) ∧ (13 * x = 15) → x = 1.23 :=
by sorry

end cost_of_one_book_l618_618392


namespace maximum_value_of_A_l618_618647

theorem maximum_value_of_A (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
    (a^4 + b^4 + c^4) / ((a + b + c)^4 - 80 * (a * b * c)^(4 / 3)) ≤ 3 :=
sorry

end maximum_value_of_A_l618_618647


namespace solutions_to_h_eq_1_l618_618802

noncomputable def h (x : ℝ) : ℝ :=
if x ≤ 0 then 5 * x + 10 else 3 * x - 5

theorem solutions_to_h_eq_1 : {x : ℝ | h x = 1} = {-9/5, 2} :=
by
  sorry

end solutions_to_h_eq_1_l618_618802


namespace Kolya_mistake_l618_618021

def boys := ["Vasya", "Kolya", "Petya", "Misha"]

constant num_blue_pencils : ℕ
constant num_green_pencils : ℕ

axiom Vasya_statement : num_blue_pencils >= 4
axiom Kolya_statement : num_green_pencils >= 5
axiom Petya_statement : num_blue_pencils >= 3 ∧ num_green_pencils >= 4
axiom Misha_statement : num_blue_pencils >= 4 ∧ num_green_pencils >= 4

axiom three_truths_one_mistake : 
  (Vasya_statement ∨ ¬Vasya_statement) ∧
  (Kolya_statement ∨ ¬Kolya_statement) ∧
  (Petya_statement ∨ ¬Petya_statement) ∧
  (Misha_statement ∨ ¬Misha_statement) ∧
  ((Vasya_statement ? true : 1) + 
   (Kolya_statement ? true : 1) + 
   (Petya_statement ? true : 1) +
   (Misha_statement ? true : 1) == 3)

theorem Kolya_mistake : ¬Kolya_statement :=
by
  sorry

end Kolya_mistake_l618_618021


namespace speed_of_other_child_l618_618837

-- Define the speeds of Chukov and Gekov
variables {V1 V2 : ℝ}

-- Given conditions as hypotheses in Lean
-- 1. They meet four times more often than they overtake each other
axiom frequency_condition : (V1 + V2) / |V1 - V2| = 4

-- 2. One of the children's speed is 6 m/s
axiom one_child_speed : V1 = 6 ∨ V2 = 6

-- Goal: Prove the speed of the other child is either 10 m/s or 3.6 m/s
theorem speed_of_other_child (V1 V2 : ℝ) (h1 : frequency_condition) (h2 : one_child_speed) :
  (V1 = 6 ∧ V2 = 10) ∨ (V1 = 10 ∧ V2 = 6) ∨ (V1 = 6 ∧ V2 = 3.6) ∨ (V1 = 3.6 ∧ V2 = 6) :=
sorry

end speed_of_other_child_l618_618837


namespace part_I_f_expression_part_I_monotonic_increase_part_II_b_length_l618_618199

variables (ω x A : ℝ) (k : ℤ)
def a_vector := (2 * Real.sin (ω * x), Real.sin (ω * x) + Real.cos (ω * x))
def b_vector := (Real.cos (ω * x), Real.sqrt 3 * (Real.sin (ω * x) - Real.cos (ω * x)))
def f (x : ℝ) := (a_vector ω x).fst * (b_vector ω x).fst + (a_vector ω x).snd * (b_vector ω x).snd

axiom ω_condition : 0 < ω ∧ ω < 1
axiom symmetry_axis : f (5 * Real.pi / 6) = 2 ∨ f (5 * Real.pi / 6) = -2

theorem part_I_f_expression : 
  f x = 2 * Real.sin (x - Real.pi / 3) :=
sorry

theorem part_I_monotonic_increase : 
  ∀ (k : ℤ), - Real.pi / 6 + 2 * k * Real.pi ≤ x ∧ x ≤ 2 * k * Real.pi + 5 * Real.pi / 6 :=
sorry

variables (c a b : ℝ)
axiom triangle_conditions : f A = 0 ∧ c = 3 ∧ a = Real.sqrt 13

theorem part_II_b_length : 
  b = 4 :=
sorry

end part_I_f_expression_part_I_monotonic_increase_part_II_b_length_l618_618199


namespace paint_house_together_time_l618_618722

-- Definitions of the conditions
def Sally_rate := 1 / 4
def John_rate := 1 / 6

-- The theorem statement to prove the combined time
theorem paint_house_together_time : 
  (1 / (Sally_rate + John_rate)) = 2.4 :=
by
  sorry

end paint_house_together_time_l618_618722


namespace largest_subset_no_member_is_4_times_another_l618_618984

-- Define the predicate that characterizes the subset
def valid_subset (S : Set ℕ) : Prop :=
  ∀ (x ∈ S) (y ∈ S), x ≠ 4 * y ∧ y ≠ 4 * x

-- Define the set of integers from 1 to 150
def range_1_to_150 : Set ℕ := {n | 1 ≤ n ∧ n ≤ 150}

-- State the theorem
theorem largest_subset_no_member_is_4_times_another :
  ∃ S ⊆ range_1_to_150, valid_subset S ∧ S.card = 140 := sorry

end largest_subset_no_member_is_4_times_another_l618_618984


namespace problem_statement_l618_618659

noncomputable def a : ℝ := 2 ^ 0.2
noncomputable def b : ℝ := 0.4 ^ 0.2
noncomputable def c : ℝ := 0.4 ^ 0.6

theorem problem_statement : a > b ∧ b > c :=
by {
    -- Proof goes here
    sorry
}

end problem_statement_l618_618659


namespace sum_of_a_b_l618_618719

theorem sum_of_a_b :
  ∃ a b : ℚ, (1 + real.sqrt 2)^5 = a + b * real.sqrt 2 ∧ a + b = 70 := by
  sorry

end sum_of_a_b_l618_618719


namespace binary_addition_l618_618904

theorem binary_addition :
  let num1 := 0b111111111
  let num2 := 0b101010101
  num1 + num2 = 852 := by
  sorry

end binary_addition_l618_618904


namespace kolya_is_wrong_l618_618026

def pencils_problem_statement (at_least_four_blue : Prop) 
                              (at_least_five_green : Prop) 
                              (at_least_three_blue_and_four_green : Prop) 
                              (at_least_four_blue_and_four_green : Prop) : 
                              Prop :=
  ∃ (B G : ℕ), -- B represents the number of blue pencils, G represents the number of green pencils
    ((B ≥ 4) ∧ (G ≥ 4)) ∧ -- Vasya's statement (at least 4 blue), Petya's and Misha's combined statement (at least 4 green)
    at_least_four_blue ∧ -- Vasya's statement (there are at least 4 blue pencils)
    (at_least_five_green ↔ G ≥ 5) ∧ -- Kolya's statement (there are at least 5 green pencils)
    at_least_three_blue_and_four_green ∧ -- Petya's statement (at least 3 blue and 4 green)
    at_least_four_blue_and_four_green -- Misha's statement (at least 4 blue and 4 green)

theorem kolya_is_wrong (at_least_four_blue : Prop) 
                        (at_least_five_green : Prop) 
                        (at_least_three_blue_and_four_green : Prop) 
                        (at_least_four_blue_and_four_green : Prop) : 
                        pencils_problem_statement at_least_four_blue 
                                                  at_least_five_green 
                                                  at_least_three_blue_and_four_green 
                                                  at_least_four_blue_and_four_green :=
sorry

end kolya_is_wrong_l618_618026


namespace sequence_convergence_l618_618718

def sequence (a0 : ℝ) (n : ℕ) : ℝ :=
  match n with
  | 0     => a0
  | n + 1 => (sequence a0 n)^2 - 1 / (n + 1)

theorem sequence_convergence (a0 : ℝ) : 
  a0 > 0 → 
  (∀ a, a > 0 → 
      ( (a0 ≥ a → ∀ n, sequence a0 n → ℝ → ∞) ∧ 
        (a0 < a → ∀ n, sequence a0 n → 0) ))) :=
begin
  sorry
end

end sequence_convergence_l618_618718


namespace find_a_l618_618670

theorem find_a (a : ℝ) : 
  (∃ x y : ℝ, (x^2 + y^2 = 4) ∧ (abs (x + y - a) / real.sqrt 2 = 1)) →
  (a = -real.sqrt 2 ∨ a = real.sqrt 2) := by
  sorry

end find_a_l618_618670


namespace integer_binom_sum_sum_integer_values_l618_618607

theorem integer_binom_sum (n : ℕ) (h : nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13) : n = 13 :=
sorry

theorem sum_integer_values (h : nat.choose 25 13 + nat.choose 25 12 = nat.choose 26 13) : 
  ∑ (x : ℕ) in {n | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13}.to_finset, x = 13 :=
sorry

end integer_binom_sum_sum_integer_values_l618_618607


namespace valid_outfits_number_l618_618714

def num_shirts := 7
def num_pants := 7
def num_hats := 7
def num_colors := 7

def total_outfits (num_shirts num_pants num_hats : ℕ) := num_shirts * num_pants * num_hats
def matching_color_outfits (num_colors : ℕ) := num_colors
def valid_outfits (num_shirts num_pants num_hats num_colors : ℕ) := 
  total_outfits num_shirts num_pants num_hats - matching_color_outfits num_colors

theorem valid_outfits_number : valid_outfits num_shirts num_pants num_hats num_colors = 336 := 
by
  sorry

end valid_outfits_number_l618_618714


namespace jade_transactions_correct_l618_618082

-- Definitions for the conditions
def mabel_transactions : ℕ := 90
def anthony_transactions : ℕ := mabel_transactions + (mabel_transactions * 10 / 100)
def cal_transactions : ℕ := (2 * anthony_transactions) / 3
def jade_transactions : ℕ := cal_transactions + 16

-- The theorem stating what we want to prove
theorem jade_transactions_correct : jade_transactions = 82 := by
  sorry

end jade_transactions_correct_l618_618082


namespace Travis_payment_l618_618896

variable (T L B F P_d P_l : ℕ) (H : T = 638 ∧ F = 100 ∧ P_d = 3 ∧ P_l = 4 ∧ L = 12 ∧ B = 15)

theorem Travis_payment (T L B F P_d P_l : ℕ) (H : T = 638 ∧ F = 100 ∧ P_d = 3 ∧ P_l = 4 ∧ L = 12 ∧ B = 15) : 
  let total_bowls_lost_broken := L + B in
  let amount_owed_for_lost_broken := total_bowls_lost_broken * P_l in
  let bowls_delivered_safely := T - total_bowls_lost_broken in
  let payment_for_safely_delivered := bowls_delivered_safely * P_d in
  let total_payment_before_deductions := payment_for_safely_delivered + F in
  let final_amount := total_payment_before_deductions - amount_owed_for_lost_broken in
  final_amount = 1825 :=
by {
  have total_bowls_lost_broken := L + B,
  have amount_owed_for_lost_broken := total_bowls_lost_broken * P_l,
  have bowls_delivered_safely := T - total_bowls_lost_broken,
  have payment_for_safely_delivered := bowls_delivered_safely * P_d,
  have total_payment_before_deductions := payment_for_safely_delivered + F,
  have final_amount := total_payment_before_deductions - amount_owed_for_lost_broken,
  sorry
}

end Travis_payment_l618_618896


namespace two_digit_sabroso_numbers_l618_618954

theorem two_digit_sabroso_numbers :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ a b : ℕ, n = 10 * a + b ∧ (n + (10 * b + a) = k^2)} =
  {29, 38, 47, 56, 65, 74, 83, 92} :=
sorry

end two_digit_sabroso_numbers_l618_618954


namespace condition_neither_necessary_nor_sufficient_l618_618229

noncomputable def condition (a1 a2 b1 b2 c1 c2 : ℝ) : Prop :=
  (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2)

noncomputable def inequality_solution_set (a b c : ℝ) : set ℝ :=
  {x | a * x^2 + b * x + c > 0}

noncomputable def equivalent_solution_sets (a1 a2 b1 b2 c1 c2 : ℝ) : Prop :=
  (inequality_solution_set a1 b1 c1) = (inequality_solution_set a2 b2 c2)

theorem condition_neither_necessary_nor_sufficient (a1 a2 b1 b2 c1 c2 : ℝ) :
  ¬ (condition a1 a2 b1 b2 c1 c2 → equivalent_solution_sets a1 a2 b1 b2 c1 c2)
  ∧ ¬ (equivalent_solution_sets a1 a2 b1 b2 c1 c2 → condition a1 a2 b1 b2 c1 c2) :=
sorry

end condition_neither_necessary_nor_sufficient_l618_618229


namespace Kolya_made_the_mistake_l618_618001

def pencils_in_box (blue green : ℕ) : Prop :=
  (blue ≥ 4 ∨ blue < 4) ∧ (green ≥ 4 ∨ green < 4)

def boys_statements (blue green : ℕ) : Prop :=
  (Vasya : blue ≥ 4) ∧
  (Kolya : green ≥ 5) ∧
  (Petya : blue ≥ 3 ∧ green ≥ 4) ∧
  (Misha : blue ≥ 4 ∧ green ≥ 4)

theorem Kolya_made_the_mistake:
  ∀ {blue green : ℕ},
  pencils_in_box blue green →
  boys_statements blue green →
  ∃ (Vasya_truth Petya_truth Misha_truth : Prop),
  Vasya_truth ∧ Petya_truth ∧ Misha_truth ∧ ¬ Kolya_truth :=
begin
  sorry
end

end Kolya_made_the_mistake_l618_618001


namespace collinear_P_I_Q_l618_618383

theorem collinear_P_I_Q
  (ABC : Type*)
  [EuclideanDomain ABC]
  {A B C I D E F P Y Z Q : ABC}
  (h_incenter : is_incenter I A B C)
  (h_incircle_tangent_BC : is_tangent D (incircle I A B C) (line B C))
  (h_incircle_tangent_CA : is_tangent E (incircle I A B C) (line C A))
  (h_incircle_tangent_AB : is_tangent F (incircle I A B C) (line A B))
  (h_P_on_EF : lies_on P (line E F))
  (h_DP_perp_EF : is_perpendicular (line D P) (line E F))
  (h_BP_meet_AC_at_Y : meets (ray B P) (line A C) Y)
  (h_CP_meet_AB_at_Z : meets (ray C P) (line A B) Z)
  (h_Q_on_circumcircle_AYZ : lies_on Q (circumcircle A Y Z))
  (h_AQ_perp_BC : is_perpendicular (line A Q) (line B C)):
  collinear P I Q :=
sorry

end collinear_P_I_Q_l618_618383


namespace equal_triangles_in_square_have_specific_angles_l618_618898

-- Define geometric entities and angle measures
def square (A B C D M : Type) := 
  (AD_eq_AB : AD = AB) ∧
  (MD_eq_MB : MD = MB) ∧
  (MA_eq_common : MA = MA) ∧
  (MAD_eq_MAB : triangle.congruent (triangle.build M A D) (triangle.build M A B))

-- Define the proof of the angles in the triangles given the conditions
theorem equal_triangles_in_square_have_specific_angles :
  ∀ {A B C D M : Type}, 
  square A B C D M → -- The geometric conditions
  (angle A M D = 120) ∧ (angle A D M = 45) ∧ (angle D M A = 15) := 
by
  intro A B C D M h
  sorry

end equal_triangles_in_square_have_specific_angles_l618_618898


namespace max_value_sqrt_sum_l618_618226

theorem max_value_sqrt_sum (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_sum : a + b + c = 1) :
  sqrt (4 * a + 1) + sqrt (4 * b + 1) + sqrt (4 * c + 1) ≤ sqrt 21 :=
sorry

end max_value_sqrt_sum_l618_618226


namespace find_angle_A_correct_l618_618733

noncomputable def find_angle_A (a b : ℝ) (angle_B : ℝ) : ℝ :=
  let sin_B := Real.sin angle_B in
  let sin_A := (a * sin_B) / b in
  Real.arcsin sin_A

theorem find_angle_A_correct :
  find_angle_A (Real.sqrt 3) 1 (Real.pi / 6) = Real.pi / 3 :=
by
  sorry

end find_angle_A_correct_l618_618733


namespace length_of_AP_l618_618748

theorem length_of_AP (ABCD : set ℝ) (WXYZ : set ℝ) (A D AP : ℝ)
  (side_ABCD : ∀ A B C D ∈ ABCD, dist A B = 8 ∧ dist B C = 8 ∧ dist C D = 8 ∧ dist D A = 8)
  (ZY : dist Z Y = 12) (XY : dist X Y = 8) (perpendicular_AD_WX : ∀ A D W X, ∠ ADX = π / 2)
  (shaded_area_third_WXYZ_area : ∃ shaded_area, shaded_area = (1/3) * 96) : 
  AP = 4 :=
begin
  sorry,
end

end length_of_AP_l618_618748


namespace arithmetic_sequence_count_l618_618655

-- Define the set M
def M : Set ℕ := { k | k ≤ 9 }

-- Define what it means to be an arithmetic sequence with common difference -3
def is_arithmetic_seq (a b c : ℕ) : Prop :=
  a - b = 3 ∧ b - c = 3

-- The final theorem statement to be proved
theorem arithmetic_sequence_count :
  { (a, b, c) ∈ (M × M × M) | is_arithmetic_seq a b c }.to_finset.card = 4 :=
  sorry

end arithmetic_sequence_count_l618_618655


namespace value_of_3a_plus_6b_l618_618720

theorem value_of_3a_plus_6b (a b : ℝ) (h : a + 2 * b = 1) : 3 * a + 6 * b = 3 :=
sorry

end value_of_3a_plus_6b_l618_618720


namespace Kolya_made_the_mistake_l618_618004

def pencils_in_box (blue green : ℕ) : Prop :=
  (blue ≥ 4 ∨ blue < 4) ∧ (green ≥ 4 ∨ green < 4)

def boys_statements (blue green : ℕ) : Prop :=
  (Vasya : blue ≥ 4) ∧
  (Kolya : green ≥ 5) ∧
  (Petya : blue ≥ 3 ∧ green ≥ 4) ∧
  (Misha : blue ≥ 4 ∧ green ≥ 4)

theorem Kolya_made_the_mistake:
  ∀ {blue green : ℕ},
  pencils_in_box blue green →
  boys_statements blue green →
  ∃ (Vasya_truth Petya_truth Misha_truth : Prop),
  Vasya_truth ∧ Petya_truth ∧ Misha_truth ∧ ¬ Kolya_truth :=
begin
  sorry
end

end Kolya_made_the_mistake_l618_618004


namespace count_divisible_by_five_l618_618755

theorem count_divisible_by_five : 
  ∃ n : ℕ, (∀ x, 1 ≤ x ∧ x ≤ 1000 → (x % 5 = 0 → (n = 200))) :=
by
  sorry

end count_divisible_by_five_l618_618755


namespace largest_subset_no_four_times_another_l618_618964

theorem largest_subset_no_four_times_another :
  ∃ (S : set ℕ), S ⊆ {1, 2, ..., 150} ∧ (∀ (a b : ℕ), a ∈ S → b ∈ S → (a ≠ 4 * b ∧ b ≠ 4 * a)) ∧ (S.card = 141) :=
sorry

end largest_subset_no_four_times_another_l618_618964


namespace Travis_payment_l618_618895

variable (T L B F P_d P_l : ℕ) (H : T = 638 ∧ F = 100 ∧ P_d = 3 ∧ P_l = 4 ∧ L = 12 ∧ B = 15)

theorem Travis_payment (T L B F P_d P_l : ℕ) (H : T = 638 ∧ F = 100 ∧ P_d = 3 ∧ P_l = 4 ∧ L = 12 ∧ B = 15) : 
  let total_bowls_lost_broken := L + B in
  let amount_owed_for_lost_broken := total_bowls_lost_broken * P_l in
  let bowls_delivered_safely := T - total_bowls_lost_broken in
  let payment_for_safely_delivered := bowls_delivered_safely * P_d in
  let total_payment_before_deductions := payment_for_safely_delivered + F in
  let final_amount := total_payment_before_deductions - amount_owed_for_lost_broken in
  final_amount = 1825 :=
by {
  have total_bowls_lost_broken := L + B,
  have amount_owed_for_lost_broken := total_bowls_lost_broken * P_l,
  have bowls_delivered_safely := T - total_bowls_lost_broken,
  have payment_for_safely_delivered := bowls_delivered_safely * P_d,
  have total_payment_before_deductions := payment_for_safely_delivered + F,
  have final_amount := total_payment_before_deductions - amount_owed_for_lost_broken,
  sorry
}

end Travis_payment_l618_618895


namespace min_sum_l618_618274

variable (x y : ℝ)

open Real

theorem min_sum (h : log 3 x + log 3 y ≥ 8) : x + y = 162 :=
sorry

end min_sum_l618_618274


namespace value_of_expression_l618_618272

theorem value_of_expression (x y : ℝ) (h : x + 2 * y = 30) : (x / 5 + 2 * y / 3 + 2 * y / 5 + x / 3) = 16 :=
by
  sorry

end value_of_expression_l618_618272


namespace probability_Al_double_Bill_and_Bill_double_Cal_l618_618995

theorem probability_Al_double_Bill_and_Bill_double_Cal
  (Al Bill Cal : ℕ)
  (h1 : Al ≠ Bill ∧ Al ≠ Cal ∧ Bill ≠ Cal)
  (h2 : 1 ≤ Al ∧ Al ≤ 12)
  (h3 : 1 ≤ Bill ∧ Bill ≤ 12)
  (h4 : 1 ≤ Cal ∧ Cal ≤ 12) :
  (∃ N : ℕ, N = 1320 ∧ 
            (Cal = 1 ∧ Bill = 2 ∧ Al = 4 ∨
             Cal = 2 ∧ Bill = 4 ∧ Al = 8 ∨
             Cal = 3 ∧ Bill = 6 ∧ Al = 12) ∧ 
             probability_correct (3 / 1320) N) :=
begin
  sorry -- Proof is not required, only the statement.
end

noncomputable def probability_correct (prob : ℚ) (N : ℕ) : Prop := 
  prob = 1 / 440

end probability_Al_double_Bill_and_Bill_double_Cal_l618_618995


namespace sum_of_fourth_powers_less_than_150_l618_618471

-- Definition of the required sum
def sum_of_fourth_powers (n : ℕ) : ℕ :=
  (Finset.range n).filter (λ x => ∃ (k : ℕ), x = k^4).sum

-- Statement of the theorem to be proved
theorem sum_of_fourth_powers_less_than_150 : sum_of_fourth_powers 150 = 98 :=
by
  sorry

end sum_of_fourth_powers_less_than_150_l618_618471


namespace total_marbles_l618_618736

theorem total_marbles (number_of_boys : ℕ) (marbles_per_boy : ℕ) (H1 : number_of_boys = 11) (H2 : marbles_per_boy = 9) : number_of_boys * marbles_per_boy = 99 :=
by
  rw [H1, H2]
  rfl

end total_marbles_l618_618736


namespace difference_shares_l618_618556

theorem difference_shares (p q r x : ℕ) 
    (hp : p = 3 * x) 
    (hq : q = 7 * x) 
    (hr : r = 12 * x)
    (h_diff_pq : q - p = 2400) : 
    q - r = 3000 := 
begin
  sorry
end

end difference_shares_l618_618556


namespace little_ma_probability_l618_618291

theorem little_ma_probability :
  let options : List Char := ['A', 'B', 'C', 'D']
  let total_outcomes := options.length
  let favorable_outcomes := 1
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = 1 / 4 :=
by
  sorry

end little_ma_probability_l618_618291


namespace find_DZ_l618_618209

-- Assuming the given vertices, edges, points, and distances
variable (A B C D A1 B1 C1 D1 X Y Z : Point) -- Parallelepiped vertices and points
variable [geometry] -- Assuming some basic geometric setup

/-- Definition for given distances and conditions -/
def conditions : Prop :=
  (distance A1 X = 5) ∧
  (distance B Y = 3) ∧
  (distance B1 C1 = 14) ∧
  line (A1, D1) ∧
  line (B, C) ∧
  line (D, A)

theorem find_DZ (h : conditions) : distance D Z = 20 :=
  sorry

end find_DZ_l618_618209


namespace number_of_small_companies_l618_618286

theorem number_of_small_companies
  (large_companies : ℕ)
  (medium_companies : ℕ)
  (inspected_companies : ℕ)
  (inspected_medium_companies : ℕ)
  (total_inspected_companies : ℕ)
  (small_companies : ℕ)
  (inspection_fraction : ℕ → ℚ)
  (proportion : inspection_fraction 20 = 1 / 4)
  (H1 : large_companies = 4)
  (H2 : medium_companies = 20)
  (H3 : inspected_medium_companies = 5)
  (H4 : total_inspected_companies = 40)
  (H5 : inspected_companies = total_inspected_companies - large_companies - inspected_medium_companies)
  (H6 : small_companies = inspected_companies * 4)
  (correct_result : small_companies = 136) :
  small_companies = 136 :=
by sorry

end number_of_small_companies_l618_618286


namespace kolya_is_wrong_l618_618029

def pencils_problem_statement (at_least_four_blue : Prop) 
                              (at_least_five_green : Prop) 
                              (at_least_three_blue_and_four_green : Prop) 
                              (at_least_four_blue_and_four_green : Prop) : 
                              Prop :=
  ∃ (B G : ℕ), -- B represents the number of blue pencils, G represents the number of green pencils
    ((B ≥ 4) ∧ (G ≥ 4)) ∧ -- Vasya's statement (at least 4 blue), Petya's and Misha's combined statement (at least 4 green)
    at_least_four_blue ∧ -- Vasya's statement (there are at least 4 blue pencils)
    (at_least_five_green ↔ G ≥ 5) ∧ -- Kolya's statement (there are at least 5 green pencils)
    at_least_three_blue_and_four_green ∧ -- Petya's statement (at least 3 blue and 4 green)
    at_least_four_blue_and_four_green -- Misha's statement (at least 4 blue and 4 green)

theorem kolya_is_wrong (at_least_four_blue : Prop) 
                        (at_least_five_green : Prop) 
                        (at_least_three_blue_and_four_green : Prop) 
                        (at_least_four_blue_and_four_green : Prop) : 
                        pencils_problem_statement at_least_four_blue 
                                                  at_least_five_green 
                                                  at_least_three_blue_and_four_green 
                                                  at_least_four_blue_and_four_green :=
sorry

end kolya_is_wrong_l618_618029


namespace Alice_needs_to_add_l618_618559

noncomputable theory

def stamps_of_Alice := 62
def stamps_of_Danny := stamps_of_Alice + 3
def stamps_of_Peggy := 1.5 * stamps_of_Danny
def stamps_of_Ernie := 2.5 * stamps_of_Peggy
def stamps_of_Bert := 4.5 * stamps_of_Ernie

theorem Alice_needs_to_add :
  62 + 1035 = Nat.ceil (4.5 * (2.5 * (1.5 * (62 + 3)))) :=
by
  sorry

end Alice_needs_to_add_l618_618559


namespace regular_25gon_symmetry_l618_618156

theorem regular_25gon_symmetry :
  let L := 25
  let R := (360 : ℝ) / 25
  L + (R / 2) = 32.2 :=
by
  let L := 25
  let R := (360 : ℝ) / 25
  have h1 : (R : ℝ) = 14.4 := by sorry
  have h2 : (R / 2 : ℝ) = 7.2 := by sorry
  show (L + (R / 2) : ℝ) = 32.2 from by do
    unfold L R
    sorry

end regular_25gon_symmetry_l618_618156


namespace domain_of_f_l618_618630

def q (x : ℝ) : ℝ := x^2 - 5 * x + 6

def f (x : ℝ) : ℝ := (x^3 - 3 * x^2 + 5 * x - 2) / q x 

theorem domain_of_f :
  ∀ x : ℝ, q x ≠ 0 ↔ (x < 2 ∨ (2 < x ∧ x < 3) ∨ 3 < x) := by
  sorry

end domain_of_f_l618_618630


namespace largest_subset_size_l618_618986

theorem largest_subset_size :
  ∃ S : Set ℕ, S ⊆ {i | 1 ≤ i ∧ i ≤ 150} ∧ 
  (∀ (a b : ℕ), a ∈ S → b ∈ S → ¬ (a = 4 * b ∨ b = 4 * a)) ∧ 
  S.card = 141 :=
sorry

end largest_subset_size_l618_618986


namespace digits_of_2_120_l618_618182

theorem digits_of_2_120 (h : ∀ n : ℕ, (10 : ℝ)^(n - 1) ≤ (2 : ℝ)^200 ∧ (2 : ℝ)^200 < (10 : ℝ)^n → n = 61) :
  ∀ m : ℕ, (10 : ℝ)^(m - 1) ≤ (2 : ℝ)^120 ∧ (2 : ℝ)^120 < (10 : ℝ)^m → m = 37 :=
by
  sorry

end digits_of_2_120_l618_618182


namespace compute_ratio_sum_eq_14_l618_618362

noncomputable theory

open Real

variables (x y : ℝ) (θ : ℝ)
variables (h_x_pos : 0 < x) (h_y_pos : 0 < y)
variable (h_θ : ∀ n : ℤ, θ ≠ (π / 2) * n)
variable (h_eq1 : sin θ / x = cos θ / y)
variable (h_eq2 : cos θ ^ 4 / x ^ 4 + sin θ ^ 4 / y ^ 4 = 97 * sin (2 * θ) / (x ^ 3 * y + y ^ 3 * x))

theorem compute_ratio_sum_eq_14 : x / y + y / x = 14 :=
sorry

end compute_ratio_sum_eq_14_l618_618362


namespace solve_for_x_l618_618847

theorem solve_for_x : ∃ x : ℚ, (x - 30) / 3 = (4 - 3 * x) / 7 ∧ x = 111 / 8 :=
by
  use (111 / 8)
  split
  { calc 
      (111 / 8 - 30) / 3
        = -69 / 8 / 3 : by norm_num
      ... = -69 / 24 : by norm_num
      ... = -23 / 8 / 7 : by norm_num
      ... = (4 - 3 * 111 / 8) / 7 : by norm_num
      ... = (4 - 333 / 8) / 7 : by norm_num
      ... = (32 / 8 - 333 / 8) / 7 : by norm_num
      ... = (-301 / 8) / 7 : by norm_num
      ... = -301 / 56 : by norm_num },

  { exact rfl }

end solve_for_x_l618_618847


namespace negation_equivalence_l618_618061

variable {α : Type} (S : set α)
variable (x : α)

-- The statement, let's assume the property p(x) is x^3 ≠ 0.
def prop (x : α) : Prop := x^3 ≠ 0

-- Statement of the proof problem
theorem negation_equivalence (S : set α) :
  (∀ x ∈ S, prop x) ↔ (∃ x ∈ S, ¬ prop x) := 
sorry

end negation_equivalence_l618_618061


namespace P_is_perfect_square_l618_618653

theorem P_is_perfect_square (m : ℤ) : 
  let P := 1 + 2 * m + 3 * m^2 + 4 * m^3 + 5 * m^4 + 4 * m^5 + 3 * m^6 + 2 * m^7 + m^8
  (∃ k : ℤ, P = k * k) :=
by 
  let k := 1 + m + m^2 + m^3 + m^4
  have h : P = k * k
  sorry

end P_is_perfect_square_l618_618653


namespace inequality_proof_l618_618406

theorem inequality_proof
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a^3 + b^3 + c^3 = 3) :
  (1 / (a^4 + 3) + 1 / (b^4 + 3) + 1 / (c^4 + 3) >= 3 / 4) :=
by
  sorry

end inequality_proof_l618_618406


namespace no_real_solutions_l618_618625

theorem no_real_solutions (x : ℝ) : x^2 + 4 * x + 4 * x * real.sqrt (x + 3) ≠ 17 :=
by
  sorry


end no_real_solutions_l618_618625


namespace opposite_reciprocal_abs_value_l618_618225

theorem opposite_reciprocal_abs_value (a b c d m : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) (h3 : abs m = 3) : 
  (a + b) / m + c * d + m = 4 ∨ (a + b) / m + c * d + m = -2 := by 
  sorry

end opposite_reciprocal_abs_value_l618_618225


namespace find_function_expression_l618_618691

theorem find_function_expression (f : ℝ → ℝ) (h : ∀ x : ℝ, f(x+1) = x^2 + 2x + 2) : 
  ∀ x : ℝ, f x = x^2 + 1 := 
sorry

end find_function_expression_l618_618691


namespace train_speed_A_to_B_l618_618992

-- Define the constants
def distance : ℝ := 480
def return_speed : ℝ := 120
def return_time_longer : ℝ := 1

-- Define the train's speed function on its way from A to B
noncomputable def train_speed : ℝ := distance / (4 - return_time_longer) -- This simplifies directly to 160 based on the provided conditions.

-- State the theorem
theorem train_speed_A_to_B :
  distance / train_speed + return_time_longer = distance / return_speed :=
by
  -- Result follows from the given conditions directly
  sorry

end train_speed_A_to_B_l618_618992


namespace g_of_f_three_l618_618370

def f (x : ℝ) : ℝ := x^3 - 2
def g (x : ℝ) : ℝ := 3 * x^2 + x + 2

theorem g_of_f_three : g (f 3) = 1902 :=
by
  sorry

end g_of_f_three_l618_618370


namespace simplify_vec_expr1_simplify_vec_expr2_l618_618845

variables {ℝ : Type} [Field ℝ] [AddGroup ℝ] {n : Type} [AddCommGroup n] [Module ℝ n]

/-- Simplify vector expression 1 -/
theorem simplify_vec_expr1 (BA BC ED EC DA : n) :
  (BA - BC) - (ED - EC) = DA := 
sorry

/-- Simplify vector expression 2 -/
theorem simplify_vec_expr2 (AC BO OA DC DO OB : n) :
  (AC + BO + OA) - (DC - DO - OB) = 0 :=
sorry

end simplify_vec_expr1_simplify_vec_expr2_l618_618845


namespace perimeter_triangle_lt_ABC_l618_618806

theorem perimeter_triangle_lt_ABC
  (A B C A₁ B₁ C₁ : Point)
  (λ : ℝ)
  (hλ : 1 / 2 < λ ∧ λ < 1)
  (hAB₁ : dist A B₁ = λ * dist B C)
  (hCB₁ : dist C B₁ = λ * dist C A)
  (hAC₁ : dist A C₁ = λ * dist A B) :
  (perimeter A₁ B₁ C₁ < λ * perimeter A B C) :=
sorry

end perimeter_triangle_lt_ABC_l618_618806


namespace probability_X_equals_Y_l618_618134

theorem probability_X_equals_Y :
  let f := λ x : ℝ, cos (cos x)
  let domain := Icc (-5 * π) (5 * π)
  let prob := (11 : ℝ) / (14641 : ℝ)
  (∀ x y ∈ domain, f x = f y → x = y) →
  ((∃ X Y : domain, X = Y) → X ∈ domain ∧ Y ∈ domain ∧ f X = f Y) →
  ((11 : ℕ) / ((11 ^ 2)^2) = prob) :=
by
  sorry

end probability_X_equals_Y_l618_618134


namespace outfits_not_same_color_l618_618709

/--
Given:
- 7 shirts, 7 pairs of pants, and 7 hats.
- Each item comes in 7 colors (one of each item of each color).
- No outfit is allowed where all 3 items are the same color.

Prove:
The number of possible outfits where not all items are the same color is 336.
-/
theorem outfits_not_same_color : 
  let total_outfits := 7 * 7 * 7 in
  let same_color_outfits := 7 in
  total_outfits - same_color_outfits = 336 :=
by
  let total_outfits := 7 * 7 * 7
  let same_color_outfits := 7
  have h1 : total_outfits = 343 := by norm_num
  have h2 : total_outfits - same_color_outfits = 336 := by norm_num
  exact h2

end outfits_not_same_color_l618_618709


namespace positive_integer_perfect_square_l618_618604

theorem positive_integer_perfect_square (n : ℕ) (h1: n > 0) (h2 : ∃ k : ℕ, n^2 - 19 * n - 99 = k^2) : n = 199 :=
sorry

end positive_integer_perfect_square_l618_618604


namespace power_quotient_l618_618581

theorem power_quotient (a m n : ℕ) (h_a : a = 19) (h_m : m = 11) (h_n : n = 8) : a^m / a^n = 6859 := by
  sorry

end power_quotient_l618_618581


namespace find_a_b_l618_618793

-- Define the conditions
variables (a b : ℕ)
hypothesis ha_prime : Prime a
hypothesis hb_pos : 0 < b

-- The main theorem to prove
theorem find_a_b (ha : a = 251) (hb : b = 7) : 
  9 * (2 * a + b)^2 = 509 * (4 * a + 511 * b) := 
by sorry

end find_a_b_l618_618793


namespace decreasing_interval_of_f_x_minus_1_l618_618201

def decreasing_interval_of_shifted_function (f : ℝ → ℝ) (shift : ℝ) (I : set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f (x - shift) > f (y - shift)

theorem decreasing_interval_of_f_x_minus_1 :
  let f := λ x : ℝ, x^2 + 2 * x - 5 in
  decreasing_interval_of_shifted_function f 1 (set.Iic 0) :=
by
  let f := λ x : ℝ, x^2 + 2 * x - 5
  -- sorry


end decreasing_interval_of_f_x_minus_1_l618_618201


namespace min_value_range_l618_618248

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem min_value_range (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 2 * a^2 - a - 1) → (0 < a ∧ a ≤ 1) :=
by 
  sorry

end min_value_range_l618_618248


namespace shaded_area_l618_618750

-- Define the problem in Lean
theorem shaded_area (area_large_square area_small_square : ℝ) (H_large_square : area_large_square = 10) (H_small_square : area_small_square = 4) (diagonals_contain : True) : 
  (area_large_square - area_small_square) / 4 = 1.5 :=
by
  sorry -- proof not required

end shaded_area_l618_618750


namespace max_value_quadratic_l618_618609

theorem max_value_quadratic :
  (∃ x : ℝ, ∀ y : ℝ, -3*y^2 + 9*y + 24 ≤ -3*x^2 + 9*x + 24) ∧ (∃ x : ℝ, x = 3/2) :=
sorry

end max_value_quadratic_l618_618609


namespace first_house_bottles_l618_618120

theorem first_house_bottles (total_bottles : ℕ) 
  (cider_only : ℕ) (beer_only : ℕ) (half : ℕ → ℕ)
  (mixture : ℕ)
  (half_cider_bottles : ℕ)
  (half_beer_bottles : ℕ)
  (half_mixture_bottles : ℕ) : 
  total_bottles = 180 →
  cider_only = 40 →
  beer_only = 80 →
  mixture = total_bottles - (cider_only + beer_only) →
  half c = c / 2 →
  half_cider_bottles = half cider_only →
  half_beer_bottles = half beer_only →
  half_mixture_bottles = half mixture →
  half_cider_bottles + half_beer_bottles + half_mixture_bottles = 90 :=
by
  intros h_tot h_cid h_beer h_mix h_half half_cid half_beer half_mix
  sorry

end first_house_bottles_l618_618120


namespace f_even_l618_618782

variable (g : ℝ → ℝ)

def is_odd (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = -g x

def f (x : ℝ) := |g (x^2)|

theorem f_even (h_g_odd : is_odd g) : ∀ x : ℝ, f g x = f g (-x) :=
by
  intro x
  -- Proof can be added here
  sorry

end f_even_l618_618782


namespace number_of_buses_l618_618101

-- Define the conditions
def vans : ℕ := 9
def people_per_van : ℕ := 8
def people_per_bus : ℕ := 27
def total_people : ℕ := 342

-- Translate the mathematical proof problem
theorem number_of_buses : ∃ buses : ℕ, (vans * people_per_van + buses * people_per_bus = total_people) ∧ (buses = 10) :=
by
  -- calculations to prove the theorem
  sorry

end number_of_buses_l618_618101


namespace largest_subset_no_member_is_4_times_another_l618_618980

-- Define the predicate that characterizes the subset
def valid_subset (S : Set ℕ) : Prop :=
  ∀ (x ∈ S) (y ∈ S), x ≠ 4 * y ∧ y ≠ 4 * x

-- Define the set of integers from 1 to 150
def range_1_to_150 : Set ℕ := {n | 1 ≤ n ∧ n ≤ 150}

-- State the theorem
theorem largest_subset_no_member_is_4_times_another :
  ∃ S ⊆ range_1_to_150, valid_subset S ∧ S.card = 140 := sorry

end largest_subset_no_member_is_4_times_another_l618_618980


namespace calculate_green_paint_l618_618656

theorem calculate_green_paint {green white : ℕ} (ratio_white_to_green : 5 * green = 3 * white) (use_white_paint : white = 15) : green = 9 :=
by
  sorry

end calculate_green_paint_l618_618656


namespace triangle_angle_bisector_ratio_l618_618047

theorem triangle_angle_bisector_ratio
  (A B C M : Type)
  [triangle A B C]
  (hAB : AB = 4) (hAC : AC = 5) (hBC : BC = 6)
  (hangle_bisector : angle_bisector (∠BAC) M B C) :
  100 * (AM / CM) = 80 :=
by
  sorry

end triangle_angle_bisector_ratio_l618_618047


namespace range_of_m_l618_618698

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - ((Real.exp x - 1) / (Real.exp x + 1))

theorem range_of_m (m : ℝ) (h : f (4 - m) - f m ≥ 8 - 4 * m) : 2 ≤ m := by
  sorry

end range_of_m_l618_618698


namespace inscribed_circle_ratio_l618_618516

theorem inscribed_circle_ratio (a b c u v : ℕ) (h_triangle : a = 10 ∧ b = 24 ∧ c = 26) 
    (h_tangent_segments : u < v) (h_side_sum : u + v = a) : u / v = 2 / 3 :=
by
    sorry

end inscribed_circle_ratio_l618_618516


namespace f_is_even_l618_618784

-- Given an odd function g
def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g (x)

-- Define the function f as given by the problem
def f (g : ℝ → ℝ) (x : ℝ) : ℝ :=
  abs (g (x^2))

-- The theorem stating that f is an even function
theorem f_is_even (g : ℝ → ℝ) (h_odd : is_odd_function g) : ∀ x, f g x = f g (-x) :=
by
  sorry

end f_is_even_l618_618784


namespace digits_of_2_120_l618_618183

theorem digits_of_2_120 (h : ∀ n : ℕ, (10 : ℝ)^(n - 1) ≤ (2 : ℝ)^200 ∧ (2 : ℝ)^200 < (10 : ℝ)^n → n = 61) :
  ∀ m : ℕ, (10 : ℝ)^(m - 1) ≤ (2 : ℝ)^120 ∧ (2 : ℝ)^120 < (10 : ℝ)^m → m = 37 :=
by
  sorry

end digits_of_2_120_l618_618183


namespace count_of_D_eq_2_l618_618191

def D (n : ℕ) : ℕ := -- Implementation of D(n) goes here
  sorry

theorem count_of_D_eq_2 : {n : ℕ | n ≤ 1000 ∧ D n = 2}.card = 36 :=
by
  sorry

end count_of_D_eq_2_l618_618191


namespace initial_black_water_bottles_l618_618883

-- Define the conditions
variables (red black blue taken left total : ℕ)
variables (hred : red = 2) (hblue : blue = 4) (htaken : taken = 5) (hleft : left = 4)

-- State the theorem with the correct answer given the conditions
theorem initial_black_water_bottles : (red + black + blue = taken + left) → black = 3 :=
by
  intros htotal
  rw [hred, hblue, htaken, hleft] at htotal
  sorry

end initial_black_water_bottles_l618_618883


namespace square_in_triangle_ratio_l618_618746

variables {x a b : ℝ}

def is_isosceles_right_triangle (A B O : ℝ) : Prop :=
A = B ∧ 2 * (A * B) = 4

theorem square_in_triangle_ratio
  (h1 : is_isosceles_right_triangle A B)
  (h2 : ∃ (P Q S : ℝ), P = A ∧ Q = B)
  (h3 : ∀ P Q : ℝ, OP = a ∧ OQ = b)
  (h4 : s = a + b)
  (h5 : s^2 = (2 / 5) * (x^2 / 2))
: a / b = 2 / 1 :=
begin
  sorry
end

end square_in_triangle_ratio_l618_618746


namespace finite_group_moves_l618_618128

noncomputable theory

variables {α : Type*} [fintype α] (G : finset (finset α)) (r : α → ℝ)
  [∀ x, is_fraction r x] (n : ℕ) (G_nempty : ∀ Gk ∈ G, Gk.nonempty)
  (rating_pos : ∀ x ∈ ⋃₀ G, 0 < r x) (relative_rating : ℝ)
  (can_move : ∀ x ∈ ⋃₀ G, ∃ H ∈ G, (∀ Gk ∈ G, relative_rating x Gk < relative_rating x H))

theorem finite_group_moves : 
  ∀ G : finset (finset α), ∀ r : α → ℝ, 
  (∀ x ∈ ⋃₀ G, 0 < r x) → 
  (∃ n > 1, G.card = n ∧ ∀ Gk ∈ G, Gk.nonempty) → 
  (∀ x ∈ ⋃₀ G, ∃ H ∈ G, (∀ Gk ∈ G, relative_rating x Gk < relative_rating x H)) → 
  ∃ m : ℕ, ∀ Gm, can_move x_1 Gm → can_move x_2 Gm → ⋯ → can_move x_m Gm → m < n := 
sorry

end finite_group_moves_l618_618128


namespace expression_a_equals_half_expression_c_equals_half_l618_618546

theorem expression_a_equals_half :
  (A : ℝ) = (1 / 2) :=
by
  let A := (Real.sqrt 2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180))
  sorry

theorem expression_c_equals_half :
  (C : ℝ) = (1 / 2) :=
by
  let C := (Real.tan (22.5 * Real.pi / 180)) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)
  sorry

end expression_a_equals_half_expression_c_equals_half_l618_618546


namespace find_n_in_sequence_l618_618254

theorem find_n_in_sequence (n : ℕ) : (∀ n, a_n = 2 * real.sqrt 17) → (∀ n, a_n = real.sqrt (3 * n - 1)) → n = 23 := 
sorry

end find_n_in_sequence_l618_618254


namespace hoopit_toes_l618_618402

theorem hoopit_toes (h : ℕ) : 
  (7 * (4 * h) + 8 * (2 * 5) = 164) -> h = 3 :=
by
  sorry

end hoopit_toes_l618_618402


namespace shorter_diagonal_of_trapezoid_l618_618046

theorem shorter_diagonal_of_trapezoid
  (EF GH : ℝ)
  (side1 side2 : ℝ)
  (right_angle : ∀ (x y : ℝ), angle x y = 90)
  (h_diagonal : ℝ) 
  (correct_diagonal : h_diagonal = 15 + 2 * Real.sqrt 2)
  : ∃ diagonal : ℝ, diagonal = h_diagonal :=
by
  have hyp1 : EF = 40 := sorry
  have hyp2 : GH = 28 := sorry
  have hyp3 : side1 = 15 := sorry
  have hyp4 : side2 = 17 := sorry
  have hyp5 : ∀ (s1 s2 : ℝ), right_angle s1 s2 = 90 := sorry
  exact ⟨h_diagonal, correct_diagonal⟩

end shorter_diagonal_of_trapezoid_l618_618046


namespace sum_of_squares_of_roots_l618_618153

theorem sum_of_squares_of_roots 
  (r s t : ℝ) 
  (hr : y^3 - 8 * y^2 + 9 * y - 2 = 0) 
  (hs : y ≥ 0) 
  (ht : y ≥ 0):
  r^2 + s^2 + t^2 = 46 :=
sorry

end sum_of_squares_of_roots_l618_618153


namespace number_of_arrangements_l618_618139

theorem number_of_arrangements :
  let all_numbers := [1, 2, 3, 4, 5, 6],
      valid_arrangements := filter (λ (l : List ℕ), l.length = 6 ∧
        l.get! 0 ≠ 1 ∧ l.get! 2 ≠ 3 ∧ l.get! 4 ≠ 5 ∧
        l.get! 0 < l.get! 2 ∧ l.get! 2 < l.get! 4) (all_permutations all_numbers)
  in valid_arrangements.length = 30 := sorry

end number_of_arrangements_l618_618139


namespace Xiaoyong_age_solution_l618_618489

theorem Xiaoyong_age_solution :
  ∃ (x y : ℕ), 1 ≤ y ∧ y < x ∧ x < 20 ∧ 2 * x + 5 * y = 97 ∧ x = 16 ∧ y = 13 :=
by
  -- You should provide a suitable proof here
  sorry

end Xiaoyong_age_solution_l618_618489


namespace ultra_prime_classification_l618_618373

-- Define the sum of divisors function
def divisors_sum (n : ℕ) : ℕ :=
  (Finset.range (n+1)).filter (λ d, n % d = 0).sum id

-- Define the function g as given in the problem
def g (n : ℕ) : ℕ := n + divisors_sum n

-- Define the ultra-prime property
def ultra_prime (n : ℕ) : Prop := divisors_sum (g n) = 2 * n + 3

-- Theorem stating the solution
theorem ultra_prime_classification : ∀ n, n < 100 → ultra_prime n ↔ n = 1 :=
by
  sorry

end ultra_prime_classification_l618_618373


namespace alpha_centauri_puzzle_l618_618932

open Nat

def max_number_count (A B N : ℕ) : ℕ :=
  let pairs_count := ((B - A) / N) * (N / 2)
  pairs_count + 1  -- Adding the single remainder

theorem alpha_centauri_puzzle :
  let A := 1353
  let B := 2134
  let N := 11
  max_number_count A B N = 356 :=
by 
  let A := 1353
  let B := 2134
  let N := 11
  -- Using the helper function max_number_count defined above
  have h : max_number_count A B N = 356 := 
   by sorry  -- skips detailed proof for illustrative purposes
  exact h


end alpha_centauri_puzzle_l618_618932


namespace Julie_work_hours_per_week_l618_618767

variable (hours_per_week_summer : ℕ) (weeks_summer : ℕ)
variable (earnings_summer : ℕ)
variable (weeks_school_year : ℕ)
variable (earnings_school_year : ℕ)

theorem Julie_work_hours_per_week :
  hours_per_week_summer = 40 →
  weeks_summer = 10 →
  earnings_summer = 4000 →
  weeks_school_year = 40 →
  earnings_school_year = 4000 →
  (∀ rate_per_hour, rate_per_hour = earnings_summer / (hours_per_week_summer * weeks_summer) →
  (earnings_school_year / (weeks_school_year * rate_per_hour) = 10)) :=
by intros h1 h2 h3 h4 h5 rate_per_hour hr; sorry

end Julie_work_hours_per_week_l618_618767


namespace rainy_days_l618_618081

namespace Mo

def drinks (R NR n : ℕ) :=
  -- Condition 3: Total number of days in the week equation
  R + NR = 7 ∧
  -- Condition 1-2: Total cups of drinks equation
  n * R + 3 * NR = 26 ∧
  -- Condition 4: Difference in cups of tea and hot chocolate equation
  3 * NR - n * R = 10

theorem rainy_days (R NR n : ℕ) (h: drinks R NR n) : 
  R = 1 := sorry

end Mo

end rainy_days_l618_618081


namespace olivers_friend_gave_l618_618812

variable (initial_amount saved_amount spent_frisbee spent_puzzle final_amount : ℕ) 

theorem olivers_friend_gave (h1 : initial_amount = 9) 
                           (h2 : saved_amount = 5) 
                           (h3 : spent_frisbee = 4) 
                           (h4 : spent_puzzle = 3) 
                           (h5 : final_amount = 15) : 
                           final_amount - (initial_amount + saved_amount - (spent_frisbee + spent_puzzle)) = 8 := 
by 
  sorry

end olivers_friend_gave_l618_618812


namespace range_of_a_l618_618451

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 2 * x^2 - a * x + 1 > 0) ↔ -2 * Real.sqrt 2 < a ∧ a < 2 * Real.sqrt 2 :=
by
  sorry

end range_of_a_l618_618451


namespace product_divisible_by_12_l618_618409

theorem product_divisible_by_12 (a b c d : ℤ) : 
  12 ∣ ((b - a) * (c - a) * (d - a) * (b - c) * (d - c) * (d - b)) :=
  sorry

end product_divisible_by_12_l618_618409


namespace cube_vertices_count_l618_618708

-- Given conditions
variables (a : ℝ) (x y z : ℝ)

-- Definition of the cube
def cube := {p : ℝ × ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ a ∧ 0 ≤ p.2 ∧ p.2 ≤ a ∧ 0 ≤ p.3 ∧ p.3 ≤ a}

-- Proof module
theorem cube_vertices_count (a : ℝ) : 
    (∃ vertices : List (ℝ × ℝ × ℝ), vertices.length = 8 ∧ 
        vertices = [(0, 0, 0), (0, 0, a), (0, a, 0), (a, 0, 0), (0, a, a), (a, 0, a), (a, a, 0), (a, a, a)]) 
:= by
    sorry

end cube_vertices_count_l618_618708


namespace sally_quarters_l618_618833

noncomputable def initial_quarters : ℕ := 760
noncomputable def spent_quarters : ℕ := 418
noncomputable def remaining_quarters : ℕ := 342

theorem sally_quarters : initial_quarters - spent_quarters = remaining_quarters :=
by sorry

end sally_quarters_l618_618833


namespace largest_subset_no_member_is_4_times_another_l618_618982

-- Define the predicate that characterizes the subset
def valid_subset (S : Set ℕ) : Prop :=
  ∀ (x ∈ S) (y ∈ S), x ≠ 4 * y ∧ y ≠ 4 * x

-- Define the set of integers from 1 to 150
def range_1_to_150 : Set ℕ := {n | 1 ≤ n ∧ n ≤ 150}

-- State the theorem
theorem largest_subset_no_member_is_4_times_another :
  ∃ S ⊆ range_1_to_150, valid_subset S ∧ S.card = 140 := sorry

end largest_subset_no_member_is_4_times_another_l618_618982


namespace avg_annual_reduction_l618_618095

noncomputable def reduction_equation_correct (x : ℝ) : Prop :=
  (1 - x)^2 = 0.64

theorem avg_annual_reduction (x : ℝ) (hx : reduction_equation_correct x) : 
  (1 - x)^2 = 1 - 0.36 :=
by
  rw [reduction_equation_correct x] at hx
  rw [one_sub] at hx
  exact hx

end avg_annual_reduction_l618_618095


namespace circus_total_tickets_sold_l618_618859

-- Definitions from the conditions
def revenue_total : ℕ := 2100
def lower_seat_tickets_sold : ℕ := 50
def price_lower : ℕ := 30
def price_upper : ℕ := 20

-- Definition derived from the conditions
def tickets_total (L U : ℕ) : ℕ := L + U

-- The theorem we need to prove
theorem circus_total_tickets_sold (L U : ℕ) (hL: L = lower_seat_tickets_sold)
    (h₁ : price_lower * L + price_upper * U = revenue_total) : 
    tickets_total L U = 80 :=
by
  sorry  -- Proof omitted

end circus_total_tickets_sold_l618_618859


namespace problem_solution_l618_618819

noncomputable def pairwise_sums : List ℕ := [5, 8, 9, 13, 14, 14, 15, 17, 18, 23]

def find_integers_and_product : Prop := 
  ∃ a b c d e : ℕ, 
  a + b + c + d + e = 34 ∧ 
  {a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e}.toFinset = pairwise_sums.toFinset ∧ 
  a * b * c * d * e = 4752

theorem problem_solution : find_integers_and_product := sorry

end problem_solution_l618_618819


namespace power_quotient_l618_618582

theorem power_quotient (a m n : ℕ) (h_a : a = 19) (h_m : m = 11) (h_n : n = 8) : a^m / a^n = 6859 := by
  sorry

end power_quotient_l618_618582


namespace set_union_complement_l618_618808

-- Define the universal set U
def U := {x : ℕ | x < 4}

-- Define sets A and B
def A := {0, 1, 2}
def B := {2, 3}

-- Define complement of A in U
def complement_U_A := {x ∈ U | x ∉ A}

-- Statement of the theorem to be proved
theorem set_union_complement :
  B ∪ complement_U_A = {2, 3} :=
by
  sorry

end set_union_complement_l618_618808


namespace tangent_line_parabola_l618_618251

theorem tangent_line_parabola (P : ℝ × ℝ) (hP : P = (-1, 0)) :
  ∀ x y : ℝ, (y^2 = 4 * x) ∧ (P = (-1, 0)) → (x + y + 1 = 0) ∨ (x - y + 1 = 0) := by
  sorry

end tangent_line_parabola_l618_618251


namespace min_value_abs_sum_exists_min_value_abs_sum_l618_618438

theorem min_value_abs_sum (x : ℝ) : |x - 1| + |x - 4| ≥ 3 :=
by sorry

theorem exists_min_value_abs_sum : ∃ x : ℝ, |x - 1| + |x - 4| = 3 :=
by sorry

end min_value_abs_sum_exists_min_value_abs_sum_l618_618438


namespace minimize_f_l618_618610

def f (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 7

theorem minimize_f : ∃ y : ℝ, y = 1.5 ∧ ∀ y' : ℝ, f(1.5) ≤ f(y') := sorry

end minimize_f_l618_618610


namespace infinite_pairs_for_equation_l618_618829

theorem infinite_pairs_for_equation :
  ∃ᶠ (a b : ℤ) in filter.cofinite ℤ, ∃ (x y : ℝ),
  x ≠ y ∧ x * y = 1 ∧ x ^ 2012 = a * x + b ∧ y ^ 2012 = a * y + b :=
sorry

end infinite_pairs_for_equation_l618_618829


namespace compound_interest_principal_l618_618060

theorem compound_interest_principal 
    (CI : Real)
    (r : Real)
    (n : Nat)
    (t : Nat)
    (A : Real)
    (P : Real) :
  CI = 945.0000000000009 →
  r = 0.10 →
  n = 1 →
  t = 2 →
  A = P * (1 + r / n) ^ (n * t) →
  CI = A - P →
  P = 4500.0000000000045 :=
by intros
   sorry

end compound_interest_principal_l618_618060


namespace expression_a_equals_half_expression_c_equals_half_l618_618547

theorem expression_a_equals_half :
  (A : ℝ) = (1 / 2) :=
by
  let A := (Real.sqrt 2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180))
  sorry

theorem expression_c_equals_half :
  (C : ℝ) = (1 / 2) :=
by
  let C := (Real.tan (22.5 * Real.pi / 180)) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)
  sorry

end expression_a_equals_half_expression_c_equals_half_l618_618547


namespace group_members_l618_618950

theorem group_members (n : ℕ) (hn : n * n = 1369) : n = 37 :=
by
  sorry

end group_members_l618_618950


namespace value_of_a_l618_618299

-- Define the curves in terms of the conditions
def C1 (ρ θ : ℝ) : Prop :=
  ρ * (sqrt 2 * cos θ + sin θ) = 1

def C2 (ρ a : ℝ) [lt : fact (0 < a)] : Prop :=
  ρ = a

-- The point on the polar axis means y=0, which translates to θ=0 or θ=π in polar coordinates.
-- Cartesian conversion for verification
def cartesian_intersection (a : ℝ) : Prop :=
  let x := sqrt 2 / 2 in
  x^2 = a^2

-- The theorem statement
theorem value_of_a (a : ℝ) [lt : fact (0 < a)] : 
  C1 (sqrt 2 / 2) 0 ∧ C2 (sqrt 2 / 2) a → a = sqrt 2 / 2 :=
by
  intros h
  -- Proof steps would be filled in here
  sorry

end value_of_a_l618_618299


namespace no_option_is_perfect_square_l618_618484

theorem no_option_is_perfect_square : 
  ¬ (∃ n, (n = 19 ∨ n = 20 ∨ n = 21 ∨ n = 22 ∨ n = 23) ∧ 
           (∃ k, k * k = (n! * (n+1)!) / 2)) := 
by 
  sorry

end no_option_is_perfect_square_l618_618484


namespace correct_option_l618_618919

-- Define the four conditions as propositions
def option_A (a b : ℝ) : Prop := (a + b) ^ 2 = a ^ 2 + b ^ 2
def option_B (a : ℝ) : Prop := 2 * a ^ 2 + a = 3 * a ^ 3
def option_C (a : ℝ) : Prop := a ^ 3 * a ^ 2 = a ^ 5
def option_D (a : ℝ) (h : a ≠ 0) : Prop := 2 * a⁻¹ = 1 / (2 * a)

-- Prove which operation is the correct one
theorem correct_option (a b : ℝ) (h : a ≠ 0) : option_C a :=
by {
  -- Placeholder for actual proofs, each option needs to be verified
  sorry
}

end correct_option_l618_618919


namespace magnitude_a_minus_b_l618_618200

variables (m : ℝ)
def a : ℝ × ℝ := (m, 2)
def b : ℝ × ℝ := (4, -2)

axiom parallel_vectors (mval : ℝ) : a mval = (mval, 2) ∧ b = (4, -2) ∧ (mval ≠ 0 → -2 * mval = 8)
axiom value_of_m : m = -4 

theorem magnitude_a_minus_b : 
  |let va := a m, vb := b in 
  (va.1 - vb.1, va.2 - vb.2)| = 4 * Real.sqrt 5 :=
by
  sorry

end magnitude_a_minus_b_l618_618200


namespace red_grapes_count_l618_618288

theorem red_grapes_count (G : ℕ) (total_fruit : ℕ) (red_grapes : ℕ) (raspberries : ℕ)
  (h1 : red_grapes = 3 * G + 7) 
  (h2 : raspberries = G - 5) 
  (h3 : total_fruit = G + red_grapes + raspberries) 
  (h4 : total_fruit = 102) : 
  red_grapes = 67 :=
by
  sorry

end red_grapes_count_l618_618288


namespace power_quotient_l618_618583

theorem power_quotient (a m n : ℕ) (h_a : a = 19) (h_m : m = 11) (h_n : n = 8) : a^m / a^n = 6859 := by
  sorry

end power_quotient_l618_618583


namespace solve_trig_problem_l618_618171

-- Definition of the given problem for trigonometric identities
def problem_statement : Prop :=
  (1 - Real.tan (Real.pi / 12)) / (1 + Real.tan (Real.pi / 12)) = Real.sqrt 3 / 3

theorem solve_trig_problem : problem_statement :=
  by
  sorry -- No proof is needed here

end solve_trig_problem_l618_618171


namespace find_center_and_tangent_slope_l618_618238

theorem find_center_and_tangent_slope :
  let C := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 6 * p.1 + 8 = 0 }
  let center := (3, 0)
  let k := - (Real.sqrt 2 / 4)
  (∃ c ∈ C, c = center) ∧
  (∃ q ∈ C, q.2 < 0 ∧ q.2 = k * q.1 ∧
             |3 * k| / Real.sqrt (k ^ 2 + 1) = 1) :=
by
  sorry

end find_center_and_tangent_slope_l618_618238


namespace largest_subset_size_l618_618973

-- Definition of the subset condition
def valid_subset (s : Set ℕ) : Prop :=
  ∀ (a ∈ s) (b ∈ s), a ≠ 4 * b

-- Setting the range
def range_set : Set ℕ := { n | 1 ≤ n ∧ n ≤ 150 }

-- The main theorem statement
theorem largest_subset_size : ∃ (s : Set ℕ), s ⊆ range_set ∧ valid_subset (s) ∧ #(s) = 150 :=
by
  sorry

end largest_subset_size_l618_618973


namespace power_quotient_l618_618579

theorem power_quotient (a m n : ℕ) (h_a : a = 19) (h_m : m = 11) (h_n : n = 8) : a^m / a^n = 6859 := by
  sorry

end power_quotient_l618_618579


namespace g_of_f_three_l618_618371

def f (x : ℝ) : ℝ := x^3 - 2
def g (x : ℝ) : ℝ := 3 * x^2 + x + 2

theorem g_of_f_three : g (f 3) = 1902 :=
by
  sorry

end g_of_f_three_l618_618371


namespace perpendiculars_concurrent_l618_618210

-- Defining the quadrilateral and relevant points
variables {A B C D A_1 B_1 C_1 : Type} [geometry.AffineSpace ℝ (euclidean_space ℝ (fin 4))]

-- Condition: A_1, B_1, and C_1 are the orthocenters of triangles BCD, ACD, and ABD respectively
def orthocenter_BCD (B C D : Type) : Type := sorry
def orthocenter_ACD (A C D : Type) : Type := sorry
def orthocenter_ABD (A B D : Type) : Type := sorry

-- Perpendiculars from A to B_1C_1, B to C_1A_1, and C to A_1B_1
def perp_from_A_to_B1C1 (A B_1 C_1 : Type) : Prop := sorry
def perp_from_B_to_C1A1 (B C_1 A_1 : Type) : Prop := sorry
def perp_from_C_to_A1B1 (C A_1 B_1 : Type) : Prop := sorry

-- Statement: The perpendiculars intersect at a single point
theorem perpendiculars_concurrent :
  orthocenter_BCD B C D = A_1 →
  orthocenter_ACD A C D = B_1 →
  orthocenter_ABD A B D = C_1 →
  perp_from_A_to_B1C1 A B_1 C_1 →
  perp_from_B_to_C1A1 B C_1 A_1 →
  perp_from_C_to_A1B1 C A_1 B_1 →
  ∃ P, P ∈ (line_through (line_through A (line_intersect A B_1 C_1)) (line_through B (line_intersect B C_1 A_1))) ∧
       P ∈ (line_through (line_through C (line_intersect C A_1 B_1))).
Proof:
  sorry

end perpendiculars_concurrent_l618_618210


namespace largest_subset_no_member_is_4_times_another_l618_618981

-- Define the predicate that characterizes the subset
def valid_subset (S : Set ℕ) : Prop :=
  ∀ (x ∈ S) (y ∈ S), x ≠ 4 * y ∧ y ≠ 4 * x

-- Define the set of integers from 1 to 150
def range_1_to_150 : Set ℕ := {n | 1 ≤ n ∧ n ≤ 150}

-- State the theorem
theorem largest_subset_no_member_is_4_times_another :
  ∃ S ⊆ range_1_to_150, valid_subset S ∧ S.card = 140 := sorry

end largest_subset_no_member_is_4_times_another_l618_618981


namespace exponent_division_l618_618575

theorem exponent_division (a : ℕ) (m n : ℕ) (h1 : 19 = a) (h2 : 11 = m) (h3 : 8 = n) : a^(m - n) = 6859 := by
  sorry

end exponent_division_l618_618575


namespace probability_smallest_divides_larger_l618_618888

/-- Given three distinct numbers selected from the set {1, 2, 3, 4, 5, 6}, 
the probability that the smallest number divides at least one of the larger ones is 9/10. -/
theorem probability_smallest_divides_larger :
  (∃ (A : finset ℕ) (hA : A.card = 3) (hA_sub : A ⊆ {1, 2, 3, 4, 5, 6})
       (h_distinct : A.pairwise (≠)) (h_min_div : ∃ (a b c : ℕ) (h_dist : a ≠ b ∧ a ≠ c ∧ b ≠ c),
         a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ 
         (a < b ∧ a < c) ∧ (a ∣ b ∨ a ∣ c)),
    (finset.card {A | ∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧
      (a < b ∧ a < c) ∧
      (a ∣ b ∨ a ∣ c) ∧ 
      A ⊆ {1, 2, 3, 4, 5, 6} ∧ A.card = 3}) = 9 * 20) :=
sorry

end probability_smallest_divides_larger_l618_618888


namespace corrected_sum_with_unknown_S_l618_618875

variable (S : ℝ)

def initial_sum : ℝ := 675
def correction := (65 - 25) + (78 - 38) + (45 - 73)
def corrected_pre_sum : ℝ := initial_sum + correction
def final_corrected_sum : ℝ := corrected_pre_sum + S/2

theorem corrected_sum_with_unknown_S
  (S : ℝ)
  (initial_sum : ℝ := 675)
  (correction : ℝ := (65 - 25) + (78 - 38) + (45 - 73))
  (corrected_pre_sum : ℝ := initial_sum + correction)
  (final_corrected_sum : ℝ := corrected_pre_sum + S/2)
  : final_corrected_sum = 727 + S / 2 := by
  sorry

end corrected_sum_with_unknown_S_l618_618875


namespace sum_reciprocals_of_roots_l618_618381

theorem sum_reciprocals_of_roots :
  let S := ∑ n in (Finset.fin_range 2020), (1 / (1 - a n))
  (∀ (n : ℕ), a n ∈ ({x | polynomial.eval x (polynomial.C 1 * (polynomial.X 2020 + polynomial.X 2019 + ... + polynomial.X) - polynomial.C 2022) = 0 })) →
  S = -2041210 :=
by
  sorry

end sum_reciprocals_of_roots_l618_618381


namespace least_possible_sum_24_l618_618375

noncomputable def leastSum (m n : ℕ) 
  (h1: m > 0)
  (h2: n > 0)
  (h3: Nat.gcd (m + n) 231 = 1)
  (h4: m^m % n^n = 0)
  (h5: ¬ (m % n = 0))
  : ℕ :=
  m + n

theorem least_possible_sum_24 : ∃ (m n : ℕ), 
  m > 0 ∧ 
  n > 0 ∧ 
  Nat.gcd (m + n) 231 = 1 ∧ 
  m^m % n^n = 0 ∧ 
  ¬(m % n = 0) ∧ 
  leastSum m n m_pos n_pos gcd_cond mult_cond not_mult_cond = 24 :=
begin
  sorry
end

end least_possible_sum_24_l618_618375


namespace sequence_positive_from_26_l618_618212

-- Define the sequence with its general term formula.
def a_n (n : ℕ) : ℤ := 4 * n - 102

-- State the theorem that for all n ≥ 26, a_n > 0.
theorem sequence_positive_from_26 (n : ℕ) (h : n ≥ 26) : a_n n > 0 := by
  sorry

end sequence_positive_from_26_l618_618212


namespace find_B_find_a_l618_618772

-- Definitions
def set_A : Set ℝ := { x | (8*x - 1)*(x - 1) ≤ 0 }
def set_C (a : ℝ) : Set ℝ := { x | a < x ∧ x < 2*a + 5 }

-- Statements to prove: B = [0, 3/2], given conditions
theorem find_B (t : ℝ) : ( (8 * (1/4)^t - 1) * ((1/4)^t - 1) ≤ 0 ) → (0 ≤ t ∧ t ≤ 3/2) :=
by
  sorry

-- Statements to prove: range of a where (A ∪ B) ⊆ C
theorem find_a (a : ℝ) (t : ℝ) : (0 ≤ t ∧ t ≤ 3/2) → (∀ x, x ∈ set_A ∪ { t | 0 ≤ t ∧ t ≤ 3/2 } → x ∈ set_C a) → (-7/4 ≤ a ∧ a ≤ 0) :=
by
  sorry

end find_B_find_a_l618_618772


namespace polynomials_exists_l618_618770

theorem polynomials_exists (n : ℕ) (h : n > 0) : 
  ∃ f g : ℤ[X], f * (X + 1)^(2^n) + g * (X^(2^n) + 1) = (2 : ℤ[X]) :=
begin
  sorry
end

end polynomials_exists_l618_618770


namespace exponent_division_l618_618569

theorem exponent_division : (19 ^ 11) / (19 ^ 8) = 6859 :=
by
  -- Here we assume the properties of powers and arithmetic operations
  sorry

end exponent_division_l618_618569


namespace polynomial_degree_bounds_l618_618382

open nat

theorem polynomial_degree_bounds (p : ℕ) (hp : prime p) (f : polynomial ℤ) (hf_degree : f.degree = d)
  (hf_0 : f.eval 0 = 0) (hf_1 : f.eval 1 = 1) 
  (hf_mod : ∀ (n : ℕ), n > 0 → f.eval n % p = 0 ∨ f.eval n % p = 1) :
  d ≥ p - 1 :=
sorry

end polynomial_degree_bounds_l618_618382


namespace speed_of_other_child_l618_618836

-- Define the speeds of Chukov and Gekov
variables {V1 V2 : ℝ}

-- Given conditions as hypotheses in Lean
-- 1. They meet four times more often than they overtake each other
axiom frequency_condition : (V1 + V2) / |V1 - V2| = 4

-- 2. One of the children's speed is 6 m/s
axiom one_child_speed : V1 = 6 ∨ V2 = 6

-- Goal: Prove the speed of the other child is either 10 m/s or 3.6 m/s
theorem speed_of_other_child (V1 V2 : ℝ) (h1 : frequency_condition) (h2 : one_child_speed) :
  (V1 = 6 ∧ V2 = 10) ∨ (V1 = 10 ∧ V2 = 6) ∨ (V1 = 6 ∧ V2 = 3.6) ∨ (V1 = 3.6 ∧ V2 = 6) :=
sorry

end speed_of_other_child_l618_618836


namespace deposits_exceed_10_on_second_Tuesday_l618_618390

noncomputable def deposits_exceed_10 (n : ℕ) : ℕ :=
2 * (2^n - 1)

theorem deposits_exceed_10_on_second_Tuesday :
  ∃ n, deposits_exceed_10 n > 1000 ∧ 1 + (n - 1) % 7 = 2 ∧ n < 21 :=
sorry

end deposits_exceed_10_on_second_Tuesday_l618_618390


namespace problem_a_problem_b_l618_618792

noncomputable def sequence (a : ℕ) : ℕ → ℕ
| 0       := 1
| (n + 1) := a + ∏ i in Finset.range (n + 1), sequence a i

theorem problem_a (a : ℕ) (h_a : a > 0) : ∃ P : ℕ → Prop, 
  (∀ p, P p → prime p) ∧ (∀ n, ∃ p, P p ∧ p ∣ sequence a n) :=
sorry

theorem problem_b (a : ℕ) (h_a : a > 0) : ∃ p, prime p ∧ (∀ n, ¬ (p ∣ sequence a n)) :=
sorry

end problem_a_problem_b_l618_618792


namespace limit_S_squared_l618_618361

noncomputable def S (n m : ℤ) : ℚ :=
  if |m| ≤ n then
    let a := (1 : ℚ)
    let r := (1 / 4)
    (4 / (3 * 2^m)) * (1 - r^(n-m+1))
  else
    0

theorem limit_S_squared :
  ∀ n : ℕ, 
    (∀ m : ℤ, |m| ≤ n → 
      S n m = (4 / (3 * 2^m)) * (1 - (1 / (4))^(n - m + 1))): 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, | (∑ k in finset.range (2 * n + 1), 
  (S n (k - n))^2) - (176 / 27) | < ε := sorry

end limit_S_squared_l618_618361


namespace correct_analogy_statement_l618_618996

-- Define the problem: check analogies and decide which statement is correct.
-- Definitions related to the problem conditions
def analogy_log (a x y : ℝ) : Prop := log a (x + y) = log a x + log a y
def analogy_dot_product (a b : ℝ) : Prop := abs (a * b) = abs a * abs b
def analogy_power (a b : ℝ) (n : ℕ) : Prop := (a + b) ^ n = a ^ n + b ^ n
def analogy_cuboid_diagonal (l w h : ℝ) : Prop := (l^2 + w^2 + h^2 = (l * l + w * w + h * h))

-- Defining which analogy is correct
theorem correct_analogy_statement (l w h : ℝ) : analogy_cuboid_diagonal l w h :=
by sorry

end correct_analogy_statement_l618_618996


namespace largest_initial_number_l618_618322

theorem largest_initial_number :
  ∃ n : ℕ, 
    (∀ (a1 a2 a3 a4 a5 : ℕ),
      (n + a1).gcd a1 = 1 ∧
      (n + a1 + a2).gcd a2 = 1 ∧
      (n + a1 + a2 + a3).gcd a3 = 1 ∧
      (n + a1 + a2 + a3 + a4).gcd a4 = 1 ∧
      (n + a1 + a2 + a3 + a4 + a5).gcd a5 = 1 ∧
      n + a1 + a2 + a3 + a4 + a5 = 200) 
    → n = 189 :=
begin
  sorry
end

end largest_initial_number_l618_618322


namespace largest_initial_number_l618_618309

theorem largest_initial_number : ∃ n : ℕ, (n + 5 ∑ k : ℕ, k ≠ 0 ∧ ¬ (n % k = 0)) = 200 ∧ n = 189 :=
begin
  sorry
end

end largest_initial_number_l618_618309


namespace number_of_divisors_of_2744_l618_618185

-- Definition of the integer and its prime factorization
def two := 2
def seven := 7
def n := two^3 * seven^3

-- Define the property for the number of divisors
def num_divisors (n : ℕ) : ℕ := (3 + 1) * (3 + 1)

-- Main proof statement
theorem number_of_divisors_of_2744 : num_divisors n = 16 := by
  sorry

end number_of_divisors_of_2744_l618_618185


namespace find_b_value_l618_618372

theorem find_b_value (f : ℝ → ℝ) (f_inv : ℝ → ℝ) (b : ℝ) :
  (∀ x, f x = 1 / (3 * x + b)) →
  (∀ x, f_inv x = (2 - 3 * x) / (3 * x)) →
  b = -3 :=
by
  intros h1 h2
  sorry

end find_b_value_l618_618372


namespace average_of_consecutive_odds_l618_618278

theorem average_of_consecutive_odds {n : ℕ} {a : ℤ} (h1 : n = 10) (h2 : a = 145) : 
  let seq := list.range n
  let odds := list.map (λ i, a + 2 * i) seq
  let total := list.sum odds
  let avg := total / n in
  avg = 154 := 
by
  sorry

end average_of_consecutive_odds_l618_618278


namespace dice_sum_coverage_l618_618886

def dice1 := {1, 2, 3, 4, 5, 6}
def dice2 := {0, 6, 12, 18, 24, 30}

theorem dice_sum_coverage :
  let sums := {n | ∃ x ∈ dice1, ∃ y ∈ dice2, n = x + y } in 
  sums = {1, 2, ..., 36} :=
by {
  sorry
}

end dice_sum_coverage_l618_618886


namespace outfits_not_same_color_l618_618710

/--
Given:
- 7 shirts, 7 pairs of pants, and 7 hats.
- Each item comes in 7 colors (one of each item of each color).
- No outfit is allowed where all 3 items are the same color.

Prove:
The number of possible outfits where not all items are the same color is 336.
-/
theorem outfits_not_same_color : 
  let total_outfits := 7 * 7 * 7 in
  let same_color_outfits := 7 in
  total_outfits - same_color_outfits = 336 :=
by
  let total_outfits := 7 * 7 * 7
  let same_color_outfits := 7
  have h1 : total_outfits = 343 := by norm_num
  have h2 : total_outfits - same_color_outfits = 336 := by norm_num
  exact h2

end outfits_not_same_color_l618_618710


namespace both_pieces_no_shorter_than_1m_l618_618852

noncomputable def rope_cut_probability : ℝ :=
let total_length: ℝ := 3 in
let favorable_length: ℝ := 1 in
favorable_length / total_length

theorem both_pieces_no_shorter_than_1m : rope_cut_probability = 1 / 3 := 
by
  sorry

end both_pieces_no_shorter_than_1m_l618_618852


namespace determine_mr_l618_618862

open Real

-- Define the 5x5 grid of points
def points : List (ℕ × ℕ) := 
  List.product (List.range 5) (List.range 5)

-- Distance between two points in the grid
def dist (p1 p2 : ℕ × ℕ) : ℝ :=
  (Real.sqrt ((p2.1 - p1.1 : ℕ)^2 + (p2.2 - p1.2 : ℕ)^2) : ℝ)

-- Describe a growing path in the points
def isGrowingPath (path : List (ℕ × ℕ)) : Prop :=
  List.chain' (fun p1 p2 => dist p1 p2 > dist p1 (path.head!)) path

-- Define the maximum number of points in a growing path
def m : ℕ := 15

-- Define the number of such growing paths consisting of exactly m points
def r : ℕ := 12

-- Define the product of m and r
def mr : ℕ := m * r

theorem determine_mr : mr = 180 :=
by 
  -- Define the maximum length
  have m_def : m = 15 := rfl
  -- Define the number of such paths
  have r_def : r = 12 := rfl
  -- Define the product
  have product_def : mr = m * r := rfl
  -- Calculate the product
  rw [m_def, r_def, product_def]
  exact rfl

end determine_mr_l618_618862


namespace sqrt_sum_l618_618067

theorem sqrt_sum :
  (√(2 * (5^3) + 2 * (5^3) + 2 * (5^3))) = 5 * √30 :=
by
  sorry

end sqrt_sum_l618_618067


namespace option_a_equals_half_option_c_equals_half_l618_618551

theorem option_a_equals_half : 
  ( ∃ x : ℝ, x = (√2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180)) ∧ x = 1 / 2 ) := 
sorry

theorem option_c_equals_half : 
  ( ∃ y : ℝ, y = (Real.tan (22.5 * Real.pi / 180) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)) ∧ y = 1 / 2 ) := 
sorry

end option_a_equals_half_option_c_equals_half_l618_618551


namespace TedLoses_l618_618563

-- Definitions of the conditions given in the problem
def CarlWins := 5
def CarlLoses := 0
def JamesWins := 4
def JamesLoses := 2
def SaifWins := 1
def SaifLoses := 6
def TedWins := 4

-- Theorem stating that the number of games Ted loses is 6
theorem TedLoses : ∑ (x : list ℕ) → x = [CarlWins, JamesWins, SaifWins, TedWins] = ∑ (y : list ℕ) → y = [CarlLoses, JamesLoses, SaifLoses, 6] :=
by
  sorry

end TedLoses_l618_618563


namespace largest_initial_number_l618_618351

theorem largest_initial_number :
  ∃ n : ℕ, (∀ k : ℕ, (n % k ≠ 0 → k ∈ {2, 2, 2, 2, 3}) ∧ (n + 11 = 200)) ∧ (n = 189) :=
begin
  sorry -- Proof not required per instruction
end

end largest_initial_number_l618_618351


namespace find_modulus_squared_l618_618801

theorem find_modulus_squared
  (z : ℂ)
  (h : z^2 + complex.abs z^2 = 2 - 3 * complex.I) :
  complex.abs z^2 = 13 / 4 :=
sorry

end find_modulus_squared_l618_618801


namespace square_ratio_short_to_long_side_l618_618879

theorem square_ratio_short_to_long_side (a b : ℝ) (h : a / b + 1 / 2 = b / (Real.sqrt (a^2 + b^2))) : (a / b)^2 = (3 - Real.sqrt 5) / 2 := by
  sorry

end square_ratio_short_to_long_side_l618_618879


namespace outfits_not_same_color_l618_618711

/--
Given:
- 7 shirts, 7 pairs of pants, and 7 hats.
- Each item comes in 7 colors (one of each item of each color).
- No outfit is allowed where all 3 items are the same color.

Prove:
The number of possible outfits where not all items are the same color is 336.
-/
theorem outfits_not_same_color : 
  let total_outfits := 7 * 7 * 7 in
  let same_color_outfits := 7 in
  total_outfits - same_color_outfits = 336 :=
by
  let total_outfits := 7 * 7 * 7
  let same_color_outfits := 7
  have h1 : total_outfits = 343 := by norm_num
  have h2 : total_outfits - same_color_outfits = 336 := by norm_num
  exact h2

end outfits_not_same_color_l618_618711


namespace children_getting_on_bus_l618_618941

theorem children_getting_on_bus (a b c: ℕ) (ha : a = 64) (hb : b = 78) (hc : c = b - a) : c = 14 :=
by
  sorry

end children_getting_on_bus_l618_618941


namespace red_grapes_count_l618_618287

theorem red_grapes_count (G : ℕ) (total_fruit : ℕ) (red_grapes : ℕ) (raspberries : ℕ)
  (h1 : red_grapes = 3 * G + 7) 
  (h2 : raspberries = G - 5) 
  (h3 : total_fruit = G + red_grapes + raspberries) 
  (h4 : total_fruit = 102) : 
  red_grapes = 67 :=
by
  sorry

end red_grapes_count_l618_618287


namespace sum_f_from_1_to_2000_l618_618600

noncomputable def f (n : ℕ) : ℝ :=
if h : (Real.log n / Real.log 4).is_rational then Real.log n / Real.log 4 else 0

theorem sum_f_from_1_to_2000 : ∑ n in Finset.range 2000, f (n + 1) = 15 := sorry

end sum_f_from_1_to_2000_l618_618600


namespace incorrect_statement_l618_618293

def data_set : List ℤ := [10, 8, 6, 9, 8, 7, 8]

theorem incorrect_statement : 
  let mode := 8
  let median := 8
  let mean := 8
  let variance := 8
  (∃ x ∈ data_set, x ≠ 8) → -- suppose there is at least one element in the dataset not equal to 8
  (1 / 7 : ℚ) * (4 + 0 + 4 + 1 + 0 + 1 + 0) ≠ 8 := -- calculating real variance from dataset
by
  sorry

end incorrect_statement_l618_618293


namespace cost_of_one_book_l618_618391

theorem cost_of_one_book (x : ℝ) : 
  (9 * x = 11) ∧ (13 * x = 15) → x = 1.23 :=
by sorry

end cost_of_one_book_l618_618391


namespace coefficient_6th_term_l618_618228

open Real

noncomputable def a : ℝ := ∫ x in 1..2, (3 * x^2 - 2 * x)

theorem coefficient_6th_term :
  let a := (∫ x in 1..2, 3 * x^2 - 2 * x)
  (a = 4) → 
  (∑ i in finset.range (6 + 1), (nat.choose 6 i) * (4^i) * ((-1)^(6-i)) * (x^(2*(6-i)-i)))[5] = -24 :=
by
  sorry

end coefficient_6th_term_l618_618228


namespace rook_cannot_return_to_a1_rook_can_end_at_a2_l618_618295

-- Definition of the conditions
variable (board : Fin 8 × Fin 8 -> ℕ) (start : Fin 8 × Fin 8) (n : ℕ)

-- Part (1): Question 1
theorem rook_cannot_return_to_a1 :
  start = (⟨0, by norm_num⟩, ⟨0, by norm_num⟩) →
  (∀ i j, (i, j) ∈ ∃ seq : List (Fin 8 × Fin 8), seq.head = (⟨0, by norm_num⟩, ⟨0, by norm_num⟩) ∧ seq.nodup ∧ seq.length = 64) →
  ¬ (∃ seq : List (Fin 8 × Fin 8), seq.head = start ∧ seq.last = start ∧ seq.nodup ∧ seq.length = 64) :=
by sorry

-- Part (2): Question 2
theorem rook_can_end_at_a2 :
  start = (⟨0, by norm_num⟩, ⟨0, by norm_num⟩) →
  (∀ i j, (i, j) ∈ ∃ seq : List (Fin 8 × Fin 8), seq.head = (⟨0, by norm_num⟩, ⟨0, by norm_num⟩) ∧ seq.nodup ∧ seq.length = 64) →
  ∃ seq : List (Fin 8 × Fin 8), seq.head = start ∧ seq.last = (⟨0, by norm_num⟩, ⟨1, by norm_num⟩) ∧ seq.nodup ∧ seq.length = 64 :=
by sorry

end rook_cannot_return_to_a1_rook_can_end_at_a2_l618_618295


namespace find_f_neg_2_l618_618601

-- Definitions from conditions
def odd_function_on_reals (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 + 1 else sorry -- to be defined

-- Theorem to state the problem
theorem find_f_neg_2 (f : ℝ → ℝ) (h1 : odd_function_on_reals f)
  (h2 : ∀ x : ℝ, 0 < x → f x = x^2 + 1) :
  f (-2) = -5 :=
begin
  sorry -- proof to be provided
end

end find_f_neg_2_l618_618601


namespace sum_of_squares_of_roots_l618_618150

theorem sum_of_squares_of_roots :
  (∀ y : ℝ, y^3 - 8*y^2 + 9*y - 2 = 0 → y ≥ 0) →
  (∃ r s t : ℝ, (r^3 - 8*r^2 + 9*r - 2 = 0) ∧ (s^3 - 8*s^2 + 9*s - 2 = 0) ∧ (t^3 - 8*t^2 + 9*t - 2 = 0) ∧
          r + s + t = 8 ∧ r * s + s * t + t * r = 9) →
  r^2 + s^2 + t^2 = 46 :=
by transition
by_paths
by tactic
show ex.

Sorry: Sign-of- root {.no.root,y} good.tags -lean.error

end sum_of_squares_of_roots_l618_618150


namespace kolya_is_wrong_l618_618032

def pencils_problem_statement (at_least_four_blue : Prop) 
                              (at_least_five_green : Prop) 
                              (at_least_three_blue_and_four_green : Prop) 
                              (at_least_four_blue_and_four_green : Prop) : 
                              Prop :=
  ∃ (B G : ℕ), -- B represents the number of blue pencils, G represents the number of green pencils
    ((B ≥ 4) ∧ (G ≥ 4)) ∧ -- Vasya's statement (at least 4 blue), Petya's and Misha's combined statement (at least 4 green)
    at_least_four_blue ∧ -- Vasya's statement (there are at least 4 blue pencils)
    (at_least_five_green ↔ G ≥ 5) ∧ -- Kolya's statement (there are at least 5 green pencils)
    at_least_three_blue_and_four_green ∧ -- Petya's statement (at least 3 blue and 4 green)
    at_least_four_blue_and_four_green -- Misha's statement (at least 4 blue and 4 green)

theorem kolya_is_wrong (at_least_four_blue : Prop) 
                        (at_least_five_green : Prop) 
                        (at_least_three_blue_and_four_green : Prop) 
                        (at_least_four_blue_and_four_green : Prop) : 
                        pencils_problem_statement at_least_four_blue 
                                                  at_least_five_green 
                                                  at_least_three_blue_and_four_green 
                                                  at_least_four_blue_and_four_green :=
sorry

end kolya_is_wrong_l618_618032


namespace min_value_expression_l618_618378

noncomputable def w : ℂ := sorry

theorem min_value_expression (hw : |(w - 3 + 3 * (complex.I : ℂ))| = 3) :
  (∃ (v : ℂ), (|v + 1 - (complex.I)|^2 + |v - 7 + 2 * (complex.I)|^2 = 17) ∧
      ∀ z: ℂ, |(z - 3 + 3 * (complex.I : ℂ))| = 3 → 
    (|z + 1 - (complex.I)|^2 + |z - 7 + 2 * (complex.I)|^2 ≥ 17)) := 
sorry

end min_value_expression_l618_618378


namespace solve_part1_solve_part2_l618_618652

noncomputable def a_n (n : ℕ) : ℝ :=
if n = 0 then 0 else 2 * n

noncomputable def S_n (n : ℕ) : ℝ :=
∀ n, S_n n = (n^2 + n)

theorem solve_part1 (n : ℕ) (n_pos : n > 0) :
  S_n n = n^2 + n → a_n n = 2 * n :=
sorry

noncomputable def b_n (n : ℕ) : ℝ :=
(n + 1) / (4 * n^2 * (n + 2)^2 * (a_n n)^2)

noncomputable def T_n (n : ℕ) : ℝ :=
  ∑ i in finset.range n, b_n i

theorem solve_part2 (n : ℕ) (n_pos : n > 0) :
  T_n n < 5 / 64 :=
sorry

end solve_part1_solve_part2_l618_618652


namespace smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_20_l618_618796

theorem smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_20 :
  ∃ n : ℕ, n > 1 ∧ ¬ prime n ∧ (∀ p : ℕ, p.prime → p ∣ n → p ≥ 20) ∧ n = 529 :=
sorry

end smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_20_l618_618796


namespace consecutive_integers_product_sum_l618_618442

theorem consecutive_integers_product_sum (a b c d : ℕ) :
  a * b * c * d = 3024 ∧ b = a + 1 ∧ c = b + 1 ∧ d = c + 1 → a + b + c + d = 30 :=
by
  sorry

end consecutive_integers_product_sum_l618_618442


namespace discount_is_25_l618_618165

def original_price : ℕ := 76
def discounted_price : ℕ := 51
def discount_amount : ℕ := original_price - discounted_price

theorem discount_is_25 : discount_amount = 25 := by
  sorry

end discount_is_25_l618_618165


namespace car_return_speed_l618_618945

theorem car_return_speed
  (distance_cd : ℕ) (speed_cd : ℕ) (average_speed_rt : ℕ)
  (h_distance : distance_cd = 150) (h_speed : speed_cd = 75) (h_avg_speed : average_speed_rt = 50) :
  ∃ r, 2 * distance_cd / (distance_cd / speed_cd + distance_cd / r) = average_speed_rt ∧ r = 37.5 := by
  sorry

end car_return_speed_l618_618945


namespace max_initial_number_l618_618341

theorem max_initial_number (n : ℕ) : 
  (∃ (a b c d e : ℕ), 
    200 = n + a + b + c + d + e ∧ 
    ¬ (n % a = 0) ∧ 
    ¬ ((n + a) % b = 0) ∧ 
    ¬ ((n + a + b) % c = 0) ∧ 
    ¬ ((n + a + b + c) % d = 0) ∧ 
    ¬ ((n + a + b + c + d) % e = 0)) → 
  n ≤ 189 := 
sorry

end max_initial_number_l618_618341


namespace discard_sacks_l618_618263

theorem discard_sacks (harvested_sacks_per_day : ℕ) (oranges_per_day : ℕ) (oranges_per_sack : ℕ) :
  harvested_sacks_per_day = 76 → oranges_per_day = 600 → oranges_per_sack = 50 → 
  harvested_sacks_per_day - oranges_per_day / oranges_per_sack = 64 :=
by
  intros h1 h2 h3
  -- Automatically passes the proof as a placeholder
  sorry

end discard_sacks_l618_618263


namespace geometric_sequence_S6_l618_618663

open_locale big_operators

noncomputable def geometric_sequence_sum := λ (n : ℕ) (a1 : ℝ) (q : ℝ), a1 * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_S6 :
  ∃ (a1 q : ℝ), q ≠ 1 ∧ a1 + a1 * q = 3 / 4 ∧ a1 * q^3 + a1 * q^4 = 6 ∧ geometric_sequence_sum 6 a1 q = 63 / 4 :=
begin
  use [1/4, 2],
  split,
  { -- Proof that q ≠ 1
    norm_num },

  split,
  { -- Proof that a1 + a1 * q = 3 / 4
    norm_num,
    ring },

  split,
  { -- Proof that a1 * q^3 + a1 * q^4 = 6
    norm_num,
    ring },

  { -- Proof that geometric_sequence_sum 6 a1 q = 63 / 4
    rw [geometric_sequence_sum, if_neg],
    { norm_num, ring },
    { norm_num } }
end

end geometric_sequence_S6_l618_618663


namespace calculate_box_sum_l618_618190

def box (a b c : Int) : ℚ := a^b + b^c - c^a

theorem calculate_box_sum :
  box 2 3 (-1) + box (-1) 2 3 = 16 := by
  sorry

end calculate_box_sum_l618_618190


namespace remainder_n_plus_2023_mod_7_l618_618069

theorem remainder_n_plus_2023_mod_7 (n : ℤ) (h : n % 7 = 2) : (n + 2023) % 7 = 2 :=
by
  sorry

end remainder_n_plus_2023_mod_7_l618_618069


namespace chord_length_and_area_correct_l618_618514

noncomputable def chord_length_and_area (r d : ℝ) (h_r: r = 5) (h_d: d = 4) : ℝ × ℝ :=
let PQ := 2 * real.sqrt (r^2 - d^2) in
let area := real.pi * r^2 in
(PQ, area)

theorem chord_length_and_area_correct : chord_length_and_area 5 4 (by rfl) (by rfl) = (6, 25 * real.pi) :=
by {
  sorry -- proof required
}

end chord_length_and_area_correct_l618_618514


namespace range_of_a_l618_618522

noncomputable theory

variable {R : Type*} [Field R]

def function_conditions (f : R → R) :=
  (∀ x : R, x > 0 → f(x) > 0) ∧
  f(1) = 2 ∧
  (∀ m n : R, f(m + n) = f(m) + f(n))

def set_A (f : R → R) : Set (R × R) :=
  {p : R × R | f(3 * p.1 ^ 2) + f(4 * p.2 ^ 2) ≤ 24}

def set_B (f : R → R) (a : R) : Set (R × R) :=
  {p : R × R | f(p.1) - f(a * p.2) + f(3) = 0}

def set_C (f : R → R) (a : R) : Set (R × R) :=
  {p : R × R | f(p.1) = (1 / 2) * f(p.2 ^ 2) + f(a)}

theorem range_of_a (a : R) (f : R → R) :
  function_conditions f →
  (set_A f ∩ set_B f a ≠ ∅) →
  (set_A f ∩ set_C f a ≠ ∅) →
  a ∈ (Icc (-13/6 : R) (-sqrt 15 / 3 : R)) ∨ 
  a ∈ (Icc (sqrt 15 / 3 : R) (2 : R)) :=
by
  sorry

end range_of_a_l618_618522


namespace lucas_payment_l618_618395

noncomputable def payment (windows_per_floor : ℕ) (floors : ℕ) (days : ℕ) 
  (earn_per_window : ℝ) (delay_penalty : ℝ) (period : ℕ) : ℝ :=
  let total_windows := windows_per_floor * floors
  let earnings := total_windows * earn_per_window
  let penalty_periods := days / period
  let total_penalty := penalty_periods * delay_penalty
  earnings - total_penalty

theorem lucas_payment :
  payment 3 3 6 2 1 3 = 16 := by
  sorry

end lucas_payment_l618_618395


namespace minimum_value_ineq_l618_618800

theorem minimum_value_ineq (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 1) :
  (a^2 / b) + (b^2 / c) + (c^2 / a) ≥ 3 :=
by
  sorry

end minimum_value_ineq_l618_618800


namespace kolya_mistaken_l618_618011

-- Definitions relating to the conditions
def at_least_four_blue_pencils (blue_pencils : ℕ) : Prop := blue_pencils >= 4
def at_least_five_green_pencils (green_pencils : ℕ) : Prop := green_pencils >= 5
def at_least_three_blue_pencils_and_four_green_pencils (blue_pencils green_pencils : ℕ) : Prop := blue_pencils >= 3 ∧ green_pencils >= 4
def at_least_four_blue_and_four_green_pencils (blue_pencils green_pencils : ℕ) : Prop := blue_pencils >= 4 ∧ green_pencils >= 4

-- Speaking truth conditions
variables (blue_pencils green_pencils : ℕ)
def vasya_true : Prop := at_least_four_blue_pencils blue_pencils
def kolya_true : Prop := at_least_five_green_pencils green_pencils
def petya_true : Prop := at_least_three_blue_pencils_and_four_green_pencils blue_pencils green_pencils
def misha_true : Prop := at_least_four_blue_and_four_green_pencils blue_pencils green_pencils

-- Given known information: three are true, one is false
def known_information : Prop := (vasya_true blue_pencils green_pencils ∧ petya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬kolya_true blue_pencils green_pencils)
                            ∨ (vasya_true blue_pencils green_pencils ∧ petya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ ¬misha_true blue_pencils green_pencils)
                            ∨ (vasya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬petya_true blue_pencils green_pencils)
                            ∨ (petya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬vasya_true blue_pencils green_pencils)

-- Theorem to be proved
theorem kolya_mistaken : known_information blue_pencils green_pencils → ¬kolya_true blue_pencils green_pencils :=
sorry

end kolya_mistaken_l618_618011


namespace sum_of_integers_solving_equation_equals_4_l618_618874

theorem sum_of_integers_solving_equation_equals_4 :
  let S := {x : ℤ | x^2 = 5 * x + 310} in 
  ∑ x in S, x = 4 :=
by
  sorry

end sum_of_integers_solving_equation_equals_4_l618_618874


namespace two_circles_positional_relationship_l618_618444

theorem two_circles_positional_relationship
  (R r d : ℝ)
  (h_eq : ∀ x, x^2 - 3*x + 2 = 0 → x = R ∨ x = r)
  (h_dist : d = 3)
  (h_sum : R + r = d) :
  d = R + r → "externally tangent" :=
by sorry

end two_circles_positional_relationship_l618_618444


namespace x_in_C_necessary_not_sufficient_for_x_in_A_l618_618679

variables {A B C : Set} (x : α) [Nonempty A] [Nonempty B] [Nonempty C]

-- Conditions
axiom union_eq : A ∪ B = C
axiom B_not_subset_A : ¬ (B ⊆ A)

-- Theorem statement
theorem x_in_C_necessary_not_sufficient_for_x_in_A :
  (x ∈ A → x ∈ C) ∧ ¬ (x ∈ C → x ∈ A) :=
sorry

end x_in_C_necessary_not_sufficient_for_x_in_A_l618_618679


namespace increasing_interval_a_geq_neg_2_l618_618697

theorem increasing_interval_a_geq_neg_2 (a : ℝ) :
  (∀ x : ℝ, 4 < x → deriv (λ x, x^2 + 2*(a-2)*x + 5) x ≥ 0) → a ≥ -2 :=
by
  sorry

end increasing_interval_a_geq_neg_2_l618_618697


namespace right_triangle_segment_ratio_l618_618282

-- Definitions of the triangle sides and hypotenuse
def right_triangle (AB BC : ℝ) : Prop :=
  AB/BC = 4/3

def hypotenuse (AB BC AC : ℝ) : Prop :=
  AC^2 = AB^2 + BC^2

def perpendicular_segment_ratio (AD CD : ℝ) : Prop :=
  AD / CD = 9/16

-- Final statement of the problem
theorem right_triangle_segment_ratio
  (AB BC AC AD CD : ℝ)
  (h1 : right_triangle AB BC)
  (h2 : hypotenuse AB BC AC)
  (h3 : perpendicular_segment_ratio AD CD) :
  CD / AD = 16/9 := sorry

end right_triangle_segment_ratio_l618_618282


namespace general_term_formula_arith_geo_seq_l618_618745

noncomputable def arithmetic_geometric_seq : Nat → Rat
| n => if q = (2/5) then 125 * (q ^ (n-1)) else 8 * (q ^ (n-1))

theorem general_term_formula_arith_geo_seq (a : Rat) (n : Nat) (q : Rat) :
  (a + a * q^3 = 133) →
  (a * q + a * q^2 = 70) →
  (q = 2/5 ∨ q = 5/2) →
  (arithmetic_geometric_seq n = if q = (2/5) then 125 * (q ^ (n-1)) else 8 * (q ^ (n-1))) :=
begin
  -- Proof would go here
  sorry
end

end general_term_formula_arith_geo_seq_l618_618745


namespace find_coordinates_of_B_l618_618198

-- Define points and circle
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨2, 2⟩
def circle (P : Point) : Prop := P.x^2 + P.y^2 = 4

-- Distance function
def distance (P₁ P₂ : Point) : ℝ := real.sqrt ((P₁.x - P₂.x)^2 + (P₁.y - P₂.y)^2)

-- Given condition for any point P on the circle
def condition (P : Point) (B : Point) : Prop :=
  circle P → distance P A / distance P B = real.sqrt 2

-- Proof statement
theorem find_coordinates_of_B (B : Point) : 
  (∀ P : Point, condition P B) → B = ⟨1, 1⟩ :=
by
  sorry

end find_coordinates_of_B_l618_618198


namespace sum_of_j_10_values_l618_618786

def h (x : ℝ) : ℝ := x^2 - 8*x + 22

def j (x : ℝ) : ℝ := 3*x + 4

theorem sum_of_j_10_values : (∑ x in {2, 6}, j x) = 32 :=
by
  sorry

end sum_of_j_10_values_l618_618786


namespace incorrect_statement_B_l618_618486

theorem incorrect_statement_B {
  x y x1 y1 x2 y2 : ℝ
  (hA : ∀ a, (2, 0) satisfies x - a * y - 2 = 0)
  (hB : ∀ (x1 y1 x2 y2 : ℝ), x1 ≠ x2 ∧ y1 ≠ y2 → (∃ x y, (x, y) ⊨ (y - y1) / (y2 - y1) = (x - x1) / (x2 - x1)))
  (hC : (1, 1) satisfies x + y - 2 = 0)
  (hD : ∀ (a : ℝ), let area := 1 / 2 * abs (4 * (-4)) in area = 8)
  : B is incorrect
:= sorry

end incorrect_statement_B_l618_618486


namespace quadratic_equation_in_terms_of_x_l618_618481

-- Define what it means to be a quadratic equation in terms of x.
def is_quadratic (eq : ℕ → ℕ → ℕ → Prop) : Prop :=
  ∃ a b c : ℕ, a ≠ 0 ∧ ∀ x : ℕ, eq x = a * x^2 + b * x + c

-- Define each condition as an equation.
def equation_A (x : ℕ) : ℕ := x - 1 / x + 2

def equation_B (x y : ℕ) : ℕ := x^2 + 2 * x + y

def equation_C (a b c x : ℕ ) : ℕ := a * x^2 + b * x + c

def equation_D (x : ℕ) : ℕ := x^2 - x + 1

theorem quadratic_equation_in_terms_of_x : is_quadratic equation_D :=
sorry

end quadratic_equation_in_terms_of_x_l618_618481


namespace exponent_division_l618_618572

theorem exponent_division : (19 ^ 11) / (19 ^ 8) = 6859 :=
by
  -- Here we assume the properties of powers and arithmetic operations
  sorry

end exponent_division_l618_618572


namespace max_initial_number_l618_618345

theorem max_initial_number (n : ℕ) : 
  (∃ (a b c d e : ℕ), 
    200 = n + a + b + c + d + e ∧ 
    ¬ (n % a = 0) ∧ 
    ¬ ((n + a) % b = 0) ∧ 
    ¬ ((n + a + b) % c = 0) ∧ 
    ¬ ((n + a + b + c) % d = 0) ∧ 
    ¬ ((n + a + b + c + d) % e = 0)) → 
  n ≤ 189 := 
sorry

end max_initial_number_l618_618345


namespace largest_subset_no_four_times_another_l618_618962

theorem largest_subset_no_four_times_another :
  ∃ (S : set ℕ), S ⊆ {1, 2, ..., 150} ∧ (∀ (a b : ℕ), a ∈ S → b ∈ S → (a ≠ 4 * b ∧ b ≠ 4 * a)) ∧ (S.card = 141) :=
sorry

end largest_subset_no_four_times_another_l618_618962


namespace range_DF_l618_618269

-- Define the points A, B, C, D, E, F and the triangles ABC and DEF
variables (A B C D E F : Type) [linear_order A] [linear_order B] [linear_order C] [linear_order D] [linear_order E] [linear_order F]

-- Declare the segments AB, BC, and DF
variables (AB BC DF : ℝ)

-- Define the congruence of the triangles
def triangles_congruent (P Q R S T U : Type) [linear_order P] [linear_order Q] [linear_order R] [linear_order S] [linear_order T] [linear_order U] : Prop :=
  ∀ (a b c d e f : ℝ), a = d → b = e → c = f

-- Problem statement
theorem range_DF : 
  triangles_congruent A B C D E F → AB = 4 → BC = 7 → 3 < DF ∧ DF < 11 :=
by
  sorry

end range_DF_l618_618269


namespace largest_initial_number_l618_618324

theorem largest_initial_number :
  ∃ n : ℕ, 
    (∀ (a1 a2 a3 a4 a5 : ℕ),
      (n + a1).gcd a1 = 1 ∧
      (n + a1 + a2).gcd a2 = 1 ∧
      (n + a1 + a2 + a3).gcd a3 = 1 ∧
      (n + a1 + a2 + a3 + a4).gcd a4 = 1 ∧
      (n + a1 + a2 + a3 + a4 + a5).gcd a5 = 1 ∧
      n + a1 + a2 + a3 + a4 + a5 = 200) 
    → n = 189 :=
begin
  sorry
end

end largest_initial_number_l618_618324


namespace equation_of_l3_line_l1_through_fixed_point_existence_of_T_l618_618255

-- Question 1: The equation of the line \( l_{3} \)
theorem equation_of_l3 
  (F : ℝ × ℝ) 
  (H_focus : F = (2, 0))
  (k : ℝ) 
  (H_slope : k = 1) : 
  (∀ x y : ℝ, y = k * x + -2 ↔ y = x - 2) := 
sorry

-- Question 2: Line \( l_{1} \) passes through the fixed point (8, 0)
theorem line_l1_through_fixed_point 
  (k m1 : ℝ)
  (H_km1 : k * m1 ≠ 0)
  (H_m1lt : m1 < -t)
  (H_condition : ∃ x y : ℝ, y = k * x + m1 ∧ x^2 + (8/k) * x + (8 * m1 / k) = 0 ∧ ((x, y) = A1 ∨ (x, y) = B1))
  (H_dot_product : (x1 - 0)*(x2 - 0) + (y1 - 0)*(y2 - 0) = 0) : 
  ∀ P : ℝ × ℝ, P = (8, 0) := 
sorry

-- Question 3: Existence of point T such that S_i and d_i form geometric sequences
theorem existence_of_T
  (k : ℝ)
  (H_k : k = 1)
  (m1 m2 m3 : ℝ)
  (H_m_ordered : m1 < m2 ∧ m2 < m3 ∧ m3 < -t)
  (t : ℝ)
  (S1 S2 S3 d1 d2 d3 : ℝ)
  (H_S_geom_seq : S2^2 = S1 * S3)
  (H_d_geom_seq : d2^2 = d1 * d3)
  : ∃ t : ℝ, t = -2 :=
sorry

end equation_of_l3_line_l1_through_fixed_point_existence_of_T_l618_618255


namespace remainder_sum_mod_53_l618_618916

theorem remainder_sum_mod_53 (a b c d : ℕ)
  (h1 : a % 53 = 31)
  (h2 : b % 53 = 45)
  (h3 : c % 53 = 17)
  (h4 : d % 53 = 6) :
  (a + b + c + d) % 53 = 46 := 
sorry

end remainder_sum_mod_53_l618_618916


namespace ellipse_equation_is_standard_form_l618_618426

theorem ellipse_equation_is_standard_form (m n : ℝ) (h_m_pos : m > 0) (h_n_pos : n > 0) (h_mn_neq : m ≠ n) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧ (∀ x y : ℝ, mx^2 + ny^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end ellipse_equation_is_standard_form_l618_618426


namespace who_made_a_mistake_l618_618037

-- Definitions of the conditions
def at_least_four_blue_pencils (B : ℕ) : Prop := B ≥ 4
def at_least_five_green_pencils (G : ℕ) : Prop := G ≥ 5
def at_least_three_blue_and_four_green_pencils (B G : ℕ) : Prop := B ≥ 3 ∧ G ≥ 4
def at_least_four_blue_and_four_green_pencils (B G : ℕ) : Prop := B ≥ 4 ∧ G ≥ 4

-- The main theorem stating who made a mistake
theorem who_made_a_mistake (B G : ℕ) 
  (hv : at_least_four_blue_pencils B)
  (hk : at_least_five_green_pencils G)
  (hp : at_least_three_blue_and_four_green_pencils B G)
  (hm : at_least_four_blue_and_four_green_pencils B G) 
  (h_truth : (hv ∧ hk ∧ hp ∧ hm) ∨ (¬hv ∧ hk ∧ hp ∧ hm) ∨ (hv ∧ ¬hk ∧ hp ∧ hm) ∨ (hv ∧ hk ∧ ¬hp ∧ hm) ∨ (hv ∧ hk ∧ hp ∧ ¬hm))
  (h_truthful: ∑ b in [hv, hk, hp, hm], (if b then 1 else 0) = 3) : 
  hk = false := 
sorry

end who_made_a_mistake_l618_618037


namespace largest_initial_number_l618_618347

theorem largest_initial_number :
  ∃ n : ℕ, (∀ k : ℕ, (n % k ≠ 0 → k ∈ {2, 2, 2, 2, 3}) ∧ (n + 11 = 200)) ∧ (n = 189) :=
begin
  sorry -- Proof not required per instruction
end

end largest_initial_number_l618_618347


namespace find_fraction_l618_618914

-- Define the problem conditions
def N : ℕ := 180
def condition (f : ℚ) : Prop :=
  1/2 * f * 1/5 * N + 6 = 1/15 * N

-- Define the theorem statement we need to prove
theorem find_fraction : ∃ f : ℚ, condition f ∧ f = 1/3 :=
by
  existsi (1/3 : ℚ)
  split
  · unfold condition
    change 1/2 * (1/3) * 1/5 * 180 + 6 = 1/15 * 180
    norm_num
  · rfl

end find_fraction_l618_618914


namespace ellipse_correct_eq_l618_618603

noncomputable
def ellipse_equation (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ 
  (∀ (x y : ℝ), 
  (x, y) = (0, 4) → 
  (x^2 / a^2 + y^2 / b^2 = 1)) ∧ 
  (∃ (e : ℝ), e = 3/5 ∧ e = real.sqrt ((a^2 - b^2) / a^2))

theorem ellipse_correct_eq (a b : ℝ) (h : ellipse_equation a b) : 
  a = 5 ∧ b = 4 → 
  ∀ (x y : ℝ), (x^2 / 25 + y^2 / 16 = 1) := 
by
  intro h_ellipse
  intros x y
  sorry

end ellipse_correct_eq_l618_618603


namespace middle_number_of_consecutive_sum_30_l618_618284

theorem middle_number_of_consecutive_sum_30 (n : ℕ) (h : n + (n + 1) + (n + 2) = 30) : n + 1 = 10 :=
by
  sorry

end middle_number_of_consecutive_sum_30_l618_618284


namespace fraction_of_recipe_l618_618955

theorem fraction_of_recipe 
  (recipe_sugar recipe_milk recipe_flour : ℚ)
  (have_sugar have_milk have_flour : ℚ)
  (h1 : recipe_sugar = 3/4) (h2 : recipe_milk = 2/3) (h3 : recipe_flour = 3/8)
  (h4 : have_sugar = 2/4) (h5 : have_milk = 1/2) (h6 : have_flour = 1/4) : 
  (min ((have_sugar / recipe_sugar)) (min ((have_milk / recipe_milk)) (have_flour / recipe_flour)) = 2/3) := 
by sorry

end fraction_of_recipe_l618_618955


namespace log_expression_identity_l618_618068

-- Statement of the problem in Lean:
theorem log_expression_identity :
  let log4_18 := log 18 / log 4
  let log9_18 := log 18 / log 9
  let log2_9 := log 9 / log 2
  sqrt (log4_18 - log9_18 + log2_9) =
  (3 * log 3 - log 2) / sqrt (2 * log 3 * log 2) :=
sorry

end log_expression_identity_l618_618068


namespace sixthDiagramShadedFraction_l618_618157

-- Definitions based on conditions
def numTriangles (n : ℕ) : ℕ := n^2
def sumOddNumbers (n : ℕ) : ℕ := n^2  -- Using the known formula for the sum of the first n odd numbers

-- Problem statement: proving the fraction of the shaded area in the sixth diagram is 1
theorem sixthDiagramShadedFraction : 
  let n := 6 
  in (sumOddNumbers n / numTriangles n : ℝ) = 1 := by
  sorry

end sixthDiagramShadedFraction_l618_618157


namespace find_19_Diamond_98_l618_618219

noncomputable def Diamond (x y : ℝ) : ℝ := sorry

axiom Diamond_def (x y : ℝ) (h : x > 0) (h' : y > 0) : 
  (x * y) ∆ y = x * (y ∆ y)

axiom Diamond_one (x : ℝ) (h : x > 0) : 
  (x ∆ 1) ∆ x = x ∆ 1

axiom Diamond_one_one : 
  1 ∆ 1 = 1

theorem find_19_Diamond_98 : 
  (19 : ℝ) ∆ (98 : ℝ) = 19 := 
by 
  sorry

end find_19_Diamond_98_l618_618219


namespace smallest_period_f_max_value_h_l618_618243

-- Define the functions f and g
def f (x : ℝ) : ℝ := cos (π / 3 + x) * cos (π / 3 - x)
def g (x : ℝ) : ℝ := 1 / 2 * sin (2 * x) - 1 / 4
def h (x : ℝ) : ℝ := f x - g x

-- Statement 1: The smallest positive period of f(x) is π
theorem smallest_period_f : (∀ x : ℝ, f (x + π) = f x) ∧ (∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T ≥ π) :=
begin
  sorry
end

-- Statement 2: The maximum value of h(x) is sqrt(2)/2, and the set of x values where h(x) attains its maximum
theorem max_value_h :
  (∀ x : ℝ, h x ≤ sqrt 2 / 2) ∧
  (∀ x : ℝ, h x = sqrt 2 / 2 ↔ ∃ k : ℤ, x = 3 * π / 8 + k * π) :=
begin
  sorry
end

end smallest_period_f_max_value_h_l618_618243


namespace quadratic_b_is_negative_sqrt_three_div_three_l618_618592

noncomputable def quadratic_b_value (b : ℝ) (m : ℝ) (c : ℝ) : Prop :=
  c = 1 / 6 ∧
  (x^2 + b * x + c = (x + m)^2 + 1 / 12) ∧
  b < 0 ∧
  b = - (Real.sqrt 3) / 3

theorem quadratic_b_is_negative_sqrt_three_div_three : 
  ∃ (b : ℝ), ∃ (m : ℝ), ∃ (c : ℝ), quadratic_b_value b m c :=
by
  sorry

end quadratic_b_is_negative_sqrt_three_div_three_l618_618592


namespace min_value_of_ratio_l618_618685

theorem min_value_of_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) :
  (4 / x + 1 / y) ≥ 6 + 4 * Real.sqrt 2 :=
sorry

end min_value_of_ratio_l618_618685


namespace single_elimination_games_l618_618991

theorem single_elimination_games (n : ℕ) (h : n = 23) : 
  ∃ g : ℕ, g = n - 1 :=
by
  sorry

end single_elimination_games_l618_618991


namespace GCF_LCM_proof_l618_618174

open Nat

theorem GCF_LCM_proof : 
  let lcm1 := lcm 10 21 in
  let lcm2 := lcm 14 15 in
  gcd lcm1 lcm2 = 210 :=
by
  let lcm1 := lcm 10 21
  let lcm2 := lcm 14 15
  have h1 : lcm1 = 210 := by sorry
  have h2 : lcm2 = 210 := by sorry
  rw [h1, h2]
  exact gcd_self 210

end GCF_LCM_proof_l618_618174


namespace area_of_triangle_hpk_is_500_l618_618749

theorem area_of_triangle_hpk_is_500
  (APK_straight : ∃ P K, ∀ A, ∀ B, A ≠ B → P ≠ K → collinear_points A P K)
  (BPH_straight : ∃ P H, ∀ B, ∀ A, B ≠ A → P ≠ H → collinear_points B P H)
  (angle_BAP_eq_angle_KHP : ∃ A B P K H, angle A B P = 30 ∧ angle K H P = 30)
  (angle_APB_eq_angle_KPH : ∃ A P B K H, angle A P B = angle K P H)
  (length_AB : ∃ A B, line_segment_length A B = 75)
  (length_PH : ∃ P H, line_segment_length P H = 50) :
  ∃ d, d = 500 := 
by sorry

end area_of_triangle_hpk_is_500_l618_618749


namespace num_symmetric_scanning_codes_l618_618593

theorem num_symmetric_scanning_codes:
  let grid := finTuple (8 × 8) bool 
  let is_symmetric (g: grid) := 
    g = rotate90 g ∧ g = rotate180 g ∧ g = rotate270 g ∧ 
    g = reflect_horizontal g ∧ g = reflect_vertical g ∧
    g = reflect_diagonal1 g ∧ g = reflect_diagonal2 g
  let valid_color (g: grid) := ∃ (b w: grid), g ≠ b ∧ g ≠ w
  ∀ g: grid, is_symmetric g → valid_color g → (finset.univ.card = 254) :=
begin
  sorry
end

end num_symmetric_scanning_codes_l618_618593


namespace total_cost_of_crayons_l618_618763

theorem total_cost_of_crayons (crayons_per_half_dozen : ℕ)
    (number_of_half_dozens : ℕ)
    (cost_per_crayon : ℕ)
    (total_cost : ℕ) :
  crayons_per_half_dozen = 6 →
  number_of_half_dozens = 4 →
  cost_per_crayon = 2 →
  total_cost = crayons_per_half_dozen * number_of_half_dozens * cost_per_crayon →
  total_cost = 48 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end total_cost_of_crayons_l618_618763


namespace Kolya_made_the_mistake_l618_618002

def pencils_in_box (blue green : ℕ) : Prop :=
  (blue ≥ 4 ∨ blue < 4) ∧ (green ≥ 4 ∨ green < 4)

def boys_statements (blue green : ℕ) : Prop :=
  (Vasya : blue ≥ 4) ∧
  (Kolya : green ≥ 5) ∧
  (Petya : blue ≥ 3 ∧ green ≥ 4) ∧
  (Misha : blue ≥ 4 ∧ green ≥ 4)

theorem Kolya_made_the_mistake:
  ∀ {blue green : ℕ},
  pencils_in_box blue green →
  boys_statements blue green →
  ∃ (Vasya_truth Petya_truth Misha_truth : Prop),
  Vasya_truth ∧ Petya_truth ∧ Misha_truth ∧ ¬ Kolya_truth :=
begin
  sorry
end

end Kolya_made_the_mistake_l618_618002


namespace largest_initial_number_l618_618332

-- Let's define the conditions and the result
def valid_addition (n a : ℕ) : Prop := ∃ k : ℕ, n = a * k + r ∧ 0 < r ∧ r < a

def valid_operations (initial : ℕ) (final : ℕ) (steps : ℕ → ℕ → ℕ) : Prop :=
  ∃ (a b c d e : ℕ), valid_addition initial a ∧
                      valid_addition (initial + a) b ∧
                      valid_addition (initial + a + b) c ∧
                      valid_addition (initial + a + b + c) d ∧
                      valid_addition (initial + a + b + c + d) e ∧
                      initial + a + b + c + d + e = final

theorem largest_initial_number :
  ∃ n : ℕ, (valid_operations n 200 (λn a, n + a)) ∧ (∀ m : ℕ, valid_operations m 200 (λn a, n + a) → m ≤ n) :=
sorry

end largest_initial_number_l618_618332


namespace net_rate_of_pay_l618_618541

/-- The net rate of pay in dollars per hour for a truck driver after deducting gasoline expenses. -/
theorem net_rate_of_pay
  (hrs : ℕ) (speed : ℕ) (miles_per_gallon : ℕ) (pay_per_mile : ℚ) (cost_per_gallon : ℚ) 
  (H1 : hrs = 3)
  (H2 : speed = 50)
  (H3 : miles_per_gallon = 25)
  (H4 : pay_per_mile = 0.6)
  (H5 : cost_per_gallon = 2.50) :
  pay_per_mile * (hrs * speed) - cost_per_gallon * ((hrs * speed) / miles_per_gallon) = 25 * hrs :=
by sorry

end net_rate_of_pay_l618_618541


namespace angle_BAM_eq_angle_CAN_l618_618667

open Real EuclideanGeometry

theorem angle_BAM_eq_angle_CAN
  (A B C M T N : Point)
  (O : Circle)
  (h_triangle_inscribed : IsInscribedTriangle O A B C)
  (h_angles_acute : IsAcuteAngle B ∧ IsAcuteAngle C)
  (h_M_arc_BC : IsOnArcNotContainingA O B C M)
  (h_AM_not_perpendicular_BC : ¬IsPerpendicular AM BC)
  (h_T : IsIntersection AM (PerpendicularBisector B C) T)
  (h_N : OtherIntersection (Circumcircle A O T) O N A) :
  ∠BAM = ∠CAN :=
sorry

end angle_BAM_eq_angle_CAN_l618_618667


namespace largest_initial_number_l618_618338

-- Let's define the conditions and the result
def valid_addition (n a : ℕ) : Prop := ∃ k : ℕ, n = a * k + r ∧ 0 < r ∧ r < a

def valid_operations (initial : ℕ) (final : ℕ) (steps : ℕ → ℕ → ℕ) : Prop :=
  ∃ (a b c d e : ℕ), valid_addition initial a ∧
                      valid_addition (initial + a) b ∧
                      valid_addition (initial + a + b) c ∧
                      valid_addition (initial + a + b + c) d ∧
                      valid_addition (initial + a + b + c + d) e ∧
                      initial + a + b + c + d + e = final

theorem largest_initial_number :
  ∃ n : ℕ, (valid_operations n 200 (λn a, n + a)) ∧ (∀ m : ℕ, valid_operations m 200 (λn a, n + a) → m ≤ n) :=
sorry

end largest_initial_number_l618_618338


namespace minimum_M_l618_618450

def sequence (n : ℕ) : ℚ := 1 / ((2 * n + 1) * (2 * n + 3))

def partial_sum (n : ℕ) : ℚ :=
  ∑ i in finset.range n, sequence i

theorem minimum_M (M : ℚ) (h : ∀ n, partial_sum n < M) : M ≥ 1 / 6 :=
by {
  sorry
}

end minimum_M_l618_618450


namespace length_AB_distance_PM_l618_618296

-- Definitions for the parametric line and curve
def parametric_line (t : ℝ) : ℝ × ℝ := (-2 - 3 * t, 2 - 4 * t)

def curve (x y : ℝ) : Prop := (y - 2) ^ 2 - x ^ 2 = 1

-- Parameters t1 and t2 from the intersection points
variable (t1 t2 : ℝ)
-- Their properties
axiom t_params : 7 * t1^2 - 12 * t1 - 5 = 0 ∧ 7 * t2^2 - 12 * t2 - 5 = 0

-- Midpoint M's parameter
def tM : ℝ := (t1 + t2) / 2

-- Conditions on t1 and t2
axiom t_relation : t1 + t2 = 12 / 7 ∧ t1 * t2 = -5 / 7

-- Theorem 1: Prove length of |AB|
theorem length_AB : ∃ t1 t2, (t_params t1 t2 ∧ t_relation t1 t2) → 
  let A := parametric_line t1, B := parametric_line t2 in
  real.dist A B = (10 * real.sqrt 71) / 7 := by
  sorry

-- Polar coordinates of point P
def P : ℝ × ℝ := (-2, 2)

-- Cartesian coordinates of midpoint M
def M : ℝ × ℝ := parametric_line tM

-- Theorem 2: Prove distance between P and M
theorem distance_PM : ∃ t1 t2, (t_params t1 t2 ∧ t_relation t1 t2) → 
  real.dist P M = 30 / 7 := by
  sorry

end length_AB_distance_PM_l618_618296


namespace largest_subset_no_member_is_4_times_another_l618_618979

-- Define the predicate that characterizes the subset
def valid_subset (S : Set ℕ) : Prop :=
  ∀ (x ∈ S) (y ∈ S), x ≠ 4 * y ∧ y ≠ 4 * x

-- Define the set of integers from 1 to 150
def range_1_to_150 : Set ℕ := {n | 1 ≤ n ∧ n ≤ 150}

-- State the theorem
theorem largest_subset_no_member_is_4_times_another :
  ∃ S ⊆ range_1_to_150, valid_subset S ∧ S.card = 140 := sorry

end largest_subset_no_member_is_4_times_another_l618_618979


namespace kolya_is_wrong_l618_618031

def pencils_problem_statement (at_least_four_blue : Prop) 
                              (at_least_five_green : Prop) 
                              (at_least_three_blue_and_four_green : Prop) 
                              (at_least_four_blue_and_four_green : Prop) : 
                              Prop :=
  ∃ (B G : ℕ), -- B represents the number of blue pencils, G represents the number of green pencils
    ((B ≥ 4) ∧ (G ≥ 4)) ∧ -- Vasya's statement (at least 4 blue), Petya's and Misha's combined statement (at least 4 green)
    at_least_four_blue ∧ -- Vasya's statement (there are at least 4 blue pencils)
    (at_least_five_green ↔ G ≥ 5) ∧ -- Kolya's statement (there are at least 5 green pencils)
    at_least_three_blue_and_four_green ∧ -- Petya's statement (at least 3 blue and 4 green)
    at_least_four_blue_and_four_green -- Misha's statement (at least 4 blue and 4 green)

theorem kolya_is_wrong (at_least_four_blue : Prop) 
                        (at_least_five_green : Prop) 
                        (at_least_three_blue_and_four_green : Prop) 
                        (at_least_four_blue_and_four_green : Prop) : 
                        pencils_problem_statement at_least_four_blue 
                                                  at_least_five_green 
                                                  at_least_three_blue_and_four_green 
                                                  at_least_four_blue_and_four_green :=
sorry

end kolya_is_wrong_l618_618031


namespace functional_eq_solution_l618_618624

open Real

theorem functional_eq_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = f (x - y) + 4 * x * y) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x^2 + c := 
sorry

end functional_eq_solution_l618_618624


namespace plastering_cost_correct_l618_618536

noncomputable def tank_length : ℝ := 25
noncomputable def tank_width : ℝ := 12
noncomputable def tank_depth : ℝ := 6
noncomputable def cost_per_sqm_paise : ℝ := 75
noncomputable def cost_per_sqm_rupees : ℝ := cost_per_sqm_paise / 100

noncomputable def total_cost_plastering : ℝ :=
  let long_wall_area := 2 * (tank_length * tank_depth)
  let short_wall_area := 2 * (tank_width * tank_depth)
  let bottom_area := tank_length * tank_width
  let total_area := long_wall_area + short_wall_area + bottom_area
  total_area * cost_per_sqm_rupees

theorem plastering_cost_correct : total_cost_plastering = 558 := by
  sorry

end plastering_cost_correct_l618_618536


namespace unique_solution_to_system_l618_618589

theorem unique_solution_to_system :
  ∀ (a b c d : ℝ),
    a + b + c + d = real.pi →
    (∃! (x y z w : ℝ), 
      x = real.sin (a + b) ∧ 
      y = real.sin (b + c) ∧ 
      z = real.sin (c + d) ∧ 
      w = real.sin (d + a)) :=
by
  sorry

end unique_solution_to_system_l618_618589


namespace average_points_top_three_teams_l618_618617

theorem average_points_top_three_teams :
  let points (wins ties : ℕ) := 2 * wins + ties
  let points_A := points 14 3
  let points_B := points 13 2
  let points_C := points 11 4
  let points_D := points 8 10
  let points_E := points 10 5
  let points_F := points 12 1
  let points_G := points 9 7
  let points_H := points 6 8
  let top_points := [points_A, points_B, points_C, points_D].removeAll [points_E, points_F, points_G, points_H]
  let average_points := (top_points.sum) / 4
  average_points = 27.75 := 
by
  -- The proof will verify this statement
  sorry

end average_points_top_three_teams_l618_618617


namespace numNonCongruentNormalPolygons_l618_618149

-- Definition of a normal polygon (inscribed in a unit circle with specific side lengths)
def isNormalPolygon (poly : Polygon) : Prop :=
  (∀ side ∈ poly.sides, side ∈ {1, sqrt 2, sqrt 3, 2}) ∧ 
  (poly.canBeInscribedInUnitCircle)

-- Theorem stating the number of such non-congruent polygons
theorem numNonCongruentNormalPolygons : 
  ∃ (count : ℕ), count = 14 ∧ 
    (∀ (poly1 poly2 : Polygon), 
      isNormalPolygon poly1 → isNormalPolygon poly2 → 
      (poly1 ≃ poly2 ↔ poly1 = poly2)) :=
by
  sorry

end numNonCongruentNormalPolygons_l618_618149


namespace jasmine_added_l618_618508

theorem jasmine_added (J : ℝ) : 
  let initial_volume := 90
      initial_percent_jasmine := 0.05
      initial_jasmine := initial_volume * initial_percent_jasmine
      water_added := 2
      final_volume := initial_volume + J + water_added
      final_percent_jasmine := 0.125
      final_jasmine := final_volume * final_percent_jasmine
  in
  initial_jasmine + J = final_jasmine → 
  J = 8 :=
begin
  sorry
end

end jasmine_added_l618_618508


namespace ant_reaches_bottom_central_vertex_prob_l618_618155

noncomputable def probability_reaches_bottom_central_vertex : ℝ :=
  1 / 3

theorem ant_reaches_bottom_central_vertex_prob :
  ∃ (P : ℝ), P = probability_reaches_bottom_central_vertex := by
  use probability_reaches_bottom_central_vertex
  sorry

end ant_reaches_bottom_central_vertex_prob_l618_618155


namespace greatest_possible_value_l618_618907

theorem greatest_possible_value (x y : ℝ) (h1 : x^2 + y^2 = 98) (h2 : x * y = 40) : x + y = Real.sqrt 178 :=
by sorry

end greatest_possible_value_l618_618907


namespace hypotenuse_length_l618_618499

-- Definitions based on the conditions
def is_isosceles_right_triangle (a c : ℝ) :=
  a^2 + a^2 = c^2

def perimeter (a c : ℝ) :=
  2 * a + c

-- Given conditions
def given_perimeter : ℝ :=
  14 + 14 * Real.sqrt 2

def a_in_terms_of_c (c : ℝ) : ℝ :=
  c / Real.sqrt 2

-- The proof statement
theorem hypotenuse_length (c : ℝ) (h : perimeter (a_in_terms_of_c c) c = given_perimeter)
  (h_iso : is_isosceles_right_triangle (a_in_terms_of_c c) c) :
  c = 28 :=
sorry

end hypotenuse_length_l618_618499


namespace max_initial_number_l618_618331

noncomputable def verify_addition (n x : ℕ) : Prop := 
  ∀ (a : ℕ), (a ∣ n) → (a ≠ 1) → (n + a = x) → False

theorem max_initial_number :
  ∃ (n : ℕ), 
  (∀ (a1 a2 a3 a4 a5 : ℕ), 
    verify_addition n a1 ∧ verify_addition (n + a1) a2 ∧
    verify_addition (n + a1 + a2) a3 ∧ verify_addition (n + a1 + a2 + a3) a4 ∧
    verify_addition (n + a1 + a2 + a3 + a4) a5 ∧
    (n + a1 + a2 + a3 + a4 + a5 = 200)) ∧
  (∀ m : ℕ, 
    (∃ (a1 a2 a3 a4 a5 : ℕ), 
      verify_addition m a1 ∧ verify_addition (m + a1) a2 ∧
      verify_addition (m + a1 + a2) a3 ∧ verify_addition (m + a1 + a2 + a3) a4 ∧
      verify_addition (m + a1 + a2 + a3 + a4) a5 ∧
      (m + a1 + a2 + a3 + a4 + a5 = 200)) →
    m ≤ 189)
: ∃ n, n = 189 := by
  sorry

end max_initial_number_l618_618331


namespace symmetry_translation_l618_618207

variable {Plane : Type} [AffinePlane Plane]
variable {Point : Type} [AffinePoint Point]
variable (l : Line Plane) (M M₁ M₂ : Point)
variable (h : ℝ)

-- Definitions of symmetries and translations
noncomputable def symmetric_point (p : Point) (l : Line Plane) : Point := sorry
noncomputable def translate_line (l : Line Plane) (h : ℝ) : Line Plane := sorry
noncomputable def translate_point (p : Point) (h : ℝ) : Point := sorry

theorem symmetry_translation {l l₁ : Line Plane} {M M₁ M₂ : Point} (h : ℝ) :
  symmetric_point M l = M₁ ∧
  l₁ = translate_line l h ∧
  symmetric_point M l₁ = M₂ →
  M₂ = translate_point M₁ (2 * h) := 
sorry

end symmetry_translation_l618_618207


namespace clock_hands_overlap_24_hours_l618_618266

theorem clock_hands_overlap_24_hours : 
  (∀ t : ℕ, t < 12 →  ∃ n : ℕ, (n = 11 ∧ (∃ h m : ℕ, h * 60 + m = t * 60 + m))) →
  (∃ k : ℕ, k = 22) :=
by
  sorry

end clock_hands_overlap_24_hours_l618_618266


namespace origin_moves_distance_l618_618519

noncomputable def origin_distance_moved : ℝ :=
  let B := (3, 1)
  let B' := (7, 9)
  let k := 1.5
  let center_of_dilation := (-1, -3)
  let d0 := Real.sqrt ((-1)^2 + (-3)^2)
  let d1 := k * d0
  d1 - d0

theorem origin_moves_distance :
  origin_distance_moved = 0.5 * Real.sqrt 10 :=
by 
  sorry

end origin_moves_distance_l618_618519


namespace chef_total_guests_served_l618_618948

def chef_served_guests : ℕ :=
  let adults := 58
  let children := 58 - 35
  let seniors := 2 * children
  let teenagers := seniors - 15
  let toddlers := teenagers / 2
  let vip_guests := max (takeWhile (fun x => (x:ℕ) * x < teenagers) (List.range teenagers.default)).default
  adults + children + seniors + teenagers + toddlers + vip_guests = 198

theorem chef_total_guests_served : chef_served_guests = 198 := by
  sorry

end chef_total_guests_served_l618_618948


namespace security_guards_night_shift_l618_618094

/-- 
A certain number of security guards were hired for the night shift at a factory. 
They agreed to a rotating schedule to cover the nine hours of the night shift. 
The first guard would take three hours, the last guard would take two hours, 
and the middle guards would split the remaining hours each taking two hours.
Prove that the total number of guards hired for the night shift is 4. 
-/
theorem security_guards_night_shift : 
  ∀ (night_shift_hours first_guard_hours last_guard_hours middle_guard_hours : ℕ),
  night_shift_hours = 9 →
  first_guard_hours = 3 →
  last_guard_hours = 2 →
  middle_guard_hours = 2 →
  let remaining_hours := night_shift_hours - (first_guard_hours + last_guard_hours) in
  let middle_guards := remaining_hours / middle_guard_hours in
  1 + middle_guards + 1 = 4 :=
by
  intros night_shift_hours first_guard_hours last_guard_hours middle_guard_hours h1 h2 h3 h4
  simp
  sorry

end security_guards_night_shift_l618_618094


namespace evaluate_expr_l618_618597

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5

theorem evaluate_expr : 3 * g 2 + 2 * g (-4) = 169 :=
by
  sorry

end evaluate_expr_l618_618597


namespace p_neg_one_zero_geometric_p_linear_arithmetic_l618_618666

/-- Define the sequence as a geometric sequence with the first term 1 and common ratio 2 -/
def a_geometric (n : ℕ) : ℕ := 2^n

/-- Define the polynomial p(x) based on the sequence a_n -/
def p (x : ℝ) (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), (a_geometric (k) * Nat.choose n k) * x^k * (1-x)^(n-k)

/-- Proof Problem (1): Prove that p(-1) = 0 if a_n is a geometric sequence with a common ratio of 2 -/
theorem p_neg_one_zero_geometric : ∀ n : ℕ, p (-1) n = 0 :=
by
  sorry

/-- Define the sequence as an arithmetic sequence with the first term 1 and common difference 2 -/
def a_arithmetic (n : ℕ) : ℕ := 2*n - 1

/-- Define the polynomial p(x) based on the sequence a_n -/
def p_arith (x : ℝ) (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), (a_arithmetic (k) * Nat.choose n k) * x^k * (1-x)^(n-k)

/-- Proof Problem (2): Prove that p(x) is a linear polynomial if a_n is an arithmetic sequence with a common difference of 2 -/
theorem p_linear_arithmetic : ∀ n : ℕ, ∃ a b : ℝ, ∀ x : ℝ, p_arith x n = a + b * x :=
by
  sorry

end p_neg_one_zero_geometric_p_linear_arithmetic_l618_618666


namespace minimum_coefficient_x2_sum_odd_powers_x_l618_618660

def f (x : ℝ) (m n : ℕ) : ℝ := (1 + x)^m + (1 + x)^n

theorem minimum_coefficient_x2_sum_odd_powers_x
    (m n : ℕ)
    (hm : m > 0) (hn : n > 0)
    (h_min_coeff : (m = 5 ∧ n = 3) ∨ (m = 3 ∧ n = 5)) :
  let c2 := (m^2 - m)/2 + 2*n*(n - 1)
  let sum_odd := (1 + 1 : ℝ)^m + (1 + 2)^n in
  c2 = 22 ∧ sum_odd = 30 :=
by
  sorry

end minimum_coefficient_x2_sum_odd_powers_x_l618_618660


namespace Kolya_mistake_l618_618024

def boys := ["Vasya", "Kolya", "Petya", "Misha"]

constant num_blue_pencils : ℕ
constant num_green_pencils : ℕ

axiom Vasya_statement : num_blue_pencils >= 4
axiom Kolya_statement : num_green_pencils >= 5
axiom Petya_statement : num_blue_pencils >= 3 ∧ num_green_pencils >= 4
axiom Misha_statement : num_blue_pencils >= 4 ∧ num_green_pencils >= 4

axiom three_truths_one_mistake : 
  (Vasya_statement ∨ ¬Vasya_statement) ∧
  (Kolya_statement ∨ ¬Kolya_statement) ∧
  (Petya_statement ∨ ¬Petya_statement) ∧
  (Misha_statement ∨ ¬Misha_statement) ∧
  ((Vasya_statement ? true : 1) + 
   (Kolya_statement ? true : 1) + 
   (Petya_statement ? true : 1) +
   (Misha_statement ? true : 1) == 3)

theorem Kolya_mistake : ¬Kolya_statement :=
by
  sorry

end Kolya_mistake_l618_618024


namespace VolunteersAssignment_l618_618854

-- Definitions of volunteers and helper functions
inductive Volunteer
| A
| B
| C
| D
| E

def Group := List Volunteer

def assignment (groups : List Group) : Prop :=
  groups.length = 3 ∧
  (∀ g, g ∈ groups → g.length ≥ 1) ∧
  (∀ g, Volunteer.A ∈ g → Volunteer.B ∉ g) ∧
  (∀ g, Volunteer.C ∈ g → Volunteer.D ∉ g)

-- The proof statement
theorem VolunteersAssignment :
  ∃ (groups : List Group), assignment groups ∧ (number_of_assignments = 288) := by
  sorry

end VolunteersAssignment_l618_618854


namespace geometric_sequence_common_ratio_l618_618741

variable {α : Type*} [Field α]
variable (a : ℕ → α) (q : α)

def geometric_sequence (a : ℕ → α) (q : α) : Prop :=
  ∀ n, a (n + 1) = a n * q

def sum_of_first_n_terms (a : ℕ → α) (q : α) (n : ℕ) : α := 
  if h : q = 1 then
    (n + 1) * a 0
  else
    a 0 * (1 - q^(n + 1)) / (1 - q)

theorem geometric_sequence_common_ratio {a : ℕ → α} {q : α}
  (h_geo : geometric_sequence a q)
  (h_sum3 : sum_of_first_n_terms a q 2 = 15)
  (h_a3 : a 2 = 5) :
  q = 1 ∨ q = - (1 / 2) := 
sorry

end geometric_sequence_common_ratio_l618_618741


namespace arithmetic_expression_value_l618_618477

theorem arithmetic_expression_value : 4 * (8 - 3 + 2) - 7 = 21 := by
  sorry

end arithmetic_expression_value_l618_618477


namespace largest_subset_no_four_times_another_l618_618961

theorem largest_subset_no_four_times_another :
  ∃ (S : set ℕ), S ⊆ {1, 2, ..., 150} ∧ (∀ (a b : ℕ), a ∈ S → b ∈ S → (a ≠ 4 * b ∧ b ≠ 4 * a)) ∧ (S.card = 141) :=
sorry

end largest_subset_no_four_times_another_l618_618961


namespace find_coefficients_l618_618301

-- Define points A, B, C and their corresponding vectors
variables (A B C P D E : Point)
variables (x y z : Rat)

-- Given conditions as hypotheses
axiom BD_DC_ratio: BD : DC = 4 : 1
axiom AE_EC_ratio: AE : EC = 2 : 1
axiom P_intersection: Intersect (line BE) (line AD) = P

-- The theorem to prove
theorem find_coefficients :
  P = x • A + y • B + z • C ∧ x + y + z = 1 :=
  sorry

end find_coefficients_l618_618301


namespace parabola_boundaries_l618_618401

theorem parabola_boundaries (n : ℕ) (parabolas : Fin n → Polynomial ℝ) 
  (h_non_touching : ∀ i j, i ≠ j → ∀ x, parabolas i x ≠ parabolas j x)
  (h_quadratic : ∀ i, parabolas i.degree = 2) :
  ∃ R : Set (ℝ × ℝ), 
    (∀ x y, (x, y) ∈ R → ∀ i, y ≥ parabolas i x) ∧
    (∃ S : Set (ℝ × ℝ), R ⊆ S ∧ (∀ p ∈ S, ∃ q r, q ≠ r ∧ parabolas q p.1 = p.2 ∧ parabolas r p.1 = p.2)
    ∧ Finset.card (S ∩ { p : ℝ × ℝ | ∃ i j, i ≠ j ∧ parabolas i p.1 = p.2 ∧ parabolas j p.1 = p.2 }) ≤ 2 * (n - 1)) :=
sorry

end parabola_boundaries_l618_618401


namespace correct_avg_weight_of_class_l618_618737

theorem correct_avg_weight_of_class :
  ∀ (n : ℕ) (avg_wt : ℝ) (mis_A mis_B mis_C actual_A actual_B actual_C : ℝ),
  n = 30 →
  avg_wt = 60.2 →
  mis_A = 54 → actual_A = 64 →
  mis_B = 58 → actual_B = 68 →
  mis_C = 50 → actual_C = 60 →
  (n * avg_wt + (actual_A - mis_A) + (actual_B - mis_B) + (actual_C - mis_C)) / n = 61.2 :=
by
  intros n avg_wt mis_A mis_B mis_C actual_A actual_B actual_C h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end correct_avg_weight_of_class_l618_618737


namespace largest_subset_size_l618_618971

def largest_subset_with_property (s : Set ℤ) : Prop :=
  ∀ (a b : ℤ), a ∈ s → b ∈ s → (a ≠ 4 * b ∧ b ≠ 4 * a)

theorem largest_subset_size : ∃ s : Set ℤ, (∀ x : ℤ, x ∈ s → 1 ≤ x ∧ x ≤ 150) ∧ largest_subset_with_property s ∧ s.card = 120 :=
by
  sorry

end largest_subset_size_l618_618971


namespace max_books_single_student_l618_618739

theorem max_books_single_student (total_students : ℕ) (students_0_books : ℕ) (students_1_book : ℕ) (students_2_books : ℕ) (avg_books_per_student : ℕ) :
  total_students = 20 →
  students_0_books = 3 →
  students_1_book = 9 →
  students_2_books = 4 →
  avg_books_per_student = 2 →
  ∃ max_books : ℕ, max_books = 14 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end max_books_single_student_l618_618739


namespace who_made_mistake_l618_618015

-- Defining conditions for the colored pencils
def has_at_least_four_blue_pencils (b : Nat) : Prop := b >= 4
def has_at_least_five_green_pencils (g : Nat) : Prop := g >= 5
def has_at_least_three_blue_four_green_pencils (b g : Nat) : Prop := b >= 3 ∧ g >= 4
def has_at_least_four_blue_four_green_pencils (b g : Nat) : Prop := b >= 4 ∧ g >= 4

-- Statement of the problem
theorem who_made_mistake (b g : Nat) (vasya kolya petya misha : Prop) :
  has_at_least_four_blue_pencils b →
  has_at_least_five_green_pencils g →
  has_at_least_three_blue_four_green_pencils b g →
  has_at_least_four_blue_four_green_pencils b g →
  (∃ T : Set Prop, {vasya, kolya, petya, misha}.Erase T = {vasya, kolya, petya, misha} ∧
    T.Card = 3) →
  (kolya ↔ ¬ g >= 5) := 
sorry

end who_made_mistake_l618_618015


namespace max_initial_number_l618_618325

noncomputable def verify_addition (n x : ℕ) : Prop := 
  ∀ (a : ℕ), (a ∣ n) → (a ≠ 1) → (n + a = x) → False

theorem max_initial_number :
  ∃ (n : ℕ), 
  (∀ (a1 a2 a3 a4 a5 : ℕ), 
    verify_addition n a1 ∧ verify_addition (n + a1) a2 ∧
    verify_addition (n + a1 + a2) a3 ∧ verify_addition (n + a1 + a2 + a3) a4 ∧
    verify_addition (n + a1 + a2 + a3 + a4) a5 ∧
    (n + a1 + a2 + a3 + a4 + a5 = 200)) ∧
  (∀ m : ℕ, 
    (∃ (a1 a2 a3 a4 a5 : ℕ), 
      verify_addition m a1 ∧ verify_addition (m + a1) a2 ∧
      verify_addition (m + a1 + a2) a3 ∧ verify_addition (m + a1 + a2 + a3) a4 ∧
      verify_addition (m + a1 + a2 + a3 + a4) a5 ∧
      (m + a1 + a2 + a3 + a4 + a5 = 200)) →
    m ≤ 189)
: ∃ n, n = 189 := by
  sorry

end max_initial_number_l618_618325


namespace production_cost_increase_maximize_profit_value_l618_618042

noncomputable def last_year_production_cost : ℝ := 100000
noncomputable def last_year_factory_price : ℝ := 130000
noncomputable def last_year_sales_volume : ℝ := 5000
noncomputable def last_year_profit : ℝ := (last_year_factory_price - last_year_production_cost) * last_year_sales_volume / 1000

noncomputable def this_year_production_cost (x : ℝ) : ℝ := 100000 * (1 + x)
noncomputable def this_year_factory_price (x : ℝ) : ℝ := 130000 * (1 + 0.7 * x)
noncomputable def this_year_sales_volume (x : ℝ) : ℝ := 5000 * (1 + 0.4 * x)
noncomputable def this_year_profit (x : ℝ) : ℝ := (this_year_factory_price x - this_year_production_cost x) * this_year_sales_volume x / 1000

noncomputable def profit_increase_condition (x : ℝ) : Prop :=
  this_year_profit x > last_year_profit

noncomputable def f (x : ℝ) : ℝ :=
  3240 * (0.9 * x^3 - 4.8 * x^2 + 4.5 * x + 5)

noncomputable def maximize_profit_condition (x : ℝ) : Prop :=
  ∀ y ∈ Ioo 0 1, f x ≥ f y

theorem production_cost_increase (x : ℝ) (hx : 0 < x ∧ x < 15/18) :
  profit_increase_condition x := sorry

theorem maximize_profit_value : ∃ x ∈ Ioo 0 1, maximize_profit_condition x ∧ f x = 20000 := sorry

end production_cost_increase_maximize_profit_value_l618_618042


namespace largest_initial_number_l618_618315

theorem largest_initial_number (n : ℕ) (h : (∃ a b c d e : ℕ, n ≠ 0 ∧ n + a + b + c + d + e = 200 
                                              ∧ n % a ≠ 0 ∧ n % b ≠ 0 ∧ n % c ≠ 0 ∧ n % d ≠ 0 ∧ n % e ≠ 0)) 
: n ≤ 189 :=
sorry

end largest_initial_number_l618_618315


namespace ships_meeting_count_l618_618816

theorem ships_meeting_count :
  ∀ (n : ℕ) (east_sailing west_sailing : ℕ),
    n = 10 →
    east_sailing = 5 →
    west_sailing = 5 →
    east_sailing + west_sailing = n →
    (∀ (v : ℕ), v > 0) →
    25 = east_sailing * west_sailing :=
by
  intros n east_sailing west_sailing h1 h2 h3 h4 h5
  sorry

end ships_meeting_count_l618_618816


namespace skater_speeds_l618_618834

-- Definitions
def constant_speeds (V1 V2 : ℝ) : Prop := V1 > 0 ∧ V2 > 0
def meeting_time (L V1 V2 : ℝ) : ℝ := L / (V1 + V2)
def overtaking_time (L V1 V2 : ℝ) : ℝ := L / (abs (V1 - V2))
def frequency_relation (L V1 V2 : ℝ) : Prop := (overtaking_time L V1 V2) / (meeting_time L V1 V2) = 4

-- Theorem
theorem skater_speeds (L V1 V2 : ℝ) (h1 : constant_speeds V1 V2) 
    (h2 : frequency_relation L V1 V2) 
    (h3 : V1 = 6 ∨ V2 = 6) : V1 = 6 ∧ V2 = 3.6 ∨ V1 = 10 ∧ V2 = 6 :=
by
  sorry

end skater_speeds_l618_618834


namespace compare_f_values_l618_618521

-- Define the function and its properties
def f : ℝ → ℝ := sorry

-- Define the conditions
axiom even_f : ∀ x, f(x) = f(-x)
axiom periodic_f : ∀ x, f(x + 1) = -f(x)
axiom mono_f : ∀ x1 x2, 0 < x1 ∧ x1 ≤ 1 → 0 < x2 ∧ x2 ≤ 1 → x1 < x2 → f(x1) < f(x2)

-- Define the proof problem
theorem compare_f_values : f(1/3) < f(5/2) ∧ f(5/2) < f(-5) := 
by {
  sorry
}

end compare_f_values_l618_618521


namespace inequality_l618_618196

variable {n : ℕ}
variable {t : Fin n → ℝ}

def valid_sequence (t : Fin n → ℝ) : Prop :=
  (∀ i, 0 < t i ∧ t i < 1) ∧
  (∀ i j, i ≤ j → t i ≤ t j)

theorem inequality (h : valid_sequence t) :
  (1 - t (Fin.last n)) * (∑ i : Fin n, t i / (1 - t i ^ (i + 1 + 1))^2) < 1 :=
sorry

end inequality_l618_618196


namespace area_of_B_l618_618158

open Complex

def in_region (z : ℂ) : Prop :=
  let x := z.re
  let y := z.im
  (0 ≤ x ∧ x ≤ 60) ∧ (0 ≤ y ∧ y ≤ 60) ∧ 
  (60 * x / (x^2 + y^2) ∈ Icc 0 1) ∧ (60 * y / (x^2 + y^2) ∈ Icc 0 1) ∧
  (x^2 + (y - 30)^2 ≥ 30^2)

def region_B_area : ℝ := 1800 + 450 * Real.pi

theorem area_of_B : ∀ (z : ℂ), in_region z → (region_B_area = 1800 + 450 * Real.pi) :=
by
  sorry

end area_of_B_l618_618158


namespace continuous_arrow_loop_encircling_rectangle_l618_618614

def total_orientations : ℕ := 2^4

def favorable_orientations : ℕ := 2 * 2

def probability_loop : ℚ := favorable_orientations / total_orientations

theorem continuous_arrow_loop_encircling_rectangle : probability_loop = 1 / 4 := by
  sorry

end continuous_arrow_loop_encircling_rectangle_l618_618614


namespace distance_between_lines_l618_618428

theorem distance_between_lines : 
  ∀ (x y : ℝ),
  3 * x + 4 * y - 2 = 0 →
  6 * x + 8 * y + 1 = 0 →
  distance_between_parallel_lines 3 4 (-2) 6 8 1 = 1/2 := 
by sorry

end distance_between_lines_l618_618428


namespace max_initial_number_l618_618328

noncomputable def verify_addition (n x : ℕ) : Prop := 
  ∀ (a : ℕ), (a ∣ n) → (a ≠ 1) → (n + a = x) → False

theorem max_initial_number :
  ∃ (n : ℕ), 
  (∀ (a1 a2 a3 a4 a5 : ℕ), 
    verify_addition n a1 ∧ verify_addition (n + a1) a2 ∧
    verify_addition (n + a1 + a2) a3 ∧ verify_addition (n + a1 + a2 + a3) a4 ∧
    verify_addition (n + a1 + a2 + a3 + a4) a5 ∧
    (n + a1 + a2 + a3 + a4 + a5 = 200)) ∧
  (∀ m : ℕ, 
    (∃ (a1 a2 a3 a4 a5 : ℕ), 
      verify_addition m a1 ∧ verify_addition (m + a1) a2 ∧
      verify_addition (m + a1 + a2) a3 ∧ verify_addition (m + a1 + a2 + a3) a4 ∧
      verify_addition (m + a1 + a2 + a3 + a4) a5 ∧
      (m + a1 + a2 + a3 + a4 + a5 = 200)) →
    m ≤ 189)
: ∃ n, n = 189 := by
  sorry

end max_initial_number_l618_618328


namespace travis_should_be_paid_l618_618893

theorem travis_should_be_paid :
  let (total_bowls, lost_bowls, broken_bowls) := (638, 12, 15) in
  let (fee_per_safe_bowl, fee_for_lost_bowl, fixed_fee) := (3, 4, 100) in
  (total_bowls - lost_bowls - broken_bowls) * fee_per_safe_bowl + fixed_fee - (lost_bowls + broken_bowls) * fee_for_lost_bowl = 1825 :=
by
  -- sorry as placeholder for the proof
  sorry

end travis_should_be_paid_l618_618893


namespace hyperbola_eccentricity_l618_618866

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
    (h3 : ∀ x, y = 2 * x) 
    (h4 : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1) :
    (∃ e, e = sqrt 5) :=
by
  sorry

end hyperbola_eccentricity_l618_618866


namespace trigonometric_identity_l618_618585

noncomputable def trigonometric_identity_proof : Prop :=
  let cos_30 := Real.sqrt 3 / 2;
  let sin_60 := Real.sqrt 3 / 2;
  let sin_30 := 1 / 2;
  let cos_60 := 1 / 2;
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 1

theorem trigonometric_identity : trigonometric_identity_proof :=
  sorry

end trigonometric_identity_l618_618585


namespace tiling_condition_l618_618902

theorem tiling_condition (a b n : ℕ) : 
  (∃ f : ℕ → ℕ × ℕ, ∀ i < (a * b) / n, (f i).fst < a ∧ (f i).snd < b) ↔ (n ∣ a ∨ n ∣ b) :=
sorry

end tiling_condition_l618_618902


namespace largest_initial_number_l618_618305

theorem largest_initial_number : ∃ n : ℕ, (n + 5 ∑ k : ℕ, k ≠ 0 ∧ ¬ (n % k = 0)) = 200 ∧ n = 189 :=
begin
  sorry
end

end largest_initial_number_l618_618305


namespace triangle_trig_proof_l618_618732

-- Define the problem statements
theorem triangle_trig_proof (A B C : ℝ) (AB AC BC : ℝ) (cosB sinB cosA sinA sinC : ℝ) :
  A = 2 * B →
  sin B = 1 / 3 →
  AB = 23 →
  sin A = 2 * sin B * cos B →
  cos B = sqrt (1 - sin B ^ 2) →
  sin A = 2 * (1 / 3) * (2 * sqrt 2 / 3) →
  sin C = sin A * cos B + cos A * sin B →
  AC = AB * sin B / sin C →
  BC = AB * sin A / sin C →
  cos C = - (cos A * cos B) + (sin A * sin B) →
  sin A = 4 * sqrt 2 / 9 → 
  sin C = 23 / 27 →
  cos A = cos (2 * B) →
  cos A = ((2 * sqrt 2 / 3) ^ 2) - (1 / 3 ^ 2) →
  cos C = - (7 / 9 * 2 * sqrt 2 / 3) + 4 * sqrt 2 / 9 * 1 / 3 →
  (AC * BC * cos C) = -80 :=
begin
  -- conditions and proofs would be here
  sorry,
end

end triangle_trig_proof_l618_618732


namespace count_propositions_l618_618555

def is_proposition (s : Prop) : Bool := true

def is_true_proposition (s : Prop) : Bool := Bool.ofNat (if s then 1 else 0)

theorem count_propositions :
  let p1 := ¬ is_proposition (∃ x : ℕ, |x + 2|)
  let p2 := is_proposition ( -5 ∈ ℤ)
  let p3 := is_proposition (π ∉ ℝ)
  let p4 := is_proposition ({0} ∈ ℕ) in
  p1 + p2 + p3 + p4 = 3 := sorry

end count_propositions_l618_618555


namespace max_initial_number_l618_618343

theorem max_initial_number (n : ℕ) : 
  (∃ (a b c d e : ℕ), 
    200 = n + a + b + c + d + e ∧ 
    ¬ (n % a = 0) ∧ 
    ¬ ((n + a) % b = 0) ∧ 
    ¬ ((n + a + b) % c = 0) ∧ 
    ¬ ((n + a + b + c) % d = 0) ∧ 
    ¬ ((n + a + b + c + d) % e = 0)) → 
  n ≤ 189 := 
sorry

end max_initial_number_l618_618343


namespace max_rented_trucks_l618_618498

-- Definitions according to the conditions
def total_trucks := 20
def percentage_returned := 0.5
def returned_trucks (R : ℕ) := (percentage_returned * R).to_nat
def trucks_on_lot_sat (N R : ℕ) := N + returned_trucks R

-- Lean statement to prove that the maximum number of rented trucks is 20
theorem max_rented_trucks :
  ∀ (N R : ℕ), trucks_on_lot_sat N R >= 10 ∧ N + R <= total_trucks  → R ≤ total_trucks := 
by
  intros
  sorry

end max_rented_trucks_l618_618498


namespace expected_prize_money_l618_618285

theorem expected_prize_money :
  let a1 := 1 / 7
  let prob1 := a1
  let prob2 := 2 * a1
  let prob3 := 4 * a1
  let prize1 := 700
  let prize2 := 700 - 140
  let prize3 := 700 - 140 * 2
  let expected_money := prize1 * prob1 + prize2 * prob2 + prize3 * prob3
  expected_money = 500 := 
by
  -- Definitions
  let a1 := 1 / 7
  let prob1 := a1
  let prob2 := 2 * a1
  let prob3 := 4 * a1
  let prize1 := 700
  let prize2 := 700 - 140
  let prize3 := 700 - 140 * 2
  let expected_money := prize1 * prob1 + prize2 * prob2 + prize3 * prob3

  -- Calculate
  sorry -- Proof to show expected_money equals 500

end expected_prize_money_l618_618285


namespace who_made_mistake_l618_618018

-- Defining conditions for the colored pencils
def has_at_least_four_blue_pencils (b : Nat) : Prop := b >= 4
def has_at_least_five_green_pencils (g : Nat) : Prop := g >= 5
def has_at_least_three_blue_four_green_pencils (b g : Nat) : Prop := b >= 3 ∧ g >= 4
def has_at_least_four_blue_four_green_pencils (b g : Nat) : Prop := b >= 4 ∧ g >= 4

-- Statement of the problem
theorem who_made_mistake (b g : Nat) (vasya kolya petya misha : Prop) :
  has_at_least_four_blue_pencils b →
  has_at_least_five_green_pencils g →
  has_at_least_three_blue_four_green_pencils b g →
  has_at_least_four_blue_four_green_pencils b g →
  (∃ T : Set Prop, {vasya, kolya, petya, misha}.Erase T = {vasya, kolya, petya, misha} ∧
    T.Card = 3) →
  (kolya ↔ ¬ g >= 5) := 
sorry

end who_made_mistake_l618_618018


namespace remainder_of_sum_division_l618_618470

theorem remainder_of_sum_division 
  (a : ℕ → ℤ) 
  (S : ℤ)
  (h_a : ∀ n, a n = 6 * (n + 1) - 3)
  (h_S : S = ∑ n in finset.range 46, a n) :
  S % 8 = 2 := 
sorry

end remainder_of_sum_division_l618_618470


namespace part_I_part_II_l618_618616

noncomputable def f (x : ℝ) : ℝ := abs (x - 2) + abs (x + 1) + 2 * abs (x + 2)

theorem part_I : ∀ x : ℝ, f x ≥ 5 :=
sorry

theorem part_II : ∀ a : ℝ, (∀ x : ℝ, 15 - 2 * f x < a^2 + 9 / (a^2 + 1)) → ¬ (a = sqrt 2 ∨ a = - sqrt 2) :=
sorry

end part_I_part_II_l618_618616


namespace sufficiency_condition_l618_618204

-- Definitions of p and q
def p (a b : ℝ) : Prop := a > |b|
def q (a b : ℝ) : Prop := a^2 > b^2

-- Main theorem statement
theorem sufficiency_condition (a b : ℝ) : (p a b → q a b) ∧ (¬(q a b → p a b)) := 
by
  sorry

end sufficiency_condition_l618_618204


namespace largest_initial_number_l618_618319

theorem largest_initial_number :
  ∃ n : ℕ, 
    (∀ (a1 a2 a3 a4 a5 : ℕ),
      (n + a1).gcd a1 = 1 ∧
      (n + a1 + a2).gcd a2 = 1 ∧
      (n + a1 + a2 + a3).gcd a3 = 1 ∧
      (n + a1 + a2 + a3 + a4).gcd a4 = 1 ∧
      (n + a1 + a2 + a3 + a4 + a5).gcd a5 = 1 ∧
      n + a1 + a2 + a3 + a4 + a5 = 200) 
    → n = 189 :=
begin
  sorry
end

end largest_initial_number_l618_618319


namespace who_made_a_mistake_l618_618038

-- Definitions of the conditions
def at_least_four_blue_pencils (B : ℕ) : Prop := B ≥ 4
def at_least_five_green_pencils (G : ℕ) : Prop := G ≥ 5
def at_least_three_blue_and_four_green_pencils (B G : ℕ) : Prop := B ≥ 3 ∧ G ≥ 4
def at_least_four_blue_and_four_green_pencils (B G : ℕ) : Prop := B ≥ 4 ∧ G ≥ 4

-- The main theorem stating who made a mistake
theorem who_made_a_mistake (B G : ℕ) 
  (hv : at_least_four_blue_pencils B)
  (hk : at_least_five_green_pencils G)
  (hp : at_least_three_blue_and_four_green_pencils B G)
  (hm : at_least_four_blue_and_four_green_pencils B G) 
  (h_truth : (hv ∧ hk ∧ hp ∧ hm) ∨ (¬hv ∧ hk ∧ hp ∧ hm) ∨ (hv ∧ ¬hk ∧ hp ∧ hm) ∨ (hv ∧ hk ∧ ¬hp ∧ hm) ∨ (hv ∧ hk ∧ hp ∧ ¬hm))
  (h_truthful: ∑ b in [hv, hk, hp, hm], (if b then 1 else 0) = 3) : 
  hk = false := 
sorry

end who_made_a_mistake_l618_618038


namespace maximum_value_of_f_l618_618437

noncomputable def f (x : ℝ) : ℝ := x / (x - 1)

theorem maximum_value_of_f : ∀ x, (2 ≤ x) → f x ≤ 2 :=
begin
  -- We define the function f as given
  have h₁ : ∀ x, f'(x) = -1 / (x - 1) ^ 2 := sorry,
  -- We know that for x ≥ 2, f'(x) < 0
  have h₂ : ∀ x, (2 ≤ x) → -1 / (x - 1) ^ 2 < 0 := sorry,
  -- f(x) is a decreasing function on [2, +∞)
  have h₃ : ∀ x, (2 ≤ x) → f x ≤ f 2 := sorry,
  -- Therefore the maximum value is f(2) = 2
  show ∀ x, (2 ≤ x) → f x ≤ 2, by sorry
end

end maximum_value_of_f_l618_618437


namespace gcd_of_228_and_1995_l618_618633

theorem gcd_of_228_and_1995 : Nat.gcd 228 1995 = 57 :=
by
  sorry

end gcd_of_228_and_1995_l618_618633


namespace solve_real_y_two_distinct_l618_618849

noncomputable def quadratic_sol (a b c : ℤ) : List ℝ :=
[(b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a), (b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)]

theorem solve_real_y_two_distinct (y : ℝ) :
  ∃ (y1 y2 : ℝ), y1 ≠ y2 ∧ 
  (4 ^ (2 * y) + 256 = 34 * 4 ^ y → 
  4 ^ y = (34 + Real.sqrt (34 ^ 2 - 4 * 256)) / 2 ∨ 
  4 ^ y = (34 - Real.sqrt (34 ^ 2 - 4 * 256)) / 2) := 
sorry

end solve_real_y_two_distinct_l618_618849


namespace part_a_exists_point_X_part_b_exists_point_X_l618_618662

namespace GeometryConstructions

/-- Part (a) -/
theorem part_a_exists_point_X (S : Set Point)
  (A B C D : Point) (a : ℝ)
  (h_circle : is_circle S)
  (h_chords : is_chord S A B ∧ is_chord S C D):
  ∃ X : Point, on_circle X S ∧ 
  (segment_intersection_length (line_through A X) (line_through B X) (chord CD) = a) :=
sorry

/-- Part (b) -/
theorem part_b_exists_point_X (S : Set Point)
  (A B C D E : Point)
  (h_circle : is_circle S)
  (h_chords : is_chord S A B ∧ is_chord S C D)
  (h_midpoint : is_midpoint E (chord CD)):
  ∃ X : Point, on_circle X S ∧ 
  (segment_bisected_at (line_through A X) (line_through B X) (chord CD) E) :=
sorry

end GeometryConstructions

end part_a_exists_point_X_part_b_exists_point_X_l618_618662


namespace Zs_share_in_profit_l618_618072

theorem Zs_share_in_profit (x_investment : ℕ) (y_investment : ℕ) (z_investment : ℕ) (z_months : ℕ) (total_profit : ℕ) :
  x_investment = 36000 → y_investment = 42000 → z_investment = 48000 → z_months = 8 → total_profit = 13970 →
  let x_investment_months := x_investment * 12,
      y_investment_months := y_investment * 12,
      z_investment_months := z_investment * z_months,
      total_investment_months := x_investment_months + y_investment_months + z_investment_months,
      z_ratio := z_investment_months.to_rat / total_investment_months.to_rat,
      z_share := z_ratio * total_profit in
  z_share.to_int = 4065 :=
by intros; simp; sorry

end Zs_share_in_profit_l618_618072


namespace problem_statement_l618_618669

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

variable (f g : ℝ → ℝ)

axiom f_odd : odd_function f
axiom f_neg : ∀ x : ℝ, x < 0 → f x = x^3 - 1
axiom f_pos : ∀ x : ℝ, x > 0 → f x = g x

theorem problem_statement : f (-1) + g 2 = 7 :=
by
  sorry

end problem_statement_l618_618669


namespace coeff_sum_zero_l618_618657

theorem coeff_sum_zero (a₀ a₁ a₂ a₃ a₄ : ℝ) (h : ∀ x : ℝ, (2*x + 1)^4 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4) :
  a₁ + a₂ + a₃ + a₄ = 0 :=
by
  sorry

end coeff_sum_zero_l618_618657


namespace perpendicular_lines_l618_618642

theorem perpendicular_lines (b : ℝ) :
  (∀ x y : ℝ, 4 * y + 2 * x - 6 = 0 → 5 * y + b * x - 15 = 0 → (1/2) * (b/5) = -1) →
  b = -10 :=
by
  intro h
  specialize h 0 1 (by norm_num) (by norm_num)
  simpa using h

end perpendicular_lines_l618_618642


namespace tory_sold_each_toy_gun_for_l618_618558

theorem tory_sold_each_toy_gun_for :
  ∃ (x : ℤ), 8 * 18 = 7 * x + 4 ∧ x = 20 := 
by
  use 20
  constructor
  · sorry
  · sorry

end tory_sold_each_toy_gun_for_l618_618558


namespace batsman_average_is_18_l618_618943
noncomputable def average_after_18_innings (score_18th: ℕ) (average_17th: ℕ) (innings: ℕ) : ℕ :=
  let total_runs_17 := average_17th * 17
  let total_runs_18 := total_runs_17 + score_18th
  total_runs_18 / innings

theorem batsman_average_is_18 {score_18th: ℕ} {average_17th: ℕ} {expected_average: ℕ} :
  score_18th = 1 → average_17th = 19 → expected_average = 18 →
  average_after_18_innings score_18th average_17th 18 = expected_average := by
  sorry

end batsman_average_is_18_l618_618943


namespace no_solution_A_and_B_l618_618918

theorem no_solution_A_and_B :
  (¬ ∃ x : ℝ, (x - 5) ^ 2 = -1) ∧ (¬ ∃ x : ℝ, | -2 * x | + 7 = 0) :=
by
  sorry

end no_solution_A_and_B_l618_618918


namespace sin_addition_l618_618416

theorem sin_addition (x y : ℝ) : sin (x - y) * cos y + cos (x - y) * sin y = sin x :=
by
  sorry

end sin_addition_l618_618416


namespace D_times_C_eq_l618_618778

-- Define the matrices C and D
variable (C D : Matrix (Fin 2) (Fin 2) ℚ)

-- Add the conditions
axiom h1 : C * D = C + D
axiom h2 : C * D = ![![15/2, 9/2], ![-6/2, 12/2]]

-- Define the goal
theorem D_times_C_eq : D * C = ![![15/2, 9/2], ![-6/2, 12/2]] :=
sorry

end D_times_C_eq_l618_618778


namespace tree_difference_l618_618130

-- Given constants
def Hassans_apple_trees : Nat := 1
def Hassans_orange_trees : Nat := 2

def Ahmeds_orange_trees : Nat := 8
def Ahmeds_apple_trees : Nat := 4 * Hassans_apple_trees

-- Total trees computations
def Ahmeds_total_trees : Nat := Ahmeds_apple_trees + Ahmeds_orange_trees
def Hassans_total_trees : Nat := Hassans_apple_trees + Hassans_orange_trees

-- Theorem to prove the difference in total trees
theorem tree_difference : Ahmeds_total_trees - Hassans_total_trees = 9 := by
  sorry

end tree_difference_l618_618130


namespace largest_initial_number_l618_618334

-- Let's define the conditions and the result
def valid_addition (n a : ℕ) : Prop := ∃ k : ℕ, n = a * k + r ∧ 0 < r ∧ r < a

def valid_operations (initial : ℕ) (final : ℕ) (steps : ℕ → ℕ → ℕ) : Prop :=
  ∃ (a b c d e : ℕ), valid_addition initial a ∧
                      valid_addition (initial + a) b ∧
                      valid_addition (initial + a + b) c ∧
                      valid_addition (initial + a + b + c) d ∧
                      valid_addition (initial + a + b + c + d) e ∧
                      initial + a + b + c + d + e = final

theorem largest_initial_number :
  ∃ n : ℕ, (valid_operations n 200 (λn a, n + a)) ∧ (∀ m : ℕ, valid_operations m 200 (λn a, n + a) → m ≤ n) :=
sorry

end largest_initial_number_l618_618334


namespace sum_y_values_l618_618044

theorem sum_y_values 
  (x1 x2 x3 x4 x5 : ℝ) 
  (y1 y2 y3 y4 y5 : ℝ)
  (h1 : x1 + x2 + x3 + x4 + x5 = 150)
  (h2 : ∀ (x : ℝ), (1 / 5) * (x1 + x2 + x3 + x4 + x5) = x → ∑ i in [y1, y2, y3, y4, y5], i = 5 * (0.67 * x + 24.9)) :
  y1 + y2 + y3 + y4 + y5 = 225 :=
sorry

end sum_y_values_l618_618044


namespace largest_subset_size_l618_618967

def largest_subset_with_property (s : Set ℤ) : Prop :=
  ∀ (a b : ℤ), a ∈ s → b ∈ s → (a ≠ 4 * b ∧ b ≠ 4 * a)

theorem largest_subset_size : ∃ s : Set ℤ, (∀ x : ℤ, x ∈ s → 1 ≤ x ∧ x ≤ 150) ∧ largest_subset_with_property s ∧ s.card = 120 :=
by
  sorry

end largest_subset_size_l618_618967


namespace f_is_even_f_symmetric_about_pi_f_l618_618843

noncomputable theory

def f (x : ℝ) : ℝ := cos x + (cos (2 * x) / 2) + (cos (4 * x) / 4)

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

theorem f_symmetric_about_pi : ∀ x : ℝ, f (2 * π - x) = f x := by
  sorry

theorem f'_less_than_3 : ∀ x : ℝ, (derivative f x) < 3 := by
  sorry

end f_is_even_f_symmetric_about_pi_f_l618_618843


namespace green_marbles_after_replacement_l618_618525

def total_marbles (white_marbles : ℕ) (white_percentage : ℚ) : ℚ :=
  white_marbles / white_percentage

/-- Given the percentage of white marbles and the total number of 6 colors of marbles, calculate the total marbles,
the initial counts of red, green, and the new count of green marbles after replacement. -/
theorem green_marbles_after_replacement:
  ∀ (white_marbles : ℕ) (white_percentage : ℚ) (percent_red percent_green percent_blue percent_yellow percent_purple : ℚ),
  white_percentage = 0.15 →
  white_marbles = 35 →
  percent_red = 0.25 →
  percent_green = 0.15 →
  percent_blue = 0.20 →
  percent_yellow = 0.10 →
  percent_purple = 0.15 →
  let total := total_marbles white_marbles white_percentage in
  let red_marbles := percent_red * total in
  let green_marbles := percent_green * total in
  let replaced_green_marbles := green_marbles + (red_marbles / 3) in
  replaced_green_marbles ≈ 55 :=
begin
  intros,
  sorry
end

end green_marbles_after_replacement_l618_618525


namespace optimal_use_period_l618_618734

noncomputable def daily_maintenance_cost (n : ℕ) : ℝ :=
  if n = 0 then 0 else n / 2 + 99.5

noncomputable def average_daily_cost (n : ℕ) : ℝ :=
  if n = 0 then 0 else
  (90000 + (100 + n / 2 + 99.5) * n / 2) / n

theorem optimal_use_period :
  (average_daily_cost 600 = 399.75) :=
by
  have : average_daily_cost 600 = (90000 + (100 + 600 / 2 + 99.5) * 600 / 2) / 600,
  { sorry },
  have : (90000 + (100 + 600 / 2 + 99.5) * 600 / 2) / 600 = 399.75,
  { sorry },
  exact Eq.trans this rfl

end optimal_use_period_l618_618734


namespace problem1_problem2_problem3_l618_618695

section
variables (b c : ℝ)

-- Function definition
def f (x : ℝ) : ℝ := (1/3) * x ^ 3 - b * x + c

-- 1. Prove that if the tangent line at (1, f(1)) is y = 2x + 1, then b = -1 and c = 5/3
theorem problem1 (h_tangent : ∀ x, f 1 = 2 * x + 1 ∧ deriv (f x) = 2) : 
  b = -1 ∧ c = 5/3 :=
sorry

-- 2. Prove that if b = 1, the function has a unique zero in (0, 2) if and only if c = 2/3 or -2/3 < c ≤ 0
theorem problem2 (unique_zero : ∀ x ∈ (0, 2), f x = 0) : 
  b = 1 ↔ c = 2/3 ∨ (-2/3 < c ∧ c ≤ 0) :=
sorry

-- 3. Prove that if |f(x₁) - f(x₂)| ≤ 4/3 for any x₁, x₂ ∈ [-1,1], then -1/3 ≤ b ≤ 1
theorem problem3 (bounded : ∀ (x₁ x₂ ∈ [-1,1]), |f x₁ - f x₂| ≤ 4/3) : 
  -1/3 ≤ b ∧ b ≤ 1 :=
sorry

end

end problem1_problem2_problem3_l618_618695


namespace disjoint_intervals_l618_618100

theorem disjoint_intervals (intervals : list (set ℝ)) (h : set.Icc 0 50 ⊆ ⋃ I ∈ intervals, I) (hl : ∀ I ∈ intervals, ∃ a b : ℝ, I = set.Icc a b ∧ b - a = 1) :
  ∃ subset : list (set ℝ), (∀ I J ∈ subset, I ≠ J → I ∩ J = ∅) ∧ subset.length ≥ 25 :=
by sorry

end disjoint_intervals_l618_618100


namespace kolya_is_wrong_l618_618027

def pencils_problem_statement (at_least_four_blue : Prop) 
                              (at_least_five_green : Prop) 
                              (at_least_three_blue_and_four_green : Prop) 
                              (at_least_four_blue_and_four_green : Prop) : 
                              Prop :=
  ∃ (B G : ℕ), -- B represents the number of blue pencils, G represents the number of green pencils
    ((B ≥ 4) ∧ (G ≥ 4)) ∧ -- Vasya's statement (at least 4 blue), Petya's and Misha's combined statement (at least 4 green)
    at_least_four_blue ∧ -- Vasya's statement (there are at least 4 blue pencils)
    (at_least_five_green ↔ G ≥ 5) ∧ -- Kolya's statement (there are at least 5 green pencils)
    at_least_three_blue_and_four_green ∧ -- Petya's statement (at least 3 blue and 4 green)
    at_least_four_blue_and_four_green -- Misha's statement (at least 4 blue and 4 green)

theorem kolya_is_wrong (at_least_four_blue : Prop) 
                        (at_least_five_green : Prop) 
                        (at_least_three_blue_and_four_green : Prop) 
                        (at_least_four_blue_and_four_green : Prop) : 
                        pencils_problem_statement at_least_four_blue 
                                                  at_least_five_green 
                                                  at_least_three_blue_and_four_green 
                                                  at_least_four_blue_and_four_green :=
sorry

end kolya_is_wrong_l618_618027


namespace compute_h_at_3_l618_618389

def f (x : ℝ) : ℝ := 3 * x + 4
def g (x : ℝ) : ℝ := Real.sqrt (f x) - 3
def h (x : ℝ) : ℝ := f (g x)

theorem compute_h_at_3 : h 3 = 3 * Real.sqrt 13 - 5 := by
  sorry

end compute_h_at_3_l618_618389


namespace decimal_arithmetic_l618_618126

theorem decimal_arithmetic : 0.45 - 0.03 + 0.008 = 0.428 := by
  sorry

end decimal_arithmetic_l618_618126


namespace trains_meet_at_noon_l618_618900

noncomputable def meeting_time_of_trains : Prop :=
  let distance_between_stations := 200
  let speed_of_train_A := 20
  let starting_time_A := 7
  let speed_of_train_B := 25
  let starting_time_B := 8
  let initial_distance_covered_by_A := speed_of_train_A * (starting_time_B - starting_time_A)
  let remaining_distance := distance_between_stations - initial_distance_covered_by_A
  let relative_speed := speed_of_train_A + speed_of_train_B
  let time_to_meet_after_B_starts := remaining_distance / relative_speed
  let meeting_time := starting_time_B + time_to_meet_after_B_starts
  meeting_time = 12

theorem trains_meet_at_noon : meeting_time_of_trains :=
by
  sorry

end trains_meet_at_noon_l618_618900


namespace exam_cutoff_mark_l618_618740

theorem exam_cutoff_mark
  (num_students : ℕ)
  (absent_percentage : ℝ)
  (fail_percentage : ℝ)
  (fail_mark_diff : ℝ)
  (just_pass_percentage : ℝ)
  (remaining_avg_mark : ℝ)
  (class_avg_mark : ℝ)
  (absent_students : ℕ)
  (fail_students : ℕ)
  (just_pass_students : ℕ)
  (remaining_students : ℕ)
  (total_marks : ℝ)
  (P : ℝ) :
  absent_percentage = 0.2 →
  fail_percentage = 0.3 →
  fail_mark_diff = 20 →
  just_pass_percentage = 0.1 →
  remaining_avg_mark = 65 →
  class_avg_mark = 36 →
  absent_students = (num_students * absent_percentage) →
  fail_students = (num_students * fail_percentage) →
  just_pass_students = (num_students * just_pass_percentage) →
  remaining_students = num_students - absent_students - fail_students - just_pass_students →
  total_marks = (absent_students * 0) + (fail_students * (P - fail_mark_diff)) + (just_pass_students * P) + (remaining_students * remaining_avg_mark) →
  class_avg_mark = total_marks / num_students →
  P = 40 :=
by
  intros
  sorry

end exam_cutoff_mark_l618_618740


namespace Petya_can_ask_one_question_l618_618405

structure FootballMatch where
  goals_shinnik : ℕ
  goals_dynamo : ℕ

def total_goals (match : FootballMatch) : ℕ :=
  match.goals_shinnik + match.goals_dynamo

axiom Roma_statement (match : FootballMatch) :
  ∀ m : FootballMatch, total_goals m = total_goals match → m = match

axiom Oleg_statement (match : FootballMatch) :
  ∀ m : FootballMatch, (total_goals m = total_goals match + 1 ∧
                         m.goals_shinnik > match.goals_shinnik) ∨
                        (total_goals m = total_goals match + 1 ∧
                         m.goals_dynamo > match.goals_dynamo) → m = match

axiom Seryozha_statement (match : FootballMatch) :
  match.goals_shinnik > 0

theorem Petya_can_ask_one_question (match : FootballMatch) :
  ∃ p : FootballMatch, (p.goals_shinnik = 1 ∧ p.goals_dynamo = 0) ∨
                       (p.goals_shinnik = 2 ∧ p.goals_dynamo = 0) ∨
                       (p.goals_shinnik = 1 ∧ p.goals_dynamo = 1) := by
  sorry

end Petya_can_ask_one_question_l618_618405


namespace pq_through_fixed_point_l618_618137

variables {A B C X P Q : Type} [circle : Circle ABC] [circumcircle : Circle ABC] [circle_diff : ¬(P = X ∨ P = B ∨ Q = X ∨ Q = B)]
variables [hPQ : on_circle P X B] [hPQ' : on_circle Q X B] [linePQ : Line P Q] [linePQ_circumcircle_intersect : ∃ S, on_circle S ABC]

theorem pq_through_fixed_point :
  ∃ S, ∀ P Q, on_circle P X B → on_circle Q X B → ¬(P = X ∨ P = B ∨ Q = X ∨ Q = B) → Line_through P Q S := 
sorry

end pq_through_fixed_point_l618_618137


namespace right_triangle_with_25_points_l618_618742

/-- Defined structure for a right triangle with given conditions -/
structure RightTriangle :=
  (hypotenuse : ℝ)
  (angle_30 : Prop)
  (angle_90 : Prop := true)

noncomputable def exists_circle_covering_three_points 
  (T : RightTriangle) 
  (points : Finset ℝ × ℝ)
  (points_count : points.card = 25)
  (diameter : ℝ)
  : Prop :=
  ∃ circle_center circle_radius,
    circle_radius = diameter / 2 ∧
    ∃ p₁ p₂ p₃ ∈ points, 
      dist circle_center p₁ ≤ circle_radius ∧
      dist circle_center p₂ ≤ circle_radius ∧
      dist circle_center p₃ ≤ circle_radius

/-- Main theorem: Proving the existence of a circle of given diameter covering 3 points -/
theorem right_triangle_with_25_points 
  : ∀ (T : RightTriangle) (points : Finset (ℝ × ℝ)),
    (T.hypotenuse = 1) →
    (T.angle_30 = (π / 6)) →
    (points.card = 25) →
    exists_circle_covering_three_points T points (5/17) :=
by sorry

end right_triangle_with_25_points_l618_618742


namespace integer_part_ratio_l618_618743

-- Define the chessboard and the circle conditions
def side_length : ℝ := 8
def radius : ℝ := 4
def π := Real.pi

-- Define the areas 
def quarter_circle_area : ℝ := (1/4) * π * radius^2
def square_area : ℝ := 1
def number_of_quadrants : ℝ := 4
def S₁' : ℝ := quarter_circle_area
def S₂' : ℝ := 4 * square_area - quarter_circle_area

def S₁ : ℝ := number_of_quadrants * S₁'
def S₂ : ℝ := number_of_quadrants * S₂'

-- Define the problem of finding the integer part of S₁ / S₂
def target_ratio : ℝ := S₁ / S₂

theorem integer_part_ratio : (⌊ target_ratio ⌋ : ℝ) = 1 := sorry

end integer_part_ratio_l618_618743


namespace who_made_a_mistake_l618_618035

-- Definitions of the conditions
def at_least_four_blue_pencils (B : ℕ) : Prop := B ≥ 4
def at_least_five_green_pencils (G : ℕ) : Prop := G ≥ 5
def at_least_three_blue_and_four_green_pencils (B G : ℕ) : Prop := B ≥ 3 ∧ G ≥ 4
def at_least_four_blue_and_four_green_pencils (B G : ℕ) : Prop := B ≥ 4 ∧ G ≥ 4

-- The main theorem stating who made a mistake
theorem who_made_a_mistake (B G : ℕ) 
  (hv : at_least_four_blue_pencils B)
  (hk : at_least_five_green_pencils G)
  (hp : at_least_three_blue_and_four_green_pencils B G)
  (hm : at_least_four_blue_and_four_green_pencils B G) 
  (h_truth : (hv ∧ hk ∧ hp ∧ hm) ∨ (¬hv ∧ hk ∧ hp ∧ hm) ∨ (hv ∧ ¬hk ∧ hp ∧ hm) ∨ (hv ∧ hk ∧ ¬hp ∧ hm) ∨ (hv ∧ hk ∧ hp ∧ ¬hm))
  (h_truthful: ∑ b in [hv, hk, hp, hm], (if b then 1 else 0) = 3) : 
  hk = false := 
sorry

end who_made_a_mistake_l618_618035


namespace speed_of_man_in_still_water_l618_618073

theorem speed_of_man_in_still_water (v_m v_s : ℝ) (h1 : 5 * (v_m + v_s) = 45) (h2 : 5 * (v_m - v_s) = 25) : v_m = 7 :=
by
  sorry

end speed_of_man_in_still_water_l618_618073


namespace alpha_centauri_puzzle_max_numbers_l618_618935

theorem alpha_centauri_puzzle_max_numbers (A B N : ℕ) (h1 : A = 1353) (h2 : B = 2134) (h3 : N = 11) : 
  ∃ S : set ℕ, S ⊆ {n | A ≤ n ∧ n ≤ B} ∧ (∀ x y ∈ S, x ≠ y → (x + y) % N ≠ 0) ∧ S.card = 356 :=
sorry

end alpha_centauri_puzzle_max_numbers_l618_618935


namespace projection_of_dodecahedron_is_pentagon_l618_618412

-- Define the dodecahedron structure and its properties
structure Dodecahedron :=
  (center : Point)
  (edges : List Edge)
  (faces : List Face)
  -- Additional properties can be added as necessary

-- Define the projection function
def projection (d : Dodecahedron) (plane : Plane) : Shape :=
  sorry -- Detailed implementation of the projection is not required here

-- Conditions for the problem
def is_valid_projection_plane (d : Dodecahedron) (plane : Plane) (line : Line) : Prop :=
  line.passes_through d.center ∧ line.passes_through (midpoint of some edge of d) ∧ plane.is_perpendicular_to line

-- Main theorem to prove
theorem projection_of_dodecahedron_is_pentagon (d : Dodecahedron) (line : Line) (plane : Plane) :
  is_valid_projection_plane d plane line →
  projection d plane = Shape.pentagon :=
by
  sorry -- Proof is not required

end

end projection_of_dodecahedron_is_pentagon_l618_618412


namespace sum_of_squares_of_roots_l618_618152

theorem sum_of_squares_of_roots 
  (r s t : ℝ) 
  (hr : y^3 - 8 * y^2 + 9 * y - 2 = 0) 
  (hs : y ≥ 0) 
  (ht : y ≥ 0):
  r^2 + s^2 + t^2 = 46 :=
sorry

end sum_of_squares_of_roots_l618_618152


namespace correct_proportions_l618_618099

variable (initial_sauce_total : ℝ)
variable (initial_chocolate_percentage : ℝ)
variable (initial_raspberry_percentage : ℝ)
variable (initial_cream_percentage : ℝ)
variable (target_chocolate_percentage : ℝ)
variable (target_raspberry_percentage : ℝ)
variable (target_cream_percentage : ℝ)
variable (replacement_ratio_chocolate : ℝ)
variable (replacement_ratio_cream : ℝ)

def remove_replace (x : ℝ) : ℝ → ℝ → ℝ → ℝ :=
  λ choco raspberry cream =>
    let changed_choco := (initial_chocolate_percentage / 100 * initial_sauce_total - initial_chocolate_percentage / 100 * x + replacement_ratio_chocolate / (replacement_ratio_chocolate + replacement_ratio_cream) * x)
    let changed_raspberry := (initial_raspberry_percentage / 100 * initial_sauce_total - initial_raspberry_percentage / 100 * x)
    let changed_cream := (initial_cream_percentage / 100 * initial_sauce_total - initial_cream_percentage / 100 * x + replacement_ratio_cream / (replacement_ratio_chocolate + replacement_ratio_cream) * x)
    changed_choco = target_chocolate_percentage / 100 * initial_sauce_total ∧
    changed_raspberry = target_raspberry_percentage / 100 * initial_sauce_total ∧
    changed_cream = target_cream_percentage / 100 * initial_sauce_total

theorem correct_proportions : ∃ x, remove_replace x 15 0.30 0.60 0.10 0.40 0.40 0.20 2 1 :=
    sorry

end correct_proportions_l618_618099


namespace mixture_ratios_l618_618757

theorem mixture_ratios
  (p q x y m n : ℕ)
  (hp : 5 * x + m * y)
  (hq : 3 * x + n * y)
  (hfinal : 5 * x + m * y = 3 * x + n * y)
  (h_ratio : p / q = 2) : 
  p / q = 2 :=
by
  sorry

end mixture_ratios_l618_618757


namespace largest_gold_coins_l618_618490

theorem largest_gold_coins (k : ℤ) (h1 : 13 * k + 3 < 100) : 91 ≤ 13 * k + 3 :=
by
  sorry

end largest_gold_coins_l618_618490


namespace BigJoe_is_8_feet_l618_618144

variable (Pepe_height : ℝ) (h1 : Pepe_height = 4.5)
variable (Frank_height : ℝ) (h2 : Frank_height = Pepe_height + 0.5)
variable (Larry_height : ℝ) (h3 : Larry_height = Frank_height + 1)
variable (Ben_height : ℝ) (h4 : Ben_height = Larry_height + 1)
variable (BigJoe_height : ℝ) (h5 : BigJoe_height = Ben_height + 1)

theorem BigJoe_is_8_feet : BigJoe_height = 8 := by
  sorry

end BigJoe_is_8_feet_l618_618144


namespace largest_initial_number_l618_618346

theorem largest_initial_number :
  ∃ n : ℕ, (∀ k : ℕ, (n % k ≠ 0 → k ∈ {2, 2, 2, 2, 3}) ∧ (n + 11 = 200)) ∧ (n = 189) :=
begin
  sorry -- Proof not required per instruction
end

end largest_initial_number_l618_618346


namespace particle_position_after_120_moves_l618_618953

noncomputable def cis (θ : ℝ) : ℂ := complex.of_real (Real.cos θ) + complex.I * (complex.of_real (Real.sin θ))

theorem particle_position_after_120_moves :
  let ω := cis (Real.pi / 3),
      move := λ z: ℂ, ω * z + 8,
      initial_pos : ℂ := 6 in
  (move^[120] initial_pos).re = 6 ∧ (move^[120] initial_pos).im = 0 :=
by
  let ω := cis (Real.pi / 3);
  let move := λ z: ℂ, ω * z + 8;
  let initial_pos : ℂ := 6;
  have h_final : (move^[120] initial_pos) = 6 := sorry;
  exact ⟨by rw [← h_final, complex.re, add_zero 6], by rw [← h_final, complex.im, zero I]⟩

end particle_position_after_120_moves_l618_618953


namespace tree_difference_l618_618129

-- Given constants
def Hassans_apple_trees : Nat := 1
def Hassans_orange_trees : Nat := 2

def Ahmeds_orange_trees : Nat := 8
def Ahmeds_apple_trees : Nat := 4 * Hassans_apple_trees

-- Total trees computations
def Ahmeds_total_trees : Nat := Ahmeds_apple_trees + Ahmeds_orange_trees
def Hassans_total_trees : Nat := Hassans_apple_trees + Hassans_orange_trees

-- Theorem to prove the difference in total trees
theorem tree_difference : Ahmeds_total_trees - Hassans_total_trees = 9 := by
  sorry

end tree_difference_l618_618129


namespace train_departure_times_l618_618540

theorem train_departure_times
  (average_speed : ℝ)
  (distance_traveled : ℝ)
  (overlap_times : set ℝ)
  (h_speed : average_speed = 33)
  (h_distance : distance_traveled = 8)
  (h_overlap : overlap_times = {t | ∃ k : ℤ, t = k * (360 / 5.5)}) :
  let time_to_travel := distance_traveled / average_speed in
  time_to_travel = (480 / 33) / 60 →
  (11 + 45/11 ≤ time_to_travel ∧ time_to_travel ≤ 22 + 40 / 11)
  ∨ (10 + 40/60 = time_to_travel) 
  ∨ (22 + 40/60 = time_to_travel) :=
by {
  sorry
}

end train_departure_times_l618_618540


namespace nine_point_circle_center_hyperbola_l618_618411

theorem nine_point_circle_center_hyperbola
    (A B C O : Point)
    (H : rectangular_hyperbola_contains A B C O):
  nine_point_circle_contains (nine_point_circle A B C) O :=
sorry

end nine_point_circle_center_hyperbola_l618_618411


namespace find_sum_l618_618791

variable {a b : ℝ}
variable {h1 : a ≠ b}
variable {h2 : det ![![2, 3, 5], ![4, a, b], ![4, b, a]] = 0}

theorem find_sum (a b : ℝ) (h1 : a ≠ b) (h2 : det ![![2, 3, 5], ![4, a, b], ![4, b, a]] = 0) : a + b = 32 := 
sorry

end find_sum_l618_618791


namespace Kolya_mistake_l618_618025

def boys := ["Vasya", "Kolya", "Petya", "Misha"]

constant num_blue_pencils : ℕ
constant num_green_pencils : ℕ

axiom Vasya_statement : num_blue_pencils >= 4
axiom Kolya_statement : num_green_pencils >= 5
axiom Petya_statement : num_blue_pencils >= 3 ∧ num_green_pencils >= 4
axiom Misha_statement : num_blue_pencils >= 4 ∧ num_green_pencils >= 4

axiom three_truths_one_mistake : 
  (Vasya_statement ∨ ¬Vasya_statement) ∧
  (Kolya_statement ∨ ¬Kolya_statement) ∧
  (Petya_statement ∨ ¬Petya_statement) ∧
  (Misha_statement ∨ ¬Misha_statement) ∧
  ((Vasya_statement ? true : 1) + 
   (Kolya_statement ? true : 1) + 
   (Petya_statement ? true : 1) +
   (Misha_statement ? true : 1) == 3)

theorem Kolya_mistake : ¬Kolya_statement :=
by
  sorry

end Kolya_mistake_l618_618025


namespace alpha_centauri_puzzle_max_numbers_l618_618934

theorem alpha_centauri_puzzle_max_numbers (A B N : ℕ) (h1 : A = 1353) (h2 : B = 2134) (h3 : N = 11) : 
  ∃ S : set ℕ, S ⊆ {n | A ≤ n ∧ n ≤ B} ∧ (∀ x y ∈ S, x ≠ y → (x + y) % N ≠ 0) ∧ S.card = 356 :=
sorry

end alpha_centauri_puzzle_max_numbers_l618_618934


namespace shop_sold_for_270_l618_618096

variable (C : ℝ)
variable (buy_back_price : ℝ := 0.60 * C)
variable (difference : ℝ := C - buy_back_price)

theorem shop_sold_for_270 (h1 : difference = 100) : let resell_price := buy_back_price + 0.80 * buy_back_price in resell_price = 270 :=
by
  assume resell_price
  sorry

end shop_sold_for_270_l618_618096


namespace reservoir_percentage_before_storm_l618_618076

/-- Given conditions:
1. After a storm deposits 115 billion gallons into the reservoir, the reservoir is 80% full.
2. The original contents of the reservoir were 245 billion gallons.
Prove that the reservoir was approximately 54.44% full before the storm. -/
theorem reservoir_percentage_before_storm :
  let C := (360 / 0.80 : ℝ) in 
  (245 / C) * 100 ≈ 54.44 :=
by
  sorry

end reservoir_percentage_before_storm_l618_618076


namespace twin_functions_count_l618_618724

theorem twin_functions_count : 
  (∀ (f : ℝ → ℝ), (∀ x, f x = 2 * x^2 + 1) ∧ (∀ y, y = 5 ∨ y = 19 → ∃ x, f x = y)) →
  {f : ℝ → ℝ // (∀ x, f x = 2 * x^2 + 1) ∧ (∀ y, y = 5 ∨ y = 19 → ∃ x, f x = y)}.card = 9 :=
begin
  sorry
end

end twin_functions_count_l618_618724


namespace line_l2_fixed_point_l618_618674

noncomputable def length_OQ {n : ℝ} (h1 : n = 2) : ℝ :=
  let P := (2, 2) in
  let Q := (-2, 0) in
  Real.dist (0, 0) Q

theorem line_l2_fixed_point {n : ℝ} (h1 : n = 2) (b : ℝ) (m : ℝ) (h2 : b ≠ -2) 
  (P := (2, 2)) (Q := (-2, 0)) (l1_eq : ∀ (E : ℝ × ℝ), E.1 = -2)
  (l2_eq : ∀ (A B : ℝ × ℝ), A.1 = m * A.2 + b ∧ B.1 = m * B.2 + b)
  (slopes_arithmetic_seq : ∀ (A B E : ℝ × ℝ), 
    let PA_slope := (A.2 - P.2) / (A.1 - P.1) in
    let PB_slope := (B.2 - P.2) / (B.1 - P.1) in
    let PE_slope := (E.2 - P.2) / (E.1 - P.1) in
    PA_slope + PB_slope = 2 * PE_slope) : 
  ∃ (k : ℝ), ∀ (y : ℝ), y * 0 + 2 = 2 :=
by sorry

end line_l2_fixed_point_l618_618674


namespace sum_of_fourth_powers_lt_150_l618_618473

theorem sum_of_fourth_powers_lt_150 : (∑ n in finset.range 4, n^4) = 98 := by
  sorry

end sum_of_fourth_powers_lt_150_l618_618473


namespace intersection_S_T_eq_S_l618_618805

def S := { y : ℝ | ∃ x : ℝ, y = 3^x }
def T := { y : ℝ | ∃ x : ℝ, x > 0 ∧ y = log x / log 3 }

theorem intersection_S_T_eq_S : S ∩ T = S :=
by
  sorry

end intersection_S_T_eq_S_l618_618805


namespace analytical_expression_l618_618246

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem analytical_expression (ω : ℝ) (φ : ℝ) :
  (∀ x : ℝ, f (x) ω φ = f (x + π)) ∧ (∀ x : ℝ, f (x) (2*π/3) φ = f ((2*π/3) - x)) →
  (ω = 2 ∧ φ = -π/6 ∨ ω = -2 ∧ φ = π/6) :=
sorry

end analytical_expression_l618_618246


namespace largest_median_of_ten_numbers_l618_618591

-- Define the list of ten positive integers where six elements are known
def known_list : List ℕ := [3, 5, 1, 4, 9, 6]

-- Define the condition that we have a sorted list with six known integers and four additional integers
def largest_possible_median : ℕ := 75 / 10

-- The theorem we need to prove
theorem largest_median_of_ten_numbers (L : List ℕ) (h_len : L.length = 10)
  (h_contains : ∀ x ∈ [3, 5, 1, 4, 9, 6], x ∈ L) : 
  median (sort L) = largest_possible_median := by
  sorry

end largest_median_of_ten_numbers_l618_618591


namespace reflect_across_x_axis_l618_618425

theorem reflect_across_x_axis (M : ℝ × ℝ) (h : M = (1, 2)) :
  let M_reflected := (M.1, -M.2) in M_reflected = (1, -2) :=
by
  subst h  -- Replace M with (1, 2)
  simp    -- Simplify the reflected coordinates
  sorry   -- Placeholder for proof

end reflect_across_x_axis_l618_618425


namespace f_is_even_l618_618785

-- Given an odd function g
def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g (x)

-- Define the function f as given by the problem
def f (g : ℝ → ℝ) (x : ℝ) : ℝ :=
  abs (g (x^2))

-- The theorem stating that f is an even function
theorem f_is_even (g : ℝ → ℝ) (h_odd : is_odd_function g) : ∀ x, f g x = f g (-x) :=
by
  sorry

end f_is_even_l618_618785


namespace problem_a_problem_b_problem_c_problem_d_l618_618842

noncomputable def f (x : ℝ) : ℝ := cos x + (cos (2 * x)) / 2 + (cos (4 * x)) / 4

theorem problem_a : (∀ x : ℝ, f (-x) = f x) :=
sorry

theorem problem_b : (∀ x : ℝ, f (2 * π - x) = f x) :=
sorry

theorem problem_c : ¬ (∀ x : ℝ, f (x + π) = f x) :=
sorry

theorem problem_d : (∀ x : ℝ, f' x < 3) :=
sorry

end problem_a_problem_b_problem_c_problem_d_l618_618842


namespace range_of_m_for_quadratic_sol_in_interval_l618_618638

theorem range_of_m_for_quadratic_sol_in_interval :
  {m : ℝ // ∀ x, (x^2 + (m-1)*x + 1 = 0) → (0 ≤ x ∧ x ≤ 2)} = {m : ℝ // m < -1} :=
by
  sorry

end range_of_m_for_quadratic_sol_in_interval_l618_618638


namespace Gage_total_cubes_l618_618259

-- Define the conditions
def Grady.red_cubes : ℕ := 20
def Grady.blue_cubes : ℕ := 15
def Grady.red_to_Gage_ratio : ℚ := 2 / 5
def Grady.blue_to_Gage_ratio : ℚ := 1 / 3
def Gage.initial_red_cubes : ℕ := 10
def Gage.initial_blue_cubes : ℕ := 12

-- Define the theorem to be proved
theorem Gage_total_cubes : 
  let received_red_cubes := Grady.red_to_Gage_ratio * Grady.red_cubes,
      received_blue_cubes := Grady.blue_to_Gage_ratio * Grady.blue_cubes,
      total_red_cubes := Gage.initial_red_cubes + received_red_cubes,
      total_blue_cubes := Gage.initial_blue_cubes + received_blue_cubes
  in total_red_cubes + total_blue_cubes = 35 :=
by
  sorry

end Gage_total_cubes_l618_618259


namespace smallest_pos_integer_l618_618388

-- Definitions based on the given conditions
def arithmetic_seq (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d
def sum_seq (a1 d : ℤ) (n : ℕ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

-- Given conditions
def condition1 (a1 d : ℤ) : Prop := arithmetic_seq a1 d 11 - arithmetic_seq a1 d 8 = 3
def condition2 (a1 d : ℤ) : Prop := sum_seq a1 d 11 - sum_seq a1 d 8 = 3

-- The claim we want to prove
theorem smallest_pos_integer 
  (n : ℕ) (a1 d : ℤ) 
  (h1 : condition1 a1 d) 
  (h2 : condition2 a1 d) : n = 10 :=
by
  sorry

end smallest_pos_integer_l618_618388


namespace smallest_b_for_factorization_l618_618186

theorem smallest_b_for_factorization : ∃ b : ℕ, 0 < b ∧ 
  (∃ r s : ℕ, 
    r * s = 2008 ∧ 
    r + s = b ∧ 
    (x^2 + b * x + 2008 = (x + r) * (x + s))) ∧ 
  ∀ b' : ℕ, 0 < b' ∧ 
    (∃ r s : ℕ, 
      r * s = 2008 ∧ 
      r + s = b') -> b ≤ b' :=
begin
  sorry
end

end smallest_b_for_factorization_l618_618186


namespace max_value_range_l618_618687

theorem max_value_range (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h_deriv : ∀ x, f' x = a * (x - 1) * (x - a))
  (h_max : ∀ x, (x = a → (∀ y, f y ≤ f x))) : 0 < a ∧ a < 1 :=
sorry

end max_value_range_l618_618687


namespace conjugate_solutions_and_min_value_l618_618861

theorem conjugate_solutions_and_min_value (z1 z2 z : ℂ) (x y : ℝ) :
  z1^2 = -4 ∧ z2^2 = -4 ∧ z1 = 2 * complex.I ∧ z2 = -2 * complex.I ∧ 
  |z| = 1 → 
  (z1.re = z2.re ∧ z1.im = -z2.im) ∧
  (∃ z : ℂ, |z - (z1 * z2)| = 3) :=
by 
  sorry

end conjugate_solutions_and_min_value_l618_618861


namespace proof_problem_l618_618281

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

def foci (F_1 F_2 : ℝ × ℝ) : Prop := F_1 = (-3, 0) ∧ F_2 = (3, 0)

def point_on_hyperbola (P : ℝ × ℝ) : Prop := ∃ x y, P = (x, y) ∧ hyperbola x y

def incenter (P F_1 F_2 I : ℝ × ℝ) : Prop := -- Assuming a definition for incenter
  sorry

def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

def condition (P F_1 F_2 I : ℝ × ℝ) (x y : ℝ) : Prop :=
  distance P F_1 = 2 * distance P F_2 ∧
  (I.1 - P.1, I.2 - P.2) = (x * (F_1.1 - P.1) + y * (F_2.1 - P.1), x * (F_1.2 - P.2) + y * (F_2.2 - P.2))

theorem proof_problem (P F_1 F_2 I : ℝ × ℝ) (x y : ℝ) :
  (∃ P : ℝ × ℝ, point_on_hyperbola P) →
  foci F_1 F_2 →
  incenter P F_1 F_2 I →
  condition P F_1 F_2 I x y →
  y - x = 2 / 9 :=
sorry

end proof_problem_l618_618281


namespace intersection_of_A_and_B_l618_618809

def universal_set := set ℝ
def U : universal_set := set.univ
def A : set ℕ := {x | 1 ≤ x ∧ x ≤ 10}
def B : set ℝ := {x | x^2 + x - 6 = 0}

theorem intersection_of_A_and_B : A ∩ B = {2} :=
by sorry

end intersection_of_A_and_B_l618_618809


namespace ingrid_tax_rate_proof_l618_618356

namespace TaxProblem

-- Define the given conditions
def john_income : ℝ := 56000
def ingrid_income : ℝ := 72000
def combined_income := john_income + ingrid_income

def john_tax_rate : ℝ := 0.30
def combined_tax_rate : ℝ := 0.35625

-- Calculate John's tax
def john_tax := john_tax_rate * john_income

-- Calculate total tax paid
def total_tax_paid := combined_tax_rate * combined_income

-- Calculate Ingrid's tax
def ingrid_tax := total_tax_paid - john_tax

-- Prove Ingrid's tax rate
theorem ingrid_tax_rate_proof (r : ℝ) :
  (ingrid_tax / ingrid_income) * 100 = 40 :=
  by sorry

end TaxProblem

end ingrid_tax_rate_proof_l618_618356


namespace rectangular_field_area_l618_618108

theorem rectangular_field_area (w l : ℝ) (h1 : w = l / 3) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end rectangular_field_area_l618_618108


namespace largest_prized_number_smallest_prized_number_sum_prized_numbers_divisible_by_13_l618_618957

def is_prized (n : ℕ) : Prop :=
  let digits := n.digits 10
  n ≥ 100000 ∧ n < 1000000 ∧
  digits.length = 6 ∧
  let (a, b, c, d, e, f) := (digits.get! 0, digits.get! 1, digits.get! 2, digits.get! 3, digits.get! 4, digits.get! 5)
  a + b + c = d + e + f ∧
  [a, b, c, d, e, f].nodup

noncomputable def prized_numbers : list ℕ :=
  (list.range (1000000 - 100000)).filter is_prized

theorem largest_prized_number : ∃ n ∈ prized_numbers, ∀ m ∈ prized_numbers, n ≥ m ∧ n = 981765 :=
by
  use 981765
  split
  { unfold is_prized
    simp
    split, { linarith }
    split, { linarith }
    split, { simp }
    split, { repeat { split }; linarith }
  }
  { intros m hm
    unfold is_prized at hm
    cases hm
    sorry }

theorem smallest_prized_number : ∃ n ∈ prized_numbers, ∀ m ∈ prized_numbers, n ≤ m ∧ n = 102345 :=
by
  use 102345
  split
  { unfold is_prized
    simp
    split, { linarith }
    split, { linarith }
    split, { simp }
    split, { repeat { split }; linarith }
  }
  { intros m hm
    unfold is_prized at hm
    cases hm
    sorry }

theorem sum_prized_numbers_divisible_by_13 : 13 ∣ prized_numbers.sum :=
by
  sorry

end largest_prized_number_smallest_prized_number_sum_prized_numbers_divisible_by_13_l618_618957


namespace determinant_matrices_equivalence_l618_618223

-- Define the problem as a Lean theorem statement
theorem determinant_matrices_equivalence (p q r s : ℝ) 
  (h : p * s - q * r = 3) : 
  p * (5 * r + 4 * s) - r * (5 * p + 4 * q) = 12 := 
by 
  sorry

end determinant_matrices_equivalence_l618_618223


namespace inequality_solution_l618_618161

theorem inequality_solution (x : ℝ) : 
  (x < -4 ∨ x > 2) ↔ (x^2 + 3 * x - 4) / (x^2 - x - 2) > 0 :=
sorry

end inequality_solution_l618_618161


namespace q_alone_completes_in_24_days_l618_618928

-- Conditions
variable {W_p W_q W_r : ℝ}

-- Given conditions in the problem
def condition1 : Prop := W_p = W_q + W_r
def condition2 : Prop := W_p + W_q = 1 / 10
def condition3 : Prop := W_r = 1 / 60

-- The theorem we want to prove
theorem q_alone_completes_in_24_days
    (h1 : condition1)
    (h2 : condition2)
    (h3 : condition3) : W_q = 1 / 24 := by
    sorry

end q_alone_completes_in_24_days_l618_618928


namespace length_of_interval_l618_618435

theorem length_of_interval (a b : ℝ) (h : 10 = (b - a) / 2) : b - a = 20 :=
by 
  sorry

end length_of_interval_l618_618435


namespace sqrt_diff_ineq_l618_618825

theorem sqrt_diff_ineq : 
  sqrt 6 - sqrt 5 > 2 * sqrt 2 - sqrt 7 :=
begin
  sorry
end

end sqrt_diff_ineq_l618_618825


namespace cubic_inches_needed_l618_618915

/-- The dimensions of each box are 20 inches by 20 inches by 12 inches. -/
def box_length : ℝ := 20
def box_width : ℝ := 20
def box_height : ℝ := 12

/-- The cost of each box is $0.40. -/
def box_cost : ℝ := 0.40

/-- The minimum spending required by the university on boxes is $200. -/
def min_spending : ℝ := 200

/-- Given the above conditions, the total cubic inches needed to package the collection is 2,400,000 cubic inches. -/
theorem cubic_inches_needed :
  (min_spending / box_cost) * (box_length * box_width * box_height) = 2400000 := by
  sorry

end cubic_inches_needed_l618_618915


namespace monotonic_intervals_l618_618386

noncomputable def f (x a b : ℝ) : ℝ := (x - 1)^3 - a * x - b

def derivative_f (x a : ℝ) : ℝ := 3 * (x - 1)^2 - a

theorem monotonic_intervals (a b : ℝ) :
  (∀ x : ℝ, 0 ≤ derivative_f x a) ∨
  (∃ (x1 x2 : ℝ), x1 = 1 - sqrt (a / 3) ∧ x2 = 1 + sqrt (a / 3) ∧
  (∀ x : ℝ, (x < x1 → 0 < derivative_f x a) ∧ (x1 < x ∧ x < x2 → derivative_f x a < 0) ∧ (x2 < x → 0 < derivative_f x a))) :=
by
  sorry

end monotonic_intervals_l618_618386


namespace total_ages_is_32_l618_618924

variable (a b c : ℕ)
variable (h_b : b = 12)
variable (h_a : a = b + 2)
variable (h_c : b = 2 * c)

theorem total_ages_is_32 (h_b : b = 12) (h_a : a = b + 2) (h_c : b = 2 * c) : a + b + c = 32 :=
by
  sorry

end total_ages_is_32_l618_618924


namespace number_of_green_cards_l618_618457

theorem number_of_green_cards :
  ∀ (total_cards red_fraction black_fraction : ℕ),
    total_cards = 120 →
    red_fraction = 2 / 5 →
    black_fraction = 5 / 9 →
    let red_cards := (red_fraction * total_cards) in
    let non_red_cards := (total_cards - red_cards) in
    let black_cards := (black_fraction * non_red_cards) in
    let green_cards := (non_red_cards - black_cards) in
    green_cards = 32 := by
  sorry

end number_of_green_cards_l618_618457


namespace function_even_l618_618870

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem function_even : is_even (λ x : ℝ, -3 * x^4) :=
by
  sorry

end function_even_l618_618870


namespace dealership_vans_expected_l618_618518

theorem dealership_vans_expected (trucks vans : ℕ) (h_ratio : 3 * vans = 5 * trucks) (h_trucks : trucks = 45) : vans = 75 :=
by
  sorry

end dealership_vans_expected_l618_618518


namespace cos_angle_between_diagonals_l618_618952

/-- Definitions for the vectors that define the parallelogram -/
def a : ℝ × ℝ × ℝ := ⟨3, 2, 1⟩
def b : ℝ × ℝ × ℝ := ⟨1, 3, 2⟩

/-- Definitions for the diagonal vectors -/
def diagonal1 : ℝ × ℝ × ℝ := ⟨a.1 + b.1, a.2 + b.2, a.3 + b.3⟩
def diagonal2 : ℝ × ℝ × ℝ := ⟨b.1 - a.1, b.2 - a.2, b.3 - a.3⟩

/-- Cosine of the angle between the diagonals of the parallelogram -/
noncomputable def cos_theta : ℝ :=
(∑ i in (finset.range 3), (diagonal1.η i) * (diagonal2.η i)) /
(real.sqrt (∑ i in (finset.range 3), (diagonal1.η i)^2) *
 real.sqrt (∑ i in (finset.range 3), (diagonal2.η i)^2))

/-- Theorem stating the problem conclusion -/
theorem cos_angle_between_diagonals : cos_theta = 0 :=
by sorry

end cos_angle_between_diagonals_l618_618952


namespace Kolya_made_the_mistake_l618_618003

def pencils_in_box (blue green : ℕ) : Prop :=
  (blue ≥ 4 ∨ blue < 4) ∧ (green ≥ 4 ∨ green < 4)

def boys_statements (blue green : ℕ) : Prop :=
  (Vasya : blue ≥ 4) ∧
  (Kolya : green ≥ 5) ∧
  (Petya : blue ≥ 3 ∧ green ≥ 4) ∧
  (Misha : blue ≥ 4 ∧ green ≥ 4)

theorem Kolya_made_the_mistake:
  ∀ {blue green : ℕ},
  pencils_in_box blue green →
  boys_statements blue green →
  ∃ (Vasya_truth Petya_truth Misha_truth : Prop),
  Vasya_truth ∧ Petya_truth ∧ Misha_truth ∧ ¬ Kolya_truth :=
begin
  sorry
end

end Kolya_made_the_mistake_l618_618003


namespace other_root_of_quadratic_l618_618664

theorem other_root_of_quadratic (k : ℝ) :
  ((2:ℝ) satisfies_roots (2:ℝ) (k - 5) (4 - k)) →
  (∃ (r : ℝ), r satisfies_roots (2:ℝ) (k - 5) (4 - k) ∧ r ≠ 2 ∧ r = 1) :=
sorry

end other_root_of_quadratic_l618_618664


namespace problem1_solution_correct_problem2_solution_correct_problem3_solution_correct_l618_618055

-- Problem 1
theorem problem1_solution_correct : 
  (∀ x, deriv (λ x, 3 * sin (5 * x)) x = 15 * cos (5 * x)) ∧ 
  (∀ x, deriv (λ x, 15 * cos (5 * x)) x = -75 * sin (5 * x)) ∧
  (∀ x, deriv (λ x, (1 / 5) * cos (5 * x)) x = -sin (5 * x)) ∧
  (∀ x, deriv (λ x, -sin (5 * x)) x = -5 * cos (5 * x)) ∧
  (∀ x, deriv (λ x, 15 * cos (5 * x) + 25 * 3 * sin (5 * x)) x = 0) ∧
  (∀ x, deriv (λ x, -5 * cos (5 * x) + 25 * (1 / 5) * cos (5 * x)) x = 0) →
  ∀ (C1 C2 : ℝ) (x : ℝ), 
  ∃ (y : ℝ → ℝ), y = C1 * sin (5 * x) + C2 * cos (5 * x) := 
  sorry

-- Problem 2
theorem problem2_solution_correct :
  (∀ x, deriv (λ x, exp (3 * x)) x = 3 * exp (3 * x)) ∧
  (∀ x, deriv (λ x, x * exp (3 * x)) x = exp (3 * x) + 3 * x * exp (3 * x)) ∧
  (∀ x, deriv (λ x, exp (3 * x) + 3 * x * exp (3 * x)) x = 3 * exp (3 * x) + 3 * (exp (3 * x) + 3 * x * exp (3 * x))) ∧
  (∀ x, deriv (λ x, 3 * exp (3 * x) + 3 * exp (3 * x) + 9 * x * exp (3 * x) - 6 * (exp (3 * x) + 3 * x * exp (3 * x)) + 9 * x * exp (3 * x)) x = 0) →
  ∀ (C1 C2 : ℝ) (x : ℝ),
  ∃ (y : ℝ → ℝ), y = C1 * exp (3 * x) + C2 * x * exp (3 * x) :=
  sorry

-- Problem 3
theorem problem3_solution_correct : 
  (∀ x, deriv (λ x, exp (2 * x)) x = 2 * exp (2 * x)) ∧ 
  (∀ x, deriv (λ x, exp (-3 * x)) x = -3 * exp (-3 * x)) ∧ 
  (∀ x, deriv (λ x, -3 * exp (-3 * x)) x = 9 * exp (-3 * x)) ∧
  (∀ x, deriv (λ x, 9 * exp (-3 * x) - 3 * exp (-3 * x) - 6 * exp (-3 * x)) x = 0) →
  ∀ (C1 C2 : ℝ) (x : ℝ), 
  ∃ (y : ℝ → ℝ), y = C1 * exp (2 * x) + C2 * exp (-3 * x) := 
  sorry

end problem1_solution_correct_problem2_solution_correct_problem3_solution_correct_l618_618055


namespace increasing_sequence_a1_range_l618_618283

theorem increasing_sequence_a1_range
  (a : ℕ → ℝ) 
  (h : ∀ n, a (n + 1) = (4 * a n - 2) / (a n + 1))
  (strictly_increasing : ∀ n, a (n + 1) > a n) :
  1 < a 1 ∧ a 1 < 2 :=
sorry

end increasing_sequence_a1_range_l618_618283


namespace largest_initial_number_l618_618337

-- Let's define the conditions and the result
def valid_addition (n a : ℕ) : Prop := ∃ k : ℕ, n = a * k + r ∧ 0 < r ∧ r < a

def valid_operations (initial : ℕ) (final : ℕ) (steps : ℕ → ℕ → ℕ) : Prop :=
  ∃ (a b c d e : ℕ), valid_addition initial a ∧
                      valid_addition (initial + a) b ∧
                      valid_addition (initial + a + b) c ∧
                      valid_addition (initial + a + b + c) d ∧
                      valid_addition (initial + a + b + c + d) e ∧
                      initial + a + b + c + d + e = final

theorem largest_initial_number :
  ∃ n : ℕ, (valid_operations n 200 (λn a, n + a)) ∧ (∀ m : ℕ, valid_operations m 200 (λn a, n + a) → m ≤ n) :=
sorry

end largest_initial_number_l618_618337


namespace integer_roots_of_quadratic_l618_618175

theorem integer_roots_of_quadratic (b : ℤ) :
  (∃ x : ℤ, x^2 + 4 * x + b = 0) ↔ b = -12 ∨ b = -5 ∨ b = 3 ∨ b = 4 :=
sorry

end integer_roots_of_quadratic_l618_618175


namespace g_f_eval_l618_618369

def f (x : ℤ) := x^3 - 2
def g (x : ℤ) := 3 * x^2 + x + 2

theorem g_f_eval : g (f 3) = 1902 := by
  sorry

end g_f_eval_l618_618369


namespace total_pages_in_book_l618_618768

theorem total_pages_in_book (P : ℕ)
  (first_day : P - (P / 5) - 12 = remaining_1)
  (second_day : remaining_1 - (remaining_1 / 4) - 15 = remaining_2)
  (third_day : remaining_2 - (remaining_2 / 3) - 18 = 42) :
  P = 190 := 
sorry

end total_pages_in_book_l618_618768


namespace infinite_pairs_for_equation_l618_618828

theorem infinite_pairs_for_equation :
  ∃ᶠ (a b : ℤ) in filter.cofinite ℤ, ∃ (x y : ℝ),
  x ≠ y ∧ x * y = 1 ∧ x ^ 2012 = a * x + b ∧ y ^ 2012 = a * y + b :=
sorry

end infinite_pairs_for_equation_l618_618828


namespace valid_outfits_number_l618_618712

def num_shirts := 7
def num_pants := 7
def num_hats := 7
def num_colors := 7

def total_outfits (num_shirts num_pants num_hats : ℕ) := num_shirts * num_pants * num_hats
def matching_color_outfits (num_colors : ℕ) := num_colors
def valid_outfits (num_shirts num_pants num_hats num_colors : ℕ) := 
  total_outfits num_shirts num_pants num_hats - matching_color_outfits num_colors

theorem valid_outfits_number : valid_outfits num_shirts num_pants num_hats num_colors = 336 := 
by
  sorry

end valid_outfits_number_l618_618712


namespace rightmost_three_digits_of_3_pow_2023_l618_618059

theorem rightmost_three_digits_of_3_pow_2023 :
  (3^2023) % 1000 = 787 := 
sorry

end rightmost_three_digits_of_3_pow_2023_l618_618059


namespace length_of_perpendicular_segment_l618_618838

-- Define the problem conditions
theorem length_of_perpendicular_segment
  (AD BE CF : ℝ)
  (hAD : AD = 8)
  (hBE : BE = 10)
  (hCF : CF = 20)
  (angle_RS : ℝ)
  (h_angle : angle_RS = 45) :
  let y_A := AD * real.sqrt 2
  let y_B := BE * real.sqrt 2
  let y_C := CF * real.sqrt 2
  let y_G := (y_A + y_B + y_C) / 3
  in GH = y_G :=
begin
  -- We have conditions given in the problem
  -- Now need to prove x = (38 * sqrt 2) / 3
  -- Proof skipped
  sorry
end

end length_of_perpendicular_segment_l618_618838


namespace min_sum_2310_vol_l618_618427

noncomputable def minimum_sum_of_dimensions : ℕ :=
  let V := 2310
  infi (λ (abc : ℕ × ℕ × ℕ), if abc.1 * abc.2 * abc.3 = V then abc.1 + abc.2 + abc.3 else V + 1) -- effectively a large number

theorem min_sum_2310_vol (a b c : ℕ) (h : a * b * c = 2310) : a + b + c ≥ 48 :=
  sorry

end min_sum_2310_vol_l618_618427


namespace max_initial_number_l618_618327

noncomputable def verify_addition (n x : ℕ) : Prop := 
  ∀ (a : ℕ), (a ∣ n) → (a ≠ 1) → (n + a = x) → False

theorem max_initial_number :
  ∃ (n : ℕ), 
  (∀ (a1 a2 a3 a4 a5 : ℕ), 
    verify_addition n a1 ∧ verify_addition (n + a1) a2 ∧
    verify_addition (n + a1 + a2) a3 ∧ verify_addition (n + a1 + a2 + a3) a4 ∧
    verify_addition (n + a1 + a2 + a3 + a4) a5 ∧
    (n + a1 + a2 + a3 + a4 + a5 = 200)) ∧
  (∀ m : ℕ, 
    (∃ (a1 a2 a3 a4 a5 : ℕ), 
      verify_addition m a1 ∧ verify_addition (m + a1) a2 ∧
      verify_addition (m + a1 + a2) a3 ∧ verify_addition (m + a1 + a2 + a3) a4 ∧
      verify_addition (m + a1 + a2 + a3 + a4) a5 ∧
      (m + a1 + a2 + a3 + a4 + a5 = 200)) →
    m ≤ 189)
: ∃ n, n = 189 := by
  sorry

end max_initial_number_l618_618327


namespace integral_abs_x2_minus_x_coeff_ab2c3_expansion_ball_distribution_range_of_a_l618_618089

-- 1. Integral Problem
theorem integral_abs_x2_minus_x : 
  ∫ x in -1..1, |x^2 - x| = 1 :=
sorry

-- 2. Coefficient Problem
theorem coeff_ab2c3_expansion : 
  polynomial.coeff ((a + 2 * b - 3 * c)^6) (finsupp.single a 1 + finsupp.single b 2 + finsupp.single c 3) = -6480 :=
sorry

-- 3. Combinatorics Problem
theorem ball_distribution : 
  ∃ (f : ℕ → ℕ), fintype.card {f // f 1 + f 2 + f 3 + f 4 = 13 ∧ f 1 ≥ 1 ∧ f 2 ≥ 2 ∧ f 3 ≥ 3 ∧ f 4 ≥ 4} = 20 :=
sorry

-- 4. Inequality Problem
theorem range_of_a (m : ℝ) (hm : 0 < m) : 
  (∃ a : ℝ, (∀ x : ℝ, x * (1 + 2 * a * ((x+m)/x - 2 * exp(1)) * log((x+m)/x)) = 0) → a ∈ (1 / (2 * exp(1)), ∞)) :=
sorry


end integral_abs_x2_minus_x_coeff_ab2c3_expansion_ball_distribution_range_of_a_l618_618089


namespace compute_pounds_of_cotton_l618_618504

theorem compute_pounds_of_cotton (x : ℝ) :
  (5 * 30 + 10 * x = 640) → (x = 49) := by
  intro h
  sorry

end compute_pounds_of_cotton_l618_618504


namespace factorial_root_inequality_l618_618487

theorem factorial_root_inequality (h : fact 8! < 9^8) : (8!) ^ (1 / 8 : ℝ) < (9!) ^ (1 / 9 : ℝ) :=
sorry

end factorial_root_inequality_l618_618487


namespace seq_bn_arithmetic_seq_an_formula_sum_an_terms_l618_618754

-- (1) Prove that the sequence {b_n} is an arithmetic sequence
theorem seq_bn_arithmetic (a : ℕ → ℕ) (b : ℕ → ℤ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = 2 * a n + 2^n)
  (h3 : ∀ n, b n = a n / 2^(n - 1)) :
  ∀ n, b (n + 1) - b n = 1 := by
  sorry

-- (2) Find the general formula for the sequence {a_n}
theorem seq_an_formula (a : ℕ → ℕ) (b : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = 2 * a n + 2^n)
  (h3 : ∀ n, b n = a n / 2^(n - 1)) :
  ∀ n, a n = n * 2^(n - 1) := by
  sorry

-- (3) Find the sum of the first n terms of the sequence {a_n}
theorem sum_an_terms (a : ℕ → ℕ) (S : ℕ → ℤ) (h1 : ∀ n, a n = n * 2^(n - 1)) :
  ∀ n, S n = (n - 1) * 2^n + 1 := by
  sorry

end seq_bn_arithmetic_seq_an_formula_sum_an_terms_l618_618754


namespace smallest_x_value_l618_618640

theorem smallest_x_value : ∀ x : ℚ, (14 * x^2 - 40 * x + 18) / (4 * x - 3) + 6 * x = 7 * x - 2 → x = 4 / 5 :=
by
  intros x hx
  sorry

end smallest_x_value_l618_618640


namespace DC_value_l618_618297

theorem DC_value (AB BD BC CD : ℝ)
  (A B C D : Type*)
  (h_AB : AB = 30)
  (h_angle_ADB : ∠ ADB = 90)
  (h_sin_A : sin A = 4 / 5)
  (h_sin_C : sin C = 1 / 4)
  (h_BD : BD = (4 / 5) * 30)
  (h_BC : BC = 4 * BD)
  (h_CD : CD = sqrt (BC^2 - BD^2)) :
  CD = 24 * sqrt 15 :=
by
  sorry

end DC_value_l618_618297


namespace maximum_value_of_A_l618_618648

theorem maximum_value_of_A (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
    (a^4 + b^4 + c^4) / ((a + b + c)^4 - 80 * (a * b * c)^(4 / 3)) ≤ 3 :=
sorry

end maximum_value_of_A_l618_618648


namespace smallest_n_modulo_l618_618911

theorem smallest_n_modulo :
  ∃ n : ℕ, 0 < n ∧ 5 * n % 26 = 1846 % 26 ∧ n = 26 :=
by
  sorry

end smallest_n_modulo_l618_618911


namespace abs_nonneg_l618_618087

theorem abs_nonneg (a : ℝ) : |a| ≥ 0 := sorry

end abs_nonneg_l618_618087


namespace who_made_a_mistake_l618_618034

-- Definitions of the conditions
def at_least_four_blue_pencils (B : ℕ) : Prop := B ≥ 4
def at_least_five_green_pencils (G : ℕ) : Prop := G ≥ 5
def at_least_three_blue_and_four_green_pencils (B G : ℕ) : Prop := B ≥ 3 ∧ G ≥ 4
def at_least_four_blue_and_four_green_pencils (B G : ℕ) : Prop := B ≥ 4 ∧ G ≥ 4

-- The main theorem stating who made a mistake
theorem who_made_a_mistake (B G : ℕ) 
  (hv : at_least_four_blue_pencils B)
  (hk : at_least_five_green_pencils G)
  (hp : at_least_three_blue_and_four_green_pencils B G)
  (hm : at_least_four_blue_and_four_green_pencils B G) 
  (h_truth : (hv ∧ hk ∧ hp ∧ hm) ∨ (¬hv ∧ hk ∧ hp ∧ hm) ∨ (hv ∧ ¬hk ∧ hp ∧ hm) ∨ (hv ∧ hk ∧ ¬hp ∧ hm) ∨ (hv ∧ hk ∧ hp ∧ ¬hm))
  (h_truthful: ∑ b in [hv, hk, hp, hm], (if b then 1 else 0) = 3) : 
  hk = false := 
sorry

end who_made_a_mistake_l618_618034


namespace largest_subset_size_l618_618987

theorem largest_subset_size :
  ∃ S : Set ℕ, S ⊆ {i | 1 ≤ i ∧ i ≤ 150} ∧ 
  (∀ (a b : ℕ), a ∈ S → b ∈ S → ¬ (a = 4 * b ∨ b = 4 * a)) ∧ 
  S.card = 141 :=
sorry

end largest_subset_size_l618_618987


namespace toll_booth_ratio_l618_618744

theorem toll_booth_ratio (total_cars : ℕ) (monday_cars tuesday_cars friday_cars saturday_cars sunday_cars : ℕ)
  (x : ℕ) (h1 : total_cars = 450) (h2 : monday_cars = 50) (h3 : tuesday_cars = 50) (h4 : friday_cars = 50)
  (h5 : saturday_cars = 50) (h6 : sunday_cars = 50) (h7 : monday_cars + tuesday_cars + x + x + friday_cars + saturday_cars + sunday_cars = total_cars) :
  x = 100 ∧ x / monday_cars = 2 :=
by
  sorry

end toll_booth_ratio_l618_618744


namespace football_goals_even_more_probable_l618_618520

-- Define the problem statement and conditions
variable (p_1 : ℝ) (h₀ : 0 ≤ p_1 ∧ p_1 ≤ 1) (h₁ : q_1 = 1 - p_1)

-- Define even and odd goal probabilities for the total match
def p : ℝ := p_1^2 + (1 - p_1)^2
def q : ℝ := 2 * p_1 * (1 - p_1)

-- The main statement to prove
theorem football_goals_even_more_probable (h₂ : q_1 = 1 - p_1) : p_1^2 + (1 - p_1)^2 ≥ 2 * p_1 * (1 - p_1) :=
  sorry

end football_goals_even_more_probable_l618_618520


namespace number_of_zeros_of_f_l618_618871

noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x

theorem number_of_zeros_of_f :
  {x : ℝ | f x = 0} ∩ {x : ℝ | x ≠ 0} = {-1} :=
begin
  sorry
end

end number_of_zeros_of_f_l618_618871


namespace repeating_decimal_sum_l618_618599

/-- 
Prove that the repeating decimal 0.787878... can be expressed in its simplest fractional form 
as 26/33, and the sum of its numerator and denominator is 59.
-/
theorem repeating_decimal_sum (x : ℚ) (h : x = 0.7878): (numer : ℕ) × (denom : ℕ) × (numer + denom = 59) :=
by
  let numer := 26
  let denom := 33
  have h_frac : x = 26 / 33, sorry
  have h_sum : 26 + 33 = 59, sorry
  exact ⟨numer, denom, h_sum⟩

end repeating_decimal_sum_l618_618599


namespace orange_marbles_l618_618462

-- Definitions based on the given conditions
def total_marbles : ℕ := 24
def blue_marbles : ℕ := total_marbles / 2
def red_marbles : ℕ := 6

-- The statement to prove: the number of orange marbles is 6
theorem orange_marbles : (total_marbles - (blue_marbles + red_marbles)) = 6 := 
  by 
  sorry

end orange_marbles_l618_618462


namespace avg_remaining_two_l618_618422

theorem avg_remaining_two (avg5 avg3 : ℝ) (h1 : avg5 = 12) (h2 : avg3 = 4) : (5 * avg5 - 3 * avg3) / 2 = 24 :=
by sorry

end avg_remaining_two_l618_618422


namespace problem_part1_problem_part2_l618_618227

theorem problem_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
  a + b + c ≥ 1 / Real.sqrt a + 1 / Real.sqrt b + 1 / Real.sqrt c := 
sorry

theorem problem_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
  a^2 + b^2 + c^2 ≥ Real.sqrt a + Real.sqrt b + Real.sqrt c :=
sorry

end problem_part1_problem_part2_l618_618227


namespace right_triangle_incenter_length_l618_618453

noncomputable def isRightTriangle (A B C : Point) : Prop := sorry
noncomputable def incenter (ABC : Triangle) : Point := sorry
noncomputable def length (A B : Point) : ℝ := sorry
noncomputable def Point := sorry
noncomputable def Triangle := sorry

theorem right_triangle_incenter_length {A B C I : Point} (h : isRightTriangle A B C) (hAB : length A B = 6) (hBC : length B C = 8) (hI : I = incenter ⟨A, B, C⟩) : length B I = 5 :=
sorry

end right_triangle_incenter_length_l618_618453


namespace power_quotient_l618_618580

theorem power_quotient (a m n : ℕ) (h_a : a = 19) (h_m : m = 11) (h_n : n = 8) : a^m / a^n = 6859 := by
  sorry

end power_quotient_l618_618580


namespace job_completion_time_l618_618927

theorem job_completion_time
  (A C : ℝ)
  (A_rate : A = 1 / 6)
  (C_rate : C = 1 / 12)
  (B_share : 390 / 1170 = 1 / 3) :
  ∃ B : ℝ, B = 1 / 8 ∧ (B * 8 = 1) :=
by
  -- Proof omitted
  sorry

end job_completion_time_l618_618927


namespace expression_approx_l618_618937

noncomputable def simplified_expression : ℝ :=
  (Real.sqrt 97 + 9 * Real.sqrt 6 + 5 * Real.sqrt 5) / (3 * Real.sqrt 6 + 7)

theorem expression_approx : abs (simplified_expression - 3.002) < 0.001 :=
by
  -- Proof omitted
  sorry

end expression_approx_l618_618937


namespace number_of_perfect_square_n_l618_618650

theorem number_of_perfect_square_n : 
  { n : ℕ // 0 ≤ n ∧ n < 30 ∧ ∃ k : ℕ, (n / (30 - n) = k^2) }.to_finset.card = 3 := 
sorry

end number_of_perfect_square_n_l618_618650


namespace ChernovHairColor_l618_618810

-- Definitions for characters and their roles
def Sedov := "Master of Sports"
def Chernov := "Candidate Master"
def Ryzhov := "First-rank Sportsman"

-- Hair colors
inductive HairColor 
| grey
| red
| black

open HairColor

-- Define that no one's hair color matches their surname
axiom SedovNotGrey : Sedov ≠ HairColor.grey
axiom RyzhovNotRed : Ryzhov ≠ HairColor.red
axiom ChernovNotBlack : Chernov ≠ HairColor.black

-- Proof problem to show Chernov's hair color
theorem ChernovHairColor : (Sedov ≠ HairColor.grey ∧ Ryzhov ≠ HairColor.red ∧ Chernov ≠ HairColor.black) → ∃ (color : HairColor), color = HairColor.grey :=
by
  intro h
  sorry

end ChernovHairColor_l618_618810


namespace third_pipe_empty_time_l618_618500

theorem third_pipe_empty_time :
  (∀ (A B C : ℝ), 
     A = 1/60 ∧ 
     B = 1/75 ∧ 
     A + B - C = 1/50 → 
     C = 1/100) :=
begin
  intros A B C h,
  cases h with hA hBC,
  cases hBC with hB hC,
  rw [hA, hB] at hC,
  sorry
end

end third_pipe_empty_time_l618_618500


namespace reginald_apples_sold_l618_618830

variable (price_per_apple : ℚ)
variable (bike_cost : ℚ)
variable (repair_percentage : ℚ)
variable (remaining_percentage : ℚ)
variable (apples_sold : ℕ)

-- Constants
def price_per_apple := 1.25
def bike_cost := 80
def repair_percentage := 0.25
def remaining_percentage := 0.20  -- Since 1 - 0.20 = 0.80, we have 4/5 of the total earnings used for repair

-- Calculations
def repair_cost := repair_percentage * bike_cost
def total_earnings := (repair_cost / (1 - remaining_percentage))
def expected_apples_sold := total_earnings / price_per_apple

-- Theorem
theorem reginald_apples_sold : apples_sold = 20 :=
by
  -- Setting the constants in a consistent state
  let price_per_apple := 1.25
  let bike_cost := 80
  let repair_percentage := 0.25
  let remaining_percentage := 0.20
  let apples_sold := 20
  
  -- Calculate repair cost
  have repair_cost_eq : repair_cost = 20 :=
    by sorry
  
  -- Calculate total earnings
  have total_earnings_eq : total_earnings = 25 :=
    by sorry
  
  -- Calculate expected apples sold
  have expected_apples_eq : expected_apples_sold = 20 :=
    by sorry

  -- Thus, proven
  exact sorry

end reginald_apples_sold_l618_618830


namespace find_b_l618_618357

def g (x b : ℝ) : ℝ := x / (b * x + 1)

theorem find_b (b : ℝ) : (∀ x : ℝ, x ≠ -1/b → g (g x b) b = x) ↔ b = -1 :=
by
  sorry

end find_b_l618_618357


namespace line_not_in_fourth_quadrant_l618_618526

-- Let the line be defined as y = 3x + 2
def line_eq (x : ℝ) : ℝ := 3 * x + 2

-- The Fourth quadrant is defined by x > 0 and y < 0
def in_fourth_quadrant (x : ℝ) (y : ℝ) : Prop := x > 0 ∧ y < 0

-- Prove that the line does not intersect the Fourth quadrant
theorem line_not_in_fourth_quadrant : ¬ (∃ x : ℝ, in_fourth_quadrant x (line_eq x)) :=
by
  -- Proof goes here (abstracted)
  sorry

end line_not_in_fourth_quadrant_l618_618526


namespace f_odd_and_increasing_l618_618245

noncomputable def f (x : ℝ) : ℝ := 3^x - (1/3)^x

theorem f_odd_and_increasing : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
by
  sorry

end f_odd_and_increasing_l618_618245


namespace distance_AB_l618_618753

-- Definitions based on conditions
def curve (φ : ℝ) : ℝ × ℝ := (2 * Real.cos φ, 2 + 2 * Real.sin φ)

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

def pointA : ℝ × ℝ := polar_to_cartesian (2 * Real.sqrt 3) (Real.pi / 3)
def pointB : ℝ × ℝ := polar_to_cartesian 2 (5 * Real.pi / 6)

-- Distance formula
def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- The statement to be proved
theorem distance_AB : distance pointA pointB = 4 := by
  sorry

end distance_AB_l618_618753


namespace correct_number_of_outfits_l618_618717

def num_shirts : ℕ := 7
def num_pants : ℕ := 7
def num_hats : ℕ := 7
def num_colors : ℕ := 7

def total_outfits : ℕ := num_shirts * num_pants * num_hats
def invalid_outfits : ℕ := num_colors
def valid_outfits : ℕ := total_outfits - invalid_outfits

theorem correct_number_of_outfits : valid_outfits = 336 :=
by {
  -- sorry can be removed when providing the proof.
  sorry
}

end correct_number_of_outfits_l618_618717


namespace work_completion_days_l618_618159

theorem work_completion_days (david_days john_days mary_days : ℕ) (david_work john_work mary_work: ℕ)
    (david_days_eq : david_days = 5)
    (john_days_eq : john_days = 9)
    (mary_days_eq : mary_days = 7)
    (david_work_eq : david_work = 1)
    (john_work_eq : john_work = 1)
    (mary_work_eq : mary_work = 1) :
  let total_work := 1000,
      work_per_day := (david_work / david_days) + (john_work / john_days) + (mary_work / mary_days),
      days_needed := total_work / work_per_day
  in days_needed ≈ 3 := 
by
  sorry

end work_completion_days_l618_618159


namespace evaluate_expression_l618_618170

theorem evaluate_expression : abs (abs (abs (-2 + 2) - 2) * 2) = 4 := 
by
  sorry

end evaluate_expression_l618_618170


namespace trajectory_curve_eqn_line_eqn_intersecting_curve_l618_618672

-- Conditions
variables {x₀ y₀ : ℝ} (A B : ℝ × ℝ) (OP OA OB : ℝ × ℝ)
variables {x y : ℝ}

-- Definitions of the points and their binary relationships
def A := (x₀, 0)
def B := (0, y₀)
def P := (x, y)
def OP := P
def OA := A
def OB := B

-- Given distance |AB| = 1
axiom AB_eq_1 : Real.sqrt (x₀ ^ 2 + y₀ ^ 2) = 1

-- Given vector equation
axiom OP_eq_eqn : P = (2 * x₀, Real.sqrt 3 * y₀)

-- Equivalent math proof problems rewritten in Lean
theorem trajectory_curve_eqn :
  (x₀ = x / 2) ∧ (y₀ = y * Real.sqrt 3 / 3) → 
  (x ^ 2 / 4 + y ^ 2 / 3 = 1) :=
sorry

theorem line_eqn_intersecting_curve :
  (∀ k : ℝ, ∃ x₁ x₂ y₁ y₂ : ℝ, 
      (y = k * x + 2) ∧ 
      (x₁ ^ 2 / 4 + (k * x₁ + 2) ^ 2 / 3 = 1) ∧
      (x₂ ^ 2 / 4 + (k * x₂ + 2) ^ 2 / 3 = 1) ∧
      (x₁ + x₂ = 16 * k / (3 + 4 * k ^ 2)) ∧
      (x₁ * x₂ = 4 / (3 + 4 * k ^ 2)) ∧
      (x₁ * x₂ + (k * x₁ + 2) * (k * x₂ + 2) = 0)) →
  (k = 2 * Real.sqrt 3 / 3 ∨ k = -2 * Real.sqrt 3 / 3) :=
sorry

end trajectory_curve_eqn_line_eqn_intersecting_curve_l618_618672


namespace solve_quadratic_l618_618850

noncomputable section

def a : ℝ := 4
def b : ℝ := -8
def c : ℝ := 1

def discriminant : ℝ := b^2 - 4 * a * c
def root1 : ℝ := (2 + Math.sqrt 3) / 2
def root2 : ℝ := (2 - Math.sqrt 3) / 2

theorem solve_quadratic :
  (∃ x : ℝ, 4 * x^2 - 8 * x + 1 = 0) ∧
  (root1 = (2 + Real.sqrt 3) / 2) ∧
  (root2 = (2 - Real.sqrt 3) / 2) :=
by
  sorry

end solve_quadratic_l618_618850


namespace ch4_moles_formed_l618_618184

theorem ch4_moles_formed : 
    ∀ (Be2C H2O CH4 BeOH2 : Type) 
      (moles_Be2C moles_H2O moles_CH4 : ℕ), 
    (∀ (x y : ℕ), (x * y = moles_Be2C) → (4 * y = moles_H2O)) →
    (∀ (a : ℕ), a = 3 → (a * moles_CH4 = 9)) → 
    (moles_Be2C = 3) ∧ (moles_H2O = 12) → 
    moles_CH4 = 9 :=
by
  intros Be2C H2O CH4 BeOH2 moles_Be2C moles_H2O moles_CH4 h1 h2 h3
  cases h3 with h3_1 h3_2
  have moles_CH4_by_react := h2 3 rfl
  sorry

end ch4_moles_formed_l618_618184


namespace minimum_distance_sum_l618_618364

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem minimum_distance_sum
  (A B : ℝ × ℝ)
  (A_x A_y B_x B_y : ℝ) 
  (A : A = (A_x, A_y) := by rfl)
  (B : B = (B_x, B_y) := by rfl):
  A = (1, 1) → B = (3, 4) → (P : ℝ × ℝ) → P.2 = 0 →
  distance P A + distance P B = distance (1, -1) B :=
sorry

end minimum_distance_sum_l618_618364


namespace no_arithmetic_progression_roots_l618_618176

theorem no_arithmetic_progression_roots (a : ℝ) :
  ¬ ∃ (x d : ℝ), 
    (∃ r : Fin 4 → ℝ, 
       (r 0 = x - d ∧ 
        r 1 = x ∧ 
        r 2 = x + d ∧ 
        r 3 = x + 2 * d ∧ 
        Polynomial.eval ↑a 
          (Polynomial.C 16 * X^4 
                       - Polynomial.C a * X^3 
                       + Polynomial.C (2 * a + 17) * X^2 
                       - Polynomial.C a * X 
                       + Polynomial.C 16) = 0)) := 
begin
  sorry
end

end no_arithmetic_progression_roots_l618_618176


namespace eccentricity_correct_l618_618217

noncomputable def eccentricity_of_ellipse
  (a b : ℝ) (h₀ : a > b) (h₁ : b > 0)
  (h₂ : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1)
  (h₃ : ∀ x y : ℝ, x^2 + y^2 = a^2)
  (h₄ : ∀ x y : ℝ, bx - ay + 2 * a * b = 0) : ℝ :=
  let c := sqrt (a^2 - b^2) in
  (c/a)

theorem eccentricity_correct (a b : ℝ) (h₀ : a > b) (h₁ : b > 0)
  (h₂ : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (h₃ : ∀ x y : ℝ, x^2 + y^2 = a^2)
  (h₄ : ∀ x y : ℝ, bx - ay + 2 * a * b = 0) :
  eccentricity_of_ellipse a b h₀ h₁ h₂ h₃ h₄ = sqrt 6 / 3 :=
sorry

end eccentricity_correct_l618_618217


namespace bell_pepper_slices_l618_618418

theorem bell_pepper_slices :
  ∀ (num_peppers : ℕ) (slices_per_pepper : ℕ) (total_slices_pieces : ℕ) (half_slices : ℕ),
  num_peppers = 5 → slices_per_pepper = 20 → total_slices_pieces = 200 →
  half_slices = (num_peppers * slices_per_pepper) / 2 →
  (total_slices_pieces - (num_peppers * slices_per_pepper)) / half_slices = 2 :=
by
  intros num_peppers slices_per_pepper total_slices_pieces half_slices h1 h2 h3 h4
  -- skip the proof with sorry as instructed
  sorry

end bell_pepper_slices_l618_618418


namespace larger_integer_is_14_l618_618899

noncomputable def isLargerInteger (a b : ℕ) : Prop :=
  a * b = 168 ∧ abs (a - b) = 4

theorem larger_integer_is_14 (a b : ℕ) (h : isLargerInteger a b) : a = 14 ∨ b = 14 :=
by
  sorry

end larger_integer_is_14_l618_618899


namespace kolya_mistaken_l618_618006

-- Definitions relating to the conditions
def at_least_four_blue_pencils (blue_pencils : ℕ) : Prop := blue_pencils >= 4
def at_least_five_green_pencils (green_pencils : ℕ) : Prop := green_pencils >= 5
def at_least_three_blue_pencils_and_four_green_pencils (blue_pencils green_pencils : ℕ) : Prop := blue_pencils >= 3 ∧ green_pencils >= 4
def at_least_four_blue_and_four_green_pencils (blue_pencils green_pencils : ℕ) : Prop := blue_pencils >= 4 ∧ green_pencils >= 4

-- Speaking truth conditions
variables (blue_pencils green_pencils : ℕ)
def vasya_true : Prop := at_least_four_blue_pencils blue_pencils
def kolya_true : Prop := at_least_five_green_pencils green_pencils
def petya_true : Prop := at_least_three_blue_pencils_and_four_green_pencils blue_pencils green_pencils
def misha_true : Prop := at_least_four_blue_and_four_green_pencils blue_pencils green_pencils

-- Given known information: three are true, one is false
def known_information : Prop := (vasya_true blue_pencils green_pencils ∧ petya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬kolya_true blue_pencils green_pencils)
                            ∨ (vasya_true blue_pencils green_pencils ∧ petya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ ¬misha_true blue_pencils green_pencils)
                            ∨ (vasya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬petya_true blue_pencils green_pencils)
                            ∨ (petya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬vasya_true blue_pencils green_pencils)

-- Theorem to be proved
theorem kolya_mistaken : known_information blue_pencils green_pencils → ¬kolya_true blue_pencils green_pencils :=
sorry

end kolya_mistaken_l618_618006


namespace vision_statistics_l618_618892

noncomputable def average (values : List ℝ) : ℝ := (List.sum values) / (List.length values)

noncomputable def variance (values : List ℝ) : ℝ :=
  let mean := average values
  (List.sum (values.map (λ x => (x - mean) ^ 2))) / (List.length values)

def classA_visions : List ℝ := [4.3, 5.1, 4.6, 4.1, 4.9]
def classB_visions : List ℝ := [5.1, 4.9, 4.0, 4.0, 4.5]

theorem vision_statistics :
  average classA_visions = 4.6 ∧
  average classB_visions = 4.5 ∧
  variance classA_visions = 0.136 ∧
  (let count := List.length classB_visions
   let total := count.choose 2
   let favorable := 3  -- (5.1, 4.5), (5.1, 4.9), (4.9, 4.5)
   7 / 10 = 1 - (favorable / total)) :=
by
  sorry

end vision_statistics_l618_618892


namespace angle_between_asymptotes_l618_618951

theorem angle_between_asymptotes
  (a : ℝ) 
  (focus_parabola : ℝ × ℝ := (-2, 0))
  (focus_hyperbola : ℝ × ℝ := (-a, 0))
  (h1 : focus_parabola = focus_hyperbola)
  (h2 : a^2 + 1 = 4)
  : real.arctan (real.sqrt 3) = real.pi / 3 :=
by
  sorry

end angle_between_asymptotes_l618_618951


namespace exponent_calculation_l618_618566

-- Define the necessary exponents and base
def base : ℕ := 19
def exp1 : ℕ := 11
def exp2 : ℕ := 8

-- Given condition
lemma power_property (a : ℕ) (m n : ℕ) : a^m / a^n = a^(m - n) := by sorry

-- Proof that 19^11 / 19^8 = 6859
theorem exponent_calculation : base^exp1 / base^exp2 = 6859 := by
  have : base^exp1 / base^exp2 = base^(exp1 - exp2) := power_property base exp1 exp2
  have : base^(exp1 - exp2) = base^3 := by rw [nat.sub_eq_iff_eq_add.mpr rfl]
  have : base^3 = 6859 := by -- This would be an arithmetic computation
    rfl -- Placeholder for the actual arithmetic; ideally, you'd verify this step.
  sorry

end exponent_calculation_l618_618566


namespace average_reduction_each_time_better_discount_option_l618_618172

noncomputable def average_percentage_reduction (p_initial p_final : ℝ) (n : ℕ) : ℝ :=
  1 - (p_final / p_initial).pow (1 / n)

theorem average_reduction_each_time (p_initial p_final : ℝ) (h1 : p_initial = 5) (h2 : p_final = 3.2) : 
  average_percentage_reduction p_initial p_final 2 = 0.2 :=
by
  rw [h1, h2]
  unfold average_percentage_reduction
  sorry

theorem better_discount_option (price_per_kg : ℝ) (qty_kg : ℕ) (discount_percentage : ℝ) (cash_discount_per_ton : ℝ) 
  (h1 : price_per_kg = 3.2) (h2 : qty_kg = 5000) (h3 : discount_percentage = 0.1) (h4 : cash_discount_per_ton = 200) : 
  (price_per_kg * (1 - discount_percentage) * qty_kg) < (price_per_kg * qty_kg - (cash_discount_per_ton * (qty_kg / 1000))) :=
by
  rw [h1, h2, h3, h4]
  sorry

end average_reduction_each_time_better_discount_option_l618_618172


namespace largest_initial_number_l618_618310

theorem largest_initial_number : ∃ n : ℕ, (n + 5 ∑ k : ℕ, k ≠ 0 ∧ ¬ (n % k = 0)) = 200 ∧ n = 189 :=
begin
  sorry
end

end largest_initial_number_l618_618310


namespace jamal_total_cost_l618_618761

-- Definitions based on conditions
def dozen := 12
def half_dozen := dozen / 2
def crayons_bought := 4 * half_dozen
def cost_per_crayon := 2
def total_cost := crayons_bought * cost_per_crayon

-- Proof statement (the question translated to a Lean theorem)
theorem jamal_total_cost : total_cost = 48 := by
  -- Proof skipped
  sorry

end jamal_total_cost_l618_618761


namespace sufficient_condition_inequality_l618_618535

theorem sufficient_condition_inequality (k : ℝ) :
  (k = 0 ∨ (-3 < k ∧ k < 0)) → ∀ x : ℝ, 2 * k * x^2 + k * x - 3 / 8 < 0 :=
sorry

end sufficient_condition_inequality_l618_618535


namespace tangent_line_and_parabola_equation_l618_618690

open Function

noncomputable def f (x : ℝ) : ℝ := real.sqrt x

theorem tangent_line_and_parabola_equation :
  (∃ (l : ℝ → ℝ), ∀ (x y : ℝ), y = l x ↔ x - 2*y + 1 = 0) ∧ 
  (∃ (p : ℝ), p = 1 / 2 ∧ (∀ (x y : ℝ), y = p ∧ x^2 = 4 * p * y ↔ x^2 = 2 * y)) :=
by
  sorry

end tangent_line_and_parabola_equation_l618_618690


namespace problem_solution_l618_618818

noncomputable def pairwise_sums : List ℕ := [5, 8, 9, 13, 14, 14, 15, 17, 18, 23]

def find_integers_and_product : Prop := 
  ∃ a b c d e : ℕ, 
  a + b + c + d + e = 34 ∧ 
  {a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e}.toFinset = pairwise_sums.toFinset ∧ 
  a * b * c * d * e = 4752

theorem problem_solution : find_integers_and_product := sorry

end problem_solution_l618_618818


namespace nine_point_circle_tangent_incircle_excircles_l618_618410

theorem nine_point_circle_tangent_incircle_excircles 
  (T : Triangle) 
  (nine_point_circle : Circle) 
  (incircle : Circle) 
  (excircle_A : Circle) 
  (excircle_B : Circle) 
  (excircle_C : Circle) : 
  touches nine_point_circle incircle ∧ 
  touches nine_point_circle excircle_A ∧ 
  touches nine_point_circle excircle_B ∧ 
  touches nine_point_circle excircle_C :=
sorry

end nine_point_circle_tangent_incircle_excircles_l618_618410


namespace option_a_equals_half_option_c_equals_half_l618_618550

theorem option_a_equals_half : 
  ( ∃ x : ℝ, x = (√2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180)) ∧ x = 1 / 2 ) := 
sorry

theorem option_c_equals_half : 
  ( ∃ y : ℝ, y = (Real.tan (22.5 * Real.pi / 180) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)) ∧ y = 1 / 2 ) := 
sorry

end option_a_equals_half_option_c_equals_half_l618_618550


namespace third_smallest_number_l618_618731

/-- 
  The third smallest two-decimal-digit number that can be made
  using the digits 3, 8, 2, and 7 each exactly once is 27.38.
-/
theorem third_smallest_number (digits : List ℕ) (h : digits = [3, 8, 2, 7]) : 
  ∃ x y, 
  x < y ∧
  x = 23.78 ∧
  y = 23.87 ∧
  ∀ z, z > x ∧ z < y → z = 27.38 :=
by 
  sorry

end third_smallest_number_l618_618731


namespace find_n_l618_618621

theorem find_n (n : ℕ) (h : n * n.factorial + 2 * n.factorial = 5040) : n = 5 :=
by {
  sorry
}

end find_n_l618_618621


namespace largest_initial_number_l618_618304

theorem largest_initial_number : ∃ n : ℕ, (n + 5 ∑ k : ℕ, k ≠ 0 ∧ ¬ (n % k = 0)) = 200 ∧ n = 189 :=
begin
  sorry
end

end largest_initial_number_l618_618304


namespace find_distance_city_A_B_l618_618355

-- Variables and givens
variable (D : ℝ)

-- Conditions from the problem
variable (JohnSpeed : ℝ := 40) (LewisSpeed : ℝ := 60)
variable (MeetDistance : ℝ := 160)
variable (TimeJohn : ℝ := (D - MeetDistance) / JohnSpeed)
variable (TimeLewis : ℝ := (D + MeetDistance) / LewisSpeed)

-- Lean 4 theorem statement for the proof
theorem find_distance_city_A_B :
  TimeJohn = TimeLewis → D = 800 :=
by
  sorry

end find_distance_city_A_B_l618_618355


namespace find_length_of_floor_l618_618530

noncomputable def length_of_floor (breadth : ℝ) : ℝ :=
  7 * breadth

noncomputable def area_of_floor (breadth : ℝ) : ℝ :=
  7 * breadth ^ 2

theorem find_length_of_floor (cost_per_sq_m : ℝ) (total_cost : ℝ) (h1 : cost_per_sq_m = 7) (h2 : total_cost = 2520) :
  ∃ length : ℝ, length ≈ 50.19 :=
by
  have h_area : area_of_floor (7 / √(51.4285714)) ≈ 360 :=
    sorry -- This should be deduced by the area calculation process.
  have h_length : length_of_floor (7 / √(51.4285714)) ≈ 50.19 :=
    sorry -- This should be deduced by direct computation from the area.
  use length_of_floor (7 / √(51.4285714))
  exact h_length

end find_length_of_floor_l618_618530


namespace find_n_of_lcm_gcf_l618_618052

open Nat

theorem find_n_of_lcm_gcf (n : ℕ) (h1 : lcm n 24 = 48) (h2 : gcd n 24 = 8) : n = 16 :=
by
  sorry

end find_n_of_lcm_gcf_l618_618052


namespace sales_worth_l618_618112

theorem sales_worth (S: ℝ) : 
  (1300 + 0.025 * (S - 4000) = 0.05 * S + 600) → S = 24000 :=
by
  sorry

end sales_worth_l618_618112


namespace coefficient_of_x4_in_expansion_l618_618178

theorem coefficient_of_x4_in_expansion :
  let expanded_expr := (2 - x) * (Polynomial.monomial 1 2 * x + 1)^6,
      term_x4 := expanded_expr.coeff 4
  in term_x4 = 320 := by
  sorry

end coefficient_of_x4_in_expansion_l618_618178


namespace difference_place_value_6_in_7669_l618_618501

theorem difference_place_value_6_in_7669 : 
  let numeral := 7669 
  in (let tens_place_value := 6 * 10 in
      let hundreds_place_value := 6 * 100 in
      hundreds_place_value - tens_place_value = 540) := 
by sorry

end difference_place_value_6_in_7669_l618_618501


namespace capacity_of_other_bottle_l618_618942

theorem capacity_of_other_bottle 
  (total_milk : ℕ) (capacity_bottle_one : ℕ) (fraction_filled_other_bottle : ℚ)
  (equal_fraction : ℚ) (other_bottle_milk : ℚ) (capacity_other_bottle : ℚ) : 
  total_milk = 8 ∧ capacity_bottle_one = 4 ∧ other_bottle_milk = 16/3 ∧ 
  (equal_fraction * capacity_bottle_one + equal_fraction * capacity_other_bottle = total_milk) ∧ 
  (fraction_filled_other_bottle = 5.333333333333333) → capacity_other_bottle = 8 :=
by
  intro h
  sorry

end capacity_of_other_bottle_l618_618942


namespace f_even_l618_618783

variable (g : ℝ → ℝ)

def is_odd (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = -g x

def f (x : ℝ) := |g (x^2)|

theorem f_even (h_g_odd : is_odd g) : ∀ x : ℝ, f g x = f g (-x) :=
by
  intro x
  -- Proof can be added here
  sorry

end f_even_l618_618783


namespace correct_statement_only_four_l618_618485

theorem correct_statement_only_four :
  (∃ (q : ℚ), q = 0) ∧
  (∀ (a b : ℝ), (a = -b) ∧ (b ≠ 0) → (a / b = -1)) ∧
  (∀ (x : ℝ), (|x| = x) → (x ≥ 0)) ∧
  (∀ (p q : ℚ), (p < q) → ∃ (r : ℚ), (p < r ∧ r < q)) →
  (④ is the only correct statement) :=
by
  sorry

end correct_statement_only_four_l618_618485


namespace kolya_mistaken_l618_618008

-- Definitions relating to the conditions
def at_least_four_blue_pencils (blue_pencils : ℕ) : Prop := blue_pencils >= 4
def at_least_five_green_pencils (green_pencils : ℕ) : Prop := green_pencils >= 5
def at_least_three_blue_pencils_and_four_green_pencils (blue_pencils green_pencils : ℕ) : Prop := blue_pencils >= 3 ∧ green_pencils >= 4
def at_least_four_blue_and_four_green_pencils (blue_pencils green_pencils : ℕ) : Prop := blue_pencils >= 4 ∧ green_pencils >= 4

-- Speaking truth conditions
variables (blue_pencils green_pencils : ℕ)
def vasya_true : Prop := at_least_four_blue_pencils blue_pencils
def kolya_true : Prop := at_least_five_green_pencils green_pencils
def petya_true : Prop := at_least_three_blue_pencils_and_four_green_pencils blue_pencils green_pencils
def misha_true : Prop := at_least_four_blue_and_four_green_pencils blue_pencils green_pencils

-- Given known information: three are true, one is false
def known_information : Prop := (vasya_true blue_pencils green_pencils ∧ petya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬kolya_true blue_pencils green_pencils)
                            ∨ (vasya_true blue_pencils green_pencils ∧ petya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ ¬misha_true blue_pencils green_pencils)
                            ∨ (vasya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬petya_true blue_pencils green_pencils)
                            ∨ (petya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬vasya_true blue_pencils green_pencils)

-- Theorem to be proved
theorem kolya_mistaken : known_information blue_pencils green_pencils → ¬kolya_true blue_pencils green_pencils :=
sorry

end kolya_mistaken_l618_618008


namespace who_made_mistake_l618_618013

-- Defining conditions for the colored pencils
def has_at_least_four_blue_pencils (b : Nat) : Prop := b >= 4
def has_at_least_five_green_pencils (g : Nat) : Prop := g >= 5
def has_at_least_three_blue_four_green_pencils (b g : Nat) : Prop := b >= 3 ∧ g >= 4
def has_at_least_four_blue_four_green_pencils (b g : Nat) : Prop := b >= 4 ∧ g >= 4

-- Statement of the problem
theorem who_made_mistake (b g : Nat) (vasya kolya petya misha : Prop) :
  has_at_least_four_blue_pencils b →
  has_at_least_five_green_pencils g →
  has_at_least_three_blue_four_green_pencils b g →
  has_at_least_four_blue_four_green_pencils b g →
  (∃ T : Set Prop, {vasya, kolya, petya, misha}.Erase T = {vasya, kolya, petya, misha} ∧
    T.Card = 3) →
  (kolya ↔ ¬ g >= 5) := 
sorry

end who_made_mistake_l618_618013


namespace largest_subset_size_l618_618988

theorem largest_subset_size :
  ∃ S : Set ℕ, S ⊆ {i | 1 ≤ i ∧ i ≤ 150} ∧ 
  (∀ (a b : ℕ), a ∈ S → b ∈ S → ¬ (a = 4 * b ∨ b = 4 * a)) ∧ 
  S.card = 141 :=
sorry

end largest_subset_size_l618_618988


namespace triangle_circle_geometry_l618_618543

variable {α : Type*}
variables (A B C O X Y : α)

-- Definitions of points and equilateral triangle
def is_equilateral_triangle (A B C : α) : Prop :=
  ∃ (d : ℝ), dist A B = d ∧ dist B C = d ∧ dist C A = d

-- Definitions of the conditions
axiom point_C_in_circle (h_eq_tri : is_equilateral_triangle A B C) 
  (h_circle: dist O A = dist O B) (C : α) : Prop

axiom points_on_circle (X Y onCircle: α) (h_AB_BX : dist A B = dist B X) : Prop

axiom point_C_on_chord (C XY : α) : Prop

-- The final statement to prove
theorem triangle_circle_geometry (h_eq_tri : is_equilateral_triangle A B C)
  (h_circle : point_C_in_circle h_eq_tri (dist O A = dist O B) C)
  (h_points_circle: points_on_circle X Y (dist A B = dist B X))
  (h_chord : point_C_on_chord C XY) :
  dist C Y = dist A O :=
sorry

end triangle_circle_geometry_l618_618543


namespace no_valid_x_l618_618240

-- Definitions based on given conditions
variables {m n x : ℝ}
variables (hm : m > 0) (hn : n < 0)

-- Theorem statement
theorem no_valid_x (hm : m > 0) (hn : n < 0) :
  ¬ ∃ x, (x - m)^2 - (x - n)^2 = (m - n)^2 :=
by
  sorry

end no_valid_x_l618_618240


namespace find_a_for_parabola_l618_618643

theorem find_a_for_parabola (a : ℝ) :
  (∃ y : ℝ, y = a * (-1 / 2)^2) → a = 1 / 2 :=
by
  sorry

end find_a_for_parabola_l618_618643


namespace men_in_first_group_l618_618277

theorem men_in_first_group (x : ℕ) :
  (20 * 48 = x * 80) → x = 12 :=
by
  intro h_eq
  have : x = (20 * 48) / 80 := sorry
  exact this

end men_in_first_group_l618_618277


namespace roots_abs_less_than_one_l618_618505

theorem roots_abs_less_than_one {a b : ℝ} 
    (h : |a| + |b| < 1) 
    (x1 x2 : ℝ) 
    (h_roots : x1 * x1 + a * x1 + b = 0) 
    (h_roots' : x2 * x2 + a * x2 + b = 0) 
    : |x1| < 1 ∧ |x2| < 1 := 
sorry

end roots_abs_less_than_one_l618_618505


namespace triangle_area_determinant_l618_618905

theorem triangle_area_determinant (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  let area := (1/2 : ℝ) * |x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)|
  in (x₁, y₁) = (1, 1) ∧ (x₂, y₂) = (1, 6) ∧ (x₃, y₃) = (8, 13) → area = 17.5 :=
by
  intros x₁ y₁ x₂ y₂ x₃ y₃ h
  -- rest of the proof skipped
  sorry

end triangle_area_determinant_l618_618905


namespace smallest_n_for_property_l618_618136

theorem smallest_n_for_property (n x : ℕ) (d : ℕ) (c : ℕ) 
  (hx : x = 10 * c + d) 
  (hx_prop : 10^(n-1) * d + c = 2 * x) :
  n = 18 := 
sorry

end smallest_n_for_property_l618_618136


namespace largest_subset_size_l618_618975

-- Definition of the subset condition
def valid_subset (s : Set ℕ) : Prop :=
  ∀ (a ∈ s) (b ∈ s), a ≠ 4 * b

-- Setting the range
def range_set : Set ℕ := { n | 1 ≤ n ∧ n ≤ 150 }

-- The main theorem statement
theorem largest_subset_size : ∃ (s : Set ℕ), s ⊆ range_set ∧ valid_subset (s) ∧ #(s) = 150 :=
by
  sorry

end largest_subset_size_l618_618975


namespace least_value_m_n_l618_618377

theorem least_value_m_n :
  ∃ m n : ℕ, (m > 0 ∧ n > 0) ∧
            (Nat.gcd (m + n) 231 = 1) ∧
            (n^n ∣ m^m) ∧
            ¬ (m % n = 0) ∧
            m + n = 377 :=
by 
  sorry

end least_value_m_n_l618_618377


namespace alpha_centauri_puzzle_l618_618933

open Nat

def max_number_count (A B N : ℕ) : ℕ :=
  let pairs_count := ((B - A) / N) * (N / 2)
  pairs_count + 1  -- Adding the single remainder

theorem alpha_centauri_puzzle :
  let A := 1353
  let B := 2134
  let N := 11
  max_number_count A B N = 356 :=
by 
  let A := 1353
  let B := 2134
  let N := 11
  -- Using the helper function max_number_count defined above
  have h : max_number_count A B N = 356 := 
   by sorry  -- skips detailed proof for illustrative purposes
  exact h


end alpha_centauri_puzzle_l618_618933


namespace frog_arrangement_l618_618885

theorem frog_arrangement :
  let n := 7 
  let g := 3 -- number of green frogs
  let r := 3 -- number of red frogs
  let b := 1 -- number of blue frog
  in
  n = g + r + b ∧ g = 3 ∧ r = 3 ∧ b = 1 →
  ∀ arrangements : Finset (List (Fin n)),
    (∀ arr ∈ arrangements, 
        (∀ i, 
          (arr[i] < g → arr[i+1] >= g) ∧ (arr[i] >= g + r → arr[i+1] < g + r + b)) →
        arrangements.card = 72
  sorry

end frog_arrangement_l618_618885


namespace largest_initial_number_l618_618335

-- Let's define the conditions and the result
def valid_addition (n a : ℕ) : Prop := ∃ k : ℕ, n = a * k + r ∧ 0 < r ∧ r < a

def valid_operations (initial : ℕ) (final : ℕ) (steps : ℕ → ℕ → ℕ) : Prop :=
  ∃ (a b c d e : ℕ), valid_addition initial a ∧
                      valid_addition (initial + a) b ∧
                      valid_addition (initial + a + b) c ∧
                      valid_addition (initial + a + b + c) d ∧
                      valid_addition (initial + a + b + c + d) e ∧
                      initial + a + b + c + d + e = final

theorem largest_initial_number :
  ∃ n : ℕ, (valid_operations n 200 (λn a, n + a)) ∧ (∀ m : ℕ, valid_operations m 200 (λn a, n + a) → m ≤ n) :=
sorry

end largest_initial_number_l618_618335


namespace largest_subset_size_l618_618969

def largest_subset_with_property (s : Set ℤ) : Prop :=
  ∀ (a b : ℤ), a ∈ s → b ∈ s → (a ≠ 4 * b ∧ b ≠ 4 * a)

theorem largest_subset_size : ∃ s : Set ℤ, (∀ x : ℤ, x ∈ s → 1 ≤ x ∧ x ≤ 150) ∧ largest_subset_with_property s ∧ s.card = 120 :=
by
  sorry

end largest_subset_size_l618_618969


namespace treasures_found_second_level_l618_618998

theorem treasures_found_second_level:
  ∀ (P T1 S T2 : ℕ), 
    P = 4 → 
    T1 = 6 → 
    S = 32 → 
    S = P * T1 + P * T2 → 
    T2 = 2 := 
by
  intros P T1 S T2 hP hT1 hS hTotal
  sorry

end treasures_found_second_level_l618_618998


namespace second_player_prevents_first_from_winning_l618_618400

theorem second_player_prevents_first_from_winning :
  ∃ strategy : (ℤ × ℤ → bool) → ℤ × ℤ → bool,
  ∀ (grid : ℤ × ℤ → option bool), ∀ (pos : ℤ × ℤ),
    (∃ n, n > 10 ∧ ∀ i ∈ list.range (n + 1), ∃ dir ∈ [{(1, 0), (0, 1), (1, 1), (-1, 1)}], 
       all_eq(grid, pos, dir, n, some ff)) → false :=
sorry

noncomputable def all_eq (grid : ℤ × ℤ → option bool) (pos : ℤ × ℤ) (dir : ℤ × ℤ) (n : ℤ) (s : option bool) : Prop :=
  ∀ i ∈ list.range (n + 1), grid (pos + i * dir) = s

end second_player_prevents_first_from_winning_l618_618400


namespace angles_satisfy_system_l618_618479

theorem angles_satisfy_system (k : ℤ) : 
  let x := Real.pi / 3 + k * Real.pi
  let y := k * Real.pi
  x - y = Real.pi / 3 ∧ Real.tan x - Real.tan y = Real.sqrt 3 := 
by 
  sorry

end angles_satisfy_system_l618_618479


namespace value_of_expression_l618_618476

theorem value_of_expression {a b : ℤ} (h₁ : a = -4) (h₂ : b = 3) : 
  -2 * a - b^3 + 2 * a * b + b^2 = -34 :=
by {
  rw [h₁, h₂],
  norm_num,
  sorry
}

end value_of_expression_l618_618476


namespace percent_of_z_l618_618730

variable {x y z : ℝ}

theorem percent_of_z (h₁ : x = 1.20 * y) (h₂ : y = 0.50 * z) : x = 0.60 * z :=
by
  sorry

end percent_of_z_l618_618730


namespace sqrt_198_bound_l618_618167

theorem sqrt_198_bound :
  14 < Real.sqrt 198 ∧ Real.sqrt 198 < 15 := 
by 
  have h1 : (14 : Real) = Real.sqrt 196, from Real.sqrt_eq_rfl.mp rfl,
  have h2 : (15 : Real) = Real.sqrt 225, from Real.sqrt_eq_rfl.mp rfl,
  rw [h1, h2],
  exact ⟨Real.sqrt_lt_sqrt Real.zero_lt_two Real.h'_lt_h₂, Real.sqrt_lt_sqrt Real.zero_lt_two Real.h'_lt_h₂⟩,
  sorry

end sqrt_198_bound_l618_618167


namespace prime_remainders_count_correct_l618_618707

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def prime_remainders (n : ℕ) : set ℕ :=
  { k | ∃ p, is_prime p ∧ 50 ≤ p ∧ p ≤ 100 ∧ p % 9 = k }

noncomputable def count_prime_remainders_between_50_and_100 : ℕ :=
  (prime_remainders 2 ∪ prime_remainders 3 ∪ prime_remainders 5 ∪ prime_remainders 7).card

theorem prime_remainders_count_correct :
  count_prime_remainders_between_50_and_100 = 5 :=
sorry

end prime_remainders_count_correct_l618_618707


namespace cannot_place_on_smaller_board_can_place_on_7x7_board_l618_618354

noncomputable theory

def smallest_square_board_battleship : ℕ :=
  let ship1 := 1 * 4,
      ship2 := 2 * (1 * 3),
      ship3 := 3 * (1 * 2),
      ship4 := 4 * (1 * 1),
      total_nodes := ship1 + ship2 + ship3 + ship4
  in 7

theorem cannot_place_on_smaller_board :
  ∀ (n : ℕ), n < 7 → n * n < smallest_square_board_battleship → false :=
by
  sorry

theorem can_place_on_7x7_board :
  smallest_square_board_battleship = 7 :=
by 
  sorry

end cannot_place_on_smaller_board_can_place_on_7x7_board_l618_618354


namespace max_initial_number_l618_618326

noncomputable def verify_addition (n x : ℕ) : Prop := 
  ∀ (a : ℕ), (a ∣ n) → (a ≠ 1) → (n + a = x) → False

theorem max_initial_number :
  ∃ (n : ℕ), 
  (∀ (a1 a2 a3 a4 a5 : ℕ), 
    verify_addition n a1 ∧ verify_addition (n + a1) a2 ∧
    verify_addition (n + a1 + a2) a3 ∧ verify_addition (n + a1 + a2 + a3) a4 ∧
    verify_addition (n + a1 + a2 + a3 + a4) a5 ∧
    (n + a1 + a2 + a3 + a4 + a5 = 200)) ∧
  (∀ m : ℕ, 
    (∃ (a1 a2 a3 a4 a5 : ℕ), 
      verify_addition m a1 ∧ verify_addition (m + a1) a2 ∧
      verify_addition (m + a1 + a2) a3 ∧ verify_addition (m + a1 + a2 + a3) a4 ∧
      verify_addition (m + a1 + a2 + a3 + a4) a5 ∧
      (m + a1 + a2 + a3 + a4 + a5 = 200)) →
    m ≤ 189)
: ∃ n, n = 189 := by
  sorry

end max_initial_number_l618_618326


namespace magnitude_of_a_plus_2b_l618_618704

noncomputable def a : ℝ × ℝ := (2, 0)
def b : ℝ × ℝ := (x, y)
axiom angle_between_a_b : (2 * x) + (0 * y) = 0
axiom magnitude_b : real.sqrt (x^2 + y^2) = 1

theorem magnitude_of_a_plus_2b :
  real.sqrt ((2 + 2 * x) ^ 2 + (2 * y) ^ 2) = 2 * real.sqrt 2 := by
  sorry

end magnitude_of_a_plus_2b_l618_618704


namespace max_factors_of_b_pow_n_l618_618881

theorem max_factors_of_b_pow_n (b n : ℕ) (h_b : b ≤ 15) (h_n : n ≤ 20) : 
  ∃ k, k = 861 ∧ (∀ m, m ≤ 15 → m^n = m → (factors_count (m^n) ≤ 861)) :=
sorry

noncomputable def factors_count (k : ℕ) : ℕ :=
  k.factors.prod (λ p e, e + 1)
-- The above definition needs a separate Lean library to accurately count the number of factors.

end max_factors_of_b_pow_n_l618_618881


namespace who_made_mistake_l618_618014

-- Defining conditions for the colored pencils
def has_at_least_four_blue_pencils (b : Nat) : Prop := b >= 4
def has_at_least_five_green_pencils (g : Nat) : Prop := g >= 5
def has_at_least_three_blue_four_green_pencils (b g : Nat) : Prop := b >= 3 ∧ g >= 4
def has_at_least_four_blue_four_green_pencils (b g : Nat) : Prop := b >= 4 ∧ g >= 4

-- Statement of the problem
theorem who_made_mistake (b g : Nat) (vasya kolya petya misha : Prop) :
  has_at_least_four_blue_pencils b →
  has_at_least_five_green_pencils g →
  has_at_least_three_blue_four_green_pencils b g →
  has_at_least_four_blue_four_green_pencils b g →
  (∃ T : Set Prop, {vasya, kolya, petya, misha}.Erase T = {vasya, kolya, petya, misha} ∧
    T.Card = 3) →
  (kolya ↔ ¬ g >= 5) := 
sorry

end who_made_mistake_l618_618014


namespace older_sister_age_l618_618043

theorem older_sister_age (x : ℕ) (older_sister_age : ℕ) (h1 : older_sister_age = 3 * x)
  (h2 : older_sister_age + 2 = 2 * (x + 2)) : older_sister_age = 6 :=
by
  sorry

end older_sister_age_l618_618043


namespace divide_example_conditions_l618_618705

variables (D q d : ℝ)

theorem divide_example_conditions (h1 : q = D / 5) (h2 : q = 7 * d) : 
  d = D / 35 ∧ q = D / 5 :=
by {
  split,
  { 
    -- from h1 and h2, q = D / 5 and q = 7 * d
    -- thus, D / 5 = 7 * d
    -- solving for d, we get d = D / 35
    calc 
      d = D / (7 * 5) : by rw [←h2, h1]
      ... = D / 35    : by ring,
  },
  { 
    -- from h1 we directly have q = D / 5
    exact h1,
  }
}

end divide_example_conditions_l618_618705


namespace expected_value_is_correct_l618_618944

-- Define probabilities for each outcome
def prob_odd := (1 : ℚ) / 3
def prob_2 := (1 : ℚ) / 9
def prob_4 := (1 : ℚ) / 18
def prob_6 := (1 : ℚ) / 9

-- Define monetary outcomes
def gain := 4 : ℚ
def loss := -6 : ℚ

-- Define expected monetary outcome
noncomputable def expected_value := 
  3 * (prob_odd * gain) + (prob_2 * loss) + (prob_4 * loss) + (prob_6 * loss)

-- Prove the expected value equals 7/3
theorem expected_value_is_correct : expected_value = 7 / 3 :=
by
  sorry

end expected_value_is_correct_l618_618944


namespace shaded_area_is_three_l618_618735

theorem shaded_area_is_three (π : ℝ) (hπ : 0 < π) :
  let AB := sqrt((8 + sqrt(64 - π^2)) / π)
  let BC := sqrt((8 - sqrt(64 - π^2)) / π)
  let area_rect := AB * BC
  let area_quarter_circle_AB := (π * AB^2) / 4
  let area_quarter_circle_BC := (π * BC^2) / 4
  let total_circular_area := area_quarter_circle_AB + area_quarter_circle_BC
  let shaded_area := total_circular_area - area_rect
  in shaded_area = 3 := sorry

end shaded_area_is_three_l618_618735


namespace count_three_digit_odd_numbers_with_even_tens_place_and_distinct_digits_l618_618265

theorem count_three_digit_odd_numbers_with_even_tens_place_and_distinct_digits : 
  (∑ n in finset.filter (λ n, 
    let d2 := (n / 10) % 10 in  -- tens place
    let d1 := (n / 100) % 10 in -- hundreds place
    let d0 := n % 10 in         -- units place
    (d0 % 2 = 1) ∧                -- units place is odd
    (d2 % 2 = 0) ∧                -- tens place is even
    (d1 ≠ d2) ∧ (d1 ≠ d0) ∧ (d2 ≠ d0) -- all digits are distinct
  ) (finset.Icc 100 999)) = 200 := sorry

end count_three_digit_odd_numbers_with_even_tens_place_and_distinct_digits_l618_618265


namespace line_circle_intersect_l618_618700

theorem line_circle_intersect (m : ℝ) :
  (∀ (A B : ℝ × ℝ), A ≠ B → (A.1 + A.2 + m = 0) ∧ (B.1 + B.2 + m = 0) →
   A.1^2 + A.2^2 = 4 ∧ B.1^2 + B.2^2 = 4 → 
   (|((0:ℝ), (0:ℝ)) + (A.1, A.2)| ≥ |A - B|)) →
  m ∈ Icc (-2*sqrt 2) (-2) ∪ Icc 2 (2*sqrt 2) :=
sorry

end line_circle_intersect_l618_618700


namespace inscribed_square_area_l618_618534

-- Define the conditions and the problem
theorem inscribed_square_area
  (side_length : ℝ)
  (square_area : ℝ) :
  side_length = 24 →
  square_area = 576 :=
by
  sorry

end inscribed_square_area_l618_618534


namespace num_integers_sum_j_consecutive_integers_l618_618706

theorem num_integers_sum_j_consecutive_integers :
  let is_valid_N (N : ℕ) : Prop :=
    N < 2000 ∧
    (∃! (j : ℕ), j ≥ 1 ∧
      ∃ k, N = j * (2 * k + j - 1) / 2)
  in ∃! (N : ℕ), is_valid_N N := sorry

end num_integers_sum_j_consecutive_integers_l618_618706


namespace chords_ratio_l618_618048

theorem chords_ratio (EQ GQ FQ HQ : ℝ) (h_eq : EQ = 5) (h_gq : GQ = 7) : EQ * FQ = GQ * HQ → FQ / HQ = 7 / 5 :=
by
  intros h_pow
  rw [h_eq, h_gq] at h_pow
  have h := congr_arg (λ x => x / HQ) h_pow
  rw [← mul_div_assoc, ← mul_div_assoc, div_eq_one_of_eq] at h, {
    assumption,
  },
  sorry

end chords_ratio_l618_618048


namespace geometric_sequence_property_l618_618751

variable {α : Type*} [LinearOrderedField α]

-- Conditions
axiom a1_a2 : α := 30
axiom a3_a4 : α := 60

-- Definitions for the geometric sequence terms
def a (n : ℕ) : α := a1_a2 * (q : α)^(n - 1)

theorem geometric_sequence_property 
  (a1_a2 : α) (a3_a4 : α) (h1 : a1_a2 = 30) (h2 : a3_a4 = 60)
  (q : α) (hq : q^2 = 2) :
  a(7) + a(8) = 240 :=
by
  sorry

end geometric_sequence_property_l618_618751


namespace max_initial_number_l618_618344

theorem max_initial_number (n : ℕ) : 
  (∃ (a b c d e : ℕ), 
    200 = n + a + b + c + d + e ∧ 
    ¬ (n % a = 0) ∧ 
    ¬ ((n + a) % b = 0) ∧ 
    ¬ ((n + a + b) % c = 0) ∧ 
    ¬ ((n + a + b + c) % d = 0) ∧ 
    ¬ ((n + a + b + c + d) % e = 0)) → 
  n ≤ 189 := 
sorry

end max_initial_number_l618_618344


namespace square_perimeter_l618_618513

theorem square_perimeter (A_total : ℕ) (A_common : ℕ) (A_circle : ℕ) 
  (H1 : A_total = 329)
  (H2 : A_common = 101)
  (H3 : A_circle = 234) :
  4 * (Int.sqrt (A_total - A_circle + A_common)) = 56 :=
by
  -- Since we are only required to provide the statement, we can skip the proof steps.
  -- sorry to skip the proof.
  sorry

end square_perimeter_l618_618513


namespace Kolya_mistake_l618_618020

def boys := ["Vasya", "Kolya", "Petya", "Misha"]

constant num_blue_pencils : ℕ
constant num_green_pencils : ℕ

axiom Vasya_statement : num_blue_pencils >= 4
axiom Kolya_statement : num_green_pencils >= 5
axiom Petya_statement : num_blue_pencils >= 3 ∧ num_green_pencils >= 4
axiom Misha_statement : num_blue_pencils >= 4 ∧ num_green_pencils >= 4

axiom three_truths_one_mistake : 
  (Vasya_statement ∨ ¬Vasya_statement) ∧
  (Kolya_statement ∨ ¬Kolya_statement) ∧
  (Petya_statement ∨ ¬Petya_statement) ∧
  (Misha_statement ∨ ¬Misha_statement) ∧
  ((Vasya_statement ? true : 1) + 
   (Kolya_statement ? true : 1) + 
   (Petya_statement ? true : 1) +
   (Misha_statement ? true : 1) == 3)

theorem Kolya_mistake : ¬Kolya_statement :=
by
  sorry

end Kolya_mistake_l618_618020


namespace train_speed_l618_618117

theorem train_speed (lt_train : ℝ) (lt_bridge : ℝ) (time_cross : ℝ) (total_speed_kmph : ℝ) :
  lt_train = 150 ∧ lt_bridge = 225 ∧ time_cross = 30 ∧ total_speed_kmph = (375 / 30) * 3.6 → 
  total_speed_kmph = 45 := 
by
  sorry

end train_speed_l618_618117


namespace probability_of_sum_at_least_10_l618_618478

def total_outcomes : ℕ := 36
def favorable_outcomes : ℕ := 6

theorem probability_of_sum_at_least_10 :
  (favorable_outcomes : ℝ) / (total_outcomes : ℝ) = 1 / 6 := by
  sorry

end probability_of_sum_at_least_10_l618_618478


namespace triangle_ABC_properties_l618_618239

-- Given conditions
def A : ℝ × ℝ := (5, 1)

def line_CM (p : ℝ × ℝ) : Prop := 2 * p.1 - p.2 - 5 = 0
def line_BH (p : ℝ × ℝ) : Prop := p.1 - 2 * p.2 - 5 = 0

-- Prove coordinates of vertex C
def C : ℝ × ℝ := (4, 3)

-- Define condition for vertex C
def is_vertex_C (p : ℝ × ℝ) : Prop :=
  2 * p.1 + p.2 - 11 = 0 ∧
  2 * p.1 - p.2 - 5 = 0

-- Prove length |AC|
def length_AC : ℝ := Real.sqrt 5

def equation_of_AC (p : ℝ × ℝ) : Prop :=
  (p.2 - 1) = -2 * (p.1 - 5)

-- Prove the equation of line BC
def B : ℝ × ℝ := (-1, -3)

def equation_of_BC (p : ℝ × ℝ) : Prop :=
  6 * p.1 - 5 * p.2 - 9 = 0

-- Lean theorem statement
theorem triangle_ABC_properties :
  is_vertex_C C ∧
  |AC| = length_AC ∧
  ∀ p : ℝ × ℝ, line_CM A ∧ line_CM B ∨ line_BH A ∧ line_BH B → equation_of_BC p :=
by
  sorry

end triangle_ABC_properties_l618_618239


namespace rabbit_hid_carrots_l618_618188

theorem rabbit_hid_carrots (h_r h_f : ℕ) (x : ℕ)
  (rabbit_holes : 5 * h_r = x) 
  (fox_holes : 7 * h_f = x)
  (holes_relation : h_r = h_f + 6) :
  x = 105 :=
by
  sorry

end rabbit_hid_carrots_l618_618188


namespace police_catches_thief_in_two_hours_l618_618494

noncomputable def time_to_catch (speed_thief speed_police distance_police_start lead_time : ℝ) : ℝ :=
  let distance_thief := speed_thief * lead_time
  let initial_distance := distance_police_start - distance_thief
  let relative_speed := speed_police - speed_thief
  initial_distance / relative_speed

theorem police_catches_thief_in_two_hours :
  time_to_catch 20 40 60 1 = 2 := by
  sorry

end police_catches_thief_in_two_hours_l618_618494


namespace dvd_pack_price_after_discount_l618_618613

def dvd_pack_cost : ℕ := 76
def discount : ℕ := 25
def final_cost : ℕ := dvd_pack_cost - discount

theorem dvd_pack_price_after_discount : final_cost = 51 :=
by
  -- Calculation
  rw [final_cost, dvd_pack_cost, discount]
  -- Perform the arithmetic
  norm_num
-- The proof will finally end here

end dvd_pack_price_after_discount_l618_618613


namespace luna_badges_correct_l618_618264

-- conditions
def total_badges : ℕ := 83
def hermione_badges : ℕ := 14
def celestia_badges : ℕ := 52

-- question and answer
theorem luna_badges_correct : total_badges - (hermione_badges + celestia_badges) = 17 :=
by
  sorry

end luna_badges_correct_l618_618264


namespace sum_of_sequence_l618_618154

theorem sum_of_sequence :
  let seq := λ (n : ℕ), n - (1 / n)
  (∑ n in Finset.range 9 \ 1, seq (n + 2)) = 52.358 :=
by
  let seq := λ (n : ℕ), n - (1 / n)
  have h_seq_2 : Real := seq 2
  have h_seq_3 : Real := seq 3
  have h_seq_4 : Real := seq 4
  have h_seq_5 : Real := seq 5
  have h_seq_6 : Real := seq 6
  have h_seq_7 : Real := seq 7
  have h_seq_8 : Real := seq 8
  have h_seq_9 : Real := seq 9
  have h_seq_10 : Real := seq 10
  sorry

end sum_of_sequence_l618_618154


namespace weakly_increasing_f_g_h_min_value_and_weakly_increasing_l618_618727

-- Question 1: Prove whether f(x) = x + 4 is a "weakly increasing function" in (1,2) and g(x) = x^2 + 4x + 2 is not.
theorem weakly_increasing_f_g :
  (∀ x ∈ set.Ioo 1 2, deriv (λ x, x + 4) x > 0) ∧ (∀ x ∈ set.Ioo 1 2, deriv (λ x, x + 4) x > 0 → deriv (λ x, 1 + 4 / x) x < 0) →
  (∀ x ∈ set.Ioo 1 2, ¬(deriv (λ x, x^2 + 4 * x + 2) x ≤ 0) ∧ ¬(deriv (λ x, x + 4 + 2 / x) x ≤ 0)) :=
sorry

-- Question 2: Minimum value of h(x) in [0, 1/4] and conditions for it to be weakly increasing in (0,1]
theorem h_min_value_and_weakly_increasing (θ b : ℝ) (h : ∀ x, x^2 + (sin θ - 1/2) * x + b)
  (h_range : θ ∈ set.Icc 0 (π / 2)) (x_range : ∀ x, x ∈ set.Icc 0 (1 / 4)) :
  (∃ θ ∈ set.Icc 0 (π / 2), ∀ x ∈ set.Icc 0 (1 / 4), h x ≥ b) ∧ 
  (∀ x ∈ set.Icc (0 : ℝ) 1, h x ^.deriv ≥ 0 ∧ (h x / x).^deriv ≤ 0 → θ ∈ set.Icc (2 * 0 * π + π / 6) (2 * 0 * π + 5 * π / 6) ∧ b ≥ 1) :=
sorry

end weakly_increasing_f_g_h_min_value_and_weakly_increasing_l618_618727


namespace possible_combinations_l618_618769

noncomputable def dark_chocolate_price : ℝ := 5
noncomputable def milk_chocolate_price : ℝ := 4.50
noncomputable def white_chocolate_price : ℝ := 6
noncomputable def sales_tax_rate : ℝ := 0.07
noncomputable def leonardo_money : ℝ := 4 + 0.59

noncomputable def total_money := leonardo_money

noncomputable def dark_chocolate_with_tax := dark_chocolate_price * (1 + sales_tax_rate)
noncomputable def milk_chocolate_with_tax := milk_chocolate_price * (1 + sales_tax_rate)
noncomputable def white_chocolate_with_tax := white_chocolate_price * (1 + sales_tax_rate)

theorem possible_combinations :
  total_money = 4.59 ∧ (total_money >= 0 ∧ total_money < dark_chocolate_with_tax ∧ total_money < white_chocolate_with_tax ∧
  total_money ≥ milk_chocolate_with_tax ∧ milk_chocolate_with_tax = 4.82) :=
by
  sorry

end possible_combinations_l618_618769


namespace tiger_initial_leaps_behind_l618_618537

theorem tiger_initial_leaps_behind (tiger_leap_distance deer_leap_distance tiger_leaps_per_minute deer_leaps_per_minute total_distance_to_catch initial_leaps_behind : ℕ) 
  (h1 : tiger_leap_distance = 8) 
  (h2 : deer_leap_distance = 5) 
  (h3 : tiger_leaps_per_minute = 5) 
  (h4 : deer_leaps_per_minute = 4) 
  (h5 : total_distance_to_catch = 800) :
  initial_leaps_behind = 40 := 
by
  -- Leaving proof body incomplete as it is not required
  sorry

end tiger_initial_leaps_behind_l618_618537


namespace EquationD_is_quadratic_l618_618482

variable (x a b c y : ℝ)

-- Definition of equations
def EquationA := x - (1 / x) + 2 = 0
def EquationB := x^2 + 2 * x + y = 0
def EquationC := a * x^2 + b * x + c = 0
def EquationD := x^2 - x + 1 = 0

-- Statement that Equation D is a quadratic equation
theorem EquationD_is_quadratic : EquationD := 
sorry

end EquationD_is_quadratic_l618_618482


namespace largest_initial_number_l618_618336

-- Let's define the conditions and the result
def valid_addition (n a : ℕ) : Prop := ∃ k : ℕ, n = a * k + r ∧ 0 < r ∧ r < a

def valid_operations (initial : ℕ) (final : ℕ) (steps : ℕ → ℕ → ℕ) : Prop :=
  ∃ (a b c d e : ℕ), valid_addition initial a ∧
                      valid_addition (initial + a) b ∧
                      valid_addition (initial + a + b) c ∧
                      valid_addition (initial + a + b + c) d ∧
                      valid_addition (initial + a + b + c + d) e ∧
                      initial + a + b + c + d + e = final

theorem largest_initial_number :
  ∃ n : ℕ, (valid_operations n 200 (λn a, n + a)) ∧ (∀ m : ℕ, valid_operations m 200 (λn a, n + a) → m ≤ n) :=
sorry

end largest_initial_number_l618_618336


namespace red_grapes_in_salad_l618_618290

theorem red_grapes_in_salad {G R B : ℕ} 
  (h1 : R = 3 * G + 7)
  (h2 : B = G - 5)
  (h3 : G + R + B = 102) : R = 67 :=
sorry

end red_grapes_in_salad_l618_618290


namespace limit_is_exp_l618_618588

noncomputable def limit_expression (α β : ℝ) :=
  (λ x : ℝ, (1 + (Real.sin x) * (Real.cos (α * x))) / (1 + (Real.sin x) * (Real.cos (β * x))) ) ** (Nat.cubic (Real.cot x))

theorem limit_is_exp (α β : ℝ) : 
  Filter.Tendsto (λ x : ℝ, (limit_expression α β x)) (Filter.nhds_within 0 set.univ) (𝓝 (Real.exp ((β^2 - α^2) / 2))) :=
  sorry

end limit_is_exp_l618_618588


namespace least_value_m_n_l618_618376

theorem least_value_m_n :
  ∃ m n : ℕ, (m > 0 ∧ n > 0) ∧
            (Nat.gcd (m + n) 231 = 1) ∧
            (n^n ∣ m^m) ∧
            ¬ (m % n = 0) ∧
            m + n = 377 :=
by 
  sorry

end least_value_m_n_l618_618376


namespace unique_k_value_l618_618273

noncomputable def findK (k : ℝ) : Prop :=
  ∃ (x : ℝ), (x^2 - k) * (x + k + 1) = x^3 + k * (x^2 - x - 4) ∧ k ≠ 0 ∧ k = -3

theorem unique_k_value : ∀ (k : ℝ), findK k :=
by
  intro k
  sorry

end unique_k_value_l618_618273


namespace locus_of_Q_is_ellipse_l618_618688

noncomputable def ellipse (x y : ℝ) := (x^2) / 24 + (y^2) / 16 = 1
noncomputable def line (x y : ℝ) := x / 12 + y / 8 = 1

theorem locus_of_Q_is_ellipse :
  (∀ (x y : ℝ),
    ∃ (P : ℝ × ℝ) (R : ℝ × ℝ) (Q : ℝ × ℝ),
      (line P.1 P.2) ∧
      (∃ (θ ρ_1 ρ_2 : ℝ), (P.1 = ρ_2 * cos θ ∧ P.2 = ρ_2 * sin θ) ∧
      (ellipse R.1 R.2) ∧ 
      (R.1 = ρ_1 * cos θ ∧ R.2 = ρ_1 * sin θ) ∧
      (let ρ := (ρ_1^2) / ρ_2 in Q.1 = ρ * cos θ ∧ Q.2 = ρ * sin θ) ∧
      |Q.1 + Q.2| * |P.1 + P.2| = |R.1 + R.2|^2)) →
  ∃ (x y : ℝ), (2 * (x - 1)^2 / (5 / 2)) + (3 * (y - 1)^2 / (5 / 3)) = 1 :=
sorry

end locus_of_Q_is_ellipse_l618_618688


namespace original_cube_volume_l618_618049

variable (a : ℝ)

def original_volume : ℝ := a^3
def new_volume : ℝ := (a + 2) * (a + 2) * (a - 2)
def volume_difference_condition : Prop := new_volume = original_volume - 16

theorem original_cube_volume
  (h : volume_difference_condition a) :
  original_volume a = 9 + 12 * Real.sqrt 5 :=
sorry

end original_cube_volume_l618_618049


namespace no_isosceles_triangle_exists_l618_618590

-- Define the grid size
def grid_size : ℕ := 5

-- Define points A and B such that AB is three units horizontally
structure Point where
  x : ℕ
  y : ℕ

-- Define specific points A and B
def A : Point := ⟨2, 2⟩
def B : Point := ⟨5, 2⟩

-- Define a function to check if a triangle is isosceles
def is_isosceles (p1 p2 p3 : Point) : Prop :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 = (p1.x - p3.x)^2 + (p1.y - p3.y)^2 ∨
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 = (p2.x - p3.x)^2 + (p2.y - p3.y)^2 ∨
  (p1.x - p3.x)^2 + (p1.y - p3.y)^2 = (p2.x - p3.x)^2 + (p2.y - p3.y)^2

-- Prove that there are no points C that make triangle ABC isosceles
theorem no_isosceles_triangle_exists :
  ¬ ∃ C : Point, C.x ≤ grid_size ∧ C.y ≤ grid_size ∧ is_isosceles A B C :=
by
  sorry

end no_isosceles_triangle_exists_l618_618590


namespace g_g_g_g_2_eq_1406_l618_618385

def g (x : ℕ) : ℕ :=
if x % 3 = 0 then x / 3 else 5 * x + 1

theorem g_g_g_g_2_eq_1406 : g (g (g (g 2))) = 1406 := by
  sorry

end g_g_g_g_2_eq_1406_l618_618385


namespace symmetric_points_x_axis_l618_618276

theorem symmetric_points_x_axis (m n : ℤ) (h1 : m + 1 = 1) (h2 : 3 = -(n - 2)) : m - n = 1 :=
by
  sorry

end symmetric_points_x_axis_l618_618276


namespace teamB_wins_second_game_l618_618419

noncomputable def prob_win_second_game (first_teamB : Prop) 
(series_winner_A : Prop) 
(last_game_fifth : Prop) 
(until_final_no_consecutive_wins_A : Prop)
(team_A_win_three : Prop) 
(team_B_win_less_three : Prop): ℝ :=
if first_teamB ∧ series_winner_A ∧ last_game_fifth ∧ until_final_no_consecutive_wins_A ∧ team_A_win_three ∧ team_B_win_less_three then 1 else 0

theorem teamB_wins_second_game (h1 : first_teamB = True) 
(h2 : series_winner_A = True) 
(h3 : last_game_fifth = True) 
(h4 : until_final_no_consecutive_wins_A = True)
(h5 : team_A_win_three = True) 
(h6 : team_B_win_less_three = True) : 
  prob_win_second_game first_teamB series_winner_A last_game_fifth until_final_no_consecutive_wins_A team_A_win_three team_B_win_less_three = 1 :=
by
  sorry

end teamB_wins_second_game_l618_618419


namespace train_speed_calculation_l618_618114

noncomputable def speed_of_train_in_kmph
  (length_of_train : ℝ)
  (length_of_bridge : ℝ)
  (time_to_cross_bridge : ℝ) : ℝ :=
(length_of_train + length_of_bridge) / time_to_cross_bridge * 3.6

theorem train_speed_calculation
  (length_of_train : ℝ)
  (length_of_bridge : ℝ)
  (time_to_cross_bridge : ℝ)
  (h_train : length_of_train = 150)
  (h_bridge : length_of_bridge = 225)
  (h_time : time_to_cross_bridge = 30) :
  speed_of_train_in_kmph length_of_train length_of_bridge time_to_cross_bridge = 45 := by
  simp [speed_of_train_in_kmph, h_train, h_bridge, h_time]
  norm_num
  sorry

end train_speed_calculation_l618_618114


namespace derivative_of_y_l618_618628

noncomputable def y (x : ℝ) :=
  (real.cot 2) ^ (1 / 5 : ℝ) * (cos (18 * x)) ^ 2 / (36 * sin (36 * x))

noncomputable def y' (x : ℝ) :=
  -((real.cot 2) ^ (1 / 5 : ℝ)) / (4 * (sin (18 * x)) ^ 2)

theorem derivative_of_y (x : ℝ) :
  deriv y x = y' x :=
sorry

end derivative_of_y_l618_618628


namespace OM_perp_OH_l618_618858

-- Given conditions
variables {O A B C M H : Type*} [metric_space O]

-- The circumcircle of triangle ABC is \odot O
variable (circumcircle : ∀ {X : Type*} [metric_space X], ∀ {A B C : X}, metric.ball O _ = metric.ball O _)

-- Angle C
variable (angleC : ∠ B C A = 60)

-- M is the midpoint of the arc AB on the circumcircle \odot O.
axiom midpoint_M {X : Type*} [metric_space X] {A B : X} : segment.midpoint O A B = M

-- H is the orthocenter of \triangle ABC.
axiom orthocenter_H {X : Type*} [metric_space X] {A B C : X} : orthocenter O A B C = H

-- Prove OM ⊥ OH
theorem OM_perp_OH : orthogonal (vector_to_point O M) (vector_to_point O H) :=
sorry

end OM_perp_OH_l618_618858


namespace first_house_gets_90_bottles_l618_618123

def bottles_of_drinks (total_bottles bottles_cider_only bottles_beer_only : ℕ) : ℕ :=
  let bottles_mixture := total_bottles - bottles_cider_only - bottles_beer_only
  let first_house_cider := bottles_cider_only / 2
  let first_house_beer := bottles_beer_only / 2
  let first_house_mixture := bottles_mixture / 2
  first_house_cider + first_house_beer + first_house_mixture
  
theorem first_house_gets_90_bottles :
  bottles_of_drinks 180 40 80 = 90 :=
by
  rw [bottles_of_drinks]
  sorry

end first_house_gets_90_bottles_l618_618123


namespace find_daps_from_dips_l618_618721

-- Definitions based on the conditions
def daps := ℝ
def dops := ℝ
def dips := ℝ

-- Conditions given in the problem
def cond1 (daps_eq : daps) (dops_eq : dops) : Prop := 6 * daps_eq = 5 * dops_eq
def cond2 (dops_eq : dops) (dips_eq : dips) : Prop := 3 * dops_eq = 10 * dips_eq

-- Find x (number of daps that are equivalent to 60 dips)
theorem find_daps_from_dips : ∃ x : daps, (∀ daps_eq dops_eq dips_eq, cond1 daps_eq dops_eq ∧ cond2 dops_eq dips_eq → 60 * dips_eq = x * daps_eq) → x = 21.6 :=
by
  sorry

end find_daps_from_dips_l618_618721


namespace sin_double_angle_l618_618236

theorem sin_double_angle (θ : ℝ) (h : ∃ (θ : ℝ), tan θ = 3) : sin (2 * θ) = 3 / 5 :=
by
  sorry

end sin_double_angle_l618_618236


namespace find_n_l618_618594

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem find_n (x a : ℝ) (h1 : binomial ?n 2 * x^(?n - 2) * a^2 = 210)
    (h2 : binomial ?n 3 * x^(?n - 3) * a^3 = 504)
    (h3 : binomial ?n 4 * x^(?n - 4) * a^4 = 1260) : ?n = 7 :=
  sorry

end find_n_l618_618594


namespace octahedron_intersection_hex_area_l618_618111

noncomputable def hexagon_area (a b c : ℕ) (side_length : ℝ) (height : ℝ) : Prop :=
  height = (side_length * real.sqrt 3 / 2) ∧
  (2 * height / 3) = (2 * real.sqrt 3 / 3) ∧
  (side_length * (2 / 3)) = (4 / 3) ∧
  (3 * real.sqrt 3 / 2 * (4 / 3)^2 / a = 3) ∧
  b = 3 ∧
  c = 3

theorem octahedron_intersection_hex_area (side_length : ℝ) (height : ℝ) (a b c : ℕ) 
  (h₁ : side_length = 2)
  (h₂ : a = 8)
  (h₃ : b = 3)
  (h₄ : c = 3):
  hexagon_area a b c side_length height ↔ ((8 * real.sqrt 3) / 3 = (8 * real.sqrt 3) / 3) :=
by
  rw h₂
  rw h₃
  rw h₄
  exact sorry

end octahedron_intersection_hex_area_l618_618111


namespace border_area_is_72_l618_618956

def livingRoomLength : ℝ := 12
def livingRoomWidth : ℝ := 10
def borderWidth : ℝ := 2

def livingRoomArea : ℝ := livingRoomLength * livingRoomWidth
def carpetLength : ℝ := livingRoomLength - 2 * borderWidth
def carpetWidth : ℝ := livingRoomWidth - 2 * borderWidth
def carpetArea : ℝ := carpetLength * carpetWidth
def borderArea : ℝ := livingRoomArea - carpetArea

theorem border_area_is_72 : borderArea = 72 := 
by
  sorry

end border_area_is_72_l618_618956


namespace expression_a_equals_half_expression_c_equals_half_l618_618548

theorem expression_a_equals_half :
  (A : ℝ) = (1 / 2) :=
by
  let A := (Real.sqrt 2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180))
  sorry

theorem expression_c_equals_half :
  (C : ℝ) = (1 / 2) :=
by
  let C := (Real.tan (22.5 * Real.pi / 180)) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)
  sorry

end expression_a_equals_half_expression_c_equals_half_l618_618548


namespace lowest_possible_price_l618_618107

theorem lowest_possible_price (MSRP : ℝ) (discount1 : ℝ) (discount2 : ℝ) (final_price : ℝ) 
  (h_MSRP : MSRP = 35) (h_discount1 : discount1 = 0.30) (h_discount2 : discount2 = 0.20) 
  (h_final_price : final_price = MSRP * (1 - discount1) * (1 - discount2)) : 
  final_price = 19.60 := 
by {
  -- Identifying the intermediate discounted prices
  let discounted_price1 := MSRP * (1 - discount1),
  let additional_discount := discounted_price1 * discount2,
  let final_discounted_price := discounted_price1 - additional_discount,

  -- We can derive the final price directly
  calc
    final_price = MSRP * (1 - discount1) * (1 - discount2) : h_final_price
             ... = 35.00 * (1 - 0.30) * (1 - 0.20) : by { congr; assumption }
             ... = 35.00 * 0.70 * 0.80 : by norm_num
             ... = 19.60 : by norm_num,
}

end lowest_possible_price_l618_618107


namespace problem_l618_618247

def f (x : ℝ) : ℝ := min (-x^2) (x - 2)

theorem problem (
  (x : ℝ) 
  (hx0_eq : f 0 = -2)
  (hx4_eq : f 4 = -16)
  (hx_sol : ∀ (x : ℝ), f x > -4 ↔ -2 < x ∧ x < 2)) : ∃ x, f x =? :=
by
  -- Proofs would go here
  sorry

end problem_l618_618247


namespace alpha_parallel_beta_suff_but_not_nec_for_beta_perp_gamma_l618_618222

variable (α β γ : Plane) -- where Plane is understood to be some geometric type representing a plane

-- Condition: α ⟂ γ
axiom alpha_perp_gamma : α ⟂ γ

-- Proof problem statement
theorem alpha_parallel_beta_suff_but_not_nec_for_beta_perp_gamma (h : α ∥ β) : 
  ∀ (h1 : β ⟂ γ), (α ∥ β) → (β ⟂ γ) ∧ ¬((β ⟂ γ) → (α ∥ β)) := 
sorry

end alpha_parallel_beta_suff_but_not_nec_for_beta_perp_gamma_l618_618222


namespace binomial_sum_l618_618605

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_sum (n : ℤ) (h1 : binomial 25 n.natAbs + binomial 25 12 = binomial 26 13 ∧ n ≥ 0) : 
    (n = 12 ∨ n = 13) → n.succ + n = 25 := 
    sorry

end binomial_sum_l618_618605


namespace contest_end_time_l618_618517

def start_time : ℕ := 15 * 60 -- 3:00 p.m. in minutes from midnight
def duration : ℕ := 765 -- duration of the contest in minutes

theorem contest_end_time : start_time + duration = 3 * 60 + 45 := by
  -- start_time is 15 * 60 (3:00 p.m. in minutes)
  -- duration is 765 minutes
  -- end_time should be 3:45 a.m. which is 3 * 60 + 45 minutes from midnight
  sorry

end contest_end_time_l618_618517


namespace product_abml_l618_618774

open Real

theorem product_abml (A B M L : ℝ) (h1 : log 10 (A * M) + log 10 (B * M) = 3)
  (h2 : log 10 (M * L) + log 10 (B * L) = 4)
  (h3 : log 10 (A * L) + log 10 (A * B) = 5)
  (h_pos : 0 < A ∧ 0 < B ∧ 0 < M ∧ 0 < L) : A * B * M * L = 10^4 :=
by
  sorry

end product_abml_l618_618774


namespace log10_sum_correct_l618_618562

noncomputable def log10_sum : ℝ :=
  (Real.log 16 / Real.log 10)
  + 3 * (Real.log 5 / Real.log 10)
  + (1 / 2) * (Real.log 64 / Real.log 10)
  + 4 * (Real.log 5 / Real.log 10)
  + (Real.log 32 / Real.log 10)

theorem log10_sum_correct : log10_sum = 8.505 := 
by
  have log_identity1 : ∀ {b x : ℝ}, x > 0 → Real.log b * Real.log x = Real.log (x ^ Real.log b) :=
    sorry,
  have log_identity2 : ∀ {b x y : ℝ}, x > 0 → y > 0 → Real.log x / Real.log b + Real.log y / Real.log b = Real.log (x * y) / Real.log b :=
    sorry,
  sorry

end log10_sum_correct_l618_618562


namespace four_digit_number_difference_l618_618208

theorem four_digit_number_difference
    (digits : List ℕ) (h_digits : digits = [2, 0, 1, 3, 1, 2, 2, 1, 0, 8, 4, 0])
    (max_val : ℕ) (h_max_val : max_val = 3840)
    (min_val : ℕ) (h_min_val : min_val = 1040) :
    max_val - min_val = 2800 :=
by
    sorry

end four_digit_number_difference_l618_618208


namespace largest_initial_number_l618_618311

theorem largest_initial_number (n : ℕ) (h : (∃ a b c d e : ℕ, n ≠ 0 ∧ n + a + b + c + d + e = 200 
                                              ∧ n % a ≠ 0 ∧ n % b ≠ 0 ∧ n % c ≠ 0 ∧ n % d ≠ 0 ∧ n % e ≠ 0)) 
: n ≤ 189 :=
sorry

end largest_initial_number_l618_618311


namespace radius_of_tangents_circles_l618_618467

noncomputable def calculate_radius (A B P Q R S : Type) [metric_space A] [metric_space B] [metric_space P] [metric_space Q] [metric_space R] [metric_space S] (r : ℝ) : ℝ :=
  if (dist A B = 2) ∧ 
     (dist A P = 1 + r) ∧ (dist B P = 1 + r) ∧ (dist B Q = 1 + r) ∧ 
     (dist P Q = 2 * r) ∧ 
     (dist P R = 1 + r) ∧ (dist R S = 2 * r) -- and other conditions needed to fully characterize the problem
  then r else 0

theorem radius_of_tangents_circles 
  {A B P Q R S : Type} [metric_space A] [metric_space B] [metric_space P] [metric_space Q] [metric_space R] [metric_space S]
  (r : ℝ) 
  (h1 : dist A B = 2) 
  (h2 : dist A P = 1 + r) 
  (h3 : dist B P = 1 + r) 
  (h4 : dist B Q = 1 + r) 
  (h5 : dist P Q = 2 * r) 
  (h6 : dist P S = 1 + r) 
  (h7 : dist R S = 2 * r) -- and other conditions needed
  : calculate_radius A B P Q R S r = 2 := 
sorry

end radius_of_tangents_circles_l618_618467


namespace expected_parts_within_range_is_997_l618_618053

open ProbabilityTheory

noncomputable def normal_expected_parts_within_range (mu sigma : ℝ) : ℝ :=
  let range := (mu - 3 * sigma, mu + 3 * sigma)
  ∫ x in set.Icc (mu - 3 * sigma) (mu + 3 * sigma), NormalDist.pdf mu sigma x

theorem expected_parts_within_range_is_997 (mu sigma : ℝ) :
  normal_expected_parts_within_range mu sigma * 1000 = 997 :=
by
  sorry

end expected_parts_within_range_is_997_l618_618053


namespace complex_modulus_problem_l618_618686

open Complex

theorem complex_modulus_problem {z : ℂ} (h : (2 - I) * z = 4 + 3 * I) : abs(z - I) = Real.sqrt 2 :=
by
  sorry

end complex_modulus_problem_l618_618686


namespace largest_subset_no_four_times_another_l618_618966

theorem largest_subset_no_four_times_another :
  ∃ (S : set ℕ), S ⊆ {1, 2, ..., 150} ∧ (∀ (a b : ℕ), a ∈ S → b ∈ S → (a ≠ 4 * b ∧ b ≠ 4 * a)) ∧ (S.card = 141) :=
sorry

end largest_subset_no_four_times_another_l618_618966


namespace max_spherical_segment_volume_l618_618880

theorem max_spherical_segment_volume (S : ℝ) (h : ℝ):
  (V = 1/2 * S * h - 1/3 * π * h^3) → 
  (2 * h^2 = S / π) → 
  V ≤ S / 3 * sqrt(S / (2 * π)) :=
by
  sorry

end max_spherical_segment_volume_l618_618880


namespace ceo_salary_calculation_l618_618884

def total_salary (salaries : list ℕ) := salaries.sum

variable (employees : list ℕ)
variable (ceo_salary : ℕ)

noncomputable def average_salary (total_salary : ℕ) (num_employees : ℕ) := 
  total_salary / num_employees

theorem ceo_salary_calculation (h1 : employees.length = 50)
  (h2 : ∀ (s : ℕ), s ∈ employees → 1500 ≤ s ∧ s ≤ 5500)
  (h3 : average_salary (total_salary employees) 50 = 3000)
  (h4 : average_salary (total_salary employees + ceo_salary) 51 = 3500)
  (h5 : ∀ (s : ℕ), s ∈ employees → s < ceo_salary) :
  ceo_salary = 28500 :=
by
  sorry

end ceo_salary_calculation_l618_618884


namespace find_value_of_n_l618_618665

def random_variable_probability (X : ℕ → Prop) (n : ℕ) : Prop :=
  ∀ k, X k ↔ (1 ≤ k ∧ k ≤ n)

def probability_X_less_than_4 (n : ℕ) : Prop :=
  (3 : ℝ) / (n : ℝ) = 0.3

theorem find_value_of_n (n : ℕ) 
  (h1 : random_variable_probability (λ k, k ≤ n) n)
  (h2 : probability_X_less_than_4 n) : n = 10 :=
sorry

end find_value_of_n_l618_618665


namespace temperature_on_Saturday_l618_618452

theorem temperature_on_Saturday 
  (avg_temp : ℕ)
  (sun_temp : ℕ) 
  (mon_temp : ℕ) 
  (tue_temp : ℕ) 
  (wed_temp : ℕ) 
  (thu_temp : ℕ) 
  (fri_temp : ℕ)
  (saturday_temp : ℕ)
  (h_avg : avg_temp = 53)
  (h_sun : sun_temp = 40)
  (h_mon : mon_temp = 50) 
  (h_tue : tue_temp = 65) 
  (h_wed : wed_temp = 36) 
  (h_thu : thu_temp = 82) 
  (h_fri : fri_temp = 72) 
  (h_week : 7 * avg_temp = sun_temp + mon_temp + tue_temp + wed_temp + thu_temp + fri_temp + saturday_temp) :
  saturday_temp = 26 := 
by
  sorry

end temperature_on_Saturday_l618_618452


namespace determine_x_l618_618163

theorem determine_x :
  (∑ n in Finset.range 2023 + 1, n * (2024 - n)) = 2023 * 1012 * 675 :=
by
  sorry

end determine_x_l618_618163


namespace three_digit_multiples_of_5_count_l618_618054

theorem three_digit_multiples_of_5_count : 
  let digits := {2, 0, 1, 5}
  (count (n : ℕ) (n < 1000 ∧ n ≥ 100 ∧ n % 5 = 0 ∧ ∀ i j : ℕ, i ≠ j → nth_digit i n digits ≠ nth_digit j n digits)) = 12 :=
by
  sorry

end three_digit_multiples_of_5_count_l618_618054


namespace ball_hits_ground_l618_618863

theorem ball_hits_ground (t : ℝ) :
  (∃ t, -(16 * t^2) + 32 * t + 30 = 0 ∧ t = 1 + (Real.sqrt 46) / 4) :=
sorry

end ball_hits_ground_l618_618863


namespace count_values_n_prime_f_l618_618795

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d => d ∣ n) |>.sum

def count_primes (range : Finset ℕ) (f : ℕ → ℕ) : ℕ :=
  range.filter (λ n => is_prime (f n)) |>.card

theorem count_values_n_prime_f :
  count_primes (Finset.range 31) sum_of_divisors = 5 := sorry

end count_values_n_prime_f_l618_618795


namespace sequence_even_numbers_sequence_odd_numbers_sequence_square_numbers_sequence_arithmetic_progression_l618_618488

-- Problem 1: Prove the general formula for the sequence of all positive even numbers
theorem sequence_even_numbers (n : ℕ) : ∃ a_n, a_n = 2 * n := by 
  sorry

-- Problem 2: Prove the general formula for the sequence of all positive odd numbers
theorem sequence_odd_numbers (n : ℕ) : ∃ b_n, b_n = 2 * n - 1 := by 
  sorry

-- Problem 3: Prove the general formula for the sequence 1, 4, 9, 16, ...
theorem sequence_square_numbers (n : ℕ) : ∃ a_n, a_n = n^2 := by
  sorry

-- Problem 4: Prove the general formula for the sequence -4, -1, 2, 5, ...
theorem sequence_arithmetic_progression (n : ℕ) : ∃ a_n, a_n = 3 * n - 7 := by
  sorry

end sequence_even_numbers_sequence_odd_numbers_sequence_square_numbers_sequence_arithmetic_progression_l618_618488


namespace angle_CBD_is_10_degrees_l618_618366

theorem angle_CBD_is_10_degrees (angle_ABC angle_ABD : ℝ) (h1 : angle_ABC = 40) (h2 : angle_ABD = 30) :
  angle_ABC - angle_ABD = 10 :=
by
  sorry

end angle_CBD_is_10_degrees_l618_618366


namespace largest_partner_share_l618_618644

def total_profit : ℕ := 48000
def partner_ratios : List ℕ := [3, 4, 4, 6, 7]
def value_per_part : ℕ := total_profit / partner_ratios.sum
def largest_share : ℕ := 7 * value_per_part

theorem largest_partner_share :
  largest_share = 14000 := by
  sorry

end largest_partner_share_l618_618644


namespace limit_is_exp_l618_618587

noncomputable def limit_expression (α β : ℝ) :=
  (λ x : ℝ, (1 + (Real.sin x) * (Real.cos (α * x))) / (1 + (Real.sin x) * (Real.cos (β * x))) ) ** (Nat.cubic (Real.cot x))

theorem limit_is_exp (α β : ℝ) : 
  Filter.Tendsto (λ x : ℝ, (limit_expression α β x)) (Filter.nhds_within 0 set.univ) (𝓝 (Real.exp ((β^2 - α^2) / 2))) :=
  sorry

end limit_is_exp_l618_618587


namespace butterfly_total_distance_l618_618509

open Real

noncomputable def total_distance_flights (radius : ℝ) (back_to_start : ℝ) : ℝ :=
  let diameter := 2 * radius
  let other_leg := sqrt (diameter^2 - back_to_start^2)
  diameter + back_to_start + other_leg

theorem butterfly_total_distance
  (radius : ℝ) (back_to_start : ℝ)
  (h_radius : radius = 75) (h_back_to_start : back_to_start = 100) :
  total_distance_flights radius back_to_start = 361.803 :=
by
  rw [h_radius, h_back_to_start]
  norm_num
  have h1: 2 * 75 = 150 := by norm_num
  rw h1
  have h2: sqrt (150 ^ 2 - 100 ^ 2) = 111.803 := sorry  -- Calculation already shown
  rw h2
  norm_num
  sorry

end butterfly_total_distance_l618_618509


namespace expr1_simplified_expr2_simplified_l618_618088

variable (a x : ℝ)

theorem expr1_simplified : (-a^3 + (-4 * a^2) * a) = -5 * a^3 := 
by
  sorry

theorem expr2_simplified : (-x^2 * (-x)^2 * (-x^2)^3 - 2 * x^10) = -x^10 := 
by
  sorry

end expr1_simplified_expr2_simplified_l618_618088


namespace ship_meetings_l618_618815

/-- 
On an east-west shipping lane are ten ships sailing individually. The first five from the west are sailing eastwards while the other five ships are sailing westwards. They sail at the same constant speed at all times. Whenever two ships meet, each turns around and sails in the opposite direction. 

When all ships have returned to port, how many meetings of two ships have taken place? 

Proof: The total number of meetings is 25.
-/
theorem ship_meetings (east_ships west_ships : ℕ) (h_east : east_ships = 5) (h_west : west_ships = 5) : 
  east_ships * west_ships = 25 :=
by
  rw [h_east, h_west]
  exact Mul.mul 5 5
  exact eq.refl 25

end ship_meetings_l618_618815


namespace stop_signs_per_mile_correct_l618_618398

def townA_population_density := 500
def townA_speed_limit := 30
def townA_distance_traveled := 7 -- 5 + 2
def townA_stop_signs := townA_population_density / 10
def townA_miles_counted := townA_distance_traveled / 3 -- counting every 3rd mile

def townB_population_density := 1000
def townB_speed_limit := 25
def townB_distance_traveled := 10 -- 7 + 3
def townB_stop_signs := 1.2 * townA_stop_signs
def townB_miles_counted := townB_distance_traveled / 2 -- counting every 2nd mile

def townC_population_density := 1500
def townC_speed_limit := 20
def townC_distance_traveled := 13 -- 9 + 4
def townC_stop_signs := (townA_stop_signs + townB_stop_signs) * townC_speed_limit
def townC_miles_counted := townC_distance_traveled -- counting all miles

theorem stop_signs_per_mile_correct :
  (townA_stop_signs / townA_miles_counted = 25) ∧
  (townB_stop_signs / townB_miles_counted = 12) ∧
  (townC_stop_signs / townC_distance_traveled = 169.23) :=
by
  sorry

end stop_signs_per_mile_correct_l618_618398


namespace Alyssa_total_spent_l618_618133

/-- Definition of fruit costs -/
def cost_grapes : ℝ := 12.08
def cost_cherries : ℝ := 9.85
def cost_mangoes : ℝ := 7.50
def cost_pineapple : ℝ := 4.25
def cost_starfruit : ℝ := 3.98

/-- Definition of tax and discount -/
def tax_rate : ℝ := 0.10
def discount : ℝ := 3.00

/-- Calculation of the total cost Alyssa spent after applying tax and discount -/
def total_spent : ℝ := 
  let total_cost_before_tax := cost_grapes + cost_cherries + cost_mangoes + cost_pineapple + cost_starfruit
  let tax := tax_rate * total_cost_before_tax
  let total_cost_with_tax := total_cost_before_tax + tax
  total_cost_with_tax - discount

/-- Statement that needs to be proven -/
theorem Alyssa_total_spent : total_spent = 38.43 := by 
  sorry

end Alyssa_total_spent_l618_618133


namespace shirts_count_l618_618765

theorem shirts_count (S : ℕ) (hours_per_shirt hours_per_pant cost_per_hour total_pants total_cost : ℝ) :
  hours_per_shirt = 1.5 →
  hours_per_pant = 3 →
  cost_per_hour = 30 →
  total_pants = 12 →
  total_cost = 1530 →
  45 * S + 1080 = total_cost →
  S = 10 :=
by
  intros hps hpp cph tp tc cost_eq
  sorry

end shirts_count_l618_618765


namespace largest_initial_number_l618_618323

theorem largest_initial_number :
  ∃ n : ℕ, 
    (∀ (a1 a2 a3 a4 a5 : ℕ),
      (n + a1).gcd a1 = 1 ∧
      (n + a1 + a2).gcd a2 = 1 ∧
      (n + a1 + a2 + a3).gcd a3 = 1 ∧
      (n + a1 + a2 + a3 + a4).gcd a4 = 1 ∧
      (n + a1 + a2 + a3 + a4 + a5).gcd a5 = 1 ∧
      n + a1 + a2 + a3 + a4 + a5 = 200) 
    → n = 189 :=
begin
  sorry
end

end largest_initial_number_l618_618323


namespace perpendicular_lines_find_b_l618_618051

theorem perpendicular_lines_find_b :
  let v1 := (-4 : ℤ, 5 : ℤ)
  let v2 := (b : ℤ, 3 : ℤ)
  (v1.1 * v2.1 + v1.2 * v2.2 = 0) → b = 15 / 4 := sorry

end perpendicular_lines_find_b_l618_618051


namespace correct_number_of_outfits_l618_618715

def num_shirts : ℕ := 7
def num_pants : ℕ := 7
def num_hats : ℕ := 7
def num_colors : ℕ := 7

def total_outfits : ℕ := num_shirts * num_pants * num_hats
def invalid_outfits : ℕ := num_colors
def valid_outfits : ℕ := total_outfits - invalid_outfits

theorem correct_number_of_outfits : valid_outfits = 336 :=
by {
  -- sorry can be removed when providing the proof.
  sorry
}

end correct_number_of_outfits_l618_618715


namespace correct_number_of_outfits_l618_618716

def num_shirts : ℕ := 7
def num_pants : ℕ := 7
def num_hats : ℕ := 7
def num_colors : ℕ := 7

def total_outfits : ℕ := num_shirts * num_pants * num_hats
def invalid_outfits : ℕ := num_colors
def valid_outfits : ℕ := total_outfits - invalid_outfits

theorem correct_number_of_outfits : valid_outfits = 336 :=
by {
  -- sorry can be removed when providing the proof.
  sorry
}

end correct_number_of_outfits_l618_618716


namespace train_speed_calculation_l618_618115

noncomputable def speed_of_train_in_kmph
  (length_of_train : ℝ)
  (length_of_bridge : ℝ)
  (time_to_cross_bridge : ℝ) : ℝ :=
(length_of_train + length_of_bridge) / time_to_cross_bridge * 3.6

theorem train_speed_calculation
  (length_of_train : ℝ)
  (length_of_bridge : ℝ)
  (time_to_cross_bridge : ℝ)
  (h_train : length_of_train = 150)
  (h_bridge : length_of_bridge = 225)
  (h_time : time_to_cross_bridge = 30) :
  speed_of_train_in_kmph length_of_train length_of_bridge time_to_cross_bridge = 45 := by
  simp [speed_of_train_in_kmph, h_train, h_bridge, h_time]
  norm_num
  sorry

end train_speed_calculation_l618_618115


namespace area_relationship_l618_618491

variables (A B C A' B' C' A1 A2 B1 B2 C1 C2 : Point)
variables (a b c : ℝ) -- Side lengths
variables (S : ℝ) -- Area of triangle ABC
variables (R : ℝ) -- Circumradius of triangle ABC

-- Altitudes
hypothesis hA_A' : Altitude A A'
hypothesis hB_B' : Altitude B B'
hypothesis hC_C' : Altitude C C'

-- Projections
hypothesis hA'AB_AC : Projection A' A1 A2 AB AC
hypothesis hB'BC_BA : Projection B' B1 B2 BC BA
hypothesis hC'CA_CB : Projection C' C1 C2 CA CB

-- Areas of the smaller triangles
variables (S_A'A1A2 S_B'B1B2 S_C'C1C2 : ℝ)
hypothesis hArea_A'A1A2 : S_A'A1A2 = Area A' A1 A2
hypothesis hArea_B'B1B2 : S_B'B1B2 = Area B' B1 B2
hypothesis hArea_C'C1C2 : S_C'C1C2 = Area C' C1 C2

theorem area_relationship :
  a^2 * S_A'A1A2 + b^2 * S_B'B1B2 + c^2 * S_C'C1C2 = S^3 / R^2 := 
sorry

end area_relationship_l618_618491


namespace min_value_fraction_l618_618680

theorem min_value_fraction (x y : ℝ) (h : (x + 2)^2 + y^2 = 1) :
  ∀ k : ℝ, 0 ≤ k ∧ k ≤ 4 / 3 → ∃ k = 0, (y - 1) / (x - 2) = k :=
by
  sorry

end min_value_fraction_l618_618680


namespace evaluate_expression_l618_618168

theorem evaluate_expression :
  let a := 3 * 4 * 5
  let b := (1 : ℝ) / 3
  let c := (1 : ℝ) / 4
  let d := (1 : ℝ) / 5
  (a : ℝ) * (b + c - d) = 23 := by
  sorry

end evaluate_expression_l618_618168


namespace train_speed_l618_618118

-- Definitions for the conditions
def distance_meters : ℕ := 1600
def time_seconds : ℕ := 40

-- Conversion factors
def meters_to_kilometers (d_m : ℕ) : ℝ := d_m / 1000.0
def seconds_to_hours (t_s : ℕ) : ℝ := t_s / 3600.0

-- Proof problem statement
theorem train_speed (d : ℕ) (t : ℕ) (h₁ : d = distance_meters) (h₂ : t = time_seconds) :
  meters_to_kilometers d / seconds_to_hours t = 144 :=
begin
  sorry
end

end train_speed_l618_618118


namespace first_house_gets_90_bottles_l618_618122

def bottles_of_drinks (total_bottles bottles_cider_only bottles_beer_only : ℕ) : ℕ :=
  let bottles_mixture := total_bottles - bottles_cider_only - bottles_beer_only
  let first_house_cider := bottles_cider_only / 2
  let first_house_beer := bottles_beer_only / 2
  let first_house_mixture := bottles_mixture / 2
  first_house_cider + first_house_beer + first_house_mixture
  
theorem first_house_gets_90_bottles :
  bottles_of_drinks 180 40 80 = 90 :=
by
  rw [bottles_of_drinks]
  sorry

end first_house_gets_90_bottles_l618_618122


namespace ratio_of_a_over_4_to_b_over_3_l618_618270

noncomputable def a (c : ℝ) : ℝ := 2 * c^2 + 3 * c + real.sqrt c
noncomputable def b (c : ℝ) : ℝ := c^2 + 5 * c - c^(3 / 2)

theorem ratio_of_a_over_4_to_b_over_3 (c : ℝ) (hc : c ≠ 0) (hab : (a c) * (b c) * (c^2 + 2 * c + 1) ≠ 0) :
  3 * (a c)^2 = 4 * (b c)^2 → (a c / 4) / (b c / 3) = real.sqrt 3 / 2 :=
by
  intro h
  sorry

end ratio_of_a_over_4_to_b_over_3_l618_618270


namespace digits_in_2_pow_120_l618_618181

theorem digits_in_2_pow_120 {a b : ℕ} (h : 10^a ≤ 2^200 ∧ 2^200 < 10^b) (ha : a = 60) (hb : b = 61) : 
  ∃ n : ℕ, 10^(n-1) ≤ 2^120 ∧ 2^120 < 10^n ∧ n = 37 :=
by {
  sorry
}

end digits_in_2_pow_120_l618_618181


namespace cos_value_of_2alpha_plus_5pi_over_12_l618_618675

theorem cos_value_of_2alpha_plus_5pi_over_12
  (α : ℝ) (h1 : Real.pi / 2 < α ∧ α < Real.pi)
  (h2 : Real.sin (α + Real.pi / 3) = -4 / 5) :
  Real.cos (2 * α + 5 * Real.pi / 12) = 17 * Real.sqrt 2 / 50 :=
by 
  sorry

end cos_value_of_2alpha_plus_5pi_over_12_l618_618675


namespace green_cards_count_l618_618459

theorem green_cards_count (total_cards red_frac black_frac : ℕ → ℝ)
  (h_total : total_cards = 120)
  (h_red_frac : red_frac = 2 / 5)
  (h_black_frac : black_frac = 5 / 9)
  (h_red_cards : ∀ total red_frac, red_frac * total = 48)
  (h_black_cards : ∀ remainder black_frac, black_frac * remainder = 40) :
  total_cards - (48 + 40) = 32 :=
by
  rw [h_total, h_red_frac, h_black_frac, ←h_red_cards, ←h_black_cards]
  real.smul
  refl 
  sorry

end green_cards_count_l618_618459


namespace find_bca_l618_618447

/--
The repeating decimals 0.bcbc... and 0.bcabc... satisfy
0.bcbc... + 0.bcabc... = 41/111, where b, c, and a are digits.
Find the three-digit number bca.
-/

noncomputable def repeating_fraction_bc (b c : ℕ) : ℚ :=
  ⟨10 * b + c, 99⟩

noncomputable def repeating_fraction_bcabc (b c a : ℕ) : ℚ :=
  ⟨10000 * b + 1000 * c + 100 * a + 10 * b + c, 99999⟩

def is_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

theorem find_bca (b c a : ℕ) (h_b : is_digit b) (h_c : is_digit c) (h_a : is_digit a)
  (h : repeating_fraction_bc b c + repeating_fraction_bcabc b c a = 41 / 111) :
  b * 100 + c * 10 + a = 341 :=
by
  sorry

end find_bca_l618_618447


namespace least_four_digit_palindrome_divisible_by_11_l618_618064

theorem least_four_digit_palindrome_divisible_by_11 : 
  ∃ (A B : ℕ), (A ≠ 0 ∧ A < 10 ∧ B < 10 ∧ 1000 * A + 100 * B + 10 * B + A = 1111 ∧ (2 * A - 2 * B) % 11 = 0) := 
by
  sorry

end least_four_digit_palindrome_divisible_by_11_l618_618064


namespace cannot_realize_degree_sequence_l618_618882

theorem cannot_realize_degree_sequence :
  ¬ ∃ (G : SimpleGraph (Fin 10)), 
    (G.degree ∘ G.vertices = [9, 7, 6, 5, 5, 3, 3, 2, 1, 1]) :=
by
  sorry

end cannot_realize_degree_sequence_l618_618882


namespace average_speed_of_train_l618_618119

-- Define conditions
def traveled_distance1 : ℝ := 240
def traveled_distance2 : ℝ := 450
def time_period1 : ℝ := 3
def time_period2 : ℝ := 5

-- Define total distance and total time based on the conditions
def total_distance : ℝ := traveled_distance1 + traveled_distance2
def total_time : ℝ := time_period1 + time_period2

-- Prove that the average speed is 86.25 km/h
theorem average_speed_of_train : total_distance / total_time = 86.25 := by
  -- Here should be the proof, but we put sorry since we only need the statement
  sorry

end average_speed_of_train_l618_618119


namespace sum_of_fourth_powers_less_than_150_l618_618472

-- Definition of the required sum
def sum_of_fourth_powers (n : ℕ) : ℕ :=
  (Finset.range n).filter (λ x => ∃ (k : ℕ), x = k^4).sum

-- Statement of the theorem to be proved
theorem sum_of_fourth_powers_less_than_150 : sum_of_fourth_powers 150 = 98 :=
by
  sorry

end sum_of_fourth_powers_less_than_150_l618_618472


namespace fixed_circle_Q_l618_618206

-- Given a fixed circle Γ and three fixed points A, B, and C on it
variables (Γ : Type*) [circle Γ] (A B C : point Γ)
-- λ is a real number in the interval (0,1)
variable (λ : ℝ) (hλ : 0 < λ ∧ λ < 1)
-- P is a moving point on Γ distinct from A, B, and C
variable (P : point Γ) (hP : circle.mem Γ P) (hPA : P ≠ A) (hPB : P ≠ B) (hPC : P ≠ C)
-- M is a point on line segment CP such that CM = λ * CP
variable (M : point Γ) (hM : line_segment.condition C P M (λ * (distance C P)))
-- Q is the second intersection point of the circumcircles of triangles AMP and BMC
variable (Q : point Γ)
variable (hQ : second_intersection_circumcircle Γ A M P B C Q)

theorem fixed_circle_Q : ∀ (P : point Γ), circle.mem Γ P ∧ P ≠ A ∧ P ≠ B ∧ P ≠ C →
  ∃ (K : circle), ∀ (Q : point Γ), hQ Q → circle.mem K Q :=
sorry

end fixed_circle_Q_l618_618206


namespace students_taking_both_courses_l618_618738

theorem students_taking_both_courses (total_students students_french students_german students_neither both_courses : ℕ) 
(h1 : total_students = 94) 
(h2 : students_french = 41) 
(h3 : students_german = 22) 
(h4 : students_neither = 40) 
(h5 : total_students = students_french + students_german - both_courses + students_neither) :
both_courses = 9 :=
by
  -- sorry can be replaced with the actual proof if necessary
  sorry

end students_taking_both_courses_l618_618738


namespace focal_length_of_ellipse_l618_618430

theorem focal_length_of_ellipse {x y : ℝ} (h : 2*x^2 + 3*y^2 = 6) : 
  ∃ c, c = 2 ∧ ellipse_focal_length 2 3 = c :=
begin
  sorry
end

end focal_length_of_ellipse_l618_618430


namespace prime_of_token_movement_l618_618931

theorem prime_of_token_movement (n : ℕ) (A : fin (2 * n) → ℤ)
  (movable : ∀ k : fin (2 * n), ∃ k' : fin (2 * n), ∃ m : ℤ, k' = ⟨(k.1 + m) % (2 * n), by linarith⟩ ∧ k' ≠ k) :
  nat.prime (2 * n + 1) :=
sorry

end prime_of_token_movement_l618_618931


namespace copy_pages_l618_618760

theorem copy_pages : 
  (fixed_fee: ℕ) (copy_cost: ℕ) (pages_per_cost: ℕ) (total_money: ℕ)
  (h1: fixed_fee = 500) (h2: copy_cost = 7) (h3: pages_per_cost = 2) (h4: total_money = 3500) :
  ((total_money - fixed_fee) * pages_per_cost) / copy_cost = 857 :=
by
  sorry

end copy_pages_l618_618760


namespace range_of_m_l618_618789

noncomputable def f (x m : ℝ) := (1/2) * x^2 + m * x + Real.log x

noncomputable def f_prime (x m : ℝ) := x + 1/x + m

theorem range_of_m (x0 m : ℝ) 
  (h1 : (1/2) ≤ x0 ∧ x0 ≤ 3) 
  (unique_x0 : ∀ y, f_prime y m = 0 → y = x0) 
  (cond1 : f_prime (1/2) m < 0) 
  (cond2 : f_prime 3 m ≥ 0) 
  : -10 / 3 ≤ m ∧ m < -5 / 2 :=
sorry

end range_of_m_l618_618789


namespace bus_ride_blocks_to_coffee_shop_l618_618138

variable (total_blocks walked_blocks bus_blocks : ℕ)

def find_bus_blocks_to_coffee_shop (total_blocks walked_blocks : ℕ) : ℕ := 
  (total_blocks - 2 * walked_blocks) / 2

theorem bus_ride_blocks_to_coffee_shop
  (total_blocks : ℕ) (walked_blocks : ℕ) (h1 : walked_blocks = 5) (h2 : total_blocks = 24) : bus_blocks = 7 :=
by
  rw [h1, h2]
  show bus_blocks = find_bus_blocks_to_coffee_shop total_blocks walked_blocks
  rfl

end

end bus_ride_blocks_to_coffee_shop_l618_618138


namespace arithmetic_mean_of_three_digit_multiples_of_9_l618_618906

theorem arithmetic_mean_of_three_digit_multiples_of_9 :
  let a := 108
  let l := 999
  let d := 9
  let n := (l - a) / d + 1
  let sum := n * (a + l) / 2
  (sum : ℝ) / n = 553.5 :=
by {
  let a := 108,
  let l := 999,
  let d := 9,
  let n := (l - a) / d + 1,
  let sum := n * (a + l) / 2,
  have n_eq_100 : n = 100 := by sorry,  -- this is where proof for n = 100 would go
  have sum_eq_55350 : sum = 55350 := by sorry,  -- this is where proof for sum = 55350 would go
  have mean_eq_553_5 : (sum : ℝ) / n = 553.5 := by sorry,  -- finally, we show the mean is 553.5
  exact mean_eq_553_5,
}

end arithmetic_mean_of_three_digit_multiples_of_9_l618_618906


namespace monochromatic_triangle_probability_l618_618612

-- Define the coloring of the edges
inductive Color
| Red : Color
| Blue : Color

-- Define an edge
structure Edge :=
(v1 v2 : Nat)
(color : Color)

-- Define the hexagon with its sides and diagonals
def hexagonEdges : List Edge := [
  -- Sides of the hexagon
  { v1 := 1, v2 := 2, color := sorry }, { v1 := 2, v2 := 3, color := sorry },
  { v1 := 3, v2 := 4, color := sorry }, { v1 := 4, v2 := 5, color := sorry },
  { v1 := 5, v2 := 6, color := sorry }, { v1 := 6, v2 := 1, color := sorry },
  -- Diagonals of the hexagon
  { v1 := 1, v2 := 3, color := sorry }, { v1 := 1, v2 := 4, color := sorry },
  { v1 := 1, v2 := 5, color := sorry }, { v1 := 2, v2 := 4, color := sorry },
  { v1 := 2, v2 := 5, color := sorry }, { v1 := 2, v2 := 6, color := sorry },
  { v1 := 3, v2 := 5, color := sorry }, { v1 := 3, v2 := 6, color := sorry },
  { v1 := 4, v2 := 6, color := sorry }
]

-- Define what a triangle is
structure Triangle :=
(v1 v2 v3 : Nat)

-- List all possible triangles formed by vertices of the hexagon
def hexagonTriangles : List Triangle := [
  { v1 := 1, v2 := 2, v3 := 3 }, { v1 := 1, v2 := 2, v3 := 4 },
  { v1 := 1, v2 := 2, v3 := 5 }, { v1 := 1, v2 := 2, v3 := 6 },
  { v1 := 1, v2 := 3, v3 := 4 }, { v1 := 1, v2 := 3, v3 := 5 },
  { v1 := 1, v2 := 3, v3 := 6 }, { v1 := 1, v2 := 4, v3 := 5 },
  { v1 := 1, v2 := 4, v3 := 6 }, { v1 := 1, v2 := 5, v3 := 6 },
  { v1 := 2, v2 := 3, v3 := 4 }, { v1 := 2, v2 := 3, v3 := 5 },
  { v1 := 2, v2 := 3, v3 := 6 }, { v1 := 2, v2 := 4, v3 := 5 },
  { v1 := 2, v2 := 4, v3 := 6 }, { v1 := 2, v2 := 5, v3 := 6 },
  { v1 := 3, v2 := 4, v3 := 5 }, { v1 := 3, v2 := 4, v3 := 6 },
  { v1 := 3, v2 := 5, v3 := 6 }, { v1 := 4, v2 := 5, v3 := 6 }
]

-- Define the probability calculation, with placeholders for terms that need proving
noncomputable def probabilityMonochromaticTriangle : ℚ :=
  1 - (3 / 4) ^ 20

-- The theorem to prove the probability matches the given answer
theorem monochromatic_triangle_probability :
  probabilityMonochromaticTriangle = 253 / 256 :=
by sorry

end monochromatic_triangle_probability_l618_618612


namespace find_B_l618_618771

noncomputable def parabola (x : ℝ) : ℝ := 2 * x^2

noncomputable def normal_line (x : ℝ) (y : ℝ) : ℝ := 
  - (1 / 8) * (x - 2) + 4

noncomputable def point_A := (2 : ℝ, 4 : ℝ)
noncomputable def point_B := (-17 / 16 : ℝ, 289 / 128 : ℝ)

theorem find_B :
  ∃ B : ℝ × ℝ,
    (parabola B.1 = B.2) ∧
    (normal_line B.1 B.2 = B.2) ∧
    B ≠ point_A ∧
    B = point_B :=
  sorry

end find_B_l618_618771


namespace derivative_problem_1_derivative_problem_2_derivative_problem_3_l618_618629

open Real

-- Problem 1
theorem derivative_problem_1 (x: ℝ) :
  deriv (fun x => 2 * x^3 + x^(1/3) + cos x - 1) x = 6 * x^2 + (1/3) * x^(-2/3) - sin x :=
by 
  sorry

-- Problem 2
theorem derivative_problem_2 (x: ℝ) :
  deriv (fun x => (x^3 + 1) * (2 * x^2 + 8 * x - 5)) x = 10 * x^4 + 32 * x^3 - 15 * x^2 + 4 * x + 8 :=
by 
  sorry

-- Problem 3
theorem derivative_problem_3 (x: ℝ) (hx: 0 < x) :
  deriv (fun x => (log x + 2^x) / x^2) x = (1 - 2 * log x + (x * log 2 - 2) * 2^x) / x^3 :=
by 
  sorry

end derivative_problem_1_derivative_problem_2_derivative_problem_3_l618_618629


namespace area_of_rhombus_l618_618627

theorem area_of_rhombus (x : ℝ) :
  let d1 := 3 * x + 5
  let d2 := 2 * x + 4
  (d1 * d2) / 2 = 3 * x^2 + 11 * x + 10 :=
by
  let d1 := 3 * x + 5
  let d2 := 2 * x + 4
  have h1 : d1 = 3 * x + 5 := rfl
  have h2 : d2 = 2 * x + 4 := rfl
  simp [h1, h2]
  sorry

end area_of_rhombus_l618_618627


namespace largest_initial_number_l618_618308

theorem largest_initial_number : ∃ n : ℕ, (n + 5 ∑ k : ℕ, k ≠ 0 ∧ ¬ (n % k = 0)) = 200 ∧ n = 189 :=
begin
  sorry
end

end largest_initial_number_l618_618308


namespace solve_for_x_l618_618078

theorem solve_for_x (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 10) : x = 15 := by
  sorry

end solve_for_x_l618_618078


namespace Serena_age_proof_l618_618839

-- Define the variables and conditions
def SerenaAgeCurrent := Nat
def MotherAgeCurrent := 39
def MotherAgeInSixYears : Nat := MotherAgeCurrent + 6
def SerenaAgeInSixYears (S : SerenaAgeCurrent) : Nat := S + 6

-- The condition that in 6 years, the mother will be three times as old as Serena
def age_condition (S : SerenaAgeCurrent) : Prop := MotherAgeInSixYears = 3 * SerenaAgeInSixYears S

-- Declare the main statement to prove
theorem Serena_age_proof : ∃ (S : SerenaAgeCurrent), age_condition S ∧ S = 9 :=
by
  -- Sorry used to skip the proof
  sorry

end Serena_age_proof_l618_618839


namespace value_added_to_reverse_digits_of_number_is_18_l618_618542

theorem value_added_to_reverse_digits_of_number_is_18 :
  ∀ (x y a : ℕ), (10 * x + y = 24) → (x * y = 8) → (10 * x + y + a = 10 * y + x) → a = 18 :=
by
  -- assume x, y, a are natural numbers
  assume x y a
  -- assume that 10 * x + y = 24
  assume h1 : 10 * x + y = 24
  -- assume that x * y = 8
  assume h2 : x * y = 8
  -- assume that 10 * x + y + a = 10 * y + x
  assume h3 : 10 * x + y + a = 10 * y + x
  -- conclusion that a = 18
  sorry

end value_added_to_reverse_digits_of_number_is_18_l618_618542


namespace largest_subset_no_four_times_another_l618_618965

theorem largest_subset_no_four_times_another :
  ∃ (S : set ℕ), S ⊆ {1, 2, ..., 150} ∧ (∀ (a b : ℕ), a ∈ S → b ∈ S → (a ≠ 4 * b ∧ b ≠ 4 * a)) ∧ (S.card = 141) :=
sorry

end largest_subset_no_four_times_another_l618_618965


namespace Kolya_mistake_l618_618019

def boys := ["Vasya", "Kolya", "Petya", "Misha"]

constant num_blue_pencils : ℕ
constant num_green_pencils : ℕ

axiom Vasya_statement : num_blue_pencils >= 4
axiom Kolya_statement : num_green_pencils >= 5
axiom Petya_statement : num_blue_pencils >= 3 ∧ num_green_pencils >= 4
axiom Misha_statement : num_blue_pencils >= 4 ∧ num_green_pencils >= 4

axiom three_truths_one_mistake : 
  (Vasya_statement ∨ ¬Vasya_statement) ∧
  (Kolya_statement ∨ ¬Kolya_statement) ∧
  (Petya_statement ∨ ¬Petya_statement) ∧
  (Misha_statement ∨ ¬Misha_statement) ∧
  ((Vasya_statement ? true : 1) + 
   (Kolya_statement ? true : 1) + 
   (Petya_statement ? true : 1) +
   (Misha_statement ? true : 1) == 3)

theorem Kolya_mistake : ¬Kolya_statement :=
by
  sorry

end Kolya_mistake_l618_618019


namespace fraction_red_mushrooms_white_spots_l618_618560

variable (red brown green blue whiteSpotted : ℕ)
variable (f : ℚ)

-- Conditions
def Bill_gathered_red_mushrooms := red = 12
def Bill_gathered_brown_mushrooms := brown = 6
def Ted_gathered_green_mushrooms := green = 14
def Ted_gathered_blue_mushrooms := blue = 6
def Half_blue_have_white_spots := whiteSpotted + whiteSpotted = blue / 2
def All_brown_have_white_spots := whiteSpotted = brown
def Total_white_spotted_is_17 := 17 = red + brown + blue / 2

-- Theorem statement
theorem fraction_red_mushrooms_white_spots :
  Bill_gathered_red_mushrooms red →
  Bill_gathered_brown_mushrooms brown →
  Ted_gathered_green_mushrooms green →
  Ted_gathered_blue_mushrooms blue →
  Half_blue_have_white_spots blue →
  All_brown_have_white_spots brown →
  Total_white_spotted_is_17 red brown blue →
  f = 2 / 3 :=
begin
  sorry
end

end fraction_red_mushrooms_white_spots_l618_618560


namespace smallest_solution_l618_618641

def condition (x : ℝ) := |5 * x + 15| = 40

theorem smallest_solution : ∃ x : ℝ, condition x ∧ ∀ y : ℝ, condition y → y ≥ x :=
begin
  use -11,
  split,
  {
    unfold condition,
    norm_num,
    linarith,
  },
  {
    intros y hy,
    unfold condition at hy,
    split_ifs at hy with hy' hy'',
    {
      linarith,
    },
    {
      linarith,
    }
  }
end

end smallest_solution_l618_618641


namespace sum_first_10_log_a_terms_eq_minus_50_l618_618211

noncomputable def T (n : ℕ) : ℕ := 2^(n^2 - 15*n)

noncomputable def a (n : ℕ) : ℕ :=
  if n = 1 then 2^(-14)
  else 2^( -(16 - 2*n) )

-- Define the sequence log_a in Lean
noncomputable def log_a (n : ℕ) : ℕ :=
  2*n - 16

theorem sum_first_10_log_a_terms_eq_minus_50 :
  ∑ i in Finset.range 10, log_a (i + 1) = -50 := by
  sorry

end sum_first_10_log_a_terms_eq_minus_50_l618_618211


namespace largest_subset_size_l618_618976

-- Definition of the subset condition
def valid_subset (s : Set ℕ) : Prop :=
  ∀ (a ∈ s) (b ∈ s), a ≠ 4 * b

-- Setting the range
def range_set : Set ℕ := { n | 1 ≤ n ∧ n ≤ 150 }

-- The main theorem statement
theorem largest_subset_size : ∃ (s : Set ℕ), s ⊆ range_set ∧ valid_subset (s) ∧ #(s) = 150 :=
by
  sorry

end largest_subset_size_l618_618976


namespace union_sets_l618_618220

def A : Set ℕ := {2, 1, 3}
def B : Set ℕ := {2, 3, 5}

theorem union_sets : A ∪ B = {1, 2, 3, 5} := 
by {
  sorry
}

end union_sets_l618_618220


namespace rabbit_travel_time_l618_618528

theorem rabbit_travel_time (distance : ℕ) (speed : ℕ) (time_in_minutes : ℕ) 
  (h_distance : distance = 3) 
  (h_speed : speed = 6) 
  (h_time_eqn : time_in_minutes = (distance * 60) / speed) : 
  time_in_minutes = 30 := 
by 
  sorry

end rabbit_travel_time_l618_618528


namespace probability_zero_l618_618379

noncomputable def Q (x : ℝ) : ℝ := x^2 - 4 * x - 6

def interval : set ℝ := {x | 3 ≤ x ∧ x ≤ 10}

def floor_sqrt_eq (x : ℝ) : Prop :=
  Real.floor (Real.sqrt (Q x)) = Real.sqrt (Q (Real.floor x))

theorem probability_zero :
  ∀ x ∈ interval, ¬floor_sqrt_eq x :=
by sorry

end probability_zero_l618_618379


namespace Kolya_mistake_l618_618022

def boys := ["Vasya", "Kolya", "Petya", "Misha"]

constant num_blue_pencils : ℕ
constant num_green_pencils : ℕ

axiom Vasya_statement : num_blue_pencils >= 4
axiom Kolya_statement : num_green_pencils >= 5
axiom Petya_statement : num_blue_pencils >= 3 ∧ num_green_pencils >= 4
axiom Misha_statement : num_blue_pencils >= 4 ∧ num_green_pencils >= 4

axiom three_truths_one_mistake : 
  (Vasya_statement ∨ ¬Vasya_statement) ∧
  (Kolya_statement ∨ ¬Kolya_statement) ∧
  (Petya_statement ∨ ¬Petya_statement) ∧
  (Misha_statement ∨ ¬Misha_statement) ∧
  ((Vasya_statement ? true : 1) + 
   (Kolya_statement ? true : 1) + 
   (Petya_statement ? true : 1) +
   (Misha_statement ? true : 1) == 3)

theorem Kolya_mistake : ¬Kolya_statement :=
by
  sorry

end Kolya_mistake_l618_618022


namespace max_initial_number_l618_618340

theorem max_initial_number (n : ℕ) : 
  (∃ (a b c d e : ℕ), 
    200 = n + a + b + c + d + e ∧ 
    ¬ (n % a = 0) ∧ 
    ¬ ((n + a) % b = 0) ∧ 
    ¬ ((n + a + b) % c = 0) ∧ 
    ¬ ((n + a + b + c) % d = 0) ∧ 
    ¬ ((n + a + b + c + d) % e = 0)) → 
  n ≤ 189 := 
sorry

end max_initial_number_l618_618340


namespace point_on_sphere_l618_618380

variables (a b c : ℝ) (a_pos : a ≠ 0) (b_pos : b ≠ 0) (c_pos : c ≠ 0)

def surface (x y z : ℝ) : ℝ := a * x^2 + b * y^2 + c * z^2 - 1
def sphere (x y z : ℝ) : ℝ := x^2 + y^2 + z^2 - (1 / a + 1 / b + 1 / c)

theorem point_on_sphere
  (x0 y0 z0 : ℝ)
  (h_tangent_planes : ∃ u1 u2 u3 v1 v2 v3 w1 w2 w3 λ1 λ2 λ3 : ℝ,
    (a * u1 * x0 + b * v1 * y0 + c * w1 * z0 = λ1) ∧
    (a * u2 * x0 + b * v2 * y0 + c * w2 * z0 = λ2) ∧
    (a * u3 * x0 + b * v3 * y0 + c * w3 * z0 = λ3) ∧
    (u1 ≠ 0 ∧ v1 ≠ 0 ∧ w1 ≠ 0) ∧
    (u2 ≠ 0 ∧ v2 ≠ 0 ∧ w2 ≠ 0) ∧
    (u3 ≠ 0 ∧ v3 ≠ 0 ∧ w3 ≠ 0) ∧
    (u1 = u2 = u3 ∧ v1 = v2 = v3 ∧ w1 = w2 = w3)) :
  sphere x0 y0 z0 = 0 :=
sorry

end point_on_sphere_l618_618380


namespace part1_part2_l618_618701

-- Define the sequence a_n
def a : ℕ → ℤ
| 0       := 2
| (n + 1) := 3 * a n + 2

-- Problem 1: Prove that the sequence {a_n + 1} is a geometric sequence.
theorem part1 (n : ℕ) : (a (n + 1) + 1) = 3 * (a n + 1) :=
by
  sorry

-- Define the sequence b_n as b_n = n * a_n
def b (n : ℕ) : ℤ := n * a n

-- Define the sum of the first n terms of the sequence {b_n}
def T (n : ℕ) : ℤ := ∑ i in Finset.range n, b (i + 1)

-- Problem 2: Prove the sum formula for the first n terms of the sequence {b_n}
theorem part2 (n : ℕ) : 
  T n = (n / 2 - 1 / 4) * 3 ^ (n + 1) + 3 / 4 - n * (n + 1) / 2 :=
by
  sorry

end part1_part2_l618_618701


namespace percentage_of_cookies_remaining_l618_618041

theorem percentage_of_cookies_remaining :
  let total_cookies := 1200
  let nicole_ratio := 7 / 15
  let eduardo_ratio := 5 / 8
  let sophia_ratio := 2 / 7
  let cookies_eaten_by_nicole := nicole_ratio * total_cookies
  let remaining_after_nicole := total_cookies - cookies_eaten_by_nicole
  let cookies_eaten_by_eduardo := eduardo_ratio * remaining_after_nicole
  let remaining_after_eduardo := remaining_after_nicole - cookies_eaten_by_eduardo
  let cookies_eaten_by_sophia := sophia_ratio * remaining_after_eduardo
  let remaining_after_sophia := remaining_after_eduardo - cookies_eaten_by_sophia
  let percentage_remaining := (remaining_after_sophia / total_cookies) * 100
  in percentage_remaining ≈ 14.29 :=
by
  sorry

end percentage_of_cookies_remaining_l618_618041


namespace countValidLabelings_l618_618611

-- Define the cube structure
structure Cube where
  edges : Fin 12 → Bool

-- Define properties of the cube
def validLabeling (cube : Cube) : Prop :=
  ∀ (faceEdges : Fin 6 → Fin 4), 
    (faceEdges.sum (fun edge => if cube.edges edge then 1 else 0)) = 3

-- The main theorem
theorem countValidLabelings : 
  ∃ (count : ℕ), count = 4 ∧ 
  (∀ cube : Cube, validLabeling cube → 
    (∃ (labelings : Fin count → Cube), ∀ n : Fin count, validLabeling (labelings n)))
 := by
  sorry

end countValidLabelings_l618_618611


namespace exponent_calculation_l618_618567

-- Define the necessary exponents and base
def base : ℕ := 19
def exp1 : ℕ := 11
def exp2 : ℕ := 8

-- Given condition
lemma power_property (a : ℕ) (m n : ℕ) : a^m / a^n = a^(m - n) := by sorry

-- Proof that 19^11 / 19^8 = 6859
theorem exponent_calculation : base^exp1 / base^exp2 = 6859 := by
  have : base^exp1 / base^exp2 = base^(exp1 - exp2) := power_property base exp1 exp2
  have : base^(exp1 - exp2) = base^3 := by rw [nat.sub_eq_iff_eq_add.mpr rfl]
  have : base^3 = 6859 := by -- This would be an arithmetic computation
    rfl -- Placeholder for the actual arithmetic; ideally, you'd verify this step.
  sorry

end exponent_calculation_l618_618567


namespace euclidean_algorithm_divisions_l618_618267

theorem euclidean_algorithm_divisions : 
  let a := 394
  let b := 82
  EuclideanAlgorithm.divsteps a b = 4 :=
by
  sorry

end euclidean_algorithm_divisions_l618_618267


namespace general_term_seq_l618_618432

theorem general_term_seq (n : ℕ) :
  (\<Seq 1, -1/3, 1/7, -1/15> n = \frac{(-1)^{n-1}}{2^n - 1}) := sorry

end general_term_seq_l618_618432


namespace gcd_m_l618_618787

def m' : ℕ := 33333333
def n' : ℕ := 555555555

theorem gcd_m'_n' : Nat.gcd m' n' = 3 := by
  sorry

end gcd_m_l618_618787


namespace BH_eq_DE_l618_618857

-- Variables
variables {A B C D E H O : Point}
noncomputable def omega : Circle := sorry -- Circumcircle of triangle ABC.
noncomputable def line_BO : Line := sorry -- The line passing through B and O.
noncomputable def point_on_BO : Point := sorry -- A point on the extension of BO beyond O.
noncomputable def parallel_line_H : Line := sorry -- A line passing through H parallel to BO.
noncomputable def intersects_arc (c : Circle) (p1 p2 : Point) : Set Point := sorry 
-- Intersection of the arc AC of the circumcircle at some point.

-- Given Conditions
axiom abc_acute : is_acute_angle (triangle A B C)
axiom AB_LT_BC : AB < BC
axiom altitude_intersect_H : intersection_of_altitudes (triangle A B C) = H
axiom D_prop : (D ∈ line_BO) ∧ (∠ADC = ∠ABC)
axiom parallel_prop : (parallel_line_H ∋ H) ∧ (parallel parallel_line_H line_BO)
axiom E_prop_intersects : E ∈ intersects_arc omega A C

-- To Prove
theorem BH_eq_DE : distance B H = distance D E :=
by
  sorry

end BH_eq_DE_l618_618857


namespace jamal_total_cost_l618_618762

-- Definitions based on conditions
def dozen := 12
def half_dozen := dozen / 2
def crayons_bought := 4 * half_dozen
def cost_per_crayon := 2
def total_cost := crayons_bought * cost_per_crayon

-- Proof statement (the question translated to a Lean theorem)
theorem jamal_total_cost : total_cost = 48 := by
  -- Proof skipped
  sorry

end jamal_total_cost_l618_618762


namespace smallest_positive_period_f_maximum_value_f_increasing_interval_f_l618_618242

noncomputable def f (x : ℝ) :=
  (cos (π + x)) * (cos (3 / 2 * π - x)) - (√3) * (cos x)^2 + (√3) / 2

theorem smallest_positive_period_f :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = π :=
sorry

theorem maximum_value_f :
  ∃ m, ∀ x : ℝ, f x ≤ m ∧ m = 1 :=
sorry

theorem increasing_interval_f :
  ∀ x ∈ [π / 6, 2/3 * π], (differentiable ℝ f x → (deriv f x) > 0 → x ∈ [π / 6, 5 * π / 12]) :=
sorry

end smallest_positive_period_f_maximum_value_f_increasing_interval_f_l618_618242


namespace set_complement_intersection_l618_618729

variable (U : Set ℕ) (M N : Set ℕ)

theorem set_complement_intersection
  (hU : U = {1, 2, 3, 4, 5, 6})
  (hM : M = {1, 4, 5})
  (hN : N = {2, 3}) :
  ((U \ N) ∩ M) = {1, 4, 5} :=
by
  sorry

end set_complement_intersection_l618_618729


namespace sum_of_zeros_eq_zero_l618_618876

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * Real.pi * Real.sin x)

theorem sum_of_zeros_eq_zero :
  ∑ z in {x : ℝ | f x = 0 ∧ x ∈ Ioo (-Real.pi / 2) (Real.pi / 2)}, z = 0 := 
  sorry

end sum_of_zeros_eq_zero_l618_618876


namespace sum_of_fourth_powers_lt_150_l618_618474

theorem sum_of_fourth_powers_lt_150 : (∑ n in finset.range 4, n^4) = 98 := by
  sorry

end sum_of_fourth_powers_lt_150_l618_618474


namespace solve_problem_statement_l618_618146

def problem_statement : Prop :=
  ∃ n, 3^19 % n = 7 ∧ n = 1162261460

theorem solve_problem_statement : problem_statement :=
  sorry

end solve_problem_statement_l618_618146


namespace largest_initial_number_l618_618316

theorem largest_initial_number (n : ℕ) (h : (∃ a b c d e : ℕ, n ≠ 0 ∧ n + a + b + c + d + e = 200 
                                              ∧ n % a ≠ 0 ∧ n % b ≠ 0 ∧ n % c ≠ 0 ∧ n % d ≠ 0 ∧ n % e ≠ 0)) 
: n ≤ 189 :=
sorry

end largest_initial_number_l618_618316


namespace modulus_of_complex_number_l618_618279

theorem modulus_of_complex_number :
  let z := 2 / (1 + real.sqrt 3 * complex.i) in
  complex.abs z = 1 :=
by
  let z := 2 / (1 + real.sqrt 3 * complex.i)
  exact sorry

end modulus_of_complex_number_l618_618279


namespace razorback_tshirt_shop_sales_l618_618421

theorem razorback_tshirt_shop_sales :
  let price_per_tshirt := 16 
  let tshirts_sold := 45 
  price_per_tshirt * tshirts_sold = 720 :=
by
  sorry

end razorback_tshirt_shop_sales_l618_618421


namespace border_area_is_correct_l618_618110

def framed_area (height width border: ℝ) : ℝ :=
  (height + 2 * border) * (width + 2 * border)

def photograph_area (height width: ℝ) : ℝ :=
  height * width

theorem border_area_is_correct (h w b : ℝ) (h6 : h = 6) (w8 : w = 8) (b3 : b = 3) :
  (framed_area h w b - photograph_area h w) = 120 := by
  sorry

end border_area_is_correct_l618_618110


namespace inverse_contrapositive_l618_618434

theorem inverse_contrapositive (a b c : ℝ) (h : a > b → a + c > b + c) :
  a + c ≤ b + c → a ≤ b :=
sorry

end inverse_contrapositive_l618_618434


namespace largest_initial_number_l618_618348

theorem largest_initial_number :
  ∃ n : ℕ, (∀ k : ℕ, (n % k ≠ 0 → k ∈ {2, 2, 2, 2, 3}) ∧ (n + 11 = 200)) ∧ (n = 189) :=
begin
  sorry -- Proof not required per instruction
end

end largest_initial_number_l618_618348


namespace exponent_division_l618_618576

theorem exponent_division (a : ℕ) (m n : ℕ) (h1 : 19 = a) (h2 : 11 = m) (h3 : 8 = n) : a^(m - n) = 6859 := by
  sorry

end exponent_division_l618_618576


namespace focus_on_line_BD_incircle_equation_l618_618250

-- Definitions for the conditions
def parabola_C := { p : ℝ × ℝ // p.2^2 = 4 * p.1 }
def focus_F := (1, 0)
def point_K := (-1, 0)
def line_l (m : ℝ) := { p : ℝ × ℝ // p.1 = m * p.2 - 1 }
def symmetric (A D : ℝ × ℝ) := A.1 = D.1 ∧ A.2 = -D.2

variable (m : ℝ)
def intersects (A B : ℝ × ℝ) := line_l m A ∧ parabola_C A ∧ line_l m B ∧ parabola_C B

variable (A B D : ℝ × ℝ)
def dot_product_condition := (A.1 - 1) * (B.1 - 1) + A.2 * B.2 = 8 / 9

-- Theorems for the questions to be proved

theorem focus_on_line_BD 
  (A B D : ℝ × ℝ) (m : ℝ)
  (h₁ : symmetric A D) 
  (h₂ : intersects m A B)
  : F ∈ line_BD B D := sorry

theorem incircle_equation
  (A B D : ℝ × ℝ) (m : ℝ)
  (h₁ : symmetric A D) 
  (h₂ : intersects m A B)
  (h₃ : dot_product_condition A B)
  : ∃ a r, a = (1/9) ∧ r = (2/3) ∧ (x - a)^2 + y^2 = r :=
sorry

end focus_on_line_BD_incircle_equation_l618_618250


namespace coefficient_x_neg_2_l618_618298

theorem coefficient_x_neg_2 : 
  let poly1 := (2 * x - 3)^2
  let poly2 := (1 - 1 / x)^6
  let expansion := poly1 * poly2
  coefficient_of_x_neg_2 expansion = 435 := sorry

end coefficient_x_neg_2_l618_618298


namespace back_wheel_revolutions_l618_618399

theorem back_wheel_revolutions (radius_front_wheel radius_back_wheel : ℝ)
                               (radius_front_eq : radius_front_wheel = 3)
                               (radius_back_eq : radius_back_wheel = 0.5)
                               (front_wheel_revolutions : ℕ)
                               (front_wheel_revolutions_eq : front_wheel_revolutions = 200) :
                               let circumference_front_wheel := 2 * Real.pi * radius_front_wheel,
                                   distance_traveled_front_wheel := circumference_front_wheel * front_wheel_revolutions,
                                   circumference_back_wheel := 2 * Real.pi * radius_back_wheel,
                                   back_wheel_revolutions := distance_traveled_front_wheel / circumference_back_wheel
                               in back_wheel_revolutions = 1200 :=
by
  sorry

end back_wheel_revolutions_l618_618399


namespace surface_area_RMO_l618_618959

theorem surface_area_RMO (h_prism : 10)
(base_triangle_base : 10)
(base_triangle_sides : 12)
(M_midpoint_PR : True)
(N_midpoint_QR : True)
(O_midpoint_TR : True) :
surface_area_RMO = 52.25 := by
  sorry

end surface_area_RMO_l618_618959


namespace best_graph_accelerating_decrease_l618_618654

/-
  Given percentages of working adults working remotely in Riverdale Town at different years:
  - In 2000, 40% of the adults worked remotely.
  - By 2005, this number decreased to 35%.
  - In 2010, the percentage further decreased to 25%.
  - By 2020, only 10% were working remotely.

  Prove that the graph that best illustrates this data shows an accelerating decrease.
-/

def remote_work_percentages : List (ℕ × ℕ) :=
  [(2000, 40), (2005, 35), (2010, 25), (2020, 10)]

theorem best_graph_accelerating_decrease (p : List (ℕ × ℕ)) 
  (h : p = remote_work_percentages) : 
  (∃ graph, graph = "accelerating decrease" ∧ illustrates p graph) := 
sorry

end best_graph_accelerating_decrease_l618_618654


namespace value_of_a2016_l618_618702

def a : ℕ → ℤ
| 0     := 1
| 1     := 3
| (n+2) := a (n+1) - a n

theorem value_of_a2016 : a 2016 = -2 := by
  sorry

end value_of_a2016_l618_618702


namespace length_of_square_side_l618_618831

theorem length_of_square_side (AB AC : ℝ) (h : AB = 8) (k : AC = 15) :
  ∃ s : ℝ, s = 120 / 17 :=
by
  have BC := real.sqrt (AB^2 + AC^2)
  have eq1 := AB = 8
  have eq2 := AC = 15
  sorry

end length_of_square_side_l618_618831


namespace general_term_of_geom_seq_sum_of_first_n_terms_of_geom_seq_l618_618449

variable {a_n : ℕ → ℝ} -- Define the sequence
variable {S_n : ℕ → ℝ} -- Define the sum of the first n terms
variable (a1 : ℝ) (q : ℝ) (n : ℕ) -- Variables for first term, common ratio, and index

-- Given conditions
def is_geometric_progression (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = a 1 * q^(n - 1)

def positive_terms (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in finset.range n, a i

def arithmetic_sequence_condition (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop :=
  S 3 + a 3 = (1 : ℝ) / 2 ∧ 
  S 5 + a 5 = 3 ↔ 
  S 4 + a 4 = 2

-- Definitions used in proof statements
axiom geom_seq (h1 : is_geometric_progression a_n)
  (h2 : positive_terms a_n)
  (h3 : ∀ n, a_n n > 0)
  (h4 : a_n 1 = a1)

-- Part I: Prove the general term formula for the sequence
theorem general_term_of_geom_seq :
  geom_seq → 
  is_geometric_progression a_n → 
  a1 = (1 : ℝ) / 2 →
  arithmetic_sequence_condition S_n a_n n →
  ∃ (general_term : ℕ → ℝ), general_term = λ n, (1 : ℝ) / 2 ^ n :=
sorry

-- Part II: Prove the sum of the first n terms of the sequence {na_n}
theorem sum_of_first_n_terms_of_geom_seq (a_n : ℕ → ℝ) (T_n : ℕ → ℝ) :
  (∀ n, a_n n = 1 / 2 ^ n) →
  T_n n = 2 - (n + 2) / 2^n :=
sorry

end general_term_of_geom_seq_sum_of_first_n_terms_of_geom_seq_l618_618449


namespace curve_eq_chord_length_l618_618253

theorem curve_eq (x y : ℝ) :
  (∃ (ρ θ : ℝ), (ρ = 2 * real.sin θ - 2 * real.cos θ) ∧ (x = ρ * real.cos θ) ∧ (y = ρ * real.sin θ)) ↔
  (x + 1)^2 + (y - 1)^2 = 2 :=
sorry

theorem chord_length : 
  let l := { t : ℝ // ∃ x y : ℝ, (x = 2 + (real.sqrt 2)/2 * t) ∧ (y = (real.sqrt 2)/2 * t) } in
  let C := (λ (x y : ℝ), (x + 1)^2 + (y - 1)^2 = 2) in
  ∃ A B : ℝ × ℝ, 
  (C A.1 A.2 ∧ C B.1 B.2 ∧
   (∃ (t : ℝ), (A = (2 + (real.sqrt 2)/2 * t , (real.sqrt 2)/2 * t)) ∧ 
               B = (-2, 0)) ∧
   (real.dist A B = 2 * real.sqrt 2)) :=
sorry

end curve_eq_chord_length_l618_618253


namespace derivative_of_f_domain_of_f_range_of_f_l618_618241

open Real

noncomputable def f (x : ℝ) := 1 / (x + sqrt (1 + 2 * x^2))

theorem derivative_of_f (x : ℝ) : 
  deriv f x = - ((sqrt (1 + 2 * x^2) + 2 * x) / (sqrt (1 + 2 * x^2) * (x + sqrt (1 + 2 * x^2))^2)) :=
by
  sorry

theorem domain_of_f : ∀ x : ℝ, f x ≠ 0 :=
by
  sorry

theorem range_of_f : 
  ∀ y : ℝ, 0 < y ∧ y ≤ sqrt 2 → ∃ x : ℝ, f x = y :=
by
  sorry

end derivative_of_f_domain_of_f_range_of_f_l618_618241


namespace proof_problem_l618_618776

noncomputable def T : Type := {x : ℝ // 0 < x}

def g (x : T) : T := sorry

theorem proof_problem (m t : ℝ) (h : m * t = 1 / 3) :
  (∀ x y : T, x.val + y.val ≠ 1 → g x + g y = x.val * y.val * g (g x + g y)) →
  (∀ x : T, g x = 1 / x.val) →
  m = 1 ∧ t = 1 / 3 ∧ m * t = 1 / 3 :=
by
  intro h1 h2
  sorry

end proof_problem_l618_618776


namespace inscribed_circle_radius_squared_l618_618515

theorem inscribed_circle_radius_squared 
  (X Y Z W R S : Type) 
  (XR RY WS SZ : ℝ)
  (hXR : XR = 23) 
  (hRY : RY = 29)
  (hWS : WS = 41) 
  (hSZ : SZ = 31)
  (tangent_at_XY : true) (tangent_at_WZ : true) -- since tangents are assumed by problem
  : ∃ (r : ℝ), r^2 = 905 :=
by sorry

end inscribed_circle_radius_squared_l618_618515


namespace exponent_calculation_l618_618565

-- Define the necessary exponents and base
def base : ℕ := 19
def exp1 : ℕ := 11
def exp2 : ℕ := 8

-- Given condition
lemma power_property (a : ℕ) (m n : ℕ) : a^m / a^n = a^(m - n) := by sorry

-- Proof that 19^11 / 19^8 = 6859
theorem exponent_calculation : base^exp1 / base^exp2 = 6859 := by
  have : base^exp1 / base^exp2 = base^(exp1 - exp2) := power_property base exp1 exp2
  have : base^(exp1 - exp2) = base^3 := by rw [nat.sub_eq_iff_eq_add.mpr rfl]
  have : base^3 = 6859 := by -- This would be an arithmetic computation
    rfl -- Placeholder for the actual arithmetic; ideally, you'd verify this step.
  sorry

end exponent_calculation_l618_618565


namespace quadratic_two_distinct_roots_l618_618439

theorem quadratic_two_distinct_roots :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 2 * x1^2 - 3 = 0 ∧ 2 * x2^2 - 3 = 0) :=
by
  sorry

end quadratic_two_distinct_roots_l618_618439


namespace women_in_business_class_l618_618840

theorem women_in_business_class (total_passengers : ℕ) (percentage_women percentage_business_class : ℝ) :
  total_passengers = 300 →
  percentage_women = 0.7 →
  percentage_business_class = 0.15 →
  (total_passengers * percentage_women * percentage_business_class).round = 32 :=
by
  intros h_total h_women h_business
  sorry

end women_in_business_class_l618_618840


namespace baseball_card_decrease_l618_618921

theorem baseball_card_decrease :
  ∀ (V0 : ℝ) (r : ℝ), r = 0.1 → 
  let V1 := V0 * (1 - r) in
  let V2 := V1 * (1 - r) in
  let decrease := (V0 - V2) / V0 in
  decrease = 0.19 :=
by {
  intros V0 r hr,
  let V1 := V0 * (1 - r),
  let V2 := V1 * (1 - r),
  let decrease := (V0 - V2) / V0,
  sorry
}

end baseball_card_decrease_l618_618921


namespace triangle_reflection_area_l618_618469

variable (A B C A' B' C' : Type)
variable [PlaneGeometry A B C]
variable [ReflectionPoint A B C A' B' C']

theorem triangle_reflection_area {ABC_area : ℝ} {A'B'C'_area : ℝ} :
  reflection_triangle_area A B C A' B' C' A'B'C'_area ABC_area ∧ 
  reflection_triangle_smaller_than_five_times_initial (triangle_area A B C ABC_area) :=
sorry

end triangle_reflection_area_l618_618469


namespace num_planes_determined_l618_618050

-- Definitions: Representing the conditions in the problem
def Line (α : Type) := set α -- Define what a line is in terms of a set of points (assuming Type α)
variables {α : Type} (l₁ l₂ : Line α) [parallel : l₁ || l₂]
variables (p₁ p₂ p₃ : α) (q₁ q₂ : α)
variables (on_l1 : p₁ ∈ l₁ ∧ p₂ ∈ l₁ ∧ p₃ ∈ l₁) (on_l2 : q₁ ∈ l₂ ∧ q₂ ∈ l₂)

-- Theorem: The number of planes determined by these points
theorem num_planes_determined : 
  ∃! plane : set α, (p₁ ∈ plane ∧ p₂ ∈ plane ∧ p₃ ∈ plane ∧ q₁ ∈ plane ∧ q₂ ∈ plane) := 
by
  sorry

end num_planes_determined_l618_618050


namespace find_a_b_find_T_l618_618113

variable (a b : ℕ → ℝ) (a_val b_val : ℝ) (a_n : ℕ)
variable (b_n : ℕ → ℝ)

-- Given conditions
axiom a_pos (n : ℕ) : 0 < a n
axiom Sn_def (n : ℕ) : S n = ∑ i in range n, a i
axiom S1_eq : S 1 = 2
axiom a7_eq : a 6 = 20 -- Due to 0 indexing in Lean, a_7 is written as a 6
axiom main_condition (n : ℕ) : 2 * (a_val + b_val) * S n = (a n + a_val) * (a n + b_val)
axiom b_gt : b_val > (3 / 2)
axiom a_gt : (3 / 2) > a_val

theorem find_a_b : a_val = 1 ∧ b_val = 2 :=
sorry

-- For second problem conditions
noncomputable def b_def (n : ℕ) : ℝ := (a n + 1) / (3 * 2^n)
noncomputable def T (n : ℕ) : ℝ := ∑ i in range n, b_def i

theorem find_T (n : ℕ) : T n = 2 - (2 + n) / 2^n :=
sorry

end find_a_b_find_T_l618_618113


namespace imaginary_part_of_z_l618_618205

def z : Complex := Complex.i / (1 - Complex.i)

theorem imaginary_part_of_z : z.im = 1 / 2 :=
by
  sorry

end imaginary_part_of_z_l618_618205


namespace min_val_inequality_no_exist_inequality_l618_618676

noncomputable def minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (a + b) * sqrt (a * b) = 1) : ℝ :=
  if h' : a = b then 4 * sqrt 2 else 999999 -- placeholder for the actual calculation

theorem min_val_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (a + b) * sqrt (a * b) = 1) :
  minimum_value a b ha hb h = 4 * sqrt 2 :=
sorry

theorem no_exist_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ¬ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1 / (2 * a) + 1 / (3 * b) = sqrt 6 / 3 :=
sorry

end min_val_inequality_no_exist_inequality_l618_618676


namespace decreasing_function_in_interval_l618_618135

open Real

theorem decreasing_function_in_interval (f : ℝ → ℝ) (interval : Set ℝ) :
  (interval = Ioo 0 1) →
  (f = λ x, 1 / x ∨ f = log x ∨ f = λ x, 2 ^ x ∨ f = λ x, x ^ (1 / 3)) →
  (∀ x ∈ interval, f' x < 0 ↔ f = λ x, 1 / x) :=
by sorry

end decreasing_function_in_interval_l618_618135


namespace find_hyperbola_m_l618_618699

theorem find_hyperbola_m (m : ℝ) (h : m > 0) : 
  (∀ x y : ℝ, (x^2 / m - y^2 / 3 = 1 → y = 1 / 2 * x)) → m = 12 :=
by
  intros
  sorry

end find_hyperbola_m_l618_618699


namespace g_f_eval_l618_618368

def f (x : ℤ) := x^3 - 2
def g (x : ℤ) := 3 * x^2 + x + 2

theorem g_f_eval : g (f 3) = 1902 := by
  sorry

end g_f_eval_l618_618368


namespace function_order_l618_618202

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x
noncomputable def f_prime (x : ℝ) : ℝ := (1 - Real.log x) / (x ^ 2)

theorem function_order (h₁ : ∀ x > 0, f' x = 1 / (x ^ 2) - (Real.log x) / (x ^ 2)) :
  f e > f 3 ∧ f 3 > f 2 := by
  have h₂ : ∀ x > 0, f_prime x > 0 ↔ x ∈ set.Ioo 0 e := sorry
  have h₃ : ∀ x > e, f_prime x < 0 := sorry
  sorry

end function_order_l618_618202


namespace exists_set_B_l618_618790

theorem exists_set_B (n : ℕ) (A : Finset (Zmod (n^2))) (hA : A.card = n) :
  ∃ B : Finset (Zmod (n^2)), B.card = n ∧ 
  ∀ r : Zmod (n^2), (∃ a ∈ A, ∃ b ∈ B, a + b = r) → r ∈ (Finset.univ : Finset (Zmod (n^2))) :=
sorry

end exists_set_B_l618_618790


namespace number_of_orange_marbles_l618_618461

/--
There are 24 marbles in a jar. Half are blue. There are 6 red marbles.
The rest of the marbles are orange.
Prove that the number of orange marbles is 6.
-/
theorem number_of_orange_marbles :
  ∀ (total_marbles blue_marbles red_marbles orange_marbles : ℕ),
  total_marbles = 24 → 
  blue_marbles = total_marbles / 2 →
  red_marbles = 6 → 
  orange_marbles = total_marbles - (blue_marbles + red_marbles) →
  orange_marbles = 6 :=
by 
  intros total_marbles blue_marbles red_marbles orange_marbles h_total h_blue h_red h_orange 
  rw [h_total, h_blue, h_red, h_orange]
  norm_num
  rw [nat.div_self] 
  reflexivity sorry

end number_of_orange_marbles_l618_618461


namespace age_double_after_5_years_l618_618897

-- Defining the current ages of the brothers
def older_brother_age := 15
def younger_brother_age := 5

-- Defining the condition
def after_x_years (x : ℕ) := older_brother_age + x = 2 * (younger_brother_age + x)

-- The main theorem with the condition
theorem age_double_after_5_years : after_x_years 5 :=
by sorry

end age_double_after_5_years_l618_618897


namespace remainder_of_3_pow_19_mod_10_l618_618065

-- Definition of the problem and conditions
def q := 3^19

-- Statement to prove
theorem remainder_of_3_pow_19_mod_10 : q % 10 = 7 :=
by
  sorry

end remainder_of_3_pow_19_mod_10_l618_618065


namespace two_pair_probability_l618_618058

theorem two_pair_probability (total_cards : ℕ)
  (ranks : ℕ)
  (cards_per_rank : ℕ)
  (choose_five : ℕ → ℕ → ℕ := Nat.choose)
  (S : ℕ := 13 * (choose_five 4 2) * 12 * (choose_five 4 2) * 11 * (choose_five 4 1))
  (N : ℕ := choose_five 52 5) :
  total_cards = 52 → ranks = 13 → cards_per_rank = 4 → S = 247104 → N = 2598960 →
  (S / N : ℚ) = 95/999 := 
by
  intros
  sorry

end two_pair_probability_l618_618058


namespace symmetric_difference_associative_l618_618649

variable {α : Type*} -- We assume α is a type, and A, B, C are subsets of α
variables A B C : Set α

theorem symmetric_difference_associative : 
  (A ∆ B) ∆ C = A ∆ (B ∆ C) := 
sorry

end symmetric_difference_associative_l618_618649


namespace max_initial_number_l618_618329

noncomputable def verify_addition (n x : ℕ) : Prop := 
  ∀ (a : ℕ), (a ∣ n) → (a ≠ 1) → (n + a = x) → False

theorem max_initial_number :
  ∃ (n : ℕ), 
  (∀ (a1 a2 a3 a4 a5 : ℕ), 
    verify_addition n a1 ∧ verify_addition (n + a1) a2 ∧
    verify_addition (n + a1 + a2) a3 ∧ verify_addition (n + a1 + a2 + a3) a4 ∧
    verify_addition (n + a1 + a2 + a3 + a4) a5 ∧
    (n + a1 + a2 + a3 + a4 + a5 = 200)) ∧
  (∀ m : ℕ, 
    (∃ (a1 a2 a3 a4 a5 : ℕ), 
      verify_addition m a1 ∧ verify_addition (m + a1) a2 ∧
      verify_addition (m + a1 + a2) a3 ∧ verify_addition (m + a1 + a2 + a3) a4 ∧
      verify_addition (m + a1 + a2 + a3 + a4) a5 ∧
      (m + a1 + a2 + a3 + a4 + a5 = 200)) →
    m ≤ 189)
: ∃ n, n = 189 := by
  sorry

end max_initial_number_l618_618329


namespace find_constants_l618_618626

theorem find_constants (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 5 → (x^2 - 7) / ((x - 2) * (x - 3) * (x - 5)) = A / (x - 2) + B / (x - 3) + C / (x - 5))
  ↔ (A = -1 ∧ B = -1 ∧ C = 3) :=
by
  sorry

end find_constants_l618_618626


namespace find_x_minus_y_l618_618203

/-
Given that:
  2 * x + y = 7
  x + 2 * y = 8
We want to prove:
  x - y = -1
-/

theorem find_x_minus_y (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : x + 2 * y = 8) : x - y = -1 :=
by
  sorry

end find_x_minus_y_l618_618203


namespace log_fraction_simplify_l618_618169

theorem log_fraction_simplify :
  (3 / log 8 (5000 ^ 5) + 4 / log 9 (5000 ^ 5) = 1 / 5) :=
by
  sorry

end log_fraction_simplify_l618_618169


namespace polyhedron_vertex_assignment_l618_618758

-- Define a polyhedron graph structure
structure Polyhedron (V : Type) :=
  (edges : V → V → Prop)
  (symm_edges : ∀ {v w : V}, edges v w → edges w v)
  (loopless : ∀ v : V, ¬ edges v v)

-- Define the problem statement in Lean 4
theorem polyhedron_vertex_assignment (V : Type) (P : Polyhedron V) :
  ∃ f : V → ℕ, 
    (∀ v, 0 < f v) ∧ 
    (∀ v w, gcd (f v) (f w) = 1 ↔ P.edges v w) :=
by 
  sorry

end polyhedron_vertex_assignment_l618_618758


namespace units_digit_of_52_cubed_plus_29_cubed_l618_618475

-- Define the units digit of a number n
def units_digit (n : ℕ) : ℕ := n % 10

-- Given conditions as definitions in Lean
def units_digit_of_2_cubed : ℕ := units_digit (2^3)  -- 8
def units_digit_of_9_cubed : ℕ := units_digit (9^3)  -- 9

-- The main theorem to prove
theorem units_digit_of_52_cubed_plus_29_cubed : units_digit (52^3 + 29^3) = 7 :=
by
  sorry

end units_digit_of_52_cubed_plus_29_cubed_l618_618475


namespace fruit_salad_servings_l618_618949

theorem fruit_salad_servings :
  let cantaloupe_chunks := 30,
      honeydew_chunks := 42,
      pineapple_chunks := 12,
      watermelon_chunks := 56,
      ratio_cantaloupe := 3,
      ratio_honeydew := 2,
      ratio_pineapple := 1,
      ratio_watermelon := 4 in
  min (cantaloupe_chunks / ratio_cantaloupe)
      (min (honeydew_chunks / ratio_honeydew)
           (min (pineapple_chunks / ratio_pineapple)
                (watermelon_chunks / ratio_watermelon))) = 10 := by
  sorry

end fruit_salad_servings_l618_618949


namespace ball_box_problem_l618_618822

theorem ball_box_problem : 
  let balls := {1, 2, 3, 4} in
  let boxes := {1, 2, 3} in 
  (∃ f : balls → boxes, function.surjective f) ∧ (set.univ.bij_on_subtype fun (b : balls) => 4) = 36 :=
sorry

end ball_box_problem_l618_618822


namespace largest_subset_no_four_times_another_l618_618963

theorem largest_subset_no_four_times_another :
  ∃ (S : set ℕ), S ⊆ {1, 2, ..., 150} ∧ (∀ (a b : ℕ), a ∈ S → b ∈ S → (a ≠ 4 * b ∧ b ≠ 4 * a)) ∧ (S.card = 141) :=
sorry

end largest_subset_no_four_times_another_l618_618963


namespace part1_part2_l618_618197

theorem part1 (α : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi) (h_trig : Real.sin α + Real.cos α = 1 / 5) :
  Real.sin α - Real.cos α = 7 / 5 := sorry

theorem part2 (α : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi) (h_trig : Real.sin α + Real.cos α = 1 / 5) :
  Real.sin (2 * α + Real.pi / 3) = -12 / 25 - 7 * Real.sqrt 3 / 50 := sorry

end part1_part2_l618_618197


namespace school_points_l618_618891

theorem school_points (a b c : ℕ) (h1 : a + b + c = 285)
  (h2 : ∃ x : ℕ, a - 8 = x ∧ b - 12 = x ∧ c - 7 = x) : a + c = 187 :=
sorry

end school_points_l618_618891


namespace one_hundred_fifth_does_not_contain_five_l618_618595

def rising_number (n : ℕ) : Prop := 
  ∀ i j : ℕ, (1 ≤ i ∧ i < j ∧ j ≤ 4) → (nat.digit_of_num n i < nat.digit_of_num n j)

def count_rising_numbers : ℕ := nat.choose 9 4

def nth_rising_number (n : ℕ) : ℕ := sorry  -- Placeholder for the correct implementation

theorem one_hundred_fifth_does_not_contain_five :
  rising_number (nth_rising_number 105) → 
  count_rising_numbers = 126 → 
  ∀ d, d ∈ nat.digits 10 (nth_rising_number 105) → d ≠ 5 :=
begin
  sorry
end

end one_hundred_fifth_does_not_contain_five_l618_618595


namespace number_of_orange_marbles_l618_618460

/--
There are 24 marbles in a jar. Half are blue. There are 6 red marbles.
The rest of the marbles are orange.
Prove that the number of orange marbles is 6.
-/
theorem number_of_orange_marbles :
  ∀ (total_marbles blue_marbles red_marbles orange_marbles : ℕ),
  total_marbles = 24 → 
  blue_marbles = total_marbles / 2 →
  red_marbles = 6 → 
  orange_marbles = total_marbles - (blue_marbles + red_marbles) →
  orange_marbles = 6 :=
by 
  intros total_marbles blue_marbles red_marbles orange_marbles h_total h_blue h_red h_orange 
  rw [h_total, h_blue, h_red, h_orange]
  norm_num
  rw [nat.div_self] 
  reflexivity sorry

end number_of_orange_marbles_l618_618460


namespace find_z_find_magnitude_z_l618_618230

def z : ℂ := (1 - I) / (1 + I) + 2 * I

theorem find_z : z = I := 
by
  sorry

theorem find_magnitude_z : Complex.abs z = 1 :=
by
  sorry

end find_z_find_magnitude_z_l618_618230


namespace num_non_phd_scientists_l618_618143

-- Total number of participants
def num_participants (n : ℕ) : Prop :=
  198 < n ∧ n < 230

-- Each participant plays exactly once against every other participant
def total_battles (n : ℕ) : ℕ := n * (n - 1) / 2

-- Function to calculate points in battles
def points_scored (n m : ℕ) : ℕ :=
  m * (m - 1) / 2 + (n - m) * (n - m - 1) / 2

-- Given condition: each participant scores half of their points against PhDs
def points_condition (n m : ℕ) : Prop := 
  2 * (points_scored n m) = total_battles n

-- Proof that the smallest number of non-PhD scientists is 105
theorem num_non_phd_scientists (n m : ℕ) (h : num_participants n) (p : points_condition n m) : (n - m) = 105 :=
by
sory

end num_non_phd_scientists_l618_618143


namespace num_non_phd_scientists_l618_618142

-- Total number of participants
def num_participants (n : ℕ) : Prop :=
  198 < n ∧ n < 230

-- Each participant plays exactly once against every other participant
def total_battles (n : ℕ) : ℕ := n * (n - 1) / 2

-- Function to calculate points in battles
def points_scored (n m : ℕ) : ℕ :=
  m * (m - 1) / 2 + (n - m) * (n - m - 1) / 2

-- Given condition: each participant scores half of their points against PhDs
def points_condition (n m : ℕ) : Prop := 
  2 * (points_scored n m) = total_battles n

-- Proof that the smallest number of non-PhD scientists is 105
theorem num_non_phd_scientists (n m : ℕ) (h : num_participants n) (p : points_condition n m) : (n - m) = 105 :=
by
sory

end num_non_phd_scientists_l618_618142


namespace extreme_value_at_1_imp_a_eq_one_third_l618_618694

theorem extreme_value_at_1_imp_a_eq_one_third
  (a : ℝ) (f : ℝ → ℝ) (h : f = λ x, (a * x ^ 2 - 1) * real.exp x)
  (h_extreme : ∃ c, c = 1 ∧ ∀ x, 0 ≤ x - c → f x - f c = 0) :
  a = 1 / 3 :=
by
  sorry

end extreme_value_at_1_imp_a_eq_one_third_l618_618694


namespace total_age_l618_618922

-- Define the ages of a, b, and c based on the conditions given
variables (a b c : ℕ)

-- Condition 1: a is two years older than b
def age_condition1 := a = b + 2

-- Condition 2: b is twice as old as c
def age_condition2 := b = 2 * c

-- Condition 3: b is 12 years old
def age_condition3 := b = 12

-- Prove that the total of the ages of a, b, and c is 32 years
theorem total_age : age_condition1 → age_condition2 → age_condition3 → a + b + c = 32 :=
by
  intros h1 h2 h3 
  -- Proof would go here
  sorry

end total_age_l618_618922


namespace total_cost_of_crayons_l618_618764

theorem total_cost_of_crayons (crayons_per_half_dozen : ℕ)
    (number_of_half_dozens : ℕ)
    (cost_per_crayon : ℕ)
    (total_cost : ℕ) :
  crayons_per_half_dozen = 6 →
  number_of_half_dozens = 4 →
  cost_per_crayon = 2 →
  total_cost = crayons_per_half_dozen * number_of_half_dozens * cost_per_crayon →
  total_cost = 48 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end total_cost_of_crayons_l618_618764


namespace length_of_second_train_is_correct_l618_618901

noncomputable def length_of_second_train 
  (speed_first_train_kmph : ℝ) 
  (speed_second_train_kmph : ℝ) 
  (length_first_train_m : ℝ) 
  (time_cross_s : ℝ) : ℝ :=
  let relative_speed_mps := (speed_first_train_kmph + speed_second_train_kmph) * (1000 / 3600) in
  let total_length_m := relative_speed_mps * time_cross_s in
  total_length_m - length_first_train_m

theorem length_of_second_train_is_correct 
  (speed_first_train_kmph : ℝ) 
  (speed_second_train_kmph : ℝ) 
  (length_first_train_m : ℝ) 
  (time_cross_s : ℝ) 
  (h_first_train_speed : speed_first_train_kmph = 60) 
  (h_second_train_speed : speed_second_train_kmph = 90) 
  (h_length_first_train : length_first_train_m = 1100) 
  (h_time_cross : time_cross_s = 47.99999999999999) : 
  length_of_second_train speed_first_train_kmph speed_second_train_kmph length_first_train_m time_cross_s = 900 :=
by
  sorry

end length_of_second_train_is_correct_l618_618901


namespace ships_meeting_count_l618_618817

theorem ships_meeting_count :
  ∀ (n : ℕ) (east_sailing west_sailing : ℕ),
    n = 10 →
    east_sailing = 5 →
    west_sailing = 5 →
    east_sailing + west_sailing = n →
    (∀ (v : ℕ), v > 0) →
    25 = east_sailing * west_sailing :=
by
  intros n east_sailing west_sailing h1 h2 h3 h4 h5
  sorry

end ships_meeting_count_l618_618817


namespace ratio_a3_a4_l618_618300

noncomputable def a : ℕ → ℝ
| 0     := 1  -- Since a₁ = 1
| (n+1) := if n = 0 then 1 else (a n + (-1)^(n + 1)) / a n

theorem ratio_a3_a4 : a 2 / a 3 = 1 / 6 :=
by
  have h1 : a 1 = 1 := rfl
  have h2 : a 2 = 2 := by simp [a, h1]
  have h3 : a 3 = 1 / 2 := by simp [a, h2]
  have h4 : a 4 = 3 := by simp [a, h3]
  simp [h3, h4]
  sorry

end ratio_a3_a4_l618_618300


namespace red_grapes_in_salad_l618_618289

theorem red_grapes_in_salad {G R B : ℕ} 
  (h1 : R = 3 * G + 7)
  (h2 : B = G - 5)
  (h3 : G + R + B = 102) : R = 67 :=
sorry

end red_grapes_in_salad_l618_618289


namespace train_speed_l618_618116

theorem train_speed (lt_train : ℝ) (lt_bridge : ℝ) (time_cross : ℝ) (total_speed_kmph : ℝ) :
  lt_train = 150 ∧ lt_bridge = 225 ∧ time_cross = 30 ∧ total_speed_kmph = (375 / 30) * 3.6 → 
  total_speed_kmph = 45 := 
by
  sorry

end train_speed_l618_618116


namespace rhombus_area_l618_618497

/-- Given a rhombus with vertices (0, 3.5), (12, 0), (0, -3.5), and (-12, 0),
prove that its area is 84 square units. -/
theorem rhombus_area (A B C D : ℝ × ℝ) 
  (hA : A = (0, 3.5)) 
  (hB : B = (12, 0)) 
  (hC : C = (0, -3.5)) 
  (hD : D = (-12, 0)) : 
  let d1 := 7
      d2 := 24 in
  (d1 * d2) / 2 = 84 :=
by 
  sorry

end rhombus_area_l618_618497


namespace problem_a_problem_b_problem_c_problem_d_l618_618841

noncomputable def f (x : ℝ) : ℝ := cos x + (cos (2 * x)) / 2 + (cos (4 * x)) / 4

theorem problem_a : (∀ x : ℝ, f (-x) = f x) :=
sorry

theorem problem_b : (∀ x : ℝ, f (2 * π - x) = f x) :=
sorry

theorem problem_c : ¬ (∀ x : ℝ, f (x + π) = f x) :=
sorry

theorem problem_d : (∀ x : ℝ, f' x < 3) :=
sorry

end problem_a_problem_b_problem_c_problem_d_l618_618841


namespace train_speed_is_correct_l618_618539

-- Definitions for conditions
def train_length : ℝ := 150  -- length of the train in meters
def time_to_cross_pole : ℝ := 3  -- time to cross the pole in seconds

-- Proof statement
theorem train_speed_is_correct : (train_length / time_to_cross_pole) = 50 := by
  sorry

end train_speed_is_correct_l618_618539


namespace valid_outfits_number_l618_618713

def num_shirts := 7
def num_pants := 7
def num_hats := 7
def num_colors := 7

def total_outfits (num_shirts num_pants num_hats : ℕ) := num_shirts * num_pants * num_hats
def matching_color_outfits (num_colors : ℕ) := num_colors
def valid_outfits (num_shirts num_pants num_hats num_colors : ℕ) := 
  total_outfits num_shirts num_pants num_hats - matching_color_outfits num_colors

theorem valid_outfits_number : valid_outfits num_shirts num_pants num_hats num_colors = 336 := 
by
  sorry

end valid_outfits_number_l618_618713


namespace probability_of_both_l618_618496

variable (A B : Prop)

-- Assumptions
def p_A : ℝ := 0.55
def p_B : ℝ := 0.60

-- Probability of both A and B telling the truth at the same time
theorem probability_of_both : p_A * p_B = 0.33 := by
  sorry

end probability_of_both_l618_618496


namespace quadratic_equation_in_terms_of_x_l618_618480

-- Define what it means to be a quadratic equation in terms of x.
def is_quadratic (eq : ℕ → ℕ → ℕ → Prop) : Prop :=
  ∃ a b c : ℕ, a ≠ 0 ∧ ∀ x : ℕ, eq x = a * x^2 + b * x + c

-- Define each condition as an equation.
def equation_A (x : ℕ) : ℕ := x - 1 / x + 2

def equation_B (x y : ℕ) : ℕ := x^2 + 2 * x + y

def equation_C (a b c x : ℕ ) : ℕ := a * x^2 + b * x + c

def equation_D (x : ℕ) : ℕ := x^2 - x + 1

theorem quadratic_equation_in_terms_of_x : is_quadratic equation_D :=
sorry

end quadratic_equation_in_terms_of_x_l618_618480


namespace who_made_a_mistake_l618_618036

-- Definitions of the conditions
def at_least_four_blue_pencils (B : ℕ) : Prop := B ≥ 4
def at_least_five_green_pencils (G : ℕ) : Prop := G ≥ 5
def at_least_three_blue_and_four_green_pencils (B G : ℕ) : Prop := B ≥ 3 ∧ G ≥ 4
def at_least_four_blue_and_four_green_pencils (B G : ℕ) : Prop := B ≥ 4 ∧ G ≥ 4

-- The main theorem stating who made a mistake
theorem who_made_a_mistake (B G : ℕ) 
  (hv : at_least_four_blue_pencils B)
  (hk : at_least_five_green_pencils G)
  (hp : at_least_three_blue_and_four_green_pencils B G)
  (hm : at_least_four_blue_and_four_green_pencils B G) 
  (h_truth : (hv ∧ hk ∧ hp ∧ hm) ∨ (¬hv ∧ hk ∧ hp ∧ hm) ∨ (hv ∧ ¬hk ∧ hp ∧ hm) ∨ (hv ∧ hk ∧ ¬hp ∧ hm) ∨ (hv ∧ hk ∧ hp ∧ ¬hm))
  (h_truthful: ∑ b in [hv, hk, hp, hm], (if b then 1 else 0) = 3) : 
  hk = false := 
sorry

end who_made_a_mistake_l618_618036


namespace largest_initial_number_l618_618320

theorem largest_initial_number :
  ∃ n : ℕ, 
    (∀ (a1 a2 a3 a4 a5 : ℕ),
      (n + a1).gcd a1 = 1 ∧
      (n + a1 + a2).gcd a2 = 1 ∧
      (n + a1 + a2 + a3).gcd a3 = 1 ∧
      (n + a1 + a2 + a3 + a4).gcd a4 = 1 ∧
      (n + a1 + a2 + a3 + a4 + a5).gcd a5 = 1 ∧
      n + a1 + a2 + a3 + a4 + a5 = 200) 
    → n = 189 :=
begin
  sorry
end

end largest_initial_number_l618_618320


namespace who_made_mistake_l618_618012

-- Defining conditions for the colored pencils
def has_at_least_four_blue_pencils (b : Nat) : Prop := b >= 4
def has_at_least_five_green_pencils (g : Nat) : Prop := g >= 5
def has_at_least_three_blue_four_green_pencils (b g : Nat) : Prop := b >= 3 ∧ g >= 4
def has_at_least_four_blue_four_green_pencils (b g : Nat) : Prop := b >= 4 ∧ g >= 4

-- Statement of the problem
theorem who_made_mistake (b g : Nat) (vasya kolya petya misha : Prop) :
  has_at_least_four_blue_pencils b →
  has_at_least_five_green_pencils g →
  has_at_least_three_blue_four_green_pencils b g →
  has_at_least_four_blue_four_green_pencils b g →
  (∃ T : Set Prop, {vasya, kolya, petya, misha}.Erase T = {vasya, kolya, petya, misha} ∧
    T.Card = 3) →
  (kolya ↔ ¬ g >= 5) := 
sorry

end who_made_mistake_l618_618012


namespace book_price_l618_618393

theorem book_price (x : ℕ) : 
  9 * x ≤ 1100 ∧ 13 * x ≤ 1500 → x = 123 :=
sorry

end book_price_l618_618393


namespace arithmetic_mean_of_pairs_l618_618811

theorem arithmetic_mean_of_pairs :
  let a := (7 : ℚ) / 8
  let b := (9 : ℚ) / 10
  let c := (4 : ℚ) / 5
  let d := (17 : ℚ) / 20
  (a, b, c, d) ∈ { (7/8, 9/10, 4/5, 17/20), (9/10, 4/5, 17/20, 7/8), (4/5, 17/20, 7/8, 9/10), (17/20, 7/8, 9/10, 4/5) }
  → 2 * d = b + c :=
begin
  intros a b c d h,
  change (2 * (17/20) = (9/10) + (4/5)) at h,
  linarith,
end

end arithmetic_mean_of_pairs_l618_618811


namespace fractions_arithmetic_lemma_l618_618173

theorem fractions_arithmetic_lemma : (8 / 15 : ℚ) - (7 / 9) + (3 / 4) = 1 / 2 := 
by
  sorry

end fractions_arithmetic_lemma_l618_618173


namespace largest_subset_size_l618_618978

-- Definition of the subset condition
def valid_subset (s : Set ℕ) : Prop :=
  ∀ (a ∈ s) (b ∈ s), a ≠ 4 * b

-- Setting the range
def range_set : Set ℕ := { n | 1 ≤ n ∧ n ≤ 150 }

-- The main theorem statement
theorem largest_subset_size : ∃ (s : Set ℕ), s ⊆ range_set ∧ valid_subset (s) ∧ #(s) = 150 :=
by
  sorry

end largest_subset_size_l618_618978


namespace largest_initial_number_l618_618333

-- Let's define the conditions and the result
def valid_addition (n a : ℕ) : Prop := ∃ k : ℕ, n = a * k + r ∧ 0 < r ∧ r < a

def valid_operations (initial : ℕ) (final : ℕ) (steps : ℕ → ℕ → ℕ) : Prop :=
  ∃ (a b c d e : ℕ), valid_addition initial a ∧
                      valid_addition (initial + a) b ∧
                      valid_addition (initial + a + b) c ∧
                      valid_addition (initial + a + b + c) d ∧
                      valid_addition (initial + a + b + c + d) e ∧
                      initial + a + b + c + d + e = final

theorem largest_initial_number :
  ∃ n : ℕ, (valid_operations n 200 (λn a, n + a)) ∧ (∀ m : ℕ, valid_operations m 200 (λn a, n + a) → m ≤ n) :=
sorry

end largest_initial_number_l618_618333


namespace geometric_sequence_seventh_term_l618_618639

theorem geometric_sequence_seventh_term : 
  let a := 5
  let r := -⅕
  a * r^(7 - 1) = ⅓125 := 
by
  sorry

end geometric_sequence_seventh_term_l618_618639


namespace physics_marks_l618_618075

theorem physics_marks (P C M : ℕ) (h1 : P + C + M = 180) (h2 : P + M = 180) (h3 : P + C = 140) : P = 140 :=
by
  sorry

end physics_marks_l618_618075


namespace binomial_sum_l618_618606

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_sum (n : ℤ) (h1 : binomial 25 n.natAbs + binomial 25 12 = binomial 26 13 ∧ n ≥ 0) : 
    (n = 12 ∨ n = 13) → n.succ + n = 25 := 
    sorry

end binomial_sum_l618_618606


namespace retail_price_l618_618074

theorem retail_price (R : ℝ) (wholesale_price : ℝ)
  (discount_rate : ℝ) (profit_rate : ℝ)
  (selling_price : ℝ) :
  wholesale_price = 81 →
  discount_rate = 0.10 →
  profit_rate = 0.20 →
  selling_price = wholesale_price * (1 + profit_rate) →
  selling_price = R * (1 - discount_rate) →
  R = 108 := 
by 
  intros h_wholesale h_discount h_profit h_selling_price h_discounted_selling_price
  sorry

end retail_price_l618_618074


namespace circumradius_of_tetrahedron_is_3_l618_618960

open Real -- Open the Real namespace for trigonometric functions and square roots

namespace Geometry

def is_tangent_to_plane (s : Sphere) (p : Plane) (D : Point) : Prop :=
  -- Define what it means for a sphere to be tangent to a plane at a point D
  sorry

def is_tangent_to_sphere (s1 s2 : Sphere) (D : Point) : Prop :=
  -- Define what it means for a sphere to be tangent to another sphere at a point D
  sorry

noncomputable def circumradius_of_tetrahedron_3 (A B C D : Point) : ℝ :=
  -- Define the circumradius of a tetrahedron given its vertices
  sorry

noncomputable def radius_of_circumscribed_sphere_3 (A B C D : Point) : ℝ :=
  -- Define the radius of the circumscribed sphere for tetrahedron ABCD
  circumradius_of_tetrahedron_3 A B C D

theorem circumradius_of_tetrahedron_is_3 (A B C D : Point)
  (h1 : AD A D = 2 * sqrt 3)
  (h2 : ∠BAC A B C = 60 * Real.pi / 180)
  (h3 : ∠BAD A B D = 45 * Real.pi / 180)
  (h4 : ∠CAD C A D = 45 * Real.pi / 180)
  (h5 : ∃ (s : Sphere), is_tangent_to_sphere s (circumsphere A B C D) D ∧ radius s = 1)
  (h6 : ∃ (plane_ABC : Plane), is_tangent_to_plane (sphere A D h5) plane_ABC D) :
  radius_of_circumscribed_sphere_3 A B C D = 3 :=
begin
  -- Exact proof formulation
  sorry
end

end Geometry

end circumradius_of_tetrahedron_is_3_l618_618960


namespace parking_garage_capacity_l618_618106

open Nat

-- Definitions from the conditions
def first_level_spaces : Nat := 90
def second_level_spaces : Nat := first_level_spaces + 8
def third_level_spaces : Nat := second_level_spaces + 12
def fourth_level_spaces : Nat := third_level_spaces - 9
def initial_parked_cars : Nat := 100

-- The proof statement
theorem parking_garage_capacity : 
  (first_level_spaces + second_level_spaces + third_level_spaces + fourth_level_spaces - initial_parked_cars) = 299 := 
  by 
    sorry

end parking_garage_capacity_l618_618106


namespace minimal_k_to_unique_determine_tromino_l618_618404

-- Definitions
def Board (n : Nat) := List (List (Option Bool))
def Position := Nat × Nat   -- Position on the board as (row, column)

-- Constants given the problem
def boardSize : Nat := 9
def trominoSize : Nat := 3

-- Conditions for the game
def marks_cells (board : Board boardSize) (k : Nat) : Prop :=
  ∃ marked : List Position, marked.length = k ∧ ∀ p ∈ marked, board.nth p.fst = some (List.replicate boardSize (some true)).nth p.snd

-- Definition of tromino placements and winning condition
def tromino_placements (board : Board boardSize) :=
  List Position -- List of positions that define possible placements of trominoes

def can_determine_tromino (board : Board boardSize) (tromino : List Position) : Prop :=
  ∀ t1 t2 ∈ tromino_placements board, t1 ≠ t2 → ∃ p1 ∈ t1, p2 ∈ t2, board.nth p1.fst ≠ board.nth p2.fst

-- Main theorem statement
theorem minimal_k_to_unique_determine_tromino : ∃ k : Nat, marks_cells (List.replicate boardSize (List.replicate boardSize none)) k ∧ k = 68 ∧
  ∀ (board : Board boardSize) (tromino : List Position), tromino_placements board → can_determine_tromino board tromino :=
sorry

end minimal_k_to_unique_determine_tromino_l618_618404


namespace solution_to_equation_l618_618177

theorem solution_to_equation :
  ∃ x : ℝ, x = (11 - 3 * Real.sqrt 5) / 2 ∧ x^2 + 6 * x + 6 * x * Real.sqrt (x + 4) = 31 :=
by
  sorry

end solution_to_equation_l618_618177


namespace total_ages_is_32_l618_618925

variable (a b c : ℕ)
variable (h_b : b = 12)
variable (h_a : a = b + 2)
variable (h_c : b = 2 * c)

theorem total_ages_is_32 (h_b : b = 12) (h_a : a = b + 2) (h_c : b = 2 * c) : a + b + c = 32 :=
by
  sorry

end total_ages_is_32_l618_618925


namespace complex_division_correct_l618_618584

theorem complex_division_correct : (3 - 1 * Complex.I) / (1 + Complex.I) = 1 - 2 * Complex.I := 
by
  sorry

end complex_division_correct_l618_618584


namespace launch_country_is_soviet_union_l618_618813

-- Definitions of conditions
def launch_date : String := "October 4, 1957"
def satellite_launched_on (date : String) : Prop := date = "October 4, 1957"
def choices : List String := ["A. United States", "B. Soviet Union", "C. European Union", "D. Germany"]

-- Problem statement
theorem launch_country_is_soviet_union : 
  satellite_launched_on launch_date → 
  "B. Soviet Union" ∈ choices := 
by
  sorry

end launch_country_is_soviet_union_l618_618813


namespace largest_initial_number_l618_618306

theorem largest_initial_number : ∃ n : ℕ, (n + 5 ∑ k : ℕ, k ≠ 0 ∧ ¬ (n % k = 0)) = 200 ∧ n = 189 :=
begin
  sorry
end

end largest_initial_number_l618_618306


namespace problem_inequality_A_problem_inequality_B_problem_inequality_D_problem_inequality_E_l618_618417

variable {a b c : ℝ}

theorem problem_inequality_A (h1 : a > 0) (h2 : a < b) (h3 : b < c) : a * b < b * c :=
by sorry

theorem problem_inequality_B (h1 : a > 0) (h2 : a < b) (h3 : b < c) : a * c < b * c :=
by sorry

theorem problem_inequality_D (h1 : a > 0) (h2 : a < b) (h3 : b < c) : a + b < b + c :=
by sorry

theorem problem_inequality_E (h1 : a > 0) (h2 : a < b) (h3 : b < c) : c / a > 1 :=
by sorry

end problem_inequality_A_problem_inequality_B_problem_inequality_D_problem_inequality_E_l618_618417


namespace cannot_determine_if_counterfeit_coin_is_lighter_or_heavier_l618_618424

/-- 
Vasiliy has 2019 coins, one of which is counterfeit (differing in weight). 
Using balance scales without weights and immediately paying out identified genuine coins, 
it is impossible to determine whether the counterfeit coin is lighter or heavier.
-/
theorem cannot_determine_if_counterfeit_coin_is_lighter_or_heavier 
  (num_coins : ℕ)
  (num_counterfeit : ℕ)
  (balance_scale : Bool → Bool → Bool)
  (immediate_payment : Bool → Bool) :
  num_coins = 2019 →
  num_counterfeit = 1 →
  (∀ coins_w1 coins_w2, balance_scale coins_w1 coins_w2 = (coins_w1 = coins_w2)) →
  (∀ coin_p coin_q, (immediate_payment coin_p = true) → ¬ coin_p = coin_q) →
  ¬ ∃ (is_lighter_or_heavier : Bool), true :=
by
  intro h1 h2 h3 h4
  sorry

end cannot_determine_if_counterfeit_coin_is_lighter_or_heavier_l618_618424


namespace f_x_leq_zero_l618_618232

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x * (1 - x) else x * (1 + x)

theorem f_x_leq_zero (x : ℝ) (h : x ≤ 0) : f x = x * (1 + x) := by
  have odd_property : ∀ x : ℝ, f (-x) = -f (x) := by
    intro x
    by_cases h : x >= 0
    · rw [if_pos h, if_neg (lt_of_not_ge h)]
      ring
    · rw [if_neg h, if_pos (le_of_not_gt h)]
      ring
  rw [if_neg (lt_of_not_ge (not_le_of_lt (lt_of_le_of_ne h (ne_of_lt h).symm)))]
  exact odd_property (-x) ▸ by
    rw [neg_neg, if_pos (neg_nonneg.mpr h)]
    ring

end f_x_leq_zero_l618_618232


namespace kolya_is_wrong_l618_618028

def pencils_problem_statement (at_least_four_blue : Prop) 
                              (at_least_five_green : Prop) 
                              (at_least_three_blue_and_four_green : Prop) 
                              (at_least_four_blue_and_four_green : Prop) : 
                              Prop :=
  ∃ (B G : ℕ), -- B represents the number of blue pencils, G represents the number of green pencils
    ((B ≥ 4) ∧ (G ≥ 4)) ∧ -- Vasya's statement (at least 4 blue), Petya's and Misha's combined statement (at least 4 green)
    at_least_four_blue ∧ -- Vasya's statement (there are at least 4 blue pencils)
    (at_least_five_green ↔ G ≥ 5) ∧ -- Kolya's statement (there are at least 5 green pencils)
    at_least_three_blue_and_four_green ∧ -- Petya's statement (at least 3 blue and 4 green)
    at_least_four_blue_and_four_green -- Misha's statement (at least 4 blue and 4 green)

theorem kolya_is_wrong (at_least_four_blue : Prop) 
                        (at_least_five_green : Prop) 
                        (at_least_three_blue_and_four_green : Prop) 
                        (at_least_four_blue_and_four_green : Prop) : 
                        pencils_problem_statement at_least_four_blue 
                                                  at_least_five_green 
                                                  at_least_three_blue_and_four_green 
                                                  at_least_four_blue_and_four_green :=
sorry

end kolya_is_wrong_l618_618028


namespace wire_ratio_bonnie_roark_l618_618145

-- Definitions from the conditions
def bonnie_wire_length : ℕ := 12 * 8
def bonnie_volume : ℕ := 8 ^ 3
def roark_cube_side : ℕ := 2
def roark_cube_volume : ℕ := roark_cube_side ^ 3
def num_roark_cubes : ℕ := bonnie_volume / roark_cube_volume
def roark_wire_length_per_cube : ℕ := 12 * roark_cube_side
def roark_total_wire_length : ℕ := num_roark_cubes * roark_wire_length_per_cube

-- Statement to prove
theorem wire_ratio_bonnie_roark : 
  ((bonnie_wire_length : ℚ) / roark_total_wire_length) = (1 / 16) :=
by
  sorry

end wire_ratio_bonnie_roark_l618_618145


namespace largest_initial_number_l618_618321

theorem largest_initial_number :
  ∃ n : ℕ, 
    (∀ (a1 a2 a3 a4 a5 : ℕ),
      (n + a1).gcd a1 = 1 ∧
      (n + a1 + a2).gcd a2 = 1 ∧
      (n + a1 + a2 + a3).gcd a3 = 1 ∧
      (n + a1 + a2 + a3 + a4).gcd a4 = 1 ∧
      (n + a1 + a2 + a3 + a4 + a5).gcd a5 = 1 ∧
      n + a1 + a2 + a3 + a4 + a5 = 200) 
    → n = 189 :=
begin
  sorry
end

end largest_initial_number_l618_618321


namespace exists_X_point_l618_618775

-- Define the structure of a convex polygon given its vertices
structure ConvexPolygon (n : ℕ) :=
(vertices : fin n → ℝ × ℝ)
(is_convex : ConvexHull (Set.range vertices))

-- Define the properties of the points A_i and B_i as mentioned in the problem
variables {n : ℕ} (P : ConvexPolygon n)

def valid_X_point (X : ℝ × ℝ) : Prop :=
∀ i : fin n, 
let Ai := P.vertices i,
    Bi := -- Some appropriate function defining Bi based on Ai and X
in dist X Ai / dist X Bi ≤ 2

-- The main theorem statement
theorem exists_X_point (P : ConvexPolygon n) : ∃ X : ℝ × ℝ, valid_X_point P X :=
sorry

end exists_X_point_l618_618775


namespace max_value_of_expression_l618_618645

theorem max_value_of_expression (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) : 
  (∃ x : ℝ, x = 3 → 
    ∀ A : ℝ, A = (a^4 + b^4 + c^4) / ((a + b + c)^4 - 80 * (a * b * c)^(4/3)) → 
      A ≤ x) :=
by
  sorry

end max_value_of_expression_l618_618645


namespace problem_solution_A_problem_solution_C_l618_618552

noncomputable def expr_A : ℝ :=
  (Real.sqrt 2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180))

noncomputable def expr_C : ℝ :=
  Real.tan (22.5 * Real.pi / 180) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)

theorem problem_solution_A :
  expr_A = 1 / 2 :=
by
  sorry

theorem problem_solution_C :
  expr_C = 1 / 2 :=
by
  sorry

end problem_solution_A_problem_solution_C_l618_618552


namespace express_1997_using_elevent_fours_l618_618085

def number_expression_uses_eleven_fours : Prop :=
  (4 * 444 + 44 * 4 + 44 + 4 / 4 = 1997)
  
theorem express_1997_using_elevent_fours : number_expression_uses_eleven_fours :=
by
  sorry

end express_1997_using_elevent_fours_l618_618085


namespace who_made_mistake_l618_618017

-- Defining conditions for the colored pencils
def has_at_least_four_blue_pencils (b : Nat) : Prop := b >= 4
def has_at_least_five_green_pencils (g : Nat) : Prop := g >= 5
def has_at_least_three_blue_four_green_pencils (b g : Nat) : Prop := b >= 3 ∧ g >= 4
def has_at_least_four_blue_four_green_pencils (b g : Nat) : Prop := b >= 4 ∧ g >= 4

-- Statement of the problem
theorem who_made_mistake (b g : Nat) (vasya kolya petya misha : Prop) :
  has_at_least_four_blue_pencils b →
  has_at_least_five_green_pencils g →
  has_at_least_three_blue_four_green_pencils b g →
  has_at_least_four_blue_four_green_pencils b g →
  (∃ T : Set Prop, {vasya, kolya, petya, misha}.Erase T = {vasya, kolya, petya, misha} ∧
    T.Card = 3) →
  (kolya ↔ ¬ g >= 5) := 
sorry

end who_made_mistake_l618_618017


namespace eat_4_pounds_in_time_l618_618397

-- Define the eating rates
def rate_fat : ℝ := 1 / 20
def rate_thin : ℝ := 1 / 25

-- Combined eating rate
def combined_rate : ℝ := rate_fat + rate_thin

-- Time to eat 4 pounds
def time_to_eat (amount : ℝ) : ℝ := amount / combined_rate

-- The problem to be proved
theorem eat_4_pounds_in_time : time_to_eat 4 = 400 / 9 :=
by
  unfold time_to_eat combined_rate rate_fat rate_thin
  norm_num
  sorry

end eat_4_pounds_in_time_l618_618397


namespace first_house_bottles_l618_618121

theorem first_house_bottles (total_bottles : ℕ) 
  (cider_only : ℕ) (beer_only : ℕ) (half : ℕ → ℕ)
  (mixture : ℕ)
  (half_cider_bottles : ℕ)
  (half_beer_bottles : ℕ)
  (half_mixture_bottles : ℕ) : 
  total_bottles = 180 →
  cider_only = 40 →
  beer_only = 80 →
  mixture = total_bottles - (cider_only + beer_only) →
  half c = c / 2 →
  half_cider_bottles = half cider_only →
  half_beer_bottles = half beer_only →
  half_mixture_bottles = half mixture →
  half_cider_bottles + half_beer_bottles + half_mixture_bottles = 90 :=
by
  intros h_tot h_cid h_beer h_mix h_half half_cid half_beer half_mix
  sorry

end first_house_bottles_l618_618121


namespace incorrect_triangle_condition_l618_618917

theorem incorrect_triangle_condition (A B C : Type) [has_mul A] [has_add A] [has_pow A ℕ]
  (AB AC BC : A) (triangle_right : (∃ (angle : A), angle = (90 : A))) :
  ¬((AB ^ 2 + AC ^ 2 = BC ^ 2) ↔ triangle_right) := 
sorry

end incorrect_triangle_condition_l618_618917


namespace kolya_is_wrong_l618_618030

def pencils_problem_statement (at_least_four_blue : Prop) 
                              (at_least_five_green : Prop) 
                              (at_least_three_blue_and_four_green : Prop) 
                              (at_least_four_blue_and_four_green : Prop) : 
                              Prop :=
  ∃ (B G : ℕ), -- B represents the number of blue pencils, G represents the number of green pencils
    ((B ≥ 4) ∧ (G ≥ 4)) ∧ -- Vasya's statement (at least 4 blue), Petya's and Misha's combined statement (at least 4 green)
    at_least_four_blue ∧ -- Vasya's statement (there are at least 4 blue pencils)
    (at_least_five_green ↔ G ≥ 5) ∧ -- Kolya's statement (there are at least 5 green pencils)
    at_least_three_blue_and_four_green ∧ -- Petya's statement (at least 3 blue and 4 green)
    at_least_four_blue_and_four_green -- Misha's statement (at least 4 blue and 4 green)

theorem kolya_is_wrong (at_least_four_blue : Prop) 
                        (at_least_five_green : Prop) 
                        (at_least_three_blue_and_four_green : Prop) 
                        (at_least_four_blue_and_four_green : Prop) : 
                        pencils_problem_statement at_least_four_blue 
                                                  at_least_five_green 
                                                  at_least_three_blue_and_four_green 
                                                  at_least_four_blue_and_four_green :=
sorry

end kolya_is_wrong_l618_618030


namespace harmonic_sequences_have_zeros_l618_618387

noncomputable def is_harmonic (A B : ℕ → ℕ) (N : ℕ) : Prop :=
  ∀ i, A i = (1 / (2 * B i + 1)) * (∑ s in finset.range (2 * B i + 1), A (i + s - B i))

theorem harmonic_sequences_have_zeros (N : ℕ) (N_ge_two : N ≥ 2)
  (A B : ℕ → ℕ)
  (h_nonneg_A : ∀ i : ℕ, i ≥ 1 ∧ i ≤ N → A i ≥ 0) 
  (h_nonneg_B : ∀ i : ℕ, i ≥ 1 ∧ i ≤ N → B i ≥ 0)
  (h_periodic_A : ∀ i : ℕ, i ≥ 1 ∧ i ≤ N → ∃ k : ℕ, k ≥ 1 ∧ k ≤ N ∧ (i - k) % N = 0 ∧ A i = A k) 
  (h_periodic_B : ∀ i : ℕ, i ≥ 1 ∧ i ≤ N → ∃ k : ℕ, k ≥ 1 ∧ k ≤ N ∧ (i - k) % N = 0 ∧ B i = B k)
  (h_not_const_A : ∃ i j : ℕ, i ≠ j ∧ A i ≠ A j)
  (h_not_const_B : ∃ i j : ℕ, i ≠ j ∧ B i ≠ B j)
  (h_harmonic_A_B : is_harmonic A B N)
  (h_harmonic_B_A : is_harmonic B A N) :
  (∑ i in finset.range (N+1), if A i = 0 then 1 else 0) + (∑ i in finset.range (N+1), if B i = 0 then 1 else 0) ≥ N + 1 :=
  sorry

end harmonic_sequences_have_zeros_l618_618387


namespace midpoint_dist_to_y_axis_l618_618221

noncomputable def focus := (1 / 4 : ℝ, 0)

def parabola (x y : ℝ) := y^2 = x

def on_parabola (p : ℝ × ℝ) := parabola p.1 p.2 

def condition (A B : ℝ × ℝ) : Prop :=
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist A focus + dist B focus = 3

theorem midpoint_dist_to_y_axis {A B : ℝ × ℝ}
  (hA : on_parabola A) (hB : on_parabola B) (h : condition A B) :
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  in abs (midpoint.1) = 5 / 4 :=
by
  sorry

end midpoint_dist_to_y_axis_l618_618221


namespace limit_trigonometric_substitution_l618_618561

open Real

theorem limit_trigonometric_substitution :
  tendsto (λ x : ℝ, (1 - 2 * cos x) / (π - 3 * x)) (𝓝 (π / 3)) (𝓝 (- (sqrt 3) / 3)) :=
sorry

end limit_trigonometric_substitution_l618_618561


namespace largest_initial_number_l618_618318

theorem largest_initial_number :
  ∃ n : ℕ, 
    (∀ (a1 a2 a3 a4 a5 : ℕ),
      (n + a1).gcd a1 = 1 ∧
      (n + a1 + a2).gcd a2 = 1 ∧
      (n + a1 + a2 + a3).gcd a3 = 1 ∧
      (n + a1 + a2 + a3 + a4).gcd a4 = 1 ∧
      (n + a1 + a2 + a3 + a4 + a5).gcd a5 = 1 ∧
      n + a1 + a2 + a3 + a4 + a5 = 200) 
    → n = 189 :=
begin
  sorry
end

end largest_initial_number_l618_618318


namespace locus_of_midpoints_is_circle_l618_618887

-- Definitions based on problem conditions
noncomputable def circle_locus_midpoints
  (O P : Point) (r : ℝ) (O_center : O ≠ P) (circle : ∀ (M : Point), dist O M = r) : Prop :=
  ∀ (M1 : Point), (∃ (M : Point), dist O M = r ∧ M1 = midpoint P M) →
  dist (midpoint O P) M1 = r / 2

-- Theorem statement to prove the locus of midpoints is a circle with specified center and radius
theorem locus_of_midpoints_is_circle (O P : Point) (r : ℝ) (O_center : O ≠ P)
  (circle : ∀ (M : Point), dist O M = r) :
  circle_locus_midpoints O P r O_center circle :=
by sorry

end locus_of_midpoints_is_circle_l618_618887


namespace xyz_inequality_l618_618723

theorem xyz_inequality (x y z : ℝ) (h : x + y + z > 0) : x^3 + y^3 + z^3 > 3 * x * y * z :=
by
  sorry

end xyz_inequality_l618_618723


namespace probability_of_minY_gt_maxX_l618_618360

noncomputable def problem_statement
  (n k : ℕ) (a : ℕ) 
  (hn : 0 < n) 
  (hk : 0 < k) 
  (ha : 0 ≤ a) 
  (X : finset (fin (k + a + 1))) 
  (Y : finset (fin (k + a + n + 1)))
  [fintype (fin (k + a + 1))] 
  [fintype (fin (k + a + n + 1))] 
  (hX : X.card = k) 
  (hY : Y.card = n) : ℝ :=
  let P (A : finset (fin (k + a + n + 1))) := (A.min' sorry > X.max' sorry) in
  ∑ (Y : finset (fin (k + a + n + 1))) in (finset.powerset_len n (finset.univ : finset (fin (k + a + n + 1)))), 
    if P Y then 1 else 0

theorem probability_of_minY_gt_maxX :
  ∀ (n k : ℕ) (a : ℕ)
  (hn : 0 < n) 
  (hk : 0 < k) 
  (ha : 0 ≤ a) 
  (X : finset (fin (k + a + 1))) 
  (Y : finset (fin (k + a + n + 1)))
  [fintype (fin (k + a + 1))] 
  [fintype (fin (k + a + n + 1))] 
  (hX : X.card = k) 
  (hY : Y.card = n), 
  problem_statement n k a hn hk ha X Y hX hY = (k.factorial * n.factorial) / ((k + n).factorial) := 
sorry

end probability_of_minY_gt_maxX_l618_618360


namespace largest_average_set_l618_618920

noncomputable def average (S : Set ℕ) : ℚ := (S.min + S.max) / 2

theorem largest_average_set :
  let S2 := { x | 1 ≤ x ∧ x ≤ 201 ∧ x % 2 = 0 },
      S3 := { x | 1 ≤ x ∧ x ≤ 201 ∧ x % 3 = 0 },
      S4 := { x | 1 ≤ x ∧ x ≤ 201 ∧ x % 4 = 0 },
      S5 := { x | 1 ≤ x ∧ x ≤ 201 ∧ x % 5 = 0 },
      S6 := { x | 1 ≤ x ∧ x ≤ 201 ∧ x % 6 = 0 }
  in average S5 > average S2 ∧ average S5 > average S3 ∧ average S5 > average S4 ∧ average S5 > average S6 :=
by
  sorry

end largest_average_set_l618_618920


namespace complex_power_equality_l618_618677

namespace ComplexProof

open Complex

noncomputable def cos5 : ℂ := cos (5 * Real.pi / 180)

theorem complex_power_equality (w : ℂ) (h : w + 1 / w = 2 * cos5) : 
  w ^ 1000 + 1 / (w ^ 1000) = -((Real.sqrt 5 + 1) / 2) :=
sorry

end ComplexProof

end complex_power_equality_l618_618677


namespace geometric_seq_increasing_l618_618683

theorem geometric_seq_increasing (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) → 
  (a 1 > a 0) = (∃ a1, (a1 > 0 ∧ q > 1) ∨ (a1 < 0 ∧ 0 < q ∧ q < 1)) :=
sorry

end geometric_seq_increasing_l618_618683


namespace problem_solution_A_problem_solution_C_l618_618554

noncomputable def expr_A : ℝ :=
  (Real.sqrt 2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180))

noncomputable def expr_C : ℝ :=
  Real.tan (22.5 * Real.pi / 180) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)

theorem problem_solution_A :
  expr_A = 1 / 2 :=
by
  sorry

theorem problem_solution_C :
  expr_C = 1 / 2 :=
by
  sorry

end problem_solution_A_problem_solution_C_l618_618554


namespace parabola_focus_hyperbola_equation_l618_618939

-- Problem 1
theorem parabola_focus (p : ℝ) (h₀ : p > 0) (h₁ : 2 * p - 0 - 4 = 0) : p = 2 :=
by
  sorry

-- Problem 2
theorem hyperbola_equation (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
  (h₂ : b / a = 3 / 4) (h₃ : a^2 / a = 16 / 5) (h₄ : a^2 + b^2 = 1) :
  (x^2 / 16) - (y^2 / 9) = 1 :=
by
  sorry

end parabola_focus_hyperbola_equation_l618_618939


namespace distribute_awards_l618_618455

theorem distribute_awards :
  (∑ k in finset.range 11, if k ≤ 3 then 1 else 0) = 84 :=
by
  sorry

end distribute_awards_l618_618455


namespace infinite_pairs_exist_l618_618826

theorem infinite_pairs_exist : 
  ∃^∞ (a b : ℤ), ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ * x₂ = 1 ∧ x₁^2012 = a * x₁ + b ∧ x₂^2012 = a * x₂ + b :=
sorry

end infinite_pairs_exist_l618_618826


namespace average_age_of_combined_rooms_l618_618832

theorem average_age_of_combined_rooms
  (num_people_A : ℕ) (avg_age_A : ℕ)
  (num_people_B : ℕ) (avg_age_B : ℕ)
  (num_people_C : ℕ) (avg_age_C : ℕ)
  (hA : num_people_A = 8) (hAA : avg_age_A = 35)
  (hB : num_people_B = 5) (hBB : avg_age_B = 30)
  (hC : num_people_C = 7) (hCC : avg_age_C = 50) :
  ((num_people_A * avg_age_A + num_people_B * avg_age_B + num_people_C * avg_age_C) / 
  (num_people_A + num_people_B + num_people_C) = 39) :=
by
  sorry

end average_age_of_combined_rooms_l618_618832


namespace integer_binom_sum_sum_integer_values_l618_618608

theorem integer_binom_sum (n : ℕ) (h : nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13) : n = 13 :=
sorry

theorem sum_integer_values (h : nat.choose 25 13 + nat.choose 25 12 = nat.choose 26 13) : 
  ∑ (x : ℕ) in {n | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13}.to_finset, x = 13 :=
sorry

end integer_binom_sum_sum_integer_values_l618_618608


namespace integers_multiples_of_d_l618_618602

theorem integers_multiples_of_d (d m n : ℕ) 
  (h1 : 2 ≤ m) 
  (h2 : 1 ≤ n) 
  (gcd_m_n : Nat.gcd m n = d) 
  (gcd_m_4n1 : Nat.gcd m (4 * n + 1) = 1) : 
  m % d = 0 :=
sorry

end integers_multiples_of_d_l618_618602


namespace smallest_b_perfect_fourth_power_l618_618940

theorem smallest_b_perfect_fourth_power:
  ∃ b : ℕ, (∀ n : ℕ, 5 * n = (7 * b^2 + 7 * b + 7) → ∃ x : ℕ, n = x^4) 
  ∧ b = 41 :=
sorry

end smallest_b_perfect_fourth_power_l618_618940
