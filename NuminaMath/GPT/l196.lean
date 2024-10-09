import Mathlib

namespace floor_sqrt_80_eq_8_l196_19646

theorem floor_sqrt_80_eq_8
  (h1 : 8^2 = 64)
  (h2 : 9^2 = 81)
  (h3 : 64 < 80 ∧ 80 < 81)
  (h4 : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) : 
  Int.floor (Real.sqrt 80) = 8 := by
  sorry

end floor_sqrt_80_eq_8_l196_19646


namespace runner_time_second_half_l196_19689

theorem runner_time_second_half (v : ℝ) (h1 : 20 / v + 4 = 40 / v) : 40 / v = 8 :=
by
  sorry

end runner_time_second_half_l196_19689


namespace rahul_matches_l196_19654

theorem rahul_matches
  (initial_avg : ℕ)
  (runs_today : ℕ)
  (final_avg : ℕ)
  (n : ℕ)
  (H1 : initial_avg = 50)
  (H2 : runs_today = 78)
  (H3 : final_avg = 54)
  (H4 : (initial_avg * n + runs_today) = final_avg * (n + 1)) :
  n = 6 :=
by
  sorry

end rahul_matches_l196_19654


namespace quadratic_eq_solutions_l196_19625

theorem quadratic_eq_solutions : ∃ x1 x2 : ℝ, (x^2 = x) ∨ (x = 0 ∧ x = 1) := by
  sorry

end quadratic_eq_solutions_l196_19625


namespace solve_equation_l196_19694

theorem solve_equation :
  {x : ℝ | (x + 1) * (x + 3) = x + 1} = {-1, -2} :=
sorry

end solve_equation_l196_19694


namespace find_inner_circle_radius_of_trapezoid_l196_19690

noncomputable def radius_of_inner_circle (k m n p : ℤ) : ℝ :=
  (-k + m * Real.sqrt n) / p

def is_equivalent (a b : ℝ) : Prop := a = b

theorem find_inner_circle_radius_of_trapezoid :
  ∃ (r : ℝ), is_equivalent r (radius_of_inner_circle 123 104 3 29) :=
by
  let r := radius_of_inner_circle 123 104 3 29
  have h1 :  (4^2 + (Real.sqrt (r^2 + 8 * r))^2 = (r + 4)^2) := sorry
  have h2 :  (3^2 + (Real.sqrt (r^2 + 6 * r))^2 = (r + 3)^2) := sorry
  have height_eq : Real.sqrt 13 = (Real.sqrt (r^2 + 6 * r) + Real.sqrt (r^2 + 8 * r)) := sorry
  use r
  exact sorry

end find_inner_circle_radius_of_trapezoid_l196_19690


namespace brick_height_correct_l196_19670

-- Definitions
def wall_length : ℝ := 8
def wall_height : ℝ := 6
def wall_thickness : ℝ := 0.02 -- converted from 2 cm to meters
def brick_length : ℝ := 0.05 -- converted from 5 cm to meters
def brick_width : ℝ := 0.11 -- converted from 11 cm to meters
def brick_height : ℝ := 0.06 -- converted from 6 cm to meters
def number_of_bricks : ℝ := 2909.090909090909

-- Statement to prove
theorem brick_height_correct : brick_height = 0.06 := by
  sorry

end brick_height_correct_l196_19670


namespace goldfish_equal_in_seven_months_l196_19617

/-- Define the growth of Alice's goldfish: they triple every month. -/
def alice_goldfish (n : ℕ) : ℕ := 3 * 3 ^ n

/-- Define the growth of Bob's goldfish: they quadruple every month. -/
def bob_goldfish (n : ℕ) : ℕ := 256 * 4 ^ n

/-- The main theorem we want to prove: For Alice and Bob's goldfish count to be equal,
    it takes 7 months. -/
theorem goldfish_equal_in_seven_months : ∃ n : ℕ, alice_goldfish n = bob_goldfish n ∧ n = 7 := 
by
  sorry

end goldfish_equal_in_seven_months_l196_19617


namespace three_digit_square_ends_with_self_l196_19636

theorem three_digit_square_ends_with_self (A : ℕ) (hA1 : 100 ≤ A) (hA2 : A ≤ 999) (hA3 : A^2 % 1000 = A) : 
  A = 376 ∨ A = 625 :=
sorry

end three_digit_square_ends_with_self_l196_19636


namespace luke_earning_problem_l196_19693

variable (WeedEarning Weeks SpendPerWeek MowingEarning : ℤ)

theorem luke_earning_problem
  (h1 : WeedEarning = 18)
  (h2 : Weeks = 9)
  (h3 : SpendPerWeek = 3)
  (h4 : MowingEarning + WeedEarning = Weeks * SpendPerWeek) :
  MowingEarning = 9 := by
  sorry

end luke_earning_problem_l196_19693


namespace cannot_form_optionE_l196_19656

-- Define the 4x4 tile
structure Tile4x4 :=
(matrix : Fin 4 → Fin 4 → Bool) -- Boolean to represent black or white

-- Define the condition of alternating rows and columns
def alternating_pattern (tile : Tile4x4) : Prop :=
  (∀ i, tile.matrix i 0 ≠ tile.matrix i 1 ∧
         tile.matrix i 2 ≠ tile.matrix i 3) ∧
  (∀ j, tile.matrix 0 j ≠ tile.matrix 1 j ∧
         tile.matrix 2 j ≠ tile.matrix 3 j)

-- Example tiles for options A, B, C, D, E
def optionA : Tile4x4 := sorry
def optionB : Tile4x4 := sorry
def optionC : Tile4x4 := sorry
def optionD : Tile4x4 := sorry
def optionE : Tile4x4 := sorry

-- Given pieces that can form a 4x4 alternating tile
axiom given_piece1 : Tile4x4
axiom given_piece2 : Tile4x4

-- Combining given pieces to form a 4x4 tile
def combine_pieces (p1 p2 : Tile4x4) : Tile4x4 := sorry -- Combination logic here

-- Proposition stating the problem
theorem cannot_form_optionE :
  (∀ tile, tile = optionA ∨ tile = optionB ∨ tile = optionC ∨ tile = optionD ∨ tile = optionE →
    (tile = optionA ∨ tile = optionB ∨ tile = optionC ∨ tile = optionD → alternating_pattern tile) ∧
    tile = optionE → ¬alternating_pattern tile) :=
sorry

end cannot_form_optionE_l196_19656


namespace sin_of_angle_F_l196_19663

theorem sin_of_angle_F 
  (DE EF DF : ℝ) 
  (h : DE = 12) 
  (h0 : EF = 20) 
  (h1 : DF = Real.sqrt (DE^2 + EF^2)) : 
  Real.sin (Real.arctan (DF / EF)) = 12 / Real.sqrt (DE^2 + EF^2) := 
by 
  sorry

end sin_of_angle_F_l196_19663


namespace distance_of_coming_down_stairs_l196_19602

noncomputable def totalTimeAscendingDescending (D : ℝ) : ℝ :=
  (D / 2) + ((D + 2) / 3)

theorem distance_of_coming_down_stairs : ∃ D : ℝ, totalTimeAscendingDescending D = 4 ∧ (D + 2) = 6 :=
by
  sorry

end distance_of_coming_down_stairs_l196_19602


namespace relationship_among_a_b_c_l196_19650

noncomputable def a : ℝ := (0.8 : ℝ)^(5.2 : ℝ)
noncomputable def b : ℝ := (0.8 : ℝ)^(5.5 : ℝ)
noncomputable def c : ℝ := (5.2 : ℝ)^(0.1 : ℝ)

theorem relationship_among_a_b_c : b < a ∧ a < c := by
  sorry

end relationship_among_a_b_c_l196_19650


namespace expression_evaluation_l196_19692

theorem expression_evaluation :
  5 * 402 + 4 * 402 + 3 * 402 + 401 = 5225 := by
  sorry

end expression_evaluation_l196_19692


namespace geometric_sequence_sum_of_first_four_terms_l196_19641

theorem geometric_sequence_sum_of_first_four_terms (a r : ℝ) 
  (h1 : a + a * r = 7) 
  (h2 : a * (1 + r + r^2 + r^3 + r^4 + r^5) = 91) : 
  a * (1 + r + r^2 + r^3) = 32 :=
by
  sorry

end geometric_sequence_sum_of_first_four_terms_l196_19641


namespace sinA_value_triangle_area_l196_19678

-- Definitions of the given variables
variables (A B C : ℝ)
variables (a b c : ℝ)
variables (sinA sinC cosC : ℝ)

-- Given conditions
axiom h_c : c = Real.sqrt 2
axiom h_a : a = 1
axiom h_cosC : cosC = 3 / 4
axiom h_sinC : sinC = Real.sqrt 7 / 4
axiom h_b : b = 2

-- Question 1: Prove sin A = sqrt 14 / 8
theorem sinA_value : sinA = Real.sqrt 14 / 8 :=
sorry

-- Question 2: Prove the area of triangle ABC is sqrt 7 / 4
theorem triangle_area : 1/2 * a * b * sinC = Real.sqrt 7 / 4 :=
sorry

end sinA_value_triangle_area_l196_19678


namespace find_fractions_l196_19649

-- Define the numerators and denominators
def p1 := 75
def p2 := 70
def q1 := 34
def q2 := 51

-- Define the fractions
def frac1 := p1 / q1
def frac2 := p1 / q2

-- Define the greatest common divisor (gcd) condition
def gcd_condition := Nat.gcd p1 p2 = p1 - p2

-- Define the least common multiple (lcm) condition
def lcm_condition := Nat.lcm p1 p2 = 1050

-- Define the difference condition
def difference_condition := (frac1 - frac2) = (5 / 6)

-- Lean proof statement
theorem find_fractions :
  gcd_condition ∧ lcm_condition ∧ difference_condition :=
by
  sorry

end find_fractions_l196_19649


namespace merchant_gross_profit_l196_19697

theorem merchant_gross_profit :
  ∃ S : ℝ, (42 + 0.30 * S = S) ∧ ((0.80 * S) - 42 = 6) :=
by
  sorry

end merchant_gross_profit_l196_19697


namespace not_or_false_implies_or_true_l196_19613

variable (p q : Prop)

theorem not_or_false_implies_or_true (h : ¬(p ∨ q) = False) : p ∨ q :=
by
  sorry

end not_or_false_implies_or_true_l196_19613


namespace number_of_m_l196_19684

theorem number_of_m (k : ℕ) : 
  (∀ m a b : ℤ, 
      (a ≠ 0 ∧ b ≠ 0) ∧ 
      (a + b = m) ∧ 
      (a * b = m + 2006) → k = 5) :=
sorry

end number_of_m_l196_19684


namespace points_on_line_l196_19630

-- Define the points
def P1 : (ℝ × ℝ) := (8, 16)
def P2 : (ℝ × ℝ) := (2, 4)

-- Define the line equation as a predicate
def on_line (m b : ℝ) (p : ℝ × ℝ) : Prop := p.2 = m * p.1 + b

-- Define the given points to be checked
def P3 : (ℝ × ℝ) := (5, 10)
def P4 : (ℝ × ℝ) := (7, 14)
def P5 : (ℝ × ℝ) := (4, 7)
def P6 : (ℝ × ℝ) := (10, 20)
def P7 : (ℝ × ℝ) := (3, 6)

theorem points_on_line :
  let m := 2
  let b := 0
  on_line m b P3 ∧
  on_line m b P4 ∧
  ¬ on_line m b P5 ∧
  on_line m b P6 ∧
  on_line m b P7 :=
by
  sorry

end points_on_line_l196_19630


namespace lucille_paint_cans_needed_l196_19676

theorem lucille_paint_cans_needed :
  let wall1_area := 3 * 2
  let wall2_area := 3 * 2
  let wall3_area := 5 * 2
  let wall4_area := 4 * 2
  let total_area := wall1_area + wall2_area + wall3_area + wall4_area
  let coverage_per_can := 2
  let cans_needed := total_area / coverage_per_can
  cans_needed = 15 := 
by 
  sorry

end lucille_paint_cans_needed_l196_19676


namespace complementSetM_l196_19640

open Set Real

-- The universal set U is the set of all real numbers
def universalSet : Set ℝ := univ

-- The set M is defined as {x | |x - 1| ≤ 2}
def setM : Set ℝ := {x : ℝ | |x - 1| ≤ 2}

-- We need to prove that the complement of M with respect to U is {x | x < -1 ∨ x > 3}
theorem complementSetM :
  (universalSet \ setM) = {x : ℝ | x < -1 ∨ x > 3} :=
by
  sorry

end complementSetM_l196_19640


namespace limit_tanxy_over_y_l196_19632

theorem limit_tanxy_over_y (f : ℝ×ℝ → ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x y, abs (x - 3) < δ ∧ abs y < δ → abs (f (x, y) - 3) < ε) :=
sorry

end limit_tanxy_over_y_l196_19632


namespace problems_left_to_grade_l196_19687

theorem problems_left_to_grade 
  (problems_per_worksheet : ℕ)
  (total_worksheets : ℕ)
  (graded_worksheets : ℕ)
  (h1 : problems_per_worksheet = 2)
  (h2 : total_worksheets = 14)
  (h3 : graded_worksheets = 7) : 
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 14 :=
by 
  sorry

end problems_left_to_grade_l196_19687


namespace Noah_age_in_10_years_is_22_l196_19622

def Joe_age : Nat := 6
def Noah_age := 2 * Joe_age
def Noah_age_after_10_years := Noah_age + 10

theorem Noah_age_in_10_years_is_22 : Noah_age_after_10_years = 22 := by
  sorry

end Noah_age_in_10_years_is_22_l196_19622


namespace brother_and_sister_ages_l196_19643

theorem brother_and_sister_ages :
  ∃ (b s : ℕ), (b - 3 = 7 * (s - 3)) ∧ (b - 2 = 4 * (s - 2)) ∧ (b - 1 = 3 * (s - 1)) ∧ (b = 5 / 2 * s) ∧ b = 10 ∧ s = 4 :=
by 
  sorry

end brother_and_sister_ages_l196_19643


namespace intersection_M_N_l196_19681

def M : Set ℝ := {x | (x - 1) * (x - 4) = 0}
def N : Set ℝ := {x | (x + 1) * (x - 3) < 0}

theorem intersection_M_N :
  M ∩ N = {1} :=
sorry

end intersection_M_N_l196_19681


namespace ratio_of_segments_l196_19611

theorem ratio_of_segments (a b c r s : ℝ) (h : a / b = 1 / 4)
  (h₁ : c ^ 2 = a ^ 2 + b ^ 2)
  (h₂ : r = a ^ 2 / c)
  (h₃ : s = b ^ 2 / c) :
  r / s = 1 / 16 :=
by
  sorry

end ratio_of_segments_l196_19611


namespace even_function_f_l196_19673

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^3 - x^2 else -(-x)^3 - (-x)^2

theorem even_function_f (x : ℝ) (h : ∀ x ≤ 0, f x = x^3 - x^2) :
  (∀ x, f x = f (-x)) ∧ (∀ x > 0, f x = -x^3 - x^2) :=
by
  sorry

end even_function_f_l196_19673


namespace confectioner_pastry_l196_19699

theorem confectioner_pastry (P : ℕ) (h : P / 28 - 6 = P / 49) : P = 378 :=
sorry

end confectioner_pastry_l196_19699


namespace area_quadrilateral_ABCDE_correct_l196_19660

noncomputable def area_quadrilateral_ABCDE (AM NM AN BN BO OC CP CD EP DE : ℝ) : ℝ :=
  (0.5 * AM * NM * Real.sqrt 2) + (0.5 * BN * BO) + (0.5 * OC * CP * Real.sqrt 2) - (0.5 * DE * EP)

theorem area_quadrilateral_ABCDE_correct :
  ∀ (AM NM AN BN BO OC CP CD EP DE : ℝ),
    DE = 12 ∧ 
    AM = 36 ∧ 
    NM = 36 ∧ 
    AN = 36 * Real.sqrt 2 ∧
    BN = 36 * Real.sqrt 2 - 36 ∧
    BO = 36 ∧
    OC = 36 ∧
    CP = 36 * Real.sqrt 2 ∧
    CD = 24 ∧
    EP = 24
    → area_quadrilateral_ABCDE AM NM AN BN BO OC CP CD EP DE = 2311.2 * Real.sqrt 2 + 504 :=
by intro AM NM AN BN BO OC CP CD EP DE h;
   cases h;
   sorry

end area_quadrilateral_ABCDE_correct_l196_19660


namespace inequality_in_triangle_l196_19619

variables {a b c : ℝ}

namespace InequalityInTriangle

-- Define the condition that a, b, c are sides of a triangle
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem inequality_in_triangle (a b c : ℝ) (h : is_triangle a b c) :
  1 / (b + c - a) + 1 / (c + a - b) + 1 / (a + b - c) > 9 / (a + b + c) :=
sorry

end InequalityInTriangle

end inequality_in_triangle_l196_19619


namespace total_cost_of_tickets_l196_19637

-- Conditions
def normal_price : ℝ := 50
def website_tickets_cost : ℝ := 2 * normal_price
def scalper_tickets_cost : ℝ := 2 * (2.4 * normal_price) - 10
def discounted_ticket_cost : ℝ := 0.6 * normal_price

-- Proof Statement
theorem total_cost_of_tickets :
  website_tickets_cost + scalper_tickets_cost + discounted_ticket_cost = 360 :=
by
  sorry

end total_cost_of_tickets_l196_19637


namespace domain_ln_x_plus_one_l196_19642

theorem domain_ln_x_plus_one :
  ∀ (x : ℝ), ∃ (y : ℝ), y = Real.log (x + 1) ↔ x > -1 :=
by sorry

end domain_ln_x_plus_one_l196_19642


namespace find_original_number_l196_19688

/-- The difference between a number increased by 18.7% and the same number decreased by 32.5% is 45. -/
theorem find_original_number (w : ℝ) (h : 1.187 * w - 0.675 * w = 45) : w = 45 / 0.512 :=
by
  sorry

end find_original_number_l196_19688


namespace race_car_cost_l196_19647

variable (R : ℝ)
variable (Mater_cost SallyMcQueen_cost : ℝ)

-- Conditions
def Mater_cost_def : Mater_cost = 0.10 * R := by sorry
def SallyMcQueen_cost_def : SallyMcQueen_cost = 3 * Mater_cost := by sorry
def SallyMcQueen_cost_val : SallyMcQueen_cost = 42000 := by sorry

-- Theorem to prove the race car cost
theorem race_car_cost : R = 140000 :=
  by
    -- Use the conditions to prove
    sorry

end race_car_cost_l196_19647


namespace miles_per_dollar_l196_19610

def car_mpg : ℝ := 32
def gas_cost_per_gallon : ℝ := 4

theorem miles_per_dollar (X : ℝ) : 
  (X / gas_cost_per_gallon) * car_mpg = 8 * X :=
by
  sorry

end miles_per_dollar_l196_19610


namespace f_four_l196_19616

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq (a b : ℝ) : f (a + b) + f (a - b) = 2 * f a + 2 * f b
axiom f_two : f 2 = 9 
axiom not_identically_zero : ¬ ∀ x : ℝ, f x = 0

theorem f_four : f 4 = 36 :=
by sorry

end f_four_l196_19616


namespace divide_composite_products_l196_19667

def first_eight_composites : List ℕ := [4, 6, 8, 9, 10, 12, 14, 15]
def next_eight_composites : List ℕ := [16, 18, 20, 21, 22, 24, 25, 26]

def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

theorem divide_composite_products :
  product first_eight_composites * 3120 = product next_eight_composites :=
by
  -- This would be the place for the proof solution
  sorry

end divide_composite_products_l196_19667


namespace condition_iff_absolute_value_l196_19634

theorem condition_iff_absolute_value (a b : ℝ) : (a > b) ↔ (a * |a| > b * |b|) :=
sorry

end condition_iff_absolute_value_l196_19634


namespace f_not_surjective_l196_19674

def f : ℝ → ℕ → Prop := sorry

theorem f_not_surjective (f : ℝ → ℕ) 
  (h : ∀ x y : ℝ, f (x + (1 / f y)) = f (y + (1 / f x))) : 
  ¬ (∀ n : ℕ, ∃ x : ℝ, f x = n) :=
sorry

end f_not_surjective_l196_19674


namespace find_h_l196_19633

theorem find_h {a b c n k : ℝ} (x : ℝ) (h_val : ℝ) 
  (h_quad : a * x^2 + b * x + c = 3 * (x - 5)^2 + 15) :
  (4 * a) * x^2 + (4 * b) * x + (4 * c) = n * (x - h_val)^2 + k → h_val = 5 :=
sorry

end find_h_l196_19633


namespace water_to_concentrate_ratio_l196_19600

theorem water_to_concentrate_ratio (servings : ℕ) (serving_size_oz concentrate_size_oz : ℕ)
                                (cans_of_concentrate required_juice_oz : ℕ)
                                (h_servings : servings = 280)
                                (h_serving_size : serving_size_oz = 6)
                                (h_concentrate_size : concentrate_size_oz = 12)
                                (h_cans_of_concentrate : cans_of_concentrate = 35)
                                (h_required_juice : required_juice_oz = servings * serving_size_oz)
                                (h_made_juice : required_juice_oz = 1680)
                                (h_concentrate_volume : cans_of_concentrate * concentrate_size_oz = 420)
                                (h_water_volume : required_juice_oz - (cans_of_concentrate * concentrate_size_oz) = 1260)
                                (h_water_cans : 1260 / concentrate_size_oz = 105) :
                                105 / 35 = 3 :=
by
  sorry

end water_to_concentrate_ratio_l196_19600


namespace athlete_difference_is_30_l196_19615

def initial_athletes : ℕ := 600
def leaving_rate : ℕ := 35
def leaving_duration : ℕ := 6
def arrival_rate : ℕ := 20
def arrival_duration : ℕ := 9

def athletes_left : ℕ := leaving_rate * leaving_duration
def new_athletes : ℕ := arrival_rate * arrival_duration
def remaining_athletes : ℕ := initial_athletes - athletes_left
def final_athletes : ℕ := remaining_athletes + new_athletes
def athlete_difference : ℕ := initial_athletes - final_athletes

theorem athlete_difference_is_30 : athlete_difference = 30 :=
by
  show athlete_difference = 30
  -- Proof goes here
  sorry

end athlete_difference_is_30_l196_19615


namespace correct_option_is_B_l196_19695

noncomputable def correct_calculation (x : ℝ) : Prop :=
  (x ≠ 1) → (x ≠ 0) → (x ≠ -1) → (-2 / (2 * x - 2) = 1 / (1 - x))

theorem correct_option_is_B (x : ℝ) : correct_calculation x := by
  intros hx1 hx2 hx3
  sorry

end correct_option_is_B_l196_19695


namespace qinJiushao_value_l196_19624

/-- A specific function f(x) with given a and b -/
def f (x : ℤ) : ℤ :=
  x^5 + 47 * x^4 - 37 * x^2 + 1

/-- Qin Jiushao algorithm to find V3 at x = -1 -/
def qinJiushao (x : ℤ) : ℤ :=
  let V0 := 1
  let V1 := V0 * x + 47
  let V2 := V1 * x + 0
  let V3 := V2 * x - 37
  V3

theorem qinJiushao_value :
  qinJiushao (-1) = 9 :=
by
  sorry

end qinJiushao_value_l196_19624


namespace domain_of_tan_function_l196_19679

theorem domain_of_tan_function :
  (∀ x : ℝ, ∀ k : ℤ, 2 * x - π / 4 ≠ k * π + π / 2 ↔ x ≠ (k * π) / 2 + 3 * π / 8) :=
sorry

end domain_of_tan_function_l196_19679


namespace taqeesha_grade_correct_l196_19691

-- Definitions for conditions
def total_score_of_24_students := 24 * 82
def total_score_of_25_students (T: ℕ) := 25 * 84
def taqeesha_grade := 132

-- Theorem statement forming the proof problem
theorem taqeesha_grade_correct
    (h1: total_score_of_24_students + taqeesha_grade = total_score_of_25_students taqeesha_grade): 
    taqeesha_grade = 132 :=
by
  sorry

end taqeesha_grade_correct_l196_19691


namespace predicted_height_at_age_10_l196_19608

-- Define the regression model as a function
def regression_model (x : ℝ) : ℝ := 7.19 * x + 73.93

-- Assert the predicted height at age 10
theorem predicted_height_at_age_10 : abs (regression_model 10 - 145.83) < 0.01 := 
by
  -- Here, we would prove the calculation steps
  sorry

end predicted_height_at_age_10_l196_19608


namespace annie_crayons_l196_19655

def initial_crayons : ℕ := 4
def additional_crayons : ℕ := 36
def total_crayons : ℕ := initial_crayons + additional_crayons

theorem annie_crayons : total_crayons = 40 :=
by
  sorry

end annie_crayons_l196_19655


namespace probability_each_university_at_least_one_admission_l196_19604

def total_students := 4
def total_universities := 3

theorem probability_each_university_at_least_one_admission :
  ∃ (p : ℚ), p = 4 / 9 :=
by
  sorry

end probability_each_university_at_least_one_admission_l196_19604


namespace sum_of_missing_angles_l196_19609

theorem sum_of_missing_angles (angle_sum_known : ℕ) (divisor : ℕ) (total_sides : ℕ) (missing_angles_sum : ℕ)
  (h1 : angle_sum_known = 1620)
  (h2 : divisor = 180)
  (h3 : total_sides = 12)
  (h4 : angle_sum_known + missing_angles_sum = divisor * (total_sides - 2)) :
  missing_angles_sum = 180 :=
by
  -- Skipping the proof for this theorem
  sorry

end sum_of_missing_angles_l196_19609


namespace puzzles_sold_eq_36_l196_19627

def n_science_kits : ℕ := 45
def n_puzzles : ℕ := n_science_kits - 9

theorem puzzles_sold_eq_36 : n_puzzles = 36 := by
  sorry

end puzzles_sold_eq_36_l196_19627


namespace arithmetic_mean_solution_l196_19685

theorem arithmetic_mean_solution (x : ℚ) :
  (x + 10 + 20 + 3*x + 18 + 3*x + 6) / 5 = 30 → x = 96 / 7 :=
by
  intros h
  sorry

end arithmetic_mean_solution_l196_19685


namespace total_bottles_ordered_in_april_and_may_is_1000_l196_19605

-- Define the conditions
def casesInApril : Nat := 20
def casesInMay : Nat := 30
def bottlesPerCase : Nat := 20

-- The total number of bottles ordered in April and May
def totalBottlesOrdered : Nat := (casesInApril + casesInMay) * bottlesPerCase

-- The main statement to be proved
theorem total_bottles_ordered_in_april_and_may_is_1000 :
  totalBottlesOrdered = 1000 :=
sorry

end total_bottles_ordered_in_april_and_may_is_1000_l196_19605


namespace number_of_smaller_cubes_l196_19666

theorem number_of_smaller_cubes 
  (volume_large_cube : ℝ)
  (volume_small_cube : ℝ)
  (surface_area_difference : ℝ)
  (h1 : volume_large_cube = 216)
  (h2 : volume_small_cube = 1)
  (h3 : surface_area_difference = 1080) :
  ∃ n : ℕ, n * 6 - 6 * (volume_large_cube^(1/3))^2 = surface_area_difference ∧ n = 216 :=
by
  sorry

end number_of_smaller_cubes_l196_19666


namespace average_of_175_results_l196_19623

theorem average_of_175_results (x y : ℕ) (hx : x = 100) (hy : y = 75) 
(a b : ℚ) (ha : a = 45) (hb : b = 65) :
  ((x * a + y * b) / (x + y) = 53.57) :=
sorry

end average_of_175_results_l196_19623


namespace value_of_m_l196_19657

theorem value_of_m (m : ℝ) : (3 = 2 * m + 1) → m = 1 :=
by
  intro h
  -- skipped proof due to requirement
  sorry

end value_of_m_l196_19657


namespace remainder_when_sum_divided_by_5_l196_19653

/-- Reinterpreting the same conditions and question: -/
theorem remainder_when_sum_divided_by_5 (a b c : ℕ) 
    (ha : a < 5) (hb : b < 5) (hc : c < 5) 
    (h1 : a * b * c % 5 = 1) 
    (h2 : 3 * c % 5 = 2)
    (h3 : 4 * b % 5 = (3 + b) % 5): 
    (a + b + c) % 5 = 4 := 
sorry

end remainder_when_sum_divided_by_5_l196_19653


namespace max_M_min_N_l196_19618

noncomputable def M (x y : ℝ) : ℝ := x / (2 * x + y) + y / (x + 2 * y)
noncomputable def N (x y : ℝ) : ℝ := x / (x + 2 * y) + y / (2 * x + y)

theorem max_M_min_N (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∃ t : ℝ, (∀ x y, 0 < x → 0 < y → M x y ≤ t) ∧ (∀ x y, 0 < x → 0 < y → N x y ≥ t) ∧ t = 2 / 3) :=
sorry

end max_M_min_N_l196_19618


namespace daria_still_owes_l196_19662

-- Definitions of the given conditions
def saved_amount : ℝ := 500
def couch_cost : ℝ := 750
def table_cost : ℝ := 100
def lamp_cost : ℝ := 50

-- Calculation of total cost of the furniture
def total_cost : ℝ := couch_cost + table_cost + lamp_cost

-- Calculation of the remaining amount owed
def remaining_owed : ℝ := total_cost - saved_amount

-- Proof statement that Daria still owes $400 before interest
theorem daria_still_owes : remaining_owed = 400 := by
  -- Skipping the proof
  sorry

end daria_still_owes_l196_19662


namespace factorization_identity_l196_19606

theorem factorization_identity (m : ℝ) : 
  -4 * m^3 + 4 * m^2 - m = -m * (2 * m - 1)^2 :=
sorry

end factorization_identity_l196_19606


namespace initial_number_of_persons_l196_19661

theorem initial_number_of_persons (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) (weight_diff : ℝ)
  (h1 : avg_increase = 2.5) 
  (h2 : old_weight = 75) 
  (h3 : new_weight = 95)
  (h4 : weight_diff = new_weight - old_weight)
  (h5 : weight_diff = avg_increase * n) : n = 8 := 
sorry

end initial_number_of_persons_l196_19661


namespace solve_quadratic_equation_l196_19621

theorem solve_quadratic_equation :
  ∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 2 ∧ x₂ = 1 - Real.sqrt 2 ∧ ∀ x : ℝ, (x^2 - 2*x - 1 = 0) ↔ (x = x₁ ∨ x = x₂) :=
by
  sorry

end solve_quadratic_equation_l196_19621


namespace fraction_of_number_l196_19644

theorem fraction_of_number : (7 / 8) * 64 = 56 := by
  sorry

end fraction_of_number_l196_19644


namespace integer_div_product_l196_19607

theorem integer_div_product (n : ℤ) : ∃ (k : ℤ), n * (n + 1) * (n + 2) = 6 * k := by
  sorry

end integer_div_product_l196_19607


namespace total_students_l196_19651

theorem total_students (n1 n2 : ℕ) (h1 : (158 - 140)/(n1 + 1) = 2) (h2 : (158 - 140)/(n2 + 1) = 3) :
  n1 + n2 + 2 = 15 :=
sorry

end total_students_l196_19651


namespace find_expression_l196_19696

variables (x y z : ℝ) (ω : ℂ)

theorem find_expression
  (h1 : x ≠ -1) (h2 : y ≠ -1) (h3 : z ≠ -1)
  (h4 : ω^3 = 1) (h5 : ω ≠ 1)
  (h6 : (1 / (x + ω) + 1 / (y + ω) + 1 / (z + ω) = ω)) :
  1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) = -1 / 3 :=
sorry

end find_expression_l196_19696


namespace part1_solution_part2_solution_l196_19671

-- Definitions for costs
variables (x y : ℝ)
variables (cost_A cost_B : ℝ)

-- Conditions
def condition1 : 80 * x + 35 * y = 2250 :=
  sorry

def condition2 : x = y - 15 :=
  sorry

-- Part 1: Cost of one bottle of each disinfectant
theorem part1_solution : x = cost_A ∧ y = cost_B :=
  sorry

-- Additional conditions for part 2
variables (m : ℕ)
variables (total_bottles : ℕ := 50)
variables (budget : ℝ := 1200)

-- Conditions for part 2
def condition3 : m + (total_bottles - m) = total_bottles :=
  sorry

def condition4 : 15 * m + 30 * (total_bottles - m) ≤ budget :=
  sorry

-- Part 2: Minimum number of bottles of Class A disinfectant
theorem part2_solution : m ≥ 20 :=
  sorry

end part1_solution_part2_solution_l196_19671


namespace prices_correct_minimum_cost_correct_l196_19672

-- Define the prices of the mustard brands
variables (x y m : ℝ)

def brandACost : ℝ := 9 * x + 6 * y
def brandBCost : ℝ := 5 * x + 8 * y

-- Conditions for prices
axiom cost_condition1 : brandACost x y = 390
axiom cost_condition2 : brandBCost x y = 310

-- Solution for prices
def priceA : ℝ := 30
def priceB : ℝ := 20

theorem prices_correct : x = priceA ∧ y = priceB :=
sorry

-- Conditions for minimizing cost
def totalCost (m : ℝ) : ℝ := 30 * m + 20 * (30 - m)
def totalPacks : ℝ := 30

-- Constraints
def constraint1 (m : ℝ) : Prop := m ≥ 5 + (30 - m)
def constraint2 (m : ℝ) : Prop := m ≤ 2 * (30 - m)

-- Minimum cost condition
def min_cost : ℝ := 780
def optimal_m : ℝ := 18

theorem minimum_cost_correct : constraint1 optimal_m ∧ constraint2 optimal_m ∧ totalCost optimal_m = min_cost :=
sorry

end prices_correct_minimum_cost_correct_l196_19672


namespace A_inter_B_eq_A_union_C_U_B_eq_l196_19669

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

end A_inter_B_eq_A_union_C_U_B_eq_l196_19669


namespace triangle_ABC_area_l196_19652

-- definition of points A, B, and C
def A : (ℝ × ℝ) := (0, 2)
def B : (ℝ × ℝ) := (6, 0)
def C : (ℝ × ℝ) := (3, 7)

-- helper function to calculate area of triangle given vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_ABC_area :
  triangle_area A B C = 18 := by
  sorry

end triangle_ABC_area_l196_19652


namespace boxes_remaining_to_sell_l196_19686

-- Define the conditions
def first_customer_boxes : ℕ := 5 
def second_customer_boxes : ℕ := 4 * first_customer_boxes
def third_customer_boxes : ℕ := second_customer_boxes / 2
def fourth_customer_boxes : ℕ := 3 * third_customer_boxes
def final_customer_boxes : ℕ := 10
def sales_goal : ℕ := 150

-- Total boxes sold
def total_boxes_sold : ℕ := first_customer_boxes + second_customer_boxes + third_customer_boxes + fourth_customer_boxes + final_customer_boxes

-- Boxes left to sell to hit the sales goal
def boxes_left_to_sell : ℕ := sales_goal - total_boxes_sold

-- Prove the number of boxes left to sell is 75
theorem boxes_remaining_to_sell : boxes_left_to_sell = 75 :=
by
  -- Step to prove goes here
  sorry

end boxes_remaining_to_sell_l196_19686


namespace root_in_interval_l196_19639

def polynomial (x : ℝ) := x^3 + 3 * x^2 - x + 1

noncomputable def A : ℤ := -4
noncomputable def B : ℤ := -3

theorem root_in_interval : (∃ x : ℝ, polynomial x = 0 ∧ (A : ℝ) < x ∧ x < (B : ℝ)) :=
sorry

end root_in_interval_l196_19639


namespace john_books_per_day_l196_19620

theorem john_books_per_day (books_total : ℕ) (total_weeks : ℕ) (days_per_week : ℕ) (total_days : ℕ)
  (read_days_eq : total_days = total_weeks * days_per_week)
  (books_per_day_eq : books_total = total_days * 4) : (books_total / total_days = 4) :=
by
  -- The conditions state the following:
  -- books_total = 48 (total books read)
  -- total_weeks = 6 (total number of weeks)
  -- days_per_week = 2 (number of days John reads per week)
  -- total_days = 12 (total number of days in which John reads books)
  -- read_days_eq :- total_days = total_weeks * days_per_week
  -- books_per_day_eq :- books_total = total_days * 4
  sorry

end john_books_per_day_l196_19620


namespace point_in_fourth_quadrant_l196_19638

def point : ℝ × ℝ := (4, -3)

def is_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : is_fourth_quadrant point :=
by
  sorry

end point_in_fourth_quadrant_l196_19638


namespace solve_fraction_eq_for_x_l196_19668

theorem solve_fraction_eq_for_x (x : ℝ) (hx : (x + 6) / (x - 3) = 4) : x = 6 :=
by sorry

end solve_fraction_eq_for_x_l196_19668


namespace cosine_ab_ac_l196_19629

noncomputable def vector_a := (-2, 4, -6)
noncomputable def vector_b := (0, 2, -4)
noncomputable def vector_c := (-6, 8, -10)

noncomputable def a_b : ℝ × ℝ × ℝ := (2, -2, 2)
noncomputable def a_c : ℝ × ℝ × ℝ := (-4, 4, -4)

noncomputable def ab_dot_ac : ℝ := -24

noncomputable def mag_a_b : ℝ := 2 * Real.sqrt 3
noncomputable def mag_a_c : ℝ := 4 * Real.sqrt 3

theorem cosine_ab_ac :
  (ab_dot_ac / (mag_a_b * mag_a_c) = -1) :=
sorry

end cosine_ab_ac_l196_19629


namespace monthly_income_of_labourer_l196_19659

variable (I : ℕ) -- Monthly income

-- Conditions: 
def condition1 := (85 * 6) - (6 * I) -- A boolean expression depicting the labourer fell into debt
def condition2 := (60 * 4) + (85 * 6 - 6 * I) + 30 -- Total income covers debt and saving 30

-- Statement to be proven
theorem monthly_income_of_labourer : 
  ∃ I : ℕ, condition1 I = 0 ∧ condition2 I = 4 * I → I = 78 :=
by
  sorry

end monthly_income_of_labourer_l196_19659


namespace pipe_B_fill_time_l196_19665

theorem pipe_B_fill_time
  (rate_A : ℝ)
  (rate_B : ℝ)
  (t : ℝ)
  (h_rate_A : rate_A = 2 / 75)
  (h_rate_B : rate_B = 1 / t)
  (h_fill_total : 9 * (rate_A + rate_B) + 21 * rate_A = 1) :
  t = 45 := 
sorry

end pipe_B_fill_time_l196_19665


namespace gcd_times_xyz_is_square_l196_19658

theorem gcd_times_xyz_is_square (x y z : ℕ) (h : 1 / (x : ℚ) - 1 / (y : ℚ) = 1 / (z : ℚ)) : 
  ∃ k : ℕ, (Nat.gcd x (Nat.gcd y z) * x * y * z) = k ^ 2 :=
sorry

end gcd_times_xyz_is_square_l196_19658


namespace original_number_of_turtles_l196_19683

-- Define the problem
theorem original_number_of_turtles (T : ℕ) (h1 : 17 = (T + 3 * T - 2) / 2) : T = 9 := by
  sorry

end original_number_of_turtles_l196_19683


namespace find_line_equation_l196_19628

noncomputable def line_equation (x y : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * x - 4

theorem find_line_equation :
  ∃ (x₁ y₁ : ℝ), x₁ = Real.sqrt 3 ∧ y₁ = -3 ∧ ∀ x y, (line_equation x y ↔ 
  (y + 3 = (Real.sqrt 3 / 3) * (x - Real.sqrt 3))) :=
sorry

end find_line_equation_l196_19628


namespace number_of_solutions_l196_19680

-- Define the relevant trigonometric equation
def trig_equation (x : ℝ) : Prop := (Real.cos x)^2 + 3 * (Real.sin x)^2 = 1

-- Define the range for x
def in_range (x : ℝ) : Prop := -20 < x ∧ x < 100

-- Define the predicate that x satisfies both the trig equation and the range condition
def satisfies_conditions (x : ℝ) : Prop := trig_equation x ∧ in_range x

-- The final theorem statement (proof is omitted)
theorem number_of_solutions : 
  ∃ (count : ℕ), count = 38 ∧ ∀ (x : ℝ), satisfies_conditions x ↔ x = k * Real.pi ∧ -20 < k * Real.pi ∧ k * Real.pi < 100 := sorry

end number_of_solutions_l196_19680


namespace complex_neither_sufficient_nor_necessary_real_l196_19603

noncomputable def quadratic_equation_real_roots (a : ℝ) : Prop := 
  (a^2 - 4 * a ≥ 0)

noncomputable def quadratic_equation_complex_roots (a : ℝ) : Prop := 
  (a^2 - 4 * (-a) < 0)

theorem complex_neither_sufficient_nor_necessary_real (a : ℝ) :
  (quadratic_equation_complex_roots a ↔ quadratic_equation_real_roots a) = false := 
sorry

end complex_neither_sufficient_nor_necessary_real_l196_19603


namespace derivative_at_one_l196_19664

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)

theorem derivative_at_one :
  deriv f 1 = -1 / 4 :=
by
  sorry

end derivative_at_one_l196_19664


namespace cube_eq_minus_one_l196_19682

theorem cube_eq_minus_one (x : ℝ) (h : x = -2) : (x + 1) ^ 3 = -1 :=
by
  sorry

end cube_eq_minus_one_l196_19682


namespace Elise_paid_23_dollars_l196_19635

-- Definitions and conditions
def base_price := 3
def cost_per_mile := 4
def distance := 5

-- Desired conclusion (total cost)
def total_cost := base_price + cost_per_mile * distance

-- Theorem statement
theorem Elise_paid_23_dollars : total_cost = 23 := by
  sorry

end Elise_paid_23_dollars_l196_19635


namespace asymptote_hole_sum_l196_19698

noncomputable def number_of_holes (f : ℝ → ℝ) : ℕ := -- Assume a definition to count holes
sorry

noncomputable def number_of_vertical_asymptotes (f : ℝ → ℝ) : ℕ := -- Assume a definition to count vertical asymptotes
sorry

noncomputable def number_of_horizontal_asymptotes (f : ℝ → ℝ) : ℕ := -- Assume a definition to count horizontal asymptotes
sorry

noncomputable def number_of_oblique_asymptotes (f : ℝ → ℝ) : ℕ := -- Assume a definition to count oblique asymptotes
sorry

theorem asymptote_hole_sum :
  let f := λ x => (x^2 + 4*x + 3) / (x^3 - 2*x^2 - x + 2)
  let a := number_of_holes f
  let b := number_of_vertical_asymptotes f
  let c := number_of_horizontal_asymptotes f
  let d := number_of_oblique_asymptotes f
  a + 2 * b + 3 * c + 4 * d = 8 :=
by
  sorry

end asymptote_hole_sum_l196_19698


namespace incorrect_conclusion_C_l196_19677

noncomputable def f (x : ℝ) := (x - 1)^2 * Real.exp x

theorem incorrect_conclusion_C : 
  ¬(∀ x, ∀ ε > 0, ∃ δ > 0, ∀ y, abs (y - x) < δ → abs (f y - f x) ≥ ε) :=
by
  sorry

end incorrect_conclusion_C_l196_19677


namespace density_change_l196_19626

theorem density_change (V : ℝ) (Δa : ℝ) (decrease_percent : ℝ) (initial_volume : V = 27) (edge_increase : Δa = 0.9) : 
    decrease_percent = 8 := 
by 
  sorry

end density_change_l196_19626


namespace total_bedrooms_is_correct_l196_19631

def bedrooms_second_floor : Nat := 2
def bedrooms_first_floor : Nat := 8
def total_bedrooms (b1 b2 : Nat) : Nat := b1 + b2

theorem total_bedrooms_is_correct : total_bedrooms bedrooms_second_floor bedrooms_first_floor = 10 := 
by
  sorry

end total_bedrooms_is_correct_l196_19631


namespace anne_total_bottle_caps_l196_19648

/-- 
Anne initially has 10 bottle caps 
and then finds another 5 bottle caps.
-/
def anne_initial_bottle_caps : ℕ := 10
def anne_found_bottle_caps : ℕ := 5

/-- 
Prove that the total number of bottle caps
Anne ends with is equal to 15.
-/
theorem anne_total_bottle_caps : 
  anne_initial_bottle_caps + anne_found_bottle_caps = 15 :=
by 
  sorry

end anne_total_bottle_caps_l196_19648


namespace trailing_zeros_in_15_factorial_base_15_are_3_l196_19675

/--
Compute the number of trailing zeros in \( 15! \) when expressed in base 15.
-/
def compute_trailing_zeros_in_factorial_base_15 : ℕ :=
  let num_factors_3 := (15 / 3) + (15 / 9)
  let num_factors_5 := (15 / 5)
  min num_factors_3 num_factors_5

theorem trailing_zeros_in_15_factorial_base_15_are_3 :
  compute_trailing_zeros_in_factorial_base_15 = 3 :=
sorry

end trailing_zeros_in_15_factorial_base_15_are_3_l196_19675


namespace marbles_count_l196_19612

variables {g y : ℕ}

theorem marbles_count (h1 : (g - 1)/(g + y - 1) = 1/8)
                      (h2 : g/(g + y - 3) = 1/6) :
                      g + y = 9 :=
by
-- This is just setting up the statements we need to prove the theorem. The actual proof is to be completed.
sorry

end marbles_count_l196_19612


namespace vasya_numbers_l196_19601

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : 
  (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) ∨ (x = y ∧ x = 0) :=
by 
  -- Placeholder to show where the proof would go
  sorry

end vasya_numbers_l196_19601


namespace solve_system_of_equations_l196_19645

theorem solve_system_of_equations :
  ∃ (x y : ℕ), (x + 2 * y = 5) ∧ (3 * x + y = 5) ∧ (x = 1) ∧ (y = 2) :=
by {
  sorry
}

end solve_system_of_equations_l196_19645


namespace soda_preference_count_eq_243_l196_19614

def total_respondents : ℕ := 540
def soda_angle : ℕ := 162
def total_circle_angle : ℕ := 360

theorem soda_preference_count_eq_243 :
  (total_respondents * soda_angle / total_circle_angle) = 243 := 
by 
  sorry

end soda_preference_count_eq_243_l196_19614
