import Mathlib

namespace NUMINAMATH_GPT_tangent_slope_at_1_l771_77191

def f (x : ℝ) : ℝ := x^3 + x^2 + 1

theorem tangent_slope_at_1 : (deriv f 1) = 5 := by
  sorry

end NUMINAMATH_GPT_tangent_slope_at_1_l771_77191


namespace NUMINAMATH_GPT_sqrt_plus_inv_sqrt_eq_l771_77188

noncomputable def sqrt_plus_inv_sqrt (x : ℝ) : ℝ :=
  Real.sqrt x + 1 / Real.sqrt x

theorem sqrt_plus_inv_sqrt_eq (x : ℝ) (h₁ : 0 < x) (h₂ : x + 1 / x = 50) :
  sqrt_plus_inv_sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_GPT_sqrt_plus_inv_sqrt_eq_l771_77188


namespace NUMINAMATH_GPT_compute_difference_of_squares_l771_77173

theorem compute_difference_of_squares :
  (23 + 15) ^ 2 - (23 - 15) ^ 2 = 1380 := by
  sorry

end NUMINAMATH_GPT_compute_difference_of_squares_l771_77173


namespace NUMINAMATH_GPT_derivative_of_y_l771_77182

noncomputable def y (x : ℝ) : ℝ :=
  1/2 * Real.tanh x + 1/(4 * Real.sqrt 2) * Real.log ((1 + Real.sqrt 2 * Real.tanh x) / (1 - Real.sqrt 2 * Real.tanh x))

theorem derivative_of_y (x : ℝ) : 
  (deriv y x) = 1/(Real.cosh x ^ 2 * (1 - Real.sinh x ^ 2)) := 
by
  sorry

end NUMINAMATH_GPT_derivative_of_y_l771_77182


namespace NUMINAMATH_GPT_original_price_of_computer_l771_77197

theorem original_price_of_computer :
  ∃ (P : ℝ), (1.30 * P = 377) ∧ (2 * P = 580) ∧ (P = 290) :=
by
  existsi (290 : ℝ)
  sorry

end NUMINAMATH_GPT_original_price_of_computer_l771_77197


namespace NUMINAMATH_GPT_parabola_through_origin_l771_77155

theorem parabola_through_origin {a b c : ℝ} :
  (c = 0 ↔ ∀ x, (0, 0) = (x, a * x^2 + b * x + c)) :=
sorry

end NUMINAMATH_GPT_parabola_through_origin_l771_77155


namespace NUMINAMATH_GPT_scientific_notation_correct_l771_77112

def distance_moon_km : ℕ := 384000

def scientific_notation (n : ℕ) : ℝ := 3.84 * 10^5

theorem scientific_notation_correct : scientific_notation distance_moon_km = 3.84 * 10^5 := by
  sorry

end NUMINAMATH_GPT_scientific_notation_correct_l771_77112


namespace NUMINAMATH_GPT_candy_initial_count_l771_77124

theorem candy_initial_count (candy_given_first candy_given_second candy_given_third candy_bought candy_eaten candy_left initial_candy : ℕ) 
    (h1 : candy_given_first = 18) 
    (h2 : candy_given_second = 12)
    (h3 : candy_given_third = 25)
    (h4 : candy_bought = 10)
    (h5 : candy_eaten = 7)
    (h6 : candy_left = 16)
    (h_initial : candy_left + candy_eaten = initial_candy - candy_bought - candy_given_first - candy_given_second - candy_given_third):
    initial_candy = 68 := 
by 
  sorry

end NUMINAMATH_GPT_candy_initial_count_l771_77124


namespace NUMINAMATH_GPT_students_with_dog_and_cat_only_l771_77127

theorem students_with_dog_and_cat_only
  (U : Finset (ℕ)) -- Universe of students
  (D C B : Finset (ℕ)) -- Sets of students with dogs, cats, and birds respectively
  (hU : U.card = 50)
  (hD : D.card = 30)
  (hC : C.card = 35)
  (hB : B.card = 10)
  (hIntersection : (D ∩ C ∩ B).card = 5) :
  ((D ∩ C) \ B).card = 25 := 
sorry

end NUMINAMATH_GPT_students_with_dog_and_cat_only_l771_77127


namespace NUMINAMATH_GPT_competition_inequality_l771_77185

variable (a b k : ℕ)

-- Conditions
variable (h1 : b % 2 = 1) 
variable (h2 : b ≥ 3)
variable (h3 : ∀ (J1 J2 : ℕ), J1 ≠ J2 → ∃ num_students : ℕ, num_students ≤ a ∧ num_students ≤ k)

theorem competition_inequality (h1: b % 2 = 1) (h2: b ≥ 3) (h3: ∀ (J1 J2 : ℕ), J1 ≠ J2 → ∃ num_students : ℕ, num_students ≤ a ∧ num_students ≤ k) :
  (k: ℝ) / (a: ℝ) ≥ (b-1: ℝ) / (2*b: ℝ) := sorry

end NUMINAMATH_GPT_competition_inequality_l771_77185


namespace NUMINAMATH_GPT_triangle_area_example_l771_77169

noncomputable def area_triangle (BC AB : ℝ) (B : ℝ) : ℝ :=
  (1 / 2) * BC * AB * Real.sin B

theorem triangle_area_example
  (BC AB : ℝ) (B : ℝ)
  (hBC : BC = 2)
  (hAB : AB = 3)
  (hB : B = Real.pi / 3) :
  area_triangle BC AB B = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_example_l771_77169


namespace NUMINAMATH_GPT_second_horse_revolutions_l771_77184

noncomputable def circumference (radius : ℝ) : ℝ := 2 * Real.pi * radius
noncomputable def distance_traveled (circumference : ℝ) (revolutions : ℕ) : ℝ := circumference * (revolutions : ℝ)
noncomputable def revolutions_needed (distance : ℝ) (circumference : ℝ) : ℕ := ⌊distance / circumference⌋₊

theorem second_horse_revolutions :
  let r1 := 30
  let r2 := 10
  let revolutions1 := 40
  let c1 := circumference r1
  let c2 := circumference r2
  let d1 := distance_traveled c1 revolutions1
  (revolutions_needed d1 c2) = 120 :=
by
  sorry

end NUMINAMATH_GPT_second_horse_revolutions_l771_77184


namespace NUMINAMATH_GPT_max_value_part1_l771_77103

theorem max_value_part1 (a : ℝ) (h : a < 3 / 2) : 2 * a + 4 / (2 * a - 3) + 3 ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_value_part1_l771_77103


namespace NUMINAMATH_GPT_solve_for_x_l771_77111

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 1) (h1 : y = 1 / (4 * x^2 + 2 * x + 1)) : 
  x = 0 ∨ x = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l771_77111


namespace NUMINAMATH_GPT_composite_square_perimeter_l771_77177

theorem composite_square_perimeter (p1 p2 : ℝ) (h1 : p1 = 40) (h2 : p2 = 100) : 
  let s1 := p1 / 4
  let s2 := p2 / 4
  (p1 + p2 - 2 * s1) = 120 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_composite_square_perimeter_l771_77177


namespace NUMINAMATH_GPT_positive_integer_power_of_two_l771_77179

theorem positive_integer_power_of_two (n : ℕ) (hn : 0 < n) :
  (∃ m : ℤ, (2^n - 1) ∣ (m^2 + 9)) ↔ (∃ k : ℕ, n = 2^k) :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_power_of_two_l771_77179


namespace NUMINAMATH_GPT_total_ways_to_choose_gifts_l771_77175

/-- The 6 pairs of zodiac signs -/
def zodiac_pairs : Set (Set String) :=
  {{"Rat", "Ox"}, {"Tiger", "Rabbit"}, {"Dragon", "Snake"}, {"Horse", "Sheep"}, {"Monkey", "Rooster"}, {"Dog", "Pig"}}

/-- The preferences of Students A, B, and C -/
def A_likes : Set String := {"Ox", "Horse"}
def B_likes : Set String := {"Ox", "Dog", "Sheep"}
def C_likes : Set String := {"Rat", "Ox", "Tiger", "Rabbit", "Dragon", "Snake", "Horse", "Sheep", "Monkey", "Rooster", "Dog", "Pig"}

theorem total_ways_to_choose_gifts : 
  True := 
by
  -- We prove that the number of ways is 16
  sorry

end NUMINAMATH_GPT_total_ways_to_choose_gifts_l771_77175


namespace NUMINAMATH_GPT_solution_set_of_inequality_l771_77154

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / x ≤ 3} = {x : ℝ | x < 0} ∪ {x : ℝ | x ≥ 1 / 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l771_77154


namespace NUMINAMATH_GPT_parabola_equation_with_left_focus_l771_77105

theorem parabola_equation_with_left_focus (x y : ℝ) :
  (∀ x y : ℝ, (x^2)/25 + (y^2)/9 = 1 → (y^2 = -16 * x)) :=
by
  sorry

end NUMINAMATH_GPT_parabola_equation_with_left_focus_l771_77105


namespace NUMINAMATH_GPT_max_streetlights_l771_77161

theorem max_streetlights {road_length streetlight_length : ℝ} 
  (h1 : road_length = 1000)
  (h2 : streetlight_length = 1)
  (fully_illuminated : ∀ (n : ℕ), (n * streetlight_length) < road_length)
  : ∃ max_n, max_n = 1998 ∧ (∀ n, n > max_n → (∃ i, streetlight_length * i > road_length)) :=
sorry

end NUMINAMATH_GPT_max_streetlights_l771_77161


namespace NUMINAMATH_GPT_polynomial_roots_r_eq_18_l771_77198

theorem polynomial_roots_r_eq_18
  (a b c : ℂ) 
  (h_roots : Polynomial.roots (Polynomial.C (0 : ℂ) * Polynomial.X^3 + Polynomial.C (5 : ℂ) * Polynomial.X^2 + Polynomial.C (2 : ℂ) * Polynomial.X + Polynomial.C (-8 : ℂ)) = {a, b, c}) 
  (h_ab_roots : Polynomial.roots (Polynomial.C (0 : ℂ) * Polynomial.X^3 + Polynomial.C p * Polynomial.X^2 + Polynomial.C q * Polynomial.X + Polynomial.C r) = {2 * a + b, 2 * b + c, 2 * c + a}) :
  r = 18 := sorry

end NUMINAMATH_GPT_polynomial_roots_r_eq_18_l771_77198


namespace NUMINAMATH_GPT_extra_men_needed_l771_77131

theorem extra_men_needed (total_length : ℝ) (total_days : ℕ) (initial_men : ℕ) (completed_length : ℝ) (days_passed : ℕ) 
  (remaining_length := total_length - completed_length)
  (remaining_days := total_days - days_passed)
  (current_rate := completed_length / days_passed)
  (required_rate := remaining_length / remaining_days)
  (rate_increase := required_rate / current_rate)
  (total_men_needed := initial_men * rate_increase)
  (extra_men_needed := ⌈total_men_needed⌉ - initial_men) :
  total_length = 15 → 
  total_days = 300 → 
  initial_men = 35 → 
  completed_length = 2.5 → 
  days_passed = 100 → 
  extra_men_needed = 53 :=
by
-- Prove that given the conditions, the number of extra men needed is 53
sorry

end NUMINAMATH_GPT_extra_men_needed_l771_77131


namespace NUMINAMATH_GPT_marching_band_total_weight_l771_77147

noncomputable def total_weight : ℕ :=
  let trumpet_weight := 5
  let clarinet_weight := 5
  let trombone_weight := 10
  let tuba_weight := 20
  let drum_weight := 15
  let trumpets := 6
  let clarinets := 9
  let trombones := 8
  let tubas := 3
  let drummers := 2
  (trumpets + clarinets) * trumpet_weight + trombones * trombone_weight + tubas * tuba_weight + drummers * drum_weight

theorem marching_band_total_weight : total_weight = 245 := by
  sorry

end NUMINAMATH_GPT_marching_band_total_weight_l771_77147


namespace NUMINAMATH_GPT_ratio_of_second_to_third_l771_77183

theorem ratio_of_second_to_third (A B C : ℕ) (h1 : A + B + C = 98) (h2 : A * 3 = B * 2) (h3 : B = 30) :
  B * 8 = C * 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_second_to_third_l771_77183


namespace NUMINAMATH_GPT_coplanar_values_l771_77146

namespace CoplanarLines

-- Define parametric equations of the lines
def line1 (t : ℝ) (m : ℝ) : ℝ × ℝ × ℝ := (3 + 2 * t, 2 - t, 5 + m * t)
def line2 (u : ℝ) (m : ℝ) : ℝ × ℝ × ℝ := (4 - m * u, 5 + 3 * u, 6 + 2 * u)

-- Define coplanarity condition
def coplanar_condition (m : ℝ) : Prop :=
  ∃ t u : ℝ, line1 t m = line2 u m

-- Theorem to prove the specific values of m for coplanarity
theorem coplanar_values (m : ℝ) : coplanar_condition m ↔ (m = -13/9 ∨ m = 1) :=
sorry

end CoplanarLines

end NUMINAMATH_GPT_coplanar_values_l771_77146


namespace NUMINAMATH_GPT_largest_decimal_number_l771_77162

theorem largest_decimal_number :
  max (0.9123 : ℝ) (max (0.9912 : ℝ) (max (0.9191 : ℝ) (max (0.9301 : ℝ) (0.9091 : ℝ)))) = 0.9912 :=
by
  sorry

end NUMINAMATH_GPT_largest_decimal_number_l771_77162


namespace NUMINAMATH_GPT_solve_z_l771_77158

variable (z : ℂ) -- Define the variable z in the complex number system
variable (i : ℂ) -- Define the variable i in the complex number system

-- State the conditions: 2 - 3i * z = 4 + 5i * z and i^2 = -1
axiom cond1 : 2 - 3 * i * z = 4 + 5 * i * z
axiom cond2 : i^2 = -1

-- The theorem to prove: z = i / 4
theorem solve_z : z = i / 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_z_l771_77158


namespace NUMINAMATH_GPT_claudia_filled_5oz_glasses_l771_77118

theorem claudia_filled_5oz_glasses :
  ∃ (n : ℕ), n = 6 ∧ 4 * 8 + 15 * 4 + n * 5 = 122 :=
by
  sorry

end NUMINAMATH_GPT_claudia_filled_5oz_glasses_l771_77118


namespace NUMINAMATH_GPT_electricity_consumption_l771_77133

variable (x y : ℝ)

-- y = 0.55 * x
def electricity_fee := 0.55 * x

-- if y = 40.7 then x should be 74
theorem electricity_consumption :
  (∃ x, electricity_fee x = 40.7) → (x = 74) :=
by
  sorry

end NUMINAMATH_GPT_electricity_consumption_l771_77133


namespace NUMINAMATH_GPT_f_m_eq_five_l771_77193

def f (x : ℝ) (a : ℝ) : ℝ :=
  x^3 + a * x + 3

axiom f_neg_m : ∀ (m a : ℝ), f (-m) a = 1

theorem f_m_eq_five (m a : ℝ) (h : f (-m) a = 1) : f m a = 5 :=
  by sorry

end NUMINAMATH_GPT_f_m_eq_five_l771_77193


namespace NUMINAMATH_GPT_sqrt_two_irrational_l771_77137

theorem sqrt_two_irrational : ¬ ∃ (a b : ℕ), b ≠ 0 ∧ (a / b) ^ 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_two_irrational_l771_77137


namespace NUMINAMATH_GPT_other_coin_value_l771_77129

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

end NUMINAMATH_GPT_other_coin_value_l771_77129


namespace NUMINAMATH_GPT_q_l771_77108

-- Definitions for the problem conditions
def slips := 50
def numbers := 12
def slips_per_number := 5
def drawn_slips := 5
def binom := Nat.choose -- Lean function for binomial coefficients

-- Define the probabilities p' and q'
def p' := 12 / (binom slips drawn_slips)
def favorable_q' := (binom numbers 2) * (binom slips_per_number 3) * (binom slips_per_number 2)
def q' := favorable_q' / (binom slips drawn_slips)

-- The statement we need to prove
theorem q'_over_p'_equals_550 : q' / p' = 550 :=
by sorry

end NUMINAMATH_GPT_q_l771_77108


namespace NUMINAMATH_GPT_max_area_of_triangle_l771_77151

theorem max_area_of_triangle (a b c : ℝ) 
  (h1 : ∀ (a b c : ℝ), S = a^2 - (b - c)^2)
  (h2 : b + c = 8) : 
  S ≤ 64 / 17 :=
sorry

end NUMINAMATH_GPT_max_area_of_triangle_l771_77151


namespace NUMINAMATH_GPT_mode_I_swaps_mode_II_swaps_l771_77190

-- Define the original and target strings
def original_sign := "MEGYEI TAKARÉKPÉNZTÁR R. T."
def target_sign := "TATÁR GYERMEK A PÉNZT KÉRI."

-- Define a function for adjacent swaps needed to convert original_sign to target_sign
def adjacent_swaps (orig : String) (target : String) : ℕ := sorry

-- Define a function for any distant swaps needed to convert original_sign to target_sign
def distant_swaps (orig : String) (target : String) : ℕ := sorry

-- The theorems we want to prove
theorem mode_I_swaps : adjacent_swaps original_sign target_sign = 85 := sorry

theorem mode_II_swaps : distant_swaps original_sign target_sign = 11 := sorry

end NUMINAMATH_GPT_mode_I_swaps_mode_II_swaps_l771_77190


namespace NUMINAMATH_GPT_pipe_fill_time_without_leak_l771_77150

theorem pipe_fill_time_without_leak (T : ℝ) (h1 : (1 / 9 : ℝ) = 1 / T - 1 / 4.5) : T = 3 := 
by
  sorry

end NUMINAMATH_GPT_pipe_fill_time_without_leak_l771_77150


namespace NUMINAMATH_GPT_complement_U_A_l771_77160

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {3, 4, 5}

theorem complement_U_A :
  U \ A = {1, 2, 6} := by
  sorry

end NUMINAMATH_GPT_complement_U_A_l771_77160


namespace NUMINAMATH_GPT_super_k_teams_l771_77149

theorem super_k_teams (n : ℕ) (h : n * (n - 1) / 2 = 45) : n = 10 :=
sorry

end NUMINAMATH_GPT_super_k_teams_l771_77149


namespace NUMINAMATH_GPT_cost_of_adult_ticket_is_15_l771_77178

variable (A : ℕ) -- Cost of an adult ticket
variable (total_tickets : ℕ) (cost_child_ticket : ℕ) (total_revenue : ℕ)
variable (adult_tickets_sold : ℕ)

theorem cost_of_adult_ticket_is_15
  (h1 : total_tickets = 522)
  (h2 : cost_child_ticket = 8)
  (h3 : total_revenue = 5086)
  (h4 : adult_tickets_sold = 130) 
  (h5 : (total_tickets - adult_tickets_sold) * cost_child_ticket + adult_tickets_sold * A = total_revenue) :
  A = 15 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_adult_ticket_is_15_l771_77178


namespace NUMINAMATH_GPT_savings_with_discount_l771_77181

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

end NUMINAMATH_GPT_savings_with_discount_l771_77181


namespace NUMINAMATH_GPT_fraction_of_number_l771_77148

variable (N : ℝ) (F : ℝ)

theorem fraction_of_number (h1 : 0.5 * N = F * N + 2) (h2 : N = 8.0) : F = 0.25 := by
  sorry

end NUMINAMATH_GPT_fraction_of_number_l771_77148


namespace NUMINAMATH_GPT_range_of_x_l771_77128

def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem range_of_x :
  {x : ℝ | odot x (x - 2) < 0} = {x : ℝ | -2 < x ∧ x < 1} := 
by sorry

end NUMINAMATH_GPT_range_of_x_l771_77128


namespace NUMINAMATH_GPT_complex_product_l771_77156

theorem complex_product (a b c d : ℤ) (i : ℂ) (h : i^2 = -1) :
  (6 - 7 * i) * (3 + 6 * i) = 60 + 15 * i :=
  by
    -- proof statements would go here
    sorry

end NUMINAMATH_GPT_complex_product_l771_77156


namespace NUMINAMATH_GPT_sum_of_products_l771_77152

theorem sum_of_products {a b c : ℝ}
  (h1 : a ^ 2 + b ^ 2 + c ^ 2 = 138)
  (h2 : a + b + c = 20) :
  a * b + b * c + c * a = 131 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_products_l771_77152


namespace NUMINAMATH_GPT_backpack_pencil_case_combinations_l771_77165

theorem backpack_pencil_case_combinations (backpacks pencil_cases : Fin 2) : 
  (backpacks * pencil_cases) = 4 :=
by 
  sorry

end NUMINAMATH_GPT_backpack_pencil_case_combinations_l771_77165


namespace NUMINAMATH_GPT_dave_non_working_games_l771_77138

def total_games : ℕ := 10
def price_per_game : ℕ := 4
def total_earnings : ℕ := 32

theorem dave_non_working_games : (total_games - (total_earnings / price_per_game)) = 2 := by
  sorry

end NUMINAMATH_GPT_dave_non_working_games_l771_77138


namespace NUMINAMATH_GPT_f_at_4_l771_77142

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f (x-1) = g_inv (x-3)
axiom h2 : ∀ x : ℝ, g_inv (g x) = x
axiom h3 : ∀ x : ℝ, g (g_inv x) = x
axiom h4 : g 5 = 2005

theorem f_at_4 : f 4 = 2008 :=
by
  sorry

end NUMINAMATH_GPT_f_at_4_l771_77142


namespace NUMINAMATH_GPT_exists_lattice_midpoint_among_five_points_l771_77180

-- Definition of lattice points
structure LatticePoint where
  x : ℤ
  y : ℤ

open LatticePoint

-- The theorem we want to prove
theorem exists_lattice_midpoint_among_five_points (A B C D E : LatticePoint) :
    ∃ P Q : LatticePoint, P ≠ Q ∧ (P.x + Q.x) % 2 = 0 ∧ (P.y + Q.y) % 2 = 0 := 
  sorry

end NUMINAMATH_GPT_exists_lattice_midpoint_among_five_points_l771_77180


namespace NUMINAMATH_GPT_halfway_between_fractions_l771_77101

-- Definitions used in the conditions
def one_eighth := (1 : ℚ) / 8
def three_tenths := (3 : ℚ) / 10

-- The mathematical assertion to prove
theorem halfway_between_fractions : (one_eighth + three_tenths) / 2 = 17 / 80 := by
  sorry

end NUMINAMATH_GPT_halfway_between_fractions_l771_77101


namespace NUMINAMATH_GPT_find_x_if_friendly_l771_77170

theorem find_x_if_friendly (x : ℚ) :
    (∃ m n : ℚ, m + n = 66 ∧ m = 7 * x ∧ n = -18) →
    x = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_x_if_friendly_l771_77170


namespace NUMINAMATH_GPT_bus_time_one_way_l771_77153

-- define conditions
def walk_time_one_way := 5 -- 5 minutes for one walk
def total_annual_travel_time_hours := 365 -- 365 hours per year
def work_days_per_year := 365 -- works every day

-- convert annual travel time from hours to minutes
def total_annual_travel_time_minutes := total_annual_travel_time_hours * 60

-- calculate total daily travel time
def total_daily_travel_time := total_annual_travel_time_minutes / work_days_per_year

-- walking time per day
def total_daily_walking_time := (walk_time_one_way * 4)

-- total bus travel time per day
def total_daily_bus_time := total_daily_travel_time - total_daily_walking_time

-- one-way bus time
theorem bus_time_one_way : total_daily_bus_time / 2 = 20 := by
  sorry

end NUMINAMATH_GPT_bus_time_one_way_l771_77153


namespace NUMINAMATH_GPT_perpendicular_condition_l771_77130

-- Definitions of lines
def line1 (x y : ℝ) : Prop := x + y = 0
def line2 (x y : ℝ) (a : ℝ) : Prop := x - a * y = 0

-- Theorem: Prove that a = 1 is a necessary and sufficient condition for the lines
-- line1 and line2 to be perpendicular.
theorem perpendicular_condition (a : ℝ) : 
  (∀ x y : ℝ, line1 x y → line2 x y a) ↔ (a = 1) :=
sorry

end NUMINAMATH_GPT_perpendicular_condition_l771_77130


namespace NUMINAMATH_GPT_mosquito_feedings_to_death_l771_77107

theorem mosquito_feedings_to_death 
  (drops_per_feeding : ℕ := 20) 
  (drops_per_liter : ℕ := 5000) 
  (lethal_blood_loss_liters : ℝ := 3) 
  (drops_per_feeding_liters : ℝ := drops_per_feeding / drops_per_liter) 
  (lethal_feedings : ℝ := lethal_blood_loss_liters / drops_per_feeding_liters) :
  lethal_feedings = 750 := 
by
  sorry

end NUMINAMATH_GPT_mosquito_feedings_to_death_l771_77107


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l771_77113

theorem hyperbola_eccentricity (a b c : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_hyperbola: ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_asymptotes_l1: ∀ x : ℝ, y = (b / a) * x)
  (h_asymptotes_l2: ∀ x : ℝ, y = -(b / a) * x)
  (h_focus: c^2 = a^2 + b^2)
  (h_symmetric: ∀ m : ℝ, m = -c / 2 ∧ (m, (b * c) / (2 * a)) ∈ { p : ℝ × ℝ | p.2 = -(b / a) * p.1 }) :
  (c / a) = 2 := sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l771_77113


namespace NUMINAMATH_GPT_problem_statement_l771_77192

theorem problem_statement (g : ℝ → ℝ) (m k : ℝ) (h₀ : ∀ x, g x = 5 * x - 3)
  (h₁ : 0 < k) (h₂ : 0 < m)
  (h₃ : ∀ x, |g x - 2| < k ↔ |x - 1| < m) : m ≤ k / 5 :=
sorry

end NUMINAMATH_GPT_problem_statement_l771_77192


namespace NUMINAMATH_GPT_P_subsetneq_M_l771_77110

def M := {x : ℝ | x > 1}
def P := {x : ℝ | x^2 - 6*x + 9 = 0}

theorem P_subsetneq_M : P ⊂ M := by
  sorry

end NUMINAMATH_GPT_P_subsetneq_M_l771_77110


namespace NUMINAMATH_GPT_length_of_platform_l771_77174

/--
Problem statement:
A train 450 m long running at 108 km/h crosses a platform in 25 seconds.
Prove that the length of the platform is 300 meters.

Given:
- The train is 450 meters long.
- The train's speed is 108 km/h.
- The train crosses the platform in 25 seconds.

To prove:
The length of the platform is 300 meters.
-/
theorem length_of_platform :
  let train_length := 450
  let train_speed := 108 * (1000 / 3600) -- converting km/h to m/s
  let crossing_time := 25
  let total_distance_covered := train_speed * crossing_time
  let platform_length := total_distance_covered - train_length
  platform_length = 300 := by
  sorry

end NUMINAMATH_GPT_length_of_platform_l771_77174


namespace NUMINAMATH_GPT_problem1_problem2_l771_77102

-- Problem 1
theorem problem1 : (1/4 / 1/5) - 1/4 = 1 := 
by 
  sorry

-- Problem 2
theorem problem2 : ∃ x : ℚ, x + 1/2 * x = 12/5 ∧ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l771_77102


namespace NUMINAMATH_GPT_find_x_value_l771_77144

theorem find_x_value {C S x : ℝ}
  (h1 : C = 100 * (1 + x / 100))
  (h2 : S - C = 10 / 9)
  (h3 : S = 100 * (1 + x / 100)):
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_x_value_l771_77144


namespace NUMINAMATH_GPT_salary_increase_percentage_l771_77163

variable {P : ℝ} (initial_salary : P > 0)

def salary_after_first_year (P: ℝ) : ℝ :=
  P * 1.12

def salary_after_second_year (P: ℝ) : ℝ :=
  (salary_after_first_year P) * 1.12

def salary_after_third_year (P: ℝ) : ℝ :=
  (salary_after_second_year P) * 1.15

theorem salary_increase_percentage (P: ℝ) (h: P > 0) : 
  (salary_after_third_year P - P) / P * 100 = 44 :=
by 
  sorry

end NUMINAMATH_GPT_salary_increase_percentage_l771_77163


namespace NUMINAMATH_GPT_measles_cases_1993_l771_77122

theorem measles_cases_1993 :
  ∀ (cases_1970 cases_1986 cases_2000 : ℕ)
    (rate1 rate2 : ℕ),
  cases_1970 = 600000 →
  cases_1986 = 30000 →
  cases_2000 = 600 →
  rate1 = 35625 →
  rate2 = 2100 →
  cases_1986 - 7 * rate2 = 15300 :=
by {
  sorry
}

end NUMINAMATH_GPT_measles_cases_1993_l771_77122


namespace NUMINAMATH_GPT_smallest_four_digit_number_l771_77139

noncomputable def smallest_four_digit_solution : ℕ := 1011

theorem smallest_four_digit_number (x : ℕ) (h1 : 5 * x ≡ 25 [MOD 20]) (h2 : 3 * x + 10 ≡ 19 [MOD 7]) (h3 : x + 3 ≡ 2 * x [MOD 12]) :
  x = smallest_four_digit_solution :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_number_l771_77139


namespace NUMINAMATH_GPT_min_value_y_l771_77126

theorem min_value_y (x : ℝ) (h : x > 5 / 4) : 
  ∃ y, y = 4*x - 1 + 1 / (4*x - 5) ∧ y ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_min_value_y_l771_77126


namespace NUMINAMATH_GPT_tan_of_45_deg_l771_77168

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_GPT_tan_of_45_deg_l771_77168


namespace NUMINAMATH_GPT_average_weight_of_remaining_boys_l771_77100

theorem average_weight_of_remaining_boys (avg_weight_16: ℝ) (avg_weight_total: ℝ) (weight_16: ℝ) (total_boys: ℝ) (avg_weight_8: ℝ) : 
  (avg_weight_16 = 50.25) → (avg_weight_total = 48.55) → (weight_16 = 16 * avg_weight_16) → (total_boys = 24) → 
  (total_weight = total_boys * avg_weight_total) → (weight_16 + 8 * avg_weight_8 = total_weight) → avg_weight_8 = 45.15 :=
by
  intros h_avg_weight_16 h_avg_weight_total h_weight_16 h_total_boys h_total_weight h_equation
  sorry

end NUMINAMATH_GPT_average_weight_of_remaining_boys_l771_77100


namespace NUMINAMATH_GPT_solution_of_inequality_is_correct_l771_77117

-- Inequality condition (x-1)/(2x+1) ≤ 0
def inequality (x : ℝ) : Prop := (x - 1) / (2 * x + 1) ≤ 0 

-- Conditions
def condition1 (x : ℝ) : Prop := (x - 1) * (2 * x + 1) ≤ 0
def condition2 (x : ℝ) : Prop := 2 * x + 1 ≠ 0

-- Combined condition
def combined_condition (x : ℝ) : Prop := condition1 x ∧ condition2 x

-- Solution set
def solution_set : Set ℝ := { x | -1/2 < x ∧ x ≤ 1 }

-- Theorem statement
theorem solution_of_inequality_is_correct :
  ∀ x : ℝ, inequality x ↔ combined_condition x ∧ x ∈ solution_set :=
by
  sorry

end NUMINAMATH_GPT_solution_of_inequality_is_correct_l771_77117


namespace NUMINAMATH_GPT_sum_sequence_conjecture_l771_77132

theorem sum_sequence_conjecture (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n : ℕ+, a n = (8 * n) / ((2 * n - 1) ^ 2 * (2 * n + 1) ^ 2)) →
  (∀ n : ℕ+, S n = (S n + a (n + 1))) →
  (∀ n : ℕ+, S 1 = 8 / 9) →
  (∀ n : ℕ+, S n = ((2 * n + 1) ^ 2 - 1) / (2 * n + 1) ^ 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_sequence_conjecture_l771_77132


namespace NUMINAMATH_GPT_chord_intersects_inner_circle_l771_77120

noncomputable def probability_chord_intersects_inner_circle
  (r1 r2 : ℝ) (h1 : r1 = 2) (h2 : r2 = 5) : ℝ :=
0.098

theorem chord_intersects_inner_circle :
  probability_chord_intersects_inner_circle 2 5 rfl rfl = 0.098 :=
sorry

end NUMINAMATH_GPT_chord_intersects_inner_circle_l771_77120


namespace NUMINAMATH_GPT_denis_dartboard_score_l771_77143

theorem denis_dartboard_score :
  ∀ P1 P2 P3 P4 : ℕ,
  P1 = 30 → 
  P2 = 38 → 
  P3 = 41 → 
  P1 + P2 + P3 + P4 = 4 * ((P1 + P2 + P3 + P4) / 4) → 
  P4 = 34 :=
by
  intro P1 P2 P3 P4 hP1 hP2 hP3 hTotal
  have hSum := hP1.symm ▸ hP2.symm ▸ hP3.symm ▸ hTotal
  sorry

end NUMINAMATH_GPT_denis_dartboard_score_l771_77143


namespace NUMINAMATH_GPT_graph_n_plus_k_odd_l771_77134

-- Definitions and assumptions
variable {V : Type} [Fintype V] [DecidableEq V] (G : SimpleGraph V)
variable (n k : ℕ)
variable (hG : Fintype.card V = n)
variable (hCond : ∀ (S : Finset V), S.card = k → (G.commonNeighborsFinset S).card % 2 = 1)

-- Goal
theorem graph_n_plus_k_odd :
  (n + k) % 2 = 1 :=
sorry

end NUMINAMATH_GPT_graph_n_plus_k_odd_l771_77134


namespace NUMINAMATH_GPT_initial_number_of_people_l771_77116

theorem initial_number_of_people (X : ℕ) (h : ((X - 10) + 15 = 17)) : X = 12 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_people_l771_77116


namespace NUMINAMATH_GPT_largest_integer_base8_square_l771_77121

theorem largest_integer_base8_square :
  ∃ (N : ℕ), (N^2 >= 8^3) ∧ (N^2 < 8^4) ∧ (N = 63 ∧ N % 8 = 7) := sorry

end NUMINAMATH_GPT_largest_integer_base8_square_l771_77121


namespace NUMINAMATH_GPT_total_wheels_in_parking_lot_l771_77194

def num_cars : ℕ := 14
def num_bikes : ℕ := 10
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

theorem total_wheels_in_parking_lot :
  (num_cars * wheels_per_car) + (num_bikes * wheels_per_bike) = 76 :=
by
  sorry

end NUMINAMATH_GPT_total_wheels_in_parking_lot_l771_77194


namespace NUMINAMATH_GPT_find_a_l771_77159

def diamond (a b : ℝ) : ℝ := 3 * a - b^2

theorem find_a (a : ℝ) (h : diamond a 6 = 15) : a = 17 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l771_77159


namespace NUMINAMATH_GPT_total_birds_on_fence_l771_77187

theorem total_birds_on_fence (initial_birds additional_birds storks : ℕ) 
  (h1 : initial_birds = 6) 
  (h2 : additional_birds = 4) 
  (h3 : storks = 8) :
  initial_birds + additional_birds + storks = 18 :=
by
  sorry

end NUMINAMATH_GPT_total_birds_on_fence_l771_77187


namespace NUMINAMATH_GPT_route_time_difference_l771_77166

-- Define the conditions of the problem
def first_route_time : ℕ := 
  let time_uphill := 6
  let time_path := 2 * time_uphill
  let time_finish := (time_uphill + time_path) / 3
  time_uphill + time_path + time_finish

def second_route_time : ℕ := 
  let time_flat_path := 14
  let time_finish := 2 * time_flat_path
  time_flat_path + time_finish

-- Prove the question
theorem route_time_difference : second_route_time - first_route_time = 18 :=
by
  sorry

end NUMINAMATH_GPT_route_time_difference_l771_77166


namespace NUMINAMATH_GPT_third_person_profit_share_l771_77104

noncomputable def investment_first : ℤ := 9000
noncomputable def investment_second : ℤ := investment_first + 2000
noncomputable def investment_third : ℤ := investment_second - 3000
noncomputable def investment_fourth : ℤ := 2 * investment_third
noncomputable def investment_fifth : ℤ := investment_fourth + 4000
noncomputable def total_investment : ℤ := investment_first + investment_second + investment_third + investment_fourth + investment_fifth

noncomputable def total_profit : ℤ := 25000
noncomputable def third_person_share : ℚ := (investment_third : ℚ) / (total_investment : ℚ) * (total_profit : ℚ)

theorem third_person_profit_share :
  third_person_share = 3076.92 := sorry

end NUMINAMATH_GPT_third_person_profit_share_l771_77104


namespace NUMINAMATH_GPT_garden_fencing_l771_77189

/-- A rectangular garden has a length of 50 yards and the width is half the length.
    Prove that the total amount of fencing needed to enclose the garden is 150 yards. -/
theorem garden_fencing : 
  ∀ (length width : ℝ), 
  length = 50 ∧ width = length / 2 → 
  2 * (length + width) = 150 :=
by
  intros length width
  rintro ⟨h1, h2⟩
  sorry

end NUMINAMATH_GPT_garden_fencing_l771_77189


namespace NUMINAMATH_GPT_simplifyExpression_l771_77119

theorem simplifyExpression (a b c d : Int) (ha : a = -2) (hb : b = -6) (hc : c = -3) (hd : d = 2) :
  (a + b - c - d = -2 - 6 + 3 - 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_simplifyExpression_l771_77119


namespace NUMINAMATH_GPT_arccos_sin_eq_l771_77172

open Real

-- Definitions from the problem conditions
noncomputable def radians := π / 180

-- The theorem we need to prove
theorem arccos_sin_eq : arccos (sin 3) = 3 - (π / 2) :=
by
  sorry

end NUMINAMATH_GPT_arccos_sin_eq_l771_77172


namespace NUMINAMATH_GPT_decagon_interior_angle_measure_l771_77106

-- Define the type for a regular polygon
structure RegularPolygon (n : Nat) :=
  (interior_angle_sum : Nat := (n - 2) * 180)
  (side_count : Nat := n)
  (regularity : Prop := True)  -- All angles are equal

-- Define the degree measure of an interior angle of a regular polygon
def interiorAngle (p : RegularPolygon 10) : Nat :=
  (p.interior_angle_sum) / p.side_count

-- The theorem to be proved
theorem decagon_interior_angle_measure : 
  ∀ (p : RegularPolygon 10), interiorAngle p = 144 := by
  -- The proof will be here, but for now, we use sorry
  sorry

end NUMINAMATH_GPT_decagon_interior_angle_measure_l771_77106


namespace NUMINAMATH_GPT_angle_B_in_triangle_is_pi_over_6_l771_77125

theorem angle_B_in_triangle_is_pi_over_6
  (a b c : ℝ)
  (A B C : ℝ)
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : A + B + C = π)
  (h₅ : b * (Real.cos C) / (Real.cos B) + c = (2 * Real.sqrt 3 / 3) * a) :
  B = π / 6 :=
by sorry

end NUMINAMATH_GPT_angle_B_in_triangle_is_pi_over_6_l771_77125


namespace NUMINAMATH_GPT_sqrt_one_over_four_eq_pm_half_l771_77164

theorem sqrt_one_over_four_eq_pm_half : Real.sqrt (1 / 4) = 1 / 2 ∨ Real.sqrt (1 / 4) = - (1 / 2) := by
  sorry

end NUMINAMATH_GPT_sqrt_one_over_four_eq_pm_half_l771_77164


namespace NUMINAMATH_GPT_taco_cost_l771_77157

theorem taco_cost (T E : ℝ) (h1 : 2 * T + 3 * E = 7.80) (h2 : 3 * T + 5 * E = 12.70) : T = 0.90 := 
by 
  sorry

end NUMINAMATH_GPT_taco_cost_l771_77157


namespace NUMINAMATH_GPT_hyperbola_equation_of_focus_and_asymptote_l771_77167

theorem hyperbola_equation_of_focus_and_asymptote :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (2 * a) ^ 2 + (2 * b) ^ 2 = 25 ∧ b / a = 2 ∧ 
  (∀ x y : ℝ, (y = 2 * x + 10) → (x = -5) ∧ (y = 0)) ∧ 
  (∀ x y : ℝ, (x ^ 2 / 5 - y ^ 2 / 20 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_equation_of_focus_and_asymptote_l771_77167


namespace NUMINAMATH_GPT_williams_tips_fraction_l771_77186

theorem williams_tips_fraction
  (A : ℝ) -- average tips for months other than August
  (h : ∀ A, A > 0) -- assuming some positivity constraint for non-degenerate mean
  (h_august : A ≠ 0) -- assuming average can’t be zero
  (august_tips : ℝ := 10 * A)
  (other_months_tips : ℝ := 6 * A)
  (total_tips : ℝ := 16 * A) :
  (august_tips / total_tips) = (5 / 8) := 
sorry

end NUMINAMATH_GPT_williams_tips_fraction_l771_77186


namespace NUMINAMATH_GPT_angleC_is_36_l771_77135

theorem angleC_is_36 
  (p q r : ℝ)  -- fictitious types for lines, as Lean needs a type here
  (A B C : ℝ)  -- Angles as Real numbers
  (hpq : p = q)  -- Line p is parallel to line q (represented equivalently for Lean)
  (h : A = 1/4 * B)
  (hr : B + C = 180)
  (vert_opposite : C = A) :
  C = 36 := 
by
  sorry

end NUMINAMATH_GPT_angleC_is_36_l771_77135


namespace NUMINAMATH_GPT_apple_selling_price_l771_77171

theorem apple_selling_price (CP SP Loss : ℝ) (h₀ : CP = 18) (h₁ : Loss = (1/6) * CP) (h₂ : SP = CP - Loss) : SP = 15 :=
  sorry

end NUMINAMATH_GPT_apple_selling_price_l771_77171


namespace NUMINAMATH_GPT_probability_blue_face_eq_one_third_l771_77145

-- Define the necessary conditions
def numberOfFaces : Nat := 12
def numberOfBlueFaces : Nat := 4

-- Define the term representing the probability
def probabilityOfBlueFace : ℚ := numberOfBlueFaces / numberOfFaces

-- The theorem to prove that the probability is 1/3
theorem probability_blue_face_eq_one_third :
  probabilityOfBlueFace = (1 : ℚ) / 3 :=
  by
  sorry

end NUMINAMATH_GPT_probability_blue_face_eq_one_third_l771_77145


namespace NUMINAMATH_GPT_other_ticket_price_l771_77109

theorem other_ticket_price (total_tickets : ℕ) (total_sales : ℝ) (cheap_tickets : ℕ) (cheap_price : ℝ) (expensive_tickets : ℕ) (expensive_price : ℝ) :
  total_tickets = 380 →
  total_sales = 1972.50 →
  cheap_tickets = 205 →
  cheap_price = 4.50 →
  expensive_tickets = 380 - 205 →
  205 * 4.50 + expensive_tickets * expensive_price = 1972.50 →
  expensive_price = 6.00 :=
by
  intros
  -- proof will be filled here
  sorry

end NUMINAMATH_GPT_other_ticket_price_l771_77109


namespace NUMINAMATH_GPT_composite_divisor_bound_l771_77140

theorem composite_divisor_bound (n : ℕ) (hn : ¬Prime n ∧ 1 < n) : 
  ∃ a : ℕ, 1 < a ∧ a ≤ Int.sqrt (n : ℤ) ∧ a ∣ n :=
sorry

end NUMINAMATH_GPT_composite_divisor_bound_l771_77140


namespace NUMINAMATH_GPT_train_speed_kph_l771_77195

-- Definitions based on conditions
def time_seconds : ℕ := 9
def length_meters : ℕ := 135
def conversion_factor : ℕ := 36 -- 3.6 represented as an integer by multiplying both sides by 10

-- The proof statement
theorem train_speed_kph : (length_meters * conversion_factor / 10 / time_seconds = 54) :=
by
  sorry

end NUMINAMATH_GPT_train_speed_kph_l771_77195


namespace NUMINAMATH_GPT_find_r_s_l771_77114

theorem find_r_s (r s : ℚ) :
  (-3)^5 - 2*(-3)^4 + 3*(-3)^3 - r*(-3)^2 + s*(-3) - 8 = 0 ∧
  2^5 - 2*(2^4) + 3*(2^3) - r*(2^2) + s*2 - 8 = 0 →
  (r, s) = (-482/15, -1024/15) :=
by
  sorry

end NUMINAMATH_GPT_find_r_s_l771_77114


namespace NUMINAMATH_GPT_consecutive_log_sum_l771_77136

theorem consecutive_log_sum : 
  ∃ c d: ℤ, (c + 1 = d) ∧ (c < Real.logb 5 125) ∧ (Real.logb 5 125 < d) ∧ (c + d = 5) :=
sorry

end NUMINAMATH_GPT_consecutive_log_sum_l771_77136


namespace NUMINAMATH_GPT_three_digit_sum_of_factorials_l771_77199

theorem three_digit_sum_of_factorials : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ (n = 145) ∧ 
  (∃ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 ∧ 
    1 ≤ d1 ∧ d1 < 10 ∧ 1 ≤ d2 ∧ d2 < 10 ∧ 1 ≤ d3 ∧ d3 < 10 ∧ 
    (d1 * d1.factorial + d2 * d2.factorial + d3 * d3.factorial = n)) :=
  by
  sorry

end NUMINAMATH_GPT_three_digit_sum_of_factorials_l771_77199


namespace NUMINAMATH_GPT_Tammy_runs_10_laps_per_day_l771_77123

theorem Tammy_runs_10_laps_per_day
  (total_distance_per_week : ℕ)
  (track_length : ℕ)
  (days_per_week : ℕ)
  (h1 : total_distance_per_week = 3500)
  (h2 : track_length = 50)
  (h3 : days_per_week = 7) :
  (total_distance_per_week / track_length) / days_per_week = 10 := by
  sorry

end NUMINAMATH_GPT_Tammy_runs_10_laps_per_day_l771_77123


namespace NUMINAMATH_GPT_ratio_of_areas_l771_77115

theorem ratio_of_areas (x y l : ℝ)
  (h1 : 2 * (x + 3 * y) = 2 * (l + y))
  (h2 : 2 * x + l = 3 * y) :
  (x * 3 * y) / (l * y) = 3 / 7 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l771_77115


namespace NUMINAMATH_GPT_extra_profit_is_60000_l771_77141

theorem extra_profit_is_60000 (base_house_cost special_house_cost base_house_price special_house_price : ℝ) :
  (special_house_cost = base_house_cost + 100000) →
  (special_house_price = 1.5 * base_house_price) →
  (base_house_price = 320000) →
  (special_house_price - base_house_price - 100000 = 60000) :=
by
  -- Definitions and conditions
  intro h1 h2 h3
  -- Placeholder for the eventual proof
  sorry

end NUMINAMATH_GPT_extra_profit_is_60000_l771_77141


namespace NUMINAMATH_GPT_sets_are_equal_l771_77176

theorem sets_are_equal :
  let M := {x | ∃ k : ℤ, x = 2 * k + 1}
  let N := {x | ∃ k : ℤ, x = 4 * k + 1 ∨ x = 4 * k - 1}
  M = N :=
by
  sorry

end NUMINAMATH_GPT_sets_are_equal_l771_77176


namespace NUMINAMATH_GPT_range_of_p_l771_77196

theorem range_of_p (p : ℝ) (a_n b_n : ℕ → ℝ)
  (ha : ∀ n, a_n n = -n + p)
  (hb : ∀ n, b_n n = 3^(n-4))
  (C_n : ℕ → ℝ)
  (hC : ∀ n, C_n n = if a_n n ≥ b_n n then a_n n else b_n n)
  (hc : ∀ n : ℕ, n ≥ 1 → C_n n > C_n 4) :
  4 < p ∧ p < 7 :=
sorry

end NUMINAMATH_GPT_range_of_p_l771_77196
