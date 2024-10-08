import Mathlib

namespace dinosaur_book_cost_l96_96758

-- Define the constants for costs and savings/needs
def dict_cost : ℕ := 11
def cookbook_cost : ℕ := 7
def savings : ℕ := 8
def needed : ℕ := 29
def total_cost : ℕ := savings + needed
def dino_cost : ℕ := 19

-- Mathematical statement to prove
theorem dinosaur_book_cost :
  dict_cost + dino_cost + cookbook_cost = total_cost :=
by
  -- The proof steps would go here
  sorry

end dinosaur_book_cost_l96_96758


namespace smallest_four_digit_divisible_by_25_l96_96566

theorem smallest_four_digit_divisible_by_25 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 25 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 25 = 0 → n ≤ m := by
  -- Prove that the smallest four-digit number divisible by 25 is 1000
  sorry

end smallest_four_digit_divisible_by_25_l96_96566


namespace lcm_eq_792_l96_96896

-- Define the integers
def a : ℕ := 8
def b : ℕ := 9
def c : ℕ := 11

-- Define their prime factorizations (included for clarity, though not directly necessary)
def a_factorization : a = 2^3 := rfl
def b_factorization : b = 3^2 := rfl
def c_factorization : c = 11 := rfl

-- Define the LCM function
def lcm_abc := Nat.lcm (Nat.lcm a b) c

-- Prove that lcm of a, b, c is 792
theorem lcm_eq_792 : lcm_abc = 792 := 
by
  -- Include the necessary properties of LCM and prime factorizations if necessary
  sorry

end lcm_eq_792_l96_96896


namespace part1_part2_l96_96584

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 5 * Real.log x + a * x^2 - 6 * x
noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := 5 / x + 2 * a * x - 6

theorem part1 (a : ℝ) (h_tangent : f_prime 1 a = 0) : a = 1 / 2 :=
by {
  sorry
}

theorem part2 (a : ℝ) (h_a : a = 1/2) :
  (∀ x, 0 < x → x < 1 → f_prime x a > 0) ∧
  (∀ x, 5 < x → f_prime x a > 0) ∧
  (∀ x, 1 < x → x < 5 → f_prime x a < 0) :=
by {
  sorry
}

end part1_part2_l96_96584


namespace Billie_has_2_caps_l96_96691

-- Conditions as definitions in Lean
def Sammy_caps : ℕ := 8
def Janine_caps : ℕ := Sammy_caps - 2
def Billie_caps : ℕ := Janine_caps / 3

-- Problem statement to prove
theorem Billie_has_2_caps : Billie_caps = 2 := by
  sorry

end Billie_has_2_caps_l96_96691


namespace Davey_Barbeck_ratio_is_1_l96_96463

-- Assume the following given conditions as definitions in Lean
variables (guitars Davey Barbeck : ℕ)

-- Condition 1: Davey has 18 guitars
def Davey_has_18 : Prop := Davey = 18

-- Condition 2: Barbeck has the same number of guitars as Davey
def Davey_eq_Barbeck : Prop := Davey = Barbeck

-- The problem statement: Prove the ratio of the number of guitars Davey has to the number of guitars Barbeck has is 1:1
theorem Davey_Barbeck_ratio_is_1 (h1 : Davey_has_18 Davey) (h2 : Davey_eq_Barbeck Davey Barbeck) :
  Davey / Barbeck = 1 :=
by
  sorry

end Davey_Barbeck_ratio_is_1_l96_96463


namespace truncated_pyramid_distance_l96_96891

noncomputable def distance_from_plane_to_base
  (a b : ℝ) (α : ℝ) : ℝ :=
  (a * (a - b) * Real.tan α) / (3 * a - b)

theorem truncated_pyramid_distance
  (a b : ℝ) (α : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_α : 0 < α) :
  (a * (a - b) * Real.tan α) / (3 * a - b) = distance_from_plane_to_base a b α :=
by
  sorry

end truncated_pyramid_distance_l96_96891


namespace bob_total_spend_in_usd_l96_96390

theorem bob_total_spend_in_usd:
  let coffee_cost_yen := 250
  let sandwich_cost_yen := 150
  let yen_to_usd := 110
  (coffee_cost_yen + sandwich_cost_yen) / yen_to_usd = 3.64 := by
  sorry

end bob_total_spend_in_usd_l96_96390


namespace domain_of_sqrt_fraction_l96_96407

theorem domain_of_sqrt_fraction {x : ℝ} (h1 : x - 3 ≥ 0) (h2 : 7 - x > 0) :
  3 ≤ x ∧ x < 7 :=
by {
  sorry
}

end domain_of_sqrt_fraction_l96_96407


namespace total_cost_function_range_of_x_minimum_cost_when_x_is_2_l96_96538

def transportation_cost (x : ℕ) : ℕ :=
  300 * x + 500 * (12 - x) + 400 * (10 - x) + 800 * (x - 2)

theorem total_cost_function (x : ℕ) : transportation_cost x = 200 * x + 8400 := by
  -- Simply restate the definition in the theorem form
  sorry

theorem range_of_x (x : ℕ) : 2 ≤ x ∧ x ≤ 10 := by
  -- Provide necessary constraints in theorem form
  sorry

theorem minimum_cost_when_x_is_2 : transportation_cost 2 = 8800 := by
  -- Final cost at minimum x
  sorry

end total_cost_function_range_of_x_minimum_cost_when_x_is_2_l96_96538


namespace truncated_trigonal_pyramid_circumscribed_sphere_l96_96444

theorem truncated_trigonal_pyramid_circumscribed_sphere
  (h R_1 R_2 : ℝ)
  (O_1 T_1 O_2 T_2 : ℝ)
  (circumscribed : ∃ r : ℝ, h = 2 * r)
  (sphere_touches_lower_base : ∀ P, dist P T_1 = r)
  (sphere_touches_upper_base : ∀ Q, dist Q T_2 = r)
  (dist_O1_T1 : ℝ)
  (dist_O2_T2 : ℝ) :
  R_1 * R_2 * h^2 = (R_1^2 - dist_O1_T1^2) * (R_2^2 - dist_O2_T2^2) :=
sorry

end truncated_trigonal_pyramid_circumscribed_sphere_l96_96444


namespace sum_of_solutions_eqn_l96_96835

theorem sum_of_solutions_eqn : 
  (∀ x : ℝ, -48 * x^2 + 100 * x + 200 = 0 → False) → 
  (-100 / -48) = (25 / 12) :=
by
  intros
  sorry

end sum_of_solutions_eqn_l96_96835


namespace john_sleep_total_hours_l96_96401

-- Defining the conditions provided in the problem statement
def days_with_3_hours : ℕ := 2
def sleep_per_day_3_hours : ℕ := 3
def remaining_days : ℕ := 7 - days_with_3_hours
def recommended_sleep : ℕ := 8
def percentage_sleep : ℝ := 0.6

-- Expressing the proof problem statement
theorem john_sleep_total_hours :
  (days_with_3_hours * sleep_per_day_3_hours
  + remaining_days * (percentage_sleep * recommended_sleep)) = 30 := by
  sorry

end john_sleep_total_hours_l96_96401


namespace fraction_given_away_is_three_fifths_l96_96081

variable (initial_bunnies : ℕ) (final_bunnies : ℕ) (kittens_per_bunny : ℕ)

def fraction_given_away (given_away : ℕ) (initial_bunnies : ℕ) : ℚ :=
  given_away / initial_bunnies

theorem fraction_given_away_is_three_fifths 
  (initial_bunnies : ℕ := 30) (final_bunnies : ℕ := 54) (kittens_per_bunny : ℕ := 2)
  (h : final_bunnies = initial_bunnies + kittens_per_bunny * (initial_bunnies - 18)) : 
  fraction_given_away 18 initial_bunnies = 3 / 5 :=
by
  sorry

end fraction_given_away_is_three_fifths_l96_96081


namespace flight_height_l96_96124

theorem flight_height (flights : ℕ) (step_height_in_inches : ℕ) (total_steps : ℕ) 
    (H1 : flights = 9) (H2 : step_height_in_inches = 18) (H3 : total_steps = 60) : 
    (total_steps * step_height_in_inches) / 12 / flights = 10 :=
by
  sorry

end flight_height_l96_96124


namespace sequence_general_formula_l96_96723

-- Definitions according to conditions in a)
def seq (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * n + 1

def S (n : ℕ) (seq : ℕ → ℕ) : ℕ :=
  n * seq (n + 1) - 3 * n^2 - 4 * n

-- The proof goal
theorem sequence_general_formula (n : ℕ) (h : 0 < n) :
  seq n = 2 * n + 1 :=
by
  sorry

end sequence_general_formula_l96_96723


namespace one_fifth_greater_than_decimal_by_term_l96_96053

noncomputable def one_fifth := (1 : ℝ) / 5
noncomputable def decimal_value := 20000001 / 10^8
noncomputable def term := 1 / (5 * 10^8)

theorem one_fifth_greater_than_decimal_by_term :
  one_fifth > decimal_value ∧ one_fifth - decimal_value = term :=
  sorry

end one_fifth_greater_than_decimal_by_term_l96_96053


namespace contrapositive_equiv_l96_96513

variable (x : Type)

theorem contrapositive_equiv (Q R : x → Prop) :
  (∀ x, Q x → R x) ↔ (∀ x, ¬ (R x) → ¬ (Q x)) :=
by
  sorry

end contrapositive_equiv_l96_96513


namespace planar_molecules_l96_96533

structure Molecule :=
  (name : String)
  (formula : String)
  (is_planar : Bool)

def propylene : Molecule := 
  { name := "Propylene", formula := "C3H6", is_planar := False }

def vinyl_chloride : Molecule := 
  { name := "Vinyl Chloride", formula := "C2H3Cl", is_planar := True }

def benzene : Molecule := 
  { name := "Benzene", formula := "C6H6", is_planar := True }

def toluene : Molecule := 
  { name := "Toluene", formula := "C7H8", is_planar := False }

theorem planar_molecules : 
  (vinyl_chloride.is_planar = True) ∧ (benzene.is_planar = True) := 
by
  sorry

end planar_molecules_l96_96533


namespace soccer_team_games_l96_96227

theorem soccer_team_games :
  ∃ G : ℕ, G % 2 = 0 ∧ 
           45 / 100 * 36 = 16 ∧ 
           ∀ R, R = G - 36 → (16 + 75 / 100 * R) = 62 / 100 * G ∧
           G = 84 :=
sorry

end soccer_team_games_l96_96227


namespace not_black_cows_count_l96_96680

theorem not_black_cows_count (total_cows : ℕ) (black_cows : ℕ) (h1 : total_cows = 18) (h2 : black_cows = 5 + total_cows / 2) :
  total_cows - black_cows = 4 :=
by 
  -- Insert the actual proof here
  sorry

end not_black_cows_count_l96_96680


namespace Intersect_A_B_l96_96416

-- Defining the sets A and B according to the problem's conditions
def A : Set ℤ := {1, 2}
def B : Set ℤ := {x ∈ Set.univ | x^2 - 5*x + 4 < 0}

-- Prove that the intersection of A and B is {2}
theorem Intersect_A_B : A ∩ B = {2} := by
  sorry

end Intersect_A_B_l96_96416


namespace number_of_cloth_bags_l96_96223

-- Definitions based on the conditions
def dozen := 12

def total_peaches : ℕ := 5 * dozen
def peaches_in_knapsack : ℕ := 12
def peaches_per_bag : ℕ := 2 * peaches_in_knapsack

-- The proof statement
theorem number_of_cloth_bags :
  (total_peaches - peaches_in_knapsack) / peaches_per_bag = 2 := by
  sorry

end number_of_cloth_bags_l96_96223


namespace johns_age_l96_96940

-- Define variables for ages of John and Matt
variables (J M : ℕ)

-- Define the conditions based on the problem statement
def condition1 : Prop := M = 4 * J - 3
def condition2 : Prop := J + M = 52

-- The goal: prove that John is 11 years old
theorem johns_age (J M : ℕ) (h1 : condition1 J M) (h2 : condition2 J M) : J = 11 := by
  -- proof will go here
  sorry

end johns_age_l96_96940


namespace max_jars_in_crate_l96_96945

-- Define the conditions given in the problem
def side_length_cardboard_box := 20 -- in cm
def jars_per_box := 8
def crate_width := 80 -- in cm
def crate_length := 120 -- in cm
def crate_height := 60 -- in cm
def volume_box := side_length_cardboard_box ^ 3
def volume_crate := crate_width * crate_length * crate_height
def boxes_per_crate := volume_crate / volume_box
def max_jars_per_crate := boxes_per_crate * jars_per_box

-- Statement that needs to be proved
theorem max_jars_in_crate : max_jars_per_crate = 576 := sorry

end max_jars_in_crate_l96_96945


namespace total_coins_last_month_l96_96664

theorem total_coins_last_month (m s : ℝ) : 
  (100 = 1.25 * m) ∧ (100 = 0.80 * s) → m + s = 205 :=
by sorry

end total_coins_last_month_l96_96664


namespace identity_function_l96_96046

theorem identity_function {f : ℕ → ℕ} (h : ∀ a b : ℕ, 0 < a → 0 < b → a - f b ∣ a * f a - b * f b) :
  ∀ a : ℕ, 0 < a → f a = a :=
by
  sorry

end identity_function_l96_96046


namespace first_player_winning_strategy_l96_96650

def game_strategy (S : ℕ) : Prop :=
  ∃ k, (1 ≤ k ∧ k ≤ 5 ∧ (S - k) % 6 = 1)

theorem first_player_winning_strategy : game_strategy 100 :=
sorry

end first_player_winning_strategy_l96_96650


namespace divisor_is_four_l96_96667

theorem divisor_is_four (d n : ℤ) (k j : ℤ) 
  (h1 : n % d = 3) 
  (h2 : 2 * n % d = 2): d = 4 :=
sorry

end divisor_is_four_l96_96667


namespace some_number_value_l96_96586

theorem some_number_value (a : ℤ) (x1 x2 : ℤ)
  (h1 : x1 + a = 10) (h2 : x2 + a = -10) (h_sum : x1 + x2 = 20) : a = -10 :=
by
  sorry

end some_number_value_l96_96586


namespace line_through_point_area_T_l96_96199

variable (a T : ℝ)

def triangle_line_equation (a T : ℝ) : Prop :=
  ∃ y x : ℝ, (a^2 * y + 2 * T * x - 2 * a * T = 0) ∧ (y = -((2 * T)/a^2) * x + (2 * T) / a) ∧ (x ≥ 0) ∧ (y ≥ 0)

theorem line_through_point_area_T (a T : ℝ) (h₁ : a > 0) (h₂ : T > 0) :
  triangle_line_equation a T :=
sorry

end line_through_point_area_T_l96_96199


namespace proof_statements_BCD_l96_96847

variable (a b : ℝ)

theorem proof_statements_BCD (h1 : a > b) (h2 : b > 0) :
  (-1 / b < -1 / a) ∧ (a^2 * b > a * b^2) ∧ (a / b > b / a) :=
by
  sorry

end proof_statements_BCD_l96_96847


namespace no_solution_when_k_equals_7_l96_96863

noncomputable def no_solution_eq (k x : ℝ) : Prop :=
  (x - 3) / (x - 4) = (x - k) / (x - 8)
  
theorem no_solution_when_k_equals_7 :
  ∀ x : ℝ, x ≠ 4 → x ≠ 8 → ¬ no_solution_eq 7 x :=
by
  sorry

end no_solution_when_k_equals_7_l96_96863


namespace question1_question2_l96_96393

variables (θ : ℝ)

-- Condition: tan θ = 2
def tan_theta_eq : Prop := Real.tan θ = 2

-- Question 1: Prove (4 * sin θ - 2 * cos θ) / (3 * sin θ + 5 * cos θ) = 6 / 11
theorem question1 (h : tan_theta_eq θ) : (4 * Real.sin θ - 2 * Real.cos θ) / (3 * Real.sin θ + 5 * Real.cos θ) = 6 / 11 :=
by
  sorry

-- Question 2: Prove 1 - 4 * sin θ * cos θ + 2 * cos² θ = -1 / 5
theorem question2 (h : tan_theta_eq θ) : 1 - 4 * Real.sin θ * Real.cos θ + 2 * (Real.cos θ)^2 = -1 / 5 :=
by
  sorry

end question1_question2_l96_96393


namespace hockey_championship_max_k_volleyball_championship_max_k_l96_96984

theorem hockey_championship_max_k : ∃ (k : ℕ), 0 < k ∧ k ≤ 20 ∧ k = 18 :=
by
  -- proof goes here
  sorry

theorem volleyball_championship_max_k : ∃ (k : ℕ), 0 < k ∧ k ≤ 20 ∧ k = 15 :=
by
  -- proof goes here
  sorry

end hockey_championship_max_k_volleyball_championship_max_k_l96_96984


namespace cookies_with_five_cups_of_flour_l96_96405

-- Define the conditions
def initial_cookies : ℕ := 24
def initial_flour : ℕ := 3
def additional_flour : ℕ := 5

-- State the problem
theorem cookies_with_five_cups_of_flour :
  (initial_cookies / initial_flour) * additional_flour = 40 :=
by
  -- Placeholder for proof
  sorry

end cookies_with_five_cups_of_flour_l96_96405


namespace integer_roots_l96_96008

-- Define the polynomial
def polynomial (x : ℤ) : ℤ := x^3 - 4 * x^2 - 7 * x + 10

-- Define the proof problem statement
theorem integer_roots :
  {x : ℤ | polynomial x = 0} = {1, -2, 5} :=
by
  sorry

end integer_roots_l96_96008


namespace imaginary_part_of_z_is_1_l96_96340

def z := Complex.ofReal 0 + Complex.ofReal 1 * Complex.I * (Complex.ofReal 1 + Complex.ofReal 2 * Complex.I)
theorem imaginary_part_of_z_is_1 : z.im = 1 := by
  sorry

end imaginary_part_of_z_is_1_l96_96340


namespace profit_last_month_l96_96732

variable (gas_expenses earnings_per_lawn lawns_mowed extra_income profit : ℤ)

def toms_profit (gas_expenses earnings_per_lawn lawns_mowed extra_income : ℤ) : ℤ :=
  (lawns_mowed * earnings_per_lawn + extra_income) - gas_expenses

theorem profit_last_month :
  toms_profit 17 12 3 10 = 29 :=
by
  rw [toms_profit]
  sorry

end profit_last_month_l96_96732


namespace initial_percentage_of_milk_l96_96528

theorem initial_percentage_of_milk (P : ℝ) :
  (P / 100) * 60 = (68 / 100) * 74.11764705882354 → P = 84 :=
by
  sorry

end initial_percentage_of_milk_l96_96528


namespace angle_A_is_pi_over_4_l96_96120

theorem angle_A_is_pi_over_4
  (A B C : ℝ)
  (a b c : ℝ)
  (h : a^2 = b^2 + c^2 - 2 * b * c * Real.sin A) :
  A = Real.pi / 4 :=
  sorry

end angle_A_is_pi_over_4_l96_96120


namespace find_number_l96_96615

def valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  ((n % 10 = 6 ∧ ¬ (n % 7 = 0)) ∨ (¬ (n % 10 = 6) ∧ n % 7 = 0)) ∧
  ((n > 26 ∧ ¬ (n % 10 = 8)) ∨ (¬ (n > 26) ∧ n % 10 = 8)) ∧
  ((n % 13 = 0 ∧ ¬ (n < 27)) ∨ (¬ (n % 13 = 0) ∧ n < 27))

theorem find_number : ∃ n : ℕ, valid_number n ∧ n = 91 := by
  sorry

end find_number_l96_96615


namespace range_of_sum_l96_96928

theorem range_of_sum (x y : ℝ) (h : 9 * x^2 + 16 * y^2 = 144) : 
  ∃ a b : ℝ, (x + y + 10 ≥ a) ∧ (x + y + 10 ≤ b) ∧ a = 5 ∧ b = 15 := 
sorry

end range_of_sum_l96_96928


namespace cube_volume_in_pyramid_and_cone_l96_96256

noncomputable def volume_of_cube
  (base_side : ℝ)
  (pyramid_height : ℝ)
  (cone_radius : ℝ)
  (cone_height : ℝ)
  (cube_side_length : ℝ) : ℝ := 
  cube_side_length^3

theorem cube_volume_in_pyramid_and_cone :
  let base_side := 2
  let pyramid_height := Real.sqrt 3
  let cone_radius := Real.sqrt 2
  let cone_height := Real.sqrt 3
  let cube_side_length := (Real.sqrt 6) / (Real.sqrt 2 + Real.sqrt 3)
  volume_of_cube base_side pyramid_height cone_radius cone_height cube_side_length = (6 * Real.sqrt 6) / 17 :=
by sorry

end cube_volume_in_pyramid_and_cone_l96_96256


namespace f_is_even_f_monotonic_increase_range_of_a_for_solutions_l96_96708

-- Define the function f(x) = x^2 - 2a|x|
def f (a x : ℝ) : ℝ := x^2 - 2 * a * |x|

-- Given a > 0
variable (a : ℝ) (ha : a > 0)

-- 1. Prove that f(x) is an even function.
theorem f_is_even : ∀ x : ℝ, f a x = f a (-x) := sorry

-- 2. Prove the interval of monotonic increase for f(x) when x > 0 is [a, +∞).
theorem f_monotonic_increase (x : ℝ) (hx : x > 0) : a ≤ x → ∃ c : ℝ, x ≤ c := sorry

-- 3. Prove the range of values for a for which the equation f(x) = -1 has solutions is a ≥ 1.
theorem range_of_a_for_solutions : (∃ x : ℝ, f a x = -1) ↔ 1 ≤ a := sorry

end f_is_even_f_monotonic_increase_range_of_a_for_solutions_l96_96708


namespace point_in_second_quadrant_l96_96117

theorem point_in_second_quadrant (m : ℝ) (h : 2 > 0 ∧ m < 0) : m < 0 :=
by
  sorry

end point_in_second_quadrant_l96_96117


namespace parabola_hyperbola_tangent_l96_96020

-- Definitions of the parabola and hyperbola
def parabola (x : ℝ) : ℝ := x^2 + 4
def hyperbola (x y : ℝ) (m : ℝ) : Prop := y^2 - m*x^2 = 1

-- Tangency condition stating that the parabola and hyperbola are tangent implies m = 8 + 2*sqrt(15)
theorem parabola_hyperbola_tangent (m : ℝ) :
  (∀ x y : ℝ, parabola x = y → hyperbola x y m) → m = 8 + 2 * Real.sqrt 15 :=
by
  sorry

end parabola_hyperbola_tangent_l96_96020


namespace multiple_statements_l96_96233

theorem multiple_statements (c d : ℤ)
  (hc4 : ∃ k : ℤ, c = 4 * k)
  (hd8 : ∃ k : ℤ, d = 8 * k) :
  (∃ k : ℤ, d = 4 * k) ∧
  (∃ k : ℤ, c + d = 4 * k) ∧
  (∃ k : ℤ, c + d = 2 * k) :=
by
  sorry

end multiple_statements_l96_96233


namespace division_quotient_l96_96766

theorem division_quotient (dividend divisor remainder quotient : ℕ)
  (H1 : dividend = 190)
  (H2 : divisor = 21)
  (H3 : remainder = 1)
  (H4 : dividend = divisor * quotient + remainder) : quotient = 9 :=
by {
  sorry
}

end division_quotient_l96_96766


namespace angle_sum_around_point_l96_96493

theorem angle_sum_around_point (x : ℝ) (h : 2 * x + 140 = 360) : x = 110 := 
  sorry

end angle_sum_around_point_l96_96493


namespace gcd_n4_plus_27_n_plus_3_l96_96682

theorem gcd_n4_plus_27_n_plus_3 (n : ℕ) (h_pos : n > 9) : 
  gcd (n^4 + 27) (n + 3) = if n % 3 = 0 then 3 else 1 := 
by
  sorry

end gcd_n4_plus_27_n_plus_3_l96_96682


namespace triangle_BDC_is_isosceles_l96_96675

-- Define the given conditions
variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (AB AC BC AD DC : ℝ)
variables (a : ℝ)
variables (α : ℝ)

-- Given conditions
def is_isosceles_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (AB AC : ℝ) : Prop :=
AB = AC

def angle_BAC_120 (α : ℝ) : Prop :=
α = 120

def point_D_extension (AD AB : ℝ) : Prop :=
AD = 2 * AB

-- Let triangle ABC be isosceles with AB = AC and angle BAC = 120 degrees
axiom isosceles_triangle_ABC (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (AB AC : ℝ) : is_isosceles_triangle A B C AB AC

axiom angle_BAC (α : ℝ) : angle_BAC_120 α

axiom point_D (AD AB : ℝ) : point_D_extension AD AB

-- Prove that triangle BDC is isosceles
theorem triangle_BDC_is_isosceles 
  (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AB AC BC AD DC : ℝ) 
  (α : ℝ) 
  (h1 : is_isosceles_triangle A B C AB AC)
  (h2 : angle_BAC_120 α)
  (h3 : point_D_extension AD AB) :
  BC = DC :=
sorry

end triangle_BDC_is_isosceles_l96_96675


namespace f_g_2_eq_36_l96_96623

def f (x : ℤ) : ℤ := x * x
def g (x : ℤ) : ℤ := 4 * x - 2

theorem f_g_2_eq_36 : f (g 2) = 36 :=
by
  sorry

end f_g_2_eq_36_l96_96623


namespace range_of_a_l96_96833

def decreasing_range (a : ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ 4 → y ≤ 4 → x < y → (x^2 + 2 * (a - 1) * x + 2) ≥ (y^2 + 2 * (a - 1) * y + 2)

theorem range_of_a (a : ℝ) : decreasing_range a ↔ a ≤ -3 := 
  sorry

end range_of_a_l96_96833


namespace math_problem_l96_96953

theorem math_problem : 2^5 + (5^2 / 5^1) - 3^3 = 10 :=
by
  sorry

end math_problem_l96_96953


namespace log_identity_l96_96170

theorem log_identity :
  (Real.log 25 / Real.log 10) - 2 * (Real.log (1 / 2) / Real.log 10) = 2 :=
by
  sorry

end log_identity_l96_96170


namespace wendy_boxes_l96_96841

theorem wendy_boxes (x : ℕ) (w_brother : ℕ) (total : ℕ) (candy_per_box : ℕ) 
    (h_w_brother : w_brother = 6) 
    (h_candy_per_box : candy_per_box = 3) 
    (h_total : total = 12) 
    (h_equation : 3 * x + w_brother = total) : 
    x = 2 :=
by
  -- Proof would go here
  sorry

end wendy_boxes_l96_96841


namespace new_salary_correct_l96_96621

-- Define the initial salary and percentage increase as given in the conditions
def initial_salary : ℝ := 10000
def percentage_increase : ℝ := 0.02

-- Define the function that calculates the new salary after a percentage increase
def new_salary (initial_salary : ℝ) (percentage_increase : ℝ) : ℝ :=
  initial_salary + (initial_salary * percentage_increase)

-- The theorem statement that proves the new salary is €10,200
theorem new_salary_correct :
  new_salary initial_salary percentage_increase = 10200 := by
  sorry

end new_salary_correct_l96_96621


namespace initial_bacteria_count_l96_96114

theorem initial_bacteria_count (doubling_time : ℕ) (initial_time : ℕ) (initial_bacteria : ℕ) 
(final_bacteria : ℕ) (doubling_rate : initial_time / doubling_time = 8 ∧ final_bacteria = 524288) : 
  initial_bacteria = 2048 :=
by
  sorry

end initial_bacteria_count_l96_96114


namespace factorial_expression_l96_96088

theorem factorial_expression :
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / Nat.factorial 8 = 1 := by
  sorry

end factorial_expression_l96_96088


namespace total_expense_in_decade_l96_96910

/-- Definition of yearly expense on car insurance -/
def yearly_expense : ℕ := 2000

/-- Definition of the number of years in a decade -/
def years_in_decade : ℕ := 10

/-- Proof that the total expense in a decade is 20000 dollars -/
theorem total_expense_in_decade : yearly_expense * years_in_decade = 20000 :=
by
  sorry

end total_expense_in_decade_l96_96910


namespace ratio_second_shop_to_shirt_l96_96653

-- Define the initial conditions in Lean
def initial_amount : ℕ := 55
def spent_on_shirt : ℕ := 7
def final_amount : ℕ := 27

-- Define the amount spent in the second shop calculation
def spent_in_second_shop (i_amt s_shirt f_amt : ℕ) : ℕ :=
  (i_amt - s_shirt) - f_amt

-- Define the ratio calculation
def ratio (a b : ℕ) : ℕ := a / b

-- Lean 4 statement proving the ratio of amounts
theorem ratio_second_shop_to_shirt : 
  ratio (spent_in_second_shop initial_amount spent_on_shirt final_amount) spent_on_shirt = 3 := 
by
  sorry

end ratio_second_shop_to_shirt_l96_96653


namespace wheat_flour_one_third_l96_96540

theorem wheat_flour_one_third (recipe_cups: ℚ) (third_recipe: ℚ) 
  (h1: recipe_cups = 5 + 2 / 3) (h2: third_recipe = recipe_cups / 3) :
  third_recipe = 1 + 8 / 9 :=
by
  sorry

end wheat_flour_one_third_l96_96540


namespace paint_room_alone_l96_96165

theorem paint_room_alone (x : ℝ) (hx : (1 / x) + (1 / 4) = 1 / 1.714) : x = 3 :=
by sorry

end paint_room_alone_l96_96165


namespace quadratic_solution_l96_96903

theorem quadratic_solution (x : ℝ) : x^2 - 2 * x = 0 ↔ (x = 0 ∨ x = 2) := by
  sorry

end quadratic_solution_l96_96903


namespace probability_X_eq_Y_l96_96150

theorem probability_X_eq_Y
  (x y : ℝ)
  (h1 : -5 * Real.pi ≤ x ∧ x ≤ 5 * Real.pi)
  (h2 : -5 * Real.pi ≤ y ∧ y ≤ 5 * Real.pi)
  (h3 : Real.cos (Real.cos x) = Real.cos (Real.cos y)) :
  (∃ N : ℕ, N = 100 ∧ ∃ M : ℕ, M = 11 ∧ M / N = (11 : ℝ) / 100) :=
by sorry

end probability_X_eq_Y_l96_96150


namespace range_of_a_l96_96221

theorem range_of_a :
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) ↔ -1 ≤ a ∧ a ≤ 3 := 
by
  sorry

end range_of_a_l96_96221


namespace gcd_n4_plus_16_n_plus_3_eq_1_l96_96536

theorem gcd_n4_plus_16_n_plus_3_eq_1 (n : ℕ) (h : n > 16) : gcd (n^4 + 16) (n + 3) = 1 := 
sorry

end gcd_n4_plus_16_n_plus_3_eq_1_l96_96536


namespace woman_work_completion_days_l96_96899

def work_completion_days_man := 6
def work_completion_days_boy := 9
def work_completion_days_combined := 3

theorem woman_work_completion_days : 
  (1 / work_completion_days_man + W + 1 / work_completion_days_boy = 1 / work_completion_days_combined) →
  W = 1 / 18 → 
  1 / W = 18 :=
by
  intros h₁ h₂
  sorry

end woman_work_completion_days_l96_96899


namespace parabola_vertex_l96_96358

theorem parabola_vertex :
  (∃ h k : ℝ, ∀ x : ℝ, (y : ℝ) = (x - 2)^2 + 5 ∧ h = 2 ∧ k = 5) :=
sorry

end parabola_vertex_l96_96358


namespace product_of_prs_l96_96265

theorem product_of_prs
  (p r s : ℕ)
  (H1 : 4 ^ p + 4 ^ 3 = 272)
  (H2 : 3 ^ r + 27 = 54)
  (H3 : 2 ^ (s + 2) + 10 = 42) : 
  p * r * s = 27 :=
sorry

end product_of_prs_l96_96265


namespace Rockham_Soccer_League_members_l96_96245

theorem Rockham_Soccer_League_members (sock_cost tshirt_cost cap_cost total_cost members : ℕ) (h1 : sock_cost = 6) (h2 : tshirt_cost = sock_cost + 10) (h3 : cap_cost = 3) (h4 : total_cost = 4620) (h5 : total_cost = 50 * members) : members = 92 :=
by
  sorry

end Rockham_Soccer_League_members_l96_96245


namespace negation_of_proposition_range_of_m_l96_96033

noncomputable def proposition (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * x - m - 1 < 0

theorem negation_of_proposition (m : ℝ) : ¬ proposition m ↔ ∀ x : ℝ, x^2 + 2 * x - m - 1 ≥ 0 :=
sorry

theorem range_of_m (m : ℝ) : proposition m → m > -2 :=
sorry

end negation_of_proposition_range_of_m_l96_96033


namespace length_of_second_platform_l96_96175

-- Given conditions
def length_of_train : ℕ := 310
def length_of_first_platform : ℕ := 110
def time_to_cross_first_platform : ℕ := 15
def time_to_cross_second_platform : ℕ := 20

-- Calculated based on conditions
def total_distance_first_platform : ℕ :=
  length_of_train + length_of_first_platform

def speed_of_train : ℕ :=
  total_distance_first_platform / time_to_cross_first_platform

def total_distance_second_platform : ℕ :=
  speed_of_train * time_to_cross_second_platform

-- Statement to prove
theorem length_of_second_platform :
  total_distance_second_platform = length_of_train + 250 := sorry

end length_of_second_platform_l96_96175


namespace length_width_difference_l96_96741

theorem length_width_difference (L W : ℝ) 
  (h1 : W = 1/2 * L) 
  (h2 : L * W = 578) : L - W = 17 :=
sorry

end length_width_difference_l96_96741


namespace inequality_solution_l96_96951

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, a * x^2 + (a - 1) * x + (a - 1) < 0) ↔ a < -1/3 :=
by
  sorry

end inequality_solution_l96_96951


namespace fractional_equation_root_l96_96195

theorem fractional_equation_root (k : ℚ) (x : ℚ) (h : (2 * k) / (x - 1) - 3 / (1 - x) = 1) : k = -3 / 2 :=
sorry

end fractional_equation_root_l96_96195


namespace geometric_sequence_min_value_l96_96997

theorem geometric_sequence_min_value
  (s : ℝ) (b1 b2 b3 : ℝ)
  (h1 : b1 = 2)
  (h2 : b2 = 2 * s)
  (h3 : b3 = 2 * s ^ 2) :
  ∃ (s : ℝ), 3 * b2 + 4 * b3 = -9 / 8 :=
by
  sorry

end geometric_sequence_min_value_l96_96997


namespace flagpole_breaking_height_l96_96618

theorem flagpole_breaking_height (x : ℝ) (h_pos : 0 < x) (h_ineq : x < 6)
    (h_pythagoras : (x^2 + 2^2 = 6^2)) : x = Real.sqrt 10 :=
by sorry

end flagpole_breaking_height_l96_96618


namespace ratio_red_to_yellow_l96_96456

structure MugCollection where
  total_mugs : ℕ
  red_mugs : ℕ
  blue_mugs : ℕ
  yellow_mugs : ℕ
  other_mugs : ℕ
  colors : ℕ

def HannahCollection : MugCollection :=
  { total_mugs := 40,
    red_mugs := 6,
    blue_mugs := 6 * 3,
    yellow_mugs := 12,
    other_mugs := 4,
    colors := 4 }

theorem ratio_red_to_yellow
  (hc : MugCollection)
  (h_total : hc.total_mugs = 40)
  (h_blue : hc.blue_mugs = 3 * hc.red_mugs)
  (h_yellow : hc.yellow_mugs = 12)
  (h_other : hc.other_mugs = 4)
  (h_colors : hc.colors = 4) :
  hc.red_mugs / hc.yellow_mugs = 1 / 2 := by
  sorry

end ratio_red_to_yellow_l96_96456


namespace anne_ben_charlie_difference_l96_96746

def sales_tax_rate : ℝ := 0.08
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25
def charlie_discount_rate : ℝ := 0.15

def anne_total : ℝ := (original_price * (1 + sales_tax_rate)) * (1 - discount_rate)
def ben_total : ℝ := (original_price * (1 - discount_rate)) * (1 + sales_tax_rate)
def charlie_total : ℝ := (original_price * (1 - charlie_discount_rate)) * (1 + sales_tax_rate)

def anne_minus_ben_minus_charlie : ℝ := anne_total - ben_total - charlie_total

theorem anne_ben_charlie_difference : anne_minus_ben_minus_charlie = -12.96 :=
by
  sorry

end anne_ben_charlie_difference_l96_96746


namespace angle_E_degree_l96_96784

-- Given conditions
variables {E F G H : ℝ} -- degrees of the angles in quadrilateral EFGH

-- Condition 1: The angles satisfy a specific ratio
axiom angle_ratio : E = 3 * F ∧ E = 2 * G ∧ E = 6 * H

-- Condition 2: The sum of the angles in the quadrilateral is 360 degrees
axiom angle_sum : E + (E / 3) + (E / 2) + (E / 6) = 360

-- Prove the degree measure of angle E is 180 degrees
theorem angle_E_degree : E = 180 :=
by
  sorry

end angle_E_degree_l96_96784


namespace change_is_13_82_l96_96239

def sandwich_cost : ℝ := 5
def num_sandwiches : ℕ := 3
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.05
def payment : ℝ := 20 + 5 + 3

def total_cost_before_discount : ℝ := num_sandwiches * sandwich_cost
def discount_amount : ℝ := total_cost_before_discount * discount_rate
def discounted_cost : ℝ := total_cost_before_discount - discount_amount
def tax_amount : ℝ := discounted_cost * tax_rate
def total_cost_after_tax : ℝ := discounted_cost + tax_amount

def change (payment total_cost : ℝ) : ℝ := payment - total_cost

theorem change_is_13_82 : change payment total_cost_after_tax = 13.82 := 
by
  -- Proof will be provided here
  sorry

end change_is_13_82_l96_96239


namespace exists_hamiltonian_path_l96_96783

theorem exists_hamiltonian_path (n : ℕ) (cities : Fin n → Type) (roads : ∀ (i j : Fin n), cities i → cities j → Prop) 
(road_one_direction : ∀ i j (c1 : cities i) (c2 : cities j), roads i j c1 c2 → ¬ roads j i c2 c1) :
∃ start : Fin n, ∃ path : Fin n → Fin n, ∀ i j : Fin n, i ≠ j → path i ≠ path j :=
sorry

end exists_hamiltonian_path_l96_96783


namespace sum_ge_six_l96_96261

theorem sum_ge_six (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b + b * c + c * a ≥ 12) : a + b + c ≥ 6 :=
by
  sorry

end sum_ge_six_l96_96261


namespace power_of_three_divides_an_l96_96315

theorem power_of_three_divides_an (a : ℕ → ℕ) (k : ℕ) (h1 : a 1 = 3)
  (h2 : ∀ n, a (n + 1) = ((3 * (a n)^2 + 1) / 2) - a n)
  (h3 : ∃ m, n = 3^m) :
  3^(k + 1) ∣ a (3^k) :=
sorry

end power_of_three_divides_an_l96_96315


namespace find_k_solution_l96_96645

noncomputable def vec1 : ℝ × ℝ := (3, -4)
noncomputable def vec2 : ℝ × ℝ := (5, 8)
noncomputable def target_norm : ℝ := 3 * Real.sqrt 10

theorem find_k_solution : ∃ k : ℝ, 0 ≤ k ∧ ‖(k * vec1.1 - vec2.1, k * vec1.2 - vec2.2)‖ = target_norm ∧ k = 0.0288 :=
by
  sorry

end find_k_solution_l96_96645


namespace linear_equation_check_l96_96441

theorem linear_equation_check : 
  (∃ a b : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x + b = 1)) ∧ 
  ¬ (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (∀ x y : ℝ, a * x + b * y = 3)) ∧ 
  ¬ (∀ x : ℝ, x^2 - 2 * x = 0) ∧ 
  ¬ (∀ x : ℝ, x - 1 / x = 0) := 
sorry

end linear_equation_check_l96_96441


namespace initial_alcohol_percentage_l96_96745

theorem initial_alcohol_percentage (P : ℚ) (initial_volume : ℚ) (added_alcohol : ℚ) (added_water : ℚ)
  (final_percentage : ℚ) (final_volume : ℚ) (alcohol_volume_in_initial_solution : ℚ) :
  initial_volume = 40 ∧ 
  added_alcohol = 3.5 ∧ 
  added_water = 6.5 ∧ 
  final_percentage = 0.11 ∧ 
  final_volume = 50 ∧ 
  alcohol_volume_in_initial_solution = (P / 100) * initial_volume ∧ 
  alcohol_volume_in_initial_solution + added_alcohol = final_percentage * final_volume
  → P = 5 :=
by
  sorry

end initial_alcohol_percentage_l96_96745


namespace ratio_of_pentagon_side_to_rectangle_width_l96_96526

-- Definitions based on the conditions
def pentagon_perimeter : ℝ := 60
def rectangle_perimeter : ℝ := 60
def rectangle_length (w : ℝ) : ℝ := 2 * w

-- The statement to be proven
theorem ratio_of_pentagon_side_to_rectangle_width :
  ∀ w : ℝ, 2 * (rectangle_length w + w) = rectangle_perimeter → (pentagon_perimeter / 5) / w = 6 / 5 :=
by
  sorry

end ratio_of_pentagon_side_to_rectangle_width_l96_96526


namespace function_symmetry_l96_96812

noncomputable def f (ω : ℝ) (x : ℝ) := 2 * Real.sin (ω * x + (Real.pi / 6))

theorem function_symmetry (ω : ℝ) (hω : ω > 0) (hT : (2 * Real.pi / ω) = 4 * Real.pi) :
  ∃ (k : ℤ), f ω (2 * k * Real.pi - Real.pi / 3) = f ω 0 := by
  sorry

end function_symmetry_l96_96812


namespace total_point_value_of_test_l96_96855

theorem total_point_value_of_test (total_questions : ℕ) (five_point_questions : ℕ) 
  (ten_point_questions : ℕ) (points_5 : ℕ) (points_10 : ℕ) 
  (h1 : total_questions = 30) (h2 : five_point_questions = 20) 
  (h3 : ten_point_questions = total_questions - five_point_questions) 
  (h4 : points_5 = 5) (h5 : points_10 = 10) : 
  five_point_questions * points_5 + ten_point_questions * points_10 = 200 :=
by
  sorry

end total_point_value_of_test_l96_96855


namespace percentage_both_correct_l96_96734

variable (A B : Type) 

noncomputable def percentage_of_test_takers_correct_first : ℝ := 0.85
noncomputable def percentage_of_test_takers_correct_second : ℝ := 0.70
noncomputable def percentage_of_test_takers_neither_correct : ℝ := 0.05

theorem percentage_both_correct :
  percentage_of_test_takers_correct_first + 
  percentage_of_test_takers_correct_second - 
  (1 - percentage_of_test_takers_neither_correct) = 0.60 := by
  sorry

end percentage_both_correct_l96_96734


namespace cat_litter_container_weight_l96_96639

theorem cat_litter_container_weight :
  (∀ (cost_container : ℕ) (pounds_per_litterbox : ℕ) (cost_total : ℕ) (days : ℕ),
    cost_container = 21 ∧ pounds_per_litterbox = 15 ∧ cost_total = 210 ∧ days = 210 → 
    ∀ (weeks : ℕ), weeks = days / 7 →
    ∀ (containers : ℕ), containers = cost_total / cost_container →
    ∀ (cost_per_container : ℕ), cost_per_container = cost_total / containers →
    (∃ (pounds_per_container : ℕ), pounds_per_container = cost_container / cost_per_container ∧ pounds_per_container = 3)) :=
by
  intros cost_container pounds_per_litterbox cost_total days
  intros h weeks hw containers hc containers_cost hc_cost
  sorry

end cat_litter_container_weight_l96_96639


namespace geometric_sequence_a5_l96_96127

theorem geometric_sequence_a5 {a : ℕ → ℝ} 
  (h_geom : ∃ r, ∀ n, a (n + 1) = r * a n) 
  (h_a3 : a 3 = -4) 
  (h_a7 : a 7 = -16) : 
  a 5 = -8 := 
sorry

end geometric_sequence_a5_l96_96127


namespace remainder_55_57_div_8_l96_96834

def remainder (a b n : ℕ) := (a * b) % n

theorem remainder_55_57_div_8 : remainder 55 57 8 = 7 := by
  -- proof omitted
  sorry

end remainder_55_57_div_8_l96_96834


namespace find_D_l96_96378

variables (A B C D : ℤ)
axiom h1 : A + C = 15
axiom h2 : A - B = 1
axiom h3 : C + C = A
axiom h4 : B - D = 2
axiom h5 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

theorem find_D : D = 7 :=
by sorry

end find_D_l96_96378


namespace pairs_of_old_roller_skates_l96_96156

def cars := 2
def bikes := 2
def trash_can := 1
def tricycle := 1
def car_wheels := 4
def bike_wheels := 2
def trash_can_wheels := 2
def tricycle_wheels := 3
def total_wheels := 25

def roller_skates_wheels := 2
def skates_per_pair := 2

theorem pairs_of_old_roller_skates : (total_wheels - (cars * car_wheels + bikes * bike_wheels + trash_can * trash_can_wheels + tricycle * tricycle_wheels)) / roller_skates_wheels / skates_per_pair = 2 := by
  sorry

end pairs_of_old_roller_skates_l96_96156


namespace alice_steps_l96_96224

noncomputable def num_sticks (n : ℕ) : ℕ :=
  (n + 1 : ℕ) ^ 2

theorem alice_steps (n : ℕ) (h : num_sticks n = 169) : n = 13 :=
by sorry

end alice_steps_l96_96224


namespace range_of_m_min_value_of_7a_4b_l96_96417

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x - 1| + |x + 1| - m ≥ 0) → m ≤ 2 :=
sorry

theorem min_value_of_7a_4b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
    (h_eq : 2 / (3 * a + b) + 1 / (a + 2 * b) = 2) : 7 * a + 4 * b ≥ 9 / 2 :=
sorry

end range_of_m_min_value_of_7a_4b_l96_96417


namespace range_of_a_minus_abs_b_l96_96285

theorem range_of_a_minus_abs_b (a b : ℝ) (h₁ : 1 < a ∧ a < 3) (h₂ : -4 < b ∧ b < 2) : 
  -3 < a - |b| ∧ a - |b| < 3 :=
sorry

end range_of_a_minus_abs_b_l96_96285


namespace isosceles_right_triangle_sums_l96_96913

theorem isosceles_right_triangle_sums (m n : ℝ)
  (h1: (1 * 2 + m * m + 2 * n) = 0)
  (h2: (1 + m^2 + 4) = (4 + m^2 + n^2)) :
  m + n = -1 :=
by {
  sorry
}

end isosceles_right_triangle_sums_l96_96913


namespace gcd_of_1237_and_1849_l96_96637

def gcd_1237_1849 : ℕ := 1

theorem gcd_of_1237_and_1849 : Nat.gcd 1237 1849 = gcd_1237_1849 := by
  sorry

end gcd_of_1237_and_1849_l96_96637


namespace distance_from_origin_to_line_AB_is_sqrt6_div_3_l96_96206

open Real

structure Point where
  x : ℝ
  y : ℝ

def ellipse (p : Point) : Prop :=
  p.x^2 / 2 + p.y^2 = 1

def left_focus : Point := ⟨-1, 0⟩

def line_through_focus (t : ℝ) (p : Point) : Prop :=
  p.x = t * p.y - 1

def origin : Point := ⟨0, 0⟩

def perpendicular (A B : Point) : Prop :=
  A.x * B.x + A.y * B.y = 0

noncomputable def distance (O : Point) (A B : Point) : ℝ :=
  let a := A.y - B.y
  let b := B.x - A.x
  let c := A.x * B.y - A.y * B.x
  abs (a * O.x + b * O.y + c) / sqrt (a^2 + b^2)

theorem distance_from_origin_to_line_AB_is_sqrt6_div_3 
  (A B : Point)
  (hA_on_ellipse : ellipse A)
  (hB_on_ellipse : ellipse B)
  (h_line_through_focus : ∃ t : ℝ, line_through_focus t A ∧ line_through_focus t B)
  (h_perpendicular : perpendicular A B) :
  distance origin A B = sqrt 6 / 3 := sorry

end distance_from_origin_to_line_AB_is_sqrt6_div_3_l96_96206


namespace a_4_value_l96_96744

def seq (n : ℕ) : ℚ :=
  if n = 0 then 0 -- To handle ℕ index starting from 0.
  else if n = 1 then 1
  else seq (n - 1) + 1 / ((n:ℚ) * (n-1))

noncomputable def a_4 : ℚ := seq 4

theorem a_4_value : a_4 = 7 / 4 := 
  by sorry

end a_4_value_l96_96744


namespace pool_capacity_percentage_l96_96263

noncomputable def hose_rate := 60 -- cubic feet per minute
noncomputable def pool_width := 80 -- feet
noncomputable def pool_length := 150 -- feet
noncomputable def pool_depth := 10 -- feet
noncomputable def drainage_time := 2000 -- minutes
noncomputable def pool_volume := pool_width * pool_length * pool_depth -- cubic feet
noncomputable def removed_water_volume := hose_rate * drainage_time -- cubic feet

theorem pool_capacity_percentage :
  (removed_water_volume / pool_volume) * 100 = 100 :=
by
  -- the proof steps would go here
  sorry

end pool_capacity_percentage_l96_96263


namespace total_pencils_l96_96539

def num_boxes : ℕ := 12
def pencils_per_box : ℕ := 17

theorem total_pencils : num_boxes * pencils_per_box = 204 := by
  sorry

end total_pencils_l96_96539


namespace caitlinAgeIsCorrect_l96_96600

-- Define Aunt Anna's age
def auntAnnAge : Nat := 48

-- Define the difference between Aunt Anna's age and 18
def ageDifference : Nat := auntAnnAge - 18

-- Define Brianna's age as twice the difference
def briannaAge : Nat := 2 * ageDifference

-- Define Caitlin's age as 6 years younger than Brianna
def caitlinAge : Nat := briannaAge - 6

-- Theorem to prove Caitlin's age
theorem caitlinAgeIsCorrect : caitlinAge = 54 := by
  sorry -- Proof to be filled in

end caitlinAgeIsCorrect_l96_96600


namespace solution_z_sq_eq_neg_4_l96_96871

theorem solution_z_sq_eq_neg_4 (x y : ℝ) (i : ℂ) (z : ℂ) (h : z = x + y * i) (hi : i^2 = -1) : 
  z^2 = -4 ↔ z = 2 * i ∨ z = -2 * i := 
by
  sorry

end solution_z_sq_eq_neg_4_l96_96871


namespace sum_of_angles_l96_96626

theorem sum_of_angles (x y : ℝ) (n : ℕ) :
  n = 16 →
  (∃ k l : ℕ, k = 3 ∧ l = 5 ∧ 
  x = (k * (360 / n)) / 2 ∧ y = (l * (360 / n)) / 2) →
  x + y = 90 :=
by
  intros
  sorry

end sum_of_angles_l96_96626


namespace circle_radius_range_l96_96324

theorem circle_radius_range (r : ℝ) : 
  (∃ P₁ P₂ : ℝ × ℝ, (P₁.2 = 1 ∨ P₁.2 = -1) ∧ (P₂.2 = 1 ∨ P₂.2 = -1) ∧ 
  (P₁.1 - 3) ^ 2 + (P₁.2 + 5) ^ 2 = r^2 ∧ (P₂.1 - 3) ^ 2 + (P₂.2 + 5) ^ 2 = r^2) → (4 < r ∧ r < 6) :=
by
  sorry

end circle_radius_range_l96_96324


namespace smaller_angle_at_10_oclock_l96_96530

def degreeMeasureSmallerAngleAt10 := 
  let totalDegrees := 360
  let numHours := 12
  let degreesPerHour := totalDegrees / numHours
  let hourHandPosition := 10
  let minuteHandPosition := 12
  let divisionsBetween := if hourHandPosition < minuteHandPosition then minuteHandPosition - hourHandPosition else hourHandPosition - minuteHandPosition
  degreesPerHour * divisionsBetween

theorem smaller_angle_at_10_oclock : degreeMeasureSmallerAngleAt10 = 60 :=
  by 
    let totalDegrees := 360
    let numHours := 12
    let degreesPerHour := totalDegrees / numHours
    have h1 : degreesPerHour = 30 := by norm_num
    let hourHandPosition := 10
    let minuteHandPosition := 12
    let divisionsBetween := minuteHandPosition - hourHandPosition
    have h2 : divisionsBetween = 2 := by norm_num
    show 30 * divisionsBetween = 60
    calc 
      30 * 2 = 60 := by norm_num

end smaller_angle_at_10_oclock_l96_96530


namespace verify_probabilities_l96_96527

/-- A bag contains 2 red balls, 3 black balls, and 4 white balls, all of the same size.
    A ball is drawn from the bag at a time, and once drawn, it is not replaced. -/
def total_balls := 9
def red_balls := 2
def black_balls := 3
def white_balls := 4

/-- Calculate the probability that the first ball is black and the second ball is white. -/
def prob_first_black_second_white :=
  (black_balls / total_balls) * (white_balls / (total_balls - 1))

/-- Calculate the probability that the number of draws does not exceed 3, 
    given that drawing a red ball means stopping. -/
def prob_draws_not_exceed_3 :=
  (red_balls / total_balls) +
  ((total_balls - red_balls) / total_balls) * (red_balls / (total_balls - 1)) +
  ((total_balls - red_balls - 1) / total_balls) *
  ((total_balls - red_balls) / (total_balls - 1)) *
  (red_balls / (total_balls - 2))

/-- Theorem that verifies the probabilities based on the given conditions. -/
theorem verify_probabilities :
  prob_first_black_second_white = 1 / 6 ∧
  prob_draws_not_exceed_3 = 7 / 12 :=
by
  sorry

end verify_probabilities_l96_96527


namespace chuck_team_leads_by_2_l96_96516

open Nat

noncomputable def chuck_team_score_first_quarter := 9 * 2 + 5 * 1
noncomputable def yellow_team_score_first_quarter := 7 * 2 + 4 * 3

noncomputable def chuck_team_score_second_quarter := 6 * 2 + 3 * 3
noncomputable def yellow_team_score_second_quarter := 5 * 2 + 2 * 3 + 3 * 1

noncomputable def chuck_team_score_third_quarter := 4 * 2 + 2 * 3 + 6 * 1
noncomputable def yellow_team_score_third_quarter := 6 * 2 + 2 * 3

noncomputable def chuck_team_score_fourth_quarter := 8 * 2 + 1 * 3
noncomputable def yellow_team_score_fourth_quarter := 4 * 2 + 3 * 3 + 2 * 1

noncomputable def chuck_team_technical_fouls := 3
noncomputable def yellow_team_technical_fouls := 2

noncomputable def total_chuck_team_score :=
  chuck_team_score_first_quarter + chuck_team_score_second_quarter + 
  chuck_team_score_third_quarter + chuck_team_score_fourth_quarter + 
  chuck_team_technical_fouls

noncomputable def total_yellow_team_score :=
  yellow_team_score_first_quarter + yellow_team_score_second_quarter + 
  yellow_team_score_third_quarter + yellow_team_score_fourth_quarter + 
  yellow_team_technical_fouls

noncomputable def chuck_team_lead :=
  total_chuck_team_score - total_yellow_team_score

theorem chuck_team_leads_by_2 :
  chuck_team_lead = 2 :=
by
  sorry

end chuck_team_leads_by_2_l96_96516


namespace total_value_of_assets_l96_96242

variable (value_expensive_stock : ℕ)
variable (shares_expensive_stock : ℕ)
variable (shares_other_stock : ℕ)
variable (value_other_stock : ℕ)

theorem total_value_of_assets
    (h1: value_expensive_stock = 78)
    (h2: shares_expensive_stock = 14)
    (h3: shares_other_stock = 26)
    (h4: value_other_stock = value_expensive_stock / 2) :
    shares_expensive_stock * value_expensive_stock + shares_other_stock * value_other_stock = 2106 := by
    sorry

end total_value_of_assets_l96_96242


namespace two_digit_number_reversed_l96_96714

theorem two_digit_number_reversed :
  ∃ (x y : ℕ), (10 * x + y = 73) ∧ (10 * x + y = 2 * (10 * y + x) - 1) ∧ (x < 10) ∧ (y < 10) := 
by
  sorry

end two_digit_number_reversed_l96_96714


namespace probability_same_color_of_two_12_sided_dice_l96_96938

-- Define the conditions
def sides := 12
def red_sides := 3
def blue_sides := 5
def green_sides := 3
def golden_sides := 1

-- Calculate the probabilities for each color being rolled
def pr_both_red := (red_sides / sides) ^ 2
def pr_both_blue := (blue_sides / sides) ^ 2
def pr_both_green := (green_sides / sides) ^ 2
def pr_both_golden := (golden_sides / sides) ^ 2

-- Total probability calculation
def total_probability_same_color := pr_both_red + pr_both_blue + pr_both_green + pr_both_golden

theorem probability_same_color_of_two_12_sided_dice :
  total_probability_same_color = 11 / 36 := by
  sorry

end probability_same_color_of_two_12_sided_dice_l96_96938


namespace missing_digit_l96_96760

theorem missing_digit (B : ℕ) (h : B < 10) : 
  (15 ∣ (200 + 10 * B)) ↔ B = 1 ∨ B = 4 :=
by sorry

end missing_digit_l96_96760


namespace max_sides_of_polygon_in_1950_gon_l96_96823

theorem max_sides_of_polygon_in_1950_gon (n : ℕ) (h : n = 1950) :
  ∃ (m : ℕ), (m ≤ 1949) ∧ (∀ k, k > m → k ≤ 1949) :=
sorry

end max_sides_of_polygon_in_1950_gon_l96_96823


namespace hyperbola_through_C_l96_96069

noncomputable def equation_of_hyperbola_passing_through_C : Prop :=
  let A := (-1/2, 1/4)
  let B := (2, 4)
  let C := (-1/2, 4)
  ∃ (k : ℝ), k = -2 ∧ (∀ x : ℝ, x ≠ 0 → x * (4) = k)

theorem hyperbola_through_C :
  equation_of_hyperbola_passing_through_C :=
by
  sorry

end hyperbola_through_C_l96_96069


namespace students_remaining_after_fourth_stop_l96_96967

variable (n : ℕ)
variable (frac : ℚ)

def initial_students := (64 : ℚ)
def fraction_remaining := (2/3 : ℚ)

theorem students_remaining_after_fourth_stop : 
  let after_first_stop := initial_students * fraction_remaining
  let after_second_stop := after_first_stop * fraction_remaining
  let after_third_stop := after_second_stop * fraction_remaining
  let after_fourth_stop := after_third_stop * fraction_remaining
  after_fourth_stop = (1024 / 81) := 
by 
  sorry

end students_remaining_after_fourth_stop_l96_96967


namespace regina_final_earnings_l96_96186

-- Define the number of animals Regina has
def cows := 20
def pigs := 4 * cows
def goats := pigs / 2
def chickens := 2 * cows
def rabbits := 30

-- Define sale prices for each animal
def cow_price := 800
def pig_price := 400
def goat_price := 600
def chicken_price := 50
def rabbit_price := 25

-- Define annual earnings from animal products
def cow_milk_income := 500
def rabbit_meat_income := 10

-- Define annual farm maintenance and animal feed costs
def maintenance_cost := 10000

-- Define a calculation for the final earnings
def final_earnings : ℕ :=
  let cow_income := cows * cow_price
  let pig_income := pigs * pig_price
  let goat_income := goats * goat_price
  let chicken_income := chickens * chicken_price
  let rabbit_income := rabbits * rabbit_price
  let total_animal_sale_income := cow_income + pig_income + goat_income + chicken_income + rabbit_income

  let cow_milk_earning := cows * cow_milk_income
  let rabbit_meat_earning := rabbits * rabbit_meat_income
  let total_annual_income := cow_milk_earning + rabbit_meat_earning

  let total_income := total_animal_sale_income + total_annual_income
  let final_income := total_income - maintenance_cost

  final_income

-- Prove that the final earnings is as calculated
theorem regina_final_earnings : final_earnings = 75050 := by
  sorry

end regina_final_earnings_l96_96186


namespace reciprocal_of_abs_neg_two_l96_96872

theorem reciprocal_of_abs_neg_two : 1 / |(-2: ℤ)| = (1 / 2: ℚ) := by
  sorry

end reciprocal_of_abs_neg_two_l96_96872


namespace ratio_ab_l96_96838

theorem ratio_ab (a b : ℚ) (h : b / a = 5 / 13) : (a - b) / (a + b) = 4 / 9 :=
by
  sorry

end ratio_ab_l96_96838


namespace measured_diagonal_length_l96_96916

theorem measured_diagonal_length (a b c d diag : Real)
  (h1 : a = 1) (h2 : b = 2) (h3 : c = 2.8) (h4 : d = 5) (hd : diag = 7.5) :
  diag = 2.8 :=
sorry

end measured_diagonal_length_l96_96916


namespace percent_nurses_with_neither_l96_96749

-- Define the number of nurses in each category
def total_nurses : ℕ := 150
def nurses_with_hbp : ℕ := 90
def nurses_with_ht : ℕ := 50
def nurses_with_both : ℕ := 30

-- Define a predicate that checks the conditions of the problem
theorem percent_nurses_with_neither :
  ((total_nurses - (nurses_with_hbp + nurses_with_ht - nurses_with_both)) * 100 : ℚ) / total_nurses = 2667 / 100 :=
by sorry

end percent_nurses_with_neither_l96_96749


namespace part_I_solution_part_II_solution_l96_96096

def f (x a m : ℝ) : ℝ := |x - a| + m * |x + a|

theorem part_I_solution (x : ℝ) :
  (|x + 1| - |x - 1| >= x) ↔ (x <= -2 ∨ (0 <= x ∧ x <= 2)) :=
by
  sorry

theorem part_II_solution (m : ℝ) :
  (∀ (x a : ℝ), (0 < m ∧ m < 1 ∧ (a <= -3 ∨ 3 <= a)) → (f x a m >= 2)) ↔ (m = 1/3) :=
by
  sorry

end part_I_solution_part_II_solution_l96_96096


namespace reciprocal_of_neg_2023_l96_96657

theorem reciprocal_of_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by sorry

end reciprocal_of_neg_2023_l96_96657


namespace least_number_to_subtract_l96_96649

theorem least_number_to_subtract (x : ℕ) :
  (2590 - x) % 9 = 6 ∧ 
  (2590 - x) % 11 = 6 ∧ 
  (2590 - x) % 13 = 6 ↔ 
  x = 16 := 
sorry

end least_number_to_subtract_l96_96649


namespace geometric_sequence_b_l96_96892

theorem geometric_sequence_b (a b c : Real) (h1 : a = 5 + 2 * Real.sqrt 6) (h2 : c = 5 - 2 * Real.sqrt 6) (h3 : ∃ r, b = r * a ∧ c = r * b) :
  b = 1 ∨ b = -1 :=
by
  sorry

end geometric_sequence_b_l96_96892


namespace solve_for_question_mark_l96_96510

theorem solve_for_question_mark :
  let question_mark := 4135 / 45
  (45 * question_mark) + (625 / 25) - (300 * 4) = 2950 + (1500 / (75 * 2)) :=
by
  let question_mark := 4135 / 45
  sorry

end solve_for_question_mark_l96_96510


namespace solve_prime_equation_l96_96015

theorem solve_prime_equation (p q r : ℕ) (hp : p.Prime) (hq : q.Prime) (hr : r.Prime) 
(h_eq : p^3 - q^3 = 5 * r) : p = 7 ∧ q = 2 ∧ r = 67 := 
sorry

end solve_prime_equation_l96_96015


namespace tangent_periodic_solution_l96_96884

theorem tangent_periodic_solution :
  ∃ n : ℤ, -180 < n ∧ n < 180 ∧ (Real.tan (n * Real.pi / 180) = Real.tan (345 * Real.pi / 180)) := by
  sorry

end tangent_periodic_solution_l96_96884


namespace molecular_weight_1_mole_l96_96933

-- Define the molecular weight of 3 moles
def molecular_weight_3_moles : ℕ := 222

-- Prove that the molecular weight of 1 mole is 74 given the molecular weight of 3 moles
theorem molecular_weight_1_mole (mw3 : ℕ) (h : mw3 = 222) : mw3 / 3 = 74 :=
by
  sorry

end molecular_weight_1_mole_l96_96933


namespace tank_capacity_l96_96353

theorem tank_capacity (w c : ℕ) (h1 : w = c / 3) (h2 : w + 7 = 2 * c / 5) : c = 105 :=
sorry

end tank_capacity_l96_96353


namespace find_seventh_term_l96_96027

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}

-- Define arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define sum of the first n terms of the sequence
def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * a 0) + (d * (n * (n - 1)) / 2)

-- Now state the theorem
theorem find_seventh_term
  (h_arith_seq : arithmetic_sequence a d)
  (h_nonzero_d : d ≠ 0)
  (h_sum_five : S 5 = 5)
  (h_squares_eq : a 0 ^ 2 + a 1 ^ 2 = a 2 ^ 2 + a 3 ^ 2) :
  a 6 = 9 :=
sorry

end find_seventh_term_l96_96027


namespace geometric_reasoning_l96_96318

-- Definitions of relationships between geometric objects
inductive GeometricObject
  | Line
  | Plane

open GeometricObject

def perpendicular (a b : GeometricObject) : Prop := 
  match a, b with
  | Plane, Plane => True  -- Planes can be perpendicular
  | Line, Plane => True   -- Lines can be perpendicular to planes
  | Plane, Line => True   -- Planes can be perpendicular to lines
  | Line, Line => True    -- Lines can be perpendicular to lines (though normally in a 3D space specific context)

def parallel (a b : GeometricObject) : Prop := 
  match a, b with
  | Plane, Plane => True  -- Planes can be parallel
  | Line, Plane => True   -- Lines can be parallel to planes under certain interpretation
  | Plane, Line => True
  | Line, Line => True    -- Lines can be parallel

axiom x : GeometricObject
axiom y : GeometricObject
axiom z : GeometricObject

-- Main theorem statement
theorem geometric_reasoning (hx : perpendicular x y) (hy : parallel y z) 
  : ¬ (perpendicular x z) → (x = Plane ∧ y = Plane ∧ z = Line) :=
  sorry

end geometric_reasoning_l96_96318


namespace rational_point_partition_exists_l96_96876

open Set

-- Define rational numbers
noncomputable def Q : Set ℚ :=
  {x | True}

-- Define the set of rational points in the plane
def I : Set (ℚ × ℚ) := 
  {p | p.1 ∈ Q ∧ p.2 ∈ Q}

-- Statement of the theorem
theorem rational_point_partition_exists :
  ∃ (A B : Set (ℚ × ℚ)),
    (∀ (y : ℚ), {p ∈ A | p.1 = y}.Finite) ∧
    (∀ (x : ℚ), {p ∈ B | p.2 = x}.Finite) ∧
    (A ∪ B = I) ∧
    (A ∩ B = ∅) :=
sorry

end rational_point_partition_exists_l96_96876


namespace max_ratio_of_odd_integers_is_nine_l96_96377

-- Define odd positive integers x and y whose mean is 55
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_positive (n : ℕ) : Prop := 0 < n
def mean_is_55 (x y : ℕ) : Prop := (x + y) / 2 = 55

-- The problem statement
theorem max_ratio_of_odd_integers_is_nine (x y : ℕ) 
  (hx : is_positive x) (hy : is_positive y)
  (ox : is_odd x) (oy : is_odd y)
  (mean : mean_is_55 x y) : 
  ∀ r, r = (x / y : ℚ) → r ≤ 9 :=
by
  sorry

end max_ratio_of_odd_integers_is_nine_l96_96377


namespace carlton_outfit_count_l96_96902

-- Definitions of conditions
def sweater_vests (s : ℕ) : ℕ := 2 * s
def button_up_shirts : ℕ := 3
def outfits (v s : ℕ) : ℕ := v * s

-- Theorem statement
theorem carlton_outfit_count : outfits (sweater_vests button_up_shirts) button_up_shirts = 18 :=
by
  sorry

end carlton_outfit_count_l96_96902


namespace complement_M_l96_96785

section ComplementSet

variable (x : ℝ)

def M : Set ℝ := {x | 1 / x < 1}

theorem complement_M : {x | 0 ≤ x ∧ x ≤ 1} = Mᶜ := sorry

end ComplementSet

end complement_M_l96_96785


namespace Jonathan_typing_time_l96_96740

theorem Jonathan_typing_time
  (J : ℝ)
  (HJ : 0 < J)
  (rate_Jonathan : ℝ := 1 / J)
  (rate_Susan : ℝ := 1 / 30)
  (rate_Jack : ℝ := 1 / 24)
  (combined_rate : ℝ := 1 / 10)
  (combined_rate_eq : rate_Jonathan + rate_Susan + rate_Jack = combined_rate)
  : J = 40 :=
sorry

end Jonathan_typing_time_l96_96740


namespace frosting_sugar_calc_l96_96668

theorem frosting_sugar_calc (total_sugar cake_sugar : ℝ) (h1 : total_sugar = 0.8) (h2 : cake_sugar = 0.2) : 
  total_sugar - cake_sugar = 0.6 :=
by
  rw [h1, h2]
  sorry  -- Proof should go here

end frosting_sugar_calc_l96_96668


namespace paper_clips_distribution_l96_96132

theorem paper_clips_distribution (total_clips : ℕ) (num_boxes : ℕ) (clip_per_box : ℕ) 
  (h1 : total_clips = 81) (h2 : num_boxes = 9) : clip_per_box = 9 :=
by sorry

end paper_clips_distribution_l96_96132


namespace lower_limit_tip_percentage_l96_96800

namespace meal_tip

def meal_cost : ℝ := 35.50
def total_paid : ℝ := 40.825
def tip_limit : ℝ := 15

-- Define the lower limit tip percentage as the solution to the given conditions.
theorem lower_limit_tip_percentage :
  ∃ x : ℝ, x > 0 ∧ x < 25 ∧ (meal_cost + (x / 100) * meal_cost = total_paid) → 
  x = tip_limit :=
sorry

end meal_tip

end lower_limit_tip_percentage_l96_96800


namespace total_amount_l96_96395

theorem total_amount
  (x y z : ℝ)
  (hy : y = 0.45 * x)
  (hz : z = 0.50 * x)
  (y_share : y = 27) :
  x + y + z = 117 :=
by
  sorry

end total_amount_l96_96395


namespace acid_solution_replacement_percentage_l96_96201

theorem acid_solution_replacement_percentage 
  (original_concentration fraction_replaced final_concentration replaced_percentage : ℝ)
  (h₁ : original_concentration = 0.50)
  (h₂ : fraction_replaced = 0.5)
  (h₃ : final_concentration = 0.40)
  (h₄ : 0.25 + fraction_replaced * replaced_percentage = final_concentration) :
  replaced_percentage = 0.30 :=
by
  sorry

end acid_solution_replacement_percentage_l96_96201


namespace cos_tan_values_l96_96780

theorem cos_tan_values (α : ℝ) (h : Real.sin α = -1 / 2) :
  (∃ (quadrant : ℕ), 
    (quadrant = 3 ∧ Real.cos α = -Real.sqrt 3 / 2 ∧ Real.tan α = Real.sqrt 3 / 3) ∨ 
    (quadrant = 4 ∧ Real.cos α = Real.sqrt 3 / 2 ∧ Real.tan α = -Real.sqrt 3 / 3)) :=
sorry

end cos_tan_values_l96_96780


namespace log_identity_proof_l96_96035

theorem log_identity_proof (lg : ℝ → ℝ) (h1 : lg 50 = lg 2 + lg 25) (h2 : lg 25 = 2 * lg 5) :
  (lg 2)^2 + lg 2 * lg 50 + lg 25 = 2 :=
by sorry

end log_identity_proof_l96_96035


namespace actual_positions_correct_l96_96434

-- Define the five athletes
inductive Athlete
| A | B | C | D | E
deriving DecidableEq, Repr

open Athlete

-- Define the two predictions as lists
def first_prediction : List Athlete := [A, B, C, D, E]
def second_prediction : List Athlete := [C, E, A, B, D]

-- Define the actual positions
def actual_positions : List Athlete := [C, B, A, D, E]

-- Prove that the first prediction correctly predicted exactly three athletes
def first_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD first_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

-- Prove that the second prediction correctly predicted exactly two athletes
def second_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD second_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

theorem actual_positions_correct :
  first_prediction_correct = 3 ∧ second_prediction_correct = 2 ∧
  actual_positions = [C, B, A, D, E] :=
by
  -- Placeholder for actual proof
  sorry

end actual_positions_correct_l96_96434


namespace rate_of_decrease_l96_96733

theorem rate_of_decrease (x : ℝ) (h : 400 * (1 - x) ^ 2 = 361) : x = 0.05 :=
by {
  sorry -- The proof is omitted as requested.
}

end rate_of_decrease_l96_96733


namespace time_to_fill_cistern_l96_96585

def pipe_p_rate := (1: ℚ) / 10
def pipe_q_rate := (1: ℚ) / 15
def pipe_r_rate := - (1: ℚ) / 30
def combined_rate_p_q := pipe_p_rate + pipe_q_rate
def combined_rate_q_r := pipe_q_rate + pipe_r_rate
def initial_fill := 2 * combined_rate_p_q
def remaining_fill := 1 - initial_fill
def remaining_time := remaining_fill / combined_rate_q_r

theorem time_to_fill_cistern :
  remaining_time = 20 := by sorry

end time_to_fill_cistern_l96_96585


namespace max_value_in_interval_l96_96620

variable {R : Type*} [OrderedCommRing R]

variables (f : R → R)
variables (odd_f : ∀ x, f (-x) = -f (x))
variables (f_increasing : ∀ x y, 0 < x → x < y → f x < f y)
variables (additive_f : ∀ x y, f (x + y) = f x + f y)
variables (f1_eq_2 : f 1 = 2)

theorem max_value_in_interval : ∀ x ∈ Set.Icc (-3 : R) (-2 : R), f x ≤ f (-2) ∧ f (-2) = -4 :=
by
  sorry

end max_value_in_interval_l96_96620


namespace value_of_f1_plus_g3_l96_96482

def f (x : ℝ) := 3 * x - 4
def g (x : ℝ) := x + 2

theorem value_of_f1_plus_g3 : f (1 + g 3) = 14 := by
  sorry

end value_of_f1_plus_g3_l96_96482


namespace compute_f_at_5_l96_96706

def f : ℝ → ℝ := sorry

axiom f_property : ∀ x : ℝ, f (10 ^ x) = x

theorem compute_f_at_5 : f 5 = Real.log 5 / Real.log 10 :=
by
  sorry

end compute_f_at_5_l96_96706


namespace C_increases_with_n_l96_96274

noncomputable def C (e n R r : ℝ) : ℝ := (e * n) / (R + n * r)

theorem C_increases_with_n (e R r : ℝ) (h_e : 0 < e) (h_R : 0 < R) (h_r : 0 < r) :
  ∀ {n₁ n₂ : ℝ}, 0 < n₁ → n₁ < n₂ → C e n₁ R r < C e n₂ R r :=
by
  sorry

end C_increases_with_n_l96_96274


namespace initial_money_given_l96_96690

def bracelet_cost : ℕ := 15
def necklace_cost : ℕ := 10
def mug_cost : ℕ := 20
def num_bracelets : ℕ := 3
def num_necklaces : ℕ := 2
def num_mugs : ℕ := 1
def change_received : ℕ := 15

theorem initial_money_given : num_bracelets * bracelet_cost + num_necklaces * necklace_cost + num_mugs * mug_cost + change_received = 100 := 
sorry

end initial_money_given_l96_96690


namespace smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5_l96_96084

theorem smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5 :
  ∃ n : ℕ, n > 0 ∧ (7^n % 5 = n^7 % 5) ∧
  ∀ m : ℕ, m > 0 ∧ (7^m % 5 = m^7 % 5) → n ≤ m :=
by
  sorry

end smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5_l96_96084


namespace average_price_correct_l96_96423

-- Define the conditions
def books_shop1 : ℕ := 65
def price_shop1 : ℕ := 1480
def books_shop2 : ℕ := 55
def price_shop2 : ℕ := 920

-- Define the total books and total price based on conditions
def total_books : ℕ := books_shop1 + books_shop2
def total_price : ℕ := price_shop1 + price_shop2

-- Define the average price based on total books and total price
def average_price : ℕ := total_price / total_books

-- Theorem stating the average price per book Sandy paid
theorem average_price_correct : average_price = 20 :=
  by
  sorry

end average_price_correct_l96_96423


namespace expression_evaluation_l96_96688

theorem expression_evaluation : 
  76 + (144 / 12) + (15 * 19)^2 - 350 - (270 / 6) = 80918 :=
by
  sorry

end expression_evaluation_l96_96688


namespace journey_ratio_proof_l96_96474

def journey_ratio (x y : ℝ) : Prop :=
  (x + y = 448) ∧ (x / 21 + y / 24 = 20) → (x / y = 1)

theorem journey_ratio_proof : ∃ x y : ℝ, journey_ratio x y :=
by
  sorry

end journey_ratio_proof_l96_96474


namespace longest_diagonal_length_l96_96840

-- Defining conditions
variable (d1 d2 : ℝ)
variable (x : ℝ)
variable (area : ℝ)
variable (h_area : area = 144)
variable (h_ratio : d1 = 4 * x)
variable (h_ratio' : d2 = 3 * x)
variable (h_area_eq : area = 1 / 2 * d1 * d2)

-- The Lean statement, asserting the length of the longest diagonal is 8 * sqrt(6)
theorem longest_diagonal_length (x : ℝ) (h_area : 1 / 2 * (4 * x) * (3 * x) = 144) :
  4 * x = 8 * Real.sqrt 6 := by
sorry

end longest_diagonal_length_l96_96840


namespace compare_abc_l96_96875

noncomputable def a : ℝ := Real.log 5 / Real.log 2
noncomputable def b : ℝ := Real.log 6 / Real.log 2
noncomputable def c : ℝ := 9 ^ (1 / 2 : ℝ)

theorem compare_abc : c > b ∧ b > a := 
by
  sorry

end compare_abc_l96_96875


namespace person_B_catches_up_after_meeting_point_on_return_l96_96679
noncomputable def distance_A := 46
noncomputable def speed_A := 15
noncomputable def speed_B := 40
noncomputable def initial_gap_time := 1

-- Prove that Person B catches up to Person A after 3/5 hours.
theorem person_B_catches_up_after : 
  ∃ x : ℚ, 40 * x = 15 * (x + 1) ∧ x = 3 / 5 := 
by
  sorry

-- Prove that they meet 10 kilometers away from point B on the return journey.
theorem meeting_point_on_return : 
  ∃ y : ℚ, (46 - y) / 15 - (46 + y) / 40 = 1 ∧ y = 10 := 
by 
  sorry

end person_B_catches_up_after_meeting_point_on_return_l96_96679


namespace simplify_abs_expr_l96_96927

theorem simplify_abs_expr : |(-4 ^ 2 + 6)| = 10 := by
  sorry

end simplify_abs_expr_l96_96927


namespace basketball_court_width_l96_96087

variable (width length : ℕ)

-- Given conditions
axiom h1 : length = width + 14
axiom h2 : 2 * length + 2 * width = 96

-- Prove the width is 17 meters
theorem basketball_court_width : width = 17 :=
by {
  sorry
}

end basketball_court_width_l96_96087


namespace smallest_n_for_purple_l96_96878

-- The conditions as definitions
def red := 18
def green := 20
def blue := 22
def purple_cost := 24

-- The mathematical proof problem statement
theorem smallest_n_for_purple : 
  ∃ n : ℕ, purple_cost * n = Nat.lcm (Nat.lcm red green) blue ∧
            ∀ m : ℕ, (purple_cost * m = Nat.lcm (Nat.lcm red green) blue → m ≥ n) ↔ n = 83 := 
by
  sorry

end smallest_n_for_purple_l96_96878


namespace initial_contestants_proof_l96_96759

noncomputable def initial_contestants (final_round : ℕ) : ℕ :=
  let fraction_remaining := 2 / 5
  let fraction_advancing := 1 / 2
  let fraction_final := fraction_remaining * fraction_advancing
  (final_round : ℕ) / fraction_final

theorem initial_contestants_proof : initial_contestants 30 = 150 :=
sorry

end initial_contestants_proof_l96_96759


namespace beads_initial_state_repeats_l96_96931

-- Define the setup of beads on a circular wire
structure BeadConfig (n : ℕ) :=
(beads : Fin n → ℝ)  -- Each bead's position indexed by a finite set, ℝ denotes angular position

-- Define the instantaneous collision swapping function
def swap (n : ℕ) (i j : Fin n) (config : BeadConfig n) : BeadConfig n :=
⟨fun k => if k = i then config.beads j else if k = j then config.beads i else config.beads k⟩

-- Define what it means for a configuration to return to its initial state
def returns_to_initial (n : ℕ) (initial : BeadConfig n) (t : ℝ) : Prop :=
  ∃ (config : BeadConfig n), (∀ k, config.beads k = initial.beads k) ∧ (config = initial)

-- Specification of the problem
theorem beads_initial_state_repeats (n : ℕ) (initial : BeadConfig n) (ω : Fin n → ℝ) :
  (∀ k, ω k > 0) →  -- condition that all beads have positive angular speed, either clockwise or counterclockwise
  ∃ t : ℝ, t > 0 ∧ returns_to_initial n initial t := 
by
  sorry

end beads_initial_state_repeats_l96_96931


namespace intersection_of_M_and_N_l96_96787

namespace ProofProblem

def M := { x : ℝ | x^2 < 4 }
def N := { x : ℝ | x < 1 }

theorem intersection_of_M_and_N :
  M ∩ N = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end ProofProblem

end intersection_of_M_and_N_l96_96787


namespace minimum_x_plus_y_l96_96852

variable (x y : ℝ)
variable (h1 : 0 < x)
variable (h2 : 0 < y)
variable (h3 : (1 / (2 * x + y)) + (4 / (2 * x + 3 * y)) = 1)

theorem minimum_x_plus_y (hx : 0 < x) (hy : 0 < y) (h : (1 / (2 * x + y)) + (4 / (2 * x + 3 * y)) = 1) : x + y ≥ 9 / 4 :=
sorry

end minimum_x_plus_y_l96_96852


namespace mowing_lawn_time_l96_96701

theorem mowing_lawn_time (mary_time tom_time tom_solo_work : ℝ) 
  (mary_rate tom_rate : ℝ)
  (combined_rate remaining_lawn total_time : ℝ) :
  mary_time = 3 → 
  tom_time = 6 → 
  tom_solo_work = 3 → 
  mary_rate = 1 / mary_time → 
  tom_rate = 1 / tom_time → 
  combined_rate = mary_rate + tom_rate →
  remaining_lawn = 1 - (tom_solo_work * tom_rate) →
  total_time = tom_solo_work + (remaining_lawn / combined_rate) →
  total_time = 4 :=
by sorry

end mowing_lawn_time_l96_96701


namespace gift_wrapping_combinations_l96_96004

theorem gift_wrapping_combinations :
  (10 * 4 * 5 * 2 = 400) := by
  sorry

end gift_wrapping_combinations_l96_96004


namespace factorization_correct_l96_96070

theorem factorization_correct (x : ℝ) :
  (x - 3) * (x - 1) * (x - 2) * (x + 4) + 24 = (x - 2) * (x + 3) * (x^2 + x - 8) := 
sorry

end factorization_correct_l96_96070


namespace tangent_line_at_P_range_of_a_l96_96861

-- Define the function f(x)
noncomputable def f (x a : ℝ) : ℝ := a * (x - 1/x) - Real.log x

-- Problem (Ⅰ): Tangent line equation at P(1, f(1)) for a = 1
theorem tangent_line_at_P (x : ℝ) (h : x = 1) : (∃ y : ℝ, f x 1 = y ∧ x - y - 1 = 0) := sorry

-- Problem (Ⅱ): Range of a for f(x) ≥ 0 ∀ x ≥ 1
theorem range_of_a {a : ℝ} (h : ∀ x : ℝ, x ≥ 1 → f x a ≥ 0) : a ≥ 1/2 := sorry

end tangent_line_at_P_range_of_a_l96_96861


namespace allocate_plots_l96_96994

theorem allocate_plots (x y : ℕ) (h : x > y) : 
  ∃ u v : ℕ, (u^2 + v^2 = 2 * (x^2 + y^2)) :=
by
  sorry

end allocate_plots_l96_96994


namespace no_solutions_for_specific_a_l96_96499

theorem no_solutions_for_specific_a (a : ℝ) :
  (a < -9) ∨ (a > 0) →
  ¬ ∃ x : ℝ, 5 * |x - 4 * a| + |x - a^2| + 4 * x - 3 * a = 0 :=
by sorry

end no_solutions_for_specific_a_l96_96499


namespace prob_black_yellow_l96_96210

theorem prob_black_yellow:
  ∃ (x y : ℚ), 12 > 0 ∧
  (∃ (r b y' : ℚ), r = 1/3 ∧ b - y' = 1/6 ∧ b + y' = 2/3 ∧ r + b + y' = 1) ∧
  x = 5/12 ∧ y = 1/4 :=
by
  sorry

end prob_black_yellow_l96_96210


namespace sum_powers_of_i_l96_96611

-- Define the conditions
def i : ℂ := Complex.I -- Complex.I is the imaginary unit in ℂ (ℂ is the set of complex numbers)

-- The theorem statement
theorem sum_powers_of_i : (i + i^2 + i^3 + i^4) * 150 + 1 + i + i^2 + i^3 = 0 := by
  sorry

end sum_powers_of_i_l96_96611


namespace b_is_multiple_of_5_a_plus_b_is_multiple_of_5_l96_96858

variable (a b : ℕ)

-- Conditions
def is_multiple_of_5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k
def is_multiple_of_10 (n : ℕ) : Prop := ∃ k : ℕ, n = 10 * k

-- Given conditions in the problem
axiom h_a : is_multiple_of_5 a
axiom h_b : is_multiple_of_10 b

-- Statements to be proved
theorem b_is_multiple_of_5 : is_multiple_of_5 b :=
sorry

theorem a_plus_b_is_multiple_of_5 : is_multiple_of_5 (a + b) :=
sorry

end b_is_multiple_of_5_a_plus_b_is_multiple_of_5_l96_96858


namespace cost_per_order_of_pakoras_l96_96137

noncomputable def samosa_cost : ℕ := 2
noncomputable def samosa_count : ℕ := 3
noncomputable def mango_lassi_cost : ℕ := 2
noncomputable def pakora_count : ℕ := 4
noncomputable def tip_percentage : ℚ := 0.25
noncomputable def total_cost_with_tax : ℚ := 25

theorem cost_per_order_of_pakoras (P : ℚ)
  (h1 : samosa_cost * samosa_count = 6)
  (h2 : mango_lassi_cost = 2)
  (h3 : 1.25 * (samosa_cost * samosa_count + mango_lassi_cost + pakora_count * P) = total_cost_with_tax) :
  P = 3 :=
by
  -- sorry ⟹ sorry
  sorry

end cost_per_order_of_pakoras_l96_96137


namespace triangle_area_ellipse_l96_96122

open Real

noncomputable def ellipse_foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := sqrt (a^2 + b^2)
  ((-c, 0), (c, 0))

theorem triangle_area_ellipse 
  (a : ℝ) (b : ℝ) 
  (h1 : a = sqrt 2) (h2 : b = 1) 
  (F1 F2 : ℝ × ℝ) 
  (hfoci : ellipse_foci a b = (F1, F2))
  (hF2 : F2 = (sqrt 3, 0))
  (A B : ℝ × ℝ)
  (hA : A = (0, -1))
  (hB : B = (0, -1))
  (h_inclination : ∃ θ, θ = pi / 4 ∧ (B.1 - A.1) / (B.2 - A.2) = tan θ) :
  F1 = (-sqrt 3, 0) → 
  1/2 * (B.1 - A.1) * (B.2 - A.2) = 4/3 :=
sorry

end triangle_area_ellipse_l96_96122


namespace lcm_of_18_and_24_l96_96337

noncomputable def lcm_18_24 : ℕ :=
  Nat.lcm 18 24

theorem lcm_of_18_and_24 : lcm_18_24 = 72 :=
by
  sorry

end lcm_of_18_and_24_l96_96337


namespace right_triangle_legs_l96_96632

theorem right_triangle_legs (a b : ℕ) (hypotenuse : ℕ) (h : hypotenuse = 39) : a^2 + b^2 = 39^2 → (a = 15 ∧ b = 36) ∨ (a = 36 ∧ b = 15) :=
by
  sorry

end right_triangle_legs_l96_96632


namespace minimum_cuts_for_11_sided_polygons_l96_96036

theorem minimum_cuts_for_11_sided_polygons (k : ℕ) :
  (∀ k, (11 * 252 + 3 * (k + 1 - 252) ≤ 4 * k + 4)) ∧ (252 ≤ (k + 1)) ∧ (4 * k + 4 ≥ 11 * 252 + 3 * (k + 1 - 252))
  ∧ (11 * 252 + 3 * (k + 1 - 252) ≤ 4 * k + 4) → (k ≥ 2012) ∧ (k = 2015) := 
sorry

end minimum_cuts_for_11_sided_polygons_l96_96036


namespace highest_degree_divisibility_l96_96730

-- Definition of the problem settings
def prime_number := 1991
def number_1 := 1990 ^ (1991 ^ 1002)
def number_2 := 1992 ^ (1501 ^ 1901)
def combined_number := number_1 + number_2

-- Statement of the proof to be formalized
theorem highest_degree_divisibility (k : ℕ) : k = 1001 ∧ prime_number ^ k ∣ combined_number := by
  sorry

end highest_degree_divisibility_l96_96730


namespace distance_between_home_and_school_l96_96304

variable (D T : ℝ)

def boy_travel_5kmhr : Prop :=
  5 * (T + 7 / 60) = D

def boy_travel_10kmhr : Prop :=
  10 * (T - 8 / 60) = D

theorem distance_between_home_and_school :
  (boy_travel_5kmhr D T) ∧ (boy_travel_10kmhr D T) → D = 2.5 :=
by
  intro h
  sorry

end distance_between_home_and_school_l96_96304


namespace jasper_hot_dogs_fewer_l96_96384

theorem jasper_hot_dogs_fewer (chips drinks hot_dogs : ℕ)
  (h1 : chips = 27)
  (h2 : drinks = 31)
  (h3 : drinks = hot_dogs + 12) : 27 - hot_dogs = 8 := by
  sorry

end jasper_hot_dogs_fewer_l96_96384


namespace difference_of_squares_divisible_by_9_l96_96031

theorem difference_of_squares_divisible_by_9 (a b : ℤ) : ∃ k : ℤ, (3 * a + 2)^2 - (3 * b + 2)^2 = 9 * k := by
  sorry

end difference_of_squares_divisible_by_9_l96_96031


namespace slices_leftover_l96_96503

def total_slices (small_pizzas large_pizzas : ℕ) : ℕ :=
  (3 * 4) + (2 * 8)

def slices_eaten_by_people (george bob susie bill fred mark : ℕ) : ℕ :=
  george + bob + susie + bill + fred + mark

theorem slices_leftover :
  total_slices 3 2 - slices_eaten_by_people 3 4 2 3 3 3 = 10 :=
by sorry

end slices_leftover_l96_96503


namespace hiker_speed_third_day_l96_96662

-- Define the conditions
def first_day_distance : ℕ := 18
def first_day_speed : ℕ := 3
def second_day_distance : ℕ :=
  let first_day_hours := first_day_distance / first_day_speed
  let second_day_hours := first_day_hours - 1
  let second_day_speed := first_day_speed + 1
  second_day_hours * second_day_speed
def total_distance : ℕ := 53
def third_day_hours : ℕ := 3

-- Define the speed on the third day based on given conditions
def speed_on_third_day : ℕ :=
  let third_day_distance := total_distance - first_day_distance - second_day_distance
  third_day_distance / third_day_hours

-- The theorem we need to prove
theorem hiker_speed_third_day : speed_on_third_day = 5 := by
  sorry

end hiker_speed_third_day_l96_96662


namespace percentage_increase_B_over_C_l96_96460

noncomputable def A_m : ℕ := 537600 / 12
noncomputable def C_m : ℕ := 16000
noncomputable def ratio : ℚ := 5 / 2

noncomputable def B_m (A_m : ℕ) : ℚ := (2 * A_m) / 5

theorem percentage_increase_B_over_C :
  B_m A_m = 17920 →
  C_m = 16000 →
  (B_m A_m - C_m) / C_m * 100 = 12 :=
by
  sorry

end percentage_increase_B_over_C_l96_96460


namespace Mitch_needs_to_keep_500_for_license_and_registration_l96_96663

-- Define the constants and variables
def total_savings : ℕ := 20000
def cost_per_foot : ℕ := 1500
def longest_boat_length : ℕ := 12
def docking_fee_factor : ℕ := 3

-- Define the price of the longest boat
def cost_longest_boat : ℕ := longest_boat_length * cost_per_foot

-- Define the amount for license and registration
def license_and_registration (L : ℕ) : Prop :=
  total_savings - cost_longest_boat = L * (docking_fee_factor + 1)

-- The statement to be proved
theorem Mitch_needs_to_keep_500_for_license_and_registration :
  ∃ L : ℕ, license_and_registration L ∧ L = 500 :=
by
  -- Conditions and setup have already been defined, we now state the proof goal.
  sorry

end Mitch_needs_to_keep_500_for_license_and_registration_l96_96663


namespace inequality_holds_l96_96747

theorem inequality_holds (x y : ℝ) : (y - x^2 < abs x) ↔ (y < x^2 + abs x) := by
  sorry

end inequality_holds_l96_96747


namespace tan_double_angle_solution_l96_96983

theorem tan_double_angle_solution (x : ℝ) (h : Real.tan (x + Real.pi / 4) = 2) :
  (Real.tan x) / (Real.tan (2 * x)) = 4 / 9 :=
sorry

end tan_double_angle_solution_l96_96983


namespace consecutive_numbers_product_l96_96534

theorem consecutive_numbers_product (a b c d : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c + 1 = d) (h4 : a + d = 109) :
  b * c = 2970 :=
by {
  -- Proof goes here
  sorry
}

end consecutive_numbers_product_l96_96534


namespace min_value_of_sum_l96_96535

theorem min_value_of_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 2 * a + b) : a + b ≥ 2 * Real.sqrt 2 + 3 :=
sorry

end min_value_of_sum_l96_96535


namespace arithmetic_fraction_subtraction_l96_96443

theorem arithmetic_fraction_subtraction :
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 :=
by
  sorry

end arithmetic_fraction_subtraction_l96_96443


namespace find_f2_l96_96232

-- Define the function f and the condition it satisfies
variable (f : ℝ → ℝ)

-- Define the condition given in the problem
def condition : Prop := ∀ x, x ≠ 1 / 3 → f x + f ((x + 1) / (1 - 3 * x)) = x

-- State the theorem to prove the value of f(2)
theorem find_f2 (h : condition f) : f 2 = 48 / 35 := 
by
  sorry

end find_f2_l96_96232


namespace fifth_graders_more_than_eighth_graders_l96_96485

theorem fifth_graders_more_than_eighth_graders 
  (cost : ℕ) 
  (h_cost : cost > 0) 
  (h_div_234 : 234 % cost = 0) 
  (h_div_312 : 312 % cost = 0) 
  (h_40_fifth_graders : 40 > 0) : 
  (312 / cost) - (234 / cost) = 6 := 
by 
  sorry

end fifth_graders_more_than_eighth_graders_l96_96485


namespace roses_cut_from_garden_l96_96093

-- Define the variables and conditions
variables {x : ℕ} -- x is the number of freshly cut roses

def initial_roses : ℕ := 17
def roses_thrown_away : ℕ := 8
def roses_final_vase : ℕ := 42
def roses_given_away : ℕ := 6

-- The condition that describes the total roses now
def condition (x : ℕ) : Prop :=
  initial_roses - roses_thrown_away + (1/3 : ℚ) * x = roses_final_vase

-- The verification step that checks the total roses concerning given away roses
def verification (x : ℕ) : Prop :=
  (1/3 : ℚ) * x + roses_given_away = roses_final_vase + roses_given_away

-- The main theorem to prove the number of roses cut
theorem roses_cut_from_garden (x : ℕ) (h1 : condition x) (h2 : verification x) : x = 99 :=
  sorry

end roses_cut_from_garden_l96_96093


namespace line_equation_l96_96435

theorem line_equation 
  (m b k : ℝ) 
  (h1 : ∀ k, abs ((k^2 + 4 * k + 4) - (m * k + b)) = 4)
  (h2 : m * 2 + b = 8) 
  (h3 : b ≠ 0) : 
  m = 8 ∧ b = -8 :=
by sorry

end line_equation_l96_96435


namespace samantha_more_posters_l96_96066

theorem samantha_more_posters :
  ∃ S : ℕ, S > 18 ∧ 18 + S = 51 ∧ S - 18 = 15 :=
by
  sorry

end samantha_more_posters_l96_96066


namespace perimeter_of_triangle_AF2B_l96_96721

theorem perimeter_of_triangle_AF2B (a : ℝ) (m n : ℝ) (F1 F2 A B : ℝ × ℝ) 
  (h_hyperbola : ∀ x y : ℝ, (x^2 - 4*y^2 = 4) ↔ (x^2 / 4 - y^2 = 1)) 
  (h_mn : m + n = 3) 
  (h_AF1 : dist A F1 = m) 
  (h_BF1 : dist B F1 = n) 
  (h_AF2 : dist A F2 = 4 + m) 
  (h_BF2 : dist B F2 = 4 + n) 
  : dist A F1 + dist A F2 + dist B F2 + dist B F1 = 14 :=
by
  sorry

end perimeter_of_triangle_AF2B_l96_96721


namespace find_min_length_seg_O1O2_l96_96455

noncomputable def minimum_length_O1O2 
  (X Y Z W : ℝ × ℝ) 
  (dist_XY : ℝ) (dist_YZ : ℝ) (dist_YW : ℝ)
  (O1 O2 : ℝ × ℝ) 
  (circumcenter1 : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ × ℝ)
  (circumcenter2 : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ × ℝ)
  (h1 : dist X Y = dist_XY) 
  (h2 : dist Y Z = dist_YZ) 
  (h3 : dist Y W = dist_YW) 
  (hO1 : O1 = circumcenter1 W X Y)
  (hO2 : O2 = circumcenter2 W Y Z)
  : ℝ :=
  dist O1 O2

theorem find_min_length_seg_O1O2 
  (X Y Z W : ℝ × ℝ) 
  (dist_XY : ℝ := 1)
  (dist_YZ : ℝ := 3)
  (dist_YW : ℝ := 5)
  (O1 O2 : ℝ × ℝ) 
  (circumcenter1 : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ × ℝ)
  (circumcenter2 : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ × ℝ)
  (h1 : dist X Y = dist_XY) 
  (h2 : dist Y Z = dist_YZ) 
  (h3 : dist Y W = dist_YW) 
  (hO1 : O1 = circumcenter1 W X Y)
  (hO2 : O2 = circumcenter2 W Y Z)
  : minimum_length_O1O2 X Y Z W dist_XY dist_YZ dist_YW O1 O2 circumcenter1 circumcenter2 h1 h2 h3 hO1 hO2 = 2 :=
sorry

end find_min_length_seg_O1O2_l96_96455


namespace jindra_initial_dice_count_l96_96490

-- Given conditions about the dice stacking
def number_of_dice_per_layer : ℕ := 36
def layers_stacked_completely : ℕ := 6
def dice_received : ℕ := 18

-- We need to prove that the initial number of dice Jindra had is 234
theorem jindra_initial_dice_count : 
    (layers_stacked_completely * number_of_dice_per_layer + dice_received) = 234 :=
    by 
        sorry

end jindra_initial_dice_count_l96_96490


namespace sequence_a4_value_l96_96107

theorem sequence_a4_value :
  ∃ (a : ℕ → ℕ), (a 1 = 1) ∧ (∀ n, a (n+1) = 2 * a n + 1) ∧ (a 4 = 15) :=
by
  sorry

end sequence_a4_value_l96_96107


namespace find_m_range_l96_96885

noncomputable def ellipse_symmetric_points_range (m : ℝ) : Prop :=
  -((2:ℝ) * Real.sqrt (13:ℝ) / 13) < m ∧ m < ((2:ℝ) * Real.sqrt (13:ℝ) / 13)

theorem find_m_range :
  ∃ m : ℝ, ellipse_symmetric_points_range m :=
sorry

end find_m_range_l96_96885


namespace range_of_x_l96_96014

-- Define the ceiling function for ease of use.
noncomputable def ceil (x : ℝ) : ℤ := ⌈x⌉

theorem range_of_x (x : ℝ) (h1 : ceil (2 * x + 1) = 5) (h2 : ceil (2 - 3 * x) = -3) :
  (5 / 3 : ℝ) ≤ x ∧ x < 2 :=
by
  sorry

end range_of_x_l96_96014


namespace largest_allowed_set_size_correct_l96_96447

noncomputable def largest_allowed_set_size (N : ℕ) : ℕ :=
  N - Nat.floor (N / 4)

def is_allowed (S : Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → (a ∣ b → b ∣ c → False)

theorem largest_allowed_set_size_correct (N : ℕ) (hN : 0 < N) : 
  ∃ S : Finset ℕ, is_allowed S ∧ S.card = largest_allowed_set_size N := sorry

end largest_allowed_set_size_correct_l96_96447


namespace percentage_decrease_in_price_l96_96887

theorem percentage_decrease_in_price (original_price new_price decrease percentage : ℝ) :
  original_price = 1300 → new_price = 988 →
  decrease = original_price - new_price →
  percentage = (decrease / original_price) * 100 →
  percentage = 24 := by
  sorry

end percentage_decrease_in_price_l96_96887


namespace lift_time_15_minutes_l96_96098

theorem lift_time_15_minutes (t : ℕ) (h₁ : 5 = 5) (h₂ : 6 * (t + 5) = 120) : t = 15 :=
by {
  sorry
}

end lift_time_15_minutes_l96_96098


namespace max_magnitude_z3_plus_3z_plus_2i_l96_96305

open Complex

theorem max_magnitude_z3_plus_3z_plus_2i (z : ℂ) (h : Complex.abs z = 1) :
  ∃ M, M = 3 * Real.sqrt 3 ∧ ∀ (z : ℂ), Complex.abs z = 1 → Complex.abs (z^3 + 3 * z + 2 * Complex.I) ≤ M :=
by
  sorry

end max_magnitude_z3_plus_3z_plus_2i_l96_96305


namespace calories_per_orange_is_correct_l96_96491

noncomputable def calories_per_orange
  (oranges pieces_per_orange num_people calories_per_person : ℕ)
  (h_oranges : oranges = 5)
  (h_pieces_per_orange : pieces_per_orange = 8)
  (h_num_people : num_people = 4)
  (h_calories_per_person : calories_per_person = 100) : ℕ :=
by
  -- Definitions derived from conditions
  let total_pieces := oranges * pieces_per_orange
  let pieces_per_person := total_pieces / num_people
  let total_calories := calories_per_person
  have calories_per_piece := total_calories / pieces_per_person

  -- Conclusion
  have calories_per_orange := pieces_per_orange * calories_per_piece
  exact calories_per_orange

theorem calories_per_orange_is_correct
  (oranges pieces_per_orange num_people calories_per_person : ℕ)
  (h_oranges : oranges = 5)
  (h_pieces_per_orange : pieces_per_orange = 8)
  (h_num_people : num_people = 4)
  (h_calories_per_person : calories_per_person = 100) :
  calories_per_orange oranges pieces_per_orange num_people calories_per_person
    h_oranges h_pieces_per_orange h_num_people h_calories_per_person = 100 :=
by
  simp [calories_per_orange]
  sorry  -- Proof omitted

end calories_per_orange_is_correct_l96_96491


namespace frog_jump_distance_l96_96471

variable (grasshopper_jump frog_jump mouse_jump : ℕ)
variable (H1 : grasshopper_jump = 19)
variable (H2 : grasshopper_jump = frog_jump + 4)
variable (H3 : mouse_jump = frog_jump - 44)

theorem frog_jump_distance : frog_jump = 15 := by
  sorry

end frog_jump_distance_l96_96471


namespace matrix_eq_value_satisfied_for_two_values_l96_96292

variable (a b c d x : ℝ)

def matrix_value (a b c d : ℝ) : ℝ := a * b - c * d

-- Define the specific instance for the given matrix problem
def matrix_eq_value (x : ℝ) : Prop :=
  matrix_value (2 * x) x 1 x = 3

-- Prove that the equation is satisfied for exactly two values of x
theorem matrix_eq_value_satisfied_for_two_values :
  (∃! (x : ℝ), matrix_value (2 * x) x 1 x = 3) :=
sorry

end matrix_eq_value_satisfied_for_two_values_l96_96292


namespace real_solutions_eq_l96_96371

theorem real_solutions_eq :
  ∀ x : ℝ, (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 12) → (x = 10 ∨ x = -1) :=
by
  sorry

end real_solutions_eq_l96_96371


namespace rikki_poetry_sales_l96_96865

theorem rikki_poetry_sales :
  let words_per_5min := 25
  let total_minutes := 2 * 60
  let intervals := total_minutes / 5
  let total_words := words_per_5min * intervals
  let total_earnings := 6
  let price_per_word := total_earnings / total_words
  price_per_word = 0.01 :=
by
  sorry

end rikki_poetry_sales_l96_96865


namespace arithmetic_seq_contains_geometric_seq_l96_96836

theorem arithmetic_seq_contains_geometric_seq (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (∃ (ns : ℕ → ℕ) (k : ℝ), k ≠ 1 ∧ (∀ n, a + b * (ns (n + 1)) = k * (a + b * (ns n)))) ↔ (∃ (q : ℚ), a = q * b) :=
sorry

end arithmetic_seq_contains_geometric_seq_l96_96836


namespace hyperbola_focus_l96_96757

theorem hyperbola_focus (m : ℝ) :
  (∃ (F : ℝ × ℝ), F = (0, 5) ∧ F ∈ {P : ℝ × ℝ | ∃ x y : ℝ, 
  x = P.1 ∧ y = P.2 ∧ (y^2 / m - x^2 / 9 = 1)}) → 
  m = 16 :=
by
  sorry

end hyperbola_focus_l96_96757


namespace consecutive_probability_is_two_fifths_l96_96357

-- Conditions
def total_days : ℕ := 5
def select_days : ℕ := 2

-- Total number of basic events (number of ways to choose 2 days out of 5)
def total_events : ℕ := Nat.choose total_days select_days -- This is C(5, 2)

-- Number of basic events where 2 selected days are consecutive
def consecutive_events : ℕ := 4

-- Probability that the selected 2 days are consecutive
def consecutive_probability : ℚ := consecutive_events / total_events

-- Theorem to be proved
theorem consecutive_probability_is_two_fifths :
  consecutive_probability = 2 / 5 :=
by
  sorry

end consecutive_probability_is_two_fifths_l96_96357


namespace range_of_a_l96_96617

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (hf_odd : ∀ x, f (-x) = -f x)
  (hf_expr_pos : ∀ x, x > 0 → f x = -x^2 + ax - 1 - a)
  (hf_monotone : ∀ x y, x < y → f y ≤ f x) :
  -1 ≤ a ∧ a ≤ 0 := 
sorry

end range_of_a_l96_96617


namespace quadratic_equation_roots_l96_96038

theorem quadratic_equation_roots (a b c : ℝ) (h_a_nonzero : a ≠ 0) 
  (h_roots : ∀ x, a * x^2 + b * x + c = 0 ↔ x = 1 ∨ x = -1) : 
  a + b + c = 0 ∧ b = 0 :=
by
  -- Using Vieta's formulas and the properties given, we should show:
  -- h_roots means the sum of roots = -(b/a) = 0 → b = 0
  -- and the product of roots = (c/a) = -1/a → c = -a
  -- Substituting these into ax^2 + bx + c = 0 should give us:
  -- a + b + c = 0 → we need to show both parts to complete the proof.
  sorry

end quadratic_equation_roots_l96_96038


namespace exists_root_in_interval_l96_96797

noncomputable def f (x : ℝ) := 3^x + 3 * x - 8

theorem exists_root_in_interval :
  f 1 < 0 → f 1.5 > 0 → f 1.25 < 0 → ∃ x ∈ (Set.Ioo 1.25 1.5), f x = 0 :=
by
  intros h1 h2 h3
  sorry

end exists_root_in_interval_l96_96797


namespace syllogism_example_l96_96279

-- Definitions based on the conditions
def is_even (n : ℕ) := n % 2 = 0
def is_divisible_by_2 (n : ℕ) := n % 2 = 0

-- Given conditions:
axiom even_implies_divisible_by_2 : ∀ n : ℕ, is_even n → is_divisible_by_2 n
axiom h2012_is_even : is_even 2012

-- Proving the conclusion and the syllogism pattern
theorem syllogism_example : is_divisible_by_2 2012 :=
by
  apply even_implies_divisible_by_2
  apply h2012_is_even

end syllogism_example_l96_96279


namespace inequality_solution_m_range_l96_96829

def f (x : ℝ) : ℝ := abs (x - 2)
def g (x : ℝ) (m : ℝ) : ℝ := -abs (x + 3) + m

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, a = 1 → f x + a - 1 > 0 ↔ x ≠ 2) ∧
  (a > 1 → ∀ x : ℝ, f x + a - 1 > 0 ↔ True) ∧
  (a < 1 → ∀ x : ℝ, f x + a - 1 > 0 ↔ x < a + 1 ∨ x > 3 - a) :=
by
  sorry

theorem m_range (m : ℝ) : (∀ x : ℝ, f x ≥ g x m) → m < 5 :=
by
  sorry

end inequality_solution_m_range_l96_96829


namespace sticks_per_pot_is_181_l96_96202

/-- Define the problem conditions -/
def number_of_pots : ℕ := 466
def flowers_per_pot : ℕ := 53
def total_flowers_and_sticks : ℕ := 109044

/-- Define the function to calculate the number of sticks per pot -/
def sticks_per_pot (S : ℕ) : Prop :=
  (number_of_pots * flowers_per_pot + number_of_pots * S = total_flowers_and_sticks)

/-- State the theorem -/
theorem sticks_per_pot_is_181 : sticks_per_pot 181 :=
by
  sorry

end sticks_per_pot_is_181_l96_96202


namespace D_72_l96_96901

def D (n : ℕ) : ℕ :=
  -- Definition of D(n) should be provided here
  sorry

theorem D_72 : D 72 = 121 :=
  sorry

end D_72_l96_96901


namespace Jill_talking_time_total_l96_96128

-- Definition of the sequence of talking times
def talking_time : ℕ → ℕ 
| 0 => 5
| (n+1) => 2 * talking_time n

-- The statement we need to prove
theorem Jill_talking_time_total :
  (talking_time 0) + (talking_time 1) + (talking_time 2) + (talking_time 3) + (talking_time 4) = 155 :=
by
  sorry

end Jill_talking_time_total_l96_96128


namespace solve_for_x_l96_96325

theorem solve_for_x (x : ℝ) (h₁ : (x + 2) ≠ 0) (h₂ : (|x| - 2) / (x + 2) = 0) : x = 2 := by
  sorry

end solve_for_x_l96_96325


namespace inequality_satisfaction_l96_96284

theorem inequality_satisfaction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x / y + 1 / x + y ≥ y / x + 1 / y + x) ↔ 
  ((x = y) ∨ (x = 1 ∧ y ≠ 0) ∨ (y = 1 ∧ x ≠ 0)) ∧ (x ≠ 0 ∧ y ≠ 0) :=
by
  sorry

end inequality_satisfaction_l96_96284


namespace find_x_l96_96699

theorem find_x (x : ℕ) (h : 5 * x + 4 * x + x + 2 * x = 360) : x = 30 :=
by
  sorry

end find_x_l96_96699


namespace valid_quadratic_polynomials_l96_96999

theorem valid_quadratic_polynomials (b c : ℤ)
  (h₁ : ∃ x₁ x₂ : ℤ, b = -(x₁ + x₂) ∧ c = x₁ * x₂)
  (h₂ : 1 + b + c = 10) :
  (b = -13 ∧ c = 22) ∨ (b = -9 ∧ c = 18) ∨ (b = 9 ∧ c = 0) ∨ (b = 5 ∧ c = 4) := sorry

end valid_quadratic_polynomials_l96_96999


namespace find_other_root_l96_96074

theorem find_other_root (m : ℝ) (h : 2^2 - 2 + m = 0) : 
  ∃ α : ℝ, α = -1 ∧ (α^2 - α + m = 0) :=
by
  -- Assuming x = 2 is a root, prove that the other root is -1.
  sorry

end find_other_root_l96_96074


namespace light_distance_200_years_l96_96765

-- Define the distance light travels in one year.
def distance_one_year := 5870000000000

-- Define the scientific notation representation for distance in one year
def distance_one_year_sci := 587 * 10^10

-- Define the distance light travels in 200 years.
def distance_200_years := distance_one_year * 200

-- Define the expected distance in scientific notation for 200 years.
def expected_distance := 1174 * 10^12

-- The theorem stating the given condition and the conclusion to prove
theorem light_distance_200_years : distance_200_years = expected_distance :=
by
  -- skipping the proof
  sorry

end light_distance_200_years_l96_96765


namespace gcd_18_30_l96_96450

theorem gcd_18_30 : Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l96_96450


namespace cos_90_eq_zero_l96_96108

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l96_96108


namespace exist_three_sum_eq_third_l96_96959

theorem exist_three_sum_eq_third
  (A : Finset ℕ)
  (h_card : A.card = 52)
  (h_cond : ∀ (a : ℕ), a ∈ A → a ≤ 100) :
  ∃ (x y z : ℕ), x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x + y = z :=
sorry

end exist_three_sum_eq_third_l96_96959


namespace regression_line_estimate_l96_96590

theorem regression_line_estimate:
  (∀ (x y : ℝ), y = 1.23 * x + a ↔ a = 5 - 1.23 * 4) →
  ∃ (y : ℝ), y = 1.23 * 2 + 0.08 :=
by
  intro h
  use 2.54
  simp
  sorry

end regression_line_estimate_l96_96590


namespace actual_price_per_gallon_l96_96418

variable (x : ℝ)
variable (expected_price : ℝ := x) -- price per gallon that the motorist expected to pay
variable (total_cash : ℝ := 12 * x) -- total cash to buy 12 gallons at expected price
variable (actual_price : ℝ := x + 0.30) -- actual price per gallon
variable (equation : 12 * x = 10 * (x + 0.30)) -- total cash equals the cost of 10 gallons at actual price

theorem actual_price_per_gallon (x : ℝ) (h : 12 * x = 10 * (x + 0.30)) : x + 0.30 = 1.80 := 
by 
  sorry

end actual_price_per_gallon_l96_96418


namespace age_relation_l96_96599

variable (x y z : ℕ)

theorem age_relation (h1 : x > y) : (z > y) ↔ (∃ w, w > 0 ∧ y + z > 2 * x) :=
sorry

end age_relation_l96_96599


namespace problem_l96_96898

theorem problem (a b : ℕ) (h1 : ∃ k : ℕ, a * b = k * k) (h2 : ∃ m : ℕ, (2 * a + 1) * (2 * b + 1) = m * m) :
  ∃ n : ℕ, n % 2 = 0 ∧ n > 2 ∧ ∃ p : ℕ, (a + n) * (b + n) = p * p :=
by
  sorry

end problem_l96_96898


namespace find_a_b_find_solution_set_l96_96703

-- Conditions
variable {a b c x : ℝ}

-- Given inequality condition
def given_inequality (x : ℝ) (a b : ℝ) : Prop := a * x^2 + x + b > 0

-- Define the solution set
def solution_set (x : ℝ) (a b : ℝ) : Prop :=
  (x < -2 ∨ x > 1) ↔ given_inequality x a b

-- Part I: Prove values of a and b
theorem find_a_b
  (H : ∀ x, solution_set x a b) :
  a = 1 ∧ b = -2 := by sorry

-- Define the second inequality
def second_inequality (x : ℝ) (c : ℝ) : Prop := x^2 - (c - 2) * x - 2 * c < 0

-- Solution set for the second inequality
def second_solution_set (x : ℝ) (c : ℝ) : Prop :=
  (c = -2 → False) ∧
  (c > -2 → -2 < x ∧ x < c) ∧
  (c < -2 → c < x ∧ x < -2)

-- Part II: Prove the solution set
theorem find_solution_set
  (H : a = 1)
  (H1 : b = -2) :
  ∀ x, second_solution_set x c ↔ second_inequality x c := by sorry

end find_a_b_find_solution_set_l96_96703


namespace fourth_root_equiv_l96_96694

theorem fourth_root_equiv (x : ℝ) (hx : 0 < x) : (x * (x ^ (3 / 4))) ^ (1 / 4) = x ^ (7 / 16) :=
sorry

end fourth_root_equiv_l96_96694


namespace intersect_at_0_intersect_at_180_intersect_at_90_l96_96795

-- Define radii R and r, and the distance c
variables {R r c : ℝ}

-- Formalize the conditions and corresponding angles
theorem intersect_at_0 (h : c = R - r) : True := 
sorry

theorem intersect_at_180 (h : c = R + r) : True := 
sorry

theorem intersect_at_90 (h : c = Real.sqrt (R^2 + r^2)) : True := 
sorry

end intersect_at_0_intersect_at_180_intersect_at_90_l96_96795


namespace find_g_neg_6_l96_96518

noncomputable def g : ℤ → ℤ := sorry

-- Conditions from the problem
axiom g_condition_1 : g 1 - 1 > 0
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The statement we need to prove
theorem find_g_neg_6 : g (-6) = 723 :=
by { sorry }

end find_g_neg_6_l96_96518


namespace train_length_correct_l96_96934

noncomputable def speed_kmph : ℝ := 60
noncomputable def time_sec : ℝ := 6

-- Conversion factor from km/hr to m/s
noncomputable def conversion_factor := (1000 : ℝ) / 3600

-- Speed in m/s
noncomputable def speed_mps := speed_kmph * conversion_factor

-- Length of the train
noncomputable def train_length := speed_mps * time_sec

theorem train_length_correct :
  train_length = 100.02 :=
by
  sorry

end train_length_correct_l96_96934


namespace y_eq_fraction_x_l96_96900

theorem y_eq_fraction_x (p : ℝ) (x y : ℝ) (hx : x = 1 + 2^p) (hy : y = 1 + 2^(-p)) : y = x / (x - 1) :=
sorry

end y_eq_fraction_x_l96_96900


namespace Maria_high_school_students_l96_96541

theorem Maria_high_school_students (M J : ℕ) (h1 : M = 4 * J) (h2 : M + J = 3600) : M = 2880 :=
sorry

end Maria_high_school_students_l96_96541


namespace stratified_sampling_correct_l96_96313

-- Definitions for the conditions
def total_employees : ℕ := 750
def young_employees : ℕ := 350
def middle_aged_employees : ℕ := 250
def elderly_employees : ℕ := 150
def sample_size : ℕ := 15
def sampling_proportion : ℚ := sample_size / total_employees

-- Statement to prove
theorem stratified_sampling_correct :
  (young_employees * sampling_proportion = 7) ∧
  (middle_aged_employees * sampling_proportion = 5) ∧
  (elderly_employees * sampling_proportion = 3) :=
by
  sorry

end stratified_sampling_correct_l96_96313


namespace retailer_marked_price_percentage_above_cost_l96_96605

noncomputable def cost_price : ℝ := 100
noncomputable def discount_rate : ℝ := 0.15
noncomputable def sales_profit_rate : ℝ := 0.275

theorem retailer_marked_price_percentage_above_cost :
  ∃ (MP : ℝ), ((MP - cost_price) / cost_price = 0.5) ∧ (((MP * (1 - discount_rate)) - cost_price) / cost_price = sales_profit_rate) :=
sorry

end retailer_marked_price_percentage_above_cost_l96_96605


namespace initial_oranges_l96_96839

theorem initial_oranges (O : ℕ) (h1 : O + 6 - 3 = 6) : O = 3 :=
by
  sorry

end initial_oranges_l96_96839


namespace max_product_l96_96398

def geometric_sequence (a1 q : ℝ) (n : ℕ) :=
  a1 * q ^ (n - 1)

def product_of_terms (a1 q : ℝ) (n : ℕ) :=
  (List.range n).foldr (λ i acc => acc * geometric_sequence a1 q (i + 1)) 1

theorem max_product (n : ℕ) (a1 q : ℝ) (h₁ : a1 = 1536) (h₂ : q = -1/2) :
  n = 11 ↔ ∀ m : ℕ, m ≤ 11 → product_of_terms a1 q m ≤ product_of_terms a1 q 11 :=
by
  sorry

end max_product_l96_96398


namespace correct_option_is_C_l96_96022

variable (a b : ℝ)

def option_A : Prop := (a - b) ^ 2 = a ^ 2 - b ^ 2
def option_B : Prop := a ^ 2 + a ^ 2 = a ^ 4
def option_C : Prop := (a ^ 2) ^ 3 = a ^ 6
def option_D : Prop := a ^ 2 * a ^ 2 = a ^ 6

theorem correct_option_is_C : option_C a :=
by
  sorry

end correct_option_is_C_l96_96022


namespace find_parameter_a_exactly_two_solutions_l96_96333

noncomputable def system_has_two_solutions (a : ℝ) : Prop :=
∃ (x y : ℝ), |y - 3 - x| + |y - 3 + x| = 6 ∧ (|x| - 4)^2 + (|y| - 3)^2 = a

theorem find_parameter_a_exactly_two_solutions :
  {a : ℝ | system_has_two_solutions a} = {1, 25} :=
by
  sorry

end find_parameter_a_exactly_two_solutions_l96_96333


namespace sum_of_roots_is_three_l96_96034

theorem sum_of_roots_is_three :
  ∀ (x1 x2 : ℝ), (x1^2 - 3 * x1 - 4 = 0) ∧ (x2^2 - 3 * x2 - 4 = 0) → x1 + x2 = 3 :=
by sorry

end sum_of_roots_is_three_l96_96034


namespace cos_2theta_l96_96508

theorem cos_2theta (θ : ℝ) (h : Real.tan θ = Real.sqrt 5) : Real.cos (2 * θ) = -2 / 3 :=
by
  sorry

end cos_2theta_l96_96508


namespace total_red_beads_l96_96480

theorem total_red_beads (total_beads : ℕ) (pattern_length : ℕ) (green_beads : ℕ) (red_beads : ℕ) (yellow_beads : ℕ) 
                         (h_total: total_beads = 85) 
                         (h_pattern: pattern_length = green_beads + red_beads + yellow_beads) 
                         (h_cycle: green_beads = 3 ∧ red_beads = 4 ∧ yellow_beads = 1) : 
                         (red_beads * (total_beads / pattern_length)) + (min red_beads (total_beads % pattern_length)) = 42 :=
by
  sorry

end total_red_beads_l96_96480


namespace find_sachin_age_l96_96029

variables (S R : ℕ)

def sachin_young_than_rahul_by_4_years (S R : ℕ) : Prop := R = S + 4
def ratio_of_ages (S R : ℕ) : Prop := 7 * R = 9 * S

theorem find_sachin_age (S R : ℕ) (h1 : sachin_young_than_rahul_by_4_years S R) (h2 : ratio_of_ages S R) : S = 14 := 
by sorry

end find_sachin_age_l96_96029


namespace sum_of_inserted_numbers_l96_96557

theorem sum_of_inserted_numbers (x y : ℝ) (h1 : x^2 = 2 * y) (h2 : 2 * y = x + 20) :
  x + y = 4 ∨ x + y = 17.5 :=
sorry

end sum_of_inserted_numbers_l96_96557


namespace find_y_minus_x_l96_96606

theorem find_y_minus_x (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x < y) 
  (h4 : Real.sqrt x + Real.sqrt y = 1) 
  (h5 : Real.sqrt (x / y) + Real.sqrt (y / x) = 10 / 3) : 
  y - x = 1 / 2 :=
sorry

end find_y_minus_x_l96_96606


namespace calculate_Al2O3_weight_and_H2_volume_l96_96180

noncomputable def weight_of_Al2O3 (moles : ℕ) : ℝ :=
  moles * ((2 * 26.98) + (3 * 16.00))

noncomputable def volume_of_H2_at_STP (moles_of_Al2O3 : ℕ) : ℝ :=
  (moles_of_Al2O3 * 3) * 22.4

theorem calculate_Al2O3_weight_and_H2_volume :
  weight_of_Al2O3 6 = 611.76 ∧ volume_of_H2_at_STP 6 = 403.2 :=
by
  sorry

end calculate_Al2O3_weight_and_H2_volume_l96_96180


namespace inequality_proof_l96_96573

theorem inequality_proof (a b c d : ℝ) 
  (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d)
  (a_geq_1 : 1 ≤ a) (b_geq_1 : 1 ≤ b) (c_geq_1 : 1 ≤ c)
  (abcd_eq_1 : a * b * c * d = 1)
  : 
  1 / (a^2 - a + 1)^2 + 1 / (b^2 - b + 1)^2 + 1 / (c^2 - c + 1)^2 + 1 / (d^2 - d + 1)^2 ≤ 4
  := sorry

end inequality_proof_l96_96573


namespace at_least_one_truth_and_not_knight_l96_96302

def isKnight (n : Nat) : Prop := n = 1   -- Identifier for knights
def isKnave (n : Nat) : Prop := n = 0    -- Identifier for knaves
def isRegular (n : Nat) : Prop := n = 2  -- Identifier for regular persons

def A := 2     -- Initially define A's type as regular (this can be adjusted)
def B := 2     -- Initially define B's type as regular (this can be adjusted)

def statementA : Prop := isKnight B
def statementB : Prop := ¬ isKnight A

theorem at_least_one_truth_and_not_knight :
  statementA ∧ ¬ isKnight A ∨ statementB ∧ ¬ isKnight B :=
sorry

end at_least_one_truth_and_not_knight_l96_96302


namespace fare_collected_from_I_class_l96_96246

theorem fare_collected_from_I_class (x y : ℝ) 
  (h1 : ∀i, i = x → ∀ii, ii = 4 * x)
  (h2 : ∀f1, f1 = 3 * y)
  (h3 : ∀f2, f2 = y)
  (h4 : x * 3 * y + 4 * x * y = 224000) : 
  x * 3 * y = 96000 :=
by
  sorry

end fare_collected_from_I_class_l96_96246


namespace value_of_f_sin_7pi_over_6_l96_96189

def f (x : ℝ) : ℝ := 4 * x^2 + 2 * x

theorem value_of_f_sin_7pi_over_6 :
  f (Real.sin (7 * Real.pi / 6)) = 0 :=
by
  sorry

end value_of_f_sin_7pi_over_6_l96_96189


namespace Fermat_numbers_are_not_cubes_l96_96260

def F (n : ℕ) : ℕ := 2^(2^n) + 1

theorem Fermat_numbers_are_not_cubes : ∀ n : ℕ, ¬ ∃ k : ℕ, F n = k^3 :=
by
  sorry

end Fermat_numbers_are_not_cubes_l96_96260


namespace solve_integers_l96_96964

theorem solve_integers (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  x^(2 * y) + (x + 1)^(2 * y) = (x + 2)^(2 * y) → (x = 3 ∧ y = 1) :=
by
  sorry

end solve_integers_l96_96964


namespace remainder_of_expression_l96_96654

theorem remainder_of_expression (n : ℤ) (h : n % 60 = 1) : (n^2 + 2 * n + 3) % 60 = 6 := 
by
  sorry

end remainder_of_expression_l96_96654


namespace shoe_store_total_shoes_l96_96389

theorem shoe_store_total_shoes (b k : ℕ) (h1 : b = 22) (h2 : k = 2 * b) : b + k = 66 :=
by
  sorry

end shoe_store_total_shoes_l96_96389


namespace true_statement_for_f_l96_96271

variable (c : ℝ) (f : ℝ → ℝ)

theorem true_statement_for_f :
  (∀ x : ℝ, f x = x^2 - 2 * x + c) → (∀ x : ℝ, f x ≥ c - 1) :=
by
  sorry

end true_statement_for_f_l96_96271


namespace range_of_independent_variable_l96_96130

theorem range_of_independent_variable (x : ℝ) : 
  (∃ y, y = x / (Real.sqrt (x + 4)) + 1 / (x - 1)) ↔ x > -4 ∧ x ≠ 1 := 
by
  sorry

end range_of_independent_variable_l96_96130


namespace remainder_43_pow_43_plus_43_mod_44_l96_96822

theorem remainder_43_pow_43_plus_43_mod_44 : (43^43 + 43) % 44 = 42 :=
by 
    sorry

end remainder_43_pow_43_plus_43_mod_44_l96_96822


namespace product_no_xx_x_eq_x_cube_plus_one_l96_96391

theorem product_no_xx_x_eq_x_cube_plus_one (a c : ℝ) (h1 : a - 1 = 0) (h2 : c - a = 0) : 
  (x + a) * (x ^ 2 - x + c) = x ^ 3 + 1 :=
by {
  -- Here would be the proof steps, which we omit with "sorry"
  sorry
}

end product_no_xx_x_eq_x_cube_plus_one_l96_96391


namespace range_of_m_l96_96788

open Real

noncomputable def satisfies_inequality (m : ℝ) : Prop :=
  ∀ (x : ℝ), x > 0 → log x ≤ x * exp (m^2 - m - 1)

theorem range_of_m : 
  {m : ℝ | satisfies_inequality m} = {m : ℝ | m ≤ 0 ∨ m ≥ 1} :=
by 
  sorry

end range_of_m_l96_96788


namespace find_divisor_l96_96880

theorem find_divisor (q r D : ℕ) (hq : q = 120) (hr : r = 333) (hD : 55053 = D * q + r) : D = 456 :=
by
  sorry

end find_divisor_l96_96880


namespace area_of_triangle_abe_l96_96362

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
10 -- Dummy definition, in actual scenario appropriate area calculation will be required.

def length_AD : ℝ := 2
def length_BD : ℝ := 3

def areas_equal (S_ABE S_DBFE : ℝ) : Prop :=
    S_ABE = S_DBFE

theorem area_of_triangle_abe
  (area_abc : ℝ)
  (length_ad length_bd : ℝ)
  (equal_areas : areas_equal (triangle_area 1 1 1) 1) -- Dummy values, should be substituted with correct arguments
  : triangle_area 1 1 1 = 6 :=
sorry -- proof will be filled later

end area_of_triangle_abe_l96_96362


namespace a_plus_d_eq_zero_l96_96197

noncomputable def f (a b c d x : ℝ) : ℝ := (2 * a * x + b) / (c * x + 2 * d)

theorem a_plus_d_eq_zero (a b c d : ℝ) (h : a * b * c * d ≠ 0) (hff : ∀ x, f a b c d (f a b c d x) = 3 * x - 4) : a + d = 0 :=
by
  sorry

end a_plus_d_eq_zero_l96_96197


namespace minimum_value_of_a_l96_96958

theorem minimum_value_of_a :
  ∀ (x : ℝ), (2 * x + 2 / (x - 1) ≥ 7) ↔ (3 ≤ x) :=
sorry

end minimum_value_of_a_l96_96958


namespace factorize_quadratic_l96_96689

theorem factorize_quadratic (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := 
by {
  sorry
}

end factorize_quadratic_l96_96689


namespace parabola_equation_l96_96731

theorem parabola_equation {p : ℝ} (hp : 0 < p)
  (h_cond : ∃ A B : ℝ × ℝ, (A.1^2 = 2 * A.2 * p) ∧ (B.1^2 = 2 * B.2 * p) ∧ (A.2 = A.1 - p / 2) ∧ (B.2 = B.1 - p / 2) ∧ (|A.1 - B.1|^2 + |A.2 - B.2|^2 = 4))
  : y^2 = 2 * x := sorry

end parabola_equation_l96_96731


namespace original_number_l96_96726

theorem original_number (x : ℝ) (h : 1.50 * x = 165) : x = 110 :=
sorry

end original_number_l96_96726


namespace no_integer_n_squared_plus_one_div_by_seven_l96_96360

theorem no_integer_n_squared_plus_one_div_by_seven (n : ℤ) : ¬ (n^2 + 1) % 7 = 0 := 
sorry

end no_integer_n_squared_plus_one_div_by_seven_l96_96360


namespace opposite_sides_line_range_a_l96_96063

theorem opposite_sides_line_range_a (a : ℝ) :
  (3 * 2 - 2 * 1 + a) * (3 * -1 - 2 * 3 + a) < 0 → -4 < a ∧ a < 9 := by
  sorry

end opposite_sides_line_range_a_l96_96063


namespace jessica_deposited_fraction_l96_96817

-- Definitions based on conditions
def original_balance (B : ℝ) : Prop :=
  B * (3 / 5) = B - 200

def final_balance (B : ℝ) (F : ℝ) : Prop :=
  ((3 / 5) * B) + (F * ((3 / 5) * B)) = 360

-- Theorem statement proving that the fraction deposited is 1/5
theorem jessica_deposited_fraction (B : ℝ) (F : ℝ) (h1 : original_balance B) (h2 : final_balance B F) : F = 1 / 5 :=
  sorry

end jessica_deposited_fraction_l96_96817


namespace value_of_m_l96_96059

def p (m : ℝ) : Prop :=
  4 < m ∧ m < 10

def q (m : ℝ) : Prop :=
  8 < m ∧ m < 12

theorem value_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬ (p m ∧ q m) ↔ (4 < m ∧ m ≤ 8) ∨ (10 ≤ m ∧ m < 12) :=
by
  sorry

end value_of_m_l96_96059


namespace june_1_friday_l96_96531

open Nat

-- Define the days of the week as data type
inductive DayOfWeek : Type
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

open DayOfWeek

-- Define that June has 30 days
def june_days := 30

-- Hypotheses that June has exactly three Mondays and exactly three Thursdays
def three_mondays (d : DayOfWeek) : Prop := 
  ∃ days : Fin 30 → DayOfWeek, 
    (∀ n : Fin 30, days n = Monday → 3 ≤ n / 7) -- there are exactly three Mondays
  
def three_thursdays (d : DayOfWeek) : Prop := 
  ∃ days : Fin 30 → DayOfWeek, 
    (∀ n : Fin 30, days n = Thursday → 3 ≤ n / 7) -- there are exactly three Thursdays

-- Theorem to prove June 1 falls on a Friday given those conditions
theorem june_1_friday : ∀ (d : DayOfWeek), 
  three_mondays d → three_thursdays d → (d = Friday) :=
by
  sorry

end june_1_friday_l96_96531


namespace highest_daily_profit_and_total_profit_l96_96594

def cost_price : ℕ := 6
def standard_price : ℕ := 10

def price_relative (day : ℕ) : ℤ := 
  match day with
  | 1 => 3
  | 2 => 2
  | 3 => 1
  | 4 => -1
  | 5 => -2
  | _ => 0

def quantity_sold (day : ℕ) : ℕ :=
  match day with
  | 1 => 7
  | 2 => 12
  | 3 => 15
  | 4 => 32
  | 5 => 34
  | _ => 0

noncomputable def selling_price (day : ℕ) : ℤ := standard_price + price_relative day

noncomputable def profit_per_pen (day : ℕ) : ℤ := (selling_price day) - cost_price

noncomputable def daily_profit (day : ℕ) : ℤ := (profit_per_pen day) * (quantity_sold day)

theorem highest_daily_profit_and_total_profit 
  (h_highest_profit: daily_profit 4 = 96) 
  (h_total_profit: daily_profit 1 + daily_profit 2 + daily_profit 3 + daily_profit 4 + daily_profit 5 = 360) : 
  True :=
by
  sorry

end highest_daily_profit_and_total_profit_l96_96594


namespace range_of_x_l96_96012

theorem range_of_x (a x : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  (x^2 + a * x > 4 * x + a - 3) ↔ (x < -1 ∨ x > 3) :=
by
  sorry

end range_of_x_l96_96012


namespace ordered_pairs_condition_l96_96789

theorem ordered_pairs_condition (m n : ℕ) (hmn : m ≥ n) (hm_pos : 0 < m) (hn_pos : 0 < n) (h_eq : 3 * m * n = 8 * (m + n - 1)) :
    (m, n) = (16, 3) ∨ (m, n) = (6, 4) := by
  sorry

end ordered_pairs_condition_l96_96789


namespace value_y1_y2_l96_96676

variable {x1 x2 y1 y2 : ℝ}

-- Points on the inverse proportion function
def on_graph (x y : ℝ) : Prop := y = -3 / x

-- Given conditions
theorem value_y1_y2 (hx1 : on_graph x1 y1) (hx2 : on_graph x2 y2) (hxy : x1 * x2 = 2) : y1 * y2 = 9 / 2 :=
by
  sorry

end value_y1_y2_l96_96676


namespace range_of_a_l96_96077

variable {α : Type} [LinearOrderedField α]

def A (a : α) : Set α := {x | |x - a| ≤ 1}

def B : Set α := {x | x^2 - 5*x + 4 ≥ 0}

theorem range_of_a (a : α) (h : A a ∩ B = ∅) : 2 < a ∧ a < 3 := sorry

end range_of_a_l96_96077


namespace decreased_price_correct_l96_96501

def actual_cost : ℝ := 250
def percentage_decrease : ℝ := 0.2

theorem decreased_price_correct : actual_cost - (percentage_decrease * actual_cost) = 200 :=
by
  sorry

end decreased_price_correct_l96_96501


namespace tan_150_eq_neg_inv_sqrt3_l96_96125

theorem tan_150_eq_neg_inv_sqrt3 :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  have cos_30 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
    by sorry
  sorry

end tan_150_eq_neg_inv_sqrt3_l96_96125


namespace ratio_pages_l96_96974

theorem ratio_pages (pages_Selena pages_Harry : ℕ) (h₁ : pages_Selena = 400) (h₂ : pages_Harry = 180) : 
  pages_Harry / pages_Selena = 9 / 20 := 
by
  -- proof goes here
  sorry

end ratio_pages_l96_96974


namespace find_a8_in_arithmetic_sequence_l96_96228

variable {a : ℕ → ℕ} -- Define a as a function from natural numbers to natural numbers

-- Assume a is an arithmetic sequence
axiom arithmetic_sequence (a : ℕ → ℕ) : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a8_in_arithmetic_sequence (h : a 4 + a 6 + a 8 + a 10 + a 12 = 120) : a 8 = 24 :=
by
  sorry  -- Proof to be filled in separately

end find_a8_in_arithmetic_sequence_l96_96228


namespace product_mod_23_l96_96351

theorem product_mod_23 :
  (2003 * 2004 * 2005 * 2006 * 2007 * 2008) % 23 = 3 :=
by 
  sorry

end product_mod_23_l96_96351


namespace roots_sum_cubes_l96_96019

theorem roots_sum_cubes (a b c d : ℝ) 
  (h_eqn : ∀ x : ℝ, (x = a ∨ x = b ∨ x = c ∨ x = d) → 
    3 * x^4 + 6 * x^3 + 1002 * x^2 + 2005 * x + 4010 = 0) :
  (a + b)^3 + (b + c)^3 + (c + d)^3 + (d + a)^3 = 9362 :=
by { sorry }

end roots_sum_cubes_l96_96019


namespace sum_pos_integers_9_l96_96113

theorem sum_pos_integers_9 (x y z : ℕ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_eq : 30 / 7 = x + 1 / (y + 1 / z)) : x + y + z = 9 :=
sorry

end sum_pos_integers_9_l96_96113


namespace new_volume_proof_l96_96472

variable (r h : ℝ)
variable (π : ℝ := Real.pi) -- Lean's notation for π
variable (original_volume : ℝ := 15) -- given original volume

-- Define original volume of the cylinder
def V := π * r^2 * h

-- Define new volume of the cylinder using new dimensions
def new_V := π * (3 * r)^2 * (2 * h)

-- Prove that new_V is 270 when V = 15
theorem new_volume_proof (hV : V = 15) : new_V = 270 :=
by
  -- Proof will go here
  sorry

end new_volume_proof_l96_96472


namespace geometric_mean_of_negatives_l96_96549

theorem geometric_mean_of_negatives :
  ∃ x : ℝ, x^2 = (-2) * (-8) ∧ (x = 4 ∨ x = -4) := by
  sorry

end geometric_mean_of_negatives_l96_96549


namespace xyz_value_l96_96470

theorem xyz_value (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 21) :
  x * y * z = 28 / 3 :=
by
  sorry

end xyz_value_l96_96470


namespace probability_white_ball_is_two_fifths_l96_96288

-- Define the total number of each type of balls.
def white_balls : ℕ := 6
def yellow_balls : ℕ := 5
def red_balls : ℕ := 4

-- Calculate the total number of balls in the bag.
def total_balls : ℕ := white_balls + yellow_balls + red_balls

-- Define the probability calculation.
noncomputable def probability_of_white_ball : ℚ := white_balls / total_balls

-- The theorem statement asserting the probability of drawing a white ball.
theorem probability_white_ball_is_two_fifths :
  probability_of_white_ball = 2 / 5 :=
sorry

end probability_white_ball_is_two_fifths_l96_96288


namespace kellan_wax_remaining_l96_96013

def remaining_wax (initial_A : ℕ) (initial_B : ℕ)
                  (spill_A : ℕ) (spill_B : ℕ)
                  (use_car_A : ℕ) (use_suv_B : ℕ) : ℕ :=
  let remaining_A := initial_A - spill_A - use_car_A
  let remaining_B := initial_B - spill_B - use_suv_B
  remaining_A + remaining_B

theorem kellan_wax_remaining
  (initial_A : ℕ := 10) 
  (initial_B : ℕ := 15)
  (spill_A : ℕ := 3) 
  (spill_B : ℕ := 4)
  (use_car_A : ℕ := 4) 
  (use_suv_B : ℕ := 5) :
  remaining_wax initial_A initial_B spill_A spill_B use_car_A use_suv_B = 9 :=
by sorry

end kellan_wax_remaining_l96_96013


namespace factorize_x2_minus_9_l96_96695

theorem factorize_x2_minus_9 (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := 
sorry

end factorize_x2_minus_9_l96_96695


namespace circle_equation_center_line_l96_96287

theorem circle_equation_center_line (x y : ℝ) :
  -- Conditions
  (∀ (x1 y1 : ℝ), x1 + y1 - 2 = 0 → (x = 1 ∧ y = 1)) ∧
  ((x - 1)^2 + (y - 1)^2 = 4) ∧
  -- Points A and B
  (∀ (xA yA : ℝ), xA = 1 ∧ yA = -1 ∨ xA = -1 ∧ yA = 1 →
    ((xA - x)^2 + (yA - y)^2 = 4)) :=
by
  sorry

end circle_equation_center_line_l96_96287


namespace pablo_puzzle_l96_96429

open Nat

theorem pablo_puzzle (pieces_per_hour : ℕ) (hours_per_day : ℕ) (days : ℕ) 
    (pieces_per_five_puzzles : ℕ) (num_five_puzzles : ℕ) (total_pieces : ℕ) 
    (num_eight_puzzles : ℕ) :

    pieces_per_hour = 100 →
    hours_per_day = 7 →
    days = 7 →
    pieces_per_five_puzzles = 500 →
    num_five_puzzles = 5 →
    num_eight_puzzles = 8 →
    total_pieces = (pieces_per_hour * hours_per_day * days) →
    num_eight_puzzles * (total_pieces - num_five_puzzles * pieces_per_five_puzzles) / num_eight_puzzles = 300 :=
by
  intros
  sorry

end pablo_puzzle_l96_96429


namespace cost_price_per_meter_l96_96387

def total_length : ℝ := 9.25
def total_cost : ℝ := 397.75

theorem cost_price_per_meter : total_cost / total_length = 43 := sorry

end cost_price_per_meter_l96_96387


namespace simplify_and_evaluate_l96_96574

theorem simplify_and_evaluate : 
  (1 / (3 - 2) - 1 / (3 + 1)) / (3 / (3^2 - 1)) = 2 :=
by
  sorry

end simplify_and_evaluate_l96_96574


namespace tracy_first_week_books_collected_l96_96873

-- Definitions for collection multipliers
def first_week (T : ℕ) := T
def second_week (T : ℕ) := 2 * T + 3 * T
def third_week (T : ℕ) := 3 * T + 4 * T + (T / 2)
def fourth_week (T : ℕ) := 4 * T + 5 * T + T
def fifth_week (T : ℕ) := 5 * T + 6 * T + 2 * T
def sixth_week (T : ℕ) := 6 * T + 7 * T + 3 * T

-- Summing up total books collected
def total_books_collected (T : ℕ) : ℕ :=
  first_week T + second_week T + third_week T + fourth_week T + fifth_week T + sixth_week T

-- Proof statement (unchanged for now)
theorem tracy_first_week_books_collected (T : ℕ) :
  total_books_collected T = 1025 → T = 20 :=
by
  sorry

end tracy_first_week_books_collected_l96_96873


namespace chips_recoloring_impossible_l96_96276

theorem chips_recoloring_impossible :
  (∀ a b c : ℕ, a = 2008 ∧ b = 2009 ∧ c = 2010 →
   ¬(∃ k : ℕ, a + b + c = k ∧ (a = k ∨ b = k ∨ c = k))) :=
by sorry

end chips_recoloring_impossible_l96_96276


namespace no_positive_numbers_satisfy_conditions_l96_96466

theorem no_positive_numbers_satisfy_conditions :
  ¬ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b + c = ab + ac + bc) ∧ (ab + ac + bc = abc) :=
by
  sorry

end no_positive_numbers_satisfy_conditions_l96_96466


namespace smallest_n_terminating_contains_9_l96_96257

def isTerminatingDecimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2 ^ a * 5 ^ b

def containsDigit9 (n : ℕ) : Prop :=
  (Nat.digits 10 n).contains 9

theorem smallest_n_terminating_contains_9 : ∃ n : ℕ, 
  isTerminatingDecimal n ∧
  containsDigit9 n ∧
  (∀ m : ℕ, isTerminatingDecimal m ∧ containsDigit9 m → n ≤ m) ∧
  n = 5120 :=
  sorry

end smallest_n_terminating_contains_9_l96_96257


namespace find_triangle_sides_l96_96141

noncomputable def triangle_sides (x : ℝ) : Prop :=
  let a := x - 2
  let b := x
  let c := x + 2
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  a + 2 = b ∧ b + 2 = c ∧ area = 6 ∧
  a = 2 * Real.sqrt 6 - 2 ∧
  b = 2 * Real.sqrt 6 ∧
  c = 2 * Real.sqrt 6 + 2

theorem find_triangle_sides :
  ∃ x : ℝ, triangle_sides x := by
  sorry

end find_triangle_sides_l96_96141


namespace cheaper_price_difference_is_75_cents_l96_96681

noncomputable def list_price := 42.50
noncomputable def store_a_discount := 12.00
noncomputable def store_b_discount_percent := 0.30

noncomputable def store_a_price := list_price - store_a_discount
noncomputable def store_b_price := (1 - store_b_discount_percent) * list_price
noncomputable def price_difference_in_dollars := store_a_price - store_b_price
noncomputable def price_difference_in_cents := price_difference_in_dollars * 100

theorem cheaper_price_difference_is_75_cents :
  price_difference_in_cents = 75 := by
  sorry

end cheaper_price_difference_is_75_cents_l96_96681


namespace not_possible_perimeter_l96_96259

theorem not_possible_perimeter :
  ∀ (x : ℝ), 13 < x ∧ x < 37 → ¬ (37 + x = 50) :=
by
  intros x h
  sorry

end not_possible_perimeter_l96_96259


namespace tan_alpha_minus_beta_l96_96102

theorem tan_alpha_minus_beta
  (α β : ℝ)
  (tan_alpha : Real.tan α = 2)
  (tan_beta : Real.tan β = -7) :
  Real.tan (α - β) = -9 / 13 :=
by sorry

end tan_alpha_minus_beta_l96_96102


namespace quotient_of_large_div_small_l96_96282

theorem quotient_of_large_div_small (L S : ℕ) (h1 : L - S = 1365)
  (h2 : L = S * (L / S) + 20) (h3 : L = 1634) : (L / S) = 6 := by
  sorry

end quotient_of_large_div_small_l96_96282


namespace students_at_year_end_l96_96220

theorem students_at_year_end (initial_students left_students new_students end_students : ℕ)
  (h_initial : initial_students = 31)
  (h_left : left_students = 5)
  (h_new : new_students = 11)
  (h_end : end_students = initial_students - left_students + new_students) :
  end_students = 37 :=
by
  sorry

end students_at_year_end_l96_96220


namespace roses_count_l96_96907

def total_roses : Nat := 80
def red_roses : Nat := 3 * total_roses / 4
def remaining_roses : Nat := total_roses - red_roses
def yellow_roses : Nat := remaining_roses / 4
def white_roses : Nat := remaining_roses - yellow_roses

theorem roses_count :
  red_roses + white_roses = 75 :=
by
  sorry

end roses_count_l96_96907


namespace probability_both_selected_is_correct_l96_96052

def prob_selection_x : ℚ := 1 / 7
def prob_selection_y : ℚ := 2 / 9
def prob_both_selected : ℚ := prob_selection_x * prob_selection_y

theorem probability_both_selected_is_correct : prob_both_selected = 2 / 63 := 
by 
  sorry

end probability_both_selected_is_correct_l96_96052


namespace rectangle_area_is_140_l96_96099

noncomputable def area_of_square (a : ℝ) : ℝ := a * a
noncomputable def length_of_rectangle (r : ℝ) : ℝ := (2 / 5) * r
noncomputable def area_of_rectangle (l : ℝ) (b : ℝ) : ℝ := l * b

theorem rectangle_area_is_140 :
  ∃ (a r l b : ℝ), area_of_square a = 1225 ∧ r = a ∧ l = length_of_rectangle r ∧ b = 10 ∧ area_of_rectangle l b = 140 :=
by
  use 35, 35, 14, 10
  simp [area_of_square, length_of_rectangle, area_of_rectangle]
  sorry

end rectangle_area_is_140_l96_96099


namespace magnitude_product_l96_96237

-- Definitions based on conditions
def z1 : Complex := ⟨7, -4⟩
def z2 : Complex := ⟨3, 10⟩

-- Statement of the theorem to be proved
theorem magnitude_product :
  Complex.abs (z1 * z2) = Real.sqrt 7085 := by
  sorry

end magnitude_product_l96_96237


namespace product_of_triangle_areas_not_end_in_1988_l96_96007

theorem product_of_triangle_areas_not_end_in_1988
  (a b c d : ℕ)
  (h1 : a * c = b * d)
  (hp : (a * b * c * d) = (a * c)^2)
  : ¬(∃ k : ℕ, (a * b * c * d) = 10000 * k + 1988) :=
sorry

end product_of_triangle_areas_not_end_in_1988_l96_96007


namespace total_sheets_folded_l96_96082

theorem total_sheets_folded (initially_folded : ℕ) (additionally_folded : ℕ) (total_folded : ℕ) :
  initially_folded = 45 → additionally_folded = 18 → total_folded = initially_folded + additionally_folded → total_folded = 63 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end total_sheets_folded_l96_96082


namespace quadratic_points_relation_l96_96660

theorem quadratic_points_relation
  (k y₁ y₂ y₃ : ℝ)
  (hA : y₁ = -((-1) - 1)^2 + k)
  (hB : y₂ = -(2 - 1)^2 + k)
  (hC : y₃ = -(4 - 1)^2 + k) : y₃ < y₁ ∧ y₁ < y₂ :=
by
  sorry

end quadratic_points_relation_l96_96660


namespace difference_in_percentage_l96_96914

noncomputable def principal : ℝ := 600
noncomputable def timePeriod : ℝ := 10
noncomputable def interestDifference : ℝ := 300

theorem difference_in_percentage (R D : ℝ) (h : 60 * (R + D) - 60 * R = 300) : D = 5 := 
by
  -- Proof is not provided, as instructed
  sorry

end difference_in_percentage_l96_96914


namespace complex_value_of_product_l96_96166

theorem complex_value_of_product (r : ℂ) (hr : r^7 = 1) (hr1 : r ≠ 1) : 
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 8 := 
by sorry

end complex_value_of_product_l96_96166


namespace set_difference_example_l96_96922

-- Define P and Q based on the given conditions
def P : Set ℝ := {x | 0 < x ∧ x < 2}
def Q : Set ℝ := {x | 1 < x ∧ x < 3}

-- State the theorem: P - Q equals to the set {x | 0 < x ≤ 1}
theorem set_difference_example : P \ Q = {x | 0 < x ∧ x ≤ 1} := 
  by
  sorry

end set_difference_example_l96_96922


namespace polynomial_divisible_x_minus_2_l96_96154

theorem polynomial_divisible_x_minus_2 (m : ℝ) : 
  (3 * 2^2 - 9 * 2 + m = 0) → m = 6 :=
by
  sorry

end polynomial_divisible_x_minus_2_l96_96154


namespace friends_meeting_time_l96_96484

noncomputable def speed_B (t : ℕ) : ℝ := 4 + 0.75 * (t - 1)

noncomputable def distance_B (t : ℕ) : ℝ :=
  t * 4 + (0.375 * t * (t - 1))

noncomputable def distance_A (t : ℕ) : ℝ := 5 * t

theorem friends_meeting_time :
  ∃ t : ℝ, 5 * t + (t / 2) * (7.25 + 0.75 * t) = 120 ∧ t = 8 :=
by
  sorry

end friends_meeting_time_l96_96484


namespace find_y_l96_96960

theorem find_y {x y : ℤ} (h1 : x - y = 12) (h2 : x + y = 6) : y = -3 := 
by
  sorry

end find_y_l96_96960


namespace area_of_T_shaped_region_l96_96025

theorem area_of_T_shaped_region :
  let ABCD_area : ℝ := 48
  let EFHG_area : ℝ := 4
  let EFGI_area : ℝ := 8
  let EFCD_area : ℝ := 12
  (ABCD_area - (EFHG_area + EFGI_area + EFCD_area)) = 24 :=
by
  let ABCD_area : ℝ := 48
  let EFHG_area : ℝ := 4
  let EFGI_area : ℝ := 8
  let EFCD_area : ℝ := 12
  exact sorry

end area_of_T_shaped_region_l96_96025


namespace power_function_decreasing_m_eq_2_l96_96965

theorem power_function_decreasing_m_eq_2 (x : ℝ) (m : ℝ) (hx : 0 < x) 
  (h_decreasing : ∀ x₁ x₂, 0 < x₁ → 0 < x₂ → x₁ < x₂ → 
                    (m^2 - m - 1) * x₁^(-m+1) > (m^2 - m - 1) * x₂^(-m+1))
  (coeff_positive : m^2 - m - 1 > 0)
  (expo_condition : -m + 1 < 0) : 
  m = 2 :=
by
  sorry

end power_function_decreasing_m_eq_2_l96_96965


namespace vector_scalar_operations_l96_96722

-- Define the vectors
def v1 : ℤ × ℤ := (2, -9)
def v2 : ℤ × ℤ := (-1, -6)

-- Define the scalars
def c1 : ℤ := 4
def c2 : ℤ := 3

-- Define the scalar multiplication of vectors
def scale (c : ℤ) (v : ℤ × ℤ) : ℤ × ℤ := (c * v.1, c * v.2)

-- Define the vector subtraction
def sub (v w : ℤ × ℤ) : ℤ × ℤ := (v.1 - w.1, v.2 - w.2)

-- State the theorem
theorem vector_scalar_operations :
  sub (scale c1 v1) (scale c2 v2) = (11, -18) :=
by
  sorry

end vector_scalar_operations_l96_96722


namespace hyperbola_equation_l96_96142

theorem hyperbola_equation
  (a b : ℝ) 
  (a_pos : a > 0) 
  (b_pos : b > 0) 
  (focus_at_five : a^2 + b^2 = 25) 
  (asymptote_ratio : b / a = 3 / 4) :
  (a = 4 ∧ b = 3 ∧ ∀ x y : ℝ, x^2 / 16 - y^2 / 9 = 1) ↔ ( ∀ x y : ℝ, x^2 / 16 - y^2 / 9 = 1 ):=
sorry 

end hyperbola_equation_l96_96142


namespace sum_place_values_of_7s_l96_96575

theorem sum_place_values_of_7s (n : ℝ) (h : n = 87953.0727) : 
  let a := 7000
  let b := 0.07
  let c := 0.0007
  a + b + c = 7000.0707 :=
by
  sorry

end sum_place_values_of_7s_l96_96575


namespace measure_of_angle_B_l96_96519

theorem measure_of_angle_B (A B C a b c : ℝ) (h₁ : a = A.sin) (h₂ : b = B.sin) (h₃ : c = C.sin)
  (h₄ : (b - a) / (c + a) = c / (a + b)) :
  B = 2 * π / 3 :=
by
  sorry

end measure_of_angle_B_l96_96519


namespace junior_girls_count_l96_96131

theorem junior_girls_count 
  (total_players : ℕ) 
  (boys_percentage : ℝ) 
  (junior_girls : ℕ)
  (h_team : total_players = 50)
  (h_boys_pct : boys_percentage = 0.6)
  (h_junior_girls : junior_girls = ((total_players : ℝ) * (1 - boys_percentage) * 0.5)) : 
  junior_girls = 10 := 
by 
  sorry

end junior_girls_count_l96_96131


namespace most_stable_performance_l96_96850

theorem most_stable_performance :
  ∀ (σ2_A σ2_B σ2_C σ2_D : ℝ), 
  σ2_A = 0.56 → 
  σ2_B = 0.78 → 
  σ2_C = 0.42 → 
  σ2_D = 0.63 → 
  σ2_C ≤ σ2_A ∧ σ2_C ≤ σ2_B ∧ σ2_C ≤ σ2_D :=
by
  intros σ2_A σ2_B σ2_C σ2_D hA hB hC hD
  sorry

end most_stable_performance_l96_96850


namespace altered_prism_edges_l96_96054

theorem altered_prism_edges :
  let original_edges := 12
  let vertices := 8
  let edges_per_vertex := 3
  let faces := 6
  let edges_per_face := 1
  let total_edges := original_edges + edges_per_vertex * vertices + edges_per_face * faces
  total_edges = 42 :=
by
  let original_edges := 12
  let vertices := 8
  let edges_per_vertex := 3
  let faces := 6
  let edges_per_face := 1
  let total_edges := original_edges + edges_per_vertex * vertices + edges_per_face * faces
  show total_edges = 42
  sorry

end altered_prism_edges_l96_96054


namespace garden_area_l96_96267

theorem garden_area (w l : ℕ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end garden_area_l96_96267


namespace employees_cycle_l96_96709

theorem employees_cycle (total_employees : ℕ) (drivers_percentage walkers_percentage cyclers_percentage: ℕ) (walk_cycle_ratio_walk walk_cycle_ratio_cycle: ℕ)
    (h_total : total_employees = 500)
    (h_drivers_perc : drivers_percentage = 35)
    (h_transit_perc : walkers_percentage = 25)
    (h_walkers_cyclers_ratio_walk : walk_cycle_ratio_walk = 3)
    (h_walkers_cyclers_ratio_cycle : walk_cycle_ratio_cycle = 7) :
    cyclers_percentage = 140 :=
by
  sorry

end employees_cycle_l96_96709


namespace classify_tangents_through_point_l96_96361

-- Definitions for the Lean theorem statement
noncomputable def curve (x : ℝ) : ℝ :=
  x^3 - x

noncomputable def phi (t x₀ y₀ : ℝ) : ℝ :=
  2*t^3 - 3*x₀*t^2 + (x₀ + y₀)

theorem classify_tangents_through_point (x₀ y₀ : ℝ) :
  (if (x₀ + y₀ < 0 ∨ y₀ > x₀^3 - x₀)
   then 1
   else if (x₀ + y₀ > 0 ∧ x₀ + y₀ - x₀^3 < 0)
   then 3
   else if (x₀ + y₀ = 0 ∨ y₀ = x₀^3 - x₀)
   then 2
   else 0) = 
  (if (x₀ + y₀ < 0 ∨ y₀ > x₀^3 - x₀)
   then 1
   else if (x₀ + y₀ > 0 ∧ x₀ + y₀ - x₀^3 < 0)
   then 3
   else if (x₀ + y₀ = 0 ∨ y₀ = x₀^3 - x₀)
   then 2
   else 0) :=
  sorry

end classify_tangents_through_point_l96_96361


namespace Winnie_the_Pooh_honey_consumption_l96_96370

theorem Winnie_the_Pooh_honey_consumption (W0 W1 W2 W3 W4 : ℝ) (pot_empty : ℝ) 
  (h1 : W1 = W0 / 2)
  (h2 : W2 = W1 / 2)
  (h3 : W3 = W2 / 2)
  (h4 : W4 = W3 / 2)
  (h5 : W4 = 200)
  (h6 : pot_empty = 200) : 
  W0 - 200 = 3000 := by
  sorry

end Winnie_the_Pooh_honey_consumption_l96_96370


namespace find_third_discount_percentage_l96_96565

noncomputable def third_discount_percentage (x : ℝ) : Prop :=
  let item_price := 68
  let num_items := 3
  let first_discount := 0.15
  let second_discount := 0.10
  let total_initial_price := num_items * item_price
  let price_after_first_discount := total_initial_price * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  price_after_second_discount * (1 - x / 100) = 105.32

theorem find_third_discount_percentage : ∃ x : ℝ, third_discount_percentage x ∧ x = 32.5 :=
by
  sorry

end find_third_discount_percentage_l96_96565


namespace find_y_eq_7_5_l96_96763

theorem find_y_eq_7_5 (y : ℝ) (hy1 : 0 < y) (hy2 : ∃ z : ℤ, ((z : ℝ) ≤ y) ∧ (y < z + 1))
  (hy3 : (Int.floor y : ℝ) * y = 45) : y = 7.5 :=
sorry

end find_y_eq_7_5_l96_96763


namespace div_by_9_implies_not_div_by_9_l96_96299

/-- If 9 divides 10^n + 1, then it also divides 10^(n+1) + 1 -/
theorem div_by_9_implies:
  ∀ n: ℕ, (9 ∣ (10^n + 1)) → (9 ∣ (10^(n + 1) + 1)) :=
by
  intro n
  intro h
  sorry

/-- 9 does not divide 10^1 + 1 -/
theorem not_div_by_9:
  ¬(9 ∣ (10^1 + 1)) :=
by 
  sorry

end div_by_9_implies_not_div_by_9_l96_96299


namespace remainder_of_polynomial_division_l96_96597

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 5 * x^3 - 18 * x^2 + 31 * x - 40

-- Define the divisor D(x)
def D (x : ℝ) : ℝ := 5 * x - 10

-- Prove that the remainder when P(x) is divided by D(x) is -10
theorem remainder_of_polynomial_division : (P 2) = -10 := by
  sorry

end remainder_of_polynomial_division_l96_96597


namespace triangle_side_condition_angle_condition_l96_96651

variable (a b c A B C : ℝ)

theorem triangle_side_condition (a_eq : a = 2) (b_eq : b = Real.sqrt 7) (h : a = b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B) :
  c = 3 :=
  sorry

theorem angle_condition (angle_eq : Real.sqrt 3 * Real.sin (2 * A - π / 6) - 2 * Real.sin (C - π / 12)^2 = 0) :
  A = π / 4 :=
  sorry

end triangle_side_condition_angle_condition_l96_96651


namespace another_representation_l96_96414

def positive_int_set : Set ℕ := {x | x > 0}

theorem another_representation :
  {x ∈ positive_int_set | x - 3 < 2} = {1, 2, 3, 4} :=
by
  sorry

end another_representation_l96_96414


namespace silly_bills_count_l96_96403

theorem silly_bills_count (x : ℕ) (h1 : x + 2 * (x + 11) + 3 * (x - 18) = 100) : x = 22 :=
by { sorry }

end silly_bills_count_l96_96403


namespace sum_arithmetic_sequence_l96_96992

noncomputable def is_arithmetic (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∃ a1 : ℚ, ∀ n : ℕ, a n = a1 + n * d

noncomputable def sum_of_first_n_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  n * (a 0 + a (n - 1)) / 2

theorem sum_arithmetic_sequence (a : ℕ → ℚ) (h_arith : is_arithmetic a)
  (h1 : 2 * a 3 = 5) (h2 : a 4 + a 12 = 9) : sum_of_first_n_terms a 10 = 35 :=
by
  -- Proof omitted
  sorry

end sum_arithmetic_sequence_l96_96992


namespace compute_sum_bk_ck_l96_96427

theorem compute_sum_bk_ck 
  (b1 b2 b3 c1 c2 c3 : ℝ)
  (h : ∀ x : ℝ, x^6 - 2*x^5 + 3*x^4 - 3*x^3 + 3*x^2 - 2*x + 1 =
                (x^2 + b1*x + c1) * (x^2 + b2*x + c2) * (x^2 + b3*x + c3)) :
  b1 * c1 + b2 * c2 + b3 * c3 = -2 := 
sorry

end compute_sum_bk_ck_l96_96427


namespace range_of_a_l96_96010

-- Conditions for sets A and B
def SetA := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def SetB (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ a + 2}

-- Main statement to show that A ∪ B = A implies the range of a is [-2, 0]
theorem range_of_a (a : ℝ) : (SetB a ⊆ SetA) → (-2 ≤ a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l96_96010


namespace cookies_left_over_l96_96247

def abigail_cookies : Nat := 53
def beatrice_cookies : Nat := 65
def carson_cookies : Nat := 26
def pack_size : Nat := 10

theorem cookies_left_over : (abigail_cookies + beatrice_cookies + carson_cookies) % pack_size = 4 := 
by
  sorry

end cookies_left_over_l96_96247


namespace inequality_holds_for_all_x_iff_l96_96367

theorem inequality_holds_for_all_x_iff (m : ℝ) :
  (∀ (x : ℝ), m * x^2 + m * x - 4 < 2 * x^2 + 2 * x - 1) ↔ -10 < m ∧ m ≤ 2 :=
by
  sorry

end inequality_holds_for_all_x_iff_l96_96367


namespace solve_abs_eq_2x_plus_1_l96_96392

theorem solve_abs_eq_2x_plus_1 (x : ℝ) (h : |x| = 2 * x + 1) : x = -1 / 3 :=
by 
  sorry

end solve_abs_eq_2x_plus_1_l96_96392


namespace probability_qualified_from_A_is_correct_l96_96406

-- Given conditions:
def p_A : ℝ := 0.7
def pass_A : ℝ := 0.95

-- Define what we need to prove:
def qualified_from_A : ℝ := p_A * pass_A

-- Theorem statement
theorem probability_qualified_from_A_is_correct :
  qualified_from_A = 0.665 :=
by
  sorry

end probability_qualified_from_A_is_correct_l96_96406


namespace proof_problem_l96_96769

noncomputable def problem_expression : ℝ :=
  50 * 39.96 * 3.996 * 500

theorem proof_problem : problem_expression = (3996 : ℝ)^2 :=
by
  sorry

end proof_problem_l96_96769


namespace fraction_calculation_l96_96579

theorem fraction_calculation : ( ( (1/2 : ℚ) + (1/5) ) / ( (3/7) - (1/14) ) * (2/3) ) = 98/75 :=
by
  sorry

end fraction_calculation_l96_96579


namespace common_difference_is_half_l96_96511

variable (a : ℕ → ℚ) (d : ℚ) (a₁ : ℚ) (q p : ℕ)

-- Conditions
def condition1 : Prop := a p = 4
def condition2 : Prop := a q = 2
def condition3 : Prop := p = 4 + q
def arithmetic_sequence : Prop := ∀ n : ℕ, a n = a₁ + (n - 1) * d

-- Proof statement
theorem common_difference_is_half 
  (h1 : condition1 a p)
  (h2 : condition2 a q)
  (h3 : condition3 p q)
  (as : arithmetic_sequence a a₁ d)
  : d = 1 / 2 := 
sorry

end common_difference_is_half_l96_96511


namespace Jessica_cut_roses_l96_96382

theorem Jessica_cut_roses
  (initial_roses : ℕ) (initial_orchids : ℕ)
  (new_roses : ℕ) (new_orchids : ℕ)
  (cut_roses : ℕ) :
  initial_roses = 15 → initial_orchids = 62 →
  new_roses = 17 → new_orchids = 96 →
  new_roses = initial_roses + cut_roses →
  cut_roses = 2 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h3] at h5
  linarith

end Jessica_cut_roses_l96_96382


namespace cosine_sine_inequality_theorem_l96_96990

theorem cosine_sine_inequality_theorem (θ : ℝ) :
  (∀ x : ℝ, 0 ≤ x → x ≤ 1 → 
    x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) ↔
    (π / 12 < θ ∧ θ < 5 * π / 12) :=
by
  sorry

end cosine_sine_inequality_theorem_l96_96990


namespace range_of_m_inequality_system_l96_96421

theorem range_of_m_inequality_system (m : ℝ) :
  (∀ x : ℤ, (-5 < x ∧ x ≤ m + 1) ↔ (x = -4 ∨ x = -3 ∨ x = -2)) →
  -3 ≤ m ∧ m < -2 :=
by
  sorry

end range_of_m_inequality_system_l96_96421


namespace actual_length_of_road_l96_96843

-- Define the conditions
def scale_factor : ℝ := 2500000
def length_on_map : ℝ := 6
def cm_to_km : ℝ := 100000

-- State the theorem
theorem actual_length_of_road : (length_on_map * scale_factor) / cm_to_km = 150 := by
  sorry

end actual_length_of_road_l96_96843


namespace find_a4_l96_96273

theorem find_a4 (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n + 1) = a n - 3) : a 4 = -8 :=
by {
  sorry
}

end find_a4_l96_96273


namespace avg_weight_section_B_l96_96665

theorem avg_weight_section_B 
  (W_B : ℝ) 
  (num_students_A : ℕ := 36) 
  (avg_weight_A : ℝ := 30) 
  (num_students_B : ℕ := 24) 
  (total_students : ℕ := 60) 
  (avg_weight_class : ℝ := 30) 
  (h1 : num_students_A * avg_weight_A + num_students_B * W_B = total_students * avg_weight_class) :
  W_B = 30 :=
sorry

end avg_weight_section_B_l96_96665


namespace system1_solution_system2_solution_l96_96432

-- For Question 1

theorem system1_solution (x y : ℝ) :
  (2 * x - y = 5) ∧ (7 * x - 3 * y = 20) ↔ (x = 5 ∧ y = 5) := 
sorry

-- For Question 2

theorem system2_solution (x y : ℝ) :
  (3 * (x + y) - 4 * (x - y) = 16) ∧ ((x + y)/2 + (x - y)/6 = 1) ↔ (x = 1/3 ∧ y = 7/3) := 
sorry

end system1_solution_system2_solution_l96_96432


namespace new_water_intake_recommendation_l96_96136

noncomputable def current_consumption : ℝ := 25
noncomputable def increase_percentage : ℝ := 0.75
noncomputable def increased_amount : ℝ := increase_percentage * current_consumption
noncomputable def new_recommended_consumption : ℝ := current_consumption + increased_amount

theorem new_water_intake_recommendation :
  new_recommended_consumption = 43.75 := 
by 
  sorry

end new_water_intake_recommendation_l96_96136


namespace rational_root_of_p_l96_96561

noncomputable def p (n : ℕ) (x : ℚ) : ℚ :=
  x^n + (2 + x)^n + (2 - x)^n

theorem rational_root_of_p :
  ∀ n : ℕ, n > 0 → (∃ x : ℚ, p n x = 0) ↔ n = 1 := by
  sorry

end rational_root_of_p_l96_96561


namespace sqrt_sum_fractions_eq_l96_96987

theorem sqrt_sum_fractions_eq :
  (Real.sqrt ((1 / 25) + (1 / 36)) = (Real.sqrt 61) / 30) :=
by
  sorry

end sqrt_sum_fractions_eq_l96_96987


namespace problem_condition_implies_statement_l96_96932

variable {a b c : ℝ}

theorem problem_condition_implies_statement :
  a^3 + a * b + a * c < 0 → b^5 - 4 * a * c > 0 :=
by
  intros h
  sorry

end problem_condition_implies_statement_l96_96932


namespace roots_of_polynomial_fraction_l96_96303

theorem roots_of_polynomial_fraction (a b c : ℝ)
  (h1 : a + b + c = 6)
  (h2 : a * b + a * c + b * c = 11)
  (h3 : a * b * c = 6) :
  a / (b * c + 2) + b / (a * c + 2) + c / (a * b + 2) = 3 / 2 := 
by
  sorry

end roots_of_polynomial_fraction_l96_96303


namespace find_y_in_set_l96_96017

noncomputable def arithmetic_mean (s : List ℝ) : ℝ :=
  s.sum / s.length

theorem find_y_in_set :
  ∀ (y : ℝ), arithmetic_mean [8, 15, 20, 5, y] = 12 ↔ y = 12 :=
by
  intro y
  unfold arithmetic_mean
  simp [List.sum_cons, List.length_cons]
  sorry

end find_y_in_set_l96_96017


namespace number_of_ways_to_take_pieces_l96_96569

theorem number_of_ways_to_take_pieces : 
  (Nat.choose 6 4) = 15 := 
by
  sorry

end number_of_ways_to_take_pieces_l96_96569


namespace sum_first_2018_terms_of_given_sequence_l96_96672

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_first_2018_terms_of_given_sequence :
  let a := 1
  let d := -1 / 2017
  S_2018 = 1009 :=
by
  sorry

end sum_first_2018_terms_of_given_sequence_l96_96672


namespace return_speed_is_33_33_l96_96309

noncomputable def return_speed (d: ℝ) (speed_to_b: ℝ) (avg_speed: ℝ): ℝ :=
  d / (3 + (d / avg_speed))

-- Conditions
def distance := 150
def speed_to_b := 50
def avg_speed := 40

-- Prove that the return speed is 33.33 miles per hour
theorem return_speed_is_33_33:
  return_speed distance speed_to_b avg_speed = 33.33 :=
by
  unfold return_speed
  sorry

end return_speed_is_33_33_l96_96309


namespace parabola_equation_l96_96043

theorem parabola_equation :
  (∃ h k : ℝ, h^2 = 3 ∧ k^2 = 6) →
  (∃ c : ℝ, c^2 = (3 + 6)) →
  (∃ x y : ℝ, x = 3 ∧ y = 0) →
  (y^2 = 12 * x) :=
sorry

end parabola_equation_l96_96043


namespace remainder_x3_minus_4x2_plus_3x_plus_2_div_x_minus_1_l96_96155

def p (x : ℝ) : ℝ := x^3 - 4 * x^2 + 3 * x + 2

theorem remainder_x3_minus_4x2_plus_3x_plus_2_div_x_minus_1 :
  p 1 = 2 := by
  -- solution needed, for now we put a placeholder
  sorry

end remainder_x3_minus_4x2_plus_3x_plus_2_div_x_minus_1_l96_96155


namespace vector_dot_product_l96_96229

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)

-- Prove that the scalar product a · (a - 2b) equals 2
theorem vector_dot_product :
  let u := a
  let v := b
  u • (u - (2 • v)) = 2 :=
by 
  -- Placeholder for the proof
  sorry

end vector_dot_product_l96_96229


namespace number_of_people_who_selected_dog_l96_96790

theorem number_of_people_who_selected_dog 
  (total : ℕ) 
  (cat : ℕ) 
  (fish : ℕ) 
  (bird : ℕ) 
  (other : ℕ) 
  (h_total : total = 90) 
  (h_cat : cat = 25) 
  (h_fish : fish = 10) 
  (h_bird : bird = 15) 
  (h_other : other = 5) :
  (total - (cat + fish + bird + other) = 35) :=
by
  sorry

end number_of_people_who_selected_dog_l96_96790


namespace hexagonal_H5_find_a_find_t_find_m_l96_96268

section problem1

-- Define the hexagonal number formula
def hexagonal_number (n : ℕ) : ℕ :=
  2 * n^2 - n

-- Define that H_5 should equal 45
theorem hexagonal_H5 : hexagonal_number 5 = 45 := sorry

end problem1

section problem2

variables (a b c : ℕ)

-- Given hexagonal number equations
def H1 := a + b + c
def H2 := 4 * a + 2 * b + c
def H3 := 9 * a + 3 * b + c

-- Conditions given in problem
axiom H1_def : H1 = 1
axiom H2_def : H2 = 7
axiom H3_def : H3 = 19

-- Prove that a = 3
theorem find_a : a = 3 := sorry

end problem2

section problem3

variables (p q r t : ℕ)

-- Given ratios in problem
axiom ratio1 : p * 3 = 2 * q
axiom ratio2 : q * 5 = 4 * r

-- Prove that t = 12
theorem find_t : t = 12 := sorry

end problem3

section problem4

variables (x y m : ℕ)

-- Given proportional conditions
axiom ratio3 : x * 3 = y * 4
axiom ratio4 : (x + y) * 3 = x * m

-- Prove that m = 7
theorem find_m : m = 7 := sorry

end problem4

end hexagonal_H5_find_a_find_t_find_m_l96_96268


namespace inequality_solution_l96_96250

-- Define the inequality condition
def fraction_inequality (x : ℝ) : Prop :=
  (3 * x - 1) / (x - 2) ≤ 0

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  1 / 3 ≤ x ∧ x < 2

-- The theorem to prove that the inequality's solution matches the given solution set
theorem inequality_solution (x : ℝ) (h : fraction_inequality x) : solution_set x :=
  sorry

end inequality_solution_l96_96250


namespace fraction_to_decimal_l96_96631

theorem fraction_to_decimal :
  (58 / 200 : ℝ) = 1.16 := by
  sorry

end fraction_to_decimal_l96_96631


namespace cricket_bat_cost_price_l96_96939

theorem cricket_bat_cost_price (CP_A : ℝ) (SP_B : ℝ) (SP_C : ℝ) (h1 : SP_B = CP_A * 1.20) (h2 : SP_C = SP_B * 1.25) (h3 : SP_C = 222) : CP_A = 148 := 
by
  sorry

end cricket_bat_cost_price_l96_96939


namespace num_distinct_five_digit_integers_with_product_of_digits_18_l96_96888

theorem num_distinct_five_digit_integers_with_product_of_digits_18 :
  ∃ (n : ℕ), n = 70 ∧ ∀ (a b c d e : ℕ),
    a * b * c * d * e = 18 ∧ 
    1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9 ∧ 1 ≤ e ∧ e ≤ 9 → 
    (∃ (s : Finset (Fin 100000)), s.card = n) :=
  sorry

end num_distinct_five_digit_integers_with_product_of_digits_18_l96_96888


namespace train_length_l96_96473

theorem train_length
  (t1 : ℕ) (t2 : ℕ)
  (d_platform : ℕ)
  (h1 : t1 = 8)
  (h2 : t2 = 20)
  (h3 : d_platform = 279)
  : ∃ (L : ℕ), (L : ℕ) = 186 :=
by
  sorry

end train_length_l96_96473


namespace tom_should_pay_times_original_price_l96_96167

-- Definitions of the given conditions
def original_price : ℕ := 3
def amount_paid : ℕ := 9

-- The theorem to prove
theorem tom_should_pay_times_original_price : ∃ k : ℕ, amount_paid = k * original_price ∧ k = 3 :=
by 
  -- Using sorry to skip the proof for now
  sorry

end tom_should_pay_times_original_price_l96_96167


namespace june_ride_time_l96_96487

theorem june_ride_time (d1 d2 : ℝ) (t1 : ℝ) (rate : ℝ) (t2 : ℝ) :
  d1 = 2 ∧ t1 = 6 ∧ rate = (d1 / t1) ∧ d2 = 5 ∧ t2 = d2 / rate → t2 = 15 := by
  intros h
  sorry

end june_ride_time_l96_96487


namespace number_of_family_members_l96_96842

noncomputable def total_money : ℝ :=
  123 * 0.01 + 85 * 0.05 + 35 * 0.10 + 26 * 0.25

noncomputable def leftover_money : ℝ := 0.48

noncomputable def double_scoop_cost : ℝ := 3.0

noncomputable def amount_spent : ℝ := total_money - leftover_money

noncomputable def number_of_double_scoops : ℝ := amount_spent / double_scoop_cost

theorem number_of_family_members :
  number_of_double_scoops = 5 := by
  sorry

end number_of_family_members_l96_96842


namespace power_function_not_pass_origin_l96_96727

noncomputable def does_not_pass_through_origin (m : ℝ) : Prop :=
  ∀ x:ℝ, (m^2 - 3 * m + 3) * x^(m^2 - m - 2) ≠ 0

theorem power_function_not_pass_origin (m : ℝ) :
  does_not_pass_through_origin m ↔ (m = 1 ∨ m = 2) :=
sorry

end power_function_not_pass_origin_l96_96727


namespace bike_travel_distance_l96_96956

def avg_speed : ℝ := 3  -- average speed in m/s
def time : ℝ := 7       -- time in seconds

theorem bike_travel_distance : avg_speed * time = 21 := by
  sorry

end bike_travel_distance_l96_96956


namespace farm_field_area_l96_96942

theorem farm_field_area
  (planned_daily_plough : ℕ)
  (actual_daily_plough : ℕ)
  (extra_days : ℕ)
  (remaining_area : ℕ)
  (total_days_hectares : ℕ → ℕ) :
  planned_daily_plough = 260 →
  actual_daily_plough = 85 →
  extra_days = 2 →
  remaining_area = 40 →
  total_days_hectares (total_days_hectares (1 + 2) * 85 + 40) = 312 :=
by
  sorry

end farm_field_area_l96_96942


namespace graph_passes_through_point_l96_96523

theorem graph_passes_through_point (a : ℝ) (x y : ℝ) (h : a < 0) : (1 - a)^0 - 1 = -1 :=
by
  sorry

end graph_passes_through_point_l96_96523


namespace original_amount_of_solution_y_l96_96307

theorem original_amount_of_solution_y (Y : ℝ) 
  (h1 : 0 < Y) -- We assume Y > 0 
  (h2 : 0.3 * (Y - 4) + 1.2 = 0.45 * Y) :
  Y = 8 := 
sorry

end original_amount_of_solution_y_l96_96307


namespace opposite_of_neg_five_halves_l96_96771

theorem opposite_of_neg_five_halves : -(- (5 / 2: ℝ)) = 5 / 2 :=
by
    sorry

end opposite_of_neg_five_halves_l96_96771


namespace scalene_triangle_angles_l96_96952

theorem scalene_triangle_angles (x y z : ℝ) (h1 : x + y + z = 180) (h2 : x ≠ y ∧ y ≠ z ∧ x ≠ z)
(h3 : x = 36 ∨ y = 36 ∨ z = 36) (h4 : x = 2 * y ∨ y = 2 * x ∨ z = 2 * x ∨ x = 2 * z ∨ y = 2 * z ∨ z = 2 * y) :
(x = 36 ∧ y = 48 ∧ z = 96) ∨ (x = 18 ∧ y = 36 ∧ z = 126) ∨ (x = 36 ∧ z = 48 ∧ y = 96) ∨ (y = 18 ∧ x = 36 ∧ z = 126) :=
sorry

end scalene_triangle_angles_l96_96952


namespace intersection_nonempty_iff_l96_96633

/-- Define sets A and B as described in the problem. -/
def A (x : ℝ) : Prop := -2 < x ∧ x ≤ 1
def B (x : ℝ) (k : ℝ) : Prop := x ≥ k

/-- The main theorem to prove the range of k where the intersection of A and B is non-empty. -/
theorem intersection_nonempty_iff (k : ℝ) : (∃ x, A x ∧ B x k) ↔ k ≤ 1 :=
by
  sorry

end intersection_nonempty_iff_l96_96633


namespace find_a_from_expansion_l96_96614

theorem find_a_from_expansion :
  (∃ a : ℝ, (∃ c : ℝ, (∃ d : ℝ, (∃ e : ℝ, (20 - 30 * a + 6 * a^2 = -16 ∧ (a = 2 ∨ a = 3))))))
:= sorry

end find_a_from_expansion_l96_96614


namespace max_area_quadrilateral_cdfg_l96_96854

theorem max_area_quadrilateral_cdfg (s : ℝ) (x : ℝ)
  (h1 : s = 1) (h2 : x > 0) (h3 : x < s) (h4 : AE = x) (h5 : AF = x) : 
  ∃ x, x > 0 ∧ x < 1 ∧ (1 - x) * x ≤ 5 / 8 :=
sorry

end max_area_quadrilateral_cdfg_l96_96854


namespace path_length_of_B_l96_96807

noncomputable def lengthPathB (BC : ℝ) : ℝ :=
  let radius := BC
  let circumference := 2 * Real.pi * radius
  circumference

theorem path_length_of_B (BC : ℝ) (h : BC = 4 / Real.pi) : lengthPathB BC = 8 := by
  rw [lengthPathB, h]
  simp [Real.pi_ne_zero, div_mul_cancel]
  sorry

end path_length_of_B_l96_96807


namespace ages_correct_in_2018_l96_96200

-- Define the initial ages in the year 2000
def age_marianne_2000 : ℕ := 20
def age_bella_2000 : ℕ := 8
def age_carmen_2000 : ℕ := 15

-- Define the birth year of Elli
def birth_year_elli : ℕ := 2003

-- Define the target year when Bella turns 18
def year_bella_turns_18 : ℕ := 2000 + 18

-- Define the ages to be proven
def age_marianne_2018 : ℕ := 30
def age_carmen_2018 : ℕ := 33
def age_elli_2018 : ℕ := 15

theorem ages_correct_in_2018 :
  age_marianne_2018 = age_marianne_2000 + (year_bella_turns_18 - 2000) ∧
  age_carmen_2018 = age_carmen_2000 + (year_bella_turns_18 - 2000) ∧
  age_elli_2018 = year_bella_turns_18 - birth_year_elli :=
by 
  -- The proof would go here
  sorry

end ages_correct_in_2018_l96_96200


namespace prove_a_eq_b_l96_96821

theorem prove_a_eq_b (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h_eq : a^b = b^a) (h_a_lt_1 : a < 1) : a = b :=
by
  sorry

end prove_a_eq_b_l96_96821


namespace arithmetic_sequence_sum_l96_96886

theorem arithmetic_sequence_sum :
  ∃ a b : ℕ, ∀ d : ℕ,
    d = 5 →
    a = 28 →
    b = 33 →
    a + b = 61 :=
by
  sorry

end arithmetic_sequence_sum_l96_96886


namespace find_n_l96_96889

noncomputable def factorial : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * factorial n

theorem find_n (n : ℕ) (h : n * factorial (n + 1) + factorial (n + 1) = 5040) : n = 5 :=
sorry

end find_n_l96_96889


namespace problem_divisibility_l96_96169

theorem problem_divisibility (n : ℕ) : ∃ k : ℕ, 2 ^ (3 ^ n) + 1 = 3 ^ (n + 1) * k :=
sorry

end problem_divisibility_l96_96169


namespace k_value_of_polynomial_square_l96_96638

theorem k_value_of_polynomial_square (k : ℤ) :
  (∃ (f : ℤ → ℤ), ∀ x, f x = x^2 + 6 * x + k^2) → (k = 3 ∨ k = -3) :=
by
  sorry

end k_value_of_polynomial_square_l96_96638


namespace find_purchase_price_l96_96537

noncomputable def purchase_price (total_paid : ℝ) (interest_percent : ℝ) : ℝ :=
    total_paid / (1 + interest_percent)

theorem find_purchase_price :
  purchase_price 130 0.09090909090909092 = 119.09 :=
by
  -- Normally we would provide the full proof here, but it is omitted as per instructions
  sorry

end find_purchase_price_l96_96537


namespace solve_for_M_l96_96119

def M : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ 2 * x + y = 2 ∧ x - y = 1 }

theorem solve_for_M : M = { (1, 0) } := by
  sorry

end solve_for_M_l96_96119


namespace units_digit_of_3_pow_1987_l96_96979

theorem units_digit_of_3_pow_1987 : 3 ^ 1987 % 10 = 7 := by
  sorry

end units_digit_of_3_pow_1987_l96_96979


namespace percentage_increase_in_sales_l96_96810

theorem percentage_increase_in_sales (P S : ℝ) (hP : P > 0) (hS : S > 0) :
  (∃ X : ℝ, (0.8 * (1 + X / 100) = 1.44) ∧ X = 80) :=
sorry

end percentage_increase_in_sales_l96_96810


namespace opposite_of_one_over_2023_l96_96187

def one_over_2023 : ℚ := 1 / 2023

theorem opposite_of_one_over_2023 : -one_over_2023 = -1 / 2023 :=
by
  sorry

end opposite_of_one_over_2023_l96_96187


namespace motorcycle_licenses_count_l96_96005

theorem motorcycle_licenses_count : (3 * (10 ^ 6) = 3000000) :=
by
  sorry -- Proof would go here.

end motorcycle_licenses_count_l96_96005


namespace hunting_dog_catches_fox_l96_96529

theorem hunting_dog_catches_fox :
  ∀ (V_1 V_2 : ℝ) (t : ℝ),
  V_1 / V_2 = 10 ∧ 
  t * V_2 = (10 / (V_2) + t) →
  (V_1 * t) = 100 / 9 :=
by
  intros V_1 V_2 t h
  sorry

end hunting_dog_catches_fox_l96_96529


namespace simplify_and_evaluate_l96_96280

noncomputable def simplifyExpression (a : ℚ) : ℚ :=
  (a - 3 + (1 / (a - 1))) / ((a^2 - 4) / (a^2 + 2*a)) * (1 / (a - 2))

theorem simplify_and_evaluate
  (h : ∀ a, a ∈ [-2, -1, 0, 1, 2]) :
  ∀ a, (a - 1) ≠ 0 → a ≠ 0 → a ≠ 2  →
  simplifyExpression a = a / (a - 1) ∧ simplifyExpression (-1) = 1 / 2 :=
by
  intro a ha_ne_zero ha_ne_two
  sorry

end simplify_and_evaluate_l96_96280


namespace quadratic_no_real_roots_l96_96068

theorem quadratic_no_real_roots (a b : ℝ) (h : ∃ x : ℝ, x^2 + b * x + a = 0) : false :=
sorry

end quadratic_no_real_roots_l96_96068


namespace length_of_AB_in_triangle_l96_96550

open Real

theorem length_of_AB_in_triangle
  (AC BC : ℝ)
  (area : ℝ) :
  AC = 4 →
  BC = 3 →
  area = 3 * sqrt 3 →
  ∃ AB : ℝ, AB = sqrt 13 :=
by
  sorry

end length_of_AB_in_triangle_l96_96550


namespace find_x_l96_96372

theorem find_x (x : ℝ) (h : 3 * x + 15 = (1 / 3) * (7 * x + 42)) : x = -3 / 2 :=
sorry

end find_x_l96_96372


namespace find_value_of_expression_l96_96056

noncomputable def roots_g : Set ℂ := { x | x^2 - 3*x - 2 = 0 }

theorem find_value_of_expression:
  ∀ γ δ : ℂ, γ ∈ roots_g → δ ∈ roots_g →
  (γ + δ = 3) → (7 * γ^4 + 10 * δ^3 = 1363) :=
by
  intros γ δ hγ hδ hsum
  -- Proof skipped
  sorry

end find_value_of_expression_l96_96056


namespace path_traveled_is_correct_l96_96677

-- Define the original triangle and the circle.
def side_a : ℝ := 8
def side_b : ℝ := 10
def side_c : ℝ := 12.5
def radius : ℝ := 1.5

-- Define the condition that the circle is rolling inside the triangle.
def new_side (original_side : ℝ) (r : ℝ) : ℝ := original_side - 2 * r

-- Calculate the new sides of the smaller triangle path.
def new_side_a := new_side side_a radius
def new_side_b := new_side side_b radius
def new_side_c := new_side side_c radius

-- Calculate the perimeter of the path traced by the circle's center.
def path_perimeter := new_side_a + new_side_b + new_side_c

-- Prove that this perimeter equals 21.5 units under given conditions.
theorem path_traveled_is_correct : path_perimeter = 21.5 := by
  simp [new_side, new_side_a, new_side_b, new_side_c, path_perimeter]
  sorry

end path_traveled_is_correct_l96_96677


namespace how_many_eyes_do_I_see_l96_96374

def boys : ℕ := 23
def eyes_per_boy : ℕ := 2
def total_eyes : ℕ := boys * eyes_per_boy

theorem how_many_eyes_do_I_see : total_eyes = 46 := by
  sorry

end how_many_eyes_do_I_see_l96_96374


namespace multiplication_result_l96_96798

theorem multiplication_result :
  10 * 9.99 * 0.999 * 100 = (99.9)^2 := 
by
  sorry

end multiplication_result_l96_96798


namespace percentage_deducted_from_list_price_l96_96162

-- Definitions based on conditions
def cost_price : ℝ := 85.5
def marked_price : ℝ := 112.5
def profit_rate : ℝ := 0.25 -- 25% profit

noncomputable def selling_price : ℝ := cost_price * (1 + profit_rate)

theorem percentage_deducted_from_list_price:
  ∃ d : ℝ, d = 5 ∧ selling_price = marked_price * (1 - d / 100) :=
by
  sorry

end percentage_deducted_from_list_price_l96_96162


namespace bills_equal_at_80_minutes_l96_96018

variable (m : ℝ)

def C_U : ℝ := 8 + 0.25 * m
def C_A : ℝ := 12 + 0.20 * m

theorem bills_equal_at_80_minutes (h : C_U m = C_A m) : m = 80 :=
by {
  sorry
}

end bills_equal_at_80_minutes_l96_96018


namespace prime_sum_diff_condition_unique_l96_96425

-- Definitions and conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n

def can_be_written_as_sum_of_two_primes (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime (p + q)

def can_be_written_as_difference_of_two_primes (p r : ℕ) : Prop :=
  is_prime p ∧ is_prime r ∧ is_prime (p - r)

-- Question rewritten as Lean statement
theorem prime_sum_diff_condition_unique (p q r : ℕ) :
  is_prime p →
  can_be_written_as_sum_of_two_primes (p - 2) p →
  can_be_written_as_difference_of_two_primes (p + 2) p →
  p = 5 :=
sorry

end prime_sum_diff_condition_unique_l96_96425


namespace interest_rate_l96_96893

theorem interest_rate (SI P : ℝ) (T : ℕ) (h₁: SI = 70) (h₂ : P = 700) (h₃ : T = 4) : 
  (SI / (P * T)) * 100 = 2.5 :=
by
  sorry

end interest_rate_l96_96893


namespace percentage_increase_l96_96388

theorem percentage_increase (S P : ℝ) (h1 : (S * (1 + P / 100)) * 0.8 = 1.04 * S) : P = 30 :=
by 
  sorry

end percentage_increase_l96_96388


namespace line_circle_intersect_l96_96498

theorem line_circle_intersect (m : ℤ) :
  (∃ x y : ℝ, 4 * x + 3 * y + 2 * m = 0 ∧ (x + 3)^2 + (y - 1)^2 = 1) ↔ 2 < m ∧ m < 7 :=
by
  sorry

end line_circle_intersect_l96_96498


namespace time_to_cross_pole_l96_96567

-- Conditions
def train_speed_kmh : ℕ := 108
def train_length_m : ℕ := 210

-- Conversion functions
def km_per_hr_to_m_per_sec (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 1000 / 3600

-- Theorem to be proved
theorem time_to_cross_pole : (train_length_m : ℕ) / (km_per_hr_to_m_per_sec train_speed_kmh) = 7 := by
  -- we'll use sorry here to skip the actual proof steps.
  sorry

end time_to_cross_pole_l96_96567


namespace mark_total_cents_l96_96152

theorem mark_total_cents (dimes nickels : ℕ) (h1 : nickels = dimes + 3) (h2 : dimes = 5) : 
  dimes * 10 + nickels * 5 = 90 := by
  sorry

end mark_total_cents_l96_96152


namespace speed_second_hour_l96_96094

noncomputable def speed_in_first_hour : ℝ := 95
noncomputable def average_speed : ℝ := 77.5
noncomputable def total_time : ℝ := 2
def speed_in_second_hour : ℝ := sorry -- to be deduced

theorem speed_second_hour :
  speed_in_second_hour = 60 :=
by
  sorry

end speed_second_hour_l96_96094


namespace remainder_modulo_l96_96255

theorem remainder_modulo (N k q r : ℤ) (h1 : N = 1423 * k + 215) (h2 : N = 109 * q + r) : 
  (N - q ^ 2) % 109 = 106 := by
  sorry

end remainder_modulo_l96_96255


namespace dividend_rate_correct_l96_96160

-- Define the stock's yield and market value
def stock_yield : ℝ := 0.08
def market_value : ℝ := 175

-- Dividend rate definition based on given yield and market value
def dividend_rate (yield market_value : ℝ) : ℝ :=
  (yield * market_value)

-- The problem statement to be proven in Lean
theorem dividend_rate_correct :
  dividend_rate stock_yield market_value = 14 := by
  sorry

end dividend_rate_correct_l96_96160


namespace maria_candy_remaining_l96_96459

theorem maria_candy_remaining :
  let c := 520.75
  let e := c / 2
  let g := 234.56
  let r := e - g
  r = 25.815 := by
  sorry

end maria_candy_remaining_l96_96459


namespace arithmetic_sequence_sum_l96_96001

variable {S : ℕ → ℕ}

theorem arithmetic_sequence_sum (h1 : S 3 = 15) (h2 : S 9 = 153) : S 6 = 66 :=
sorry

end arithmetic_sequence_sum_l96_96001


namespace cookie_cost_per_day_l96_96881

theorem cookie_cost_per_day
    (days_in_April : ℕ)
    (cookies_per_day : ℕ)
    (total_spent : ℕ)
    (total_cookies : ℕ := days_in_April * cookies_per_day)
    (cost_per_cookie : ℕ := total_spent / total_cookies) :
  days_in_April = 30 ∧ cookies_per_day = 3 ∧ total_spent = 1620 → cost_per_cookie = 18 :=
by
  sorry

end cookie_cost_per_day_l96_96881


namespace approximation_example1_approximation_example2_approximation_example3_l96_96103

theorem approximation_example1 (α β : ℝ) (hα : α = 0.0023) (hβ : β = 0.0057) :
  (1 + α) * (1 + β) = 1.008 := sorry

theorem approximation_example2 (α β : ℝ) (hα : α = 0.05) (hβ : β = -0.03) :
  (1 + α) * (10 + β) = 10.02 := sorry

theorem approximation_example3 (α β γ : ℝ) (hα : α = 0.03) (hβ : β = -0.01) (hγ : γ = -0.02) :
  (1 + α) * (1 + β) * (1 + γ) = 1 := sorry

end approximation_example1_approximation_example2_approximation_example3_l96_96103


namespace units_digit_of_quotient_l96_96671

theorem units_digit_of_quotient : 
  let n := 1993
  let term1 := 4 ^ n
  let term2 := 6 ^ n
  (term1 + term2) % 5 = 0 →
  let quotient := (term1 + term2) / 5
  (quotient % 10 = 0) := 
by 
  sorry

end units_digit_of_quotient_l96_96671


namespace difference_of_numbers_l96_96446

theorem difference_of_numbers 
  (a b : ℕ) 
  (h1 : a + b = 23976)
  (h2 : b % 8 = 0)
  (h3 : a = 7 * b / 8) : 
  b - a = 1598 :=
sorry

end difference_of_numbers_l96_96446


namespace percentage_of_girls_l96_96837

theorem percentage_of_girls (B G : ℕ) (h1 : B + G = 400) (h2 : B = 80) :
  (G * 100) / (B + G) = 80 :=
by sorry

end percentage_of_girls_l96_96837


namespace field_trip_buses_l96_96115

-- Definitions of conditions
def fifth_graders : ℕ := 109
def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def teachers_per_grade : ℕ := 4
def parents_per_grade : ℕ := 2
def grades : ℕ := 3
def seats_per_bus : ℕ := 72

-- Total calculations
def total_students : ℕ := fifth_graders + sixth_graders + seventh_graders
def chaperones_per_grade : ℕ := teachers_per_grade + parents_per_grade
def total_chaperones : ℕ := chaperones_per_grade * grades
def total_people : ℕ := total_students + total_chaperones
def buses_needed : ℕ := (total_people + seats_per_bus - 1) / seats_per_bus

theorem field_trip_buses : buses_needed = 6 := by
  unfold buses_needed
  unfold total_people total_students total_chaperones chaperones_per_grade
  norm_num
  sorry

end field_trip_buses_l96_96115


namespace card_statements_true_l96_96133

def statement1 (statements : Fin 5 → Prop) : Prop :=
  ∃! i, i < 5 ∧ statements i

def statement2 (statements : Fin 5 → Prop) : Prop :=
  (∃ i j, i < 5 ∧ j < 5 ∧ i ≠ j ∧ statements i ∧ statements j) ∧ ¬(∃ h k l, h < 5 ∧ k < 5 ∧ l < 5 ∧ h ≠ k ∧ h ≠ l ∧ k ≠ l ∧ statements h ∧ statements k ∧ statements l)

def statement3 (statements : Fin 5 → Prop) : Prop :=
  (∃ i j k, i < 5 ∧ j < 5 ∧ k < 5 ∧ i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ statements i ∧ statements j ∧ statements k) ∧ ¬(∃ a b c d, a < 5 ∧ b < 5 ∧ c < 5 ∧ d < 5 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ statements a ∧ statements b ∧ statements c ∧ statements d)

def statement4 (statements : Fin 5 → Prop) : Prop :=
  (∃ i j k l, i < 5 ∧ j < 5 ∧ k < 5 ∧ l < 5 ∧ i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ statements i ∧ statements j ∧ statements k ∧ statements l) ∧ ¬(∃ a b c d e, a < 5 ∧ b < 5 ∧ c < 5 ∧ d < 5 ∧ e < 5 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ statements a ∧ statements b ∧ statements c ∧ statements d ∧ statements e)

def statement5 (statements : Fin 5 → Prop) : Prop :=
  ∀ i, i < 5 → statements i

theorem card_statements_true : ∃ (statements : Fin 5 → Prop), 
  statement1 statements ∨ statement2 statements ∨ statement3 statements ∨ statement4 statements ∨ statement5 statements 
  ∧ statement3 statements := 
sorry

end card_statements_true_l96_96133


namespace constant_term_of_expansion_l96_96801

open BigOperators

noncomputable def binomialCoeff (n k : ℕ) : ℕ := Nat.choose n k

theorem constant_term_of_expansion :
  ∑ r in Finset.range (6 + 1), binomialCoeff 6 r * (2^r * (x : ℚ)^r) / (x^3 : ℚ) = 160 :=
by
  sorry

end constant_term_of_expansion_l96_96801


namespace total_complaints_l96_96311

-- Conditions as Lean definitions
def normal_complaints : ℕ := 120
def short_staffed_20 (c : ℕ) := c + c / 3
def short_staffed_40 (c : ℕ) := c + 2 * c / 3
def self_checkout_partial (c : ℕ) := c + c / 10
def self_checkout_complete (c : ℕ) := c + c / 5
def day1_complaints : ℕ := normal_complaints + normal_complaints / 3 + normal_complaints / 5
def day2_complaints : ℕ := normal_complaints + 2 * normal_complaints / 3 + normal_complaints / 10
def day3_complaints : ℕ := normal_complaints + 2 * normal_complaints / 3 + normal_complaints / 5

-- Prove the total complaints
theorem total_complaints : day1_complaints + day2_complaints + day3_complaints = 620 :=
by
  sorry

end total_complaints_l96_96311


namespace everett_weeks_worked_l96_96583

theorem everett_weeks_worked (daily_hours : ℕ) (total_hours : ℕ) (days_in_week : ℕ) 
  (h1 : daily_hours = 5) (h2 : total_hours = 140) (h3 : days_in_week = 7) : 
  (total_hours / (daily_hours * days_in_week) = 4) :=
by
  sorry

end everett_weeks_worked_l96_96583


namespace problem_statement_l96_96969

noncomputable def f (a x : ℝ) := a * (x ^ 2 + 1) + Real.log x

theorem problem_statement (a m : ℝ) (x : ℝ) 
  (h_a : -4 < a) (h_a' : a < -2) (h_x1 : 1 ≤ x) (h_x2 : x ≤ 3) :
  (m * a - f a x > a ^ 2) ↔ (m ≤ -2) :=
by
  sorry

end problem_statement_l96_96969


namespace integral_exp_neg_l96_96176

theorem integral_exp_neg : ∫ x in (Set.Ioi 0), Real.exp (-x) = 1 := sorry

end integral_exp_neg_l96_96176


namespace crayon_count_l96_96762

theorem crayon_count (initial_crayons eaten_crayons : ℕ) (h1 : initial_crayons = 62) (h2 : eaten_crayons = 52) : initial_crayons - eaten_crayons = 10 := 
by 
  sorry

end crayon_count_l96_96762


namespace equilibrium_stability_l96_96936

noncomputable def f (x : ℝ) : ℝ := x * (Real.exp x - 2)

theorem equilibrium_stability (x : ℝ) :
  (x = 0 → HasDerivAt f (-1) 0 ∧ (-1 < 0)) ∧
  (x = Real.log 2 → HasDerivAt f (2 * Real.log 2) (Real.log 2) ∧ (2 * Real.log 2 > 0)) :=
by
  sorry

end equilibrium_stability_l96_96936


namespace smallest_x_l96_96545

theorem smallest_x (M x : ℕ) (h : 720 * x = M^3) : x = 300 :=
by
  sorry

end smallest_x_l96_96545


namespace probability_black_or_white_l96_96641

-- Defining the probabilities of drawing red and white balls
def prob_red : ℝ := 0.45
def prob_white : ℝ := 0.25

-- Defining the total probability
def total_prob : ℝ := 1.0

-- Define the probability of drawing a black or white ball
def prob_black_or_white : ℝ := total_prob - prob_red

-- The theorem stating the required proof
theorem probability_black_or_white : 
  prob_black_or_white = 0.55 := by
    sorry

end probability_black_or_white_l96_96641


namespace value_of_a_l96_96341

theorem value_of_a (a b c : ℂ) (h_real : a.im = 0)
  (h1 : a + b + c = 5) 
  (h2 : a * b + b * c + c * a = 7) 
  (h3 : a * b * c = 2) : a = 2 := by
  sorry

end value_of_a_l96_96341


namespace new_average_after_increase_and_bonus_l96_96058

theorem new_average_after_increase_and_bonus 
  (n : ℕ) (initial_avg : ℝ) (k : ℝ) (bonus : ℝ) 
  (h1: n = 37) 
  (h2: initial_avg = 73) 
  (h3: k = 1.65) 
  (h4: bonus = 15) 
  : (initial_avg * k) + bonus = 135.45 := 
sorry

end new_average_after_increase_and_bonus_l96_96058


namespace smallest_r_for_B_in_C_l96_96213

def A : Set ℝ := {t | 0 < t ∧ t < 2 * Real.pi}

def B : Set (ℝ × ℝ) := 
  {p | ∃ t ∈ A, p.1 = Real.sin t ∧ p.2 = 2 * Real.sin t * Real.cos t}

def C (r : ℝ) : Set (ℝ × ℝ) := 
  {p | p.1^2 + p.2^2 ≤ r^2 ∧ r > 0}

theorem smallest_r_for_B_in_C : ∃ r, (B ⊆ C r ∧ ∀ r', r' < r → ¬ (B ⊆ C r')) :=
  sorry

end smallest_r_for_B_in_C_l96_96213


namespace no_sol_n4_minus_m4_eq_42_l96_96319

theorem no_sol_n4_minus_m4_eq_42 :
  ¬ ∃ (n m : ℕ), 0 < n ∧ 0 < m ∧ n^4 - m^4 = 42 :=
by
  sorry

end no_sol_n4_minus_m4_eq_42_l96_96319


namespace triangle_areas_l96_96581

-- Define points based on the conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Triangle DEF vertices
def D : Point := { x := 0, y := 4 }
def E : Point := { x := 6, y := 0 }
def F : Point := { x := 6, y := 5 }

-- Triangle GHI vertices
def G : Point := { x := 0, y := 8 }
def H : Point := { x := 0, y := 6 }
def I : Point := F  -- I and F are the same point

-- Auxiliary function to calculate area of a triangle given its vertices
def area (A B C : Point) : ℝ :=
  0.5 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

-- Prove that the areas are correct
theorem triangle_areas :
  area D E F = 15 ∧ area G H I = 6 :=
by
  sorry

end triangle_areas_l96_96581


namespace series_sum_equality_l96_96217

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, 12^k / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))

theorem series_sum_equality : sum_series = 1 := 
by sorry

end series_sum_equality_l96_96217


namespace trains_meeting_distance_l96_96674

theorem trains_meeting_distance :
  ∃ D T : ℕ, (D = 20 * T) ∧ (D + 60 = 25 * T) ∧ (2 * D + 60 = 540) :=
by
  sorry

end trains_meeting_distance_l96_96674


namespace trig_identity_l96_96587

variable (α : ℝ)

theorem trig_identity (h : Real.sin (α - 70 * Real.pi / 180) = α) : 
  Real.cos (α + 20 * Real.pi / 180) = -α := by
  sorry

end trig_identity_l96_96587


namespace polygons_intersection_l96_96080

/-- In a square with an area of 5, nine polygons, each with an area of 1, are placed. 
    Prove that some two of them must have an intersection area of at least 1 / 9. -/
theorem polygons_intersection 
  (S : ℝ) (hS : S = 5)
  (n : ℕ) (hn : n = 9)
  (polygons : Fin n → ℝ) (hpolys : ∀ i, polygons i = 1)
  (intersection : Fin n → Fin n → ℝ) : 
  ∃ i j : Fin n, i ≠ j ∧ intersection i j ≥ 1 / 9 := 
sorry

end polygons_intersection_l96_96080


namespace water_on_wednesday_l96_96051

-- Define the total water intake for the week.
def total_water : ℕ := 60

-- Define the water intake amounts for specific days.
def water_on_mon_thu_sat : ℕ := 9
def water_on_tue_fri_sun : ℕ := 8

-- Define the number of days for each intake.
def days_mon_thu_sat : ℕ := 3
def days_tue_fri_sun : ℕ := 3

-- Define the water intake calculated for specific groups of days.
def total_water_mon_thu_sat := water_on_mon_thu_sat * days_mon_thu_sat
def total_water_tue_fri_sun := water_on_tue_fri_sun * days_tue_fri_sun

-- Define the total water intake for these days combined.
def total_water_other_days := total_water_mon_thu_sat + total_water_tue_fri_sun

-- Define the water intake for Wednesday, which we need to prove to be 9 liters.
theorem water_on_wednesday : total_water - total_water_other_days = 9 := by
  -- Proof omitted.
  sorry

end water_on_wednesday_l96_96051


namespace none_of_the_above_option_l96_96700

-- Define integers m and n
variables (m n: ℕ)

-- Define P and R in terms of m and n
def P : ℕ := 2^m
def R : ℕ := 5^n

-- Define the statement to prove
theorem none_of_the_above_option : ∀ (m n : ℕ), 15^(m + n) ≠ P^(m + n) * R ∧ 15^(m + n) ≠ (3^m * 3^n * 5^m) ∧ 15^(m + n) ≠ (3^m * P^n) ∧ 15^(m + n) ≠ (2^m * 5^n * 5^m) :=
by sorry

end none_of_the_above_option_l96_96700


namespace melting_point_of_ice_in_Celsius_l96_96241

theorem melting_point_of_ice_in_Celsius :
  ∀ (boiling_point_F boiling_point_C melting_point_F temperature_C temperature_F : ℤ),
    (boiling_point_F = 212) →
    (boiling_point_C = 100) →
    (melting_point_F = 32) →
    (temperature_C = 60) →
    (temperature_F = 140) →
    (5 * melting_point_F = 9 * 0 + 160) →         -- Using the given equation F = (9/5)C + 32 and C = 0
    melting_point_F = 32 ∧ 0 = 0 :=
by
  intros
  sorry

end melting_point_of_ice_in_Celsius_l96_96241


namespace percentage_yield_l96_96978

theorem percentage_yield (market_price annual_dividend : ℝ) (yield : ℝ) 
  (H1 : yield = 0.12)
  (H2 : market_price = 125)
  (H3 : annual_dividend = yield * market_price) :
  (annual_dividend / market_price) * 100 = 12 := 
sorry

end percentage_yield_l96_96978


namespace boxes_needed_l96_96906

def num_red_pencils := 45
def num_yellow_pencils := 80
def num_pencils_per_red_box := 15
def num_pencils_per_blue_box := 25
def num_pencils_per_yellow_box := 10
def num_pencils_per_green_box := 30

def num_blue_pencils (x : Nat) := 3 * x + 6
def num_green_pencils (red : Nat) (blue : Nat) := 2 * (red + blue)

def total_boxes_needed : Nat :=
  let red_boxes := num_red_pencils / num_pencils_per_red_box
  let blue_boxes := (num_blue_pencils num_red_pencils) / num_pencils_per_blue_box + 
                    if ((num_blue_pencils num_red_pencils) % num_pencils_per_blue_box) = 0 then 0 else 1
  let yellow_boxes := num_yellow_pencils / num_pencils_per_yellow_box
  let green_boxes := (num_green_pencils num_red_pencils (num_blue_pencils num_red_pencils)) / num_pencils_per_green_box + 
                     if ((num_green_pencils num_red_pencils (num_blue_pencils num_red_pencils)) % num_pencils_per_green_box) = 0 then 0 else 1
  red_boxes + blue_boxes + yellow_boxes + green_boxes

theorem boxes_needed : total_boxes_needed = 30 := sorry

end boxes_needed_l96_96906


namespace simplify_expression_l96_96044

theorem simplify_expression : (1 / (1 / ((1 / 3) ^ 1) + 1 / ((1 / 3) ^ 2) + 1 / ((1 / 3) ^ 3))) = (1 / 39) :=
by
  sorry

end simplify_expression_l96_96044


namespace sum_first_n_odd_eq_n_squared_l96_96101

theorem sum_first_n_odd_eq_n_squared (n : ℕ) : (Finset.sum (Finset.range n) (fun k => (2 * k + 1)) = n^2) := sorry

end sum_first_n_odd_eq_n_squared_l96_96101


namespace probability_of_red_ball_l96_96505

theorem probability_of_red_ball (total_balls red_balls black_balls white_balls : ℕ)
  (h1 : total_balls = 7)
  (h2 : red_balls = 2)
  (h3 : black_balls = 4)
  (h4 : white_balls = 1) :
  (red_balls / total_balls : ℚ) = 2 / 7 :=
by {
  sorry
}

end probability_of_red_ball_l96_96505


namespace total_shaded_area_l96_96895

theorem total_shaded_area 
  (side': ℝ) (d: ℝ) (s: ℝ)
  (h1: 12 / d = 4)
  (h2: d / s = 4) : 
  d = 3 →
  s = 3 / 4 →
  (π * (d / 2) ^ 2 + 8 * s ^ 2) = 9 * π / 4 + 9 / 2 :=
by
  intro h3 h4
  have h5 : d = 3 := h3
  have h6 : s = 3 / 4 := h4
  rw [h5, h6]
  sorry

end total_shaded_area_l96_96895


namespace gcd_1978_2017_l96_96332

theorem gcd_1978_2017 : Int.gcd 1978 2017 = 1 :=
sorry

end gcd_1978_2017_l96_96332


namespace first_woman_hours_l96_96394

-- Definitions and conditions
variables (W k y t η : ℝ)
variables (work_rate : k * y * 45 = W)
variables (total_work : W = k * (t * ((y-1) * y) / 2 + y * η))
variables (first_vs_last : (y-1) * t + η = 5 * η)

-- The goal to prove
theorem first_woman_hours :
  (y - 1) * t + η = 75 := 
by
  sorry

end first_woman_hours_l96_96394


namespace cone_height_l96_96354

theorem cone_height (r_sphere : ℝ) (r_cone : ℝ) (waste_percentage : ℝ) 
  (h : ℝ) (V_sphere : ℝ) (V_cone : ℝ) : 
  r_sphere = 9 → r_cone = 9 → waste_percentage = 0.75 → 
  V_sphere = (4 / 3) * Real.pi * r_sphere^3 → 
  V_cone = (1 / 3) * Real.pi * r_cone^2 * h → 
  V_cone = waste_percentage * V_sphere → 
  h = 27 :=
by
  intros r_sphere_eq r_cone_eq waste_eq V_sphere_eq V_cone_eq V_cone_waste_eq
  sorry

end cone_height_l96_96354


namespace money_left_is_correct_l96_96042

noncomputable def total_income : ℝ := 800000
noncomputable def children_pct : ℝ := 0.2
noncomputable def num_children : ℝ := 3
noncomputable def wife_pct : ℝ := 0.3
noncomputable def donation_pct : ℝ := 0.05

noncomputable def remaining_income_after_donations : ℝ := 
  let distributed_to_children := total_income * children_pct * num_children
  let distributed_to_wife := total_income * wife_pct
  let total_distributed := distributed_to_children + distributed_to_wife
  let remaining_after_family := total_income - total_distributed
  let donation := remaining_after_family * donation_pct
  remaining_after_family - donation

theorem money_left_is_correct :
  remaining_income_after_donations = 76000 := 
by 
  sorry

end money_left_is_correct_l96_96042


namespace hyperbola_eccentricity_l96_96777

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (a^2) / (b^2))

theorem hyperbola_eccentricity {b : ℝ} (hb_pos : b > 0)
  (h_area : b = 1) :
  eccentricity 1 b = Real.sqrt 3 :=
by
  sorry

end hyperbola_eccentricity_l96_96777


namespace sin_1200_eq_sqrt3_div_2_l96_96188

theorem sin_1200_eq_sqrt3_div_2 : Real.sin (1200 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end sin_1200_eq_sqrt3_div_2_l96_96188


namespace prove_tan_570_eq_sqrt_3_over_3_l96_96386

noncomputable def tan_570_eq_sqrt_3_over_3 : Prop :=
  Real.tan (570 * Real.pi / 180) = Real.sqrt 3 / 3

theorem prove_tan_570_eq_sqrt_3_over_3 : tan_570_eq_sqrt_3_over_3 :=
by
  sorry

end prove_tan_570_eq_sqrt_3_over_3_l96_96386


namespace power_mod_remainder_l96_96310

theorem power_mod_remainder (a b c : ℕ) (h1 : 7^40 % 500 = 1) (h2 : 7^4 % 40 = 1) : (7^(7^25) % 500 = 43) :=
sorry

end power_mod_remainder_l96_96310


namespace sum_of_coefficients_shifted_function_l96_96595

def original_function (x : ℝ) : ℝ :=
  3*x^2 - 2*x + 6

def shifted_function (x : ℝ) : ℝ :=
  original_function (x + 5)

theorem sum_of_coefficients_shifted_function : 
  let a := 3
  let b := 28
  let c := 71
  a + b + c = 102 :=
by
  -- Placeholder for the proof
  sorry

end sum_of_coefficients_shifted_function_l96_96595


namespace odd_expression_is_odd_l96_96275

theorem odd_expression_is_odd (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : (4 * p * q + 1) % 2 = 1 :=
sorry

end odd_expression_is_odd_l96_96275


namespace sum_mod_20_l96_96926

theorem sum_mod_20 : 
  (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92 + 93 + 94) % 20 = 15 :=
by 
  -- The proof goes here
  sorry

end sum_mod_20_l96_96926


namespace value_to_add_l96_96803

theorem value_to_add (a b c n m : ℕ) (h₁ : a = 510) (h₂ : b = 4590) (h₃ : c = 105) (h₄ : n = 627) (h₅ : m = Nat.lcm a (Nat.lcm b c)) :
  m - n = 31503 :=
by
  sorry

end value_to_add_l96_96803


namespace simplify_expression_l96_96642

variable (b c d x y : ℝ)

theorem simplify_expression :
  (cx * (b^2 * x^3 + 3 * b^2 * y^3 + c^3 * y^3) + dy * (b^2 * x^3 + 3 * c^3 * x^3 + c^3 * y^3)) / (cx + dy) 
  = b^2 * x^3 + 3 * c^2 * xy^3 + c^3 * y^3 :=
by sorry

end simplify_expression_l96_96642


namespace either_x_or_y_is_even_l96_96346

theorem either_x_or_y_is_even (x y z : ℤ) (h : x^2 + y^2 = z^2) : (2 ∣ x) ∨ (2 ∣ y) :=
by
  sorry

end either_x_or_y_is_even_l96_96346


namespace range_of_m_cond_l96_96149

noncomputable def quadratic_inequality (x m : ℝ) : Prop :=
  x^2 + m * x + 2 * m - 3 ≥ 0

theorem range_of_m_cond (m : ℝ) (h1 : 2 ≤ m) (h2 : m ≤ 6) (x : ℝ) :
  quadratic_inequality x m :=
sorry

end range_of_m_cond_l96_96149


namespace simplify_and_evaluate_expr_l96_96918

theorem simplify_and_evaluate_expr (a b : ℝ) (h1 : a = 1 / 2) (h2 : b = -4) :
  5 * (3 * a ^ 2 * b - a * b ^ 2) - 4 * (-a * b ^ 2 + 3 * a ^ 2 * b) = -11 :=
by
  sorry

end simplify_and_evaluate_expr_l96_96918


namespace fraction_zero_value_x_l96_96512

theorem fraction_zero_value_x (x : ℝ) (h1 : (x - 2) / (1 - x) = 0) (h2 : 1 - x ≠ 0) : x = 2 := 
sorry

end fraction_zero_value_x_l96_96512


namespace profit_sharing_l96_96711

-- Define constants and conditions
def Tom_investment : ℕ := 30000
def Tom_share : ℝ := 0.40

def Jose_investment : ℕ := 45000
def Jose_start_month : ℕ := 2
def Jose_share : ℝ := 0.30

def Sarah_investment : ℕ := 60000
def Sarah_start_month : ℕ := 5
def Sarah_share : ℝ := 0.20

def Ravi_investment : ℕ := 75000
def Ravi_start_month : ℕ := 8
def Ravi_share : ℝ := 0.10

def total_profit : ℕ := 120000

-- Define expected shares
def Tom_expected_share : ℕ := 48000
def Jose_expected_share : ℕ := 36000
def Sarah_expected_share : ℕ := 24000
def Ravi_expected_share : ℕ := 12000

-- Theorem statement
theorem profit_sharing :
  let Tom_contribution := Tom_investment * 12
  let Jose_contribution := Jose_investment * (12 - Jose_start_month)
  let Sarah_contribution := Sarah_investment * (12 - Sarah_start_month)
  let Ravi_contribution := Ravi_investment * (12 - Ravi_start_month)
  Tom_share * total_profit = Tom_expected_share ∧
  Jose_share * total_profit = Jose_expected_share ∧
  Sarah_share * total_profit = Sarah_expected_share ∧
  Ravi_share * total_profit = Ravi_expected_share := by {
    sorry
  }

end profit_sharing_l96_96711


namespace Diamond_result_l96_96057

-- Define the binary operation Diamond
def Diamond (a b : ℕ) : ℕ := a * b^2 - b + 1

theorem Diamond_result : Diamond (Diamond 3 4) 2 = 179 := 
by 
  sorry

end Diamond_result_l96_96057


namespace product_sum_of_roots_l96_96231

theorem product_sum_of_roots
  {p q r : ℝ}
  (h : (∀ x : ℝ, (4 * x^3 - 8 * x^2 + 16 * x - 12) = 0 → (x = p ∨ x = q ∨ x = r))) :
  p * q + q * r + r * p = 4 := 
sorry

end product_sum_of_roots_l96_96231


namespace hapok_max_coins_l96_96402

/-- The maximum number of coins Hapok can guarantee himself regardless of Glazok's actions is 46 coins. -/
theorem hapok_max_coins (total_coins : ℕ) (max_handfuls : ℕ) (coins_per_handful : ℕ) :
  total_coins = 100 ∧ max_handfuls = 9 ∧ (∀ h : ℕ, h ≤ max_handfuls) ∧ coins_per_handful ≤ total_coins →
  ∃ k : ℕ, k ≤ total_coins ∧ k = 46 :=
by {
  sorry
}

end hapok_max_coins_l96_96402


namespace factorial_inequality_l96_96627

theorem factorial_inequality (n : ℕ) : 2^n * n! < (n+1)^n :=
by
  sorry

end factorial_inequality_l96_96627


namespace repeating_decimal_sum_l96_96494

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_4 : ℚ := 4 / 9

theorem repeating_decimal_sum : repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_4 = 4 / 9 := by
  sorry

end repeating_decimal_sum_l96_96494


namespace roots_numerically_equal_but_opposite_signs_l96_96339

noncomputable def value_of_m (a b c : ℝ) : ℝ := (a - b) / (a + b)

theorem roots_numerically_equal_but_opposite_signs
  (a b c m : ℝ)
  (h : ∀ x : ℝ, (a ≠ 0 ∧ a + b ≠ 0) ∧ (x^2 - b*x = (ax - c) * (m - 1) / (m + 1))) 
  (root_condition : ∃ x₁ x₂ : ℝ, x₁ = -x₂ ∧ x₁ * x₂ != 0) :
  m = value_of_m a b c :=
by
  sorry

end roots_numerically_equal_but_opposite_signs_l96_96339


namespace probability_no_practice_l96_96761

def prob_has_practice : ℚ := 5 / 8

theorem probability_no_practice : 
  1 - prob_has_practice = 3 / 8 := 
by
  sorry

end probability_no_practice_l96_96761


namespace z_max_plus_z_min_l96_96244

theorem z_max_plus_z_min {x y z : ℝ} 
  (h1 : x^2 + y^2 + z^2 = 3) 
  (h2 : x + 2 * y - 2 * z = 4) : 
  z + z = -4 :=
by 
  sorry

end z_max_plus_z_min_l96_96244


namespace john_steps_l96_96436

/-- John climbs up 9 flights of stairs. Each flight is 10 feet. -/
def flights := 9
def flight_height_feet := 10

/-- Conversion factor between feet and inches. -/
def feet_to_inches := 12

/-- Each step is 18 inches. -/
def step_height_inches := 18

/-- The total number of steps John climbs. -/
theorem john_steps :
  (flights * flight_height_feet * feet_to_inches) / step_height_inches = 60 :=
by
  sorry

end john_steps_l96_96436


namespace range_of_k_l96_96411

-- Definitions for the conditions of p and q
def is_ellipse (k : ℝ) : Prop := (0 < k) ∧ (k < 4)
def is_hyperbola (k : ℝ) : Prop := 1 < k ∧ k < 3

-- The main proposition
theorem range_of_k (k : ℝ) : (is_ellipse k ∨ is_hyperbola k) → (1 < k ∧ k < 4) :=
by
  sorry

end range_of_k_l96_96411


namespace find_angle_C_l96_96039

open Real -- Opening Real to directly use real number functions and constants

noncomputable def triangle_angles_condition (A B C: ℝ) : Prop :=
  2 * sin A + 5 * cos B = 5 ∧ 5 * sin B + 2 * cos A = 2

-- Theorem statement
theorem find_angle_C (A B C: ℝ) (h: triangle_angles_condition A B C):
  C = arcsin (1 / 5) ∨ C = 180 - arcsin (1 / 5) :=
sorry

end find_angle_C_l96_96039


namespace cookies_remaining_percentage_l96_96204

theorem cookies_remaining_percentage: 
  ∀ (total initial_remaining eduardo_remaining final_remaining: ℕ),
  total = 600 → 
  initial_remaining = total - (2 * total / 5) → 
  eduardo_remaining = initial_remaining - (3 * initial_remaining / 5) → 
  final_remaining = eduardo_remaining → 
  (final_remaining * 100) / total = 24 := 
by
  intros total initial_remaining eduardo_remaining final_remaining h_total h_initial_remaining h_eduardo_remaining h_final_remaining
  sorry

end cookies_remaining_percentage_l96_96204


namespace paula_candies_distribution_l96_96110

-- Defining the given conditions and the question in Lean
theorem paula_candies_distribution :
  ∀ (initial_candies additional_candies friends : ℕ),
  initial_candies = 20 →
  additional_candies = 4 →
  friends = 6 →
  (initial_candies + additional_candies) / friends = 4 :=
by
  -- We skip the actual proof here
  intros initial_candies additional_candies friends h1 h2 h3
  sorry

end paula_candies_distribution_l96_96110


namespace find_m_l96_96415

noncomputable def vector_sum (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

noncomputable def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem find_m (m : ℝ) :
  let a := (1, m)
  let b := (3, -2)
  are_parallel (vector_sum a b) b → m = -2 / 3 :=
by
  sorry

end find_m_l96_96415


namespace generatrix_length_l96_96560

theorem generatrix_length (r : ℝ) (l : ℝ) 
  (h_radius : r = Real.sqrt 2) 
  (h_surface : 2 * Real.pi * r = Real.pi * l) : 
  l = 2 * Real.sqrt 2 := 
by 
  sorry

end generatrix_length_l96_96560


namespace max_distance_l96_96525

noncomputable def polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ := 
  (rho * Real.cos theta, rho * Real.sin theta)

noncomputable def curve_C (p : ℝ × ℝ) : Prop := 
  let x := p.1 
  let y := p.2 
  x^2 + y^2 - 2*y = 0

noncomputable def line_l (t : ℝ) : ℝ × ℝ := 
  (-3/5 * t + 2, 4/5 * t)

def x_axis_intersection (l : ℝ → ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := l 0 
  (x, 0)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem max_distance {M : ℝ × ℝ} {N : ℝ × ℝ}
  (curve_c : (ℝ × ℝ) → Prop)
  (line_l : ℝ → ℝ × ℝ)
  (h1 : curve_c = curve_C)
  (h2 : line_l = line_l)
  (M_def : x_axis_intersection line_l = M)
  (hNP : curve_c N) :
  distance M N ≤ Real.sqrt 5 + 1 :=
sorry

end max_distance_l96_96525


namespace product_of_decimal_numbers_l96_96851

theorem product_of_decimal_numbers 
  (h : 213 * 16 = 3408) : 
  1.6 * 21.3 = 34.08 :=
by
  sorry

end product_of_decimal_numbers_l96_96851


namespace vehicles_with_at_least_80_kmh_equal_50_l96_96342

variable (num_vehicles_80_to_89 : ℕ := 15)
variable (num_vehicles_90_to_99 : ℕ := 30)
variable (num_vehicles_100_to_109 : ℕ := 5)

theorem vehicles_with_at_least_80_kmh_equal_50 :
  num_vehicles_80_to_89 + num_vehicles_90_to_99 + num_vehicles_100_to_109 = 50 := by
  sorry

end vehicles_with_at_least_80_kmh_equal_50_l96_96342


namespace transport_cost_in_euros_l96_96568

def cost_per_kg : ℝ := 18000
def weight_g : ℝ := 300
def exchange_rate : ℝ := 0.95

theorem transport_cost_in_euros :
  (cost_per_kg * (weight_g / 1000) * exchange_rate) = 5130 :=
by sorry

end transport_cost_in_euros_l96_96568


namespace minimum_tenth_game_score_l96_96610

theorem minimum_tenth_game_score (S5 : ℕ) (score10 : ℕ) 
  (h1 : 18 + 15 + 16 + 19 = 68)
  (h2 : S5 ≤ 85)
  (h3 : (S5 + 68 + score10) / 10 > 17) : 
  score10 ≥ 18 := sorry

end minimum_tenth_game_score_l96_96610


namespace determine_N_l96_96258

/-- 
Each row and two columns in the grid forms distinct arithmetic sequences.
Given:
- First column values: 10 and 18 (arithmetic sequence).
- Second column top value: N, bottom value: -23 (arithmetic sequence).
Prove that N = -15.
 -/
theorem determine_N : ∃ N : ℤ, (∀ n : ℕ, 10 + n * 8 = 10 ∨ 10 + n * 8 = 18) ∧ (∀ m : ℕ, N + m * 8 = N ∨ N + m * 8 = -23) ∧ N = -15 :=
by {
  sorry
}

end determine_N_l96_96258


namespace int_solutions_exist_for_x2_plus_15y2_eq_4n_l96_96962

theorem int_solutions_exist_for_x2_plus_15y2_eq_4n (n : ℕ) (hn : n > 0) : 
  ∃ S : Finset (ℤ × ℤ), S.card ≥ n ∧ ∀ (xy : ℤ × ℤ), xy ∈ S → xy.1^2 + 15 * xy.2^2 = 4^n :=
by
  sorry

end int_solutions_exist_for_x2_plus_15y2_eq_4n_l96_96962


namespace max_x_value_l96_96686

variables {x y : ℝ}
variables (data : list (ℝ × ℝ))
variables (linear_relation : ℝ → ℝ → Prop)

def max_y : ℝ := 10

-- Given conditions
axiom linear_data :
  (data = [(16, 11), (14, 9), (12, 8), (8, 5)]) ∧
  (∀ (p : ℝ × ℝ), p ∈ data → linear_relation p.1 p.2)

-- Prove the maximum value of x for which y ≤ max_y
theorem max_x_value (h : ∀ (x y : ℝ), linear_relation x y → y = 11 - (16 - x) / 3):
  ∀ (x : ℝ), (∃ y : ℝ, linear_relation x y) → y ≤ max_y → x ≤ 15 :=
sorry

end max_x_value_l96_96686


namespace exists_right_triangle_area_eq_perimeter_l96_96635

theorem exists_right_triangle_area_eq_perimeter :
  ∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ a + b + c = (a * b) / 2 ∧ a ≠ b ∧ 
  ((a = 5 ∧ b = 12 ∧ c = 13) ∨ (a = 12 ∧ b = 5 ∧ c = 13) ∨ 
  (a = 6 ∧ b = 8 ∧ c = 10) ∨ (a = 8 ∧ b = 6 ∧ c = 10)) :=
by
  sorry

end exists_right_triangle_area_eq_perimeter_l96_96635


namespace hypotenuse_length_l96_96779

theorem hypotenuse_length (a b : ℝ) (c : ℝ) (h₁ : a = Real.sqrt 5) (h₂ : b = Real.sqrt 12) : c = Real.sqrt 17 :=
by
  -- Proof not required, hence skipped with 'sorry'
  sorry

end hypotenuse_length_l96_96779


namespace tobias_swimming_distance_l96_96041

def swimming_time_per_100_meters : ℕ := 5
def pause_time : ℕ := 5
def swimming_period : ℕ := 25
def total_visit_hours : ℕ := 3

theorem tobias_swimming_distance :
  let total_visit_minutes := total_visit_hours * 60
  let sequence_time := swimming_period + pause_time
  let number_of_sequences := total_visit_minutes / sequence_time
  let total_pause_time := number_of_sequences * pause_time
  let total_swimming_time := total_visit_minutes - total_pause_time
  let number_of_100m_lengths := total_swimming_time / swimming_time_per_100_meters
  let total_distance := number_of_100m_lengths * 100
  total_distance = 3000 :=
by
  sorry

end tobias_swimming_distance_l96_96041


namespace remainder_13_pow_2000_mod_1000_l96_96905

theorem remainder_13_pow_2000_mod_1000 :
  (13^2000) % 1000 = 1 := 
by 
  sorry

end remainder_13_pow_2000_mod_1000_l96_96905


namespace largest_prime_factor_of_85_l96_96060

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) (p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q ∧ q ∣ n → q ≤ p

theorem largest_prime_factor_of_85 :
  let a := 65
  let b := 85
  let c := 91
  let d := 143
  let e := 169
  largest_prime_factor b 17 :=
by
  sorry

end largest_prime_factor_of_85_l96_96060


namespace degrees_for_basic_astrophysics_correct_l96_96178

-- Definitions for conditions
def percentage_allocations : List ℚ := [13, 24, 15, 29, 8]
def total_percentage : ℚ := percentage_allocations.sum
def remaining_percentage : ℚ := 100 - total_percentage

-- The question to answer
def total_degrees : ℚ := 360
def degrees_for_basic_astrophysics : ℚ := remaining_percentage / 100 * total_degrees

-- Prove that the degrees for basic astrophysics is 39.6
theorem degrees_for_basic_astrophysics_correct :
  degrees_for_basic_astrophysics = 39.6 :=
by
  sorry

end degrees_for_basic_astrophysics_correct_l96_96178


namespace combined_list_correct_l96_96300

def james_friends : ℕ := 75
def john_friends : ℕ := 3 * james_friends
def shared_friends : ℕ := 25

def combined_list : ℕ :=
  james_friends + john_friends - shared_friends

theorem combined_list_correct : combined_list = 275 := by
  unfold combined_list
  unfold james_friends
  unfold john_friends
  unfold shared_friends
  sorry

end combined_list_correct_l96_96300


namespace maximum_n_l96_96827

variable (x y z : ℝ)

theorem maximum_n (h1 : x + y + z = 12) (h2 : x * y + y * z + z * x = 30) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ n, n = min (x * y) (min (y * z) (z * x)) ∧ n = 2 :=
by
  sorry

end maximum_n_l96_96827


namespace cube_volume_of_surface_area_l96_96517

theorem cube_volume_of_surface_area (S : ℝ) (V : ℝ) (a : ℝ) (h1 : S = 150) (h2 : S = 6 * a^2) (h3 : V = a^3) : V = 125 := by
  sorry

end cube_volume_of_surface_area_l96_96517


namespace complement_U_M_inter_N_eq_l96_96995

def U : Set ℝ := Set.univ

def M : Set ℝ := { y | ∃ x, y = 2 * x + 1 ∧ -1/2 ≤ x ∧ x ≤ 1/2 }

def N : Set ℝ := { x | ∃ y, y = Real.log (x^2 + 3 * x) ∧ (x < -3 ∨ x > 0) }

def complement_U_M : Set ℝ := U \ M

theorem complement_U_M_inter_N_eq :
  (complement_U_M ∩ N) = ((Set.Iio (-3 : ℝ)) ∪ (Set.Ioi (2 : ℝ))) :=
sorry

end complement_U_M_inter_N_eq_l96_96995


namespace percentage_increase_l96_96542

theorem percentage_increase (A B : ℝ) (y : ℝ) (h : A > B) (h1 : B > 0) (h2 : C = A + B) (h3 : C = (1 + y / 100) * B) : y = 100 * (A / B) := 
sorry

end percentage_increase_l96_96542


namespace network_structure_l96_96138

theorem network_structure 
  (n : ℕ)
  (is_acquainted : Fin n → Fin n → Prop)
  (H_symmetric : ∀ x y, is_acquainted x y = is_acquainted y x) 
  (H_common_acquaintance : ∀ x y, ¬ is_acquainted x y → ∃! z : Fin n, is_acquainted x z ∧ is_acquainted y z) :
  ∃ (G : SimpleGraph (Fin n)), (∀ x y, G.Adj x y = is_acquainted x y) ∧
    (∀ x y, ¬ G.Adj x y → (∃ (z1 z2 : Fin n), G.Adj x z1 ∧ G.Adj y z1 ∧ G.Adj x z2 ∧ G.Adj y z2)) :=
by
  sorry

end network_structure_l96_96138


namespace total_amount_invested_l96_96126

variable (T : ℝ)

def income_first (T : ℝ) : ℝ :=
  0.10 * (T - 700)

def income_second : ℝ :=
  0.08 * 700

theorem total_amount_invested :
  income_first T - income_second = 74 → T = 2000 :=
by
  intros h
  sorry 

end total_amount_invested_l96_96126


namespace range_of_quadratic_function_l96_96383

noncomputable def quadratic_function_range : Set ℝ :=
  { y : ℝ | ∃ x : ℝ, y = x^2 - 6 * x + 7 }

theorem range_of_quadratic_function :
  quadratic_function_range = { y : ℝ | y ≥ -2 } :=
by
  -- Insert proof here
  sorry

end range_of_quadratic_function_l96_96383


namespace solve_equation_frac_l96_96291

theorem solve_equation_frac (x : ℝ) (h : x ≠ 2) : (3 / (x - 2) = 1) ↔ (x = 5) :=
by
  sorry -- proof is to be constructed

end solve_equation_frac_l96_96291


namespace find_x2_div_c2_add_y2_div_a2_add_z2_div_b2_l96_96002

variable (a b c x y z : ℝ)

theorem find_x2_div_c2_add_y2_div_a2_add_z2_div_b2 
  (h1 : a * (x / c) + b * (y / a) + c * (z / b) = 5) 
  (h2 : c / x + a / y + b / z = 0) : 
  x^2 / c^2 + y^2 / a^2 + z^2 / b^2 = 25 := 
sorry

end find_x2_div_c2_add_y2_div_a2_add_z2_div_b2_l96_96002


namespace number_of_cows_l96_96469

variable (x y z : ℕ)

theorem number_of_cows (h1 : 4 * x + 2 * y + 2 * z = 24 + 2 * (x + y + z)) (h2 : z = y / 2) : x = 12 := 
sorry

end number_of_cows_l96_96469


namespace expenditure_fraction_l96_96848

variable (B : ℝ)
def cost_of_book (x y : ℝ) (B : ℝ) := x = 0.30 * (B - 2 * y)
def cost_of_coffee (x y : ℝ) (B : ℝ) := y = 0.10 * (B - x)

theorem expenditure_fraction (x y : ℝ) (B : ℝ) 
  (hx : cost_of_book x y B) 
  (hy : cost_of_coffee x y B) : 
  (x + y) / B = 31 / 94 :=
sorry

end expenditure_fraction_l96_96848


namespace find_primes_pqr_eq_5_sum_l96_96497

theorem find_primes_pqr_eq_5_sum (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) :
  p * q * r = 5 * (p + q + r) → (p = 2 ∧ q = 5 ∧ r = 7) ∨ (p = 2 ∧ q = 7 ∧ r = 5) ∨
                                         (p = 5 ∧ q = 2 ∧ r = 7) ∨ (p = 5 ∧ q = 7 ∧ r = 2) ∨
                                         (p = 7 ∧ q = 2 ∧ r = 5) ∨ (p = 7 ∧ q = 5 ∧ r = 2) :=
by
  sorry

end find_primes_pqr_eq_5_sum_l96_96497


namespace number_of_ways_to_make_125_quacks_using_1_5_25_125_coins_l96_96774

def num_ways_to_make_125_quacks_using_coins : ℕ :=
  have h : ∃ (a b c d : ℕ), a + 5 * b + 25 * c + 125 * d = 125 := sorry
  82

theorem number_of_ways_to_make_125_quacks_using_1_5_25_125_coins : num_ways_to_make_125_quacks_using_coins = 82 := 
  sorry

end number_of_ways_to_make_125_quacks_using_1_5_25_125_coins_l96_96774


namespace sequence_sum_l96_96147

def alternating_sum : List ℤ := [2, -7, 10, -15, 18, -23, 26, -31, 34, -39, 40, -45, 48]

theorem sequence_sum : alternating_sum.sum = 13 := by
  sorry

end sequence_sum_l96_96147


namespace percent_calculation_l96_96011

theorem percent_calculation (Part Whole : ℝ) (hPart : Part = 14) (hWhole : Whole = 70) : 
  (Part / Whole) * 100 = 20 := 
by 
  sorry

end percent_calculation_l96_96011


namespace fraction_of_time_to_cover_distance_l96_96643

-- Definitions for the given conditions
def distance : ℝ := 540
def initial_time : ℝ := 12
def new_speed : ℝ := 60

-- The statement we need to prove
theorem fraction_of_time_to_cover_distance :
  ∃ (x : ℝ), (x = 3 / 4) ∧ (distance / (initial_time * x) = new_speed) :=
by
  -- Proof steps would go here
  sorry

end fraction_of_time_to_cover_distance_l96_96643


namespace food_initially_meant_to_last_22_days_l96_96767

variable (D : ℕ)   -- Denoting the initial number of days the food was meant to last
variable (m : ℕ := 760)  -- Initial number of men
variable (total_men : ℕ := 1520)  -- Total number of men after 2 days

-- The first condition derived from the problem: total amount of food
def total_food := m * D

-- The second condition derived from the problem: Remaining food after 2 days
def remaining_food_after_2_days := total_food - m * 2

-- The third condition derived from the problem: Remaining food to last for 10 more days
def remaining_food_to_last_10_days := total_men * 10

-- Statement to prove
theorem food_initially_meant_to_last_22_days :
  D - 2 = 10 →
  D = 22 :=
by
  sorry

end food_initially_meant_to_last_22_days_l96_96767


namespace rental_cost_equal_mileage_l96_96076

theorem rental_cost_equal_mileage :
  ∃ m : ℝ, 
    (21.95 + 0.19 * m = 18.95 + 0.21 * m) ∧ 
    m = 150 :=
by
  sorry

end rental_cost_equal_mileage_l96_96076


namespace sector_central_angle_l96_96089

-- The conditions
def r : ℝ := 2
def S : ℝ := 4

-- The question
theorem sector_central_angle : ∃ α : ℝ, |α| = 2 ∧ S = 0.5 * α * r * r :=
by
  sorry

end sector_central_angle_l96_96089


namespace total_cans_to_collect_l96_96225

def cans_for_project (marthas_cans : ℕ) (additional_cans_needed : ℕ) (total_cans_needed : ℕ) : Prop :=
  ∃ diegos_cans : ℕ, diegos_cans = (marthas_cans / 2) + 10 ∧ 
  total_cans_needed = marthas_cans + diegos_cans + additional_cans_needed

theorem total_cans_to_collect : 
  cans_for_project 90 5 150 :=
by
  -- Insert proof here in actual usage
  sorry

end total_cans_to_collect_l96_96225


namespace fraction_sum_l96_96464

variable (a b : ℝ)

theorem fraction_sum
  (hb : b + 1 ≠ 0) :
  (a / (b + 1)) + (2 * a / (b + 1)) - (3 * a / (b + 1)) = 0 :=
by sorry

end fraction_sum_l96_96464


namespace ages_correct_l96_96738

variables (Son Daughter Wife Man Father : ℕ)

theorem ages_correct :
  (Man = Son + 20) ∧
  (Man = Daughter + 15) ∧
  (Man + 2 = 2 * (Son + 2)) ∧
  (Man + 2 = 3 * (Daughter + 2)) ∧
  (Wife = Man - 5) ∧
  (Wife + 6 = 2 * (Daughter + 6)) ∧
  (Father = Man + 32) →
  (Son = 7 ∧ Daughter = 12 ∧ Wife = 22 ∧ Man = 27 ∧ Father = 59) :=
by
  intros h
  sorry

end ages_correct_l96_96738


namespace line_through_points_C_D_has_undefined_slope_and_angle_90_l96_96986

theorem line_through_points_C_D_has_undefined_slope_and_angle_90 (m : ℝ) (n : ℝ) (hn : n ≠ 0) :
  ∃ θ : ℝ, (∀ (slope : ℝ), false) ∧ θ = 90 :=
by { sorry }

end line_through_points_C_D_has_undefined_slope_and_angle_90_l96_96986


namespace profit_percentage_l96_96500

theorem profit_percentage (cost_price selling_price profit_percentage : ℚ) 
  (h_cost_price : cost_price = 240) 
  (h_selling_price : selling_price = 288) 
  (h_profit_percentage : profit_percentage = 20) : 
  profit_percentage = ((selling_price - cost_price) / cost_price) * 100 := 
by 
  sorry

end profit_percentage_l96_96500


namespace min_value_of_expr_l96_96161

noncomputable def min_value_expr (x y : ℝ) : ℝ :=
  ((x^2 + 1 / y^2 + 1) * (x^2 + 1 / y^2 - 1000)) +
  ((y^2 + 1 / x^2 + 1) * (y^2 + 1 / x^2 - 1000))

theorem min_value_of_expr :
  ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ min_value_expr x y = -498998 :=
by
  sorry

end min_value_of_expr_l96_96161


namespace initial_house_cats_l96_96016

theorem initial_house_cats (H : ℕ) (H_condition : 13 + H - 10 = 8) : H = 5 :=
by
-- sorry provides a placeholder to skip the actual proof
sorry

end initial_house_cats_l96_96016


namespace cylinder_height_percentage_l96_96095

-- Lean 4 statement for the problem
theorem cylinder_height_percentage (h : ℝ) (r : ℝ) (H : ℝ) :
  (7 / 8) * h = (3 / 5) * (1.25 * r)^2 * H → H = 0.9333 * h :=
by 
  sorry

end cylinder_height_percentage_l96_96095


namespace tan_105_degree_is_neg_sqrt3_minus_2_l96_96090

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end tan_105_degree_is_neg_sqrt3_minus_2_l96_96090


namespace circles_ordering_l96_96364

theorem circles_ordering :
  let rA := 3
  let AB := 12 * Real.pi
  let AC := 28 * Real.pi
  let rB := Real.sqrt 12
  let rC := Real.sqrt 28
  (rA < rB) ∧ (rB < rC) :=
by
  let rA := 3
  let AB := 12 * Real.pi
  let AC := 28 * Real.pi
  let rB := Real.sqrt 12
  let rC := Real.sqrt 28
  have rA_lt_rB: rA < rB := by sorry
  have rB_lt_rC: rB < rC := by sorry
  exact ⟨rA_lt_rB, rB_lt_rC⟩

end circles_ordering_l96_96364


namespace greatest_four_digit_number_divisible_by_3_and_4_l96_96976

theorem greatest_four_digit_number_divisible_by_3_and_4 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ (n % 12 = 0) ∧ (∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ (m % 12 = 0) → m ≤ 9996) :=
by sorry

end greatest_four_digit_number_divisible_by_3_and_4_l96_96976


namespace inequality_always_true_l96_96502

theorem inequality_always_true (a : ℝ) : (∀ x : ℝ, |x - 1| - |x + 2| ≤ a) ↔ 3 ≤ a :=
by
  sorry

end inequality_always_true_l96_96502


namespace superior_points_in_Omega_l96_96026

-- Define the set Omega
def Omega : Set (ℝ × ℝ) := { p | let (x, y) := p; x^2 + y^2 ≤ 2008 }

-- Definition of the superior relation
def superior (P P' : ℝ × ℝ) : Prop :=
  let (x, y) := P
  let (x', y') := P'
  x ≤ x' ∧ y ≥ y'

-- Definition of the set of points Q such that no other point in Omega is superior to Q
def Q_set : Set (ℝ × ℝ) :=
  { p | let (x, y) := p; x^2 + y^2 = 2008 ∧ x ≤ 0 ∧ y ≥ 0 }

theorem superior_points_in_Omega :
  { p | p ∈ Omega ∧ ¬ (∃ q ∈ Omega, superior q p) } = Q_set :=
by
  sorry

end superior_points_in_Omega_l96_96026


namespace remainder_when_divided_by_product_l96_96442

noncomputable def Q : Polynomial ℝ := sorry

theorem remainder_when_divided_by_product (Q : Polynomial ℝ)
    (h1 : Q.eval 20 = 100)
    (h2 : Q.eval 100 = 20) :
    ∃ R : Polynomial ℝ, ∃ a b : ℝ, Q = (Polynomial.X - 20) * (Polynomial.X - 100) * R + Polynomial.C a * Polynomial.X + Polynomial.C b ∧
    a = -1 ∧ b = 120 :=
by
  sorry

end remainder_when_divided_by_product_l96_96442


namespace prime_eq_sol_l96_96612

theorem prime_eq_sol {p q x y z : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by
  sorry

end prime_eq_sol_l96_96612


namespace correct_factorization_l96_96802

theorem correct_factorization : 
  (¬ (6 * x^2 * y^3 = 2 * x^2 * 3 * y^3)) ∧ 
  (¬ (x^2 + 2 * x + 1 = x * (x^2 + 2) + 1)) ∧ 
  (¬ ((x + 2) * (x - 3) = x^2 - x - 6)) ∧ 
  (x^2 - 9 = (x - 3) * (x + 3)) :=
by 
  sorry

end correct_factorization_l96_96802


namespace percentage_not_caught_l96_96420

theorem percentage_not_caught (x : ℝ) (h1 : 22 + x = 25.88235294117647) : x = 3.88235294117647 :=
sorry

end percentage_not_caught_l96_96420


namespace sum_of_ten_numbers_l96_96381

theorem sum_of_ten_numbers (average count : ℝ) (h_avg : average = 5.3) (h_count : count = 10) : 
  average * count = 53 :=
by
  sorry

end sum_of_ten_numbers_l96_96381


namespace solve_problem_l96_96630

def problem_statement (x y : ℕ) : Prop :=
  (x = 3) ∧ (y = 2) → (x^8 + 2 * x^4 * y^2 + y^4) / (x^4 + y^2) = 85

theorem solve_problem : problem_statement 3 2 :=
  by sorry

end solve_problem_l96_96630


namespace sector_area_l96_96030

-- Given conditions
variables {l r : ℝ}

-- Definitions (conditions from the problem)
def arc_length (l : ℝ) := l
def radius (r : ℝ) := r

-- Problem statement
theorem sector_area (l r : ℝ) : 
    (1 / 2) * l * r = (1 / 2) * l * r :=
by
  sorry

end sector_area_l96_96030


namespace five_person_lineup_l96_96496

theorem five_person_lineup : 
  let total_ways := Nat.factorial 5
  let invalid_first := Nat.factorial 4
  let invalid_last := Nat.factorial 4
  let valid_ways := total_ways - (invalid_first + invalid_last)
  valid_ways = 72 :=
by
  sorry

end five_person_lineup_l96_96496


namespace sum_x_y_z_w_l96_96924

-- Define the conditions in Lean
variables {x y z w : ℤ}
axiom h1 : x - y + z = 7
axiom h2 : y - z + w = 8
axiom h3 : z - w + x = 4
axiom h4 : w - x + y = 3

-- Prove the result
theorem sum_x_y_z_w : x + y + z + w = 22 := by
  sorry

end sum_x_y_z_w_l96_96924


namespace condition_an_necessary_but_not_sufficient_l96_96326

-- Definitions for the sequence and properties
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, n ≥ 1 → a (n + 1) = r * (a n)

def condition_an (a : ℕ → ℝ) : Prop :=
  ∀ n, n ≥ 2 → a n = 2 * a (n - 1)

-- The theorem statement
theorem condition_an_necessary_but_not_sufficient (a : ℕ → ℝ) :
  (∀ n, n ≥ 1 → a (n + 1) = 2 * (a n)) → (condition_an a) ∧ ¬(is_geometric_sequence a 2) :=
by
  sorry

end condition_an_necessary_but_not_sufficient_l96_96326


namespace triangle_area_interval_l96_96572

theorem triangle_area_interval (s : ℝ) :
  10 ≤ (s - 1)^(3 / 2) ∧ (s - 1)^(3 / 2) ≤ 50 → (5.64 ≤ s ∧ s ≤ 18.32) :=
by
  sorry

end triangle_area_interval_l96_96572


namespace fraction_result_l96_96404

theorem fraction_result (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (2 * x + 3 * y) / (x - 2 * y) = 3) : 
  (x + 2 * y) / (2 * x - y) = 11 / 17 :=
sorry

end fraction_result_l96_96404


namespace always_positive_sum_reciprocal_inequality_l96_96314

-- Problem 1
theorem always_positive (x : ℝ) : x^6 - x^3 + x^2 - x + 1 > 0 :=
sorry

-- Problem 2
theorem sum_reciprocal_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  1/a + 1/b + 1/c ≥ 9 :=
sorry

end always_positive_sum_reciprocal_inequality_l96_96314


namespace num_of_terms_in_arithmetic_sequence_l96_96601

-- Define the arithmetic sequence
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the first term, common difference, and last term of the sequence
def a : ℕ := 15
def d : ℕ := 4
def last_term : ℕ := 99

-- Define the number of terms in the sequence
def n : ℕ := 22

-- State the theorem
theorem num_of_terms_in_arithmetic_sequence : arithmetic_seq a d n = last_term :=
by
  sorry

end num_of_terms_in_arithmetic_sequence_l96_96601


namespace scientific_notation_example_l96_96506

theorem scientific_notation_example :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 218000000 = a * 10 ^ n ∧ a = 2.18 ∧ n = 8 :=
by {
  -- statement of the problem conditions
  sorry
}

end scientific_notation_example_l96_96506


namespace diff_of_two_numbers_l96_96295

theorem diff_of_two_numbers :
  ∃ D S : ℕ, (1650 = 5 * S + 5) ∧ (D = 1650 - S) ∧ (D = 1321) :=
sorry

end diff_of_two_numbers_l96_96295


namespace sum_of_factors_eq_12_l96_96448

-- Define the polynomial for n = 1
def poly (x : ℤ) : ℤ := x^5 + x + 1

-- Define the two factors when x = 2
def factor1 (x : ℤ) : ℤ := x^3 - x^2 + 1
def factor2 (x : ℤ) : ℤ := x^2 + x + 1

-- State the sum of the two factors at x = 2 equals 12
theorem sum_of_factors_eq_12 (x : ℤ) (h : x = 2) : factor1 x + factor2 x = 12 :=
by {
  sorry
}

end sum_of_factors_eq_12_l96_96448


namespace michael_remaining_money_l96_96655

variables (m b n : ℝ) (h1 : (1 : ℝ) / 3 * m = 1 / 2 * n * b) (h2 : 5 = m / 15)

theorem michael_remaining_money : m - (2 / 3 * m + m / 15) = 4 / 15 * m :=
by
  have hb1 : 2 / 3 * m = (2 * m) / 3 := by ring
  have hb2 : m / 15 = (1 * m) / 15 := by ring
  rw [hb1, hb2]
  sorry

end michael_remaining_money_l96_96655


namespace maximum_m_value_l96_96312

theorem maximum_m_value (a : ℕ → ℤ) (m : ℕ) :
  (∀ n, a (n + 1) - a n = 3) →
  a 3 = -2 →
  (∀ k : ℕ, k ≥ 4 → (3 * k - 8) * (3 * k - 5) / (3 * k - 11) ≥ 3 * m - 11) →
  m ≤ 9 :=
by
  sorry

end maximum_m_value_l96_96312


namespace fraction_sum_l96_96336

theorem fraction_sum : (1 / 3 : ℚ) + (5 / 9 : ℚ) = (8 / 9 : ℚ) :=
by
  sorry

end fraction_sum_l96_96336


namespace sector_max_angle_l96_96705

variables (r l : ℝ)

theorem sector_max_angle (h : 2 * r + l = 40) : (l / r) = 2 :=
sorry

end sector_max_angle_l96_96705


namespace candy_total_cost_l96_96050

theorem candy_total_cost
    (grape_candies cherry_candies apple_candies : ℕ)
    (cost_per_candy : ℝ)
    (h1 : grape_candies = 3 * cherry_candies)
    (h2 : apple_candies = 2 * grape_candies)
    (h3 : cost_per_candy = 2.50)
    (h4 : grape_candies = 24) :
    (grape_candies + cherry_candies + apple_candies) * cost_per_candy = 200 := 
by
  sorry

end candy_total_cost_l96_96050


namespace simplify_expression_l96_96975

-- Define the original expression and the simplified version
def original_expr (x y : ℤ) : ℤ := 7 * x + 3 - 2 * x + 15 + y
def simplified_expr (x y : ℤ) : ℤ := 5 * x + y + 18

-- The equivalence to be proved
theorem simplify_expression (x y : ℤ) : original_expr x y = simplified_expr x y :=
by sorry

end simplify_expression_l96_96975


namespace aaron_and_carson_scoops_l96_96735

def initial_savings (a c : ℕ) : Prop :=
  a = 150 ∧ c = 150

def total_savings (t a c : ℕ) : Prop :=
  t = a + c

def restaurant_expense (r t : ℕ) : Prop :=
  r = 3 * t / 4

def service_charge_inclusive (r sc : ℕ) : Prop :=
  r = sc * 115 / 100

def remaining_money (t r rm : ℕ) : Prop :=
  rm = t - r

def money_left (al cl : ℕ) : Prop :=
  al = 4 ∧ cl = 4

def ice_cream_scoop_cost (s : ℕ) : Prop :=
  s = 4

def total_scoops (rm ml s scoop_total : ℕ) : Prop :=
  scoop_total = (rm - (ml - 4 - 4)) / s

theorem aaron_and_carson_scoops :
  ∃ a c t r sc rm al cl s scoop_total, initial_savings a c ∧
  total_savings t a c ∧
  restaurant_expense r t ∧
  service_charge_inclusive r sc ∧
  remaining_money t r rm ∧
  money_left al cl ∧
  ice_cream_scoop_cost s ∧
  total_scoops rm (al + cl) s scoop_total ∧
  scoop_total = 16 :=
sorry

end aaron_and_carson_scoops_l96_96735


namespace one_third_of_nine_times_x_decreased_by_three_is_3x_minus_1_l96_96546

-- Definition of the conditions.
variable (x : ℝ)

-- Statement of the problem in Lean.
theorem one_third_of_nine_times_x_decreased_by_three_is_3x_minus_1 (x : ℝ) :
    (1 / 3) * (9 * x - 3) = 3 * x - 1 :=
by sorry

end one_third_of_nine_times_x_decreased_by_three_is_3x_minus_1_l96_96546


namespace longest_side_of_triangle_l96_96696

-- Definitions of the conditions in a)
def side1 : ℝ := 9
def side2 (x : ℝ) : ℝ := x + 5
def side3 (x : ℝ) : ℝ := 2 * x + 3
def perimeter : ℝ := 40

-- Statement of the mathematically equivalent proof problem.
theorem longest_side_of_triangle (x : ℝ) (h : side1 + side2 x + side3 x = perimeter) : 
  max side1 (max (side2 x) (side3 x)) = side3 x := 
sorry

end longest_side_of_triangle_l96_96696


namespace quadratic_completing_square_l96_96453

theorem quadratic_completing_square (b p : ℝ) (hb : b < 0)
  (h_quad_eq : ∀ x : ℝ, x^2 + b * x + (1 / 6) = (x + p)^2 + (1 / 18)) :
  b = - (2 / 3) :=
by
  sorry

end quadratic_completing_square_l96_96453


namespace total_apples_picked_l96_96191

-- Define the number of apples picked by Benny
def applesBenny : Nat := 2

-- Define the number of apples picked by Dan
def applesDan : Nat := 9

-- The theorem we want to prove
theorem total_apples_picked : applesBenny + applesDan = 11 := 
by 
  sorry

end total_apples_picked_l96_96191


namespace jane_picked_fraction_l96_96439

-- Define the total number of tomatoes initially
def total_tomatoes : ℕ := 100

-- Define the number of tomatoes remaining at the end
def remaining_tomatoes : ℕ := 15

-- Define the number of tomatoes picked in the second week
def second_week_tomatoes : ℕ := 20

-- Define the number of tomatoes picked in the third week
def third_week_tomatoes : ℕ := 2 * second_week_tomatoes

theorem jane_picked_fraction :
  ∃ (f : ℚ), f = 1 / 4 ∧
    (f * total_tomatoes + second_week_tomatoes + third_week_tomatoes + remaining_tomatoes = total_tomatoes) :=
sorry

end jane_picked_fraction_l96_96439


namespace proposition_equivalence_l96_96521

-- Definition of propositions p and q
variables (p q : Prop)

-- Statement of the problem in Lean 4
theorem proposition_equivalence :
  (p ∨ q) → ¬(p ∧ q) ↔ (¬((p ∨ q) → ¬(p ∧ q)) ∧ ¬(¬(p ∧ q) → (p ∨ q))) :=
sorry

end proposition_equivalence_l96_96521


namespace isosceles_triangle_perimeter_correct_l96_96832

-- Definitions based on conditions
def equilateral_triangle_side_length (perimeter : ℕ) : ℕ :=
  perimeter / 3

def isosceles_triangle_perimeter (side1 side2 base : ℕ) : ℕ :=
  side1 + side2 + base

-- Given conditions
def equilateral_triangle_perimeter : ℕ := 45
def equilateral_triangle_side : ℕ := equilateral_triangle_side_length equilateral_triangle_perimeter

-- The side of the equilateral triangle is also a leg of the isosceles triangle
def isosceles_triangle_leg : ℕ := equilateral_triangle_side
def isosceles_triangle_base : ℕ := 10

-- The problem to prove
theorem isosceles_triangle_perimeter_correct : 
  isosceles_triangle_perimeter isosceles_triangle_leg isosceles_triangle_leg isosceles_triangle_base = 40 :=
by
  sorry

end isosceles_triangle_perimeter_correct_l96_96832


namespace exists_universal_accessible_city_l96_96226

-- Define the basic structure for cities and flights
structure Country :=
  (City : Type)
  (accessible : City → City → Prop)

namespace Country

-- Define the properties of accessibility in the country
variables {C : Country}

-- Axiom: Each city is accessible from itself
axiom self_accessible (A : C.City) : C.accessible A A

-- Axiom: For any two cities, there exists a city from which both are accessible
axiom exists_intermediate (P Q : C.City) : ∃ R : C.City, C.accessible R P ∧ C.accessible R Q

-- Definition of the main theorem
theorem exists_universal_accessible_city :
  ∃ U : C.City, ∀ A : C.City, C.accessible U A :=
sorry

end Country

end exists_universal_accessible_city_l96_96226


namespace man_average_interest_rate_l96_96778

noncomputable def average_rate_of_interest (total_investment : ℝ) (rate1 rate2 rate_average : ℝ) 
    (x : ℝ) (same_return : (rate1 * (total_investment - x) = rate2 * x)) : Prop :=
  (rate_average = ((rate1 * (total_investment - x) + rate2 * x) / total_investment))

theorem man_average_interest_rate
    (total_investment : ℝ) 
    (rate1 : ℝ)
    (rate2 : ℝ)
    (rate_average : ℝ)
    (x : ℝ)
    (same_return : rate1 * (total_investment - x) = rate2 * x) :
    total_investment = 4500 ∧ rate1 = 0.04 ∧ rate2 = 0.06 ∧ x = 1800 ∧ rate_average = 0.048 → 
    average_rate_of_interest total_investment rate1 rate2 rate_average x same_return := 
by
  sorry

end man_average_interest_rate_l96_96778


namespace fifth_flower_is_e_l96_96317

def flowers : List String := ["a", "b", "c", "d", "e", "f", "g"]

theorem fifth_flower_is_e : flowers.get! 4 = "e" := sorry

end fifth_flower_is_e_l96_96317


namespace determine_v6_l96_96238

variable (v : ℕ → ℝ)

-- Given initial conditions: v₄ = 12 and v₇ = 471
def initial_conditions := v 4 = 12 ∧ v 7 = 471

-- Recurrence relation definition: vₙ₊₂ = 3vₙ₊₁ + vₙ
def recurrence_relation := ∀ n : ℕ, v (n + 2) = 3 * v (n + 1) + v n

-- The target is to prove that v₆ = 142.5
theorem determine_v6 (h1 : initial_conditions v) (h2 : recurrence_relation v) : 
  v 6 = 142.5 :=
sorry

end determine_v6_l96_96238


namespace side_length_of_square_ground_l96_96219

theorem side_length_of_square_ground
    (radius : ℝ)
    (Q_area : ℝ)
    (pi : ℝ)
    (quarter_circle_area : Q_area = (pi * (radius^2) / 4))
    (pi_approx : pi = 3.141592653589793)
    (Q_area_val : Q_area = 15393.804002589986)
    (radius_val : radius = 140) :
    ∃ (s : ℝ), s^2 = radius^2 :=
by
  sorry -- Proof not required per the instructions

end side_length_of_square_ground_l96_96219


namespace parabola_line_non_intersect_l96_96335

theorem parabola_line_non_intersect (r s : ℝ) (Q : ℝ × ℝ) (P : ℝ → ℝ)
  (hP : ∀ x, P x = x^2)
  (hQ : Q = (10, 6))
  (h_cond : ∀ m : ℝ, ¬∃ x : ℝ, (Q.snd - 6 = m * (Q.fst - 10)) ∧ (P x = x^2) ↔ r < m ∧ m < s) :
  r + s = 40 :=
sorry

end parabola_line_non_intersect_l96_96335


namespace matrix_det_eq_seven_l96_96061

theorem matrix_det_eq_seven (p q r s : ℝ) (h : p * s - q * r = 7) : 
  (p - 2 * r) * s - (q - 2 * s) * r = 7 := 
sorry

end matrix_det_eq_seven_l96_96061


namespace polynomial_degree_l96_96786

variable {P : Polynomial ℝ}

theorem polynomial_degree (h1 : ∀ x : ℝ, (x - 4) * P.eval (2 * x) = 4 * (x - 1) * P.eval x) (h2 : P.eval 0 ≠ 0) : P.degree = 2 := 
sorry

end polynomial_degree_l96_96786


namespace find_n_from_binomial_variance_l96_96813

variable (ξ : Type)
variable (n : ℕ)
variable (p : ℝ := 0.3)
variable (Var : ℕ → ℝ → ℝ := λ n p => n * p * (1 - p))

-- Given conditions
axiom binomial_distribution : p = 0.3 ∧ Var n p = 2.1

-- Prove n = 10
theorem find_n_from_binomial_variance (ξ : Type) (n : ℕ) (p : ℝ := 0.3) (Var : ℕ → ℝ → ℝ := λ n p => n * p * (1 - p)) :
  p = 0.3 ∧ Var n p = 2.1 → n = 10 :=
by
  sorry

end find_n_from_binomial_variance_l96_96813


namespace sawing_time_determination_l96_96634

variable (totalLength pieceLength sawTime : Nat)

theorem sawing_time_determination
  (h1 : totalLength = 10)
  (h2 : pieceLength = 2)
  (h3 : sawTime = 10) :
  (totalLength / pieceLength - 1) * sawTime = 40 := by
  sorry

end sawing_time_determination_l96_96634


namespace sum_first_10_terms_l96_96468

-- Define the conditions for the problem
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def arithmetic_sequence (b c d : ℝ) : Prop :=
  2 * c = b + d

def conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 1 = 1 ∧
  geometric_sequence a q ∧
  arithmetic_sequence (4 * a 1) (2 * a 2) (a 3)

-- Define the sum of the first n terms of a geometric sequence
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

-- Prove the final result
theorem sum_first_10_terms (a : ℕ → ℝ) (q : ℝ) (h : conditions a q) :
  sum_first_n_terms a 10 = 1023 :=
sorry

end sum_first_10_terms_l96_96468


namespace shopper_savings_percentage_l96_96173

theorem shopper_savings_percentage
  (amount_saved : ℝ) (final_price : ℝ)
  (h_saved : amount_saved = 3)
  (h_final : final_price = 27) :
  (amount_saved / (final_price + amount_saved)) * 100 = 10 := 
by
  sorry

end shopper_savings_percentage_l96_96173


namespace boy_present_age_l96_96955

theorem boy_present_age : ∃ x : ℕ, (x + 4 = 2 * (x - 6)) ∧ x = 16 := by
  sorry

end boy_present_age_l96_96955


namespace inscribed_circle_quadrilateral_l96_96430

theorem inscribed_circle_quadrilateral
  (AB CD BC AD AC BD E : ℝ)
  (r1 r2 r3 r4 : ℝ)
  (h1 : BC = AD)
  (h2 : AB + CD = BC + AD)
  (h3 : ∃ E, ∃ AC BD, AC * BD = E∧ AC > 0 ∧ BD > 0)
  (h_r1 : r1 > 0)
  (h_r2 : r2 > 0)
  (h_r3 : r3 > 0)
  (h_r4 : r4 > 0):
  1 / r1 + 1 / r3 = 1 / r2 + 1 / r4 := 
by
  sorry

end inscribed_circle_quadrilateral_l96_96430


namespace inequality_x_y_l96_96845

theorem inequality_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + x * y = 3) : x + y ≥ 2 := 
  sorry

end inequality_x_y_l96_96845


namespace find_value_of_d_l96_96278

theorem find_value_of_d
  (a b c d : ℕ) 
  (h1 : 0 < a) 
  (h2 : a < b) 
  (h3 : b < c) 
  (h4 : c < d) 
  (h5 : ab + bc + ac = abc) 
  (h6 : abc = d) : 
  d = 36 := 
sorry

end find_value_of_d_l96_96278


namespace triangle_angle_bisector_sum_l96_96736

theorem triangle_angle_bisector_sum (P Q R : ℝ × ℝ)
  (hP : P = (-8, 5)) (hQ : Q = (-15, -19)) (hR : R = (1, -7)) 
  (a b c : ℕ) (h : a + c = 89) 
  (gcd_abc : Int.gcd (Int.gcd a b) c = 1) :
  a + c = 89 :=
by
  sorry

end triangle_angle_bisector_sum_l96_96736


namespace number_of_meetings_l96_96920

-- Define the data for the problem
def pool_length : ℕ := 120
def swimmer_A_speed : ℕ := 4
def swimmer_B_speed : ℕ := 3
def total_time_seconds : ℕ := 15 * 60
def swimmer_A_turn_break_seconds : ℕ := 2
def swimmer_B_turn_break_seconds : ℕ := 0

-- Define the round trip time for each swimmer
def swimmer_A_round_trip_time : ℕ := 2 * (pool_length / swimmer_A_speed) + 2 * swimmer_A_turn_break_seconds
def swimmer_B_round_trip_time : ℕ := 2 * (pool_length / swimmer_B_speed) + 2 * swimmer_B_turn_break_seconds

-- Define the least common multiple of the round trip times
def lcm_round_trip_time : ℕ := Nat.lcm swimmer_A_round_trip_time swimmer_B_round_trip_time

-- Define the statement to prove
theorem number_of_meetings (lcm_round_trip_time : ℕ) : 
  (24 * (total_time_seconds / lcm_round_trip_time) + ((total_time_seconds % lcm_round_trip_time) / (pool_length / (swimmer_A_speed + swimmer_B_speed)))) = 51 := 
sorry

end number_of_meetings_l96_96920


namespace average_cd_l96_96006

theorem average_cd (c d : ℝ) (h : (4 + 6 + 8 + c + d) / 5 = 18) : (c + d) / 2 = 36 := 
by
  -- The proof goes here
  sorry

end average_cd_l96_96006


namespace tan_monotone_increasing_interval_l96_96629

theorem tan_monotone_increasing_interval :
  ∀ k : ℤ, ∀ x : ℝ, 
  (-π / 2 + k * π < x + π / 4 ∧ x + π / 4 < π / 2 + k * π) ↔
  (k * π - 3 * π / 4 < x ∧ x < k * π + π / 4) :=
by sorry

end tan_monotone_increasing_interval_l96_96629


namespace max_min_values_l96_96363

noncomputable def y (x : ℝ) : ℝ :=
  3 - 4 * Real.sin x - 4 * (Real.cos x)^2

theorem max_min_values :
  (∀ k : ℤ, y (- (Real.pi/2) + 2 * k * Real.pi) = 7) ∧
  (∀ k : ℤ, y (Real.pi/6 + 2 * k * Real.pi) = -2) ∧
  (∀ k : ℤ, y (5 * Real.pi/6 + 2 * k * Real.pi) = -2) := by
  sorry

end max_min_values_l96_96363


namespace candies_problem_l96_96190

theorem candies_problem (emily jennifer bob : ℕ) (h1 : emily = 6) 
  (h2 : jennifer = 2 * emily) (h3 : jennifer = 3 * bob) : bob = 4 := by
  -- Lean code to skip the proof
  sorry

end candies_problem_l96_96190


namespace bug_return_probability_twelfth_move_l96_96321

-- Conditions
def P : ℕ → ℚ
| 0       => 1
| (n + 1) => (1 : ℚ) / 3 * (1 - P n)

theorem bug_return_probability_twelfth_move :
  P 12 = 14762 / 59049 := by
sorry

end bug_return_probability_twelfth_move_l96_96321


namespace B_initial_investment_l96_96684

-- Definitions for investments and conditions
def A_init_invest : Real := 3000
def A_later_invest := 2 * A_init_invest

def A_yearly_investment := (A_init_invest * 6) + (A_later_invest * 6)

-- The amount B needs to invest for the yearly investment to be equal in the profit ratio 1:1
def B_investment (x : Real) := x * 12 

-- Definition of the proof problem
theorem B_initial_investment (x : Real) : A_yearly_investment = B_investment x → x = 4500 := 
by 
  sorry

end B_initial_investment_l96_96684


namespace relationship_among_a_b_c_l96_96874

noncomputable def a : ℝ := Real.log 4 / Real.log 5
noncomputable def b : ℝ := (Real.log 3 / Real.log 5)^2
noncomputable def c : ℝ := Real.log 5 / Real.log 4

theorem relationship_among_a_b_c : b < a ∧ a < c := by
  sorry

end relationship_among_a_b_c_l96_96874


namespace complement_U_A_l96_96704

def U := {x : ℝ | x < 2}
def A := {x : ℝ | x^2 < x}

theorem complement_U_A :
  (U \ A) = {x : ℝ | x ≤ 0 ∨ (1 ≤ x ∧ x < 2)} :=
sorry

end complement_U_A_l96_96704


namespace sum_of_squares_l96_96971

theorem sum_of_squares :
  ∃ p q r s t u : ℤ, (∀ x : ℤ, 729 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) ∧ 
    (p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210) :=
sorry

end sum_of_squares_l96_96971


namespace loss_per_metre_l96_96552

-- Definitions for given conditions
def TSP : ℕ := 15000           -- Total Selling Price
def CPM : ℕ := 40              -- Cost Price per Metre
def TMS : ℕ := 500             -- Total Metres Sold

-- Definition for the expected Loss Per Metre
def LPM : ℕ := 10

-- Statement to prove that the loss per metre is 10
theorem loss_per_metre :
  (CPM * TMS - TSP) / TMS = LPM :=
by
sorry

end loss_per_metre_l96_96552


namespace sam_dimes_example_l96_96547

theorem sam_dimes_example (x y : ℕ) (h₁ : x = 9) (h₂ : y = 7) : x + y = 16 :=
by 
  sorry

end sam_dimes_example_l96_96547


namespace smallest_ninequality_l96_96083

theorem smallest_ninequality 
  (n : ℕ) 
  (h : ∀ x : ℝ, (Real.sin x) ^ n + (Real.cos x) ^ n ≤ 2 ^ (1 - n)) : 
  n = 2 := 
by
  sorry

end smallest_ninequality_l96_96083


namespace Laura_running_speed_l96_96023

noncomputable def running_speed (x : ℝ) :=
  let biking_time := 30 / (3 * x + 2)
  let running_time := 10 / x
  let total_time := biking_time + running_time
  total_time = 3

theorem Laura_running_speed : ∃ x : ℝ, running_speed x ∧ abs (x - 6.35) < 0.01 :=
sorry

end Laura_running_speed_l96_96023


namespace find_savings_l96_96894

-- Define the problem statement
def income_expenditure_problem (income expenditure : ℝ) (ratio : ℝ) : Prop :=
  (income / ratio = expenditure) ∧ (income = 20000)

-- Define the theorem for savings
theorem find_savings (income expenditure : ℝ) (ratio : ℝ) (h_ratio : ratio = 4 / 5) (h_income : income = 20000) : 
  income_expenditure_problem income expenditure ratio → income - expenditure = 4000 :=
by
  sorry

end find_savings_l96_96894


namespace c_is_11_years_younger_than_a_l96_96673

variable (A B C : ℕ) (h : A + B = B + C + 11)

theorem c_is_11_years_younger_than_a (A B C : ℕ) (h : A + B = B + C + 11) : C = A - 11 := by
  sorry

end c_is_11_years_younger_than_a_l96_96673


namespace bobby_consumption_l96_96104

theorem bobby_consumption :
  let initial_candy := 28
  let additional_candy_portion := 3/4 * 42
  let chocolate_portion := 1/2 * 63
  initial_candy + additional_candy_portion + chocolate_portion = 91 := 
by {
  let initial_candy : ℝ := 28
  let additional_candy_portion : ℝ := 3/4 * 42
  let chocolate_portion : ℝ := 1/2 * 63
  sorry
}

end bobby_consumption_l96_96104


namespace sufficient_but_not_necessary_condition_l96_96943

-- Define the conditions as predicates
def p (x : ℝ) : Prop := x^2 - 3 * x - 4 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 6 * x + 9 - m^2 ≤ 0

-- Range for m where p is sufficient but not necessary for q
def m_range (m : ℝ) : Prop := m ≤ -4 ∨ m ≥ 4

-- The main goal to be proven
theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x, p x → q x m) ∧ ¬(∀ x, q x m → p x) ↔ m_range m :=
sorry

end sufficient_but_not_necessary_condition_l96_96943


namespace magazine_cost_l96_96139

theorem magazine_cost (C M : ℝ) 
  (h1 : 4 * C = 8 * M) 
  (h2 : 12 * C = 24) : 
  M = 1 :=
by
  sorry

end magazine_cost_l96_96139


namespace tangent_line_solution_l96_96818

variables (x y : ℝ)

noncomputable def circle_equation (m : ℝ) : Prop :=
  x^2 + y^2 - 4 * x + m * y = 0

def point_on_circle (m : ℝ) : Prop :=
  circle_equation 1 1 m

noncomputable def tangent_line_equation (x y : ℝ) : Prop :=
  x - 2 * y + 1 = 0

theorem tangent_line_solution (m : ℝ) :
  point_on_circle m →
  m = 2 →
  tangent_line_equation 1 1 :=
by
  sorry

end tangent_line_solution_l96_96818


namespace pirates_total_coins_l96_96458

theorem pirates_total_coins (x : ℕ) (h : (x * (x + 1)) / 2 = 5 * x) : 6 * x = 54 := by
  -- The proof will go here, but it's currently omitted with 'sorry'
  sorry

end pirates_total_coins_l96_96458


namespace g_inverse_sum_l96_96264

-- Define the function g and its inverse
def g (x : ℝ) : ℝ := x ^ 3
noncomputable def g_inv (y : ℝ) : ℝ := y ^ (1/3 : ℝ)

-- State the theorem to be proved
theorem g_inverse_sum : g_inv 8 + g_inv (-64) = -2 := by 
  sorry

end g_inverse_sum_l96_96264


namespace number_of_valid_5_digit_numbers_l96_96452

def is_multiple_of_16 (n : Nat) : Prop := 
  n % 16 = 0

theorem number_of_valid_5_digit_numbers : Nat := 
  sorry

example : number_of_valid_5_digit_numbers = 90 :=
  sorry

end number_of_valid_5_digit_numbers_l96_96452


namespace mean_after_removal_l96_96791

variable {n : ℕ}
variable {S : ℝ}
variable {S' : ℝ}
variable {mean_original : ℝ}
variable {size_original : ℕ}
variable {x1 : ℝ}
variable {x2 : ℝ}

theorem mean_after_removal (h_mean_original : mean_original = 42)
    (h_size_original : size_original = 60)
    (h_x1 : x1 = 50)
    (h_x2 : x2 = 60)
    (h_S : S = mean_original * size_original)
    (h_S' : S' = S - (x1 + x2)) :
    S' / (size_original - 2) = 41.55 :=
by
  sorry

end mean_after_removal_l96_96791


namespace largest_k_consecutive_sum_l96_96461

theorem largest_k_consecutive_sum (k : ℕ) (h1 : (∃ n : ℕ, 3^12 = k * n + (k*(k-1))/2)) : k ≤ 729 :=
by
  -- Proof omitted for brevity
  sorry

end largest_k_consecutive_sum_l96_96461


namespace ball_bounce_height_l96_96009

theorem ball_bounce_height (b : ℕ) : 
  ∃ b : ℕ, 400 * (3 / 4 : ℝ)^b < 50 ∧ ∀ b' < b, 400 * (3 / 4 : ℝ)^b' ≥ 50 :=
sorry

end ball_bounce_height_l96_96009


namespace find_common_ratio_l96_96347

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

axiom a2 : a 2 = 9
axiom a3_plus_a4 : a 3 + a 4 = 18
axiom q_not_one : q ≠ 1

-- Proof problem
theorem find_common_ratio
  (h : is_geometric_sequence a q)
  (ha2 : a 2 = 9)
  (ha3a4 : a 3 + a 4 = 18)
  (hq : q ≠ 1) :
  q = -2 :=
sorry

end find_common_ratio_l96_96347


namespace train_speed_correct_l96_96548

-- Define the length of the train
def train_length : ℝ := 200

-- Define the time taken to cross the telegraph post
def cross_time : ℝ := 8

-- Define the expected speed of the train
def expected_speed : ℝ := 25

-- Prove that the speed of the train is as expected
theorem train_speed_correct (length time : ℝ) (h_length : length = train_length) (h_time : time = cross_time) : 
  (length / time = expected_speed) :=
by
  rw [h_length, h_time]
  sorry

end train_speed_correct_l96_96548


namespace darnell_texts_l96_96693

theorem darnell_texts (T : ℕ) (unlimited_plan_cost alternative_text_cost alternative_call_cost : ℕ) 
    (call_minutes : ℕ) (cost_difference : ℕ) :
    unlimited_plan_cost = 12 →
    alternative_text_cost = 1 →
    alternative_call_cost = 3 →
    call_minutes = 60 →
    cost_difference = 1 →
    (alternative_text_cost * T / 30 + alternative_call_cost * call_minutes / 20) = 
      unlimited_plan_cost - cost_difference →
    T = 60 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end darnell_texts_l96_96693


namespace polynomial_sum_l96_96856

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l96_96856


namespace find_n_l96_96768

theorem find_n (x : ℝ) (h1 : x = 596.95) (h2 : ∃ n : ℝ, n + 11.95 - x = 3054) : ∃ n : ℝ, n = 3639 :=
by
  sorry

end find_n_l96_96768


namespace mary_fruits_l96_96982

noncomputable def totalFruitsLeft 
    (initial_apples: ℕ) (initial_oranges: ℕ) (initial_blueberries: ℕ) (initial_grapes: ℕ) (initial_kiwis: ℕ)
    (salad_apples: ℕ) (salad_oranges: ℕ) (salad_blueberries: ℕ)
    (snack_apples: ℕ) (snack_oranges: ℕ) (snack_kiwis: ℕ)
    (given_apples: ℕ) (given_oranges: ℕ) (given_blueberries: ℕ) (given_grapes: ℕ) (given_kiwis: ℕ) : ℕ :=
  let remaining_apples := initial_apples - salad_apples - snack_apples - given_apples
  let remaining_oranges := initial_oranges - salad_oranges - snack_oranges - given_oranges
  let remaining_blueberries := initial_blueberries - salad_blueberries - given_blueberries
  let remaining_grapes := initial_grapes - given_grapes
  let remaining_kiwis := initial_kiwis - snack_kiwis - given_kiwis
  remaining_apples + remaining_oranges + remaining_blueberries + remaining_grapes + remaining_kiwis

theorem mary_fruits :
    totalFruitsLeft 26 35 18 12 22 6 10 8 2 3 1 5 7 4 3 3 = 61 := by
  sorry

end mary_fruits_l96_96982


namespace arun_weight_average_l96_96582

theorem arun_weight_average (w : ℝ) 
  (h1 : 64 < w ∧ w < 72) 
  (h2 : 60 < w ∧ w < 70) 
  (h3 : w ≤ 67) : 
  (64 + 67) / 2 = 65.5 := 
  by sorry

end arun_weight_average_l96_96582


namespace measure_angle_C_l96_96968

noncomputable def triangle_angles_sum (a b c : ℝ) : Prop :=
  a + b + c = 180

noncomputable def angle_B_eq_twice_angle_C (b c : ℝ) : Prop :=
  b = 2 * c

noncomputable def angle_A_eq_40 : ℝ := 40

theorem measure_angle_C :
  ∀ (B C : ℝ), triangle_angles_sum angle_A_eq_40 B C → angle_B_eq_twice_angle_C B C → C = 140 / 3 :=
by
  intros B C h1 h2
  sorry

end measure_angle_C_l96_96968


namespace same_terminal_side_angles_l96_96145

theorem same_terminal_side_angles (α : ℝ) : 
  (∃ k : ℤ, α = -457 + k * 360) ↔ (∃ k : ℤ, α = 263 + k * 360) :=
sorry

end same_terminal_side_angles_l96_96145


namespace rotated_point_l96_96754

def point := (ℝ × ℝ × ℝ)

def rotate_point (A P : point) (θ : ℝ) : point :=
  -- Function implementing the rotation (the full definition would normally be placed here)
  sorry

def A : point := (1, 1, 1)
def P : point := (1, 1, 0)

theorem rotated_point (θ : ℝ) (hθ : θ = 60) :
  rotate_point A P θ = (1/3, 4/3, 1/3) :=
sorry

end rotated_point_l96_96754


namespace washing_whiteboards_l96_96379

/-- Define the conditions from the problem:
1. Four kids can wash three whiteboards in 20 minutes.
2. It takes one kid 160 minutes to wash a certain number of whiteboards. -/
def four_kids_wash_in_20_min : ℕ := 3
def time_per_batch : ℕ := 20
def one_kid_time : ℕ := 160
def intervals : ℕ := one_kid_time / time_per_batch

/-- Proving the answer based on the conditions:
one kid can wash six whiteboards in 160 minutes given these conditions. -/
theorem washing_whiteboards : intervals * (four_kids_wash_in_20_min / 4) = 6 :=
by
  sorry

end washing_whiteboards_l96_96379


namespace dan_gave_marbles_l96_96619

-- Conditions as definitions in Lean 4
def original_marbles : ℕ := 64
def marbles_left : ℕ := 50
def marbles_given : ℕ := original_marbles - marbles_left

-- Theorem statement proving the question == answer given the conditions.
theorem dan_gave_marbles : marbles_given = 14 := by
  sorry

end dan_gave_marbles_l96_96619


namespace gopi_turbans_annual_salary_l96_96032

variable (T : ℕ) (annual_salary_turbans : ℕ)
variable (annual_salary_money : ℕ := 90)
variable (months_worked : ℕ := 9)
variable (total_months_in_year : ℕ := 12)
variable (received_money : ℕ := 55)
variable (turban_price : ℕ := 50)
variable (received_turbans : ℕ := 1)
variable (servant_share_fraction : ℚ := 3 / 4)

theorem gopi_turbans_annual_salary 
    (annual_salary_turbans : ℕ)
    (H : (servant_share_fraction * (annual_salary_money + turban_price * annual_salary_turbans) = received_money + turban_price * received_turbans))
    : annual_salary_turbans = 1 :=
sorry

end gopi_turbans_annual_salary_l96_96032


namespace solution_set_of_inequality_l96_96652

theorem solution_set_of_inequality :
  {x : ℝ | (x - 1) * (2 - x) ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end solution_set_of_inequality_l96_96652


namespace solve_inequality_l96_96000

theorem solve_inequality (x : ℝ) : -7/3 < x ∧ x < 7 → |x+2| + |x-2| < x + 7 :=
by
  intro h
  sorry

end solve_inequality_l96_96000


namespace log_inequality_l96_96553

theorem log_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) : 
  Real.log b / Real.log a + Real.log a / Real.log b ≤ -2 := sorry

end log_inequality_l96_96553


namespace part1_part2_l96_96919

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

end part1_part2_l96_96919


namespace unknown_number_l96_96208

theorem unknown_number (n : ℕ) (h1 : Nat.lcm 24 n = 168) (h2 : Nat.gcd 24 n = 4) : n = 28 :=
by
  sorry

end unknown_number_l96_96208


namespace sufficient_condition_frac_ineq_inequality_transformation_problem_equivalence_l96_96543

theorem sufficient_condition_frac_ineq (x : ℝ) : (1 < x ∧ x < 2) → ( (x + 1) / (x - 1) > 2) :=
by
  -- Given that 1 < x and x < 2, we need to show (x + 1) / (x - 1) > 2
  sorry

theorem inequality_transformation (x : ℝ) : ( (x + 1) / (x - 1) > 2) ↔ ( (x - 1) * (x - 3) < 0 ) :=
by
  -- Prove that (x + 1) / (x - 1) > 2 is equivalent to (x - 1)(x - 3) < 0
  sorry

theorem problem_equivalence (x : ℝ) : ( (x + 1) / (x - 1) > 2) → (1 < x ∧ x < 3) :=
by
  -- Prove that (x + 1) / (x - 1) > 2 implies 1 < x < 3
  sorry

end sufficient_condition_frac_ineq_inequality_transformation_problem_equivalence_l96_96543


namespace initial_chips_in_bag_l96_96515

-- Definitions based on conditions
def chips_given_to_brother : ℕ := 7
def chips_given_to_sister : ℕ := 5
def chips_kept_by_nancy : ℕ := 10

-- Theorem statement
theorem initial_chips_in_bag (total_chips := chips_given_to_brother + chips_given_to_sister + chips_kept_by_nancy) : total_chips = 22 := 
by 
  -- we state the assertion
  sorry

end initial_chips_in_bag_l96_96515


namespace kay_exercise_time_l96_96109

variable (A W : ℕ)
variable (exercise_total : A + W = 250) 
variable (ratio_condition : A * 2 = 3 * W)

theorem kay_exercise_time :
  A = 150 ∧ W = 100 :=
by
  sorry

end kay_exercise_time_l96_96109


namespace james_received_stickers_l96_96647

theorem james_received_stickers (initial_stickers given_away final_stickers received_stickers : ℕ) 
  (h_initial : initial_stickers = 269)
  (h_given : given_away = 48)
  (h_final : final_stickers = 423)
  (h_total_before_giving_away : initial_stickers + received_stickers = given_away + final_stickers) :
  received_stickers = 202 :=
by
  sorry

end james_received_stickers_l96_96647


namespace problem_solution_l96_96065

noncomputable def quadratic_symmetric_b (a : ℝ) : ℝ :=
  2 * (1 - a)

theorem problem_solution (a : ℝ) (h1 : quadratic_symmetric_b a = 6) :
  b = 6 :=
by
  sorry

end problem_solution_l96_96065


namespace value_of_product_l96_96072

theorem value_of_product (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 2) : (x + 2) * (y + 2) = 16 := by
  sorry

end value_of_product_l96_96072


namespace problem_l96_96578

variable {x y : ℝ}

theorem problem (h : x < y) : 3 - x > 3 - y :=
sorry

end problem_l96_96578


namespace total_distance_walked_l96_96725

-- Define the conditions
def walking_rate : ℝ := 4
def time_before_break : ℝ := 2
def time_after_break : ℝ := 0.5

-- Define the required theorem
theorem total_distance_walked : 
  walking_rate * time_before_break + walking_rate * time_after_break = 10 := 
sorry

end total_distance_walked_l96_96725


namespace two_numbers_are_opposites_l96_96692

theorem two_numbers_are_opposites (x y z : ℝ) (h : (1 / x) + (1 / y) + (1 / z) = 1 / (x + y + z)) :
  (x + y = 0) ∨ (x + z = 0) ∨ (y + z = 0) :=
by
  sorry

end two_numbers_are_opposites_l96_96692


namespace eval_f_at_3_l96_96492

-- Define the polynomial function
def f (x : ℝ) : ℝ := 3 * x^3 - 5 * x^2 + 2 * x - 1

-- State the theorem to prove f(3) = 41
theorem eval_f_at_3 : f 3 = 41 :=
by
  -- Proof would go here
  sorry

end eval_f_at_3_l96_96492


namespace AMHSE_1988_l96_96481

theorem AMHSE_1988 (x y : ℝ) (h1 : |x| + x + y = 10) (h2 : x + |y| - y = 12) : x + y = 18 / 5 :=
sorry

end AMHSE_1988_l96_96481


namespace cellphone_gifting_l96_96294

theorem cellphone_gifting (n m : ℕ) (h1 : n = 20) (h2 : m = 3) : 
    (Finset.range n).card * (Finset.range (n - 1)).card * (Finset.range (n - 2)).card = 6840 := by
  sorry

end cellphone_gifting_l96_96294


namespace simplify_x_cubed_simplify_expr_l96_96451

theorem simplify_x_cubed (x : ℝ) : x * (x + 3) * (x + 5) = x^3 + 8 * x^2 + 15 * x := by
  sorry

theorem simplify_expr (x y : ℝ) : (5 * x + 2 * y) * (5 * x - 2 * y) - 5 * x * (5 * x - 3 * y) = -4 * y^2 + 15 * x * y := by
  sorry

end simplify_x_cubed_simplify_expr_l96_96451


namespace heptagon_isosceles_same_color_l96_96985

theorem heptagon_isosceles_same_color 
  (color : Fin 7 → Prop) (red blue : Prop)
  (h_heptagon : ∀ i : Fin 7, color i = red ∨ color i = blue) :
  ∃ (i j k : Fin 7), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ color i = color j ∧ color j = color k ∧ ((i + j) % 7 = k ∨ (j + k) % 7 = i ∨ (k + i) % 7 = j) :=
sorry

end heptagon_isosceles_same_color_l96_96985


namespace side_length_of_base_l96_96719

variable (s : ℕ) -- side length of the square base
variable (A : ℕ) -- area of one lateral face
variable (h : ℕ) -- slant height

-- Given conditions
def area_of_lateral_face (s h : ℕ) : ℕ := 20 * s

axiom lateral_face_area_given : A = 120
axiom slant_height_given : h = 40

theorem side_length_of_base (A : ℕ) (h : ℕ) (s : ℕ) : 20 * s = A → s = 6 :=
by
  -- The proof part is omitted, only required the statement as per guidelines
  sorry

end side_length_of_base_l96_96719


namespace intersect_is_one_l96_96045

def SetA : Set ℝ := {x | 0 < x ∧ x < 2}

def SetB : Set ℝ := {0, 1, 2, 3}

theorem intersect_is_one : SetA ∩ SetB = {1} :=
by
  sorry

end intersect_is_one_l96_96045


namespace find_other_number_l96_96729

theorem find_other_number
  (x y lcm hcf : ℕ)
  (h_lcm : Nat.lcm x y = lcm)
  (h_hcf : Nat.gcd x y = hcf)
  (h_x : x = 462)
  (h_lcm_value : lcm = 2310)
  (h_hcf_value : hcf = 30) :
  y = 150 :=
by
  sorry

end find_other_number_l96_96729


namespace division_result_l96_96123

theorem division_result : 180 / 6 / 3 / 2 = 5 := by
  sorry

end division_result_l96_96123


namespace trigonometric_identity_l96_96808

theorem trigonometric_identity (t : ℝ) : 
  5.43 * Real.cos (22 * Real.pi / 180 - t) * Real.cos (82 * Real.pi / 180 - t) +
  Real.cos (112 * Real.pi / 180 - t) * Real.cos (172 * Real.pi / 180 - t) = 
  0.5 * (Real.sin t + Real.cos t) :=
sorry

end trigonometric_identity_l96_96808


namespace find_n_l96_96743

   theorem find_n (n : ℕ) : 
     (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 3 * x + 3 * y + z = n) → 
     (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 3 * x + 3 * y + z = n) → 
     (n = 34 ∨ n = 37) :=
   by
     intros
     sorry
   
end find_n_l96_96743


namespace quadratic_eq_with_given_roots_l96_96483

theorem quadratic_eq_with_given_roots (a b : ℝ) (h1 : (a + b) / 2 = 8) (h2 : Real.sqrt (a * b) = 12) :
    (a + b = 16) ∧ (a * b = 144) ∧ (∀ (x : ℝ), x^2 - (a + b) * x + (a * b) = 0 ↔ x^2 - 16 * x + 144 = 0) := by
  sorry

end quadratic_eq_with_given_roots_l96_96483


namespace monotonic_increasing_f_l96_96426

theorem monotonic_increasing_f (f g : ℝ → ℝ) (hf : ∀ x, f (-x) = -f x) 
  (hg : ∀ x, g (-x) = g x) (hfg : ∀ x, f x + g x = 3^x) :
  ∀ a b : ℝ, a > b → f a > f b :=
sorry

end monotonic_increasing_f_l96_96426


namespace perpendicular_lines_iff_l96_96316

theorem perpendicular_lines_iff (a : ℝ) : 
  (∀ b₁ b₂ : ℝ, b₁ ≠ b₂ → ¬ (∀ x : ℝ, a * x + b₁ = (a - 2) * x + b₂) ∧ 
   (a * (a - 2) = -1)) ↔ a = 1 :=
by
  sorry

end perpendicular_lines_iff_l96_96316


namespace counting_error_l96_96182

theorem counting_error
  (b g : ℕ)
  (initial_balloons := 5 * b + 4 * g)
  (popped_balloons := g + 2 * b)
  (remaining_balloons := initial_balloons - popped_balloons)
  (Dima_count := 100) :
  remaining_balloons ≠ Dima_count := by
  sorry

end counting_error_l96_96182


namespace free_endpoints_eq_1001_l96_96467

theorem free_endpoints_eq_1001 : 
  ∃ k : ℕ, 1 + 4 * k = 1001 :=
by {
  sorry
}

end free_endpoints_eq_1001_l96_96467


namespace intersection_of_A_and_B_l96_96207

def A : Set ℝ := {x | x - 1 > 1}
def B : Set ℝ := {x | x < 3}

theorem intersection_of_A_and_B : (A ∩ B) = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_of_A_and_B_l96_96207


namespace number_142857_has_property_l96_96644

noncomputable def has_desired_property (n : ℕ) : Prop :=
∀ m ∈ [1, 2, 3, 4, 5, 6], ∀ d ∈ (Nat.digits 10 (n * m)), d ∈ (Nat.digits 10 n)

theorem number_142857_has_property : has_desired_property 142857 :=
sorry

end number_142857_has_property_l96_96644


namespace find_g_l96_96946

theorem find_g (x : ℝ) (g : ℝ → ℝ) :
  2 * x^5 - 4 * x^3 + 3 * x^2 + g x = 7 * x^4 - 5 * x^3 + x^2 - 9 * x + 2 →
  g x = -2 * x^5 + 7 * x^4 - x^3 - 2 * x^2 - 9 * x + 2 :=
by
  intro h
  sorry

end find_g_l96_96946


namespace remaining_lawn_mowing_l96_96140

-- Definitions based on the conditions in the problem.
def Mary_mowing_time : ℝ := 3  -- Mary can mow the lawn in 3 hours
def John_mowing_time : ℝ := 6  -- John can mow the lawn in 6 hours
def John_work_time : ℝ := 3    -- John works for 3 hours

-- Question: How much of the lawn remains to be mowed?
theorem remaining_lawn_mowing : (Mary_mowing_time = 3) ∧ (John_mowing_time = 6) ∧ (John_work_time = 3) →
  (1 - (John_work_time / John_mowing_time) = 1 / 2) :=
by
  sorry

end remaining_lawn_mowing_l96_96140


namespace mowing_difference_l96_96563

-- Define the number of times mowed in spring and summer
def mowedSpring : ℕ := 8
def mowedSummer : ℕ := 5

-- Prove the difference between spring and summer mowing is 3
theorem mowing_difference : mowedSpring - mowedSummer = 3 := by
  sorry

end mowing_difference_l96_96563


namespace instantaneous_velocity_at_1_l96_96773

noncomputable def S (t : ℝ) : ℝ := t^2 + 2 * t

theorem instantaneous_velocity_at_1 : (deriv S 1) = 4 :=
by 
  -- The proof is left as an exercise
  sorry

end instantaneous_velocity_at_1_l96_96773


namespace find_n_l96_96067

theorem find_n (n : ℕ) (x y a b : ℕ) (hx : x = 1) (hy : y = 1) (ha : a = 1) (hb : b = 1)
  (h : (x + 3 * y) ^ n = (7 * a + b) ^ 10) : n = 5 :=
by
  sorry

end find_n_l96_96067


namespace complex_fraction_eval_l96_96345

theorem complex_fraction_eval (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + a * b + b^2 = 0) :
  (a^15 + b^15) / (a + b)^15 = -2 := by
sorry

end complex_fraction_eval_l96_96345


namespace smallest_integer_mod_inverse_l96_96909

theorem smallest_integer_mod_inverse (n : ℕ) (h1 : n > 1) (h2 : gcd n 1001 = 1) : n = 2 :=
sorry

end smallest_integer_mod_inverse_l96_96909


namespace negation_of_exactly_one_even_l96_96283

variable (a b c : ℕ)

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def exactly_one_even (a b c : ℕ) : Prop :=
  (is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
  (¬ is_even a ∧ is_even b ∧ ¬ is_even c) ∨
  (¬ is_even a ∧ ¬ is_even b ∧ is_even c)

theorem negation_of_exactly_one_even :
  ¬ exactly_one_even a b c ↔ (¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
                                 (is_even a ∧ is_even b) ∨
                                 (is_even a ∧ is_even c) ∨
                                 (is_even b ∧ is_even c) :=
by sorry

end negation_of_exactly_one_even_l96_96283


namespace max_common_ratio_arithmetic_geometric_sequence_l96_96752

open Nat

theorem max_common_ratio_arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (k : ℕ) (q : ℝ) 
  (hk : k ≥ 2) (ha : ∀ n, a (n + 1) = a n + d)
  (hg : (a 1) * (a (2 * k)) = (a k) ^ 2) :
  q ≤ 2 :=
by
  sorry

end max_common_ratio_arithmetic_geometric_sequence_l96_96752


namespace multiply_correct_l96_96253

theorem multiply_correct : 2.4 * 0.2 = 0.48 := by
  sorry

end multiply_correct_l96_96253


namespace line_intersects_curve_l96_96559

theorem line_intersects_curve (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ ax₁ + 16 = x₁^3 ∧ ax₂ + 16 = x₂^3) →
  a = 12 :=
by
  sorry

end line_intersects_curve_l96_96559


namespace digit_divisibility_l96_96475

theorem digit_divisibility : 
  (∃ (A : ℕ), A < 10 ∧ 
   (4573198080 + A) % 2 = 0 ∧ 
   (4573198080 + A) % 5 = 0 ∧ 
   (4573198080 + A) % 8 = 0 ∧ 
   (4573198080 + A) % 10 = 0 ∧ 
   (4573198080 + A) % 16 = 0 ∧ A = 0) := 
by { use 0; sorry }

end digit_divisibility_l96_96475


namespace area_arccos_cos_eq_pi_sq_l96_96793

noncomputable def area_bounded_by_arccos_cos : ℝ :=
  ∫ x in (0 : ℝ)..2 * Real.pi, Real.arccos (Real.cos x)

theorem area_arccos_cos_eq_pi_sq :
  area_bounded_by_arccos_cos = Real.pi ^ 2 :=
sorry

end area_arccos_cos_eq_pi_sq_l96_96793


namespace shortest_path_l96_96118

noncomputable def diameter : ℝ := 18
noncomputable def radius : ℝ := diameter / 2
noncomputable def AC : ℝ := 7
noncomputable def BD : ℝ := 7
noncomputable def CD : ℝ := diameter - AC - BD
noncomputable def CP : ℝ := Real.sqrt (radius ^ 2 - (CD / 2) ^ 2)
noncomputable def DP : ℝ := CP

theorem shortest_path (C P D : ℝ) :
  (C - 7) ^ 2 + (D - 7) ^ 2 = CD ^ 2 →
  (C = AC) ∧ (D = BD) →
  2 * CP = 2 * Real.sqrt 77 :=
by
  intros h1 h2
  sorry

end shortest_path_l96_96118


namespace prime_for_all_k_l96_96433

theorem prime_for_all_k (n : ℕ) (h_n : n ≥ 2) (h_prime : ∀ k : ℕ, k ≤ Nat.sqrt (n / 3) → Prime (k^2 + k + n)) :
  ∀ k : ℕ, k ≤ n - 2 → Prime (k^2 + k + n) :=
by
  intros
  sorry

end prime_for_all_k_l96_96433


namespace b_2016_value_l96_96941

theorem b_2016_value : 
  ∃ (a b : ℕ → ℝ), 
    a 1 = 1 / 2 ∧ 
    (∀ n : ℕ, 0 < n → a n + b n = 1) ∧
    (∀ n : ℕ, 0 < n → b (n + 1) = b n / (1 - (a n)^2)) → 
    b 2016 = 2016 / 2017 :=
by
  sorry

end b_2016_value_l96_96941


namespace max_distance_with_optimal_tire_swapping_l96_96100

theorem max_distance_with_optimal_tire_swapping
  (front_tires_last : ℕ)
  (rear_tires_last : ℕ)
  (front_tires_last_eq : front_tires_last = 20000)
  (rear_tires_last_eq : rear_tires_last = 30000) :
  ∃ D : ℕ, D = 30000 :=
by
  sorry

end max_distance_with_optimal_tire_swapping_l96_96100


namespace max_area_of_rectangle_with_perimeter_60_l96_96950

theorem max_area_of_rectangle_with_perimeter_60 :
  ∃ (x y : ℝ), 2 * (x + y) = 60 ∧ x * y = 225 :=
by
  sorry

end max_area_of_rectangle_with_perimeter_60_l96_96950


namespace number_of_participants_l96_96570

-- Define the conditions and theorem
theorem number_of_participants (n : ℕ) (h : n * (n - 1) / 2 = 231) : n = 22 :=
  sorry

end number_of_participants_l96_96570


namespace find_x_in_triangle_l96_96742

theorem find_x_in_triangle 
  (P Q R S: Type) 
  (PQS_is_straight: PQS) 
  (angle_PQR: ℝ)
  (h1: angle_PQR = 110) 
  (angle_RQS : ℝ)
  (h2: angle_RQS = 70)
  (angle_QRS : ℝ)
  (h3: angle_QRS = 3 * angle_x)
  (angle_QSR : ℝ)
  (h4: angle_QSR = angle_x + 14) 
  (triangle_angles_sum : ∀ (a b c: ℝ), a + b + c = 180) : 
  angle_x = 24 :=
by
  sorry

end find_x_in_triangle_l96_96742


namespace ceil_evaluation_l96_96334

theorem ceil_evaluation : 
  (Int.ceil (((-7 : ℚ) / 4) ^ 2 - (1 / 8)) = 3) :=
sorry

end ceil_evaluation_l96_96334


namespace Faye_total_pencils_l96_96944

def pencils_per_row : ℕ := 8
def number_of_rows : ℕ := 4
def total_pencils : ℕ := pencils_per_row * number_of_rows

theorem Faye_total_pencils : total_pencils = 32 := by
  sorry

end Faye_total_pencils_l96_96944


namespace boris_possible_amount_l96_96462

theorem boris_possible_amount (k : ℕ) : ∃ k : ℕ, 1 + 74 * k = 823 :=
by
  use 11
  sorry

end boris_possible_amount_l96_96462


namespace find_correct_answer_l96_96064

theorem find_correct_answer (x : ℕ) (h : 3 * x = 135) : x / 3 = 15 :=
sorry

end find_correct_answer_l96_96064


namespace polynomial_roots_and_coefficients_l96_96522

theorem polynomial_roots_and_coefficients 
  (a b c d e : ℝ)
  (h1 : a = 2)
  (h2 : 256 * a + 64 * b + 16 * c + 4 * d + e = 0)
  (h3 : -81 * a + 27 * b - 9 * c + 3 * d + e = 0)
  (h4 : 625 * a + 125 * b + 25 * c + 5 * d + e = 0) :
  (b + c + d) / a = 151 := 
by
  sorry

end polynomial_roots_and_coefficients_l96_96522


namespace sqrt_fraction_equiv_l96_96866

-- Define the fractions
def frac1 : ℚ := 25 / 36
def frac2 : ℚ := 16 / 9

-- Define the expression under the square root
def sum_frac : ℚ := frac1 + (frac2 * 36 / 36)

-- State the problem
theorem sqrt_fraction_equiv : (Real.sqrt sum_frac) = Real.sqrt 89 / 6 :=
by
  -- Steps and proof are omitted; we use sorry to indicate the proof is skipped
  sorry

end sqrt_fraction_equiv_l96_96866


namespace dad_steps_eq_90_l96_96071

-- Define the conditions given in the problem
variables (masha_steps yasha_steps dad_steps : ℕ)

-- Conditions:
-- 1. Dad takes 3 steps while Masha takes 5 steps
-- 2. Masha takes 3 steps while Yasha takes 5 steps
-- 3. Together, Masha and Yasha made 400 steps
def conditions := dad_steps * 5 = 3 * masha_steps ∧ masha_steps * yasha_steps = 3 * yasha_steps ∧ 3 * yasha_steps = 400

-- Theorem stating the proof problem
theorem dad_steps_eq_90 : conditions masha_steps yasha_steps dad_steps → dad_steps = 90 :=
by
  sorry

end dad_steps_eq_90_l96_96071


namespace find_z_l96_96980

theorem find_z (x y : ℤ) (h1 : x * y + x + y = 106) (h2 : x^2 * y + x * y^2 = 1320) :
  x^2 + y^2 = 748 ∨ x^2 + y^2 = 5716 :=
sorry

end find_z_l96_96980


namespace polar_to_rectangular_l96_96814

noncomputable def curve_equation (θ : ℝ) : ℝ := 2 * Real.cos θ

theorem polar_to_rectangular (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ Real.pi / 2) :
  ∃ (x y : ℝ), (x - 1) ^ 2 + y ^ 2 = 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧
  (x = curve_equation θ * Real.cos θ ∧ y = curve_equation θ * Real.sin θ) :=
sorry

end polar_to_rectangular_l96_96814


namespace pencils_in_boxes_l96_96091

theorem pencils_in_boxes (total_pencils : ℕ) (pencils_per_box : ℕ) (boxes_required : ℕ) 
    (h1 : total_pencils = 648) (h2 : pencils_per_box = 4) : boxes_required = 162 :=
sorry

end pencils_in_boxes_l96_96091


namespace total_goals_in_five_matches_is_4_l96_96683

theorem total_goals_in_five_matches_is_4
    (A : ℚ) -- defining the average number of goals before the fifth match as rational
    (h1 : A * 4 + 2 = (A + 0.3) * 5) : -- condition representing total goals equation
    4 = (4 * A + 2) := -- statement that the total number of goals in 5 matches is 4
by
  sorry

end total_goals_in_five_matches_is_4_l96_96683


namespace miles_mike_l96_96235

def cost_mike (M : ℕ) : ℝ := 2.50 + 0.25 * M
def cost_annie (A : ℕ) : ℝ := 2.50 + 5.00 + 0.25 * A

theorem miles_mike {M A : ℕ} (annie_ride_miles : A = 16) (same_cost : cost_mike M = cost_annie A) : M = 36 :=
by
  rw [cost_annie, annie_ride_miles] at same_cost
  simp [cost_mike] at same_cost
  sorry

end miles_mike_l96_96235


namespace distinct_equilateral_triangles_in_polygon_l96_96301

noncomputable def num_distinct_equilateral_triangles (P : Finset (Fin 10)) : Nat :=
  90

theorem distinct_equilateral_triangles_in_polygon (P : Finset (Fin 10)) :
  P.card = 10 →
  num_distinct_equilateral_triangles P = 90 :=
by
  intros
  sorry

end distinct_equilateral_triangles_in_polygon_l96_96301


namespace product_mod_self_inverse_l96_96212

theorem product_mod_self_inverse 
  {n : ℕ} (hn : 0 < n) (a b : ℤ) (ha : a * a % n = 1) (hb : b * b % n = 1) :
  (a * b) % n = 1 := 
sorry

end product_mod_self_inverse_l96_96212


namespace circle_standard_equation_l96_96877

noncomputable def circle_equation (a : ℝ) : Prop :=
  ∃ (x y : ℝ), (x - a)^2 + y^2 = 1

theorem circle_standard_equation : circle_equation 2 := by
  sorry

end circle_standard_equation_l96_96877


namespace countFibSequences_l96_96047

-- Define what it means for a sequence to be Fibonacci-type
def isFibType (a : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, a n = a (n - 1) + a (n - 2)

-- Define a Fibonacci-type sequence condition with given constraints
def fibSeqCondition (a : ℤ → ℤ) (N : ℤ) : Prop :=
  isFibType a ∧ ∃ n : ℤ, 0 < a n ∧ a n ≤ N ∧ 0 < a (n + 1) ∧ a (n + 1) ≤ N

-- Main theorem
theorem countFibSequences (N : ℤ) :
  ∃ count : ℤ,
    (N % 2 = 0 → count = (N / 2) * (N / 2 + 1)) ∧
    (N % 2 = 1 → count = ((N + 1) / 2) ^ 2) ∧
    (∀ a : ℤ → ℤ, fibSeqCondition a N → (∃ n : ℤ, a n = count)) :=
by
  sorry

end countFibSequences_l96_96047


namespace smallest_N_l96_96988

-- Definitions corresponding to the conditions
def circular_table (chairs : ℕ) : Prop := chairs = 72

def proper_seating (N chairs : ℕ) : Prop :=
  ∀ (new_person : ℕ), new_person < chairs →
    (∃ seated, seated < N ∧ (seated - new_person).gcd chairs = 1)

-- Problem statement
theorem smallest_N (chairs : ℕ) :
  circular_table chairs →
  ∃ N, proper_seating N chairs ∧ (∀ M < N, ¬ proper_seating M chairs) ∧ N = 18 :=
by
  intro h
  sorry

end smallest_N_l96_96988


namespace correct_equation_l96_96181

variable (x : ℝ) (h1 : x > 0)

def length_pipeline : ℝ := 3000
def efficiency_increase : ℝ := 0.2
def days_ahead : ℝ := 10

theorem correct_equation :
  (length_pipeline / x) - (length_pipeline / ((1 + efficiency_increase) * x)) = days_ahead :=
by
  sorry

end correct_equation_l96_96181


namespace haleys_car_distance_l96_96993

theorem haleys_car_distance (fuel_ratio : ℕ) (distance_ratio : ℕ) (fuel_used : ℕ) (distance_covered : ℕ) 
   (h_ratio : fuel_ratio = 4) (h_distance_ratio : distance_ratio = 7) (h_fuel_used : fuel_used = 44) :
   distance_covered = 77 := by
  -- Proof to be filled in
  sorry

end haleys_car_distance_l96_96993


namespace least_five_digit_congruent_to_5_mod_15_l96_96764

theorem least_five_digit_congruent_to_5_mod_15 : ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 15 = 5 ∧ n = 10010 := by
  sorry

end least_five_digit_congruent_to_5_mod_15_l96_96764


namespace distance_between_stations_l96_96720

theorem distance_between_stations (x y t : ℝ) 
(start_same_hour : t > 0)
(speed_slow_train : ∀ t, x = 16 * t)
(speed_fast_train : ∀ t, y = 21 * t)
(distance_difference : y = x + 60) : 
  x + y = 444 := 
sorry

end distance_between_stations_l96_96720


namespace parabola_focus_l96_96328

theorem parabola_focus (p : ℝ) (hp : ∃ (p : ℝ), ∀ x y : ℝ, x^2 = 2 * p * y) : (∀ (hf : (0, 2) = (0, p / 2)), p = 4) :=
sorry

end parabola_focus_l96_96328


namespace quadratic_vertex_l96_96603

theorem quadratic_vertex (x y : ℝ) (h : y = -3 * x^2 + 2) : (x, y) = (0, 2) :=
sorry

end quadratic_vertex_l96_96603


namespace lowry_earnings_l96_96972

def small_bonsai_cost : ℕ := 30
def big_bonsai_cost : ℕ := 20
def small_bonsai_sold : ℕ := 3
def big_bonsai_sold : ℕ := 5

def total_earnings (small_cost : ℕ) (big_cost : ℕ) (small_sold : ℕ) (big_sold : ℕ) : ℕ :=
  small_cost * small_sold + big_cost * big_sold

theorem lowry_earnings :
  total_earnings small_bonsai_cost big_bonsai_cost small_bonsai_sold big_bonsai_sold = 190 := 
by
  sorry

end lowry_earnings_l96_96972


namespace P_Ravi_is_02_l96_96075

def P_Ram : ℚ := 6 / 7
def P_Ram_and_Ravi : ℚ := 0.17142857142857143

theorem P_Ravi_is_02 (P_Ravi : ℚ) : P_Ram_and_Ravi = P_Ram * P_Ravi → P_Ravi = 0.2 :=
by
  intro h
  sorry

end P_Ravi_is_02_l96_96075


namespace total_spending_l96_96756

theorem total_spending :
  let price_per_pencil := 0.20
  let tolu_pencils := 3
  let robert_pencils := 5
  let melissa_pencils := 2
  let tolu_cost := tolu_pencils * price_per_pencil
  let robert_cost := robert_pencils * price_per_pencil
  let melissa_cost := melissa_pencils * price_per_pencil
  let total_cost := tolu_cost + robert_cost + melissa_cost
  total_cost = 2.00 := by
  sorry

end total_spending_l96_96756


namespace cos_sq_sub_sin_sq_pi_div_12_l96_96728

theorem cos_sq_sub_sin_sq_pi_div_12 : 
  (Real.cos (π / 12))^2 - (Real.sin (π / 12))^2 = Real.cos (π / 6) :=
by
  sorry

end cos_sq_sub_sin_sq_pi_div_12_l96_96728


namespace inner_cube_surface_area_l96_96243

theorem inner_cube_surface_area (S_outer : ℝ) (h_outer : S_outer = 54) : 
  ∃ S_inner : ℝ, S_inner = 27 := by
  -- The proof will go here
  sorry

end inner_cube_surface_area_l96_96243


namespace proof_problem_l96_96251

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < (π / 2))
variable (hβ : 0 < β ∧ β < (π / 2))
variable (htan : tan α = (1 + sin β) / cos β)

theorem proof_problem : 2 * α - β = π / 2 :=
by
  sorry

end proof_problem_l96_96251


namespace compl_union_eq_l96_96596

-- Definitions
def U : Set ℤ := {x | 1 ≤ x ∧ x ≤ 6}
def A : Set ℤ := {1, 3, 4}
def B : Set ℤ := {2, 4}

-- The statement
theorem compl_union_eq : (Aᶜ ∩ U) ∪ B = {2, 4, 5, 6} :=
by sorry

end compl_union_eq_l96_96596


namespace problem_solution_l96_96454

theorem problem_solution (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (h : x - y = x / y) : 
  (1 / x - 1 / y = -1 / y^2) := 
by sorry

end problem_solution_l96_96454


namespace eq_frac_l96_96085

noncomputable def g : ℝ → ℝ := sorry

theorem eq_frac (h1 : ∀ c d : ℝ, c^3 * g d = d^3 * g c)
                (h2 : g 3 ≠ 0) : (g 7 - g 4) / g 3 = 279 / 27 :=
by
  sorry

end eq_frac_l96_96085


namespace jellybeans_initial_amount_l96_96870

theorem jellybeans_initial_amount (x : ℝ) 
  (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end jellybeans_initial_amount_l96_96870


namespace sqrt_factorial_product_l96_96424

theorem sqrt_factorial_product :
  Nat.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := 
sorry

end sqrt_factorial_product_l96_96424


namespace fill_time_l96_96477

def inflow_rate : ℕ := 24 -- gallons per second
def outflow_rate : ℕ := 4 -- gallons per second
def basin_volume : ℕ := 260 -- gallons

theorem fill_time (inflow_rate outflow_rate basin_volume : ℕ) (h₁ : inflow_rate = 24) (h₂ : outflow_rate = 4) 
  (h₃ : basin_volume = 260) : basin_volume / (inflow_rate - outflow_rate) = 13 :=
by
  sorry

end fill_time_l96_96477


namespace range_of_b_l96_96781

theorem range_of_b (a b c m : ℝ) (h_ge_seq : c = b * b / a) (h_sum : a + b + c = m) (h_pos_a : a > 0) (h_pos_m : m > 0) : 
  (-m ≤ b ∧ b < 0) ∨ (0 < b ∧ b ≤ m / 3) :=
by
  sorry

end range_of_b_l96_96781


namespace min_deg_q_l96_96616

-- Definitions of polynomials requirements
variables (p q r : Polynomial ℝ)

-- Given Conditions
def polynomials_relation : Prop := 5 * p + 6 * q = r
def deg_p : Prop := p.degree = 10
def deg_r : Prop := r.degree = 12

-- The main theorem we want to prove
theorem min_deg_q (h1 : polynomials_relation p q r) (h2 : deg_p p) (h3 : deg_r r) : q.degree ≥ 12 :=
sorry

end min_deg_q_l96_96616


namespace find_length_of_other_diagonal_l96_96296

theorem find_length_of_other_diagonal
  (area : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h1: area = 75)
  (h2: d1 = 10) :
  d2 = 15 :=
by 
  sorry

end find_length_of_other_diagonal_l96_96296


namespace smallest_positive_multiple_of_45_is_45_l96_96216

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l96_96216


namespace EmilySixthQuizScore_l96_96509

theorem EmilySixthQuizScore (x : ℕ) : 
  let scores := [85, 92, 88, 90, 93]
  let total_scores_with_x := scores.sum + x
  let desired_average := 91
  total_scores_with_x = 6 * desired_average → x = 98 := by
  sorry

end EmilySixthQuizScore_l96_96509


namespace students_correct_answers_l96_96792

theorem students_correct_answers
  (total_questions : ℕ)
  (correct_score per_question : ℕ)
  (incorrect_penalty : ℤ)
  (xiao_ming_score xiao_hong_score xiao_hua_score : ℤ)
  (xm_correct_answers xh_correct_answers xh_correct_answers : ℕ)
  (total : ℕ)
  (h_1 : total_questions = 10)
  (h_2 : correct_score = 10)
  (h_3 : incorrect_penalty = -3)
  (h_4 : xiao_ming_score = 87)
  (h_5 : xiao_hong_score = 74)
  (h_6 : xiao_hua_score = 9)
  (h_xm : xm_correct_answers = total_questions - (xiao_ming_score - total_questions * correct_score) / (correct_score - incorrect_penalty))
  (h_xh : xh_correct_answers = total_questions - (xiao_hong_score - total_questions * correct_score) / (correct_score - incorrect_penalty))
  (h_xh : xh_correct_answers = total_questions - (xiao_hua_score - total_questions * correct_score) / (correct_score - incorrect_penalty))
  (expected : total = 20) :
  xm_correct_answers + xh_correct_answers + xh_correct_answers = total := 
sorry

end students_correct_answers_l96_96792


namespace angle_tuvels_equiv_l96_96991

-- Defining the conditions
def full_circle_tuvels : ℕ := 400
def degree_angle_in_circle : ℕ := 360
def specific_angle_degrees : ℕ := 45

-- Proof statement showing the equivalence
theorem angle_tuvels_equiv :
  (specific_angle_degrees * full_circle_tuvels) / degree_angle_in_circle = 50 :=
by
  sorry

end angle_tuvels_equiv_l96_96991


namespace sqrt_conjecture_l96_96811

theorem sqrt_conjecture (n : ℕ) (h : n ≥ 1) : 
  (Real.sqrt (n + (1 / (n + 2)))) = ((n + 1) * Real.sqrt (1 / (n + 2))) :=
sorry

end sqrt_conjecture_l96_96811


namespace z_real_iff_z_complex_iff_z_pure_imaginary_iff_l96_96184

-- Definitions for the problem conditions
def z_real (m : ℝ) : Prop := (m^2 - 2 * m - 15 = 0)
def z_pure_imaginary (m : ℝ) : Prop := (m^2 - 9 * m - 36 = 0) ∧ (m^2 - 2 * m - 15 ≠ 0)

-- Question 1: Prove that z is a real number if and only if m = -3 or m = 5
theorem z_real_iff (m : ℝ) : z_real m ↔ m = -3 ∨ m = 5 := sorry

-- Question 2: Prove that z is a complex number with non-zero imaginary part if and only if m ≠ -3 and m ≠ 5
theorem z_complex_iff (m : ℝ) : ¬z_real m ↔ m ≠ -3 ∧ m ≠ 5 := sorry

-- Question 3: Prove that z is a pure imaginary number if and only if m = 12
theorem z_pure_imaginary_iff (m : ℝ) : z_pure_imaginary m ↔ m = 12 := sorry

end z_real_iff_z_complex_iff_z_pure_imaginary_iff_l96_96184


namespace positive_number_l96_96249

theorem positive_number (n : ℕ) (h : n^2 + 2 * n = 170) : n = 12 :=
sorry

end positive_number_l96_96249


namespace alpha_beta_sum_pi_over_2_l96_96121

theorem alpha_beta_sum_pi_over_2 (α β : ℝ) (hα : 0 < α) (hα_lt : α < π / 2) (hβ : 0 < β) (hβ_lt : β < π / 2) (h : Real.sin (α + β) = Real.sin α ^ 2 + Real.sin β ^ 2) : α + β = π / 2 :=
by
  -- Proof steps would go here
  sorry

end alpha_beta_sum_pi_over_2_l96_96121


namespace determine_a_l96_96966

noncomputable def imaginary_unit : ℂ := Complex.I

def is_on_y_axis (z : ℂ) : Prop :=
  z.re = 0

theorem determine_a (a : ℝ) : 
  is_on_y_axis (⟨(a - 3 * imaginary_unit.re), -(a - 3 * imaginary_unit.im)⟩ / ⟨(1 - imaginary_unit.re), -(1 - imaginary_unit.im)⟩) → 
  a = -3 :=
sorry

end determine_a_l96_96966


namespace tree_cost_calculation_l96_96183

theorem tree_cost_calculation :
  let c := 1500 -- park circumference in meters
  let i := 30 -- interval distance in meters
  let p := 5000 -- price per tree in mill
  let n := c / i -- number of trees
  let cost := n * p -- total cost in mill
  cost = 250000 :=
by
  sorry

end tree_cost_calculation_l96_96183


namespace even_function_value_l96_96234

theorem even_function_value (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x))
  (h_neg : ∀ x, x < 0 → f x = 2 ^ x) :
  f (Real.log 9 / Real.log 4) = 1 / 3 :=
by
  sorry

end even_function_value_l96_96234


namespace initial_money_l96_96562

def cost_of_game : Nat := 47
def cost_of_toy : Nat := 7
def number_of_toys : Nat := 3

theorem initial_money (initial_amount : Nat) (remaining_amount : Nat) :
  initial_amount = cost_of_game + remaining_amount →
  remaining_amount = number_of_toys * cost_of_toy →
  initial_amount = 68 := by
    sorry

end initial_money_l96_96562


namespace number_of_six_digit_numbers_formable_by_1_2_3_4_l96_96640

theorem number_of_six_digit_numbers_formable_by_1_2_3_4
  (digits : Finset ℕ := {1, 2, 3, 4})
  (pairs_count : ℕ := 2)
  (non_adjacent_pair : ℕ := 1)
  (adjacent_pair : ℕ := 1)
  (six_digit_numbers : ℕ := 432) :
  ∃ (n : ℕ), n = 432 :=
by
  -- Proof will go here
  sorry

end number_of_six_digit_numbers_formable_by_1_2_3_4_l96_96640


namespace select_7_jury_l96_96028

theorem select_7_jury (students : Finset ℕ) (jury : Finset ℕ)
  (likes : ℕ → Finset ℕ) (h_students : students.card = 100)
  (h_jury : jury.card = 25) (h_likes : ∀ s ∈ students, (likes s).card = 10) :
  ∃ (selected_jury : Finset ℕ), selected_jury.card = 7 ∧ ∀ s ∈ students, ∃ j ∈ selected_jury, j ∈ (likes s) :=
sorry

end select_7_jury_l96_96028


namespace foreign_objects_total_sum_l96_96996

-- define the conditions
def dog_burrs : Nat := 12
def dog_ticks := 6 * dog_burrs
def dog_fleas := 3 * dog_ticks

def cat_burrs := 2 * dog_burrs
def cat_ticks := dog_ticks / 3
def cat_fleas := 4 * cat_ticks

-- calculate the total foreign objects
def total_dog := dog_burrs + dog_ticks + dog_fleas
def total_cat := cat_burrs + cat_ticks + cat_fleas

def total_objects := total_dog + total_cat

-- state the theorem
theorem foreign_objects_total_sum : total_objects = 444 := by
  sorry

end foreign_objects_total_sum_l96_96996


namespace common_difference_is_1_over_10_l96_96365

open Real

noncomputable def a_n (a₁ d: ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def S_n (a₁ d : ℝ) (n : ℕ) : ℝ := 
  n * a₁ + (n * (n - 1)) * d / 2

theorem common_difference_is_1_over_10 (a₁ d : ℝ) 
  (h : (S_n a₁ d 2017 / 2017) - (S_n a₁ d 17 / 17) = 100) : 
  d = 1 / 10 :=
by
  sorry

end common_difference_is_1_over_10_l96_96365


namespace find_xy_l96_96323

-- Define the conditions as constants for clarity
def condition1 (x : ℝ) : Prop := 0.60 / x = 6 / 2
def condition2 (x y : ℝ) : Prop := x / y = 8 / 12

theorem find_xy (x y : ℝ) (hx : condition1 x) (hy : condition2 x y) : 
  x = 0.20 ∧ y = 0.30 :=
by
  sorry

end find_xy_l96_96323


namespace sequence_term_500_l96_96272

theorem sequence_term_500 (a : ℕ → ℤ) (h1 : a 1 = 3009) (h2 : a 2 = 3010) 
  (h3 : ∀ n : ℕ, 1 ≤ n → a n + a (n + 1) + a (n + 2) = 2 * n) : 
  a 500 = 3341 := 
sorry

end sequence_term_500_l96_96272


namespace calc_quotient_l96_96554

theorem calc_quotient (a b : ℕ) (h1 : a - b = 177) (h2 : 14^2 = 196) : (a - b)^2 / 196 = 144 := 
by sorry

end calc_quotient_l96_96554


namespace max_cities_visited_l96_96824

theorem max_cities_visited (n k : ℕ) : ∃ t, t = n - k :=
by
  sorry

end max_cities_visited_l96_96824


namespace find_the_number_l96_96948

noncomputable def special_expression (x : ℝ) : ℝ :=
  9 - 8 / x * 5 + 10

theorem find_the_number (x : ℝ) (h : special_expression x = 13.285714285714286) : x = 7 := by
  sorry

end find_the_number_l96_96948


namespace f_at_neg_one_l96_96911

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2 + 3 * x + 16

noncomputable def f_with_r (x : ℝ) (a r : ℝ) : ℝ := (x^3 + a * x^2 + 3 * x + 16) * (x - r)

theorem f_at_neg_one (a b c r : ℝ) (h1 : ∀ x, g x a = 0 → f_with_r x a r = 0)
  (h2 : a - r = 5) (h3 : 16 - 3 * r = 150) (h4 : -16 * r = c) :
  f_with_r (-1) a r = -1347 :=
by
  sorry

end f_at_neg_one_l96_96911


namespace function_zero_interval_l96_96750

noncomputable def f (x : ℝ) : ℝ := 1 / 4^x - Real.log x / Real.log 4

theorem function_zero_interval :
  ∃ (c : ℝ), 1 < c ∧ c < 2 ∧ f c = 0 := by
  sorry

end function_zero_interval_l96_96750


namespace Alice_wins_no_matter_what_Bob_does_l96_96853

theorem Alice_wins_no_matter_what_Bob_does (a b c : ℝ) :
  (∀ d : ℝ, (b + d) ^ 2 - 4 * (a + d) * (c + d) ≤ 0) → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  intro h
  sorry

end Alice_wins_no_matter_what_Bob_does_l96_96853


namespace inequality_condition_l96_96532

variables {a b c : ℝ} {x : ℝ}

theorem inequality_condition (h : a * a + b * b < c * c) : ∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0 :=
sorry

end inequality_condition_l96_96532


namespace max_third_altitude_l96_96174

theorem max_third_altitude (h1 h2 : ℕ) (h1_eq : h1 = 6) (h2_eq : h2 = 18) (triangle_scalene : true)
: (exists h3 : ℕ, (∀ h3_alt > h3, h3_alt > 8)) := 
sorry

end max_third_altitude_l96_96174


namespace truck_initial_gas_ratio_l96_96908

-- Definitions and conditions
def truck_total_capacity : ℕ := 20

def car_total_capacity : ℕ := 12

def car_initial_gas : ℕ := car_total_capacity / 3

def added_gas : ℕ := 18

-- Goal: The ratio of the gas in the truck's tank to its total capacity before she fills it up is 1:2
theorem truck_initial_gas_ratio :
  ∃ T : ℕ, (T + car_initial_gas + added_gas = truck_total_capacity + car_total_capacity) ∧ (T : ℚ) / truck_total_capacity = 1 / 2 :=
by
  sorry

end truck_initial_gas_ratio_l96_96908


namespace lcm_18_24_eq_72_l96_96882

-- Conditions
def factorization_18 : Nat × Nat := (1, 2) -- 18 = 2^1 * 3^2
def factorization_24 : Nat × Nat := (3, 1) -- 24 = 2^3 * 3^1

-- Definition of LCM using the highest powers from factorizations
def LCM (a b : Nat × Nat) : Nat :=
  let (p1, q1) := a
  let (p2, q2) := b
  (2^max p1 p2) * (3^max q1 q2)

-- Proof statement
theorem lcm_18_24_eq_72 : LCM factorization_18 factorization_24 = 72 :=
by
  sorry

end lcm_18_24_eq_72_l96_96882


namespace distance_AC_in_terms_of_M_l96_96431

-- Define the given constants and the relevant equations
variables (M x : ℝ) (AB BC AC : ℝ)
axiom distance_eq_add : AB = M + BC
axiom time_AB : (M + x) / 7 = x / 5
axiom time_BC : BC = x
axiom time_S : (M + x + x) = AC

theorem distance_AC_in_terms_of_M : AC = 6 * M :=
by
  sorry

end distance_AC_in_terms_of_M_l96_96431


namespace find_a5_l96_96507

-- Define the sequence and its properties
def geom_sequence (a : ℕ → ℕ) : Prop :=
∀ n m : ℕ, a (n + m) = (2^m) * a n

-- Define the problem statement
def sum_of_first_five_terms_is_31 (a : ℕ → ℕ) : Prop :=
a 1 + a 2 + a 3 + a 4 + a 5 = 31

-- State the theorem to prove
theorem find_a5 (a : ℕ → ℕ) (h_geom : geom_sequence a) (h_sum : sum_of_first_five_terms_is_31 a) : a 5 = 16 :=
by
  sorry

end find_a5_l96_96507


namespace keith_stored_bales_l96_96897

theorem keith_stored_bales (initial_bales added_bales final_bales : ℕ) :
  initial_bales = 22 → final_bales = 89 → final_bales = initial_bales + added_bales → added_bales = 67 :=
by
  intros h_initial h_final h_eq
  sorry

end keith_stored_bales_l96_96897


namespace bouquet_combinations_l96_96193

theorem bouquet_combinations :
  ∃ n : ℕ, (∀ r c t : ℕ, 4 * r + 3 * c + 2 * t = 60 → true) ∧ n = 13 :=
sorry

end bouquet_combinations_l96_96193


namespace value_of_expression_l96_96831

theorem value_of_expression : (85 + 32 / 113) * 113 = 9635 :=
by
  sorry

end value_of_expression_l96_96831


namespace value_of_a_l96_96021

variable (a : ℝ)

noncomputable def f (x : ℝ) := x^2 + 8
noncomputable def g (x : ℝ) := x^2 - 4

theorem value_of_a
  (h0 : a > 0)
  (h1 : f (g a) = 8) : a = 2 :=
by
  -- conditions are used as assumptions
  let f := f
  let g := g
  sorry

end value_of_a_l96_96021


namespace circles_touch_each_other_l96_96608

-- Define the radii of the two circles and the distance between their centers.
variables (R r d : ℝ)

-- Hypotheses: the condition and the relationships derived from the solution.
variables (x y t : ℝ)

-- The core relationships as conditions based on the problem and the solution.
axiom h1 : x + y = t
axiom h2 : x / y = R / r
axiom h3 : t / d = x / R

-- The proof statement
theorem circles_touch_each_other 
  (h1 : x + y = t) 
  (h2 : x / y = R / r) 
  (h3 : t / d = x / R) : 
  d = R + r := 
by 
  sorry

end circles_touch_each_other_l96_96608


namespace sale_savings_l96_96666

theorem sale_savings (price_fox : ℝ) (price_pony : ℝ) 
(discount_fox : ℝ) (discount_pony : ℝ) 
(total_discount : ℝ) (num_fox : ℕ) (num_pony : ℕ) 
(price_saved_during_sale : ℝ) :
price_fox = 15 → 
price_pony = 18 → 
num_fox = 3 → 
num_pony = 2 → 
total_discount = 22 → 
discount_pony = 15 → 
discount_fox = total_discount - discount_pony → 
price_saved_during_sale = num_fox * price_fox * (discount_fox / 100) + num_pony * price_pony * (discount_pony / 100) →
price_saved_during_sale = 8.55 := 
by sorry

end sale_savings_l96_96666


namespace negation_of_forall_exp_positive_l96_96670

theorem negation_of_forall_exp_positive :
  ¬ (∀ x : ℝ, Real.exp x > 0) ↔ ∃ x : ℝ, Real.exp x ≤ 0 :=
by {
  sorry
}

end negation_of_forall_exp_positive_l96_96670


namespace arithmetic_sequence_proof_l96_96281

noncomputable def a (n : ℕ) (a₁ d : ℝ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_proof
  (a₁ d : ℝ)
  (h : a 4 a₁ d + a 6 a₁ d + a 8 a₁ d + a 10 a₁ d + a 12 a₁ d = 120) :
  a 7 a₁ d - (1 / 3) * a 5 a₁ d = 16 :=
by
  sorry

end arithmetic_sequence_proof_l96_96281


namespace intersection_eq_singleton_l96_96222

-- Defining the sets M and N
def M : Set ℤ := {-1, 1, -2, 2}
def N : Set ℤ := {1, 4}

-- Stating the intersection problem
theorem intersection_eq_singleton :
  M ∩ N = {1} := 
by 
  sorry

end intersection_eq_singleton_l96_96222


namespace cost_of_parakeet_l96_96329

theorem cost_of_parakeet
  (P Py K : ℕ) -- defining the costs of parakeet, puppy, and kitten
  (h1 : Py = 3 * P) -- puppy is three times the cost of parakeet
  (h2 : P = K / 2) -- parakeet is half the cost of kitten
  (h3 : 2 * Py + 2 * K + 3 * P = 130) -- total cost equation
  : P = 10 := 
sorry

end cost_of_parakeet_l96_96329


namespace lucas_change_l96_96860

-- Define the given conditions as constants in Lean
def num_bananas : ℕ := 5
def cost_per_banana : ℝ := 0.70
def num_oranges : ℕ := 2
def cost_per_orange : ℝ := 0.80
def amount_paid : ℝ := 10.00

-- Define a noncomputable constant to represent the change received
noncomputable def change_received : ℝ := 
  amount_paid - (num_bananas * cost_per_banana + num_oranges * cost_per_orange)

-- State the theorem to be proved
theorem lucas_change : change_received = 4.90 := 
by 
  -- Dummy proof since the actual proof is not required
  sorry

end lucas_change_l96_96860


namespace total_pencils_owned_l96_96864

def SetA_pencils := 10
def SetB_pencils := 20
def SetC_pencils := 30

def friends_SetA_Buys := 3
def friends_SetB_Buys := 2
def friends_SetC_Buys := 2

def Chloe_SetA_Buys := 1
def Chloe_SetB_Buys := 1
def Chloe_SetC_Buys := 1

def total_friends_pencils := friends_SetA_Buys * SetA_pencils + friends_SetB_Buys * SetB_pencils + friends_SetC_Buys * SetC_pencils
def total_Chloe_pencils := Chloe_SetA_Buys * SetA_pencils + Chloe_SetB_Buys * SetB_pencils + Chloe_SetC_Buys * SetC_pencils
def total_pencils := total_friends_pencils + total_Chloe_pencils

theorem total_pencils_owned : total_pencils = 190 :=
by
  sorry

end total_pencils_owned_l96_96864


namespace equation_one_solution_equation_two_solution_l96_96949

theorem equation_one_solution (x : ℝ) (h : 7 * x - 20 = 2 * (3 - 3 * x)) : x = 2 :=
by {
  sorry
}

theorem equation_two_solution (x : ℝ) (h : (2 * x - 3) / 5 = (3 * x - 1) / 2 + 1) : x = -1 :=
by {
  sorry
}

end equation_one_solution_equation_two_solution_l96_96949


namespace polynomial_factorization_l96_96624

theorem polynomial_factorization (a b c : ℝ) :
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (a^2 + ab + ac + b^2 + bc + c^2) :=
sorry

end polynomial_factorization_l96_96624


namespace compute_expression_l96_96923

theorem compute_expression :
  23 ^ 12 / 23 ^ 5 + 5 = 148035894 :=
  sorry

end compute_expression_l96_96923


namespace find_circle_equation_l96_96799

noncomputable def center_of_parabola : ℝ × ℝ := (1, 0)

noncomputable def tangent_line (x y : ℝ) : Prop := 3 * x + 4 * y + 2 = 0

noncomputable def equation_of_circle (x y : ℝ) : Prop := (x - 1) ^ 2 + y ^ 2 = 1

theorem find_circle_equation 
  (center_c : ℝ × ℝ := center_of_parabola)
  (tangent : ∀ x y, tangent_line x y → (x - 1) ^ 2 + (y - 0) ^ 2 = 1) :
  equation_of_circle = (fun x y => sorry) :=
sorry

end find_circle_equation_l96_96799


namespace function_increasing_in_range_l96_96457

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - m) * x - m else Real.log x / Real.log m

theorem function_increasing_in_range (m : ℝ) :
  (3 / 2 ≤ m ∧ m < 3) ↔ (∀ x y : ℝ, x < y → f m x < f m y) := by
  sorry

end function_increasing_in_range_l96_96457


namespace abs_diff_x_y_l96_96748

variables {x y : ℝ}

noncomputable def floor (z : ℝ) : ℤ := Int.floor z
noncomputable def fract (z : ℝ) : ℝ := z - floor z

theorem abs_diff_x_y 
  (h1 : floor x + fract y = 3.7) 
  (h2 : fract x + floor y = 4.6) : 
  |x - y| = 1.1 :=
by
  sorry

end abs_diff_x_y_l96_96748


namespace ratio_problem_l96_96409

theorem ratio_problem (x n : ℕ) (h1 : 5 * x = n) (h2 : n = 65) : x = 13 :=
by
  sorry

end ratio_problem_l96_96409


namespace range_of_F_l96_96862

theorem range_of_F (A B C : ℝ) (h1 : 0 < A) (h2 : A ≤ B) (h3 : B ≤ C) (h4 : C < π / 2) :
  1 + (Real.sqrt 2) / 2 < (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) ∧
  (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) < 2 :=
  sorry

end range_of_F_l96_96862


namespace total_apples_l96_96707

-- Define the number of apples given to each person
def apples_per_person : ℝ := 15.0

-- Define the number of people
def number_of_people : ℝ := 3.0

-- Goal: Prove that the total number of apples is 45.0
theorem total_apples : apples_per_person * number_of_people = 45.0 := by
  sorry

end total_apples_l96_96707


namespace convoy_length_after_checkpoint_l96_96564

theorem convoy_length_after_checkpoint
  (L_initial : ℝ) (v_initial : ℝ) (v_final : ℝ) (t_fin : ℝ)
  (H_initial_len : L_initial = 300)
  (H_initial_speed : v_initial = 60)
  (H_final_speed : v_final = 40)
  (H_time_last_car : t_fin = (300 / 1000) / 60) :
  L_initial * v_final / v_initial - (v_final * ((300 / 1000) / 60)) = 200 :=
by
  sorry

end convoy_length_after_checkpoint_l96_96564


namespace extremum_at_neg3_l96_96376

variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 + 5 * x^2 + a * x
def f_deriv (x : ℝ) : ℝ := 3 * x^2 + 10 * x + a

theorem extremum_at_neg3 (h : f_deriv a (-3) = 0) : a = 3 := 
  by
  sorry

end extremum_at_neg3_l96_96376


namespace painting_together_time_l96_96437

theorem painting_together_time (jamshid_time taimour_time time_together : ℝ) 
  (h1 : jamshid_time = taimour_time / 2)
  (h2 : taimour_time = 21)
  (h3 : time_together = 7) :
  (1 / taimour_time + 1 / jamshid_time) * time_together = 1 := 
sorry

end painting_together_time_l96_96437


namespace dennis_teaching_years_l96_96198

noncomputable def years_taught (V A D E N : ℕ) := V + A + D + E + N
noncomputable def sum_of_ages := 375
noncomputable def teaching_years : Prop :=
  ∃ (A V D E N : ℕ),
    V + A + D + E + N = 225 ∧
    V = A + 9 ∧
    V = D - 15 ∧
    E = A - 3 ∧
    E = 2 * N ∧
    D = 101

theorem dennis_teaching_years : teaching_years :=
by
  sorry

end dennis_teaching_years_l96_96198


namespace sum_of_first_10_terms_of_arithmetic_sequence_l96_96776

theorem sum_of_first_10_terms_of_arithmetic_sequence :
  ∀ (a n : ℕ) (a₁ : ℤ) (d : ℤ),
  (d = -2) →
  (a₇ : ℤ := a₁ + 6 * d) →
  (a₃ : ℤ := a₁ + 2 * d) →
  (a₁₀ : ℤ := a₁ + 9 * d) →
  (a₇ * a₇ = a₃ * a₁₀) →
  (S₁₀ : ℤ := 10 * a₁ + 45 * d) →
  S₁₀ = 270 :=
by
  intros a n a₁ d hd ha₇ ha₃ ha₁₀ hgm hS₁₀
  sorry

end sum_of_first_10_terms_of_arithmetic_sequence_l96_96776


namespace soda_cost_l96_96134

theorem soda_cost (b s f : ℕ) (h1 : 3 * b + 2 * s + 2 * f = 590) (h2 : 2 * b + 3 * s + f = 610) : s = 140 :=
sorry

end soda_cost_l96_96134


namespace fraction_of_seats_taken_l96_96556

theorem fraction_of_seats_taken : 
  ∀ (total_seats broken_fraction available_seats : ℕ), 
    total_seats = 500 → 
    broken_fraction = 1 / 10 → 
    available_seats = 250 → 
    (total_seats - available_seats - total_seats * broken_fraction) / total_seats = 2 / 5 :=
by
  intro total_seats broken_fraction available_seats
  intro h1 h2 h3
  sorry

end fraction_of_seats_taken_l96_96556


namespace jim_ran_16_miles_in_2_hours_l96_96086

-- Given conditions
variables (j f : ℝ) -- miles Jim ran in 2 hours, miles Frank ran in 2 hours
variables (h1 : f = 20) -- Frank ran 20 miles in 2 hours
variables (h2 : f / 2 = (j / 2) + 2) -- Frank ran 2 miles more than Jim in an hour

-- Statement to prove
theorem jim_ran_16_miles_in_2_hours (j f : ℝ) (h1 : f = 20) (h2 : f / 2 = (j / 2) + 2) : j = 16 :=
by
  sorry

end jim_ran_16_miles_in_2_hours_l96_96086


namespace minimize_water_tank_construction_cost_l96_96366

theorem minimize_water_tank_construction_cost 
  (volume : ℝ := 4800)
  (depth : ℝ := 3)
  (cost_bottom_per_m2 : ℝ := 150)
  (cost_walls_per_m2 : ℝ := 120)
  (x : ℝ) :
  (volume = x * x * depth) →
  (∀ y, y = cost_bottom_per_m2 * x * x + cost_walls_per_m2 * 4 * x * depth) →
  (x = 40) ∧ (y = 297600) :=
by
  sorry

end minimize_water_tank_construction_cost_l96_96366


namespace chloe_total_score_l96_96828

theorem chloe_total_score :
  let first_level_treasure_points := 9
  let first_level_bonus_points := 15
  let first_level_treasures := 6
  let second_level_treasure_points := 11
  let second_level_bonus_points := 20
  let second_level_treasures := 3

  let first_level_score := first_level_treasures * first_level_treasure_points + first_level_bonus_points
  let second_level_score := second_level_treasures * second_level_treasure_points + second_level_bonus_points

  first_level_score + second_level_score = 122 :=
by
  sorry

end chloe_total_score_l96_96828


namespace golden_section_AP_l96_96320

-- Definitions of the golden ratio and its reciprocal
noncomputable def phi := (1 + Real.sqrt 5) / 2
noncomputable def phi_inv := (Real.sqrt 5 - 1) / 2

-- Conditions of the problem
def isGoldenSectionPoint (A B P : ℝ) := ∃ AP BP AB, AP < BP ∧ BP = 10 ∧ P = AB ∧ AP = BP * phi_inv

theorem golden_section_AP (A B P : ℝ) (h1 : isGoldenSectionPoint A B P) : 
  ∃ AP, AP = 5 * Real.sqrt 5 - 5 :=
by
  sorry

end golden_section_AP_l96_96320


namespace rectangle_side_l96_96589

theorem rectangle_side (x : ℝ) (w : ℝ) (P : ℝ) (hP : P = 30) (h : 2 * (x + w) = P) : w = 15 - x :=
by
  -- Proof goes here
  sorry

end rectangle_side_l96_96589


namespace division_result_l96_96055

-- Define n in terms of the given condition
def n : ℕ := 9^2023

theorem division_result : n / 3 = 3^4045 :=
by
  sorry

end division_result_l96_96055


namespace sum_of_coefficients_l96_96604

theorem sum_of_coefficients (a b c : ℤ) (h : a - b + c = -1) : a + b + c = -1 := sorry

end sum_of_coefficients_l96_96604


namespace ratio_of_two_numbers_l96_96961

theorem ratio_of_two_numbers (A B : ℕ) (x y : ℕ) (h1 : lcm A B = 60) (h2 : A + B = 50) (h3 : A / B = x / y) (hx : x = 3) (hy : y = 2) : x = 3 ∧ y = 2 := 
by
  -- Conditions provided in the problem
  sorry

end ratio_of_two_numbers_l96_96961


namespace length_on_ninth_day_l96_96419

-- Define relevant variables and conditions.
variables (a1 d : ℕ)

-- Define conditions as hypotheses.
def problem_conditions : Prop :=
  (7 * a1 + 21 * d = 28) ∧ 
  (a1 + d + a1 + 4 * d + a1 + 7 * d = 15)

theorem length_on_ninth_day (h : problem_conditions a1 d) : (a1 + 8 * d = 9) :=
  sorry

end length_on_ninth_day_l96_96419


namespace new_avg_weight_l96_96524

-- Definition of the conditions
def original_team_avg_weight : ℕ := 94
def original_team_size : ℕ := 7
def new_player_weight_1 : ℕ := 110
def new_player_weight_2 : ℕ := 60
def total_new_team_size : ℕ := original_team_size + 2

-- Computation of the total weight
def total_weight_original_team : ℕ := original_team_avg_weight * original_team_size
def total_weight_new_team : ℕ := total_weight_original_team + new_player_weight_1 + new_player_weight_2

-- Statement of the theorem
theorem new_avg_weight : total_weight_new_team / total_new_team_size = 92 := by
  -- Proof is omitted
  sorry

end new_avg_weight_l96_96524


namespace distance_from_B_to_center_is_74_l96_96973

noncomputable def circle_radius := 10
noncomputable def B_distance (a b : ℝ) := a^2 + b^2

theorem distance_from_B_to_center_is_74 
  (a b : ℝ)
  (hA : a^2 + (b + 6)^2 = 100)
  (hC : (a + 4)^2 + b^2 = 100) :
  B_distance a b = 74 :=
sorry

end distance_from_B_to_center_is_74_l96_96973


namespace at_least_one_non_negative_l96_96168

theorem at_least_one_non_negative 
  (a b c d e f g h : ℝ) : 
  ac + bd ≥ 0 ∨ ae + bf ≥ 0 ∨ ag + bh ≥ 0 ∨ ce + df ≥ 0 ∨ cg + dh ≥ 0 ∨ eg + fh ≥ 0 := 
sorry

end at_least_one_non_negative_l96_96168


namespace problem_statement_l96_96164

variables {x y x1 y1 a b c d : ℝ}

-- The main theorem statement
theorem problem_statement (h0 : ∀ (x y : ℝ), 6 * y ^ 2 = 2 * x ^ 3 + 3 * x ^ 2 + x) 
                           (h1 : x1 = a * x + b) 
                           (h2 : y1 = c * y + d) 
                           (h3 : y1 ^ 2 = x1 ^ 3 - 36 * x1) : 
                           a + b + c + d = 90 := sorry

end problem_statement_l96_96164


namespace high_school_sampling_problem_l96_96465

theorem high_school_sampling_problem :
  let first_year_classes := 20
  let first_year_students_per_class := 50
  let first_year_total_students := first_year_classes * first_year_students_per_class
  let second_year_classes := 24
  let second_year_students_per_class := 45
  let second_year_total_students := second_year_classes * second_year_students_per_class
  let total_students := first_year_total_students + second_year_total_students
  let survey_students := 208
  let first_year_sample := (first_year_total_students * survey_students) / total_students
  let second_year_sample := (second_year_total_students * survey_students) / total_students
  let A_selected_probability := first_year_sample / first_year_total_students
  let B_selected_probability := second_year_sample / second_year_total_students
  (survey_students = 208) →
  (first_year_sample = 100) →
  (second_year_sample = 108) →
  (A_selected_probability = 1 / 10) →
  (B_selected_probability = 1 / 10) →
  (A_selected_probability = B_selected_probability) →
  (student_A_in_first_year : true) →
  (student_B_in_second_year : true) →
  true :=
  by sorry

end high_school_sampling_problem_l96_96465


namespace game_completion_days_l96_96656

theorem game_completion_days (initial_playtime hours_per_day : ℕ) (initial_days : ℕ) (completion_percentage : ℚ) (increased_playtime : ℕ) (remaining_days : ℕ) :
  initial_playtime = 4 →
  hours_per_day = 2 * 7 →
  completion_percentage = 0.4 →
  increased_playtime = 7 →
  ((initial_playtime * hours_per_day) / completion_percentage) - (initial_playtime * hours_per_day) = increased_playtime * remaining_days →
  remaining_days = 12 :=
by
  intros
  sorry

end game_completion_days_l96_96656


namespace Harold_spending_l96_96782

theorem Harold_spending
  (num_shirt_boxes : ℕ)
  (num_xl_boxes : ℕ)
  (wraps_shirt_boxes : ℕ)
  (wraps_xl_boxes : ℕ)
  (cost_per_roll : ℕ)
  (h1 : num_shirt_boxes = 20)
  (h2 : num_xl_boxes = 12)
  (h3 : wraps_shirt_boxes = 5)
  (h4 : wraps_xl_boxes = 3)
  (h5 : cost_per_roll = 4) :
  num_shirt_boxes / wraps_shirt_boxes + num_xl_boxes / wraps_xl_boxes * cost_per_roll = 32 :=
by
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end Harold_spending_l96_96782


namespace train_speed_l96_96998

noncomputable def train_speed_kmph (L_t L_b : ℝ) (T : ℝ) : ℝ :=
  (L_t + L_b) / T * 3.6

theorem train_speed (L_t L_b : ℝ) (T : ℝ) :
  L_t = 110 ∧ L_b = 190 ∧ T = 17.998560115190784 → train_speed_kmph L_t L_b T = 60 :=
by
  intro h
  sorry

end train_speed_l96_96998


namespace sqrt_infinite_nest_eq_two_l96_96330

theorem sqrt_infinite_nest_eq_two (y : ℝ) (h : y = Real.sqrt (2 + y)) : y = 2 := 
sorry

end sqrt_infinite_nest_eq_two_l96_96330


namespace prob_kong_meng_is_one_sixth_l96_96344

variable (bag : List String := ["孔", "孟", "之", "乡"])
variable (draws : List String := [])
def total_events : ℕ := 4 * 3
def favorable_events : ℕ := 2
def probability_kong_meng : ℚ := favorable_events / total_events

theorem prob_kong_meng_is_one_sixth :
  (probability_kong_meng = 1 / 6) :=
by
  sorry

end prob_kong_meng_is_one_sixth_l96_96344


namespace incorrect_statement_B_l96_96327

def two_times_root_equation (a b c x1 x2 : ℝ) : Prop :=
  a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 ∧ (x1 = 2 * x2 ∨ x2 = 2 * x1)

theorem incorrect_statement_B (m n : ℝ) (h : (x - 2) * (m * x + n) = 0) :
  ¬(two_times_root_equation 1 (-m+n) (-mn) 2 (-n / m) -> m + n = 0) :=
sorry

end incorrect_statement_B_l96_96327


namespace probability_of_two_one_color_and_one_other_color_l96_96593

theorem probability_of_two_one_color_and_one_other_color
    (black_balls white_balls : ℕ)
    (total_drawn : ℕ)
    (draw_two_black_one_white : ℕ)
    (draw_one_black_two_white : ℕ)
    (total_ways : ℕ)
    (favorable_ways : ℕ)
    (probability : ℚ) :
    black_balls = 8 →
    white_balls = 7 →
    total_drawn = 3 →
    draw_two_black_one_white = 196 →
    draw_one_black_two_white = 168 →
    total_ways = 455 →
    favorable_ways = draw_two_black_one_white + draw_one_black_two_white →
    probability = favorable_ways / total_ways →
    probability = 4 / 5 :=
by sorry

end probability_of_two_one_color_and_one_other_color_l96_96593


namespace math_problem_l96_96622

theorem math_problem
  (a b c d m : ℝ)
  (h1 : a = -b)
  (h2 : c = (1 / d) ∨ d = (1 / c))
  (h3 : |m| = 4) :
  (a + b = 0) ∧ (c * d = 1) ∧ (m = 4 ∨ m = -4) ∧
  ((a + b) / 3 + m^2 - 5 * (c * d) = 11) := by
  sorry

end math_problem_l96_96622


namespace distribute_tickets_among_people_l96_96270

noncomputable def distribution_ways : ℕ := 84

theorem distribute_tickets_among_people (tickets : Fin 5 → ℕ) (persons : Fin 4 → ℕ)
  (h1 : ∀ p : Fin 4, ∃ t : Fin 5, tickets t = persons p)
  (h2 : ∀ p : Fin 4, ∀ t1 t2 : Fin 5, tickets t1 = persons p ∧ tickets t2 = persons p → (t1.val + 1 = t2.val ∨ t2.val + 1 = t1.val)) :
  ∃ n : ℕ, n = distribution_ways := by
  use 84
  trivial

end distribute_tickets_among_people_l96_96270


namespace dad_additional_money_l96_96935

-- Define the conditions in Lean
def daily_savings : ℕ := 35
def days : ℕ := 7
def total_savings_before_doubling := daily_savings * days
def doubled_savings := 2 * total_savings_before_doubling
def total_amount_after_7_days : ℕ := 500

-- Define the theorem to prove
theorem dad_additional_money : (total_amount_after_7_days - doubled_savings) = 10 := by
  sorry

end dad_additional_money_l96_96935


namespace greatest_x_l96_96551

theorem greatest_x (x : ℕ) (h_pos : 0 < x) (h_ineq : (x^6) / (x^3) < 18) : x = 2 :=
by sorry

end greatest_x_l96_96551


namespace interest_calculated_years_l96_96970

variable (P T : ℝ)

-- Given conditions
def principal_sum_positive : Prop := P > 0
def simple_interest_condition : Prop := (P * 5 * T) / 100 = P / 5

-- Theorem statement
theorem interest_calculated_years (h1 : principal_sum_positive P) (h2 : simple_interest_condition P T) : T = 4 :=
  sorry

end interest_calculated_years_l96_96970


namespace commission_percentage_l96_96205

def commission_rate (amount: ℕ) : ℚ :=
  if amount <= 500 then
    0.20 * amount
  else
    0.20 * 500 + 0.50 * (amount - 500)

theorem commission_percentage (total_sale : ℕ) (h : total_sale = 800) :
  (commission_rate total_sale) / total_sale * 100 = 31.25 :=
by
  sorry

end commission_percentage_l96_96205


namespace monotonically_decreasing_intervals_max_and_min_values_on_interval_l96_96248

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x + a

theorem monotonically_decreasing_intervals (a : ℝ) : 
  ∀ x : ℝ, (x < -1 ∨ x > 3) → f x a < f (x+1) a :=
sorry

theorem max_and_min_values_on_interval : 
  (f (-1) (-2) = -7) ∧ (max (f (-2) (-2)) (f 2 (-2)) = 20) :=
sorry

end monotonically_decreasing_intervals_max_and_min_values_on_interval_l96_96248


namespace alloy_problem_solution_l96_96628

theorem alloy_problem_solution (x y k n : ℝ) (H_weight : k * 4 * x + n * 3 * y = 10)
    (H_ratio : (kx + ny)/(k * 3 * x + n * 2 * y) = 3/7) :
    k * 4 * x = 4 :=
by
  -- Proof to be provided
  sorry

end alloy_problem_solution_l96_96628


namespace frisbee_total_distance_l96_96352

-- Definitions for the conditions
def bess_initial_distance : ℝ := 20
def bess_throws : ℕ := 4
def bess_reduction : ℝ := 0.90
def holly_initial_distance : ℝ := 8
def holly_throws : ℕ := 5
def holly_reduction : ℝ := 0.95

-- Function to calculate the total distance for Bess
def total_distance_bess : ℝ :=
  let distances := List.range bess_throws |>.map (λ i => bess_initial_distance * bess_reduction ^ i)
  (distances.sum) * 2

-- Function to calculate the total distance for Holly
def total_distance_holly : ℝ :=
  let distances := List.range holly_throws |>.map (λ i => holly_initial_distance * holly_reduction ^ i)
  distances.sum

-- Proof statement
theorem frisbee_total_distance : 
  total_distance_bess + total_distance_holly = 173.76 :=
by
  sorry

end frisbee_total_distance_l96_96352


namespace sufficient_but_not_necessary_condition_l96_96254

variable (a₁ d : ℝ)

def S₄ := 4 * a₁ + 6 * d
def S₅ := 5 * a₁ + 10 * d
def S₆ := 6 * a₁ + 15 * d

theorem sufficient_but_not_necessary_condition (h : d > 1) :
  S₄ a₁ d + S₆ a₁ d > 2 * S₅ a₁ d :=
by
  -- proof omitted
  sorry

end sufficient_but_not_necessary_condition_l96_96254


namespace neg_p_eq_exist_l96_96963

theorem neg_p_eq_exist:
  (¬ ∀ a b : ℝ, a^2 + b^2 ≥ 2 * a * b) ↔ ∃ a b : ℝ, a^2 + b^2 < 2 * a * b := by
  sorry

end neg_p_eq_exist_l96_96963


namespace wire_attachment_distance_l96_96135

theorem wire_attachment_distance :
  ∃ x : ℝ, 
    (∀ z y : ℝ, z = Real.sqrt (x ^ 2 + 3.6 ^ 2) ∧ y = Real.sqrt ((x + 5) ^ 2 + 3.6 ^ 2) →
      z + y = 13) ∧
    abs ((x : ℝ) - 2.7) < 0.01 := -- Assuming numerical closeness within a small epsilon for practical solutions.
sorry -- Proof not provided.

end wire_attachment_distance_l96_96135


namespace values_of_xyz_l96_96702

theorem values_of_xyz (x y z : ℝ) (h1 : 2 * x - y + z = 14) (h2 : y = 2) (h3 : x + z = 3 * y + 5) : 
  x = 5 ∧ y = 2 ∧ z = 6 := 
by
  sorry

end values_of_xyz_l96_96702


namespace equilibrium_proof_l96_96558

noncomputable def equilibrium_constant (Γ_eq B_eq : ℝ) : ℝ :=
(Γ_eq ^ 3) / (B_eq ^ 3)

theorem equilibrium_proof (Γ_eq B_eq : ℝ) (K_c : ℝ) (B_initial : ℝ) (Γ_initial : ℝ)
  (hΓ : Γ_eq = 0.25) (hB : B_eq = 0.15) (hKc : K_c = 4.63) 
  (ratio : Γ_eq = B_eq + B_initial) (hΓ_initial : Γ_initial = 0) :
  equilibrium_constant Γ_eq B_eq = K_c ∧ 
  B_initial = 0.4 ∧ 
  Γ_initial = 0 := 
by
  sorry

end equilibrium_proof_l96_96558


namespace yz_zx_xy_minus_2xyz_leq_7_27_l96_96989

theorem yz_zx_xy_minus_2xyz_leq_7_27 (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) :
  (y * z + z * x + x * y - 2 * x * y * z) ≤ 7 / 27 := 
by 
  sorry

end yz_zx_xy_minus_2xyz_leq_7_27_l96_96989


namespace incorrect_conclusion_l96_96215

-- Define the linear regression model
def model (x : ℝ) : ℝ := 0.85 * x - 85.71

-- Define the conditions
axiom linear_correlation : ∀ (x y : ℝ), ∃ (x_i y_i : ℝ) (i : ℕ), model x = y

-- The theorem to prove the statement for x = 170 is false
theorem incorrect_conclusion (x : ℝ) (h : x = 170) : ¬ (model x = 58.79) :=
  by sorry

end incorrect_conclusion_l96_96215


namespace xy_max_value_l96_96252

theorem xy_max_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 4 * y = 12) :
  xy <= 9 := by
  sorry

end xy_max_value_l96_96252


namespace line_equation_with_equal_intercepts_l96_96240

theorem line_equation_with_equal_intercepts 
  (a : ℝ) 
  (l : ℝ → ℝ → Prop) 
  (h : ∀ x y, l x y ↔ (a+1)*x + y + 2 - a = 0) 
  (intercept_condition : ∀ x y, l x 0 = l 0 y) : 
  (∀ x y, l x y ↔ x + y + 2 = 0) ∨ (∀ x y, l x y ↔ 3*x + y = 0) :=
sorry

end line_equation_with_equal_intercepts_l96_96240


namespace min_value_of_function_l96_96917

theorem min_value_of_function (x : ℝ) (hx : x > 0) :
  ∃ y, y = (3 + x + x^2) / (1 + x) ∧ y = -1 + 2 * Real.sqrt 3 :=
sorry

end min_value_of_function_l96_96917


namespace outlet_pipe_empties_2_over_3_in_16_min_l96_96306

def outlet_pipe_part_empty_in_t (t : ℕ) (part_per_8_min : ℚ) : ℚ :=
  (part_per_8_min / 8) * t

theorem outlet_pipe_empties_2_over_3_in_16_min (
  part_per_8_min : ℚ := 1/3
) : outlet_pipe_part_empty_in_t 16 part_per_8_min = 2/3 :=
by
  sorry

end outlet_pipe_empties_2_over_3_in_16_min_l96_96306


namespace problem_1_problem_2_l96_96369

-- Proof Problem 1: Prove A ∩ B = {x | -3 ≤ x ≤ -2} given m = -3
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | 2 * m - 1 ≤ x ∧ x ≤ m + 1}

theorem problem_1 : B (-3) ∩ A = {x | -3 ≤ x ∧ x ≤ -2} := sorry

-- Proof Problem 2: Prove m ≥ -1 given B ⊆ A
theorem problem_2 (m : ℝ) : (B m ⊆ A) → m ≥ -1 := sorry

end problem_1_problem_2_l96_96369


namespace no_solutions_exist_l96_96737

theorem no_solutions_exist : ¬ ∃ (x y z : ℝ), x + y = 3 ∧ xy - z^2 = 2 :=
by sorry

end no_solutions_exist_l96_96737


namespace dadAgeWhenXiaoHongIs7_l96_96079

variable {a : ℕ}

-- Condition: Dad's age is given as 'a'
-- Condition: Dad's age is 4 times plus 3 years more than Xiao Hong's age
def xiaoHongAge (a : ℕ) : ℕ := (a - 3) / 4

theorem dadAgeWhenXiaoHongIs7 : xiaoHongAge a = 7 → a = 31 := by
  intro h
  have h1 : a - 3 = 28 := by sorry   -- Algebraic manipulation needed
  have h2 : a = 31 := by sorry       -- Algebraic manipulation needed
  exact h2

end dadAgeWhenXiaoHongIs7_l96_96079


namespace determinant_A_l96_96413

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![
    ![  2,  4, -2],
    ![  3, -1,  5],
    ![-1,  3,  2]
  ]

theorem determinant_A : det A = -94 := by
  sorry

end determinant_A_l96_96413


namespace pythagorean_triple_example_l96_96770

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_example :
  is_pythagorean_triple 7 24 25 :=
sorry

end pythagorean_triple_example_l96_96770


namespace gcd_f100_f101_l96_96514

def f (x : ℤ) : ℤ := x^2 - 3 * x + 2023

theorem gcd_f100_f101 : Int.gcd (f 100) (f 101) = 2 :=
by
  sorry

end gcd_f100_f101_l96_96514


namespace total_seashells_l96_96697

theorem total_seashells :
  let initial_seashells : ℝ := 6.5
  let more_seashells : ℝ := 4.25
  initial_seashells + more_seashells = 10.75 :=
by
  sorry

end total_seashells_l96_96697


namespace coin_flip_sequences_l96_96806

theorem coin_flip_sequences :
  let total_sequences := 2^10
  let sequences_starting_with_two_heads := 2^8
  total_sequences - sequences_starting_with_two_heads = 768 :=
by
  sorry

end coin_flip_sequences_l96_96806


namespace fifth_graders_more_than_seventh_l96_96857

theorem fifth_graders_more_than_seventh (price_per_pencil : ℕ) (price_per_pencil_pos : price_per_pencil > 0)
    (total_cents_7th : ℕ) (total_cents_7th_val : total_cents_7th = 201)
    (total_cents_5th : ℕ) (total_cents_5th_val : total_cents_5th = 243)
    (pencil_price_div_7th : total_cents_7th % price_per_pencil = 0)
    (pencil_price_div_5th : total_cents_5th % price_per_pencil = 0) :
    (total_cents_5th / price_per_pencil - total_cents_7th / price_per_pencil = 14) := 
by
    sorry

end fifth_graders_more_than_seventh_l96_96857


namespace geometric_sequence_sum_l96_96144

theorem geometric_sequence_sum
  (a r : ℝ)
  (h1 : a * (1 - r ^ 3000) / (1 - r) = 500)
  (h2 : a * (1 - r ^ 6000) / (1 - r) = 950) :
  a * (1 - r ^ 9000) / (1 - r) = 1355 :=
sorry

end geometric_sequence_sum_l96_96144


namespace solve_abc_l96_96179

theorem solve_abc (a b c : ℝ) (h1 : a ≤ b ∧ b ≤ c) (h2 : a + b + c = -1) (h3 : a * b + b * c + a * c = -4) (h4 : a * b * c = -2) :
  a = -1 - Real.sqrt 3 ∧ b = -1 + Real.sqrt 3 ∧ c = 1 :=
by
  -- Proof goes here
  sorry

end solve_abc_l96_96179


namespace num_rows_of_gold_bars_l96_96297

-- Definitions from the problem conditions
def num_bars_per_row : ℕ := 20
def total_worth : ℕ := 1600000

-- Statement to prove
theorem num_rows_of_gold_bars :
  (total_worth / (total_worth / num_bars_per_row)) = 1 := 
by sorry

end num_rows_of_gold_bars_l96_96297


namespace remainder_of_expression_l96_96658

theorem remainder_of_expression (m : ℤ) (h : m % 9 = 3) : (3 * m + 2436) % 9 = 0 := 
by 
  sorry

end remainder_of_expression_l96_96658


namespace real_y_iff_x_l96_96349

open Real

-- Definitions based on the conditions
def quadratic_eq (y x : ℝ) : ℝ := 9 * y^2 - 3 * x * y + x + 8

-- The main theorem to prove
theorem real_y_iff_x (x : ℝ) : (∃ y : ℝ, quadratic_eq y x = 0) ↔ x ≤ -4 ∨ x ≥ 8 := 
sorry

end real_y_iff_x_l96_96349


namespace inequality_holds_and_equality_occurs_l96_96375

theorem inequality_holds_and_equality_occurs (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 4) :
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5) ∧ (x = 2 ∧ y = 2 → 1 / (x + 3) + 1 / (y + 3) = 2 / 5) :=
by
  sorry

end inequality_holds_and_equality_occurs_l96_96375


namespace thabo_paperback_diff_l96_96930

variable (total_books : ℕ) (H_books : ℕ) (P_books : ℕ) (F_books : ℕ)

def thabo_books_conditions :=
  total_books = 160 ∧
  H_books = 25 ∧
  P_books > H_books ∧
  F_books = 2 * P_books ∧
  total_books = F_books + P_books + H_books 

theorem thabo_paperback_diff :
  thabo_books_conditions total_books H_books P_books F_books → 
  (P_books - H_books) = 20 :=
by
  sorry

end thabo_paperback_diff_l96_96930


namespace general_term_an_l96_96716

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 2
noncomputable def S_n (n : ℕ) : ℕ := n^2 + 3 * n

theorem general_term_an (n : ℕ) (h : 1 ≤ n) : a_n n = (S_n n) - (S_n (n-1)) :=
by sorry

end general_term_an_l96_96716


namespace functional_equation_solution_l96_96148

open Function

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f (x ^ 2 + f y) = y + f x ^ 2) → (∀ x : ℝ, f x = x) :=
by
  sorry

end functional_equation_solution_l96_96148


namespace paws_on_ground_are_correct_l96_96815

-- Problem statement
def num_paws_on_ground (total_dogs : ℕ) (half_on_all_fours : ℕ) (paws_on_all_fours : ℕ) (half_on_two_legs : ℕ) (paws_on_two_legs : ℕ) : ℕ :=
  half_on_all_fours * paws_on_all_fours + half_on_two_legs * paws_on_two_legs

theorem paws_on_ground_are_correct :
  let total_dogs := 12
  let half_on_all_fours := 6
  let half_on_two_legs := 6
  let paws_on_all_fours := 4
  let paws_on_two_legs := 2
  num_paws_on_ground total_dogs half_on_all_fours paws_on_all_fours half_on_two_legs paws_on_two_legs = 36 :=
by sorry

end paws_on_ground_are_correct_l96_96815


namespace probability_white_given_popped_l96_96350

theorem probability_white_given_popped :
  let P_white := 3 / 5
  let P_yellow := 2 / 5
  let P_popped_given_white := 2 / 5
  let P_popped_given_yellow := 4 / 5
  let P_white_and_popped := P_white * P_popped_given_white
  let P_yellow_and_popped := P_yellow * P_popped_given_yellow
  let P_popped := P_white_and_popped + P_yellow_and_popped
  let P_white_given_popped := P_white_and_popped / P_popped
  P_white_given_popped = 3 / 7 :=
by sorry

end probability_white_given_popped_l96_96350


namespace solution_1_solution_2_l96_96151

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 1) + abs (2 * x + 3)

lemma f_piecewise (x : ℝ) : 
  f x = if x ≤ -3 / 2 then -4 * x - 2
        else if -3 / 2 < x ∧ x < 1 / 2 then 4
        else 4 * x + 2 := 
by
-- This lemma represents the piecewise definition of f(x)
sorry

theorem solution_1 : 
  (∀ x : ℝ, f x < 5 ↔ (-7 / 4 < x ∧ x < 3 / 4)) := 
by 
-- Proof of the inequality solution
sorry

theorem solution_2 : 
  (∀ t : ℝ, (∀ x : ℝ, f x - t ≥ 0) → t ≤ 4) :=
by
-- Proof that the maximum value of t is 4
sorry

end solution_1_solution_2_l96_96151


namespace cara_total_debt_l96_96293

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

end cara_total_debt_l96_96293


namespace column_heights_achievable_l96_96097

open Int

noncomputable def number_of_column_heights (n : ℕ) (h₁ h₂ h₃ : ℕ) : ℕ :=
  let min_height := n * h₁
  let max_height := n * h₃
  max_height - min_height + 1

theorem column_heights_achievable :
  number_of_column_heights 80 3 8 15 = 961 := by
  -- Proof goes here.
  sorry

end column_heights_achievable_l96_96097


namespace transformed_ellipse_l96_96112

-- Define the original equation and the transformation
def orig_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

noncomputable def trans_x (x' : ℝ) : ℝ := x' / 5
noncomputable def trans_y (y' : ℝ) : ℝ := y' / 4

-- Prove that the transformed equation is an ellipse with specified properties
theorem transformed_ellipse :
  (∃ x' y' : ℝ, (trans_x x')^2 + (trans_y y')^2 = 1) →
  ∃ a b : ℝ, (a = 10) ∧ (b = 8) ∧ (∀ x' y' : ℝ, x'^2 / (a/2)^2 + y'^2 / (b/2)^2 = 1) :=
sorry

end transformed_ellipse_l96_96112


namespace intersection_of_complements_l96_96343

-- Define the universal set U as a natural set with numbers <= 8
def U : Set ℕ := { x | x ≤ 8 }

-- Define the set A
def A : Set ℕ := { 1, 3, 7 }

-- Define the set B
def B : Set ℕ := { 2, 3, 8 }

-- Prove the statement for the intersection of the complements of A and B with respect to U
theorem intersection_of_complements : 
  ((U \ A) ∩ (U \ B)) = ({ 0, 4, 5, 6 } : Set ℕ) :=
by
  sorry

end intersection_of_complements_l96_96343


namespace Tahir_contribution_l96_96715

theorem Tahir_contribution
  (headphone_cost : ℕ := 200)
  (kenji_yen : ℕ := 15000)
  (exchange_rate : ℕ := 100)
  (kenji_contribution : ℕ := kenji_yen / exchange_rate)
  (tahir_contribution : ℕ := headphone_cost - kenji_contribution) :
  tahir_contribution = 50 := 
  by sorry

end Tahir_contribution_l96_96715


namespace sum_and_product_of_three_numbers_l96_96678

variables (a b c : ℝ)

-- Conditions
axiom h1 : a + b = 35
axiom h2 : b + c = 47
axiom h3 : c + a = 52

-- Prove the sum and product
theorem sum_and_product_of_three_numbers : a + b + c = 67 ∧ a * b * c = 9600 :=
by {
  sorry
}

end sum_and_product_of_three_numbers_l96_96678


namespace true_proposition_B_l96_96159

theorem true_proposition_B : (3 > 4) ∨ (3 < 4) :=
sorry

end true_proposition_B_l96_96159


namespace problem_l96_96397

theorem problem (a b : ℝ) (h : ∀ x : ℝ, (x + a) * (x + b) = x^2 + 4 * x + 3) : a + b = 4 :=
by
  sorry

end problem_l96_96397


namespace men_in_first_group_l96_96925

theorem men_in_first_group (M : ℕ) (h1 : ∀ W, W = M * 30) (h2 : ∀ W, W = 10 * 36) : 
  M = 12 :=
by
  sorry

end men_in_first_group_l96_96925


namespace negation_of_universal_proposition_l96_96214

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 1) ↔ (∃ x : ℝ, x^2 ≤ 1) :=
by
  sorry

end negation_of_universal_proposition_l96_96214


namespace line_quadrants_l96_96143

theorem line_quadrants (k b : ℝ) (h : ∃ x y : ℝ, y = k * x + b ∧ 
                                          ((x > 0 ∧ y > 0) ∧   -- First quadrant
                                           (x < 0 ∧ y < 0) ∧   -- Third quadrant
                                           (x > 0 ∧ y < 0))) : -- Fourth quadrant
  k > 0 :=
sorry

end line_quadrants_l96_96143


namespace find_x_l96_96809

-- Define the condition as a Lean equation
def equation (x : ℤ) : Prop :=
  45 - (28 - (37 - (x - 19))) = 58

-- The proof statement: if the equation holds, then x = 15
theorem find_x (x : ℤ) (h : equation x) : x = 15 := by
  sorry

end find_x_l96_96809


namespace sum_m_n_is_192_l96_96844

def smallest_prime : ℕ := 2

def largest_four_divisors_under_200 : ℕ :=
  -- we assume this as 190 based on the provided problem's solution
  190

theorem sum_m_n_is_192 :
  smallest_prime = 2 →
  largest_four_divisors_under_200 = 190 →
  smallest_prime + largest_four_divisors_under_200 = 192 :=
by
  intros h1 h2
  sorry

end sum_m_n_is_192_l96_96844


namespace factorial_of_6_is_720_l96_96153

theorem factorial_of_6_is_720 : (Nat.factorial 6) = 720 := by
  sorry

end factorial_of_6_is_720_l96_96153


namespace vec_subtraction_l96_96659

variables (a b : Prod ℝ ℝ)
def vec1 : Prod ℝ ℝ := (1, 2)
def vec2 : Prod ℝ ℝ := (3, 1)

theorem vec_subtraction : (2 * (vec1.fst, vec1.snd) - (vec2.fst, vec2.snd)) = (-1, 3) := by
  -- Proof here, skipped
  sorry

end vec_subtraction_l96_96659


namespace find_DG_l96_96073

theorem find_DG 
  (a b : ℕ) -- sides DE and EC
  (S : ℕ := 19 * (a + b)) -- area of each rectangle
  (k l : ℕ) -- sides DG and CH
  (h1 : S = a * k) 
  (h2 : S = b * l) 
  (h_bc : 19 * (a + b) = S)
  (h_div_a : S % a = 0)
  (h_div_b : S % b = 0)
  : DG = 380 :=
sorry

end find_DG_l96_96073


namespace baseball_batter_at_bats_left_l96_96157

theorem baseball_batter_at_bats_left (L R H_L H_R : ℕ) (h1 : L + R = 600)
    (h2 : H_L + H_R = 192) (h3 : H_L = 25 / 100 * L) (h4 : H_R = 35 / 100 * R) : 
    L = 180 :=
by
  sorry

end baseball_batter_at_bats_left_l96_96157


namespace only_option_d_determines_location_l96_96921

-- Define the problem conditions in Lean
inductive LocationOption where
  | OptionA : LocationOption
  | OptionB : LocationOption
  | OptionC : LocationOption
  | OptionD : LocationOption

-- Define a function that takes a LocationOption and returns whether it can determine a specific location
def determine_location (option : LocationOption) : Prop :=
  match option with
  | LocationOption.OptionD => True
  | LocationOption.OptionA => False
  | LocationOption.OptionB => False
  | LocationOption.OptionC => False

-- Prove that only option D can determine a specific location
theorem only_option_d_determines_location : ∀ (opt : LocationOption), determine_location opt ↔ opt = LocationOption.OptionD := by
  intro opt
  cases opt
  · simp [determine_location, LocationOption.OptionA]
  · simp [determine_location, LocationOption.OptionB]
  · simp [determine_location, LocationOption.OptionC]
  · simp [determine_location, LocationOption.OptionD]

end only_option_d_determines_location_l96_96921


namespace negation_proposition_l96_96355

-- Define the original proposition
def unique_solution (a b : ℝ) (h : a ≠ 0) : Prop :=
  ∀ x1 x2 : ℝ, (a * x1 = b ∧ a * x2 = b) → (x1 = x2)

-- Define the negation of the proposition
def negation_unique_solution (a b : ℝ) (h : a ≠ 0) : Prop :=
  ¬ unique_solution a b h

-- Define a proposition for "no unique solution"
def no_unique_solution (a b : ℝ) (h : a ≠ 0) : Prop :=
  ∃ x1 x2 : ℝ, (a * x1 = b ∧ a * x2 = b) ∧ (x1 ≠ x2)

-- The Lean 4 statement
theorem negation_proposition (a b : ℝ) (h : a ≠ 0) :
  negation_unique_solution a b h :=
sorry

end negation_proposition_l96_96355


namespace cost_of_apples_l96_96804

def cost_per_kilogram (m : ℝ) : ℝ := m
def number_of_kilograms : ℝ := 3

theorem cost_of_apples (m : ℝ) : cost_per_kilogram m * number_of_kilograms = 3 * m :=
by
  unfold cost_per_kilogram number_of_kilograms
  sorry

end cost_of_apples_l96_96804


namespace cookies_baked_on_monday_is_32_l96_96816

-- Definitions for the problem.
variable (X : ℕ)

-- Conditions.
def cookies_baked_on_monday := X
def cookies_baked_on_tuesday := X / 2
def cookies_baked_on_wednesday := 3 * (X / 2) - 4

-- Total cookies at the end of three days.
def total_cookies := cookies_baked_on_monday X + cookies_baked_on_tuesday X + cookies_baked_on_wednesday X

-- Theorem statement to prove the number of cookies baked on Monday.
theorem cookies_baked_on_monday_is_32 : total_cookies X = 92 → cookies_baked_on_monday X = 32 :=
by
  -- We would add the proof steps here.
  sorry

end cookies_baked_on_monday_is_32_l96_96816


namespace correct_factorization_l96_96399

theorem correct_factorization (a b : ℝ) : 
  ((x + 6) * (x - 1) = x^2 + 5 * x - 6) →
  ((x - 2) * (x + 1) = x^2 - x - 2) →
  (a = 1 ∧ b = -6) →
  (x^2 - x - 6 = (x + 2) * (x - 3)) :=
sorry

end correct_factorization_l96_96399


namespace carlos_more_miles_than_dana_after_3_hours_l96_96915

-- Define the conditions
variable (carlos_total_distance : ℕ)
variable (carlos_advantage : ℕ)
variable (dana_total_distance : ℕ)
variable (time_hours : ℕ)

-- State the condition values that are given in the problem
def conditions : Prop :=
  carlos_total_distance = 50 ∧
  carlos_advantage = 5 ∧
  dana_total_distance = 40 ∧
  time_hours = 3

-- State the proof goal
theorem carlos_more_miles_than_dana_after_3_hours
  (h : conditions carlos_total_distance carlos_advantage dana_total_distance time_hours) :
  carlos_total_distance - dana_total_distance = 10 :=
by
  sorry

end carlos_more_miles_than_dana_after_3_hours_l96_96915


namespace tan_30_eq_sqrt3_div_3_l96_96912

/-- Statement that proves the value of tang of 30 degrees, given the cosine
    and sine values. -/
theorem tan_30_eq_sqrt3_div_3 
  (cos_30 : Real) (sin_30 : Real) 
  (hcos : cos_30 = Real.sqrt 3 / 2) 
  (hsin : sin_30 = 1 / 2) : 
    Real.tan 30 = Real.sqrt 3 / 3 := 
by 
  sorry

end tan_30_eq_sqrt3_div_3_l96_96912


namespace highest_visitors_at_4pm_yellow_warning_time_at_12_30pm_l96_96146

-- Definitions for cumulative visitors entering and leaving
def y (x : ℕ) : ℕ := 850 * x + 100
def z (x : ℕ) : ℕ := 200 * x - 200

-- Definition for total number of visitors at time x
def w (x : ℕ) : ℕ := y x - z x

-- Proof problem statements
theorem highest_visitors_at_4pm :
  ∀x, x ≤ 9 → w 9 ≥ w x :=
sorry

theorem yellow_warning_time_at_12_30pm :
  ∃x, w x = 2600 :=
sorry

end highest_visitors_at_4pm_yellow_warning_time_at_12_30pm_l96_96146


namespace find_a_plus_b_l96_96203

noncomputable def f (a b x : ℝ) : ℝ := a * x ^ 2 + b * x + 3 * a + b

theorem find_a_plus_b (a b : ℝ) (h1 : ∀ x : ℝ, f a b x = f a b (-x)) (h2 : 2 * a = 3 - a) : a + b = 1 :=
by
  unfold f at h1
  sorry

end find_a_plus_b_l96_96203


namespace lower_bound_fraction_sum_l96_96555

open Real

theorem lower_bound_fraction_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) : 
  (1 / (3 * a) + 3 / b) ≥ 8 / 3 :=
by 
  sorry

end lower_bound_fraction_sum_l96_96555


namespace number_of_chickens_l96_96062

def cost_per_chicken := 3
def total_cost := 15
def potato_cost := 6
def remaining_amount := total_cost - potato_cost

theorem number_of_chickens : (total_cost - potato_cost) / cost_per_chicken = 3 := by
  sorry

end number_of_chickens_l96_96062


namespace an_general_term_sum_bn_l96_96724

open Nat

variable (a : ℕ → ℕ)
variable (b : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (T : ℕ → ℕ)

-- Conditions
axiom a3 : a 3 = 3
axiom S6 : S 6 = 21
axiom Sn : ∀ n, S n = n * (a 1 + a n) / 2

-- Define bn based on the given condition for bn = an + 2^n
def bn (n : ℕ) : ℕ := a n + 2^n

-- Define Tn based on the given condition for Tn.
def Tn (n : ℕ) : ℕ := (n * (n + 1)) / 2 + (2^(n + 1) - 2)

-- Prove the general term formula of the arithmetic sequence an
theorem an_general_term (n : ℕ) : a n = n :=
by
  sorry

-- Prove the sum of the first n terms of the sequence bn
theorem sum_bn (n : ℕ) : T n = Tn n :=
by
  sorry

end an_general_term_sum_bn_l96_96724


namespace asep_wins_in_at_most_n_minus_5_div_4_steps_l96_96400

theorem asep_wins_in_at_most_n_minus_5_div_4_steps (n : ℕ) (h : n ≥ 14) : 
  ∃ f : ℕ → ℕ, (∀ X d : ℕ, 0 < d → d ∣ X → (X' = X + d ∨ X' = X - d) → (f X' ≤ f X + 1)) ∧ f n ≤ (n - 5) / 4 := 
sorry

end asep_wins_in_at_most_n_minus_5_div_4_steps_l96_96400


namespace total_apple_trees_l96_96428

-- Definitions and conditions
def ava_trees : ℕ := 9
def lily_trees : ℕ := ava_trees - 3
def total_trees : ℕ := ava_trees + lily_trees

-- Statement to be proved
theorem total_apple_trees :
  total_trees = 15 := by
  sorry

end total_apple_trees_l96_96428


namespace Johnson_family_seating_l96_96489

theorem Johnson_family_seating : 
  ∃ n : ℕ, number_of_ways_to_seat_Johnson_family = n ∧ n = 288 :=
sorry

end Johnson_family_seating_l96_96489


namespace average_speed_correct_l96_96445

def biking_time : ℕ := 30 -- in minutes
def biking_speed : ℕ := 16 -- in mph
def walking_time : ℕ := 90 -- in minutes
def walking_speed : ℕ := 4 -- in mph

theorem average_speed_correct :
  (biking_time / 60 * biking_speed + walking_time / 60 * walking_speed) / ((biking_time + walking_time) / 60) = 7 := by
  sorry

end average_speed_correct_l96_96445


namespace proof_problem_l96_96298

theorem proof_problem (s t: ℤ) (h : 514 - s = 600 - t) : s < t ∧ t - s = 86 :=
by
  sorry

end proof_problem_l96_96298


namespace compound_interest_doubling_time_l96_96588

theorem compound_interest_doubling_time :
  ∃ (t : ℕ), (0.15 : ℝ) = 0.15 ∧ ∀ (n : ℕ), (n = 1) →
               (2 : ℝ) < (1 + 0.15) ^ t ∧ t = 5 :=
by
  sorry

end compound_interest_doubling_time_l96_96588


namespace sum_of_24_terms_l96_96904

variable (a_1 d : ℝ)

def a (n : ℕ) : ℝ := a_1 + (n - 1) * d

theorem sum_of_24_terms 
  (h : (a 5 + a 10 + a 15 + a 20 = 20)) : 
  (12 * (2 * a_1 + 23 * d) = 120) :=
by
  sorry

end sum_of_24_terms_l96_96904


namespace point_in_first_quadrant_l96_96947

/-- In the Cartesian coordinate system, if a point P has x-coordinate 2 and y-coordinate 4, it lies in the first quadrant. -/
theorem point_in_first_quadrant (x y : ℝ) (h1 : x = 2) (h2 : y = 4) : 
  x > 0 ∧ y > 0 → 
  (x, y).1 = 2 ∧ (x, y).2 = 4 → 
  (x > 0 ∧ y > 0) := 
by
  intros
  sorry

end point_in_first_quadrant_l96_96947


namespace greatest_possible_remainder_l96_96607

theorem greatest_possible_remainder (x : ℕ) : ∃ r, r < 11 ∧ x % 11 = r ∧ r = 10 :=
by
  exists 10
  sorry

end greatest_possible_remainder_l96_96607


namespace bouquets_sold_on_Monday_l96_96869

theorem bouquets_sold_on_Monday
  (tuesday_three_times_monday : ∀ (x : ℕ), bouquets_sold_Tuesday = 3 * x)
  (wednesday_third_of_tuesday : ∀ (bouquets_sold_Tuesday : ℕ), bouquets_sold_Wednesday = bouquets_sold_Tuesday / 3)
  (total_bouquets : bouquets_sold_Monday + bouquets_sold_Tuesday + bouquets_sold_Wednesday = 60)
  : bouquets_sold_Monday = 12 := 
sorry

end bouquets_sold_on_Monday_l96_96869


namespace power_mean_inequality_l96_96846

variables {a b c : ℝ}
variables {n p q r : ℕ}

theorem power_mean_inequality 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hn : 0 < n)
  (hpqr_nonneg : 0 ≤ p ∧ 0 ≤ q ∧ 0 ≤ r)
  (sum_pqr : p + q + r = n) :
  a^n + b^n + c^n ≥ a^p * b^q * c^r + a^r * b^p * c^q + a^q * b^r * c^p :=
sorry

end power_mean_inequality_l96_96846


namespace wire_leftover_length_l96_96106

-- Define given conditions as variables/constants
def initial_wire_length : ℝ := 60
def side_length : ℝ := 9
def sides_in_square : ℕ := 4

-- Define the theorem: prove leftover wire length is 24 after creating the square
theorem wire_leftover_length :
  initial_wire_length - sides_in_square * side_length = 24 :=
by
  -- proof steps are not required, so we use sorry to indicate where the proof should be
  sorry

end wire_leftover_length_l96_96106


namespace a8_div_b8_l96_96356

variable (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)

-- Given Conditions
axiom sum_a (n : ℕ) : S n = (n * (a 1 + (n - 1) * a 2)) / 2 -- Sum of first n terms of arithmetic sequence a_n
axiom sum_b (n : ℕ) : T n = (n * (b 1 + (n - 1) * b 2)) / 2 -- Sum of first n terms of arithmetic sequence b_n
axiom ratio (n : ℕ) : S n / T n = (7 * n + 3) / (n + 3)

-- Proof statement
theorem a8_div_b8 : a 8 / b 8 = 6 := by
  sorry

end a8_div_b8_l96_96356


namespace trees_left_after_typhoon_l96_96037

-- Define the initial count of trees and the number of trees that died
def initial_trees := 150
def trees_died := 24

-- Define the expected number of trees left
def expected_trees_left := 126

-- The statement to be proven: after trees died, the number of trees left is as expected
theorem trees_left_after_typhoon : (initial_trees - trees_died) = expected_trees_left := by
  sorry

end trees_left_after_typhoon_l96_96037


namespace bounces_to_below_30_cm_l96_96396

theorem bounces_to_below_30_cm :
  ∃ (b : ℕ), (256 * (3 / 4)^b < 30) ∧
            (∀ (k : ℕ), k < b -> 256 * (3 / 4)^k ≥ 30) :=
by 
  sorry

end bounces_to_below_30_cm_l96_96396


namespace solution_one_solution_two_l96_96209

section

variables {a x : ℝ}

def f (x : ℝ) (a : ℝ) := |2 * x - a| - |x + 1|

-- (1) Prove the solution set for f(x) > 2 when a = 1 is (-∞, -2/3) ∪ (4, ∞)
theorem solution_one (x : ℝ) : f x 1 > 2 ↔ x < -2/3 ∨ x > 4 :=
by sorry

-- (2) Prove the range of a for which f(x) + |x + 1| + x > a² - 1/2 always holds for x ∈ ℝ is (-1/2, 1)
theorem solution_two (a : ℝ) : 
  (∀ x, f x a + |x + 1| + x > a^2 - 1/2) ↔ -1/2 < a ∧ a < 1 :=
by sorry

end

end solution_one_solution_two_l96_96209


namespace perfect_square_trinomial_l96_96890

theorem perfect_square_trinomial (m : ℤ) : 
  (x^2 - (m - 3) * x + 16 = (x - 4)^2) ∨ (x^2 - (m - 3) * x + 16 = (x + 4)^2) ↔ (m = -5 ∨ m = 11) := by
  sorry

end perfect_square_trinomial_l96_96890


namespace problem1_l96_96048

theorem problem1 (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) :
  (Real.sin (2 * α) - Real.cos α ^ 2) / (1 + Real.cos (2 * α)) = -5 / 6 := 
  sorry

end problem1_l96_96048


namespace car_distribution_l96_96092

theorem car_distribution :
  let total_cars := 5650000
  let first_supplier := 1000000
  let second_supplier := first_supplier + 500000
  let third_supplier := first_supplier + second_supplier
  let total_distributed_first_three := first_supplier + second_supplier + third_supplier
  let remaining_cars := total_cars - total_distributed_first_three
  let fourth_supplier := remaining_cars / 2
  let fifth_supplier := remaining_cars / 2
  fourth_supplier = 325000 :=
by
  let total_cars := 5650000
  let first_supplier := 1000000
  let second_supplier := first_supplier + 500000
  let third_supplier := first_supplier + second_supplier
  let total_distributed_first_three := first_supplier + second_supplier + third_supplier
  let remaining_cars := total_cars - total_distributed_first_three
  let fourth_supplier := remaining_cars / 2
  let fifth_supplier := remaining_cars / 2
  sorry

end car_distribution_l96_96092


namespace triangle_side_lengths_consecutive_l96_96544

theorem triangle_side_lengths_consecutive (n : ℕ) (a b c A : ℕ) 
  (h1 : a = n - 1) (h2 : b = n) (h3 : c = n + 1) (h4 : A = n + 2)
  (h5 : 2 * A * A = 3 * n^2 * (n^2 - 4)) :
  a = 3 ∧ b = 4 ∧ c = 5 :=
sorry

end triangle_side_lengths_consecutive_l96_96544


namespace smaller_rectangle_area_l96_96338

theorem smaller_rectangle_area (L_h S_h : ℝ) (L_v S_v : ℝ) 
  (ratio_h : L_h = (8 / 7) * S_h) 
  (ratio_v : L_v = (9 / 4) * S_v) 
  (area_large : L_h * L_v = 108) :
  S_h * S_v = 42 :=
sorry

end smaller_rectangle_area_l96_96338


namespace inequality_for_pos_a_b_c_d_l96_96591

theorem inequality_for_pos_a_b_c_d
  (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b) * (b + c) * (c + d) * (d + a) * (1 + (abcd ^ (1/4)))^4
  ≥ 16 * abcd * (1 + a) * (1 + b) * (1 + c) * (1 + d) :=
by
  sorry

end inequality_for_pos_a_b_c_d_l96_96591


namespace rhombus_side_length_l96_96820

-- Definitions
def is_rhombus_perimeter (P s : ℝ) : Prop := P = 4 * s

-- Theorem to prove
theorem rhombus_side_length (P : ℝ) (hP : P = 4) : ∃ s : ℝ, is_rhombus_perimeter P s ∧ s = 1 :=
by
  sorry

end rhombus_side_length_l96_96820


namespace coffee_equals_milk_l96_96669

theorem coffee_equals_milk (S : ℝ) (h : 0 < S ∧ S < 1/2) :
  let initial_milk := 1 / 2
  let initial_coffee := 1 / 2
  let glass1_initial := initial_milk
  let glass2_initial := initial_coffee
  let glass2_after_first_transfer := glass2_initial + S
  let coffee_transferred_back := (S * initial_coffee) / (initial_coffee + S)
  let milk_transferred_back := (S^2) / (initial_coffee + S)
  let glass1_after_second_transfer := glass1_initial - S + milk_transferred_back
  let glass2_after_second_transfer := glass2_initial + S - coffee_transferred_back
  (glass1_initial - S + milk_transferred_back) = (glass2_initial + S - coffee_transferred_back) :=
sorry

end coffee_equals_milk_l96_96669


namespace find_a4_l96_96410

variables {a : ℕ → ℝ} (q : ℝ) (h_positive : ∀ n, 0 < a n)
variables (h_seq : ∀ n, a (n+1) = q * a n)
variables (h1 : a 1 + (2/3) * a 2 = 3)
variables (h2 : (a 4)^2 = (1/9) * a 3 * a 7)

-- Proof problem statement
theorem find_a4 : a 4 = 27 :=
sorry

end find_a4_l96_96410


namespace find_point_P_l96_96177

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

end find_point_P_l96_96177


namespace range_of_m_n_l96_96308

noncomputable def tangent_condition (m n : ℝ) : Prop :=
  ∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 1 → (m + 1) * x + (n + 1) * y - 2 = 0

theorem range_of_m_n (m n : ℝ) :
  tangent_condition m n →
  (m + n ≤ 2 - 2 * Real.sqrt 2 ∨ m + n ≥ 2 + 2 * Real.sqrt 2) :=
sorry

end range_of_m_n_l96_96308


namespace group_C_forms_triangle_l96_96937

theorem group_C_forms_triangle :
  ∀ (a b c : ℕ), (a + b > c ∧ a + c > b ∧ b + c > a) ↔ ((a, b, c) = (2, 3, 4)) :=
by
  -- we'll prove the forward and backward directions separately
  sorry

end group_C_forms_triangle_l96_96937


namespace guest_bedroom_area_l96_96576

theorem guest_bedroom_area 
  (master_bedroom_bath_area : ℝ)
  (kitchen_guest_bath_living_area : ℝ)
  (total_rent : ℝ)
  (rate_per_sqft : ℝ)
  (num_guest_bedrooms : ℕ)
  (area_guest_bedroom : ℝ) :
  master_bedroom_bath_area = 500 →
  kitchen_guest_bath_living_area = 600 →
  total_rent = 3000 →
  rate_per_sqft = 2 →
  num_guest_bedrooms = 2 →
  (total_rent / rate_per_sqft) - (master_bedroom_bath_area + kitchen_guest_bath_living_area) / num_guest_bedrooms = area_guest_bedroom → 
  area_guest_bedroom = 200 := by
  sorry

end guest_bedroom_area_l96_96576


namespace buttons_on_first_type_of_shirt_l96_96129

/--
The GooGoo brand of clothing manufactures two types of shirts.
- The first type of shirt has \( x \) buttons.
- The second type of shirt has 5 buttons.
- The department store ordered 200 shirts of each type.
- A total of 1600 buttons are used for the entire order.

Prove that the first type of shirt has exactly 3 buttons.
-/
theorem buttons_on_first_type_of_shirt (x : ℕ) 
  (h1 : 200 * x + 200 * 5 = 1600) : 
  x = 3 :=
  sorry

end buttons_on_first_type_of_shirt_l96_96129


namespace average_of_remaining_two_numbers_l96_96040

theorem average_of_remaining_two_numbers 
(A B C D E F G H : ℝ) 
(h_avg1 : (A + B + C + D + E + F + G + H) / 8 = 4.5) 
(h_avg2 : (A + B + C) / 3 = 5.2) 
(h_avg3 : (D + E + F) / 3 = 3.6) : 
  ((G + H) / 2 = 4.8) :=
sorry

end average_of_remaining_two_numbers_l96_96040


namespace customer_bought_two_pens_l96_96495

noncomputable def combination (n k : ℕ) : ℝ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem customer_bought_two_pens :
  ∃ n : ℕ, combination 5 n / combination 8 n = 0.3571428571428571 ↔ n = 2 := by
  sorry

end customer_bought_two_pens_l96_96495


namespace cost_of_first_20_kgs_l96_96003

theorem cost_of_first_20_kgs (l q : ℕ)
  (h1 : 30 * l + 3 * q = 168)
  (h2 : 30 * l + 6 * q = 186) :
  20 * l = 100 :=
by
  sorry

end cost_of_first_20_kgs_l96_96003


namespace tan_alpha_implication_l96_96236

theorem tan_alpha_implication (α : ℝ) (h : Real.tan α = 2) :
    (2 * Real.sin α - Real.cos α) / (2 * Real.sin α + Real.cos α) = 3 / 5 := 
by 
  sorry

end tan_alpha_implication_l96_96236


namespace articles_produced_l96_96290

theorem articles_produced (a b c d f p q r g : ℕ) :
  (a * b * c = d) → 
  ((p * q * r * d * g) / (a * b * c * f) = pqr * d * g / (abc * f)) :=
by
  sorry

end articles_produced_l96_96290


namespace zoey_preparation_months_l96_96171
open Nat

-- Define months as integers assuming 1 = January, 5 = May, 9 = September, etc.
def month_start : ℕ := 5 -- May
def month_exam : ℕ := 9 -- September

-- The function to calculate the number of months of preparation excluding the exam month.
def months_of_preparation (start : ℕ) (exam : ℕ) : ℕ := (exam - start)

theorem zoey_preparation_months :
  months_of_preparation month_start month_exam = 4 := by
  sorry

end zoey_preparation_months_l96_96171


namespace part1_solution_set_part2_range_of_a_l96_96954

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_solution_set_part2_range_of_a_l96_96954


namespace proof_problem_l96_96609

variable (a b c : ℝ)
variable (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h_prod : a * b * c = 1)
variable (h_ineq : a^2011 + b^2011 + c^2011 < (1 / a)^2011 + (1 / b)^2011 + (1 / c)^2011)

theorem proof_problem : a + b + c < 1 / a + 1 / b + 1 / c := 
  sorry

end proof_problem_l96_96609


namespace change_in_responses_max_min_diff_l96_96478

open Classical

theorem change_in_responses_max_min_diff :
  let initial_yes := 40
  let initial_no := 40
  let initial_undecided := 20
  let end_yes := 60
  let end_no := 30
  let end_undecided := 10
  let min_change := 20
  let max_change := 80
  max_change - min_change = 60 := by
  intros; sorry

end change_in_responses_max_min_diff_l96_96478


namespace total_chocolate_bars_l96_96262

theorem total_chocolate_bars :
  let num_large_boxes := 45
  let num_small_boxes_per_large_box := 36
  let num_chocolate_bars_per_small_box := 72
  num_large_boxes * num_small_boxes_per_large_box * num_chocolate_bars_per_small_box = 116640 :=
by
  sorry

end total_chocolate_bars_l96_96262


namespace area_original_is_504_l96_96661

-- Define the sides of the three rectangles
variable (a1 b1 a2 b2 a3 b3 : ℕ)

-- Define the perimeters of the three rectangles
def P1 := 2 * (a1 + b1)
def P2 := 2 * (a2 + b2)
def P3 := 2 * (a3 + b3)

-- Define the conditions given in the problem
axiom P1_equal_P2_plus_20 : P1 = P2 + 20
axiom P2_equal_P3_plus_16 : P2 = P3 + 16

-- Define the calculation for the area of the original rectangle
def area_original := a1 * b1

-- Proof goal: the area of the original rectangle is 504
theorem area_original_is_504 : area_original = 504 := 
sorry

end area_original_is_504_l96_96661


namespace y_order_of_quadratic_l96_96796

theorem y_order_of_quadratic (k : ℝ) (y1 y2 y3 : ℝ) :
  (y1 = (-4)^2 + 4 * (-4) + k) → 
  (y2 = (-1)^2 + 4 * (-1) + k) → 
  (y3 = (1)^2 + 4 * (1) + k) → 
  y2 < y1 ∧ y1 < y3 :=
by
  intro hy1 hy2 hy3
  sorry

end y_order_of_quadratic_l96_96796


namespace first_grade_sample_count_l96_96438

-- Defining the total number of students and their ratio in grades 1, 2, and 3.
def total_students : ℕ := 2400
def ratio_grade1 : ℕ := 5
def ratio_grade2 : ℕ := 4
def ratio_grade3 : ℕ := 3
def total_ratio := ratio_grade1 + ratio_grade2 + ratio_grade3

-- Defining the sample size
def sample_size : ℕ := 120

-- Proving that the number of first-grade students sampled should be 50.
theorem first_grade_sample_count : 
  (sample_size * ratio_grade1) / total_ratio = 50 :=
by
  -- sorry is added here to skip the proof
  sorry

end first_grade_sample_count_l96_96438


namespace ratio_a_over_b_l96_96625

-- Definitions of conditions
def func (a b x : ℝ) : ℝ := a * x^2 + b
def derivative (a b x : ℝ) : ℝ := 2 * a * x

-- Given conditions
variables (a b : ℝ)
axiom tangent_slope : derivative a b 1 = 2
axiom point_on_graph : func a b 1 = 3

-- Statement to prove
theorem ratio_a_over_b : a / b = 1 / 2 :=
by sorry

end ratio_a_over_b_l96_96625


namespace total_chairs_calculation_l96_96794

-- Definitions of the conditions
def numIndoorTables : Nat := 9
def numOutdoorTables : Nat := 11
def chairsPerIndoorTable : Nat := 10
def chairsPerOutdoorTable : Nat := 3

-- The proposition we want to prove
theorem total_chairs_calculation :
  numIndoorTables * chairsPerIndoorTable + numOutdoorTables * chairsPerOutdoorTable = 123 := by
sorry

end total_chairs_calculation_l96_96794


namespace donuts_eaten_on_monday_l96_96111

theorem donuts_eaten_on_monday (D : ℕ) (h1 : D + D / 2 + 4 * D = 49) : 
  D = 9 :=
sorry

end donuts_eaten_on_monday_l96_96111


namespace two_digit_number_multiple_l96_96322

theorem two_digit_number_multiple (x : ℕ) (h1 : x ≥ 10) (h2 : x < 100) 
(h3 : ∃ k : ℕ, x + 1 = 3 * k) 
(h4 : ∃ k : ℕ, x + 1 = 4 * k) 
(h5 : ∃ k : ℕ, x + 1 = 5 * k) 
(h6 : ∃ k : ℕ, x + 1 = 7 * k) 
: x = 83 := 
sorry

end two_digit_number_multiple_l96_96322


namespace complex_number_quadrant_l96_96348

theorem complex_number_quadrant (a b : ℝ) (h1 : (2 + a * (0+1*I)) / (1 + 1*I) = b + 1*I) (h2: a = 4) (h3: b = 3) : 
  0 < a ∧ 0 < b :=
by
  sorry

end complex_number_quadrant_l96_96348


namespace length_squared_of_segment_CD_is_196_l96_96385

theorem length_squared_of_segment_CD_is_196 :
  ∃ (C D : ℝ × ℝ), 
    (C.2 = 3 * C.1 ^ 2 + 6 * C.1 - 2) ∧
    (D.2 = 3 * (2 - C.1) ^ 2 + 6 * (2 - C.1) - 2) ∧
    (1 : ℝ) = (C.1 + D.1) / 2 ∧
    (0 : ℝ) = (C.2 + D.2) / 2 ∧
    ((C.1 - D.1) ^ 2 + (C.2 - D.2) ^ 2 = 196) :=
by
  -- The proof would go here
  sorry

end length_squared_of_segment_CD_is_196_l96_96385


namespace find_coefficients_l96_96078

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x^2 + 3 * x + 4

noncomputable def h (a b c x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_coefficients :
  ∃ a b c : ℝ, (∀ s : ℝ, f s = 0 → h a b c (s^3) = 0) ∧
    (a, b, c) = (-6, -9, 20) :=
sorry

end find_coefficients_l96_96078


namespace initial_fish_l96_96277

-- Define the conditions of the problem
def fish_bought : Float := 280.0
def current_fish : Float := 492.0

-- Define the question to be proved
theorem initial_fish (x : Float) (h : x + fish_bought = current_fish) : x = 212 :=
by 
  sorry

end initial_fish_l96_96277


namespace parabola_focus_coordinates_l96_96192

theorem parabola_focus_coordinates (x y : ℝ) (h : y = -2 * x^2) : (0, -1 / 8) = (0, (-1 / 2) * (y: ℝ)) :=
sorry

end parabola_focus_coordinates_l96_96192


namespace circle_tangent_line_l96_96218

noncomputable def line_eq (x : ℝ) : ℝ := 2 * x + 1
noncomputable def circle_eq (x y b : ℝ) : ℝ := x^2 + (y - b)^2

theorem circle_tangent_line 
  (b : ℝ) 
  (tangency : ∃ b, (1 - b) / (0 - 1) = -(1 / 2)) 
  (center_point : 1^2 + (3 - b)^2 = 5 / 4) : 
  circle_eq 1 3 b = circle_eq 0 b (7/2) :=
sorry

end circle_tangent_line_l96_96218


namespace rate_is_correct_l96_96592

noncomputable def rate_of_interest (P A T : ℝ) : ℝ :=
  let SI := A - P
  (SI * 100) / (P * T)

theorem rate_is_correct :
  rate_of_interest 10000 18500 8 = 10.625 := 
by
  sorry

end rate_is_correct_l96_96592


namespace root_of_equation_imp_expression_eq_one_l96_96929

variable (m : ℝ)

theorem root_of_equation_imp_expression_eq_one
  (h : m^2 - m - 1 = 0) : m^2 - m = 1 :=
  sorry

end root_of_equation_imp_expression_eq_one_l96_96929


namespace intersection_complementA_setB_l96_96739

noncomputable def setA : Set ℝ := { x | abs x > 1 }

noncomputable def setB : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

noncomputable def complementA : Set ℝ := { x | abs x ≤ 1 }

theorem intersection_complementA_setB : 
  (complementA ∩ setB) = { x | 0 ≤ x ∧ x ≤ 1 } := by
  sorry

end intersection_complementA_setB_l96_96739


namespace largest_divisor_of_10000_not_dividing_9999_l96_96479

theorem largest_divisor_of_10000_not_dividing_9999 : ∃ d, d ∣ 10000 ∧ ¬ (d ∣ 9999) ∧ ∀ y, (y ∣ 10000 ∧ ¬ (y ∣ 9999)) → y ≤ d := 
by
  sorry

end largest_divisor_of_10000_not_dividing_9999_l96_96479


namespace find_integer_pairs_l96_96830

theorem find_integer_pairs :
  ∀ (a b : ℕ), 0 < a → 0 < b → a * b + 2 = a^3 + 2 * b →
  (a = 1 ∧ b = 1) ∨ (a = 3 ∧ b = 25) ∨ (a = 4 ∧ b = 31) ∨ (a = 5 ∧ b = 41) ∨ (a = 8 ∧ b = 85) :=
by
  intros a b ha hb hab_eq
  -- Proof goes here
  sorry

end find_integer_pairs_l96_96830


namespace exists_ij_aij_gt_ij_l96_96859

theorem exists_ij_aij_gt_ij (a : ℕ → ℕ → ℕ) 
  (h_a_positive : ∀ i j, 0 < a i j)
  (h_a_distribution : ∀ k, (∃ S : Finset (ℕ × ℕ), S.card = 8 ∧ ∀ ij : ℕ × ℕ, ij ∈ S ↔ a ij.1 ij.2 = k)) :
  ∃ i j, a i j > i * j :=
by
  sorry

end exists_ij_aij_gt_ij_l96_96859


namespace problem_solution_l96_96685

-- Define the ellipse equation and foci positions.
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + (y^2 / 2) = 1
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

-- Define the line equation
def line (x y k : ℝ) : Prop := y = k * x + 1

-- Define the intersection points A and B
variable (A B : ℝ × ℝ)
variable (k : ℝ)

-- Define the points lie on the line and ellipse
def A_on_line := ∃ x y, A = (x, y) ∧ line x y k
def B_on_line := ∃ x y, B = (x, y) ∧ line x y k

-- Define the parallel and perpendicular conditions
def parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k, v1.1 = k * v2.1 ∧ v1.2 = k * v2.2
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Lean theorem for the conclusions of the problem
theorem problem_solution (A_cond : A_on_line A k ∧ ellipse A.1 A.2) 
                          (B_cond : B_on_line B k ∧ ellipse B.1 B.2) :

  -- Prove these two statements
  ¬ parallel (A.1 + 1, A.2) (B.1 - 1, B.2) ∧
  ¬ perpendicular (A.1 + 1, A.2) (A.1 - 1, A.2) :=
sorry

end problem_solution_l96_96685


namespace find_t_l96_96331

variable (t : ℚ)

def point_on_line (p1 p2 p3 : ℚ × ℚ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem find_t (t : ℚ) : point_on_line (3, 0) (0, 7) (t, 8) → t = -3 / 7 := by
  sorry

end find_t_l96_96331


namespace monica_total_savings_l96_96698

noncomputable def weekly_savings (week: ℕ) : ℕ :=
  if week < 6 then 15 + 5 * week
  else if week < 11 then 40 - 5 * (week - 5)
  else weekly_savings (week % 10)

theorem monica_total_savings : 
  let cycle_savings := (15 + 20 + 25 + 30 + 35 + 40) + (40 + 35 + 30 + 25 + 20 + 15) - 40 
  let total_savings := 5 * cycle_savings
  total_savings = 1450 := by
  sorry

end monica_total_savings_l96_96698


namespace tangent_line_with_smallest_slope_l96_96849

-- Define the given curve
def curve (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x - 10

-- Define the derivative of the given curve
def curve_derivative (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 6

-- Define the equation of the tangent line with the smallest slope
def tangent_line (x y : ℝ) : Prop := 3 * x - y = 11

-- Prove that the equation of the tangent line with the smallest slope on the curve is 3x - y - 11 = 0
theorem tangent_line_with_smallest_slope :
  ∃ x y : ℝ, curve x = y ∧ curve_derivative x = 3 ∧ tangent_line x y :=
by
  sorry

end tangent_line_with_smallest_slope_l96_96849


namespace min_value_f_min_achieved_l96_96504

noncomputable def f (x : ℝ) : ℝ := (1 / (x - 3)) + x

theorem min_value_f : ∀ x : ℝ, x > 3 → f x ≥ 5 :=
by
  intro x hx
  sorry

theorem min_achieved : f 4 = 5 :=
by
  sorry

end min_value_f_min_achieved_l96_96504


namespace find_m_l96_96981

def g (x : ℤ) (A : ℤ) (B : ℤ) (C : ℤ) : ℤ := A * x^2 + B * x + C

theorem find_m (A B C m : ℤ) 
  (h1 : g 2 A B C = 0)
  (h2 : 100 < g 9 A B C ∧ g 9 A B C < 110)
  (h3 : 150 < g 10 A B C ∧ g 10 A B C < 160)
  (h4 : 10000 * m < g 200 A B C ∧ g 200 A B C < 10000 * (m + 1)) : 
  m = 16 :=
sorry

end find_m_l96_96981


namespace sine_shift_l96_96571

variable (m : ℝ)

theorem sine_shift (h : Real.sin 5.1 = m) : Real.sin 365.1 = m :=
by
  sorry

end sine_shift_l96_96571


namespace cinematic_academy_member_count_l96_96879

theorem cinematic_academy_member_count (M : ℝ) 
  (h : (1 / 4) * M = 192.5) : M = 770 := 
by 
  -- proof omitted
  sorry

end cinematic_academy_member_count_l96_96879


namespace graduating_class_total_students_l96_96775

theorem graduating_class_total_students (boys girls students : ℕ) (h1 : girls = boys + 69) (h2 : boys = 208) :
  students = boys + girls → students = 485 :=
by
  sorry

end graduating_class_total_students_l96_96775


namespace solve_for_z_l96_96819

theorem solve_for_z (z : ℂ) (i : ℂ) (h : i^2 = -1) : 3 + 2 * i * z = 5 - 3 * i * z → z = - (2 * i) / 5 :=
by
  intro h_equation
  -- Proof steps will be provided here.
  sorry

end solve_for_z_l96_96819


namespace carlotta_tantrum_time_l96_96977

theorem carlotta_tantrum_time :
  (∀ (T P S : ℕ), 
   S = 6 ∧ T + P + S = 54 ∧ P = 3 * S → T = 5 * S) :=
by
  intro T P S
  rintro ⟨hS, hTotal, hPractice⟩
  sorry

end carlotta_tantrum_time_l96_96977


namespace smart_charging_piles_growth_l96_96867

noncomputable def a : ℕ := 301
noncomputable def b : ℕ := 500
variable (x : ℝ) -- Monthly average growth rate

theorem smart_charging_piles_growth :
  a * (1 + x) ^ 2 = b :=
by
  -- Proof should go here
  sorry

end smart_charging_piles_growth_l96_96867


namespace amount_over_budget_l96_96266

-- Define the prices of each item
def cost_necklace_A : ℕ := 34
def cost_necklace_B : ℕ := 42
def cost_necklace_C : ℕ := 50
def cost_first_book := cost_necklace_A + 20
def cost_second_book := cost_necklace_C - 10

-- Define Bob's budget
def budget : ℕ := 100

-- Define the total cost
def total_cost := cost_necklace_A + cost_necklace_B + cost_necklace_C + cost_first_book + cost_second_book

-- Prove the amount over budget
theorem amount_over_budget : total_cost - budget = 120 := by
  sorry

end amount_over_budget_l96_96266


namespace half_radius_circle_y_l96_96269

-- Conditions
def circle_x_circumference (C : ℝ) : Prop :=
  C = 20 * Real.pi

def circle_x_and_y_same_area (r R : ℝ) : Prop :=
  Real.pi * r^2 = Real.pi * R^2

-- Problem statement: Prove that half the radius of circle y is 5
theorem half_radius_circle_y (r R : ℝ) (hx : circle_x_circumference (2 * Real.pi * r)) (hy : circle_x_and_y_same_area r R) : R / 2 = 5 :=
by sorry

end half_radius_circle_y_l96_96269


namespace lowest_possible_number_of_students_l96_96717

theorem lowest_possible_number_of_students : ∃ n : ℕ, (n > 0) ∧ (∃ k1 : ℕ, n = 10 * k1) ∧ (∃ k2 : ℕ, n = 24 * k2) ∧ n = 120 :=
by
  sorry

end lowest_possible_number_of_students_l96_96717


namespace workers_in_workshop_l96_96049

theorem workers_in_workshop :
  (∀ (W : ℕ), 8000 * W = 12000 * 7 + 6000 * (W - 7) → W = 21) :=
by
  intro W h
  sorry

end workers_in_workshop_l96_96049


namespace sequence_n_value_l96_96449

theorem sequence_n_value (a : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 3) (h3 : a n = 2008) : n = 670 :=
by
 sorry

end sequence_n_value_l96_96449


namespace find_p_l96_96185

theorem find_p
  (A B C r s p q : ℝ)
  (h1 : A ≠ 0)
  (h2 : r + s = -B / A)
  (h3 : r * s = C / A)
  (h4 : r^3 + s^3 = -p) :
  p = (B^3 - 3 * A * B * C) / A^3 :=
by {
  sorry
}

end find_p_l96_96185


namespace find_d_minus_b_l96_96289

theorem find_d_minus_b (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a^5 = b^4) (h2 : c^3 = d^2) (h3 : c - a = 19) : d - b = 757 := 
by sorry

end find_d_minus_b_l96_96289


namespace pentagon_position_3010_l96_96805

def rotate_72 (s : String) : String :=
match s with
| "ABCDE" => "EABCD"
| "EABCD" => "DCBAE"
| "DCBAE" => "EDABC"
| "EDABC" => "ABCDE"
| _ => s

def reflect_vertical (s : String) : String :=
match s with
| "EABCD" => "DCBAE"
| "DCBAE" => "EABCD"
| _ => s

def transform (s : String) (n : Nat) : String :=
match n % 5 with
| 0 => s
| 1 => reflect_vertical (rotate_72 s)
| 2 => rotate_72 (reflect_vertical (rotate_72 s))
| 3 => reflect_vertical (rotate_72 (reflect_vertical (rotate_72 s)))
| 4 => rotate_72 (reflect_vertical (rotate_72 (reflect_vertical (rotate_72 s))))
| _ => s

theorem pentagon_position_3010 :
  transform "ABCDE" 3010 = "ABCDE" :=
by 
  sorry

end pentagon_position_3010_l96_96805


namespace percent_flowers_are_carnations_l96_96196

-- Define the conditions
def one_third_pink_are_roses (total_flower pink_flower pink_roses : ℕ) : Prop :=
  pink_roses = (1/3) * pink_flower

def three_fourths_red_are_carnations (total_flower red_flower red_carnations : ℕ) : Prop :=
  red_carnations = (3/4) * red_flower

def six_tenths_are_pink (total_flower pink_flower : ℕ) : Prop :=
  pink_flower = (6/10) * total_flower

-- Define the proof problem statement
theorem percent_flowers_are_carnations (total_flower pink_flower pink_roses red_flower red_carnations : ℕ) :
  one_third_pink_are_roses total_flower pink_flower pink_roses →
  three_fourths_red_are_carnations total_flower red_flower red_carnations →
  six_tenths_are_pink total_flower pink_flower →
  (red_flower = total_flower - pink_flower) →
  (pink_flower - pink_roses + red_carnations = (4/10) * total_flower) →
  ((pink_flower - pink_roses) + red_carnations) * 100 / total_flower = 40 := 
sorry

end percent_flowers_are_carnations_l96_96196


namespace find_p_l96_96486

theorem find_p (a : ℕ) (ha : a = 2030) : 
  let p := 2 * a + 1;
  let q := a * (a + 1);
  p = 4061 ∧ Nat.gcd p q = 1 := by
  sorry

end find_p_l96_96486


namespace flight_duration_l96_96753

theorem flight_duration (takeoff landing : Nat)
  (h m : Nat) (h_pos : 0 < m) (m_lt_60 : m < 60)
  (time_takeoff : takeoff = 9 * 60 + 27)
  (time_landing : landing = 11 * 60 + 56)
  (flight_duration : (landing - takeoff) = h * 60 + m) :
  h + m = 31 :=
sorry

end flight_duration_l96_96753


namespace acute_angle_of_rhombus_l96_96368

theorem acute_angle_of_rhombus (a α : ℝ) (V1 V2 : ℝ) (OA BD AN AB : ℝ) 
  (h_volumes : V1 / V2 = 1 / (2 * Real.sqrt 5)) 
  (h_V1 : V1 = (1 / 3) * Real.pi * (OA^2) * BD)
  (h_V2 : V2 = Real.pi * (AN^2) * AB)
  (h_OA : OA = a * Real.sin (α / 2))
  (h_BD : BD = 2 * a * Real.cos (α / 2))
  (h_AN : AN = a * Real.sin α)
  (h_AB : AB = a)
  : α = Real.arccos (1 / 9) :=
sorry

end acute_angle_of_rhombus_l96_96368


namespace square_distance_from_B_to_center_l96_96957

-- Defining the conditions
structure Circle (α : Type _) :=
(center : α × α)
(radius2 : ℝ)

structure Point (α : Type _) :=
(x : α)
(y : α)

def is_right_angle (a b c : Point ℝ) : Prop :=
(b.x - a.x) * (c.x - b.x) + (b.y - a.y) * (c.y - b.y) = 0

noncomputable def distance2 (p1 p2 : Point ℝ) : ℝ :=
(p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2

theorem square_distance_from_B_to_center :
  ∀ (c : Circle ℝ) (A B C : Point ℝ), 
    c.radius2 = 65 →
    distance2 A B = 49 →
    distance2 B C = 9 →
    is_right_angle A B C →
    distance2 B {x:=0, y:=0} = 80 := 
by
  intros c A B C h_radius h_AB h_BC h_right_angle
  sorry

end square_distance_from_B_to_center_l96_96957


namespace alphazia_lost_words_l96_96602

def alphazia_letters := 128
def forbidden_letters := 2
def total_forbidden_pairs := forbidden_letters * alphazia_letters

theorem alphazia_lost_words :
  let one_letter_lost := forbidden_letters
  let two_letter_lost := 2 * alphazia_letters
  one_letter_lost + two_letter_lost = 258 :=
by
  sorry

end alphazia_lost_words_l96_96602


namespace expand_remains_same_l96_96230

variable (m n : ℤ)

-- Define a function that represents expanding m and n by a factor of 3
def expand_by_factor_3 (m n : ℤ) : ℤ := 
  2 * (3 * m) / (3 * m - 3 * n)

-- Define the original fraction
def original_fraction (m n : ℤ) : ℤ :=
  2 * m / (m - n)

-- Theorem to prove that expanding m and n by a factor of 3 does not change the fraction
theorem expand_remains_same (m n : ℤ) : 
  expand_by_factor_3 m n = original_fraction m n := 
by sorry

end expand_remains_same_l96_96230


namespace length_of_24_l96_96408

def length_of_integer (k : ℕ) : ℕ :=
  k.factors.length

theorem length_of_24 : length_of_integer 24 = 4 :=
by
  sorry

end length_of_24_l96_96408


namespace comic_cost_is_4_l96_96105

-- Define initial amount of money Raul had.
def initial_money : ℕ := 87

-- Define number of comics bought by Raul.
def num_comics : ℕ := 8

-- Define the amount of money left after buying comics.
def money_left : ℕ := 55

-- Define the hypothesis condition about the money spent.
def total_spent : ℕ := initial_money - money_left

-- Define the main assertion that each comic cost $4.
def cost_per_comic (total_spent : ℕ) (num_comics : ℕ) : Prop :=
  total_spent / num_comics = 4

-- Main theorem statement
theorem comic_cost_is_4 : cost_per_comic total_spent num_comics :=
by
  -- Here we're skipping the proof for this exercise.
  sorry

end comic_cost_is_4_l96_96105


namespace triangle_perimeter_l96_96577

theorem triangle_perimeter
  (x : ℝ) 
  (h : x^2 - 6 * x + 8 = 0)
  (a b c : ℝ)
  (ha : a = 2)
  (hb : b = 4)
  (hc : c = x)
  (triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a) :
  a + b + c = 10 := 
sorry

end triangle_perimeter_l96_96577


namespace ellipse_focus_coordinates_l96_96520

theorem ellipse_focus_coordinates (a b c : ℝ) (x1 y1 x2 y2 : ℝ) 
  (major_axis_length : 2 * a = 20) 
  (focal_relationship : c^2 = a^2 - b^2)
  (focus1_location : x1 = 3 ∧ y1 = 4) 
  (focus_c_calculation : c = Real.sqrt (x1^2 + y1^2)) :
  (x2 = -3 ∧ y2 = -4) := by
  sorry

end ellipse_focus_coordinates_l96_96520


namespace domain_of_logarithmic_function_l96_96883

theorem domain_of_logarithmic_function :
  ∀ x : ℝ, 2 - x > 0 ↔ x < 2 := 
by
  intro x
  sorry

end domain_of_logarithmic_function_l96_96883


namespace rectangle_perimeter_is_104_l96_96359

noncomputable def perimeter_of_rectangle (b : ℝ) (h1 : b > 0) (h2 : 3 * b * b = 507) : ℝ :=
  2 * (3 * b) + 2 * b

theorem rectangle_perimeter_is_104 {b : ℝ} (h1 : b > 0) (h2 : 3 * b * b = 507) :
  perimeter_of_rectangle b h1 h2 = 104 :=
by
  sorry

end rectangle_perimeter_is_104_l96_96359


namespace greatest_number_of_balloons_l96_96718

-- Let p be the regular price of one balloon, and M be the total amount of money Orvin has
variable (p M : ℝ)

-- Initial condition: Orvin can buy 45 balloons at the regular price.
-- Thus, he has money M = 45 * p
def orvin_has_enough_money : Prop :=
  M = 45 * p

-- Special Sale condition: The first balloon costs p and the second balloon costs p/2,
-- so total cost for 2 balloons = 1.5 * p
def special_sale_condition : Prop :=
  ∀ pairs : ℝ, M / (1.5 * p) = pairs ∧ pairs * 2 = 60

-- Given the initial condition and the special sale condition, prove the greatest 
-- number of balloons Orvin could purchase is 60
theorem greatest_number_of_balloons (p : ℝ) (M : ℝ) (h1 : orvin_has_enough_money p M) (h2 : special_sale_condition p M) : 
∀ N : ℝ, N = 60 :=
sorry

end greatest_number_of_balloons_l96_96718


namespace multiple_of_michael_trophies_l96_96024

-- Conditions
def michael_current_trophies : ℕ := 30
def michael_trophies_increse : ℕ := 100
def total_trophies_in_three_years : ℕ := 430

-- Proof statement
theorem multiple_of_michael_trophies (x : ℕ) :
  (michael_current_trophies + michael_trophies_increse) + (michael_current_trophies * x) = total_trophies_in_three_years → x = 10 := 
by
  sorry

end multiple_of_michael_trophies_l96_96024


namespace maximum_n_l96_96712

noncomputable def a1 : ℝ := sorry -- define a1 solving a_5 equations
noncomputable def q : ℝ := sorry -- define q solving a_5 and a_6 + a_7 equations
noncomputable def sn (n : ℕ) : ℝ := (a1 * (1 - q^n)) / (1 - q)  -- S_n of geometric series with a1 and q
noncomputable def pin (n : ℕ) : ℝ := (a1 * (q^((1 + n) * n / 2 - (11 * n) / 2 + 19 / 2)))  -- Pi solely in terms of n, a1, and q

theorem maximum_n (n : ℕ) (h1 : (a1 : ℝ) > 0) (h2 : q > 0) (h3 : q ≠ 1)
(h4 : a1 * q^4 = 1 / 4) (h5 : a1 * q^5 + a1 * q^6 = 3 / 2) :
  ∃ n : ℕ, sn n > pin n ∧ ∀ m : ℕ, m > 13 → sn m ≤ pin m := sorry

end maximum_n_l96_96712


namespace jenna_less_than_bob_l96_96648

def bob_amount : ℕ := 60
def phil_amount : ℕ := (1 / 3) * bob_amount
def jenna_amount : ℕ := 2 * phil_amount

theorem jenna_less_than_bob : bob_amount - jenna_amount = 20 := by
  sorry

end jenna_less_than_bob_l96_96648


namespace min_reciprocal_sum_l96_96755

theorem min_reciprocal_sum (a b x y : ℝ) (h1 : 8 * x - y - 4 ≤ 0) (h2 : x + y + 1 ≥ 0) (h3 : y - 4 * x ≤ 0) 
    (ha : a > 0) (hb : b > 0) (hz : a * x + b * y = 2) : 
    1 / a + 1 / b = 9 / 2 := 
    sorry

end min_reciprocal_sum_l96_96755


namespace arithmetic_sequence_and_sum_properties_l96_96158

noncomputable def a_n (n : ℕ) : ℤ := 30 - 2 * n
noncomputable def S_n (n : ℕ) : ℤ := -n^2 + 29 * n

theorem arithmetic_sequence_and_sum_properties :
  (a_n 3 = 24 ∧ a_n 6 = 18) ∧
  (∀ n : ℕ, (S_n n = (n * (a_n 1 + a_n n)) / 2) ∧ ((a_n 3 = 24 ∧ a_n 6 = 18) → ∀ n : ℕ, a_n n = 30 - 2 * n)) ∧
  (S_n 14 = 210) :=
by 
  -- Proof omitted.
  sorry

end arithmetic_sequence_and_sum_properties_l96_96158


namespace min_value_of_reciprocals_l96_96646

theorem min_value_of_reciprocals (m n : ℝ) (h1 : m + n = 2) (h2 : m * n > 0) : 
  (1 / m) + (1 / n) = 2 :=
by
  -- the proof needs to be completed here.
  sorry

end min_value_of_reciprocals_l96_96646


namespace snail_kite_eats_35_snails_l96_96710

theorem snail_kite_eats_35_snails : 
  let day1 := 3
  let day2 := day1 + 2
  let day3 := day2 + 2
  let day4 := day3 + 2
  let day5 := day4 + 2
  day1 + day2 + day3 + day4 + day5 = 35 := 
by
  sorry

end snail_kite_eats_35_snails_l96_96710


namespace downstream_distance_80_l96_96172

-- Conditions
variables (Speed_boat Speed_stream Distance_upstream : ℝ)

-- Assign given values
def speed_boat := 36 -- kmph
def speed_stream := 12 -- kmph
def distance_upstream := 40 -- km

-- Effective speeds
def speed_downstream := speed_boat + speed_stream -- kmph
def speed_upstream := speed_boat - speed_stream -- kmph

-- Downstream distance
noncomputable def distance_downstream : ℝ := 80 -- km

-- Theorem
theorem downstream_distance_80 :
  speed_boat = 36 → speed_stream = 12 → distance_upstream = 40 →
  (distance_upstream / speed_upstream = distance_downstream / speed_downstream) :=
by
  sorry

end downstream_distance_80_l96_96172


namespace simplify_fraction_l96_96580

theorem simplify_fraction : (3 / 462 + 17 / 42) = 95 / 231 :=
by 
  sorry

end simplify_fraction_l96_96580


namespace smallest_even_divisible_by_20_and_60_l96_96422

theorem smallest_even_divisible_by_20_and_60 : ∃ x, (Even x) ∧ (x % 20 = 0) ∧ (x % 60 = 0) ∧ (∀ y, (Even y) ∧ (y % 20 = 0) ∧ (y % 60 = 0) → x ≤ y) → x = 60 :=
by
  sorry

end smallest_even_divisible_by_20_and_60_l96_96422


namespace investment_interests_l96_96380

theorem investment_interests (x y : ℝ) (h₁ : x + y = 24000)
  (h₂ : 0.045 * x + 0.06 * y = 0.05 * 24000) : (x = 16000) ∧ (y = 8000) :=
  by
  sorry

end investment_interests_l96_96380


namespace find_a_l96_96194

noncomputable def f (a : ℝ) (x : ℝ) := (1/2) * a * x^2 + Real.log x

theorem find_a (h_max : ∃ (x : Set.Icc (0 : ℝ) 1), f (-Real.exp 1) x = -1) : 
  ∀ a : ℝ, (∀ x : ℝ, 0 < x → x ≤ 1 → f a x ≤ -1) → a = -Real.exp 1 :=
sorry

end find_a_l96_96194


namespace number_of_people_after_10_years_l96_96440

def number_of_people_after_n_years (n : ℕ) : ℕ :=
  Nat.recOn n 30 (fun k a_k => 3 * a_k - 20)

theorem number_of_people_after_10_years :
  number_of_people_after_n_years 10 = 1180990 := by
  sorry

end number_of_people_after_10_years_l96_96440


namespace calc1_calc2_l96_96636

noncomputable def calculation1 := -4^2

theorem calc1 : calculation1 = -16 := by
  sorry

noncomputable def calculation2 := (-3) - (-6)

theorem calc2 : calculation2 = 3 := by
  sorry

end calc1_calc2_l96_96636


namespace total_envelopes_l96_96488

def total_stamps : ℕ := 52
def lighter_envelopes : ℕ := 6
def stamps_per_lighter_envelope : ℕ := 2
def stamps_per_heavier_envelope : ℕ := 5

theorem total_envelopes (total_stamps lighter_envelopes stamps_per_lighter_envelope stamps_per_heavier_envelope : ℕ) 
  (h : total_stamps = 52 ∧ lighter_envelopes = 6 ∧ stamps_per_lighter_envelope = 2 ∧ stamps_per_heavier_envelope = 5) : 
  lighter_envelopes + (total_stamps - (stamps_per_lighter_envelope * lighter_envelopes)) / stamps_per_heavier_envelope = 14 :=
by
  sorry

end total_envelopes_l96_96488


namespace range_of_a_l96_96163

theorem range_of_a
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : x + 2 * y + 4 = 4 * x * y)
  (h2 : ∀ a : ℝ, (x + 2 * y) * a ^ 2 + 2 * a + 2 * x * y - 34 ≥ 0) : 
  ∀ a : ℝ, a ≤ -3 ∨ a ≥ 5 / 2 :=
by
  sorry

end range_of_a_l96_96163


namespace suraj_average_after_17th_innings_l96_96713

theorem suraj_average_after_17th_innings (A : ℕ) :
  (16 * A + 92) / 17 = A + 4 -> A + 4 = 28 := 
by 
  sorry

end suraj_average_after_17th_innings_l96_96713


namespace sum_of_odd_powers_divisible_by_six_l96_96868

theorem sum_of_odd_powers_divisible_by_six (a1 a2 a3 a4 : ℤ)
    (h : a1^3 + a2^3 + a3^3 + a4^3 = 0) :
    ∀ k : ℕ, k % 2 = 1 → 6 ∣ (a1^k + a2^k + a3^k + a4^k) :=
by
  intros k hk
  sorry

end sum_of_odd_powers_divisible_by_six_l96_96868


namespace find_X_l96_96373

theorem find_X (X : ℝ) (h : 45 * 8 = 0.40 * X) : X = 900 :=
sorry

end find_X_l96_96373


namespace degrees_to_radians_216_l96_96412

theorem degrees_to_radians_216 : (216 / 180 : ℝ) * Real.pi = (6 / 5 : ℝ) * Real.pi := by
  sorry

end degrees_to_radians_216_l96_96412


namespace simplify_expression_l96_96211

noncomputable def x : ℝ := Real.sqrt 3 + 1

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  ((1 - (x / (x + 1))) / ((x^2 - 1) / (x^2 + 2*x + 1))) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_expression_l96_96211


namespace cos_alpha_in_second_quadrant_l96_96687

variable (α : Real) -- Define the variable α as a Real number (angle in radians)
variable (h1 : α > π / 2 ∧ α < π) -- Condition that α is in the second quadrant
variable (h2 : Real.sin α = 2 / 3) -- Condition that sin(α) = 2/3

theorem cos_alpha_in_second_quadrant (α : Real) (h1 : α > π / 2 ∧ α < π)
  (h2 : Real.sin α = 2 / 3) : Real.cos α = - Real.sqrt (1 - (2 / 3) ^ 2) :=
by
  sorry

end cos_alpha_in_second_quadrant_l96_96687


namespace florist_first_picking_l96_96772

theorem florist_first_picking (x : ℝ) (h1 : 37.0 + x + 19.0 = 72.0) : x = 16.0 :=
by
  sorry

end florist_first_picking_l96_96772


namespace eeshas_usual_time_l96_96825

/-- Eesha's usual time to reach her office from home is 60 minutes,
given that she started 30 minutes late and reached her office
50 minutes late while driving 25% slower than her usual speed. -/
theorem eeshas_usual_time (T T' : ℝ) (h1 : T' = T / 0.75) (h2 : T' = T + 20) : T = 60 := by
  sorry

end eeshas_usual_time_l96_96825


namespace sin_eq_cos_example_l96_96286

theorem sin_eq_cos_example 
  (n : ℤ) (h_range : -180 ≤ n ∧ n ≤ 180)
  (h_eq : Real.sin (n * Real.pi / 180) = Real.cos (682 * Real.pi / 180)) :
  n = 128 :=
sorry

end sin_eq_cos_example_l96_96286


namespace modulo_residue_l96_96598

theorem modulo_residue : 
  ∃ (x : ℤ), 0 ≤ x ∧ x < 31 ∧ (-1237 % 31) = x := 
  sorry

end modulo_residue_l96_96598


namespace accurate_bottle_weight_l96_96826

-- Define the options as constants
def OptionA : ℕ := 500 -- milligrams
def OptionB : ℕ := 500 * 1000 -- grams
def OptionC : ℕ := 500 * 1000 * 1000 -- kilograms
def OptionD : ℕ := 500 * 1000 * 1000 * 1000 -- tons

-- Define a threshold range for the weight of a standard bottle of mineral water in grams
def typicalBottleWeightMin : ℕ := 400 -- for example
def typicalBottleWeightMax : ℕ := 600 -- for example

-- Translate the question and conditions into a proof statement
theorem accurate_bottle_weight : OptionB = 500 * 1000 :=
by
  -- Normally, we would add the necessary steps here to prove the statement
  sorry

end accurate_bottle_weight_l96_96826


namespace chromium_percentage_l96_96751

theorem chromium_percentage (x : ℝ) : 
  (15 * x / 100 + 35 * 8 / 100 = 50 * 8.6 / 100) → 
  x = 10 := 
sorry

end chromium_percentage_l96_96751


namespace range_of_a_l96_96613

noncomputable def quadratic_inequality_holds (a : ℝ) : Prop :=
  ∀ (x : ℝ), a * x^2 - a * x - 1 < 0 

theorem range_of_a (a : ℝ) : quadratic_inequality_holds a ↔ -4 < a ∧ a ≤ 0 := 
sorry

end range_of_a_l96_96613


namespace sin_alpha_eq_sqrt5_over_3_l96_96476

theorem sin_alpha_eq_sqrt5_over_3 {α : ℝ} (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.sin α = (Real.sqrt 5) / 3 :=
sorry

end sin_alpha_eq_sqrt5_over_3_l96_96476


namespace original_number_l96_96116

theorem original_number (x : ℕ) : x * 16 = 3408 → x = 213 := by
  intro h
  sorry

end original_number_l96_96116
