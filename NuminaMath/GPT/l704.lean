import Mathlib

namespace NUMINAMATH_GPT_pond_contains_total_money_correct_l704_70488

def value_of_dime := 10
def value_of_quarter := 25
def value_of_nickel := 5
def value_of_penny := 1

def cindy_dimes := 5
def eric_quarters := 3
def garrick_nickels := 8
def ivy_pennies := 60

def total_money : ℕ := 
  cindy_dimes * value_of_dime + 
  eric_quarters * value_of_quarter + 
  garrick_nickels * value_of_nickel + 
  ivy_pennies * value_of_penny

theorem pond_contains_total_money_correct:
  total_money = 225 := by
  sorry

end NUMINAMATH_GPT_pond_contains_total_money_correct_l704_70488


namespace NUMINAMATH_GPT_largest_prime_divisor_25_sq_plus_72_sq_l704_70423

theorem largest_prime_divisor_25_sq_plus_72_sq : ∃ p : ℕ, Nat.Prime p ∧ p ∣ (25^2 + 72^2) ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ (25^2 + 72^2) → q ≤ p :=
sorry

end NUMINAMATH_GPT_largest_prime_divisor_25_sq_plus_72_sq_l704_70423


namespace NUMINAMATH_GPT_elder_twice_as_old_l704_70410

theorem elder_twice_as_old (Y E : ℕ) (hY : Y = 35) (hDiff : E - Y = 20) : ∃ (X : ℕ),  X = 15 ∧ E - X = 2 * (Y - X) := 
by
  sorry

end NUMINAMATH_GPT_elder_twice_as_old_l704_70410


namespace NUMINAMATH_GPT_total_ladders_climbed_in_inches_l704_70498

-- Define the conditions as hypotheses
def keaton_ladder_length := 30
def keaton_climbs := 20
def reece_ladder_difference := 4
def reece_climbs := 15
def feet_to_inches := 12

-- Define the lengths climbed by Keaton and Reece
def keaton_total_feet := keaton_ladder_length * keaton_climbs
def reece_ladder_length := keaton_ladder_length - reece_ladder_difference
def reece_total_feet := reece_ladder_length * reece_climbs

-- Calculate the total feet climbed and convert to inches
def total_feet := keaton_total_feet + reece_total_feet
def total_inches := total_feet * feet_to_inches

-- Prove the final result
theorem total_ladders_climbed_in_inches : total_inches = 11880 := by
  sorry

end NUMINAMATH_GPT_total_ladders_climbed_in_inches_l704_70498


namespace NUMINAMATH_GPT_original_divisor_in_terms_of_Y_l704_70458

variables (N D Y : ℤ)
variables (h1 : N = 45 * D + 13) (h2 : N = 6 * Y + 4)

theorem original_divisor_in_terms_of_Y (h1 : N = 45 * D + 13) (h2 : N = 6 * Y + 4) : 
  D = (2 * Y - 3) / 15 :=
sorry

end NUMINAMATH_GPT_original_divisor_in_terms_of_Y_l704_70458


namespace NUMINAMATH_GPT_find_value_of_y_l704_70494

noncomputable def angle_sum_triangle (A B C : ℝ) : Prop :=
A + B + C = 180

noncomputable def triangle_ABC : angle_sum_triangle 80 60 x := by
  sorry

noncomputable def triangle_CDE (x y : ℝ) : Prop :=
(x = 40) ∧ (90 + x + y = 180)

theorem find_value_of_y (x y : ℝ) 
  (h1 : angle_sum_triangle 80 60 x)
  (h2 : triangle_CDE x y) : 
  y = 50 := 
by
  sorry

end NUMINAMATH_GPT_find_value_of_y_l704_70494


namespace NUMINAMATH_GPT_original_price_l704_70402

theorem original_price (x : ℝ) (h1 : 0.95 * x * 1.40 = 1.33 * x) (h2 : 1.33 * x = 2 * x - 1352.06) : x = 2018 := sorry

end NUMINAMATH_GPT_original_price_l704_70402


namespace NUMINAMATH_GPT_gold_distribution_l704_70442

theorem gold_distribution :
  ∃ (d : ℚ), 
    (4 * (a1: ℚ) + 6 * d = 3) ∧ 
    (3 * (a1: ℚ) + 24 * d = 4) ∧
    d = 7 / 78 :=
by {
  sorry
}

end NUMINAMATH_GPT_gold_distribution_l704_70442


namespace NUMINAMATH_GPT_prime_square_mod_12_l704_70490

theorem prime_square_mod_12 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) : p^2 % 12 = 1 := 
by
  sorry

end NUMINAMATH_GPT_prime_square_mod_12_l704_70490


namespace NUMINAMATH_GPT_distance_between_A_and_B_is_90_l704_70430

variable (A B : Type)
variables (v_A v_B v'_A v'_B : ℝ)
variable (d : ℝ)

-- Conditions
axiom starts_simultaneously : True
axiom speed_ratio : v_A / v_B = 4 / 5
axiom A_speed_decrease : v'_A = 0.75 * v_A
axiom B_speed_increase : v'_B = 1.2 * v_B
axiom distance_when_B_reaches_A : ∃ k : ℝ, k = 30 -- Person A is 30 km away from location B

-- Goal
theorem distance_between_A_and_B_is_90 : d = 90 := by 
  sorry

end NUMINAMATH_GPT_distance_between_A_and_B_is_90_l704_70430


namespace NUMINAMATH_GPT_value_of_x_plus_2y_l704_70493

theorem value_of_x_plus_2y 
  (x y : ℝ) 
  (h : (x + 5)^2 = -(|y - 2|)) : 
  x + 2 * y = -1 :=
sorry

end NUMINAMATH_GPT_value_of_x_plus_2y_l704_70493


namespace NUMINAMATH_GPT_salt_percentage_l704_70454

theorem salt_percentage :
  ∀ (salt water : ℝ), salt = 10 → water = 90 → 
  100 * (salt / (salt + water)) = 10 :=
by
  intros salt water h_salt h_water
  sorry

end NUMINAMATH_GPT_salt_percentage_l704_70454


namespace NUMINAMATH_GPT_luncheon_cost_l704_70440

variables (s c p : ℝ)

def eq1 := 5 * s + 8 * c + 2 * p = 5.10
def eq2 := 6 * s + 11 * c + 2 * p = 6.45

theorem luncheon_cost (h₁ : 5 * s + 8 * c + 2 * p = 5.10) (h₂ : 6 * s + 11 * c + 2 * p = 6.45) : 
  s + c + p = 1.35 :=
  sorry

end NUMINAMATH_GPT_luncheon_cost_l704_70440


namespace NUMINAMATH_GPT_total_sand_correct_l704_70427

-- Define the conditions as variables and equations:
variables (x : ℕ) -- original days scheduled to complete
variables (total_sand : ℕ) -- total amount of sand in tons

-- Define the conditions in the problem:
def original_daily_amount := 15  -- tons per day as scheduled
def actual_daily_amount := 20  -- tons per day in reality
def days_ahead := 3  -- days finished ahead of schedule

-- Equation representing the planned and actual transportation:
def planned_sand := original_daily_amount * x
def actual_sand := actual_daily_amount * (x - days_ahead)

-- The goal is to prove:
theorem total_sand_correct : planned_sand = actual_sand → total_sand = 180 :=
by
  sorry

end NUMINAMATH_GPT_total_sand_correct_l704_70427


namespace NUMINAMATH_GPT_emily_num_dresses_l704_70450

theorem emily_num_dresses (M : ℕ) (D : ℕ) (E : ℕ) 
  (h1 : D = M + 12) 
  (h2 : M = E / 2) 
  (h3 : M + D + E = 44) : 
  E = 16 := 
by 
  sorry

end NUMINAMATH_GPT_emily_num_dresses_l704_70450


namespace NUMINAMATH_GPT_find_a_2016_l704_70432

-- Define the sequence a_n and its sum S_n
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Given conditions
axiom S_n_eq : ∀ n : ℕ, S n + (1 + (2 / n)) * a n = 4
axiom a_1_eq : a 1 = 1
axiom a_rec : ∀ n : ℕ, n ≥ 2 → a n = (n / (2 * (n - 1))) * a (n - 1)

-- The theorem to prove
theorem find_a_2016 : a 2016 = 2016 / 2^2015 := by
  sorry

end NUMINAMATH_GPT_find_a_2016_l704_70432


namespace NUMINAMATH_GPT_find_third_number_l704_70481

-- Definitions
def A : ℕ := 600
def B : ℕ := 840
def LCM : ℕ := 50400
def HCF : ℕ := 60

-- Theorem to be proven
theorem find_third_number (C : ℕ) (h_lcm : Nat.lcm (Nat.lcm A B) C = LCM) (h_hcf : Nat.gcd (Nat.gcd A B) C = HCF) : C = 6 :=
by -- proof
  sorry

end NUMINAMATH_GPT_find_third_number_l704_70481


namespace NUMINAMATH_GPT_trig_identity_tan_solutions_l704_70412

open Real

theorem trig_identity_tan_solutions :
  ∃ α β : ℝ, (tan α) * (tan β) = -3 ∧ (tan α) + (tan β) = 3 ∧
  abs (sin (α + β) ^ 2 - 3 * sin (α + β) * cos (α + β) - 3 * cos (α + β) ^ 2) = 3 :=
by
  have: ∀ x : ℝ, x^2 - 3*x - 3 = 0 → x = (3 + sqrt 21) / 2 ∨ x = (3 - sqrt 21) / 2 := sorry
  sorry

end NUMINAMATH_GPT_trig_identity_tan_solutions_l704_70412


namespace NUMINAMATH_GPT_speed_with_current_l704_70497

-- Define the constants
def speed_of_current : ℝ := 2.5
def speed_against_current : ℝ := 20

-- Define the man's speed in still water
axiom speed_in_still_water : ℝ
axiom speed_against_current_eq : speed_in_still_water - speed_of_current = speed_against_current

-- The statement we need to prove
theorem speed_with_current : speed_in_still_water + speed_of_current = 25 := sorry

end NUMINAMATH_GPT_speed_with_current_l704_70497


namespace NUMINAMATH_GPT_middle_number_of_consecutive_squares_l704_70478

theorem middle_number_of_consecutive_squares (x : ℕ ) (h : x^2 + (x+1)^2 + (x+2)^2 = 2030) : x + 1 = 26 :=
sorry

end NUMINAMATH_GPT_middle_number_of_consecutive_squares_l704_70478


namespace NUMINAMATH_GPT_tangent_line_equation_at_point_l704_70408

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1) - x

theorem tangent_line_equation_at_point :
  ∃ a b c : ℝ, (∀ x y : ℝ, a * x + b * y + c = 0 ↔ (x = 1 → y = -1 → f x = y)) ∧ (a * 1 + b * (-1) + c = 0) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_equation_at_point_l704_70408


namespace NUMINAMATH_GPT_combined_area_rectangle_triangle_l704_70419

/-- 
  Given a rectangle ABCD with vertices A = (10, -30), 
  B = (2010, 170), D = (12, -50), and a right triangle
  ADE with vertex E = (12, -30), prove that the combined
  area of the rectangle and the triangle is 
  40400 + 20√101.
-/
theorem combined_area_rectangle_triangle :
  let A := (10, -30)
  let B := (2010, 170)
  let D := (12, -50)
  let E := (12, -30)
  let length_AB := Real.sqrt ((2010 - 10)^2 + (170 + 30)^2)
  let length_AD := Real.sqrt ((12 - 10)^2 + (-50 + 30)^2)
  let area_rectangle := length_AB * length_AD
  let length_DE := Real.sqrt ((12 - 12)^2 + (-50 + 30)^2)
  let area_triangle := 1/2 * length_DE * length_AD
  area_rectangle + area_triangle = 40400 + 20 * Real.sqrt 101 :=
by
  sorry

end NUMINAMATH_GPT_combined_area_rectangle_triangle_l704_70419


namespace NUMINAMATH_GPT_third_circle_radius_l704_70468

theorem third_circle_radius (r1 r2 d : ℝ) (τ : ℝ) (h1: r1 = 1) (h2: r2 = 9) (h3: d = 17) : 
  τ = 225 / 64 :=
by
  sorry

end NUMINAMATH_GPT_third_circle_radius_l704_70468


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l704_70492

-- Definitions of the sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

-- Statement to prove the intersection of sets A and B is {3}
theorem intersection_of_A_and_B : A ∩ B = {3} :=
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l704_70492


namespace NUMINAMATH_GPT_prime_sum_divisors_l704_70409

theorem prime_sum_divisors (p : ℕ) (s : ℕ) : 
  (2 ≤ s ∧ s ≤ 10) → 
  (p = 2^s - 1) → 
  (p = 3 ∨ p = 7 ∨ p = 31 ∨ p = 127) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_prime_sum_divisors_l704_70409


namespace NUMINAMATH_GPT_James_baked_muffins_l704_70416

theorem James_baked_muffins (arthur_muffins : Nat) (multiplier : Nat) (james_muffins : Nat) : 
  arthur_muffins = 115 → 
  multiplier = 12 → 
  james_muffins = arthur_muffins * multiplier → 
  james_muffins = 1380 :=
by
  intros haf ham hmul
  rw [haf, ham] at hmul
  simp at hmul
  exact hmul

end NUMINAMATH_GPT_James_baked_muffins_l704_70416


namespace NUMINAMATH_GPT_Sally_lost_20_Pokemon_cards_l704_70469

theorem Sally_lost_20_Pokemon_cards (original_cards : ℕ) (received_cards : ℕ) (final_cards : ℕ) (lost_cards : ℕ) 
  (h1 : original_cards = 27) 
  (h2 : received_cards = 41) 
  (h3 : final_cards = 48) 
  (h4 : original_cards + received_cards - lost_cards = final_cards) : 
  lost_cards = 20 := 
sorry

end NUMINAMATH_GPT_Sally_lost_20_Pokemon_cards_l704_70469


namespace NUMINAMATH_GPT_parabola_directrix_l704_70441

theorem parabola_directrix (a : ℝ) (h : -1 / (4 * a) = 2) : a = -1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_l704_70441


namespace NUMINAMATH_GPT_gcd_of_4410_and_10800_l704_70473

theorem gcd_of_4410_and_10800 : Nat.gcd 4410 10800 = 90 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_of_4410_and_10800_l704_70473


namespace NUMINAMATH_GPT_sum_remainders_l704_70443

theorem sum_remainders :
  ∀ (a b c d : ℕ),
  a % 53 = 31 →
  b % 53 = 44 →
  c % 53 = 6 →
  d % 53 = 2 →
  (a + b + c + d) % 53 = 30 :=
by
  intros a b c d ha hb hc hd
  sorry

end NUMINAMATH_GPT_sum_remainders_l704_70443


namespace NUMINAMATH_GPT_complete_the_square_b_l704_70405

theorem complete_the_square_b (x : ℝ) : (x ^ 2 - 6 * x + 7 = 0) → ∃ a b : ℝ, (x + a) ^ 2 = b ∧ b = 2 :=
by
sorry

end NUMINAMATH_GPT_complete_the_square_b_l704_70405


namespace NUMINAMATH_GPT_selection_ways_l704_70437

-- Define the problem parameters
def male_students : ℕ := 4
def female_students : ℕ := 3
def total_selected : ℕ := 3

-- Define the binomial coefficient function for combinatorial calculations
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define conditions
def both_genders_must_be_represented : Prop :=
  total_selected = 3 ∧ male_students >= 1 ∧ female_students >= 1

-- Problem statement: proof that the total ways to select 3 students is 30
theorem selection_ways : both_genders_must_be_represented → 
  (binomial male_students 2 * binomial female_students 1 +
   binomial male_students 1 * binomial female_students 2) = 30 :=
by
  sorry

end NUMINAMATH_GPT_selection_ways_l704_70437


namespace NUMINAMATH_GPT_jason_flames_per_minute_l704_70466

theorem jason_flames_per_minute :
  (∀ (t : ℕ), t % 15 = 0 -> (5 * (t / 15) = 20)) :=
sorry

end NUMINAMATH_GPT_jason_flames_per_minute_l704_70466


namespace NUMINAMATH_GPT_math_expression_equivalent_l704_70449

theorem math_expression_equivalent :
  ((π - 1)^0 + 4 * (Real.sin (Real.pi / 4)) - Real.sqrt 8 + abs (-3) = 4) :=
by
  sorry

end NUMINAMATH_GPT_math_expression_equivalent_l704_70449


namespace NUMINAMATH_GPT_notepad_days_last_l704_70463

def fold_paper (n : Nat) : Nat := 2 ^ n

def lettersize_paper_pieces : Nat := 5
def folds : Nat := 3
def notes_per_day : Nat := 10

def smaller_note_papers_per_piece : Nat := fold_paper folds
def total_smaller_note_papers : Nat := lettersize_paper_pieces * smaller_note_papers_per_piece
def total_days : Nat := total_smaller_note_papers / notes_per_day

theorem notepad_days_last : total_days = 4 := by
  sorry

end NUMINAMATH_GPT_notepad_days_last_l704_70463


namespace NUMINAMATH_GPT_total_apples_l704_70483

def packs : ℕ := 2
def apples_per_pack : ℕ := 4

theorem total_apples : packs * apples_per_pack = 8 := by
  sorry

end NUMINAMATH_GPT_total_apples_l704_70483


namespace NUMINAMATH_GPT_vasya_correct_l704_70496

theorem vasya_correct (x : ℝ) (h : x^2 + x + 1 = 0) : 
  x^2000 + x^1999 + x^1998 + 1000*x^1000 + 1000*x^999 + 1000*x^998 + 2000*x^3 + 2000*x^2 + 2000*x + 3000 = 3000 :=
by 
  sorry

end NUMINAMATH_GPT_vasya_correct_l704_70496


namespace NUMINAMATH_GPT_initial_total_cards_l704_70479

theorem initial_total_cards (x y : ℕ) (h1 : x / (x + y) = 1 / 3) (h2 : x / (x + y + 4) = 1 / 4) : x + y = 12 := 
sorry

end NUMINAMATH_GPT_initial_total_cards_l704_70479


namespace NUMINAMATH_GPT_inequality_holds_l704_70447

variable (a b c : ℝ)

theorem inequality_holds (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + b*c) / (a * (b + c)) + 
  (b^2 + c*a) / (b * (c + a)) + 
  (c^2 + a*b) / (c * (a + b)) ≥ 3 :=
sorry

end NUMINAMATH_GPT_inequality_holds_l704_70447


namespace NUMINAMATH_GPT_max_notebooks_15_dollars_l704_70495

noncomputable def max_notebooks (money : ℕ) : ℕ :=
  let cost_individual   := 2
  let cost_pack_4       := 6
  let cost_pack_7       := 9
  let notebooks_budget  := 15
  if money >= 9 then 
    7 + max_notebooks (money - 9)
  else if money >= 6 then 
    4 + max_notebooks (money - 6)
  else 
    money / 2

theorem max_notebooks_15_dollars : max_notebooks 15 = 11 :=
by
  sorry

end NUMINAMATH_GPT_max_notebooks_15_dollars_l704_70495


namespace NUMINAMATH_GPT_correct_statements_count_l704_70491

theorem correct_statements_count (x : ℝ) :
  let inverse := (x > 0) → (x^2 > 0)
  let converse := (x^2 ≤ 0) → (x ≤ 0)
  let contrapositive := (x ≤ 0) → (x^2 ≤ 0)
  (∃ p : Prop, p = inverse ∨ p = converse ∧ p) ↔ 
  ¬ contrapositive →
  2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_count_l704_70491


namespace NUMINAMATH_GPT_fred_spent_18_42_l704_70435

variable (football_price : ℝ) (pokemon_price : ℝ) (baseball_price : ℝ)
variable (football_packs : ℕ) (pokemon_packs : ℕ) (baseball_decks : ℕ)

def total_cost (football_price : ℝ) (football_packs : ℕ) (pokemon_price : ℝ) (pokemon_packs : ℕ) (baseball_price : ℝ) (baseball_decks : ℕ) : ℝ :=
  football_packs * football_price + pokemon_packs * pokemon_price + baseball_decks * baseball_price

theorem fred_spent_18_42 :
  total_cost 2.73 2 4.01 1 8.95 1 = 18.42 :=
by
  sorry

end NUMINAMATH_GPT_fred_spent_18_42_l704_70435


namespace NUMINAMATH_GPT_triangular_angles_l704_70445

noncomputable def measure_of_B (A : ℝ) : ℝ :=
  Real.arcsin (Real.sqrt ((1 + Real.sin (2 * A)) / 3))

noncomputable def length_of_c (A : ℝ) : ℝ := 
  Real.sqrt (22 - 6 * Real.sqrt 13 * Real.cos (measure_of_B A))

noncomputable def area_of_triangle_ABC (A : ℝ) : ℝ := 
  (3 * Real.sqrt 13 / 2) * Real.sqrt ((1 + Real.sin (2 * A)) / 3)

theorem triangular_angles 
  (a b c : ℝ) (b_pos : b = Real.sqrt 13) (a_pos : a = 3) (h : b * Real.cos c = (2 * a - c) * Real.cos (measure_of_B c)) :
  c = length_of_c c ∧
  (3 * Real.sqrt 13 / 2) * Real.sqrt ((1 + Real.sin (2 * c)) / 3) = area_of_triangle_ABC c :=
by
  sorry

end NUMINAMATH_GPT_triangular_angles_l704_70445


namespace NUMINAMATH_GPT_sum_inverse_one_minus_roots_eq_half_l704_70487

noncomputable def cubic_eq_roots (x : ℝ) : ℝ := 10 * x^3 - 25 * x^2 + 8 * x - 1

theorem sum_inverse_one_minus_roots_eq_half
  {p q s : ℝ} (hpqseq : cubic_eq_roots p = 0 ∧ cubic_eq_roots q = 0 ∧ cubic_eq_roots s = 0)
  (hpospq : 0 < p ∧ 0 < q ∧ 0 < s) (hlespq : p < 1 ∧ q < 1 ∧ s < 1) :
  (1 / (1 - p)) + (1 / (1 - q)) + (1 / (1 - s)) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_sum_inverse_one_minus_roots_eq_half_l704_70487


namespace NUMINAMATH_GPT_profit_percentage_l704_70456

theorem profit_percentage (CP SP : ℝ) (h1 : CP = 500) (h2 : SP = 650) : 
  (SP - CP) / CP * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_l704_70456


namespace NUMINAMATH_GPT_harry_worked_total_hours_l704_70476

theorem harry_worked_total_hours (x : ℝ) (H : ℝ) (H_total : ℝ) :
  (24 * x + 1.5 * x * H = 42 * x) → (H_total = 24 + H) → H_total = 36 :=
by
sorry

end NUMINAMATH_GPT_harry_worked_total_hours_l704_70476


namespace NUMINAMATH_GPT_find_a_l704_70480

theorem find_a (a : ℝ) (i : ℂ) (h : i * i = -1) (h1 : (1 + a * i) * i = -3 + i) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l704_70480


namespace NUMINAMATH_GPT_rhombus_diagonal_l704_70420

theorem rhombus_diagonal (d1 d2 area : ℝ) (h1 : d1 = 20) (h2 : area = 160) (h3 : area = (d1 * d2) / 2) :
  d2 = 16 :=
by
  rw [h1, h2] at h3
  linarith

end NUMINAMATH_GPT_rhombus_diagonal_l704_70420


namespace NUMINAMATH_GPT_sin_alpha_eq_three_fifths_l704_70461

theorem sin_alpha_eq_three_fifths (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.tan α = -3 / 4) 
  (h3 : Real.sin α > 0) 
  (h4 : Real.cos α < 0) 
  (h5 : Real.sin α ^ 2 + Real.cos α ^ 2 = 1) : 
  Real.sin α = 3 / 5 := 
sorry

end NUMINAMATH_GPT_sin_alpha_eq_three_fifths_l704_70461


namespace NUMINAMATH_GPT_bounded_poly_constant_l704_70459

theorem bounded_poly_constant (P : Polynomial ℤ) (B : ℕ) (h_bounded : ∀ x : ℤ, abs (P.eval x) ≤ B) : 
  P.degree = 0 :=
sorry

end NUMINAMATH_GPT_bounded_poly_constant_l704_70459


namespace NUMINAMATH_GPT_Rhett_rent_expense_l704_70421

-- Define the problem statement using given conditions
theorem Rhett_rent_expense
  (late_payments : ℕ := 2)
  (no_late_fees : Bool := true)
  (fraction_of_salary : ℝ := 3 / 5)
  (monthly_salary : ℝ := 5000)
  (tax_rate : ℝ := 0.1) :
  let salary_after_taxes := monthly_salary * (1 - tax_rate)
  let total_late_rent := fraction_of_salary * salary_after_taxes
  let monthly_rent_expense := total_late_rent / late_payments
  monthly_rent_expense = 1350 := by
  sorry

end NUMINAMATH_GPT_Rhett_rent_expense_l704_70421


namespace NUMINAMATH_GPT_correct_operation_among_given_ones_l704_70431

theorem correct_operation_among_given_ones
  (a : ℝ) :
  (a^2)^3 = a^6 :=
by {
  sorry
}

-- Auxiliary lemmas if needed (based on conditions):
lemma mul_powers_add_exponents (a : ℝ) (m n : ℕ) : a^m * a^n = a^(m + n) := by sorry

lemma power_of_a_power (a : ℝ) (m n : ℕ) : (a^m)^n = a^(m * n) := by sorry

lemma div_powers_subtract_exponents (a : ℝ) (m n : ℕ) : a^m / a^n = a^(m - n) := by sorry

lemma square_of_product (x y : ℝ) : (x * y)^2 = x^2 * y^2 := by sorry

end NUMINAMATH_GPT_correct_operation_among_given_ones_l704_70431


namespace NUMINAMATH_GPT_storks_more_than_birds_l704_70438

def initial_birds := 2
def additional_birds := 3
def total_birds := initial_birds + additional_birds
def storks := 6
def difference := storks - total_birds

theorem storks_more_than_birds : difference = 1 :=
by
  sorry

end NUMINAMATH_GPT_storks_more_than_birds_l704_70438


namespace NUMINAMATH_GPT_change_in_f_when_x_increased_by_2_change_in_f_when_x_decreased_by_2_l704_70446

-- Given f(x) = x^2 - 5x
def f (x : ℝ) : ℝ := x^2 - 5 * x

-- Prove the change in f(x) when x is increased by 2 is 4x - 6
theorem change_in_f_when_x_increased_by_2 (x : ℝ) : f (x + 2) - f x = 4 * x - 6 := by
  sorry

-- Prove the change in f(x) when x is decreased by 2 is -4x + 14
theorem change_in_f_when_x_decreased_by_2 (x : ℝ) : f (x - 2) - f x = -4 * x + 14 := by
  sorry

end NUMINAMATH_GPT_change_in_f_when_x_increased_by_2_change_in_f_when_x_decreased_by_2_l704_70446


namespace NUMINAMATH_GPT_problem_statement_l704_70413

noncomputable def f (x : ℝ) : ℝ := (1/3)^x - x^2

theorem problem_statement (x0 x1 x2 m : ℝ) (h0 : f x0 = m) (h1 : 0 < x1) (h2 : x1 < x0) (h3 : x0 < x2) :
    f x1 > m ∧ f x2 < m :=
sorry

end NUMINAMATH_GPT_problem_statement_l704_70413


namespace NUMINAMATH_GPT_candles_ratio_l704_70485

-- Conditions
def kalani_bedroom_candles : ℕ := 20
def donovan_candles : ℕ := 20
def total_candles_house : ℕ := 50

-- Definitions for the number of candles in the living room and the ratio
def living_room_candles : ℕ := total_candles_house - kalani_bedroom_candles - donovan_candles
def ratio_of_candles : ℚ := kalani_bedroom_candles / living_room_candles

theorem candles_ratio : ratio_of_candles = 2 :=
by
  sorry

end NUMINAMATH_GPT_candles_ratio_l704_70485


namespace NUMINAMATH_GPT_illuminated_cube_surface_area_l704_70455

noncomputable def edge_length : ℝ := Real.sqrt (2 + Real.sqrt 3)
noncomputable def radius : ℝ := Real.sqrt 2
noncomputable def illuminated_area (a ρ : ℝ) : ℝ := Real.sqrt 3 * (Real.pi + 3)

theorem illuminated_cube_surface_area :
  illuminated_area edge_length radius = Real.sqrt 3 * (Real.pi + 3) := sorry

end NUMINAMATH_GPT_illuminated_cube_surface_area_l704_70455


namespace NUMINAMATH_GPT_boys_and_girls_l704_70429

theorem boys_and_girls (B G : ℕ) (h1 : B + G = 30)
  (h2 : ∀ (i j : ℕ), i < B → j < B → i ≠ j → ∃ k, k < G ∧ ∀ l < B, l ≠ i → k ≠ l)
  (h3 : ∀ (i j : ℕ), i < G → j < G → i ≠ j → ∃ k, k < B ∧ ∀ l < G, l ≠ i → k ≠ l) :
  B = 15 ∧ G = 15 :=
by
  have hB : B ≤ G := sorry
  have hG : G ≤ B := sorry
  exact ⟨by linarith, by linarith⟩

end NUMINAMATH_GPT_boys_and_girls_l704_70429


namespace NUMINAMATH_GPT_segment_EC_length_l704_70486

noncomputable def length_of_segment_EC (a b c : ℕ) (angle_A_deg BC : ℝ) (BD_perp_AC CE_perp_AB : Prop) (angle_DBC_eq_3_angle_ECB : Prop) : ℝ :=
  a * (Real.sqrt b + Real.sqrt c)

theorem segment_EC_length
  (a b c : ℕ)
  (angle_A_deg BC : ℝ)
  (BD_perp_AC CE_perp_AB : Prop)
  (angle_DBC_eq_3_angle_ECB : Prop)
  (h1 : angle_A_deg = 45)
  (h2 : BC = 10)
  (h3 : BD_perp_AC)
  (h4 : CE_perp_AB)
  (h5 : angle_DBC_eq_3_angle_ECB)
  (h6 : length_of_segment_EC a b c angle_A_deg BC BD_perp_AC CE_perp_AB angle_DBC_eq_3_angle_ECB = 5 * (Real.sqrt 3 + Real.sqrt 1)) :
  a + b + c = 9 :=
  by
    sorry

end NUMINAMATH_GPT_segment_EC_length_l704_70486


namespace NUMINAMATH_GPT_symmetric_points_origin_l704_70482

theorem symmetric_points_origin (a b : ℝ) (h1 : 1 = -b) (h2 : a = 2) : a + b = 1 := by
  sorry

end NUMINAMATH_GPT_symmetric_points_origin_l704_70482


namespace NUMINAMATH_GPT_find_a_l704_70451

open Set
open Real

def A : Set ℝ := {-1, 1}
def B (a : ℝ) : Set ℝ := {x | a * x ^ 2 = 1}

theorem find_a (a : ℝ) (h : (A ∩ (B a)) = (B a)) : a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_l704_70451


namespace NUMINAMATH_GPT_total_stars_l704_70470

theorem total_stars (g s : ℕ) (hg : g = 10^11) (hs : s = 10^11) : g * s = 10^22 :=
by
  rw [hg, hs]
  sorry

end NUMINAMATH_GPT_total_stars_l704_70470


namespace NUMINAMATH_GPT_problem1_problem2_l704_70415

-- Problem 1: Sequence "Seven six five four three two one" is a descending order
theorem problem1 : ∃ term: String, term = "Descending Order" ∧ "Seven six five four three two one" = "Descending Order" := sorry

-- Problem 2: Describing a computing tool that knows 0 and 1 and can calculate large numbers (computer)
theorem problem2 : ∃ tool: String, tool = "Computer" ∧ "I only know 0 and 1, can calculate millions and billions, available in both software and hardware" = "Computer" := sorry

end NUMINAMATH_GPT_problem1_problem2_l704_70415


namespace NUMINAMATH_GPT_abcd_value_l704_70414

noncomputable def abcd_eval (a b c d : ℂ) : ℂ := a * b * c * d

theorem abcd_value (a b c d : ℂ) 
  (h1 : a + b + c + d = 5)
  (h2 : (5 - a)^4 + (5 - b)^4 + (5 - c)^4 + (5 - d)^4 = 125)
  (h3 : (a + b)^4 + (b + c)^4 + (c + d)^4 + (d + a)^4 + (a + c)^4 + (b + d)^4 = 1205)
  (h4 : a^4 + b^4 + c^4 + d^4 = 25) : 
  abcd_eval a b c d = 70 := 
sorry

end NUMINAMATH_GPT_abcd_value_l704_70414


namespace NUMINAMATH_GPT_rectangle_dimensions_l704_70472

theorem rectangle_dimensions (x : ℝ) (h : 3 * x * x = 8 * x) : (x = 8 / 3 ∧ 3 * x = 8) :=
by {
  sorry
}

end NUMINAMATH_GPT_rectangle_dimensions_l704_70472


namespace NUMINAMATH_GPT_log_bounds_l704_70400

-- Definitions and assumptions
def tenCubed : Nat := 1000
def tenFourth : Nat := 10000
def twoNine : Nat := 512
def twoFourteen : Nat := 16384

-- Statement that encapsulates the proof problem
theorem log_bounds (h1 : 10^3 = tenCubed) 
                   (h2 : 10^4 = tenFourth) 
                   (h3 : 2^9 = twoNine) 
                   (h4 : 2^14 = twoFourteen) : 
  (2 / 7 : ℝ) < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < (1 / 3 : ℝ) :=
sorry

end NUMINAMATH_GPT_log_bounds_l704_70400


namespace NUMINAMATH_GPT_fruit_seller_price_l704_70448

theorem fruit_seller_price (CP SP : ℝ) (h1 : SP = 0.90 * CP) (h2 : 1.10 * CP = 13.444444444444445) : 
  SP = 11 :=
sorry

end NUMINAMATH_GPT_fruit_seller_price_l704_70448


namespace NUMINAMATH_GPT_competition_end_time_l704_70422

def time := ℕ × ℕ -- Representing time as a pair of hours and minutes

def start_time : time := (15, 15) -- 3:15 PM is represented as 15:15 in 24-hour format
def duration := 1825 -- Duration in minutes
def end_time : time := (21, 40) -- 9:40 PM is represented as 21:40 in 24-hour format

def add_minutes (t : time) (m : ℕ) : time :=
  let (h, min) := t
  let total_minutes := h * 60 + min + m
  (total_minutes / 60 % 24, total_minutes % 60)

theorem competition_end_time :
  add_minutes start_time duration = end_time :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_competition_end_time_l704_70422


namespace NUMINAMATH_GPT_elevator_translation_l704_70425

-- Definitions based on conditions
def turning_of_steering_wheel : Prop := False
def rotation_of_bicycle_wheels : Prop := False
def motion_of_pendulum : Prop := False
def movement_of_elevator : Prop := True

-- Theorem statement
theorem elevator_translation :
  movement_of_elevator := by
  exact True.intro

end NUMINAMATH_GPT_elevator_translation_l704_70425


namespace NUMINAMATH_GPT_adult_dog_cost_is_100_l704_70460

-- Define the costs for cats, puppies, and dogs.
def cat_cost : ℕ := 50
def puppy_cost : ℕ := 150

-- Define the number of each type of animal.
def number_of_cats : ℕ := 2
def number_of_adult_dogs : ℕ := 3
def number_of_puppies : ℕ := 2

-- The total cost
def total_cost : ℕ := 700

-- Define what needs to be proven: the cost of getting each adult dog ready for adoption.
theorem adult_dog_cost_is_100 (D : ℕ) (h : number_of_cats * cat_cost + number_of_adult_dogs * D + number_of_puppies * puppy_cost = total_cost) : D = 100 :=
by 
  sorry

end NUMINAMATH_GPT_adult_dog_cost_is_100_l704_70460


namespace NUMINAMATH_GPT_club_additional_members_l704_70477

theorem club_additional_members (current_members : ℕ) (additional_members : ℕ) 
  (h1 : current_members = 10) 
  (h2 : additional_members = 5 + 2 * current_members - current_members) : 
  additional_members = 15 :=
by 
  rw [h1] at h2
  norm_num at h2
  exact h2

end NUMINAMATH_GPT_club_additional_members_l704_70477


namespace NUMINAMATH_GPT_range_of_f_l704_70453

def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem range_of_f : Set.Icc 0 3 → (Set.Ico 1 5) :=
by
  sorry
  -- Here the proof steps would go, which are omitted based on your guidelines.

end NUMINAMATH_GPT_range_of_f_l704_70453


namespace NUMINAMATH_GPT_part_a_l704_70457

def is_tricubic (k : ℕ) : Prop :=
  ∃ a b c : ℕ, k = a^3 + b^3 + c^3

theorem part_a : ∃ (n : ℕ), is_tricubic n ∧ ¬ is_tricubic (n + 2) ∧ ¬ is_tricubic (n + 28) :=
by 
  let n := 3 * (3*1+1)^3
  exists n
  sorry

end NUMINAMATH_GPT_part_a_l704_70457


namespace NUMINAMATH_GPT_area_of_fourth_rectangle_l704_70433

variable (x y z w : ℝ)
variable (Area_EFGH Area_EIKJ Area_KLMN Perimeter : ℝ)

def conditions :=
  (Area_EFGH = x * y ∧ Area_EFGH = 20 ∧
   Area_EIKJ = x * w ∧ Area_EIKJ = 25 ∧
   Area_KLMN = z * w ∧ Area_KLMN = 15 ∧
   Perimeter = 2 * (x + z + y + w) ∧ Perimeter = 40)

theorem area_of_fourth_rectangle (h : conditions x y z w Area_EFGH Area_EIKJ Area_KLMN Perimeter) :
  (y * w = 340) :=
by
  sorry

end NUMINAMATH_GPT_area_of_fourth_rectangle_l704_70433


namespace NUMINAMATH_GPT_total_pennies_thrown_l704_70499

theorem total_pennies_thrown (R G X M T : ℝ) (hR : R = 1500)
  (hG : G = (2 / 3) * R) (hX : X = (3 / 4) * G) 
  (hM : M = 3.5 * X) (hT : T = (4 / 5) * M) : 
  R + G + X + M + T = 7975 :=
by
  sorry

end NUMINAMATH_GPT_total_pennies_thrown_l704_70499


namespace NUMINAMATH_GPT_percentage_discount_l704_70475

-- Define the given conditions
def equal_contribution (total: ℕ) (num_people: ℕ) := total / num_people

def original_contribution (amount_paid: ℕ) (discount: ℕ) := amount_paid + discount

def total_original_cost (individual_original: ℕ) (num_people: ℕ) := individual_original * num_people

def discount_amount (original_cost: ℕ) (discounted_cost: ℕ) := original_cost - discounted_cost

def discount_percentage (discount: ℕ) (original_cost: ℕ) := (discount * 100) / original_cost

-- Given conditions
def given_total := 48
def given_num_people := 3
def amount_paid_each := equal_contribution given_total given_num_people
def discount_each := 4
def original_payment_each := original_contribution amount_paid_each discount_each
def original_total_cost := total_original_cost original_payment_each given_num_people
def paid_total := 48

-- Question: What is the percentage discount
theorem percentage_discount :
  discount_percentage (discount_amount original_total_cost paid_total) original_total_cost = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_discount_l704_70475


namespace NUMINAMATH_GPT_number_of_ways_to_assign_friends_to_teams_l704_70418

theorem number_of_ways_to_assign_friends_to_teams (n m : ℕ) (h_n : n = 7) (h_m : m = 4) : m ^ n = 16384 :=
by
  rw [h_n, h_m]
  exact pow_succ' 4 6

end NUMINAMATH_GPT_number_of_ways_to_assign_friends_to_teams_l704_70418


namespace NUMINAMATH_GPT_Ronaldinho_age_2018_l704_70484

variable (X : ℕ)

theorem Ronaldinho_age_2018 (h : X^2 = 2025) : X - (2025 - 2018) = 38 := by
  sorry

end NUMINAMATH_GPT_Ronaldinho_age_2018_l704_70484


namespace NUMINAMATH_GPT_calculate_expression_l704_70401

def x : Float := 3.241
def y : Float := 14
def z : Float := 100
def expected_result : Float := 0.45374

theorem calculate_expression : (x * y) / z = expected_result := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l704_70401


namespace NUMINAMATH_GPT_cards_given_l704_70467

-- Defining the conditions
def initial_cards : ℕ := 4
def final_cards : ℕ := 12

-- The theorem to be proved
theorem cards_given : final_cards - initial_cards = 8 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_cards_given_l704_70467


namespace NUMINAMATH_GPT_A_lt_B_l704_70428

variable (x y : ℝ)

def A (x y : ℝ) : ℝ := - y^2 + 4 * x - 3
def B (x y : ℝ) : ℝ := x^2 + 2 * x + 2 * y

theorem A_lt_B (x y : ℝ) : A x y < B x y := 
by
  sorry

end NUMINAMATH_GPT_A_lt_B_l704_70428


namespace NUMINAMATH_GPT_apples_in_each_basket_l704_70489

-- Definitions based on the conditions
def total_apples : ℕ := 64
def baskets : ℕ := 4
def apples_taken_per_basket : ℕ := 3

-- Theorem statement based on the question and correct answer
theorem apples_in_each_basket (h1 : total_apples = 64) 
                              (h2 : baskets = 4) 
                              (h3 : apples_taken_per_basket = 3) : 
    (total_apples / baskets - apples_taken_per_basket) = 13 := 
by
  sorry

end NUMINAMATH_GPT_apples_in_each_basket_l704_70489


namespace NUMINAMATH_GPT_least_positive_int_satisfies_congruence_l704_70465

theorem least_positive_int_satisfies_congruence :
  ∃ x : ℕ, (x + 3001) % 15 = 1723 % 15 ∧ x = 12 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_int_satisfies_congruence_l704_70465


namespace NUMINAMATH_GPT_tom_coins_worth_l704_70471

-- Definitions based on conditions:
def total_coins : ℕ := 30
def value_difference_cents : ℕ := 90
def nickel_value_cents : ℕ := 5
def dime_value_cents : ℕ := 10

-- Main theorem statement:
theorem tom_coins_worth (n d : ℕ) (h1 : d = total_coins - n) 
    (h2 : (nickel_value_cents * n + dime_value_cents * d) - (dime_value_cents * n + nickel_value_cents * d) = value_difference_cents) : 
    (nickel_value_cents * n + dime_value_cents * d) = 180 :=
by
  sorry -- Proof omitted.

end NUMINAMATH_GPT_tom_coins_worth_l704_70471


namespace NUMINAMATH_GPT_interval_of_x_l704_70434

theorem interval_of_x (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) → (2 < 5 * x ∧ 5 * x < 3) → (1 / 2 < x ∧ x < 3 / 5) :=
by
  sorry

end NUMINAMATH_GPT_interval_of_x_l704_70434


namespace NUMINAMATH_GPT_cos_sum_nonneg_one_l704_70407

theorem cos_sum_nonneg_one (x y z : ℝ) (h : x + y + z = 0) : abs (Real.cos x) + abs (Real.cos y) + abs (Real.cos z) ≥ 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_cos_sum_nonneg_one_l704_70407


namespace NUMINAMATH_GPT_trig_identity_eq_one_l704_70439

theorem trig_identity_eq_one :
  (Real.sin (160 * Real.pi / 180) + Real.sin (40 * Real.pi / 180)) *
  (Real.sin (140 * Real.pi / 180) + Real.sin (20 * Real.pi / 180)) +
  (Real.sin (50 * Real.pi / 180) - Real.sin (70 * Real.pi / 180)) *
  (Real.sin (130 * Real.pi / 180) - Real.sin (110 * Real.pi / 180)) =
  1 :=
sorry

end NUMINAMATH_GPT_trig_identity_eq_one_l704_70439


namespace NUMINAMATH_GPT_findMultipleOfSamsMoney_l704_70444

-- Define the conditions specified in the problem
def SamMoney : ℕ := 75
def TotalMoney : ℕ := 200
def BillyHasLess (x : ℕ) : ℕ := x * SamMoney - 25

-- State the theorem to prove
theorem findMultipleOfSamsMoney (x : ℕ) 
  (h1 : SamMoney + BillyHasLess x = TotalMoney) : x = 2 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_findMultipleOfSamsMoney_l704_70444


namespace NUMINAMATH_GPT_brian_gallons_usage_l704_70474

/-
Brian’s car gets 20 miles per gallon. 
On his last trip, he traveled 60 miles. 
How many gallons of gas did he use?
-/

theorem brian_gallons_usage (miles_per_gallon : ℝ) (total_miles : ℝ) (gallons_used : ℝ) 
    (h1 : miles_per_gallon = 20) 
    (h2 : total_miles = 60) 
    (h3 : gallons_used = total_miles / miles_per_gallon) : 
    gallons_used = 3 := 
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end NUMINAMATH_GPT_brian_gallons_usage_l704_70474


namespace NUMINAMATH_GPT_inequality_with_sum_of_one_l704_70411

theorem inequality_with_sum_of_one
  (a b c d : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_sum: a + b + c + d = 1) :
  (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) >= 1 / 2) :=
sorry

end NUMINAMATH_GPT_inequality_with_sum_of_one_l704_70411


namespace NUMINAMATH_GPT_equation_1_solve_equation_2_solve_l704_70464

-- The first equation
theorem equation_1_solve (x : ℝ) (h : 4 * (x - 2) = 2 * x) : x = 4 :=
by
  sorry

-- The second equation
theorem equation_2_solve (x : ℝ) (h : (x + 1) / 4 = 1 - (1 - x) / 3) : x = -5 :=
by
  sorry

end NUMINAMATH_GPT_equation_1_solve_equation_2_solve_l704_70464


namespace NUMINAMATH_GPT_max_radius_of_inscribable_circle_l704_70462

theorem max_radius_of_inscribable_circle
  (AB BC CD DA : ℝ) (x y z w : ℝ)
  (h1 : AB = 10) (h2 : BC = 12) (h3 : CD = 8) (h4 : DA = 14)
  (h5 : x + y = 10) (h6 : y + z = 12)
  (h7 : z + w = 8) (h8 : w + x = 14)
  (h9 : x + z = y + w) :
  ∃ r : ℝ, r = Real.sqrt 24.75 :=
by
  sorry

end NUMINAMATH_GPT_max_radius_of_inscribable_circle_l704_70462


namespace NUMINAMATH_GPT_evaluate_number_l704_70452

theorem evaluate_number (n : ℝ) (h : 22 + Real.sqrt (-4 + 6 * 4 * n) = 24) : n = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_number_l704_70452


namespace NUMINAMATH_GPT_find_total_kids_l704_70436

-- Given conditions
def total_kids_in_camp (X : ℕ) : Prop :=
  let soccer_kids := X / 2
  let morning_soccer_kids := soccer_kids / 4
  let afternoon_soccer_kids := soccer_kids - morning_soccer_kids
  afternoon_soccer_kids = 750

-- Theorem statement
theorem find_total_kids (X : ℕ) (h : total_kids_in_camp X) : X = 2000 :=
by
  sorry

end NUMINAMATH_GPT_find_total_kids_l704_70436


namespace NUMINAMATH_GPT_log2_a_plus_log2_b_zero_l704_70426

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem log2_a_plus_log2_b_zero 
    (a b : ℝ) 
    (h : (Nat.choose 6 3) * (a^3) * (b^3) = 20) 
    (hc : (a^2 + b / a)^(3) = 20 * x^(3)) :
  log2 a + log2 b = 0 :=
by
  sorry

end NUMINAMATH_GPT_log2_a_plus_log2_b_zero_l704_70426


namespace NUMINAMATH_GPT_keith_bought_cards_l704_70424

theorem keith_bought_cards (orig : ℕ) (now : ℕ) (bought : ℕ) 
  (h1 : orig = 40) (h2 : now = 18) (h3 : bought = orig - now) : bought = 22 := by
  sorry

end NUMINAMATH_GPT_keith_bought_cards_l704_70424


namespace NUMINAMATH_GPT_probability_correct_l704_70404

noncomputable def probability_all_players_have_5_after_2023_rings 
    (initial_money : ℕ)
    (num_rings : ℕ) 
    (target_money : ℕ)
    : ℝ := 
    if initial_money = 5 ∧ num_rings = 2023 ∧ target_money = 5 
    then 1 / 4 
    else 0

theorem probability_correct : 
        probability_all_players_have_5_after_2023_rings 5 2023 5 = 1 / 4 := 
by 
    sorry

end NUMINAMATH_GPT_probability_correct_l704_70404


namespace NUMINAMATH_GPT_union_A_B_subset_B_A_l704_70406

-- Condition definitions
def A : Set ℝ := {x | 2 * x - 8 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2 * (m + 1) * x + m^2 = 0}

-- Problem 1: If m = 4, prove A ∪ B = {2, 4, 8}
theorem union_A_B (m : ℝ) (h : m = 4) : A ∪ B m = {2, 4, 8} :=
sorry

-- Problem 2: If B ⊆ A, find the range for m
theorem subset_B_A (m : ℝ) (h : B m ⊆ A) : 
  m = 4 + 2 * Real.sqrt 2 ∨ m = 4 - 2 * Real.sqrt 2 ∨ m < -1 / 2 :=
sorry

end NUMINAMATH_GPT_union_A_B_subset_B_A_l704_70406


namespace NUMINAMATH_GPT_prize_winners_l704_70417

theorem prize_winners (total_people : ℕ) (percent_envelope : ℝ) (percent_win : ℝ) 
  (h_total : total_people = 100) (h_percent_envelope : percent_envelope = 0.40) 
  (h_percent_win : percent_win = 0.20) : 
  (percent_win * (percent_envelope * total_people)) = 8 := by
  sorry

end NUMINAMATH_GPT_prize_winners_l704_70417


namespace NUMINAMATH_GPT_primes_with_prime_remainders_l704_70403

namespace PrimePuzzle

open Nat

def primes_between (a b : Nat) : List Nat :=
  (List.range' (a + 1) (b - a)).filter Nat.Prime

def prime_remainders (lst : List Nat) (m : Nat) : List Nat :=
  (lst.map (λ n => n % m)).filter Nat.Prime

theorem primes_with_prime_remainders : 
  primes_between 40 85 = [41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83] ∧ 
  prime_remainders [41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83] 12 = [5, 7, 7, 11, 11, 7, 11] ∧ 
  (prime_remainders [41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83] 12).toFinset.card = 9 := 
by 
  sorry

end PrimePuzzle

end NUMINAMATH_GPT_primes_with_prime_remainders_l704_70403
