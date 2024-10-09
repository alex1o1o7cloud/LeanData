import Mathlib

namespace nancy_first_album_pictures_l2072_207230

theorem nancy_first_album_pictures (total_pics : ℕ) (total_albums : ℕ) (pics_per_album : ℕ)
    (h1 : total_pics = 51) (h2 : total_albums = 8) (h3 : pics_per_album = 5) :
    (total_pics - total_albums * pics_per_album = 11) :=
by
    sorry

end nancy_first_album_pictures_l2072_207230


namespace distinct_real_numbers_a_l2072_207203

theorem distinct_real_numbers_a (a x y z : ℝ) (h_distinct: x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  (a = x + 1 / y ∧ a = y + 1 / z ∧ a = z + 1 / x) ↔ (a = 1 ∨ a = -1) :=
by sorry

end distinct_real_numbers_a_l2072_207203


namespace cubic_identity_l2072_207283

theorem cubic_identity (a b c : ℝ) (h1 : a + b + c = 7) (h2 : ab + ac + bc = 11) (h3 : abc = -6) : a^3 + b^3 + c^3 = 94 :=
by
  sorry

end cubic_identity_l2072_207283


namespace num_pass_students_is_85_l2072_207273

theorem num_pass_students_is_85 (T P F : ℕ) (avg_all avg_pass avg_fail : ℕ) (weight_pass weight_fail : ℕ) 
  (h_total_students : T = 150)
  (h_avg_all : avg_all = 40)
  (h_avg_pass : avg_pass = 45)
  (h_avg_fail : avg_fail = 20)
  (h_weight_ratio : weight_pass = 3 ∧ weight_fail = 1)
  (h_total_marks : (weight_pass * avg_pass * P + weight_fail * avg_fail * F) / (weight_pass * P + weight_fail * F) = avg_all)
  (h_students_sum : P + F = T) :
  P = 85 :=
by
  sorry

end num_pass_students_is_85_l2072_207273


namespace ab_difference_l2072_207220

theorem ab_difference (a b : ℤ) (h1 : |a| = 5) (h2 : |b| = 3) (h3 : a + b > 0) : a - b = 2 ∨ a - b = 8 :=
sorry

end ab_difference_l2072_207220


namespace John_height_l2072_207255

open Real

variable (John Mary Tom Angela Helen Amy Becky Carl : ℝ)

axiom h1 : John = 1.5 * Mary
axiom h2 : Mary = 2 * Tom
axiom h3 : Tom = Angela - 70
axiom h4 : Angela = Helen + 4
axiom h5 : Helen = Amy + 3
axiom h6 : Amy = 1.2 * Becky
axiom h7 : Becky = 2 * Carl
axiom h8 : Carl = 120

theorem John_height : John = 675 := by
  sorry

end John_height_l2072_207255


namespace remainder_101_pow_50_mod_100_l2072_207293

theorem remainder_101_pow_50_mod_100 : (101 ^ 50) % 100 = 1 := by
  sorry

end remainder_101_pow_50_mod_100_l2072_207293


namespace stating_martha_painting_time_l2072_207242

/-- 
  Theorem stating the time it takes for Martha to paint the kitchen is 42 hours.
-/
theorem martha_painting_time :
  let width1 := 12
  let width2 := 16
  let height := 10
  let area_pair1 := 2 * width1 * height
  let area_pair2 := 2 * width2 * height
  let total_area := area_pair1 + area_pair2
  let coats := 3
  let total_paint_area := total_area * coats
  let painting_speed := 40
  let time_required := total_paint_area / painting_speed
  time_required = 42 := by
    -- Since we are asked not to provide the proof steps, we use sorry to skip the proof.
    sorry

end stating_martha_painting_time_l2072_207242


namespace find_C_l2072_207234

theorem find_C (A B C : ℝ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 340) : C = 40 :=
by sorry

end find_C_l2072_207234


namespace correct_statement_four_l2072_207279

variable {α : Type*} (A B S : Set α) (U : Set α)

theorem correct_statement_four (h1 : U = Set.univ) (h2 : A ∩ B = U) : A = U ∧ B = U := by
  sorry

end correct_statement_four_l2072_207279


namespace minimum_positive_period_minimum_value_l2072_207240

noncomputable def f (x : Real) : Real :=
  Real.sin (x / 5) - Real.cos (x / 5)

theorem minimum_positive_period (T : Real) : (∀ x, f (x + T) = f x) ∧ T > 0 → T = 10 * Real.pi :=
  sorry

theorem minimum_value : ∃ x, f x = -Real.sqrt 2 :=
  sorry

end minimum_positive_period_minimum_value_l2072_207240


namespace candy_left_l2072_207228

-- Define the number of candies each sibling has
def debbyCandy : ℕ := 32
def sisterCandy : ℕ := 42
def brotherCandy : ℕ := 48

-- Define the total candies collected
def totalCandy : ℕ := debbyCandy + sisterCandy + brotherCandy

-- Define the number of candies eaten
def eatenCandy : ℕ := 56

-- Define the remaining candies after eating some
def remainingCandy : ℕ := totalCandy - eatenCandy

-- The hypothesis stating the initial condition
theorem candy_left (h1 : debbyCandy = 32) (h2 : sisterCandy = 42) (h3 : brotherCandy = 48) (h4 : eatenCandy = 56) : remainingCandy = 66 :=
by
  -- Proof can be filled in here
  sorry

end candy_left_l2072_207228


namespace polynomial_factorization_l2072_207208

noncomputable def polynomial_expr (a b c : ℝ) :=
  a^4 * (b^2 - c^2) + b^4 * (c^2 - a^2) + c^4 * (a^2 - b^2)

noncomputable def factored_form (a b c : ℝ) :=
  (a - b) * (b - c) * (c - a) * (b^2 + c^2 + a^2)

theorem polynomial_factorization (a b c : ℝ) :
  polynomial_expr a b c = factored_form a b c :=
by {
  sorry
}

end polynomial_factorization_l2072_207208


namespace ratio_problem_l2072_207289

-- Given condition: a, b, c are in the ratio 2:3:4
theorem ratio_problem (a b c : ℝ) (h1 : a / b = 2 / 3) (h2 : a / c = 2 / 4) : 
  (a - b + c) / b = 1 :=
by
  -- Placeholder for the actual proof
  sorry

end ratio_problem_l2072_207289


namespace grasshopper_jump_distance_l2072_207243

theorem grasshopper_jump_distance (frog_jump grasshopper_jump : ℝ) (h_frog : frog_jump = 40) (h_difference : frog_jump = grasshopper_jump + 15) : grasshopper_jump = 25 :=
by sorry

end grasshopper_jump_distance_l2072_207243


namespace water_current_speed_l2072_207281

-- Definitions based on the conditions
def swimmer_speed : ℝ := 4  -- The swimmer's speed in still water (km/h)
def swim_time : ℝ := 2  -- Time taken to swim against the current (hours)
def swim_distance : ℝ := 6  -- Distance swum against the current (km)

-- The effective speed against the current
noncomputable def effective_speed_against_current (v : ℝ) : ℝ := swimmer_speed - v

-- Lean statement that formalizes proving the speed of the current
theorem water_current_speed (v : ℝ) (h : effective_speed_against_current v = swim_distance / swim_time) : v = 1 :=
by
  sorry

end water_current_speed_l2072_207281


namespace wendys_sales_are_205_l2072_207280

def price_of_apple : ℝ := 1.5
def price_of_orange : ℝ := 1.0
def apples_sold_morning : ℕ := 40
def oranges_sold_morning : ℕ := 30
def apples_sold_afternoon : ℕ := 50
def oranges_sold_afternoon : ℕ := 40

/-- Wendy's total sales for the day are $205 given the conditions about the prices of apples and oranges,
and the number of each sold in the morning and afternoon. -/
def wendys_total_sales : ℝ :=
  let total_apples_sold := apples_sold_morning + apples_sold_afternoon
  let total_oranges_sold := oranges_sold_morning + oranges_sold_afternoon
  let sales_from_apples := total_apples_sold * price_of_apple
  let sales_from_oranges := total_oranges_sold * price_of_orange
  sales_from_apples + sales_from_oranges

theorem wendys_sales_are_205 : wendys_total_sales = 205 := by
  sorry

end wendys_sales_are_205_l2072_207280


namespace solution_set_for_rational_inequality_l2072_207261

theorem solution_set_for_rational_inequality (x : ℝ) :
  (x - 2) / (x - 1) > 0 ↔ x < 1 ∨ x > 2 := 
sorry

end solution_set_for_rational_inequality_l2072_207261


namespace number_of_yogurts_l2072_207264

def slices_per_yogurt : Nat := 8
def slices_per_banana : Nat := 10
def number_of_bananas : Nat := 4

theorem number_of_yogurts (slices_per_yogurt slices_per_banana number_of_bananas : Nat) : 
  slices_per_yogurt = 8 → 
  slices_per_banana = 10 → 
  number_of_bananas = 4 → 
  (number_of_bananas * slices_per_banana) / slices_per_yogurt = 5 :=
by
  intros h1 h2 h3
  sorry

end number_of_yogurts_l2072_207264


namespace impossibility_of_4_level_ideal_interval_tan_l2072_207212

def has_ideal_interval (f : ℝ → ℝ) (D : Set ℝ) (k : ℝ) :=
  ∃ (a b : ℝ), a ≤ b ∧ Set.Icc a b ⊆ D ∧ (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y ∨ f y ≤ f x) ∧
  (Set.image f (Set.Icc a b) = Set.Icc (k * a) (k * b))

def option_D_incorrect : Prop :=
  ¬ has_ideal_interval (fun x => Real.tan x) (Set.Ioc (-(Real.pi / 2)) (Real.pi / 2)) 4

theorem impossibility_of_4_level_ideal_interval_tan :
  option_D_incorrect :=
sorry

end impossibility_of_4_level_ideal_interval_tan_l2072_207212


namespace divisor_between_l2072_207221

theorem divisor_between (n a b : ℕ) (h_n_gt_8 : n > 8) (h_a_dvd_n : a ∣ n) (h_b_dvd_n : b ∣ n) 
    (h_a_lt_b : a < b) (h_n_eq_asq_plus_b : n = a^2 + b) (h_a_ne_b : a ≠ b) :
  ∃ d : ℕ, d ∣ n ∧ a < d ∧ d < b :=
sorry

end divisor_between_l2072_207221


namespace coeff_fourth_term_expansion_l2072_207214

theorem coeff_fourth_term_expansion :
  (3 : ℚ) ^ 2 * (-1 : ℚ) / 8 * (Nat.choose 8 3) = -63 :=
by
  sorry

end coeff_fourth_term_expansion_l2072_207214


namespace solve_mod_equation_l2072_207256

def is_two_digit_positive_integer (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem solve_mod_equation (u : ℕ) (h1 : is_two_digit_positive_integer u) (h2 : 13 * u % 100 = 52) : u = 4 :=
sorry

end solve_mod_equation_l2072_207256


namespace min_box_coeff_l2072_207218

theorem min_box_coeff (a b c d : ℤ) (h_ac : a * c = 40) (h_bd : b * d = 40) : 
  ∃ (min_val : ℤ), min_val = 89 ∧ (a * d + b * c) ≥ min_val :=
sorry

end min_box_coeff_l2072_207218


namespace problem_1_problem_2_l2072_207277

noncomputable def complete_residue_system (n : ℕ) (as : Fin n → ℕ) :=
  ∀ i j : Fin n, i ≠ j → as i % n ≠ as j % n

theorem problem_1 (n : ℕ) (hn : 0 < n) :
  ∃ as : Fin n → ℕ, complete_residue_system n as ∧ complete_residue_system n (λ i => as i + i) := 
sorry

theorem problem_2 (n : ℕ) (hn : 0 < n) :
  ∃ as : Fin n → ℕ, complete_residue_system n as ∧ complete_residue_system n (λ i => as i + i) ∧ complete_residue_system n (λ i => as i - i) := 
sorry

end problem_1_problem_2_l2072_207277


namespace impossible_to_use_up_components_l2072_207254

theorem impossible_to_use_up_components 
  (p q r x y z : ℕ) 
  (condition1 : 2 * x + 2 * z = 2 * p + 2 * r + 2)
  (condition2 : 2 * x + y = 2 * p + q + 1)
  (condition3 : y + z = q + r) : 
  False :=
by sorry

end impossible_to_use_up_components_l2072_207254


namespace find_treasure_island_l2072_207202

-- Define the types for the three islands
inductive Island : Type
| A | B | C

-- Define the possible inhabitants of island A
inductive Inhabitant : Type
| Knight  -- always tells the truth
| Liar    -- always lies
| Normal  -- might tell the truth or lie

-- Define the conditions
def no_treasure_on_A : Prop := ¬ ∃ (x : Island), x = Island.A ∧ (x = Island.A)
def normal_people_on_A_two_treasures : Prop := ∀ (h : Inhabitant), h = Inhabitant.Normal → (∃ (x y : Island), x ≠ y ∧ (x ≠ Island.A ∧ y ≠ Island.A))

-- The question to ask
def question_to_ask (h : Inhabitant) : Prop :=
  (h = Inhabitant.Knight) ↔ (∃ (x : Island), (x = Island.B) ∧ (¬ ∃ (y : Island), (y = Island.A) ∧ (y = Island.A)))

-- The theorem statement
theorem find_treasure_island (inh : Inhabitant) :
  no_treasure_on_A ∧ normal_people_on_A_two_treasures →
  (question_to_ask inh → (∃ (x : Island), x = Island.B)) ∧ (¬ question_to_ask inh → (∃ (x : Island), x = Island.C)) :=
by
  intro h
  sorry

end find_treasure_island_l2072_207202


namespace solve_equation_1_solve_equation_2_l2072_207298

namespace Proofs

theorem solve_equation_1 (x : ℝ) :
  (x + 1)^2 = 9 ↔ x = 2 ∨ x = -4 :=
by
  sorry

theorem solve_equation_2 (x : ℝ) :
  x * (x - 6) = 6 ↔ x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15 :=
by
  sorry

end Proofs

end solve_equation_1_solve_equation_2_l2072_207298


namespace alex_minus_sam_eq_negative_2_50_l2072_207217

def original_price : ℝ := 120.00
def tax_rate : ℝ := 0.07
def discount_rate : ℝ := 0.15
def packaging_fee : ℝ := 2.50

def alex_total (original_price tax_rate discount_rate : ℝ) : ℝ :=
  let price_with_tax := original_price * (1 + tax_rate)
  let final_price := price_with_tax * (1 - discount_rate)
  final_price

def sam_total (original_price tax_rate discount_rate packaging_fee : ℝ) : ℝ :=
  let price_with_discount := original_price * (1 - discount_rate)
  let price_with_tax := price_with_discount * (1 + tax_rate)
  let final_price := price_with_tax + packaging_fee
  final_price

theorem alex_minus_sam_eq_negative_2_50 :
  alex_total original_price tax_rate discount_rate - sam_total original_price tax_rate discount_rate packaging_fee = -2.50 := by
  sorry

end alex_minus_sam_eq_negative_2_50_l2072_207217


namespace ratio_of_rectangle_to_square_l2072_207270

theorem ratio_of_rectangle_to_square (s w h : ℝ) 
  (hs : h = s / 2)
  (shared_area_ABCD_EFGH_1 : 0.25 * s^2 = 0.4 * w * h)
  (shared_area_ABCD_EFGH_2 : 0.25 * s^2 = 0.4 * w * h) :
  w / h = 2.5 :=
by
  -- Proof goes here
  sorry

end ratio_of_rectangle_to_square_l2072_207270


namespace rectangle_width_l2072_207232

theorem rectangle_width (length : ℕ) (perimeter : ℕ) (h1 : length = 20) (h2 : perimeter = 70) :
  2 * (length + width) = perimeter → width = 15 :=
by
  intro h
  rw [h1, h2] at h
  -- Continue the steps to solve for width (can be simplified if not requesting the whole proof)
  sorry

end rectangle_width_l2072_207232


namespace random_event_proof_l2072_207299

def is_certain_event (event: Prop) : Prop := ∃ h: event → true, ∃ h': true → event, true
def is_impossible_event (event: Prop) : Prop := event → false
def is_random_event (event: Prop) : Prop := ¬is_certain_event event ∧ ¬is_impossible_event event

def cond1 : Prop := sorry -- Yingying encounters a green light
def cond2 : Prop := sorry -- A non-transparent bag contains one ping-pong ball and two glass balls of the same size, and a ping-pong ball is drawn from it.
def cond3 : Prop := sorry -- You are currently answering question 12 of this test paper.
def cond4 : Prop := sorry -- The highest temperature in our city tomorrow will be 60°C.

theorem random_event_proof : 
  is_random_event cond1 ∧ 
  ¬is_random_event cond2 ∧ 
  ¬is_random_event cond3 ∧ 
  ¬is_random_event cond4 :=
by
  sorry

end random_event_proof_l2072_207299


namespace rectangle_relationships_l2072_207207

theorem rectangle_relationships (x y S : ℝ) (h1 : 2 * x + 2 * y = 10) (h2 : S = x * y) :
  y = 5 - x ∧ S = 5 * x - x ^ 2 :=
by
  sorry

end rectangle_relationships_l2072_207207


namespace probability_first_head_second_tail_l2072_207249

-- Conditions
def fair_coin := true
def prob_heads := 1 / 2
def prob_tails := 1 / 2
def independent_events (A B : Prop) := true

-- Statement
theorem probability_first_head_second_tail :
  fair_coin →
  independent_events (prob_heads = 1/2) (prob_tails = 1/2) →
  (prob_heads * prob_tails) = 1/4 :=
by
  sorry

end probability_first_head_second_tail_l2072_207249


namespace remainder_n_squared_plus_3n_plus_4_l2072_207225

theorem remainder_n_squared_plus_3n_plus_4 (n : ℤ) (h : n % 100 = 99) : (n^2 + 3*n + 4) % 100 = 2 := 
by sorry

end remainder_n_squared_plus_3n_plus_4_l2072_207225


namespace solve_inequality_l2072_207263

theorem solve_inequality (a : ℝ) : 
  (if a = 0 ∨ a = 1 then { x : ℝ | false }
   else if a < 0 ∨ a > 1 then { x : ℝ | a < x ∧ x < a^2 }
   else if 0 < a ∧ a < 1 then { x : ℝ | a^2 < x ∧ x < a }
   else ∅) = 
  { x : ℝ | (x - a) / (x - a^2) < 0 } :=
by sorry

end solve_inequality_l2072_207263


namespace minimum_time_to_cook_l2072_207290

def wash_pot_fill_water : ℕ := 2
def wash_vegetables : ℕ := 3
def prepare_noodles_seasonings : ℕ := 2
def boil_water : ℕ := 7
def cook_noodles_vegetables : ℕ := 3

theorem minimum_time_to_cook : wash_pot_fill_water + boil_water + cook_noodles_vegetables = 12 :=
by
  sorry

end minimum_time_to_cook_l2072_207290


namespace sqrt_meaningful_value_x_l2072_207216

theorem sqrt_meaningful_value_x (x : ℝ) (h : x-1 ≥ 0) : x = 2 :=
by
  sorry

end sqrt_meaningful_value_x_l2072_207216


namespace quinton_cupcakes_l2072_207278

theorem quinton_cupcakes (students_Delmont : ℕ) (students_Donnelly : ℕ)
                         (num_teachers_nurse_principal : ℕ) (leftover : ℕ) :
  students_Delmont = 18 → students_Donnelly = 16 →
  num_teachers_nurse_principal = 4 → leftover = 2 →
  students_Delmont + students_Donnelly + num_teachers_nurse_principal + leftover = 40 :=
by
  intros h1 h2 h3 h4
  sorry

end quinton_cupcakes_l2072_207278


namespace hyperbola_asymptote_equation_l2072_207236

variable (a b : ℝ)
variable (x y : ℝ)

def arithmetic_mean := (a + b) / 2 = 5
def geometric_mean := (a * b) ^ (1 / 2) = 4
def a_greater_b := a > b
def hyperbola_asymptote := (y = (1 / 2) * x) ∨ (y = -(1 / 2) * x)

theorem hyperbola_asymptote_equation :
  arithmetic_mean a b ∧ geometric_mean a b ∧ a_greater_b a b → hyperbola_asymptote x y :=
by
  sorry

end hyperbola_asymptote_equation_l2072_207236


namespace no_integer_points_on_circle_l2072_207266

theorem no_integer_points_on_circle : 
  ∀ x : ℤ, ¬ ((x - 3)^2 + (x + 1 + 2)^2 ≤ 64) := by
  sorry

end no_integer_points_on_circle_l2072_207266


namespace polynomial_remainder_l2072_207200

theorem polynomial_remainder :
  ∀ (q : Polynomial ℚ), (3 * X^5 - 2 * X^3 + 5 * X - 9) = (X - 1) * (X - 2) * q + (92 * X - 95) :=
by
  intro q
  sorry

end polynomial_remainder_l2072_207200


namespace angle_A_is_70_l2072_207257

-- Definitions of angles given as conditions in the problem
variables (BAD BAC ACB : ℝ)

def angle_BAD := 150
def angle_BAC := 80

-- The Lean 4 statement to prove the measure of angle ACB
theorem angle_A_is_70 (h1 : BAD = 150) (h2 : BAC = 80) : ACB = 70 :=
by {
  sorry
}

end angle_A_is_70_l2072_207257


namespace ryan_hours_on_english_l2072_207259

-- Given the conditions
def hours_on_chinese := 2
def hours_on_spanish := 4
def extra_hours_between_english_and_spanish := 3

-- We want to find out the hours on learning English
def hours_on_english := hours_on_spanish + extra_hours_between_english_and_spanish

-- Proof statement
theorem ryan_hours_on_english : hours_on_english = 7 := by
  -- This is where the proof would normally go.
  sorry

end ryan_hours_on_english_l2072_207259


namespace correct_proposition_is_D_l2072_207286

theorem correct_proposition_is_D (A B C D : Prop) :
  (∀ (H : Prop), (H = A ∨ H = B ∨ H = C) → ¬H) → D :=
by
  -- We assume that A, B, and C are false.
  intro h
  -- Now we need to prove that D is true.
  sorry

end correct_proposition_is_D_l2072_207286


namespace range_of_k_for_quadratic_inequality_l2072_207241

theorem range_of_k_for_quadratic_inequality (k : ℝ) :
  (∀ x : ℝ, k * x^2 + 2 * k * x - 1 < 0) ↔ (-1 < k ∧ k ≤ 0) :=
  sorry

end range_of_k_for_quadratic_inequality_l2072_207241


namespace gcd_square_of_difference_l2072_207252

theorem gcd_square_of_difference (x y z : ℕ) (h : 1/x - 1/y = 1/z) :
  ∃ k : ℕ, (Nat.gcd (Nat.gcd x y) z) * (y - x) = k^2 :=
by
  sorry

end gcd_square_of_difference_l2072_207252


namespace common_divisor_of_differences_l2072_207250

theorem common_divisor_of_differences 
  (a1 a2 b1 b2 c1 c2 d : ℤ) 
  (h1: d ∣ (a1 - a2)) 
  (h2: d ∣ (b1 - b2)) 
  (h3: d ∣ (c1 - c2)) : 
  d ∣ (a1 * b1 * c1 - a2 * b2 * c2) := 
by sorry

end common_divisor_of_differences_l2072_207250


namespace solve_quadratic_inequality_l2072_207294

theorem solve_quadratic_inequality (x : ℝ) (h : x^2 - 7 * x + 6 < 0) : 1 < x ∧ x < 6 :=
  sorry

end solve_quadratic_inequality_l2072_207294


namespace positive_integer_solutions_of_inequality_system_l2072_207201

theorem positive_integer_solutions_of_inequality_system :
  {x : ℤ | 2 * (x - 1) < x + 1 ∧ 1 - (2 * x + 5) / 3 ≤ x ∧ x > 0} = {1, 2} :=
by
  sorry

end positive_integer_solutions_of_inequality_system_l2072_207201


namespace Q_2_plus_Q_neg_2_l2072_207262

noncomputable def cubic_polynomial (a b c k : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + k

theorem Q_2_plus_Q_neg_2 (a b c k : ℝ) 
  (h0 : cubic_polynomial a b c k 0 = k)
  (h1 : cubic_polynomial a b c k 1 = 3 * k)
  (hneg1 : cubic_polynomial a b c k (-1) = 4 * k) :
  cubic_polynomial a b c k 2 + cubic_polynomial a b c k (-2) = 22 * k :=
sorry

end Q_2_plus_Q_neg_2_l2072_207262


namespace cricket_player_average_l2072_207275

theorem cricket_player_average (A : ℕ)
  (H1 : 10 * A + 62 = 11 * (A + 4)) : A = 18 :=
by {
  sorry -- The proof itself
}

end cricket_player_average_l2072_207275


namespace purse_multiple_of_wallet_l2072_207222

theorem purse_multiple_of_wallet (W P : ℤ) (hW : W = 22) (hc : W + P = 107) : ∃ n : ℤ, n * W > P ∧ n = 4 :=
by
  sorry

end purse_multiple_of_wallet_l2072_207222


namespace A_inter_B_is_correct_l2072_207215

def set_A : Set ℤ := { x : ℤ | x^2 - x - 2 ≤ 0 }
def set_B : Set ℤ := { x : ℤ | True }

theorem A_inter_B_is_correct : set_A ∩ set_B = { -1, 0, 1, 2 } := by
  sorry

end A_inter_B_is_correct_l2072_207215


namespace nonneg_int_solution_coprime_l2072_207271

theorem nonneg_int_solution_coprime (a b c : ℕ) (h1 : Nat.gcd a b = 1) (h2 : c ≥ (a - 1) * (b - 1)) :
  ∃ (x y : ℕ), c = a * x + b * y :=
sorry

end nonneg_int_solution_coprime_l2072_207271


namespace polynomial_roots_l2072_207282

theorem polynomial_roots :
  ∀ x : ℝ, (4 * x^4 - 28 * x^3 + 53 * x^2 - 28 * x + 4 = 0) ↔ (x = 4 ∨ x = 2 ∨ x = 1/4 ∨ x = 1/2) := 
by
  sorry

end polynomial_roots_l2072_207282


namespace range_of_a_l2072_207285

variable {a x : ℝ}

theorem range_of_a (h : ∀ x, (a - 5) * x > a - 5 ↔ x < 1) : a < 5 := 
sorry

end range_of_a_l2072_207285


namespace combination_of_students_l2072_207292

-- Define the conditions
def num_boys := 4
def num_girls := 3

-- Define the combination function
def combination (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Calculate possible combinations
def two_boys_one_girl : ℕ :=
  combination num_boys 2 * combination num_girls 1

def one_boy_two_girls : ℕ :=
  combination num_boys 1 * combination num_girls 2

-- Total combinations
def total_combinations : ℕ :=
  two_boys_one_girl + one_boy_two_girls

-- Lean statement to be proven
theorem combination_of_students :
  total_combinations = 30 :=
by sorry

end combination_of_students_l2072_207292


namespace gymnast_scores_difference_l2072_207219

theorem gymnast_scores_difference
  (s1 s2 s3 s4 s5 : ℝ)
  (h1 : (s2 + s3 + s4 + s5) / 4 = 9.46)
  (h2 : (s1 + s2 + s3 + s4) / 4 = 9.66)
  (h3 : (s2 + s3 + s4) / 3 = 9.58)
  : |s5 - s1| = 8.3 :=
sorry

end gymnast_scores_difference_l2072_207219


namespace average_age_of_contestants_l2072_207297

theorem average_age_of_contestants :
  let numFemales := 12
  let avgAgeFemales := 25
  let numMales := 18
  let avgAgeMales := 40
  let sumAgesFemales := avgAgeFemales * numFemales
  let sumAgesMales := avgAgeMales * numMales
  let totalSumAges := sumAgesFemales + sumAgesMales
  let totalContestants := numFemales + numMales
  (totalSumAges / totalContestants) = 34 := by
  sorry

end average_age_of_contestants_l2072_207297


namespace selection_ways_l2072_207239

-- The statement of the problem in Lean 4
theorem selection_ways :
  (Nat.choose 50 4) - (Nat.choose 47 4) = 
  (Nat.choose 3 1) * (Nat.choose 47 3) + 
  (Nat.choose 3 2) * (Nat.choose 47 2) + 
  (Nat.choose 3 3) * (Nat.choose 47 1) := 
sorry

end selection_ways_l2072_207239


namespace find_weight_of_B_l2072_207274

theorem find_weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 44) : B = 33 :=
by 
  sorry

end find_weight_of_B_l2072_207274


namespace cos_diff_proof_l2072_207246

noncomputable def cos_diff (α β : ℝ) : ℝ := Real.cos (α - β)

theorem cos_diff_proof (α β : ℝ) 
  (h1 : Real.cos α - Real.cos β = 1 / 2)
  (h2 : Real.sin α - Real.sin β = 1 / 3) :
  cos_diff α β = 59 / 72 := by
  sorry

end cos_diff_proof_l2072_207246


namespace carson_circles_theorem_l2072_207231

-- Define the dimensions of the warehouse
def warehouse_length : ℕ := 600
def warehouse_width : ℕ := 400

-- Define the perimeter calculation
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)

-- Define the distance Carson walked
def distance_walked : ℕ := 16000

-- Define the number of circles Carson skipped
def circles_skipped : ℕ := 2

-- Define the expected number of circles Carson was supposed to circle
def expected_circles :=
  let actual_circles := distance_walked / (perimeter warehouse_length warehouse_width)
  actual_circles + circles_skipped

-- The theorem we want to prove
theorem carson_circles_theorem : expected_circles = 10 := by
  sorry

end carson_circles_theorem_l2072_207231


namespace leftover_value_is_5_30_l2072_207213

variable (q_per_roll d_per_roll : ℕ)
variable (j_quarters j_dimes l_quarters l_dimes : ℕ)
variable (value_per_quarter value_per_dime : ℝ)

def total_leftover_value (q_per_roll d_per_roll : ℕ) 
  (j_quarters l_quarters j_dimes l_dimes : ℕ)
  (value_per_quarter value_per_dime : ℝ) : ℝ :=
  let total_quarters := j_quarters + l_quarters
  let total_dimes := j_dimes + l_dimes
  let leftover_quarters := total_quarters % q_per_roll
  let leftover_dimes := total_dimes % d_per_roll
  (leftover_quarters * value_per_quarter) + (leftover_dimes * value_per_dime)

theorem leftover_value_is_5_30 :
  total_leftover_value 45 55 95 140 173 285 0.25 0.10 = 5.3 := 
by
  sorry

end leftover_value_is_5_30_l2072_207213


namespace divisible_by_133_l2072_207265

theorem divisible_by_133 (n : ℕ) : (11^(n + 2) + 12^(2*n + 1)) % 133 = 0 :=
by
  sorry

end divisible_by_133_l2072_207265


namespace apple_allocation_proof_l2072_207229

theorem apple_allocation_proof : 
    ∃ (ann mary jane kate ned tom bill jack : ℕ), 
    ann = 1 ∧
    mary = 2 ∧
    jane = 3 ∧
    kate = 4 ∧
    ned = jane ∧
    tom = 2 * kate ∧
    bill = 3 * ann ∧
    jack = 4 * mary ∧
    ann + mary + jane + ned + kate + tom + bill + jack = 32 :=
by {
    sorry
}

end apple_allocation_proof_l2072_207229


namespace connectivity_within_square_l2072_207268

theorem connectivity_within_square (side_length : ℝ) (highway1 highway2 : ℝ) 
  (A1 A2 A3 A4 : ℝ → ℝ → Prop) : 
  side_length = 10 → 
  highway1 ≠ highway2 → 
  (∀ x y, (0 ≤ x ∧ x ≤ side_length ∧ 0 ≤ y ∧ y ≤ side_length) → 
    (A1 x y ∨ A2 x y ∨ A3 x y ∨ A4 x y)) →
  ∃ (road_length : ℝ), road_length ≤ 25 := 
sorry

end connectivity_within_square_l2072_207268


namespace min_value_of_diff_squares_l2072_207267

noncomputable def C (x y z : ℝ) : ℝ := (Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12))
noncomputable def D (x y z : ℝ) : ℝ := (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))

theorem min_value_of_diff_squares (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  ∃ minimum_value, minimum_value = 36 ∧ ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → (C x y z)^2 - (D x y z)^2 ≥ minimum_value :=
sorry

end min_value_of_diff_squares_l2072_207267


namespace sufficient_but_not_necessary_condition_l2072_207224

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.tan (ω * x + φ)
def P (f : ℝ → ℝ) : Prop := f 0 = 0
def Q (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem sufficient_but_not_necessary_condition (ω : ℝ) (φ : ℝ) (hω : ω > 0) :
  (P (f ω φ) → Q (f ω φ)) ∧ ¬(Q (f ω φ) → P (f ω φ)) := by
  sorry

end sufficient_but_not_necessary_condition_l2072_207224


namespace total_profit_at_100_max_profit_price_l2072_207288

noncomputable def sales_volume (x : ℝ) : ℝ := 15 - 0.1 * x
noncomputable def floating_price (S : ℝ) : ℝ := 10 / S
noncomputable def supply_price (x : ℝ) : ℝ := 30 + floating_price (sales_volume x)
noncomputable def profit_per_set (x : ℝ) : ℝ := x - supply_price x
noncomputable def total_profit (x : ℝ) : ℝ := profit_per_set x * sales_volume x

-- Theorem 1: Total profit when each set is priced at 100 yuan is 340 ten thousand yuan
theorem total_profit_at_100 : total_profit 100 = 340 := by
  sorry

-- Theorem 2: The price per set that maximizes profit per set is 140 yuan
theorem max_profit_price : ∃ x, profit_per_set x = 100 ∧ x = 140 := by
  sorry

end total_profit_at_100_max_profit_price_l2072_207288


namespace min_value_of_fraction_l2072_207284

noncomputable def problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : ℝ :=
  1/a + 2/b

theorem min_value_of_fraction (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  problem_statement a b h1 h2 h3 = 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_of_fraction_l2072_207284


namespace determinant_transformation_l2072_207206

theorem determinant_transformation 
  (p q r s : ℝ)
  (h : Matrix.det ![![p, q], ![r, s]] = 6) :
  Matrix.det ![![p, 9 * p + 4 * q], ![r, 9 * r + 4 * s]] = 24 := 
sorry

end determinant_transformation_l2072_207206


namespace projectile_first_reaches_70_feet_l2072_207205

theorem projectile_first_reaches_70_feet :
  ∃ t : ℝ, t = 7/4 ∧ 0 < t ∧ ∀ s : ℝ, s < t → -16 * s^2 + 80 * s < 70 :=
by 
  sorry

end projectile_first_reaches_70_feet_l2072_207205


namespace mark_parking_tickets_eq_l2072_207248

def total_tickets : ℕ := 24
def sarah_speeding_tickets : ℕ := 6
def mark_speeding_tickets : ℕ := 6
def sarah_parking_tickets (S : ℕ) := S
def mark_parking_tickets (S : ℕ) := 2 * S
def total_traffic_tickets (S : ℕ) := S + 2 * S + sarah_speeding_tickets + mark_speeding_tickets

theorem mark_parking_tickets_eq (S : ℕ) (h1 : total_traffic_tickets S = total_tickets)
  (h2 : sarah_speeding_tickets = 6) (h3 : mark_speeding_tickets = 6) :
  mark_parking_tickets S = 8 :=
sorry

end mark_parking_tickets_eq_l2072_207248


namespace number_of_paths_l2072_207272

open Nat

def f : ℕ → ℕ → ℕ
| 0, 0 => 1
| x, 0 => 1
| 0, y => 1
| (x + 1), (y + 1) => f x (y + 1) + f (x + 1) y

theorem number_of_paths (n : ℕ) : f n 2 = (n^2 + 3 * n + 2) / 2 := by sorry

end number_of_paths_l2072_207272


namespace largest_sum_is_5_over_6_l2072_207235

def sum_1 := (1/3) + (1/7)
def sum_2 := (1/3) + (1/8)
def sum_3 := (1/3) + (1/2)
def sum_4 := (1/3) + (1/9)
def sum_5 := (1/3) + (1/4)

theorem largest_sum_is_5_over_6 : (sum_3 = 5/6) ∧ ((sum_3 > sum_1) ∧ (sum_3 > sum_2) ∧ (sum_3 > sum_4) ∧ (sum_3 > sum_5)) :=
by
  sorry

end largest_sum_is_5_over_6_l2072_207235


namespace inequality_proof_equality_case_l2072_207276

variables (x y z : ℝ)
  
theorem inequality_proof 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x + y + z ≥ 3) : 
  (1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) ≤ 1 := 
sorry

theorem equality_case 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x + y + z ≥ 3) 
  (h_eq : (1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) = 1) :
  x = 1 ∧ y = 1 ∧ z = 1 := 
sorry

end inequality_proof_equality_case_l2072_207276


namespace problem_statement_l2072_207210

theorem problem_statement (x y z : ℝ) (h1 : x = 2) (h2 : y = -1) (h3 : z = 3) :
  x^2 + y^2 + z^2 + 2*x*z = 26 :=
by
  rw [h1, h2, h3]
  norm_num

end problem_statement_l2072_207210


namespace count_valid_prime_pairs_l2072_207291

theorem count_valid_prime_pairs (x y : ℕ) (h₁ : Prime x) (h₂ : Prime y) (h₃ : x ≠ y) (h₄ : (621 * x * y) % (x + y) = 0) : 
  ∃ p, p = 6 := by
  sorry

end count_valid_prime_pairs_l2072_207291


namespace minimum_cuts_for_polygons_l2072_207269

theorem minimum_cuts_for_polygons (initial_pieces desired_pieces : ℕ) (sides : ℕ)
    (h_initial_pieces : initial_pieces = 1) (h_desired_pieces : desired_pieces = 100)
    (h_sides : sides = 20) :
    ∃ (cuts : ℕ), cuts = 1699 ∧
    (∀ current_pieces, current_pieces < desired_pieces → current_pieces + cuts ≥ desired_pieces) :=
by
    sorry

end minimum_cuts_for_polygons_l2072_207269


namespace boys_in_art_class_l2072_207227

noncomputable def number_of_boys (ratio_girls_to_boys : ℕ × ℕ) (total_students : ℕ) : ℕ :=
  let (g, b) := ratio_girls_to_boys
  let k := total_students / (g + b)
  b * k

theorem boys_in_art_class (h : number_of_boys (4, 3) 35 = 15) : true := 
  sorry

end boys_in_art_class_l2072_207227


namespace inequality_reciprocal_l2072_207223

theorem inequality_reciprocal (a b : ℝ) (hab : a < b) (hb : b < 0) : (1 / a) > (1 / b) :=
by
  sorry

end inequality_reciprocal_l2072_207223


namespace rationalize_fraction_l2072_207211

open BigOperators

theorem rationalize_fraction :
  (3 : ℝ) / (Real.sqrt 50 + 2) = (15 * Real.sqrt 2 - 6) / 46 :=
by
  -- Our proof intention will be inserted here.
  sorry

end rationalize_fraction_l2072_207211


namespace complementary_event_probability_l2072_207226

-- Define A and B as events such that B is the complement of A.
section
variables (A B : Prop) -- A and B are propositions representing events.
variable (P : Prop → ℝ) -- P is a function that gives the probability of an event.

-- Define the conditions for the problem.
variable (h_complementary : ∀ A B, A ∧ B = false ∧ A ∨ B = true) 
variable (h_PA : P A = 1 / 5)

-- The statement to be proved.
theorem complementary_event_probability : P B = 4 / 5 :=
by
  -- Here we would provide the proof, but for now, we use 'sorry' to bypass it.
  sorry
end

end complementary_event_probability_l2072_207226


namespace num_red_balls_l2072_207287

theorem num_red_balls (x : ℕ) (h : 4 / (4 + x) = 1 / 5) : x = 16 :=
by
  sorry

end num_red_balls_l2072_207287


namespace mary_turnips_grown_l2072_207295

variable (sally_turnips : ℕ)
variable (total_turnips : ℕ)
variable (mary_turnips : ℕ)

theorem mary_turnips_grown (h_sally : sally_turnips = 113)
                          (h_total : total_turnips = 242) :
                          mary_turnips = total_turnips - sally_turnips := by
  sorry

end mary_turnips_grown_l2072_207295


namespace total_cost_l2072_207245

theorem total_cost (a b : ℕ) : 30 * a + 20 * b = 30 * a + 20 * b :=
by
  sorry

end total_cost_l2072_207245


namespace remainder_when_x_plus_4uy_div_y_l2072_207251

theorem remainder_when_x_plus_4uy_div_y (x y u v : ℕ) (h₀: x = u * y + v) (h₁: 0 ≤ v) (h₂: v < y) : 
  ((x + 4 * u * y) % y) = v := 
by 
  sorry

end remainder_when_x_plus_4uy_div_y_l2072_207251


namespace five_x_ge_seven_y_iff_exists_abcd_l2072_207244

theorem five_x_ge_seven_y_iff_exists_abcd (x y : ℕ) :
  (5 * x ≥ 7 * y) ↔ ∃ (a b c d : ℕ), x = a + 2 * b + 3 * c + 7 * d ∧ y = b + 2 * c + 5 * d :=
by sorry

end five_x_ge_seven_y_iff_exists_abcd_l2072_207244


namespace problem_statement_l2072_207258

theorem problem_statement (x y : ℝ) (h : x * y < 0) : abs (x + y) < abs (x - y) :=
sorry

end problem_statement_l2072_207258


namespace paint_walls_l2072_207260

theorem paint_walls (d h e : ℕ) : 
  ∃ (x : ℕ), (d * d * e = 2 * h * h * x) ↔ x = (d^2 * e) / (2 * h^2) := by
  sorry

end paint_walls_l2072_207260


namespace percentage_error_in_calculated_area_l2072_207204

theorem percentage_error_in_calculated_area :
  let initial_length_error := 0.03 -- 3%
  let initial_width_error := -0.02 -- 2% deficit
  let temperature_change := 15 -- °C
  let humidity_increase := 20 -- %
  let length_error_temp_increase := (temperature_change / 5) * 0.01
  let width_error_humidity_increase := (humidity_increase / 10) * 0.005
  let total_length_error := initial_length_error + length_error_temp_increase
  let total_width_error := initial_width_error + width_error_humidity_increase
  let total_percentage_error := total_length_error + total_width_error
  total_percentage_error * 100 = 3 -- 3%
:= by
  sorry

end percentage_error_in_calculated_area_l2072_207204


namespace domestic_probability_short_haul_probability_long_haul_probability_l2072_207253

variable (P_internet_domestic P_snacks_domestic P_entertainment_domestic P_legroom_domestic : ℝ)
variable (P_internet_short_haul P_snacks_short_haul P_entertainment_short_haul P_legroom_short_haul : ℝ)
variable (P_internet_long_haul P_snacks_long_haul P_entertainment_long_haul P_legroom_long_haul : ℝ)

noncomputable def P_domestic :=
  P_internet_domestic * P_snacks_domestic * P_entertainment_domestic * P_legroom_domestic

theorem domestic_probability :
  P_domestic 0.40 0.60 0.70 0.50 = 0.084 := by
  sorry

noncomputable def P_short_haul :=
  P_internet_short_haul * P_snacks_short_haul * P_entertainment_short_haul * P_legroom_short_haul

theorem short_haul_probability :
  P_short_haul 0.50 0.75 0.55 0.60 = 0.12375 := by
  sorry

noncomputable def P_long_haul :=
  P_internet_long_haul * P_snacks_long_haul * P_entertainment_long_haul * P_legroom_long_haul

theorem long_haul_probability :
  P_long_haul 0.65 0.80 0.75 0.70 = 0.273 := by
  sorry

end domestic_probability_short_haul_probability_long_haul_probability_l2072_207253


namespace upload_time_l2072_207209

theorem upload_time (file_size upload_speed : ℕ) (h_file_size : file_size = 160) (h_upload_speed : upload_speed = 8) : file_size / upload_speed = 20 :=
by
  sorry

end upload_time_l2072_207209


namespace count_valid_outfits_l2072_207238

/-
Problem:
I have 5 shirts, 3 pairs of pants, and 5 hats. The pants come in red, green, and blue. 
The shirts and hats come in those colors, plus orange and purple. 
I refuse to wear an outfit where the shirt and the hat are the same color. 
How many choices for outfits, consisting of one shirt, one hat, and one pair of pants, do I have?
-/

def num_shirts := 5
def num_pants := 3
def num_hats := 5
def valid_outfits := 66

-- The set of colors available for shirts and hats
inductive color
| red | green | blue | orange | purple

-- Conditions and properties translated into Lean
def pants_colors : List color := [color.red, color.green, color.blue]
def shirt_hat_colors : List color := [color.red, color.green, color.blue, color.orange, color.purple]

theorem count_valid_outfits (h1 : num_shirts = 5) 
                            (h2 : num_pants = 3) 
                            (h3 : num_hats = 5) 
                            (h4 : ∀ (s : color), s ∈ shirt_hat_colors) 
                            (h5 : ∀ (p : color), p ∈ pants_colors) 
                            (h6 : ∀ (s h : color), s ≠ h) :
  valid_outfits = 66 :=
by
  sorry

end count_valid_outfits_l2072_207238


namespace kernels_needed_for_movie_night_l2072_207233

structure PopcornPreferences where
  caramel_popcorn: ℝ
  butter_popcorn: ℝ
  cheese_popcorn: ℝ
  kettle_corn_popcorn: ℝ

noncomputable def total_kernels_needed (preferences: PopcornPreferences) : ℝ :=
  (preferences.caramel_popcorn / 6) * 3 +
  (preferences.butter_popcorn / 4) * 2 +
  (preferences.cheese_popcorn / 8) * 4 +
  (preferences.kettle_corn_popcorn / 3) * 1

theorem kernels_needed_for_movie_night :
  let preferences := PopcornPreferences.mk 3 4 6 3
  total_kernels_needed preferences = 7.5 :=
sorry

end kernels_needed_for_movie_night_l2072_207233


namespace find_angle_sum_l2072_207247

theorem find_angle_sum
  {α β : ℝ}
  (hα_acute : 0 < α ∧ α < π / 2)
  (hβ_acute : 0 < β ∧ β < π / 2)
  (h_tan_α : Real.tan α = 1 / 3)
  (h_cos_β : Real.cos β = 3 / 5) :
  α + 2 * β = π - Real.arctan (13 / 9) :=
sorry

end find_angle_sum_l2072_207247


namespace find_fraction_of_original_flow_rate_l2072_207296

noncomputable def fraction_of_original_flow_rate (f : ℚ) : Prop :=
  let original_flow_rate := 5
  let reduced_flow_rate := 2
  reduced_flow_rate = f * original_flow_rate - 1

theorem find_fraction_of_original_flow_rate : ∃ (f : ℚ), fraction_of_original_flow_rate f ∧ f = 3 / 5 :=
by
  sorry

end find_fraction_of_original_flow_rate_l2072_207296


namespace debby_pictures_l2072_207237

theorem debby_pictures : 
  let zoo_pics := 24
  let museum_pics := 12
  let pics_deleted := 14
  zoo_pics + museum_pics - pics_deleted = 22 := 
by
  sorry

end debby_pictures_l2072_207237
