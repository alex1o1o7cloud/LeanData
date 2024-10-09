import Mathlib

namespace always_true_inequality_l1063_106317

theorem always_true_inequality (a b : ℝ) : a^2 + b^2 ≥ -2 * a * b :=
by
  sorry

end always_true_inequality_l1063_106317


namespace melissa_bonus_points_l1063_106383

/-- Given that Melissa scored 109 points per game and a total of 15089 points in 79 games,
    prove that she got 82 bonus points per game. -/
theorem melissa_bonus_points (points_per_game : ℕ) (total_points : ℕ) (num_games : ℕ)
  (H1 : points_per_game = 109)
  (H2 : total_points = 15089)
  (H3 : num_games = 79) : 
  (total_points - points_per_game * num_games) / num_games = 82 := by
  sorry

end melissa_bonus_points_l1063_106383


namespace number_in_circle_Y_l1063_106339

section
variables (a b c d X Y : ℕ)

theorem number_in_circle_Y :
  a + b + X = 30 ∧
  c + d + Y = 30 ∧
  a + b + c + d = 40 ∧
  X + Y + c + b = 40 ∧
  X = 9 → Y = 11 := by
  intros h
  sorry
end

end number_in_circle_Y_l1063_106339


namespace trapezoid_base_ratio_l1063_106311

-- Define the context of the problem
variables (AB CD : ℝ) (h : AB < CD)

-- Define the main theorem to be proved
theorem trapezoid_base_ratio (h : AB / CD = 1 / 2) :
  ∃ (E F G H I J : ℝ), 
    EJ - EI = FI - FH / 5 ∧ -- These points create segments that divide equally as per the conditions 
    FI - FH = GH / 5 ∧
    GH - GI = HI / 5 ∧
    HI - HJ = JI / 5 ∧
    JI - JE = EJ / 5 :=
sorry

end trapezoid_base_ratio_l1063_106311


namespace max_lessons_l1063_106309

-- Declaring noncomputable variables for the number of shirts, pairs of pants, and pairs of shoes.
noncomputable def s : ℕ := sorry
noncomputable def p : ℕ := sorry
noncomputable def b : ℕ := sorry

lemma conditions_satisfied :
  2 * (s + 1) * p * b = 2 * s * p * b + 36 ∧
  2 * s * (p + 1) * b = 2 * s * p * b + 72 ∧
  2 * s * p * (b + 1) = 2 * s * p * b + 54 ∧
  s * p * b = 27 ∧
  s * b = 36 ∧
  p * b = 18 := by
  sorry

theorem max_lessons : (2 * s * p * b) = 216 :=
by
  have h := conditions_satisfied
  sorry

end max_lessons_l1063_106309


namespace find_a_45_l1063_106368

theorem find_a_45 (a : ℕ → ℝ) 
  (h0 : a 0 = 11) 
  (h1 : a 1 = 11) 
  (h_rec : ∀ m n : ℕ, a (m + n) = (1 / 2) * (a (2 * m) + a (2 * n)) - (m - n) ^ 2) 
  : a 45 = 1991 :=
sorry

end find_a_45_l1063_106368


namespace find_selling_price_l1063_106342

variable (SP CP : ℝ)

def original_selling_price (SP CP : ℝ) : Prop :=
  0.9 * SP = CP + 0.08 * CP

theorem find_selling_price (h1 : CP = 17500)
  (h2 : original_selling_price SP CP) : SP = 21000 :=
by
  sorry

end find_selling_price_l1063_106342


namespace Jamie_earns_10_per_hour_l1063_106357

noncomputable def JamieHourlyRate (days_per_week : ℕ) (hours_per_day : ℕ) (weeks : ℕ) (total_earnings : ℕ) : ℕ :=
  let total_hours := days_per_week * hours_per_day * weeks
  total_earnings / total_hours

theorem Jamie_earns_10_per_hour :
  JamieHourlyRate 2 3 6 360 = 10 := by
  sorry

end Jamie_earns_10_per_hour_l1063_106357


namespace neither_drinkers_eq_nine_l1063_106387

-- Define the number of businessmen at the conference
def total_businessmen : Nat := 30

-- Define the number of businessmen who drank coffee
def coffee_drinkers : Nat := 15

-- Define the number of businessmen who drank tea
def tea_drinkers : Nat := 13

-- Define the number of businessmen who drank both coffee and tea
def both_drinkers : Nat := 7

-- Prove the number of businessmen who drank neither coffee nor tea
theorem neither_drinkers_eq_nine : 
  total_businessmen - ((coffee_drinkers + tea_drinkers) - both_drinkers) = 9 := 
by
  sorry

end neither_drinkers_eq_nine_l1063_106387


namespace fraction_value_eq_l1063_106347

theorem fraction_value_eq : (5 * 8) / 10 = 4 := 
by 
  sorry

end fraction_value_eq_l1063_106347


namespace total_canoes_built_l1063_106343

theorem total_canoes_built (boats_jan : ℕ) (h : boats_jan = 5)
    (boats_feb : ℕ) (h1 : boats_feb = boats_jan * 3)
    (boats_mar : ℕ) (h2 : boats_mar = boats_feb * 3)
    (boats_apr : ℕ) (h3 : boats_apr = boats_mar * 3) :
  boats_jan + boats_feb + boats_mar + boats_apr = 200 :=
sorry

end total_canoes_built_l1063_106343


namespace gcd_of_2535_5929_11629_l1063_106312

theorem gcd_of_2535_5929_11629 : Nat.gcd (Nat.gcd 2535 5929) 11629 = 1 := by
  sorry

end gcd_of_2535_5929_11629_l1063_106312


namespace jasmine_percent_after_addition_l1063_106379

-- Variables definition based on the problem
def original_volume : ℕ := 90
def original_jasmine_percent : ℚ := 0.05
def added_jasmine : ℕ := 8
def added_water : ℕ := 2

-- Total jasmine amount calculation in original solution
def original_jasmine_amount : ℚ := original_jasmine_percent * original_volume

-- New total jasmine amount after addition
def new_jasmine_amount : ℚ := original_jasmine_amount + added_jasmine

-- New total volume calculation after addition
def new_total_volume : ℕ := original_volume + added_jasmine + added_water

-- New jasmine percent in the solution
def new_jasmine_percent : ℚ := (new_jasmine_amount / new_total_volume) * 100

-- The proof statement
theorem jasmine_percent_after_addition : new_jasmine_percent = 12.5 :=
by
  sorry

end jasmine_percent_after_addition_l1063_106379


namespace john_money_left_l1063_106382

theorem john_money_left 
  (start_amount : ℝ := 100) 
  (price_roast : ℝ := 17)
  (price_vegetables : ℝ := 11)
  (price_wine : ℝ := 12)
  (price_dessert : ℝ := 8)
  (price_bread : ℝ := 4)
  (price_milk : ℝ := 2)
  (discount_rate : ℝ := 0.15)
  (tax_rate : ℝ := 0.05)
  (total_cost := price_roast + price_vegetables + price_wine + price_dessert + price_bread + price_milk)
  (discount_amount := discount_rate * total_cost)
  (discounted_total := total_cost - discount_amount)
  (tax_amount := tax_rate * discounted_total)
  (final_amount := discounted_total + tax_amount)
  : start_amount - final_amount = 51.80 := sorry

end john_money_left_l1063_106382


namespace problem1_problem2_l1063_106355

-- Definitions of the polynomials A and B
def A (x y : ℝ) := x^2 + x * y + 3 * y
def B (x y : ℝ) := x^2 - x * y

-- Problem 1 Statement: 
theorem problem1 (x y : ℝ) (h : (x - 2)^2 + |y + 5| = 0) : 2 * (A x y) - (B x y) = -56 := by
  sorry

-- Problem 2 Statement:
theorem problem2 (x : ℝ) (h : ∀ y, 2 * (A x y) - (B x y) = 0) : x = -2 := by
  sorry

end problem1_problem2_l1063_106355


namespace isabel_earned_l1063_106329

theorem isabel_earned :
  let bead_necklace_price := 4
  let gemstone_necklace_price := 8
  let bead_necklace_count := 3
  let gemstone_necklace_count := 3
  let sales_tax_rate := 0.05
  let discount_rate := 0.10

  let total_cost_before_tax := bead_necklace_count * bead_necklace_price + gemstone_necklace_count * gemstone_necklace_price
  let sales_tax := total_cost_before_tax * sales_tax_rate
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  let discount := total_cost_after_tax * discount_rate
  let final_amount_earned := total_cost_after_tax - discount

  final_amount_earned = 34.02 :=
by {
  sorry
}

end isabel_earned_l1063_106329


namespace smallest_portion_proof_l1063_106386

theorem smallest_portion_proof :
  ∃ (a d : ℚ), 5 * a = 100 ∧ 3 * (a + d) = 2 * d + 7 * (a - 2 * d) ∧ a - 2 * d = 5 / 3 :=
by
  sorry

end smallest_portion_proof_l1063_106386


namespace government_subsidy_per_hour_l1063_106364

-- Given conditions:
def cost_first_employee : ℕ := 20
def cost_second_employee : ℕ := 22
def hours_per_week : ℕ := 40
def weekly_savings : ℕ := 160

-- To prove:
theorem government_subsidy_per_hour (S : ℕ) : S = 2 :=
by
  -- Proof steps go here.
  sorry

end government_subsidy_per_hour_l1063_106364


namespace mandarin_ducks_total_l1063_106349

theorem mandarin_ducks_total : (3 * 2) = 6 := by
  sorry

end mandarin_ducks_total_l1063_106349


namespace isosceles_triangle_largest_angle_l1063_106360

theorem isosceles_triangle_largest_angle (A B C : Type) (α β γ : ℝ)
  (h_iso : α = β) (h_angles : α = 50) (triangle: α + β + γ = 180) : γ = 80 :=
sorry

end isosceles_triangle_largest_angle_l1063_106360


namespace dimes_max_diff_l1063_106326

-- Definitions and conditions
def num_coins (a b c : ℕ) : Prop := a + b + c = 120
def coin_values (a b c : ℕ) : Prop := 5 * a + 10 * b + 50 * c = 1050
def dimes_difference (a1 a2 b1 b2 c1 c2 : ℕ) : Prop := num_coins a1 b1 c1 ∧ num_coins a2 b2 c2 ∧ coin_values a1 b1 c1 ∧ coin_values a2 b2 c2 ∧ a1 = a2 ∧ c1 = c2

-- Theorem statement
theorem dimes_max_diff : ∃ (a b1 b2 c : ℕ), dimes_difference a a b1 b2 c c ∧ b1 - b2 = 90 :=
by sorry

end dimes_max_diff_l1063_106326


namespace split_tips_evenly_l1063_106305

theorem split_tips_evenly :
  let julie_cost := 10
  let letitia_cost := 20
  let anton_cost := 30
  let total_cost := julie_cost + letitia_cost + anton_cost
  let tip_rate := 0.2
  let total_tip := total_cost * tip_rate
  let tip_per_person := total_tip / 3
  tip_per_person = 4 := by
  sorry

end split_tips_evenly_l1063_106305


namespace toilet_paper_squares_per_roll_l1063_106351

theorem toilet_paper_squares_per_roll
  (trips_per_day : ℕ)
  (squares_per_trip : ℕ)
  (num_rolls : ℕ)
  (supply_days : ℕ)
  (total_squares : ℕ)
  (squares_per_roll : ℕ)
  (h1 : trips_per_day = 3)
  (h2 : squares_per_trip = 5)
  (h3 : num_rolls = 1000)
  (h4 : supply_days = 20000)
  (h5 : total_squares = trips_per_day * squares_per_trip * supply_days)
  (h6 : squares_per_roll = total_squares / num_rolls) :
  squares_per_roll = 300 :=
by sorry

end toilet_paper_squares_per_roll_l1063_106351


namespace general_term_a_general_term_b_l1063_106340

def arithmetic_sequence (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) :=
∀ n, a_n n = n ∧ S_n n = (n^2 + n) / 2

def sequence_b (b_n : ℕ → ℝ) (T_n : ℕ → ℝ) :=
  (b_n 1 = 1/2) ∧
  (∀ n, b_n (n+1) = (n+1) / n * b_n n) ∧ 
  (∀ n, b_n n = n / 2) ∧ 
  (∀ n, T_n n = (n^2 + n) / 4) ∧ 
  (∀ m, m = 1 → T_n m = 1/2)

-- Arithmetic sequence {a_n}
theorem general_term_a (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 2 = 2) (h2 : S 5 = 15) :
  arithmetic_sequence a S := sorry

-- Sequence {b_n}
theorem general_term_b (b : ℕ → ℝ) (T : ℕ → ℝ) (h1 : b 1 = 1/2) (h2 : ∀ n, b (n+1) = (n+1) / n * b n) :
  sequence_b b T := sorry

end general_term_a_general_term_b_l1063_106340


namespace lowest_price_for_16_oz_butter_l1063_106365

-- Define the constants
def price_single_16_oz_package : ℝ := 7
def price_8_oz_package : ℝ := 4
def price_4_oz_package : ℝ := 2
def discount_4_oz_package : ℝ := 0.5

-- Calculate the discounted price for a 4 oz package
def discounted_price_4_oz_package : ℝ := price_4_oz_package * discount_4_oz_package

-- Calculate the total price for two discounted 4 oz packages
def total_price_two_discounted_4_oz_packages : ℝ := 2 * discounted_price_4_oz_package

-- Calculate the total price using the 8 oz package and two discounted 4 oz packages
def total_price_using_coupon : ℝ := price_8_oz_package + total_price_two_discounted_4_oz_packages

-- State the property to prove
theorem lowest_price_for_16_oz_butter :
  min price_single_16_oz_package total_price_using_coupon = 6 :=
sorry

end lowest_price_for_16_oz_butter_l1063_106365


namespace quadrilateral_EFGH_l1063_106377

variable {EF FG GH HE EH : ℤ}

theorem quadrilateral_EFGH (h1 : EF = 6) (h2 : FG = 18) (h3 : GH = 6) (h4 : HE = 10) (h5 : 12 < EH) (h6 : EH < 24) : EH = 12 := 
sorry

end quadrilateral_EFGH_l1063_106377


namespace negative_option_is_B_l1063_106353

-- Define the options as constants
def optionA : ℤ := -( -2 )
def optionB : ℤ := (-1) ^ 2023
def optionC : ℤ := |(-1) ^ 2|
def optionD : ℤ := (-5) ^ 2

-- Prove that the negative number among the options is optionB
theorem negative_option_is_B : optionB = -1 := 
by
  rw [optionB]
  sorry

end negative_option_is_B_l1063_106353


namespace speedster_convertibles_count_l1063_106346

-- Definitions of conditions
def total_inventory (T : ℕ) : Prop := (T / 3) = 60
def number_of_speedsters (T S : ℕ) : Prop := S = (2 / 3) * T
def number_of_convertibles (S C : ℕ) : Prop := C = (4 / 5) * S

-- Primary statement to prove
theorem speedster_convertibles_count (T S C : ℕ) (h1 : total_inventory T) (h2 : number_of_speedsters T S) (h3 : number_of_convertibles S C) : C = 96 :=
by
  -- Conditions and given values are defined
  sorry

end speedster_convertibles_count_l1063_106346


namespace derek_walk_time_l1063_106328

theorem derek_walk_time (x : ℕ) :
  (∀ y : ℕ, (y = 9) → (∀ d₁ d₂ : ℕ, (d₁ = 20 ∧ d₂ = 60) →
    (20 * x = d₁ * y + d₂))) → x = 12 :=
by
  intro h
  sorry

end derek_walk_time_l1063_106328


namespace B_visits_A_l1063_106306

/-- Students A, B, and C were surveyed on whether they have visited cities A, B, and C -/
def student_visits_city (student : Type) (city : Type) : Prop := sorry -- assume there's a definition

variables (A_student B_student C_student : Type) (city_A city_B city_C : Type)

variables 
  -- A's statements
  (A_visits_more_than_B : student_visits_city A_student city_A → ¬ student_visits_city A_student city_B → ∃ city, student_visits_city B_student city ∧ ¬ student_visits_city A_student city)
  (A_not_visit_B : ¬ student_visits_city A_student city_B)
  -- B's statement
  (B_not_visit_C : ¬ student_visits_city B_student city_C)
  -- C's statement
  (all_three_same_city : student_visits_city A_student city_A → student_visits_city B_student city_A → student_visits_city C_student city_A)

theorem B_visits_A : student_visits_city B_student city_A :=
by
  sorry

end B_visits_A_l1063_106306


namespace probability_A_given_B_l1063_106314

namespace ProbabilityProof

def total_parts : ℕ := 100
def A_parts_produced : ℕ := 0
def A_parts_qualified : ℕ := 35
def B_parts_produced : ℕ := 60
def B_parts_qualified : ℕ := 50

def event_A (x : ℕ) : Prop := x ≤ B_parts_qualified + A_parts_qualified
def event_B (x : ℕ) : Prop := x ≤ A_parts_produced

-- Formalizing the probability condition P(A | B) = 7/8, logically this should be revised with practical events.
theorem probability_A_given_B : (event_B x → event_A x) := sorry

end ProbabilityProof

end probability_A_given_B_l1063_106314


namespace false_proposition_l1063_106375

-- Definitions of the conditions
def p1 := ∃ x0 : ℝ, x0^2 - 2*x0 + 1 ≤ 0
def p2 := ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - 1 ≥ 0

-- Statement to prove
theorem false_proposition : ¬ (¬ p1 ∧ ¬ p2) :=
by sorry

end false_proposition_l1063_106375


namespace difference_between_second_and_third_levels_l1063_106332

def total_parking_spots : ℕ := 400
def first_level_open_spots : ℕ := 58
def second_level_open_spots : ℕ := first_level_open_spots + 2
def fourth_level_open_spots : ℕ := 31
def total_full_spots : ℕ := 186

def total_open_spots : ℕ := total_parking_spots - total_full_spots

def third_level_open_spots : ℕ := 
  total_open_spots - (first_level_open_spots + second_level_open_spots + fourth_level_open_spots)

def difference_open_spots : ℕ := third_level_open_spots - second_level_open_spots

theorem difference_between_second_and_third_levels : difference_open_spots = 5 :=
sorry

end difference_between_second_and_third_levels_l1063_106332


namespace cubic_expression_identity_l1063_106336

theorem cubic_expression_identity (x : ℝ) (hx : x + 1/x = 8) : 
  x^3 + 1/x^3 = 332 :=
sorry

end cubic_expression_identity_l1063_106336


namespace ducks_in_marsh_l1063_106395

theorem ducks_in_marsh (total_birds geese : ℕ) (h1 : total_birds = 95) (h2 : geese = 58) : total_birds - geese = 37 := by
  sorry

end ducks_in_marsh_l1063_106395


namespace worker_saves_one_third_l1063_106393

variable {P : ℝ} 
variable {f : ℝ}

theorem worker_saves_one_third (h : P ≠ 0) (h_eq : 12 * f * P = 6 * (1 - f) * P) : 
  f = 1 / 3 :=
sorry

end worker_saves_one_third_l1063_106393


namespace angle_A_value_l1063_106316

/-- 
In triangle ABC, the sides opposite to angles A, B, C are a, b, and c respectively.
Given:
  - C = π / 3,
  - b = √6,
  - c = 3,
Prove that A = 5π / 12.
-/
theorem angle_A_value (a b c : ℝ) (A B C : ℝ) (hC : C = Real.pi / 3) (hb : b = Real.sqrt 6) (hc : c = 3) :
  A = 5 * Real.pi / 12 :=
sorry

end angle_A_value_l1063_106316


namespace partial_fraction_sum_zero_l1063_106327

variable {A B C D E : ℝ}
variable {x : ℝ}

theorem partial_fraction_sum_zero (h : 
  (1:ℝ) / ((x-1)*x*(x+1)*(x+2)*(x+3)) = 
  A / (x-1) + B / x + C / (x+1) + D / (x+2) + E / (x+3)) : 
  A + B + C + D + E = 0 :=
by sorry

end partial_fraction_sum_zero_l1063_106327


namespace total_blood_cells_correct_l1063_106389

def first_sample : ℕ := 4221
def second_sample : ℕ := 3120
def total_blood_cells : ℕ := first_sample + second_sample

theorem total_blood_cells_correct : total_blood_cells = 7341 := by
  -- proof goes here
  sorry

end total_blood_cells_correct_l1063_106389


namespace x_y_differ_by_one_l1063_106354

theorem x_y_differ_by_one (x y : ℚ) (h : (1 + y) / (x - y) = x) : y = x - 1 :=
by
sorry

end x_y_differ_by_one_l1063_106354


namespace smallest_n_45_l1063_106358

def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, x = k * k

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m * m * m

theorem smallest_n_45 :
  ∃ n : ℕ, n > 0 ∧ (is_perfect_square (5 * n)) ∧ (is_perfect_cube (3 * n)) ∧ ∀ m : ℕ, (m > 0 ∧ (is_perfect_square (5 * m)) ∧ (is_perfect_cube (3 * m))) → n ≤ m :=
sorry

end smallest_n_45_l1063_106358


namespace sin_B_over_sin_A_eq_two_max_value_sin_A_sin_B_l1063_106396

-- Given conditions for the triangle ABC
variables {A B C a b c : ℝ}
axiom angle_C_eq_two_pi_over_three : C = 2 * Real.pi / 3
axiom c_squared_eq_five_a_squared_plus_ab : c^2 = 5 * a^2 + a * b

-- Proof statements
theorem sin_B_over_sin_A_eq_two (hAC: C = 2 * Real.pi / 3) (hCond: c^2 = 5 * a^2 + a * b) :
  Real.sin B / Real.sin A = 2 :=
sorry

theorem max_value_sin_A_sin_B (hAC: C = 2 * Real.pi / 3) :
  ∃ A B : ℝ, 0 < A ∧ A < Real.pi / 3 ∧ B = (Real.pi / 3 - A) ∧ Real.sin A * Real.sin B ≤ 1 / 4 :=
sorry

end sin_B_over_sin_A_eq_two_max_value_sin_A_sin_B_l1063_106396


namespace tan_420_eq_sqrt3_l1063_106361

theorem tan_420_eq_sqrt3 : Real.tan (420 * Real.pi / 180) = Real.sqrt 3 := 
by 
  -- Additional mathematical justification can go here.
  sorry

end tan_420_eq_sqrt3_l1063_106361


namespace means_imply_sum_of_squares_l1063_106310

noncomputable def arithmetic_mean (x y z : ℝ) : ℝ :=
(x + y + z) / 3

noncomputable def geometric_mean (x y z : ℝ) : ℝ :=
(x * y * z) ^ (1/3)

noncomputable def harmonic_mean (x y z : ℝ) : ℝ :=
3 / ((1/x) + (1/y) + (1/z))

theorem means_imply_sum_of_squares (x y z : ℝ) :
  arithmetic_mean x y z = 10 →
  geometric_mean x y z = 6 →
  harmonic_mean x y z = 4 →
  x^2 + y^2 + z^2 = 576 :=
by
  -- Proof is omitted for now
  exact sorry

end means_imply_sum_of_squares_l1063_106310


namespace one_fourths_in_one_eighth_l1063_106362

theorem one_fourths_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 := 
by 
  sorry

end one_fourths_in_one_eighth_l1063_106362


namespace weight_of_purple_ring_l1063_106369

noncomputable section

def orange_ring_weight : ℝ := 0.08333333333333333
def white_ring_weight : ℝ := 0.4166666666666667
def total_weight : ℝ := 0.8333333333

theorem weight_of_purple_ring :
  total_weight - orange_ring_weight - white_ring_weight = 0.3333333333 :=
by
  -- We'll place the statement here, leave out the proof for skipping.
  sorry

end weight_of_purple_ring_l1063_106369


namespace max_value_expression_l1063_106385

theorem max_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + 3 * b = 5) : 
  (∀ x y : ℝ, x = 2 * a + 2 → y = 3 * b + 1 → x * y ≤ 16) := by
  sorry

end max_value_expression_l1063_106385


namespace max_sum_of_multiplication_table_l1063_106301

theorem max_sum_of_multiplication_table :
  let numbers := [3, 5, 7, 11, 17, 19]
  let repeated_num := 19
  ∃ d e f, d ∈ numbers ∧ e ∈ numbers ∧ f ∈ numbers ∧ d ≠ e ∧ e ≠ f ∧ d ≠ f ∧
  3 * repeated_num * (d + e + f) = 1995 := 
by {
  sorry
}

end max_sum_of_multiplication_table_l1063_106301


namespace mary_time_l1063_106388

-- Define the main entities for the problem
variables (mary_days : ℕ) (rosy_days : ℕ)
variable (rosy_efficiency_factor : ℝ) -- Rosy's efficiency factor compared to Mary

-- Given conditions
def rosy_efficient := rosy_efficiency_factor = 1.4
def rosy_time := rosy_days = 20

-- Problem Statement
theorem mary_time (h1 : rosy_efficient rosy_efficiency_factor) (h2 : rosy_time rosy_days) : mary_days = 28 :=
by
  sorry

end mary_time_l1063_106388


namespace count_elements_in_A_l1063_106391

variables (a b : ℕ)

def condition1 : Prop := a = 3 * b / 2
def condition2 : Prop := a + b - 1200 = 4500

theorem count_elements_in_A (h1 : condition1 a b) (h2 : condition2 a b) : a = 3420 :=
by sorry

end count_elements_in_A_l1063_106391


namespace dan_marbles_l1063_106390

theorem dan_marbles (original_marbles : ℕ) (given_marbles : ℕ) (remaining_marbles : ℕ) : 
  original_marbles = 64 ∧ given_marbles = 14 → remaining_marbles = 50 := 
by 
  sorry

end dan_marbles_l1063_106390


namespace parabola_tangent_xsum_l1063_106322

theorem parabola_tangent_xsum
  (p : ℝ) (hp : p > 0) 
  (X_A X_B X_M : ℝ) 
  (hxM_line : ∃ y, y = -2 * p ∧ y = -2 * p)
  (hxA_tangent : ∃ y, y = (X_A / p) * (X_A - X_M) - 2 * p)
  (hxB_tangent : ∃ y, y = (X_B / p) * (X_B - X_M) - 2 * p) :
  2 * X_M = X_A + X_B :=
by
  sorry

end parabola_tangent_xsum_l1063_106322


namespace find_angle_E_l1063_106363

def trapezoid_angles (E H F G : ℝ) : Prop :=
  E + H = 180 ∧ E = 3 * H ∧ G = 4 * F

theorem find_angle_E (E H F G : ℝ) 
  (h1 : E + H = 180)
  (h2 : E = 3 * H)
  (h3 : G = 4 * F) : 
  E = 135 := by
    sorry

end find_angle_E_l1063_106363


namespace cost_per_pie_eq_l1063_106307

-- We define the conditions
def price_per_piece : ℝ := 4
def pieces_per_pie : ℕ := 3
def pies_per_hour : ℕ := 12
def actual_revenue : ℝ := 138

-- Lean theorem statement
theorem cost_per_pie_eq : (price_per_piece * pieces_per_pie * pies_per_hour - actual_revenue) / pies_per_hour = 0.50 := by
  -- Proof would go here
  sorry

end cost_per_pie_eq_l1063_106307


namespace binary_multiplication_binary_result_l1063_106333

-- Definitions for binary numbers
def bin_11011 : ℕ := 27 -- 11011 in binary is 27 in decimal
def bin_101 : ℕ := 5 -- 101 in binary is 5 in decimal

-- Theorem statement to prove the product of two binary numbers
theorem binary_multiplication : (bin_11011 * bin_101) = 135 := by
  sorry

-- Convert the result back to binary, expected to be 10000111
theorem binary_result : 135 = 8 * 16 + 7 := by
  sorry

end binary_multiplication_binary_result_l1063_106333


namespace russian_players_pairing_probability_l1063_106359

theorem russian_players_pairing_probability :
  let total_players := 10
  let russian_players := 4
  (russian_players * (russian_players - 1)) / (total_players * (total_players - 1)) * 
  ((russian_players - 2) * (russian_players - 3)) / ((total_players - 2) * (total_players - 3)) = 1 / 21 :=
by
  sorry

end russian_players_pairing_probability_l1063_106359


namespace units_digit_div_product_l1063_106392

theorem units_digit_div_product :
  (30 * 31 * 32 * 33 * 34 * 35) / 14000 % 10 = 2 :=
by
  sorry

end units_digit_div_product_l1063_106392


namespace number_of_apples_l1063_106321

theorem number_of_apples (C : ℝ) (A : ℕ) (total_cost : ℝ) (price_diff : ℝ) (num_oranges : ℕ)
  (h_price : C = 0.26)
  (h_price_diff : price_diff = 0.28)
  (h_num_oranges : num_oranges = 7)
  (h_total_cost : total_cost = 4.56) :
  A * C + num_oranges * (C + price_diff) = total_cost → A = 3 := 
by
  sorry

end number_of_apples_l1063_106321


namespace difference_in_price_l1063_106399

noncomputable def total_cost : ℝ := 70.93
noncomputable def pants_price : ℝ := 34.00

theorem difference_in_price (total_cost pants_price : ℝ) (h_total : total_cost = 70.93) (h_pants : pants_price = 34.00) :
  (total_cost - pants_price) - pants_price = 2.93 :=
by
  sorry

end difference_in_price_l1063_106399


namespace total_jellybeans_needed_l1063_106384

def large_glass_jellybeans : ℕ := 50
def small_glass_jellybeans : ℕ := large_glass_jellybeans / 2
def num_large_glasses : ℕ := 5
def num_small_glasses : ℕ := 3

theorem total_jellybeans_needed : 
  (num_large_glasses * large_glass_jellybeans) + (num_small_glasses * small_glass_jellybeans) = 325 := 
by
  sorry

end total_jellybeans_needed_l1063_106384


namespace product_of_y_coordinates_on_line_l1063_106320

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem product_of_y_coordinates_on_line (y1 y2 : ℝ) (h1 : distance (4, -1) (-2, y1) = 8) (h2 : distance (4, -1) (-2, y2) = 8) :
  y1 * y2 = -27 :=
sorry

end product_of_y_coordinates_on_line_l1063_106320


namespace twelve_integers_divisible_by_eleven_l1063_106334

theorem twelve_integers_divisible_by_eleven (a : Fin 12 → ℤ) : 
  ∃ (i j : Fin 12), i ≠ j ∧ 11 ∣ (a i - a j) :=
by
  sorry

end twelve_integers_divisible_by_eleven_l1063_106334


namespace angle_B_of_right_triangle_l1063_106308

theorem angle_B_of_right_triangle (B C : ℝ) (hA : A = 90) (hC : C = 3 * B) (h_sum : A + B + C = 180) : B = 22.5 :=
sorry

end angle_B_of_right_triangle_l1063_106308


namespace toys_gained_l1063_106325

theorem toys_gained
  (sp : ℕ) -- selling price of 18 toys
  (cp_per_toy : ℕ) -- cost price per toy
  (sp_val : sp = 27300) -- given selling price value
  (cp_per_val : cp_per_toy = 1300) -- given cost price per toy value
  : (sp - 18 * cp_per_toy) / cp_per_toy = 3 := by
  -- Conditions of the problem are stated
  -- Proof is omitted with 'sorry'
  sorry

end toys_gained_l1063_106325


namespace order_of_trig_values_l1063_106380

noncomputable def a := Real.sin (Real.sin (2008 * Real.pi / 180))
noncomputable def b := Real.sin (Real.cos (2008 * Real.pi / 180))
noncomputable def c := Real.cos (Real.sin (2008 * Real.pi / 180))
noncomputable def d := Real.cos (Real.cos (2008 * Real.pi / 180))

theorem order_of_trig_values : b < a ∧ a < d ∧ d < c :=
by
  sorry

end order_of_trig_values_l1063_106380


namespace monthly_salary_is_correct_l1063_106350

noncomputable def man's_salary : ℝ :=
  let S : ℝ := 6500
  S

theorem monthly_salary_is_correct (S : ℝ) (h1 : S * 0.20 = S * 0.20) (h2 : S * 0.80 * 1.20 + 260 = S):
  S = man's_salary :=
by sorry

end monthly_salary_is_correct_l1063_106350


namespace real_number_value_of_m_pure_imaginary_value_of_m_l1063_106370

def is_real (z : ℂ) : Prop := z.im = 0
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem real_number_value_of_m (m : ℝ) : 
  is_real ((m^2 + 2 * m - 8) + (m^2 - 2 * m) * I) ↔ (m = 0 ∨ m = 2) := 
by sorry

theorem pure_imaginary_value_of_m (m : ℝ) : 
  is_pure_imaginary ((m^2 + 2 * m - 8) + (m^2 - 2 * m) * I) ↔ (m = -4) := 
by sorry

end real_number_value_of_m_pure_imaginary_value_of_m_l1063_106370


namespace johns_outfit_cost_l1063_106338

theorem johns_outfit_cost (pants_cost shirt_cost outfit_cost : ℝ)
    (h_pants : pants_cost = 50)
    (h_shirt : shirt_cost = pants_cost + 0.6 * pants_cost)
    (h_outfit : outfit_cost = pants_cost + shirt_cost) :
    outfit_cost = 130 :=
by
  sorry

end johns_outfit_cost_l1063_106338


namespace intersection_A_B_l1063_106371

def A : Set ℤ := {x | abs x < 2}
def B : Set ℤ := {-1, 0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {-1, 0, 1} := by
  sorry

end intersection_A_B_l1063_106371


namespace sets_equivalence_l1063_106378

theorem sets_equivalence :
  (∀ M N, (M = {(3, 2)} ∧ N = {(2, 3)} → M ≠ N) ∧
          (M = {4, 5} ∧ N = {5, 4} → M = N) ∧
          (M = {1, 2} ∧ N = {(1, 2)} → M ≠ N) ∧
          (M = {(x, y) | x + y = 1} ∧ N = {y | ∃ x, x + y = 1} → M ≠ N)) :=
by sorry

end sets_equivalence_l1063_106378


namespace length_of_segment_AB_l1063_106381

variables (h : ℝ) (AB CD : ℝ)

-- Defining the conditions
def condition_one : Prop := (AB / CD = 5 / 2)
def condition_two : Prop := (AB + CD = 280)

-- The theorem to prove
theorem length_of_segment_AB (h : ℝ) (AB CD : ℝ) :
  condition_one AB CD ∧ condition_two AB CD → AB = 200 :=
by
  sorry

end length_of_segment_AB_l1063_106381


namespace probability_complement_B_probability_union_A_B_l1063_106374

variable (Ω : Type) [MeasurableSpace Ω] {P : ProbabilityMeasure Ω}
variable (A B : Set Ω)

theorem probability_complement_B
  (hB : P B = 1 / 3) : P Bᶜ = 2 / 3 :=
by
  sorry

theorem probability_union_A_B
  (hA : P A = 1 / 2) (hB : P B = 1 / 3) : P (A ∪ B) ≤ 5 / 6 :=
by
  sorry

end probability_complement_B_probability_union_A_B_l1063_106374


namespace friends_bought_color_box_l1063_106300

variable (total_pencils : ℕ) (pencils_per_box : ℕ) (chloe_pencils : ℕ)

theorem friends_bought_color_box : 
  (total_pencils = 42) → 
  (pencils_per_box = 7) → 
  (chloe_pencils = pencils_per_box) → 
  (total_pencils - chloe_pencils) / pencils_per_box = 5 := 
by 
  intros ht hb hc
  sorry

end friends_bought_color_box_l1063_106300


namespace three_digit_sum_permutations_l1063_106344

theorem three_digit_sum_permutations (a b c : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9) (h₃ : 1 ≤ b) (h₄ : b ≤ 9) (h₅ : 1 ≤ c) (h₆ : c ≤ 9)
  (h₇ : n = 100 * a + 10 * b + c)
  (h₈ : 222 * (a + b + c) - n = 1990) :
  n = 452 :=
by
  sorry

end three_digit_sum_permutations_l1063_106344


namespace double_x_value_l1063_106318

theorem double_x_value (x : ℝ) (h : x / 2 = 32) : 2 * x = 128 := by
  sorry

end double_x_value_l1063_106318


namespace find_ordered_pair_l1063_106341

open Polynomial

theorem find_ordered_pair (a b : ℝ) :
  (∀ x : ℝ, (((x^3 + a * x^2 + 17 * x + 10 = 0) ∧ (x^3 + b * x^2 + 20 * x + 12 = 0)) → 
  (x = -6 ∧ y = -7))) :=
sorry

end find_ordered_pair_l1063_106341


namespace triangle_XYZ_median_inequalities_l1063_106303

theorem triangle_XYZ_median_inequalities :
  ∀ (XY XZ : ℝ), 
  (∀ (YZ : ℝ), YZ = 10 → 
  ∀ (XM : ℝ), XM = 6 → 
  ∃ (x : ℝ), x = (XY + XZ - 20)/4 → 
  ∃ (N n : ℝ), 
  N = 192 ∧ n = 92 → 
  N - n = 100) :=
by sorry

end triangle_XYZ_median_inequalities_l1063_106303


namespace find_real_numbers_l1063_106315

theorem find_real_numbers (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
sorry

end find_real_numbers_l1063_106315


namespace number_of_days_same_l1063_106304

-- Defining volumes as given in the conditions.
def volume_project1 : ℕ := 100 * 25 * 30
def volume_project2 : ℕ := 75 * 20 * 50

-- The mathematical statement we want to prove.
theorem number_of_days_same : volume_project1 = volume_project2 → ∀ d : ℕ, d > 0 → d = d :=
by
  sorry

end number_of_days_same_l1063_106304


namespace xiaolin_distance_l1063_106330

theorem xiaolin_distance (speed : ℕ) (time : ℕ) (distance : ℕ)
    (h1 : speed = 80) (h2 : time = 28) : distance = 2240 :=
by
  have h3 : distance = time * speed := by sorry
  rw [h1, h2] at h3
  exact h3

end xiaolin_distance_l1063_106330


namespace shenille_scores_points_l1063_106394

theorem shenille_scores_points :
  ∀ (x y : ℕ), (x + y = 45) → (x = 2 * y) → 
  (25/100 * x + 40/100 * y) * 3 + (40/100 * y) * 2 = 33 :=
by 
  intros x y h1 h2
  sorry

end shenille_scores_points_l1063_106394


namespace count_four_digit_numbers_without_1_or_4_l1063_106373

-- Define a function to check if a digit is allowed (i.e., not 1 or 4)
def allowed_digit (d : ℕ) : Prop := d ≠ 1 ∧ d ≠ 4

-- Function to count four-digit numbers without digits 1 or 4
def count_valid_four_digit_numbers : ℕ :=
  let valid_first_digits := [2, 3, 5, 6, 7, 8, 9]
  let valid_other_digits := [0, 2, 3, 5, 6, 7, 8, 9]
  (valid_first_digits.length) * (valid_other_digits.length ^ 3)

-- The main theorem stating that the number of valid four-digit integers is 3072
theorem count_four_digit_numbers_without_1_or_4 : count_valid_four_digit_numbers = 3072 :=
by
  sorry

end count_four_digit_numbers_without_1_or_4_l1063_106373


namespace equivalent_statements_l1063_106324

variables (P Q : Prop)

theorem equivalent_statements : (¬Q → ¬P) ∧ (¬P ∨ Q) ↔ (P → Q) :=
by
  -- Proof goes here
  sorry

end equivalent_statements_l1063_106324


namespace dimes_paid_l1063_106397

theorem dimes_paid (cost_in_dollars : ℕ) (dollars_to_dimes : ℕ) (h₁ : cost_in_dollars = 5) (h₂ : dollars_to_dimes = 10) :
  cost_in_dollars * dollars_to_dimes = 50 :=
by
  sorry

end dimes_paid_l1063_106397


namespace opening_night_ticket_price_l1063_106372

theorem opening_night_ticket_price :
  let matinee_customers := 32
  let evening_customers := 40
  let opening_night_customers := 58
  let matinee_price := 5
  let evening_price := 7
  let popcorn_price := 10
  let total_revenue := 1670
  let total_customers := matinee_customers + evening_customers + opening_night_customers
  let popcorn_customers := total_customers / 2
  let total_matinee_revenue := matinee_customers * matinee_price
  let total_evening_revenue := evening_customers * evening_price
  let total_popcorn_revenue := popcorn_customers * popcorn_price
  let known_revenue := total_matinee_revenue + total_evening_revenue + total_popcorn_revenue
  let opening_night_revenue := total_revenue - known_revenue
  let opening_night_price := opening_night_revenue / opening_night_customers
  opening_night_price = 10 := by
  sorry

end opening_night_ticket_price_l1063_106372


namespace third_neigh_uses_100_more_l1063_106398

def total_water : Nat := 1200
def first_neigh_usage : Nat := 150
def second_neigh_usage : Nat := 2 * first_neigh_usage
def fourth_neigh_remaining : Nat := 350

def third_neigh_usage := total_water - (first_neigh_usage + second_neigh_usage + fourth_neigh_remaining)
def diff_third_second := third_neigh_usage - second_neigh_usage

theorem third_neigh_uses_100_more :
  diff_third_second = 100 := by
  sorry

end third_neigh_uses_100_more_l1063_106398


namespace expected_winnings_l1063_106345

theorem expected_winnings (roll_1_2: ℝ) (roll_3_4: ℝ) (roll_5_6: ℝ) (p1_2 p3_4 p5_6: ℝ) :
    roll_1_2 = 2 →
    roll_3_4 = 4 →
    roll_5_6 = -6 →
    p1_2 = 1 / 8 →
    p3_4 = 1 / 4 →
    p5_6 = 1 / 8 →
    (2 * p1_2 + 2 * p1_2 + 4 * p3_4 + 4 * p3_4 + roll_5_6 * p5_6 + roll_5_6 * p5_6) = 1 := by
  intros
  sorry

end expected_winnings_l1063_106345


namespace kevin_birth_year_l1063_106348

theorem kevin_birth_year (year_first_amc: ℕ) (annual: ∀ n, year_first_amc + n = year_first_amc + n) (age_tenth_amc: ℕ) (year_tenth_amc: ℕ) (year_kevin_took_amc: ℕ) 
  (h_first_amc: year_first_amc = 1988) (h_age_tenth_amc: age_tenth_amc = 13) (h_tenth_amc: year_tenth_amc = year_first_amc + 9) (h_kevin_took_amc: year_kevin_took_amc = year_tenth_amc) :
  year_kevin_took_amc - age_tenth_amc = 1984 :=
by
  sorry

end kevin_birth_year_l1063_106348


namespace first_group_size_l1063_106313

theorem first_group_size
  (x : ℕ)
  (h1 : 2 * x + 22 + 16 + 14 = 68) : 
  x = 8 :=
by
  sorry

end first_group_size_l1063_106313


namespace orchestra_french_horn_players_l1063_106337

open Nat

theorem orchestra_french_horn_players :
  ∃ (french_horn_players : ℕ), 
  french_horn_players = 1 ∧
  1 + 6 + 5 + 7 + 1 + french_horn_players = 21 :=
by
  sorry

end orchestra_french_horn_players_l1063_106337


namespace quadrilateral_area_l1063_106352

theorem quadrilateral_area (a b : ℤ) (h1 : a > b) (h2 : b > 0) 
  (h3 : 2 * |a - b| * |a + b| = 32) : a + b = 8 :=
by
  sorry

end quadrilateral_area_l1063_106352


namespace simplify_expression_l1063_106319

theorem simplify_expression (c : ℤ) : (3 * c + 6 - 6 * c) / 3 = -c + 2 := by
  sorry

end simplify_expression_l1063_106319


namespace total_puppies_is_74_l1063_106366

-- Define the number of puppies adopted each week based on the given conditions
def number_of_puppies_first_week : Nat := 20
def number_of_puppies_second_week : Nat := 2 * number_of_puppies_first_week / 5
def number_of_puppies_third_week : Nat := 2 * number_of_puppies_second_week
def number_of_puppies_fourth_week : Nat := number_of_puppies_first_week + 10

-- Define the total number of puppies
def total_number_of_puppies : Nat :=
  number_of_puppies_first_week + number_of_puppies_second_week + number_of_puppies_third_week + number_of_puppies_fourth_week

-- Proof statement: Prove that the total number of puppies is 74
theorem total_puppies_is_74 : total_number_of_puppies = 74 := by
  sorry

end total_puppies_is_74_l1063_106366


namespace original_cost_of_tshirt_l1063_106331

theorem original_cost_of_tshirt
  (backpack_cost : ℕ := 10)
  (cap_cost : ℕ := 5)
  (total_spent_after_discount : ℕ := 43)
  (discount : ℕ := 2)
  (tshirt_cost_before_discount : ℕ) :
  total_spent_after_discount + discount - (backpack_cost + cap_cost) = tshirt_cost_before_discount :=
by
  sorry

end original_cost_of_tshirt_l1063_106331


namespace sum_of_first_five_integers_l1063_106335

theorem sum_of_first_five_integers : (1 + 2 + 3 + 4 + 5) = 15 := 
by 
  sorry

end sum_of_first_five_integers_l1063_106335


namespace a_squared_plus_b_squared_equals_61_l1063_106323

theorem a_squared_plus_b_squared_equals_61 (a b : ℝ) (h1 : a + b = -9) (h2 : a = 30 / b) : a^2 + b^2 = 61 :=
sorry

end a_squared_plus_b_squared_equals_61_l1063_106323


namespace find_angle_l1063_106367

theorem find_angle (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by 
  sorry

end find_angle_l1063_106367


namespace largest_angle_in_triangle_l1063_106302

theorem largest_angle_in_triangle (y : ℝ) (h : 60 + 70 + y = 180) :
    70 > 60 ∧ 70 > y :=
by {
  sorry
}

end largest_angle_in_triangle_l1063_106302


namespace sum_f_1_2021_l1063_106376

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom equation_f : ∀ x : ℝ, f (2 + x) + f (2 - x) = 0
axiom interval_f : ∀ x : ℝ, -1 ≤ x ∧ x < 0 → f x = Real.log (1 - x) / Real.log 2

theorem sum_f_1_2021 : (List.sum (List.map f (List.range' 1 2021))) = -1 := sorry

end sum_f_1_2021_l1063_106376


namespace carlos_meeting_percentage_l1063_106356

-- Definitions for the given conditions
def work_day_minutes : ℕ := 10 * 60
def first_meeting_minutes : ℕ := 80
def second_meeting_minutes : ℕ := 3 * first_meeting_minutes
def break_minutes : ℕ := 15
def total_meeting_and_break_minutes : ℕ := first_meeting_minutes + second_meeting_minutes + break_minutes

-- Statement to prove
theorem carlos_meeting_percentage : 
  (total_meeting_and_break_minutes * 100 / work_day_minutes) = 56 := 
by
  sorry

end carlos_meeting_percentage_l1063_106356
