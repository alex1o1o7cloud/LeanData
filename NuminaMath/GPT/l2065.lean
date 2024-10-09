import Mathlib

namespace ratio_of_percentage_change_l2065_206582

theorem ratio_of_percentage_change
  (P U U' : ℝ)
  (h_price_decrease : U' = 4 * U)
  : (300 / 75) = 4 := 
by
  sorry

end ratio_of_percentage_change_l2065_206582


namespace f_2015_l2065_206580

noncomputable def f (x : ℝ) : ℝ := sorry

axiom even_f : ∀ x : ℝ, f x = f (-x)

axiom periodic_f : ∀ x : ℝ, f (x - 2) = -f x

axiom f_interval : ∀ x : ℝ, (-1 ≤ x ∧ x ≤ 0) → f x = 2 ^ x

theorem f_2015 : f 2015 = 1 / 2 :=
sorry

end f_2015_l2065_206580


namespace largest_4_digit_divisible_by_98_l2065_206577

theorem largest_4_digit_divisible_by_98 :
  ∃ n, (n ≤ 9999 ∧ 9999 < n + 98) ∧ 98 ∣ n :=
sorry

end largest_4_digit_divisible_by_98_l2065_206577


namespace hyperbola_focal_point_k_l2065_206551

theorem hyperbola_focal_point_k (k : ℝ) :
  (∃ (c : ℝ), c = 2 ∧ (5 : ℝ) * 2 ^ 2 - k * 0 ^ 2 = 5) →
  k = (5 : ℝ) / 3 :=
by
  sorry

end hyperbola_focal_point_k_l2065_206551


namespace lauren_annual_income_l2065_206502

open Real

theorem lauren_annual_income (p : ℝ) (A : ℝ) (T : ℝ) :
  (T = (p + 0.45)/100 * A) →
  (T = (p/100) * 20000 + ((p + 1)/100) * 15000 + ((p + 3)/100) * (A - 35000)) →
  A = 36000 :=
by
  intros
  sorry

end lauren_annual_income_l2065_206502


namespace find_digit_A_l2065_206505

theorem find_digit_A : ∃ A : ℕ, A < 10 ∧ (200 + 10 * A + 4) % 13 = 0 ∧ A = 7 :=
by
  sorry

end find_digit_A_l2065_206505


namespace find_base_l2065_206506

theorem find_base (x y : ℕ) (b : ℕ) (h1 : 3 ^ x * b ^ y = 19683) (h2 : x - y = 9) (h3 : x = 9) : b = 1 := 
by
  sorry

end find_base_l2065_206506


namespace swim_time_l2065_206510

-- Definitions based on conditions:
def speed_in_still_water : ℝ := 6.5 -- speed of the man in still water (km/h)
def distance_downstream : ℝ := 16 -- distance swam downstream (km)
def distance_upstream : ℝ := 10 -- distance swam upstream (km)
def time_downstream := 2 -- time taken to swim downstream (hours)
def time_upstream := 2 -- time taken to swim upstream (hours)

-- Defining the speeds taking the current into account:
def speed_downstream (c : ℝ) : ℝ := speed_in_still_water + c
def speed_upstream (c : ℝ) : ℝ := speed_in_still_water - c

-- Assumption that the time took for both downstream and upstream are equal
def time_eq (c : ℝ) : Prop :=
  distance_downstream / (speed_downstream c) = distance_upstream / (speed_upstream c)

-- The proof we need to establish:
theorem swim_time (c : ℝ) (h : time_eq c) : time_downstream = time_upstream := by
  sorry

end swim_time_l2065_206510


namespace product_of_hypotenuse_segments_eq_area_l2065_206576

theorem product_of_hypotenuse_segments_eq_area (x y c t : ℝ) : 
  -- Conditions
  (c = x + y) → 
  (t = x * y) →
  -- Conclusion
  x * y = t :=
by
  intros
  sorry

end product_of_hypotenuse_segments_eq_area_l2065_206576


namespace correct_quotient_l2065_206552

theorem correct_quotient (Q : ℤ) (D : ℤ) (h1 : D = 21 * Q) (h2 : D = 12 * 35) : Q = 20 :=
by {
  sorry
}

end correct_quotient_l2065_206552


namespace total_rent_correct_recoup_investment_period_maximize_average_return_l2065_206593

noncomputable def initialInvestment := 720000
noncomputable def firstYearRent := 54000
noncomputable def annualRentIncrease := 4000
noncomputable def maxRentalPeriod := 40

-- Conditions on the rental period
variable (x : ℝ) (hx : 0 < x ∧ x ≤ 40)

-- Function for total rent after x years
noncomputable def total_rent (x : ℝ) := 0.2 * x^2 + 5.2 * x

-- Condition for investment recoup period
noncomputable def recoupInvestmentTime := ∃ x : ℝ, x ≥ 10 ∧ total_rent x ≥ initialInvestment

-- Function for transfer price
noncomputable def transfer_price (x : ℝ) := -0.3 * x^2 + 10.56 * x + 57.6

-- Function for average return on investment
noncomputable def annual_avg_return (x : ℝ) := (transfer_price x + total_rent x - initialInvestment) / x

-- Statement of theorems
theorem total_rent_correct (x : ℝ) (hx : 0 < x ∧ x ≤ 40) :
  total_rent x = 0.2 * x^2 + 5.2 * x := sorry

theorem recoup_investment_period :
  ∃ x : ℝ, x ≥ 10 ∧ total_rent x ≥ initialInvestment := sorry

theorem maximize_average_return :
  ∃ x : ℝ, x = 12 ∧ (∀ y : ℝ, annual_avg_return x ≥ annual_avg_return y) := sorry

end total_rent_correct_recoup_investment_period_maximize_average_return_l2065_206593


namespace true_inverse_propositions_count_l2065_206574

-- Let P1, P2, P3, P4 denote the original propositions
def P1 := "Supplementary angles are congruent, and two lines are parallel."
def P2 := "If |a| = |b|, then a = b."
def P3 := "Right angles are congruent."
def P4 := "Congruent angles are vertical angles."

-- Let IP1, IP2, IP3, IP4 denote the inverse propositions
def IP1 := "Two lines are parallel, and supplementary angles are congruent."
def IP2 := "If a = b, then |a| = |b|."
def IP3 := "Congruent angles are right angles."
def IP4 := "Vertical angles are congruent angles."

-- Counting the number of true inverse propositions
def countTrueInversePropositions : ℕ :=
  let p1_inverse_true := true  -- IP1 is true
  let p2_inverse_true := true  -- IP2 is true
  let p3_inverse_true := false -- IP3 is false
  let p4_inverse_true := true  -- IP4 is true
  [p1_inverse_true, p2_inverse_true, p4_inverse_true].length

-- The statement to be proved
theorem true_inverse_propositions_count : countTrueInversePropositions = 3 := by
  sorry

end true_inverse_propositions_count_l2065_206574


namespace jimmy_cards_left_l2065_206538

theorem jimmy_cards_left :
  ∀ (initial_cards jimmy_cards bob_cards mary_cards : ℕ),
    initial_cards = 18 →
    bob_cards = 3 →
    mary_cards = 2 * bob_cards →
    jimmy_cards = initial_cards - bob_cards - mary_cards →
    jimmy_cards = 9 := 
by
  intros initial_cards jimmy_cards bob_cards mary_cards h1 h2 h3 h4
  sorry

end jimmy_cards_left_l2065_206538


namespace market_value_decrease_l2065_206588

noncomputable def percentage_decrease_each_year : ℝ :=
  let original_value := 8000
  let value_after_two_years := 3200
  let p := 1 - (value_after_two_years / original_value)^(1 / 2)
  p * 100

theorem market_value_decrease :
  let p := percentage_decrease_each_year
  abs (p - 36.75) < 0.01 :=
by
  sorry

end market_value_decrease_l2065_206588


namespace cistern_depth_l2065_206553

noncomputable def length : ℝ := 9
noncomputable def width : ℝ := 4
noncomputable def total_wet_surface_area : ℝ := 68.5

theorem cistern_depth (h : ℝ) (h_def : 68.5 = 36 + 18 * h + 8 * h) : h = 1.25 :=
by sorry

end cistern_depth_l2065_206553


namespace weight_of_green_peppers_l2065_206546

-- Definitions for conditions and question
def total_weight : ℝ := 0.6666666667
def is_split_equally (x y : ℝ) : Prop := x = y

-- Theorem statement that needs to be proved
theorem weight_of_green_peppers (g r : ℝ) (h_split : is_split_equally g r) (h_total : g + r = total_weight) :
  g = 0.33333333335 :=
by sorry

end weight_of_green_peppers_l2065_206546


namespace competition_scores_order_l2065_206548

theorem competition_scores_order (A B C D : ℕ) (h1 : A + B = C + D) (h2 : C + A > D + B) (h3 : B > A + D) : (B > A) ∧ (A > C) ∧ (C > D) := 
by 
  sorry

end competition_scores_order_l2065_206548


namespace library_books_total_l2065_206596

-- Definitions for the conditions
def books_purchased_last_year : Nat := 50
def books_purchased_this_year : Nat := 3 * books_purchased_last_year
def books_before_last_year : Nat := 100

-- The library's current number of books
def total_books_now : Nat :=
  books_before_last_year + books_purchased_last_year + books_purchased_this_year

-- The proof statement
theorem library_books_total : total_books_now = 300 :=
by
  -- Placeholder for actual proof
  sorry

end library_books_total_l2065_206596


namespace terry_problems_wrong_l2065_206571

theorem terry_problems_wrong (R W : ℕ) 
  (h1 : R + W = 25) 
  (h2 : 4 * R - W = 85) : 
  W = 3 := 
by
  sorry

end terry_problems_wrong_l2065_206571


namespace increasing_function_fA_increasing_function_fB_increasing_function_fC_increasing_function_fD_l2065_206589

noncomputable def fA (x : ℝ) : ℝ := -x
noncomputable def fB (x : ℝ) : ℝ := (2/3)^x
noncomputable def fC (x : ℝ) : ℝ := x^2
noncomputable def fD (x : ℝ) : ℝ := x^(1/3)

theorem increasing_function_fA : ¬∀ x y : ℝ, x < y → fA x < fA y := sorry
theorem increasing_function_fB : ¬∀ x y : ℝ, x < y → fB x < fB y := sorry
theorem increasing_function_fC : ¬∀ x y : ℝ, x < y → fC x < fC y := sorry
theorem increasing_function_fD : ∀ x y : ℝ, x < y → fD x < fD y := sorry

end increasing_function_fA_increasing_function_fB_increasing_function_fC_increasing_function_fD_l2065_206589


namespace sin_gamma_plus_delta_l2065_206534

theorem sin_gamma_plus_delta (γ δ : ℝ) (hγ : Complex.exp (Complex.I * γ) = (4/5 : ℂ) + (3/5 : ℂ) * Complex.I)
                             (hδ : Complex.exp (Complex.I * δ) = (-5/13 : ℂ) + (12/13 : ℂ) * Complex.I) :
  Real.sin (γ + δ) = 33 / 65 :=
by
  sorry

end sin_gamma_plus_delta_l2065_206534


namespace eggs_left_after_cupcakes_l2065_206523

-- Definitions derived from the given conditions
def dozen := 12
def initial_eggs := 3 * dozen
def crepes_fraction := 1 / 4
def cupcakes_fraction := 2 / 3

theorem eggs_left_after_cupcakes :
  let eggs_after_crepes := initial_eggs - crepes_fraction * initial_eggs;
  let eggs_after_cupcakes := eggs_after_crepes - cupcakes_fraction * eggs_after_crepes;
  eggs_after_cupcakes = 9 := sorry

end eggs_left_after_cupcakes_l2065_206523


namespace sara_oranges_l2065_206517

-- Conditions
def joan_oranges : Nat := 37
def total_oranges : Nat := 47

-- Mathematically equivalent proof problem: Prove that the number of oranges picked by Sara is 10
theorem sara_oranges : total_oranges - joan_oranges = 10 :=
by
  sorry

end sara_oranges_l2065_206517


namespace selling_price_correct_l2065_206501

theorem selling_price_correct (cost_price : ℝ) (loss_percent : ℝ) (selling_price : ℝ) 
  (h_cost : cost_price = 600) 
  (h_loss : loss_percent = 25)
  (h_selling_price : selling_price = cost_price - (loss_percent / 100) * cost_price) : 
  selling_price = 450 := 
by 
  rw [h_cost, h_loss] at h_selling_price
  norm_num at h_selling_price
  exact h_selling_price

#check selling_price_correct

end selling_price_correct_l2065_206501


namespace question1_question2_l2065_206503

noncomputable def setA := {x : ℝ | -2 < x ∧ x < 4}
noncomputable def setB (m : ℝ) := {x : ℝ | x < -m}

-- (1) If A ∩ B = ∅, find the range of the real number m.
theorem question1 (m : ℝ) (h : setA ∩ setB m = ∅) : 2 ≤ m := by
  sorry

-- (2) If A ⊂ B, find the range of the real number m.
theorem question2 (m : ℝ) (h : setA ⊂ setB m) : m ≤ 4 := by
  sorry

end question1_question2_l2065_206503


namespace mogs_and_mags_to_migs_l2065_206568

theorem mogs_and_mags_to_migs:
  (∀ mags migs, 1 * mags = 8 * migs) ∧ 
  (∀ mogs mags, 1 * mogs = 6 * mags) → 
  10 * (6 * 8) + 6 * 8 = 528 := by 
  sorry

end mogs_and_mags_to_migs_l2065_206568


namespace population_net_increase_per_day_l2065_206561

theorem population_net_increase_per_day (birth_rate death_rate : ℚ) (seconds_per_day : ℕ) (net_increase : ℚ) :
  birth_rate = 7 / 2 ∧
  death_rate = 2 / 2 ∧
  seconds_per_day = 24 * 60 * 60 ∧
  net_increase = (birth_rate - death_rate) * seconds_per_day →
  net_increase = 216000 := 
by
  sorry

end population_net_increase_per_day_l2065_206561


namespace composite_proposition_l2065_206598

theorem composite_proposition :
  (∀ x : ℝ, x^2 ≥ 0) ∧ ¬ (1 < 0) :=
by
  sorry

end composite_proposition_l2065_206598


namespace necessary_sufficient_condition_l2065_206529

theorem necessary_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ a ≥ 4 :=
by
  sorry

end necessary_sufficient_condition_l2065_206529


namespace fraction_value_l2065_206547

theorem fraction_value
  (m n : ℕ)
  (h : m / n = 2 / 3) :
  m / (m + n) = 2 / 5 :=
sorry

end fraction_value_l2065_206547


namespace continuous_function_form_l2065_206515

noncomputable def f (t : ℝ) : ℝ := sorry

theorem continuous_function_form (f : ℝ → ℝ) (h1 : f 0 = -1 / 2) (h2 : ∀ x y, f (x + y) ≥ f x + f y + f (x * y) + 1) :
  ∃ (a : ℝ), ∀ x, f x = 1 / 2 + a * x + (a/2) * x ^ 2 := sorry

end continuous_function_form_l2065_206515


namespace convert_base_8_to_10_l2065_206541

theorem convert_base_8_to_10 :
  let n := 4532
  let b := 8
  n = 4 * b^3 + 5 * b^2 + 3 * b^1 + 2 * b^0 → 4 * 512 + 5 * 64 + 3 * 8 + 2 * 1 = 2394 :=
by
  sorry

end convert_base_8_to_10_l2065_206541


namespace integer_solutions_of_inequality_l2065_206527

theorem integer_solutions_of_inequality :
  {x : ℤ | 3 ≤ 5 - 2 * x ∧ 5 - 2 * x ≤ 9} = {-2, -1, 0, 1} :=
by
  sorry

end integer_solutions_of_inequality_l2065_206527


namespace actual_cost_of_article_l2065_206567

theorem actual_cost_of_article (x : ℝ) (h : 0.76 * x = 760) : x = 1000 :=
by 
  sorry

end actual_cost_of_article_l2065_206567


namespace problem_1_problem_2_l2065_206557

theorem problem_1 (x : ℝ) : (2 * x + 3)^2 = 16 ↔ x = 1/2 ∨ x = -7/2 := by
  sorry

theorem problem_2 (x : ℝ) : x^2 - 4 * x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 := by
  sorry

end problem_1_problem_2_l2065_206557


namespace square_rem_1_mod_9_l2065_206562

theorem square_rem_1_mod_9 (N : ℤ) (h : N % 9 = 1 ∨ N % 9 = 8) : (N * N) % 9 = 1 :=
by sorry

end square_rem_1_mod_9_l2065_206562


namespace arithmetic_mean_16_24_40_32_l2065_206592

theorem arithmetic_mean_16_24_40_32 : (16 + 24 + 40 + 32) / 4 = 28 :=
by
  sorry

end arithmetic_mean_16_24_40_32_l2065_206592


namespace probability_divisible_by_3_l2065_206560

theorem probability_divisible_by_3 (a b c : ℕ) (h : a ∈ Finset.range 2008 ∧ b ∈ Finset.range 2008 ∧ c ∈ Finset.range 2008) :
  (∃ p : ℚ, p = 1265/2007 ∧ (abc + ac + a) % 3 = 0) :=
sorry

end probability_divisible_by_3_l2065_206560


namespace find_E_coordinates_l2065_206511

structure Point :=
(x : ℚ)
(y : ℚ)

def A : Point := { x := -2, y := 1 }
def B : Point := { x := 1, y := 4 }
def C : Point := { x := 4, y := -3 }

def D : Point := 
  let m : ℚ := 1
  let n : ℚ := 2
  let x1 := A.x
  let y1 := A.y
  let x2 := B.x
  let y2 := B.y
  { x := (m * x2 + n * x1) / (m + n), y := (m * y2 + n * y1) / (m + n) }

theorem find_E_coordinates : 
  let k : ℚ := 4
  let x_E : ℚ := (k * C.x + D.x) / (k + 1)
  let y_E : ℚ := (k * C.y + D.y) / (k + 1)
  ∃ E : Point, E.x = (17:ℚ) / 3 ∧ E.y = -(14:ℚ) / 3 :=
sorry

end find_E_coordinates_l2065_206511


namespace cars_meet_after_5_hours_l2065_206522

theorem cars_meet_after_5_hours :
  ∀ (t : ℝ), (40 * t + 60 * t = 500) → t = 5 := 
by
  intro t
  intro h
  sorry

end cars_meet_after_5_hours_l2065_206522


namespace slope_y_intercept_product_eq_neg_five_over_two_l2065_206559

theorem slope_y_intercept_product_eq_neg_five_over_two :
  let A := (0, 10)
  let B := (0, 0)
  let C := (10, 0)
  let D := ((0 + 0) / 2, (10 + 0) / 2) -- midpoint of A and B
  let slope := (D.2 - C.2) / (D.1 - C.1)
  let y_intercept := D.2
  slope * y_intercept = -5 / 2 := 
by 
  sorry

end slope_y_intercept_product_eq_neg_five_over_two_l2065_206559


namespace triangle_perimeter_is_correct_l2065_206555

open Real

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (S : ℝ)

def triangle_perimeter (a b c : ℝ) := a + b + c

theorem triangle_perimeter_is_correct :
  c = sqrt 7 → C = π / 3 → S = 3 * sqrt 3 / 2 →
  S = (1 / 2) * a * b * sin (C) → c^2 = a^2 + b^2 - 2 * a * b * cos (C) →
  ∃ a b : ℝ, triangle_perimeter a b c = 5 + sqrt 7 :=
  by
    intros h1 h2 h3 h4 h5
    sorry

end triangle_perimeter_is_correct_l2065_206555


namespace earnings_per_hour_l2065_206512

-- Define the conditions
def widgetsProduced : Nat := 750
def hoursWorked : Nat := 40
def totalEarnings : ℝ := 620
def earningsPerWidget : ℝ := 0.16

-- Define the proof goal
theorem earnings_per_hour :
  ∃ H : ℝ, (hoursWorked * H + widgetsProduced * earningsPerWidget = totalEarnings) ∧ H = 12.5 :=
by
  sorry

end earnings_per_hour_l2065_206512


namespace width_at_bottom_l2065_206564

-- Defining the given values and conditions
def top_width : ℝ := 14
def area : ℝ := 770
def depth : ℝ := 70

-- The proof problem
theorem width_at_bottom (b : ℝ) (h : area = (1/2) * (top_width + b) * depth) : b = 8 :=
by
  sorry

end width_at_bottom_l2065_206564


namespace paperback_copies_sold_l2065_206500

theorem paperback_copies_sold
  (H P : ℕ)
  (h1 : H = 36000)
  (h2 : H + P = 440000) :
  P = 404000 :=
by
  rw [h1] at h2
  sorry

end paperback_copies_sold_l2065_206500


namespace soccer_team_games_played_l2065_206586

theorem soccer_team_games_played (t : ℝ) (h1 : 0.40 * t = 63.2) : t = 158 :=
sorry

end soccer_team_games_played_l2065_206586


namespace total_limes_l2065_206508

-- Define the number of limes picked by Alyssa, Mike, and Tom's plums
def alyssa_limes : ℕ := 25
def mike_limes : ℕ := 32
def tom_plums : ℕ := 12

theorem total_limes : alyssa_limes + mike_limes = 57 := by
  -- The proof is omitted as per the instruction
  sorry

end total_limes_l2065_206508


namespace sampling_method_is_systematic_sampling_l2065_206595

-- Definitions based on the problem's conditions
def produces_products (factory : Type) : Prop := sorry
def uses_conveyor_belt (factory : Type) : Prop := sorry
def takes_item_every_5_minutes (inspector : Type) : Prop := sorry

-- Lean 4 statement to prove the question equals the answer given the conditions
theorem sampling_method_is_systematic_sampling
  (factory : Type)
  (inspector : Type)
  (h1 : produces_products factory)
  (h2 : uses_conveyor_belt factory)
  (h3 : takes_item_every_5_minutes inspector) :
  systematic_sampling_method := 
sorry

end sampling_method_is_systematic_sampling_l2065_206595


namespace limit_equivalence_l2065_206573

open Nat
open Real

variable {u : ℕ → ℝ} {L : ℝ}

def original_def (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |L - u n| ≤ ε

def def1 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε : ℝ, ε ≤ 0 ∨ (∃ N : ℕ, ∀ n : ℕ, n < N ∨ |L - u n| ≤ ε)

def def2 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∀ n : ℕ, ∃ N : ℕ, n ≥ N → |L - u n| ≤ ε

def def3 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n > N, |L - u n| < ε

def def4 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∃ N : ℕ, ∀ ε > 0, ∀ n ≥ N, |L - u n| ≤ ε

theorem limit_equivalence :
  original_def u L ↔ def1 u L ∧ def3 u L ∧ ¬def2 u L ∧ ¬def4 u L :=
by
  sorry

end limit_equivalence_l2065_206573


namespace simplify_expression_l2065_206542

theorem simplify_expression (x : ℝ) (h₁ : x ≠ -2) (h₂ : x ≠ 3) :
  (3 * x ^ 2 - 2 * x - 4) / ((x + 2) * (x - 3)) - (5 + x) / ((x + 2) * (x - 3)) =
  3 * (x ^ 2 - x - 3) / ((x + 2) * (x - 3)) :=
by
  sorry

end simplify_expression_l2065_206542


namespace area_of_figure_l2065_206545

def equation (x y : ℝ) : Prop := |15 * x| + |8 * y| + |120 - 15 * x - 8 * y| = 120

theorem area_of_figure : ∃ (A : ℝ), A = 60 ∧ 
  (∃ (x y : ℝ), equation x y) :=
sorry

end area_of_figure_l2065_206545


namespace cube_root_of_4913_has_unit_digit_7_cube_root_of_50653_is_37_cube_root_of_110592_is_48_l2065_206520

theorem cube_root_of_4913_has_unit_digit_7 :
  (∃ (y : ℕ), y^3 = 4913 ∧ y % 10 = 7) :=
sorry

theorem cube_root_of_50653_is_37 :
  (∃ (y : ℕ), y = 37 ∧ y^3 = 50653) :=
sorry

theorem cube_root_of_110592_is_48 :
  (∃ (y : ℕ), y = 48 ∧ y^3 = 110592) :=
sorry

end cube_root_of_4913_has_unit_digit_7_cube_root_of_50653_is_37_cube_root_of_110592_is_48_l2065_206520


namespace correct_average_l2065_206591

theorem correct_average
  (incorrect_avg : ℝ)
  (incorrect_num correct_num : ℝ)
  (n : ℕ)
  (h1 : incorrect_avg = 16)
  (h2 : incorrect_num = 26)
  (h3 : correct_num = 46)
  (h4 : n = 10) :
  (incorrect_avg * n - incorrect_num + correct_num) / n = 18 :=
sorry

end correct_average_l2065_206591


namespace problem_statement_l2065_206585

theorem problem_statement :
  (∃ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ),
    (∀ x : ℝ, 1 + x^5 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + 
              a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5) ∧
    (a_0 = 2) ∧
    (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 33)) →
  (∃ a_1 a_2 a_3 a_4 a_5 : ℝ, a_1 + a_2 + a_3 + a_4 + a_5 = 31) :=
by
  sorry

end problem_statement_l2065_206585


namespace reflect_point_P_l2065_206549

-- Define the point P
def P : ℝ × ℝ := (-3, 2)

-- Define the reflection across the x-axis
def reflect_x_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.1, -point.2)

-- Theorem to prove the coordinates of the point P with respect to the x-axis
theorem reflect_point_P : reflect_x_axis P = (-3, -2) := by
  sorry

end reflect_point_P_l2065_206549


namespace binomial_20_5_l2065_206533

theorem binomial_20_5 : Nat.choose 20 5 = 15504 := by
  sorry

end binomial_20_5_l2065_206533


namespace problem_statement_l2065_206543

variable {x y z : ℝ}

theorem problem_statement 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z)
  (hxyz : x * y * z = 1) :
  1 / (x ^ 3 * y) + 1 / (y ^ 3 * z) + 1 / (z ^ 3 * x) ≥ x * y + y * z + z * x :=
by sorry

end problem_statement_l2065_206543


namespace most_frequent_data_is_mode_l2065_206524

def most_frequent_data_name (dataset : Type) : String := "Mode"

theorem most_frequent_data_is_mode (dataset : Type) :
  most_frequent_data_name dataset = "Mode" :=
by
  sorry

end most_frequent_data_is_mode_l2065_206524


namespace percentage_error_computation_l2065_206575

theorem percentage_error_computation (x : ℝ) (h : 0 < x) : 
  let correct_result := 8 * x
  let erroneous_result := x / 8
  let error := |correct_result - erroneous_result|
  let error_percentage := (error / correct_result) * 100
  error_percentage = 98 :=
by
  sorry

end percentage_error_computation_l2065_206575


namespace subset_0_in_X_l2065_206514

-- Define the set X
def X : Set ℤ := { x | -2 ≤ x ∧ x ≤ 2 }

-- Define the theorem to prove
theorem subset_0_in_X : {0} ⊆ X :=
by
  sorry

end subset_0_in_X_l2065_206514


namespace coupon_discount_l2065_206570

theorem coupon_discount (total_before_coupon : ℝ) (amount_paid_per_friend : ℝ) (number_of_friends : ℕ) :
  total_before_coupon = 100 ∧ amount_paid_per_friend = 18.8 ∧ number_of_friends = 5 →
  ∃ discount_percentage : ℝ, discount_percentage = 6 :=
by
  sorry

end coupon_discount_l2065_206570


namespace isosceles_triangle_circum_incenter_distance_l2065_206599

variable {R r d : ℝ}

/-- The distance \(d\) between the centers of the circumscribed circle and the inscribed circle of an isosceles triangle satisfies \(d = \sqrt{R(R - 2r)}\) --/
theorem isosceles_triangle_circum_incenter_distance (hR : 0 < R) (hr : 0 < r) 
  (hIso : ∃ (A B C : ℝ × ℝ), (A ≠ B) ∧ (A ≠ C) ∧ (B ≠ C) ∧ (dist A B = dist A C)) 
  : d = Real.sqrt (R * (R - 2 * r)) :=
sorry

end isosceles_triangle_circum_incenter_distance_l2065_206599


namespace f_divides_f_2k_plus_1_f_coprime_f_multiple_l2065_206563

noncomputable def f (g n : ℕ) : ℕ := g ^ n + 1

theorem f_divides_f_2k_plus_1 (g : ℕ) (k n : ℕ) :
  f g n ∣ f g ((2 * k + 1) * n) :=
by sorry

theorem f_coprime_f_multiple (g n : ℕ) :
  Nat.Coprime (f g n) (f g (2 * n)) ∧
  Nat.Coprime (f g n) (f g (4 * n)) ∧
  Nat.Coprime (f g n) (f g (6 * n)) :=
by sorry

end f_divides_f_2k_plus_1_f_coprime_f_multiple_l2065_206563


namespace min_value_ineq_l2065_206531

theorem min_value_ineq (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) : 
  (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) ≥ 3 / 2 := 
sorry

end min_value_ineq_l2065_206531


namespace balance_difference_l2065_206513

def compounded_balance (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n

def simple_interest_balance (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * (1 + r * t)

/-- Cedric deposits $15,000 into an account that pays 6% interest compounded annually,
    Daniel deposits $15,000 into an account that pays 8% simple annual interest.
    After 10 years, the positive difference between their balances is $137. -/
theorem balance_difference :
  let P : ℝ := 15000
  let r_cedric : ℝ := 0.06
  let r_daniel : ℝ := 0.08
  let t : ℕ := 10
  compounded_balance P r_cedric t - simple_interest_balance P r_daniel t = 137 := 
sorry

end balance_difference_l2065_206513


namespace find_x_when_y_equals_2_l2065_206550

theorem find_x_when_y_equals_2 :
  ∀ (y x k : ℝ),
  (y * (Real.sqrt x + 1) = k) →
  (y = 5 → x = 1 → k = 10) →
  (y = 2 → x = 16) := by
  intros y x k h_eq h_initial h_final
  sorry

end find_x_when_y_equals_2_l2065_206550


namespace tiffany_won_lives_l2065_206532
-- Step d: Lean 4 statement incorporating the conditions and the proof goal


-- Define initial lives, lives won in the hard part and the additional lives won
def initial_lives : Float := 43.0
def additional_lives : Float := 27.0
def total_lives_after_wins : Float := 84.0

open Classical

theorem tiffany_won_lives (x : Float) :
    initial_lives + x + additional_lives = total_lives_after_wins →
    x = 14.0 :=
by
  intros h
  -- This "sorry" indicates that the proof is skipped.
  sorry

end tiffany_won_lives_l2065_206532


namespace option_A_option_D_l2065_206535

variable {a : ℕ → ℤ} -- The arithmetic sequence
variable {S : ℕ → ℤ} -- Sum of the first n terms
variable {a1 d : ℤ} -- First term and common difference

-- Conditions for arithmetic sequence
axiom a_n (n : ℕ) : a n = a1 + ↑(n-1) * d
axiom S_n (n : ℕ) : S n = n * a1 + (n * (n - 1) / 2) * d
axiom condition : a 4 + 2 * a 8 = a 6

theorem option_A : a 7 = 0 :=
by
  -- Proof to be done
  sorry

theorem option_D : S 13 = 0 :=
by
  -- Proof to be done
  sorry

end option_A_option_D_l2065_206535


namespace intersection_empty_l2065_206583

def A : Set ℝ := {x | x > -1 ∧ x ≤ 3}
def B : Set ℝ := {2, 4}

theorem intersection_empty : A ∩ B = ∅ := 
by
  sorry

end intersection_empty_l2065_206583


namespace milkman_cows_l2065_206539

theorem milkman_cows (x : ℕ) (c : ℕ) :
  (3 * x * c = 720) ∧ (3 * x * c + 50 * c + 140 * c + 63 * c = 3250) → x = 24 :=
by
  sorry

end milkman_cows_l2065_206539


namespace keiko_speed_l2065_206528

theorem keiko_speed (wA wB tA tB : ℝ) (v : ℝ)
    (h1: wA = 4)
    (h2: wB = 8)
    (h3: tA = 48)
    (h4: tB = 72)
    (h5: v = (24 * π) / 60) :
    v = 2 * π / 5 :=
by
  sorry

end keiko_speed_l2065_206528


namespace circles_intersect_l2065_206530

theorem circles_intersect
  (r : ℝ) (R : ℝ) (d : ℝ)
  (hr : r = 4)
  (hR : R = 5)
  (hd : d = 6) :
  1 < d ∧ d < r + R :=
by
  sorry

end circles_intersect_l2065_206530


namespace problem1_problem2_l2065_206572

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

end problem1_problem2_l2065_206572


namespace cos_product_identity_l2065_206597

theorem cos_product_identity :
  (Real.cos (20 * Real.pi / 180)) * (Real.cos (40 * Real.pi / 180)) *
  (Real.cos (60 * Real.pi / 180)) * (Real.cos (80 * Real.pi / 180)) = 1 / 16 := 
by
  sorry

end cos_product_identity_l2065_206597


namespace gcd_36_60_l2065_206558

theorem gcd_36_60 : Nat.gcd 36 60 = 12 := by
  sorry

end gcd_36_60_l2065_206558


namespace sequence_a500_l2065_206565

theorem sequence_a500 (a : ℕ → ℤ)
  (h1 : a 1 = 2010)
  (h2 : a 2 = 2011)
  (h3 : ∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = n) :
  a 500 = 2177 :=
sorry

end sequence_a500_l2065_206565


namespace find_m_l2065_206525

open Complex

theorem find_m (m : ℝ) : (re ((1 + I) / (1 - I) + m * (1 - I) / (1 + I)) = ((1 + I) / (1 - I) + m * (1 - I) / (1 + I))) → m = 1 :=
by
  sorry

end find_m_l2065_206525


namespace total_detergent_is_19_l2065_206566

-- Define the quantities and usage of detergent
def detergent_per_pound_cotton := 2
def detergent_per_pound_woolen := 3
def detergent_per_pound_synthetic := 1

def pounds_of_cotton := 4
def pounds_of_woolen := 3
def pounds_of_synthetic := 2

-- Define the function to calculate the total amount of detergent needed
def total_detergent_needed := 
  detergent_per_pound_cotton * pounds_of_cotton +
  detergent_per_pound_woolen * pounds_of_woolen +
  detergent_per_pound_synthetic * pounds_of_synthetic

-- The theorem to prove the total amount of detergent used
theorem total_detergent_is_19 : total_detergent_needed = 19 :=
  by { sorry }

end total_detergent_is_19_l2065_206566


namespace complex_modulus_product_l2065_206590

noncomputable def z1 : ℂ := 4 - 3 * Complex.I
noncomputable def z2 : ℂ := 4 + 3 * Complex.I

theorem complex_modulus_product : Complex.abs z1 * Complex.abs z2 = 25 := by 
  sorry

end complex_modulus_product_l2065_206590


namespace adam_earnings_after_taxes_l2065_206507

theorem adam_earnings_after_taxes
  (daily_earnings : ℕ) 
  (tax_pct : ℕ)
  (workdays : ℕ)
  (H1 : daily_earnings = 40) 
  (H2 : tax_pct = 10) 
  (H3 : workdays = 30) : 
  (daily_earnings - daily_earnings * tax_pct / 100) * workdays = 1080 := 
by
  -- Proof to be filled in
  sorry

end adam_earnings_after_taxes_l2065_206507


namespace find_correct_t_l2065_206518

theorem find_correct_t (t : ℝ) :
  (∃! x1 x2 x3 : ℝ, x1^2 - 4*|x1| + 3 = t ∧
                     x2^2 - 4*|x2| + 3 = t ∧
                     x3^2 - 4*|x3| + 3 = t) → t = 3 :=
by
  sorry

end find_correct_t_l2065_206518


namespace total_money_together_is_l2065_206544

def Sam_has : ℚ := 750.50
def Billy_has (S : ℚ) : ℚ := 4.5 * S - 345.25
def Lila_has (B S : ℚ) : ℚ := 2.25 * (B - S)
def Total_money (S B L : ℚ) : ℚ := S + B + L

theorem total_money_together_is :
  Total_money Sam_has (Billy_has Sam_has) (Lila_has (Billy_has Sam_has) Sam_has) = 8915.88 :=
by sorry

end total_money_together_is_l2065_206544


namespace least_number_of_shoes_l2065_206504

theorem least_number_of_shoes (num_inhabitants : ℕ) 
  (one_legged_percentage : ℚ) 
  (barefooted_proportion : ℚ) 
  (h_num_inhabitants : num_inhabitants = 10000) 
  (h_one_legged_percentage : one_legged_percentage = 0.05) 
  (h_barefooted_proportion : barefooted_proportion = 0.5) : 
  ∃ (shoes_needed : ℕ), shoes_needed = 10000 := 
by
  sorry

end least_number_of_shoes_l2065_206504


namespace remainder_when_x_plus_2uy_div_y_l2065_206536

theorem remainder_when_x_plus_2uy_div_y (x y u v : ℕ) (h1 : x = u * y + v) (h2 : 0 ≤ v) (h3 : v < y) :
  (x + 2 * u * y) % y = v := 
sorry

end remainder_when_x_plus_2uy_div_y_l2065_206536


namespace circumcircle_eq_of_triangle_vertices_l2065_206594

theorem circumcircle_eq_of_triangle_vertices (A B C: ℝ × ℝ) (hA : A = (0, 4)) (hB : B = (0, 0)) (hC : C = (3, 0)) :
  ∃ D E F : ℝ,
    x^2 + y^2 + D*x + E*y + F = 0 ∧
    (x - 3/2)^2 + (y - 2)^2 = 25/4 :=
by 
  sorry

end circumcircle_eq_of_triangle_vertices_l2065_206594


namespace a_2_value_general_terms_T_n_value_l2065_206509

-- Definitions based on conditions
def S (n : ℕ) : ℕ := sorry  -- Sum of the first n terms of sequence {a_n}

def a (n : ℕ) : ℕ := (S n + 2) / 2  -- a_n is the arithmetic mean of S_n and 2

def b (n : ℕ) : ℕ := 2 * n - 1  -- Given general term for b_n

-- Prove a_2 = 4
theorem a_2_value : a 2 = 4 := 
by
  sorry

-- Prove the general terms
theorem general_terms (n : ℕ) : a n = 2^n ∧ b n = 2 * n - 1 := 
by
  sorry

-- Definition and sum of the first n terms of c_n
def c (n : ℕ) : ℕ := a n * b n

def T (n : ℕ) : ℕ := (2 * n - 3) * 2^(n + 1) + 6  -- Given sum of the first n terms of {c_n}

-- Prove T_n = (2n - 3)2^(n+1) + 6
theorem T_n_value (n : ℕ) : T n = (2 * n - 3) * 2^(n + 1) + 6 :=
by
  sorry

end a_2_value_general_terms_T_n_value_l2065_206509


namespace carousel_revolutions_l2065_206579

/-- Prove that the number of revolutions a horse 4 feet from the center needs to travel the same distance
as a horse 16 feet from the center making 40 revolutions is 160 revolutions. -/
theorem carousel_revolutions (r₁ : ℕ := 16) (revolutions₁ : ℕ := 40) (r₂ : ℕ := 4) :
  (revolutions₁ * (r₁ / r₂) = 160) :=
sorry

end carousel_revolutions_l2065_206579


namespace increasing_interval_of_y_l2065_206584

noncomputable def y (x : ℝ) : ℝ := (Real.log x) / x

theorem increasing_interval_of_y :
  ∃ (a b : ℝ), 0 < a ∧ a < e ∧ (∀ x : ℝ, a < x ∧ x < e → y x < y (x + ε)) :=
sorry

end increasing_interval_of_y_l2065_206584


namespace cost_of_cookbook_l2065_206581

def cost_of_dictionary : ℕ := 11
def cost_of_dinosaur_book : ℕ := 19
def amount_saved : ℕ := 8
def amount_needed : ℕ := 29

theorem cost_of_cookbook :
  let total_cost := amount_saved + amount_needed
  let accounted_cost := cost_of_dictionary + cost_of_dinosaur_book
  total_cost - accounted_cost = 7 :=
by
  sorry

end cost_of_cookbook_l2065_206581


namespace trigonometric_inequalities_l2065_206537

noncomputable def a : ℝ := Real.sin (21 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (72 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (23 * Real.pi / 180)

-- The proof statement
theorem trigonometric_inequalities : c > a ∧ a > b :=
by
  sorry

end trigonometric_inequalities_l2065_206537


namespace simplify_fraction_l2065_206540

theorem simplify_fraction :
  (4^5 + 4^3) / (4^4 - 4^2) = 68 / 15 :=
by sorry

end simplify_fraction_l2065_206540


namespace fraction_BC_AD_l2065_206516

-- Defining points and segments
variables (A B C D : Point)
variable (len : Point → Point → ℝ) -- length function

-- Conditions
axiom AB_eq_3BD : len A B = 3 * len B D
axiom AC_eq_7CD : len A C = 7 * len C D
axiom B_mid_AD : 2 * len A B = len A D

-- Theorem: Proving the fraction of BC relative to AD is 2/3
theorem fraction_BC_AD : (len B C) / (len A D) = 2 / 3 :=
sorry

end fraction_BC_AD_l2065_206516


namespace roots_in_interval_l2065_206587

theorem roots_in_interval (a b : ℝ) (hb : b > 0) (h_discriminant : a^2 - 4 * b > 0)
  (h_root_interval : ∃ r1 r2 : ℝ, r1 + r2 = -a ∧ r1 * r2 = b ∧ ((-1 ≤ r1 ∧ r1 ≤ 1 ∧ (r2 < -1 ∨ 1 < r2)) ∨ (-1 ≤ r2 ∧ r2 ≤ 1 ∧ (r1 < -1 ∨ 1 < r1)))) : 
  ∃ r : ℝ, (r + a) * r + b = 0 ∧ -b < r ∧ r < b :=
by
  sorry

end roots_in_interval_l2065_206587


namespace least_number_subtracted_l2065_206521

/--
  What least number must be subtracted from 9671 so that the remaining number is divisible by 5, 7, and 11?
-/
theorem least_number_subtracted
  (x : ℕ) :
  (9671 - x) % 5 = 0 ∧ (9671 - x) % 7 = 0 ∧ (9671 - x) % 11 = 0 ↔ x = 46 :=
sorry

end least_number_subtracted_l2065_206521


namespace length_of_AB_l2065_206519

def parabola_eq (y : ℝ) : Prop := y^2 = 8 * y

def directrix_x : ℝ := 2

def dist_to_y_axis (E : ℝ × ℝ) : ℝ := E.1

theorem length_of_AB (A B F E : ℝ × ℝ)
  (p : parabola_eq A.2) (q : parabola_eq B.2) 
  (F_focus : F.1 = 2 ∧ F.2 = 0) 
  (midpoint_E : E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (E_distance_from_y_axis : dist_to_y_axis E = 3) : 
  (abs (A.1 - B.1) + abs (A.2 - B.2)) = 10 := 
sorry

end length_of_AB_l2065_206519


namespace find_z_l2065_206526

theorem find_z (z : ℝ) (h : (z^2 - 5 * z + 6) / (z - 2) + (5 * z^2 + 11 * z - 32) / (5 * z - 16) = 1) : z = 1 :=
sorry

end find_z_l2065_206526


namespace tangent_line_at_2_eq_l2065_206578

noncomputable def f (x : ℝ) : ℝ := x^3 - 4 * x^2 + 5 * x - 4

theorem tangent_line_at_2_eq :
  let x := (2 : ℝ)
  let slope := (deriv f) x
  let y := f x
  ∃ (m y₀ : ℝ), m = slope ∧ y₀ = y ∧ 
    (∀ (x y : ℝ), y = m * (x - 2) + y₀ → x - y - 4 = 0)
:= sorry

end tangent_line_at_2_eq_l2065_206578


namespace polynomial_difference_of_squares_l2065_206556

theorem polynomial_difference_of_squares:
  (∀ a b : ℝ, ¬ ∃ x1 x2 : ℝ, a^2 + (-b)^2 = (x1 - x2) * (x1 + x2)) ∧
  (∀ m n : ℝ, ¬ ∃ x1 x2 : ℝ, 5 * m^2 - 20 * m * n = (x1 - x2) * (x1 + x2)) ∧
  (∀ x y : ℝ, ¬ ∃ x1 x2 : ℝ, -x^2 - y^2 = (x1 - x2) * (x1 + x2)) →
  ∃ x1 x2 : ℝ, -x^2 + 9 = (x1 - x2) * (x1 + x2) :=
by 
  sorry

end polynomial_difference_of_squares_l2065_206556


namespace arithmetic_sequence_a14_eq_41_l2065_206554

theorem arithmetic_sequence_a14_eq_41 (a : ℕ → ℤ) (d : ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d) 
  (h_a2 : a 2 = 5) 
  (h_a6 : a 6 = 17) : 
  a 14 = 41 :=
sorry

end arithmetic_sequence_a14_eq_41_l2065_206554


namespace total_fortunate_numbers_is_65_largest_odd_fortunate_number_is_1995_l2065_206569

-- Definition of properties required as per the given conditions
def is_fortunate_number (abcd ab cd : ℕ) : Prop :=
  abcd = 100 * ab + cd ∧
  ab ≠ cd ∧
  ab ∣ cd ∧
  cd ∣ abcd

-- Total number of fortunate numbers is 65
theorem total_fortunate_numbers_is_65 : 
  ∃ n : ℕ, n = 65 ∧ 
  ∀(abcd ab cd : ℕ), is_fortunate_number abcd ab cd → n = 65 :=
sorry

-- Largest odd fortunate number is 1995
theorem largest_odd_fortunate_number_is_1995 : 
  ∃ abcd : ℕ, abcd = 1995 ∧ 
  ∀(abcd' ab cd : ℕ), is_fortunate_number abcd' ab cd ∧ cd % 2 = 1 → abcd = 1995 :=
sorry

end total_fortunate_numbers_is_65_largest_odd_fortunate_number_is_1995_l2065_206569
