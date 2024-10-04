import Mathlib

namespace color_preference_l214_214385

-- Define the conditions
def total_students := 50
def girls := 30
def boys := 20

def girls_pref_pink := girls / 3
def girls_pref_purple := 2 * girls / 5
def girls_pref_blue := girls - girls_pref_pink - girls_pref_purple

def boys_pref_red := 2 * boys / 5
def boys_pref_green := 3 * boys / 10
def boys_pref_orange := boys - boys_pref_red - boys_pref_green

-- Proof statement
theorem color_preference :
  girls_pref_pink = 10 ∧
  girls_pref_purple = 12 ∧
  girls_pref_blue = 8 ∧
  boys_pref_red = 8 ∧
  boys_pref_green = 6 ∧
  boys_pref_orange = 6 :=
by
  sorry

end color_preference_l214_214385


namespace find_N_l214_214740

theorem find_N (N : ℕ) (hN : N > 1) (h1 : 2019 ≡ 1743 [MOD N]) (h2 : 3008 ≡ 2019 [MOD N]) : N = 23 :=
by
  sorry

end find_N_l214_214740


namespace intersection_M_N_l214_214515

def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | (1/3) ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x | (1/3) ≤ x ∧ x < 4} := by
  sorry

end intersection_M_N_l214_214515


namespace strawberry_pancakes_l214_214587

theorem strawberry_pancakes (total blueberry banana chocolate : ℕ) (h_total : total = 150) (h_blueberry : blueberry = 45) (h_banana : banana = 60) (h_chocolate : chocolate = 25) :
  total - (blueberry + banana + chocolate) = 20 :=
by
  sorry

end strawberry_pancakes_l214_214587


namespace goldfinch_percentage_l214_214001

def number_of_goldfinches := 6
def number_of_sparrows := 9
def number_of_grackles := 5
def total_birds := number_of_goldfinches + number_of_sparrows + number_of_grackles
def goldfinch_fraction := (number_of_goldfinches : ℚ) / total_birds

theorem goldfinch_percentage : goldfinch_fraction * 100 = 30 := 
by
  sorry

end goldfinch_percentage_l214_214001


namespace remainder_when_divided_by_32_l214_214968

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l214_214968


namespace simplify_fraction_l214_214193

def a : ℕ := 2016
def b : ℕ := 2017

theorem simplify_fraction :
  (a^4 - 2 * a^3 * b + 3 * a^2 * b^2 - a * b^3 + 1) / (a^2 * b^2) = 1 - 1 / b^2 :=
by
  sorry

end simplify_fraction_l214_214193


namespace total_handshakes_is_316_l214_214720

def number_of_couples : ℕ := 15
def number_of_people : ℕ := number_of_couples * 2

def handshakes_among_men (n : ℕ) : ℕ := n * (n - 1) / 2
def handshakes_men_women (n : ℕ) : ℕ := n * (n - 1)
def handshakes_between_women : ℕ := 1
def total_handshakes (n : ℕ) : ℕ := handshakes_among_men n + handshakes_men_women n + handshakes_between_women

theorem total_handshakes_is_316 : total_handshakes number_of_couples = 316 :=
by
  sorry

end total_handshakes_is_316_l214_214720


namespace total_votes_l214_214188

theorem total_votes (Ben_votes Matt_votes total_votes : ℕ)
  (h_ratio : 2 * Matt_votes = 3 * Ben_votes)
  (h_Ben_votes : Ben_votes = 24) :
  total_votes = Ben_votes + Matt_votes :=
sorry

end total_votes_l214_214188


namespace mode_and_median_correct_l214_214315

open Finset

noncomputable def data_set := {2, 7, 6, 3, 4, 7}

def mode (s : Finset ℕ) : ℕ := 7 -- as 7 appears most frequently
def median (s : Finset ℕ) : ℚ := 5 -- the median is calculated as (4 + 6) / 2

theorem mode_and_median_correct : 
  mode data_set = 7 ∧ median data_set = 5 := by
  -- proof will go here
  sorry

end mode_and_median_correct_l214_214315


namespace good_walker_catch_up_l214_214106

theorem good_walker_catch_up :
  ∀ x y : ℕ, 
    (x = (100:ℕ) + y) ∧ (x = ((100:ℕ)/(60:ℕ) : ℚ) * y) := 
by
  sorry

end good_walker_catch_up_l214_214106


namespace gcd_of_180_and_450_l214_214601

theorem gcd_of_180_and_450 : Int.gcd 180 450 = 90 := 
  sorry

end gcd_of_180_and_450_l214_214601


namespace helicopter_rental_cost_l214_214685

theorem helicopter_rental_cost :
  let hours_per_day := 2
  let days := 3
  let rate_first_day := 85
  let rate_second_day := 75
  let rate_third_day := 65
  let total_cost_before_discount := hours_per_day * rate_first_day + hours_per_day * rate_second_day + hours_per_day * rate_third_day
  let discount := 0.05
  let discounted_amount := total_cost_before_discount * discount
  let total_cost_after_discount := total_cost_before_discount - discounted_amount
  total_cost_after_discount = 427.50 :=
by
  sorry

end helicopter_rental_cost_l214_214685


namespace correct_reasoning_l214_214854

-- Define that every multiple of 9 is a multiple of 3
def multiple_of_9_is_multiple_of_3 : Prop :=
  ∀ n : ℤ, n % 9 = 0 → n % 3 = 0

-- Define that a certain odd number is a multiple of 9
def odd_multiple_of_9 (n : ℤ) : Prop :=
  n % 2 = 1 ∧ n % 9 = 0

-- The goal: Prove that the reasoning process is completely correct
theorem correct_reasoning (H1 : multiple_of_9_is_multiple_of_3)
                          (n : ℤ)
                          (H2 : odd_multiple_of_9 n) : 
                          (n % 3 = 0) :=
by
  -- Explanation of the proof here
  sorry

end correct_reasoning_l214_214854


namespace dino_second_gig_hourly_rate_l214_214887

theorem dino_second_gig_hourly_rate (h1 : 20 * 10 = 200)
  (h2 : 5 * 40 = 200) (h3 : 500 + 500 = 1000) : 
  let total_income := 1000 
  let income_first_gig := 200 
  let income_third_gig := 200 
  let income_second_gig := total_income - income_first_gig - income_third_gig 
  let hours_second_gig := 30 
  let hourly_rate := income_second_gig / hours_second_gig 
  hourly_rate = 20 := 
by 
  sorry

end dino_second_gig_hourly_rate_l214_214887


namespace deposit_percentage_l214_214131

-- Define the conditions of the problem
def amount_deposited : ℕ := 5000
def monthly_income : ℕ := 25000

-- Define the percentage deposited formula
def percentage_deposited (amount_deposited monthly_income : ℕ) : ℚ :=
  (amount_deposited / monthly_income) * 100

-- State the theorem to be proved
theorem deposit_percentage :
  percentage_deposited amount_deposited monthly_income = 20 := by
  sorry

end deposit_percentage_l214_214131


namespace chicken_nuggets_order_l214_214127

theorem chicken_nuggets_order (cost_per_box : ℕ) (nuggets_per_box : ℕ) (total_amount_paid : ℕ) 
  (h1 : cost_per_box = 4) (h2 : nuggets_per_box = 20) (h3 : total_amount_paid = 20) : 
  total_amount_paid / cost_per_box * nuggets_per_box = 100 :=
by
  -- This is where the proof would go
  sorry

end chicken_nuggets_order_l214_214127


namespace intersection_M_N_l214_214517

def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | (1/3) ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x | (1/3) ≤ x ∧ x < 4} := by
  sorry

end intersection_M_N_l214_214517


namespace C_share_of_profit_l214_214867

def A_investment : ℕ := 12000
def B_investment : ℕ := 16000
def C_investment : ℕ := 20000
def total_profit : ℕ := 86400

theorem C_share_of_profit: 
  (C_investment / (A_investment + B_investment + C_investment) * total_profit) = 36000 :=
by
  sorry

end C_share_of_profit_l214_214867


namespace P_work_time_l214_214712

theorem P_work_time (T : ℝ) (hT : T > 0) : 
  (1 / T + 1 / 6 = 1 / 2.4) → T = 4 :=
by
  intros h
  sorry

end P_work_time_l214_214712


namespace original_polynomial_l214_214435

theorem original_polynomial {x y : ℝ} (P : ℝ) :
  P - (-x^2 * y) = 3 * x^2 * y - 2 * x * y - 1 → P = 2 * x^2 * y - 2 * x * y - 1 :=
sorry

end original_polynomial_l214_214435


namespace find_perpendicular_line_l214_214593

-- Define the point P
structure Point where
  x : ℤ
  y : ℤ

-- Define the line
structure Line where
  a : ℤ
  b : ℤ
  c : ℤ

-- The given problem conditions
def P : Point := { x := -1, y := 3 }

def given_line : Line := { a := 1, b := -2, c := 3 }

def perpendicular_line (line : Line) (point : Point) : Line :=
  ⟨ -line.b, line.a, -(line.a * point.y - line.b * point.x) ⟩

-- Theorem statement to prove
theorem find_perpendicular_line :
  perpendicular_line given_line P = { a := 2, b := 1, c := -1 } :=
by
  sorry

end find_perpendicular_line_l214_214593


namespace trace_ellipse_l214_214169

open Complex

theorem trace_ellipse (z : ℂ) (θ : ℝ) (h₁ : z = 3 * exp (θ * I))
  (h₂ : abs z = 3) : ∃ a b : ℝ, ∀ θ, z + 1/z = a * Real.cos θ + b * (I * Real.sin θ) :=
sorry

end trace_ellipse_l214_214169


namespace isabel_initial_candy_l214_214509

theorem isabel_initial_candy (total_candy : ℕ) (candy_given : ℕ) (initial_candy : ℕ) :
  candy_given = 25 → total_candy = 93 → total_candy = initial_candy + candy_given → initial_candy = 68 :=
by
  intros h_candy_given h_total_candy h_eq
  rw [h_candy_given, h_total_candy] at h_eq
  sorry

end isabel_initial_candy_l214_214509


namespace correct_average_l214_214264

theorem correct_average (initial_avg : ℝ) (n : ℕ) (error1 : ℝ) (wrong_num : ℝ) (correct_num : ℝ) :
  initial_avg = 40.2 → n = 10 → error1 = 19 → wrong_num = 13 → correct_num = 31 →
  (initial_avg * n - error1 - wrong_num + correct_num) / n = 40.1 :=
by
  intros
  sorry

end correct_average_l214_214264


namespace proof_problem_l214_214476

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)

theorem proof_problem (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) : f x * f (-x) = 1 := 
by 
  sorry

end proof_problem_l214_214476


namespace weekly_milk_production_l214_214306

-- Conditions
def number_of_cows : ℕ := 52
def milk_per_cow_per_day : ℕ := 1000
def days_in_week : ℕ := 7

-- Statement to prove
theorem weekly_milk_production : (number_of_cows * milk_per_cow_per_day * days_in_week) = 364000 := by
  sorry

end weekly_milk_production_l214_214306


namespace remainder_when_M_divided_by_32_l214_214945

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l214_214945


namespace intersection_M_N_l214_214526

def M : Set ℝ := { x : ℝ | 0 < x ∧ x < 4 }
def N : Set ℝ := { x : ℝ | (1 / 3) ≤ x ∧ x ≤ 5 }

theorem intersection_M_N : M ∩ N = { x : ℝ | (1 / 3) ≤ x ∧ x < 4 } :=
by
  sorry

end intersection_M_N_l214_214526


namespace min_odd_is_1_l214_214686

def min_odd_integers (a b c d e f : ℤ) : ℤ :=
  if (a + b) % 2 = 0 ∧ 
     (a + b + c + d) % 2 = 1 ∧ 
     (a + b + c + d + e + f) % 2 = 0 then
    1
  else
    sorry -- This should be replaced by a calculation of the true minimum based on conditions.

def satisfies_conditions (a b c d e f : ℤ) :=
  a + b = 30 ∧ 
  a + b + c + d = 47 ∧ 
  a + b + c + d + e + f = 65

theorem min_odd_is_1 (a b c d e f : ℤ) (h : satisfies_conditions a b c d e f) : 
  min_odd_integers a b c d e f = 1 := 
sorry

end min_odd_is_1_l214_214686


namespace intersection_of_sets_example_l214_214519

theorem intersection_of_sets_example :
  let M := { x : ℝ | 0 < x ∧ x < 4 }
  let N := { x : ℝ | 1 / 3 ≤ x ∧ x ≤ 5 }
  let expected := { x : ℝ | 1 / 3 ≤ x ∧ x < 4 }
  (M ∩ N) = expected :=
by
  sorry

end intersection_of_sets_example_l214_214519


namespace fraction_of_people_under_21_correct_l214_214572

variable (P : ℕ) (frac_over_65 : ℚ) (num_under_21 : ℕ) (frac_under_21 : ℚ)

def total_people_in_range (P : ℕ) : Prop := 50 < P ∧ P < 100

def fraction_of_people_over_65 (frac_over_65 : ℚ) : Prop := frac_over_65 = 5/12

def number_of_people_under_21 (num_under_21 : ℕ) : Prop := num_under_21 = 36

def fraction_of_people_under_21 (frac_under_21 : ℚ) : Prop := frac_under_21 = 3/7

theorem fraction_of_people_under_21_correct :
  ∀ (P : ℕ),
  total_people_in_range P →
  fraction_of_people_over_65 (5 / 12) →
  number_of_people_under_21 36 →
  P = 84 →
  fraction_of_people_under_21 (36 / P) :=
by
  intros P h_range h_over_65 h_under_21 h_P
  sorry

end fraction_of_people_under_21_correct_l214_214572


namespace computer_additions_per_hour_l214_214057

def operations_per_second : ℕ := 15000
def additions_per_second : ℕ := operations_per_second / 2
def seconds_per_hour : ℕ := 3600

theorem computer_additions_per_hour : 
  additions_per_second * seconds_per_hour = 27000000 := by
  sorry

end computer_additions_per_hour_l214_214057


namespace min_sum_of_factors_l214_214548

theorem min_sum_of_factors (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z = 1806) :
  x + y + z ≥ 72 := 
sorry

end min_sum_of_factors_l214_214548


namespace single_interval_condition_l214_214156

-- Definitions: k and l are integers
variables (k l : ℤ)

-- Condition: The given condition for l
theorem single_interval_condition : l = Int.floor (k ^ 2 / 4) :=
sorry

end single_interval_condition_l214_214156


namespace tim_bought_two_appetizers_l214_214837

-- Definitions of the conditions.
def total_spending : ℝ := 50
def portion_spent_on_entrees : ℝ := 0.80
def entree_cost : ℝ := total_spending * portion_spent_on_entrees
def appetizer_cost : ℝ := 5
def appetizer_spending : ℝ := total_spending - entree_cost

-- The statement to prove: that Tim bought 2 appetizers.
theorem tim_bought_two_appetizers :
  appetizer_spending / appetizer_cost = 2 := 
by
  sorry

end tim_bought_two_appetizers_l214_214837


namespace find_a_l214_214276

theorem find_a (a b c : ℤ) (h_vertex : ∀ x, (x - 2)*(x - 2) * a + 3 = a*x*x + b*x + c) 
  (h_point : (a*(3 - 2)*(3 -2) + 3 = 6)) : a = 3 :=
by
  sorry

end find_a_l214_214276


namespace find_N_l214_214737

theorem find_N (N : ℕ) (h1 : N > 1)
  (h2 : ∀ (a b c : ℕ), (a ≡ b [MOD N]) ∧ (b ≡ c [MOD N]) ∧ (a ≡ c [MOD N])
  → (a = 1743) ∧ (b = 2019) ∧ (c = 3008)) :
  N = 23 :=
by
  have h3 : Nat.gcd (2019 - 1743) (3008 - 2019) = 23 := sorry
  show N = 23 from sorry

end find_N_l214_214737


namespace number_of_australians_l214_214443

-- Conditions are given here as definitions
def total_people : ℕ := 49
def number_americans : ℕ := 16
def number_chinese : ℕ := 22

-- Goal is to prove the number of Australians is 11
theorem number_of_australians : total_people - (number_americans + number_chinese) = 11 := by
  sorry

end number_of_australians_l214_214443


namespace point_coordinates_l214_214679

def point : Type := ℝ × ℝ

def x_coordinate (P : point) : ℝ := P.1

def y_coordinate (P : point) : ℝ := P.2

theorem point_coordinates (P : point) (h1 : x_coordinate P = -3) (h2 : abs (y_coordinate P) = 5) :
  P = (-3, 5) ∨ P = (-3, -5) :=
by
  sorry

end point_coordinates_l214_214679


namespace donation_amount_is_correct_l214_214187

def stuffed_animals_barbara : ℕ := 9
def stuffed_animals_trish : ℕ := 2 * stuffed_animals_barbara
def stuffed_animals_sam : ℕ := stuffed_animals_barbara + 5
def stuffed_animals_linda : ℕ := stuffed_animals_sam - 7

def price_per_barbara : ℝ := 2
def price_per_trish : ℝ := 1.5
def price_per_sam : ℝ := 2.5
def price_per_linda : ℝ := 3

def total_amount_collected : ℝ := 
  stuffed_animals_barbara * price_per_barbara +
  stuffed_animals_trish * price_per_trish +
  stuffed_animals_sam * price_per_sam +
  stuffed_animals_linda * price_per_linda

def discount : ℝ := 0.10

def final_amount : ℝ := total_amount_collected * (1 - discount)

theorem donation_amount_is_correct : final_amount = 90.90 := sorry

end donation_amount_is_correct_l214_214187


namespace shopkeeper_discount_l214_214578

theorem shopkeeper_discount :
  let CP := 100
  let SP_with_discount := 119.7
  let SP_without_discount := 126
  let discount := SP_without_discount - SP_with_discount
  let discount_percentage := (discount / SP_without_discount) * 100
  discount_percentage = 5 := sorry

end shopkeeper_discount_l214_214578


namespace function_with_prop_M_l214_214915

noncomputable def prop_M (f : ℝ → ℝ) := ∀ x, (deriv (fun x => exp x * f x) x) ≥ 0

theorem function_with_prop_M :
  ∀ f ∈ ({(fun x : ℝ => 2^x): ℝ → ℝ, (fun x : ℝ => x^2): ℝ → ℝ, (fun x: ℝ => 3^(-x)): ℝ → ℝ, (fun x: ℝ => cos x): ℝ → ℝ}),
  f = (fun x => 2^x) ↔ prop_M f :=
by
  intros
  sorry

end function_with_prop_M_l214_214915


namespace tan_ratio_of_triangle_l214_214753

theorem tan_ratio_of_triangle (A B C : ℝ) (a b c : ℝ) (h1 : a * Real.cos B - b * Real.cos A = (3 / 5) * c) : 
  Real.tan A / Real.tan B = 4 :=
sorry

end tan_ratio_of_triangle_l214_214753


namespace triangle_ratio_perimeter_l214_214782

theorem triangle_ratio_perimeter (AC BC : ℝ) (CD : ℝ) (AB : ℝ) (m n : ℕ) :
  AC = 15 → BC = 20 → AB = 25 → CD = 10 * Real.sqrt 3 →
  gcd m n = 1 → (2 * Real.sqrt ((AC * BC) / AB) + AB) / AB = m / n → m + n = 7 :=
by
  intros hAC hBC hAB hCD hmn hratio
  sorry

end triangle_ratio_perimeter_l214_214782


namespace cost_price_is_975_l214_214302

-- Definitions from the conditions
def selling_price : ℝ := 1170
def profit_percentage : ℝ := 0.20

-- The proof statement
theorem cost_price_is_975 : (selling_price / (1 + profit_percentage)) = 975 := by
  sorry

end cost_price_is_975_l214_214302


namespace fraction_of_students_who_walk_l214_214063

def fraction_by_bus : ℚ := 2 / 5
def fraction_by_car : ℚ := 1 / 5
def fraction_by_scooter : ℚ := 1 / 8
def total_fraction_not_walk := fraction_by_bus + fraction_by_car + fraction_by_scooter

theorem fraction_of_students_who_walk :
  (1 - total_fraction_not_walk) = 11 / 40 :=
by
  sorry

end fraction_of_students_who_walk_l214_214063


namespace grid_square_division_l214_214110

theorem grid_square_division (m n k : ℕ) (h : m * m = n * k) : ℕ := sorry

end grid_square_division_l214_214110


namespace gcd_180_450_l214_214603

theorem gcd_180_450 : gcd 180 450 = 90 :=
by sorry

end gcd_180_450_l214_214603


namespace three_over_x_solution_l214_214770

theorem three_over_x_solution (x : ℝ) (h : 1 - 9 / x + 9 / (x^2) = 0) :
  3 / x = (3 - Real.sqrt 5) / 2 ∨ 3 / x = (3 + Real.sqrt 5) / 2 :=
by
  sorry

end three_over_x_solution_l214_214770


namespace positive_sqrt_729_l214_214629

theorem positive_sqrt_729 (x : ℝ) (h_pos : 0 < x) (h_eq : x^2 = 729) : x = 27 :=
by
  sorry

end positive_sqrt_729_l214_214629


namespace tournament_chromatic_index_l214_214716

noncomputable def chromaticIndex {n : ℕ} (k : ℕ) (h₁ : 2^(k-1) < n) (h₂ : n ≤ 2^k) : ℕ :=
k

theorem tournament_chromatic_index (n k : ℕ) (h₁ : 2^(k-1) < n) (h₂ : n ≤ 2^k) :
  chromaticIndex k h₁ h₂ = k :=
by sorry

end tournament_chromatic_index_l214_214716


namespace remainder_of_M_l214_214981

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l214_214981


namespace a_2021_2022_2023_product_l214_214085

noncomputable def a_seq : ℕ → ℚ
| 0       := 1 -- dummy value, since sequence defined from a_1
| 1       := (1 : ℚ) / 2
| (n + 2) := 1 / (1 - a_seq (n + 1))

theorem a_2021_2022_2023_product : a_seq 2021 * a_seq 2022 * a_seq 2023 = -1 :=
sorry

end a_2021_2022_2023_product_l214_214085


namespace average_bowling_score_l214_214908

theorem average_bowling_score 
    (gretchen_score : ℕ) (mitzi_score : ℕ) (beth_score : ℕ)
    (gretchen_eq : gretchen_score = 120)
    (mitzi_eq : mitzi_score = 113)
    (beth_eq : beth_score = 85) :
    (gretchen_score + mitzi_score + beth_score) / 3 = 106 := 
by
  sorry

end average_bowling_score_l214_214908


namespace composite_exists_for_x_64_l214_214477

-- Define the conditions
def is_composite (n : ℕ) : Prop := ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

-- Main statement
theorem composite_exists_for_x_64 :
  ∃ n : ℕ, is_composite (n^4 + 64) :=
sorry

end composite_exists_for_x_64_l214_214477


namespace remainder_when_divided_by_32_l214_214965

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l214_214965


namespace bank_policy_advantageous_for_retirees_l214_214399

theorem bank_policy_advantageous_for_retirees
  (special_programs : Prop)
  (higher_deposit_rates : Prop)
  (lower_credit_rates : Prop)
  (reliable_loan_payers : Prop)
  (stable_income : Prop)
  (family_interest : Prop)
  (savings_tendency : Prop)
  (regular_income : Prop)
  (long_term_deposits : Prop) :
  reliable_loan_payers ∧ stable_income ∧ family_interest ∧ savings_tendency ∧ regular_income ∧ long_term_deposits → 
  special_programs ∧ higher_deposit_rates ∧ lower_credit_rates :=
sorry

end bank_policy_advantageous_for_retirees_l214_214399


namespace remainder_of_M_when_divided_by_32_l214_214934

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l214_214934


namespace initial_pups_per_mouse_l214_214724

-- Definitions from the problem's conditions
def initial_mice : ℕ := 8
def stress_factor : ℕ := 2
def second_round_pups : ℕ := 6
def total_mice : ℕ := 280

-- Define a variable for the initial number of pups each mouse had
variable (P : ℕ)

-- Lean statement to prove the number of initial pups per mouse
theorem initial_pups_per_mouse (P : ℕ) (initial_mice stress_factor second_round_pups total_mice : ℕ) :
  total_mice = initial_mice + initial_mice * P + (initial_mice + initial_mice * P) * second_round_pups - stress_factor * (initial_mice + initial_mice * P) → 
  P = 6 := 
by
  sorry

end initial_pups_per_mouse_l214_214724


namespace sequence_general_formula_l214_214083

theorem sequence_general_formula (a : ℕ+ → ℝ) (h₀ : a 1 = 7 / 8)
  (h₁ : ∀ n : ℕ+, a (n + 1) = 1 / 2 * a n + 1 / 3) :
  ∀ n : ℕ+, a n = 5 / 24 * (1 / 2)^(n - 1 : ℕ) + 2 / 3 :=
by
  sorry

end sequence_general_formula_l214_214083


namespace avgPercentageSpentOnFoodCorrect_l214_214723

-- Definitions for given conditions
def JanuaryIncome : ℕ := 3000
def JanuaryPetrolExpenditure : ℕ := 300
def JanuaryHouseRentPercentage : ℕ := 14
def JanuaryClothingPercentage : ℕ := 10
def JanuaryUtilityBillsPercentage : ℕ := 5
def FebruaryIncome : ℕ := 4000
def FebruaryPetrolExpenditure : ℕ := 400
def FebruaryHouseRentPercentage : ℕ := 14
def FebruaryClothingPercentage : ℕ := 10
def FebruaryUtilityBillsPercentage : ℕ := 5

-- Calculate percentage spent on food over January and February
noncomputable def avgPercentageSpentOnFood : ℝ :=
  let totalIncome := (JanuaryIncome + FebruaryIncome: ℝ)
  let totalFoodExpenditure :=
    let remainingJan := (JanuaryIncome - JanuaryPetrolExpenditure: ℝ) 
                         - (JanuaryHouseRentPercentage / 100 * (JanuaryIncome - JanuaryPetrolExpenditure: ℝ))
                         - (JanuaryClothingPercentage / 100 * JanuaryIncome)
                         - (JanuaryUtilityBillsPercentage / 100 * JanuaryIncome)
    let remainingFeb := (FebruaryIncome - FebruaryPetrolExpenditure: ℝ)
                         - (FebruaryHouseRentPercentage / 100 * (FebruaryIncome - FebruaryPetrolExpenditure: ℝ))
                         - (FebruaryClothingPercentage / 100 * FebruaryIncome)
                         - (FebruaryUtilityBillsPercentage / 100 * FebruaryIncome)
    remainingJan + remainingFeb
  (totalFoodExpenditure / totalIncome) * 100

theorem avgPercentageSpentOnFoodCorrect : avgPercentageSpentOnFood = 62.4 := by
  sorry

end avgPercentageSpentOnFoodCorrect_l214_214723


namespace scientific_notation_of_0_000000032_l214_214829

theorem scientific_notation_of_0_000000032 :
  0.000000032 = 3.2 * 10^(-8) :=
by
  -- skipping the proof
  sorry

end scientific_notation_of_0_000000032_l214_214829


namespace find_value_of_a4_plus_a5_l214_214527

variables {S_n : ℕ → ℕ} {a_n : ℕ → ℕ} {d : ℤ} 

-- Conditions
def arithmetic_sequence_sum (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) (d : ℤ) : Prop :=
∀ n : ℕ, S_n n = n * a_n 1 + (n * (n - 1) / 2) * d

def a_3_S_3_condition (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) : Prop := 
a_n 3 = 3 ∧ S_n 3 = 3

-- Question
theorem find_value_of_a4_plus_a5 (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (d : ℤ):
  arithmetic_sequence_sum S_n a_n d →
  a_3_S_3_condition a_n S_n →
  a_n 4 + a_n 5 = 12 :=
by
  sorry

end find_value_of_a4_plus_a5_l214_214527


namespace common_difference_l214_214822

def Sn (S : Nat → ℝ) (n : Nat) : ℝ := S n

theorem common_difference (S : Nat → ℝ) (H : Sn S 2016 / 2016 = Sn S 2015 / 2015 + 1) : 2 = 2 := 
by
  sorry

end common_difference_l214_214822


namespace valid_license_plates_count_l214_214865

def num_valid_license_plates := (26 ^ 3) * (10 ^ 4)

theorem valid_license_plates_count : num_valid_license_plates = 175760000 :=
by
  sorry

end valid_license_plates_count_l214_214865


namespace combined_area_correct_l214_214311

noncomputable def breadth : ℝ := 20
noncomputable def length : ℝ := 1.15 * breadth
noncomputable def area_rectangle : ℝ := 460
noncomputable def radius_semicircle : ℝ := breadth / 2
noncomputable def area_semicircle : ℝ := (1/2) * Real.pi * radius_semicircle^2
noncomputable def combined_area : ℝ := area_rectangle + area_semicircle

theorem combined_area_correct : combined_area = 460 + 50 * Real.pi :=
by
  sorry

end combined_area_correct_l214_214311


namespace two_boys_same_price_l214_214556

open Real Finset

noncomputable def cats_20 := {r : ℝ | 12 ≤ r ∧ r ≤ 15 ∧ r ∈ {n/100 | n ∈ Icc 1200 1500}} .toFinset

noncomputable def sacks_20 := {r : ℝ | 0.10 ≤ r ∧ r ≤ 1 ∧ r ∈ {n/100 | n ∈ Icc 10 100}} .toFinset

theorem two_boys_same_price:
  ∃ (cat1 cat2 sack1 sack2 : ℝ), 
    cat1 ∈ cats_20 ∧ cat2 ∈ cats_20 ∧ sack1 ∈ sacks_20 ∧ sack2 ∈ sacks_20 ∧ 
    (cat1 + sack1 = cat2 + sack2) ∧ ¬ (cat1 = cat2 ∧ sack1 = sack2) :=
by
  sorry

end two_boys_same_price_l214_214556


namespace total_cookies_prepared_l214_214873

-- Definition of conditions
def cookies_per_guest : ℕ := 19
def number_of_guests : ℕ := 2

-- Theorem statement
theorem total_cookies_prepared : (cookies_per_guest * number_of_guests) = 38 :=
by
  sorry

end total_cookies_prepared_l214_214873


namespace problem_part1_problem_part2_l214_214486

open Real

noncomputable def f (x : ℝ) := (sqrt 3) * sin x * cos x - (1 / 2) * cos (2 * x)

theorem problem_part1 : 
  (∀ x : ℝ, -1 ≤ f x) ∧ 
  (∃ T : ℝ, (T > 0) ∧ ∀ x : ℝ, f (x + T) = f x ∧ T = π) := 
sorry

theorem problem_part2 (C A B c : ℝ) :
  (f C = 1) → 
  (B = π / 6) → 
  (c = 2 * sqrt 3) → 
  ∃ b : ℝ, ∃ area : ℝ, b = 2 ∧ area = (1 / 2) * b * c * sin A ∧ area = 2 * sqrt 3 := 
sorry

end problem_part1_problem_part2_l214_214486


namespace find_S_l214_214028

noncomputable def S : ℕ+ → ℝ := sorry
noncomputable def a : ℕ+ → ℝ := sorry

axiom h : ∀ n : ℕ+, 2 * S n = 3 * a n + 4

theorem find_S : ∀ n : ℕ+, S n = 2 - 2 * 3 ^ (n : ℕ) :=
  sorry

end find_S_l214_214028


namespace find_p_l214_214613

-- Definitions and conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_valid_configuration (p q s r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime s ∧ is_prime r ∧ 
  1 < p ∧ p < q ∧ q < s ∧ p + q + s = r

-- The theorem statement
theorem find_p (p q s r : ℕ) (h : is_valid_configuration p q s r) : p = 2 :=
by
  sorry

end find_p_l214_214613


namespace shared_friends_count_l214_214376

theorem shared_friends_count (james_friends : ℕ) (total_combined : ℕ) (john_factor : ℕ) 
  (h1 : james_friends = 75) 
  (h2 : john_factor = 3) 
  (h3 : total_combined = 275) : 
  james_friends + (john_factor * james_friends) - total_combined = 25 := 
by
  sorry

end shared_friends_count_l214_214376


namespace points_comparison_l214_214354

def quadratic_function (m x : ℝ) : ℝ :=
  (x + m - 3) * (x - m) + 3

def point_on_graph (m x y : ℝ) : Prop :=
  y = quadratic_function m x

theorem points_comparison (m x1 x2 y1 y2 : ℝ)
  (h1 : point_on_graph m x1 y1)
  (h2 : point_on_graph m x2 y2)
  (hx : x1 < x2)
  (h_sum : x1 + x2 < 3) :
  y1 > y2 := 
  sorry

end points_comparison_l214_214354


namespace remainder_when_divided_by_32_l214_214970

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l214_214970


namespace polynomial_coefficient_sum_l214_214096

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (1 - 2 * x) ^ 4 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 →
  a₀ + a₁ + a₃ = -39 :=
by
  sorry

end polynomial_coefficient_sum_l214_214096


namespace minimum_n_for_3_zeros_l214_214571

theorem minimum_n_for_3_zeros :
  ∃ n : ℕ, (∀ m : ℕ, (m < n → ∀ k < 10, m + k ≠ 5 * m ∧ m + k ≠ 5 * m + 25)) ∧
  (∀ k < 10, n + k = 16 ∨ n + k = 16 + 9) ∧
  n = 16 :=
sorry

end minimum_n_for_3_zeros_l214_214571


namespace hypotenuse_is_18_point_8_l214_214313

def hypotenuse_of_right_triangle (a b c : ℝ) : Prop :=
  a + b + c = 40 ∧ (1/2) * a * b = 24 ∧ a^2 + b^2 = c^2

theorem hypotenuse_is_18_point_8 (a b c : ℝ) (h : hypotenuse_of_right_triangle a b c) : c = 18.8 :=
  sorry

end hypotenuse_is_18_point_8_l214_214313


namespace determine_a_l214_214749

noncomputable def polynomial_factorization (a : ℝ) : Prop :=
  ∃ b : ℝ, (y^2 + 3 * y - a) = (y - 3) * (y + b)

theorem determine_a (a : ℝ) :
  polynomial_factorization a → a = 18 :=
by
  -- Need proof here
  sorry

end determine_a_l214_214749


namespace remainder_13_pow_150_mod_11_l214_214157

theorem remainder_13_pow_150_mod_11 : (13^150) % 11 = 1 := 
by 
  sorry

end remainder_13_pow_150_mod_11_l214_214157


namespace sum_of_roots_l214_214043

theorem sum_of_roots (z1 z2 : ℂ) (h : z1^2 + 5*z1 - 14 = 0 ∧ z2^2 + 5*z2 - 14 = 0) :
  z1 + z2 = -5 :=
sorry

end sum_of_roots_l214_214043


namespace factorization_a_minus_b_l214_214396

-- Define the problem in Lean 4
theorem factorization_a_minus_b (a b : ℤ) : 
  (∀ y : ℝ, 2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b))
  → a - b = 7 := 
by 
  sorry

end factorization_a_minus_b_l214_214396


namespace union_card_ge_165_l214_214930

open Finset

variable (A : Finset ℕ) (A_i : Fin (11) → Finset ℕ)
variable (hA : A.card = 225)
variable (hA_i_card : ∀ i, (A_i i).card = 45)
variable (hA_i_intersect : ∀ i j, i < j → ((A_i i) ∩ (A_i j)).card = 9)

theorem union_card_ge_165 : (Finset.biUnion Finset.univ A_i).card ≥ 165 := by sorry

end union_card_ge_165_l214_214930


namespace russel_carousel_rides_l214_214535

variable (tickets_used : Nat) (tickets_shooting : Nat) (tickets_carousel : Nat)
variable (total_tickets : Nat)
variable (times_shooting : Nat)

theorem russel_carousel_rides :
    times_shooting = 2 →
    tickets_shooting = 5 →
    tickets_carousel = 3 →
    total_tickets = 19 →
    tickets_used = total_tickets - (times_shooting * tickets_shooting) →
    tickets_used / tickets_carousel = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end russel_carousel_rides_l214_214535


namespace Roberta_spent_on_shoes_l214_214660

-- Define the conditions as per the problem statement
variables (S B L : ℝ) (h1 : B = S - 17) (h2 : L = B / 4) (h3 : 158 - (S + B + L) = 78)

-- State the theorem to be proved
theorem Roberta_spent_on_shoes : S = 45 :=
by
  -- use variables and conditions
  have := h1
  have := h2
  have := h3
  sorry -- Proof steps can be filled later

end Roberta_spent_on_shoes_l214_214660


namespace max_intersections_three_circles_two_lines_l214_214690

noncomputable def max_intersections_3_circles_2_lines : ℕ :=
  3 * 2 * 1 + 2 * 3 * 2 + 1

theorem max_intersections_three_circles_two_lines :
  max_intersections_3_circles_2_lines = 19 :=
by
  sorry

end max_intersections_three_circles_two_lines_l214_214690


namespace parabola_functions_eq_l214_214658

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := x^2 + b * x + c
noncomputable def g (x : ℝ) (c : ℝ) (b : ℝ) : ℝ := x^2 + c * x + b

theorem parabola_functions_eq : ∀ (x₁ x₂ : ℝ), 
  (∃ t : ℝ, (f t b c = g t c b) ∧ (t = 1)) → 
    (f x₁ 2 (-3) = x₁^2 + 2 * x₁ - 3) ∧ (g x₂ (-3) 2 = x₂^2 - 3 * x₂ + 2) :=
sorry

end parabola_functions_eq_l214_214658


namespace product_of_odd_primes_mod_32_l214_214959

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l214_214959


namespace more_larger_boxes_l214_214448

theorem more_larger_boxes (S L : ℕ) 
  (h1 : 12 * S + 16 * L = 480)
  (h2 : S + L = 32)
  (h3 : L > S) : L - S = 16 := 
sorry

end more_larger_boxes_l214_214448


namespace bounded_area_l214_214458

noncomputable def f (x : ℝ) : ℝ := (x + Real.sqrt (x^2 + 1))^(1/3) + (x - Real.sqrt (x^2 + 1))^(1/3)

def g (y : ℝ) : ℝ := y + 1

theorem bounded_area : 
  (∫ y in (0:ℝ)..(1:ℝ), (g y - f (g y))) = (5/8 : ℝ) := by
  sorry

end bounded_area_l214_214458


namespace periodic_symmetry_mono_f_l214_214855

-- Let f be a function from ℝ to ℝ.
variable (f : ℝ → ℝ)

-- f has the domain of ℝ.
-- f(x) = f(x + 6) for all x ∈ ℝ.
axiom periodic_f : ∀ x : ℝ, f x = f (x + 6)

-- f is monotonically decreasing in (0, 3).
axiom mono_f : ∀ ⦃x y : ℝ⦄, 0 < x → x < y → y < 3 → f y < f x

-- The graph of f is symmetric about the line x = 3.
axiom symmetry_f : ∀ x : ℝ, f x = f (6 - x)

-- Prove that f(3.5) < f(1.5) < f(6.5).
theorem periodic_symmetry_mono_f : f 3.5 < f 1.5 ∧ f 1.5 < f 6.5 :=
sorry

end periodic_symmetry_mono_f_l214_214855


namespace integer_solution_for_system_l214_214536

theorem integer_solution_for_system 
    (x y z : ℕ) 
    (h1 : 3 * x - 4 * y + 5 * z = 10) 
    (h2 : 7 * y + 8 * x - 3 * z = 13) : 
    x = 1 ∧ y = 2 ∧ z = 3 :=
by 
  sorry

end integer_solution_for_system_l214_214536


namespace vector_calculation_l214_214882

def v1 : ℝ × ℝ := (3, -5)
def v2 : ℝ × ℝ := (-1, 6)
def v3 : ℝ × ℝ := (2, -1)

theorem vector_calculation :
  (5:ℝ) • v1 - (3:ℝ) • v2 + v3 = (20, -44) :=
by
  sorry

end vector_calculation_l214_214882


namespace bookstore_floor_l214_214016

theorem bookstore_floor
  (academy_floor : ℤ)
  (reading_room_floor : ℤ)
  (bookstore_floor : ℤ)
  (h1 : academy_floor = 7)
  (h2 : reading_room_floor = academy_floor + 4)
  (h3 : bookstore_floor = reading_room_floor - 9) :
  bookstore_floor = 2 :=
by
  sorry

end bookstore_floor_l214_214016


namespace combined_area_of_walls_l214_214150

theorem combined_area_of_walls (A : ℕ) 
  (h1: ∃ (A : ℕ), A ≥ 0)
  (h2 : (A - 2 * 40 - 40 = 180)) :
  A = 300 := 
sorry

end combined_area_of_walls_l214_214150


namespace area_of_original_triangle_l214_214177

variable (H : ℝ) (H' : ℝ := 0.65 * H) 
variable (A' : ℝ := 14.365)
variable (k : ℝ := 0.65) 
variable (A : ℝ)

theorem area_of_original_triangle (h₁ : H' = k * H) (h₂ : A' = 14.365) (h₃ : k = 0.65) : A = 34 := by
  sorry

end area_of_original_triangle_l214_214177


namespace gcd_180_450_l214_214597

theorem gcd_180_450 : Nat.gcd 180 450 = 90 :=
by
  sorry

end gcd_180_450_l214_214597


namespace tent_cost_solution_l214_214155

-- We define the prices of the tents and other relevant conditions.
def tent_costs (m n : ℕ) : Prop :=
  2 * m + 4 * n = 5200 ∧ 3 * m + n = 2800

-- Define the condition for the number of tents and constraints.
def optimal_tent_count (x : ℕ) (w : ℕ) : Prop :=
  x + (20 - x) = 20 ∧ x ≤ (20 - x) / 3 ∧ w = 600 * x + 1000 * (20 - x)

-- The main theorem to be proven in Lean.
theorem tent_cost_solution :
  ∃ m n, tent_costs m n ∧ m = 600 ∧ n = 1000 ∧
  ∃ x, optimal_tent_count x 18000 ∧ x = 5 ∧ (20 - x) = 15 :=
by
  sorry

end tent_cost_solution_l214_214155


namespace find_c_l214_214067

-- Definitions for the conditions
def line1 (x y : ℝ) : Prop := 4 * y + 2 * x + 6 = 0
def line2 (x y : ℝ) (c : ℝ) : Prop := 5 * y + c * x + 4 = 0
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Main theorem
theorem find_c (c : ℝ) : 
  (∀ x y : ℝ, line1 x y → y = -1/2 * x - 3/2) ∧ 
  (∀ x y : ℝ, line2 x y c → y = -c/5 * x - 4/5) ∧ 
  perpendicular (-1/2) (-c/5) → 
  c = -10 := by
  sorry

end find_c_l214_214067


namespace neg_p_l214_214093

theorem neg_p :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by
  sorry

end neg_p_l214_214093


namespace jane_performance_l214_214792

theorem jane_performance :
  ∃ (p w e : ℕ), 
  p + w + e = 15 ∧ 
  2 * p + 4 * w + 6 * e = 66 ∧ 
  e = p + 4 ∧ 
  w = 11 :=
by
  sorry

end jane_performance_l214_214792


namespace geometric_sequence_sum_l214_214231

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum {a : ℕ → ℝ}
  (h_geo : is_geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_condition : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) :
  a 3 + a 5 = 5 := 
sorry

end geometric_sequence_sum_l214_214231


namespace toy_factory_max_profit_l214_214182

theorem toy_factory_max_profit :
  ∃ x y : ℕ,    -- x: number of bears, y: number of cats
  15 * x + 10 * y ≤ 450 ∧    -- labor hours constraint
  20 * x + 5 * y ≤ 400 ∧     -- raw materials constraint
  80 * x + 45 * y = 2200 :=  -- total selling price
by
  sorry

end toy_factory_max_profit_l214_214182


namespace right_triangle_area_l214_214101

theorem right_triangle_area (x : ℝ) (h : 3 * x + 4 * x = 10) : 
  (1 / 2) * (3 * x) * (4 * x) = 24 :=
sorry

end right_triangle_area_l214_214101


namespace jackson_has_1900_more_than_brandon_l214_214790

-- Conditions
def initial_investment : ℝ := 500
def jackson_multiplier : ℝ := 4
def brandon_multiplier : ℝ := 0.20

-- Final values
def jackson_final_value := jackson_multiplier * initial_investment
def brandon_final_value := brandon_multiplier * initial_investment

-- Statement to prove the difference
theorem jackson_has_1900_more_than_brandon : jackson_final_value - brandon_final_value = 1900 := 
    by sorry

end jackson_has_1900_more_than_brandon_l214_214790


namespace correct_average_marks_l214_214851

def incorrect_average := 100
def number_of_students := 10
def incorrect_mark := 60
def correct_mark := 10
def difference := incorrect_mark - correct_mark
def incorrect_total := incorrect_average * number_of_students
def correct_total := incorrect_total - difference

theorem correct_average_marks : correct_total / number_of_students = 95 := by
  sorry

end correct_average_marks_l214_214851


namespace pirate_prob_exactly_four_treasure_no_traps_l214_214178

open ProbabilityMeasure

def probability_island_treasure_no_traps : ℚ := 3/10
def probability_island_traps_no_treasure : ℚ := 1/10
def probability_island_both_treasure_traps : ℚ := 1/5
def probability_island_neither_traps_nor_treasure : ℚ := 2/5

theorem pirate_prob_exactly_four_treasure_no_traps :
  ∀ (total_islands : ℕ) (islands_with_treasure_no_traps : ℕ),
    total_islands = 8 →
    islands_with_treasure_no_traps = 4 →
    (choose total_islands islands_with_treasure_no_traps : ℚ) *
    (probability_island_treasure_no_traps ^ islands_with_treasure_no_traps) *
    (probability_island_neither_traps_nor_treasure ^ (total_islands - islands_with_treasure_no_traps)) =
    9072 / 6250000 := sorry

end pirate_prob_exactly_four_treasure_no_traps_l214_214178


namespace elder_child_age_l214_214831

theorem elder_child_age (x : ℕ) (h : x + (x + 4) + (x + 8) + (x + 12) = 48) : (x + 12) = 18 :=
by
  sorry

end elder_child_age_l214_214831


namespace A_alone_time_l214_214172

theorem A_alone_time (x : ℕ) (h1 : 3 * x / 4  = 12) : x / 3 = 16 := by
  sorry

end A_alone_time_l214_214172


namespace distance_from_Asheville_to_Darlington_l214_214868

theorem distance_from_Asheville_to_Darlington (BC AC BD AD : ℝ) 
(h0 : BC = 12) 
(h1 : BC = (1/3) * AC) 
(h2 : BC = (1/4) * BD) :
AD = 72 :=
sorry

end distance_from_Asheville_to_Darlington_l214_214868


namespace selection_methods_l214_214318

theorem selection_methods (fitters turners master_workers : Nat)
  (h_fitters : fitters = 5)
  (h_turners : turners = 4)
  (h_master_workers : master_workers = 2) :
  ∃ (methods : Nat), methods = 185 :=
by
  let C := Nat.choose
  have scenario1 : Nat := C 7 4
  have scenario2 : Nat := C 4 3 * C 2 1 * C 6 4
  have scenario3 : Nat := C 4 2 * C 5 4 * C 2 2
  let total_methods := scenario1 + scenario2 + scenario3
  use total_methods
  sorry

end selection_methods_l214_214318


namespace find_f_107_l214_214760

-- Define the conditions
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 3) = -f x

def piecewise_function (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x / 5

-- Main theorem to prove based on the conditions
theorem find_f_107 (f : ℝ → ℝ)
  (h_periodic : periodic_function f)
  (h_piece : piecewise_function f)
  (h_even : even_function f) : f 107 = 1 / 5 :=
sorry

end find_f_107_l214_214760


namespace relationship_a_e_l214_214480

theorem relationship_a_e (a : ℝ) (h : 0 < a ∧ a < 1) : a < Real.exp a - 1 ∧ Real.exp a - 1 < a ^ Real.exp 1 := by
  sorry

end relationship_a_e_l214_214480


namespace number_of_strawberries_in_each_basket_l214_214125

variable (x : ℕ) (Lilibeth_picks : 6 * x)
variable (total_strawberries : 4 * 6 * x = 1200)

theorem number_of_strawberries_in_each_basket : x = 50 := by
  sorry

end number_of_strawberries_in_each_basket_l214_214125


namespace lcm_factor_l214_214670

-- Define the variables and conditions
variables (A B H L x : ℕ)
variable (hcf_23 : Nat.gcd A B = 23)
variable (larger_number_391 : A = 391)
variable (lcm_hcf_mult_factors : L = Nat.lcm A B)
variable (lcm_factors : L = 23 * x * 17)

-- The proof statement
theorem lcm_factor (hcf_23 : Nat.gcd A B = 23) (larger_number_391 : A = 391) (lcm_hcf_mult_factors : L = Nat.lcm A B) (lcm_factors : L = 23 * x * 17) :
  x = 17 :=
sorry

end lcm_factor_l214_214670


namespace cost_of_pants_is_250_l214_214876

variable (costTotal : ℕ) (costTShirt : ℕ) (numTShirts : ℕ) (numPants : ℕ)

def costPants (costTotal costTShirt numTShirts numPants : ℕ) : ℕ :=
  let costTShirts := numTShirts * costTShirt
  let costPantsTotal := costTotal - costTShirts
  costPantsTotal / numPants

-- Given conditions
axiom h1 : costTotal = 1500
axiom h2 : costTShirt = 100
axiom h3 : numTShirts = 5
axiom h4 : numPants = 4

-- Prove each pair of pants costs $250
theorem cost_of_pants_is_250 : costPants costTotal costTShirt numTShirts numPants = 250 :=
by
  -- Place proof here
  sorry

end cost_of_pants_is_250_l214_214876


namespace original_mixture_percentage_l214_214299

variables (a w : ℝ)

-- Conditions given
def condition1 : Prop := a / (a + w + 2) = 0.3
def condition2 : Prop := (a + 2) / (a + w + 4) = 0.4

theorem original_mixture_percentage (h1 : condition1 a w) (h2 : condition2 a w) : (a / (a + w)) * 100 = 36 :=
by
sorry

end original_mixture_percentage_l214_214299


namespace gcd_180_450_l214_214596

theorem gcd_180_450 : Nat.gcd 180 450 = 90 :=
by
  sorry

end gcd_180_450_l214_214596


namespace hyperbola_sum_l214_214728

theorem hyperbola_sum
  (h k a b : ℝ)
  (center : h = 3 ∧ k = 1)
  (vertex : ∃ (v : ℝ), (v = 4 ∧ h = 3 ∧ a = |k - v|))
  (focus : ∃ (f : ℝ), (f = 10 ∧ h = 3 ∧ (f - k) = 9 ∧ ∃ (c : ℝ), c = |k - f|))
  (relationship : ∀ (c : ℝ), c = 9 → c^2 = a^2 + b^2): 
  h + k + a + b = 7 + 6 * Real.sqrt 2 :=
by 
  sorry

end hyperbola_sum_l214_214728


namespace solve_quadratic_eq_l214_214538

theorem solve_quadratic_eq (x : ℝ) : x^2 - 2 * x - 15 = 0 ↔ (x = -3 ∨ x = 5) :=
by
  sorry

end solve_quadratic_eq_l214_214538


namespace not_prime_sum_l214_214236

theorem not_prime_sum (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_square : ∃ k : ℕ, a^2 - b * c = k^2) : ¬ Nat.Prime (2 * a + b + c) := 
sorry

end not_prime_sum_l214_214236


namespace line_through_P_perpendicular_l214_214592

theorem line_through_P_perpendicular 
  (P : ℝ × ℝ) (a b c : ℝ) (hP : P = (-1, 3)) (hline : a = 1 ∧ b = -2 ∧ c = 3) :
  ∃ (a' b' c' : ℝ), (a' * P.1 + b' * P.2 + c' = 0) ∧ (a = b' ∧ b = -a') ∧ (a' = 2 ∧ b' = 1 ∧ c' = -1) := 
by
  use 2, 1, -1
  sorry

end line_through_P_perpendicular_l214_214592


namespace rectangle_width_is_3_l214_214541

-- Define the given conditions
def length_square : ℝ := 9
def length_rectangle : ℝ := 27

-- Calculate the area based on the given conditions
def area_square : ℝ := length_square * length_square

-- Define the area equality condition
def area_equality (width_rectangle : ℝ) : Prop :=
  area_square = length_rectangle * width_rectangle

-- The theorem stating the width of the rectangle
theorem rectangle_width_is_3 (width_rectangle: ℝ) :
  area_equality width_rectangle → width_rectangle = 3 :=
by
  -- Skipping the proof itself as instructed
  intro h
  sorry

end rectangle_width_is_3_l214_214541


namespace expansion_dissimilar_terms_count_l214_214372

def number_of_dissimilar_terms (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

theorem expansion_dissimilar_terms_count :
  number_of_dissimilar_terms 7 4 = 120 := by
  sorry

end expansion_dissimilar_terms_count_l214_214372


namespace find_a3_l214_214785

variable {α : Type*} [AddCommGroup α] [Module ℤ α]

noncomputable
def arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem find_a3 {a : ℕ → ℤ} (d : ℤ) (h6 : a 6 = 6) (h9 : a 9 = 9) :
  (∃ d : ℤ, arithmetic_sequence a d) →
  a 3 = 3 :=
by
  intro h_arith_seq
  sorry

end find_a3_l214_214785


namespace right_triangle_side_lengths_l214_214192

theorem right_triangle_side_lengths (x : ℝ) :
  (2 * x + 2)^2 + (x + 2)^2 = (x + 4)^2 ∨ (2 * x + 2)^2 + (x + 4)^2 = (x + 2)^2 ↔ (x = 1 ∨ x = 4) :=
by sorry

end right_triangle_side_lengths_l214_214192


namespace intersection_of_sets_example_l214_214521

theorem intersection_of_sets_example :
  let M := { x : ℝ | 0 < x ∧ x < 4 }
  let N := { x : ℝ | 1 / 3 ≤ x ∧ x ≤ 5 }
  let expected := { x : ℝ | 1 / 3 ≤ x ∧ x < 4 }
  (M ∩ N) = expected :=
by
  sorry

end intersection_of_sets_example_l214_214521


namespace new_person_weight_l214_214161

theorem new_person_weight (W : ℝ) (N : ℝ) (avg_increase : ℝ := 2.5) (replaced_weight : ℝ := 35) :
  (W - replaced_weight + N) = (W + (8 * avg_increase)) → N = 55 := sorry

end new_person_weight_l214_214161


namespace range_of_k_if_f_monotonically_increasing_l214_214633

noncomputable def f (k x : ℝ) : ℝ := k * x - Real.log x

theorem range_of_k_if_f_monotonically_increasing :
  (∀ (x : ℝ), 1 < x → 0 ≤ (k - 1 / x)) → k ∈ Set.Ici (1: ℝ) :=
by
  intro hyp
  have : ∀ (x : ℝ), 1 < x → 0 ≤ k - 1 / x := hyp
  sorry

end range_of_k_if_f_monotonically_increasing_l214_214633


namespace lcm_36_65_l214_214607

-- Definitions based on conditions
def number1 : ℕ := 36
def number2 : ℕ := 65

-- The prime factorization conditions can be implied through deriving LCM hence added as comments to clarify the conditions.
-- 36 = 2^2 * 3^2
-- 65 = 5 * 13

-- Theorem statement that the LCM of number1 and number2 is 2340
theorem lcm_36_65 : Nat.lcm number1 number2 = 2340 := 
by 
  sorry

end lcm_36_65_l214_214607


namespace distance_from_pointM_to_xaxis_l214_214823

-- Define the point M with coordinates (2, -3)
def pointM : ℝ × ℝ := (2, -3)

-- Define the function to compute the distance from a point to the x-axis.
def distanceToXAxis (p : ℝ × ℝ) : ℝ := |p.2|

-- Formalize the proof statement.
theorem distance_from_pointM_to_xaxis : distanceToXAxis pointM = 3 := by
  -- Proof goes here
  sorry

end distance_from_pointM_to_xaxis_l214_214823


namespace swap_numbers_l214_214014

-- Define the initial state
variables (a b c : ℕ)
axiom initial_state : a = 8 ∧ b = 17

-- Define the assignment sequence
axiom swap_statement1 : c = b 
axiom swap_statement2 : b = a
axiom swap_statement3 : a = c

-- Define the theorem to be proved
theorem swap_numbers (a b c : ℕ) (initial_state : a = 8 ∧ b = 17)
  (swap_statement1 : c = b) (swap_statement2 : b = a) (swap_statement3 : a = c) :
  (a = 17 ∧ b = 8) :=
sorry

end swap_numbers_l214_214014


namespace smallest_positive_angle_l214_214065

theorem smallest_positive_angle :
  ∃ y : ℝ, 0 < y ∧ y < 90 ∧ (6 * Real.sin y * (Real.cos y)^3 - 6 * (Real.sin y)^3 * Real.cos y = 3 / 2) ∧ y = 22.5 :=
by
  sorry

end smallest_positive_angle_l214_214065


namespace product_of_odd_primes_mod_32_l214_214957

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l214_214957


namespace intersection_M_N_l214_214524

def M : Set ℝ := { x : ℝ | 0 < x ∧ x < 4 }
def N : Set ℝ := { x : ℝ | (1 / 3) ≤ x ∧ x ≤ 5 }

theorem intersection_M_N : M ∩ N = { x : ℝ | (1 / 3) ≤ x ∧ x < 4 } :=
by
  sorry

end intersection_M_N_l214_214524


namespace intersection_M_N_l214_214523

def M : Set ℝ := { x : ℝ | 0 < x ∧ x < 4 }
def N : Set ℝ := { x : ℝ | (1 / 3) ≤ x ∧ x ≤ 5 }

theorem intersection_M_N : M ∩ N = { x : ℝ | (1 / 3) ≤ x ∧ x < 4 } :=
by
  sorry

end intersection_M_N_l214_214523


namespace final_probability_l214_214361

def total_cards := 52
def kings := 4
def aces := 4
def chosen_cards := 3

namespace probability

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

def prob_three_kings : ℚ :=
  (4 / 52) * (3 / 51) * (2 / 50)

def prob_exactly_two_aces : ℚ :=
  (choose 4 2 * choose 48 1) / choose 52 3

def prob_exactly_three_aces : ℚ :=
  (choose 4 3) / choose 52 3

def prob_at_least_two_aces : ℚ :=
  prob_exactly_two_aces + prob_exactly_three_aces

def prob_three_kings_or_two_aces : ℚ :=
  prob_three_kings + prob_at_least_two_aces

theorem final_probability :
  prob_three_kings_or_two_aces = 6 / 425 :=
by
  sorry

end probability

end final_probability_l214_214361


namespace point_has_zero_measure_measure_disjoint_intervals_measure_union_intervals_rationals_have_zero_measure_measure_even_integer_part_measure_translation_invariant_measure_no_digit_9_has_zero_measure_homothety_measure_l214_214703

open MeasureTheory

-- 1. Prove that a point has zero measure.
theorem point_has_zero_measure (x : ℝ) : measure_of_Ixx volume (Ioo x x) = 0 :=
sorry

-- 2. What is the measure of [0, 1[ ∩ ]3/2, 4]?
theorem measure_disjoint_intervals : measure_of_Ixx volume (Ico 0 1 ∩ Ioc (3/2) 4) = 0 :=
sorry

-- 3. What is the measure of ∪_{n=0}^∞ [2^{-2n}, 2^{-2n-1}]
theorem measure_union_intervals : measure_of_Ixx volume (⋃ n, Icc (Real.exp (-(2*n))) (Real.exp (-(2*n) - 1))) = 2/3 :=
sorry

-- 4. Prove that ℚ has zero measure.
theorem rationals_have_zero_measure : measure_of_Ixx volume (⋃ q ∈ ℚ, Ioo q q) = 0 :=
sorry

-- 5. What is the measure of the set of numbers whose integer part is even?
def even_integer_part (x : ℝ) : Prop := ∃ k : ℤ, x ∈ Ico (2*k : ℝ) (2*k + 1)
theorem measure_even_integer_part : measure_of_Ixx volume {x | even_integer_part x} = ∞ :=
sorry

-- 6. Prove that the measure is invariant under translation.
theorem measure_translation_invariant (A : Set ℝ) (t : ℝ) : measurable_set A → measure_of_Ixx volume (A + t) = measure_of_Ixx volume A :=
sorry

-- 7. Prove that the set of numbers whose decimal representation does not contain the digit 9 has zero measure.
theorem measure_no_digit_9_has_zero_measure : measure_of_Ixx volume {x : ℝ | ¬ 9 ∈ (to_digits (floor x))} = 0 :=
sorry

-- 8. Prove that if A is a measurable set and λ is a real number, then λA has measure λ times the measure of A.
theorem homothety_measure (A : Set ℝ) (λ : ℝ) : measurable_set A → measure_of_Ixx volume (λ • A) = λ * measure_of_Ixx volume A :=
sorry

end point_has_zero_measure_measure_disjoint_intervals_measure_union_intervals_rationals_have_zero_measure_measure_even_integer_part_measure_translation_invariant_measure_no_digit_9_has_zero_measure_homothety_measure_l214_214703


namespace calculate_expression_l214_214878

theorem calculate_expression : 16^4 * 8^2 / 4^12 = (1 : ℚ) / 4 := by
  sorry

end calculate_expression_l214_214878


namespace prime_product_mod_32_l214_214972

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l214_214972


namespace gcd_840_1764_l214_214825

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l214_214825


namespace earnings_difference_l214_214846

-- Definitions:
def investments_ratio := (3, 4, 5)
def return_ratio := (6, 5, 4)
def total_earnings := 5800

-- Target statement:
theorem earnings_difference (x y : ℝ)
  (h_investment_ratio : investments_ratio = (3, 4, 5))
  (h_return_ratio : return_ratio = (6, 5, 4))
  (h_total_earnings : (3 * x * 6 * y) / 100 + (4 * x * 5 * y) / 100 + (5 * x * 4 * y) / 100 = total_earnings) :
  ((4 * x * 5 * y) / 100 - (3 * x * 6 * y) / 100) = 200 := 
by
  sorry

end earnings_difference_l214_214846


namespace work_problem_l214_214697

theorem work_problem (W : ℝ) (A_rate : ℝ) (AB_rate : ℝ) : A_rate = W / 14 ∧ AB_rate = W / 10 → 1 / (AB_rate - A_rate) = 35 :=
by
  sorry

end work_problem_l214_214697


namespace solve_equation_l214_214454

theorem solve_equation (a b : ℕ) : 
  (a^2 = b * (b + 7) ∧ a ≥ 0 ∧ b ≥ 0) ↔ (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) := by
  sorry

end solve_equation_l214_214454


namespace initial_acorns_l214_214808

theorem initial_acorns (T : ℝ) (h1 : 0.35 * T = 7) (h2 : 0.45 * T = 9) : T = 20 :=
sorry

end initial_acorns_l214_214808


namespace triangle_area_eq_l214_214788

/-- In a triangle ABC, given that A = arccos(7/8), BC = a, and the altitude from vertex A 
     is equal to the sum of the other two altitudes, show that the area of triangle ABC 
     is (a^2 * sqrt(15)) / 4. -/
theorem triangle_area_eq (a : ℝ) (angle_A : ℝ) (h_angle : angle_A = Real.arccos (7/8))
    (BC : ℝ) (h_BC : BC = a) (H : ∀ (AC AB altitude_A altitude_C altitude_B : ℝ),
    AC = X → AB = Y → 
    altitude_A = (altitude_C + altitude_B) → 
    ∃ (S : ℝ), 
    S = (1/2) * X * Y * Real.sin angle_A ∧ 
    altitude_A = (2 * S / X) + (2 * S / Y) 
    → (X * Y) = 4 * (a^2) 
    → S = ((a^2 * Real.sqrt 15) / 4)) :
S = (a^2 * Real.sqrt 15) / 4 := sorry

end triangle_area_eq_l214_214788


namespace ratio_of_small_square_to_shaded_area_l214_214266

theorem ratio_of_small_square_to_shaded_area :
  let small_square_area := 2 * 2
  let large_square_area := 5 * 5
  let shaded_area := (large_square_area / 2) - (small_square_area / 2)
  (small_square_area : ℚ) / shaded_area = 8 / 21 :=
by
  sorry

end ratio_of_small_square_to_shaded_area_l214_214266


namespace no_integer_solutions_l214_214893

theorem no_integer_solutions (x y z : ℤ) : ¬ (4 * x^2 + 77 * y^2 = 487 * z^2) :=
by {
  -- The proof would go here, but for the generated statement, we leave it as a sorry
  sorry
}

end no_integer_solutions_l214_214893


namespace rectangle_x_is_18_l214_214479

-- Definitions for the conditions
def rectangle (a b x : ℕ) : Prop := 
  (a = 2 * b) ∧
  (x = 2 * (a + b)) ∧
  (x = a * b)

-- Theorem to prove the equivalence of the conditions and the answer \( x = 18 \)
theorem rectangle_x_is_18 : ∀ a b x : ℕ, rectangle a b x → x = 18 :=
by
  sorry

end rectangle_x_is_18_l214_214479


namespace factor_expression_l214_214050

variable (x y : ℝ)

theorem factor_expression :
  4 * x ^ 2 - 4 * x - y ^ 2 + 4 * y - 3 = (2 * x + y - 3) * (2 * x - y + 1) := by
  sorry

end factor_expression_l214_214050


namespace part_1_part_2_l214_214082

theorem part_1 (a : ℕ → ℚ) (S : ℕ → ℚ) (h1 : ∀ n, S (n + 1) = 4 * a n - 2) (h2 : a 1 = 2) (n : ℕ) (hn_pos : 0 < n) : 
  a (n + 1) - 2 * a n = 0 :=
sorry

theorem part_2 (a : ℕ → ℚ) (b : ℕ → ℚ) (S : ℕ → ℚ) (h1 : ∀ n, S (n + 1) = 4 * a n - 2) (h2 : a 1 = 2) :
  (∀ n, b n = 1 / (a n * a (n + 1))) → ∀ n, S n = (1/6) * (1 - (1/4)^n) :=
sorry

end part_1_part_2_l214_214082


namespace WalterWorksDaysAWeek_l214_214561

theorem WalterWorksDaysAWeek (hourlyEarning : ℕ) (hoursPerDay : ℕ) (schoolAllocationFraction : ℚ) (schoolAllocation : ℕ) 
  (dailyEarning : ℕ) (weeklyEarning : ℕ) (daysWorked : ℕ) :
  hourlyEarning = 5 →
  hoursPerDay = 4 →
  schoolAllocationFraction = 3 / 4 →
  schoolAllocation = 75 →
  dailyEarning = hourlyEarning * hoursPerDay →
  weeklyEarning = (schoolAllocation : ℚ) / schoolAllocationFraction →
  daysWorked = weeklyEarning / dailyEarning →
  daysWorked = 5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end WalterWorksDaysAWeek_l214_214561


namespace possible_values_of_q_l214_214097

theorem possible_values_of_q {q : ℕ} (hq : q > 0) :
  (∃ k : ℕ, (5 * q + 35) = k * (3 * q - 7) ∧ k > 0) ↔
  q = 3 ∨ q = 4 ∨ q = 5 ∨ q = 7 ∨ q = 9 ∨ q = 15 ∨ q = 21 ∨ q = 31 :=
by
  sorry

end possible_values_of_q_l214_214097


namespace Lavinia_daughter_age_difference_l214_214649

-- Define the ages of the individuals involved
variables (Ld Ls Kd : ℕ)

-- Conditions given in the problem
variables (H1 : Kd = 12)
variables (H2 : Ls = 2 * Kd)
variables (H3 : Ls = Ld + 22)

-- Statement we need to prove
theorem Lavinia_daughter_age_difference(Ld Ls Kd : ℕ) (H1 : Kd = 12) (H2 : Ls = 2 * Kd) (H3 : Ls = Ld + 22) : 
  Kd - Ld = 10 :=
sorry

end Lavinia_daughter_age_difference_l214_214649


namespace initial_dolphins_l214_214029

variable (D : ℕ)

theorem initial_dolphins (h1 : 3 * D + D = 260) : D = 65 :=
by
  sorry

end initial_dolphins_l214_214029


namespace minimum_k_l214_214758

variable {a b k : ℝ}

theorem minimum_k (h_a : a > 0) (h_b : b > 0) (h : ∀ a b : ℝ, a > 0 → b > 0 → (1 / a) + (1 / b) + (k / (a + b)) ≥ 0) : k ≥ -4 :=
sorry

end minimum_k_l214_214758


namespace smallest_a_l214_214847

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem smallest_a (a : ℕ) (h1 : 5880 = 2^3 * 3^1 * 5^1 * 7^2)
                    (h2 : ∀ b : ℕ, b < a → ¬ is_perfect_square (5880 * b))
                    : a = 15 :=
by
  sorry

end smallest_a_l214_214847


namespace remainder_when_M_divided_by_32_l214_214947

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l214_214947


namespace exists_c_gt_zero_l214_214759

theorem exists_c_gt_zero (a b : ℝ) (h : a < b) : ∃ c > 0, a < b + c := 
sorry

end exists_c_gt_zero_l214_214759


namespace lateral_edges_in_same_plane_edges_in_planes_for_all_vertices_l214_214303

-- Define a cube with edge length a
structure Cube :=
  (a : ℝ) -- Edge length of the cube

-- Define a pyramid with a given height
structure Pyramid :=
  (h : ℝ) -- Height of the pyramid

-- The main theorem statement for part 4A
theorem lateral_edges_in_same_plane (c : Cube) (p : Pyramid) : p.h = c.a ↔ (∃ O1 O2 O3 : ℝ × ℝ × ℝ,
  O1 = (c.a / 2, c.a / 2, -p.h) ∧
  O2 = (c.a / 2, -p.h, c.a / 2) ∧
  O3 = (-p.h, c.a / 2, c.a / 2)) := sorry

-- The main theorem statement for part 4B
theorem edges_in_planes_for_all_vertices (c : Cube) (p : Pyramid) : p.h = c.a ↔ ∀ (v : ℝ × ℝ × ℝ), -- Iterate over cube vertices
  (∃ O1 O2 O3 : ℝ × ℝ × ℝ,
    O1 = (c.a / 2, c.a / 2, -p.h) ∧
    O2 = (c.a / 2, -p.h, c.a / 2) ∧
    O3 = (-p.h, c.a / 2, c.a / 2)) := sorry

end lateral_edges_in_same_plane_edges_in_planes_for_all_vertices_l214_214303


namespace product_cos_angles_l214_214834

theorem product_cos_angles :
  (Real.cos (π / 15) * Real.cos (2 * π / 15) * Real.cos (3 * π / 15) * Real.cos (4 * π / 15) * Real.cos (5 * π / 15) * Real.cos (6 * π / 15) * Real.cos (7 * π / 15) = 1 / 128) :=
sorry

end product_cos_angles_l214_214834


namespace hyperbola_equation_chord_length_l214_214351

noncomputable def length_real_axis := 2
noncomputable def eccentricity := Real.sqrt 3
noncomputable def a := 1
noncomputable def b := Real.sqrt 2
noncomputable def hyperbola_eq (x y : ℝ) := x^2 - y^2 / 2 = 1

theorem hyperbola_equation : 
  (∀ x y : ℝ, hyperbola_eq x y ↔ x^2 - (y^2 / 2) = 1) :=
by
  intros x y
  sorry

theorem chord_length (m : ℝ) : 
  ∀ x1 x2 y1 y2 : ℝ, y1 = x1 + m → y2 = x2 + m →
    x1^2 - y1^2 / 2 = 1 → x2^2 - y2^2 / 2 = 1 →
    Real.sqrt (2 * ((x1 + x2)^2 - 4 * x1 * x2)) = 4 * Real.sqrt 2 →
    m = 1 ∨ m = -1 :=
by
  intros x1 x2 y1 y2 h1 h2 h3 h4 h5
  sorry

end hyperbola_equation_chord_length_l214_214351


namespace intersection_AB_l214_214623

/-- Define the set A based on the given condition -/
def setA : Set ℝ := {x | 2 * x ^ 2 + x > 0}

/-- Define the set B based on the given condition -/
def setB : Set ℝ := {x | 2 * x + 1 > 0}

/-- Prove that A ∩ B = {x | x > 0} -/
theorem intersection_AB : (setA ∩ setB) = {x | x > 0} :=
sorry

end intersection_AB_l214_214623


namespace crayons_selection_l214_214284

-- Definition for the problem constraints
def number_of_ways_to_select_crayons (crayons : ℕ) (select_1 : ℕ) (select_2 : ℕ) : ℕ :=
  (Nat.choose crayons select_1) * (Nat.choose (crayons - select_1) select_2)

-- Problem statement in Lean 4
theorem crayons_selection (h : number_of_ways_to_select_crayons 15 3 4 = 225225) : true :=
  by
    sorry

end crayons_selection_l214_214284


namespace seating_arrangement_l214_214718

def valid_arrangements := 6

def Alice_refusal (A B C : Prop) := (¬ (A ∧ B)) ∧ (¬ (A ∧ C))
def Derek_refusal (D E C : Prop) := (¬ (D ∧ E)) ∧ (¬ (D ∧ C))

theorem seating_arrangement (A B C D E : Prop) : 
  Alice_refusal A B C ∧ Derek_refusal D E C → valid_arrangements = 6 := 
  sorry

end seating_arrangement_l214_214718


namespace increase_in_average_l214_214559

theorem increase_in_average {a1 a2 a3 a4 : ℕ} 
                            (h1 : a1 = 92) 
                            (h2 : a2 = 89) 
                            (h3 : a3 = 91) 
                            (h4 : a4 = 93) : 
    ((a1 + a2 + a3 + a4 : ℚ) / 4) - ((a1 + a2 + a3 : ℚ) / 3) = 0.58 := 
by
  sorry

end increase_in_average_l214_214559


namespace higher_concentration_acid_solution_l214_214573

theorem higher_concentration_acid_solution (x : ℝ) (h1 : 2 * (8 / 100 : ℝ) = 1.2 * (x / 100) + 0.8 * (5 / 100)) : x = 10 :=
sorry

end higher_concentration_acid_solution_l214_214573


namespace div_by_90_l214_214109

def N : ℤ := 19^92 - 91^29

theorem div_by_90 : ∃ k : ℤ, N = 90 * k := 
sorry

end div_by_90_l214_214109


namespace area_triangle_possible_values_l214_214638

noncomputable def area_of_triangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1 / 2 * a * c * Real.sin B

theorem area_triangle_possible_values (a b c : ℝ) (A B C : ℝ) (ha : a = 2) (hc : c = 2 * Real.sqrt 3) (hA : A = Real.pi / 6) :
  ∃ S, S = 2 * Real.sqrt 3 ∨ S = Real.sqrt 3 :=
by
  -- Define the area using the given values
  sorry

end area_triangle_possible_values_l214_214638


namespace arithmetic_sqrt_of_9_l214_214392

theorem arithmetic_sqrt_of_9 : ∃ y : ℝ, y ^ 2 = 9 ∧ y ≥ 0 ∧ y = 3 := by
  sorry

end arithmetic_sqrt_of_9_l214_214392


namespace value_of_z_plus_one_over_y_l214_214801

theorem value_of_z_plus_one_over_y
  (x y z : ℝ)
  (h1 : 0 < x) 
  (h2 : 0 < y) 
  (h3 : 0 < z)
  (h4 : x * y * z = 1)
  (h5 : x + 1 / z = 3)
  (h6 : y + 1 / x = 31) :
  z + 1 / y = 9 / 23 :=
by
  sorry

end value_of_z_plus_one_over_y_l214_214801


namespace bekah_days_left_l214_214722

theorem bekah_days_left 
  (total_pages : ℕ)
  (pages_read : ℕ)
  (pages_per_day : ℕ)
  (remaining_pages : ℕ := total_pages - pages_read)
  (days_left : ℕ := remaining_pages / pages_per_day) :
  total_pages = 408 →
  pages_read = 113 →
  pages_per_day = 59 →
  days_left = 5 :=
by {
  sorry
}

end bekah_days_left_l214_214722


namespace find_certain_number_l214_214055

theorem find_certain_number (x : ℕ) (certain_number : ℕ)
  (h1 : certain_number * x = 675)
  (h2 : x = 27) : certain_number = 25 :=
by
  -- Proof goes here
  sorry

end find_certain_number_l214_214055


namespace smallest_total_squares_l214_214710

theorem smallest_total_squares (n : ℕ) (h : 4 * n - 4 = 2 * n) : n^2 = 4 :=
by
  sorry

end smallest_total_squares_l214_214710


namespace remainder_of_M_l214_214990

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l214_214990


namespace relationship_between_y1_y2_l214_214087

theorem relationship_between_y1_y2 (b y1 y2 : ℝ) 
  (h1 : y1 = -(-2) + b) 
  (h2 : y2 = -(3) + b) : 
  y1 > y2 := 
by {
  sorry
}

end relationship_between_y1_y2_l214_214087


namespace five_coins_all_heads_or_tails_l214_214464

theorem five_coins_all_heads_or_tails : 
  (1 / 2) ^ 5 + (1 / 2) ^ 5 = 1 / 16 := 
by 
  sorry

end five_coins_all_heads_or_tails_l214_214464


namespace smallest_prime_fifth_term_of_arithmetic_sequence_l214_214747

theorem smallest_prime_fifth_term_of_arithmetic_sequence :
  ∃ (a d : ℕ) (seq : ℕ → ℕ), 
    (∀ n, seq n = a + n * d) ∧ 
    (∀ n < 5, Prime (seq n)) ∧ 
    d = 6 ∧ 
    a = 5 ∧ 
    seq 4 = 29 := by
  sorry

end smallest_prime_fifth_term_of_arithmetic_sequence_l214_214747


namespace factor_quadratic_l214_214589

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := 16 * x^2 - 56 * x + 49

-- The goal is to prove that the quadratic expression is equal to (4x - 7)^2
theorem factor_quadratic (x : ℝ) : quadratic_expr x = (4 * x - 7)^2 :=
by
  sorry

end factor_quadratic_l214_214589


namespace factor_sum_l214_214337

theorem factor_sum : 
  (∃ d e, x^2 + 9 * x + 20 = (x + d) * (x + e)) ∧ 
  (∃ e f, x^2 - x - 56 = (x + e) * (x - f)) → 
  ∃ d e f, d + e + f = 19 :=
by
  sorry

end factor_sum_l214_214337


namespace necessary_condition_l214_214204

theorem necessary_condition {x m : ℝ} 
  (p : |1 - (x - 1) / 3| ≤ 2)
  (q : x^2 - 2 * x + 1 - m^2 ≤ 0)
  (hm : m > 0)
  (h_np_nq : ¬(|1 - (x - 1) / 3| ≤ 2) → ¬(x^2 - 2 * x + 1 - m^2 ≤ 0))
  : m ≥ 9 :=
sorry

end necessary_condition_l214_214204


namespace free_fall_time_l214_214026

theorem free_fall_time (h : ℝ) (t : ℝ) (h_eq : h = 4.9 * t^2) (h_val : h = 490) : t = 10 :=
by
  sorry

end free_fall_time_l214_214026


namespace second_layer_ratio_l214_214322

theorem second_layer_ratio
  (first_layer_sugar third_layer_sugar : ℕ)
  (third_layer_factor : ℕ)
  (h1 : first_layer_sugar = 2)
  (h2 : third_layer_sugar = 12)
  (h3 : third_layer_factor = 3) :
  third_layer_sugar = third_layer_factor * (2 * first_layer_sugar) →
  second_layer_factor = 2 :=
by
  sorry

end second_layer_ratio_l214_214322


namespace evaluate_expression_l214_214051

theorem evaluate_expression : 
  908 * 501 - (731 * 1389 - (547 * 236 + 842 * 731 - 495 * 361)) = 5448 := by
  sorry

end evaluate_expression_l214_214051


namespace prime_product_mod_32_l214_214978

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l214_214978


namespace length_squared_t_graph_interval_l214_214662

noncomputable def p (x : ℝ) : ℝ := -x + 2
noncomputable def q (x : ℝ) : ℝ := x + 2
noncomputable def r (x : ℝ) : ℝ := 2
noncomputable def t (x : ℝ) : ℝ :=
  if x ≤ -2 then p x
  else if x ≤ 2 then r x
  else q x

theorem length_squared_t_graph_interval :
  let segment_length (f : ℝ → ℝ) (a b : ℝ) : ℝ := Real.sqrt ((f b - f a)^2 + (b - a)^2)
  segment_length t (-4) (-2) + segment_length t (-2) 2 + segment_length t 2 4 = 4 + 2 * Real.sqrt 32 →
  (4 + 2 * Real.sqrt 32)^2 = 80 :=
sorry

end length_squared_t_graph_interval_l214_214662


namespace relationship_a_b_c_l214_214470

noncomputable def a := Real.log 3 / Real.log (1/2)
noncomputable def b := Real.log (1/2) / Real.log 3
noncomputable def c := Real.exp (0.3 * Real.log 2)

theorem relationship_a_b_c : 
  a < b ∧ b < c := 
by {
  sorry
}

end relationship_a_b_c_l214_214470


namespace remainder_when_divided_by_32_l214_214967

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l214_214967


namespace remainder_when_M_divided_by_32_l214_214946

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l214_214946


namespace remainder_of_M_l214_214983

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l214_214983


namespace novel_corona_high_students_l214_214557

theorem novel_corona_high_students (students_know_it_all students_karen_high total_students students_novel_corona : ℕ)
  (h1 : students_know_it_all = 50)
  (h2 : students_karen_high = 3 / 5 * students_know_it_all)
  (h3 : total_students = 240)
  (h4 : students_novel_corona = total_students - (students_know_it_all + students_karen_high))
  : students_novel_corona = 160 :=
sorry

end novel_corona_high_students_l214_214557


namespace total_weight_of_shells_l214_214510

noncomputable def initial_weight : ℝ := 5.25
noncomputable def weight_large_shell_g : ℝ := 700
noncomputable def grams_per_pound : ℝ := 453.592
noncomputable def additional_weight : ℝ := 4.5

/-
We need to prove:
5.25 pounds (initial weight) + (700 grams * (1 pound / 453.592 grams)) (weight of large shell in pounds) + 4.5 pounds (additional weight) = 11.293235835 pounds
-/
theorem total_weight_of_shells :
  initial_weight + (weight_large_shell_g / grams_per_pound) + additional_weight = 11.293235835 := by
    -- Proof will be inserted here
    sorry

end total_weight_of_shells_l214_214510


namespace minimum_value_of_function_l214_214403

-- Define the function y = 2x + 1/(x - 1) with the constraint x > 1
noncomputable def f (x : ℝ) : ℝ := 2 * x + 1 / (x - 1)

-- Prove that the minimum value of the function for x > 1 is 2√2 + 2
theorem minimum_value_of_function : 
  ∃ x : ℝ, x > 1 ∧ ∀ y : ℝ, (y = f x) → y ≥ 2 * Real.sqrt 2 + 2 := 
  sorry

end minimum_value_of_function_l214_214403


namespace total_dolphins_l214_214366

theorem total_dolphins (initial_dolphins : ℕ) (triple_of_initial : ℕ) (final_dolphins : ℕ) 
    (h1 : initial_dolphins = 65) (h2 : triple_of_initial = 3 * initial_dolphins) (h3 : final_dolphins = initial_dolphins + triple_of_initial) : 
    final_dolphins = 260 :=
by
  -- Proof goes here
  sorry

end total_dolphins_l214_214366


namespace count_pos_integers_three_digits_l214_214359

/-- The number of positive integers less than 50,000 having at most three distinct digits equals 7862. -/
theorem count_pos_integers_three_digits : 
  ∃ n : ℕ, n < 50000 ∧ (∀ d1 d2 d3 d4 d5 : ℕ, d1 ≠ d2 ∨ d1 ≠ d3 ∨ d1 ≠ d4 ∨ d1 ≠ d5 ∨ d2 ≠ d3 ∨ d2 ≠ d4 ∨ d2 ≠ d5 ∨ d3 ≠ d4 ∨ d3 ≠ d5 ∨ d4 ≠ d5) ∧ n = 7862 :=
sorry

end count_pos_integers_three_digits_l214_214359


namespace problem_statement_l214_214114

   noncomputable def g (x : ℝ) : ℝ := (3 * x + 4) / (x + 3)

   def T := {y : ℝ | ∃ (x : ℝ), x ≥ 0 ∧ y = g x}

   theorem problem_statement :
     (∃ N, (∀ y ∈ T, y ≤ N) ∧ N = 3 ∧ N ∉ T) ∧
     (∃ n, (∀ y ∈ T, y ≥ n) ∧ n = 4/3 ∧ n ∈ T) :=
   by
     sorry
   
end problem_statement_l214_214114


namespace ef_length_l214_214229

theorem ef_length (FR RG : ℝ) (cos_ERH : ℝ) (h1 : FR = 12) (h2 : RG = 6) (h3 : cos_ERH = 1 / 5) : EF = 30 :=
by
  sorry

end ef_length_l214_214229


namespace sum_of_areas_of_two_parks_l214_214013

theorem sum_of_areas_of_two_parks :
  let side1 := 11
  let side2 := 5
  let area1 := side1 * side1
  let area2 := side2 * side2
  area1 + area2 = 146 := 
by 
  sorry

end sum_of_areas_of_two_parks_l214_214013


namespace minewaska_state_park_l214_214500

variable (B H : Nat)

theorem minewaska_state_park (hikers_bike_riders_sum : H + B = 676) (hikers_more_than_bike_riders : H = B + 178) : H = 427 :=
sorry

end minewaska_state_park_l214_214500


namespace people_who_didnt_show_up_l214_214857

-- Definitions based on the conditions
def invited_people : ℕ := 68
def people_per_table : ℕ := 3
def tables_needed : ℕ := 6

-- Theorem statement
theorem people_who_didnt_show_up : 
  (invited_people - tables_needed * people_per_table = 50) :=
by 
  sorry

end people_who_didnt_show_up_l214_214857


namespace compute_expression_l214_214451

theorem compute_expression :
  20 * (150 / 3 + 40 / 5 + 16 / 25 + 2) = 1212.8 :=
by
  -- skipping the proof steps
  sorry

end compute_expression_l214_214451


namespace sufficient_but_not_necessary_condition_for_negativity_l214_214619

variable (b c : ℝ)

def f (x : ℝ) : ℝ := x^2 + b*x + c

theorem sufficient_but_not_necessary_condition_for_negativity (b c : ℝ) :
  (c < 0 → ∃ x : ℝ, f b c x < 0) ∧ (∃ b c : ℝ, ∃ x : ℝ, c ≥ 0 ∧ f b c x < 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_negativity_l214_214619


namespace pencil_cost_l214_214071

-- Definitions of given conditions
def has_amount : ℝ := 5.00  -- Elizabeth has 5 dollars
def borrowed_amount : ℝ := 0.53  -- She borrowed 53 cents
def needed_amount : ℝ := 0.47  -- She needs 47 cents more

-- Theorem to prove the cost of the pencil
theorem pencil_cost : has_amount + borrowed_amount + needed_amount = 6.00 := by 
  sorry

end pencil_cost_l214_214071


namespace euclidean_algorithm_steps_fibonacci_inequalities_l214_214530

theorem euclidean_algorithm_steps_fibonacci_inequalities
  (a b d : ℕ) 
  (n : ℕ) 
  (h_gcd : Nat.gcd a b = d) 
  (h_ab : a > b)
  (h_steps : -- condition that Euclidean algorithm for (a, b) stops after 'n' steps)
  : (a ≥ d * fibonacci (n + 2)) ∧ (b ≥ d * fibonacci (n + 1)) :=
sorry

end euclidean_algorithm_steps_fibonacci_inequalities_l214_214530


namespace maximum_of_f_attain_maximum_of_f_l214_214401

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 - 4

theorem maximum_of_f : ∀ x : ℝ, f x ≤ 0 :=
sorry

theorem attain_maximum_of_f : ∃ x : ℝ, f x = 0 :=
sorry

end maximum_of_f_attain_maximum_of_f_l214_214401


namespace median_room_number_l214_214584

open Finset

theorem median_room_number :
  let rooms : Finset ℕ := (range 25).erase 15 |> erase 20
  ∃ median : ℕ, median = 12 ∧ median ∈ rooms :=
by 
  let rooms : Finset ℕ := (range 25).erase 15 |> erase 20
  -- given that rooms now contain 1 to 14, 16 to 19, 21 to 25
  have : rooms.card = 23,
  { 
    -- ([0, 1, 2, ..., 24] \ {15, 20}) = 25 - 2 = 23
    rw [card_erase_of_mem, card_erase_of_mem, card_range 25], 
    { exact nat.lt_succ_self 25, },
    { exact mem_range.mpr (nat.le_succ 24), },
    { exact nat.lt_succ_self 25, },
  },
  -- find the 12th element of this ordered set, which should have 23-1 elements before it 
  -- and 23-12 elements after it
  use ((rooms.to_list).nth_le 11 sorry),
  split,
  { refl, },
  {
    suffices H : (rooms.to_list).sorted (<),
    { exact nth_le_mem _ H _ sorry, },
    -- sort the list of room numbers: [1, 2, ..., 14, 16, ..., 19, 21, ..., 25]
    sorry, 
  }

end median_room_number_l214_214584


namespace time_of_free_fall_l214_214023

theorem time_of_free_fall (h : ℝ) (t : ℝ) (height_fall_eq : h = 4.9 * t^2) (initial_height : h = 490) : t = 10 :=
by
  -- Proof is omitted
  sorry

end time_of_free_fall_l214_214023


namespace largest_possible_percent_error_l214_214540

theorem largest_possible_percent_error
  (d : ℝ) (error_percent : ℝ) (actual_area : ℝ)
  (h_d : d = 30) (h_error_percent : error_percent = 0.1)
  (h_actual_area : actual_area = 225 * Real.pi) :
  ∃ max_error_percent : ℝ,
    (max_error_percent = 21) :=
by
  sorry

end largest_possible_percent_error_l214_214540


namespace rationalize_denominator_l214_214253

theorem rationalize_denominator :
  ∃ A B C D : ℝ, 
    (1 / (real.cbrt 5 - real.cbrt 3) = (real.cbrt A + real.cbrt B + real.cbrt C) / D) ∧ 
    A + B + C + D = 51 :=
  sorry

end rationalize_denominator_l214_214253


namespace total_handshakes_proof_l214_214064

-- Define the groups
def groupA : Finset ℕ := {i | i < 25}
def groupB : Finset ℕ := {i | 25 ≤ i ∧ i < 35}
def groupC : Finset ℕ := {i | 35 ≤ i ∧ i < 40}

-- Calculate the individual group interactions
noncomputable def handshakes_AB := 10 * 25
noncomputable def handshakes_AC := 5 * 25
noncomputable def handshakes_BC := 10 * 5
noncomputable def handshakes_B := (Finset.card groupB) * (Finset.card groupB - 1) / 2
noncomputable def handshakes_C := 0

-- Total handshakes
noncomputable def total_handshakes := handshakes_AB + handshakes_AC + handshakes_BC + handshakes_B + handshakes_C

-- The statement to prove
theorem total_handshakes_proof : total_handshakes = 470 := by
  sorry

end total_handshakes_proof_l214_214064


namespace number_of_integer_exponent_terms_l214_214787

noncomputable def terms_with_integer_exponent : ℕ → ℕ 
| n := (range n).countp (λ r, (12 - (5 * r / 6)) ∈ ℕ)

theorem number_of_integer_exponent_terms : terms_with_integer_exponent 24 = 5 := 
sorry

end number_of_integer_exponent_terms_l214_214787


namespace find_LCM_l214_214848

-- Given conditions
def A := ℕ
def B := ℕ
def h := 22
def productAB := 45276

-- The theorem we want to prove
theorem find_LCM (a b lcm : ℕ) (hcf : ℕ) 
  (H_product : a * b = productAB) (H_hcf : hcf = h) : 
  (lcm = productAB / hcf) → 
  (a * b = hcf * lcm) :=
by
  intros H_lcm
  sorry

end find_LCM_l214_214848


namespace alyssa_plums_correct_l214_214319

def total_plums : ℕ := 27
def jason_plums : ℕ := 10
def alyssa_plums : ℕ := 17

theorem alyssa_plums_correct : alyssa_plums = total_plums - jason_plums := by
  sorry

end alyssa_plums_correct_l214_214319


namespace solve_for_x_l214_214258

theorem solve_for_x : ∀ (x : ℤ), (5 * x - 2) * 4 = (3 * (6 * x - 6)) → x = -5 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l214_214258


namespace verify_smallest_x_l214_214296

noncomputable def smallest_positive_integer_for_product_multiple_of_576 : ℕ :=
  let x := 36 in
  x

theorem verify_smallest_x :
  ∃ x : ℕ, x = smallest_positive_integer_for_product_multiple_of_576 ∧ (400 * x) % 576 = 0 :=
by
  use 36
  split
  { refl }
  { show (400 * 36) % 576 = 0
    sorry }

end verify_smallest_x_l214_214296


namespace remainder_when_M_divided_by_32_l214_214942

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l214_214942


namespace total_books_l214_214002

def books_per_shelf_mystery : ℕ := 7
def books_per_shelf_picture : ℕ := 5
def books_per_shelf_sci_fi : ℕ := 8
def books_per_shelf_biography : ℕ := 6

def shelves_mystery : ℕ := 8
def shelves_picture : ℕ := 2
def shelves_sci_fi : ℕ := 3
def shelves_biography : ℕ := 4

theorem total_books :
  (books_per_shelf_mystery * shelves_mystery) + 
  (books_per_shelf_picture * shelves_picture) + 
  (books_per_shelf_sci_fi * shelves_sci_fi) + 
  (books_per_shelf_biography * shelves_biography) = 114 :=
by
  sorry

end total_books_l214_214002


namespace perfect_square_proof_l214_214695

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem perfect_square_proof :
  isPerfectSquare (factorial 22 * factorial 23 * factorial 24 / 12) :=
sorry

end perfect_square_proof_l214_214695


namespace equation_solution_count_l214_214910

open Real

theorem equation_solution_count :
  ∃ s : Finset ℝ, (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * π ∧ sin (π / 4 * sin x) = cos (π / 4 * cos x)) ∧ s.card = 4 :=
by
  sorry

end equation_solution_count_l214_214910


namespace decreasing_power_function_has_specific_m_l214_214610

theorem decreasing_power_function_has_specific_m (m : ℝ) (x : ℝ) : 
  (∀ x > 0, (m^2 - m - 1) * x^(m^2 - 2 * m - 3) < 0) → 
  m = 2 :=
by
  sorry

end decreasing_power_function_has_specific_m_l214_214610


namespace tangent_triangle_perimeter_acute_tangent_triangle_perimeter_obtuse_l214_214232

theorem tangent_triangle_perimeter_acute (a b c: ℝ) (h1: a^2 + b^2 > c^2) (h2: b^2 + c^2 > a^2) (h3: c^2 + a^2 > b^2) :
  2 * a * b * c * (1 / (b^2 + c^2 - a^2) + 1 / (c^2 + a^2 - b^2) + 1 / (a^2 + b^2 - c^2)) = 
  2 * a * b * c * (1 / (b^2 + c^2 - a^2) + 1 / (c^2 + a^2 - b^2) + 1 / (a^2 + b^2 - c^2)) := 
by sorry -- proof goes here

theorem tangent_triangle_perimeter_obtuse (a b c: ℝ) (h1: a^2 > b^2 + c^2) :
  2 * a * b * c / (a^2 - b^2 - c^2) = 2 * a * b * c / (a^2 - b^2 - c^2) := 
by sorry -- proof goes here

end tangent_triangle_perimeter_acute_tangent_triangle_perimeter_obtuse_l214_214232


namespace necessary_but_not_sufficient_condition_l214_214805

open Set

variable {α : Type*}

def M : Set ℝ := { x | 0 < x ∧ x ≤ 4 }
def N : Set ℝ := { x | 2 ≤ x ∧ x ≤ 3 }

theorem necessary_but_not_sufficient_condition :
  (N ⊆ M) ∧ (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) :=
by
  sorry

end necessary_but_not_sufficient_condition_l214_214805


namespace option_d_is_correct_l214_214421

theorem option_d_is_correct (a b : ℝ) : -3 * (a - b) = -3 * a + 3 * b :=
by
  sorry

end option_d_is_correct_l214_214421


namespace selection_and_arrangement_l214_214008

-- Defining the problem conditions
def volunteers : Nat := 5
def roles : Nat := 4
def A_excluded_role : String := "music_composer"
def total_methods : Nat := 96

theorem selection_and_arrangement (h1 : volunteers = 5) (h2 : roles = 4) (h3 : A_excluded_role = "music_composer") :
  total_methods = 96 :=
by
  sorry

end selection_and_arrangement_l214_214008


namespace find_value_l214_214626

variable (x y z : ℕ)

-- Condition: x / 4 = y / 3 = z / 2
def ratio_condition := x / 4 = y / 3 ∧ y / 3 = z / 2

-- Theorem: Given the ratio condition, prove that (x - y + 3z) / x = 7 / 4.
theorem find_value (h : ratio_condition x y z) : (x - y + 3 * z) / x = 7 / 4 := 
  by sorry

end find_value_l214_214626


namespace least_number_of_shoes_l214_214657

theorem least_number_of_shoes (num_inhabitants : ℕ) 
  (one_legged_percentage : ℚ) 
  (barefooted_proportion : ℚ) 
  (h_num_inhabitants : num_inhabitants = 10000) 
  (h_one_legged_percentage : one_legged_percentage = 0.05) 
  (h_barefooted_proportion : barefooted_proportion = 0.5) : 
  ∃ (shoes_needed : ℕ), shoes_needed = 10000 := 
by
  sorry

end least_number_of_shoes_l214_214657


namespace find_a10_l214_214205

variable {n : ℕ}
variable (a : ℕ → ℝ)
variable (h_pos : ∀ (n : ℕ), 0 < a n)
variable (h_mul : ∀ (p q : ℕ), a (p + q) = a p * a q)
variable (h_a8 : a 8 = 16)

theorem find_a10 : a 10 = 32 :=
by
  sorry

end find_a10_l214_214205


namespace smaller_than_neg3_l214_214440

theorem smaller_than_neg3 :
  (∃ x ∈ ({0, -1, -5, -1/2} : Set ℚ), x < -3) ∧ ∀ x ∈ ({0, -1, -5, -1/2} : Set ℚ), x < -3 → x = -5 :=
by
  sorry

end smaller_than_neg3_l214_214440


namespace find_angle_C_max_area_triangle_l214_214777

-- Part I: Proving angle C
theorem find_angle_C (a b c : ℝ) (A B C : ℝ)
    (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
    (h4 : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C) :
    C = Real.pi / 3 :=
sorry

-- Part II: Finding maximum area of triangle ABC
theorem max_area_triangle (a b : ℝ) (c : ℝ) (h_c : c = 2 * Real.sqrt 3) (A B C : ℝ)
    (h_A : A > 0) (h_B : B > 0) (h_C : C = Real.pi / 3)
    (h : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C) :
    0.5 * a * b * Real.sin C ≤ 3 * Real.sqrt 3 :=
sorry

end find_angle_C_max_area_triangle_l214_214777


namespace area_triangle_ABF_proof_area_triangle_AFD_proof_l214_214664

variable (A B C D M F : Type)
variable (area_square : Real) (midpoint_D_CM : Prop) (lies_on_line_BC : Prop)

-- Given conditions
axiom area_ABCD_300 : area_square = 300
axiom M_midpoint_DC : midpoint_D_CM
axiom F_on_line_BC : lies_on_line_BC

-- Define areas for the triangles
def area_triangle_ABF : Real := 300
def area_triangle_AFD : Real := 150

-- Prove that given the conditions, the area of triangle ABF is 300 cm²
theorem area_triangle_ABF_proof : area_square = 300 ∧ midpoint_D_CM ∧ lies_on_line_BC → area_triangle_ABF = 300 :=
by
  intro h
  sorry

-- Prove that given the conditions, the area of triangle AFD is 150 cm²
theorem area_triangle_AFD_proof : area_square = 300 ∧ midpoint_D_CM ∧ lies_on_line_BC → area_triangle_AFD = 150 :=
by
  intro h
  sorry

end area_triangle_ABF_proof_area_triangle_AFD_proof_l214_214664


namespace range_of_m_plus_n_l214_214344

noncomputable def f (m n x : ℝ) : ℝ := m * 2^x + x^2 + n * x

theorem range_of_m_plus_n (m n : ℝ) :
  (∃ x : ℝ, f m n x = 0 ∧ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
by
  sorry

end range_of_m_plus_n_l214_214344


namespace arnold_and_danny_age_l214_214583

theorem arnold_and_danny_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 9) : x = 4 :=
sorry

end arnold_and_danny_age_l214_214583


namespace cylinder_h_over_r_equals_one_l214_214345

theorem cylinder_h_over_r_equals_one
  (A : ℝ) (r h : ℝ)
  (h_surface_area : A = 2 * π * r^2 + 2 * π * r * h)
  (V : ℝ := π * r^2 * h)
  (max_V : ∀ r' h', (A = 2 * π * r'^2 + 2 * π * r' * h') → (π * r'^2 * h' ≤ V) → (r' = r ∧ h' = h)) :
  h / r = 1 := by
sorry

end cylinder_h_over_r_equals_one_l214_214345


namespace forty_percent_of_n_l214_214251

theorem forty_percent_of_n (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 16) : 0.40 * N = 384 :=
by
  sorry

end forty_percent_of_n_l214_214251


namespace simplify_expression_l214_214292

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : 
  (15 * x^2) * (6 * x) * (1 / (3 * x)^2) = 10 * x := 
by
  sorry

end simplify_expression_l214_214292


namespace P_plus_Q_l214_214238

theorem P_plus_Q (P Q : ℝ) (h : (P / (x - 3) + Q * (x - 2)) = (-5 * x^2 + 18 * x + 27) / (x - 3)) : P + Q = 31 := 
by {
  sorry
}

end P_plus_Q_l214_214238


namespace negation_of_proposition_l214_214404

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x^2 + 3*x + 2 < 0) ↔ ∃ x : ℝ, x^2 + 3*x + 2 ≥ 0 :=
by
  sorry

end negation_of_proposition_l214_214404


namespace MutualExclusivity_Of_A_C_l214_214200

-- Definitions of events using conditions from a)
def EventA (products : List Bool) : Prop :=
  products.all (λ p => p = true)

def EventB (products : List Bool) : Prop :=
  products.all (λ p => p = false)

def EventC (products : List Bool) : Prop :=
  products.any (λ p => p = false)

-- The main theorem using correct answer from b)
theorem MutualExclusivity_Of_A_C (products : List Bool) :
  EventA products → ¬ EventC products :=
by
  sorry

end MutualExclusivity_Of_A_C_l214_214200


namespace largest_of_three_l214_214034

structure RealTriple (x y z : ℝ) where
  h1 : x + y + z = 3
  h2 : x * y + y * z + z * x = -8
  h3 : x * y * z = -18

theorem largest_of_three {x y z : ℝ} (h : RealTriple x y z) : max x (max y z) = Real.sqrt 5 :=
  sorry

end largest_of_three_l214_214034


namespace solve_for_A_l214_214650

def f (A B x : ℝ) : ℝ := A * x ^ 2 - 3 * B ^ 3
def g (B x : ℝ) : ℝ := 2 * B * x + B ^ 2

theorem solve_for_A (B : ℝ) (hB : B ≠ 0) (h : f A B (g B 2) = 0) :
  A = 3 / (16 / B + 8 + B ^ 3) :=
by
  sorry

end solve_for_A_l214_214650


namespace rectangular_plot_breadth_l214_214277

theorem rectangular_plot_breadth (b : ℝ) 
    (h1 : ∃ l : ℝ, l = 3 * b)
    (h2 : 432 = 3 * b * b) : b = 12 :=
by
  sorry

end rectangular_plot_breadth_l214_214277


namespace sum_of_x_values_l214_214537

theorem sum_of_x_values :
  (2^(x^2 + 6*x + 9) = 16^(x + 3)) → ∃ x1 x2 : ℝ, x1 + x2 = -2 :=
by
  sorry

end sum_of_x_values_l214_214537


namespace sin_squared_alpha_eq_one_add_sin_squared_beta_l214_214202

variable {α θ β : ℝ}

theorem sin_squared_alpha_eq_one_add_sin_squared_beta
  (h1 : Real.sin α = Real.sin θ + Real.cos θ)
  (h2 : Real.sin β ^ 2 = 2 * Real.sin θ * Real.cos θ) :
  Real.sin α ^ 2 = 1 + Real.sin β ^ 2 := 
sorry

end sin_squared_alpha_eq_one_add_sin_squared_beta_l214_214202


namespace intersection_in_first_quadrant_l214_214357

theorem intersection_in_first_quadrant (a : ℝ) :
  (∃ x y : ℝ, ax - y + 2 = 0 ∧ x + y - a = 0 ∧ x > 0 ∧ y > 0) ↔ a > 2 := 
by
  sorry

end intersection_in_first_quadrant_l214_214357


namespace first_player_wins_l214_214287

def initial_piles (p1 p2 : Nat) : Prop :=
  p1 = 33 ∧ p2 = 35

def winning_strategy (p1 p2 : Nat) : Prop :=
  ∃ moves : List (Nat × Nat), 
  (initial_piles p1 p2) →
  (∀ (p1' p2' : Nat), 
    (p1', p2') ∈ moves →
    p1' = 1 ∧ p2' = 1 ∨ p1' = 2 ∧ p2' = 1)

theorem first_player_wins : winning_strategy 33 35 :=
sorry

end first_player_wins_l214_214287


namespace projectile_height_l214_214269

theorem projectile_height (t : ℝ) (h : (-16 * t^2 + 80 * t = 100)) : t = 2.5 :=
sorry

end projectile_height_l214_214269


namespace sqrt_expression_eval_l214_214189

theorem sqrt_expression_eval :
    (Real.sqrt 8 - 2 * Real.sqrt (1 / 2) + (2 - Real.sqrt 3) * (2 + Real.sqrt 3)) = Real.sqrt 2 + 1 := 
by
  sorry

end sqrt_expression_eval_l214_214189


namespace find_N_l214_214739

theorem find_N (N : ℕ) (h1 : N > 1)
  (h2 : ∀ (a b c : ℕ), (a ≡ b [MOD N]) ∧ (b ≡ c [MOD N]) ∧ (a ≡ c [MOD N])
  → (a = 1743) ∧ (b = 2019) ∧ (c = 3008)) :
  N = 23 :=
by
  have h3 : Nat.gcd (2019 - 1743) (3008 - 2019) = 23 := sorry
  show N = 23 from sorry

end find_N_l214_214739


namespace arithmetic_square_root_of_second_consecutive_l214_214154

theorem arithmetic_square_root_of_second_consecutive (x : ℝ) : 
  real.sqrt (x^2 + 1) = real.sqrt (x^2 + 1) :=
sorry

end arithmetic_square_root_of_second_consecutive_l214_214154


namespace cans_difference_l214_214881

theorem cans_difference 
  (n_cat_packages : ℕ) (n_dog_packages : ℕ) 
  (n_cat_cans_per_package : ℕ) (n_dog_cans_per_package : ℕ) :
  n_cat_packages = 6 →
  n_dog_packages = 2 →
  n_cat_cans_per_package = 9 →
  n_dog_cans_per_package = 3 →
  (n_cat_packages * n_cat_cans_per_package) - (n_dog_packages * n_dog_cans_per_package) = 48 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end cans_difference_l214_214881


namespace remainder_of_product_of_odd_primes_mod_32_l214_214991

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l214_214991


namespace calculate_perimeter_l214_214288

-- Definitions based on conditions
def num_posts : ℕ := 36
def post_width : ℕ := 2
def gap_width : ℕ := 4
def sides : ℕ := 4

-- Computations inferred from the conditions (not using solution steps directly)
def posts_per_side : ℕ := num_posts / sides
def gaps_per_side : ℕ := posts_per_side - 1
def side_length : ℕ := posts_per_side * post_width + gaps_per_side * gap_width

-- Theorem statement, proving the perimeter is 200 feet
theorem calculate_perimeter : 4 * side_length = 200 := by
  sorry

end calculate_perimeter_l214_214288


namespace cost_per_night_l214_214796

variable (x : ℕ)

theorem cost_per_night (h : 3 * x - 100 = 650) : x = 250 :=
sorry

end cost_per_night_l214_214796


namespace number_of_teams_l214_214681

theorem number_of_teams (n : ℕ) (h : (n * (n - 1)) / 2 * 10 = 1050) : n = 15 :=
by 
  sorry

end number_of_teams_l214_214681


namespace remainder_of_product_of_odd_primes_mod_32_l214_214995

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l214_214995


namespace find_g2_l214_214019

variable {R : Type*} [Nonempty R] [Field R]

-- Define the function g
def g (x : R) : R := sorry

-- Given conditions
axiom condition1 : ∀ x y : R, x * g y = 2 * y * g x
axiom condition2 : g 10 = 5

-- The statement to be proved
theorem find_g2 : g 2 = 2 :=
by
  sorry

end find_g2_l214_214019


namespace incorrect_observation_value_l214_214402

-- Definitions stemming from the given conditions
def initial_mean : ℝ := 100
def corrected_mean : ℝ := 99.075
def number_of_observations : ℕ := 40
def correct_observation_value : ℝ := 50

-- Lean theorem statement to prove the incorrect observation value
theorem incorrect_observation_value (initial_mean corrected_mean correct_observation_value : ℝ) (number_of_observations : ℕ) :
  (initial_mean * number_of_observations - corrected_mean * number_of_observations + correct_observation_value) = 87 := 
sorry

end incorrect_observation_value_l214_214402


namespace find_sin_θ_l214_214210

open Real

noncomputable def θ_in_range_and_sin_2θ (θ : ℝ) : Prop :=
  (θ ∈ Set.Icc (π / 4) (π / 2)) ∧ (sin (2 * θ) = 3 * sqrt 7 / 8)

theorem find_sin_θ (θ : ℝ) (h : θ_in_range_and_sin_2θ θ) : sin θ = 3 / 4 :=
  sorry

end find_sin_θ_l214_214210


namespace points_comparison_l214_214355

def quadratic_function (m x : ℝ) : ℝ :=
  (x + m - 3) * (x - m) + 3

def point_on_graph (m x y : ℝ) : Prop :=
  y = quadratic_function m x

theorem points_comparison (m x1 x2 y1 y2 : ℝ)
  (h1 : point_on_graph m x1 y1)
  (h2 : point_on_graph m x2 y2)
  (hx : x1 < x2)
  (h_sum : x1 + x2 < 3) :
  y1 > y2 := 
  sorry

end points_comparison_l214_214355


namespace bullet_trains_crossing_time_l214_214560

theorem bullet_trains_crossing_time
  (length_train1 : ℝ) (length_train2 : ℝ)
  (speed_train1_km_hr : ℝ) (speed_train2_km_hr : ℝ)
  (opposite_directions : Prop)
  (h_length1 : length_train1 = 140)
  (h_length2 : length_train2 = 170)
  (h_speed1 : speed_train1_km_hr = 60)
  (h_speed2 : speed_train2_km_hr = 40)
  (h_opposite : opposite_directions = true) :
  ∃ t : ℝ, t = 11.16 :=
by
  sorry

end bullet_trains_crossing_time_l214_214560


namespace vecMA_dotProduct_vecBA_range_l214_214207

-- Define the conditions
def pointM : ℝ × ℝ := (1, 0)

def onEllipse (p : ℝ × ℝ) : Prop := (p.1^2 / 4 + p.2^2 = 1)

def vecMA (A : ℝ × ℝ) := (A.1 - pointM.1, A.2 - pointM.2)
def vecMB (B : ℝ × ℝ) := (B.1 - pointM.1, B.2 - pointM.2)
def vecBA (A B : ℝ × ℝ) := (A.1 - B.1, A.2 - B.2)

def dotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the statement
theorem vecMA_dotProduct_vecBA_range (A B : ℝ × ℝ) (α : ℝ) :
  onEllipse A → onEllipse B → dotProduct (vecMA A) (vecMB B) = 0 → 
  A = (2 * Real.cos α, Real.sin α) → 
  (2/3 ≤ dotProduct (vecMA A) (vecBA A B) ∧ dotProduct (vecMA A) (vecBA A B) ≤ 9) :=
sorry

end vecMA_dotProduct_vecBA_range_l214_214207


namespace quadratic_inequality_solution_set_l214_214553

theorem quadratic_inequality_solution_set :
  {x : ℝ | - x ^ 2 + 4 * x + 12 > 0} = {x : ℝ | -2 < x ∧ x < 6} :=
sorry

end quadratic_inequality_solution_set_l214_214553


namespace distance_between_points_l214_214460

theorem distance_between_points :
  let x1 := 1
  let y1 := 16
  let x2 := 9
  let y2 := 3
  let distance := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance = Real.sqrt 233 :=
by
  sorry

end distance_between_points_l214_214460


namespace factorization_l214_214335

theorem factorization (x : ℝ) : 4 * x^3 - x = x * (2 * x + 1) * (2 * x - 1) :=
by sorry

end factorization_l214_214335


namespace age_difference_is_16_l214_214138

-- Variables
variables (y : ℕ) -- y represents the present age of the younger person

-- Conditions from the problem
def elder_present_age := 30
def elder_age_6_years_ago := elder_present_age - 6
def younger_age_6_years_ago := y - 6

-- Given condition 6 years ago:
def condition_6_years_ago := elder_age_6_years_ago = 3 * younger_age_6_years_ago

-- The theorem to prove the difference in ages is 16 years
theorem age_difference_is_16
  (h1 : elder_present_age = 30)
  (h2 : condition_6_years_ago) :
  elder_present_age - y = 16 :=
by sorry

end age_difference_is_16_l214_214138


namespace find_non_zero_real_x_satisfies_equation_l214_214198

theorem find_non_zero_real_x_satisfies_equation :
  ∃! x : ℝ, x ≠ 0 ∧ (9 * x) ^ 18 - (18 * x) ^ 9 = 0 ∧ x = 2 :=
by
  sorry

end find_non_zero_real_x_satisfies_equation_l214_214198


namespace harry_annual_pet_feeding_cost_l214_214358

def monthly_cost_snake := 10
def monthly_cost_iguana := 5
def monthly_cost_gecko := 15
def num_snakes := 4
def num_iguanas := 2
def num_geckos := 3
def months_in_year := 12

theorem harry_annual_pet_feeding_cost :
  (num_snakes * monthly_cost_snake + 
   num_iguanas * monthly_cost_iguana + 
   num_geckos * monthly_cost_gecko) * 
   months_in_year = 1140 := 
sorry

end harry_annual_pet_feeding_cost_l214_214358


namespace option_d_is_correct_l214_214422

theorem option_d_is_correct (a b : ℝ) : -3 * (a - b) = -3 * a + 3 * b :=
by
  sorry

end option_d_is_correct_l214_214422


namespace series_sum_eq_14_div_15_l214_214072

theorem series_sum_eq_14_div_15 : 
  (∑ n in Finset.range 14, (1 : ℚ) / ((n + 1) * (n + 2))) = 14 / 15 := 
by
  sorry

end series_sum_eq_14_div_15_l214_214072


namespace arithmetic_geometric_sequence_l214_214062

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : a 2 * a 5 = (a 4) ^ 2)
  (h4 : d ≠ 0) : d = -1 / 5 :=
by
  sorry

end arithmetic_geometric_sequence_l214_214062


namespace miranda_savings_l214_214809

theorem miranda_savings:
  ∀ (months : ℕ) (sister_contribution price shipping total paid_per_month : ℝ),
    months = 3 →
    sister_contribution = 50 →
    price = 210 →
    shipping = 20 →
    total = 230 →
    total - sister_contribution = price + shipping →
    paid_per_month = (total - sister_contribution) / months →
    paid_per_month = 60 :=
by
  intros months sister_contribution price shipping total paid_per_month h1 h2 h3 h4 h5 h6 h7
  sorry

end miranda_savings_l214_214809


namespace remainder_of_M_when_divided_by_32_l214_214939

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l214_214939


namespace division_equivalence_l214_214843

theorem division_equivalence (a b c d : ℝ) (h1 : a = 11.7) (h2 : b = 2.6) (h3 : c = 117) (h4 : d = 26) :
  (11.7 / 2.6) = (117 / 26) ∧ (117 / 26) = 4.5 := 
by 
  sorry

end division_equivalence_l214_214843


namespace speed_ratio_l214_214427

theorem speed_ratio (a b v1 v2 S : ℝ) (h1 : S = a * (v1 + v2)) (h2 : S = b * (v1 - v2)) (h3 : a ≠ b) : 
  v1 / v2 = (a + b) / (b - a) :=
by
  -- proof skipped
  sorry

end speed_ratio_l214_214427


namespace plums_added_l214_214163

-- Definitions of initial and final plum counts
def initial_plums : ℕ := 17
def final_plums : ℕ := 21

-- The mathematical statement to be proved
theorem plums_added (initial_plums final_plums : ℕ) : final_plums - initial_plums = 4 := by
  -- The proof will be inserted here
  sorry

end plums_added_l214_214163


namespace traveler_never_returns_home_l214_214184

variable (City : Type)
variable (Distance : City → City → ℝ)

variables (A B C : City)
variables (C_i C_i_plus_one C_i_minus_one : City)

-- Given conditions
axiom travel_far_from_A : ∀ (C : City), C ≠ B → Distance A B > Distance A C
axiom travel_far_from_B : ∀ (D : City), D ≠ C → Distance B C > Distance B D
axiom increasing_distance : ∀ i : ℕ, Distance C_i C_i_plus_one > Distance C_i_minus_one C_i

-- Given condition that C is not A
axiom C_not_eq_A : C ≠ A

-- Proof statement
theorem traveler_never_returns_home : ∀ i : ℕ, C_i ≠ A := sorry

end traveler_never_returns_home_l214_214184


namespace total_percentage_failed_exam_l214_214223

theorem total_percentage_failed_exam :
  let total_candidates := 2000
  let general_candidates := 1000
  let obc_candidates := 600
  let sc_candidates := 300
  let st_candidates := total_candidates - (general_candidates + obc_candidates + sc_candidates)
  let general_pass_percentage := 0.35
  let obc_pass_percentage := 0.50
  let sc_pass_percentage := 0.25
  let st_pass_percentage := 0.30
  let general_failed := general_candidates - (general_candidates * general_pass_percentage)
  let obc_failed := obc_candidates - (obc_candidates * obc_pass_percentage)
  let sc_failed := sc_candidates - (sc_candidates * sc_pass_percentage)
  let st_failed := st_candidates - (st_candidates * st_pass_percentage)
  let total_failed := general_failed + obc_failed + sc_failed + st_failed
  let failed_percentage := (total_failed / total_candidates) * 100
  failed_percentage = 62.25 :=
by
  sorry

end total_percentage_failed_exam_l214_214223


namespace johny_distance_l214_214928

noncomputable def distance_south : ℕ := 40
variable (E : ℕ)
noncomputable def distance_east : ℕ := E
noncomputable def distance_north (E : ℕ) : ℕ := 2 * E
noncomputable def total_distance (E : ℕ) : ℕ := distance_south + distance_east E + distance_north E

theorem johny_distance :
  ∀ E : ℕ, total_distance E = 220 → E - distance_south = 20 :=
by
  intro E
  intro h
  rw [total_distance, distance_north, distance_east, distance_south] at h
  sorry

end johny_distance_l214_214928


namespace remainder_of_M_when_divided_by_32_l214_214936

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l214_214936


namespace factorize_expression_l214_214891

theorem factorize_expression (a : ℝ) : a^2 + 2*a + 1 = (a + 1)^2 :=
by
  sorry

end factorize_expression_l214_214891


namespace hikers_rate_l214_214432

noncomputable def rate_up (rate_down := 15) : ℝ := 5

theorem hikers_rate :
  let R := rate_up
  let distance_down := rate_down
  let time := 2
  let rate_down := 1.5 * R
  distance_down = rate_down * time → R = 5 :=
by
  intro h
  sorry

end hikers_rate_l214_214432


namespace quadratic_value_l214_214860

theorem quadratic_value (a b c : ℝ) 
  (h1 : a + b + c = 2)
  (h2 : 4 * a + 2 * b + c = 3) :
  a + 2 * b + 3 * c = 7 :=
by
  sorry

end quadratic_value_l214_214860


namespace intersection_M_N_l214_214525

def M : Set ℝ := { x : ℝ | 0 < x ∧ x < 4 }
def N : Set ℝ := { x : ℝ | (1 / 3) ≤ x ∧ x ≤ 5 }

theorem intersection_M_N : M ∩ N = { x : ℝ | (1 / 3) ≤ x ∧ x < 4 } :=
by
  sorry

end intersection_M_N_l214_214525


namespace ratio_of_boys_l214_214921

theorem ratio_of_boys (p : ℝ) (h : p = (3 / 5) * (1 - p)) 
  : p = 3 / 8 := 
by
  sorry

end ratio_of_boys_l214_214921


namespace sum_of_transformed_numbers_l214_214133

theorem sum_of_transformed_numbers (x y S : ℝ) (h : x + y = S) : 
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 := 
by
  sorry

end sum_of_transformed_numbers_l214_214133


namespace solve_system_of_equations_l214_214816

theorem solve_system_of_equations (a b c x y z : ℝ):
  (x - a * y + a^2 * z = a^3) →
  (x - b * y + b^2 * z = b^3) →
  (x - c * y + c^2 * z = c^3) →
  x = a * b * c ∧ y = a * b + a * c + b * c ∧ z = a + b + c :=
by
  intros h1 h2 h3
  -- proof goes here
  sorry

end solve_system_of_equations_l214_214816


namespace factorize_expression_l214_214336

theorem factorize_expression (x y : ℝ) : 25 * x - x * y ^ 2 = x * (5 + y) * (5 - y) := by
  sorry

end factorize_expression_l214_214336


namespace number_of_valid_subsets_l214_214119

open Finset

theorem number_of_valid_subsets : 
  {S : Finset ℕ // ∀ n ∈ S, n ∈ range 1 11 ∧ S.card = 4 ∧
  (∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
   a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
   a > b ∧ a > d ∧
   (b = c + 4 ∨ d = c + 4 ∨ b = d + 4 ∨ d = b + 4)) } = 36 :=
by
  sorry

end number_of_valid_subsets_l214_214119


namespace find_point_A_l214_214245

theorem find_point_A (x : ℝ) (h : x + 7 - 4 = 0) : x = -3 :=
sorry

end find_point_A_l214_214245


namespace dice_sum_not_22_l214_214627

theorem dice_sum_not_22 (a b c d e : ℕ) (h₀ : 1 ≤ a ∧ a ≤ 6) (h₁ : 1 ≤ b ∧ b ≤ 6)
  (h₂ : 1 ≤ c ∧ c ≤ 6) (h₃ : 1 ≤ d ∧ d ≤ 6) (h₄ : 1 ≤ e ∧ e ≤ 6) 
  (h₅ : a * b * c * d * e = 432) : a + b + c + d + e ≠ 22 :=
sorry

end dice_sum_not_22_l214_214627


namespace rachel_picture_books_shelves_l214_214659

theorem rachel_picture_books_shelves (mystery_shelves : ℕ) (books_per_shelf : ℕ) (total_books : ℕ) 
  (h1 : mystery_shelves = 6) 
  (h2 : books_per_shelf = 9) 
  (h3 : total_books = 72) : 
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 2 :=
by sorry

end rachel_picture_books_shelves_l214_214659


namespace student_missed_20_l214_214317

theorem student_missed_20 {n : ℕ} (S_correct : ℕ) (S_incorrect : ℕ) 
    (h1 : S_correct = n * (n + 1) / 2)
    (h2 : S_incorrect = S_correct - 20) : 
    S_incorrect = n * (n + 1) / 2 - 20 := 
sorry

end student_missed_20_l214_214317


namespace jerry_age_l214_214531

theorem jerry_age (M J : ℝ) (h₁ : M = 17) (h₂ : M = 2.5 * J - 3) : J = 8 :=
by
  -- The proof is omitted.
  sorry

end jerry_age_l214_214531


namespace inequality_transformation_l214_214035

theorem inequality_transformation (x : ℝ) :
  x - 2 > 1 → x > 3 :=
by
  intro h
  linarith

end inequality_transformation_l214_214035


namespace annulus_divide_l214_214677

theorem annulus_divide (r : ℝ) (h₁ : 2 < 14) (h₂ : 2 > 0) (h₃ : 14 > 0)
    (h₄ : π * 196 - π * r^2 = π * r^2 - π * 4) : r = 10 := 
sorry

end annulus_divide_l214_214677


namespace midpoint_sum_l214_214654

theorem midpoint_sum (x y : ℝ) (h1 : (x + 0) / 2 = 2) (h2 : (y + 9) / 2 = 4) : x + y = 3 := by
  sorry

end midpoint_sum_l214_214654


namespace probability_all_heads_or_tails_l214_214466

def coin_outcomes := {heads, tails}

def total_outcomes (n : ℕ) : ℕ := 2^n

def favorable_outcomes : ℕ := 2

def probability_five_heads_or_tails (n : ℕ) (total : ℕ) (favorable : ℕ) : ℚ :=
  favorable / total

theorem probability_all_heads_or_tails :
  probability_five_heads_or_tails 5 (total_outcomes 5) favorable_outcomes = 1 / 16 :=
by
  sorry

end probability_all_heads_or_tails_l214_214466


namespace remainder_when_M_divided_by_32_l214_214941

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l214_214941


namespace interest_years_l214_214864

theorem interest_years (P : ℝ) (R : ℝ) (N : ℝ) (H1 : P = 2400) (H2 : (P * (R + 1) * N) / 100 - (P * R * N) / 100 = 72) : N = 3 :=
by
  -- Proof can be filled in here
  sorry

end interest_years_l214_214864


namespace smallest_d_for_inequality_l214_214331

open Real

theorem smallest_d_for_inequality :
  (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → exp (x * y) + 1 * |x^2 - y^2| ≥ exp ((x + y) / 2)) ∧
  (∀ d > 0, (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → exp (x * y) + d * |x^2 - y^2| ≥ exp ((x + y) / 2)) → d ≥ 1) :=
by
  sorry

end smallest_d_for_inequality_l214_214331


namespace free_fall_time_l214_214025

theorem free_fall_time (h : ℝ) (t : ℝ) (h_eq : h = 4.9 * t^2) (h_val : h = 490) : t = 10 :=
by
  sorry

end free_fall_time_l214_214025


namespace degree_odd_of_polynomials_l214_214743

theorem degree_odd_of_polynomials 
  (d : ℕ) 
  (P Q : Polynomial ℝ) 
  (hP_deg : P.degree = d) 
  (h_eq : P^2 + 1 = (X^2 + 1) * Q^2) 
  : Odd d :=
sorry

end degree_odd_of_polynomials_l214_214743


namespace sampling_method_is_systematic_l214_214680

-- Define the conditions
structure Grade where
  num_classes : Nat
  students_per_class : Nat
  required_student_num : Nat

-- Define our specific problem's conditions
def problem_conditions : Grade :=
  { num_classes := 12, students_per_class := 50, required_student_num := 14 }

-- State the theorem
theorem sampling_method_is_systematic (G : Grade) (h1 : G.num_classes = 12) (h2 : G.students_per_class = 50) (h3 : G.required_student_num = 14) : 
  "Systematic sampling" = "Systematic sampling" :=
by
  sorry

end sampling_method_is_systematic_l214_214680


namespace remainder_of_product_of_odd_primes_mod_32_l214_214999

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l214_214999


namespace number_of_rows_in_theater_l214_214099

theorem number_of_rows_in_theater 
  (x : ℕ)
  (h1 : ∀ (students : ℕ), students = 30 → ∃ row : ℕ, row < x ∧ ∃ a b : ℕ, a ≠ b ∧ row = a ∧ row = b)
  (h2 : ∀ (students : ℕ), students = 26 → ∃ empties : ℕ, empties ≥ 3 ∧ x - students = empties)
  : x = 29 :=
by
  sorry

end number_of_rows_in_theater_l214_214099


namespace time_for_A_l214_214715

-- Given rates of pipes A, B, and C filling the tank
variable (A B C : ℝ)

-- Condition 1: Tank filled by all three pipes in 8 hours
def combined_rate := (A + B + C = 1/8)

-- Condition 2: Pipe C is twice as fast as B
def rate_C := (C = 2 * B)

-- Condition 3: Pipe B is twice as fast as A
def rate_B := (B = 2 * A)

-- Question: To prove that pipe A alone will take 56 hours to fill the tank
theorem time_for_A (h₁ : combined_rate A B C) (h₂ : rate_C B C) (h₃ : rate_B A B) : 
  1 / A = 56 :=
by {
  sorry
}

end time_for_A_l214_214715


namespace no_integer_solutions_for_system_l214_214455

theorem no_integer_solutions_for_system :
  ∀ (y z : ℤ),
    (2 * y^2 - 2 * y * z - z^2 = 15) ∧ 
    (6 * y * z + 2 * z^2 = 60) ∧ 
    (y^2 + 8 * z^2 = 90) 
    → False :=
by 
  intro y z
  simp
  sorry

end no_integer_solutions_for_system_l214_214455


namespace capital_formula_minimum_m_l214_214321

-- Define initial conditions
def initial_capital : ℕ := 50000  -- in thousand yuan
def annual_growth_rate : ℝ := 0.5
def submission_amount : ℕ := 10000  -- in thousand yuan

-- Define remaining capital after nth year
noncomputable def remaining_capital (n : ℕ) : ℝ :=
  4500 * (3 / 2)^(n - 1) + 2000  -- in thousand yuan

-- Prove the formula for a_n
theorem capital_formula (n : ℕ) : 
  remaining_capital n = 4500 * (3 / 2)^(n - 1) + 2000 := 
by
  sorry

-- Prove the minimum value of m for which a_m > 30000
theorem minimum_m (m : ℕ) : 
  remaining_capital m > 30000 ↔ m ≥ 6 := 
by
  sorry

end capital_formula_minimum_m_l214_214321


namespace susan_ate_6_candies_l214_214332

-- Definitions based on the problem conditions
def candies_tuesday := 3
def candies_thursday := 5
def candies_friday := 2
def candies_left := 4

-- The total number of candies bought
def total_candies_bought := candies_tuesday + candies_thursday + candies_friday

-- The number of candies eaten
def candies_eaten := total_candies_bought - candies_left

-- Theorem statement to prove that Susan ate 6 candies
theorem susan_ate_6_candies : candies_eaten = 6 :=
by
  -- Proof will be provided here
  sorry

end susan_ate_6_candies_l214_214332


namespace prime_product_mod_32_l214_214974

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l214_214974


namespace new_years_day_more_frequent_l214_214235

-- Define conditions
def common_year_days : ℕ := 365
def leap_year_days : ℕ := 366
def century_is_leap_year (year : ℕ) : Prop := (year % 400 = 0)

-- Given: 23 October 1948 was a Saturday
def october_23_1948 : ℕ := 5 -- 5 corresponds to Saturday

-- Define the question proof statement
theorem new_years_day_more_frequent :
  (frequency_Sunday : ℕ) > (frequency_Monday : ℕ) :=
sorry

end new_years_day_more_frequent_l214_214235


namespace bounces_to_below_30_cm_l214_214165

theorem bounces_to_below_30_cm :
  ∃ (b : ℕ), (256 * (3 / 4)^b < 30) ∧
            (∀ (k : ℕ), k < b -> 256 * (3 / 4)^k ≥ 30) :=
by 
  sorry

end bounces_to_below_30_cm_l214_214165


namespace steven_total_seeds_l214_214818

-- Definitions based on the conditions
def apple_seed_count := 6
def pear_seed_count := 2
def grape_seed_count := 3

def apples_set_aside := 4
def pears_set_aside := 3
def grapes_set_aside := 9

def additional_seeds_needed := 3

-- The total seeds Steven already has
def total_seeds_from_fruits : ℕ :=
  apples_set_aside * apple_seed_count +
  pears_set_aside * pear_seed_count +
  grapes_set_aside * grape_seed_count

-- The total number of seeds Steven needs to collect, as given by the problem's solution
def total_seeds_needed : ℕ :=
  total_seeds_from_fruits + additional_seeds_needed

-- The actual proof statement
theorem steven_total_seeds : total_seeds_needed = 60 :=
  by
    sorry

end steven_total_seeds_l214_214818


namespace range_of_function_l214_214883

noncomputable def range_of_y : Set ℝ :=
  {y | ∃ x : ℝ, y = |x + 5| - |x - 3|}

theorem range_of_function : range_of_y = Set.Icc (-2) 12 :=
by
  sorry

end range_of_function_l214_214883


namespace smallest_scalene_triangle_perimeter_l214_214861

-- Define what it means for a number to be a prime number greater than 3
def prime_gt_3 (n : ℕ) : Prop := Prime n ∧ 3 < n

-- Define the main theorem
theorem smallest_scalene_triangle_perimeter : ∃ (a b c : ℕ), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  prime_gt_3 a ∧ prime_gt_3 b ∧ prime_gt_3 c ∧
  Prime (a + b + c) ∧ 
  (∀ (x y z : ℕ), 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    prime_gt_3 x ∧ prime_gt_3 y ∧ prime_gt_3 z ∧
    Prime (x + y + z) → (a + b + c) ≤ (x + y + z)) ∧
  a + b + c = 23 := by
    sorry

end smallest_scalene_triangle_perimeter_l214_214861


namespace berries_ratio_l214_214391

theorem berries_ratio (total_berries : ℕ) (stacy_berries : ℕ) (ratio_stacy_steve : ℕ)
  (h_total : total_berries = 1100) (h_stacy : stacy_berries = 800)
  (h_ratio : stacy_berries = 4 * ratio_stacy_steve) :
  ratio_stacy_steve / (total_berries - stacy_berries - ratio_stacy_steve) = 2 :=
by {
  sorry
}

end berries_ratio_l214_214391


namespace reinforcement_1600_l214_214310

/-- A garrison of 2000 men has provisions for 54 days. After 18 days, a reinforcement arrives, and it is now found that the provisions will last only for 20 days more. We define the initial total provisions, remaining provisions after 18 days, and form equations to solve for the unknown reinforcement R.
We need to prove that R = 1600 given these conditions.
-/
theorem reinforcement_1600 (P : ℕ) (M1 M2 : ℕ) (D1 D2 : ℕ) (R : ℕ) :
  M1 = 2000 →
  D1 = 54 →
  D2 = 20 →
  M2 = 2000 + R →
  P = M1 * D1 →
  (M1 * (D1 - 18) = M2 * D2) →
  R = 1600 :=
by
  intros hM1 hD1 hD2 hM2 hP hEquiv
  sorry

end reinforcement_1600_l214_214310


namespace value_of_b_l214_214501

-- Definitions
def A := 45  -- in degrees
def B := 60  -- in degrees
def a := 10  -- length of side a

-- Assertion
theorem value_of_b : (b : ℝ) = 5 * Real.sqrt 6 :=
by
  -- Definitions used in previous problem conditions
  let sin_A := Real.sin (Real.pi * A / 180)
  let sin_B := Real.sin (Real.pi * B / 180)
  -- Applying the Law of Sines
  have law_of_sines := (a / sin_A) = (b / sin_B)
  -- Simplified calculation of b (not provided here; proof required later)
  sorry

end value_of_b_l214_214501


namespace tomatoes_price_per_pound_l214_214280

noncomputable def price_per_pound (cost_per_pound : ℝ) (loss_percent : ℝ) (profit_percent : ℝ) : ℝ :=
  let remaining_percent := 1 - loss_percent / 100
  let desired_total := (1 + profit_percent / 100) * cost_per_pound
  desired_total / remaining_percent

theorem tomatoes_price_per_pound :
  price_per_pound 0.80 15 8 = 1.02 :=
by
  sorry

end tomatoes_price_per_pound_l214_214280


namespace time_of_free_fall_l214_214024

theorem time_of_free_fall (h : ℝ) (t : ℝ) (height_fall_eq : h = 4.9 * t^2) (initial_height : h = 490) : t = 10 :=
by
  -- Proof is omitted
  sorry

end time_of_free_fall_l214_214024


namespace find_k_multiple_l214_214698

theorem find_k_multiple (a b k : ℕ) (h1 : a = b + 5) (h2 : a + b = 13) 
  (h3 : 3 * (a + 7) = k * (b + 7)) : k = 4 := sorry

end find_k_multiple_l214_214698


namespace robin_packages_l214_214661

theorem robin_packages (p t n : ℕ) (h1 : p = 18) (h2 : t = 486) : t / p = n ↔ n = 27 :=
by
  rw [h1, h2]
  norm_num
  sorry

end robin_packages_l214_214661


namespace rectangle_semicircle_area_split_l214_214815

open Real

/-- The main problem statement -/
theorem rectangle_semicircle_area_split 
  (A B D C N U T : ℝ)
  (AU_AN_UAlengths : AU = 84 ∧ AN = 126 ∧ UB = 168)
  (area_ratio : ∃ (ℓ : ℝ), ∃ (N U T : ℝ), 1 / 2 = area_differ / (area_left + area_right))
  (DA_calculation : DA = 63 * sqrt 6) :
  63 + 6 = 69
:=
sorry

end rectangle_semicircle_area_split_l214_214815


namespace teams_in_BIG_M_l214_214230

theorem teams_in_BIG_M (n : ℕ) (h : n * (n - 1) / 2 = 36) : n = 9 :=
sorry

end teams_in_BIG_M_l214_214230


namespace difference_of_smallest_integers_l214_214836

open Nat

theorem difference_of_smallest_integers :
  ∃ n₁ n₂, (∀ k, 2 ≤ k ∧ k ≤ 12 → n₁ % k = 1) ∧ (∀ k, 2 ≤ k ∧ k ≤ 12 → n₂ % k = 1) ∧ (n₂ - n₁ = 27720) := by
  -- The proof would go here
  sorry

end difference_of_smallest_integers_l214_214836


namespace factorization_a_minus_b_l214_214397

-- Define the problem in Lean 4
theorem factorization_a_minus_b (a b : ℤ) : 
  (∀ y : ℝ, 2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b))
  → a - b = 7 := 
by 
  sorry

end factorization_a_minus_b_l214_214397


namespace Dima_broke_more_l214_214732

theorem Dima_broke_more (D F : ℕ) (h : 2 * D + 7 * F = 3 * (D + F)) : D = 4 * F :=
sorry

end Dima_broke_more_l214_214732


namespace typists_initial_group_l214_214914

theorem typists_initial_group
  (T : ℕ) 
  (h1 : 0 < T) 
  (h2 : T * (240 / 40 * 20) = 2400) : T = 10 :=
by
  sorry

end typists_initial_group_l214_214914


namespace smallest_circle_radius_polygonal_chain_l214_214853

theorem smallest_circle_radius_polygonal_chain (l : ℝ) (hl : l = 1) : ∃ (r : ℝ), r = 0.5 := 
sorry

end smallest_circle_radius_polygonal_chain_l214_214853


namespace chord_length_range_l214_214496

open Real

def chord_length_ge (t : ℝ) : Prop :=
  let r := sqrt 8
  let l := (4 * sqrt 2) / 3
  let d := abs t / sqrt 2
  let s := l / 2
  s ≤ sqrt (r^2 - d^2)

theorem chord_length_range (t : ℝ) : chord_length_ge t ↔ -((8 * sqrt 2) / 3) ≤ t ∧ t ≤ (8 * sqrt 2) / 3 :=
by
  sorry

end chord_length_range_l214_214496


namespace ellipse_focal_length_l214_214349

theorem ellipse_focal_length {m : ℝ} : 
  (m > 2 ∧ 4 ≤ 10 - m ∧ 4 ≤ m - 2) → 
  (10 - m - (m - 2) = 4) ∨ (m - 2 - (10 - m) = 4) :=
by
  sorry

end ellipse_focal_length_l214_214349


namespace prob_eliminated_after_semifinal_expected_value_ξ_l214_214779

-- Define the problem conditions
variables (Ω : Type*) [MeasureSpace Ω]
variables (A B C : Event Ω)
variables (P : MeasureTheory.Measure Ω)
variable hA : P(A) = 2/3
variable hB : P(B) = 1/2
variable hC : P(C) = 1/3
variable h_ind : pairwise (independent P)

-- Problem 1: Probability of elimination during semifinal stage is 1/3
theorem prob_eliminated_after_semifinal : P(A ∩ Bᶜ) = 1 / 3 :=
sorry

-- Define the random variable ξ
def ξ : Ω → ℕ
| ω := if ¬A ω then 1 else if ¬B ω then 2 else 3

-- Problem 2: Expected value of ξ is 2
theorem expected_value_ξ : MeasureTheory.ProbabilityTheory.ProbabilityMassFunction.expected_value (MeasureTheory.ProbabilityTheory.ProbabilityMassFunction.from_fun ξ) = 2 :=
sorry

end prob_eliminated_after_semifinal_expected_value_ξ_l214_214779


namespace natural_number_x_l214_214752

theorem natural_number_x (x : ℕ) (A : ℕ → ℕ) (h : 3 * (A (x + 1))^3 = 2 * (A (x + 2))^2 + 6 * (A (x + 1))^2) : x = 4 :=
sorry

end natural_number_x_l214_214752


namespace point_relationship_l214_214352

variables {m x1 x2 y1 y2 : ℝ}

def quadratic_function (x : ℝ) (m : ℝ) : ℝ :=
  (x + m - 3)*(x - m) + 3

theorem point_relationship 
  (hx1_lt_x2 : x1 < x2)
  (hA : y1 = quadratic_function x1 m)
  (hB : y2 = quadratic_function x2 m)
  (h_sum_lt : x1 + x2 < 3) :
  y1 > y2 :=
sorry

end point_relationship_l214_214352


namespace one_product_success_profit_distribution_l214_214054

namespace CompanyResearch

-- Conditions
def probs : probability_space ℝ :=
{ sample_space := { successesA := 0.75, successesB := 0.6},
  prob := sorry }

def successA : events ℝ := { event := λ ω, ω.successesA}
def successB : events ℝ := { event := λ ω, ω.successesB}

-- (1) Probability that exactly one new product is successfully developed
theorem one_product_success (probs : ℝ) (successA successB : prob_event probs) : 
  (Pr[successA] = 3/4) → 
  (Pr[successB] = 3/5) → 
  (independent successA successB) → 
  Pr[successA ∩ successBᶜ ∪ successAᶜ ∩ successB] = 3/20 + 6/20 := 
sorry

-- (2) Distribution of the company's profit
inductive Profit
| neg_90 : Profit
| pos_50 : Profit
| pos_80 : Profit
| pos_220 : Profit

def profit_dist : probability_space Profit :=
{ sample_space := { Profit.neg_90, Profit.pos_50, Profit.pos_80, Profit.pos_220 },
  prob := λ x, match x with
    | Profit.neg_90 := 0.1
    | Profit.pos_50 := 0.15
    | Profit.pos_80 := 0.3
    | Profit.pos_220 := 0.45
  end := sorry }

theorem profit_distribution (probs : ℝ) (successA successB : prob_event probs) : 
  (Pr[successA] = 3/4) → 
  (Pr[successB] = 3/5) → 
  (independent successA successB) →
  immeasurable (profit_dist.ProbabilitySizeSpace) :=
sorry

end CompanyResearch

end one_product_success_profit_distribution_l214_214054


namespace distance_from_plate_to_bottom_edge_l214_214049

theorem distance_from_plate_to_bottom_edge :
  ∀ (W T d : ℕ), W = 73 ∧ T = 20 ∧ (T + d = W) → d = 53 :=
by
  intros W T d
  rintro ⟨hW, hT, h⟩
  rw [hW, hT] at h
  linarith

end distance_from_plate_to_bottom_edge_l214_214049


namespace boxes_of_apples_l214_214407

theorem boxes_of_apples (apples_per_crate crates_delivered rotten_apples apples_per_box : ℕ) 
       (h1 : apples_per_crate = 42) 
       (h2 : crates_delivered = 12) 
       (h3 : rotten_apples = 4) 
       (h4 : apples_per_box = 10) : 
       crates_delivered * apples_per_crate - rotten_apples = 500 ∧
       (crates_delivered * apples_per_crate - rotten_apples) / apples_per_box = 50 := by
  sorry

end boxes_of_apples_l214_214407


namespace abs_expr_evaluation_l214_214077

theorem abs_expr_evaluation : abs (abs (-abs (-1 + 2) - 2) + 3) = 6 := by
  sorry

end abs_expr_evaluation_l214_214077


namespace distance_between_stripes_l214_214316

theorem distance_between_stripes
  (h1 : ∀ (curbs_are_parallel : Prop), curbs_are_parallel → true)
  (h2 : ∀ (distance_between_curbs : ℝ), distance_between_curbs = 60 → true)
  (h3 : ∀ (length_of_curb : ℝ), length_of_curb = 20 → true)
  (h4 : ∀ (stripe_length : ℝ), stripe_length = 75 → true) :
  ∃ (d : ℝ), d = 16 :=
by
  sorry

end distance_between_stripes_l214_214316


namespace remainder_of_M_l214_214988

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l214_214988


namespace intersection_of_sets_example_l214_214520

theorem intersection_of_sets_example :
  let M := { x : ℝ | 0 < x ∧ x < 4 }
  let N := { x : ℝ | 1 / 3 ≤ x ∧ x ≤ 5 }
  let expected := { x : ℝ | 1 / 3 ≤ x ∧ x < 4 }
  (M ∩ N) = expected :=
by
  sorry

end intersection_of_sets_example_l214_214520


namespace initial_books_calculation_l214_214812

-- Definitions based on conditions
def total_books : ℕ := 77
def additional_books : ℕ := 23

-- Statement of the problem
theorem initial_books_calculation : total_books - additional_books = 54 :=
by
  sorry

end initial_books_calculation_l214_214812


namespace cube_volume_given_surface_area_l214_214283

theorem cube_volume_given_surface_area (A : ℝ) (V : ℝ) :
  A = 96 → V = 64 :=
by
  sorry

end cube_volume_given_surface_area_l214_214283


namespace joan_and_karl_sofas_l214_214234

variable (J K : ℝ)

theorem joan_and_karl_sofas (hJ : J = 230) (hSum : J + K = 600) :
  2 * J - K = 90 :=
by
  sorry

end joan_and_karl_sofas_l214_214234


namespace unique_plants_in_all_beds_l214_214612

theorem unique_plants_in_all_beds:
  let A := 600
  let B := 500
  let C := 400
  let D := 300
  let AB := 80
  let AC := 70
  let ABD := 40
  let BC := 0
  let AD := 0
  let BD := 0
  let CD := 0
  let ABC := 0
  let ACD := 0
  let BCD := 0
  let ABCD := 0
  A + B + C + D - AB - AC - BC - AD - BD - CD + ABC + ABD + ACD + BCD - ABCD = 1690 :=
by
  sorry

end unique_plants_in_all_beds_l214_214612


namespace problem_solution_l214_214756

def satisfies_conditions (x y : ℚ) : Prop :=
  (3 * x + y = 6) ∧ (x + 3 * y = 6)

theorem problem_solution :
  ∃ (x y : ℚ), satisfies_conditions x y ∧ 3 * x^2 + 5 * x * y + 3 * y^2 = 24.75 :=
by
  sorry

end problem_solution_l214_214756


namespace remainder_when_M_divided_by_32_l214_214944

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l214_214944


namespace cost_of_each_toy_l214_214301

theorem cost_of_each_toy (initial_money spent_money remaining_money toys_count toy_cost : ℕ) 
  (h1 : initial_money = 57)
  (h2 : spent_money = 27)
  (h3 : remaining_money = initial_money - spent_money)
  (h4 : toys_count = 5)
  (h5 : remaining_money / toys_count = toy_cost) :
  toy_cost = 6 :=
by
  sorry

end cost_of_each_toy_l214_214301


namespace probability_second_roll_twice_first_l214_214708

theorem probability_second_roll_twice_first :
  let outcomes := [(1, 2), (2, 4), (3, 6)]
  let total_outcomes := 36
  let favorable_outcomes := 3
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 12 :=
by
  sorry

end probability_second_roll_twice_first_l214_214708


namespace copy_pages_count_l214_214018

-- Definitions and conditions
def cost_per_page : ℕ := 5  -- Cost per page in cents
def total_money : ℕ := 50 * 100  -- Total money in cents

-- Proof goal
theorem copy_pages_count : total_money / cost_per_page = 1000 := 
by sorry

end copy_pages_count_l214_214018


namespace profit_percentage_of_cp_is_75_percent_of_sp_l214_214699

/-- If the cost price (CP) is 75% of the selling price (SP), then the profit percentage is 33.33% -/
theorem profit_percentage_of_cp_is_75_percent_of_sp (SP : ℝ) (h : SP > 0) (CP : ℝ) (hCP : CP = 0.75 * SP) :
  (SP - CP) / CP * 100 = 33.33 :=
by
  sorry

end profit_percentage_of_cp_is_75_percent_of_sp_l214_214699


namespace recurrent_sequence_solution_l214_214622

theorem recurrent_sequence_solution (a : ℕ → ℕ) : 
  (a 1 = 1 ∧ ∀ n, n ≥ 2 → a n = 2 * a (n - 1) + 2^n) →
  (∀ n, n ≥ 1 → a n = (2 * n - 1) * 2^(n - 1)) :=
by
  sorry

end recurrent_sequence_solution_l214_214622


namespace pens_distribution_l214_214342

theorem pens_distribution (friends : ℕ) (pens : ℕ) (at_least_one : ℕ) 
  (h1 : friends = 4) (h2 : pens = 10) (h3 : at_least_one = 1) 
  (h4 : ∀ f : ℕ, f < friends → at_least_one ≤ f) :
  ∃ ways : ℕ, ways = 142 := 
sorry

end pens_distribution_l214_214342


namespace ratio_of_thermometers_to_hotwater_bottles_l214_214037

theorem ratio_of_thermometers_to_hotwater_bottles (T H : ℕ) (thermometer_price hotwater_bottle_price total_sales : ℕ) 
  (h1 : thermometer_price = 2) (h2 : hotwater_bottle_price = 6) (h3 : total_sales = 1200) (h4 : H = 60) 
  (h5 : total_sales = thermometer_price * T + hotwater_bottle_price * H) : 
  T / H = 7 :=
by
  sorry

end ratio_of_thermometers_to_hotwater_bottles_l214_214037


namespace mod_inverse_3_40_l214_214339

theorem mod_inverse_3_40 : 3 * 27 % 40 = 1 := by
  sorry

end mod_inverse_3_40_l214_214339


namespace minimum_value_of_a_l214_214211

theorem minimum_value_of_a (a b : ℕ) (h₁ : b - a = 2013) 
(h₂ : ∃ x : ℕ, x^2 - a * x + b = 0) : a = 93 :=
sorry

end minimum_value_of_a_l214_214211


namespace remainder_when_M_divided_by_32_l214_214950

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l214_214950


namespace positive_integer_solution_lcm_eq_sum_l214_214453

def is_lcm (x y z m : Nat) : Prop :=
  ∃ (d : Nat), x = d * (Nat.gcd y z) ∧ y = d * (Nat.gcd x z) ∧ z = d * (Nat.gcd x y) ∧
  x * y * z / Nat.gcd x (Nat.gcd y z) = m

theorem positive_integer_solution_lcm_eq_sum :
  ∀ (a b c : Nat), 0 < a → 0 < b → 0 < c → is_lcm a b c (a + b + c) → (a, b, c) = (a, 2 * a, 3 * a) := by
    sorry

end positive_integer_solution_lcm_eq_sum_l214_214453


namespace evaluate_expression_l214_214456

theorem evaluate_expression : 
  1 + 2 / (3 + 4 / (5 + 6 / 7)) = 233 / 151 := 
by 
  sorry

end evaluate_expression_l214_214456


namespace maria_total_baggies_l214_214806

def choc_chip_cookies := 33
def oatmeal_cookies := 2
def cookies_per_bag := 5

def total_cookies := choc_chip_cookies + oatmeal_cookies

def total_baggies (total_cookies : Nat) (cookies_per_bag : Nat) : Nat :=
  total_cookies / cookies_per_bag

theorem maria_total_baggies : total_baggies total_cookies cookies_per_bag = 7 :=
  by
    -- Steps proving the equivalence can be done here
    sorry

end maria_total_baggies_l214_214806


namespace inequality_solution_range_l214_214775

theorem inequality_solution_range (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 5 → x ^ 2 + a * x + 4 < 0) ↔ a < -4 :=
by 
  sorry

end inequality_solution_range_l214_214775


namespace train_b_leaves_after_train_a_l214_214411

noncomputable def time_difference := 2

theorem train_b_leaves_after_train_a 
  (speedA speedB distance t : ℝ) 
  (h1 : speedA = 30)
  (h2 : speedB = 38)
  (h3 : distance = 285)
  (h4 : distance = speedB * t)
  : time_difference = (distance - speedA * t) / speedA := 
by 
  sorry

end train_b_leaves_after_train_a_l214_214411


namespace peter_fraction_is_1_8_l214_214813

-- Define the total number of slices, slices Peter ate alone, and slices Peter shared with Paul
def total_slices := 16
def peter_alone_slices := 1
def shared_slices := 2

-- Define the fraction of the pizza Peter ate alone
def peter_fraction_alone := peter_alone_slices / total_slices

-- Define the fraction of the pizza Peter ate from the shared slices
def shared_fraction := shared_slices * (1 / 2) / total_slices

-- Define the total fraction of the pizza Peter ate
def total_fraction_peter_ate := peter_fraction_alone + shared_fraction

-- Prove that the total fraction of the pizza Peter ate is 1/8
theorem peter_fraction_is_1_8 : total_fraction_peter_ate = 1/8 := by
  sorry

end peter_fraction_is_1_8_l214_214813


namespace triangle_interior_angle_contradiction_l214_214045

theorem triangle_interior_angle_contradiction :
  (∀ (A B C : ℝ), A + B + C = 180 ∧ A > 60 ∧ B > 60 ∧ C > 60 → false) :=
by
  sorry

end triangle_interior_angle_contradiction_l214_214045


namespace multiple_of_9_l214_214844

theorem multiple_of_9 (a : ℤ) :
  a ∈ {50, 40, 35, 45, 55} → (∃ k : ℤ, a = 9 * k) ↔ a = 45 :=
by
  intro h
  cases h
  · constructor
    · intro ⟨k, hk⟩
      contradiction
    · intro ha
      rw ha
      use 5
  · constructor
    · intro ⟨k, hk⟩
      contradiction
    · intro ha
      rw ha
      use 5
  · constructor
    · intro ⟨k, hk⟩
      contradiction
    · intro ha
      rw ha
      use 5
  · constructor
    · intro ⟨k, hk⟩
      use k
      exact hk
    · intro ha
      use 5
      exact ha
  · constructor
    · intro ⟨k, hk⟩
      contradiction
    · intro ha
      rw ha
      use 5
  sorry

end multiple_of_9_l214_214844


namespace expansion_coefficient_l214_214373

theorem expansion_coefficient (x : ℝ) (h : x ≠ 0): 
  (∃ r : ℕ, (7 - (3 / 2 : ℝ) * r = 1) ∧ Nat.choose 7 r = 35) := 
  sorry

end expansion_coefficient_l214_214373


namespace num_chords_num_triangles_l214_214821

noncomputable def num_points : ℕ := 10

theorem num_chords (n : ℕ) (h : n = num_points) : (n.choose 2) = 45 := by
  sorry

theorem num_triangles (n : ℕ) (h : n = num_points) : (n.choose 3) = 120 := by
  sorry

end num_chords_num_triangles_l214_214821


namespace total_boys_in_camp_l214_214850

theorem total_boys_in_camp (T : ℝ) (h : 0.70 * (0.20 * T) = 28) : T = 200 := 
by
  sorry

end total_boys_in_camp_l214_214850


namespace projectile_height_l214_214270

theorem projectile_height (t : ℝ) (h : (-16 * t^2 + 80 * t = 100)) : t = 2.5 :=
sorry

end projectile_height_l214_214270


namespace option_d_correct_l214_214492

theorem option_d_correct (a b : ℝ) (h : a > b) : -b > -a :=
sorry

end option_d_correct_l214_214492


namespace ratio_p_q_l214_214069

section ProbabilityProof

-- Definitions and constants as per conditions
def N := Nat.factorial 15

def num_ways_A : ℕ := 4 * (Nat.factorial 9 / (Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 3))
def num_ways_B : ℕ := 4 * 3

def p : ℚ := num_ways_A / N
def q : ℚ := num_ways_B / N

-- Theorem: Prove that the ratio p/q is 560
theorem ratio_p_q : p / q = 560 := by
  sorry

end ProbabilityProof

end ratio_p_q_l214_214069


namespace at_least_one_heart_or_king_l214_214304

noncomputable def probability_heart_or_king_in_three_draws : ℚ := 
  1 - (36 / 52)^3

theorem at_least_one_heart_or_king :
  probability_heart_or_king_in_three_draws = 1468 / 2197 := 
by
  sorry

end at_least_one_heart_or_king_l214_214304


namespace decimal_6_to_binary_is_110_l214_214562

def decimal_to_binary (n : ℕ) : ℕ :=
  -- This is just a placeholder definition. Adjust as needed for formalization.
  sorry

theorem decimal_6_to_binary_is_110 :
  decimal_to_binary 6 = 110 := 
sorry

end decimal_6_to_binary_is_110_l214_214562


namespace product_of_odd_primes_mod_32_l214_214951

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l214_214951


namespace find_N_l214_214735

theorem find_N :
  ∃ N : ℕ, N > 1 ∧
    (1743 % N = 2019 % N) ∧ (2019 % N = 3008 % N) ∧ N = 23 :=
by
  sorry

end find_N_l214_214735


namespace total_amount_l214_214164

variable (A B C : ℕ)
variable (h1 : C = 495)
variable (h2 : (A - 10) * 18 = (B - 20) * 11)
variable (h3 : (B - 20) * 24 = (C - 15) * 18)

theorem total_amount (A B C : ℕ) (h1 : C = 495)
  (h2 : (A - 10) * 18 = (B - 20) * 11)
  (h3 : (B - 20) * 24 = (C - 15) * 18) :
  A + B + C = 1105 :=
sorry

end total_amount_l214_214164


namespace probabilityOfWearingSunglassesGivenCap_l214_214249

-- Define the conditions as Lean constants
def peopleWearingSunglasses : ℕ := 80
def peopleWearingCaps : ℕ := 60
def probabilityOfWearingCapGivenSunglasses : ℚ := 3 / 8
def peopleWearingBoth : ℕ := (3 / 8) * 80

-- Prove the desired probability
theorem probabilityOfWearingSunglassesGivenCap : (peopleWearingBoth / peopleWearingCaps = 1 / 2) :=
by
  -- sorry is used here to skip the proof
  sorry

end probabilityOfWearingSunglassesGivenCap_l214_214249


namespace probability_no_adjacent_birch_l214_214058

theorem probability_no_adjacent_birch:
  let total_trees := 12
  let maple_trees := 3
  let oak_trees := 4
  let birch_trees := 5

  let total_arrangements := Nat.factorial total_trees
  
  -- Arrangements of maples and oaks
  let non_birch_trees := maple_trees + oak_trees
  let arrangements_without_birch := Nat.factorial non_birch_trees
  
  -- Possible slots to place birch trees
  let slots_for_birch := non_birch_trees + 1
  let ways_to_choose_slots := Nat.choose slots_for_birch birch_trees

  -- Permutation of birch trees
  let permutation_of_birch := Nat.factorial birch_trees

  -- Favorable arrangements
  let favorable_arrangements := arrangements_without_birch * ways_to_choose_slots * permutation_of_birch

  -- Probability
  let probability := favorable_arrangements / total_arrangements
  probability = 7 / 99 :=
by {
  sorry
}

end probability_no_adjacent_birch_l214_214058


namespace choose_president_and_committee_l214_214370

theorem choose_president_and_committee (n : ℕ) (k : ℕ) (hn : n = 10) (hk : k = 3) : 
  (finset.card (finset.univ : finset (fin (n - 1))).choose k) * n = 840 :=
by
  rw [hn, hk]
  have h1 : finset.card (finset.univ : finset (fin 9)).choose 3 = nat.choose 9 3 := rfl
  rw [h1, nat.choose]
  exact dec_trivial

end choose_president_and_committee_l214_214370


namespace part1_parallel_vectors_part2_perpendicular_vectors_l214_214625

def vect_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

def vect_dot (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem part1_parallel_vectors 
  (m : ℝ)
  (OA OB OC : ℝ × ℝ)
  (HOA : OA = (3, -4))
  (HOB : OB = (6, -3))
  (HOC : OC = (5 - m, -3 - m))
  (HAB_parallel_BC : vect_sub OB OA = (3, 1) ∧ vect_sub OC OB = (-1 - m, -m)) :
  m = 1 / 2 :=
sorry

theorem part2_perpendicular_vectors 
  (m : ℝ)
  (OA OB OC : ℝ × ℝ)
  (HOA : OA = (3, -4))
  (HOB : OB = (6, -3))
  (HOC : OC = (5 - m, -3 - m))
  (HAB_perp_AC : vect_dot (vect_sub OB OA) (vect_sub OC OA) = 0) :
  m = 7 / 4 :=
sorry

end part1_parallel_vectors_part2_perpendicular_vectors_l214_214625


namespace perpendicular_line_through_point_l214_214341

def point : ℝ × ℝ := (1, 0)

def given_line (x y : ℝ) : Prop := x - y + 2 = 0

def is_perpendicular_to (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, l1 x y → l2 (y - x) (-x - y + 2)

def target_line (x y : ℝ) : Prop := x + y - 1 = 0

theorem perpendicular_line_through_point (l1 : ℝ → ℝ → Prop) (p : ℝ × ℝ) :
  given_line = l1 ∧ p = point →
  (∃ l2 : ℝ → ℝ → Prop, is_perpendicular_to l1 l2 ∧ l2 p.1 p.2) →
  target_line p.1 p.2 :=
by
  intro hp hl2
  sorry

end perpendicular_line_through_point_l214_214341


namespace find_baking_soda_boxes_l214_214053

-- Define the quantities and costs
def num_flour_boxes := 3
def cost_per_flour_box := 3
def num_egg_trays := 3
def cost_per_egg_tray := 10
def num_milk_liters := 7
def cost_per_milk_liter := 5
def baking_soda_cost_per_box := 3
def total_cost := 80

-- Define the total cost of flour, eggs, and milk
def total_flour_cost := num_flour_boxes * cost_per_flour_box
def total_egg_cost := num_egg_trays * cost_per_egg_tray
def total_milk_cost := num_milk_liters * cost_per_milk_liter

-- Define the total cost of non-baking soda items
def total_non_baking_soda_cost := total_flour_cost + total_egg_cost + total_milk_cost

-- Define the remaining cost for baking soda
def baking_soda_total_cost := total_cost - total_non_baking_soda_cost

-- Define the number of baking soda boxes
def num_baking_soda_boxes := baking_soda_total_cost / baking_soda_cost_per_box

theorem find_baking_soda_boxes : num_baking_soda_boxes = 2 :=
by
  sorry

end find_baking_soda_boxes_l214_214053


namespace more_stable_yield_A_l214_214705

theorem more_stable_yield_A (s_A s_B : ℝ) (hA : s_A * s_A = 794) (hB : s_B * s_B = 958) : s_A < s_B :=
by {
  sorry -- Details of the proof would go here
}

end more_stable_yield_A_l214_214705


namespace running_track_diameter_l214_214190

theorem running_track_diameter 
  (running_track_width : ℕ) 
  (garden_ring_width : ℕ) 
  (play_area_diameter : ℕ) 
  (h1 : running_track_width = 4) 
  (h2 : garden_ring_width = 6) 
  (h3 : play_area_diameter = 14) :
  (2 * ((play_area_diameter / 2) + garden_ring_width + running_track_width)) = 34 := 
by
  sorry

end running_track_diameter_l214_214190


namespace article_production_l214_214630

-- Conditions
variables (x z : ℕ) (hx : 0 < x) (hz : 0 < z)
-- The given condition: x men working x hours a day for x days produce 2x^2 articles.
def articles_produced_x (x : ℕ) : ℕ := 2 * x^2

-- The question: the number of articles produced by z men working z hours a day for z days
def articles_produced_z (x z : ℕ) : ℕ := 2 * z^3 / x

-- Prove that the number of articles produced by z men working z hours a day for z days is 2 * (z^3) / x
theorem article_production (hx : 0 < x) (hz : 0 < z) :
  articles_produced_z x z = 2 * z^3 / x :=
sorry

end article_production_l214_214630


namespace smallest_portion_bread_l214_214426

theorem smallest_portion_bread (a d : ℚ) (h1 : 5 * a = 100) (h2 : 24 * d = 11 * a) :
  a - 2 * d = 5 / 3 :=
by
  -- Solution proof goes here...
  sorry -- placeholder for the proof

end smallest_portion_bread_l214_214426


namespace intersection_of_M_and_N_l214_214514

variable {α : Type*} [LinearOrder α] [OrderBot α] [OrderTop α]

def M (x : α) : Prop := 0 < x ∧ x < 4

def N (x : α) : Prop := (1 / 3 : α) ≤ x ∧ x ≤ 5

theorem intersection_of_M_and_N {x : α} : (M x ∧ N x) ↔ ((1 / 3 : α) ≤ x ∧ x < 4) :=
by 
  sorry

end intersection_of_M_and_N_l214_214514


namespace freddy_travel_time_l214_214888

theorem freddy_travel_time (dist_A_B : ℝ) (time_Eddy : ℝ) (dist_A_C : ℝ) (speed_ratio : ℝ) (travel_time_Freddy : ℝ) :
  dist_A_B = 540 ∧ time_Eddy = 3 ∧ dist_A_C = 300 ∧ speed_ratio = 2.4 →
  travel_time_Freddy = dist_A_C / (dist_A_B / time_Eddy / speed_ratio) :=
  sorry

end freddy_travel_time_l214_214888


namespace solution_correct_l214_214118

variable (a b c d : ℝ)

theorem solution_correct (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end solution_correct_l214_214118


namespace find_N_l214_214734

theorem find_N :
  ∃ N : ℕ, N > 1 ∧
    (1743 % N = 2019 % N) ∧ (2019 % N = 3008 % N) ∧ N = 23 :=
by
  sorry

end find_N_l214_214734


namespace intersection_of_M_and_N_l214_214511

variable {α : Type*} [LinearOrder α] [OrderBot α] [OrderTop α]

def M (x : α) : Prop := 0 < x ∧ x < 4

def N (x : α) : Prop := (1 / 3 : α) ≤ x ∧ x ≤ 5

theorem intersection_of_M_and_N {x : α} : (M x ∧ N x) ↔ ((1 / 3 : α) ≤ x ∧ x < 4) :=
by 
  sorry

end intersection_of_M_and_N_l214_214511


namespace boxes_of_apples_l214_214408

theorem boxes_of_apples (apples_per_crate crates_delivered rotten_apples apples_per_box : ℕ) 
       (h1 : apples_per_crate = 42) 
       (h2 : crates_delivered = 12) 
       (h3 : rotten_apples = 4) 
       (h4 : apples_per_box = 10) : 
       crates_delivered * apples_per_crate - rotten_apples = 500 ∧
       (crates_delivered * apples_per_crate - rotten_apples) / apples_per_box = 50 := by
  sorry

end boxes_of_apples_l214_214408


namespace smallest_lcm_not_multiple_of_25_l214_214297

theorem smallest_lcm_not_multiple_of_25 (n : ℕ) (h1 : n % 36 = 0) (h2 : n % 45 = 0) (h3 : n % 25 ≠ 0) : n = 180 := 
by 
  sorry

end smallest_lcm_not_multiple_of_25_l214_214297


namespace absolute_value_equation_solution_l214_214611

theorem absolute_value_equation_solution (a b c : ℝ) :
  (∀ x y z : ℝ, |a * x + b * y + c * z| + |b * x + c * y + a * z| + |c * x + a * y + b * z| = |x| + |y| + |z|) ↔
  ((a = 0 ∧ b = 0 ∧ (c = 1 ∨ c = -1)) ∨ 
   (a = 0 ∧ c = 0 ∧ (b = 1 ∨ b = -1)) ∨ 
   (b = 0 ∧ c = 0 ∧ (a = 1 ∨ a = -1))) :=
by
  sorry

end absolute_value_equation_solution_l214_214611


namespace probability_x_gt_3y_in_rectangle_l214_214003

noncomputable def probability_of_x_gt_3y :ℝ :=
  let base := 2010
  let height := 2011
  let triangle_height := 670
  (1/2 * base * triangle_height) / (base * height)

theorem probability_x_gt_3y_in_rectangle:
  probability_of_x_gt_3y = 335 / 2011 := 
by
  sorry

end probability_x_gt_3y_in_rectangle_l214_214003


namespace six_rational_right_triangles_same_perimeter_l214_214449

theorem six_rational_right_triangles_same_perimeter :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ a₄ b₄ c₄ a₅ b₅ c₅ a₆ b₆ c₆ : ℕ),
    a₁^2 + b₁^2 = c₁^2 ∧ a₂^2 + b₂^2 = c₂^2 ∧ a₃^2 + b₃^2 = c₃^2 ∧
    a₄^2 + b₄^2 = c₄^2 ∧ a₅^2 + b₅^2 = c₅^2 ∧ a₆^2 + b₆^2 = c₆^2 ∧
    a₁ + b₁ + c₁ = 720 ∧ a₂ + b₂ + c₂ = 720 ∧ a₃ + b₃ + c₃ = 720 ∧
    a₄ + b₄ + c₄ = 720 ∧ a₅ + b₅ + c₅ = 720 ∧ a₆ + b₆ + c₆ = 720 ∧
    (a₁, b₁, c₁) ≠ (a₂, b₂, c₂) ∧ (a₁, b₁, c₁) ≠ (a₃, b₃, c₃) ∧
    (a₁, b₁, c₁) ≠ (a₄, b₄, c₄) ∧ (a₁, b₁, c₁) ≠ (a₅, b₅, c₅) ∧
    (a₁, b₁, c₁) ≠ (a₆, b₆, c₆) ∧ (a₂, b₂, c₂) ≠ (a₃, b₃, c₃) ∧
    (a₂, b₂, c₂) ≠ (a₄, b₄, c₄) ∧ (a₂, b₂, c₂) ≠ (a₅, b₅, c₅) ∧
    (a₂, b₂, c₂) ≠ (a₆, b₆, c₆) ∧ (a₃, b₃, c₃) ≠ (a₄, b₄, c₄) ∧
    (a₃, b₃, c₃) ≠ (a₅, b₅, c₅) ∧ (a₃, b₃, c₃) ≠ (a₆, b₆, c₆) ∧
    (a₄, b₄, c₄) ≠ (a₅, b₅, c₅) ∧ (a₄, b₄, c₄) ≠ (a₆, b₆, c₆) ∧
    (a₅, b₅, c₅) ≠ (a₆, b₆, c₆) :=
sorry

end six_rational_right_triangles_same_perimeter_l214_214449


namespace garden_length_l214_214917

theorem garden_length (P b l : ℕ) (h1 : P = 500) (h2 : b = 100) : l = 150 :=
by
  sorry

end garden_length_l214_214917


namespace range_of_a_l214_214117

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = Real.exp x + a * x) ∧ (∃ x, 0 < x ∧ (DifferentiableAt ℝ f x) ∧ (deriv f x = 0)) → a < -1 :=
by
  sorry

end range_of_a_l214_214117


namespace tanya_number_75_less_l214_214663

def rotate180 (d : ℕ) : ℕ :=
  match d with
  | 0 => 0
  | 1 => 1
  | 6 => 9
  | 8 => 8
  | 9 => 6
  | _ => 0 -- invalid assumption for digits outside the defined scope

def two_digit_upside_down (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  10 * rotate180 units + rotate180 tens

theorem tanya_number_75_less (n : ℕ) : 
  ∀ n, (∃ a b, n = 10 * a + b ∧ (a = 0 ∨ a = 1 ∨ a = 6 ∨ a = 8 ∨ a = 9) ∧ 
      (b = 0 ∨ b = 1 ∨ b = 6 ∨ b = 8 ∨ b = 9) ∧  
      n - two_digit_upside_down n = 75) :=
by {
  sorry
}

end tanya_number_75_less_l214_214663


namespace equation_holds_l214_214838

variable (a b : ℝ)

theorem equation_holds : a^2 - b^2 - (-2 * b^2) = a^2 + b^2 :=
by sorry

end equation_holds_l214_214838


namespace projectile_height_l214_214268

theorem projectile_height (t : ℝ) (h : (-16 * t^2 + 80 * t = 100)) : t = 2.5 :=
sorry

end projectile_height_l214_214268


namespace inequality_abc_l214_214240

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b))) ≥ 3 / 2 := 
sorry

end inequality_abc_l214_214240


namespace sum_of_number_and_reverse_l214_214667

theorem sum_of_number_and_reverse (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) : (10 * a + b) + (10 * b + a) = 99 := by
  sorry

end sum_of_number_and_reverse_l214_214667


namespace no_solution_xy_l214_214132

theorem no_solution_xy (x y : ℕ) : ¬ (x * (x + 1) = 4 * y * (y + 1)) :=
sorry

end no_solution_xy_l214_214132


namespace part1_part2_l214_214000

-- Define the solution set M for the inequality
def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

-- Define the problem conditions
variables {a b : ℝ} (ha : a ∈ M) (hb : b ∈ M)

-- First part: Prove that |(1/3)a + (1/6)b| < 1/4
theorem part1 : |(1/3 : ℝ) * a + (1/6 : ℝ) * b| < 1/4 :=
sorry

-- Second part: Prove that |1 - 4 * a * b| > 2 * |a - b|
theorem part2 : |1 - 4 * a * b| > 2 * |a - b| :=
sorry

end part1_part2_l214_214000


namespace remainder_of_power_l214_214415

theorem remainder_of_power :
  (4^215) % 9 = 7 := by
sorry

end remainder_of_power_l214_214415


namespace intersection_M_N_l214_214516

def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | (1/3) ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x | (1/3) ≤ x ∧ x < 4} := by
  sorry

end intersection_M_N_l214_214516


namespace remainder_of_product_of_odd_primes_mod_32_l214_214993

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l214_214993


namespace white_marbles_in_C_equals_15_l214_214032

variables (A_red A_yellow B_green B_yellow C_yellow : ℕ) (w : ℕ)

-- Conditions from the problem
def conditions : Prop :=
  A_red = 4 ∧ A_yellow = 2 ∧
  B_green = 6 ∧ B_yellow = 1 ∧
  C_yellow = 9 ∧
  (A_red - A_yellow = 2) ∧
  (B_green - B_yellow = 5) ∧
  (w - C_yellow = 6)

-- Proving w = 15 given the conditions
theorem white_marbles_in_C_equals_15 (h : conditions A_red A_yellow B_green B_yellow C_yellow w) : w = 15 :=
  sorry

end white_marbles_in_C_equals_15_l214_214032


namespace race_positions_l214_214639

theorem race_positions :
  ∀ (M J T R H D : ℕ),
    (M = J + 3) →
    (J = T + 1) →
    (T = R + 3) →
    (H = R + 5) →
    (D = H + 4) →
    (M = 9) →
    H = 7 :=
by sorry

end race_positions_l214_214639


namespace bicycle_distance_l214_214700

theorem bicycle_distance (P_b P_f : ℝ) (h1 : P_b = 9) (h2 : P_f = 7) (h3 : ∀ D : ℝ, D / P_f = D / P_b + 10) :
  315 = 315 :=
by
  sorry

end bicycle_distance_l214_214700


namespace probability_of_two_primary_schools_from_selected_l214_214702

-- Definitions
def num_primary_schools : ℕ := 21
def num_middle_schools : ℕ := 14
def num_universities : ℕ := 7

def num_selected_primary_schools : ℕ := 3
def num_selected_middle_schools : ℕ := 2
def num_selected_universities : ℕ := 1

def selected_schools : Finset ℕ := Finset.range 6

-- Total possible outcomes of selecting 2 schools from 6 selected schools
def total_possible_outcomes : Finset (Finset ℕ) := Finset.powersetLen 2 selected_schools

-- Outcomes where both selected schools are primary
def primary_school_outcomes : Finset (Finset ℕ) := {Finset.mk [0, 1], Finset.mk [0, 2], Finset.mk [1, 2]}

-- Total combination count
def total_outcome_count : ℕ := Finset.card total_possible_outcomes

-- Favorable outcome count
def favorable_outcome_count : ℕ := Finset.card primary_school_outcomes

-- Probability calculation
def selection_probability : ℚ := favorable_outcome_count / total_outcome_count

theorem probability_of_two_primary_schools_from_selected : selection_probability = 1 / 5 := by
  -- Sorry is used to skip the proof.
  sorry

end probability_of_two_primary_schools_from_selected_l214_214702


namespace evaluate_expression_l214_214729

def g (x : ℝ) := 3 * x^2 - 5 * x + 7

theorem evaluate_expression : 3 * g 2 + 2 * g (-4) = 177 := by
  sorry

end evaluate_expression_l214_214729


namespace prob_all_heads_or_tails_five_coins_l214_214465

theorem prob_all_heads_or_tails_five_coins :
  (number_of_favorable_outcomes : ℕ) (total_number_of_outcomes : ℕ) (probability : ℚ) 
  (h_favorable : number_of_favorable_outcomes = 2)
  (h_total : total_number_of_outcomes = 32)
  (h_probability : probability = number_of_favorable_outcomes / total_number_of_outcomes) :
  probability = 1 / 16 :=
by
  sorry

end prob_all_heads_or_tails_five_coins_l214_214465


namespace projectile_reaches_100_feet_l214_214271

noncomputable def projectile_height (t : ℝ) : ℝ :=
  -16 * t^2 + 80 * t

theorem projectile_reaches_100_feet :
  ∃ t : ℝ, t = 2.5 ∧ projectile_height t = 100 :=
by
  use 2.5
  sorry

end projectile_reaches_100_feet_l214_214271


namespace sqrt_x_minus_5_meaningful_iff_x_ge_5_l214_214636

theorem sqrt_x_minus_5_meaningful_iff_x_ge_5 (x : ℝ) : (∃ y : ℝ, y^2 = x - 5) ↔ (x ≥ 5) :=
sorry

end sqrt_x_minus_5_meaningful_iff_x_ge_5_l214_214636


namespace remainder_of_M_when_divided_by_32_l214_214937

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l214_214937


namespace prime_product_mod_32_l214_214980

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l214_214980


namespace number_smaller_than_neg3_exists_l214_214438

def numbers := [0, -1, -5, -1/2]

theorem number_smaller_than_neg3_exists : ∃ x ∈ numbers, x < -3 :=
by
  let x := -5
  have h : x ∈ numbers := by simp [numbers]
  have h_lt : x < -3 := by norm_num
  exact ⟨x, h, h_lt⟩ -- show that -5 meets the criteria

end number_smaller_than_neg3_exists_l214_214438


namespace gcd_180_450_l214_214598

theorem gcd_180_450 : Nat.gcd 180 450 = 90 :=
by
  sorry

end gcd_180_450_l214_214598


namespace magnitude_of_resultant_vector_is_sqrt_5_l214_214095

-- We denote the vectors a and b
def a : ℝ × ℝ := (1, -2)
def b (y : ℝ) : ℝ × ℝ := (-2, y)

-- We encode the condition that vectors are parallel
def parallel_vectors (y : ℝ) : Prop := 1 * y = (-2) * (-2)

-- We calculate the resultant vector and its magnitude
def resultant_vector (y : ℝ) : ℝ × ℝ :=
  ((3 * 1 + 2 * -2), (3 * -2 + 2 * y))

def magnitude_square (v : ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2

-- The target statement
theorem magnitude_of_resultant_vector_is_sqrt_5 (y : ℝ) (hy : parallel_vectors y) :
  magnitude_square (resultant_vector y) = 5 := by
  sorry

end magnitude_of_resultant_vector_is_sqrt_5_l214_214095


namespace largest_n_with_triangle_property_l214_214066

/-- Triangle property: For any subset {a, b, c} with a ≤ b ≤ c, a + b > c -/
def triangle_property (s : Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s → a ≤ b → b ≤ c → a + b > c

/-- Definition of the set {3, 4, ..., n} -/
def consecutive_set (n : ℕ) : Finset ℕ :=
  Finset.range (n + 1) \ Finset.range 3

/-- The problem statement: The largest possible value of n where all eleven-element
 subsets of {3, 4, ..., n} have the triangle property -/
theorem largest_n_with_triangle_property : ∃ n, (∀ s ⊆ consecutive_set n, s.card = 11 → triangle_property s) ∧ n = 321 := sorry

end largest_n_with_triangle_property_l214_214066


namespace xy_extrema_l214_214022

noncomputable def xy_product (a : ℝ) : ℝ := a^2 - 1

theorem xy_extrema (x y a : ℝ) 
  (h1 : x + y = a) 
  (h2 : x^2 + y^2 = -a^2 + 2) : 
  -1 ≤ xy_product a ∧ xy_product a ≤ 1/3 :=
by
  sorry

end xy_extrema_l214_214022


namespace remainder_of_M_l214_214987

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l214_214987


namespace minimum_cost_is_8600_l214_214009

-- Defining the conditions
def shanghai_units : ℕ := 12
def nanjing_units : ℕ := 6
def suzhou_needs : ℕ := 10
def changsha_needs : ℕ := 8
def cost_shanghai_suzhou : ℕ := 400
def cost_shanghai_changsha : ℕ := 800
def cost_nanjing_suzhou : ℕ := 300
def cost_nanjing_changsha : ℕ := 500

-- Defining the function for total shipping cost
def total_shipping_cost (x : ℕ) : ℕ :=
  cost_shanghai_suzhou * x +
  cost_shanghai_changsha * (shanghai_units - x) +
  cost_nanjing_suzhou * (suzhou_needs - x) +
  cost_nanjing_changsha * (x - (shanghai_units - suzhou_needs))

-- Define the minimum shipping cost function
def minimum_shipping_cost : ℕ :=
  total_shipping_cost 10

-- State the theorem to prove
theorem minimum_cost_is_8600 : minimum_shipping_cost = 8600 :=
sorry

end minimum_cost_is_8600_l214_214009


namespace five_digit_number_with_integer_cube_root_l214_214632

theorem five_digit_number_with_integer_cube_root (n : ℕ) 
  (h1 : n ≥ 10000 ∧ n < 100000) 
  (h2 : n % 10 = 3) 
  (h3 : ∃ k : ℕ, k^3 = n) : 
  n = 19683 ∨ n = 50653 :=
sorry

end five_digit_number_with_integer_cube_root_l214_214632


namespace average_bowling_score_l214_214909

theorem average_bowling_score 
    (gretchen_score : ℕ) (mitzi_score : ℕ) (beth_score : ℕ)
    (gretchen_eq : gretchen_score = 120)
    (mitzi_eq : mitzi_score = 113)
    (beth_eq : beth_score = 85) :
    (gretchen_score + mitzi_score + beth_score) / 3 = 106 := 
by
  sorry

end average_bowling_score_l214_214909


namespace solve_system_of_equations_l214_214260

-- Define the given system of equations and conditions
theorem solve_system_of_equations (a b c x y z : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : yz / (y + z) = a) 
  (h2 : xz / (x + z) = b) 
  (h3 : xy / (x + y) = c) :
  x = 2 * a * b * c / (a * c + a * b - b * c) ∧ 
  y = 2 * a * b * c / (a * b + b * c - a * c) ∧ 
  z = 2 * a * b * c / (a * c + b * c - a * b) := sorry

end solve_system_of_equations_l214_214260


namespace geometric_sum_S5_l214_214116

variable (a_n : ℕ → ℝ)
variable (S : ℕ → ℝ)

def geometric_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a_n (n+1) = a_n n * q

theorem geometric_sum_S5 (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (h_geom : geometric_sequence a_n)
  (h_cond1 : a_n 2 * a_n 3 = 8 * a_n 1)
  (h_cond2 : (a_n 4 + 2 * a_n 5) / 2 = 20) :
  S 5 = 31 :=
sorry

end geometric_sum_S5_l214_214116


namespace sphere_radius_of_melted_cone_l214_214031

noncomputable def coneVolume (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * π * r^3

theorem sphere_radius_of_melted_cone :
  ∀ (r_cone s_cone l_cone : ℝ) (r_sphere : ℝ),
  r_cone = 3 →
  s_cone = 5 →
  l_cone = real.sqrt (s_cone^2 - r_cone^2) →
  coneVolume r_cone l_cone = 12 * π →
  sphereVolume r_sphere = 12 * π →
  r_sphere = real.cbrt 9 :=
by
  intros r_cone s_cone l_cone r_sphere h_rcone h_scone h_lcone h_coneVolume h_sphereVolume   
  sorry

end sphere_radius_of_melted_cone_l214_214031


namespace find_width_of_metallic_sheet_l214_214176

noncomputable def width_of_metallic_sheet (w : ℝ) : Prop :=
  let length := 48
  let square_side := 8
  let new_length := length - 2 * square_side
  let new_width := w - 2 * square_side
  let height := square_side
  let volume := new_length * new_width * height
  volume = 5120

theorem find_width_of_metallic_sheet (w : ℝ) :
  width_of_metallic_sheet w -> w = 36 := 
sorry

end find_width_of_metallic_sheet_l214_214176


namespace exists_polynomial_perfect_powers_l214_214885

theorem exists_polynomial_perfect_powers (m n : ℕ) (hm : 2 ≤ m) (hn : 1 ≤ n) :
  ∃ P : Polynomial ℤ, P.degree = n ∧ ∀ x ∈ Finset.range (n + 1), ∃ k : ℕ, P.eval x = m ^ k :=
by { sorry }

end exists_polynomial_perfect_powers_l214_214885


namespace width_of_road_correct_l214_214056

-- Define the given conditions
def sum_of_circumferences (r R : ℝ) : Prop := 2 * Real.pi * r + 2 * Real.pi * R = 88
def radius_relation (r R : ℝ) : Prop := r = (1/3) * R
def width_of_road (R r : ℝ) := R - r

-- State the main theorem
theorem width_of_road_correct (R r : ℝ) (h1 : sum_of_circumferences r R) (h2 : radius_relation r R) :
    width_of_road R r = 22 / Real.pi := by
  sorry

end width_of_road_correct_l214_214056


namespace remainder_when_divided_by_32_l214_214969

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l214_214969


namespace power_function_solution_l214_214221

def power_function_does_not_pass_through_origin (m : ℝ) : Prop :=
  (m^2 - m - 2) ≤ 0

def condition (m : ℝ) : Prop :=
  m^2 - 3 * m + 3 = 1

theorem power_function_solution (m : ℝ) :
  power_function_does_not_pass_through_origin m ∧ condition m → (m = 1 ∨ m = 2) :=
by sorry

end power_function_solution_l214_214221


namespace sin_cos_relation_l214_214769

theorem sin_cos_relation 
  (α β : Real) 
  (h : 2 * Real.sin α - Real.cos β = 2) 
  : Real.sin α + 2 * Real.cos β = 1 ∨ Real.sin α + 2 * Real.cos β = -1 := 
sorry

end sin_cos_relation_l214_214769


namespace weekly_milk_production_l214_214305

theorem weekly_milk_production (num_cows : ℕ) (milk_per_cow_per_day : ℕ) (days_in_week : ℕ) 
  (h_num_cows : num_cows = 52) (h_milk_per_cow_per_day : milk_per_cow_per_day = 1000) 
  (h_days_in_week : days_in_week = 7) :
  num_cows * milk_per_cow_per_day * days_in_week = 364000 :=
by
  rw [h_num_cows, h_milk_per_cow_per_day, h_days_in_week]
  norm_num
  sorry

end weekly_milk_production_l214_214305


namespace TimSpentThisMuch_l214_214849

/-- Tim's lunch cost -/
def lunchCost : ℝ := 50.50

/-- Tip percentage -/
def tipPercent : ℝ := 0.20

/-- Calculate the tip amount -/
def tipAmount := tipPercent * lunchCost

/-- Calculate the total amount spent -/
def totalAmountSpent := lunchCost + tipAmount

/-- Prove that the total amount spent is as expected -/
theorem TimSpentThisMuch : totalAmountSpent = 60.60 :=
  sorry

end TimSpentThisMuch_l214_214849


namespace value_of_a7_l214_214641

-- Define an arithmetic sequence
structure ArithmeticSeq (a : Nat → ℤ) :=
  (d : ℤ)
  (a_eq : ∀ n, a (n+1) = a n + d)

-- Lean statement of the equivalent proof problem
theorem value_of_a7 (a : ℕ → ℤ) (H : ArithmeticSeq a) :
  (2 * a 4 - a 7 ^ 2 + 2 * a 10 = 0) → a 7 = 4 * H.d :=
by
  sorry

end value_of_a7_l214_214641


namespace triangle_acute_angle_sufficient_condition_triangle_acute_angle_not_necessary_condition_l214_214375

theorem triangle_acute_angle_sufficient_condition
  (a b c : ℝ) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  a ≤ (b + c) / 2 → b^2 + c^2 > a^2 :=
sorry

theorem triangle_acute_angle_not_necessary_condition
  (a b c : ℝ) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  b^2 + c^2 > a^2 → ¬ (a ≤ (b + c) / 2) :=
sorry

end triangle_acute_angle_sufficient_condition_triangle_acute_angle_not_necessary_condition_l214_214375


namespace find_a1_in_geometric_sequence_l214_214107

noncomputable def geometric_sequence_first_term (a : ℕ → ℝ) (r : ℝ) (h : ∀ n : ℕ, a (n + 1) = a n * r) : ℝ :=
  a 0

theorem find_a1_in_geometric_sequence (a : ℕ → ℝ) (h_geo : ∀ n : ℕ, a (n + 1) = a n * (1 / 2)) :
  a 2 = 16 → a 3 = 8 → geometric_sequence_first_term a (1 / 2) h_geo = 64 :=
by
  intros h2 h3
  -- Proof would go here
  sorry

end find_a1_in_geometric_sequence_l214_214107


namespace remainder_of_M_l214_214982

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l214_214982


namespace harry_basketball_points_l214_214778

theorem harry_basketball_points :
  ∃ (x y : ℕ), 
    (x < 15) ∧ 
    (y < 15) ∧ 
    (62 + x) % 11 = 0 ∧ 
    (62 + x + y) % 12 = 0 ∧ 
    (x * y = 24) :=
by
  sorry

end harry_basketball_points_l214_214778


namespace neg_p_l214_214094

theorem neg_p :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by
  sorry

end neg_p_l214_214094


namespace monotonic_intervals_max_min_values_l214_214487

noncomputable def f : ℝ → ℝ := λ x => (1 / 3) * x^3 + x^2 - 3 * x + 1

theorem monotonic_intervals :
  (∀ x, x < -3 → deriv f x > 0) ∧
  (∀ x, x > 1 → deriv f x > 0) ∧
  (∀ x, -3 < x ∧ x < 1 → deriv f x < 0) :=
by
  sorry

theorem max_min_values :
  f 2 = 5 / 3 ∧ f 1 = -2 / 3 :=
by
  sorry

end monotonic_intervals_max_min_values_l214_214487


namespace prime_product_mod_32_l214_214976

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l214_214976


namespace distance_from_unselected_vertex_l214_214061

-- Define the problem statement
theorem distance_from_unselected_vertex
  (base length : ℝ) (area : ℝ) (h : ℝ) 
  (h_area : area = (base * h) / 2) 
  (h_base : base = 8) 
  (h_area_given : area = 24) : 
  h = 6 :=
by
  -- The proof here is skipped
  sorry

end distance_from_unselected_vertex_l214_214061


namespace count_mixed_4_digit_numbers_l214_214007

-- Defining what constitutes a mixed number according to conditions stated
def is_mixed (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  n ≥ 1000 ∧ n < 10000 ∧
  (List.length digits = 4) ∧
  (List.nodup digits) ∧
  (digits.head! ≠ List.maximum digits) ∧
  (digits.head! ≠ List.minimum digits) ∧
  (digits.last! ≠ List.minimum digits)

-- Statement: Prove that the number of 4-digit mixed integers is 1680
theorem count_mixed_4_digit_numbers :
  let mixed_count := (Nat.choose 10 4) * 8 in
  mixed_count = 1680 :=
by
  sorry

end count_mixed_4_digit_numbers_l214_214007


namespace proper_subsets_count_l214_214142

open Finset

def S : Finset (ℤ × ℤ) := 
  {(0,0), (-1,0), (0,-1), (1,0), (0,1)}

theorem proper_subsets_count : 
  S.card = 5 → 2 ^ S.card - 1 = 31 :=
by intros h; rw h; norm_num

end proper_subsets_count_l214_214142


namespace remainder_when_divided_by_32_l214_214962

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l214_214962


namespace num_sequences_of_student_helpers_l214_214870

-- Define the conditions
def num_students : ℕ := 15
def num_meetings : ℕ := 3

-- Define the statement to prove
theorem num_sequences_of_student_helpers : 
  (num_students ^ num_meetings) = 3375 :=
by sorry

end num_sequences_of_student_helpers_l214_214870


namespace find_other_number_l214_214145

theorem find_other_number (x y : ℕ) (h1 : x + y = 72) (h2 : y = x + 12) (h3 : y = 42) : x = 30 := by
  sorry

end find_other_number_l214_214145


namespace mapping_image_l214_214751

theorem mapping_image (f : ℕ → ℕ) (h : ∀ x, f x = x + 1) : f 3 = 4 :=
by {
  sorry
}

end mapping_image_l214_214751


namespace crocodile_length_in_meters_l214_214398

-- Definitions based on conditions
def ken_to_cm : ℕ := 180
def shaku_to_cm : ℕ := 30
def ken_to_shaku : ℕ := 6
def cm_to_m : ℕ := 100

-- Lengths given in the problem expressed in ken
def head_to_tail_in_ken (L : ℚ) : Prop := 3 * L = 10
def tail_to_head_in_ken (L : ℚ) : Prop := L = (3 + (2 / ken_to_shaku : ℚ))

-- Final length conversion to meters
def length_in_m (L : ℚ) : ℚ := L * ken_to_cm / cm_to_m

-- The length of the crocodile in meters
theorem crocodile_length_in_meters (L : ℚ) : head_to_tail_in_ken L → tail_to_head_in_ken L → length_in_m L = 6 :=
by
  intros _ _
  sorry

end crocodile_length_in_meters_l214_214398


namespace prime_product_mod_32_l214_214973

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l214_214973


namespace nearest_integer_to_power_sum_l214_214295

theorem nearest_integer_to_power_sum :
  let x := (3 + Real.sqrt 5)
  Int.floor ((x ^ 4) + 1 / 2) = 752 :=
by
  sorry

end nearest_integer_to_power_sum_l214_214295


namespace cos_double_angle_l214_214481

theorem cos_double_angle (α : ℝ) (h : Real.cos (π - α) = -3/5) : Real.cos (2 * α) = -7/25 :=
  sorry

end cos_double_angle_l214_214481


namespace diophantine_solution_exists_l214_214005

theorem diophantine_solution_exists (D : ℤ) : 
  ∃ (x y z : ℕ), x^2 - D * y^2 = z^2 ∧ ∃ m n : ℕ, m^2 > D * n^2 :=
sorry

end diophantine_solution_exists_l214_214005


namespace alyosha_possible_l214_214436

theorem alyosha_possible (current_date : ℕ) (day_before_yesterday_age current_year_age next_year_age : ℕ) : 
  (next_year_age = 12 ∧ day_before_yesterday_age = 9 ∧ current_year_age = 12 - 1)
  → (current_date = 1 ∧ current_year_age = 11 → (∃ bday : ℕ, bday = 31)) := 
by
  sorry

end alyosha_possible_l214_214436


namespace max_matrix_det_l214_214461

noncomputable def matrix_det (θ : ℝ) : ℝ :=
  by
    let M := ![
      ![1, 1, 1],
      ![1, 1 + Real.sin θ ^ 2, 1],
      ![1 + Real.cos θ ^ 2, 1, 1]
    ]
    exact Matrix.det M

theorem max_matrix_det : ∃ θ : ℝ, matrix_det θ = 3/4 :=
  sorry

end max_matrix_det_l214_214461


namespace intersection_complement_eq_three_l214_214241

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem intersection_complement_eq_three : N ∩ (U \ M) = {3} := by
  sorry

end intersection_complement_eq_three_l214_214241


namespace range_of_a_l214_214100

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - 2| ≥ a) → a ≤ 3 :=
by
  intro h
  sorry

end range_of_a_l214_214100


namespace find_number_l214_214894

-- Define the number x that satisfies the given condition
theorem find_number (x : ℤ) (h : x + 12 - 27 = 24) : x = 39 :=
by {
  -- This is where the proof steps will go, but we'll use sorry to indicate it's incomplete
  sorry
}

end find_number_l214_214894


namespace locker_number_problem_l214_214282

theorem locker_number_problem 
  (cost_per_digit : ℝ)
  (total_cost : ℝ)
  (one_digit_cost : ℝ)
  (two_digit_cost : ℝ)
  (three_digit_cost : ℝ) :
  cost_per_digit = 0.03 →
  one_digit_cost = 0.27 →
  two_digit_cost = 5.40 →
  three_digit_cost = 81.00 →
  total_cost = 206.91 →
  10 * cost_per_digit = six_cents →
  9 * cost_per_digit = three_cents →
  1 * 9 * cost_per_digit = one_digit_cost →
  2 * 45 * cost_per_digit = two_digit_cost →
  3 * 300 * cost_per_digit = three_digit_cost →
  (999 * 3 + x * 4 = 6880) →
  ∀ total_locker : ℕ, total_locker = 2001 := sorry

end locker_number_problem_l214_214282


namespace integer_solutions_determinant_l214_214320

theorem integer_solutions_determinant (a b c d : ℤ)
    (h : ∀ (m n : ℤ), ∃ (x y : ℤ), a * x + b * y = m ∧ c * x + d * y = n) :
    a * d - b * c = 1 ∨ a * d - b * c = -1 :=
sorry

end integer_solutions_determinant_l214_214320


namespace base_salary_is_1600_l214_214112

theorem base_salary_is_1600 (B : ℝ) (C : ℝ) (sales : ℝ) (fixed_salary : ℝ) :
  C = 0.04 ∧ sales = 5000 ∧ fixed_salary = 1800 ∧ (B + C * sales = fixed_salary) → B = 1600 :=
by sorry

end base_salary_is_1600_l214_214112


namespace base7_to_base10_l214_214689

theorem base7_to_base10 : 
  let digit0 := 2
  let digit1 := 3
  let digit2 := 4
  let digit3 := 5
  let base := 7
  digit0 * base^0 + digit1 * base^1 + digit2 * base^2 + digit3 * base^3 = 1934 :=
by
  let digit0 := 2
  let digit1 := 3
  let digit2 := 4
  let digit3 := 5
  let base := 7
  sorry

end base7_to_base10_l214_214689


namespace vector_addition_proof_l214_214748

variables {Point : Type} [AddCommGroup Point]

variables (A B C D : Point)

theorem vector_addition_proof :
  (D - A) + (C - D) - (C - B) = B - A :=
by
  sorry

end vector_addition_proof_l214_214748


namespace line_through_fixed_point_fixed_points_with_constant_slope_l214_214764

-- Point structure definition
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define curves C1 and C2
def curve_C1 (p : Point) : Prop :=
  p.x^2 + (p.y - 1/4)^2 = 1 ∧ p.y ≥ 1/4

def curve_C2 (p : Point) : Prop :=
  p.x^2 = 8 * p.y - 1 ∧ abs p.x ≥ 1

-- Line passing through fixed point for given perpendicularity condition
theorem line_through_fixed_point (A B M : Point) (l : ℝ → ℝ → Prop) :
  curve_C2 A → curve_C2 B →
  (∃ k b, ∀ x y, l x y ↔ y = k * x + b) →
  (M = ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩) →
  ((M.x = A.x ∧ M.y = (A.y + B.y) / 2) → A.x * B.x = -16) →
  ∀ x y, l x y → y = (17 / 8) := sorry

-- Existence of two fixed points on y-axis with constant slope product
theorem fixed_points_with_constant_slope (P T1 T2 M : Point) (l : ℝ → ℝ → Prop) :
  curve_C1 P →
  (T1 = ⟨0, -1⟩) →
  (T2 = ⟨0, 1⟩) →
  l P.x P.y →
  (∃ k b, ∀ x y, l x y ↔ y = k * x + b) →
  (M.y^2 - (M.x^2 / 16) = 1) →
  (M.x ≠ 0) →
  ((M.y + 1) / M.x) * ((M.y - 1) / M.x) = (1 / 16) := sorry

end line_through_fixed_point_fixed_points_with_constant_slope_l214_214764


namespace proof_problem_l214_214668

-- Definitions of the conditions
def domain_R (f : ℝ → ℝ) : Prop := ∀ x : ℝ, true

def symmetric_graph_pt (f : ℝ → ℝ) (a : ℝ) (b : ℝ) : Prop :=
  ∀ x : ℝ, f (a - x) = 2 * b - f (a + x)

def symmetric (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = -f (x)

def symmetric_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (2*a - x) = f (x)

-- Definitions of the statements to prove
def statement_1 (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (y = f (x - 1) → y = f (1 - x) → x = 1)

def statement_2 (f : ℝ → ℝ) : Prop :=
  symmetric_line f (3 / 2)

def statement_3 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 3) = -f (x)

-- Main proof problem
theorem proof_problem (f : ℝ → ℝ) 
  (h_domain : domain_R f)
  (h_symmetric_pt : symmetric_graph_pt f (-3 / 4) 0)
  (h_symmetric : ∀ x : ℝ, f (x + 3 / 2) = -f (x))
  (h_property : ∀ x : ℝ, f (x + 2) = -f (-x + 4)) :
  statement_1 f ∧ statement_2 f ∧ statement_3 f :=
sorry

end proof_problem_l214_214668


namespace min_T_tiles_needed_l214_214582

variable {a b c d : Nat}
variable (total_blocks : Nat := a + b + c + d)
variable (board_size : Nat := 8 * 10)
variable (block_size : Nat := 4)
variable (tile_types := ["T_horizontal", "T_vertical", "S_horizontal", "S_vertical"])
variable (conditions : Prop := total_blocks = 20 ∧ a + c ≥ 5)

theorem min_T_tiles_needed
    (h : conditions)
    (covering : total_blocks * block_size = board_size)
    (T_tiles : a ≥ 6) :
    a = 6 := sorry

end min_T_tiles_needed_l214_214582


namespace second_chapter_pages_is_80_l214_214429

def first_chapter_pages : ℕ := 37
def second_chapter_pages : ℕ := first_chapter_pages + 43

theorem second_chapter_pages_is_80 : second_chapter_pages = 80 :=
by
  sorry

end second_chapter_pages_is_80_l214_214429


namespace prob_of_drawing_one_red_ball_distribution_of_X_l214_214856

-- Definitions for conditions
def red_balls : ℕ := 2
def white_balls : ℕ := 3
def total_balls : ℕ := red_balls + white_balls
def balls_drawn : ℕ := 3

-- Combinations 
noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Probabilities
noncomputable def prob_ex_one_red_ball : ℚ :=
  (combination red_balls 1 * combination white_balls 2) / combination total_balls balls_drawn

noncomputable def prob_X_0 : ℚ := (combination white_balls 3) / combination total_balls balls_drawn
noncomputable def prob_X_1 : ℚ := prob_ex_one_red_ball
noncomputable def prob_X_2 : ℚ := (combination red_balls 2 * combination white_balls 1) / combination total_balls balls_drawn

-- Theorem statements
theorem prob_of_drawing_one_red_ball : prob_ex_one_red_ball = 3/5 := by
  sorry

theorem distribution_of_X : prob_X_0 = 1/10 ∧ prob_X_1 = 3/5 ∧ prob_X_2 = 3/10 := by
  sorry

end prob_of_drawing_one_red_ball_distribution_of_X_l214_214856


namespace min_value_of_m_n_squared_l214_214081

theorem min_value_of_m_n_squared 
  (a b c : ℝ)
  (triangle_cond : a^2 + b^2 = c^2)
  (m n : ℝ)
  (line_cond : a * m + b * n + 3 * c = 0) 
  : m^2 + n^2 = 9 := 
by
  sorry

end min_value_of_m_n_squared_l214_214081


namespace P_at_6_l214_214379

noncomputable def P (x : ℕ) : ℚ := (720 * x) / (x^2 - 1)

theorem P_at_6 : P 6 = 48 :=
by
  -- Definitions and conditions derived from the problem.
  -- Establishing given condition and deriving P(6) value.
  sorry

end P_at_6_l214_214379


namespace math_problem_l214_214618

theorem math_problem 
  (x y : ℝ) 
  (h1 : 4 * x + y = 12) 
  (h2 : x + 4 * y = 18) : 
  20 * x^2 + 24 * x * y + 20 * y^2 = 468 := 
by
  sorry

end math_problem_l214_214618


namespace remainder_when_M_divided_by_32_l214_214948

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l214_214948


namespace find_x0_l214_214750

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem find_x0 (x0 : ℝ) (h : deriv f x0 = 0) : x0 = Real.exp 1 :=
by 
  sorry

end find_x0_l214_214750


namespace rectangle_area_l214_214020

theorem rectangle_area (l w : ℕ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 120) : l * w = 800 :=
by
  -- proof to be filled in
  sorry

end rectangle_area_l214_214020


namespace inverse_proportional_example_l214_214348

variable (x y : ℝ)

def inverse_proportional (x y : ℝ) := y = 8 / (x - 1)

theorem inverse_proportional_example
  (h1 : y = 4)
  (h2 : x = 3) :
  inverse_proportional x y :=
by
  sorry

end inverse_proportional_example_l214_214348


namespace evaluate_expression_l214_214727

theorem evaluate_expression :
  -(12 * 2) - (3 * 2) + ((-18 / 3) * -4) = -6 := 
by
  sorry

end evaluate_expression_l214_214727


namespace correct_operation_l214_214419

theorem correct_operation (a b : ℤ) : -3 * (a - b) = -3 * a + 3 * b := 
sorry

end correct_operation_l214_214419


namespace zeros_indeterminate_in_interval_l214_214898

noncomputable def f : ℝ → ℝ := sorry

variables (a b : ℝ) (ha : a < b) (hf : f a * f b < 0)

-- The theorem statement
theorem zeros_indeterminate_in_interval :
  (∀ (f : ℝ → ℝ), f a * f b < 0 → (∃ (x : ℝ), a < x ∧ x < b ∧ f x = 0) ∨ (∀ (x : ℝ), a < x ∧ x < b → f x ≠ 0) ∨ (∃ (x1 x2 : ℝ), a < x1 ∧ x1 < x2 ∧ x2 < b ∧ f x1 = 0 ∧ f x2 = 0)) :=
by sorry

end zeros_indeterminate_in_interval_l214_214898


namespace number_of_subsets_of_union_l214_214617

open Finset

variable (A B : Finset ℕ)
variable hypA : A = {1, 2}
variable hypB : B = {0, 1}

theorem number_of_subsets_of_union : (A ∪ B).card = 3 → (A ∪ B).powerset.card = 8 :=
by
  intros h
  rw [←powerset_card]
  rw [h]
  norm_num
  sorry

end number_of_subsets_of_union_l214_214617


namespace probability_is_correct_l214_214170

-- Define the ratios for the colors: red, yellow, blue, black
def red_ratio := 6
def yellow_ratio := 2
def blue_ratio := 1
def black_ratio := 4

-- Define the total ratio
def total_ratio := red_ratio + yellow_ratio + blue_ratio + black_ratio

-- Define the ratio of red or blue regions
def red_or_blue_ratio := red_ratio + blue_ratio

-- Define the probability of landing on a red or blue region
def probability_red_or_blue := red_or_blue_ratio / total_ratio

-- State the theorem to prove
theorem probability_is_correct : probability_red_or_blue = 7 / 13 := 
by 
  -- Proof will go here
  sorry

end probability_is_correct_l214_214170


namespace intersection_in_quadrants_I_and_II_l214_214551

open Set

def in_quadrants_I_and_II (x y : ℝ) : Prop :=
  (0 ≤ x ∧ 0 ≤ y) ∨ (x ≤ 0 ∧ 0 ≤ y)

theorem intersection_in_quadrants_I_and_II :
  ∀ (x y : ℝ),
    y > 3 * x → y > -2 * x + 3 → in_quadrants_I_and_II x y :=
by
  intros x y h1 h2
  sorry

end intersection_in_quadrants_I_and_II_l214_214551


namespace smallest_b_in_ap_l214_214237

-- Definition of an arithmetic progression
def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b

-- Problem statement in Lean
theorem smallest_b_in_ap (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h_ap : is_arithmetic_progression a b c) 
  (h_prod : a * b * c = 216) : 
  b ≥ 6 :=
by
  sorry

end smallest_b_in_ap_l214_214237


namespace weekly_earnings_l214_214725

def phone_repair_cost : ℕ := 11
def laptop_repair_cost : ℕ := 15
def computer_repair_cost : ℕ := 18

def phone_repairs : ℕ := 5
def laptop_repairs : ℕ := 2
def computer_repairs : ℕ := 2

def total_earnings : ℕ := 
  phone_repairs * phone_repair_cost + 
  laptop_repairs * laptop_repair_cost + 
  computer_repairs * computer_repair_cost

theorem weekly_earnings : total_earnings = 121 := by
  sorry

end weekly_earnings_l214_214725


namespace problem1_problem2_l214_214209

-- Define the conditions as noncomputable definitions
noncomputable def A : Real := sorry
noncomputable def tan_A : Real := 2
noncomputable def sin_A_plus_cos_A : Real := 1 / 5

-- Define the trigonometric identities
noncomputable def sin (x : Real) : Real := sorry
noncomputable def cos (x : Real) : Real := sorry
noncomputable def tan (x : Real) : Real := sin x / cos x

-- Ensure the conditions
axiom tan_A_condition : tan A = tan_A
axiom sin_A_plus_cos_A_condition : sin A + cos A = sin_A_plus_cos_A

-- Proof problem 1:
theorem problem1 : 
  (sin (π - A) + cos (-A)) / (sin A - sin (π / 2 + A)) = 3 := by
  sorry

-- Proof problem 2:
theorem problem2 : 
  sin A - cos A = 7 / 5 := by
  sorry

end problem1_problem2_l214_214209


namespace find_g_three_l214_214546

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_three (h : ∀ x : ℝ, g (3^x) + (x + 1) * g (3^(-x)) = 3) : g 3 = -3 :=
sorry

end find_g_three_l214_214546


namespace intersection_M_N_l214_214356

variable (x : ℝ)

def M := {x : ℝ | -2 < x ∧ x < 2}
def N := {x : ℝ | x > 1}

theorem intersection_M_N :
  {x | x ∈ M ∧ x ∈ N} = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end intersection_M_N_l214_214356


namespace distance_between_cars_l214_214503

theorem distance_between_cars (t : ℝ) (v_kmh : ℝ) (v_ms : ℝ) :
  t = 1 ∧ v_kmh = 180 ∧ v_ms = v_kmh * 1000 / 3600 → 
  v_ms * t = 50 := 
by 
  sorry

end distance_between_cars_l214_214503


namespace bank_policies_for_retirees_favorable_l214_214400

-- Define the problem statement
theorem bank_policies_for_retirees_favorable :
  (∃ banks : Type,
    (∃ retirees: Type,
      (∃ policies : banks → Prop,
        (∀ bank, (policies bank → ∃ retiree: retirees,
          (let conscientious := True
          in let stable_income := True 
          in let attract_funds := True 
          in let save_preference := True 
          in let regular_income := True 
          in let long_term_deposit := True
          in conscientious ∧ stable_income ∧ attract_funds ∧ save_preference ∧ regular_income ∧ long_term_deposit))))) :=
begin
  sorry
end

end bank_policies_for_retirees_favorable_l214_214400


namespace compare_exponents_l214_214343

theorem compare_exponents :
  let a := (3 / 2) ^ 0.1
  let b := (3 / 2) ^ 0.2
  let c := (3 / 2) ^ 0.08
  c < a ∧ a < b := by
  sorry

end compare_exponents_l214_214343


namespace point_relationship_l214_214353

variables {m x1 x2 y1 y2 : ℝ}

def quadratic_function (x : ℝ) (m : ℝ) : ℝ :=
  (x + m - 3)*(x - m) + 3

theorem point_relationship 
  (hx1_lt_x2 : x1 < x2)
  (hA : y1 = quadratic_function x1 m)
  (hB : y2 = quadratic_function x2 m)
  (h_sum_lt : x1 + x2 < 3) :
  y1 > y2 :=
sorry

end point_relationship_l214_214353


namespace range_f_pos_l214_214616

noncomputable def f : ℝ → ℝ := sorry
axiom even_f : ∀ x : ℝ, f x = f (-x)
axiom increasing_f : ∀ x y : ℝ, x < y → x ≤ 0 → y ≤ 0 → f x ≤ f y
axiom f_at_neg_one : f (-1) = 0

theorem range_f_pos : {x : ℝ | f x > 0} = Set.Ioo (-1) 1 := 
by
  sorry

end range_f_pos_l214_214616


namespace gcd_of_180_and_450_l214_214600

theorem gcd_of_180_and_450 : Int.gcd 180 450 = 90 := 
  sorry

end gcd_of_180_and_450_l214_214600


namespace smaller_than_neg3_l214_214441

theorem smaller_than_neg3 :
  (∃ x ∈ ({0, -1, -5, -1/2} : Set ℚ), x < -3) ∧ ∀ x ∈ ({0, -1, -5, -1/2} : Set ℚ), x < -3 → x = -5 :=
by
  sorry

end smaller_than_neg3_l214_214441


namespace geometric_sequence_product_l214_214472

variable {a : ℕ → ℝ}
variable {r : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_product 
  (h : is_geometric_sequence a r)
  (h_cond : a 4 * a 6 = 10) :
  a 2 * a 8 = 10 := 
sorry

end geometric_sequence_product_l214_214472


namespace total_flowers_in_3_hours_l214_214286

-- Constants representing the number of each type of flower
def roses : ℕ := 12
def sunflowers : ℕ := 15
def tulips : ℕ := 9
def daisies : ℕ := 18
def orchids : ℕ := 6
def total_flowers : ℕ := 60

-- Number of flowers each bee can pollinate in an hour
def bee_A_rate (roses sunflowers tulips: ℕ) : ℕ := 2 + 3 + 1
def bee_B_rate (daisies orchids: ℕ) : ℕ := 4 + 1
def bee_C_rate (roses sunflowers tulips daisies orchids: ℕ) : ℕ := 1 + 2 + 2 + 3 + 1

-- Total number of flowers pollinated by all bees in an hour
def total_bees_rate (bee_A_rate bee_B_rate bee_C_rate: ℕ) : ℕ := bee_A_rate + bee_B_rate + bee_C_rate

-- Proving the total flowers pollinated in 3 hours
theorem total_flowers_in_3_hours : total_bees_rate 6 5 9 * 3 = total_flowers := 
by {
  sorry
}

end total_flowers_in_3_hours_l214_214286


namespace sin_value_l214_214471

theorem sin_value (alpha : ℝ) (h1 : -π / 6 < alpha ∧ alpha < π / 6)
  (h2 : Real.cos (alpha + π / 6) = 4 / 5) :
  Real.sin (2 * alpha + π / 12) = 17 * Real.sqrt 2 / 50 :=
by
    sorry

end sin_value_l214_214471


namespace decipher_rebus_l214_214647

theorem decipher_rebus (a b c d : ℕ) :
  (a = 10 ∧ b = 14 ∧ c = 12 ∧ d = 13) ↔
  (∀ (x y z w: ℕ), 
    (x = 10 → 5 + 5 * 7 = 49) ∧
    (y = 14 → 2 - 4 * 3 = 9) ∧
    (z = 12 → 12 - 1 - 1 * 2 = 20) ∧
    (w = 13 → 13 - 1 + 10 - 5 = 17) ∧
    (49 + 9 + 20 + 17 = 95)) :=
by sorry

end decipher_rebus_l214_214647


namespace average_bowling_score_l214_214906

-- Definitions of the scores
def g : ℕ := 120
def m : ℕ := 113
def b : ℕ := 85

-- Theorem statement: The average score is 106
theorem average_bowling_score : (g + m + b) / 3 = 106 := by
  sorry

end average_bowling_score_l214_214906


namespace find_second_number_l214_214852

theorem find_second_number (a b c : ℕ) 
  (h1 : a + b + c = 550) 
  (h2 : a = 2 * b) 
  (h3 : c = a / 3) :
  b = 150 :=
by
  sorry

end find_second_number_l214_214852


namespace min_a2_plus_b2_l214_214772

theorem min_a2_plus_b2 (a b : ℝ) (h : a * b = 1) : a^2 + b^2 ≥ 2 :=
sorry

end min_a2_plus_b2_l214_214772


namespace evaluate_fraction_l214_214075

theorem evaluate_fraction :
  1 + (2 / (3 + (6 / (7 + (8 / 9))))) = 409 / 267 :=
by
  sorry

end evaluate_fraction_l214_214075


namespace simplify_expression_l214_214388

noncomputable def cube_root (x : ℝ) : ℝ := x ^ (1/3 : ℝ)

theorem simplify_expression :
  (cube_root 512) * (cube_root 343) = 56 := by
  -- conditions
  let h1 : 512 = 2^9 := by rfl
  let h2 : 343 = 7^3 := by rfl
  -- goal
  sorry

end simplify_expression_l214_214388


namespace total_length_circle_l214_214011

-- Definitions based on conditions
def num_strips : ℕ := 16
def length_each_strip : ℝ := 10.4
def overlap_each_strip : ℝ := 3.5

-- Theorem stating the total length of the circle-shaped colored tape
theorem total_length_circle : 
  (num_strips * length_each_strip) - (num_strips * overlap_each_strip) = 110.4 := 
by 
  sorry

end total_length_circle_l214_214011


namespace solve_for_x_l214_214259

theorem solve_for_x : ∀ (x : ℤ), (5 * x - 2) * 4 = (3 * (6 * x - 6)) → x = -5 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l214_214259


namespace smaller_integer_l214_214139

noncomputable def m : ℕ := 1
noncomputable def n : ℕ := 1998 * m

lemma two_digit_number (m: ℕ) : 10 ≤ m ∧ m < 100 := by sorry
lemma three_digit_number (n: ℕ) : 100 ≤ n ∧ n < 1000 := by sorry

theorem smaller_integer 
  (two_digit_m: 10 ≤ m ∧ m < 100)
  (three_digit_n: 100 ≤ n ∧ n < 1000)
  (avg_eq_decimal: (m + n) / 2 = m + n / 1000)
  : m = 1 := by 
  sorry

end smaller_integer_l214_214139


namespace find_integer_l214_214745

def satisfies_conditions (x : ℕ) (m n : ℕ) : Prop :=
  x + 100 = m ^ 2 ∧ x + 168 = n ^ 2 ∧ m > 0 ∧ n > 0

theorem find_integer (x m n : ℕ) (h : satisfies_conditions x m n) : x = 156 :=
sorry

end find_integer_l214_214745


namespace remainder_of_M_l214_214989

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l214_214989


namespace num_frisbees_more_than_deck_cards_l214_214585

variables (M F D x : ℕ)
variable (bought_fraction : ℝ)

theorem num_frisbees_more_than_deck_cards :
  M = 60 ∧ M = 2 * F ∧ F = D + x ∧
  M + bought_fraction * M + F + bought_fraction * F + D + bought_fraction * D = 140 ∧ bought_fraction = 2/5 →
  x = 20 :=
by
  sorry

end num_frisbees_more_than_deck_cards_l214_214585


namespace euler_distance_formula_l214_214120

theorem euler_distance_formula 
  (d R r : ℝ) 
  (h₁ : d = distance_between_centers_of_inscribed_and_circumscribed_circles_of_triangle)
  (h₂ : R = circumradius_of_triangle)
  (h₃ : r = inradius_of_triangle) : 
  d^2 = R^2 - 2 * R * r := 
sorry

end euler_distance_formula_l214_214120


namespace binomial_variance_l214_214762

variables {p : ℝ} {ξ : ℝ}
variables (h1 : E ξ = 9) (h2: ∀ ξ, ξ ∼ B 18 p)

theorem binomial_variance : D ξ = 9 / 2 :=
by
  sorry

end binomial_variance_l214_214762


namespace max_min_g_l214_214489

open Real

noncomputable def f (t : ℝ) : ℝ := (t^2 + 2 * t - 1) / (t^2 + 1)

noncomputable def g (x : ℝ) : ℝ := f(x) * f(1 - x)

theorem max_min_g :
  (∀ x, -1 ≤ x ∧ x ≤ 1 → g(x) ≤ 1 / 25) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 1 ∧ g(x) = 1 / 25) ∧
  (∀ x, -1 ≤ x ∧ x ≤ 1 → g(x) ≥ 4 - sqrt 34) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 1 ∧ g(x) = 4 - sqrt 34) :=
by
  sorry

end max_min_g_l214_214489


namespace triangle_ABI_ratio_l214_214783

theorem triangle_ABI_ratio:
  ∀ (AC BC : ℝ) (hAC : AC = 15) (hBC : BC = 20),
  let AB := Real.sqrt (AC^2 + BC^2) in
  let CD :=  (AC * BC) / AB in
  let r := CD / 2 in
  let x := Real.sqrt (r^2 + (Real.sqrt (r^2 + (AB/2)^2) - r)^2) in
  let P := 2 * x + AB in
  (P / AB = 177 / 100) ∧ (177 + 100 = 277) :=
by
  intros AC BC hAC hBC
  let AB := Real.sqrt (AC^2 + BC^2)
  let CD := (AC * BC) / AB
  let r := CD / 2
  let x := Real.sqrt (r^2 + (Real.sqrt (r^2 + (AB/2)^2) - r)^2)
  let P := 2 * x + AB
  have ratio : P / AB = 177 / 100 := sorry
  exact ⟨ratio, rfl⟩


end triangle_ABI_ratio_l214_214783


namespace mult_closest_l214_214160

theorem mult_closest :
  0.0004 * 9000000 = 3600 := sorry

end mult_closest_l214_214160


namespace complement_intersection_l214_214767

/-- Given the universal set U={1,2,3,4,5},
    A={2,3,4}, and B={1,2,3}, 
    Prove the complement of (A ∩ B) in U is {1,4,5}. -/
theorem complement_intersection 
    (U : Set ℕ) (A : Set ℕ) (B : Set ℕ) 
    (hU : U = {1, 2, 3, 4, 5})
    (hA : A = {2, 3, 4})
    (hB : B = {1, 2, 3}) :
    U \ (A ∩ B) = {1, 4, 5} :=
by
  -- proof goes here
  sorry

end complement_intersection_l214_214767


namespace min_eq_neg_one_implies_x_eq_two_l214_214528

theorem min_eq_neg_one_implies_x_eq_two (x : ℝ) (h : min (2*x - 5) (x + 1) = -1) : x = 2 :=
sorry

end min_eq_neg_one_implies_x_eq_two_l214_214528


namespace line_through_P_perpendicular_l214_214591

theorem line_through_P_perpendicular 
  (P : ℝ × ℝ) (a b c : ℝ) (hP : P = (-1, 3)) (hline : a = 1 ∧ b = -2 ∧ c = 3) :
  ∃ (a' b' c' : ℝ), (a' * P.1 + b' * P.2 + c' = 0) ∧ (a = b' ∧ b = -a') ∧ (a' = 2 ∧ b' = 1 ∧ c' = -1) := 
by
  use 2, 1, -1
  sorry

end line_through_P_perpendicular_l214_214591


namespace triangle_formation_l214_214923

theorem triangle_formation (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : c^2 = a^2 + b^2 + a * b) : 
  a + b > c ∧ a + c > b ∧ c + (a + b) > a :=
by
  sorry

end triangle_formation_l214_214923


namespace total_profit_or_loss_is_negative_175_l214_214577

theorem total_profit_or_loss_is_negative_175
    (price_A price_B selling_price : ℝ)
    (profit_A loss_B : ℝ)
    (h1 : selling_price = 2100)
    (h2 : profit_A = 0.2)
    (h3 : loss_B = 0.2)
    (hA : price_A * (1 + profit_A) = selling_price)
    (hB : price_B * (1 - loss_B) = selling_price) :
    (selling_price + selling_price) - (price_A + price_B) = -175 := 
by 
  -- The proof is omitted
  sorry

end total_profit_or_loss_is_negative_175_l214_214577


namespace daily_production_l214_214168

-- Define the conditions
def bottles_per_case : ℕ := 9
def num_cases : ℕ := 8000

-- State the theorem with the question and the calculated answer
theorem daily_production : bottles_per_case * num_cases = 72000 :=
by
  sorry

end daily_production_l214_214168


namespace difference_of_sums_1000_l214_214294

def sum_first_n_even (n : ℕ) : ℕ :=
  n * (n + 1)

def sum_first_n_odd_not_divisible_by_5 (n : ℕ) : ℕ :=
  (n * n) - 5 * ((n / 5) * ((n / 5) + 1))

theorem difference_of_sums_1000 :
  (sum_first_n_even 1000) - (sum_first_n_odd_not_divisible_by_5 1000) = 51000 :=
by
  sorry

end difference_of_sums_1000_l214_214294


namespace inequality_proof_l214_214378

variable (a b c : ℝ)

noncomputable def specific_condition (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (1 / a + 1 / b + 1 / c = 1)

theorem inequality_proof (h : specific_condition a b c) :
  (a^a * b * c + b^b * c * a + c^c * a * b) ≥ 27 * (b * c + c * a + a * b) := 
by {
  sorry
}

end inequality_proof_l214_214378


namespace must_true_l214_214213

axiom p : Prop
axiom q : Prop
axiom h1 : ¬ (p ∧ q)
axiom h2 : p ∨ q

theorem must_true : (¬ p) ∨ (¬ q) := by
  sorry

end must_true_l214_214213


namespace minimize_segment_sum_l214_214558

theorem minimize_segment_sum (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  ∃ x y : ℝ, x = Real.sqrt (a * b) ∧ y = Real.sqrt (a * b) ∧ x * y = a * b ∧ x + y = 2 * Real.sqrt (a * b) := 
by
  sorry

end minimize_segment_sum_l214_214558


namespace no_hamiltonian_cycle_l214_214285

-- Define the problem constants
def n : ℕ := 2016
def a : ℕ := 2
def b : ℕ := 3

-- Define the circulant graph and the conditions of the Hamiltonian cycle theorem
theorem no_hamiltonian_cycle (s t : ℕ) (h1 : s + t = Int.gcd n (a - b)) :
  ¬ (Int.gcd n (s * a + t * b) = 1) :=
by
  sorry  -- Proof not required as per instructions

end no_hamiltonian_cycle_l214_214285


namespace max_k_exists_l214_214631

noncomputable def max_possible_k (x y k : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ k > 0)
  (h_eq : 5 = k^3 * ((x^2 / y^2) + (y^2 / x^2)) + k^2 * ((x / y) + (y / x))) : ℝ :=
sorry

theorem max_k_exists (x y k : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ k > 0)
  (h_eq : 5 = k^3 * ((x^2 / y^2) + (y^2 / x^2)) + k^2 * ((x / y) + (y / x))) :
  ∃ k_max : ℝ, k_max = max_possible_k x y k h_pos h_eq :=
sorry

end max_k_exists_l214_214631


namespace age_sum_l214_214570

theorem age_sum (P Q : ℕ) (h1 : P - 12 = (1 / 2 : ℚ) * (Q - 12)) (h2 : (P : ℚ) / Q = (3 / 4 : ℚ)) : P + Q = 42 :=
sorry

end age_sum_l214_214570


namespace min_value_expression_l214_214475

theorem min_value_expression (m n : ℝ) (h : m > 2 * n) :
  m + (4 * n^2 - 2 * m * n + 9) / (m - 2 * n) ≥ 6 :=
by
  sorry

end min_value_expression_l214_214475


namespace intersection_of_sets_example_l214_214522

theorem intersection_of_sets_example :
  let M := { x : ℝ | 0 < x ∧ x < 4 }
  let N := { x : ℝ | 1 / 3 ≤ x ∧ x ≤ 5 }
  let expected := { x : ℝ | 1 / 3 ≤ x ∧ x < 4 }
  (M ∩ N) = expected :=
by
  sorry

end intersection_of_sets_example_l214_214522


namespace find_k_in_expression_l214_214554

theorem find_k_in_expression :
  (2^1004 + 5^1005)^2 - (2^1004 - 5^1005)^2 = 20 * 10^1004 :=
by
  sorry

end find_k_in_expression_l214_214554


namespace original_example_intended_l214_214922

theorem original_example_intended (x : ℝ) : (3 * x - 4 = x / 3 + 4) → x = 3 :=
by
  sorry

end original_example_intended_l214_214922


namespace remainder_of_M_when_divided_by_32_l214_214938

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l214_214938


namespace large_square_area_l214_214186

theorem large_square_area (l w : ℕ) (h1 : 2 * (l + w) = 28) : (l + w) * (l + w) = 196 :=
by {
  sorry
}

end large_square_area_l214_214186


namespace usual_time_is_24_l214_214841

variable (R T : ℝ)
variable (usual_rate fraction_of_rate early_min : ℝ)
variable (h1 : fraction_of_rate = 6 / 7)
variable (h2 : early_min = 4)
variable (h3 : (R / (fraction_of_rate * R)) = 7 / 6)
variable (h4 : ((T - early_min) / T) = fraction_of_rate)

theorem usual_time_is_24 {R T : ℝ} (fraction_of_rate := 6/7) (early_min := 4) :
  fraction_of_rate = 6 / 7 ∧ early_min = 4 → 
  (T - early_min) / T = fraction_of_rate → 
  T = 24 :=
by
  intros hfraction_hearly htime_eq_fraction
  sorry

end usual_time_is_24_l214_214841


namespace triangle_tangency_perimeter_l214_214413

def triangle_perimeter (a b c : ℝ) (s : ℝ) (t : ℝ) (u : ℝ) : ℝ :=
  s + t + u

theorem triangle_tangency_perimeter (a b c : ℝ) (D E F : ℝ) (s : ℝ) (t : ℝ) (u : ℝ)
  (h1 : a = 5) (h2 : b = 7) (h3 : c = 8) 
  (h4 : s + t + u = 3) : triangle_perimeter a b c s t u = 3 :=
by
  sorry

end triangle_tangency_perimeter_l214_214413


namespace unit_digit_G_1000_l214_214886

def G (n : ℕ) : ℕ := 3^(3^n) + 1

theorem unit_digit_G_1000 : (G 1000) % 10 = 2 :=
by
  sorry

end unit_digit_G_1000_l214_214886


namespace least_positive_integer_l214_214041

theorem least_positive_integer (a : ℕ) :
  (a % 2 = 0) ∧ (a % 5 = 1) ∧ (a % 4 = 2) → a = 6 :=
by
  sorry

end least_positive_integer_l214_214041


namespace find_number_of_white_balls_l214_214228

theorem find_number_of_white_balls (n : ℕ) (h : 6 / (6 + n) = 2 / 5) : n = 9 :=
sorry

end find_number_of_white_balls_l214_214228


namespace binary_to_decimal_is_1023_l214_214017

-- Define the binary number 1111111111 in terms of its decimal representation
def binary_to_decimal : ℕ :=
  (1 * 2^9 + 1 * 2^8 + 1 * 2^7 + 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0)

-- The theorem statement
theorem binary_to_decimal_is_1023 : binary_to_decimal = 1023 :=
by
  sorry

end binary_to_decimal_is_1023_l214_214017


namespace prime_product_mod_32_l214_214975

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l214_214975


namespace N_eq_M_union_P_l214_214490

def M : Set ℝ := {x : ℝ | ∃ (n : ℤ), x = n}
def N : Set ℝ := {x : ℝ | ∃ (n : ℤ), x = n / 2}
def P : Set ℝ := {x : ℝ | ∃ (n : ℤ), x = n + 1 / 2}

theorem N_eq_M_union_P : N = M ∪ P :=
  sorry

end N_eq_M_union_P_l214_214490


namespace greatest_divisor_remainders_l214_214196

theorem greatest_divisor_remainders (x : ℕ) (h1 : 1255 % x = 8) (h2 : 1490 % x = 11) : x = 29 :=
by
  -- The proof steps would go here, but for now, we use sorry.
  sorry

end greatest_divisor_remainders_l214_214196


namespace quadratic_value_at_sum_of_roots_is_five_l214_214757

noncomputable def quadratic_func (a b x : ℝ) : ℝ := a * x^2 + b * x + 5

theorem quadratic_value_at_sum_of_roots_is_five
  (a b x₁ x₂ : ℝ)
  (hA : quadratic_func a b x₁ = 2023)
  (hB : quadratic_func a b x₂ = 2023)
  (ha : a ≠ 0) :
  quadratic_func a b (x₁ + x₂) = 5 :=
sorry

end quadratic_value_at_sum_of_roots_is_five_l214_214757


namespace first_common_digit_three_digit_powers_l214_214688

theorem first_common_digit_three_digit_powers (m n: ℕ) (hm: 100 ≤ 2^m ∧ 2^m < 1000) (hn: 100 ≤ 3^n ∧ 3^n < 1000) :
  (∃ d, (2^m).div 100 = d ∧ (3^n).div 100 = d ∧ d = 2) :=
sorry

end first_common_digit_three_digit_powers_l214_214688


namespace total_bricks_required_l214_214567

def courtyard_length : ℕ := 24 * 100  -- convert meters to cm
def courtyard_width : ℕ := 14 * 100  -- convert meters to cm
def brick_length : ℕ := 25
def brick_width : ℕ := 15

-- Calculate the area of the courtyard in square centimeters
def courtyard_area : ℕ := courtyard_length * courtyard_width

-- Calculate the area of one brick in square centimeters
def brick_area : ℕ := brick_length * brick_width

theorem total_bricks_required :  courtyard_area / brick_area = 8960 := by
  -- This part will have the proof, for now, we use sorry to skip it
  sorry

end total_bricks_required_l214_214567


namespace geometric_sequence_a6_l214_214327

noncomputable def a_n (a_1 q : ℝ) (n : ℕ) : ℝ :=
  a_1 * q ^ (n - 1)

theorem geometric_sequence_a6 
  (a_1 q : ℝ) 
  (a2_eq : a_1 + a_1 * q = -1)
  (a3_eq : a_1 - a_1 * q ^ 2 = -3) : 
  a_n a_1 q 6 = -32 :=
sorry

end geometric_sequence_a6_l214_214327


namespace remainder_of_product_of_odd_primes_mod_32_l214_214998

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l214_214998


namespace inequality_transformation_range_of_a_l214_214350

-- Define the given function f(x) = |x + 2|
def f (x : ℝ) : ℝ := abs (x + 2)

-- State the inequality transformation problem
theorem inequality_transformation (x : ℝ) :  (2 * abs (x + 2) < 4 - abs (x - 1)) ↔ (-7 / 3 < x ∧ x < -1) :=
by sorry

-- State the implication problem involving m, n, and a
theorem range_of_a (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (hmn : m + n = 1) (a : ℝ) :
  (∀ x : ℝ, abs (x - a) - f x ≤ 1 / m + 1 / n) → (-6 ≤ a ∧ a ≤ 2) :=
by sorry

end inequality_transformation_range_of_a_l214_214350


namespace variance_of_score_is_0_16_l214_214704

-- Define the probabilities for the single free throw event
variable (p_hit : ℝ) (p_miss : ℝ)
variable (X : Type) [Probability X]
variable (X : Probability.Space.Discrete)

def P_X_1 : ℝ := 0.8
def P_X_0 : ℝ := 0.2

-- Define the random variable and its expected value
def score (x : X) : ℝ :=
  if x then 1 else 0

def E_X : ℝ := E (score p_hit p_miss : X → ℝ)

-- Define the expected value of X squared
def E_X2 : ℝ := E (λ x, (score p_hit p_miss : X → ℝ) x ^ 2)

-- Define the variance of X
def Var_X : ℝ := E_X2 - E_X^2

-- Lean 4 statement to prove the variance is 0.16
theorem variance_of_score_is_0_16 (h1 : P_X_1 = 0.8) (h2 : P_X_0 = 0.2) : Var_X = 0.16 :=
by {
  sorry
}

end variance_of_score_is_0_16_l214_214704


namespace shortest_track_length_l214_214384

open Nat

def Melanie_track_length := 8
def Martin_track_length := 20

theorem shortest_track_length :
  Nat.lcm Melanie_track_length Martin_track_length = 40 :=
by
  sorry

end shortest_track_length_l214_214384


namespace curve_symmetries_and_point_l214_214763

theorem curve_symmetries_and_point :
  ∀ (x y : ℝ), (x ^ 2 + x * y + y ^ 2 = 4) → 
  (∀ (x y : ℝ), x ^ 2 + x * y + y ^ 2 = 4 → (-x) ^ 2 + (-x) * (-y) + (-y) ^ 2 = 4) ∧ 
  (∀ (x y : ℝ), x ^ 2 + x * y + y ^ 2 = 4 → y ^ 2 + x * y + x ^ 2 = 4) ∧ 
  ( (2 : ℝ) ^ 2 + 2 * (-2 : ℝ) + (-2 : ℝ) ^ 2 = 4) :=
sorry

end curve_symmetries_and_point_l214_214763


namespace pants_cost_is_250_l214_214875

-- Define the cost of a T-shirt
def tshirt_cost := 100

-- Define the total amount spent
def total_amount := 1500

-- Define the number of T-shirts bought
def num_tshirts := 5

-- Define the number of pants bought
def num_pants := 4

-- Define the total cost of T-shirts
def total_tshirt_cost := tshirt_cost * num_tshirts

-- Define the total cost of pants
def total_pants_cost := total_amount - total_tshirt_cost

-- Define the cost per pair of pants
def pants_cost_per_pair := total_pants_cost / num_pants

-- Proving that the cost per pair of pants is $250
theorem pants_cost_is_250 : pants_cost_per_pair = 250 := by
  sorry

end pants_cost_is_250_l214_214875


namespace smallest_positive_multiple_of_45_divisible_by_3_l214_214298

theorem smallest_positive_multiple_of_45_divisible_by_3 
  (x : ℕ) (hx: x > 0) : ∃ y : ℕ, y = 45 ∧ 45 ∣ y ∧ 3 ∣ y ∧ ∀ z : ℕ, (45 ∣ z ∧ 3 ∣ z ∧ z > 0) → z ≥ y :=
by
  sorry

end smallest_positive_multiple_of_45_divisible_by_3_l214_214298


namespace gcd_lcm_sum_l214_214159

theorem gcd_lcm_sum :
  ∀ (a b c d : ℕ), gcd a b + lcm c d = 74 :=
by
  let a := 42
  let b := 70
  let c := 20
  let d := 15
  sorry

end gcd_lcm_sum_l214_214159


namespace problem_statement_l214_214121

noncomputable def g : ℝ → ℝ := sorry

theorem problem_statement 
  (h : ∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - x * y^2 - x + 2) :
  ∃ (m t : ℕ), (m = 1) ∧ (t = 3) ∧ (m * t = 3) :=
sorry

end problem_statement_l214_214121


namespace total_bill_l214_214015

/-
Ten friends dined at a restaurant and split the bill equally.
One friend, Chris, forgets his money.
Each of the remaining nine friends agreed to pay an extra $3 to cover Chris's share.
How much was the total bill?

Correct answer: 270
-/

theorem total_bill (t : ℕ) (h1 : ∀ x, t = 10 * x) (h2 : ∀ x, t = 9 * (x + 3)) : t = 270 := by
  sorry

end total_bill_l214_214015


namespace paper_clips_in_two_cases_l214_214424

-- Defining the problem statement in Lean 4
theorem paper_clips_in_two_cases (c b : ℕ) :
  2 * (c * b * 300) = 2 * c * b * 300 :=
by
  sorry

end paper_clips_in_two_cases_l214_214424


namespace meaningful_sqrt_x_minus_5_l214_214103

theorem meaningful_sqrt_x_minus_5 (x : ℝ) (h : sqrt (x - 5) ∈ ℝ) : x = 6 ∨ x ≥ 5 := by
  sorry

end meaningful_sqrt_x_minus_5_l214_214103


namespace gcd_180_450_l214_214595

theorem gcd_180_450 : Nat.gcd 180 450 = 90 :=
by
  sorry

end gcd_180_450_l214_214595


namespace find_side_lengths_l214_214215

variable (a b : ℝ)

-- Conditions
def diff_side_lengths := a - b = 2
def diff_areas := a^2 - b^2 = 40

-- Theorem to prove
theorem find_side_lengths (h1 : diff_side_lengths a b) (h2 : diff_areas a b) :
  a = 11 ∧ b = 9 := by
  -- Proof skipped
  sorry

end find_side_lengths_l214_214215


namespace find_z_l214_214098

variable {x y z : ℝ}

theorem find_z (h : (1/x + 1/y = 1/z)) : z = (x * y) / (x + y) :=
  sorry

end find_z_l214_214098


namespace projectile_reaches_100_feet_l214_214273

noncomputable def projectile_height (t : ℝ) : ℝ :=
  -16 * t^2 + 80 * t

theorem projectile_reaches_100_feet :
  ∃ t : ℝ, t = 2.5 ∧ projectile_height t = 100 :=
by
  use 2.5
  sorry

end projectile_reaches_100_feet_l214_214273


namespace remainder_of_product_of_odd_primes_mod_32_l214_214997

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l214_214997


namespace hyperbola_eccentricity_l214_214824

theorem hyperbola_eccentricity (x y : ℝ) :
  (∃ a b : ℝ, a^2 = 4 ∧ b^2 = 12 ∧ (∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 4) ∧
  ∃ e : ℝ, e = c / a ∧ e = 2) :=
sorry

end hyperbola_eccentricity_l214_214824


namespace acute_angle_sine_l214_214363

theorem acute_angle_sine (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : Real.sin α = 0.58) : (π / 6) < α ∧ α < (π / 4) :=
by
  sorry

end acute_angle_sine_l214_214363


namespace greatest_natural_number_exists_l214_214076

noncomputable def sum_of_squares (n : ℕ) : ℕ :=
    n * (n + 1) * (2 * n + 1) / 6

noncomputable def squared_sum_from_to (a b : ℕ) : ℕ :=
    sum_of_squares b - sum_of_squares (a - 1)

noncomputable def is_perfect_square (n : ℕ) : Prop :=
    ∃ k, k * k = n

theorem greatest_natural_number_exists :
    ∃ n : ℕ, n = 1921 ∧ n ≤ 2008 ∧ 
    is_perfect_square ((sum_of_squares n) * (squared_sum_from_to (n + 1) (2 * n))) :=
by
  sorry

end greatest_natural_number_exists_l214_214076


namespace find_N_l214_214741

theorem find_N (N : ℕ) (hN : N > 1) (h1 : 2019 ≡ 1743 [MOD N]) (h2 : 3008 ≡ 2019 [MOD N]) : N = 23 :=
by
  sorry

end find_N_l214_214741


namespace least_positive_integer_remainder_l214_214038

theorem least_positive_integer_remainder :
  ∃ n : ℕ, (n > 0) ∧ (n % 5 = 1) ∧ (n % 4 = 2) ∧ (∀ m : ℕ, (m > 0) ∧ (m % 5 = 1) ∧ (m % 4 = 2) → n ≤ m) :=
sorry

end least_positive_integer_remainder_l214_214038


namespace squirrels_in_tree_l214_214030

theorem squirrels_in_tree (N S : ℕ) (h₁ : N = 2) (h₂ : S - N = 2) : S = 4 :=
by
  sorry

end squirrels_in_tree_l214_214030


namespace expression_equality_l214_214078

theorem expression_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : (1 / x + 1 / y) = 1 :=
by
  sorry

end expression_equality_l214_214078


namespace geometric_sequence_common_ratio_l214_214214

theorem geometric_sequence_common_ratio {a : ℕ+ → ℝ} (q : ℝ) (h_geom : ∀ n, a (n + 1) = q * a n) 
  (h_a3 : a 3 = 1) (h_a5 : a 5 = 4) : q = 2 ∨ q = -2 :=
by
  sorry

end geometric_sequence_common_ratio_l214_214214


namespace a_minus_b_7_l214_214394

theorem a_minus_b_7 (a b : ℤ) : (2 * y + a) * (y + b) = 2 * y^2 - 5 * y - 12 → a - b = 7 :=
by
  sorry

end a_minus_b_7_l214_214394


namespace smallest_t_eq_3_over_4_l214_214896

theorem smallest_t_eq_3_over_4 (t : ℝ) :
  (∀ t : ℝ,
    (16 * t^3 - 49 * t^2 + 35 * t - 6) / (4 * t - 3) + 7 * t = 8 * t - 2 → t >= (3 / 4)) ∧
  (∃ t₀ : ℝ, (16 * t₀^3 - 49 * t₀^2 + 35 * t₀ - 6) / (4 * t₀ - 3) + 7 * t₀ = 8 * t₀ - 2 ∧ t₀ = (3 / 4)) :=
sorry

end smallest_t_eq_3_over_4_l214_214896


namespace mary_total_earnings_l214_214807

-- Define the earnings for each job
def cleaning_earnings (homes_cleaned : ℕ) : ℕ := 46 * homes_cleaned
def babysitting_earnings (days_babysat : ℕ) : ℕ := 35 * days_babysat
def petcare_earnings (days_petcare : ℕ) : ℕ := 60 * days_petcare

-- Define the total earnings
def total_earnings (homes_cleaned days_babysat days_petcare : ℕ) : ℕ :=
  cleaning_earnings homes_cleaned + babysitting_earnings days_babysat + petcare_earnings days_petcare

-- Given values
def homes_cleaned_last_week : ℕ := 4
def days_babysat_last_week : ℕ := 5
def days_petcare_last_week : ℕ := 3

-- Prove the total earnings
theorem mary_total_earnings : total_earnings homes_cleaned_last_week days_babysat_last_week days_petcare_last_week = 539 :=
by
  -- We just state the theorem; the proof is not required
  sorry

end mary_total_earnings_l214_214807


namespace walter_coins_value_l214_214687

theorem walter_coins_value :
  let pennies : ℕ := 2
  let nickels : ℕ := 2
  let dimes : ℕ := 1
  let quarters : ℕ := 1
  let half_dollars : ℕ := 1
  let penny_value : ℕ := 1
  let nickel_value : ℕ := 5
  let dime_value : ℕ := 10
  let quarter_value : ℕ := 25
  let half_dollar_value : ℕ := 50
  (pennies * penny_value + nickels * nickel_value + dimes * dime_value + quarters * quarter_value + half_dollars * half_dollar_value) = 97 := 
sorry

end walter_coins_value_l214_214687


namespace cathy_wins_probability_l214_214444

theorem cathy_wins_probability : 
  -- Definitions of the problem conditions
  let p_win := (1 : ℚ) / 6
  let p_not_win := (5 : ℚ) / 6
  -- The probability that Cathy wins
  (p_not_win ^ 2 * p_win) / (1 - p_not_win ^ 3) = 25 / 91 :=
by
  sorry

end cathy_wins_probability_l214_214444


namespace gymnastics_performance_participation_l214_214675

def total_people_in_gym_performance (grades : ℕ) (classes_per_grade : ℕ) (students_per_class : ℕ) : ℕ :=
  grades * classes_per_grade * students_per_class

theorem gymnastics_performance_participation :
  total_people_in_gym_performance 3 4 15 = 180 :=
by
  -- This is where the proof would go
  sorry

end gymnastics_performance_participation_l214_214675


namespace total_cups_sold_l214_214859

theorem total_cups_sold (plastic_cups : ℕ) (ceramic_cups : ℕ) (total_sold : ℕ) :
  plastic_cups = 284 ∧ ceramic_cups = 284 → total_sold = 568 :=
by
  intros h
  cases h
  sorry

end total_cups_sold_l214_214859


namespace integer_range_2014_l214_214671

theorem integer_range_2014 : 1000 < 2014 ∧ 2014 < 10000 := by
  sorry

end integer_range_2014_l214_214671


namespace age_comparison_l214_214709

variable (P A F X : ℕ)

theorem age_comparison :
  P = 50 →
  P = 5 / 4 * A →
  P = 5 / 6 * F →
  X = 50 - A →
  X = 10 :=
by { sorry }

end age_comparison_l214_214709


namespace total_rainfall_cm_l214_214872

theorem total_rainfall_cm :
  let monday := 0.12962962962962962
  let tuesday := 3.5185185185185186 * 0.1
  let wednesday := 0.09259259259259259
  let thursday := 0.10222222222222223 * 2.54
  let friday := 12.222222222222221 * 0.1
  let saturday := 0.2222222222222222
  let sunday := 0.17444444444444446 * 2.54
  monday + tuesday + wednesday + thursday + friday + saturday + sunday = 2.721212629851652 :=
by
  sorry

end total_rainfall_cm_l214_214872


namespace distance_from_point_to_x_axis_l214_214105

def distance_to_x_axis (p : ℝ × ℝ) : ℝ :=
  |p.2|

theorem distance_from_point_to_x_axis :
  let p := (-2, -Real.sqrt 5)
  distance_to_x_axis p = Real.sqrt 5 := by
  sorry

end distance_from_point_to_x_axis_l214_214105


namespace miss_davis_items_left_l214_214810

theorem miss_davis_items_left 
  (popsicle_sticks_per_group : ℕ := 15) 
  (straws_per_group : ℕ := 20) 
  (num_groups : ℕ := 10) 
  (total_items_initial : ℕ := 500) : 
  total_items_initial - (num_groups * (popsicle_sticks_per_group + straws_per_group)) = 150 :=
by 
  sorry

end miss_davis_items_left_l214_214810


namespace remainder_of_8_pow_2050_mod_100_l214_214414

theorem remainder_of_8_pow_2050_mod_100 :
  (8 ^ 2050) % 100 = 24 := 
by
  have h1: 8 % 100 = 8 := rfl
  have h2: (8 ^ 2) % 100 = 64 := by norm_num
  have h3: (8 ^ 3) % 100 = 12 := by norm_num
  have h4: (8 ^ 4) % 100 = 96 := by norm_num
  have h5: (8 ^ 5) % 100 = 68 := by norm_num
  have h6: (8 ^ 6) % 100 = 44 := by norm_num
  have h7: (8 ^ 7) % 100 = 52 := by norm_num
  have h8: (8 ^ 8) % 100 = 16 := by norm_num
  have h9: (8 ^ 9) % 100 = 28 := by norm_num
  have h10: (8 ^ 10) % 100 = 24 := by norm_num
  have h20: (8 ^ 20) % 100 = 76 := by norm_num
  have h40: (8 ^ 40) % 100 = 76 := by norm_num
  
  -- Given periodicity and expressing 2050 in terms of periodicity
  have h_periodicity : (8 ^ 2050) % 100 = (8 ^ (20 * 102 + 10)) % 100 := by
    rw [pow_add, pow_mul]
    rw [(8 ^ 2010 % 100), (8 ^ 10) % 100]
    sorry

end remainder_of_8_pow_2050_mod_100_l214_214414


namespace negation_of_existential_proposition_l214_214828

theorem negation_of_existential_proposition :
  ¬(∃ x_0 : ℝ, x_0^2 > Real.exp x_0) ↔ ∀ (x : ℝ), x^2 ≤ Real.exp x :=
by
  sorry

end negation_of_existential_proposition_l214_214828


namespace sum_of_values_of_x_l214_214122

noncomputable def g (x : ℝ) : ℝ :=
if x < 3 then 7 * x + 10 else 3 * x - 18

theorem sum_of_values_of_x (h : ∃ x : ℝ, g x = 5) :
  (∃ x1 x2 : ℝ, g x1 = 5 ∧ g x2 = 5) → (x1 + x2 = 18 / 7) :=
sorry

end sum_of_values_of_x_l214_214122


namespace remainder_of_product_of_odd_primes_mod_32_l214_214992

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l214_214992


namespace find_N_l214_214742

theorem find_N (N : ℕ) (hN : N > 1) (h1 : 2019 ≡ 1743 [MOD N]) (h2 : 3008 ≡ 2019 [MOD N]) : N = 23 :=
by
  sorry

end find_N_l214_214742


namespace ratio_second_to_first_l214_214148

theorem ratio_second_to_first (F S T : ℕ) 
  (hT : T = 2 * F)
  (havg : (F + S + T) / 3 = 77)
  (hmin : F = 33) :
  S / F = 4 :=
by
  sorry

end ratio_second_to_first_l214_214148


namespace sum_of_midpoint_coordinates_l214_214267

theorem sum_of_midpoint_coordinates :
  let x1 := 3
  let y1 := 4
  let x2 := 9
  let y2 := 18
  let midpoint := ((x1 + x2) / 2, (y1 + y2) / 2)
  let sum_of_coordinates := midpoint.fst + midpoint.snd
  sum_of_coordinates = 17 :=
by
  let x1 := 3
  let y1 := 4
  let x2 := 9
  let y2 := 18
  let midpoint := ((x1 + x2) / 2, (y1 + y2) / 2)
  let sum_of_coordinates := midpoint.fst + midpoint.snd
  show sum_of_coordinates = 17
  sorry

end sum_of_midpoint_coordinates_l214_214267


namespace library_books_new_releases_l214_214919

theorem library_books_new_releases (P Q R S : Prop) 
  (h : ¬P) 
  (P_iff_Q : P ↔ Q)
  (Q_implies_R : Q → R)
  (S_iff_notP : S ↔ ¬P) : 
  Q ∧ S := by 
  sorry

end library_books_new_releases_l214_214919


namespace find_values_of_M_l214_214580

theorem find_values_of_M :
  ∃ M : ℕ, 
    (M = 81 ∨ M = 92) ∧ 
    (∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ M = 10 * a + b ∧
     (∃ k : ℕ, k ^ 3 = 9 * (a - b) ∧ k > 0)) :=
sorry

end find_values_of_M_l214_214580


namespace arithmetic_geometric_sequence_l214_214902

theorem arithmetic_geometric_sequence (a b c : ℝ) 
  (a_ne_b : a ≠ b) (b_ne_c : b ≠ c) (a_ne_c : a ≠ c)
  (h1 : 2 * b = a + c)
  (h2 : (a * b)^2 = a * b * c^2)
  (h3 : a + b + c = 15) : a = 20 := 
by 
  sorry

end arithmetic_geometric_sequence_l214_214902


namespace kyoko_payment_l214_214798

noncomputable def total_cost (balls skipropes frisbees : ℕ) (ball_cost rope_cost frisbee_cost : ℝ) : ℝ :=
  (balls * ball_cost) + (skipropes * rope_cost) + (frisbees * frisbee_cost)

noncomputable def final_amount (total_cost discount_rate : ℝ) : ℝ :=
  total_cost - (discount_rate * total_cost)

theorem kyoko_payment :
  let balls := 3
  let skipropes := 2
  let frisbees := 4
  let ball_cost := 1.54
  let rope_cost := 3.78
  let frisbee_cost := 2.63
  let discount_rate := 0.07
  final_amount (total_cost balls skipropes frisbees ball_cost rope_cost frisbee_cost) discount_rate = 21.11 :=
by
  sorry

end kyoko_payment_l214_214798


namespace remainder_when_divided_by_32_l214_214961

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l214_214961


namespace watermelons_eaten_l214_214130

theorem watermelons_eaten (original left : ℕ) (h1 : original = 4) (h2 : left = 1) :
  original - left = 3 :=
by {
  -- Providing the proof steps is not necessary as per the instructions
  sorry
}

end watermelons_eaten_l214_214130


namespace find_x_in_gp_l214_214340

theorem find_x_in_gp :
  ∃ x : ℤ, (30 + x)^2 = (10 + x) * (90 + x) ∧ x = 0 :=
by
  sorry

end find_x_in_gp_l214_214340


namespace remainder_of_M_l214_214984

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l214_214984


namespace angle_relation_l214_214926

-- Definitions for the triangle properties and angles.
variables {α : Type*} [LinearOrderedField α]
variables {A B C D E F : α}

-- Definitions stating the properties of the triangles.
def is_isosceles_triangle (a b c : α) : Prop :=
  a = b ∨ b = c ∨ c = a

def triangle_ABC_is_isosceles (AB AC : α) (ABC : α) : Prop :=
  is_isosceles_triangle AB AC ABC

def triangle_DEF_is_isosceles (DE DF : α) (DEF : α) : Prop :=
  is_isosceles_triangle DE DF DEF

-- Condition that gives the specific angle measure in triangle DEF.
def angle_DEF_is_100 (DEF : α) : Prop :=
  DEF = 100

-- The main theorem to prove.
theorem angle_relation (AB AC DE DF DEF a b c : α) :
  triangle_ABC_is_isosceles AB AC (AB + AC) →
  triangle_DEF_is_isosceles DE DF DEF →
  angle_DEF_is_100 DEF →
  a = c :=
by
  -- Assuming the conditions define the angles and state the relationship.
  sorry

end angle_relation_l214_214926


namespace quadrilateral_probability_l214_214070

def sticks : List ℕ := [1, 3, 4, 6, 8, 9, 10, 12]

def isValidSet (s : Finset ℕ) : Prop :=
  s.card = 4 ∧ 
  ∀ (x ∈ s), (s.sum - x) > x

def validCombCount : ℕ := 
  (Finset.powerset (Finset.of_list sticks)).filter (λ s => isValidSet s).card

def totalCombCount : ℕ := Nat.choose 8 4

theorem quadrilateral_probability :
  (validCombCount : ℚ) / totalCombCount = 9 / 14 :=
by
  sorry

end quadrilateral_probability_l214_214070


namespace remainder_of_M_l214_214985

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l214_214985


namespace cindy_correct_answer_l214_214726

noncomputable def cindy_number (x : ℝ) : Prop :=
  (x - 10) / 5 = 40

theorem cindy_correct_answer (x : ℝ) (h : cindy_number x) : (x - 4) / 10 = 20.6 :=
by
  -- The proof is omitted as instructed
  sorry

end cindy_correct_answer_l214_214726


namespace product_of_odd_primes_mod_32_l214_214958

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l214_214958


namespace track_width_track_area_l214_214179

theorem track_width (r1 r2 : ℝ) (h1 : 2 * π * r1 - 2 * π * r2 = 24 * π) : r1 - r2 = 12 :=
by sorry

theorem track_area (r1 r2 : ℝ) (h1 : r1 = r2 + 12) : π * (r1^2 - r2^2) = π * (24 * r2 + 144) :=
by sorry

end track_width_track_area_l214_214179


namespace initial_money_is_correct_l214_214534

-- Given conditions
def spend_per_trip : ℕ := 2
def trips_per_month : ℕ := 4
def months_per_year : ℕ := 12
def money_left_after_year : ℕ := 104

-- Define the initial amount of money
def initial_amount_of_money (spend_per_trip trips_per_month months_per_year money_left_after_year : ℕ) : ℕ :=
  money_left_after_year + (spend_per_trip * trips_per_month * months_per_year)

-- Theorem stating that under the given conditions, the initial amount of money is 200
theorem initial_money_is_correct :
  initial_amount_of_money spend_per_trip trips_per_month months_per_year money_left_after_year = 200 :=
  sorry

end initial_money_is_correct_l214_214534


namespace ajax_weight_after_exercise_l214_214581

theorem ajax_weight_after_exercise :
  ∀ (initial_weight_kg : ℕ) (conversion_rate : ℝ) (daily_exercise_hours : ℕ) (exercise_loss_rate : ℝ) (days_in_week : ℕ) (weeks : ℕ),
    initial_weight_kg = 80 →
    conversion_rate = 2.2 →
    daily_exercise_hours = 2 →
    exercise_loss_rate = 1.5 →
    days_in_week = 7 →
    weeks = 2 →
    initial_weight_kg * conversion_rate - daily_exercise_hours * exercise_loss_rate * (days_in_week * weeks) = 134 :=
by
  intros
  sorry

end ajax_weight_after_exercise_l214_214581


namespace value_of_v3_at_2_l214_214879

def f (x : ℝ) : ℝ := x^5 - 2 * x^4 + 3 * x^3 - 7 * x^2 + 6 * x - 3

def v3 (x : ℝ) := (x - 2) * x + 3 
def v3_eval_at_2 : ℝ := (2 - 2) * 2 + 3

theorem value_of_v3_at_2 : v3 2 - 7 = -1 := by
    sorry

end value_of_v3_at_2_l214_214879


namespace find_cost_per_kg_l214_214140

-- Define the conditions given in the problem
def side_length : ℕ := 30
def coverage_per_kg : ℕ := 20
def total_cost : ℕ := 10800

-- The cost per kg we need to find
def cost_per_kg := total_cost / ((6 * side_length^2) / coverage_per_kg)

-- We need to prove that cost_per_kg = 40
theorem find_cost_per_kg : cost_per_kg = 40 := by
  sorry

end find_cost_per_kg_l214_214140


namespace number_of_girls_in_school_l214_214544

theorem number_of_girls_in_school :
  ∃ G B : ℕ, 
    G + B = 1600 ∧
    (G * 200 / 1600) - 20 = (B * 200 / 1600) ∧
    G = 860 :=
by
  sorry

end number_of_girls_in_school_l214_214544


namespace find_x_l214_214819

theorem find_x (c d : ℝ) (y z x : ℝ) 
  (h1 : y^2 = c * z^2) 
  (h2 : y = d / x)
  (h3 : y = 3) 
  (h4 : x = 4) 
  (h5 : z = 6) 
  (h6 : y = 2) 
  (h7 : z = 12) 
  : x = 6 := 
by
  sorry

end find_x_l214_214819


namespace unique_solution_l214_214194

theorem unique_solution (x y z : ℝ) (h₁ : x^2 + y^2 + z^2 = 2) (h₂ : x = z + 2) :
  x = 1 ∧ y = 0 ∧ z = -1 :=
by
  sorry

end unique_solution_l214_214194


namespace susan_hours_per_day_l214_214820

theorem susan_hours_per_day (h : ℕ) 
  (works_five_days_a_week : Prop)
  (paid_vacation_days : ℕ)
  (unpaid_vacation_days : ℕ)
  (missed_pay : ℕ)
  (hourly_rate : ℕ)
  (total_vacation_days : ℕ)
  (total_workdays_in_2_weeks : ℕ)
  (paid_vacation_days_eq : paid_vacation_days = 6)
  (unpaid_vacation_days_eq : unpaid_vacation_days = 4)
  (missed_pay_eq : missed_pay = 480)
  (hourly_rate_eq : hourly_rate = 15)
  (total_vacation_days_eq : total_vacation_days = 14)
  (total_workdays_in_2_weeks_eq : total_workdays_in_2_weeks = 10)
  (total_unpaid_hours_in_4_days : unpaid_vacation_days * hourly_rate = missed_pay) :
  h = 8 :=
by 
  -- We need to show that Susan works 8 hours a day
  sorry

end susan_hours_per_day_l214_214820


namespace mod_residue_17_l214_214333

theorem mod_residue_17 : (513 + 3 * 68 + 9 * 289 + 2 * 34 - 10) % 17 = 7 := by
  -- We first compute the modulo 17 residue of each term given in the problem:
  -- 513 == 0 % 17
  -- 68 == 0 % 17
  -- 289 == 0 % 17
  -- 34 == 0 % 17
  -- -10 == 7 % 17
  sorry

end mod_residue_17_l214_214333


namespace minimize_blue_surface_l214_214858

noncomputable def fraction_blue_surface_area : ℚ := 1 / 8

theorem minimize_blue_surface
  (total_cubes : ℕ)
  (blue_cubes : ℕ)
  (green_cubes : ℕ)
  (edge_length : ℕ)
  (surface_area : ℕ)
  (blue_surface_area : ℕ)
  (fraction_blue : ℚ)
  (h1 : total_cubes = 64)
  (h2 : blue_cubes = 20)
  (h3 : green_cubes = 44)
  (h4 : edge_length = 4)
  (h5 : surface_area = 6 * edge_length^2)
  (h6 : blue_surface_area = 12)
  (h7 : fraction_blue = blue_surface_area / surface_area) :
  fraction_blue = fraction_blue_surface_area :=
by
  sorry

end minimize_blue_surface_l214_214858


namespace tan_theta_minus_pi_over_4_l214_214086

theorem tan_theta_minus_pi_over_4 (θ : Real) (h1 : θ ∈ Set.Ioc (-(π / 2)) 0)
  (h2 : Real.sin (θ + π / 4) = 3 / 5) : Real.tan (θ - π / 4) = - (4 / 3) :=
by
  /- Proof goes here -/
  sorry

end tan_theta_minus_pi_over_4_l214_214086


namespace product_of_odd_primes_mod_32_l214_214955

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l214_214955


namespace square_area_less_than_circle_area_l214_214684

theorem square_area_less_than_circle_area (a : ℝ) (ha : 0 < a) :
    let S1 := (a / 4) ^ 2
    let r := a / (2 * Real.pi)
    let S2 := Real.pi * r^2
    (S1 < S2) := by
sorry

end square_area_less_than_circle_area_l214_214684


namespace ratio_age_difference_to_pencils_l214_214136

-- Definitions of the given problem conditions
def AsafAge : ℕ := 50
def SumOfAges : ℕ := 140
def AlexanderAge : ℕ := SumOfAges - AsafAge

def PencilDifference : ℕ := 60
def TotalPencils : ℕ := 220
def AsafPencils : ℕ := (TotalPencils - PencilDifference) / 2
def AlexanderPencils : ℕ := AsafPencils + PencilDifference

-- Define the age difference and the ratio
def AgeDifference : ℕ := AlexanderAge - AsafAge
def Ratio : ℚ := AgeDifference / AsafPencils

theorem ratio_age_difference_to_pencils : Ratio = 1 / 2 := by
  sorry

end ratio_age_difference_to_pencils_l214_214136


namespace f_value_at_4_l214_214545

def f : ℝ → ℝ := sorry  -- Define f as a function from ℝ to ℝ

-- Specify the condition that f satisfies for all real numbers x
axiom f_condition (x : ℝ) : f (2^x) + x * f (2^(-x)) = 3

-- Statement to be proven: f(4) = -3
theorem f_value_at_4 : f 4 = -3 :=
by {
  -- Proof goes here
  sorry
}

end f_value_at_4_l214_214545


namespace domain_of_function_l214_214730

theorem domain_of_function (x : ℝ) : 
  {x | ∃ k : ℤ, - (Real.pi / 3) + (2 : ℝ) * k * Real.pi ≤ x ∧ x ≤ (Real.pi / 3) + (2 : ℝ) * k * Real.pi} :=
by
  -- Proof omitted
  sorry

end domain_of_function_l214_214730


namespace perpendicular_lines_slope_condition_l214_214761

theorem perpendicular_lines_slope_condition (k : ℝ) :
  (∀ x y : ℝ, y = k * x - 1 ↔ x + 2 * y + 3 = 0) → k = 2 :=
by
  sorry

end perpendicular_lines_slope_condition_l214_214761


namespace greatest_divisor_l214_214565

theorem greatest_divisor (d : ℕ) :
  (690 % d = 10) ∧ (875 % d = 25) ∧ ∀ e : ℕ, (690 % e = 10) ∧ (875 % e = 25) → (e ≤ d) :=
  sorry

end greatest_divisor_l214_214565


namespace number_of_planes_l214_214925

-- Definitions based on the conditions
def Line (space: Type) := space → space → Prop

variables {space: Type} [MetricSpace space]

-- Given conditions
variable (l1 l2 l3 : Line space)
variable (intersects : ∃ p : space, l1 p p ∧ l2 p p ∧ l3 p p)

-- The theorem stating the conclusion
theorem number_of_planes (h: ∃ p : space, l1 p p ∧ l2 p p ∧ l3 p p) :
  (1 = 1 ∨ 1 = 2 ∨ 1 = 3) ∨ (2 = 1 ∨ 2 = 2 ∨ 2 = 3) ∨ (3 = 1 ∨ 3 = 2 ∨ 3 = 3) := 
sorry

end number_of_planes_l214_214925


namespace number_line_problem_l214_214248

theorem number_line_problem (x : ℤ) (h : x + 7 - 4 = 0) : x = -3 :=
by
  -- The proof is omitted as only the statement is required.
  sorry

end number_line_problem_l214_214248


namespace find_full_price_l214_214382

-- Defining the conditions
variables (P : ℝ) 
-- The condition that 20% of the laptop's total cost is $240.
def condition : Prop := 0.2 * P = 240

-- The proof goal is to show that the full price P is $1200 given the condition
theorem find_full_price (h : condition P) : P = 1200 :=
sorry

end find_full_price_l214_214382


namespace Zhukov_birth_year_l214_214669

-- Define the conditions
def years_lived_total : ℕ := 78
def years_lived_20th_more_than_19th : ℕ := 70

-- Define the proof problem
theorem Zhukov_birth_year :
  ∃ y19 y20 : ℕ, y19 + y20 = years_lived_total ∧ y20 = y19 + years_lived_20th_more_than_19th ∧ (1900 - y19) = 1896 :=
by
  sorry

end Zhukov_birth_year_l214_214669


namespace parabola_directrix_eq_l214_214092

theorem parabola_directrix_eq (a : ℝ) (h : - a / 4 = - (1 : ℝ) / 4) : a = 1 := by
  sorry

end parabola_directrix_eq_l214_214092


namespace angle_405_eq_45_l214_214046

def same_terminal_side (angle1 angle2 : ℝ) : Prop :=
  ∃ k : ℤ, angle1 = angle2 + k * 360

theorem angle_405_eq_45 (k : ℤ) : same_terminal_side 405 45 := 
sorry

end angle_405_eq_45_l214_214046


namespace mediant_fraction_of_6_11_and_5_9_minimized_is_31_l214_214380

theorem mediant_fraction_of_6_11_and_5_9_minimized_is_31 
  (p q : ℕ) (h_pos : 0 < p ∧ 0 < q)
  (h_bounds : (6 : ℝ) / 11 < p / q ∧ p / q < 5 / 9)
  (h_min_q : ∀ r s : ℕ, (6 : ℝ) / 11 < r / s ∧ r / s < 5 / 9 → s ≥ q) :
  p + q = 31 :=
sorry

end mediant_fraction_of_6_11_and_5_9_minimized_is_31_l214_214380


namespace rectangles_containment_existence_l214_214291

theorem rectangles_containment_existence :
  (∃ (rects : ℕ → ℕ × ℕ), (∀ n : ℕ, (rects n).fst > 0 ∧ (rects n).snd > 0) ∧
   (∀ n m : ℕ, n ≠ m → ¬((rects n).fst ≤ (rects m).fst ∧ (rects n).snd ≤ (rects m).snd))) →
  false :=
by
  sorry

end rectangles_containment_existence_l214_214291


namespace sequence_formula_l214_214904

-- Definitions of the sequence and conditions
def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) a

def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S a n + a n = 2 * n + 1

-- Proposition to prove
theorem sequence_formula (a : ℕ → ℝ) (h : satisfies_condition a) : 
  ∀ n : ℕ, a n = 2 - 1 / 2^n := sorry

end sequence_formula_l214_214904


namespace distance_from_P_to_y_axis_l214_214765

theorem distance_from_P_to_y_axis 
  (x y : ℝ)
  (h1 : (x^2 / 16) + (y^2 / 25) = 1)
  (F1 : ℝ × ℝ := (0, -3))
  (F2 : ℝ × ℝ := (0, 3))
  (h2 : (F1.1 - x)^2 + (F1.2 - y)^2 = 9 ∨ (F2.1 - x)^2 + (F2.2 - y)^2 = 9 
          ∨ (F1.1 - x)^2 + (F1.2 - y)^2 + (F2.1 - x)^2 + (F2.2 - y)^2 = (F1.1 - F2.1)^2 + (F1.2 - F2.2)^2) :
  |x| = 16 / 5 :=
by
  sorry

end distance_from_P_to_y_axis_l214_214765


namespace set_membership_l214_214074

theorem set_membership :
  {m : ℤ | ∃ k : ℤ, 10 = k * (m + 1)} = {-11, -6, -3, -2, 0, 1, 4, 9} :=
by sorry

end set_membership_l214_214074


namespace max_pizzas_l214_214927

theorem max_pizzas (dough_available cheese_available sauce_available pepperoni_available mushroom_available olive_available sausage_available: ℝ)
  (dough_per_pizza cheese_per_pizza sauce_per_pizza toppings_per_pizza: ℝ)
  (total_toppings: ℝ)
  (toppings_per_pizza_sum: total_toppings = pepperoni_available + mushroom_available + olive_available + sausage_available)
  (dough_cond: dough_available = 200)
  (cheese_cond: cheese_available = 20)
  (sauce_cond: sauce_available = 20)
  (pepperoni_cond: pepperoni_available = 15)
  (mushroom_cond: mushroom_available = 5)
  (olive_cond: olive_available = 5)
  (sausage_cond: sausage_available = 10)
  (dough_per_pizza_cond: dough_per_pizza = 1)
  (cheese_per_pizza_cond: cheese_per_pizza = 1/4)
  (sauce_per_pizza_cond: sauce_per_pizza = 1/6)
  (toppings_per_pizza_cond: toppings_per_pizza = 1/3)
  : (min (dough_available / dough_per_pizza) (min (cheese_available / cheese_per_pizza) (min (sauce_available / sauce_per_pizza) (total_toppings / toppings_per_pizza))) = 80) :=
by
  sorry

end max_pizzas_l214_214927


namespace intersection_M_N_l214_214518

def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | (1/3) ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x | (1/3) ≤ x ∧ x < 4} := by
  sorry

end intersection_M_N_l214_214518


namespace probability_of_non_defective_is_seven_ninetyninths_l214_214369

-- Define the number of total pencils, defective pencils, and the number of pencils selected
def total_pencils : ℕ := 12
def defective_pencils : ℕ := 4
def selected_pencils : ℕ := 5

-- Define the number of ways to choose k elements from n elements (the combination function)
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculate the total number of ways to choose 5 pencils out of 12
def total_ways : ℕ := combination total_pencils selected_pencils

-- Calculate the number of non-defective pencils
def non_defective_pencils : ℕ := total_pencils - defective_pencils

-- Calculate the number of ways to choose 5 non-defective pencils out of 8
def non_defective_ways : ℕ := combination non_defective_pencils selected_pencils

-- Calculate the probability that all 5 chosen pencils are non-defective
def probability_non_defective : ℚ :=
  non_defective_ways / total_ways

-- Prove that this probability equals 7/99
theorem probability_of_non_defective_is_seven_ninetyninths :
  probability_non_defective = 7 / 99 :=
by
  -- The proof is left as an exercise
  sorry

end probability_of_non_defective_is_seven_ninetyninths_l214_214369


namespace nat_set_satisfy_inequality_l214_214676

theorem nat_set_satisfy_inequality :
  {x : ℕ | -3 < 2 * x - 1 ∧ 2 * x - 1 ≤ 3} = {0, 1, 2} :=
by
  sorry

end nat_set_satisfy_inequality_l214_214676


namespace least_positive_integer_l214_214040

theorem least_positive_integer (a : ℕ) :
  (a % 2 = 0) ∧ (a % 5 = 1) ∧ (a % 4 = 2) → a = 6 :=
by
  sorry

end least_positive_integer_l214_214040


namespace distance_between_stripes_correct_l214_214863

noncomputable def distance_between_stripes : ℝ :=
  let base1 := 20
  let height1 := 50
  let base2 := 65
  let area := base1 * height1
  let d := area / base2
  d

theorem distance_between_stripes_correct : distance_between_stripes = 200 / 13 := by
  sorry

end distance_between_stripes_correct_l214_214863


namespace find_AD_l214_214646

-- Given conditions as definitions
def AB := 5 -- given length in meters
def angle_ABC := 85 -- given angle in degrees
def angle_BCA := 45 -- given angle in degrees
def angle_DBC := 20 -- given angle in degrees

-- Lean theorem statement to prove the result
theorem find_AD : AD = AB := by
  -- The proof will be filled in afterwards; currently, we leave it as sorry.
  sorry

end find_AD_l214_214646


namespace apples_used_l214_214167

theorem apples_used (apples_before : ℕ) (apples_left : ℕ) (apples_used_for_pie : ℕ) 
                    (h1 : apples_before = 19) 
                    (h2 : apples_left = 4) 
                    (h3 : apples_used_for_pie = apples_before - apples_left) : 
  apples_used_for_pie = 15 :=
by
  -- Since we are instructed to leave the proof out, we put sorry here
  sorry

end apples_used_l214_214167


namespace part1_solution_part2_solution_l214_214488

-- Part (1)
theorem part1_solution (x : ℝ) : (|x - 2| + |x - 1| ≥ 2) ↔ (x ≥ 2.5 ∨ x ≤ 0.5) := sorry

-- Part (2)
theorem part2_solution (a : ℝ) (h : a > 0) : (∀ x, |a * x - 2| + |a * x - a| ≥ 2) → a ≥ 4 := sorry

end part1_solution_part2_solution_l214_214488


namespace sum_of_consecutive_integers_l214_214744

theorem sum_of_consecutive_integers (a : ℤ) (h₁ : a = 18) (h₂ : a + 1 = 19) (h₃ : a + 2 = 20) : a + (a + 1) + (a + 2) = 57 :=
by
  -- Add a sorry to focus on creating the statement successfully
  sorry

end sum_of_consecutive_integers_l214_214744


namespace sheets_in_stack_l214_214862

theorem sheets_in_stack 
  (num_sheets : ℕ) 
  (initial_thickness final_thickness : ℝ) 
  (t_per_sheet : ℝ) 
  (h_initial : num_sheets = 800) 
  (h_thickness : initial_thickness = 4) 
  (h_thickness_per_sheet : initial_thickness / num_sheets = t_per_sheet) 
  (h_final_thickness : final_thickness = 6) 
  : num_sheets * (final_thickness / t_per_sheet) = 1200 := 
by 
  sorry

end sheets_in_stack_l214_214862


namespace queenie_daily_earnings_l214_214533

/-- Define the overtime earnings per hour. -/
def overtime_pay_per_hour : ℤ := 5

/-- Define the total amount received. -/
def total_received : ℤ := 770

/-- Define the number of days worked. -/
def days_worked : ℤ := 5

/-- Define the number of overtime hours. -/
def overtime_hours : ℤ := 4

/-- State the theorem to find out Queenie's daily earnings. -/
theorem queenie_daily_earnings :
  ∃ D : ℤ, days_worked * D + overtime_hours * overtime_pay_per_hour = total_received ∧ D = 150 :=
by
  use 150
  sorry

end queenie_daily_earnings_l214_214533


namespace gcd_of_180_and_450_l214_214599

theorem gcd_of_180_and_450 : Int.gcd 180 450 = 90 := 
  sorry

end gcd_of_180_and_450_l214_214599


namespace range_of_a_l214_214203

noncomputable def p (x : ℝ) : Prop := (1 / (x - 3)) ≥ 1

noncomputable def q (x a : ℝ) : Prop := abs (x - a) < 1

theorem range_of_a (a : ℝ) : (∀ x, p x → q x a) ∧ (∃ x, ¬ (p x) ∧ (q x a)) ↔ (3 < a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l214_214203


namespace stan_run_duration_l214_214817

def run_duration : ℕ := 100

def num_3_min_songs : ℕ := 10
def num_2_min_songs : ℕ := 15
def time_per_3_min_song : ℕ := 3
def time_per_2_min_song : ℕ := 2
def additional_time_needed : ℕ := 40

theorem stan_run_duration :
  (num_3_min_songs * time_per_3_min_song) + (num_2_min_songs * time_per_2_min_song) + additional_time_needed = run_duration := by
  sorry

end stan_run_duration_l214_214817


namespace weight_of_new_person_l214_214425

variable (avg_increase : ℝ) (n_persons : ℕ) (weight_replaced : ℝ)

theorem weight_of_new_person (h1 : avg_increase = 3.5) (h2 : n_persons = 8) (h3 : weight_replaced = 65) :
  let total_weight_increase := n_persons * avg_increase
  let weight_new := weight_replaced + total_weight_increase
  weight_new = 93 := by
  sorry

end weight_of_new_person_l214_214425


namespace Kevin_crates_per_week_l214_214797

theorem Kevin_crates_per_week (a b c : ℕ) (h₁ : a = 13) (h₂ : b = 20) (h₃ : c = 17) :
  a + b + c = 50 :=
by 
  sorry

end Kevin_crates_per_week_l214_214797


namespace repeating_decimal_to_fraction_l214_214047

theorem repeating_decimal_to_fraction :
  (0.512341234123412341234 : ℝ) = (51229 / 99990 : ℝ) :=
sorry

end repeating_decimal_to_fraction_l214_214047


namespace remainder_when_M_divided_by_32_l214_214949

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l214_214949


namespace find_r_in_arithmetic_sequence_l214_214784

-- Define an arithmetic sequence
def is_arithmetic_sequence (a b c d e f : ℤ) : Prop :=
  (b - a = c - b) ∧ (c - b = d - c) ∧ (d - c = e - d) ∧ (e - d = f - e)

-- Define the given problem
theorem find_r_in_arithmetic_sequence :
  ∃ r : ℤ, ∀ p q s : ℤ, is_arithmetic_sequence 23 p q r s 59 → r = 41 :=
by
  sorry

end find_r_in_arithmetic_sequence_l214_214784


namespace pants_cost_is_250_l214_214874

-- Define the cost of a T-shirt
def tshirt_cost := 100

-- Define the total amount spent
def total_amount := 1500

-- Define the number of T-shirts bought
def num_tshirts := 5

-- Define the number of pants bought
def num_pants := 4

-- Define the total cost of T-shirts
def total_tshirt_cost := tshirt_cost * num_tshirts

-- Define the total cost of pants
def total_pants_cost := total_amount - total_tshirt_cost

-- Define the cost per pair of pants
def pants_cost_per_pair := total_pants_cost / num_pants

-- Proving that the cost per pair of pants is $250
theorem pants_cost_is_250 : pants_cost_per_pair = 250 := by
  sorry

end pants_cost_is_250_l214_214874


namespace half_of_one_point_zero_one_l214_214842

theorem half_of_one_point_zero_one : (1.01 / 2) = 0.505 := 
by
  sorry

end half_of_one_point_zero_one_l214_214842


namespace total_dolphins_l214_214367

theorem total_dolphins (initial_dolphins : ℕ) (triple_of_initial : ℕ) (final_dolphins : ℕ) 
    (h1 : initial_dolphins = 65) (h2 : triple_of_initial = 3 * initial_dolphins) (h3 : final_dolphins = initial_dolphins + triple_of_initial) : 
    final_dolphins = 260 :=
by
  -- Proof goes here
  sorry

end total_dolphins_l214_214367


namespace max_correct_answers_l214_214368

theorem max_correct_answers :
  ∀ (a b c : ℕ), a + b + c = 60 ∧ 4 * a - c = 112 → a ≤ 34 :=
by
  sorry

end max_correct_answers_l214_214368


namespace solve_for_x_l214_214257

theorem solve_for_x (x : ℝ) : (5 * x - 2) / (6 * x - 6) = 3 / 4 ↔ x = -5 := by
  sorry

end solve_for_x_l214_214257


namespace pages_read_first_day_l214_214185

-- Alexa is reading a Nancy Drew mystery with 95 pages.
def total_pages : ℕ := 95

-- She read 58 pages the next day.
def pages_read_second_day : ℕ := 58

-- She has 19 pages left to read.
def pages_left_to_read : ℕ := 19

-- How many pages did she read on the first day?
theorem pages_read_first_day : total_pages - pages_read_second_day - pages_left_to_read = 18 := by
  -- Proof is omitted as instructed
  sorry

end pages_read_first_day_l214_214185


namespace saree_discount_l214_214830

theorem saree_discount (x : ℝ) : 
  let original_price := 495
  let final_price := 378.675
  let discounted_price := original_price * ((100 - x) / 100) * 0.9
  discounted_price = final_price -> x = 15 := 
by
  intro h
  sorry

end saree_discount_l214_214830


namespace temp_on_Monday_l214_214666

variable (M T W Th F : ℤ)

-- Given conditions
axiom sum_MTWT : M + T + W + Th = 192
axiom sum_TWTF : T + W + Th + F = 184
axiom temp_F : F = 34
axiom exists_day_temp_42 : ∃ (day : String), 
  (day = "Monday" ∨ day = "Tuesday" ∨ day = "Wednesday" ∨ day = "Thursday" ∨ day = "Friday") ∧
  (if day = "Monday" then M else if day = "Tuesday" then T else if day = "Wednesday" then W else if day = "Thursday" then Th else F) = 42

-- Prove temperature of Monday is 42
theorem temp_on_Monday : M = 42 := 
by
  sorry

end temp_on_Monday_l214_214666


namespace arithmetic_sequence_general_term_and_sum_l214_214371

theorem arithmetic_sequence_general_term_and_sum :
  (∃ (a₁ d : ℤ), a₁ + d = 14 ∧ a₁ + 4 * d = 5 ∧ ∀ n : ℤ, a_n = a₁ + (n - 1) * d ∧ (∀ N : ℤ, N ≥ 1 → S_N = N * ((2 * a₁ + (N - 1) * d) / 2) ∧ N = 10 → S_N = 35)) :=
sorry

end arithmetic_sequence_general_term_and_sum_l214_214371


namespace xy_plus_y_square_l214_214800

theorem xy_plus_y_square {x y : ℝ} (h1 : x * y = 16) (h2 : x + y = 8) : x^2 + y^2 = 32 :=
sorry

end xy_plus_y_square_l214_214800


namespace proof_expectation_red_balls_drawn_l214_214640

noncomputable def expectation_red_balls_drawn : Prop :=
  let total_ways := Nat.choose 5 2
  let ways_2_red := Nat.choose 3 2
  let ways_1_red_1_yellow := Nat.choose 3 1 * Nat.choose 2 1
  let p_X_eq_2 := (ways_2_red : ℝ) / total_ways
  let p_X_eq_1 := (ways_1_red_1_yellow : ℝ) / total_ways
  let expectation := 2 * p_X_eq_2 + 1 * p_X_eq_1
  expectation = 1.2

theorem proof_expectation_red_balls_drawn :
  expectation_red_balls_drawn :=
by
  sorry

end proof_expectation_red_balls_drawn_l214_214640


namespace handshake_problem_l214_214721

theorem handshake_problem :
  let team_size := 6
  let teams := 2
  let referees := 3
  let handshakes_between_teams := team_size * team_size
  let handshakes_within_teams := teams * (team_size * (team_size - 1)) / 2
  let handshakes_with_referees := (teams * team_size) * referees
  handshakes_between_teams + handshakes_within_teams + handshakes_with_referees = 102 := by
  sorry

end handshake_problem_l214_214721


namespace lily_ducks_l214_214129

variable (D G : ℕ)
variable (Rayden_ducks : ℕ := 3 * D)
variable (Rayden_geese : ℕ := 4 * G)
variable (Lily_geese : ℕ := 10) -- Given G = 10
variable (Rayden_extra : ℕ := 70) -- Given Rayden has 70 more ducks and geese

theorem lily_ducks (h : 3 * D + 4 * Lily_geese = D + Lily_geese + Rayden_extra) : D = 20 :=
by sorry

end lily_ducks_l214_214129


namespace ratio_students_sent_home_to_remaining_l214_214314

theorem ratio_students_sent_home_to_remaining (total_students : ℕ) (students_taken_to_beach : ℕ)
    (students_still_in_school : ℕ) (students_sent_home : ℕ) 
    (h1 : total_students = 1000) (h2 : students_taken_to_beach = total_students / 2)
    (h3 : students_still_in_school = 250) 
    (h4 : students_sent_home = total_students / 2 - students_still_in_school) :
    (students_sent_home / students_still_in_school) = 1 := 
by
    sorry

end ratio_students_sent_home_to_remaining_l214_214314


namespace jill_spending_on_clothing_l214_214811

theorem jill_spending_on_clothing (C : ℝ) (T : ℝ)
  (h1 : 0.2 * T = 0.2 * T)
  (h2 : 0.3 * T = 0.3 * T)
  (h3 : (C / 100) * T * 0.04 + 0.3 * T * 0.08 = 0.044 * T) :
  C = 50 :=
by
  -- This line indicates the point where the proof would typically start
  sorry

end jill_spending_on_clothing_l214_214811


namespace stratified_sampling_l214_214707

noncomputable def employees := 500
noncomputable def under_35 := 125
noncomputable def between_35_and_49 := 280
noncomputable def over_50 := 95
noncomputable def sample_size := 100

theorem stratified_sampling : 
  under_35 * sample_size / employees = 25 := by
  sorry

end stratified_sampling_l214_214707


namespace odd_function_fixed_point_l214_214493

noncomputable def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

theorem odd_function_fixed_point 
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f) :
  f (0) = 0 → f (-1 + 1) - 2 = -2 :=
by
  sorry

end odd_function_fixed_point_l214_214493


namespace f_fe_eq_neg1_f_x_gt_neg1_solution_l214_214474

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log (1 / x)
  else if x < 0 then 1 / x
  else 0  -- handle the case for x = 0 explicitly if needed

theorem f_fe_eq_neg1 : 
  f (f (Real.exp 1)) = -1 := 
by
  -- proof to be filled in
  sorry

theorem f_x_gt_neg1_solution :
  {x : ℝ | f x > -1} = {x : ℝ | (x < -1) ∨ (0 < x ∧ x < Real.exp 1)} :=
by
  -- proof to be filled in
  sorry

end f_fe_eq_neg1_f_x_gt_neg1_solution_l214_214474


namespace number_smaller_than_neg3_exists_l214_214439

def numbers := [0, -1, -5, -1/2]

theorem number_smaller_than_neg3_exists : ∃ x ∈ numbers, x < -3 :=
by
  let x := -5
  have h : x ∈ numbers := by simp [numbers]
  have h_lt : x < -3 := by norm_num
  exact ⟨x, h, h_lt⟩ -- show that -5 meets the criteria

end number_smaller_than_neg3_exists_l214_214439


namespace pool_capacity_l214_214227

variables {T : ℕ} {A B C : ℕ → ℕ}

-- Conditions
def valve_rate_A (T : ℕ) : ℕ := T / 180
def valve_rate_B (T : ℕ) := valve_rate_A T + 60
def valve_rate_C (T : ℕ) := valve_rate_A T + 75

def combined_rate (T : ℕ) := valve_rate_A T + valve_rate_B T + valve_rate_C T

-- Theorem to prove
theorem pool_capacity (T : ℕ) (h1 : combined_rate T = T / 40) : T = 16200 :=
by
  sorry

end pool_capacity_l214_214227


namespace jose_investment_l214_214151

theorem jose_investment (P T : ℝ) (X : ℝ) (months_tom months_jose : ℝ) (profit_total profit_jose profit_tom : ℝ) :
  T = 30000 →
  months_tom = 12 →
  months_jose = 10 →
  profit_total = 54000 →
  profit_jose = 30000 →
  profit_tom = profit_total - profit_jose →
  profit_tom / profit_jose = (T * months_tom) / (X * months_jose) →
  X = 45000 :=
by sorry

end jose_investment_l214_214151


namespace find_slope_of_line_l214_214217

-- Define the parabola, point M, and the conditions leading to the slope k.
theorem find_slope_of_line (k : ℝ) :
  let C := {p : ℝ × ℝ | p.2^2 = 4 * p.1}
  let focus : ℝ × ℝ := (1, 0)
  let M : ℝ × ℝ := (-1, 1)
  let line (k : ℝ) (x : ℝ) := k * (x - 1)
  ∃ A B : (ℝ × ℝ), 
    A ∈ C ∧ B ∈ C ∧
    A ≠ B ∧
    A.1 + 1 = B.1 + 1 ∧ 
    A.2 - 1 = B.2 - 1 ∧
    ((A.1 + 1) * (B.1 + 1) + (A.2 - 1) * (B.2 - 1) = 0) -> k = 2 := 
by
  sorry

end find_slope_of_line_l214_214217


namespace intersection_of_A_and_B_l214_214347

def A : Set ℝ := { x | x^2 - x > 0 }
def B : Set ℝ := { x | Real.log x / Real.log 2 < 2 }

theorem intersection_of_A_and_B : A ∩ B = { x | 1 < x ∧ x < 4 } :=
by sorry

end intersection_of_A_and_B_l214_214347


namespace packs_needed_l214_214789

def pouches_per_pack : ℕ := 6
def team_members : ℕ := 13
def coaches : ℕ := 3
def helpers : ℕ := 2
def total_people : ℕ := team_members + coaches + helpers

theorem packs_needed (people : ℕ) (pouches_per_pack : ℕ) : ℕ :=
  (people + pouches_per_pack - 1) / pouches_per_pack

example : packs_needed total_people pouches_per_pack = 3 :=
by
  have h1 : total_people = 18 := rfl
  have h2 : pouches_per_pack = 6 := rfl
  rw [h1, h2]
  norm_num
  sorry

end packs_needed_l214_214789


namespace range_of_a_condition_l214_214498

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0

theorem range_of_a_condition :
  range_of_a a → -1 < a ∧ a < 3 := sorry

end range_of_a_condition_l214_214498


namespace width_of_metallic_sheet_l214_214173

theorem width_of_metallic_sheet 
  (length : ℕ)
  (new_volume : ℕ) 
  (side_length_of_square : ℕ)
  (height_of_box : ℕ)
  (new_length : ℕ)
  (new_width : ℕ)
  (w : ℕ) : 
  length = 48 → 
  new_volume = 5120 → 
  side_length_of_square = 8 → 
  height_of_box = 8 → 
  new_length = length - 2 * side_length_of_square → 
  new_width = w - 2 * side_length_of_square → 
  new_volume = new_length * new_width * height_of_box → 
  w = 36 := 
by 
  intros _ _ _ _ _ _ _ 
  sorry

end width_of_metallic_sheet_l214_214173


namespace inverse_proportion_l214_214261

variable {x y x1 x2 y1 y2 : ℝ}
variable {k : ℝ}

theorem inverse_proportion {h1 : x1 ≠ 0} {h2 : x2 ≠ 0} {h3 : y1 ≠ 0} {h4 : y2 ≠ 0}
  (h5 : (∃ k, ∀ (x y : ℝ), x * y = k))
  (h6 : x1 / x2 = 4 / 5) : 
  y1 / y2 = 5 / 4 :=
sorry

end inverse_proportion_l214_214261


namespace subcommitteesWithAtLeastOneTeacher_l214_214405

namespace Subcommittee

def totalCombinations : ℕ := 12.choose 5
def nonTeacherCombinations : ℕ := 7.choose 5

theorem subcommitteesWithAtLeastOneTeacher : totalCombinations - nonTeacherCombinations = 771 :=
by
  sorry

end Subcommittee

end subcommitteesWithAtLeastOneTeacher_l214_214405


namespace min_abs_sum_l214_214755

theorem min_abs_sum (a b c : ℝ) (h₁ : a + b + c = -2) (h₂ : a * b * c = -4) :
  ∃ (m : ℝ), m = min (abs a + abs b + abs c) 6 :=
sorry

end min_abs_sum_l214_214755


namespace tan_of_angle_in_third_quadrant_l214_214483

theorem tan_of_angle_in_third_quadrant 
  (α : ℝ) 
  (h1 : α < -π / 2 ∧ α > -π) 
  (h2 : Real.sin α = -Real.sqrt 5 / 5) :
  Real.tan α = 1 / 2 := 
sorry

end tan_of_angle_in_third_quadrant_l214_214483


namespace find_N_l214_214738

theorem find_N (N : ℕ) (h1 : N > 1)
  (h2 : ∀ (a b c : ℕ), (a ≡ b [MOD N]) ∧ (b ≡ c [MOD N]) ∧ (a ≡ c [MOD N])
  → (a = 1743) ∧ (b = 2019) ∧ (c = 3008)) :
  N = 23 :=
by
  have h3 : Nat.gcd (2019 - 1743) (3008 - 2019) = 23 := sorry
  show N = 23 from sorry

end find_N_l214_214738


namespace smallest_n_leq_l214_214042

theorem smallest_n_leq (n : ℤ) : (n ^ 2 - 13 * n + 40 ≤ 0) → (n = 5) :=
sorry

end smallest_n_leq_l214_214042


namespace expand_and_solve_solve_quadratic_l214_214890

theorem expand_and_solve (x : ℝ) :
  6 * (x - 3) * (x + 5) = 6 * x^2 + 12 * x - 90 :=
by sorry

theorem solve_quadratic (x : ℝ) :
  6 * x^2 + 12 * x - 90 = 0 ↔ x = -5 ∨ x = 3 :=
by sorry

end expand_and_solve_solve_quadratic_l214_214890


namespace remainder_of_product_of_odd_primes_mod_32_l214_214996

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l214_214996


namespace total_time_to_fill_tank_l214_214568

-- Definitions as per conditions
def tank_fill_time_for_one_tap (total_time : ℕ) : Prop :=
  total_time = 16

def number_of_taps_for_second_half (num_taps : ℕ) : Prop :=
  num_taps = 4

-- Theorem statement to prove the total time taken to fill the tank
theorem total_time_to_fill_tank : ∀ (time_one_tap time_total : ℕ),
  tank_fill_time_for_one_tap time_one_tap →
  number_of_taps_for_second_half 4 →
  time_total = 10 :=
by
  intros time_one_tap time_total h1 h2
  -- Proof needed here
  sorry

end total_time_to_fill_tank_l214_214568


namespace mary_final_weight_l214_214244

def initial_weight : Int := 99
def weight_lost_initially : Int := 12
def weight_gained_back_twice_initial : Int := 2 * weight_lost_initially
def weight_lost_thrice_initial : Int := 3 * weight_lost_initially
def weight_gained_back_half_dozen : Int := 12 / 2

theorem mary_final_weight :
  let final_weight := 
      initial_weight 
      - weight_lost_initially 
      + weight_gained_back_twice_initial 
      - weight_lost_thrice_initial 
      + weight_gained_back_half_dozen
  in final_weight = 78 :=
by
  sorry

end mary_final_weight_l214_214244


namespace correct_exp_operation_l214_214845

theorem correct_exp_operation (a b : ℝ) : (-a^3 * b) ^ 2 = a^6 * b^2 :=
  sorry

end correct_exp_operation_l214_214845


namespace smallest_positive_a_integer_root_l214_214731

theorem smallest_positive_a_integer_root :
  ∀ x a : ℚ, (exists x : ℚ, (x > 0) ∧ (a > 0) ∧ 
    (
      ((x - a) / 2 + (x - 2 * a) / 3) / ((x + 4 * a) / 5 - (x + 3 * a) / 4) =
      ((x - 3 * a) / 4 + (x - 4 * a) / 5) / ((x + 2 * a) / 3 - (x + a) / 2)
    )
  ) → a = 419 / 421 :=
by sorry

end smallest_positive_a_integer_root_l214_214731


namespace range_of_a_l214_214088

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 4 → x^2 - 2 * x + 1 - a^2 < 0) ↔ (a < -3 ∨ a > 3) :=
sorry

end range_of_a_l214_214088


namespace find_width_of_metallic_sheet_l214_214175

noncomputable def width_of_metallic_sheet (w : ℝ) : Prop :=
  let length := 48
  let square_side := 8
  let new_length := length - 2 * square_side
  let new_width := w - 2 * square_side
  let height := square_side
  let volume := new_length * new_width * height
  volume = 5120

theorem find_width_of_metallic_sheet (w : ℝ) :
  width_of_metallic_sheet w -> w = 36 := 
sorry

end find_width_of_metallic_sheet_l214_214175


namespace arithmetic_sequence_a13_l214_214786

theorem arithmetic_sequence_a13 (a : ℕ → ℤ) (d : ℤ) (a1 : ℤ) 
  (h1 : a 5 = 3) (h2 : a 9 = 6) 
  (h3 : ∀ n, a n = a1 + (n - 1) * d) : 
  a 13 = 9 :=
sorry

end arithmetic_sequence_a13_l214_214786


namespace ratio_of_fixing_times_is_two_l214_214233

noncomputable def time_per_shirt : ℝ := 1.5
noncomputable def number_of_shirts : ℕ := 10
noncomputable def number_of_pants : ℕ := 12
noncomputable def hourly_rate : ℝ := 30
noncomputable def total_cost : ℝ := 1530

theorem ratio_of_fixing_times_is_two :
  let total_hours := total_cost / hourly_rate
  let shirt_hours := number_of_shirts * time_per_shirt
  let pant_hours := total_hours - shirt_hours
  let time_per_pant := pant_hours / number_of_pants
  (time_per_pant / time_per_shirt) = 2 :=
by
  sorry

end ratio_of_fixing_times_is_two_l214_214233


namespace det_modulo_matrix_l214_214586

noncomputable def matrix_100x100 : Matrix (Fin 100) (Fin 100) ℕ :=
λ i j => (i : ℕ) * (j : ℕ)

theorem det_modulo_matrix :
  (matrix.det matrix_100x100 : ℤ) % 101 = 1 :=
sorry

end det_modulo_matrix_l214_214586


namespace opposite_of_neg2016_l214_214672

theorem opposite_of_neg2016 : -(-2016) = 2016 := 
by 
  sorry

end opposite_of_neg2016_l214_214672


namespace number_of_dogs_l214_214504

theorem number_of_dogs 
  (d c b : Nat) 
  (ratio : d / c / b = 3 / 7 / 12) 
  (total_dogs_and_bunnies : d + b = 375) :
  d = 75 :=
by
  -- Using the hypothesis and given conditions to prove d = 75.
  sorry

end number_of_dogs_l214_214504


namespace probability_heart_and_face_card_club_l214_214152

-- Conditions
def num_cards : ℕ := 52
def num_hearts : ℕ := 13
def num_face_card_clubs : ℕ := 3

-- Define the probabilities
def prob_heart_first : ℚ := num_hearts / num_cards
def prob_face_card_club_given_heart : ℚ := num_face_card_clubs / (num_cards - 1)

-- Proof statement
theorem probability_heart_and_face_card_club :
  prob_heart_first * prob_face_card_club_given_heart = 3 / 204 :=
by
  sorry

end probability_heart_and_face_card_club_l214_214152


namespace width_of_metallic_sheet_l214_214174

theorem width_of_metallic_sheet 
  (length : ℕ)
  (new_volume : ℕ) 
  (side_length_of_square : ℕ)
  (height_of_box : ℕ)
  (new_length : ℕ)
  (new_width : ℕ)
  (w : ℕ) : 
  length = 48 → 
  new_volume = 5120 → 
  side_length_of_square = 8 → 
  height_of_box = 8 → 
  new_length = length - 2 * side_length_of_square → 
  new_width = w - 2 * side_length_of_square → 
  new_volume = new_length * new_width * height_of_box → 
  w = 36 := 
by 
  intros _ _ _ _ _ _ _ 
  sorry

end width_of_metallic_sheet_l214_214174


namespace range_of_x_l214_214222

theorem range_of_x (a : ℝ) (x : ℝ) (h_a : 1 ≤ a ∧ a ≤ 3) (h : a * x^2 + (a - 2) * x - 2 > 0) :
  x < -1 ∨ x > 2 / 3 :=
sorry

end range_of_x_l214_214222


namespace electricity_bill_written_as_decimal_l214_214048

-- Definitions as conditions
def number : ℝ := 71.08

-- Proof statement
theorem electricity_bill_written_as_decimal : number = 71.08 :=
by sorry

end electricity_bill_written_as_decimal_l214_214048


namespace trains_meet_at_distance_360_km_l214_214183

-- Define the speeds of the trains
def speed_A : ℕ := 30 -- speed of train A in kmph
def speed_B : ℕ := 40 -- speed of train B in kmph
def speed_C : ℕ := 60 -- speed of train C in kmph

-- Define the head starts in hours for trains A and B
def head_start_A : ℕ := 9 -- head start for train A in hours
def head_start_B : ℕ := 3 -- head start for train B in hours

-- Define the distances traveled by trains A and B by the time train C starts at 6 p.m.
def distance_A_start : ℕ := speed_A * head_start_A -- distance traveled by train A by 6 p.m.
def distance_B_start : ℕ := speed_B * head_start_B -- distance traveled by train B by 6 p.m.

-- The formula to calculate the distance after t hours from 6 p.m. for each train
def distance_A (t : ℕ) : ℕ := distance_A_start + speed_A * t
def distance_B (t : ℕ) : ℕ := distance_B_start + speed_B * t
def distance_C (t : ℕ) : ℕ := speed_C * t

-- Problem statement to prove the point where all three trains meet
theorem trains_meet_at_distance_360_km : ∃ t : ℕ, distance_A t = 360 ∧ distance_B t = 360 ∧ distance_C t = 360 := by
  sorry

end trains_meet_at_distance_360_km_l214_214183


namespace find_m_l214_214648

-- Define the points M and N and the normal vector n
structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def M (m : ℝ) : Point3D := { x := m, y := -2, z := 1 }
def N (m : ℝ) : Point3D := { x := 0, y := m, z := 3 }
def n : Point3D := { x := 3, y := 1, z := 2 }

-- Define the dot product
def dot_product (v1 v2 : Point3D) : ℝ :=
  (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z)

-- Define the vector MN
def MN (m : ℝ) : Point3D := { x := -(m), y := m + 2, z := 2 }

-- Prove the dot product condition is zero implies m = 3
theorem find_m (m : ℝ) (h : dot_product n (MN m) = 0) : m = 3 :=
by
  sorry

end find_m_l214_214648


namespace krish_spent_on_sweets_l214_214126

noncomputable def initial_amount := 200.50
noncomputable def amount_per_friend := 25.20
noncomputable def remaining_amount := 114.85

noncomputable def total_given_to_friends := amount_per_friend * 2
noncomputable def amount_before_sweets := initial_amount - total_given_to_friends
noncomputable def amount_spent_on_sweets := amount_before_sweets - remaining_amount

theorem krish_spent_on_sweets : amount_spent_on_sweets = 35.25 :=
by
  sorry

end krish_spent_on_sweets_l214_214126


namespace line_common_chord_eq_l214_214773

theorem line_common_chord_eq (a b : ℝ) :
  (∀ (x1 x2 y1 y2 : ℝ), x1^2 + y1^2 = 1 → (x2 - a)^2 + (y2 - b)^2 = 1 → 
    2 * a * x2 + 2 * b * y2 - 3 = 0) :=
sorry

end line_common_chord_eq_l214_214773


namespace max_students_l214_214637

theorem max_students (A B C : ℕ) (A_left B_left C_left : ℕ)
  (hA : A = 38) (hB : B = 78) (hC : C = 128)
  (hA_left : A_left = 2) (hB_left : B_left = 6) (hC_left : C_left = 20) :
  gcd (A - A_left) (gcd (B - B_left) (C - C_left)) = 36 :=
by {
  sorry
}

end max_students_l214_214637


namespace train_length_correct_l214_214060

noncomputable def length_of_train (speed_train_kmph : ℕ) (time_to_cross_bridge_sec : ℝ) (length_of_bridge_m : ℝ) : ℝ :=
let speed_train_mps := (speed_train_kmph : ℝ) * (1000 / 3600)
let total_distance := speed_train_mps * time_to_cross_bridge_sec
total_distance - length_of_bridge_m

theorem train_length_correct :
  length_of_train 90 32.99736021118311 660 = 164.9340052795778 :=
by
  have speed_train_mps : ℝ := 90 * (1000 / 3600)
  have total_distance := speed_train_mps * 32.99736021118311
  have length_of_train := total_distance - 660
  exact sorry

end train_length_correct_l214_214060


namespace product_of_odd_primes_mod_32_l214_214954

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l214_214954


namespace product_of_odd_primes_mod_32_l214_214952

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l214_214952


namespace new_person_weight_l214_214162

theorem new_person_weight (N : ℝ) (h : N - 65 = 22.5) : N = 87.5 :=
by
  sorry

end new_person_weight_l214_214162


namespace remainder_when_M_divided_by_32_l214_214943

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end remainder_when_M_divided_by_32_l214_214943


namespace slope_range_l214_214374

open Real

theorem slope_range (k : ℝ) :
  (∃ b : ℝ, 
    ∃ x1 x2 x3 : ℝ,
      (x1 + x2 + x3 = 0) ∧
      (x1 ≥ 0) ∧ (x2 ≥ 0) ∧ (x3 < 0) ∧
      ((kx1 + b) = ((x1 + 1) / (|x1| + 1))) ∧
      ((kx2 + b) = ((x2 + 1) / (|x2| + 1))) ∧
      ((kx3 + b) = ((x3 + 1) / (|x3| + 1)))) →
  (0 < k ∧ k < (2 / 9)) :=
sorry

end slope_range_l214_214374


namespace measure_of_angle_H_in_parallelogram_l214_214506

theorem measure_of_angle_H_in_parallelogram (H : Type) [AddGroup H] {E F G : H}
  (h_parallelogram : parallelogram E F G H)
  (angle_F : measure_of_angle F = 125) :
  measure_of_angle H = 55 := 
sorry

end measure_of_angle_H_in_parallelogram_l214_214506


namespace find_N_l214_214736

theorem find_N :
  ∃ N : ℕ, N > 1 ∧
    (1743 % N = 2019 % N) ∧ (2019 % N = 3008 % N) ∧ N = 23 :=
by
  sorry

end find_N_l214_214736


namespace inequality_le_one_equality_case_l214_214569

open Real

theorem inequality_le_one (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 1) :
    (ab / (a^5 + b^5 + ab) + bc / (b^5 + c^5 + bc) + ca / (c^5 + a^5 + ca) ≤ 1) :=
sorry

theorem equality_case (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 1) :
    (ab / (a^5 + b^5 + ab) + bc / (b^5 + c^5 + bc) + ca / (c^5 + a^5 + ca) = 1) ↔ (a = 1 ∧ b = 1 ∧ c = 1) :=
sorry

end inequality_le_one_equality_case_l214_214569


namespace atomic_weight_Br_correct_l214_214462

def atomic_weight_Ba : ℝ := 137.33
def molecular_weight_compound : ℝ := 297
def atomic_weight_Br : ℝ := 79.835

theorem atomic_weight_Br_correct :
  molecular_weight_compound = atomic_weight_Ba + 2 * atomic_weight_Br :=
by
  sorry

end atomic_weight_Br_correct_l214_214462


namespace age_difference_is_16_l214_214137

variable (y : ℕ)  -- the present age of the younger person

-- Conditions
def elder_age_now : ℕ := 30
def elder_age_6_years_ago := elder_age_now - 6
def younger_age_6_years_ago := y - 6
def condition := elder_age_6_years_ago = 3 * younger_age_6_years_ago

-- Theorem to prove the difference in ages is 16 years
theorem age_difference_is_16 (h : condition y) : elder_age_now - y = 16 :=
by
  sorry

end age_difference_is_16_l214_214137


namespace minimize_product_l214_214208

theorem minimize_product
    (a b c : ℕ) 
    (h_positive: a > 0 ∧ b > 0 ∧ c > 0)
    (h_eq: 10 * a^2 - 3 * a * b + 7 * c^2 = 0) : 
    (gcd a b) * (gcd b c) * (gcd c a) = 3 :=
sorry

end minimize_product_l214_214208


namespace sum_of_first_15_squares_l214_214833

noncomputable def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_first_15_squares :
  sum_of_squares 15 = 1240 :=
by
  sorry

end sum_of_first_15_squares_l214_214833


namespace candy_vs_chocolate_l214_214871

theorem candy_vs_chocolate
  (candy1 candy2 chocolate : ℕ)
  (h1 : candy1 = 38)
  (h2 : candy2 = 36)
  (h3 : chocolate = 16) :
  (candy1 + candy2) - chocolate = 58 :=
by
  sorry

end candy_vs_chocolate_l214_214871


namespace natural_number_sum_of_coprimes_l214_214004

theorem natural_number_sum_of_coprimes (n : ℕ) (h : n ≥ 2) : ∃ a b : ℕ, n = a + b ∧ Nat.gcd a b = 1 :=
by
  use (n - 1), 1
  sorry

end natural_number_sum_of_coprimes_l214_214004


namespace mr_brown_class_problem_l214_214920

theorem mr_brown_class_problem :
  ∀ (total_students boys girls : ℕ), 
  boys = 3 * (total_students / 7) ∧ 
  girls = 4 * (total_students / 7) ∧ 
  total_students = 56 -> 
  (boys : ℚ) / total_students = 42.86 / 100 ∧ 
  girls = 32 :=
by
  sorry

end mr_brown_class_problem_l214_214920


namespace nat_know_albums_l214_214588

/-- Define the number of novels, comics, documentaries and crates properties --/
def novels := 145
def comics := 271
def documentaries := 419
def crates := 116
def items_per_crate := 9

/-- Define the total capacity of crates --/
def total_capacity := crates * items_per_crate

/-- Define the total number of other items --/
def other_items := novels + comics + documentaries

/-- Define the number of albums --/
def albums := total_capacity - other_items

/-- Theorem: Prove that the number of albums is equal to 209 --/
theorem nat_know_albums : albums = 209 := by
  sorry

end nat_know_albums_l214_214588


namespace rectangle_sides_l214_214278

theorem rectangle_sides (x y : ℕ) (h_diff : x ≠ y) (h_eq : x * y = 2 * x + 2 * y) : 
  (x = 3 ∧ y = 6) ∨ (x = 6 ∧ y = 3) :=
sorry

end rectangle_sides_l214_214278


namespace certain_number_is_213_l214_214220

theorem certain_number_is_213 (x : ℝ) (h1 : x * 16 = 3408) (h2 : x * 1.6 = 340.8) : x = 213 :=
sorry

end certain_number_is_213_l214_214220


namespace tangents_parallel_l214_214328

variable {R : Type*} [Field R]

-- Let f be a function from ratios to slopes
variable (φ : R -> R)

-- Given points (x, y) and (x₁, y₁) with corresponding conditions
variable (x x₁ y y₁ : R)

-- Conditions
def corresponding_points := y / x = y₁ / x₁
def homogeneous_diff_eqn := ∀ x y, (y / x) = φ (y / x)

-- Prove that the tangents are parallel
theorem tangents_parallel (h_corr : corresponding_points x x₁ y y₁)
  (h_diff_eqn : ∀ (x x₁ y y₁ : R), y' = φ (y / x) ∧ y₁' = φ (y₁ / x₁)) :
  y' = y₁' :=
by
  sorry

end tangents_parallel_l214_214328


namespace find_vector_at_t_zero_l214_214433

def vector_at_t (a d : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (a.1 + t*d.1, a.2 + t*d.2)

theorem find_vector_at_t_zero :
  ∃ (a d : ℝ × ℝ),
    vector_at_t a d 1 = (2, 3) ∧
    vector_at_t a d 4 = (8, -5) ∧
    vector_at_t a d 5 = (10, -9) ∧
    vector_at_t a d 0 = (0, 17/3) :=
by
  sorry

end find_vector_at_t_zero_l214_214433


namespace combination_20_choose_3_eq_1140_l214_214644

theorem combination_20_choose_3_eq_1140 :
  (Nat.choose 20 3) = 1140 := 
by sorry

end combination_20_choose_3_eq_1140_l214_214644


namespace rocket_altitude_time_l214_214135

theorem rocket_altitude_time (a₁ d : ℕ) (n : ℕ) (h₁ : a₁ = 2) (h₂ : d = 2)
  (h₃ : n * a₁ + (n * (n - 1) * d) / 2 = 240) : n = 15 :=
by
  -- The proof is ignored as per instruction.
  sorry

end rocket_altitude_time_l214_214135


namespace rahul_matches_played_l214_214701

theorem rahul_matches_played
  (current_avg runs_today new_avg : ℕ)
  (h1 : current_avg = 51)
  (h2 : runs_today = 69)
  (h3 : new_avg = 54)
  : ∃ m : ℕ, ((51 * m + 69) / (m + 1) = 54) ∧ (m = 5) :=
by
  sorry

end rahul_matches_played_l214_214701


namespace remainder_when_divided_by_32_l214_214963

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l214_214963


namespace remainder_of_M_when_divided_by_32_l214_214940

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l214_214940


namespace trigonometric_identity_l214_214494

theorem trigonometric_identity
  (α : Real)
  (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 :=
by
  sorry

end trigonometric_identity_l214_214494


namespace negation_of_proposition_l214_214827

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x ≥ 1 → x^2 ≥ 1)) ↔ (∃ x : ℝ, x ≥ 1 ∧ x^2 < 1) := 
sorry

end negation_of_proposition_l214_214827


namespace parallelogram_angle_H_l214_214507

theorem parallelogram_angle_H (F H : ℝ) (h1 : F = 125) (h2 : F + H = 180) : H = 55 :=
by
  have h3 : H = 180 - F := by linarith
  rw [h1] at h3
  rw [h3]
  norm_num

end parallelogram_angle_H_l214_214507


namespace interval_of_monotonic_decrease_minimum_value_in_interval_l214_214620

noncomputable def f (x a : ℝ) : ℝ := 1 / x + a * Real.log x

-- Define the derivative of f
noncomputable def f_prime (x a : ℝ) : ℝ := (a * x - 1) / x^2

-- Prove that the interval of monotonic decrease is as specified
theorem interval_of_monotonic_decrease (a : ℝ) :
  if a ≤ 0 then ∀ x ∈ Set.Ioi (0 : ℝ), f_prime x a < 0
  else ∀ x ∈ Set.Ioo 0 (1/a), f_prime x a < 0 := sorry

-- Prove that, given x in [1/2, 1], the minimum value of f(x) is 0 when a = 2 / log 2
theorem minimum_value_in_interval :
  ∃ a : ℝ, (a = 2 / Real.log 2) ∧ ∀ x ∈ Set.Icc (1/2 : ℝ) 1, f x a ≥ 0 ∧ (∃ y ∈ Set.Icc (1/2 : ℝ) 1, f y a = 0) := sorry

end interval_of_monotonic_decrease_minimum_value_in_interval_l214_214620


namespace solution_set_of_inequality_l214_214678

theorem solution_set_of_inequality : {x : ℝ // |x - 2| > x - 2} = {x : ℝ // x < 2} :=
sorry

end solution_set_of_inequality_l214_214678


namespace find_point_A_l214_214246

theorem find_point_A (x : ℝ) (h : x + 7 - 4 = 0) : x = -3 :=
sorry

end find_point_A_l214_214246


namespace xy_extrema_l214_214021

noncomputable def xy_product (a : ℝ) : ℝ := a^2 - 1

theorem xy_extrema (x y a : ℝ) 
  (h1 : x + y = a) 
  (h2 : x^2 + y^2 = -a^2 + 2) : 
  -1 ≤ xy_product a ∧ xy_product a ≤ 1/3 :=
by
  sorry

end xy_extrema_l214_214021


namespace subcommittee_combinations_l214_214052

open Nat

theorem subcommittee_combinations :
  (choose 8 3) * (choose 6 2) = 840 := by
  sorry

end subcommittee_combinations_l214_214052


namespace smallest_natural_number_l214_214149

theorem smallest_natural_number :
  ∃ n : ℕ, (n > 0) ∧ (7 * n % 10000 = 2012) ∧ ∀ m : ℕ, (7 * m % 10000 = 2012) → (n ≤ m) :=
sorry

end smallest_natural_number_l214_214149


namespace correct_option_C_l214_214696

theorem correct_option_C (m n : ℤ) : 
  (4 * m + 1) * 2 * m = 8 * m^2 + 2 * m :=
by
  sorry

end correct_option_C_l214_214696


namespace remainder_9_plus_y_mod_31_l214_214653

theorem remainder_9_plus_y_mod_31 (y : ℕ) (hy : 7 * y ≡ 1 [MOD 31]) : (9 + y) % 31 = 18 :=
sorry

end remainder_9_plus_y_mod_31_l214_214653


namespace correct_operation_B_l214_214418

theorem correct_operation_B (x : ℝ) : 
  x - 2 * x = -x :=
sorry

end correct_operation_B_l214_214418


namespace meaningful_sqrt_l214_214102

theorem meaningful_sqrt (x : ℝ) (h : x - 5 ≥ 0) : x = 6 :=
sorry

end meaningful_sqrt_l214_214102


namespace parking_spaces_in_the_back_l214_214143

theorem parking_spaces_in_the_back
  (front_spaces : ℕ)
  (cars_parked : ℕ)
  (half_back_filled : ℕ → ℚ)
  (spaces_available : ℕ)
  (B : ℕ)
  (h1 : front_spaces = 52)
  (h2 : cars_parked = 39)
  (h3 : half_back_filled B = B / 2)
  (h4 : spaces_available = 32) :
  B = 38 :=
by
  -- Here you can provide the proof steps.
  sorry

end parking_spaces_in_the_back_l214_214143


namespace find_x_when_y_equals_2_l214_214913

theorem find_x_when_y_equals_2 (x : ℚ) (y : ℚ) : 
  y = (1 / (4 * x + 2)) ∧ y = 2 -> x = -3 / 8 := 
by 
  sorry

end find_x_when_y_equals_2_l214_214913


namespace apples_to_eat_raw_l214_214655

/-- Proof of the number of apples left to eat raw given the conditions -/
theorem apples_to_eat_raw 
  (total_apples : ℕ)
  (pct_wormy : ℕ)
  (pct_moldy : ℕ)
  (wormy_apples_offset : ℕ)
  (wormy_apples bruised_apples moldy_apples apples_left : ℕ) 
  (h1 : total_apples = 120)
  (h2 : pct_wormy = 20)
  (h3 : pct_moldy = 30)
  (h4 : wormy_apples = pct_wormy * total_apples / 100)
  (h5 : moldy_apples = pct_moldy * total_apples / 100)
  (h6 : bruised_apples = wormy_apples + wormy_apples_offset)
  (h7 : wormy_apples_offset = 9)
  (h8 : apples_left = total_apples - (wormy_apples + moldy_apples + bruised_apples))
  : apples_left = 27 :=
sorry

end apples_to_eat_raw_l214_214655


namespace solve_for_x_l214_214256

theorem solve_for_x (x : ℝ) : (5 * x - 2) / (6 * x - 6) = 3 / 4 ↔ x = -5 := by
  sorry

end solve_for_x_l214_214256


namespace irreducible_fraction_l214_214651

theorem irreducible_fraction (n : ℤ) : Int.gcd (3 * n + 10) (4 * n + 13) = 1 := 
sorry

end irreducible_fraction_l214_214651


namespace remainder_when_divided_by_32_l214_214964

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l214_214964


namespace count_integers_with_at_most_three_digits_l214_214360

/-- The number of positive integers less than 50,000 with at most three different digits is 7503. -/
theorem count_integers_with_at_most_three_digits : 
  (finset.filter (λ n : ℕ, n < 50000 ∧ (finset.card (finset.image (λ d : ℕ, (n / 10^d) % 10) (finset.range 5)) ≤ 3)) (finset.range 50000)).card = 7503 := 
sorry

end count_integers_with_at_most_three_digits_l214_214360


namespace prime_product_mod_32_l214_214971

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l214_214971


namespace vector_dot_product_value_l214_214115

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 4)
def c : ℝ × ℝ := (3, 2)

def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem vector_dot_product_value : dot_product (add (scalar_mul 2 a) b) c = -3 := by
  sorry

end vector_dot_product_value_l214_214115


namespace total_artworks_l214_214123

theorem total_artworks (students : ℕ) (group1_artworks : ℕ) (group2_artworks : ℕ) (total_students : students = 10) 
    (artwork_group1 : group1_artworks = 5 * 3) (artwork_group2 : group2_artworks = 5 * 4) : 
    group1_artworks + group2_artworks = 35 :=
by
  sorry

end total_artworks_l214_214123


namespace sqrt_combination_l214_214437

theorem sqrt_combination : 
    ∃ (k : ℝ), (k * Real.sqrt 2 = Real.sqrt 8) ∧ 
    (¬(∃ (k : ℝ), (k * Real.sqrt 2 = Real.sqrt 3))) ∧ 
    (¬(∃ (k : ℝ), (k * Real.sqrt 2 = Real.sqrt 12))) ∧ 
    (¬(∃ (k : ℝ), (k * Real.sqrt 2 = Real.sqrt 0.2))) :=
by
  sorry

end sqrt_combination_l214_214437


namespace gcd_360_128_is_8_l214_214563

def gcd_360_128 : ℕ :=
  gcd 360 128

theorem gcd_360_128_is_8 : gcd_360_128 = 8 :=
  by
    -- Proof goes here (use sorry for now)
    sorry

end gcd_360_128_is_8_l214_214563


namespace solution_set_of_abs_inequality_l214_214027

theorem solution_set_of_abs_inequality : 
  {x : ℝ | abs (x - 1) - abs (x - 5) < 2} = {x : ℝ | x < 4} := 
by 
  sorry

end solution_set_of_abs_inequality_l214_214027


namespace prime_product_mod_32_l214_214977

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l214_214977


namespace length_of_first_platform_l214_214059

theorem length_of_first_platform 
  (t1 t2 : ℝ) 
  (length_train : ℝ) 
  (length_second_platform : ℝ) 
  (time1 : ℝ) 
  (time2 : ℝ) 
  (speed_eq : (t1 + length_train) / time1 = (length_second_platform + length_train) / time2) 
  (time1_eq : time1 = 15) 
  (time2_eq : time2 = 20) 
  (length_train_eq : length_train = 100) 
  (length_second_platform_eq: length_second_platform = 500) :
  t1 = 350 := 
  by 
  sorry

end length_of_first_platform_l214_214059


namespace sale_in_fifth_month_l214_214171

theorem sale_in_fifth_month (sale1 sale2 sale3 sale4 sale6 avg_sale num_months total_sales known_sales_five_months sale5: ℕ) :
  sale1 = 6400 →
  sale2 = 7000 →
  sale3 = 6800 →
  sale4 = 7200 →
  sale6 = 5100 →
  avg_sale = 6500 →
  num_months = 6 →
  total_sales = avg_sale * num_months →
  known_sales_five_months = sale1 + sale2 + sale3 + sale4 + sale6 →
  sale5 = total_sales - known_sales_five_months →
  sale5 = 6500 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end sale_in_fifth_month_l214_214171


namespace gcd_1213_1985_eq_1_l214_214195

theorem gcd_1213_1985_eq_1
  (h1: ¬ (1213 % 2 = 0))
  (h2: ¬ (1213 % 3 = 0))
  (h3: ¬ (1213 % 5 = 0))
  (h4: ¬ (1985 % 2 = 0))
  (h5: ¬ (1985 % 3 = 0))
  (h6: ¬ (1985 % 5 = 0)):
  Nat.gcd 1213 1985 = 1 := by
  sorry

end gcd_1213_1985_eq_1_l214_214195


namespace solve_for_x_l214_214012

theorem solve_for_x (x : ℤ) : 3 * (5 - x) = 9 → x = 2 :=
by {
  sorry
}

end solve_for_x_l214_214012


namespace tan_sum_trig_identity_l214_214643

variable {A B C : ℝ} -- Angles
variable {a b c : ℝ} -- Sides opposite to angles A, B and C

-- Acute triangle implies A, B, C are all less than π/2 and greater than 0
variable (hAcute : 0 < A ∧ A < pi / 2 ∧ 0 < B ∧ B < pi / 2 ∧ 0 < C ∧ C < pi / 2)

-- Given condition in the problem
variable (hCondition : b / a + a / b = 6 * Real.cos C)

theorem tan_sum_trig_identity : 
  Real.tan C / Real.tan A + Real.tan C / Real.tan B = 4 :=
sorry

end tan_sum_trig_identity_l214_214643


namespace numerator_denominator_added_l214_214694

theorem numerator_denominator_added (n : ℕ) : (3 + n) / (5 + n) = 9 / 11 → n = 6 :=
by
  sorry

end numerator_denominator_added_l214_214694


namespace smallest_b_l214_214006

theorem smallest_b (a b : ℝ) (h1 : 2 < a) (h2 : a < b) 
(h3 : 2 + a ≤ b) (h4 : 1 / a + 1 / b ≤ 2) : b = 2 :=
sorry

end smallest_b_l214_214006


namespace triangle_angle_B_l214_214108

theorem triangle_angle_B (A B C : ℕ) (h₁ : B + C = 110) (h₂ : A + B + C = 180) (h₃ : A = 70) :
  B = 70 ∨ B = 55 ∨ B = 40 :=
by
  sorry

end triangle_angle_B_l214_214108


namespace lizard_eyes_fewer_than_spots_and_wrinkles_l214_214377

noncomputable def lizard_problem : Nat :=
  let eyes_jan := 3
  let wrinkles_jan := 3 * eyes_jan
  let spots_jan := 7 * (wrinkles_jan ^ 2)
  let eyes_cousin := 3
  let wrinkles_cousin := 2 * eyes_cousin
  let spots_cousin := 5 * (wrinkles_cousin ^ 2)
  let total_eyes := eyes_jan + eyes_cousin
  let total_wrinkles := wrinkles_jan + wrinkles_cousin
  let total_spots := spots_jan + spots_cousin
  (total_spots + total_wrinkles) - total_eyes

theorem lizard_eyes_fewer_than_spots_and_wrinkles :
  lizard_problem = 756 :=
by
  sorry

end lizard_eyes_fewer_than_spots_and_wrinkles_l214_214377


namespace shop_width_correct_l214_214547

-- Definition of the shop's monthly rent
def monthly_rent : ℝ := 2400

-- Definition of the shop's length in feet
def shop_length : ℝ := 10

-- Definition of the annual rent per square foot
def annual_rent_per_sq_ft : ℝ := 360

-- The mathematical assertion that the width of the shop is 8 feet
theorem shop_width_correct (width : ℝ) :
  (monthly_rent * 12) / annual_rent_per_sq_ft / shop_length = width :=
by
  sorry

end shop_width_correct_l214_214547


namespace area_of_circle_above_below_lines_l214_214323

noncomputable def circle_area : ℝ :=
  40 * Real.pi

theorem area_of_circle_above_below_lines :
  ∃ (x y : ℝ), (x^2 + y^2 - 16*x - 8*y = 0) ∧ (y > x - 4) ∧ (y < -x + 4) ∧
  (circle_area = 40 * Real.pi) :=
  sorry

end area_of_circle_above_below_lines_l214_214323


namespace root_of_polynomial_l214_214674

theorem root_of_polynomial :
  ∀ x : ℝ, (x^2 - 3 * x + 2) * x * (x - 4) = 0 ↔ (x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 4) :=
by 
  sorry

end root_of_polynomial_l214_214674


namespace decreasing_sufficient_condition_l214_214473

theorem decreasing_sufficient_condition {a : ℝ} (h_pos : 0 < a) (h_neq_one : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^x > a^y) →
  (∀ x y : ℝ, x < y → (a-2)*x^3 > (a-2)*y^3) :=
by
  sorry

end decreasing_sufficient_condition_l214_214473


namespace primes_between_30_and_60_l214_214717

theorem primes_between_30_and_60 (list_of_primes : List ℕ) 
  (H1 : list_of_primes = [31, 37, 41, 43, 47, 53, 59]) :
  (list_of_primes.headI * list_of_primes.reverse.headI) = 1829 := by
  sorry

end primes_between_30_and_60_l214_214717


namespace year_when_P_costs_40_paise_more_than_Q_l214_214281

def price_of_P (n : ℕ) : ℝ := 4.20 + 0.40 * n
def price_of_Q (n : ℕ) : ℝ := 6.30 + 0.15 * n

theorem year_when_P_costs_40_paise_more_than_Q :
  ∃ n : ℕ, price_of_P n = price_of_Q n + 0.40 ∧ 2001 + n = 2011 :=
by
  sorry

end year_when_P_costs_40_paise_more_than_Q_l214_214281


namespace mary_final_weight_l214_214242

theorem mary_final_weight : 
  let initial_weight := 99
  let weight_loss1 := 12
  let weight_gain1 := 2 * weight_loss1
  let weight_loss2 := 3 * weight_loss1
  let weight_gain2 := 6
  initial_weight - weight_loss1 + weight_gain1 - weight_loss2 + weight_gain2 = 81 := by 
  sorry

end mary_final_weight_l214_214242


namespace extremum_range_a_l214_214916

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - a * x^2 + x

theorem extremum_range_a :
  (∀ x : ℝ, -1 < x ∧ x < 0 → (f a x = 0 → ∃ x0 : ℝ, f a x0 = 0 ∧ -1 < x0 ∧ x0 < 0)) →
  a < -1/5 ∨ a = -1 :=
sorry

end extremum_range_a_l214_214916


namespace minimum_value_l214_214090

noncomputable def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem minimum_value (a b c d : ℝ) (h1 : a < (2 / 3) * b) 
  (h2 : ∀ x, 3 * a * x^2 + 2 * b * x + c ≥ 0) : 
  ∃ (x : ℝ), ∀ c, 2 * b - 3 * a ≠ 0 → (c = (b^2 / 3 / a)) → (c / (2 * b - 3 * a) ≥ 1) :=
by
  sorry

end minimum_value_l214_214090


namespace correct_operation_l214_214566

theorem correct_operation :
  (3 * a^3 - 2 * a^3 = a^3) ∧ ¬(m - 4 * m = -3) ∧ ¬(a^2 * b - a * b^2 = 0) ∧ ¬(2 * x + 3 * x = 5 * x^2) :=
by
  sorry

end correct_operation_l214_214566


namespace total_balloons_l214_214080

theorem total_balloons (fred_balloons sam_balloons dan_balloons : ℕ) 
  (h1 : fred_balloons = 10) 
  (h2 : sam_balloons = 46) 
  (h3 : dan_balloons = 16) 
  (total : fred_balloons + sam_balloons + dan_balloons = 72) :
  fred_balloons + sam_balloons + dan_balloons = 72 := 
sorry

end total_balloons_l214_214080


namespace gcd_840_1764_l214_214826

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l214_214826


namespace production_analysis_l214_214308

def daily_change (day: ℕ) : ℤ :=
  match day with
  | 0 => 40    -- Monday
  | 1 => -30   -- Tuesday
  | 2 => 90    -- Wednesday
  | 3 => -50   -- Thursday
  | 4 => -20   -- Friday
  | 5 => -10   -- Saturday
  | 6 => 20    -- Sunday
  | _ => 0     -- Invalid day, just in case

def planned_daily_production : ℤ := 500

def actual_production (day: ℕ) : ℤ :=
  planned_daily_production + (List.sum (List.map daily_change (List.range (day + 1))))

def total_production : ℤ :=
  List.sum (List.map actual_production (List.range 7))

theorem production_analysis :
  ∃ largest_increase_day smallest_increase_day : ℕ,
    largest_increase_day = 2 ∧  -- Wednesday
    smallest_increase_day = 1 ∧  -- Tuesday
    total_production = 3790 ∧
    total_production > 7 * planned_daily_production := by
  sorry

end production_analysis_l214_214308


namespace students_algebra_or_drafting_not_both_not_geography_l214_214892

variables (A D G : Finset ℕ)
-- Condition 1: Fifteen students are taking both algebra and drafting
variable (h1 : (A ∩ D).card = 15)
-- Condition 2: There are 30 students taking algebra
variable (h2 : A.card = 30)
-- Condition 3: There are 12 students taking drafting only
variable (h3 : (D \ A).card = 12)
-- Condition 4: There are eight students taking a geography class
variable (h4 : G.card = 8)
-- Condition 5: Two students are also taking both algebra and drafting and geography
variable (h5 : ((A ∩ D) ∩ G).card = 2)

-- Question: Prove the final count of students taking algebra or drafting but not both, and not taking geography is 25
theorem students_algebra_or_drafting_not_both_not_geography :
  ((A \ D) ∪ (D \ A)).card - ((A ∩ D) ∩ G).card = 25 :=
by
  sorry

end students_algebra_or_drafting_not_both_not_geography_l214_214892


namespace gcd_180_450_l214_214604

theorem gcd_180_450 : gcd 180 450 = 90 :=
by sorry

end gcd_180_450_l214_214604


namespace boxes_of_apples_l214_214410

theorem boxes_of_apples (n_crates apples_per_crate rotten_apples apples_per_box : ℕ) 
  (h1 : n_crates = 12) 
  (h2 : apples_per_crate = 42)
  (h3: rotten_apples = 4) 
  (h4 : apples_per_box = 10) : 
  (n_crates * apples_per_crate - rotten_apples) / apples_per_box = 50 :=
by
  sorry

end boxes_of_apples_l214_214410


namespace largest_k_dividing_A_l214_214197

def A : ℤ := 1990^(1991^1992) + 1991^(1990^1992) + 1992^(1991^1990)

theorem largest_k_dividing_A :
  1991^(1991) ∣ A := sorry

end largest_k_dividing_A_l214_214197


namespace total_artworks_l214_214124

theorem total_artworks (students art_kits : ℕ)
    (students_per_kit : ℕ → ℕ → ℕ) 
    (artworks_group1 : ℕ → ℕ) 
    (artworks_group2 : ℕ → ℕ)
    (h_students : students = 10)
    (h_art_kits : art_kits = 20)
    (h_students_per_kit : students_per_kit students art_kits = 2)
    (h_group1_size : (students / 2) = 5)
    (h_group2_size : (students / 2) = 5)
    (h_artworks_group1 : (5 * 3) = artworks_group1 5)
    (h_artworks_group2 : (5 * 4) = artworks_group2 5)
    : (artworks_group1 5 + artworks_group2 5) = 35 := 
by 
  rw [h_students, h_art_kits, h_students_per_kit, h_group1_size, h_group2_size, h_artworks_group1, h_artworks_group2]
  sorry

end total_artworks_l214_214124


namespace smallest_integer_sum_consecutive_l214_214406

theorem smallest_integer_sum_consecutive
  (l m n a : ℤ)
  (h1 : a = 9 * l + 36)
  (h2 : a = 10 * m + 45)
  (h3 : a = 11 * n + 55)
  : a = 495 :=
sorry

end smallest_integer_sum_consecutive_l214_214406


namespace distance_between_points_l214_214459

def distance_on_line (a b : ℝ) : ℝ := |b - a|

theorem distance_between_points (a b : ℝ) : distance_on_line a b = |b - a| :=
by sorry

end distance_between_points_l214_214459


namespace cistern_height_l214_214706

theorem cistern_height (l w A : ℝ) (h : ℝ) (hl : l = 8) (hw : w = 6) (hA : 48 + 2 * (l * h) + 2 * (w * h) = 99.8) : h = 1.85 := by
  sorry

end cistern_height_l214_214706


namespace average_bowling_score_l214_214907

-- Definitions of the scores
def g : ℕ := 120
def m : ℕ := 113
def b : ℕ := 85

-- Theorem statement: The average score is 106
theorem average_bowling_score : (g + m + b) / 3 = 106 := by
  sorry

end average_bowling_score_l214_214907


namespace ratio_of_expenditure_l214_214673

variable (A B AE BE : ℕ)

theorem ratio_of_expenditure (h1 : A = 2000) 
    (h2 : A / B = 5 / 4) 
    (h3 : A - AE = 800) 
    (h4: B - BE = 800) :
    AE / BE = 3 / 2 := by
  sorry

end ratio_of_expenditure_l214_214673


namespace rowing_speed_l214_214387

theorem rowing_speed :
  ∀ (initial_width final_width increase_per_10m : ℝ) (time_seconds : ℝ)
  (yards_to_meters : ℝ → ℝ) (width_increase_in_yards : ℝ) (distance_10m_segments : ℝ) 
  (total_distance : ℝ),
  initial_width = 50 →
  final_width = 80 →
  increase_per_10m = 2 →
  time_seconds = 30 →
  yards_to_meters 1 = 0.9144 →
  width_increase_in_yards = (final_width - initial_width) →
  width_increase_in_yards * (yards_to_meters 1) = 27.432 →
  distance_10m_segments = (width_increase_in_yards * (yards_to_meters 1)) / 10 →
  total_distance = distance_10m_segments * 10 →
  (total_distance / time_seconds) = 0.9144 :=
by
  intros initial_width final_width increase_per_10m time_seconds yards_to_meters 
        width_increase_in_yards distance_10m_segments total_distance
  sorry

end rowing_speed_l214_214387


namespace boxes_of_apples_l214_214409

theorem boxes_of_apples (n_crates apples_per_crate rotten_apples apples_per_box : ℕ) 
  (h1 : n_crates = 12) 
  (h2 : apples_per_crate = 42)
  (h3: rotten_apples = 4) 
  (h4 : apples_per_box = 10) : 
  (n_crates * apples_per_crate - rotten_apples) / apples_per_box = 50 :=
by
  sorry

end boxes_of_apples_l214_214409


namespace product_of_odd_primes_mod_32_l214_214960

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l214_214960


namespace primes_sum_product_condition_l214_214624

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem primes_sum_product_condition (m n p : ℕ) (hm : is_prime m) (hn : is_prime n) (hp : is_prime p)  
  (h : m * n * p = 5 * (m + n + p)) : 
  m^2 + n^2 + p^2 = 78 :=
sorry

end primes_sum_product_condition_l214_214624


namespace inverse_proportion_m_range_l214_214609

theorem inverse_proportion_m_range (m : ℝ) :
  (∀ x : ℝ, x < 0 → ∀ y1 y2 : ℝ, y1 = (1 - 2 * m) / x → y2 = (1 - 2 * m) / (x + 1) → y1 < y2) 
  ↔ (m > 1 / 2) :=
by sorry

end inverse_proportion_m_range_l214_214609


namespace complement_of_A_l214_214485

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≤ -3} ∪ {x | x ≥ 0}

theorem complement_of_A :
  U \ A = {x | -3 < x ∧ x < 0} :=
sorry

end complement_of_A_l214_214485


namespace monkey_count_l214_214431

theorem monkey_count (piles_1 piles_2 hands_1 hands_2 bananas_1_per_hand bananas_2_per_hand total_bananas_per_monkey : ℕ) 
  (h1 : piles_1 = 6) 
  (h2 : piles_2 = 4) 
  (h3 : hands_1 = 9) 
  (h4 : hands_2 = 12) 
  (h5 : bananas_1_per_hand = 14) 
  (h6 : bananas_2_per_hand = 9) 
  (h7 : total_bananas_per_monkey = 99) : 
  (piles_1 * hands_1 * bananas_1_per_hand + piles_2 * hands_2 * bananas_2_per_hand) / total_bananas_per_monkey = 12 := 
by 
  sorry

end monkey_count_l214_214431


namespace kimiko_watched_4_videos_l214_214929

/-- Kimiko's videos. --/
def first_video_length := 120
def second_video_length := 270
def last_two_video_length := 60
def total_time_watched := 510

theorem kimiko_watched_4_videos :
  first_video_length + second_video_length + last_two_video_length + last_two_video_length = total_time_watched → 
  4 = 4 :=
by
  intro h
  sorry

end kimiko_watched_4_videos_l214_214929


namespace polynomial_degree_rational_coefficients_l214_214262

theorem polynomial_degree_rational_coefficients :
  ∃ p : Polynomial ℚ,
    (Polynomial.aeval (2 - 3 * Real.sqrt 3) p = 0) ∧
    (Polynomial.aeval (-2 - 3 * Real.sqrt 3) p = 0) ∧
    (Polynomial.aeval (3 + Real.sqrt 11) p = 0) ∧
    (Polynomial.aeval (3 - Real.sqrt 11) p = 0) ∧
    p.degree = 6 :=
sorry

end polynomial_degree_rational_coefficients_l214_214262


namespace smallest_integral_value_k_l214_214693

-- Define the discriminant of the quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := 3 * x * (k * x - 5) - x^2 + 4

-- Define the condition for the quadratic equation having no real roots
def no_real_roots (k : ℝ) : Prop :=
  let a := 3 * k - 1
  let b := -15
  let c := 4
  discriminant a b c < 0

-- The Lean 4 statement to find the smallest integral value of k such that the quadratic has no real roots
theorem smallest_integral_value_k : ∃ (k : ℤ), no_real_roots k ∧ (∀ (m : ℤ), no_real_roots m → k ≤ m) :=
  sorry

end smallest_integral_value_k_l214_214693


namespace value_of_expression_l214_214776

theorem value_of_expression (x : ℝ) (h : x = 1 + Real.sqrt 2) : x^4 - 4 * x^3 + 4 * x^2 + 4 = 5 := by
  sorry

end value_of_expression_l214_214776


namespace value_of_k_l214_214918

theorem value_of_k (x y k : ℤ) (h1 : 2 * x - y = 5 * k + 6) (h2 : 4 * x + 7 * y = k) (h3 : x + y = 2024)
: k = 2023 :=
by
  sorry

end value_of_k_l214_214918


namespace triangle_is_right_l214_214508

-- Definitions based on the conditions given in the problem
variables {a b c A B C : ℝ}

-- Introduction of the conditions in Lean
def is_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180 ∧
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)

def given_condition (A b c : ℝ) : Prop :=
  (Real.cos (A / 2))^2 = (b + c) / (2 * c)

-- Theorem statement to prove the conclusion based on given conditions
theorem triangle_is_right (a b c A B C : ℝ) 
  (h_triangle : is_triangle a b c A B C)
  (h_given : given_condition A b c) :
  A = 90 := sorry

end triangle_is_right_l214_214508


namespace Peter_total_distance_l214_214252

theorem Peter_total_distance 
  (total_time : ℝ) 
  (speed1 speed2 fraction1 fraction2 : ℝ) 
  (h_time : total_time = 1.4) 
  (h_speed1 : speed1 = 4) 
  (h_speed2 : speed2 = 5) 
  (h_fraction1 : fraction1 = 2/3) 
  (h_fraction2 : fraction2 = 1/3) 
  (D : ℝ) : 
  (fraction1 * D / speed1 + fraction2 * D / speed2 = total_time) → D = 6 :=
by
  intros h_eq
  sorry

end Peter_total_distance_l214_214252


namespace least_positive_integer_remainder_l214_214039

theorem least_positive_integer_remainder :
  ∃ n : ℕ, (n > 0) ∧ (n % 5 = 1) ∧ (n % 4 = 2) ∧ (∀ m : ℕ, (m > 0) ∧ (m % 5 = 1) ∧ (m % 4 = 2) → n ≤ m) :=
sorry

end least_positive_integer_remainder_l214_214039


namespace stratified_sampling_third_grade_l214_214180

theorem stratified_sampling_third_grade 
  (N : ℕ) (N3 : ℕ) (S : ℕ) (x : ℕ)
  (h1 : N = 1600)
  (h2 : N3 = 400)
  (h3 : S = 80)
  (h4 : N3 / N = x / S) :
  x = 20 := 
by {
  sorry
}

end stratified_sampling_third_grade_l214_214180


namespace prime_product_mod_32_l214_214979

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l214_214979


namespace remainder_of_M_l214_214986

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l214_214986


namespace inequality_solution_function_min_value_l214_214381

theorem inequality_solution (a : ℕ) (h₁ : abs ((3 / 2 : ℚ) - 2) < a) (h₂ : abs ((1 / 2 : ℚ) - 2) ≥ a) : a = 1 := 
by
  -- proof omitted
  sorry

theorem function_min_value (a : ℕ) (h₁ : abs ((3 / 2 : ℚ) - 2) < a) (h₂ : abs ((1 / 2 : ℚ) - 2) ≥ a)
  (h₃ : a = 1) : ∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → (abs (x + a) + abs (x - 2)) = 3 :=
by
  -- proof omitted
  use 0
  -- proof omitted
  sorry

end inequality_solution_function_min_value_l214_214381


namespace maximum_profit_is_achieved_at_14_yuan_l214_214575

-- Define the initial conditions
def cost_per_unit : ℕ := 8
def initial_selling_price : ℕ := 10
def initial_selling_quantity : ℕ := 100

-- Define the sales volume decrease per price increase
def decrease_per_yuan_increase : ℕ := 10

-- Define the profit function
def profit (price_increase : ℕ) : ℕ :=
  let new_selling_price := initial_selling_price + price_increase
  let new_selling_quantity := initial_selling_quantity - (decrease_per_yuan_increase * price_increase)
  (new_selling_price - cost_per_unit) * new_selling_quantity

-- Define the statement to be proved
theorem maximum_profit_is_achieved_at_14_yuan :
  ∃ price_increase : ℕ, price_increase = 4 ∧ profit price_increase = profit 4 := by
  sorry

end maximum_profit_is_achieved_at_14_yuan_l214_214575


namespace no_triangle_tangent_l214_214905

open Real

/-- Given conditions --/
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def C2 (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1 ∧ a > b ∧ b > 0 ∧ (1 / a^2) + (1 / b^2) = 1

theorem no_triangle_tangent (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : (1 : ℝ) / a^2 + 1 / b^2 = 1) :
  ¬∃ (A B C : ℝ × ℝ), 
    (C1 A.1 A.2) ∧ (C1 B.1 B.2) ∧ (C1 C.1 C.2) ∧
    (∃ (l : ℝ) (m : ℝ) (n : ℝ), C2 l m a b ∧ C2 n l a b) :=
by
  sorry

end no_triangle_tangent_l214_214905


namespace range_of_a_l214_214766

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x > 1 then a^x else (4 - a/2)*x + 2

theorem range_of_a (a : ℝ) :
  (∀ x1 x2, x1 < x2 → f a x1 ≤ f a x2) ↔ (4 ≤ a ∧ a < 8) :=
by
  sorry

end range_of_a_l214_214766


namespace xyz_sum_l214_214912

theorem xyz_sum (x y z : ℝ) 
  (h1 : y + z = 17 - 2 * x) 
  (h2 : x + z = 1 - 2 * y) 
  (h3 : x + y = 8 - 2 * z) : 
  x + y + z = 6.5 :=
sorry

end xyz_sum_l214_214912


namespace white_balls_count_l214_214307

theorem white_balls_count (total_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ) (prob_neither_red_nor_purple : ℝ) (W : ℕ)
    (h_total : total_balls = 100)
    (h_green : green_balls = 30)
    (h_yellow : yellow_balls = 10)
    (h_red : red_balls = 37)
    (h_purple : purple_balls = 3)
    (h_prob : prob_neither_red_nor_purple = 0.6)
    (h_computation : W = total_balls * prob_neither_red_nor_purple - (green_balls + yellow_balls)) :
    W = 20 := 
sorry

end white_balls_count_l214_214307


namespace f_zero_eq_zero_f_one_eq_one_f_n_is_n_l214_214326

variable (f : ℤ → ℤ)

axiom functional_eq : ∀ m n : ℤ, f (m^2 + f n) = f (f m) + n

theorem f_zero_eq_zero : f 0 = 0 :=
sorry

theorem f_one_eq_one : f 1 = 1 :=
sorry

theorem f_n_is_n : ∀ n : ℤ, f n = n :=
sorry

end f_zero_eq_zero_f_one_eq_one_f_n_is_n_l214_214326


namespace average_effective_increase_correct_l214_214033

noncomputable def effective_increase (initial_price: ℕ) (price_increase_percent: ℕ) (discount_percent: ℕ) : ℕ :=
let increased_price := initial_price + (initial_price * price_increase_percent / 100)
let final_price := increased_price - (increased_price * discount_percent / 100)
(final_price - initial_price) * 100 / initial_price

noncomputable def average_effective_increase : ℕ :=
let increase1 := effective_increase 300 10 5
let increase2 := effective_increase 450 15 7
let increase3 := effective_increase 600 20 10
(increase1 + increase2 + increase3) / 3

theorem average_effective_increase_correct :
  average_effective_increase = 6483 / 100 :=
by
  sorry

end average_effective_increase_correct_l214_214033


namespace product_of_odd_primes_mod_32_l214_214953

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l214_214953


namespace crossing_time_l214_214574

-- Define the conditions
def walking_speed_kmh : Float := 10
def bridge_length_m : Float := 1666.6666666666665

-- Convert the man's walking speed to meters per minute
def walking_speed_mpm : Float := walking_speed_kmh * (1000 / 60)

-- State the theorem we want to prove
theorem crossing_time 
  (ws_kmh : Float := walking_speed_kmh)
  (bl_m : Float := bridge_length_m)
  (ws_mpm : Float := walking_speed_mpm) :
  bl_m / ws_mpm = 10 :=
by
  sorry

end crossing_time_l214_214574


namespace yellow_tiled_area_is_correct_l214_214423

noncomputable def length : ℝ := 3.6
noncomputable def width : ℝ := 2.5 * length
noncomputable def total_area : ℝ := length * width
noncomputable def yellow_tiled_area : ℝ := total_area / 2

theorem yellow_tiled_area_is_correct (length_eq : length = 3.6)
    (width_eq : width = 2.5 * length)
    (total_area_eq : total_area = length * width)
    (yellow_area_eq : yellow_tiled_area = total_area / 2) :
    yellow_tiled_area = 16.2 := 
by sorry

end yellow_tiled_area_is_correct_l214_214423


namespace cost_of_72_tulips_is_115_20_l214_214869

/-
Conditions:
1. A package containing 18 tulips costs $36.
2. The price of a package is directly proportional to the number of tulips it contains.
3. There is a 20% discount applied for packages containing more than 50 tulips.
Question:
What is the cost of 72 tulips?

Correct answer:
$115.20
-/

def costOfTulips (numTulips : ℕ)  : ℚ :=
  if numTulips ≤ 50 then
    36 * numTulips / 18
  else
    (36 * numTulips / 18) * 0.8 -- apply 20% discount for more than 50 tulips

theorem cost_of_72_tulips_is_115_20 :
  costOfTulips 72 = 115.2 := 
sorry

end cost_of_72_tulips_is_115_20_l214_214869


namespace discount_is_25_percent_l214_214141

noncomputable def discount_percentage (M : ℝ) (C : ℝ) (SP : ℝ) : ℝ :=
  ((M - SP) / M) * 100

theorem discount_is_25_percent (M : ℝ) (C : ℝ) (SP : ℝ) 
  (h1 : C = 0.64 * M) 
  (h2 : SP = C * 1.171875) : 
  discount_percentage M C SP = 25 := 
by 
  sorry

end discount_is_25_percent_l214_214141


namespace packing_peanuts_per_large_order_l214_214290

/-- Definitions of conditions as stated -/
def large_orders : ℕ := 3
def small_orders : ℕ := 4
def total_peanuts_used : ℕ := 800
def peanuts_per_small : ℕ := 50

/-- The statement to prove, ensuring all conditions are utilized in the definitions -/
theorem packing_peanuts_per_large_order : 
  ∃ L, large_orders * L + small_orders * peanuts_per_small = total_peanuts_used ∧ L = 200 := 
by
  use 200
  -- Adding the necessary proof steps
  have h1 : large_orders = 3 := rfl
  have h2 : small_orders = 4 := rfl
  have h3 : peanuts_per_small = 50 := rfl
  have h4 : total_peanuts_used = 800 := rfl
  sorry

end packing_peanuts_per_large_order_l214_214290


namespace total_dolphins_correct_l214_214365

-- Define the initial number of dolphins
def initialDolphins : Nat := 65

-- Define the multiplier for the dolphins joining from the river
def joiningMultiplier : Nat := 3

-- Define the total number of dolphins after joining
def totalDolphins : Nat := initialDolphins + (joiningMultiplier * initialDolphins)

-- Prove that the total number of dolphins is 260
theorem total_dolphins_correct : totalDolphins = 260 := by
  sorry

end total_dolphins_correct_l214_214365


namespace distinct_prime_sum_product_l214_214555

open Nat

-- Definitions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- The problem statement
theorem distinct_prime_sum_product (a b c : ℕ) (h1 : is_prime a) (h2 : is_prime b) 
    (h3 : is_prime c) (h4 : a ≠ 1) (h5 : b ≠ 1) (h6 : c ≠ 1) 
    (h7 : a ≠ b) (h8 : b ≠ c) (h9 : a ≠ c) : 

    1994 + a + b + c = a * b * c :=
sorry

end distinct_prime_sum_product_l214_214555


namespace sqrt_of_second_number_l214_214153

-- Given condition: the arithmetic square root of a natural number n is x
variable (x : ℕ)
def first_number := x ^ 2
def second_number := first_number + 1

-- The theorem statement we want to prove
theorem sqrt_of_second_number (x : ℕ) : Real.sqrt (x^2 + 1) = Real.sqrt (first_number x + 1) :=
by
  sorry

end sqrt_of_second_number_l214_214153


namespace sum_of_squares_divisible_by_sum_l214_214499

theorem sum_of_squares_divisible_by_sum (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
    (h_bound : a < 2017 ∧ b < 2017 ∧ c < 2017)
    (h_mod : (a^3 - b^3) % 2017 = 0 ∧ (b^3 - c^3) % 2017 = 0 ∧ (c^3 - a^3) % 2017 = 0) :
    (a^2 + b^2 + c^2) % (a + b + c) = 0 :=
by
  sorry

end sum_of_squares_divisible_by_sum_l214_214499


namespace rationalize_denominator_l214_214254

theorem rationalize_denominator :
  let a := (∛5 : ℝ)
  let b := (∛3 : ℝ)
  let expr := 1 / (a - b)
  let num := a^2 + a*b + b^2
  ∃ A B C D : ℝ,
    expr * num = (∛A + ∛B + ∛C) / D ∧
    A = 25 ∧ B = 15 ∧ C = 9 ∧ D = 2 ∧
    A + B + C + D = 51 :=
by {
  let a := (∛5 : ℝ)
  let b := (∛3 : ℝ)
  let expr := 1 / (a - b)
  let num := a^2 + a*b + b^2

  exists (25 : ℝ)
  exists (15 : ℝ)
  exists (9 : ℝ)
  exists (2 : ℝ)

  split
  { sorry, }
  { split
    { exact rfl, }
    { split
      { exact rfl, }
      { split
        { exact rfl, }
        { split
          { exact rfl, }
          { norm_num }}}}
}

end rationalize_denominator_l214_214254


namespace find_teacher_age_l214_214263

theorem find_teacher_age (S T : ℕ) (h1 : S / 19 = 20) (h2 : (S + T) / 20 = 21) : T = 40 :=
sorry

end find_teacher_age_l214_214263


namespace remainder_of_product_of_odd_primes_mod_32_l214_214994

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l214_214994


namespace factorize_poly_l214_214334

theorem factorize_poly (x : ℝ) : 4 * x^3 - x = x * (2 * x + 1) * (2 * x - 1) := by
  sorry

end factorize_poly_l214_214334


namespace smallest_three_digit_number_multiple_of_conditions_l214_214181

theorem smallest_three_digit_number_multiple_of_conditions :
  ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧
  (x % 2 = 0) ∧ ((x + 1) % 3 = 0) ∧ ((x + 2) % 4 = 0) ∧ ((x + 3) % 5 = 0) ∧ ((x + 4) % 6 = 0) 
  ∧ x = 122 := 
by
  sorry

end smallest_three_digit_number_multiple_of_conditions_l214_214181


namespace total_dolphins_correct_l214_214364

-- Define the initial number of dolphins
def initialDolphins : Nat := 65

-- Define the multiplier for the dolphins joining from the river
def joiningMultiplier : Nat := 3

-- Define the total number of dolphins after joining
def totalDolphins : Nat := initialDolphins + (joiningMultiplier * initialDolphins)

-- Prove that the total number of dolphins is 260
theorem total_dolphins_correct : totalDolphins = 260 := by
  sorry

end total_dolphins_correct_l214_214364


namespace gcd_180_450_l214_214605

theorem gcd_180_450 : gcd 180 450 = 90 :=
by sorry

end gcd_180_450_l214_214605


namespace sphere_circumscribed_around_cone_radius_l214_214199

-- Definitions of the given conditions
variable (r h : ℝ)

-- Theorem statement (without the proof)
theorem sphere_circumscribed_around_cone_radius :
  ∃ R : ℝ, R = (Real.sqrt (r^2 + h^2)) / 2 :=
sorry

end sphere_circumscribed_around_cone_radius_l214_214199


namespace age_conversion_in_base_l214_214442

theorem age_conversion_in_base (n : ℕ) (h : n = 7231) :
  Nat.ofDigits 16 [9, 9, 14] = Nat.ofDigits 8 [7, 2, 3, 1] :=
by
  have h_eq : 7231 = Nat.ofDigits 8 [7, 2, 3, 1] := rfl
  rw [h, h_eq]
  sorry

end age_conversion_in_base_l214_214442


namespace minimum_value_of_a_l214_214621

noncomputable def inequality_valid_for_all_x (a : ℝ) : Prop :=
  ∀ (x : ℝ), 1 < x → x + a * Real.log x - x^a + 1 / Real.exp x ≥ 0

theorem minimum_value_of_a : ∃ a, inequality_valid_for_all_x a ∧ a = -Real.exp 1 := sorry

end minimum_value_of_a_l214_214621


namespace sequence_formula_correct_l214_214468

noncomputable def S (n : ℕ) : ℕ := 2^n - 3

def a (n : ℕ) : ℤ :=
  if n = 1 then -1
  else 2^(n-1)

theorem sequence_formula_correct (n : ℕ) :
  a n = (if n = 1 then -1 else 2^(n-1)) :=
by
  sorry

end sequence_formula_correct_l214_214468


namespace margie_change_l214_214383

def cost_of_banana_cents : ℕ := 30
def cost_of_orange_cents : ℕ := 60
def num_bananas : ℕ := 4
def num_oranges : ℕ := 2
def amount_paid_dollars : ℝ := 10.0

noncomputable def cost_of_banana_dollars := (cost_of_banana_cents : ℝ) / 100
noncomputable def cost_of_orange_dollars := (cost_of_orange_cents : ℝ) / 100

noncomputable def total_cost := 
  (num_bananas * cost_of_banana_dollars) + (num_oranges * cost_of_orange_dollars)

noncomputable def change_received := amount_paid_dollars - total_cost

theorem margie_change : change_received = 7.60 := 
by sorry

end margie_change_l214_214383


namespace larry_correct_evaluation_l214_214656

theorem larry_correct_evaluation (a b c d e : ℝ) 
(Ha : a = 5) (Hb : b = 3) (Hc : c = 6) (Hd : d = 4) :
a - b + c + d - e = a - (b - (c + (d - e))) → e = 0 :=
by
  -- Not providing the actual proof
  sorry

end larry_correct_evaluation_l214_214656


namespace product_divisible_by_12_l214_214901

theorem product_divisible_by_12 (a b c d : ℤ) :
  12 ∣ (b - a) * (c - a) * (d - a) * (d - c) * (d - b) * (c - b) := 
by {
  sorry
}

end product_divisible_by_12_l214_214901


namespace prime_integer_roots_l214_214746

theorem prime_integer_roots (p : ℕ) (hp : Prime p) 
  (hroots : ∀ (x1 x2 : ℤ), x1 * x2 = -512 * p ∧ x1 + x2 = -p) : p = 2 :=
by
  -- Proof omitted
  sorry

end prime_integer_roots_l214_214746


namespace isosceles_triangle_angle_sum_l214_214719

theorem isosceles_triangle_angle_sum (y : ℕ) (a : ℕ) (b : ℕ) 
  (h_isosceles : a = b ∨ a = y ∨ b = y)
  (h_sum : a + b + y = 180) :
  a = 80 → b = 80 → y = 50 ∨ y = 20 ∨ y = 80 → y + y + y = 150 :=
by
  sorry

end isosceles_triangle_angle_sum_l214_214719


namespace gcd_180_450_l214_214606

theorem gcd_180_450 : gcd 180 450 = 90 :=
by sorry

end gcd_180_450_l214_214606


namespace focus_of_parabola_l214_214330

theorem focus_of_parabola (x y : ℝ) : (y^2 = 4 * x) → (x = 2 ∧ y = 0) :=
by
  sorry

end focus_of_parabola_l214_214330


namespace coefficient_of_x3_l214_214497

theorem coefficient_of_x3 (n : ℕ) (h1 : ∀ n, (∃ r, 3 * r - n = 0 ∧ nat.binomial n r * (-1)^r = 15) → n = 6) :
  (∀ r, 3 * r - 6 = 3 → r = 3) → - nat.binomial 6 3 = -20 :=
by
  intro h2
  specialize h2 3
  sorry -- Proof is omitted as per the instructions

end coefficient_of_x3_l214_214497


namespace how_many_pairs_of_shoes_l214_214733

theorem how_many_pairs_of_shoes (l k : ℕ) (h_l : l = 52) (h_k : k = 2) : l / k = 26 := by
  sorry

end how_many_pairs_of_shoes_l214_214733


namespace circle_problem_l214_214903

theorem circle_problem 
  (x y : ℝ)
  (h : x^2 + 8*x - 10*y = 10 - y^2 + 6*x) :
  let a := -1
  let b := 5
  let r := 6
  a + b + r = 10 :=
by sorry

end circle_problem_l214_214903


namespace greatest_triangle_perimeter_l214_214781

theorem greatest_triangle_perimeter :
  ∃ (x : ℕ), 3 < x ∧ x < 6 ∧ max (x + 4 * x + 17) (5 + 4 * 5 + 17) = 42 :=
by
  sorry

end greatest_triangle_perimeter_l214_214781


namespace chef_initial_eggs_l214_214393

-- Define the conditions
def eggs_in_fridge := 10
def eggs_per_cake := 5
def cakes_made := 10

-- Prove that the number of initial eggs is 60
theorem chef_initial_eggs : (eggs_per_cake * cakes_made + eggs_in_fridge) = 60 :=
by
  sorry

end chef_initial_eggs_l214_214393


namespace number_of_raccoons_l214_214793

/-- Jason pepper-sprays some raccoons and 6 times as many squirrels. 
Given that he pepper-sprays a total of 84 animals, the number of raccoons he pepper-sprays is 12. -/
theorem number_of_raccoons (R : Nat) (h1 : 84 = R + 6 * R) : R = 12 :=
by
  sorry

end number_of_raccoons_l214_214793


namespace inequality_problem_l214_214652

open Real

theorem inequality_problem
  (a b c x y z : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z)
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
  (h_condition : 1 / x + 1 / y + 1 / z = 1) :
  a^x + b^y + c^z ≥ 4 * a * b * c * x * y * z / (x + y + z - 3) ^ 2 :=
by
  sorry

end inequality_problem_l214_214652


namespace angle_B_l214_214212

/-- 
  Given that the area of triangle ABC is (sqrt 3 / 2) 
  and the dot product of vectors AB and BC is 3, 
  prove that the measure of angle B is 5π/6. 
--/
theorem angle_B (A B C : ℝ) (a c : ℝ) (h1 : 0 ≤ B ∧ B ≤ π)
  (h_area : (1 / 2) * a * c * (Real.sin B) = (Real.sqrt 3 / 2))
  (h_dot : a * c * (Real.cos B) = -3) :
  B = 5 * Real.pi / 6 :=
sorry

end angle_B_l214_214212


namespace ab_value_l214_214495

theorem ab_value (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 29) : a * b = 2 :=
by
  -- proof will be provided here
  sorry

end ab_value_l214_214495


namespace option_D_is_correct_option_A_is_incorrect_option_B_is_incorrect_option_C_is_incorrect_l214_214300

variable (a b x : ℝ)

theorem option_D_is_correct :
  (2 * x + 1) * (x - 2) = 2 * x^2 - 3 * x - 2 :=
by sorry

theorem option_A_is_incorrect :
  2 * a^2 * b * 3 * a^2 * b^2 ≠ 6 * a^6 * b^3 :=
by sorry

theorem option_B_is_incorrect :
  0.00076 ≠ 7.6 * 10^4 :=
by sorry

theorem option_C_is_incorrect :
  -2 * a * (a + b) ≠ -2 * a^2 + 2 * a * b :=
by sorry

end option_D_is_correct_option_A_is_incorrect_option_B_is_incorrect_option_C_is_incorrect_l214_214300


namespace tan_beta_half_l214_214206

theorem tan_beta_half (α β : ℝ)
    (h1 : Real.tan α = 1 / 3)
    (h2 : Real.sin β = 2 * Real.cos (α + β) * Real.sin α) : 
    Real.tan β = 1 / 2 := 
sorry

end tan_beta_half_l214_214206


namespace number_of_true_propositions_is_2_l214_214452

-- Definitions for the propositions
def original_proposition (x : ℝ) : Prop := x > -3 → x > -6
def converse_proposition (x : ℝ) : Prop := x > -6 → x > -3
def inverse_proposition (x : ℝ) : Prop := x ≤ -3 → x ≤ -6
def contrapositive_proposition (x : ℝ) : Prop := x ≤ -6 → x ≤ -3

-- The theorem we need to prove
theorem number_of_true_propositions_is_2 :
  (∀ x, original_proposition x) ∧ (∀ x, contrapositive_proposition x) ∧ 
  ¬ (∀ x, converse_proposition x) ∧ ¬ (∀ x, inverse_proposition x) → 2 = 2 := 
sorry

end number_of_true_propositions_is_2_l214_214452


namespace sum_of_c_and_d_l214_214239

theorem sum_of_c_and_d (c d : ℝ) :
  (∀ x : ℝ, x ≠ 2 → x ≠ -3 → (x - 2) * (x + 3) = x^2 + c * x + d) →
  c + d = -5 :=
by
  intros h
  sorry

end sum_of_c_and_d_l214_214239


namespace Reeta_pencils_l214_214447

-- Let R be the number of pencils Reeta has
variable (R : ℕ)

-- Condition 1: Anika has 4 more than twice the number of pencils as Reeta
def Anika_pencils := 2 * R + 4

-- Condition 2: Together, Anika and Reeta have 64 pencils
def combined_pencils := R + Anika_pencils R

theorem Reeta_pencils (h : combined_pencils R = 64) : R = 20 :=
by
  sorry

end Reeta_pencils_l214_214447


namespace good_carrots_l214_214338

theorem good_carrots (Faye_picked : ℕ) (Mom_picked : ℕ) (bad_carrots : ℕ)
    (total_carrots : Faye_picked + Mom_picked = 28)
    (bad_carrots_count : bad_carrots = 16) : 
    28 - bad_carrots = 12 := by
  -- Proof goes here
  sorry

end good_carrots_l214_214338


namespace odd_number_diff_of_squares_l214_214010

theorem odd_number_diff_of_squares (k : ℕ) : ∃ n : ℕ, k = (n+1)^2 - n^2 ↔ ∃ m : ℕ, k = 2 * m + 1 := 
by 
  sorry

end odd_number_diff_of_squares_l214_214010


namespace last_digit_of_7_power_7_power_7_l214_214691

theorem last_digit_of_7_power_7_power_7 : (7 ^ (7 ^ 7)) % 10 = 3 :=
by
  sorry

end last_digit_of_7_power_7_power_7_l214_214691


namespace math_problem_l214_214482

-- Define functions f and g
def f (x : ℝ) : ℝ := x^2 - 3 * x + 6
def g (x : ℝ) : ℝ := 2 * x + 4

-- State the theorem
theorem math_problem : f(g(3)) - g(f(3)) = 60 := by
  sorry

end math_problem_l214_214482


namespace three_lines_intersect_single_point_l214_214542

theorem three_lines_intersect_single_point (a : ℝ) :
  (∀ x y : ℝ, (x + 2*y + a) * (x^2 - y^2) = 0) ↔ a = 0 := by
  sorry

end three_lines_intersect_single_point_l214_214542


namespace probability_seat_7_l214_214250

open ProbabilityTheory

noncomputable def probability_last_passenger_seat (n : ℕ) : ℝ :=
if h: n = 1 then 1 else 1 / (n:ℝ)

theorem probability_seat_7 : probability_last_passenger_seat 7 = 1 / 7 := by
  sorry

end probability_seat_7_l214_214250


namespace Karlson_max_candies_l214_214147

theorem Karlson_max_candies (f : Fin 25 → ℕ) (g : Fin 25 → Fin 25 → ℕ) :
  (∀ i, f i = 1) →
  (∀ i j, g i j = f i * f j) →
  (∃ (S : ℕ), S = 300) :=
by
  intros h1 h2
  sorry

end Karlson_max_candies_l214_214147


namespace integers_with_product_72_and_difference_4_have_sum_20_l214_214549

theorem integers_with_product_72_and_difference_4_have_sum_20 :
  ∃ (x y : ℕ), (x * y = 72) ∧ (x - y = 4) ∧ (x + y = 20) :=
sorry

end integers_with_product_72_and_difference_4_have_sum_20_l214_214549


namespace a_minus_b_7_l214_214395

theorem a_minus_b_7 (a b : ℤ) : (2 * y + a) * (y + b) = 2 * y^2 - 5 * y - 12 → a - b = 7 :=
by
  sorry

end a_minus_b_7_l214_214395


namespace fraction_girls_at_dance_l214_214325

theorem fraction_girls_at_dance :
  let total_students_colfax := 300
  let boys_to_girls_ratio_colfax := (3, 2)
  let total_students_winthrop := 200
  let boys_to_girls_ratio_winthrop := (3, 4)
  
  let total_students_dance := total_students_colfax + total_students_winthrop
  let girls_colfax := (boys_to_girls_ratio_colfax.snd * total_students_colfax) /
                       (boys_to_girls_ratio_colfax.fst + boys_to_girls_ratio_colfax.snd)
  let girls_winthrop := (boys_to_girls_ratio_winthrop.snd * total_students_winthrop) /
                         (boys_to_girls_ratio_winthrop.fst + boys_to_girls_ratio_winthrop.snd)
  let total_girls := girls_colfax + girls_winthrop
  (total_girls / total_students_dance) = (328 : ℚ) / 700 := by
  sorry

end fraction_girls_at_dance_l214_214325


namespace exists_rectangular_parallelepiped_with_equal_surface_area_and_edge_sum_l214_214068

theorem exists_rectangular_parallelepiped_with_equal_surface_area_and_edge_sum :
  ∃ (a b c : ℤ), 2 * (a * b + b * c + c * a) = 4 * (a + b + c) :=
by
  -- Here we prove the existence of such integers a, b, c, which is stated in the theorem
  sorry

end exists_rectangular_parallelepiped_with_equal_surface_area_and_edge_sum_l214_214068


namespace sufficient_but_not_necessary_l214_214754

-- Definitions of propositions p and q
def p (a b m : ℝ) : Prop := a * m^2 < b * m^2
def q (a b : ℝ) : Prop := a < b

-- Problem statement as a Lean theorem
theorem sufficient_but_not_necessary (a b m : ℝ) : 
  (p a b m → q a b) ∧ (¬ (q a b → p a b m)) :=
by
  sorry

end sufficient_but_not_necessary_l214_214754


namespace abs_diff_squares_l214_214293

theorem abs_diff_squares (a b : ℤ) (ha : a = 103) (hb : b = 97) : |a^2 - b^2| = 1200 :=
by
  sorry

end abs_diff_squares_l214_214293


namespace pie_eating_contest_l214_214683

theorem pie_eating_contest:
  let pie1 := 4/5
  let pie2 := 5/6
  let pie3 := 3/4
  max pie1 (max pie2 pie3) - min pie1 (min pie2 pie3) = 1/12 := by
    sorry

end pie_eating_contest_l214_214683


namespace remainder_of_M_when_divided_by_32_l214_214932

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l214_214932


namespace at_least_one_head_probability_l214_214324

open ProbabilityTheory

noncomputable def probability_at_least_one_head := 
  let p_tails := (1 / 2) ^ 4  -- Probability of getting four tails
  let p_at_least_one_head := 1 - p_tails -- Probability of getting at least one head
  p_at_least_one_head

theorem at_least_one_head_probability : probability_at_least_one_head = (15 / 16) := by
  sorry

end at_least_one_head_probability_l214_214324


namespace al_sandwich_combinations_l214_214543

def types_of_bread : ℕ := 5
def types_of_meat : ℕ := 6
def types_of_cheese : ℕ := 5

def restricted_turkey_swiss_combinations : ℕ := 5
def restricted_white_chicken_combinations : ℕ := 5
def restricted_rye_turkey_combinations : ℕ := 5

def total_sandwich_combinations : ℕ := types_of_bread * types_of_meat * types_of_cheese

def valid_sandwich_combinations : ℕ :=
  total_sandwich_combinations - restricted_turkey_swiss_combinations
  - restricted_white_chicken_combinations - restricted_rye_turkey_combinations

theorem al_sandwich_combinations : valid_sandwich_combinations = 135 := 
  by
  sorry

end al_sandwich_combinations_l214_214543


namespace prove_Praveen_present_age_l214_214814

-- Definitions based on the conditions identified in a)
def PraveenAge (P : ℝ) := P + 10 = 3 * (P - 3)

-- The equivalent proof problem statement
theorem prove_Praveen_present_age : ∃ P : ℝ, PraveenAge P ∧ P = 9.5 :=
by
  sorry

end prove_Praveen_present_age_l214_214814


namespace house_orderings_l214_214226

/-- Ralph walks past five houses each painted in a different color: 
orange, red, blue, yellow, and green.
Conditions:
1. Ralph passed the orange house before the red house.
2. Ralph passed the blue house before the yellow house.
3. The blue house was not next to the yellow house.
4. Ralph passed the green house before the red house and after the blue house.
Given these conditions, prove that there are exactly 3 valid orderings of the houses.
-/
theorem house_orderings : 
  ∃ (orderings : Finset (List String)), 
  orderings.card = 3 ∧
  (∀ (o : List String), 
   o ∈ orderings ↔ 
    ∃ (idx_o idx_r idx_b idx_y idx_g : ℕ), 
    o = ["orange", "red", "blue", "yellow", "green"] ∧
    idx_o < idx_r ∧ 
    idx_b < idx_y ∧ 
    (idx_b + 1 < idx_y ∨ idx_y + 1 < idx_b) ∧ 
    idx_b < idx_g ∧ idx_g < idx_r) := sorry

end house_orderings_l214_214226


namespace runner_overtake_time_l214_214682

theorem runner_overtake_time
  (L : ℝ)
  (v1 v2 v3 : ℝ)
  (h1 : v1 = v2 + L / 6)
  (h2 : v1 = v3 + L / 10) :
  L / (v3 - v2) = 15 := by
  sorry

end runner_overtake_time_l214_214682


namespace correct_operation_l214_214420

theorem correct_operation (a b : ℤ) : -3 * (a - b) = -3 * a + 3 * b := 
sorry

end correct_operation_l214_214420


namespace part1_part2_l214_214615

-- Conditions: Definitions of A and B
def A (a b : ℝ) : ℝ := 2 * a^2 - 5 * a * b + 3 * b
def B (a b : ℝ) : ℝ := 4 * a^2 - 6 * a * b - 8 * a

-- Theorem statements
theorem part1 (a b : ℝ) :  2 * A a b - B a b = -4 * a * b + 6 * b + 8 * a := sorry

theorem part2 (a : ℝ) (h : ∀ a, 2 * A a 2 - B a 2 = - 4 * a * 2 + 6 * 2 + 8 * a) : 2 = 2 := sorry

end part1_part2_l214_214615


namespace remainder_of_M_when_divided_by_32_l214_214931

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l214_214931


namespace max_value_2ab_sqrt2_plus_2ac_l214_214804

theorem max_value_2ab_sqrt2_plus_2ac (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a^2 + b^2 + c^2 = 1) : 
  2 * a * b * Real.sqrt 2 + 2 * a * c ≤ 1 :=
sorry

end max_value_2ab_sqrt2_plus_2ac_l214_214804


namespace linear_regression_passes_through_centroid_l214_214279

noncomputable def linear_regression (a b : ℝ) (x : ℝ) : ℝ := a + b * x

theorem linear_regression_passes_through_centroid 
  (a b : ℝ) (x_bar y_bar : ℝ) 
  (h_centroid : ∀ (x y : ℝ), (x = x_bar ∧ y = y_bar) → y = linear_regression a b x) :
  linear_regression a b x_bar = y_bar :=
by
  -- proof omitted
  sorry

end linear_regression_passes_through_centroid_l214_214279


namespace height_of_bottom_step_l214_214289

variable (h l w : ℝ)

theorem height_of_bottom_step
  (h l w : ℝ)
  (eq1 : l + h - w / 2 = 42)
  (eq2 : 2 * l + h = 38)
  (w_value : w = 4) : h = 34 := by
sorry

end height_of_bottom_step_l214_214289


namespace positive_integer_cases_l214_214469

theorem positive_integer_cases (x : ℝ) (hx : x ≠ 0) :
  (∃ n : ℤ, (abs (x^2 - abs x)) / x = n ∧ n > 0) ↔ (∃ m : ℤ, (x = m) ∧ (m > 1 ∨ m < -1)) :=
by
  sorry

end positive_integer_cases_l214_214469


namespace number_line_problem_l214_214247

theorem number_line_problem (x : ℤ) (h : x + 7 - 4 = 0) : x = -3 :=
by
  -- The proof is omitted as only the statement is required.
  sorry

end number_line_problem_l214_214247


namespace smallest_n_for_three_nested_rectangles_l214_214608

/-- Rectangle represented by its side lengths -/
structure Rectangle where
  x : ℕ
  y : ℕ
  h1 : 1 ≤ x
  h2 : x ≤ y
  h3 : y ≤ 100

/-- Define the nesting relation between rectangles -/
def nested (R1 R2 : Rectangle) : Prop :=
  R1.x < R2.x ∧ R1.y < R2.y

/-- Prove the smallest n such that there exist 3 nested rectangles out of n rectangles where n = 101 -/
theorem smallest_n_for_three_nested_rectangles (n : ℕ) (h : n ≥ 101) :
  ∀ (rectangles : Fin n → Rectangle), 
    ∃ (R1 R2 R3 : Fin n), nested (rectangles R1) (rectangles R2) ∧ nested (rectangles R2) (rectangles R3) :=
  sorry

end smallest_n_for_three_nested_rectangles_l214_214608


namespace solution_set_of_inequality_l214_214552

theorem solution_set_of_inequality :
  { x : ℝ | - (1 : ℝ) / 2 < x ∧ x <= 1 } =
  { x : ℝ | (x - 1) / (2 * x + 1) <= 0 ∧ x ≠ - (1 : ℝ) / 2 } :=
by
  sorry

end solution_set_of_inequality_l214_214552


namespace equation_has_exactly_one_real_solution_l214_214899

-- Definitions for the problem setup
def equation (k : ℝ) (x : ℝ) : Prop := (3 * x + 8) * (x - 6) = -54 + k * x

-- The property that we need to prove
theorem equation_has_exactly_one_real_solution (k : ℝ) :
  (∀ x : ℝ, equation k x → ∃! x : ℝ, equation k x) ↔ k = 6 * Real.sqrt 2 - 10 ∨ k = -6 * Real.sqrt 2 - 10 := 
sorry

end equation_has_exactly_one_real_solution_l214_214899


namespace Aimee_escalator_time_l214_214457

theorem Aimee_escalator_time (d : ℝ) (v_esc : ℝ) (v_walk : ℝ) :
  v_esc = d / 60 → v_walk = d / 90 → (d / (v_esc + v_walk)) = 36 :=
by
  intros h1 h2
  sorry

end Aimee_escalator_time_l214_214457


namespace sum_of_odd_integers_l214_214832

theorem sum_of_odd_integers (n : ℕ) (h1 : 4970 = n * (1 + n)) : (n ^ 2 = 4900) :=
by
  sorry

end sum_of_odd_integers_l214_214832


namespace find_k_l214_214113

theorem find_k (k : ℝ) (A B : ℝ × ℝ) 
  (hA : A = (2, 3)) (hB : B = (4, k)) 
  (hAB_parallel : A.2 = B.2) : k = 3 := 
by 
  have hA_def : A = (2, 3) := hA 
  have hB_def : B = (4, k) := hB 
  have parallel_condition: A.2 = B.2 := hAB_parallel
  simp at parallel_condition
  sorry

end find_k_l214_214113


namespace find_perpendicular_line_l214_214594

-- Define the point P
structure Point where
  x : ℤ
  y : ℤ

-- Define the line
structure Line where
  a : ℤ
  b : ℤ
  c : ℤ

-- The given problem conditions
def P : Point := { x := -1, y := 3 }

def given_line : Line := { a := 1, b := -2, c := 3 }

def perpendicular_line (line : Line) (point : Point) : Line :=
  ⟨ -line.b, line.a, -(line.a * point.y - line.b * point.x) ⟩

-- Theorem statement to prove
theorem find_perpendicular_line :
  perpendicular_line given_line P = { a := 2, b := 1, c := -1 } :=
by
  sorry

end find_perpendicular_line_l214_214594


namespace remainder_of_M_when_divided_by_32_l214_214935

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l214_214935


namespace cost_of_pants_is_250_l214_214877

variable (costTotal : ℕ) (costTShirt : ℕ) (numTShirts : ℕ) (numPants : ℕ)

def costPants (costTotal costTShirt numTShirts numPants : ℕ) : ℕ :=
  let costTShirts := numTShirts * costTShirt
  let costPantsTotal := costTotal - costTShirts
  costPantsTotal / numPants

-- Given conditions
axiom h1 : costTotal = 1500
axiom h2 : costTShirt = 100
axiom h3 : numTShirts = 5
axiom h4 : numPants = 4

-- Prove each pair of pants costs $250
theorem cost_of_pants_is_250 : costPants costTotal costTShirt numTShirts numPants = 250 :=
by
  -- Place proof here
  sorry

end cost_of_pants_is_250_l214_214877


namespace max_height_reached_l214_214713

def h (t : ℝ) : ℝ := -20 * t ^ 2 + 120 * t + 36

theorem max_height_reached :
  ∃ t : ℝ, h t = 216 ∧ t = 3 :=
sorry

end max_height_reached_l214_214713


namespace euler_phi_divisibility_l214_214803

def euler_phi (n : ℕ) : ℕ := sorry -- Placeholder for the Euler phi-function

theorem euler_phi_divisibility (n : ℕ) (hn : n > 0) :
    2^(n * (n + 1)) ∣ 32 * euler_phi (2^(2^n) - 1) :=
sorry

end euler_phi_divisibility_l214_214803


namespace greatest_b_for_no_minus_nine_in_range_l214_214564

theorem greatest_b_for_no_minus_nine_in_range :
  ∃ b_max : ℤ, (b_max = 16) ∧ (∀ b : ℤ, (b^2 < 288) ↔ (b ≤ 16)) :=
by
  sorry

end greatest_b_for_no_minus_nine_in_range_l214_214564


namespace james_pays_37_50_l214_214791

/-- 
James gets 20 singing lessons.
First lesson is free.
After the first 10 paid lessons, he only needs to pay for every other lesson.
Each lesson costs $5.
His uncle pays for half.
Prove that James pays $37.50.
--/

theorem james_pays_37_50 :
  let first_lessons := 1
  let total_lessons := 20
  let paid_lessons := 10
  let remaining_lessons := total_lessons - first_lessons - paid_lessons
  let paid_remaining_lessons := (remaining_lessons + 1) / 2
  let total_paid_lessons := paid_lessons + paid_remaining_lessons
  let cost_per_lesson := 5
  let total_payment := total_paid_lessons * cost_per_lesson
  let payment_by_james := total_payment / 2
  payment_by_james = 37.5 := 
by
  sorry

end james_pays_37_50_l214_214791


namespace more_divisible_by_7_than_11_l214_214417

open Nat

theorem more_divisible_by_7_than_11 :
  let N := 10000
  let count_7_not_11 := (N / 7) - (N / 77)
  let count_11_not_7 := (N / 11) - (N / 77)
  count_7_not_11 > count_11_not_7 := 
  by
    let N := 10000
    let count_7_not_11 := (N / 7) - (N / 77)
    let count_11_not_7 := (N / 11) - (N / 77)
    sorry

end more_divisible_by_7_than_11_l214_214417


namespace ducks_cows_legs_l214_214780

theorem ducks_cows_legs (D C : ℕ) (L H X : ℤ)
  (hC : C = 13)
  (hL : L = 2 * D + 4 * C)
  (hH : H = D + C)
  (hCond : L = 3 * H + X) : X = 13 := by
  sorry

end ducks_cows_legs_l214_214780


namespace f_odd_and_increasing_l214_214216

noncomputable def f (x : ℝ) : ℝ := 3^x - (1/3)^x

theorem f_odd_and_increasing : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) := sorry

end f_odd_and_increasing_l214_214216


namespace cubic_yards_to_cubic_feet_l214_214218

theorem cubic_yards_to_cubic_feet :
  (1 : ℝ) * 3^3 * 5 = 135 := by
sorry

end cubic_yards_to_cubic_feet_l214_214218


namespace platform_length_proof_l214_214428

noncomputable def train_length : ℝ := 1200
noncomputable def time_to_cross_tree : ℝ := 120
noncomputable def time_to_cross_platform : ℝ := 240
noncomputable def speed_of_train : ℝ := train_length / time_to_cross_tree
noncomputable def platform_length : ℝ := 2400 - train_length

theorem platform_length_proof (h1 : train_length = 1200) (h2 : time_to_cross_tree = 120) (h3 : time_to_cross_platform = 240) :
  platform_length = 1200 := by
  sorry

end platform_length_proof_l214_214428


namespace initial_minutes_under_plan_A_l214_214166

theorem initial_minutes_under_plan_A (x : ℕ) (planA_initial : ℝ) (planA_rate : ℝ) (planB_rate : ℝ) (call_duration : ℕ) :
  planA_initial = 0.60 ∧ planA_rate = 0.06 ∧ planB_rate = 0.08 ∧ call_duration = 3 ∧
  (planA_initial + planA_rate * (call_duration - x) = planB_rate * call_duration) →
  x = 9 := 
by
  intros h
  obtain ⟨h1, h2, h3, h4, heq⟩ := h
  -- Skipping the proof
  sorry

end initial_minutes_under_plan_A_l214_214166


namespace find_PS_length_l214_214924

theorem find_PS_length 
  (PT TR QS QP PQ : ℝ)
  (h1 : PT = 5)
  (h2 : TR = 10)
  (h3 : QS = 16)
  (h4 : QP = 13)
  (h5 : PQ = 7) : 
  PS = Real.sqrt 703 := 
sorry

end find_PS_length_l214_214924


namespace gcd_a_b_l214_214799

def a : ℕ := 6666666
def b : ℕ := 999999999

theorem gcd_a_b : Nat.gcd a b = 3 := by
  sorry

end gcd_a_b_l214_214799


namespace line_through_center_and_perpendicular_l214_214274

theorem line_through_center_and_perpendicular 
(C : ℝ × ℝ) 
(HC : ∀ (x y : ℝ), x ^ 2 + (y - 1) ^ 2 = 4 → C = (0, 1))
(l : ℝ → ℝ)
(Hl : ∀ x y : ℝ, 3 * x + 2 * y + 1 = 0 → y = l x)
: ∃ k b : ℝ, (∀ x : ℝ, y = k * x + b ↔ 2 * x - 3 * y + 3 = 0) :=
by 
  sorry

end line_through_center_and_perpendicular_l214_214274


namespace sum_of_roots_eq_three_l214_214044

theorem sum_of_roots_eq_three (x : ℝ) :
  (∃ x : ℝ, x^2 - 3 * x + 2 = 12) →
  ((roots (X^2 - 3 * X - 10)).sum = 3) :=
by
  intros h,
  sorry

end sum_of_roots_eq_three_l214_214044


namespace right_triangle_acute_angle_l214_214225

theorem right_triangle_acute_angle (a b : ℝ) (h1 : a + b = 90) (h2 : a = 55) : b = 35 := 
by sorry

end right_triangle_acute_angle_l214_214225


namespace arithmetic_sequence_1001th_term_l214_214275

theorem arithmetic_sequence_1001th_term (p q : ℤ)
  (h1 : 9 - p = (2 * q - 5))
  (h2 : (3 * p - q + 7) - 9 = (2 * q - 5)) :
  p + (1000 * (2 * q - 5)) = 5004 :=
by
  sorry

end arithmetic_sequence_1001th_term_l214_214275


namespace product_of_sequence_l214_214889

theorem product_of_sequence : 
  (1 / 2) * (4 / 1) * (1 / 8) * (16 / 1) * (1 / 32) * (64 / 1) * (1 / 128) * (256 / 1) * (1 / 512) * (1024 / 1) * (1 / 2048) * (4096 / 1) = 64 := 
by
  sorry

end product_of_sequence_l214_214889


namespace right_triangle_legs_from_medians_l214_214329

theorem right_triangle_legs_from_medians
  (a b : ℝ) (x y : ℝ)
  (h1 : x^2 + 4 * y^2 = 4 * a^2)
  (h2 : 4 * x^2 + y^2 = 4 * b^2) :
  y^2 = (16 * a^2 - 4 * b^2) / 15 ∧ x^2 = (16 * b^2 - 4 * a^2) / 15 :=
by
  sorry

end right_triangle_legs_from_medians_l214_214329


namespace intersection_of_M_and_N_l214_214512

variable {α : Type*} [LinearOrder α] [OrderBot α] [OrderTop α]

def M (x : α) : Prop := 0 < x ∧ x < 4

def N (x : α) : Prop := (1 / 3 : α) ≤ x ∧ x ≤ 5

theorem intersection_of_M_and_N {x : α} : (M x ∧ N x) ↔ ((1 / 3 : α) ≤ x ∧ x < 4) :=
by 
  sorry

end intersection_of_M_and_N_l214_214512


namespace min_convex_cover_area_l214_214714

-- Define the dimensions of the box and the hole
def box_side := 5
def hole_side := 1

-- Define a function to represent the minimum area convex cover
def min_area_convex_cover (box_side hole_side : ℕ) : ℕ :=
  5 -- As given in the problem, the minimum area is concluded to be 5.

-- Theorem to state that the minimum area of the convex cover is 5
theorem min_convex_cover_area : min_area_convex_cover box_side hole_side = 5 :=
by
  -- Proof of the theorem
  sorry

end min_convex_cover_area_l214_214714


namespace mary_final_weight_l214_214243

theorem mary_final_weight :
    let initial_weight := 99
    let initial_loss := 12
    let first_gain := 2 * initial_loss
    let second_loss := 3 * initial_loss
    let final_gain := 6
    let weight_after_first_loss := initial_weight - initial_loss
    let weight_after_first_gain := weight_after_first_loss + first_gain
    let weight_after_second_loss := weight_after_first_gain - second_loss
    let final_weight := weight_after_second_loss + final_gain
    in final_weight = 81 :=
by
    sorry

end mary_final_weight_l214_214243


namespace total_tiles_needed_l214_214450

-- Define the dimensions of the dining room
def dining_room_length : ℕ := 15
def dining_room_width : ℕ := 20

-- Define the width of the border
def border_width : ℕ := 2

-- Areas for one-foot by one-foot border tiles
def one_foot_tile_border_tiles : ℕ :=
  2 * (dining_room_width + (dining_room_width - 2 * border_width)) + 
  2 * ((dining_room_length - 2) + (dining_room_length - 2 * border_width))

-- Dimensions of the inner area
def inner_length : ℕ := dining_room_length - 2 * border_width
def inner_width : ℕ := dining_room_width - 2 * border_width

-- Area for two-foot by two-foot tiles
def inner_area : ℕ := inner_length * inner_width
def two_foot_tile_inner_tiles : ℕ := inner_area / 4

-- Total number of tiles
def total_tiles : ℕ := one_foot_tile_border_tiles + two_foot_tile_inner_tiles

-- Prove that the total number of tiles needed is 168
theorem total_tiles_needed : total_tiles = 168 := sorry

end total_tiles_needed_l214_214450


namespace max_integer_inequality_l214_214412

theorem max_integer_inequality (a b c: ℝ) (h₀: 0 < a) (h₁: 0 < b) (h₂: 0 < c) :
  (a^2 / (b / 29 + c / 31) + b^2 / (c / 29 + a / 31) + c^2 / (a / 29 + b / 31)) ≥ 14 * (a + b + c) :=
sorry

end max_integer_inequality_l214_214412


namespace sequence_bound_l214_214550

theorem sequence_bound (a b c : ℕ → ℝ) :
  (a 0 = 1) ∧ (b 0 = 0) ∧ (c 0 = 0) ∧
  (∀ n, n ≥ 1 → a n = a (n-1) + c (n-1) / n) ∧
  (∀ n, n ≥ 1 → b n = b (n-1) + a (n-1) / n) ∧
  (∀ n, n ≥ 1 → c n = c (n-1) + b (n-1) / n) →
  ∀ n, n ≥ 1 → |a n - (n + 1) / 3| < 2 / Real.sqrt (3 * n) :=
by sorry

end sequence_bound_l214_214550


namespace sum_of_coordinates_D_l214_214386

theorem sum_of_coordinates_D (M C D : ℝ × ℝ)
  (h1 : M = (5, 5))
  (h2 : C = (10, 10))
  (h3 : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
  D.1 + D.2 = 0 := 
sorry

end sum_of_coordinates_D_l214_214386


namespace proof_problem_l214_214502

open Real

noncomputable def problem_statement (PQ QR PR QS : ℝ) (S : Set ℝ) : Prop :=
  PQ = 6 ∧ QR = 8 ∧ PR = 10 ∧ QS ∈ S ∧ QS = 6 → 
  (PS PR : ℝ) (PS_rat SR_rat : ℝ) (angle_QSR : ℝ) =>
  PS = 18 / 5 ∧ 
  SR = PR - PS ∧ 
  PS_rat = PS / SR ∧ 
  SR_rat = SR / PR ∧ 
  angle_QSR = asin (64 / 75) ∧ 
  PS_rat = 9 / 16 ∧ 
  angle_QSR = asin (64 / 75)

theorem proof_problem : problem_statement 6 8 10 6 {PS | PS < 10} :=
  by sorry

end proof_problem_l214_214502


namespace find_x2_y2_l214_214911

theorem find_x2_y2 (x y : ℝ) (h₁ : (x + y)^2 = 9) (h₂ : x * y = -6) : x^2 + y^2 = 21 := 
by
  sorry

end find_x2_y2_l214_214911


namespace product_of_odd_primes_mod_32_l214_214956

theorem product_of_odd_primes_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31) in
  M % 32 = 25 :=
by
  let M := (3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31)
  sorry

end product_of_odd_primes_mod_32_l214_214956


namespace proof_equivalence_l214_214484

variables {A B C D : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
variables {α β γ δ : ℝ} -- angles are real numbers

-- Definition of cyclic quadrilateral
def cyclic_quadrilateral (α β γ δ : ℝ) : Prop :=
α + γ = 180 ∧ β + δ = 180

-- Definition of the problem statements
def statement1 (α γ : ℝ) : Prop :=
α = γ → α = 90

def statement3 (α γ : ℝ) : Prop :=
180 - α + 180 - γ = 180

def statement2 (α β : ℝ) (ψ χ : ℝ) : Prop := 
α = β → cyclic_quadrilateral α β ψ χ → ψ = χ ∨ (α = β ∧ α = ψ ∧ α = χ)

def statement4 (α β γ δ : ℝ) : Prop :=
1*α + 2*β + 3*γ + 4*δ = 360

-- Theorem statement
theorem proof_equivalence (α β γ δ : ℝ) :
  cyclic_quadrilateral α β γ δ →
  (statement1 α γ) ∧ (statement3 α γ) ∧ ¬(statement2 α β γ δ) ∧ ¬(statement4 α β γ δ) :=
by
  sorry

end proof_equivalence_l214_214484


namespace minimally_competent_subsets_count_l214_214144

def A : Finset ℕ := Finset.range 11 \{0}

def is_competent (s : Finset ℕ) (n : ℕ) : Prop :=
  n ∈ s ∧ s.card = n

def is_minimally_competent (s : Finset ℕ) : Prop :=
  ∃ n, is_competent s n ∧ ∀ t ⊂ s, ¬ is_competent t n

theorem minimally_competent_subsets_count :
  (Finset.filter is_minimally_competent (Finset.powerset A)).card = 129 :=
sorry

end minimally_competent_subsets_count_l214_214144


namespace daniella_lap_time_l214_214036

theorem daniella_lap_time
  (T_T : ℕ) (H_TT : T_T = 56)
  (meet_time : ℕ) (H_meet : meet_time = 24) :
  ∃ T_D : ℕ, T_D = 42 :=
by
  sorry

end daniella_lap_time_l214_214036


namespace min_elements_in_as_l214_214802

noncomputable def min_elems_in_A_s (n : ℕ) (S : Finset ℝ) (hS : S.card = n) : ℕ :=
  if 2 ≤ n then 2 * n - 3 else 0

theorem min_elements_in_as (n : ℕ) (S : Finset ℝ) (hS : S.card = n) (hn: 2 ≤ n) :
  ∃ (A_s : Finset ℝ), A_s.card = min_elems_in_A_s n S hS := sorry

end min_elements_in_as_l214_214802


namespace Angela_height_is_157_l214_214445

variable (height_Amy height_Helen height_Angela : ℕ)

-- The conditions
axiom h_Amy : height_Amy = 150
axiom h_Helen : height_Helen = height_Amy + 3
axiom h_Angela : height_Angela = height_Helen + 4

-- The proof to show Angela's height is 157 cm
theorem Angela_height_is_157 : height_Angela = 157 :=
by
  rw [h_Amy] at h_Helen
  rw [h_Helen] at h_Angela
  exact h_Angela

end Angela_height_is_157_l214_214445


namespace rationalize_denominator_correct_l214_214255

noncomputable def rationalize_denominator_sum : ℕ :=
  let a := real.root (5 : ℝ) 3;
  let b := real.root (3 : ℝ) 3;
  let A := real.root (25 : ℝ) 3;
  let B := real.root (15 : ℝ) 3;
  let C := real.root (9 : ℝ) 3;
  let D := 2;
  (25 + 15 + 9 + 2)

theorem rationalize_denominator_correct :
  rationalize_denominator_sum = 51 :=
  by sorry

end rationalize_denominator_correct_l214_214255


namespace arithmetic_sequence_201_is_61_l214_214645

def is_arithmetic_sequence_term (a_5 a_45 : ℤ) (n : ℤ) (a_n : ℤ) : Prop :=
  ∃ d a_1, a_1 + 4 * d = a_5 ∧ a_1 + 44 * d = a_45 ∧ a_1 + (n - 1) * d = a_n

theorem arithmetic_sequence_201_is_61 : is_arithmetic_sequence_term 33 153 61 201 :=
sorry

end arithmetic_sequence_201_is_61_l214_214645


namespace prob_geometry_given_algebra_l214_214835

variable (algebra geometry : ℕ) (total : ℕ)

/-- Proof of the probability of selecting a geometry question on the second draw,
    given that an algebra question is selected on the first draw. -/
theorem prob_geometry_given_algebra : 
  algebra = 3 ∧ geometry = 2 ∧ total = 5 →
  (algebra / (total : ℚ)) * (geometry / (total - 1 : ℚ)) = 1 / 2 :=
by
  intro h
  sorry

end prob_geometry_given_algebra_l214_214835


namespace gcd_of_180_and_450_l214_214602

theorem gcd_of_180_and_450 : Int.gcd 180 450 = 90 := 
  sorry

end gcd_of_180_and_450_l214_214602


namespace find_triplets_l214_214590

theorem find_triplets (x y z : ℕ) (h1 : x ≤ y) (h2 : x^2 + y^2 = 3 * 2016^z + 77) :
  (x, y, z) = (4, 8, 0) ∨ (x, y, z) = (14, 77, 1) ∨ (x, y, z) = (35, 70, 1) :=
  sorry

end find_triplets_l214_214590


namespace Jessie_initial_weight_l214_214111

def lost_first_week : ℕ := 56
def after_first_week : ℕ := 36

theorem Jessie_initial_weight :
  (after_first_week + lost_first_week = 92) :=
by
  sorry

end Jessie_initial_weight_l214_214111


namespace sum_of_all_three_digit_positive_even_integers_l214_214158

def sum_of_three_digit_even_integers : ℕ :=
  let a := 100
  let l := 998
  let d := 2
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem sum_of_all_three_digit_positive_even_integers :
  sum_of_three_digit_even_integers = 247050 :=
by
  -- proof to be completed
  sorry

end sum_of_all_three_digit_positive_even_integers_l214_214158


namespace find_B_coords_l214_214346

-- Define point A and vector a
def A : (ℝ × ℝ) := (1, -3)
def a : (ℝ × ℝ) := (3, 4)

-- Assume B is at coordinates (m, n) and AB = 2a
def B : (ℝ × ℝ) := (7, 5)
def AB : (ℝ × ℝ) := (B.1 - A.1, B.2 - A.2)

-- Prove point B has the correct coordinates
theorem find_B_coords : AB = (2 * a.1, 2 * a.2) → B = (7, 5) :=
by
  intro h
  sorry

end find_B_coords_l214_214346


namespace product_of_solutions_abs_eq_l214_214895

theorem product_of_solutions_abs_eq (x : ℝ) :
  (∃ x1 x2 : ℝ, |6 * x1 + 2| + 5 = 47 ∧ |6 * x2 + 2| + 5 = 47 ∧ x ≠ x1 ∧ x ≠ x2 ∧ x1 * x2 = -440 / 9) :=
by
  sorry

end product_of_solutions_abs_eq_l214_214895


namespace angela_height_l214_214446

def height_of_Amy : ℕ := 150
def height_of_Helen : ℕ := height_of_Amy + 3
def height_of_Angela : ℕ := height_of_Helen + 4

theorem angela_height : height_of_Angela = 157 := by
  sorry

end angela_height_l214_214446


namespace train_speed_kmph_l214_214579

/-- Define the lengths of the train and bridge, as well as the time taken to cross the bridge. --/
def train_length : ℝ := 150
def bridge_length : ℝ := 150
def crossing_time_seconds : ℝ := 29.997600191984642

/-- Calculate the speed of the train in km/h. --/
theorem train_speed_kmph : 
  let total_distance := train_length + bridge_length
  let time_in_hours := crossing_time_seconds / 3600
  let speed_mph := total_distance / time_in_hours
  let speed_kmph := speed_mph / 1000
  speed_kmph = 36 := by
  /- Proof omitted -/
  sorry

end train_speed_kmph_l214_214579


namespace hamburgers_left_over_l214_214312

theorem hamburgers_left_over (h_made : ℕ) (h_served : ℕ) (h_total : h_made = 9) (h_served_count : h_served = 3) : h_made - h_served = 6 :=
by
  sorry

end hamburgers_left_over_l214_214312


namespace problem_proof_l214_214628

variable (P Q M N : ℝ)

axiom hp1 : M = 0.40 * Q
axiom hp2 : Q = 0.30 * P
axiom hp3 : N = 1.20 * P

theorem problem_proof : (M / N) = (1 / 10) := by
  sorry

end problem_proof_l214_214628


namespace basketball_game_total_points_l214_214104

theorem basketball_game_total_points :
  ∃ (a d b: ℕ) (r: ℝ), 
      a = b + 2 ∧     -- Eagles lead by 2 points at the end of the first quarter
      (a + d < 100) ∧ -- Points scored by Eagles in each quarter form an increasing arithmetic sequence
      (b * r < 100) ∧ -- Points scored by Lions in each quarter form an increasing geometric sequence
      (a + (a + d) + (a + 2 * d)) = b * (1 + r + r^2) ∧ -- Aggregate score tied at the end of the third quarter
      (a + (a + d) + (a + 2 * d) + (a + 3 * d) + b * (1 + r + r^2 + r^3) = 144) -- Total points scored by both teams 
   :=
sorry

end basketball_game_total_points_l214_214104


namespace area_of_triangle_PQR_l214_214434

-- Define point P
structure Point where
  x : ℝ
  y : ℝ

def P : Point := { x := 2, y := 5 }

-- Define the lines using their slopes and the point P
def line1 (x : ℝ) : ℝ := -x + 7
def line2 (x : ℝ) : ℝ := -2 * x + 9

-- Definitions of points Q and R, which are the x-intercepts
def Q : Point := { x := 7, y := 0 }
def R : Point := { x := 4.5, y := 0 }

-- Theorem statement
theorem area_of_triangle_PQR : 
  let base := 7 - 4.5
  let height := 5
  (1 / 2) * base * height = 6.25 := by
  sorry

end area_of_triangle_PQR_l214_214434


namespace largest_vertex_sum_of_parabola_l214_214467

theorem largest_vertex_sum_of_parabola 
  (a T : ℤ)
  (hT : T ≠ 0)
  (h1 : 0 = a * 0^2 + b * 0 + c)
  (h2 : 0 = a * (2 * T) ^ 2 + b * (2 * T) + c)
  (h3 : 36 = a * (2 * T + 2) ^ 2 + b * (2 * T + 2) + c) :
  ∃ N : ℚ, N = -5 / 4 :=
sorry

end largest_vertex_sum_of_parabola_l214_214467


namespace intersection_of_M_and_N_l214_214513

variable {α : Type*} [LinearOrder α] [OrderBot α] [OrderTop α]

def M (x : α) : Prop := 0 < x ∧ x < 4

def N (x : α) : Prop := (1 / 3 : α) ≤ x ∧ x ≤ 5

theorem intersection_of_M_and_N {x : α} : (M x ∧ N x) ↔ ((1 / 3 : α) ≤ x ∧ x < 4) :=
by 
  sorry

end intersection_of_M_and_N_l214_214513


namespace friendships_structure_count_l214_214505

/-- In a group of 8 individuals, where each person has exactly 3 friends within the group,
there are 420 different ways to structure these friendships. -/
theorem friendships_structure_count : 
  ∃ (structure_count : ℕ), 
    structure_count = 420 ∧ 
    (∀ (G : Fin 8 → Fin 8 → Prop), 
      (∀ i, ∃! (j₁ j₂ j₃ : Fin 8), G i j₁ ∧ G i j₂ ∧ G i j₃) ∧ 
      (∀ i j, G i j → G j i) ∧ 
      (structure_count = 420)) := 
by
  sorry

end friendships_structure_count_l214_214505


namespace problem1_problem2_l214_214389

theorem problem1 (x : ℚ) (h : x ≠ -4) : (3 - x) / (x + 4) = 1 / 2 → x = 2 / 3 :=
by
  sorry

theorem problem2 (x : ℚ) (h : x ≠ 1) : x / (x - 1) - 2 * x / (3 * (x - 1)) = 1 → x = 3 / 2 :=
by
  sorry

end problem1_problem2_l214_214389


namespace perfect_square_trinomial_k_l214_214635

theorem perfect_square_trinomial_k (k : ℤ) : 
  (∀ x : ℝ, x^2 - k*x + 64 = (x + 8)^2 ∨ x^2 - k*x + 64 = (x - 8)^2) → 
  (k = 16 ∨ k = -16) :=
by
  sorry

end perfect_square_trinomial_k_l214_214635


namespace distinct_ordered_pair_count_l214_214089

theorem distinct_ordered_pair_count (x y : ℕ) (h1 : x + y = 50) (h2 : 1 ≤ x) (h3 : 1 ≤ y) : 
  ∃! (x y : ℕ), x + y = 50 ∧ 1 ≤ x ∧ 1 ≤ y :=
by
  sorry

end distinct_ordered_pair_count_l214_214089


namespace solve_for_x_l214_214219

theorem solve_for_x (x t : ℝ)
  (h₁ : t = 9)
  (h₂ : (3 * (x + 5)) / 4 = t + (3 - 3 * x) / 2) :
  x = 3 :=
by
  sorry

end solve_for_x_l214_214219


namespace weight_range_correct_l214_214839

noncomputable def combined_weight : ℕ := 158
noncomputable def tracy_weight : ℕ := 52
noncomputable def jake_weight : ℕ := tracy_weight + 8
noncomputable def john_weight : ℕ := combined_weight - (tracy_weight + jake_weight)
noncomputable def weight_range : ℕ := jake_weight - john_weight

theorem weight_range_correct : weight_range = 14 := 
by
  sorry

end weight_range_correct_l214_214839


namespace car_overtakes_truck_l214_214576

theorem car_overtakes_truck 
  (car_speed : ℝ)
  (truck_speed : ℝ)
  (car_arrival_time : ℝ)
  (truck_arrival_time : ℝ)
  (route_same : Prop)
  (time_difference : ℝ)
  (car_speed_km_min : car_speed = 66 / 60)
  (truck_speed_km_min : truck_speed = 42 / 60)
  (arrival_time_difference : truck_arrival_time - car_arrival_time = 18 / 60) :
  ∃ d : ℝ, d = 34.65 := 
by {
  sorry
}

end car_overtakes_truck_l214_214576


namespace temperature_on_tuesday_l214_214265

variable (T W Th F : ℝ)

theorem temperature_on_tuesday :
  (T + W + Th) / 3 = 45 →
  (W + Th + F) / 3 = 50 →
  F = 53 →
  T = 38 :=
by 
  intros h1 h2 h3
  sorry

end temperature_on_tuesday_l214_214265


namespace final_amoeba_is_blue_l214_214224

-- We define the initial counts of each type of amoeba
def initial_red : ℕ := 26
def initial_blue : ℕ := 31
def initial_yellow : ℕ := 16

-- We define the final count of amoebas
def final_amoebas : ℕ := 1

-- The type of the final amoeba (we're proving it's 'blue')
inductive AmoebaColor
| Red
| Blue
| Yellow

-- Given initial counts, we aim to prove the final amoeba is blue
theorem final_amoeba_is_blue :
  initial_red = 26 ∧ initial_blue = 31 ∧ initial_yellow = 16 ∧ final_amoebas = 1 → 
  ∃ c : AmoebaColor, c = AmoebaColor.Blue :=
by sorry

end final_amoeba_is_blue_l214_214224


namespace ways_to_distribute_5_balls_in_3_boxes_with_conditions_l214_214768

noncomputable def num_ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) (min_in_box_c : ℕ) : ℕ :=
  let total_ways := boxes ^ balls
  let invalid_ways := (2 ^ balls) + (balls * (2 ^ (balls - 1)))
  total_ways - invalid_ways

theorem ways_to_distribute_5_balls_in_3_boxes_with_conditions : num_ways_to_distribute_balls 5 3 2 = 131 := by
  sorry

end ways_to_distribute_5_balls_in_3_boxes_with_conditions_l214_214768


namespace quadratic_inequality_solution_l214_214390

theorem quadratic_inequality_solution :
  (∀ x : ℝ, x ∈ Set.Ioo ((1 - Real.sqrt 2) / 3) ((1 + Real.sqrt 2) / 3) → -9 * x^2 + 6 * x + 1 < 0) ∧
  (∀ x : ℝ, -9 * x^2 + 6 * x + 1 < 0 → x ∈ Set.Ioo ((1 - Real.sqrt 2) / 3) ((1 + Real.sqrt 2) / 3)) :=
by
  sorry

end quadratic_inequality_solution_l214_214390


namespace sum_of_squares_of_projections_constant_l214_214128

-- Defines a function that calculates the sum of the squares of the projections of the edges of a cube onto any plane.
def sum_of_squares_of_projections (a : ℝ) (n : ℝ × ℝ × ℝ) : ℝ :=
  let α := n.1
  let β := n.2.1
  let γ := n.2.2
  4 * (a^2) * (2)

-- Define the theorem statement that proves the sum of the squares of the projections is constant and equal to 8a^2
theorem sum_of_squares_of_projections_constant (a : ℝ) (n : ℝ × ℝ × ℝ) :
  sum_of_squares_of_projections a n = 8 * a^2 :=
by
  -- Since we assume the trigonometric identity holds, directly match the sum_of_squares_of_projections function result.
  sorry

end sum_of_squares_of_projections_constant_l214_214128


namespace different_outcomes_count_l214_214079

-- Define the number of students and competitions
def num_students : ℕ := 4
def num_competitions : ℕ := 3

-- Define the proof statement
theorem different_outcomes_count : (num_competitions ^ num_students) = 81 := 
by
  -- Proof will be here
  sorry

end different_outcomes_count_l214_214079


namespace monotonic_intervals_l214_214463

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_intervals :
  (∀ x (h : 0 < x ∧ x < Real.exp 1), 0 < f x) ∧
  (∀ x (h : Real.exp 1 < x), f x < 0) :=
by
  sorry

end monotonic_intervals_l214_214463


namespace remainder_of_M_when_divided_by_32_l214_214933

open Nat

def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
def M : ℕ := List.foldl (*) 1 oddPrimes

theorem remainder_of_M_when_divided_by_32 :
  M % 32 = 1 :=
sorry

end remainder_of_M_when_divided_by_32_l214_214933


namespace sum_of_fractions_l214_214073

theorem sum_of_fractions : 
  (∑ n in Finset.range 14, (1 : ℚ) / ((n + 1) * (n + 2))) = 14 / 15 := 
by
  sorry

end sum_of_fractions_l214_214073


namespace remainder_when_divided_by_32_l214_214966

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l214_214966


namespace lcm_3_15_is_15_l214_214614

theorem lcm_3_15_is_15 : Nat.lcm 3 15 = 15 :=
sorry

end lcm_3_15_is_15_l214_214614


namespace even_function_k_value_l214_214774

theorem even_function_k_value (k : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = k * x^2 + (k - 1) * x + 2)
  (even_f : ∀ x : ℝ, f x = f (-x)) : k = 1 :=
by
  -- Proof would go here
  sorry

end even_function_k_value_l214_214774


namespace museum_rid_paintings_l214_214711

def initial_paintings : ℕ := 1795
def leftover_paintings : ℕ := 1322

theorem museum_rid_paintings : initial_paintings - leftover_paintings = 473 := by
  sorry

end museum_rid_paintings_l214_214711


namespace find_a_solution_l214_214897

open Complex

noncomputable def find_a : Prop := 
  ∃ a : ℂ, ((1 + a * I) / (2 + I) = 1 + 2 * I) ∧ (a = 5 + I)

theorem find_a_solution : find_a := 
  by
    sorry

end find_a_solution_l214_214897


namespace problem1_solution_set_problem2_range_of_m_l214_214091

open Real

noncomputable def f (x : ℝ) := abs (x + 1) - abs (x - 2)

theorem problem1_solution_set :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} :=
sorry

theorem problem2_range_of_m (m : ℝ) :
  (∃ x : ℝ, f x ≥ x^2 - x + m) → m ≤ 5 / 4 :=
sorry

end problem1_solution_set_problem2_range_of_m_l214_214091


namespace teairras_pants_count_l214_214134

-- Definitions according to the given conditions
def total_shirts := 5
def plaid_shirts := 3
def purple_pants := 5
def neither_plaid_nor_purple := 21

-- The theorem we need to prove
theorem teairras_pants_count :
  ∃ (pants : ℕ), pants = (neither_plaid_nor_purple - (total_shirts - plaid_shirts)) + purple_pants ∧ pants = 24 :=
by
  sorry

end teairras_pants_count_l214_214134


namespace seq_fifth_term_l214_214478

def seq (a : ℕ → ℤ) : Prop :=
  (a 1 = 3) ∧ (a 2 = 6) ∧ (∀ n : ℕ, a (n + 2) = a (n + 1) - a n)

theorem seq_fifth_term (a : ℕ → ℤ) (h : seq a) : a 5 = -6 :=
by
  sorry

end seq_fifth_term_l214_214478


namespace find_second_dimension_of_smaller_box_l214_214642

def volume_large_box : ℕ := 12 * 14 * 16
def volume_small_box (x : ℕ) : ℕ := 3 * x * 2
def max_small_boxes : ℕ := 64

theorem find_second_dimension_of_smaller_box (x : ℕ) : volume_large_box = max_small_boxes * volume_small_box x → x = 7 :=
by
  intros h
  unfold volume_large_box at h
  unfold volume_small_box at h
  sorry

end find_second_dimension_of_smaller_box_l214_214642


namespace max_contestants_l214_214539

theorem max_contestants (n : ℕ) (h1 : n = 55) (h2 : ∀ (i j : ℕ), i < j → j < n → (j - i) % 5 ≠ 4) : ∃(k : ℕ), k = 30 := 
  sorry

end max_contestants_l214_214539


namespace whiskers_count_l214_214201

variable (P C S : ℕ)

theorem whiskers_count :
  P = 14 →
  C = 2 * P - 6 →
  S = P + C + 8 →
  C = 22 ∧ S = 44 :=
by
  intros hP hC hS
  rw [hP] at hC
  rw [hP, hC] at hS
  exact ⟨hC, hS⟩

end whiskers_count_l214_214201


namespace find_expression_for_a_n_l214_214900

noncomputable def seq (n : ℕ) : ℕ := sorry
def sumFirstN (n : ℕ) : ℕ := sorry

theorem find_expression_for_a_n (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h_pos : ∀ n, 0 < a n)
  (h_arith_seq : ∀ n, S n + 1 = 2 * a n) :
  ∀ n, a n = 2^(n-1) :=
sorry

end find_expression_for_a_n_l214_214900


namespace twenty_five_percent_less_than_eighty_equals_one_fourth_more_than_what_number_l214_214840

theorem twenty_five_percent_less_than_eighty_equals_one_fourth_more_than_what_number :
  ∃ n : ℝ, (80 - 0.25 * 80) = (5 / 4) * n ∧ n = 48 := 
by
  sorry

end twenty_five_percent_less_than_eighty_equals_one_fourth_more_than_what_number_l214_214840


namespace number_of_people_l214_214532

theorem number_of_people
  (x y : ℕ)
  (h1 : x + y = 28)
  (h2 : 2 * x + 4 * y = 92) :
  x = 10 :=
by
  sorry

end number_of_people_l214_214532


namespace quadratic_unique_root_l214_214529

theorem quadratic_unique_root (b c : ℝ)
  (h₁ : b = c^2 + 1)
  (h₂ : (x^2 + b * x + c = 0) → ∃! x : ℝ, x^2 + b * x + c = 0) :
  c = 1 ∨ c = -1 := 
sorry

end quadratic_unique_root_l214_214529


namespace projectile_reaches_100_feet_l214_214272

noncomputable def projectile_height (t : ℝ) : ℝ :=
  -16 * t^2 + 80 * t

theorem projectile_reaches_100_feet :
  ∃ t : ℝ, t = 2.5 ∧ projectile_height t = 100 :=
by
  use 2.5
  sorry

end projectile_reaches_100_feet_l214_214272


namespace remaining_wire_length_l214_214866

theorem remaining_wire_length (total_length : ℝ) (fraction_cut : ℝ) (remaining_length : ℝ) (h1 : total_length = 3) (h2 : fraction_cut = 1 / 3) (h3 : remaining_length = 2) :
  total_length * (1 - fraction_cut) = remaining_length :=
by
  -- Proof goes here
  sorry

end remaining_wire_length_l214_214866


namespace average_of_remaining_two_numbers_l214_214665

theorem average_of_remaining_two_numbers (a b c d e f : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 2.5)
  (h2 : (a + b) / 2 = 1.1)
  (h3 : (c + d) / 2 = 1.4) : 
  (e + f) / 2 = 5 :=
by
  sorry

end average_of_remaining_two_numbers_l214_214665


namespace gold_copper_ratio_l214_214491

theorem gold_copper_ratio (G C : ℕ) (h : 19 * G + 9 * C = 17 * (G + C)) : G = 4 * C :=
by
  sorry

end gold_copper_ratio_l214_214491


namespace proof_problem_l214_214084

variables (α : ℝ)

-- Condition: tan(α) = 2
def tan_condition : Prop := Real.tan α = 2

-- First expression: (sin α + 2 cos α) / (4 cos α - sin α) = 2
def expression1 : Prop := (Real.sin α + 2 * Real.cos α) / (4 * Real.cos α - Real.sin α) = 2

-- Second expression: sqrt(2) * sin(2α + π/4) + 1 = 6/5
def expression2 : Prop := Real.sqrt 2 * Real.sin (2 * α + Real.pi / 4) + 1 = 6 / 5

-- Theorem: Prove the expressions given the condition
theorem proof_problem :
  tan_condition α → expression1 α ∧ expression2 α :=
by
  intro tan_cond
  have h1 : expression1 α := sorry
  have h2 : expression2 α := sorry
  exact ⟨h1, h2⟩

end proof_problem_l214_214084


namespace max_value_quadratic_max_value_quadratic_attained_l214_214692

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem max_value_quadratic : ∀ (x : ℝ), quadratic (-8) 32 (-1) x ≤ 31 :=
by
  sorry

theorem max_value_quadratic_attained : 
  quadratic (-8) 32 (-1) 2 = 31 :=
by
  sorry

end max_value_quadratic_max_value_quadratic_attained_l214_214692


namespace syntheticMethod_correct_l214_214146

-- Definition: The synthetic method leads from cause to effect.
def syntheticMethod (s : String) : Prop :=
  s = "The synthetic method leads from cause to effect, gradually searching for the necessary conditions that are known."

-- Question: Is the statement correct?
def question : String :=
  "The thought process of the synthetic method is to lead from cause to effect, gradually searching for the necessary conditions that are known."

-- Options given
def options : List String := ["Correct", "Incorrect", "", ""]

-- Correct answer is Option A - "Correct"
def correctAnswer : String := "Correct"

theorem syntheticMethod_correct :
  syntheticMethod question → options.head? = some correctAnswer :=
sorry

end syntheticMethod_correct_l214_214146


namespace distinct_elements_in_T_l214_214191

def sequence1 (k : ℕ) : ℤ := 3 * k - 1
def sequence2 (m : ℕ) : ℤ := 8 * m + 2

def setC : Finset ℤ := Finset.image sequence1 (Finset.range 3000)
def setD : Finset ℤ := Finset.image sequence2 (Finset.range 3000)
def setT : Finset ℤ := setC ∪ setD

theorem distinct_elements_in_T : setT.card = 3000 := by
  sorry

end distinct_elements_in_T_l214_214191


namespace base_b_equivalence_l214_214362

theorem base_b_equivalence (b : ℕ) (h : (2 * b + 4) ^ 2 = 5 * b ^ 2 + 5 * b + 4) : b = 12 :=
sorry

end base_b_equivalence_l214_214362


namespace range_of_k_l214_214634

variable (k x : ℝ)

def f (k x : ℝ) : ℝ := k * x - Real.log x

def f' (k x : ℝ) : ℝ := k - 1/x

theorem range_of_k :
  (∀ x : ℝ, 1 < x → f' k x ≥ 0) ↔ k ∈ Set.Ici 1 := by
  sorry

end range_of_k_l214_214634


namespace cat_food_more_than_dog_food_l214_214880

theorem cat_food_more_than_dog_food :
  let cat_food_packs := 6
  let cans_per_cat_pack := 9
  let dog_food_packs := 2
  let cans_per_dog_pack := 3
  let total_cat_food_cans := cat_food_packs * cans_per_cat_pack
  let total_dog_food_cans := dog_food_packs * cans_per_dog_pack
  total_cat_food_cans - total_dog_food_cans = 48 :=
by
  sorry

end cat_food_more_than_dog_food_l214_214880


namespace number_of_red_balls_eq_47_l214_214430

theorem number_of_red_balls_eq_47
  (T : ℕ) (white green yellow purple : ℕ)
  (neither_red_nor_purple_prob : ℚ)
  (hT : T = 100)
  (hWhite : white = 10)
  (hGreen : green = 30)
  (hYellow : yellow = 10)
  (hPurple : purple = 3)
  (hProb : neither_red_nor_purple_prob = 0.5)
  : T - (white + green + yellow + purple) = 47 :=
by
  -- Sorry is used to skip the actual proof
  sorry

end number_of_red_balls_eq_47_l214_214430


namespace miles_left_to_reach_E_l214_214794

-- Given conditions as definitions
def total_journey : ℕ := 2500
def miles_driven : ℕ := 642
def miles_B_to_C : ℕ := 400
def miles_C_to_D : ℕ := 550
def detour_D_to_E : ℕ := 200

-- Proof statement
theorem miles_left_to_reach_E : 
  (miles_B_to_C + miles_C_to_D + detour_D_to_E) = 1150 :=
by
  sorry

end miles_left_to_reach_E_l214_214794


namespace factorial_division_l214_214416

-- Definitions of factorial used in Lean according to math problem statement.
open Nat

-- Statement of the proof problem in Lean 4.
theorem factorial_division : (12! - 11!) / 10! = 121 := by
  sorry

end factorial_division_l214_214416


namespace max_value_under_constraint_l214_214771

noncomputable def max_value_expression (a b c : ℝ) : ℝ :=
3 * a * b - 3 * b * c + 2 * c^2

theorem max_value_under_constraint
  (a b c : ℝ)
  (h : a^2 + b^2 + c^2 = 1) :
  max_value_expression a b c ≤ 3 :=
sorry

end max_value_under_constraint_l214_214771


namespace minerals_now_l214_214795

def minerals_yesterday (M : ℕ) : Prop := (M / 2 = 21)

theorem minerals_now (M : ℕ) (H : minerals_yesterday M) : (M + 6 = 48) :=
by 
  unfold minerals_yesterday at H
  sorry

end minerals_now_l214_214795


namespace water_usage_l214_214309

def fee (x : ℕ) : ℕ :=
  if x ≤ 8 then 2 * x else 4 * x - 16

theorem water_usage (h : fee 9 = 20) : fee 9 = 20 := by
  sorry

end water_usage_l214_214309


namespace middle_digit_is_zero_l214_214884

noncomputable def N_in_base8 (a b c : ℕ) : ℕ := 512 * a + 64 * b + 8 * c
noncomputable def N_in_base10 (a b c : ℕ) : ℕ := 100 * b + 10 * c + a

theorem middle_digit_is_zero (a b c : ℕ) (h : N_in_base8 a b c = N_in_base10 a b c) :
  b = 0 :=
by 
  sorry

end middle_digit_is_zero_l214_214884
