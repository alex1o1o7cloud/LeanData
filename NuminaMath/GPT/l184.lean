import Mathlib

namespace fair_die_proba_l184_18407
noncomputable def probability_of_six : ℚ := 1 / 6

theorem fair_die_proba : 
  (1 / 6 : ℚ) = probability_of_six :=
by
  sorry

end fair_die_proba_l184_18407


namespace evaporation_period_l184_18430

theorem evaporation_period
  (total_water : ℕ)
  (daily_evaporation_rate : ℝ)
  (percentage_evaporated : ℝ)
  (evaporation_period_days : ℕ)
  (h_total_water : total_water = 10)
  (h_daily_evaporation_rate : daily_evaporation_rate = 0.006)
  (h_percentage_evaporated : percentage_evaporated = 0.03)
  (h_evaporation_period_days : evaporation_period_days = 50):
  (percentage_evaporated * total_water) / daily_evaporation_rate = evaporation_period_days := by
  sorry

end evaporation_period_l184_18430


namespace train_speed_is_260_kmph_l184_18461

-- Define the conditions: length of the train and time to cross the pole
def length_of_train : ℝ := 130
def time_to_cross_pole : ℝ := 9

-- Define the conversion factor from meters per second to kilometers per hour
def conversion_factor : ℝ := 3.6

-- Define the expected speed in kilometers per hour
def expected_speed_kmph : ℝ := 260

-- The theorem statement
theorem train_speed_is_260_kmph :
  (length_of_train / time_to_cross_pole) * conversion_factor = expected_speed_kmph :=
sorry

end train_speed_is_260_kmph_l184_18461


namespace samantha_born_in_1979_l184_18486

-- Condition definitions
def first_AMC8_year := 1985
def annual_event (n : ℕ) : ℕ := first_AMC8_year + n
def seventh_AMC8_year := annual_event 6

variable (Samantha_age_in_seventh_AMC8 : ℕ)
def Samantha_age_when_seventh_AMC8 := 12
def Samantha_birth_year := seventh_AMC8_year - Samantha_age_when_seventh_AMC8

-- Proof statement
theorem samantha_born_in_1979 : Samantha_birth_year = 1979 :=
by
  sorry

end samantha_born_in_1979_l184_18486


namespace weight_of_replaced_person_l184_18441

theorem weight_of_replaced_person
  (avg_increase : ∀ W : ℝ, W + 8 * 2.5 = W - X + 80)
  (new_person_weight : 80 = 80):
  X = 60 := by
  sorry

end weight_of_replaced_person_l184_18441


namespace find_ab_l184_18458

theorem find_ab (a b : ℝ) : 
  (∀ x : ℝ, (3 * x - a) * (2 * x + 5) - x = 6 * x^2 + 2 * (5 * x - b)) → a = 2 ∧ b = 5 :=
by
  intro h
  -- We assume the condition holds for all x
  sorry -- Proof not needed as per instructions

end find_ab_l184_18458


namespace bonnie_egg_count_indeterminable_l184_18485

theorem bonnie_egg_count_indeterminable
    (eggs_Kevin : ℕ)
    (eggs_George : ℕ)
    (eggs_Cheryl : ℕ)
    (diff_Cheryl_combined : ℕ)
    (c1 : eggs_Kevin = 5)
    (c2 : eggs_George = 9)
    (c3 : eggs_Cheryl = 56)
    (c4 : diff_Cheryl_combined = 29)
    (h₁ : eggs_Cheryl = diff_Cheryl_combined + (eggs_Kevin + eggs_George + some_children)) :
    ∀ (eggs_Bonnie : ℕ), ∃ some_children : ℕ, eggs_Bonnie = eggs_Bonnie :=
by
  -- The proof is omitted here
  sorry

end bonnie_egg_count_indeterminable_l184_18485


namespace probability_product_even_gt_one_fourth_l184_18476

def n := 100
def is_even (x : ℕ) : Prop := x % 2 = 0
def is_odd (x : ℕ) : Prop := ¬ is_even x

theorem probability_product_even_gt_one_fourth :
  (∃ (p : ℝ), p > 0 ∧ p = 1 - (50 * 49 * 48 : ℝ) / (100 * 99 * 98) ∧ p > 1 / 4) :=
sorry

end probability_product_even_gt_one_fourth_l184_18476


namespace find_x_l184_18471

theorem find_x (x y z : ℚ) (h1 : (x * y) / (x + y) = 4) (h2 : (x * z) / (x + z) = 5) (h3 : (y * z) / (y + z) = 6) : x = 40 / 9 :=
by
  -- Structure the proof here
  sorry

end find_x_l184_18471


namespace fold_paper_crease_length_l184_18412

theorem fold_paper_crease_length 
    (w l : ℝ) (w_pos : w = 12) (l_pos : l = 16) 
    (F G : ℝ × ℝ) (F_on_AD : F = (0, 12))
    (G_on_BC : G = (16, 12)) :
    dist F G = 20 := 
by
  sorry

end fold_paper_crease_length_l184_18412


namespace arithmetic_progression_root_difference_l184_18455

theorem arithmetic_progression_root_difference (a b c : ℚ) (h : 81 * a * a * a - 225 * a * a + 164 * a - 30 = 0)
  (hb : b = 5/3) (hprog : ∃ d : ℚ, a = b - d ∧ c = b + d) :
  c - a = 5 / 9 :=
sorry

end arithmetic_progression_root_difference_l184_18455


namespace general_term_seq_l184_18449

open Nat

-- Definition of the sequence given conditions
def seq (a : ℕ → ℕ) : Prop :=
  a 2 = 2 ∧ ∀ n, n ≥ 1 → (n - 1) * a (n + 1) - n * a n + 1 = 0

-- To prove that the general term is a_n = n
theorem general_term_seq (a : ℕ → ℕ) (h : seq a) : ∀ n, n ≥ 1 → a n = n := 
by
  sorry

end general_term_seq_l184_18449


namespace john_writing_time_l184_18415

def pages_per_day : ℕ := 20
def pages_per_book : ℕ := 400
def number_of_books : ℕ := 3

theorem john_writing_time : (pages_per_book / pages_per_day) * number_of_books = 60 :=
by
  -- The proof should be placed here.
  sorry

end john_writing_time_l184_18415


namespace star_operation_possible_l184_18427

noncomputable def star_operation_exists : Prop := 
  ∃ (star : ℤ → ℤ → ℤ), 
  (∀ (a b c : ℤ), star (star a b) c = star a (star b c)) ∧ 
  (∀ (x y : ℤ), star (star x x) y = y ∧ star y (star x x) = y)

theorem star_operation_possible : star_operation_exists :=
sorry

end star_operation_possible_l184_18427


namespace train_speed_l184_18483

theorem train_speed (length_bridge : ℕ) (time_total : ℕ) (time_on_bridge : ℕ) (speed_of_train : ℕ) 
  (h1 : length_bridge = 800)
  (h2 : time_total = 60)
  (h3 : time_on_bridge = 40)
  (h4 : length_bridge + (time_total - time_on_bridge) * speed_of_train = time_total * speed_of_train) :
  speed_of_train = 20 := sorry

end train_speed_l184_18483


namespace mary_investment_amount_l184_18462

theorem mary_investment_amount
  (A : ℝ := 100000) -- Future value in dollars
  (r : ℝ := 0.08) -- Annual interest rate
  (n : ℕ := 12) -- Compounded monthly
  (t : ℝ := 10) -- Time in years
  : (⌈A / (1 + r / n) ^ (n * t)⌉₊ = 45045) :=
by
  sorry

end mary_investment_amount_l184_18462


namespace find_xy_l184_18422

theorem find_xy (x y : ℝ) (h1 : (x / 6) * 12 = 11) (h2 : 4 * (x - y) + 5 = 11) : 
  x = 5.5 ∧ y = 4 :=
sorry

end find_xy_l184_18422


namespace kiki_scarves_count_l184_18425

variable (money : ℝ) (scarf_cost : ℝ) (hat_spending_ratio : ℝ) (scarves : ℕ) (hats : ℕ)

-- Condition: Kiki has $90.
axiom kiki_money : money = 90

-- Condition: Kiki spends 60% of her money on hats.
axiom kiki_hat_spending_ratio : hat_spending_ratio = 0.60

-- Condition: Each scarf costs $2.
axiom scarf_price : scarf_cost = 2

-- Condition: Kiki buys twice as many hats as scarves.
axiom hat_scarf_relationship : hats = 2 * scarves

theorem kiki_scarves_count 
  (kiki_money : money = 90)
  (kiki_hat_spending_ratio : hat_spending_ratio = 0.60)
  (scarf_price : scarf_cost = 2)
  (hat_scarf_relationship : hats = 2 * scarves)
  : scarves = 18 := 
sorry

end kiki_scarves_count_l184_18425


namespace person_A_work_days_l184_18484

theorem person_A_work_days (x : ℝ) (h1 : 0 < x) 
                                 (h2 : ∃ b_work_rate, b_work_rate = 1 / 30) 
                                 (h3 : 5 * (1 / x + 1 / 30) = 0.5) : 
  x = 15 :=
by
-- Proof omitted
sorry

end person_A_work_days_l184_18484


namespace option_C_incorrect_l184_18445

variable (a b : ℝ)

theorem option_C_incorrect : ((-a^3)^2 * (-b^2)^3) ≠ (a^6 * b^6) :=
by {
  sorry
}

end option_C_incorrect_l184_18445


namespace parallel_lines_count_l184_18423

theorem parallel_lines_count (n : ℕ) (h : 7 * (n - 1) = 588) : n = 85 :=
sorry

end parallel_lines_count_l184_18423


namespace sum_of_possible_values_of_N_l184_18482

theorem sum_of_possible_values_of_N :
  ∃ a b c : ℕ, (a > 0 ∧ b > 0 ∧ c > 0) ∧ (abc = 8 * (a + b + c)) ∧ (c = a + b)
  ∧ (2560 = 560) :=
by
  sorry

end sum_of_possible_values_of_N_l184_18482


namespace percent_increase_l184_18428

variable (E : ℝ)

-- Given conditions
def enrollment_1992 := 1.20 * E
def enrollment_1993 := 1.26 * E

-- Theorem to prove
theorem percent_increase :
  ((enrollment_1993 E - enrollment_1992 E) / enrollment_1992 E) * 100 = 5 := by
  sorry

end percent_increase_l184_18428


namespace factorization_of_polynomial_l184_18474

theorem factorization_of_polynomial :
  ∀ x : ℝ, (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) = (x - 1)^4 :=
by
  intro x
  sorry

end factorization_of_polynomial_l184_18474


namespace possible_values_l184_18478

theorem possible_values (a b : ℕ → ℕ) (h1 : ∀ n, a n < (a (n + 1)))
  (h2 : ∀ n, b n < (b (n + 1)))
  (h3 : a 10 = b 10)
  (h4 : a 10 < 2017)
  (h5 : ∀ n, a (n + 2) = a (n + 1) + a n)
  (h6 : ∀ n, b (n + 1) = 2 * b n) :
  ∃ (a1 b1 : ℕ), (a 1 = a1) ∧ (b 1 = b1) ∧ (a1 + b1 = 13 ∨ a1 + b1 = 20) := sorry

end possible_values_l184_18478


namespace jacqueline_has_29_percent_more_soda_than_liliane_l184_18433

variable (A : ℝ) -- A is the amount of soda Alice has

-- Define the amount of soda Jacqueline has
def J (A : ℝ) : ℝ := 1.80 * A

-- Define the amount of soda Liliane has
def L (A : ℝ) : ℝ := 1.40 * A

-- The statement that needs to be proven
theorem jacqueline_has_29_percent_more_soda_than_liliane (A : ℝ) (hA : A > 0) : 
  ((J A - L A) / L A) * 100 = 29 :=
by
  sorry

end jacqueline_has_29_percent_more_soda_than_liliane_l184_18433


namespace total_cars_produced_l184_18401

def CarCompanyA_NorthAmerica := 3884
def CarCompanyA_Europe := 2871
def CarCompanyA_Asia := 1529

def CarCompanyB_NorthAmerica := 4357
def CarCompanyB_Europe := 3690
def CarCompanyB_Asia := 1835

def CarCompanyC_NorthAmerica := 2937
def CarCompanyC_Europe := 4210
def CarCompanyC_Asia := 977

def TotalNorthAmerica :=
  CarCompanyA_NorthAmerica + CarCompanyB_NorthAmerica + CarCompanyC_NorthAmerica

def TotalEurope :=
  CarCompanyA_Europe + CarCompanyB_Europe + CarCompanyC_Europe

def TotalAsia :=
  CarCompanyA_Asia + CarCompanyB_Asia + CarCompanyC_Asia

def TotalProduction := TotalNorthAmerica + TotalEurope + TotalAsia

theorem total_cars_produced : TotalProduction = 26290 := 
by sorry

end total_cars_produced_l184_18401


namespace michael_total_cost_l184_18475

def rental_fee : ℝ := 20.99
def charge_per_mile : ℝ := 0.25
def miles_driven : ℕ := 299

def total_cost (rental_fee : ℝ) (charge_per_mile : ℝ) (miles_driven : ℕ) : ℝ :=
  rental_fee + (charge_per_mile * miles_driven)

theorem michael_total_cost :
  total_cost rental_fee charge_per_mile miles_driven = 95.74 :=
by
  sorry

end michael_total_cost_l184_18475


namespace multiplication_addition_example_l184_18456

theorem multiplication_addition_example :
  469138 * 9999 + 876543 * 12345 = 15512230997 :=
by
  sorry

end multiplication_addition_example_l184_18456


namespace lauren_total_money_made_is_correct_l184_18400

-- Define the rate per commercial view
def rate_per_commercial_view : ℝ := 0.50
-- Define the rate per subscriber
def rate_per_subscriber : ℝ := 1.00
-- Define the number of commercial views on Tuesday
def commercial_views : ℕ := 100
-- Define the number of new subscribers on Tuesday
def subscribers : ℕ := 27
-- Calculate the total money Lauren made on Tuesday
def total_money_made (rate_com_view : ℝ) (rate_sub : ℝ) (com_views : ℕ) (subs : ℕ) : ℝ :=
  (rate_com_view * com_views) + (rate_sub * subs)

-- Theorem stating that the total money Lauren made on Tuesday is $77.00
theorem lauren_total_money_made_is_correct : total_money_made rate_per_commercial_view rate_per_subscriber commercial_views subscribers = 77.00 :=
by
  sorry

end lauren_total_money_made_is_correct_l184_18400


namespace find_p_l184_18479

theorem find_p (m n p : ℝ)
  (h1 : m = 4 * n + 5)
  (h2 : m + 2 = 4 * (n + p) + 5) : 
  p = 1 / 2 :=
sorry

end find_p_l184_18479


namespace volume_of_rectangular_prism_l184_18408

theorem volume_of_rectangular_prism (a b c : ℝ)
  (h1 : a * b = Real.sqrt 2)
  (h2 : b * c = Real.sqrt 3)
  (h3 : a * c = Real.sqrt 6) :
  a * b * c = Real.sqrt 6 := by
sorry

end volume_of_rectangular_prism_l184_18408


namespace coordinates_of_B_l184_18409

/--
Given point A with coordinates (2, -3) and line segment AB parallel to the x-axis,
and the length of AB being 4, prove that the coordinates of point B are either (-2, -3)
or (6, -3).
-/
theorem coordinates_of_B (x1 y1 : ℝ) (d : ℝ) (h1 : x1 = 2) (h2 : y1 = -3) (h3 : d = 4) (hx : 0 ≤ d) :
  ∃ x2 : ℝ, ∃ y2 : ℝ, (y2 = y1) ∧ ((x2 = x1 + d) ∨ (x2 = x1 - d)) :=
by
  sorry

end coordinates_of_B_l184_18409


namespace largest_divisor_product_of_consecutive_odds_l184_18444

theorem largest_divisor_product_of_consecutive_odds (n : ℕ) (h : Even n) (h_pos : 0 < n) : 
  15 ∣ (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) :=
sorry

end largest_divisor_product_of_consecutive_odds_l184_18444


namespace number_in_marked_square_is_10_l184_18480

theorem number_in_marked_square_is_10 : 
  ∃ f : ℕ × ℕ → ℕ, 
    (f (0,0) = 5 ∧ f (0,1) = 6 ∧ f (0,2) = 7) ∧ 
    (∀ r c, r > 0 → 
      f (r,c) = f (r-1,c) + f (r-1,c+1)) 
    ∧ f (1, 1) = 13 
    ∧ f (2, 1) = 10 :=
    sorry

end number_in_marked_square_is_10_l184_18480


namespace john_and_lisa_meet_at_midpoint_l184_18460

-- Define the conditions
def john_position : ℝ × ℝ := (2, 9)
def lisa_position : ℝ × ℝ := (-6, 1)

-- Assertion for their meeting point
theorem john_and_lisa_meet_at_midpoint :
  ∃ (x y : ℝ), (x, y) = ((john_position.1 + lisa_position.1) / 2,
                         (john_position.2 + lisa_position.2) / 2) :=
sorry

end john_and_lisa_meet_at_midpoint_l184_18460


namespace rainfall_ratio_l184_18448

theorem rainfall_ratio (S M T : ℝ) (h1 : M = S + 3) (h2 : S = 4) (h3 : S + M + T = 25) : T / M = 2 :=
by
  sorry

end rainfall_ratio_l184_18448


namespace assignment_schemes_correct_l184_18435

-- Define the total number of students
def total_students : ℕ := 6

-- Define the total number of tasks
def total_tasks : ℕ := 4

-- Define a predicate that checks if a student can be assigned to task A
def can_assign_to_task_A (student : ℕ) : Prop := student ≠ 1 ∧ student ≠ 2

-- Calculate the total number of unrestricted assignments
def total_unrestricted_assignments : ℕ := 6 * 5 * 4 * 3

-- Calculate the restricted number of assignments if student A or B is assigned to task A
def restricted_assignments : ℕ := 2 * 5 * 4 * 3

-- Define the problem statement
def number_of_assignment_schemes : ℕ :=
  total_unrestricted_assignments - restricted_assignments

-- The theorem to prove
theorem assignment_schemes_correct :
  number_of_assignment_schemes = 240 :=
by
  -- We acknowledge the problem statement is correct
  sorry

end assignment_schemes_correct_l184_18435


namespace find_a_and_b_l184_18402

theorem find_a_and_b (a b : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = x^3 - a * x^2 - b * x + a^2) →
  f 1 = 10 →
  deriv f 1 = 0 →
  (a = -4 ∧ b = 11) :=
by
  intros hf hf1 hderiv
  sorry

end find_a_and_b_l184_18402


namespace infinite_primes_of_form_m2_mn_n2_l184_18469

theorem infinite_primes_of_form_m2_mn_n2 : ∀ m n : ℤ, ∃ p : ℕ, ∃ k : ℕ, (p = k^2 + k * m + n^2) ∧ Prime k :=
sorry

end infinite_primes_of_form_m2_mn_n2_l184_18469


namespace log9_log11_lt_one_l184_18467

theorem log9_log11_lt_one (log9_pos : 0 < Real.log 9) (log11_pos : 0 < Real.log 11) : 
  Real.log 9 * Real.log 11 < 1 :=
by
  sorry

end log9_log11_lt_one_l184_18467


namespace curves_intersect_at_three_points_l184_18418

theorem curves_intersect_at_three_points :
  (∀ x y a : ℝ, (x^2 + y^2 = 4 * a^2) ∧ (y = x^2 - 2 * a) → a = 1) := sorry

end curves_intersect_at_three_points_l184_18418


namespace find_constants_l184_18421

-- Given definitions based on the conditions and conjecture
def S (n : ℕ) : ℕ := 
  match n with
  | 1 => 1
  | 2 => 5
  | 3 => 15
  | 4 => 34
  | 5 => 65
  | _ => 0

noncomputable def conjecture_S (n a b c : ℤ) := (2 * n - 1) * (a * n^2 + b * n + c)

theorem find_constants (a b c : ℤ) (h1 : conjecture_S 1 a b c = 1) (h2 : conjecture_S 2 a b c = 5) (h3 : conjecture_S 3 a b c = 15) : 3 * a + b = 4 :=
by
  -- Proof omitted
  sorry

end find_constants_l184_18421


namespace P_zero_eq_zero_l184_18470

open Polynomial

noncomputable def P (x : ℝ) : ℝ := sorry

axiom distinct_roots : ∃ y : Fin 17 → ℝ, Function.Injective y ∧ ∀ i, P (y i ^ 2) = 0

theorem P_zero_eq_zero : P 0 = 0 :=
by
  sorry

end P_zero_eq_zero_l184_18470


namespace smallest_number_diminished_by_2_divisible_12_16_18_21_28_l184_18424

def conditions_holds (n : ℕ) : Prop :=
  (n - 2) % 12 = 0 ∧ (n - 2) % 16 = 0 ∧ (n - 2) % 18 = 0 ∧ (n - 2) % 21 = 0 ∧ (n - 2) % 28 = 0

theorem smallest_number_diminished_by_2_divisible_12_16_18_21_28 :
  ∃ (n : ℕ), conditions_holds n ∧ (∀ m, conditions_holds m → n ≤ m) ∧ n = 1009 :=
by
  sorry

end smallest_number_diminished_by_2_divisible_12_16_18_21_28_l184_18424


namespace four_times_num_mod_nine_l184_18487

theorem four_times_num_mod_nine (n : ℤ) (h : n % 9 = 4) : (4 * n - 3) % 9 = 4 :=
sorry

end four_times_num_mod_nine_l184_18487


namespace sum_of_fourth_powers_eq_174_fourth_l184_18413

theorem sum_of_fourth_powers_eq_174_fourth :
  120 ^ 4 + 97 ^ 4 + 84 ^ 4 + 27 ^ 4 = 174 ^ 4 :=
by
  sorry

end sum_of_fourth_powers_eq_174_fourth_l184_18413


namespace specialPermutationCount_l184_18477

def countSpecialPerms (n : ℕ) : ℕ := 2 ^ (n - 1)

theorem specialPermutationCount (n : ℕ) : 
  (countSpecialPerms n = 2 ^ (n - 1)) := 
by 
  sorry

end specialPermutationCount_l184_18477


namespace players_taking_chemistry_l184_18492

theorem players_taking_chemistry (total_players biology_players both_sci_players: ℕ) 
  (h1 : total_players = 12)
  (h2 : biology_players = 7)
  (h3 : both_sci_players = 2)
  (h4 : ∀ p, p <= total_players) : 
  ∃ chemistry_players, chemistry_players = 7 := 
sorry

end players_taking_chemistry_l184_18492


namespace radius_range_of_sector_l184_18429

theorem radius_range_of_sector (a : ℝ) (h : a > 0) :
  ∃ (R : ℝ), (a / (2 * (1 + π)) < R ∧ R < a / 2) :=
sorry

end radius_range_of_sector_l184_18429


namespace find_a_range_l184_18497

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then |x| + 2 else x + 2 / x

theorem find_a_range (a : ℝ) :
  (∀ x : ℝ, f x ≥ |x / 2 + a|) ↔ (-2 ≤ a ∧ a ≤ 2) :=
by
  sorry

end find_a_range_l184_18497


namespace largest_angle_in_convex_pentagon_l184_18489

theorem largest_angle_in_convex_pentagon (x : ℕ) (h : (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 540) : 
  x + 2 = 110 :=
by
  sorry

end largest_angle_in_convex_pentagon_l184_18489


namespace ab_zero_if_conditions_l184_18490

theorem ab_zero_if_conditions 
  (a b : ℤ)
  (h : |a - b| + |a * b| = 2) : a * b = 0 :=
  sorry

end ab_zero_if_conditions_l184_18490


namespace lena_more_than_nicole_l184_18447

theorem lena_more_than_nicole :
  ∀ (L K N : ℝ),
    L = 37.5 →
    (L + 9.5) = 5 * K →
    K = N - 8.5 →
    (L - N) = 19.6 :=
by
  intros L K N hL hLK hK
  sorry

end lena_more_than_nicole_l184_18447


namespace residue_of_neg_1001_mod_37_l184_18488

theorem residue_of_neg_1001_mod_37 : (-1001 : ℤ) % 37 = 35 :=
by
  sorry

end residue_of_neg_1001_mod_37_l184_18488


namespace percent_of_b_is_50_l184_18494

variable (a b c : ℝ)

-- Conditions
def c_is_25_percent_of_a : Prop := c = 0.25 * a
def b_is_50_percent_of_a : Prop := b = 0.50 * a

-- Proof
theorem percent_of_b_is_50 :
  c_is_25_percent_of_a c a → b_is_50_percent_of_a b a → c = 0.50 * b :=
by sorry

end percent_of_b_is_50_l184_18494


namespace B_work_days_l184_18436

/-- 
  A and B undertake to do a piece of work for $500.
  A alone can do it in 5 days while B alone can do it in a certain number of days.
  With the help of C, they finish it in 2 days. C's share is $200.
  Prove B alone can do the work in 10 days.
-/
theorem B_work_days (x : ℕ) (h1 : (1/5 : ℝ) + (1/x : ℝ) = 3/10) : x = 10 := 
  sorry

end B_work_days_l184_18436


namespace second_percentage_increase_l184_18450

theorem second_percentage_increase 
  (P : ℝ) 
  (x : ℝ) 
  (h1: 1.20 * P * (1 + x / 100) = 1.38 * P) : 
  x = 15 := 
  sorry

end second_percentage_increase_l184_18450


namespace equal_total_areas_of_checkerboard_pattern_l184_18496

-- Definition representing the convex quadrilateral and its subdivisions
structure ConvexQuadrilateral :=
  (A B C D : ℝ × ℝ) -- vertices of the quadrilateral

-- Predicate indicating the subdivision and coloring pattern
inductive CheckerboardColor
  | Black
  | White

-- Function to determine the area of the resulting smaller quadrilateral
noncomputable def area_of_subquadrilateral 
  (quad : ConvexQuadrilateral) 
  (subdivision : ℕ) -- subdivision factor
  (color : CheckerboardColor) 
  : ℝ := -- returns the area based on the subdivision and color
  -- Simplified implementation of area calculation
  -- (detailed geometric computation should replace this placeholder)
  sorry

-- Function to determine the total area of quadrilaterals of a given color
noncomputable def total_area_of_color 
  (quad : ConvexQuadrilateral) 
  (substution : ℕ) 
  (color : CheckerboardColor) 
  : ℝ := -- Total area of subquadrilaterals of the given color
  sorry

-- Theorem stating the required proof
theorem equal_total_areas_of_checkerboard_pattern
  (quad : ConvexQuadrilateral)
  (subdivision : ℕ)
  : total_area_of_color quad subdivision CheckerboardColor.Black = total_area_of_color quad subdivision CheckerboardColor.White :=
  sorry

end equal_total_areas_of_checkerboard_pattern_l184_18496


namespace Darren_paints_432_feet_l184_18451

theorem Darren_paints_432_feet (t : ℝ) (h : t = 792) (paint_ratio : ℝ) 
  (h_ratio : paint_ratio = 1.20) : 
  let d := t / (1 + paint_ratio)
  let D := d * paint_ratio
  D = 432 :=
by
  sorry

end Darren_paints_432_feet_l184_18451


namespace geom_seq_sum_l184_18431

noncomputable def geom_seq (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
a₁ * r^(n-1)

theorem geom_seq_sum (a₁ r : ℝ) (h_pos : 0 < a₁) (h_pos_r : 0 < r)
  (h : a₁ * (geom_seq a₁ r 5) + 2 * (geom_seq a₁ r 3) * (geom_seq a₁ r 6) + a₁ * (geom_seq a₁ r 11) = 16) :
  (geom_seq a₁ r 3 + geom_seq a₁ r 6) = 4 :=
sorry

end geom_seq_sum_l184_18431


namespace find_nat_int_l184_18417

theorem find_nat_int (x y : ℕ) (h : x^2 = y^2 + 7 * y + 6) : x = 6 ∧ y = 3 := 
by
  sorry

end find_nat_int_l184_18417


namespace intersection_of_asymptotes_l184_18481

theorem intersection_of_asymptotes :
  ∃ x y : ℝ, (y = 1) ∧ (x = 3) ∧ (y = (x^2 - 6*x + 8) / (x^2 - 6*x + 9)) := 
by {
  sorry
}

end intersection_of_asymptotes_l184_18481


namespace area_of_complex_polygon_l184_18419

-- Defining the problem
def area_of_polygon (side1 side2 side3 : ℝ) (rot1 rot2 : ℝ) : ℝ :=
  -- This is a placeholder definition.
  -- In a complete proof, here we would calculate the area based on the input conditions.
  sorry

-- Main theorem statement
theorem area_of_complex_polygon :
  area_of_polygon 4 5 6 (π / 4) (-π / 6) = 72 :=
by sorry

end area_of_complex_polygon_l184_18419


namespace total_students_l184_18439

-- Definitions extracted from the conditions 
def ratio_boys_girls := 8 / 5
def number_of_boys := 128

-- Theorem to prove the total number of students
theorem total_students : 
  (128 + (5 / 8) * 128 = 208) ∧ ((128 : ℝ) * (13 / 8) = 208) :=
by
  sorry

end total_students_l184_18439


namespace division_criterion_based_on_stroke_l184_18411

-- Definition of a drawable figure with a single stroke
def drawable_in_one_stroke (figure : Type) : Prop := sorry -- exact conditions can be detailed with figure representation

-- Example figures for the groups (types can be extended based on actual representation)
def Group1 := {fig1 : Type // drawable_in_one_stroke fig1}
def Group2 := {fig2 : Type // ¬drawable_in_one_stroke fig2}

-- Problem Statement:
theorem division_criterion_based_on_stroke (fig : Type) :
  (drawable_in_one_stroke fig ∨ ¬drawable_in_one_stroke fig) := by
  -- We state that every figure belongs to either Group1 or Group2
  sorry

end division_criterion_based_on_stroke_l184_18411


namespace diana_total_earnings_l184_18406

-- Define the earnings in each month
def july_earnings : ℕ := 150
def august_earnings : ℕ := 3 * july_earnings
def september_earnings : ℕ := 2 * august_earnings

-- State the theorem that the total earnings over the three months is $1500
theorem diana_total_earnings : july_earnings + august_earnings + september_earnings = 1500 :=
by
  have h1 : august_earnings = 3 * july_earnings := rfl
  have h2 : september_earnings = 2 * august_earnings := rfl
  sorry

end diana_total_earnings_l184_18406


namespace negation_of_universal_prop_l184_18446

theorem negation_of_universal_prop :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by
  sorry

end negation_of_universal_prop_l184_18446


namespace ted_gathered_10_blue_mushrooms_l184_18466

noncomputable def blue_mushrooms_ted_gathered : ℕ :=
  let bill_red_mushrooms := 12
  let bill_brown_mushrooms := 6
  let ted_green_mushrooms := 14
  let total_white_spotted_mushrooms := 17
  
  let bill_white_spotted_red_mushrooms := bill_red_mushrooms / 2
  let bill_white_spotted_brown_mushrooms := bill_brown_mushrooms

  let total_bill_white_spotted_mushrooms := bill_white_spotted_red_mushrooms + bill_white_spotted_brown_mushrooms
  let ted_white_spotted_mushrooms := total_white_spotted_mushrooms - total_bill_white_spotted_mushrooms

  ted_white_spotted_mushrooms * 2

theorem ted_gathered_10_blue_mushrooms :
  blue_mushrooms_ted_gathered = 10 :=
by
  sorry

end ted_gathered_10_blue_mushrooms_l184_18466


namespace least_number_to_add_l184_18440

theorem least_number_to_add (n : ℕ) (h₁ : n = 1054) :
  ∃ k : ℕ, (n + k) % 23 = 0 ∧ k = 4 :=
by
  use 4
  have h₂ : n % 23 = 19 := by sorry
  have h₃ : (n + 4) % 23 = 0 := by sorry
  exact ⟨h₃, rfl⟩

end least_number_to_add_l184_18440


namespace max_students_distributing_pens_and_pencils_l184_18465

theorem max_students_distributing_pens_and_pencils :
  Nat.gcd 1001 910 = 91 :=
by
  -- remaining proof required
  sorry

end max_students_distributing_pens_and_pencils_l184_18465


namespace trapezoid_area_l184_18405

-- Definitions based on the problem conditions
def Vertex := (Real × Real)

structure Triangle :=
(A : Vertex)
(B : Vertex)
(C : Vertex)
(area : Real)

structure Trapezoid :=
(AB : Real)
(CD : Real)
(M : Vertex)
(area_triangle_ABM : Real)
(area_triangle_CDM : Real)

-- The main theorem we want to prove
theorem trapezoid_area (T : Trapezoid)
  (parallel_sides : T.AB < T.CD)
  (intersect_at_M : ∃ M : Vertex, M = T.M)
  (area_ABM : T.area_triangle_ABM = 2)
  (area_CDM : T.area_triangle_CDM = 8) :
  T.AB * T.CD / (T.CD - T.AB) + T.CD * T.AB / (T.CD - T.AB) = 18 :=
sorry

end trapezoid_area_l184_18405


namespace sum_of_integers_mod_59_l184_18426

theorem sum_of_integers_mod_59 (a b c : ℕ) (h1 : a % 59 = 29) (h2 : b % 59 = 31) (h3 : c % 59 = 7)
  (h4 : a^2 % 59 = 29) (h5 : b^2 % 59 = 31) (h6 : c^2 % 59 = 7) :
  (a + b + c) % 59 = 8 :=
by
  sorry

end sum_of_integers_mod_59_l184_18426


namespace smaller_cube_volume_l184_18459

theorem smaller_cube_volume
  (V_L : ℝ) (N : ℝ) (SA_diff : ℝ) 
  (h1 : V_L = 8)
  (h2 : N = 8)
  (h3 : SA_diff = 24) :
  (∀ V_S : ℝ, V_L = N * V_S → V_S = 1) :=
by
  sorry

end smaller_cube_volume_l184_18459


namespace probability_red_ball_l184_18493

def total_balls : ℕ := 3
def red_balls : ℕ := 1
def yellow_balls : ℕ := 2

theorem probability_red_ball : (red_balls : ℚ) / (total_balls : ℚ) = 1 / 3 :=
by
  sorry

end probability_red_ball_l184_18493


namespace intersection_A_B_l184_18434

def A (x : ℝ) : Prop := (x ≥ 2 ∧ x ≠ 3)
def B (x : ℝ) : Prop := (3 ≤ x ∧ x ≤ 5)
def C := {x : ℝ | 3 < x ∧ x ≤ 5}

theorem intersection_A_B : {x : ℝ | A x} ∩ {x : ℝ | B x} = C :=
  by sorry

end intersection_A_B_l184_18434


namespace length_of_box_l184_18495

theorem length_of_box 
  (width height num_cubes length : ℕ)
  (h_width : width = 16)
  (h_height : height = 13)
  (h_cubes : num_cubes = 3120)
  (h_volume : length * width * height = num_cubes) :
  length = 15 :=
by
  sorry

end length_of_box_l184_18495


namespace nearby_island_banana_production_l184_18420

theorem nearby_island_banana_production
  (x : ℕ)
  (h_prod: 10 * x + x = 99000) :
  x = 9000 :=
sorry

end nearby_island_banana_production_l184_18420


namespace N_perfect_square_l184_18438

theorem N_perfect_square (N : ℕ) (hN_pos : N > 0) 
  (h_pairs : ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 2005 ∧ 
  ∀ p ∈ pairs, (1 : ℚ) / (p.1 : ℚ) + (1 : ℚ) / (p.2 : ℚ) = (1 : ℚ) / N ∧ p.1 > 0 ∧ p.2 > 0) : 
  ∃ k : ℕ, N = k^2 := 
sorry

end N_perfect_square_l184_18438


namespace proof_one_third_of_seven_times_nine_subtract_three_l184_18416

def one_third_of_seven_times_nine_subtract_three : ℕ :=
  let product := 7 * 9
  let one_third := product / 3
  one_third - 3

theorem proof_one_third_of_seven_times_nine_subtract_three : one_third_of_seven_times_nine_subtract_three = 18 := by
  sorry

end proof_one_third_of_seven_times_nine_subtract_three_l184_18416


namespace f_2018_eq_2017_l184_18498

-- Define f(1) and f(2)
def f : ℕ → ℕ 
| 1 => 1
| 2 => 1
| n => if h : n ≥ 3 then (f (n - 1) - f (n - 2) + n) else 0

-- State the theorem to prove f(2018) = 2017
theorem f_2018_eq_2017 : f 2018 = 2017 := 
by 
  sorry

end f_2018_eq_2017_l184_18498


namespace age_difference_is_eight_l184_18432

theorem age_difference_is_eight (A B k : ℕ)
  (h1 : A = B + k)
  (h2 : A - 1 = 3 * (B - 1))
  (h3 : A = 2 * B + 3) :
  k = 8 :=
by sorry

end age_difference_is_eight_l184_18432


namespace sufficient_drivers_and_completion_time_l184_18410

noncomputable def one_way_trip_minutes : ℕ := 2 * 60 + 40
noncomputable def round_trip_minutes : ℕ := 2 * one_way_trip_minutes
noncomputable def rest_period_minutes : ℕ := 60
noncomputable def twelve_forty_pm : ℕ := 12 * 60 + 40 -- in minutes from midnight
noncomputable def one_forty_pm : ℕ := twelve_forty_pm + rest_period_minutes
noncomputable def thirteen_five_pm : ℕ := 13 * 60 + 5 -- 1:05 PM
noncomputable def sixteen_ten_pm : ℕ := 16 * 60 + 10 -- 4:10 PM
noncomputable def sixteen_pm : ℕ := 16 * 60 -- 4:00 PM
noncomputable def seventeen_thirty_pm : ℕ := 17 * 60 + 30 -- 5:30 PM
noncomputable def twenty_one_thirty_pm : ℕ := sixteen_ten_pm + round_trip_minutes -- 9:30 PM (21:30)

theorem sufficient_drivers_and_completion_time :
  4 = 4 ∧ twenty_one_thirty_pm = 21 * 60 + 30 := by
  sorry 

end sufficient_drivers_and_completion_time_l184_18410


namespace inequality_solution_l184_18472

noncomputable def inequality_proof (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 2) : Prop :=
  (1 / (1 + a * b) + 1 / (1 + b * c) + 1 / (1 + c * a)) ≥ (27 / 13)

theorem inequality_solution (a b c : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a + b + c = 2) : 
  inequality_proof a b c h_positive h_sum :=
sorry

end inequality_solution_l184_18472


namespace eval_infinite_product_l184_18499

noncomputable def infinite_product : ℝ :=
  ∏' n : ℕ, (3:ℝ)^(2 * n / (3:ℝ)^n)

theorem eval_infinite_product : infinite_product = (3:ℝ)^(9 / 2) := by
  sorry

end eval_infinite_product_l184_18499


namespace gardening_project_total_cost_l184_18464

noncomputable def cost_gardening_project : ℕ := 
  let number_rose_bushes := 20
  let cost_per_rose_bush := 150
  let cost_fertilizer_per_bush := 25
  let gardener_work_hours := [6, 5, 4, 7]
  let gardener_hourly_rate := 30
  let soil_amount := 100
  let cost_per_cubic_foot := 5

  let cost_roses := number_rose_bushes * cost_per_rose_bush
  let cost_fertilizer := number_rose_bushes * cost_fertilizer_per_bush
  let total_work_hours := List.sum gardener_work_hours
  let cost_labor := total_work_hours * gardener_hourly_rate
  let cost_soil := soil_amount * cost_per_cubic_foot

  cost_roses + cost_fertilizer + cost_labor + cost_soil

theorem gardening_project_total_cost : cost_gardening_project = 4660 := by
  sorry

end gardening_project_total_cost_l184_18464


namespace batch_production_equation_l184_18414

theorem batch_production_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 20) :
  (500 / x) = (300 / (x - 20)) :=
sorry

end batch_production_equation_l184_18414


namespace container_capacity_l184_18491

theorem container_capacity (C : ℝ) 
  (h1 : 0.30 * C + 18 = 0.75 * C) : 
  C = 40 :=
by
  -- proof steps would go here
  sorry

end container_capacity_l184_18491


namespace find_A_l184_18437

theorem find_A (
  A B C A' r : ℕ
) (hA : A = 312) (hB : B = 270) (hC : C = 211)
  (hremA : A % A' = 4 * r)
  (hremB : B % A' = 2 * r)
  (hremC : C % A' = r) :
  A' = 19 :=
by
  sorry

end find_A_l184_18437


namespace ratio_saturday_friday_l184_18403

variable (S : ℕ)
variable (soldOnFriday : ℕ := 30)
variable (soldOnSunday : ℕ := S - 15)
variable (totalSold : ℕ := 135)

theorem ratio_saturday_friday (h1 : soldOnFriday = 30)
                              (h2 : totalSold = 135)
                              (h3 : soldOnSunday = S - 15)
                              (h4 : soldOnFriday + S + soldOnSunday = totalSold) :
  (S / soldOnFriday) = 2 :=
by
  -- Prove the theorem here...
  sorry

end ratio_saturday_friday_l184_18403


namespace compounding_frequency_l184_18452

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem compounding_frequency (P A r t n : ℝ) 
  (principal : P = 6000) 
  (amount : A = 6615)
  (rate : r = 0.10)
  (time : t = 1) 
  (comp_freq : n = 2) :
  compound_interest P r n t = A := 
by 
  simp [compound_interest, principal, rate, time, comp_freq, amount]
  -- calculations and proof omitted
  sorry

end compounding_frequency_l184_18452


namespace marks_lost_per_incorrect_sum_l184_18454

variables (marks_per_correct : ℕ) (total_attempts total_marks correct_sums : ℕ)
variable (marks_per_incorrect : ℕ)
variable (incorrect_sums : ℕ)

def calc_marks_per_incorrect_sum : Prop :=
  marks_per_correct = 3 ∧ 
  total_attempts = 30 ∧ 
  total_marks = 50 ∧ 
  correct_sums = 22 ∧ 
  incorrect_sums = total_attempts - correct_sums ∧ 
  (marks_per_correct * correct_sums) - (marks_per_incorrect * incorrect_sums) = total_marks ∧ 
  marks_per_incorrect = 2

theorem marks_lost_per_incorrect_sum : calc_marks_per_incorrect_sum 3 30 50 22 2 (30 - 22) :=
sorry

end marks_lost_per_incorrect_sum_l184_18454


namespace number_of_tables_l184_18463

/-- Problem Statement
  In a hall used for a conference, each table is surrounded by 8 stools and 4 chairs. Each stool has 3 legs,
  each chair has 4 legs, and each table has 4 legs. If the total number of legs for all tables, stools, and chairs is 704,
  the number of tables in the hall is 16. -/
theorem number_of_tables (legs_per_stool legs_per_chair legs_per_table total_legs t : ℕ) 
  (Hstools : ∀ tables, stools = 8 * tables)
  (Hchairs : ∀ tables, chairs = 4 * tables)
  (Hlegs : 3 * stools + 4 * chairs + 4 * t = total_legs)
  (Hleg_values : legs_per_stool = 3 ∧ legs_per_chair = 4 ∧ legs_per_table = 4)
  (Htotal_legs : total_legs = 704) :
  t = 16 := by
  sorry

end number_of_tables_l184_18463


namespace sin_x_eq_x_has_unique_root_in_interval_l184_18468

theorem sin_x_eq_x_has_unique_root_in_interval :
  ∃! x : ℝ, x ∈ Set.Icc (-Real.pi) Real.pi ∧ x = Real.sin x :=
sorry

end sin_x_eq_x_has_unique_root_in_interval_l184_18468


namespace infinite_n_multiples_of_six_available_l184_18442

theorem infinite_n_multiples_of_six_available :
  ∃ (S : Set ℕ), (∀ n ∈ S, ∃ (A : Matrix (Fin 3) (Fin (n : ℕ)) Nat),
    (∀ (i : Fin n), (A 0 i + A 1 i + A 2 i) % 6 = 0) ∧ 
    (∀ (i : Fin 3), (Finset.univ.sum (λ j => A i j)) % 6 = 0)) ∧
  Set.Infinite S :=
sorry

end infinite_n_multiples_of_six_available_l184_18442


namespace trigonometric_identity_l184_18453

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := 
by
  sorry

end trigonometric_identity_l184_18453


namespace find_interest_rate_l184_18443

-- Define the conditions
def total_amount : ℝ := 2500
def second_part_rate : ℝ := 0.06
def annual_income : ℝ := 145
def first_part_amount : ℝ := 500.0000000000002
noncomputable def interest_rate (r : ℝ) : Prop :=
  first_part_amount * r + (total_amount - first_part_amount) * second_part_rate = annual_income

-- State the theorem
theorem find_interest_rate : interest_rate 0.05 :=
by
  sorry

end find_interest_rate_l184_18443


namespace necessary_but_not_sufficient_condition_l184_18457
-- Import the required Mathlib library in Lean 4

-- State the equivalent proof problem
theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (|a| ≤ 1 → a ≤ 1) ∧ ¬ (a ≤ 1 → |a| ≤ 1) :=
by
  sorry

end necessary_but_not_sufficient_condition_l184_18457


namespace dumpling_probability_l184_18404

theorem dumpling_probability :
  let total_dumplings := 15
  let choose4 := Nat.choose total_dumplings 4
  let choose1 := Nat.choose 3 1
  let choose5_2 := Nat.choose 5 2
  let choose5_1 := Nat.choose 5 1
  (choose1 * choose5_2 * choose5_1 * choose5_1) / choose4 = 50 / 91 := by
  sorry

end dumpling_probability_l184_18404


namespace wall_length_proof_l184_18473

noncomputable def volume_of_brick (length width height : ℝ) : ℝ := length * width * height

noncomputable def total_volume (brick_volume num_of_bricks : ℝ) : ℝ := brick_volume * num_of_bricks

theorem wall_length_proof
  (height_of_wall : ℝ) (width_of_walls : ℝ) (num_of_bricks : ℝ)
  (length_of_brick width_of_brick height_of_brick : ℝ)
  (total_volume_of_bricks : ℝ) :
  total_volume (volume_of_brick length_of_brick width_of_brick height_of_brick) num_of_bricks = total_volume_of_bricks →
  volume_of_brick length_of_wall height_of_wall width_of_walls = total_volume_of_bricks →
  height_of_wall = 600 →
  width_of_walls = 2 →
  num_of_bricks = 2909.090909090909 →
  length_of_brick = 5 →
  width_of_brick = 11 →
  height_of_brick = 6 →
  total_volume_of_bricks = 960000 →
  length_of_wall = 800 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end wall_length_proof_l184_18473
