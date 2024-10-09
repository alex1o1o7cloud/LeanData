import Mathlib

namespace cost_of_coffee_B_per_kg_l275_27505

-- Define the cost of coffee A per kilogram
def costA : ℝ := 10

-- Define the amount of coffee A used in the mixture
def amountA : ℝ := 240

-- Define the amount of coffee B used in the mixture
def amountB : ℝ := 240

-- Define the total amount of the mixture
def totalAmount : ℝ := 480

-- Define the selling price of the mixture per kilogram
def sellingPrice : ℝ := 11

-- Define the cost of coffee B per kilogram as a variable B
variable (B : ℝ)

-- Define the total cost of the mixture
def totalCost : ℝ := totalAmount * sellingPrice

-- Define the cost of coffee A used
def costOfA : ℝ := amountA * costA

-- Define the cost of coffee B used as total cost minus the cost of A
def costOfB : ℝ := totalCost - costOfA

-- Calculate the cost of coffee B per kilogram
theorem cost_of_coffee_B_per_kg : B = 12 :=
by
  have h1 : costOfA = 2400 := by sorry
  have h2 : totalCost = 5280 := by sorry
  have h3 : costOfB = 2880 := by sorry
  have h4 : B = costOfB / amountB := by sorry
  have h5 : B = 2880 / 240 := by sorry
  have h6 : B = 12 := by sorry
  exact h6

end cost_of_coffee_B_per_kg_l275_27505


namespace peter_has_4_finches_l275_27586

variable (parakeet_eats_per_day : ℕ) (parrot_eats_per_day : ℕ) (finch_eats_per_day : ℕ)
variable (num_parakeets : ℕ) (num_parrots : ℕ) (num_finches : ℕ)
variable (total_birdseed : ℕ)

theorem peter_has_4_finches
    (h1 : parakeet_eats_per_day = 2)
    (h2 : parrot_eats_per_day = 14)
    (h3 : finch_eats_per_day = 1)
    (h4 : num_parakeets = 3)
    (h5 : num_parrots = 2)
    (h6 : total_birdseed = 266)
    (h7 : total_birdseed = (num_parakeets * parakeet_eats_per_day + num_parrots * parrot_eats_per_day) * 7 + num_finches * finch_eats_per_day * 7) :
    num_finches = 4 :=
by
  sorry

end peter_has_4_finches_l275_27586


namespace rahul_spends_10_percent_on_clothes_l275_27570

theorem rahul_spends_10_percent_on_clothes 
    (salary : ℝ) (house_rent_percent : ℝ) (education_percent : ℝ) (remaining_after_expense : ℝ) (expenses : ℝ) (clothes_percent : ℝ) 
    (h_salary : salary = 2125) 
    (h_house_rent_percent : house_rent_percent = 0.20)
    (h_education_percent : education_percent = 0.10)
    (h_remaining_after_expense : remaining_after_expense = 1377)
    (h_expenses : expenses = salary * house_rent_percent + (salary - salary * house_rent_percent) * education_percent + (salary - salary * house_rent_percent - (salary - salary * house_rent_percent) * education_percent) * clothes_percent)
    (h_clothes_expense : remaining_after_expense = salary - (salary * house_rent_percent + (salary - salary * house_rent_percent) * education_percent + (salary - salary * house_rent_percent - (salary - salary * house_rent_percent) * education_percent) * clothes_percent)) :
    clothes_percent = 0.10 := 
by 
  sorry

end rahul_spends_10_percent_on_clothes_l275_27570


namespace number_of_sandwiches_l275_27563

-- Definitions based on conditions
def breads : Nat := 5
def meats : Nat := 7
def cheeses : Nat := 6
def total_sandwiches : Nat := breads * meats * cheeses
def turkey_mozzarella_exclusions : Nat := breads
def rye_beef_exclusions : Nat := cheeses

-- The proof problem statement
theorem number_of_sandwiches (total_sandwiches := 210) 
  (turkey_mozzarella_exclusions := 5) 
  (rye_beef_exclusions := 6) : 
  total_sandwiches - turkey_mozzarella_exclusions - rye_beef_exclusions = 199 := 
by sorry

end number_of_sandwiches_l275_27563


namespace index_card_area_reduction_l275_27555

theorem index_card_area_reduction :
  ∀ (length width : ℕ),
  (length = 5 ∧ width = 7) →
  ((length - 2) * width = 21) →
  (length * (width - 2) = 25) :=
by
  intros length width h1 h2
  rcases h1 with ⟨h_length, h_width⟩
  sorry

end index_card_area_reduction_l275_27555


namespace point_distance_units_l275_27538

theorem point_distance_units (d : ℝ) (h : |d| = 4) : d = 4 ∨ d = -4 := 
sorry

end point_distance_units_l275_27538


namespace right_triangle_area_l275_27588

theorem right_triangle_area (a : ℝ) (r : ℝ) (area : ℝ) :
  a = 3 → r = 3 / 8 → area = 21 / 16 :=
by 
  sorry

end right_triangle_area_l275_27588


namespace subset_condition_l275_27507

noncomputable def A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}

def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem subset_condition (m : ℝ) : (B m ⊆ A) ↔ m ≤ 3 :=
sorry

end subset_condition_l275_27507


namespace find_divisor_l275_27559

theorem find_divisor (Q R D V : ℤ) (hQ : Q = 65) (hR : R = 5) (hV : V = 1565) (hEquation : V = D * Q + R) : D = 24 :=
by
  sorry

end find_divisor_l275_27559


namespace problem_l275_27514

theorem problem (a b : ℝ) :
  (∀ x : ℝ, 3 * x - 1 ≤ a ∧ 2 * x ≥ 6 - b → -1 ≤ x ∧ x ≤ 2) →
  a + b = 13 := by
  sorry

end problem_l275_27514


namespace gcd_459_357_l275_27585

theorem gcd_459_357 :
  Nat.gcd 459 357 = 51 :=
by
  sorry

end gcd_459_357_l275_27585


namespace dice_probability_l275_27574

theorem dice_probability (D1 D2 D3 : ℕ) (hD1 : 0 ≤ D1) (hD1' : D1 < 10) (hD2 : 0 ≤ D2) (hD2' : D2 < 10) (hD3 : 0 ≤ D3) (hD3' : D3 < 10) :
  ∃ p : ℚ, p = 1 / 10 :=
by
  let outcomes := 10 * 10 * 10
  let favorable := 100
  let expected_probability : ℚ := favorable / outcomes
  use expected_probability
  sorry

end dice_probability_l275_27574


namespace dangerous_animals_remaining_in_swamp_l275_27556

-- Define the initial counts of each dangerous animals
def crocodiles_initial := 42
def alligators_initial := 35
def vipers_initial := 10
def water_moccasins_initial := 28
def cottonmouth_snakes_initial := 15
def piranha_fish_initial := 120

-- Define the counts of migrating animals
def crocodiles_migrating := 9
def alligators_migrating := 7
def vipers_migrating := 3

-- Define the total initial dangerous animals
def total_initial : Nat :=
  crocodiles_initial + alligators_initial + vipers_initial + water_moccasins_initial + cottonmouth_snakes_initial + piranha_fish_initial

-- Define the total migrating dangerous animals
def total_migrating : Nat :=
  crocodiles_migrating + alligators_migrating + vipers_migrating

-- Define the total remaining dangerous animals
def total_remaining : Nat :=
  total_initial - total_migrating

theorem dangerous_animals_remaining_in_swamp :
  total_remaining = 231 :=
by
  -- simply using the calculation we know
  sorry

end dangerous_animals_remaining_in_swamp_l275_27556


namespace average_age_l275_27552

theorem average_age (Jared Molly Hakimi : ℕ) (h1 : Jared = Hakimi + 10) (h2 : Molly = 30) (h3 : Hakimi = 40) :
  (Jared + Molly + Hakimi) / 3 = 40 :=
by
  sorry

end average_age_l275_27552


namespace min_value_of_expression_l275_27524

open Classical

theorem min_value_of_expression (x : ℝ) (hx : x > 0) : 
  ∃ y, x + 16 / (x + 1) = y ∧ ∀ z, (z > 0 → z + 16 / (z + 1) ≥ y) := 
by
  use 7
  sorry

end min_value_of_expression_l275_27524


namespace find_divisor_l275_27532

-- Definitions
def dividend := 199
def quotient := 11
def remainder := 1

-- Statement of the theorem
theorem find_divisor : ∃ x : ℕ, dividend = (x * quotient) + remainder ∧ x = 18 := by
  sorry

end find_divisor_l275_27532


namespace evaluate_101_times_101_l275_27547

theorem evaluate_101_times_101 : (101 * 101 = 10201) :=
by {
  sorry
}

end evaluate_101_times_101_l275_27547


namespace average_test_score_of_remainder_l275_27554

variable (score1 score2 score3 totalAverage : ℝ)
variable (percentage1 percentage2 percentage3 : ℝ)

def equation (score1 score2 score3 totalAverage : ℝ) (percentage1 percentage2 percentage3: ℝ) : Prop :=
  (percentage1 * score1) + (percentage2 * score2) + (percentage3 * score3) = totalAverage

theorem average_test_score_of_remainder
  (h1 : percentage1 = 0.15)
  (h2 : score1 = 100)
  (h3 : percentage2 = 0.5)
  (h4 : score2 = 78)
  (h5 : percentage3 = 0.35)
  (total : totalAverage = 76.05) :
  (score3 = 63) :=
sorry

end average_test_score_of_remainder_l275_27554


namespace find_f_l275_27598

theorem find_f (q f : ℕ) (h_digit_q : q ≤ 9) (h_digit_f : f ≤ 9)
  (h_distinct : q ≠ f) 
  (h_div_by_36 : (457 * 1000 + q * 100 + 89 * 10 + f) % 36 = 0)
  (h_sum_3 : q + f = 3) :
  f = 2 :=
sorry

end find_f_l275_27598


namespace locus_of_C_l275_27512

variable (a : ℝ) (h : a > 0)

theorem locus_of_C : 
  ∃ (x y : ℝ), 
  (1 - a) * x^2 - 2 * a * x + (1 + a) * y^2 = 0 :=
sorry

end locus_of_C_l275_27512


namespace work_together_zero_days_l275_27592

theorem work_together_zero_days (a b : ℝ) (ha : a = 1/18) (hb : b = 1/9) (x : ℝ) (hx : 1 - x * a = 2/3) : x = 6 →
  (a - a) * (b - b) = 0 := by
  sorry

end work_together_zero_days_l275_27592


namespace largest_of_three_l275_27571

theorem largest_of_three (a b c : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : ab + ac + bc = -8) 
  (h3 : abc = -20) : 
  max a (max b c) = (1 + Real.sqrt 41) / 2 := 
by 
  sorry

end largest_of_three_l275_27571


namespace meters_to_centimeters_l275_27581

theorem meters_to_centimeters : (3.5 : ℝ) * 100 = 350 :=
by
  sorry

end meters_to_centimeters_l275_27581


namespace mikails_age_l275_27568

-- Define the conditions
def dollars_per_year_old : ℕ := 5
def total_dollars_given : ℕ := 45

-- Main theorem statement
theorem mikails_age (age : ℕ) : (age * dollars_per_year_old = total_dollars_given) → age = 9 :=
by
  sorry

end mikails_age_l275_27568


namespace faye_science_problems_l275_27536

variable (total_problems math_problems science_problems : Nat)
variable (finished_at_school left_for_homework : Nat)

theorem faye_science_problems :
  finished_at_school = 40 ∧ left_for_homework = 15 ∧ math_problems = 46 →
  total_problems = finished_at_school + left_for_homework →
  science_problems = total_problems - math_problems →
  science_problems = 9 :=
by
  sorry

end faye_science_problems_l275_27536


namespace sin_of_300_degrees_l275_27565

theorem sin_of_300_degrees : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 := by
  sorry

end sin_of_300_degrees_l275_27565


namespace number_of_primes_l275_27517

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem number_of_primes (p : ℕ)
  (H_prime : is_prime p)
  (H_square : is_perfect_square (1 + p + p^2 + p^3 + p^4)) :
  p = 3 :=
sorry

end number_of_primes_l275_27517


namespace m_plus_n_sum_l275_27566

theorem m_plus_n_sum :
  let m := 271
  let n := 273
  m + n = 544 :=
by {
  -- sorry included to skip the proof steps
  sorry
}

end m_plus_n_sum_l275_27566


namespace max_area_of_triangle_l275_27504

theorem max_area_of_triangle 
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : a = 2)
  (h2 : 4 * (Real.cos (A / 2))^2 -  Real.cos (2 * (B + C)) = 7 / 2)
  (h3 : A + B + C = Real.pi) :
  (Real.sqrt 3 / 2 * b * c) ≤ Real.sqrt 3 :=
sorry

end max_area_of_triangle_l275_27504


namespace Katie_cupcakes_l275_27567

theorem Katie_cupcakes (initial_cupcakes sold_cupcakes final_cupcakes : ℕ) (h1 : initial_cupcakes = 26) (h2 : sold_cupcakes = 20) (h3 : final_cupcakes = 26) :
  (final_cupcakes - (initial_cupcakes - sold_cupcakes)) = 20 :=
by
  sorry

end Katie_cupcakes_l275_27567


namespace tangent_line_circle_l275_27513

theorem tangent_line_circle (a : ℝ) : (∀ x y : ℝ, a * x + y + 1 = 0) → (∀ x y : ℝ, x^2 + y^2 - 4 * x = 0) → a = 3 / 4 :=
by
  sorry

end tangent_line_circle_l275_27513


namespace meeting_point_l275_27575

/-- Along a straight alley with 400 streetlights placed at equal intervals, numbered consecutively from 1 to 400,
    Alla and Boris set out towards each other from opposite ends of the alley with different constant speeds.
    Alla starts at streetlight number 1 and Boris starts at streetlight number 400. When Alla is at the 55th streetlight,
    Boris is at the 321st streetlight. The goal is to prove that they will meet at the 163rd streetlight.
-/
theorem meeting_point (n : ℕ) (h1 : n = 400) (h2 : ∀ i j k l : ℕ, i = 55 → j = 321 → k = 1 → l = 400) : 
  ∃ m, m = 163 := 
by
  sorry

end meeting_point_l275_27575


namespace first_number_percentage_of_second_l275_27594

theorem first_number_percentage_of_second (X : ℝ) (h1 : First = 0.06 * X) (h2 : Second = 0.18 * X) : 
  (First / Second) * 100 = 33.33 := 
by 
  sorry

end first_number_percentage_of_second_l275_27594


namespace employees_working_abroad_l275_27553

theorem employees_working_abroad
  (total_employees : ℕ)
  (fraction_abroad : ℝ)
  (h_total : total_employees = 450)
  (h_fraction : fraction_abroad = 0.06) :
  total_employees * fraction_abroad = 27 := 
by
  sorry

end employees_working_abroad_l275_27553


namespace replace_90_percent_in_3_days_cannot_replace_all_banknotes_l275_27572

-- Define constants and conditions
def total_old_banknotes : ℕ := 3628800
def daily_cost : ℕ := 90000
def major_repair_cost : ℕ := 700000
def max_daily_print_after_repair : ℕ := 1000000
def budget_limit : ℕ := 1000000

-- Define the day's print capability function (before repair)
def daily_print (num_days : ℕ) (banknotes_remaining : ℕ) : ℕ :=
  if num_days = 1 then banknotes_remaining / 2
  else (banknotes_remaining / (num_days + 1))

-- Define the budget calculation before repair
def print_costs (num_days : ℕ) (banknotes_remaining : ℕ) : ℕ :=
  daily_cost * num_days

-- Lean theorem to be stated proving that 90% of the banknotes can be replaced within 3 days
theorem replace_90_percent_in_3_days :
  ∃ (days : ℕ) (banknotes_replaced : ℕ), days = 3 ∧ banknotes_replaced = 3265920 ∧ print_costs days total_old_banknotes ≤ budget_limit :=
sorry

-- Lean theorem to be stated proving that not all banknotes can be replaced within the given budget
theorem cannot_replace_all_banknotes :
  ∀ banknotes_replaced cost : ℕ,
  banknotes_replaced < total_old_banknotes ∧ cost ≤ budget_limit →
  banknotes_replaced + (total_old_banknotes / (4 + 1)) < total_old_banknotes :=
sorry

end replace_90_percent_in_3_days_cannot_replace_all_banknotes_l275_27572


namespace sum_of_four_digit_numbers_l275_27597

theorem sum_of_four_digit_numbers :
  let digits := [2, 4, 5, 3]
  let factorial := Nat.factorial (List.length digits)
  let each_appearance := factorial / (List.length digits)
  (each_appearance * (2 + 4 + 5 + 3) * (1000 + 100 + 10 + 1)) = 93324 :=
by
  let digits := [2, 4, 5, 3]
  let factorial := Nat.factorial (List.length digits)
  let each_appearance := factorial / (List.length digits)
  show (each_appearance * (2 + 4 + 5 + 3) * (1000 + 100 + 10 + 1)) = 93324
  sorry

end sum_of_four_digit_numbers_l275_27597


namespace exist_positive_integers_x_y_z_l275_27527

theorem exist_positive_integers_x_y_z (n : ℕ) (hn : n > 0) : 
  ∃ (x y z : ℕ), 
    x = 2^(n^2) * 3^(n+1) ∧
    y = 2^(n^2 - n) * 3^n ∧
    z = 2^(n^2 - 2*n + 2) * 3^(n-1) ∧
    x^(n-1) + y^n = z^(n+1) :=
by {
  -- placeholder for the proof
  sorry
}

end exist_positive_integers_x_y_z_l275_27527


namespace line_ellipse_tangent_l275_27569

theorem line_ellipse_tangent (m : ℝ) : 
  (∀ x y : ℝ, (y = m * x + 2) → (x^2 + (y^2 / 4) = 1)) → m^2 = 0 :=
sorry

end line_ellipse_tangent_l275_27569


namespace different_genre_pairs_count_l275_27529

theorem different_genre_pairs_count 
  (mystery_books : Finset ℕ)
  (fantasy_books : Finset ℕ)
  (biographies : Finset ℕ)
  (h1 : mystery_books.card = 4)
  (h2 : fantasy_books.card = 4)
  (h3 : biographies.card = 4) :
  (mystery_books.product (fantasy_books ∪ biographies)).card +
  (fantasy_books.product (mystery_books ∪ biographies)).card +
  (biographies.product (mystery_books ∪ fantasy_books)).card = 48 := 
sorry

end different_genre_pairs_count_l275_27529


namespace distance_swim_downstream_correct_l275_27518

def speed_man_still_water : ℝ := 7
def time_taken : ℝ := 5
def distance_upstream : ℝ := 25

lemma distance_swim_downstream (V_m : ℝ) (t : ℝ) (d_up : ℝ) : 
  t * ((V_m + (V_m - d_up / t)) / 2) = 45 :=
by
  have h_speed_upstream : (V_m - (d_up / t)) = d_up / t := by sorry
  have h_speed_stream : (d_up / t) = (V_m - (d_up / t)) := by sorry
  have h_distance_downstream : t * ((V_m + (V_m - (d_up / t)) / 2)) = t * (V_m + (V_m - (V_m - d_up / t))) := by sorry
  sorry

noncomputable def distance_swim_downstream_value : ℝ :=
  9 * 5

theorem distance_swim_downstream_correct :
  distance_swim_downstream_value = 45 :=
by
  sorry

end distance_swim_downstream_correct_l275_27518


namespace map_scale_l275_27501

theorem map_scale (map_distance : ℝ) (time : ℝ) (speed : ℝ) (actual_distance : ℝ) (scale : ℝ) 
  (h1 : map_distance = 5) 
  (h2 : time = 1.5) 
  (h3 : speed = 60) 
  (h4 : actual_distance = speed * time) 
  (h5 : scale = map_distance / actual_distance) : 
  scale = 1 / 18 :=
by 
  sorry

end map_scale_l275_27501


namespace fraction_meaningful_condition_l275_27543

theorem fraction_meaningful_condition (x : ℝ) : (4 / (x + 2) ≠ 0) ↔ (x ≠ -2) := 
by 
  sorry

end fraction_meaningful_condition_l275_27543


namespace inequality_solution_l275_27583

theorem inequality_solution (x : ℝ) : |2 * x - 7| < 3 → 2 < x ∧ x < 5 :=
by
  sorry

end inequality_solution_l275_27583


namespace multiples_of_15_between_17_and_158_l275_27587

theorem multiples_of_15_between_17_and_158 : 
  let first := 30
  let last := 150
  let step := 15
  Nat.succ ((last - first) / step) = 9 := 
by
  sorry

end multiples_of_15_between_17_and_158_l275_27587


namespace n_eq_14_l275_27522

variable {a : ℕ → ℕ}  -- the arithmetic sequence
variable {S : ℕ → ℕ}  -- the sum function of the first n terms
variable {d : ℕ}      -- the common difference of the arithmetic sequence

-- Given Conditions
axiom Sn_eq_4 : S 4 = 40
axiom Sn_eq_210 : ∃ (n : ℕ), S n = 210
axiom Sn_minus_4_eq_130 : ∃ (n : ℕ), S (n - 4) = 130

-- Main theorem to prove
theorem n_eq_14 : ∃ (n : ℕ),  S n = 210 ∧ S (n - 4) = 130 ∧ n = 14 :=
by
  sorry

end n_eq_14_l275_27522


namespace related_sequence_exists_l275_27578

theorem related_sequence_exists :
  ∃ b : Fin 5 → ℕ, b = ![11, 10, 9, 8, 7] :=
by
  let a : Fin 5 → ℕ := ![1, 5, 9, 13, 17]
  let b : Fin 5 → ℕ := ![
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 0) / 4,
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 1) / 4,
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 2) / 4,
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 3) / 4,
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 4) / 4
  ]
  existsi b
  sorry

end related_sequence_exists_l275_27578


namespace find_number_l275_27584

theorem find_number : ∃ x : ℝ, (6 * ((x / 8 + 8) - 30) = 12) ∧ x = 192 :=
by sorry

end find_number_l275_27584


namespace pipe_B_fill_time_l275_27531

theorem pipe_B_fill_time (T : ℕ) (h1 : 50 > 0) (h2 : 30 > 0)
  (h3 : (1/50 + 1/T = 1/30)) : T = 75 := 
sorry

end pipe_B_fill_time_l275_27531


namespace solution_inequality_l275_27545

-- Conditions
variables {a b x : ℝ}
theorem solution_inequality (h1 : a < 0) (h2 : b = a) :
  {x : ℝ | (ax + b) ≤ 0} = {x : ℝ | x ≥ -1} →
  {x : ℝ | (ax + b) / (x - 2) > 0} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end solution_inequality_l275_27545


namespace smallest_four_digit_integer_l275_27521

theorem smallest_four_digit_integer (n : ℕ) :
  (75 * n ≡ 225 [MOD 450]) ∧ (1000 ≤ n ∧ n < 10000) → n = 1005 :=
sorry

end smallest_four_digit_integer_l275_27521


namespace model_to_reality_length_l275_27576

-- Defining conditions
def scale_factor := 50 -- one centimeter represents 50 meters
def model_length := 7.5 -- line segment in the model is 7.5 centimeters

-- Statement of the problem
theorem model_to_reality_length (scale_factor model_length : ℝ) 
  (scale_condition : scale_factor = 50) (length_condition : model_length = 7.5) :
  model_length * scale_factor = 375 := 
by
  rw [length_condition, scale_condition]
  norm_num

end model_to_reality_length_l275_27576


namespace initial_men_count_l275_27503

variable (M : ℕ)

theorem initial_men_count
  (work_completion_time : ℕ)
  (men_leaving : ℕ)
  (remaining_work_time : ℕ)
  (completion_days : ℕ) :
  work_completion_time = 40 →
  men_leaving = 20 →
  remaining_work_time = 40 →
  completion_days = 10 →
  M = 80 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_men_count_l275_27503


namespace sum_of_coordinates_of_B_l275_27561

-- Definitions
def Point := (ℝ × ℝ)
def isMidpoint (M A B : Point) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Given conditions
def M : Point := (4, 8)
def A : Point := (10, 4)

-- Statement to prove
theorem sum_of_coordinates_of_B (B : Point) (h : isMidpoint M A B) :
  B.1 + B.2 = 10 :=
by
  sorry

end sum_of_coordinates_of_B_l275_27561


namespace total_books_for_girls_l275_27582

theorem total_books_for_girls (num_girls : ℕ) (num_boys : ℕ) (total_books : ℕ)
  (h_girls : num_girls = 15)
  (h_boys : num_boys = 10)
  (h_books : total_books = 375) :
  num_girls * (total_books / (num_girls + num_boys)) = 225 :=
by
  sorry

end total_books_for_girls_l275_27582


namespace total_ear_muffs_bought_l275_27506

-- Define the number of ear muffs bought before December
def ear_muffs_before_dec : ℕ := 1346

-- Define the number of ear muffs bought during December
def ear_muffs_during_dec : ℕ := 6444

-- The total number of ear muffs bought by customers
theorem total_ear_muffs_bought : ear_muffs_before_dec + ear_muffs_during_dec = 7790 :=
by
  sorry

end total_ear_muffs_bought_l275_27506


namespace ordered_pairs_count_l275_27516

theorem ordered_pairs_count : 
    ∃ (s : Finset (ℝ × ℝ)), 
        (∀ (x y : ℝ), (x + 3 * y = 3 ∧ abs (abs x - abs y) = 1 ↔ (x, y) ∈ s)) ∧ 
        s.card = 3 :=
    by
    sorry

end ordered_pairs_count_l275_27516


namespace parallelogram_not_symmetrical_l275_27530

-- Define the shapes
inductive Shape
| circle
| rectangle
| isosceles_trapezoid
| parallelogram

-- Define what it means for a shape to be symmetrical
def is_symmetrical (s: Shape) : Prop :=
  match s with
  | Shape.circle => True
  | Shape.rectangle => True
  | Shape.isosceles_trapezoid => True
  | Shape.parallelogram => False -- The condition we're interested in proving

-- The main theorem stating the problem
theorem parallelogram_not_symmetrical : is_symmetrical Shape.parallelogram = False :=
by
  sorry

end parallelogram_not_symmetrical_l275_27530


namespace no_integer_roots_l275_27599

theorem no_integer_roots (x : ℤ) : ¬ (x^3 - 5 * x^2 - 11 * x + 35 = 0) := 
sorry

end no_integer_roots_l275_27599


namespace range_of_function_l275_27544

theorem range_of_function : ∀ (y : ℝ), (0 < y ∧ y ≤ 1 / 2) ↔ ∃ (x : ℝ), y = 1 / (x^2 + 2) := 
by
  sorry

end range_of_function_l275_27544


namespace two_digit_primes_with_digit_sum_10_count_l275_27577

def digits_sum_to_ten (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_digit_sum_10_count : 
  ∃ count : ℕ, count = 4 ∧ ∀ n, (two_digit_number n ∧ digits_sum_to_ten n ∧ Prime n) → count = 4 := 
by
  sorry

end two_digit_primes_with_digit_sum_10_count_l275_27577


namespace focal_distance_of_ellipse_l275_27580

theorem focal_distance_of_ellipse :
  ∀ (x y : ℝ), (x^2 / 16) + (y^2 / 9) = 1 → (2 * Real.sqrt 7) = 2 * Real.sqrt 7 :=
by
  intros x y hxy
  sorry

end focal_distance_of_ellipse_l275_27580


namespace distance_difference_l275_27560

theorem distance_difference (t : ℕ) (speed_alice speed_bob : ℕ) :
  speed_alice = 15 → speed_bob = 10 → t = 6 → (speed_alice * t) - (speed_bob * t) = 30 :=
by
  intros h1 h2 h3
  sorry

end distance_difference_l275_27560


namespace probability_of_symmetry_line_l275_27519

-- Define the conditions of the problem.
def is_on_symmetry_line (P Q : (ℤ × ℤ)) :=
  (Q.fst = P.fst) ∨ (Q.snd = P.snd) ∨ (Q.fst - P.fst = Q.snd - P.snd) ∨ (Q.fst - P.fst = P.snd - Q.snd)

-- Define the main statement of the theorem to be proved.
theorem probability_of_symmetry_line :
  let grid_size := 11
  let total_points := grid_size * grid_size
  let center : (ℤ × ℤ) := (grid_size / 2, grid_size / 2)
  let other_points := total_points - 1
  let symmetric_points := 40
  /- Here we need to calculate the probability, which is the ratio of symmetric points to other points,
     and this should equal 1/3 -/
  (symmetric_points : ℚ) / other_points = 1 / 3 :=
by sorry

end probability_of_symmetry_line_l275_27519


namespace sandwiches_difference_l275_27509

-- Conditions definitions
def sandwiches_at_lunch_monday : ℤ := 3
def sandwiches_at_dinner_monday : ℤ := 2 * sandwiches_at_lunch_monday
def total_sandwiches_monday : ℤ := sandwiches_at_lunch_monday + sandwiches_at_dinner_monday
def sandwiches_on_tuesday : ℤ := 1

-- Proof goal
theorem sandwiches_difference :
  total_sandwiches_monday - sandwiches_on_tuesday = 8 :=
  by
  sorry

end sandwiches_difference_l275_27509


namespace trig_expression_eq_zero_l275_27515

theorem trig_expression_eq_zero (α : ℝ) (h1 : Real.sin α = -2 / Real.sqrt 5) (h2 : Real.cos α = 1 / Real.sqrt 5) :
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 0 := by
  sorry

end trig_expression_eq_zero_l275_27515


namespace michael_savings_l275_27500

theorem michael_savings :
  let price := 45
  let tax_rate := 0.08
  let promo_A_dis := 0.40
  let promo_B_dis := 15
  let before_tax_A := price + price * (1 - promo_A_dis)
  let before_tax_B := price + (price - promo_B_dis)
  let after_tax_A := before_tax_A * (1 + tax_rate)
  let after_tax_B := before_tax_B * (1 + tax_rate)
  after_tax_B - after_tax_A = 3.24 :=
by
  sorry

end michael_savings_l275_27500


namespace no_rational_roots_of_odd_coeffs_l275_27526

theorem no_rational_roots_of_odd_coeffs (a b c : ℤ) (h_a_odd : a % 2 = 1) (h_b_odd : b % 2 = 1) (h_c_odd : c % 2 = 1)
  (h_rational_root : ∃ (p q : ℤ), q ≠ 0 ∧ (a * (p / q : ℚ)^2 + b * (p / q : ℚ) + c = 0)) : false :=
sorry

end no_rational_roots_of_odd_coeffs_l275_27526


namespace keith_attended_games_l275_27535

-- Definitions based on the given conditions
def total_games : ℕ := 8
def missed_games : ℕ := 4

-- The proof goal: Keith's attendance
def attended_games : ℕ := total_games - missed_games

-- Main statement to prove the total games Keith attended
theorem keith_attended_games : attended_games = 4 := by
  -- Sorry is a placeholder for the proof
  sorry

end keith_attended_games_l275_27535


namespace history_homework_time_l275_27540

def total_time := 180
def math_homework := 45
def english_homework := 30
def science_homework := 50
def special_project := 30

theorem history_homework_time : total_time - (math_homework + english_homework + science_homework + special_project) = 25 := by
  sorry

end history_homework_time_l275_27540


namespace molecular_weight_correct_l275_27510

def potassium_weight : ℝ := 39.10
def chromium_weight : ℝ := 51.996
def oxygen_weight : ℝ := 16.00

def num_potassium_atoms : ℕ := 2
def num_chromium_atoms : ℕ := 2
def num_oxygen_atoms : ℕ := 7

def molecular_weight_of_compound : ℝ :=
  (num_potassium_atoms * potassium_weight) +
  (num_chromium_atoms * chromium_weight) +
  (num_oxygen_atoms * oxygen_weight)

theorem molecular_weight_correct :
  molecular_weight_of_compound = 294.192 :=
by
  sorry

end molecular_weight_correct_l275_27510


namespace compute_expression_l275_27573
-- Import the standard math library to avoid import errors.

-- Define the theorem statement based on the given conditions and the correct answer.
theorem compute_expression :
  (75 * 2424 + 25 * 2424) / 2 = 121200 :=
by
  sorry

end compute_expression_l275_27573


namespace det_is_18_l275_27549

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![4, 1],
    ![2, 5]]

theorem det_is_18 : det A = 18 := by
  sorry

end det_is_18_l275_27549


namespace equation_of_perpendicular_line_intersection_l275_27520

theorem equation_of_perpendicular_line_intersection  :
  ∃ (x y : ℝ), 4 * x + 2 * y + 5 = 0 ∧ 3 * x - 2 * y + 9 = 0 ∧ 
               (∃ (m : ℝ), m = 2 ∧ 4 * x - 2 * y + 11 = 0) := 
sorry

end equation_of_perpendicular_line_intersection_l275_27520


namespace find_b_n_find_T_n_l275_27551

-- Conditions
def S (n : ℕ) : ℕ := 3 * n^2 + 8 * n
def a (n : ℕ) : ℕ := S n - S (n - 1) -- provided n > 1
def b : ℕ → ℕ := sorry -- This is what we need to prove
def c (n : ℕ) : ℕ := (a n + 1)^(n + 1) / (b n + 2)^n  -- Definition of c_n
def T (n : ℕ) : ℕ := sorry -- The sum of the first n terms of c_n

-- Proof requirements
def proof_b_n := ∀ n : ℕ, b n = 3 * n + 1
def proof_T_n := ∀ n : ℕ, T n = 3 * n * 2^(n+2)

theorem find_b_n : proof_b_n := 
by sorry

theorem find_T_n : proof_T_n := 
by sorry

end find_b_n_find_T_n_l275_27551


namespace manufacturing_percentage_l275_27558

theorem manufacturing_percentage (deg_total : ℝ) (deg_manufacturing : ℝ) (h1 : deg_total = 360) (h2 : deg_manufacturing = 126) : 
  (deg_manufacturing / deg_total * 100) = 35 := by
  sorry

end manufacturing_percentage_l275_27558


namespace problem_solution_l275_27564

open Real

noncomputable def length_and_slope_MP 
    (length_MN : ℝ) 
    (slope_MN : ℝ) 
    (length_NP : ℝ) 
    (slope_NP : ℝ) 
    : (ℝ × ℝ) := sorry

theorem problem_solution :
  length_and_slope_MP 6 14 7 8 = (5.55, 25.9) :=
  sorry

end problem_solution_l275_27564


namespace intersection_and_complement_l275_27528

open Set

def A := {x : ℝ | -4 ≤ x ∧ x ≤ -2}
def B := {x : ℝ | x + 3 ≥ 0}

theorem intersection_and_complement : 
  (A ∩ B = {x | -3 ≤ x ∧ x ≤ -2}) ∧ (compl (A ∩ B) = {x | x < -3 ∨ x > -2}) :=
by
  sorry

end intersection_and_complement_l275_27528


namespace find_amount_l275_27546

theorem find_amount (amount : ℝ) (h : 0.25 * amount = 75) : amount = 300 :=
sorry

end find_amount_l275_27546


namespace complete_the_square_l275_27534

theorem complete_the_square (x : ℝ) :
  (x^2 - 6 * x + 8 = 0) → ((x - 3)^2 = 1) := 
by
  sorry

end complete_the_square_l275_27534


namespace moles_of_ammonium_nitrate_formed_l275_27557

def ammonia := ℝ
def nitric_acid := ℝ
def ammonium_nitrate := ℝ

-- Define the stoichiometric coefficients from the balanced equation.
def stoichiometric_ratio_ammonia : ℝ := 1
def stoichiometric_ratio_nitric_acid : ℝ := 1
def stoichiometric_ratio_ammonium_nitrate : ℝ := 1

-- Define the initial moles of reactants.
def initial_moles_ammonia (moles : ℝ) : Prop := moles = 3
def initial_moles_nitric_acid (moles : ℝ) : Prop := moles = 3

-- The reaction goes to completion as all reactants are used:
theorem moles_of_ammonium_nitrate_formed :
  ∀ (moles_ammonia moles_nitric_acid : ℝ),
    initial_moles_ammonia moles_ammonia →
    initial_moles_nitric_acid moles_nitric_acid →
    (moles_ammonia / stoichiometric_ratio_ammonia) = 
    (moles_nitric_acid / stoichiometric_ratio_nitric_acid) →
    (moles_ammonia / stoichiometric_ratio_ammonia) * stoichiometric_ratio_ammonium_nitrate = 3 :=
by
  intros moles_ammonia moles_nitric_acid h_ammonia h_nitric_acid h_ratio
  rw [h_ammonia, h_nitric_acid] at *
  simp only [stoichiometric_ratio_ammonia, stoichiometric_ratio_nitric_acid, stoichiometric_ratio_ammonium_nitrate] at *
  sorry

end moles_of_ammonium_nitrate_formed_l275_27557


namespace problem_statement_l275_27596

variable (a b c : ℝ)
variable (x : ℝ)

theorem problem_statement (h : ∀ x ∈ Set.Icc (-1 : ℝ) 1, |a * x^2 - b * x + c| < 1) :
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, |(a + b) * x^2 + c| < 1 :=
by
  intros x hx
  let f := fun x => a * x^2 - b * x + c
  let g := fun x => (a + b) * x^2 + c
  have h1 : ∀ x ∈ Set.Icc (-1 : ℝ) 1, |f x| < 1 := h
  sorry

end problem_statement_l275_27596


namespace kate_needs_more_money_l275_27591

theorem kate_needs_more_money
  (pen_price : ℝ)
  (notebook_price : ℝ)
  (artset_price : ℝ)
  (kate_pen_money_fraction : ℝ)
  (notebook_discount : ℝ)
  (artset_discount : ℝ)
  (kate_artset_money : ℝ) :
  pen_price = 30 →
  notebook_price = 20 →
  artset_price = 50 →
  kate_pen_money_fraction = 1/3 →
  notebook_discount = 0.15 →
  artset_discount = 0.4 →
  kate_artset_money = 10 →
  (pen_price - kate_pen_money_fraction * pen_price) +
  (notebook_price * (1 - notebook_discount)) +
  (artset_price * (1 - artset_discount) - kate_artset_money) = 57 :=
by
  sorry

end kate_needs_more_money_l275_27591


namespace interval_for_systematic_sampling_l275_27595

-- Define the total population size
def total_population : ℕ := 1203

-- Define the sample size
def sample_size : ℕ := 40

-- Define the interval for systematic sampling
def interval (n m : ℕ) : ℕ := (n - (n % m)) / m

-- The proof statement that the interval \( k \) for segmenting is 30
theorem interval_for_systematic_sampling : interval total_population sample_size = 30 :=
by
  show interval 1203 40 = 30
  sorry

end interval_for_systematic_sampling_l275_27595


namespace beds_with_fewer_beds_l275_27542

theorem beds_with_fewer_beds:
  ∀ (total_rooms rooms_with_fewer_beds rooms_with_three_beds total_beds x : ℕ),
    total_rooms = 13 →
    rooms_with_fewer_beds = 8 →
    rooms_with_three_beds = total_rooms - rooms_with_fewer_beds →
    total_beds = 31 →
    8 * x + 3 * (total_rooms - rooms_with_fewer_beds) = total_beds →
    x = 2 :=
by
  intros total_rooms rooms_with_fewer_beds rooms_with_three_beds total_beds x
  intros ht_rooms hrwb hrwtb htb h_eq
  sorry

end beds_with_fewer_beds_l275_27542


namespace find_s_l275_27548

variable {t s : Real}

theorem find_s (h1 : t = 8 * s^2) (h2 : t = 4) : s = Real.sqrt 2 / 2 :=
by
  sorry

end find_s_l275_27548


namespace kolya_time_segment_DE_l275_27533

-- Definitions representing the conditions
def time_petya_route : ℝ := 12  -- Petya takes 12 minutes
def time_kolya_route : ℝ := 12  -- Kolya also takes 12 minutes
def kolya_speed_factor : ℝ := 1.2

-- Proof problem: Prove that Kolya spends 1 minute traveling the segment D-E
theorem kolya_time_segment_DE 
    (v : ℝ)  -- Assume v is Petya's speed
    (time_petya_A_B_C : ℝ := time_petya_route)  
    (time_kolya_A_D_E_F_C : ℝ := time_kolya_route)
    (kolya_fast_factor : ℝ := kolya_speed_factor)
    : (time_petya_A_B_C / kolya_fast_factor - time_petya_A_B_C) / (2 / kolya_fast_factor) = 1 := 
by 
    sorry

end kolya_time_segment_DE_l275_27533


namespace tan_45_deg_l275_27562

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l275_27562


namespace kelly_snacks_l275_27550

theorem kelly_snacks (peanuts raisins : ℝ) (h_peanuts : peanuts = 0.1) (h_raisins : raisins = 0.4) : peanuts + raisins = 0.5 :=
by
  sorry

end kelly_snacks_l275_27550


namespace determine_q_l275_27523

theorem determine_q (q : ℝ → ℝ) :
  (∀ x, q x = 0 ↔ x = 3 ∨ x = -3) ∧
  (∀ k : ℝ, k < 3) ∧ -- indicating degree considerations for asymptotes
  (q 2 = 18) →
  q = (fun x => (-18 / 5) * x ^ 2 + 162 / 5) :=
by
  sorry

end determine_q_l275_27523


namespace equal_pair_c_l275_27539

theorem equal_pair_c : (-4)^3 = -(4^3) := 
by {
  sorry
}

end equal_pair_c_l275_27539


namespace boat_speed_in_still_water_l275_27541

-- Boat's speed in still water in km/hr
variable (B S : ℝ)

-- Conditions given for the boat's speed along and against the stream
axiom cond1 : B + S = 11
axiom cond2 : B - S = 5

-- Prove that the speed of the boat in still water is 8 km/hr
theorem boat_speed_in_still_water : B = 8 :=
by
  sorry

end boat_speed_in_still_water_l275_27541


namespace remainder_of_division_l275_27508

theorem remainder_of_division (dividend divisor quotient remainder : ℕ)
  (h1 : dividend = 55053)
  (h2 : divisor = 456)
  (h3 : quotient = 120)
  (h4 : remainder = dividend - divisor * quotient) : 
  remainder = 333 := by
  sorry

end remainder_of_division_l275_27508


namespace petya_run_time_l275_27593

-- Definitions
def time_petya_4_to_1 : ℕ := 12

-- Conditions
axiom time_mom_condition : ∃ (time_mom : ℕ), time_petya_4_to_1 = time_mom - 2
axiom time_mom_5_to_1_condition : ∃ (time_petya_5_to_1 : ℕ), ∀ time_mom : ℕ, time_mom = time_petya_5_to_1 - 2

-- Proof statement
theorem petya_run_time :
  ∃ (time_petya_4_to_1 : ℕ), time_petya_4_to_1 = 12 :=
sorry

end petya_run_time_l275_27593


namespace sum_eq_zero_l275_27590

variable {a b c : ℝ}

theorem sum_eq_zero (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
    (h4 : a ≠ b ∨ b ≠ c ∨ c ≠ a)
    (h5 : (a^2) / (2 * (a^2) + b * c) + (b^2) / (2 * (b^2) + c * a) + (c^2) / (2 * (c^2) + a * b) = 1) :
  a + b + c = 0 :=
sorry

end sum_eq_zero_l275_27590


namespace problem_equivalence_l275_27537

theorem problem_equivalence (n : ℕ) (H₁ : 2 * 2006 = 1) (H₂ : ∀ n : ℕ, (2 * n + 2) * 2006 = 3 * (2 * n * 2006)) :
  2008 * 2006 = 3 ^ 1003 :=
by
  sorry

end problem_equivalence_l275_27537


namespace x_plus_p_eq_2p_plus_2_l275_27525

-- Define the conditions and the statement to be proved
theorem x_plus_p_eq_2p_plus_2 (x p : ℝ) (h1 : x > 2) (h2 : |x - 2| = p) : x + p = 2 * p + 2 :=
by
  -- Proof goes here
  sorry

end x_plus_p_eq_2p_plus_2_l275_27525


namespace Zachary_did_47_pushups_l275_27511

-- Define the conditions and the question
def Zachary_pushups (David_pushups difference : ℕ) : ℕ :=
  David_pushups - difference

theorem Zachary_did_47_pushups :
  Zachary_pushups 62 15 = 47 :=
by
  -- Provide the proof here (we'll use sorry for now)
  sorry

end Zachary_did_47_pushups_l275_27511


namespace cylinder_base_radius_l275_27502

theorem cylinder_base_radius (l w : ℝ) (h_l : l = 6) (h_w : w = 4) (h_circ : l = 2 * Real.pi * r ∨ w = 2 * Real.pi * r) : 
    r = 3 / Real.pi ∨ r = 2 / Real.pi := by
  sorry

end cylinder_base_radius_l275_27502


namespace rectangle_new_area_l275_27589

theorem rectangle_new_area
  (L W : ℝ) (h1 : L * W = 600) :
  let L' := 0.8 * L
  let W' := 1.3 * W
  (L' * W' = 624) :=
by
  -- Let L' = 0.8 * L
  -- Let W' = 1.3 * W
  -- Proof goes here
  sorry

end rectangle_new_area_l275_27589


namespace cars_to_sell_l275_27579

theorem cars_to_sell (clients : ℕ) (selections_per_client : ℕ) (selections_per_car : ℕ) (total_clients : ℕ) (h1 : selections_per_client = 2) 
  (h2 : selections_per_car = 3) (h3 : total_clients = 24) : (total_clients * selections_per_client / selections_per_car = 16) :=
by
  sorry

end cars_to_sell_l275_27579
