import Mathlib

namespace prove_b_zero_l289_289898

variables {a b c : ℕ}

theorem prove_b_zero (h1 : ∃ (a b c : ℕ), a^5 + 4 * b^5 = c^5 ∧ c % 2 = 0) : b = 0 :=
sorry

end prove_b_zero_l289_289898


namespace probability_at_least_one_male_l289_289711

-- Definitions according to the problem conditions
def total_finalists : ℕ := 8
def female_finalists : ℕ := 5
def male_finalists : ℕ := 3
def num_selected : ℕ := 3

-- Binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Probabilistic statement
theorem probability_at_least_one_male :
  let total_ways := binom total_finalists num_selected
  let ways_all_females := binom female_finalists num_selected
  let ways_at_least_one_male := total_ways - ways_all_females
  (ways_at_least_one_male : ℚ) / total_ways = 23 / 28 :=
by
  sorry

end probability_at_least_one_male_l289_289711


namespace apple_lovers_l289_289369

theorem apple_lovers :
  ∃ (x y : ℕ), 22 * x = 1430 ∧ 13 * (x + y) = 1430 ∧ y = 45 :=
by
  sorry

end apple_lovers_l289_289369


namespace simplify_exponents_l289_289918

theorem simplify_exponents (x : ℝ) : x^5 * x^3 = x^8 :=
by
  sorry

end simplify_exponents_l289_289918


namespace height_of_wall_l289_289490

-- Definitions
def brick_length : ℝ := 25
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6
def wall_length : ℝ := 850
def wall_width : ℝ := 22.5
def num_bricks : ℝ := 6800

-- Total volume of bricks
def total_brick_volume : ℝ := num_bricks * brick_length * brick_width * brick_height

-- Volume of the wall
def wall_volume (height : ℝ) : ℝ := wall_length * wall_width * height

-- Proof statement
theorem height_of_wall : ∃ h : ℝ, wall_volume h = total_brick_volume ∧ h = 600 := 
sorry

end height_of_wall_l289_289490


namespace maximize_profit_constraints_l289_289958

variable (a1 a2 b1 b2 d1 d2 c1 c2 x y z : ℝ)

theorem maximize_profit_constraints (a1 a2 b1 b2 d1 d2 c1 c2 x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (a1 * x + a2 * y ≤ c1) ∧ (b1 * x + b2 * y ≤ c2) :=
sorry

end maximize_profit_constraints_l289_289958


namespace eq_decrease_in_area_l289_289658

noncomputable def decrease_in_area (A : ℝ) (decrease_length : ℝ) : ℝ :=
  let s := real.sqrt (4 * A / real.sqrt 3)
  let s' := s - decrease_length
  let A' := s'^2 * real.sqrt 3 / 4
  A - A'

theorem eq_decrease_in_area :
  decrease_in_area (121 * real.sqrt 3) 6 = 57 * real.sqrt 3 :=
by
  sorry

end eq_decrease_in_area_l289_289658


namespace sum_of_cubes_of_consecutive_numbers_divisible_by_9_l289_289589

theorem sum_of_cubes_of_consecutive_numbers_divisible_by_9 (a : ℕ) (h : a > 1) : 
  9 ∣ ((a - 1)^3 + a^3 + (a + 1)^3) := 
by 
  sorry

end sum_of_cubes_of_consecutive_numbers_divisible_by_9_l289_289589


namespace geometric_series_sum_l289_289805

theorem geometric_series_sum :
  let a := 3
  let r := 3
  let n := 9
  let last_term := a * r^(n - 1)
  last_term = 19683 →
  let S := a * (r^n - 1) / (r - 1)
  S = 29523 :=
by
  intros
  sorry

end geometric_series_sum_l289_289805


namespace complement_U_M_l289_289288

def U : Set ℕ := {x | x > 0 ∧ ∃ y : ℝ, y = Real.sqrt (5 - x)}
def M : Set ℕ := {x ∈ U | 4^x ≤ 16}

theorem complement_U_M : U \ M = {3, 4, 5} := by
  sorry

end complement_U_M_l289_289288


namespace incorrect_guess_at_20_Iskander_incorrect_guess_20_l289_289468

def is_color (col : String) (pos : Nat) : Prop := sorry
def valid_guesses : Prop :=
  (is_color "white" 2) ∧
  (is_color "brown" 20) ∧
  (is_color "black" 400) ∧
  (is_color "brown" 600) ∧
  (is_color "white" 800)

theorem incorrect_guess_at_20 :
  (∃ x, (x ∈ [2, 20, 400, 600, 800]) ∧ ¬ is_color_correct x) :=
begin
  sorry -- proof is not required
end

/-- Main theorem to identify the incorrect guess position. -/
theorem Iskander_incorrect_guess_20 :
  valid_guesses →
  (∃! x ∈ [2, 20, 400, 600, 800], ¬ is_color_correct x) →
  ¬ is_color "brown" 20 :=
begin
  admit -- proof is not required
end

end incorrect_guess_at_20_Iskander_incorrect_guess_20_l289_289468


namespace complex_number_solution_l289_289058

theorem complex_number_solution (z : ℂ) (i : ℂ) (h1 : i * z = (1 - 2 * i) ^ 2) (h2 : i * i = -1) : z = -4 + 3 * i := by
  sorry

end complex_number_solution_l289_289058


namespace probability_neither_alive_l289_289215

-- Define the probabilities for the man and his wife being alive for 10 more years.
def man_alive_10_years : ℝ := 1 / 4
def wife_alive_10_years : ℝ := 1 / 3

-- Define the probability of them being not alive for 10 more years.
def man_not_alive_10_years : ℝ := 1 - man_alive_10_years
def wife_not_alive_10_years : ℝ := 1 - wife_alive_10_years

-- Define the independence of their lifespans.
def independent_events : ℝ := man_not_alive_10_years * wife_not_alive_10_years

theorem probability_neither_alive :
  independent_events = 1 / 2 :=
by {
  -- Conditional definitions
  have h_man : man_not_alive_10_years = 3 / 4 := by sorry,
  have h_wife : wife_not_alive_10_years = 2 / 3 := by sorry,
  -- Calculation using independence and given probabilities
  calc
    independent_events = man_not_alive_10_years * wife_not_alive_10_years : by rfl
    ... = (3 / 4) * (2 / 3) : by rw [h_man, h_wife]
    ... = 1 / 2 : by norm_num
}

end probability_neither_alive_l289_289215


namespace least_value_divisibility_l289_289196

theorem least_value_divisibility : ∃ (x : ℕ), (23 * x) % 3 = 0  ∧ (∀ y : ℕ, ((23 * y) % 3 = 0 → x ≤ y)) := 
  sorry

end least_value_divisibility_l289_289196


namespace conditional_probability_law_bayes_theorem_l289_289899

variables {Ω : Type*} [ProbabilitySpace Ω]
variables (M N : Event Ω)

theorem conditional_probability_law :
  P(M ∩ N) = P(M) * P(N | M) :=
sorry

theorem bayes_theorem :
  P(M | N) = (P(N | M) * P(M)) / P(N) :=
sorry

end conditional_probability_law_bayes_theorem_l289_289899


namespace chen_steps_recorded_correct_l289_289631

-- Define the standard for steps per day
def standard : ℕ := 5000

-- Define the steps walked by Xia
def xia_steps : ℕ := 6200

-- Define the recorded steps for Xia
def xia_recorded : ℤ := xia_steps - standard

-- Assert that Xia's recorded steps are +1200
lemma xia_steps_recorded_correct : xia_recorded = 1200 := by
  sorry

-- Define the steps walked by Chen
def chen_steps : ℕ := 4800

-- Define the recorded steps for Chen
def chen_recorded : ℤ := standard - chen_steps

-- State and prove that Chen's recorded steps are -200
theorem chen_steps_recorded_correct : chen_recorded = -200 :=
  sorry

end chen_steps_recorded_correct_l289_289631


namespace last_digit_sum_l289_289477

theorem last_digit_sum :
  (2^2 % 10 + 20^20 % 10 + 200^200 % 10 + 2006^2006 % 10) % 10 = 0 := 
by
  sorry

end last_digit_sum_l289_289477


namespace dentist_cleaning_cost_l289_289769

theorem dentist_cleaning_cost
  (F: ℕ)
  (C: ℕ)
  (B: ℕ)
  (tooth_extraction_cost: ℕ)
  (HC1: F = 120)
  (HC2: B = 5 * F)
  (HC3: tooth_extraction_cost = 290)
  (HC4: B = C + 2 * F + tooth_extraction_cost) :
  C = 70 :=
by
  sorry

end dentist_cleaning_cost_l289_289769


namespace quadratic_has_real_root_l289_289125

theorem quadratic_has_real_root (b : ℝ) : 
  (b^2 - 100 ≥ 0) ↔ (b ≤ -10 ∨ b ≥ 10) :=
by
  sorry

end quadratic_has_real_root_l289_289125


namespace james_carrot_sticks_l289_289715

theorem james_carrot_sticks (carrots_before : ℕ) (carrots_after : ℕ) 
(h_before : carrots_before = 22) (h_after : carrots_after = 15) : 
carrots_before + carrots_after = 37 := 
by 
  -- Placeholder for proof
  sorry

end james_carrot_sticks_l289_289715


namespace coefficients_of_quadratic_function_l289_289924

-- Define the quadratic function.
def quadratic_function (x : ℝ) : ℝ :=
  2 * (x - 3) ^ 2 + 2

-- Define the expected expanded form.
def expanded_form (x : ℝ) : ℝ :=
  2 * x ^ 2 - 12 * x + 20

-- State the proof problem.
theorem coefficients_of_quadratic_function :
  ∀ (x : ℝ), quadratic_function x = expanded_form x := by
  sorry

end coefficients_of_quadratic_function_l289_289924


namespace relationship_among_abc_l289_289059

theorem relationship_among_abc (x : ℝ) (e : ℝ) (ln : ℝ → ℝ) (half_pow : ℝ → ℝ) (exp : ℝ → ℝ) 
  (x_in_e_e2 : x > e ∧ x < exp 2) 
  (def_a : ln x = ln x)
  (def_b : half_pow (ln x) = ((1/2)^(ln x)))
  (def_c : exp (ln x) = x):
  (exp (ln x)) > (ln x) ∧ (ln x) > ((1/2)^(ln x)) :=
by 
  sorry

end relationship_among_abc_l289_289059


namespace total_transaction_loss_l289_289230

-- Define the cost and selling prices given the conditions
def cost_price_house (h : ℝ) := (7 / 10) * h = 15000
def cost_price_store (s : ℝ) := (5 / 4) * s = 15000

-- Define the loss calculation for the transaction
def transaction_loss : Prop :=
  ∃ (h s : ℝ),
    (7 / 10) * h = 15000 ∧
    (5 / 4) * s = 15000 ∧
    h + s - 2 * 15000 = 3428.57

-- The theorem stating the transaction resulted in a loss of $3428.57
theorem total_transaction_loss : transaction_loss :=
by
  sorry

end total_transaction_loss_l289_289230


namespace triangle_ratio_l289_289977

theorem triangle_ratio
  (D E F X : Type)
  [DecidableEq D] [DecidableEq E] [DecidableEq F] [DecidableEq X]
  (DE DF : ℝ)
  (hDE : DE = 36)
  (hDF : DF = 40)
  (DX_bisects_EDF : ∀ EX FX, (DE * FX = DF * EX)) :
  ∃ (EX FX : ℝ), EX / FX = 9 / 10 :=
sorry

end triangle_ratio_l289_289977


namespace evaluate_expression_l289_289620

theorem evaluate_expression :
  3 * 307 + 4 * 307 + 2 * 307 + 307 * 307 = 97012 := by
  sorry

end evaluate_expression_l289_289620


namespace fishing_tomorrow_l289_289888

-- Conditions
def every_day_fishers : Nat := 7
def every_other_day_fishers : Nat := 8
def every_three_days_fishers : Nat := 3
def yesterday_fishers : Nat := 12
def today_fishers : Nat := 10

-- Determine the number who will fish tomorrow
def fishers_tomorrow : Nat :=
  let every_day_tomorrow := every_day_fishers
  let every_three_day_tomorrow := every_three_days_fishers
  let every_other_day_yesterday := yesterday_fishers - every_day_fishers
  let every_other_day_tomorrow := every_other_day_fishers - every_other_day_yesterday
  every_day_tomorrow + every_three_day_tomorrow + every_other_day_tomorrow

theorem fishing_tomorrow : fishers_tomorrow = 15 :=
  by
    -- skipping the actual proof with sorry
    sorry

end fishing_tomorrow_l289_289888


namespace donation_to_first_home_l289_289180

theorem donation_to_first_home :
  let total_donation := 700
  let donation_to_second := 225
  let donation_to_third := 230
  total_donation - donation_to_second - donation_to_third = 245 :=
by
  sorry

end donation_to_first_home_l289_289180


namespace sum_of_cubes_divisible_by_9_l289_289586

theorem sum_of_cubes_divisible_by_9 (n : ℕ) : 9 ∣ (n^3 + (n + 1)^3 + (n + 2)^3) := 
  sorry

end sum_of_cubes_divisible_by_9_l289_289586


namespace min_f_value_f_achieves_min_l289_289527

noncomputable def f (x : ℝ) : ℝ :=
  x + (2 * x) / (x ^ 2 + 1) + (x * (x + 3)) / (x ^ 2 + 2) + (3 * (x + 1)) / (x * (x ^ 2 + 2))

theorem min_f_value (x : ℝ) (hx : x > 0) : f x ≥ 3 :=
sorry

theorem f_achieves_min (x : ℝ) (hx : x > 0) : ∃ x, f x = 3 :=
sorry

end min_f_value_f_achieves_min_l289_289527


namespace find_m_range_l289_289823

def A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 4 }
def B (m : ℝ) : Set ℝ := { x | -1 ≤ x ∧ x ≤ m + 1 }

theorem find_m_range (m : ℝ) : (B m ⊆ A) ↔ (-2 ≤ m ∧ m ≤ 3) := by
  sorry

end find_m_range_l289_289823


namespace normal_distribution_probability_l289_289158

theorem normal_distribution_probability (σ : ℝ) (hσ : σ > 0) :
  ∀ (ζ : ℝ → Type) [normal_distribution ζ 4 σ] (P : set ℝ → ℝ),
  (P {x | 4 < x ∧ x < 8} = 0.3) →
  (P {x | x < 0} = 0.2) :=
by
  assume ζ hζ P hP,
  sorry

end normal_distribution_probability_l289_289158


namespace complementary_combinations_count_l289_289550

-- Definitions for the problem
structure Card where
  shape   : ℕ -- 3 types (indexed by 0, 1, 2)
  color   : ℕ -- 3 types (indexed by 0, 1, 2)
  pattern : ℕ -- 3 types (indexed by 0, 1, 2)

def isComplementary (c1 c2 c3 : Card) : Prop :=
  (c1.shape = c2.shape ∧ c2.shape = c3.shape ∨ c1.shape ≠ c2.shape ∧ c2.shape ≠ c3.shape ∧ c1.shape ≠ c3.shape) ∧
  (c1.color = c2.color ∧ c2.color = c3.color ∨ c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c1.color ≠ c3.color) ∧
  (c1.pattern = c2.pattern ∧ c2.pattern = c3.pattern ∨ c1.pattern ≠ c2.pattern ∧ c2.pattern ≠ c3.pattern ∧ c1.pattern ≠ c3.pattern)

theorem complementary_combinations_count :
  (∃ (cards : Finset Card), cards.card = 27 ∧ 
  ∀ c ∈ cards, c.shape < 3 ∧ c.color < 3 ∧ c.pattern < 3) → 
  (Finset.univ.card = 27 → 
  ∃ (complementary_comb : Finset (Card × Card × Card)),
    complementary_comb.card = 117 ∧
    (∀ xyz ∈ complementary_comb, isComplementary xyz.1 xyz.2.1 xyz.2.2)) := by
  sorry

end complementary_combinations_count_l289_289550


namespace diamond_op_example_l289_289297

def diamond_op (x y : ℕ) : ℕ := 3 * x + 5 * y

theorem diamond_op_example : diamond_op 2 7 = 41 :=
by {
    -- proof goes here
    sorry
}

end diamond_op_example_l289_289297


namespace simplify_exponent_multiplication_l289_289919

theorem simplify_exponent_multiplication (x : ℝ) : x^5 * x^3 = x^8 :=
by sorry

end simplify_exponent_multiplication_l289_289919


namespace range_of_a_l289_289285

theorem range_of_a (a : ℝ) : (¬ (∃ x0 : ℝ, a * x0^2 + x0 + 1/2 ≤ 0)) → a > 1/2 :=
by
  sorry

end range_of_a_l289_289285


namespace ways_to_write_1800_as_sum_of_4s_and_5s_l289_289430

theorem ways_to_write_1800_as_sum_of_4s_and_5s : 
  ∃ S : Finset (ℕ × ℕ), S.card = 91 ∧ ∀ (nm : ℕ × ℕ), nm ∈ S ↔ 4 * nm.1 + 5 * nm.2 = 1800 ∧ nm.1 ≥ 0 ∧ nm.2 ≥ 0 :=
by
  sorry

end ways_to_write_1800_as_sum_of_4s_and_5s_l289_289430


namespace area_of_circumscribed_circle_l289_289635

noncomputable def circumradius_of_equilateral_triangle (s : ℝ) : ℝ :=
  s / Real.sqrt 3

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem area_of_circumscribed_circle (s : ℝ) (h : s = 12) : area_of_circle (circumradius_of_equilateral_triangle s) = 48 * Real.pi := by
  sorry

end area_of_circumscribed_circle_l289_289635


namespace fishing_tomorrow_l289_289882

theorem fishing_tomorrow (yesterday_fishers today_fishers : ℕ)
  (every_day_fishers every_other_day_fishers every_three_days_fishers : ℕ)
  (total_population : ℕ):
  yesterday_fishers = 12 → 
  today_fishers = 10 → 
  every_day_fishers = 7 → 
  every_other_day_fishers = 8 → 
  every_three_days_fishers = 3 → 
  total_population = yesterday_fishers + today_fishers + (total_population - (every_day_fishers + every_other_day_fishers + every_three_days_fishers)) →
  ∃ tomorrow_fishers : ℕ, tomorrow_fishers = 15 :=
by {
  -- This is a statement definition, the proof is not required and thus marked as "sorry:"
  sorry
}

end fishing_tomorrow_l289_289882


namespace inequality_solution_l289_289325

theorem inequality_solution (x : ℝ) (h : 3 * x - 5 > 11 - 2 * x) : x > 16 / 5 := 
sorry

end inequality_solution_l289_289325


namespace circle_area_from_circumference_l289_289491

theorem circle_area_from_circumference (C : ℝ) (hC : C = 48 * Real.pi) : 
  ∃ m : ℝ, (∀ r : ℝ, C = 2 * Real.pi * r → (Real.pi * r^2 = m * Real.pi)) ∧ m = 576 :=
by
  sorry

end circle_area_from_circumference_l289_289491


namespace paint_problem_l289_289585

-- Definitions based on conditions
def roomsInitiallyPaintable := 50
def roomsAfterLoss := 40
def cansLost := 5

-- The number of rooms each can could paint
def roomsPerCan := (roomsInitiallyPaintable - roomsAfterLoss) / cansLost

-- The total number of cans originally owned
def originalCans := roomsInitiallyPaintable / roomsPerCan

-- Theorem to prove the number of original cans equals 25
theorem paint_problem : originalCans = 25 := by
  sorry

end paint_problem_l289_289585


namespace wine_cost_today_l289_289220

theorem wine_cost_today (C : ℝ) (h1 : ∀ (new_tariff : ℝ), new_tariff = 0.25) (h2 : ∀ (total_increase : ℝ), total_increase = 25) (h3 : C = 20) : 5 * (1.25 * C - C) = 25 :=
by
  sorry

end wine_cost_today_l289_289220


namespace find_x_l289_289290

def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vectors_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, (u.1 * k = v.1) ∧ (u.2 * k = v.2)

theorem find_x :
  let a := (1, -2)
  let b := (3, -1)
  let c := (x, 4)
  vectors_parallel (vector_add a c) (vector_add b c) → x = 3 :=
by intros; sorry

end find_x_l289_289290


namespace solution_set_of_inequality_l289_289933

theorem solution_set_of_inequality :
  {x : ℝ | 4 * x ^ 2 - 4 * x + 1 ≤ 0} = {1 / 2} :=
by
  sorry

end solution_set_of_inequality_l289_289933


namespace quadratic_real_roots_l289_289129

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := by 
  sorry

end quadratic_real_roots_l289_289129


namespace find_percent_defective_l289_289894

def percent_defective (D : ℝ) : Prop :=
  (0.04 * D = 0.32)

theorem find_percent_defective : ∃ D, percent_defective D ∧ D = 8 := by
  sorry

end find_percent_defective_l289_289894


namespace total_distance_12_hours_l289_289949

-- Define the initial conditions for the speed and distance calculation
def speed_increase : ℕ → ℕ
  | 0 => 50
  | n + 1 => speed_increase n + 2

def distance_in_hour (n : ℕ) : ℕ := speed_increase n

-- Define the total distance traveled in 12 hours
def total_distance (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n + 1 => total_distance n + distance_in_hour n

theorem total_distance_12_hours :
  total_distance 12 = 732 := by
  sorry

end total_distance_12_hours_l289_289949


namespace quadratic_real_roots_l289_289112

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := 
by 
  sorry

end quadratic_real_roots_l289_289112


namespace calculate_weekly_charge_l289_289334

-- Defining conditions as constraints
def daily_charge : ℕ := 30
def total_days : ℕ := 11
def total_cost : ℕ := 310

-- Defining the weekly charge
def weekly_charge : ℕ := 190

-- Prove that the weekly charge for the first week of rental is $190
theorem calculate_weekly_charge (daily_charge total_days total_cost weekly_charge: ℕ) (daily_charge_eq : daily_charge = 30) (total_days_eq : total_days = 11) (total_cost_eq : total_cost = 310) : 
  weekly_charge = 190 :=
by
  sorry

end calculate_weekly_charge_l289_289334


namespace no_real_values_of_p_for_equal_roots_l289_289514

theorem no_real_values_of_p_for_equal_roots (p : ℝ) : ¬ ∃ (p : ℝ), (p^2 - 2*p + 5 = 0) :=
by sorry

end no_real_values_of_p_for_equal_roots_l289_289514


namespace avg_equivalence_l289_289179

-- Definition of binary average [a, b]
def avg2 (a b : ℤ) : ℤ := (a + b) / 2

-- Definition of ternary average {a, b, c}
def avg3 (a b c : ℤ) : ℤ := (a + b + c) / 3

-- Lean statement for proving the given problem
theorem avg_equivalence : avg3 (avg3 2 2 (-1)) (avg2 3 (-1)) 1 = 1 := by
  sorry

end avg_equivalence_l289_289179


namespace quadratic_has_real_root_l289_289124

theorem quadratic_has_real_root (b : ℝ) : 
  (b^2 - 100 ≥ 0) ↔ (b ≤ -10 ∨ b ≥ 10) :=
by
  sorry

end quadratic_has_real_root_l289_289124


namespace max_volume_day1_l289_289372

-- Define volumes of the containers
def volumes : List ℕ := [9, 13, 17, 19, 20, 38]

-- Define conditions: sold containers volumes
def condition_on_first_day (s: List ℕ) := s.length = 3
def condition_on_second_day (s: List ℕ) := s.length = 2

-- Define condition: total and relative volumes sold
def volume_sold_first_day (s: List ℕ) : ℕ := s.foldr (λ x acc => x + acc) 0
def volume_sold_second_day (s: List ℕ) : ℕ := s.foldr (λ x acc => x + acc) 0

def volume_sold_total (s1 s2: List ℕ) := volume_sold_first_day s1 + volume_sold_second_day s2 = 116
def volume_ratio (s1 s2: List ℕ) := volume_sold_first_day s1 = 2 * volume_sold_second_day s2 

-- The goal is to prove the maximum possible volume_sold_first_day
theorem max_volume_day1 (s1 s2: List ℕ) 
  (h1: condition_on_first_day s1)
  (h2: condition_on_second_day s2)
  (h3: volume_sold_total s1 s2)
  (h4: volume_ratio s1 s2) : 
  ∃(max_volume: ℕ), max_volume = 66 :=
sorry

end max_volume_day1_l289_289372


namespace find_a_and_c_range_of_m_l289_289539

theorem find_a_and_c (a c : ℝ) 
  (h : ∀ x, 1 < x ∧ x < 3 ↔ ax^2 + x + c > 0) 
  : a = -1/4 ∧ c = -3/4 := 
sorry

theorem range_of_m (m : ℝ) 
  (h : ∀ x, (-1/4)*x^2 + 2*x - 3 > 0 → x + m > 0) 
  : m ≥ -2 :=
sorry

end find_a_and_c_range_of_m_l289_289539


namespace total_feet_is_140_l289_289496

def total_heads : ℕ := 48
def number_of_hens : ℕ := 26
def number_of_cows : ℕ := total_heads - number_of_hens
def feet_per_hen : ℕ := 2
def feet_per_cow : ℕ := 4

theorem total_feet_is_140 : ((number_of_hens * feet_per_hen) + (number_of_cows * feet_per_cow)) = 140 := by
  sorry

end total_feet_is_140_l289_289496


namespace consecutive_digits_sum_190_to_199_l289_289260

-- Define the digits sum function
def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define ten consecutive numbers starting from m
def ten_consecutive_sum (m : ℕ) : ℕ :=
  (List.range 10).map (λ i => digits_sum (m + i)) |>.sum

theorem consecutive_digits_sum_190_to_199:
  ten_consecutive_sum 190 = 145 :=
by
  sorry

end consecutive_digits_sum_190_to_199_l289_289260


namespace find_positive_integer_n_l289_289481

theorem find_positive_integer_n (n : ℕ) (h₁ : 200 % n = 5) (h₂ : 395 % n = 5) : n = 13 :=
sorry

end find_positive_integer_n_l289_289481


namespace num_O_atoms_l289_289012

def compound_molecular_weight : ℕ := 62
def atomic_weight_H : ℕ := 1
def atomic_weight_C : ℕ := 12
def atomic_weight_O : ℕ := 16
def num_H_atoms : ℕ := 2
def num_C_atoms : ℕ := 1

theorem num_O_atoms (H_weight : ℕ := num_H_atoms * atomic_weight_H)
                    (C_weight : ℕ := num_C_atoms * atomic_weight_C)
                    (total_weight : ℕ := compound_molecular_weight)
                    (O_weight := atomic_weight_O) : 
    (total_weight - (H_weight + C_weight)) / O_weight = 3 :=
by
  sorry

end num_O_atoms_l289_289012


namespace smallest_y_square_factor_l289_289852

theorem smallest_y_square_factor (y n : ℕ) (h₀ : y = 10) 
  (h₁ : ∀ m : ℕ, ∃ k : ℕ, k * k = m * y)
  (h₂ : ∀ (y' : ℕ), (∀ m : ℕ, ∃ k : ℕ, k * k = m * y') → y ≤ y') : 
  n = 10 :=
by sorry

end smallest_y_square_factor_l289_289852


namespace compute_expr_l289_289251

-- Definitions
def a := 150 / 5
def b := 40 / 8
def c := 16 / 32
def d := 3

def expr := 20 * (a - b + c + d)

-- Theorem
theorem compute_expr : expr = 570 :=
by
  sorry

end compute_expr_l289_289251


namespace negation_proof_l289_289750

theorem negation_proof (a b : ℝ) : 
  (¬ (a > b → 2 * a > 2 * b - 1)) = (a ≤ b → 2 * a ≤ 2 * b - 1) :=
by
  sorry

end negation_proof_l289_289750


namespace yangyang_departure_time_l289_289483

noncomputable def departure_time : Nat := 373 -- 6:13 in minutes from midnight (6 * 60 + 13)

theorem yangyang_departure_time :
  let arrival_at_60_mpm := 413 -- 6:53 in minutes from midnight
  let arrival_at_75_mpm := 405 -- 6:45 in minutes from midnight
  let difference := arrival_at_60_mpm - arrival_at_75_mpm -- time difference
  let x := 40 -- time taken to walk to school at 60 meters per minute
  departure_time = arrival_at_60_mpm - x :=
by
  -- Definitions
  let arrival_at_60_mpm := 413
  let arrival_at_75_mpm := 405
  let difference := 8
  let x := 40
  have h : departure_time = (413 - 40) := rfl
  sorry

end yangyang_departure_time_l289_289483


namespace inequality_for_pos_reals_l289_289596

-- Definitions for positive real numbers
variables {x y : ℝ}
def is_pos_real (x : ℝ) : Prop := x > 0

-- Theorem statement
theorem inequality_for_pos_reals (hx : is_pos_real x) (hy : is_pos_real y) : 
  2 * (x^2 + y^2) ≥ (x + y)^2 :=
by
  sorry

end inequality_for_pos_reals_l289_289596


namespace expected_value_constant_random_variable_l289_289646

noncomputable def X : ℕ → ℝ := λ n, 7

theorem expected_value_constant_random_variable :
  (∀ n, X n = 7) → ⁅X⁆ = 7 :=
begin
  intro h,
  sorry
end

end expected_value_constant_random_variable_l289_289646


namespace abby_bridget_adjacent_probability_l289_289241

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def probability_adjacent : ℚ :=
  let total_seats := 9
  let ab_adj_same_row_pairs := 9
  let ab_adj_diagonal_pairs := 4
  let favorable_outcomes := (ab_adj_same_row_pairs + ab_adj_diagonal_pairs) * 2 * factorial 7
  let total_outcomes := factorial total_seats
  favorable_outcomes / total_outcomes

theorem abby_bridget_adjacent_probability :
  probability_adjacent = 13 / 36 :=
by
  sorry

end abby_bridget_adjacent_probability_l289_289241


namespace find_h_plus_k_l289_289812

theorem find_h_plus_k (h k : ℝ) :
  (∀ (x y : ℝ),
    (x - 3) ^ 2 + (y + 4) ^ 2 = 49) → 
  h = 3 ∧ k = -4 → 
  h + k = -1 :=
by
  sorry

end find_h_plus_k_l289_289812


namespace real_solution_count_l289_289156

/-- Given \( \lfloor x \rfloor \) is the greatest integer less than or equal to \( x \),
prove that the number of real solutions to the equation \( 9x^2 - 36\lfloor x \rfloor + 20 = 0 \) is 2. --/
theorem real_solution_count (x : ℝ) (h : ⌊x⌋ = Int.floor x) :
  ∃ (S : Finset ℝ), S.card = 2 ∧ ∀ a ∈ S, 9 * a^2 - 36 * ⌊a⌋ + 20 = 0 :=
sorry

end real_solution_count_l289_289156


namespace no_rational_solutions_l289_289045

theorem no_rational_solutions : 
  ¬ ∃ (x y z : ℚ), 11 = x^5 + 2 * y^5 + 5 * z^5 := 
sorry

end no_rational_solutions_l289_289045


namespace hyperbolic_identity_l289_289437

noncomputable def sh (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2
noncomputable def ch (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2

theorem hyperbolic_identity (x : ℝ) : (ch x) ^ 2 - (sh x) ^ 2 = 1 := 
sorry

end hyperbolic_identity_l289_289437


namespace solve_inequality_l289_289814

open Set Real

def condition1 (x : ℝ) : Prop := 6 * x + 2 < (x + 2) ^ 2
def condition2 (x : ℝ) : Prop := (x + 2) ^ 2 < 8 * x + 4

theorem solve_inequality (x : ℝ) : condition1 x ∧ condition2 x ↔ x ∈ Ioo (2 + Real.sqrt 2) 4 := by
  sorry

end solve_inequality_l289_289814


namespace quadratic_roots_interval_l289_289091

theorem quadratic_roots_interval (b : ℝ) :
  ∃ (x : ℝ), x^2 + b * x + 25 = 0 → b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_roots_interval_l289_289091


namespace arithmetic_sequence_squares_l289_289168

theorem arithmetic_sequence_squares (a b c : ℝ) :
  (1 / (a + b) - 1 / (b + c) = 1 / (c + a) - 1 / (b + c)) →
  (2 * b^2 = a^2 + c^2) :=
by
  intro h
  sorry

end arithmetic_sequence_squares_l289_289168


namespace count_even_factors_is_correct_l289_289804

def prime_factors_444_533_72 := (2^8 * 5^3 * 7^2)

def range_a := {a : ℕ | 0 ≤ a ∧ a ≤ 8}
def range_b := {b : ℕ | 0 ≤ b ∧ b ≤ 3}
def range_c := {c : ℕ | 0 ≤ c ∧ c ≤ 2}

def even_factors_count : ℕ :=
  (8 - 1 + 1) * (3 - 0 + 1) * (2 - 0 + 1)

theorem count_even_factors_is_correct :
  even_factors_count = 96 := by
  sorry

end count_even_factors_is_correct_l289_289804


namespace trey_total_hours_l289_289580

def num_clean_house := 7
def num_shower := 1
def num_make_dinner := 4
def minutes_per_item := 10
def total_items := num_clean_house + num_shower + num_make_dinner
def total_minutes := total_items * minutes_per_item
def minutes_in_hour := 60

theorem trey_total_hours : total_minutes / minutes_in_hour = 2 := by
  sorry

end trey_total_hours_l289_289580


namespace sum_of_areas_l289_289453

def base_width : ℕ := 3
def lengths : List ℕ := [1, 8, 27, 64, 125, 216]
def area (w l : ℕ) : ℕ := w * l
def total_area : ℕ := (lengths.map (area base_width)).sum

theorem sum_of_areas : total_area = 1323 := 
by sorry

end sum_of_areas_l289_289453


namespace least_number_of_groups_l289_289504

def num_students : ℕ := 24
def max_students_per_group : ℕ := 10

theorem least_number_of_groups : ∃ x, ∀ y, y ≤ max_students_per_group ∧ num_students = x * y → x = 3 := by
  sorry

end least_number_of_groups_l289_289504


namespace problem_l289_289703

-- Helper definition for point on a line
def point_on_line (x y : ℝ) (a b : ℝ) : Prop := y = a * x + b

-- Given condition: Point P(1, 3) lies on the line y = 2x + b
def P_on_l (b : ℝ) : Prop := point_on_line 1 3 2 b

-- The proof problem: Proving (2, 5) also lies on the line y = 2x + b where b is the constant found using P
theorem problem (b : ℝ) (h: P_on_l b) : point_on_line 2 5 2 b :=
by
  sorry

end problem_l289_289703


namespace abigail_monthly_saving_l289_289799

-- Definitions based on the conditions
def total_saving := 48000
def months_in_year := 12

-- The statement to be proved
theorem abigail_monthly_saving : total_saving / months_in_year = 4000 :=
by sorry

end abigail_monthly_saving_l289_289799


namespace probability_of_perfect_square_is_correct_l289_289965

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def probability_perfect_square (p : ℚ) : ℚ :=
  let less_than_equal_60 := 7 * p
  let greater_than_60 := 4 * 4 * p
  less_than_equal_60 + greater_than_60

theorem probability_of_perfect_square_is_correct :
  let p : ℚ := 1 / 300
  probability_perfect_square p = 23 / 300 :=
sorry

end probability_of_perfect_square_is_correct_l289_289965


namespace perpendicular_vectors_solution_l289_289072

theorem perpendicular_vectors_solution (m : ℝ) (a : ℝ × ℝ := (m-1, 2)) (b : ℝ × ℝ := (m, -3)) 
  (h : a.1 * b.1 + a.2 * b.2 = 0) : m = 3 ∨ m = -2 :=
by sorry

end perpendicular_vectors_solution_l289_289072


namespace am_gm_inequality_l289_289700

theorem am_gm_inequality (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) : 
  8 * a * b * c ≤ (a + b) * (b + c) * (c + a) := 
by
  sorry

end am_gm_inequality_l289_289700


namespace attendees_count_l289_289518

def n_students_seated : ℕ := 300
def n_students_standing : ℕ := 25
def n_teachers_seated : ℕ := 30

def total_attendees : ℕ :=
  n_students_seated + n_students_standing + n_teachers_seated

theorem attendees_count :
  total_attendees = 355 := by
  sorry

end attendees_count_l289_289518


namespace employee_count_l289_289182

theorem employee_count (avg_salary : ℕ) (manager_salary : ℕ) (new_avg_increase : ℕ) (E : ℕ) :
  (avg_salary = 1500) ∧ (manager_salary = 4650) ∧ (new_avg_increase = 150) →
  1500 * E + 4650 = 1650 * (E + 1) → E = 20 :=
by
  sorry

end employee_count_l289_289182


namespace radius_of_inner_circle_l289_289473

def right_triangle_legs (AC BC : ℝ) : Prop :=
  AC = 3 ∧ BC = 4

theorem radius_of_inner_circle (AC BC : ℝ) (h : right_triangle_legs AC BC) :
  ∃ r : ℝ, r = 2 :=
by
  sorry

end radius_of_inner_circle_l289_289473


namespace checkerboard_fraction_checkerboard_sum_mn_l289_289368

open Nat

-- Condition definitions
def checkerboard_size : ℕ := 10
def horizontal_lines : ℕ := checkerboard_size + 1
def vertical_lines : ℕ := checkerboard_size + 1

-- Number of rectangles calculation
def num_rectangles : ℕ :=
  (choose horizontal_lines 2) * (choose vertical_lines 2)

-- Number of squares calculation
def num_squares : ℕ :=
  ∑ i in Icc 1 checkerboard_size, i * i

-- Simplified fraction of squares to rectangles
def fraction_squares_to_rectangles : ℚ :=
  (num_squares : ℚ) / num_rectangles

theorem checkerboard_fraction :
  fraction_squares_to_rectangles = 7 / 55 :=
by {
  -- The proof would go here, but it's skipped with sorry
  sorry
}

theorem checkerboard_sum_mn :
  7 + 55 = 62 :=
by {
  -- This is a simple arithmetic proof
  exact rfl
}

end checkerboard_fraction_checkerboard_sum_mn_l289_289368


namespace fishing_tomorrow_l289_289862

theorem fishing_tomorrow 
  (P1 : ∀ day : ℕ, 7 ∈ {p | p goes fishing on day})
  (P2 : ∀ day : ℕ, day % 2 = 0 → 8 ∈ {p | p goes fishing on day})
  (P3 : ∀ day : ℕ, day % 3 = 0 → 3 ∈ {p | p goes fishing on day})
  (P4 : ℕ)
  (yesterday : ℕ)
  (today : ℕ)
  (tomorrow : ℕ)
  (hyesterday : yesterday = 12)
  (htoday : today = 10)
  : tomorrow = 15 := by
  sorry

end fishing_tomorrow_l289_289862


namespace find_three_numbers_l289_289817

theorem find_three_numbers (x y z : ℝ)
  (h1 : x - y = (1 / 3) * z)
  (h2 : y - z = (1 / 3) * x)
  (h3 : z - 10 = (1 / 3) * y) :
  x = 45 ∧ y = 37.5 ∧ z = 22.5 :=
by
  sorry

end find_three_numbers_l289_289817


namespace oil_needed_to_half_fill_tanker_l289_289768

theorem oil_needed_to_half_fill_tanker :
  let initial_tank_capacity := 4000
  let poured_fraction := 3 / 4
  let initial_tanker_capacity := 20000
  let initial_tanker_oil := 3000
  let poured_oil := poured_fraction * initial_tank_capacity
  let current_tanker_oil := initial_tanker_oil + poured_oil
  let half_tanker_capacity := initial_tanker_capacity / 2
  let needed_oil := half_tanker_capacity - current_tanker_oil
  needed_oil = 4000 :=
by
  let initial_tank_capacity := 4000
  let poured_fraction := 3 / 4
  let initial_tanker_capacity := 20000
  let initial_tanker_oil := 3000
  let poured_oil := poured_fraction * initial_tank_capacity
  have h1 : poured_oil = 3000 := by sorry
  let current_tanker_oil := initial_tanker_oil + poured_oil
  have h2 : current_tanker_oil = 6000 := by sorry
  let half_tanker_capacity := initial_tanker_capacity / 2
  have h3 : half_tanker_capacity = 10000 := by sorry
  let needed_oil := half_tanker_capacity - current_tanker_oil
  have h4 : needed_oil = 4000 := by sorry
  exact h4

end oil_needed_to_half_fill_tanker_l289_289768


namespace soccer_team_lineups_l289_289162

theorem soccer_team_lineups :
  let total_players := 18
  let goalkeeper_choices := total_players
  let defender_choices := Nat.choose (total_players - 1) 4
  let midfield_or_forward_choices := Nat.choose (total_players - 1 - 4) 4
  goalkeeper_choices * defender_choices * midfield_or_forward_choices = 30_544_200 :=
by
  sorry

end soccer_team_lineups_l289_289162


namespace selling_price_of_book_l289_289791

theorem selling_price_of_book (cost_price : ℕ) (profit_rate : ℕ) (profit : ℕ) (selling_price : ℕ) :
  cost_price = 50 → profit_rate = 80 → profit = (profit_rate * cost_price) / 100 → selling_price = cost_price + profit → selling_price = 90 :=
by
  intros h_cost_price h_profit_rate h_profit h_selling_price
  rw [h_cost_price, h_profit_rate] at h_profit
  simp at h_profit
  rw [h_cost_price, h_profit] at h_selling_price
  exact h_selling_price

end selling_price_of_book_l289_289791


namespace number_is_37_5_l289_289546

theorem number_is_37_5 (y : ℝ) (h : 0.4 * y = 15) : y = 37.5 :=
sorry

end number_is_37_5_l289_289546


namespace base_salary_l289_289009

theorem base_salary {B : ℝ} {C : ℝ} :
  (B + 200 * C = 2000) → 
  (B + 200 * 15 = 4000) → 
  B = 1000 :=
by
  sorry

end base_salary_l289_289009


namespace peg_arrangement_l289_289343

theorem peg_arrangement :
  let Y := 5
  let R := 4
  let G := 3
  let B := 2
  let O := 1
  (Y! * R! * G! * B! * O!) = 34560 :=
by
  sorry

end peg_arrangement_l289_289343


namespace license_plates_count_l289_289291

def numConsonantsExcludingY : Nat := 19
def numVowelsIncludingY : Nat := 6
def numConsonantsIncludingY : Nat := 21
def numEvenDigits : Nat := 5

theorem license_plates_count : 
  numConsonantsExcludingY * numVowelsIncludingY * numConsonantsIncludingY * numEvenDigits = 11970 := by
  sorry

end license_plates_count_l289_289291


namespace evaluate_expression_l289_289035

theorem evaluate_expression :
  let a := 2020
  let b := 2016
  (2^a + 2^b) / (2^a - 2^b) = 17 / 15 :=
by
  sorry

end evaluate_expression_l289_289035


namespace singh_gain_l289_289029

def initial_amounts (B A S : ℕ) : Prop :=
  B = 70 ∧ A = 70 ∧ S = 70

def ratio_Ashtikar_Singh (A S : ℕ) : Prop :=
  2 * A = S

def ratio_Singh_Bhatia (S B : ℕ) : Prop :=
  4 * B = S

def total_conservation (A S B : ℕ) : Prop :=
  A + S + B = 210

theorem singh_gain : ∀ B A S fA fB fS : ℕ,
  initial_amounts B A S →
  ratio_Ashtikar_Singh fA fS →
  ratio_Singh_Bhatia fS fB →
  total_conservation fA fS fB →
  fS - S = 50 :=
by
  intros B A S fA fB fS
  intros i rA rS tC
  sorry

end singh_gain_l289_289029


namespace math_proof_statement_l289_289428

open Real

noncomputable def proof_problem (x : ℝ) : Prop :=
  let a := (cos x, sin x)
  let b := (sqrt 2, sqrt 2)
  (a.1 * b.1 + a.2 * b.2 = 8 / 5) ∧ (π / 4 < x ∧ x < π / 2) ∧ 
  (cos (x - π / 4) = 4 / 5) ∧ (tan (x - π / 4) = 3 / 4) ∧ 
  (sin (2 * x) * (1 - tan x) / (1 + tan x) = -21 / 100)

theorem math_proof_statement (x : ℝ) : proof_problem x := 
by
  unfold proof_problem
  sorry

end math_proof_statement_l289_289428


namespace sector_area_l289_289834

theorem sector_area (r l : ℝ) (h1 : l + 2 * r = 8) (h2 : l = 2 * r) : 
  (1 / 2) * l * r = 4 := 
by sorry

end sector_area_l289_289834


namespace comb_identity_a_l289_289592

theorem comb_identity_a (r m k : ℕ) (h : 0 ≤ k ∧ k ≤ m ∧ m ≤ r) :
  Nat.choose r m * Nat.choose m k = Nat.choose r k * Nat.choose (r - k) (m - k) :=
sorry

end comb_identity_a_l289_289592


namespace ab_finish_job_in_15_days_l289_289361

theorem ab_finish_job_in_15_days (A B C : ℝ) (h1 : A + B + C = 1/12) (h2 : C = 1/60) : 1 / (A + B) = 15 := 
by
  sorry

end ab_finish_job_in_15_days_l289_289361


namespace more_supermarkets_in_us_l289_289611

-- Definitions based on conditions
def total_supermarkets : ℕ := 84
def us_supermarkets : ℕ := 47
def canada_supermarkets : ℕ := total_supermarkets - us_supermarkets

-- Prove that the number of more FGH supermarkets in the US than in Canada is 10
theorem more_supermarkets_in_us : us_supermarkets - canada_supermarkets = 10 :=
by
  -- adding 'sorry' as the proof
  sorry

end more_supermarkets_in_us_l289_289611


namespace problem_quadratic_has_real_root_l289_289115

theorem problem_quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end problem_quadratic_has_real_root_l289_289115


namespace certain_number_l289_289136

theorem certain_number (x y : ℝ) (h1 : 0.20 * x = 0.15 * y - 15) (h2 : x = 1050) : y = 1500 :=
by
  sorry

end certain_number_l289_289136


namespace quadratic_has_real_root_iff_b_in_interval_l289_289097

theorem quadratic_has_real_root_iff_b_in_interval (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ Set.Iic (-10) ∪ Set.Ici 10) := 
sorry

end quadratic_has_real_root_iff_b_in_interval_l289_289097


namespace sum_of_min_max_l289_289562

-- Define the necessary parameters and conditions
variables (n k : ℕ)
  (h_pos_nk : 0 < n ∧ 0 < k)
  (f : ℕ → ℕ)
  (h_toppings : ∀ t, (0 ≤ f t ∧ f t ≤ n) ∧ (f t + f (t + k) % (2 * k) = n))
  (m M : ℕ)
  (h_m : ∀ t, m ≤ f t)
  (h_M : ∀ t, f t ≤ M)
  (h_min_max : ∃ t_min t_max, m = f t_min ∧ M = f t_max)

-- The goal is to prove that the sum of m and M equals n
theorem sum_of_min_max (n k : ℕ) (h_pos_nk : 0 < n ∧ 0 < k)
  (f : ℕ → ℕ) (h_toppings : ∀ t, (0 ≤ f t ∧ f t ≤ n) ∧ (f t + f (t + k) % (2 * k) = n))
  (m M : ℕ) (h_m : ∀ t, m ≤ f t)
  (h_M : ∀ t, f t ≤ M)
  (h_min_max : ∃ t_min t_max, m = f t_min ∧ M = f t_max) :
  m + M = n := 
sorry

end sum_of_min_max_l289_289562


namespace sin_X_value_l289_289181

theorem sin_X_value (a b X : ℝ) (h₁ : (1/2) * a * b * Real.sin X = 72) (h₂ : Real.sqrt (a * b) = 16) :
  Real.sin X = 9 / 16 := by
  sorry

end sin_X_value_l289_289181


namespace circus_tent_capacity_l289_289938

theorem circus_tent_capacity (num_sections : ℕ) (people_per_section : ℕ) 
  (h1 : num_sections = 4) (h2 : people_per_section = 246) :
  num_sections * people_per_section = 984 :=
by
  sorry

end circus_tent_capacity_l289_289938


namespace substitution_and_elimination_l289_289174

theorem substitution_and_elimination {x y : ℝ} :
  y = 2 * x + 1 → 5 * x - 2 * y = 7 → 5 * x - 4 * x - 2 = 7 :=
by
  intros h₁ h₂
  rw [h₁] at h₂
  exact h₂

end substitution_and_elimination_l289_289174


namespace volume_expansion_rate_l289_289797

theorem volume_expansion_rate (R m : ℝ) (h1 : R = 1) (h2 : (4 * π * (m^3 - 1) / 3) / (m - 1) = 28 * π / 3) : m = 2 :=
sorry

end volume_expansion_rate_l289_289797


namespace find_numbers_l289_289412

theorem find_numbers (x y : ℝ) (r : ℝ) (d : ℝ) 
  (h_geom_x : x = 5 * r) 
  (h_geom_y : y = 5 * r^2)
  (h_arith_1 : y = x + d) 
  (h_arith_2 : 15 = y + d) : 
  x + y = 10 :=
by
  sorry

end find_numbers_l289_289412


namespace angle_sum_proof_l289_289712

theorem angle_sum_proof (x y : ℝ) (h : 3 * x + 6 * x + (x + y) + 4 * y = 360) : x = 0 ∧ y = 72 :=
by {
  sorry
}

end angle_sum_proof_l289_289712


namespace connections_in_computer_lab_l289_289192

theorem connections_in_computer_lab (n : ℕ) (d : ℕ) (h1 : n = 30) (h2 : d = 4) :
  (n * d) / 2 = 60 := by
  sorry

end connections_in_computer_lab_l289_289192


namespace averageTemperature_is_99_l289_289973

-- Define the daily temperatures
def tempSunday : ℝ := 99.1
def tempMonday : ℝ := 98.2
def tempTuesday : ℝ := 98.7
def tempWednesday : ℝ := 99.3
def tempThursday : ℝ := 99.8
def tempFriday : ℝ := 99
def tempSaturday : ℝ := 98.9

-- Define the number of days
def numDays : ℝ := 7

-- Define the total temperature
def totalTemp : ℝ := tempSunday + tempMonday + tempTuesday + tempWednesday + tempThursday + tempFriday + tempSaturday

-- Define the average temperature
def averageTemp : ℝ := totalTemp / numDays

-- The theorem to prove
theorem averageTemperature_is_99 : averageTemp = 99 := by
  sorry

end averageTemperature_is_99_l289_289973


namespace find_a_l289_289315

theorem find_a (a : ℝ) :
  let A := {5}
  let B := { x : ℝ | a * x - 1 = 0 }
  A ∩ B = B ↔ (a = 0 ∨ a = 1 / 5) :=
by
  sorry

end find_a_l289_289315


namespace place_signs_correct_l289_289450

theorem place_signs_correct :
  1 * 3 / 3^2 / 3^4 / 3^8 * 3^16 * 3^32 * 3^64 = 3^99 :=
by
  sorry

end place_signs_correct_l289_289450


namespace find_x_squared_plus_inv_squared_l289_289278

noncomputable def x : ℝ := sorry

theorem find_x_squared_plus_inv_squared (h : x^4 + 1 / x^4 = 240) : x^2 + 1 / x^2 = Real.sqrt 242 := by
  sorry

end find_x_squared_plus_inv_squared_l289_289278


namespace Trent_tears_l289_289348

def onions_per_pot := 4
def pots_of_soup := 6
def tears_per_3_onions := 2

theorem Trent_tears:
  (onions_per_pot * pots_of_soup) / 3 * tears_per_3_onions = 16 :=
by
  sorry

end Trent_tears_l289_289348


namespace find_x_l289_289300

def myOperation (x y : ℝ) : ℝ := 2 * x * y

theorem find_x (x : ℝ) (h : myOperation 9 (myOperation 4 x) = 720) : x = 5 :=
by
  sorry

end find_x_l289_289300


namespace circumscribed_circle_radius_l289_289709

theorem circumscribed_circle_radius (b c : ℝ) (cosA : ℝ)
  (hb : b = 2) (hc : c = 3) (hcosA : cosA = 1 / 3) : 
  R = 9 * Real.sqrt 2 / 8 :=
by
  sorry

end circumscribed_circle_radius_l289_289709


namespace unique_four_digit_numbers_l289_289247

theorem unique_four_digit_numbers (digits : Finset ℕ) (odd_digits : Finset ℕ) :
  digits = {2, 3, 4, 5, 6} → 
  odd_digits = {3, 5} → 
  ∃ (n : ℕ), n = 14 :=
by
  sorry

end unique_four_digit_numbers_l289_289247


namespace imaginary_part_of_z_l289_289277

namespace ComplexNumberProof

-- Define the imaginary unit
def i : ℂ := ⟨0, 1⟩

-- Define the complex number
def z : ℂ := i^2 * (1 + i)

-- Prove the imaginary part of z is -1
theorem imaginary_part_of_z : z.im = -1 := by
    -- Proof goes here
    sorry

end ComplexNumberProof

end imaginary_part_of_z_l289_289277


namespace probability_of_odd_die_roll_probability_of_double_tail_coin_flip_l289_289771

theorem probability_of_odd_die_roll : 
  let outcomes := {1, 2, 3, 4, 5, 6}
  let odd_outcomes := {1, 3, 5}
  outcomes.size = 6 → odd_outcomes.size = 3 →
  (odd_outcomes.size.to_rat / outcomes.size.to_rat) = 1 / 2 :=
by
  sorry

theorem probability_of_double_tail_coin_flip :
  let coin_outcomes := {'HH', 'HT', 'TH', 'TT'}
  let tail_outcomes := {'TT'}
  coin_outcomes.size = 4 → tail_outcomes.size = 1 →
  (tail_outcomes.size.to_rat / coin_outcomes.size.to_rat) = 1 / 4 :=
by
  sorry

end probability_of_odd_die_roll_probability_of_double_tail_coin_flip_l289_289771


namespace find_q_l289_289565

-- Define the roots of the polynomial 2x^2 - 6x + 1 = 0
def roots_of_first_poly (a b : ℝ) : Prop :=
    2 * a^2 - 6 * a + 1 = 0 ∧ 2 * b^2 - 6 * b + 1 = 0

-- Conditions from Vieta's formulas for the first polynomial
def sum_of_roots (a b : ℝ) : Prop := a + b = 3
def product_of_roots (a b : ℝ) : Prop := a * b = 0.5

-- Define the roots of the second polynomial x^2 + px + q = 0
def roots_of_second_poly (a b : ℝ) (p q : ℝ) : Prop :=
    (λ x => x^2 + p * x + q) (3 * a - 1) = 0 ∧ 
    (λ x => x^2 + p * x + q) (3 * b - 1) = 0

-- Proof that q = -0.5 given the conditions
theorem find_q (a b p q : ℝ) (h1 : roots_of_first_poly a b) (h2 : sum_of_roots a b)
    (h3 : product_of_roots a b) (h4 : roots_of_second_poly a b p q) : q = -0.5 :=
by
  sorry

end find_q_l289_289565


namespace number_of_operations_to_equal_l289_289764

theorem number_of_operations_to_equal (a b : ℤ) (da db : ℤ) (initial_diff change_per_operation : ℤ) (n : ℤ) 
(h1 : a = 365) 
(h2 : b = 24) 
(h3 : da = 19) 
(h4 : db = 12) 
(h5 : initial_diff = a - b) 
(h6 : change_per_operation = da + db) 
(h7 : initial_diff = 341) 
(h8 : change_per_operation = 31) 
(h9 : initial_diff = change_per_operation * n) :
n = 11 := 
by
  sorry

end number_of_operations_to_equal_l289_289764


namespace y_coordinate_of_point_on_line_l289_289307

theorem y_coordinate_of_point_on_line (x y : ℝ) (h1 : -4 = x) (h2 : ∃ m b : ℝ, y = m * x + b ∧ y = 3 ∧ x = 10 ∧ m * 4 + b = 0) : y = -4 :=
sorry

end y_coordinate_of_point_on_line_l289_289307


namespace dimes_in_piggy_bank_l289_289964

variable (q d : ℕ)

def total_coins := q + d = 100
def total_amount := 25 * q + 10 * d = 1975

theorem dimes_in_piggy_bank (h1 : total_coins q d) (h2 : total_amount q d) : d = 35 := by
  sorry

end dimes_in_piggy_bank_l289_289964


namespace total_bottles_in_market_l289_289763

theorem total_bottles_in_market (j w : ℕ) (hj : j = 34) (hw : w = 3 / 2 * j + 3) : j + w = 88 :=
by
  sorry

end total_bottles_in_market_l289_289763


namespace quad_form_unique_solution_l289_289145

theorem quad_form_unique_solution (d e f : ℤ) (h1 : d * d = 16) (h2 : 2 * d * e = -40) (h3 : e * e + f = -56) : d * e = -20 :=
by sorry

end quad_form_unique_solution_l289_289145


namespace pastries_made_l289_289970

theorem pastries_made (P cakes_sold pastries_sold extra_pastries : ℕ)
  (h1 : cakes_sold = 78)
  (h2 : pastries_sold = 154)
  (h3 : extra_pastries = 76)
  (h4 : pastries_sold = cakes_sold + extra_pastries) :
  P = 154 := sorry

end pastries_made_l289_289970


namespace items_counted_l289_289014

def convert_counter (n : Nat) : Nat := sorry

theorem items_counted
  (counter_reading : Nat) 
  (condition_1 : ∀ d, d ∈ [5, 6, 7] → ¬(d ∈ [0, 1, 2, 3, 4, 8, 9]))
  (condition_2 : ∀ d1 d2, d1 = 4 → d2 = 8 → ¬(d2 = 5 ∨ d2 = 6 ∨ d2 = 7)) :
  convert_counter 388 = 151 :=
sorry

end items_counted_l289_289014


namespace greatest_possible_perimeter_l289_289303

theorem greatest_possible_perimeter (x : ℤ) (hx1 : 3 * x > 17) (hx2 : 17 > x) : 
  (3 * x + 17 ≤ 65) :=
by
  have Hx : x ≤ 16 := sorry -- Derived from inequalities hx1 and hx2
  have Hx_ge_6 : x ≥ 6 := sorry -- Derived from integer constraint and hx1, hx2
  sorry -- Show 3 * x + 17 has maximum value 65 when x = 16

end greatest_possible_perimeter_l289_289303


namespace largest_divisor_of_n_l289_289624

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : ∃ k : ℕ, n^2 = 18 * k) : ∃ l : ℕ, n = 6 * l :=
sorry

end largest_divisor_of_n_l289_289624


namespace triangle_area_l289_289375

/-- The area of the triangle enclosed by a line with slope -1/2 passing through (2, -3) and the coordinate axes is 4. -/
theorem triangle_area {l : ℝ → ℝ} (h1 : ∀ x, l x = -1/2 * x + b)
  (h2 : l 2 = -3) : 
  ∃ (A : ℝ) (B : ℝ), 
  ((l 0 = B) ∧ (l A = 0) ∧ (A ≠ 0) ∧ (B ≠ 0)) ∧
  (1/2 * |A| * |B| = 4) := 
sorry

end triangle_area_l289_289375


namespace emily_small_gardens_l289_289402

theorem emily_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) (num_small_gardens : ℕ) :
  total_seeds = 41 →
  big_garden_seeds = 29 →
  seeds_per_small_garden = 4 →
  num_small_gardens = (total_seeds - big_garden_seeds) / seeds_per_small_garden →
  num_small_gardens = 3 :=
by
  intros h_total h_big h_seeds_per_small h_num_small
  rw [h_total, h_big, h_seeds_per_small] at h_num_small
  exact h_num_small

end emily_small_gardens_l289_289402


namespace quadratic_polynomial_l289_289540

theorem quadratic_polynomial (x y : ℝ) (hx : x + y = 12) (hy : x * (3 * y) = 108) : 
  (t : ℝ) → t^2 - 12 * t + 36 = 0 :=
by 
  sorry

end quadratic_polynomial_l289_289540


namespace interest_years_eq_three_l289_289502

theorem interest_years_eq_three :
  ∀ (x y : ℝ),
    (x + 1720 = 2795) →
    (x * (3 / 100) * 8 = 1720 * (5 / 100) * y) →
    y = 3 :=
by
  intros x y hsum heq
  sorry

end interest_years_eq_three_l289_289502


namespace greatest_three_digit_multiple_of_thirteen_l289_289195

theorem greatest_three_digit_multiple_of_thirteen : 
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (13 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000) ∧ (13 ∣ m) → m ≤ n) ∧ n = 988 :=
  sorry

end greatest_three_digit_multiple_of_thirteen_l289_289195


namespace maria_bottles_proof_l289_289907

theorem maria_bottles_proof 
    (initial_bottles : ℕ)
    (drank_bottles : ℕ)
    (current_bottles : ℕ)
    (bought_bottles : ℕ) 
    (h1 : initial_bottles = 14)
    (h2 : drank_bottles = 8)
    (h3 : current_bottles = 51)
    (h4 : current_bottles = initial_bottles - drank_bottles + bought_bottles) :
  bought_bottles = 45 :=
by
  sorry

end maria_bottles_proof_l289_289907


namespace solution_sets_and_range_l289_289954

theorem solution_sets_and_range 
    (x a : ℝ) 
    (A : Set ℝ)
    (M : Set ℝ) :
    (∀ x, x ∈ A ↔ 1 ≤ x ∧ x ≤ 4) ∧
    (M = {x | (x - a) * (x - 2) ≤ 0} ) ∧
    (M ⊆ A) → (1 ≤ a ∧ a ≤ 4) :=
by
  sorry

end solution_sets_and_range_l289_289954


namespace quadratic_real_root_iff_b_range_l289_289104

open Real

theorem quadratic_real_root_iff_b_range (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_real_root_iff_b_range_l289_289104


namespace max_value_of_f_l289_289749

def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 1 / x else -x^2 + 2

theorem max_value_of_f : ∃ m, ∀ x, f(x) ≤ m ∧ m = 2 :=
by
  sorry

end max_value_of_f_l289_289749


namespace find_theta_l289_289836

variable (x : ℝ) (θ : ℝ) (k : ℤ)

def condition := (3 - 3^(-|x - 3|))^2 = 3 - Real.cos θ

theorem find_theta (h : condition x θ) : ∃ k : ℤ, θ = (2 * k + 1) * Real.pi :=
by
  sorry

end find_theta_l289_289836


namespace f_odd_f_decreasing_on_01_l289_289927

noncomputable def f (x : ℝ) (hx : x ≠ 0) : ℝ := x + 1 / x

theorem f_odd (x : ℝ) (hx : x ≠ 0) : f (-x) hx = -f x hx :=
by sorry

theorem f_decreasing_on_01 : ∀ x ∈ Ioo (0 : ℝ) 1, deriv (λ x, f x (ne_of_gt (lt_of_lt_of_le zero_lt_one x))) x < 0 :=
by sorry

end f_odd_f_decreasing_on_01_l289_289927


namespace problem1_problem2_l289_289366

noncomputable def calculate_a (b : ℝ) (angleB : ℝ) (angleC : ℝ) : ℝ :=
  let angleA := 180 - (angleB + angleC)
  let sinA := Real.sin (angleA * Real.pi / 180)
  (b * sinA) / (Real.sin(angleB * Real.pi / 180))

theorem problem1 (b : ℝ) (angleB : ℝ) (angleC : ℝ) : 
  b = 2 → angleB = 30 → angleC = 135 → 
  calculate_a b angleB angleC = Real.sqrt 6 - Real.sqrt 2 := 
sorry

noncomputable def calculate_angleC (a b c : ℝ) : ℝ :=
  let S_triangle_ABC := (a^2 + b^2 - c^2) / 4
  let sinC := (a^2 + b^2 - c^2) / (2 * a * b)
  let cosC := (a^2 + b^2 - c^2) / (2 * a * b)
  if sinC = cosC then
    Real.pi / 4
  else
    0 -- this condition catches non-matching (impossible in given context)

theorem problem2 (a b c : ℝ) (S_triangle_ABC : ℝ) : 
  S_triangle_ABC = 1/4 * (a^2 + b^2 - c^2) → 
  calculate_angleC a b c = Real.pi / 4 := 
sorry

end problem1_problem2_l289_289366


namespace mos_to_ory_bus_encounter_l289_289031

def encounter_buses (departure_time : Nat) (encounter_bus_time : Nat) (travel_time : Nat) : Nat := sorry

theorem mos_to_ory_bus_encounter :
  encounter_buses 0 30 5 = 10 :=
sorry

end mos_to_ory_bus_encounter_l289_289031


namespace proof_problem_l289_289157

variable {a b c : ℝ}
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
variable (h4 : (a+1) * (b+1) * (c+1) = 8)

theorem proof_problem :
  a + b + c ≥ 3 ∧ abc ≤ 1 :=
by
  sorry

end proof_problem_l289_289157


namespace fishers_tomorrow_l289_289876

-- Definitions based on conditions
def people_every_day : ℕ := 7
def people_every_other_day : ℕ := 8
def people_every_three_days : ℕ := 3
def people_yesterday : ℕ := 12
def people_today : ℕ := 10

-- Theorem to be proved
theorem fishers_tomorrow (people_every_day people_every_other_day people_every_three_days people_yesterday people_today : ℕ) : 
  people_every_day = 7 ∧ 
  people_every_other_day = 8 ∧ 
  people_every_three_days = 3 ∧
  people_yesterday = 12 ∧ 
  people_today = 10 →
  15 := 
by {
  sorry
}

end fishers_tomorrow_l289_289876


namespace magnitude_of_conjugate_z_l289_289843

theorem magnitude_of_conjugate_z (a b : ℝ) (h₀: b ≠ 0) 
  (h₁ : (2 + a * Complex.I) / (3 - Complex.I) = b * Complex.I) :
  Complex.abs (Complex.conj (a + b * Complex.I)) = 2 * Real.sqrt 10 :=
by
  sorry

end magnitude_of_conjugate_z_l289_289843


namespace percent_decrease_in_square_area_l289_289556

theorem percent_decrease_in_square_area (A B C D : Type) 
  (side_length_AD side_length_AB side_length_CD : ℝ) 
  (area_square_original new_side_length new_area : ℝ) 
  (h1 : side_length_AD = side_length_AB) (h2 : side_length_AD = side_length_CD) 
  (h3 : area_square_original = side_length_AD^2)
  (h4 : new_side_length = side_length_AD * 0.8)
  (h5 : new_area = new_side_length^2)
  (h6 : side_length_AD = 9) : 
  (area_square_original - new_area) / area_square_original * 100 = 36 := 
  by 
    sorry

end percent_decrease_in_square_area_l289_289556


namespace biscuit_dimensions_l289_289943

theorem biscuit_dimensions (sheet_length : ℝ) (sheet_width : ℝ) (num_biscuits : ℕ) 
  (h₁ : sheet_length = 12) (h₂ : sheet_width = 12) (h₃ : num_biscuits = 16) :
  ∃ biscuit_length : ℝ, biscuit_length = 3 :=
by
  sorry

end biscuit_dimensions_l289_289943


namespace ratio_jake_to_clementine_l289_289249

-- Definitions based on conditions
def ClementineCookies : Nat := 72
def ToryCookies (J : Nat) : Nat := (J + ClementineCookies) / 2
def TotalCookies (J : Nat) : Nat := ClementineCookies + J + ToryCookies J
def TotalRevenue : Nat := 648
def CookiePrice : Nat := 2
def TotalCookiesSold : Nat := TotalRevenue / CookiePrice

-- The main proof statement
theorem ratio_jake_to_clementine : 
  ∃ J : Nat, TotalCookies J = TotalCookiesSold ∧ J / ClementineCookies = 2 :=
by
  sorry

end ratio_jake_to_clementine_l289_289249


namespace time_to_fill_one_barrel_with_leak_l289_289494

-- Define the conditions
def normal_time_per_barrel := 3
def time_to_fill_12_barrels_no_leak := normal_time_per_barrel * 12
def additional_time_due_to_leak := 24
def time_to_fill_12_barrels_with_leak (t : ℕ) := 12 * t

-- Define the theorem
theorem time_to_fill_one_barrel_with_leak :
  ∃ t : ℕ, time_to_fill_12_barrels_with_leak t = time_to_fill_12_barrels_no_leak + additional_time_due_to_leak ∧ t = 5 :=
by {
  use 5, 
  sorry
}

end time_to_fill_one_barrel_with_leak_l289_289494


namespace fishing_tomorrow_l289_289872

theorem fishing_tomorrow (every_day_fishers every_other_day_fishers every_three_days_fishers fishers_yesterday fishers_today : ℕ) :
  (every_day_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (every_three_days_fishers + every_day_fishers + (every_other_day_fishers - (fishers_yesterday - every_day_fishers)) = 15) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  simp
  sorry

end fishing_tomorrow_l289_289872


namespace find_pq_l289_289319

noncomputable def find_k_squared (x y : ℝ) : ℝ :=
  let u1 := x^2 + y^2 - 12 * x + 16 * y - 160
  let u2 := x^2 + y^2 + 12 * x + 16 * y - 36
  let k_sq := 741 / 324
  k_sq

theorem find_pq : (741 + 324) = 1065 := by
  sorry

end find_pq_l289_289319


namespace incorrect_guess_20_l289_289467

-- Define the assumptions and conditions
def bears : Nat → String := sorry -- function that determines the color of the bear at position n
axiom bears_color_constraint : ∀ n:Nat, exists b:List String, b.length = 3 ∧ Set ("W" "B" "Bk") = List.toSet b ∧ 
  List.all (List.sublist b (n, n+1, n+2) bears = fun c=> c = "W" or c = "B" or c = "Bk") 

-- Iskander's guesses
def guess1 := (2, "W")
def guess2 := (20, "B")
def guess3 := (400, "Bk")
def guess4 := (600, "B")
def guess5 := (800, "W")

-- Function to check the bear at each position
def check_bear (n:Nat) : String := sorry

-- Iskander's guess correctness, exactly one is wrong
axiom one_wrong : count (check_bear 2 =="W") 
                         + count (check_bear 20 == "B") 
                         + count (check_bear 400 =="Bk") 
                         + count (check_bear 600 =="B") 
                         + count (check_bear 800 =="W") = 4

-- Prove that the guess for the 20th bear is incorrect
theorem incorrect_guess_20 : ∀ {n:Nat} (h : n=20), (check_bear n != "B") := sorry

end incorrect_guess_20_l289_289467


namespace simplify_exponents_l289_289917

theorem simplify_exponents (x : ℝ) : x^5 * x^3 = x^8 :=
by
  sorry

end simplify_exponents_l289_289917


namespace side_length_uncovered_l289_289498

theorem side_length_uncovered (L W : ℝ) (h₁ : L * W = 50) (h₂ : 2 * W + L = 25) : L = 20 :=
by {
  sorry
}

end side_length_uncovered_l289_289498


namespace units_digit_sum_factorials_l289_289990

-- Definitions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def units_digit (n : ℕ) : ℕ := n % 10

-- Theorem to prove
theorem units_digit_sum_factorials : 
  units_digit (∑ i in Finset.range 2011, factorial i) = 3 :=
by
  sorry

end units_digit_sum_factorials_l289_289990


namespace n_squared_plus_d_not_perfect_square_l289_289906

theorem n_squared_plus_d_not_perfect_square (n d : ℕ) (h1 : n > 0)
  (h2 : d > 0) (h3 : d ∣ 2 * n^2) : ¬ ∃ x : ℕ, n^2 + d = x^2 := 
sorry

end n_squared_plus_d_not_perfect_square_l289_289906


namespace total_candies_l289_289654

def candies_in_boxes (num_boxes: Nat) (pieces_per_box: Nat) : Nat :=
  num_boxes * pieces_per_box

theorem total_candies :
  candies_in_boxes 3 6 + candies_in_boxes 5 8 + candies_in_boxes 4 10 = 98 := by
  sorry

end total_candies_l289_289654


namespace vacation_cost_l289_289006

theorem vacation_cost (C P : ℕ) 
    (h1 : C = 5 * P)
    (h2 : C = 7 * (P - 40))
    (h3 : C = 8 * (P - 60)) : C = 700 := 
by 
    sorry

end vacation_cost_l289_289006


namespace find_side_and_area_l289_289301

-- Conditions
variables {A B C a b c : ℝ} (S : ℝ)
axiom angle_sum : A + B + C = Real.pi
axiom side_a : a = 4
axiom side_b : b = 5
axiom angle_relation : C = 2 * A

-- Proven equalities
theorem find_side_and_area :
  c = 6 ∧ S = 5 * 6 * (Real.sqrt 7) / 4 / 2 := by
  sorry

end find_side_and_area_l289_289301


namespace number_of_pairs_l289_289723

theorem number_of_pairs (n : ℕ) (h : n ≥ 3) : 
  ∃ a : ℕ, a = (n-2) * 2^(n-1) + 1 :=
by
  sorry

end number_of_pairs_l289_289723


namespace total_items_at_bakery_l289_289344

theorem total_items_at_bakery (bread_rolls : ℕ) (croissants : ℕ) (bagels : ℕ) (h1 : bread_rolls = 49) (h2 : croissants = 19) (h3 : bagels = 22) : bread_rolls + croissants + bagels = 90 :=
by
  sorry

end total_items_at_bakery_l289_289344


namespace arithmetic_sequence_k_value_l289_289718

theorem arithmetic_sequence_k_value (a1 d : ℤ) (S : ℕ → ℤ)
  (h1 : a1 = 1)
  (h2 : d = 2)
  (h3 : ∀ k : ℕ, S (k+2) - S k = 24) : k = 5 := 
sorry

end arithmetic_sequence_k_value_l289_289718


namespace desired_digit_set_l289_289234

noncomputable def prob_digit (d : ℕ) : ℝ := if d > 0 then Real.log (d + 1) - Real.log d else 0

theorem desired_digit_set : 
  (prob_digit 5 = (1 / 2) * (prob_digit 5 + prob_digit 6 + prob_digit 7 + prob_digit 8)) ↔
  {d | d = 5 ∨ d = 6 ∨ d = 7 ∨ d = 8} = {5, 6, 7, 8} :=
by
  sorry

end desired_digit_set_l289_289234


namespace find_natural_numbers_l289_289813

theorem find_natural_numbers (n : ℕ) (x : ℕ) (y : ℕ) (hx : n = 10 * x + y) (hy : 10 * x + y = 14 * x) : n = 14 ∨ n = 28 :=
by
  sorry

end find_natural_numbers_l289_289813


namespace number_of_dogs_is_112_l289_289362

-- Definitions based on the given conditions.
def ratio_dogs_to_cats_to_bunnies (D C B : ℕ) : Prop := 4 * C = 7 * D ∧ 9 * C = 7 * B
def total_dogs_and_bunnies (D B : ℕ) (total : ℕ) : Prop := D + B = total

-- The hypothesis and conclusion of the problem.
theorem number_of_dogs_is_112 (D C B : ℕ) (x : ℕ) (h1: ratio_dogs_to_cats_to_bunnies D C B) (h2: total_dogs_and_bunnies D B 364) : D = 112 :=
by 
  sorry

end number_of_dogs_is_112_l289_289362


namespace solve_eq_l289_289745

theorem solve_eq (x : ℝ) : (x - 2)^2 = 9 * x^2 ↔ x = -1 ∨ x = 1 / 2 := by
  sorry

end solve_eq_l289_289745


namespace find_b_values_l289_289082

noncomputable def has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + b*x + 25 = 0

theorem find_b_values (b : ℝ) : has_real_root b ↔ b ∈ set.Iic (-10) ∪ set.Ici 10 := by
  sorry

end find_b_values_l289_289082


namespace find_rate_l289_289510

noncomputable def national_bank_interest_rate (total_income: ℚ) (investment_national: ℚ) (investment_additional: ℚ) (additional_rate: ℚ) (total_investment_rate: ℚ): ℚ :=
  (total_income - (investment_additional * additional_rate)) / investment_national

theorem find_rate (total_income: ℚ) (investment_national: ℚ) (investment_additional: ℚ) (additional_rate: ℚ) (total_investment_rate: ℚ) (correct_rate: ℚ):
  investment_national = 2400 → investment_additional = 600 → additional_rate = 0.10 → total_investment_rate = 0.06 → total_income = total_investment_rate * (investment_national + investment_additional) → correct_rate = 0.05 → national_bank_interest_rate total_income investment_national investment_additional additional_rate total_investment_rate = correct_rate :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end find_rate_l289_289510


namespace ben_fewer_pints_than_kathryn_l289_289255

-- Define the conditions
def annie_picked := 8
def kathryn_picked := annie_picked + 2
def total_picked := 25

-- Add noncomputable because constants are involved
noncomputable def ben_picked : ℕ := total_picked - (annie_picked + kathryn_picked)

theorem ben_fewer_pints_than_kathryn : ben_picked = kathryn_picked - 3 := 
by 
  -- The problem statement does not require proof body
  sorry

end ben_fewer_pints_than_kathryn_l289_289255


namespace divisor_count_l289_289903

theorem divisor_count (m : ℕ) (h : m = 2^15 * 5^12) :
  let m_squared := m * m
  let num_divisors_m := (15 + 1) * (12 + 1)
  let num_divisors_m_squared := (30 + 1) * (24 + 1)
  let divisors_of_m_squared_less_than_m := (num_divisors_m_squared - 1) / 2
  num_divisors_m_squared - num_divisors_m = 179 :=
by
  subst h
  sorry

end divisor_count_l289_289903


namespace smallest_a_inequality_l289_289263

theorem smallest_a_inequality 
  (x : ℝ)
  (h1 : x ∈ Set.Ioo (-3 * Real.pi / 2) (-Real.pi)) : 
  (∃ a : ℝ, a = -2.52 ∧ (∀ x ∈ Set.Ioo (-3 * Real.pi / 2) (-Real.pi), 
    ( ((Real.sqrt (Real.cos x / Real.sin x)^2) - (Real.sqrt (Real.sin x / Real.cos x)^2))
    / ((Real.sqrt (Real.sin x)^2) - (Real.sqrt (Real.cos x)^2)) ) < a )) :=
  sorry

end smallest_a_inequality_l289_289263


namespace sqrt_sum_eq_l289_289953

theorem sqrt_sum_eq : sqrt 12 + sqrt (1 / 3) = (7 * sqrt 3) / 3 := sorry

end sqrt_sum_eq_l289_289953


namespace interval_solution_length_l289_289668

theorem interval_solution_length (a b : ℝ) (h : (b - a) / 3 = 8) : b - a = 24 := by
  sorry

end interval_solution_length_l289_289668


namespace prove_equation_l289_289845

theorem prove_equation (x : ℚ) (h : 5 * x - 3 = 15 * x + 21) : 3 * (2 * x + 5) = 3 / 5 :=
by
  sorry

end prove_equation_l289_289845


namespace perimeter_of_square_D_l289_289740

-- Definitions based on conditions
def perimeter_C : ℝ := 32
def side_len_C : ℝ := perimeter_C / 4
def area_C : ℝ := side_len_C * side_len_C
def area_D : ℝ := area_C / 3
def side_len_D : ℝ := Real.sqrt area_D
def perimeter_D : ℝ := 4 * side_len_D

-- Theorem to prove the perimeter of square D
theorem perimeter_of_square_D : perimeter_D = (32 * Real.sqrt 3) / 3 :=
by
  sorry

end perimeter_of_square_D_l289_289740


namespace blossom_room_area_l289_289762

theorem blossom_room_area
  (ft_to_in : ℕ)
  (length_ft : ℕ)
  (width_ft : ℕ)
  (ft_to_in_def : ft_to_in = 12)
  (length_width_def : length_ft = 10)
  (room_square : length_ft = width_ft) :
  (length_ft * ft_to_in) * (width_ft * ft_to_in) = 14400 := 
by
  -- ft_to_in is the conversion factor from feet to inches
  -- length_ft and width_ft are both 10 according to length_width_def and room_square
  -- So, we have (10 * 12) * (10 * 12) = 14400
  sorry

end blossom_room_area_l289_289762


namespace moles_of_CH4_needed_l289_289046

theorem moles_of_CH4_needed
  (moles_C6H6_needed : ℕ)
  (reaction_balance : ∀ (C6H6 CH4 C6H5CH3 H2 : ℕ), 
    C6H6 + CH4 = C6H5CH3 + H2 → C6H6 = 1 ∧ CH4 = 1 ∧ C6H5CH3 = 1 ∧ H2 = 1)
  (H : moles_C6H6_needed = 3) :
  (3 : ℕ) = 3 :=
by 
  -- The actual proof would go here
  sorry

end moles_of_CH4_needed_l289_289046


namespace space_per_bookshelf_l289_289150

-- Defining the conditions
def S_room : ℕ := 400
def S_reserved : ℕ := 160
def n_shelves : ℕ := 3

-- Theorem statement
theorem space_per_bookshelf (S_room S_reserved n_shelves : ℕ)
  (h1 : S_room = 400) (h2 : S_reserved = 160) (h3 : n_shelves = 3) :
  (S_room - S_reserved) / n_shelves = 80 :=
by
  -- Placeholder for the proof
  sorry

end space_per_bookshelf_l289_289150


namespace paula_bought_fewer_cookies_l289_289446
-- Import the necessary libraries

-- Definitions
def paul_cookies : ℕ := 45
def total_cookies : ℕ := 87

-- Theorem statement
theorem paula_bought_fewer_cookies : ∃ (paula_cookies : ℕ), paul_cookies + paula_cookies = total_cookies ∧ paul_cookies - paula_cookies = 3 := by
  sorry

end paula_bought_fewer_cookies_l289_289446


namespace percentage_of_green_ducks_l289_289890

theorem percentage_of_green_ducks (ducks_small_pond ducks_large_pond : ℕ) 
  (green_fraction_small_pond green_fraction_large_pond : ℚ) 
  (h1 : ducks_small_pond = 20) 
  (h2 : ducks_large_pond = 80) 
  (h3 : green_fraction_small_pond = 0.20) 
  (h4 : green_fraction_large_pond = 0.15) :
  let total_ducks := ducks_small_pond + ducks_large_pond
  let green_ducks := (green_fraction_small_pond * ducks_small_pond) + 
                     (green_fraction_large_pond * ducks_large_pond)
  (green_ducks / total_ducks) * 100 = 16 := 
by 
  sorry

end percentage_of_green_ducks_l289_289890


namespace Jenny_ate_65_l289_289148

theorem Jenny_ate_65 (mike_squares : ℕ) (jenny_squares : ℕ)
  (h1 : mike_squares = 20)
  (h2 : jenny_squares = 3 * mike_squares + 5) :
  jenny_squares = 65 :=
by
  sorry

end Jenny_ate_65_l289_289148


namespace blossom_room_area_l289_289761

theorem blossom_room_area
  (ft_to_in : ℕ)
  (length_ft : ℕ)
  (width_ft : ℕ)
  (ft_to_in_def : ft_to_in = 12)
  (length_width_def : length_ft = 10)
  (room_square : length_ft = width_ft) :
  (length_ft * ft_to_in) * (width_ft * ft_to_in) = 14400 := 
by
  -- ft_to_in is the conversion factor from feet to inches
  -- length_ft and width_ft are both 10 according to length_width_def and room_square
  -- So, we have (10 * 12) * (10 * 12) = 14400
  sorry

end blossom_room_area_l289_289761


namespace reduced_price_per_kg_l289_289377

/-- Given that:
1. There is a reduction of 25% in the price of oil.
2. The housewife can buy 5 kgs more for Rs. 700 after the reduction.

Prove that the reduced price per kg of oil is Rs. 35. -/
theorem reduced_price_per_kg (P : ℝ) (R : ℝ) (X : ℝ)
  (h1 : R = 0.75 * P)
  (h2 : 700 = X * P)
  (h3 : 700 = (X + 5) * R)
  : R = 35 := 
sorry

end reduced_price_per_kg_l289_289377


namespace complement_A_is_closed_interval_l289_289070

-- Define the universal set U as the set of all real numbers
def U : Set ℝ := Set.univ

-- Define the set A with the given condition
def A : Set ℝ := { x | x^2 - 2 * x - 3 > 0 }

-- Define the complement of A with respect to U
def complement_A : Set ℝ := Set.compl A

theorem complement_A_is_closed_interval :
  complement_A = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
by
  sorry  -- Proof to be inserted

end complement_A_is_closed_interval_l289_289070


namespace color_dots_l289_289978

-- Define the vertices and the edges of the graph representing the figure
inductive Color : Type
| red : Color
| white : Color
| blue : Color

structure Dot :=
  (color : Color)

structure Edge :=
  (u : Dot)
  (v : Dot)

def valid_coloring (dots : List Dot) (edges : List Edge) : Prop :=
  ∀ e ∈ edges, e.u.color ≠ e.v.color

def count_colorings : Nat :=
  6 * 2

theorem color_dots (dots : List Dot) (edges : List Edge)
  (h1 : ∀ d ∈ dots, d.color = Color.red ∨ d.color = Color.white ∨ d.color = Color.blue)
  (h2 : valid_coloring dots edges) :
  count_colorings = 12 :=
by
  sorry

end color_dots_l289_289978


namespace exists_polynomial_degree_n_l289_289441

theorem exists_polynomial_degree_n (n : ℕ) (hn : 0 < n) : 
  ∃ (ω ψ : Polynomial ℤ), ω.degree = n ∧ (ω^2 = (X^2 - 1) * ψ^2 + 1) := 
sorry

end exists_polynomial_degree_n_l289_289441


namespace line_passes_through_2nd_and_4th_quadrants_l289_289238

theorem line_passes_through_2nd_and_4th_quadrants (b : ℝ) :
  (∀ x : ℝ, x > 0 → -2 * x + b < 0) ∧ (∀ x : ℝ, x < 0 → -2 * x + b > 0) :=
by
  sorry

end line_passes_through_2nd_and_4th_quadrants_l289_289238


namespace measure_angle_ABC_l289_289603

theorem measure_angle_ABC (x : ℝ) (h1 : ∃ θ, θ = 180 - x ∧ x / 2 = (180 - x) / 3) : x = 72 :=
by
  sorry

end measure_angle_ABC_l289_289603


namespace combination_5_3_eq_10_l289_289663

-- Define the combination function according to its formula
noncomputable def combination (n k : ℕ) : ℕ :=
  (n.factorial) / (k.factorial * (n - k).factorial)

-- Theorem stating the required result
theorem combination_5_3_eq_10 : combination 5 3 = 10 := by
  sorry

end combination_5_3_eq_10_l289_289663


namespace nine_x_five_y_multiple_l289_289626

theorem nine_x_five_y_multiple (x y : ℤ) (h : 2 * x + 3 * y ≡ 0 [ZMOD 17]) : 
  9 * x + 5 * y ≡ 0 [ZMOD 17] := 
by
  sorry

end nine_x_five_y_multiple_l289_289626


namespace total_pay_is_186_l289_289233

-- Define the conditions
def regular_rate : ℕ := 3 -- dollars per hour
def regular_hours : ℕ := 40 -- hours
def overtime_rate_multiplier : ℕ := 2
def overtime_hours : ℕ := 11

-- Calculate the regular pay
def regular_pay : ℕ := regular_hours * regular_rate

-- Calculate the overtime pay
def overtime_rate : ℕ := regular_rate * overtime_rate_multiplier
def overtime_pay : ℕ := overtime_hours * overtime_rate

-- Calculate the total pay
def total_pay : ℕ := regular_pay + overtime_pay

-- The statement to be proved
theorem total_pay_is_186 : total_pay = 186 :=
by 
  sorry

end total_pay_is_186_l289_289233


namespace whole_process_time_is_6_hours_l289_289577

def folding_time_per_fold : ℕ := 5
def number_of_folds : ℕ := 4
def resting_time_per_rest : ℕ := 75
def number_of_rests : ℕ := 4
def mixing_time : ℕ := 10
def baking_time : ℕ := 30

def total_time_process_in_minutes : ℕ :=
  mixing_time + 
  (folding_time_per_fold * number_of_folds) + 
  (resting_time_per_rest * number_of_rests) + 
  baking_time

def total_time_process_in_hours : ℕ := total_time_process_in_minutes / 60

theorem whole_process_time_is_6_hours :
  total_time_process_in_hours = 6 :=
by sorry

end whole_process_time_is_6_hours_l289_289577


namespace regular_seven_gon_l289_289856

theorem regular_seven_gon 
    (A : Fin 7 → ℝ × ℝ)
    (cong_diagonals_1 : ∀ (i : Fin 7), dist (A i) (A ((i + 2) % 7)) = dist (A 0) (A 2))
    (cong_diagonals_2 : ∀ (i : Fin 7), dist (A i) (A ((i + 3) % 7)) = dist (A 0) (A 3))
    : ∀ (i j : Fin 7), dist (A i) (A ((i + 1) % 7)) = dist (A j) (A ((j + 1) % 7)) :=
by sorry

end regular_seven_gon_l289_289856


namespace positive_solution_x_l289_289840

theorem positive_solution_x (x y z : ℝ) (h1 : x * y = 10 - 3 * x - 2 * y) 
(h2 : y * z = 10 - 5 * y - 3 * z) 
(h3 : x * z = 40 - 5 * x - 2 * z) 
(h_pos : x > 0) : 
  x = 8 :=
sorry

end positive_solution_x_l289_289840


namespace boxes_in_case_correct_l289_289418

-- Given conditions
def total_boxes : Nat := 2
def blocks_per_box : Nat := 6
def total_blocks : Nat := 12

-- Define the number of boxes in a case as a result of total_blocks divided by blocks_per_box
def boxes_in_case : Nat := total_blocks / blocks_per_box

-- Prove the number of boxes in a case is 2
theorem boxes_in_case_correct : boxes_in_case = 2 := by
  -- Place the actual proof here
  sorry

end boxes_in_case_correct_l289_289418


namespace final_hair_length_is_14_l289_289571

def initial_hair_length : ℕ := 24

def half_hair_cut (l : ℕ) : ℕ := l / 2

def hair_growth (l : ℕ) : ℕ := l + 4

def final_hair_cut (l : ℕ) : ℕ := l - 2

theorem final_hair_length_is_14 :
  final_hair_cut (hair_growth (half_hair_cut initial_hair_length)) = 14 := by
  sorry

end final_hair_length_is_14_l289_289571


namespace exists_super_number_B_l289_289219

-- Define a function is_super_number to identify super numbers.
def is_super_number (A : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 ≤ A n ∧ A n < 10

-- Define a function zero_super_number to represent the super number with all digits zero.
def zero_super_number (n : ℕ) := 0

-- Task: Prove the existence of B such that A + B = zero_super_number.
theorem exists_super_number_B (A : ℕ → ℕ) (hA : is_super_number A) :
  ∃ B : ℕ → ℕ, is_super_number B ∧ (∀ n : ℕ, (A n + B n) % 10 = zero_super_number n) :=
sorry

end exists_super_number_B_l289_289219


namespace monotone_intervals_max_floor_a_l289_289283

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x + a

theorem monotone_intervals (a : ℝ) (h : a = 1) :
  (∀ x, 0 < x ∧ x < 1 → deriv (λ x => f x 1) x > 0) ∧
  (∀ x, 1 ≤ x → deriv (λ x => f x 1) x < 0) :=
by
  sorry

theorem max_floor_a (a : ℝ) (h : ∀ x > 0, f x a ≤ x) : ⌊a⌋ = 1 :=
by
  sorry

end monotone_intervals_max_floor_a_l289_289283


namespace evaluate_at_two_l289_289901

def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 10) / (x^2 - x + 5)
def g (x : ℚ) : ℚ := x - 2

theorem evaluate_at_two : f (g 2) + g (f 2) = 38 / 7 := by
  sorry

end evaluate_at_two_l289_289901


namespace quadratic_has_real_root_l289_289126

theorem quadratic_has_real_root (b : ℝ) : 
  (b^2 - 100 ≥ 0) ↔ (b ≤ -10 ∨ b ≥ 10) :=
by
  sorry

end quadratic_has_real_root_l289_289126


namespace find_ab_l289_289701

theorem find_ab (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 48) : a * b = 6 :=
by
  sorry

end find_ab_l289_289701


namespace triangle_right_isosceles_l289_289308

theorem triangle_right_isosceles {A B C : Type*} [LinearOrderedField A] (a b c ha hb : A)
  (h_a_ge_a : ha ≥ b) (h_b_ge_b : hb ≥ c) :
  ∃ (angles : Finset (A)), angles = {90, 45, 45} :=
by sorry

end triangle_right_isosceles_l289_289308


namespace total_pages_l289_289262

theorem total_pages (x : ℕ) (h : 9 + 180 + 3 * (x - 99) = 1392) : x = 500 :=
by
  sorry

end total_pages_l289_289262


namespace number_of_outfits_l289_289208

-- Define the counts of each item
def redShirts : Nat := 6
def greenShirts : Nat := 4
def pants : Nat := 7
def greenHats : Nat := 10
def redHats : Nat := 9

-- Total number of outfits satisfying the conditions
theorem number_of_outfits :
  (redShirts * greenHats * pants) + (greenShirts * redHats * pants) = 672 :=
by
  sorry

end number_of_outfits_l289_289208


namespace wolf_does_not_catch_hare_l289_289380

-- Define the distance the hare needs to cover
def distanceHare := 250 -- meters

-- Define the initial separation between the wolf and the hare
def separation := 30 -- meters

-- Define the speed of the hare
def speedHare := 550 -- meters per minute

-- Define the speed of the wolf
def speedWolf := 600 -- meters per minute

-- Define the time it takes for the hare to reach the refuge
def tHare := (distanceHare : ℚ) / speedHare

-- Define the total distance the wolf needs to cover
def totalDistanceWolf := distanceHare + separation

-- Define the time it takes for the wolf to cover the total distance
def tWolf := (totalDistanceWolf : ℚ) / speedWolf

-- Final proposition to be proven
theorem wolf_does_not_catch_hare : tHare < tWolf :=
by
  sorry

end wolf_does_not_catch_hare_l289_289380


namespace ice_cream_eaten_l289_289023

variables (f : ℝ)

theorem ice_cream_eaten (h : f + 0.25 = 3.5) : f = 3.25 :=
sorry

end ice_cream_eaten_l289_289023


namespace geometric_sequence_product_l289_289835

/-- Given a geometric sequence with positive terms where a_3 = 3 and a_6 = 1/9,
    prove that a_4 * a_5 = 1/3. -/
theorem geometric_sequence_product (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, a n > 0)
    (h_geometric : ∀ n, a (n + 1) = a n * q) (ha3 : a 3 = 3) (ha6 : a 6 = 1 / 9) :
  a 4 * a 5 = 1 / 3 := 
by
  sorry

end geometric_sequence_product_l289_289835


namespace fishing_tomorrow_l289_289885

-- Conditions
def every_day_fishers : Nat := 7
def every_other_day_fishers : Nat := 8
def every_three_days_fishers : Nat := 3
def yesterday_fishers : Nat := 12
def today_fishers : Nat := 10

-- Determine the number who will fish tomorrow
def fishers_tomorrow : Nat :=
  let every_day_tomorrow := every_day_fishers
  let every_three_day_tomorrow := every_three_days_fishers
  let every_other_day_yesterday := yesterday_fishers - every_day_fishers
  let every_other_day_tomorrow := every_other_day_fishers - every_other_day_yesterday
  every_day_tomorrow + every_three_day_tomorrow + every_other_day_tomorrow

theorem fishing_tomorrow : fishers_tomorrow = 15 :=
  by
    -- skipping the actual proof with sorry
    sorry

end fishing_tomorrow_l289_289885


namespace fraction_of_boys_participated_l289_289629

-- Definitions based on given conditions
def total_students (B G : ℕ) : Prop := B + G = 800
def participating_girls (G : ℕ) : Prop := (3 / 4 : ℚ) * G = 150
def total_participants (P : ℕ) : Prop := P = 550
def participating_girls_count (PG : ℕ) : Prop := PG = 150

-- Definition of the fraction of participating boys
def fraction_participating_boys (X : ℚ) (B : ℕ) (PB : ℕ) : Prop := X * B = PB

-- The problem of proving the fraction of boys who participated
theorem fraction_of_boys_participated (B G PB : ℕ) (X : ℚ)
  (h1 : total_students B G)
  (h2 : participating_girls G)
  (h3 : total_participants 550)
  (h4 : participating_girls_count 150)
  (h5 : PB = 550 - 150) :
  fraction_participating_boys X B PB → X = 2 / 3 := by
  sorry

end fraction_of_boys_participated_l289_289629


namespace problem_statement_l289_289904

-- Definitions of the conditions
variables (x y z w : ℕ)

-- The proof problem
theorem problem_statement
  (hx : x^3 = y^2)
  (hz : z^4 = w^3)
  (hzx : z - x = 17)
  (hx_pos : x > 0)
  (hy_pos : y > 0)
  (hz_pos : z > 0)
  (hw_pos : w > 0) :
  w - y = 229 :=
sorry

end problem_statement_l289_289904


namespace circle_area_of_equilateral_triangle_l289_289640

-- Define the side length of the equilateral triangle.
def side_length : ℝ := 12

-- Define the circumradius of an equilateral triangle.
def circumradius (s : ℝ) : ℝ := s / Real.sqrt 3

-- Define the area of a circle given its radius.
def circle_area (R : ℝ) : ℝ := Real.pi * R^2

-- Define the problem statement.
theorem circle_area_of_equilateral_triangle :
  circle_area (circumradius side_length) = 48 * Real.pi :=
by
  sorry

end circle_area_of_equilateral_triangle_l289_289640


namespace simplify_expression_l289_289974

variable (x y : ℝ)

theorem simplify_expression :
  (2 * x + 3 * y) ^ 2 - 2 * x * (2 * x - 3 * y) = 18 * x * y + 9 * y ^ 2 :=
by
  sorry

end simplify_expression_l289_289974


namespace percentage_milk_in_B_l289_289024

theorem percentage_milk_in_B :
  ∀ (A B C : ℕ),
  A = 1200 → B + C = A → B + 150 = C - 150 →
  (B:ℝ) / (A:ℝ) * 100 = 37.5 :=
by
  intros A B C hA hBC hE
  sorry

end percentage_milk_in_B_l289_289024


namespace part_a_part_b_part_c_l289_289976

def op (a b : ℕ) : ℕ := a ^ b + b ^ a

theorem part_a (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : op a b = op b a :=
by
  dsimp [op]
  rw [add_comm]

theorem part_b (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ¬ (op a (op b c) = op (op a b) c) :=
by
  -- example counter: a = 2, b = 2, c = 2 
  -- 2 ^ (2^2 + 2^2) + (2^2 + 2^2) ^ 2 ≠ (2^2 + 2 ^ 2) ^ 2 + 8 ^ 2
  sorry

theorem part_c (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ¬ (op (op a b) (op b c) = op (op b a) (op c b)) :=
by
  -- example counter: a = 2, b = 3, c = 2 
  -- This will involve specific calculations showing the inequality.
  sorry

end part_a_part_b_part_c_l289_289976


namespace point_not_on_graph_l289_289053

theorem point_not_on_graph : 
  ∀ (k : ℝ), (k ≠ 0) → (∀ x y : ℝ, y = k * x → (x, y) = (1, 2)) → ¬ (∀ x y : ℝ, y = k * x → (x, y) = (1, -2)) :=
by
  sorry

end point_not_on_graph_l289_289053


namespace perpendicular_lines_slope_l289_289543

theorem perpendicular_lines_slope (a : ℝ)
  (h : (a * (a + 2)) = -1) : a = -1 :=
sorry

end perpendicular_lines_slope_l289_289543


namespace ed_money_left_l289_289400

theorem ed_money_left
  (cost_per_hour_night : ℝ := 1.5)
  (cost_per_hour_morning : ℝ := 2)
  (initial_money : ℝ := 80)
  (hours_night : ℝ := 6)
  (hours_morning : ℝ := 4) :
  initial_money - (cost_per_hour_night * hours_night + cost_per_hour_morning * hours_morning) = 63 := 
  by
  sorry

end ed_money_left_l289_289400


namespace exists_integers_a_b_c_d_l289_289742

-- Define the problem statement in Lean 4

theorem exists_integers_a_b_c_d (x y z : ℕ) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), x = a^2 + b^2 ∧ y = c^2 + d^2 ∧ z = a * c + b * d :=
by
  sorry

end exists_integers_a_b_c_d_l289_289742


namespace police_coverage_l289_289438

-- Define the intersections and streets
inductive Intersection : Type
| A | B | C | D | E | F | G | H | I | J | K

open Intersection

-- Define the streets
def Streets : List (List Intersection) :=
  [ [A, B, C, D],    -- Horizontal street 1
    [E, F, G],       -- Horizontal street 2
    [H, I, J, K],    -- Horizontal street 3
    [A, E, H],       -- Vertical street 1
    [B, F, I],       -- Vertical street 2
    [D, G, J],       -- Vertical street 3
    [H, F, C],       -- Diagonal street 1
    [C, G, K]        -- Diagonal street 2
  ]

-- Define the set of intersections where police officers are 
def policeIntersections : List Intersection := [B, G, H]

-- State the theorem to be proved
theorem police_coverage : 
  ∀ (street : List Intersection), street ∈ Streets → 
  ∃ (i : Intersection), i ∈ policeIntersections ∧ i ∈ street := 
sorry

end police_coverage_l289_289438


namespace circumscribed_circle_area_l289_289638

open Real

/-- Define the side length of the equilateral triangle --/
def side_length : ℝ := 12

/-- Define the circumradius of an equilateral triangle given side length s --/
def circumradius (s : ℝ) : ℝ := s / sqrt 3

/-- Define the area of a circle given its radius --/
def circle_area (r : ℝ) : ℝ := π * r^2

/-- Theorem: The area of the circle circumscribed about an equilateral triangle 
    with side length 12 units is 48π square units. --/
theorem circumscribed_circle_area :
  circle_area (circumradius side_length) = 48 * π :=
by
  /- Proof will be inserted here -/
  sorry

end circumscribed_circle_area_l289_289638


namespace double_chess_first_player_can_draw_l289_289612

-- Define the basic structure and rules of double chess
structure Game :=
  (state : Type)
  (move : state → state)
  (turn : ℕ → state → state)

-- Define the concept of double move
def double_move (g : Game) (s : g.state) : g.state :=
  g.move (g.move s)

-- Define a condition stating that the first player can at least force a draw
theorem double_chess_first_player_can_draw
  (game : Game)
  (initial_state : game.state)
  (double_move_valid : ∀ s : game.state, ∃ s' : game.state, s' = double_move game s) :
  ∃ draw : game.state, ∀ second_player_strategy : game.state → game.state, 
    double_move game initial_state = draw :=
  sorry

end double_chess_first_player_can_draw_l289_289612


namespace croissant_process_time_in_hours_l289_289575

-- Conditions as definitions
def num_folds : ℕ := 4
def fold_time : ℕ := 5
def rest_time : ℕ := 75
def mix_time : ℕ := 10
def bake_time : ℕ := 30

-- The main theorem statement
theorem croissant_process_time_in_hours :
  (num_folds * (fold_time + rest_time) + mix_time + bake_time) / 60 = 6 := 
sorry

end croissant_process_time_in_hours_l289_289575


namespace unique_solution_nat_triplet_l289_289044

theorem unique_solution_nat_triplet (x y l : ℕ) (h : x^3 + y^3 - 53 = 7^l) : (x, y, l) = (3, 3, 0) :=
sorry

end unique_solution_nat_triplet_l289_289044


namespace ratio_of_radii_l289_289144

variables (a b : ℝ) (h : π * b ^ 2 - π * a ^ 2 = 4 * π * a ^ 2)

theorem ratio_of_radii (h : π * b ^ 2 - π * a ^ 2 = 4 * π * a ^ 2) : 
  a / b = Real.sqrt 5 / 5 :=
sorry

end ratio_of_radii_l289_289144


namespace find_y_when_x4_l289_289365

theorem find_y_when_x4 : 
  (∀ x y : ℚ, 5 * y + 3 = 344 / (x ^ 3)) ∧ (5 * (8:ℚ) + 3 = 344 / (2 ^ 3)) → 
  (∃ y : ℚ, 5 * y + 3 = 344 / (4 ^ 3) ∧ y = 19 / 40) := 
by
  sorry

end find_y_when_x4_l289_289365


namespace binary_to_octal_equivalence_l289_289617

theorem binary_to_octal_equivalence : (1 * 2^6 + 0 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) 
                                    = (1 * 8^2 + 1 * 8^1 + 5 * 8^0) :=
by sorry

end binary_to_octal_equivalence_l289_289617


namespace quadratic_max_value_4_at_2_l289_289928

theorem quadratic_max_value_4_at_2 (a b c : ℝ) (m : ℝ)
  (h1 : ∀ x : ℝ, x ≠ 2 → (a * 2^2 + b * 2 + c) = 4)
  (h2 : a * 0^2 + b * 0 + c = -20)
  (h3 : a * 5^2 + b * 5 + c = m) :
  m = -50 :=
sorry

end quadratic_max_value_4_at_2_l289_289928


namespace range_of_fx₂_l289_289691

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2 * x + a * Real.log x

def is_extreme_point (a x : ℝ) : Prop := 
  (2 * x^2 - 2 * x + a) / x = 0

theorem range_of_fx₂ (a x₁ x₂ : ℝ) (h₀ : 0 < a) (h₁ : a < 1 / 2) 
  (h₂ : 0 < x₁ ∧ x₁ < x₂) (h₃ : is_extreme_point a x₁)
  (h₄ : is_extreme_point a x₂) : 
  (f a x₂) ∈ (Set.Ioo (-(3 + 2 * Real.log 2) / 4) (-1)) :=
sorry

end range_of_fx₂_l289_289691


namespace isosceles_triangle_l289_289328

theorem isosceles_triangle 
  {a b : ℝ} {α β : ℝ} 
  (h : a / (Real.cos α) = b / (Real.cos β)) : 
  a = b :=
sorry

end isosceles_triangle_l289_289328


namespace find_n_mod_10_l289_289523

theorem find_n_mod_10 :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n % 10 = (-2023) % 10 ∧ n = 7 :=
sorry

end find_n_mod_10_l289_289523


namespace average_temperature_week_l289_289971

theorem average_temperature_week :
  let sunday := 99.1
  let monday := 98.2
  let tuesday := 98.7
  let wednesday := 99.3
  let thursday := 99.8
  let friday := 99.0
  let saturday := 98.9
  (sunday + monday + tuesday + wednesday + thursday + friday + saturday) / 7 = 99.0 :=
by
  sorry

end average_temperature_week_l289_289971


namespace a_can_complete_in_6_days_l289_289622

noncomputable def rate_b : ℚ := 1/8
noncomputable def rate_c : ℚ := 1/12
noncomputable def earnings_total : ℚ := 2340
noncomputable def earnings_b : ℚ := 780.0000000000001

theorem a_can_complete_in_6_days :
  ∃ (rate_a : ℚ), 
    (1 / rate_a) = 6 ∧
    rate_a + rate_b + rate_c = 3 * rate_b ∧
    earnings_b = (rate_b / (rate_a + rate_b + rate_c)) * earnings_total := sorry

end a_can_complete_in_6_days_l289_289622


namespace quadratic_has_real_root_iff_b_in_interval_l289_289099

theorem quadratic_has_real_root_iff_b_in_interval (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ Set.Iic (-10) ∪ Set.Ici 10) := 
sorry

end quadratic_has_real_root_iff_b_in_interval_l289_289099


namespace sum_of_intercepts_l289_289232

theorem sum_of_intercepts (x y : ℝ) (h : y + 3 = -2 * (x + 5)) : 
  (- (13 / 2) : ℝ) + (- 13 : ℝ) = - (39 / 2) :=
by sorry

end sum_of_intercepts_l289_289232


namespace quadratic_has_real_root_l289_289128

theorem quadratic_has_real_root (b : ℝ) : 
  (b^2 - 100 ≥ 0) ↔ (b ≤ -10 ∨ b ≥ 10) :=
by
  sorry

end quadratic_has_real_root_l289_289128


namespace tank_capacity_l289_289503

theorem tank_capacity (x : ℝ) 
  (h1 : 1/4 * x + 180 = 2/3 * x) : 
  x = 432 :=
by
  sorry

end tank_capacity_l289_289503


namespace cube_volume_l289_289479

theorem cube_volume (SA : ℕ) (h : SA = 294) : 
  ∃ V : ℕ, V = 343 := 
by
  sorry

end cube_volume_l289_289479


namespace add_ten_to_certain_number_l289_289616

theorem add_ten_to_certain_number (x : ℤ) (h : x + 36 = 71) : x + 10 = 45 :=
by
  sorry

end add_ten_to_certain_number_l289_289616


namespace points_on_opposite_sides_of_line_l289_289684

theorem points_on_opposite_sides_of_line (a : ℝ) :
  let A := (3, 1)
  let B := (-4, 6)
  (3 * A.1 - 2 * A.2 + a) * (3 * B.1 - 2 * B.2 + a) < 0 ↔ -7 < a ∧ a < 24 :=
by
  let A := (3, 1)
  let B := (-4, 6)
  have hA : 3 * A.1 - 2 * A.2 + a = 7 + a := by sorry
  have hB : 3 * B.1 - 2 * B.2 + a = -24 + a := by sorry
  exact sorry

end points_on_opposite_sides_of_line_l289_289684


namespace average_time_for_relay_race_l289_289216

noncomputable def average_leg_time (y_time z_time w_time x_time : ℕ) : ℚ :=
  (y_time + z_time + w_time + x_time) / 4

theorem average_time_for_relay_race :
  let y_time := 58
  let z_time := 26
  let w_time := 2 * z_time
  let x_time := 35
  average_leg_time y_time z_time w_time x_time = 42.75 := by
    sorry

end average_time_for_relay_race_l289_289216


namespace sum_of_cubes_divisible_by_9_l289_289587

theorem sum_of_cubes_divisible_by_9 (n : ℕ) : 9 ∣ (n^3 + (n + 1)^3 + (n + 2)^3) := 
  sorry

end sum_of_cubes_divisible_by_9_l289_289587


namespace no_snow_three_days_l289_289751

noncomputable def probability_no_snow_first_two_days : ℚ := 1 - 2/3
noncomputable def probability_no_snow_third_day : ℚ := 1 - 3/5

theorem no_snow_three_days : 
  (probability_no_snow_first_two_days) * 
  (probability_no_snow_first_two_days) * 
  (probability_no_snow_third_day) = 2/45 :=
by
  sorry

end no_snow_three_days_l289_289751


namespace second_month_sale_l289_289373

theorem second_month_sale 
  (sale_1st: ℕ) (sale_3rd: ℕ) (sale_4th: ℕ) (sale_5th: ℕ) (sale_6th: ℕ) (avg_sale: ℕ)
  (h1: sale_1st = 5266) (h3: sale_3rd = 5864)
  (h4: sale_4th = 6122) (h5: sale_5th = 6588)
  (h6: sale_6th = 4916) (h_avg: avg_sale = 5750) :
  ∃ sale_2nd, (sale_1st + sale_2nd + sale_3rd + sale_4th + sale_5th + sale_6th) / 6 = avg_sale :=
by
  sorry

end second_month_sale_l289_289373


namespace avg_of_6_10_N_is_10_if_even_l289_289566

theorem avg_of_6_10_N_is_10_if_even (N : ℕ) (h1 : 9 ≤ N) (h2 : N ≤ 17) (h3 : (6 + 10 + N) % 2 = 0) : (6 + 10 + N) / 3 = 10 :=
by
-- sorry is placed here since we are not including the actual proof
sorry

end avg_of_6_10_N_is_10_if_even_l289_289566


namespace quadratic_real_roots_l289_289130

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := by 
  sorry

end quadratic_real_roots_l289_289130


namespace garden_perimeter_l289_289647

-- formally defining the conditions of the problem
variables (x y : ℝ)
def diagonal_of_garden : Prop := x^2 + y^2 = 900
def area_of_garden : Prop := x * y = 216

-- final statement to prove the perimeter of the garden
theorem garden_perimeter (h1 : diagonal_of_garden x y) (h2 : area_of_garden x y) : 2 * (x + y) = 73 := sorry

end garden_perimeter_l289_289647


namespace rewrite_equation_l289_289329

theorem rewrite_equation (x y : ℝ) (h : 2 * x - y = 4) : y = 2 * x - 4 :=
by
  sorry

end rewrite_equation_l289_289329


namespace certain_number_is_51_l289_289508

theorem certain_number_is_51 (G C : ℤ) 
  (h1 : G = 33) 
  (h2 : 3 * G = 2 * C - 3) : 
  C = 51 := 
by
  sorry

end certain_number_is_51_l289_289508


namespace simplest_radical_form_l289_289242

def is_simplest_radical_form (r : ℝ) : Prop :=
  ∀ x : ℝ, x * x = r → ∃ y : ℝ, y * y ≠ r

theorem simplest_radical_form :
   (is_simplest_radical_form 6) :=
by
  sorry

end simplest_radical_form_l289_289242


namespace ant_probability_after_10_minutes_l289_289026

-- Definitions based on the conditions given in the problem
def ant_start_at_A := true
def moves_each_minute (n : ℕ) := n == 10
def blue_dots (x y : ℤ) : Prop := 
  (x == 0 ∨ y == 0) ∧ (x + y) % 2 == 0
def A_at_center (x y : ℤ) : Prop := x == 0 ∧ y == 0
def B_north_of_A (x y : ℤ) : Prop := x == 0 ∧ y == 1

-- The probability we need to prove
def probability_ant_at_B_after_10_minutes := 1 / 9

-- We state our proof problem
theorem ant_probability_after_10_minutes :
  ant_start_at_A ∧ moves_each_minute 10 ∧ blue_dots 0 0 ∧ blue_dots 0 1 ∧ A_at_center 0 0 ∧ B_north_of_A 0 1
  → probability_ant_at_B_after_10_minutes = 1 / 9 := 
sorry

end ant_probability_after_10_minutes_l289_289026


namespace stock_reaches_N_fourth_time_l289_289340

noncomputable def stock_at_k (c0 a b : ℝ) (k : ℕ) : ℝ :=
  if k % 2 = 0 then c0 + (k / 2) * (a - b)
  else c0 + (k / 2 + 1) * a - (k / 2) * b

theorem stock_reaches_N_fourth_time (c0 a b N : ℝ) (hN3 : ∃ k1 k2 k3 : ℕ, k1 ≠ k2 ∧ k2 ≠ k3 ∧ k1 ≠ k3 ∧ stock_at_k c0 a b k1 = N ∧ stock_at_k c0 a b k2 = N ∧ stock_at_k c0 a b k3 = N) :
  ∃ k4 : ℕ, k4 ≠ k1 ∧ k4 ≠ k2 ∧ k4 ≠ k3 ∧ stock_at_k c0 a b k4 = N := 
sorry

end stock_reaches_N_fourth_time_l289_289340


namespace total_sticks_needed_l289_289731

theorem total_sticks_needed :
  let simon_sticks := 36
  let gerry_sticks := 2 * (simon_sticks / 3)
  let total_simon_and_gerry := simon_sticks + gerry_sticks
  let micky_sticks := total_simon_and_gerry + 9
  total_simon_and_gerry + micky_sticks = 129 :=
by
  sorry

end total_sticks_needed_l289_289731


namespace exponents_to_99_l289_289448

theorem exponents_to_99 :
  (1 * 3 / 3^2 / 3^4 / 3^8 * 3^16 * 3^32 * 3^64 = 3^99) :=
sorry

end exponents_to_99_l289_289448


namespace shifts_needed_l289_289016

-- Given definitions
def total_workers : ℕ := 12
def workers_per_shift : ℕ := 2
def total_ways_to_assign : ℕ := 23760

-- Prove the number of shifts needed
theorem shifts_needed : total_workers / workers_per_shift = 6 := by
  sorry

end shifts_needed_l289_289016


namespace boxes_in_case_correct_l289_289417

-- Given conditions
def total_boxes : Nat := 2
def blocks_per_box : Nat := 6
def total_blocks : Nat := 12

-- Define the number of boxes in a case as a result of total_blocks divided by blocks_per_box
def boxes_in_case : Nat := total_blocks / blocks_per_box

-- Prove the number of boxes in a case is 2
theorem boxes_in_case_correct : boxes_in_case = 2 := by
  -- Place the actual proof here
  sorry

end boxes_in_case_correct_l289_289417


namespace velocity_at_t_2_l289_289692

variable {t : ℝ}
noncomputable def motion_eq (t : ℝ) : ℝ := t^2 + 3 / t

theorem velocity_at_t_2 : deriv motion_eq 2 = 13 / 4 :=
by {
  sorry
}

end velocity_at_t_2_l289_289692


namespace fourier_series_decomposition_l289_289513

open Real

noncomputable def f : ℝ → ℝ :=
  λ x => if (x < 0) then -1 else (if (0 < x) then 1/2 else 0)

theorem fourier_series_decomposition :
    ∀ x, -π ≤ x ∧ x ≤ π →
         f x = -1/4 + (3/π) * ∑' k, (sin ((2*k+1)*x)) / (2*k+1) :=
by
  sorry

end fourier_series_decomposition_l289_289513


namespace tangency_condition_l289_289392

-- Definitions based on conditions
def ellipse (x y : ℝ) : Prop := 2 * x^2 + 3 * y^2 = 6
def hyperbola (x y n : ℝ) : Prop := 3 * x^2 - n * (y - 1)^2 = 3

-- The theorem statement based on the question and correct answer:
theorem tangency_condition (n : ℝ) (x y : ℝ) : 
  ellipse x y → hyperbola x y n → n = -6 :=
sorry

end tangency_condition_l289_289392


namespace num_of_valid_numbers_l289_289934

def is_valid_number (n : ℕ) : Prop :=
  let a := n / 10
  let b := n % 10
  a >= 1 ∧ a <= 9 ∧ b >= 0 ∧ b <= 9 ∧ (9 * a) % 10 = 4

theorem num_of_valid_numbers : ∃ n, n = 10 :=
by {
  sorry
}

end num_of_valid_numbers_l289_289934


namespace sufficient_not_necessary_l289_289830

theorem sufficient_not_necessary (a b : ℝ) (h : b > a ∧ a > 0) : (1 / a > 1 / b) :=
by {
  sorry -- the proof steps are intentionally omitted
}

end sufficient_not_necessary_l289_289830


namespace distance_between_towns_proof_l289_289926

noncomputable def distance_between_towns : ℕ :=
  let distance := 300
  let time_after_departure := 2
  let remaining_distance := 40
  let speed_difference := 10
  let total_distance_covered := distance - remaining_distance
  let speed_slower_train := 60
  let speed_faster_train := speed_slower_train + speed_difference
  let relative_speed := speed_slower_train + speed_faster_train
  distance

theorem distance_between_towns_proof 
  (distance : ℕ) 
  (time_after_departure : ℕ) 
  (remaining_distance : ℕ) 
  (speed_difference : ℕ) 
  (h1 : distance = 300) 
  (h2 : time_after_departure = 2) 
  (h3 : remaining_distance = 40) 
  (h4 : speed_difference = 10) 
  (speed_slower_train speed_faster_train relative_speed : ℕ)
  (h_speed_faster : speed_faster_train = speed_slower_train + speed_difference)
  (h_relative_speed : relative_speed = speed_slower_train + speed_faster_train) :
  distance = 300 :=
by {
  sorry
}

end distance_between_towns_proof_l289_289926


namespace perfect_square_trinomial_l289_289077

theorem perfect_square_trinomial (a k : ℝ) : (∃ b : ℝ, (a^2 + 2*k*a + 9 = (a + b)^2)) ↔ (k = 3 ∨ k = -3) := 
by
  sorry

end perfect_square_trinomial_l289_289077


namespace ratio_Bill_Cary_l289_289390

noncomputable def Cary_height : ℝ := 72
noncomputable def Jan_height : ℝ := 42
noncomputable def Bill_height : ℝ := Jan_height - 6

theorem ratio_Bill_Cary : Bill_height / Cary_height = 1 / 2 :=
by
  sorry

end ratio_Bill_Cary_l289_289390


namespace fishing_tomorrow_l289_289883

theorem fishing_tomorrow (yesterday_fishers today_fishers : ℕ)
  (every_day_fishers every_other_day_fishers every_three_days_fishers : ℕ)
  (total_population : ℕ):
  yesterday_fishers = 12 → 
  today_fishers = 10 → 
  every_day_fishers = 7 → 
  every_other_day_fishers = 8 → 
  every_three_days_fishers = 3 → 
  total_population = yesterday_fishers + today_fishers + (total_population - (every_day_fishers + every_other_day_fishers + every_three_days_fishers)) →
  ∃ tomorrow_fishers : ℕ, tomorrow_fishers = 15 :=
by {
  -- This is a statement definition, the proof is not required and thus marked as "sorry:"
  sorry
}

end fishing_tomorrow_l289_289883


namespace geric_bills_l289_289419

variable (G K J : ℕ)

theorem geric_bills (h1 : G = 2 * K) 
                    (h2 : K = J - 2) 
                    (h3 : J = 7 + 3) : 
    G = 16 := by
  sorry

end geric_bills_l289_289419


namespace gcd_of_polynomials_l289_289276

theorem gcd_of_polynomials (b : ℤ) (h : b % 2 = 1 ∧ 8531 ∣ b) :
  Int.gcd (8 * b^2 + 33 * b + 125) (4 * b + 15) = 5 :=
by
  sorry

end gcd_of_polynomials_l289_289276


namespace total_gray_area_trees_l289_289345

/-- 
Three aerial photos were taken by the drone, each capturing the same number of trees.
First rectangle has 100 trees in total and 82 trees in the white area.
Second rectangle has 90 trees in total and 82 trees in the white area.
Prove that the number of trees in gray areas in both rectangles is 26.
-/
theorem total_gray_area_trees : (100 - 82) + (90 - 82) = 26 := 
by sorry

end total_gray_area_trees_l289_289345


namespace feb1_is_wednesday_l289_289844

-- Define the days of the week as a data type
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open DayOfWeek

-- Define a function that models the backward count for days of the week from a given day
def days_backward (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days % 7 with
  | 0 => start
  | 1 => match start with
         | Sunday => Saturday
         | Monday => Sunday
         | Tuesday => Monday
         | Wednesday => Tuesday
         | Thursday => Wednesday
         | Friday => Thursday
         | Saturday => Friday
  | 2 => match start with
         | Sunday => Friday
         | Monday => Saturday
         | Tuesday => Sunday
         | Wednesday => Monday
         | Thursday => Tuesday
         | Friday => Wednesday
         | Saturday => Thursday
  | 3 => match start with
         | Sunday => Thursday
         | Monday => Friday
         | Tuesday => Saturday
         | Wednesday => Sunday
         | Thursday => Monday
         | Friday => Tuesday
         | Saturday => Wednesday
  | 4 => match start with
         | Sunday => Wednesday
         | Monday => Thursday
         | Tuesday => Friday
         | Wednesday => Saturday
         | Thursday => Sunday
         | Friday => Monday
         | Saturday => Tuesday
  | 5 => match start with
         | Sunday => Tuesday
         | Monday => Wednesday
         | Tuesday => Thursday
         | Wednesday => Friday
         | Thursday => Saturday
         | Friday => Sunday
         | Saturday => Monday
  | 6 => match start with
         | Sunday => Monday
         | Monday => Tuesday
         | Tuesday => Wednesday
         | Wednesday => Thursday
         | Thursday => Friday
         | Friday => Saturday
         | Saturday => Sunday
  | _ => start  -- This case is unreachable because days % 7 is always between 0 and 6

-- Proof statement: given February 28 is a Tuesday, prove that February 1 is a Wednesday
theorem feb1_is_wednesday (h : days_backward Tuesday 27 = Wednesday) : True :=
by
  sorry

end feb1_is_wednesday_l289_289844


namespace divisibility_by_seven_l289_289364

theorem divisibility_by_seven : (∃ k : ℤ, (-8)^2019 + (-8)^2018 = 7 * k) :=
sorry

end divisibility_by_seven_l289_289364


namespace isosceles_triangle_k_value_l289_289605

theorem isosceles_triangle_k_value 
(side1 : ℝ)
(side2 side3 : ℝ)
(k : ℝ)
(h1 : side1 = 3 ∨ side2 = 3 ∨ side3 = 3)
(h2 : side1 = side2 ∨ side1 = side3 ∨ side2 = side3)
(h3 : Polynomial.eval side1 (Polynomial.C k + Polynomial.X ^ 2) = 0 
    ∨ Polynomial.eval side2 (Polynomial.C k + Polynomial.X ^ 2) = 0 
    ∨ Polynomial.eval side3 (Polynomial.C k + Polynomial.X ^ 2) = 0) :
k = 3 ∨ k = 4 :=
sorry

end isosceles_triangle_k_value_l289_289605


namespace quotient_of_2213_div_13_in_base4_is_53_l289_289811

-- Definitions of the numbers in base 4
def n₁ : ℕ := 2 * 4^3 + 2 * 4^2 + 1 * 4^1 + 3 * 4^0  -- 2213_4 in base 10
def n₂ : ℕ := 1 * 4^1 + 3 * 4^0  -- 13_4 in base 10

-- The correct quotient in base 4 (converted from quotient in base 10)
def expected_quotient : ℕ := 5 * 4^1 + 3 * 4^0  -- 53_4 in base 10

-- The proposition we want to prove
theorem quotient_of_2213_div_13_in_base4_is_53 : n₁ / n₂ = expected_quotient := by
  sorry

end quotient_of_2213_div_13_in_base4_is_53_l289_289811


namespace average_value_of_series_l289_289389

theorem average_value_of_series (z : ℤ) :
  let series := [0^2, (2*z)^2, (4*z)^2, (8*z)^2]
  let sum_series := series.sum
  let n := series.length
  sum_series / n = 21 * z^2 :=
by
  let series := [0^2, (2*z)^2, (4*z)^2, (8*z)^2]
  let sum_series := series.sum
  let n := series.length
  sorry

end average_value_of_series_l289_289389


namespace fishers_tomorrow_l289_289867

-- Define the groups of fishers and their fishing pattern
def everyday_fishers := 7
def every_other_day_fishers := 8
def every_three_days_fishers := 3

-- Given counts for yesterday and today
def fishers_yesterday := 12
def fishers_today := 10

-- The problem to prove: 15 people will fish tomorrow
theorem fishers_tomorrow : 
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (everyday_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  15 = everyday_fishers + every_other_day_fishers / 2 + every_three_days_fishers / 3 :=
begin
  sorry
end

end fishers_tomorrow_l289_289867


namespace factor_polynomial_l289_289848

theorem factor_polynomial (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) = 
  (x - y) * (y - z) * (z - x) * (x * y + x * z + y * z) := by
  sorry

end factor_polynomial_l289_289848


namespace point_reflection_x_axis_l289_289142

-- Definition of the original point P
def P : ℝ × ℝ := (-2, 5)

-- Function to reflect a point across the x-axis
def reflect_x_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.1, -point.2)

-- Our theorem
theorem point_reflection_x_axis :
  reflect_x_axis P = (-2, -5) := by
  sorry

end point_reflection_x_axis_l289_289142


namespace part_a_part_b_l289_289952

-- Part (a): Proving that 91 divides n^37 - n for all integers n
theorem part_a (n : ℤ) : 91 ∣ (n ^ 37 - n) := 
sorry

-- Part (b): Finding the largest k that divides n^37 - n for all integers n is 3276
theorem part_b (n : ℤ) : ∀ k : ℤ, (k > 0) → (∀ n : ℤ, k ∣ (n ^ 37 - n)) → k ≤ 3276 :=
sorry

end part_a_part_b_l289_289952


namespace triangle_shape_l289_289708

theorem triangle_shape (a b : ℝ) (A B : ℝ) (hA : 0 < A) (hB : A < π) (h : a * Real.cos A = b * Real.cos B) :
  (A = B ∨ A + B = π / 2 ∨ a = b) :=
by
  sorry

end triangle_shape_l289_289708


namespace jade_and_julia_total_money_l289_289311

theorem jade_and_julia_total_money (x : ℕ) : 
  let jade_initial := 38 
  let julia_initial := jade_initial / 2 
  let jade_after := jade_initial + x 
  let julia_after := julia_initial + x 
  jade_after + julia_after = 57 + 2 * x := by
  sorry

end jade_and_julia_total_money_l289_289311


namespace problem_quadratic_has_real_root_l289_289118

theorem problem_quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end problem_quadratic_has_real_root_l289_289118


namespace range_of_a_l289_289697

theorem range_of_a (a : ℝ) (x : ℝ) :
  ((a < x ∧ x < a + 2) → x > 3) ∧ ¬(∀ x, (x > 3) → (a < x ∧ x < a + 2)) → a ≥ 3 :=
by
  sorry

end range_of_a_l289_289697


namespace sum_of_prime_factors_l289_289947

theorem sum_of_prime_factors (n : ℕ) (h : n = 257040) : 
  (2 + 5 + 3 + 107 = 117) :=
by sorry

end sum_of_prime_factors_l289_289947


namespace certain_number_is_correct_l289_289411

def m : ℕ := 72483

theorem certain_number_is_correct : 9999 * m = 724827405 := by
  sorry

end certain_number_is_correct_l289_289411


namespace ed_money_left_after_hotel_stay_l289_289398

theorem ed_money_left_after_hotel_stay 
  (night_rate : ℝ) (morning_rate : ℝ) 
  (initial_money : ℝ) (hours_night : ℕ) (hours_morning : ℕ) 
  (remaining_money : ℝ) : 
  night_rate = 1.50 → morning_rate = 2.00 → initial_money = 80 → 
  hours_night = 6 → hours_morning = 4 → 
  remaining_money = 63 :=
by
  intros h1 h2 h3 h4 h5
  let cost_night := night_rate * hours_night
  let cost_morning := morning_rate * hours_morning
  let total_cost := cost_night + cost_morning
  let money_left := initial_money - total_cost
  sorry

end ed_money_left_after_hotel_stay_l289_289398


namespace fishers_tomorrow_l289_289874

-- Definitions based on conditions
def people_every_day : ℕ := 7
def people_every_other_day : ℕ := 8
def people_every_three_days : ℕ := 3
def people_yesterday : ℕ := 12
def people_today : ℕ := 10

-- Theorem to be proved
theorem fishers_tomorrow (people_every_day people_every_other_day people_every_three_days people_yesterday people_today : ℕ) : 
  people_every_day = 7 ∧ 
  people_every_other_day = 8 ∧ 
  people_every_three_days = 3 ∧
  people_yesterday = 12 ∧ 
  people_today = 10 →
  15 := 
by {
  sorry
}

end fishers_tomorrow_l289_289874


namespace rachel_math_homework_pages_l289_289593

-- Define the number of pages of math homework and reading homework
def pagesReadingHomework : ℕ := 4

theorem rachel_math_homework_pages (M : ℕ) (h1 : M + 1 = pagesReadingHomework) : M = 3 :=
by
  sorry

end rachel_math_homework_pages_l289_289593


namespace remainder_mod56_l289_289160

theorem remainder_mod56 (N : ℕ) (h : N % 8 = 5) : ∃ r, r ∈ {5, 13, 21, 29, 37, 45, 53} ∧ N % 56 = r :=
by
  sorry

end remainder_mod56_l289_289160


namespace trajectory_of_M_ellipse_trajectory_l289_289996

variable {x y : ℝ}

theorem trajectory_of_M (hx : x ≠ 5) (hnx : x ≠ -5)
  (h : (y / (x + 5)) * (y / (x - 5)) = -2) : 
  (2 * x^2 + y^2 = 50) :=
by
  -- Proof is omitted.
  sorry

theorem ellipse_trajectory (hx : x ≠ 5) (hnx : x ≠ -5) 
  (h : (y / (x + 5)) * (y / (x - 5)) = -2) : 
  (x^2 / 25 + y^2 / 50 = 1) :=
by
  -- Using the previous theorem to derive.
  have h1 : (2 * x^2 + y^2 = 50) := trajectory_of_M hx hnx h
  -- Proof of transformation is omitted.
  sorry

end trajectory_of_M_ellipse_trajectory_l289_289996


namespace integer_roots_of_quadratic_l289_289423

theorem integer_roots_of_quadratic (a : ℚ) :
  (∃ x₁ x₂ : ℤ, 
    a * x₁ * x₁ + (a + 1) * x₁ + (a - 1) = 0 ∧ 
    a * x₂ * x₂ + (a + 1) * x₂ + (a - 1) = 0 ∧ 
    x₁ ≠ x₂) ↔ 
      a = 0 ∨ a = -1/7 ∨ a = 1 :=
by
  sorry

end integer_roots_of_quadratic_l289_289423


namespace Q1_no_such_a_b_Q2_no_such_a_b_c_l289_289218

theorem Q1_no_such_a_b :
  ∀ (a b : ℕ), (0 < a) ∧ (0 < b) → ¬ (∀ n : ℕ, 0 < n → ∃ k : ℕ, k^2 = 2^n * a + 5^n * b) := sorry

theorem Q2_no_such_a_b_c :
  ∀ (a b c : ℕ), (0 < a) ∧ (0 < b) ∧ (0 < c) → ¬ (∀ n : ℕ, 0 < n → ∃ k : ℕ, k^2 = 2^n * a + 5^n * b + c) := sorry

end Q1_no_such_a_b_Q2_no_such_a_b_c_l289_289218


namespace midpoint_range_l289_289060

variable {x0 y0 : ℝ}

-- Conditions
def point_on_line1 (P : ℝ × ℝ) := P.1 + 2 * P.2 - 1 = 0
def point_on_line2 (Q : ℝ × ℝ) := Q.1 + 2 * Q.2 + 3 = 0
def is_midpoint (P Q M : ℝ × ℝ) := P.1 + Q.1 = 2 * M.1 ∧ P.2 + Q.2 = 2 * M.2
def midpoint_condition (M : ℝ × ℝ) := M.2 > M.1 + 2

-- Theorem
theorem midpoint_range
  (P Q M : ℝ × ℝ)
  (hP : point_on_line1 P)
  (hQ : point_on_line2 Q)
  (hM : is_midpoint P Q M)
  (h_cond : midpoint_condition M)
  (hx0 : x0 = M.1)
  (hy0 : y0 = M.2)
  : - (1 / 2) < y0 / x0 ∧ y0 / x0 < - (1 / 5) :=
sorry

end midpoint_range_l289_289060


namespace percentage_of_exceedance_l289_289235

theorem percentage_of_exceedance (x p : ℝ) (h : x = (p / 100) * x + 52.8) (hx : x = 60) : p = 12 :=
by 
  sorry

end percentage_of_exceedance_l289_289235


namespace fishers_tomorrow_l289_289868

-- Define the groups of fishers and their fishing pattern
def everyday_fishers := 7
def every_other_day_fishers := 8
def every_three_days_fishers := 3

-- Given counts for yesterday and today
def fishers_yesterday := 12
def fishers_today := 10

-- The problem to prove: 15 people will fish tomorrow
theorem fishers_tomorrow : 
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (everyday_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  15 = everyday_fishers + every_other_day_fishers / 2 + every_three_days_fishers / 3 :=
begin
  sorry
end

end fishers_tomorrow_l289_289868


namespace find_alcohol_quantity_l289_289484

theorem find_alcohol_quantity 
  (A W : ℝ) 
  (h1 : A / W = 2 / 5)
  (h2 : A / (W + 10) = 2 / 7) : 
  A = 10 :=
sorry

end find_alcohol_quantity_l289_289484


namespace relatively_prime_divisibility_l289_289983

theorem relatively_prime_divisibility (x y : ℕ) (h1 : Nat.gcd x y = 1) (h2 : y^2 * (y - x)^2 ∣ x^2 * (x + y)) :
  (x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 1) :=
sorry

end relatively_prime_divisibility_l289_289983


namespace total_cost_at_discount_l289_289968

-- Definitions for conditions
def original_price_notebook : ℕ := 15
def original_price_planner : ℕ := 10
def discount_rate : ℕ := 20
def number_of_notebooks : ℕ := 4
def number_of_planners : ℕ := 8

-- Theorem statement for the proof
theorem total_cost_at_discount :
  let discounted_price_notebook := original_price_notebook - (original_price_notebook * discount_rate / 100)
  let discounted_price_planner := original_price_planner - (original_price_planner * discount_rate / 100)
  let total_cost := (number_of_notebooks * discounted_price_notebook) + (number_of_planners * discounted_price_planner)
  total_cost = 112 :=
by
  sorry

end total_cost_at_discount_l289_289968


namespace tom_books_read_in_may_l289_289942

def books_read_in_june := 6
def books_read_in_july := 10
def total_books_read := 18

theorem tom_books_read_in_may : total_books_read - (books_read_in_june + books_read_in_july) = 2 :=
by sorry

end tom_books_read_in_may_l289_289942


namespace find_b_values_l289_289083

noncomputable def has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + b*x + 25 = 0

theorem find_b_values (b : ℝ) : has_real_root b ↔ b ∈ set.Iic (-10) ∪ set.Ici 10 := by
  sorry

end find_b_values_l289_289083


namespace intersection_of_A_and_B_l289_289286

open Set

variable {α : Type}

-- Definitions of the sets A and B
def A : Set ℤ := {-1, 0, 2, 3, 5}
def B : Set ℤ := {x | -1 < x ∧ x < 3}

-- Define the proof problem as a theorem
theorem intersection_of_A_and_B : A ∩ B = {0, 2} :=
by
  sorry

end intersection_of_A_and_B_l289_289286


namespace zoo_ticket_problem_l289_289936

def students_6A (total_cost_6A : ℕ) (saved_tickets_6A : ℕ) (ticket_price : ℕ) : ℕ :=
  let paid_tickets := (total_cost_6A / ticket_price)
  (paid_tickets + saved_tickets_6A)

def students_6B (total_cost_6B : ℕ) (total_students_6A : ℕ) (ticket_price : ℕ) : ℕ :=
  let paid_tickets := (total_cost_6B / ticket_price)
  let total_students := paid_tickets + (paid_tickets / 4)
  (total_students - total_students_6A)

theorem zoo_ticket_problem :
  (students_6A 1995 4 105 = 23) ∧
  (students_6B 4410 23 105 = 29) :=
by {
  -- The proof will follow the steps to confirm the calculations and final result
  sorry
}

end zoo_ticket_problem_l289_289936


namespace group_value_21_le_a_lt_41_l289_289001

theorem group_value_21_le_a_lt_41 : 
  (∀ a: ℤ, 21 ≤ a ∧ a < 41 → (21 + 41) / 2 = 31) :=
by 
  sorry

end group_value_21_le_a_lt_41_l289_289001


namespace total_books_l289_289470

theorem total_books (x : ℕ) (h1 : 3 * x + 2 * x + (3 / 2) * x > 3000) : 
  ∃ (T : ℕ), T = 3 * x + 2 * x + (3 / 2) * x ∧ T > 3000 ∧ T = 3003 := 
by 
  -- Our theorem states there exists an integer T such that the total number of books is 3003.
  sorry

end total_books_l289_289470


namespace inequality_solution_l289_289519

theorem inequality_solution (x : ℝ) (h₀ : x ≠ 0) (h₂ : x ≠ 2) : 
  (x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 2 ↔ (0 < x ∧ x ≤ 0.5) ∨ (6 ≤ x) :=
by { sorry }

end inequality_solution_l289_289519


namespace initial_budget_calculation_l289_289790

variable (flaskCost testTubeCost safetyGearCost totalExpenses remainingAmount initialBudget : ℕ)

theorem initial_budget_calculation (h1 : flaskCost = 150)
                               (h2 : testTubeCost = 2 * flaskCost / 3)
                               (h3 : safetyGearCost = testTubeCost / 2)
                               (h4 : totalExpenses = flaskCost + testTubeCost + safetyGearCost)
                               (h5 : remainingAmount = 25)
                               (h6 : initialBudget = totalExpenses + remainingAmount) :
                               initialBudget = 325 := by
  sorry

end initial_budget_calculation_l289_289790


namespace problem_l289_289443

def p (x y : Int) : Int :=
  if x ≥ 0 ∧ y ≥ 0 then x * y
  else if x < 0 ∧ y < 0 then x - 2 * y
  else if x ≥ 0 ∧ y < 0 then 2 * x + 3 * y
  else if x < 0 ∧ y ≥ 0 then x + 3 * y
  else 3 * x + y

theorem problem : p (p 2 (-3)) (p (-1) 4) = 28 := by
  sorry

end problem_l289_289443


namespace shop_sold_price_l289_289227

noncomputable def clock_selling_price (C : ℝ) : ℝ :=
  let buy_back_price := 0.60 * C
  let maintenance_cost := 0.10 * buy_back_price
  let total_spent := buy_back_price + maintenance_cost
  let selling_price := 1.80 * total_spent
  selling_price

theorem shop_sold_price (C : ℝ) (h1 : C - 0.60 * C = 100) :
  clock_selling_price C = 297 := by
  sorry

end shop_sold_price_l289_289227


namespace problem_l289_289339

theorem problem (a b : ℝ) (h₁ : a = -a) (h₂ : b = 1 / b) : a + b = 1 ∨ a + b = -1 :=
  sorry

end problem_l289_289339


namespace equivalent_integer_l289_289076

theorem equivalent_integer (a b n : ℤ) (h1 : a ≡ 33 [ZMOD 60]) (h2 : b ≡ 85 [ZMOD 60]) (hn : 200 ≤ n ∧ n ≤ 251) : 
  a - b ≡ 248 [ZMOD 60] :=
sorry

end equivalent_integer_l289_289076


namespace mike_games_l289_289908

theorem mike_games (init_money spent_money game_cost : ℕ) (h1 : init_money = 42) (h2 : spent_money = 10) (h3 : game_cost = 8) :
  (init_money - spent_money) / game_cost = 4 :=
by
  sorry

end mike_games_l289_289908


namespace number_of_students_playing_both_l289_289163

open Set

variable (U : Type) (F C : Set U)

theorem number_of_students_playing_both (T : ℕ) (nF : ℕ) (nC : ℕ) (nNeither : ℕ)
  (H1 : T = 470) 
  (H2 : nF = 325) 
  (H3 : nC = 175) 
  (H4 : nNeither = 50) :
  Nat.card (F ∩ C) = 80 := by 
  sorry

end number_of_students_playing_both_l289_289163


namespace inequality_solution_l289_289739

theorem inequality_solution (x : ℝ) :
    (x < 1 ∨ (3 < x ∧ x < 4) ∨ (4 < x ∧ x < 5) ∨ (5 < x ∧ x < 6) ∨ x > 6) ↔
    ((x - 1) * (x - 3) * (x - 4) / ((x - 2) * (x - 5) * (x - 6)) > 0) := by
  sorry

end inequality_solution_l289_289739


namespace radius_of_inscribed_circle_l289_289225

variable (height : ℝ) (alpha : ℝ)

theorem radius_of_inscribed_circle (h : ℝ) (α : ℝ) : 
∃ r : ℝ, r = (h / 2) * (Real.tan (Real.pi / 4 - α / 4)) ^ 2 := 
sorry

end radius_of_inscribed_circle_l289_289225


namespace quadratic_real_roots_l289_289135

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := by 
  sorry

end quadratic_real_roots_l289_289135


namespace fishing_tomorrow_l289_289881

theorem fishing_tomorrow (yesterday_fishers today_fishers : ℕ)
  (every_day_fishers every_other_day_fishers every_three_days_fishers : ℕ)
  (total_population : ℕ):
  yesterday_fishers = 12 → 
  today_fishers = 10 → 
  every_day_fishers = 7 → 
  every_other_day_fishers = 8 → 
  every_three_days_fishers = 3 → 
  total_population = yesterday_fishers + today_fishers + (total_population - (every_day_fishers + every_other_day_fishers + every_three_days_fishers)) →
  ∃ tomorrow_fishers : ℕ, tomorrow_fishers = 15 :=
by {
  -- This is a statement definition, the proof is not required and thus marked as "sorry:"
  sorry
}

end fishing_tomorrow_l289_289881


namespace number_of_penguins_l289_289553

-- Define the number of animals and zookeepers
def zebras : ℕ := 22
def tigers : ℕ := 8
def zookeepers : ℕ := 12
def headsLessThanFeetBy : ℕ := 132

-- Define the theorem to prove the number of penguins (P)
theorem number_of_penguins (P : ℕ) (H : P + zebras + tigers + zookeepers + headsLessThanFeetBy = 4 * P + 4 * zebras + 4 * tigers + 2 * zookeepers) : P = 10 :=
by
  sorry

end number_of_penguins_l289_289553


namespace f_zero_eq_one_f_positive_f_increasing_f_range_x_l289_289667

noncomputable def f : ℝ → ℝ := sorry
axiom f_condition1 : f 0 ≠ 0
axiom f_condition2 : ∀ x : ℝ, x > 0 → f x > 1
axiom f_condition3 : ∀ a b : ℝ, f (a + b) = f a * f b

theorem f_zero_eq_one : f 0 = 1 :=
sorry

theorem f_positive : ∀ x : ℝ, f x > 0 :=
sorry

theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y :=
sorry

theorem f_range_x (x : ℝ) (h : f x * f (2 * x - x^2) > 1) : x ∈ { x : ℝ | f x > 1 ∧ f (2 * x - x^2) > 1 } :=
sorry

end f_zero_eq_one_f_positive_f_increasing_f_range_x_l289_289667


namespace power_quotient_example_l289_289807

theorem power_quotient_example (a : ℕ) (m n : ℕ) (h : 23^11 / 23^8 = 23^(11 - 8)) : 23^3 = 12167 := by
  sorry

end power_quotient_example_l289_289807


namespace center_of_hyperbola_l289_289815

theorem center_of_hyperbola :
  (∃ h k : ℝ, ∀ x y : ℝ, (3*y + 3)^2 / 49 - (2*x - 5)^2 / 9 = 1 ↔ x = h ∧ y = k) → 
  h = 5 / 2 ∧ k = -1 :=
by
  sorry

end center_of_hyperbola_l289_289815


namespace find_x_l289_289413

theorem find_x (x : ℝ) (h : x > 0) (area : 1 / 2 * (2 * x) * x = 72) : x = 6 * Real.sqrt 2 :=
by
  sorry

end find_x_l289_289413


namespace solve_inequality_l289_289172

open Set

theorem solve_inequality (a x : ℝ) : 
  (x - 2) * (a * x - 2) > 0 → 
  (a = 0 ∧ x < 2) ∨ 
  (a < 0 ∧ (2/a) < x ∧ x < 2) ∨ 
  (0 < a ∧ a < 1 ∧ ((x < 2 ∨ x > 2/a))) ∨ 
  (a = 1 ∧ x ≠ 2) ∨ 
  (a > 1 ∧ ((x < 2/a ∨ x > 2)))
  := sorry

end solve_inequality_l289_289172


namespace sum_of_cubes_of_consecutive_numbers_divisible_by_9_l289_289591

theorem sum_of_cubes_of_consecutive_numbers_divisible_by_9 (a : ℕ) (h : a > 1) : 
  9 ∣ ((a - 1)^3 + a^3 + (a + 1)^3) := 
by 
  sorry

end sum_of_cubes_of_consecutive_numbers_divisible_by_9_l289_289591


namespace second_derivative_l289_289705

noncomputable def y (x : ℝ) : ℝ := x^3 + Real.log x / Real.log 2 + Real.exp (-x)

theorem second_derivative (x : ℝ) : (deriv^[2] y x) = 3 * x^2 + (1 / (x * Real.log 2)) - Real.exp (-x) :=
by
  sorry

end second_derivative_l289_289705


namespace taimour_paints_fence_alone_in_15_hours_l289_289213

theorem taimour_paints_fence_alone_in_15_hours :
  ∀ (T : ℝ), (∀ (J : ℝ), J = T / 2 → (1 / J + 1 / T = 1 / 5)) → T = 15 :=
by
  intros T h
  have h1 := h (T / 2) rfl
  sorry

end taimour_paints_fence_alone_in_15_hours_l289_289213


namespace general_formula_a_n_general_formula_b_n_sum_c_n_T_n_l289_289322

open Classical

axiom S_n : ℕ → ℝ
axiom a_n : ℕ → ℝ
axiom b_n : ℕ → ℝ
axiom c_n : ℕ → ℝ
axiom T_n : ℕ → ℝ

noncomputable def general_a_n (n : ℕ) : ℝ :=
  sorry

axiom h1 : ∀ n, S_n n + a_n n = 2

theorem general_formula_a_n : ∀ n, a_n n = 1 / 2^(n-1) :=
  sorry

axiom h2 : b_n 1 = a_n 1
axiom h3 : ∀ n ≥ 2, b_n n = 3 * b_n (n-1) / (b_n (n-1) + 3)

theorem general_formula_b_n : ∀ n, b_n n = 3 / (n + 2) ∧
  (∀ n, 1 / b_n n = 1 + (n - 1) / 3) :=
  sorry

axiom h4 : ∀ n, c_n n = a_n n / b_n n

theorem sum_c_n_T_n : ∀ n, T_n n = 8 / 3 - (n + 4) / (3 * 2^(n-1)) :=
  sorry

end general_formula_a_n_general_formula_b_n_sum_c_n_T_n_l289_289322


namespace combined_weight_l289_289435

variable (J S : ℝ)

-- Given conditions
def jake_current_weight := (J = 152)
def lose_weight_equation := (J - 32 = 2 * S)

-- Question: combined weight of Jake and his sister
theorem combined_weight (h1 : jake_current_weight J) (h2 : lose_weight_equation J S) : J + S = 212 :=
by
  sorry

end combined_weight_l289_289435


namespace sum_first_seven_terms_of_arith_seq_l289_289683

-- Define an arithmetic sequence
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Conditions: a_2 = 10 and a_5 = 1
def a_2 := 10
def a_5 := 1

-- The sum of the first 7 terms of the sequence
theorem sum_first_seven_terms_of_arith_seq (a d : ℤ) :
  arithmetic_seq a d 1 = a_2 →
  arithmetic_seq a d 4 = a_5 →
  (7 * a + (7 * 6 / 2) * d = 28) :=
by
  sorry

end sum_first_seven_terms_of_arith_seq_l289_289683


namespace machine_copies_l289_289492

theorem machine_copies (x : ℕ) (h1 : ∀ t : ℕ, t = 30 → 30 * t = 900)
  (h2 : 900 + 30 * 30 = 2550) : x = 55 :=
by
  sorry

end machine_copies_l289_289492


namespace power_function_solution_l289_289186

theorem power_function_solution (m : ℝ) 
    (h1 : m^2 - 3 * m + 3 = 1) 
    (h2 : m - 1 ≠ 0) : m = 2 := 
by
  sorry

end power_function_solution_l289_289186


namespace average_price_of_fruit_l289_289509

theorem average_price_of_fruit 
  (price_apple price_orange : ℝ)
  (total_fruits initial_fruits kept_oranges kept_fruits : ℕ)
  (average_price_kept average_price_initial : ℝ)
  (h1 : price_apple = 40)
  (h2 : price_orange = 60)
  (h3 : initial_fruits = 10)
  (h4 : kept_oranges = initial_fruits - 6)
  (h5 : average_price_kept = 50) :
  average_price_initial = 56 := 
sorry

end average_price_of_fruit_l289_289509


namespace problem_quadratic_has_real_root_l289_289121

theorem problem_quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end problem_quadratic_has_real_root_l289_289121


namespace fishing_problem_l289_289860

theorem fishing_problem
  (everyday : ℕ)
  (every_other_day : ℕ)
  (every_three_days : ℕ)
  (yesterday_fishing : ℕ)
  (today_fishing : ℕ)
  (h_everyday : everyday = 7)
  (h_every_other_day : every_other_day = 8)
  (h_every_three_days : every_three_days = 3)
  (h_yesterday_fishing : yesterday_fishing = 12)
  (h_today_fishing : today_fishing = 10) :
  (every_three_days + everyday + (every_other_day - (yesterday_fishing - everyday))) = 15 := by
  sorry

end fishing_problem_l289_289860


namespace intersection_complement_eq_singleton_l289_289289

def U : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) }
def M : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ (y - 3) / (x - 2) = 1 }
def N : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ y = x + 1 }
def complement_U (M : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := { p | p ∈ U ∧ p ∉ M }

theorem intersection_complement_eq_singleton :
  N ∩ complement_U M = {(2,3)} :=
by
  sorry

end intersection_complement_eq_singleton_l289_289289


namespace pieces_per_box_correct_l289_289482

-- Define the number of boxes Will bought
def total_boxes_bought := 7

-- Define the number of boxes Will gave to his brother
def boxes_given := 3

-- Define the number of pieces left with Will
def pieces_left := 16

-- Define the function to find the pieces per box
def pieces_per_box (total_boxes : Nat) (given_away : Nat) (remaining_pieces : Nat) : Nat :=
  remaining_pieces / (total_boxes - given_away)

-- Prove that each box contains 4 pieces of chocolate candy
theorem pieces_per_box_correct : pieces_per_box total_boxes_bought boxes_given pieces_left = 4 :=
by
  sorry

end pieces_per_box_correct_l289_289482


namespace ratio_of_volumes_cone_cylinder_l289_289408

theorem ratio_of_volumes_cone_cylinder (r h_cylinder : ℝ) (h_cone : ℝ) (h_radius : r = 4) (h_height_cylinder : h_cylinder = 12) (h_height_cone : h_cone = h_cylinder / 2) :
  ((1/3) * (π * r^2 * h_cone)) / (π * r^2 * h_cylinder) = 1 / 6 :=
by
  -- Definitions and assumptions are directly included from the conditions.
  sorry

end ratio_of_volumes_cone_cylinder_l289_289408


namespace handshakes_count_l289_289659

def num_teams : ℕ := 4
def players_per_team : ℕ := 2
def total_players : ℕ := num_teams * players_per_team
def shakeable_players (total : ℕ) : ℕ := total * (total - players_per_team) / 2

theorem handshakes_count :
  shakeable_players total_players = 24 :=
by
  sorry

end handshakes_count_l289_289659


namespace correlation_comparison_l289_289183

/-- The data for variables x and y are (1, 3), (2, 5.3), (3, 6.9), (4, 9.1), and (5, 10.8) -/
def xy_data : List (Int × Float) := [(1, 3), (2, 5.3), (3, 6.9), (4, 9.1), (5, 10.8)]

/-- The data for variables U and V are (1, 12.7), (2, 10.2), (3, 7), (4, 3.6), and (5, 1) -/
def UV_data : List (Int × Float) := [(1, 12.7), (2, 10.2), (3, 7), (4, 3.6), (5, 1)]

/-- r1 is the linear correlation coefficient between y and x -/
noncomputable def r1 : Float := sorry

/-- r2 is the linear correlation coefficient between V and U -/
noncomputable def r2 : Float := sorry

/-- The problem is to prove that r2 < 0 < r1 given the data conditions -/
theorem correlation_comparison : r2 < 0 ∧ 0 < r1 := 
by 
  sorry

end correlation_comparison_l289_289183


namespace greatest_of_3_consecutive_integers_l289_289356

theorem greatest_of_3_consecutive_integers (x : ℤ) (h : x + (x + 1) + (x + 2) = 24) : (x + 2) = 9 :=
by
-- Proof would go here.
sorry

end greatest_of_3_consecutive_integers_l289_289356


namespace fishing_tomorrow_l289_289861

theorem fishing_tomorrow 
  (P1 : ∀ day : ℕ, 7 ∈ {p | p goes fishing on day})
  (P2 : ∀ day : ℕ, day % 2 = 0 → 8 ∈ {p | p goes fishing on day})
  (P3 : ∀ day : ℕ, day % 3 = 0 → 3 ∈ {p | p goes fishing on day})
  (P4 : ℕ)
  (yesterday : ℕ)
  (today : ℕ)
  (tomorrow : ℕ)
  (hyesterday : yesterday = 12)
  (htoday : today = 10)
  : tomorrow = 15 := by
  sorry

end fishing_tomorrow_l289_289861


namespace distances_from_median_l289_289608

theorem distances_from_median (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  ∃ (x y : ℝ), x = (b * c) / (a + b) ∧ y = (a * c) / (a + b) ∧ x + y = c :=
by
  sorry

end distances_from_median_l289_289608


namespace solve_inequality_l289_289599

theorem solve_inequality : { x : ℝ // (x < -1) ∨ (-2/3 < x) } :=
sorry

end solve_inequality_l289_289599


namespace cos_six_arccos_two_fifths_l289_289245

noncomputable def arccos (x : ℝ) : ℝ := Real.arccos x
noncomputable def cos (x : ℝ) : ℝ := Real.cos x

theorem cos_six_arccos_two_fifths : cos (6 * arccos (2 / 5)) = 12223 / 15625 := 
by
  sorry

end cos_six_arccos_two_fifths_l289_289245


namespace smallest_N_l289_289370

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

end smallest_N_l289_289370


namespace min_value_frac_sum_l289_289685

open Real

theorem min_value_frac_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 1) :
  3 ≤ (1 / (x + y)) + ((x + y) / z) := 
sorry

end min_value_frac_sum_l289_289685


namespace quadratic_real_root_iff_b_range_l289_289102

open Real

theorem quadratic_real_root_iff_b_range (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_real_root_iff_b_range_l289_289102


namespace relation_between_3a5_3b5_l289_289681

theorem relation_between_3a5_3b5 (a b : ℝ) (h : a > b) : 3 * a + 5 > 3 * b + 5 := by
  sorry

end relation_between_3a5_3b5_l289_289681


namespace problem1_problem2_l289_289737

open Real

theorem problem1 : sin (420 * π / 180) * cos (330 * π / 180) + sin (-690 * π / 180) * cos (-660 * π / 180) = 1 := by
  sorry

theorem problem2 (α : ℝ) : 
  (sin (π / 2 + α) * cos (π / 2 - α) / cos (π + α)) + 
  (sin (π - α) * cos (π / 2 + α) / sin (π + α)) = 0 := by
  sorry

end problem1_problem2_l289_289737


namespace DM_eq_r_plus_R_l289_289516

noncomputable def radius_incircle (A B D : ℝ) (s K : ℝ) : ℝ := K / s

noncomputable def radius_excircle (A C D : ℝ) (s' K' : ℝ) (AD : ℝ) : ℝ := K' / (s' - AD)

theorem DM_eq_r_plus_R 
  (A B C D M : ℝ)
  (h1 : A ≠ B)
  (h2 : B ≠ C)
  (h3 : A ≠ C)
  (h4 : D = (B + C) / 2)
  (h5 : M = (B + C) / 2)
  (r : ℝ)
  (h6 : r = radius_incircle A B D ((A + B + D) / 2) (abs ((A - B) * (A - D) / 2)))
  (R : ℝ)
  (h7 : R = radius_excircle A C D ((A + C + D) / 2) (abs ((A - C) * (A - D) / 2)) (abs (A - D))) :
  dist D M =r + R :=
by sorry

end DM_eq_r_plus_R_l289_289516


namespace floor_alpha_six_eq_three_l289_289298

noncomputable def floor_of_alpha_six (α : ℝ) (h : α^5 - α^3 + α - 2 = 0) : ℤ :=
  Int.floor (α^6)

theorem floor_alpha_six_eq_three (α : ℝ) (h : α^5 - α^3 + α - 2 = 0) : floor_of_alpha_six α h = 3 :=
sorry

end floor_alpha_six_eq_three_l289_289298


namespace point_on_line_eq_l289_289688

theorem point_on_line_eq (a b : ℝ) (h : b = -3 * a - 4) : b + 3 * a + 4 = 0 :=
by
  sorry

end point_on_line_eq_l289_289688


namespace race_winner_and_liar_l289_289821

def Alyosha_statement (pos : ℕ → Prop) : Prop := ¬ pos 1 ∧ ¬ pos 4
def Borya_statement (pos : ℕ → Prop) : Prop := ¬ pos 4
def Vanya_statement (pos : ℕ → Prop) : Prop := pos 1
def Grisha_statement (pos : ℕ → Prop) : Prop := pos 4

def three_true_one_false (s1 s2 s3 s4 : Prop) : Prop := 
  (s1 ∧ s2 ∧ s3 ∧ ¬ s4) ∨
  (s1 ∧ s2 ∧ ¬ s3 ∧ s4) ∨
  (s1 ∧ ¬ s2 ∧ s3 ∧ s4) ∨
  (¬ s1 ∧ s2 ∧ s3 ∧ s4)

def race_result (pos : ℕ → Prop) : Prop :=
  Vanya_statement pos ∧
  three_true_one_false (Alyosha_statement pos) (Borya_statement pos) (Vanya_statement pos) (Grisha_statement pos) ∧
  Borya_statement pos = false

theorem race_winner_and_liar:
  ∃ (pos : ℕ → Prop), race_result pos :=
sorry

end race_winner_and_liar_l289_289821


namespace tan_alpha_tan_beta_l289_289073

theorem tan_alpha_tan_beta (α β : ℝ) (h1 : Real.cos (α + β) = 3 / 5) (h2 : Real.cos (α - β) = 4 / 5) :
  Real.tan α * Real.tan β = 1 / 7 := by
  sorry

end tan_alpha_tan_beta_l289_289073


namespace negation_of_p_is_correct_l289_289838

variable (c : ℝ)

-- Proposition p defined as: there exists c > 0 such that x^2 - x + c = 0 has a solution
def proposition_p : Prop :=
  ∃ c > 0, ∃ x : ℝ, x^2 - x + c = 0

-- Negation of proposition p
def neg_proposition_p : Prop :=
  ∀ c > 0, ¬ ∃ x : ℝ, x^2 - x + c = 0

-- The Lean statement to prove
theorem negation_of_p_is_correct :
  neg_proposition_p ↔ (∀ c > 0, ¬ ∃ x : ℝ, x^2 - x + c = 0) :=
by
  sorry

end negation_of_p_is_correct_l289_289838


namespace prob_both_societies_have_at_least_one_participant_prob_AB_same_CD_not_same_l289_289190

-- Define the students and societies
inductive Student
| A | B | C | D

inductive Society
| LoveHeart | LiteraryStyle

open Student Society

-- Probability that both societies have at least one participant
theorem prob_both_societies_have_at_least_one_participant :
  P (has_participants LoveHeart ∧ has_participants LiteraryStyle) = 1 / 8 := 
sorry 

-- Probability that A and B are in the same society, while C and D are not
theorem prob_AB_same_CD_not_same :
  P (same_society A B ∧ ¬ same_society C D) = 1 / 4 := 
sorry

-- Helper definition: A student being in a society
def in_society (s : Student) (soc : Society) : Prop := sorry

-- Helper definition: A society having at least one participant
def has_participants (soc : Society) : Prop := sorry

-- Helper definition: Two students being in the same society
def same_society (s1 s2 : Student) : Prop := sorry

end prob_both_societies_have_at_least_one_participant_prob_AB_same_CD_not_same_l289_289190


namespace fishing_tomorrow_l289_289870

theorem fishing_tomorrow (every_day_fishers every_other_day_fishers every_three_days_fishers fishers_yesterday fishers_today : ℕ) :
  (every_day_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (every_three_days_fishers + every_day_fishers + (every_other_day_fishers - (fishers_yesterday - every_day_fishers)) = 15) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  simp
  sorry

end fishing_tomorrow_l289_289870


namespace ex_sq_sum_l289_289075

theorem ex_sq_sum (x y : ℝ) (h1 : (x + y)^2 = 9) (h2 : x * y = -1) : x^2 + y^2 = 11 :=
by
  sorry

end ex_sq_sum_l289_289075


namespace quarters_for_soda_l289_289579

def quarters_for_chips := 4
def total_dollars := 4

theorem quarters_for_soda :
  (total_dollars * 4) - quarters_for_chips = 12 :=
by
  sorry

end quarters_for_soda_l289_289579


namespace arithmetic_sequence_a14_l289_289271

theorem arithmetic_sequence_a14 (a : ℕ → ℤ) (h1 : a 4 = 5) (h2 : a 9 = 17) (h3 : 2 * a 9 = a 14 + a 4) : a 14 = 29 := sorry

end arithmetic_sequence_a14_l289_289271


namespace smallest_number_among_four_l289_289025

theorem smallest_number_among_four (a b c d : ℤ) (h1 : a = 2023) (h2 : b = 2022) (h3 : c = -2023) (h4 : d = -2022) : 
  min (min a (min b c)) d = -2023 :=
by
  rw [h1, h2, h3, h4]
  sorry

end smallest_number_among_four_l289_289025


namespace three_x_plus_y_eq_zero_l289_289779

theorem three_x_plus_y_eq_zero (x y : ℝ) (h : (2 * x + y) ^ 3 + x ^ 3 + 3 * x + y = 0) : 3 * x + y = 0 :=
sorry

end three_x_plus_y_eq_zero_l289_289779


namespace sum_of_repeating_decimals_l289_289041

noncomputable def x : ℚ := 1 / 9
noncomputable def y : ℚ := 2 / 99
noncomputable def z : ℚ := 3 / 999

theorem sum_of_repeating_decimals :
  x + y + z = 134 / 999 := by
  sorry

end sum_of_repeating_decimals_l289_289041


namespace angle_C_measure_l289_289800

theorem angle_C_measure
  (D C : ℝ)
  (h1 : C + D = 90)
  (h2 : C = 3 * D) :
  C = 67.5 :=
by
  sorry

end angle_C_measure_l289_289800


namespace employee_discount_percentage_l289_289499

def wholesale_cost : ℝ := 200
def retail_markup : ℝ := 0.20
def employee_paid_price : ℝ := 228

theorem employee_discount_percentage :
  let retail_price := wholesale_cost * (1 + retail_markup)
  let discount := retail_price - employee_paid_price
  (discount / retail_price) * 100 = 5 := by
  sorry

end employee_discount_percentage_l289_289499


namespace guide_is_knight_l289_289969

-- Definitions
def knight (p : Prop) : Prop := p
def liar (p : Prop) : Prop := ¬p

-- Conditions
variable (GuideClaimsKnight : Prop)
variable (SecondResidentClaimsKnight : Prop)
variable (GuideReportsAccurately : Prop)

-- Proof problem
theorem guide_is_knight
  (GuideClaimsKnight : Prop)
  (SecondResidentClaimsKnight : Prop)
  (GuideReportsAccurately : (GuideClaimsKnight ↔ SecondResidentClaimsKnight)) :
  GuideClaimsKnight := 
sorry

end guide_is_knight_l289_289969


namespace area_reduction_is_correct_l289_289384

-- Define the original area of the equilateral triangle
def original_area := 100 * Real.sqrt 3

-- Define the reduction in side length of the triangle
def side_reduction := 6

-- Calculate the side length of the original equilateral triangle
noncomputable def original_side_length : ℝ := Real.sqrt (4 * original_area / Real.sqrt 3)

-- Define the new side length after reduction
def new_side_length := original_side_length - side_reduction

-- Define the area of an equilateral triangle given its side length
noncomputable def area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

-- Calculate the new area after the side length reduction
noncomputable def new_area : ℝ := area new_side_length

-- The decrease in area of the equilateral triangle
noncomputable def area_decrease : ℝ := original_area - new_area

-- The proof statement showing the decrease in area is 51√3 cm²
theorem area_reduction_is_correct : area_decrease = 51 * Real.sqrt 3 := 
by sorry

end area_reduction_is_correct_l289_289384


namespace probability_two_queens_or_at_least_one_king_l289_289434

/-- Prove that the probability of either drawing two queens or drawing at least one king 
    when 2 cards are selected randomly from a standard deck of 52 cards is 2/13. -/
theorem probability_two_queens_or_at_least_one_king :
  (∃ (kq pk pq : ℚ), kq = 4 ∧
                     pk = 4 ∧
                     pq = 52 ∧
                     (∃ (p : ℚ), p = (kq*(kq-1))/(pq*(pq-1)) + (pk/pq)*(pq-pk)/(pq-1) + (kq*(kq-1))/(pq*(pq-1)) ∧
                            p = 2/13)) :=
by {
  sorry
}

end probability_two_queens_or_at_least_one_king_l289_289434


namespace sequence_diff_exists_l289_289717

theorem sequence_diff_exists (x : ℕ → ℕ) (h1 : x 1 = 1) (h2 : ∀ n : ℕ, 1 ≤ n → x n < x (n + 1) ∧ x (n + 1) ≤ 2 * n) :
  ∀ k : ℕ, ∃ r s : ℕ, x r - x s = k :=
by
  sorry

end sequence_diff_exists_l289_289717


namespace James_weight_after_gain_l289_289312

theorem James_weight_after_gain 
    (initial_weight : ℕ)
    (muscle_gain_perc : ℕ)
    (fat_gain_fraction : ℚ)
    (weight_after_gain : ℕ) :
    initial_weight = 120 →
    muscle_gain_perc = 20 →
    fat_gain_fraction = 1/4 →
    weight_after_gain = 150 :=
by
  intros
  sorry

end James_weight_after_gain_l289_289312


namespace balls_to_boxes_distribution_l289_289544

theorem balls_to_boxes_distribution :
  (StirlingS2 5 1) + (StirlingS2 5 2) + (StirlingS2 5 3) = 41 :=
by
  sorry

end balls_to_boxes_distribution_l289_289544


namespace rectangle_perimeter_l289_289966

theorem rectangle_perimeter (a b : ℚ) (ha : ¬ a.den = 1) (hb : ¬ b.den = 1) (hab : a ≠ b) (h : (a - 2) * (b - 2) = -7) : 2 * (a + b) = 20 :=
by
  sorry

end rectangle_perimeter_l289_289966


namespace jenny_ate_65_chocolates_l289_289146

-- Define the number of chocolate squares Mike ate
def MikeChoc := 20

-- Define the function that calculates the chocolates Jenny ate
def JennyChoc (mikeChoc : ℕ) := 3 * mikeChoc + 5

-- The theorem stating the solution
theorem jenny_ate_65_chocolates (h : MikeChoc = 20) : JennyChoc MikeChoc = 65 := by
  -- Automatic proof step
  sorry

end jenny_ate_65_chocolates_l289_289146


namespace distance_between_parallel_lines_l289_289349

theorem distance_between_parallel_lines (a d : ℝ) (d_pos : 0 ≤ d) (a_pos : 0 ≤ a) :
  {d_ | d_ = d + a ∨ d_ = |d - a|} = {d + a, abs (d - a)} :=
by
  sorry

end distance_between_parallel_lines_l289_289349


namespace range_of_a_l289_289055

noncomputable def A (x : ℝ) : Prop := x < -2 ∨ x ≥ 1
noncomputable def B (x : ℝ) (a : ℝ) : Prop := x ≥ a

theorem range_of_a (a : ℝ) : (∀ x, A x ∨ B x a) ↔ a ≤ -2 :=
by sorry

end range_of_a_l289_289055


namespace nearest_integer_x_sub_y_l289_289153

theorem nearest_integer_x_sub_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : |x| - y = 4) 
  (h2 : |x| * y - x^3 = 1) : 
  abs (x - y - 4) < 1 :=
sorry

end nearest_integer_x_sub_y_l289_289153


namespace cos_double_angle_of_tangent_is_2_l289_289062

theorem cos_double_angle_of_tangent_is_2
  (θ : ℝ)
  (h_tan : Real.tan θ = 2) :
  Real.cos (2 * θ) = -3 / 5 := 
by
  sorry

end cos_double_angle_of_tangent_is_2_l289_289062


namespace frank_fencemaker_fence_length_l289_289212

theorem frank_fencemaker_fence_length :
  ∃ (L W : ℕ), W = 40 ∧
               (L * W = 200) ∧
               (2 * L + W = 50) :=
by
  sorry

end frank_fencemaker_fence_length_l289_289212


namespace find_angle_B_l289_289436

theorem find_angle_B (a b c : ℝ) (h : a^2 + c^2 - b^2 = a * c) : 
  ∃ B : ℝ, 0 < B ∧ B < 180 ∧ B = 60 :=
by 
  sorry

end find_angle_B_l289_289436


namespace find_xyz_l289_289433

variable (x y z : ℝ)

theorem find_xyz (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * (y + z) = 168)
  (h2 : y * (z + x) = 180)
  (h3 : z * (x + y) = 192) : x * y * z = 842 :=
sorry

end find_xyz_l289_289433


namespace pencil_total_length_l289_289699

-- Definitions of the colored sections
def purple_length : ℝ := 3.5
def black_length : ℝ := 2.8
def blue_length : ℝ := 1.6
def green_length : ℝ := 0.9
def yellow_length : ℝ := 1.2

-- The theorem stating the total length of the pencil
theorem pencil_total_length : purple_length + black_length + blue_length + green_length + yellow_length = 10 := 
by
  sorry

end pencil_total_length_l289_289699


namespace sin_of_angle_F_l289_289141

theorem sin_of_angle_F 
  (DE EF DF : ℝ) 
  (h : DE = 12) 
  (h0 : EF = 20) 
  (h1 : DF = Real.sqrt (DE^2 + EF^2)) : 
  Real.sin (Real.arctan (DF / EF)) = 12 / Real.sqrt (DE^2 + EF^2) := 
by 
  sorry

end sin_of_angle_F_l289_289141


namespace quadratic_real_roots_l289_289114

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := 
by 
  sorry

end quadratic_real_roots_l289_289114


namespace solve_system_l289_289753

theorem solve_system : ∃ (x y : ℝ), x + 3 * y = 7 ∧ y = 2 * x ∧ x = 1 ∧ y = 2 :=
by {
  use 1,
  use 2,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { refl },
  { refl },
}

end solve_system_l289_289753


namespace sum_of_squares_l289_289987

theorem sum_of_squares (a b : ℝ) (h1 : a + b = 16) (h2 : a * b = 20) : a^2 + b^2 = 216 :=
by
  sorry

end sum_of_squares_l289_289987


namespace quadratic_roots_interval_l289_289092

theorem quadratic_roots_interval (b : ℝ) :
  ∃ (x : ℝ), x^2 + b * x + 25 = 0 → b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_roots_interval_l289_289092


namespace roof_ratio_l289_289752

theorem roof_ratio (L W : ℝ) (h1 : L * W = 576) (h2 : L - W = 36) : L / W = 4 := 
by
  sorry

end roof_ratio_l289_289752


namespace workshops_participation_l289_289189

variable (x y z a b c d : ℕ)
variable (A B C : Finset ℕ)

theorem workshops_participation:
  (A.card = 15) →
  (B.card = 14) →
  (C.card = 11) →
  (25 = x + y + z + a + b + c + d) →
  (12 = a + b + c + d) →
  (A.card = x + a + c + d) →
  (B.card = y + a + b + d) →
  (C.card = z + b + c + d) →
  d = 0 :=
by
  intro hA hB hC hTotal hAtLeastTwo hAkA hBkA hCkA
  -- The proof will go here
  -- Parsing these inputs shall lead to establishing d = 0
  sorry

end workshops_participation_l289_289189


namespace cos_alpha_beta_l289_289048

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x * cos x * (sin x) ^ 2 - (1 / 2)

theorem cos_alpha_beta :
  ∀ (α β : ℝ), 
    (0 < α ∧ α < π / 2) →
    (0 < β ∧ β < π / 2) →
    f (α / 2) = sqrt 5 / 5 →
    f (β / 2) = 3 * sqrt 10 / 10 →
    cos (α - β) = sqrt 2 / 2 :=
by
  intros α β hα hβ h1 h2
  sorry

end cos_alpha_beta_l289_289048


namespace expectation_equality_variance_inequality_l289_289489

noncomputable def X1_expectation : ℚ :=
  2 * (2 / 5 : ℚ)

noncomputable def X1_variance : ℚ :=
  2 * (2 / 5) * (1 - 2 / 5)

noncomputable def P_X2_0 : ℚ :=
  (3 * 2) / (5 * 4)

noncomputable def P_X2_1 : ℚ :=
  (2 * 3) / (5 * 4)

noncomputable def P_X2_2 : ℚ :=
  (2 * 1) / (5 * 4)

noncomputable def X2_expectation : ℚ :=
  0 * P_X2_0 + 1 * P_X2_1 + 2 * P_X2_2

noncomputable def X2_variance : ℚ :=
  P_X2_0 * (0 - X2_expectation)^2 + P_X2_1 * (1 - X2_expectation)^2 + P_X2_2 * (2 - X2_expectation)^2

theorem expectation_equality : X1_expectation = X2_expectation :=
  by sorry

theorem variance_inequality : X1_variance > X2_variance :=
  by sorry

end expectation_equality_variance_inequality_l289_289489


namespace existence_of_indices_l289_289320

theorem existence_of_indices 
  (a1 a2 a3 a4 a5 : ℝ) 
  (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) (h4 : 0 < a4) (h5 : 0 < a5) : 
  ∃ (i j k l : Fin 5), 
    (i ≠ j) ∧ (i ≠ k) ∧ (i ≠ l) ∧ (j ≠ k) ∧ (j ≠ l) ∧ (k ≠ l) ∧ 
    |(a1 / a2) - (a3 / a4)| < 1/2 :=
by 
  sorry

end existence_of_indices_l289_289320


namespace quadratic_has_real_root_iff_b_in_interval_l289_289098

theorem quadratic_has_real_root_iff_b_in_interval (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ Set.Iic (-10) ∪ Set.Ici 10) := 
sorry

end quadratic_has_real_root_iff_b_in_interval_l289_289098


namespace ex_sq_sum_l289_289074

theorem ex_sq_sum (x y : ℝ) (h1 : (x + y)^2 = 9) (h2 : x * y = -1) : x^2 + y^2 = 11 :=
by
  sorry

end ex_sq_sum_l289_289074


namespace garden_area_eq_450_l289_289959

theorem garden_area_eq_450
  (width length : ℝ)
  (fencing : ℝ := 60) 
  (length_eq_twice_width : length = 2 * width)
  (fencing_eq : 2 * width + length = fencing) :
  width * length = 450 := by
  sorry

end garden_area_eq_450_l289_289959


namespace parabola_directrix_l289_289184

theorem parabola_directrix (y x : ℝ) : y^2 = -8 * x → x = -1 :=
by
  sorry

end parabola_directrix_l289_289184


namespace nonzero_fraction_power_zero_l289_289355

theorem nonzero_fraction_power_zero (a b : ℤ) (h : a ≠ 0 ∧ b ≠ 0) : ((a : ℚ) / b)^0 = 1 := 
by
  -- proof goes here
  sorry

end nonzero_fraction_power_zero_l289_289355


namespace simplify_exponent_multiplication_l289_289920

theorem simplify_exponent_multiplication (x : ℝ) : x^5 * x^3 = x^8 :=
by sorry

end simplify_exponent_multiplication_l289_289920


namespace b_share_is_approx_1885_71_l289_289209

noncomputable def investment_problem (x : ℝ) : ℝ := 
  let c_investment := x
  let b_investment := (2 / 3) * c_investment
  let a_investment := 3 * b_investment
  let total_investment := a_investment + b_investment + c_investment
  let b_share := (b_investment / total_investment) * 6600
  b_share

theorem b_share_is_approx_1885_71 (x : ℝ) : abs (investment_problem x - 1885.71) < 0.01 := sorry

end b_share_is_approx_1885_71_l289_289209


namespace no_integer_triplets_for_equation_l289_289678

theorem no_integer_triplets_for_equation (a b c : ℤ) : ¬ (a^2 + b^2 + 1 = 4 * c) :=
by
  sorry

end no_integer_triplets_for_equation_l289_289678


namespace num_ways_to_write_360_as_increasing_seq_l289_289554

def is_consecutive_sum (n k : ℕ) : Prop :=
  let seq_sum := k * n + k * (k - 1) / 2
  seq_sum = 360

def valid_k (k : ℕ) : Prop :=
  k ≥ 2 ∧ k ∣ 360 ∧ (k = 2 ∨ (k - 1) % 2 = 0)

noncomputable def count_consecutive_sums : ℕ :=
  Nat.card {k // valid_k k ∧ ∃ n : ℕ, is_consecutive_sum n k}

theorem num_ways_to_write_360_as_increasing_seq : count_consecutive_sums = 4 :=
sorry

end num_ways_to_write_360_as_increasing_seq_l289_289554


namespace whole_process_time_is_6_hours_l289_289576

def folding_time_per_fold : ℕ := 5
def number_of_folds : ℕ := 4
def resting_time_per_rest : ℕ := 75
def number_of_rests : ℕ := 4
def mixing_time : ℕ := 10
def baking_time : ℕ := 30

def total_time_process_in_minutes : ℕ :=
  mixing_time + 
  (folding_time_per_fold * number_of_folds) + 
  (resting_time_per_rest * number_of_rests) + 
  baking_time

def total_time_process_in_hours : ℕ := total_time_process_in_minutes / 60

theorem whole_process_time_is_6_hours :
  total_time_process_in_hours = 6 :=
by sorry

end whole_process_time_is_6_hours_l289_289576


namespace fishers_tomorrow_l289_289866

-- Define the groups of fishers and their fishing pattern
def everyday_fishers := 7
def every_other_day_fishers := 8
def every_three_days_fishers := 3

-- Given counts for yesterday and today
def fishers_yesterday := 12
def fishers_today := 10

-- The problem to prove: 15 people will fish tomorrow
theorem fishers_tomorrow : 
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (everyday_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  15 = everyday_fishers + every_other_day_fishers / 2 + every_three_days_fishers / 3 :=
begin
  sorry
end

end fishers_tomorrow_l289_289866


namespace solve_trig_eq_l289_289171

open Real

theorem solve_trig_eq (x : ℝ) (m k : ℤ) : 
  (∀ k : ℤ, x ≠ (π / 2) * k) → 
  (sin (3 * x) ^ 2 / sin x ^ 2 = 8 * cos (4 * x) + cos (3 * x) ^ 2 / cos x ^ 2) ↔ 
  ∃ m : ℤ, x = (π / 3) * (2 * m + 1) :=
by
  sorry

end solve_trig_eq_l289_289171


namespace members_with_both_non_athletic_parents_l289_289552

-- Let's define the conditions
variable (total_members athletic_dads athletic_moms both_athletic none_have_dads : ℕ)
variable (H1 : total_members = 50)
variable (H2 : athletic_dads = 25)
variable (H3 : athletic_moms = 30)
variable (H4 : both_athletic = 10)
variable (H5 : none_have_dads = 5)

-- Define the conclusion we want to prove
theorem members_with_both_non_athletic_parents : 
  (total_members - (athletic_dads + athletic_moms - both_athletic) + none_have_dads - total_members) = 10 :=
sorry

end members_with_both_non_athletic_parents_l289_289552


namespace figure_perimeter_l289_289197

-- Define the side length of the square and the triangles.
def square_side_length : ℕ := 3
def triangle_side_length : ℕ := 2

-- Calculate the perimeter of the figure
def perimeter (a b : ℕ) : ℕ := 2 * a + 2 * b

-- Statement to prove
theorem figure_perimeter : perimeter square_side_length triangle_side_length = 10 := 
by 
  -- "sorry" denotes that the proof is omitted.
  sorry

end figure_perimeter_l289_289197


namespace farmer_initial_plan_days_l289_289229

def initialDaysPlan
    (daily_hectares : ℕ)
    (increased_productivity : ℕ)
    (hectares_ploughed_first_two_days : ℕ)
    (hectares_remaining : ℕ)
    (days_ahead_schedule : ℕ)
    (total_hectares : ℕ)
    (days_actual : ℕ) : ℕ :=
  days_actual + days_ahead_schedule

theorem farmer_initial_plan_days : 
  ∀ (x days_ahead_schedule : ℕ) 
    (daily_hectares hectares_ploughed_first_two_days increased_productivity hectares_remaining total_hectares days_actual : ℕ),
  daily_hectares = 120 →
  increased_productivity = daily_hectares + daily_hectares / 4 →
  hectares_ploughed_first_two_days = 2 * daily_hectares →
  total_hectares = 1440 →
  days_ahead_schedule = 2 →
  days_actual = 10 →
  hectares_remaining = total_hectares - hectares_ploughed_first_two_days →
  hectares_remaining = increased_productivity * (days_actual - 2) →
  x = 12 :=
by
  intros x days_ahead_schedule daily_hectares hectares_ploughed_first_two_days increased_productivity hectares_remaining total_hectares days_actual
  intros h_daily_hectares h_increased_productivity h_hectares_ploughed_first_two_days h_total_hectares h_days_ahead_schedule h_days_actual h_hectares_remaining h_hectares_ploughed
  sorry

end farmer_initial_plan_days_l289_289229


namespace minimal_distance_ln_x_x_minimal_distance_graphs_ex_ln_x_l289_289625

theorem minimal_distance_ln_x_x :
  ∀ (x : ℝ), x > 0 → ∃ (d : ℝ), d = |Real.log x - x| → d ≥ 0 :=
by sorry

theorem minimal_distance_graphs_ex_ln_x :
  ∀ (x : ℝ), x > 0 → ∀ (y : ℝ), ∃ (d : ℝ), y = d → d = 2 :=
by sorry

end minimal_distance_ln_x_x_minimal_distance_graphs_ex_ln_x_l289_289625


namespace wrong_guess_is_20_l289_289463

-- Define the colors
inductive Color
| white
| brown
| black

-- Assume we have a sequence of 1000 bears
def bears : fin 1000 → Color := sorry

-- Hypotheses
axiom colors_per_three : ∀ (i : fin 998), 
  ({bears i, bears (i + 1), bears (i + 2)} = {Color.white, Color.brown, Color.black} ∨ 
   {bears i, bears (i + 1), bears (i + 2)} = {Color.black, Color.white, Color.brown} ∨ 
   {bears i, bears (i + 1), bears (i + 2)} = {Color.brown, Color.black, Color.white})

axiom exactly_one_wrong : 
  (bears 1 = Color.white ∧ bears 19 ≠ Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 ≠ Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 ≠ Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 ≠ Color.brown ∧ bears 799 = Color.white) ∨
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 ≠ Color.white)

-- Define the theorem to prove
theorem wrong_guess_is_20 : 
  (bears 1 = Color.white ∧ bears 19 = Color.brown ∧ bears 399 = Color.black ∧ bears 599 = Color.brown ∧ bears 799 = Color.white) →
  ¬(bears 19 = Color.brown) := 
sorry

end wrong_guess_is_20_l289_289463


namespace chen_recording_l289_289632

variable (standard xia_steps chen_steps : ℕ)
variable (xia_record : ℤ)

-- Conditions: 
-- standard = 5000
-- Xia walked 6200 steps, recorded as +1200 steps
def met_standard (s : ℕ) : Prop :=
  s >= 5000

def xia_condition := (xia_steps = 6200) ∧ (xia_record = 1200) ∧ (xia_record = (xia_steps : ℤ) - 5000)

-- Question and solution combined into a statement: 
-- Chen walked 4800 steps, recorded as -200 steps
def chen_condition := (chen_steps = 4800) ∧ (met_standard chen_steps = false) → (((standard : ℤ) - chen_steps) * -1 = -200)

-- Proof goal:
theorem chen_recording (h₁ : standard = 5000) (h₂ : xia_condition xia_steps xia_record):
  chen_condition standard chen_steps :=
by
  sorry

end chen_recording_l289_289632


namespace smallest_number_increased_by_3_divisible_l289_289198

theorem smallest_number_increased_by_3_divisible (n : ℤ) 
    (h1 : (n + 3) % 18 = 0)
    (h2 : (n + 3) % 70 = 0)
    (h3 : (n + 3) % 25 = 0)
    (h4 : (n + 3) % 21 = 0) : 
    n = 3147 :=
by
  sorry

end smallest_number_increased_by_3_divisible_l289_289198


namespace fishers_tomorrow_l289_289865

-- Define the groups of fishers and their fishing pattern
def everyday_fishers := 7
def every_other_day_fishers := 8
def every_three_days_fishers := 3

-- Given counts for yesterday and today
def fishers_yesterday := 12
def fishers_today := 10

-- The problem to prove: 15 people will fish tomorrow
theorem fishers_tomorrow : 
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (everyday_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  15 = everyday_fishers + every_other_day_fishers / 2 + every_three_days_fishers / 3 :=
begin
  sorry
end

end fishers_tomorrow_l289_289865


namespace cookies_leftover_l289_289657

def amelia_cookies := 52
def benjamin_cookies := 63
def chloe_cookies := 25
def total_cookies := amelia_cookies + benjamin_cookies + chloe_cookies
def package_size := 15

theorem cookies_leftover :
  total_cookies % package_size = 5 := by
  sorry

end cookies_leftover_l289_289657


namespace proof_problem_l289_289915

variables {AO' AO_1 AB AC t s s_1 s_2 s_3 : ℝ}
variables {alpha : ℝ}

-- Conditions
def condition1 : Prop := AO' * Real.sin (alpha / 2) = t / s
def condition2 : Prop := AO_1 * Real.sin (alpha / 2) = t / s_1
def condition3 : Prop := AO' * AO_1 = t^2 / (s * s_1 * (Real.sin (alpha / 2))^2)
def condition4 : Prop := (Real.sin (alpha / 2))^2 = (s_2 * s_3) / (AB * AC)

-- Statement to prove
theorem proof_problem (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : 
  AO' * AO_1 = AB * AC :=
by
  sorry

end proof_problem_l289_289915


namespace express_y_in_terms_of_y_l289_289042

variable (x : ℝ)

theorem express_y_in_terms_of_y (y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 :=
sorry

end express_y_in_terms_of_y_l289_289042


namespace number_of_handshakes_l289_289662

-- Define the context of the problem
def total_women := 8
def teams (n : Nat) := 4

-- Define the number of people each woman will shake hands with (excluding her partner)
def handshakes_per_woman := total_women - 2

-- Define the total number of handshakes
def total_handshakes := (total_women * handshakes_per_woman) / 2

-- The theorem that we're to prove
theorem number_of_handshakes : total_handshakes = 24 :=
by
  sorry

end number_of_handshakes_l289_289662


namespace integer_pairs_sum_product_l289_289261

theorem integer_pairs_sum_product (x y : ℤ) (h : x + y = x * y) : (x = 2 ∧ y = 2) ∨ (x = 0 ∧ y = 0) :=
by
  sorry

end integer_pairs_sum_product_l289_289261


namespace probability_of_color_change_l289_289239

theorem probability_of_color_change :
  let cycle_duration := 100
  let green_duration := 45
  let yellow_duration := 5
  let red_duration := 50
  let green_to_yellow_interval := 5
  let yellow_to_red_interval := 5
  let red_to_green_interval := 5
  let total_color_change_duration := green_to_yellow_interval + yellow_to_red_interval + red_to_green_interval
  let observation_probability := total_color_change_duration / cycle_duration
  observation_probability = 3 / 20 := by sorry

end probability_of_color_change_l289_289239


namespace time_spent_on_road_l289_289359

theorem time_spent_on_road (Total_time_hours Stop1_minutes Stop2_minutes Stop3_minutes : ℕ) 
  (h1: Total_time_hours = 13) 
  (h2: Stop1_minutes = 25) 
  (h3: Stop2_minutes = 10) 
  (h4: Stop3_minutes = 25) : 
  Total_time_hours - (Stop1_minutes + Stop2_minutes + Stop3_minutes) / 60 = 12 :=
by
  sorry

end time_spent_on_road_l289_289359


namespace ab_cd_value_l289_289294

theorem ab_cd_value (a b c d: ℝ)
  (h1 : a + b + c = 1)
  (h2 : a + b + d = 5)
  (h3 : a + c + d = 14)
  (h4 : b + c + d = 9) :
  a * b + c * d = 338 / 9 := 
sorry

end ab_cd_value_l289_289294


namespace points_above_y_eq_x_l289_289772

theorem points_above_y_eq_x (x y : ℝ) : (y > x) → (y, x) ∈ {p : ℝ × ℝ | p.2 < p.1} :=
by
  intro h
  sorry

end points_above_y_eq_x_l289_289772


namespace find_b_values_l289_289084

noncomputable def has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + b*x + 25 = 0

theorem find_b_values (b : ℝ) : has_real_root b ↔ b ∈ set.Iic (-10) ∪ set.Ici 10 := by
  sorry

end find_b_values_l289_289084


namespace range_of_a_l289_289997

noncomputable def A : Set ℝ := {x | x ≥ abs (x^2 - 2 * x)}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + a ≤ 0}

theorem range_of_a (a : ℝ) : (A ∩ B a = B a) → (0 ≤ a ∧ a ≤ 1) :=
by
  sorry

end range_of_a_l289_289997


namespace competition_votes_l289_289551

/-- 
In a revival competition, if B's number of votes is 20/21 of A's, and B wins by
gaining at least 4 votes more than A, prove the possible valid votes counts.
-/
theorem competition_votes (x : ℕ) 
  (hx : x > 0) 
  (hx_mod_21 : x % 21 = 0) 
  (hB_wins : ∀ b : ℕ, b = (20 * x / 21) + 4 → b > x - 4) :
  (x = 147 ∧ 140 = 20 * x / 21) ∨ (x = 126 ∧ 120 = 20 * x / 21) := 
by 
  sorry

end competition_votes_l289_289551


namespace Evan_earnings_Markese_less_than_Evan_l289_289572

-- Definitions from conditions
def MarkeseEarnings : ℕ := 16
def TotalEarnings : ℕ := 37

-- Theorem statements
theorem Evan_earnings (E : ℕ) (h : E + MarkeseEarnings = TotalEarnings) : E = 21 :=
by {
  sorry
}

theorem Markese_less_than_Evan (E : ℕ) (h : E + MarkeseEarnings = TotalEarnings) : E - MarkeseEarnings = 5 :=
by {
  sorry
}

end Evan_earnings_Markese_less_than_Evan_l289_289572


namespace range_of_s_l289_289530

def is_composite (n : ℕ) : Prop := (n > 1) ∧ ∃ p k : ℕ, p.prime ∧ k > 1 ∧ n = p^k

def s (n : ℕ) : ℕ :=
  if hn : 2 ≤ n ∧ ¬ n.prime then
    let prime_factors : List (ℕ × ℕ) :=
      (UniqueFactorizationMonoid.normalizedFactors n).groupBy id in
    prime_factors.foldl (λ acc (p, k), acc + k * p^2) 0
  else
    0

theorem range_of_s :
  ∀ n : ℕ, is_composite n → 12 ≤ s n ∧ ∀ m : ℕ, m > 11 → ∃ n, is_composite n ∧ s n = m :=
by
  sorry

end range_of_s_l289_289530


namespace question1_question2_l289_289248

theorem question1 :
  (1:ℝ) * (Real.sqrt 12 + Real.sqrt 20) + (Real.sqrt 3 - Real.sqrt 5) = 3 * Real.sqrt 3 + Real.sqrt 5 := 
by sorry

theorem question2 :
  (4 * Real.sqrt 2 - 3 * Real.sqrt 6) / (2 * Real.sqrt 2) - (Real.sqrt 8 + Real.pi)^0 = 1 - 3 * Real.sqrt 3 / 2 :=
by sorry

end question1_question2_l289_289248


namespace only_one_P_Q_l289_289071

def P (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def Q (a : ℝ) : Prop := ∀ x : ℝ, x^2 - x + a = 0

theorem only_one_P_Q (a : ℝ) :
  (P a ∧ ¬ Q a) ∨ (Q a ∧ ¬ P a) ↔
  (a < 0) ∨ (1/4 < a ∧ a < 4) :=
sorry

end only_one_P_Q_l289_289071


namespace find_n_l289_289524

theorem find_n (n : ℤ) (h₀ : 0 ≤ n) (h₁ : n ≤ 9) : n ≡ -2023 [MOD 10] → n = 7 :=
by
  sorry

end find_n_l289_289524


namespace people_believing_mostly_purple_l289_289780

theorem people_believing_mostly_purple :
  ∀ (total : ℕ) (mostly_pink : ℕ) (both_mostly_pink_purple : ℕ) (neither : ℕ),
  total = 150 →
  mostly_pink = 80 →
  both_mostly_pink_purple = 40 →
  neither = 25 →
  (total - neither + both_mostly_pink_purple - mostly_pink) = 85 :=
by
  intros total mostly_pink both_mostly_pink_purple neither h_total h_mostly_pink h_both h_neither
  have people_identified_without_mostly_purple : ℕ := mostly_pink + both_mostly_pink_purple - mostly_pink + neither
  have leftover_people : ℕ := total - people_identified_without_mostly_purple
  have people_mostly_purple := both_mostly_pink_purple + leftover_people
  suffices people_mostly_purple = 85 by sorry
  sorry

end people_believing_mostly_purple_l289_289780


namespace joggers_meetings_l289_289164

theorem joggers_meetings (road_length : ℝ)
  (speed_A speed_B : ℝ)
  (start_time : ℝ)
  (meeting_time : ℝ) :
  road_length = 400 → 
  speed_A = 3 → 
  speed_B = 2.5 →
  start_time = 0 → 
  meeting_time = 1200 → 
  ∃ y : ℕ, y = 8 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end joggers_meetings_l289_289164


namespace sum_of_squares_l289_289988

theorem sum_of_squares (a b : ℝ) (h1 : a + b = 16) (h2 : a * b = 20) : a^2 + b^2 = 216 :=
by
  sorry

end sum_of_squares_l289_289988


namespace amount_of_flour_per_large_tart_l289_289324

-- Statement without proof
theorem amount_of_flour_per_large_tart 
  (num_small_tarts : ℕ) (flour_per_small_tart : ℚ) 
  (num_large_tarts : ℕ) (total_flour : ℚ) 
  (h1 : num_small_tarts = 50) 
  (h2 : flour_per_small_tart = 1/8) 
  (h3 : num_large_tarts = 25) 
  (h4 : total_flour = num_small_tarts * flour_per_small_tart) : 
  total_flour = num_large_tarts * (1/4) := 
sorry

end amount_of_flour_per_large_tart_l289_289324


namespace sum_and_gap_l289_289850

-- Define the gap condition
def gap_condition (x : ℝ) : Prop :=
  |5.46 - x| = 3.97

-- Define the main theorem to be proved 
theorem sum_and_gap :
  ∀ (x : ℝ), gap_condition x → x < 5.46 → x + 5.46 = 6.95 := 
by 
  intros x hx hlt
  sorry

end sum_and_gap_l289_289850


namespace gcd_lcm_product_l289_289664

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 30) (h2 : b = 75) :
  (Nat.gcd a b) * (Nat.lcm a b) = 2250 := by
  sorry

end gcd_lcm_product_l289_289664


namespace max_profundity_eq_Fib_l289_289143

open Nat

-- Definitions as per conditions
def profundity (word : List Char) : Nat :=
  (word.sublists.map List.length).length
  
def dog_dictionary (n : Nat) : Set (List Char) :=
  { word | word.length = n ∧ word.all (fun c => c = 'A' ∨ c = 'U') }

def max_profundity (n : Nat) : Nat :=
  Sup (profundity '' (dog_dictionary n))

-- Theorem statement
theorem max_profundity_eq_Fib (n : Nat) : max_profundity n = Fibonacci (n + 3) - 3 := 
  sorry

end max_profundity_eq_Fib_l289_289143


namespace sum_of_squares_l289_289986

theorem sum_of_squares (a b : ℝ) (h1 : (a + b) / 2 = 8) (h2 : Real.sqrt (a * b) = 2 * Real.sqrt 5) :
  a^2 + b^2 = 216 :=
by
  sorry

end sum_of_squares_l289_289986


namespace complement_union_l289_289287

variable (U : Set ℤ) (A : Set ℤ) (B : Set ℤ)

theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3})
                         (hA : A = {-1, 0, 1})
                         (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} := 
by 
  -- Proof is omitted
  sorry

end complement_union_l289_289287


namespace cadastral_value_of_land_l289_289511

theorem cadastral_value_of_land (tax_amount_paid : ℝ) (tax_rate : ℝ) (V : ℝ) :
  (tax_amount_paid = 4500) → (tax_rate = 0.003) → (V = tax_amount_paid / tax_rate) → (V = 1500000) :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  have hV : V = 4500 / 0.003 := h3
  have hV_value : 4500 / 0.003 = 1500000 := 
    by norm_num
  rw hV_value at hV
  exact hV

#eval cadastral_value_of_land 4500 0.003 1500000 sorry -- Testing purpose to ensure it compiles.

end cadastral_value_of_land_l289_289511


namespace farmer_land_l289_289214

-- Define A to be the total land owned by the farmer
variables (A : ℝ)

-- Define the conditions of the problem
def condition_1 (A : ℝ) : ℝ := 0.90 * A
def condition_2 (cleared_land : ℝ) : ℝ := 0.20 * cleared_land
def condition_3 (cleared_land : ℝ) : ℝ := 0.70 * cleared_land
def condition_4 (cleared_land : ℝ) : ℝ := cleared_land - condition_2 cleared_land - condition_3 cleared_land

-- Define the assertion we need to prove
theorem farmer_land (h : condition_4 (condition_1 A) = 630) : A = 7000 :=
by
  sorry

end farmer_land_l289_289214


namespace abs_eq_solution_diff_l289_289253

theorem abs_eq_solution_diff : 
  ∀ x₁ x₂ : ℝ, 
  (2 * x₁ - 3 = 18 ∨ 2 * x₁ - 3 = -18) → 
  (2 * x₂ - 3 = 18 ∨ 2 * x₂ - 3 = -18) → 
  |x₁ - x₂| = 18 :=
by
  sorry

end abs_eq_solution_diff_l289_289253


namespace triangle_area_l289_289406

def point := ℝ × ℝ

def A : point := (-3, 3)
def B : point := (5, -1)
def C : point := (13, 6)

def vec (p1 p2 : point) : point := (p2.1 - p1.1, p2.2 - p1.2)

def area_of_parallelogram (v w : point) : ℝ :=
  real.abs (v.1 * w.2 - v.2 * w.1)

def area_of_triangle (A B C : point) : ℝ :=
  area_of_parallelogram (vec C A) (vec C B) / 2

theorem triangle_area :
  area_of_triangle A B C = 44 :=
sorry

end triangle_area_l289_289406


namespace problem_A_minus_B_no_x_or_x2_implies_a2_plus_b2_eq_13_l289_289267

variable (x y a b : ℝ)

def A : ℝ := 2*x^2 + a*x - y + 6
def B : ℝ := b*x^2 - 3*x + 5*y - 1

theorem problem_A_minus_B_no_x_or_x2_implies_a2_plus_b2_eq_13 
  (h : A x y a - B x y b = -6*y + 7) : a^2 + b^2 = 13 := by
  sorry

end problem_A_minus_B_no_x_or_x2_implies_a2_plus_b2_eq_13_l289_289267


namespace smallest_positive_perfect_square_divisible_by_2_and_5_l289_289200

theorem smallest_positive_perfect_square_divisible_by_2_and_5 :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, n = k^2) ∧ (2 ∣ n) ∧ (5 ∣ n) ∧ n = 100 :=
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_and_5_l289_289200


namespace find_y_payment_l289_289778

-- Definitions for the conditions in the problem
def total_payment (X Y : ℝ) : Prop := X + Y = 560
def x_is_120_percent_of_y (X Y : ℝ) : Prop := X = 1.2 * Y

-- Problem statement converted to a Lean proof problem
theorem find_y_payment (X Y : ℝ) (h1 : total_payment X Y) (h2 : x_is_120_percent_of_y X Y) : Y = 255 := 
by sorry

end find_y_payment_l289_289778


namespace proof_problem_l289_289272

variable {x y : ℝ}

def conditions (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + y = 1

theorem proof_problem (h : conditions x y) :
  x + y - 4 * x * y ≥ 0 ∧ (1 / x) + 4 / (1 + y) ≥ 9 / 2 :=
by
  sorry

end proof_problem_l289_289272


namespace chen_recording_l289_289633

variable (standard xia_steps chen_steps : ℕ)
variable (xia_record : ℤ)

-- Conditions: 
-- standard = 5000
-- Xia walked 6200 steps, recorded as +1200 steps
def met_standard (s : ℕ) : Prop :=
  s >= 5000

def xia_condition := (xia_steps = 6200) ∧ (xia_record = 1200) ∧ (xia_record = (xia_steps : ℤ) - 5000)

-- Question and solution combined into a statement: 
-- Chen walked 4800 steps, recorded as -200 steps
def chen_condition := (chen_steps = 4800) ∧ (met_standard chen_steps = false) → (((standard : ℤ) - chen_steps) * -1 = -200)

-- Proof goal:
theorem chen_recording (h₁ : standard = 5000) (h₂ : xia_condition xia_steps xia_record):
  chen_condition standard chen_steps :=
by
  sorry

end chen_recording_l289_289633


namespace eval_expression_l289_289039

theorem eval_expression (x y z : ℝ) (hx : x = 1/3) (hy : y = 2/3) (hz : z = -9) :
  x^2 * y^3 * z = -8/27 :=
by
  subst hx
  subst hy
  subst hz
  sorry

end eval_expression_l289_289039


namespace inequality_problem_l289_289720

theorem inequality_problem
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ (3 / 2) :=
sorry

end inequality_problem_l289_289720


namespace number_of_incorrect_statements_l289_289395

-- Conditions
def cond1 (p q : Prop) : Prop := (p ∨ q) → (p ∧ q)

def cond2 (x : ℝ) : Prop := x > 5 → x^2 - 4*x - 5 > 0

def cond3 : Prop := ∃ x0 : ℝ, x0^2 + x0 - 1 < 0

def cond3_neg : Prop := ∀ x : ℝ, x^2 + x - 1 ≥ 0

def cond4 (x : ℝ) : Prop := (x ≠ 1 ∨ x ≠ 2) → (x^2 - 3*x + 2 ≠ 0)

-- Proof problem
theorem number_of_incorrect_statements : 
  (¬ cond1 (p := true) (q := false)) ∧ (cond2 (x := 6)) ∧ (cond3 → cond3_neg) ∧ (¬ cond4 (x := 0)) → 
  2 = 2 :=
by
  sorry

end number_of_incorrect_statements_l289_289395


namespace quadratic_real_roots_l289_289108

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := 
by 
  sorry

end quadratic_real_roots_l289_289108


namespace mn_necessary_not_sufficient_l289_289054

variable (m n : ℝ)

def is_ellipse (m n : ℝ) : Prop := 
  (m > 0) ∧ (n > 0) ∧ (m ≠ n)

theorem mn_necessary_not_sufficient : (mn > 0) → (is_ellipse m n) ↔ false := 
by sorry

end mn_necessary_not_sufficient_l289_289054


namespace pebbles_game_invariant_l289_289747

/-- 
The game of pebbles is played on an infinite board of lattice points (i, j).
Initially, there is a pebble at (0, 0).
A move consists of removing a pebble from point (i, j) and placing a pebble at each of the points (i+1, j) and (i, j+1) provided both are vacant.
Show that at any stage of the game there is a pebble at some lattice point (a, b) with 0 ≤ a + b ≤ 3. 
-/
theorem pebbles_game_invariant :
  ∀ (board : ℕ × ℕ → Prop) (initial_state : board (0, 0)) (move : (ℕ × ℕ) → Prop → Prop → Prop),
  (∀ (i j : ℕ), board (i, j) → ¬ board (i+1, j) ∧ ¬ board (i, j+1) → board (i+1, j) ∧ board (i, j+1)) →
  ∃ (a b : ℕ), (0 ≤ a + b ∧ a + b ≤ 3) ∧ board (a, b) :=
by
  intros board initial_state move move_rule
  sorry 

end pebbles_game_invariant_l289_289747


namespace simplify_and_evaluate_expression_l289_289921

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = (Real.sqrt 2) + 1) : 
  (1 - (1 / a)) / ((a ^ 2 - 2 * a + 1) / a) = (Real.sqrt 2) / 2 :=
by
  sorry

end simplify_and_evaluate_expression_l289_289921


namespace melons_count_l289_289755

theorem melons_count (w_apples_total w_apple w_2apples w_watermelons w_total w_melons : ℕ) :
  w_apples_total = 4500 →
  9 * w_apple = w_apples_total →
  2 * w_apple = w_2apples →
  5 * 1050 = w_watermelons →
  w_total = w_2apples + w_melons →
  w_total = w_watermelons →
  w_melons / 850 = 5 :=
by
  sorry

end melons_count_l289_289755


namespace triangle_side_lengths_expression_neg_l289_289280

theorem triangle_side_lengths_expression_neg {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^4 + b^4 + c^4 - 2 * a^2 * b^2 - 2 * b^2 * c^2 - 2 * c^2 * a^2 < 0 := 
by 
  sorry

end triangle_side_lengths_expression_neg_l289_289280


namespace min_value_complex_mod_one_l289_289833

/-- Given that the modulus of the complex number \( z \) is 1, prove that the minimum value of
    \( |z - 4|^2 + |z + 3 * Complex.I|^2 \) is \( 17 \). -/
theorem min_value_complex_mod_one (z : ℂ) (h : ‖z‖ = 1) : 
  ∃ α : ℝ, (‖z - 4‖^2 + ‖z + 3 * Complex.I‖^2) = 17 :=
sorry

end min_value_complex_mod_one_l289_289833


namespace area_of_circumscribed_circle_l289_289643

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  let R := s / Real.sqrt 3 in 
  Real.pi * R^2 = 48 * Real.pi :=
by
  sorry

end area_of_circumscribed_circle_l289_289643


namespace employees_salaries_l289_289891

theorem employees_salaries (M N P : ℝ)
  (hM : M = 1.20 * N)
  (hN_median : N = N) -- Indicates N is the median
  (hP : P = 0.65 * M)
  (h_total : N + M + P = 3200) :
  M = 1288.58 ∧ N = 1073.82 ∧ P = 837.38 :=
by
  sorry

end employees_salaries_l289_289891


namespace temperature_lower_than_freezing_point_is_minus_three_l289_289613

-- Define the freezing point of water
def freezing_point := 0 -- in degrees Celsius

-- Define the temperature lower by a certain value
def lower_temperature (t: Int) (delta: Int) := t - delta

-- State the theorem to be proved
theorem temperature_lower_than_freezing_point_is_minus_three:
  lower_temperature freezing_point 3 = -3 := by
  sorry

end temperature_lower_than_freezing_point_is_minus_three_l289_289613


namespace total_tiles_in_room_l289_289500

theorem total_tiles_in_room (s : ℕ) (hs : 6 * s - 5 = 193) : s^2 = 1089 :=
by sorry

end total_tiles_in_room_l289_289500


namespace shorter_side_length_l289_289967

theorem shorter_side_length (a b : ℕ) (h1 : 2 * a + 2 * b = 50) (h2 : a * b = 126) : b = 9 :=
sorry

end shorter_side_length_l289_289967


namespace quadratic_real_roots_l289_289111

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := 
by 
  sorry

end quadratic_real_roots_l289_289111


namespace Teresa_age_when_Michiko_born_l289_289455

theorem Teresa_age_when_Michiko_born 
  (Teresa_age : ℕ) (Morio_age : ℕ) (Michiko_born_age : ℕ) (Kenji_diff : ℕ)
  (Emiko_diff : ℕ) (Hideki_same_as_Kenji : Prop) (Ryuji_age_same_as_Morio : Prop)
  (h1 : Teresa_age = 59) 
  (h2 : Morio_age = 71) 
  (h3 : Morio_age = Michiko_born_age + 33)
  (h4 : Kenji_diff = 4)
  (h5 : Emiko_diff = 10)
  (h6 : Hideki_same_as_Kenji = True)
  (h7 : Ryuji_age_same_as_Morio = True) : 
  ∃ Michiko_age Hideki_age Michiko_Hideki_diff Teresa_birth_age,
    Michiko_age = 33 ∧ 
    Hideki_age = 29 ∧ 
    Michiko_Hideki_diff = 4 ∧ 
    Teresa_birth_age = 26 :=
sorry

end Teresa_age_when_Michiko_born_l289_289455


namespace range_of_a_I_minimum_value_of_a_II_l289_289268

open Real

def f (x a : ℝ) : ℝ := abs (x - a)

theorem range_of_a_I (a : ℝ) :
  (∀ x, -1 ≤ x → x ≤ 3 → f x a ≤ 3) ↔ 0 ≤ a ∧ a ≤ 2 := sorry

theorem minimum_value_of_a_II :
  ∀ a : ℝ, (∀ x : ℝ, f (x - a) a + f (x + a) a ≥ 1 - 2 * a) ↔ a ≥ (1 / 4) :=
sorry

end range_of_a_I_minimum_value_of_a_II_l289_289268


namespace water_remaining_l289_289493

theorem water_remaining (initial_water : ℕ) (evap_rate : ℕ) (days : ℕ) : 
  initial_water = 500 → evap_rate = 1 → days = 50 → 
  initial_water - evap_rate * days = 450 :=
by
  intros h₁ h₂ h₃
  sorry

end water_remaining_l289_289493


namespace future_age_ratio_l289_289346

theorem future_age_ratio (j e x : ℕ) 
  (h1 : j - 3 = 5 * (e - 3)) 
  (h2 : j - 7 = 6 * (e - 7)) 
  (h3 : x = 17) : (j + x) / (e + x) = 3 := 
by
  sorry

end future_age_ratio_l289_289346


namespace probability_of_winning_pair_l289_289013

/-- Definition of the deck and conditions --/
def deck : Finset (Σ (c : Fin 3), Fin 4) :=
  { (0, 0), (0, 1), (0, 2), (0, 3), -- red cards
    (1, 0), (1, 1), (1, 2), (1, 3), -- green cards
    (2, 0), (2, 1), (2, 2), (2, 3)  -- blue cards
  }

/-- A pair of cards is winning if they have the same color or the same label --/
def winning_pair (a b : (Σ (c : Fin 3), Fin 4)) : Prop :=
  a.1 = b.1 ∨ a.2 = b.2

/-- The theorem to prove - The probability of drawing a winning pair is 5/11 --/
theorem probability_of_winning_pair :
  ((Finset.card ((deck.product deck).filter (λ pair, winning_pair pair.1 pair.2)) : ℚ) /
   (Finset.card (deck.product deck) : ℚ)) = 5 / 11 :=
by
  -- Outline logic goes here (proof skipped with sorry)
  sorry

end probability_of_winning_pair_l289_289013


namespace correct_proportion_l289_289138

theorem correct_proportion {a b c x y : ℝ} 
  (h1 : x + y = b)
  (h2 : x * c = y * a) :
  y / a = b / (a + c) :=
sorry

end correct_proportion_l289_289138


namespace star_of_15_star_eq_neg_15_l289_289049

def y_star (y : ℤ) : ℤ := 10 - y
def star_y (y : ℤ) : ℤ := y - 10

theorem star_of_15_star_eq_neg_15 : star_y (y_star 15) = -15 :=
by {
  -- applying given definitions;
  sorry
}

end star_of_15_star_eq_neg_15_l289_289049


namespace quadratic_root_shift_c_value_l289_289154

theorem quadratic_root_shift_c_value
  (r s : ℝ)
  (h1 : r + s = 2)
  (h2 : r * s = -5) :
  ∃ b : ℝ, x^2 + b * x - 2 = 0 :=
by
  sorry

end quadratic_root_shift_c_value_l289_289154


namespace total_cost_of_tires_and_battery_l289_289050

theorem total_cost_of_tires_and_battery :
  (4 * 42 + 56 = 224) := 
  by
    sorry

end total_cost_of_tires_and_battery_l289_289050


namespace max_difference_y_coords_l289_289525

noncomputable def maximumDifference : ℝ :=
  (4 * Real.sqrt 6) / 9

theorem max_difference_y_coords :
  let f1 (x : ℝ) := 3 - 2 * x^2 + x^3
  let f2 (x : ℝ) := 1 + x^2 + x^3
  let x1 := Real.sqrt (2/3)
  let x2 := - Real.sqrt (2/3)
  let y1 := f1 x1
  let y2 := f1 x2
  |y1 - y2| = maximumDifference := sorry

end max_difference_y_coords_l289_289525


namespace odd_exponent_divisibility_l289_289352

theorem odd_exponent_divisibility (x y : ℤ) (k : ℕ) (h : (x^(2*k-1) + y^(2*k-1)) % (x + y) = 0) : 
  (x^(2*k+1) + y^(2*k+1)) % (x + y) = 0 :=
sorry

end odd_exponent_divisibility_l289_289352


namespace g_at_pi_over_4_l289_289837

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := f x + 1

theorem g_at_pi_over_4 : g (Real.pi / 4) = 3 / 2 :=
by 
  sorry

end g_at_pi_over_4_l289_289837


namespace length_of_jordans_rectangle_l289_289975

theorem length_of_jordans_rectangle 
  (h1 : ∃ (length width : ℕ), length = 5 ∧ width = 24) 
  (h2 : ∃ (width_area : ℕ), width_area = 30 ∧ ∃ (area : ℕ), area = 5 * 24 ∧ ∃ (L : ℕ), area = L * width_area) :
  ∃ L, L = 4 := by 
  sorry

end length_of_jordans_rectangle_l289_289975


namespace find_value_divided_by_4_l289_289367

theorem find_value_divided_by_4 (x : ℝ) (h : 812 / x = 25) : x / 4 = 8.12 :=
by
  sorry

end find_value_divided_by_4_l289_289367


namespace parabola_vertex_y_axis_opens_upwards_l289_289618

theorem parabola_vertex_y_axis_opens_upwards :
  ∃ (a b c : ℝ), (a > 0) ∧ (b = 0) ∧ y = a * x^2 + b * x + c := 
sorry

end parabola_vertex_y_axis_opens_upwards_l289_289618


namespace sarah_initial_bake_l289_289594

theorem sarah_initial_bake (todd_ate : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) 
  (initial_cupcakes : ℕ)
  (h1 : todd_ate = 14)
  (h2 : packages = 3)
  (h3 : cupcakes_per_package = 8)
  (h4 : packages * cupcakes_per_package + todd_ate = initial_cupcakes) :
  initial_cupcakes = 38 :=
by sorry

end sarah_initial_bake_l289_289594


namespace correct_card_ordering_l289_289767

structure CardOrder where
  left : String
  middle : String
  right : String

def is_right_of (a b : String) : Prop := (a = "club" ∧ (b = "heart" ∨ b = "diamond")) ∨ (a = "8" ∧ b = "4")

def is_left_of (a b : String) : Prop := a = "5" ∧ b = "heart"

def correct_order : CardOrder :=
  { left := "5 of diamonds", middle := "4 of hearts", right := "8 of clubs" }

theorem correct_card_ordering : 
  ∀ order : CardOrder, 
  is_right_of order.right order.middle ∧ is_right_of order.right order.left ∧ is_left_of order.left order.middle 
  → order = correct_order := 
by
  intro order
  intro h
  sorry

end correct_card_ordering_l289_289767


namespace conditional_probability_heads_then_tails_twice_l289_289204

theorem conditional_probability_heads_then_tails_twice :
  let p_heads := 1/2
  let p_tails := 1/2 
  P(B | A) = 1/2 :=
by
  have p_A : P(A) = p_heads := sorry
  have p_B_given_A : P(B | A) = (P(A) * p_tails) / P(A) := sorry
  rw p_A at p_B_given_A
  norm_num at p_B_given_A
  exact p_B_given_A

end conditional_probability_heads_then_tails_twice_l289_289204


namespace find_t_l289_289037

theorem find_t (c o u n t s : ℕ)
    (hc : c ≠ 0) (ho : o ≠ 0) (hn : n ≠ 0) (ht : t ≠ 0) (hs : s ≠ 0)
    (h1 : c + o = u)
    (h2 : u + n = t + 1)
    (h3 : t + c = s)
    (h4 : o + n + s = 15) :
    t = 7 := 
sorry

end find_t_l289_289037


namespace find_a_f_odd_f_increasing_l289_289064

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x - a / x

theorem find_a : (f 1 a = 3) → (a = -1) :=
by
  sorry

noncomputable def f_1 (x : ℝ) : ℝ := 2 * x + 1 / x

theorem f_odd : ∀ x : ℝ, f_1 (-x) = -f_1 x :=
by
  sorry

theorem f_increasing : ∀ x1 x2 : ℝ, (x1 > 1) → (x2 > 1) → (x1 > x2) → (f_1 x1 > f_1 x2) :=
by
  sorry

end find_a_f_odd_f_increasing_l289_289064


namespace find_b_values_l289_289081

noncomputable def has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + b*x + 25 = 0

theorem find_b_values (b : ℝ) : has_real_root b ↔ b ∈ set.Iic (-10) ∪ set.Ici 10 := by
  sorry

end find_b_values_l289_289081


namespace total_price_of_order_l289_289957

-- Define the price of each item
def price_ice_cream_bar : ℝ := 0.60
def price_sundae : ℝ := 1.40

-- Define the quantity of each item
def quantity_ice_cream_bar : ℕ := 125
def quantity_sundae : ℕ := 125

-- Calculate the costs
def cost_ice_cream_bar := quantity_ice_cream_bar * price_ice_cream_bar
def cost_sundae := quantity_sundae * price_sundae

-- Calculate the total cost
def total_cost := cost_ice_cream_bar + cost_sundae

-- Statement of the theorem
theorem total_price_of_order : total_cost = 250 := 
by {
  sorry
}

end total_price_of_order_l289_289957


namespace proportional_segments_l289_289536

theorem proportional_segments (a b c d : ℝ) (h1 : a = 4) (h2 : b = 2) (h3 : c = 3) (h4 : a / b = c / d) : d = 3 / 2 :=
by
  -- proof steps here
  sorry

end proportional_segments_l289_289536


namespace substitution_result_l289_289173

theorem substitution_result (x y : ℝ) (h1 : y = 2 * x + 1) (h2 : 5 * x - 2 * y = 7) : 5 * x - 4 * x - 2 = 7 :=
by
  sorry

end substitution_result_l289_289173


namespace train_cross_bridge_in_56_seconds_l289_289020

noncomputable def train_pass_time (length_train length_bridge : ℝ) (speed_train_kmh : ℝ) : ℝ :=
  let total_distance := length_train + length_bridge
  let speed_train_ms := speed_train_kmh * (1000 / 3600)
  total_distance / speed_train_ms

theorem train_cross_bridge_in_56_seconds :
  train_pass_time 560 140 45 = 56 :=
by
  -- The proof can be added here
  sorry

end train_cross_bridge_in_56_seconds_l289_289020


namespace avg_cans_used_per_game_l289_289652

theorem avg_cans_used_per_game (total_rounds : ℕ) (games_first_round : ℕ) (games_second_round : ℕ)
  (games_third_round : ℕ) (games_finals : ℕ) (total_tennis_balls : ℕ) (balls_per_can : ℕ)
  (h1 : total_rounds = 4) (h2 : games_first_round = 8) (h3 : games_second_round = 4) 
  (h4 : games_third_round = 2) (h5 : games_finals = 1) (h6 : total_tennis_balls = 225) 
  (h7 : balls_per_can = 3) :
  let total_games := games_first_round + games_second_round + games_third_round + games_finals
  let total_cans_used := total_tennis_balls / balls_per_can
  let avg_cans_per_game := total_cans_used / total_games
  avg_cans_per_game = 5 :=
by {
  -- proof steps here
  sorry
}

end avg_cans_used_per_game_l289_289652


namespace total_books_l289_289360

theorem total_books (Zig_books : ℕ) (Flo_books : ℕ) (Tim_books : ℕ) 
  (hz : Zig_books = 60) (hf : Zig_books = 4 * Flo_books) (ht : Tim_books = Flo_books / 2) :
  Zig_books + Flo_books + Tim_books = 82 := by
  sorry

end total_books_l289_289360


namespace smallest_positive_perfect_square_divisible_by_2_and_5_l289_289199

theorem smallest_positive_perfect_square_divisible_by_2_and_5 :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, n = k^2) ∧ (2 ∣ n) ∧ (5 ∣ n) ∧ n = 100 :=
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_and_5_l289_289199


namespace people_per_table_l289_289787

def total_people_invited : ℕ := 68
def people_who_didn't_show_up : ℕ := 50
def number_of_tables_needed : ℕ := 6

theorem people_per_table (total_people_invited people_who_didn't_show_up number_of_tables_needed : ℕ) : 
  total_people_invited - people_who_didn't_show_up = 18 ∧
  (total_people_invited - people_who_didn't_show_up) / number_of_tables_needed = 3 :=
by
  sorry

end people_per_table_l289_289787


namespace transformed_inequality_solution_l289_289533

variable {a b c d : ℝ}

theorem transformed_inequality_solution (H : ∀ x : ℝ, ((-1 < x ∧ x < -1/3) ∨ (1/2 < x ∧ x < 1)) → 
  (b / (x + a) + (x + d) / (x + c) < 0)) :
  ∀ x : ℝ, ((1 < x ∧ x < 3) ∨ (-2 < x ∧ x < -1)) ↔ (bx / (ax - 1) + (dx - 1) / (cx - 1) < 0) :=
sorry

end transformed_inequality_solution_l289_289533


namespace dave_non_working_games_l289_289393

def total_games : ℕ := 10
def price_per_game : ℕ := 4
def total_earnings : ℕ := 32

theorem dave_non_working_games : (total_games - (total_earnings / price_per_game)) = 2 := by
  sorry

end dave_non_working_games_l289_289393


namespace inequality_range_l289_289529

theorem inequality_range (k : ℝ) : (∀ x : ℝ, abs (x + 1) - abs (x - 2) > k) → k < -3 :=
by
  sorry

end inequality_range_l289_289529


namespace xy_sum_greater_two_l289_289607

theorem xy_sum_greater_two (x y : ℝ) (h1 : x^3 > y^2) (h2 : y^3 > x^2) : x + y > 2 := 
by 
  sorry

end xy_sum_greater_two_l289_289607


namespace tv_sets_sales_decrease_l289_289584

theorem tv_sets_sales_decrease
  (P Q P' Q' R R': ℝ)
  (h1 : P' = 1.6 * P)
  (h2 : R' = 1.28 * R)
  (h3 : R = P * Q)
  (h4 : R' = P' * Q')
  (h5 : Q' = Q * (1 - D / 100)) :
  D = 20 :=
by
  sorry

end tv_sets_sales_decrease_l289_289584


namespace fishing_tomorrow_l289_289884

theorem fishing_tomorrow (yesterday_fishers today_fishers : ℕ)
  (every_day_fishers every_other_day_fishers every_three_days_fishers : ℕ)
  (total_population : ℕ):
  yesterday_fishers = 12 → 
  today_fishers = 10 → 
  every_day_fishers = 7 → 
  every_other_day_fishers = 8 → 
  every_three_days_fishers = 3 → 
  total_population = yesterday_fishers + today_fishers + (total_population - (every_day_fishers + every_other_day_fishers + every_three_days_fishers)) →
  ∃ tomorrow_fishers : ℕ, tomorrow_fishers = 15 :=
by {
  -- This is a statement definition, the proof is not required and thus marked as "sorry:"
  sorry
}

end fishing_tomorrow_l289_289884


namespace find_speed_of_boat_l289_289462

theorem find_speed_of_boat (r d t : ℝ) (x : ℝ) (h_rate : r = 4) (h_dist : d = 33.733333333333334) (h_time : t = 44 / 60) 
  (h_eq : d = (x + r) * t) : x = 42.09090909090909 :=
  sorry

end find_speed_of_boat_l289_289462


namespace find_b_values_l289_289085

noncomputable def has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + b*x + 25 = 0

theorem find_b_values (b : ℝ) : has_real_root b ↔ b ∈ set.Iic (-10) ∪ set.Ici 10 := by
  sorry

end find_b_values_l289_289085


namespace union_eq_universal_set_l289_289995

-- Define the sets U, M, and N
def U : Set ℕ := {2, 3, 4, 5, 6}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {2, 4, 6}

-- The theorem stating the desired equality
theorem union_eq_universal_set : M ∪ N = U := 
sorry

end union_eq_universal_set_l289_289995


namespace fishing_problem_l289_289859

theorem fishing_problem
  (everyday : ℕ)
  (every_other_day : ℕ)
  (every_three_days : ℕ)
  (yesterday_fishing : ℕ)
  (today_fishing : ℕ)
  (h_everyday : everyday = 7)
  (h_every_other_day : every_other_day = 8)
  (h_every_three_days : every_three_days = 3)
  (h_yesterday_fishing : yesterday_fishing = 12)
  (h_today_fishing : today_fishing = 10) :
  (every_three_days + everyday + (every_other_day - (yesterday_fishing - everyday))) = 15 := by
  sorry

end fishing_problem_l289_289859


namespace arithmetic_sequence_a4_l289_289900

theorem arithmetic_sequence_a4 (S : ℕ → ℚ) (a : ℕ → ℚ) (a1 : ℚ) (d : ℚ) :
  S 8 = 30 → S 4 = 7 → 
      (∀ n, S n = n * a1 + (n * (n - 1) / 2) * d) → 
      a 4 = a1 + 3 * d → 
      a 4 = 13 / 4 := by
  intros hS8 hS4 hS_formula ha4_formula
  -- Formal proof to be filled in
  sorry

end arithmetic_sequence_a4_l289_289900


namespace number_of_groups_is_correct_l289_289460

-- Defining the conditions
def new_players : Nat := 48
def returning_players : Nat := 6
def players_per_group : Nat := 6
def total_players : Nat := new_players + returning_players

-- Theorem to prove the number of groups
theorem number_of_groups_is_correct : total_players / players_per_group = 9 := by
  sorry

end number_of_groups_is_correct_l289_289460


namespace fixed_monthly_fee_l289_289531

def FebruaryBill (x y : ℝ) : Prop := x + y = 18.72
def MarchBill (x y : ℝ) : Prop := x + 3 * y = 28.08

theorem fixed_monthly_fee (x y : ℝ) (h1 : FebruaryBill x y) (h2 : MarchBill x y) : x = 14.04 :=
by 
  sorry

end fixed_monthly_fee_l289_289531


namespace quadratic_roots_interval_l289_289090

theorem quadratic_roots_interval (b : ℝ) :
  ∃ (x : ℝ), x^2 + b * x + 25 = 0 → b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_roots_interval_l289_289090


namespace travel_distance_correct_l289_289651

noncomputable def traveler_distance : ℝ :=
  let x1 : ℝ := -4
  let y1 : ℝ := 0
  let x2 : ℝ := x1 + 5 * Real.cos (-(Real.pi / 3))
  let y2 : ℝ := y1 + 5 * Real.sin (-(Real.pi / 3))
  let x3 : ℝ := x2 + 2
  let y3 : ℝ := y2
  Real.sqrt (x3^2 + y3^2)

theorem travel_distance_correct : traveler_distance = Real.sqrt 19 := by
  sorry

end travel_distance_correct_l289_289651


namespace cos_diff_l289_289421

theorem cos_diff (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.tan α = 2) : 
  Real.cos (α - π / 4) = 3 * Real.sqrt 10 / 10 :=
sorry

end cos_diff_l289_289421


namespace sequence_is_odd_l289_289336

theorem sequence_is_odd (a : ℕ → ℤ) 
  (h1 : a 1 = 2) 
  (h2 : a 2 = 7) 
  (h3 : ∀ n ≥ 2, -1/2 < (a (n + 1)) - (a n) * (a n) / a (n-1) ∧
                (a (n + 1)) - (a n) * (a n) / a (n-1) ≤ 1/2) :
  ∀ n > 1, (a n) % 2 = 1 :=
by
  sorry

end sequence_is_odd_l289_289336


namespace walking_distance_l289_289354

theorem walking_distance (west east : ℤ) (h_west : west = 5) (h_east : east = -5) : west + east = 10 := 
by 
  rw [h_west, h_east] 
  sorry

end walking_distance_l289_289354


namespace intersection_of_sets_l289_289056

open Set

theorem intersection_of_sets :
  let A := {x : ℤ | |x| < 3}
  let B := {x : ℤ | |x| > 1}
  A ∩ B = ({-2, 2} : Set ℤ) := by
  sorry

end intersection_of_sets_l289_289056


namespace ratio_of_plums_to_peaches_is_three_l289_289822

theorem ratio_of_plums_to_peaches_is_three :
  ∃ (L P W : ℕ), W = 1 ∧ P = W + 12 ∧ L = 3 * P ∧ W + P + L = 53 ∧ (L / P) = 3 :=
by
  sorry

end ratio_of_plums_to_peaches_is_three_l289_289822


namespace range_of_a_monotonically_decreasing_l289_289849

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 3 * x

theorem range_of_a_monotonically_decreasing (a : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 1 → deriv (f a) x ≤ 0) ↔ 0 < a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_monotonically_decreasing_l289_289849


namespace base9_to_base10_l289_289788

def num_base9 : ℕ := 521 -- Represents 521_9
def base : ℕ := 9

theorem base9_to_base10 : 
  (1 * base^0 + 2 * base^1 + 5 * base^2) = 424 := 
by
  -- Sorry allows us to skip the proof.
  sorry

end base9_to_base10_l289_289788


namespace problem_quadratic_has_real_root_l289_289117

theorem problem_quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end problem_quadratic_has_real_root_l289_289117


namespace max_profit_at_35_l289_289224

-- Define the conditions
def unit_purchase_price : ℝ := 20
def base_selling_price : ℝ := 30
def base_sales_volume : ℕ := 400
def price_increase_effect : ℝ := 1
def sales_volume_decrease_per_dollar : ℝ := 20

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - unit_purchase_price) * (base_sales_volume - sales_volume_decrease_per_dollar * (x - base_selling_price))

-- Lean statement to prove that the selling price which maximizes the profit is 35
theorem max_profit_at_35 : ∃ x : ℝ, x = 35 ∧ ∀ y : ℝ, profit y ≤ profit 35 := 
  sorry

end max_profit_at_35_l289_289224


namespace similar_iff_condition_l289_289451

-- Define the similarity of triangles and the necessary conditions.
variables {α : Type*} [LinearOrderedField α]
variables (a b c a' b' c' : α)

-- Statement of the problem in Lean 4
theorem similar_iff_condition : 
  (∃ z w : α, a' = a * z + w ∧ b' = b * z + w ∧ c' = c * z + w) ↔ 
  (a' * (b - c) + b' * (c - a) + c' * (a - b) = 0) :=
sorry

end similar_iff_condition_l289_289451


namespace quadratic_has_real_root_iff_b_in_interval_l289_289094

theorem quadratic_has_real_root_iff_b_in_interval (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ Set.Iic (-10) ∪ Set.Ici 10) := 
sorry

end quadratic_has_real_root_iff_b_in_interval_l289_289094


namespace hens_count_l289_289960

theorem hens_count (H C : ℕ) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 144) 
  (h3 : H ≥ 10) (h4 : C ≥ 5) : H = 24 :=
by
  sorry

end hens_count_l289_289960


namespace a4_in_factorial_base_945_l289_289932

open Nat

def factorial_base_representation (n : Nat) : List Nat :=
  let rec aux (n k : Nat) (acc : List Nat) : List Nat :=
    if k = 0 then acc.reverse
    else
      let coeff := n / (factorial k)
      let rem := n % (factorial k)
      aux rem (k - 1) (coeff :: acc)
  aux n n []

theorem a4_in_factorial_base_945 : (factorial_base_representation 945).nth 4 = some 4 :=
by
  sorry

end a4_in_factorial_base_945_l289_289932


namespace find_k_for_maximum_value_l289_289065

theorem find_k_for_maximum_value (k : ℝ) :
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 2 → k * x^2 + 2 * k * x + 1 ≤ 5) ∧
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 2 ∧ k * x^2 + 2 * k * x + 1 = 5) ↔
  k = 1 / 2 ∨ k = -4 :=
by
  sorry

end find_k_for_maximum_value_l289_289065


namespace largest_N_l289_289420

-- Definition of the problem conditions
def problem_conditions (n : ℕ) (N : ℕ) (a : Fin (N + 1) → ℝ) : Prop :=
  (n ≥ 2) ∧
  (a 0 + a 1 = -(1 : ℝ) / n) ∧  
  (∀ k : ℕ, 1 ≤ k → k ≤ N - 1 → (a k + a (k - 1)) * (a k + a (k + 1)) = a (k - 1) - a (k + 1))

-- The theorem stating that the largest integer N is n
theorem largest_N (n : ℕ) (N : ℕ) (a : Fin (N + 1) → ℝ) :
  problem_conditions n N a → N = n :=
sorry

end largest_N_l289_289420


namespace fishing_tomorrow_l289_289887

-- Conditions
def every_day_fishers : Nat := 7
def every_other_day_fishers : Nat := 8
def every_three_days_fishers : Nat := 3
def yesterday_fishers : Nat := 12
def today_fishers : Nat := 10

-- Determine the number who will fish tomorrow
def fishers_tomorrow : Nat :=
  let every_day_tomorrow := every_day_fishers
  let every_three_day_tomorrow := every_three_days_fishers
  let every_other_day_yesterday := yesterday_fishers - every_day_fishers
  let every_other_day_tomorrow := every_other_day_fishers - every_other_day_yesterday
  every_day_tomorrow + every_three_day_tomorrow + every_other_day_tomorrow

theorem fishing_tomorrow : fishers_tomorrow = 15 :=
  by
    -- skipping the actual proof with sorry
    sorry

end fishing_tomorrow_l289_289887


namespace calc_result_l289_289951

theorem calc_result (initial_number : ℕ) (square : ℕ → ℕ) (subtract_five : ℕ → ℕ) : 
  initial_number = 7 ∧ (square 7 = 49) ∧ (subtract_five 49 = 44) → 
  subtract_five (square initial_number) = 44 := 
by
  sorry

end calc_result_l289_289951


namespace ann_age_l289_289801

variable (A T : ℕ)

-- Condition 1: Tom is currently two times older than Ann
def tom_older : Prop := T = 2 * A

-- Condition 2: The sum of their ages 10 years later will be 38
def age_sum_later : Prop := (A + 10) + (T + 10) = 38

-- Theorem: Ann's current age
theorem ann_age (h1 : tom_older A T) (h2 : age_sum_later A T) : A = 6 :=
by
  sorry

end ann_age_l289_289801


namespace percentage_decrease_l289_289892

theorem percentage_decrease 
  (P0 : ℕ) (P2 : ℕ) (H0 : P0 = 10000) (H2 : P2 = 9600) 
  (P1 : ℕ) (H1 : P1 = P0 + (20 * P0) / 100) :
  ∃ (D : ℕ), P2 = P1 - (D * P1) / 100 ∧ D = 20 :=
by
  sorry

end percentage_decrease_l289_289892


namespace find_smaller_number_l289_289342

theorem find_smaller_number (a b : ℤ) (h1 : a + b = 18) (h2 : a - b = 24) : b = -3 :=
by
  sorry

end find_smaller_number_l289_289342


namespace quadratic_root_difference_l289_289738

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

noncomputable def root_difference (a b c : ℝ) : ℝ :=
  (Real.sqrt (discriminant a b c)) / a

theorem quadratic_root_difference :
  root_difference (3 + 2 * Real.sqrt 2) (5 + Real.sqrt 2) (-4) = Real.sqrt (177 - 122 * Real.sqrt 2) :=
by
  sorry

end quadratic_root_difference_l289_289738


namespace smallest_y_square_l289_289854

theorem smallest_y_square (y n : ℕ) (h1 : y = 10) (h2 : ∀ m : ℕ, (∃ z : ℕ, m * y = z^2) ↔ (m = n)) : n = 10 :=
sorry

end smallest_y_square_l289_289854


namespace same_color_combination_probability_l289_289231

-- Defining the number of each color candy 
def num_red : Nat := 12
def num_blue : Nat := 12
def num_green : Nat := 6

-- Terry and Mary each pick 3 candies at random
def total_pick : Nat := 3

-- The total number of candies in the jar
def total_candies : Nat := num_red + num_blue + num_green

-- Probability of Terry and Mary picking the same color combination
def probability_same_combination : ℚ := 2783 / 847525

-- The theorem statement
theorem same_color_combination_probability :
  let terry_picks_red := (num_red * (num_red - 1) * (num_red - 2)) / (total_candies * (total_candies - 1) * (total_candies - 2))
  let remaining_red := num_red - total_pick
  let mary_picks_red := (remaining_red * (remaining_red - 1) * (remaining_red - 2)) / (27 * 26 * 25)
  let combined_red := terry_picks_red * mary_picks_red

  let terry_picks_blue := (num_blue * (num_blue - 1) * (num_blue - 2)) / (total_candies * (total_candies - 1) * (total_candies - 2))
  let remaining_blue := num_blue - total_pick
  let mary_picks_blue := (remaining_blue * (remaining_blue - 1) * (remaining_blue - 2)) / (27 * 26 * 25)
  let combined_blue := terry_picks_blue * mary_picks_blue

  let terry_picks_green := (num_green * (num_green - 1) * (num_green - 2)) / (total_candies * (total_candies - 1) * (total_candies - 2))
  let remaining_green := num_green - total_pick
  let mary_picks_green := (remaining_green * (remaining_green - 1) * (remaining_green - 2)) / (27 * 26 * 25)
  let combined_green := terry_picks_green * mary_picks_green

  let total_probability := 2 * combined_red + 2 * combined_blue + combined_green
  total_probability = probability_same_combination := sorry

end same_color_combination_probability_l289_289231


namespace solve_for_x_l289_289358

theorem solve_for_x (x : ℝ) (h : (2012 + x)^2 = x^2) : x = -1006 := 
sorry

end solve_for_x_l289_289358


namespace quadratic_real_roots_l289_289113

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := 
by 
  sorry

end quadratic_real_roots_l289_289113


namespace find_part_length_in_inches_find_part_length_in_feet_and_inches_l289_289648

def feetToInches (feet : ℕ) : ℕ := feet * 12

def totalLengthInInches (feet : ℕ) (inches : ℕ) : ℕ := feetToInches feet + inches

def partLengthInInches (totalLength : ℕ) (parts : ℕ) : ℕ := totalLength / parts

def inchesToFeetAndInches (inches : ℕ) : Nat × Nat := (inches / 12, inches % 12)

theorem find_part_length_in_inches (feet : ℕ) (inches : ℕ) (parts : ℕ)
    (h1 : feet = 10) (h2 : inches = 5) (h3 : parts = 5) :
    partLengthInInches (totalLengthInInches feet inches) parts = 25 := by
  sorry

theorem find_part_length_in_feet_and_inches (feet : ℕ) (inches : ℕ) (parts : ℕ)
    (h1 : feet = 10) (h2 : inches = 5) (h3 : parts = 5) :
    inchesToFeetAndInches (partLengthInInches (totalLengthInInches feet inches) parts) = (2, 1) := by
  sorry

end find_part_length_in_inches_find_part_length_in_feet_and_inches_l289_289648


namespace cappuccino_cost_l289_289573

theorem cappuccino_cost 
  (total_order_cost drip_price espresso_price latte_price syrup_price cold_brew_price total_other_cost : ℝ)
  (h1 : total_order_cost = 25)
  (h2 : drip_price = 2 * 2.25)
  (h3 : espresso_price = 3.50)
  (h4 : latte_price = 2 * 4.00)
  (h5 : syrup_price = 0.50)
  (h6 : cold_brew_price = 2 * 2.50)
  (h7 : total_other_cost = drip_price + espresso_price + latte_price + syrup_price + cold_brew_price) :
  total_order_cost - total_other_cost = 3.50 := 
by
  sorry

end cappuccino_cost_l289_289573


namespace find_f_neg2_l289_289532

-- Define the function f and the given conditions
noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x - 4

theorem find_f_neg2 (a b : ℝ) (h₁ : f 2 a b = 6) : f (-2) a b = -14 :=
by
  sorry

end find_f_neg2_l289_289532


namespace range_of_a_l289_289284

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) ↔ (a ≤ -2 ∨ a = 1) := 
sorry

end range_of_a_l289_289284


namespace circumcircle_and_nine_point_circle_right_angle_intersection_l289_289743

-- Definitions for angles and circles
variables {A B C : ℝ} -- Angles in radians
variables {R : ℝ} -- Radius of the circumcircle
variables {O J H : EuclideanGeometry.Point} -- Centers O (circumcenter), J (nine-point center), H (orthocenter) in 2D Euclidean space

-- Given condition on the angles in a triangle
axiom sine_sum_eq_one : sin A ^ 2 + sin B ^ 2 + sin C ^ 2 = 1

-- Theorem to be proven
theorem circumcircle_and_nine_point_circle_right_angle_intersection
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h_angle_sum : A + B + C = π)
  (h_sin_sum : sin A ^ 2 + sin B ^ 2 + sin C ^ 2 = 1)
  (h_OJ : dist O J = R * sqrt ((1:ℝ) / 2 * (1 + cos (A + B) + cos (B + C) + cos (C + A))))
  : ∠ O H J = π / 2 :=
begin
  sorry, -- Proof goes here
end

end circumcircle_and_nine_point_circle_right_angle_intersection_l289_289743


namespace youtube_dislikes_calculation_l289_289627

theorem youtube_dislikes_calculation :
  ∀ (l d_initial d_final : ℕ),
    l = 3000 →
    d_initial = (l / 2) + 100 →
    d_final = d_initial + 1000 →
    d_final = 2600 :=
by
  intros l d_initial d_final h_l h_d_initial h_d_final
  sorry

end youtube_dislikes_calculation_l289_289627


namespace chicks_increased_l289_289702

theorem chicks_increased (chicks_day1 chicks_day2: ℕ) (H1 : chicks_day1 = 23) (H2 : chicks_day2 = 12) : 
  chicks_day1 + chicks_day2 = 35 :=
by
  sorry

end chicks_increased_l289_289702


namespace units_digit_of_factorial_sum_l289_289991

theorem units_digit_of_factorial_sum : 
  (1 + 2 + 6 + 4) % 10 = 3 := sorry

end units_digit_of_factorial_sum_l289_289991


namespace Znayka_sufficient_numbers_l289_289619

theorem Znayka_sufficient_numbers :
  ∀ (p q : ℝ),
  (p > 4 ∧ p > q) →
  (x^2 + p * x + q).root_count > 0 ∧
  ∀ (r s : ℝ),
  (0 < q ∧ 0 < r ∧ 0 < s ∧ q < r ∧ r < s < p) →
  (x^2 + p * x + q).root_count > 0 →
  (x^2 + r * x + s).root_count > 0 →
  (distinct_roots (x^2 + p * x + q) ∧ distinct_roots (x^2 + r * x + s)) :=
sorry

end Znayka_sufficient_numbers_l289_289619


namespace min_sum_squares_roots_l289_289410

theorem min_sum_squares_roots (m : ℝ) :
  (∃ (α β : ℝ), 2 * α^2 - 3 * α + m = 0 ∧ 2 * β^2 - 3 * β + m = 0 ∧ α ≠ β) → 
  (9 - 8 * m ≥ 0) →
  (α^2 + β^2 = (3/2)^2 - 2 * (m/2)) →
  (α^2 + β^2 = 9/8) ↔ m = 9/8 :=
by
  sorry

end min_sum_squares_roots_l289_289410


namespace smallest_y_square_l289_289855

theorem smallest_y_square (y n : ℕ) (h1 : y = 10) (h2 : ∀ m : ℕ, (∃ z : ℕ, m * y = z^2) ↔ (m = n)) : n = 10 :=
sorry

end smallest_y_square_l289_289855


namespace ways_to_return_0_non_negative_ways_to_return_0_l289_289139

open Nat

-- Define the number of ways to return to flower 0 after k jumps using binomial coefficients
def ways_to_return_to_0_after_k_jumps (k : ℕ) : ℕ :=
  if h : Even k then binomial k (k / 2) else 0

-- Define the number of ways to return to flower 0 after k jumps without landing on a negative index using Catalan numbers
def non_negative_ways_to_return_to_0_after_k_jumps (k : ℕ) : ℕ :=
  if h : Even k then Catalan (k / 2) else 0

-- Prove the number of ways to return to flower 0 after k jumps is binomial coefficient when k is even
theorem ways_to_return_0 (k : ℕ) (h : Even k) :
  ways_to_return_to_0_after_k_jumps k = binomial k (k / 2) :=
by {
  rw [ways_to_return_to_0_after_k_jumps],
  exact if_pos h
}

-- Prove the number of ways to return to flower 0 after k jumps without landing on a negative index
-- is the Catalan number when k is even
theorem non_negative_ways_to_return_0 (k : ℕ) (h : Even k) :
  non_negative_ways_to_return_to_0_after_k_jumps k = Catalan (k / 2) :=
by {
  rw [non_negative_ways_to_return_to_0_after_k_jumps],
  exact if_pos h
}

end ways_to_return_0_non_negative_ways_to_return_0_l289_289139


namespace cars_count_l289_289766

theorem cars_count
  (distance : ℕ)
  (time_between_cars : ℕ)
  (total_time_hours : ℕ)
  (cars_per_hour : ℕ)
  (expected_cars_count : ℕ) :
  distance = 3 →
  time_between_cars = 20 →
  total_time_hours = 10 →
  cars_per_hour = 3 →
  expected_cars_count = total_time_hours * cars_per_hour →
  expected_cars_count = 30 :=
by
  intros h1 h2 h3 h4 h5
  rw [h3, h4] at h5
  exact h5


end cars_count_l289_289766


namespace speed_in_kmh_l289_289792

def distance : ℝ := 550.044
def time : ℝ := 30
def conversion_factor : ℝ := 3.6

theorem speed_in_kmh : (distance / time) * conversion_factor = 66.00528 := 
by
  sorry

end speed_in_kmh_l289_289792


namespace solve_abs_inequality_l289_289515

theorem solve_abs_inequality (x : ℝ) : abs ((7 - x) / 4) < 3 → 2 < x ∧ x < 19 :=
by 
  sorry

end solve_abs_inequality_l289_289515


namespace additional_cost_per_pint_proof_l289_289019

-- Definitions based on the problem conditions
def pints_sold := 54
def total_revenue_on_sale := 216
def revenue_difference := 108

-- Derived definitions
def revenue_if_not_on_sale := total_revenue_on_sale + revenue_difference
def cost_per_pint_on_sale := total_revenue_on_sale / pints_sold
def cost_per_pint_not_on_sale := revenue_if_not_on_sale / pints_sold
def additional_cost_per_pint := cost_per_pint_not_on_sale - cost_per_pint_on_sale

-- Proof statement
theorem additional_cost_per_pint_proof :
  additional_cost_per_pint = 2 :=
by
  -- Placeholder to indicate that the proof is not provided
  sorry

end additional_cost_per_pint_proof_l289_289019


namespace series_sum_l289_289563

variable (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_gt : b < a)

noncomputable def infinite_series : ℝ := 
∑' n, 1 / ( ((n - 1) * a^2 - (n - 2) * b^2) * (n * a^2 - (n - 1) * b^2) )

theorem series_sum : infinite_series a b = 1 / ((a^2 - b^2) * b^2) := 
by 
  sorry

end series_sum_l289_289563


namespace simplify_expression_l289_289330

theorem simplify_expression
  (h0 : (Real.pi / 2) < 2 ∧ 2 < Real.pi)  -- Given conditions on 2 related to π.
  (h1 : Real.sin 2 > 0)  -- Given condition that sin 2 is positive.
  (h2 : Real.cos 2 < 0)  -- Given condition that cos 2 is negative.
  : 2 * Real.sqrt (1 + Real.sin 4) + Real.sqrt (2 + 2 * Real.cos 4) = 2 * Real.sin 2 :=
sorry

end simplify_expression_l289_289330


namespace find_vector_from_origin_to_line_l289_289033

theorem find_vector_from_origin_to_line :
  ∃ t : ℝ, (3 * t + 1, 2 * t + 3) = (16, 32 / 3) ∧
  ∃ k : ℝ, (16, 32 / 3) = (3 * k, 2 * k) :=
sorry

end find_vector_from_origin_to_line_l289_289033


namespace billy_raspberry_juice_billy_raspberry_juice_quarts_l289_289030

theorem billy_raspberry_juice (V : ℚ) (h : V / 12 + 1 = 3) : V = 24 :=
by sorry

theorem billy_raspberry_juice_quarts (V : ℚ) (h : V / 12 + 1 = 3) : V / 4 = 6 :=
by sorry

end billy_raspberry_juice_billy_raspberry_juice_quarts_l289_289030


namespace car_late_speed_l289_289010

theorem car_late_speed :
  ∀ (d : ℝ) (t_on_time : ℝ) (t_late : ℝ) (v_on_time : ℝ) (v_late : ℝ),
  d = 225 →
  v_on_time = 60 →
  t_on_time = d / v_on_time →
  t_late = t_on_time + 0.75 →
  v_late = d / t_late →
  v_late = 50 :=
by
  intros d t_on_time t_late v_on_time v_late hd hv_on_time ht_on_time ht_late hv_late
  sorry

end car_late_speed_l289_289010


namespace tilings_of_3_by_5_rectangle_l289_289252

def num_tilings_of_3_by_5_rectangle : ℕ := 96

theorem tilings_of_3_by_5_rectangle (h : ℕ := 96) :
  (∃ (tiles : List (ℕ × ℕ)),
    tiles = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)] ∧
    -- Whether we are counting tiles in the context of a 3x5 rectangle
    -- with all distinct rotations and reflections allowed.
    True
  ) → num_tilings_of_3_by_5_rectangle = h :=
by {
  sorry -- Proof goes here
}

end tilings_of_3_by_5_rectangle_l289_289252


namespace principal_amount_l289_289614

theorem principal_amount (SI R T : ℕ) (P : ℕ) : SI = 160 ∧ R = 5 ∧ T = 4 → P = 800 :=
by
  sorry

end principal_amount_l289_289614


namespace total_animal_legs_is_12_l289_289445

-- Define the number of legs per dog and chicken
def legs_per_dog : Nat := 4
def legs_per_chicken : Nat := 2

-- Define the number of dogs and chickens Mrs. Hilt saw
def number_of_dogs : Nat := 2
def number_of_chickens : Nat := 2

-- Calculate the total number of legs seen
def total_legs_seen : Nat :=
  (number_of_dogs * legs_per_dog) + (number_of_chickens * legs_per_chicken)

-- The theorem to be proven
theorem total_animal_legs_is_12 : total_legs_seen = 12 :=
by
  sorry

end total_animal_legs_is_12_l289_289445


namespace trigonometric_identity_l289_289057

noncomputable def tan_sum (alpha : ℝ) : Prop :=
  Real.tan (alpha + Real.pi / 4) = 2

noncomputable def trigonometric_expression (alpha : ℝ) : ℝ :=
  (Real.sin alpha + 2 * Real.cos alpha) / (Real.sin alpha - 2 * Real.cos alpha)

theorem trigonometric_identity (alpha : ℝ) (h : tan_sum alpha) : 
  trigonometric_expression alpha = -7 / 5 :=
sorry

end trigonometric_identity_l289_289057


namespace height_average_inequality_l289_289610

theorem height_average_inequality 
    (a b c d : ℝ)
    (h1 : 3 * a + 2 * b = 2 * c + 3 * d)
    (h2 : a > d) : 
    (|c + d| / 2 > |a + b| / 2) :=
sorry

end height_average_inequality_l289_289610


namespace sin_2x_eq_7_div_25_l289_289831

theorem sin_2x_eq_7_div_25 (x : ℝ) (h : Real.sin (Real.pi / 4 - x) = 3 / 5) :
    Real.sin (2 * x) = 7 / 25 := by
  sorry

end sin_2x_eq_7_div_25_l289_289831


namespace number_of_feet_l289_289497

theorem number_of_feet (H C F : ℕ) (hH : H = 26) (hHeads : H + C = 48) : F = 140 :=
by
  have hC : C = 48 - 26, from calc
    C = 48 - H    : by linarith
    ... = 48 - 26 : by rw hH
  have hFeet : F = (H * 2) + (C * 4), from sorry -- Define the equation for F
  rw [hH, hC] at hFeet
  linarith

end number_of_feet_l289_289497


namespace remainder_17_pow_2037_mod_20_l289_289770

theorem remainder_17_pow_2037_mod_20:
      (17^1) % 20 = 17 ∧
      (17^2) % 20 = 9 ∧
      (17^3) % 20 = 13 ∧
      (17^4) % 20 = 1 → 
      (17^2037) % 20 = 17 := sorry

end remainder_17_pow_2037_mod_20_l289_289770


namespace find_integer_pairs_l289_289043

noncomputable def satisfies_equation (x y : ℤ) :=
  12 * x ^ 2 + 6 * x * y + 3 * y ^ 2 = 28 * (x + y)

theorem find_integer_pairs (m n : ℤ) :
  satisfies_equation (3 * m - 4 * n) (4 * n) :=
sorry

end find_integer_pairs_l289_289043


namespace range_of_m_l289_289686

def P (m : ℝ) : Prop := m^2 - 4 > 0
def Q (m : ℝ) : Prop := 16 * (m - 2)^2 - 16 < 0

theorem range_of_m (m : ℝ) :
  (P m ∨ Q m) ∧ ¬(P m ∧ Q m) ↔ (m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) :=
by
  sorry

end range_of_m_l289_289686


namespace number_of_1989_periodic_points_l289_289567

noncomputable def f (z : ℂ) (m : ℕ) : ℂ := z ^ m

noncomputable def is_periodic_point (z : ℂ) (f : ℂ → ℂ) (n : ℕ) : Prop :=
f^[n] z = z ∧ ∀ k : ℕ, k < n → (f^[k] z) ≠ z

noncomputable def count_periodic_points (m n : ℕ) : ℕ :=
m^n - m^(n / 3) - m^(n / 13) - m^(n / 17) + m^(n / 39) + m^(n / 51) + m^(n / 117) - m^(n / 153)

theorem number_of_1989_periodic_points (m : ℕ) (hm : 1 < m) :
  count_periodic_points m 1989 = m^1989 - m^663 - m^153 - m^117 + m^51 + m^39 + m^9 - m^3 :=
sorry

end number_of_1989_periodic_points_l289_289567


namespace smallest_y_square_factor_l289_289853

theorem smallest_y_square_factor (y n : ℕ) (h₀ : y = 10) 
  (h₁ : ∀ m : ℕ, ∃ k : ℕ, k * k = m * y)
  (h₂ : ∀ (y' : ℕ), (∀ m : ℕ, ∃ k : ℕ, k * k = m * y') → y ≤ y') : 
  n = 10 :=
by sorry

end smallest_y_square_factor_l289_289853


namespace prob_same_color_seven_red_and_five_green_l289_289842

noncomputable def probability_same_color (red_plat : ℕ) (green_plat : ℕ) : ℚ :=
  let total_plates := red_plat + green_plat
  let total_pairs := (total_plates.choose 2) -- total ways to select 2 plates
  let red_pairs := (red_plat.choose 2) -- ways to select 2 red plates
  let green_pairs := (green_plat.choose 2) -- ways to select 2 green plates
  (red_pairs + green_pairs) / total_pairs

theorem prob_same_color_seven_red_and_five_green :
  probability_same_color 7 5 = 31 / 66 :=
by
  sorry

end prob_same_color_seven_red_and_five_green_l289_289842


namespace bananas_left_l289_289394

-- Definitions based on conditions
def original_bananas : ℕ := 46
def bananas_removed : ℕ := 5

-- Statement of the problem using the definitions
theorem bananas_left : original_bananas - bananas_removed = 41 :=
by sorry

end bananas_left_l289_289394


namespace inequality_system_solution_l289_289922

theorem inequality_system_solution (x : ℝ) :
  (3 * x > x + 6) ∧ ((1 / 2) * x < -x + 5) ↔ (3 < x) ∧ (x < 10 / 3) :=
by
  sorry

end inequality_system_solution_l289_289922


namespace num_teachers_l289_289488

variable (num_students : ℕ) (ticket_cost : ℕ) (total_cost : ℕ)

theorem num_teachers (h1 : num_students = 20) (h2 : ticket_cost = 5) (h3 : total_cost = 115) :
  (total_cost / ticket_cost - num_students = 3) :=
by
  sorry

end num_teachers_l289_289488


namespace minimize_wood_frame_l289_289240

noncomputable def min_wood_frame (x y : ℝ) : Prop :=
  let area_eq : Prop := x * y + x^2 / 4 = 8
  let length := 2 * (x + y) + Real.sqrt 2 * x
  let y_expr := 8 / x - x / 4
  let length_expr := (3 / 2 + Real.sqrt 2) * x + 16 / x
  let min_x := Real.sqrt (16 / (3 / 2 + Real.sqrt 2))
  area_eq ∧ y = y_expr ∧ length = length_expr ∧ x = 2.343 ∧ y = 2.828

theorem minimize_wood_frame : ∃ x y : ℝ, min_wood_frame x y :=
by
  use 2.343
  use 2.828
  unfold min_wood_frame
  -- we leave the proof of the properties as sorry
  sorry

end minimize_wood_frame_l289_289240


namespace find_orange_juice_amount_l289_289256

variable (s y t oj : ℝ)

theorem find_orange_juice_amount (h1 : s = 0.2) (h2 : y = 0.1) (h3 : t = 0.5) (h4 : oj = t - (s + y)) : oj = 0.2 :=
by
  sorry

end find_orange_juice_amount_l289_289256


namespace quadratic_roots_interval_l289_289088

theorem quadratic_roots_interval (b : ℝ) :
  ∃ (x : ℝ), x^2 + b * x + 25 = 0 → b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_roots_interval_l289_289088


namespace number_of_children_l289_289152

def male_adults : ℕ := 60
def female_adults : ℕ := 60
def total_people : ℕ := 200

def total_adults : ℕ := male_adults + female_adults

theorem number_of_children : total_people - total_adults = 80 :=
by sorry

end number_of_children_l289_289152


namespace ShielaDrawingsPerNeighbor_l289_289595

-- Defining our problem using the given conditions:
def ShielaTotalDrawings : ℕ := 54
def ShielaNeighbors : ℕ := 6

-- Mathematically restating the problem:
theorem ShielaDrawingsPerNeighbor : (ShielaTotalDrawings / ShielaNeighbors) = 9 := by
  sorry

end ShielaDrawingsPerNeighbor_l289_289595


namespace perimeter_of_square_C_l289_289176

theorem perimeter_of_square_C (s_A s_B s_C : ℝ)
  (h1 : 4 * s_A = 16)
  (h2 : 4 * s_B = 32)
  (h3 : s_C = s_B - s_A) :
  4 * s_C = 16 :=
by
  sorry

end perimeter_of_square_C_l289_289176


namespace overall_percent_change_in_stock_l289_289387

noncomputable def stock_change (initial_value : ℝ) : ℝ :=
  let value_after_first_day := 0.85 * initial_value
  let value_after_second_day := 1.25 * value_after_first_day
  (value_after_second_day - initial_value) / initial_value * 100

theorem overall_percent_change_in_stock (x : ℝ) : stock_change x = 6.25 :=
by
  sorry

end overall_percent_change_in_stock_l289_289387


namespace shaded_area_quadrilateral_l289_289185

theorem shaded_area_quadrilateral :
  let large_square_area := 11 * 11
  let small_square_area_1 := 1 * 1
  let small_square_area_2 := 2 * 2
  let small_square_area_3 := 3 * 3
  let small_square_area_4 := 4 * 4
  let other_non_shaded_areas := 12 + 15 + 14
  let total_non_shaded := small_square_area_1 + small_square_area_2 + small_square_area_3 + small_square_area_4 + other_non_shaded_areas
  let shaded_area := large_square_area - total_non_shaded
  shaded_area = 35 := by
  sorry

end shaded_area_quadrilateral_l289_289185


namespace johns_profit_l289_289560

theorem johns_profit
  (trees_chopped : ℕ)
  (planks_per_tree : ℕ)
  (planks_per_table : ℕ)
  (price_per_table : ℕ)
  (labor_cost : ℕ)
  (profit : ℕ) :
  trees_chopped = 30 →
  planks_per_tree = 25 →
  planks_per_table = 15 →
  price_per_table = 300 →
  labor_cost = 3000 →
  profit = 12000 :=
begin
  sorry
end

end johns_profit_l289_289560


namespace mary_mac_download_time_l289_289725

theorem mary_mac_download_time (x : ℕ) (windows_download : ℕ) (total_glitch : ℕ) (time_without_glitches : ℕ) (total_time : ℕ) :
  windows_download = 3 * x ∧
  total_glitch = 14 ∧
  time_without_glitches = 2 * total_glitch ∧
  total_time = 82 ∧
  x + windows_download + total_glitch + time_without_glitches = total_time →
  x = 10 :=
by 
  sorry

end mary_mac_download_time_l289_289725


namespace Karen_tote_weight_l289_289440

variable (B T F : ℝ)
variable (Papers Laptop : ℝ)

theorem Karen_tote_weight (h1: T = 2 * B)
                         (h2: F = 2 * T)
                         (h3: Papers = (1 / 6) * F)
                         (h4: Laptop = T + 2)
                         (h5: F = B + Laptop + Papers):
  T = 12 := 
sorry

end Karen_tote_weight_l289_289440


namespace intersection_A_B_l289_289545

def A : Set ℝ := { x | (x + 1) / (x - 1) ≤ 0 }
def B : Set ℝ := { x | Real.log x ≤ 0 }

theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_A_B_l289_289545


namespace more_children_got_off_than_got_on_l289_289008

-- Define the initial number of children on the bus
def initial_children : ℕ := 36

-- Define the number of children who got off the bus
def children_got_off : ℕ := 68

-- Define the total number of children on the bus after changes
def final_children : ℕ := 12

-- Define the unknown number of children who got on the bus
def children_got_on : ℕ := sorry -- We will use the conditions to solve for this in the proof

-- The main proof statement
theorem more_children_got_off_than_got_on : (children_got_off - children_got_on = 24) :=
by
  -- Write the equation describing the total number of children after changes
  have h1 : initial_children - children_got_off + children_got_on = final_children := sorry
  -- Solve for the number of children who got on the bus (children_got_on)
  have h2 : children_got_on = final_children + (children_got_off - initial_children) := sorry
  -- Substitute to find the required difference
  have h3 : children_got_off - final_children - (children_got_off - initial_children) = 24 := sorry
  -- Conclude the proof
  exact sorry


end more_children_got_off_than_got_on_l289_289008


namespace jack_bought_apples_l289_289557

theorem jack_bought_apples :
  ∃ n : ℕ, 
    (∃ k : ℕ, k = 10 ∧ ∃ m : ℕ, m = 5 * 9 ∧ n = k + m) ∧ n = 55 :=
by
  sorry

end jack_bought_apples_l289_289557


namespace max_value_xyz_l289_289526

theorem max_value_xyz (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : 2 * x + 3 * x * y^2 + 2 * z = 36) : 
  x^2 * y^2 * z ≤ 144 :=
sorry

end max_value_xyz_l289_289526


namespace alice_students_count_l289_289506

variable (S : ℕ)
variable (students_with_own_vests := 0.20 * S)
variable (students_needing_vests := 0.80 * S)
variable (instructors : ℕ := 10)
variable (life_vests_on_hand : ℕ := 20)
variable (additional_life_vests_needed : ℕ := 22)
variable (total_life_vests_needed := life_vests_on_hand + additional_life_vests_needed)
variable (life_vests_needed_for_instructors := instructors)
variable (life_vests_needed_for_students := total_life_vests_needed - life_vests_needed_for_instructors)

theorem alice_students_count : S = 40 :=
by
  -- proof steps would go here
  sorry

end alice_students_count_l289_289506


namespace abs_reciprocal_inequality_l289_289682

theorem abs_reciprocal_inequality (a b : ℝ) (h : 1 / |a| < 1 / |b|) : |a| > |b| :=
sorry

end abs_reciprocal_inequality_l289_289682


namespace necessary_condition_l289_289273

theorem necessary_condition (m : ℝ) : 
  (∀ x > 0, (x / 2) + (1 / (2 * x)) - (3 / 2) > m) → (m ≤ -1 / 2) :=
by
  -- Proof omitted
  sorry

end necessary_condition_l289_289273


namespace quadrilateral_area_l289_289893

theorem quadrilateral_area {AB BC : ℝ} (hAB : AB = 4) (hBC : BC = 8) :
  ∃ area : ℝ, area = 16 := by
  sorry

end quadrilateral_area_l289_289893


namespace prob_rain_at_least_one_day_l289_289756

noncomputable def prob_rain_saturday := 0.35
noncomputable def prob_rain_sunday := 0.45

theorem prob_rain_at_least_one_day : 
  (1 - (1 - prob_rain_saturday) * (1 - prob_rain_sunday)) * 100 = 64.25 := 
by 
  sorry

end prob_rain_at_least_one_day_l289_289756


namespace bamboo_break_height_l289_289304

-- Conditions provided in the problem
def original_height : ℝ := 20  -- 20 chi
def distance_tip_to_root : ℝ := 6  -- 6 chi

-- Function to check if the height of the break satisfies the equation
def equationHolds (x : ℝ) : Prop :=
  (original_height - x) ^ 2 - x ^ 2 = distance_tip_to_root ^ 2

-- Main statement to prove the height of the break is 9.1 chi
theorem bamboo_break_height : equationHolds 9.1 :=
by
  sorry

end bamboo_break_height_l289_289304


namespace No_of_boxes_in_case_l289_289416

-- Define the conditions
def George_has_total_blocks : ℕ := 12
def blocks_per_box : ℕ := 6
def George_has_boxes : ℕ := George_has_total_blocks / blocks_per_box

-- The theorem to prove
theorem No_of_boxes_in_case : George_has_boxes = 2 :=
by
  sorry

end No_of_boxes_in_case_l289_289416


namespace geometric_seq_sum_l289_289444

-- Definitions of the conditions
def a (n : ℕ) : ℤ :=
  match n with
  | 0 => 1
  | _ => (-3)^(n - 1)

theorem geometric_seq_sum : 
  a 0 + |a 1| + a 2 + |a 3| + a 4 = 121 := by
  sorry

end geometric_seq_sum_l289_289444


namespace abs_sq_lt_self_iff_l289_289984

theorem abs_sq_lt_self_iff {x : ℝ} : abs x * abs x < x ↔ (0 < x ∧ x < 1) ∨ (x < -1) :=
by
  sorry

end abs_sq_lt_self_iff_l289_289984


namespace quadratic_has_real_root_l289_289127

theorem quadratic_has_real_root (b : ℝ) : 
  (b^2 - 100 ≥ 0) ↔ (b ≤ -10 ∨ b ≥ 10) :=
by
  sorry

end quadratic_has_real_root_l289_289127


namespace molecular_weight_of_one_mole_l289_289946

theorem molecular_weight_of_one_mole (total_weight : ℝ) (number_of_moles : ℕ) 
    (h : total_weight = 204) (n : number_of_moles = 3) : 
    (total_weight / number_of_moles) = 68 :=
by
  have h_weight : total_weight = 204 := h
  have h_moles : number_of_moles = 3 := n
  rw [h_weight, h_moles]
  norm_num

end molecular_weight_of_one_mole_l289_289946


namespace incorrect_guess_l289_289465

-- Define the conditions
def bears : ℕ := 1000

inductive Color
| White
| Brown
| Black

constant bear_color : ℕ → Color -- The color of the bear at each position

axiom condition : ∀ n : ℕ, n < bears - 2 → 
  ∃ i j k, (i, j, k ∈ {Color.White, Color.Brown, Color.Black}) ∧ 
  (i ≠ j ∧ j ≠ k ∧ i ≠ k) ∧ 
  (bear_color n = i ∧ bear_color (n+1) = j ∧ bear_color (n+2) = k) 

constants (g1 : bear_color 2 = Color.White)
          (g2 : bear_color 20 = Color.Brown)
          (g3 : bear_color 400 = Color.Black)
          (g4 : bear_color 600 = Color.Brown)
          (g5 : bear_color 800 = Color.White)

-- The proof problem
theorem incorrect_guess : bear_color 20 ≠ Color.Brown :=
by sorry

end incorrect_guess_l289_289465


namespace number_of_shelves_l289_289022

-- Given conditions
def booksBeforeTrip : ℕ := 56
def booksBought : ℕ := 26
def avgBooksPerShelf : ℕ := 20
def booksLeftOver : ℕ := 2
def totalBooks : ℕ := booksBeforeTrip + booksBought

-- Statement to prove
theorem number_of_shelves :
  totalBooks - booksLeftOver = 80 →
  80 / avgBooksPerShelf = 4 := by
  intros h
  sorry

end number_of_shelves_l289_289022


namespace units_digit_of_sequence_l289_289714

theorem units_digit_of_sequence : 
  (2 + 2 * 3 + 2 * 3^2 + 2 * 3^3 + 2 * 3^4 + 2 * 3^5 + 2 * 3^6 + 2 * 3^7 + 2 * 3^8 + 2 * 3^9) % 10 = 8 := 
by 
  sorry

end units_digit_of_sequence_l289_289714


namespace cell_phone_plan_cost_l289_289222

theorem cell_phone_plan_cost:
  let base_cost : ℕ := 25
  let text_cost : ℕ := 8
  let extra_min_cost : ℕ := 12
  let texts_sent : ℕ := 150
  let hours_talked : ℕ := 27
  let extra_minutes := (hours_talked - 25) * 60
  let total_cost := (base_cost * 100) + (texts_sent * text_cost) + (extra_minutes * extra_min_cost)
  (total_cost = 5140) :=
by
  sorry

end cell_phone_plan_cost_l289_289222


namespace problem_l289_289687

universe u

def U : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {x ∈ U | x ≠ 0} -- Placeholder, B itself is a generic subset of U
def A : Set ℕ := {x ∈ U | x = 3 ∨ x = 5 ∨ x = 9}

noncomputable def C_U (B : Set ℕ) : Set ℕ := {x ∈ U | ¬ (x ∈ B)}

axiom h1 : A ∩ B = {3, 5}
axiom h2 : A ∩ C_U B = {9}

theorem problem : A = {3, 5, 9} :=
by
  sorry

end problem_l289_289687


namespace surface_area_reduction_of_spliced_cuboid_l289_289175

theorem surface_area_reduction_of_spliced_cuboid 
  (initial_faces : ℕ := 12)
  (faces_lost : ℕ := 2)
  (percentage_reduction : ℝ := (2 / 12) * 100) :
  percentage_reduction = 16.7 :=
by
  sorry

end surface_area_reduction_of_spliced_cuboid_l289_289175


namespace train_speed_proof_l289_289650

theorem train_speed_proof
  (length_of_train : ℕ)
  (length_of_bridge : ℕ)
  (time_to_cross_bridge : ℕ)
  (h_train_length : length_of_train = 145)
  (h_bridge_length : length_of_bridge = 230)
  (h_time : time_to_cross_bridge = 30) :
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 18 / 5 = 45 :=
by
  sorry

end train_speed_proof_l289_289650


namespace train_A_total_distance_l289_289350

variables (Speed_A : ℝ) (Time_meet : ℝ) (Total_Distance : ℝ)

def Distance_A_to_C (Speed_A Time_meet : ℝ) : ℝ := Speed_A * Time_meet
def Distance_B_to_C (Total_Distance Distance_A_to_C : ℝ) : ℝ := Total_Distance - Distance_A_to_C
def Additional_Distance_A (Speed_A Time_meet : ℝ) : ℝ := Speed_A * Time_meet
def Total_Distance_A (Distance_A_to_C Additional_Distance_A : ℝ) : ℝ :=
  Distance_A_to_C + Additional_Distance_A

theorem train_A_total_distance
  (h1 : Speed_A = 50)
  (h2 : Time_meet = 0.5)
  (h3 : Total_Distance = 120) :
  Total_Distance_A (Distance_A_to_C Speed_A Time_meet)
                   (Additional_Distance_A Speed_A Time_meet) = 50 :=
by 
  rw [Distance_A_to_C, Additional_Distance_A, Total_Distance_A]
  rw [h1, h2]
  norm_num

end train_A_total_distance_l289_289350


namespace linear_eq_find_m_l289_289061

theorem linear_eq_find_m (m : ℤ) (x : ℝ) 
  (h : (m - 5) * x^(|m| - 4) + 5 = 0) 
  (h_linear : |m| - 4 = 1) 
  (h_nonzero : m - 5 ≠ 0) : m = -5 :=
by
  sorry

end linear_eq_find_m_l289_289061


namespace inheritance_division_l289_289914

variables {M P Q R : ℝ} {p q r : ℕ}

theorem inheritance_division (hP : P < 99 * (p : ℝ))
                             (hR : R > 10000 * (r : ℝ))
                             (hM : M = P + Q + R)
                             (hRichPoor : R ≥ P) : 
                             R ≥ 100 * P := 
sorry

end inheritance_division_l289_289914


namespace discount_is_100_l289_289726

-- Define the constants for the problem conditions
def suit_cost : ℕ := 430
def shoes_cost : ℕ := 190
def amount_paid : ℕ := 520

-- Total cost before discount
def total_cost_before_discount (a b : ℕ) : ℕ := a + b

-- Discount amount
def discount_amount (total paid : ℕ) : ℕ := total - paid

-- Main theorem statement
theorem discount_is_100 : discount_amount (total_cost_before_discount suit_cost shoes_cost) amount_paid = 100 := 
by
sorry

end discount_is_100_l289_289726


namespace find_orig_denominator_l289_289379

-- Definitions as per the conditions
def orig_numer : ℕ := 2
def mod_numer : ℕ := orig_numer + 3

-- The modified fraction yields 1/3
def new_fraction (d : ℕ) : Prop :=
  (mod_numer : ℚ) / (d + 4) = 1 / 3

-- Proof Problem Statement
theorem find_orig_denominator (d : ℕ) : new_fraction d → d = 11 :=
  sorry

end find_orig_denominator_l289_289379


namespace jenny_ate_65_chocolates_l289_289147

-- Define the number of chocolate squares Mike ate
def MikeChoc := 20

-- Define the function that calculates the chocolates Jenny ate
def JennyChoc (mikeChoc : ℕ) := 3 * mikeChoc + 5

-- The theorem stating the solution
theorem jenny_ate_65_chocolates (h : MikeChoc = 20) : JennyChoc MikeChoc = 65 := by
  -- Automatic proof step
  sorry

end jenny_ate_65_chocolates_l289_289147


namespace actual_road_length_l289_289728

theorem actual_road_length
  (scale_factor : ℕ → ℕ → Prop)
  (map_length_cm : ℕ)
  (actual_length_km : ℝ) : 
  (scale_factor 1 50000) →
  (map_length_cm = 15) →
  (actual_length_km = 7.5) :=
by
  sorry

end actual_road_length_l289_289728


namespace sum_of_coefficients_l289_289266

theorem sum_of_coefficients (a a1 a2 a3 a4 a5 : ℤ)
  (h : (1 - 2 * X)^5 = a + a1 * X + a2 * X^2 + a3 * X^3 + a4 * X^4 + a5 * X^5) :
  a1 + a2 + a3 + a4 + a5 = -2 :=
by {
  -- the proof steps would go here
  sorry
}

end sum_of_coefficients_l289_289266


namespace quadrilateral_side_difference_l289_289237

variable (a b c d : ℝ)

theorem quadrilateral_side_difference :
  a + b + c + d = 120 →
  a + c = 50 →
  (a^2 + c^2 = 1600) →
  (b + d = 70 ∧ b * d = 450) →
  |b - d| = 2 * Real.sqrt 775 :=
by
  intros ha hb hc hd
  sorry

end quadrilateral_side_difference_l289_289237


namespace frequency_of_group_of_samples_l289_289018

def sample_capacity : ℝ := 32
def frequency_rate : ℝ := 0.125

theorem frequency_of_group_of_samples : frequency_rate * sample_capacity = 4 :=
by 
  sorry

end frequency_of_group_of_samples_l289_289018


namespace fishing_tomorrow_l289_289869

theorem fishing_tomorrow (every_day_fishers every_other_day_fishers every_three_days_fishers fishers_yesterday fishers_today : ℕ) :
  (every_day_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (every_three_days_fishers + every_day_fishers + (every_other_day_fishers - (fishers_yesterday - every_day_fishers)) = 15) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  simp
  sorry

end fishing_tomorrow_l289_289869


namespace candy_lollipops_l289_289191

theorem candy_lollipops (κ c l : ℤ) 
  (h1 : κ = l + c - 8)
  (h2 : c = l + κ - 14) :
  l = 11 :=
by
  sorry

end candy_lollipops_l289_289191


namespace vegetable_planting_methods_l289_289665

theorem vegetable_planting_methods :
  let vegetables := ["cucumber", "cabbage", "rape", "lentils"]
  let cucumber := "cucumber"
  let other_vegetables := ["cabbage", "rape", "lentils"]
  let choose_2_out_of_3 := Nat.choose 3 2
  let arrangements := Nat.factorial 3
  total_methods = choose_2_out_of_3 * arrangements := by
  let total_methods := 3 * 6
  sorry

end vegetable_planting_methods_l289_289665


namespace cube_greater_than_quadratic_minus_linear_plus_one_l289_289564

variable (x : ℝ)

theorem cube_greater_than_quadratic_minus_linear_plus_one (h : x > 1) :
  x^3 > x^2 - x + 1 := by
  sorry

end cube_greater_than_quadratic_minus_linear_plus_one_l289_289564


namespace area_of_circumscribed_circle_l289_289642

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  let R := s / Real.sqrt 3 in 
  Real.pi * R^2 = 48 * Real.pi :=
by
  sorry

end area_of_circumscribed_circle_l289_289642


namespace sum_of_squares_of_coefficients_l289_289203

theorem sum_of_squares_of_coefficients :
  let poly := 5 * (X^6 + 4 * X^4 + 2 * X^2 + 1)
  let coeffs := [5, 20, 10, 5]
  (coeffs.map (λ c => c * c)).sum = 550 := 
by
  sorry

end sum_of_squares_of_coefficients_l289_289203


namespace find_solutions_l289_289485

def system_solutions (x y z : ℝ) : Prop :=
  (x + 1) * y * z = 12 ∧
  (y + 1) * z * x = 4 ∧
  (z + 1) * x * y = 4

theorem find_solutions :
  ∃ (x y z : ℝ), system_solutions x y z ∧ ((x = 2 ∧ y = -2 ∧ z = -2) ∨ (x = 1/3 ∧ y = 3 ∧ z = 3)) :=
by
  sorry

end find_solutions_l289_289485


namespace final_price_relative_l289_289783

-- Definitions of the conditions
variable (x : ℝ)
#check x * 1.30  -- original price increased by 30%
#check x * 1.30 * 0.85  -- after 15% discount on increased price
#check x * 1.30 * 0.85 * 1.05  -- after applying 5% tax on discounted price

-- Theorem to prove the final price relative to the original price
theorem final_price_relative (x : ℝ) : 
  (x * 1.30 * 0.85 * 1.05) = (1.16025 * x) :=
by
  sorry

end final_price_relative_l289_289783


namespace triangle_angles_l289_289309

-- Definitions for altitude and angles in a triangle
variables {A B C : Type} [linear_ordered_field A]

-- Lean statement for the problem
theorem triangle_angles (a b c : A)
    (hA : a > 0)
    (hB : b > 0)
    (hC : c > 0)
    (h_alt_A : a ≤ b)
    (h_alt_B : b ≤ a) :
  (a = 90 ∧ b = 45 ∧ c = 45) :=
begin
  sorry
end

end triangle_angles_l289_289309


namespace mod_remainder_l289_289478

theorem mod_remainder (a b c x: ℤ):
    a = 9 → b = 5 → c = 3 → x = 7 →
    (a^6 + b^7 + c^8) % x = 4 :=
by
  intros
  sorry

end mod_remainder_l289_289478


namespace simplify_evaluate_l289_289736

def f (x y : ℝ) : ℝ := 4 * x^2 * y - (6 * x * y - 3 * (4 * x - 2) - x^2 * y) + 1

theorem simplify_evaluate : f (-2) (1/2) = -13 := by
  sorry

end simplify_evaluate_l289_289736


namespace range_of_m_l289_289706

def has_solution_in_interval (m : ℝ) : Prop :=
  ∃ x ∈ Set.Icc (0 : ℝ) (3 : ℝ), x^2 - 2 * x - 1 + m ≤ 0 

theorem range_of_m (m : ℝ) : has_solution_in_interval m ↔ m ≤ 2 := by 
  sorry

end range_of_m_l289_289706


namespace total_cost_of_oranges_and_mangoes_l289_289670

theorem total_cost_of_oranges_and_mangoes
  (original_price_orange : ℝ)
  (original_price_mango : ℝ)
  (price_increase_percentage : ℝ)
  (quantity_oranges : ℕ)
  (quantity_mangoes : ℕ) :
  original_price_orange = 40 →
  original_price_mango = 50 →
  price_increase_percentage = 0.15 →
  quantity_oranges = 10 →
  quantity_mangoes = 10 →
  (quantity_oranges * (original_price_orange * (1 + price_increase_percentage)) +
   quantity_mangoes * (original_price_mango * (1 + price_increase_percentage))
  ) = 1035 :=
begin
  intros,
  sorry
end

end total_cost_of_oranges_and_mangoes_l289_289670


namespace negation_of_proposition_l289_289426

theorem negation_of_proposition :
  (∀ x : ℝ, x > 0 → x + (1 / x) ≥ 2) →
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ + (1 / x₀) < 2) :=
sorry

end negation_of_proposition_l289_289426


namespace quadratic_root_conditions_l289_289254

theorem quadratic_root_conditions : ∃ p q : ℝ, (p - 1)^2 - 4 * q > 0 ∧ (p + 1)^2 - 4 * q > 0 ∧ p^2 - 4 * q < 0 := 
sorry

end quadratic_root_conditions_l289_289254


namespace min_value_xy_l289_289296

theorem min_value_xy {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : (2 / x) + (8 / y) = 1) : x * y ≥ 64 :=
sorry

end min_value_xy_l289_289296


namespace running_speed_l289_289236

theorem running_speed (R : ℝ) (walking_speed : ℝ) (total_distance : ℝ) (total_time : ℝ) (half_distance : ℝ) (walking_time : ℝ) (running_time : ℝ)
  (h1 : walking_speed = 4)
  (h2 : total_distance = 16)
  (h3 : total_time = 3)
  (h4 : half_distance = total_distance / 2)
  (h5 : walking_time = half_distance / walking_speed)
  (h6 : running_time = half_distance / R)
  (h7 : walking_time + running_time = total_time) :
  R = 8 := 
sorry

end running_speed_l289_289236


namespace problem_quadratic_has_real_root_l289_289116

theorem problem_quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end problem_quadratic_has_real_root_l289_289116


namespace distance_traveled_l289_289005

-- Let T be the time in hours taken to travel the actual distance D at 10 km/hr.
-- Let D be the actual distance traveled by the person.
-- Given: D = 10 * T and D + 40 = 20 * T prove that D = 40.

theorem distance_traveled (T : ℝ) (D : ℝ) 
  (h1 : D = 10 * T)
  (h2 : D + 40 = 20 * T) : 
  D = 40 := by
  sorry

end distance_traveled_l289_289005


namespace complex_numbers_right_triangle_l289_289696

theorem complex_numbers_right_triangle (z : ℂ) (hz : z ≠ 0) :
  (∃ z₁ z₂ : ℂ, z₁ ≠ 0 ∧ z₂ ≠ 0 ∧ z₁^3 = z₂ ∧
                 (∃ θ₁ θ₂ : ℝ, z₁ = Complex.exp (Complex.I * θ₁) ∧
                               z₂ = Complex.exp (Complex.I * θ₂) ∧
                               (θ₂ - θ₁ = π/2 ∨ θ₂ - θ₁ = 3 * π/2))) →
  ∃ n : ℕ, n = 2 :=
by
  sorry

end complex_numbers_right_triangle_l289_289696


namespace find_repair_cost_l289_289913

variable (P : ℕ) (T : ℕ) (SP : ℕ) (PP : ℕ) (R : ℕ)

-- Define the conditions as hypotheses
def conditions : Prop :=
  P = 11000 ∧
  T = 1000 ∧
  SP = 25500 ∧
  PP = 50

-- State the goal to prove
def goal : Prop :=
  SP = 1.5 * (P + T + R) → R = 5000

-- Combine them into a theorem
theorem find_repair_cost (h : conditions) : goal := by
  sorry

end find_repair_cost_l289_289913


namespace shortest_side_of_right_triangle_l289_289021

theorem shortest_side_of_right_triangle
  (a b c : ℝ)
  (h : a = 5) (k : b = 13) (rightangled : a^2 + c^2 = b^2) : c = 12 := 
sorry

end shortest_side_of_right_triangle_l289_289021


namespace probability_symmetric_interval_l289_289795

noncomputable def X : Type := sorry -- Define the type of the random variable X

variable [MeasureTheory.ProbabilityMeasure X]
variable (a : ℝ) (h_mean : a = 10)
variable (h_interval1 : ∀ (X : ℝ), (10 < X ∧ X < 20) → MeasureTheory.ProbabilityMeasure X = 0.3)

theorem probability_symmetric_interval :
  MeasureTheory.Probability (Set.Ioo 0 10) = 0.3 :=
by
  sorry -- The actual proof would go here, but is omitted for this example.

end probability_symmetric_interval_l289_289795


namespace tan_double_angle_l289_289282

theorem tan_double_angle (θ : ℝ) (P : ℝ × ℝ) 
  (h_vertex : θ = 0) 
  (h_initial_side : ∀ x, θ = x)
  (h_terminal_side : P = (-1, 2)) : 
  Real.tan (2 * θ) = 4 / 3 := 
by 
  sorry

end tan_double_angle_l289_289282


namespace trey_total_hours_l289_289581

def num_clean_house := 7
def num_shower := 1
def num_make_dinner := 4
def minutes_per_item := 10
def total_items := num_clean_house + num_shower + num_make_dinner
def total_minutes := total_items * minutes_per_item
def minutes_in_hour := 60

theorem trey_total_hours : total_minutes / minutes_in_hour = 2 := by
  sorry

end trey_total_hours_l289_289581


namespace ratio_of_fish_cat_to_dog_l289_289151

theorem ratio_of_fish_cat_to_dog (fish_dog : ℕ) (cost_per_fish : ℕ) (total_spent : ℕ)
  (h1 : fish_dog = 40)
  (h2 : cost_per_fish = 4)
  (h3 : total_spent = 240) :
  (total_spent / cost_per_fish - fish_dog) / fish_dog = 1 / 2 := by
  sorry

end ratio_of_fish_cat_to_dog_l289_289151


namespace solve_absolute_inequality_l289_289454

theorem solve_absolute_inequality (x : ℝ) : |x - 1| - |x - 2| > 1 / 2 ↔ x > 7 / 4 :=
by sorry

end solve_absolute_inequality_l289_289454


namespace time_to_school_gate_l289_289727

theorem time_to_school_gate (total_time gate_to_building building_to_room time_to_gate : ℕ) 
                            (h1 : total_time = 30)
                            (h2 : gate_to_building = 6)
                            (h3 : building_to_room = 9)
                            (h4 : total_time = time_to_gate + gate_to_building + building_to_room) :
  time_to_gate = 15 :=
  sorry

end time_to_school_gate_l289_289727


namespace line_through_parabola_vertex_unique_value_l289_289819

theorem line_through_parabola_vertex_unique_value :
  ∃! a : ℝ, ∃ y : ℝ, y = x + a ∧ y = x^2 - 2*a*x + a^2 :=
sorry

end line_through_parabola_vertex_unique_value_l289_289819


namespace determine_b_l289_289672

-- Define the problem conditions
variable (n b : ℝ)
variable (h_pos_b : b > 0)
variable (h_eq : ∀ x : ℝ, (x + n) ^ 2 + 16 = x^2 + b * x + 88)

-- State that we want to prove that b equals 12 * sqrt(2)
theorem determine_b : b = 12 * Real.sqrt 2 :=
by
  sorry

end determine_b_l289_289672


namespace payments_option1_option2_option1_more_effective_combined_option_cost_l289_289786

variable {x : ℕ}

-- Condition 1: Prices and discount options
def badminton_rackets_price : ℕ := 40
def shuttlecocks_price : ℕ := 10
def discount_option1_free_shuttlecocks (pairs : ℕ): ℕ := pairs
def discount_option2_price (price : ℕ) : ℕ := price * 9 / 10

-- Condition 2: Buying requirements
def pairs_needed : ℕ := 10
def shuttlecocks_needed (n : ℕ) : ℕ := n
axiom x_gt_10 : x > 10

-- Proof Problem 1: Payment calculations
theorem payments_option1_option2 (x : ℕ) (h : x > 10) :
  (shuttlecocks_price * (shuttlecocks_needed x - discount_option1_free_shuttlecocks pairs_needed) + badminton_rackets_price * pairs_needed =
    10 * x + 300) ∧
  (discount_option2_price (shuttlecocks_price * shuttlecocks_needed x + badminton_rackets_price * pairs_needed) =
    9 * x + 360) :=
sorry

-- Proof Problem 2: More cost-effective option when x=30
theorem option1_more_effective (x : ℕ) (h : x = 30) :
  (10 * x + 300 < 9 * x + 360) :=
sorry

-- Proof Problem 3: Another cost-effective method when x=30
theorem combined_option_cost (x : ℕ) (h : x = 30) :
  (badminton_rackets_price * pairs_needed + discount_option2_price (shuttlecocks_price * (shuttlecocks_needed x - 10)) = 580) :=
sorry

end payments_option1_option2_option1_more_effective_combined_option_cost_l289_289786


namespace theo_selling_price_l289_289570

theorem theo_selling_price:
  ∀ (maddox_price theo_cost maddox_sell theo_profit maddox_profit theo_sell: ℕ),
    maddox_price = 20 → 
    theo_cost = 20 → 
    maddox_sell = 28 →
    maddox_profit = (maddox_sell - maddox_price) * 3 →
    (theo_sell - theo_cost) * 3 = (maddox_profit - 15) →
    theo_sell = 23 := by
  intros maddox_price theo_cost maddox_sell theo_profit maddox_profit theo_sell
  intros maddox_price_eq theo_cost_eq maddox_sell_eq maddox_profit_eq theo_profit_eq

  -- Use given assumptions
  rw [maddox_price_eq, theo_cost_eq, maddox_sell_eq] at *
  simp at *

  -- Final goal
  sorry

end theo_selling_price_l289_289570


namespace solve_problem_l289_289680

def spadesuit (x y : ℝ) : ℝ := x^2 + y^2

theorem solve_problem : spadesuit (spadesuit 3 5) 4 = 1172 := by
  sorry

end solve_problem_l289_289680


namespace percentage_increase_l289_289707

theorem percentage_increase (X Y Z : ℝ) (h1 : X = 1.25 * Y) (h2 : Z = 100) (h3 : X + Y + Z = 370) :
  ((Y - Z) / Z) * 100 = 20 :=
by
  sorry

end percentage_increase_l289_289707


namespace gcd_288_123_l289_289193

theorem gcd_288_123 : gcd 288 123 = 3 :=
by
  sorry

end gcd_288_123_l289_289193


namespace camel_water_ratio_l289_289649

theorem camel_water_ratio (gallons_water : ℕ) (ounces_per_gallon : ℕ) (traveler_ounces : ℕ)
  (total_ounces : ℕ) (camel_ounces : ℕ) (ratio : ℕ) 
  (h1 : gallons_water = 2) 
  (h2 : ounces_per_gallon = 128) 
  (h3 : traveler_ounces = 32) 
  (h4 : total_ounces = gallons_water * ounces_per_gallon) 
  (h5 : camel_ounces = total_ounces - traveler_ounces)
  (h6 : ratio = camel_ounces / traveler_ounces) : 
  ratio = 7 := 
by
  sorry

end camel_water_ratio_l289_289649


namespace convex_polyhedron_formula_l289_289979

theorem convex_polyhedron_formula
  (V E F t h T H : ℕ)
  (hF : F = 40)
  (hFaces : F = t + h)
  (hVertex : 2 * T + H = 7)
  (hEdges : E = (3 * t + 6 * h) / 2)
  (hEuler : V - E + F = 2)
  : 100 * H + 10 * T + V = 367 := 
sorry

end convex_polyhedron_formula_l289_289979


namespace quadratic_has_real_root_iff_b_in_interval_l289_289096

theorem quadratic_has_real_root_iff_b_in_interval (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ Set.Iic (-10) ∪ Set.Ici 10) := 
sorry

end quadratic_has_real_root_iff_b_in_interval_l289_289096


namespace find_added_amount_l289_289962

theorem find_added_amount (x y : ℕ) (h1 : x = 18) (h2 : 3 * (2 * x + y) = 123) : y = 5 :=
by
  sorry

end find_added_amount_l289_289962


namespace quadratic_has_real_root_iff_b_in_interval_l289_289095

theorem quadratic_has_real_root_iff_b_in_interval (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ Set.Iic (-10) ∪ Set.Ici 10) := 
sorry

end quadratic_has_real_root_iff_b_in_interval_l289_289095


namespace find_n_l289_289675

open Nat

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def twin_primes (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ q = p + 2

def is_twins_prime_sum (n p q : ℕ) : Prop :=
  twin_primes p q ∧ is_prime (2^n + p) ∧ is_prime (2^n + q)

theorem find_n :
  ∀ (n : ℕ), (∃ (p q : ℕ), is_twins_prime_sum n p q) → (n = 1 ∨ n = 3) :=
sorry

end find_n_l289_289675


namespace find_second_offset_l289_289520

-- Define the given constants
def diagonal : ℝ := 30
def offset1 : ℝ := 10
def area : ℝ := 240

-- The theorem we want to prove
theorem find_second_offset : ∃ (offset2 : ℝ), area = (1 / 2) * diagonal * (offset1 + offset2) ∧ offset2 = 6 :=
sorry

end find_second_offset_l289_289520


namespace discriminant_of_quadratic_5x2_minus_2x_minus_7_l289_289194

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b ^ 2 - 4 * a * c

theorem discriminant_of_quadratic_5x2_minus_2x_minus_7 :
  quadratic_discriminant 5 (-2) (-7) = 144 :=
by
  sorry

end discriminant_of_quadratic_5x2_minus_2x_minus_7_l289_289194


namespace croissant_process_time_in_hours_l289_289574

-- Conditions as definitions
def num_folds : ℕ := 4
def fold_time : ℕ := 5
def rest_time : ℕ := 75
def mix_time : ℕ := 10
def bake_time : ℕ := 30

-- The main theorem statement
theorem croissant_process_time_in_hours :
  (num_folds * (fold_time + rest_time) + mix_time + bake_time) / 60 = 6 := 
sorry

end croissant_process_time_in_hours_l289_289574


namespace vector_addition_l289_289068

variable (a : ℝ × ℝ)
variable (b : ℝ × ℝ)

theorem vector_addition (h1 : a = (-1, 2)) (h2 : b = (1, 0)) :
  3 • a + b = (-2, 6) :=
by
  -- proof goes here
  sorry

end vector_addition_l289_289068


namespace constant_difference_of_equal_derivatives_l289_289078

theorem constant_difference_of_equal_derivatives
  {f g : ℝ → ℝ}
  (h : ∀ x, deriv f x = deriv g x) :
  ∃ C : ℝ, ∀ x, f x - g x = C := 
sorry

end constant_difference_of_equal_derivatives_l289_289078


namespace oranges_cost_l289_289939

def cost_for_multiple_dozens (price_per_dozen: ℝ) (dozens: ℝ) : ℝ := 
    price_per_dozen * dozens

theorem oranges_cost (price_for_4_dozens: ℝ) (price_for_5_dozens: ℝ) :
  price_for_4_dozens = 28.80 →
  price_for_5_dozens = cost_for_multiple_dozens (28.80 / 4) 5 →
  price_for_5_dozens = 36 :=
by
  intros h1 h2
  sorry

end oranges_cost_l289_289939


namespace average_age_after_person_leaves_l289_289333

theorem average_age_after_person_leaves
  (average_age_seven : ℕ := 28)
  (num_people_initial : ℕ := 7)
  (person_leaves : ℕ := 20) :
  (average_age_seven * num_people_initial - person_leaves) / (num_people_initial - 1) = 29 := by
  sorry

end average_age_after_person_leaves_l289_289333


namespace euler_totient_divisibility_l289_289486

theorem euler_totient_divisibility (a n: ℕ) (h1 : a ≥ 2) : (n ∣ Nat.totient (a^n - 1)) :=
sorry

end euler_totient_divisibility_l289_289486


namespace dan_initial_money_l289_289666

theorem dan_initial_money 
  (cost_chocolate : ℕ) 
  (cost_candy_bar : ℕ) 
  (h1 : cost_chocolate = 3) 
  (h2 : cost_candy_bar = 7)
  (h3 : cost_candy_bar - cost_chocolate = 4) : 
  cost_candy_bar + cost_chocolate = 10 := 
by
  sorry

end dan_initial_money_l289_289666


namespace complex_number_powers_l289_289007

theorem complex_number_powers (i : ℂ) (hi : i^2 = -1) : i + i^2 + i^3 = -1 :=
sorry

end complex_number_powers_l289_289007


namespace yield_difference_correct_l289_289028

noncomputable def tomato_yield (initial : ℝ) (growth_rate : ℝ) : ℝ := initial * (1 + growth_rate / 100)
noncomputable def corn_yield (initial : ℝ) (growth_rate : ℝ) : ℝ := initial * (1 + growth_rate / 100)
noncomputable def onion_yield (initial : ℝ) (growth_rate : ℝ) : ℝ := initial * (1 + growth_rate / 100)
noncomputable def carrot_yield (initial : ℝ) (growth_rate : ℝ) : ℝ := initial * (1 + growth_rate / 100)

theorem yield_difference_correct :
  let tomato_initial := 2073
  let corn_initial := 4112
  let onion_initial := 985
  let carrot_initial := 6250
  let tomato_growth := 12
  let corn_growth := 15
  let onion_growth := 8
  let carrot_growth := 10
  let tomato_total := tomato_yield tomato_initial tomato_growth
  let corn_total := corn_yield corn_initial corn_growth
  let onion_total := onion_yield onion_initial onion_growth
  let carrot_total := carrot_yield carrot_initial carrot_growth
  let highest_yield := max (max tomato_total corn_total) (max onion_total carrot_total)
  let lowest_yield := min (min tomato_total corn_total) (min onion_total carrot_total)
  highest_yield - lowest_yield = 5811.2 := by
  sorry

end yield_difference_correct_l289_289028


namespace james_total_oop_correct_l289_289558

-- Define the costs and insurance coverage percentages as given conditions.
def cost_consultation : ℝ := 300
def coverage_consultation : ℝ := 0.80

def cost_xray : ℝ := 150
def coverage_xray : ℝ := 0.70

def cost_prescription : ℝ := 75
def coverage_prescription : ℝ := 0.50

def cost_therapy : ℝ := 120
def coverage_therapy : ℝ := 0.60

-- Define the out-of-pocket calculation for each service
def oop_consultation := cost_consultation * (1 - coverage_consultation)
def oop_xray := cost_xray * (1 - coverage_xray)
def oop_prescription := cost_prescription * (1 - coverage_prescription)
def oop_therapy := cost_therapy * (1 - coverage_therapy)

-- Define the total out-of-pocket cost
def total_oop : ℝ := oop_consultation + oop_xray + oop_prescription + oop_therapy

-- Proof statement
theorem james_total_oop_correct : total_oop = 190.50 := by
  sorry

end james_total_oop_correct_l289_289558


namespace divisible_by_17_l289_289993

theorem divisible_by_17 (k : ℕ) : 17 ∣ (2^(2*k+3) + 3^(k+2) * 7^k) :=
  sorry

end divisible_by_17_l289_289993


namespace sum_of_cubes_of_consecutive_numbers_divisible_by_9_l289_289590

theorem sum_of_cubes_of_consecutive_numbers_divisible_by_9 (a : ℕ) (h : a > 1) : 
  9 ∣ ((a - 1)^3 + a^3 + (a + 1)^3) := 
by 
  sorry

end sum_of_cubes_of_consecutive_numbers_divisible_by_9_l289_289590


namespace average_large_basket_weight_l289_289015

-- Definitions derived from the conditions
def small_basket_capacity := 25  -- Capacity of each small basket in kilograms
def num_small_baskets := 28      -- Number of small baskets used
def num_large_baskets := 10      -- Number of large baskets used
def leftover_weight := 50        -- Leftover weight in kilograms

-- Statement of the problem
theorem average_large_basket_weight :
  (small_basket_capacity * num_small_baskets - leftover_weight) / num_large_baskets = 65 :=
by
  sorry

end average_large_basket_weight_l289_289015


namespace fishing_tomorrow_l289_289880

theorem fishing_tomorrow (seven_every_day eight_every_other_day three_every_three_days twelve_yesterday ten_today : ℕ)
  (h1 : seven_every_day = 7)
  (h2 : eight_every_other_day = 8)
  (h3 : three_every_three_days = 3)
  (h4 : twelve_yesterday = 12)
  (h5 : ten_today = 10) :
  (seven_every_day + (eight_every_other_day - (twelve_yesterday - seven_every_day)) + three_every_three_days) = 15 :=
by
  sorry

end fishing_tomorrow_l289_289880


namespace quadratic_real_roots_l289_289109

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := 
by 
  sorry

end quadratic_real_roots_l289_289109


namespace sum_of_three_numbers_l289_289930

theorem sum_of_three_numbers (a b c : ℕ) (mean_least difference greatest_diff : ℕ)
  (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : mean_least = 8) (h4 : greatest_diff = 25)
  (h5 : c - a = 26)
  (h6 : (a + b + c) / 3 = a + mean_least) 
  (h7 : (a + b + c) / 3 = c - greatest_diff) : 
a + b + c = 81 := 
sorry

end sum_of_three_numbers_l289_289930


namespace find_n_divides_2n_plus_2_l289_289677

theorem find_n_divides_2n_plus_2 :
  ∃ n : ℕ, (100 ≤ n ∧ n ≤ 1997 ∧ n ∣ (2 * n + 2)) ∧ n = 946 :=
by {
  sorry
}

end find_n_divides_2n_plus_2_l289_289677


namespace Tim_driving_hours_l289_289941

theorem Tim_driving_hours (D T : ℕ) (h1 : T = 2 * D) (h2 : D + T = 15) : D = 5 :=
by
  sorry

end Tim_driving_hours_l289_289941


namespace geometric_sequence_l289_289568

open Nat

-- Define the sequence and conditions for the problem
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {m p : ℕ}
variable (h1 : a 1 ≠ 0)
variable (h2 : ∀ n : ℕ, 2 * S (n + 1) - 3 * S n = 2 * a 1)
variable (h3 : S 0 = 0)
variable (h4 : ∀ n : ℕ, S (n + 1) = S n + a (n + 1))
variable (h5 : a 1 ≥ m^(p-1))
variable (h6 : a p ≤ (m+1)^(p-1))

-- The theorem that we need to prove
theorem geometric_sequence (n : ℕ) : 
  (exists r : ℕ → ℕ, ∀ k : ℕ, a (k + 1) = r (k + 1) * a k) ∧ 
  (∀ k : ℕ, a k = sorry) := sorry

end geometric_sequence_l289_289568


namespace deriv_y1_deriv_y2_deriv_y3_l289_289803

variable (x : ℝ)

-- Prove the derivative of y = 3x^3 - 4x is 9x^2 - 4
theorem deriv_y1 : deriv (λ x => 3 * x^3 - 4 * x) x = 9 * x^2 - 4 := by
sorry

-- Prove the derivative of y = (2x - 1)(3x + 2) is 12x + 1
theorem deriv_y2 : deriv (λ x => (2 * x - 1) * (3 * x + 2)) x = 12 * x + 1 := by
sorry

-- Prove the derivative of y = x^2 (x^3 - 4) is 5x^4 - 8x
theorem deriv_y3 : deriv (λ x => x^2 * (x^3 - 4)) x = 5 * x^4 - 8 * x := by
sorry


end deriv_y1_deriv_y2_deriv_y3_l289_289803


namespace total_sum_lent_l289_289798

noncomputable def interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem total_sum_lent 
  (x y : ℝ)
  (h1 : interest x (3 / 100) 5 = interest y (5 / 100) 3) 
  (h2 : y = 1332.5) : 
  x + y = 2665 :=
by
  -- We would continue the proof steps here.
  sorry

end total_sum_lent_l289_289798


namespace rectangle_area_ratio_l289_289748

noncomputable def area_ratio (s : ℝ) : ℝ :=
  let area_square := s^2
  let longer_side := 1.15 * s
  let shorter_side := 0.95 * s
  let area_rectangle := longer_side * shorter_side
  area_rectangle / area_square

theorem rectangle_area_ratio (s : ℝ) : area_ratio s = 109.25 / 100 := by
  sorry

end rectangle_area_ratio_l289_289748


namespace kennedy_softball_park_miles_l289_289561

theorem kennedy_softball_park_miles :
  let miles_per_gallon := 19
  let gallons_of_gas := 2
  let total_drivable_miles := miles_per_gallon * gallons_of_gas
  let miles_to_school := 15
  let miles_to_burger_restaurant := 2
  let miles_to_friends_house := 4
  let miles_home := 11
  total_drivable_miles - (miles_to_school + miles_to_burger_restaurant + miles_to_friends_house + miles_home) = 6 :=
by
  sorry

end kennedy_softball_park_miles_l289_289561


namespace minimum_value_expression_l289_289279

theorem minimum_value_expression {a b c : ℤ} (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  4 * (a^2 + b^2 + c^2) - (a + b + c)^2 = 8 := 
sorry

end minimum_value_expression_l289_289279


namespace identifyIncorrectGuess_l289_289466

-- Define the colors of the bears
inductive BearColor
| white
| brown
| black

-- Conditions as defined in the problem statement
def isValidBearRow (bears : Fin 1000 → BearColor) : Prop :=
  ∀ (i : Fin 998), 
    (bears i = BearColor.white ∨ bears i = BearColor.brown ∨ bears i = BearColor.black) ∧
    (bears ⟨i + 1, by linarith⟩ = BearColor.white ∨ bears ⟨i + 1, by linarith⟩ = BearColor.brown ∨ bears ⟨i + 1, by linarith⟩ = BearColor.black) ∧
    (bears ⟨i + 2, by linarith⟩ = BearColor.white ∨ bears ⟨i + 2, by linarith⟩ = BearColor.brown ∨ bears ⟨i + 2, by linarith⟩ = BearColor.black)

-- Iskander's guesses
def iskanderGuesses (bears : Fin 1000 → BearColor) : Prop :=
  bears 1 = BearColor.white ∧
  bears 19 = BearColor.brown ∧
  bears 399 = BearColor.black ∧
  bears 599 = BearColor.brown ∧
  bears 799 = BearColor.white

-- Exactly one guess is incorrect
def oneIncorrectGuess (bears : Fin 1000 → BearColor) : Prop :=
  ∃ (idx : Fin 5), 
    ¬iskanderGuesses bears ∧
    ∀ (j : Fin 5), (j ≠ idx → (bearGuessesIdx j bears = true))

-- The proof problem
theorem identifyIncorrectGuess (bears : Fin 1000 → BearColor) :
  isValidBearRow bears → iskanderGuesses bears → oneIncorrectGuess bears := sorry

end identifyIncorrectGuess_l289_289466


namespace peanut_butter_last_days_l289_289912

-- Definitions for the problem conditions
def daily_consumption : ℕ := 2
def servings_per_jar : ℕ := 15
def num_jars : ℕ := 4

-- The statement to prove
theorem peanut_butter_last_days : 
  (num_jars * servings_per_jar) / daily_consumption = 30 :=
by
  sorry

end peanut_butter_last_days_l289_289912


namespace power_quotient_example_l289_289808

theorem power_quotient_example (a : ℕ) (m n : ℕ) (h : 23^11 / 23^8 = 23^(11 - 8)) : 23^3 = 12167 := by
  sorry

end power_quotient_example_l289_289808


namespace fishing_tomorrow_l289_289886

-- Conditions
def every_day_fishers : Nat := 7
def every_other_day_fishers : Nat := 8
def every_three_days_fishers : Nat := 3
def yesterday_fishers : Nat := 12
def today_fishers : Nat := 10

-- Determine the number who will fish tomorrow
def fishers_tomorrow : Nat :=
  let every_day_tomorrow := every_day_fishers
  let every_three_day_tomorrow := every_three_days_fishers
  let every_other_day_yesterday := yesterday_fishers - every_day_fishers
  let every_other_day_tomorrow := every_other_day_fishers - every_other_day_yesterday
  every_day_tomorrow + every_three_day_tomorrow + every_other_day_tomorrow

theorem fishing_tomorrow : fishers_tomorrow = 15 :=
  by
    -- skipping the actual proof with sorry
    sorry

end fishing_tomorrow_l289_289886


namespace geometric_progression_condition_l289_289809

theorem geometric_progression_condition
  (a b k : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : k > 0)
  (a_seq : ℕ → ℝ) 
  (h_def : ∀ n, a_seq (n+2) = k * a_seq n * a_seq (n+1)) :
  (a_seq 1 = a ∧ a_seq 2 = b) ↔ a_seq 1 = a_seq 2 :=
by
  sorry

end geometric_progression_condition_l289_289809


namespace jill_study_hours_l289_289716

theorem jill_study_hours (x : ℕ) (h_condition : x + 2*x + (2*x - 1) = 9) : x = 2 :=
by
  sorry

end jill_study_hours_l289_289716


namespace units_digit_sum_factorials_l289_289989

theorem units_digit_sum_factorials : 
  (∑ n in finset.range 2011, (n.factorial % 10)) % 10 = 3 := 
by
  sorry

end units_digit_sum_factorials_l289_289989


namespace train_average_speed_with_stoppages_l289_289003

theorem train_average_speed_with_stoppages :
  (∀ d t_without_stops t_with_stops : ℝ, t_without_stops = d / 400 → 
  t_with_stops = d / (t_without_stops * (10/9)) → 
  t_with_stops = d / 360) :=
sorry

end train_average_speed_with_stoppages_l289_289003


namespace num_good_triples_at_least_l289_289155

noncomputable def num_good_triples (S : Finset (ℕ × ℕ)) (n m : ℕ) : ℕ :=
  4 * m * (m - n^2 / 4) / (3 * n)

theorem num_good_triples_at_least
  (S : Finset (ℕ × ℕ))
  (n m : ℕ)
  (h_S : ∀ (x : ℕ × ℕ), x ∈ S → 1 ≤ x.1 ∧ x.1 < x.2 ∧ x.2 ≤ n)
  (h_m : S.card = m)
  : ∃ t ≤ num_good_triples S n m, True := 
sorry

end num_good_triples_at_least_l289_289155


namespace negation_of_exists_l289_289931

theorem negation_of_exists {x : ℝ} :
  (¬ ∃ x : ℝ, x^2 - 2 ≤ 0) ↔ ∀ x : ℝ, x^2 - 2 > 0 :=
by
  sorry

end negation_of_exists_l289_289931


namespace solve_inequality_l289_289992

theorem solve_inequality (x : ℝ) : (2 ≤ |3 * x - 6| ∧ |3 * x - 6| ≤ 12) ↔ (x ∈ Set.Icc (-2 : ℝ) (4 / 3) ∨ x ∈ Set.Icc (8 / 3) (6 : ℝ)) :=
sorry

end solve_inequality_l289_289992


namespace find_multiple_sales_l289_289897

theorem find_multiple_sales 
  (A : ℝ) 
  (M : ℝ)
  (h : M * A = 0.35294117647058826 * (11 * A + M * A)) 
  : M = 6 :=
sorry

end find_multiple_sales_l289_289897


namespace monotonically_decreasing_when_a_half_l289_289424

noncomputable def f (x a : ℝ) : ℝ := x * (Real.log x - a * x)

theorem monotonically_decreasing_when_a_half :
  ∀ x : ℝ, 0 < x → (f x (1 / 2)) ≤ 0 :=
by
  sorry

end monotonically_decreasing_when_a_half_l289_289424


namespace difference_mean_median_l289_289302

theorem difference_mean_median :
  let percentage_scored_60 : ℚ := 0.20
  let percentage_scored_70 : ℚ := 0.30
  let percentage_scored_85 : ℚ := 0.25
  let percentage_scored_95 : ℚ := 1 - (percentage_scored_60 + percentage_scored_70 + percentage_scored_85)
  let score_60 : ℚ := 60
  let score_70 : ℚ := 70
  let score_85 : ℚ := 85
  let score_95 : ℚ := 95
  let mean : ℚ := percentage_scored_60 * score_60 + percentage_scored_70 * score_70 + percentage_scored_85 * score_85 + percentage_scored_95 * score_95
  let median : ℚ := 85
  (median - mean) = 7 := 
by 
  sorry

end difference_mean_median_l289_289302


namespace hcf_of_two_numbers_l289_289548

theorem hcf_of_two_numbers 
  (x y : ℕ) 
  (h1 : x + y = 45)
  (h2 : Nat.lcm x y = 120)
  (h3 : (1/x : ℚ) + (1/y : ℚ) = 11/120) : 
  Nat.gcd x y = 1 := 
sorry

end hcf_of_two_numbers_l289_289548


namespace determinant_of_given_matrix_l289_289258

-- Define the given matrix
def given_matrix (z : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![z + 2, z, z], ![z, z + 3, z], ![z, z, z + 4]]

-- Define the proof statement
theorem determinant_of_given_matrix (z : ℂ) : Matrix.det (given_matrix z) = 22 * z + 24 :=
by
  sorry

end determinant_of_given_matrix_l289_289258


namespace value_of_f_at_2_l289_289318

def f (x : ℝ) := x^2 + 2 * x - 3

theorem value_of_f_at_2 : f 2 = 5 :=
by
  sorry

end value_of_f_at_2_l289_289318


namespace factorization_proof_l289_289981

theorem factorization_proof (a : ℝ) : 2 * a^2 + 4 * a + 2 = 2 * (a + 1)^2 :=
by { sorry }

end factorization_proof_l289_289981


namespace coeff_x4_expansion_l289_289299

def binom_expansion (a : ℚ) : ℚ :=
  let term1 : ℚ := a * 28
  let term2 : ℚ := -56
  term1 + term2

theorem coeff_x4_expansion (a : ℚ) : (binom_expansion a = -42) → a = 1/2 := 
by 
  intro h
  -- continuation of proof will go here.
  sorry

end coeff_x4_expansion_l289_289299


namespace find_x_from_exponential_eq_l289_289274

theorem find_x_from_exponential_eq (x : ℕ) (h : 3^x + 3^x + 3^x + 3^x = 6561) : x = 6 := 
sorry

end find_x_from_exponential_eq_l289_289274


namespace trey_total_time_is_two_hours_l289_289582

-- Define the conditions
def num_cleaning_tasks := 7
def num_shower_tasks := 1
def num_dinner_tasks := 4
def time_per_task := 10 -- in minutes
def minutes_per_hour := 60

-- Total tasks
def total_tasks := num_cleaning_tasks + num_shower_tasks + num_dinner_tasks

-- Total time in minutes
def total_time_minutes := total_tasks * time_per_task

-- Total time in hours
def total_time_hours := total_time_minutes / minutes_per_hour

-- Prove that the total time Trey will need to complete his list is 2 hours
theorem trey_total_time_is_two_hours : total_time_hours = 2 := by
  sorry

end trey_total_time_is_two_hours_l289_289582


namespace four_integers_product_sum_l289_289689

theorem four_integers_product_sum (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
(h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_prod : a * b * c * d = 2002) (h_sum : a + b + c + d < 40) :
  (a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨ (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) :=
sorry

end four_integers_product_sum_l289_289689


namespace solve_quadratic_l289_289597

theorem solve_quadratic (x : ℝ) : x^2 - 6*x + 5 = 0 ↔ x = 1 ∨ x = 5 := by
  sorry

end solve_quadratic_l289_289597


namespace original_distance_cycled_l289_289794

theorem original_distance_cycled
  (x t d : ℝ)
  (h1 : d = x * t)
  (h2 : d = (x + 1/4) * (3/4 * t))
  (h3 : d = (x - 1/4) * (t + 3)) :
  d = 4.5 := 
sorry

end original_distance_cycled_l289_289794


namespace value_of_x_squared_plus_reciprocal_squared_l289_289846

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (h : 45 = x^4 + 1 / x^4) : 
  x^2 + 1 / x^2 = Real.sqrt 47 :=
by
  sorry

end value_of_x_squared_plus_reciprocal_squared_l289_289846


namespace fifth_iteration_perimeter_l289_289244

theorem fifth_iteration_perimeter :
  let A1_side_length := 1
  let P1 := 3 * A1_side_length
  let P2 := 3 * (A1_side_length * 4 / 3)
  ∀ n : ℕ, P_n = 3 * (4 / 3) ^ (n - 1) →
  P_5 = 3 * (4 / 3) ^ 4 :=
  by sorry

end fifth_iteration_perimeter_l289_289244


namespace geometric_series_sum_l289_289246

theorem geometric_series_sum :
  let a := (1 : ℚ) / 3
  let r := (-1 : ℚ) / 4
  let n := 6
  let S6 := a * (1 - r^n) / (1 - r)
  S6 = 4095 / 30720 :=
by
  sorry

end geometric_series_sum_l289_289246


namespace system_of_equations_solution_l289_289754

theorem system_of_equations_solution (x y : ℝ) 
  (h1 : x + 3 * y = 7) 
  (h2 : y = 2 * x) : 
  x = 1 ∧ y = 2 :=
by
  sorry

end system_of_equations_solution_l289_289754


namespace simplify_expression_l289_289735

theorem simplify_expression (x : ℝ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 :=
by
  sorry

end simplify_expression_l289_289735


namespace sqrt_sum_l289_289802

theorem sqrt_sum : (Real.sqrt 50) + (Real.sqrt 32) = 9 * (Real.sqrt 2) :=
by
  sorry

end sqrt_sum_l289_289802


namespace P_iff_nonQ_l289_289825

-- Given conditions
def P (x y : ℝ) : Prop := x^2 + y^2 = 0
def Q (x y : ℝ) : Prop := x ≠ 0 ∨ y ≠ 0
def nonQ (x y : ℝ) : Prop := x = 0 ∧ y = 0

-- Main statement
theorem P_iff_nonQ (x y : ℝ) : P x y ↔ nonQ x y :=
sorry

end P_iff_nonQ_l289_289825


namespace ribbon_per_gift_l289_289896

-- Definitions for the conditions in the problem
def total_ribbon_used : ℚ := 4/15
def num_gifts: ℕ := 5

-- Statement to prove
theorem ribbon_per_gift : total_ribbon_used / num_gifts = 4 / 75 :=
by
  sorry

end ribbon_per_gift_l289_289896


namespace sum_ages_in_five_years_l289_289323

theorem sum_ages_in_five_years (L J : ℕ) (hL : L = 13) (h_relation : L = 2 * J + 3) : 
  (L + 5) + (J + 5) = 28 := 
by 
  sorry

end sum_ages_in_five_years_l289_289323


namespace sum_of_first_K_natural_numbers_is_perfect_square_l289_289036

noncomputable def values_K (K : ℕ) : Prop := 
  ∃ N : ℕ, (K * (K + 1)) / 2 = N^2 ∧ (N + K < 120)

theorem sum_of_first_K_natural_numbers_is_perfect_square :
  ∀ K : ℕ, values_K K ↔ (K = 1 ∨ K = 8 ∨ K = 49) := by
  sorry

end sum_of_first_K_natural_numbers_is_perfect_square_l289_289036


namespace grapes_difference_l289_289507

theorem grapes_difference (R A_i A_l : ℕ) 
  (hR : R = 25) 
  (hAi : A_i = R + 2) 
  (hTotal : R + A_i + A_l = 83) : 
  A_l - A_i = 4 := 
by
  sorry

end grapes_difference_l289_289507


namespace nate_total_time_l289_289910

/-- Definitions for the conditions -/
def sectionG : ℕ := 18 * 12
def sectionH : ℕ := 25 * 10
def sectionI : ℕ := 17 * 11
def sectionJ : ℕ := 20 * 9
def sectionK : ℕ := 15 * 13

def speedGH : ℕ := 8
def speedIJ : ℕ := 10
def speedK : ℕ := 6

/-- Compute the time spent in each section, rounding up where necessary -/
def timeG : ℕ := (sectionG + speedGH - 1) / speedGH
def timeH : ℕ := (sectionH + speedGH - 1) / speedGH
def timeI : ℕ := (sectionI + speedIJ - 1) / speedIJ
def timeJ : ℕ := (sectionJ + speedIJ - 1) / speedIJ
def timeK : ℕ := (sectionK + speedK - 1) / speedK

/-- Compute the total time spent -/
def totalTime : ℕ := timeG + timeH + timeI + timeJ + timeK

/-- The proof statement -/
theorem nate_total_time : totalTime = 129 := by
  -- the proof goes here
  sorry

end nate_total_time_l289_289910


namespace household_waste_per_day_l289_289774

theorem household_waste_per_day (total_waste_4_weeks : ℝ) (h : total_waste_4_weeks = 30.8) : 
  (total_waste_4_weeks / 4 / 7) = 1.1 :=
by
  sorry

end household_waste_per_day_l289_289774


namespace LindasTrip_l289_289569

theorem LindasTrip (x : ℝ) :
    (1 / 4) * x + 30 + (1 / 6) * x = x →
    x = 360 / 7 :=
by
  intros h
  sorry

end LindasTrip_l289_289569


namespace james_total_points_l289_289177

def points_per_correct_answer : ℕ := 2
def bonus_points_per_round : ℕ := 4
def total_rounds : ℕ := 5
def questions_per_round : ℕ := 5
def total_questions : ℕ := total_rounds * questions_per_round
def questions_missed_by_james : ℕ := 1
def questions_answered_by_james : ℕ := total_questions - questions_missed_by_james
def points_for_correct_answers : ℕ := questions_answered_by_james * points_per_correct_answer
def complete_rounds_by_james : ℕ := total_rounds - 1  -- Since James missed one question, he has 4 complete rounds
def bonus_points_by_james : ℕ := complete_rounds_by_james * bonus_points_per_round
def total_points : ℕ := points_for_correct_answers + bonus_points_by_james

theorem james_total_points : total_points = 64 := by
  sorry

end james_total_points_l289_289177


namespace only_set_d_forms_triangle_l289_289206

/-- Definition of forming a triangle given three lengths -/
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem only_set_d_forms_triangle :
  ¬ can_form_triangle 3 5 10 ∧ ¬ can_form_triangle 5 4 9 ∧ 
  ¬ can_form_triangle 5 5 10 ∧ can_form_triangle 4 6 9 :=
by {
  sorry
}

end only_set_d_forms_triangle_l289_289206


namespace add_three_digits_l289_289655

theorem add_three_digits (x : ℕ) :
  (x = 152 ∨ x = 656) →
  (523000 + x) % 504 = 0 := 
by
  sorry

end add_three_digits_l289_289655


namespace abcd_sum_l289_289765

theorem abcd_sum : 
  ∃ (a b c d : ℕ), 
    (∃ x y : ℝ, x + y = 5 ∧ 2 * x * y = 6 ∧ 
      (x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d)) →
    a + b + c + d = 21 :=
by
  sorry

end abcd_sum_l289_289765


namespace monkey_slips_2_feet_each_hour_l289_289376

/-- 
  A monkey climbs a 17 ft tree, hopping 3 ft and slipping back a certain distance each hour.
  The monkey takes 15 hours to reach the top. Prove that the monkey slips back 2 feet each hour.
-/
def monkey_slips_back_distance (s : ℝ) : Prop :=
  ∃ s : ℝ, (14 * (3 - s) + 3 = 17) ∧ s = 2

theorem monkey_slips_2_feet_each_hour : monkey_slips_back_distance 2 := by
  -- Sorry, proof omitted
  sorry

end monkey_slips_2_feet_each_hour_l289_289376


namespace inscribed_squares_ratio_l289_289501

theorem inscribed_squares_ratio (x y : ℝ) (h1 : ∃ (x : ℝ), x * (13 * 12 + 13 * 5 - 5 * 12) = 60) 
  (h2 : ∃ (y : ℝ), 30 * y = 13 ^ 2) :
  x / y = 1800 / 2863 := 
sorry

end inscribed_squares_ratio_l289_289501


namespace find_f_comp_f_l289_289690

def f (x : ℚ) : ℚ :=
  if x ≤ 1 then x + 1 else -x + 3

theorem find_f_comp_f (h : f (f (5/2)) = 3/2) :
  f (f (5/2)) = 3/2 := by
  sorry

end find_f_comp_f_l289_289690


namespace pencil_cost_is_correct_l289_289471

-- Defining the cost of a pen as x and the cost of a pencil as y in cents
def cost_of_pen_and_pencil (x y : ℕ) : Prop :=
  3 * x + 5 * y = 345 ∧ 4 * x + 2 * y = 280

-- Stating the theorem that proves y = 39
theorem pencil_cost_is_correct (x y : ℕ) (h : cost_of_pen_and_pencil x y) : y = 39 :=
by
  sorry

end pencil_cost_is_correct_l289_289471


namespace sum_of_squares_l289_289985

theorem sum_of_squares (a b : ℝ) (h1 : (a + b) / 2 = 8) (h2 : Real.sqrt (a * b) = 2 * Real.sqrt 5) :
  a^2 + b^2 = 216 :=
by
  sorry

end sum_of_squares_l289_289985


namespace quadratic_real_roots_l289_289134

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := by 
  sorry

end quadratic_real_roots_l289_289134


namespace total_food_items_donated_l289_289818

def FosterFarmsDonation : ℕ := 45
def AmericanSummitsDonation : ℕ := 2 * FosterFarmsDonation
def HormelDonation : ℕ := 3 * FosterFarmsDonation
def BoudinButchersDonation : ℕ := HormelDonation / 3
def DelMonteFoodsDonation : ℕ := AmericanSummitsDonation - 30

theorem total_food_items_donated :
  FosterFarmsDonation + AmericanSummitsDonation + HormelDonation + BoudinButchersDonation + DelMonteFoodsDonation = 375 :=
by
  sorry

end total_food_items_donated_l289_289818


namespace Blossom_room_area_square_inches_l289_289758

def feet_to_inches (feet : ℕ) : ℕ := feet * 12

theorem Blossom_room_area_square_inches :
  (let length_feet := 10 in
   let width_feet := 10 in
   let length_inches := feet_to_inches length_feet in
   let width_inches := feet_to_inches width_feet in
   length_inches * width_inches = 14400) :=
by
  let length_feet := 10
  let width_feet := 10
  let length_inches := feet_to_inches length_feet
  let width_inches := feet_to_inches width_feet
  show length_inches * width_inches = 14400
  sorry

end Blossom_room_area_square_inches_l289_289758


namespace jessica_withdraw_fraq_l289_289895

theorem jessica_withdraw_fraq {B : ℝ} (h : B - 200 + (1 / 2) * (B - 200) = 450) :
  (200 / B) = 2 / 5 := by
  sorry

end jessica_withdraw_fraq_l289_289895


namespace pies_from_apples_l289_289363

theorem pies_from_apples 
  (initial_apples : ℕ) (handed_out_apples : ℕ) (apples_per_pie : ℕ) 
  (remaining_apples := initial_apples - handed_out_apples) 
  (pies := remaining_apples / apples_per_pie) 
  (h1 : initial_apples = 75) 
  (h2 : handed_out_apples = 19) 
  (h3 : apples_per_pie = 8) : 
  pies = 7 :=
by
  rw [h1, h2, h3]
  sorry

end pies_from_apples_l289_289363


namespace one_third_percent_of_150_l289_289945

theorem one_third_percent_of_150 : (1/3) * (150 / 100) = 0.5 := by
  sorry

end one_third_percent_of_150_l289_289945


namespace fishing_tomorrow_l289_289877

theorem fishing_tomorrow (seven_every_day eight_every_other_day three_every_three_days twelve_yesterday ten_today : ℕ)
  (h1 : seven_every_day = 7)
  (h2 : eight_every_other_day = 8)
  (h3 : three_every_three_days = 3)
  (h4 : twelve_yesterday = 12)
  (h5 : ten_today = 10) :
  (seven_every_day + (eight_every_other_day - (twelve_yesterday - seven_every_day)) + three_every_three_days) = 15 :=
by
  sorry

end fishing_tomorrow_l289_289877


namespace linear_function_value_l289_289902

theorem linear_function_value (g : ℝ → ℝ) (h_linear : ∀ x y, g (x + y) = g x + g y)
  (h_scale : ∀ c x, g (c * x) = c * g x) (h : g 10 - g 0 = 20) : g 20 - g 0 = 40 :=
by
  sorry

end linear_function_value_l289_289902


namespace union_A_B_intersection_complement_A_B_l289_289535

open Set Real

noncomputable def A : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}
noncomputable def B : Set ℝ := {x : ℝ | abs (2 * x + 1) ≤ 1}

theorem union_A_B : A ∪ B = {x : ℝ | x ≤ 0 ∨ x ≥ 2} := by
  sorry

theorem intersection_complement_A_B : (Aᶜ) ∩ (Bᶜ) = {x : ℝ | 0 < x ∧ x < 2} := by
  sorry

end union_A_B_intersection_complement_A_B_l289_289535


namespace trajectory_parabola_l289_289719

noncomputable def otimes (x1 x2 : ℝ) : ℝ := (x1 + x2)^2 - (x1 - x2)^2

theorem trajectory_parabola (x : ℝ) (h : 0 ≤ x) : 
  ∃ (y : ℝ), y^2 = 8 * x ∧ (∀ P : ℝ × ℝ, P = (x, y) → (P.snd^2 = 8 * P.fst)) :=
by
  sorry

end trajectory_parabola_l289_289719


namespace prove_county_growth_condition_l289_289306

variable (x : ℝ)
variable (investment2014 : ℝ) (investment2016 : ℝ)

def county_growth_condition
  (h1 : investment2014 = 2500)
  (h2 : investment2016 = 3500) : Prop :=
  investment2014 * (1 + x)^2 = investment2016

theorem prove_county_growth_condition
  (x : ℝ)
  (h1 : investment2014 = 2500)
  (h2 : investment2016 = 3500) : county_growth_condition x investment2014 investment2016 h1 h2 :=
by
  sorry

end prove_county_growth_condition_l289_289306


namespace part1_part2_l289_289066

noncomputable def f (x : ℝ) : ℝ := |x + 1| + 2 * |x - 1|

theorem part1 (x : ℝ) : f x < 4 ↔ -1 < x ∧ x < (5:ℝ)/3 := by
  sorry

theorem part2 (a : ℝ) : (∀ x, f x ≥ |a + 1|) ↔ -3 ≤ a ∧ a ≤ 1 := by
  sorry

end part1_part2_l289_289066


namespace find_cost_per_batch_l289_289011

noncomputable def cost_per_tire : ℝ := 8
noncomputable def selling_price_per_tire : ℝ := 20
noncomputable def profit_per_tire : ℝ := 10.5
noncomputable def number_of_tires : ℕ := 15000

noncomputable def total_cost (C : ℝ) : ℝ := C + cost_per_tire * number_of_tires
noncomputable def total_revenue : ℝ := selling_price_per_tire * number_of_tires
noncomputable def total_profit : ℝ := profit_per_tire * number_of_tires

theorem find_cost_per_batch (C : ℝ) :
  total_profit = total_revenue - total_cost C → C = 22500 := by
  sorry

end find_cost_per_batch_l289_289011


namespace product_of_square_roots_l289_289293

theorem product_of_square_roots (a b : ℝ) (h₁ : a^2 = 9) (h₂ : b^2 = 9) (h₃ : a ≠ b) : a * b = -9 :=
by
  -- Proof skipped
  sorry

end product_of_square_roots_l289_289293


namespace calc_expression_l289_289250

theorem calc_expression :
  (12^4 + 375) * (24^4 + 375) * (36^4 + 375) * (48^4 + 375) * (60^4 + 375) /
  ((6^4 + 375) * (18^4 + 375) * (30^4 + 375) * (42^4 + 375) * (54^4 + 375)) = 159 :=
by
  sorry

end calc_expression_l289_289250


namespace average_weight_increase_l289_289777

theorem average_weight_increase
  (initial_weight replaced_weight : ℝ)
  (num_persons : ℕ)
  (avg_increase : ℝ)
  (h₁ : num_persons = 5)
  (h₂ : replaced_weight = 65)
  (h₃ : avg_increase = 1.5)
  (total_increase : ℝ)
  (new_weight : ℝ)
  (h₄ : total_increase = num_persons * avg_increase)
  (h₅ : total_increase = new_weight - replaced_weight) :
  new_weight = 72.5 :=
by
  sorry

end average_weight_increase_l289_289777


namespace problem_quadratic_has_real_root_l289_289119

theorem problem_quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end problem_quadratic_has_real_root_l289_289119


namespace problem_1_problem_2_l289_289427

-- Definitions for the sets A and B:

def set_A : Set ℝ := { x | x^2 - x - 12 ≤ 0 }
def set_B (m : ℝ) : Set ℝ := { x | 2 * m - 1 < x ∧ x < 1 + m }

-- Problem 1: When m = -2, find A ∪ B
theorem problem_1 : set_A ∪ set_B (-2) = { x | -5 < x ∧ x ≤ 4 } :=
sorry

-- Problem 2: If A ∩ B = B, find the range of the real number m
theorem problem_2 : (∀ x, x ∈ set_B m → x ∈ set_A) ↔ m ≥ -1 :=
sorry

end problem_1_problem_2_l289_289427


namespace area_of_room_in_square_inches_l289_289759

-- Defining the conversion from feet to inches
def feet_to_inches (ft : ℕ) : ℕ := ft * 12

-- Given conditions
def length_in_feet : ℕ := 10
def width_in_feet : ℕ := 10

-- Calculate length and width in inches
def length_in_inches := feet_to_inches length_in_feet
def width_in_inches := feet_to_inches width_in_feet

-- Calculate area in square inches
def area_in_square_inches := length_in_inches * width_in_inches

-- Theorem statement
theorem area_of_room_in_square_inches
  (h1 : length_in_feet = 10)
  (h2 : width_in_feet = 10)
  (conversion : feet_to_inches 1 = 12) :
  area_in_square_inches = 14400 :=
sorry

end area_of_room_in_square_inches_l289_289759


namespace area_of_circumscribed_circle_l289_289645

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  ∃ A : ℝ, A = 48 * Real.pi := 
by
  sorry

end area_of_circumscribed_circle_l289_289645


namespace arithmetic_sequence_geometric_sequence_l289_289487

-- Problem 1
theorem arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) (Sₙ : ℝ) 
  (h₁ : a₁ = 3 / 2) (h₂ : d = -1 / 2) (h₃ : Sₙ = -15) :
  n = 12 ∧ (a₁ + (n - 1) * d) = -4 := 
sorry

-- Problem 2
theorem geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) (aₙ Sₙ : ℝ) 
  (h₁ : q = 2) (h₂ : aₙ = 96) (h₃ : Sₙ = 189) :
  a₁ = 3 ∧ n = 6 := 
sorry

end arithmetic_sequence_geometric_sequence_l289_289487


namespace part_a_l289_289217

theorem part_a (a b c : ℝ) (m : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hm : 0 < m) :
  (a + b)^m + (b + c)^m + (c + a)^m ≤ 2^m * (a^m + b^m + c^m) :=
by
  sorry

end part_a_l289_289217


namespace equilateral_triangle_area_decrease_l289_289385

theorem equilateral_triangle_area_decrease :
  let original_area : ℝ := 100 * Real.sqrt 3
  let side_length_s := 20
  let decreased_side_length := side_length_s - 6
  let new_area := (decreased_side_length * decreased_side_length * Real.sqrt 3) / 4
  let decrease_in_area := original_area - new_area
  decrease_in_area = 51 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_decrease_l289_289385


namespace jen_scored_more_l289_289388

def bryan_score : ℕ := 20
def total_points : ℕ := 35
def sammy_mistakes : ℕ := 7
def sammy_score : ℕ := total_points - sammy_mistakes
def jen_score : ℕ := sammy_score + 2

theorem jen_scored_more :
  jen_score - bryan_score = 10 := by
  -- Proof to be filled in
  sorry

end jen_scored_more_l289_289388


namespace quadratic_has_real_root_l289_289122

theorem quadratic_has_real_root (b : ℝ) : 
  (b^2 - 100 ≥ 0) ↔ (b ≤ -10 ∨ b ≥ 10) :=
by
  sorry

end quadratic_has_real_root_l289_289122


namespace cos_value_proof_l289_289275

variable (α : Real)
variable (h1 : -Real.pi / 2 < α ∧ α < 0)
variable (h2 : Real.sin (α + Real.pi / 3) + Real.sin α = -(4 * Real.sqrt 3) / 5)

theorem cos_value_proof : Real.cos (α + 2 * Real.pi / 3) = 4 / 5 :=
by
  sorry

end cos_value_proof_l289_289275


namespace quadratic_real_root_iff_b_range_l289_289107

open Real

theorem quadratic_real_root_iff_b_range (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_real_root_iff_b_range_l289_289107


namespace three_digit_number_divisible_by_eleven_l289_289480

theorem three_digit_number_divisible_by_eleven
  (x : ℕ) (n : ℕ)
  (units_digit_is_two : n % 10 = 2)
  (hundreds_digit_is_seven : n / 100 = 7)
  (tens_digit : n = 700 + x * 10 + 2)
  (divisibility_condition : (7 - x + 2) % 11 = 0) :
  n = 792 := by
  sorry

end three_digit_number_divisible_by_eleven_l289_289480


namespace intersection_of_A_and_B_l289_289694

def A : Set ℤ := {-1, 0, 3, 5}
def B : Set ℤ := {x | x - 2 > 0}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} := 
by 
  sorry

end intersection_of_A_and_B_l289_289694


namespace sum_of_reciprocals_of_roots_l289_289409

theorem sum_of_reciprocals_of_roots (s₁ s₂ : ℝ) (h₀ : s₁ + s₂ = 15) (h₁ : s₁ * s₂ = 36) :
  (1 / s₁) + (1 / s₂) = 5 / 12 :=
by
  sorry

end sum_of_reciprocals_of_roots_l289_289409


namespace inequality_solution_l289_289600

theorem inequality_solution (x : ℝ) (h : 3 * x + 2 ≠ 0) : 
  3 - 2/(3 * x + 2) < 5 ↔ x > -2/3 := 
sorry

end inequality_solution_l289_289600


namespace original_population_multiple_of_3_l289_289188

theorem original_population_multiple_of_3 (x y z : ℕ) (h1 : x^2 + 121 = y^2) (h2 : y^2 + 121 = z^2) :
  3 ∣ x^2 :=
sorry

end original_population_multiple_of_3_l289_289188


namespace infinite_integer_triples_solution_l289_289916

theorem infinite_integer_triples_solution (a b c : ℤ) : 
  ∃ (a b c : ℤ), ∀ n : ℤ, a^2 + b^2 = c^2 + 3 :=
sorry

end infinite_integer_triples_solution_l289_289916


namespace horner_value_at_2_l289_289955

noncomputable def f (x : ℝ) := 2 * x^5 - 3 * x^3 + 2 * x^2 + x - 3

theorem horner_value_at_2 : f 2 = 12 := sorry

end horner_value_at_2_l289_289955


namespace compare_powers_l289_289205

theorem compare_powers :
  100^100 > 50^50 * 150^50 := sorry

end compare_powers_l289_289205


namespace quadratic_roots_interval_l289_289093

theorem quadratic_roots_interval (b : ℝ) :
  ∃ (x : ℝ), x^2 + b * x + 25 = 0 → b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_roots_interval_l289_289093


namespace number_of_handshakes_l289_289661

-- Define the context of the problem
def total_women := 8
def teams (n : Nat) := 4

-- Define the number of people each woman will shake hands with (excluding her partner)
def handshakes_per_woman := total_women - 2

-- Define the total number of handshakes
def total_handshakes := (total_women * handshakes_per_woman) / 2

-- The theorem that we're to prove
theorem number_of_handshakes : total_handshakes = 24 :=
by
  sorry

end number_of_handshakes_l289_289661


namespace expression_evaluation_correct_l289_289673

theorem expression_evaluation_correct (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
  ( ( ( (x - 2) ^ 2 * (x ^ 2 + x + 1) ^ 2 ) / (x ^ 3 - 1) ^ 2 ) ^ 2 *
    ( ( (x + 2) ^ 2 * (x ^ 2 - x + 1) ^ 2 ) / (x ^ 3 + 1) ^ 2 ) ^ 2 ) 
  = (x^2 - 4)^4 := 
sorry

end expression_evaluation_correct_l289_289673


namespace sum_of_cubes_inequality_l289_289721

theorem sum_of_cubes_inequality (a b c : ℝ) (h1 : a >= -1) (h2 : b >= -1) (h3 : c >= -1) (h4 : a^3 + b^3 + c^3 = 1) :
  a + b + c + a^2 + b^2 + c^2 <= 4 := 
sorry

end sum_of_cubes_inequality_l289_289721


namespace find_principal_l289_289950

theorem find_principal :
  ∃ P r : ℝ, (8820 = P * (1 + r) ^ 2) ∧ (9261 = P * (1 + r) ^ 3) → (P = 8000) :=
by
  sorry

end find_principal_l289_289950


namespace smallest_perfect_square_divisible_by_2_and_5_l289_289201

theorem smallest_perfect_square_divisible_by_2_and_5 :
  ∃ n : ℕ, 0 < n ∧ (∃ k : ℕ, n = k ^ 2) ∧ (n % 2 = 0) ∧ (n % 5 = 0) ∧ 
  (∀ m : ℕ, 0 < m ∧ (∃ k : ℕ, m = k ^ 2) ∧ (m % 2 = 0) ∧ (m % 5 = 0) → n ≤ m) :=
sorry

end smallest_perfect_square_divisible_by_2_and_5_l289_289201


namespace milkman_cows_l289_289785

theorem milkman_cows (x : ℕ) (c : ℕ) :
  (3 * x * c = 720) ∧ (3 * x * c + 50 * c + 140 * c + 63 * c = 3250) → x = 24 :=
by
  sorry

end milkman_cows_l289_289785


namespace inverse_proportion_function_increasing_l289_289729

theorem inverse_proportion_function_increasing (m : ℝ) :
  (∀ x1 x2 : ℝ, (0 < x1) → (x1 < x2) → (y = (m - 5) / x1) < (y = (m - 5) / x2)) ↔ m < 5 :=
by
  sorry

end inverse_proportion_function_increasing_l289_289729


namespace length_GH_l289_289458

def length_AB : ℕ := 11
def length_FE : ℕ := 13
def length_CD : ℕ := 5

theorem length_GH : length_AB + length_CD + length_FE = 29 :=
by
  refine rfl -- This will unroll the constants and perform arithmetic

end length_GH_l289_289458


namespace find_x_l289_289781

theorem find_x (x : ℝ) : 
  0.65 * x = 0.20 * 682.50 → x = 210 := 
by 
  sorry

end find_x_l289_289781


namespace cos_alpha_add_beta_over_two_l289_289537

theorem cos_alpha_add_beta_over_two (
  α β : ℝ) 
  (h1 : 0 < α ∧ α < (Real.pi / 2)) 
  (h2 : - (Real.pi / 2) < β ∧ β < 0) 
  (hcos1 : Real.cos (α + (Real.pi / 4)) = 1 / 3) 
  (hcos2 : Real.cos ((β / 2) - (Real.pi / 4)) = Real.sqrt 3 / 3) : 
  Real.cos (α + β / 2) = 5 * Real.sqrt 3 / 9 :=
sorry

end cos_alpha_add_beta_over_two_l289_289537


namespace fishing_tomorrow_l289_289878

theorem fishing_tomorrow (seven_every_day eight_every_other_day three_every_three_days twelve_yesterday ten_today : ℕ)
  (h1 : seven_every_day = 7)
  (h2 : eight_every_other_day = 8)
  (h3 : three_every_three_days = 3)
  (h4 : twelve_yesterday = 12)
  (h5 : ten_today = 10) :
  (seven_every_day + (eight_every_other_day - (twelve_yesterday - seven_every_day)) + three_every_three_days) = 15 :=
by
  sorry

end fishing_tomorrow_l289_289878


namespace find_initial_shells_l289_289314

theorem find_initial_shells (x : ℕ) (h : x + 23 = 28) : x = 5 :=
by
  sorry

end find_initial_shells_l289_289314


namespace debby_deleted_pictures_l289_289773

theorem debby_deleted_pictures :
  ∀ (zoo_pics museum_pics remaining_pics : ℕ), 
  zoo_pics = 24 →
  museum_pics = 12 →
  remaining_pics = 22 →
  (zoo_pics + museum_pics) - remaining_pics = 14 :=
by
  intros zoo_pics museum_pics remaining_pics hz hm hr
  sorry

end debby_deleted_pictures_l289_289773


namespace amount_subtracted_is_30_l289_289223

-- Definitions based on conditions
def N : ℕ := 200
def subtracted_amount (A : ℕ) : Prop := 0.40 * (N : ℝ) - (A : ℝ) = 50

-- The theorem statement
theorem amount_subtracted_is_30 : subtracted_amount 30 :=
by 
  -- proof will be completed here
  sorry

end amount_subtracted_is_30_l289_289223


namespace probability_of_event_A_l289_289956

def total_balls : ℕ := 10
def white_balls : ℕ := 7
def black_balls : ℕ := 3

def event_A : Prop := (black_balls / total_balls) * (white_balls / (total_balls - 1)) = 7 / 30

theorem probability_of_event_A : event_A := by
  sorry

end probability_of_event_A_l289_289956


namespace initial_paint_amount_l289_289439

theorem initial_paint_amount (P : ℝ) (h1 : P > 0) (h2 : (1 / 4) * P + (1 / 3) * (3 / 4) * P = 180) : P = 360 := by
  sorry

end initial_paint_amount_l289_289439


namespace cristina_pace_correct_l289_289578

-- Definitions of the conditions
def head_start : ℕ := 30
def nicky_pace : ℕ := 3  -- meters per second
def time_for_catch_up : ℕ := 15  -- seconds

-- Distance covers by Nicky
def nicky_distance : ℕ := nicky_pace * time_for_catch_up

-- Total distance covered by Cristina to catch up Nicky
def cristina_distance : ℕ := nicky_distance + head_start

-- Cristina's pace
def cristina_pace : ℕ := cristina_distance / time_for_catch_up

-- Theorem statement
theorem cristina_pace_correct : cristina_pace = 5 := by 
  sorry

end cristina_pace_correct_l289_289578


namespace continuous_tape_length_l289_289982

theorem continuous_tape_length :
  let num_sheets := 15
  let sheet_length_cm := 25
  let overlap_cm := 0.5 
  let total_length_without_overlap := num_sheets * sheet_length_cm
  let num_overlaps := num_sheets - 1
  let total_overlap_length := num_overlaps * overlap_cm
  let total_length_cm := total_length_without_overlap - total_overlap_length
  let total_length_m := total_length_cm / 100
  total_length_m = 3.68 := 
by {
  sorry
}

end continuous_tape_length_l289_289982


namespace chen_steps_recorded_correct_l289_289630

-- Define the standard for steps per day
def standard : ℕ := 5000

-- Define the steps walked by Xia
def xia_steps : ℕ := 6200

-- Define the recorded steps for Xia
def xia_recorded : ℤ := xia_steps - standard

-- Assert that Xia's recorded steps are +1200
lemma xia_steps_recorded_correct : xia_recorded = 1200 := by
  sorry

-- Define the steps walked by Chen
def chen_steps : ℕ := 4800

-- Define the recorded steps for Chen
def chen_recorded : ℤ := standard - chen_steps

-- State and prove that Chen's recorded steps are -200
theorem chen_steps_recorded_correct : chen_recorded = -200 :=
  sorry

end chen_steps_recorded_correct_l289_289630


namespace tangent_line_intercept_l289_289472

theorem tangent_line_intercept:
  ∃ (m b : ℚ), 
    m > 0 ∧ 
    b = 135 / 28 ∧ 
    (∀ x y : ℚ, (y - 3)^2 + (x - 1)^2 ≥ 3^2 → (y - 8)^2 + (x - 10)^2 ≥ 6^2 → y = m * x + b) := 
sorry

end tangent_line_intercept_l289_289472


namespace evaluate_g_neg5_l289_289432

def g (x : ℝ) : ℝ := 4 * x - 2

theorem evaluate_g_neg5 : g (-5) = -22 := 
  by sorry

end evaluate_g_neg5_l289_289432


namespace inclination_angle_of_line_l289_289929

open Real

theorem inclination_angle_of_line (x y : ℝ) (h : x + y - 3 = 0) : 
  ∃ θ : ℝ, θ = 3 * π / 4 :=
by
  sorry

end inclination_angle_of_line_l289_289929


namespace possible_numbers_erased_one_digit_reduce_sixfold_l289_289793

theorem possible_numbers_erased_one_digit_reduce_sixfold (N : ℕ) :
  (∃ N' : ℕ, N = 6 * N' ∧ N % 10 ≠ 0 ∧ ¬N = N') ↔
  N = 12 ∨ N = 24 ∨ N = 36 ∨ N = 48 ∨ N = 108 :=
by {
  sorry
}

end possible_numbers_erased_one_digit_reduce_sixfold_l289_289793


namespace range_of_a_l289_289693

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l289_289693


namespace simplify_trig_expr_l289_289169

theorem simplify_trig_expr : 
  let θ := 60
  let tan_θ := Real.sqrt 3
  let cot_θ := (Real.sqrt 3)⁻¹
  (tan_θ^3 + cot_θ^3) / (tan_θ + cot_θ) = 7 / 3 :=
by
  sorry

end simplify_trig_expr_l289_289169


namespace problem_quadratic_has_real_root_l289_289120

theorem problem_quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end problem_quadratic_has_real_root_l289_289120


namespace chess_tournament_games_l289_289847

theorem chess_tournament_games (n : ℕ) (h : n * (n - 1) / 2 = 190) : n = 20 :=
sorry

end chess_tournament_games_l289_289847


namespace evaluate_expression_l289_289040

theorem evaluate_expression :
  (∃ (a b c : ℕ), a = 18 ∧ b = 3 ∧ c = 54 ∧ c = a * b ∧ (18^36 / 54^18) = (6^18)) :=
sorry

end evaluate_expression_l289_289040


namespace mostWaterIntake_l289_289517

noncomputable def dailyWaterIntakeDongguk : ℝ := 5 * 0.2 -- Total water intake in liters per day for Dongguk
noncomputable def dailyWaterIntakeYoonji : ℝ := 6 * 0.3 -- Total water intake in liters per day for Yoonji
noncomputable def dailyWaterIntakeHeejin : ℝ := 4 * 500 / 1000 -- Total water intake in liters per day for Heejin (converted from milliliters)

theorem mostWaterIntake :
  dailyWaterIntakeHeejin = max dailyWaterIntakeDongguk (max dailyWaterIntakeYoonji dailyWaterIntakeHeejin) :=
by
  sorry

end mostWaterIntake_l289_289517


namespace quadratic_has_real_root_l289_289123

theorem quadratic_has_real_root (b : ℝ) : 
  (b^2 - 100 ≥ 0) ↔ (b ≤ -10 ∨ b ≥ 10) :=
by
  sorry

end quadratic_has_real_root_l289_289123


namespace depth_multiple_of_rons_height_l289_289457

theorem depth_multiple_of_rons_height (h d : ℕ) (Ron_height : h = 13) (water_depth : d = 208) : d = 16 * h := by
  sorry

end depth_multiple_of_rons_height_l289_289457


namespace james_total_points_l289_289178

def points_per_correct_answer : ℕ := 2
def bonus_points_per_round : ℕ := 4
def total_rounds : ℕ := 5
def questions_per_round : ℕ := 5
def total_questions : ℕ := total_rounds * questions_per_round
def questions_missed_by_james : ℕ := 1
def questions_answered_by_james : ℕ := total_questions - questions_missed_by_james
def points_for_correct_answers : ℕ := questions_answered_by_james * points_per_correct_answer
def complete_rounds_by_james : ℕ := total_rounds - 1  -- Since James missed one question, he has 4 complete rounds
def bonus_points_by_james : ℕ := complete_rounds_by_james * bonus_points_per_round
def total_points : ℕ := points_for_correct_answers + bonus_points_by_james

theorem james_total_points : total_points = 64 := by
  sorry

end james_total_points_l289_289178


namespace find_x_ge_0_l289_289404

-- Defining the condition and the proof problem
theorem find_x_ge_0 :
  {x : ℝ | (x^2 + 2*x^4 - 3*x^5) / (x + 2*x^3 - 3*x^4) ≥ 0} = {x : ℝ | 0 ≤ x} :=
by
  sorry -- proof steps not included

end find_x_ge_0_l289_289404


namespace smallest_irreducible_l289_289528

def is_irreducible (n : ℕ) : Prop :=
  ∀ k : ℕ, 19 ≤ k ∧ k ≤ 91 → Nat.gcd k (n + k + 2) = 1

theorem smallest_irreducible : ∃ n : ℕ, is_irreducible n ∧ ∀ m : ℕ, m < n → ¬ is_irreducible m :=
  by
  exists 95
  sorry

end smallest_irreducible_l289_289528


namespace minimum_value_inequality_l289_289295

theorem minimum_value_inequality {a b : ℝ} (h1 : b > a) (h2 : a > 1) 
  (h3 : 3 * Real.log b / Real.log a + 2 * Real.log a / Real.log b = 7) :
  a^2 + 3 / (b - 1) ≥ 2 * Real.sqrt 3 + 1 :=
sorry

end minimum_value_inequality_l289_289295


namespace age_ratio_l289_289547

theorem age_ratio (B A : ℕ) (h1 : B = 4) (h2 : A - B = 12) :
  A / B = 4 :=
by
  sorry

end age_ratio_l289_289547


namespace roots_abs_gt_4_or_l289_289698

theorem roots_abs_gt_4_or
    (r1 r2 : ℝ)
    (q : ℝ) 
    (h1 : r1 ≠ r2)
    (h2 : r1 + r2 = -q)
    (h3 : r1 * r2 = -10) :
    |r1| > 4 ∨ |r2| > 4 :=
sorry

end roots_abs_gt_4_or_l289_289698


namespace susie_investment_l289_289923

theorem susie_investment :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2000 ∧
  (x * 1.04 + (2000 - x) * 1.06 = 2120) → (x = 0) :=
by
  sorry

end susie_investment_l289_289923


namespace hank_newspaper_reading_time_l289_289429

theorem hank_newspaper_reading_time
  (n_days_weekday : ℕ := 5)
  (novel_reading_time_weekday : ℕ := 60)
  (n_days_weekend : ℕ := 2)
  (total_weekly_reading_time : ℕ := 810)
  (x : ℕ)
  (h1 : n_days_weekday * x + n_days_weekday * novel_reading_time_weekday +
        n_days_weekend * 2 * x + n_days_weekend * 2 * novel_reading_time_weekday = total_weekly_reading_time) :
  x = 30 := 
by {
  sorry -- Proof would go here
}

end hank_newspaper_reading_time_l289_289429


namespace complex_div_l289_289391

theorem complex_div (i : ℂ) (hi : i^2 = -1) : (1 + i) / i = 1 - i := by
  sorry

end complex_div_l289_289391


namespace find_x_when_y_is_10_l289_289281

-- Definitions of inverse proportionality and initial conditions
def inversely_proportional (x y : ℝ) (k : ℝ) : Prop := x * y = k

-- Given constants
def k : ℝ := 160
def x_initial : ℝ := 40
def y_initial : ℝ := 4

-- Theorem statement to prove the value of x when y = 10
theorem find_x_when_y_is_10 (h : inversely_proportional x_initial y_initial k) : 
  ∃ (x : ℝ), inversely_proportional x 10 k :=
sorry

end find_x_when_y_is_10_l289_289281


namespace find_room_width_l289_289335

noncomputable def unknown_dimension (total_cost per_sqft_cost area_excluding_openings room_height room_length width_wall1 : ℝ) : ℝ :=
  (total_cost - area_excluding_openings * per_sqft_cost) / (2 * room_height * per_sqft_cost) - width_wall1

theorem find_room_width (x : ℝ) :
  let door_area : ℝ := 6 * 3
      window_area : ℝ := 3 * (4 * 3)
      room_height : ℝ := 12
      room_length : ℝ := 25
      total_area : ℝ := 2 * (room_length * room_height) + 2 * (x * room_height)
      area_excluding_openings : ℝ := total_area - door_area - window_area
      per_sqft_cost : ℝ := 3
      total_cost : ℝ := 2718
  in unknown_dimension total_cost per_sqft_cost area_excluding_openings room_height room_length 0 = 15 :=
by
  -- Here we would have the proof steps, but we use 'sorry' as required.
  sorry

end find_room_width_l289_289335


namespace ratio_of_other_triangle_to_square_l289_289017

noncomputable def ratio_of_triangle_areas (m : ℝ) : ℝ :=
  let side_of_square := 2
  let area_of_square := side_of_square ^ 2
  let area_of_smaller_triangle := m * area_of_square
  let r := area_of_smaller_triangle / (side_of_square / 2)
  let s := side_of_square * side_of_square / r
  let area_of_other_triangle := side_of_square * s / 2
  area_of_other_triangle / area_of_square

theorem ratio_of_other_triangle_to_square (m : ℝ) (h : m > 0) :
  ratio_of_triangle_areas m = 1 / (4 * m) :=
sorry

end ratio_of_other_triangle_to_square_l289_289017


namespace sum_of_x_coordinates_mod_20_l289_289161

theorem sum_of_x_coordinates_mod_20 (y x : ℤ) (h1 : y ≡ 7 * x + 3 [ZMOD 20]) (h2 : y ≡ 13 * x + 17 [ZMOD 20]) 
: ∃ (x1 x2 : ℤ), (0 ≤ x1 ∧ x1 < 20) ∧ (0 ≤ x2 ∧ x2 < 20) ∧ x1 ≡ 1 [ZMOD 10] ∧ x2 ≡ 11 [ZMOD 10] ∧ x1 + x2 = 12 := sorry

end sum_of_x_coordinates_mod_20_l289_289161


namespace find_f_2_l289_289442

theorem find_f_2 (f : ℝ → ℝ) (h₁ : f 1 = 0)
  (h₂ : ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x + f y)) :
  f 2 = 0 :=
sorry

end find_f_2_l289_289442


namespace change_of_b_l289_289925

variable {t b1 b2 C C_new : ℝ}

theorem change_of_b (hC : C = t * b1^4) 
                   (hC_new : C_new = 16 * C) 
                   (hC_new_eq : C_new = t * b2^4) : 
                   b2 = 2 * b1 :=
by
  sorry

end change_of_b_l289_289925


namespace quadratic_real_roots_l289_289131

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := by 
  sorry

end quadratic_real_roots_l289_289131


namespace parabola_line_intersection_l289_289538

theorem parabola_line_intersection (x1 x2 : ℝ) (h1 : x1 * x2 = 1) (h2 : x1 + 1 = 4) : x2 + 1 = 4 / 3 :=
by
  sorry

end parabola_line_intersection_l289_289538


namespace evaluate_expression_l289_289935

theorem evaluate_expression : (4 - 3) * 2 = 2 := by
  sorry

end evaluate_expression_l289_289935


namespace circle_area_of_equilateral_triangle_l289_289641

-- Define the side length of the equilateral triangle.
def side_length : ℝ := 12

-- Define the circumradius of an equilateral triangle.
def circumradius (s : ℝ) : ℝ := s / Real.sqrt 3

-- Define the area of a circle given its radius.
def circle_area (R : ℝ) : ℝ := Real.pi * R^2

-- Define the problem statement.
theorem circle_area_of_equilateral_triangle :
  circle_area (circumradius side_length) = 48 * Real.pi :=
by
  sorry

end circle_area_of_equilateral_triangle_l289_289641


namespace quadratic_has_real_root_iff_b_in_interval_l289_289100

theorem quadratic_has_real_root_iff_b_in_interval (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ Set.Iic (-10) ∪ Set.Ici 10) := 
sorry

end quadratic_has_real_root_iff_b_in_interval_l289_289100


namespace find_m_n_l289_289259

open Nat

-- Define binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem find_m_n (m n : ℕ) (h1 : binom (n+1) (m+1) / binom (n+1) m = 5 / 3) 
  (h2 : binom (n+1) m / binom (n+1) (m-1) = 5 / 3) : m = 3 ∧ n = 6 :=
  sorry

end find_m_n_l289_289259


namespace triangle_side_lengths_l289_289067

theorem triangle_side_lengths (a : ℝ) :
  (∃ (b c : ℝ), b = 1 - 2 * a ∧ c = 8 ∧ (3 + b > c ∧ 3 + c > b ∧ b + c > 3)) ↔ (-5 < a ∧ a < -2) :=
sorry

end triangle_side_lengths_l289_289067


namespace find_n_mod_10_l289_289522

theorem find_n_mod_10 : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [MOD 10] ∧ n = 7 := by
  sorry

end find_n_mod_10_l289_289522


namespace simplify_fractions_l289_289733

theorem simplify_fractions :
  (240 / 20) * (6 / 180) * (10 / 4) = 1 :=
by sorry

end simplify_fractions_l289_289733


namespace domino_placement_l289_289228
  
theorem domino_placement (n : ℕ) :
  let K := 2 * n choose n in
  (K * K) = (nat.choose (2 * n) n) ^ 2 := by
  sorry

end domino_placement_l289_289228


namespace handshakes_count_l289_289660

def num_teams : ℕ := 4
def players_per_team : ℕ := 2
def total_players : ℕ := num_teams * players_per_team
def shakeable_players (total : ℕ) : ℕ := total * (total - players_per_team) / 2

theorem handshakes_count :
  shakeable_players total_players = 24 :=
by
  sorry

end handshakes_count_l289_289660


namespace ed_money_left_after_hotel_stay_l289_289399

theorem ed_money_left_after_hotel_stay 
  (night_rate : ℝ) (morning_rate : ℝ) 
  (initial_money : ℝ) (hours_night : ℕ) (hours_morning : ℕ) 
  (remaining_money : ℝ) : 
  night_rate = 1.50 → morning_rate = 2.00 → initial_money = 80 → 
  hours_night = 6 → hours_morning = 4 → 
  remaining_money = 63 :=
by
  intros h1 h2 h3 h4 h5
  let cost_night := night_rate * hours_night
  let cost_morning := morning_rate * hours_morning
  let total_cost := cost_night + cost_morning
  let money_left := initial_money - total_cost
  sorry

end ed_money_left_after_hotel_stay_l289_289399


namespace compare_logarithms_l289_289824

theorem compare_logarithms (a b c : ℝ) (h1 : a = Real.log 2 / Real.log 3) 
                           (h2 : b = (Real.log 2 / Real.log 3)^2) 
                           (h3 : c = Real.log (2/3) / Real.log 4) : c < b ∧ b < a :=
by
  sorry

end compare_logarithms_l289_289824


namespace one_third_percent_of_150_l289_289944

theorem one_third_percent_of_150 : (1/3) * (150 / 100) = 0.5 := by
  sorry

end one_third_percent_of_150_l289_289944


namespace hose_rate_l289_289994

theorem hose_rate (V : ℝ) (T : ℝ) (r_fixed : ℝ) (total_rate : ℝ) (R : ℝ) :
  V = 15000 ∧ T = 25 ∧ r_fixed = 3 ∧ total_rate = 10 ∧
  (2 * R + 2 * r_fixed = total_rate) → R = 2 :=
by
  -- Given conditions:
  -- Volume V = 15000 gallons
  -- Time T = 25 hours
  -- Rate of fixed hoses r_fixed = 3 gallons per minute each
  -- Total rate of filling the pool total_rate = 10 gallons per minute
  -- Relationship: 2 * rate of first two hoses + 2 * rate of fixed hoses = total rate
  
  sorry

end hose_rate_l289_289994


namespace triangle_equilateral_of_constraints_l289_289828

theorem triangle_equilateral_of_constraints {a b c : ℝ}
  (h1 : a^4 = b^4 + c^4 - b^2 * c^2)
  (h2 : b^4 = c^4 + a^4 - a^2 * c^2) : 
  a = b ∧ b = c :=
by 
  sorry

end triangle_equilateral_of_constraints_l289_289828


namespace frac_eval_eq_l289_289980

theorem frac_eval_eq :
  let a := 19
  let b := 8
  let c := 35
  let d := 19 * 8 / 35
  ( (⌈a / b - ⌈c / d⌉⌉) / ⌈c / b + ⌈d⌉⌉) = (1 / 10) := by
  sorry

end frac_eval_eq_l289_289980


namespace berry_average_temperature_l289_289972

def sunday_temp : ℝ := 99.1
def monday_temp : ℝ := 98.2
def tuesday_temp : ℝ := 98.7
def wednesday_temp : ℝ := 99.3
def thursday_temp : ℝ := 99.8
def friday_temp : ℝ := 99.0
def saturday_temp : ℝ := 98.9

def total_temp : ℝ := sunday_temp + monday_temp + tuesday_temp + wednesday_temp + thursday_temp + friday_temp + saturday_temp
def average_temp : ℝ := total_temp / 7

theorem berry_average_temperature : average_temp = 99 := by
  sorry

end berry_average_temperature_l289_289972


namespace average_rate_of_change_l289_289425

noncomputable def f (x : ℝ) : ℝ := x^2 + 1

theorem average_rate_of_change (Δx : ℝ) : 
  (f (1 + Δx) - f 1) / Δx = 2 + Δx := 
by
  sorry

end average_rate_of_change_l289_289425


namespace part_a_l289_289004

theorem part_a (m : ℕ) (A B : ℕ) (hA : A = (10^(2 * m) - 1) / 9) (hB : B = 4 * ((10^m - 1) / 9)) :
  ∃ k : ℕ, A + B + 1 = k^2 :=
sorry

end part_a_l289_289004


namespace interest_percentage_calculation_l289_289159

-- Definitions based on problem conditions
def purchase_price : ℝ := 110
def down_payment : ℝ := 10
def monthly_payment : ℝ := 10
def number_of_monthly_payments : ℕ := 12

-- Theorem statement:
theorem interest_percentage_calculation :
  let total_paid := down_payment + (monthly_payment * number_of_monthly_payments)
  let interest_paid := total_paid - purchase_price
  let interest_percent := (interest_paid / purchase_price) * 100
  interest_percent = 18.2 :=
by sorry

end interest_percentage_calculation_l289_289159


namespace find_sum_of_squares_l289_289079

theorem find_sum_of_squares (x y z : ℝ)
  (h1 : x^2 + 3 * y = 8)
  (h2 : y^2 + 5 * z = -9)
  (h3 : z^2 + 7 * x = -16) : x^2 + y^2 + z^2 = 20.75 :=
sorry

end find_sum_of_squares_l289_289079


namespace fuel_A_volume_l289_289027

-- Let V_A and V_B be defined as the volumes of fuel A and B respectively.
def V_A : ℝ := sorry
def V_B : ℝ := sorry

-- Given conditions:
axiom h1 : V_A + V_B = 214
axiom h2 : 0.12 * V_A + 0.16 * V_B = 30

-- Prove that the volume of fuel A added, V_A, is 106 gallons.
theorem fuel_A_volume : V_A = 106 := 
by
  sorry

end fuel_A_volume_l289_289027


namespace solve_inequality_l289_289598

theorem solve_inequality : { x : ℝ // (x < -1) ∨ (-2/3 < x) } :=
sorry

end solve_inequality_l289_289598


namespace Katona_theorem_l289_289167

noncomputable def binomial (n k : ℕ) : ℕ := nat.choose n k

theorem Katona_theorem {n k : ℕ} (h : k ≤ n / 2) 
(F : set (finset (fin n) × finset (fin n))) 
(hF : ∀ A B ∈ F, finset.inter A.1 B.1 ≠ ∅ ∧ finset.inter A.2 B.2 ≠ ∅) :
  F.card ≤ (binomial (n-1) (k-1)) * (binomial (n-1) (k-1)) :=
sorry

end Katona_theorem_l289_289167


namespace fishing_tomorrow_l289_289864

theorem fishing_tomorrow 
  (P1 : ∀ day : ℕ, 7 ∈ {p | p goes fishing on day})
  (P2 : ∀ day : ℕ, day % 2 = 0 → 8 ∈ {p | p goes fishing on day})
  (P3 : ∀ day : ℕ, day % 3 = 0 → 3 ∈ {p | p goes fishing on day})
  (P4 : ℕ)
  (yesterday : ℕ)
  (today : ℕ)
  (tomorrow : ℕ)
  (hyesterday : yesterday = 12)
  (htoday : today = 10)
  : tomorrow = 15 := by
  sorry

end fishing_tomorrow_l289_289864


namespace first_nonzero_digit_right_of_decimal_1_199_l289_289475

theorem first_nonzero_digit_right_of_decimal_1_199 :
  let x := (1 / 199 : ℚ)
  let first_nonzero_digit := 2
  (∃ (n : ℕ), x * 10^n - (x * 10^n).floor = first_nonzero_digit * 10^(-(n-1))) :=
begin
  sorry
end


end first_nonzero_digit_right_of_decimal_1_199_l289_289475


namespace pages_left_l289_289313

theorem pages_left (total_pages read_fraction : ℕ) (h_total_pages : total_pages = 396) (h_read_fraction : read_fraction = 1/3) : total_pages * (1 - read_fraction) = 264 := 
by
  sorry

end pages_left_l289_289313


namespace quadratic_real_roots_l289_289270

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + 4 * x - 1 = 0) ↔ m ≥ -3 ∧ m ≠ 1 := 
by 
  sorry

end quadratic_real_roots_l289_289270


namespace range_of_b_l289_289063

-- Define the conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 16
def line_eq (x y b : ℝ) : Prop := y = x + b
def distance_point_line_eq (x y b d : ℝ) : Prop := 
  d = abs (b) / (Real.sqrt 2)
def at_least_three_points_on_circle_at_distance_one (b : ℝ) : Prop := 
  ∃ p1 p2 p3 : ℝ × ℝ, circle_eq p1.1 p1.2 ∧ circle_eq p2.1 p2.2 ∧ circle_eq p3.1 p3.2 ∧ 
  distance_point_line_eq p1.1 p1.2 b 1 ∧ distance_point_line_eq p2.1 p2.2 b 1 ∧ distance_point_line_eq p3.1 p3.2 b 1

-- The theorem statement to prove
theorem range_of_b (b : ℝ) (h : at_least_three_points_on_circle_at_distance_one b) : 
  -3 * Real.sqrt 2 ≤ b ∧ b ≤ 3 * Real.sqrt 2 := 
sorry

end range_of_b_l289_289063


namespace blood_pressure_systolic_diastolic_l289_289615

noncomputable def blood_pressure (t : ℝ) : ℝ :=
110 + 25 * Real.sin (160 * t)

theorem blood_pressure_systolic_diastolic :
  (∀ t : ℝ, blood_pressure t ≤ 135) ∧ (∀ t : ℝ, blood_pressure t ≥ 85) :=
by
  sorry

end blood_pressure_systolic_diastolic_l289_289615


namespace same_sum_sufficient_days_l289_289210

variable {S Wb Wc : ℝ}
variable (h1 : S = 12 * Wb)
variable (h2 : S = 24 * Wc)

theorem same_sum_sufficient_days : ∃ D : ℝ, D = 8 ∧ S = D * (Wb + Wc) :=
by
  use 8
  sorry

end same_sum_sufficient_days_l289_289210


namespace polynomial_value_l289_289000

theorem polynomial_value : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 :=
by
  sorry

end polynomial_value_l289_289000


namespace sixth_graders_more_than_seventh_l289_289602

def pencil_cost : ℕ := 13
def eighth_graders_total : ℕ := 208
def seventh_graders_total : ℕ := 181
def sixth_graders_total : ℕ := 234

-- Number of students in each grade who bought a pencil
def seventh_graders_count := seventh_graders_total / pencil_cost
def sixth_graders_count := sixth_graders_total / pencil_cost

-- The difference in the number of sixth graders than seventh graders who bought a pencil
theorem sixth_graders_more_than_seventh : sixth_graders_count - seventh_graders_count = 4 :=
by sorry

end sixth_graders_more_than_seventh_l289_289602


namespace series_ln2_series_1_ln2_l289_289775

theorem series_ln2 :
  ∑' n : ℕ, (1 / (n + 1) / (n + 2)) = Real.log 2 :=
sorry

theorem series_1_ln2 :
  ∑' k : ℕ, (1 / ((2 * k + 2) * (2 * k + 3))) = 1 - Real.log 2 :=
sorry

end series_ln2_series_1_ln2_l289_289775


namespace initial_numbers_unique_l289_289337

theorem initial_numbers_unique 
  (A B C A' B' C' : ℕ) 
  (h1: 1 ≤ A ∧ A ≤ 50) 
  (h2: 1 ≤ B ∧ B ≤ 50) 
  (h3: 1 ≤ C ∧ C ≤ 50) 
  (final_ana : 104 = 2 * A + B + C)
  (final_beto : 123 = A + 2 * B + C)
  (final_caio : 137 = A + B + 2 * C) : 
  A = 13 ∧ B = 32 ∧ C = 46 :=
sorry

end initial_numbers_unique_l289_289337


namespace last_digit_base5_89_l289_289034

theorem last_digit_base5_89 (n : ℕ) (h : n = 89) : (n % 5) = 4 :=
by 
  sorry

end last_digit_base5_89_l289_289034


namespace quadratic_real_roots_l289_289110

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := 
by 
  sorry

end quadratic_real_roots_l289_289110


namespace Blossom_room_area_square_inches_l289_289757

def feet_to_inches (feet : ℕ) : ℕ := feet * 12

theorem Blossom_room_area_square_inches :
  (let length_feet := 10 in
   let width_feet := 10 in
   let length_inches := feet_to_inches length_feet in
   let width_inches := feet_to_inches width_feet in
   length_inches * width_inches = 14400) :=
by
  let length_feet := 10
  let width_feet := 10
  let length_inches := feet_to_inches length_feet
  let width_inches := feet_to_inches width_feet
  show length_inches * width_inches = 14400
  sorry

end Blossom_room_area_square_inches_l289_289757


namespace sarah_bottle_caps_total_l289_289452

def initial_caps : ℕ := 450
def first_day_caps : ℕ := 175
def second_day_caps : ℕ := 95
def third_day_caps : ℕ := 220
def total_caps : ℕ := 940

theorem sarah_bottle_caps_total : 
    initial_caps + first_day_caps + second_day_caps + third_day_caps = total_caps :=
by
  sorry

end sarah_bottle_caps_total_l289_289452


namespace total_sticks_needed_l289_289732

/-
Given conditions:
1. Simon's raft needs 36 sticks.
2. Gerry's raft needs two-thirds of the number of sticks that Simon needs.
3. Micky's raft needs 9 sticks more than Simon and Gerry's rafts combined.

Prove that the total number of sticks collected by Simon, Gerry, and Micky is 129.
-/

theorem total_sticks_needed :
  let S := 36 in
  let G := (2/3) * S in
  let M := S + G + 9 in
  S + G + M = 129 :=
by
  let S := 36
  let G := (2/3) * S
  let M := S + G + 9
  have : S + G + M = 129 := sorry
  exact this

end total_sticks_needed_l289_289732


namespace sixty_percent_of_fifty_minus_forty_percent_of_thirty_l289_289841

theorem sixty_percent_of_fifty_minus_forty_percent_of_thirty : 
  (0.6 * 50) - (0.4 * 30) = 18 :=
by
  sorry

end sixty_percent_of_fifty_minus_forty_percent_of_thirty_l289_289841


namespace minimum_value_of_4a_plus_b_l289_289829

noncomputable def minimum_value (a b : ℝ) :=
  if a > 0 ∧ b > 0 ∧ a^2 + a*b - 3 = 0 then 4*a + b else 0

theorem minimum_value_of_4a_plus_b :
  ∀ (a b : ℝ), a > 0 → b > 0 → a^2 + a*b - 3 = 0 → 4*a + b ≥ 6 :=
by
  intros a b ha hb hab
  sorry

end minimum_value_of_4a_plus_b_l289_289829


namespace quadratic_real_root_iff_b_range_l289_289105

open Real

theorem quadratic_real_root_iff_b_range (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_real_root_iff_b_range_l289_289105


namespace fishing_problem_l289_289858

theorem fishing_problem
  (everyday : ℕ)
  (every_other_day : ℕ)
  (every_three_days : ℕ)
  (yesterday_fishing : ℕ)
  (today_fishing : ℕ)
  (h_everyday : everyday = 7)
  (h_every_other_day : every_other_day = 8)
  (h_every_three_days : every_three_days = 3)
  (h_yesterday_fishing : yesterday_fishing = 12)
  (h_today_fishing : today_fishing = 10) :
  (every_three_days + everyday + (every_other_day - (yesterday_fishing - everyday))) = 15 := by
  sorry

end fishing_problem_l289_289858


namespace geometric_series_sum_l289_289806

theorem geometric_series_sum :
  let a := (1 / 4 : ℚ)
  let r := (1 / 4 : ℚ)
  let n := 5
  let S := a * (1 - r^n) / (1 - r)
  S = 341 / 1024 :=
by
  let a := (1 / 4 : ℚ)
  let r := (1 / 4 : ℚ)
  let n := 5
  let S := a * (1 - r^n) / (1 - r)
  show S = 341 / 1024
  sorry

end geometric_series_sum_l289_289806


namespace maximum_BD_cyclic_quad_l289_289316

theorem maximum_BD_cyclic_quad (AB BC CD : ℤ) (BD : ℝ)
  (h_side_bounds : AB < 15 ∧ BC < 15 ∧ CD < 15)
  (h_distinct_sides : AB ≠ BC ∧ BC ≠ CD ∧ CD ≠ AB)
  (h_AB_value : AB = 13)
  (h_BC_value : BC = 5)
  (h_CD_value : CD = 8)
  (h_sides_product : BC * CD = AB * (10 : ℤ)) :
  BD = Real.sqrt 179 := 
by 
  sorry

end maximum_BD_cyclic_quad_l289_289316


namespace range_of_m_l289_289704

-- Definitions
def is_circle_eqn (d e f : ℝ) : Prop :=
  d^2 + e^2 - 4 * f > 0

-- Main statement 
theorem range_of_m (m : ℝ) : 
  is_circle_eqn (-2) (-4) m → m < 5 :=
by
  intro h
  sorry

end range_of_m_l289_289704


namespace No_of_boxes_in_case_l289_289415

-- Define the conditions
def George_has_total_blocks : ℕ := 12
def blocks_per_box : ℕ := 6
def George_has_boxes : ℕ := George_has_total_blocks / blocks_per_box

-- The theorem to prove
theorem No_of_boxes_in_case : George_has_boxes = 2 :=
by
  sorry

end No_of_boxes_in_case_l289_289415


namespace angela_age_in_5_years_l289_289243

-- Define the variables representing Angela and Beth's ages.
variable (A B : ℕ)

-- State the conditions as hypotheses.
def condition_1 : Prop := A = 4 * B
def condition_2 : Prop := (A - 5) + (B - 5) = 45

-- State the final proposition that Angela will be 49 years old in five years.
theorem angela_age_in_5_years (h1 : condition_1 A B) (h2 : condition_2 A B) : A + 5 = 49 := by
  sorry

end angela_age_in_5_years_l289_289243


namespace beka_distance_l289_289386

theorem beka_distance (jackson_distance : ℕ) (beka_more_than_jackson : ℕ) :
  jackson_distance = 563 → beka_more_than_jackson = 310 → 
  (jackson_distance + beka_more_than_jackson = 873) :=
by
  sorry

end beka_distance_l289_289386


namespace hexagon_area_l289_289474

theorem hexagon_area :
  let points := [(0, 0), (2, 4), (5, 4), (7, 0), (5, -4), (2, -4), (0, 0)]
  ∃ (area : ℝ), area = 52 := by
  sorry

end hexagon_area_l289_289474


namespace inequality_solution_l289_289601

theorem inequality_solution (x : ℝ) (h : 3 * x + 2 ≠ 0) : 
  3 - 2/(3 * x + 2) < 5 ↔ x > -2/3 := 
sorry

end inequality_solution_l289_289601


namespace abcd_hife_value_l289_289776

theorem abcd_hife_value (a b c d e f g h i : ℝ) 
  (h1 : a / b = 1 / 3) 
  (h2 : b / c = 2) 
  (h3 : c / d = 1 / 2) 
  (h4 : d / e = 3) 
  (h5 : e / f = 1 / 10) 
  (h6 : f / g = 3 / 4) 
  (h7 : g / h = 1 / 5) 
  (h8 : h / i = 5) : 
  abcd / hife = 17.28 := sorry

end abcd_hife_value_l289_289776


namespace find_three_digit_number_l289_289674

theorem find_three_digit_number :
  ∃ a b c : ℕ, 
  a + b + c = 9 ∧ 
  a * b * c = 24 ∧
  100*c + 10*b + a = (27/38) * (100*a + 10*b + c) ∧ 
  100*a + 10*b + c = 342 := sorry

end find_three_digit_number_l289_289674


namespace jill_age_l289_289341

theorem jill_age (H J : ℕ) (h1 : H + J = 41) (h2 : H - 7 = 2 * (J - 7)) : J = 16 :=
by
  sorry

end jill_age_l289_289341


namespace poodle_barks_proof_l289_289963

-- Definitions based on our conditions
def terrier_barks (hushes : Nat) : Nat := hushes * 2
def poodle_barks (terrier_barks : Nat) : Nat := terrier_barks * 2

-- Given that the terrier's owner says "hush" six times
def hushes : Nat := 6
def terrier_barks_total : Nat := terrier_barks hushes

-- The final statement that we need to prove
theorem poodle_barks_proof : 
    ∃ P, P = poodle_barks terrier_barks_total ∧ P = 24 := 
by
  -- The proof goes here
  sorry

end poodle_barks_proof_l289_289963


namespace proof_shortest_side_l289_289911

-- Definitions based on problem conditions
def side_divided (a b : ℕ) : Prop := a + b = 20

def radius (r : ℕ) : Prop := r = 5

noncomputable def shortest_side (a b c : ℕ) : ℕ :=
  if a ≤ b ∧ a ≤ c then a
  else if b ≤ a ∧ b ≤ c then b
  else c

-- Proof problem statement
theorem proof_shortest_side {a b c : ℕ} (h1 : side_divided 9 11) (h2 : radius 5) :
  shortest_side 15 (11 + 9) (2 * 6 + 9) = 14 :=
sorry

end proof_shortest_side_l289_289911


namespace pentagon_angles_sum_l289_289656

theorem pentagon_angles_sum {α β γ δ ε : ℝ} (h1 : α + β + γ + δ + ε = 180) (h2 : α = 50) :
  β + ε = 230 := 
sorry

end pentagon_angles_sum_l289_289656


namespace efficiency_and_days_l289_289166

noncomputable def sakshi_efficiency : ℝ := 1 / 25
noncomputable def tanya_efficiency : ℝ := 1.25 * sakshi_efficiency
noncomputable def ravi_efficiency : ℝ := 0.70 * sakshi_efficiency
noncomputable def combined_efficiency : ℝ := sakshi_efficiency + tanya_efficiency + ravi_efficiency
noncomputable def days_to_complete_work : ℝ := 1 / combined_efficiency

theorem efficiency_and_days:
  combined_efficiency = 29.5 / 250 ∧
  days_to_complete_work = 250 / 29.5 :=
by
  sorry

end efficiency_and_days_l289_289166


namespace find_integer_pairs_l289_289211

theorem find_integer_pairs (x y : ℤ) :
  x^4 + (y+2)^3 = (x+2)^4 ↔ (x, y) = (0, 0) ∨ (x, y) = (-1, -2) := sorry

end find_integer_pairs_l289_289211


namespace age_difference_l289_289789

theorem age_difference (A B : ℕ) (h1 : B = 38) (h2 : A + 10 = 2 * (B - 10)) : A - B = 8 :=
by
  sorry

end age_difference_l289_289789


namespace circumscribed_circle_area_l289_289639

open Real

/-- Define the side length of the equilateral triangle --/
def side_length : ℝ := 12

/-- Define the circumradius of an equilateral triangle given side length s --/
def circumradius (s : ℝ) : ℝ := s / sqrt 3

/-- Define the area of a circle given its radius --/
def circle_area (r : ℝ) : ℝ := π * r^2

/-- Theorem: The area of the circle circumscribed about an equilateral triangle 
    with side length 12 units is 48π square units. --/
theorem circumscribed_circle_area :
  circle_area (circumradius side_length) = 48 * π :=
by
  /- Proof will be inserted here -/
  sorry

end circumscribed_circle_area_l289_289639


namespace profit_bicycle_l289_289724

theorem profit_bicycle (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 650) 
  (h2 : x + 2 * y = 350) : 
  x = 150 ∧ y = 100 :=
by 
  sorry

end profit_bicycle_l289_289724


namespace win_sector_area_l289_289226

/-- Given a circular spinner with a radius of 8 cm and the probability of winning being 3/8,
    prove that the area of the WIN sector is 24π square centimeters. -/
theorem win_sector_area (r : ℝ) (P_win : ℝ) (area_WIN : ℝ) :
  r = 8 → P_win = 3 / 8 → area_WIN = 24 * Real.pi := by
sorry

end win_sector_area_l289_289226


namespace circle_area_equilateral_triangle_l289_289636

theorem circle_area_equilateral_triangle (s : ℝ) (hs : s = 12) :
  ∃ (A : ℝ), A = 48 * Real.pi :=
by
  use 48 * Real.pi
  sorry

end circle_area_equilateral_triangle_l289_289636


namespace problem_1_problem_2_problem_3_l289_289695

def U : Set ℝ := { x | 1 ≤ x ∧ x ≤ 7 }
def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 5 }
def B : Set ℝ := { x | 3 ≤ x ∧ x ≤ 7 }
def notU (s : Set ℝ) : Set ℝ := { x | x ∉ s ∧ x ∈ U }

theorem problem_1 : A ∩ B = { x | 3 ≤ x ∧ x ≤ 5 } :=
sorry

theorem problem_2 : notU A ∪ B = { x | (1 ≤ x ∧ x < 2) ∨ (3 ≤ x ∧ x ≤ 7) } :=
sorry

theorem problem_3 : A ∩ notU B = { x | 2 ≤ x ∧ x < 3 } :=
sorry

end problem_1_problem_2_problem_3_l289_289695


namespace find_students_l289_289961

theorem find_students (n : ℕ) (h1 : n % 8 = 5) (h2 : n % 6 = 1) (h3 : n < 50) : n = 13 :=
sorry

end find_students_l289_289961


namespace minimize_f_l289_289541

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x + (Real.sin x)^2

theorem minimize_f :
  ∃ x : ℝ, (-π / 4 < x ∧ x ≤ π / 2) ∧
  ∀ y : ℝ, (-π / 4 < y ∧ y ≤ π / 2) → f y ≥ f x ∧ f x = 1 ∧ x = π / 2 :=
by
  sorry

end minimize_f_l289_289541


namespace wire_cut_l289_289782

theorem wire_cut (x : ℝ) (h1 : x + (100 - x) = 100) (h2 : x = (7/13) * (100 - x)) : x = 35 :=
sorry

end wire_cut_l289_289782


namespace quadratic_roots_interval_l289_289089

theorem quadratic_roots_interval (b : ℝ) :
  ∃ (x : ℝ), x^2 + b * x + 25 = 0 → b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_roots_interval_l289_289089


namespace find_A_from_equation_l289_289937

variable (A B C D : ℕ)
variable (diff_numbers : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
variable (eq1 : A * 1000 + B * 100 + 82 - 900 + C * 10 + 9 = 4000 + 900 + 30 + D)

theorem find_A_from_equation (A B C D : ℕ) (diff_numbers : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (eq1 : A * 1000 + B * 100 + 82 - (900 + C * 10 + 9) = 4000 + 900 + 30 + D) : A = 5 :=
by sorry

end find_A_from_equation_l289_289937


namespace total_pieces_of_clothing_l289_289326

def number_of_pieces_per_drawer : ℕ := 2
def number_of_drawers : ℕ := 4

theorem total_pieces_of_clothing : 
  (number_of_pieces_per_drawer * number_of_drawers = 8) :=
by sorry

end total_pieces_of_clothing_l289_289326


namespace area_square_15_cm_l289_289405

-- Define the side length of the square
def side_length : ℝ := 15

-- Define the area calculation for a square given the side length
def area_of_square (s : ℝ) : ℝ := s * s

-- The theorem statement translating the problem to Lean
theorem area_square_15_cm :
  area_of_square side_length = 225 :=
by
  -- We need to provide proof here, but 'sorry' is used to skip the proof as per instructions
  sorry

end area_square_15_cm_l289_289405


namespace problem_statement_l289_289909

-- The conditions of the problem
variables (x : Real)

-- Define the conditions as hypotheses
def condition1 : Prop := (Real.sin (3 * x) * Real.sin (4 * x)) = (Real.cos (3 * x) * Real.cos (4 * x))
def condition2 : Prop := Real.sin (7 * x) = 0

-- The theorem we need to prove
theorem problem_statement (h1 : condition1 x) (h2 : condition2 x) : x = Real.pi / 7 :=
by sorry

end problem_statement_l289_289909


namespace value_of_a_l289_289137

theorem value_of_a (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y = 0 → 3 * x + y + a = 0) → a = 1 :=
by
  sorry

end value_of_a_l289_289137


namespace perimeter_eq_20_l289_289187

-- Define the lengths of the sides
def horizontal_sides := [2, 3]
def vertical_sides := [2, 3, 3, 2]

-- Define the perimeter calculation
def perimeter := horizontal_sides.sum + vertical_sides.sum

theorem perimeter_eq_20 : perimeter = 20 :=
by
  -- We assert that the calculations do hold
  sorry

end perimeter_eq_20_l289_289187


namespace find_income_4_l289_289221

noncomputable def income_4 (income_1 income_2 income_3 income_5 average_income num_days : ℕ) : ℕ :=
  average_income * num_days - (income_1 + income_2 + income_3 + income_5)

theorem find_income_4
  (income_1 : ℕ := 200)
  (income_2 : ℕ := 150)
  (income_3 : ℕ := 750)
  (income_5 : ℕ := 500)
  (average_income : ℕ := 400)
  (num_days : ℕ := 5) :
  income_4 income_1 income_2 income_3 income_5 average_income num_days = 400 :=
by
  unfold income_4
  sorry

end find_income_4_l289_289221


namespace circle_area_equilateral_triangle_l289_289637

theorem circle_area_equilateral_triangle (s : ℝ) (hs : s = 12) :
  ∃ (A : ℝ), A = 48 * Real.pi :=
by
  use 48 * Real.pi
  sorry

end circle_area_equilateral_triangle_l289_289637


namespace cadastral_value_of_land_l289_289512

theorem cadastral_value_of_land (tax_amount : ℝ) (tax_rate : ℝ) (V : ℝ)
    (h1 : tax_amount = 4500)
    (h2 : tax_rate = 0.003) :
    V = 1500000 :=
by
  sorry

end cadastral_value_of_land_l289_289512


namespace quadratic_roots_interval_l289_289087

theorem quadratic_roots_interval (b : ℝ) :
  ∃ (x : ℝ), x^2 + b * x + 25 = 0 → b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_roots_interval_l289_289087


namespace sum_series_l289_289722

noncomputable def b : ℕ → ℝ
| 0     => 2
| 1     => 2
| (n+2) => b (n+1) + b n

theorem sum_series : (∑' n, b n / 3^(n+1)) = 1 / 3 := by
  sorry

end sum_series_l289_289722


namespace final_price_correct_l289_289628

variable (original_price first_discount second_discount third_discount sales_tax : ℝ)
variable (final_discounted_price final_price: ℝ)

-- Define original price and discounts
def initial_price : ℝ := 20000
def discount1      : ℝ := 0.12
def discount2      : ℝ := 0.10
def discount3      : ℝ := 0.05
def tax_rate       : ℝ := 0.08

def price_after_first_discount : ℝ := initial_price * (1 - discount1)
def price_after_second_discount : ℝ := price_after_first_discount * (1 - discount2)
def price_after_third_discount : ℝ := price_after_second_discount * (1 - discount3)
def final_sale_price : ℝ := price_after_third_discount * (1 + tax_rate)

-- Prove final sale price is 16251.84
theorem final_price_correct : final_sale_price = 16251.84 := by
  sorry

end final_price_correct_l289_289628


namespace square_D_perimeter_l289_289741

theorem square_D_perimeter 
(C_perimeter: Real) 
(D_area_ratio : Real) 
(hC : C_perimeter = 32) 
(hD : D_area_ratio = 1/3) : 
    ∃ D_perimeter, D_perimeter = (32 * Real.sqrt 3) / 3 := 
by 
    sorry

end square_D_perimeter_l289_289741


namespace problem_inequality_l289_289905

theorem problem_inequality (a b c m n p : ℝ) (h1 : a + b + c = 1) (h2 : m + n + p = 1) :
  -1 ≤ a * m + b * n + c * p ∧ a * m + b * n + c * p ≤ 1 := by
  sorry

end problem_inequality_l289_289905


namespace place_signs_correct_l289_289449

theorem place_signs_correct :
  1 * 3 / 3^2 / 3^4 / 3^8 * 3^16 * 3^32 * 3^64 = 3^99 :=
by
  sorry

end place_signs_correct_l289_289449


namespace distribution_ways_l289_289382

def number_of_ways_to_distribute_problems : ℕ :=
  let friends := 10
  let problems := 7
  let max_receivers := 3
  let ways_to_choose_friends := Nat.choose friends max_receivers
  let ways_to_distribute_problems := max_receivers ^ problems
  ways_to_choose_friends * ways_to_distribute_problems

theorem distribution_ways :
  number_of_ways_to_distribute_problems = 262440 :=
by
  -- Proof is omitted
  sorry

end distribution_ways_l289_289382


namespace find_b_values_l289_289080

noncomputable def has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + b*x + 25 = 0

theorem find_b_values (b : ℝ) : has_real_root b ↔ b ∈ set.Iic (-10) ∪ set.Ici 10 := by
  sorry

end find_b_values_l289_289080


namespace points_scored_fourth_game_l289_289381

-- Define the conditions
def avg_score_3_games := 18
def avg_score_4_games := 17
def games_played_3 := 3
def games_played_4 := 4

-- Calculate total points after 3 games
def total_points_3_games := avg_score_3_games * games_played_3

-- Calculate total points after 4 games
def total_points_4_games := avg_score_4_games * games_played_4

-- Define a theorem to prove the points scored in the fourth game
theorem points_scored_fourth_game :
  total_points_4_games - total_points_3_games = 14 :=
by
  sorry

end points_scored_fourth_game_l289_289381


namespace simplify_expression_l289_289734

theorem simplify_expression (x : ℝ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 :=
by
  sorry

end simplify_expression_l289_289734


namespace sin_inequality_of_triangle_l289_289310

theorem sin_inequality_of_triangle (B C : ℝ) (hB : 0 < B) (hB_lt_pi : B < π) 
(hC : 0 < C) (hC_lt_pi : C < π) :
  (B > C) ↔ (Real.sin B > Real.sin C) := 
  sorry

end sin_inequality_of_triangle_l289_289310


namespace negation_of_prop_l289_289069

theorem negation_of_prop :
  ¬ (∀ x : ℝ, x^2 - 1 > 0) ↔ ∃ x : ℝ, x^2 - 1 ≤ 0 :=
sorry

end negation_of_prop_l289_289069


namespace fishing_tomorrow_l289_289871

theorem fishing_tomorrow (every_day_fishers every_other_day_fishers every_three_days_fishers fishers_yesterday fishers_today : ℕ) :
  (every_day_fishers = 7) →
  (every_other_day_fishers = 8) →
  (every_three_days_fishers = 3) →
  (fishers_yesterday = 12) →
  (fishers_today = 10) →
  (every_three_days_fishers + every_day_fishers + (every_other_day_fishers - (fishers_yesterday - every_day_fishers)) = 15) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  simp
  sorry

end fishing_tomorrow_l289_289871


namespace dividend_rate_is_16_l289_289495

noncomputable def dividend_rate_of_shares : ℝ :=
  let share_value := 48
  let interest_rate := 0.12
  let market_value := 36.00000000000001
  (interest_rate * share_value) / market_value * 100

theorem dividend_rate_is_16 :
  dividend_rate_of_shares = 16 := by
  sorry

end dividend_rate_is_16_l289_289495


namespace triangle_side_count_l289_289851

theorem triangle_side_count :
  {b c : ℕ} → b ≤ 5 → 5 ≤ c → c - b < 5 → ∃ t : ℕ, t = 15 :=
by
  sorry

end triangle_side_count_l289_289851


namespace first_nonzero_digit_of_one_over_199_l289_289476

theorem first_nonzero_digit_of_one_over_199 :
  (∃ n : ℕ, (n < 10) ∧ (rat.of_int 2 / rat.of_int 100 < 1 / rat.of_int 199) ∧ (1 / rat.of_int 199 < rat.of_int 3 / rat.of_int 100)) :=
sorry

end first_nonzero_digit_of_one_over_199_l289_289476


namespace sum_first_100_terms_l289_289422

def a (n : ℕ) : ℤ := (-1) ^ (n + 1) * n

def S (n : ℕ) : ℤ := Finset.sum (Finset.range n) (λ i => a (i + 1))

theorem sum_first_100_terms : S 100 = -50 := 
by 
  sorry

end sum_first_100_terms_l289_289422


namespace number_difference_l289_289730

theorem number_difference 
  (a1 a2 a3 : ℝ)
  (h1 : a1 = 2 * a2)
  (h2 : a1 = 3 * a3)
  (h3 : (a1 + a2 + a3) / 3 = 88) : 
  a1 - a3 = 96 :=
sorry

end number_difference_l289_289730


namespace exists_sum_of_three_l289_289459

theorem exists_sum_of_three {a b c d : ℕ} 
  (h1 : Nat.Coprime a b) 
  (h2 : Nat.Coprime a c) 
  (h3 : Nat.Coprime a d)
  (h4 : Nat.Coprime b c) 
  (h5 : Nat.Coprime b d) 
  (h6 : Nat.Coprime c d) 
  (h7 : a * b + c * d = a * c - 10 * b * d) :
  ∃ x y z, (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ 
           (y = a ∨ y = b ∨ y = c ∨ y = d) ∧ 
           (z = a ∨ z = b ∨ z = c ∨ z = d) ∧ 
           x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ 
           (x = y + z ∨ y = x + z ∨ z = x + y) :=
by
  sorry

end exists_sum_of_three_l289_289459


namespace sum_of_squares_and_product_l289_289609

theorem sum_of_squares_and_product (x y : ℕ) 
  (h1 : x^2 + y^2 = 130) 
  (h2 : x * y = 36) : 
  x + y = Real.sqrt 202 := 
by
  sorry

end sum_of_squares_and_product_l289_289609


namespace amount_of_money_C_l289_289653

variable (A B C : ℝ)

theorem amount_of_money_C (h1 : A + B + C = 500)
                         (h2 : A + C = 200)
                         (h3 : B + C = 360) :
    C = 60 :=
sorry

end amount_of_money_C_l289_289653


namespace equal_cubic_values_l289_289534

theorem equal_cubic_values (a b c d : ℝ) 
  (h1 : a + b + c + d = 3) 
  (h2 : a^2 + b^2 + c^2 + d^2 = 3) 
  (h3 : a * b * c + b * c * d + c * d * a + d * a * b = 1) :
  a * (1 - a)^3 = b * (1 - b)^3 ∧ 
  b * (1 - b)^3 = c * (1 - c)^3 ∧ 
  c * (1 - c)^3 = d * (1 - d)^3 :=
sorry

end equal_cubic_values_l289_289534


namespace find_x_l289_289826

-- Definitions based on conditions
def parabola_eq (y x p : ℝ) : Prop := y^2 = 2 * p * x
def point_on_parabola (p : ℝ) : Prop := ∃ y x, parabola_eq y x p ∧ (x = 1) ∧ (y = 2)
def valid_p (p : ℝ) : Prop := p > 0
def dist_to_focus (x : ℝ) : ℝ := 1
def dist_to_line (x : ℝ) : ℝ := abs (x + 1)

-- Main statement to be proven
theorem find_x (p : ℝ) (h1 : point_on_parabola p) (h2 : valid_p p) :
  ∃ x, dist_to_focus x = dist_to_line x ∧ x = 1 :=
sorry

end find_x_l289_289826


namespace value_of_V3_l289_289353

-- Define the polynomial function using Horner's rule
def f (x : ℤ) := (((((2 * x + 0) * x - 3) * x + 2) * x + 1) * x - 3)

-- Define the value of x
def x : ℤ := 2

-- Prove the value of V_3 when x = 2
theorem value_of_V3 : f x = 12 := by
  sorry

end value_of_V3_l289_289353


namespace certain_number_eq_40_l289_289332

theorem certain_number_eq_40 (x : ℝ) 
    (h : (20 + x + 60) / 3 = (20 + 60 + 25) / 3 + 5) : x = 40 := 
by
  sorry

end certain_number_eq_40_l289_289332


namespace problem_proof_l289_289002

theorem problem_proof :
  (3 ∣ 18) ∧
  (17 ∣ 187 ∧ ¬ (17 ∣ 52)) ∧
  ¬ ((24 ∣ 72) ∧ (24 ∣ 67)) ∧
  ¬ (13 ∣ 26 ∧ ¬ (13 ∣ 52)) ∧
  (8 ∣ 160) :=
by 
  sorry

end problem_proof_l289_289002


namespace root_of_equation_l289_289047

theorem root_of_equation (x : ℝ) : 
  169 * (157 - 77 * x)^2 + 100 * (201 - 100 * x)^2 = 26 * (77 * x - 157) * (1000 * x - 2010) ↔ x = 31 := 
by 
  sorry

end root_of_equation_l289_289047


namespace time_to_empty_is_109_89_hours_l289_289623

noncomputable def calculate_time_to_empty_due_to_leak : ℝ :=
  let R := 1 / 10 -- filling rate in tank/hour
  let Reffective := 1 / 11 -- effective filling rate in tank/hour
  let L := R - Reffective -- leak rate in tank/hour
  1 / L -- time to empty in hours

theorem time_to_empty_is_109_89_hours : calculate_time_to_empty_due_to_leak = 109.89 :=
by
  rw [calculate_time_to_empty_due_to_leak]
  sorry -- Proof steps can be filled in later

end time_to_empty_is_109_89_hours_l289_289623


namespace gabriel_month_days_l289_289414

theorem gabriel_month_days (forgot_days took_days : ℕ) (h_forgot : forgot_days = 3) (h_took : took_days = 28) : 
  forgot_days + took_days = 31 :=
by
  sorry

end gabriel_month_days_l289_289414


namespace fishing_problem_l289_289857

theorem fishing_problem
  (everyday : ℕ)
  (every_other_day : ℕ)
  (every_three_days : ℕ)
  (yesterday_fishing : ℕ)
  (today_fishing : ℕ)
  (h_everyday : everyday = 7)
  (h_every_other_day : every_other_day = 8)
  (h_every_three_days : every_three_days = 3)
  (h_yesterday_fishing : yesterday_fishing = 12)
  (h_today_fishing : today_fishing = 10) :
  (every_three_days + everyday + (every_other_day - (yesterday_fishing - everyday))) = 15 := by
  sorry

end fishing_problem_l289_289857


namespace area_of_room_in_square_inches_l289_289760

-- Defining the conversion from feet to inches
def feet_to_inches (ft : ℕ) : ℕ := ft * 12

-- Given conditions
def length_in_feet : ℕ := 10
def width_in_feet : ℕ := 10

-- Calculate length and width in inches
def length_in_inches := feet_to_inches length_in_feet
def width_in_inches := feet_to_inches width_in_feet

-- Calculate area in square inches
def area_in_square_inches := length_in_inches * width_in_inches

-- Theorem statement
theorem area_of_room_in_square_inches
  (h1 : length_in_feet = 10)
  (h2 : width_in_feet = 10)
  (conversion : feet_to_inches 1 = 12) :
  area_in_square_inches = 14400 :=
sorry

end area_of_room_in_square_inches_l289_289760


namespace Jenny_ate_65_l289_289149

theorem Jenny_ate_65 (mike_squares : ℕ) (jenny_squares : ℕ)
  (h1 : mike_squares = 20)
  (h2 : jenny_squares = 3 * mike_squares + 5) :
  jenny_squares = 65 :=
by
  sorry

end Jenny_ate_65_l289_289149


namespace total_students_l289_289713

theorem total_students (m d : ℕ) 
  (H1: 30 < m + d ∧ m + d < 40)
  (H2: ∃ r, r = 3 * m ∧ r = 5 * d) : 
  m + d = 32 := 
by
  sorry

end total_students_l289_289713


namespace sum_of_cubes_divisible_by_9_l289_289588

theorem sum_of_cubes_divisible_by_9 (n : ℕ) : 9 ∣ (n^3 + (n + 1)^3 + (n + 2)^3) := 
  sorry

end sum_of_cubes_divisible_by_9_l289_289588


namespace find_pairs_l289_289676

theorem find_pairs (x y : ℤ) (h : 19 / x + 96 / y = (19 * 96) / (x * y)) :
  ∃ m : ℤ, x = 19 * m ∧ y = 96 - 96 * m :=
by
  sorry

end find_pairs_l289_289676


namespace quadratic_real_roots_l289_289132

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := by 
  sorry

end quadratic_real_roots_l289_289132


namespace discount_received_l289_289505

theorem discount_received (original_cost : ℝ) (amt_spent : ℝ) (discount : ℝ) 
  (h1 : original_cost = 467) (h2 : amt_spent = 68) : 
  discount = 399 :=
by
  sorry

end discount_received_l289_289505


namespace calculate_total_cost_l289_289671

def initial_price_orange : ℝ := 40
def initial_price_mango : ℝ := 50
def price_increase_percentage : ℝ := 0.15

-- Hypotheses
def new_price (initial_price : ℝ) (percentage_increase : ℝ) : ℝ :=
  initial_price * (1 + percentage_increase)

noncomputable def total_cost (num_oranges num_mangoes : ℕ) : ℝ :=
  (num_oranges * new_price initial_price_orange price_increase_percentage) +
  (num_mangoes * new_price initial_price_mango price_increase_percentage)

theorem calculate_total_cost :
  total_cost 10 10 = 1035 := by
  sorry

end calculate_total_cost_l289_289671


namespace pascal_tenth_number_in_hundred_row_l289_289357

def pascal_row (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_tenth_number_in_hundred_row :
  pascal_row 99 9 = Nat.choose 99 9 :=
by
  sorry

end pascal_tenth_number_in_hundred_row_l289_289357


namespace empty_tank_time_l289_289621

-- Definitions based on problem conditions
def tank_full_fraction := 1 / 5
def pipeA_fill_time := 15
def pipeB_empty_time := 6

-- Derived definitions
def rate_of_pipeA := 1 / pipeA_fill_time
def rate_of_pipeB := 1 / pipeB_empty_time
def combined_rate := rate_of_pipeA - rate_of_pipeB 

-- The time to empty the tank when both pipes are open
def time_to_empty (initial_fraction : ℚ) (combined_rate : ℚ) : ℚ :=
  initial_fraction / -combined_rate

-- The main theorem to prove
theorem empty_tank_time
  (initial_fraction : ℚ := tank_full_fraction)
  (combined_rate : ℚ := combined_rate)
  (time : ℚ := time_to_empty initial_fraction combined_rate) :
  time = 2 :=
by
  sorry

end empty_tank_time_l289_289621


namespace thomas_annual_insurance_cost_l289_289940

theorem thomas_annual_insurance_cost (total_cost : ℕ) (number_of_years : ℕ) 
  (h1 : total_cost = 40000) (h2 : number_of_years = 10) : 
  total_cost / number_of_years = 4000 := 
by 
  sorry

end thomas_annual_insurance_cost_l289_289940


namespace least_positive_n_for_reducible_fraction_l289_289679

theorem least_positive_n_for_reducible_fraction :
  ∃ n : ℕ, 0 < n ∧ (∃ k : ℤ, k > 1 ∧ k ∣ (n - 17) ∧ k ∣ (6 * n + 7)) ∧ n = 126 :=
by
  sorry

end least_positive_n_for_reducible_fraction_l289_289679


namespace fishing_tomorrow_l289_289879

theorem fishing_tomorrow (seven_every_day eight_every_other_day three_every_three_days twelve_yesterday ten_today : ℕ)
  (h1 : seven_every_day = 7)
  (h2 : eight_every_other_day = 8)
  (h3 : three_every_three_days = 3)
  (h4 : twelve_yesterday = 12)
  (h5 : ten_today = 10) :
  (seven_every_day + (eight_every_other_day - (twelve_yesterday - seven_every_day)) + three_every_three_days) = 15 :=
by
  sorry

end fishing_tomorrow_l289_289879


namespace find_b_values_l289_289086

noncomputable def has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + b*x + 25 = 0

theorem find_b_values (b : ℝ) : has_real_root b ↔ b ∈ set.Iic (-10) ∪ set.Ici 10 := by
  sorry

end find_b_values_l289_289086


namespace count_squares_with_dot_l289_289407

theorem count_squares_with_dot (n : ℕ) (dot_center : (n = 5)) :
  n = 5 → ∃ k, k = 19 :=
by sorry

end count_squares_with_dot_l289_289407


namespace a_investment_l289_289374

theorem a_investment (B C total_profit A_share: ℝ) (hB: B = 7200) (hC: C = 9600) (htotal_profit: total_profit = 9000) 
  (hA_share: A_share = 1125) : ∃ x : ℝ, (A_share / total_profit) = (x / (x + B + C)) ∧ x = 2400 := 
by
  use 2400
  sorry

end a_investment_l289_289374


namespace grid_cut_990_l289_289397

theorem grid_cut_990 (grid : Matrix (Fin 1000) (Fin 1000) (Fin 2)) :
  (∃ (rows_to_remove : Finset (Fin 1000)), rows_to_remove.card = 990 ∧ 
   ∀ col : Fin 1000, ∃ row ∈ (Finset.univ \ rows_to_remove), grid row col = 1) ∨
  (∃ (cols_to_remove : Finset (Fin 1000)), cols_to_remove.card = 990 ∧ 
   ∀ row : Fin 1000, ∃ col ∈ (Finset.univ \ cols_to_remove), grid row col = 0) :=
sorry

end grid_cut_990_l289_289397


namespace range_of_k_l289_289820

theorem range_of_k (k : ℤ) (x : ℤ) 
  (h1 : -4 * x - k ≤ 0) 
  (h2 : x = -1 ∨ x = -2) : 
  8 ≤ k ∧ k < 12 :=
sorry

end range_of_k_l289_289820


namespace tangerine_boxes_l289_289469

theorem tangerine_boxes
  (num_boxes_apples : ℕ)
  (apples_per_box : ℕ)
  (num_boxes_tangerines : ℕ)
  (tangerines_per_box : ℕ)
  (total_fruits : ℕ)
  (h1 : num_boxes_apples = 19)
  (h2 : apples_per_box = 46)
  (h3 : tangerines_per_box = 170)
  (h4 : total_fruits = 1894)
  : num_boxes_tangerines = 6 := 
  sorry

end tangerine_boxes_l289_289469


namespace solution_set_of_absolute_inequality_l289_289461

theorem solution_set_of_absolute_inequality :
  {x : ℝ | |2 * x - 1| < 1} = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_absolute_inequality_l289_289461


namespace worker_assignment_l289_289257

theorem worker_assignment :
  ∃ (x y : ℕ), x + y = 85 ∧
  (16 * x) / 2 = (10 * y) / 3 ∧
  x = 25 ∧ y = 60 :=
by
  sorry

end worker_assignment_l289_289257


namespace tax_amount_is_correct_l289_289559

def camera_cost : ℝ := 200.00
def tax_rate : ℝ := 0.15

theorem tax_amount_is_correct :
  (camera_cost * tax_rate) = 30.00 :=
sorry

end tax_amount_is_correct_l289_289559


namespace smallest_perfect_square_divisible_by_2_and_5_l289_289202

theorem smallest_perfect_square_divisible_by_2_and_5 :
  ∃ n : ℕ, 0 < n ∧ (∃ k : ℕ, n = k ^ 2) ∧ (n % 2 = 0) ∧ (n % 5 = 0) ∧ 
  (∀ m : ℕ, 0 < m ∧ (∃ k : ℕ, m = k ^ 2) ∧ (m % 2 = 0) ∧ (m % 5 = 0) → n ≤ m) :=
sorry

end smallest_perfect_square_divisible_by_2_and_5_l289_289202


namespace exists_schoolchild_who_participated_in_all_competitions_l289_289305

theorem exists_schoolchild_who_participated_in_all_competitions
    (competitions : Fin 50 → Finset ℕ)
    (h_card : ∀ i, (competitions i).card = 30)
    (h_unique : ∀ i j, i ≠ j → competitions i ≠ competitions j)
    (h_intersect : ∀ S : Finset (Fin 50), S.card = 30 → 
      ∃ x, ∀ i ∈ S, x ∈ competitions i) :
    ∃ x, ∀ i, x ∈ competitions i :=
by
  sorry

end exists_schoolchild_who_participated_in_all_competitions_l289_289305


namespace unique_polynomial_P_l289_289403

open Polynomial

/-- The only polynomial P with real coefficients such that
    xP(y/x) + yP(x/y) = x + y for all nonzero real numbers x and y 
    is P(x) = x. --/
theorem unique_polynomial_P (P : ℝ[X]) (hP : ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x * P.eval (y / x) + y * P.eval (x / y) = x + y) :
P = Polynomial.C 1 * X :=
by sorry

end unique_polynomial_P_l289_289403


namespace subset_implies_a_geq_4_l289_289051

open Set

def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x^2 - a * x + 3 ≤ 0}

theorem subset_implies_a_geq_4 (a : ℝ) :
  A ⊆ B a → a ≥ 4 := sorry

end subset_implies_a_geq_4_l289_289051


namespace part1_inequality_part2_range_of_a_l289_289052

-- Definitions and conditions
def f (x a : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- First proof problem for a = 1
theorem part1_inequality (x : ℝ) : f x 1 > 1 ↔ x > 1/2 :=
by sorry

-- Second proof problem for range of a when f(x) > x in (0, 1)
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 1 → f x a > x) → 0 < a ∧ a ≤ 2 :=
by sorry

end part1_inequality_part2_range_of_a_l289_289052


namespace inequality_px_qy_l289_289292

theorem inequality_px_qy 
  (p q x y : ℝ) 
  (hp : 0 < p) 
  (hq : 0 < q) 
  (hpq : p + q < 1) 
  : (p * x + q * y) ^ 2 ≤ p * x ^ 2 + q * y ^ 2 := 
sorry

end inequality_px_qy_l289_289292


namespace quadratic_real_root_iff_b_range_l289_289103

open Real

theorem quadratic_real_root_iff_b_range (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_real_root_iff_b_range_l289_289103


namespace integral_inequality_l289_289165

open Real

theorem integral_inequality :
  ln ((sqrt 2009 + sqrt 2010) / (sqrt 2008 + sqrt 2009)) <
    ∫ x in sqrt 2008 .. sqrt 2009, (sqrt (1 - exp (-x^2)) / x) ∧
    (∫ x in sqrt 2008 .. sqrt 2009, sqrt (1 - exp (-x^2)) / x) < (sqrt 2009 - sqrt 2008) :=
sorry

end integral_inequality_l289_289165


namespace problem_statement_l289_289832

theorem problem_statement (a : Fin 2018 → ℝ) :
  (∑ i : Fin 2018, (i : ℕ + 1) * (if i % 2 = 0 then -1 else 1) * a i) = -4034 :=
by
  sorry

end problem_statement_l289_289832


namespace total_books_l289_289207

variable (a : ℕ)

theorem total_books (h₁ : 5 = 5) (h₂ : a = a) : 5 + a = 5 + a :=
by
  sorry

end total_books_l289_289207


namespace quadratic_real_root_iff_b_range_l289_289106

open Real

theorem quadratic_real_root_iff_b_range (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_real_root_iff_b_range_l289_289106


namespace haircuts_away_from_next_free_l289_289331

def free_haircut (total_paid : ℕ) : ℕ := total_paid / 14

theorem haircuts_away_from_next_free (total_haircuts : ℕ) (free_haircuts : ℕ) (haircuts_per_free : ℕ) :
  total_haircuts = 79 → free_haircuts = 5 → haircuts_per_free = 14 → 
  (haircuts_per_free - (total_haircuts - free_haircuts)) % haircuts_per_free = 10 :=
by
  intros h1 h2 h3
  sorry

end haircuts_away_from_next_free_l289_289331


namespace fishers_tomorrow_l289_289873

-- Definitions based on conditions
def people_every_day : ℕ := 7
def people_every_other_day : ℕ := 8
def people_every_three_days : ℕ := 3
def people_yesterday : ℕ := 12
def people_today : ℕ := 10

-- Theorem to be proved
theorem fishers_tomorrow (people_every_day people_every_other_day people_every_three_days people_yesterday people_today : ℕ) : 
  people_every_day = 7 ∧ 
  people_every_other_day = 8 ∧ 
  people_every_three_days = 3 ∧
  people_yesterday = 12 ∧ 
  people_today = 10 →
  15 := 
by {
  sorry
}

end fishers_tomorrow_l289_289873


namespace tina_mother_took_out_coins_l289_289347

theorem tina_mother_took_out_coins :
  let first_hour := 20
  let next_two_hours := 30 * 2
  let fourth_hour := 40
  let total_coins := first_hour + next_two_hours + fourth_hour
  let coins_left_after_fifth_hour := 100
  let coins_taken_out := total_coins - coins_left_after_fifth_hour
  coins_taken_out = 20 :=
by
  sorry

end tina_mother_took_out_coins_l289_289347


namespace integer_ratio_l289_289604

theorem integer_ratio (A B C D : ℕ) (h1 : (A + B + C + D) / 4 = 16)
  (h2 : A % B = 0) (h3 : B = C - 2) (h4 : D = 2) (h5 : A ≠ B) (h6 : B ≠ C) (h7 : C ≠ D) (h8 : D ≠ A)
  (h9: 0 < A) (h10: 0 < B) (h11: 0 < C):
  A / B = 28 := 
sorry

end integer_ratio_l289_289604


namespace problem_ABCD_cos_l289_289140

/-- In convex quadrilateral ABCD, angle A = 2 * angle C, AB = 200, CD = 200, the perimeter of 
ABCD is 720, and AD ≠ BC. Find the floor of 1000 * cos A. -/
theorem problem_ABCD_cos (A C : ℝ) (AB CD AD BC : ℝ) (h1 : AB = 200)
  (h2 : CD = 200) (h3 : AD + BC = 320) (h4 : A = 2 * C)
  (h5 : AD ≠ BC) : ⌊1000 * Real.cos A⌋ = 233 := 
sorry

end problem_ABCD_cos_l289_289140


namespace boxes_with_neither_l289_289710

-- Definitions for conditions
def total_boxes := 15
def boxes_with_crayons := 9
def boxes_with_markers := 5
def boxes_with_both := 4

-- Theorem statement
theorem boxes_with_neither :
  total_boxes - (boxes_with_crayons + boxes_with_markers - boxes_with_both) = 5 :=
by
  sorry

end boxes_with_neither_l289_289710


namespace simplify_fraction_l289_289170

open Real

theorem simplify_fraction : 
  (60 * (π / 180) = real.pi / 3) →
  (∀ x, tan x = sin x / cos x) →
  (∀ x, cot x = 1 / tan x) →
  let t := tan (real.pi / 3)
  let c := cot (real.pi / 3)
  t = sqrt 3 →
  c = 1 / sqrt 3 →
  (t ^ 3 + c ^ 3) / (t + c) = 7 / 3 :=
by
  intro h60 htan hcot t_def c_def
  sorry

end simplify_fraction_l289_289170


namespace option_C_is_correct_l289_289431

theorem option_C_is_correct (a b c : ℝ) (h : a > b) : c - a < c - b := 
by
  linarith

end option_C_is_correct_l289_289431


namespace ed_money_left_l289_289401

theorem ed_money_left
  (cost_per_hour_night : ℝ := 1.5)
  (cost_per_hour_morning : ℝ := 2)
  (initial_money : ℝ := 80)
  (hours_night : ℝ := 6)
  (hours_morning : ℝ := 4) :
  initial_money - (cost_per_hour_night * hours_night + cost_per_hour_morning * hours_morning) = 63 := 
  by
  sorry

end ed_money_left_l289_289401


namespace series_sum_correct_l289_289810

open Classical

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (2 * (k+1)) / 4^(k+1)

theorem series_sum_correct :
  series_sum = 8 / 9 :=
by
  sorry

end series_sum_correct_l289_289810


namespace fishers_tomorrow_l289_289875

-- Definitions based on conditions
def people_every_day : ℕ := 7
def people_every_other_day : ℕ := 8
def people_every_three_days : ℕ := 3
def people_yesterday : ℕ := 12
def people_today : ℕ := 10

-- Theorem to be proved
theorem fishers_tomorrow (people_every_day people_every_other_day people_every_three_days people_yesterday people_today : ℕ) : 
  people_every_day = 7 ∧ 
  people_every_other_day = 8 ∧ 
  people_every_three_days = 3 ∧
  people_yesterday = 12 ∧ 
  people_today = 10 →
  15 := 
by {
  sorry
}

end fishers_tomorrow_l289_289875


namespace n_eq_7_mod_10_l289_289521

theorem n_eq_7_mod_10 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 9) (h3 : n ≡ -2023 [MOD 10]) : n = 7 := by
  sorry

end n_eq_7_mod_10_l289_289521


namespace probability_three_aligned_l289_289269

theorem probability_three_aligned (total_arrangements favorable_arrangements : ℕ) 
  (h1 : total_arrangements = 126)
  (h2 : favorable_arrangements = 48) :
  (favorable_arrangements : ℚ) / total_arrangements = 8 / 21 :=
by sorry

end probability_three_aligned_l289_289269


namespace original_selling_price_l289_289032

-- Definitions and conditions
def cost_price (CP : ℝ) := CP
def profit (CP : ℝ) := 1.25 * CP
def loss (CP : ℝ) := 0.75 * CP
def loss_price (CP : ℝ) := 600

-- Main theorem statement
theorem original_selling_price (CP : ℝ) (h1 : loss CP = loss_price CP) : profit CP = 1000 :=
by
  -- Note: adding the proof that CP = 800 and then profit CP = 1000 would be here.
  sorry

end original_selling_price_l289_289032


namespace a4_equals_zero_l289_289839

-- Define the general term of the sequence
def a (n : ℕ) (h : n > 0) : ℤ := n^2 - 3 * n - 4

-- The theorem statement to prove a_4 = 0
theorem a4_equals_zero : a 4 (by norm_num) = 0 :=
sorry

end a4_equals_zero_l289_289839


namespace area_of_circumscribed_circle_l289_289634

noncomputable def circumradius_of_equilateral_triangle (s : ℝ) : ℝ :=
  s / Real.sqrt 3

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem area_of_circumscribed_circle (s : ℝ) (h : s = 12) : area_of_circle (circumradius_of_equilateral_triangle s) = 48 * Real.pi := by
  sorry

end area_of_circumscribed_circle_l289_289634


namespace total_amount_divided_l289_289784

theorem total_amount_divided (A B C : ℝ) (h1 : A = (2/3) * (B + C)) (h2 : B = (2/3) * (A + C)) (h3 : A = 200) :
  A + B + C = 500 :=
by
  sorry

end total_amount_divided_l289_289784


namespace nested_function_evaluation_l289_289317

def f (x : ℕ) : ℕ := x + 3
def g (x : ℕ) : ℕ := x / 2
def f_inv (x : ℕ) : ℕ := x - 3
def g_inv (x : ℕ) : ℕ := 2 * x

theorem nested_function_evaluation : 
  f (g_inv (f_inv (g_inv (f_inv (g (f 15)))))) = 21 := 
by 
  sorry

end nested_function_evaluation_l289_289317


namespace length_AB_proof_l289_289351

noncomputable def length_AB (AB BC CA : ℝ) (DEF DE EF DF : ℝ) (angle_BAC angle_DEF : ℝ) : ℝ :=
  if h : (angle_BAC = 120 ∧ angle_DEF = 120 ∧ AB = 5 ∧ BC = 17 ∧ CA = 12 ∧ DE = 9 ∧ EF = 15 ∧ DF = 12) then
    (5 * 15) / 17
  else
    0

theorem length_AB_proof : length_AB 5 17 12 9 15 12 120 120 = 75 / 17 := by
  sorry

end length_AB_proof_l289_289351


namespace square_garden_perimeter_l289_289456

theorem square_garden_perimeter (A : ℝ) (hA : A = 450) : 
    ∃ P : ℝ, P = 60 * Real.sqrt 2 :=
  by
    sorry

end square_garden_perimeter_l289_289456


namespace find_a_b_l289_289549

def curve (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

theorem find_a_b :
  ∀ (a b : ℝ),
  (∀ x, (curve x a b) = x^2 + a * x + b) →
  (tangent_line 0 (curve 0 a b)) →
  (tangent_line x y → y = x + 1) →
  (tangent_line x y → ∃ m c, y = m * x + c ∧ m = 1 ∧ c = 1) →
  (∃ a b : ℝ, a = 1 ∧ b = 1) :=
by
  intros a b h_curve h_tangent_line h_tangent_line_form h_tangent_line_eq
  sorry

end find_a_b_l289_289549


namespace total_players_on_ground_l289_289948

theorem total_players_on_ground 
  (cricket_players : ℕ) (hockey_players : ℕ) (football_players : ℕ) (softball_players : ℕ)
  (hcricket : cricket_players = 16) (hhokey : hockey_players = 12) 
  (hfootball : football_players = 18) (hsoftball : softball_players = 13) :
  cricket_players + hockey_players + football_players + softball_players = 59 :=
by
  sorry

end total_players_on_ground_l289_289948


namespace area_of_circumscribed_circle_l289_289644

theorem area_of_circumscribed_circle (s : ℝ) (hs : s = 12) : 
  ∃ A : ℝ, A = 48 * Real.pi := 
by
  sorry

end area_of_circumscribed_circle_l289_289644


namespace chocolates_exceeding_200_l289_289265

-- Define the initial amount of chocolates
def initial_chocolates : ℕ := 3

-- Define the function that computes the amount of chocolates on the nth day
def chocolates_on_day (n : ℕ) : ℕ := initial_chocolates * 3 ^ (n - 1)

-- Define the proof problem
theorem chocolates_exceeding_200 : ∃ (n : ℕ), chocolates_on_day n > 200 :=
by
  -- Proof required here
  sorry

end chocolates_exceeding_200_l289_289265


namespace b_seq_formula_c_seq_arithmetic_c_seq_formula_a_seq_formula_sum_S_5_l289_289827

-- Define the sequences a_n, b_n, and c_n along with their properties

-- Definitions
def a_seq (n : ℕ) : ℕ := sorry            -- Define a_n

def S_seq (n : ℕ) : ℕ := sorry            -- Define S_n

def b_seq (n : ℕ) : ℕ := a_seq (n+1) - 2 * a_seq n

def c_seq (n : ℕ) : ℕ := a_seq n / 2^n

-- Conditions
axiom S_n_condition (n : ℕ) : S_seq (n+1) = 4 * a_seq n + 2
axiom a_1_condition : a_seq 1 = 1

-- Goals
theorem b_seq_formula (n : ℕ) : b_seq n = 3 * 2^(n-1) := sorry

theorem c_seq_arithmetic (n : ℕ) : c_seq (n+1) - c_seq n = 3 / 4 := sorry

theorem c_seq_formula (n : ℕ) : c_seq n = (3 * n - 1) / 4 := sorry

theorem a_seq_formula (n : ℕ) : a_seq n = (3 * n - 1) * 2^(n-2) := sorry

theorem sum_S_5 : S_seq 5 = 178 := sorry

end b_seq_formula_c_seq_arithmetic_c_seq_formula_a_seq_formula_sum_S_5_l289_289827


namespace smallest_value_fraction_l289_289998

theorem smallest_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 6) :
  ∃ k : ℝ, (∀ (x y : ℝ), (-6 ≤ x ∧ x ≤ -3) → (3 ≤ y ∧ y ≤ 6) → k ≤ (x + y) / x) ∧ k = 0 :=
by
  sorry

end smallest_value_fraction_l289_289998


namespace negation_divisible_by_5_is_odd_l289_289338

theorem negation_divisible_by_5_is_odd : 
  ¬∀ n : ℤ, (n % 5 = 0) → (n % 2 ≠ 0) ↔ ∃ n : ℤ, (n % 5 = 0) ∧ (n % 2 = 0) := 
by 
  sorry

end negation_divisible_by_5_is_odd_l289_289338


namespace sphere_surface_area_l289_289378

theorem sphere_surface_area (a : ℝ) (d : ℝ) (S : ℝ) : 
  a = 3 → d = Real.sqrt 7 → S = 40 * Real.pi := by
  sorry

end sphere_surface_area_l289_289378


namespace trey_total_time_is_two_hours_l289_289583

-- Define the conditions
def num_cleaning_tasks := 7
def num_shower_tasks := 1
def num_dinner_tasks := 4
def time_per_task := 10 -- in minutes
def minutes_per_hour := 60

-- Total tasks
def total_tasks := num_cleaning_tasks + num_shower_tasks + num_dinner_tasks

-- Total time in minutes
def total_time_minutes := total_tasks * time_per_task

-- Total time in hours
def total_time_hours := total_time_minutes / minutes_per_hour

-- Prove that the total time Trey will need to complete his list is 2 hours
theorem trey_total_time_is_two_hours : total_time_hours = 2 := by
  sorry

end trey_total_time_is_two_hours_l289_289583


namespace prob_lamp_first_factory_standard_prob_lamp_standard_l289_289038

noncomputable def P_B1 : ℝ := 0.35
noncomputable def P_B2 : ℝ := 0.50
noncomputable def P_B3 : ℝ := 0.15

noncomputable def P_B1_A : ℝ := 0.70
noncomputable def P_B2_A : ℝ := 0.80
noncomputable def P_B3_A : ℝ := 0.90

-- Question A
theorem prob_lamp_first_factory_standard : P_B1 * P_B1_A = 0.245 :=
by 
  sorry

-- Question B
theorem prob_lamp_standard : (P_B1 * P_B1_A) + (P_B2 * P_B2_A) + (P_B3 * P_B3_A) = 0.78 :=
by 
  sorry

end prob_lamp_first_factory_standard_prob_lamp_standard_l289_289038


namespace math_problem_l289_289396

theorem math_problem 
  (a b c : ℝ) 
  (h0 : 0 ≤ a) (h1 : 0 ≤ b) (h2 : 0 ≤ c) 
  (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : a^2 + b^2 = c^2 + ab) : 
  c^2 + ab < a*c + b*c := 
sorry

end math_problem_l289_289396


namespace even_fn_increasing_max_val_l289_289746

variable {f : ℝ → ℝ}

theorem even_fn_increasing_max_val (h_even : ∀ x, f x = f (-x))
    (h_inc_0_5 : ∀ x y, 0 ≤ x → x ≤ y → y ≤ 5 → f x ≤ f y)
    (h_dec_5_inf : ∀ x y, 5 ≤ x → x ≤ y → f y ≤ f x)
    (h_f5 : f 5 = 2) :
    (∀ x y, -5 ≤ x → x ≤ y → y ≤ 0 → f x ≤ f y) ∧ (∀ x, -5 ≤ x → x ≤ 0 → f x ≤ 2) :=
by
    sorry

end even_fn_increasing_max_val_l289_289746


namespace cylinder_height_l289_289371

theorem cylinder_height (base_area : ℝ) (h s : ℝ)
  (h_base : base_area > 0)
  (h_ratio : (1 / 3 * base_area * 4.5) / (base_area * h) = 1 / 6)
  (h_cone_height : s = 4.5) :
  h = 9 :=
by
  -- Proof omitted
  sorry

end cylinder_height_l289_289371


namespace quadratic_real_root_iff_b_range_l289_289101

open Real

theorem quadratic_real_root_iff_b_range (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_real_root_iff_b_range_l289_289101


namespace quadratic_real_roots_l289_289133

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := by 
  sorry

end quadratic_real_roots_l289_289133


namespace abs_ineq_solution_set_l289_289816

theorem abs_ineq_solution_set (x : ℝ) : |x + 1| - |x - 5| < 4 ↔ x < 4 :=
sorry

end abs_ineq_solution_set_l289_289816


namespace tiles_needed_correct_l289_289796

noncomputable def tiles_needed (floor_length : ℝ) (floor_width : ℝ) (tile_length_inch : ℝ) (tile_width_inch : ℝ) (border_width : ℝ) : ℝ :=
  let tile_length := tile_length_inch / 12
  let tile_width := tile_width_inch / 12
  let main_length := floor_length - 2 * border_width
  let main_width := floor_width - 2 * border_width
  let main_area := main_length * main_width
  let tile_area := tile_length * tile_width
  main_area / tile_area

theorem tiles_needed_correct :
  tiles_needed 15 20 3 9 1 = 1248 := 
by 
  sorry -- Proof skipped.

end tiles_needed_correct_l289_289796


namespace n_not_composite_l289_289606

theorem n_not_composite
  (n : ℕ) (h1 : n > 1)
  (a : ℕ) (q : ℕ) (hq_prime : Nat.Prime q)
  (hq1 : q ∣ (n - 1))
  (hq2 : q > Nat.sqrt n - 1)
  (hn_div : n ∣ (a^(n-1) - 1))
  (hgcd : Nat.gcd (a^(n-1)/q - 1) n = 1) :
  ¬ Nat.Prime n :=
sorry

end n_not_composite_l289_289606


namespace cubic_has_one_real_root_iff_l289_289999

theorem cubic_has_one_real_root_iff (a : ℝ) :
  (∃! x : ℝ, x^3 + (1 - a) * x^2 - 2 * a * x + a^2 = 0) ↔ a < -1/4 := by
  sorry

end cubic_has_one_real_root_iff_l289_289999


namespace inequality_smallest_val_l289_289264

open Real

def cot (x : ℝ) := cos x / sin x
def tan (x : ℝ) := sin x / cos x

theorem inequality_smallest_val (a : ℝ) (h : a = -2.52) :
  (∀ x ∈ set.Ioo (-3 * π / 2) (-π), 
  (∛(cot x ^ 2) - ∛(tan x ^ 2)) / (∛(sin x ^ 2) - ∛(cos x ^ 2)) < a) :=
begin
  sorry
end

end inequality_smallest_val_l289_289264


namespace exponents_to_99_l289_289447

theorem exponents_to_99 :
  (1 * 3 / 3^2 / 3^4 / 3^8 * 3^16 * 3^32 * 3^64 = 3^99) :=
sorry

end exponents_to_99_l289_289447


namespace hyperbola_foci_distance_l289_289744

theorem hyperbola_foci_distance :
  (∀ (x y : ℝ), (y = 2 * x + 3) ∨ (y = -2 * x + 7)) →
  (∃ (x y : ℝ), x = 4 ∧ y = 5 ∧ ((y = 2 * x + 3) ∨ (y = -2 * x + 7))) →
  (∃ h : ℝ, h = 6 * Real.sqrt 2) :=
by
  sorry

end hyperbola_foci_distance_l289_289744


namespace determine_slope_l289_289542

theorem determine_slope (a : ℝ) : 
  let l1 := λ x : ℝ, a * x - 2 in
  let l2 := λ x : ℝ, (a + 2) * x + 1 in
  (∀ x1 x2 : ℝ, x1 ≠ x2 ↔ (l1 x1 - l1 x2) / (x1 - x2) * (l2 x1 - l2 x2) / (x1 - x2) = -1) → 
  a = -1 :=
by
  -- proof or steps are not required, use sorry
  sorry

end determine_slope_l289_289542


namespace find_wrong_guess_l289_289464

-- Define the three colors as an inductive type.
inductive Color
| white
| brown
| black

-- Define the bears as a list of colors.
def bears (n : ℕ) : Type := list Color

-- Define the conditions: 
-- There are 1000 bears and each tuple of 3 consecutive bears has all three colors.
def valid_bears (b : bears 1000) : Prop :=
  ∀ i : ℕ, i + 2 < 1000 → 
    ∃ c1 c2 c3 : Color, 
      c1 ∈ b.nth i ∧ c2 ∈ b.nth (i+1) ∧ c3 ∈ b.nth (i+2) ∧ 
      c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3

-- Define Iskander's guesses.
def guesses (b : bears 1000) : Prop :=
  b.nth 1 = some Color.white ∧
  b.nth 19 = some Color.brown ∧
  b.nth 399 = some Color.black ∧
  b.nth 599 = some Color.brown ∧
  b.nth 799 = some Color.white

-- Prove that exactly one of Iskander's guesses is wrong.
def wrong_guess (b : bears 1000) : Prop :=
  (b.nth 19 ≠ some Color.brown) ∧
  valid_bears b ∧
  guesses b →
  ∃ i, i ∈ {1, 19, 399, 599, 799} ∧ (b.nth i ≠ some Color.white ∧ b.nth i ≠ some Color.brown ∧ b.nth i ≠ some Color.black)

theorem find_wrong_guess : 
  ∀ b : bears 1000, 
  valid_bears b → guesses b → wrong_guess b :=
  by
  intros b vb gs
  sorry

end find_wrong_guess_l289_289464


namespace fishing_tomorrow_l289_289863

theorem fishing_tomorrow 
  (P1 : ∀ day : ℕ, 7 ∈ {p | p goes fishing on day})
  (P2 : ∀ day : ℕ, day % 2 = 0 → 8 ∈ {p | p goes fishing on day})
  (P3 : ∀ day : ℕ, day % 3 = 0 → 3 ∈ {p | p goes fishing on day})
  (P4 : ℕ)
  (yesterday : ℕ)
  (today : ℕ)
  (tomorrow : ℕ)
  (hyesterday : yesterday = 12)
  (htoday : today = 10)
  : tomorrow = 15 := by
  sorry

end fishing_tomorrow_l289_289863


namespace hall_width_l289_289889

theorem hall_width
  (L H E C : ℝ)
  (hL : L = 20)
  (hH : H = 5)
  (hE : E = 57000)
  (hC : C = 60) :
  ∃ w : ℝ, (w * 50 + 100) * C = E ∧ w = 17 :=
by
  use 17
  simp [hL, hH, hE, hC]
  sorry

end hall_width_l289_289889


namespace math_problem_l289_289321

-- Conditions
variables {f g : ℝ → ℝ}
axiom f_zero : f 0 = 0
axiom inequality : ∀ x y : ℝ, g (x - y) ≥ f x * f y + g x * g y

-- Problem Statement
theorem math_problem : ∀ x : ℝ, f x ^ 2008 + g x ^ 2008 ≤ 1 :=
by
  sorry

end math_problem_l289_289321


namespace real_roots_of_quad_eq_l289_289327

theorem real_roots_of_quad_eq (p q a : ℝ) (h : p^2 - 4 * q > 0) : 
  (2 * a - p)^2 + 3 * (p^2 - 4 * q) > 0 := 
by
  sorry

end real_roots_of_quad_eq_l289_289327


namespace total_cost_oranges_mangoes_l289_289669

theorem total_cost_oranges_mangoes
  (initial_price_orange : ℝ)
  (initial_price_mango : ℝ)
  (price_increase_percentage : ℝ)
  (quantity_oranges : ℕ)
  (quantity_mangoes : ℕ) :
  initial_price_orange = 40 →
  initial_price_mango = 50 →
  price_increase_percentage = 0.15 →
  quantity_oranges = 10 →
  quantity_mangoes = 10 →
  let new_price_orange := initial_price_orange * (1 + price_increase_percentage)
  let new_price_mango := initial_price_mango * (1 + price_increase_percentage)
  let total_cost_oranges := new_price_orange * quantity_oranges
  let total_cost_mangoes := new_price_mango * quantity_mangoes
  let total_cost := total_cost_oranges + total_cost_mangoes in
  total_cost = 1035 :=
by
  intros h_orange h_mango h_percentage h_qty_oranges h_qty_mangoes
  let new_price_orange := 40 * (1 + 0.15)
  let new_price_mango := 50 * (1 + 0.15)
  let total_cost_oranges := new_price_orange * 10
  let total_cost_mangoes := new_price_mango * 10
  let total_cost := total_cost_oranges + total_cost_mangoes
  sorry

end total_cost_oranges_mangoes_l289_289669


namespace total_amount_l289_289383

theorem total_amount (A N J : ℕ) (h1 : A = N - 5) (h2 : J = 4 * N) (h3 : J = 48) : A + N + J = 67 :=
by
  -- Proof will be constructed here
  sorry

end total_amount_l289_289383


namespace largest_number_by_replacement_l289_289555

theorem largest_number_by_replacement 
  (n : ℝ) (n_1 n_3 n_6 n_8 : ℝ)
  (h : n = -0.3168)
  (h1 : n_1 = -0.3468)
  (h3 : n_3 = -0.4168)
  (h6 : n_6 = -0.3148)
  (h8 : n_8 = -0.3164)
  : n_6 > n_1 ∧ n_6 > n_3 ∧ n_6 > n_8 := 
by {
  -- Proof goes here
  sorry
}

end largest_number_by_replacement_l289_289555
