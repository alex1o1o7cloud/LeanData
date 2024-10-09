import Mathlib

namespace mulch_cost_l1993_199330

-- Definitions based on conditions
def cost_per_cubic_foot : ℕ := 8
def cubic_yard_to_cubic_feet : ℕ := 27
def volume_in_cubic_yards : ℕ := 7

-- Target statement to prove
theorem mulch_cost :
    (volume_in_cubic_yards * cubic_yard_to_cubic_feet) * cost_per_cubic_foot = 1512 := by
  sorry

end mulch_cost_l1993_199330


namespace inequalities_hold_l1993_199377

theorem inequalities_hold (b : ℝ) :
  (b ∈ Set.Ioo (-(1 : ℝ) - Real.sqrt 2 / 4) (0 : ℝ) ∨ b < -(1 : ℝ) - Real.sqrt 2 / 4) →
  (∀ x y : ℝ, 2 * b * Real.cos (2 * (x - y)) + 8 * b^2 * Real.cos (x - y) + 8 * b^2 * (b + 1) + 5 * b < 0) ∧ 
  (∀ x y : ℝ, x^2 + y^2 + 1 > 2 * b * x + 2 * y + b - b^2) :=
by 
  intro h
  sorry

end inequalities_hold_l1993_199377


namespace alice_speed_exceed_l1993_199381

theorem alice_speed_exceed (d : ℝ) (t₁ t₂ : ℝ) (t₃ : ℝ) :
  d = 220 →
  t₁ = 220 / 40 →
  t₂ = t₁ - 0.5 →
  t₃ = 220 / t₂ →
  t₃ = 44 :=
by
  intros h1 h2 h3 h4
  sorry

end alice_speed_exceed_l1993_199381


namespace exponents_subtraction_l1993_199321

theorem exponents_subtraction (m n : ℕ) (hm : 3 ^ m = 8) (hn : 3 ^ n = 2) : 3 ^ (m - n) = 4 := 
by
  sorry

end exponents_subtraction_l1993_199321


namespace f_positive_when_a_1_f_negative_solution_sets_l1993_199380

section

variable (f : ℝ → ℝ) (a x : ℝ)

def f_def := f x = (x - a) * (x - 2)

-- (Ⅰ) Problem statement
theorem f_positive_when_a_1 : (∀ x, f_def f 1 x → f x > 0 ↔ (x < 1) ∨ (x > 2)) :=
by sorry

-- (Ⅱ) Problem statement
theorem f_negative_solution_sets (a : ℝ) : 
  (∀ x, f_def f a x ∧ a = 2 → False) ∧ 
  (∀ x, f_def f a x ∧ a > 2 → 2 < x ∧ x < a) ∧ 
  (∀ x, f_def f a x ∧ a < 2 → a < x ∧ x < 2) :=
by sorry

end

end f_positive_when_a_1_f_negative_solution_sets_l1993_199380


namespace plot_area_is_nine_hectares_l1993_199343

-- Definition of the dimensions of the plot
def length := 450
def width := 200

-- Definition of conversion factor from square meters to hectares
def sqMetersPerHectare := 10000

-- Calculated area in hectares
def area_hectares := (length * width) / sqMetersPerHectare

-- Theorem statement: prove that the area in hectares is 9
theorem plot_area_is_nine_hectares : area_hectares = 9 := 
by
  sorry

end plot_area_is_nine_hectares_l1993_199343


namespace product_of_areas_eq_square_of_volume_l1993_199360

theorem product_of_areas_eq_square_of_volume 
(x y z d : ℝ) 
(h1 : d^2 = x^2 + y^2 + z^2) :
  (x * y) * (y * z) * (z * x) = (x * y * z) ^ 2 :=
by sorry

end product_of_areas_eq_square_of_volume_l1993_199360


namespace hemisphere_surface_area_l1993_199375

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (h1: 0 < π) (h2: A = 3) (h3: S = 4 * π * r^2):
  ∃ t, t = 9 :=
by
  sorry

end hemisphere_surface_area_l1993_199375


namespace sum_of_variables_l1993_199399

variables (a b c d : ℝ)

theorem sum_of_variables :
  (a - 2)^2 + (b - 5)^2 + (c - 6)^2 + (d - 3)^2 = 0 → a + b + c + d = 16 :=
by
  intro h
  -- your proof goes here
  sorry

end sum_of_variables_l1993_199399


namespace coin_loading_impossible_l1993_199345

theorem coin_loading_impossible (p q : ℝ) (h1 : p ≠ 1 - p) (h2 : q ≠ 1 - q) 
    (h3 : p * q = 1/4) (h4 : p * (1 - q) = 1/4) (h5 : (1 - p) * q = 1/4) (h6 : (1 - p) * (1 - q) = 1/4) : 
    false := 
by 
  sorry

end coin_loading_impossible_l1993_199345


namespace billy_videos_within_limit_l1993_199316

def total_videos_watched_within_time_limit (time_limit : ℕ) (video_time : ℕ) (search_time : ℕ) (break_time : ℕ) (num_trials : ℕ) (videos_per_trial : ℕ) (categories : ℕ) (videos_per_category : ℕ) : ℕ :=
  let total_trial_time := videos_per_trial * video_time + search_time + break_time
  let total_category_time := videos_per_category * video_time
  let full_trial_time := num_trials * total_trial_time
  let full_category_time := categories * total_category_time
  let total_time := full_trial_time + full_category_time
  let non_watching_time := search_time * num_trials + break_time * (num_trials - 1)
  let available_time := time_limit - non_watching_time
  let max_videos := available_time / video_time
  max_videos

theorem billy_videos_within_limit : total_videos_watched_within_time_limit 90 4 3 5 5 15 2 10 = 13 := by
  sorry

end billy_videos_within_limit_l1993_199316


namespace c_n_monotonically_decreasing_l1993_199359

variable (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ)

theorem c_n_monotonically_decreasing 
    (h_a0 : a 0 = 0)
    (h_b : ∀ n ≥ 1, b n = a n - a (n - 1))
    (h_c : ∀ n ≥ 1, c n = a n / n)
    (h_bn_decrease : ∀ n ≥ 1, b n ≥ b (n + 1)) : 
    ∀ n ≥ 2, c n ≤ c (n - 1) := 
by
  sorry

end c_n_monotonically_decreasing_l1993_199359


namespace prove_fractions_sum_equal_11_l1993_199308

variable (a b c : ℝ)

-- Given conditions
axiom h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -9
axiom h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 10

-- The proof problem statement
theorem prove_fractions_sum_equal_11 : (b / (a + b) + c / (b + c) + a / (c + a)) = 11 :=
by
  sorry

end prove_fractions_sum_equal_11_l1993_199308


namespace min_value_ineq_inequality_proof_l1993_199353

variable (a b x1 x2 : ℝ)
variable (ha_pos : 0 < a) (hb_pos : 0 < b) (hx1_pos : 0 < x1) (hx2_pos : 0 < x2) (hab_sum : a + b = 1)

-- First problem: Prove that the minimum value of the given expression is 6.
theorem min_value_ineq : (x1 / a) + (x2 / b) + (2 / (x1 * x2)) ≥ 6 := by
  sorry

-- Second problem: Prove the given inequality.
theorem inequality_proof : (a * x1 + b * x2) * (a * x2 + b * x1) ≥ x1 * x2 := by
  sorry

end min_value_ineq_inequality_proof_l1993_199353


namespace solution_set_of_inequality_l1993_199398

theorem solution_set_of_inequality :
  {x : ℝ | abs (x^2 - 5 * x + 6) < x^2 - 4} = { x : ℝ | x > 2 } :=
sorry

end solution_set_of_inequality_l1993_199398


namespace train_half_speed_time_l1993_199379

-- Definitions for Lean
variables (S T D : ℝ)

-- Conditions
axiom cond1 : D = S * T
axiom cond2 : D = (1 / 2) * S * (T + 4)

-- Theorem Statement
theorem train_half_speed_time : 
  (T = 4) → (4 + 4 = 8) := 
by 
  intros hT
  simp [hT]

end train_half_speed_time_l1993_199379


namespace filling_rate_in_cubic_meters_per_hour_l1993_199303

def barrels_per_minute_filling_rate : ℝ := 3
def liters_per_barrel : ℝ := 159
def liters_per_cubic_meter : ℝ := 1000
def minutes_per_hour : ℝ := 60

theorem filling_rate_in_cubic_meters_per_hour :
  (barrels_per_minute_filling_rate * liters_per_barrel / liters_per_cubic_meter * minutes_per_hour) = 28.62 :=
sorry

end filling_rate_in_cubic_meters_per_hour_l1993_199303


namespace least_possible_area_l1993_199350

def perimeter (x y : ℕ) : ℕ := 2 * (x + y)

def area (x y : ℕ) : ℕ := x * y

theorem least_possible_area :
  ∃ (x y : ℕ), 
    perimeter x y = 120 ∧ 
    (∀ x y, perimeter x y = 120 → area x y ≥ 59) ∧ 
    area x y = 59 := 
sorry

end least_possible_area_l1993_199350


namespace bad_carrots_eq_13_l1993_199335

-- Define the number of carrots picked by Haley
def haley_picked : ℕ := 39

-- Define the number of carrots picked by her mom
def mom_picked : ℕ := 38

-- Define the number of good carrots
def good_carrots : ℕ := 64

-- Define the total number of carrots picked
def total_carrots : ℕ := haley_picked + mom_picked

-- State the theorem to prove the number of bad carrots
theorem bad_carrots_eq_13 : total_carrots - good_carrots = 13 := by
  sorry

end bad_carrots_eq_13_l1993_199335


namespace james_and_david_probability_l1993_199338

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem james_and_david_probability :
  let total_workers := 22
  let chosen_workers := 4
  let j_and_d_chosen := 2
  (choose 20 2) / (choose 22 4) = (2 / 231) :=
by
  sorry

end james_and_david_probability_l1993_199338


namespace mutually_exclusive_events_not_complementary_l1993_199305

def event_a (ball: ℕ) (box: ℕ): Prop := ball = 1 ∧ box = 1
def event_b (ball: ℕ) (box: ℕ): Prop := ball = 1 ∧ box = 2

theorem mutually_exclusive_events_not_complementary :
  (∀ ball box, event_a ball box → ¬ event_b ball box) ∧ 
  (∃ box, ¬((event_a 1 box) ∨ (event_b 1 box))) :=
by
  sorry

end mutually_exclusive_events_not_complementary_l1993_199305


namespace simplify_sqrt_square_l1993_199331

theorem simplify_sqrt_square (h : Real.sqrt 7 < 3) : Real.sqrt ((Real.sqrt 7 - 3)^2) = 3 - Real.sqrt 7 :=
by
  sorry

end simplify_sqrt_square_l1993_199331


namespace Mike_found_seashells_l1993_199337

/-!
# Problem:
Mike found some seashells on the beach, he gave Tom 49 of his seashells.
He has thirteen seashells left. How many seashells did Mike find on the beach?

# Conditions:
1. Mike gave Tom 49 seashells.
2. Mike has 13 seashells left.

# Proof statement:
Prove that Mike found 62 seashells on the beach.
-/

/-- Define the variables and conditions -/
def seashells_given_to_Tom : ℕ := 49
def seashells_left_with_Mike : ℕ := 13

/-- Prove that Mike found 62 seashells on the beach -/
theorem Mike_found_seashells : 
  seashells_given_to_Tom + seashells_left_with_Mike = 62 := 
by
  -- This is where the proof would go
  sorry

end Mike_found_seashells_l1993_199337


namespace maximize_profit_l1993_199300

def total_orders := 100
def max_days := 160
def time_per_A := 5 / 4 -- days
def time_per_B := 5 / 3 -- days
def profit_per_A := 0.5 -- (10,000 RMB)
def profit_per_B := 0.8 -- (10,000 RMB)

theorem maximize_profit : 
  ∃ (x : ℝ) (y : ℝ), 
    (time_per_A * x + time_per_B * (total_orders - x) ≤ max_days) ∧ 
    (y = -0.3 * x + 80) ∧ 
    (x = 16) ∧ 
    (y = 75.2) :=
by 
  sorry

end maximize_profit_l1993_199300


namespace no_intersection_abs_eq_l1993_199367

theorem no_intersection_abs_eq (x : ℝ) : ∀ y : ℝ, y = |3 * x + 6| → y = -|2 * x - 4| → false := 
by
  sorry

end no_intersection_abs_eq_l1993_199367


namespace max_discriminant_l1993_199320

noncomputable def f (a b c x : ℤ) := a * x^2 + b * x + c

theorem max_discriminant (a b c u v w : ℤ)
  (h1 : u ≠ v) (h2 : v ≠ w) (h3 : u ≠ w)
  (hu : f a b c u = 0)
  (hv : f a b c v = 0)
  (hw : f a b c w = 2) :
  ∃ (a b c : ℤ), b^2 - 4 * a * c = 16 :=
sorry

end max_discriminant_l1993_199320


namespace incorrect_statement_C_l1993_199373

noncomputable def f (x a b c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem incorrect_statement_C (a b c : ℝ) (x0 : ℝ) (h_local_min : ∀ y, f x0 a b c ≤ f y a b c) :
  ∃ z, z < x0 ∧ ¬ (f z a b c ≤ f (z + ε) a b c) := sorry

end incorrect_statement_C_l1993_199373


namespace chord_length_l1993_199356

theorem chord_length (ρ θ : ℝ) (p : ℝ) : 
  (∀ θ, ρ = 6 * Real.cos θ) ∧ (θ = Real.pi / 4) → 
  ∃ l : ℝ, l = 3 * Real.sqrt 2 :=
by
  sorry

end chord_length_l1993_199356


namespace hot_dogs_served_today_l1993_199336

theorem hot_dogs_served_today : 9 + 2 = 11 :=
by
  sorry

end hot_dogs_served_today_l1993_199336


namespace marilyn_total_caps_l1993_199324

def marilyn_initial_caps : ℝ := 51.0
def nancy_gives_caps : ℝ := 36.0
def total_caps (initial: ℝ) (given: ℝ) : ℝ := initial + given

theorem marilyn_total_caps : total_caps marilyn_initial_caps nancy_gives_caps = 87.0 :=
by
  sorry

end marilyn_total_caps_l1993_199324


namespace Haley_has_25_necklaces_l1993_199339

theorem Haley_has_25_necklaces (J H Q : ℕ) 
  (h1 : H = J + 5) 
  (h2 : Q = J / 2) 
  (h3 : H = Q + 15) : 
  H = 25 := 
sorry

end Haley_has_25_necklaces_l1993_199339


namespace max_buses_in_city_l1993_199347

theorem max_buses_in_city (n : ℕ) (stops : ℕ) (shared : ℕ) (condition1 : n = 9) (condition2 : stops = 3) (condition3 : shared ≤ 1) : n = 12 :=
sorry

end max_buses_in_city_l1993_199347


namespace negation_of_proposition_l1993_199307

theorem negation_of_proposition :
  (¬ ∃ x_0 : ℝ, x_0^3 - x_0^2 + 1 ≥ 0) ↔ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0 :=
by sorry

end negation_of_proposition_l1993_199307


namespace find_m_l1993_199323

open Real

namespace VectorPerpendicular

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (-1, m)

def perpendicular (v₁ v₂ : ℝ × ℝ) : Prop := (v₁.1 * v₂.1 + v₁.2 * v₂.2) = 0

theorem find_m (m : ℝ) (h : perpendicular a (b m)) : m = 1 / 2 :=
by
  sorry -- Proof is omitted

end VectorPerpendicular

end find_m_l1993_199323


namespace sun_salutations_per_year_l1993_199326

-- Definitions 
def sun_salutations_per_weekday : ℕ := 5
def weekdays_per_week : ℕ := 5
def weeks_per_year : ℕ := 52

-- Problem statement to prove
theorem sun_salutations_per_year :
  sun_salutations_per_weekday * weekdays_per_week * weeks_per_year = 1300 :=
by
  sorry

end sun_salutations_per_year_l1993_199326


namespace mixed_gender_groups_l1993_199358

theorem mixed_gender_groups (boys girls : ℕ) (h_boys : boys = 28) (h_girls : girls = 4) :
  ∃ groups : ℕ, (groups ≤ girls) ∧ (groups * 2 ≤ boys) ∧ groups = 4 :=
by
   sorry

end mixed_gender_groups_l1993_199358


namespace problem_f_of_f_neg1_eq_neg1_l1993_199318

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(1 - x) else 1 - Real.logb 2 x

-- State the proposition to be proved
theorem problem_f_of_f_neg1_eq_neg1 : f (f (-1)) = -1 := by
  sorry

end problem_f_of_f_neg1_eq_neg1_l1993_199318


namespace units_digit_of_p_is_6_l1993_199329

theorem units_digit_of_p_is_6 (p : ℕ) (h_even : Even p) (h_units_p_plus_1 : (p + 1) % 10 = 7) (h_units_p3_minus_p2 : ((p^3) % 10 - (p^2) % 10) % 10 = 0) : p % 10 = 6 := 
by 
  -- proof steps go here
  sorry

end units_digit_of_p_is_6_l1993_199329


namespace rival_awards_l1993_199395

theorem rival_awards (scott_awards jessie_awards rival_awards : ℕ)
  (h1 : scott_awards = 4)
  (h2 : jessie_awards = 3 * scott_awards)
  (h3 : rival_awards = 2 * jessie_awards) :
  rival_awards = 24 :=
by sorry

end rival_awards_l1993_199395


namespace evaluate_fraction_l1993_199302

theorem evaluate_fraction :
  1 + 1 / (2 + 1 / (3 + 1 / (3 + 3))) = 63 / 44 := 
by
  -- Skipping the proof part with 'sorry'
  sorry

end evaluate_fraction_l1993_199302


namespace tan2α_sin_β_l1993_199314

open Real

variables {α β : ℝ}

axiom α_acute : 0 < α ∧ α < π / 2
axiom β_acute : 0 < β ∧ β < π / 2
axiom sin_α : sin α = 4 / 5
axiom cos_alpha_beta : cos (α + β) = 5 / 13

theorem tan2α : tan 2 * α = -24 / 7 :=
by sorry

theorem sin_β : sin β = 16 / 65 :=
by sorry

end tan2α_sin_β_l1993_199314


namespace revenue_comparison_l1993_199309

theorem revenue_comparison 
  (D N J F : ℚ) 
  (hN : N = (2 / 5) * D) 
  (hJ : J = (2 / 25) * D) 
  (hF : F = (3 / 4) * D) : 
  D / ((N + J + F) / 3) = 100 / 41 := 
by 
  sorry

end revenue_comparison_l1993_199309


namespace adam_cat_food_packages_l1993_199388

theorem adam_cat_food_packages (c : ℕ) 
  (dog_food_packages : ℕ := 7) 
  (cans_per_cat_package : ℕ := 10) 
  (cans_per_dog_package : ℕ := 5) 
  (extra_cat_food_cans : ℕ := 55) 
  (total_dog_cans : ℕ := dog_food_packages * cans_per_dog_package) 
  (total_cat_cans : ℕ := c * cans_per_cat_package)
  (h : total_cat_cans = total_dog_cans + extra_cat_food_cans) : 
  c = 9 :=
by
  sorry

end adam_cat_food_packages_l1993_199388


namespace contains_zero_l1993_199366

theorem contains_zero (a b c d e f: ℕ) (h1: 1 ≤ a ∧ a ≤ 9) (h2: 1 ≤ b ∧ b ≤ 9) (h3: 0 ≤ c ∧ c ≤ 9) 
  (h4: 1 ≤ d ∧ d ≤ 9) (h5: 0 ≤ e ∧ e ≤ 9) (h6: 0 ≤ f ∧ f ≤ 9) 
  (h7: c ≠ f) (h8: 10^4*a + 10^3*b + 10^2*c + 10^1*d + e + 10^4*a + 10^3*b + 10^2*f + 10^1*d + e = 111111) :
  c = 0 ∨ f = 0 := 
sorry

end contains_zero_l1993_199366


namespace molly_christmas_shipping_cost_l1993_199354

def cost_per_package : ℕ := 5
def num_parents : ℕ := 2
def num_brothers : ℕ := 3
def num_sisters_in_law_per_brother : ℕ := 1
def num_children_per_brother : ℕ := 2

def total_relatives : ℕ :=
  num_parents + num_brothers + (num_brothers * num_sisters_in_law_per_brother) + (num_brothers * num_children_per_brother)

theorem molly_christmas_shipping_cost : total_relatives * cost_per_package = 70 :=
by
  sorry

end molly_christmas_shipping_cost_l1993_199354


namespace total_potatoes_l1993_199315

open Nat

theorem total_potatoes (P T R : ℕ) (h1 : P = 5) (h2 : T = 6) (h3 : R = 48) : P + (R / T) = 13 := by
  sorry

end total_potatoes_l1993_199315


namespace seating_profession_solution_l1993_199397

inductive Profession
| architect
| barista
| veterinarian
| guitarist

open Profession

inductive Friend
| Andrey
| Boris
| Vyacheslav
| Gennady

open Friend

structure SeatArrangement :=
(seat1: Friend)
(seat2: Friend)
(seat3: Friend)
(seat4: Friend)

structure ProfessionAssignment :=
(Andrey_profession: Profession)
(Boris_profession: Profession)
(Vyacheslav_profession: Profession)
(Gennady_profession: Profession)

noncomputable def correct_assignment (seats: SeatArrangement) : ProfessionAssignment :=
{ Andrey_profession := veterinarian,
  Boris_profession := architect,
  Vyacheslav_profession := guitarist,
  Gennady_profession := barista }

theorem seating_profession_solution
  (seats: SeatArrangement)
  (cond1 : ∀ f, f ∈ [seats.seat2, seats.seat3] ↔ ∃ p, p ∈ [architect, guitarist])
  (cond2 : seats.seat2 = Boris)
  (cond3 : (seats.seat3 = Vyacheslav ∧ seats.seat1 = Andrey) ∨ (seats.seat4 = Vyacheslav ∧ seats.seat2 = Andrey))
  (cond4 : (seats.seat1 = Andrey ∨ seats.seat2 = Andrey) ∧ Andrey ∉ [seats.seat1, seats.seat4])
  (cond5 : seats.seat1 ≠ Vyacheslav ∧ (seats.seat2 ≠ Gennady ∨ seats.seat3 ≠ Gennady)) :
  correct_assignment seats = 
  { Andrey_profession := veterinarian,
    Boris_profession := architect,
    Vyacheslav_profession := guitarist,
    Gennady_profession := barista } :=
sorry

end seating_profession_solution_l1993_199397


namespace max_value_expr_l1993_199370

theorem max_value_expr (x y : ℝ) : (2 * x + 3 * y + 4) / (Real.sqrt (x^4 + y^2 + 1)) ≤ Real.sqrt 29 := sorry

end max_value_expr_l1993_199370


namespace simplify_expression_l1993_199334

theorem simplify_expression (x : ℝ) (h : x ≠ 0) : 
  (x-2) ^ 2 - x * (x-1) + (x^3 - 4 * x^2) / x^2 = -2 * x := 
by 
  sorry

end simplify_expression_l1993_199334


namespace constant_ratio_arithmetic_progressions_l1993_199301

theorem constant_ratio_arithmetic_progressions
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d p a1 b1 : ℝ)
  (h_a : ∀ k : ℕ, a (k + 1) = a1 + k * d)
  (h_b : ∀ k : ℕ, b (k + 1) = b1 + k * p)
  (h_pos : ∀ k : ℕ, a (k + 1) > 0 ∧ b (k + 1) > 0)
  (h_int : ∀ k : ℕ, ∃ n : ℤ, (a (k + 1) / b (k + 1)) = n) :
  ∃ r : ℝ, ∀ k : ℕ, (a (k + 1) / b (k + 1)) = r :=
by
  sorry

end constant_ratio_arithmetic_progressions_l1993_199301


namespace salt_solution_proof_l1993_199364

theorem salt_solution_proof (x : ℝ) (P : ℝ) (hx : x = 28.571428571428573) :
  ((P / 100) * 100 + x) = 0.30 * (100 + x) → P = 10 :=
by
  sorry

end salt_solution_proof_l1993_199364


namespace a_squared_gt_b_squared_l1993_199363

theorem a_squared_gt_b_squared {a b : ℝ} (h : a ≠ 0) (hb : b ≠ 0) (hb_domain : b > -1 ∧ b < 1) (h_eq : a = Real.log (1 + b) - Real.log (1 - b)) :
  a^2 > b^2 := 
sorry

end a_squared_gt_b_squared_l1993_199363


namespace percent_of_pizza_not_crust_l1993_199312

theorem percent_of_pizza_not_crust (total_weight crust_weight : ℝ) (h_total : total_weight = 800) (h_crust : crust_weight = 200) :
  (total_weight - crust_weight) / total_weight * 100 = 75 :=
by
  sorry

end percent_of_pizza_not_crust_l1993_199312


namespace largest_stores_visited_l1993_199365

theorem largest_stores_visited 
  (stores : ℕ) (total_visits : ℕ) (shoppers : ℕ) 
  (two_store_visitors : ℕ) (min_visits_per_person : ℕ)
  (h1 : stores = 8)
  (h2 : total_visits = 22)
  (h3 : shoppers = 12)
  (h4 : two_store_visitors = 8)
  (h5 : min_visits_per_person = 1)
  : ∃ (max_stores : ℕ), max_stores = 3 := 
by 
  -- Define the exact details given in the conditions
  have h_total_two_store_visits : two_store_visitors * 2 = 16 := by sorry
  have h_remaining_visits : total_visits - 16 = 6 := by sorry
  have h_remaining_shoppers : shoppers - two_store_visitors = 4 := by sorry
  have h_each_remaining_one_visit : 4 * 1 = 4 := by sorry
  -- Prove the largest number of stores visited by any one person is 3
  have h_max_stores : 1 + 2 = 3 := by sorry
  exact ⟨3, h_max_stores⟩

end largest_stores_visited_l1993_199365


namespace parcel_total_weight_l1993_199351

theorem parcel_total_weight (x y z : ℝ) 
  (h1 : x + y = 132) 
  (h2 : y + z = 146) 
  (h3 : z + x = 140) : 
  x + y + z = 209 :=
by
  sorry

end parcel_total_weight_l1993_199351


namespace find_length_QS_l1993_199385

theorem find_length_QS 
  (cosR : ℝ) (RS : ℝ) (QR : ℝ) (QS : ℝ)
  (h1 : cosR = 3 / 5)
  (h2 : RS = 10)
  (h3 : cosR = QR / RS) :
  QS = 8 :=
by
  sorry

end find_length_QS_l1993_199385


namespace carp_and_population_l1993_199348

-- Define the characteristics of an individual and a population
structure Individual where
  birth : Prop
  death : Prop
  gender : Prop
  age : Prop

structure Population where
  birth_rate : Prop
  death_rate : Prop
  gender_ratio : Prop
  age_composition : Prop

-- Define the conditions as hypotheses
axiom a : Individual
axiom b : Population

-- State the theorem: If "a" has characteristics of an individual and "b" has characteristics
-- of a population, then "a" is a carp and "b" is a carp population
theorem carp_and_population : 
  (a.birth ∧ a.death ∧ a.gender ∧ a.age) ∧
  (b.birth_rate ∧ b.death_rate ∧ b.gender_ratio ∧ b.age_composition) →
  (a = ⟨True, True, True, True⟩ ∧ b = ⟨True, True, True, True⟩) := 
by 
  sorry

end carp_and_population_l1993_199348


namespace distance_between_andrey_and_valentin_l1993_199322

-- Definitions based on conditions
def speeds_relation_andrey_boris (a b : ℝ) := b = 0.94 * a
def speeds_relation_boris_valentin (b c : ℝ) := c = 0.95 * b

theorem distance_between_andrey_and_valentin
  (a b c : ℝ)
  (h1 : speeds_relation_andrey_boris a b)
  (h2 : speeds_relation_boris_valentin b c)
  : 1000 - 1000 * c / a = 107 :=
by
  sorry

end distance_between_andrey_and_valentin_l1993_199322


namespace gumballs_per_package_l1993_199389

theorem gumballs_per_package (total_gumballs : ℕ) (packages : ℝ) (h1 : total_gumballs = 100) (h2 : packages = 20.0) :
  total_gumballs / packages = 5 :=
by sorry

end gumballs_per_package_l1993_199389


namespace mass_percentage_O_in_N2O_is_approximately_36_35_l1993_199355

noncomputable def atomic_mass_N : ℝ := 14.01
noncomputable def atomic_mass_O : ℝ := 16.00

noncomputable def number_of_N : ℕ := 2
noncomputable def number_of_O : ℕ := 1

noncomputable def molar_mass_N2O : ℝ := (number_of_N * atomic_mass_N) + (number_of_O * atomic_mass_O)

noncomputable def mass_percentage_O : ℝ := (atomic_mass_O / molar_mass_N2O) * 100

theorem mass_percentage_O_in_N2O_is_approximately_36_35 :
  abs (mass_percentage_O - 36.35) < 0.01 := sorry

end mass_percentage_O_in_N2O_is_approximately_36_35_l1993_199355


namespace distance_ratio_l1993_199311

-- Define the distances as given in the conditions
def distance_from_city_sky_falls := 8 -- Distance in miles
def distance_from_city_rocky_mist := 400 -- Distance in miles

theorem distance_ratio : distance_from_city_rocky_mist / distance_from_city_sky_falls = 50 := 
by
  -- Proof skipped
  sorry

end distance_ratio_l1993_199311


namespace sum_of_powers_of_two_l1993_199346

theorem sum_of_powers_of_two : 2^4 + 2^4 + 2^4 + 2^4 = 2^6 :=
by
  sorry

end sum_of_powers_of_two_l1993_199346


namespace total_animals_in_jacobs_flock_l1993_199332

-- Define the conditions of the problem
def one_third_of_animals_are_goats (total goats : ℕ) : Prop := 
  3 * goats = total

def twelve_more_sheep_than_goats (goats sheep : ℕ) : Prop :=
  sheep = goats + 12

-- Define the main theorem to prove
theorem total_animals_in_jacobs_flock : 
  ∃ total goats sheep : ℕ, one_third_of_animals_are_goats total goats ∧ 
                           twelve_more_sheep_than_goats goats sheep ∧ 
                           total = 36 := 
by
  sorry

end total_animals_in_jacobs_flock_l1993_199332


namespace sum_of_intercepts_l1993_199391

theorem sum_of_intercepts (x y : ℝ) (hx : y + 3 = 5 * (x - 6)) : 
  let x_intercept := 6 + 3/5;
  let y_intercept := -33;
  x_intercept + y_intercept = -26.4 := by
  sorry

end sum_of_intercepts_l1993_199391


namespace fraction_division_l1993_199357

theorem fraction_division (a b : ℚ) (ha : a = 3) (hb : b = 4) :
  (1 / b) / (1 / a) = 3 / 4 :=
by 
  -- Solve the proof
  sorry

end fraction_division_l1993_199357


namespace range_of_m_l1993_199368

theorem range_of_m (m : ℝ) : (∀ x : ℝ, m^2 * x^2 + 2 * m * x - 4 < 2 * x^2 + 4 * x) ↔ -2 < m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_l1993_199368


namespace cakes_served_yesterday_l1993_199383

theorem cakes_served_yesterday:
  ∃ y : ℕ, (5 + 6 + y = 14) ∧ y = 3 := 
by
  sorry

end cakes_served_yesterday_l1993_199383


namespace height_of_pole_l1993_199362

/-- A telephone pole is supported by a steel cable extending from the top of the pole to a point on the ground 3 meters from its base.
When Leah, who is 1.5 meters tall, stands 2.5 meters from the base of the pole towards the point where the cable is attached to the ground,
her head just touches the cable. Prove that the height of the pole is 9 meters. -/
theorem height_of_pole 
  (cable_length_from_base : ℝ)
  (leah_distance_from_base : ℝ)
  (leah_height : ℝ)
  : cable_length_from_base = 3 → leah_distance_from_base = 2.5 → leah_height = 1.5 → 
    (∃ height_of_pole : ℝ, height_of_pole = 9) := 
by
  intros h1 h2 h3
  sorry

end height_of_pole_l1993_199362


namespace min_value_l1993_199382

theorem min_value (x y : ℝ) (h1 : xy > 0) (h2 : x + 4 * y = 3) : 
  ∃ (m : ℝ), m = 3 ∧ ∀ x y, xy > 0 → x + 4 * y = 3 → (1 / x + 1 / y) ≥ 3 := sorry

end min_value_l1993_199382


namespace total_time_to_make_cookies_l1993_199349

def time_to_make_batter := 10
def baking_time := 15
def cooling_time := 15
def white_icing_time := 30
def chocolate_icing_time := 30

theorem total_time_to_make_cookies : 
  time_to_make_batter + baking_time + cooling_time + white_icing_time + chocolate_icing_time = 100 := 
by
  sorry

end total_time_to_make_cookies_l1993_199349


namespace knight_count_l1993_199341

theorem knight_count (K L : ℕ) (h1 : K + L = 15) 
  (h2 : ∀ k, k < K → (∃ l, l < L ∧ l = 6)) 
  (h3 : ∀ l, l < L → (K > 7)) : K = 9 :=
by 
  sorry

end knight_count_l1993_199341


namespace polynomial_factorization_l1993_199325

noncomputable def polyExpression (a b c : ℕ) : ℕ := a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4

theorem polynomial_factorization (a b c : ℕ) :
  ∃ q : ℕ → ℕ → ℕ → ℕ, q a b c = (a + b + c)^3 - 3 * a * b * c ∧
  polyExpression a b c = (a - b) * (b - c) * (c - a) * q a b c := by
  -- The proof goes here
  sorry

end polynomial_factorization_l1993_199325


namespace set_equivalence_l1993_199390

theorem set_equivalence :
  {p : ℝ × ℝ | p.1 + p.2 = 1 ∧ 2 * p.1 - p.2 = 2} = {(1, 0)} :=
by
  sorry

end set_equivalence_l1993_199390


namespace calculate_product_l1993_199372

noncomputable def complex_number_r (r : ℂ) : Prop :=
r^6 = 1 ∧ r ≠ 1

theorem calculate_product (r : ℂ) (h₁ : r^6 = 1) (h₂ : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 2 := 
sorry

end calculate_product_l1993_199372


namespace toms_weekly_income_l1993_199371

variable (num_buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ)

def daily_crabs := num_buckets * crabs_per_bucket
def daily_income := daily_crabs * price_per_crab
def weekly_income := daily_income * days_per_week

theorem toms_weekly_income 
  (h1 : num_buckets = 8)
  (h2 : crabs_per_bucket = 12)
  (h3 : price_per_crab = 5)
  (h4 : days_per_week = 7) :
  weekly_income num_buckets crabs_per_bucket price_per_crab days_per_week = 3360 :=
by
  sorry

end toms_weekly_income_l1993_199371


namespace find_constant_b_l1993_199342

variable (x : ℝ)
variable (b d e : ℝ)

theorem find_constant_b   
  (h1 : (7 * x ^ 2 - 2 * x + 4 / 3) * (d * x ^ 2 + b * x + e) = 28 * x ^ 4 - 10 * x ^ 3 + 18 * x ^ 2 - 8 * x + 5 / 3)
  (h2 : d = 4) : 
  b = -2 / 7 := 
sorry

end find_constant_b_l1993_199342


namespace find_threedigit_number_l1993_199304

-- Define the three-digit number and its reverse
def original_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
def reversed_number (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

-- Define the condition of adding the number and its reverse to get 1777
def number_sum_condition (a b c : ℕ) : Prop :=
  original_number a b c + reversed_number a b c = 1777

-- Prove the existence of digits a, b, and c that satisfy the conditions
theorem find_threedigit_number :
  ∃ a b c : ℕ, a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  original_number a b c = 859 ∧ 
  reversed_number a b c = 958 ∧ 
  number_sum_condition a b c :=
sorry

end find_threedigit_number_l1993_199304


namespace hyperbola_eccentricity_l1993_199328

def hyperbola : Prop :=
  ∀ (x y : ℝ), (x^2 / 9) - (y^2 / 16) = 1

noncomputable def eccentricity : ℝ :=
  let a := 3
  let b := 4
  let c := Real.sqrt (a^2 + b^2)
  c / a

theorem hyperbola_eccentricity : 
  (∀ (x y : ℝ), (x^2 / 9) - (y^2 / 16) = 1) → eccentricity = 5 / 3 :=
by
  intros h
  funext
  exact sorry

end hyperbola_eccentricity_l1993_199328


namespace regular_price_correct_l1993_199384

noncomputable def regular_price_of_one_tire (x : ℝ) : Prop :=
  3 * x + 5 - 10 = 302

theorem regular_price_correct (x : ℝ) : regular_price_of_one_tire x → x = 307 / 3 := by
  intro h
  sorry

end regular_price_correct_l1993_199384


namespace calculateSurfaceArea_l1993_199327

noncomputable def totalSurfaceArea (r : ℝ) : ℝ :=
  let hemisphereCurvedArea := 2 * Real.pi * r^2
  let cylinderLateralArea := 2 * Real.pi * r * r
  hemisphereCurvedArea + cylinderLateralArea

theorem calculateSurfaceArea :
  ∃ r : ℝ, (Real.pi * r^2 = 144 * Real.pi) ∧ totalSurfaceArea r = 576 * Real.pi :=
by
  exists 12
  constructor
  . sorry -- Proof that 144π = π*12^2 can be shown
  . sorry -- Proof that 576π = 288π + 288π can be shown

end calculateSurfaceArea_l1993_199327


namespace Maria_selling_price_l1993_199344

-- Define the constants based on the given conditions
def brush_cost : ℕ := 20
def canvas_cost : ℕ := 3 * brush_cost
def paint_cost_per_liter : ℕ := 8
def paint_needed : ℕ := 5
def earnings : ℕ := 80

-- Calculate the total cost and the selling price
def total_cost : ℕ := brush_cost + canvas_cost + (paint_cost_per_liter * paint_needed)
def selling_price : ℕ := total_cost + earnings

-- Proof statement
theorem Maria_selling_price : selling_price = 200 := by
  sorry

end Maria_selling_price_l1993_199344


namespace triangle_side_lengths_m_range_l1993_199352

theorem triangle_side_lengths_m_range (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (m : ℝ) :
  (2 - Real.sqrt 3) < m ∧ m < (2 + Real.sqrt 3) ↔
  (x + y) + Real.sqrt (x^2 + x * y + y^2) > m * Real.sqrt (x * y) ∧
  (x + y) + m * Real.sqrt (x * y) > Real.sqrt (x^2 + x * y + y^2) ∧
  Real.sqrt (x^2 + x * y + y^2) + m * Real.sqrt (x * y) > (x + y) :=
by sorry

end triangle_side_lengths_m_range_l1993_199352


namespace values_of_x_plus_y_l1993_199396

theorem values_of_x_plus_y (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 6) (h3 : x > y) : x + y = -3 ∨ x + y = -9 :=
sorry

end values_of_x_plus_y_l1993_199396


namespace solve_quad_1_solve_quad_2_l1993_199387

theorem solve_quad_1 :
  ∀ (x : ℝ), x^2 - 5 * x - 6 = 0 ↔ x = 6 ∨ x = -1 := by
  sorry

theorem solve_quad_2 :
  ∀ (x : ℝ), (x + 1) * (x - 1) + x * (x + 2) = 7 + 6 * x ↔ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 := by
  sorry

end solve_quad_1_solve_quad_2_l1993_199387


namespace jackson_miles_l1993_199317

theorem jackson_miles (beka_miles jackson_miles : ℕ) (h1 : beka_miles = 873) (h2 : beka_miles = jackson_miles + 310) : jackson_miles = 563 := by
  sorry

end jackson_miles_l1993_199317


namespace min_value_of_f_l1993_199369

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 + 9) / sqrt (x^2 + 5)

theorem min_value_of_f : ∀ x : ℝ, f x ≥ 4 :=
by
  sorry

end min_value_of_f_l1993_199369


namespace train_length_l1993_199378

/-- Given problem conditions -/
def speed_kmh := 72
def length_platform_m := 270
def time_sec := 26

/-- Convert speed to meters per second -/
def speed_mps := speed_kmh * 1000 / 3600

/-- Calculate the total distance covered -/
def distance_covered := speed_mps * time_sec

theorem train_length :
  (distance_covered - length_platform_m) = 250 :=
by
  sorry

end train_length_l1993_199378


namespace complex_power_identity_l1993_199394

theorem complex_power_identity (i : ℂ) (hi : i^2 = -1) :
  ( (1 + i) / (1 - i) ) ^ 2013 = i :=
by sorry

end complex_power_identity_l1993_199394


namespace greatest_of_5_consec_even_numbers_l1993_199340

-- Definitions based on the conditions
def avg_of_5_consec_even_numbers (N : ℤ) : ℤ := (N - 4 + N - 2 + N + N + 2 + N + 4) / 5

-- Proof statement
theorem greatest_of_5_consec_even_numbers (N : ℤ) (h : avg_of_5_consec_even_numbers N = 35) : N + 4 = 39 :=
by
  sorry -- proof is omitted

end greatest_of_5_consec_even_numbers_l1993_199340


namespace geom_seq_product_l1993_199313

theorem geom_seq_product {a : ℕ → ℝ} (h_geom : ∀ n, a (n + 1) = a n * r)
 (h_a1 : a 1 = 1 / 2) (h_a5 : a 5 = 8) : a 2 * a 3 * a 4 = 8 := 
sorry

end geom_seq_product_l1993_199313


namespace binary_arith_proof_l1993_199333

theorem binary_arith_proof :
  let a := 0b1101110  -- binary representation of 1101110_2
  let b := 0b101010   -- binary representation of 101010_2
  let c := 0b100      -- binary representation of 100_2
  (a * b / c) = 0b11001000010 :=  -- binary representation of the final result
by
  sorry

end binary_arith_proof_l1993_199333


namespace find_g_value_l1993_199374

noncomputable def g (x : ℝ) (a b c : ℝ) : ℝ := a * x^6 + b * x^4 + c * x^2 + 7

theorem find_g_value (a b c : ℝ) (h1 : g (-4) a b c = 13) : g 4 a b c = 13 := by
  sorry

end find_g_value_l1993_199374


namespace students_like_neither_l1993_199319

theorem students_like_neither (N_Total N_Chinese N_Math N_Both N_Neither : ℕ)
  (h_total: N_Total = 62)
  (h_chinese: N_Chinese = 37)
  (h_math: N_Math = 49)
  (h_both: N_Both = 30)
  (h_neither: N_Neither = N_Total - (N_Chinese - N_Both) - (N_Math - N_Both) - N_Both) : 
  N_Neither = 6 :=
by 
  rw [h_total, h_chinese, h_math, h_both] at h_neither
  exact h_neither.trans (by norm_num)


end students_like_neither_l1993_199319


namespace find_c_quadratic_solution_l1993_199386

theorem find_c_quadratic_solution (c : ℝ) :
  (Polynomial.eval (-5) (Polynomial.C (-45) + Polynomial.X * Polynomial.C c + Polynomial.X^2) = 0) →
  c = -4 :=
by 
  intros h
  sorry

end find_c_quadratic_solution_l1993_199386


namespace subtracted_result_correct_l1993_199392

theorem subtracted_result_correct (n : ℕ) (h1 : 96 / n = 6) : 34 - n = 18 :=
by
  sorry

end subtracted_result_correct_l1993_199392


namespace ratio_c_a_l1993_199376

theorem ratio_c_a (a b c : ℚ) (h1 : a * b = 3) (h2 : b * c = 8 / 5) : c / a = 8 / 15 := 
by 
  sorry

end ratio_c_a_l1993_199376


namespace num_solution_pairs_l1993_199310

theorem num_solution_pairs : 
  ∃! (n : ℕ), 
    n = 2 ∧ 
    ∃ x y : ℕ, 
      x > 0 ∧ y >0 ∧ 
      4^x = y^2 + 15 := 
by 
  sorry

end num_solution_pairs_l1993_199310


namespace find_other_endpoint_l1993_199393

theorem find_other_endpoint :
  ∀ (A B M : ℝ × ℝ),
  M = (2, 3) →
  A = (7, -4) →
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  B = (-3, 10) :=
by
  intros A B M hM1 hA hM2
  sorry

end find_other_endpoint_l1993_199393


namespace inequality_problem_l1993_199361

open Real

theorem inequality_problem 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h1 : x + y^2016 ≥ 1) : 
  x^2016 + y > 1 - 1/100 :=
by
  sorry

end inequality_problem_l1993_199361


namespace mark_spends_47_l1993_199306

def apple_price : ℕ := 2
def apple_quantity : ℕ := 4
def bread_price : ℕ := 3
def bread_quantity : ℕ := 5
def cheese_price : ℕ := 6
def cheese_quantity : ℕ := 3
def cereal_price : ℕ := 5
def cereal_quantity : ℕ := 4
def coupon : ℕ := 10

def calculate_total_cost (apple_price apple_quantity bread_price bread_quantity cheese_price cheese_quantity cereal_price cereal_quantity coupon : ℕ) : ℕ :=
  let apples_cost := apple_price * (apple_quantity / 2)  -- Apply buy-one-get-one-free
  let bread_cost := bread_price * bread_quantity
  let cheese_cost := cheese_price * cheese_quantity
  let cereal_cost := cereal_price * cereal_quantity
  let subtotal := apples_cost + bread_cost + cheese_cost + cereal_cost
  let total_cost := if subtotal > 50 then subtotal - coupon else subtotal
  total_cost

theorem mark_spends_47 : calculate_total_cost apple_price apple_quantity bread_price bread_quantity cheese_price cheese_quantity cereal_price cereal_quantity coupon = 47 :=
  sorry

end mark_spends_47_l1993_199306
