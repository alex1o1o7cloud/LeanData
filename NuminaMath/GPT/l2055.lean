import Mathlib

namespace find_g_of_3_l2055_205536

noncomputable def g (x : ℝ) : ℝ := sorry  -- Placeholder for the function g

theorem find_g_of_3 (h : ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = 2 * x) :
  g 3 = 26 / 7 :=
by sorry

end find_g_of_3_l2055_205536


namespace arithmetic_sequence_a2_value_l2055_205524

theorem arithmetic_sequence_a2_value :
  ∃ (a : ℕ) (d : ℕ), (a = 3) ∧ (a + d + (a + 2 * d) = 12) ∧ (a + d = 5) :=
by
  sorry

end arithmetic_sequence_a2_value_l2055_205524


namespace julia_cookies_l2055_205592

theorem julia_cookies (N : ℕ) 
  (h1 : N % 6 = 5) 
  (h2 : N % 8 = 7) 
  (h3 : N < 100) : 
  N = 17 ∨ N = 41 ∨ N = 65 ∨ N = 89 → 17 + 41 + 65 + 89 = 212 :=
sorry

end julia_cookies_l2055_205592


namespace find_roots_of_parabola_l2055_205589

-- Define the conditions given in the problem
variables (a b c : ℝ)
variable (a_nonzero : a ≠ 0)
variable (passes_through_1_0 : a * 1^2 + b * 1 + c = 0)
variable (axis_of_symmetry : -b / (2 * a) = -2)

-- Lean theorem statement
theorem find_roots_of_parabola (a b c : ℝ) (a_nonzero : a ≠ 0)
(passes_through_1_0 : a * 1^2 + b * 1 + c = 0) (axis_of_symmetry : -b / (2 * a) = -2) :
  (a * (-5)^2 + b * (-5) + c = 0) ∧ (a * 1^2 + b * 1 + c = 0) :=
by
  -- Placeholder for the proof
  sorry

end find_roots_of_parabola_l2055_205589


namespace f_minimum_positive_period_and_max_value_l2055_205576

noncomputable def f (x : ℝ) : ℝ := (Real.sin x * Real.cos x) + (1 + (Real.tan x)^2) * (Real.cos x)^2

theorem f_minimum_positive_period_and_max_value :
  (∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T ≥ π) ∧ (∃ M, ∀ x : ℝ, f x ≤ M ∧ M = 3 / 2) := by
  sorry

end f_minimum_positive_period_and_max_value_l2055_205576


namespace length_of_EC_l2055_205555

theorem length_of_EC
  (AB CD AC : ℝ)
  (h1 : AB = 3 * CD)
  (h2 : AC = 15)
  (EC : ℝ)
  (h3 : AC = 4 * EC)
  : EC = 15 / 4 := 
sorry

end length_of_EC_l2055_205555


namespace max_students_distribution_l2055_205531

-- Define the four quantities
def pens : ℕ := 4261
def pencils : ℕ := 2677
def erasers : ℕ := 1759
def notebooks : ℕ := 1423

-- Prove that the greatest common divisor (GCD) of these four quantities is 1
theorem max_students_distribution : Nat.gcd (Nat.gcd (Nat.gcd pens pencils) erasers) notebooks = 1 :=
by
  sorry

end max_students_distribution_l2055_205531


namespace ways_to_place_7_balls_into_3_boxes_l2055_205564

theorem ways_to_place_7_balls_into_3_boxes :
  (Nat.choose (7 + 3 - 1) (3 - 1)) = 36 :=
by
  sorry

end ways_to_place_7_balls_into_3_boxes_l2055_205564


namespace evaluate_expression_l2055_205525

theorem evaluate_expression : 
  (3^2 - 3 * 2) - (4^2 - 4 * 2) + (5^2 - 5 * 2) - (6^2 - 6 * 2) = -14 :=
by
  sorry

end evaluate_expression_l2055_205525


namespace find_x_y_l2055_205582

theorem find_x_y (x y : ℝ) 
  (h1 : 3 * x = 0.75 * y)
  (h2 : x + y = 30) : x = 6 ∧ y = 24 := 
by
  sorry  -- Proof is omitted

end find_x_y_l2055_205582


namespace total_vitamins_in_box_correct_vitamins_per_half_bag_correct_l2055_205522

-- Define the conditions
def number_of_bags : ℕ := 9
def vitamins_per_bag : ℚ := 0.2

-- Define the total vitamins in the box
def total_vitamins_in_box : ℚ := number_of_bags * vitamins_per_bag

-- Define the vitamins intake by drinking half a bag
def vitamins_per_half_bag : ℚ := vitamins_per_bag / 2

-- Prove that the total grams of vitamins in the box is 1.8 grams
theorem total_vitamins_in_box_correct : total_vitamins_in_box = 1.8 := by
  sorry

-- Prove that the vitamins intake by drinking half a bag is 0.1 grams
theorem vitamins_per_half_bag_correct : vitamins_per_half_bag = 0.1 := by
  sorry

end total_vitamins_in_box_correct_vitamins_per_half_bag_correct_l2055_205522


namespace equal_play_time_for_students_l2055_205502

theorem equal_play_time_for_students 
  (total_students : ℕ) 
  (start_time end_time : ℕ) 
  (tables : ℕ) 
  (playing_students refereeing_students : ℕ) 
  (time_played : ℕ) :
  total_students = 6 →
  start_time = 8 * 60 →
  end_time = 11 * 60 + 30 →
  tables = 2 →
  playing_students = 4 →
  refereeing_students = 2 →
  time_played = (end_time - start_time) * tables / (total_students / refereeing_students) →
  time_played = 140 :=
by
  sorry

end equal_play_time_for_students_l2055_205502


namespace aunt_may_milk_left_l2055_205539

def morningMilkProduction (numCows numGoats numSheep : ℕ) (cowMilk goatMilk sheepMilk : ℝ) : ℝ :=
  numCows * cowMilk + numGoats * goatMilk + numSheep * sheepMilk

def eveningMilkProduction (numCows numGoats numSheep : ℕ) (cowMilk goatMilk sheepMilk : ℝ) : ℝ :=
  numCows * cowMilk + numGoats * goatMilk + numSheep * sheepMilk

def spoiledMilk (milkProduction : ℝ) (spoilageRate : ℝ) : ℝ :=
  milkProduction * spoilageRate

def freshMilk (totalMilk spoiledMilk : ℝ) : ℝ :=
  totalMilk - spoiledMilk

def soldMilk (freshMilk : ℝ) (saleRate : ℝ) : ℝ :=
  freshMilk * saleRate

def milkLeft (freshMilk soldMilk : ℝ) : ℝ :=
  freshMilk - soldMilk

noncomputable def totalMilkLeft (previousLeftover : ℝ) (morningLeft eveningLeft : ℝ) : ℝ :=
  previousLeftover + morningLeft + eveningLeft

theorem aunt_may_milk_left :
  let numCows := 5
  let numGoats := 4
  let numSheep := 10
  let cowMilkMorning := 13
  let goatMilkMorning := 0.5
  let sheepMilkMorning := 0.25
  let cowMilkEvening := 14
  let goatMilkEvening := 0.6
  let sheepMilkEvening := 0.2
  let morningSpoilageRate := 0.10
  let eveningSpoilageRate := 0.05
  let iceCreamSaleRate := 0.70
  let cheeseShopSaleRate := 0.80
  let previousLeftover := 15
  let morningMilk := morningMilkProduction numCows numGoats numSheep cowMilkMorning goatMilkMorning sheepMilkMorning
  let eveningMilk := eveningMilkProduction numCows numGoats numSheep cowMilkEvening goatMilkEvening sheepMilkEvening
  let morningSpoiled := spoiledMilk morningMilk morningSpoilageRate
  let eveningSpoiled := spoiledMilk eveningMilk eveningSpoilageRate
  let freshMorningMilk := freshMilk morningMilk morningSpoiled
  let freshEveningMilk := freshMilk eveningMilk eveningSpoiled
  let morningSold := soldMilk freshMorningMilk iceCreamSaleRate
  let eveningSold := soldMilk freshEveningMilk cheeseShopSaleRate
  let morningLeft := milkLeft freshMorningMilk morningSold
  let eveningLeft := milkLeft freshEveningMilk eveningSold
  totalMilkLeft previousLeftover morningLeft eveningLeft = 47.901 :=
by
  sorry

end aunt_may_milk_left_l2055_205539


namespace sophia_daily_saving_l2055_205587

theorem sophia_daily_saving (total_days : ℕ) (total_saving : ℝ) (h1 : total_days = 20) (h2 : total_saving = 0.20) : 
  (total_saving / total_days) = 0.01 :=
by
  sorry

end sophia_daily_saving_l2055_205587


namespace percentage_A_of_B_l2055_205581

variable {A B C D : ℝ}

theorem percentage_A_of_B (
  h1: A = 0.125 * C)
  (h2: B = 0.375 * D)
  (h3: D = 1.225 * C)
  (h4: C = 0.805 * B) :
  A = 0.100625 * B := by
  -- Sufficient proof steps would go here
  sorry

end percentage_A_of_B_l2055_205581


namespace square_is_six_l2055_205584

def represents_digit (square triangle circle : ℕ) : Prop :=
  square < 10 ∧ triangle < 10 ∧ circle < 10 ∧
  square ≠ triangle ∧ square ≠ circle ∧ triangle ≠ circle

theorem square_is_six :
  ∃ (square triangle circle : ℕ), represents_digit square triangle circle ∧ triangle = 1 ∧ circle = 9 ∧ (square + triangle + 100 * 1 + 10 * 9) = 117 ∧ square = 6 :=
by {
  sorry
}

end square_is_six_l2055_205584


namespace ball_radius_l2055_205569

theorem ball_radius 
  (r_cylinder : ℝ) (h_rise : ℝ) (v_approx : ℝ)
  (r_cylinder_value : r_cylinder = 12)
  (h_rise_value : h_rise = 6.75)
  (v_approx_value : v_approx = 3053.628) :
  ∃ (r_ball : ℝ), (4 / 3) * Real.pi * r_ball^3 = v_approx ∧ r_ball = 9 := 
by 
  use 9
  sorry

end ball_radius_l2055_205569


namespace moles_of_KOH_used_l2055_205538

variable {n_KOH : ℝ}

theorem moles_of_KOH_used :
  ∃ n_KOH, (NH4I + KOH = KI_produced) → (KI_produced = 1) → n_KOH = 1 :=
by
  sorry

end moles_of_KOH_used_l2055_205538


namespace sequence_general_term_l2055_205540

theorem sequence_general_term
  (a : ℕ → ℝ)
  (h₁ : a 1 = 1 / 2)
  (h₂ : ∀ n, a (n + 1) = 3 * a n + 7) :
  ∀ n, a n = 4 * 3^(n - 1) - 7 / 2 :=
by
  sorry

end sequence_general_term_l2055_205540


namespace triangle_problem_l2055_205535

/-- In triangle ABC, the sides opposite to angles A, B, C are a, b, c respectively.
Given that b = sqrt 2, c = 3, B + C = 3A, prove:
1. The length of side a equals sqrt 5.
2. sin (B + 3π/4) equals sqrt(10) / 10.
-/
theorem triangle_problem 
  (a b c A B C : ℝ)
  (hb : b = Real.sqrt 2)
  (hc : c = 3)
  (hBC : B + C = 3 * A)
  (hA : A = π / 4)
  : (a = Real.sqrt 5)
  ∧ (Real.sin (B + 3 * π / 4) = Real.sqrt 10 / 10) :=
sorry

end triangle_problem_l2055_205535


namespace machine_value_after_two_years_l2055_205597

noncomputable def machine_market_value (initial_value : ℝ) (years : ℕ) (decrease_rate : ℝ) : ℝ :=
  initial_value * (1 - decrease_rate) ^ years

theorem machine_value_after_two_years :
  machine_market_value 8000 2 0.2 = 5120 := by
  sorry

end machine_value_after_two_years_l2055_205597


namespace max_expression_sum_l2055_205595

open Real

theorem max_expression_sum :
  ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ 
  (2 * x^2 - 3 * x * y + 4 * y^2 = 15 ∧ 
  (3 * x^2 + 2 * x * y + y^2 = 50 * sqrt 3 + 65)) :=
sorry

#eval 65 + 50 + 3 + 1 -- this should output 119

end max_expression_sum_l2055_205595


namespace gary_money_left_l2055_205546

variable (initialAmount : Nat)
variable (amountSpent : Nat)

theorem gary_money_left (h1 : initialAmount = 73) (h2 : amountSpent = 55) : initialAmount - amountSpent = 18 :=
by
  sorry

end gary_money_left_l2055_205546


namespace actual_distance_traveled_l2055_205570

-- Given conditions
variables (D : ℝ)
variables (H : D / 5 = (D + 20) / 15)

-- The proof problem statement
theorem actual_distance_traveled : D = 10 :=
by
  sorry

end actual_distance_traveled_l2055_205570


namespace max_value_and_period_of_g_value_of_expression_if_fx_eq_2f_l2055_205561

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f' (x : ℝ) : ℝ := Real.cos x - Real.sin x
noncomputable def g (x : ℝ) : ℝ := f x * f' x - f x ^ 2

theorem max_value_and_period_of_g :
  ∃ (M : ℝ) (T : ℝ), (∀ x, g x ≤ M) ∧ (∀ x, g (x + T) = g x) ∧ M = 2 ∧ T = Real.pi :=
sorry

theorem value_of_expression_if_fx_eq_2f'x (x : ℝ) :
  f x = 2 * f' x → (1 + Real.sin x ^ 2) / (Real.cos x ^ 2 - Real.sin x * Real.cos x) = 11 / 6 :=
sorry

end max_value_and_period_of_g_value_of_expression_if_fx_eq_2f_l2055_205561


namespace percentage_of_cars_in_accident_l2055_205599

-- Define probabilities of each segment of the rally
def prob_fall_bridge := 1 / 5
def prob_off_turn := 3 / 10
def prob_crash_tunnel := 1 / 10
def prob_stuck_sand := 2 / 5

-- Define complement probabilities (successful completion)
def prob_success_bridge := 1 - prob_fall_bridge
def prob_success_turn := 1 - prob_off_turn
def prob_success_tunnel := 1 - prob_crash_tunnel
def prob_success_sand := 1 - prob_stuck_sand

-- Define overall success probability
def prob_success_total := prob_success_bridge * prob_success_turn * prob_success_tunnel * prob_success_sand

-- Define percentage function
def percentage (p: ℚ) : ℚ := p * 100

-- Prove the percentage of cars involved in accidents
theorem percentage_of_cars_in_accident : percentage (1 - prob_success_total) = 70 := by sorry

end percentage_of_cars_in_accident_l2055_205599


namespace evaluate_expression_l2055_205572

theorem evaluate_expression : (2^(2 + 1) - 4 * (2 - 1)^2)^2 = 16 :=
by
  sorry

end evaluate_expression_l2055_205572


namespace average_age_of_club_l2055_205560

theorem average_age_of_club (S_f S_m S_c : ℕ) (females males children : ℕ) (avg_females avg_males avg_children : ℕ) :
  females = 12 →
  males = 20 →
  children = 8 →
  avg_females = 28 →
  avg_males = 40 →
  avg_children = 10 →
  S_f = avg_females * females →
  S_m = avg_males * males →
  S_c = avg_children * children →
  (S_f + S_m + S_c) / (females + males + children) = 30 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end average_age_of_club_l2055_205560


namespace sufficient_not_necessary_condition_l2055_205541

theorem sufficient_not_necessary_condition (x : ℝ) : (x < -1 → x^2 - 1 > 0) ∧ (x^2 - 1 > 0 → x < -1 ∨ x > 1) :=
by
  sorry

end sufficient_not_necessary_condition_l2055_205541


namespace rope_folded_three_times_parts_l2055_205510

theorem rope_folded_three_times_parts (total_length : ℕ) :
  ∀ parts : ℕ, parts = (total_length / 8) →
  ∀ n : ℕ, n = 3 →
  (∀ length_each_part : ℚ, length_each_part = 1 / (2 ^ n) →
  length_each_part = 1 / 8) :=
by
  sorry

end rope_folded_three_times_parts_l2055_205510


namespace remainder_sum_div_6_l2055_205506

theorem remainder_sum_div_6 (n : ℤ) : ((5 - n) + (n + 4)) % 6 = 3 :=
by
  -- Placeholder for the actual proof
  sorry

end remainder_sum_div_6_l2055_205506


namespace even_num_students_count_l2055_205562

-- Define the number of students in each school
def num_students_A : Nat := 786
def num_students_B : Nat := 777
def num_students_C : Nat := 762
def num_students_D : Nat := 819
def num_students_E : Nat := 493

-- Define a predicate to check if a number is even
def is_even (n : Nat) : Prop := n % 2 = 0

-- The theorem to state the problem
theorem even_num_students_count :
  (is_even num_students_A ∧ is_even num_students_C) ∧ ¬(is_even num_students_B ∧ is_even num_students_D ∧ is_even num_students_E) →
  2 = 2 :=
by
  sorry

end even_num_students_count_l2055_205562


namespace minutes_per_mile_l2055_205559

-- Define the total distance Peter needs to walk
def total_distance : ℝ := 2.5

-- Define the distance Peter has already walked
def walked_distance : ℝ := 1.0

-- Define the remaining time Peter needs to walk to reach the grocery store
def remaining_time : ℝ := 30.0

-- Define the remaining distance Peter needs to walk
def remaining_distance : ℝ := total_distance - walked_distance

-- The desired statement to prove: it takes Peter 20 minutes to walk one mile
theorem minutes_per_mile : remaining_distance / remaining_time = 1.0 / 20.0 := by
  sorry

end minutes_per_mile_l2055_205559


namespace smallest_k_divides_l2055_205507

-- Declare the polynomial p(z)
def p (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

-- Proposition: The smallest positive integer k such that p(z) divides z^k - 1 is 120
theorem smallest_k_divides (k : ℕ) (h : ∀ z : ℂ, p z ∣ z^k - 1) : k = 120 :=
sorry

end smallest_k_divides_l2055_205507


namespace largest_element_in_A_inter_B_l2055_205573

def A : Set ℕ := { n | 1 ≤ n ∧ n ≤ 2023 }
def B : Set ℕ := { n | ∃ k : ℤ, n = 3 * k + 2 ∧ n > 0 }

theorem largest_element_in_A_inter_B : ∃ x ∈ (A ∩ B), ∀ y ∈ (A ∩ B), y ≤ x ∧ x = 2021 := by
  sorry

end largest_element_in_A_inter_B_l2055_205573


namespace radius_circle_D_eq_five_l2055_205517

-- Definitions for circles with given radii and tangency conditions
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

noncomputable def circle_C : Circle := ⟨(0, 0), 5⟩
noncomputable def circle_D (rD : ℝ) : Circle := ⟨(4 * rD, 0), 4 * rD⟩
noncomputable def circle_E (rE : ℝ) : Circle := ⟨(5 - rE, rE * 5), rE⟩

-- Prove that the radius of circle D is 5
theorem radius_circle_D_eq_five (rE : ℝ) (rD : ℝ) : circle_D rE = circle_C → rD = 5 := by
  sorry

end radius_circle_D_eq_five_l2055_205517


namespace linear_function_increasing_l2055_205509

variable (x1 x2 : ℝ)
variable (y1 y2 : ℝ)
variable (hx : x1 < x2)
variable (P1_eq : y1 = 2 * x1 + 1)
variable (P2_eq : y2 = 2 * x2 + 1)

theorem linear_function_increasing (hx : x1 < x2) (P1_eq : y1 = 2 * x1 + 1) (P2_eq : y2 = 2 * x2 + 1) 
    : y1 < y2 := sorry

end linear_function_increasing_l2055_205509


namespace find_t_l2055_205500

-- Defining variables and assumptions
variables (V V0 g S t : Real)
variable (h1 : V = g * t + V0)
variable (h2 : S = (1 / 2) * g * t^2 + V0 * t)

-- The goal: to prove t equals 2S / (V + V0)
theorem find_t (V V0 g S t : Real) (h1 : V = g * t + V0) (h2 : S = (1 / 2) * g * t^2 + V0 * t):
  t = 2 * S / (V + V0) := by
  sorry

end find_t_l2055_205500


namespace expected_coins_basilio_per_day_l2055_205586

/-- The expected number of gold coins received by Basilio per day is 5.25 -/
def expected_coins_received_by_basilio (n : ℕ) (p : ℚ) : ℚ :=
  if n = 20 ∧ p = (1 / 2 : ℚ) then 5.25 else 0

theorem expected_coins_basilio_per_day :
  expected_coins_received_by_basilio 20 (1 / 2) = 5.25 :=
by {
  -- proof goes here
  sorry
}

end expected_coins_basilio_per_day_l2055_205586


namespace negation_of_p_l2055_205528
open Classical

variable (n : ℕ)

def p : Prop := ∀ n : ℕ, n^2 < 2^n

theorem negation_of_p : ¬ p ↔ ∃ n₀ : ℕ, n₀^2 ≥ 2^n₀ := 
by
  sorry

end negation_of_p_l2055_205528


namespace problem_solution_l2055_205521

theorem problem_solution :
  { x : ℝ | x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2 ∧ (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) < 1 / 5) } 
  = { x : ℝ | x < 0 } ∪ { x : ℝ | 1 < x ∧ x < 2 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

end problem_solution_l2055_205521


namespace greatest_integer_gcd_3_l2055_205504

theorem greatest_integer_gcd_3 : ∃ n, n < 100 ∧ gcd n 18 = 3 ∧ ∀ m, m < 100 ∧ gcd m 18 = 3 → m ≤ n := by
  sorry

end greatest_integer_gcd_3_l2055_205504


namespace additional_savings_l2055_205598

-- Defining the conditions
def initial_price : ℝ := 50
def discount_one : ℝ := 6
def discount_percentage : ℝ := 0.15

-- Defining the final prices according to the two methods
def first_method : ℝ := (1 - discount_percentage) * (initial_price - discount_one)
def second_method : ℝ := (1 - discount_percentage) * initial_price - discount_one

-- Defining the savings for the two methods
def savings_first_method : ℝ := initial_price - first_method
def savings_second_method : ℝ := initial_price - second_method

-- Proving that the second method results in an additional 0.90 savings
theorem additional_savings : (savings_second_method - savings_first_method) = 0.90 :=
by
  sorry

end additional_savings_l2055_205598


namespace max_gcd_13n_plus_4_8n_plus_3_l2055_205543

theorem max_gcd_13n_plus_4_8n_plus_3 : ∃ n : ℕ, n > 0 ∧ Int.gcd (13 * n + 4) (8 * n + 3) = 11 := 
sorry

end max_gcd_13n_plus_4_8n_plus_3_l2055_205543


namespace find_Gary_gold_l2055_205577

variable (G : ℕ) -- G represents the number of grams of gold Gary has.
variable (cost_Gary_gold_per_gram : ℕ) -- The cost per gram of Gary's gold.
variable (grams_Anna_gold : ℕ) -- The number of grams of gold Anna has.
variable (cost_Anna_gold_per_gram : ℕ) -- The cost per gram of Anna's gold.
variable (combined_cost : ℕ) -- The combined cost of both Gary's and Anna's gold.

theorem find_Gary_gold (h1 : cost_Gary_gold_per_gram = 15)
                       (h2 : grams_Anna_gold = 50)
                       (h3 : cost_Anna_gold_per_gram = 20)
                       (h4 : combined_cost = 1450)
                       (h5 : combined_cost = cost_Gary_gold_per_gram * G + grams_Anna_gold * cost_Anna_gold_per_gram) :
  G = 30 :=
by 
  sorry

end find_Gary_gold_l2055_205577


namespace equivalent_systems_solution_and_value_l2055_205514

-- Definitions for the conditions
def system1 (x y a b : ℝ) : Prop := 
  (2 * (x + 1) - y = 7) ∧ (x + b * y = a)

def system2 (x y a b : ℝ) : Prop := 
  (a * x + y = b) ∧ (3 * x + 2 * (y - 1) = 9)

-- The proof problem as a Lean 4 statement
theorem equivalent_systems_solution_and_value (a b : ℝ) :
  (∃ x y : ℝ, system1 x y a b ∧ system2 x y a b) →
  ((∃ x y : ℝ, x = 3 ∧ y = 1) ∧ (3 * a - b) ^ 2023 = -1) :=
  by sorry

end equivalent_systems_solution_and_value_l2055_205514


namespace find_R_when_S_eq_5_l2055_205594

theorem find_R_when_S_eq_5
  (g : ℚ)
  (h1 : ∀ S, R = g * S^2 - 6)
  (h2 : R = 15 ∧ S = 3) :
  R = 157 / 3 := by
    sorry

end find_R_when_S_eq_5_l2055_205594


namespace remaining_garden_space_l2055_205503

theorem remaining_garden_space : 
  let Area_rectangle := 20 * 18
  let Area_square_cutout := 4 * 4
  let Area_triangle := (1 / 2) * 3 * 2
  Area_rectangle - Area_square_cutout + Area_triangle = 347 :=
by
  let Area_rectangle := 20 * 18
  let Area_square_cutout := 4 * 4
  let Area_triangle := (1 / 2) * 3 * 2
  show Area_rectangle - Area_square_cutout + Area_triangle = 347
  sorry

end remaining_garden_space_l2055_205503


namespace single_train_car_passenger_count_l2055_205563

theorem single_train_car_passenger_count (P : ℕ) 
  (h1 : ∀ (plane_capacity train_capacity : ℕ), plane_capacity = 366 →
    train_capacity = 16 * P →
      (train_capacity = (2 * plane_capacity) + 228)) : 
  P = 60 :=
by
  sorry

end single_train_car_passenger_count_l2055_205563


namespace inequality_A_only_inequality_B_not_always_l2055_205549

theorem inequality_A_only (a b c : ℝ) (h1 : 2 * b > c) (h2 : c > a) (h3 : c > b) :
  a < c / 3 := 
sorry

theorem inequality_B_not_always (a b c : ℝ) (h1 : 2 * b > c) (h2 : c > a) (h3 : c > b) :
  ¬ (b < c / 3) := 
sorry

end inequality_A_only_inequality_B_not_always_l2055_205549


namespace ratio_of_x_to_y_l2055_205556

variable (x y : ℝ)

theorem ratio_of_x_to_y (h : 0.10 * x = 0.20 * y) : x / y = 2 :=
by sorry

end ratio_of_x_to_y_l2055_205556


namespace stock_percentage_calculation_l2055_205518

noncomputable def stock_percentage (investment_amount stock_price annual_income : ℝ) : ℝ :=
  (annual_income / (investment_amount / stock_price) / stock_price) * 100

theorem stock_percentage_calculation :
  stock_percentage 6800 136 1000 = 14.71 :=
by
  sorry

end stock_percentage_calculation_l2055_205518


namespace fish_too_small_l2055_205511

theorem fish_too_small
    (ben_fish : ℕ) (judy_fish : ℕ) (billy_fish : ℕ) (jim_fish : ℕ) (susie_fish : ℕ)
    (total_filets : ℕ) (filets_per_fish : ℕ) :
    ben_fish = 4 →
    judy_fish = 1 →
    billy_fish = 3 →
    jim_fish = 2 →
    susie_fish = 5 →
    total_filets = 24 →
    filets_per_fish = 2 →
    (ben_fish + judy_fish + billy_fish + jim_fish + susie_fish) - (total_filets / filets_per_fish) = 3 := 
by 
  intros
  sorry

end fish_too_small_l2055_205511


namespace eq_implies_sq_eq_l2055_205591

theorem eq_implies_sq_eq (a b : ℝ) (h : a = b) : a^2 = b^2 :=
sorry

end eq_implies_sq_eq_l2055_205591


namespace A_inter_B_eq_C_l2055_205588

noncomputable def A : Set ℝ := { x | ∃ α β : ℤ, α ≥ 0 ∧ β ≥ 0 ∧ x = 2^α * 3^β }
def B : Set ℝ := { x | 1 ≤ x ∧ x ≤ 5 }
def C : Set ℝ := {1, 2, 3, 4}

theorem A_inter_B_eq_C : A ∩ B = C :=
by
  sorry

end A_inter_B_eq_C_l2055_205588


namespace min_inequality_l2055_205523

theorem min_inequality (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 2) :
  ∃ L, L = 9 / 4 ∧ (1 / (x + y) + 1 / (x + z) + 1 / (y + z) ≥ L) :=
sorry

end min_inequality_l2055_205523


namespace average_age_decrease_l2055_205579

-- Define the conditions as given in the problem
def original_strength : ℕ := 12
def new_students : ℕ := 12

def original_avg_age : ℕ := 40
def new_students_avg_age : ℕ := 32

def decrease_in_avg_age (O N : ℕ) (OA NA : ℕ) : ℕ :=
  let total_original_age := O * OA
  let total_new_students_age := N * NA
  let total_students := O + N
  let new_avg_age := (total_original_age + total_new_students_age) / total_students
  OA - new_avg_age

theorem average_age_decrease :
  decrease_in_avg_age original_strength new_students original_avg_age new_students_avg_age = 4 :=
sorry

end average_age_decrease_l2055_205579


namespace k_equals_10_l2055_205590

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a d : α) : ℕ → α
  | 0     => a
  | (n+1) => a + (n+1) * d

noncomputable def sum_of_first_n_terms (a d : α) (n : ℕ) : α :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem k_equals_10
  (a d : α)
  (h1 : sum_of_first_n_terms a d 9 = sum_of_first_n_terms a d 4)
  (h2 : arithmetic_sequence a d 4 + arithmetic_sequence a d 10 = 0) :
  k = 10 :=
sorry

end k_equals_10_l2055_205590


namespace percent_of_a_l2055_205544

theorem percent_of_a (a b : ℝ) (h : a = 1.2 * b) : 4 * b = (10 / 3) * a :=
sorry

end percent_of_a_l2055_205544


namespace Zelda_probability_success_l2055_205530

variable (P : ℝ → ℝ)
variable (X Y Z : ℝ)

theorem Zelda_probability_success :
  P X = 1/3 ∧ P Y = 1/2 ∧ (P X) * (P Y) * (1 - P Z) = 0.0625 → P Z = 0.625 :=
by
  sorry

end Zelda_probability_success_l2055_205530


namespace exists_positive_integers_for_equation_l2055_205583

theorem exists_positive_integers_for_equation :
  ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a^4 = b^3 + c^2 :=
by
  sorry

end exists_positive_integers_for_equation_l2055_205583


namespace inequality_x_y_l2055_205547

theorem inequality_x_y 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) : 
  (x / (x + 5 * y)) + (y / (y + 5 * x)) ≤ 1 := 
by 
  sorry

end inequality_x_y_l2055_205547


namespace patio_total_tiles_l2055_205551

theorem patio_total_tiles (s : ℕ) (red_tiles : ℕ) (h1 : s % 2 = 1) (h2 : red_tiles = 2 * s - 1) (h3 : red_tiles = 61) :
  s * s = 961 :=
by
  sorry

end patio_total_tiles_l2055_205551


namespace cookies_per_child_is_22_l2055_205526

def total_cookies (num_packages : ℕ) (cookies_per_package : ℕ) : ℕ :=
  num_packages * cookies_per_package

def total_children (num_friends : ℕ) : ℕ :=
  num_friends + 1

def cookies_per_child (total_cookies : ℕ) (total_children : ℕ) : ℕ :=
  total_cookies / total_children

theorem cookies_per_child_is_22 :
  total_cookies 5 36 / total_children 7 = 22 := 
by
  sorry

end cookies_per_child_is_22_l2055_205526


namespace g_2187_value_l2055_205542

-- Define the function properties and the goal
theorem g_2187_value (g : ℕ → ℝ) (h : ∀ x y m : ℕ, x + y = 3^m → g x + g y = m^3) :
  g 2187 = 343 :=
sorry

end g_2187_value_l2055_205542


namespace fill_blanks_l2055_205557

/-
Given the following conditions:
1. 20 * (x1 - 8) = 20
2. x2 / 2 + 17 = 20
3. 3 * x3 - 4 = 20
4. (x4 + 8) / 12 = y4
5. 4 * x5 = 20
6. 20 * (x6 - y6) = 100

Prove that:
1. x1 = 9
2. x2 = 6
3. x3 = 8
4. x4 = 4 and y4 = 1
5. x5 = 5
6. x6 = 7 and y6 = 2
-/
theorem fill_blanks (x1 x2 x3 x4 y4 x5 x6 y6 : ℕ) :
  20 * (x1 - 8) = 20 →
  x2 / 2 + 17 = 20 →
  3 * x3 - 4 = 20 →
  (x4 + 8) / 12 = y4 →
  4 * x5 = 20 →
  20 * (x6 - y6) = 100 →
  x1 = 9 ∧
  x2 = 6 ∧
  x3 = 8 ∧
  x4 = 4 ∧
  y4 = 1 ∧
  x5 = 5 ∧
  x6 = 7 ∧
  y6 = 2 :=
by
  sorry

end fill_blanks_l2055_205557


namespace possible_value_of_b_l2055_205574

theorem possible_value_of_b (a b : ℕ) (H1 : b ∣ (5 * a - 1)) (H2 : b ∣ (a - 10)) (H3 : ¬ b ∣ (3 * a + 5)) : 
  b = 49 :=
sorry

end possible_value_of_b_l2055_205574


namespace sum_of_nonnegative_reals_l2055_205545

theorem sum_of_nonnegative_reals (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 52) (h2 : a * b + b * c + c * a = 24) (h3 : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) :
  a + b + c = 10 :=
sorry

end sum_of_nonnegative_reals_l2055_205545


namespace arithmetic_sequence_general_term_l2055_205513

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (h1 : a 2 = 4) 
  (h2 : a 4 + a 7 = 15) : 
  ∃ d : ℝ, ∀ n : ℕ, a n = n + 2 := 
by
  sorry

end arithmetic_sequence_general_term_l2055_205513


namespace sequence_a_is_perfect_square_l2055_205512

theorem sequence_a_is_perfect_square :
  ∃ (a b : ℕ → ℤ),
    a 0 = 1 ∧ 
    b 0 = 0 ∧ 
    (∀ n, a (n + 1) = 7 * a n + 6 * b n - 3) ∧
    (∀ n, b (n + 1) = 8 * a n + 7 * b n - 4) ∧
    ∀ n, ∃ m : ℕ, a n = m * m := sorry

end sequence_a_is_perfect_square_l2055_205512


namespace isosceles_triangle_perimeter_l2055_205537

theorem isosceles_triangle_perimeter (m : ℝ) (a b : ℝ) 
  (h1 : 3 = a ∨ 3 = b)
  (h2 : a ≠ b)
  (h3 : a^2 - (m+1)*a + 2*m = 0)
  (h4 : b^2 - (m+1)*b + 2*m = 0) :
  (a + b + a = 11) ∨ (a + a + b = 10) := 
sorry

end isosceles_triangle_perimeter_l2055_205537


namespace odd_expression_l2055_205578

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

theorem odd_expression (p q : ℕ) (hp : is_odd p) (hq : is_odd q) : is_odd (2 * p * p - q) :=
by
  sorry

end odd_expression_l2055_205578


namespace x_minus_p_eq_2_minus_2p_l2055_205550

theorem x_minus_p_eq_2_minus_2p (x p : ℝ) (h1 : |x - 3| = p + 1) (h2 : x < 3) : x - p = 2 - 2 * p := 
sorry

end x_minus_p_eq_2_minus_2p_l2055_205550


namespace lorelei_roses_l2055_205552

theorem lorelei_roses :
  let red_flowers := 12
  let pink_flowers := 18
  let yellow_flowers := 20
  let orange_flowers := 8
  let lorelei_red := (50 / 100) * red_flowers
  let lorelei_pink := (50 / 100) * pink_flowers
  let lorelei_yellow := (25 / 100) * yellow_flowers
  let lorelei_orange := (25 / 100) * orange_flowers
  lorelei_red + lorelei_pink + lorelei_yellow + lorelei_orange = 22 :=
by
  sorry

end lorelei_roses_l2055_205552


namespace jellybean_probability_l2055_205567

theorem jellybean_probability :
  let total_jellybeans := 15
  let green_jellybeans := 6
  let purple_jellybeans := 2
  let yellow_jellybeans := 7
  let total_picked := 4
  let total_ways := Nat.choose total_jellybeans total_picked
  let ways_to_pick_two_yellow := Nat.choose yellow_jellybeans 2
  let ways_to_pick_two_non_yellow := Nat.choose (total_jellybeans - yellow_jellybeans) 2
  let successful_outcomes := ways_to_pick_two_yellow * ways_to_pick_two_non_yellow
  let probability := successful_outcomes / total_ways
  probability = 4 / 9 := by
sorry

end jellybean_probability_l2055_205567


namespace excluded_avg_mark_l2055_205529

theorem excluded_avg_mark (N A A_remaining excluded_count : ℕ)
  (hN : N = 15)
  (hA : A = 80)
  (hA_remaining : A_remaining = 90) 
  (h_excluded : excluded_count = 5) :
  (A * N - A_remaining * (N - excluded_count)) / excluded_count = 60 := sorry

end excluded_avg_mark_l2055_205529


namespace expand_product_l2055_205501

theorem expand_product (y : ℝ) : 5 * (y - 6) * (y + 9) = 5 * y^2 + 15 * y - 270 := 
by
  sorry

end expand_product_l2055_205501


namespace additional_weight_difference_l2055_205516

theorem additional_weight_difference (raw_squat sleeves_add wraps_percentage : ℝ) 
  (raw_squat_val : raw_squat = 600) 
  (sleeves_add_val : sleeves_add = 30) 
  (wraps_percentage_val : wraps_percentage = 0.25) : 
  (wraps_percentage * raw_squat) - sleeves_add = 120 :=
by
  rw [ raw_squat_val, sleeves_add_val, wraps_percentage_val ]
  norm_num

end additional_weight_difference_l2055_205516


namespace goose_eggs_count_l2055_205532

theorem goose_eggs_count (E : ℕ)
  (hatch_ratio : ℚ := 2 / 3)
  (survive_first_month_ratio : ℚ := 3 / 4)
  (survive_first_year_ratio : ℚ := 2 / 5)
  (survived_first_year : ℕ := 130) :
  (survive_first_year_ratio * survive_first_month_ratio * hatch_ratio * (E : ℚ) = survived_first_year) →
  E = 1300 := by
  sorry

end goose_eggs_count_l2055_205532


namespace number_of_teams_l2055_205568

-- Total number of players
def total_players : Nat := 12

-- Number of ways to choose one captain
def ways_to_choose_captain : Nat := total_players

-- Number of remaining players after choosing the captain
def remaining_players : Nat := total_players - 1

-- Number of players needed to form a team (excluding the captain)
def team_size : Nat := 5

-- Number of ways to choose 5 players from the remaining 11
def ways_to_choose_team (n k : Nat) : Nat := Nat.choose n k

-- Total number of different teams
def total_teams : Nat := ways_to_choose_captain * ways_to_choose_team remaining_players team_size

theorem number_of_teams : total_teams = 5544 := by
  sorry

end number_of_teams_l2055_205568


namespace circle_bisection_relation_l2055_205558

theorem circle_bisection_relation (a b : ℝ) :
  (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = b^2 + 1 → (x + 1)^2 + (y + 1)^2 = 4) ↔ 
  a^2 + 2 * a + 2 * b + 5 = 0 :=
by sorry

end circle_bisection_relation_l2055_205558


namespace initial_number_of_macaroons_l2055_205515

theorem initial_number_of_macaroons 
  (w : ℕ) (bag_count : ℕ) (eaten_bag_count : ℕ) (remaining_weight : ℕ) 
  (macaroon_weight : ℕ) (remaining_bags : ℕ) (initial_macaroons : ℕ) :
  w = 5 → bag_count = 4 → eaten_bag_count = 1 → remaining_weight = 45 → 
  macaroon_weight = w → remaining_bags = (bag_count - eaten_bag_count) → 
  initial_macaroons = (remaining_bags * remaining_weight / macaroon_weight) * bag_count / remaining_bags →
  initial_macaroons = 12 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end initial_number_of_macaroons_l2055_205515


namespace exists_d_for_m_divides_f_of_f_n_l2055_205548

noncomputable def f : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => 23 * f (n + 1) + f n

theorem exists_d_for_m_divides_f_of_f_n (m : ℕ) : 
  ∃ (d : ℕ), ∀ (n : ℕ), m ∣ f (f n) ↔ d ∣ n := 
sorry

end exists_d_for_m_divides_f_of_f_n_l2055_205548


namespace range_of_expression_l2055_205593

theorem range_of_expression (x : ℝ) : (x + 2 ≥ 0 ∧ x - 1 ≠ 0) ↔ (x ≥ -2 ∧ x ≠ 1) :=
by
  sorry

end range_of_expression_l2055_205593


namespace reeyas_first_subject_score_l2055_205534

theorem reeyas_first_subject_score
  (second_subject_score third_subject_score fourth_subject_score : ℕ)
  (num_subjects : ℕ)
  (average_score : ℕ)
  (total_subjects_score : ℕ)
  (condition1 : second_subject_score = 76)
  (condition2 : third_subject_score = 82)
  (condition3 : fourth_subject_score = 85)
  (condition4 : num_subjects = 4)
  (condition5 : average_score = 75)
  (condition6 : total_subjects_score = num_subjects * average_score) :
  67 = total_subjects_score - (second_subject_score + third_subject_score + fourth_subject_score) := 
  sorry

end reeyas_first_subject_score_l2055_205534


namespace description_of_T_l2055_205553

-- Define the conditions
def T := { p : ℝ × ℝ | (∃ (c : ℝ), ((c = 5 ∨ c = p.1 + 3 ∨ c = p.2 - 6) ∧ (5 ≥ p.1 + 3) ∧ (5 ≥ p.2 - 6))) }

-- The main theorem
theorem description_of_T : 
  ∃ p : ℝ × ℝ, 
    (p = (2, 11)) ∧ 
    ∀ q ∈ T, 
      (q.fst = 2 ∧ q.snd ≤ 11) ∨ 
      (q.snd = 11 ∧ q.fst ≤ 2) ∨ 
      (q.snd = q.fst + 9 ∧ q.fst ≤ 2) :=
sorry

end description_of_T_l2055_205553


namespace area_of_inscribed_square_in_ellipse_l2055_205571

open Real

noncomputable def inscribed_square_area : ℝ := 32

theorem area_of_inscribed_square_in_ellipse :
  ∀ (x y : ℝ),
  (x^2 / 4 + y^2 / 8 = 1) →
  (x = t - t) ∧ (y = (t + t) / sqrt 2) ∧ 
  (t = sqrt 4) → inscribed_square_area = 32 :=
  sorry

end area_of_inscribed_square_in_ellipse_l2055_205571


namespace total_share_amount_l2055_205533

theorem total_share_amount (x y z : ℝ) (hx : y = 0.45 * x) (hz : z = 0.30 * x) (hy_share : y = 63) : x + y + z = 245 := by
  sorry

end total_share_amount_l2055_205533


namespace find_k_intersects_parabola_at_one_point_l2055_205519

theorem find_k_intersects_parabola_at_one_point :
  ∃ k : ℝ, (∀ y : ℝ, -3 * y^2 - 4 * y + 7 = k ↔ y = (-4 / (2 * 3))) →
    k = 25 / 3 :=
by sorry

end find_k_intersects_parabola_at_one_point_l2055_205519


namespace bridge_length_l2055_205575

theorem bridge_length (train_length : ℕ) (train_cross_bridge_time : ℕ) (train_cross_lamp_time : ℕ) (bridge_length : ℕ) :
  train_length = 600 →
  train_cross_bridge_time = 70 →
  train_cross_lamp_time = 20 →
  bridge_length = 1500 :=
by
  intro h1 h2 h3
  sorry

end bridge_length_l2055_205575


namespace parabola_point_coordinates_l2055_205527

theorem parabola_point_coordinates (x y : ℝ) (h_parabola : y^2 = 8 * x) 
    (h_distance_focus : (x + 2)^2 + y^2 = 81) : 
    (x = 7 ∧ y = 2 * Real.sqrt 14) ∨ (x = 7 ∧ y = -2 * Real.sqrt 14) :=
by {
  -- Proof will be inserted here
  sorry
}

end parabola_point_coordinates_l2055_205527


namespace complex_multiplication_l2055_205585

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (1 - 2 * i) = 2 + i :=
by
  sorry

end complex_multiplication_l2055_205585


namespace base9_problem_l2055_205520

def base9_add (a : ℕ) (b : ℕ) : ℕ := sorry -- Define actual addition for base 9
def base9_mul (a : ℕ) (b : ℕ) : ℕ := sorry -- Define actual multiplication for base 9

theorem base9_problem : base9_mul (base9_add 35 273) 2 = 620 := sorry

end base9_problem_l2055_205520


namespace distance_to_school_l2055_205554

def jerry_one_way_time : ℝ := 15  -- Jerry's one-way time in minutes
def carson_speed_mph : ℝ := 8  -- Carson's speed in miles per hour
def minutes_per_hour : ℝ := 60  -- Number of minutes in one hour

noncomputable def carson_speed_mpm : ℝ := carson_speed_mph / minutes_per_hour -- Carson's speed in miles per minute
def carson_one_way_time : ℝ := jerry_one_way_time -- Carson's one-way time is the same as Jerry's round trip time / 2

-- Prove that the distance to the school is 2 miles.
theorem distance_to_school : carson_speed_mpm * carson_one_way_time = 2 := by
  sorry

end distance_to_school_l2055_205554


namespace douglas_won_in_Y_l2055_205596

theorem douglas_won_in_Y (percent_total_vote : ℕ) (percent_vote_X : ℕ) (ratio_XY : ℕ) (P : ℕ) :
  percent_total_vote = 54 →
  percent_vote_X = 62 →
  ratio_XY = 2 →
  P = 38 :=
by
  sorry

end douglas_won_in_Y_l2055_205596


namespace problem_l2055_205565

theorem problem (x : ℝ) (h : x + 1 / x = 5) : x ^ 2 + (1 / x) ^ 2 = 23 := 
sorry

end problem_l2055_205565


namespace greatest_q_minus_r_l2055_205566

theorem greatest_q_minus_r :
  ∃ q r : ℤ, q > 0 ∧ r > 0 ∧ 975 = 23 * q + r ∧ q - r = 33 := sorry

end greatest_q_minus_r_l2055_205566


namespace range_of_a_l2055_205580

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 > 0) ↔ (-1 < a ∧ a < 3) := 
sorry

end range_of_a_l2055_205580


namespace average_runs_l2055_205505

/-- The average runs scored by the batsman in the first 20 matches is 40,
and in the next 10 matches is 30. We want to prove the average runs scored
by the batsman in all 30 matches is 36.67. --/
theorem average_runs (avg20 avg10 : ℕ) (num_matches_20 num_matches_10 : ℕ)
  (h1 : avg20 = 40) (h2 : avg10 = 30) (h3 : num_matches_20 = 20) (h4 : num_matches_10 = 10) :
  ((num_matches_20 * avg20 + num_matches_10 * avg10 : ℕ) : ℚ) / (num_matches_20 + num_matches_10 : ℕ) = 36.67 := by
  sorry

end average_runs_l2055_205505


namespace problem1_problem2a_problem2b_problem3_l2055_205508

noncomputable def f (a x : ℝ) := -x^2 + a * x - 2
noncomputable def g (x : ℝ) := x * Real.log x

-- Problem 1
theorem problem1 {a : ℝ} : (∀ x : ℝ, 0 < x → g x ≥ f a x) → a ≤ 3 :=
sorry

-- Problem 2 
theorem problem2a (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1 / Real.exp 1) :
  ∃ xmin : ℝ, g (1 / Real.exp 1) = -1 / Real.exp 1 ∧ 
  ∃ xmax : ℝ, g (m + 1) = (m + 1) * Real.log (m + 1) :=
sorry

theorem problem2b (m : ℝ) (h₀ : 1 / Real.exp 1 ≤ m) :
  ∃ xmin ymax : ℝ, xmin = g m ∧ ymax = g (m + 1) :=
sorry

-- Problem 3
theorem problem3 (x : ℝ) (h : 0 < x) : 
  Real.log x + (2 / (Real.exp 1 * x)) ≥ 1 / Real.exp x :=
sorry

end problem1_problem2a_problem2b_problem3_l2055_205508
