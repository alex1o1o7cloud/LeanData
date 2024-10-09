import Mathlib

namespace region_area_l95_9566

noncomputable def area_of_region_outside_hexagon_inside_semicircles (s : ℝ) : ℝ :=
  let area_hexagon := (3 * Real.sqrt 3 / 2) * s^2
  let area_semicircle := (1/2) * Real.pi * (s/2)^2
  let total_area_semicircles := 6 * area_semicircle
  let total_area_circles := 6 * Real.pi * (s/2)^2
  total_area_circles - area_hexagon

theorem region_area (s := 2) : area_of_region_outside_hexagon_inside_semicircles s = (6 * Real.pi - 6 * Real.sqrt 3) :=
by
  sorry  -- Proof is skipped.

end region_area_l95_9566


namespace initial_cows_l95_9598

theorem initial_cows (x : ℕ) (h : (3 / 4 : ℝ) * (x + 5) = 42) : x = 51 :=
by
  sorry

end initial_cows_l95_9598


namespace cody_final_tickets_l95_9532

def initial_tickets : ℝ := 56.5
def lost_tickets : ℝ := 6.3
def spent_tickets : ℝ := 25.75
def won_tickets : ℝ := 10.25
def dropped_tickets : ℝ := 3.1

theorem cody_final_tickets : 
  initial_tickets - lost_tickets - spent_tickets + won_tickets - dropped_tickets = 31.6 :=
by
  sorry

end cody_final_tickets_l95_9532


namespace minimum_value_of_PA_PF_l95_9540

noncomputable def ellipse_min_distance : ℝ :=
  let F := (1, 0)
  let A := (1, 1)
  let a : ℝ := 3
  let F1 := (-1, 0)
  let d_A_F1 : ℝ := Real.sqrt ((-1 - 1)^2 + (0 - 1)^2)
  6 - d_A_F1

theorem minimum_value_of_PA_PF :
  ellipse_min_distance = 6 - Real.sqrt 5 :=
by
  sorry

end minimum_value_of_PA_PF_l95_9540


namespace Penny_total_species_identified_l95_9583

/-- Penny identified 35 species of sharks, 15 species of eels, and 5 species of whales.
    Prove that the total number of species identified is 55. -/
theorem Penny_total_species_identified :
  let sharks_species := 35
  let eels_species := 15
  let whales_species := 5
  sharks_species + eels_species + whales_species = 55 :=
by
  sorry

end Penny_total_species_identified_l95_9583


namespace opposite_neg_2023_l95_9505

theorem opposite_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_neg_2023_l95_9505


namespace find_a10_l95_9525

variable {n : ℕ}
variable (a : ℕ → ℝ)
variable (h_pos : ∀ (n : ℕ), 0 < a n)
variable (h_mul : ∀ (p q : ℕ), a (p + q) = a p * a q)
variable (h_a8 : a 8 = 16)

theorem find_a10 : a 10 = 32 :=
by
  sorry

end find_a10_l95_9525


namespace tan_sum_simplification_l95_9514

theorem tan_sum_simplification :
  (Real.tan (Real.pi / 12) + Real.tan (Real.pi / 4)) = 2 * Real.sqrt 6 - 2 * Real.sqrt 3 :=
by
  sorry

end tan_sum_simplification_l95_9514


namespace additional_books_acquired_l95_9565

def original_stock : ℝ := 40.0
def shelves_used : ℕ := 15
def books_per_shelf : ℝ := 4.0

theorem additional_books_acquired :
  (shelves_used * books_per_shelf) - original_stock = 20.0 :=
by
  sorry

end additional_books_acquired_l95_9565


namespace probability_of_one_radio_operator_per_group_l95_9513

def total_ways_to_assign_soldiers_to_groups : ℕ := 27720
def ways_to_assign_radio_operators_to_groups : ℕ := 7560

theorem probability_of_one_radio_operator_per_group :
  (ways_to_assign_radio_operators_to_groups : ℚ) / (total_ways_to_assign_soldiers_to_groups : ℚ) = 3 / 11 := 
sorry

end probability_of_one_radio_operator_per_group_l95_9513


namespace power_addition_l95_9564

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end power_addition_l95_9564


namespace problems_per_worksheet_l95_9504

theorem problems_per_worksheet (P : ℕ) (graded : ℕ) (remaining : ℕ) (total_worksheets : ℕ) (total_problems_remaining : ℕ) :
    graded = 5 →
    total_worksheets = 9 →
    total_problems_remaining = 16 →
    remaining = total_worksheets - graded →
    4 * P = total_problems_remaining →
    P = 4 :=
by
  intros h_graded h_worksheets h_problems h_remaining h_equation
  sorry

end problems_per_worksheet_l95_9504


namespace find_m_l95_9545

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def point_on_x_axis_distance (x y : ℝ) : Prop :=
  y = 14

def point_distance_from_fixed_point (x y : ℝ) : Prop :=
  distance (x, y) (3, 8) = 8

def x_coordinate_condition (x : ℝ) : Prop :=
  x > 3

def m_distance (x y m : ℝ) : Prop :=
  distance (x, y) (0, 0) = m

theorem find_m (x y m : ℝ) 
  (h1 : point_on_x_axis_distance x y) 
  (h2 : point_distance_from_fixed_point x y) 
  (h3 : x_coordinate_condition x) :
  m_distance x y m → 
  m = Real.sqrt (233 + 12 * Real.sqrt 7) := by
  sorry

end find_m_l95_9545


namespace hyperbola_real_axis_length_l95_9553

theorem hyperbola_real_axis_length (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_hyperbola : ∀ x y : ℝ, x = 1 → y = 2 → (x^2 / (a^2)) - (y^2 / (b^2)) = 1)
  (h_parabola : ∀ y : ℝ, y = 2 → (y^2) = 4 * 1)
  (h_focus : (1, 2) = (1, 2))
  (h_eq : a^2 + b^2 = 1) :
  2 * a = 2 * (Real.sqrt 2 - 1) :=
by 
-- Skipping the proof part
sorry

end hyperbola_real_axis_length_l95_9553


namespace johns_raise_percent_increase_l95_9554

theorem johns_raise_percent_increase (original_earnings new_earnings : ℝ) 
  (h₀ : original_earnings = 60) (h₁ : new_earnings = 110) : 
  ((new_earnings - original_earnings) / original_earnings) * 100 = 83.33 :=
by
  sorry

end johns_raise_percent_increase_l95_9554


namespace right_triangle_OAB_condition_l95_9516

theorem right_triangle_OAB_condition
  (a b : ℝ)
  (h1: a ≠ 0) 
  (h2: b ≠ 0) :
  (b - a^3) * (b - a^3 - 1/a) = 0 :=
sorry

end right_triangle_OAB_condition_l95_9516


namespace no_integer_solution_for_expression_l95_9570

theorem no_integer_solution_for_expression (x y z : ℤ) :
  x^4 + y^4 + z^4 - 2 * x^2 * y^2 - 2 * y^2 * z^2 - 2 * z^2 * x^2 ≠ 2000 :=
by sorry

end no_integer_solution_for_expression_l95_9570


namespace geometric_seq_common_ratio_l95_9510

theorem geometric_seq_common_ratio 
  (a : ℕ → ℝ) -- a_n is the sequence
  (S : ℕ → ℝ) -- S_n is the partial sum of the sequence
  (h1 : a 3 = 2 * S 2 + 1) -- condition a_3 = 2S_2 + 1
  (h2 : a 4 = 2 * S 3 + 1) -- condition a_4 = 2S_3 + 1
  (h3 : S 2 = a 1 / (1 / q) * (1 - q^3) / (1 - q)) -- sum of first 2 terms
  (h4 : S 3 = a 1 / (1 / q) * (1 - q^4) / (1 - q)) -- sum of first 3 terms
  : q = 3 := -- conclusion
by sorry

end geometric_seq_common_ratio_l95_9510


namespace g_g_2_equals_226_l95_9548

def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 4

theorem g_g_2_equals_226 : g (g 2) = 226 := by
  sorry

end g_g_2_equals_226_l95_9548


namespace viola_final_jump_l95_9543

variable (n : ℕ) (T : ℝ) (x : ℝ)

theorem viola_final_jump (h1 : T = 3.80 * n)
                        (h2 : (T + 3.99) / (n + 1) = 3.81)
                        (h3 : T + 3.99 + x = 3.82 * (n + 2)) : 
                        x = 4.01 :=
sorry

end viola_final_jump_l95_9543


namespace speed_of_stream_l95_9591

theorem speed_of_stream (D v : ℝ) (h1 : ∀ D, D / (54 - v) = 2 * (D / (54 + v))) : v = 18 := 
sorry

end speed_of_stream_l95_9591


namespace fran_speed_calculation_l95_9575

noncomputable def fran_speed (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) : ℝ :=
  joann_speed * joann_time / fran_time

theorem fran_speed_calculation : 
  fran_speed 15 3 2.5 = 18 := 
by
  -- Remember to write down the proof steps if needed, currently we use sorry as placeholder
  sorry

end fran_speed_calculation_l95_9575


namespace min_ab_given_parallel_l95_9530

-- Define the conditions
def parallel_vectors (a b : ℝ) : Prop :=
  4 * b - a * (b - 1) = 0 ∧ b > 1

-- Prove the main statement
theorem min_ab_given_parallel (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h_parallel : parallel_vectors a b) :
  a + b = 9 :=
sorry  -- Proof is omitted

end min_ab_given_parallel_l95_9530


namespace p_sufficient_but_not_necessary_for_q_l95_9533

-- Definitions
variable {p q : Prop}

-- The condition: ¬p is a necessary but not sufficient condition for ¬q
def necessary_but_not_sufficient (p q : Prop) : Prop :=
  (∀ q, ¬q → ¬p) ∧ (∃ q, ¬q ∧ p)

-- The theorem stating the problem
theorem p_sufficient_but_not_necessary_for_q 
  (h : necessary_but_not_sufficient (¬p) (¬q)) : 
  (∀ p, p → q) ∧ (∃ p, p ∧ ¬q) :=
sorry

end p_sufficient_but_not_necessary_for_q_l95_9533


namespace jenny_boxes_sold_l95_9515

/--
Jenny sold some boxes of Trefoils. Each box has 8.0 packs. She sold 192 packs in total.
Prove that Jenny sold 24 boxes.
-/
theorem jenny_boxes_sold (packs_per_box : Real) (total_packs_sold : Real) (num_boxes_sold : Real) 
  (h1 : packs_per_box = 8.0) (h2 : total_packs_sold = 192) : num_boxes_sold = 24 :=
by
  have h3 : num_boxes_sold = total_packs_sold / packs_per_box :=
    by sorry
  sorry

end jenny_boxes_sold_l95_9515


namespace least_number_of_cans_l95_9501

theorem least_number_of_cans (maaza : ℕ) (pepsi : ℕ) (sprite : ℕ) (gcd_val : ℕ) (total_cans : ℕ)
  (h1 : maaza = 50) (h2 : pepsi = 144) (h3 : sprite = 368) (h_gcd : gcd maaza (gcd pepsi sprite) = gcd_val)
  (h_total_cans : total_cans = maaza / gcd_val + pepsi / gcd_val + sprite / gcd_val) :
  total_cans = 281 :=
sorry

end least_number_of_cans_l95_9501


namespace remainder_cd_42_l95_9592

theorem remainder_cd_42 (c d : ℕ) (p q : ℕ) (hc : c = 84 * p + 76) (hd : d = 126 * q + 117) : 
  (c + d) % 42 = 25 :=
by
  sorry

end remainder_cd_42_l95_9592


namespace football_game_wristbands_l95_9586

theorem football_game_wristbands (total_wristbands wristbands_per_person : Nat) (h1 : total_wristbands = 290) (h2 : wristbands_per_person = 2) :
  total_wristbands / wristbands_per_person = 145 :=
by
  sorry

end football_game_wristbands_l95_9586


namespace total_people_who_eat_vegetarian_l95_9520

def people_who_eat_only_vegetarian := 16
def people_who_eat_both_vegetarian_and_non_vegetarian := 12

-- We want to prove that the total number of people who eat vegetarian is 28
theorem total_people_who_eat_vegetarian : 
  people_who_eat_only_vegetarian + people_who_eat_both_vegetarian_and_non_vegetarian = 28 :=
by 
  sorry

end total_people_who_eat_vegetarian_l95_9520


namespace mary_unanswered_questions_l95_9506

theorem mary_unanswered_questions :
  ∃ (c w u : ℕ), 150 = 6 * c + 3 * u ∧ 118 = 40 + 5 * c - 2 * w ∧ 50 = c + w + u ∧ u = 16 :=
by
  sorry

end mary_unanswered_questions_l95_9506


namespace cost_of_72_tulips_is_115_20_l95_9599

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

end cost_of_72_tulips_is_115_20_l95_9599


namespace compare_abc_l95_9546

noncomputable def a : ℝ := 2 * Real.log (21 / 20)
noncomputable def b : ℝ := Real.log (11 / 10)
noncomputable def c : ℝ := Real.sqrt 1.2 - 1

theorem compare_abc : a > b ∧ b < c ∧ a > c :=
by {
  sorry
}

end compare_abc_l95_9546


namespace no_integer_solutions_3a2_eq_b2_plus_1_l95_9558

theorem no_integer_solutions_3a2_eq_b2_plus_1 : 
  ¬ ∃ a b : ℤ, 3 * a^2 = b^2 + 1 :=
by
  intro h
  obtain ⟨a, b, hab⟩ := h
  sorry

end no_integer_solutions_3a2_eq_b2_plus_1_l95_9558


namespace f_one_zero_x_range_l95_9528

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
-- f is defined for x > 0
variable (f : ℝ → ℝ)
variables (h_domain : ∀ x, x > 0 → ∃ y, f x = y)
variables (h1 : f 2 = 1)
variables (h2 : ∀ x y, x > 0 → y > 0 → f (x * y) = f x + f y)
variables (h3 : ∀ x y, x > y → f x > f y)

-- Question 1
theorem f_one_zero (hf1 : f 1 = 0) : True := 
  by trivial
  
-- Question 2
theorem x_range (x: ℝ) (hx: f 3 + f (4 - 8 * x) > 2) : x ≤ 1/3 := sorry

end f_one_zero_x_range_l95_9528


namespace find_principal_amount_l95_9511

variable (P : ℝ)
variable (R : ℝ := 5)
variable (T : ℝ := 13)
variable (SI : ℝ := 1300)

theorem find_principal_amount (h1 : SI = (P * R * T) / 100) : P = 2000 :=
sorry

end find_principal_amount_l95_9511


namespace workers_together_time_l95_9562

-- Definition of the times taken by each worker to complete the job
def timeA : ℚ := 8
def timeB : ℚ := 10
def timeC : ℚ := 12

-- Definition of the rates based on the times
def rateA : ℚ := 1 / timeA
def rateB : ℚ := 1 / timeB
def rateC : ℚ := 1 / timeC

-- Definition of the total rate when working together
def total_rate : ℚ := rateA + rateB + rateC

-- Definition of the total time taken to complete the job when working together
def total_time : ℚ := 1 / total_rate

-- The final theorem we need to prove
theorem workers_together_time : total_time = 120 / 37 :=
by {
  -- structure of the proof will go here, but it is not required as per the instructions
  sorry
}

end workers_together_time_l95_9562


namespace find_black_balls_l95_9522

-- Define the conditions given in the problem.
def initial_balls : ℕ := 10
def all_red_balls (p_red : ℝ) : Prop := p_red = 1
def equal_red_black (p_red : ℝ) (p_black : ℝ) : Prop := p_red = 0.5 ∧ p_black = 0.5
def with_green_balls (p_red : ℝ) (green_balls : ℕ) : Prop := green_balls = 2 ∧ p_red = 0.7

-- Define the total probability condition
def total_probability (p_red : ℝ) (p_green : ℝ) (p_black : ℝ) : Prop :=
  p_red + p_green + p_black = 1

-- The final statement to prove
theorem find_black_balls :
  ∃ black_balls : ℕ,
    initial_balls = 10 ∧
    (∃ p_red : ℝ, all_red_balls p_red) ∧
    (∃ p_red p_black : ℝ, equal_red_black p_red p_black) ∧
    (∃ p_red : ℝ, ∃ green_balls : ℕ, with_green_balls p_red green_balls) ∧
    (∃ p_red p_green p_black : ℝ, total_probability p_red p_green p_black) ∧
    black_balls = 1 :=
sorry

end find_black_balls_l95_9522


namespace minimum_order_amount_to_get_discount_l95_9509

theorem minimum_order_amount_to_get_discount 
  (cost_quiche : ℝ) (cost_croissant : ℝ) (cost_biscuit : ℝ) (n_quiches : ℝ) (n_croissants : ℝ) (n_biscuits : ℝ)
  (discount_percent : ℝ) (total_with_discount : ℝ) (min_order_amount : ℝ) :
  cost_quiche = 15.0 → cost_croissant = 3.0 → cost_biscuit = 2.0 →
  n_quiches = 2 → n_croissants = 6 → n_biscuits = 6 →
  discount_percent = 0.10 → total_with_discount = 54.0 →
  (n_quiches * cost_quiche + n_croissants * cost_croissant + n_biscuits * cost_biscuit) * (1 - discount_percent) = total_with_discount →
  min_order_amount = 60.0 :=
by
  sorry

end minimum_order_amount_to_get_discount_l95_9509


namespace sara_peaches_l95_9549

theorem sara_peaches (initial_peaches : ℕ) (picked_peaches : ℕ) (total_peaches : ℕ) 
  (h1 : initial_peaches = 24) (h2 : picked_peaches = 37) : 
  total_peaches = 61 :=
by
  sorry

end sara_peaches_l95_9549


namespace reservoir_capacity_l95_9593

theorem reservoir_capacity (x : ℝ) (h1 : (3 / 8) * x - (1 / 4) * x = 100) : x = 800 :=
by
  sorry

end reservoir_capacity_l95_9593


namespace no_such_a_exists_l95_9595

def A (a : ℝ) : Set ℝ := {2, 4, a^3 - 2*a^2 - a + 7}
def B (a : ℝ) : Set ℝ := {1, 5*a - 5, -1/2*a^2 + 3/2*a + 4, a^3 + a^2 + 3*a + 7}

theorem no_such_a_exists (a : ℝ) : ¬(A a ∩ B a = {2, 5}) :=
by
  sorry

end no_such_a_exists_l95_9595


namespace simplify_fractional_equation_l95_9551

theorem simplify_fractional_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 2) : (x / (x - 2) - 2 = 3 / (2 - x)) → (x - 2 * (x - 2) = -3) :=
by
  sorry

end simplify_fractional_equation_l95_9551


namespace difference_between_numbers_l95_9529

theorem difference_between_numbers 
  (A B : ℝ)
  (h1 : 0.075 * A = 0.125 * B)
  (h2 : A = 2430 ∨ B = 2430) :
  A - B = 972 :=
by
  sorry

end difference_between_numbers_l95_9529


namespace speed_with_stream_l95_9568

variable (V_m V_s : ℝ)

def against_speed : Prop := V_m - V_s = 13
def still_water_rate : Prop := V_m = 6

theorem speed_with_stream (h1 : against_speed V_m V_s) (h2 : still_water_rate V_m) : V_m + V_s = 13 := 
sorry

end speed_with_stream_l95_9568


namespace car_more_miles_per_tank_after_modification_l95_9556

theorem car_more_miles_per_tank_after_modification (mpg_old : ℕ) (efficiency_factor : ℝ) (gallons : ℕ) :
  mpg_old = 33 →
  efficiency_factor = 1.25 →
  gallons = 16 →
  (efficiency_factor * mpg_old * gallons - mpg_old * gallons) = 132 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry  -- Proof omitted

end car_more_miles_per_tank_after_modification_l95_9556


namespace binomial_param_exact_l95_9519

variable (ξ : ℕ → ℝ) (n : ℕ) (p : ℝ)

-- Define the conditions: expectation and variance
axiom expectation_eq : n * p = 3
axiom variance_eq : n * p * (1 - p) = 2

-- Statement to prove
theorem binomial_param_exact (h1 : n * p = 3) (h2 : n * p * (1 - p) = 2) : p = 1 / 3 :=
by
  rw [expectation_eq] at h2
  sorry

end binomial_param_exact_l95_9519


namespace mother_l95_9521

def age_relations (P M : ℕ) : Prop :=
  P = (2 * M) / 5 ∧ P + 10 = (M + 10) / 2

theorem mother's_present_age (P M : ℕ) (h : age_relations P M) : M = 50 :=
by
  sorry

end mother_l95_9521


namespace megatek_manufacturing_percentage_l95_9563

theorem megatek_manufacturing_percentage :
  ∀ (total_degrees manufacturing_degrees total_percentage : ℝ),
  total_degrees = 360 → manufacturing_degrees = 216 → total_percentage = 100 →
  (manufacturing_degrees / total_degrees) * total_percentage = 60 :=
by
  intros total_degrees manufacturing_degrees total_percentage H1 H2 H3
  rw [H1, H2, H3]
  sorry

end megatek_manufacturing_percentage_l95_9563


namespace jess_double_cards_l95_9576

theorem jess_double_cards (rob_total_cards jess_doubles : ℕ) 
    (one_third_rob_cards_doubles : rob_total_cards / 3 = rob_total_cards / 3)
    (jess_times_rob_doubles : jess_doubles = 5 * (rob_total_cards / 3)) :
    rob_total_cards = 24 → jess_doubles = 40 :=
  by
  sorry

end jess_double_cards_l95_9576


namespace existence_of_solution_largest_unsolvable_n_l95_9552

-- Definitions based on the conditions provided in the problem
def equation (x y z n : ℕ) : Prop := 28 * x + 30 * y + 31 * z = n

-- There exist positive integers x, y, z such that 28x + 30y + 31z = 365
theorem existence_of_solution : ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ equation x y z 365 :=
by
  sorry

-- The largest positive integer n such that 28x + 30y + 31z = n cannot be solved in positive integers x, y, z is 370
theorem largest_unsolvable_n : ∀ (n : ℕ), (∀ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 → n ≠ 370) → ∀ (n' : ℕ), n' > 370 → (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ equation x y z n') :=
by
  sorry

end existence_of_solution_largest_unsolvable_n_l95_9552


namespace officers_count_l95_9544

theorem officers_count (average_salary_all : ℝ) (average_salary_officers : ℝ) 
    (average_salary_non_officers : ℝ) (num_non_officers : ℝ) (total_salary : ℝ) : 
    average_salary_all = 120 → 
    average_salary_officers = 470 →  
    average_salary_non_officers = 110 → 
    num_non_officers = 525 → 
    total_salary = average_salary_all * (num_non_officers + O) → 
    total_salary = average_salary_officers * O + average_salary_non_officers * num_non_officers → 
    O = 15 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end officers_count_l95_9544


namespace consecutive_integers_sum_l95_9527

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l95_9527


namespace yellow_more_than_purple_l95_9579
-- Import math library for necessary definitions.

-- Define the problem conditions in Lean
def num_purple_candies : ℕ := 10
def num_total_candies : ℕ := 36

axiom exists_yellow_and_green_candies 
  (Y G : ℕ) 
  (h1 : G = Y - 2) 
  (h2 : 10 + Y + G = 36) : True

-- The theorem to prove
theorem yellow_more_than_purple 
  (Y : ℕ) 
  (hY : exists (G : ℕ), G = Y - 2 ∧ 10 + Y + G = 36) : Y - num_purple_candies = 4 :=
by {
  sorry -- proof is not required
}

end yellow_more_than_purple_l95_9579


namespace mary_investment_l95_9537

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem mary_investment :
  ∃ (P : ℝ), P = 51346 ∧ compound_interest P 0.10 12 7 = 100000 :=
by
  sorry

end mary_investment_l95_9537


namespace fraction_equation_l95_9518

theorem fraction_equation (a : ℕ) (h : a > 0) (eq : (a : ℚ) / (a + 35) = 0.875) : a = 245 :=
by
  sorry

end fraction_equation_l95_9518


namespace schedule_courses_l95_9507

/-- Definition of valid schedule count where at most one pair of courses is consecutive. -/
def count_valid_schedules : ℕ := 180

/-- Given 7 periods and 3 courses, determine the number of valid schedules 
    where at most one pair of these courses is consecutive. -/
theorem schedule_courses (periods : ℕ) (courses : ℕ) (valid_schedules : ℕ) :
  periods = 7 → courses = 3 → valid_schedules = count_valid_schedules →
  valid_schedules = 180 :=
by
  intros h1 h2 h3
  sorry

end schedule_courses_l95_9507


namespace eq_infinite_solutions_function_satisfies_identity_l95_9541

-- First Part: Proving the equation has infinitely many positive integer solutions
theorem eq_infinite_solutions : ∃ (x y z t : ℕ), ∀ n : ℕ, x^2 + 2 * y^2 = z^2 + 2 * t^2 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 := 
sorry

-- Second Part: Finding and proving the function f
def f (n : ℕ) : ℕ := n

theorem function_satisfies_identity (f : ℕ → ℕ) (h : ∀ m n : ℕ, f (f n^2 + 2 * f m^2) = n^2 + 2 * m^2) : ∀ k : ℕ, f k = k :=
sorry

end eq_infinite_solutions_function_satisfies_identity_l95_9541


namespace jill_speed_downhill_l95_9535

theorem jill_speed_downhill 
  (up_speed : ℕ) (total_time : ℕ) (hill_distance : ℕ) 
  (up_time : ℕ) (down_time : ℕ) (down_speed : ℕ) 
  (h1 : up_speed = 9)
  (h2 : total_time = 175)
  (h3 : hill_distance = 900)
  (h4 : up_time = hill_distance / up_speed)
  (h5 : down_time = total_time - up_time)
  (h6 : down_speed = hill_distance / down_time) :
  down_speed = 12 := 
  by
    sorry

end jill_speed_downhill_l95_9535


namespace distance_traveled_l95_9571

-- Definition of the velocity function
def velocity (t : ℝ) : ℝ := 2 * t - 3

-- Prove the integral statement
theorem distance_traveled : 
  (∫ t in (0 : ℝ)..(5 : ℝ), abs (velocity t)) = 29 / 2 := by 
{ sorry }

end distance_traveled_l95_9571


namespace multiplication_addition_l95_9512

theorem multiplication_addition :
  108 * 108 + 92 * 92 = 20128 :=
by
  sorry

end multiplication_addition_l95_9512


namespace salt_concentration_solution_l95_9526

theorem salt_concentration_solution
  (x y z : ℕ)
  (h1 : x + y + z = 30)
  (h2 : 2 * x + 3 * y = 35)
  (h3 : 3 * y + 2 * z = 45) :
  x = 10 ∧ y = 5 ∧ z = 15 := by
  sorry

end salt_concentration_solution_l95_9526


namespace gcf_2550_7140_l95_9503

def gcf (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcf_2550_7140 : gcf 2550 7140 = 510 := 
  by 
    sorry

end gcf_2550_7140_l95_9503


namespace part1_part2_l95_9517

noncomputable def quadratic_eq (m x : ℝ) : Prop := m * x^2 - 2 * x + 1 = 0

theorem part1 (m : ℝ) : 
  (∃ x1 x2 : ℝ, quadratic_eq m x1 ∧ quadratic_eq m x2 ∧ x1 ≠ x2) ↔ (m ≤ 1 ∧ m ≠ 0) :=
by sorry

theorem part2 (m : ℝ) (x1 x2 : ℝ) : 
  (quadratic_eq m x1 ∧ quadratic_eq m x2 ∧ x1 * x2 - x1 - x2 = 1/2) ↔ (m = -2) :=
by sorry

end part1_part2_l95_9517


namespace part1_part1_monotonicity_intervals_part2_l95_9573

noncomputable def f (x a : ℝ) := x * Real.log x - a * (x - 1)^2 - x + 1

-- Part 1: Monotonicity and Extreme values when a = 0
theorem part1 (x : ℝ) : f x 0 = x * Real.log x - x + 1 := sorry

theorem part1_monotonicity_intervals (x : ℝ) :
  (∀ (x : ℝ), 0 < x ∧ x < 1 → f x 0 < f 1 0) ∧
  (∀ (x : ℝ), x > 1 → f 1 0 < f x 0) ∧ 
  (f 1 0 = 0) := sorry

-- Part 2: f(x) < 0 for x > 1 and a >= 1/2
theorem part2 (x a : ℝ) (hx : x > 1) (ha : a ≥ 1/2) : f x a < 0 := sorry

end part1_part1_monotonicity_intervals_part2_l95_9573


namespace carol_rectangle_length_l95_9584

theorem carol_rectangle_length (lCarol : ℝ) :
    (∃ (wCarol : ℝ), wCarol = 20 ∧ lCarol * wCarol = 300) ↔ lCarol = 15 :=
by
  have jordan_area : 6 * 50 = 300 := by norm_num
  sorry

end carol_rectangle_length_l95_9584


namespace units_digit_of_product_of_seven_consecutive_integers_is_zero_l95_9587

/-- Define seven consecutive positive integers and show the units digit of their product is 0 -/
theorem units_digit_of_product_of_seven_consecutive_integers_is_zero (n : ℕ) :
  ∃ (k : ℕ), k = (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) % 10 ∧ k = 0 :=
by {
  -- We state that the units digit k of the product of seven consecutive integers
  -- starting from n is 0
  sorry
}

end units_digit_of_product_of_seven_consecutive_integers_is_zero_l95_9587


namespace students_present_l95_9578

theorem students_present (absent_students male_students female_student_diff : ℕ) 
  (h1 : absent_students = 18) 
  (h2 : male_students = 848) 
  (h3 : female_student_diff = 49) : 
  (male_students + (male_students - female_student_diff) - absent_students = 1629) := 

by 
  sorry

end students_present_l95_9578


namespace solution_set_inequality_l95_9574

theorem solution_set_inequality (x : ℝ) : (-2 * x + 3 < 0) ↔ (x > 3 / 2) := by 
  sorry

end solution_set_inequality_l95_9574


namespace calculate_grand_total_profit_l95_9590

-- Definitions based on conditions
def cost_per_type_A : ℕ := 8 * 10
def sell_price_type_A : ℕ := 125
def cost_per_type_B : ℕ := 12 * 18
def sell_price_type_B : ℕ := 280
def cost_per_type_C : ℕ := 15 * 12
def sell_price_type_C : ℕ := 350

def num_sold_type_A : ℕ := 45
def num_sold_type_B : ℕ := 35
def num_sold_type_C : ℕ := 25

-- Definition of profit calculations
def profit_per_type_A : ℕ := sell_price_type_A - cost_per_type_A
def profit_per_type_B : ℕ := sell_price_type_B - cost_per_type_B
def profit_per_type_C : ℕ := sell_price_type_C - cost_per_type_C

def total_profit_type_A : ℕ := num_sold_type_A * profit_per_type_A
def total_profit_type_B : ℕ := num_sold_type_B * profit_per_type_B
def total_profit_type_C : ℕ := num_sold_type_C * profit_per_type_C

def grand_total_profit : ℕ := total_profit_type_A + total_profit_type_B + total_profit_type_C

-- Statement to be proved
theorem calculate_grand_total_profit : grand_total_profit = 8515 := by
  sorry

end calculate_grand_total_profit_l95_9590


namespace compute_f_1_g_3_l95_9534

def f (x : ℝ) := 3 * x - 5
def g (x : ℝ) := x + 1

theorem compute_f_1_g_3 : f (1 + g 3) = 10 := by
  sorry

end compute_f_1_g_3_l95_9534


namespace f_decreasing_on_0_1_l95_9523

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x⁻¹

theorem f_decreasing_on_0_1 : ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → x₁ < x₂ → f x₁ > f x₂ := by
  sorry

end f_decreasing_on_0_1_l95_9523


namespace tricycle_total_spokes_l95_9594

noncomputable def front : ℕ := 20
noncomputable def middle : ℕ := 2 * front
noncomputable def back : ℝ := 20 * Real.sqrt 2
noncomputable def total_spokes : ℝ := front + middle + back

theorem tricycle_total_spokes : total_spokes = 88 :=
by
  sorry

end tricycle_total_spokes_l95_9594


namespace unique_pyramid_formation_l95_9582

theorem unique_pyramid_formation:
  ∀ (positions: Finset ℕ)
    (is_position_valid: ℕ → Prop),
    (positions.card = 5) → 
    (∀ n ∈ positions, n < 5) → 
    (∃! n, is_position_valid n) :=
by
  sorry

end unique_pyramid_formation_l95_9582


namespace one_third_of_1206_is_300_percent_of_134_l95_9580

theorem one_third_of_1206_is_300_percent_of_134 :
  let number := 1206
  let fraction := 1 / 3
  let computed_one_third := fraction * number
  let whole := 134
  let expected_percent := 300
  let percent := (computed_one_third / whole) * 100
  percent = expected_percent := by
  let number := 1206
  let fraction := 1 / 3
  have computed_one_third : ℝ := fraction * number
  let whole := 134
  let expected_percent := 300
  have percent : ℝ := (computed_one_third / whole) * 100
  exact sorry

end one_third_of_1206_is_300_percent_of_134_l95_9580


namespace total_cubes_in_stack_l95_9542

theorem total_cubes_in_stack :
  let bottom_layer := 4
  let middle_layer := 2
  let top_layer := 1
  bottom_layer + middle_layer + top_layer = 7 :=
by
  sorry

end total_cubes_in_stack_l95_9542


namespace calculate_expression_l95_9569

theorem calculate_expression : 
  (3.242 * (14 + 6) - 7.234 * 7) / 20 = 0.7101 :=
by
  sorry

end calculate_expression_l95_9569


namespace parabola_vertex_sum_l95_9596

theorem parabola_vertex_sum 
  (a b c : ℝ)
  (h1 : ∀ x : ℝ, (a * x^2 + b * x + c) = (a * (x + 3)^2 + 4))
  (h2 : (a * 49 + 4) = -2)
  : a + b + c = 100 / 49 :=
by
  sorry

end parabola_vertex_sum_l95_9596


namespace trenton_commission_rate_l95_9550

noncomputable def commission_rate (fixed_earnings : ℕ) (goal : ℕ) (sales : ℕ) : ℚ :=
  ((goal - fixed_earnings : ℤ) / (sales : ℤ)) * 100

theorem trenton_commission_rate :
  commission_rate 190 500 7750 = 4 := 
  by
  sorry

end trenton_commission_rate_l95_9550


namespace marble_group_l95_9557

theorem marble_group (x : ℕ) (h1 : 144 % x = 0) (h2 : 144 % (x + 2) = (144 / x) - 1) : x = 16 :=
sorry

end marble_group_l95_9557


namespace logs_per_tree_is_75_l95_9502

-- Definitions
def logsPerDay : Nat := 5

def totalDays : Nat := 30 + 31 + 31 + 28

def totalLogs (burnRate : Nat) (days : Nat) : Nat :=
  burnRate * days

def treesNeeded : Nat := 8

def logsPerTree (totalLogs : Nat) (numTrees : Nat) : Nat :=
  totalLogs / numTrees

-- Theorem statement to prove the number of logs per tree
theorem logs_per_tree_is_75 : logsPerTree (totalLogs logsPerDay totalDays) treesNeeded = 75 :=
  by
  sorry

end logs_per_tree_is_75_l95_9502


namespace number_of_adult_tickets_l95_9577

-- Define the parameters of the problem
def price_adult_ticket : ℝ := 5.50
def price_child_ticket : ℝ := 3.50
def total_tickets : ℕ := 21
def total_cost : ℝ := 83.50

-- Define the main theorem to be proven
theorem number_of_adult_tickets : 
  ∃ (A C : ℕ), A + C = total_tickets ∧ 
                (price_adult_ticket * A + price_child_ticket * C = total_cost) ∧ 
                 A = 5 :=
by
  -- The proof content will be filled in later
  sorry

end number_of_adult_tickets_l95_9577


namespace brick_length_l95_9524

theorem brick_length (x : ℝ) (brick_width : ℝ) (brick_height : ℝ) (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ) (number_of_bricks : ℕ)
  (h_brick : brick_width = 11.25) (h_brick_height : brick_height = 6)
  (h_wall : wall_length = 800) (h_wall_width : wall_width = 600) 
  (h_wall_height : wall_height = 22.5) (h_bricks_number : number_of_bricks = 1280)
  (h_eq : (wall_length * wall_width * wall_height) = (x * brick_width * brick_height) * number_of_bricks) : 
  x = 125 := by
  sorry

end brick_length_l95_9524


namespace number_of_lines_passing_through_point_and_forming_given_area_l95_9555

theorem number_of_lines_passing_through_point_and_forming_given_area :
  ∃ l : ℝ → ℝ, (∀ x y : ℝ, l 1 = 1) ∧ (∃ (a b : ℝ), abs ((1/2) * a * b) = 2)
  → (∃ n : ℕ, n = 4) :=
by
  sorry

end number_of_lines_passing_through_point_and_forming_given_area_l95_9555


namespace divisors_72_l95_9597

theorem divisors_72 : 
  { d | d ∣ 72 ∧ 0 < d } = {1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72} := 
sorry

end divisors_72_l95_9597


namespace division_by_fraction_l95_9581

theorem division_by_fraction : 5 / (1 / 5) = 25 := by
  sorry

end division_by_fraction_l95_9581


namespace simplify_and_evaluate_l95_9589

noncomputable def x : ℕ := 2023
noncomputable def y : ℕ := 2

theorem simplify_and_evaluate :
  (x + 2 * y)^2 - ((x^3 + 4 * x^2 * y) / x) = 16 :=
by
  sorry

end simplify_and_evaluate_l95_9589


namespace cos_value_l95_9536

theorem cos_value (α : ℝ) 
  (h1 : Real.sin (α + Real.pi / 12) = 1 / 3) : 
  Real.cos (α + 7 * Real.pi / 12) = -(1 + Real.sqrt 24) / 6 :=
sorry

end cos_value_l95_9536


namespace sum_of_odd_function_at_points_l95_9561

def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

theorem sum_of_odd_function_at_points (f : ℝ → ℝ) (h : is_odd_function f) : 
  f (-2) + f (-1) + f 0 + f 1 + f 2 = 0 :=
by
  sorry

end sum_of_odd_function_at_points_l95_9561


namespace books_bought_l95_9585

noncomputable def totalCost : ℤ :=
  let numFilms := 9
  let costFilm := 5
  let numCDs := 6
  let costCD := 3
  let costBook := 4
  let totalSpent := 79
  totalSpent - (numFilms * costFilm + numCDs * costCD)

theorem books_bought : ∃ B : ℤ, B * 4 = totalCost := by
  sorry

end books_bought_l95_9585


namespace sqrt_expression_l95_9508

noncomputable def a : ℝ := 5 - 3 * Real.sqrt 2
noncomputable def b : ℝ := 5 + 3 * Real.sqrt 2

theorem sqrt_expression : 
  Real.sqrt (a^2) + Real.sqrt (b^2) + 2 = 12 :=
by
  sorry

end sqrt_expression_l95_9508


namespace negation_of_proposition_l95_9588

variable (x : ℝ)
variable (p : Prop)

def proposition : Prop := ∀ x > 0, (x + 1) * Real.exp x > 1

theorem negation_of_proposition : ¬ proposition ↔ ∃ x > 0, (x + 1) * Real.exp x ≤ 1 :=
by
  sorry

end negation_of_proposition_l95_9588


namespace symmetry_center_on_line_l95_9539

def symmetry_center_curve :=
  ∃ θ : ℝ, (∃ x y : ℝ, (x = -1 + Real.cos θ ∧ y = 2 + Real.sin θ))

-- The main theorem to prove
theorem symmetry_center_on_line : 
  (∃ cx cy : ℝ, (symmetry_center_curve ∧ (cy = -2 * cx))) :=
sorry

end symmetry_center_on_line_l95_9539


namespace rectangle_length_to_width_ratio_l95_9547

-- Define the side length of the square
def s : ℝ := 1 -- Since we only need the ratio, the actual length does not matter

-- Define the length and width of the large rectangle
def length_of_large_rectangle : ℝ := 3 * s
def width_of_large_rectangle : ℝ := 3 * s

-- Define the dimensions of the small rectangle
def length_of_rectangle : ℝ := 3 * s
def width_of_rectangle : ℝ := s

-- Proving that the length of the rectangle is 3 times its width
theorem rectangle_length_to_width_ratio : length_of_rectangle = 3 * width_of_rectangle := 
by
  -- The proof is omitted
  sorry

end rectangle_length_to_width_ratio_l95_9547


namespace probability_of_6_heads_in_10_flips_l95_9500

theorem probability_of_6_heads_in_10_flips :
  let total_outcomes := 2 ^ 10
  let favorable_outcomes := Nat.choose 10 6
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = 210 / 1024 :=
by
  sorry

end probability_of_6_heads_in_10_flips_l95_9500


namespace find_ratio_b_c_l95_9567

variable {a b c A B C : Real}

theorem find_ratio_b_c
  (h1 : a * Real.sin A - b * Real.sin B = 4 * c * Real.sin C)
  (h2 : Real.cos A = -1 / 4) :
  b / c = 6 :=
sorry

end find_ratio_b_c_l95_9567


namespace laptop_weight_l95_9538

-- Defining the weights
variables (B U L P : ℝ)
-- Karen's tote weight
def K := 8

-- Conditions from the problem
axiom tote_eq_two_briefcase : K = 2 * B
axiom umbrella_eq_half_briefcase : U = B / 2
axiom full_briefcase_eq_double_tote : B + L + P + U = 2 * K
axiom papers_eq_sixth_full_briefcase : P = (B + L + P) / 6

-- Theorem stating the weight of Kevin's laptop is 7.67 pounds
theorem laptop_weight (hB : B = 4) (hU : U = 2) (hL : L = 7.67) : 
  L - K = -0.33 :=
by
  sorry

end laptop_weight_l95_9538


namespace factor_expression_l95_9559

theorem factor_expression (c : ℝ) : 180 * c ^ 2 + 36 * c = 36 * c * (5 * c + 1) := 
by
  sorry

end factor_expression_l95_9559


namespace parallelogram_area_150deg_10_20_eq_100sqrt3_l95_9531

noncomputable def parallelogram_area (angle: ℝ) (side1: ℝ) (side2: ℝ) : ℝ :=
  side1 * side2 * Real.sin angle

theorem parallelogram_area_150deg_10_20_eq_100sqrt3 :
  parallelogram_area (150 * Real.pi / 180) 10 20 = 100 * Real.sqrt 3 := by
  sorry

end parallelogram_area_150deg_10_20_eq_100sqrt3_l95_9531


namespace tomato_price_per_kilo_l95_9572

theorem tomato_price_per_kilo 
  (initial_money: ℝ) (money_left: ℝ)
  (potato_price_per_kilo: ℝ) (potato_kilos: ℝ)
  (cucumber_price_per_kilo: ℝ) (cucumber_kilos: ℝ)
  (banana_price_per_kilo: ℝ) (banana_kilos: ℝ)
  (tomato_kilos: ℝ)
  (spent_on_potatoes: initial_money - money_left = potato_price_per_kilo * potato_kilos)
  (spent_on_cucumbers: initial_money - money_left = cucumber_price_per_kilo * cucumber_kilos)
  (spent_on_bananas: initial_money - money_left = banana_price_per_kilo * banana_kilos)
  (total_spent: initial_money - money_left = 74)
  : (74 - (potato_price_per_kilo * potato_kilos + cucumber_price_per_kilo * cucumber_kilos + banana_price_per_kilo * banana_kilos)) / tomato_kilos = 3 := 
sorry

end tomato_price_per_kilo_l95_9572


namespace camel_height_in_feet_l95_9560

theorem camel_height_in_feet (h_ht_14 : ℕ) (ratio : ℕ) (inch_to_ft : ℕ) : ℕ :=
  let hare_height := 14
  let camel_height_in_inches := hare_height * 24
  let camel_height_in_feet := camel_height_in_inches / 12
  camel_height_in_feet
#print camel_height_in_feet

example : camel_height_in_feet 14 24 12 = 28 := by sorry

end camel_height_in_feet_l95_9560
