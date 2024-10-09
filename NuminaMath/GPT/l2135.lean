import Mathlib

namespace choir_members_minimum_l2135_213557

theorem choir_members_minimum (n : Nat) (h9 : n % 9 = 0) (h10 : n % 10 = 0) (h11 : n % 11 = 0) (h14 : n % 14 = 0) : n = 6930 :=
sorry

end choir_members_minimum_l2135_213557


namespace determine_d_l2135_213573

-- Given conditions
def equation (d x : ℝ) : Prop := 3 * (5 + d * x) = 15 * x + 15

-- Proof statement
theorem determine_d (d : ℝ) : (∀ x : ℝ, equation d x) ↔ d = 5 :=
by
  sorry

end determine_d_l2135_213573


namespace proof_of_equivalence_l2135_213597

variables (x y : ℝ)

def expression := 49 * x^2 - 36 * y^2
def optionD := (-6 * y + 7 * x) * (6 * y + 7 * x)

theorem proof_of_equivalence : expression x y = optionD x y := 
by sorry

end proof_of_equivalence_l2135_213597


namespace trigonometric_expression_value_l2135_213514

variable (θ : ℝ)

-- Conditions
axiom tan_theta_eq_two : Real.tan θ = 2

-- Theorem to prove
theorem trigonometric_expression_value : 
  Real.sin θ * Real.sin θ + 
  Real.sin θ * Real.cos θ - 
  2 * Real.cos θ * Real.cos θ = 4 / 5 := 
by
  sorry

end trigonometric_expression_value_l2135_213514


namespace Andrew_has_5_more_goats_than_twice_Adam_l2135_213539

-- Definitions based on conditions
def goats_Adam := 7
def goats_Ahmed := 13
def goats_Andrew := goats_Ahmed + 6
def twice_goats_Adam := 2 * goats_Adam

-- Theorem statement
theorem Andrew_has_5_more_goats_than_twice_Adam :
  goats_Andrew - twice_goats_Adam = 5 :=
by
  sorry

end Andrew_has_5_more_goats_than_twice_Adam_l2135_213539


namespace six_times_number_eq_132_l2135_213566

theorem six_times_number_eq_132 (x : ℕ) (h : x / 11 = 2) : 6 * x = 132 :=
sorry

end six_times_number_eq_132_l2135_213566


namespace guests_not_eating_brownies_ala_mode_l2135_213582

theorem guests_not_eating_brownies_ala_mode (total_brownies : ℕ) (eaten_brownies : ℕ) (eaten_scoops : ℕ)
    (scoops_per_serving : ℕ) (scoops_per_tub : ℕ) (tubs_eaten : ℕ) : 
    total_brownies = 32 → eaten_brownies = 28 → eaten_scoops = 48 → scoops_per_serving = 2 → scoops_per_tub = 8 → tubs_eaten = 6 → (eaten_scoops - eaten_brownies * scoops_per_serving) / scoops_per_serving = 4 :=
by
  intros
  sorry

end guests_not_eating_brownies_ala_mode_l2135_213582


namespace ratio_of_numbers_l2135_213520

theorem ratio_of_numbers (x y : ℕ) (h1 : x + y = 33) (h2 : x = 22) : y / x = 1 / 2 :=
by
  sorry

end ratio_of_numbers_l2135_213520


namespace johns_burritos_l2135_213594

-- Definitions based on conditions:
def initial_burritos : Nat := 3 * 20
def burritos_given_away : Nat := initial_burritos / 3
def burritos_after_giving_away : Nat := initial_burritos - burritos_given_away
def burritos_eaten : Nat := 3 * 10
def burritos_left : Nat := burritos_after_giving_away - burritos_eaten

-- The theorem we need to prove:
theorem johns_burritos : burritos_left = 10 := by
  sorry

end johns_burritos_l2135_213594


namespace door_X_is_inner_sanctuary_l2135_213507

  variable (X Y Z W : Prop)
  variable (A B C D E F G H : Prop)
  variable (is_knight : Prop → Prop)

  -- Each statement according to the conditions in the problem.
  variable (stmt_A : X)
  variable (stmt_B : Y ∨ Z)
  variable (stmt_C : is_knight A ∧ is_knight B)
  variable (stmt_D : X ∧ Y)
  variable (stmt_E : X ∧ Y)
  variable (stmt_F : is_knight D ∨ is_knight E)
  variable (stmt_G : is_knight C → is_knight F)
  variable (stmt_H : is_knight G ∧ is_knight H → is_knight A)

  theorem door_X_is_inner_sanctuary :
    is_knight A → is_knight B → is_knight C → is_knight D → is_knight E → is_knight F → is_knight G → is_knight H → X :=
  sorry
  
end door_X_is_inner_sanctuary_l2135_213507


namespace marks_in_physics_l2135_213533

-- Definitions of the variables
variables (P C M : ℕ)

-- Conditions
def condition1 : Prop := P + C + M = 210
def condition2 : Prop := P + M = 180
def condition3 : Prop := P + C = 140

-- The statement to prove
theorem marks_in_physics (h1 : condition1 P C M) (h2 : condition2 P M) (h3 : condition3 P C) : P = 110 :=
sorry

end marks_in_physics_l2135_213533


namespace evaluate_g_at_5_l2135_213554

def g (x : ℝ) : ℝ := 5 * x + 2

theorem evaluate_g_at_5 : g 5 = 27 := by
  sorry

end evaluate_g_at_5_l2135_213554


namespace odd_function_condition_l2135_213536

noncomputable def f (x a b : ℝ) : ℝ := x * |x + a| + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x : ℝ, f (-x) a b = -f x a b) ↔ a^2 + b^2 = 0 :=
by
  sorry

end odd_function_condition_l2135_213536


namespace inequality_am_gm_l2135_213595

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (a^2 + a * b + b^2) + b^3 / (b^2 + b * c + c^2) + c^3 / (c^2 + c * a + a^2)) ≥ (a + b + c) / 3 :=
by
  sorry

end inequality_am_gm_l2135_213595


namespace find_value_of_P_l2135_213518

def f (x : ℝ) : ℝ := (x^2 + x - 2)^2002 + 3

theorem find_value_of_P :
  f ( (Real.sqrt 5) / 2 - 1 / 2 ) = 4 := by
  sorry

end find_value_of_P_l2135_213518


namespace largest_d_for_range_l2135_213556

theorem largest_d_for_range (d : ℝ) : (∃ x : ℝ, x^2 - 6*x + d = 2) ↔ d ≤ 11 := 
by
  sorry

end largest_d_for_range_l2135_213556


namespace num_ways_to_queue_ABC_l2135_213593

-- Definitions for the problem
def num_people : ℕ := 5
def fixed_order_positions : ℕ := 3

-- Lean statement to prove the problem
theorem num_ways_to_queue_ABC (h : num_people = 5) (h_fop : fixed_order_positions = 3) : 
  (Nat.factorial num_people / Nat.factorial (num_people - fixed_order_positions)) * 1 = 20 := 
by
  sorry

end num_ways_to_queue_ABC_l2135_213593


namespace bear_pies_l2135_213560

-- Lean definitions model:

variables (v_M v_B u_M u_B : ℝ)
variables (M_raspberries B_raspberries : ℝ)
variables (P_M P_B : ℝ)

-- Given conditions
axiom v_B_eq_6v_M : v_B = 6 * v_M
axiom u_B_eq_3u_M : u_B = 3 * u_M
axiom B_raspberries_eq_2M_raspberries : B_raspberries = 2 * M_raspberries
axiom P_sum : P_B + P_M = 60
axiom P_B_eq_9P_M : P_B = 9 * P_M

-- The theorem to prove
theorem bear_pies : P_B = 54 :=
sorry

end bear_pies_l2135_213560


namespace minimum_value_function_l2135_213587

theorem minimum_value_function (x : ℝ) (h : x > -1) : 
  (∃ y, y = (x^2 + 7 * x + 10) / (x + 1) ∧ y ≥ 9) :=
sorry

end minimum_value_function_l2135_213587


namespace marek_sequence_sum_l2135_213528

theorem marek_sequence_sum (x : ℝ) :
  let a := x
  let b := (a + 4) / 4 - 4
  let c := (b + 4) / 4 - 4
  let d := (c + 4) / 4 - 4
  (a + 4) / 4 * 4 + (b + 4) / 4 * 4 + (c + 4) / 4 * 4 + (d + 4) / 4 * 4 = 80 →
  x = 38 :=
by
  sorry

end marek_sequence_sum_l2135_213528


namespace circle_radius_5_l2135_213540

theorem circle_radius_5 (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 10 * x + y^2 + 2 * y + c = 0) → 
  (∀ x y : ℝ, (x + 5)^2 + (y + 1)^2 = 25) → 
  c = 51 :=
sorry

end circle_radius_5_l2135_213540


namespace perpendicular_distance_l2135_213545

structure Vertex :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def S : Vertex := ⟨6, 0, 0⟩
def P : Vertex := ⟨0, 0, 0⟩
def Q : Vertex := ⟨0, 5, 0⟩
def R : Vertex := ⟨0, 0, 4⟩

noncomputable def distance_from_point_to_plane (S P Q R : Vertex) : ℝ := sorry

theorem perpendicular_distance (S P Q R : Vertex) (hS : S = ⟨6, 0, 0⟩) (hP : P = ⟨0, 0, 0⟩) (hQ : Q = ⟨0, 5, 0⟩) (hR : R = ⟨0, 0, 4⟩) :
  distance_from_point_to_plane S P Q R = 6 :=
  sorry

end perpendicular_distance_l2135_213545


namespace part1_part2_l2135_213580

namespace ProofProblem

noncomputable def f (x : ℝ) : ℝ := Real.tan ((x / 2) - (Real.pi / 3))

-- Part (1)
theorem part1 : f (5 * Real.pi / 2) = Real.sqrt 3 - 2 :=
by
  sorry

-- Part (2)
theorem part2 (k : ℤ) : { x : ℝ | f x ≤ Real.sqrt 3 } = 
  {x | ∃ (k : ℤ), 2 * k * Real.pi - Real.pi / 3 < x ∧ x ≤ 2 * k * Real.pi + 4 * Real.pi / 3} :=
by
  sorry

end ProofProblem

end part1_part2_l2135_213580


namespace radius_of_circle_l2135_213512

variable (r M N : ℝ)

theorem radius_of_circle (h1 : M = Real.pi * r^2) 
  (h2 : N = 2 * Real.pi * r) 
  (h3 : M / N = 15) : 
  r = 30 :=
sorry

end radius_of_circle_l2135_213512


namespace sufficient_but_not_necessary_condition_l2135_213570

theorem sufficient_but_not_necessary_condition (x : ℝ) : (x > 0 → |x| > 0) ∧ (¬ (|x| > 0 → x > 0)) :=
by 
  sorry

end sufficient_but_not_necessary_condition_l2135_213570


namespace simplified_value_l2135_213500

theorem simplified_value :
  (245^2 - 205^2) / 40 = 450 := by
  sorry

end simplified_value_l2135_213500


namespace Wayne_initially_collected_blocks_l2135_213577

-- Let's denote the initial blocks collected by Wayne as 'w'.
-- According to the problem:
-- - Wayne's father gave him 6 more blocks.
-- - He now has 15 blocks in total.
--
-- We need to prove that the initial number of blocks Wayne collected (w) is 9.

theorem Wayne_initially_collected_blocks : 
  ∃ w : ℕ, (w + 6 = 15) ↔ (w = 9) := by
  sorry

end Wayne_initially_collected_blocks_l2135_213577


namespace solve_math_problem_l2135_213515

noncomputable def math_problem : Prop :=
  ∃ (ω α β : ℂ), (ω^5 = 1) ∧ (ω ≠ 1) ∧ (α = ω + ω^2) ∧ (β = ω^3 + ω^4) ∧
  (∀ x : ℂ, x^2 + x + 3 = 0 → x = α ∨ x = β) ∧ (α + β = -1) ∧ (α * β = 3)

theorem solve_math_problem : math_problem := sorry

end solve_math_problem_l2135_213515


namespace flags_left_l2135_213576

theorem flags_left (interval circumference : ℕ) (total_flags : ℕ) (h1 : interval = 20) (h2 : circumference = 200) (h3 : total_flags = 12) : 
  total_flags - (circumference / interval) = 2 := 
by 
  -- Using the conditions h1, h2, h3
  sorry

end flags_left_l2135_213576


namespace alexander_spends_total_amount_l2135_213592

theorem alexander_spends_total_amount :
  (5 * 1) + (2 * 2) = 9 :=
by
  sorry

end alexander_spends_total_amount_l2135_213592


namespace car_travel_time_l2135_213564

noncomputable def travelTimes 
  (t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime : ℝ) : Prop :=
t_Ningi_Zipra = 0.80 * t_Ngapara_Zipra ∧
t_Ngapara_Zipra = 60 ∧
totalTravelTime = t_Ngapara_Zipra + t_Ningi_Zipra

theorem car_travel_time :
  ∃ t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime,
  travelTimes t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime ∧
  totalTravelTime = 108 :=
by
  sorry

end car_travel_time_l2135_213564


namespace loss_percentage_is_five_l2135_213598

/-- Definitions -/
def original_price : ℝ := 490
def sold_price : ℝ := 465.50
def loss_amount : ℝ := original_price - sold_price

/-- Theorem -/
theorem loss_percentage_is_five :
  (loss_amount / original_price) * 100 = 5 :=
by
  sorry

end loss_percentage_is_five_l2135_213598


namespace camp_cedar_counselors_l2135_213581

theorem camp_cedar_counselors (boys : ℕ) (girls : ℕ) 
(counselors_for_boys : ℕ) (counselors_for_girls : ℕ) 
(total_counselors : ℕ) 
(h1 : boys = 80)
(h2 : girls = 6 * boys - 40)
(h3 : counselors_for_boys = boys / 5)
(h4 : counselors_for_girls = (girls + 11) / 12)  -- +11 to account for rounding up
(h5 : total_counselors = counselors_for_boys + counselors_for_girls) : 
total_counselors = 53 :=
by
  sorry

end camp_cedar_counselors_l2135_213581


namespace geom_seq_product_l2135_213541

theorem geom_seq_product (a : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h_prod : a 5 * a 14 = 5) :
  a 8 * a 9 * a 10 * a 11 = 10 := 
sorry

end geom_seq_product_l2135_213541


namespace necessary_but_not_sufficient_condition_l2135_213509

noncomputable def necessary_but_not_sufficient (x : ℝ) : Prop :=
  (3 - x >= 0 → |x - 1| ≤ 2) ∧ ¬(3 - x >= 0 ↔ |x - 1| ≤ 2)

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  necessary_but_not_sufficient x :=
sorry

end necessary_but_not_sufficient_condition_l2135_213509


namespace total_crayons_l2135_213530

theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) 
      (h1 : crayons_per_child = 18) (h2 : num_children = 36) : 
        crayons_per_child * num_children = 648 := by
  sorry

end total_crayons_l2135_213530


namespace overlap_coordinates_l2135_213563

theorem overlap_coordinates :
  ∃ m n : ℝ, 
    (m + n = 6.8) ∧ 
    ((2 * (7 + m) / 2 - 3) = (3 + n) / 2) ∧ 
    ((2 * (7 + m) / 2 - 3) = - (m - 7) / 2) :=
by
  sorry

end overlap_coordinates_l2135_213563


namespace instantaneous_velocity_at_3_l2135_213550

-- Define the displacement function s(t)
def displacement (t : ℝ) : ℝ := 2 * t^3

-- Define the time at which we want to calculate the instantaneous velocity
def time : ℝ := 3

-- Define the expected instantaneous velocity at t=3
def expected_velocity : ℝ := 54

-- Define the derivative of the displacement function as the velocity function
noncomputable def velocity (t : ℝ) : ℝ := deriv displacement t

-- Theorem: Prove that the instantaneous velocity at t=3 is 54
theorem instantaneous_velocity_at_3 : velocity time = expected_velocity := 
by {
  -- Here the detailed proof should go, but we skip it with sorry
  sorry
}

end instantaneous_velocity_at_3_l2135_213550


namespace gcd_204_85_l2135_213567

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  have h1 : 204 = 2 * 85 + 34 := by rfl
  have h2 : 85 = 2 * 34 + 17 := by rfl
  have h3 : 34 = 2 * 17 := by rfl
  sorry

end gcd_204_85_l2135_213567


namespace area_between_circles_of_octagon_l2135_213517

-- Define some necessary geometric terms and functions
noncomputable def cot (θ : ℝ) : ℝ := 1 / Real.tan θ
noncomputable def csc (θ : ℝ) : ℝ := 1 / Real.sin θ

/-- The main theorem stating the area between the inscribed and circumscribed circles of a regular octagon is π. -/
theorem area_between_circles_of_octagon :
  let side_length := 2
  let θ := Real.pi / 8 -- 22.5 degrees in radians
  let apothem := cot θ
  let circum_radius := csc θ
  let area_between_circles := π * (circum_radius^2 - apothem^2)
  area_between_circles = π :=
by
  sorry

end area_between_circles_of_octagon_l2135_213517


namespace amount_amys_money_l2135_213508

def initial_dollars : ℝ := 2
def chores_payment : ℝ := 5 * 13
def birthday_gift : ℝ := 3
def total_after_gift : ℝ := initial_dollars + chores_payment + birthday_gift

def investment_percentage : ℝ := 0.20
def invested_amount : ℝ := investment_percentage * total_after_gift

def interest_rate : ℝ := 0.10
def interest_amount : ℝ := interest_rate * invested_amount
def total_investment : ℝ := invested_amount + interest_amount

def cost_of_toy : ℝ := 12
def remaining_after_toy : ℝ := total_after_gift - cost_of_toy

def grandparents_gift : ℝ := 2 * remaining_after_toy
def total_including_investment : ℝ := grandparents_gift + total_investment

def donation_percentage : ℝ := 0.25
def donated_amount : ℝ := donation_percentage * total_including_investment

def final_amount : ℝ := total_including_investment - donated_amount

theorem amount_amys_money :
  final_amount = 98.55 := by
  sorry

end amount_amys_money_l2135_213508


namespace find_original_price_l2135_213525

-- Defining constants and variables
def original_price (P : ℝ) : Prop :=
  let cost_after_repairs := P + 13000
  let selling_price := 66900
  let profit := selling_price - cost_after_repairs
  let profit_percent := profit / P * 100
  profit_percent = 21.636363636363637

theorem find_original_price : ∃ P : ℝ, original_price P :=
  by
  sorry

end find_original_price_l2135_213525


namespace find_n_l2135_213548

noncomputable def C (n : ℕ) : ℝ :=
  352 * (1 - 1 / 2 ^ n) / (1 - 1 / 2)

noncomputable def D (n : ℕ) : ℝ :=
  992 * (1 - 1 / (-2) ^ n) / (1 + 1 / 2)

theorem find_n (n : ℕ) (h : 1 ≤ n) : C n = D n ↔ n = 1 := by
  sorry

end find_n_l2135_213548


namespace smallest_largest_multiples_l2135_213529

theorem smallest_largest_multiples : 
  ∃ l g, l >= 10 ∧ l < 100 ∧ g >= 100 ∧ g < 1000 ∧
  (2 ∣ l) ∧ (3 ∣ l) ∧ (5 ∣ l) ∧ 
  (2 ∣ g) ∧ (3 ∣ g) ∧ (5 ∣ g) ∧
  (∀ n, n >= 10 ∧ n < 100 ∧ (2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n) → l ≤ n) ∧
  (∀ n, n >= 100 ∧ n < 1000 ∧ (2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n) → g >= n) ∧
  l = 30 ∧ g = 990 := 
by 
  sorry

end smallest_largest_multiples_l2135_213529


namespace mosquito_shadow_speed_l2135_213523

theorem mosquito_shadow_speed
  (v : ℝ) (t : ℝ) (h : ℝ) (cos_theta : ℝ) (v_shadow : ℝ)
  (hv : v = 0.5) (ht : t = 20) (hh : h = 6) (hcos_theta : cos_theta = 0.6) :
  v_shadow = 0 ∨ v_shadow = 0.8 :=
  sorry

end mosquito_shadow_speed_l2135_213523


namespace carson_gets_clawed_39_times_l2135_213588

-- Conditions
def number_of_wombats : ℕ := 9
def claws_per_wombat : ℕ := 4
def number_of_rheas : ℕ := 3
def claws_per_rhea : ℕ := 1

-- Theorem statement
theorem carson_gets_clawed_39_times :
  (number_of_wombats * claws_per_wombat + number_of_rheas * claws_per_rhea) = 39 :=
by
  sorry

end carson_gets_clawed_39_times_l2135_213588


namespace range_of_a_l2135_213589

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ 1 then 2^(|x - a|) else x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ f 1 a) ↔ (1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l2135_213589


namespace max_sum_consecutive_integers_less_360_l2135_213586

theorem max_sum_consecutive_integers_less_360 :
  ∃ n : ℤ, n * (n + 1) < 360 ∧ (n + (n + 1)) = 37 :=
by
  sorry

end max_sum_consecutive_integers_less_360_l2135_213586


namespace profit_percentage_is_50_l2135_213506

/--
Assumption:
- Initial machine cost: Rs 10,000
- Repair cost: Rs 5,000
- Transportation charges: Rs 1,000
- Selling price: Rs 24,000

To prove:
- The profit percentage is 50%
-/

def initial_cost : ℕ := 10000
def repair_cost : ℕ := 5000
def transportation_charges : ℕ := 1000
def selling_price : ℕ := 24000
def total_cost : ℕ := initial_cost + repair_cost + transportation_charges
def profit : ℕ := selling_price - total_cost

theorem profit_percentage_is_50 :
  (profit * 100) / total_cost = 50 :=
by
  -- proof goes here
  sorry

end profit_percentage_is_50_l2135_213506


namespace fruits_calculation_l2135_213569

structure FruitStatus :=
  (initial_picked  : ℝ)
  (initial_eaten  : ℝ)

def apples_status : FruitStatus :=
  { initial_picked := 7.0 + 3.0 + 5.0, initial_eaten := 6.0 + 2.0 }

def pears_status : FruitStatus :=
  { initial_picked := 0, initial_eaten := 4.0 + 3.0 }  -- number of pears picked is unknown, hence 0

def oranges_status : FruitStatus :=
  { initial_picked := 8.0, initial_eaten := 8.0 }

def cherries_status : FruitStatus :=
  { initial_picked := 4.0, initial_eaten := 4.0 }

theorem fruits_calculation :
  (apples_status.initial_picked - apples_status.initial_eaten = 7.0) ∧
  (pears_status.initial_picked - pears_status.initial_eaten = 0) ∧  -- cannot be determined in the problem statement
  (oranges_status.initial_picked - oranges_status.initial_eaten = 0) ∧
  (cherries_status.initial_picked - cherries_status.initial_eaten = 0) :=
by {
  sorry
}

end fruits_calculation_l2135_213569


namespace local_extrema_l2135_213519

noncomputable def f (x : ℝ) := 3 * x^3 - 9 * x^2 + 3

theorem local_extrema :
  (∃ x, x = 0 ∧ ∀ δ > 0, ∃ ε > 0, ∀ y, abs (y - x) < ε → f y ≤ f x) ∧
  (∃ x, x = 2 ∧ ∀ δ > 0, ∃ ε > 0, ∀ y, abs (y - x) < ε → f y ≥ f x) :=
sorry

end local_extrema_l2135_213519


namespace box_ratio_l2135_213553

theorem box_ratio (h : ℤ) (l : ℤ) (w : ℤ) (v : ℤ)
  (H_height : h = 12)
  (H_length : l = 3 * h)
  (H_volume : l * w * h = 3888)
  (H_length_multiple : ∃ m, l = m * w) :
  l / w = 4 := by
  sorry

end box_ratio_l2135_213553


namespace typing_difference_l2135_213562

theorem typing_difference (m : ℕ) (h1 : 10 * m - 8 * m = 10) : m = 5 :=
by
  sorry

end typing_difference_l2135_213562


namespace ticket_savings_l2135_213501

def single_ticket_cost : ℝ := 1.50
def package_cost : ℝ := 5.75
def num_tickets_needed : ℝ := 40

theorem ticket_savings :
  (num_tickets_needed * single_ticket_cost) - 
  ((num_tickets_needed / 5) * package_cost) = 14.00 :=
by
  sorry

end ticket_savings_l2135_213501


namespace simplify_expression_l2135_213538

theorem simplify_expression :
  (1 / (Real.log 3 / Real.log 12 + 1) + 1 / (Real.log 2 / Real.log 8 + 1) + 1 / (Real.log 9 / Real.log 18 + 1)) = 7 / 4 := 
sorry

end simplify_expression_l2135_213538


namespace initial_population_l2135_213524

theorem initial_population (P : ℝ) (h1 : P * 1.05 * 0.95 = 9975) : P = 10000 :=
by
  sorry

end initial_population_l2135_213524


namespace at_least_3_students_same_score_l2135_213537

-- Conditions
def initial_points : ℕ := 6
def correct_points : ℕ := 4
def incorrect_points : ℤ := -1
def num_questions : ℕ := 6
def num_students : ℕ := 51

-- Question
theorem at_least_3_students_same_score :
  ∃ score : ℤ, ∃ students_with_same_score : ℕ, students_with_same_score ≥ 3 :=
by
  sorry

end at_least_3_students_same_score_l2135_213537


namespace focus_of_parabola_x2_eq_neg_4y_l2135_213504

theorem focus_of_parabola_x2_eq_neg_4y :
  (∀ x y : ℝ, x^2 = -4 * y → focus = (0, -1)) := 
sorry

end focus_of_parabola_x2_eq_neg_4y_l2135_213504


namespace dodecagon_diagonals_l2135_213516

def D (n : ℕ) : ℕ := n * (n - 3) / 2

theorem dodecagon_diagonals : D 12 = 54 :=
by
  sorry

end dodecagon_diagonals_l2135_213516


namespace today_is_thursday_l2135_213599

-- Define the days of the week as an enumerated type
inductive DayOfWeek
| Monday : DayOfWeek
| Tuesday : DayOfWeek
| Wednesday : DayOfWeek
| Thursday : DayOfWeek
| Friday : DayOfWeek
| Saturday : DayOfWeek
| Sunday : DayOfWeek

open DayOfWeek

-- Define the conditions for the lion and the unicorn
def lion_truth (d: DayOfWeek) : Bool :=
match d with
| Monday | Tuesday | Wednesday => false
| _ => true

def unicorn_truth (d: DayOfWeek) : Bool :=
match d with
| Thursday | Friday | Saturday => false
| _ => true

-- The statement made by the lion and the unicorn
def lion_statement (today: DayOfWeek) : Bool :=
match today with
| Monday => lion_truth Sunday
| Tuesday => lion_truth Monday
| Wednesday => lion_truth Tuesday
| Thursday => lion_truth Wednesday
| Friday => lion_truth Thursday
| Saturday => lion_truth Friday
| Sunday => lion_truth Saturday

def unicorn_statement (today: DayOfWeek) : Bool :=
match today with
| Monday => unicorn_truth Sunday
| Tuesday => unicorn_truth Monday
| Wednesday => unicorn_truth Tuesday
| Thursday => unicorn_truth Wednesday
| Friday => unicorn_truth Thursday
| Saturday => unicorn_truth Friday
| Sunday => unicorn_truth Saturday

-- Main theorem to prove the current day
theorem today_is_thursday (d: DayOfWeek) (lion_said: lion_statement d = false) (unicorn_said: unicorn_statement d = false) : d = Thursday :=
by
  -- Placeholder for actual proof
  sorry

end today_is_thursday_l2135_213599


namespace alex_score_correct_l2135_213561

-- Conditions of the problem
def num_students := 20
def average_first_19 := 78
def new_average := 79

-- Alex's score calculation
def alex_score : ℕ :=
  let total_score_first_19 := 19 * average_first_19
  let total_score_all := num_students * new_average
  total_score_all - total_score_first_19

-- Problem statement: Prove Alex's score is 98
theorem alex_score_correct : alex_score = 98 := by
  sorry

end alex_score_correct_l2135_213561


namespace inequality_holds_l2135_213579

noncomputable def f : ℝ → ℝ := sorry

theorem inequality_holds (h_cont : Continuous f) (h_diff : Differentiable ℝ f)
  (h_ineq : ∀ x : ℝ, 2 * f x - (deriv f x) > 0) : 
  f 1 > (f 2) / (Real.exp 2) :=
sorry

end inequality_holds_l2135_213579


namespace cubic_sum_of_roots_l2135_213527

theorem cubic_sum_of_roots (r s a b : ℝ) (h1 : r + s = a) (h2 : r * s = b) : 
  r^3 + s^3 = a^3 - 3 * a * b :=
by
  sorry

end cubic_sum_of_roots_l2135_213527


namespace number_of_sequences_less_than_1969_l2135_213565

theorem number_of_sequences_less_than_1969 :
  (∃ S : ℕ → ℕ, (∀ n : ℕ, S (n + 1) > (S n) * (S n)) ∧ S 1969 = 1969) →
  ∃ N : ℕ, N < 1969 :=
sorry

end number_of_sequences_less_than_1969_l2135_213565


namespace part1_part2_l2135_213534

/-- Given that in triangle ABC, sides opposite to angles A, B, and C are a, b, and c respectively
     if 2a sin B = sqrt(3) b and A is an acute angle, then A = 60 degrees. -/
theorem part1 {a b : ℝ} {A B : ℝ} (h1 : 2 * a * Real.sin B = Real.sqrt 3 * b)
  (h2 : 0 < A ∧ A < Real.pi / 2) : A = Real.pi / 3 :=
sorry

/-- Given that in triangle ABC, sides opposite to angles A, B, and C are a, b, and c respectively
     if b = 5, c = sqrt(5), and cos C = 9 / 10, then a = 4 or a = 5. -/
theorem part2 {a b c : ℝ} {C : ℝ} (h1 : b = 5) (h2 : c = Real.sqrt 5) 
  (h3 : Real.cos C = 9 / 10) : a = 4 ∨ a = 5 :=
sorry

end part1_part2_l2135_213534


namespace cos_neg_45_eq_one_over_sqrt_two_l2135_213596

theorem cos_neg_45_eq_one_over_sqrt_two : Real.cos (-(45 : ℝ)) = 1 / Real.sqrt 2 := 
by
  sorry

end cos_neg_45_eq_one_over_sqrt_two_l2135_213596


namespace calculate_arithmetic_expression_l2135_213584

noncomputable def arithmetic_sum (a d l : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem calculate_arithmetic_expression :
  3 * (arithmetic_sum 71 2 99) = 3825 :=
by
  sorry

end calculate_arithmetic_expression_l2135_213584


namespace calculate_exponentiation_l2135_213571

theorem calculate_exponentiation : (64^(0.375) * 64^(0.125) = 8) :=
by sorry

end calculate_exponentiation_l2135_213571


namespace field_length_proof_l2135_213572

noncomputable def field_width (w : ℝ) : Prop := w > 0

def pond_side_length : ℝ := 7

def pond_area : ℝ := pond_side_length * pond_side_length

def field_length (w l : ℝ) : Prop := l = 2 * w

def field_area (w l : ℝ) : ℝ := l * w

def pond_area_condition (w l : ℝ) : Prop :=
  pond_area = (1 / 8) * field_area w l

theorem field_length_proof {w l : ℝ} (hw : field_width w)
                           (hl : field_length w l)
                           (hpond : pond_area_condition w l) :
  l = 28 := by
  sorry

end field_length_proof_l2135_213572


namespace maximum_BD_cyclic_quad_l2135_213526

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

end maximum_BD_cyclic_quad_l2135_213526


namespace num_possible_radii_l2135_213502

theorem num_possible_radii :
  ∃ S : Finset ℕ, S.card = 11 ∧ ∀ r ∈ S, (∃ k : ℕ, 150 = k * r) ∧ r ≠ 150 :=
by
  sorry

end num_possible_radii_l2135_213502


namespace find_original_number_l2135_213535

theorem find_original_number (x : ℤ) (h : (x + 5) % 23 = 0) : x = 18 :=
sorry

end find_original_number_l2135_213535


namespace tv_weight_calculations_l2135_213583

theorem tv_weight_calculations
    (w1 h1 r1 : ℕ) -- Represents Bill's TV dimensions and weight ratio
    (w2 h2 r2 : ℕ) -- Represents Bob's TV dimensions and weight ratio
    (w3 h3 r3 : ℕ) -- Represents Steve's TV dimensions and weight ratio
    (ounce_to_pound: ℕ) -- Represents the conversion factor from ounces to pounds
    (bill_tv_weight bob_tv_weight steve_tv_weight : ℕ) -- Computed weights in pounds
    (weight_diff: ℕ):
  (w1 * h1 * r1) / ounce_to_pound = bill_tv_weight → -- Bill's TV weight calculation
  (w2 * h2 * r2) / ounce_to_pound = bob_tv_weight → -- Bob's TV weight calculation
  (w3 * h3 * r3) / ounce_to_pound = steve_tv_weight → -- Steve's TV weight calculation
  steve_tv_weight > (bill_tv_weight + bob_tv_weight) → -- Steve's TV is the heaviest
  steve_tv_weight - (bill_tv_weight + bob_tv_weight) = weight_diff → -- weight difference calculation
  True := sorry

end tv_weight_calculations_l2135_213583


namespace roots_of_equation_l2135_213574

theorem roots_of_equation : ∀ x : ℝ, x^2 - 3 * x = 0 ↔ x = 0 ∨ x = 3 :=
by sorry

end roots_of_equation_l2135_213574


namespace share_pizza_l2135_213549

variable (Yoojung_slices Minyoung_slices total_slices : ℕ)
variable (Y : ℕ)

theorem share_pizza :
  Yoojung_slices = Y ∧
  Minyoung_slices = Y + 2 ∧
  total_slices = 10 ∧
  Yoojung_slices + Minyoung_slices = total_slices →
  Y = 4 :=
by
  sorry

end share_pizza_l2135_213549


namespace max_volume_day1_l2135_213513

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

end max_volume_day1_l2135_213513


namespace solve_for_y_l2135_213522

theorem solve_for_y (y : ℝ) (h : 1 / 4 - 1 / 5 = 4 / y) : y = 80 :=
by
  sorry

end solve_for_y_l2135_213522


namespace area_of_smaller_part_l2135_213559

noncomputable def average (a b : ℝ) : ℝ :=
  (a + b) / 2

theorem area_of_smaller_part:
  ∃ A B : ℝ, A + B = 900 ∧ (B - A) = (1 / 5) * average A B ∧ A = 405 :=
by
  sorry

end area_of_smaller_part_l2135_213559


namespace at_least_one_f_nonnegative_l2135_213585

theorem at_least_one_f_nonnegative 
  (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m * n > 1) : 
  (m^2 - m ≥ 0) ∨ (n^2 - n ≥ 0) :=
by sorry

end at_least_one_f_nonnegative_l2135_213585


namespace initial_bananas_l2135_213542

theorem initial_bananas (x B : ℕ) (h1 : 840 * x = B) (h2 : 420 * (x + 2) = B) : x = 2 :=
by
  sorry

end initial_bananas_l2135_213542


namespace product_of_solutions_eq_neg_ten_l2135_213511

theorem product_of_solutions_eq_neg_ten :
  (∃ x₁ x₂, -20 = -2 * x₁^2 - 6 * x₁ ∧ -20 = -2 * x₂^2 - 6 * x₂ ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -10) :=
by
  sorry

end product_of_solutions_eq_neg_ten_l2135_213511


namespace fraction_comparisons_l2135_213590

theorem fraction_comparisons :
  (1 / 8 : ℝ) * (3 / 7) < (1 / 8) ∧ 
  (9 / 8 : ℝ) * (1 / 5) > (9 / 8) * (1 / 8) ∧ 
  (2 / 3 : ℝ) < (2 / 3) / (6 / 11) := by
    sorry

end fraction_comparisons_l2135_213590


namespace find_least_positive_n_l2135_213503

theorem find_least_positive_n (n : ℕ) : 
  let m := 143
  m = 11 * 13 → 
  (3^5 ≡ 1 [MOD m^2]) →
  (3^39 ≡ 1 [MOD (13^2)]) →
  n = 195 :=
sorry

end find_least_positive_n_l2135_213503


namespace reflection_transformation_l2135_213575

structure Point (α : Type) :=
(x : α)
(y : α)

def reflect_x_axis (p : Point ℝ) : Point ℝ :=
  {x := p.x, y := -p.y}

def reflect_x_eq_3 (p : Point ℝ) : Point ℝ :=
  {x := 6 - p.x, y := p.y}

def D : Point ℝ := {x := 4, y := 1}

def D' := reflect_x_axis D

def D'' := reflect_x_eq_3 D'

theorem reflection_transformation :
  D'' = {x := 2, y := -1} :=
by
  -- We skip the proof here
  sorry

end reflection_transformation_l2135_213575


namespace range_distance_PQ_l2135_213531

noncomputable def point_P (α : ℝ) : ℝ × ℝ × ℝ := (3 * Real.cos α, 3 * Real.sin α, 1)
noncomputable def point_Q (β : ℝ) : ℝ × ℝ × ℝ := (2 * Real.cos β, 2 * Real.sin β, 1)

noncomputable def distance_PQ (α β : ℝ) : ℝ :=
  Real.sqrt ((3 * Real.cos α - 2 * Real.cos β)^2 +
             (3 * Real.sin α - 2 * Real.sin β)^2 +
             (1 - 1)^2)

theorem range_distance_PQ : 
  ∀ α β : ℝ, 1 ≤ distance_PQ α β ∧ distance_PQ α β ≤ 5 := 
by
  intros
  sorry

end range_distance_PQ_l2135_213531


namespace college_girls_count_l2135_213555

theorem college_girls_count (B G : ℕ) (h1 : B / G = 8 / 5) (h2 : B + G = 546) : G = 210 :=
by
  sorry

end college_girls_count_l2135_213555


namespace find_k_l2135_213558

theorem find_k (k l : ℝ) (C : ℝ × ℝ) (OC : ℝ) (A B D : ℝ × ℝ)
  (hC_coords : C = (0, 3))
  (hl_val : l = 3)
  (line_eqn : ∀ x, y = k * x + l)
  (intersect_eqn : ∀ x, y = 1 / x)
  (hA_coords : A = (1 / 6, 6))
  (hD_coords : D = (1 / 6, 6))
  (dist_ABC : dist A B = dist B C)
  (dist_BCD : dist B C = dist C D)
  (OC_val : OC = 3) :
  k = 18 := 
sorry

end find_k_l2135_213558


namespace hexagon_inscribed_in_square_area_l2135_213547

noncomputable def hexagon_area (side_length : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * side_length^2

theorem hexagon_inscribed_in_square_area (AB BC : ℝ) (BDEF_square : BDEF_is_square) (hAB : AB = 2) (hBC : BC = 2) :
  hexagon_area (2 * Real.sqrt 2) = 12 * Real.sqrt 3 :=
by
  sorry

-- Definitions to assume the necessary conditions in the theorem (placeholders)
-- Assuming a structure of BDEF_is_square to represent the property that BDEF is a square
structure BDEF_is_square :=
(square : Prop)

end hexagon_inscribed_in_square_area_l2135_213547


namespace find_number_l2135_213544

theorem find_number (n : ℝ) (h : n - (1004 / 20.08) = 4970) : n = 5020 := 
by {
  sorry
}

end find_number_l2135_213544


namespace remove_6_maximizes_probability_l2135_213578

def original_list : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

-- Define what it means to maximize the probability of pairs summing to 12
def maximize_probability (l : List Int) : Prop :=
  ∀ x y, x ≠ y → x ∈ l → y ∈ l → x + y = 12

-- Prove that removing 6 maximizes the probability that the sum of the two chosen numbers is 12
theorem remove_6_maximizes_probability :
  maximize_probability (original_list.erase 6) :=
sorry

end remove_6_maximizes_probability_l2135_213578


namespace find_a_l2135_213521

theorem find_a (a : ℝ) :
  (∃ x y : ℝ, x - 2 * y + 1 = 0 ∧ x + 3 * y - 1 = 0 ∧ ¬(∀ x y : ℝ, ax + 2 * y - 3 = 0)) →
  (∃ p q : ℝ, ax + 2 * q - 3 = 0 ∧ (a = -1 ∨ a = 2 / 3)) :=
by {
  sorry
}

end find_a_l2135_213521


namespace A_inter_B_A_subset_C_l2135_213543

namespace MathProof

def A := {x : ℝ | x^2 - 6*x + 8 ≤ 0 }
def B := {x : ℝ | (x - 1)/(x - 3) ≥ 0 }
def C (a : ℝ) := {x : ℝ | x^2 - (2*a + 4)*x + a^2 + 4*a ≤ 0 }

theorem A_inter_B : (A ∩ B) = {x : ℝ | 3 < x ∧ x ≤ 4} := sorry

theorem A_subset_C (a : ℝ) : (A ⊆ C a) ↔ (0 ≤ a ∧ a ≤ 2) := sorry

end MathProof

end A_inter_B_A_subset_C_l2135_213543


namespace simplify_expression_l2135_213591

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : 
  (18 * x^3) * (4 * x^2) * (1 / (2 * x)^3) = 9 * x^2 :=
by
  sorry

end simplify_expression_l2135_213591


namespace bananas_left_l2135_213568

-- Definitions based on conditions
def original_bananas : ℕ := 46
def bananas_removed : ℕ := 5

-- Statement of the problem using the definitions
theorem bananas_left : original_bananas - bananas_removed = 41 :=
by sorry

end bananas_left_l2135_213568


namespace derivative_at_one_third_l2135_213546

noncomputable def f (x : ℝ) : ℝ := Real.log (2 - 3 * x)

theorem derivative_at_one_third : (deriv f (1 / 3) = -3) := by
  sorry

end derivative_at_one_third_l2135_213546


namespace soda_cost_per_ounce_l2135_213552

/-- 
  Peter brought $2 with him, left with $0.50, and bought 6 ounces of soda.
  Prove that the cost per ounce of soda is $0.25.
-/
theorem soda_cost_per_ounce (initial_money final_money : ℝ) (amount_spent ounces_soda cost_per_ounce : ℝ)
  (h1 : initial_money = 2)
  (h2 : final_money = 0.5)
  (h3 : amount_spent = initial_money - final_money)
  (h4 : amount_spent = 1.5)
  (h5 : ounces_soda = 6)
  (h6 : cost_per_ounce = amount_spent / ounces_soda) :
  cost_per_ounce = 0.25 :=
by sorry

end soda_cost_per_ounce_l2135_213552


namespace number_of_impossible_d_l2135_213510

-- Define the problem parameters and conditions
def perimeter_diff (t s : ℕ) : ℕ := 3 * t - 4 * s
def side_diff (t s d : ℕ) : ℕ := t - s - d
def square_perimeter_positive (s : ℕ) : Prop := s > 0

-- Define the proof problem
theorem number_of_impossible_d (t s d : ℕ) (h1 : perimeter_diff t s = 1575) (h2 : side_diff t s d = 0) (h3 : square_perimeter_positive s) : 
    ∃ n, n = 525 ∧ ∀ d, d ≤ 525 → ¬ (3 * d > 1575) :=
    sorry

end number_of_impossible_d_l2135_213510


namespace yearly_return_of_1500_investment_l2135_213505

theorem yearly_return_of_1500_investment 
  (combined_return_percent : ℝ)
  (total_investment : ℕ)
  (return_500 : ℕ)
  (investment_500 : ℕ)
  (investment_1500 : ℕ) :
  combined_return_percent = 0.085 →
  total_investment = (investment_500 + investment_1500) →
  return_500 = (investment_500 * 7 / 100) →
  investment_500 = 500 →
  investment_1500 = 1500 →
  total_investment = 2000 →
  (return_500 + investment_1500 * combined_return_percent * 100) = (combined_return_percent * total_investment * 100) →
  ((investment_1500 * (9 : ℝ)) / 100) + return_500 = 0.085 * total_investment →
  (investment_1500 * 7 / 100) = investment_1500 →
  (investment_1500 / investment_1500) = (13500 / 1500) →
  (9 : ℝ) = 9 :=
sorry

end yearly_return_of_1500_investment_l2135_213505


namespace find_r_l2135_213532

-- Declaring the roots of the first polynomial
variables (a b m : ℝ)
-- Declaring the roots of the second polynomial
variables (p r : ℝ)

-- Assumptions based on the given conditions
def roots_of_first_eq : Prop :=
  a + b = m ∧ a * b = 3

def roots_of_second_eq : Prop :=
  ∃ (p : ℝ), (a^2 + 1/b) * (b^2 + 1/a) = r

-- The desired theorem
theorem find_r 
  (h1 : roots_of_first_eq a b m)
  (h2 : (a^2 + 1/b) * (b^2 + 1/a) = r) :
  r = 46/3 := by sorry

end find_r_l2135_213532


namespace eq_or_sum_zero_l2135_213551

variables (a b c d : ℝ)

theorem eq_or_sum_zero (h : (3 * a + 2 * b) / (2 * b + 4 * c) = (4 * c + 3 * d) / (3 * d + 3 * a)) :
  3 * a = 4 * c ∨ 3 * a + 3 * d + 2 * b + 4 * c = 0 :=
by sorry

end eq_or_sum_zero_l2135_213551
