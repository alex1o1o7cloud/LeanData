import Mathlib

namespace sum_abs_values_of_factors_l584_58498

theorem sum_abs_values_of_factors (a w c d : ℤ)
  (h1 : 6 * (x : ℤ)^2 + x - 12 = (a * x + w) * (c * x + d)) :
  abs a + abs w + abs c + abs d = 22 :=
sorry

end sum_abs_values_of_factors_l584_58498


namespace number_of_fence_panels_is_10_l584_58484

def metal_rods_per_sheet := 10
def metal_rods_per_beam := 4
def sheets_per_panel := 3
def beams_per_panel := 2
def total_metal_rods := 380

theorem number_of_fence_panels_is_10 :
  (total_metal_rods = 380) →
  (metal_rods_per_sheet = 10) →
  (metal_rods_per_beam = 4) →
  (sheets_per_panel = 3) →
  (beams_per_panel = 2) →
  380 / (3 * 10 + 2 * 4) = 10 := 
by 
  sorry

end number_of_fence_panels_is_10_l584_58484


namespace mother_picked_38_carrots_l584_58435

theorem mother_picked_38_carrots
  (haley_carrots : ℕ)
  (good_carrots : ℕ)
  (bad_carrots : ℕ)
  (total_carrots_picked : ℕ)
  (mother_carrots : ℕ)
  (h1 : haley_carrots = 39)
  (h2 : good_carrots = 64)
  (h3 : bad_carrots = 13)
  (h4 : total_carrots_picked = good_carrots + bad_carrots)
  (h5 : total_carrots_picked = haley_carrots + mother_carrots) :
  mother_carrots = 38 :=
by
  sorry

end mother_picked_38_carrots_l584_58435


namespace probability_of_suitcase_at_60th_position_expected_waiting_time_l584_58422

/-- Part (a):
    Prove that the probability that the businesspeople's 10th suitcase 
    appears exactly at the 60th position is equal to 
    (binom 59 9) / (binom 200 10) given 200 suitcases and 10 business people's suitcases,
    and a suitcase placed on the belt every 2 seconds. -/
theorem probability_of_suitcase_at_60th_position : 
  ∃ (P : ℚ), P = (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) :=
sorry

/-- Part (b):
    Prove that the expected waiting time for the businesspeople to get 
    their last suitcase is equal to 4020 / 11 seconds given 200 suitcases and 
    10 business people's suitcases, and a suitcase placed on the belt 
    every 2 seconds. -/
theorem expected_waiting_time : 
  ∃ (E : ℚ), E = 4020 / 11 :=
sorry

end probability_of_suitcase_at_60th_position_expected_waiting_time_l584_58422


namespace num_pos_divisors_36_l584_58431

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ d : ℕ, d > 0 → d ∣ 36 → d ∣ 9) :=
by
  sorry

end num_pos_divisors_36_l584_58431


namespace complementary_angles_decrease_86_percent_l584_58460

theorem complementary_angles_decrease_86_percent (x : ℝ) (h : 10 * x = 90) :
  let small_angle := 3 * x
  let increased_small_angle := small_angle * 1.2
  let large_angle := 7 * x
  let new_large_angle := 90 - increased_small_angle
  (new_large_angle / large_angle) * 100 = 91.4 :=
by
  sorry

end complementary_angles_decrease_86_percent_l584_58460


namespace evaporation_period_length_l584_58472

def initial_water_amount : ℝ := 10
def daily_evaporation_rate : ℝ := 0.0008
def percentage_evaporated : ℝ := 0.004  -- 0.4% expressed as a decimal

theorem evaporation_period_length :
  (percentage_evaporated * initial_water_amount) / daily_evaporation_rate = 50 := by
  sorry

end evaporation_period_length_l584_58472


namespace divisibility_problem_l584_58497

theorem divisibility_problem :
  (2^62 + 1) % (2^31 + 2^16 + 1) = 0 := 
sorry

end divisibility_problem_l584_58497


namespace ranking_possibilities_l584_58409

theorem ranking_possibilities (A B C D E : Type) : 
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 5 → (n ≠ 1 → n ≠ last)) →
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 5 → (n ≠ 1)) →
  ∃ (positions : Finset (List ℕ)),
    positions.card = 54 :=
by
  sorry

end ranking_possibilities_l584_58409


namespace caterpillar_length_difference_l584_58438

-- Define the lengths of the caterpillars
def green_caterpillar_length : ℝ := 3
def orange_caterpillar_length : ℝ := 1.17

-- State the theorem we need to prove
theorem caterpillar_length_difference :
  green_caterpillar_length - orange_caterpillar_length = 1.83 :=
by
  sorry

end caterpillar_length_difference_l584_58438


namespace abc_equivalence_l584_58457

theorem abc_equivalence (n : ℕ) (k : ℤ) (a b c : ℤ)
  (hn : 0 < n) (hk : k % 2 = 1)
  (h : a^n + k * b = b^n + k * c ∧ b^n + k * c = c^n + k * a) :
  a = b ∧ b = c := 
sorry

end abc_equivalence_l584_58457


namespace solution_set_l584_58453

open BigOperators

noncomputable def f (x : ℝ) := 2016^x + Real.log (Real.sqrt (x^2 + 1) + x) / Real.log 2016 - 2016^(-x)

theorem solution_set (x : ℝ) (h1 : ∀ x, f (-x) = -f (x)) (h2 : ∀ x1 x2, x1 < x2 → f (x1) < f (x2)) :
  x > -1 / 4 ↔ f (3 * x + 1) + f (x) > 0 := 
by
  sorry

end solution_set_l584_58453


namespace sequence_eventually_periodic_modulo_l584_58430

noncomputable def a_n (n : ℕ) : ℕ :=
  n ^ n + (n - 1) ^ (n + 1)

theorem sequence_eventually_periodic_modulo (m : ℕ) (hm : m > 0) : ∃ K s : ℕ, ∀ k : ℕ, (K ≤ k → a_n (k) % m = a_n (k + s) % m) :=
sorry

end sequence_eventually_periodic_modulo_l584_58430


namespace problem_proof_l584_58408

noncomputable def triangle_expression (a b c : ℝ) (A B C : ℝ) : ℝ :=
  b^2 * (Real.cos (C / 2))^2 + c^2 * (Real.cos (B / 2))^2 + 
  2 * b * c * Real.cos (B / 2) * Real.cos (C / 2) * Real.sin (A / 2)

theorem problem_proof (a b c A B C : ℝ) (h1 : a + b + c = 16) : 
  triangle_expression a b c A B C = 64 := 
sorry

end problem_proof_l584_58408


namespace rowing_distance_l584_58480
-- Lean 4 Statement

theorem rowing_distance (v_m v_t D : ℝ) 
  (h1 : D = v_m + v_t)
  (h2 : 30 = 10 * (v_m - v_t))
  (h3 : 30 = 6 * (v_m + v_t)) :
  D = 5 :=
by sorry

end rowing_distance_l584_58480


namespace inequality_system_integer_solutions_l584_58450

theorem inequality_system_integer_solutions :
  { x : ℤ | 5 * x + 1 > 3 * (x - 1) ∧ (x - 1) / 2 ≥ 2 * x - 4 } = {-1, 0, 1, 2} := by
  sorry

end inequality_system_integer_solutions_l584_58450


namespace tom_paid_correct_amount_l584_58465

-- Define the conditions given in the problem
def kg_apples : ℕ := 8
def rate_apples : ℕ := 70
def kg_mangoes : ℕ := 9
def rate_mangoes : ℕ := 45

-- Define the cost calculations
def cost_apples : ℕ := kg_apples * rate_apples
def cost_mangoes : ℕ := kg_mangoes * rate_mangoes
def total_amount : ℕ := cost_apples + cost_mangoes

-- The proof problem statement
theorem tom_paid_correct_amount : total_amount = 965 :=
by
  -- The proof steps are omitted and replaced with sorry
  sorry

end tom_paid_correct_amount_l584_58465


namespace total_splash_width_l584_58420

def pebbles : ℚ := 1/5
def rocks : ℚ := 2/5
def boulders : ℚ := 7/5
def mini_boulders : ℚ := 4/5
def large_pebbles : ℚ := 3/5

def num_pebbles : ℚ := 10
def num_rocks : ℚ := 5
def num_boulders : ℚ := 4
def num_mini_boulders : ℚ := 3
def num_large_pebbles : ℚ := 7

theorem total_splash_width : 
  num_pebbles * pebbles + 
  num_rocks * rocks + 
  num_boulders * boulders + 
  num_mini_boulders * mini_boulders + 
  num_large_pebbles * large_pebbles = 16.2 := by
  sorry

end total_splash_width_l584_58420


namespace no_real_solution_l584_58411

theorem no_real_solution (x : ℝ) : ¬ ∃ x : ℝ, (x - 5*x + 12)^2 + 1 = -|x| := 
sorry

end no_real_solution_l584_58411


namespace contrapositive_example_l584_58418

theorem contrapositive_example (α : ℝ) : (α = Real.pi / 3 → Real.cos α = 1 / 2) → (Real.cos α ≠ 1 / 2 → α ≠ Real.pi / 3) :=
by
  sorry

end contrapositive_example_l584_58418


namespace sahil_selling_price_l584_58492

-- Definitions based on the conditions
def purchase_price : ℕ := 10000
def repair_costs : ℕ := 5000
def transportation_charges : ℕ := 1000
def profit_percentage : ℕ := 50

def total_cost : ℕ := purchase_price + repair_costs + transportation_charges
def profit : ℕ := (profit_percentage * total_cost) / 100
def selling_price : ℕ := total_cost + profit

-- The theorem we need to prove
theorem sahil_selling_price : selling_price = 24000 :=
by
  sorry

end sahil_selling_price_l584_58492


namespace range_of_m_l584_58436

noncomputable def f (x : ℝ) := |x - 3| - 2
noncomputable def g (x : ℝ) := -|x + 1| + 4

theorem range_of_m (m : ℝ) : (∀ x, f x - g x ≥ m + 1) ↔ m ≤ -3 :=
by
  sorry

end range_of_m_l584_58436


namespace seashells_left_l584_58445

-- Definitions based on conditions
def initial_seashells : ℕ := 35
def seashells_given_away : ℕ := 18

-- Theorem stating the proof problem
theorem seashells_left (initial_seashells seashells_given_away : ℕ) : initial_seashells - seashells_given_away = 17 := 
    by
        sorry

end seashells_left_l584_58445


namespace rectangle_area_invariant_l584_58459

theorem rectangle_area_invariant
    (x y : ℕ)
    (h1 : x * y = (x + 3) * (y - 1))
    (h2 : x * y = (x - 3) * (y + 2)) :
    x * y = 15 :=
by sorry

end rectangle_area_invariant_l584_58459


namespace inequality_product_geq_two_power_n_equality_condition_l584_58427

open Real BigOperators

noncomputable def is_solution (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i ∧ a i = 1

theorem inequality_product_geq_two_power_n (a : ℕ → ℝ) (n : ℕ)
  (h1 : ( ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i))
  (h2 : ∑ i in Finset.range n, a (i + 1) = n) :
  (∏ i in Finset.range n, (1 + 1 / a (i + 1))) ≥ 2 ^ n :=
sorry

theorem equality_condition (a : ℕ → ℝ) (n : ℕ)
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i)
  (h2 : ∑ i in Finset.range n, a (i + 1) = n):
  (∏ i in Finset.range n, (1 + 1 / a (i + 1))) = 2 ^ n ↔ is_solution a n :=
sorry

end inequality_product_geq_two_power_n_equality_condition_l584_58427


namespace correct_insights_l584_58479

def insight1 := ∀ connections : Type, (∃ journey : connections → Prop, ∀ (x : connections), ¬journey x)
def insight2 := ∀ connections : Type, (∃ (beneficial : connections → Prop), ∀ (x : connections), beneficial x → True)
def insight3 := ∀ connections : Type, (∃ (accidental : connections → Prop), ∀ (x : connections), accidental x → False)
def insight4 := ∀ connections : Type, (∃ (conditional : connections → Prop), ∀ (x : connections), conditional x → True)

theorem correct_insights : ¬ insight1 ∧ insight2 ∧ ¬ insight3 ∧ insight4 :=
by sorry

end correct_insights_l584_58479


namespace exists_1990_gon_with_conditions_l584_58477

/-- A polygon structure with side lengths and properties to check equality of interior angles and side lengths -/
structure Polygon (n : ℕ) :=
  (sides : Fin n → ℕ)
  (angles_equal : Prop)

/-- Given conditions -/
def condition_1 (P : Polygon 1990) : Prop := P.angles_equal
def condition_2 (P : Polygon 1990) : Prop :=
  ∃ (σ : Fin 1990 → Fin 1990), ∀ i, P.sides i = (σ i + 1)^2

/-- The main theorem to be proven -/
theorem exists_1990_gon_with_conditions :
  ∃ P : Polygon 1990, condition_1 P ∧ condition_2 P :=
sorry

end exists_1990_gon_with_conditions_l584_58477


namespace largest_square_plot_size_l584_58485

def field_side_length := 50
def available_fence_length := 4000

theorem largest_square_plot_size :
  ∃ (s : ℝ), (0 < s) ∧ (s ≤ field_side_length) ∧ 
  (100 * (field_side_length - s) = available_fence_length) →
  s = 10 :=
by
  sorry

end largest_square_plot_size_l584_58485


namespace average_increase_l584_58495

def scores : List ℕ := [92, 85, 90, 95]

def initial_average (s : List ℕ) : ℚ := (s.take 3).sum / 3

def new_average (s : List ℕ) : ℚ := s.sum / s.length

theorem average_increase :
  initial_average scores + 1.5 = new_average scores := 
by
  sorry

end average_increase_l584_58495


namespace no_club_member_is_fraternity_member_l584_58488

variable (Student : Type) (isHonest : Student → Prop) 
                       (isFraternityMember : Student → Prop) 
                       (isClubMember : Student → Prop)

axiom some_students_not_honest : ∃ x : Student, ¬ isHonest x
axiom all_frats_honest : ∀ y : Student, isFraternityMember y → isHonest y
axiom no_clubs_honest : ∀ z : Student, isClubMember z → ¬ isHonest z

theorem no_club_member_is_fraternity_member : ∀ w : Student, isClubMember w → ¬ isFraternityMember w :=
by sorry

end no_club_member_is_fraternity_member_l584_58488


namespace students_participated_in_both_l584_58412

theorem students_participated_in_both (total_students volleyball track field no_participation both: ℕ) 
  (h1 : total_students = 45) 
  (h2 : volleyball = 12) 
  (h3 : track = 20) 
  (h4 : no_participation = 19) 
  (h5 : both = volleyball + track - (total_students - no_participation)) 
  : both = 6 :=
by
  sorry

end students_participated_in_both_l584_58412


namespace product_of_integers_l584_58454

theorem product_of_integers (A B C D : ℚ)
  (h1 : A + B + C + D = 100)
  (h2 : A + 5 = B - 5)
  (h3 : A + 5 = 2 * C)
  (h4 : A + 5 = D / 2) :
  A * B * C * D = 1517000000 / 6561 := by
  sorry

end product_of_integers_l584_58454


namespace wendi_owns_rabbits_l584_58401

/-- Wendi's plot of land is 200 feet by 900 feet. -/
def area_land_in_feet : ℕ := 200 * 900

/-- One rabbit can eat enough grass to clear ten square yards of lawn area per day. -/
def rabbit_clear_per_day : ℕ := 10

/-- It would take 20 days for all of Wendi's rabbits to clear all the grass off of her grassland property. -/
def days_to_clear : ℕ := 20

/-- Convert feet to yards (3 feet in a yard). -/
def feet_to_yards (feet : ℕ) : ℕ := feet / 3

/-- Calculate the total area of the land in square yards. -/
def area_land_in_yards : ℕ := (feet_to_yards 200) * (feet_to_yards 900)

theorem wendi_owns_rabbits (total_area : ℕ := area_land_in_yards)
                            (clear_area_per_rabbit : ℕ := rabbit_clear_per_day * days_to_clear) :
  total_area / clear_area_per_rabbit = 100 := 
sorry

end wendi_owns_rabbits_l584_58401


namespace evaluate_fraction_eq_10_pow_10_l584_58405

noncomputable def evaluate_fraction (a b c : ℕ) : ℕ :=
  (a ^ 20) / ((a * b) ^ 10)

theorem evaluate_fraction_eq_10_pow_10 :
  evaluate_fraction 30 3 10 = 10 ^ 10 :=
by
  -- We define what is given and manipulate it directly to form a proof outline.
  sorry

end evaluate_fraction_eq_10_pow_10_l584_58405


namespace show_R_r_eq_l584_58469

variables {a b c R r : ℝ}

-- Conditions
def sides_of_triangle (a b c : ℝ) : Prop :=
a + b > c ∧ a + c > b ∧ b + c > a

def circumradius (R a b c : ℝ) (Δ : ℝ) : Prop :=
R = a * b * c / (4 * Δ)

def inradius (r Δ : ℝ) (s : ℝ) : Prop :=
r = Δ / s

theorem show_R_r_eq (a b c : ℝ) (R r : ℝ) (Δ : ℝ) (s : ℝ) (h_sides : sides_of_triangle a b c)
  (h_circumradius : circumradius R a b c Δ)
  (h_inradius : inradius r Δ s)
  (h_semiperimeter : s = (a + b + c) / 2) :
  R * r = a * b * c / (2 * (a + b + c)) :=
sorry

end show_R_r_eq_l584_58469


namespace total_number_of_baseball_cards_l584_58444

def baseball_cards_total : Nat :=
  let carlos := 20
  let matias := carlos - 6
  let jorge := matias
  carlos + matias + jorge
   
theorem total_number_of_baseball_cards :
  baseball_cards_total = 48 :=
by
  rfl

end total_number_of_baseball_cards_l584_58444


namespace goshawk_eurasian_reserve_hawks_l584_58467

variable (H P : ℝ)

theorem goshawk_eurasian_reserve_hawks :
  P = 100 ∧
  (35 / 100) * P = P - (H + (40 / 100) * (P - H) + (25 / 100) * (40 / 100) * (P - H))
    → H = 25 :=
by sorry

end goshawk_eurasian_reserve_hawks_l584_58467


namespace find_sum_of_variables_l584_58475

variables (a b c d : ℤ)

theorem find_sum_of_variables
    (h1 : a - b + c = 7)
    (h2 : b - c + d = 8)
    (h3 : c - d + a = 4)
    (h4 : d - a + b = 3)
    (h5 : a + b + c - d = 10) :
    a + b + c + d = 16 := 
sorry

end find_sum_of_variables_l584_58475


namespace number_of_subsets_l584_58458

theorem number_of_subsets (P : Finset ℤ) (h : P = {-1, 0, 1}) : P.powerset.card = 8 := 
by
  rw [h]
  sorry

end number_of_subsets_l584_58458


namespace original_cost_price_l584_58446

theorem original_cost_price (S P C : ℝ) (h1 : S = 260) (h2 : S = 1.20 * C) : C = 216.67 := sorry

end original_cost_price_l584_58446


namespace apple_tree_total_apples_l584_58426

def firstYear : ℕ := 40
def secondYear : ℕ := 8 + 2 * firstYear
def thirdYear : ℕ := secondYear - (secondYear / 4)

theorem apple_tree_total_apples (FirstYear := firstYear) (SecondYear := secondYear) (ThirdYear := thirdYear) :
  FirstYear + SecondYear + ThirdYear = 194 :=
by 
  sorry

end apple_tree_total_apples_l584_58426


namespace part1_part2_l584_58425

-- Define the operation * on integers
def op (a b : ℤ) : ℤ := a^2 - b + a * b

-- Prove that 2 * 3 = 7 given the defined operation
theorem part1 : op 2 3 = 7 := 
sorry

-- Prove that (-2) * (op 2 (-3)) = 1 given the defined operation
theorem part2 : op (-2) (op 2 (-3)) = 1 := 
sorry

end part1_part2_l584_58425


namespace part1_part2_part3_l584_58429

-- Define conditions
variables (n : ℕ) (h₁ : 5 ≤ n)

-- Problem part (1): Define p_n and prove its value
def p_n (n : ℕ) := (10 * n) / ((n + 5) * (n + 4))

-- Problem part (2): Define EX and prove its value for n = 5
def EX : ℚ := 5 / 3

-- Problem part (3): Prove n = 20 maximizes P
def P (n : ℕ) := 3 * ((p_n n) ^ 3 - 2 * (p_n n) ^ 2 + (p_n n))
def n_max := 20

-- Making the proof skeletons for clarity, filling in later
theorem part1 : p_n n = 10 * n / ((n + 5) * (n + 4)) :=
sorry

theorem part2 (h₂ : n = 5) : EX = 5 / 3 :=
sorry

theorem part3 : n_max = 20 :=
sorry

end part1_part2_part3_l584_58429


namespace option_b_results_in_2x_cubed_l584_58448

variable (x : ℝ)

theorem option_b_results_in_2x_cubed : |x^3| + x^3 = 2 * x^3 := 
sorry

end option_b_results_in_2x_cubed_l584_58448


namespace fewer_mpg_in_city_l584_58440

def city_miles : ℕ := 336
def highway_miles : ℕ := 462
def city_mpg : ℕ := 24

def tank_size : ℕ := city_miles / city_mpg
def highway_mpg : ℕ := highway_miles / tank_size

theorem fewer_mpg_in_city : highway_mpg - city_mpg = 9 :=
by
  sorry

end fewer_mpg_in_city_l584_58440


namespace solve_equation_l584_58441

theorem solve_equation (x : ℝ) : 2 * x - 1 = 3 * x + 3 → x = -4 :=
by
  intro h
  sorry

end solve_equation_l584_58441


namespace inequality_proof_l584_58439

theorem inequality_proof (a b c : ℝ) (ha1 : 0 ≤ a) (ha2 : a ≤ 1) (hb1 : 0 ≤ b) (hb2 : b ≤ 1) (hc1 : 0 ≤ c) (hc2 : c ≤ 1) :
  (a / (b + c + 1) + b / (a + c + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1) :=
by
  sorry

end inequality_proof_l584_58439


namespace problem_statement_l584_58414

variable (U M N : Set ℕ)

theorem problem_statement (hU : U = {1, 2, 3, 4, 5})
                         (hM : M = {1, 4})
                         (hN : N = {2, 5}) :
                         N ∪ (U \ M) = {2, 3, 5} :=
by sorry

end problem_statement_l584_58414


namespace other_x_intercept_l584_58462

-- Definition of the two foci
def f1 : ℝ × ℝ := (0, 2)
def f2 : ℝ × ℝ := (3, 0)

-- One x-intercept is given as
def intercept1 : ℝ × ℝ := (0, 0)

-- We need to prove the other x-intercept is (15/4, 0)
theorem other_x_intercept : ∃ x : ℝ, (x, 0) = (15/4, 0) ∧
  (dist (x, 0) f1 + dist (x, 0) f2 = dist intercept1 f1 + dist intercept1 f2) :=
by
  sorry

end other_x_intercept_l584_58462


namespace parallelogram_sides_l584_58483

theorem parallelogram_sides (a b : ℝ)
  (h1 : 2 * (a + b) = 32)
  (h2 : b - a = 8) :
  a = 4 ∧ b = 12 :=
by
  -- Proof is to be provided
  sorry

end parallelogram_sides_l584_58483


namespace bathing_suits_per_model_l584_58434

def models : ℕ := 6
def evening_wear_sets_per_model : ℕ := 3
def time_per_trip_minutes : ℕ := 2
def total_show_time_minutes : ℕ := 60

theorem bathing_suits_per_model : (total_show_time_minutes - (models * evening_wear_sets_per_model * time_per_trip_minutes)) / (time_per_trip_minutes * models) = 2 :=
by
  sorry

end bathing_suits_per_model_l584_58434


namespace speed_of_second_person_l584_58468

-- Definitions based on the conditions
def speed_person1 := 70 -- km/hr
def distance_AB := 600 -- km

def time_traveled := 4 -- hours (from 10 am to 2 pm)

-- The goal is to prove that the speed of the second person is 80 km/hr
theorem speed_of_second_person :
  (distance_AB - speed_person1 * time_traveled) / time_traveled = 80 := 
by 
  sorry

end speed_of_second_person_l584_58468


namespace find_speed_of_A_l584_58433

noncomputable def speed_of_A_is_7_5 (a : ℝ) : Prop :=
  -- Conditions
  ∃ (b : ℝ), b = a + 5 ∧ 
  (60 / a = 100 / b) → 
  -- Conclusion
  a = 7.5

-- Statement in Lean 4
theorem find_speed_of_A (a : ℝ) (h : speed_of_A_is_7_5 a) : a = 7.5 :=
  sorry

end find_speed_of_A_l584_58433


namespace solution_set_of_inequality_l584_58415

theorem solution_set_of_inequality (x : ℝ) : 
  (3*x^2 - 4*x + 7 > 0) → (1 - 2*x) / (3*x^2 - 4*x + 7) ≥ 0 ↔ x ≤ 1 / 2 :=
by
  intros
  sorry

end solution_set_of_inequality_l584_58415


namespace quadratic_roots_l584_58419

theorem quadratic_roots:
  ∀ x : ℝ, x^2 - 1 = 0 ↔ (x = -1 ∨ x = 1) :=
by
  sorry

end quadratic_roots_l584_58419


namespace given_expression_simplifies_to_l584_58499

-- Given conditions: a ≠ ±1, a ≠ 0, b ≠ -1, b ≠ 0
variable (a b : ℝ)
variable (ha1 : a ≠ 1)
variable (ha2 : a ≠ -1)
variable (ha3 : a ≠ 0)
variable (hb1 : b ≠ 0)
variable (hb2 : b ≠ -1)

theorem given_expression_simplifies_to (h1 : a ≠ 1) (h2 : a ≠ -1) (h3 : a ≠ 0) (h4 : b ≠ 0) (h5 : b ≠ -1) :
    (a * b^(2/3) - b^(2/3) - a + 1) / ((1 - a^(1/3)) * ((a^(1/3) + 1)^2 - a^(1/3)) * (b^(1/3) + 1))
  + (a * b)^(1/3) * (1/a^(1/3) + 1/b^(1/3)) = 1 + a^(1/3) := by
  sorry

end given_expression_simplifies_to_l584_58499


namespace probability_exactly_five_shots_expected_shots_to_hit_all_l584_58413

-- Part (a)
theorem probability_exactly_five_shots
  (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  (∃ t₁ t₂ t₃ : ℕ, t₁ ≠ t₂ ∧ t₁ ≠ t₃ ∧ t₂ ≠ t₃ ∧ t₁ + t₂ + t₃ = 5) →
  6 * p ^ 3 * (1 - p) ^ 2 = 6 * p ^ 3 * (1 - p) ^ 2 :=
by sorry

-- Part (b)
theorem expected_shots_to_hit_all
  (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  (∀ t: ℕ, (t * p * (1 - p)^(t-1)) = 1/p) →
  3 * (1/p) = 3 / p :=
by sorry

end probability_exactly_five_shots_expected_shots_to_hit_all_l584_58413


namespace miles_per_gallon_l584_58443

theorem miles_per_gallon (miles gallons : ℝ) (h : miles = 100 ∧ gallons = 5) : miles / gallons = 20 := by
  cases h with
  | intro miles_eq gallons_eq =>
    rw [miles_eq, gallons_eq]
    norm_num

end miles_per_gallon_l584_58443


namespace miles_driven_on_tuesday_l584_58496

-- Define the conditions given in the problem
theorem miles_driven_on_tuesday (T : ℕ) (h_avg : (12 + T + 21) / 3 = 17) :
  T = 18 :=
by
  -- We state what we want to prove, but we leave the proof with sorry
  sorry

end miles_driven_on_tuesday_l584_58496


namespace circular_garden_area_l584_58481

open Real

theorem circular_garden_area (r : ℝ) (h₁ : r = 8)
      (h₂ : 2 * π * r = (1 / 4) * π * r ^ 2) :
  π * r ^ 2 = 64 * π :=
by
  -- The proof will go here
  sorry

end circular_garden_area_l584_58481


namespace rate_at_which_bowls_were_bought_l584_58487

theorem rate_at_which_bowls_were_bought 
    (total_bowls : ℕ) (sold_bowls : ℕ) (price_per_sold_bowl : ℝ) (remaining_bowls : ℕ) (percentage_gain : ℝ) 
    (total_bowls_eq : total_bowls = 115) 
    (sold_bowls_eq : sold_bowls = 104) 
    (price_per_sold_bowl_eq : price_per_sold_bowl = 20) 
    (remaining_bowls_eq : remaining_bowls = 11) 
    (percentage_gain_eq : percentage_gain = 0.4830917874396135) 
  : ∃ (R : ℝ), R = 18 :=
  sorry

end rate_at_which_bowls_were_bought_l584_58487


namespace project_work_time_ratio_l584_58424

theorem project_work_time_ratio (A B C : ℕ) (h_ratio : A = x ∧ B = 2 * x ∧ C = 3 * x) (h_total : A + B + C = 120) : 
  (C - A = 40) :=
by
  sorry

end project_work_time_ratio_l584_58424


namespace rope_lengths_l584_58406

theorem rope_lengths (joey_len chad_len mandy_len : ℝ) (h1 : joey_len = 56) 
  (h2 : 8 / 3 = joey_len / chad_len) (h3 : 5 / 2 = chad_len / mandy_len) : 
  chad_len = 21 ∧ mandy_len = 8.4 :=
by
  sorry

end rope_lengths_l584_58406


namespace first_group_number_l584_58421

variable (x : ℕ)

def number_of_first_group :=
  x = 6

theorem first_group_number (H1 : ∀ k : ℕ, k = 8 * 15 + x)
                          (H2 : k = 126) : 
                          number_of_first_group x :=
by
  sorry

end first_group_number_l584_58421


namespace four_squares_cover_larger_square_l584_58491

structure Square :=
  (side : ℝ) (h_positive : side > 0)

theorem four_squares_cover_larger_square (large small : Square) 
  (h_side_relation: large.side = 2 * small.side) : 
  large.side^2 = 4 * small.side^2 :=
by
  sorry

end four_squares_cover_larger_square_l584_58491


namespace find_fixed_monthly_fee_l584_58437

noncomputable def fixed_monthly_fee (f h : ℝ) (february_bill march_bill : ℝ) : Prop :=
  (f + h = february_bill) ∧ (f + 3 * h = march_bill)

theorem find_fixed_monthly_fee (h : ℝ):
  fixed_monthly_fee 13.44 h 20.72 35.28 :=
by 
  sorry

end find_fixed_monthly_fee_l584_58437


namespace donation_calculation_l584_58447

/-- Patricia's initial hair length -/
def initial_length : ℕ := 14

/-- Patricia's hair growth -/
def growth_length : ℕ := 21

/-- Desired remaining hair length after donation -/
def remaining_length : ℕ := 12

/-- Calculate the donation length -/
def donation_length (L G R : ℕ) : ℕ := (L + G) - R

-- Theorem stating the donation length required for Patricia to achieve her goal.
theorem donation_calculation : donation_length initial_length growth_length remaining_length = 23 :=
by
  -- Proof omitted
  sorry

end donation_calculation_l584_58447


namespace average_of_all_results_is_24_l584_58461

-- Definitions translated from conditions
def average_1 := 20
def average_2 := 30
def n1 := 30
def n2 := 20
def total_sum_1 := n1 * average_1
def total_sum_2 := n2 * average_2

-- Lean 4 statement
theorem average_of_all_results_is_24
  (h1 : total_sum_1 = n1 * average_1)
  (h2 : total_sum_2 = n2 * average_2) :
  ((total_sum_1 + total_sum_2) / (n1 + n2) = 24) :=
by
  sorry

end average_of_all_results_is_24_l584_58461


namespace paint_more_expensive_than_wallpaper_l584_58442

variable (x y z : ℝ)
variable (h : 4 * x + 4 * y = 7 * x + 2 * y + z)

theorem paint_more_expensive_than_wallpaper : y > x :=
by
  sorry

end paint_more_expensive_than_wallpaper_l584_58442


namespace infinite_solutions_no_solutions_l584_58489

-- Define the geometric sequence with first term a1 = 1 and common ratio q
def a1 : ℝ := 1
def a2 (q : ℝ) : ℝ := a1 * q
def a3 (q : ℝ) : ℝ := a1 * q^2
def a4 (q : ℝ) : ℝ := a1 * q^3

-- Define the system of linear equations
def system_of_eqns (x y q : ℝ) : Prop :=
  a1 * x + a3 q * y = 3 ∧ a2 q * x + a4 q * y = -2

-- Conditions for infinitely many solutions
theorem infinite_solutions (q x y : ℝ) :
  q = -2 / 3 → ∃ x y, system_of_eqns x y q :=
by
  sorry

-- Conditions for no solutions
theorem no_solutions (q : ℝ) :
  q ≠ -2 / 3 → ¬∃ x y, system_of_eqns x y q :=
by
  sorry

end infinite_solutions_no_solutions_l584_58489


namespace f_neg_2008_value_l584_58455

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x^7 + b * x - 2

theorem f_neg_2008_value (h : f a b 2008 = 10) : f a b (-2008) = -12 := by
  sorry

end f_neg_2008_value_l584_58455


namespace range_of_m_eq_l584_58493

theorem range_of_m_eq (m: ℝ) (x: ℝ) :
  (m+1 = 0 ∧ 4 > 0) ∨ 
  ((m + 1 > 0) ∧ ((m^2 - 2 * m - 3)^2 - 4 * (m + 1) * (-m + 3) < 0)) ↔ 
  (m ∈ Set.Icc (-1 : ℝ) 1 ∪ Set.Ico (1 : ℝ) 3) := 
sorry

end range_of_m_eq_l584_58493


namespace find_integers_with_sum_and_gcd_l584_58400

theorem find_integers_with_sum_and_gcd {a b : ℕ} (h_sum : a + b = 104055) (h_gcd : Nat.gcd a b = 6937) :
  (a = 6937 ∧ b = 79118) ∨ (a = 13874 ∧ b = 90181) ∨ (a = 27748 ∧ b = 76307) ∨ (a = 48559 ∧ b = 55496) :=
sorry

end find_integers_with_sum_and_gcd_l584_58400


namespace trigonometric_identity_l584_58451

-- The main statement to prove
theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (2 * Real.sin α - 2 * Real.cos α) / (4 * Real.sin α - 9 * Real.cos α) = -2 :=
by
  sorry

end trigonometric_identity_l584_58451


namespace monotonicity_of_f_l584_58463

noncomputable def f (a x : ℝ) : ℝ := (a * x) / (x + 1)

theorem monotonicity_of_f (a : ℝ) :
  (∀ x1 x2 : ℝ, -1 < x1 → -1 < x2 → x1 < x2 → 0 < a → f a x1 < f a x2) ∧
  (∀ x1 x2 : ℝ, -1 < x1 → -1 < x2 → x1 < x2 → a < 0 → f a x1 > f a x2) :=
by {
  sorry
}

end monotonicity_of_f_l584_58463


namespace train_speed_is_correct_l584_58456

/-- Define the length of the train and the time taken to cross the telegraph post. --/
def train_length : ℕ := 240
def crossing_time : ℕ := 16

/-- Define speed calculation based on train length and crossing time. --/
def train_speed : ℕ := train_length / crossing_time

/-- Prove that the computed speed of the train is 15 meters per second. --/
theorem train_speed_is_correct : train_speed = 15 := sorry

end train_speed_is_correct_l584_58456


namespace find_a_l584_58494

-- Define the main inequality condition
def inequality_condition (a x : ℝ) : Prop := |x^2 + a * x + 4 * a| ≤ 3

-- Define the condition that there is exactly one solution to the inequality
def has_exactly_one_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, (inequality_condition a x) ∧ (∀ y : ℝ, x ≠ y → ¬(inequality_condition a y))

-- The theorem that states the specific values of a
theorem find_a (a : ℝ) : has_exactly_one_solution a ↔ a = 8 + 2 * Real.sqrt 13 ∨ a = 8 - 2 * Real.sqrt 13 := 
by
  sorry

end find_a_l584_58494


namespace necessary_conditions_l584_58402

theorem necessary_conditions (a b c d e : ℝ) (h : (a + b + e) / (b + c) = (c + d + e) / (d + a)) :
  a = c ∨ a + b + c + d + e = 0 :=
by
  sorry

end necessary_conditions_l584_58402


namespace remainder_div_1442_l584_58482

theorem remainder_div_1442 (x k l r : ℤ) (h1 : 1816 = k * x + 6) (h2 : 1442 = l * x + r) (h3 : x = Int.gcd 1810 374) : r = 0 := by
  sorry

end remainder_div_1442_l584_58482


namespace nested_sqrt_eq_two_l584_58490

theorem nested_sqrt_eq_two (x : ℝ) (h : x = Real.sqrt (2 + x)) : x = 2 :=
sorry

end nested_sqrt_eq_two_l584_58490


namespace associate_professor_charts_l584_58416

theorem associate_professor_charts (A B C : ℕ) : 
  A + B = 8 → 
  2 * A + B = 10 → 
  C * A + 2 * B = 14 → 
  C = 1 := 
by 
  intros h1 h2 h3 
  sorry

end associate_professor_charts_l584_58416


namespace fraction_meaningful_condition_l584_58452

-- Define a variable x
variable (x : ℝ)

-- State the condition that makes the fraction meaningful
def fraction_meaningful (x : ℝ) : Prop := (x - 2) ≠ 0

-- State the theorem we want to prove
theorem fraction_meaningful_condition : fraction_meaningful x ↔ x ≠ 2 := sorry

end fraction_meaningful_condition_l584_58452


namespace negative_large_base_zero_exponent_l584_58478

-- Define the problem conditions: base number and exponent
def base_number : ℤ := -2023
def exponent : ℕ := 0

-- Prove that (-2023)^0 equals 1
theorem negative_large_base_zero_exponent : base_number ^ exponent = 1 := by
  sorry

end negative_large_base_zero_exponent_l584_58478


namespace more_bags_found_l584_58423

def bags_Monday : ℕ := 7
def bags_nextDay : ℕ := 12

theorem more_bags_found : bags_nextDay - bags_Monday = 5 := by
  -- Proof Skipped
  sorry

end more_bags_found_l584_58423


namespace handshake_problem_l584_58476

theorem handshake_problem (n : ℕ) (hn : n = 11) (H : n * (n - 1) / 2 = 55) : 10 = n - 1 :=
by
  sorry

end handshake_problem_l584_58476


namespace gcd_of_repeated_three_digit_l584_58486

theorem gcd_of_repeated_three_digit : 
  ∀ (n : ℕ), 100 ≤ n ∧ n < 1000 → ∀ m ∈ {k : ℕ | ∃ n, 100 ≤ n ∧ n < 1000 ∧ k = 1001 * n}, Nat.gcd 1001 m = 1001 :=
by
  sorry

end gcd_of_repeated_three_digit_l584_58486


namespace gcd_840_1764_l584_58432

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 :=
by
  sorry

end gcd_840_1764_l584_58432


namespace opposite_of_neg_quarter_l584_58464

theorem opposite_of_neg_quarter : -(- (1 / 4)) = 1 / 4 :=
by
  sorry

end opposite_of_neg_quarter_l584_58464


namespace correctness_of_statements_l584_58410

theorem correctness_of_statements :
  (statement1 ∧ statement4 ∧ statement5) :=
by sorry

end correctness_of_statements_l584_58410


namespace distance_big_rock_correct_l584_58428

noncomputable def rower_in_still_water := 7 -- km/h
noncomputable def river_flow := 2 -- km/h
noncomputable def total_trip_time := 1 -- hour

def distance_to_big_rock (D : ℝ) :=
  (D / (rower_in_still_water - river_flow)) + (D / (rower_in_still_water + river_flow)) = total_trip_time

theorem distance_big_rock_correct {D : ℝ} (h : distance_to_big_rock D) : D = 45 / 14 :=
sorry

end distance_big_rock_correct_l584_58428


namespace distance_traveled_on_foot_l584_58449

theorem distance_traveled_on_foot (x y : ℝ) (h1 : x + y = 80) (h2 : x / 8 + y / 16 = 7) : x = 32 :=
by
  sorry

end distance_traveled_on_foot_l584_58449


namespace problem_part1_problem_part2_l584_58471

theorem problem_part1 (x y : ℝ) (h1 : x = 1 / (3 - 2 * Real.sqrt 2)) (h2 : y = 1 / (3 + 2 * Real.sqrt 2)) : 
  x^2 * y - x * y^2 = 4 * Real.sqrt 2 := 
  sorry

theorem problem_part2 (x y : ℝ) (h1 : x = 1 / (3 - 2 * Real.sqrt 2)) (h2 : y = 1 / (3 + 2 * Real.sqrt 2)) : 
  x^2 - x * y + y^2 = 33 := 
  sorry

end problem_part1_problem_part2_l584_58471


namespace cost_of_purchase_l584_58404

theorem cost_of_purchase :
  (5 * 3) + (8 * 2) = 31 :=
by
  sorry

end cost_of_purchase_l584_58404


namespace determine_numbers_l584_58466

theorem determine_numbers (A B n : ℤ) (h1 : 0 ≤ n ∧ n ≤ 9) (h2 : A = 10 * B + n) (h3 : A + B = 2022) : 
  A = 1839 ∧ B = 183 :=
by
  -- proof will be filled in here
  sorry

end determine_numbers_l584_58466


namespace sum_of_powers_sequence_l584_58473

theorem sum_of_powers_sequence (a b : ℝ) 
  (h₁ : a + b = 1)
  (h₂ : a^2 + b^2 = 3)
  (h₃ : a^3 + b^3 = 4)
  (h₄ : a^4 + b^4 = 7)
  (h₅ : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 :=
sorry

end sum_of_powers_sequence_l584_58473


namespace cat_food_more_than_dog_food_l584_58470

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

end cat_food_more_than_dog_food_l584_58470


namespace polynomial_equality_l584_58407

theorem polynomial_equality :
  (3 * x + 1) ^ 4 = a * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e →
  a - b + c - d + e = 16 :=
by
  intro h
  sorry

end polynomial_equality_l584_58407


namespace corn_increase_factor_l584_58403

noncomputable def field_area : ℝ := 1

-- Let x be the remaining part of the field
variable (x : ℝ)

-- First condition: if the remaining part is fully planted with millet
-- Millet will occupy half of the field
axiom condition1 : (field_area - x) + x = field_area / 2

-- Second condition: if the remaining part x is equally divided between oats and corn
-- Oats will occupy half of the field
axiom condition2 : (field_area - x) + 0.5 * x = field_area / 2

-- Prove the factor by which the amount of corn increases
theorem corn_increase_factor : (0.5 * x + x) / (0.5 * x / 2) = 3 :=
by
  sorry

end corn_increase_factor_l584_58403


namespace value_of_a_l584_58474

theorem value_of_a (a b : ℚ) (h₁ : b = 3 * a) (h₂ : b = 12 - 5 * a) : a = 3 / 2 :=
by
  sorry

end value_of_a_l584_58474


namespace investor_difference_l584_58417

/-
Scheme A yields 30% of the capital within a year.
Scheme B yields 50% of the capital within a year.
Investor invested $300 in scheme A.
Investor invested $200 in scheme B.
We need to prove that the difference in total money between scheme A and scheme B after a year is $90.
-/

def schemeA_yield_rate : ℝ := 0.30
def schemeB_yield_rate : ℝ := 0.50
def schemeA_investment : ℝ := 300
def schemeB_investment : ℝ := 200

def total_after_year (investment : ℝ) (yield_rate : ℝ) : ℝ :=
  investment * (1 + yield_rate)

theorem investor_difference :
  total_after_year schemeA_investment schemeA_yield_rate - total_after_year schemeB_investment schemeB_yield_rate = 90 := by
  sorry

end investor_difference_l584_58417
