import Mathlib

namespace matilda_fathers_chocolate_bars_l2369_236990

/-- Matilda had 20 chocolate bars and shared them evenly amongst herself and her 4 sisters.
    When her father got home, he was upset that they did not put aside any chocolates for him.
    They felt bad, so they each gave up half of their chocolate bars for their father.
    Their father then gave 3 chocolate bars to their mother and ate some.
    Matilda's father had 5 chocolate bars left.
    Prove that Matilda's father ate 2 chocolate bars. -/
theorem matilda_fathers_chocolate_bars:
  ∀ (total_chocolates initial_people chocolates_per_person given_to_father chocolates_left chocolates_eaten: ℕ ),
    total_chocolates = 20 →
    initial_people = 5 →
    chocolates_per_person = total_chocolates / initial_people →
    given_to_father = (chocolates_per_person / 2) * initial_people →
    chocolates_left = given_to_father - 3 →
    chocolates_left - 5 = chocolates_eaten →
    chocolates_eaten = 2 :=
by
  intros
  sorry

end matilda_fathers_chocolate_bars_l2369_236990


namespace materials_total_order_l2369_236954

theorem materials_total_order :
  let concrete := 0.16666666666666666
  let bricks := 0.16666666666666666
  let stone := 0.5
  concrete + bricks + stone = 0.8333333333333332 :=
by
  sorry

end materials_total_order_l2369_236954


namespace alice_savings_l2369_236919

def sales : ℝ := 2500
def basic_salary : ℝ := 240
def commission_rate : ℝ := 0.02
def savings_rate : ℝ := 0.10

theorem alice_savings :
  (basic_salary + (sales * commission_rate)) * savings_rate = 29 :=
by
  sorry

end alice_savings_l2369_236919


namespace ratio_tends_to_zero_as_n_tends_to_infinity_l2369_236902

def smallest_prime_not_dividing (n : ℕ) : ℕ :=
  -- Function to find the smallest prime not dividing n
  sorry

theorem ratio_tends_to_zero_as_n_tends_to_infinity :
  ∀ ε > 0, ∃ N, ∀ n > N, (smallest_prime_not_dividing n : ℝ) / (n : ℝ) < ε := by
  sorry

end ratio_tends_to_zero_as_n_tends_to_infinity_l2369_236902


namespace prob_A_winning_l2369_236904

variable (P_draw P_B : ℚ)

def P_A_winning := 1 - P_draw - P_B

theorem prob_A_winning (h1 : P_draw = 1 / 2) (h2 : P_B = 1 / 3) :
  P_A_winning P_draw P_B = 1 / 6 :=
by
  rw [P_A_winning, h1, h2]
  norm_num
  done

end prob_A_winning_l2369_236904


namespace number_of_bedrooms_l2369_236958

-- Conditions
def battery_life : ℕ := 10
def vacuum_time_per_room : ℕ := 4
def num_initial_rooms : ℕ := 2 -- kitchen and living room
def num_charges : ℕ := 2

-- Computation of total vacuuming time
def total_vacuuming_time : ℕ := battery_life * (num_charges + 1)

-- Computation of remaining time for bedrooms
def time_for_bedrooms : ℕ := total_vacuuming_time - (vacuum_time_per_room * num_initial_rooms)

-- Proof problem: Prove number of bedrooms
theorem number_of_bedrooms (B : ℕ) (h : B = time_for_bedrooms / vacuum_time_per_room) : B = 5 := by 
  sorry

end number_of_bedrooms_l2369_236958


namespace car_mass_nearest_pound_l2369_236951

def mass_of_car_kg : ℝ := 1500
def kg_to_pounds : ℝ := 0.4536

theorem car_mass_nearest_pound :
  (↑(Int.floor ((mass_of_car_kg / kg_to_pounds) + 0.5))) = 3307 :=
by
  sorry

end car_mass_nearest_pound_l2369_236951


namespace printingTime_l2369_236947

def printerSpeed : ℝ := 23
def pauseTime : ℝ := 2
def totalPages : ℝ := 350

theorem printingTime : (totalPages / printerSpeed) + ((totalPages / 50 - 1) * pauseTime) = 27 := by 
  sorry

end printingTime_l2369_236947


namespace remainder_of_division_l2369_236959

theorem remainder_of_division :
  Nat.mod 4536 32 = 24 :=
sorry

end remainder_of_division_l2369_236959


namespace minimum_value_of_expression_l2369_236918

theorem minimum_value_of_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  a^2 + b^2 + c^2 + (3 / (a + b + c)^2) ≥ 2 :=
sorry

end minimum_value_of_expression_l2369_236918


namespace range_of_a_l2369_236903

theorem range_of_a (a : ℝ) (x1 x2 : ℝ)
  (h_poly: ∀ x, x * x + (a * a - 1) * x + (a - 2) = 0 → x = x1 ∨ x = x2)
  (h_order: x1 < 1 ∧ 1 < x2) : 
  -2 < a ∧ a < 1 := 
sorry

end range_of_a_l2369_236903


namespace find_n_l2369_236994

-- Define the arithmetic series sums
def s1 (n : ℕ) : ℕ := (5 * n^2 + 5 * n) / 2
def s2 (n : ℕ) : ℕ := n^2 + n

-- The theorem to be proved
theorem find_n : ∃ n : ℕ, s1 n + s2 n = 156 ∧ n = 7 :=
by
  sorry

end find_n_l2369_236994


namespace range_of_m_l2369_236964

theorem range_of_m (x m : ℝ) (h1 : (m - 1) / (x + 1) = 1) (h2 : x < 0) : m < 2 ∧ m ≠ 1 :=
by
  sorry

end range_of_m_l2369_236964


namespace discount_percentage_l2369_236957

theorem discount_percentage (M C S : ℝ) (hC : C = 0.64 * M) (hS : S = C * 1.28125) :
  ((M - S) / M) * 100 = 18.08 := 
by
  sorry

end discount_percentage_l2369_236957


namespace problem_1_problem_2_l2369_236995

def f (x : ℝ) : ℝ := |2 * x - 1|

theorem problem_1 : {x : ℝ | f x > 2} = {x : ℝ | x < -1 / 2 ∨ x > 3 / 2} := sorry

theorem problem_2 (m : ℝ) : (∀ x : ℝ, f x + |2 * (x + 3)| - 4 > m * x) → m ≤ -11 := sorry

end problem_1_problem_2_l2369_236995


namespace tangent_line_circle_l2369_236993

theorem tangent_line_circle (m : ℝ) (h : ∀ x y : ℝ,  (x + y + m = 0) → (x^2 + y^2 = m) → m = 2) : m = 2 :=
sorry

end tangent_line_circle_l2369_236993


namespace salad_dressing_vinegar_percentage_l2369_236952

-- Define the initial conditions
def percentage_in_vinegar_in_Q : ℝ := 10
def percentage_of_vinegar_in_combined : ℝ := 12
def percentage_of_dressing_P_in_combined : ℝ := 0.10
def percentage_of_dressing_Q_in_combined : ℝ := 0.90
def percentage_of_vinegar_in_P (V : ℝ) : ℝ := V

-- The statement to prove
theorem salad_dressing_vinegar_percentage (V : ℝ) 
  (hQ : percentage_in_vinegar_in_Q = 10)
  (hCombined : percentage_of_vinegar_in_combined = 12)
  (hP_combined : percentage_of_dressing_P_in_combined = 0.10)
  (hQ_combined : percentage_of_dressing_Q_in_combined = 0.90)
  (hV_combined : 0.10 * percentage_of_vinegar_in_P V + 0.90 * percentage_in_vinegar_in_Q = 12) :
  V = 30 :=
by 
  sorry

end salad_dressing_vinegar_percentage_l2369_236952


namespace solve_for_x_l2369_236927

theorem solve_for_x (x : ℝ) (h : (2012 + x)^2 = x^2) : x = -1006 := 
sorry

end solve_for_x_l2369_236927


namespace like_terms_exponents_equal_l2369_236949

theorem like_terms_exponents_equal (a b : ℤ) :
  (∀ x y : ℝ, 2 * x^a * y^2 = -3 * x^3 * y^(b+3) → a = 3 ∧ b = -1) :=
by
  sorry

end like_terms_exponents_equal_l2369_236949


namespace solution_of_equation_l2369_236987

theorem solution_of_equation (m : ℝ) :
  (∃ x : ℝ, x = (4 - 3 * m) / 2 ∧ x > 0) ↔ m < 4 / 3 ∧ m ≠ 2 / 3 :=
by
  sorry

end solution_of_equation_l2369_236987


namespace number_of_intersections_l2369_236938

/-- 
  Define the two curves as provided in the problem:
  curve1 is defined by the equation 3x² + 2y² = 6,
  curve2 is defined by the equation x² - 2y² = 1.
  We aim to prove that there are exactly 4 distinct intersection points.
--/
def curve1 (x y : ℝ) : Prop := 3 * x^2 + 2 * y^2 = 6

def curve2 (x y : ℝ) : Prop := x^2 - 2 * y^2 = 1

theorem number_of_intersections : ∃ (points : Finset (ℝ × ℝ)), (∀ p ∈ points, curve1 p.1 p.2 ∧ curve2 p.1 p.2) ∧ points.card = 4 :=
sorry

end number_of_intersections_l2369_236938


namespace increment_in_displacement_l2369_236969

variable (d : ℝ)

def equation_of_motion (t : ℝ) : ℝ := 2 * t^2

theorem increment_in_displacement:
  let t1 := 2
  let t2 := 2 + d
  let s1 := equation_of_motion t1
  let s2 := equation_of_motion t2
  s2 - s1 = 8 * d + 2 * d^2 := by
  sorry

end increment_in_displacement_l2369_236969


namespace gcd_lcm_8951_4267_l2369_236943

theorem gcd_lcm_8951_4267 :
  gcd 8951 4267 = 1 ∧ lcm 8951 4267 = 38212917 :=
by
  sorry

end gcd_lcm_8951_4267_l2369_236943


namespace abc_sum_l2369_236940

theorem abc_sum (a b c : ℤ) 
  (h1 : ∀ x : ℤ, (x + a) * (x + b) = x^2 + 21 * x + 110)
  (h2 : ∀ x : ℤ, (x - b) * (x - c) = x^2 - 19 * x + 88) : 
  a + b + c = 29 := 
by
  sorry

end abc_sum_l2369_236940


namespace moles_of_BeOH2_l2369_236985

-- Definitions based on the given conditions
def balanced_chemical_equation (xBe2C xH2O xBeOH2 xCH4 : ℕ) : Prop :=
  xBe2C = 1 ∧ xH2O = 4 ∧ xBeOH2 = 2 ∧ xCH4 = 1

def initial_conditions (yBe2C yH2O : ℕ) : Prop :=
  yBe2C = 1 ∧ yH2O = 4

-- Lean statement to prove the number of moles of Beryllium hydroxide formed
theorem moles_of_BeOH2 (xBe2C xH2O xBeOH2 xCH4 yBe2C yH2O : ℕ) (h1 : balanced_chemical_equation xBe2C xH2O xBeOH2 xCH4) (h2 : initial_conditions yBe2C yH2O) :
  xBeOH2 = 2 :=
by
  sorry

end moles_of_BeOH2_l2369_236985


namespace max_discardable_grapes_l2369_236926

theorem max_discardable_grapes (n : ℕ) (k : ℕ) (h : k = 8) : 
  ∃ m : ℕ, m < k ∧ (∀ q : ℕ, q * k + m = n) ∧ m = 7 :=
by
  sorry

end max_discardable_grapes_l2369_236926


namespace value_of_b_add_c_l2369_236960

variables {a b c d : ℝ}

theorem value_of_b_add_c (h1 : a + b = 5) (h2 : c + d = 3) (h3 : a + d = 2) : b + c = 6 :=
sorry

end value_of_b_add_c_l2369_236960


namespace determine_values_l2369_236920

theorem determine_values (A B : ℚ) :
  (A + B = 4) ∧ (2 * A - 7 * B = 3) →
  A = 31 / 9 ∧ B = 5 / 9 :=
by
  sorry

end determine_values_l2369_236920


namespace savings_account_amount_l2369_236930

-- Definitions and conditions from the problem
def checking_account_yen : ℕ := 6359
def total_yen : ℕ := 9844

-- Question we aim to prove - the amount in the savings account
def savings_account_yen : ℕ := total_yen - checking_account_yen

-- Lean statement to prove the equality
theorem savings_account_amount : savings_account_yen = 3485 :=
by
  sorry

end savings_account_amount_l2369_236930


namespace positive_root_of_cubic_eq_l2369_236907

theorem positive_root_of_cubic_eq : ∃ (x : ℝ), x > 0 ∧ x^3 - 3 * x^2 - x - Real.sqrt 2 = 0 ∧ x = 2 + Real.sqrt 2 := by
  sorry

end positive_root_of_cubic_eq_l2369_236907


namespace pyramid_height_l2369_236906

theorem pyramid_height (perimeter_side_base : ℝ) (apex_distance_to_vertex : ℝ) (height_peak_to_center_base : ℝ) : 
  (perimeter_side_base = 32) → (apex_distance_to_vertex = 12) → 
  height_peak_to_center_base = 4 * Real.sqrt 7 := 
  by
    sorry

end pyramid_height_l2369_236906


namespace sufficient_but_not_necessary_l2369_236966

theorem sufficient_but_not_necessary (a b : ℝ) : (ab >= 2) -> a^2 + b^2 >= 4 ∧ ∃ a b : ℝ, a^2 + b^2 >= 4 ∧ ab < 2 := by
  sorry

end sufficient_but_not_necessary_l2369_236966


namespace total_container_weight_is_correct_l2369_236914

-- Definitions based on the conditions
def copper_bar_weight : ℕ := 90
def steel_bar_weight : ℕ := copper_bar_weight + 20
def tin_bar_weight : ℕ := steel_bar_weight / 2
def aluminum_bar_weight : ℕ := tin_bar_weight + 10

-- Number of bars in the container
def count_steel_bars : ℕ := 10
def count_tin_bars : ℕ := 15
def count_copper_bars : ℕ := 12
def count_aluminum_bars : ℕ := 8

-- Total weight of each type of bar
def total_steel_weight : ℕ := count_steel_bars * steel_bar_weight
def total_tin_weight : ℕ := count_tin_bars * tin_bar_weight
def total_copper_weight : ℕ := count_copper_bars * copper_bar_weight
def total_aluminum_weight : ℕ := count_aluminum_bars * aluminum_bar_weight

-- Total weight of the container
def total_container_weight : ℕ := total_steel_weight + total_tin_weight + total_copper_weight + total_aluminum_weight

-- Theorem to prove
theorem total_container_weight_is_correct : total_container_weight = 3525 := by
  sorry

end total_container_weight_is_correct_l2369_236914


namespace total_wheels_at_station_l2369_236961

/--
There are 4 trains at a train station.
Each train has 4 carriages.
Each carriage has 3 rows of wheels.
Each row of wheels has 5 wheels.
The total number of wheels at the train station is 240.
-/
theorem total_wheels_at_station : 
    let number_of_trains := 4
    let carriages_per_train := 4
    let rows_per_carriage := 3
    let wheels_per_row := 5
    number_of_trains * carriages_per_train * rows_per_carriage * wheels_per_row = 240 := 
by
    sorry

end total_wheels_at_station_l2369_236961


namespace num_ways_to_tile_3x5_is_40_l2369_236910

-- Definition of the problem
def numTilings (tiles : List (ℕ × ℕ)) (m n : ℕ) : ℕ :=
  sorry -- Placeholder for actual tiling computation

-- Condition specific to this problem
def specificTiles : List (ℕ × ℕ) :=
  [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]

-- Problem statement in Lean 4
theorem num_ways_to_tile_3x5_is_40 :
  numTilings specificTiles 3 5 = 40 :=
sorry

end num_ways_to_tile_3x5_is_40_l2369_236910


namespace max_mow_time_l2369_236996

-- Define the conditions
def timeToMow (x : ℕ) : Prop := 
  let timeToFertilize := 2 * x
  x + timeToFertilize = 120

-- State the theorem
theorem max_mow_time (x : ℕ) (h : timeToMow x) : x = 40 := by
  sorry

end max_mow_time_l2369_236996


namespace time_to_coffee_shop_is_18_l2369_236956

variable (cycle_constant_pace : Prop)
variable (time_cycle_library : ℕ)
variable (distance_cycle_library : ℕ)
variable (distance_to_coffee_shop : ℕ)

theorem time_to_coffee_shop_is_18
  (h_const_pace : cycle_constant_pace)
  (h_time_library : time_cycle_library = 30)
  (h_distance_library : distance_cycle_library = 5)
  (h_distance_coffee : distance_to_coffee_shop = 3)
  : (30 / 5) * 3 = 18 :=
by
  sorry

end time_to_coffee_shop_is_18_l2369_236956


namespace students_per_van_l2369_236998

def number_of_boys : ℕ := 60
def number_of_girls : ℕ := 80
def number_of_vans : ℕ := 5

theorem students_per_van : (number_of_boys + number_of_girls) / number_of_vans = 28 := by
  sorry

end students_per_van_l2369_236998


namespace face_opposite_to_A_l2369_236908

-- Define the faces and their relationships
inductive Face : Type
| A | B | C | D | E | F
open Face

def adjacent (x y : Face) : Prop :=
  match x, y with
  | A, B => true
  | B, A => true
  | C, A => true
  | A, C => true
  | D, A => true
  | A, D => true
  | C, D => true
  | D, C => true
  | E, F => true
  | F, E => true
  | _, _ => false

-- Theorem stating that "F" is opposite to "A" given the provided conditions.
theorem face_opposite_to_A : ∀ x : Face, (adjacent A x = false) → (x = B ∨ x = C ∨ x = D → false) → (x = E ∨ x = F) → x = F := 
  by
    intros x h1 h2 h3
    sorry

end face_opposite_to_A_l2369_236908


namespace tangent_circles_t_value_l2369_236928

theorem tangent_circles_t_value (t : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 = t^2 → x^2 + y^2 + 6 * x - 8 * y + 24 = 0 → dist (0, 0) (-3, 4) = t + 1) → t = 4 :=
by
  sorry

end tangent_circles_t_value_l2369_236928


namespace boys_girls_students_l2369_236913

theorem boys_girls_students (total_students : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ)
  (h1 : total_students = 100)
  (h2 : ratio_boys = 3)
  (h3 : ratio_girls = 2) :
  3 * (total_students / (ratio_boys + ratio_girls)) - 2 * (total_students / (ratio_boys + ratio_girls)) = 20 :=
by
  sorry

end boys_girls_students_l2369_236913


namespace book_price_l2369_236997

theorem book_price (P : ℝ) : 
  (3 * 12 * P - 500 = 220) → 
  P = 20 :=
by
  intro h
  sorry

end book_price_l2369_236997


namespace pizza_payment_difference_l2369_236915

theorem pizza_payment_difference
  (total_slices : ℕ := 12)
  (plain_cost : ℝ := 12)
  (onion_cost : ℝ := 3)
  (jack_onion_slices : ℕ := 4)
  (jack_plain_slices : ℕ := 3)
  (carl_plain_slices : ℕ := 5) :
  let total_cost := plain_cost + onion_cost
  let cost_per_slice := total_cost / total_slices
  let jack_onion_payment := jack_onion_slices * cost_per_slice
  let jack_plain_payment := jack_plain_slices * cost_per_slice
  let jack_total_payment := jack_onion_payment + jack_plain_payment
  let carl_total_payment := carl_plain_slices * cost_per_slice
  jack_total_payment - carl_total_payment = 2.5 :=
by
  sorry

end pizza_payment_difference_l2369_236915


namespace problem_statement_l2369_236972

theorem problem_statement (x : ℝ) (h₀ : 0 < x) (h₁ : x + 1 / x ≥ 2) (h₂ : x + 4 / x ^ 2 ≥ 3) (h₃ : x + 27 / x ^ 3 ≥ 4) :
  ∀ a : ℝ, (x + a / x ^ 4 ≥ 5) → a = 4 ^ 4 := 
by 
  sorry

end problem_statement_l2369_236972


namespace baseball_team_earnings_l2369_236937

theorem baseball_team_earnings (S : ℝ) (W : ℝ) (Total : ℝ) 
    (h1 : S = 2662.50) 
    (h2 : W = S - 142.50) 
    (h3 : Total = W + S) : 
  Total = 5182.50 :=
sorry

end baseball_team_earnings_l2369_236937


namespace min_holiday_days_l2369_236963

theorem min_holiday_days 
  (rained_days : ℕ) 
  (sunny_mornings : ℕ)
  (sunny_afternoons : ℕ) 
  (condition1 : rained_days = 7) 
  (condition2 : sunny_mornings = 5) 
  (condition3 : sunny_afternoons = 6) :
  ∃ (days : ℕ), days = 9 :=
by
  -- The specific steps of the proof are omitted as per the instructions
  sorry

end min_holiday_days_l2369_236963


namespace solve_for_x_l2369_236983

theorem solve_for_x (x y : ℚ) (h1 : 3 * x - 2 * y = 8) (h2 : x + 3 * y = 7) : x = 38 / 11 :=
by
  sorry

end solve_for_x_l2369_236983


namespace sum_of_fractions_l2369_236967

theorem sum_of_fractions :
  (1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8) + 1 / (8 * 9)) = (2 / 9) :=
by
  sorry

end sum_of_fractions_l2369_236967


namespace proposition_3_true_proposition_4_true_l2369_236931

def exp_pos (x : ℝ) : Prop := Real.exp x > 0

def two_power_gt_xsq (x : ℝ) : Prop := 2^x > x^2

def prod_gt_one (a b : ℝ) (ha : a > 1) (hb : b > 1) : Prop := a * b > 1

def geom_seq_nec_suff (a b c : ℝ) : Prop := ¬(b = Real.sqrt (a * c) ∨ (a * b = c * b ∧ b^2 = a * c))

theorem proposition_3_true (a b : ℝ) (ha : a > 1) (hb : b > 1) : prod_gt_one a b ha hb :=
sorry

theorem proposition_4_true (a b c : ℝ) : geom_seq_nec_suff a b c :=
sorry

end proposition_3_true_proposition_4_true_l2369_236931


namespace coronavirus_diameter_scientific_notation_l2369_236935

theorem coronavirus_diameter_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), a = 1.1 ∧ n = -7 ∧ 0.00000011 = a * 10^n := by
sorry

end coronavirus_diameter_scientific_notation_l2369_236935


namespace union_sets_l2369_236962

def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_sets : M ∪ N = {0, 1, 3, 9} := by
  sorry

end union_sets_l2369_236962


namespace volunteer_org_percentage_change_l2369_236922

theorem volunteer_org_percentage_change 
  (initial_membership : ℝ)
  (fall_increase_rate : ℝ)
  (spring_decrease_rate : ℝ) :
  (initial_membership = 100) →
  (fall_increase_rate = 0.05) →
  (spring_decrease_rate = 0.19) →
  (14.95 : ℝ) =
  ((initial_membership * (1 + fall_increase_rate)) * (1 - spring_decrease_rate)
  - initial_membership) / initial_membership * 100 := by
  sorry

end volunteer_org_percentage_change_l2369_236922


namespace initial_friends_online_l2369_236974

theorem initial_friends_online (F : ℕ) 
  (h1 : 8 + F = 13) 
  (h2 : 6 * F = 30) : 
  F = 5 :=
by
  sorry

end initial_friends_online_l2369_236974


namespace valid_numbers_l2369_236982

-- Define the conditions for three-digit numbers
def isThreeDigitNumber (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000

-- Define the splitting cases and the required property
def satisfiesFirstCase (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a ≠ 0 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧
  n = 100 * a + 10 * b + c ∧
  3 * ((10 * a + b) * c) = n

def satisfiesSecondCase (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a ≠ 0 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧
  n = 100 * a + 10 * b + c ∧
  3 * (a * (10 * b + c)) = n

-- Define the main proposition
def validThreeDigitNumber (n : ℕ) : Prop :=
  isThreeDigitNumber n ∧ (satisfiesFirstCase n ∨ satisfiesSecondCase n)

-- The theorem statement which we need to prove
theorem valid_numbers : ∀ n : ℕ, validThreeDigitNumber n ↔ n = 150 ∨ n = 240 ∨ n = 735 :=
by
  sorry

end valid_numbers_l2369_236982


namespace parallel_lines_slope_eq_l2369_236909

theorem parallel_lines_slope_eq {a : ℝ} : (∀ x : ℝ, 2*x - 1 = a*x + 1) → a = 2 :=
by
  sorry

end parallel_lines_slope_eq_l2369_236909


namespace asymptotes_equation_l2369_236933

noncomputable def hyperbola_asymptotes (x y : ℝ) : Prop :=
  x^2 / 64 - y^2 / 36 = 1

theorem asymptotes_equation :
  ∀ (x y : ℝ), hyperbola_asymptotes x y → (y = (3/4) * x ∨ y = - (3/4) * x) :=
by
  intro x y
  intro h
  sorry

end asymptotes_equation_l2369_236933


namespace find_f_neg2_l2369_236973

noncomputable def f (x : ℝ) : ℝ := -2 * (x + 1) + 1

theorem find_f_neg2 : f (-2) = 3 := by
  sorry

end find_f_neg2_l2369_236973


namespace luke_initial_money_l2369_236968

def initial_amount (X : ℤ) : Prop :=
  let spent := 11
  let received := 21
  let current_amount := 58
  X - spent + received = current_amount

theorem luke_initial_money : ∃ (X : ℤ), initial_amount X ∧ X = 48 :=
by
  sorry

end luke_initial_money_l2369_236968


namespace find_m_l2369_236921

noncomputable def inverse_proportion (x : ℝ) : ℝ := 4 / x

theorem find_m (m n : ℝ) (h1 : ∀ x, -4 ≤ x ∧ x ≤ m → inverse_proportion x = 4 / x ∧ n ≤ inverse_proportion x ∧ inverse_proportion x ≤ n + 3) :
  m = -1 :=
by
  sorry

end find_m_l2369_236921


namespace Morgan_first_SAT_score_l2369_236950

variable (S : ℝ) -- Morgan's first SAT score
variable (improved_score : ℝ := 1100) -- Improved score on second attempt
variable (improvement_rate : ℝ := 0.10) -- Improvement rate

theorem Morgan_first_SAT_score:
  improved_score = S * (1 + improvement_rate) → S = 1000 := 
by 
  sorry

end Morgan_first_SAT_score_l2369_236950


namespace draw_odds_l2369_236917

theorem draw_odds (x : ℝ) (bet_Zubilo bet_Shaiba bet_Draw payout : ℝ) (h1 : bet_Zubilo = 3 * x) (h2 : bet_Shaiba = 2 * x) (h3 : payout = 6 * x) : 
  bet_Draw * 6 = payout :=
by
  sorry

end draw_odds_l2369_236917


namespace trig_eq_solution_l2369_236923

open Real

theorem trig_eq_solution (x : ℝ) : 
  (cos (7 * x) + cos (3 * x) + sin (7 * x) - sin (3 * x) + sqrt 2 * cos (4 * x) = 0) ↔ 
  (∃ k : ℤ, 
    (x = -π / 8 + π * k / 2) ∨ 
    (x = -π / 4 + 2 * π * k / 3) ∨ 
    (x = 3 * π / 28 + 2 * π * k / 7)) :=
by sorry

end trig_eq_solution_l2369_236923


namespace complementary_angle_measure_l2369_236988

theorem complementary_angle_measure (A S C : ℝ) (h1 : A = 45) (h2 : A + S = 180) (h3 : A + C = 90) (h4 : S = 3 * C) : C = 45 :=
by
  sorry

end complementary_angle_measure_l2369_236988


namespace ratio_part_to_whole_l2369_236976

/-- One part of one third of two fifth of a number is 17, and 40% of that number is 204. 
Prove that the ratio of the part to the whole number is 1:30. -/
theorem ratio_part_to_whole 
  (N : ℝ)
  (h1 : (1 / 1) * (1 / 3) * (2 / 5) * N = 17) 
  (h2 : 0.40 * N = 204) : 
  17 / N = 1 / 30 :=
  sorry

end ratio_part_to_whole_l2369_236976


namespace valid_domain_of_x_l2369_236944

theorem valid_domain_of_x (x : ℝ) : 
  (x + 1 ≥ 0 ∧ x ≠ 0) ↔ (x ≥ -1 ∧ x ≠ 0) :=
by sorry

end valid_domain_of_x_l2369_236944


namespace valid_number_of_m_values_l2369_236970

theorem valid_number_of_m_values : 
  (∃ m : ℕ, 2 ≤ m ∧ m ∣ 420 ∧ 2 ≤ (420 / m)) ∧ ∀ m, 2 ≤ m ∧ m ∣ 420 ∧ 2 ≤ (420 / m) → m > 1  → 
  ∃ n : ℕ, n = 22 :=
by
  sorry

end valid_number_of_m_values_l2369_236970


namespace circles_intersect_l2369_236971

section PositionalRelationshipCircles

-- Define the first circle O1 with center (1, 0) and radius 1
def Circle1 (p : ℝ × ℝ) : Prop := (p.1 - 1)^2 + p.2^2 = 1

-- Define the second circle O2 with center (0, 3) and radius 3
def Circle2 (p : ℝ × ℝ) : Prop := p.1^2 + (p.2 - 3)^2 = 9

-- Prove that the positional relationship between Circle1 and Circle2 is intersecting
theorem circles_intersect : 
  ∃ p : ℝ × ℝ, Circle1 p ∧ Circle2 p :=
sorry

end PositionalRelationshipCircles

end circles_intersect_l2369_236971


namespace celsius_to_fahrenheit_conversion_l2369_236936

theorem celsius_to_fahrenheit_conversion (k b : ℝ) :
  (∀ C : ℝ, (C * k + b = C * 1.8 + 32)) → (k = 1.8 ∧ b = 32) :=
by
  intro h
  sorry

end celsius_to_fahrenheit_conversion_l2369_236936


namespace true_propositions_3_and_4_l2369_236932

-- Define the condition for Proposition ③
def prop3_statement (m : ℝ) : Prop :=
  (m > 2) → ∀ x : ℝ, (x^2 - 2*x + m > 0)

def prop3_contrapositive (m : ℝ) : Prop :=
  (∀ x : ℝ, (x^2 - 2*x + m > 0)) → (m > 2)

-- Define the condition for Proposition ④
def prop4_condition (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x : ℝ, f (1 + x) = f (1 - x))

def prop4_period_4 (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 4) = f (x))

-- Theorem to prove Propositions ③ and ④ are true
theorem true_propositions_3_and_4
  (m : ℝ) (f : ℝ → ℝ)
  (h3 : ∀ (m : ℝ), prop3_contrapositive m)
  (h4 : prop4_condition f): 
  prop3_statement m ∧ prop4_period_4 f :=
by {
  sorry
}

end true_propositions_3_and_4_l2369_236932


namespace Diamond_evaluation_l2369_236977

-- Redefine the operation Diamond
def Diamond (a b : ℕ) : ℕ := a * b^3 - b^2 + 1

-- Statement of the proof
theorem Diamond_evaluation : (Diamond 3 2) = 21 := by
  sorry

end Diamond_evaluation_l2369_236977


namespace count_valid_48_tuples_l2369_236925

open BigOperators

theorem count_valid_48_tuples : 
  ∃ n : ℕ, n = 54 ^ 48 ∧ 
  ( ∃ a : Fin 48 → ℕ, 
    (∀ i : Fin 48, 0 ≤ a i ∧ a i ≤ 100) ∧ 
    (∀ (i j : Fin 48), i < j → a i ≠ a j ∧ a i ≠ a j + 1) 
  ) :=
by
  sorry

end count_valid_48_tuples_l2369_236925


namespace vector_dot_product_l2369_236980

-- Definitions of the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-1, 2)

-- Definition of the dot product for 2D vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Main statement to prove
theorem vector_dot_product :
  dot_product (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) = 0 :=
by
  sorry

end vector_dot_product_l2369_236980


namespace price_ratio_l2369_236975

-- Definitions based on the provided conditions
variables (x y : ℕ) -- number of ballpoint pens and gel pens respectively
variables (b g T : ℝ) -- price of ballpoint pen, gel pen, and total amount paid respectively

-- The two given conditions
def cond1 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * g = 4 * (x * b + y * g)

def cond2 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * b = (x * b + y * g) / 2

-- The goal to prove
theorem price_ratio (x y : ℕ) (b g T : ℝ) (h1 : cond1 x y b g T) (h2 : cond2 x y b g T) : 
  g = 8 * b :=
sorry

end price_ratio_l2369_236975


namespace juice_packs_in_box_l2369_236948

theorem juice_packs_in_box 
  (W_box L_box H_box W_juice_pack L_juice_pack H_juice_pack : ℕ)
  (hW_box : W_box = 24) (hL_box : L_box = 15) (hH_box : H_box = 28)
  (hW_juice_pack : W_juice_pack = 4) (hL_juice_pack : L_juice_pack = 5) (hH_juice_pack : H_juice_pack = 7) : 
  (W_box * L_box * H_box) / (W_juice_pack * L_juice_pack * H_juice_pack) = 72 :=
by
  sorry

end juice_packs_in_box_l2369_236948


namespace greatest_integer_c_l2369_236999

theorem greatest_integer_c (c : ℤ) :
  (∀ x : ℝ, x^2 + (c : ℝ) * x + 10 ≠ 0) → c = 6 :=
by
  sorry

end greatest_integer_c_l2369_236999


namespace new_oranges_added_l2369_236989
-- Import the necessary library

-- Define the constants and conditions
def initial_oranges : ℕ := 5
def thrown_away : ℕ := 2
def total_oranges_now : ℕ := 31

-- Define new_oranges as the variable we want to prove
def new_oranges (x : ℕ) : Prop := x = 28

-- The theorem to prove how many new oranges were added
theorem new_oranges_added :
  ∃ (x : ℕ), new_oranges x ∧ total_oranges_now = initial_oranges - thrown_away + x :=
by
  sorry

end new_oranges_added_l2369_236989


namespace rectangular_prism_dimensions_l2369_236905

theorem rectangular_prism_dimensions 
    (a b c : ℝ) -- edges of the rectangular prism
    (h_increase_volume : (2 * a * b = 90)) -- condition 2: increasing height increases volume by 90 cm³ 
    (h_volume_proportion : (a * (c + 2)) / 2 = (3 / 5) * (a * b * c)) -- condition 3: height change results in 3/5 of original volume
    (h_edge_relation : (a = 5 * b ∨ b = 5 * a ∨ a * b = 45)) -- condition 1: one edge 5 times longer
    : 
    (a = 0.9 ∧ b = 50 ∧ c = 10) ∨ (a = 2 ∧ b = 22.5 ∧ c = 10) ∨ (a = 3 ∧ b = 15 ∧ c = 10) :=
sorry

end rectangular_prism_dimensions_l2369_236905


namespace cost_of_a_pen_l2369_236924

theorem cost_of_a_pen:
  ∃ x y : ℕ, 5 * x + 4 * y = 345 ∧ 3 * x + 6 * y = 285 ∧ x = 52 :=
by
  sorry

end cost_of_a_pen_l2369_236924


namespace least_possible_sum_l2369_236965

theorem least_possible_sum (p q : ℕ) (hp : 1 < p) (hq : 1 < q) (h : 17 * (p + 1) = 21 * (q + 1)) : p + q = 5 :=
sorry

end least_possible_sum_l2369_236965


namespace average_speed_return_trip_l2369_236942

/--
A train travels from Albany to Syracuse, a distance of 120 miles,
at an average rate of 50 miles per hour. The train then continues
to Rochester, which is 90 miles from Syracuse, before returning
to Albany. On its way to Rochester, the train's average speed is
60 miles per hour. Finally, the train travels back to Albany from
Rochester, with the total travel time of the train, including all
three legs of the journey, being 9 hours and 15 minutes. What was
the average rate of speed of the train on the return trip from
Rochester to Albany?
-/
theorem average_speed_return_trip :
  let dist_Albany_Syracuse := 120 -- miles
  let speed_Albany_Syracuse := 50 -- miles per hour
  let dist_Syracuse_Rochester := 90 -- miles
  let speed_Syracuse_Rochester := 60 -- miles per hour
  let total_travel_time := 9.25 -- hours (9 hours 15 minutes)
  let time_Albany_Syracuse := dist_Albany_Syracuse / speed_Albany_Syracuse
  let time_Syracuse_Rochester := dist_Syracuse_Rochester / speed_Syracuse_Rochester
  let total_time_so_far := time_Albany_Syracuse + time_Syracuse_Rochester
  let time_return_trip := total_travel_time - total_time_so_far
  let dist_return_trip := dist_Albany_Syracuse + dist_Syracuse_Rochester
  let average_speed_return := dist_return_trip / time_return_trip
  average_speed_return = 39.25 :=
by
  -- sorry placeholder for the actual proof
  sorry

end average_speed_return_trip_l2369_236942


namespace subset_A_implies_a_subset_B_implies_range_a_l2369_236978

variable (a : ℝ)

def A : Set ℝ := {x | x^2 - 2*x - 8 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a^2 - 12 = 0}

theorem subset_A_implies_a (h : A ⊆ B a) : a = -2 := 
sorry

theorem subset_B_implies_range_a (h : B a ⊆ A) : a >= 4 ∨ a < -4 ∨ a = -2 := 
sorry

end subset_A_implies_a_subset_B_implies_range_a_l2369_236978


namespace roots_of_quadratic_eq_l2369_236953

theorem roots_of_quadratic_eq : ∀ x : ℝ, (x^2 = 9) → (x = 3 ∨ x = -3) :=
by
  sorry

end roots_of_quadratic_eq_l2369_236953


namespace total_players_l2369_236991

theorem total_players (kabaddi : ℕ) (only_kho_kho : ℕ) (both_games : ℕ) 
  (h_kabaddi : kabaddi = 10) (h_only_kho_kho : only_kho_kho = 15) 
  (h_both_games : both_games = 5) : (kabaddi - both_games) + only_kho_kho + both_games = 25 :=
by
  sorry

end total_players_l2369_236991


namespace purple_sequins_each_row_l2369_236939

theorem purple_sequins_each_row (x : ℕ) : 
  (6 * 8) + (9 * 6) + (5 * x) = 162 → x = 12 :=
by 
  sorry

end purple_sequins_each_row_l2369_236939


namespace find_excluded_number_l2369_236912

-- Definition of the problem conditions
def avg (nums : List ℕ) : ℕ := (nums.sum / nums.length)

-- Problem condition: the average of 5 numbers is 27
def condition1 (nums : List ℕ) : Prop :=
  nums.length = 5 ∧ avg nums = 27

-- Problem condition: excluding one number, the average of remaining 4 numbers is 25
def condition2 (nums : List ℕ) (x : ℕ) : Prop :=
  let nums' := nums.filter (λ n => n ≠ x)
  nums.length = 5 ∧ nums'.length = 4 ∧ avg nums' = 25

-- Proof statement: finding the excluded number
theorem find_excluded_number (nums : List ℕ) (x : ℕ) (h1 : condition1 nums) (h2 : condition2 nums x) : x = 35 := 
by
  sorry

end find_excluded_number_l2369_236912


namespace experiment_implies_101_sq_1_equals_10200_l2369_236916

theorem experiment_implies_101_sq_1_equals_10200 :
    (5^2 - 1 = 24) →
    (7^2 - 1 = 48) →
    (11^2 - 1 = 120) →
    (13^2 - 1 = 168) →
    (101^2 - 1 = 10200) :=
by
  repeat { intro }
  sorry

end experiment_implies_101_sq_1_equals_10200_l2369_236916


namespace sin_double_alpha_zero_l2369_236946

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem sin_double_alpha_zero (α : ℝ) (h : f α = 1) : Real.sin (2 * α) = 0 :=
by 
  -- Proof would go here, but we're using sorry
  sorry

end sin_double_alpha_zero_l2369_236946


namespace simplify_expression_l2369_236901

variable (a b : ℝ)
variable (h1 : 0 < a)
variable (h2 : a ≠ 1)
variable (h3 : 0 < b)

theorem simplify_expression : a ^ Real.log (1 / b ^ Real.log a) = 1 / b ^ (Real.log a) ^ 2 :=
by
  sorry

end simplify_expression_l2369_236901


namespace find_angle_A_max_perimeter_incircle_l2369_236934

-- Definition of the triangle and the conditions
variables {A B C : Real} {a b c : Real} 

-- The conditions given in the problem
def triangle_conditions (a b c A B C : Real) : Prop :=
  (b + c = a * (Real.cos C + Real.sqrt 3 * Real.sin C)) ∧
  A + B + C = Real.pi

-- Part 1: Prove the value of angle A
theorem find_angle_A (a b c A B C : Real) 
(h : triangle_conditions a b c A B C) : 
A = Real.pi / 3 := sorry

-- Part 2: Prove the maximum perimeter of the incircle when a=2
theorem max_perimeter_incircle (b c A B C : Real) 
(h : triangle_conditions 2 b c A B C) : 
2 * Real.pi * (Real.sqrt 3 / 6 * (b + c - 2)) ≤ (2 * Real.sqrt 3 / 3) * Real.pi := sorry

end find_angle_A_max_perimeter_incircle_l2369_236934


namespace coordinates_of_C_l2369_236900

theorem coordinates_of_C (A B : ℝ × ℝ) (hA : A = (-2, -1)) (hB : B = (4, 9)) :
    ∃ C : ℝ × ℝ, (dist C A) = 4 * dist C B ∧ C = (-0.8, 1) :=
sorry

end coordinates_of_C_l2369_236900


namespace solution_set_of_inequality_l2369_236981

-- We define the inequality condition
def inequality (x : ℝ) : Prop := (x - 3) * (x + 2) < 0

-- We need to state that for all real numbers x, iff x satisfies the inequality,
-- then x must be within the interval (-2, 3).
theorem solution_set_of_inequality :
  ∀ x : ℝ, inequality x ↔ -2 < x ∧ x < 3 :=
by {
   sorry
}

end solution_set_of_inequality_l2369_236981


namespace determine_n_l2369_236941

theorem determine_n (n : ℕ) (h : 3^n = 27 * 81^3 / 9^4) : n = 7 := by
  sorry

end determine_n_l2369_236941


namespace rhombus_new_perimeter_l2369_236984

theorem rhombus_new_perimeter (d1 d2 : ℝ) (scale : ℝ) (h_d1 : d1 = 10) (h_d2 : d2 = 24) (h_scale : scale = 0.5) : 
  4 * (scale * (Real.sqrt ((d1/2)^2 + (d2/2)^2))) = 26 := 
by
  sorry

end rhombus_new_perimeter_l2369_236984


namespace Anya_loss_games_l2369_236979

noncomputable def game_indices_Anya_lost (Anya Bella Valya Galya Dasha : ℕ) : Set ℕ :=
  {4, 8, 12, 16}

theorem Anya_loss_games
  (Anya Bella Valya Galya Dasha total_games : ℕ)
  (h1 : Anya = 4)
  (h2 : Bella = 6)
  (h3 : Valya = 7)
  (h4 : Galya = 10)
  (h5 : Dasha = 11)
  (h6 : total_games = (Anya + Bella + Valya + Galya + Dasha) / 2)
  (h7 : total_games = 19) :
  game_indices_Anya_lost Anya Bella Valya Galya Dasha = {4, 8, 12, 16} :=
  by sorry

end Anya_loss_games_l2369_236979


namespace sequence_third_order_and_nth_term_l2369_236911

-- Define the given sequence
def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 4
  | 1 => 6
  | 2 => 13
  | 3 => 27
  | 4 => 50
  | 5 => 84
  | _ => sorry -- let’s define the general form for other terms later

-- Define first differences
def first_diff (n : ℕ) : ℤ := a (n + 1) - a n

-- Define second differences
def second_diff (n : ℕ) : ℤ := first_diff (n + 1) - first_diff n

-- Define third differences
def third_diff (n : ℕ) : ℤ := second_diff (n + 1) - second_diff n

-- Define the nth term formula
noncomputable def nth_term (n : ℕ) : ℚ := (1 / 6) * (2 * n^3 + 3 * n^2 - 11 * n + 30)

-- Theorem stating the least possible order is 3 and the nth term formula
theorem sequence_third_order_and_nth_term :
  (∀ n, third_diff n = 2) ∧ (∀ n, a n = nth_term n) :=
by
  sorry

end sequence_third_order_and_nth_term_l2369_236911


namespace express_y_in_terms_of_x_l2369_236945

theorem express_y_in_terms_of_x (x y : ℝ) (h : y - 2 * x = 6) : y = 2 * x + 6 :=
by
  sorry

end express_y_in_terms_of_x_l2369_236945


namespace total_floor_area_covered_l2369_236955

theorem total_floor_area_covered (combined_area : ℝ) (area_two_layers : ℝ) (area_three_layers : ℝ) : 
  combined_area = 200 → 
  area_two_layers = 22 → 
  area_three_layers = 19 → 
  (combined_area - (area_two_layers + 2 * area_three_layers)) = 140 := 
by
  sorry

end total_floor_area_covered_l2369_236955


namespace max_f_value_inequality_m_n_l2369_236986

section
variable (x : ℝ)

def f (x : ℝ) := abs (x - 1) - 2 * abs (x + 1)

theorem max_f_value : ∃ k, (∀ x : ℝ, f x ≤ k) ∧ (∃ x₀ : ℝ, f x₀ = k) ∧ k = 2 := 
by sorry

theorem inequality_m_n (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 1 / m + 1 / (2 * n) = 2) :
  m + 2 * n ≥ 2 :=
by sorry

end

end max_f_value_inequality_m_n_l2369_236986


namespace find_f2_l2369_236992

noncomputable def f (a b x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem find_f2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 :=
by
  sorry

end find_f2_l2369_236992


namespace product_formula_l2369_236929

theorem product_formula :
  (3 + 5) * (3^2 + 5^2) * (3^4 + 5^4) * (3^8 + 5^8) *
  (3^16 + 5^16) * (3^32 + 5^32) * (3^64 + 5^64) *
  (3^128 + 5^128) = 3^256 - 5^256 := by
  sorry

end product_formula_l2369_236929
