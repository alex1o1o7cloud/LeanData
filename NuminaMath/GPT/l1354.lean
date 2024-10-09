import Mathlib

namespace geometric_sum_first_six_terms_l1354_135448

theorem geometric_sum_first_six_terms : 
  let a := (1 : ℚ) / 2
  let r := (1 : ℚ) / 4
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 4095 / 6144 :=
by
  -- Definitions and properties of geometric series
  sorry

end geometric_sum_first_six_terms_l1354_135448


namespace ratio_debt_manny_to_annika_l1354_135484

-- Define the conditions
def money_jericho_has : ℕ := 30
def debt_to_annika : ℕ := 14
def remaining_money_after_debts : ℕ := 9

-- Define the amount Jericho owes Manny
def debt_to_manny : ℕ := money_jericho_has - debt_to_annika - remaining_money_after_debts

-- Prove the ratio of amount Jericho owes Manny to the amount he owes Annika is 1:2
theorem ratio_debt_manny_to_annika :
  debt_to_manny * 2 = debt_to_annika :=
by
  -- Proof goes here
  sorry

end ratio_debt_manny_to_annika_l1354_135484


namespace current_in_circuit_l1354_135412

open Complex

theorem current_in_circuit
  (V : ℂ := 2 + 3 * I)
  (Z : ℂ := 4 - 2 * I) :
  (V / Z) = (1 / 10 + 4 / 5 * I) :=
  sorry

end current_in_circuit_l1354_135412


namespace evaluate_expression_l1354_135478

theorem evaluate_expression (x y z : ℤ) (hx : x = 25) (hy : y = 33) (hz : z = 7) :
    (x - (y - z)) - ((x - y) - z) = 14 := by 
  sorry

end evaluate_expression_l1354_135478


namespace patty_coins_value_l1354_135447

theorem patty_coins_value (n d q : ℕ) (h₁ : n + d + q = 30) (h₂ : 5 * n + 15 * d - 20 * q = 120) : 
  5 * n + 10 * d + 25 * q = 315 := by
sorry

end patty_coins_value_l1354_135447


namespace solution_for_system_of_inequalities_l1354_135405

theorem solution_for_system_of_inequalities (p : ℝ) : 
  (19 * p < 10) ∧ (p > 0.5) ↔ (0.5 < p ∧ p < 10/19) := 
by {
  sorry
}

end solution_for_system_of_inequalities_l1354_135405


namespace circle_diameter_from_area_l1354_135414

theorem circle_diameter_from_area (A : ℝ) (hA : A = 400 * Real.pi) :
    ∃ D : ℝ, D = 40 := 
by
  -- Consider the formula for the area of a circle with radius r.
  -- The area is given as A = π * r^2.
  let r := Real.sqrt 400 -- Solve for radius r.
  have hr : r = 20 := by sorry
  -- The diameter D is twice the radius.
  let D := 2 * r 
  existsi D
  have hD : D = 40 := by sorry
  exact hD

end circle_diameter_from_area_l1354_135414


namespace pentagon_area_proof_l1354_135456

noncomputable def area_of_pentagon : ℕ :=
  let side1 := 18
  let side2 := 25
  let side3 := 30
  let side4 := 28
  let side5 := 25
  -- Assuming the total area calculated from problem's conditions
  950

theorem pentagon_area_proof : area_of_pentagon = 950 := by
  sorry

end pentagon_area_proof_l1354_135456


namespace protein_in_steak_is_correct_l1354_135474

-- Definitions of the conditions
def collagen_protein_per_scoop : ℕ := 18 / 2 -- 9 grams
def protein_powder_per_scoop : ℕ := 21 -- 21 grams

-- Define the total protein consumed
def total_protein (collagen_scoops protein_scoops : ℕ) (protein_from_steak : ℕ) : ℕ :=
  collagen_protein_per_scoop * collagen_scoops + protein_powder_per_scoop * protein_scoops + protein_from_steak

-- Condition in the problem
def total_protein_consumed : ℕ := 86

-- Prove that the protein in the steak is 56 grams
theorem protein_in_steak_is_correct : 
  total_protein 1 1 56 = total_protein_consumed :=
sorry

end protein_in_steak_is_correct_l1354_135474


namespace cubic_sum_l1354_135458

theorem cubic_sum (x y z : ℝ) (h1 : x + y + z = 2) (h2 : x * y + x * z + y * z = -5) (h3 : x * y * z = -6) :
  x^3 + y^3 + z^3 = 18 :=
by
  sorry

end cubic_sum_l1354_135458


namespace valid_parameterizations_l1354_135462

open Real

def is_scalar_multiple (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def lies_on_line (p : ℝ × ℝ) : Prop :=
  p.2 = 2 * p.1 - 7

def valid_parametrization (p d : ℝ × ℝ) : Prop :=
  lies_on_line p ∧ is_scalar_multiple d (1, 2)

theorem valid_parameterizations :
  valid_parametrization (4, 1) (-2, -4) ∧ 
  ¬ valid_parametrization (12, 17) (5, 10) ∧ 
  valid_parametrization (3.5, 0) (1, 2) ∧ 
  valid_parametrization (-2, -11) (0.5, 1) ∧ 
  valid_parametrization (0, -7) (10, 20) :=
by {
  sorry
}

end valid_parameterizations_l1354_135462


namespace golden_apples_first_six_months_l1354_135436

-- Use appropriate namespaces
namespace ApolloProblem

-- Define the given conditions
def total_cost : ℕ := 54
def months_in_half_year : ℕ := 6

-- Prove that the number of golden apples charged for the first six months is 18
theorem golden_apples_first_six_months (X : ℕ) 
  (h1 : 6 * X + 6 * (2 * X) = total_cost) : 
  6 * X = 18 := 
sorry

end ApolloProblem

end golden_apples_first_six_months_l1354_135436


namespace perimeter_of_rectangle_l1354_135469

-- Define the conditions
def area (l w : ℝ) : Prop := l * w = 180
def length_three_times_width (l w : ℝ) : Prop := l = 3 * w

-- Define the problem
theorem perimeter_of_rectangle (l w : ℝ) (h₁ : area l w) (h₂ : length_three_times_width l w) : 
  2 * (l + w) = 16 * Real.sqrt 15 := 
sorry

end perimeter_of_rectangle_l1354_135469


namespace largest_possible_perimeter_l1354_135492

theorem largest_possible_perimeter (x : ℕ) (h1 : 1 < x) (h2 : x < 11) : 
    5 + 6 + x ≤ 21 := 
  sorry

end largest_possible_perimeter_l1354_135492


namespace resistor_value_l1354_135439

-- Definitions based on given conditions
def U : ℝ := 9 -- Volt reading by the voltmeter
def I : ℝ := 2 -- Current reading by the ammeter
def U_total : ℝ := 2 * U -- Total voltage in the series circuit

-- Stating the theorem
theorem resistor_value (R₀ : ℝ) :
  (U_total = I * (2 * R₀)) → R₀ = 9 :=
by
  intro h
  sorry

end resistor_value_l1354_135439


namespace smallest_n_for_candy_distribution_l1354_135427

theorem smallest_n_for_candy_distribution : ∃ (n : ℕ), (∀ (a : ℕ), ∃ (x : ℕ), (x * (x + 1)) / 2 % n = a % n) ∧ n = 2 :=
sorry

end smallest_n_for_candy_distribution_l1354_135427


namespace find_cost_of_book_sold_at_loss_l1354_135443

-- Definitions from the conditions
def total_cost (C1 C2 : ℝ) : Prop := C1 + C2 = 540
def selling_price_loss (C1 : ℝ) : ℝ := 0.85 * C1
def selling_price_gain (C2 : ℝ) : ℝ := 1.19 * C2
def same_selling_price (SP1 SP2 : ℝ) : Prop := SP1 = SP2

theorem find_cost_of_book_sold_at_loss (C1 C2 : ℝ) 
  (h1 : total_cost C1 C2) 
  (h2 : same_selling_price (selling_price_loss C1) (selling_price_gain C2)) :
  C1 = 315 :=
by {
   sorry
}

end find_cost_of_book_sold_at_loss_l1354_135443


namespace complex_number_z_l1354_135490

theorem complex_number_z (z : ℂ) (h : (3 + 1 * I) * z = 4 - 2 * I) : z = 1 - I :=
by
  sorry

end complex_number_z_l1354_135490


namespace light_flash_time_l1354_135494

/--
A light flashes every few seconds. In 3/4 of an hour, it flashes 300 times.
Prove that it takes 9 seconds for the light to flash once.
-/
theorem light_flash_time : 
  (3 / 4 * 60 * 60) / 300 = 9 :=
by
  sorry

end light_flash_time_l1354_135494


namespace triangle_inequality_circumradius_l1354_135442

theorem triangle_inequality_circumradius (a b c R : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) 
  (circumradius_def : R = (a * b * c) / (4 * (Real.sqrt ((a + b + c) * (a + b - c) * (a - b + c) * (-a + b + c))))) :
  (1 / (a * b)) + (1 / (b * c)) + (1 / (c * a)) ≥ (1 / (R ^ 2)) :=
sorry

end triangle_inequality_circumradius_l1354_135442


namespace tammy_driving_rate_l1354_135438

-- Define the conditions given in the problem
def total_miles : ℕ := 1980
def total_hours : ℕ := 36

-- Define the desired rate to prove
def expected_rate : ℕ := 55

-- The theorem stating that given the conditions, Tammy's driving rate is correct
theorem tammy_driving_rate :
  total_miles / total_hours = expected_rate :=
by
  -- Detailed proof would go here
  sorry

end tammy_driving_rate_l1354_135438


namespace spending_difference_is_65_l1354_135454

-- Definitions based on conditions
def ice_cream_cones : ℕ := 15
def pudding_cups : ℕ := 5
def ice_cream_cost_per_unit : ℝ := 5
def pudding_cost_per_unit : ℝ := 2

-- The solution requires the calculation of the total cost and the difference
def total_ice_cream_cost : ℝ := ice_cream_cones * ice_cream_cost_per_unit
def total_pudding_cost : ℝ := pudding_cups * pudding_cost_per_unit
def spending_difference : ℝ := total_ice_cream_cost - total_pudding_cost

-- Theorem statement proving the difference is 65
theorem spending_difference_is_65 : spending_difference = 65 := by
  -- The proof is omitted with sorry
  sorry

end spending_difference_is_65_l1354_135454


namespace handshaking_remainder_l1354_135421

noncomputable def num_handshaking_arrangements_modulo (n : ℕ) : ℕ := sorry

theorem handshaking_remainder (N : ℕ) (h : num_handshaking_arrangements_modulo 9 = N) :
  N % 1000 = 16 :=
sorry

end handshaking_remainder_l1354_135421


namespace divisibility_by_7_l1354_135475

theorem divisibility_by_7 (m a : ℤ) (h : 0 ≤ a ∧ a ≤ 9) (B : ℤ) (hB : B = m - 2 * a) (h7 : B % 7 = 0) : (10 * m + a) % 7 = 0 := 
sorry

end divisibility_by_7_l1354_135475


namespace number_of_sodas_in_pack_l1354_135483

/-- Billy has twice as many brothers as sisters -/
def twice_as_many_brothers_as_sisters (brothers sisters : ℕ) : Prop :=
  brothers = 2 * sisters

/-- Billy has 2 sisters -/
def billy_has_2_sisters : Prop :=
  ∃ sisters : ℕ, sisters = 2

/-- Billy can give 2 sodas to each of his siblings if he wants to give out the entire pack while giving each sibling the same number of sodas -/
def divide_sodas_evenly (total_sodas siblings sodas_per_sibling : ℕ) : Prop :=
  total_sodas = siblings * sodas_per_sibling

/-- Determine the total number of sodas in the pack given the conditions -/
theorem number_of_sodas_in_pack : 
  ∃ (sisters brothers total_sodas : ℕ), 
    (twice_as_many_brothers_as_sisters brothers sisters) ∧ 
    (billy_has_2_sisters) ∧ 
    (divide_sodas_evenly total_sodas (sisters + brothers + 1) 2) ∧
    (total_sodas = 12) :=
by
  sorry

end number_of_sodas_in_pack_l1354_135483


namespace difference_of_sums_l1354_135495

def even_numbers_sum (n : ℕ) : ℕ := (n * (n + 1))
def odd_numbers_sum (n : ℕ) : ℕ := n^2

theorem difference_of_sums : 
  even_numbers_sum 3003 - odd_numbers_sum 3003 = 7999 := 
by {
  sorry 
}

end difference_of_sums_l1354_135495


namespace perpendicular_condition_l1354_135410

def line1 (a : ℝ) (x y : ℝ) : ℝ := a * x + 2 * y - 1
def line2 (a : ℝ) (x y : ℝ) : ℝ := 3 * x - a * y + 1

def perpendicular_lines (a : ℝ) : Prop := 
  ∀ (x y : ℝ), line1 a x y = 0 → line2 a x y = 0 → 3 * a - 2 * a = 0 

theorem perpendicular_condition (a : ℝ) (h : perpendicular_lines a) : a = 0 := sorry

end perpendicular_condition_l1354_135410


namespace squares_sum_l1354_135406

theorem squares_sum (a b c : ℝ) 
  (h1 : 36 - 4 * Real.sqrt 2 - 6 * Real.sqrt 3 + 12 * Real.sqrt 6 = (a * Real.sqrt 2 + b * Real.sqrt 3 + c) ^ 2) : 
  a^2 + b^2 + c^2 = 14 := 
by
  sorry

end squares_sum_l1354_135406


namespace units_digit_of_A_is_1_l1354_135446

-- Definition of A
def A : ℕ := 2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) + 1

-- Main theorem stating that the units digit of A is 1
theorem units_digit_of_A_is_1 : (A % 10) = 1 :=
by 
  -- Given conditions about powers of 3 and their properties in modulo 10
  sorry

end units_digit_of_A_is_1_l1354_135446


namespace avg_age_across_rooms_l1354_135418

namespace AverageAgeProof

def Room := Type

-- Conditions
def people_in_room_a : ℕ := 8
def avg_age_room_a : ℕ := 35

def people_in_room_b : ℕ := 5
def avg_age_room_b : ℕ := 30

def people_in_room_c : ℕ := 7
def avg_age_room_c : ℕ := 25

-- Combined Calculations
def total_people := people_in_room_a + people_in_room_b + people_in_room_c
def total_age := (people_in_room_a * avg_age_room_a) + (people_in_room_b * avg_age_room_b) + (people_in_room_c * avg_age_room_c)

noncomputable def average_age : ℚ := total_age / total_people

-- Proof that the average age of all the people across the three rooms is 30.25
theorem avg_age_across_rooms : average_age = 30.25 := 
sorry

end AverageAgeProof

end avg_age_across_rooms_l1354_135418


namespace find_common_ratio_l1354_135408

noncomputable def geometric_seq_sum (a₁ q : ℂ) (n : ℕ) :=
if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem find_common_ratio (a₁ q : ℂ) :
(geometric_seq_sum a₁ q 8) / (geometric_seq_sum a₁ q 4) = 2 → q = 1 :=
by
  intro h
  sorry

end find_common_ratio_l1354_135408


namespace total_balls_estimation_l1354_135404

theorem total_balls_estimation
  (n : ℕ)  -- Let n be the total number of balls in the bag
  (yellow_balls : ℕ)  -- Let yellow_balls be the number of yellow balls
  (frequency : ℝ)  -- Let frequency be the stabilized frequency of drawing a yellow ball
  (h1 : yellow_balls = 6)
  (h2 : frequency = 0.3)
  (h3 : (yellow_balls : ℝ) / (n : ℝ) = frequency) :
  n = 20 :=
by
  sorry

end total_balls_estimation_l1354_135404


namespace sector_angle_l1354_135459

theorem sector_angle (r : ℝ) (S : ℝ) (α : ℝ) (h₁ : r = 10) (h₂ : S = 50 * π / 3) (h₃ : S = 1 / 2 * r^2 * α) : 
  α = π / 3 :=
by
  sorry

end sector_angle_l1354_135459


namespace quadratic_touches_x_axis_l1354_135402

theorem quadratic_touches_x_axis (a : ℝ) : 
  (∃ x : ℝ, 2 * x ^ 2 - 8 * x + a = 0) ∧ (∀ y : ℝ, y^2 - 4 * a = 0 → y = 0) → a = 8 := 
by 
  sorry

end quadratic_touches_x_axis_l1354_135402


namespace digit_difference_l1354_135445

theorem digit_difference (x y : ℕ) (h : 10 * x + y - (10 * y + x) = 45) : x - y = 5 :=
sorry

end digit_difference_l1354_135445


namespace common_difference_l1354_135435

-- Definitions
variable (a₁ d : ℝ) -- First term and common difference of the arithmetic sequence

-- Conditions
def mean_nine_terms (a₁ d : ℝ) : Prop :=
  (1 / 2 * (2 * a₁ + 8 * d)) = 10

def mean_ten_terms (a₁ d : ℝ) : Prop :=
  (1 / 2 * (2 * a₁ + 9 * d)) = 13

-- Theorem to prove the common difference is 6
theorem common_difference (a₁ d : ℝ) :
  mean_nine_terms a₁ d → 
  mean_ten_terms a₁ d → 
  d = 6 := by
  intros
  sorry

end common_difference_l1354_135435


namespace clea_ride_escalator_time_l1354_135400

def clea_time_not_walking (x k y : ℝ) : Prop :=
  60 * x = y ∧ 24 * (x + k) = y ∧ 1.5 * x = k ∧ 40 = y / k

theorem clea_ride_escalator_time :
  ∀ (x y k : ℝ), 60 * x = y → 24 * (x + k) = y → (1.5 * x = k) → 40 = y / k :=
by
  intros x y k H1 H2 H3
  sorry

end clea_ride_escalator_time_l1354_135400


namespace sol_earnings_in_a_week_l1354_135423

-- Define the number of candy bars sold each day using recurrence relation
def candies_sold (n : ℕ) : ℕ :=
  match n with
  | 0     => 10  -- Day 1
  | (n+1) => candies_sold n + 4  -- Each subsequent day

-- Define the total candies sold in a week and total earnings in dollars
def total_candies_sold_in_a_week : ℕ :=
  List.sum (List.map candies_sold [0, 1, 2, 3, 4, 5])

def total_earnings_in_dollars : ℕ :=
  (total_candies_sold_in_a_week * 10) / 100

-- Proving that Sol will earn 12 dollars in a week
theorem sol_earnings_in_a_week : total_earnings_in_dollars = 12 := by
  sorry

end sol_earnings_in_a_week_l1354_135423


namespace discard_sacks_l1354_135496

theorem discard_sacks (harvested_sacks_per_day : ℕ) (oranges_per_day : ℕ) (oranges_per_sack : ℕ) :
  harvested_sacks_per_day = 76 → oranges_per_day = 600 → oranges_per_sack = 50 → 
  harvested_sacks_per_day - oranges_per_day / oranges_per_sack = 64 :=
by
  intros h1 h2 h3
  -- Automatically passes the proof as a placeholder
  sorry

end discard_sacks_l1354_135496


namespace arccos_of_half_eq_pi_over_three_l1354_135470

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l1354_135470


namespace find_y_l1354_135432

variable (a b c x : ℝ) (p q r y : ℝ)
variable (log : ℝ → ℝ) -- represents the logarithm function

-- Conditions as hypotheses
axiom log_eq : (log a) / p = (log b) / q
axiom log_eq' : (log b) / q = (log c) / r
axiom log_eq'' : (log c) / r = log x
axiom x_ne_one : x ≠ 1
axiom eq_exp : (b^3) / (a^2 * c) = x^y

-- Statement to be proven
theorem find_y : y = 3 * q - 2 * p - r := by
  sorry

end find_y_l1354_135432


namespace Trumpington_marching_band_max_l1354_135450

theorem Trumpington_marching_band_max (n : ℕ) (k : ℕ) 
  (h1 : 20 * n % 26 = 4)
  (h2 : n = 8 + 13 * k)
  (h3 : 20 * n < 1000) 
  : 20 * (8 + 13 * 3) = 940 := 
by
  sorry

end Trumpington_marching_band_max_l1354_135450


namespace ab_equiv_l1354_135466

theorem ab_equiv (a b : ℝ) (hb : b ≠ 0) (h : (a - b) / b = 3 / 7) : a / b = 10 / 7 :=
by
  sorry

end ab_equiv_l1354_135466


namespace odd_numbers_in_A_even_4k_minus_2_not_in_A_product_in_A_l1354_135422

def is_in_A (a : ℤ) : Prop := ∃ (x y : ℤ), a = x^2 - y^2

theorem odd_numbers_in_A :
  ∀ (n : ℤ), n % 2 = 1 → is_in_A n :=
sorry

theorem even_4k_minus_2_not_in_A :
  ∀ (k : ℤ), ¬ is_in_A (4 * k - 2) :=
sorry

theorem product_in_A :
  ∀ (a b : ℤ), is_in_A a → is_in_A b → is_in_A (a * b) :=
sorry

end odd_numbers_in_A_even_4k_minus_2_not_in_A_product_in_A_l1354_135422


namespace increase_in_area_is_44_percent_l1354_135457

-- Let's define the conditions first
variables {r : ℝ} -- radius of the medium pizza
noncomputable def radius_large (r : ℝ) := 1.2 * r
noncomputable def area (r : ℝ) := Real.pi * r ^ 2

-- Now we state the Lean theorem that expresses the problem
theorem increase_in_area_is_44_percent (r : ℝ) : 
  (area (radius_large r) - area r) / area r * 100 = 44 :=
by
  sorry

end increase_in_area_is_44_percent_l1354_135457


namespace triangle_XYZ_PQZ_lengths_l1354_135424

theorem triangle_XYZ_PQZ_lengths :
  ∀ (X Y Z P Q : Type) (d_XZ d_YZ d_PQ : ℝ),
  d_XZ = 9 → d_YZ = 12 → d_PQ = 3 →
  ∀ (XY YP : ℝ),
  XY = Real.sqrt (d_XZ^2 + d_YZ^2) →
  YP = (d_PQ / d_XZ) * d_YZ →
  YP = 4 :=
by
  intros X Y Z P Q d_XZ d_YZ d_PQ hXZ hYZ hPQ XY YP hXY hYP
  -- Skipping detailed proof
  sorry

end triangle_XYZ_PQZ_lengths_l1354_135424


namespace resulting_figure_has_25_sides_l1354_135488

/-- Consider a sequential construction starting with an isosceles triangle, adding a rectangle 
    on one side, then a regular hexagon on a non-adjacent side of the rectangle, followed by a
    regular heptagon, another regular hexagon, and finally, a regular nonagon. -/
def sides_sequence : List ℕ := [3, 4, 6, 7, 6, 9]

/-- The number of sides exposed to the outside in the resulting figure. -/
def exposed_sides (sides : List ℕ) : ℕ :=
  let total_sides := sides.sum
  let adjacent_count := 2 + 2 + 2 + 2 + 1
  total_sides - adjacent_count

theorem resulting_figure_has_25_sides :
  exposed_sides sides_sequence = 25 := 
by
  sorry

end resulting_figure_has_25_sides_l1354_135488


namespace smallest_x_2_abs_eq_24_l1354_135468

theorem smallest_x_2_abs_eq_24 : ∃ x : ℝ, (2 * |x - 10| = 24) ∧ (∀ y : ℝ, (2 * |y - 10| = 24) -> x ≤ y) := 
sorry

end smallest_x_2_abs_eq_24_l1354_135468


namespace remaining_puppies_l1354_135464

def initial_puppies : Nat := 8
def given_away_puppies : Nat := 4

theorem remaining_puppies : initial_puppies - given_away_puppies = 4 := 
by 
  sorry

end remaining_puppies_l1354_135464


namespace main_l1354_135433

def M (x : ℝ) : Prop := x^2 - 5 * x ≤ 0
def N (x : ℝ) (p : ℝ) : Prop := p < x ∧ x < 6
def intersection (x : ℝ) (q : ℝ) : Prop := 2 < x ∧ x ≤ q

theorem main (p q : ℝ) (hM : ∀ x, M x → 0 ≤ x ∧ x ≤ 5) (hN : ∀ x, N x p → p < x ∧ x < 6) (hMN : ∀ x, (M x ∧ N x p) ↔ intersection x q) :
  p + q = 7 :=
by
  sorry

end main_l1354_135433


namespace solve_for_x_l1354_135441

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ -1 then x + 2 
  else if x < 2 then x^2 
  else 2 * x

theorem solve_for_x (x : ℝ) : f x = 3 ↔ x = Real.sqrt 3 := by
  sorry

end solve_for_x_l1354_135441


namespace average_cost_per_trip_is_correct_l1354_135467

def oldest_pass_cost : ℕ := 100
def second_oldest_pass_cost : ℕ := 90
def third_oldest_pass_cost : ℕ := 80
def youngest_pass_cost : ℕ := 70

def oldest_trips : ℕ := 35
def second_oldest_trips : ℕ := 25
def third_oldest_trips : ℕ := 20
def youngest_trips : ℕ := 15

def total_cost : ℕ := oldest_pass_cost + second_oldest_pass_cost + third_oldest_pass_cost + youngest_pass_cost
def total_trips : ℕ := oldest_trips + second_oldest_trips + third_oldest_trips + youngest_trips

def average_cost_per_trip : ℚ := total_cost / total_trips

theorem average_cost_per_trip_is_correct : average_cost_per_trip = 340 / 95 :=
by sorry

end average_cost_per_trip_is_correct_l1354_135467


namespace number_of_blue_butterflies_l1354_135499

theorem number_of_blue_butterflies 
  (total_butterflies : ℕ)
  (B Y : ℕ)
  (H1 : total_butterflies = 11)
  (H2 : B = 2 * Y)
  (H3 : total_butterflies = B + Y + 5) : B = 4 := 
sorry

end number_of_blue_butterflies_l1354_135499


namespace cyclic_sum_inequality_l1354_135415

open BigOperators

theorem cyclic_sum_inequality {n : ℕ} (h : 0 < n) (a : ℕ → ℝ)
  (hpos : ∀ i, 0 < a i) :
  (∑ k in Finset.range n, a k / (a (k+1) + a (k+2))) > n / 4 := by
  sorry

end cyclic_sum_inequality_l1354_135415


namespace preceding_integer_binary_l1354_135403

theorem preceding_integer_binary (M : ℕ) (h : M = 0b110101) : 
  (M - 1) = 0b110100 :=
by
  sorry

end preceding_integer_binary_l1354_135403


namespace traffic_safety_team_eq_twice_fire_l1354_135487

-- Define initial members in the teams
def t0 : ℕ := 8
def f0 : ℕ := 7

-- Define the main theorem
theorem traffic_safety_team_eq_twice_fire (x : ℕ) : t0 + x = 2 * (f0 - x) :=
by sorry

end traffic_safety_team_eq_twice_fire_l1354_135487


namespace ordered_triple_unique_l1354_135431

variable (a b c : ℝ)

theorem ordered_triple_unique
  (h_pos_a : a > 4)
  (h_pos_b : b > 4)
  (h_pos_c : c > 4)
  (h_eq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) :
  (a, b, c) = (12, 10, 8) := 
sorry

end ordered_triple_unique_l1354_135431


namespace triangle_area_l1354_135440

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem triangle_area (a b c : ℕ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) (h4 : is_right_triangle a b c) :
  (1 / 2 : ℝ) * a * b = 180 :=
by sorry

end triangle_area_l1354_135440


namespace sum_not_equals_any_l1354_135465

-- Define the nine special natural numbers a1 to a9
def a1 (k : ℕ) : ℕ := (10^k - 1) / 9
def a2 (m : ℕ) : ℕ := 2 * (10^m - 1) / 9
def a3 (p : ℕ) : ℕ := 3 * (10^p - 1) / 9
def a4 (q : ℕ) : ℕ := 4 * (10^q - 1) / 9
def a5 (r : ℕ) : ℕ := 5 * (10^r - 1) / 9
def a6 (s : ℕ) : ℕ := 6 * (10^s - 1) / 9
def a7 (t : ℕ) : ℕ := 7 * (10^t - 1) / 9
def a8 (u : ℕ) : ℕ := 8 * (10^u - 1) / 9
def a9 (v : ℕ) : ℕ := 9 * (10^v - 1) / 9

-- Statement of the problem
theorem sum_not_equals_any (k m p q r s t u v : ℕ) :
  ¬ (a1 k = a2 m + a3 p + a4 q + a5 r + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a2 m = a1 k + a3 p + a4 q + a5 r + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a3 p = a1 k + a2 m + a4 q + a5 r + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a4 q = a1 k + a2 m + a3 p + a5 r + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a5 r = a1 k + a2 m + a3 p + a4 q + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a6 s = a1 k + a2 m + a3 p + a4 q + a5 r + a7 t + a8 u + a9 v) ∧
  ¬ (a7 t = a1 k + a2 m + a3 p + a4 q + a5 r + a6 s + a8 u + a9 v) ∧
  ¬ (a8 u = a1 k + a2 m + a3 p + a4 q + a5 r + a6 s + a7 t + a9 v) ∧
  ¬ (a9 v = a1 k + a2 m + a3 p + a4 q + a5 r + a6 s + a7 t + a8 u) :=
  sorry

end sum_not_equals_any_l1354_135465


namespace fisherman_caught_total_fish_l1354_135429

noncomputable def number_of_boxes : ℕ := 15
noncomputable def fish_per_box : ℕ := 20
noncomputable def fish_outside_boxes : ℕ := 6

theorem fisherman_caught_total_fish :
  number_of_boxes * fish_per_box + fish_outside_boxes = 306 :=
by
  sorry

end fisherman_caught_total_fish_l1354_135429


namespace missing_dimension_of_soap_box_l1354_135477

theorem missing_dimension_of_soap_box 
  (volume_carton : ℕ) 
  (volume_soap_box : ℕ)
  (number_of_boxes : ℕ)
  (x : ℕ) 
  (h1 : volume_carton = 25 * 48 * 60) 
  (h2 : volume_soap_box = x * 6 * 5)
  (h3: number_of_boxes = 300)
  (h4 : number_of_boxes * volume_soap_box = volume_carton) : 
  x = 8 := by 
  sorry

end missing_dimension_of_soap_box_l1354_135477


namespace p_div_q_is_12_l1354_135481

-- Definition of binomials and factorials required for the proof
open Nat

/-- Define the number of ways to distribute balls for configuration A -/
def config_A : ℕ :=
  @choose 5 1 * @choose 4 2 * @choose 2 1 * (factorial 20) / (factorial 2 * factorial 4 * factorial 4 * factorial 3 * factorial 7)

/-- Define the number of ways to distribute balls for configuration B -/
def config_B : ℕ :=
  @choose 5 2 * @choose 3 3 * (factorial 20) / (factorial 3 * factorial 3 * factorial 4 * factorial 4 * factorial 4)

/-- The ratio of probabilities p/q for the given distributions of balls into bins is 12 -/
theorem p_div_q_is_12 : config_A / config_B = 12 :=
by
  sorry

end p_div_q_is_12_l1354_135481


namespace motorcyclist_average_speed_BC_l1354_135491

theorem motorcyclist_average_speed_BC :
  ∀ (d_AB : ℝ) (theta : ℝ) (d_BC_half_d_AB : ℝ) (avg_speed_trip : ℝ)
    (time_ratio_AB_BC : ℝ) (total_speed : ℝ) (t_AB : ℝ) (t_BC : ℝ),
    d_AB = 120 →
    theta = 10 →
    d_BC_half_d_AB = 1 / 2 →
    avg_speed_trip = 30 →
    time_ratio_AB_BC = 3 →
    t_AB = 4.5 →
    t_BC = 1.5 →
    t_AB = time_ratio_AB_BC * t_BC →
    avg_speed_trip = total_speed →
    total_speed = (d_AB + (d_AB * d_BC_half_d_AB)) / (t_AB + t_BC) →
    t_AB / 3 = t_BC →
    ((d_AB * d_BC_half_d_AB) / t_BC = 40) :=
by
  intros d_AB theta d_BC_half_d_AB avg_speed_trip time_ratio_AB_BC total_speed
        t_AB t_BC h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end motorcyclist_average_speed_BC_l1354_135491


namespace sin_45_eq_sqrt2_div_2_l1354_135453

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l1354_135453


namespace inequality_solution_l1354_135493

theorem inequality_solution (x : ℝ) : (2 * x - 1) / 3 ≥ 1 → x ≥ 2 := by
  sorry

end inequality_solution_l1354_135493


namespace arithmetic_progression_11th_term_l1354_135401

theorem arithmetic_progression_11th_term:
  ∀ (a d : ℝ), (15 / 2) * (2 * a + 14 * d) = 56.25 → a + 6 * d = 3.25 → a + 10 * d = 5.25 :=
by
  intros a d h_sum h_7th
  sorry

end arithmetic_progression_11th_term_l1354_135401


namespace range_of_a_l1354_135449

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
if x < 1 then -x + 2 else a / x

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x < 1 ∧ (0 < -x + 2)) ∧ (∀ x : ℝ, x ≥ 1 → (0 < a / x)) → a ≥ 1 :=
by
  sorry

end range_of_a_l1354_135449


namespace sum_even_numbers_from_2_to_60_l1354_135411

noncomputable def sum_even_numbers_seq : ℕ :=
  let a₁ := 2
  let d := 2
  let aₙ := 60
  let n := (aₙ - a₁) / d + 1
  n / 2 * (a₁ + aₙ)

theorem sum_even_numbers_from_2_to_60:
  sum_even_numbers_seq = 930 :=
by
  sorry

end sum_even_numbers_from_2_to_60_l1354_135411


namespace grace_pennies_l1354_135407

theorem grace_pennies (dime_value nickel_value : ℕ) (dimes nickels : ℕ) 
  (h₁ : dime_value = 10) (h₂ : nickel_value = 5) (h₃ : dimes = 10) (h₄ : nickels = 10) : 
  dimes * dime_value + nickels * nickel_value = 150 := 
by 
  sorry

end grace_pennies_l1354_135407


namespace sin2_cos3_tan4_lt_zero_l1354_135471

theorem sin2_cos3_tan4_lt_zero (h1 : Real.sin 2 > 0) (h2 : Real.cos 3 < 0) (h3 : Real.tan 4 > 0) : Real.sin 2 * Real.cos 3 * Real.tan 4 < 0 :=
sorry

end sin2_cos3_tan4_lt_zero_l1354_135471


namespace jawbreakers_in_package_correct_l1354_135498

def jawbreakers_ate : Nat := 20
def jawbreakers_left : Nat := 4
def jawbreakers_in_package : Nat := jawbreakers_ate + jawbreakers_left

theorem jawbreakers_in_package_correct : jawbreakers_in_package = 24 := by
  sorry

end jawbreakers_in_package_correct_l1354_135498


namespace Ava_watched_television_for_240_minutes_l1354_135489

-- Define the conditions
def hours (h : ℕ) := h = 4

-- Define the conversion factor from hours to minutes
def convert_hours_to_minutes (h : ℕ) : ℕ := h * 60

-- State the theorem
theorem Ava_watched_television_for_240_minutes (h : ℕ) (hh : hours h) : convert_hours_to_minutes h = 240 :=
by
  -- The proof goes here but is skipped
  sorry

end Ava_watched_television_for_240_minutes_l1354_135489


namespace basketball_free_throws_l1354_135479

theorem basketball_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 2 * a)
  (h2 : b = a - 2)
  (h3 : 2 * a + 3 * b + x = 68) : x = 44 :=
by
  sorry

end basketball_free_throws_l1354_135479


namespace lanes_on_road_l1354_135473

theorem lanes_on_road (num_lanes : ℕ)
  (h1 : ∀ trucks_per_lane cars_per_lane total_vehicles, 
          cars_per_lane = 2 * (trucks_per_lane * num_lanes) ∧
          trucks_per_lane = 60 ∧
          total_vehicles = num_lanes * (trucks_per_lane + cars_per_lane) ∧
          total_vehicles = 2160) :
  num_lanes = 12 :=
by
  sorry

end lanes_on_road_l1354_135473


namespace cylindrical_to_rectangular_l1354_135461

theorem cylindrical_to_rectangular (r θ z : ℝ) (h₁ : r = 10) (h₂ : θ = Real.pi / 6) (h₃ : z = 2) :
  (r * Real.cos θ, r * Real.sin θ, z) = (5 * Real.sqrt 3, 5, 2) := 
by
  sorry

end cylindrical_to_rectangular_l1354_135461


namespace stability_comparison_l1354_135437

-- Definitions of conditions
def variance_A : ℝ := 3
def variance_B : ℝ := 1.2

-- Definition of the stability metric
def more_stable (performance_A performance_B : ℝ) : Prop :=
  performance_B < performance_A

-- Target Proposition
theorem stability_comparison (h_variance_A : variance_A = 3)
                            (h_variance_B : variance_B = 1.2) :
  more_stable variance_A variance_B = true :=
by
  sorry

end stability_comparison_l1354_135437


namespace least_odd_prime_factor_of_2023_pow_8_add_1_l1354_135480

theorem least_odd_prime_factor_of_2023_pow_8_add_1 :
  ∃ (p : ℕ), Prime p ∧ (2023^8 + 1) % p = 0 ∧ p % 2 = 1 ∧ p = 97 :=
by
  sorry

end least_odd_prime_factor_of_2023_pow_8_add_1_l1354_135480


namespace find_constants_and_calculate_result_l1354_135420

theorem find_constants_and_calculate_result :
  ∃ (a b : ℤ), 
    (∀ (x : ℤ), (x + a) * (x + 6) = x^2 + 8 * x + 12) ∧ 
    (∀ (x : ℤ), (x - a) * (x + b) = x^2 + x - 6) ∧ 
    (∀ (x : ℤ), (x + a) * (x + b) = x^2 + 5 * x + 6) :=
by
  sorry

end find_constants_and_calculate_result_l1354_135420


namespace solve_textbook_by_12th_l1354_135434

/-!
# Problem Statement
There are 91 problems in a textbook. Yura started solving them on September 6 and solves one problem less each subsequent morning. By the evening of September 8, there are 46 problems left to solve.

We need to prove that Yura finishes solving all the problems by September 12.
-/

def initial_problems : ℕ := 91
def problems_left_by_evening_of_8th : ℕ := 46

def problems_solved_by_evening_of_8th : ℕ :=
  initial_problems - problems_left_by_evening_of_8th

def z : ℕ := (problems_solved_by_evening_of_8th / 3)

theorem solve_textbook_by_12th 
    (total_problems : ℕ)
    (problems_left : ℕ)
    (solved_by_evening_8th : ℕ)
    (daily_problem_count : ℕ) :
    (total_problems = initial_problems) →
    (problems_left = problems_left_by_evening_of_8th) →
    (solved_by_evening_8th = problems_solved_by_evening_of_8th) →
    (daily_problem_count = z) →
    ∃ (finishing_date : ℕ), finishing_date = 12 :=
  by
    intros _ _ _ _
    sorry

end solve_textbook_by_12th_l1354_135434


namespace remainder_when_divided_by_7_l1354_135425

theorem remainder_when_divided_by_7 (k : ℕ) (h1 : k % 5 = 2) (h2 : k % 6 = 5) : k % 7 = 3 :=
sorry

end remainder_when_divided_by_7_l1354_135425


namespace problem_l1354_135452

theorem problem (h : ℤ) : (∃ x : ℤ, x = -2 ∧ x^3 + h * x - 12 = 0) → h = -10 := by
  sorry

end problem_l1354_135452


namespace ratio_of_votes_l1354_135444

theorem ratio_of_votes (randy_votes : ℕ) (shaun_votes : ℕ) (eliot_votes : ℕ)
  (h1 : randy_votes = 16)
  (h2 : shaun_votes = 5 * randy_votes)
  (h3 : eliot_votes = 160) :
  eliot_votes / shaun_votes = 2 :=
by
  sorry

end ratio_of_votes_l1354_135444


namespace gcd_lcm_divisible_l1354_135430

theorem gcd_lcm_divisible (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : Nat.gcd a b + Nat.lcm a b = a + b) : a % b = 0 ∨ b % a = 0 := 
sorry

end gcd_lcm_divisible_l1354_135430


namespace probability_of_success_l1354_135486

def prob_successful_attempt := 0.5

def prob_unsuccessful_attempt := 1 - prob_successful_attempt

def all_fail_prob := prob_unsuccessful_attempt ^ 4

def at_least_one_success_prob := 1 - all_fail_prob

theorem probability_of_success :
  at_least_one_success_prob = 0.9375 :=
by
  -- Proof would be here
  sorry

end probability_of_success_l1354_135486


namespace tan_beta_formula_l1354_135413

theorem tan_beta_formula (α β : ℝ) 
  (h1 : Real.tan α = -2/3)
  (h2 : Real.tan (α + β) = 1/2) :
  Real.tan β = 7/4 :=
sorry

end tan_beta_formula_l1354_135413


namespace solve_for_y_l1354_135419

theorem solve_for_y (y : ℝ)
  (h1 : 9 * y^2 + 8 * y - 1 = 0)
  (h2 : 27 * y^2 + 44 * y - 7 = 0) : 
  y = 1 / 9 :=
sorry

end solve_for_y_l1354_135419


namespace chips_probability_l1354_135485

/-- A bag contains 4 green, 3 orange, and 5 blue chips. If the 12 chips are randomly drawn from
    the bag, one at a time and without replacement, the probability that the chips are drawn such
    that the 4 green chips are drawn consecutively, the 3 orange chips are drawn consecutively,
    and the 5 blue chips are drawn consecutively, but not necessarily in the green-orange-blue
    order, is 1/4620. -/
theorem chips_probability :
  let total_chips := 12
  let factorial := Nat.factorial
  let favorable_outcomes := (factorial 3) * (factorial 4) * (factorial 3) * (factorial 5)
  let total_outcomes := factorial total_chips
  favorable_outcomes / total_outcomes = 1 / 4620 :=
by
  -- proof goes here, but we skip it
  sorry

end chips_probability_l1354_135485


namespace weeks_per_month_l1354_135463

-- Define the given conditions
def num_employees_initial : Nat := 500
def additional_employees : Nat := 200
def hourly_wage : Nat := 12
def daily_work_hours : Nat := 10
def weekly_work_days : Nat := 5
def total_monthly_pay : Nat := 1680000

-- Calculate the total number of employees after hiring
def total_employees : Nat := num_employees_initial + additional_employees

-- Calculate the pay rates
def daily_pay_per_employee : Nat := hourly_wage * daily_work_hours
def weekly_pay_per_employee : Nat := daily_pay_per_employee * weekly_work_days

-- Calculate the total weekly pay for all employees
def total_weekly_pay : Nat := weekly_pay_per_employee * total_employees

-- Define the statement to be proved
theorem weeks_per_month
  (h1 : total_employees = num_employees_initial + additional_employees)
  (h2 : daily_pay_per_employee = hourly_wage * daily_work_hours)
  (h3 : weekly_pay_per_employee = daily_pay_per_employee * weekly_work_days)
  (h4 : total_weekly_pay = weekly_pay_per_employee * total_employees)
  (h5 : total_monthly_pay = 1680000) :
  total_monthly_pay / total_weekly_pay = 4 :=
by sorry

end weeks_per_month_l1354_135463


namespace john_total_spent_l1354_135417

noncomputable def total_spent (computer_cost : ℝ) (peripheral_ratio : ℝ) (base_video_cost : ℝ) : ℝ :=
  let peripheral_cost := computer_cost * peripheral_ratio
  let upgraded_video_cost := base_video_cost * 2
  computer_cost + peripheral_cost + (upgraded_video_cost - base_video_cost)

theorem john_total_spent :
  total_spent 1500 0.2 300 = 2100 :=
by
  sorry

end john_total_spent_l1354_135417


namespace additional_wolves_in_pack_l1354_135460

-- Define the conditions
def wolves_out_hunting : ℕ := 4
def meat_per_wolf_per_day : ℕ := 8
def hunting_days : ℕ := 5
def meat_per_deer : ℕ := 200

-- Calculate total meat per wolf for hunting days
def meat_per_wolf_total : ℕ := meat_per_wolf_per_day * hunting_days

-- Calculate wolves fed per deer
def wolves_fed_per_deer : ℕ := meat_per_deer / meat_per_wolf_total

-- Calculate total deer killed by wolves out hunting
def total_deers_killed : ℕ := wolves_out_hunting

-- Calculate total meat provided by hunting wolves
def total_meat_provided : ℕ := total_deers_killed * meat_per_deer

-- Calculate number of wolves fed by total meat provided
def total_wolves_fed : ℕ := total_meat_provided / meat_per_wolf_total

-- Define the main theorem to prove the answer
theorem additional_wolves_in_pack (total_wolves_fed wolves_out_hunting : ℕ) : 
  total_wolves_fed - wolves_out_hunting = 16 :=
by
  sorry

end additional_wolves_in_pack_l1354_135460


namespace evaluate_expression_l1354_135451

theorem evaluate_expression :
  2 * 7^(-1/3 : ℝ) + (1/2 : ℝ) * Real.log (1/64) / Real.log 2 = -3 := 
  sorry

end evaluate_expression_l1354_135451


namespace max_c_value_for_f_x_range_l1354_135426

theorem max_c_value_for_f_x_range:
  (∀ c : ℝ, (∃ x : ℝ, x^2 + 4 * x + c = -2) → c ≤ 2) ∧ (∃ (x : ℝ), x^2 + 4 * x + 2 = -2) :=
sorry

end max_c_value_for_f_x_range_l1354_135426


namespace smallest_three_digit_number_l1354_135472

theorem smallest_three_digit_number (x : ℤ) (h1 : x - 7 % 7 = 0) (h2 : x - 8 % 8 = 0) (h3 : x - 9 % 9 = 0) : x = 504 := 
sorry

end smallest_three_digit_number_l1354_135472


namespace difference_of_squares_l1354_135482

theorem difference_of_squares (x y : ℕ) (h1 : x + y = 26) (h2 : x * y = 168) : x^2 - y^2 = 52 := by
  sorry

end difference_of_squares_l1354_135482


namespace functional_equation_divisibility_l1354_135409

theorem functional_equation_divisibility (f : ℕ+ → ℕ+) :
  (∀ x y : ℕ+, (f x)^2 + y ∣ f y + x^2) → (∀ x : ℕ+, f x = x) :=
by
  sorry

end functional_equation_divisibility_l1354_135409


namespace num_divisors_count_l1354_135476

theorem num_divisors_count (n : ℕ) (m : ℕ) (H : m = 32784) :
  (∃ S : Finset ℕ, (∀ x ∈ S, x ∈ (Finset.range 10) ∧ m % x = 0) ∧ S.card = n) ↔ n = 7 :=
by
  sorry

end num_divisors_count_l1354_135476


namespace solution_set_l1354_135428

-- Definitions representing the given conditions
def cond1 (x : ℝ) := x - 3 < 0
def cond2 (x : ℝ) := x + 1 ≥ 0

-- The problem: Prove the solution set is as given
theorem solution_set (x : ℝ) :
  (cond1 x) ∧ (cond2 x) ↔ -1 ≤ x ∧ x < 3 :=
by
  sorry

end solution_set_l1354_135428


namespace germination_percentage_l1354_135416

theorem germination_percentage (seeds_plot1 seeds_plot2 : ℕ) (percent_germ_plot1 : ℕ) (total_percent_germ : ℕ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  percent_germ_plot1 = 20 →
  total_percent_germ = 26 →
  ∃ (percent_germ_plot2 : ℕ), percent_germ_plot2 = 35 :=
by
  sorry

end germination_percentage_l1354_135416


namespace tiger_distance_proof_l1354_135455

-- Declare the problem conditions
def tiger_initial_speed : ℝ := 25
def tiger_initial_time : ℝ := 3
def tiger_slow_speed : ℝ := 10
def tiger_slow_time : ℝ := 4
def tiger_chase_speed : ℝ := 50
def tiger_chase_time : ℝ := 0.5

-- Compute individual distances
def distance1 := tiger_initial_speed * tiger_initial_time
def distance2 := tiger_slow_speed * tiger_slow_time
def distance3 := tiger_chase_speed * tiger_chase_time

-- Compute the total distance
def total_distance := distance1 + distance2 + distance3

-- The final theorem to prove
theorem tiger_distance_proof : total_distance = 140 := by
  sorry

end tiger_distance_proof_l1354_135455


namespace boy_age_proof_l1354_135497

theorem boy_age_proof (P X : ℕ) (hP : P = 16) (hcond : P - X = (P + 4) / 2) : X = 6 :=
by
  sorry

end boy_age_proof_l1354_135497
