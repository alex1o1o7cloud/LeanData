import Mathlib

namespace cyclist_speed_l2153_215343

theorem cyclist_speed 
  (v : ℝ) 
  (hiker1_speed : ℝ := 4)
  (hiker2_speed : ℝ := 5)
  (cyclist_overtakes_hiker2_after_hiker1 : ∃ t1 t2 : ℝ, 
      t1 = 8 / (v - hiker1_speed) ∧ 
      t2 = 5 / (v - hiker2_speed) ∧ 
      t2 - t1 = 1/6)
: (v = 20 ∨ v = 7 ∨ abs (v - 6.5) < 0.1) :=
sorry

end cyclist_speed_l2153_215343


namespace C_share_of_profit_l2153_215366

variable (A B C P Rs_36000 k : ℝ)

-- Definitions as per the conditions given in the problem statement.
def investment_A := 24000
def investment_B := 32000
def investment_C := 36000
def total_profit := 92000
def C_Share := 36000

-- The Lean statement without the proof as requested.
theorem C_share_of_profit 
  (h_A : investment_A = 24000)
  (h_B : investment_B = 32000)
  (h_C : investment_C = 36000)
  (h_P : total_profit = 92000)
  (h_C_share : C_Share = 36000)
  : C_Share = (investment_C / k) / ((investment_A / k) + (investment_B / k) + (investment_C / k)) * total_profit := 
sorry

end C_share_of_profit_l2153_215366


namespace no_third_number_for_lcm_l2153_215392

theorem no_third_number_for_lcm (a : ℕ) : ¬ (Nat.lcm (Nat.lcm 23 46) a = 83) :=
sorry

end no_third_number_for_lcm_l2153_215392


namespace unique_even_odd_decomposition_l2153_215332

def is_symmetric (s : Set ℝ) : Prop := ∀ x ∈ s, -x ∈ s

def is_even (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ x ∈ s, f (-x) = f x

def is_odd (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ x ∈ s, f (-x) = -f x

theorem unique_even_odd_decomposition (s : Set ℝ) (hs : is_symmetric s) (f : ℝ → ℝ) (hf : ∀ x ∈ s, True) :
  ∃! g h : ℝ → ℝ, (is_even g s) ∧ (is_odd h s) ∧ (∀ x ∈ s, f x = g x + h x) :=
sorry

end unique_even_odd_decomposition_l2153_215332


namespace spring_length_at_9kg_l2153_215350

theorem spring_length_at_9kg :
  (∃ (k b : ℝ), (∀ x : ℝ, y = k * x + b) ∧ 
                 (y = 10 ∧ x = 0) ∧ 
                 (y = 10.5 ∧ x = 1)) → 
  (∀ x : ℝ, x = 9 → y = 14.5) :=
sorry

end spring_length_at_9kg_l2153_215350


namespace length_of_BC_is_eight_l2153_215387

theorem length_of_BC_is_eight (a : ℝ) (h_area : (1 / 2) * (2 * a) * a^2 = 64) : 2 * a = 8 := 
by { sorry }

end length_of_BC_is_eight_l2153_215387


namespace repeating_decimal_as_fraction_l2153_215371

-- Define the repeating decimal 4.25252525... as x
def repeating_decimal : ℚ := 4 + 25 / 99

-- Theorem statement to prove the equivalence
theorem repeating_decimal_as_fraction :
  repeating_decimal = 421 / 99 :=
by
  sorry

end repeating_decimal_as_fraction_l2153_215371


namespace black_spools_l2153_215363

-- Define the given conditions
def spools_per_beret : ℕ := 3
def red_spools : ℕ := 12
def blue_spools : ℕ := 6
def berets_made : ℕ := 11

-- Define the statement to be proved using the defined conditions
theorem black_spools (spools_per_beret red_spools blue_spools berets_made : ℕ) : (spools_per_beret * berets_made) - (red_spools + blue_spools) = 15 :=
by sorry

end black_spools_l2153_215363


namespace part_a_part_b_l2153_215395

-- Define a polygon type
structure Polygon :=
  (sides : ℕ)
  (area : ℝ)
  (grid_size : ℕ)

-- Define a function to verify drawable polygon
def DrawablePolygon (p : Polygon) : Prop :=
  ∃ (n : ℕ), p.grid_size = n ∧ p.area = n ^ 2

-- Part (a): 20-sided polygon with an area of 9
theorem part_a : DrawablePolygon {sides := 20, area := 9, grid_size := 3} :=
by
  sorry

-- Part (b): 100-sided polygon with an area of 49
theorem part_b : DrawablePolygon {sides := 100, area := 49, grid_size := 7} :=
by
  sorry

end part_a_part_b_l2153_215395


namespace shape_of_triangle_l2153_215337

-- Define the problem conditions
variable {a b : ℝ}
variable {A B C : ℝ}
variable (triangle_condition : (a^2 / b^2 = tan A / tan B))

-- Define the theorem to be proved
theorem shape_of_triangle ABC
  (h : triangle_condition):
  (A = B ∨ A + B = π / 2) :=
sorry

end shape_of_triangle_l2153_215337


namespace circle_sum_condition_l2153_215325

theorem circle_sum_condition (n : ℕ) (n_ge_1 : n ≥ 1)
  (x : Fin n → ℝ) (sum_x : (Finset.univ.sum x) = n - 1) :
  ∃ j : Fin n, ∀ k : ℕ, k ≥ 1 → k ≤ n → (Finset.range k).sum (fun i => x ⟨(j + i) % n, sorry⟩) ≥ k - 1 :=
sorry

end circle_sum_condition_l2153_215325


namespace Martin_correct_answers_l2153_215330

theorem Martin_correct_answers (C K M : ℕ) 
  (h1 : C = 35)
  (h2 : K = C + 8)
  (h3 : M = K - 3) : 
  M = 40 :=
by
  sorry

end Martin_correct_answers_l2153_215330


namespace isosceles_triangle_formed_by_lines_l2153_215391

theorem isosceles_triangle_formed_by_lines :
  let P1 := (1/4, 4)
  let P2 := (-3/2, -3)
  let P3 := (2, -3)
  let d12 := ((1/4 + 3/2)^2 + (4 + 3)^2)
  let d13 := ((1/4 - 2)^2 + (4 + 3)^2)
  let d23 := ((-3/2 - 2)^2)
  (d12 = d13) ∧ (d12 ≠ d23) → 
  ∃ (A B C : ℝ × ℝ), 
    A = P1 ∧ B = P2 ∧ C = P3 ∧ 
    ((dist A B = dist A C) ∧ (dist B C ≠ dist A B)) :=
by
  sorry

end isosceles_triangle_formed_by_lines_l2153_215391


namespace hyperbola_eccentricity_l2153_215398

theorem hyperbola_eccentricity (m : ℝ) (h : 0 < m) :
  ∃ e, e = Real.sqrt (1 + m) ∧ e > Real.sqrt 2 → m > 1 :=
by
  sorry

end hyperbola_eccentricity_l2153_215398


namespace water_pressure_on_dam_l2153_215353

theorem water_pressure_on_dam :
  let a := 10 -- length of upper base in meters
  let b := 20 -- length of lower base in meters
  let h := 3 -- height in meters
  let ρg := 9810 -- natural constant for water pressure in N/m^3
  let P := ρg * ((a + 2 * b) * h^2 / 6)
  P = 735750 :=
by
  sorry

end water_pressure_on_dam_l2153_215353


namespace first_group_person_count_l2153_215357

theorem first_group_person_count
  (P : ℕ)
  (h1 : P * 24 * 5 = 30 * 26 * 6) : 
  P = 39 :=
by
  sorry

end first_group_person_count_l2153_215357


namespace panteleimon_twos_l2153_215321

-- Define the variables
variables (P_5 P_4 P_3 P_2 G_5 G_4 G_3 G_2 : ℕ)

-- Define the conditions
def conditions :=
  P_5 + P_4 + P_3 + P_2 = 20 ∧
  G_5 + G_4 + G_3 + G_2 = 20 ∧
  P_5 = G_4 ∧
  P_4 = G_3 ∧
  P_3 = G_2 ∧
  P_2 = G_5 ∧
  (5 * P_5 + 4 * P_4 + 3 * P_3 + 2 * P_2 = 5 * G_5 + 4 * G_4 + 3 * G_3 + 2 * G_2)

-- The proof goal
theorem panteleimon_twos (h : conditions P_5 P_4 P_3 P_2 G_5 G_4 G_3 G_2) : P_2 = 5 :=
sorry

end panteleimon_twos_l2153_215321


namespace B_starts_cycling_after_A_l2153_215379

theorem B_starts_cycling_after_A (t : ℝ) : 10 * t + 20 * (2 - t) = 60 → t = 2 :=
by
  intro h
  sorry

end B_starts_cycling_after_A_l2153_215379


namespace sets_equal_l2153_215375

def A : Set ℝ := {1, Real.sqrt 3, Real.pi}
def B : Set ℝ := {Real.pi, 1, abs (-(Real.sqrt 3))}

theorem sets_equal : A = B :=
by 
  sorry

end sets_equal_l2153_215375


namespace barrels_oil_total_l2153_215397

theorem barrels_oil_total :
  let A := 3 / 4
  let B := A + 1 / 10
  A + B = 8 / 5 := by
  sorry

end barrels_oil_total_l2153_215397


namespace seq_equality_iff_initial_equality_l2153_215381

variable {α : Type*} [AddGroup α]

-- Definition of sequences and their differences
def sequence_diff (u : ℕ → α) (v : ℕ → α) : Prop := ∀ n, (u (n+1) - u n) = (v (n+1) - v n)

-- Main theorem statement
theorem seq_equality_iff_initial_equality (u v : ℕ → α) :
  sequence_diff u v → (∀ n, u n = v n) ↔ (u 1 = v 1) :=
by
  sorry

end seq_equality_iff_initial_equality_l2153_215381


namespace trigonometric_expression_eval_l2153_215300

theorem trigonometric_expression_eval :
  2 * (Real.cos (5 * Real.pi / 16))^6 +
  2 * (Real.sin (11 * Real.pi / 16))^6 +
  (3 * Real.sqrt 2 / 8) = 5 / 4 :=
by
  sorry

end trigonometric_expression_eval_l2153_215300


namespace rational_sum_l2153_215311

theorem rational_sum (x y : ℚ) (h1 : |x| = 5) (h2 : |y| = 2) (h3 : |x - y| = x - y) : x + y = 7 ∨ x + y = 3 := 
sorry

end rational_sum_l2153_215311


namespace solve_polynomial_equation_l2153_215382

theorem solve_polynomial_equation :
  ∃ z, (z^5 + 40 * z^3 + 80 * z - 32 = 0) →
  ∃ x, (x = z + 4) ∧ ((x - 2)^5 + (x - 6)^5 = 32) :=
by
  sorry

end solve_polynomial_equation_l2153_215382


namespace starting_elevation_l2153_215328

variable (rate time final_elevation : ℝ)
variable (h_rate : rate = 10)
variable (h_time : time = 5)
variable (h_final_elevation : final_elevation = 350)

theorem starting_elevation (start_elevation : ℝ) :
  start_elevation = 400 :=
  by
    sorry

end starting_elevation_l2153_215328


namespace tan_x_over_tan_y_plus_tan_y_over_tan_x_l2153_215384

open Real

theorem tan_x_over_tan_y_plus_tan_y_over_tan_x (x y : ℝ) 
  (h1 : sin x / cos y + sin y / cos x = 2) 
  (h2 : cos x / sin y + cos y / sin x = 5) :
  tan x / tan y + tan y / tan x = 10 := 
by
  sorry

end tan_x_over_tan_y_plus_tan_y_over_tan_x_l2153_215384


namespace sum_bases_l2153_215313

theorem sum_bases (R1 R2 : ℕ) (F1 F2 : ℚ)
  (h1 : F1 = (4 * R1 + 5) / (R1 ^ 2 - 1))
  (h2 : F2 = (5 * R1 + 4) / (R1 ^ 2 - 1))
  (h3 : F1 = (3 * R2 + 8) / (R2 ^ 2 - 1))
  (h4 : F2 = (6 * R2 + 1) / (R2 ^ 2 - 1)) :
  R1 + R2 = 19 :=
sorry

end sum_bases_l2153_215313


namespace difference_length_width_l2153_215324

-- Definition of variables and conditions
variables (L W : ℝ)
def hall_width_half_length : Prop := W = (1/2) * L
def hall_area_578 : Prop := L * W = 578

-- Theorem to prove the desired result
theorem difference_length_width (h1 : hall_width_half_length L W) (h2 : hall_area_578 L W) : L - W = 17 :=
sorry

end difference_length_width_l2153_215324


namespace ratio_distance_l2153_215329

-- Definitions based on conditions
def speed_ferry_P : ℕ := 6 -- speed of ferry P in km/h
def time_ferry_P : ℕ := 3 -- travel time of ferry P in hours
def speed_ferry_Q : ℕ := speed_ferry_P + 3 -- speed of ferry Q in km/h
def time_ferry_Q : ℕ := time_ferry_P + 1 -- travel time of ferry Q in hours

-- Calculating the distances
def distance_ferry_P : ℕ := speed_ferry_P * time_ferry_P -- distance covered by ferry P
def distance_ferry_Q : ℕ := speed_ferry_Q * time_ferry_Q -- distance covered by ferry Q

-- Main theorem to prove
theorem ratio_distance (d_P d_Q : ℕ) (h_dP : d_P = distance_ferry_P) (h_dQ : d_Q = distance_ferry_Q) : d_Q / d_P = 2 :=
by
  sorry

end ratio_distance_l2153_215329


namespace triangle_segments_equivalence_l2153_215385

variable {a b c p : ℝ}

theorem triangle_segments_equivalence (h_acute : a^2 + b^2 > c^2) 
  (h_perpendicular : ∃ h: ℝ, h^2 = c^2 - (a - p)^2 ∧ h^2 = b^2 - p^2) :
  a / (c + b) = (c - b) / (a - 2 * p) := by
sorry

end triangle_segments_equivalence_l2153_215385


namespace power_inequality_l2153_215316

theorem power_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  a^5 + b^5 > a^2 * b^3 + a^3 * b^2 :=
sorry

end power_inequality_l2153_215316


namespace second_machine_time_l2153_215331

/-- Given:
1. A first machine can address 600 envelopes in 10 minutes.
2. Both machines together can address 600 envelopes in 4 minutes.
We aim to prove that the second machine alone would take 20/3 minutes to address 600 envelopes. -/
theorem second_machine_time (x : ℝ) 
  (first_machine_rate : ℝ := 600 / 10)
  (combined_rate_needed : ℝ := 600 / 4)
  (second_machine_rate : ℝ := combined_rate_needed - first_machine_rate) 
  (secs_envelope_rate : ℝ := second_machine_rate) 
  (envelopes : ℝ := 600) : 
  x = envelopes / secs_envelope_rate :=
sorry

end second_machine_time_l2153_215331


namespace regular_polygon_sides_l2153_215368

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i < n → (180 * (n - 2) / n) = 174) : n = 60 := by
  sorry

end regular_polygon_sides_l2153_215368


namespace probability_at_least_one_exceeds_one_dollar_l2153_215388

noncomputable def prob_A : ℚ := 2 / 3
noncomputable def prob_B : ℚ := 1 / 2
noncomputable def prob_C : ℚ := 1 / 4

theorem probability_at_least_one_exceeds_one_dollar :
  (1 - ((1 - prob_A) * (1 - prob_B) * (1 - prob_C))) = 7 / 8 :=
by
  -- The proof can be conducted here
  sorry

end probability_at_least_one_exceeds_one_dollar_l2153_215388


namespace ratio_length_to_width_l2153_215307

-- Define the given conditions and values
def width : ℕ := 75
def perimeter : ℕ := 360

-- Define the proof problem statement
theorem ratio_length_to_width (L : ℕ) (P_eq : perimeter = 2 * L + 2 * width) :
  (L / width : ℚ) = 7 / 5 :=
sorry

end ratio_length_to_width_l2153_215307


namespace exists_equal_sum_disjoint_subsets_l2153_215373

-- Define the set and conditions
def is_valid_set (S : Finset ℕ) : Prop :=
  S.card = 15 ∧ ∀ x ∈ S, x ≤ 2020

-- Define the problem statement
theorem exists_equal_sum_disjoint_subsets (S : Finset ℕ) (h : is_valid_set S) :
  ∃ (A B : Finset ℕ), A ⊆ S ∧ B ⊆ S ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by sorry

end exists_equal_sum_disjoint_subsets_l2153_215373


namespace number_of_integers_satisfying_l2153_215320

theorem number_of_integers_satisfying (n : ℤ) : 
    (25 < n^2 ∧ n^2 < 144) → Finset.card (Finset.filter (fun n => 25 < n^2 ∧ n^2 < 144) (Finset.range 25)) = 12 := by
  sorry

end number_of_integers_satisfying_l2153_215320


namespace solve_system_of_equations_l2153_215369

theorem solve_system_of_equations
  (a b c d x y z u : ℝ)
  (h1 : a^3 * x + a^2 * y + a * z + u = 0)
  (h2 : b^3 * x + b^2 * y + b * z + u = 0)
  (h3 : c^3 * x + c^2 * y + c * z + u = 0)
  (h4 : d^3 * x + d^2 * y + d * z + u = 1) :
  x = 1 / ((d - a) * (d - b) * (d - c)) ∧
  y = -(a + b + c) / ((d - a) * (d - b) * (d - c)) ∧
  z = (a * b + b * c + c * a) / ((d - a) * (d - b) * (d - c)) ∧
  u = - (a * b * c) / ((d - a) * (d - b) * (d - c)) :=
sorry

end solve_system_of_equations_l2153_215369


namespace singer_arrangements_l2153_215351

-- Let's assume the 5 singers are represented by the indices 1 through 5

theorem singer_arrangements :
  ∀ (singers : List ℕ) (no_first : ℕ) (must_last : ℕ), 
  singers = [1, 2, 3, 4, 5] →
  no_first ∈ singers →
  must_last ∈ singers →
  no_first ≠ must_last →
  ∃ (arrangements : ℕ),
    arrangements = 18 :=
by
  sorry

end singer_arrangements_l2153_215351


namespace units_digit_of_product_of_first_four_composites_l2153_215359

def units_digit (n : Nat) : Nat := n % 10

theorem units_digit_of_product_of_first_four_composites : 
    units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_product_of_first_four_composites_l2153_215359


namespace solve_y_pos_in_arithmetic_seq_l2153_215306

-- Define the first term as 4
def first_term : ℕ := 4

-- Define the third term as 36
def third_term : ℕ := 36

-- Basing on the properties of an arithmetic sequence, 
-- we solve for the positive second term (y) such that its square equals to 20
theorem solve_y_pos_in_arithmetic_seq : ∃ y : ℝ, y > 0 ∧ y ^ 2 = 20 := by
  sorry

end solve_y_pos_in_arithmetic_seq_l2153_215306


namespace lcm_of_15_18_20_is_180_l2153_215314

theorem lcm_of_15_18_20_is_180 : Nat.lcm (Nat.lcm 15 18) 20 = 180 := by
  sorry

end lcm_of_15_18_20_is_180_l2153_215314


namespace zayne_total_revenue_l2153_215322

-- Defining the constants and conditions
def price_per_bracelet := 5
def deal_price := 8
def initial_bracelets := 30
def revenue_from_five_dollar_sales := 60

-- Calculating number of bracelets sold for $5 each
def bracelets_sold_five_dollars := revenue_from_five_dollar_sales / price_per_bracelet

-- Calculating remaining bracelets after selling some for $5 each
def remaining_bracelets := initial_bracelets - bracelets_sold_five_dollars

-- Calculating number of pairs sold at two for $8
def pairs_sold := remaining_bracelets / 2

-- Calculating revenue from selling pairs
def revenue_from_deal_sales := pairs_sold * deal_price

-- Total revenue calculation
def total_revenue := revenue_from_five_dollar_sales + revenue_from_deal_sales

-- Theorem to prove the total revenue is $132
theorem zayne_total_revenue : total_revenue = 132 := by
  sorry

end zayne_total_revenue_l2153_215322


namespace students_failed_exam_l2153_215383

def total_students : ℕ := 740
def percent_passed : ℝ := 0.35
def percent_failed : ℝ := 1 - percent_passed
def failed_students : ℝ := percent_failed * total_students

theorem students_failed_exam : failed_students = 481 := 
by sorry

end students_failed_exam_l2153_215383


namespace geometric_progression_fifth_term_sum_l2153_215345

def gp_sum_fifth_term
    (p q : ℝ)
    (hpq_sum : p + q = 3)
    (hpq_6th : p^5 + q^5 = 573) : ℝ :=
p^4 + q^4

theorem geometric_progression_fifth_term_sum :
    ∃ p q : ℝ, p + q = 3 ∧ p^5 + q^5 = 573 ∧ gp_sum_fifth_term p q (by sorry) (by sorry) = 161 :=
by
  sorry

end geometric_progression_fifth_term_sum_l2153_215345


namespace distance_between_towns_l2153_215364

theorem distance_between_towns (D S : ℝ) (h1 : D = S * 3) (h2 : 200 = S * 5) : D = 120 :=
by
  sorry

end distance_between_towns_l2153_215364


namespace total_length_of_sticks_l2153_215380

-- Definitions based on conditions
def num_sticks := 30
def length_per_stick := 25
def overlap := 6
def effective_length_per_stick := length_per_stick - overlap

-- Theorem statement
theorem total_length_of_sticks : num_sticks * effective_length_per_stick - effective_length_per_stick + length_per_stick = 576 := sorry

end total_length_of_sticks_l2153_215380


namespace dice_probability_l2153_215361

theorem dice_probability (p : ℚ) (h : p = (1 / 42)) : 
  p = 0.023809523809523808 := 
sorry

end dice_probability_l2153_215361


namespace rex_lesson_schedule_l2153_215386

-- Define the total lessons and weeks
def total_lessons : ℕ := 40
def weeks_completed : ℕ := 6
def weeks_remaining : ℕ := 4

-- Define the proof statement
theorem rex_lesson_schedule : (weeks_completed + weeks_remaining) * 4 = total_lessons := by
  -- Proof placeholder, to be filled in 
  sorry

end rex_lesson_schedule_l2153_215386


namespace annual_pension_l2153_215396

theorem annual_pension (c d r s x k : ℝ) (hc : c ≠ 0) (hd : d ≠ c)
  (h1 : k * (x + c) ^ (3 / 2) = k * x ^ (3 / 2) + r)
  (h2 : k * (x + d) ^ (3 / 2) = k * x ^ (3 / 2) + s) :
  k * x ^ (3 / 2) = 4 * r^2 / (9 * c^2) :=
by
  sorry

end annual_pension_l2153_215396


namespace find_A_minus_B_l2153_215378

variables (A B : ℝ)

-- Define the conditions
def condition1 : Prop := B + A + B = 814.8
def condition2 : Prop := B = A / 10

-- Statement to prove
theorem find_A_minus_B (h1 : condition1 A B) (h2 : condition2 A B) : A - B = 611.1 :=
sorry

end find_A_minus_B_l2153_215378


namespace marion_score_is_correct_l2153_215339

-- Definition of the problem conditions
def exam_total_items := 40
def ella_incorrect_answers := 4

-- Calculate Ella's score
def ella_score := exam_total_items - ella_incorrect_answers

-- Calculate half of Ella's score
def half_ella_score := ella_score / 2

-- Marion's score is 6 more than half of Ella's score
def marion_score := half_ella_score + 6

-- The theorem we need to prove
theorem marion_score_is_correct : marion_score = 24 := by
  sorry

end marion_score_is_correct_l2153_215339


namespace inequality_proof_l2153_215302

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b)) ≥ 1 / 2 * (a + b + c) := 
by
  sorry

end inequality_proof_l2153_215302


namespace distance_between_A_and_B_l2153_215336

def scale : ℕ := 20000
def map_distance : ℕ := 6
def actual_distance_cm : ℕ := scale * map_distance
def actual_distance_m : ℕ := actual_distance_cm / 100

theorem distance_between_A_and_B : actual_distance_m = 1200 := by
  sorry

end distance_between_A_and_B_l2153_215336


namespace fermats_little_theorem_l2153_215340

theorem fermats_little_theorem (p : ℕ) (a : ℤ) (hp : Nat.Prime p) : 
  (a^p - a) % p = 0 :=
sorry

end fermats_little_theorem_l2153_215340


namespace find_n_l2153_215334

theorem find_n (n : ℕ) (h : n * Nat.factorial n + Nat.factorial n = 5040) : n = 6 :=
sorry

end find_n_l2153_215334


namespace min_value_expression_l2153_215301

theorem min_value_expression (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : x * y * z = 1) :
  (x^2 + 8 * x * y + 25 * y^2 + 16 * y * z + 9 * z^2) ≥ 403 / 9 := by
  sorry

end min_value_expression_l2153_215301


namespace diagonals_in_nine_sided_polygon_l2153_215372

-- Define the conditions
def sides : ℕ := 9
def right_angles : ℕ := 2

-- The function to calculate the number of diagonals for a polygon
def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- The theorem to prove
theorem diagonals_in_nine_sided_polygon : number_of_diagonals sides = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l2153_215372


namespace product_evaluation_l2153_215315

-- Define the conditions and the target expression
def product (a : ℕ) : ℕ := (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a

-- Main theorem statement
theorem product_evaluation : product 7 = 5040 :=
by
  -- Lean usually requires some import from the broader Mathlib to support arithmetic simplifications
  sorry

end product_evaluation_l2153_215315


namespace cab_speed_fraction_l2153_215374

def usual_time := 30 -- The usual time of the journey in minutes
def delay_time := 6   -- The delay time in minutes
def usual_speed : ℝ := sorry -- Placeholder for the usual speed
def reduced_speed : ℝ := sorry -- Placeholder for the reduced speed

-- Given the conditions:
-- 1. The usual time for the cab to cover the journey is 30 minutes.
-- 2. The cab is 6 minutes late when walking at a reduced speed.
-- Prove that the fraction of the cab's usual speed it is walking at is 5/6

theorem cab_speed_fraction : (reduced_speed / usual_speed) = (5 / 6) :=
sorry

end cab_speed_fraction_l2153_215374


namespace number_of_possible_schedules_l2153_215348

-- Define the six teams
inductive Team : Type
| A | B | C | D | E | F

open Team

-- Define the function to get the number of different schedules possible
noncomputable def number_of_schedules : ℕ := 70

-- Define the theorem statement
theorem number_of_possible_schedules (teams : Finset Team) (play_games : Team → Finset Team) (h : teams.card = 6) 
  (h2 : ∀ t ∈ teams, (play_games t).card = 3 ∧ ∀ t' ∈ (play_games t), t ≠ t') : 
  number_of_schedules = 70 :=
by sorry

end number_of_possible_schedules_l2153_215348


namespace parking_lot_problem_l2153_215352

theorem parking_lot_problem :
  let total_spaces := 50
  let cars := 2
  let total_ways := total_spaces * (total_spaces - 1)
  let adjacent_ways := (total_spaces - 1) * 2
  let valid_ways := total_ways - adjacent_ways
  valid_ways = 2352 :=
by
  sorry

end parking_lot_problem_l2153_215352


namespace line_inclination_angle_l2153_215367

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x + (Real.sqrt 3) * y - 1 = 0

-- Define the condition of inclination angle in radians
def inclination_angle (θ : ℝ) : Prop := θ = Real.arctan (-1 / Real.sqrt 3) + Real.pi

-- The theorem to prove the inclination angle of the line
theorem line_inclination_angle (x y θ : ℝ) (h : line_eq x y) : inclination_angle θ :=
by
  sorry

end line_inclination_angle_l2153_215367


namespace count_valid_numbers_l2153_215317

theorem count_valid_numbers : 
  let count_A := 10 
  let count_B := 2 
  count_A * count_B = 20 :=
by 
  let count_A := 10
  let count_B := 2
  have : count_A * count_B = 20 := by norm_num
  exact this

end count_valid_numbers_l2153_215317


namespace exp_eval_l2153_215308

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end exp_eval_l2153_215308


namespace smallest_prime_divisor_of_sum_l2153_215358

theorem smallest_prime_divisor_of_sum (a b : ℕ) (h1 : a = 3^19) (h2 : b = 11^13) (h3 : Odd a) (h4 : Odd b) : 
  ∃ p : ℕ, Prime p ∧ p ∣ (a + b) ∧ p = 2 := 
by {
  sorry
}

end smallest_prime_divisor_of_sum_l2153_215358


namespace continuous_func_unique_l2153_215305

theorem continuous_func_unique (f : ℝ → ℝ) (hf_cont : Continuous f)
  (hf_eqn : ∀ x : ℝ, f x + f (x^2) = 2) :
  ∀ x : ℝ, f x = 1 :=
by
  sorry

end continuous_func_unique_l2153_215305


namespace dogs_neither_long_furred_nor_brown_l2153_215362

theorem dogs_neither_long_furred_nor_brown :
  (∀ (total_dogs long_furred_dogs brown_dogs both_long_furred_and_brown neither_long_furred_nor_brown : ℕ),
     total_dogs = 45 →
     long_furred_dogs = 26 →
     brown_dogs = 22 →
     both_long_furred_and_brown = 11 →
     neither_long_furred_nor_brown = total_dogs - (long_furred_dogs + brown_dogs - both_long_furred_and_brown) → 
     neither_long_furred_nor_brown = 8) :=
by
  intros total_dogs long_furred_dogs brown_dogs both_long_furred_and_brown neither_long_furred_nor_brown
  sorry

end dogs_neither_long_furred_nor_brown_l2153_215362


namespace lucas_150_mod_9_l2153_215346

-- Define the Lucas sequence recursively
def lucas (n : ℕ) : ℕ :=
  match n with
  | 0 => 1 -- Since L_1 in the sequence provided is actually the first Lucas number (index starts from 1)
  | 1 => 3
  | (n + 2) => lucas n + lucas (n + 1)

-- Define the theorem for the remainder when the 150th term is divided by 9
theorem lucas_150_mod_9 : lucas 149 % 9 = 3 := by
  sorry

end lucas_150_mod_9_l2153_215346


namespace percent_profit_is_25_percent_l2153_215338

theorem percent_profit_is_25_percent
  (CP SP : ℝ)
  (h : 75 * (CP - 0.05 * CP) = 60 * SP) :
  let profit := SP - (0.95 * CP)
  let percent_profit := (profit / (0.95 * CP)) * 100
  percent_profit = 25 :=
by
  sorry

end percent_profit_is_25_percent_l2153_215338


namespace third_median_length_l2153_215376

-- Proposition stating the problem with conditions and the conclusion
theorem third_median_length (m1 m2 : ℝ) (area : ℝ) (h1 : m1 = 4) (h2 : m2 = 5) (h_area : area = 10 * Real.sqrt 3) : 
  ∃ m3 : ℝ, m3 = 3 * Real.sqrt 10 :=
by
  sorry  -- proof is not included

end third_median_length_l2153_215376


namespace problem_statement_l2153_215344

-- Given conditions
variables {p q r t n : ℕ}

axiom prime_p : Nat.Prime p
axiom prime_q : Nat.Prime q
axiom prime_r : Nat.Prime r

axiom nat_n : n ≥ 1
axiom nat_t : t ≥ 1

axiom eqn1 : p^2 + q * t = (p + t)^n
axiom eqn2 : p^2 + q * r = t^4

-- Statement to prove
theorem problem_statement : n < 3 ∧ (p = 2 ∧ q = 7 ∧ r = 11 ∧ t = 3 ∧ n = 2) :=
by
  sorry

end problem_statement_l2153_215344


namespace max_diff_x_y_l2153_215312

theorem max_diff_x_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : 
  x - y ≤ Real.sqrt (4 / 3) := 
by
  sorry

end max_diff_x_y_l2153_215312


namespace irwin_basketball_l2153_215333

theorem irwin_basketball (A B C D : ℕ) (h1 : C = 2) (h2 : 2^A * 5^B * 11^C * 13^D = 2420) : A = 2 :=
by
  sorry

end irwin_basketball_l2153_215333


namespace find_a_l2153_215356

noncomputable def f (x : ℝ) : ℝ := x^2 + 9
noncomputable def g (x : ℝ) : ℝ := x^2 - 5

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 9) : a = Real.sqrt 5 :=
by
  sorry

end find_a_l2153_215356


namespace company_salary_decrease_l2153_215360

variables {E S : ℝ} -- Let the initial number of employees be E and the initial average salary be S

theorem company_salary_decrease :
  (0.8 * E * (1.15 * S)) / (E * S) = 0.92 := 
by
  -- The proof will go here, but we use sorry to skip it for now
  sorry

end company_salary_decrease_l2153_215360


namespace altitude_line_eq_circumcircle_eq_l2153_215393

noncomputable def point := ℝ × ℝ

noncomputable def A : point := (5, 1)
noncomputable def B : point := (1, 3)
noncomputable def C : point := (4, 4)

theorem altitude_line_eq : ∃ (k b : ℝ), (k = 2 ∧ b = -4) ∧ (∀ x y : ℝ, y = k * x + b ↔ 2 * x - y - 4 = 0) :=
sorry

theorem circumcircle_eq : ∃ (h k r : ℝ), (h = 3 ∧ k = 2 ∧ r = 5) ∧ (∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r ↔ (x - 3)^2 + (y - 2)^2 = 5) :=
sorry

end altitude_line_eq_circumcircle_eq_l2153_215393


namespace parabola_hyperbola_tangent_l2153_215377

noncomputable def parabola (x : ℝ) : ℝ := x^2 + 5
noncomputable def hyperbola (x y : ℝ) (m : ℝ) : ℝ := y^2 - m * x^2 - 1

theorem parabola_hyperbola_tangent (m : ℝ) :
(∃ x y : ℝ, y = parabola x ∧ hyperbola x y m = 0) ↔ 
m = 10 + 2 * Real.sqrt 6 ∨ m = 10 - 2 * Real.sqrt 6 := by
  sorry

end parabola_hyperbola_tangent_l2153_215377


namespace probability_is_correct_l2153_215394

variables (total_items truckA_first_class truckA_second_class truckB_first_class truckB_second_class brokenA brokenB remaining_items : ℕ)

-- Setting up the problem according to the given conditions
def conditions := (total_items = 10) ∧ 
                  (truckA_first_class = 2) ∧ (truckA_second_class = 2) ∧ 
                  (truckB_first_class = 4) ∧ (truckB_second_class = 2) ∧ 
                  (brokenA = 1) ∧ (brokenB = 1) ∧
                  (remaining_items = 8)

-- Calculating the probability of selecting a first-class item from the remaining items
def probability_of_first_class : ℚ :=
  1/3 * 1/2 + 1/6 * 5/8 + 1/3 * 5/8 + 1/6 * 3/4

-- The theorem to be proved
theorem probability_is_correct : 
  conditions total_items truckA_first_class truckA_second_class truckB_first_class truckB_second_class brokenA brokenB remaining_items →
  probability_of_first_class = 29/48 :=
sorry

end probability_is_correct_l2153_215394


namespace train_length_is_correct_l2153_215323

noncomputable def length_of_train (t : ℝ) (v_train : ℝ) (v_man : ℝ) : ℝ :=
  let relative_speed : ℝ := (v_train - v_man) * (5/18)
  relative_speed * t

theorem train_length_is_correct :
  length_of_train 23.998 63 3 = 400 :=
by
  -- Placeholder for the proof
  sorry

end train_length_is_correct_l2153_215323


namespace emily_seeds_start_with_l2153_215342

-- Define the conditions as hypotheses
variables (big_garden_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ)

-- Conditions: Emily planted 29 seeds in the big garden and 4 seeds in each of her 3 small gardens.
def emily_conditions := big_garden_seeds = 29 ∧ small_gardens = 3 ∧ seeds_per_small_garden = 4

-- Define the statement to prove the total number of seeds Emily started with
theorem emily_seeds_start_with (h : emily_conditions big_garden_seeds small_gardens seeds_per_small_garden) : 
(big_garden_seeds + small_gardens * seeds_per_small_garden) = 41 :=
by
  -- Assuming the proof follows logically from conditions
  sorry

end emily_seeds_start_with_l2153_215342


namespace z_in_fourth_quadrant_l2153_215335

def complex_quadrant (re im : ℤ) : String :=
  if re > 0 ∧ im > 0 then "First Quadrant"
  else if re < 0 ∧ im > 0 then "Second Quadrant"
  else if re < 0 ∧ im < 0 then "Third Quadrant"
  else if re > 0 ∧ im < 0 then "Fourth Quadrant"
  else "Axis"

theorem z_in_fourth_quadrant : complex_quadrant 2 (-3) = "Fourth Quadrant" :=
by
  sorry

end z_in_fourth_quadrant_l2153_215335


namespace T_n_formula_l2153_215370

def a_n (n : ℕ) : ℕ := 3 * n - 1
def b_n (n : ℕ) : ℕ := 2 ^ n
def T_n (n : ℕ) : ℕ := (Finset.range n).sum (λ k => a_n (k + 1) * b_n (k + 1))

theorem T_n_formula (n : ℕ) : T_n n = 8 - 8 * 2 ^ n + 3 * n * 2 ^ (n + 1) :=
by 
  sorry

end T_n_formula_l2153_215370


namespace person_A_money_left_l2153_215365

-- We define the conditions and question in terms of Lean types.
def initial_money_ratio : ℚ := 7 / 6
def money_spent_A : ℚ := 50
def money_spent_B : ℚ := 60
def final_money_ratio : ℚ := 3 / 2
def x : ℚ := 30

-- The theorem to prove the amount of money left by person A
theorem person_A_money_left 
  (init_ratio : initial_money_ratio = 7 / 6)
  (spend_A : money_spent_A = 50)
  (spend_B : money_spent_B = 60)
  (final_ratio : final_money_ratio = 3 / 2)
  (hx : x = 30) : 3 * x = 90 := by 
  sorry

end person_A_money_left_l2153_215365


namespace ellipse_problem_l2153_215304

-- Definitions of conditions from the problem
def F1 := (0, 0)
def F2 := (6, 0)
def ellipse_equation (x y h k a b : ℝ) := ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1

-- The main statement to be proved
theorem ellipse_problem :
  let h := 3
  let k := 0
  let a := 5
  let c := 3
  let b := Real.sqrt (a^2 - c^2)
  h + k + a + b = 12 :=
by
  -- Proof would go here
  sorry

end ellipse_problem_l2153_215304


namespace large_pizza_cost_l2153_215326

theorem large_pizza_cost
  (small_side : ℕ) (small_cost : ℝ) (large_side : ℕ) (friend_money : ℝ) (extra_square_inches : ℝ)
  (A_small : small_side * small_side = 196)
  (A_large : large_side * large_side = 441)
  (small_cost_per_sq_in : 196 / small_cost = 19.6)
  (individual_area : (30 / small_cost) * 196 = 588)
  (total_individual_area : 2 * 588 = 1176)
  (pool_area_eq : (60 / (441 / x)) = 1225)
  : (x = 21.6) := 
by
  sorry

end large_pizza_cost_l2153_215326


namespace processing_times_maximum_salary_l2153_215390

def monthly_hours : ℕ := 8 * 25
def base_salary : ℕ := 800
def earnings_per_A : ℕ := 16
def earnings_per_B : ℕ := 12

theorem processing_times :
  ∃ (x y : ℕ),
    x + 3 * y = 5 ∧ 2 * x + 5 * y = 9 ∧ x = 2 ∧ y = 1 :=
by
  sorry

theorem maximum_salary :
  ∃ (a b W : ℕ),
    a ≥ 50 ∧ 
    b = monthly_hours - 2 * a ∧ 
    W = base_salary + earnings_per_A * a + earnings_per_B * b ∧ 
    a = 50 ∧ 
    b = 100 ∧ 
    W = 2800 :=
by
  sorry

end processing_times_maximum_salary_l2153_215390


namespace new_person_weight_l2153_215309

theorem new_person_weight (avg_increase : ℝ) (num_persons : ℕ) (initial_person_weight : ℝ) 
  (weight_increase : ℝ) (final_person_weight : ℝ) : 
  avg_increase = 2.5 ∧ num_persons = 8 ∧ initial_person_weight = 65 ∧ 
  weight_increase = num_persons * avg_increase ∧ final_person_weight = initial_person_weight + weight_increase 
  → final_person_weight = 85 :=
by 
  intros h
  sorry

end new_person_weight_l2153_215309


namespace wrapping_paper_area_correct_l2153_215354

structure Box :=
  (l : ℝ)  -- length of the box
  (w : ℝ)  -- width of the box
  (h : ℝ)  -- height of the box
  (h_lw : l > w)  -- condition that length is greater than width

def wrapping_paper_area (b : Box) : ℝ :=
  3 * (b.l + b.w) * b.h

theorem wrapping_paper_area_correct (b : Box) : 
  wrapping_paper_area b = 3 * (b.l + b.w) * b.h :=
sorry

end wrapping_paper_area_correct_l2153_215354


namespace sequence_a4_eq_neg3_l2153_215318

theorem sequence_a4_eq_neg3 (a : ℕ → ℤ) (h1 : a 1 = 3) (h2 : a 2 = 6)
  (h_rec : ∀ n, a (n + 2) = a (n + 1) - a n) : a 4 = -3 :=
by
  sorry

end sequence_a4_eq_neg3_l2153_215318


namespace smallest_four_digit_divisible_by_8_with_3_even_1_odd_l2153_215399

theorem smallest_four_digit_divisible_by_8_with_3_even_1_odd : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ n % 8 = 0 ∧ 
  (∃ d1 d2 d3 d4, n = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ 
    (d1 % 2 = 0) ∧ (d2 % 2 = 0 ∨ d2 % 2 ≠ 0) ∧ 
    (d3 % 2 = 0) ∧ (d4 % 2 = 0 ∨ d4 % 2 ≠ 0) ∧ 
    (d2 % 2 ≠ 0 ∨ d4 % 2 ≠ 0) ) ∧ n = 1248 :=
by
  sorry

end smallest_four_digit_divisible_by_8_with_3_even_1_odd_l2153_215399


namespace agatha_initial_money_60_l2153_215389

def Agatha_initial_money (spent_frame : ℕ) (spent_front_wheel: ℕ) (left_over: ℕ) : ℕ :=
  spent_frame + spent_front_wheel + left_over

theorem agatha_initial_money_60 :
  Agatha_initial_money 15 25 20 = 60 :=
by
  -- This line assumes $15 on frame, $25 on wheel, $20 left translates to a total of $60.
  sorry

end agatha_initial_money_60_l2153_215389


namespace negation_of_universal_proposition_l2153_215319

theorem negation_of_universal_proposition {f : ℝ → ℝ} :
  (¬ (∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0)) ↔
  ∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0 :=
by
  sorry

end negation_of_universal_proposition_l2153_215319


namespace power_function_is_odd_l2153_215327

open Function

noncomputable def power_function (a : ℝ) (b : ℝ) : ℝ → ℝ := λ x => (a - 1) * x^b

theorem power_function_is_odd (a b : ℝ) (h : power_function a b a = 1 / 8)
  :  a = 2 ∧ b = -3 → (∀ x : ℝ, power_function a b (-x) = -power_function a b x) :=
by
  intro ha hb
  -- proofs can be filled later with details
  sorry

end power_function_is_odd_l2153_215327


namespace problem_l2153_215355

def p (x y : Int) : Int :=
  if x ≥ 0 ∧ y ≥ 0 then x * y
  else if x < 0 ∧ y < 0 then x - 2 * y
  else if x ≥ 0 ∧ y < 0 then 2 * x + 3 * y
  else if x < 0 ∧ y ≥ 0 then x + 3 * y
  else 3 * x + y

theorem problem : p (p 2 (-3)) (p (-1) 4) = 28 := by
  sorry

end problem_l2153_215355


namespace range_of_a_l2153_215341

open Function

theorem range_of_a (f : ℝ → ℝ) (h_even : ∀ x : ℝ, f x = f (-x)) (h_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ 0 → f x₁ > f x₂) (a : ℝ) (h_gt : f a > f 2) : a < -2 ∨ a > 2 :=
  sorry

end range_of_a_l2153_215341


namespace total_cost_of_pens_and_notebooks_l2153_215349

theorem total_cost_of_pens_and_notebooks (a b : ℝ) : 5 * a + 8 * b = 5 * a + 8 * b := 
by 
  sorry

end total_cost_of_pens_and_notebooks_l2153_215349


namespace final_statement_l2153_215310

variable (x : ℝ)

def seven_elevenths_of_five_thirteenths_eq_48 (x : ℝ) :=
  (7/11 : ℝ) * (5/13 : ℝ) * x = 48

def solve_for_x (x : ℝ) : Prop :=
  seven_elevenths_of_five_thirteenths_eq_48 x → x = 196

def calculate_315_percent_of_x (x : ℝ) : Prop :=
  solve_for_x x → 3.15 * x = 617.4

theorem final_statement : calculate_315_percent_of_x x :=
sorry  -- Proof omitted

end final_statement_l2153_215310


namespace joe_paint_problem_l2153_215303

theorem joe_paint_problem (f : ℝ) (h₁ : 360 * f + (1 / 6) * (360 - 360 * f) = 135) : f = 1 / 4 := 
by
  sorry

end joe_paint_problem_l2153_215303


namespace pairs_of_polygons_with_angle_ratio_l2153_215347

theorem pairs_of_polygons_with_angle_ratio :
  ∃ n, n = 2 ∧ (∀ {k r : ℕ}, (k > 2 ∧ r > 2) → 
  (4 * (180 * r - 360) = 3 * (180 * k - 360) →
  ((k = 3 ∧ r = 18) ∨ (k = 2 ∧ r = 6)))) :=
by
  -- The proof should be provided here, but we skip it
  sorry

end pairs_of_polygons_with_angle_ratio_l2153_215347
