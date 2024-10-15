import Mathlib

namespace NUMINAMATH_GPT_polynomial_division_properties_l1064_106469

open Polynomial

noncomputable def g : Polynomial ℝ := 3 * X^4 + 9 * X^3 - 7 * X^2 + 2 * X + 5
noncomputable def e : Polynomial ℝ := X^2 + 2 * X - 3

theorem polynomial_division_properties (s t : Polynomial ℝ) (h : g = s * e + t) (h_deg : t.degree < e.degree) :
  s.eval 1 + t.eval (-1) = -22 :=
sorry

end NUMINAMATH_GPT_polynomial_division_properties_l1064_106469


namespace NUMINAMATH_GPT_broken_seashells_l1064_106465

-- Define the total number of seashells Tom found
def total_seashells : ℕ := 7

-- Define the number of unbroken seashells
def unbroken_seashells : ℕ := 3

-- Prove that the number of broken seashells equals 4
theorem broken_seashells : total_seashells - unbroken_seashells = 4 := by
  sorry

end NUMINAMATH_GPT_broken_seashells_l1064_106465


namespace NUMINAMATH_GPT_tens_digit_of_9_pow_1801_l1064_106416

theorem tens_digit_of_9_pow_1801 : 
  ∀ n : ℕ, (9 ^ (1801) % 100) / 10 % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_tens_digit_of_9_pow_1801_l1064_106416


namespace NUMINAMATH_GPT_price_increase_solution_l1064_106480

variable (x : ℕ)

def initial_profit := 10
def initial_sales := 500
def price_increase_effect := 20
def desired_profit := 6000

theorem price_increase_solution :
  ((initial_sales - price_increase_effect * x) * (initial_profit + x) = desired_profit) → (x = 5) :=
by
  sorry

end NUMINAMATH_GPT_price_increase_solution_l1064_106480


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l1064_106410

theorem right_triangle_hypotenuse (a b : ℝ) (m_a m_b : ℝ)
    (h1 : m_a = Real.sqrt (b^2 + (a / 2)^2))
    (h2 : m_b = Real.sqrt (a^2 + (b / 2)^2))
    (h3 : m_a = Real.sqrt 30)
    (h4 : m_b = 6) :
  Real.sqrt (4 * (a^2 + b^2)) = 2 * Real.sqrt 52.8 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_l1064_106410


namespace NUMINAMATH_GPT_convert_decimal_to_fraction_l1064_106419

theorem convert_decimal_to_fraction : (2.24 : ℚ) = 56 / 25 := by
  sorry

end NUMINAMATH_GPT_convert_decimal_to_fraction_l1064_106419


namespace NUMINAMATH_GPT_hexagon_monochromatic_triangles_l1064_106417

theorem hexagon_monochromatic_triangles :
  let hexagon_edges := 15 -- $\binom{6}{2}$
  let monochromatic_tri_prob := (1 / 3) -- Prob of one triangle being monochromatic
  let combinations := 20 -- $\binom{6}{3}$, total number of triangles in K_6
  let exactly_two_monochromatic := (combinations.choose 2) * (monochromatic_tri_prob ^ 2) * ((2 / 3) ^ 18)
  (exactly_two_monochromatic = 49807360 / 3486784401) := sorry

end NUMINAMATH_GPT_hexagon_monochromatic_triangles_l1064_106417


namespace NUMINAMATH_GPT_max_gcd_lcm_l1064_106407

theorem max_gcd_lcm (a b c : ℕ) (h : Nat.gcd (Nat.lcm a b) c * Nat.lcm (Nat.gcd a b) c = 200) :
  ∃ x : ℕ, x = Nat.gcd (Nat.lcm a b) c ∧ ∀ y : ℕ, Nat.gcd (Nat.lcm a b) c ≤ 10 :=
sorry

end NUMINAMATH_GPT_max_gcd_lcm_l1064_106407


namespace NUMINAMATH_GPT_permutations_count_l1064_106446

-- Define the conditions
variable (n : ℕ)
variable (a : Fin n → ℕ)

-- Define the main proposition
theorem permutations_count (hn : 2 ≤ n) (h_perm : ∀ k : Fin n, a k ≥ k.val - 2) :
  ∃! L, L = 2 * 3 ^ (n - 2) :=
by
  sorry

end NUMINAMATH_GPT_permutations_count_l1064_106446


namespace NUMINAMATH_GPT_chadSavingsIsCorrect_l1064_106478

noncomputable def chadSavingsAfterTaxAndConversion : ℝ :=
  let euroToUsd := 1.20
  let poundToUsd := 1.40
  let euroIncome := 600 * euroToUsd
  let poundIncome := 250 * poundToUsd
  let dollarIncome := 150 + 150
  let totalIncome := euroIncome + poundIncome + dollarIncome
  let taxRate := 0.10
  let taxedIncome := totalIncome * (1 - taxRate)
  let savingsRate := if taxedIncome ≤ 1000 then 0.20
                     else if taxedIncome ≤ 2000 then 0.30
                     else if taxedIncome ≤ 3000 then 0.40
                     else 0.50
  let savings := taxedIncome * savingsRate
  savings

theorem chadSavingsIsCorrect : chadSavingsAfterTaxAndConversion = 369.90 := by
  sorry

end NUMINAMATH_GPT_chadSavingsIsCorrect_l1064_106478


namespace NUMINAMATH_GPT_inequality_proof_l1064_106443

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  abc ≥ (a + b + c) / ((1 / a^2) + (1 / b^2) + (1 / c^2)) ∧
  (a + b + c) / ((1 / a^2) + (1 / b^2) + (1 / c^2)) ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1064_106443


namespace NUMINAMATH_GPT_set_intersection_eq_l1064_106451

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | -2 < x ∧ x < 2}

-- The proof statement
theorem set_intersection_eq :
  A ∩ B = A :=
sorry

end NUMINAMATH_GPT_set_intersection_eq_l1064_106451


namespace NUMINAMATH_GPT_remainder_of_number_of_minimally_intersecting_triples_l1064_106418

noncomputable def number_of_minimally_intersecting_triples : Nat :=
  let n := (8 * 7 * 6) * (4 ^ 5)
  n % 1000

theorem remainder_of_number_of_minimally_intersecting_triples :
  number_of_minimally_intersecting_triples = 64 := by
  sorry

end NUMINAMATH_GPT_remainder_of_number_of_minimally_intersecting_triples_l1064_106418


namespace NUMINAMATH_GPT_paint_after_third_day_l1064_106495

def initial_paint := 2
def paint_used_first_day (x : ℕ) := (1 / 2) * x
def remaining_after_first_day (x : ℕ) := x - paint_used_first_day x
def paint_used_second_day (y : ℕ) := (1 / 4) * y
def remaining_after_second_day (y : ℕ) := y - paint_used_second_day y
def paint_used_third_day (z : ℕ) := (1 / 3) * z
def remaining_after_third_day (z : ℕ) := z - paint_used_third_day z

theorem paint_after_third_day :
  remaining_after_third_day 
    (remaining_after_second_day 
      (remaining_after_first_day initial_paint)) = initial_paint / 2 := 
  by
  sorry

end NUMINAMATH_GPT_paint_after_third_day_l1064_106495


namespace NUMINAMATH_GPT_M_diff_N_eq_l1064_106458

noncomputable def A_diff_B (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∉ B }

noncomputable def M : Set ℝ := { x | -3 ≤ x ∧ x ≤ 1 }

noncomputable def N : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem M_diff_N_eq : A_diff_B M N = { x | -3 ≤ x ∧ x < 0 } :=
by
  sorry

end NUMINAMATH_GPT_M_diff_N_eq_l1064_106458


namespace NUMINAMATH_GPT_max_possible_value_of_y_l1064_106415

theorem max_possible_value_of_y (x y : ℤ) (h : x * y + 3 * x + 2 * y = 4) : y ≤ 7 :=
sorry

end NUMINAMATH_GPT_max_possible_value_of_y_l1064_106415


namespace NUMINAMATH_GPT_friendly_sequences_exist_l1064_106488

theorem friendly_sequences_exist :
  ∃ (a b : ℕ → ℕ), 
    (∀ n, a n = 2^(n-1)) ∧ 
    (∀ n, b n = 2*n - 1) ∧ 
    (∀ k : ℕ, ∃ (i j : ℕ), k = a i * b j) :=
by
  sorry

end NUMINAMATH_GPT_friendly_sequences_exist_l1064_106488


namespace NUMINAMATH_GPT_distance_light_travels_500_years_l1064_106466

def distance_light_travels_one_year : ℝ := 5.87e12
def years : ℕ := 500

theorem distance_light_travels_500_years :
  distance_light_travels_one_year * years = 2.935e15 := 
sorry

end NUMINAMATH_GPT_distance_light_travels_500_years_l1064_106466


namespace NUMINAMATH_GPT_cost_per_pizza_is_12_l1064_106409

def numberOfPeople := 15
def peoplePerPizza := 3
def earningsPerNight := 4
def nightsBabysitting := 15

-- We aim to prove that the cost per pizza is $12
theorem cost_per_pizza_is_12 : 
  (earningsPerNight * nightsBabysitting) / (numberOfPeople / peoplePerPizza) = 12 := 
by 
  sorry

end NUMINAMATH_GPT_cost_per_pizza_is_12_l1064_106409


namespace NUMINAMATH_GPT_nat_pair_solution_l1064_106404

theorem nat_pair_solution (x y : ℕ) : 7^x - 3 * 2^y = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
by
  sorry

end NUMINAMATH_GPT_nat_pair_solution_l1064_106404


namespace NUMINAMATH_GPT_solve_arrangement_equation_l1064_106462

def arrangement_numeral (x : ℕ) : ℕ :=
  x * (x - 1) * (x - 2)

theorem solve_arrangement_equation (x : ℕ) (h : 3 * (arrangement_numeral x)^3 = 2 * (arrangement_numeral (x + 1))^2 + 6 * (arrangement_numeral x)^2) : x = 5 := 
sorry

end NUMINAMATH_GPT_solve_arrangement_equation_l1064_106462


namespace NUMINAMATH_GPT_choir_average_age_l1064_106425

-- Each condition as a definition in Lean 4
def avg_age_females := 28
def num_females := 12
def avg_age_males := 32
def num_males := 18
def total_people := num_females + num_males

-- The total sum of ages calculated from the given conditions
def sum_ages_females := avg_age_females * num_females
def sum_ages_males := avg_age_males * num_males
def total_sum_ages := sum_ages_females + sum_ages_males

-- The final proof statement to be proved
theorem choir_average_age : 
  (total_sum_ages : ℝ) / (total_people : ℝ) = 30.4 := by
  sorry

end NUMINAMATH_GPT_choir_average_age_l1064_106425


namespace NUMINAMATH_GPT_tourists_number_l1064_106434

theorem tourists_number (m : ℕ) (k l : ℤ) (n : ℕ) (hn : n = 23) (hm1 : 2 * m ≡ 1 [MOD n]) (hm2 : 3 * m ≡ 13 [MOD n]) (hn_gt_13 : n > 13) : n = 23 := 
by
  sorry

end NUMINAMATH_GPT_tourists_number_l1064_106434


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l1064_106455

-- Part a
def can_ratings_increase_after_first_migration (QA_before : ℚ) (QB_before : ℚ) (QA_after : ℚ) (QB_after : ℚ) : Prop :=
  QA_before < QA_after ∧ QB_before < QB_after

-- Part b
def can_ratings_increase_after_second_migration (QA_after_first : ℚ) (QB_after_first : ℚ) (QA_after_second : ℚ) (QB_after_second : ℚ) : Prop :=
  QA_after_second ≤ QA_after_first ∨ QB_after_second ≤ QB_after_first

-- Part c
def can_all_ratings_increase_after_reversed_migration (QA_before : ℚ) (QB_before : ℚ) (QC_before : ℚ) (QA_after_first : ℚ) (QB_after_first : ℚ) (QC_after_first : ℚ)
  (QA_after_second : ℚ) (QB_after_second : ℚ) (QC_after_second : ℚ) : Prop :=
  QA_before < QA_after_first ∧ QB_before < QB_after_first ∧ QC_before < QC_after_first ∧
  QA_after_first < QA_after_second ∧ QB_after_first < QB_after_second ∧ QC_after_first <= QC_after_second


-- Lean statements
theorem part_a (QA_before QA_after QB_before QB_after : ℚ) (Q_moved : ℚ) 
  (h : QA_before < QA_after ∧ QA_after < Q_moved ∧ QB_before < QB_after ∧ QB_after < Q_moved) : 
  can_ratings_increase_after_first_migration QA_before QB_before QA_after QB_after := 
by sorry

theorem part_b (QA_after_first QB_after_first QA_after_second QB_after_second : ℚ):
  ¬ can_ratings_increase_after_second_migration QA_after_first QB_after_first QA_after_second QB_after_second := 
by sorry

theorem part_c (QA_before QB_before QC_before QA_after_first QB_after_first QC_after_first
  QA_after_second QB_after_second QC_after_second: ℚ)
  (h: QA_before < QA_after_first ∧ QB_before < QB_after_first ∧ QC_before < QC_after_first ∧ 
      QA_after_first < QA_after_second ∧ QB_after_first < QB_after_second) :
   can_all_ratings_increase_after_reversed_migration QA_before QB_before QC_before QA_after_first QB_after_first QC_after_first QA_after_second QB_after_second QC_after_second :=
by sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l1064_106455


namespace NUMINAMATH_GPT_sum_divisible_by_seventeen_l1064_106460

theorem sum_divisible_by_seventeen :
  (90 + 91 + 92 + 93 + 94 + 95 + 96 + 97) % 17 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_sum_divisible_by_seventeen_l1064_106460


namespace NUMINAMATH_GPT_exists_five_points_isosceles_exists_six_points_isosceles_exists_seven_points_isosceles_3D_l1064_106487

open Real EuclideanGeometry

def is_isosceles_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ((dist A B = dist B C) ∨ (dist A B = dist A C) ∨ (dist B C = dist A C))

def is_isosceles_triangle_3D (A B C : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ((dist A B = dist B C) ∨ (dist A B = dist A C) ∨ (dist B C = dist A C))

def five_points_isosceles (pts : Fin 5 → EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∀ i j k : Fin 5, is_isosceles_triangle (pts i) (pts j) (pts k)

def six_points_isosceles (pts : Fin 6 → EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∀ i j k : Fin 6, is_isosceles_triangle (pts i) (pts j) (pts k)

def seven_points_isosceles_3D (pts : Fin 7 → EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∀ i j k : Fin 7, is_isosceles_triangle_3D (pts i) (pts j) (pts k)

theorem exists_five_points_isosceles : ∃ (pts : Fin 5 → EuclideanSpace ℝ (Fin 2)), five_points_isosceles pts :=
sorry

theorem exists_six_points_isosceles : ∃ (pts : Fin 6 → EuclideanSpace ℝ (Fin 2)), six_points_isosceles pts :=
sorry

theorem exists_seven_points_isosceles_3D : ∃ (pts : Fin 7 → EuclideanSpace ℝ (Fin 3)), seven_points_isosceles_3D pts :=
sorry

end NUMINAMATH_GPT_exists_five_points_isosceles_exists_six_points_isosceles_exists_seven_points_isosceles_3D_l1064_106487


namespace NUMINAMATH_GPT_arithmetic_progression_of_squares_l1064_106405

theorem arithmetic_progression_of_squares 
  (a b c : ℝ)
  (h : 1 / (a + b) - 1 / (a + c) = 1 / (b + c) - 1 / (a + c)) :
  2 * b^2 = a^2 + c^2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_of_squares_l1064_106405


namespace NUMINAMATH_GPT_exists_composite_l1064_106459

theorem exists_composite (x y : ℕ) (hx : 2 ≤ x ∧ x ≤ 100) (hy : 2 ≤ y ∧ y ≤ 100) :
  ∃ n : ℕ, ∃ k : ℕ, x^(2^n) + y^(2^n) = k * (k + 1) :=
by {
  sorry -- proof goes here
}

end NUMINAMATH_GPT_exists_composite_l1064_106459


namespace NUMINAMATH_GPT_sequence_fifth_term_l1064_106492

theorem sequence_fifth_term (a : ℤ) (d : ℤ) (n : ℕ) (a_n : ℤ) :
  a_n = 89 ∧ d = 11 ∧ n = 5 → a + (n-1) * -d = 45 := 
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  exact sorry

end NUMINAMATH_GPT_sequence_fifth_term_l1064_106492


namespace NUMINAMATH_GPT_toothpicks_in_20th_stage_l1064_106429

theorem toothpicks_in_20th_stage :
  (3 + 3 * (20 - 1) = 60) :=
by
  sorry

end NUMINAMATH_GPT_toothpicks_in_20th_stage_l1064_106429


namespace NUMINAMATH_GPT_distance_between_locations_A_and_B_l1064_106406

-- Define the conditions
variables {x y s t : ℝ}

-- Conditions specified in the problem
axiom bus_a_meets_bus_b_after_85_km : 85 / x = (s - 85) / y 
axiom buses_meet_again_after_turnaround : (s - 85 + 65) / x + 1 / 2 = (85 + (s - 65)) / y + 1 / 2

-- The theorem to be proved
theorem distance_between_locations_A_and_B : s = 190 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_locations_A_and_B_l1064_106406


namespace NUMINAMATH_GPT_cylindrical_tank_volume_increase_l1064_106432

theorem cylindrical_tank_volume_increase (k : ℝ) (H R : ℝ) 
  (hR : R = 10) (hH : H = 5)
  (condition : (π * (10 * k)^2 * 5 - π * 10^2 * 5) = (π * 10^2 * (5 + k) - π * 10^2 * 5)) :
  k = (1 + Real.sqrt 101) / 10 :=
by
  sorry

end NUMINAMATH_GPT_cylindrical_tank_volume_increase_l1064_106432


namespace NUMINAMATH_GPT_average_age_remains_l1064_106477

theorem average_age_remains (total_age : ℕ) (leaving_age : ℕ) (remaining_people : ℕ) (initial_people_avg : ℕ) 
                            (total_age_eq : total_age = initial_people_avg * 8) 
                            (new_total_age : ℕ := total_age - leaving_age)
                            (new_avg : ℝ := new_total_age / remaining_people) :
  (initial_people_avg = 25) ∧ (leaving_age = 20) ∧ (remaining_people = 7) → new_avg = 180 / 7 := 
by
  sorry

end NUMINAMATH_GPT_average_age_remains_l1064_106477


namespace NUMINAMATH_GPT_qualifying_rate_l1064_106422

theorem qualifying_rate (a b : ℝ) (h1 : 0 ≤ a ∧ a ≤ 1) (h2 : 0 ≤ b ∧ b ≤ 1) :
  (1 - a) * (1 - b) = 1 - a - b + a * b :=
by sorry

end NUMINAMATH_GPT_qualifying_rate_l1064_106422


namespace NUMINAMATH_GPT_new_price_of_computer_l1064_106400

theorem new_price_of_computer (d : ℝ) (h : 2 * d = 520) : d * 1.3 = 338 := 
sorry

end NUMINAMATH_GPT_new_price_of_computer_l1064_106400


namespace NUMINAMATH_GPT_min_value_of_expression_is_6_l1064_106474

noncomputable def min_value_of_expression (a b c : ℝ) : ℝ :=
  (a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a

theorem min_value_of_expression_is_6 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : min_value_of_expression a b c = 6 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_is_6_l1064_106474


namespace NUMINAMATH_GPT_find_distance_between_stations_l1064_106445

noncomputable def distance_between_stations (D T : ℝ) : Prop :=
  D = 100 * T ∧
  D = 50 * (T + 15 / 60) ∧
  D = 70 * (T + 7 / 60)

theorem find_distance_between_stations :
  ∃ D T : ℝ, distance_between_stations D T ∧ D = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_distance_between_stations_l1064_106445


namespace NUMINAMATH_GPT_geom_seq_sum_l1064_106493

theorem geom_seq_sum (q : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 3)
  (h2 : a 1 + a 3 + a 5 = 21)
  (h_geometric : ∀ n : ℕ, a (n + 1) = a 1 * q ^ n) :
  a 3 + a 5 + a 7 = 42 :=
sorry

end NUMINAMATH_GPT_geom_seq_sum_l1064_106493


namespace NUMINAMATH_GPT_seth_spent_more_on_ice_cream_l1064_106454

-- Definitions based on the conditions
def cartons_ice_cream := 20
def cartons_yogurt := 2
def cost_per_carton_ice_cream := 6
def cost_per_carton_yogurt := 1

-- Theorem statement
theorem seth_spent_more_on_ice_cream :
  (cartons_ice_cream * cost_per_carton_ice_cream) - (cartons_yogurt * cost_per_carton_yogurt) = 118 :=
by
  sorry

end NUMINAMATH_GPT_seth_spent_more_on_ice_cream_l1064_106454


namespace NUMINAMATH_GPT_gamma_donuts_received_l1064_106437

theorem gamma_donuts_received (total_donuts delta_donuts gamma_donuts beta_donuts : ℕ) 
    (h1 : total_donuts = 40) 
    (h2 : delta_donuts = 8) 
    (h3 : beta_donuts = 3 * gamma_donuts) :
    delta_donuts + beta_donuts + gamma_donuts = total_donuts -> gamma_donuts = 8 :=
by 
  intro h4
  sorry

end NUMINAMATH_GPT_gamma_donuts_received_l1064_106437


namespace NUMINAMATH_GPT_shoes_difference_l1064_106482

theorem shoes_difference :
  let pairs_per_box := 20
  let boxes_A := 8
  let boxes_B := 5 * boxes_A
  let total_pairs_A := boxes_A * pairs_per_box
  let total_pairs_B := boxes_B * pairs_per_box
  total_pairs_B - total_pairs_A = 640 :=
by
  sorry

end NUMINAMATH_GPT_shoes_difference_l1064_106482


namespace NUMINAMATH_GPT_polygon_sides_l1064_106447

theorem polygon_sides {S n : ℕ} (h : S = 2160) (hs : S = 180 * (n - 2)) : n = 14 := 
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l1064_106447


namespace NUMINAMATH_GPT_smallest_even_integer_cube_mod_1000_l1064_106440

theorem smallest_even_integer_cube_mod_1000 :
  ∃ n : ℕ, (n % 2 = 0) ∧ (n > 0) ∧ (n^3 % 1000 = 392) ∧ (∀ m : ℕ, (m % 2 = 0) ∧ (m > 0) ∧ (m^3 % 1000 = 392) → n ≤ m) ∧ n = 892 := 
sorry

end NUMINAMATH_GPT_smallest_even_integer_cube_mod_1000_l1064_106440


namespace NUMINAMATH_GPT_middle_income_sample_count_l1064_106471

def total_households : ℕ := 600
def high_income_families : ℕ := 150
def middle_income_families : ℕ := 360
def low_income_families : ℕ := 90
def sample_size : ℕ := 80

theorem middle_income_sample_count : 
  (middle_income_families / total_households) * sample_size = 48 := 
by
  sorry

end NUMINAMATH_GPT_middle_income_sample_count_l1064_106471


namespace NUMINAMATH_GPT_bicycle_weight_l1064_106470

theorem bicycle_weight (b s : ℝ) (h1 : 9 * b = 5 * s) (h2 : 4 * s = 160) : b = 200 / 9 :=
by
  sorry

end NUMINAMATH_GPT_bicycle_weight_l1064_106470


namespace NUMINAMATH_GPT_how_many_roses_cut_l1064_106439

theorem how_many_roses_cut :
  ∀ (r_i r_f r_c : ℕ), r_i = 6 → r_f = 16 → r_c = r_f - r_i → r_c = 10 :=
by
  intros r_i r_f r_c hri hrf heq
  rw [hri, hrf] at heq
  exact heq

end NUMINAMATH_GPT_how_many_roses_cut_l1064_106439


namespace NUMINAMATH_GPT_radishes_difference_l1064_106408

theorem radishes_difference 
    (total_radishes : ℕ)
    (groups : ℕ)
    (first_basket : ℕ)
    (second_basket : ℕ)
    (total_radishes_eq : total_radishes = 88)
    (groups_eq : groups = 4)
    (first_basket_eq : first_basket = 37)
    (second_basket_eq : second_basket = total_radishes - first_basket)
  : second_basket - first_basket = 14 :=
by
  sorry

end NUMINAMATH_GPT_radishes_difference_l1064_106408


namespace NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l1064_106442

variable (a : ℝ)

def p : Prop := a > 0
def q : Prop := a^2 + a ≥ 0

theorem p_sufficient_not_necessary_for_q : (p a → q a) ∧ ¬ (q a → p a) := by
  sorry

end NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l1064_106442


namespace NUMINAMATH_GPT_gre_exam_month_l1064_106401

def months_of_year := ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

def start_month := "June"
def preparation_duration := 5

theorem gre_exam_month :
  months_of_year[(months_of_year.indexOf start_month + preparation_duration) % 12] = "November" := by
  sorry

end NUMINAMATH_GPT_gre_exam_month_l1064_106401


namespace NUMINAMATH_GPT_sufficient_and_not_necessary_condition_l1064_106420

theorem sufficient_and_not_necessary_condition (a b : ℝ) (hb: a < 0 ∧ b < 0) : a + b < 0 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_and_not_necessary_condition_l1064_106420


namespace NUMINAMATH_GPT_pencils_initial_count_l1064_106435

theorem pencils_initial_count (pencils_initially: ℕ) :
  (∀ n, n > 0 → n < 36 → 36 % n = 1) →
  pencils_initially + 30 = 36 → 
  pencils_initially = 6 :=
by
  intro h hn
  sorry

end NUMINAMATH_GPT_pencils_initial_count_l1064_106435


namespace NUMINAMATH_GPT_min_sum_of_factors_of_72_l1064_106473

theorem min_sum_of_factors_of_72 (a b: ℤ) (h: a * b = 72) : a + b = -73 :=
sorry

end NUMINAMATH_GPT_min_sum_of_factors_of_72_l1064_106473


namespace NUMINAMATH_GPT_negate_p_l1064_106433

theorem negate_p (p : Prop) :
  (∃ x : ℝ, 0 < x ∧ 3^x < x^3) ↔ (¬ (∀ x : ℝ, 0 < x → 3^x ≥ x^3)) :=
by sorry

end NUMINAMATH_GPT_negate_p_l1064_106433


namespace NUMINAMATH_GPT_negative_half_less_than_negative_third_l1064_106461

theorem negative_half_less_than_negative_third : - (1 / 2 : ℝ) < - (1 / 3 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_negative_half_less_than_negative_third_l1064_106461


namespace NUMINAMATH_GPT_unique_intersection_l1064_106414

theorem unique_intersection {m : ℝ} :
  (∃! y : ℝ, m = -3 * y^2 - 4 * y + 7) ↔ m = 25 / 3 :=
by
  sorry

end NUMINAMATH_GPT_unique_intersection_l1064_106414


namespace NUMINAMATH_GPT_smallest_prime_with_digits_sum_22_l1064_106413

def digits_sum (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem smallest_prime_with_digits_sum_22 : 
  ∃ p : ℕ, Prime p ∧ digits_sum p = 22 ∧ ∀ q : ℕ, Prime q ∧ digits_sum q = 22 → q ≥ p ∧ p = 499 :=
by sorry

end NUMINAMATH_GPT_smallest_prime_with_digits_sum_22_l1064_106413


namespace NUMINAMATH_GPT_gardener_payment_l1064_106412

theorem gardener_payment (total_cost : ℕ) (rect_area : ℕ) (rect_side1 : ℕ) (rect_side2 : ℕ)
                         (square1_area : ℕ) (square2_area : ℕ) (cost_per_are : ℕ) :
  total_cost = 570 →
  rect_area = 600 → rect_side1 = 20 → rect_side2 = 30 →
  square1_area = 400 → square2_area = 900 →
  cost_per_are * (rect_area + square1_area + square2_area) / 100 = total_cost →
  cost_per_are = 30 →
  ∃ (rect_payment : ℕ) (square1_payment : ℕ) (square2_payment : ℕ),
    rect_payment = 6 * cost_per_are ∧
    square1_payment = 4 * cost_per_are ∧
    square2_payment = 9 * cost_per_are ∧
    rect_payment + square1_payment + square2_payment = total_cost :=
by
  intros
  sorry

end NUMINAMATH_GPT_gardener_payment_l1064_106412


namespace NUMINAMATH_GPT_find_a_l1064_106456

noncomputable section

def f (x a : ℝ) : ℝ := Real.sqrt (1 + a * 4^x)

theorem find_a (a : ℝ) : 
  (∀ (x : ℝ), x ≤ -1 → 1 + a * 4^x ≥ 0) → a = -4 :=
sorry

end NUMINAMATH_GPT_find_a_l1064_106456


namespace NUMINAMATH_GPT_number_of_correct_conclusions_l1064_106457

noncomputable def floor (x : ℝ) : ℤ := Int.floor x
noncomputable def f (x : ℝ) : ℝ := x - floor x

theorem number_of_correct_conclusions : 
  ∃ n, n = 3 ∧ 
  (0 ≤ f 0) ∧ 
  (∀ x : ℝ, 0 ≤ f x) ∧ 
  (∀ x : ℝ, f x < 1) ∧ 
  (∀ x : ℝ, f (x + 1) = f x) ∧ 
  ¬ (∀ x : ℝ, f (-x) = f x) := 
sorry

end NUMINAMATH_GPT_number_of_correct_conclusions_l1064_106457


namespace NUMINAMATH_GPT_g_at_six_l1064_106497

def g (x : ℝ) : ℝ := 2 * x^4 - 19 * x^3 + 30 * x^2 - 12 * x - 72

theorem g_at_six : g 6 = 288 :=
by
  sorry

end NUMINAMATH_GPT_g_at_six_l1064_106497


namespace NUMINAMATH_GPT_height_of_flagpole_l1064_106453

theorem height_of_flagpole 
  (house_shadow : ℝ) (tree_height : ℝ) (tree_shadow : ℝ) 
  (flagpole_shadow : ℝ) (house_height : ℝ)
  (h1 : house_shadow = 70)
  (h2 : tree_height = 28)
  (h3 : tree_shadow = 40)
  (h4 : flagpole_shadow = 25)
  (h5 : house_height = (tree_height * house_shadow) / tree_shadow) :
  round ((house_height * flagpole_shadow / house_shadow) : ℝ) = 18 := 
by
  sorry

end NUMINAMATH_GPT_height_of_flagpole_l1064_106453


namespace NUMINAMATH_GPT_parabola_properties_l1064_106490

def parabola (a b x : ℝ) : ℝ :=
  a * x ^ 2 + b * x - 4

theorem parabola_properties :
  ∃ (a b : ℝ), (a = 2) ∧ (b = 2) ∧
  parabola a b (-2) = 0 ∧ 
  parabola a b (-1) = -4 ∧ 
  parabola a b 0 = -4 ∧ 
  parabola a b 1 = 0 ∧ 
  parabola a b 2 = 8 ∧ 
  parabola a b (-3) = 8 ∧ 
  (0, -4) ∈ {(x, y) | ∃ a b, y = parabola a b x} :=
sorry

end NUMINAMATH_GPT_parabola_properties_l1064_106490


namespace NUMINAMATH_GPT_angle_A_l1064_106424

variable (a b c : ℝ) (A B C : ℝ)

-- Hypothesis: In triangle ABC, (a + c)(a - c) = b(b + c)
def condition (a b c : ℝ) : Prop := (a + c) * (a - c) = b * (b + c)

-- The goal is to show that under given conditions, ∠A = 2π/3
theorem angle_A (h : condition a b c) : A = 2 * π / 3 :=
sorry

end NUMINAMATH_GPT_angle_A_l1064_106424


namespace NUMINAMATH_GPT_neighborhood_has_exactly_one_item_l1064_106475

noncomputable def neighborhood_conditions : Prop :=
  let total_households := 120
  let households_no_items := 15
  let households_car_and_bike := 28
  let households_car := 52
  let households_bike := 32
  let households_scooter := 18
  let households_skateboard := 8
  let households_at_least_one_item := total_households - households_no_items
  let households_car_only := households_car - households_car_and_bike
  let households_bike_only := households_bike - households_car_and_bike
  let households_exactly_one_item := households_car_only + households_bike_only + households_scooter + households_skateboard
  households_at_least_one_item = 105 ∧ households_exactly_one_item = 54

theorem neighborhood_has_exactly_one_item :
  neighborhood_conditions :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_neighborhood_has_exactly_one_item_l1064_106475


namespace NUMINAMATH_GPT_triangle_area_l1064_106463

theorem triangle_area (A B C : ℝ) (AB BC CA : ℝ) (sinA sinB sinC : ℝ)
    (h1 : sinA * sinB * sinC = 1 / 1000) 
    (h2 : AB * BC * CA = 1000) : 
    (AB * BC * CA / (4 * 50)) = 5 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_triangle_area_l1064_106463


namespace NUMINAMATH_GPT_boat_travel_distance_upstream_l1064_106448

noncomputable def upstream_distance (v : ℝ) : ℝ :=
  let d := 2.5191640969412834 * (v + 3)
  d

theorem boat_travel_distance_upstream :
  ∀ v : ℝ, 
  (∀ D : ℝ, D / (v + 3) = 2.5191640969412834 → D / (v - 3) = D / (v + 3) + 0.5) → 
  upstream_distance 33.2299691632954 = 91.25 :=
by
  sorry

end NUMINAMATH_GPT_boat_travel_distance_upstream_l1064_106448


namespace NUMINAMATH_GPT_overall_gain_is_correct_l1064_106479

noncomputable def overall_gain_percentage : ℝ :=
  let CP_A := 100
  let SP_A := 120 / (1 - 0.20)
  let gain_A := SP_A - CP_A

  let CP_B := 200
  let SP_B := 240 / (1 + 0.10)
  let gain_B := SP_B - CP_B

  let CP_C := 150
  let SP_C := (165 / (1 + 0.05)) / (1 - 0.10)
  let gain_C := SP_C - CP_C

  let CP_D := 300
  let SP_D := (345 / (1 - 0.05)) / (1 + 0.15)
  let gain_D := SP_D - CP_D

  let total_gain := gain_A + gain_B + gain_C + gain_D
  let total_CP := CP_A + CP_B + CP_C + CP_D
  (total_gain / total_CP) * 100

theorem overall_gain_is_correct : abs (overall_gain_percentage - 14.48) < 0.01 := by
  sorry

end NUMINAMATH_GPT_overall_gain_is_correct_l1064_106479


namespace NUMINAMATH_GPT_medium_supermarkets_in_sample_l1064_106483

-- Definitions of the conditions
def total_supermarkets : ℕ := 200 + 400 + 1400
def prop_medium_supermarkets : ℚ := 400 / total_supermarkets
def sample_size : ℕ := 100

-- Problem: Prove that the number of medium-sized supermarkets in the sample is 20.
theorem medium_supermarkets_in_sample : 
  (sample_size * prop_medium_supermarkets) = 20 :=
by
  sorry

end NUMINAMATH_GPT_medium_supermarkets_in_sample_l1064_106483


namespace NUMINAMATH_GPT_find_angle4_l1064_106476

noncomputable def angle_1 := 70
noncomputable def angle_2 := 110
noncomputable def angle_3 := 35
noncomputable def angle_4 := 35

theorem find_angle4 (h1 : angle_1 + angle_2 = 180) (h2 : angle_3 = angle_4) :
  angle_4 = 35 :=
by
  have h3: angle_1 + 70 + 40 = 180 := by sorry
  have h4: angle_2 + angle_3 + angle_4 = 180 := by sorry
  sorry

end NUMINAMATH_GPT_find_angle4_l1064_106476


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1064_106438

theorem solution_set_of_inequality (x : ℝ) :
  x^2 - |x| - 2 < 0 ↔ -2 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1064_106438


namespace NUMINAMATH_GPT_inequality_correct_l1064_106430

theorem inequality_correct (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) (h₂ : b < 1) : (1 - a) ^ a > (1 - b) ^ b :=
sorry

end NUMINAMATH_GPT_inequality_correct_l1064_106430


namespace NUMINAMATH_GPT_clown_blew_more_balloons_l1064_106486

theorem clown_blew_more_balloons :
  ∀ (initial_balloons final_balloons additional_balloons : ℕ),
    initial_balloons = 47 →
    final_balloons = 60 →
    additional_balloons = final_balloons - initial_balloons →
    additional_balloons = 13 :=
by
  intros initial_balloons final_balloons additional_balloons h1 h2 h3
  sorry

end NUMINAMATH_GPT_clown_blew_more_balloons_l1064_106486


namespace NUMINAMATH_GPT_ratio_mn_eq_x_plus_one_over_two_x_plus_one_l1064_106403

theorem ratio_mn_eq_x_plus_one_over_two_x_plus_one (x : ℝ) (m n : ℝ) 
  (hx : x > 0) 
  (hmn : m * n ≠ 0) 
  (hineq : m * x > n * x + n) : 
  m / (m + n) = (x + 1) / (2 * x + 1) := 
by 
  sorry

end NUMINAMATH_GPT_ratio_mn_eq_x_plus_one_over_two_x_plus_one_l1064_106403


namespace NUMINAMATH_GPT_find_image_point_l1064_106485

noncomputable def lens_equation (t f k : ℝ) : Prop :=
  (1 / k) + (1 / t) = (1 / f)

theorem find_image_point
  (O F T T_star K_star K : ℝ)
  (OT OTw OTw_star FK : ℝ)
  (OT_eq : OT = OTw)
  (OTw_star_eq : OTw_star = OT)
  (similarity_condition : ∀ (CTw_star OF : ℝ), CTw_star / OF = (CTw_star + OK) / OK)
  : lens_equation OTw FK K :=
sorry

end NUMINAMATH_GPT_find_image_point_l1064_106485


namespace NUMINAMATH_GPT_intersection_eq_l1064_106468

-- Conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Proof Problem
theorem intersection_eq : M ∩ N = {2, 3} := 
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l1064_106468


namespace NUMINAMATH_GPT_dima_story_retelling_count_l1064_106431

theorem dima_story_retelling_count :
  ∃ n, (26 * (2 ^ 5) * (3 ^ 4)) = 33696 ∧ n = 9 :=
by
  sorry

end NUMINAMATH_GPT_dima_story_retelling_count_l1064_106431


namespace NUMINAMATH_GPT_math_problem_l1064_106494

theorem math_problem : 
  ∃ (n m k : ℕ), 
    (∀ d : ℕ, d ∣ n → d > 0) ∧ 
    (n = m * 6^k) ∧
    (∀ d : ℕ, d ∣ m → 6 ∣ d → False) ∧
    (m + k = 60466182) ∧ 
    (n.factors.count 1 = 2023) :=
sorry

end NUMINAMATH_GPT_math_problem_l1064_106494


namespace NUMINAMATH_GPT_joan_balloon_gain_l1064_106421

theorem joan_balloon_gain
  (initial_balloons : ℕ)
  (final_balloons : ℕ)
  (h_initial : initial_balloons = 9)
  (h_final : final_balloons = 11) :
  final_balloons - initial_balloons = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_joan_balloon_gain_l1064_106421


namespace NUMINAMATH_GPT_cheryl_mms_l1064_106450

/-- Cheryl's m&m problem -/
theorem cheryl_mms (c l g d : ℕ) (h1 : c = 25) (h2 : l = 7) (h3 : g = 13) :
  (c - l - g) = d → d = 5 :=
by
  sorry

end NUMINAMATH_GPT_cheryl_mms_l1064_106450


namespace NUMINAMATH_GPT_guess_x_30_guess_y_127_l1064_106441

theorem guess_x_30 : 120 = 4 * 30 := 
  sorry

theorem guess_y_127 : 87 = 127 - 40 := 
  sorry

end NUMINAMATH_GPT_guess_x_30_guess_y_127_l1064_106441


namespace NUMINAMATH_GPT_number_of_possible_scenarios_l1064_106426

-- Definitions based on conditions
def num_companies : Nat := 5
def reps_company_A : Nat := 2
def reps_other_companies : Nat := 1
def total_speakers : Nat := 3

-- Problem statement
theorem number_of_possible_scenarios : 
  ∃ (scenarios : Nat), scenarios = 16 ∧ 
  (scenarios = 
    (Nat.choose reps_company_A 1 * Nat.choose 4 2) + 
    Nat.choose 4 3) :=
by
  sorry

end NUMINAMATH_GPT_number_of_possible_scenarios_l1064_106426


namespace NUMINAMATH_GPT_number_difference_l1064_106423

theorem number_difference (a b : ℕ) (h1 : a + b = 44) (h2 : 8 * a = 3 * b) : b - a = 20 := by
  sorry

end NUMINAMATH_GPT_number_difference_l1064_106423


namespace NUMINAMATH_GPT_exists_increasing_or_decreasing_subsequence_l1064_106489

theorem exists_increasing_or_decreasing_subsequence (n : ℕ) (a : Fin (n^2 + 1) → ℝ) :
  ∃ (b : Fin (n + 1) → ℝ), (StrictMono b ∨ StrictAnti b) :=
sorry

end NUMINAMATH_GPT_exists_increasing_or_decreasing_subsequence_l1064_106489


namespace NUMINAMATH_GPT_find_n_l1064_106467

def Point : Type := ℝ × ℝ

def A : Point := (5, -8)
def B : Point := (9, -30)
def C (n : ℝ) : Point := (n, n)

def collinear (p1 p2 p3 : Point) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem find_n (n : ℝ) (h : collinear A B (C n)) : n = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_n_l1064_106467


namespace NUMINAMATH_GPT_square_field_area_l1064_106411

def square_area (side_length : ℝ) : ℝ :=
  side_length * side_length

theorem square_field_area :
  square_area 20 = 400 := by
  sorry

end NUMINAMATH_GPT_square_field_area_l1064_106411


namespace NUMINAMATH_GPT_find_first_number_l1064_106484

theorem find_first_number (x : ℝ) : (x + 16 + 8 + 22) / 4 = 13 ↔ x = 6 :=
by 
  sorry

end NUMINAMATH_GPT_find_first_number_l1064_106484


namespace NUMINAMATH_GPT_euler_totient_divisibility_l1064_106449

theorem euler_totient_divisibility (n : ℕ) (h : n > 0) : n ∣ Nat.totient (2^n - 1) := by
  sorry

end NUMINAMATH_GPT_euler_totient_divisibility_l1064_106449


namespace NUMINAMATH_GPT_largest_among_options_l1064_106427

theorem largest_among_options (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b > (1/2) ∧ b > a^2 + b^2 ∧ b > 2*a*b := 
by
  sorry

end NUMINAMATH_GPT_largest_among_options_l1064_106427


namespace NUMINAMATH_GPT_isosceles_triangle_sides_l1064_106491

theorem isosceles_triangle_sides (a b c : ℝ) (hb : b = 3) (hc : a = 3 ∨ c = 3) (hperim : a + b + c = 7) :
  a = 2 ∨ a = 3 ∨ c = 2 ∨ c = 3 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_sides_l1064_106491


namespace NUMINAMATH_GPT_infinite_possible_matrices_A_squared_l1064_106472

theorem infinite_possible_matrices_A_squared (A : Matrix (Fin 3) (Fin 3) ℝ) (hA : A^4 = 0) :
  ∃ (S : Set (Matrix (Fin 3) (Fin 3) ℝ)), (∀ B ∈ S, B = A^2) ∧ S.Infinite :=
sorry

end NUMINAMATH_GPT_infinite_possible_matrices_A_squared_l1064_106472


namespace NUMINAMATH_GPT_worker_schedule_l1064_106428

open Nat

theorem worker_schedule (x : ℕ) :
  24 * 3 + (15 - 3) * x > 408 :=
by
  sorry

end NUMINAMATH_GPT_worker_schedule_l1064_106428


namespace NUMINAMATH_GPT_three_sleep_simultaneously_l1064_106464

noncomputable def professors := Finset.range 5

def sleeping_times (p: professors) : Finset ℕ 
-- definition to be filled in, stating that p falls asleep twice.
:= sorry 

def moment_two_asleep (p q: professors) : ℕ 
-- definition to be filled in, stating that p and q are asleep together once.
:= sorry

theorem three_sleep_simultaneously :
  ∃ t : ℕ, ∃ p1 p2 p3 : professors, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
  (t ∈ sleeping_times p1) ∧
  (t ∈ sleeping_times p2) ∧
  (t ∈ sleeping_times p3) := by
  sorry

end NUMINAMATH_GPT_three_sleep_simultaneously_l1064_106464


namespace NUMINAMATH_GPT_probability_point_in_square_l1064_106444

theorem probability_point_in_square (r : ℝ) (hr : 0 < r) :
  (∃ p : ℝ, p = 2 / Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_probability_point_in_square_l1064_106444


namespace NUMINAMATH_GPT_install_time_per_window_l1064_106481

/-- A new building needed 14 windows. The builder had already installed 8 windows.
    It will take the builder 48 hours to install the rest of the windows. -/
theorem install_time_per_window (total_windows installed_windows remaining_install_time : ℕ)
  (h_total : total_windows = 14)
  (h_installed : installed_windows = 8)
  (h_remaining_time : remaining_install_time = 48) :
  (remaining_install_time / (total_windows - installed_windows)) = 8 :=
by
  -- Insert usual proof steps here
  sorry

end NUMINAMATH_GPT_install_time_per_window_l1064_106481


namespace NUMINAMATH_GPT_areas_of_isosceles_triangles_l1064_106496

theorem areas_of_isosceles_triangles (A B C : ℝ) (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 13)
  (hA : A = 1/2 * a * a) (hB : B = 1/2 * b * b) (hC : C = 1/2 * c * c) :
  A + B = C :=
by
  sorry

end NUMINAMATH_GPT_areas_of_isosceles_triangles_l1064_106496


namespace NUMINAMATH_GPT_find_function_l1064_106452

def satisfies_functional_eqn (f : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x * f y) = f (x * y^2) - 2 * x^2 * f y - f x - 1

theorem find_function (f : ℝ → ℝ) :
  satisfies_functional_eqn f → (∀ y : ℝ, f y = y^2 - 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_function_l1064_106452


namespace NUMINAMATH_GPT_no_solution_for_m_l1064_106402

theorem no_solution_for_m (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (m : ℕ) (h3 : (ab)^2015 = (a^2 + b^2)^m) : false := 
sorry

end NUMINAMATH_GPT_no_solution_for_m_l1064_106402


namespace NUMINAMATH_GPT_proof_sets_l1064_106499

def I : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {1, 3, 6}
def complement (s : Set ℕ) : Set ℕ := {x | x ∈ I ∧ x ∉ s}

theorem proof_sets :
  M ∩ (complement N) = {4, 5} ∧ {2, 7, 8} = complement (M ∪ N) :=
by
  sorry

end NUMINAMATH_GPT_proof_sets_l1064_106499


namespace NUMINAMATH_GPT_eustace_age_in_3_years_l1064_106498

variable (E M : ℕ)

theorem eustace_age_in_3_years
  (h1 : E = 2 * M)
  (h2 : M + 3 = 21) :
  E + 3 = 39 :=
sorry

end NUMINAMATH_GPT_eustace_age_in_3_years_l1064_106498


namespace NUMINAMATH_GPT_eccentricity_hyperbola_l1064_106436

variables (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0)
variables (H : c = Real.sqrt (a^2 + b^2))
variables (L1 : ∀ x y : ℝ, x = c → (x^2/a^2 - y^2/b^2 = 1))
variables (L2 : ∀ (B C : ℝ × ℝ), (B.1 = c ∧ C.1 = c) ∧ (B.2 = -C.2) ∧ (B.2 = b^2/a))

theorem eccentricity_hyperbola : ∃ e, e = 2 :=
sorry

end NUMINAMATH_GPT_eccentricity_hyperbola_l1064_106436
