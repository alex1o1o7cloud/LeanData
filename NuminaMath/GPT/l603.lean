import Mathlib

namespace NUMINAMATH_GPT_minimize_sum_of_squares_l603_60357

theorem minimize_sum_of_squares (x1 x2 x3 : ℝ) (hpos1 : 0 < x1) (hpos2 : 0 < x2) (hpos3 : 0 < x3)
  (h_eq : x1 + 3 * x2 + 5 * x3 = 100) : x1^2 + x2^2 + x3^2 = 2000 / 7 := 
sorry

end NUMINAMATH_GPT_minimize_sum_of_squares_l603_60357


namespace NUMINAMATH_GPT_highest_number_of_years_of_service_l603_60344

theorem highest_number_of_years_of_service
  (years_of_service : Fin 8 → ℕ)
  (h_range : ∃ L, ∃ H, H - L = 14)
  (h_second_highest : ∃ second_highest, second_highest = 16) :
  ∃ highest, highest = 17 := by
  sorry

end NUMINAMATH_GPT_highest_number_of_years_of_service_l603_60344


namespace NUMINAMATH_GPT_total_amount_spent_l603_60388

def cost_of_haley_paper : ℝ := 3.75 + (3.75 * 0.5)
def cost_of_sister_paper : ℝ := (4.50 * 2) + (4.50 * 0.5)
def cost_of_haley_pens : ℝ := (1.45 * 5) - ((1.45 * 5) * 0.25)
def cost_of_sister_pens : ℝ := (1.65 * 7) - ((1.65 * 7) * 0.25)

def total_cost_of_supplies : ℝ := cost_of_haley_paper + cost_of_sister_paper + cost_of_haley_pens + cost_of_sister_pens

theorem total_amount_spent : total_cost_of_supplies = 30.975 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_spent_l603_60388


namespace NUMINAMATH_GPT_square_section_dimensions_l603_60320

theorem square_section_dimensions (x length : ℕ) :
  (250 ≤ x^2 + x * length ∧ x^2 + x * length ≤ 300) ∧ (25 ≤ length ∧ length ≤ 30) →
  (x = 7 ∨ x = 8) :=
  by
    sorry

end NUMINAMATH_GPT_square_section_dimensions_l603_60320


namespace NUMINAMATH_GPT_possible_values_of_C_l603_60328

theorem possible_values_of_C {a b C : ℤ} :
  (C = a * (a - 5) ∧ C = b * (b - 8)) ↔ (C = 0 ∨ C = 84) :=
sorry

end NUMINAMATH_GPT_possible_values_of_C_l603_60328


namespace NUMINAMATH_GPT_harry_water_per_mile_l603_60379

noncomputable def water_per_mile_during_first_3_miles (initial_water : ℝ) (remaining_water : ℝ) (leak_rate : ℝ) (hike_time : ℝ) (water_drunk_last_mile : ℝ) (first_3_miles : ℝ) : ℝ :=
  let total_leak := leak_rate * hike_time
  let remaining_before_last_mile := remaining_water + water_drunk_last_mile
  let start_last_mile := remaining_before_last_mile + total_leak
  let water_before_leak := initial_water - total_leak
  let water_drunk_first_3_miles := water_before_leak - start_last_mile
  water_drunk_first_3_miles / first_3_miles

theorem harry_water_per_mile :
  water_per_mile_during_first_3_miles 10 2 1 2 3 3 = 1 / 3 :=
by
  have initial_water := 10
  have remaining_water := 2
  have leak_rate := 1
  have hike_time := 2
  have water_drunk_last_mile := 3
  have first_3_miles := 3
  let total_leak := leak_rate * hike_time
  let remaining_before_last_mile := remaining_water + water_drunk_last_mile
  let start_last_mile := remaining_before_last_mile + total_leak
  let water_before_leak := initial_water - total_leak
  let water_drunk_first_3_miles := water_before_leak - start_last_mile
  let result := water_drunk_first_3_miles / first_3_miles
  exact sorry

end NUMINAMATH_GPT_harry_water_per_mile_l603_60379


namespace NUMINAMATH_GPT_value_by_which_number_is_multiplied_l603_60372

theorem value_by_which_number_is_multiplied (x : ℝ) : (5 / 6) * x = 10 ↔ x = 12 := by
  sorry

end NUMINAMATH_GPT_value_by_which_number_is_multiplied_l603_60372


namespace NUMINAMATH_GPT_john_pays_per_year_l603_60381

-- Define the costs and insurance parameters.
def cost_per_epipen : ℝ := 500
def insurance_coverage : ℝ := 0.75

-- Number of months in a year.
def months_in_year : ℕ := 12

-- Number of months each EpiPen lasts.
def months_per_epipen : ℕ := 6

-- Amount covered by insurance for each EpiPen.
def insurance_amount (cost : ℝ) (coverage : ℝ) : ℝ :=
  cost * coverage

-- Amount John pays after insurance for each EpiPen.
def amount_john_pays_per_epipen (cost : ℝ) (covered: ℝ) : ℝ :=
  cost - covered

-- Number of EpiPens John needs per year.
def epipens_per_year (months_in_year : ℕ) (months_per_epipen : ℕ) : ℕ :=
  months_in_year / months_per_epipen

-- Total amount John pays per year.
def total_amount_john_pays_per_year (amount_per_epipen : ℝ) (epipens_per_year : ℕ) : ℝ :=
  amount_per_epipen * epipens_per_year

-- Theorem to prove the correct answer.
theorem john_pays_per_year :
  total_amount_john_pays_per_year (amount_john_pays_per_epipen cost_per_epipen (insurance_amount cost_per_epipen insurance_coverage)) (epipens_per_year months_in_year months_per_epipen) = 250 := 
by
  sorry

end NUMINAMATH_GPT_john_pays_per_year_l603_60381


namespace NUMINAMATH_GPT_point_in_second_quadrant_iff_l603_60369

theorem point_in_second_quadrant_iff (a : ℝ) : (a - 2 < 0) ↔ (a < 2) :=
by
  sorry

end NUMINAMATH_GPT_point_in_second_quadrant_iff_l603_60369


namespace NUMINAMATH_GPT_max_value_ln_x_minus_x_on_interval_l603_60304

noncomputable def f (x : ℝ) : ℝ := Real.log x - x

theorem max_value_ln_x_minus_x_on_interval : 
  ∃ x ∈ Set.Ioc 0 (Real.exp 1), ∀ y ∈ Set.Ioc 0 (Real.exp 1), f y ≤ f x ∧ f x = -1 :=
by
  sorry

end NUMINAMATH_GPT_max_value_ln_x_minus_x_on_interval_l603_60304


namespace NUMINAMATH_GPT_gasoline_price_increase_l603_60306

theorem gasoline_price_increase (highest_price lowest_price : ℝ) (h1 : highest_price = 24) (h2 : lowest_price = 15) : 
  ((highest_price - lowest_price) / lowest_price) * 100 = 60 :=
by
  sorry

end NUMINAMATH_GPT_gasoline_price_increase_l603_60306


namespace NUMINAMATH_GPT_consecutive_integers_sum_to_thirty_unique_sets_l603_60374

theorem consecutive_integers_sum_to_thirty_unique_sets :
  (∃ a n : ℕ, a ≥ 3 ∧ n ≥ 2 ∧ n * (2 * a + n - 1) = 60) ↔ ∃! a n : ℕ, a ≥ 3 ∧ n ≥ 2 ∧ n * (2 * a + n - 1) = 60 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_integers_sum_to_thirty_unique_sets_l603_60374


namespace NUMINAMATH_GPT_final_price_correct_l603_60368

noncomputable def original_price : ℝ := 49.99
noncomputable def first_discount : ℝ := 0.10
noncomputable def second_discount : ℝ := 0.20

theorem final_price_correct :
  let price_after_first_discount := original_price * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  price_after_second_discount = 36.00 := by
    -- The proof would go here
    sorry

end NUMINAMATH_GPT_final_price_correct_l603_60368


namespace NUMINAMATH_GPT_part1_assoc_eq_part2_k_range_part3_m_range_l603_60314

-- Part 1
theorem part1_assoc_eq (x : ℝ) :
  (2 * (x + 1) - x = -3 ∧ (-4 < x ∧ x ≤ 4)) ∨ 
  ((x+1)/3 - 1 = x ∧ (-4 < x ∧ x ≤ 4)) ∨ 
  (2 * x - 7 = 0 ∧ (-4 < x ∧ x ≤ 4)) :=
sorry

-- Part 2
theorem part2_k_range (k : ℝ) :
  (∀ (x : ℝ), (x = (k + 6) / 2) → -5 < x ∧ x ≤ -3) ↔ (-16 < k) ∧ (k ≤ -12) :=
sorry 

-- Part 3
theorem part3_m_range (m : ℝ) :
  (∀ (x : ℝ), (x = 6 * m - 5) → (0 < x) ∧ (x ≤ 3 * m + 1) ∧ (1 ≤ x) ∧ (x ≤ 3)) ↔ (5/6 < m) ∧ (m < 1) :=
sorry

end NUMINAMATH_GPT_part1_assoc_eq_part2_k_range_part3_m_range_l603_60314


namespace NUMINAMATH_GPT_sophia_estimate_larger_l603_60307

theorem sophia_estimate_larger (x y a b : ℝ) (hx : x > y) (hy : y > 0) (ha : a > 0) (hb : b > 0) :
  (x + a) - (y - b) > x - y := by
  sorry

end NUMINAMATH_GPT_sophia_estimate_larger_l603_60307


namespace NUMINAMATH_GPT_area_of_quadrilateral_l603_60394

theorem area_of_quadrilateral (d h1 h2 : ℝ) (hd : d = 20) (hh1 : h1 = 9) (hh2 : h2 = 6) :
  (1 / 2) * d * (h1 + h2) = 150 :=
by
  rw [hd, hh1, hh2]
  norm_num

end NUMINAMATH_GPT_area_of_quadrilateral_l603_60394


namespace NUMINAMATH_GPT_perfect_square_trinomial_k_l603_60329

theorem perfect_square_trinomial_k (k : ℤ) : 
  (∀ x : ℝ, x^2 - k*x + 64 = (x + 8)^2 ∨ x^2 - k*x + 64 = (x - 8)^2) → 
  (k = 16 ∨ k = -16) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_k_l603_60329


namespace NUMINAMATH_GPT_plane_equation_through_point_parallel_l603_60301

theorem plane_equation_through_point_parallel (A B C D : ℤ) (hx hy hz : ℤ) (x y z : ℤ)
  (h_point : (A, B, C, D) = (-2, 1, -3, 10))
  (h_coordinates : (hx, hy, hz) = (2, -3, 1))
  (h_plane_parallel : ∀ x y z, -2 * x + y - 3 * z = 7 ↔ A * x + B * y + C * z + D = 0)
  (h_form : A > 0):
  ∃ A' B' C' D', A' * (x : ℤ) + B' * (y : ℤ) + C' * (z : ℤ) + D' = 0 :=
by
  sorry

end NUMINAMATH_GPT_plane_equation_through_point_parallel_l603_60301


namespace NUMINAMATH_GPT_scaling_transformation_l603_60347

theorem scaling_transformation (a b : ℝ) :
  (∀ x y : ℝ, (y = 1 - x → y' = b * (1 - x))
    → (y' = b - b * x)) 
  ∧
  (∀ x' y' : ℝ, (y = (2 / 3) * x' + 2)
    → (y' = (2 / 3) * (a * x) + 2))
  → a = 3 ∧ b = 2 := by
  sorry

end NUMINAMATH_GPT_scaling_transformation_l603_60347


namespace NUMINAMATH_GPT_smallest_three_digit_candy_number_l603_60335

theorem smallest_three_digit_candy_number (n : ℕ) (hn1 : 100 ≤ n) (hn2 : n ≤ 999)
    (h1 : (n + 6) % 9 = 0) (h2 : (n - 9) % 6 = 0) : n = 111 := by
  sorry

end NUMINAMATH_GPT_smallest_three_digit_candy_number_l603_60335


namespace NUMINAMATH_GPT_average_donation_proof_l603_60366

noncomputable def average_donation (total_people : ℝ) (donated_200 : ℝ) (donated_100 : ℝ) (donated_50 : ℝ) : ℝ :=
  let proportion_200 := donated_200 / total_people
  let proportion_100 := donated_100 / total_people
  let proportion_50 := donated_50 / total_people
  let total_donation := (200 * proportion_200) + (100 * proportion_100) + (50 * proportion_50)
  total_donation

theorem average_donation_proof 
  (total_people : ℝ)
  (donated_200 donated_100 donated_50 : ℝ)
  (h1 : proportion_200 = 1 / 10)
  (h2 : proportion_100 = 3 / 4)
  (h3 : proportion_50 = 1 - proportion_200 - proportion_100) :
  average_donation total_people donated_200 donated_100 donated_50 = 102.5 :=
  by 
    sorry

end NUMINAMATH_GPT_average_donation_proof_l603_60366


namespace NUMINAMATH_GPT_range_of_a_l603_60321

def prop_p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a (a : ℝ) : (prop_p a ∨ prop_q a) ∧ ¬(prop_p a ∧ prop_q a) ↔ (a < 0 ∨ (1 / 4 < a ∧ a < 4)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l603_60321


namespace NUMINAMATH_GPT_gcd_360_504_is_72_l603_60378

theorem gcd_360_504_is_72 : Nat.gcd 360 504 = 72 := by
  sorry

end NUMINAMATH_GPT_gcd_360_504_is_72_l603_60378


namespace NUMINAMATH_GPT_gmat_test_statistics_l603_60358

theorem gmat_test_statistics 
    (p1 : ℝ) (p2 : ℝ) (p12 : ℝ) (neither : ℝ) (S : ℝ) 
    (h1 : p1 = 0.85)
    (h2 : p12 = 0.60) 
    (h3 : neither = 0.05) :
    0.25 + S = 0.95 → S = 0.70 :=
by
  sorry

end NUMINAMATH_GPT_gmat_test_statistics_l603_60358


namespace NUMINAMATH_GPT_part_a_part_b_l603_60383

noncomputable def tsunami_area_center_face (l : ℝ) (v : ℝ) (t : ℝ) : ℝ :=
  180000 * Real.pi + 270000 * Real.sqrt 3

noncomputable def tsunami_area_mid_edge (l : ℝ) (v : ℝ) (t : ℝ) : ℝ :=
  720000 * Real.arccos (3 / 4) + 135000 * Real.sqrt 7

theorem part_a (l v t : ℝ) (hl : l = 900) (hv : v = 300) (ht : t = 2) :
  tsunami_area_center_face l v t = 180000 * Real.pi + 270000 * Real.sqrt 3 :=
by
  sorry

theorem part_b (l v t : ℝ) (hl : l = 900) (hv : v = 300) (ht : t = 2) :
  tsunami_area_mid_edge l v t = 720000 * Real.arccos (3 / 4) + 135000 * Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l603_60383


namespace NUMINAMATH_GPT_evan_runs_200_more_feet_l603_60343

def street_width : ℕ := 25
def block_side : ℕ := 500

def emily_path : ℕ := 4 * block_side
def evan_path : ℕ := 4 * (block_side + 2 * street_width)

theorem evan_runs_200_more_feet : evan_path - emily_path = 200 := by
  sorry

end NUMINAMATH_GPT_evan_runs_200_more_feet_l603_60343


namespace NUMINAMATH_GPT_box_made_by_Bellini_or_son_l603_60317

-- Definitions of the conditions
variable (B : Prop) -- Bellini made the box
variable (S : Prop) -- Bellini's son made the box
variable (inscription_true : Prop) -- The inscription "I made this box" is truthful

-- The problem statement in Lean: Prove that B or S given the inscription is true
theorem box_made_by_Bellini_or_son (B S inscription_true : Prop) (h1 : inscription_true → (B ∨ S)) : B ∨ S :=
by
  sorry

end NUMINAMATH_GPT_box_made_by_Bellini_or_son_l603_60317


namespace NUMINAMATH_GPT_range_of_2x_minus_y_l603_60361

theorem range_of_2x_minus_y (x y : ℝ) (hx : 0 < x ∧ x < 4) (hy : 0 < y ∧ y < 6) : -6 < 2 * x - y ∧ 2 * x - y < 8 := 
sorry

end NUMINAMATH_GPT_range_of_2x_minus_y_l603_60361


namespace NUMINAMATH_GPT_range_of_m_l603_60359

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, (m + 1) * x^2 ≥ 0) : m > -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l603_60359


namespace NUMINAMATH_GPT_small_square_perimeter_l603_60342

-- Condition Definitions
def perimeter_difference := 17
def side_length_of_square (x : ℝ) := 2 * x = perimeter_difference

-- Theorem Statement
theorem small_square_perimeter (x : ℝ) (h : side_length_of_square x) : 4 * x = 34 :=
by
  sorry

end NUMINAMATH_GPT_small_square_perimeter_l603_60342


namespace NUMINAMATH_GPT_greatest_possible_sum_of_squares_l603_60392

theorem greatest_possible_sum_of_squares (x y : ℤ) (h : x^2 + y^2 = 36) : x + y ≤ 8 :=
by sorry

end NUMINAMATH_GPT_greatest_possible_sum_of_squares_l603_60392


namespace NUMINAMATH_GPT_arcsin_add_arccos_eq_pi_div_two_l603_60351

open Real

theorem arcsin_add_arccos_eq_pi_div_two (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  arcsin x + arccos x = (π / 2) :=
sorry

end NUMINAMATH_GPT_arcsin_add_arccos_eq_pi_div_two_l603_60351


namespace NUMINAMATH_GPT_rotate_parabola_180deg_l603_60360

theorem rotate_parabola_180deg (x y : ℝ) :
  (∀ x, y = 2 * x^2 - 12 * x + 16) →
  (∀ x, y = -2 * x^2 + 12 * x - 20) :=
sorry

end NUMINAMATH_GPT_rotate_parabola_180deg_l603_60360


namespace NUMINAMATH_GPT_range_exp3_eq_l603_60356

noncomputable def exp3 (x : ℝ) : ℝ := 3^x

theorem range_exp3_eq (x : ℝ) : Set.range (exp3) = Set.Ioi 0 :=
sorry

end NUMINAMATH_GPT_range_exp3_eq_l603_60356


namespace NUMINAMATH_GPT_compound_interest_time_l603_60380

theorem compound_interest_time 
  (P : ℝ) (r : ℝ) (A₁ : ℝ) (A₂ : ℝ) (t₁ t₂ : ℕ)
  (h1 : r = 0.10)
  (h2 : A₁ = P * (1 + r) ^ t₁)
  (h3 : A₂ = P * (1 + r) ^ t₂)
  (h4 : A₁ = 2420)
  (h5 : A₂ = 2662)
  (h6 : t₂ = t₁ + 3) :
  t₁ = 3 := 
sorry

end NUMINAMATH_GPT_compound_interest_time_l603_60380


namespace NUMINAMATH_GPT_sugar_amount_indeterminate_l603_60364

-- Define the variables and conditions
variable (cups_of_flour_needed : ℕ) (cups_of_sugar_needed : ℕ)
variable (cups_of_flour_put_in : ℕ) (cups_of_flour_to_add : ℕ)

-- Conditions
axiom H1 : cups_of_flour_needed = 8
axiom H2 : cups_of_flour_put_in = 4
axiom H3 : cups_of_flour_to_add = 4

-- Problem statement: Prove that the amount of sugar cannot be determined
theorem sugar_amount_indeterminate (h : cups_of_sugar_needed > 0) :
  cups_of_flour_needed = 8 → cups_of_flour_put_in = 4 → cups_of_flour_to_add = 4 → cups_of_sugar_needed > 0 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sugar_amount_indeterminate_l603_60364


namespace NUMINAMATH_GPT_pqrs_predicate_l603_60367

noncomputable def P (a b c : ℝ) := a + b - c
noncomputable def Q (a b c : ℝ) := b + c - a
noncomputable def R (a b c : ℝ) := c + a - b

theorem pqrs_predicate (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (P a b c) * (Q a b c) * (R a b c) > 0 ↔ (P a b c > 0 ∧ Q a b c > 0 ∧ R a b c > 0) :=
sorry

end NUMINAMATH_GPT_pqrs_predicate_l603_60367


namespace NUMINAMATH_GPT_mehki_age_l603_60377

variable (Mehki Jordyn Zrinka : ℕ)

axiom h1 : Mehki = Jordyn + 10
axiom h2 : Jordyn = 2 * Zrinka
axiom h3 : Zrinka = 6

theorem mehki_age : Mehki = 22 := by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_mehki_age_l603_60377


namespace NUMINAMATH_GPT_probability_one_card_each_l603_60330

-- Define the total number of cards
def total_cards := 12

-- Define the number of cards from Adrian
def adrian_cards := 7

-- Define the number of cards from Bella
def bella_cards := 5

-- Calculate the probability of one card from each cousin when selecting two cards without replacement
theorem probability_one_card_each :
  (adrian_cards / total_cards) * (bella_cards / (total_cards - 1)) +
  (bella_cards / total_cards) * (adrian_cards / (total_cards - 1)) =
  35 / 66 := sorry

end NUMINAMATH_GPT_probability_one_card_each_l603_60330


namespace NUMINAMATH_GPT_solve_for_x_l603_60391

-- Definitions of δ and φ
def delta (x : ℚ) : ℚ := 4 * x + 9
def phi (x : ℚ) : ℚ := 9 * x + 8

-- The main proof statement
theorem solve_for_x :
  ∃ x : ℚ, delta (phi x) = 10 ∧ x = -31 / 36 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l603_60391


namespace NUMINAMATH_GPT_rectangle_side_length_relation_l603_60349

variable (x y : ℝ)

-- Condition: The area of the rectangle is 10
def is_rectangle_area_10 (x y : ℝ) : Prop := x * y = 10

-- Theorem: Given the area condition, express y in terms of x
theorem rectangle_side_length_relation (h : is_rectangle_area_10 x y) : y = 10 / x :=
sorry

end NUMINAMATH_GPT_rectangle_side_length_relation_l603_60349


namespace NUMINAMATH_GPT_ticket_costs_l603_60323

-- Define the conditions
def cost_per_ticket : ℕ := 44
def number_of_tickets : ℕ := 7

-- Define the total cost calculation
def total_cost : ℕ := cost_per_ticket * number_of_tickets

-- Prove that given the conditions, the total cost is 308
theorem ticket_costs :
  total_cost = 308 :=
by
  -- Proof steps here
  sorry

end NUMINAMATH_GPT_ticket_costs_l603_60323


namespace NUMINAMATH_GPT_circuit_disconnected_scenarios_l603_60332

def num_scenarios_solder_points_fall_off (n : Nat) : Nat :=
  2 ^ n - 1

theorem circuit_disconnected_scenarios : num_scenarios_solder_points_fall_off 6 = 63 :=
by
  sorry

end NUMINAMATH_GPT_circuit_disconnected_scenarios_l603_60332


namespace NUMINAMATH_GPT_simplify_expression_l603_60346

variable (a b : ℝ)

theorem simplify_expression : a + (3 * a - 3 * b) - (a - 2 * b) = 3 * a - b := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l603_60346


namespace NUMINAMATH_GPT_sequence_bk_bl_sum_l603_60315

theorem sequence_bk_bl_sum (b : ℕ → ℕ) (m : ℕ) 
  (h_pairwise_distinct : ∀ i j, i ≠ j → b i ≠ b j)
  (h_b0 : b 0 = 0)
  (h_b_lt_2n : ∀ n, 0 < n → b n < 2 * n) :
  ∃ k ℓ : ℕ, b k + b ℓ = m := 
  sorry

end NUMINAMATH_GPT_sequence_bk_bl_sum_l603_60315


namespace NUMINAMATH_GPT_triangle_bisector_length_l603_60396

theorem triangle_bisector_length (A B C : Type) [MetricSpace A] [MetricSpace B]
  [MetricSpace C] (angle_A angle_C : ℝ) (AC AB : ℝ) 
  (hAC : angle_A = 20) (hAngle_C : angle_C = 40) (hAC_minus_AB : AC - AB = 5) : ∃ BM : ℝ, BM = 5 :=
by
  sorry

end NUMINAMATH_GPT_triangle_bisector_length_l603_60396


namespace NUMINAMATH_GPT_divisor_is_seven_l603_60376

theorem divisor_is_seven 
  (d x : ℤ)
  (h1 : x % d = 5)
  (h2 : 4 * x % d = 6) :
  d = 7 := 
sorry

end NUMINAMATH_GPT_divisor_is_seven_l603_60376


namespace NUMINAMATH_GPT_unique_solution_l603_60362

theorem unique_solution (a n : ℕ) (h₁ : 0 < a) (h₂ : 0 < n) (h₃ : 3^n = a^2 - 16) : a = 5 ∧ n = 2 :=
by
sorry

end NUMINAMATH_GPT_unique_solution_l603_60362


namespace NUMINAMATH_GPT_geometric_sequence_a6_l603_60373

variable {α : Type} [LinearOrderedSemiring α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∃ a₁ q : α, ∀ n, a n = a₁ * q ^ n

theorem geometric_sequence_a6 
  (a : ℕ → α) 
  (h_seq : is_geometric_sequence a) 
  (h1 : a 2 + a 4 = 20) 
  (h2 : a 3 + a 5 = 40) : 
  a 6 = 64 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a6_l603_60373


namespace NUMINAMATH_GPT_lieutenant_age_l603_60339

variables (n x : ℕ) 

-- Condition 1: Number of soldiers is the same in both formations
def total_soldiers_initial (n : ℕ) : ℕ := n * (n + 5)
def total_soldiers_new (n x : ℕ) : ℕ := x * (n + 9)

-- Condition 2: The number of soldiers is the same 
-- and Condition 3: Equations relating n and x
theorem lieutenant_age (n x : ℕ) (h1: total_soldiers_initial n = total_soldiers_new n x) (h2 : x = 24) : 
  x = 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_lieutenant_age_l603_60339


namespace NUMINAMATH_GPT_geometric_sequence_fifth_term_l603_60363

theorem geometric_sequence_fifth_term 
    (a₁ : ℕ) (a₄ : ℕ) (r : ℕ) (a₅ : ℕ)
    (h₁ : a₁ = 3) (h₂ : a₄ = 240) 
    (h₃ : a₄ = a₁ * r^3) 
    (h₄ : a₅ = a₁ * r^4) : 
    a₅ = 768 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_fifth_term_l603_60363


namespace NUMINAMATH_GPT_cost_price_per_meter_l603_60336

-- Definitions based on the conditions given in the problem
def meters_of_cloth : ℕ := 45
def selling_price : ℕ := 4500
def profit_per_meter : ℕ := 12

-- Statement to prove
theorem cost_price_per_meter :
  (selling_price - (profit_per_meter * meters_of_cloth)) / meters_of_cloth = 88 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_per_meter_l603_60336


namespace NUMINAMATH_GPT_one_add_i_cubed_eq_one_sub_i_l603_60385

theorem one_add_i_cubed_eq_one_sub_i (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i :=
sorry

end NUMINAMATH_GPT_one_add_i_cubed_eq_one_sub_i_l603_60385


namespace NUMINAMATH_GPT_find_number_l603_60333

theorem find_number (x : ℝ) (h : 0.50 * x = 48 + 180) : x = 456 :=
sorry

end NUMINAMATH_GPT_find_number_l603_60333


namespace NUMINAMATH_GPT_perimeter_of_similar_triangle_l603_60353

def is_isosceles (a b c : ℕ) : Prop :=
  (a = b) ∨ (b = c) ∨ (a = c)

theorem perimeter_of_similar_triangle (a b c : ℕ) (d e f : ℕ) 
  (h1 : is_isosceles a b c) (h2 : min a b = 15) (h3 : min (min a b) c = 15)
  (h4 : d = 75) (h5 : (d / 15) = e / b) (h6 : f = e) :
  d + e + f = 375 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_similar_triangle_l603_60353


namespace NUMINAMATH_GPT_equal_elements_l603_60305

theorem equal_elements {n : ℕ} (a : ℕ → ℝ) (h₁ : n ≥ 2) (h₂ : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i ≠ -1) 
  (h₃ : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a (i + 2) = (a i ^ 2 + a i) / (a (i + 1) + 1)) 
  (hn1 : a (n + 1) = a 1) (hn2 : a (n + 2) = a 2) :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i = a 1 := by
  sorry

end NUMINAMATH_GPT_equal_elements_l603_60305


namespace NUMINAMATH_GPT_second_number_is_180_l603_60384

theorem second_number_is_180 
  (x : ℝ) 
  (first : ℝ := 2 * x) 
  (third : ℝ := (1/3) * first)
  (h : first + x + third = 660) : 
  x = 180 :=
sorry

end NUMINAMATH_GPT_second_number_is_180_l603_60384


namespace NUMINAMATH_GPT_city_mileage_per_tankful_l603_60389

theorem city_mileage_per_tankful :
  ∀ (T : ℝ), 
  ∃ (city_miles : ℝ),
    (462 = T * (32 + 12)) ∧
    (city_miles = 32 * T) ∧
    (city_miles = 336) :=
by
  sorry

end NUMINAMATH_GPT_city_mileage_per_tankful_l603_60389


namespace NUMINAMATH_GPT_g_range_l603_60386

noncomputable def g (x y z : ℝ) : ℝ :=
  (x ^ 2) / (x ^ 2 + y ^ 2) +
  (y ^ 2) / (y ^ 2 + z ^ 2) +
  (z ^ 2) / (z ^ 2 + x ^ 2)

theorem g_range (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  1 < g x y z ∧ g x y z < 2 :=
  sorry

end NUMINAMATH_GPT_g_range_l603_60386


namespace NUMINAMATH_GPT_find_first_term_l603_60382

theorem find_first_term
  (S : ℝ) (a r : ℝ)
  (h1 : S = 10)
  (h2 : a + a * r = 6)
  (h3 : a = 2 * r) :
  a = -1 + Real.sqrt 13 ∨ a = -1 - Real.sqrt 13 := by
  sorry

end NUMINAMATH_GPT_find_first_term_l603_60382


namespace NUMINAMATH_GPT_abs_neg_implies_nonpositive_l603_60341

theorem abs_neg_implies_nonpositive (a : ℝ) : |a| = -a → a ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_abs_neg_implies_nonpositive_l603_60341


namespace NUMINAMATH_GPT_maximum_value_of_function_l603_60399

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 - 2 * Real.sin x - 2

theorem maximum_value_of_function :
  ∃ x : ℝ, f x = 1 ∧ ∀ y : ℝ, -1 ≤ Real.sin y ∧ Real.sin y ≤ 1 → f y ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_function_l603_60399


namespace NUMINAMATH_GPT_women_in_luxury_suites_count_l603_60354

noncomputable def passengers : ℕ := 300
noncomputable def percentage_women : ℝ := 70 / 100
noncomputable def percentage_luxury : ℝ := 15 / 100

noncomputable def women_on_ship : ℝ := passengers * percentage_women
noncomputable def women_in_luxury_suites : ℝ := women_on_ship * percentage_luxury

theorem women_in_luxury_suites_count : 
  round women_in_luxury_suites = 32 :=
by sorry

end NUMINAMATH_GPT_women_in_luxury_suites_count_l603_60354


namespace NUMINAMATH_GPT_part1_part2_l603_60309

open Set

variable {α : Type*} [LinearOrderedField α]

def A : Set α := { x | abs (x - 1) ≤ 1 }
def B (a : α) : Set α := { x | x ≥ a }

theorem part1 {x : α} : x ∈ (A ∩ B 1) ↔ 1 ≤ x ∧ x ≤ 2 := by
  sorry

theorem part2 {a : α} : (A ⊆ B a) ↔ a ≤ 0 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l603_60309


namespace NUMINAMATH_GPT_rosie_can_make_nine_pies_l603_60387

theorem rosie_can_make_nine_pies (apples pies : ℕ) (h : apples = 12 ∧ pies = 3) : 36 / (12 / 3) * pies = 9 :=
by
  sorry

end NUMINAMATH_GPT_rosie_can_make_nine_pies_l603_60387


namespace NUMINAMATH_GPT_minimize_relative_waiting_time_l603_60319

-- Definitions of task times in seconds
def task_U : ℕ := 10
def task_V : ℕ := 120
def task_W : ℕ := 900

-- Definition of relative waiting time given a sequence of task execution times
def relative_waiting_time (times : List ℕ) : ℚ :=
  (times.head! : ℚ) / (times.head! : ℚ) + 
  (times.head! + times.tail.head! : ℚ) / (times.tail.head! : ℚ) + 
  (times.head! + times.tail.head! + times.tail.tail.head! : ℚ) / (times.tail.tail.head! : ℚ)

-- Sequences
def sequence_A : List ℕ := [task_U, task_V, task_W]
def sequence_B : List ℕ := [task_V, task_W, task_U]
def sequence_C : List ℕ := [task_W, task_U, task_V]
def sequence_D : List ℕ := [task_U, task_W, task_V]

-- Sum of relative waiting times for each sequence
def S_A := relative_waiting_time sequence_A
def S_B := relative_waiting_time sequence_B
def S_C := relative_waiting_time sequence_C
def S_D := relative_waiting_time sequence_D

-- Theorem to prove that sequence A has the minimum sum of relative waiting times
theorem minimize_relative_waiting_time : S_A < S_B ∧ S_A < S_C ∧ S_A < S_D := 
  by sorry

end NUMINAMATH_GPT_minimize_relative_waiting_time_l603_60319


namespace NUMINAMATH_GPT_number_of_x_intercepts_l603_60395

def parabola (y : ℝ) : ℝ := -3 * y ^ 2 + 2 * y + 3

theorem number_of_x_intercepts : (∃ y : ℝ, parabola y = 0) ∧ (∃! x : ℝ, parabola x = 0) :=
by
  sorry

end NUMINAMATH_GPT_number_of_x_intercepts_l603_60395


namespace NUMINAMATH_GPT_cost_price_of_watch_l603_60326

theorem cost_price_of_watch (CP : ℝ) (h1 : SP1 = CP * 0.64) (h2 : SP2 = CP * 1.04) (h3 : SP2 = SP1 + 140) : CP = 350 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_watch_l603_60326


namespace NUMINAMATH_GPT_gcd_of_polynomial_l603_60331

def multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

theorem gcd_of_polynomial (b : ℕ) (h : multiple_of b 456) :
  Nat.gcd (4 * b^3 + b^2 + 6 * b + 152) b = 152 := sorry

end NUMINAMATH_GPT_gcd_of_polynomial_l603_60331


namespace NUMINAMATH_GPT_simplify_and_rationalize_l603_60365

theorem simplify_and_rationalize :
  ( (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 10 / Real.sqrt 15) *
    (Real.sqrt 12 / Real.sqrt 20) = 1 / 3 ) :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_rationalize_l603_60365


namespace NUMINAMATH_GPT_area_of_parallelogram_l603_60312

variable (b : ℕ)
variable (h : ℕ)
variable (A : ℕ)

-- Condition: The height is twice the base.
def height_twice_base := h = 2 * b

-- Condition: The base is 9.
def base_is_9 := b = 9

-- Condition: The area of the parallelogram is base times height.
def area_formula := A = b * h

-- Question: Prove that the area of the parallelogram is 162.
theorem area_of_parallelogram 
  (h_twice : height_twice_base h b) 
  (b_val : base_is_9 b) 
  (area_form : area_formula A b h): A = 162 := 
sorry

end NUMINAMATH_GPT_area_of_parallelogram_l603_60312


namespace NUMINAMATH_GPT_choose_correct_graph_l603_60345

noncomputable def appropriate_graph : String :=
  let bar_graph := "Bar graph"
  let pie_chart := "Pie chart"
  let line_graph := "Line graph"
  let freq_dist_graph := "Frequency distribution graph"
  
  if (bar_graph = "Bar graph") ∧ (pie_chart = "Pie chart") ∧ (line_graph = "Line graph") ∧ (freq_dist_graph = "Frequency distribution graph") then
    "Line graph"
  else
    sorry

theorem choose_correct_graph :
  appropriate_graph = "Line graph" :=
by
  sorry

end NUMINAMATH_GPT_choose_correct_graph_l603_60345


namespace NUMINAMATH_GPT_sarah_rye_flour_l603_60324

-- Definitions
variables (b c p t r : ℕ)

-- Conditions
def condition1 : Prop := b = 10
def condition2 : Prop := c = 3
def condition3 : Prop := p = 2
def condition4 : Prop := t = 20

-- Proposition to prove
theorem sarah_rye_flour : condition1 b → condition2 c → condition3 p → condition4 t → r = t - (b + c + p) → r = 5 :=
by
  intros h1 h2 h3 h4 hr
  rw [h1, h2, h3, h4] at hr
  exact hr

end NUMINAMATH_GPT_sarah_rye_flour_l603_60324


namespace NUMINAMATH_GPT_geom_seq_product_l603_60318

-- Given conditions
variables (a : ℕ → ℝ)
variable (r : ℝ)
axiom geom_seq (n : ℕ) : a (n + 1) = a n * r
axiom a1_eq_1 : a 1 = 1
axiom a10_eq_3 : a 10 = 3

-- Proof goal
theorem geom_seq_product : a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 81 :=  
sorry

end NUMINAMATH_GPT_geom_seq_product_l603_60318


namespace NUMINAMATH_GPT_square_side_length_is_10_l603_60352

-- Define the side lengths of the original squares
def side_length1 : ℝ := 8
def side_length2 : ℝ := 6

-- Define the areas of the original squares
def area1 : ℝ := side_length1^2
def area2 : ℝ := side_length2^2

-- Define the total area of the combined squares
def total_area : ℝ := area1 + area2

-- Define the side length of the new square
def side_length_new_square : ℝ := 10

-- Theorem statement to prove that the side length of the new square is 10 cm
theorem square_side_length_is_10 : side_length_new_square^2 = total_area := by
  sorry

end NUMINAMATH_GPT_square_side_length_is_10_l603_60352


namespace NUMINAMATH_GPT_hyperbola_asymptote_m_value_l603_60393

theorem hyperbola_asymptote_m_value
  (m : ℝ)
  (h1 : m > 0)
  (h2 : ∀ x y : ℝ, (5 * x - 2 * y = 0) → ((x^2 / 4) - (y^2 / m^2) = 1)) :
  m = 5 :=
sorry

end NUMINAMATH_GPT_hyperbola_asymptote_m_value_l603_60393


namespace NUMINAMATH_GPT_problem_27_integer_greater_than_B_over_pi_l603_60337

noncomputable def B : ℕ := 22

theorem problem_27_integer_greater_than_B_over_pi :
  Nat.ceil (B / Real.pi) = 8 := sorry

end NUMINAMATH_GPT_problem_27_integer_greater_than_B_over_pi_l603_60337


namespace NUMINAMATH_GPT_final_cost_is_35_l603_60300

-- Definitions based on conditions
def original_price : ℕ := 50
def discount_rate : ℚ := 0.30
def discount_amount : ℚ := original_price * discount_rate
def final_cost : ℚ := original_price - discount_amount

-- The theorem we need to prove
theorem final_cost_is_35 : final_cost = 35 := by
  sorry

end NUMINAMATH_GPT_final_cost_is_35_l603_60300


namespace NUMINAMATH_GPT_consecutive_product_not_mth_power_l603_60308

theorem consecutive_product_not_mth_power (n m k : ℕ) :
  ¬ ∃ k, (n - 1) * n * (n + 1) = k^m := 
sorry

end NUMINAMATH_GPT_consecutive_product_not_mth_power_l603_60308


namespace NUMINAMATH_GPT_range_of_S_on_ellipse_l603_60398

theorem range_of_S_on_ellipse :
  ∀ (x y : ℝ),
    (x ^ 2 / 2 + y ^ 2 / 3 = 1) →
    -Real.sqrt 5 ≤ x + y ∧ x + y ≤ Real.sqrt 5 :=
by
  intro x y
  intro h
  sorry

end NUMINAMATH_GPT_range_of_S_on_ellipse_l603_60398


namespace NUMINAMATH_GPT_GreenValley_Absent_Percentage_l603_60371

theorem GreenValley_Absent_Percentage 
  (total_students boys girls absent_boys_frac absent_girls_frac : ℝ)
  (h1 : total_students = 120)
  (h2 : boys = 70)
  (h3 : girls = 50)
  (h4 : absent_boys_frac = 1 / 7)
  (h5 : absent_girls_frac = 1 / 5) :
  (absent_boys_frac * boys + absent_girls_frac * girls) / total_students * 100 = 16.67 := 
sorry

end NUMINAMATH_GPT_GreenValley_Absent_Percentage_l603_60371


namespace NUMINAMATH_GPT_num_coloring_l603_60327

-- Define the set of numbers to be colored
def numbers_to_color : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9]

-- Define the set of colors
inductive Color
| red
| green
| blue

-- Define proper divisors for the numbers in the list
def proper_divisors (n : ℕ) : List ℕ :=
  match n with
  | 2 => []
  | 3 => []
  | 4 => [2]
  | 5 => []
  | 6 => [2, 3]
  | 7 => []
  | 8 => [2, 4]
  | 9 => [3]
  | _ => []

-- The proof statement
theorem num_coloring (h : ∀ n ∈ numbers_to_color, ∀ d ∈ proper_divisors n, n ≠ d) :
  ∃ f : ℕ → Color, ∀ n ∈ numbers_to_color, ∀ d ∈ proper_divisors n, f n ≠ f d :=
  sorry

end NUMINAMATH_GPT_num_coloring_l603_60327


namespace NUMINAMATH_GPT_roger_owes_correct_amount_l603_60316

def initial_house_price : ℝ := 100000
def down_payment_percentage : ℝ := 0.20
def parents_payment_percentage : ℝ := 0.30

def down_payment : ℝ := down_payment_percentage * initial_house_price
def remaining_after_down_payment : ℝ := initial_house_price - down_payment
def parents_payment : ℝ := parents_payment_percentage * remaining_after_down_payment
def money_owed : ℝ := remaining_after_down_payment - parents_payment

theorem roger_owes_correct_amount :
  money_owed = 56000 := by
  sorry

end NUMINAMATH_GPT_roger_owes_correct_amount_l603_60316


namespace NUMINAMATH_GPT_gcd_of_differences_l603_60313

theorem gcd_of_differences (a b c : ℕ) (h1 : a = 794) (h2 : b = 858) (h3 : c = 1351) : 
  Nat.gcd (Nat.gcd (b - a) (c - b)) (c - a) = 4 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_differences_l603_60313


namespace NUMINAMATH_GPT_problem1_problem2_l603_60340

noncomputable def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}

noncomputable def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

variable (a : ℝ) (x : ℝ)

-- Problem 1: Proving intersection of sets when a = 2
theorem problem1 (ha : a = 2) : (A a ∩ B a) = {x | 4 < x ∧ x < 5} :=
sorry

-- Problem 2: Proving the range of a for which B is a subset of A
theorem problem2 : {a | B a ⊆ A a} = {a | (1 < a ∧ a ≤ 3) ∨ a = -1} :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l603_60340


namespace NUMINAMATH_GPT_det_A_l603_60390

def A : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![2, -4, 5],
  ![0, 6, -2],
  ![3, -1, 2]
]

theorem det_A : A.det = -46 := by
  sorry

end NUMINAMATH_GPT_det_A_l603_60390


namespace NUMINAMATH_GPT_no_solution_k_eq_7_l603_60350

-- Define the condition that x should not be equal to 4 and 8
def condition (x : ℝ) : Prop := x ≠ 4 ∧ x ≠ 8

-- Define the equation
def equation (x k : ℝ) : Prop := (x - 3) / (x - 4) = (x - k) / (x - 8)

-- Prove that for the equation to have no solution, k must be 7
theorem no_solution_k_eq_7 : (∀ x, condition x → ¬ equation x 7) ↔ (∃ k, k = 7) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_k_eq_7_l603_60350


namespace NUMINAMATH_GPT_smallest_multiple_l603_60310

theorem smallest_multiple (x : ℕ) (h1 : x % 24 = 0) (h2 : x % 36 = 0) (h3 : x % 20 ≠ 0) :
  x = 72 :=
by
  sorry

end NUMINAMATH_GPT_smallest_multiple_l603_60310


namespace NUMINAMATH_GPT_find_second_number_l603_60355

theorem find_second_number (x : ℕ) :
  22030 = (555 + x) * 2 * (x - 555) + 30 → 
  x = 564 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_second_number_l603_60355


namespace NUMINAMATH_GPT_continuity_at_x0_l603_60325

def f (x : ℝ) : ℝ := -5 * x ^ 2 - 8
def x0 : ℝ := 2

theorem continuity_at_x0 :
  ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧ (∀ (x : ℝ), abs (x - x0) < δ → abs (f x - f x0) < ε) :=
by
  sorry

end NUMINAMATH_GPT_continuity_at_x0_l603_60325


namespace NUMINAMATH_GPT_solve_sqrt_equation_l603_60338

open Real

theorem solve_sqrt_equation :
  ∀ x : ℝ, (sqrt ((3*x - 1) / (x + 4)) + 3 - 4 * sqrt ((x + 4) / (3*x - 1)) = 0) →
    (3*x - 1) / (x + 4) ≥ 0 →
    (x + 4) / (3*x - 1) ≥ 0 →
    x = 5 / 2 := by
  sorry

end NUMINAMATH_GPT_solve_sqrt_equation_l603_60338


namespace NUMINAMATH_GPT_mean_score_74_9_l603_60322

/-- 
In a class of 100 students, the score distribution is as follows:
- 10 students scored 100%
- 15 students scored 90%
- 20 students scored 80%
- 30 students scored 70%
- 20 students scored 60%
- 4 students scored 50%
- 1 student scored 40%

Prove that the mean percentage score of the class is 74.9.
-/
theorem mean_score_74_9 : 
  let scores := [100, 90, 80, 70, 60, 50, 40]
  let counts := [10, 15, 20, 30, 20, 4, 1]
  let total_students := 100
  let total_score := 1000 + 1350 + 1600 + 2100 + 1200 + 200 + 40
  (total_score / total_students : ℝ) = 74.9 :=
by {
  -- The detailed proof steps are omitted with sorry.
  sorry
}

end NUMINAMATH_GPT_mean_score_74_9_l603_60322


namespace NUMINAMATH_GPT_no_integer_solution_for_conditions_l603_60370

theorem no_integer_solution_for_conditions :
  ¬∃ (x : ℤ), 
    (18 + x = 2 * (5 + x)) ∧
    (18 + x = 3 * (2 + x)) ∧
    ((18 + x) + (5 + x) + (2 + x) = 50) :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solution_for_conditions_l603_60370


namespace NUMINAMATH_GPT_difference_in_roi_l603_60375

theorem difference_in_roi (E_investment : ℝ) (B_investment : ℝ) (E_rate : ℝ) (B_rate : ℝ) (years : ℕ) :
  E_investment = 300 → B_investment = 500 → E_rate = 0.15 → B_rate = 0.10 → years = 2 →
  (B_rate * B_investment * years) - (E_rate * E_investment * years) = 10 :=
by
  intros E_investment_eq B_investment_eq E_rate_eq B_rate_eq years_eq
  sorry

end NUMINAMATH_GPT_difference_in_roi_l603_60375


namespace NUMINAMATH_GPT_complete_the_square_example_l603_60334

theorem complete_the_square_example : ∀ x m n : ℝ, (x^2 - 12 * x + 33 = 0) → 
  (x + m)^2 = n → m = -6 ∧ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_complete_the_square_example_l603_60334


namespace NUMINAMATH_GPT_no_valid_angles_l603_60303

open Real

theorem no_valid_angles (θ : ℝ) (h1 : 0 < θ) (h2 : θ < 2 * π)
    (h3 : ∀ k : ℤ, θ ≠ k * (π / 2))
    (h4 : cos θ * tan θ = sin θ ^ 3) : false :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_no_valid_angles_l603_60303


namespace NUMINAMATH_GPT_ticket_distribution_l603_60348

noncomputable def num_dist_methods (n : ℕ) (A : ℕ) (B : ℕ) (C : ℕ) (D : ℕ) : ℕ := sorry

theorem ticket_distribution :
  num_dist_methods 18 5 6 7 10 = 140 := sorry

end NUMINAMATH_GPT_ticket_distribution_l603_60348


namespace NUMINAMATH_GPT_total_numbers_l603_60302

-- Setting up constants and conditions
variables (n : ℕ)
variables (s1 s2 s3 : ℕ → ℝ)

-- Conditions
axiom avg_all : (s1 n + s2 n + s3 n) / n = 2.5
axiom avg_2_1 : s1 2 / 2 = 1.1
axiom avg_2_2 : s2 2 / 2 = 1.4
axiom avg_2_3 : s3 2 / 2 = 5.0

-- Proposed theorem to prove
theorem total_numbers : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_total_numbers_l603_60302


namespace NUMINAMATH_GPT_find_fraction_l603_60397

variable (x y z : ℝ)

theorem find_fraction (h : (x - y) / (z - y) = -10) : (x - z) / (y - z) = 11 := 
by
  sorry

end NUMINAMATH_GPT_find_fraction_l603_60397


namespace NUMINAMATH_GPT_find_a_l603_60311

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + 2

theorem find_a (a : ℝ) (h : (3 * a * (-1 : ℝ)^2) = 3) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l603_60311
