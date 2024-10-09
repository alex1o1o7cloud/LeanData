import Mathlib

namespace jonah_total_ingredients_in_cups_l2093_209367

noncomputable def volume_of_ingredients_in_cups : ℝ :=
  let yellow_raisins := 0.3
  let black_raisins := 0.4
  let almonds_in_ounces := 5.5
  let pumpkin_seeds_in_grams := 150
  let ounce_to_cup_conversion := 0.125
  let gram_to_cup_conversion := 0.00423
  let almonds := almonds_in_ounces * ounce_to_cup_conversion
  let pumpkin_seeds := pumpkin_seeds_in_grams * gram_to_cup_conversion
  yellow_raisins + black_raisins + almonds + pumpkin_seeds

theorem jonah_total_ingredients_in_cups : volume_of_ingredients_in_cups = 2.022 :=
by
  sorry

end jonah_total_ingredients_in_cups_l2093_209367


namespace inequality_proof_l2093_209395

noncomputable def a := (1 / 4) * Real.logb 2 3
noncomputable def b := 1 / 2
noncomputable def c := (1 / 2) * Real.logb 5 3

theorem inequality_proof : c < a ∧ a < b :=
by
  sorry

end inequality_proof_l2093_209395


namespace rectangle_area_is_correct_l2093_209387

noncomputable def inscribed_rectangle_area (r : ℝ) (l_to_w_ratio : ℝ) : ℝ :=
  let width := 2 * r
  let length := l_to_w_ratio * width
  length * width

theorem rectangle_area_is_correct :
  inscribed_rectangle_area 7 3 = 588 :=
  by
    -- The proof goes here
    sorry

end rectangle_area_is_correct_l2093_209387


namespace train_probability_at_station_l2093_209346

-- Define time intervals
def t0 := 0 -- Train arrival start time in minutes after 1:00 PM
def t1 := 60 -- Train arrival end time in minutes after 1:00 PM
def a0 := 0 -- Alex arrival start time in minutes after 1:00 PM
def a1 := 120 -- Alex arrival end time in minutes after 1:00 PM

-- Define the probability calculation problem
theorem train_probability_at_station :
  let total_area := (t1 - t0) * (a1 - a0)
  let overlap_area := (1/2 * 50 * 50) + (10 * 55)
  (overlap_area / total_area) = 1/4 := 
by
  sorry

end train_probability_at_station_l2093_209346


namespace range_of_m_l2093_209345

theorem range_of_m (h : ¬ (∀ x : ℝ, ∃ m : ℝ, 4 ^ x - 2 ^ (x + 1) + m = 0) → false) : 
  ∀ m : ℝ, m ≤ 1 :=
by
  sorry

end range_of_m_l2093_209345


namespace time_to_cross_platform_l2093_209389

-- Definitions of the given conditions
def train_length : ℝ := 900
def time_to_cross_pole : ℝ := 18
def platform_length : ℝ := 1050

-- Goal statement in Lean 4 format
theorem time_to_cross_platform : 
  let speed := train_length / time_to_cross_pole;
  let total_distance := train_length + platform_length;
  let time := total_distance / speed;
  time = 39 := 
by
  sorry

end time_to_cross_platform_l2093_209389


namespace fraction_division_l2093_209332

theorem fraction_division :
  (1/4) / 2 = 1/8 :=
by
  sorry

end fraction_division_l2093_209332


namespace hannah_practice_hours_l2093_209312

theorem hannah_practice_hours (weekend_hours : ℕ) (total_weekly_hours : ℕ) (more_weekday_hours : ℕ)
  (h1 : weekend_hours = 8)
  (h2 : total_weekly_hours = 33)
  (h3 : more_weekday_hours = 17) :
  (total_weekly_hours - weekend_hours) - weekend_hours = more_weekday_hours :=
by
  sorry

end hannah_practice_hours_l2093_209312


namespace valid_permutations_count_l2093_209399

/-- 
Given five elements consisting of the numbers 1, 2, 3, and the symbols "+" and "-", 
we want to count the number of permutations such that no two numbers are adjacent.
-/
def count_valid_permutations : Nat := 
  let number_permutations := Nat.factorial 3 -- 3! permutations of 1, 2, 3
  let symbol_insertions := Nat.factorial 2  -- 2! permutations of "+" and "-"
  number_permutations * symbol_insertions

theorem valid_permutations_count : count_valid_permutations = 12 := by
  sorry

end valid_permutations_count_l2093_209399


namespace discount_percentage_l2093_209327

theorem discount_percentage (coach_cost sectional_cost other_cost paid : ℕ) 
  (h1 : coach_cost = 2500) 
  (h2 : sectional_cost = 3500) 
  (h3 : other_cost = 2000) 
  (h4 : paid = 7200) : 
  ((coach_cost + sectional_cost + other_cost - paid) * 100) / (coach_cost + sectional_cost + other_cost) = 10 :=
by
  sorry

end discount_percentage_l2093_209327


namespace range_of_m_l2093_209359

-- Define the conditions
theorem range_of_m (m : ℝ) : 
    (∀ x : ℝ, (m-1) * x^2 + 2 * x + 1 = 0 → 
     (m-1 ≠ 0) ∧ 
     (4 - 4 * (m - 1) > 0)) ↔ 
    (m < 2 ∧ m ≠ 1) :=
sorry

end range_of_m_l2093_209359


namespace distinct_convex_polygons_l2093_209398

def twelve_points : Finset (Fin 12) := (Finset.univ : Finset (Fin 12))

noncomputable def polygon_count_with_vertices (n : ℕ) : ℕ :=
  2^n - 1 - n - (n * (n - 1)) / 2

theorem distinct_convex_polygons :
  polygon_count_with_vertices 12 = 4017 := 
by
  sorry

end distinct_convex_polygons_l2093_209398


namespace oliver_total_earnings_l2093_209397

/-- Rates for different types of laundry items -/
def rate_regular : ℝ := 3
def rate_delicate : ℝ := 4
def rate_bulky : ℝ := 5

/-- Quantity of laundry items washed over three days -/
def quantity_day1_regular : ℝ := 7
def quantity_day1_delicate : ℝ := 4
def quantity_day1_bulky : ℝ := 2

def quantity_day2_regular : ℝ := 10
def quantity_day2_delicate : ℝ := 6
def quantity_day2_bulky : ℝ := 3

def quantity_day3_regular : ℝ := 20
def quantity_day3_delicate : ℝ := 4
def quantity_day3_bulky : ℝ := 0

/-- Discount on delicate clothes for the third day -/
def discount : ℝ := 0.2

/-- The expected earnings for each day and total -/
def earnings_day1 : ℝ :=
  rate_regular * quantity_day1_regular +
  rate_delicate * quantity_day1_delicate +
  rate_bulky * quantity_day1_bulky

def earnings_day2 : ℝ :=
  rate_regular * quantity_day2_regular +
  rate_delicate * quantity_day2_delicate +
  rate_bulky * quantity_day2_bulky

def earnings_day3 : ℝ :=
  rate_regular * quantity_day3_regular +
  (rate_delicate * quantity_day3_delicate * (1 - discount)) +
  rate_bulky * quantity_day3_bulky

def total_earnings : ℝ := earnings_day1 + earnings_day2 + earnings_day3

theorem oliver_total_earnings : total_earnings = 188.80 := by
  sorry

end oliver_total_earnings_l2093_209397


namespace rectangular_prism_width_l2093_209361

theorem rectangular_prism_width 
  (l : ℝ) (h : ℝ) (d : ℝ) (w : ℝ)
  (hl : l = 5) (hh : h = 7) (hd : d = 14) :
  d = Real.sqrt (l^2 + w^2 + h^2) → w = Real.sqrt 122 :=
by 
  sorry

end rectangular_prism_width_l2093_209361


namespace find_a_and_b_l2093_209321

-- Define the two numbers a and b and the given conditions
variables (a b : ℕ)
variables (h1 : a - b = 831) (h2 : a = 21 * b + 11)

-- State the theorem to find the values of a and b
theorem find_a_and_b (a b : ℕ) (h1 : a - b = 831) (h2 : a = 21 * b + 11) : a = 872 ∧ b = 41 :=
by
  sorry

end find_a_and_b_l2093_209321


namespace solution_set_of_inequality_l2093_209369

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 3 * x - 2 ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end solution_set_of_inequality_l2093_209369


namespace Rachel_total_books_l2093_209304

theorem Rachel_total_books :
  (8 * 15) + (4 * 15) + (3 * 15) + (5 * 15) = 300 :=
by {
  sorry
}

end Rachel_total_books_l2093_209304


namespace distance_two_from_origin_l2093_209391

theorem distance_two_from_origin (x : ℝ) (h : abs x = 2) : x = 2 ∨ x = -2 := by
  sorry

end distance_two_from_origin_l2093_209391


namespace find_function_l2093_209302

theorem find_function (f : ℝ → ℝ) :
  (∀ x y z : ℝ, x + y + z = 0 → f (x^3) + f (y)^3 + f (z)^3 = 3 * x * y * z) → 
  f = id :=
by sorry

end find_function_l2093_209302


namespace loan_amount_is_900_l2093_209370

theorem loan_amount_is_900 (P R T SI : ℕ) (hR : R = 9) (hT : T = 9) (hSI : SI = 729)
    (h_simple_interest : SI = (P * R * T) / 100) : P = 900 := by
  sorry

end loan_amount_is_900_l2093_209370


namespace first_tier_tax_rate_l2093_209358

theorem first_tier_tax_rate (price : ℕ) (total_tax : ℕ) (tier1_limit : ℕ) (tier2_rate : ℝ) (tier1_tax_rate : ℝ) :
  price = 18000 →
  total_tax = 1950 →
  tier1_limit = 11000 →
  tier2_rate = 0.09 →
  ((price - tier1_limit) * tier2_rate + tier1_tax_rate * tier1_limit = total_tax) →
  tier1_tax_rate = 0.12 :=
by
  intros hprice htotal htier1 hrate htax_eq
  sorry

end first_tier_tax_rate_l2093_209358


namespace sequence_first_term_l2093_209356

theorem sequence_first_term (a : ℕ → ℤ) 
  (h1 : a 3 = 5) 
  (h2 : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n - 1) : 
  a 1 = 2 := 
sorry

end sequence_first_term_l2093_209356


namespace distinct_m_value_l2093_209316

theorem distinct_m_value (a b : ℝ) (m : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
    (h_b_eq_2a : b = 2 * a) (h_m_eq_neg2a_b : m = -2 * a / b) : 
    ∃! (m : ℝ), m = -1 :=
by sorry

end distinct_m_value_l2093_209316


namespace tan_15_simplification_l2093_209306

theorem tan_15_simplification :
  (1 + Real.tan (Real.pi / 12)) / (1 - Real.tan (Real.pi / 12)) = Real.sqrt 3 :=
by
  sorry

end tan_15_simplification_l2093_209306


namespace train_average_speed_l2093_209317

theorem train_average_speed :
  let start_time := 9.0 -- Start time in hours (9:00 am)
  let end_time := 13.75 -- End time in hours (1:45 pm)
  let total_distance := 348.0 -- Total distance in km
  let halt_time := 0.75 -- Halt time in hours (45 minutes)
  let scheduled_time := end_time - start_time -- Total scheduled time in hours
  let actual_travel_time := scheduled_time - halt_time -- Actual travel time in hours
  let average_speed := total_distance / actual_travel_time -- Average speed formula
  average_speed = 87.0 := sorry

end train_average_speed_l2093_209317


namespace jessica_seashells_l2093_209334

theorem jessica_seashells (joan jessica total : ℕ) (h1 : joan = 6) (h2 : total = 14) (h3 : total = joan + jessica) : jessica = 8 :=
by
  -- proof steps would go here
  sorry

end jessica_seashells_l2093_209334


namespace ratio_and_equation_imp_value_of_a_l2093_209340

theorem ratio_and_equation_imp_value_of_a (a b : ℚ) (h1 : b / a = 4) (h2 : b = 20 - 7 * a) :
  a = 20 / 11 :=
by
  sorry

end ratio_and_equation_imp_value_of_a_l2093_209340


namespace average_all_results_l2093_209351

theorem average_all_results (s₁ s₂ : ℤ) (n₁ n₂ : ℤ) (h₁ : n₁ = 60) (h₂ : n₂ = 40) (avg₁ : s₁ / n₁ = 40) (avg₂ : s₂ / n₂ = 60) : 
  ((s₁ + s₂) / (n₁ + n₂) = 48) :=
sorry

end average_all_results_l2093_209351


namespace geometric_sequence_general_term_formula_no_arithmetic_sequence_l2093_209350

-- Assume we have a sequence {a_n} and its sum of the first n terms S_n where S_n = 2a_n - n (for n ∈ ℕ*)
variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}

-- Condition 1: S_n = 2a_n - n
axiom Sn_condition (n : ℕ) (h : n > 0) : S_n n = 2 * a_n n - n

-- 1. Prove that the sequence {a_n + 1} is a geometric sequence with first term and common ratio equal to 2
theorem geometric_sequence (n : ℕ) (h : n > 0) : ∃ r : ℕ, r = 2 ∧ ∀ m : ℕ, a_n (m + 1) + 1 = r * (a_n m + 1) :=
by
  sorry

-- 2. Prove the general term formula an = 2^n - 1
theorem general_term_formula (n : ℕ) (h : n > 0) : a_n n = 2^n - 1 :=
by
  sorry

-- 3. Prove that there do not exist three consecutive terms in {a_n} that form an arithmetic sequence
theorem no_arithmetic_sequence (n k : ℕ) (h : n > 0 ∧ k > 0 ∧ k + 2 < n) : ¬(a_n k + a_n (k + 2) = 2 * a_n (k + 1)) :=
by
  sorry

end geometric_sequence_general_term_formula_no_arithmetic_sequence_l2093_209350


namespace min_visible_sum_of_values_l2093_209392

-- Definitions based on the problem conditions
def is_standard_die (die : ℕ → ℕ) : Prop :=
  ∀ (i j : ℕ), (i + j = 7) → (die j + die i = 7)

def corner_cubes (cubes : ℕ) : ℕ := 8
def edge_cubes (cubes : ℕ) : ℕ := 24
def face_center_cubes (cubes : ℕ) : ℕ := 24

-- The proof statement
theorem min_visible_sum_of_values
  (m : ℕ)
  (condition1 : is_standard_die m)
  (condition2 : corner_cubes 64 = 8)
  (condition3 : edge_cubes 64 = 24)
  (condition4 : face_center_cubes 64 = 24)
  (condition5 : 64 = 8 + 24 + 24 + 8): 
  m = 144 :=
sorry

end min_visible_sum_of_values_l2093_209392


namespace initial_fish_count_l2093_209355

theorem initial_fish_count (F T : ℕ) 
  (h1 : T = 3 * F)
  (h2 : T / 2 = (F - 7) + 32) : F = 50 :=
by
  sorry

end initial_fish_count_l2093_209355


namespace arithmetic_identity_l2093_209322

theorem arithmetic_identity :
  65 * 1515 - 25 * 1515 + 1515 = 62115 :=
by
  sorry

end arithmetic_identity_l2093_209322


namespace bingley_bracelets_final_l2093_209337

-- Definitions
def initial_bingley_bracelets : Nat := 5
def kelly_bracelets_given : Nat := 16 / 4
def bingley_bracelets_after_kelly : Nat := initial_bingley_bracelets + kelly_bracelets_given
def bingley_bracelets_given_to_sister : Nat := bingley_bracelets_after_kelly / 3
def bingley_remaining_bracelets : Nat := bingley_bracelets_after_kelly - bingley_bracelets_given_to_sister

-- Theorem
theorem bingley_bracelets_final : bingley_remaining_bracelets = 6 := by
  sorry

end bingley_bracelets_final_l2093_209337


namespace proof_cost_A_B_schools_proof_renovation_plans_l2093_209338

noncomputable def cost_A_B_schools : Prop :=
  ∃ (x y : ℝ), 2 * x + 3 * y = 78 ∧ 3 * x + y = 54 ∧ x = 12 ∧ y = 18

noncomputable def renovation_plans : Prop :=
  ∃ (a : ℕ), 3 ≤ a ∧ a ≤ 5 ∧ 
    (1200 - 300) * a + (1800 - 500) * (10 - a) ≤ 11800 ∧
    300 * a + 500 * (10 - a) ≥ 4000

theorem proof_cost_A_B_schools : cost_A_B_schools :=
sorry

theorem proof_renovation_plans : renovation_plans :=
sorry

end proof_cost_A_B_schools_proof_renovation_plans_l2093_209338


namespace general_form_of_numbers_whose_square_ends_with_9_l2093_209379

theorem general_form_of_numbers_whose_square_ends_with_9 (x : ℤ) (h : (x^2 % 10 = 9)) :
  ∃ a : ℤ, x = 10 * a + 3 ∨ x = 10 * a + 7 :=
sorry

end general_form_of_numbers_whose_square_ends_with_9_l2093_209379


namespace average_age_of_students_is_14_l2093_209378

noncomputable def average_age_of_students (student_count : ℕ) (teacher_age : ℕ) (combined_avg_age : ℕ) : ℕ :=
  let total_people := student_count + 1
  let total_combined_age := total_people * combined_avg_age
  let total_student_age := total_combined_age - teacher_age
  total_student_age / student_count

theorem average_age_of_students_is_14 :
  average_age_of_students 50 65 15 = 14 :=
by
  sorry

end average_age_of_students_is_14_l2093_209378


namespace part1_part2_l2093_209311

variable (a b : ℝ)

-- Conditions
axiom abs_a_eq_4 : |a| = 4
axiom abs_b_eq_6 : |b| = 6

-- Part 1: If ab > 0, find the value of a - b
theorem part1 (h : a * b > 0) : a - b = 2 ∨ a - b = -2 := 
by
  -- Proof will go here
  sorry

-- Part 2: If |a + b| = -(a + b), find the value of a + b
theorem part2 (h : |a + b| = -(a + b)) : a + b = -10 ∨ a + b = -2 := 
by
  -- Proof will go here
  sorry

end part1_part2_l2093_209311


namespace calculate_expression_l2093_209339

theorem calculate_expression :
  ((12 ^ 12 / 12 ^ 11) ^ 2 * 4 ^ 2) / 2 ^ 4 = 144 :=
by
  sorry

end calculate_expression_l2093_209339


namespace minimal_q_for_fraction_l2093_209352

theorem minimal_q_for_fraction :
  ∃ p q : ℕ, 0 < p ∧ 0 < q ∧ 
  (3/5 : ℚ) < p / q ∧ p / q < (5/8 : ℚ) ∧
  (∀ r : ℕ, 0 < r ∧ (3/5 : ℚ) < p / r ∧ p / r < (5/8 : ℚ) → q ≤ r) ∧
  p + q = 21 :=
by
  sorry

end minimal_q_for_fraction_l2093_209352


namespace f_gt_e_plus_2_l2093_209396

noncomputable def f (x : ℝ) : ℝ := ( (Real.exp x) / x ) - ( (8 * Real.log (x / 2)) / (x^2) ) + x

lemma slope_at_2 : HasDerivAt f (Real.exp 2 / 4) 2 := 
by 
  sorry

theorem f_gt_e_plus_2 (x : ℝ) (hx : 0 < x) : f x > Real.exp 1 + 2 :=
by
  sorry

end f_gt_e_plus_2_l2093_209396


namespace eq1_solution_eq2_no_solution_l2093_209393

-- For Equation (1)
theorem eq1_solution (x : ℝ) (h : (3 / (2 * x - 2)) + (1 / (1 - x)) = 3) : 
  x = 7 / 6 :=
by sorry

-- For Equation (2)
theorem eq2_no_solution (y : ℝ) : ¬((y / (y - 1)) - (2 / (y^2 - 1)) = 1) :=
by sorry

end eq1_solution_eq2_no_solution_l2093_209393


namespace expected_score_particular_player_l2093_209310

-- Define types of dice
inductive DiceType : Type
| A | B | C

-- Define the faces of each dice type
def DiceFaces : DiceType → List ℕ
| DiceType.A => [2, 2, 4, 4, 9, 9]
| DiceType.B => [1, 1, 6, 6, 8, 8]
| DiceType.C => [3, 3, 5, 5, 7, 7]

-- Define a function to calculate the score of a player given their roll and opponents' rolls
def player_score (p_roll : ℕ) (opp_rolls : List ℕ) : ℕ :=
  opp_rolls.foldl (λ acc roll => if roll < p_roll then acc + 1 else acc) 0

-- Define a function to calculate the expected score of a player
noncomputable def expected_score (dice_choice : DiceType) : ℚ :=
  let rolls := DiceFaces dice_choice
  let total_possibilities := (rolls.length : ℚ) ^ 3
  let score_sum := rolls.foldl (λ acc p_roll =>
    acc + rolls.foldl (λ acc1 opp1_roll =>
        acc1 + rolls.foldl (λ acc2 opp2_roll =>
            acc2 + player_score p_roll [opp1_roll, opp2_roll]
          ) 0
      ) 0
    ) 0
  score_sum / total_possibilities

-- The main theorem statement
theorem expected_score_particular_player : (expected_score DiceType.A + expected_score DiceType.B + expected_score DiceType.C) / 3 = 
(8 : ℚ) / 9 := sorry

end expected_score_particular_player_l2093_209310


namespace cakes_sold_to_baked_ratio_l2093_209328

theorem cakes_sold_to_baked_ratio
  (cakes_per_day : ℕ) 
  (days : ℕ)
  (cakes_left : ℕ)
  (total_cakes : ℕ := cakes_per_day * days)
  (cakes_sold : ℕ := total_cakes - cakes_left) :
  cakes_per_day = 20 → 
  days = 9 → 
  cakes_left = 90 → 
  cakes_sold * 2 = total_cakes := 
by 
  intros 
  sorry

end cakes_sold_to_baked_ratio_l2093_209328


namespace sqrt_30_estimate_l2093_209301

theorem sqrt_30_estimate : 5 < Real.sqrt 30 ∧ Real.sqrt 30 < 6 := by
  sorry

end sqrt_30_estimate_l2093_209301


namespace walnut_trees_planted_l2093_209348

theorem walnut_trees_planted (initial_trees : ℕ) (final_trees : ℕ) (num_trees_planted : ℕ) : initial_trees = 107 → final_trees = 211 → num_trees_planted = final_trees - initial_trees → num_trees_planted = 104 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end walnut_trees_planted_l2093_209348


namespace possible_values_of_cubic_sum_l2093_209300

theorem possible_values_of_cubic_sum (x y z : ℂ) (h1 : (Matrix.of ![
    ![x, y, z],
    ![y, z, x],
    ![z, x, y]
  ] ^ 2 = 3 • (1 : Matrix (Fin 3) (Fin 3) ℂ))) (h2 : x * y * z = -1) :
  x^3 + y^3 + z^3 = -3 + 3 * Real.sqrt 3 ∨ x^3 + y^3 + z^3 = -3 - 3 * Real.sqrt 3 := by
  sorry

end possible_values_of_cubic_sum_l2093_209300


namespace divisible_by_24_l2093_209375

theorem divisible_by_24 (n : ℕ) : ∃ k : ℤ, n^4 + 2 * n^3 + 11 * n^2 + 10 * n = 24 * k := sorry

end divisible_by_24_l2093_209375


namespace total_height_increase_in_4_centuries_l2093_209347

def height_increase_per_decade : ℕ := 75
def years_per_century : ℕ := 100
def years_per_decade : ℕ := 10
def centuries : ℕ := 4

theorem total_height_increase_in_4_centuries :
  height_increase_per_decade * (centuries * years_per_century / years_per_decade) = 3000 := by
  sorry

end total_height_increase_in_4_centuries_l2093_209347


namespace students_brought_apples_l2093_209354

theorem students_brought_apples (A B C D : ℕ) (h1 : B = 8) (h2 : C = 10) (h3 : D = 5) (h4 : A - D + B - D = C) : A = 12 :=
by {
  sorry
}

end students_brought_apples_l2093_209354


namespace eva_total_marks_l2093_209388

theorem eva_total_marks
    (math_score_s2 : ℕ) (arts_score_s2 : ℕ) (science_score_s2 : ℕ)
    (math_diff : ℕ) (arts_diff : ℕ) (science_frac_diff : ℚ)
    (math_score_s2_eq : math_score_s2 = 80)
    (arts_score_s2_eq : arts_score_s2 = 90)
    (science_score_s2_eq : science_score_s2 = 90)
    (math_diff_eq : math_diff = 10)
    (arts_diff_eq : arts_diff = 15)
    (science_frac_diff_eq : science_frac_diff = 1/3) : 
  (math_score_s2 + 10 + (math_score_s2 + math_diff) + 
   (arts_score_s2 + 90 - 15) + (arts_score_s2 + arts_diff) + 
   (science_score_s2 + 90 - (1/3) * 90) + (science_score_s2 + science_score_s2 * 1/3)) = 485 := 
by
  sorry

end eva_total_marks_l2093_209388


namespace intersection_A_B_l2093_209344

def A := {x : ℝ | 2 * x - 1 ≤ 0}
def B := {x : ℝ | 1 / x > 1}

theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1 / 2} :=
  sorry

end intersection_A_B_l2093_209344


namespace log_power_relationship_l2093_209373

theorem log_power_relationship (a b c : ℝ) (m n r : ℝ)
  (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) (h4 : 1 < c)
  (hm : m = Real.log c / Real.log a)
  (hn : n = Real.log c / Real.log b)
  (hr : r = a^c) :
  r > m ∧ m > n :=
sorry

end log_power_relationship_l2093_209373


namespace pascal_28_25_eq_2925_l2093_209349

-- Define the Pascal's triangle nth-row function
def pascal (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the theorem to prove that the 25th element in the 28 element row is 2925
theorem pascal_28_25_eq_2925 :
  pascal 27 24 = 2925 :=
by
  sorry

end pascal_28_25_eq_2925_l2093_209349


namespace lorry_empty_weight_l2093_209353

-- Define variables for the weights involved
variable (lw : ℕ)  -- weight of the lorry when empty
variable (bl : ℕ)  -- number of bags of apples
variable (bw : ℕ)  -- weight of each bag of apples
variable (total_weight : ℕ)  -- total loaded weight of the lorry

-- Given conditions
axiom lorry_loaded_weight : bl = 20 ∧ bw = 60 ∧ total_weight = 1700

-- The theorem we want to prove
theorem lorry_empty_weight : (∀ lw bw, total_weight - bl * bw = lw) → lw = 500 :=
by
  intro h
  rw [←h lw bw]
  sorry

end lorry_empty_weight_l2093_209353


namespace ratio_children_to_adults_l2093_209323

variable (f m c : ℕ)

-- Conditions
def average_age_female (f : ℕ) := 35
def average_age_male (m : ℕ) := 30
def average_age_child (c : ℕ) := 10
def overall_average_age (f m c : ℕ) := 25

-- Total age sums based on given conditions
def total_age_sum_female (f : ℕ) := 35 * f
def total_age_sum_male (m : ℕ) := 30 * m
def total_age_sum_child (c : ℕ) := 10 * c

-- Total sum and average conditions
def total_age_sum (f m c : ℕ) := total_age_sum_female f + total_age_sum_male m + total_age_sum_child c
def total_members (f m c : ℕ) := f + m + c

theorem ratio_children_to_adults (f m c : ℕ) (h : (total_age_sum f m c) / (total_members f m c) = 25) :
  (c : ℚ) / (f + m) = 2 / 3 := sorry

end ratio_children_to_adults_l2093_209323


namespace factorization_of_x4_plus_16_l2093_209319

theorem factorization_of_x4_plus_16 :
  (x : ℝ) → x^4 + 16 = (x^2 + 2 * x + 2) * (x^2 - 2 * x + 2) :=
by
  intro x
  -- Placeholder for the proof
  sorry

end factorization_of_x4_plus_16_l2093_209319


namespace paul_tips_l2093_209371

theorem paul_tips (P : ℕ) (h1 : P + 16 = 30) : P = 14 :=
by
  sorry

end paul_tips_l2093_209371


namespace number_of_shares_is_25_l2093_209365

def wife_weekly_savings := 100
def husband_monthly_savings := 225
def duration_months := 4
def cost_per_share := 50

def total_savings : ℕ :=
  (wife_weekly_savings * 4 * duration_months) + (husband_monthly_savings * duration_months)

def amount_invested := total_savings / 2

def number_of_shares := amount_invested / cost_per_share

theorem number_of_shares_is_25 : number_of_shares = 25 := by
  sorry

end number_of_shares_is_25_l2093_209365


namespace total_amount_paid_l2093_209394

def grapes_quantity := 8
def grapes_rate := 80
def mangoes_quantity := 9
def mangoes_rate := 55
def apples_quantity := 6
def apples_rate := 120
def oranges_quantity := 4
def oranges_rate := 75

theorem total_amount_paid :
  grapes_quantity * grapes_rate +
  mangoes_quantity * mangoes_rate +
  apples_quantity * apples_rate +
  oranges_quantity * oranges_rate =
  2155 := by
  sorry

end total_amount_paid_l2093_209394


namespace lino_shells_total_l2093_209364

def picked_up_shells : Float := 324.0
def put_back_shells : Float := 292.0

theorem lino_shells_total : picked_up_shells - put_back_shells = 32.0 :=
by
  sorry

end lino_shells_total_l2093_209364


namespace customers_non_holiday_l2093_209366

theorem customers_non_holiday (h : ∀ n, 2 * n = 350) (H : ∃ h : ℕ, h * 8 = 2800) : (2800 / 8 / 2 = 175) :=
by sorry

end customers_non_holiday_l2093_209366


namespace face_value_of_shares_l2093_209336

theorem face_value_of_shares (investment : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ) (dividend_received : ℝ) (F : ℝ)
  (h1 : investment = 14400)
  (h2 : premium_rate = 0.20)
  (h3 : dividend_rate = 0.06)
  (h4 : dividend_received = 720) :
  (1.20 * F = investment) ∧ (0.06 * F = dividend_received) ∧ (F = 12000) :=
by
  sorry

end face_value_of_shares_l2093_209336


namespace solve_for_x_l2093_209308

def f (x : ℝ) : ℝ := 3 * x - 4

noncomputable def f_inv (x : ℝ) : ℝ := (x + 4) / 3

theorem solve_for_x : ∃ x : ℝ, f x = f_inv x ∧ x = 2 := by
  sorry

end solve_for_x_l2093_209308


namespace average_of_ABC_l2093_209343

theorem average_of_ABC (A B C : ℝ) 
  (h1 : 2002 * C - 1001 * A = 8008) 
  (h2 : 2002 * B + 3003 * A = 7007) 
  (h3 : A = 2) : (A + B + C) / 3 = 2.33 := 
by 
  sorry

end average_of_ABC_l2093_209343


namespace maximum_y_coordinate_l2093_209342

variable (x y b : ℝ)

def hyperbola (x y b : ℝ) : Prop := (x^2) / 4 - (y^2) / b = 1

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

def op_condition (x y b : ℝ) : Prop := (x^2 + y^2) = 4 + b

noncomputable def eccentricity (b : ℝ) : ℝ := (Real.sqrt (4 + b)) / 2

theorem maximum_y_coordinate (hb : b > 0) 
                            (h_ec : 1 < eccentricity b ∧ eccentricity b ≤ 2) 
                            (h_hyp : hyperbola x y b) 
                            (h_first : first_quadrant x y) 
                            (h_op : op_condition x y b) 
                            : y ≤ 3 :=
sorry

end maximum_y_coordinate_l2093_209342


namespace sum_first_15_odd_integers_from_5_l2093_209382

theorem sum_first_15_odd_integers_from_5 :
  let a := 5
  let n := 15
  let d := 2
  let last_term := a + (n - 1) * d
  let S := n * a + (n * (n - 1) * d) / 2
  last_term = 37 ∧ S = 315 := by
  sorry

end sum_first_15_odd_integers_from_5_l2093_209382


namespace max_height_l2093_209372

def h (t : ℝ) : ℝ := -20 * t ^ 2 + 80 * t + 50

theorem max_height : ∃ t : ℝ, ∀ t' : ℝ, h t' ≤ h t ∧ h t = 130 :=
by
  sorry

end max_height_l2093_209372


namespace class_trip_contributions_l2093_209326

theorem class_trip_contributions (x y : ℕ) :
  (x + 5) * (y + 6) = x * y + 792 ∧ (x - 4) * (y + 4) = x * y - 388 → x = 213 ∧ y = 120 := 
by
  sorry

end class_trip_contributions_l2093_209326


namespace line_equation_l2093_209385

-- Define the conditions: point (2,1) on the line and slope is 2
def point_on_line (x y : ℝ) (m b : ℝ) : Prop := y = m * x + b

def slope_of_line (m : ℝ) : Prop := m = 2

-- Prove the equation of the line is 2x - y - 3 = 0
theorem line_equation (b : ℝ) (h1 : point_on_line 2 1 2 b) : 2 * 2 - 1 - 3 = 0 := by
  sorry

end line_equation_l2093_209385


namespace find_cost_per_kg_l2093_209390

-- Define the conditions given in the problem
def side_length : ℕ := 30
def coverage_per_kg : ℕ := 20
def total_cost : ℕ := 10800

-- The cost per kg we need to find
def cost_per_kg := total_cost / ((6 * side_length^2) / coverage_per_kg)

-- We need to prove that cost_per_kg = 40
theorem find_cost_per_kg : cost_per_kg = 40 := by
  sorry

end find_cost_per_kg_l2093_209390


namespace simplify_expression_l2093_209305

theorem simplify_expression (a b : ℚ) : (14 * a^3 * b^2 - 7 * a * b^2) / (7 * a * b^2) = 2 * a^2 - 1 := 
by 
  sorry

end simplify_expression_l2093_209305


namespace max_min_f_triangle_area_l2093_209381

open Real

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (-2 * sin x, -1)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (-cos x, cos (2 * x))
noncomputable def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

theorem max_min_f :
  (∀ x : ℝ, f x ≤ 2) ∧ (∀ x : ℝ, -2 ≤ f x) :=
sorry

theorem triangle_area
  (A B C : ℝ)
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (hC : 0 < C ∧ C < π / 2)
  (h : A + B + C = π)
  (h_f_A : f A = 1)
  (b c : ℝ)
  (h_bc : b * c = 8) :
  (1 / 2) * b * c * sin A = 2 :=
sorry

end max_min_f_triangle_area_l2093_209381


namespace find_function_l2093_209331

theorem find_function (f : ℝ → ℝ) (c : ℝ) :
  (∀ x y : ℝ, (x - y) * f (x + y) - (x + y) * f (x - y) = 4 * x * y * (x ^ 2 - y ^ 2)) →
  (∀ x : ℝ, f x = x ^ 3 + c * x) :=
by
  -- The proof details will be filled here.
  sorry

end find_function_l2093_209331


namespace range_f_period_f_monotonic_increase_intervals_l2093_209377

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sin x) ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 1 

theorem range_f : Set.Icc 0 4 = Set.range f := sorry

theorem period_f : ∀ x, f (x + Real.pi) = f x := sorry

theorem monotonic_increase_intervals (k : ℤ) :
  ∀ x, (-π / 6 + k * π : ℝ) ≤ x ∧ x ≤ (π / 3 + k * π : ℝ) → 
        ∀ y, f y ≤ f x → y ≤ x := sorry

end range_f_period_f_monotonic_increase_intervals_l2093_209377


namespace quadratic_root_ratio_eq_l2093_209314

theorem quadratic_root_ratio_eq (k : ℝ) :
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ (x = 3 * y ∨ y = 3 * x) ∧ x + y = -10 ∧ x * y = k) → k = 18.75 := by
  sorry

end quadratic_root_ratio_eq_l2093_209314


namespace max_sum_of_cubes_l2093_209309

open Real

theorem max_sum_of_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 5 * sqrt 5 :=
by
  sorry

end max_sum_of_cubes_l2093_209309


namespace cost_of_green_lettuce_l2093_209313

-- Definitions based on the conditions given in the problem
def cost_per_pound := 2
def weight_red_lettuce := 6 / cost_per_pound
def total_weight := 7
def weight_green_lettuce := total_weight - weight_red_lettuce

-- Problem statement: Prove that the cost of green lettuce is $8
theorem cost_of_green_lettuce : (weight_green_lettuce * cost_per_pound) = 8 :=
by
  sorry

end cost_of_green_lettuce_l2093_209313


namespace measure_angle_Z_l2093_209333

-- Given conditions
def triangle_condition (X Y Z : ℝ) :=
   X = 78 ∧ Y = 4 * Z - 14

-- Triangle angle sum property
def triangle_angle_sum (X Y Z : ℝ) :=
   X + Y + Z = 180

-- Prove the measure of angle Z
theorem measure_angle_Z (X Y Z : ℝ) (h1 : triangle_condition X Y Z) (h2 : triangle_angle_sum X Y Z) : 
  Z = 23.2 :=
by
  -- Lean will expect proof steps here, ‘sorry’ is used to denote unproven parts.
  sorry

end measure_angle_Z_l2093_209333


namespace similar_triangle_perimeters_l2093_209374

theorem similar_triangle_perimeters 
  (h_ratio : ℕ) (h_ratio_eq : h_ratio = 2/3)
  (sum_perimeters : ℕ) (sum_perimeters_eq : sum_perimeters = 50)
  (a b : ℕ)
  (perimeter_ratio : ℕ) (perimeter_ratio_eq : perimeter_ratio = 2/3)
  (hyp1 : a + b = sum_perimeters)
  (hyp2 : a * 3 = b * 2) :
  (a = 20 ∧ b = 30) :=
by
  sorry

end similar_triangle_perimeters_l2093_209374


namespace part1_min_value_part2_max_value_k_lt_part2_max_value_k_geq_l2093_209318

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem part1_min_value : ∀ (x : ℝ), x > 0 → f x ≥ -1 / Real.exp 1 := 
by sorry

noncomputable def g (x k : ℝ) : ℝ := f x - k * (x - 1)

theorem part2_max_value_k_lt : ∀ (k : ℝ), k < Real.exp 1 / (Real.exp 1 - 1) → 
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → g x k ≤ Real.exp 1 - k * Real.exp 1 + k :=
by sorry

theorem part2_max_value_k_geq : ∀ (k : ℝ), k ≥ Real.exp 1 / (Real.exp 1 - 1) → 
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → g x k ≤ 0 :=
by sorry

end part1_min_value_part2_max_value_k_lt_part2_max_value_k_geq_l2093_209318


namespace range_of_a_l2093_209383

-- Define the condition function
def inequality (a x : ℝ) : Prop := a^2 * x - 2 * (a - x - 4) < 0

-- Prove that given the inequality always holds for any real x, the range of a is (-2, 2]
theorem range_of_a {a : ℝ} (h : ∀ x : ℝ, inequality a x) : -2 < a ∧ a ≤ 2 := by
  sorry

end range_of_a_l2093_209383


namespace final_result_is_four_l2093_209325

theorem final_result_is_four (x : ℕ) (h1 : x = 208) (y : ℕ) (h2 : y = x / 2) (z : ℕ) (h3 : z = y - 100) : z = 4 :=
by {
  sorry
}

end final_result_is_four_l2093_209325


namespace arithmetic_sequence_eighth_term_l2093_209384

theorem arithmetic_sequence_eighth_term (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 2) : a 8 = 15 := by
  sorry

end arithmetic_sequence_eighth_term_l2093_209384


namespace inequality_proof_l2093_209376

theorem inequality_proof (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) :
  (a^2 - b^2) / (a^2 + b^2) > (a - b) / (a + b) :=
by 
  sorry

end inequality_proof_l2093_209376


namespace sequence_a_n_definition_l2093_209303

theorem sequence_a_n_definition (a : ℕ+ → ℝ) 
  (h₀ : ∀ n : ℕ+, a (n + 1) = 2016 * a n / (2014 * a n + 2016))
  (h₁ : a 1 = 1) : 
  a 2017 = 1008 / (1007 * 2017 + 1) :=
sorry

end sequence_a_n_definition_l2093_209303


namespace solve_equation_l2093_209329

theorem solve_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1) :
  (x / (x - 1) - 2 / x = 1) ↔ x = 2 :=
sorry

end solve_equation_l2093_209329


namespace angle_C_of_triangle_l2093_209324

theorem angle_C_of_triangle (A B C : ℝ) (hA : A = 90) (hB : B = 50) (h_sum : A + B + C = 180) : C = 40 := 
by
  sorry

end angle_C_of_triangle_l2093_209324


namespace complement_A_in_U_l2093_209357

open Set

-- Definitions for sets
def U : Set ℕ := {x | 1 < x ∧ x < 5}
def A : Set ℕ := {2, 3}

-- The proof goal: prove that the complement of A in U is {4}
theorem complement_A_in_U : (U \ A) = {4} := by
  sorry

end complement_A_in_U_l2093_209357


namespace mark_final_buttons_l2093_209360

def mark_initial_buttons : ℕ := 14
def shane_factor : ℚ := 3.5
def lent_to_anna : ℕ := 7
def lost_fraction : ℚ := 0.5
def sam_fraction : ℚ := 2 / 3

theorem mark_final_buttons : 
  let shane_buttons := mark_initial_buttons * shane_factor
  let before_anna := mark_initial_buttons + shane_buttons
  let after_lending_anna := before_anna - lent_to_anna
  let anna_returned := lent_to_anna * (1 - lost_fraction)
  let after_anna_return := after_lending_anna + anna_returned
  let after_sam := after_anna_return - (after_anna_return * sam_fraction)
  round after_sam = 20 := 
by
  sorry

end mark_final_buttons_l2093_209360


namespace next_month_eggs_l2093_209341

-- Given conditions definitions
def eggs_left_last_month : ℕ := 27
def eggs_after_buying : ℕ := 58
def eggs_eaten_this_month : ℕ := 48

-- Calculate number of eggs mother buys each month
def eggs_bought_each_month : ℕ := eggs_after_buying - eggs_left_last_month

-- Remaining eggs before next purchase
def eggs_left_before_next_purchase : ℕ := eggs_after_buying - eggs_eaten_this_month

-- Final amount of eggs after mother buys next month's supply
def total_eggs_next_month : ℕ := eggs_left_before_next_purchase + eggs_bought_each_month

-- Prove the total number of eggs next month equals 41
theorem next_month_eggs : total_eggs_next_month = 41 := by
  sorry

end next_month_eggs_l2093_209341


namespace Ted_has_15_bags_l2093_209307

-- Define the parameters
def total_candy_bars : ℕ := 75
def candy_per_bag : ℝ := 5.0

-- Define the assertion to be proved
theorem Ted_has_15_bags : total_candy_bars / candy_per_bag = 15 := 
by
  sorry

end Ted_has_15_bags_l2093_209307


namespace sum_abc_l2093_209320

noncomputable def polynomial : Polynomial ℝ :=
  Polynomial.C (-6) + Polynomial.X * (Polynomial.C 11 + Polynomial.X * (Polynomial.C (-6) + Polynomial.X))

def t (k : ℕ) : ℝ :=
  match k with
  | 0 => 3
  | 1 => 6
  | 2 => 14
  | _ => 0 -- placeholder, as only t_0, t_1, t_2 are given explicitly

def a := 6
def b := -11
def c := 18

def t_rec (k : ℕ) : ℝ :=
  match k with
  | 0 => 3
  | 1 => 6
  | 2 => 14
  | n + 3 => a * t (n + 2) + b * t (n + 1) + c * t n

theorem sum_abc : a + b + c = 13 := by
  sorry

end sum_abc_l2093_209320


namespace determinant_transformation_l2093_209380

theorem determinant_transformation 
  (a b c d : ℝ)
  (h : a * d - b * c = 6) :
  (a * (5 * c + 2 * d) - c * (5 * a + 2 * b)) = 12 := by
  sorry

end determinant_transformation_l2093_209380


namespace range_of_a_l2093_209368

def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B (a : ℝ) : Set ℝ := {x | 2^(1 - x) + a ≤ 0 ∧ x^2 - 2*(a + 7)*x + 5 ≤ 0}

/-- If A ⊆ B, then the range of values for 'a' satisfies -4 ≤ a ≤ -1 -/
theorem range_of_a (a : ℝ) (h : A ⊆ B a) : -4 ≤ a ∧ a ≤ -1 :=
by
  sorry

end range_of_a_l2093_209368


namespace rachelle_meat_needed_l2093_209362

-- Define the ratio of meat per hamburger
def meat_per_hamburger (pounds : ℕ) (hamburgers : ℕ) : ℚ :=
  pounds / hamburgers

-- Define the total meat needed for a given number of hamburgers
def total_meat (meat_per_hamburger : ℚ) (hamburgers : ℕ) : ℚ :=
  meat_per_hamburger * hamburgers

-- Prove that Rachelle needs 15 pounds of meat to make 36 hamburgers
theorem rachelle_meat_needed : total_meat (meat_per_hamburger 5 12) 36 = 15 := by
  sorry

end rachelle_meat_needed_l2093_209362


namespace fraction_addition_l2093_209315

theorem fraction_addition (a b c d : ℚ) (ha : a = 2/5) (hb : b = 3/8) (hc : c = 31/40) :
  a + b = c :=
by
  rw [ha, hb, hc]
  -- The proof part is skipped here as per instructions
  sorry

end fraction_addition_l2093_209315


namespace one_third_of_seven_times_nine_l2093_209386

theorem one_third_of_seven_times_nine : (1 / 3) * (7 * 9) = 21 := by
  sorry

end one_third_of_seven_times_nine_l2093_209386


namespace expression_divisibility_l2093_209330

theorem expression_divisibility (x y : ℤ) (k_1 k_2 : ℤ) (h1 : 2 * x + 3 * y = 17 * k_1) :
    ∃ k_2 : ℤ, 9 * x + 5 * y = 17 * k_2 :=
by
  sorry

end expression_divisibility_l2093_209330


namespace cell_chain_length_l2093_209363

theorem cell_chain_length (d n : ℕ) (h₁ : d = 5 * 10^2) (h₂ : n = 2 * 10^3) : d * n = 10^6 :=
by
  sorry

end cell_chain_length_l2093_209363


namespace sum_reciprocals_of_roots_l2093_209335

theorem sum_reciprocals_of_roots (p q x₁ x₂ : ℝ) (h₀ : x₁ + x₂ = -p) (h₁ : x₁ * x₂ = q) :
  (1 / x₁ + 1 / x₂) = -p / q :=
by 
  sorry

end sum_reciprocals_of_roots_l2093_209335
