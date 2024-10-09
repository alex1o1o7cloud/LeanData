import Mathlib

namespace intersection_S_T_eq_l1633_163378

def S : Set ℝ := { x | (x - 2) * (x - 3) ≥ 0 }
def T : Set ℝ := { x | x > 0 }

theorem intersection_S_T_eq : (S ∩ T) = { x | (0 < x ∧ x ≤ 2) ∨ (x ≥ 3) } :=
by
  sorry

end intersection_S_T_eq_l1633_163378


namespace candy_mixture_price_l1633_163337

theorem candy_mixture_price
  (a : ℝ)
  (h1 : 0 < a) -- Assuming positive amount of money spent, to avoid division by zero
  (p1 p2 : ℝ)
  (h2 : p1 = 2)
  (h3 : p2 = 3)
  (h4 : p2 * (a / p2) = p1 * (a / p1)) -- Condition that the total cost for each type is equal.
  : ( (p1 * (a / p1) + p2 * (a / p2)) / (a / p1 + a / p2) = 2.4 ) :=
  sorry

end candy_mixture_price_l1633_163337


namespace one_third_recipe_ingredients_l1633_163395

noncomputable def cups_of_flour (f : ℚ) := (f : ℚ)
noncomputable def cups_of_sugar (s : ℚ) := (s : ℚ)
def original_recipe_flour := (27 / 4 : ℚ)  -- mixed number 6 3/4 converted to improper fraction
def original_recipe_sugar := (5 / 2 : ℚ)  -- mixed number 2 1/2 converted to improper fraction

theorem one_third_recipe_ingredients :
  cups_of_flour (original_recipe_flour / 3) = (9 / 4) ∧
  cups_of_sugar (original_recipe_sugar / 3) = (5 / 6) :=
by
  sorry

end one_third_recipe_ingredients_l1633_163395


namespace eq_no_sol_l1633_163302

open Nat -- Use natural number namespace

theorem eq_no_sol (k : ℤ) (x y z : ℕ) (hk1 : k ≠ 1) (hk3 : k ≠ 3) :
  ¬ (x^2 + y^2 + z^2 = k * x * y * z) := 
sorry

end eq_no_sol_l1633_163302


namespace original_number_people_l1633_163371

theorem original_number_people (n : ℕ) (h1 : n / 3 * 2 / 2 = 18) : n = 54 :=
sorry

end original_number_people_l1633_163371


namespace solve_z_l1633_163380

open Complex

theorem solve_z (z : ℂ) (h : z^2 = 3 - 4 * I) : z = 1 - 2 * I ∨ z = -1 + 2 * I :=
by
  sorry

end solve_z_l1633_163380


namespace Phoenix_roots_prod_l1633_163392

def Phoenix_eqn (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ a + b + c = 0

theorem Phoenix_roots_prod {m n : ℝ} (hPhoenix : Phoenix_eqn 1 m n)
  (hEqualRoots : (m^2 - 4 * n) = 0) : m * n = -2 :=
by sorry

end Phoenix_roots_prod_l1633_163392


namespace greatest_integer_c_not_in_range_l1633_163325

theorem greatest_integer_c_not_in_range :
  ∃ c : ℤ, (¬ ∃ x : ℝ, x^2 + (c:ℝ)*x + 18 = -6) ∧ (∀ c' : ℤ, c' > c → (∃ x : ℝ, x^2 + (c':ℝ)*x + 18 = -6)) :=
sorry

end greatest_integer_c_not_in_range_l1633_163325


namespace product_bc_l1633_163304

theorem product_bc (b c : ℤ)
    (h1 : ∀ s : ℤ, s^2 = 2 * s + 1 → s^6 - b * s - c = 0) :
    b * c = 2030 :=
sorry

end product_bc_l1633_163304


namespace total_students_correct_l1633_163390

def students_in_school : ℕ :=
  let students_per_class := 23
  let classes_per_grade := 12
  let grades_per_school := 3
  students_per_class * classes_per_grade * grades_per_school

theorem total_students_correct :
  students_in_school = 828 :=
by
  sorry

end total_students_correct_l1633_163390


namespace candidate_lost_by_1650_votes_l1633_163382

theorem candidate_lost_by_1650_votes (total_votes : ℕ) (pct_candidate : ℝ) (pct_rival : ℝ) : 
  total_votes = 5500 → 
  pct_candidate = 0.35 → 
  pct_rival = 0.65 → 
  ((pct_rival * total_votes) - (pct_candidate * total_votes)) = 1650 := 
by
  intros h1 h2 h3
  sorry

end candidate_lost_by_1650_votes_l1633_163382


namespace find_x_y_l1633_163355

theorem find_x_y (A B C : ℝ) (x y : ℝ) (hA : A = 120) (hB : B = 100) (hC : C = 150)
  (hx : A = B + (x / 100) * B) (hy : A = C - (y / 100) * C) : x = 20 ∧ y = 20 :=
by
  sorry

end find_x_y_l1633_163355


namespace problem1_l1633_163334

theorem problem1 (a b : ℤ) (h1 : abs a = 5) (h2 : abs b = 3) (h3 : abs (a - b) = b - a) : a - b = -8 ∨ a - b = -2 := by 
  sorry

end problem1_l1633_163334


namespace find_angle_4_l1633_163342

def angle_sum_180 (α β : ℝ) : Prop := α + β = 180
def angle_equality (γ δ : ℝ) : Prop := γ = δ
def triangle_angle_values (A B : ℝ) : Prop := A = 80 ∧ B = 50

theorem find_angle_4
  (A B : ℝ) (angle1 angle2 angle3 angle4 : ℝ)
  (h1 : angle_sum_180 angle1 angle2)
  (h2 : angle_equality angle3 angle4)
  (h3 : triangle_angle_values A B)
  (h4 : angle_sum_180 (angle1 + A + B) 180)
  (h5 : angle_sum_180 (angle2 + angle3 + angle4) 180) :
  angle4 = 25 :=
by sorry

end find_angle_4_l1633_163342


namespace correct_statements_l1633_163313

theorem correct_statements :
  (20 / 100 * 40 = 8) ∧
  (2^3 = 8) ∧
  (7 - 3 * 2 ≠ 8) ∧
  (3^2 - 1^2 = 8) ∧
  (2 * (6 - 4)^2 = 8) :=
by
  sorry

end correct_statements_l1633_163313


namespace necessary_condition_not_sufficient_condition_l1633_163352

variable (x : ℝ)

def quadratic_condition : Prop := x^2 - 3 * x + 2 > 0
def interval_condition : Prop := x < 1 ∨ x > 4

theorem necessary_condition : interval_condition x → quadratic_condition x := by sorry

theorem not_sufficient_condition : ¬ (quadratic_condition x → interval_condition x) := by sorry

end necessary_condition_not_sufficient_condition_l1633_163352


namespace cards_left_l1633_163318
noncomputable section

def initial_cards : ℕ := 676
def bought_cards : ℕ := 224

theorem cards_left : initial_cards - bought_cards = 452 := 
by
  sorry

end cards_left_l1633_163318


namespace adult_ticket_price_l1633_163335

/-- 
The community center sells 85 tickets and collects $275 in total.
35 of those tickets are adult tickets. Each child's ticket costs $2.
We want to find the price of an adult ticket.
-/
theorem adult_ticket_price 
  (total_tickets : ℕ) 
  (total_revenue : ℚ) 
  (adult_tickets_sold : ℕ) 
  (child_ticket_price : ℚ)
  (h1 : total_tickets = 85)
  (h2 : total_revenue = 275) 
  (h3 : adult_tickets_sold = 35) 
  (h4 : child_ticket_price = 2) 
  : ∃ A : ℚ, (35 * A + 50 * 2 = 275) ∧ (A = 5) :=
by
  sorry

end adult_ticket_price_l1633_163335


namespace last_three_digits_of_power_l1633_163324

theorem last_three_digits_of_power (h : 7^500 ≡ 1 [MOD 1250]) : 7^10000 ≡ 1 [MOD 1250] :=
by
  sorry

end last_three_digits_of_power_l1633_163324


namespace algebraic_simplification_l1633_163319

theorem algebraic_simplification (m x : ℝ) (h₀ : 0 < m) (h₁ : m < 10) (h₂ : m ≤ x) (h₃ : x ≤ 10) : 
  |x - m| + |x - 10| + |x - m - 10| = 20 - x :=
by
  sorry

end algebraic_simplification_l1633_163319


namespace total_days_on_island_correct_l1633_163357

-- Define the first, second, and third expeditions
def firstExpedition : ℕ := 3

def secondExpedition (a : ℕ) : ℕ := a + 2

def thirdExpedition (b : ℕ) : ℕ := 2 * b

-- Define the total duration in weeks
def totalWeeks : ℕ := firstExpedition + secondExpedition firstExpedition + thirdExpedition (secondExpedition firstExpedition)

-- Define the total days spent on the island
def totalDays (weeks : ℕ) : ℕ := weeks * 7

-- Prove that the total number of days spent is 126
theorem total_days_on_island_correct : totalDays totalWeeks = 126 := 
  by
    sorry

end total_days_on_island_correct_l1633_163357


namespace students_not_enrolled_in_bio_l1633_163365

theorem students_not_enrolled_in_bio (total_students : ℕ) (p : ℕ) (p_half : p = (total_students / 2)) (total_students_eq : total_students = 880) : 
  total_students - p = 440 :=
by sorry

end students_not_enrolled_in_bio_l1633_163365


namespace good_tipper_bill_amount_l1633_163362

theorem good_tipper_bill_amount {B : ℝ} 
    (h₁ : 0.05 * B + 1/20 ≥ 0.20 * B) 
    (h₂ : 0.15 * B = 3.90) : 
    B = 26.00 := 
by 
  sorry

end good_tipper_bill_amount_l1633_163362


namespace range_of_a_min_value_a_plus_4_over_a_sq_l1633_163358

noncomputable def f (x : ℝ) : ℝ :=
  |x - 10| + |x - 20|

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x < 10 * a + 10) ↔ 0 < a :=
sorry

theorem min_value_a_plus_4_over_a_sq (a : ℝ) (h : 0 < a) :
  ∃ y : ℝ, a + 4 / a ^ 2 = y ∧ y = 3 :=
sorry

end range_of_a_min_value_a_plus_4_over_a_sq_l1633_163358


namespace sum_of_reciprocals_l1633_163341

theorem sum_of_reciprocals {a b : ℕ} (h_sum: a + b = 55) (h_hcf: Nat.gcd a b = 5) (h_lcm: Nat.lcm a b = 120) :
  1 / (a : ℚ) + 1 / (b : ℚ) = 11 / 120 :=
by
  sorry

end sum_of_reciprocals_l1633_163341


namespace all_round_trips_miss_capital_same_cost_l1633_163321

open Set

variable {City : Type} [Inhabited City]
variable {f : City → City → ℝ}
variable (capital : City)
variable (round_trip_cost : List City → ℝ)

-- The conditions
axiom flight_cost_symmetric (A B : City) : f A B = f B A
axiom equal_round_trip_cost (R1 R2 : List City) :
  (∀ (city : City), city ∈ R1 ↔ city ∈ R2) → 
  round_trip_cost R1 = round_trip_cost R2

noncomputable def constant_trip_cost := 
  ∀ (cities1 cities2 : List City),
     (∀ (city : City), city ∈ cities1 ↔ city ∈ cities2) →
     ¬(capital ∈ cities1 ∨ capital ∈ cities2) →
     round_trip_cost cities1 = round_trip_cost cities2

-- Goal to prove
theorem all_round_trips_miss_capital_same_cost : constant_trip_cost capital round_trip_cost := 
  sorry

end all_round_trips_miss_capital_same_cost_l1633_163321


namespace walnuts_count_l1633_163387

def nuts_problem (p a c w : ℕ) : Prop :=
  p + a + c + w = 150 ∧
  a = p / 2 ∧
  c = 4 * a ∧
  w = 3 * c

theorem walnuts_count (p a c w : ℕ) (h : nuts_problem p a c w) : w = 96 :=
by sorry

end walnuts_count_l1633_163387


namespace sin_phi_value_l1633_163323

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x + 4 * Real.cos x
noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin x - 4 * Real.cos x

theorem sin_phi_value (φ : ℝ) (h_shift : ∀ x, g x = f (x - φ)) : Real.sin φ = 24 / 25 :=
by
  sorry

end sin_phi_value_l1633_163323


namespace sequence_exists_and_unique_l1633_163326

theorem sequence_exists_and_unique (a : ℕ → ℕ) :
  a 0 = 11 ∧ a 7 = 12 ∧
  (∀ n : ℕ, n < 6 → a n + a (n + 1) + a (n + 2) = 50) →
  (a 0 = 11 ∧ a 1 = 12 ∧ a 2 = 27 ∧ a 3 = 11 ∧ a 4 = 12 ∧
   a 5 = 27 ∧ a 6 = 11 ∧ a 7 = 12) :=
by
  sorry

end sequence_exists_and_unique_l1633_163326


namespace scientific_notation_of_5_35_million_l1633_163315

theorem scientific_notation_of_5_35_million : 
  (5.35 : ℝ) * 10^6 = 5.35 * 10^6 := 
by
  sorry

end scientific_notation_of_5_35_million_l1633_163315


namespace f_ln2_add_f_ln_half_l1633_163339

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x ^ 2) - 3 * x) + 1

theorem f_ln2_add_f_ln_half :
  f (Real.log 2) + f (Real.log (1 / 2)) = 2 :=
by
  sorry

end f_ln2_add_f_ln_half_l1633_163339


namespace average_age_of_women_l1633_163345

noncomputable def avg_age_two_women (M : ℕ) (new_avg : ℕ) (W : ℕ) :=
  let loss := 20 + 10;
  let gain := 2 * 8;
  W = loss + gain

theorem average_age_of_women (M : ℕ) (new_avg : ℕ) (W : ℕ) (avg_age : ℕ) :
  avg_age_two_women M new_avg W →
  avg_age = 23 :=
sorry

#check average_age_of_women

end average_age_of_women_l1633_163345


namespace rate_of_mixed_oil_l1633_163317

theorem rate_of_mixed_oil (V1 V2 : ℝ) (P1 P2 : ℝ) : 
  (V1 = 10) → 
  (P1 = 50) → 
  (V2 = 5) → 
  (P2 = 67) → 
  ((V1 * P1 + V2 * P2) / (V1 + V2) = 55.67) :=
by
  intros V1_eq P1_eq V2_eq P2_eq
  rw [V1_eq, P1_eq, V2_eq, P2_eq]
  norm_num
  sorry

end rate_of_mixed_oil_l1633_163317


namespace avg_weight_of_class_l1633_163316

def A_students : Nat := 36
def B_students : Nat := 44
def C_students : Nat := 50
def D_students : Nat := 30

def A_avg_weight : ℝ := 40
def B_avg_weight : ℝ := 35
def C_avg_weight : ℝ := 42
def D_avg_weight : ℝ := 38

def A_additional_students : Nat := 5
def A_additional_weight : ℝ := 10

def B_reduced_students : Nat := 7
def B_reduced_weight : ℝ := 8

noncomputable def total_weight_class : ℝ :=
  (A_students * A_avg_weight + A_additional_students * A_additional_weight) +
  (B_students * B_avg_weight - B_reduced_students * B_reduced_weight) +
  (C_students * C_avg_weight) +
  (D_students * D_avg_weight)

noncomputable def total_students_class : Nat :=
  A_students + B_students + C_students + D_students

noncomputable def avg_weight_class : ℝ :=
  total_weight_class / total_students_class

theorem avg_weight_of_class :
  avg_weight_class = 38.84 := by
    sorry

end avg_weight_of_class_l1633_163316


namespace length_of_AC_l1633_163368

-- Definitions from the problem
variable (AB BC CD DA : ℝ)
variable (angle_ADC : ℝ)
variable (AC : ℝ)

-- Conditions from the problem
def conditions : Prop :=
  AB = 10 ∧ BC = 10 ∧ CD = 17 ∧ DA = 17 ∧ angle_ADC = 120

-- The mathematically equivalent proof statement
theorem length_of_AC (h : conditions AB BC CD DA angle_ADC) : AC = Real.sqrt 867 := sorry

end length_of_AC_l1633_163368


namespace mary_income_is_128_percent_of_juan_income_l1633_163332

def juan_income : ℝ := sorry
def tim_income : ℝ := 0.80 * juan_income
def mary_income : ℝ := 1.60 * tim_income

theorem mary_income_is_128_percent_of_juan_income
  (J : ℝ) : mary_income = 1.28 * J :=
by
  sorry

end mary_income_is_128_percent_of_juan_income_l1633_163332


namespace find_price_max_profit_l1633_163364

/-
Part 1: Prove the price per unit of type A and B
-/

def price_per_unit (x y : ℕ) : Prop :=
  (2 * x + 3 * y = 690) ∧ (x + 4 * y = 720)

theorem find_price :
  ∃ x y : ℕ, price_per_unit x y ∧ x = 120 ∧ y = 150 :=
by
  sorry

/-
Part 2: Prove the maximum profit with constraints
-/

def conditions (m : ℕ) : Prop :=
  m ≤ 3 * (40 - m) ∧ 120 * m + 150 * (40 - m) ≤ 5400

def profit (m : ℕ) : ℕ :=
  (160 - 120) * m + (200 - 150) * (40 - m)

theorem max_profit :
  ∃ m : ℕ, 20 ≤ m ∧ m ≤ 30 ∧ conditions m ∧ profit m = profit 20 :=
by
  sorry

end find_price_max_profit_l1633_163364


namespace no_rational_numbers_satisfy_l1633_163396

theorem no_rational_numbers_satisfy :
  ¬ ∃ (x y z : ℚ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
    (1 / (x - y)^2 + 1 / (y - z)^2 + 1 / (z - x)^2 = 2014) :=
by
  sorry

end no_rational_numbers_satisfy_l1633_163396


namespace plane_speed_in_still_air_l1633_163385

theorem plane_speed_in_still_air (p w : ℝ) (h1 : (p + w) * 3 = 900) (h2 : (p - w) * 4 = 900) : p = 262.5 :=
by
  sorry

end plane_speed_in_still_air_l1633_163385


namespace even_n_divisible_into_equal_triangles_l1633_163338

theorem even_n_divisible_into_equal_triangles (n : ℕ) (hn : 3 < n) :
  (∃ (triangles : ℕ), triangles = n) ↔ (∃ (k : ℕ), n = 2 * k) := 
sorry

end even_n_divisible_into_equal_triangles_l1633_163338


namespace find_x_l1633_163308

theorem find_x (x : ℝ) (h : (x + 8 + 5 * x + 4 + 2 * x + 7) / 3 = 3 * x - 10) : x = 49 :=
sorry

end find_x_l1633_163308


namespace symmetric_points_existence_l1633_163343

-- Define the ellipse equation
def is_ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

-- Define the line equation parameterized by m
def line_eq (x y m : ℝ) : Prop :=
  y = 4 * x + m

-- Define the range for m such that symmetric points exist
def m_in_range (m : ℝ) : Prop :=
  - (2 * Real.sqrt 13) / 13 < m ∧ m < (2 * Real.sqrt 13) / 13

-- Prove the existence of symmetric points criteria for m
theorem symmetric_points_existence (m : ℝ) :
  (∀ (x y : ℝ), is_ellipse x y → line_eq x y m → 
    (∃ (x1 y1 x2 y2 : ℝ), is_ellipse x1 y1 ∧ is_ellipse x2 y2 ∧ line_eq x1 y1 m ∧ line_eq x2 y2 m ∧ 
      (x1 = x2) ∧ (y1 = -y2))) ↔ m_in_range m :=
sorry

end symmetric_points_existence_l1633_163343


namespace triangle_area_base_6_height_8_l1633_163336

noncomputable def triangle_area (base height : ℕ) : ℕ :=
  (base * height) / 2

theorem triangle_area_base_6_height_8 : triangle_area 6 8 = 24 := by
  sorry

end triangle_area_base_6_height_8_l1633_163336


namespace molecular_weight_of_barium_iodide_l1633_163381

-- Define the atomic weights
def atomic_weight_of_ba : ℝ := 137.33
def atomic_weight_of_i : ℝ := 126.90

-- Define the molecular weight calculation for Barium iodide
def molecular_weight_of_bai2 : ℝ := atomic_weight_of_ba + 2 * atomic_weight_of_i

-- The main theorem to prove
theorem molecular_weight_of_barium_iodide : molecular_weight_of_bai2 = 391.13 := by
  -- we are given that atomic_weight_of_ba = 137.33 and atomic_weight_of_i = 126.90
  -- hence, molecular_weight_of_bai2 = 137.33 + 2 * 126.90
  -- simplifying this, we get
  -- molecular_weight_of_bai2 = 137.33 + 253.80 = 391.13
  sorry

end molecular_weight_of_barium_iodide_l1633_163381


namespace relation_between_x_and_y_l1633_163367

-- Definitions based on the conditions
variables (r x y : ℝ)

-- Power of a Point Theorem and provided conditions
variables (AE_eq_3EC : AE = 3 * EC)
variables (x_def : x = AE)
variables (y_def : y = r)

-- Main statement to be proved
theorem relation_between_x_and_y (r x y : ℝ) (AE_eq_3EC : AE = 3 * EC) (x_def : x = AE) (y_def : y = r) :
  y^2 = x^3 / (2 * r - x) :=
sorry

end relation_between_x_and_y_l1633_163367


namespace diagonal_pairs_forming_60_degrees_l1633_163379

theorem diagonal_pairs_forming_60_degrees :
  let total_diagonals := 12
  let total_pairs := total_diagonals.choose 2
  let non_forming_pairs_per_set := 6
  let sets_of_parallel_planes := 3
  total_pairs - non_forming_pairs_per_set * sets_of_parallel_planes = 48 :=
by 
  let total_diagonals := 12
  let total_pairs := total_diagonals.choose 2
  let non_forming_pairs_per_set := 6
  let sets_of_parallel_planes := 3
  have calculation : total_pairs - non_forming_pairs_per_set * sets_of_parallel_planes = 48 := sorry
  exact calculation

end diagonal_pairs_forming_60_degrees_l1633_163379


namespace fraction_c_d_l1633_163300

theorem fraction_c_d (x y c d : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) 
  (h1 : 8 * x - 6 * y = c) (h2 : 10 * y - 15 * x = d) :
  c / d = -8 / 15 :=
sorry

end fraction_c_d_l1633_163300


namespace cost_of_orchestra_seat_l1633_163393

-- Define the variables according to the conditions in the problem
def orchestra_ticket_count (y : ℕ) : Prop := (2 * y + 115 = 355)
def total_ticket_cost (x y : ℕ) : Prop := (120 * x + 235 * 8 = 3320)
def balcony_ticket_relation (y : ℕ) : Prop := (y + 115 = 355 - y)

-- Main theorem statement: Prove that the cost of a seat in the orchestra is 12 dollars
theorem cost_of_orchestra_seat : ∃ x y : ℕ, orchestra_ticket_count y ∧ total_ticket_cost x y ∧ (x = 12) :=
by sorry

end cost_of_orchestra_seat_l1633_163393


namespace find_integer_triplets_l1633_163311

theorem find_integer_triplets (x y z : ℤ) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = 2003 ↔ (x = 668 ∧ y = 668 ∧ z = 667) ∨ (x = 668 ∧ y = 667 ∧ z = 668) ∨ (x = 667 ∧ y = 668 ∧ z = 668) :=
by
  sorry

end find_integer_triplets_l1633_163311


namespace circle_eq_l1633_163340

theorem circle_eq (x y : ℝ) (h k r : ℝ) (hc : h = 3) (kc : k = 1) (rc : r = 5) :
  (x - h)^2 + (y - k)^2 = r^2 ↔ (x - 3)^2 + (y - 1)^2 = 25 :=
by
  sorry

end circle_eq_l1633_163340


namespace arithmetic_mean_pq_is_10_l1633_163399

variables {p q r : ℝ}

theorem arithmetic_mean_pq_is_10 
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 27)
  (h3 : r - p = 34) 
  : (p + q) / 2 = 10 :=
by 
  exact h1

end arithmetic_mean_pq_is_10_l1633_163399


namespace jed_gives_2_cards_every_two_weeks_l1633_163347

theorem jed_gives_2_cards_every_two_weeks
  (starting_cards : ℕ)
  (cards_per_week : ℕ)
  (cards_after_4_weeks : ℕ)
  (number_of_two_week_intervals : ℕ)
  (cards_given_away_each_two_weeks : ℕ):
  starting_cards = 20 →
  cards_per_week = 6 →
  cards_after_4_weeks = 40 →
  number_of_two_week_intervals = 2 →
  (starting_cards + 4 * cards_per_week - number_of_two_week_intervals * cards_given_away_each_two_weeks = cards_after_4_weeks) →
  cards_given_away_each_two_weeks = 2 := 
by
  intros h_start h_week h_4weeks h_intervals h_eq
  sorry

end jed_gives_2_cards_every_two_weeks_l1633_163347


namespace isoland_license_plates_proof_l1633_163330

def isoland_license_plates : ℕ :=
  let letters := ['A', 'B', 'D', 'E', 'I', 'L', 'N', 'O', 'R', 'U']
  let valid_letters := letters.erase 'B'
  let first_letter_choices := ['A', 'I']
  let last_letter := 'R'
  let remaining_letters:= valid_letters.erase last_letter
  (first_letter_choices.length * (remaining_letters.length - first_letter_choices.length) * (remaining_letters.length - first_letter_choices.length - 1) * (remaining_letters.length - first_letter_choices.length - 2))

theorem isoland_license_plates_proof :
  isoland_license_plates = 420 := by
  sorry

end isoland_license_plates_proof_l1633_163330


namespace max_strong_boys_l1633_163305

theorem max_strong_boys (n : ℕ) (h : n = 100) (a b : Fin n → ℕ) 
  (ha : ∀ i j : Fin n, i < j → a i > a j) 
  (hb : ∀ i j : Fin n, i < j → b i < b j) : 
  ∃ k : ℕ, k = n := 
sorry

end max_strong_boys_l1633_163305


namespace least_value_of_N_l1633_163329

theorem least_value_of_N : ∃ (N : ℕ), (N % 6 = 5) ∧ (N % 5 = 4) ∧ (N % 4 = 3) ∧ (N % 3 = 2) ∧ (N % 2 = 1) ∧ N = 59 :=
by
  sorry

end least_value_of_N_l1633_163329


namespace greatest_integer_difference_l1633_163314

theorem greatest_integer_difference (x y : ℤ) (h1 : 5 < x ∧ x < 8) (h2 : 8 < y ∧ y < 13)
  (h3 : x % 3 = 0) (h4 : y % 3 = 0) : y - x = 6 :=
sorry

end greatest_integer_difference_l1633_163314


namespace value_of_a_l1633_163346

theorem value_of_a
  (x y a : ℝ)
  (h1 : x + 2 * y = 2 * a - 1)
  (h2 : x - y = 6)
  (h3 : x = -y)
  : a = -1 :=
by
  sorry

end value_of_a_l1633_163346


namespace find_a5_l1633_163328

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Condition: The sum of the first n terms of the sequence {a_n} is represented by S_n = 2a_n - 1 (n ∈ ℕ)
axiom sum_of_terms (n : ℕ) : S n = 2 * (a n) - 1

-- Prove that a_5 = 16
theorem find_a5 : a 5 = 16 :=
  sorry

end find_a5_l1633_163328


namespace range_of_a_l1633_163389

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1) * x + a^2 > 0) ↔ (a < -1 ∨ a > (1 : ℝ) / 3) := 
sorry

end range_of_a_l1633_163389


namespace probability_of_three_blue_beans_l1633_163351

-- Define the conditions
def red_jellybeans : ℕ := 10 
def blue_jellybeans : ℕ := 10 
def total_jellybeans : ℕ := red_jellybeans + blue_jellybeans 
def draws : ℕ := 3 

-- Define the events
def P_first_blue : ℚ := blue_jellybeans / total_jellybeans 
def P_second_blue : ℚ := (blue_jellybeans - 1) / (total_jellybeans - 1) 
def P_third_blue : ℚ := (blue_jellybeans - 2) / (total_jellybeans - 2) 
def P_all_three_blue : ℚ := P_first_blue * P_second_blue * P_third_blue 

-- Define the correct answer
def correct_probability : ℚ := 1 / 9.5 

-- State the theorem
theorem probability_of_three_blue_beans : 
  P_all_three_blue = correct_probability := 
sorry

end probability_of_three_blue_beans_l1633_163351


namespace train_speed_l1633_163373

theorem train_speed
  (length_of_train : ℝ)
  (time_to_cross_pole : ℝ)
  (h1 : length_of_train = 3000)
  (h2 : time_to_cross_pole = 120) :
  length_of_train / time_to_cross_pole = 25 :=
by {
  sorry
}

end train_speed_l1633_163373


namespace binomial_probability_l1633_163348

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- Define the binomial probability mass function
def binomial_pmf (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coeff n k) * (p^k) * ((1 - p)^(n - k))

-- Define the conditions of the problem
def n := 5
def k := 2
def p : ℚ := 1/3

-- State the theorem
theorem binomial_probability :
  binomial_pmf n k p = binomial_coeff 5 2 * (1/3)^2 * (2/3)^3 := by
  sorry

end binomial_probability_l1633_163348


namespace english_book_pages_l1633_163312

def numPagesInOneEnglishBook (x y : ℕ) : Prop :=
  x = y + 12 ∧ 3 * x + 4 * y = 1275 → x = 189

-- The statement with sorry as no proof is required:
theorem english_book_pages (x y : ℕ) (h1 : x = y + 12) (h2 : 3 * x + 4 * y = 1275) : x = 189 :=
  sorry

end english_book_pages_l1633_163312


namespace total_slices_left_is_14_l1633_163391

-- Define the initial conditions
def large_pizza_slices : ℕ := 12
def small_pizza_slices : ℕ := 8
def hawaiian_pizza (num_large : ℕ) : ℕ := num_large * large_pizza_slices
def cheese_pizza (num_large : ℕ) : ℕ := num_large * large_pizza_slices
def pepperoni_pizza (num_small : ℕ) : ℕ := num_small * small_pizza_slices

-- Number of large pizzas ordered (Hawaiian and cheese)
def num_large_pizzas : ℕ := 2

-- Number of small pizzas received in promotion
def num_small_pizzas : ℕ := 1

-- Slices eaten by each person
def dean_slices (hawaiian_slices : ℕ) : ℕ := hawaiian_slices / 2
def frank_slices : ℕ := 3
def sammy_slices (cheese_slices : ℕ) : ℕ := cheese_slices / 3
def nancy_cheese_slices : ℕ := 2
def nancy_pepperoni_slice : ℕ := 1
def olivia_slices : ℕ := 2

-- Total slices eaten from each pizza
def total_hawaiian_slices_eaten (hawaiian_slices : ℕ) : ℕ := dean_slices hawaiian_slices + frank_slices
def total_cheese_slices_eaten (cheese_slices : ℕ) : ℕ := sammy_slices cheese_slices + nancy_cheese_slices
def total_pepperoni_slices_eaten : ℕ := nancy_pepperoni_slice + olivia_slices

-- Total slices left over
def total_slices_left (hawaiian_slices : ℕ) (cheese_slices : ℕ) (pepperoni_slices : ℕ) : ℕ := 
  (hawaiian_slices - total_hawaiian_slices_eaten hawaiian_slices) + 
  (cheese_slices - total_cheese_slices_eaten cheese_slices) + 
  (pepperoni_slices - total_pepperoni_slices_eaten)

-- The actual Lean 4 statement to be verified
theorem total_slices_left_is_14 : total_slices_left (hawaiian_pizza num_large_pizzas) (cheese_pizza num_large_pizzas) (pepperoni_pizza num_small_pizzas) = 14 := 
  sorry

end total_slices_left_is_14_l1633_163391


namespace sixth_term_of_geometric_seq_l1633_163384

-- conditions
def is_geometric_sequence (seq : ℕ → ℕ) := 
  ∃ r : ℕ, ∀ n : ℕ, seq (n + 1) = seq n * r

def first_term (seq : ℕ → ℕ) := seq 1 = 3
def fifth_term (seq : ℕ → ℕ) := seq 5 = 243

-- question to be proved
theorem sixth_term_of_geometric_seq (seq : ℕ → ℕ) 
  (h_geom : is_geometric_sequence seq) 
  (h_first : first_term seq) 
  (h_fifth : fifth_term seq) : 
  seq 6 = 729 :=
sorry

end sixth_term_of_geometric_seq_l1633_163384


namespace selling_price_is_correct_l1633_163363

-- Definitions based on conditions
def cost_price : ℝ := 280
def profit_percentage : ℝ := 0.3
def profit_amount : ℝ := cost_price * profit_percentage

-- Selling price definition
def selling_price : ℝ := cost_price + profit_amount

-- Theorem statement
theorem selling_price_is_correct : selling_price = 364 := by
  sorry

end selling_price_is_correct_l1633_163363


namespace seats_per_bus_l1633_163331

theorem seats_per_bus (students : ℕ) (buses : ℕ) (h1 : students = 111) (h2 : buses = 37) : students / buses = 3 := by
  sorry

end seats_per_bus_l1633_163331


namespace find_total_cost_price_l1633_163320

noncomputable def cost_prices (C1 C2 C3 : ℝ) : Prop :=
  0.85 * C1 + 72.50 = 1.125 * C1 ∧
  1.20 * C2 - 45.30 = 0.95 * C2 ∧
  0.92 * C3 + 33.60 = 1.10 * C3

theorem find_total_cost_price :
  ∃ (C1 C2 C3 : ℝ), cost_prices C1 C2 C3 ∧ C1 + C2 + C3 = 631.51 := 
by
  sorry

end find_total_cost_price_l1633_163320


namespace molecular_weight_correct_l1633_163386

namespace MolecularWeight

-- Define the atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_Cl : ℝ := 35.45

-- Define the number of each atom in the compound
def n_N : ℝ := 1
def n_H : ℝ := 4
def n_Cl : ℝ := 1

-- Calculate the molecular weight of the compound
def molecular_weight : ℝ := (n_N * atomic_weight_N) + (n_H * atomic_weight_H) + (n_Cl * atomic_weight_Cl)

theorem molecular_weight_correct : molecular_weight = 53.50 := by
  -- Proof is omitted
  sorry

end MolecularWeight

end molecular_weight_correct_l1633_163386


namespace goose_eggs_laied_l1633_163301

theorem goose_eggs_laied (z : ℕ) (hatch_rate : ℚ := 2 / 3) (first_month_survival_rate : ℚ := 3 / 4) 
  (first_year_survival_rate : ℚ := 2 / 5) (geese_survived_first_year : ℕ := 126) :
  (hatch_rate * z) = 420 ∧ (first_month_survival_rate * 315 = 315) ∧ (first_year_survival_rate * 315 = 126) →
  z = 630 :=
by
  sorry

end goose_eggs_laied_l1633_163301


namespace minimize_quadratic_l1633_163310

theorem minimize_quadratic (x : ℝ) : (x = -9 / 2) → ∀ y : ℝ, y^2 + 9 * y + 7 ≥ (-9 / 2)^2 + 9 * -9 / 2 + 7 :=
by sorry

end minimize_quadratic_l1633_163310


namespace billy_watches_videos_l1633_163376

-- Conditions definitions
def num_suggestions_per_list : Nat := 15
def num_iterations : Nat := 5
def pick_index_on_final_list : Nat := 5

-- Main theorem statement
theorem billy_watches_videos : 
  num_suggestions_per_list * num_iterations + (pick_index_on_final_list - 1) = 79 :=
by
  sorry

end billy_watches_videos_l1633_163376


namespace total_time_taken_l1633_163349

theorem total_time_taken (speed_boat : ℕ) (speed_stream : ℕ) (distance : ℕ) 
    (h1 : speed_boat = 12) (h2 : speed_stream = 4) (h3 : distance = 480) : 
    ((distance / (speed_boat + speed_stream)) + (distance / (speed_boat - speed_stream)) = 90) :=
by
  -- Sorry is used to skip the proof
  sorry

end total_time_taken_l1633_163349


namespace jordan_time_to_run_7_miles_l1633_163398

def time_taken (distance time_per_unit : ℝ) : ℝ :=
  distance * time_per_unit

theorem jordan_time_to_run_7_miles :
  ∀ (t_S d_S d_J : ℝ), t_S = 36 → d_S = 6 → d_J = 4 → time_taken 7 ((t_S / 2) / d_J) = 31.5 :=
by
  intros t_S d_S d_J h_t_S h_d_S h_d_J
  -- skipping the proof
  sorry

end jordan_time_to_run_7_miles_l1633_163398


namespace cover_with_L_shapes_l1633_163306

def L_shaped (m n : ℕ) : Prop :=
  m > 1 ∧ n > 1 ∧ ∃ k, m * n = 8 * k -- Conditions and tiling pattern coverage.

-- Problem statement as a theorem
theorem cover_with_L_shapes (m n : ℕ) (h1 : m > 1) (h2 : n > 1) : (∃ k, m * n = 8 * k) ↔ L_shaped m n :=
-- Placeholder for the proof
sorry

end cover_with_L_shapes_l1633_163306


namespace find_x_l1633_163359

theorem find_x (y z : ℚ) (h1 : z = 80) (h2 : y = z / 4) (h3 : x = y / 3) : x = 20 / 3 :=
by
  sorry

end find_x_l1633_163359


namespace min_distance_origin_to_line_l1633_163394

theorem min_distance_origin_to_line (a b : ℝ) (h : a + 2 * b = Real.sqrt 5) : 
  Real.sqrt (a^2 + b^2) ≥ 1 :=
sorry

end min_distance_origin_to_line_l1633_163394


namespace kenny_played_basketball_last_week_l1633_163360

def time_practicing_trumpet : ℕ := 40
def time_running : ℕ := time_practicing_trumpet / 2
def time_playing_basketball : ℕ := time_running / 2
def answer : ℕ := 10

theorem kenny_played_basketball_last_week :
  time_playing_basketball = answer :=
by
  -- sorry to skip the proof
  sorry

end kenny_played_basketball_last_week_l1633_163360


namespace citric_acid_molecular_weight_l1633_163327

def molecular_weight_citric_acid := 192.12 -- in g/mol

theorem citric_acid_molecular_weight :
  molecular_weight_citric_acid = 192.12 :=
by sorry

end citric_acid_molecular_weight_l1633_163327


namespace isosceles_triangle_angle_sum_l1633_163350

theorem isosceles_triangle_angle_sum (y : ℕ) (a : ℕ) (b : ℕ) 
  (h_isosceles : a = b ∨ a = y ∨ b = y)
  (h_sum : a + b + y = 180) :
  a = 80 → b = 80 → y = 50 ∨ y = 20 ∨ y = 80 → y + y + y = 150 :=
by
  sorry

end isosceles_triangle_angle_sum_l1633_163350


namespace original_price_of_shirt_l1633_163375

theorem original_price_of_shirt (discounted_price : ℝ) (discount_percentage : ℝ) 
  (h_discounted_price : discounted_price = 780) (h_discount_percentage : discount_percentage = 0.20) 
  : (discounted_price / (1 - discount_percentage) = 975) := by
  sorry

end original_price_of_shirt_l1633_163375


namespace sin_60_eq_sqrt3_div_2_l1633_163377

theorem sin_60_eq_sqrt3_div_2 : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  -- proof skipped
  sorry

end sin_60_eq_sqrt3_div_2_l1633_163377


namespace remainder_1234567_div_by_137_l1633_163356

theorem remainder_1234567_div_by_137 :
  (1234567 % 137) = 102 :=
by {
  sorry
}

end remainder_1234567_div_by_137_l1633_163356


namespace area_of_shaded_part_l1633_163307

-- Define the given condition: area of the square
def area_of_square : ℝ := 100

-- Define the proof goal: area of the shaded part
theorem area_of_shaded_part : area_of_square / 2 = 50 := by
  sorry

end area_of_shaded_part_l1633_163307


namespace roots_of_polynomial_l1633_163372

theorem roots_of_polynomial :
  {x | x * (2 * x - 5) ^ 2 * (x + 3) * (7 - x) = 0} = {0, 2.5, -3, 7} :=
by {
  sorry
}

end roots_of_polynomial_l1633_163372


namespace geometric_sequence_a3a5_l1633_163303

theorem geometric_sequence_a3a5 :
  ∀ (a : ℕ → ℝ) (r : ℝ), (a 4 = 4) → (a 3 = a 0 * r ^ 3) → (a 5 = a 0 * r ^ 5) →
  a 3 * a 5 = 16 :=
by
  intros a r h1 h2 h3
  sorry

end geometric_sequence_a3a5_l1633_163303


namespace bert_initial_amount_l1633_163322

theorem bert_initial_amount (n : ℝ) (h : (1 / 2) * (3 / 4 * n - 9) = 12) : n = 44 :=
sorry

end bert_initial_amount_l1633_163322


namespace math_problem_l1633_163366

theorem math_problem (A B C : ℕ) (h_pos : A > 0 ∧ B > 0 ∧ C > 0) (h_gcd : Nat.gcd (Nat.gcd A B) C = 1) (h_eq : A * Real.log 5 / Real.log 200 + B * Real.log 2 / Real.log 200 = C) : A + B + C = 6 :=
sorry

end math_problem_l1633_163366


namespace maria_younger_than_ann_l1633_163370

variable (M A : ℕ)

def maria_current_age : Prop := M = 7

def age_relation_four_years_ago : Prop := M - 4 = (1 / 2) * (A - 4)

theorem maria_younger_than_ann :
  maria_current_age M → age_relation_four_years_ago M A → A - M = 3 :=
by
  sorry

end maria_younger_than_ann_l1633_163370


namespace team_a_vs_team_b_l1633_163361

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem team_a_vs_team_b (P1 P2 : ℝ) :
  let n_a := 5
  let x_a := 4
  let p_a := 0.5
  let n_b := 5
  let x_b := 3
  let p_b := 1/3
  let P1 := binomial_probability n_a x_a p_a
  let P2 := binomial_probability n_b x_b p_b
  P1 < P2 := by sorry

end team_a_vs_team_b_l1633_163361


namespace Paul_lost_161_crayons_l1633_163369

def total_crayons : Nat := 589
def crayons_given : Nat := 571
def extra_crayons_given : Nat := 410

theorem Paul_lost_161_crayons : ∃ L : Nat, crayons_given = L + extra_crayons_given ∧ L = 161 := by
  sorry

end Paul_lost_161_crayons_l1633_163369


namespace pounds_over_weight_limit_l1633_163344

-- Definitions based on conditions

def bookcase_max_weight : ℝ := 80

def weight_hardcover_book : ℝ := 0.5
def number_hardcover_books : ℕ := 70
def total_weight_hardcover_books : ℝ := number_hardcover_books * weight_hardcover_book

def weight_textbook : ℝ := 2
def number_textbooks : ℕ := 30
def total_weight_textbooks : ℝ := number_textbooks * weight_textbook

def weight_knick_knack : ℝ := 6
def number_knick_knacks : ℕ := 3
def total_weight_knick_knacks : ℝ := number_knick_knacks * weight_knick_knack

def total_weight_items : ℝ := total_weight_hardcover_books + total_weight_textbooks + total_weight_knick_knacks

theorem pounds_over_weight_limit : total_weight_items - bookcase_max_weight = 33 := by
  sorry

end pounds_over_weight_limit_l1633_163344


namespace quadratic_roots_equal_l1633_163353

theorem quadratic_roots_equal (m : ℝ) :
  (∃ x : ℝ, x^2 - 4 * x + m - 1 = 0 ∧ (∀ y : ℝ, y^2 - 4*y + m-1 = 0 → y = x)) ↔ (m = 5 ∧ (∀ x, x^2 - 4 * x + 4 = 0 ↔ x = 2)) :=
by
  sorry

end quadratic_roots_equal_l1633_163353


namespace lawn_chair_original_price_l1633_163309

theorem lawn_chair_original_price (sale_price : ℝ) (discount_percentage : ℝ) (original_price : ℝ) :
  sale_price = 59.95 →
  discount_percentage = 23.09 →
  original_price = sale_price / (1 - discount_percentage / 100) →
  original_price = 77.95 :=
by sorry

end lawn_chair_original_price_l1633_163309


namespace sum_values_l1633_163374

noncomputable def abs_eq_4 (x : ℝ) : Prop := |x| = 4
noncomputable def abs_eq_5 (x : ℝ) : Prop := |x| = 5

theorem sum_values (a b : ℝ) (h₁ : abs_eq_4 a) (h₂ : abs_eq_5 b) :
  a + b = 9 ∨ a + b = -1 ∨ a + b = 1 ∨ a + b = -9 := 
by
  -- Proof is omitted
  sorry

end sum_values_l1633_163374


namespace exists_positive_n_with_m_zeros_l1633_163333

theorem exists_positive_n_with_m_zeros (m : ℕ) (hm : 0 < m) :
  ∃ n : ℕ, 0 < n ∧ ∃ k : ℕ, 7^n = k * 10^m :=
sorry

end exists_positive_n_with_m_zeros_l1633_163333


namespace max_dogs_and_fish_l1633_163388

theorem max_dogs_and_fish (d c b p f : ℕ) (h_ratio : d / 7 = c / 7 ∧ d / 7 = b / 8 ∧ d / 7 = p / 3 ∧ d / 7 = f / 5)
  (h_dogs_bunnies : d + b = 330)
  (h_twice_fish : f ≥ 2 * c) :
  d = 154 ∧ f = 308 :=
by
  -- This is where the proof would go
  sorry

end max_dogs_and_fish_l1633_163388


namespace mobile_price_two_years_ago_l1633_163383

-- Definitions and conditions
def price_now : ℝ := 1000
def decrease_rate : ℝ := 0.2
def years_ago : ℝ := 2

-- Main statement
theorem mobile_price_two_years_ago :
  ∃ (a : ℝ), a * (1 - decrease_rate)^years_ago = price_now :=
sorry

end mobile_price_two_years_ago_l1633_163383


namespace parabola_points_relationship_l1633_163397

theorem parabola_points_relationship (c y1 y2 y3 : ℝ)
  (h1 : y1 = -0^2 + 2 * 0 + c)
  (h2 : y2 = -1^2 + 2 * 1 + c)
  (h3 : y3 = -3^2 + 2 * 3 + c) :
  y2 > y1 ∧ y1 > y3 := by
  sorry

end parabola_points_relationship_l1633_163397


namespace max_pieces_with_single_cut_min_cuts_to_intersect_all_pieces_l1633_163354

theorem max_pieces_with_single_cut (n : ℕ) (h : n = 4) :
  (∃ m : ℕ, m = 23) :=
sorry

theorem min_cuts_to_intersect_all_pieces (n : ℕ) (h : n = 4) :
  (∃ k : ℕ, k = 3) :=
sorry

noncomputable def pieces_of_cake : ℕ := 23

noncomputable def cuts_required : ℕ := 3

end max_pieces_with_single_cut_min_cuts_to_intersect_all_pieces_l1633_163354
