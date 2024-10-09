import Mathlib

namespace line_parallel_to_parallel_set_l1842_184200

variables {Point Line Plane : Type} 
variables (a : Line) (α : Plane)
variables (parallel : Line → Plane → Prop) (parallel_set : Line → Plane → Prop)

-- Definition for line parallel to plane
axiom line_parallel_to_plane : parallel a α

-- Goal: line a is parallel to a set of parallel lines within plane α
theorem line_parallel_to_parallel_set (h : parallel a α) : parallel_set a α := 
sorry

end line_parallel_to_parallel_set_l1842_184200


namespace travel_time_K_l1842_184206

theorem travel_time_K (d x : ℝ) (h_pos_d : d > 0) (h_x_pos : x > 0) (h_time_diff : (d / (x - 1/2)) - (d / x) = 1/2) : d / x = 40 / x :=
by
  sorry

end travel_time_K_l1842_184206


namespace congruence_problem_l1842_184299

theorem congruence_problem (x : ℤ) (h : 5 * x + 9 ≡ 4 [ZMOD 18]) : 3 * x + 15 ≡ 12 [ZMOD 18] :=
sorry

end congruence_problem_l1842_184299


namespace number_of_girls_attending_picnic_l1842_184280

variables (g b : ℕ)

def hms_conditions : Prop :=
  g + b = 1500 ∧ (3 / 4 : ℝ) * g + (3 / 5 : ℝ) * b = 975

theorem number_of_girls_attending_picnic (h : hms_conditions g b) : (3 / 4 : ℝ) * g = 375 :=
sorry

end number_of_girls_attending_picnic_l1842_184280


namespace number_of_pencils_l1842_184261

theorem number_of_pencils 
  (P Pe M : ℕ)
  (h1 : Pe = P + 4)
  (h2 : M = P + 20)
  (h3 : P / 5 = Pe / 6)
  (h4 : Pe / 6 = M / 7) : 
  Pe = 24 :=
by
  sorry

end number_of_pencils_l1842_184261


namespace Mr_Blue_potato_yield_l1842_184260

-- Definitions based on the conditions
def steps_length (steps : ℕ) : ℕ := steps * 3
def garden_length : ℕ := steps_length 18
def garden_width : ℕ := steps_length 25

def area_garden : ℕ := garden_length * garden_width
def yield_potatoes (area : ℕ) : ℚ := area * (3/4)

-- Statement of the proof
theorem Mr_Blue_potato_yield :
  yield_potatoes area_garden = 3037.5 := by
  sorry

end Mr_Blue_potato_yield_l1842_184260


namespace find_unknown_number_l1842_184290

theorem find_unknown_number :
  (0.86 ^ 3 - 0.1 ^ 3) / (0.86 ^ 2) + x + 0.1 ^ 2 = 0.76 → 
  x = 0.115296 :=
sorry

end find_unknown_number_l1842_184290


namespace point_inside_circle_range_l1842_184209

theorem point_inside_circle_range (a : ℝ) : ((1 - a)^2 + (1 + a)^2 < 4) → (-1 < a ∧ a < 1) :=
  by
  sorry

end point_inside_circle_range_l1842_184209


namespace calc_r_over_s_at_2_l1842_184242

def r (x : ℝ) := 3 * (x - 4) * (x - 1)
def s (x : ℝ) := (x - 4) * (x + 3)

theorem calc_r_over_s_at_2 : (r 2) / (s 2) = 3 / 5 := by
  sorry

end calc_r_over_s_at_2_l1842_184242


namespace tulip_to_remaining_ratio_l1842_184272

theorem tulip_to_remaining_ratio (total_flowers daisies sunflowers tulips remaining_tulips remaining_flowers : ℕ) 
  (h1 : total_flowers = 12) 
  (h2 : daisies = 2) 
  (h3 : sunflowers = 4) 
  (h4 : tulips = total_flowers - (daisies + sunflowers))
  (h5 : remaining_tulips = tulips)
  (h6 : remaining_flowers = remaining_tulips + sunflowers)
  (h7 : remaining_flowers = 10) : 
  tulips / remaining_flowers = 3 / 5 := 
by
  sorry

end tulip_to_remaining_ratio_l1842_184272


namespace total_students_l1842_184274

variable (T : ℕ)

-- Conditions
def is_girls_percentage (T : ℕ) := 60 / 100 * T
def is_boys_percentage (T : ℕ) := 40 / 100 * T
def boys_not_in_clubs (number_of_boys : ℕ) := 2 / 3 * number_of_boys

theorem total_students (h1 : is_girls_percentage T + is_boys_percentage T = T)
  (h2 : boys_not_in_clubs (is_boys_percentage T) = 40) : T = 150 :=
by
  sorry

end total_students_l1842_184274


namespace arithmetic_mean_geom_mean_ratio_l1842_184297

theorem arithmetic_mean_geom_mean_ratio {a b : ℝ} (h1 : (a + b) / 2 = 3 * Real.sqrt (a * b)) (h2 : a > b) (h3 : b > 0) : 
  (∃ k : ℤ, k = 34 ∧ abs ((a / b) - 34) ≤ 0.5) :=
sorry

end arithmetic_mean_geom_mean_ratio_l1842_184297


namespace complex_pow_imaginary_unit_l1842_184236

theorem complex_pow_imaginary_unit (i : ℂ) (h : i^2 = -1) : i^2015 = -i :=
sorry

end complex_pow_imaginary_unit_l1842_184236


namespace problem_I_problem_II_1_problem_II_2_l1842_184259

section
variables (boys_A girls_A boys_B girls_B : ℕ)
variables (total_students : ℕ)

-- Define the conditions
def conditions : Prop :=
  boys_A = 2 ∧ girls_A = 1 ∧ boys_B = 3 ∧ girls_B = 2 ∧ total_students = boys_A + girls_A + boys_B + girls_B

-- Problem (I)
theorem problem_I (h : conditions boys_A girls_A boys_B girls_B total_students) :
  ∃ arrangements, arrangements = 14400 := sorry

-- Problem (II.1)
theorem problem_II_1 (h : conditions boys_A girls_A boys_B girls_B total_students) :
  ∃ prob, prob = 13 / 14 := sorry

-- Problem (II.2)
theorem problem_II_2 (h : conditions boys_A girls_A boys_B girls_B total_students) :
  ∃ prob, prob = 6 / 35 := sorry
end

end problem_I_problem_II_1_problem_II_2_l1842_184259


namespace gcd_of_102_and_238_l1842_184256

theorem gcd_of_102_and_238 : Nat.gcd 102 238 = 34 := 
by 
  sorry

end gcd_of_102_and_238_l1842_184256


namespace difference_of_numbers_l1842_184232

theorem difference_of_numbers (a b : ℕ) (h1 : a + b = 34800) (h2 : b % 25 = 0) (h3 : b / 100 = a) : b - a = 32112 := by
  sorry

end difference_of_numbers_l1842_184232


namespace domain_of_fractional_sqrt_function_l1842_184268

theorem domain_of_fractional_sqrt_function :
  ∀ x : ℝ, (x + 4 ≥ 0) ∧ (x - 1 ≠ 0) ↔ (x ∈ (Set.Ici (-4) \ {1})) :=
by
  sorry

end domain_of_fractional_sqrt_function_l1842_184268


namespace garden_enlargement_l1842_184276

theorem garden_enlargement :
  let length := 60
  let width := 20
  let perimeter := 2 * (length + width)
  let side_square := perimeter / 4
  let area_rectangular := length * width
  let area_square := side_square * side_square
  area_square - area_rectangular = 400 := by
  -- initializing all definitions
  let length := 60
  let width := 20
  let perimeter := 2 * (length + width)
  let side_square := perimeter / 4
  let area_rectangular := length * width
  let area_square := side_square * side_square
  -- placeholder for the actual proof
  sorry

end garden_enlargement_l1842_184276


namespace maxValue_a1_l1842_184279

variable (a_1 q : ℝ)

def isGeometricSequence (a_1 q : ℝ) : Prop :=
  a_1 ≥ 1 ∧ a_1 * q ≤ 2 ∧ a_1 * q^2 ≥ 3

theorem maxValue_a1 (h : isGeometricSequence a_1 q) : a_1 ≤ 4 / 3 := 
sorry

end maxValue_a1_l1842_184279


namespace calculate_expression_l1842_184289

theorem calculate_expression :
  (-0.125)^2022 * 8^2023 = 8 :=
sorry

end calculate_expression_l1842_184289


namespace neg_four_is_square_root_of_sixteen_l1842_184243

/-
  Definitions:
  - A number y is a square root of x if y^2 = x.
  - A number y is an arithmetic square root of x if y ≥ 0 and y^2 = x.
-/

theorem neg_four_is_square_root_of_sixteen :
  -4 * -4 = 16 := 
by
  -- proof step is omitted
  sorry

end neg_four_is_square_root_of_sixteen_l1842_184243


namespace sports_field_perimeter_l1842_184287

noncomputable def perimeter_of_sports_field (a b : ℝ) (h1 : a^2 + b^2 = 400) (h2 : a * b = 120) : ℝ :=
  2 * (a + b)

theorem sports_field_perimeter {a b : ℝ} (h1 : a^2 + b^2 = 400) (h2 : a * b = 120) :
  perimeter_of_sports_field a b h1 h2 = 51 := by
  sorry

end sports_field_perimeter_l1842_184287


namespace minimize_surface_area_l1842_184228

theorem minimize_surface_area (V r h : ℝ) (hV : V = π * r^2 * h) (hA : 2 * π * r^2 + 2 * π * r * h = 2 * π * r^2 + 2 * π * r * h) : 
  (h / r) = 2 := 
by
  sorry

end minimize_surface_area_l1842_184228


namespace determine_marriages_l1842_184275

-- Definitions of the items each person bought
variable (a_items b_items c_items : ℕ) -- Number of items bought by wives a, b, and c
variable (A_items B_items C_items : ℕ) -- Number of items bought by husbands A, B, and C

-- Conditions
variable (spend_eq_square_a : a_items * a_items = a_spend) -- Spending equals square of items
variable (spend_eq_square_b : b_items * b_items = b_spend)
variable (spend_eq_square_c : c_items * c_items = c_spend)
variable (spend_eq_square_A : A_items * A_items = A_spend)
variable (spend_eq_square_B : B_items * B_items = B_spend)
variable (spend_eq_square_C : C_items * C_items = C_spend)

variable (A_spend_eq : A_spend = a_spend + 48) -- Husbands spent 48 yuan more than wives
variable (B_spend_eq : B_spend = b_spend + 48)
variable (C_spend_eq : C_spend = c_spend + 48)

variable (A_bought_9_more : A_items = b_items + 9) -- A bought 9 more items than b
variable (B_bought_7_more : B_items = a_items + 7) -- B bought 7 more items than a

-- Theorem statement
theorem determine_marriages (hA : A_items ≥ b_items + 9) (hB : B_items ≥ a_items + 7) :
  (A_spend = A_items * A_items) ∧ (B_spend = B_items * B_items) ∧ (C_spend = C_items * C_items) ∧
  (a_spend = a_items * a_items) ∧ (b_spend = b_items * b_items) ∧ (c_spend = c_items * c_items) →
  (A_spend = a_spend + 48) ∧ (B_spend = b_spend + 48) ∧ (C_spend = c_spend + 48) →
  (A_items = b_items + 9) ∧ (B_items = a_items + 7) →
  (A_items = 13 ∧ c_items = 11) ∧ (B_items = 8 ∧ b_items = 4) ∧ (C_items = 7 ∧ a_items = 1) :=
by
  sorry

end determine_marriages_l1842_184275


namespace a3_value_l1842_184217

theorem a3_value (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) (x : ℝ) :
  ( (1 + x) * (a - x) ^ 6 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 ) →
  ( a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0 ) →
  a = 1 →
  a₃ = -5 :=
by
  sorry

end a3_value_l1842_184217


namespace evaluate_five_iterates_of_f_at_one_l1842_184239

def f (x : ℕ) : ℕ :=
if x % 2 = 0 then x / 2 else 5 * x + 1

theorem evaluate_five_iterates_of_f_at_one :
  f (f (f (f (f 1)))) = 4 := by
  sorry

end evaluate_five_iterates_of_f_at_one_l1842_184239


namespace xiaoli_estimate_greater_l1842_184286

variable (p q a b : ℝ)

theorem xiaoli_estimate_greater (hpq : p > q) (hq0 : q > 0) (hab : a > b) : (p + a) - (q + b) > p - q := 
by 
  sorry

end xiaoli_estimate_greater_l1842_184286


namespace megatek_manufacturing_percentage_l1842_184295

-- Define the given conditions
def sector_deg : ℝ := 18
def full_circle_deg : ℝ := 360

-- Define the problem as a theorem statement in Lean
theorem megatek_manufacturing_percentage : 
  (sector_deg / full_circle_deg) * 100 = 5 := 
sorry

end megatek_manufacturing_percentage_l1842_184295


namespace cubic_roots_expression_l1842_184277

theorem cubic_roots_expression (p q r : ℝ) 
  (h1 : p + q + r = 4) 
  (h2 : pq + pr + qr = 6) 
  (h3 : pqr = 3) : 
  p / (qr + 2) + q / (pr + 2) + r / (pq + 2) = 4 / 5 := 
by 
  sorry

end cubic_roots_expression_l1842_184277


namespace find_teddy_dogs_l1842_184258

-- Definitions from the conditions
def teddy_cats := 8
def ben_dogs (teddy_dogs : ℕ) := teddy_dogs + 9
def dave_cats (teddy_cats : ℕ) := teddy_cats + 13
def dave_dogs (teddy_dogs : ℕ) := teddy_dogs - 5
def total_pets (teddy_dogs teddy_cats : ℕ) := teddy_dogs + teddy_cats + (ben_dogs teddy_dogs) + (dave_dogs teddy_dogs) + (dave_cats teddy_cats)

-- Theorem statement
theorem find_teddy_dogs (teddy_dogs : ℕ) (teddy_cats : ℕ) (hd : total_pets teddy_dogs teddy_cats = 54) :
  teddy_dogs = 7 := sorry

end find_teddy_dogs_l1842_184258


namespace total_new_students_l1842_184273

-- Given conditions
def number_of_schools : ℝ := 25.0
def average_students_per_school : ℝ := 9.88

-- Problem statement
theorem total_new_students : number_of_schools * average_students_per_school = 247 :=
by sorry

end total_new_students_l1842_184273


namespace total_amount_due_l1842_184246

noncomputable def original_bill : ℝ := 500
noncomputable def late_charge_rate : ℝ := 0.02
noncomputable def annual_interest_rate : ℝ := 0.05

theorem total_amount_due (n : ℕ) (initial_amount : ℝ) (late_charge_rate : ℝ) (interest_rate : ℝ) : 
  initial_amount = 500 → 
  late_charge_rate = 0.02 → 
  interest_rate = 0.05 → 
  n = 3 → 
  (initial_amount * (1 + late_charge_rate)^n * (1 + interest_rate) = 557.13) :=
by
  intros h_initial_amount h_late_charge_rate h_interest_rate h_n
  sorry

end total_amount_due_l1842_184246


namespace contrapositive_of_implication_l1842_184281

theorem contrapositive_of_implication (p q : Prop) (h : p → q) : ¬q → ¬p :=
by {
  sorry
}

end contrapositive_of_implication_l1842_184281


namespace rods_in_one_mile_l1842_184222

theorem rods_in_one_mile (chains_in_mile : ℕ) (rods_in_chain : ℕ) (mile_to_chain : 1 = 10 * chains_in_mile) (chain_to_rod : 1 = 22 * rods_in_chain) :
  1 * 220 = 10 * 22 :=
by sorry

end rods_in_one_mile_l1842_184222


namespace distance_between_foci_of_ellipse_l1842_184201

theorem distance_between_foci_of_ellipse : 
  let a := 5
  let b := 2
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 2 * Real.sqrt 21 :=
by
  sorry

end distance_between_foci_of_ellipse_l1842_184201


namespace perpendicular_lines_condition_l1842_184208

theorem perpendicular_lines_condition (m : ℝ) :
  (m = -1) ↔ ((m * 2 + 1 * m * (m - 1)) = 0) :=
sorry

end perpendicular_lines_condition_l1842_184208


namespace sum_of_youngest_and_oldest_cousins_l1842_184240

theorem sum_of_youngest_and_oldest_cousins 
  (a1 a2 a3 a4 : ℕ) 
  (h_order : a1 ≤ a2 ∧ a2 ≤ a3 ∧ a3 ≤ a4) 
  (h_mean : a1 + a2 + a3 + a4 = 36) 
  (h_median : a2 + a3 = 14) : 
  a1 + a4 = 22 :=
by sorry

end sum_of_youngest_and_oldest_cousins_l1842_184240


namespace total_volume_l1842_184283

-- Defining the volumes for different parts as per the conditions.
variables (V_A V_C V_B' V_C' : ℝ)
variables (V : ℝ)

-- The given conditions
axiom V_A_eq_40 : V_A = 40
axiom V_C_eq_300 : V_C = 300
axiom V_B'_eq_360 : V_B' = 360
axiom V_C'_eq_90 : V_C' = 90

-- The proof goal: total volume of the parallelepiped
theorem total_volume (V_A V_C V_B' V_C' : ℝ) 
  (V_A_eq_40 : V_A = 40) (V_C_eq_300 : V_C = 300) 
  (V_B'_eq_360 : V_B' = 360) (V_C'_eq_90 : V_C' = 90) :
  V = V_A + V_C + V_B' + V_C' :=
by
  sorry

end total_volume_l1842_184283


namespace eval_expression_eq_2_l1842_184247

theorem eval_expression_eq_2 :
  (10^2 + 11^2 + 12^2 + 13^2 + 14^2) / 365 = 2 :=
by
  sorry

end eval_expression_eq_2_l1842_184247


namespace arrange_in_ascending_order_l1842_184229

open Real

noncomputable def a := log 3 / log (1/2)
noncomputable def b := log 5 / log (1/2)
noncomputable def c := log (1/2) / log (1/3)

theorem arrange_in_ascending_order : b < a ∧ a < c :=
by
  sorry

end arrange_in_ascending_order_l1842_184229


namespace range_of_a_for_local_min_max_l1842_184245

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a_for_local_min_max (a e x1 x2 : ℝ) (h_a : 0 < a) (h_a_ne : a ≠ 1) (h_x1_x2 : x1 < x2) 
  (h_min : ∀ x, f a e x > f a e x1) (h_max : ∀ x, f a e x < f a e x2) : 
  (1 / Real.exp 1) < a ∧ a < 1 := 
sorry

end range_of_a_for_local_min_max_l1842_184245


namespace probability_at_least_one_die_less_3_l1842_184254

-- Definitions
def total_outcomes_dice : ℕ := 64
def outcomes_no_die_less_3 : ℕ := 36
def favorable_outcomes : ℕ := total_outcomes_dice - outcomes_no_die_less_3
def probability : ℚ := favorable_outcomes / total_outcomes_dice

-- Theorem statement
theorem probability_at_least_one_die_less_3 :
  probability = 7 / 16 :=
by
  -- Proof would go here
  sorry

end probability_at_least_one_die_less_3_l1842_184254


namespace common_ratio_of_gp_l1842_184219

theorem common_ratio_of_gp (a r : ℝ) (h1 : r ≠ 1) 
  (h2 : (a * (1 - r^6) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 343) : r = 6 := 
by
  sorry

end common_ratio_of_gp_l1842_184219


namespace first_term_of_arithmetic_sequence_l1842_184265

theorem first_term_of_arithmetic_sequence :
  ∃ (a_1 : ℤ), ∀ (d n : ℤ), d = 3 / 4 ∧ n = 30 ∧ a_n = 63 / 4 → a_1 = -6 := by
  sorry

end first_term_of_arithmetic_sequence_l1842_184265


namespace toby_total_time_l1842_184294

theorem toby_total_time (d1 d2 d3 d4 : ℕ)
  (speed_loaded speed_unloaded : ℕ)
  (time1 time2 time3 time4 total_time : ℕ)
  (h1 : d1 = 180)
  (h2 : d2 = 120)
  (h3 : d3 = 80)
  (h4 : d4 = 140)
  (h5 : speed_loaded = 10)
  (h6 : speed_unloaded = 20)
  (h7 : time1 = d1 / speed_loaded)
  (h8 : time2 = d2 / speed_unloaded)
  (h9 : time3 = d3 / speed_loaded)
  (h10 : time4 = d4 / speed_unloaded)
  (h11 : total_time = time1 + time2 + time3 + time4) :
  total_time = 39 := by
  sorry

end toby_total_time_l1842_184294


namespace graph_symmetry_l1842_184214

/-- Theorem:
The functions y = 2^x and y = 2^{-x} are symmetric about the y-axis.
-/
theorem graph_symmetry :
  ∀ (x : ℝ), (∃ (y : ℝ), y = 2^x) →
  (∃ (y' : ℝ), y' = 2^(-x)) →
  (∀ (y : ℝ), ∃ (x : ℝ), (y = 2^x ↔ y = 2^(-x)) → y = 2^x → y = 2^(-x)) :=
by
  intro x
  intro h1
  intro h2
  intro y
  exists x
  intro h3
  intro hy
  sorry

end graph_symmetry_l1842_184214


namespace probability_exactly_three_even_l1842_184270

theorem probability_exactly_three_even (p : ℕ → ℚ) (n : ℕ) (k : ℕ) (h : p 20 = 1/2 ∧ n = 5 ∧ k = 3) :
  (∃ C : ℚ, (C = (Nat.choose n k : ℚ)) ∧ (p 20)^n = 1/32) → (C * 1/32 = 5/16) :=
by
  sorry

end probability_exactly_three_even_l1842_184270


namespace probability_bc_seated_next_l1842_184216

theorem probability_bc_seated_next {P : ℝ} : 
  P = 2 / 3 :=
sorry

end probability_bc_seated_next_l1842_184216


namespace intersection_of_A_and_B_l1842_184230

def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}
def B : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-2, 0, 2} := by
  sorry

end intersection_of_A_and_B_l1842_184230


namespace tile_rectangle_condition_l1842_184252

theorem tile_rectangle_condition (k m n : ℕ) (hk : 0 < k) (hm : 0 < m) (hn : 0 < n) : 
  (∃ q, m = k * q) ∨ (∃ r, n = k * r) :=
sorry

end tile_rectangle_condition_l1842_184252


namespace num_solutions_triples_l1842_184285

theorem num_solutions_triples :
  {n : ℕ // ∃ a b c : ℤ, a^2 - a * (b + c) + b^2 - b * c + c^2 = 1 ∧ n = 10  } :=
  sorry

end num_solutions_triples_l1842_184285


namespace no_such_function_exists_l1842_184225

noncomputable def f : ℕ → ℕ := sorry

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ → ℕ), (∀ n > 1, f n = f (f (n-1)) + f (f (n+1))) ∧ (∀ n, f n > 0) :=
sorry

end no_such_function_exists_l1842_184225


namespace solution_set_l1842_184218

-- Define the conditions
variables {f : ℝ → ℝ}

-- Condition: f(x) is an odd function
axiom odd_function : ∀ x : ℝ, f (-x) = -f x

-- Condition: xf'(x) + f(x) < 0 for x in (-∞, 0)
axiom condition1 : ∀ x : ℝ, x < 0 → x * (deriv f x) + f x < 0

-- Condition: f(-2) = 0
axiom f_neg2_zero : f (-2) = 0

-- Goal: Prove the solution set of the inequality xf(x) < 0 is {x | -2 < x < 0 ∨ 0 < x < 2}
theorem solution_set : ∀ x : ℝ, (x * f x < 0) ↔ (-2 < x ∧ x < 0 ∨ 0 < x ∧ x < 2) := by
  sorry

end solution_set_l1842_184218


namespace missing_number_l1842_184205

theorem missing_number (x : ℝ) (h : 0.72 * 0.43 + x * 0.34 = 0.3504) : x = 0.12 :=
by sorry

end missing_number_l1842_184205


namespace tangent_position_is_six_l1842_184213

def clock_radius : ℝ := 30
def disk_radius : ℝ := 15
def initial_tangent_position := 12
def final_tangent_position := 6

theorem tangent_position_is_six :
  (∃ (clock_radius disk_radius : ℝ), clock_radius = 30 ∧ disk_radius = 15) →
  (initial_tangent_position = 12) →
  (final_tangent_position = 6) :=
by
  intros h1 h2
  sorry

end tangent_position_is_six_l1842_184213


namespace find_p_q_l1842_184288

def vector_a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
def vector_b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

theorem find_p_q (p q : ℝ)
  (h1 : 4 * 3 + p * 2 + (-2) * q = 0)
  (h2 : 4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2) :
  (p, q) = (-29/12, 43/12) :=
by 
  sorry

end find_p_q_l1842_184288


namespace min_deg_g_correct_l1842_184231

open Polynomial

noncomputable def min_deg_g {R : Type*} [CommRing R]
  (f g h : R[X])
  (hf : f.natDegree = 10)
  (hh : h.natDegree = 11)
  (h_eq : 5 * f + 6 * g = h) :
  Nat :=
11

theorem min_deg_g_correct {R : Type*} [CommRing R]
  (f g h : R[X])
  (hf : f.natDegree = 10)
  (hh : h.natDegree = 11)
  (h_eq : 5 * f + 6 * g = h) :
  (min_deg_g f g h hf hh h_eq = 11) :=
sorry

end min_deg_g_correct_l1842_184231


namespace hypotenuse_length_l1842_184266

theorem hypotenuse_length (a b : ℕ) (h1 : a = 36) (h2 : b = 48) : 
  ∃ c : ℕ, c * c = a * a + b * b ∧ c = 60 := 
by 
  use 60
  sorry

end hypotenuse_length_l1842_184266


namespace contrapositive_l1842_184282

theorem contrapositive (a b : ℝ) : (a ≠ 0 ∨ b ≠ 0) → a^2 + b^2 ≠ 0 :=
by
  intro h
  sorry

end contrapositive_l1842_184282


namespace bicycle_cost_after_tax_l1842_184251

theorem bicycle_cost_after_tax :
  let original_price := 300
  let first_discount := original_price * 0.40
  let price_after_first_discount := original_price - first_discount
  let second_discount := price_after_first_discount * 0.20
  let price_after_second_discount := price_after_first_discount - second_discount
  let tax := price_after_second_discount * 0.05
  price_after_second_discount + tax = 151.20 :=
by
  sorry

end bicycle_cost_after_tax_l1842_184251


namespace pensioners_painting_conditions_l1842_184269

def boardCondition (A Z : ℕ) : Prop :=
(∀ x y, (∃ i j, i ≤ 1 ∧ j ≤ 1 ∧ (x + 3 = A) ∧ (i ≤ 2 ∧ j ≤ 4 ∨ i ≤ 4 ∧ j ≤ 2) → x + 2 * y = Z))

theorem pensioners_painting_conditions (A Z : ℕ) :
  (boardCondition A Z) ↔ (A = 0 ∧ Z = 0) ∨ (A = 9 ∧ Z = 8) :=
sorry

end pensioners_painting_conditions_l1842_184269


namespace king_zenobius_more_descendants_l1842_184262

-- Conditions
def descendants_paphnutius (p2_descendants p1_descendants: ℕ) := 
  2 + 60 * p2_descendants + 20 * p1_descendants = 142

def descendants_zenobius (z3_descendants z1_descendants : ℕ) := 
  4 + 35 * z3_descendants + 35 * z1_descendants = 144

-- Main statement
theorem king_zenobius_more_descendants:
  ∀ (p2_descendants p1_descendants z3_descendants z1_descendants : ℕ),
    descendants_paphnutius p2_descendants p1_descendants →
    descendants_zenobius z3_descendants z1_descendants →
    144 > 142 :=
by
  intros
  sorry

end king_zenobius_more_descendants_l1842_184262


namespace sum_of_coefficients_l1842_184212

theorem sum_of_coefficients (a a1 a2 a3 a4 a5 : ℤ)
  (h : (1 - 2 * X)^5 = a + a1 * X + a2 * X^2 + a3 * X^3 + a4 * X^4 + a5 * X^5) :
  a1 + a2 + a3 + a4 + a5 = -2 :=
by {
  -- the proof steps would go here
  sorry
}

end sum_of_coefficients_l1842_184212


namespace quadratic_sum_l1842_184257

theorem quadratic_sum (x : ℝ) :
  (∃ a b c : ℝ, 6 * x^2 + 48 * x + 162 = a * (x + b) ^ 2 + c ∧ a + b + c = 76) :=
by
  sorry

end quadratic_sum_l1842_184257


namespace salad_quantity_percentage_difference_l1842_184238

noncomputable def Tom_rate := 2/3 -- Tom's rate (lb/min)
noncomputable def Tammy_rate := 3/2 -- Tammy's rate (lb/min)
noncomputable def Total_salad := 65 -- Total salad chopped (lb)
noncomputable def Time_to_chop := Total_salad / (Tom_rate + Tammy_rate) -- Time to chop 65 lb (min)
noncomputable def Tom_chop := Time_to_chop * Tom_rate -- Total chopped by Tom (lb)
noncomputable def Tammy_chop := Time_to_chop * Tammy_rate -- Total chopped by Tammy (lb)
noncomputable def Percent_difference := (Tammy_chop - Tom_chop) / Tom_chop * 100 -- Percent difference

theorem salad_quantity_percentage_difference : Percent_difference = 125 :=
by
  sorry

end salad_quantity_percentage_difference_l1842_184238


namespace Ed_cats_l1842_184211

variable (C F : ℕ)

theorem Ed_cats 
  (h1 : F = 2 * (C + 2))
  (h2 : 2 + C + F = 15) : 
  C = 3 := by 
  sorry

end Ed_cats_l1842_184211


namespace man_finishes_work_in_100_days_l1842_184291

variable (M W : ℝ)
variable (H1 : 10 * M * 6 + 15 * W * 6 = 1)
variable (H2 : W * 225 = 1)

theorem man_finishes_work_in_100_days (M W : ℝ) (H1 : 10 * M * 6 + 15 * W * 6 = 1) (H2 : W * 225 = 1) : M = 1 / 100 :=
by
  sorry

end man_finishes_work_in_100_days_l1842_184291


namespace lower_limit_of_range_l1842_184220

theorem lower_limit_of_range (A : Set ℕ) (range_A : ℕ) (h1 : ∀ n ∈ A, Prime n∧ n ≤ 36) (h2 : range_A = 14)
  (h3 : ∃ x, x ∈ A ∧ ¬(∃ y, y ∈ A ∧ y > x)) (h4 : ∃ x, x ∈ A ∧ x = 31): 
  ∃ m, m ∈ A ∧ m = 17 := 
sorry

end lower_limit_of_range_l1842_184220


namespace omitted_digits_correct_l1842_184255

theorem omitted_digits_correct :
  (287 * 23 = 6601) := by
  sorry

end omitted_digits_correct_l1842_184255


namespace time_to_cook_rest_of_potatoes_l1842_184241

-- Definitions of the conditions
def total_potatoes : ℕ := 12
def already_cooked : ℕ := 6
def minutes_per_potato : ℕ := 6

-- Proof statement
theorem time_to_cook_rest_of_potatoes : (total_potatoes - already_cooked) * minutes_per_potato = 36 :=
by
  sorry

end time_to_cook_rest_of_potatoes_l1842_184241


namespace find_y_l1842_184249

theorem find_y (x y : ℚ) (h1 : x = 153) (h2 : x^3 * y - 4 * x^2 * y + 4 * x * y = 350064) : 
  y = 40 / 3967 :=
by
  -- Proof to be filled in
  sorry

end find_y_l1842_184249


namespace larger_square_area_total_smaller_squares_area_l1842_184250
noncomputable def largerSquareSideLengthFromCircleRadius (r : ℝ) : ℝ :=
  2 * (2 * r)

noncomputable def squareArea (side : ℝ) : ℝ :=
  side * side

theorem larger_square_area (r : ℝ) (h : r = 3) :
  squareArea (largerSquareSideLengthFromCircleRadius r) = 144 :=
by
  sorry

theorem total_smaller_squares_area (r : ℝ) (h : r = 3) :
  4 * squareArea (2 * r) = 144 :=
by
  sorry

end larger_square_area_total_smaller_squares_area_l1842_184250


namespace man_present_age_l1842_184267

variable {P : ℝ}

theorem man_present_age (h1 : P = 1.25 * (P - 10)) (h2 : P = (5 / 6) * (P + 10)) : P = 50 :=
  sorry

end man_present_age_l1842_184267


namespace multiple_optimal_solutions_for_z_l1842_184221

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A := Point.mk 0 2
def B := Point.mk (-2) (-2)
def C := Point.mk 2 0

def z (a : ℝ) (P : Point) : ℝ := P.y - a * P.x

def maxz_mult_opt_solutions (a : ℝ) : Prop :=
  z a A = z a B ∨ z a A = z a C ∨ z a B = z a C

theorem multiple_optimal_solutions_for_z :
  (maxz_mult_opt_solutions (-1)) ∧ (maxz_mult_opt_solutions 2) :=
by
  sorry

end multiple_optimal_solutions_for_z_l1842_184221


namespace de_morgan_birth_year_jenkins_birth_year_l1842_184203

open Nat

theorem de_morgan_birth_year
  (x : ℕ) (hx : x = 43) (hx_square : x * x = 1849) :
  1849 - 43 = 1806 :=
by
  sorry

theorem jenkins_birth_year
  (a b : ℕ) (ha : a = 5) (hb : b = 6) (m : ℕ) (hm : m = 31) (n : ℕ) (hn : n = 5)
  (ha_sq : a * a = 25) (hb_sq : b * b = 36) (ha4 : a * a * a * a = 625)
  (hb4 : b * b * b * b = 1296) (hm2 : m * m = 961) (hn4 : n * n * n * n = 625) :
  1921 - 61 = 1860 ∧
  1922 - 62 = 1860 ∧
  1875 - 15 = 1860 :=
by
  sorry

end de_morgan_birth_year_jenkins_birth_year_l1842_184203


namespace roots_of_equation_l1842_184271

theorem roots_of_equation (x : ℝ) : (x - 3) ^ 2 = 4 ↔ (x = 5 ∨ x = 1) := by
  sorry

end roots_of_equation_l1842_184271


namespace cadence_worked_longer_by_5_months_l1842_184207

-- Definitions
def months_old_company : ℕ := 36

def salary_old_company : ℕ := 5000

def salary_new_company : ℕ := 6000

def total_earnings : ℕ := 426000

-- Prove that Cadence worked 5 months longer at her new company
theorem cadence_worked_longer_by_5_months :
  ∃ x : ℕ, 
  total_earnings = salary_old_company * months_old_company + 
                  salary_new_company * (months_old_company + x)
  ∧ x = 5 :=
by {
  sorry
}

end cadence_worked_longer_by_5_months_l1842_184207


namespace hydrogen_burns_oxygen_certain_l1842_184202

-- define what it means for a chemical reaction to be well-documented and known to occur
def chemical_reaction (reactants : String) (products : String) : Prop :=
  (reactants = "2H₂ + O₂") ∧ (products = "2H₂O")

-- Event description and classification
def event_is_certain (event : String) : Prop :=
  event = "Hydrogen burns in oxygen to form water"

-- Main statement
theorem hydrogen_burns_oxygen_certain :
  ∀ (reactants products : String), (chemical_reaction reactants products) → event_is_certain "Hydrogen burns in oxygen to form water" :=
by
  intros reactants products h
  have h1 : reactants = "2H₂ + O₂" := h.1
  have h2 : products = "2H₂O" := h.2
  -- proof omitted
  exact sorry

end hydrogen_burns_oxygen_certain_l1842_184202


namespace rectangle_diagonal_length_l1842_184233

theorem rectangle_diagonal_length
  (a b : ℝ)
  (h1 : a = 40 * Real.sqrt 2)
  (h2 : b = 2 * a) :
  Real.sqrt (a^2 + b^2) = 160 := by
  sorry

end rectangle_diagonal_length_l1842_184233


namespace free_throws_count_l1842_184284

-- Definitions based on the conditions
variables (a b x : ℕ) -- Number of 2-point shots, 3-point shots, and free throws respectively.

-- Condition: Points from two-point shots equal the points from three-point shots
def points_eq : Prop := 2 * a = 3 * b

-- Condition: Number of free throws is twice the number of two-point shots
def free_throws_eq : Prop := x = 2 * a

-- Condition: Total score is adjusted to 78 points
def total_score : Prop := 2 * a + 3 * b + x = 78

-- Proof problem statement
theorem free_throws_count (h1 : points_eq a b) (h2 : free_throws_eq a x) (h3 : total_score a b x) : x = 26 :=
sorry

end free_throws_count_l1842_184284


namespace distinct_sequences_count_l1842_184204

-- Define the set of available letters excluding 'M' for start and 'S' for end
def available_letters : List Char := ['A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C']

-- Define the cardinality function for the sequences under given specific conditions.
-- This will check specific prompt format; you may want to specify permutations, combinations based on calculations but in the spirit, we are sticking to detail.
def count_sequences (letters : List Char) (n : Nat) : Nat :=
  if letters = available_letters ∧ n = 5 then 
    -- based on detailed calculation in the solution
    480
  else
    0

-- Theorem statement in Lean 4 to verify the number of sequences
theorem distinct_sequences_count : count_sequences available_letters 5 = 480 := 
sorry

end distinct_sequences_count_l1842_184204


namespace number_of_correct_calculations_is_one_l1842_184234

/- Given conditions -/
def cond1 (a : ℝ) : Prop := a^2 * a^2 = 2 * a^2
def cond2 (a b : ℝ) : Prop := (a - b)^2 = a^2 - b^2
def cond3 (a : ℝ) : Prop := a^2 + a^3 = a^5
def cond4 (a b : ℝ) : Prop := (-2 * a^2 * b^3)^3 = -6 * a^6 * b^3
def cond5 (a : ℝ) : Prop := (-a^3)^2 / a = a^5

/- Statement to prove the number of correct calculations is 1 -/
theorem number_of_correct_calculations_is_one :
  (¬ (cond1 a)) ∧ (¬ (cond2 a b)) ∧ (¬ (cond3 a)) ∧ (¬ (cond4 a b)) ∧ (cond5 a) → 1 = 1 :=
by
  sorry

end number_of_correct_calculations_is_one_l1842_184234


namespace distance_between_centers_eq_l1842_184215

theorem distance_between_centers_eq (r1 r2 : ℝ) : ∃ d : ℝ, (d = r1 * Real.sqrt 2) := by
  sorry

end distance_between_centers_eq_l1842_184215


namespace simplify_sqrt_product_l1842_184223

theorem simplify_sqrt_product (y : ℝ) (hy : y > 0) : 
  (Real.sqrt (45 * y) * Real.sqrt (20 * y) * Real.sqrt (30 * y) = 30 * y * Real.sqrt (30 * y)) :=
by
  sorry

end simplify_sqrt_product_l1842_184223


namespace range_of_a_l1842_184235

noncomputable def f (a x : ℝ) : ℝ := x^2 + a * Real.log x - a * x

theorem range_of_a (a : ℝ) (h : a > 0) : 
  (∀ x : ℝ, 0 < x → 0 ≤ 2 * x^2 - a * x + a) ↔ 0 < a ∧ a ≤ 8 :=
by
  sorry

end range_of_a_l1842_184235


namespace num_teachers_l1842_184292

-- This statement involves defining the given conditions and stating the theorem to be proved.
theorem num_teachers (parents students total_people : ℕ) (h_parents : parents = 73) (h_students : students = 724) (h_total : total_people = 1541) :
  total_people - (parents + students) = 744 :=
by
  -- Including sorry to skip the proof, as required.
  sorry

end num_teachers_l1842_184292


namespace minimum_ab_ge_four_l1842_184226

variable (a b : ℝ)
variables (ha : 0 < a) (hb : 0 < b)
variable (h : 1 / a + 4 / b = Real.sqrt (a * b))

theorem minimum_ab_ge_four : a * b ≥ 4 := by
  sorry

end minimum_ab_ge_four_l1842_184226


namespace cups_of_oil_used_l1842_184253

-- Define the required amounts
def total_liquid : ℝ := 1.33
def water_used : ℝ := 1.17

-- The statement we want to prove
theorem cups_of_oil_used : total_liquid - water_used = 0.16 := by
sorry

end cups_of_oil_used_l1842_184253


namespace average_weight_of_all_players_l1842_184264

-- Definitions based on conditions
def num_forwards : ℕ := 8
def avg_weight_forwards : ℝ := 75
def num_defensemen : ℕ := 12
def avg_weight_defensemen : ℝ := 82

-- Total number of players
def total_players : ℕ := num_forwards + num_defensemen

-- Values derived from conditions
def total_weight_forwards : ℝ := avg_weight_forwards * num_forwards
def total_weight_defensemen : ℝ := avg_weight_defensemen * num_defensemen
def total_weight : ℝ := total_weight_forwards + total_weight_defensemen

-- Theorem to prove the average weight of all players
theorem average_weight_of_all_players : total_weight / total_players = 79.2 :=
by
  sorry

end average_weight_of_all_players_l1842_184264


namespace quadratic_inequality_l1842_184227

theorem quadratic_inequality (a x : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) := 
sorry

end quadratic_inequality_l1842_184227


namespace seashells_total_l1842_184293

theorem seashells_total :
    let Sam := 35
    let Joan := 18
    let Alex := 27
    Sam + Joan + Alex = 80 :=
by
    sorry

end seashells_total_l1842_184293


namespace total_books_to_put_away_l1842_184263

-- Definitions based on the conditions
def books_per_shelf := 4
def shelves_needed := 3

-- The proof problem translates to finding the total number of books
theorem total_books_to_put_away : shelves_needed * books_per_shelf = 12 := by
  sorry

end total_books_to_put_away_l1842_184263


namespace petya_wins_with_optimal_play_l1842_184298

theorem petya_wins_with_optimal_play :
  ∃ (n m : ℕ), n = 2000 ∧ m = (n * (n - 1)) / 2 ∧
  (∀ (v_cut : ℕ), ∀ (p_cut : ℕ), v_cut = 1 ∧ (p_cut = 2 ∨ p_cut = 3) ∧
  ((∃ k, m - v_cut = 4 * k) → ∃ k, m - v_cut - p_cut = 4 * k + 1) → 
  ∃ k, m - p_cut = 4 * k + 3) :=
sorry

end petya_wins_with_optimal_play_l1842_184298


namespace form_a_set_l1842_184210

def is_definitive (description: String) : Prop :=
  match description with
  | "comparatively small numbers" => False
  | "non-negative even numbers not greater than 10" => True
  | "all triangles" => True
  | "points in the Cartesian coordinate plane with an x-coordinate of zero" => True
  | "tall male students" => False
  | "students under 17 years old in a certain class" => True
  | _ => False

theorem form_a_set :
  is_definitive "comparatively small numbers" = False ∧
  is_definitive "non-negative even numbers not greater than 10" = True ∧
  is_definitive "all triangles" = True ∧
  is_definitive "points in the Cartesian coordinate plane with an x-coordinate of zero" = True ∧
  is_definitive "tall male students" = False ∧
  is_definitive "students under 17 years old in a certain class" = True :=
by
  repeat { split };
  exact sorry

end form_a_set_l1842_184210


namespace power_sum_tenth_l1842_184244

theorem power_sum_tenth (a b : ℝ) (h1 : a + b = 1)
    (h2 : a^2 + b^2 = 3)
    (h3 : a^3 + b^3 = 4)
    (h4 : a^4 + b^4 = 7)
    (h5 : a^5 + b^5 = 11) : 
    a^10 + b^10 = 123 := 
sorry

end power_sum_tenth_l1842_184244


namespace original_book_pages_l1842_184237

theorem original_book_pages (n k : ℕ) (h1 : (n * (n + 1)) / 2 - (2 * k + 1) = 4979)
: n = 100 :=
by
  sorry

end original_book_pages_l1842_184237


namespace minute_hand_length_l1842_184224

theorem minute_hand_length (r : ℝ) (h : 20 * (2 * Real.pi / 60) * r = Real.pi / 3) : r = 1 / 2 :=
by
  sorry

end minute_hand_length_l1842_184224


namespace nth_odd_and_sum_first_n_odds_l1842_184296

noncomputable def nth_odd (n : ℕ) : ℕ := 2 * n - 1

noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n ^ 2

theorem nth_odd_and_sum_first_n_odds :
  nth_odd 100 = 199 ∧ sum_first_n_odds 100 = 10000 :=
by
  sorry

end nth_odd_and_sum_first_n_odds_l1842_184296


namespace jellybean_probability_l1842_184248

/-- Abe holds 1 blue and 2 red jelly beans. 
    Bob holds 2 blue, 2 yellow, and 1 red jelly bean. 
    Each randomly picks a jelly bean to show the other. 
    What is the probability that the colors match? 
-/
theorem jellybean_probability :
  let abe_blue_prob := 1 / 3
  let bob_blue_prob := 2 / 5
  let abe_red_prob := 2 / 3
  let bob_red_prob := 1 / 5
  (abe_blue_prob * bob_blue_prob + abe_red_prob * bob_red_prob) = 4 / 15 :=
by
  sorry

end jellybean_probability_l1842_184248


namespace rectangle_area_l1842_184278

theorem rectangle_area (d : ℝ) (w : ℝ) (h : w^2 + (3*w)^2 = d^2) : (3 * w ^ 2 = 3 * d ^ 2 / 10) :=
by
  sorry

end rectangle_area_l1842_184278
