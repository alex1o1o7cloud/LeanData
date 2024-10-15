import Mathlib

namespace NUMINAMATH_GPT_triangle_PQR_PR_value_l843_84364

theorem triangle_PQR_PR_value (PQ QR PR : ℕ) (h1 : PQ = 7) (h2 : QR = 20) (h3 : 13 < PR) (h4 : PR < 27) : PR = 21 :=
by sorry

end NUMINAMATH_GPT_triangle_PQR_PR_value_l843_84364


namespace NUMINAMATH_GPT_hyperbola_equation_l843_84324

theorem hyperbola_equation:
  let F1 := (-Real.sqrt 10, 0)
  let F2 := (Real.sqrt 10, 0)
  ∃ P : ℝ × ℝ, 
    (let PF1 := (P.1 - F1.1, P.2 - F1.2);
     let PF2 := (P.1 - F2.1, P.2 - F2.2);
     (PF1.1 * PF2.1 + PF1.2 * PF2.2 = 0) ∧ 
     ((Real.sqrt (PF1.1^2 + PF1.2^2)) * (Real.sqrt (PF2.1^2 + PF2.2^2)) = 2)) →
    (∃ a b : ℝ, (a^2 = 9 ∧ b^2 = 1) ∧ 
                (∀ x y : ℝ, 
                 (a ≠ 0 ∧ (x^2 / a^2) - (y^2 / b^2) = 1 ↔ 
                  ∃ P : ℝ × ℝ, 
                    let PF1 := (P.1 - F1.1, P.2 - F1.2);
                    let PF2 := (P.1 - F2.1, P.2 - F2.2);
                    PF1.1 * PF2.1 + PF1.2 * PF2.2 = 0 ∧ 
                    (Real.sqrt (PF1.1^2 + PF1.2^2)) * (Real.sqrt (PF2.1^2 + PF2.2^2)) = 2)))
:= by
sorry

end NUMINAMATH_GPT_hyperbola_equation_l843_84324


namespace NUMINAMATH_GPT_maria_bottles_proof_l843_84362

theorem maria_bottles_proof 
    (initial_bottles : ℕ)
    (drank_bottles : ℕ)
    (current_bottles : ℕ)
    (bought_bottles : ℕ) 
    (h1 : initial_bottles = 14)
    (h2 : drank_bottles = 8)
    (h3 : current_bottles = 51)
    (h4 : current_bottles = initial_bottles - drank_bottles + bought_bottles) :
  bought_bottles = 45 :=
by
  sorry

end NUMINAMATH_GPT_maria_bottles_proof_l843_84362


namespace NUMINAMATH_GPT_total_cost_alex_had_to_pay_l843_84377

def baseCost : ℝ := 30
def costPerText : ℝ := 0.04 -- 4 cents in dollars
def textsSent : ℕ := 150
def costPerMinuteOverLimit : ℝ := 0.15 -- 15 cents in dollars
def hoursUsed : ℝ := 26
def freeHours : ℝ := 25

def totalCost : ℝ :=
  baseCost + (costPerText * textsSent) + (costPerMinuteOverLimit * (hoursUsed - freeHours) * 60)

theorem total_cost_alex_had_to_pay :
  totalCost = 45 := by
  sorry

end NUMINAMATH_GPT_total_cost_alex_had_to_pay_l843_84377


namespace NUMINAMATH_GPT_circle_radius_l843_84354

theorem circle_radius : 
  ∀ (x y : ℝ), x^2 + y^2 + 12 = 10 * x - 6 * y → ∃ r : ℝ, r = Real.sqrt 22 :=
by
  intros x y h
  -- Additional steps to complete the proof will be added here
  sorry

end NUMINAMATH_GPT_circle_radius_l843_84354


namespace NUMINAMATH_GPT_goals_per_player_is_30_l843_84390

-- Define the total number of goals scored in the league against Barca
def total_goals : ℕ := 300

-- Define the percentage of goals scored by the two players
def percentage_of_goals : ℝ := 0.20

-- Define the combined goals by the two players
def combined_goals := (percentage_of_goals * total_goals : ℝ)

-- Define the number of players
def number_of_players : ℕ := 2

-- Define the number of goals scored by each player
noncomputable def goals_per_player := combined_goals / number_of_players

-- Proof statement: Each of the two players scored 30 goals.
theorem goals_per_player_is_30 :
  goals_per_player = 30 :=
sorry

end NUMINAMATH_GPT_goals_per_player_is_30_l843_84390


namespace NUMINAMATH_GPT_surface_area_of_rectangular_solid_l843_84394

-- Conditions
variables {a b c : ℕ}
variables (h_a_prime : Nat.Prime a) (h_b_prime : Nat.Prime b) (h_c_prime : Nat.Prime c)
variables (h_volume : a * b * c = 308)

-- Question and Proof Problem
theorem surface_area_of_rectangular_solid :
  2 * (a * b + b * c + c * a) = 226 :=
sorry

end NUMINAMATH_GPT_surface_area_of_rectangular_solid_l843_84394


namespace NUMINAMATH_GPT_discount_problem_l843_84308

theorem discount_problem (m : ℝ) (h : (200 * (1 - m / 100)^2 = 162)) : m = 10 :=
sorry

end NUMINAMATH_GPT_discount_problem_l843_84308


namespace NUMINAMATH_GPT_park_cycling_time_l843_84397

def length_breadth_ratio (L B : ℕ) : Prop := L / B = 1 / 3
def area_of_park (L B : ℕ) : Prop := L * B = 120000
def speed_of_cyclist : ℕ := 200 -- meters per minute
def perimeter (L B : ℕ) : ℕ := 2 * L + 2 * B
def time_to_complete_round (P v : ℕ) : ℕ := P / v

theorem park_cycling_time
  (L B : ℕ)
  (h_ratio : length_breadth_ratio L B)
  (h_area : area_of_park L B)
  : time_to_complete_round (perimeter L B) speed_of_cyclist = 8 :=
by
  sorry

end NUMINAMATH_GPT_park_cycling_time_l843_84397


namespace NUMINAMATH_GPT_trapezoid_area_l843_84343

theorem trapezoid_area (a b H : ℝ) (h_lat1 : a = 10) (h_lat2 : b = 8) (h_height : H = b) : 
∃ S : ℝ, S = 104 :=
by sorry

end NUMINAMATH_GPT_trapezoid_area_l843_84343


namespace NUMINAMATH_GPT_function_at_neg_one_zero_l843_84379

-- Define the function f with the given conditions
variable {f : ℝ → ℝ}

-- Declare the conditions as hypotheses
def domain_condition : ∀ x : ℝ, true := by sorry
def non_zero_condition : ∃ x : ℝ, f x ≠ 0 := by sorry
def even_function_condition : ∀ x : ℝ, f (x + 2) = f (2 - x) := by sorry
def odd_function_condition : ∀ x : ℝ, f (1 - 2 * x) = -f (2 * x + 1) := by sorry

-- The main theorem to be proved
theorem function_at_neg_one_zero :
  f (-1) = 0 :=
by
  -- Use the conditions to derive the result
  sorry

end NUMINAMATH_GPT_function_at_neg_one_zero_l843_84379


namespace NUMINAMATH_GPT_boat_travel_l843_84384

theorem boat_travel (T_against T_with : ℝ) (V_b D V_c : ℝ) 
  (hT_against : T_against = 10) 
  (hT_with : T_with = 6) 
  (hV_b : V_b = 12)
  (hD1 : D = (V_b - V_c) * T_against)
  (hD2 : D = (V_b + V_c) * T_with) :
  V_c = 3 ∧ D = 90 :=
by
  sorry

end NUMINAMATH_GPT_boat_travel_l843_84384


namespace NUMINAMATH_GPT_problem_inequality_l843_84386

theorem problem_inequality (n : ℕ) (x : ℝ) (hn : n ≥ 2) (hx : |x| < 1) :
  2^n > (1 - x)^n + (1 + x)^n :=
sorry

end NUMINAMATH_GPT_problem_inequality_l843_84386


namespace NUMINAMATH_GPT_num_factors_of_60_l843_84352

-- Definition of 60 in terms of its prime factors
def n : ℕ := 60
def a : ℕ := 2
def b : ℕ := 1
def c : ℕ := 1

-- Statement for the number of positive factors
theorem num_factors_of_60 :
  (a + 1) * (b + 1) * (c + 1) = 12 :=
by 
  -- We are skipping the proof part by using sorry.
  sorry

end NUMINAMATH_GPT_num_factors_of_60_l843_84352


namespace NUMINAMATH_GPT_expression_positive_l843_84398

theorem expression_positive (x y z : ℝ) (h : x^2 + y^2 + z^2 ≠ 0) : 
  5 * x^2 + 5 * y^2 + 5 * z^2 + 6 * x * y - 8 * x * z - 8 * y * z > 0 := 
sorry

end NUMINAMATH_GPT_expression_positive_l843_84398


namespace NUMINAMATH_GPT_birds_landing_l843_84374

theorem birds_landing (initial_birds total_birds birds_landed : ℤ) 
  (h_initial : initial_birds = 12) 
  (h_total : total_birds = 20) :
  birds_landed = total_birds - initial_birds :=
by
  sorry

end NUMINAMATH_GPT_birds_landing_l843_84374


namespace NUMINAMATH_GPT_books_on_shelf_l843_84333

-- Step definitions based on the conditions
def initial_books := 38
def marta_books_removed := 10
def tom_books_removed := 5
def tom_books_added := 12

-- Final number of books on the shelf
def final_books : ℕ := initial_books - marta_books_removed - tom_books_removed + tom_books_added

-- Theorem statement to prove the final number of books
theorem books_on_shelf : final_books = 35 :=
by 
  -- Proof for the statement goes here
  sorry

end NUMINAMATH_GPT_books_on_shelf_l843_84333


namespace NUMINAMATH_GPT_thieves_cloth_equation_l843_84365

theorem thieves_cloth_equation (x y : ℤ) 
  (h1 : y = 6 * x + 5)
  (h2 : y = 7 * x - 8) :
  6 * x + 5 = 7 * x - 8 :=
by
  sorry

end NUMINAMATH_GPT_thieves_cloth_equation_l843_84365


namespace NUMINAMATH_GPT_JungMinBoughtWire_l843_84311

theorem JungMinBoughtWire
  (side_length : ℕ)
  (number_of_sides : ℕ)
  (remaining_wire : ℕ)
  (total_wire_bought : ℕ)
  (h1 : side_length = 13)
  (h2 : number_of_sides = 5)
  (h3 : remaining_wire = 8)
  (h4 : total_wire_bought = side_length * number_of_sides + remaining_wire) :
    total_wire_bought = 73 :=
by {
  sorry
}

end NUMINAMATH_GPT_JungMinBoughtWire_l843_84311


namespace NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l843_84341

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, (n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ (∀ m : ℕ, (m % 17 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≥ m)) ∧ n = 986 :=
by
  sorry

end NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l843_84341


namespace NUMINAMATH_GPT_volume_of_water_cylinder_l843_84310

theorem volume_of_water_cylinder :
  let r := 5
  let h := 10
  let depth := 3
  let θ := Real.arccos (3 / 5)
  let sector_area := (2 * θ) / (2 * Real.pi) * Real.pi * r^2
  let triangle_area := r * (2 * r * Real.sin θ)
  let water_surface_area := sector_area - triangle_area
  let volume := h * water_surface_area
  volume = 232.6 * Real.pi - 160 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_water_cylinder_l843_84310


namespace NUMINAMATH_GPT_smallest_positive_n_l843_84358

theorem smallest_positive_n
  (a x y : ℤ)
  (h1 : x ≡ a [ZMOD 9])
  (h2 : y ≡ -a [ZMOD 9]) :
  ∃ n : ℕ, n > 0 ∧ (x^2 + x * y + y^2 + n) % 9 = 0 ∧ n = 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_n_l843_84358


namespace NUMINAMATH_GPT_correct_relationship_5_25_l843_84348

theorem correct_relationship_5_25 : 5^2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_correct_relationship_5_25_l843_84348


namespace NUMINAMATH_GPT_max_value_of_expression_l843_84319

theorem max_value_of_expression 
  (a b c : ℝ)
  (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  6 * a + 3 * b + 10 * c ≤ 3.2 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l843_84319


namespace NUMINAMATH_GPT_arithmetic_seq_sum_ratio_l843_84376

theorem arithmetic_seq_sum_ratio
  (a : ℕ → ℤ)
  (S : ℕ → ℤ)
  (h1 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h2 : S 25 / a 23 = 5)
  (h3 : S 45 / a 33 = 25) :
  S 65 / a 43 = 45 :=
by sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_ratio_l843_84376


namespace NUMINAMATH_GPT_symmetrical_point_of_P_is_correct_l843_84350

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the function to get the symmetric point with respect to the origin
def symmetrical_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Prove that the symmetrical point of P with respect to the origin is (1, -2)
theorem symmetrical_point_of_P_is_correct : symmetrical_point P = (1, -2) :=
  sorry

end NUMINAMATH_GPT_symmetrical_point_of_P_is_correct_l843_84350


namespace NUMINAMATH_GPT_weight_of_gravel_l843_84334

theorem weight_of_gravel (total_weight : ℝ) (weight_sand : ℝ) (weight_water : ℝ) (weight_gravel : ℝ) 
  (h1 : total_weight = 48)
  (h2 : weight_sand = (1/3) * total_weight)
  (h3 : weight_water = (1/2) * total_weight)
  (h4 : weight_gravel = total_weight - (weight_sand + weight_water)) :
  weight_gravel = 8 :=
sorry

end NUMINAMATH_GPT_weight_of_gravel_l843_84334


namespace NUMINAMATH_GPT_find_reflection_line_l843_84366

-- Definition of the original and reflected vertices
structure Point :=
  (x : ℝ)
  (y : ℝ)

def D : Point := {x := 1, y := 2}
def E : Point := {x := 6, y := 7}
def F : Point := {x := -5, y := 5}
def D' : Point := {x := 1, y := -4}
def E' : Point := {x := 6, y := -9}
def F' : Point := {x := -5, y := -7}

theorem find_reflection_line (M : ℝ) :
  (D.y + D'.y) / 2 = M ∧ (E.y + E'.y) / 2 = M ∧ (F.y + F'.y) / 2 = M → M = -1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_reflection_line_l843_84366


namespace NUMINAMATH_GPT_rojas_speed_l843_84361

theorem rojas_speed (P R : ℝ) (h1 : P = 3) (h2 : 4 * (R + P) = 28) : R = 4 :=
by
  sorry

end NUMINAMATH_GPT_rojas_speed_l843_84361


namespace NUMINAMATH_GPT_sequence_property_l843_84338

theorem sequence_property (a : ℕ+ → ℤ) (h_add : ∀ p q : ℕ+, a (p + q) = a p + a q) (h_a2 : a 2 = -6) :
  a 10 = -30 := 
sorry

end NUMINAMATH_GPT_sequence_property_l843_84338


namespace NUMINAMATH_GPT_xiao_liang_correct_l843_84347

theorem xiao_liang_correct :
  ∀ (x : ℕ), (0 ≤ x ∧ x ≤ 26 ∧ 30 - x ≤ 24 ∧ 26 - x ≤ 20) →
  let boys_A := x
  let girls_A := 30 - x
  let boys_B := 26 - x
  let girls_B := 24 - girls_A
  ∃ k : ℤ, boys_A - girls_B = 6 := 
by 
  sorry

end NUMINAMATH_GPT_xiao_liang_correct_l843_84347


namespace NUMINAMATH_GPT_find_possible_f_one_l843_84339

noncomputable def f : ℝ → ℝ := sorry

theorem find_possible_f_one (h : ∀ x y : ℝ, f (x + y) = 2 * f x * f y) :
  f 1 = 0 ∨ (∃ c : ℝ, f 0 = 1/2 ∧ f 1 = c) :=
sorry

end NUMINAMATH_GPT_find_possible_f_one_l843_84339


namespace NUMINAMATH_GPT_roots_of_quadratic_l843_84359

theorem roots_of_quadratic (a b c : ℝ) (h : ¬ (a = 0 ∧ b = 0 ∧ c = 0)) :
  ¬ ∃ (x : ℝ), x^2 + (a + b + c) * x + a^2 + b^2 + c^2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_l843_84359


namespace NUMINAMATH_GPT_social_studies_score_l843_84363

-- Step d): Translate to Lean 4
theorem social_studies_score 
  (K E S SS : ℝ)
  (h1 : (K + E + S) / 3 = 89)
  (h2 : (K + E + S + SS) / 4 = 90) :
  SS = 93 :=
by
  -- We'll leave the mathematics formal proof details to Lean.
  sorry

end NUMINAMATH_GPT_social_studies_score_l843_84363


namespace NUMINAMATH_GPT_determine_a_for_nonnegative_function_l843_84382

def function_positive_on_interval (a : ℝ) : Prop :=
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → a * x^3 - 3 * x + 1 ≥ 0

theorem determine_a_for_nonnegative_function :
  ∀ (a : ℝ), function_positive_on_interval a ↔ a = 4 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_for_nonnegative_function_l843_84382


namespace NUMINAMATH_GPT_distance_between_foci_l843_84368

theorem distance_between_foci (x y : ℝ)
    (h : 2 * x^2 - 12 * x - 8 * y^2 + 16 * y = 100) :
    2 * Real.sqrt 68.75 =
    2 * Real.sqrt (55 + 13.75) :=
by
  sorry

end NUMINAMATH_GPT_distance_between_foci_l843_84368


namespace NUMINAMATH_GPT_simplify_frac_and_find_cd_l843_84336

theorem simplify_frac_and_find_cd :
  ∀ (m : ℤ), ∃ (c d : ℤ), 
    (c * m + d = (6 * m + 12) / 3) ∧ (c = 2) ∧ (d = 4) ∧ (c / d = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_simplify_frac_and_find_cd_l843_84336


namespace NUMINAMATH_GPT_examination_students_total_l843_84345

/-
  Problem Statement:
  Given:
  - 35% of the students passed the examination.
  - 546 students failed the examination.

  Prove:
  - The total number of students who appeared for the examination is 840.
-/

theorem examination_students_total (T : ℝ) (h1 : 0.35 * T + 0.65 * T = T) (h2 : 0.65 * T = 546) : T = 840 :=
by
  -- skipped proof part
  sorry

end NUMINAMATH_GPT_examination_students_total_l843_84345


namespace NUMINAMATH_GPT_domain_ln_2_minus_x_is_interval_l843_84332

noncomputable def domain_ln_2_minus_x : Set Real := { x : Real | 2 - x > 0 }

theorem domain_ln_2_minus_x_is_interval : domain_ln_2_minus_x = Set.Iio 2 :=
by
  sorry

end NUMINAMATH_GPT_domain_ln_2_minus_x_is_interval_l843_84332


namespace NUMINAMATH_GPT_mushroom_collection_l843_84391

variable (a b v g : ℕ)

theorem mushroom_collection : 
  (a / 2 + 2 * b = v + g) ∧ (a + b = v / 2 + 2 * g) → (v = 2 * b) ∧ (a = 2 * g) :=
by
  sorry

end NUMINAMATH_GPT_mushroom_collection_l843_84391


namespace NUMINAMATH_GPT_tail_growth_problem_l843_84342

def initial_tail_length : ℕ := 1
def final_tail_length : ℕ := 864
def transformations (ordinary_count cowardly_count : ℕ) : ℕ := initial_tail_length * 2^ordinary_count * 3^cowardly_count

theorem tail_growth_problem (ordinary_count cowardly_count : ℕ) :
  transformations ordinary_count cowardly_count = final_tail_length ↔ ordinary_count = 5 ∧ cowardly_count = 3 :=
by
  sorry

end NUMINAMATH_GPT_tail_growth_problem_l843_84342


namespace NUMINAMATH_GPT_x_cubed_plus_y_cubed_l843_84349

theorem x_cubed_plus_y_cubed (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 14) : x^3 + y^3 = 176 :=
sorry

end NUMINAMATH_GPT_x_cubed_plus_y_cubed_l843_84349


namespace NUMINAMATH_GPT_divisor_of_1076_plus_least_addend_l843_84375

theorem divisor_of_1076_plus_least_addend (a d : ℕ) (h1 : 1076 + a = 1081) (h2 : d ∣ 1081) (ha : a = 5) : d = 13 := 
sorry

end NUMINAMATH_GPT_divisor_of_1076_plus_least_addend_l843_84375


namespace NUMINAMATH_GPT_evaluate_expression_l843_84303

theorem evaluate_expression (a : ℝ) (h : a = 2) : 
    (a / (a^2 - 1) - 1 / (a^2 - 1)) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l843_84303


namespace NUMINAMATH_GPT_sum_of_roots_cubic_l843_84331

theorem sum_of_roots_cubic :
  let a := 3
  let b := 7
  let c := -12
  let d := -4
  let roots_sum := -(b / a)
  roots_sum = -2.33 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_cubic_l843_84331


namespace NUMINAMATH_GPT_original_price_hat_l843_84322

theorem original_price_hat 
  (x : ℝ)
  (discounted_price := x / 5)
  (final_price := discounted_price * 1.2)
  (h : final_price = 8) :
  x = 100 / 3 :=
by
  sorry

end NUMINAMATH_GPT_original_price_hat_l843_84322


namespace NUMINAMATH_GPT_carrie_pants_l843_84312

theorem carrie_pants (P : ℕ) (shirts := 4) (pants := P) (jackets := 2)
  (shirt_cost := 8) (pant_cost := 18) (jacket_cost := 60)
  (total_cost := shirts * shirt_cost + jackets * jacket_cost + pants * pant_cost)
  (total_cost_half := 94) :
  total_cost = 188 → total_cost_half = 94 → total_cost = 2 * total_cost_half → P = 2 :=
by
  intros h_total h_half h_relation
  sorry

end NUMINAMATH_GPT_carrie_pants_l843_84312


namespace NUMINAMATH_GPT_geom_seq_a1_l843_84370

-- Define a geometric sequence.
def geom_seq (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 * q ^ n

-- Given conditions
def a2 (a : ℕ → ℝ) : Prop := a 1 = 2 -- because a2 = a(1) in zero-indexed
def a5 (a : ℕ → ℝ) : Prop := a 4 = -54 -- because a5 = a(4) in zero-indexed

-- Prove that a1 = -2/3
theorem geom_seq_a1 (a : ℕ → ℝ) (a1 q : ℝ) (h_geom : geom_seq a a1 q)
  (h_a2 : a2 a) (h_a5 : a5 a) : a1 = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_a1_l843_84370


namespace NUMINAMATH_GPT_function_satisfies_equation_l843_84318

theorem function_satisfies_equation (y : ℝ → ℝ) (h : ∀ x : ℝ, y x = Real.exp (x + x^2) + 2 * Real.exp x) :
  ∀ x : ℝ, deriv y x - y x = 2 * x * Real.exp (x + x^2) :=
by {
  sorry
}

end NUMINAMATH_GPT_function_satisfies_equation_l843_84318


namespace NUMINAMATH_GPT_male_athletes_sampled_l843_84396

-- Define the total number of athletes
def total_athletes : Nat := 98

-- Define the number of female athletes
def female_athletes : Nat := 42

-- Define the probability of being selected
def selection_probability : ℚ := 2 / 7

-- Calculate the number of male athletes
def male_athletes : Nat := total_athletes - female_athletes

-- State the theorem about the number of male athletes sampled
theorem male_athletes_sampled : male_athletes * selection_probability = 16 :=
by
  sorry

end NUMINAMATH_GPT_male_athletes_sampled_l843_84396


namespace NUMINAMATH_GPT_pieces_per_pan_of_brownies_l843_84353

theorem pieces_per_pan_of_brownies (total_guests guests_ala_mode additional_guests total_scoops_per_tub total_tubs_eaten total_pans guests_per_pan second_pan_percentage consumed_pans : ℝ)
    (h1 : total_guests = guests_ala_mode + additional_guests)
    (h2 : total_scoops_per_tub * total_tubs_eaten = guests_ala_mode * 2)
    (h3 : consumed_pans = 1 + second_pan_percentage)
    (h4 : second_pan_percentage = 0.75)
    (h5 : total_guests = guests_per_pan * consumed_pans)
    (h6 : guests_per_pan = 28)
    : total_guests / consumed_pans = 16 :=
by
  have h7 : total_scoops_per_tub * total_tubs_eaten = 48 := by sorry
  have h8 : guests_ala_mode = 24 := by sorry
  have h9 : total_guests = 28 := by sorry
  have h10 : consumed_pans = 1.75 := by sorry
  have h11 : guests_per_pan = 28 := by sorry
  sorry


end NUMINAMATH_GPT_pieces_per_pan_of_brownies_l843_84353


namespace NUMINAMATH_GPT_conference_games_l843_84329

theorem conference_games (teams_per_division : ℕ) (divisions : ℕ) 
  (intradivision_games_per_team : ℕ) (interdivision_games_per_team : ℕ) 
  (total_teams : ℕ) (total_games : ℕ) : 
  total_teams = teams_per_division * divisions →
  intradivision_games_per_team = (teams_per_division - 1) * 2 →
  interdivision_games_per_team = teams_per_division →
  total_games = (total_teams * (intradivision_games_per_team + interdivision_games_per_team)) / 2 →
  total_games = 133 :=
by
  intros
  sorry

end NUMINAMATH_GPT_conference_games_l843_84329


namespace NUMINAMATH_GPT_magician_earnings_l843_84399

noncomputable def total_earnings (price_per_deck : ℕ) (initial_decks : ℕ) (end_decks : ℕ) (promotion_price : ℕ) (exchange_rate_start : ℚ) (exchange_rate_mid : ℚ) (foreign_sales_1 : ℕ) (domestic_sales : ℕ) (foreign_sales_2 : ℕ) : ℕ :=
  let foreign_earnings_1 := (foreign_sales_1 / 2) * promotion_price
  let foreign_earnings_2 := foreign_sales_2 * price_per_deck
  (domestic_sales / 2) * promotion_price + foreign_earnings_1 + foreign_earnings_2
  

-- Given conditions:
-- price_per_deck = 2
-- initial_decks = 5
-- end_decks = 3
-- promotion_price = 3
-- exchange_rate_start = 1
-- exchange_rate_mid = 1.5
-- foreign_sales_1 = 4
-- domestic_sales = 2
-- foreign_sales_2 = 1

theorem magician_earnings :
  total_earnings 2 5 3 3 1 1.5 4 2 1 = 11 :=
by
   sorry

end NUMINAMATH_GPT_magician_earnings_l843_84399


namespace NUMINAMATH_GPT_projected_increase_is_25_l843_84335

variable (R P : ℝ) -- variables for last year's revenue and projected increase in percentage

-- Conditions
axiom h1 : ∀ (R : ℝ), R > 0
axiom h2 : ∀ (P : ℝ), P/100 ≥ 0
axiom h3 : ∀ (R : ℝ), 0.75 * R = 0.60 * (R + (P/100) * R)

-- Goal
theorem projected_increase_is_25 (R : ℝ) : P = 25 :=
by {
    -- import the required axioms and provide the necessary proof
    apply sorry
}

end NUMINAMATH_GPT_projected_increase_is_25_l843_84335


namespace NUMINAMATH_GPT_prob1_prob2_l843_84301

variables (x y a b c : ℝ)

-- Proof for the first problem
theorem prob1 :
  3 * x^2 * (-3 * x * y)^2 - x^2 * (x^2 * y^2 - 2 * x) = 26 * x^4 * y^2 + 2 * x^3 := 
sorry

-- Proof for the second problem
theorem prob2 :
  -2 * (-a^2 * b * c)^2 * (1 / 2) * a * (b * c)^3 - (-a * b * c)^3 * (-a * b * c)^2 = 0 :=
sorry

end NUMINAMATH_GPT_prob1_prob2_l843_84301


namespace NUMINAMATH_GPT_union_of_A_and_B_l843_84315

def set_A : Set Int := {0, 1}
def set_B : Set Int := {0, -1}

theorem union_of_A_and_B : set_A ∪ set_B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l843_84315


namespace NUMINAMATH_GPT_operations_correctness_l843_84387

theorem operations_correctness (a b : ℝ) : 
  ((-ab)^2 ≠ -a^2 * b^2)
  ∧ (a^3 * a^2 ≠ a^6)
  ∧ ((a^3)^4 ≠ a^7)
  ∧ (b^2 + b^2 = 2 * b^2) :=
by
  sorry

end NUMINAMATH_GPT_operations_correctness_l843_84387


namespace NUMINAMATH_GPT_animal_shelter_kittens_count_l843_84385

def num_puppies : ℕ := 32
def num_kittens_more : ℕ := 14

theorem animal_shelter_kittens_count : 
  ∃ k : ℕ, k = (2 * num_puppies) + num_kittens_more := 
sorry

end NUMINAMATH_GPT_animal_shelter_kittens_count_l843_84385


namespace NUMINAMATH_GPT_minimum_value_of_expression_l843_84360

theorem minimum_value_of_expression (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
    ∃ (c : ℝ), (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → x^3 + y^3 - 5 * x * y ≥ c) ∧ c = -125 / 27 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l843_84360


namespace NUMINAMATH_GPT_triangle_side_lengths_l843_84355

theorem triangle_side_lengths (A B C : ℝ) (a b c : ℝ) 
  (hcosA : Real.cos A = 1/4)
  (ha : a = 4)
  (hbc_sum : b + c = 6)
  (hbc_order : b < c) :
  b = 2 ∧ c = 4 := by
  sorry

end NUMINAMATH_GPT_triangle_side_lengths_l843_84355


namespace NUMINAMATH_GPT_part1_part2_l843_84309

theorem part1 (a b : ℝ) (h1 : ∀ x, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) (hb : b > 1) : a = 1 ∧ b = 2 :=
sorry

theorem part2 (k : ℝ) (x y : ℝ) (hx : x > 0) (hy : y > 0) (a b : ℝ) 
  (ha : a = 1) (hb : b = 2) 
  (h2 : a / x + b / y = 1)
  (h3 : 2 * x + y ≥ k^2 + k + 2) : -3 ≤ k ∧ k ≤ 2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l843_84309


namespace NUMINAMATH_GPT_second_group_num_persons_l843_84327

def man_hours (num_persons : ℕ) (days : ℕ) (hours_per_day : ℕ) : ℕ :=
  num_persons * days * hours_per_day

theorem second_group_num_persons :
  ∀ (x : ℕ),
    let first_group_man_hours := man_hours 36 12 5
    let second_group_days := 12
    let second_group_hours_per_day := 6
    (first_group_man_hours = man_hours x second_group_days second_group_hours_per_day) →
    x = 30 :=
by
  intros x first_group_man_hours second_group_days second_group_hours_per_day h
  sorry

end NUMINAMATH_GPT_second_group_num_persons_l843_84327


namespace NUMINAMATH_GPT_initial_markup_percentage_l843_84369

theorem initial_markup_percentage (C M : ℝ) 
  (h1 : C > 0) 
  (h2 : (1 + M) * 1.25 * 0.92 = 1.38) :
  M = 0.2 :=
sorry

end NUMINAMATH_GPT_initial_markup_percentage_l843_84369


namespace NUMINAMATH_GPT_b_arithmetic_sequence_max_S_n_l843_84393

variables {a : ℕ → ℝ} {b : ℕ → ℝ} {S : ℕ → ℝ}

-- Conditions
noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m, a m ≠ 0 → a n = a (n + 1) * a (m-1) / (a m)

axiom a_pos_terms : ∀ n, 0 < a n
axiom a11_eight : a 11 = 8
axiom b_log : ∀ n, b n = Real.log (a n) / Real.log 2
axiom b4_seventeen : b 4 = 17

-- Question I: Prove b_n is an arithmetic sequence with common difference -2
theorem b_arithmetic_sequence (d : ℝ) (h_d : d = (-2)) :
  ∃ d, ∀ n, b (n + 1) - b n = d :=
sorry

-- Question II: Find the maximum value of S_n
theorem max_S_n : ∃ n, S n = 144 :=
sorry

end NUMINAMATH_GPT_b_arithmetic_sequence_max_S_n_l843_84393


namespace NUMINAMATH_GPT_running_time_15mph_l843_84305

theorem running_time_15mph (x y z : ℝ) (h1 : x + y + z = 14) (h2 : 15 * x + 10 * y + 8 * z = 164) :
  x = 3 :=
sorry

end NUMINAMATH_GPT_running_time_15mph_l843_84305


namespace NUMINAMATH_GPT_Andrews_age_l843_84351

theorem Andrews_age (a g : ℝ) (h1 : g = 15 * a) (h2 : g - a = 55) : a = 55 / 14 :=
by
  /- proof will go here -/
  sorry

end NUMINAMATH_GPT_Andrews_age_l843_84351


namespace NUMINAMATH_GPT_number_of_three_digit_integers_l843_84323

-- Defining the set of available digits
def digits : List ℕ := [3, 5, 8, 9]

-- Defining the property for selecting a digit without repetition
def no_repetition (l : List ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ l → l.filter (fun x => x = d) = [d]

-- The main theorem stating the number of three-digit integers that can be formed
theorem number_of_three_digit_integers (h : no_repetition digits) : 
  ∃ n : ℕ, n = 24 :=
by
  sorry

end NUMINAMATH_GPT_number_of_three_digit_integers_l843_84323


namespace NUMINAMATH_GPT_quadratic_smaller_solution_l843_84326

theorem quadratic_smaller_solution : ∀ (x : ℝ), x^2 - 9 * x + 20 = 0 → x = 4 ∨ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_smaller_solution_l843_84326


namespace NUMINAMATH_GPT_find_number_l843_84388

def number_equal_when_divided_by_3_and_subtracted : Prop :=
  ∃ x : ℝ, (x / 3 = x - 3) ∧ (x = 4.5)

theorem find_number (x : ℝ) : (x / 3 = x - 3) → x = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l843_84388


namespace NUMINAMATH_GPT_books_returned_wednesday_correct_l843_84357

def initial_books : Nat := 250
def books_taken_out_Tuesday : Nat := 120
def books_taken_out_Thursday : Nat := 15
def books_remaining_after_Thursday : Nat := 150

def books_after_tuesday := initial_books - books_taken_out_Tuesday
def books_before_thursday := books_remaining_after_Thursday + books_taken_out_Thursday
def books_returned_wednesday := books_before_thursday - books_after_tuesday

theorem books_returned_wednesday_correct : books_returned_wednesday = 35 := by
  sorry

end NUMINAMATH_GPT_books_returned_wednesday_correct_l843_84357


namespace NUMINAMATH_GPT_players_count_l843_84317

theorem players_count (total_socks : ℕ) (socks_per_player : ℕ) (h1 : total_socks = 16) (h2 : socks_per_player = 2) :
  total_socks / socks_per_player = 8 :=
by
  sorry

end NUMINAMATH_GPT_players_count_l843_84317


namespace NUMINAMATH_GPT_maximum_cookies_by_andy_l843_84314

-- Define the conditions
def total_cookies := 36
def cookies_by_andry (a : ℕ) := a
def cookies_by_alexa (a : ℕ) := 3 * a
def cookies_by_alice (a : ℕ) := 2 * a
def sum_cookies (a : ℕ) := cookies_by_andry a + cookies_by_alexa a + cookies_by_alice a

-- The theorem stating the problem and solution
theorem maximum_cookies_by_andy :
  ∃ a : ℕ, sum_cookies a = total_cookies ∧ a = 6 :=
by
  sorry

end NUMINAMATH_GPT_maximum_cookies_by_andy_l843_84314


namespace NUMINAMATH_GPT_number_of_rectangles_is_24_l843_84330

-- Define the rectangles on a 1x5 stripe
def rectangles_1x5 : ℕ := 1 + 2 + 3 + 4 + 5

-- Define the rectangles on a 1x4 stripe
def rectangles_1x4 : ℕ := 1 + 2 + 3 + 4

-- Define the overlap (intersection) adjustment
def overlap_adjustment : ℕ := 1

-- Total number of rectangles calculation
def total_rectangles : ℕ := rectangles_1x5 + rectangles_1x4 - overlap_adjustment

theorem number_of_rectangles_is_24 : total_rectangles = 24 := by
  sorry

end NUMINAMATH_GPT_number_of_rectangles_is_24_l843_84330


namespace NUMINAMATH_GPT_problem_statement_l843_84392

theorem problem_statement (a b : ℝ) (h : 3 * a - 2 * b = -1) : 3 * a - 2 * b + 2024 = 2023 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l843_84392


namespace NUMINAMATH_GPT_min_max_values_l843_84313

theorem min_max_values (x1 x2 x3 : ℝ) (h1 : x1 ≥ 0) (h2 : x2 ≥ 0) (h3 : x3 ≥ 0) (h_sum : x1 + x2 + x3 = 1) :
  1 ≤ (x1 + 3*x2 + 5*x3) * (x1 + x2 / 3 + x3 / 5) ∧ (x1 + 3*x2 + 5*x3) * (x1 + x2 / 3 + x3 / 5) ≤ 9/5 :=
by sorry

end NUMINAMATH_GPT_min_max_values_l843_84313


namespace NUMINAMATH_GPT_quadrant_of_half_angle_in_second_quadrant_l843_84367

theorem quadrant_of_half_angle_in_second_quadrant (θ : ℝ) (h : π / 2 < θ ∧ θ < π) :
  (0 < θ / 2 ∧ θ / 2 < π / 2) ∨ (π < θ / 2 ∧ θ / 2 < 3 * π / 2) :=
by
  sorry

end NUMINAMATH_GPT_quadrant_of_half_angle_in_second_quadrant_l843_84367


namespace NUMINAMATH_GPT_fixed_point_of_transformed_logarithmic_function_l843_84389

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

noncomputable def f_a (a : ℝ) (x : ℝ) : ℝ := 1 + log_a a (x - 1)

theorem fixed_point_of_transformed_logarithmic_function
  (a : ℝ) (ha : 0 < a ∧ a ≠ 1) : f_a a 2 = 1 :=
by
  -- Prove the theorem using given conditions
  sorry

end NUMINAMATH_GPT_fixed_point_of_transformed_logarithmic_function_l843_84389


namespace NUMINAMATH_GPT_obtuse_angle_probability_l843_84337

-- Defining the vertices of the pentagon
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨0, 3⟩
def B : Point := ⟨5, 0⟩
def C : Point := ⟨8, 0⟩
def D : Point := ⟨8, 5⟩
def E : Point := ⟨0, 5⟩

def is_interior (P : Point) : Prop :=
  -- A condition to define if a point is inside the pentagon
  sorry

def is_obtuse_angle (A B P : Point) : Prop :=
  -- Condition for angle APB to be obtuse
  sorry

noncomputable def probability_obtuse_angle :=
  -- Probability calculation
  let area_pentagon := 40
  let area_circle := (34 * Real.pi) / 4
  let area_outside_circle := area_pentagon - area_circle
  area_outside_circle / area_pentagon

theorem obtuse_angle_probability :
  ∀ P : Point, is_interior P → ∃! p : ℝ, p = (160 - 34 * Real.pi) / 160 :=
sorry

end NUMINAMATH_GPT_obtuse_angle_probability_l843_84337


namespace NUMINAMATH_GPT_geometric_sequence_problem_l843_84378

noncomputable def geometric_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 1 then n * a1 else a1 * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_problem
  (a1 q : ℝ) (a2 : ℝ := a1 * q) (a5 : ℝ := a1 * q^4)
  (S2 : ℝ := geometric_sum a1 q 2) (S4 : ℝ := geometric_sum a1 q 4)
  (h1 : 8 * a2 + a5 = 0) :
  S4 / S2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l843_84378


namespace NUMINAMATH_GPT_perimeter_of_square_l843_84373

/-- The perimeter of a square with side length 15 cm is 60 cm -/
theorem perimeter_of_square (side_length : ℝ) (area : ℝ) (h1 : side_length = 15) (h2 : area = 225) :
  (4 * side_length = 60) :=
by
  -- Proof steps would go here (omitted)
  sorry

end NUMINAMATH_GPT_perimeter_of_square_l843_84373


namespace NUMINAMATH_GPT_x_squared_plus_y_squared_l843_84383

theorem x_squared_plus_y_squared (x y : ℝ) (h₁ : x - y = 18) (h₂ : x * y = 9) : x^2 + y^2 = 342 := by
  sorry

end NUMINAMATH_GPT_x_squared_plus_y_squared_l843_84383


namespace NUMINAMATH_GPT_who_wears_which_dress_l843_84381

-- Define the possible girls
inductive Girl
| Katya | Olya | Liza | Rita
deriving DecidableEq

-- Define the possible dresses
inductive Dress
| Pink | Green | Yellow | Blue
deriving DecidableEq

-- Define the fact that each girl is wearing a dress
structure Wearing (girl : Girl) (dress : Dress) : Prop

-- Define the conditions
theorem who_wears_which_dress :
  (¬ Wearing Girl.Katya Dress.Pink ∧ ¬ Wearing Girl.Katya Dress.Blue) ∧
  (∀ g1 g2 g3, Wearing g1 Dress.Green → (Wearing g2 Dress.Pink ∧ Wearing g3 Dress.Yellow → (g2 = Girl.Liza ∧ (g3 = Girl.Rita)) ∨ (g3 = Girl.Liza ∧ g2 = Girl.Rita))) ∧
  (¬ Wearing Girl.Rita Dress.Green ∧ ¬ Wearing Girl.Rita Dress.Blue) ∧
  (∀ g1 g2, (Wearing g1 Dress.Pink ∧ Wearing g2 Dress.Yellow) → Girl.Olya = g2 ∧ Girl.Rita = g1) →
  (Wearing Girl.Katya Dress.Green ∧ Wearing Girl.Olya Dress.Blue ∧ Wearing Girl.Liza Dress.Pink ∧ Wearing Girl.Rita Dress.Yellow) :=
by
  sorry

end NUMINAMATH_GPT_who_wears_which_dress_l843_84381


namespace NUMINAMATH_GPT_consequence_of_implication_l843_84304

-- Define the conditions
variable (A B : Prop)

-- State the theorem to prove
theorem consequence_of_implication (h : B → A) : A → B := 
  sorry

end NUMINAMATH_GPT_consequence_of_implication_l843_84304


namespace NUMINAMATH_GPT_min_workers_for_profit_l843_84300

def revenue (n : ℕ) : ℕ := 240 * n
def cost (n : ℕ) : ℕ := 600 + 200 * n

theorem min_workers_for_profit (n : ℕ) (h : 240 * n > 600 + 200 * n) : n >= 16 :=
by {
  -- Placeholder for the proof steps (which are not required per instructions)
  sorry
}

end NUMINAMATH_GPT_min_workers_for_profit_l843_84300


namespace NUMINAMATH_GPT_angle_between_a_and_b_is_2pi_over_3_l843_84306

open Real

variables (a b c : ℝ × ℝ)

-- Given conditions
def condition1 := a.1^2 + a.2^2 = 2  -- |a| = sqrt(2)
def condition2 := b = (-1, 1)        -- b = (-1, 1)
def condition3 := c = (2, -2)        -- c = (2, -2)
def condition4 := a.1 * (b.1 + c.1) + a.2 * (b.2 + c.2) = 1  -- a · (b + c) = 1

-- Prove the angle θ between a and b is 2π/3
theorem angle_between_a_and_b_is_2pi_over_3 :
  condition1 a → condition2 b → condition3 c → condition4 a b c →
  ∃ θ, 0 ≤ θ ∧ θ ≤ π ∧ cos θ = -(1/2) ∧ θ = 2 * π / 3 :=
by
  sorry

end NUMINAMATH_GPT_angle_between_a_and_b_is_2pi_over_3_l843_84306


namespace NUMINAMATH_GPT_function_symmetry_extremum_l843_84356

noncomputable def f (x θ : ℝ) : ℝ := 3 * Real.cos (Real.pi * x + θ)

theorem function_symmetry_extremum {θ : ℝ} (H : ∀ x : ℝ, f x θ = f (2 - x) θ) : 
  f 1 θ = 3 ∨ f 1 θ = -3 :=
by
  sorry

end NUMINAMATH_GPT_function_symmetry_extremum_l843_84356


namespace NUMINAMATH_GPT_fraction_of_janes_age_is_five_eighths_l843_84372

/-- Jane's current age -/
def jane_current_age : ℕ := 34

/-- Number of years ago Jane stopped babysitting -/
def years_since_stopped_babysitting : ℕ := 10

/-- Current age of the oldest child Jane could have babysat -/
def oldest_child_current_age : ℕ := 25

/-- Calculate Jane's age when she stopped babysitting -/
def jane_age_when_stopped_babysitting : ℕ := jane_current_age - years_since_stopped_babysitting

/-- Calculate the child's age when Jane stopped babysitting -/
def oldest_child_age_when_jane_stopped : ℕ := oldest_child_current_age - years_since_stopped_babysitting 

/-- Calculate the fraction of Jane's age that the child could be at most -/
def babysitting_age_fraction : ℚ := (oldest_child_age_when_jane_stopped : ℚ) / (jane_age_when_stopped_babysitting : ℚ)

theorem fraction_of_janes_age_is_five_eighths :
  babysitting_age_fraction = 5 / 8 :=
by 
  -- Declare the proof steps (this part is the placeholder as proof is not required)
  sorry

end NUMINAMATH_GPT_fraction_of_janes_age_is_five_eighths_l843_84372


namespace NUMINAMATH_GPT_total_spent_on_computer_l843_84302

def initial_cost_of_pc : ℕ := 1200
def sale_price_old_card : ℕ := 300
def cost_new_card : ℕ := 500

theorem total_spent_on_computer : 
  (initial_cost_of_pc + (cost_new_card - sale_price_old_card)) = 1400 :=
by
  sorry

end NUMINAMATH_GPT_total_spent_on_computer_l843_84302


namespace NUMINAMATH_GPT_raft_drift_time_l843_84340

theorem raft_drift_time (s : ℝ) (v_down v_up v_c : ℝ) 
  (h1 : v_down = s / 3) 
  (h2 : v_up = s / 4) 
  (h3 : v_down = v_c + v_c)
  (h4 : v_up = v_c - v_c) :
  v_c = s / 24 → (s / v_c) = 24 := 
by
  sorry

end NUMINAMATH_GPT_raft_drift_time_l843_84340


namespace NUMINAMATH_GPT_point_c_in_second_quadrant_l843_84316

-- Definitions for the points
def PointA : ℝ × ℝ := (1, 2)
def PointB : ℝ × ℝ := (-1, -2)
def PointC : ℝ × ℝ := (-1, 2)
def PointD : ℝ × ℝ := (1, -2)

-- Definition of the second quadrant condition
def in_second_quadrant (p : ℝ × ℝ) : Prop :=
p.1 < 0 ∧ p.2 > 0

theorem point_c_in_second_quadrant : in_second_quadrant PointC :=
sorry

end NUMINAMATH_GPT_point_c_in_second_quadrant_l843_84316


namespace NUMINAMATH_GPT_isosceles_triangle_vertex_angle_l843_84344

theorem isosceles_triangle_vertex_angle (θ : ℝ) (h₀ : θ = 80) (h₁ : ∃ (x y z : ℝ), (x = y ∨ y = z ∨ z = x) ∧ x + y + z = 180) : θ = 80 ∨ θ = 20 := 
sorry

end NUMINAMATH_GPT_isosceles_triangle_vertex_angle_l843_84344


namespace NUMINAMATH_GPT_maximum_volume_prism_l843_84371

-- Define the conditions
variables {l w h : ℝ}
axiom area_sum_eq : 2 * h * l + l * w = 30

-- Define the volume of the prism
def volume (l w h : ℝ) : ℝ := l * w * h

-- Statement to be proved
theorem maximum_volume_prism : 
  (∃ l w h : ℝ, 2 * h * l + l * w = 30 ∧ 
  ∀ u v t : ℝ, 2 * t * u + u * v = 30 → l * w * h ≥ u * v * t) → volume l w h = 112.5 :=
by
  sorry

end NUMINAMATH_GPT_maximum_volume_prism_l843_84371


namespace NUMINAMATH_GPT_algebraic_expression_value_l843_84328

theorem algebraic_expression_value (m : ℝ) (h : m^2 - m - 2 = 0) : 2 * m^2 - 2 * m = 4 := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l843_84328


namespace NUMINAMATH_GPT_initial_deadline_l843_84307

theorem initial_deadline (D : ℝ) :
  (∀ (n : ℝ), (10 * 20) / 4 = n / 1) → 
  (∀ (m : ℝ), 8 * 75 = m * 3) →
  (∀ (d1 d2 : ℝ), d1 = 20 ∧ d2 = 93.75 → D = d1 + d2) →
  D = 113.75 :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_deadline_l843_84307


namespace NUMINAMATH_GPT_total_money_received_a_l843_84346

-- Define the partners and their capitals
structure Partner :=
  (name : String)
  (capital : ℕ)
  (isWorking : Bool)

def a : Partner := { name := "a", capital := 3500, isWorking := true }
def b : Partner := { name := "b", capital := 2500, isWorking := false }

-- Define the total profit
def totalProfit : ℕ := 9600

-- Define the managing fee as 10% of total profit
def managingFee (total : ℕ) : ℕ := (10 * total) / 100

-- Define the remaining profit after deducting the managing fee
def remainingProfit (total : ℕ) (fee : ℕ) : ℕ := total - fee

-- Calculate the share of remaining profit based on capital contribution
def share (capital totalCapital remaining : ℕ) : ℕ := (capital * remaining) / totalCapital

-- Theorem to prove the total money received by partner a
theorem total_money_received_a :
  let totalCapitals := a.capital + b.capital
  let fee := managingFee totalProfit
  let remaining := remainingProfit totalProfit fee
  let aShare := share a.capital totalCapitals remaining
  (fee + aShare) = 6000 :=
by
  sorry

end NUMINAMATH_GPT_total_money_received_a_l843_84346


namespace NUMINAMATH_GPT_gcd_360_504_l843_84395

theorem gcd_360_504 : Int.gcd 360 504 = 72 := by
  sorry

end NUMINAMATH_GPT_gcd_360_504_l843_84395


namespace NUMINAMATH_GPT_trigonometric_expression_value_l843_84320

theorem trigonometric_expression_value :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 4 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_expression_value_l843_84320


namespace NUMINAMATH_GPT_trapezoid_third_largest_angle_l843_84380

theorem trapezoid_third_largest_angle (a d : ℝ)
  (h1 : 2 * a + 3 * d = 200)      -- Condition: 2a + 3d = 200°
  (h2 : a + d = 70) :             -- Condition: a + d = 70°
  a + 2 * d = 130 :=              -- Question: Prove a + 2d = 130°
by
  sorry

end NUMINAMATH_GPT_trapezoid_third_largest_angle_l843_84380


namespace NUMINAMATH_GPT_center_of_large_hexagon_within_small_hexagon_l843_84321

-- Define a structure for a regular hexagon with the necessary properties
structure RegularHexagon (α : Type) [LinearOrderedField α] :=
  (center : α × α)      -- Coordinates of the center
  (side_length : α)      -- Length of the side

-- Define the conditions: two regular hexagons with specific side length relationship
variables {α : Type} [LinearOrderedField α]
def hexagon_large : RegularHexagon α := 
  {center := (0, 0), side_length := 2}

def hexagon_small : RegularHexagon α := 
  {center := (0, 0), side_length := 1}

-- The theorem to prove
theorem center_of_large_hexagon_within_small_hexagon (hl : RegularHexagon α) (hs : RegularHexagon α) 
  (hc : hs.side_length = hl.side_length / 2) : (hl.center = hs.center) → 
  (∀ (x y : α × α), x = hs.center → (∃ r, y = hl.center → (y.1 - x.1) ^ 2 + (y.2 - x.2) ^ 2 < r ^ 2)) :=
by sorry

end NUMINAMATH_GPT_center_of_large_hexagon_within_small_hexagon_l843_84321


namespace NUMINAMATH_GPT_remainder_3_pow_405_mod_13_l843_84325

theorem remainder_3_pow_405_mod_13 : (3^405) % 13 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_3_pow_405_mod_13_l843_84325
