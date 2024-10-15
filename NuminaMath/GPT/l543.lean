import Mathlib

namespace NUMINAMATH_GPT_units_digit_of_fraction_l543_54350

theorem units_digit_of_fraction :
  let numer := 30 * 31 * 32 * 33 * 34 * 35
  let denom := 1000
  (numer / denom) % 10 = 6 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_fraction_l543_54350


namespace NUMINAMATH_GPT_compute_d_for_ellipse_l543_54379

theorem compute_d_for_ellipse
  (in_first_quadrant : true)
  (is_tangent_x_axis : true)
  (is_tangent_y_axis : true)
  (focus1 : (ℝ × ℝ) := (5, 4))
  (focus2 : (ℝ × ℝ) := (d, 4)) :
  d = 3.2 := by
  sorry

end NUMINAMATH_GPT_compute_d_for_ellipse_l543_54379


namespace NUMINAMATH_GPT_max_rect_area_l543_54326

theorem max_rect_area (l w : ℤ) (h1 : 2 * l + 2 * w = 40) (h2 : 0 < l) (h3 : 0 < w) : 
  l * w ≤ 100 :=
by sorry

end NUMINAMATH_GPT_max_rect_area_l543_54326


namespace NUMINAMATH_GPT_time_equal_l543_54341

noncomputable def S : ℝ := sorry 
noncomputable def S_flat : ℝ := S
noncomputable def S_uphill : ℝ := (1 / 3) * S
noncomputable def S_downhill : ℝ := (2 / 3) * S
noncomputable def V_flat : ℝ := sorry 
noncomputable def V_uphill : ℝ := (1 / 2) * V_flat
noncomputable def V_downhill : ℝ := 2 * V_flat
noncomputable def t_flat: ℝ := S / V_flat
noncomputable def t_uphill: ℝ := S_uphill / V_uphill
noncomputable def t_downhill: ℝ := S_downhill / V_downhill
noncomputable def t_hill: ℝ := t_uphill + t_downhill

theorem time_equal: t_flat = t_hill := 
  by sorry

end NUMINAMATH_GPT_time_equal_l543_54341


namespace NUMINAMATH_GPT_a1_lt_a3_iff_an_lt_an1_l543_54381

-- Define arithmetic sequence and required properties
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variables (a : ℕ → ℝ)

-- Define the necessary and sufficient condition theorem
theorem a1_lt_a3_iff_an_lt_an1 (h_arith : is_arithmetic_sequence a) :
  (a 1 < a 3) ↔ (∀ n : ℕ, a n < a (n + 1)) :=
sorry

end NUMINAMATH_GPT_a1_lt_a3_iff_an_lt_an1_l543_54381


namespace NUMINAMATH_GPT_sugar_cubes_left_l543_54358

theorem sugar_cubes_left (h w d : ℕ) (hd1 : w * d = 77) (hd2 : h * d = 55) :
  (h - 1) * w * (d - 1) = 300 ∨ (h - 1) * w * (d - 1) = 0 :=
by
  sorry

end NUMINAMATH_GPT_sugar_cubes_left_l543_54358


namespace NUMINAMATH_GPT_speed_of_current_l543_54301

-- Definitions
def downstream_speed (m current : ℝ) := m + current
def upstream_speed (m current : ℝ) := m - current

-- Theorem
theorem speed_of_current 
  (m : ℝ) (current : ℝ) 
  (h1 : downstream_speed m current = 20) 
  (h2 : upstream_speed m current = 14) : 
  current = 3 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_speed_of_current_l543_54301


namespace NUMINAMATH_GPT_yellow_more_than_green_by_l543_54390

-- Define the problem using the given conditions.
def weight_yellow_block : ℝ := 0.6
def weight_green_block  : ℝ := 0.4

-- State the theorem that the yellow block weighs 0.2 pounds more than the green block.
theorem yellow_more_than_green_by : weight_yellow_block - weight_green_block = 0.2 :=
by sorry

end NUMINAMATH_GPT_yellow_more_than_green_by_l543_54390


namespace NUMINAMATH_GPT_chess_tournament_proof_l543_54336

-- Define the conditions
variables (i g n I G : ℕ)
variables (VI VG VD : ℕ)

-- Condition 1: The number of GMs is ten times the number of IMs
def condition1 : Prop := g = 10 * i
  
-- Condition 2: The sum of the points of all GMs is 4.5 times the sum of the points of all IMs
def condition2 : Prop := G = 5 * I + I / 2

-- Condition 3: The total number of players is the sum of IMs and GMs
def condition3 : Prop := n = i + g

-- Condition 4: Each player played only once against all other opponents
def condition4 : Prop := n * (n - 1) = 2 * (VI + VG + VD)

-- Condition 5: The sum of the points of all games is 5.5 times the sum of the points of all IMs
def condition5 : Prop := I + G = 11 * I / 2

-- Condition 6: Total games played
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

-- The questions to be proven given the conditions
theorem chess_tournament_proof:
  condition1 i g →
  condition2 I G →
  condition3 i g n →
  condition4 n VI VG VD →
  condition5 I G →
  i = 1 ∧ g = 10 ∧ total_games n = 55 :=
by
  -- The proof is left as an exercise
  sorry

end NUMINAMATH_GPT_chess_tournament_proof_l543_54336


namespace NUMINAMATH_GPT_find_other_number_l543_54359

theorem find_other_number
  (a b : ℕ)
  (HCF : ℕ)
  (LCM : ℕ)
  (h1 : HCF = 12)
  (h2 : LCM = 396)
  (h3 : a = 36)
  (h4 : HCF * LCM = a * b) :
  b = 132 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l543_54359


namespace NUMINAMATH_GPT_find_AX_l543_54324

theorem find_AX (AB AC BC : ℝ) (CX_bisects_ACB : Prop) (h1 : AB = 50) (h2 : AC = 28) (h3 : BC = 56) : AX = 50 / 3 :=
by
  -- Proof can be added here
  sorry

end NUMINAMATH_GPT_find_AX_l543_54324


namespace NUMINAMATH_GPT_eliot_votes_l543_54340

theorem eliot_votes (randy_votes shaun_votes eliot_votes : ℕ)
                    (h1 : randy_votes = 16)
                    (h2 : shaun_votes = 5 * randy_votes)
                    (h3 : eliot_votes = 2 * shaun_votes) :
                    eliot_votes = 160 :=
by {
  -- Proof will be conducted here
  sorry
}

end NUMINAMATH_GPT_eliot_votes_l543_54340


namespace NUMINAMATH_GPT_PQ_length_l543_54371

theorem PQ_length (BC AD : ℝ) (angle_A angle_D : ℝ) (P Q : ℝ) 
  (H1 : BC = 700) (H2 : AD = 1400) (H3 : angle_A = 45) (H4 : angle_D = 45) 
  (mid_BC : P = BC / 2) (mid_AD : Q = AD / 2) :
  abs (Q - P) = 350 :=
by
  sorry

end NUMINAMATH_GPT_PQ_length_l543_54371


namespace NUMINAMATH_GPT_simplify_trig_expression_l543_54372

theorem simplify_trig_expression (A : ℝ) :
  (2 - (Real.cos A / Real.sin A) + (1 / Real.sin A)) * (3 - (Real.sin A / Real.cos A) - (1 / Real.cos A)) = 
  7 * Real.sin A * Real.cos A - 2 * Real.cos A ^ 2 - 3 * Real.sin A ^ 2 - 3 * Real.cos A + Real.sin A + 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_trig_expression_l543_54372


namespace NUMINAMATH_GPT_vanessa_scored_27_points_l543_54366

variable (P : ℕ) (number_of_players : ℕ) (average_points_per_player : ℚ) (vanessa_points : ℕ)

axiom team_total_points : P = 48
axiom other_players : number_of_players = 6
axiom average_points_per_other_player : average_points_per_player = 3.5

theorem vanessa_scored_27_points 
  (h1 : P = 48)
  (h2 : number_of_players = 6)
  (h3 : average_points_per_player = 3.5)
: vanessa_points = 27 :=
sorry

end NUMINAMATH_GPT_vanessa_scored_27_points_l543_54366


namespace NUMINAMATH_GPT_students_in_high_school_l543_54323

-- Definitions from conditions
def H (L: ℝ) : ℝ := 4 * L
def middleSchoolStudents : ℝ := 300
def combinedStudents (H: ℝ) (L: ℝ) : ℝ := H + L
def combinedIsSevenTimesMiddle (H: ℝ) (L: ℝ) : Prop := combinedStudents H L = 7 * middleSchoolStudents

-- The main goal to prove
theorem students_in_high_school (L H: ℝ) (h1: H = 4 * L) (h2: combinedIsSevenTimesMiddle H L) : H = 1680 := by
  sorry

end NUMINAMATH_GPT_students_in_high_school_l543_54323


namespace NUMINAMATH_GPT_find_hours_spent_l543_54327

/-- Let 
  h : ℝ := hours Ed stayed in the hotel last night
  morning_hours : ℝ := 4 -- hours Ed stayed in the hotel this morning
  
  conditions:
  night_cost_per_hour : ℝ := 1.50 -- the cost per hour for staying at night
  morning_cost_per_hour : ℝ := 2 -- the cost per hour for staying in the morning
  initial_amount : ℝ := 80 -- initial amount Ed had
  remaining_amount : ℝ := 63 -- remaining amount after stay
  
  Then the total cost calculated by Ed is:
  total_cost : ℝ := (night_cost_per_hour * h) + (morning_cost_per_hour * morning_hours)
  spent_amount : ℝ := initial_amount - remaining_amount

  We need to prove that h = 6 given the above conditions.
-/
theorem find_hours_spent {h morning_hours night_cost_per_hour morning_cost_per_hour initial_amount remaining_amount total_cost spent_amount : ℝ}
  (hc1 : night_cost_per_hour = 1.50)
  (hc2 : morning_cost_per_hour = 2)
  (hc3 : initial_amount = 80)
  (hc4 : remaining_amount = 63)
  (hc5 : morning_hours = 4)
  (hc6 : spent_amount = initial_amount - remaining_amount)
  (hc7 : total_cost = night_cost_per_hour * h + morning_cost_per_hour * morning_hours)
  (hc8 : spent_amount = 17)
  (hc9 : total_cost = spent_amount) :
  h = 6 :=
by 
  sorry

end NUMINAMATH_GPT_find_hours_spent_l543_54327


namespace NUMINAMATH_GPT_abc_sum_equals_9_l543_54316

theorem abc_sum_equals_9 (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a * b + c = 57) (h5 : b * c + a = 57) (h6 : a * c + b = 57) :
  a + b + c = 9 := 
sorry

end NUMINAMATH_GPT_abc_sum_equals_9_l543_54316


namespace NUMINAMATH_GPT_build_time_40_workers_l543_54367

theorem build_time_40_workers (r : ℝ) : 
  (60 * r) * 5 = 1 → (40 * r) * t = 1 → t = 7.5 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_build_time_40_workers_l543_54367


namespace NUMINAMATH_GPT_union_of_sets_l543_54351

def A := { x : ℝ | -1 ≤ x ∧ x < 3 }
def B := { x : ℝ | 2 < x ∧ x ≤ 5 }

theorem union_of_sets : A ∪ B = { x : ℝ | -1 ≤ x ∧ x ≤ 5 } := 
by sorry

end NUMINAMATH_GPT_union_of_sets_l543_54351


namespace NUMINAMATH_GPT_coverage_is_20_l543_54306

noncomputable def cost_per_kg : ℝ := 60
noncomputable def total_cost : ℝ := 1800
noncomputable def side_length : ℝ := 10

-- Surface area of one side of the cube
noncomputable def area_side : ℝ := side_length * side_length

-- Total surface area of the cube
noncomputable def total_area : ℝ := 6 * area_side

-- Kilograms of paint used
noncomputable def kg_paint_used : ℝ := total_cost / cost_per_kg

-- Coverage per kilogram of paint
noncomputable def coverage_per_kg (total_area : ℝ) (kg_paint_used : ℝ) : ℝ := total_area / kg_paint_used

theorem coverage_is_20 : coverage_per_kg total_area kg_paint_used = 20 := by
  sorry

end NUMINAMATH_GPT_coverage_is_20_l543_54306


namespace NUMINAMATH_GPT_max_right_angle_triangles_in_pyramid_l543_54337

noncomputable def pyramid_max_right_angle_triangles : Nat :=
  let pyramid : Type := { faces : Nat // faces = 4 }
  1

theorem max_right_angle_triangles_in_pyramid (p : pyramid) : pyramid_max_right_angle_triangles = 1 :=
  sorry

end NUMINAMATH_GPT_max_right_angle_triangles_in_pyramid_l543_54337


namespace NUMINAMATH_GPT_city_cleaning_total_l543_54342

variable (A B C D : ℕ)

theorem city_cleaning_total : 
  A = 54 →
  A = B + 17 →
  C = 2 * B →
  D = A / 3 →
  A + B + C + D = 183 := 
by 
  intros hA hAB hC hD
  sorry

end NUMINAMATH_GPT_city_cleaning_total_l543_54342


namespace NUMINAMATH_GPT_roots_of_quadratic_eq_l543_54330

theorem roots_of_quadratic_eq:
  (8 * γ^3 + 15 * δ^2 = 179) ↔ (γ^2 - 3 * γ + 1 = 0 ∧ δ^2 - 3 * δ + 1 = 0) :=
sorry

end NUMINAMATH_GPT_roots_of_quadratic_eq_l543_54330


namespace NUMINAMATH_GPT_dragon_jewels_l543_54377

theorem dragon_jewels (x : ℕ) (h1 : (x / 3 = 6)) : x + 6 = 24 :=
sorry

end NUMINAMATH_GPT_dragon_jewels_l543_54377


namespace NUMINAMATH_GPT_negation_of_p_l543_54355

theorem negation_of_p :
  ¬(∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔ ∃ x : ℝ, x > 0 ∧ (x + 1) * Real.exp x ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l543_54355


namespace NUMINAMATH_GPT_find_values_of_a2_b2_l543_54382

-- Define the conditions
variables {a b : ℝ}
variable (h1 : a > b)
variable (h2 : b > 0)
variable (hP : (-2, (Real.sqrt 14) / 2) ∈ { p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1 })
variable (hCircle : ∀ Q : ℝ × ℝ, (Q ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 2 }) → (∃ tA tB : ℝ × ℝ, (tA ∈ { p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1 }) ∧ (tB ∈ { p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1 }) ∧ (tA = - tB ∨ tB = - tA) ∧ ((tA.1 + tB.1)/2 = (-2 + tA.1)/2) ))

-- The theorem to be proven
theorem find_values_of_a2_b2 : a^2 + b^2 = 15 :=
sorry

end NUMINAMATH_GPT_find_values_of_a2_b2_l543_54382


namespace NUMINAMATH_GPT_train_distance_covered_l543_54304

-- Definitions based on the given conditions
def average_speed := 3   -- in meters per second
def total_time := 9      -- in seconds

-- Theorem statement: Given the average speed and total time, the total distance covered is 27 meters
theorem train_distance_covered : average_speed * total_time = 27 := 
by
  sorry

end NUMINAMATH_GPT_train_distance_covered_l543_54304


namespace NUMINAMATH_GPT_max_black_balls_C_is_22_l543_54398

-- Define the given parameters
noncomputable def balls_A : ℕ := 100
noncomputable def black_balls_A : ℕ := 15
noncomputable def balls_B : ℕ := 50
noncomputable def balls_C : ℕ := 80
noncomputable def probability : ℚ := 101 / 600

-- Define the maximum number of black balls in box C given the conditions
theorem max_black_balls_C_is_22 (y : ℕ) (h : (1/3 * (black_balls_A / balls_A) + 1/3 * (y / balls_B) + 1/3 * (22 / balls_C)) = probability  ) :
  ∃ (x : ℕ), x ≤ 22 := sorry

end NUMINAMATH_GPT_max_black_balls_C_is_22_l543_54398


namespace NUMINAMATH_GPT_maple_tree_taller_than_pine_tree_pine_tree_height_in_one_year_l543_54375

def pine_tree_height : ℚ := 37 / 4  -- 9 1/4 feet
def maple_tree_height : ℚ := 62 / 4  -- 15 1/2 feet (converted directly to common denominator)
def growth_rate : ℚ := 7 / 4  -- 1 3/4 feet per year

theorem maple_tree_taller_than_pine_tree : maple_tree_height - pine_tree_height = 25 / 4 := 
by sorry

theorem pine_tree_height_in_one_year : pine_tree_height + growth_rate = 44 / 4 := 
by sorry

end NUMINAMATH_GPT_maple_tree_taller_than_pine_tree_pine_tree_height_in_one_year_l543_54375


namespace NUMINAMATH_GPT_total_time_preparing_games_l543_54309

def time_A_game : ℕ := 15
def time_B_game : ℕ := 25
def time_C_game : ℕ := 30
def num_each_type : ℕ := 5

theorem total_time_preparing_games : 
  (num_each_type * time_A_game + num_each_type * time_B_game + num_each_type * time_C_game) = 350 := 
  by sorry

end NUMINAMATH_GPT_total_time_preparing_games_l543_54309


namespace NUMINAMATH_GPT_no_digit_satisfies_equations_l543_54343

-- Define the conditions as predicates.
def is_digit (x : ℤ) : Prop := 0 ≤ x ∧ x < 10

-- Formulate the proof problem based on the given problem conditions and conclusion
theorem no_digit_satisfies_equations : 
  ¬ (∃ x : ℤ, is_digit x ∧ (x - (10 * x + x) = 801 ∨ x - (10 * x + x) = 812)) :=
by
  sorry

end NUMINAMATH_GPT_no_digit_satisfies_equations_l543_54343


namespace NUMINAMATH_GPT_s_mores_graham_crackers_l543_54303

def graham_crackers_per_smore (total_graham_crackers total_marshmallows : ℕ) : ℕ :=
total_graham_crackers / total_marshmallows

theorem s_mores_graham_crackers :
  let total_graham_crackers := 48
  let available_marshmallows := 6
  let additional_marshmallows := 18
  let total_marshmallows := available_marshmallows + additional_marshmallows
  graham_crackers_per_smore total_graham_crackers total_marshallows = 2 := sorry

end NUMINAMATH_GPT_s_mores_graham_crackers_l543_54303


namespace NUMINAMATH_GPT_simplify_div_expression_l543_54353

theorem simplify_div_expression (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (x - 1) / (x^2 + 2 * x + 1) / (1 - 2 / (x + 1)) = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_GPT_simplify_div_expression_l543_54353


namespace NUMINAMATH_GPT_price_of_kid_ticket_l543_54360

theorem price_of_kid_ticket (k a : ℤ) (hk : k = 6) (ha : a = 2)
  (price_kid price_adult : ℤ)
  (hprice_adult : price_adult = 2 * price_kid)
  (hcost_total : 6 * price_kid + 2 * price_adult = 50) :
  price_kid = 5 :=
by
  sorry

end NUMINAMATH_GPT_price_of_kid_ticket_l543_54360


namespace NUMINAMATH_GPT_negB_sufficient_for_A_l543_54352

variables {A B : Prop}

theorem negB_sufficient_for_A (h : ¬A → B) (hnotsuff : ¬(B → ¬A)) : ¬ B → A :=
by
  sorry

end NUMINAMATH_GPT_negB_sufficient_for_A_l543_54352


namespace NUMINAMATH_GPT_find_y_l543_54317

theorem find_y (x y : ℝ) (h1 : x = 202) (h2 : x^3 * y - 4 * x^2 * y + 2 * x * y = 808080) : y = 1 / 10 := by
  sorry

end NUMINAMATH_GPT_find_y_l543_54317


namespace NUMINAMATH_GPT_coefficient_B_is_1_l543_54388

-- Definitions based on the conditions
def g (A B C D : ℝ) (x : ℝ) : ℝ := A * x^3 + B * x^2 + C * x + D

-- Given conditions
def condition1 (A B C D : ℝ) := g A B C D (-2) = 0 
def condition2 (A B C D : ℝ) := g A B C D 0 = -1
def condition3 (A B C D : ℝ) := g A B C D 2 = 0

-- The main theorem to prove
theorem coefficient_B_is_1 (A B C D : ℝ) 
  (h1 : condition1 A B C D) 
  (h2 : condition2 A B C D) 
  (h3 : condition3 A B C D) : 
  B = 1 :=
sorry

end NUMINAMATH_GPT_coefficient_B_is_1_l543_54388


namespace NUMINAMATH_GPT_cameron_books_ratio_l543_54363

theorem cameron_books_ratio (Boris_books : ℕ) (Cameron_books : ℕ)
  (Boris_after_donation : ℕ) (Cameron_after_donation : ℕ)
  (total_books_after_donation : ℕ) (ratio : ℚ) :
  Boris_books = 24 → 
  Cameron_books = 30 → 
  Boris_after_donation = Boris_books - (Boris_books / 4) →
  total_books_after_donation = 38 →
  Cameron_after_donation = total_books_after_donation - Boris_after_donation →
  ratio = (Cameron_books - Cameron_after_donation) / Cameron_books →
  ratio = 1 / 3 :=
by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_cameron_books_ratio_l543_54363


namespace NUMINAMATH_GPT_supplement_of_complementary_angle_of_35_deg_l543_54344

theorem supplement_of_complementary_angle_of_35_deg :
  let A := 35
  let C := 90 - A
  let S := 180 - C
  S = 125 :=
by
  let A := 35
  let C := 90 - A
  let S := 180 - C
  -- we need to prove S = 125
  sorry

end NUMINAMATH_GPT_supplement_of_complementary_angle_of_35_deg_l543_54344


namespace NUMINAMATH_GPT_find_original_price_l543_54384

-- Define the original price and conditions
def original_price (P : ℝ) : Prop :=
  ∃ discount final_price, discount = 0.55 ∧ final_price = 450000 ∧ ((1 - discount) * P = final_price)

-- The theorem to prove the original price before discount
theorem find_original_price (P : ℝ) (h : original_price P) : P = 1000000 :=
by
  sorry

end NUMINAMATH_GPT_find_original_price_l543_54384


namespace NUMINAMATH_GPT_gcd_1680_1683_l543_54313

theorem gcd_1680_1683 :
  ∀ (n : ℕ), n = 1683 →
  (∀ m, (m = 5 ∨ m = 67 ∨ m = 8) → n % m = 3) →
  (∃ d, d > 1 ∧ d ∣ 1683 ∧ d = Nat.gcd 1680 n ∧ Nat.gcd 1680 n = 3) :=
by
  sorry

end NUMINAMATH_GPT_gcd_1680_1683_l543_54313


namespace NUMINAMATH_GPT_find_x_l543_54361

noncomputable def a : ℝ × ℝ := (2, 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (x, 2)
noncomputable def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
noncomputable def scalar_vec_mul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)
noncomputable def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

theorem find_x (x : ℝ) :
  (vec_add a (b x)).1 * (vec_sub a (scalar_vec_mul 2 (b x))).2 =
  (vec_add a (b x)).2 * (vec_sub a (scalar_vec_mul 2 (b x))).1 →
  x = 4 :=
by sorry

end NUMINAMATH_GPT_find_x_l543_54361


namespace NUMINAMATH_GPT_find_n_l543_54369

theorem find_n (n : ℤ) (hn_range : -150 < n ∧ n < 150) (h_tan : Real.tan (n * Real.pi / 180) = Real.tan (286 * Real.pi / 180)) : 
  n = -74 :=
sorry

end NUMINAMATH_GPT_find_n_l543_54369


namespace NUMINAMATH_GPT_remainder_div_l543_54365

theorem remainder_div (N : ℤ) (k : ℤ) (h : N = 39 * k + 18) :
  N % 13 = 5 := 
by
  sorry

end NUMINAMATH_GPT_remainder_div_l543_54365


namespace NUMINAMATH_GPT_bob_equals_alice_l543_54349

-- Define conditions as constants
def original_price : ℝ := 120.00
def tax_rate : ℝ := 0.08
def discount_rate : ℝ := 0.25

-- Bob's total calculation
def bob_total : ℝ := (original_price * (1 + tax_rate)) * (1 - discount_rate)

-- Alice's total calculation
def alice_total : ℝ := (original_price * (1 - discount_rate)) * (1 + tax_rate)

-- Theorem statement to be proved
theorem bob_equals_alice : bob_total = alice_total := by sorry

end NUMINAMATH_GPT_bob_equals_alice_l543_54349


namespace NUMINAMATH_GPT_geometric_sequence_value_l543_54339

theorem geometric_sequence_value 
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_nonzero_diff : d ≠ 0)
  (h_condition : 2 * a 3 - (a 7) ^ 2 + 2 * a 11 = 0)
  (h_geom_seq : ∀ n, b (n + 1) = b n * (b 1 / b 0))
  (h_b7_eq_a7 : b 7 = a 7) :
  b 6 * b 8 = 16 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_value_l543_54339


namespace NUMINAMATH_GPT_inverse_proportion_indeterminate_l543_54305

theorem inverse_proportion_indeterminate (k : ℝ) (x1 x2 y1 y2 : ℝ) (h1 : x1 < x2)
  (h2 : y1 = k / x1) (h3 : y2 = k / x2) : 
  (y1 > 0 ∧ y2 > 0) ∨ (y1 < 0 ∧ y2 < 0) ∨ (y1 * y2 < 0) → false :=
sorry

end NUMINAMATH_GPT_inverse_proportion_indeterminate_l543_54305


namespace NUMINAMATH_GPT_prime_sequence_constant_l543_54348

open Nat

-- Define a predicate for prime numbers
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Define the recurrence relation
def recurrence_relation (p : ℕ → ℕ) (k : ℤ) : Prop :=
  ∀ n : ℕ, p (n + 2) = p (n + 1) + p n + k

-- Define the proof problem
theorem prime_sequence_constant (p : ℕ → ℕ) (k : ℤ) : 
  (∀ n, is_prime (p n)) →
  recurrence_relation p k →
  ∃ (q : ℕ), is_prime q ∧ (∀ n, p n = q) ∧ k = -q :=
by
  -- Sorry proof here
  sorry

end NUMINAMATH_GPT_prime_sequence_constant_l543_54348


namespace NUMINAMATH_GPT_best_shooter_l543_54354

noncomputable def avg_A : ℝ := 9
noncomputable def avg_B : ℝ := 8
noncomputable def avg_C : ℝ := 9
noncomputable def avg_D : ℝ := 9

noncomputable def var_A : ℝ := 1.2
noncomputable def var_B : ℝ := 0.4
noncomputable def var_C : ℝ := 1.8
noncomputable def var_D : ℝ := 0.4

theorem best_shooter :
  (avg_A = 9 ∧ var_A = 1.2) →
  (avg_B = 8 ∧ var_B = 0.4) →
  (avg_C = 9 ∧ var_C = 1.8) →
  (avg_D = 9 ∧ var_D = 0.4) →
  avg_D = 9 ∧ var_D = 0.4 :=
by {
  sorry
}

end NUMINAMATH_GPT_best_shooter_l543_54354


namespace NUMINAMATH_GPT_largest_expression_is_A_l543_54397

noncomputable def A : ℝ := 3009 / 3008 + 3009 / 3010
noncomputable def B : ℝ := 3011 / 3010 + 3011 / 3012
noncomputable def C : ℝ := 3010 / 3009 + 3010 / 3011

theorem largest_expression_is_A : A > B ∧ A > C := by
  sorry

end NUMINAMATH_GPT_largest_expression_is_A_l543_54397


namespace NUMINAMATH_GPT_greatest_possible_sum_of_two_consecutive_even_integers_l543_54392

theorem greatest_possible_sum_of_two_consecutive_even_integers
  (n : ℤ) (h1 : Even n) (h2 : n * (n + 2) < 800) :
  n + (n + 2) = 54 := 
sorry

end NUMINAMATH_GPT_greatest_possible_sum_of_two_consecutive_even_integers_l543_54392


namespace NUMINAMATH_GPT_real_part_zero_implies_a_eq_one_l543_54356

open Complex

theorem real_part_zero_implies_a_eq_one (a : ℝ) : 
  (1 + (1 : ℂ) * I) * (1 + a * I) = 0 ↔ a = 1 := by
  sorry

end NUMINAMATH_GPT_real_part_zero_implies_a_eq_one_l543_54356


namespace NUMINAMATH_GPT_mutually_exclusive_not_opposite_l543_54335

-- Define the given conditions
def boys := 6
def girls := 5
def total_students := boys + girls
def selection := 3

-- Define the mutually exclusive and not opposite events
def event_at_least_2_boys := ∃ (b: ℕ), ∃ (g: ℕ), (b + g = selection) ∧ (b ≥ 2) ∧ (g ≤ (selection - b))
def event_at_least_2_girls := ∃ (b: ℕ), ∃ (g: ℕ), (b + g = selection) ∧ (g ≥ 2) ∧ (b ≤ (selection - g))

-- Statement that these events are mutually exclusive but not opposite
theorem mutually_exclusive_not_opposite :
  (event_at_least_2_boys ∧ event_at_least_2_girls) → 
  (¬ ((∃ (b: ℕ) (g: ℕ), b + g = selection ∧ b ≥ 2 ∧ g ≥ 2) ∧ ¬(event_at_least_2_boys))) :=
sorry

end NUMINAMATH_GPT_mutually_exclusive_not_opposite_l543_54335


namespace NUMINAMATH_GPT_incident_ray_slope_in_circle_problem_l543_54320

noncomputable def slope_of_incident_ray : ℚ := sorry

theorem incident_ray_slope_in_circle_problem :
  ∃ (P : ℝ × ℝ) (C : ℝ × ℝ) (D : ℝ × ℝ),
  P = (-1, -3) ∧
  C = (2, -1) ∧
  (D = (C.1, -C.2)) ∧
  (D = (2, 1)) ∧
  ∀ (m : ℚ), (m = (D.2 - P.2) / (D.1 - P.1)) → m = 4 / 3 := 
sorry

end NUMINAMATH_GPT_incident_ray_slope_in_circle_problem_l543_54320


namespace NUMINAMATH_GPT_solve_for_b_l543_54399

noncomputable def g (a b : ℝ) (x : ℝ) := 1 / (2 * a * x + 3 * b)

theorem solve_for_b (a b : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) :
  (g a b (2) = 1 / (4 * a + 3 * b)) → (4 * a + 3 * b = 1 / 2) → b = (1 - 4 * a) / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_b_l543_54399


namespace NUMINAMATH_GPT_range_of_a_l543_54334

noncomputable def min_expr (x: ℝ) : ℝ := x + 2/(x - 2)

theorem range_of_a (a: ℝ) : 
  (∀ x > 2, a ≤ min_expr x) ↔ a ≤ 2 + 2 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l543_54334


namespace NUMINAMATH_GPT_solve_abs_equation_l543_54368

theorem solve_abs_equation (x : ℝ) :
  (|2 * x + 1| - |x - 5| = 6) ↔ (x = -12 ∨ x = 10 / 3) :=
by sorry

end NUMINAMATH_GPT_solve_abs_equation_l543_54368


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l543_54325

-- Proof statement for Problem 1
theorem problem1 : 23 * (-5) - (-3) / (3 / 108) = -7 := 
by 
  sorry

-- Proof statement for Problem 2
theorem problem2 : (-7) * (-3) * (-0.5) + (-12) * (-2.6) = 20.7 := 
by 
  sorry

-- Proof statement for Problem 3
theorem problem3 : ((-1 / 2) - (1 / 12) + (3 / 4) - (1 / 6)) * (-48) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l543_54325


namespace NUMINAMATH_GPT_problem_solution_l543_54322

theorem problem_solution (k a b : ℝ) (h1 : k = a + Real.sqrt b) 
  (h2 : abs (Real.logb 5 k - Real.logb 5 (k^2 + 3)) = 0.6) : 
  a + b = 15 :=
sorry

end NUMINAMATH_GPT_problem_solution_l543_54322


namespace NUMINAMATH_GPT_total_yards_thrown_l543_54364

-- Definitions for the conditions
def distance_50_degrees : ℕ := 20
def distance_80_degrees : ℕ := distance_50_degrees * 2

def throws_on_saturday : ℕ := 20
def throws_on_sunday : ℕ := 30

def headwind_penalty : ℕ := 5
def tailwind_bonus : ℕ := 10

-- Theorem for the total yards thrown in two days
theorem total_yards_thrown :
  ((distance_50_degrees - headwind_penalty) * throws_on_saturday) + 
  ((distance_80_degrees + tailwind_bonus) * throws_on_sunday) = 1800 :=
by
  sorry

end NUMINAMATH_GPT_total_yards_thrown_l543_54364


namespace NUMINAMATH_GPT_base_8_subtraction_l543_54321

def subtract_in_base_8 (a b : ℕ) : ℕ := 
  -- Implementing the base 8 subtraction
  sorry

theorem base_8_subtraction : subtract_in_base_8 0o652 0o274 = 0o356 :=
by 
  -- Faking the proof to ensure it can compile.
  sorry

end NUMINAMATH_GPT_base_8_subtraction_l543_54321


namespace NUMINAMATH_GPT_geometric_mean_unique_solution_l543_54300

-- Define the conditions
variable (k : ℕ) -- k is a natural number
variable (hk_pos : 0 < k) -- k is a positive natural number

-- The geometric mean condition translated to Lean
def geometric_mean_condition (k : ℕ) : Prop :=
  (2 * k)^2 = (k + 9) * (6 - k)

-- The main statement to prove
theorem geometric_mean_unique_solution (k : ℕ) (hk_pos : 0 < k) (h: geometric_mean_condition k) : k = 3 :=
sorry -- proof placeholder

end NUMINAMATH_GPT_geometric_mean_unique_solution_l543_54300


namespace NUMINAMATH_GPT_bill_sunday_miles_l543_54331

variables (B J M S : ℝ)

-- Conditions
def condition_1 := B + 4
def condition_2 := 2 * (B + 4)
def condition_3 := J = 0 ∧ M = 5 ∧ (M + 2 = 7)
def condition_4 := (B + 5) + (B + 4) + 2 * (B + 4) + 7 = 50

-- The main theorem to prove the number of miles Bill ran on Sunday
theorem bill_sunday_miles (h1 : S = B + 4) (h2 : ∀ B, J = 0 → M = 5 → S + 2 = 7 → (B + 5) + S + 2 * S + 7 = 50) : S = 10.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_bill_sunday_miles_l543_54331


namespace NUMINAMATH_GPT_find_coordinates_of_M_l543_54357

-- Definitions of the points A, B, C
def A : (ℝ × ℝ) := (2, -4)
def B : (ℝ × ℝ) := (-1, 3)
def C : (ℝ × ℝ) := (3, 4)

-- Definitions of vectors CA and CB
def vector_CA : (ℝ × ℝ) := (A.1 - C.1, A.2 - C.2)
def vector_CB : (ℝ × ℝ) := (B.1 - C.1, B.2 - C.2)

-- Definition of the point M
def M : (ℝ × ℝ) := (-11, -15)

-- Definition of vector CM
def vector_CM : (ℝ × ℝ) := (M.1 - C.1, M.2 - C.2)

-- The condition to prove
theorem find_coordinates_of_M : vector_CM = (2 * vector_CA.1 + 3 * vector_CB.1, 2 * vector_CA.2 + 3 * vector_CB.2) :=
by
  sorry

end NUMINAMATH_GPT_find_coordinates_of_M_l543_54357


namespace NUMINAMATH_GPT_maximum_perimeter_triangle_area_l543_54319

-- Part 1: Maximum Perimeter
theorem maximum_perimeter (a b c : ℝ) (A B C : ℝ) 
  (h_c : c = 2) 
  (h_C : C = Real.pi / 3) :
  (a + b + c) ≤ 6 :=
sorry

-- Part 2: Area under given trigonometric condition
theorem triangle_area (A B C a b c : ℝ) 
  (h_c : 2 * Real.sin (2 * A) + Real.sin (2 * B + C) = Real.sin C) :
  (1/2 * a * b * Real.sin C) = (2 * Real.sqrt 6) / 3 :=
sorry

end NUMINAMATH_GPT_maximum_perimeter_triangle_area_l543_54319


namespace NUMINAMATH_GPT_mode_and_median_of_survey_l543_54387

/-- A data structure representing the number of students corresponding to each sleep time. -/
structure SleepSurvey :=
  (time7 : ℕ)
  (time8 : ℕ)
  (time9 : ℕ)
  (time10 : ℕ)

def survey : SleepSurvey := { time7 := 6, time8 := 9, time9 := 11, time10 := 4 }

theorem mode_and_median_of_survey (s : SleepSurvey) :
  (mode=9 ∧ median = 8.5) :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_mode_and_median_of_survey_l543_54387


namespace NUMINAMATH_GPT_tommys_family_members_l543_54380

-- Definitions
def ounces_per_member : ℕ := 16
def ounces_per_steak : ℕ := 20
def steaks_needed : ℕ := 4

-- Theorem statement
theorem tommys_family_members : (steaks_needed * ounces_per_steak) / ounces_per_member = 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_tommys_family_members_l543_54380


namespace NUMINAMATH_GPT_final_price_percentage_l543_54345

theorem final_price_percentage (original_price sale_price final_price : ℝ) (h1 : sale_price = 0.9 * original_price) 
(h2 : final_price = sale_price - 0.1 * sale_price) : final_price / original_price = 0.81 :=
by
  sorry

end NUMINAMATH_GPT_final_price_percentage_l543_54345


namespace NUMINAMATH_GPT_original_number_of_players_l543_54310

theorem original_number_of_players 
    (n : ℕ) (W : ℕ)
    (h1 : W = n * 112)
    (h2 : W + 110 + 60 = (n + 2) * 106) : 
    n = 7 :=
by
  sorry

end NUMINAMATH_GPT_original_number_of_players_l543_54310


namespace NUMINAMATH_GPT_good_carrots_l543_54312

-- Definitions
def vanessa_carrots : ℕ := 17
def mother_carrots : ℕ := 14
def bad_carrots : ℕ := 7

-- Proof statement
theorem good_carrots : (vanessa_carrots + mother_carrots) - bad_carrots = 24 := by
  sorry

end NUMINAMATH_GPT_good_carrots_l543_54312


namespace NUMINAMATH_GPT_trajectory_of_point_l543_54315

theorem trajectory_of_point (x y k : ℝ) (hx : x ≠ 0) (hk : k ≠ 0) (h : |y| / |x| = k) : y = k * x ∨ y = -k * x :=
by
  sorry

end NUMINAMATH_GPT_trajectory_of_point_l543_54315


namespace NUMINAMATH_GPT_min_focal_length_l543_54391

theorem min_focal_length (a b c : ℝ) (h : a > 0 ∧ b > 0) 
    (hyperbola_eq : ∀ x y, ((x^2 / a^2) - (y^2 / b^2) = 1))
    (line_intersects_asymptotes_at : x = a)
    (area_of_triangle : 1/2 * a * (2 * b) = 8) :
    2 * c = 8 :=
by
  sorry

end NUMINAMATH_GPT_min_focal_length_l543_54391


namespace NUMINAMATH_GPT_coronavirus_transmission_l543_54395

theorem coronavirus_transmission (x : ℝ) 
  (H: (1 + x) ^ 2 = 225) : (1 + x) ^ 2 = 225 :=
  by
    sorry

end NUMINAMATH_GPT_coronavirus_transmission_l543_54395


namespace NUMINAMATH_GPT_find_sum_of_smallest_multiples_l543_54393

-- Define c as the smallest positive two-digit multiple of 5
def is_smallest_two_digit_multiple_of_5 (c : ℕ) : Prop :=
  c ≥ 10 ∧ c % 5 = 0 ∧ ∀ n, (n ≥ 10 ∧ n % 5 = 0) → n ≥ c

-- Define d as the smallest positive three-digit multiple of 7
def is_smallest_three_digit_multiple_of_7 (d : ℕ) : Prop :=
  d ≥ 100 ∧ d % 7 = 0 ∧ ∀ n, (n ≥ 100 ∧ n % 7 = 0) → n ≥ d

theorem find_sum_of_smallest_multiples :
  ∃ c d : ℕ, is_smallest_two_digit_multiple_of_5 c ∧ is_smallest_three_digit_multiple_of_7 d ∧ c + d = 115 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_of_smallest_multiples_l543_54393


namespace NUMINAMATH_GPT_distance_between_points_l543_54346

/-- Given points P1 and P2 in the plane, prove that the distance between 
P1 and P2 is 5 units. -/
theorem distance_between_points : 
  let P1 : ℝ × ℝ := (-1, 1)
  let P2 : ℝ × ℝ := (2, 5)
  dist P1 P2 = 5 :=
by 
  sorry

end NUMINAMATH_GPT_distance_between_points_l543_54346


namespace NUMINAMATH_GPT_find_x_range_l543_54338

noncomputable def p (x : ℝ) := x^2 + 2*x - 3 > 0
noncomputable def q (x : ℝ) := 1/(3 - x) > 1

theorem find_x_range (x : ℝ) : (¬q x ∧ p x) → (x ≥ 3 ∨ (1 < x ∧ x ≤ 2) ∨ x < -3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_x_range_l543_54338


namespace NUMINAMATH_GPT_find_number_l543_54302

variable (x : ℕ)
variable (result : ℕ)

theorem find_number (h : x * 9999 = 4690640889) : x = 469131 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l543_54302


namespace NUMINAMATH_GPT_solution_sets_l543_54389

-- These are the hypotheses derived from the problem conditions.
structure Conditions (a b c d : ℕ) : Prop :=
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (positive_even : ∃ u v w x : ℕ, a = 2*u ∧ b = 2*v ∧ c = 2*w ∧ d = 2*x ∧ 
                   u > 0 ∧ v > 0 ∧ w > 0 ∧ x > 0)
  (sum_100 : a + b + c + d = 100)
  (third_fourth_single_digit : c < 20 ∧ d < 20)
  (sum_2000 : 12 * a + 30 * b + 52 * c = 2000)

-- The main theorem in Lean asserting that these are the only possible sets of numbers.
theorem solution_sets :
  ∃ (a b c d : ℕ), Conditions a b c d ∧
  ( 
    (a = 62 ∧ b = 14 ∧ c = 4 ∧ d = 1) ∨ 
    (a = 48 ∧ b = 22 ∧ c = 2 ∧ d = 3)
  ) :=
  sorry

end NUMINAMATH_GPT_solution_sets_l543_54389


namespace NUMINAMATH_GPT_max_sqrt_sum_l543_54333

theorem max_sqrt_sum (x y : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hxy : x + y = 8) :
  abs (Real.sqrt (x - 1 / y) + Real.sqrt (y - 1 / x)) ≤ Real.sqrt 15 :=
sorry

end NUMINAMATH_GPT_max_sqrt_sum_l543_54333


namespace NUMINAMATH_GPT_exponent_rule_example_l543_54311

theorem exponent_rule_example {a : ℝ} : (a^3)^4 = a^12 :=
by {
  sorry
}

end NUMINAMATH_GPT_exponent_rule_example_l543_54311


namespace NUMINAMATH_GPT_students_in_class_l543_54307

theorem students_in_class (n : ℕ) 
  (h1 : 15 = 15)
  (h2 : ∃ m, n = m + 20 - 1)
  (h3 : ∃ x : ℕ, x = 3) :
  n = 38 :=
by
  sorry

end NUMINAMATH_GPT_students_in_class_l543_54307


namespace NUMINAMATH_GPT_differential_equation_solution_l543_54385

theorem differential_equation_solution (x y : ℝ) (C : ℝ) :
  (∀ dx dy, 2 * x * y * dx + x^2 * dy = 0) → x^2 * y = C :=
sorry

end NUMINAMATH_GPT_differential_equation_solution_l543_54385


namespace NUMINAMATH_GPT_negative_values_of_x_l543_54373

theorem negative_values_of_x : 
  let f (x : ℤ) := Int.sqrt (x + 196)
  ∃ (n : ℕ), (f (n ^ 2 - 196) > 0 ∧ f (n ^ 2 - 196) = n) ∧ ∃ k : ℕ, k = 13 :=
by
  sorry

end NUMINAMATH_GPT_negative_values_of_x_l543_54373


namespace NUMINAMATH_GPT_joan_carrots_grown_correct_l543_54396

variable (total_carrots : ℕ) (jessica_carrots : ℕ) (joan_carrots : ℕ)

theorem joan_carrots_grown_correct (h1 : total_carrots = 40) (h2 : jessica_carrots = 11) (h3 : total_carrots = joan_carrots + jessica_carrots) : joan_carrots = 29 :=
by
  sorry

end NUMINAMATH_GPT_joan_carrots_grown_correct_l543_54396


namespace NUMINAMATH_GPT_sum_of_digits_l543_54374

theorem sum_of_digits (a b c d : ℕ) (h1 : a + c = 11) (h2 : b + c = 9) (h3 : a + d = 10) (h_d : d - c = 1) : 
  a + b + c + d = 21 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_l543_54374


namespace NUMINAMATH_GPT_min_value_of_y_min_value_achieved_l543_54314

noncomputable def y (x : ℝ) : ℝ := x + 1/x + 16*x / (x^2 + 1)

theorem min_value_of_y : ∀ x > 1, y x ≥ 8 :=
  sorry

theorem min_value_achieved : ∃ x, (x > 1) ∧ (y x = 8) :=
  sorry

end NUMINAMATH_GPT_min_value_of_y_min_value_achieved_l543_54314


namespace NUMINAMATH_GPT_simplify_complex_fraction_l543_54386

theorem simplify_complex_fraction : 
  ∀ (i : ℂ), 
  i^2 = -1 → 
  (2 - 2 * i) / (3 + 4 * i) = -(2 / 25 : ℝ) - (14 / 25) * i :=
by
  intros
  sorry

end NUMINAMATH_GPT_simplify_complex_fraction_l543_54386


namespace NUMINAMATH_GPT_range_of_a_l543_54394

-- Definitions capturing the given conditions
variables (a b c : ℝ)

-- Conditions are stated as assumptions
def condition1 := a^2 - b * c - 8 * a + 7 = 0
def condition2 := b^2 + c^2 + b * c - 6 * a + 6 = 0

-- The mathematically equivalent proof problem
theorem range_of_a (h1 : condition1 a b c) (h2 : condition2 a b c) : 1 ≤ a ∧ a ≤ 9 := 
sorry

end NUMINAMATH_GPT_range_of_a_l543_54394


namespace NUMINAMATH_GPT_problem_a_l543_54370

theorem problem_a (k l m : ℝ) : 
  (k + l + m) ^ 2 >= 3 * (k * l + l * m + m * k) :=
by sorry

end NUMINAMATH_GPT_problem_a_l543_54370


namespace NUMINAMATH_GPT_fraction_expression_l543_54383

theorem fraction_expression :
  ((3 / 7) + (5 / 8)) / ((5 / 12) + (2 / 9)) = (531 / 322) :=
by
  sorry

end NUMINAMATH_GPT_fraction_expression_l543_54383


namespace NUMINAMATH_GPT_max_value_of_function_l543_54378

theorem max_value_of_function : ∀ x : ℝ, (0 < x ∧ x < 1) → x * (1 - x) ≤ 1 / 4 :=
sorry

end NUMINAMATH_GPT_max_value_of_function_l543_54378


namespace NUMINAMATH_GPT_total_turtles_l543_54329

theorem total_turtles (G H L : ℕ) (h_G : G = 800) (h_H : H = 2 * G) (h_L : L = 3 * G) : G + H + L = 4800 :=
by
  sorry

end NUMINAMATH_GPT_total_turtles_l543_54329


namespace NUMINAMATH_GPT_penalty_kicks_l543_54308

-- Define the soccer team data
def total_players : ℕ := 16
def goalkeepers : ℕ := 2
def players_shooting : ℕ := total_players - goalkeepers -- 14

-- Function to calculate total penalty kicks
def total_penalty_kicks (total_players goalkeepers : ℕ) : ℕ :=
  let players_shooting := total_players - goalkeepers
  players_shooting * goalkeepers

-- Theorem stating the number of penalty kicks
theorem penalty_kicks : total_penalty_kicks total_players goalkeepers = 30 :=
by
  sorry

end NUMINAMATH_GPT_penalty_kicks_l543_54308


namespace NUMINAMATH_GPT_expression_divisible_by_1968_l543_54332

theorem expression_divisible_by_1968 (n : ℕ) : 
  ( -1 ^ (2 * n) +  9 ^ (4 * n) - 6 ^ (8 * n) + 8 ^ (16 * n) ) % 1968 = 0 :=
by
  sorry

end NUMINAMATH_GPT_expression_divisible_by_1968_l543_54332


namespace NUMINAMATH_GPT_complement_intersection_eq_l543_54347

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection_eq :
  (U \ (M ∩ N)) = {1, 3, 4} := by
  sorry

end NUMINAMATH_GPT_complement_intersection_eq_l543_54347


namespace NUMINAMATH_GPT_no_C_makes_2C7_even_and_multiple_of_5_l543_54318

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

theorem no_C_makes_2C7_even_and_multiple_of_5 : ∀ C : ℕ, ¬(C < 10) ∨ ¬(is_even (2 * 100 + C * 10 + 7) ∧ is_multiple_of_5 (2 * 100 + C * 10 + 7)) :=
by
  intro C
  sorry

end NUMINAMATH_GPT_no_C_makes_2C7_even_and_multiple_of_5_l543_54318


namespace NUMINAMATH_GPT_percentage_games_won_l543_54376

def total_games_played : ℕ := 75
def win_rate_first_100_games : ℝ := 0.65

theorem percentage_games_won : 
  (win_rate_first_100_games * total_games_played / total_games_played * 100) = 65 := 
by
  sorry

end NUMINAMATH_GPT_percentage_games_won_l543_54376


namespace NUMINAMATH_GPT_black_area_after_six_transformations_l543_54328

noncomputable def remaining_fraction_after_transformations (initial_fraction : ℚ) (transforms : ℕ) (reduction_factor : ℚ) : ℚ :=
  reduction_factor ^ transforms * initial_fraction

theorem black_area_after_six_transformations :
  remaining_fraction_after_transformations 1 6 (2 / 3) = 64 / 729 := 
by
  sorry

end NUMINAMATH_GPT_black_area_after_six_transformations_l543_54328


namespace NUMINAMATH_GPT_infinitely_many_solutions_l543_54362

def circ (x y : ℝ) : ℝ := 4 * x - 3 * y + x * y

theorem infinitely_many_solutions : ∀ y : ℝ, circ 3 y = 12 := by
  sorry

end NUMINAMATH_GPT_infinitely_many_solutions_l543_54362
