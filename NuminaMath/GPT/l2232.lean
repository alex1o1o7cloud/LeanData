import Mathlib

namespace NUMINAMATH_GPT_max_distance_eq_of_l1_l2232_223208

noncomputable def equation_of_l1 (l1 l2 : ℝ → ℝ) (A B : ℝ × ℝ) : Prop :=
  A = (1, 3) ∧ B = (2, 4) ∧ -- Points A and B
  l1 A.1 = A.2 ∧ l2 B.1 = B.2 ∧ -- l1 passes through A and l2 passes through B
  (∀ (x : ℝ), l1 x - l2 x = 1) ∧ -- l1 and l2 are parallel (constant difference in y-values)
  (∃ (c : ℝ), ∀ (x : ℝ), l1 x = -x + c ∧ l2 x = -x + c + 1) -- distance maximized

theorem max_distance_eq_of_l1 : 
  ∃ (l1 l2 : ℝ → ℝ), equation_of_l1 l1 l2 (1, 3) (2, 4) ∧
  ∀ (x : ℝ), l1 x = -x + 4 := 
sorry

end NUMINAMATH_GPT_max_distance_eq_of_l1_l2232_223208


namespace NUMINAMATH_GPT_toll_for_18_wheel_truck_l2232_223241

theorem toll_for_18_wheel_truck : 
  let x := 5 
  let w := 15 
  let y := 2 
  let T := 3.50 + 0.50 * (x - 2) + 0.10 * w + y 
  T = 8.50 := 
by 
  -- let x := 5 
  -- let w := 15 
  -- let y := 2 
  -- let T := 3.50 + 0.50 * (x - 2) + 0.10 * w + y 
  -- Note: the let statements within the brackets above
  sorry

end NUMINAMATH_GPT_toll_for_18_wheel_truck_l2232_223241


namespace NUMINAMATH_GPT_prove_difference_l2232_223263

theorem prove_difference (x y : ℝ) (h1 : x + y = 500) (h2 : x * y = 22000) : y - x = -402.5 :=
sorry

end NUMINAMATH_GPT_prove_difference_l2232_223263


namespace NUMINAMATH_GPT_questionnaire_visitors_l2232_223268

theorem questionnaire_visitors (V E : ℕ) (H1 : 140 = V - E) 
  (H2 : E = (3 * V) / 4) : V = 560 :=
by
  sorry

end NUMINAMATH_GPT_questionnaire_visitors_l2232_223268


namespace NUMINAMATH_GPT_find_n_l2232_223252

theorem find_n 
  (num_engineers : ℕ) (num_technicians : ℕ) (num_workers : ℕ)
  (total_population : ℕ := num_engineers + num_technicians + num_workers)
  (systematic_sampling_inclusion_exclusion : ∀ n : ℕ, ∃ k : ℕ, n ∣ total_population ↔ n + 1 ≠ total_population) 
  (stratified_sampling_lcm : ∃ lcm : ℕ, lcm = Nat.lcm (Nat.lcm num_engineers num_technicians) num_workers)
  (total_population_is_36 : total_population = 36)
  (num_engineers_is_6 : num_engineers = 6)
  (num_technicians_is_12 : num_technicians = 12)
  (num_workers_is_18 : num_workers = 18) :
  ∃ n : ℕ, n = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l2232_223252


namespace NUMINAMATH_GPT_constant_function_of_horizontal_tangent_l2232_223290

theorem constant_function_of_horizontal_tangent (f : ℝ → ℝ) (h : ∀ x, deriv f x = 0) : ∃ c : ℝ, ∀ x, f x = c :=
sorry

end NUMINAMATH_GPT_constant_function_of_horizontal_tangent_l2232_223290


namespace NUMINAMATH_GPT_value_of_q_l2232_223278

-- Define the problem in Lean 4

variable (a d q : ℝ) (h0 : a ≠ 0)
variables (M P : Set ℝ)
variable (hM : M = {a, a + d, a + 2 * d})
variable (hP : P = {a, a * q, a * q * q})
variable (hMP : M = P)

theorem value_of_q : q = -1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_q_l2232_223278


namespace NUMINAMATH_GPT_number_of_diagonals_l2232_223202

-- Define the regular pentagonal prism and its properties
def regular_pentagonal_prism : Type := sorry

-- Define what constitutes a diagonal in this context
def is_diagonal (p : regular_pentagonal_prism) (v1 v2 : Nat) : Prop :=
  sorry -- We need to detail what counts as a diagonal based on the conditions

-- Hypothesis on the structure specifying that there are 5 vertices on the top and 5 on the bottom
axiom vertices_on_top_and_bottom (p : regular_pentagonal_prism) : sorry -- We need the precise formalization

-- The main theorem
theorem number_of_diagonals (p : regular_pentagonal_prism) : ∃ n, n = 10 :=
  sorry

end NUMINAMATH_GPT_number_of_diagonals_l2232_223202


namespace NUMINAMATH_GPT_planar_graph_edge_bound_l2232_223224

structure Graph :=
  (V E : ℕ) -- vertices and edges

def planar_connected (G : Graph) : Prop := 
  sorry -- Planarity and connectivity conditions are complex to formalize

def num_faces (G : Graph) : ℕ :=
  sorry -- Number of faces based on V, E and planarity

theorem planar_graph_edge_bound (G : Graph) (h_planar : planar_connected G) 
  (euler : G.V - G.E + num_faces G = 2) 
  (face_bound : 2 * G.E ≥ 3 * num_faces G) : 
  G.E ≤ 3 * G.V - 6 :=
sorry

end NUMINAMATH_GPT_planar_graph_edge_bound_l2232_223224


namespace NUMINAMATH_GPT_right_triangle_area_l2232_223214

open Real

theorem right_triangle_area
  (a : ℝ) 
  (h₁ : a > 0) 
  (h₂ : a < 24)
  (h₃ : 24^2 + a^2 = (48 - a)^2) : 
  1/2 * 24 * a = 216 :=
by
  -- This is just a statement, the proof is omitted
  sorry

end NUMINAMATH_GPT_right_triangle_area_l2232_223214


namespace NUMINAMATH_GPT_remainder_when_divided_by_39_l2232_223258

theorem remainder_when_divided_by_39 (N k : ℤ) (h : N = 13 * k + 4) : (N % 39) = 4 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_39_l2232_223258


namespace NUMINAMATH_GPT_conference_center_capacity_l2232_223227

theorem conference_center_capacity (n_rooms : ℕ) (fraction_full : ℚ) (current_people : ℕ) (full_capacity : ℕ) (people_per_room : ℕ) 
  (h1 : n_rooms = 6) (h2 : fraction_full = 2/3) (h3 : current_people = 320) (h4 : current_people = fraction_full * full_capacity) 
  (h5 : people_per_room = full_capacity / n_rooms) : people_per_room = 80 :=
by
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_conference_center_capacity_l2232_223227


namespace NUMINAMATH_GPT_parabola_properties_l2232_223203

theorem parabola_properties (p : ℝ) (h : p > 0) (F : ℝ × ℝ) (l : ℝ → ℝ) (A B : ℝ × ℝ) (M : ℝ × ℝ)
  (hp : p = 4) 
  (hF : F = (p / 2, 0)) 
  (hA : A.2^2 = 2 * p * A.1) 
  (hB : B.2^2 = 2 * p * B.1) 
  (hM : M = ((A.1 + B.1) / 2, 2)) 
  (hl : ∀ x, l x = 2 * x - 4) 
  : (p = 4) ∧ (l 0 = -4) ∧ (A ≠ B) → 
    (p = 4) ∧ (l 0 = -4) ∧ (A ≠ B) ∧ (|A.1 - B.1| + |A.2 - B.2| = 10) :=
by 
  sorry

end NUMINAMATH_GPT_parabola_properties_l2232_223203


namespace NUMINAMATH_GPT_find_cost_of_baseball_l2232_223206

noncomputable def total_amount : ℝ := 20.52
noncomputable def cost_of_marbles : ℝ := 9.05
noncomputable def cost_of_football : ℝ := 4.95
noncomputable def cost_of_baseball : ℝ := total_amount - (cost_of_marbles + cost_of_football)

theorem find_cost_of_baseball : cost_of_baseball = 6.52 := sorry

end NUMINAMATH_GPT_find_cost_of_baseball_l2232_223206


namespace NUMINAMATH_GPT_graph_through_origin_y_increases_with_x_not_pass_third_quadrant_l2232_223228

-- Part 1: Prove that if the graph passes through the origin, then m ≠ 2/3 and n = 1
theorem graph_through_origin {m n : ℝ} : 
  (3 * m - 2 ≠ 0) → (1 - n = 0) ↔ (m ≠ 2/3 ∧ n = 1) :=
by sorry

-- Part 2: Prove that if y increases as x increases, then m > 2/3 and n is any real number
theorem y_increases_with_x {m n : ℝ} : 
  (3 * m - 2 > 0) ↔ (m > 2/3 ∧ ∀ n : ℝ, True) :=
by sorry

-- Part 3: Prove that if the graph does not pass through the third quadrant, then m < 2/3 and n ≤ 1
theorem not_pass_third_quadrant {m n : ℝ} : 
  (3 * m - 2 < 0) ∧ (1 - n ≥ 0) ↔ (m < 2/3 ∧ n ≤ 1) :=
by sorry

end NUMINAMATH_GPT_graph_through_origin_y_increases_with_x_not_pass_third_quadrant_l2232_223228


namespace NUMINAMATH_GPT_will_has_123_pieces_of_candy_l2232_223272

def initial_candy_pieces (chocolate_boxes mint_boxes caramel_boxes : ℕ)
  (pieces_per_chocolate_box pieces_per_mint_box pieces_per_caramel_box : ℕ) : ℕ :=
  chocolate_boxes * pieces_per_chocolate_box + mint_boxes * pieces_per_mint_box + caramel_boxes * pieces_per_caramel_box

def given_away_candy_pieces (given_chocolate_boxes given_mint_boxes given_caramel_boxes : ℕ)
  (pieces_per_chocolate_box pieces_per_mint_box pieces_per_caramel_box : ℕ) : ℕ :=
  given_chocolate_boxes * pieces_per_chocolate_box + given_mint_boxes * pieces_per_mint_box + given_caramel_boxes * pieces_per_caramel_box

def remaining_candy : ℕ :=
  let initial := initial_candy_pieces 7 5 4 12 15 10
  let given_away := given_away_candy_pieces 3 2 1 12 15 10
  initial - given_away

theorem will_has_123_pieces_of_candy : remaining_candy = 123 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_will_has_123_pieces_of_candy_l2232_223272


namespace NUMINAMATH_GPT_prism_volume_l2232_223256

theorem prism_volume (a b c : ℝ) (h1 : a * b = 18) (h2 : b * c = 12) (h3 : a * c = 8) : a * b * c = 12 :=
by sorry

end NUMINAMATH_GPT_prism_volume_l2232_223256


namespace NUMINAMATH_GPT_probability_same_outcomes_l2232_223222

-- Let us define the event space for a fair coin
inductive CoinTossOutcome
| H : CoinTossOutcome
| T : CoinTossOutcome

open CoinTossOutcome

-- Definition of an event where the outcomes are the same (HHH or TTT)
def same_outcomes (t1 t2 t3 : CoinTossOutcome) : Prop :=
  (t1 = H ∧ t2 = H ∧ t3 = H) ∨ (t1 = T ∧ t2 = T ∧ t3 = T)

-- Number of all possible outcomes for three coin tosses
def total_outcomes : ℕ := 2 ^ 3

-- Number of favorable outcomes where all outcomes are the same
def favorable_outcomes : ℕ := 2

-- Calculation of probability
def prob_same_outcomes : ℚ := favorable_outcomes / total_outcomes

-- The statement to be proved in Lean 4
theorem probability_same_outcomes : prob_same_outcomes = 1 / 4 := 
by sorry

end NUMINAMATH_GPT_probability_same_outcomes_l2232_223222


namespace NUMINAMATH_GPT_fraction_product_eq_l2232_223253

theorem fraction_product_eq : (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 2 / 7 := by
  sorry

end NUMINAMATH_GPT_fraction_product_eq_l2232_223253


namespace NUMINAMATH_GPT_range_of_a_l2232_223267

noncomputable def prop_p (a x : ℝ) : Prop := 3 * a < x ∧ x < a

noncomputable def prop_q (x : ℝ) : Prop := x^2 - x - 6 < 0

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, ¬ prop_p a x) ∧ ¬ (∃ x : ℝ, ¬ prop_p a x) → ¬ (∃ x : ℝ, ¬ prop_q x) → -2/3 ≤ a ∧ a < 0 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2232_223267


namespace NUMINAMATH_GPT_algebraic_expression_value_l2232_223266

theorem algebraic_expression_value (x : ℝ) (h : 3 / (x^2 + x) - x^2 = 2 + x) :
  2 * x^2 + 2 * x = 2 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l2232_223266


namespace NUMINAMATH_GPT_min_value_of_a_l2232_223275

theorem min_value_of_a (a : ℝ) :
  (¬ ∃ x0 : ℝ, -1 < x0 ∧ x0 ≤ 2 ∧ x0 - a > 0) → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_a_l2232_223275


namespace NUMINAMATH_GPT_addition_belongs_to_Q_l2232_223251

def P : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}
def Q : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def R : Set ℤ := {x | ∃ k : ℤ, x = 4 * k + 1}

theorem addition_belongs_to_Q (a b : ℤ) (ha : a ∈ P) (hb : b ∈ Q) : a + b ∈ Q := by
  sorry

end NUMINAMATH_GPT_addition_belongs_to_Q_l2232_223251


namespace NUMINAMATH_GPT_calculation1_calculation2_calculation3_calculation4_l2232_223200

-- Define the problem and conditions
theorem calculation1 : 9.5 * 101 = 959.5 := 
by 
  sorry

theorem calculation2 : 12.5 * 8.8 = 110 := 
by 
  sorry

theorem calculation3 : 38.4 * 187 - 15.4 * 384 + 3.3 * 16 = 1320 := 
by 
  sorry

theorem calculation4 : 5.29 * 73 + 52.9 * 2.7 = 529 := 
by 
  sorry

end NUMINAMATH_GPT_calculation1_calculation2_calculation3_calculation4_l2232_223200


namespace NUMINAMATH_GPT_average_rate_of_interest_l2232_223205

theorem average_rate_of_interest (total_investment : ℝ) (rate1 rate2 average_rate : ℝ) (amount1 amount2 : ℝ)
  (H1 : total_investment = 6000)
  (H2 : rate1 = 0.03)
  (H3 : rate2 = 0.07)
  (H4 : average_rate = 0.042)
  (H5 : amount1 + amount2 = total_investment)
  (H6 : rate1 * amount1 = rate2 * amount2) :
  (rate1 * amount1 + rate2 * amount2) / total_investment = average_rate := 
sorry

end NUMINAMATH_GPT_average_rate_of_interest_l2232_223205


namespace NUMINAMATH_GPT_cost_of_each_book_l2232_223242

noncomputable def cost_of_book (money_given money_left notebook_cost notebook_count book_count : ℕ) : ℕ :=
  (money_given - money_left - (notebook_count * notebook_cost)) / book_count

-- Conditions
def money_given : ℕ := 56
def money_left : ℕ := 14
def notebook_cost : ℕ := 4
def notebook_count : ℕ := 7
def book_count : ℕ := 2

-- Theorem stating that the cost of each book is $7 under given conditions
theorem cost_of_each_book : cost_of_book money_given money_left notebook_cost notebook_count book_count = 7 := by
  sorry

end NUMINAMATH_GPT_cost_of_each_book_l2232_223242


namespace NUMINAMATH_GPT_geese_survived_first_year_l2232_223287

-- Definitions based on the conditions
def total_eggs := 900
def hatch_rate := 2 / 3
def survive_first_month_rate := 3 / 4
def survive_first_year_rate := 2 / 5

-- Definitions derived from the conditions
def hatched_geese := total_eggs * hatch_rate
def survived_first_month := hatched_geese * survive_first_month_rate
def survived_first_year := survived_first_month * survive_first_year_rate

-- Target proof statement
theorem geese_survived_first_year : survived_first_year = 180 := by
  sorry

end NUMINAMATH_GPT_geese_survived_first_year_l2232_223287


namespace NUMINAMATH_GPT_p_minus_q_l2232_223280

theorem p_minus_q (p q : ℚ) (h1 : 3 / p = 6) (h2 : 3 / q = 18) : p - q = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_p_minus_q_l2232_223280


namespace NUMINAMATH_GPT_principal_amount_l2232_223259

theorem principal_amount (SI : ℝ) (T : ℝ) (R : ℝ) (P : ℝ) (h1 : SI = 140) (h2 : T = 2) (h3 : R = 17.5) :
  P = 400 :=
by
  -- Formal proof would go here
  sorry

end NUMINAMATH_GPT_principal_amount_l2232_223259


namespace NUMINAMATH_GPT_rationalize_expression_l2232_223254

theorem rationalize_expression :
  (2 * Real.sqrt 3) / (Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5) = 
  (Real.sqrt 6 + 3 - Real.sqrt 15) / 2 :=
sorry

end NUMINAMATH_GPT_rationalize_expression_l2232_223254


namespace NUMINAMATH_GPT_range_of_a_l2232_223221

theorem range_of_a (x a : ℝ) (p : 0 < x ∧ x < 1)
  (q : (x - a) * (x - (a + 2)) ≤ 0) (h : ∀ x, (0 < x ∧ x < 1) → (x - a) * (x - (a + 2)) ≤ 0) :
  -1 ≤ a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2232_223221


namespace NUMINAMATH_GPT_share_ratio_l2232_223293

theorem share_ratio (A B C x : ℝ)
  (h1 : A = 280)
  (h2 : A + B + C = 700)
  (h3 : A = x * (B + C))
  (h4 : B = (6 / 9) * (A + C)) :
  A / (B + C) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_share_ratio_l2232_223293


namespace NUMINAMATH_GPT_no_positive_divisor_of_2n2_square_l2232_223231

theorem no_positive_divisor_of_2n2_square (n : ℕ) (hn : n > 0) : 
  ∀ d : ℕ, d > 0 → d ∣ 2 * n ^ 2 → ¬∃ x : ℕ, x ^ 2 = d ^ 2 * n ^ 2 + d ^ 3 := 
by
  sorry

end NUMINAMATH_GPT_no_positive_divisor_of_2n2_square_l2232_223231


namespace NUMINAMATH_GPT_first_digit_of_sum_l2232_223220

theorem first_digit_of_sum (n : ℕ) (a : ℕ) (hs : 9 * a = n)
  (h_sum : n = 43040102 - (10^7 * d - 10^7 * 4)) : 
  (10^7 * d - 10^7 * 4) / 10^7 = 8 :=
by
  sorry

end NUMINAMATH_GPT_first_digit_of_sum_l2232_223220


namespace NUMINAMATH_GPT_polynomial_roots_fraction_sum_l2232_223240

theorem polynomial_roots_fraction_sum (a b c : ℝ) 
  (h1 : a + b + c = 12) 
  (h2 : ab + ac + bc = 20) 
  (h3 : abc = 3) : 
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 328 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_roots_fraction_sum_l2232_223240


namespace NUMINAMATH_GPT_correct_product_of_a_b_l2232_223262

theorem correct_product_of_a_b (a b : ℕ) (h1 : (a - (10 * (a / 10 % 10) + 1)) * b = 255)
                              (h2 : (a - (10 * (a / 100 % 10 * 10 + a % 10 - (a / 100 % 10 * 10 + 5 * 10)))) * b = 335) :
  a * b = 285 := sorry

end NUMINAMATH_GPT_correct_product_of_a_b_l2232_223262


namespace NUMINAMATH_GPT_number_of_nintendo_games_to_give_away_l2232_223230

-- Define the conditions
def initial_nintendo_games : ℕ := 20
def desired_nintendo_games_left : ℕ := 12

-- Define the proof problem as a Lean theorem
theorem number_of_nintendo_games_to_give_away :
  initial_nintendo_games - desired_nintendo_games_left = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_nintendo_games_to_give_away_l2232_223230


namespace NUMINAMATH_GPT_total_number_of_games_in_season_l2232_223244

def number_of_games_per_month : ℕ := 13
def number_of_months_in_season : ℕ := 14

theorem total_number_of_games_in_season :
  number_of_games_per_month * number_of_months_in_season = 182 := by
  sorry

end NUMINAMATH_GPT_total_number_of_games_in_season_l2232_223244


namespace NUMINAMATH_GPT_morgan_olivia_same_debt_l2232_223234

theorem morgan_olivia_same_debt (t : ℝ) : 
  (200 * (1 + 0.12 * t) = 300 * (1 + 0.04 * t)) → 
  t = 25 / 3 :=
by
  sorry

end NUMINAMATH_GPT_morgan_olivia_same_debt_l2232_223234


namespace NUMINAMATH_GPT_anna_initial_stamps_l2232_223295

theorem anna_initial_stamps (final_stamps : ℕ) (alison_stamps : ℕ) (alison_to_anna : ℕ) : 
  final_stamps = 50 ∧ alison_stamps = 28 ∧ alison_to_anna = 14 → (final_stamps - alison_to_anna = 36) :=
by
  sorry

end NUMINAMATH_GPT_anna_initial_stamps_l2232_223295


namespace NUMINAMATH_GPT_circle_ratio_l2232_223219

theorem circle_ratio (R r a c : ℝ) (hR : 0 < R) (hr : 0 < r) (h_c_lt_a : 0 < c ∧ c < a) 
  (condition : π * R^2 = (a - c) * (π * R^2 - π * r^2)) :
  R / r = Real.sqrt ((a - c) / (c + 1 - a)) :=
by
  sorry

end NUMINAMATH_GPT_circle_ratio_l2232_223219


namespace NUMINAMATH_GPT_opposite_event_of_hitting_target_at_least_once_is_both_shots_miss_l2232_223299

/-- A person is shooting at a target, firing twice in succession. 
    The opposite event of "hitting the target at least once" is "both shots miss". -/
theorem opposite_event_of_hitting_target_at_least_once_is_both_shots_miss :
  ∀ (A B : Prop) (hits_target_at_least_once both_shots_miss : Prop), 
    (hits_target_at_least_once → (A ∨ B)) → (both_shots_miss ↔ ¬hits_target_at_least_once) ∧ 
    (¬(A ∧ B) → both_shots_miss) :=
by
  sorry

end NUMINAMATH_GPT_opposite_event_of_hitting_target_at_least_once_is_both_shots_miss_l2232_223299


namespace NUMINAMATH_GPT_conditional_probability_l2232_223247

def slips : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_odd (n : ℕ) : Prop := n % 2 ≠ 0

def P_A : ℚ := 5/9

def P_A_and_B : ℚ := 5/9 * 4/8

theorem conditional_probability :
  (5 / 18) / (5 / 9) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_conditional_probability_l2232_223247


namespace NUMINAMATH_GPT_system_is_inconsistent_l2232_223232

def system_of_equations (x1 x2 x3 : ℝ) : Prop :=
  (x1 + 4*x2 + 10*x3 = 1) ∧
  (0*x1 - 5*x2 - 13*x3 = -1.25) ∧
  (0*x1 + 0*x2 + 0*x3 = 1.25)

theorem system_is_inconsistent : 
  ∀ x1 x2 x3, ¬ system_of_equations x1 x2 x3 :=
by
  intro x1 x2 x3
  sorry

end NUMINAMATH_GPT_system_is_inconsistent_l2232_223232


namespace NUMINAMATH_GPT_compute_custom_op_l2232_223269

def custom_op (x y : ℤ) : ℤ := 
  x * y - y * x - 3 * x + 2 * y

theorem compute_custom_op : (custom_op 9 5) - (custom_op 5 9) = -20 := 
by
  sorry

end NUMINAMATH_GPT_compute_custom_op_l2232_223269


namespace NUMINAMATH_GPT_auntie_em_can_park_l2232_223237

noncomputable def parking_probability : ℚ :=
  let total_ways := (Nat.choose 20 5)
  let unfavorables := (Nat.choose 14 5)
  let probability_cannot_park := (unfavorables : ℚ) / total_ways
  1 - probability_cannot_park

theorem auntie_em_can_park :
  parking_probability = 964 / 1107 :=
by
  sorry

end NUMINAMATH_GPT_auntie_em_can_park_l2232_223237


namespace NUMINAMATH_GPT_milk_for_flour_l2232_223270

theorem milk_for_flour (milk flour use_flour : ℕ) (h1 : milk = 75) (h2 : flour = 300) (h3 : use_flour = 900) : (use_flour/flour * milk) = 225 :=
by sorry

end NUMINAMATH_GPT_milk_for_flour_l2232_223270


namespace NUMINAMATH_GPT_trigonometric_identity_l2232_223264

theorem trigonometric_identity :
  3 * Real.arcsin (Real.sqrt 3 / 2) - Real.arctan (-1) - Real.arccos 0 = (3 * Real.pi) / 4 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2232_223264


namespace NUMINAMATH_GPT_max_area_of_rectangle_l2232_223296

theorem max_area_of_rectangle (x y : ℝ) (h1 : 2 * (x + y) = 40) : 
  (x * y) ≤ 100 :=
by
  sorry

end NUMINAMATH_GPT_max_area_of_rectangle_l2232_223296


namespace NUMINAMATH_GPT_geometric_sequence_sum_l2232_223225

def geometric_sequence_props (a : ℕ → ℤ) (S : ℕ → ℤ) := 
  (∀ n, a n = a 1 * 2 ^ (n - 1)) ∧ 
  (∀ n, S n = (1 - 2^n) / (1 - 2)) ∧ 
  (a 5 - a 3 = 12) ∧ 
  (a 6 - a 4 = 24)

theorem geometric_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h : geometric_sequence_props a S) :
  ∀ n, S n / a n = 2 - 2^(1 - n) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l2232_223225


namespace NUMINAMATH_GPT_smallest_k_l2232_223273

def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k (k : ℕ) : (∀ z : ℂ, z ≠ 0 → f z ∣ z^k - 1) ↔ k = 40 :=
by sorry

end NUMINAMATH_GPT_smallest_k_l2232_223273


namespace NUMINAMATH_GPT_maynard_filled_percentage_l2232_223265

theorem maynard_filled_percentage (total_holes : ℕ) (unfilled_holes : ℕ) (filled_holes : ℕ) (p : ℚ) :
  total_holes = 8 →
  unfilled_holes = 2 →
  filled_holes = total_holes - unfilled_holes →
  p = (filled_holes : ℚ) / (total_holes : ℚ) * 100 →
  p = 75 := 
by {
  -- proofs and calculations would go here
  sorry
}

end NUMINAMATH_GPT_maynard_filled_percentage_l2232_223265


namespace NUMINAMATH_GPT_integral_cos_square_div_one_plus_cos_minus_sin_squared_l2232_223236

theorem integral_cos_square_div_one_plus_cos_minus_sin_squared:
  ∫ x in (-2 * Real.pi / 3 : Real)..0, (Real.cos x)^2 / (1 + Real.cos x - Real.sin x)^2 = (Real.sqrt 3) / 2 - Real.log 2 := 
by
  sorry

end NUMINAMATH_GPT_integral_cos_square_div_one_plus_cos_minus_sin_squared_l2232_223236


namespace NUMINAMATH_GPT_edward_spent_money_l2232_223211

-- Definitions based on the conditions
def books := 2
def cost_per_book := 3

-- Statement of the proof problem
theorem edward_spent_money : 
  (books * cost_per_book = 6) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_edward_spent_money_l2232_223211


namespace NUMINAMATH_GPT_combined_depths_underwater_l2232_223279

theorem combined_depths_underwater :
  let Ron_height := 12
  let Dean_height := Ron_height - 11
  let Sam_height := Dean_height + 2
  let Ron_depth := Ron_height / 2
  let Sam_depth := Sam_height
  let Dean_depth := Dean_height + 3
  Ron_depth + Sam_depth + Dean_depth = 13 :=
by
  let Ron_height := 12
  let Dean_height := Ron_height - 11
  let Sam_height := Dean_height + 2
  let Ron_depth := Ron_height / 2
  let Sam_depth := Sam_height
  let Dean_depth := Dean_height + 3
  show Ron_depth + Sam_depth + Dean_depth = 13
  sorry

end NUMINAMATH_GPT_combined_depths_underwater_l2232_223279


namespace NUMINAMATH_GPT_find_r_s_l2232_223216

noncomputable def parabola_line_intersection (x y m : ℝ) : Prop :=
  y = x^2 + 5*x ∧ y + 6 = m*(x - 10)

theorem find_r_s (r s m : ℝ) (Q : ℝ × ℝ)
  (hq : Q = (10, -6))
  (h_parabola : ∀ x, ∃ y, y = x^2 + 5*x)
  (h_line : ∀ x, ∃ y, y + 6 = m*(x - 10)) :
  parabola_line_intersection x y m → (r < m ∧ m < s) ∧ (r + s = 50) :=
sorry

end NUMINAMATH_GPT_find_r_s_l2232_223216


namespace NUMINAMATH_GPT_no_even_integers_of_form_3k_plus_4_and_5m_plus_2_l2232_223245

theorem no_even_integers_of_form_3k_plus_4_and_5m_plus_2 (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 1000) (h2 : ∃ k : ℕ, n = 3 * k + 4) (h3 : ∃ m : ℕ, n = 5 * m + 2) (h4 : n % 2 = 0) : false :=
sorry

end NUMINAMATH_GPT_no_even_integers_of_form_3k_plus_4_and_5m_plus_2_l2232_223245


namespace NUMINAMATH_GPT_abs_neg_three_l2232_223255

theorem abs_neg_three : |(-3 : ℤ)| = 3 := by
  sorry

end NUMINAMATH_GPT_abs_neg_three_l2232_223255


namespace NUMINAMATH_GPT_sequence_increasing_range_l2232_223297

theorem sequence_increasing_range (a : ℝ) (n : ℕ) : 
  (∀ n ≤ 5, (a - 1) ^ (n - 4) < (a - 1) ^ ((n+1) - 4)) ∧
  (∀ n > 5, (7 - a) * n - 1 < (7 - a) * (n + 1) - 1) ∧
  (a - 1 < (7 - a) * 6 - 1) 
  → 2 < a ∧ a < 6 := 
sorry

end NUMINAMATH_GPT_sequence_increasing_range_l2232_223297


namespace NUMINAMATH_GPT_john_pack_count_l2232_223274

-- Defining the conditions
def utensilsInPack : Nat := 30
def knivesInPack : Nat := utensilsInPack / 3
def forksInPack : Nat := utensilsInPack / 3
def spoonsInPack : Nat := utensilsInPack / 3
def requiredKnivesRatio : Nat := 2
def requiredForksRatio : Nat := 3
def requiredSpoonsRatio : Nat := 5
def minimumSpoons : Nat := 50

-- Proving the solution
theorem john_pack_count : 
  ∃ packs : Nat, 
    (packs * spoonsInPack >= minimumSpoons) ∧
    (packs * foonsInPack / packs * knivesInPack = requiredForksRatio / requiredKnivesRatio) ∧
    (packs * spoonsInPack / packs * forksInPack = requiredForksRatio / requiredSpoonsRatio) ∧
    (packs * spoonsInPack / packs * knivesInPack = requiredSpoonsRatio / requiredKnivesRatio) ∧
    packs = 5 :=
sorry

end NUMINAMATH_GPT_john_pack_count_l2232_223274


namespace NUMINAMATH_GPT_total_photos_in_gallery_l2232_223257

def initial_photos : ℕ := 800
def photos_first_day : ℕ := (2 * initial_photos) / 3
def photos_second_day : ℕ := photos_first_day + 180

theorem total_photos_in_gallery : initial_photos + photos_first_day + photos_second_day = 2046 := by
  -- the proof can be provided here
  sorry

end NUMINAMATH_GPT_total_photos_in_gallery_l2232_223257


namespace NUMINAMATH_GPT_sin_690_degree_l2232_223233

theorem sin_690_degree : Real.sin (690 * Real.pi / 180) = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_sin_690_degree_l2232_223233


namespace NUMINAMATH_GPT_max_projection_area_tetrahedron_l2232_223207

-- Define the side length of the tetrahedron
variable (a : ℝ)

-- Define a theorem stating the maximum projection area of a tetrahedron
theorem max_projection_area_tetrahedron (h : a > 0) : 
  ∃ A, A = (a^2 / 2) :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_max_projection_area_tetrahedron_l2232_223207


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l2232_223284

-- Definitions from conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Proof problem statement
theorem intersection_of_M_and_N : M ∩ N = {2, 3} :=
sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l2232_223284


namespace NUMINAMATH_GPT_total_potatoes_sold_is_322kg_l2232_223215

-- Define the given conditions
def bags_morning := 29
def bags_afternoon := 17
def weight_per_bag := 7

-- The theorem to prove the total kilograms sold is 322kg
theorem total_potatoes_sold_is_322kg : (bags_morning + bags_afternoon) * weight_per_bag = 322 :=
by
  sorry -- Placeholder for the actual proof

end NUMINAMATH_GPT_total_potatoes_sold_is_322kg_l2232_223215


namespace NUMINAMATH_GPT_repeatingDecimal_as_fraction_l2232_223204

def repeatingDecimal : ℚ := 0.136513513513

theorem repeatingDecimal_as_fraction : repeatingDecimal = 136377 / 999000 := 
by 
  sorry

end NUMINAMATH_GPT_repeatingDecimal_as_fraction_l2232_223204


namespace NUMINAMATH_GPT_convert_yahs_to_bahs_l2232_223288

noncomputable section

def bahs_to_rahs (bahs : ℕ) : ℕ := bahs * (36/24)
def rahs_to_bahs (rahs : ℕ) : ℕ := rahs * (24/36)
def rahs_to_yahs (rahs : ℕ) : ℕ := rahs * (18/12)
def yahs_to_rahs (yahs : ℕ) : ℕ := yahs * (12/18)
def yahs_to_bahs (yahs : ℕ) : ℕ := rahs_to_bahs (yahs_to_rahs yahs)

theorem convert_yahs_to_bahs :
  yahs_to_bahs 1500 = 667 :=
sorry

end NUMINAMATH_GPT_convert_yahs_to_bahs_l2232_223288


namespace NUMINAMATH_GPT_isosceles_triangle_angle_sum_l2232_223285

theorem isosceles_triangle_angle_sum (x : ℝ) (h1 : x = 50 ∨ x = 65 ∨ x = 80) : (50 + 65 + 80 = 195) :=
by sorry

end NUMINAMATH_GPT_isosceles_triangle_angle_sum_l2232_223285


namespace NUMINAMATH_GPT_twentieth_century_years_as_powers_of_two_diff_l2232_223282

theorem twentieth_century_years_as_powers_of_two_diff :
  ∀ (y : ℕ), (1900 ≤ y ∧ y < 2000) →
    ∃ (n k : ℕ), y = 2^n - 2^k ↔ y = 1984 ∨ y = 1920 := 
by
  sorry

end NUMINAMATH_GPT_twentieth_century_years_as_powers_of_two_diff_l2232_223282


namespace NUMINAMATH_GPT_cos_alpha_value_l2232_223250

theorem cos_alpha_value (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos (α + π / 4) = 4 / 5) :
  Real.cos α = 7 * Real.sqrt 2 / 10 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_value_l2232_223250


namespace NUMINAMATH_GPT_number_of_pages_in_each_chapter_l2232_223238

variable (x : ℕ)  -- Variable for number of pages in each chapter

-- Definitions based on the problem conditions
def pages_read_before_4_o_clock := 10 * x
def pages_read_at_4_o_clock := 20
def pages_read_after_4_o_clock := 2 * x
def total_pages_read := pages_read_before_4_o_clock x + pages_read_at_4_o_clock + pages_read_after_4_o_clock x

-- The theorem statement
theorem number_of_pages_in_each_chapter (h : total_pages_read x = 500) : x = 40 :=
sorry

end NUMINAMATH_GPT_number_of_pages_in_each_chapter_l2232_223238


namespace NUMINAMATH_GPT_distinct_paths_in_grid_l2232_223243

def number_of_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

theorem distinct_paths_in_grid :
  number_of_paths 7 8 = 6435 :=
by
  sorry

end NUMINAMATH_GPT_distinct_paths_in_grid_l2232_223243


namespace NUMINAMATH_GPT_find_f_2023_4_l2232_223213

noncomputable def f : ℕ → ℝ → ℝ
| 0, x => 2 * Real.sqrt x
| (n + 1), x => 4 / (2 - f n x)

theorem find_f_2023_4 : f 2023 4 = -2 := sorry

end NUMINAMATH_GPT_find_f_2023_4_l2232_223213


namespace NUMINAMATH_GPT_factorization_correct_l2232_223212

theorem factorization_correct (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) :=
by 
  sorry

end NUMINAMATH_GPT_factorization_correct_l2232_223212


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l2232_223218

-- Define the sides of the isosceles triangle
def side1 : ℝ := 4
def side2 : ℝ := 8

-- Hypothesis: The perimeter of an isosceles triangle with the given sides
-- Given condition
def is_isosceles_triangle (a b c : ℝ) : Prop := (a = b ∨ b = c ∨ c = a) ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = side1 ∨ a = side2) (h2 : b = side1 ∨ b = side2) :
  ∃ p : ℝ, is_isosceles_triangle a b side2 ∧ p = a + b + side2 → p = 20 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l2232_223218


namespace NUMINAMATH_GPT_find_b_value_l2232_223261

-- Definitions based on given conditions
def original_line (x : ℝ) (b : ℝ) : ℝ := 2 * x + b
def shifted_line (x : ℝ) (b : ℝ) : ℝ := 2 * (x - 2) + b
def passes_through_origin (b : ℝ) := shifted_line 0 b = 0

-- Main proof statement
theorem find_b_value (b : ℝ) (h : passes_through_origin b) : b = 4 := by
  sorry

end NUMINAMATH_GPT_find_b_value_l2232_223261


namespace NUMINAMATH_GPT_no_such_arrangement_exists_l2232_223217

theorem no_such_arrangement_exists :
  ¬ ∃ (f : ℕ → ℕ) (c : ℕ), 
    (∀ n, 1 ≤ f n ∧ f n ≤ 1331) ∧
    (∀ x y z, f (x + 11 * y + 121 * z) = c → f ((x+1) + 11 * y + 121 * z) = c + 8) ∧
    (∀ x y z, f (x + 11 * y + 121 * z) = c → f (x + 11 * (y+1) + 121 * z) = c + 9) :=
sorry

end NUMINAMATH_GPT_no_such_arrangement_exists_l2232_223217


namespace NUMINAMATH_GPT_month_length_l2232_223246

def treats_per_day : ℕ := 2
def cost_per_treat : ℝ := 0.1
def total_cost : ℝ := 6

theorem month_length : (total_cost / cost_per_treat) / treats_per_day = 30 := by
  sorry

end NUMINAMATH_GPT_month_length_l2232_223246


namespace NUMINAMATH_GPT_polynomial_roots_l2232_223249

theorem polynomial_roots :
  (∀ x : ℝ, (x^3 - 2*x^2 - 5*x + 6 = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3)) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_roots_l2232_223249


namespace NUMINAMATH_GPT_find_value_of_question_mark_l2232_223298

theorem find_value_of_question_mark (q : ℕ) : q * 40 = 173 * 240 → q = 1036 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_value_of_question_mark_l2232_223298


namespace NUMINAMATH_GPT_equal_focal_distances_l2232_223294

def ellipse1 (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1
def ellipse2 (k x y : ℝ) (hk : k < 9) : Prop := x^2 / (25 - k) + y^2 / (9 - k) = 1

theorem equal_focal_distances (k : ℝ) (hk : k < 9) : 
  let f1 := 8
  let f2 := 8 
  f1 = f2 :=
by 
  sorry

end NUMINAMATH_GPT_equal_focal_distances_l2232_223294


namespace NUMINAMATH_GPT_radius_of_congruent_spheres_in_cone_l2232_223277

noncomputable def radius_of_congruent_spheres (base_radius height : ℝ) : ℝ := 
  let slant_height := Real.sqrt (height^2 + base_radius^2)
  let r := (4 : ℝ) / (10 + 4) * slant_height
  r

theorem radius_of_congruent_spheres_in_cone :
  radius_of_congruent_spheres 4 10 = 4 * Real.sqrt 29 / 7 := by
  sorry

end NUMINAMATH_GPT_radius_of_congruent_spheres_in_cone_l2232_223277


namespace NUMINAMATH_GPT_probability_of_top_card_heart_l2232_223271

-- Define the total number of cards in the deck.
def total_cards : ℕ := 39

-- Define the number of hearts in the deck.
def hearts : ℕ := 13

-- Define the probability that the top card is a heart.
def probability_top_card_heart : ℚ := hearts / total_cards

-- State the theorem to prove.
theorem probability_of_top_card_heart : probability_top_card_heart = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_top_card_heart_l2232_223271


namespace NUMINAMATH_GPT_div_eq_implies_eq_l2232_223286

theorem div_eq_implies_eq (a b : ℕ) (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b :=
sorry

end NUMINAMATH_GPT_div_eq_implies_eq_l2232_223286


namespace NUMINAMATH_GPT_range_of_a_l2232_223226

theorem range_of_a (a : ℝ) (A : Set ℝ) (h : A = {x | a * x^2 - 3 * x + 1 = 0} ∧ ∃ (n : ℕ), 2 ^ n - 1 = 3) :
  a ∈ Set.Ioo (-(1:ℝ)/0) 0 ∪ Set.Ioo 0 (9 / 4) :=
sorry

end NUMINAMATH_GPT_range_of_a_l2232_223226


namespace NUMINAMATH_GPT_total_digits_l2232_223291

theorem total_digits (n S S6 S4 : ℕ) 
  (h1 : S = 80 * n)
  (h2 : S6 = 6 * 58)
  (h3 : S4 = 4 * 113)
  (h4 : S = S6 + S4) : 
  n = 10 :=
by 
  sorry

end NUMINAMATH_GPT_total_digits_l2232_223291


namespace NUMINAMATH_GPT_coconut_grove_average_yield_l2232_223260

theorem coconut_grove_average_yield :
  ∀ (x : ℕ),
  40 * (x + 2) + 120 * x + 180 * (x - 2) = 100 * 3 * x →
  x = 7 :=
by
  intro x
  intro h
  /- sorry proof -/
  sorry

end NUMINAMATH_GPT_coconut_grove_average_yield_l2232_223260


namespace NUMINAMATH_GPT_distance_to_mothers_house_l2232_223201

theorem distance_to_mothers_house 
  (D : ℝ) 
  (h1 : (2 / 3) * D = 156.0) : 
  D = 234.0 := 
sorry

end NUMINAMATH_GPT_distance_to_mothers_house_l2232_223201


namespace NUMINAMATH_GPT_smallest_rel_prime_l2232_223239

theorem smallest_rel_prime (n : ℕ) (h : n > 1) (rel_prime : ∀ p ∈ [2, 3, 5, 7], ¬ p ∣ n) : n = 11 :=
by sorry

end NUMINAMATH_GPT_smallest_rel_prime_l2232_223239


namespace NUMINAMATH_GPT_min_sum_abc_l2232_223223

theorem min_sum_abc (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1020) : a + b + c = 33 :=
sorry

end NUMINAMATH_GPT_min_sum_abc_l2232_223223


namespace NUMINAMATH_GPT_quadratic_has_single_solution_l2232_223235

theorem quadratic_has_single_solution (k : ℚ) : 
  (∀ x : ℚ, 3 * x^2 - 7 * x + k = 0 → x = 7 / 6) ↔ k = 49 / 12 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_has_single_solution_l2232_223235


namespace NUMINAMATH_GPT_intersection_eq_l2232_223229

def setM : Set ℝ := { x | x^2 - 2*x < 0 }
def setN : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

theorem intersection_eq : setM ∩ setN = { x | 0 < x ∧ x ≤ 1 } := sorry

end NUMINAMATH_GPT_intersection_eq_l2232_223229


namespace NUMINAMATH_GPT_radio_loss_percentage_l2232_223283

theorem radio_loss_percentage :
  ∀ (cost_price selling_price : ℝ), 
    cost_price = 1500 → 
    selling_price = 1290 → 
    ((cost_price - selling_price) / cost_price) * 100 = 14 :=
by
  intros cost_price selling_price h_cp h_sp
  sorry

end NUMINAMATH_GPT_radio_loss_percentage_l2232_223283


namespace NUMINAMATH_GPT_payment_plan_months_l2232_223248

theorem payment_plan_months 
  (M T : ℝ) (r : ℝ) 
  (hM : M = 100)
  (hT : T = 1320)
  (hr : r = 0.10)
  : ∃ t : ℕ, t = 12 ∧ T = (M * t) + (M * t * r) :=
by
  sorry

end NUMINAMATH_GPT_payment_plan_months_l2232_223248


namespace NUMINAMATH_GPT_find_expression_value_l2232_223210

theorem find_expression_value (x : ℝ) (h : x^2 - 3 * x - 1 = 0) : -3 * x^2 + 9 * x + 4 = 1 :=
by sorry

end NUMINAMATH_GPT_find_expression_value_l2232_223210


namespace NUMINAMATH_GPT_relay_race_solution_l2232_223209

variable (Sadie_time : ℝ) (Sadie_speed : ℝ)
variable (Ariana_time : ℝ) (Ariana_speed : ℝ)
variable (Sarah_speed : ℝ)
variable (total_distance : ℝ)

def relay_race_time : Prop :=
  let Sadie_distance := Sadie_time * Sadie_speed
  let Ariana_distance := Ariana_time * Ariana_speed
  let Sarah_distance := total_distance - Sadie_distance - Ariana_distance
  let Sarah_time := Sarah_distance / Sarah_speed
  Sadie_time + Ariana_time + Sarah_time = 4.5

theorem relay_race_solution (h1: Sadie_time = 2) (h2: Sadie_speed = 3)
  (h3: Ariana_time = 0.5) (h4: Ariana_speed = 6)
  (h5: Sarah_speed = 4) (h6: total_distance = 17) :
  relay_race_time Sadie_time Sadie_speed Ariana_time Ariana_speed Sarah_speed total_distance :=
by
  sorry

end NUMINAMATH_GPT_relay_race_solution_l2232_223209


namespace NUMINAMATH_GPT_lowest_test_score_dropped_l2232_223292

theorem lowest_test_score_dropped (A B C D : ℝ) 
  (h1: A + B + C + D = 280)
  (h2: A + B + C = 225) : D = 55 := 
by 
  sorry

end NUMINAMATH_GPT_lowest_test_score_dropped_l2232_223292


namespace NUMINAMATH_GPT_value_of_m_l2232_223276

theorem value_of_m (m : ℝ) : (m + 1, 3) ∈ {p : ℝ × ℝ | p.1 + p.2 + 1 = 0} → m = -5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_value_of_m_l2232_223276


namespace NUMINAMATH_GPT_remainder_of_22_divided_by_3_l2232_223281

theorem remainder_of_22_divided_by_3 : ∃ (r : ℕ), 22 = 3 * 7 + r ∧ r = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_of_22_divided_by_3_l2232_223281


namespace NUMINAMATH_GPT_correct_operation_l2232_223289

variables {x y : ℝ}

theorem correct_operation : -2 * x * 3 * y = -6 * x * y :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l2232_223289
