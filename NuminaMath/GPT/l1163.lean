import Mathlib

namespace NUMINAMATH_GPT_solve_for_x_l1163_116317

theorem solve_for_x (x : ℝ) (h : x ≠ 2) : (7 * x) / (x - 2) - 5 / (x - 2) = 3 / (x - 2) → x = 8 / 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1163_116317


namespace NUMINAMATH_GPT_solve_system_l1163_116350

def system_of_equations (x y : ℤ) : Prop :=
  (x^2 * y + x * y^2 + 3 * x + 3 * y + 24 = 0) ∧
  (x^3 * y - x * y^3 + 3 * x^2 - 3 * y^2 - 48 = 0)

theorem solve_system : system_of_equations (-3) (-1) :=
by {
  -- Proof details are omitted
  sorry
}

end NUMINAMATH_GPT_solve_system_l1163_116350


namespace NUMINAMATH_GPT_problem1_problem2_l1163_116358

-- Define the universe U
def U : Set ℝ := Set.univ

-- Define the sets A and B
def A : Set ℝ := { x | -4 < x ∧ x < 4 }
def B : Set ℝ := { x | x ≤ 1 ∨ x ≥ 3 }

-- Statement of the first proof problem: Prove A ∩ B is equal to the given set
theorem problem1 : A ∩ B = { x | -4 < x ∧ x ≤ 1 ∨ 4 > x ∧ x ≥ 3 } :=
by
  sorry

-- Statement of the second proof problem: Prove the complement of (A ∪ B) in the universe U is ∅
theorem problem2 : Set.compl (A ∪ B) = ∅ :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1163_116358


namespace NUMINAMATH_GPT_total_investment_sum_l1163_116337

theorem total_investment_sum :
  let R : ℝ := 2200
  let T : ℝ := R - 0.1 * R
  let V : ℝ := T + 0.1 * T
  R + T + V = 6358 := by
  sorry

end NUMINAMATH_GPT_total_investment_sum_l1163_116337


namespace NUMINAMATH_GPT_part1_part2_l1163_116364

noncomputable def f (x : ℝ) : ℝ := |x| + |x + 1|

theorem part1 (x : ℝ) : f x > 3 ↔ x > 1 ∨ x < -2 :=
by
  sorry

theorem part2 (m : ℝ) (hx : ∀ x : ℝ, m^2 + 3 * m + 2 * f x ≥ 0) : m ≤ -2 ∨ m ≥ -1 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1163_116364


namespace NUMINAMATH_GPT_f_g_relationship_l1163_116390

def f (x : ℝ) : ℝ := 3 * x ^ 2 - x + 1
def g (x : ℝ) : ℝ := 2 * x ^ 2 + x - 1

theorem f_g_relationship (x : ℝ) : f x > g x :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_f_g_relationship_l1163_116390


namespace NUMINAMATH_GPT_probability_of_green_l1163_116388

theorem probability_of_green : 
  ∀ (P_red P_orange P_yellow P_green : ℝ), 
    P_red = 0.25 → P_orange = 0.35 → P_yellow = 0.1 → 
    P_red + P_orange + P_yellow + P_green = 1 →
    P_green = 0.3 :=
by
  intros P_red P_orange P_yellow P_green h_red h_orange h_yellow h_total
  sorry

end NUMINAMATH_GPT_probability_of_green_l1163_116388


namespace NUMINAMATH_GPT_power_of_negative_125_l1163_116302

theorem power_of_negative_125 : (-125 : ℝ)^(4/3) = 625 := by
  sorry

end NUMINAMATH_GPT_power_of_negative_125_l1163_116302


namespace NUMINAMATH_GPT_total_cost_is_9_43_l1163_116391

def basketball_game_cost : ℝ := 5.20
def racing_game_cost : ℝ := 4.23
def total_cost : ℝ := basketball_game_cost + racing_game_cost

theorem total_cost_is_9_43 : total_cost = 9.43 := by
  sorry

end NUMINAMATH_GPT_total_cost_is_9_43_l1163_116391


namespace NUMINAMATH_GPT_simplify_expression_l1163_116374

variable {R : Type} [AddCommGroup R] [Module ℤ R]

theorem simplify_expression (a b : R) :
  (25 • a + 70 • b) + (15 • a + 34 • b) - (12 • a + 55 • b) = 28 • a + 49 • b :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1163_116374


namespace NUMINAMATH_GPT_max_abs_diff_f_l1163_116389

noncomputable def f (x : ℝ) : ℝ := (x + 1) ^ 2 * Real.exp x

theorem max_abs_diff_f {k x1 x2 : ℝ} (hk : -3 ≤ k ∧ k ≤ -1) 
    (hx1 : k ≤ x1 ∧ x1 ≤ k + 2) (hx2 : k ≤ x2 ∧ x2 ≤ k + 2) : 
    |f x1 - f x2| ≤ 4 * Real.exp 1 := 
sorry

end NUMINAMATH_GPT_max_abs_diff_f_l1163_116389


namespace NUMINAMATH_GPT_speed_of_A_l1163_116333

theorem speed_of_A (B_speed : ℕ) (crossings : ℕ) (H : B_speed = 3 ∧ crossings = 5 ∧ 5 * (1 / (x + B_speed)) = 1) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_A_l1163_116333


namespace NUMINAMATH_GPT_oliver_boxes_total_l1163_116378

theorem oliver_boxes_total (initial_boxes : ℕ := 8) (additional_boxes : ℕ := 6) : initial_boxes + additional_boxes = 14 := 
by 
  sorry

end NUMINAMATH_GPT_oliver_boxes_total_l1163_116378


namespace NUMINAMATH_GPT_product_sum_condition_l1163_116379

theorem product_sum_condition (a b c : ℝ) (h1 : a * b * c = 1) (h2 : a + b + c > (1/a) + (1/b) + (1/c)) : 
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
sorry

end NUMINAMATH_GPT_product_sum_condition_l1163_116379


namespace NUMINAMATH_GPT_find_unit_price_B_l1163_116387

variable (x : ℕ)

def unit_price_B := x
def unit_price_A := x + 50

theorem find_unit_price_B (h : (2000 / unit_price_A x = 1500 / unit_price_B x)) : unit_price_B x = 150 :=
by
  sorry

end NUMINAMATH_GPT_find_unit_price_B_l1163_116387


namespace NUMINAMATH_GPT_find_a2_plus_b2_l1163_116316

theorem find_a2_plus_b2 (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 15) : a^2 + b^2 = 39 :=
by
  sorry

end NUMINAMATH_GPT_find_a2_plus_b2_l1163_116316


namespace NUMINAMATH_GPT_friends_count_l1163_116342

noncomputable def university_students := 1995

theorem friends_count (students : ℕ)
  (knows_each_other : (ℕ → ℕ → Prop))
  (acquaintances : ℕ → ℕ)
  (h_university_students : students = university_students)
  (h_knows_iff_same_acq : ∀ a b, knows_each_other a b ↔ acquaintances a = acquaintances b)
  (h_not_knows_iff_diff_acq : ∀ a b, ¬ knows_each_other a b ↔ acquaintances a ≠ acquaintances b) :
  ∃ a, acquaintances a ≥ 62 ∧ ¬ ∃ a, acquaintances a ≥ 63 :=
sorry

end NUMINAMATH_GPT_friends_count_l1163_116342


namespace NUMINAMATH_GPT_fiona_reaches_pad_thirteen_without_predators_l1163_116356

noncomputable def probability_reach_pad_thirteen : ℚ := sorry

theorem fiona_reaches_pad_thirteen_without_predators :
  probability_reach_pad_thirteen = 3 / 2048 :=
sorry

end NUMINAMATH_GPT_fiona_reaches_pad_thirteen_without_predators_l1163_116356


namespace NUMINAMATH_GPT_trapezoid_shorter_base_length_l1163_116346

theorem trapezoid_shorter_base_length 
  (a b : ℕ) 
  (mid_segment_length longer_base : ℕ) 
  (h1 : mid_segment_length = 5) 
  (h2 : longer_base = 103) 
  (trapezoid_property : mid_segment_length = (longer_base - a) / 2) : 
  a = 93 := 
sorry

end NUMINAMATH_GPT_trapezoid_shorter_base_length_l1163_116346


namespace NUMINAMATH_GPT_candy_bar_cost_l1163_116315

theorem candy_bar_cost :
  ∃ C : ℕ, (C + 1 = 3) → (C = 2) :=
by
  use 2
  intros h
  linarith

end NUMINAMATH_GPT_candy_bar_cost_l1163_116315


namespace NUMINAMATH_GPT_megan_markers_l1163_116323

def initial_markers : ℕ := 217
def roberts_gift : ℕ := 109
def sarah_took : ℕ := 35

def final_markers : ℕ := initial_markers + roberts_gift - sarah_took

theorem megan_markers : final_markers = 291 := by
  sorry

end NUMINAMATH_GPT_megan_markers_l1163_116323


namespace NUMINAMATH_GPT_solution_set_ineq_l1163_116386

theorem solution_set_ineq (x : ℝ) : 
  x * (x + 2) > 0 → abs x < 1 → 0 < x ∧ x < 1 := by
sorry

end NUMINAMATH_GPT_solution_set_ineq_l1163_116386


namespace NUMINAMATH_GPT_minimum_questions_two_l1163_116371

structure Person :=
  (is_liar : Bool)

structure Decagon :=
  (people : Fin 10 → Person)

def minimumQuestionsNaive (d : Decagon) : Nat :=
  match d with 
  -- add the logic here later
  | _ => sorry

theorem minimum_questions_two (d : Decagon) : minimumQuestionsNaive d = 2 :=
  sorry

end NUMINAMATH_GPT_minimum_questions_two_l1163_116371


namespace NUMINAMATH_GPT_edward_final_money_l1163_116354

theorem edward_final_money 
  (spring_earnings : ℕ)
  (summer_earnings : ℕ)
  (supplies_cost : ℕ)
  (h_spring : spring_earnings = 2)
  (h_summer : summer_earnings = 27)
  (h_supplies : supplies_cost = 5)
  : spring_earnings + summer_earnings - supplies_cost = 24 := 
sorry

end NUMINAMATH_GPT_edward_final_money_l1163_116354


namespace NUMINAMATH_GPT_income_expenditure_ratio_l1163_116328

theorem income_expenditure_ratio
  (I : ℕ) (E : ℕ) (S : ℕ)
  (h1 : I = 18000)
  (h2 : S = 3600)
  (h3 : S = I - E) : I / E = 5 / 4 :=
by
  -- The actual proof is skipped.
  sorry

end NUMINAMATH_GPT_income_expenditure_ratio_l1163_116328


namespace NUMINAMATH_GPT_emily_did_not_sell_bars_l1163_116362

-- Definitions based on conditions
def cost_per_bar : ℕ := 4
def total_bars : ℕ := 8
def total_earnings : ℕ := 20

-- The statement to be proved
theorem emily_did_not_sell_bars :
  (total_bars - (total_earnings / cost_per_bar)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_emily_did_not_sell_bars_l1163_116362


namespace NUMINAMATH_GPT_pure_imaginary_implies_a_neg_one_l1163_116382

theorem pure_imaginary_implies_a_neg_one (a : ℝ) 
  (h_pure_imaginary : ∃ (y : ℝ), z = 0 + y * I) : 
  z = a + 1 - a * I → a = -1 :=
by
  sorry

end NUMINAMATH_GPT_pure_imaginary_implies_a_neg_one_l1163_116382


namespace NUMINAMATH_GPT_angle_between_hands_at_seven_l1163_116327

-- Define the conditions
def clock_parts := 12 -- The clock is divided into 12 parts
def degrees_per_part := 30 -- Each part is 30 degrees

-- Define the position of the hour and minute hands at 7:00 AM
def hour_position_at_seven := 7 -- Hour hand points to 7
def minute_position_at_seven := 0 -- Minute hand points to 12

-- Calculate the number of parts between the two positions
def parts_between_hands := if minute_position_at_seven = 0 then hour_position_at_seven else 12 - hour_position_at_seven

-- Calculate the angle between the hour hand and the minute hand at 7:00 AM
def angle_at_seven := degrees_per_part * parts_between_hands

-- State the theorem
theorem angle_between_hands_at_seven : angle_at_seven = 150 :=
by
  sorry

end NUMINAMATH_GPT_angle_between_hands_at_seven_l1163_116327


namespace NUMINAMATH_GPT_relation_between_a_b_c_l1163_116345

theorem relation_between_a_b_c :
  let a := (3/7 : ℝ) ^ (2/7)
  let b := (2/7 : ℝ) ^ (3/7)
  let c := (2/7 : ℝ) ^ (2/7)
  a > c ∧ c > b :=
by {
  let a := (3/7 : ℝ) ^ (2/7)
  let b := (2/7 : ℝ) ^ (3/7)
  let c := (2/7 : ℝ) ^ (2/7)
  sorry
}

end NUMINAMATH_GPT_relation_between_a_b_c_l1163_116345


namespace NUMINAMATH_GPT_Nancy_shelved_biographies_l1163_116351

def NancyBooks.shelved_books_from_top : Nat := 12 + 8 + 4 -- history + romance + poetry
def NancyBooks.total_books_on_cart : Nat := 46
def NancyBooks.bottom_books_after_top_shelved : Nat := 46 - 24
def NancyBooks.mystery_books_on_bottom : Nat := NancyBooks.bottom_books_after_top_shelved / 2
def NancyBooks.western_novels_on_bottom : Nat := 5
def NancyBooks.biographies : Nat := NancyBooks.bottom_books_after_top_shelved - NancyBooks.mystery_books_on_bottom - NancyBooks.western_novels_on_bottom

theorem Nancy_shelved_biographies : NancyBooks.biographies = 6 := by
  sorry

end NUMINAMATH_GPT_Nancy_shelved_biographies_l1163_116351


namespace NUMINAMATH_GPT_percentage_of_A_l1163_116375

-- Define variables and assumptions
variables (A B : ℕ)
def total_payment := 580
def payment_B := 232

-- Define the proofs of the conditions provided in the problem
axiom total_payment_eq : A + B = total_payment
axiom B_eq : B = payment_B
noncomputable def percentage_paid_to_A := (A / B) * 100

-- Theorem to prove the percentage of the payment to A compared to B
theorem percentage_of_A : percentage_paid_to_A = 150 :=
by
 sorry

end NUMINAMATH_GPT_percentage_of_A_l1163_116375


namespace NUMINAMATH_GPT_find_remainder_l1163_116313

variable (x y remainder : ℕ)
variable (h1 : x = 7 * y + 3)
variable (h2 : 2 * x = 18 * y + remainder)
variable (h3 : 11 * y - x = 1)

theorem find_remainder : remainder = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_remainder_l1163_116313


namespace NUMINAMATH_GPT_length_more_than_breadth_by_200_percent_l1163_116310

noncomputable def length: ℝ := 19.595917942265423
noncomputable def total_cost: ℝ := 640
noncomputable def rate_per_sq_meter: ℝ := 5

theorem length_more_than_breadth_by_200_percent
  (area : ℝ := total_cost / rate_per_sq_meter)
  (breadth : ℝ := area / length) :
  ((length - breadth) / breadth) * 100 = 200 := by
  have h1 : area = 128 := by sorry
  have h2 : breadth = 128 / 19.595917942265423 := by sorry
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_length_more_than_breadth_by_200_percent_l1163_116310


namespace NUMINAMATH_GPT_block_of_flats_l1163_116373

theorem block_of_flats :
  let total_floors := 12
  let half_floors := total_floors / 2
  let apartments_per_half_floor := 6
  let max_residents_per_apartment := 4
  let total_max_residents := 264
  let apartments_on_half_floors := half_floors * apartments_per_half_floor
  ∃ (x : ℝ), 
    4 * (apartments_on_half_floors + half_floors * x) = total_max_residents ->
    x = 5 :=
sorry

end NUMINAMATH_GPT_block_of_flats_l1163_116373


namespace NUMINAMATH_GPT_minimum_value_inequality_l1163_116305

theorem minimum_value_inequality {a b : ℝ} (h1 : b > a) (h2 : a > 1) 
  (h3 : 3 * Real.log b / Real.log a + 2 * Real.log a / Real.log b = 7) :
  a^2 + 3 / (b - 1) ≥ 2 * Real.sqrt 3 + 1 :=
sorry

end NUMINAMATH_GPT_minimum_value_inequality_l1163_116305


namespace NUMINAMATH_GPT_levels_for_blocks_l1163_116398

theorem levels_for_blocks (S : ℕ → ℕ) (n : ℕ) (h1 : S n = n * (n + 1)) (h2 : S 10 = 110) : n = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_levels_for_blocks_l1163_116398


namespace NUMINAMATH_GPT_number_of_lizards_l1163_116303

theorem number_of_lizards (total_geckos : ℕ) (insects_per_gecko : ℕ) (total_insects_eaten : ℕ) (insects_per_lizard : ℕ) 
  (gecko_total_insects : total_geckos * insects_per_gecko = 5 * 6) (lizard_insects: insects_per_lizard = 2 * insects_per_gecko)
  (total_insects : total_insects_eaten = 66) : 
  (total_insects_eaten - total_geckos * insects_per_gecko) / insects_per_lizard = 3 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_lizards_l1163_116303


namespace NUMINAMATH_GPT_collinear_c1_c2_l1163_116380

def vec3 := (ℝ × ℝ × ℝ)

def a : vec3 := (8, 3, -1)
def b : vec3 := (4, 1, 3)

def c1 : vec3 := (2 * 8 - 4, 2 * 3 - 1, 2 * (-1) - 3) -- (12, 5, -5)
def c2 : vec3 := (2 * 4 - 4 * 8, 2 * 1 - 4 * 3, 2 * 3 - 4 * (-1)) -- (-24, -10, 10)

theorem collinear_c1_c2 : ∃ γ : ℝ, c1 = (γ * -24, γ * -10, γ * 10) :=
  sorry

end NUMINAMATH_GPT_collinear_c1_c2_l1163_116380


namespace NUMINAMATH_GPT_second_alloy_amount_l1163_116319

theorem second_alloy_amount (x : ℝ) :
  (0.12 * 15 + 0.08 * x = 0.092 * (15 + x)) → x = 35 :=
by
  sorry

end NUMINAMATH_GPT_second_alloy_amount_l1163_116319


namespace NUMINAMATH_GPT_functional_equation_solution_l1163_116384

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x, 2 * f (f x) = (x^2 - x) * f x + 4 - 2 * x) :
  f 2 = 2 ∧ (f 1 = 1 ∨ f 1 = 4) :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l1163_116384


namespace NUMINAMATH_GPT_problem_statement_l1163_116340

variable {S R p a b c : ℝ}
variable (τ τ_a τ_b τ_c : ℝ)

theorem problem_statement
  (h1: S = τ * p)
  (h2: S = τ_a * (p - a))
  (h3: S = τ_b * (p - b))
  (h4: S = τ_c * (p - c))
  (h5: τ = S / p)
  (h6: τ_a = S / (p - a))
  (h7: τ_b = S / (p - b))
  (h8: τ_c = S / (p - c))
  (h9: abc / S = 4 * R) :
  1 / τ^3 - 1 / τ_a^3 - 1 / τ_b^3 - 1 / τ_c^3 = 12 * R / S^2 :=
  sorry

end NUMINAMATH_GPT_problem_statement_l1163_116340


namespace NUMINAMATH_GPT_average_age_of_omi_kimiko_arlette_l1163_116359

theorem average_age_of_omi_kimiko_arlette (Kimiko Omi Arlette : ℕ) (hK : Kimiko = 28) (hO : Omi = 2 * Kimiko) (hA : Arlette = (3 * Kimiko) / 4) : 
  (Omi + Kimiko + Arlette) / 3 = 35 := 
by
  sorry

end NUMINAMATH_GPT_average_age_of_omi_kimiko_arlette_l1163_116359


namespace NUMINAMATH_GPT_binomial_square_l1163_116336

theorem binomial_square (a b : ℝ) : (2 * a - 3 * b)^2 = 4 * a^2 - 12 * a * b + 9 * b^2 :=
by
  sorry

end NUMINAMATH_GPT_binomial_square_l1163_116336


namespace NUMINAMATH_GPT_original_number_is_80_l1163_116353

theorem original_number_is_80 (t : ℝ) (h : t * 1.125 - t * 0.75 = 30) : t = 80 := by
  sorry

end NUMINAMATH_GPT_original_number_is_80_l1163_116353


namespace NUMINAMATH_GPT_value_of_x_l1163_116367

noncomputable def k := 9

theorem value_of_x (y : ℝ) (h1 : y = 3) (h2 : ∀ (x : ℝ), x = 2.25 → x = k / (2 : ℝ)^2) : 
  ∃ (x : ℝ), x = 1 := by
  sorry

end NUMINAMATH_GPT_value_of_x_l1163_116367


namespace NUMINAMATH_GPT_slope_tangent_line_at_x1_l1163_116348

def f (x c : ℝ) : ℝ := (x-2)*(x^2 + c)
def f_prime (x c : ℝ) := (x^2 + c) + (x-2) * 2 * x

theorem slope_tangent_line_at_x1 (c : ℝ) (h : f_prime 2 c = 0) : f_prime 1 c = -5 := by
  sorry

end NUMINAMATH_GPT_slope_tangent_line_at_x1_l1163_116348


namespace NUMINAMATH_GPT_cubic_inequality_solution_l1163_116383

theorem cubic_inequality_solution :
  ∀ x : ℝ, (x + 1) * (x + 2)^2 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 := 
by 
  sorry

end NUMINAMATH_GPT_cubic_inequality_solution_l1163_116383


namespace NUMINAMATH_GPT_preimage_of_3_1_l1163_116366

theorem preimage_of_3_1 (a b : ℝ) (f : ℝ × ℝ → ℝ × ℝ) (h : ∀ (a b : ℝ), f (a, b) = (a + 2 * b, 2 * a - b)) :
  f (1, 1) = (3, 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_preimage_of_3_1_l1163_116366


namespace NUMINAMATH_GPT_geometric_seq_min_3b2_7b3_l1163_116349

theorem geometric_seq_min_3b2_7b3 (b_1 b_2 b_3 : ℝ) (r : ℝ) 
  (h_seq : b_1 = 2) (h_geom : b_2 = b_1 * r) (h_geom2 : b_3 = b_1 * r^2) :
  3 * b_2 + 7 * b_3 ≥ -16 / 7 :=
by
  -- Include the necessary definitions to support the setup
  have h_b1 : b_1 = 2 := h_seq
  have h_b2 : b_2 = 2 * r := by rw [h_geom, h_b1]
  have h_b3 : b_3 = 2 * r^2 := by rw [h_geom2, h_b1]
  sorry

end NUMINAMATH_GPT_geometric_seq_min_3b2_7b3_l1163_116349


namespace NUMINAMATH_GPT_geometric_seq_min_value_l1163_116325

theorem geometric_seq_min_value (b : ℕ → ℝ) (s : ℝ) (h1 : b 1 = 1) (h2 : ∀ n : ℕ, b (n + 1) = s * b n) : 
  ∃ s : ℝ, 3 * b 1 + 4 * b 2 = -9 / 16 :=
by
  sorry

end NUMINAMATH_GPT_geometric_seq_min_value_l1163_116325


namespace NUMINAMATH_GPT_smallest_s_for_F_l1163_116341

def F (a b c d : ℕ) : ℕ := a * b^(c^d)

theorem smallest_s_for_F :
  ∃ s : ℕ, F s s 2 2 = 65536 ∧ ∀ t : ℕ, F t t 2 2 = 65536 → s ≤ t :=
sorry

end NUMINAMATH_GPT_smallest_s_for_F_l1163_116341


namespace NUMINAMATH_GPT_jonathan_needs_12_bottles_l1163_116385

noncomputable def fl_oz_to_liters (fl_oz : ℝ) : ℝ :=
  fl_oz / 33.8

noncomputable def liters_to_ml (liters : ℝ) : ℝ :=
  liters * 1000

noncomputable def num_bottles_needed (ml : ℝ) : ℝ :=
  ml / 150

theorem jonathan_needs_12_bottles :
  num_bottles_needed (liters_to_ml (fl_oz_to_liters 60)) = 12 := 
by
  sorry

end NUMINAMATH_GPT_jonathan_needs_12_bottles_l1163_116385


namespace NUMINAMATH_GPT_sin_cos_sixth_power_sum_l1163_116355

theorem sin_cos_sixth_power_sum (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 2) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 0.8125 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_sixth_power_sum_l1163_116355


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1163_116338

theorem sufficient_but_not_necessary (a b : ℝ) :
  (a > 2 ∧ b > 1) → (a + b > 3 ∧ a * b > 2) ∧ ¬((a + b > 3 ∧ a * b > 2) → (a > 2 ∧ b > 1)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1163_116338


namespace NUMINAMATH_GPT_final_cards_l1163_116324

def initial_cards : ℝ := 47.0
def lost_cards : ℝ := 7.0

theorem final_cards : (initial_cards - lost_cards) = 40.0 :=
by
  sorry

end NUMINAMATH_GPT_final_cards_l1163_116324


namespace NUMINAMATH_GPT_mitch_family_milk_l1163_116365

variable (total_milk soy_milk regular_milk : ℚ)

-- Conditions
axiom cond1 : total_milk = 0.6
axiom cond2 : soy_milk = 0.1
axiom cond3 : regular_milk + soy_milk = total_milk

-- Theorem statement
theorem mitch_family_milk : regular_milk = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_mitch_family_milk_l1163_116365


namespace NUMINAMATH_GPT_slope_of_line_AB_is_pm_4_3_l1163_116361

noncomputable def slope_of_line_AB : ℝ := sorry

theorem slope_of_line_AB_is_pm_4_3 (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : y₁^2 = 4 * x₁)
  (h₂ : y₂^2 = 4 * x₂)
  (h₃ : (x₁, y₁) ≠ (x₂, y₂))
  (h₄ : (x₁ - 1, y₁) = -4 * (x₂ - 1, y₂)) :
  slope_of_line_AB = 4 / 3 ∨ slope_of_line_AB = -4 / 3 :=
sorry

end NUMINAMATH_GPT_slope_of_line_AB_is_pm_4_3_l1163_116361


namespace NUMINAMATH_GPT_prob_fourth_black_ball_is_half_l1163_116300

-- Define the conditions
def num_red_balls : ℕ := 4
def num_black_balls : ℕ := 4
def total_balls : ℕ := num_red_balls + num_black_balls

-- The theorem stating that the probability of drawing a black ball on the fourth draw is 1/2
theorem prob_fourth_black_ball_is_half : 
  (num_black_balls : ℚ) / (total_balls : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_prob_fourth_black_ball_is_half_l1163_116300


namespace NUMINAMATH_GPT_selling_price_is_80000_l1163_116339

-- Given the conditions of the problem
def purchasePrice : ℕ := 45000
def repairCosts : ℕ := 12000
def profitPercent : ℚ := 40.35 / 100

-- Total cost calculation
def totalCost := purchasePrice + repairCosts

-- Profit calculation
def profit := profitPercent * totalCost

-- Selling price calculation
def sellingPrice := totalCost + profit

-- Statement of the proof problem
theorem selling_price_is_80000 : round sellingPrice = 80000 := by
  sorry

end NUMINAMATH_GPT_selling_price_is_80000_l1163_116339


namespace NUMINAMATH_GPT_one_fourth_of_6_8_is_fraction_l1163_116360

theorem one_fourth_of_6_8_is_fraction :
  (6.8 / 4 : ℚ) = 17 / 10 :=
sorry

end NUMINAMATH_GPT_one_fourth_of_6_8_is_fraction_l1163_116360


namespace NUMINAMATH_GPT_rosie_pies_l1163_116301

theorem rosie_pies (apples_per_pie : ℕ) (apples_total : ℕ) (pies_initial : ℕ) 
  (h1 : 3 = pies_initial) (h2 : 12 = apples_total) : 
  (36 / (apples_total / pies_initial)) * pies_initial = 27 := 
by
  sorry

end NUMINAMATH_GPT_rosie_pies_l1163_116301


namespace NUMINAMATH_GPT_largest_integer_value_n_l1163_116377

theorem largest_integer_value_n (n : ℤ) : 
  (n^2 - 9 * n + 18 < 0) → n ≤ 5 := sorry

end NUMINAMATH_GPT_largest_integer_value_n_l1163_116377


namespace NUMINAMATH_GPT_pow_mod_cycle_l1163_116322

theorem pow_mod_cycle (n : ℕ) : 3^250 % 13 = 3 := 
by
  sorry

end NUMINAMATH_GPT_pow_mod_cycle_l1163_116322


namespace NUMINAMATH_GPT_class_average_correct_l1163_116372

-- Define the constants as per the problem data
def total_students : ℕ := 30
def students_group_1 : ℕ := 24
def students_group_2 : ℕ := 6
def avg_score_group_1 : ℚ := 85 / 100  -- 85%
def avg_score_group_2 : ℚ := 92 / 100  -- 92%

-- Calculate total scores and averages based on the defined constants
def total_score_group_1 : ℚ := students_group_1 * avg_score_group_1
def total_score_group_2 : ℚ := students_group_2 * avg_score_group_2
def total_class_score : ℚ := total_score_group_1 + total_score_group_2
def class_average : ℚ := total_class_score / total_students

-- Goal: Prove that class_average is 86.4%
theorem class_average_correct : class_average = 86.4 / 100 := sorry

end NUMINAMATH_GPT_class_average_correct_l1163_116372


namespace NUMINAMATH_GPT_age_difference_l1163_116307

variable (A B C : ℕ)

def condition1 := C = B / 2
def condition2 := A + B + C = 22
def condition3 := B = 8

theorem age_difference (h1 : condition1 C B)
                       (h2 : condition2 A B C) 
                       (h3 : condition3 B) : A - B = 2 := by
  sorry

end NUMINAMATH_GPT_age_difference_l1163_116307


namespace NUMINAMATH_GPT_addition_of_decimals_l1163_116334

theorem addition_of_decimals (a b : ℚ) (h1 : a = 7.56) (h2 : b = 4.29) : a + b = 11.85 :=
by
  -- The proof will be provided here
  sorry

end NUMINAMATH_GPT_addition_of_decimals_l1163_116334


namespace NUMINAMATH_GPT_factorize_polynomial_l1163_116326

noncomputable def zeta : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

theorem factorize_polynomial :
  (zeta^3 = 1) ∧ (zeta^2 + zeta + 1 = 0) → (x : ℂ) → (x^15 + x^10 + x) = (x^3 - 1) * (x^12 + x^9 + x^6 + x^3 + 1)
:= sorry

end NUMINAMATH_GPT_factorize_polynomial_l1163_116326


namespace NUMINAMATH_GPT_flag_distance_false_l1163_116308

theorem flag_distance_false (track_length : ℕ) (num_flags : ℕ) (flag1_flagN : 2 ≤ num_flags)
  (h1 : track_length = 90) (h2 : num_flags = 10) :
  ¬ (track_length / (num_flags - 1) = 9) :=
by
  sorry

end NUMINAMATH_GPT_flag_distance_false_l1163_116308


namespace NUMINAMATH_GPT_klay_to_draymond_ratio_l1163_116392

-- Let us define the points earned by each player
def draymond_points : ℕ := 12
def curry_points : ℕ := 2 * draymond_points
def kelly_points : ℕ := 9
def durant_points : ℕ := 2 * kelly_points

-- Total points of the Golden State Team
def total_points_team : ℕ := 69

theorem klay_to_draymond_ratio :
  ∃ klay_points : ℕ,
    klay_points = total_points_team - (draymond_points + curry_points + kelly_points + durant_points) ∧
    klay_points / draymond_points = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_klay_to_draymond_ratio_l1163_116392


namespace NUMINAMATH_GPT_correct_log_conclusions_l1163_116332

variables {x₁ x₂ : ℝ} (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (h_diff : x₁ ≠ x₂)
noncomputable def f (x : ℝ) : ℝ := Real.log x

theorem correct_log_conclusions :
  ¬ (f (x₁ + x₂) = f x₁ * f x₂) ∧
  (f (x₁ * x₂) = f x₁ + f x₂) ∧
  ¬ ((f x₁ - f x₂) / (x₁ - x₂) < 0) ∧
  (f ((x₁ + x₂) / 2) > (f x₁ + f x₂) / 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_log_conclusions_l1163_116332


namespace NUMINAMATH_GPT_percentage_difference_l1163_116304

variable (p : ℝ) (j : ℝ) (t : ℝ)

def condition_1 := j = 0.75 * p
def condition_2 := t = 0.9375 * p

theorem percentage_difference : (j = 0.75 * p) → (t = 0.9375 * p) → ((t - j) / t * 100 = 20) :=
by
  intros h1 h2
  rw [h1, h2]
  -- This will use the derived steps from the solution, and ultimately show 20
  sorry

end NUMINAMATH_GPT_percentage_difference_l1163_116304


namespace NUMINAMATH_GPT_ratio_change_factor_is_5_l1163_116331

-- Definitions based on problem conditions
def original_bleach : ℕ := 4
def original_detergent : ℕ := 40
def original_water : ℕ := 100

-- Simplified original ratio
def original_bleach_ratio : ℕ := original_bleach / 4
def original_detergent_ratio : ℕ := original_detergent / 4
def original_water_ratio : ℕ := original_water / 4

-- Altered conditions
def altered_detergent : ℕ := 60
def altered_water : ℕ := 300

-- Simplified altered ratio of detergent to water
def altered_detergent_ratio : ℕ := altered_detergent / 60
def altered_water_ratio : ℕ := altered_water / 60

-- Proof that the ratio change factor is 5
theorem ratio_change_factor_is_5 : 
  (original_water_ratio / altered_water_ratio) = 5
  := by
    have original_detergent_ratio : ℕ := 10
    have original_water_ratio : ℕ := 25
    have altered_detergent_ratio : ℕ := 1
    have altered_water_ratio : ℕ := 5
    sorry

end NUMINAMATH_GPT_ratio_change_factor_is_5_l1163_116331


namespace NUMINAMATH_GPT_value_of_c_div_b_l1163_116394

theorem value_of_c_div_b (a b c : ℕ) (h1 : a = 0) (h2 : a < b) (h3 : b < c) 
  (h4 : b ≠ a + 1) (h5 : (a + b + c) / 3 = 4 * b) : c / b = 11 :=
by
  sorry

end NUMINAMATH_GPT_value_of_c_div_b_l1163_116394


namespace NUMINAMATH_GPT_problem_statement_l1163_116357

theorem problem_statement (x : ℝ) (h : 8 * x = 4) : 150 * (1 / x) = 300 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1163_116357


namespace NUMINAMATH_GPT_min_value_expression_l1163_116376

theorem min_value_expression (θ φ : ℝ) :
  ∃ (θ φ : ℝ), (3 * Real.cos θ + 6 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 6 * Real.cos φ - 18)^2 = 121 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1163_116376


namespace NUMINAMATH_GPT_lesser_fraction_l1163_116320

theorem lesser_fraction (x y : ℚ) (h₁ : x + y = 3 / 4) (h₂ : x * y = 1 / 8) : min x y = 1 / 4 :=
sorry

end NUMINAMATH_GPT_lesser_fraction_l1163_116320


namespace NUMINAMATH_GPT_correct_equation_l1163_116312

theorem correct_equation (a b : ℝ) : (a - b) ^ 3 * (b - a) ^ 4 = (a - b) ^ 7 :=
sorry

end NUMINAMATH_GPT_correct_equation_l1163_116312


namespace NUMINAMATH_GPT_louie_monthly_payment_l1163_116381

noncomputable def compound_interest_payment (P : ℝ) (r : ℝ) (n : ℕ) (t_months : ℕ) : ℝ :=
  let t_years := t_months / 12
  let A := P * (1 + r / ↑n)^(↑n * t_years)
  A / t_months

theorem louie_monthly_payment : compound_interest_payment 1000 0.10 1 3 = 444 :=
by
  sorry

end NUMINAMATH_GPT_louie_monthly_payment_l1163_116381


namespace NUMINAMATH_GPT_inverse_proportion_quadrants_l1163_116393

theorem inverse_proportion_quadrants (x : ℝ) (y : ℝ) (h : y = 6/x) : 
  (x > 0 -> y > 0) ∧ (x < 0 -> y < 0) := 
sorry

end NUMINAMATH_GPT_inverse_proportion_quadrants_l1163_116393


namespace NUMINAMATH_GPT_smallest_n_boxes_cookies_l1163_116368

theorem smallest_n_boxes_cookies (n : ℕ) (h : (17 * n - 1) % 12 = 0) : n = 5 :=
sorry

end NUMINAMATH_GPT_smallest_n_boxes_cookies_l1163_116368


namespace NUMINAMATH_GPT_trigonometric_identity_l1163_116330

theorem trigonometric_identity (x : ℝ) (h : Real.tan (3 * π - x) = 2) :
    (2 * Real.cos (x / 2) ^ 2 - Real.sin x - 1) / (Real.sin x + Real.cos x) = -3 := by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1163_116330


namespace NUMINAMATH_GPT_value_range_of_f_l1163_116370

open Set

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem value_range_of_f : {y : ℝ | ∃ x ∈ Icc (-2 : ℝ) (2 : ℝ), f x = y} = Icc (-1 : ℝ) 8 := 
by
  sorry

end NUMINAMATH_GPT_value_range_of_f_l1163_116370


namespace NUMINAMATH_GPT_jerry_total_hours_at_field_l1163_116369
-- Import the entire necessary library

-- Lean statement of the problem
theorem jerry_total_hours_at_field 
  (games_per_daughter : ℕ)
  (practice_hours_per_game : ℕ)
  (game_duration : ℕ)
  (daughters : ℕ)
  (h1: games_per_daughter = 8)
  (h2: practice_hours_per_game = 4)
  (h3: game_duration = 2)
  (h4: daughters = 2)
 : (game_duration * games_per_daughter * daughters + practice_hours_per_game * games_per_daughter * daughters) = 96 :=
by
  -- Proof not required, so we skip it with sorry
  sorry

end NUMINAMATH_GPT_jerry_total_hours_at_field_l1163_116369


namespace NUMINAMATH_GPT_present_age_of_son_l1163_116343

theorem present_age_of_son (F S : ℕ) (h1 : F = S + 24) (h2 : F + 2 = 2 * (S + 2)) : S = 22 := by
  sorry

end NUMINAMATH_GPT_present_age_of_son_l1163_116343


namespace NUMINAMATH_GPT_combined_prism_volume_is_66_l1163_116335

noncomputable def volume_of_combined_prisms
  (length_rect : ℝ) (width_rect : ℝ) (height_rect : ℝ)
  (base_tri : ℝ) (height_tri : ℝ) (length_tri : ℝ) : ℝ :=
  let volume_rect := length_rect * width_rect * height_rect
  let area_tri := (1 / 2) * base_tri * height_tri
  let volume_tri := area_tri * length_tri
  volume_rect + volume_tri

theorem combined_prism_volume_is_66 :
  volume_of_combined_prisms 6 4 2 3 3 4 = 66 := by
  sorry

end NUMINAMATH_GPT_combined_prism_volume_is_66_l1163_116335


namespace NUMINAMATH_GPT_tangent_line_eq_at_P_tangent_lines_through_P_l1163_116329

-- Define the function and point of interest
def f (x : ℝ) : ℝ := x^3
def P : ℝ × ℝ := (1, 1)

-- State the first part: equation of the tangent line at (1, 1)
theorem tangent_line_eq_at_P : 
  (∀ x : ℝ, x = P.1 → (f x) = P.2) → 
  (∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ y = f x ∧ x = 1 → y = 3 * x - 2) :=
sorry

-- State the second part: equations of tangent lines passing through (1, 1)
theorem tangent_lines_through_P :
  (∀ x : ℝ, x = P.1 → (f x) = P.2) → 
  (∀ (x₀ y₀ : ℝ), y₀ = x₀^3 → 
  (x₀ ≠ 1 → ∃ k : ℝ,  k = 3 * (x₀)^2 → 
  (∀ x y : ℝ, y = k * (x - 1) + 1 ∧ y = f x₀ → y = y₀))) → 
  (∃ m b m' b' : ℝ, 
    (¬ ∀ x : ℝ, ∀ y : ℝ, (y = m *x + b ∧ y = 3 * x - 2) → y = m' * x + b') ∧ 
    ((m = 3 ∧ b = -2) ∧ (m' = 3/4 ∧ b' = 1/4))) :=
sorry

end NUMINAMATH_GPT_tangent_line_eq_at_P_tangent_lines_through_P_l1163_116329


namespace NUMINAMATH_GPT_value_expression_l1163_116347

theorem value_expression (x : ℝ) (h : x = 1 + Real.sqrt 2) : x^4 - 4 * x^3 + 4 * x^2 + 4 = 5 :=
by
  sorry

end NUMINAMATH_GPT_value_expression_l1163_116347


namespace NUMINAMATH_GPT_problem_equivalent_statement_l1163_116396

-- Define the operations provided in the problem
inductive Operation
| add
| sub
| mul
| div

open Operation

-- Represents the given equation with the specified operation
def applyOperation (op : Operation) (a b : ℕ) : ℕ :=
  match op with
  | add => a + b
  | sub => a - b
  | mul => a * b
  | div => a / b

theorem problem_equivalent_statement : 
  (∀ (op : Operation), applyOperation op 8 2 - 5 + 7 - (3^2 - 4) ≠ 6) → (¬ ∃ op : Operation, applyOperation op 8 2 = 9) := 
by
  sorry

end NUMINAMATH_GPT_problem_equivalent_statement_l1163_116396


namespace NUMINAMATH_GPT_lines_non_intersect_l1163_116352

theorem lines_non_intersect (k : ℝ) : 
  (¬∃ t s : ℝ, (1 + 2 * t = -1 + 3 * s ∧ 3 - 5 * t = 4 + k * s)) → 
  k = -15 / 2 :=
by
  intro h
  -- Now left to define proving steps using sorry
  sorry

end NUMINAMATH_GPT_lines_non_intersect_l1163_116352


namespace NUMINAMATH_GPT_parabola_equation_l1163_116363

theorem parabola_equation (a b c : ℝ)
  (h_p : (a + b + c = 1))
  (h_q : (4 * a + 2 * b + c = -1))
  (h_tangent : (4 * a + b = 1)) :
  y = 3 * x^2 - 11 * x + 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_parabola_equation_l1163_116363


namespace NUMINAMATH_GPT_triangle_third_side_l1163_116309

noncomputable def length_of_third_side
  (a b : ℝ) (θ : ℝ) (cosθ : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2 - 2 * a * b * cosθ)

theorem triangle_third_side : 
  length_of_third_side 8 15 (Real.pi / 6) (Real.cos (Real.pi / 6)) = Real.sqrt (289 - 120 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_triangle_third_side_l1163_116309


namespace NUMINAMATH_GPT_b_should_pay_360_l1163_116321

theorem b_should_pay_360 :
  let total_cost : ℝ := 870
  let a_horses  : ℝ := 12
  let a_months  : ℝ := 8
  let b_horses  : ℝ := 16
  let b_months  : ℝ := 9
  let c_horses  : ℝ := 18
  let c_months  : ℝ := 6
  let a_horse_months := a_horses * a_months
  let b_horse_months := b_horses * b_months
  let c_horse_months := c_horses * c_months
  let total_horse_months := a_horse_months + b_horse_months + c_horse_months
  let cost_per_horse_month := total_cost / total_horse_months
  let b_cost := b_horse_months * cost_per_horse_month
  b_cost = 360 :=
by sorry

end NUMINAMATH_GPT_b_should_pay_360_l1163_116321


namespace NUMINAMATH_GPT_mat_inverse_sum_l1163_116395

theorem mat_inverse_sum (a b c d : ℝ)
  (h1 : -2 * a + 3 * d = 1)
  (h2 : a * c - 12 = 0)
  (h3 : -8 + b * d = 0)
  (h4 : 4 * c - 4 * b = 0)
  (abc : a = 3 * Real.sqrt 2)
  (bb : b = 2 * Real.sqrt 2)
  (cc : c = 2 * Real.sqrt 2)
  (dd : d = (1 + 6 * Real.sqrt 2) / 3) :
  a + b + c + d = 9 * Real.sqrt 2 + 1 / 3 := by
  sorry

end NUMINAMATH_GPT_mat_inverse_sum_l1163_116395


namespace NUMINAMATH_GPT_remaining_balance_is_correct_l1163_116399

def total_price (deposit amount sales_tax_rate discount_rate service_charge P : ℝ) :=
  let sales_tax := sales_tax_rate * P
  let price_after_tax := P + sales_tax
  let discount := discount_rate * price_after_tax
  let price_after_discount := price_after_tax - discount
  let total_price := price_after_discount + service_charge
  total_price

theorem remaining_balance_is_correct (deposit : ℝ) (amount_paid : ℝ) (sales_tax_rate : ℝ) (discount_rate : ℝ) (service_charge : ℝ)
  (P : ℝ) : deposit = 0.10 * P →
         amount_paid = 110 →
         sales_tax_rate = 0.15 →
         discount_rate = 0.05 →
         service_charge = 50 →
         total_price deposit amount_paid sales_tax_rate discount_rate service_charge P - amount_paid = 1141.75 :=
by
  sorry

end NUMINAMATH_GPT_remaining_balance_is_correct_l1163_116399


namespace NUMINAMATH_GPT_isabella_paint_area_l1163_116311

-- Lean 4 statement for the proof problem based on given conditions and question:
theorem isabella_paint_area :
  let length := 15
  let width := 12
  let height := 9
  let door_and_window_area := 80
  let number_of_bedrooms := 4
  (2 * (length * height) + 2 * (width * height) - door_and_window_area) * number_of_bedrooms = 1624 :=
by
  sorry

end NUMINAMATH_GPT_isabella_paint_area_l1163_116311


namespace NUMINAMATH_GPT_valentines_given_l1163_116314

-- Let x be the number of boys and y be the number of girls
variables (x y : ℕ)

-- Condition 1: the number of valentines is 28 more than the total number of students.
axiom valentines_eq : x * y = x + y + 28

-- Theorem: Prove that the total number of valentines given is 60.
theorem valentines_given : x * y = 60 :=
by
  sorry

end NUMINAMATH_GPT_valentines_given_l1163_116314


namespace NUMINAMATH_GPT_find_value_of_m_l1163_116318

-- Define the quadratic function and the values in the given table
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Given conditions
variables (a b c m : ℝ)
variables (h1 : quadratic_function a b c (-1) = m)
variables (h2 : quadratic_function a b c 0 = 2)
variables (h3 : quadratic_function a b c 1 = 1)
variables (h4 : quadratic_function a b c 2 = 2)
variables (h5 : quadratic_function a b c 3 = 5)
variables (h6 : quadratic_function a b c 4 = 10)

-- Theorem stating that the value of m is 5
theorem find_value_of_m : m = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_m_l1163_116318


namespace NUMINAMATH_GPT_shape_descibed_by_z_eq_c_in_cylindrical_coords_is_cylinder_l1163_116306

theorem shape_descibed_by_z_eq_c_in_cylindrical_coords_is_cylinder
  (c : ℝ)
  (r : ℝ)
  (θ : ℝ)
  (hr : r ≥ 0)
  (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi) :
  ∃ (x y z : ℝ), (z = c) ∧ 
    x = r * Real.cos θ ∧ 
    y = r * Real.sin θ :=
by
  sorry

end NUMINAMATH_GPT_shape_descibed_by_z_eq_c_in_cylindrical_coords_is_cylinder_l1163_116306


namespace NUMINAMATH_GPT_larger_integer_is_72_l1163_116397

theorem larger_integer_is_72 (x y : ℤ) (h1 : y = 4 * x) (h2 : (x + 6) * 3 = y) : y = 72 :=
sorry

end NUMINAMATH_GPT_larger_integer_is_72_l1163_116397


namespace NUMINAMATH_GPT_intersection_product_is_15_l1163_116344

-- Define the first circle equation as a predicate
def first_circle (x y : ℝ) : Prop :=
  x^2 - 4 * x + y^2 - 6 * y + 12 = 0

-- Define the second circle equation as a predicate
def second_circle (x y : ℝ) : Prop :=
  x^2 - 10 * x + y^2 - 6 * y + 34 = 0

-- The Lean statement for the proof problem
theorem intersection_product_is_15 :
  ∃ x y : ℝ, first_circle x y ∧ second_circle x y ∧ (x * y = 15) :=
by
  sorry

end NUMINAMATH_GPT_intersection_product_is_15_l1163_116344
