import Mathlib

namespace NUMINAMATH_GPT_election_votes_total_l1764_176400

theorem election_votes_total 
  (winner_votes : ℕ) (opponent1_votes opponent2_votes opponent3_votes : ℕ)
  (excess1 excess2 excess3 : ℕ)
  (h1 : winner_votes = opponent1_votes + excess1)
  (h2 : winner_votes = opponent2_votes + excess2)
  (h3 : winner_votes = opponent3_votes + excess3)
  (votes_winner : winner_votes = 195)
  (votes_opponent1 : opponent1_votes = 142)
  (votes_opponent2 : opponent2_votes = 116)
  (votes_opponent3 : opponent3_votes = 90)
  (he1 : excess1 = 53)
  (he2 : excess2 = 79)
  (he3 : excess3 = 105) :
  winner_votes + opponent1_votes + opponent2_votes + opponent3_votes = 543 :=
by sorry

end NUMINAMATH_GPT_election_votes_total_l1764_176400


namespace NUMINAMATH_GPT_complex_div_eq_i_l1764_176411

noncomputable def i := Complex.I

theorem complex_div_eq_i : (1 + i) / (1 - i) = i := 
by
  sorry

end NUMINAMATH_GPT_complex_div_eq_i_l1764_176411


namespace NUMINAMATH_GPT_kim_monthly_expenses_l1764_176495

-- Define the conditions

def initial_cost : ℝ := 25000
def monthly_revenue : ℝ := 4000
def payback_period : ℕ := 10

-- Define the proof statement
theorem kim_monthly_expenses :
  ∃ (E : ℝ), 
    (payback_period * (monthly_revenue - E) = initial_cost) → (E = 1500) :=
by
  sorry

end NUMINAMATH_GPT_kim_monthly_expenses_l1764_176495


namespace NUMINAMATH_GPT_find_digit_l1764_176408

theorem find_digit:
  ∃ d: ℕ, d < 1000 ∧ 1995 * d = 610470 :=
  sorry

end NUMINAMATH_GPT_find_digit_l1764_176408


namespace NUMINAMATH_GPT_tangent_parallel_x_axis_tangent_45_degrees_x_axis_l1764_176483

-- Condition: Define the curve
def curve (x : ℝ) : ℝ := x^2 - 1

-- Condition: Calculate derivative
def derivative_curve (x : ℝ) : ℝ := 2 * x

-- Part (a): Point where tangent is parallel to the x-axis
theorem tangent_parallel_x_axis :
  (∃ x y : ℝ, y = curve x ∧ derivative_curve x = 0 ∧ x = 0 ∧ y = -1) :=
  sorry

-- Part (b): Point where tangent forms a 45 degree angle with the x-axis
theorem tangent_45_degrees_x_axis :
  (∃ x y : ℝ, y = curve x ∧ derivative_curve x = 1 ∧ x = 1/2 ∧ y = -3/4) :=
  sorry

end NUMINAMATH_GPT_tangent_parallel_x_axis_tangent_45_degrees_x_axis_l1764_176483


namespace NUMINAMATH_GPT_percent_decrease_of_y_l1764_176477

theorem percent_decrease_of_y (k x y q : ℝ) (h_inv_prop : x * y = k) (h_pos : 0 < x ∧ 0 < y) (h_q : 0 < q) :
  let x' := x * (1 + q / 100)
  let y' := y * 100 / (100 + q)
  (y - y') / y * 100 = (100 * q) / (100 + q) :=
by
  sorry

end NUMINAMATH_GPT_percent_decrease_of_y_l1764_176477


namespace NUMINAMATH_GPT_evaluate_expression_l1764_176476

theorem evaluate_expression : ⌈(7 : ℝ) / 3⌉ + ⌊- (7 : ℝ) / 3⌋ = 0 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1764_176476


namespace NUMINAMATH_GPT_find_cheese_calories_l1764_176432

noncomputable def lettuce_calories := 50
noncomputable def carrots_calories := 2 * lettuce_calories
noncomputable def dressing_calories := 210

noncomputable def crust_calories := 600
noncomputable def pepperoni_calories := crust_calories / 3

noncomputable def total_salad_calories := lettuce_calories + carrots_calories + dressing_calories
noncomputable def total_pizza_calories (cheese_calories : ℕ) := crust_calories + pepperoni_calories + cheese_calories

theorem find_cheese_calories (consumed_calories : ℕ) (cheese_calories : ℕ) :
  consumed_calories = 330 →
  1/4 * total_salad_calories + 1/5 * total_pizza_calories cheese_calories = consumed_calories →
  cheese_calories = 400 := by
  sorry

end NUMINAMATH_GPT_find_cheese_calories_l1764_176432


namespace NUMINAMATH_GPT_avg_ballpoint_pens_per_day_l1764_176440

theorem avg_ballpoint_pens_per_day (bundles_sold : ℕ) (pens_per_bundle : ℕ) (days : ℕ) (total_pens : ℕ) (avg_per_day : ℕ) 
  (h1 : bundles_sold = 15)
  (h2 : pens_per_bundle = 40)
  (h3 : days = 5)
  (h4 : total_pens = bundles_sold * pens_per_bundle)
  (h5 : avg_per_day = total_pens / days) :
  avg_per_day = 120 :=
by
  -- placeholder proof
  sorry

end NUMINAMATH_GPT_avg_ballpoint_pens_per_day_l1764_176440


namespace NUMINAMATH_GPT_arithmetic_mean_of_4_and_16_l1764_176413

-- Define the arithmetic mean condition
def is_arithmetic_mean (a b x : ℝ) : Prop :=
  x = (a + b) / 2

-- Theorem to prove that x = 10 if it is the mean of 4 and 16
theorem arithmetic_mean_of_4_and_16 (x : ℝ) (h : is_arithmetic_mean 4 16 x) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_4_and_16_l1764_176413


namespace NUMINAMATH_GPT_count_final_numbers_l1764_176452

def initial_numbers : List ℕ := List.range' 1 20

def erased_even_numbers : List ℕ :=
  initial_numbers.filter (λ n => n % 2 ≠ 0)

def final_numbers : List ℕ :=
  erased_even_numbers.filter (λ n => n % 5 ≠ 4)

theorem count_final_numbers : final_numbers.length = 8 :=
by
  -- Here, we would proceed to prove the statement.
  sorry

end NUMINAMATH_GPT_count_final_numbers_l1764_176452


namespace NUMINAMATH_GPT_target_runs_l1764_176499

theorem target_runs (r1 r2 : ℝ) (o1 o2 : ℕ) (target : ℝ) :
  r1 = 3.6 ∧ o1 = 10 ∧ r2 = 6.15 ∧ o2 = 40 → target = (r1 * o1) + (r2 * o2) := by
  sorry

end NUMINAMATH_GPT_target_runs_l1764_176499


namespace NUMINAMATH_GPT_part1_l1764_176425

   noncomputable def sin_20_deg_sq : ℝ := (Real.sin (20 * Real.pi / 180))^2
   noncomputable def cos_80_deg_sq : ℝ := (Real.sin (10 * Real.pi / 180))^2
   noncomputable def sqrt3_sin20_cos80 : ℝ := Real.sqrt 3 * Real.sin (20 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)
   noncomputable def value : ℝ := sin_20_deg_sq + cos_80_deg_sq + sqrt3_sin20_cos80

   theorem part1 : value = 1 / 4 := by
     sorry
   
end NUMINAMATH_GPT_part1_l1764_176425


namespace NUMINAMATH_GPT_f_2023_value_l1764_176492

noncomputable def f : ℕ → ℝ := sorry

axiom f_condition (a b n : ℕ) (ha : a > 0) (hb : b > 0) (hn : 2^n = a + b) : f a + f b = n^2 + 1

theorem f_2023_value : f 2023 = 107 :=
by 
  sorry

end NUMINAMATH_GPT_f_2023_value_l1764_176492


namespace NUMINAMATH_GPT_find_2u_plus_3v_l1764_176451

theorem find_2u_plus_3v (u v : ℚ) (h1 : 5 * u - 6 * v = 28) (h2 : 3 * u + 5 * v = -13) :
  2 * u + 3 * v = -7767 / 645 := 
sorry

end NUMINAMATH_GPT_find_2u_plus_3v_l1764_176451


namespace NUMINAMATH_GPT_trains_cross_time_l1764_176410

theorem trains_cross_time (length : ℝ) (time1 time2 : ℝ) (speed1 speed2 relative_speed : ℝ) 
  (H1 : length = 120) 
  (H2 : time1 = 12) 
  (H3 : time2 = 20) 
  (H4 : speed1 = length / time1) 
  (H5 : speed2 = length / time2) 
  (H6 : relative_speed = speed1 + speed2) 
  (total_distance : ℝ) (H7 : total_distance = length + length) 
  (T : ℝ) (H8 : T = total_distance / relative_speed) :
  T = 15 := 
sorry

end NUMINAMATH_GPT_trains_cross_time_l1764_176410


namespace NUMINAMATH_GPT_inequality_reciprocal_of_negatives_l1764_176438

theorem inequality_reciprocal_of_negatives (a b : ℝ) (ha : a < b) (hb : b < 0) : (1 / a) > (1 / b) :=
sorry

end NUMINAMATH_GPT_inequality_reciprocal_of_negatives_l1764_176438


namespace NUMINAMATH_GPT_not_age_of_child_l1764_176497

theorem not_age_of_child (ages : Set ℕ) (h_ages : ∀ x ∈ ages, 4 ≤ x ∧ x ≤ 10) : 
  5 ∉ ages := by
  let number := 1122
  have h_number : number % 5 ≠ 0 := by decide
  have h_divisible : ∀ x ∈ ages, number % x = 0 := sorry
  exact sorry

end NUMINAMATH_GPT_not_age_of_child_l1764_176497


namespace NUMINAMATH_GPT_problem_integer_square_l1764_176419

theorem problem_integer_square 
  (a b c d A : ℤ) 
  (H1 : a^2 + A = b^2) 
  (H2 : c^2 + A = d^2) : 
  ∃ (k : ℕ), 2 * (a + b) * (c + d) * (a * c + b * d - A) = k^2 :=
by
  sorry

end NUMINAMATH_GPT_problem_integer_square_l1764_176419


namespace NUMINAMATH_GPT_ocean_depth_l1764_176402

/-
  Problem:
  Determine the depth of the ocean at the current location of the ship.
  
  Given conditions:
  - The signal sent by the echo sounder was received after 5 seconds.
  - The speed of sound in water is 1.5 km/s.

  Correct answer to prove:
  - The depth of the ocean is 3750 meters.
-/

theorem ocean_depth
  (v : ℝ) (t : ℝ) (depth : ℝ) 
  (hv : v = 1500) 
  (ht : t = 5) 
  (hdepth : depth = 3750) :
  depth = (v * t) / 2 :=
sorry

end NUMINAMATH_GPT_ocean_depth_l1764_176402


namespace NUMINAMATH_GPT_number_of_sides_of_polygon_24_deg_exterior_angle_l1764_176454

theorem number_of_sides_of_polygon_24_deg_exterior_angle :
  (∀ (n : ℕ), (∀ (k : ℕ), k = 360 / 24 → n = k)) :=
by
  sorry

end NUMINAMATH_GPT_number_of_sides_of_polygon_24_deg_exterior_angle_l1764_176454


namespace NUMINAMATH_GPT_chantal_gain_l1764_176426

variable (sweaters balls cost_selling cost_yarn total_gain : ℕ)

def chantal_knits_sweaters : Prop :=
  sweaters = 28 ∧
  balls = 4 ∧
  cost_yarn = 6 ∧
  cost_selling = 35 ∧
  total_gain = (sweaters * cost_selling) - (sweaters * balls * cost_yarn)

theorem chantal_gain : chantal_knits_sweaters sweaters balls cost_selling cost_yarn total_gain → total_gain = 308 :=
by sorry

end NUMINAMATH_GPT_chantal_gain_l1764_176426


namespace NUMINAMATH_GPT_sum_of_coordinates_of_A_l1764_176409

theorem sum_of_coordinates_of_A
  (A B C : ℝ × ℝ)
  (AC AB BC : ℝ)
  (h1 : AC / AB = 1 / 3)
  (h2 : BC / AB = 2 / 3)
  (hB : B = (2, 5))
  (hC : C = (5, 8)) :
  (A.1 + A.2) = 16 :=
sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_A_l1764_176409


namespace NUMINAMATH_GPT_integer_solution_exists_l1764_176403

theorem integer_solution_exists (a : ℤ) : 
  (∃ k : ℤ, 2 * a^2 = 7 * k + 2) ↔ (a % 7 = 1 ∨ a % 7 = 6) :=
by sorry

end NUMINAMATH_GPT_integer_solution_exists_l1764_176403


namespace NUMINAMATH_GPT_radius_of_cone_base_l1764_176478

theorem radius_of_cone_base {R : ℝ} {theta : ℝ} (hR : R = 6) (htheta : theta = 120) :
  ∃ r : ℝ, r = 2 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_cone_base_l1764_176478


namespace NUMINAMATH_GPT_find_k_l1764_176462

theorem find_k (m n : ℝ) 
  (h₁ : m = k * n + 5) 
  (h₂ : m + 2 = k * (n + 0.5) + 5) : 
  k = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1764_176462


namespace NUMINAMATH_GPT_find_remainder_l1764_176424

-- Given conditions
def dividend : ℕ := 144
def divisor : ℕ := 11
def quotient : ℕ := 13

-- Theorem statement
theorem find_remainder (dividend divisor quotient : ℕ) (h1 : dividend = divisor * quotient + 1):
  ∃ r, r = dividend % divisor := 
by 
  exists 1
  sorry

end NUMINAMATH_GPT_find_remainder_l1764_176424


namespace NUMINAMATH_GPT_best_fit_model_l1764_176488

theorem best_fit_model 
  (R2_model1 R2_model2 R2_model3 R2_model4 : ℝ)
  (h1 : R2_model1 = 0.976)
  (h2 : R2_model2 = 0.776)
  (h3 : R2_model3 = 0.076)
  (h4 : R2_model4 = 0.351) : 
  (R2_model1 > R2_model2) ∧ (R2_model1 > R2_model3) ∧ (R2_model1 > R2_model4) :=
by
  sorry

end NUMINAMATH_GPT_best_fit_model_l1764_176488


namespace NUMINAMATH_GPT_problem_complement_intersection_l1764_176423

open Set

-- Define the universal set U
def U : Set ℕ := {0, 2, 4, 6, 8, 10}

-- Define set A
def A : Set ℕ := {0, 2, 4, 6}

-- Define set B based on A
def B : Set ℕ := {x | x ∈ A ∧ x < 4}

-- Define the complement of set A within U
def complement_A_U : Set ℕ := U \ A

-- Define the complement of set B within U
def complement_B_U : Set ℕ := U \ B

-- Prove the given equations
theorem problem_complement_intersection :
  (complement_A_U = {8, 10}) ∧ (A ∩ complement_B_U = {4, 6}) := 
by
  sorry

end NUMINAMATH_GPT_problem_complement_intersection_l1764_176423


namespace NUMINAMATH_GPT_length_of_chord_l1764_176468

theorem length_of_chord {x1 x2 : ℝ} (h1 : ∃ (y : ℝ), y^2 = 8 * x1)
                                   (h2 : ∃ (y : ℝ), y^2 = 8 * x2)
                                   (h_midpoint : (x1 + x2) / 2 = 3) :
  x1 + x2 + 4 = 10 :=
sorry

end NUMINAMATH_GPT_length_of_chord_l1764_176468


namespace NUMINAMATH_GPT_hash_of_hash_of_hash_of_70_l1764_176414

def hash (N : ℝ) : ℝ := 0.4 * N + 2

theorem hash_of_hash_of_hash_of_70 : hash (hash (hash 70)) = 8 := by
  sorry

end NUMINAMATH_GPT_hash_of_hash_of_hash_of_70_l1764_176414


namespace NUMINAMATH_GPT_no_integer_roots_l1764_176422

theorem no_integer_roots (a b c : ℤ) (h1 : a ≠ 0) (h2 : a % 2 = 1) (h3 : b % 2 = 1) (h4 : c % 2 = 1) :
  ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_roots_l1764_176422


namespace NUMINAMATH_GPT_hamburger_cost_l1764_176457

variable (H : ℝ)

theorem hamburger_cost :
  (H + 2 + 3 = 20 - 11) → (H = 4) :=
by
  sorry

end NUMINAMATH_GPT_hamburger_cost_l1764_176457


namespace NUMINAMATH_GPT_horizontal_distance_parabola_l1764_176455

theorem horizontal_distance_parabola :
  ∀ x_p x_q : ℝ, 
  (x_p^2 + 3*x_p - 4 = 8) → 
  (x_q^2 + 3*x_q - 4 = 0) → 
  x_p ≠ x_q → 
  abs (x_p - x_q) = 2 :=
sorry

end NUMINAMATH_GPT_horizontal_distance_parabola_l1764_176455


namespace NUMINAMATH_GPT_cricket_avg_score_l1764_176448

theorem cricket_avg_score
  (avg_first_two : ℕ)
  (num_first_two : ℕ)
  (avg_all_five : ℕ)
  (num_all_five : ℕ)
  (avg_first_two_eq : avg_first_two = 40)
  (num_first_two_eq : num_first_two = 2)
  (avg_all_five_eq : avg_all_five = 22)
  (num_all_five_eq : num_all_five = 5) :
  ((num_all_five * avg_all_five - num_first_two * avg_first_two) / (num_all_five - num_first_two) = 10) :=
by
  sorry

end NUMINAMATH_GPT_cricket_avg_score_l1764_176448


namespace NUMINAMATH_GPT_expression_value_l1764_176439

theorem expression_value (x y : ℝ) (h : x - 2 * y = 3) : 1 - 2 * x + 4 * y = -5 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l1764_176439


namespace NUMINAMATH_GPT_total_sections_l1764_176418

theorem total_sections (boys girls gcd sections_boys sections_girls : ℕ) 
  (h_boys : boys = 408) 
  (h_girls : girls = 264) 
  (h_gcd: gcd = Nat.gcd boys girls)
  (h_sections_boys : sections_boys = boys / gcd)
  (h_sections_girls : sections_girls = girls / gcd)
  (h_total_sections : sections_boys + sections_girls = 28)
: sections_boys + sections_girls = 28 := by
  sorry

end NUMINAMATH_GPT_total_sections_l1764_176418


namespace NUMINAMATH_GPT_max_value_3absx_2absy_l1764_176473

theorem max_value_3absx_2absy (x y : ℝ) (h : x^2 + y^2 = 9) : 
  3 * abs x + 2 * abs y ≤ 9 :=
sorry

end NUMINAMATH_GPT_max_value_3absx_2absy_l1764_176473


namespace NUMINAMATH_GPT_notebook_and_pen_prices_l1764_176485

theorem notebook_and_pen_prices (x y : ℕ) (h1 : 2 * x + y = 30) (h2 : x = 2 * y) :
  x = 12 ∧ y = 6 :=
by
  sorry

end NUMINAMATH_GPT_notebook_and_pen_prices_l1764_176485


namespace NUMINAMATH_GPT_largest_sum_of_three_largest_angles_l1764_176445

-- Definitions and main theorem statement
theorem largest_sum_of_three_largest_angles (EFGH : Type*)
    (a b c d : ℝ) 
    (h1 : a + b + c + d = 360)
    (h2 : b = 3 * c)
    (h3 : ∃ (common_diff : ℝ), (c - a = common_diff) ∧ (b - c = common_diff) ∧ (d - b = common_diff))
    (h4 : ∀ (x y z : ℝ), (x = y + z) ↔ (∃ (progression_diff : ℝ), x - y = y - z ∧ y - z = z - x)) :
    (∃ (A B C D : ℝ), A = a ∧ B = b ∧ C = c ∧ D = d ∧ A + B + C + D = 360 ∧ A = max a (max b (max c d)) ∧ B = 2 * D ∧ A + B + C = 330) :=
sorry

end NUMINAMATH_GPT_largest_sum_of_three_largest_angles_l1764_176445


namespace NUMINAMATH_GPT_distance_upstream_l1764_176470

variable (v : ℝ) -- speed of the stream in km/h
variable (t : ℝ := 6) -- time of each trip in hours
variable (d_down : ℝ := 24) -- distance for downstream trip in km
variable (u : ℝ := 3) -- speed of man in still water in km/h

/- The distance the man swam upstream -/
theorem distance_upstream : 
  24 = (u + v) * t → 
  ∃ (d_up : ℝ), 
    d_up = (u - v) * t ∧
    d_up = 12 :=
by
  sorry

end NUMINAMATH_GPT_distance_upstream_l1764_176470


namespace NUMINAMATH_GPT_scientific_notation_of_million_l1764_176446

theorem scientific_notation_of_million (x : ℝ) (h : x = 2600000) : x = 2.6 * 10^6 := by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_million_l1764_176446


namespace NUMINAMATH_GPT_polynomial_value_at_2_l1764_176406

def f (x : ℤ) : ℤ := 7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

theorem polynomial_value_at_2:
  f 2 = 1538 := by
  sorry

end NUMINAMATH_GPT_polynomial_value_at_2_l1764_176406


namespace NUMINAMATH_GPT_ratio_of_shoppers_l1764_176444

theorem ratio_of_shoppers (boxes ordered_of_yams: ℕ) (packages_per_box shoppers total_shoppers: ℕ)
  (h1 : packages_per_box = 25)
  (h2 : ordered_of_yams = 5)
  (h3 : total_shoppers = 375)
  (h4 : shoppers = ordered_of_yams * packages_per_box):
  (shoppers : ℕ) / total_shoppers = 1 / 3 := 
sorry

end NUMINAMATH_GPT_ratio_of_shoppers_l1764_176444


namespace NUMINAMATH_GPT_sub_seq_arithmetic_l1764_176443

variable (a : ℕ → ℝ) (d : ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sub_seq (a : ℕ → ℝ) (k : ℕ) : ℝ :=
  a (3 * k - 1)

theorem sub_seq_arithmetic (h : is_arithmetic_sequence a d) : is_arithmetic_sequence (sub_seq a) (3 * d) := 
sorry


end NUMINAMATH_GPT_sub_seq_arithmetic_l1764_176443


namespace NUMINAMATH_GPT_simplify_expression_l1764_176441

theorem simplify_expression (x : ℝ) : 7 * x + 8 - 3 * x + 14 = 4 * x + 22 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1764_176441


namespace NUMINAMATH_GPT_mark_total_cost_is_correct_l1764_176459

variable (hours : ℕ) (hourly_rate part_cost : ℕ)

def total_cost (hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) :=
  hours * hourly_rate + part_cost

theorem mark_total_cost_is_correct : 
  hours = 2 → hourly_rate = 75 → part_cost = 150 → total_cost hours hourly_rate part_cost = 300 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_mark_total_cost_is_correct_l1764_176459


namespace NUMINAMATH_GPT_tadd_2019th_number_l1764_176437

def next_start_point (n : ℕ) : ℕ := 
    1 + (n * (2 * 3 + (n - 1) * 9)) / 2

def block_size (n : ℕ) : ℕ := 
    1 + 3 * (n - 1)

def nth_number_said_by_tadd (n : ℕ) (k : ℕ) : ℕ :=
    let block_n := next_start_point n
    block_n + k - 1

theorem tadd_2019th_number :
    nth_number_said_by_tadd 37 2019 = 5979 := 
sorry

end NUMINAMATH_GPT_tadd_2019th_number_l1764_176437


namespace NUMINAMATH_GPT_trapezoid_smallest_angle_l1764_176449

theorem trapezoid_smallest_angle (a d : ℝ) 
  (h1 : a + 3 * d = 140)
  (h2 : 2 * a + 3 * d = 180) : 
  a = 20 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_smallest_angle_l1764_176449


namespace NUMINAMATH_GPT_subtracted_value_l1764_176482

-- Given conditions
def chosen_number : ℕ := 110
def result_number : ℕ := 110

-- Statement to prove
theorem subtracted_value : ∃ y : ℕ, 3 * chosen_number - y = result_number ∧ y = 220 :=
by
  sorry

end NUMINAMATH_GPT_subtracted_value_l1764_176482


namespace NUMINAMATH_GPT_tan_identity_at_30_degrees_l1764_176421

theorem tan_identity_at_30_degrees :
  let A := 30
  let B := 30
  let deg_to_rad := pi / 180
  let tan := fun x : ℝ => Real.tan (x * deg_to_rad)
  (1 + tan A) * (1 + tan B) = (4 + 2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_GPT_tan_identity_at_30_degrees_l1764_176421


namespace NUMINAMATH_GPT_one_cow_one_bag_in_34_days_l1764_176431

-- Definitions: 34 cows eat 34 bags in 34 days, each cow eats one bag in those 34 days.
def cows : Nat := 34
def bags : Nat := 34
def days : Nat := 34

-- Hypothesis: each cow eats one bag in 34 days.
def one_bag_days (c : Nat) (b : Nat) : Nat := days

-- Theorem: One cow will eat one bag of husk in 34 days.
theorem one_cow_one_bag_in_34_days : one_bag_days 1 1 = 34 := sorry

end NUMINAMATH_GPT_one_cow_one_bag_in_34_days_l1764_176431


namespace NUMINAMATH_GPT_intersecting_circle_radius_l1764_176430

-- Definitions representing the conditions
def non_intersecting_circles (O_i : Fin 6 → ℕ) (r_i : Fin 6 → ℝ) : Prop :=
  ∀ i j, i ≠ j → dist (O_i i) (O_i j) ≥ r_i i + r_i j

def min_radius_one (O_i : Fin 6 → ℕ) (r_i : Fin 6 → ℝ) := 
  ∀ i, r_i i ≥ 1

-- The main theorem stating the proof goal
theorem intersecting_circle_radius 
  (O_i : Fin 6 → ℕ) (r_i : Fin 6 → ℝ) (O : ℕ) (r : ℝ)
  (h_non_intersecting : non_intersecting_circles O_i r_i)
  (h_min_radius : min_radius_one O_i r_i)
  (h_intersecting : ∀ i, dist O (O_i i) ≤ r + r_i i) :
  r ≥ 1 := 
sorry

end NUMINAMATH_GPT_intersecting_circle_radius_l1764_176430


namespace NUMINAMATH_GPT_monotonic_decreasing_interval_l1764_176461

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_decreasing_interval :
  {x : ℝ | x > 0} ∩ {x : ℝ | deriv f x < 0} = {x : ℝ | x > Real.exp 1} :=
by sorry

end NUMINAMATH_GPT_monotonic_decreasing_interval_l1764_176461


namespace NUMINAMATH_GPT_max_real_root_lt_100_l1764_176496

theorem max_real_root_lt_100 (k a b c : ℕ) (r : ℝ)
  (ha : ∃ m : ℕ, a = k^m)
  (hb : ∃ n : ℕ, b = k^n)
  (hc : ∃ l : ℕ, c = k^l)
  (one_real_solution : b^2 = 4 * a * c)
  (r_is_root : ∃ r : ℝ, a * r^2 - b * r + c = 0)
  (r_lt_100 : r < 100) :
  r ≤ 64 := sorry

end NUMINAMATH_GPT_max_real_root_lt_100_l1764_176496


namespace NUMINAMATH_GPT_inequality_solutions_l1764_176469

theorem inequality_solutions :
  (∀ x : ℝ, 2 * x / (x + 1) < 1 ↔ -1 < x ∧ x < 1) ∧
  (∀ a x : ℝ,
    (x^2 + (2 - a) * x - 2 * a ≥ 0 ↔
      (a = -2 → True) ∧
      (a > -2 → (x ≤ -2 ∨ x ≥ a)) ∧
      (a < -2 → (x ≤ a ∨ x ≥ -2)))) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solutions_l1764_176469


namespace NUMINAMATH_GPT_grid_diagonal_segments_l1764_176447

theorem grid_diagonal_segments (m n : ℕ) (hm : m = 100) (hn : n = 101) :
    let d := m + n - gcd m n
    d = 200 := by
  sorry

end NUMINAMATH_GPT_grid_diagonal_segments_l1764_176447


namespace NUMINAMATH_GPT_distance_from_point_to_circle_center_l1764_176474

noncomputable def polar_to_rect (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def circle_center : ℝ × ℝ := (1, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem distance_from_point_to_circle_center :
  distance (polar_to_rect 2 (Real.pi / 3)) circle_center = Real.sqrt 3 := sorry

end NUMINAMATH_GPT_distance_from_point_to_circle_center_l1764_176474


namespace NUMINAMATH_GPT_discount_is_25_l1764_176486

def original_price : ℕ := 76
def discounted_price : ℕ := 51
def discount_amount : ℕ := original_price - discounted_price

theorem discount_is_25 : discount_amount = 25 := by
  sorry

end NUMINAMATH_GPT_discount_is_25_l1764_176486


namespace NUMINAMATH_GPT_workers_work_5_days_a_week_l1764_176427

def total_weekly_toys : ℕ := 5500
def daily_toys : ℕ := 1100
def days_worked : ℕ := total_weekly_toys / daily_toys

theorem workers_work_5_days_a_week : days_worked = 5 := 
by 
  sorry

end NUMINAMATH_GPT_workers_work_5_days_a_week_l1764_176427


namespace NUMINAMATH_GPT_selling_price_same_loss_as_profit_l1764_176458

theorem selling_price_same_loss_as_profit (cost_price selling_price_with_profit selling_price_with_loss profit loss : ℝ)
  (h1 : selling_price_with_profit - cost_price = profit)
  (h2 : cost_price - selling_price_with_loss = loss)
  (h3 : profit = loss) :
  selling_price_with_loss = 52 :=
by
  have h4 : selling_price_with_profit = 66 := by sorry
  have h5 : cost_price = 59 := by sorry
  have h6 : profit = 66 - 59 := by sorry
  have h7 : profit = 7 := by sorry
  have h8 : loss = 59 - selling_price_with_loss := by sorry
  have h9 : loss = 7 := by sorry
  have h10 : selling_price_with_loss = 59 - loss := by sorry
  have h11 : selling_price_with_loss = 59 - 7 := by sorry
  have h12 : selling_price_with_loss = 52 := by sorry
  exact h12

end NUMINAMATH_GPT_selling_price_same_loss_as_profit_l1764_176458


namespace NUMINAMATH_GPT_smallest_four_digit_equivalent_6_mod_7_l1764_176401

theorem smallest_four_digit_equivalent_6_mod_7 :
  (∃ (n : ℕ), n >= 1000 ∧ n < 10000 ∧ n % 7 = 6 ∧ (∀ (m : ℕ), m >= 1000 ∧ m < 10000 ∧ m % 7 = 6 → m >= n)) ∧ ∃ (n : ℕ), n = 1000 :=
sorry

end NUMINAMATH_GPT_smallest_four_digit_equivalent_6_mod_7_l1764_176401


namespace NUMINAMATH_GPT_sectionB_seats_correct_l1764_176435

-- Definitions for the number of seats in Section A
def seatsA_subsection1 : Nat := 60
def seatsA_subsection2 : Nat := 3 * 80
def totalSeatsA : Nat := seatsA_subsection1 + seatsA_subsection2

-- Condition for the number of seats in Section B
def seatsB : Nat := 3 * totalSeatsA + 20

-- Theorem statement to prove the number of seats in Section B
theorem sectionB_seats_correct : seatsB = 920 := by
  sorry

end NUMINAMATH_GPT_sectionB_seats_correct_l1764_176435


namespace NUMINAMATH_GPT_region_area_proof_l1764_176404

noncomputable def region_area := 
  let region := {p : ℝ × ℝ | abs (p.1 - p.2^2 / 2) + p.1 + p.2^2 / 2 ≤ 2 - p.2}
  2 * (0.5 * (3 * (2 + 0.5)))

theorem region_area_proof : region_area = 15 / 2 :=
by
  sorry

end NUMINAMATH_GPT_region_area_proof_l1764_176404


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1764_176415

theorem sufficient_but_not_necessary (x : ℝ) : (x > 0 → x * (x + 1) > 0) ∧ ¬ (x * (x + 1) > 0 → x > 0) := 
by 
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1764_176415


namespace NUMINAMATH_GPT_find_y_l1764_176494

theorem find_y (x y : ℕ) (hx_positive : 0 < x) (hy_positive : 0 < y) (hmod : x % y = 9) (hdiv : (x : ℝ) / (y : ℝ) = 96.25) : y = 36 :=
sorry

end NUMINAMATH_GPT_find_y_l1764_176494


namespace NUMINAMATH_GPT_base3_composite_numbers_l1764_176464

theorem base3_composite_numbers:
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 12002110 = a * b) ∧ 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 2210121012 = a * b) ∧ 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 121212 = a * b) ∧ 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 102102 = a * b) ∧ 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 1001 * AB = a * b) :=
by {
  sorry
}

end NUMINAMATH_GPT_base3_composite_numbers_l1764_176464


namespace NUMINAMATH_GPT_no_primes_divisible_by_45_l1764_176472

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_primes_divisible_by_45 : 
  ∀ p, is_prime p → ¬ (45 ∣ p) := 
by
  sorry

end NUMINAMATH_GPT_no_primes_divisible_by_45_l1764_176472


namespace NUMINAMATH_GPT_total_beakers_count_l1764_176466

variable (total_beakers_with_ions : ℕ) 
variable (drops_per_test : ℕ)
variable (total_drops_used : ℕ) 
variable (beakers_without_ions : ℕ)

theorem total_beakers_count
  (h1 : total_beakers_with_ions = 8)
  (h2 : drops_per_test = 3)
  (h3 : total_drops_used = 45)
  (h4 : beakers_without_ions = 7) : 
  (total_drops_used / drops_per_test) = (total_beakers_with_ions + beakers_without_ions) :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_total_beakers_count_l1764_176466


namespace NUMINAMATH_GPT_work_rate_ab_l1764_176450

variables (A B C : ℝ)

-- Defining the work rates as per the conditions
def work_rate_bc := 1 / 6 -- (b and c together in 6 days)
def work_rate_ca := 1 / 3 -- (c and a together in 3 days)
def work_rate_c := 1 / 8 -- (c alone in 8 days)

-- The main theorem that proves a and b together can complete the work in 4 days,
-- based on the above conditions.
theorem work_rate_ab : 
  (B + C = work_rate_bc) ∧ (C + A = work_rate_ca) ∧ (C = work_rate_c) 
  → (A + B = 1 / 4) :=
by sorry

end NUMINAMATH_GPT_work_rate_ab_l1764_176450


namespace NUMINAMATH_GPT_count_unbroken_matches_l1764_176467

theorem count_unbroken_matches :
  let n_1 := 5 * 12  -- number of boxes in the first set
  let matches_1 := n_1 * 20  -- total matches in first set of boxes
  let broken_1 := n_1 * 3  -- total broken matches in first set of boxes
  let unbroken_1 := matches_1 - broken_1  -- unbroken matches in first set of boxes

  let n_2 := 4  -- number of extra boxes
  let matches_2 := n_2 * 25  -- total matches in extra boxes
  let broken_2 := (matches_2 / 5)  -- total broken matches in extra boxes (20%)
  let unbroken_2 := matches_2 - broken_2  -- unbroken matches in extra boxes

  let total_unbroken := unbroken_1 + unbroken_2  -- total unbroken matches

  total_unbroken = 1100 := 
by
  sorry

end NUMINAMATH_GPT_count_unbroken_matches_l1764_176467


namespace NUMINAMATH_GPT_sphere_segment_volume_l1764_176463

theorem sphere_segment_volume (r : ℝ) (ratio_surface_to_base : ℝ) : r = 10 → ratio_surface_to_base = 10 / 7 → ∃ V : ℝ, V = 288 * π :=
by
  intros
  sorry

end NUMINAMATH_GPT_sphere_segment_volume_l1764_176463


namespace NUMINAMATH_GPT_double_bed_heavier_than_single_bed_l1764_176433

theorem double_bed_heavier_than_single_bed 
  (S D : ℝ) 
  (h1 : 5 * S = 50) 
  (h2 : 2 * S + 4 * D = 100) 
  : D - S = 10 :=
sorry

end NUMINAMATH_GPT_double_bed_heavier_than_single_bed_l1764_176433


namespace NUMINAMATH_GPT_find_f_one_l1764_176498

noncomputable def f_inv (x : ℝ) : ℝ := 2^(x + 1)

theorem find_f_one : ∃ f : ℝ → ℝ, (∀ y, f (f_inv y) = y) ∧ f 1 = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_f_one_l1764_176498


namespace NUMINAMATH_GPT_polar_to_cartesian_l1764_176428

theorem polar_to_cartesian (ρ θ : ℝ) (h : ρ = 4 * Real.cos θ) :
  ∃ x y : ℝ, (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧
  (x - 2)^2 + y^2 = 4) :=
sorry

end NUMINAMATH_GPT_polar_to_cartesian_l1764_176428


namespace NUMINAMATH_GPT_find_y_from_x_squared_l1764_176416

theorem find_y_from_x_squared (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = 6) : y = 29 :=
by
  sorry

end NUMINAMATH_GPT_find_y_from_x_squared_l1764_176416


namespace NUMINAMATH_GPT_gcd_9155_4892_l1764_176479

theorem gcd_9155_4892 : Nat.gcd 9155 4892 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_9155_4892_l1764_176479


namespace NUMINAMATH_GPT_total_people_in_group_l1764_176460

theorem total_people_in_group (men women children : ℕ)
  (h1 : men = 2 * women)
  (h2 : women = 3 * children)
  (h3 : children = 30) :
  men + women + children = 300 :=
by
  sorry

end NUMINAMATH_GPT_total_people_in_group_l1764_176460


namespace NUMINAMATH_GPT_pascal_row_with_ratio_456_exists_at_98_l1764_176412

theorem pascal_row_with_ratio_456_exists_at_98 :
  ∃ n, ∃ r, 0 ≤ r ∧ r + 2 ≤ n ∧ 
  ((Nat.choose n r : ℚ) / Nat.choose n (r + 1) = 4 / 5) ∧
  ((Nat.choose n (r + 1) : ℚ) / Nat.choose n (r + 2) = 5 / 6) ∧ 
  n = 98 := by
  sorry

end NUMINAMATH_GPT_pascal_row_with_ratio_456_exists_at_98_l1764_176412


namespace NUMINAMATH_GPT_fin_solutions_l1764_176490

theorem fin_solutions (u : ℕ) (hu : u > 0) :
  ∃ N : ℕ, ∀ n a b : ℕ, n > N → ¬ (n! = u^a - u^b) :=
sorry

end NUMINAMATH_GPT_fin_solutions_l1764_176490


namespace NUMINAMATH_GPT_quadratic_equations_with_common_root_l1764_176480

theorem quadratic_equations_with_common_root :
  ∃ (p1 q1 p2 q2 : ℝ),
    p1 ≠ p2 ∧ q1 ≠ q2 ∧
    ∀ x : ℝ,
      (x^2 + p1 * x + q1 = 0 ∧ x^2 + p2 * x + q2 = 0) →
      (x = 2 ∨ (∃ r1 r2 : ℝ, r1 ≠ r2 ∧ ((x = r1 ∧ x == 2) ∨ (x = r2 ∧ x == 2)))) :=
sorry

end NUMINAMATH_GPT_quadratic_equations_with_common_root_l1764_176480


namespace NUMINAMATH_GPT_negation_of_P_l1764_176442

-- Defining the original proposition
def P : Prop := ∃ x₀ : ℝ, x₀^2 = 1

-- The problem is to prove the negation of the proposition
theorem negation_of_P : (¬P) ↔ (∀ x : ℝ, x^2 ≠ 1) :=
  by sorry

end NUMINAMATH_GPT_negation_of_P_l1764_176442


namespace NUMINAMATH_GPT_correct_statements_l1764_176484

theorem correct_statements (a b c x : ℝ) (h1 : a < 0) 
  (h2 : ∀ x, ax^2 + bx + c ≤ 0 ↔ x ≤ -2 ∨ x ≥ 6)
  (hb : b = -4 * a)
  (hc : c = -12 * a) : 
  (a < 0) ∧ 
  (∀ x, cx^2 - bx + a < 0 ↔ -1/6 < x ∧ x < 1/2) ∧ 
  (a + b + c > 0) :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_l1764_176484


namespace NUMINAMATH_GPT_trucks_more_than_buses_l1764_176489

theorem trucks_more_than_buses (b t : ℕ) (h₁ : b = 9) (h₂ : t = 17) : t - b = 8 :=
by
  sorry

end NUMINAMATH_GPT_trucks_more_than_buses_l1764_176489


namespace NUMINAMATH_GPT_abs_m_minus_n_l1764_176429

theorem abs_m_minus_n (m n : ℝ) (h_avg : (m + n + 9 + 8 + 10) / 5 = 9) (h_var : (1 / 5 * (m^2 + n^2 + 81 + 64 + 100) - 81) = 2) : |m - n| = 4 :=
  sorry

end NUMINAMATH_GPT_abs_m_minus_n_l1764_176429


namespace NUMINAMATH_GPT_price_of_sports_equipment_l1764_176407

theorem price_of_sports_equipment (x y : ℕ) (a b : ℕ) :
  (2 * x + y = 330) → (5 * x + 2 * y = 780) → x = 120 ∧ y = 90 ∧
  (120 * a + 90 * b = 810) → a = 3 ∧ b = 5 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_price_of_sports_equipment_l1764_176407


namespace NUMINAMATH_GPT_starting_even_number_l1764_176453

def is_even (n : ℤ) : Prop := n % 2 = 0

def span_covered_by_evens (count : ℤ) : ℤ := count * 2 - 2

theorem starting_even_number
  (count : ℤ)
  (end_num : ℤ)
  (H1 : is_even end_num)
  (H2 : count = 20)
  (H3 : end_num = 55) :
  ∃ start_num, is_even start_num ∧ start_num = end_num - span_covered_by_evens count + 1 := 
sorry

end NUMINAMATH_GPT_starting_even_number_l1764_176453


namespace NUMINAMATH_GPT_tan_add_pi_over_3_l1764_176420

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_tan_add_pi_over_3_l1764_176420


namespace NUMINAMATH_GPT_algebraic_expression_value_l1764_176417

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y^2 = 1) : -2 * x + 4 * y^2 + 1 = -1 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1764_176417


namespace NUMINAMATH_GPT_sum_three_digit_even_integers_l1764_176481

theorem sum_three_digit_even_integers :
  let a := 100
  let l := 998
  let d := 2
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  S = 247050 :=
by
  let a := 100
  let d := 2
  let l := 998
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  sorry

end NUMINAMATH_GPT_sum_three_digit_even_integers_l1764_176481


namespace NUMINAMATH_GPT_average_is_correct_l1764_176493

def numbers : List ℕ := [12, 13, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

def sum_of_numbers : ℕ := numbers.foldr (· + ·) 0

def number_of_values : ℕ := numbers.length

def average : ℚ := sum_of_numbers / number_of_values

theorem average_is_correct : average = 114391.82 := by
  sorry

end NUMINAMATH_GPT_average_is_correct_l1764_176493


namespace NUMINAMATH_GPT_cost_of_each_shirt_l1764_176471

theorem cost_of_each_shirt (initial_money : ℕ) (cost_pants : ℕ) (money_left : ℕ) (shirt_cost : ℕ)
  (h1 : initial_money = 109)
  (h2 : cost_pants = 13)
  (h3 : money_left = 74)
  (h4 : initial_money - (2 * shirt_cost + cost_pants) = money_left) :
  shirt_cost = 11 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_each_shirt_l1764_176471


namespace NUMINAMATH_GPT_set_equality_l1764_176465

theorem set_equality (M P : Set (ℝ × ℝ))
  (hM : M = {p : ℝ × ℝ | p.1 + p.2 < 0 ∧ p.1 * p.2 > 0})
  (hP : P = {p : ℝ × ℝ | p.1 < 0 ∧ p.2 < 0}) : M = P :=
by
  sorry

end NUMINAMATH_GPT_set_equality_l1764_176465


namespace NUMINAMATH_GPT_arrows_from_530_to_533_l1764_176434

-- Define what it means for the pattern to be cyclic with period 5
def cycle_period (n m : Nat) : Prop := n % m = 0

-- Define the equivalent points on the circular track
def equiv_point (n : Nat) (m : Nat) : Nat := n % m

-- Given conditions
def arrow_pattern : Prop :=
  ∀ n : Nat, cycle_period n 5 ∧
  (equiv_point 530 5 = 0) ∧ (equiv_point 533 5 = 3)

-- The theorem to be proved
theorem arrows_from_530_to_533 :
  (∃ seq : List (Nat × Nat),
    seq = [(0, 1), (1, 2), (2, 3)]) :=
sorry

end NUMINAMATH_GPT_arrows_from_530_to_533_l1764_176434


namespace NUMINAMATH_GPT_bryce_raisins_l1764_176475

theorem bryce_raisins:
  ∃ x : ℕ, (x - 8 = x / 3) ∧ x = 12 :=
by 
  sorry

end NUMINAMATH_GPT_bryce_raisins_l1764_176475


namespace NUMINAMATH_GPT_hydrogen_atoms_in_compound_l1764_176487

theorem hydrogen_atoms_in_compound :
  ∀ (H_atoms Br_atoms O_atoms total_molecular_weight weight_H weight_Br weight_O : ℝ),
  Br_atoms = 1 ∧ O_atoms = 3 ∧ total_molecular_weight = 129 ∧ 
  weight_H = 1 ∧ weight_Br = 79.9 ∧ weight_O = 16 →
  H_atoms = 1 :=
by
  sorry

end NUMINAMATH_GPT_hydrogen_atoms_in_compound_l1764_176487


namespace NUMINAMATH_GPT_percentage_customers_not_pay_tax_l1764_176405

theorem percentage_customers_not_pay_tax
  (daily_shoppers : ℕ)
  (weekly_tax_payers : ℕ)
  (h1 : daily_shoppers = 1000)
  (h2 : weekly_tax_payers = 6580)
  : ((7000 - weekly_tax_payers) / 7000) * 100 = 6 := 
by sorry

end NUMINAMATH_GPT_percentage_customers_not_pay_tax_l1764_176405


namespace NUMINAMATH_GPT_sin_of_right_angle_l1764_176491

theorem sin_of_right_angle (A B C : Type)
  (angle_A : Real) (AB BC : Real)
  (h_angleA : angle_A = 90)
  (h_AB : AB = 16)
  (h_BC : BC = 24) :
  Real.sin (angle_A) = 1 :=
by
  sorry

end NUMINAMATH_GPT_sin_of_right_angle_l1764_176491


namespace NUMINAMATH_GPT_length_of_BA_is_sqrt_557_l1764_176436

-- Define the given conditions
def AD : ℝ := 6
def DC : ℝ := 11
def CB : ℝ := 6
def AC : ℝ := 14

-- Define the theorem statement
theorem length_of_BA_is_sqrt_557 (x : ℝ) (H1 : AD = 6) (H2 : DC = 11) (H3 : CB = 6) (H4 : AC = 14) :
  x = Real.sqrt 557 :=
  sorry

end NUMINAMATH_GPT_length_of_BA_is_sqrt_557_l1764_176436


namespace NUMINAMATH_GPT_determine_n_l1764_176456

-- Define the condition
def eq1 := (1 : ℚ) / (2 ^ 10) + (1 : ℚ) / (2 ^ 9) + (1 : ℚ) / (2 ^ 8)
def eq2 (n : ℚ) := n / (2 ^ 10)

-- The lean statement for the proof problem
theorem determine_n : ∃ (n : ℤ), eq1 = eq2 n ∧ n > 0 ∧ n = 7 := by
  sorry

end NUMINAMATH_GPT_determine_n_l1764_176456
