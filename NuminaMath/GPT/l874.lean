import Mathlib

namespace NUMINAMATH_GPT_movies_left_to_watch_l874_87476

theorem movies_left_to_watch (total_movies watched_movies : Nat) (h_total : total_movies = 12) (h_watched : watched_movies = 6) : total_movies - watched_movies = 6 :=
by
  sorry

end NUMINAMATH_GPT_movies_left_to_watch_l874_87476


namespace NUMINAMATH_GPT_triangle_perimeter_l874_87438

variable (r A p : ℝ)

-- Define the conditions from the problem
def inradius (r : ℝ) := r = 3
def area (A : ℝ) := A = 30
def perimeter (A r p : ℝ) := A = r * (p / 2)

-- The theorem stating the problem
theorem triangle_perimeter (h1 : inradius r) (h2 : area A) (h3 : perimeter A r p) : p = 20 := 
by
  -- Proof is provided by the user, so we skip it with sorry
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l874_87438


namespace NUMINAMATH_GPT_total_cost_of_books_l874_87411

theorem total_cost_of_books
  (C1 : ℝ)
  (C2 : ℝ)
  (H1 : C1 = 285.8333333333333)
  (H2 : 0.85 * C1 = 1.19 * C2) :
  C1 + C2 = 2327.5 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_books_l874_87411


namespace NUMINAMATH_GPT_remainder_when_divided_by_15_l874_87425

theorem remainder_when_divided_by_15 (c d : ℤ) (h1 : c % 60 = 47) (h2 : d % 45 = 14) : (c + d) % 15 = 1 :=
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_15_l874_87425


namespace NUMINAMATH_GPT_number_of_dogs_l874_87459

theorem number_of_dogs (h1 : 24 = 2 * 2 + 4 * n) : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_dogs_l874_87459


namespace NUMINAMATH_GPT_find_complex_number_l874_87481

namespace ComplexProof

open Complex

def satisfies_conditions (z : ℂ) : Prop :=
  (z^2).im = 0 ∧ abs (z - I) = 1

theorem find_complex_number (z : ℂ) (h : satisfies_conditions z) : z = 0 ∨ z = 2 * I :=
sorry

end ComplexProof

end NUMINAMATH_GPT_find_complex_number_l874_87481


namespace NUMINAMATH_GPT_tan_of_sin_in_interval_l874_87420

theorem tan_of_sin_in_interval (α : ℝ) (h1 : Real.sin α = 4 / 5) (h2 : 0 < α ∧ α < Real.pi) :
  Real.tan α = 4 / 3 ∨ Real.tan α = -4 / 3 :=
  sorry

end NUMINAMATH_GPT_tan_of_sin_in_interval_l874_87420


namespace NUMINAMATH_GPT_problem_l874_87430

theorem problem 
  (a : ℝ) 
  (h_a : ∀ x : ℝ, |x + 1| - |2 - x| ≤ a ∧ a ≤ |x + 1| + |2 - x|)
  {m n : ℝ} 
  (h_mn : m > n) 
  (h_n : n > 0)
  (h: a = 3) 
  : 2 * m + 1 / (m^2 - 2 * m * n + n^2) ≥ 2 * n + a :=
by
  sorry

end NUMINAMATH_GPT_problem_l874_87430


namespace NUMINAMATH_GPT_solution_set_a_neg5_solution_set_general_l874_87418

theorem solution_set_a_neg5 (x : ℝ) : (-5 * x^2 + 3 * x + 2 > 0) ↔ (-2/5 < x ∧ x < 1) := 
sorry

theorem solution_set_general (a x : ℝ) : 
  (ax^2 + (a + 3) * x + 3 > 0) ↔
  ((0 < a ∧ a < 3 ∧ (x < -1 ∨ x > -3/a)) ∨ 
   (a = 3 ∧ x ≠ -1) ∨ 
   (a > 3 ∧ (x < -1 ∨ x > -3/a)) ∨ 
   (a = 0 ∧ x > -1) ∨ 
   (a < 0 ∧ -1 < x ∧ x < -3/a)) := 
sorry

end NUMINAMATH_GPT_solution_set_a_neg5_solution_set_general_l874_87418


namespace NUMINAMATH_GPT_distance_to_lightning_l874_87465

noncomputable def distance_from_lightning (time_delay : ℕ) (speed_of_sound : ℕ) (feet_per_mile : ℕ) : ℚ :=
  (time_delay * speed_of_sound : ℕ) / feet_per_mile

theorem distance_to_lightning (time_delay : ℕ) (speed_of_sound : ℕ) (feet_per_mile : ℕ) :
  time_delay = 12 → speed_of_sound = 1120 → feet_per_mile = 5280 → distance_from_lightning time_delay speed_of_sound feet_per_mile = 2.5 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end NUMINAMATH_GPT_distance_to_lightning_l874_87465


namespace NUMINAMATH_GPT_minimum_loaves_arithmetic_sequence_l874_87455

theorem minimum_loaves_arithmetic_sequence :
  ∃ a d : ℚ, 
    (5 * a = 100) ∧ (3 * a + 3 * d = 7 * (2 * a - 3 * d)) ∧ (a - 2 * d = 5/3) :=
sorry

end NUMINAMATH_GPT_minimum_loaves_arithmetic_sequence_l874_87455


namespace NUMINAMATH_GPT_eccentricity_range_of_ellipse_l874_87467

theorem eccentricity_range_of_ellipse 
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (P : ℝ × ℝ) (hP_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (h_foci_relation : ∀(θ₁ θ₂ : ℝ), a / (Real.sin θ₁) = c / (Real.sin θ₂)) :
  ∃ (e : ℝ), e = c / a ∧ (Real.sqrt 2 - 1 < e ∧ e < 1) := 
sorry

end NUMINAMATH_GPT_eccentricity_range_of_ellipse_l874_87467


namespace NUMINAMATH_GPT_range_of_m_l874_87483

/-- Define the domain set A where the function f(x) = 1 / sqrt(4 + 3x - x^2) is defined. -/
def A : Set ℝ := {x | -1 < x ∧ x < 4}

/-- Define the range set B where the function g(x) = - x^2 - 2x + 2, with x in [-1, 1], is defined. -/
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

/-- Define the set C in terms of m. -/
def C (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 2}

/-- Prove the range of the real number m such that C ∩ (A ∪ B) = C. -/
theorem range_of_m : {m : ℝ | C m ⊆ A ∪ B} = {m | -1 ≤ m ∧ m < 2} :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l874_87483


namespace NUMINAMATH_GPT_arrangements_correctness_l874_87494

noncomputable def arrangements_of_groups (total mountaineers : ℕ) (familiar_with_route : ℕ) (required_in_each_group : ℕ) : ℕ :=
  sorry

theorem arrangements_correctness :
  arrangements_of_groups 10 4 2 = 120 :=
sorry

end NUMINAMATH_GPT_arrangements_correctness_l874_87494


namespace NUMINAMATH_GPT_simplify_expression_zero_l874_87424

noncomputable def simplify_expression (a b c d : ℝ) : ℝ :=
  1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2)

theorem simplify_expression_zero (a b c d : ℝ) (h : a + b + c = d)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  simplify_expression a b c d = 0 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_zero_l874_87424


namespace NUMINAMATH_GPT_kekai_ratio_l874_87480

/-
Kekai sells 5 shirts at $1 each,
5 pairs of pants at $3 each,
and he has $10 left after giving some money to his parents.
Our goal is to prove the ratio of the money Kekai gives to his parents
to the total money he earns from selling his clothes is 1:2.
-/

def shirts_sold : ℕ := 5
def pants_sold : ℕ := 5
def shirt_price : ℕ := 1
def pants_price : ℕ := 3
def money_left : ℕ := 10

def total_earnings : ℕ := (shirts_sold * shirt_price) + (pants_sold * pants_price)
def money_given_to_parents : ℕ := total_earnings - money_left
def ratio (a b : ℕ) := (a / Nat.gcd a b, b / Nat.gcd a b)

theorem kekai_ratio : ratio money_given_to_parents total_earnings = (1, 2) :=
  by
    sorry

end NUMINAMATH_GPT_kekai_ratio_l874_87480


namespace NUMINAMATH_GPT_no_three_nat_sum_pair_is_pow_of_three_l874_87405

theorem no_three_nat_sum_pair_is_pow_of_three :
  ¬ ∃ (a b c : ℕ) (m n p : ℕ), a + b = 3 ^ m ∧ b + c = 3 ^ n ∧ c + a = 3 ^ p := 
by 
  sorry

end NUMINAMATH_GPT_no_three_nat_sum_pair_is_pow_of_three_l874_87405


namespace NUMINAMATH_GPT_find_uv_non_integer_l874_87482

noncomputable def q (x y : ℝ) (b : ℕ → ℝ) := 
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 + 
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3

theorem find_uv_non_integer (b : ℕ → ℝ) 
  (h0 : q 0 0 b = 0) 
  (h1 : q 1 0 b = 0) 
  (h2 : q (-1) 0 b = 0) 
  (h3 : q 0 1 b = 0) 
  (h4 : q 0 (-1) b = 0) 
  (h5 : q 1 1 b = 0) 
  (h6 : q 1 (-1) b = 0) 
  (h7 : q 3 3 b = 0) : 
  ∃ u v : ℝ, q u v b = 0 ∧ u = 17/19 ∧ v = 18/19 := 
  sorry

end NUMINAMATH_GPT_find_uv_non_integer_l874_87482


namespace NUMINAMATH_GPT_mileage_interval_l874_87474

-- Define the distances driven each day
def d1 : ℕ := 135
def d2 : ℕ := 135 + 124
def d3 : ℕ := 159
def d4 : ℕ := 189

-- Define the total distance driven
def total_distance : ℕ := d1 + d2 + d3 + d4

-- Define the number of intervals (charges)
def number_of_intervals : ℕ := 6

-- Define the expected mileage interval for charging
def expected_interval : ℕ := 124

-- The theorem to prove that the mileage interval is approximately 124 miles
theorem mileage_interval : total_distance / number_of_intervals = expected_interval := by
  sorry

end NUMINAMATH_GPT_mileage_interval_l874_87474


namespace NUMINAMATH_GPT_vec_eq_l874_87441

def a : ℝ × ℝ := (-1, 0)
def b : ℝ × ℝ := (0, 2)

theorem vec_eq : (2 * a.1 - 3 * b.1, 2 * a.2 - 3 * b.2) = (-2, -6) := by
  sorry

end NUMINAMATH_GPT_vec_eq_l874_87441


namespace NUMINAMATH_GPT_real_roots_condition_l874_87456

theorem real_roots_condition (a : ℝ) (h : a ≠ -1) : 
    (∃ x : ℝ, x^2 + a * x + (a + 1)^2 = 0) ↔ a ∈ Set.Icc (-2 : ℝ) (-2 / 3) :=
sorry

end NUMINAMATH_GPT_real_roots_condition_l874_87456


namespace NUMINAMATH_GPT_smallest_c_plus_d_l874_87432

theorem smallest_c_plus_d :
  ∃ (c d : ℕ), (8 * c + 3 = 3 * d + 8) ∧ c + d = 27 :=
by
  sorry

end NUMINAMATH_GPT_smallest_c_plus_d_l874_87432


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l874_87426

theorem arithmetic_sequence_sum :
  ∃ (c d e : ℕ), 
  c = 15 + (9 - 3) ∧ 
  d = c + (9 - 3) ∧ 
  e = d + (9 - 3) ∧ 
  c + d + e = 81 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l874_87426


namespace NUMINAMATH_GPT_profit_at_original_price_l874_87417

theorem profit_at_original_price (x : ℝ) (h : 0.8 * x = 1.2) : x - 1 = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_profit_at_original_price_l874_87417


namespace NUMINAMATH_GPT_correct_option_is_C_l874_87475

-- Definitions for given conditions
def optionA (x y : ℝ) : Prop := 3 * x + 3 * y = 6 * x * y
def optionB (x y : ℝ) : Prop := 4 * x * y^2 - 5 * x * y^2 = -1
def optionC (x : ℝ) : Prop := -2 * (x - 3) = -2 * x + 6
def optionD (a : ℝ) : Prop := 2 * a + a = 3 * a^2

-- The proof statement to show that Option C is the correct calculation
theorem correct_option_is_C (x y a : ℝ) : 
  ¬ optionA x y ∧ ¬ optionB x y ∧ optionC x ∧ ¬ optionD a :=
by
  -- Proof not required, using sorry to compile successfully
  sorry

end NUMINAMATH_GPT_correct_option_is_C_l874_87475


namespace NUMINAMATH_GPT_determine_pairs_l874_87472

theorem determine_pairs (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔
  (∃ k : ℕ, k > 0 ∧ (a = 7 * k^2 ∧ b = 7 * k) ∨ (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1)) :=
by
  sorry

end NUMINAMATH_GPT_determine_pairs_l874_87472


namespace NUMINAMATH_GPT_range_of_a_l874_87462

theorem range_of_a (a : ℝ) (a_seq : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : ∀ n : ℕ, a_seq n = a + n - 1)
  (h2 : ∀ n : ℕ, b n = (1 + a_seq n) / a_seq n)
  (h3 : ∀ n : ℕ, n > 0 → b n ≤ b 5) :
  -4 < a ∧ a < -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l874_87462


namespace NUMINAMATH_GPT_ratio_of_larger_to_smaller_l874_87461

theorem ratio_of_larger_to_smaller (x y : ℝ) (hx : x > y) (hx_pos : x > 0) (hy_pos : y > 0) 
  (h : x + y = 7 * (x - y)) : x / y = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_larger_to_smaller_l874_87461


namespace NUMINAMATH_GPT_cube_root_of_neg_27_over_8_l874_87497

theorem cube_root_of_neg_27_over_8 :
  (- (3 : ℝ) / 2) ^ 3 = - (27 / 8 : ℝ) := 
by
  sorry

end NUMINAMATH_GPT_cube_root_of_neg_27_over_8_l874_87497


namespace NUMINAMATH_GPT_number_of_ways_to_partition_22_as_triangle_pieces_l874_87434

theorem number_of_ways_to_partition_22_as_triangle_pieces : 
  (∃ (a b c : ℕ), a + b + c = 22 ∧ a + b > c ∧ a + c > b ∧ b + c > a) → 
  ∃! (count : ℕ), count = 10 :=
by sorry

end NUMINAMATH_GPT_number_of_ways_to_partition_22_as_triangle_pieces_l874_87434


namespace NUMINAMATH_GPT_ferris_wheel_seats_l874_87442

variable (total_people : ℕ) (people_per_seat : ℕ)

theorem ferris_wheel_seats (h1 : total_people = 18) (h2 : people_per_seat = 9) : total_people / people_per_seat = 2 := by
  sorry

end NUMINAMATH_GPT_ferris_wheel_seats_l874_87442


namespace NUMINAMATH_GPT_kite_cost_l874_87427

variable (initial_amount : ℕ) (cost_frisbee : ℕ) (amount_left : ℕ)

theorem kite_cost (initial_amount : ℕ) (cost_frisbee : ℕ) (amount_left : ℕ) (h_initial_amount : initial_amount = 78) (h_cost_frisbee : cost_frisbee = 9) (h_amount_left : amount_left = 61) : 
  initial_amount - amount_left - cost_frisbee = 8 :=
by
  -- Proof can be completed here
  sorry

end NUMINAMATH_GPT_kite_cost_l874_87427


namespace NUMINAMATH_GPT_population_of_missing_village_eq_945_l874_87477

theorem population_of_missing_village_eq_945
  (pop1 pop2 pop3 pop4 pop5 pop6 : ℕ)
  (avg_pop total_population missing_population : ℕ)
  (h1 : pop1 = 803)
  (h2 : pop2 = 900)
  (h3 : pop3 = 1100)
  (h4 : pop4 = 1023)
  (h5 : pop5 = 980)
  (h6 : pop6 = 1249)
  (h_avg : avg_pop = 1000)
  (h_total_population : total_population = avg_pop * 7)
  (h_missing_population : missing_population = total_population - (pop1 + pop2 + pop3 + pop4 + pop5 + pop6)) :
  missing_population = 945 :=
by {
  -- Here would go the proof steps if needed
  sorry 
}

end NUMINAMATH_GPT_population_of_missing_village_eq_945_l874_87477


namespace NUMINAMATH_GPT_total_selling_price_l874_87484

theorem total_selling_price
  (cost1 : ℝ) (cost2 : ℝ) (cost3 : ℝ) 
  (profit_percent1 : ℝ) (profit_percent2 : ℝ) (profit_percent3 : ℝ) :
  cost1 = 600 → cost2 = 450 → cost3 = 750 →
  profit_percent1 = 0.08 → profit_percent2 = 0.10 → profit_percent3 = 0.15 →
  (cost1 * (1 + profit_percent1) + cost2 * (1 + profit_percent2) + cost3 * (1 + profit_percent3)) = 2005.50 :=
by
  intros h1 h2 h3 p1 p2 p3
  simp [h1, h2, h3, p1, p2, p3]
  sorry

end NUMINAMATH_GPT_total_selling_price_l874_87484


namespace NUMINAMATH_GPT_value_of_y_square_plus_inverse_square_l874_87413

variable {y : ℝ}
variable (h : 35 = y^4 + 1 / y^4)

theorem value_of_y_square_plus_inverse_square (h : 35 = y^4 + 1 / y^4) : y^2 + 1 / y^2 = Real.sqrt 37 := 
sorry

end NUMINAMATH_GPT_value_of_y_square_plus_inverse_square_l874_87413


namespace NUMINAMATH_GPT_max_a4_l874_87439

variable (a1 d : ℝ)

theorem max_a4 (h1 : 2 * a1 + 6 * d ≥ 10) (h2 : 2.5 * a1 + 10 * d ≤ 15) :
  ∃ max_a4, max_a4 = 4 ∧ a1 + 3 * d ≤ max_a4 :=
by
  sorry

end NUMINAMATH_GPT_max_a4_l874_87439


namespace NUMINAMATH_GPT_patrick_savings_l874_87416

theorem patrick_savings :
  let bicycle_price := 150
  let saved_money := bicycle_price / 2
  let lent_money := 50
  saved_money - lent_money = 25 := by
  let bicycle_price := 150
  let saved_money := bicycle_price / 2
  let lent_money := 50
  sorry

end NUMINAMATH_GPT_patrick_savings_l874_87416


namespace NUMINAMATH_GPT_triangle_inequality_l874_87470

theorem triangle_inequality (a b c : ℝ) (h : a < b + c) : a^2 - b^2 - c^2 - 2*b*c < 0 := by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l874_87470


namespace NUMINAMATH_GPT_inequality_solution_set_l874_87422

theorem inequality_solution_set :
  {x : ℝ | (x / (x ^ 2 - 8 * x + 15) ≥ 2) ∧ (x ^ 2 - 8 * x + 15 ≠ 0)} =
  {x : ℝ | (5 / 2 ≤ x ∧ x < 3) ∨ (5 < x ∧ x ≤ 6)} :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l874_87422


namespace NUMINAMATH_GPT_side_length_of_S2_is_1001_l874_87473

-- Definitions and Conditions
variables (R1 R2 : Type) (S1 S2 S3 : Type)
variables (r s : ℤ)
variables (h_total_width : 2 * r + 3 * s = 4422)
variables (h_total_height : 2 * r + s = 2420)

theorem side_length_of_S2_is_1001 (R1 R2 S1 S2 S3 : Type) (r s : ℤ)
  (h_total_width : 2 * r + 3 * s = 4422)
  (h_total_height : 2 * r + s = 2420) : s = 1001 :=
by
  sorry -- proof to be provided

end NUMINAMATH_GPT_side_length_of_S2_is_1001_l874_87473


namespace NUMINAMATH_GPT_community_cleaning_children_l874_87404

theorem community_cleaning_children (total_members adult_men_ratio adult_women_ratio : ℕ) 
(h_total : total_members = 2000)
(h_men_ratio : adult_men_ratio = 30) 
(h_women_ratio : adult_women_ratio = 2) :
  (total_members - (adult_men_ratio * total_members / 100 + 
  adult_women_ratio * (adult_men_ratio * total_members / 100))) = 200 :=
by
  sorry

end NUMINAMATH_GPT_community_cleaning_children_l874_87404


namespace NUMINAMATH_GPT_area_of_triangle_CM_N_l874_87464

noncomputable def triangle_area (a : ℝ) : ℝ :=
  let M := (a / 2, a, a)
  let N := (a, a / 2, a)
  let MN := Real.sqrt ((a - a / 2) ^ 2 + (a / 2 - a) ^ 2)
  let CK := Real.sqrt (a ^ 2 + (a * Real.sqrt 2 / 4) ^ 2)
  (1/2) * MN * CK

theorem area_of_triangle_CM_N 
  (a : ℝ) :
  (a > 0) →
  triangle_area a = (3 * a^2) / 8 :=
by
  intro h
  -- Proof will go here.
  sorry

end NUMINAMATH_GPT_area_of_triangle_CM_N_l874_87464


namespace NUMINAMATH_GPT_reciprocal_neg_one_div_2022_l874_87419

theorem reciprocal_neg_one_div_2022 : (1 / (-1 / 2022)) = -2022 :=
by sorry

end NUMINAMATH_GPT_reciprocal_neg_one_div_2022_l874_87419


namespace NUMINAMATH_GPT_quadratic_complete_square_l874_87400

theorem quadratic_complete_square : 
  ∃ d e : ℝ, ((x^2 - 16*x + 15) = ((x + d)^2 + e)) ∧ (d + e = -57) := by
  sorry

end NUMINAMATH_GPT_quadratic_complete_square_l874_87400


namespace NUMINAMATH_GPT_jackson_vacuuming_time_l874_87468

-- Definitions based on the conditions
def hourly_wage : ℕ := 5
def washing_dishes_time : ℝ := 0.5
def cleaning_bathroom_time : ℝ := 3 * washing_dishes_time
def total_earnings : ℝ := 30

-- The total time spent on chores
def total_chore_time (V : ℝ) : ℝ :=
  2 * V + washing_dishes_time + cleaning_bathroom_time

-- The main theorem that needs to be proven
theorem jackson_vacuuming_time :
  ∃ V : ℝ, hourly_wage * total_chore_time V = total_earnings ∧ V = 2 :=
by
  sorry

end NUMINAMATH_GPT_jackson_vacuuming_time_l874_87468


namespace NUMINAMATH_GPT_find_b_l874_87487

theorem find_b (a b c : ℝ) (k₁ k₂ k₃ : ℤ) :
  (a + b) / 2 = 40 ∧
  (b + c) / 2 = 43 ∧
  (a + c) / 2 = 44 ∧
  a + b = 5 * k₁ ∧
  b + c = 5 * k₂ ∧
  a + c = 5 * k₃
  → b = 40 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_b_l874_87487


namespace NUMINAMATH_GPT_logarithmic_AMGM_inequality_l874_87466

theorem logarithmic_AMGM_inequality (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) :
  2 * ((Real.log b / (a * Real.log a)) / (a + b) + 
       (Real.log c / (b * Real.log b)) / (b + c) + 
       (Real.log a / (c * Real.log c)) / (c + a)) 
  ≥ 9 / (a + b + c) := 
sorry

end NUMINAMATH_GPT_logarithmic_AMGM_inequality_l874_87466


namespace NUMINAMATH_GPT_find_b_minus_a_l874_87444

noncomputable def rotate_90_counterclockwise (x y xc yc : ℝ) : ℝ × ℝ :=
  (xc + (-(y - yc)), yc + (x - xc))

noncomputable def reflect_about_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem find_b_minus_a (a b : ℝ) :
  let xc := 2
  let yc := 3
  let P := (a, b)
  let P_rotated := rotate_90_counterclockwise a b xc yc
  let P_reflected := reflect_about_y_eq_x P_rotated.1 P_rotated.2
  P_reflected = (4, 1) →
  b - a = 1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_b_minus_a_l874_87444


namespace NUMINAMATH_GPT_lines_intersect_l874_87453

theorem lines_intersect (m : ℝ) : ∃ (x y : ℝ), 3 * x + 2 * y + m = 0 ∧ (m^2 + 1) * x - 3 * y - 3 * m = 0 := 
by {
  sorry
}

end NUMINAMATH_GPT_lines_intersect_l874_87453


namespace NUMINAMATH_GPT_hyperbola_foci_distance_l874_87469

theorem hyperbola_foci_distance (a b : ℝ) (h1 : a^2 = 25) (h2 : b^2 = 9) :
  2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 34 := 
by
  sorry

end NUMINAMATH_GPT_hyperbola_foci_distance_l874_87469


namespace NUMINAMATH_GPT_total_books_from_library_l874_87492

def initialBooks : ℕ := 54
def additionalBooks : ℕ := 23

theorem total_books_from_library : initialBooks + additionalBooks = 77 := by
  sorry

end NUMINAMATH_GPT_total_books_from_library_l874_87492


namespace NUMINAMATH_GPT_initial_apps_l874_87421

-- Define the initial condition stating the number of files Dave had initially
def files_initial : ℕ := 21

-- Define the condition after deletion
def apps_after_deletion : ℕ := 3
def files_after_deletion : ℕ := 7

-- Define the number of files deleted
def files_deleted : ℕ := 14

-- Prove that the initial number of apps Dave had was 3
theorem initial_apps (a : ℕ) (h1 : files_initial = 21) 
(h2 : files_after_deletion = 7) 
(h3 : files_deleted = 14) 
(h4 : a - 3 = 0) : a = 3 :=
by sorry

end NUMINAMATH_GPT_initial_apps_l874_87421


namespace NUMINAMATH_GPT_option_c_correct_l874_87495

theorem option_c_correct (α x1 x2 : ℝ) (hα1 : 0 < α) (hα2 : α < π) (hx1 : 0 < x1) (hx2 : x1 < x2) : 
  (x2 / x1) ^ Real.sin α > 1 :=
by
  sorry

end NUMINAMATH_GPT_option_c_correct_l874_87495


namespace NUMINAMATH_GPT_initial_bucket_capacity_l874_87458

theorem initial_bucket_capacity (x : ℕ) (h1 : x - 3 = 2) : x = 5 := sorry

end NUMINAMATH_GPT_initial_bucket_capacity_l874_87458


namespace NUMINAMATH_GPT_imaginary_power_sum_zero_l874_87449

theorem imaginary_power_sum_zero (i : ℂ) (n : ℤ) (h : i^2 = -1) :
  i^(2*n - 3) + i^(2*n - 1) + i^(2*n + 1) + i^(2*n + 3) = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_imaginary_power_sum_zero_l874_87449


namespace NUMINAMATH_GPT_gcd_35_91_840_l874_87407

theorem gcd_35_91_840 : Nat.gcd (Nat.gcd 35 91) 840 = 7 :=
by
  sorry

end NUMINAMATH_GPT_gcd_35_91_840_l874_87407


namespace NUMINAMATH_GPT_number_of_refuels_needed_l874_87437

noncomputable def fuelTankCapacity : ℕ := 50
noncomputable def distanceShanghaiHarbin : ℕ := 2560
noncomputable def fuelConsumptionRate : ℕ := 8
noncomputable def safetyFuel : ℕ := 6

theorem number_of_refuels_needed
  (fuelTankCapacity : ℕ)
  (distanceShanghaiHarbin : ℕ)
  (fuelConsumptionRate : ℕ)
  (safetyFuel : ℕ) :
  (fuelTankCapacity = 50) →
  (distanceShanghaiHarbin = 2560) →
  (fuelConsumptionRate = 8) →
  (safetyFuel = 6) →
  ∃ n : ℕ, n = 4 := by
  sorry

end NUMINAMATH_GPT_number_of_refuels_needed_l874_87437


namespace NUMINAMATH_GPT_circumcenter_rational_l874_87451

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} :
  ∃ (x y : ℚ), 
    ((x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2) ∧
    ((x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2) :=
sorry

end NUMINAMATH_GPT_circumcenter_rational_l874_87451


namespace NUMINAMATH_GPT_smallest_class_size_l874_87412

theorem smallest_class_size (n : ℕ) (h : 5 * n + 2 > 40) : 5 * n + 2 ≥ 42 :=
by
  sorry

end NUMINAMATH_GPT_smallest_class_size_l874_87412


namespace NUMINAMATH_GPT_find_original_prices_and_discount_l874_87478

theorem find_original_prices_and_discount :
  ∃ x y a : ℝ,
  (6 * x + 5 * y = 1140) ∧
  (3 * x + 7 * y = 1110) ∧
  (((9 * x + 8 * y) - 1062) / (9 * x + 8 * y) = a) ∧
  x = 90 ∧
  y = 120 ∧
  a = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_find_original_prices_and_discount_l874_87478


namespace NUMINAMATH_GPT_slices_per_person_eq_three_l874_87489

variables (num_people : ℕ) (slices_per_pizza : ℕ) (num_pizzas : ℕ)

theorem slices_per_person_eq_three (h1 : num_people = 18) (h2 : slices_per_pizza = 9) (h3 : num_pizzas = 6) : 
  (num_pizzas * slices_per_pizza) / num_people = 3 :=
sorry

end NUMINAMATH_GPT_slices_per_person_eq_three_l874_87489


namespace NUMINAMATH_GPT_second_machine_time_l874_87423

theorem second_machine_time (x : ℝ) : 
  (600 / 10) + (1000 / x) = 1000 / 4 ↔ 
  1 / 10 + 1 / x = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_second_machine_time_l874_87423


namespace NUMINAMATH_GPT_product_fraction_simplification_l874_87435

theorem product_fraction_simplification : 
  (1^4 - 1) / (1^4 + 1) * (2^4 - 1) / (2^4 + 1) * (3^4 - 1) / (3^4 + 1) *
  (4^4 - 1) / (4^4 + 1) * (5^4 - 1) / (5^4 + 1) * (6^4 - 1) / (6^4 + 1) *
  (7^4 - 1) / (7^4 + 1) = 50 := 
  sorry

end NUMINAMATH_GPT_product_fraction_simplification_l874_87435


namespace NUMINAMATH_GPT_polynomial_coeff_sums_l874_87460

theorem polynomial_coeff_sums (g h : ℤ) (d : ℤ) :
  (7 * d^2 - 3 * d + g) * (3 * d^2 + h * d - 8) = 21 * d^4 - 44 * d^3 - 35 * d^2 + 14 * d - 16 →
  g + h = -3 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_coeff_sums_l874_87460


namespace NUMINAMATH_GPT_min_bottles_needed_l874_87409

theorem min_bottles_needed (bottle_size : ℕ) (min_ounces : ℕ) (n : ℕ) 
  (h1 : bottle_size = 15) 
  (h2 : min_ounces = 195) 
  (h3 : 15 * n >= 195) : n = 13 :=
sorry

end NUMINAMATH_GPT_min_bottles_needed_l874_87409


namespace NUMINAMATH_GPT_expression_simplifies_to_zero_l874_87402

theorem expression_simplifies_to_zero (x y : ℝ) (h : x = 2024) :
    5 * (x ^ 3 - 3 * x ^ 2 * y - 2 * x * y ^ 2) -
    3 * (x ^ 3 - 5 * x ^ 2 * y + 2 * y ^ 3) +
    2 * (-x ^ 3 + 5 * x * y ^ 2 + 3 * y ^ 3) = 0 :=
by {
    sorry
}

end NUMINAMATH_GPT_expression_simplifies_to_zero_l874_87402


namespace NUMINAMATH_GPT_tea_sale_price_correct_l874_87471

noncomputable def cost_price (weight: ℕ) (unit_price: ℕ) : ℕ := weight * unit_price
noncomputable def desired_profit (cost: ℕ) (percentage: ℕ) : ℕ := cost * percentage / 100
noncomputable def sale_price (cost: ℕ) (profit: ℕ) : ℕ := cost + profit
noncomputable def sale_price_per_kg (total_sale_price: ℕ) (weight: ℕ) : ℚ := total_sale_price / weight

theorem tea_sale_price_correct :
  ∀ (weight_A weight_B weight_C weight_D cost_per_kg_A cost_per_kg_B cost_per_kg_C cost_per_kg_D
     profit_percent_A profit_percent_B profit_percent_C profit_percent_D : ℕ),

  weight_A = 80 →
  weight_B = 20 →
  weight_C = 50 →
  weight_D = 30 →
  cost_per_kg_A = 15 →
  cost_per_kg_B = 20 →
  cost_per_kg_C = 25 →
  cost_per_kg_D = 30 →
  profit_percent_A = 25 →
  profit_percent_B = 30 →
  profit_percent_C = 20 →
  profit_percent_D = 15 →
  
  sale_price_per_kg (sale_price (cost_price weight_A cost_per_kg_A) (desired_profit (cost_price weight_A cost_per_kg_A) profit_percent_A)) weight_A = 18.75 →
  sale_price_per_kg (sale_price (cost_price weight_B cost_per_kg_B) (desired_profit (cost_price weight_B cost_per_kg_B) profit_percent_B)) weight_B = 26 →
  sale_price_per_kg (sale_price (cost_price weight_C cost_per_kg_C) (desired_profit (cost_price weight_C cost_per_kg_C) profit_percent_C)) weight_C = 30 →
  sale_price_per_kg (sale_price (cost_price weight_D cost_per_kg_D) (desired_profit (cost_price weight_D cost_per_kg_D) profit_percent_D)) weight_D = 34.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_tea_sale_price_correct_l874_87471


namespace NUMINAMATH_GPT_evaluate_expression_l874_87428

noncomputable def M (x y : ℝ) : ℝ := if x < y then y else x
noncomputable def m (x y : ℝ) : ℝ := if x < y then x else y

theorem evaluate_expression
  (p q r s t : ℝ)
  (h1 : p < q)
  (h2 : q < r)
  (h3 : r < s)
  (h4 : s < t)
  (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ s ∧ s ≠ t ∧ t ≠ p ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ s ∧ q ≠ t ∧ r ≠ t):
  M (M p (m q r)) (m s (m p t)) = q := 
sorry

end NUMINAMATH_GPT_evaluate_expression_l874_87428


namespace NUMINAMATH_GPT_combined_tax_rate_is_correct_l874_87493

noncomputable def combined_tax_rate (john_income : ℝ) (ingrid_income : ℝ) (john_tax_rate : ℝ) (ingrid_tax_rate : ℝ) : ℝ :=
  let john_tax := john_tax_rate * john_income
  let ingrid_tax := ingrid_tax_rate * ingrid_income
  let total_tax := john_tax + ingrid_tax
  let total_income := john_income + ingrid_income
  total_tax / total_income

theorem combined_tax_rate_is_correct :
  combined_tax_rate 56000 72000 0.30 0.40 = 0.35625 := 
by
  sorry

end NUMINAMATH_GPT_combined_tax_rate_is_correct_l874_87493


namespace NUMINAMATH_GPT_solve_for_y_l874_87414

theorem solve_for_y (y : ℚ) (h : 2 * y + 3 * y = 500 - (4 * y + 5 * y)) : y = 250 / 7 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l874_87414


namespace NUMINAMATH_GPT_interest_difference_l874_87440

theorem interest_difference (P R T : ℝ) (SI : ℝ) (Diff : ℝ) :
  P = 250 ∧ R = 4 ∧ T = 8 ∧ SI = (P * R * T) / 100 ∧ Diff = P - SI → Diff = 170 :=
by sorry

end NUMINAMATH_GPT_interest_difference_l874_87440


namespace NUMINAMATH_GPT_product_of_special_triplet_l874_87410

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_triangular (n : ℕ) : Prop := ∃ k : ℕ, n = k * (k + 1) / 2

def three_consecutive (a b c : ℕ) : Prop := b = a + 1 ∧ c = b + 1

theorem product_of_special_triplet :
  ∃ a b c : ℕ, a < b ∧ b < c ∧ c < 20 ∧ three_consecutive a b c ∧
   is_prime a ∧ is_even b ∧ is_triangular c ∧ a * b * c = 2730 :=
sorry

end NUMINAMATH_GPT_product_of_special_triplet_l874_87410


namespace NUMINAMATH_GPT_smallest_missing_digit_units_place_cube_l874_87431

theorem smallest_missing_digit_units_place_cube :
  ∀ d : Fin 10, ∃ n : ℕ, (n ^ 3) % 10 = d :=
by
  sorry

end NUMINAMATH_GPT_smallest_missing_digit_units_place_cube_l874_87431


namespace NUMINAMATH_GPT_find_minimum_r_l874_87498

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem find_minimum_r (r : ℕ) (h_pos : r > 0) (h_perfect : is_perfect_square (4^3 + 4^r + 4^4)) : r = 4 :=
sorry

end NUMINAMATH_GPT_find_minimum_r_l874_87498


namespace NUMINAMATH_GPT_probability_of_red_ball_is_correct_l874_87485

noncomputable def probability_of_drawing_red_ball (white_balls : ℕ) (red_balls : ℕ) :=
  let total_balls := white_balls + red_balls
  let favorable_outcomes := red_balls
  (favorable_outcomes : ℚ) / total_balls

theorem probability_of_red_ball_is_correct :
  probability_of_drawing_red_ball 5 2 = 2 / 7 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_red_ball_is_correct_l874_87485


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l874_87490

theorem isosceles_triangle_base_length (s a b : ℕ) (h1 : 3 * s = 45)
  (h2 : 2 * a + b = 40) (h3 : a = s) : b = 10 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l874_87490


namespace NUMINAMATH_GPT_sum_of_fourth_powers_of_consecutive_integers_l874_87408

-- Definitions based on conditions
def consecutive_squares_sum (x : ℤ) : Prop :=
  (x - 1)^2 + x^2 + (x + 1)^2 = 12246

-- Statement of the problem
theorem sum_of_fourth_powers_of_consecutive_integers (x : ℤ)
  (h : consecutive_squares_sum x) : 
  (x - 1)^4 + x^4 + (x + 1)^4 = 50380802 :=
sorry

end NUMINAMATH_GPT_sum_of_fourth_powers_of_consecutive_integers_l874_87408


namespace NUMINAMATH_GPT_blue_red_area_ratio_l874_87452

theorem blue_red_area_ratio (d1 d2 : ℝ) (h1 : d1 = 2) (h2 : d2 = 6) :
  let r1 := d1 / 2
  let r2 := d2 / 2
  let a_red := π * r1^2
  let a_large := π * r2^2
  let a_blue := a_large - a_red
  a_blue / a_red = 8 :=
by
  have r1 := d1 / 2
  have r2 := d2 / 2
  have a_red := π * r1^2
  have a_large := π * r2^2
  have a_blue := a_large - a_red
  sorry

end NUMINAMATH_GPT_blue_red_area_ratio_l874_87452


namespace NUMINAMATH_GPT_max_digit_sum_in_24_hour_format_l874_87446

theorem max_digit_sum_in_24_hour_format : 
  ∃ t : ℕ × ℕ, (0 ≤ t.fst ∧ t.fst < 24 ∧ 0 ≤ t.snd ∧ t.snd < 60 ∧ (t.fst / 10 + t.fst % 10 + t.snd / 10 + t.snd % 10 = 24)) :=
sorry

end NUMINAMATH_GPT_max_digit_sum_in_24_hour_format_l874_87446


namespace NUMINAMATH_GPT_standard_equation_of_parabola_l874_87445

theorem standard_equation_of_parabola (x : ℝ) (y : ℝ) (directrix : ℝ) (eq_directrix : directrix = 1) :
  y^2 = -4 * x :=
sorry

end NUMINAMATH_GPT_standard_equation_of_parabola_l874_87445


namespace NUMINAMATH_GPT_alyssa_picked_42_l874_87486

variable (totalPears nancyPears : ℕ)
variable (total_picked : totalPears = 59)
variable (nancy_picked : nancyPears = 17)

theorem alyssa_picked_42 (h1 : totalPears = 59) (h2 : nancyPears = 17) :
  totalPears - nancyPears = 42 :=
by
  sorry

end NUMINAMATH_GPT_alyssa_picked_42_l874_87486


namespace NUMINAMATH_GPT_check_3x5_board_cannot_be_covered_l874_87488

/-- Define the concept of a checkerboard with a given number of rows and columns. -/
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Define the number of squares on a checkerboard. -/
def num_squares (cb : Checkerboard) : ℕ :=
  cb.rows * cb.cols

/-- Define whether a board can be completely covered by dominoes. -/
def can_be_covered_by_dominoes (cb : Checkerboard) : Prop :=
  (num_squares cb) % 2 = 0

/-- Instantiate the specific checkerboard scenarios. -/
def board_3x4 := Checkerboard.mk 3 4
def board_3x5 := Checkerboard.mk 3 5
def board_4x4 := Checkerboard.mk 4 4
def board_4x5 := Checkerboard.mk 4 5
def board_6x3 := Checkerboard.mk 6 3

/-- Statement to prove which board cannot be covered completely by dominoes. -/
theorem check_3x5_board_cannot_be_covered : ¬ can_be_covered_by_dominoes board_3x5 :=
by
  /- We leave out the proof steps here as requested. -/
  sorry

end NUMINAMATH_GPT_check_3x5_board_cannot_be_covered_l874_87488


namespace NUMINAMATH_GPT_inverse_function_properties_l874_87443

theorem inverse_function_properties {f : ℝ → ℝ} 
  (h_monotonic_decreasing : ∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 3 → f x2 < f x1)
  (h_range : ∀ y : ℝ, 4 ≤ y ∧ y ≤ 7 ↔ ∃ x : ℝ, 1 ≤ x ∧ x ≤ 3 ∧ y = f x)
  (h_inverse_exists : ∃ g : ℝ → ℝ, ∀ x : ℝ, f (g x) = x ∧ g (f x) = x) :
  ∃ g : ℝ → ℝ, (∀ y1 y2 : ℝ, 4 ≤ y1 ∧ y1 < y2 ∧ y2 ≤ 7 → g y2 < g y1) ∧ (∀ y : ℝ, 4 ≤ y ∧ y ≤ 7 → g y ≤ 3) :=
sorry

end NUMINAMATH_GPT_inverse_function_properties_l874_87443


namespace NUMINAMATH_GPT_reading_proof_l874_87479

noncomputable def reading (arrow_pos : ℝ) : ℝ :=
  if arrow_pos > 9.75 ∧ arrow_pos < 10.0 then 9.95 else 0

theorem reading_proof
  (arrow_pos : ℝ)
  (h0 : 9.75 < arrow_pos)
  (h1 : arrow_pos < 10.0)
  (possible_readings : List ℝ)
  (h2 : possible_readings = [9.80, 9.90, 9.95, 10.0, 9.85]) :
  reading arrow_pos = 9.95 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_reading_proof_l874_87479


namespace NUMINAMATH_GPT_percentage_difference_l874_87403

theorem percentage_difference :
  let a1 := 0.12 * 24.2
  let a2 := 0.10 * 14.2
  a1 - a2 = 1.484 := 
by
  -- Definitions
  let a1 := 0.12 * 24.2
  let a2 := 0.10 * 14.2
  -- Proof body (skipped for this task)
  sorry

end NUMINAMATH_GPT_percentage_difference_l874_87403


namespace NUMINAMATH_GPT_fourth_term_correct_l874_87499

def fourth_term_sequence : Nat :=
  4^0 + 4^1 + 4^2 + 4^3

theorem fourth_term_correct : fourth_term_sequence = 85 :=
by
  sorry

end NUMINAMATH_GPT_fourth_term_correct_l874_87499


namespace NUMINAMATH_GPT_geometric_sequence_a_eq_neg4_l874_87496

theorem geometric_sequence_a_eq_neg4 
    (a : ℝ)
    (h : (2 * a + 2) ^ 2 = a * (3 * a + 3)) : 
    a = -4 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a_eq_neg4_l874_87496


namespace NUMINAMATH_GPT_solution_inequality_l874_87454

theorem solution_inequality
  (a a' b b' c : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : a' ≠ 0)
  (h₃ : (c - b) / a > (c - b') / a') :
  (c - b') / a' < (c - b) / a :=
by
  sorry

end NUMINAMATH_GPT_solution_inequality_l874_87454


namespace NUMINAMATH_GPT_board_numbers_l874_87457

theorem board_numbers (a b c : ℕ) (h1 : a = 3) (h2 : b = 9) (h3 : c = 15)
    (op : ∀ x y z : ℕ, (x = y + z - t) → true)  -- simplifying the operation representation
    (min_number : ∃ x, x = 2013) : ∃ n m, n = 2019 ∧ m = 2025 := 
sorry

end NUMINAMATH_GPT_board_numbers_l874_87457


namespace NUMINAMATH_GPT_contradiction_proof_l874_87448

theorem contradiction_proof (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : 
  ¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) :=
sorry

end NUMINAMATH_GPT_contradiction_proof_l874_87448


namespace NUMINAMATH_GPT_probability_of_drawing_white_ball_is_zero_l874_87436

theorem probability_of_drawing_white_ball_is_zero
  (red_balls blue_balls : ℕ)
  (h1 : red_balls = 3)
  (h2 : blue_balls = 5)
  (white_balls : ℕ)
  (h3 : white_balls = 0) : 
  (0 / (red_balls + blue_balls + white_balls) = 0) :=
sorry

end NUMINAMATH_GPT_probability_of_drawing_white_ball_is_zero_l874_87436


namespace NUMINAMATH_GPT_total_canoes_built_by_april_l874_87401

theorem total_canoes_built_by_april
  (initial : ℕ)
  (production_increase : ℕ → ℕ) 
  (total_canoes : ℕ) :
  initial = 5 →
  (∀ n, production_increase n = 3 * n) →
  total_canoes = initial + production_increase initial + production_increase (production_increase initial) + production_increase (production_increase (production_increase initial)) →
  total_canoes = 200 :=
by
  intros h_initial h_production h_total
  sorry

end NUMINAMATH_GPT_total_canoes_built_by_april_l874_87401


namespace NUMINAMATH_GPT_chickens_and_rabbits_l874_87450

theorem chickens_and_rabbits (c r : ℕ) (h1 : c + r = 15) (h2 : 2 * c + 4 * r = 40) : c = 10 ∧ r = 5 :=
sorry

end NUMINAMATH_GPT_chickens_and_rabbits_l874_87450


namespace NUMINAMATH_GPT_find_first_term_of_sequence_l874_87429

theorem find_first_term_of_sequence
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n, a (n+1) = a n + d)
  (h2 : a 0 + a 1 + a 2 = 12)
  (h3 : a 0 * a 1 * a 2 = 48)
  (h4 : ∀ n m, n < m → a n ≤ a m) :
  a 0 = 2 :=
sorry

end NUMINAMATH_GPT_find_first_term_of_sequence_l874_87429


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l874_87415

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, a (n+1) = a n * q)
  (h3 : 2 * a 0 + a 1 = a 2)
  : q = 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l874_87415


namespace NUMINAMATH_GPT_gcd_154_308_462_l874_87447

theorem gcd_154_308_462 : Nat.gcd (Nat.gcd 154 308) 462 = 154 := by
  sorry

end NUMINAMATH_GPT_gcd_154_308_462_l874_87447


namespace NUMINAMATH_GPT_father_dig_time_l874_87463

-- Definitions based on the conditions
variable (T : ℕ) -- Time taken by the father to dig the hole in hours
variable (D : ℕ) -- Depth of the hole dug by the father in feet
variable (M : ℕ) -- Depth of the hole dug by Michael in feet

-- Conditions
def father_hole_depth : Prop := D = 4 * T
def michael_hole_depth : Prop := M = 2 * D - 400
def michael_dig_time : Prop := M = 4 * 700

-- The proof statement, proving T = 400 given the conditions
theorem father_dig_time (T D M : ℕ)
  (h1 : father_hole_depth T D)
  (h2 : michael_hole_depth D M)
  (h3 : michael_dig_time M) : T = 400 := 
by
  sorry

end NUMINAMATH_GPT_father_dig_time_l874_87463


namespace NUMINAMATH_GPT_pow_mod_3_225_l874_87491

theorem pow_mod_3_225 :
  (3 ^ 225) % 11 = 1 :=
by
  -- Given condition from problem:
  have h : 3 ^ 5 % 11 = 1 := by norm_num
  -- Proceed to prove based on this condition
  sorry

end NUMINAMATH_GPT_pow_mod_3_225_l874_87491


namespace NUMINAMATH_GPT_seedling_costs_and_purchase_l874_87406

variable (cost_A cost_B : ℕ)
variable (m n : ℕ)

-- Conditions
def conditions : Prop :=
  (cost_A = cost_B + 5) ∧ 
  (400 / cost_A = 300 / cost_B)

-- Prove costs and purchase for minimal costs
theorem seedling_costs_and_purchase (cost_A cost_B : ℕ) (m n : ℕ)
  (h1 : conditions cost_A cost_B)
  (h2 : m + n = 150)
  (h3 : m ≥ n / 2)
  : cost_A = 20 ∧ cost_B = 15 ∧ 5 * 50 + 2250 = 2500 
  := by
  sorry

end NUMINAMATH_GPT_seedling_costs_and_purchase_l874_87406


namespace NUMINAMATH_GPT_abs_neg_two_l874_87433

def absolute_value (x : Int) : Int :=
  if x >= 0 then x else -x

theorem abs_neg_two : absolute_value (-2) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_abs_neg_two_l874_87433
