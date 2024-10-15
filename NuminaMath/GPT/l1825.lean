import Mathlib

namespace NUMINAMATH_GPT_difference_is_1365_l1825_182592

-- Define the conditions as hypotheses
def difference_between_numbers (L S : ℕ) : Prop :=
  L = 1637 ∧ L = 6 * S + 5

-- State the theorem to prove the difference is 1365
theorem difference_is_1365 {L S : ℕ} (h₁ : L = 1637) (h₂ : L = 6 * S + 5) :
  L - S = 1365 :=
by
  sorry

end NUMINAMATH_GPT_difference_is_1365_l1825_182592


namespace NUMINAMATH_GPT_lines_forming_angle_bamboo_pole_longest_shadow_angle_l1825_182573

-- Define the angle between sunlight and ground
def angle_sunlight_ground : ℝ := 60

-- Proof problem 1 statement
theorem lines_forming_angle (A : ℝ) : 
  (A > angle_sunlight_ground → ∃ l : ℕ, l = 0) ∧ (A < angle_sunlight_ground → ∃ l : ℕ, ∀ n : ℕ, n > l) :=
  sorry

-- Proof problem 2 statement
theorem bamboo_pole_longest_shadow_angle : 
  ∀ bamboo_pole_angle ground_angle : ℝ, 
  (ground_angle = 60 → bamboo_pole_angle = 30) :=
  sorry

end NUMINAMATH_GPT_lines_forming_angle_bamboo_pole_longest_shadow_angle_l1825_182573


namespace NUMINAMATH_GPT_tangent_line_at_one_minimum_a_range_of_a_l1825_182519

-- Definitions for the given functions
def g (a x : ℝ) := a * x^2 - (a + 2) * x
noncomputable def h (x : ℝ) := Real.log x
noncomputable def f (a x : ℝ) := g a x + h x

-- Part (1): Prove the tangent line equation at x = 1 for a = 1
theorem tangent_line_at_one (x y : ℝ) (h_x : x = 1) (h_a : 1 = (1 : ℝ)) :
  x + y + 1 = 0 := by
  sorry

-- Part (2): Prove the minimum value of a given certain conditions
theorem minimum_a (a : ℝ) (h_a_pos : 0 < a) (h_x : 1 ≤ x ∧ x ≤ Real.exp 1)
  (h_fmin : ∀ x, f a x ≥ -2) : 
  a = 1 := by
  sorry

-- Part (3): Prove the range of values for a given a condition
theorem range_of_a (a x₁ x₂ : ℝ) (h_x : 0 < x₁ ∧ x₁ < x₂) 
  (h_f : ∀ x₁ x₂, (f a x₁ - f a x₂) / (x₁ - x₂) > -2) :
  0 ≤ a ∧ a ≤ 8 := by
  sorry

end NUMINAMATH_GPT_tangent_line_at_one_minimum_a_range_of_a_l1825_182519


namespace NUMINAMATH_GPT_sum_of_ages_l1825_182562

-- Definitions for conditions
def age_product (a b c : ℕ) : Prop := a * b * c = 72
def younger_than_10 (k : ℕ) : Prop := k < 10

-- Main statement
theorem sum_of_ages (a b k : ℕ) (h_product : age_product a b k) (h_twin : a = b) (h_kiana : younger_than_10 k) : 
  a + b + k = 14 := sorry

end NUMINAMATH_GPT_sum_of_ages_l1825_182562


namespace NUMINAMATH_GPT_line_in_slope_intercept_form_l1825_182546

def vec1 : ℝ × ℝ := (3, -7)
def point : ℝ × ℝ := (-2, 4)
def line_eq (x y : ℝ) : Prop := vec1.1 * (x - point.1) + vec1.2 * (y - point.2) = 0

theorem line_in_slope_intercept_form (x y : ℝ) : line_eq x y → y = (3 / 7) * x - (34 / 7) :=
by
  sorry

end NUMINAMATH_GPT_line_in_slope_intercept_form_l1825_182546


namespace NUMINAMATH_GPT_arrests_per_day_in_each_city_l1825_182579

-- Define the known conditions
def daysOfProtest := 30
def numberOfCities := 21
def daysInJailBeforeTrial := 4
def daysInJailAfterTrial := 7 / 2 * 7 -- half of a 2-week sentence in days, converted from weeks to days
def combinedJailTimeInWeeks := 9900
def combinedJailTimeInDays := combinedJailTimeInWeeks * 7

-- Define the proof statement
theorem arrests_per_day_in_each_city :
  (combinedJailTimeInDays / (daysInJailBeforeTrial + daysInJailAfterTrial)) / daysOfProtest / numberOfCities = 10 := 
by
  sorry

end NUMINAMATH_GPT_arrests_per_day_in_each_city_l1825_182579


namespace NUMINAMATH_GPT_minimum_chess_pieces_l1825_182582

theorem minimum_chess_pieces (n : ℕ) : 
  (n % 3 = 1) ∧ (n % 5 = 3) ∧ (n % 7 = 5) → 
  n = 103 :=
by 
  sorry

end NUMINAMATH_GPT_minimum_chess_pieces_l1825_182582


namespace NUMINAMATH_GPT_cosine_value_parallel_vectors_l1825_182510

theorem cosine_value_parallel_vectors (α : ℝ) (h1 : ∃ (a : ℝ × ℝ) (b : ℝ × ℝ), a = (Real.cos (Real.pi / 3 + α), 1) ∧ b = (1, 4) ∧ a.1 * b.2 - a.2 * b.1 = 0) : 
  Real.cos (Real.pi / 3 - 2 * α) = 7 / 8 := by
  sorry

end NUMINAMATH_GPT_cosine_value_parallel_vectors_l1825_182510


namespace NUMINAMATH_GPT_parts_of_second_liquid_l1825_182595

theorem parts_of_second_liquid (x : ℝ) :
    (0.10 * 5 + 0.15 * x) / (5 + x) = 11.42857142857143 / 100 ↔ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_parts_of_second_liquid_l1825_182595


namespace NUMINAMATH_GPT_correct_statements_identification_l1825_182564

-- Definitions based on given conditions
def syntheticMethodCauseToEffect := True
def syntheticMethodForward := True
def analyticMethodEffectToCause := True
def analyticMethodIndirect := False
def analyticMethodBackward := True

-- The main statement to be proved
theorem correct_statements_identification :
  (syntheticMethodCauseToEffect = True) ∧ 
  (syntheticMethodForward = True) ∧ 
  (analyticMethodEffectToCause = True) ∧ 
  (analyticMethodBackward = True) ∧ 
  (analyticMethodIndirect = False) :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_identification_l1825_182564


namespace NUMINAMATH_GPT_find_sticker_price_l1825_182576

-- Define the conditions and the question
def storeA_price (x : ℝ) : ℝ := 0.80 * x - 80
def storeB_price (x : ℝ) : ℝ := 0.70 * x - 40
def heather_saves_30 (x : ℝ) : Prop := storeA_price x = storeB_price x + 30

-- Define the main theorem
theorem find_sticker_price : ∃ x : ℝ, heather_saves_30 x ∧ x = 700 :=
by
  sorry

end NUMINAMATH_GPT_find_sticker_price_l1825_182576


namespace NUMINAMATH_GPT_Cornelia_current_age_l1825_182537

theorem Cornelia_current_age (K : ℕ) (C : ℕ) (h1 : K = 20) (h2 : C + 10 = 3 * (K + 10)) : C = 80 :=
by
  sorry

end NUMINAMATH_GPT_Cornelia_current_age_l1825_182537


namespace NUMINAMATH_GPT_mr_c_gain_1000_l1825_182542

-- Define the initial conditions
def initial_mr_c_cash := 15000
def initial_mr_c_house := 12000
def initial_mrs_d_cash := 16000

-- Define the changes in the house value
def house_value_appreciated := 13000
def house_value_depreciated := 11000

-- Define the cash changes after transactions
def mr_c_cash_after_first_sale := initial_mr_c_cash + house_value_appreciated
def mrs_d_cash_after_first_sale := initial_mrs_d_cash - house_value_appreciated
def mrs_d_cash_after_second_sale := mrs_d_cash_after_first_sale + house_value_depreciated
def mr_c_cash_after_second_sale := mr_c_cash_after_first_sale - house_value_depreciated

-- Define the final net worth for Mr. C
def final_mr_c_cash := mr_c_cash_after_second_sale
def final_mr_c_house := house_value_depreciated
def final_mr_c_net_worth := final_mr_c_cash + final_mr_c_house
def initial_mr_c_net_worth := initial_mr_c_cash + initial_mr_c_house

-- Statement to prove
theorem mr_c_gain_1000 : final_mr_c_net_worth = initial_mr_c_net_worth + 1000 := by
  sorry

end NUMINAMATH_GPT_mr_c_gain_1000_l1825_182542


namespace NUMINAMATH_GPT_at_least_one_travels_l1825_182505

-- Define the probabilities of A and B traveling
def P_A := 1 / 3
def P_B := 1 / 4

-- Define the probability that person A does not travel
def P_not_A := 1 - P_A

-- Define the probability that person B does not travel
def P_not_B := 1 - P_B

-- Define the probability that neither person A nor person B travels
def P_neither := P_not_A * P_not_B

-- Define the probability that at least one of them travels
def P_at_least_one := 1 - P_neither

theorem at_least_one_travels : P_at_least_one = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_at_least_one_travels_l1825_182505


namespace NUMINAMATH_GPT_divisible_by_12_l1825_182548

theorem divisible_by_12 (n : ℤ) : 12 ∣ (n^4 - n^2) := sorry

end NUMINAMATH_GPT_divisible_by_12_l1825_182548


namespace NUMINAMATH_GPT_no_integer_cube_eq_3n_squared_plus_3n_plus_7_l1825_182559

theorem no_integer_cube_eq_3n_squared_plus_3n_plus_7 :
  ¬ ∃ x n : ℤ, x^3 = 3 * n^2 + 3 * n + 7 := 
sorry

end NUMINAMATH_GPT_no_integer_cube_eq_3n_squared_plus_3n_plus_7_l1825_182559


namespace NUMINAMATH_GPT_nth_term_arithmetic_seq_l1825_182566

variable (a_n : Nat → Int)
variable (S : Nat → Int)
variable (a_1 : Int)

-- Conditions
def is_arithmetic_sequence (a_n : Nat → Int) : Prop :=
  ∃ d : Int, ∀ n : Nat, a_n (n + 1) = a_n n + d

def first_term (a_1 : Int) : Prop :=
  a_1 = 1

def sum_first_three_terms (S : Nat → Int) : Prop :=
  S 3 = 9

theorem nth_term_arithmetic_seq :
  (is_arithmetic_sequence a_n) →
  (first_term 1) →
  (sum_first_three_terms S) →
  ∀ n : Nat, a_n n = 2 * n - 1 :=
  sorry

end NUMINAMATH_GPT_nth_term_arithmetic_seq_l1825_182566


namespace NUMINAMATH_GPT_polynomial_zero_pairs_l1825_182551

theorem polynomial_zero_pairs (r s : ℝ) :
  (∀ x : ℝ, (x = 0 ∨ x = 0) ↔ x^2 - 2 * r * x + r = 0) ∧
  (∀ x : ℝ, (x = 0 ∨ x = 0 ∨ x = 0) ↔ 27 * x^3 - 27 * r * x^2 + s * x - r^6 = 0) → 
  (r, s) = (0, 0) ∨ (r, s) = (1, 9) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_zero_pairs_l1825_182551


namespace NUMINAMATH_GPT_animal_legs_count_l1825_182575

-- Let's define the conditions first.
def total_animals : ℕ := 12
def chickens : ℕ := 5
def chicken_legs : ℕ := 2
def sheep_legs : ℕ := 4

-- Define the statement that we need to prove.
theorem animal_legs_count :
  ∃ (total_legs : ℕ), total_legs = 38 :=
by
  -- Adding the condition for total number of legs
  let sheep := total_animals - chickens
  let total_legs := (chickens * chicken_legs) + (sheep * sheep_legs)
  existsi total_legs
  -- Question proves the correct answer
  sorry

end NUMINAMATH_GPT_animal_legs_count_l1825_182575


namespace NUMINAMATH_GPT_can_determine_number_of_spies_l1825_182554

def determine_spies (V : Fin 15 → ℕ) (S : Fin 15 → ℕ) : Prop :=
  V 0 = S 0 + S 1 ∧ 
  ∀ i : Fin 13, V (Fin.succ (Fin.succ i)) = S i + S (Fin.succ i) + S (Fin.succ (Fin.succ i)) ∧
  V 14 = S 13 + S 14

theorem can_determine_number_of_spies :
  ∃ S : Fin 15 → ℕ, ∀ V : Fin 15 → ℕ, determine_spies V S :=
sorry

end NUMINAMATH_GPT_can_determine_number_of_spies_l1825_182554


namespace NUMINAMATH_GPT_single_room_cost_l1825_182526

theorem single_room_cost (total_rooms : ℕ) (single_rooms : ℕ) (double_room_cost : ℕ) 
  (total_revenue : ℤ) (x : ℤ) : 
  total_rooms = 260 → 
  single_rooms = 64 → 
  double_room_cost = 60 → 
  total_revenue = 14000 → 
  64 * x + (total_rooms - single_rooms) * double_room_cost = total_revenue → 
  x = 35 := 
by 
  intros h_total_rooms h_single_rooms h_double_room_cost h_total_revenue h_eqn 
  -- Add steps for proving if necessary
  sorry

end NUMINAMATH_GPT_single_room_cost_l1825_182526


namespace NUMINAMATH_GPT_can_encode_number_l1825_182540

theorem can_encode_number : ∃ (m n : ℕ), (0.07 = 1 / (m : ℝ) + 1 / (n : ℝ)) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_can_encode_number_l1825_182540


namespace NUMINAMATH_GPT_find_m_l1825_182527

variable {a b c m : ℝ}

theorem find_m (h1 : a + b = 4)
               (h2 : a * b = m)
               (h3 : b + c = 8)
               (h4 : b * c = 5 * m) : m = 0 ∨ m = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_m_l1825_182527


namespace NUMINAMATH_GPT_household_A_bill_bill_formula_household_B_usage_household_C_usage_l1825_182504

-- Definition of the tiered water price system
def water_bill (x : ℕ) : ℕ :=
if x <= 22 then 3 * x
else if x <= 30 then 3 * 22 + 5 * (x - 22)
else 3 * 22 + 5 * 8 + 7 * (x - 30)

-- Prove that if a household uses 25m^3 of water, the water bill is 81 yuan.
theorem household_A_bill : water_bill 25 = 81 := by 
  sorry

-- Prove that the formula for the water bill when x > 30 is y = 7x - 104.
theorem bill_formula (x : ℕ) (hx : x > 30) : water_bill x = 7 * x - 104 := by 
  sorry

-- Prove that if a household paid 120 yuan for water, their usage was 32m^3.
theorem household_B_usage : ∃ x : ℕ, water_bill x = 120 ∧ x = 32 := by 
  sorry

-- Prove that if household C uses a total of 50m^3 over May and June with a total bill of 174 yuan, their usage was 18m^3 in May and 32m^3 in June.
theorem household_C_usage (a b : ℕ) (ha : a + b = 50) (hb : a < b) (total_bill : water_bill a + water_bill b = 174) :
  a = 18 ∧ b = 32 := by
  sorry

end NUMINAMATH_GPT_household_A_bill_bill_formula_household_B_usage_household_C_usage_l1825_182504


namespace NUMINAMATH_GPT_initial_dimes_l1825_182520

theorem initial_dimes (dimes_received_from_dad : ℕ) (dimes_received_from_mom : ℕ) (total_dimes_now : ℕ) : 
  dimes_received_from_dad = 8 → dimes_received_from_mom = 4 → total_dimes_now = 19 → 
  total_dimes_now - (dimes_received_from_dad + dimes_received_from_mom) = 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_initial_dimes_l1825_182520


namespace NUMINAMATH_GPT_y_work_days_eq_10_l1825_182502

noncomputable def work_days_y (W d : ℝ) : Prop :=
  let work_rate_x := W / 30
  let work_rate_y := W / 15
  let days_x_remaining := 10.000000000000002
  let work_done_by_y := d * work_rate_y
  let work_done_by_x := days_x_remaining * work_rate_x
  work_done_by_y + work_done_by_x = W

/-- The number of days y worked before leaving the job is 10 -/
theorem y_work_days_eq_10 (W : ℝ) : work_days_y W 10 :=
by
  sorry

end NUMINAMATH_GPT_y_work_days_eq_10_l1825_182502


namespace NUMINAMATH_GPT_solution_exists_l1825_182570

def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x + a^2

def f' (x a b : ℝ) : ℝ := 3 * x^2 - 2 * a * x - b

theorem solution_exists (a b : ℝ) :
    f 1 a b = 10 ∧ f' 1 a b = 0 ↔ (a = -4 ∧ b = 11) :=
by 
  sorry

end NUMINAMATH_GPT_solution_exists_l1825_182570


namespace NUMINAMATH_GPT_inequality_three_var_l1825_182514

theorem inequality_three_var
  (a b c : ℝ)
  (ha : 0 ≤ a)
  (hb : 0 ≤ b)
  (hc : 0 ≤ c) :
  2 * (a^3 + b^3 + c^3) ≥ a^2 * b + a * b^2 + a^2 * c + a * c^2 + b^2 * c + b * c^2 :=
by sorry

end NUMINAMATH_GPT_inequality_three_var_l1825_182514


namespace NUMINAMATH_GPT_cannot_invert_all_signs_l1825_182558

structure RegularDecagon :=
  (vertices : Fin 10 → ℤ)
  (diagonals : Fin 45 → ℤ) -- Assume we encode the intersections as unique indices for simplicity.
  (all_positives : ∀ v, vertices v = 1 ∧ ∀ d, diagonals d = 1)

def isValidSignChange (t : List ℤ) : Prop :=
  t.length % 2 = 0

theorem cannot_invert_all_signs (D : RegularDecagon) :
  ¬ (∃ f : Fin 10 → ℤ → ℤ, ∀ (side : Fin 10) (val : ℤ), f side val = -val) :=
sorry

end NUMINAMATH_GPT_cannot_invert_all_signs_l1825_182558


namespace NUMINAMATH_GPT_cats_to_dogs_ratio_l1825_182532

theorem cats_to_dogs_ratio (cats dogs : ℕ) (h1 : 2 * dogs = 3 * cats) (h2 : cats = 14) : dogs = 21 :=
by
  sorry

end NUMINAMATH_GPT_cats_to_dogs_ratio_l1825_182532


namespace NUMINAMATH_GPT_integer_root_b_l1825_182515

theorem integer_root_b (a1 a2 a3 a4 a5 b : ℤ)
  (h_diff : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a3 ≠ a4 ∧ a3 ≠ a5 ∧ a4 ≠ a5)
  (h_sum : a1 + a2 + a3 + a4 + a5 = 9)
  (h_prod : (b - a1) * (b - a2) * (b - a3) * (b - a4) * (b - a5) = 2009) :
  b = 10 :=
sorry

end NUMINAMATH_GPT_integer_root_b_l1825_182515


namespace NUMINAMATH_GPT_original_class_strength_l1825_182544

theorem original_class_strength (x : ℕ) 
    (avg_original : ℕ)
    (num_new : ℕ) 
    (avg_new : ℕ) 
    (decrease : ℕ)
    (h1 : avg_original = 40)
    (h2 : num_new = 17)
    (h3 : avg_new = 32)
    (h4 : decrease = 4)
    (h5 : (40 * x + 17 * avg_new) = (x + num_new) * (40 - decrease))
    : x = 17 := 
by {
  sorry
}

end NUMINAMATH_GPT_original_class_strength_l1825_182544


namespace NUMINAMATH_GPT_domain_of_h_l1825_182534

noncomputable def h (x : ℝ) : ℝ :=
  (x^2 - 9) / (abs (x - 4) + x^2 - 1)

theorem domain_of_h :
  ∀ (x : ℝ), x ≠ (1 + Real.sqrt 13) / 2 → (abs (x - 4) + x^2 - 1) ≠ 0 :=
sorry

end NUMINAMATH_GPT_domain_of_h_l1825_182534


namespace NUMINAMATH_GPT_solve_equation_l1825_182545

open Function

theorem solve_equation (m n : ℕ) (h_gcd : gcd m n = 2) (h_lcm : lcm m n = 4) :
  m * n = (gcd m n)^2 + lcm m n ↔ (m = 2 ∧ n = 4) ∨ (m = 4 ∧ n = 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1825_182545


namespace NUMINAMATH_GPT_curve_intersection_l1825_182567

noncomputable def C1 (t : ℝ) (a : ℝ) : ℝ × ℝ :=
  (2 * t + 2 * a, -t)

noncomputable def C2 (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.sin θ, 1 + 2 * Real.cos θ)

theorem curve_intersection (a : ℝ) :
  (∃ t θ : ℝ, C1 t a = C2 θ) ↔ 1 - Real.sqrt 5 ≤ a ∧ a ≤ 1 + Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_curve_intersection_l1825_182567


namespace NUMINAMATH_GPT_number_of_ordered_pairs_l1825_182547

theorem number_of_ordered_pairs (x y : ℕ) : (x * y = 1716) → 
  (∃! n : ℕ, n = 18) :=
by
  sorry

end NUMINAMATH_GPT_number_of_ordered_pairs_l1825_182547


namespace NUMINAMATH_GPT_expression_positive_l1825_182596

theorem expression_positive (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) : 5 * a ^ 2 - 6 * a * b + 5 * b ^ 2 > 0 :=
by
  sorry

end NUMINAMATH_GPT_expression_positive_l1825_182596


namespace NUMINAMATH_GPT_part1_a_eq_neg1_inter_part1_a_eq_neg1_union_part2_a_range_l1825_182583

def A (x : ℝ) : Prop := x^2 - 4 * x - 5 ≥ 0
def B (x a : ℝ) : Prop := 2 * a ≤ x ∧ x ≤ a + 2

theorem part1_a_eq_neg1_inter (a : ℝ) (h : a = -1) : 
  {x : ℝ | A x} ∩ {x : ℝ | B x a} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} :=
by sorry

theorem part1_a_eq_neg1_union (a : ℝ) (h : a = -1) : 
  {x : ℝ | A x} ∪ {x : ℝ | B x a} = {x : ℝ | x ≤ 1 ∨ x ≥ 5} :=
by sorry

theorem part2_a_range (a : ℝ) : 
  ({x : ℝ | A x} ∩ {x : ℝ | B x a} = {x : ℝ | B x a}) → 
  a ∈ {a : ℝ | a > 2 ∨ a ≤ -3} :=
by sorry

end NUMINAMATH_GPT_part1_a_eq_neg1_inter_part1_a_eq_neg1_union_part2_a_range_l1825_182583


namespace NUMINAMATH_GPT_question1_question2_question3_question4_l1825_182550

theorem question1 : (2 * 3) ^ 2 = 2 ^ 2 * 3 ^ 2 := by admit

theorem question2 : (-1 / 2 * 2) ^ 3 = (-1 / 2) ^ 3 * 2 ^ 3 := by admit

theorem question3 : (3 / 2) ^ 2019 * (-2 / 3) ^ 2019 = -1 := by admit

theorem question4 (a b : ℝ) (n : ℕ) (h : 0 < n): (a * b) ^ n = a ^ n * b ^ n := by admit

end NUMINAMATH_GPT_question1_question2_question3_question4_l1825_182550


namespace NUMINAMATH_GPT_max_cart_length_l1825_182506

-- Definitions for the hallway and cart dimensions
def hallway_width : ℝ := 1.5
def cart_width : ℝ := 1

-- The proposition stating the maximum length of the cart that can smoothly navigate the hallway
theorem max_cart_length : ∃ L : ℝ, L = 3 * Real.sqrt 2 ∧
  (∀ (a b : ℝ), a > 0 ∧ b > 0 → (3 / a) + (3 / b) = 2 → Real.sqrt (a^2 + b^2) = L) :=
  sorry

end NUMINAMATH_GPT_max_cart_length_l1825_182506


namespace NUMINAMATH_GPT_final_sign_is_minus_l1825_182524

theorem final_sign_is_minus 
  (plus_count : ℕ) 
  (minus_count : ℕ) 
  (h_plus : plus_count = 2004) 
  (h_minus : minus_count = 2005) 
  (transform : (ℕ → ℕ → ℕ × ℕ) → Prop) :
  transform (fun plus minus =>
    if plus >= 2 then (plus - 1, minus)
    else if minus >= 2 then (plus, minus - 1)
    else if plus > 0 && minus > 0 then (plus - 1, minus - 1)
    else (0, 0)) →
  (plus_count = 0 ∧ minus_count = 1) := sorry

end NUMINAMATH_GPT_final_sign_is_minus_l1825_182524


namespace NUMINAMATH_GPT_smallest_possible_x2_plus_y2_l1825_182597

theorem smallest_possible_x2_plus_y2 (x y : ℝ) (h : (x + 3) * (y - 3) = 0) : x^2 + y^2 = 18 :=
sorry

end NUMINAMATH_GPT_smallest_possible_x2_plus_y2_l1825_182597


namespace NUMINAMATH_GPT_positive_solution_l1825_182584

variable {x y z : ℝ}

theorem positive_solution (h1 : x * y = 8 - 2 * x - 3 * y)
    (h2 : y * z = 8 - 4 * y - 2 * z)
    (h3 : x * z = 40 - 5 * x - 3 * z) :
    x = 10 := by
  sorry

end NUMINAMATH_GPT_positive_solution_l1825_182584


namespace NUMINAMATH_GPT_sum_of_remainders_and_smallest_n_l1825_182585

theorem sum_of_remainders_and_smallest_n (n : ℕ) (h : n % 20 = 11) :
    (n % 4 + n % 5 = 4) ∧ (∃ (k : ℕ), k > 2 ∧ n = 20 * k + 11 ∧ n > 50) := by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_and_smallest_n_l1825_182585


namespace NUMINAMATH_GPT_temperature_below_zero_l1825_182574

-- Assume the basic definitions and context needed
def above_zero (temp : Int) := temp > 0
def below_zero (temp : Int) := temp < 0

theorem temperature_below_zero (t1 t2 : Int) (h1 : above_zero t1) (h2 : t2 = -7) :
  below_zero t2 := by 
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_temperature_below_zero_l1825_182574


namespace NUMINAMATH_GPT_range_of_a1_l1825_182516

theorem range_of_a1 {a : ℕ → ℝ} (h_seq : ∀ n : ℕ, 2 * a (n + 1) * a n + a (n + 1) - 3 * a n = 0)
  (h_a1_positive : a 1 > 0) :
  (0 < a 1) ∧ (a 1 < 1) ↔ ∀ m n : ℕ, m < n → a m < a n := by
  sorry

end NUMINAMATH_GPT_range_of_a1_l1825_182516


namespace NUMINAMATH_GPT_binomial_expansion_l1825_182591

theorem binomial_expansion : 
  (102: ℕ)^4 - 4 * (102: ℕ)^3 + 6 * (102: ℕ)^2 - 4 * (102: ℕ) + 1 = (101: ℕ)^4 :=
by sorry

end NUMINAMATH_GPT_binomial_expansion_l1825_182591


namespace NUMINAMATH_GPT_arithmetic_sequence_a7_l1825_182586

theorem arithmetic_sequence_a7 :
  ∀ (a : ℕ → ℕ) (d : ℕ),
  (∀ n, a (n + 1) = a n + d) →
  a 1 = 2 →
  a 3 + a 5 = 10 →
  a 7 = 8 :=
by
  intros a d h_seq h_a1 h_sum
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a7_l1825_182586


namespace NUMINAMATH_GPT_base_b_prime_digits_l1825_182533

theorem base_b_prime_digits (b' : ℕ) (h1 : b'^4 ≤ 216) (h2 : 216 < b'^5) : b' = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_base_b_prime_digits_l1825_182533


namespace NUMINAMATH_GPT_quotient_of_division_l1825_182539

theorem quotient_of_division (S L Q : ℕ) (h1 : S = 270) (h2 : L - S = 1365) (h3 : L % S = 15) : Q = 6 :=
by
  sorry

end NUMINAMATH_GPT_quotient_of_division_l1825_182539


namespace NUMINAMATH_GPT_compute_area_ratio_l1825_182500

noncomputable def area_ratio (K : ℝ) : ℝ :=
  let small_triangle_area := 2 * K
  let large_triangle_area := 8 * K
  small_triangle_area / large_triangle_area

theorem compute_area_ratio (K : ℝ) : area_ratio K = 1 / 4 :=
by
  unfold area_ratio
  sorry

end NUMINAMATH_GPT_compute_area_ratio_l1825_182500


namespace NUMINAMATH_GPT_y_intercept_of_line_l1825_182530

theorem y_intercept_of_line (x y : ℝ) (eq : 2 * x - 3 * y = 6) (hx : x = 0) : y = -2 := 
by
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l1825_182530


namespace NUMINAMATH_GPT_intersection_three_points_l1825_182557

def circle_eq (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = a^2
def parabola_eq (a : ℝ) (x y : ℝ) : Prop := y = x^2 - 3 * a

theorem intersection_three_points (a : ℝ) :
  (∃ x1 y1 x2 y2 x3 y3 : ℝ, 
    circle_eq a x1 y1 ∧ parabola_eq a x1 y1 ∧
    circle_eq a x2 y2 ∧ parabola_eq a x2 y2 ∧
    circle_eq a x3 y3 ∧ parabola_eq a x3 y3 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    (x1 ≠ x3 ∨ y1 ≠ y3) ∧
    (x2 ≠ x3 ∨ y2 ≠ y3)) ↔ 
  a = 1/3 := by
  sorry

end NUMINAMATH_GPT_intersection_three_points_l1825_182557


namespace NUMINAMATH_GPT_valid_votes_election_l1825_182509

-- Definition of the problem
variables (V : ℝ) -- the total number of valid votes
variables (hvoting_percentage : V > 0 ∧ V ≤ 1) -- constraints for voting percentage in general
variables (h_winning_votes : 0.70 * V) -- 70% of the votes
variables (h_losing_votes : 0.30 * V) -- 30% of the votes

-- Given condition: the winning candidate won by a majority of 184 votes
variables (majority : ℝ) (h_majority : 0.70 * V - 0.30 * V = 184)

/-- The total number of valid votes in the election. -/
theorem valid_votes_election : V = 460 :=
by
  sorry

end NUMINAMATH_GPT_valid_votes_election_l1825_182509


namespace NUMINAMATH_GPT_kevin_hopped_distance_after_four_hops_l1825_182590

noncomputable def kevin_total_hopped_distance : ℚ :=
  let hop1 := 1
  let hop2 := 1 / 2
  let hop3 := 1 / 4
  let hop4 := 1 / 8
  hop1 + hop2 + hop3 + hop4

theorem kevin_hopped_distance_after_four_hops :
  kevin_total_hopped_distance = 15 / 8 :=
by
  sorry

end NUMINAMATH_GPT_kevin_hopped_distance_after_four_hops_l1825_182590


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1825_182522

-- Definitions for sides opposite angles A, B, and C in a triangle.
variables {A B C : Real} {a b c : Real}

-- Condition p: sides a, b related to angles A, B via cosine
def condition_p (a b : Real) (A B : Real) : Prop := a / Real.cos A = b / Real.cos B

-- Condition q: sides a and b are equal
def condition_q (a b : Real) : Prop := a = b

theorem necessary_and_sufficient_condition (h1 : condition_p a b A B) : condition_q a b ↔ condition_p a b A B :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1825_182522


namespace NUMINAMATH_GPT_soccer_tournament_matches_l1825_182577

theorem soccer_tournament_matches (n: ℕ):
  n = 20 → ∃ m: ℕ, m = 19 := sorry

end NUMINAMATH_GPT_soccer_tournament_matches_l1825_182577


namespace NUMINAMATH_GPT_dominic_domino_problem_l1825_182521

theorem dominic_domino_problem 
  (num_dominoes : ℕ)
  (pips_pairs : ℕ → ℕ)
  (hexagonal_ring : ℕ → ℕ → Prop) : 
  ∀ (adj : ℕ → ℕ → Prop), 
  num_dominoes = 6 → 
  (∀ i j, hexagonal_ring i j → pips_pairs i = pips_pairs j) →
  ∃ k, k = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_dominic_domino_problem_l1825_182521


namespace NUMINAMATH_GPT_xy_equals_twelve_l1825_182531

theorem xy_equals_twelve (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 :=
by
  sorry

end NUMINAMATH_GPT_xy_equals_twelve_l1825_182531


namespace NUMINAMATH_GPT_intersection_x_sum_l1825_182569

theorem intersection_x_sum :
  ∃ x : ℤ, (0 ≤ x ∧ x < 17) ∧ (4 * x + 3 ≡ 13 * x + 14 [ZMOD 17]) ∧ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_intersection_x_sum_l1825_182569


namespace NUMINAMATH_GPT_prod_mod_6_l1825_182541

theorem prod_mod_6 (h1 : 2015 % 6 = 3) (h2 : 2016 % 6 = 0) (h3 : 2017 % 6 = 1) (h4 : 2018 % 6 = 2) : 
  (2015 * 2016 * 2017 * 2018) % 6 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_prod_mod_6_l1825_182541


namespace NUMINAMATH_GPT_trig_identity_l1825_182523

theorem trig_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  (Real.sin θ + Real.cos θ) / Real.sin θ + Real.sin θ * Real.sin θ = 23 / 10 :=
sorry

end NUMINAMATH_GPT_trig_identity_l1825_182523


namespace NUMINAMATH_GPT_choose_roles_from_8_l1825_182503

-- Define the number of people
def num_people : ℕ := 8
-- Define the function to count the number of ways to choose different persons for the roles
def choose_roles (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

theorem choose_roles_from_8 : choose_roles num_people = 336 := by
  -- sorry acts as a placeholder for the proof
  sorry

end NUMINAMATH_GPT_choose_roles_from_8_l1825_182503


namespace NUMINAMATH_GPT_largest_decimal_of_four_digit_binary_l1825_182598

theorem largest_decimal_of_four_digit_binary : ∀ n : ℕ, (n < 16) → n ≤ 15 :=
by {
  -- conditions: a four-digit binary number implies \( n \) must be less than \( 2^4 = 16 \)
  sorry
}

end NUMINAMATH_GPT_largest_decimal_of_four_digit_binary_l1825_182598


namespace NUMINAMATH_GPT_smallest_multiple_of_37_smallest_multiple_of_37_verification_l1825_182561

theorem smallest_multiple_of_37 (x : ℕ) (h : 37 * x % 97 = 3) :
  x = 15 := sorry

theorem smallest_multiple_of_37_verification :
  37 * 15 = 555 := rfl

end NUMINAMATH_GPT_smallest_multiple_of_37_smallest_multiple_of_37_verification_l1825_182561


namespace NUMINAMATH_GPT_return_trip_avg_speed_l1825_182580

noncomputable def avg_speed_return_trip : ℝ := 
  let distance_ab_to_sy := 120
  let rate_ab_to_sy := 50
  let total_time := 5.5
  let time_ab_to_sy := distance_ab_to_sy / rate_ab_to_sy
  let time_return_trip := total_time - time_ab_to_sy
  distance_ab_to_sy / time_return_trip

theorem return_trip_avg_speed 
  (distance_ab_to_sy : ℝ := 120)
  (rate_ab_to_sy : ℝ := 50)
  (total_time : ℝ := 5.5) 
  : avg_speed_return_trip = 38.71 :=
by
  sorry

end NUMINAMATH_GPT_return_trip_avg_speed_l1825_182580


namespace NUMINAMATH_GPT_power_mod_remainder_l1825_182587

theorem power_mod_remainder (a : ℕ) (n : ℕ) (h1 : 3^5 % 11 = 1) (h2 : 221 % 5 = 1) : 3^221 % 11 = 3 :=
by
  sorry

end NUMINAMATH_GPT_power_mod_remainder_l1825_182587


namespace NUMINAMATH_GPT_bookseller_fiction_books_count_l1825_182568

theorem bookseller_fiction_books_count (n : ℕ) (h1 : n.factorial * 6 = 36) : n = 3 :=
sorry

end NUMINAMATH_GPT_bookseller_fiction_books_count_l1825_182568


namespace NUMINAMATH_GPT_largest_possible_M_l1825_182556

theorem largest_possible_M (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h_cond : x * y + y * z + z * x = 1) :
    ∃ M, ∀ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y + y * z + z * x = 1 → 
    (x / (1 + yz/x) + y / (1 + zx/y) + z / (1 + xy/z) ≥ M) → 
        M = 3 / (Real.sqrt 3 + 1) :=
by
  sorry        

end NUMINAMATH_GPT_largest_possible_M_l1825_182556


namespace NUMINAMATH_GPT_phase_shift_of_sine_l1825_182535

theorem phase_shift_of_sine (b c : ℝ) (h_b : b = 4) (h_c : c = - (Real.pi / 2)) :
  (-c / b) = Real.pi / 8 :=
by
  rw [h_b, h_c]
  sorry

end NUMINAMATH_GPT_phase_shift_of_sine_l1825_182535


namespace NUMINAMATH_GPT_convert_mps_to_kmph_l1825_182571

theorem convert_mps_to_kmph (v_mps : ℝ) (c : ℝ) (h_c : c = 3.6) (h_v_mps : v_mps = 20) : (v_mps * c = 72) :=
by
  rw [h_v_mps, h_c]
  sorry

end NUMINAMATH_GPT_convert_mps_to_kmph_l1825_182571


namespace NUMINAMATH_GPT_max_distance_circle_to_line_l1825_182529

open Real

-- Definitions of polar equations and transformations to Cartesian coordinates
def circle_eq (ρ θ : ℝ) : Prop := (ρ = 8 * sin θ)
def line_eq (θ : ℝ) : Prop := (θ = π / 3)

-- Cartesian coordinate transformations
def circle_cartesian (x y : ℝ) : Prop := (x^2 + (y - 4)^2 = 16)
def line_cartesian (x y : ℝ) : Prop := (y = sqrt 3 * x)

-- Maximum distance problem statement
theorem max_distance_circle_to_line : 
  ∀ (x y : ℝ), circle_cartesian x y → 
  (∀ x y, line_cartesian x y → 
  ∃ d : ℝ, d = 6) :=
by
  sorry

end NUMINAMATH_GPT_max_distance_circle_to_line_l1825_182529


namespace NUMINAMATH_GPT_tan_ratio_l1825_182581

open Real

variables (p q : ℝ)

-- Conditions
def cond1 := (sin p / cos q + sin q / cos p = 2)
def cond2 := (cos p / sin q + cos q / sin p = 3)

-- Proof statement
theorem tan_ratio (hpq : cond1 p q) (hq : cond2 p q) :
  (tan p / tan q + tan q / tan p = 8 / 5) :=
sorry

end NUMINAMATH_GPT_tan_ratio_l1825_182581


namespace NUMINAMATH_GPT_carl_gave_beth_35_coins_l1825_182536

theorem carl_gave_beth_35_coins (x : ℕ) (h1 : ∃ n, n = 125) (h2 : ∃ m, m = (125 + x) / 2) (h3 : m = 80) : x = 35 :=
by
  sorry

end NUMINAMATH_GPT_carl_gave_beth_35_coins_l1825_182536


namespace NUMINAMATH_GPT_triangle_third_side_length_l1825_182589

theorem triangle_third_side_length (A B C : Type) 
  (AB : ℝ) (AC : ℝ) 
  (angle_ABC angle_ACB : ℝ) 
  (BC : ℝ) 
  (h1 : AB = 7) 
  (h2 : AC = 21) 
  (h3 : angle_ABC = 3 * angle_ACB) 
  : 
  BC = (some_correct_value ) := 
sorry

end NUMINAMATH_GPT_triangle_third_side_length_l1825_182589


namespace NUMINAMATH_GPT_apple_production_l1825_182543

variable {S1 S2 S3 : ℝ}

theorem apple_production (h1 : S2 = 0.8 * S1) 
                         (h2 : S3 = 2 * S2) 
                         (h3 : S1 + S2 + S3 = 680) : 
                         S1 = 200 := 
by
  sorry

end NUMINAMATH_GPT_apple_production_l1825_182543


namespace NUMINAMATH_GPT_original_percentage_alcohol_l1825_182511

-- Definitions of the conditions
def original_mixture_volume : ℝ := 15
def additional_water_volume : ℝ := 3
def final_percentage_alcohol : ℝ := 20.833333333333336
def final_mixture_volume : ℝ := original_mixture_volume + additional_water_volume

-- Lean statement to prove
theorem original_percentage_alcohol (A : ℝ) :
  (A / 100 * original_mixture_volume) = (final_percentage_alcohol / 100 * final_mixture_volume) →
  A = 25 :=
by
  sorry

end NUMINAMATH_GPT_original_percentage_alcohol_l1825_182511


namespace NUMINAMATH_GPT_divisibility_equivalence_distinct_positive_l1825_182517

variable (a b c : ℕ)

theorem divisibility_equivalence_distinct_positive (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a + b + c) ∣ (a^3 * b + b^3 * c + c^3 * a)) ↔ ((a + b + c) ∣ (a * b^3 + b * c^3 + c * a^3)) :=
by sorry

end NUMINAMATH_GPT_divisibility_equivalence_distinct_positive_l1825_182517


namespace NUMINAMATH_GPT_calculate_difference_l1825_182552

theorem calculate_difference (x y : ℝ) (h1 : x + y = 520) (h2 : x / y = 0.75) : y - x = 74 :=
by
  sorry

end NUMINAMATH_GPT_calculate_difference_l1825_182552


namespace NUMINAMATH_GPT_ratio_shiny_igneous_to_total_l1825_182512

-- Define the conditions
variable (S I SI : ℕ)
variable (SS : ℕ)
variable (h1 : I = S / 2)
variable (h2 : SI = 40)
variable (h3 : S + I = 180)
variable (h4 : SS = S / 5)

-- Statement to prove
theorem ratio_shiny_igneous_to_total (S I SI SS : ℕ) 
  (h1 : I = S / 2) 
  (h2 : SI = 40) 
  (h3 : S + I = 180) 
  (h4 : SS = S / 5) : 
  SI / I = 2 / 3 := 
sorry

end NUMINAMATH_GPT_ratio_shiny_igneous_to_total_l1825_182512


namespace NUMINAMATH_GPT_nine_digit_positive_integers_l1825_182553

theorem nine_digit_positive_integers :
  (∃ n : Nat, 10^8 * 9 = n ∧ n = 900000000) :=
sorry

end NUMINAMATH_GPT_nine_digit_positive_integers_l1825_182553


namespace NUMINAMATH_GPT_problem_statement_l1825_182565

theorem problem_statement (r p q : ℝ) (h1 : r > 0) (h2 : p * q ≠ 0) (h3 : p^2 * r > q^2 * r) : p^2 > q^2 := 
sorry

end NUMINAMATH_GPT_problem_statement_l1825_182565


namespace NUMINAMATH_GPT_count_congruent_to_3_mod_7_lt_500_l1825_182501

theorem count_congruent_to_3_mod_7_lt_500 : 
  ∃ n, n = 71 ∧ ∀ x, 0 < x ∧ x < 500 ∧ x % 7 = 3 ↔ ∃ k, 0 ≤ k ∧ k ≤ 70 ∧ x = 3 + 7 * k :=
sorry

end NUMINAMATH_GPT_count_congruent_to_3_mod_7_lt_500_l1825_182501


namespace NUMINAMATH_GPT_ice_cream_vendor_l1825_182549

theorem ice_cream_vendor (choco : ℕ) (mango : ℕ) (sold_choco : ℚ) (sold_mango : ℚ) 
  (h_choco : choco = 50) (h_mango : mango = 54) (h_sold_choco : sold_choco = 3/5) 
  (h_sold_mango : sold_mango = 2/3) : 
  choco - (choco * sold_choco) + mango - (mango * sold_mango) = 38 := 
by 
  sorry

end NUMINAMATH_GPT_ice_cream_vendor_l1825_182549


namespace NUMINAMATH_GPT_point_A_is_minus_five_l1825_182508

theorem point_A_is_minus_five 
  (A B C : ℝ)
  (h1 : A + 4 = B)
  (h2 : B - 2 = C)
  (h3 : C = -3) : 
  A = -5 := 
by 
  sorry

end NUMINAMATH_GPT_point_A_is_minus_five_l1825_182508


namespace NUMINAMATH_GPT_range_of_m_l1825_182538

theorem range_of_m (m : ℝ) (h : ∃ x : ℝ, x < 0 ∧ mx^2 + 2*x + 1 = 0) : m ∈ Set.Iic 1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1825_182538


namespace NUMINAMATH_GPT_decimal_to_fraction_l1825_182528

theorem decimal_to_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by sorry

end NUMINAMATH_GPT_decimal_to_fraction_l1825_182528


namespace NUMINAMATH_GPT_abs_value_equation_l1825_182525

-- Define the main proof problem
theorem abs_value_equation (a b c d : ℝ)
  (h : ∀ x : ℝ, |2 * x + 4| + |a * x + b| = |c * x + d|) :
  d = 2 * c :=
sorry -- Proof skipped for this exercise

end NUMINAMATH_GPT_abs_value_equation_l1825_182525


namespace NUMINAMATH_GPT_num_divisors_720_l1825_182518

-- Define the number 720 and its prime factorization
def n : ℕ := 720
def pf : List (ℕ × ℕ) := [(2, 4), (3, 2), (5, 1)]

-- Define the function to calculate the number of divisors from prime factorization
def num_divisors (pf : List (ℕ × ℕ)) : ℕ :=
  pf.foldr (λ p acc => acc * (p.snd + 1)) 1

-- Statement to prove
theorem num_divisors_720 : num_divisors pf = 30 :=
  by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_num_divisors_720_l1825_182518


namespace NUMINAMATH_GPT_identify_wrong_operator_l1825_182594

def original_expr (x y z w u v p q : Int) : Int := x + y - z + w - u + v - p + q
def wrong_expr (x y z w u v p q : Int) : Int := x + y - z - w - u + v - p + q

theorem identify_wrong_operator :
  original_expr 3 5 7 9 11 13 15 17 ≠ -4 →
  wrong_expr 3 5 7 9 11 13 15 17 = -4 :=
by
  sorry

end NUMINAMATH_GPT_identify_wrong_operator_l1825_182594


namespace NUMINAMATH_GPT_largest_digit_divisible_by_6_l1825_182563

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem largest_digit_divisible_by_6 (N : ℕ) (hN : N ≤ 9) :
  (∃ m : ℕ, 56780 + N = m * 6) ∧ is_even N ∧ is_divisible_by_3 (26 + N) → N = 4 := by
  sorry

end NUMINAMATH_GPT_largest_digit_divisible_by_6_l1825_182563


namespace NUMINAMATH_GPT_sum_of_squares_five_consecutive_ints_not_perfect_square_l1825_182572

theorem sum_of_squares_five_consecutive_ints_not_perfect_square (n : ℤ) :
  ∀ k : ℤ, k^2 ≠ 5 * (n^2 + 2) := 
sorry

end NUMINAMATH_GPT_sum_of_squares_five_consecutive_ints_not_perfect_square_l1825_182572


namespace NUMINAMATH_GPT_point_on_y_axis_l1825_182588

theorem point_on_y_axis (m n : ℝ) (h : (m, n).1 = 0) : m = 0 :=
by
  sorry

end NUMINAMATH_GPT_point_on_y_axis_l1825_182588


namespace NUMINAMATH_GPT_interior_diagonals_sum_l1825_182513

theorem interior_diagonals_sum (a b c : ℝ) 
  (h1 : 4 * (a + b + c) = 52)
  (h2 : 2 * (a * b + b * c + c * a) = 118) :
  4 * Real.sqrt (a^2 + b^2 + c^2) = 4 * Real.sqrt 51 := 
by
  sorry

end NUMINAMATH_GPT_interior_diagonals_sum_l1825_182513


namespace NUMINAMATH_GPT_fifth_equation_pattern_l1825_182507

theorem fifth_equation_pattern :
  (1 = 1) →
  (2 + 3 + 4 = 9) →
  (3 + 4 + 5 + 6 + 7 = 25) →
  (4 + 5 + 6 + 7 + 8 + 9 + 10 = 49) →
  (5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 = 81) :=
by 
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_fifth_equation_pattern_l1825_182507


namespace NUMINAMATH_GPT_donor_multiple_l1825_182555

def cost_per_box (food_cost : ℕ) (supplies_cost : ℕ) : ℕ := food_cost + supplies_cost

def total_initial_cost (num_boxes : ℕ) (cost_per_box : ℕ) : ℕ := num_boxes * cost_per_box

def additional_boxes (total_boxes : ℕ) (initial_boxes : ℕ) : ℕ := total_boxes - initial_boxes

def donor_contribution (additional_boxes : ℕ) (cost_per_box : ℕ) : ℕ := additional_boxes * cost_per_box

def multiple (donor_contribution : ℕ) (initial_cost : ℕ) : ℕ := donor_contribution / initial_cost

theorem donor_multiple 
    (initial_boxes : ℕ) (box_cost : ℕ) (total_boxes : ℕ) (donor_multi : ℕ)
    (h1 : initial_boxes = 400) 
    (h2 : box_cost = 245) 
    (h3 : total_boxes = 2000)
    : donor_multi = 4 :=
by
    let initial_cost := total_initial_cost initial_boxes box_cost
    let additional_boxes := additional_boxes total_boxes initial_boxes
    let contribution := donor_contribution additional_boxes box_cost
    have h4 : contribution = 392000 := sorry
    have h5 : initial_cost = 98000 := sorry
    have h6 : donor_multi = contribution / initial_cost := sorry
    -- Therefore, the multiple should be 4
    exact sorry

end NUMINAMATH_GPT_donor_multiple_l1825_182555


namespace NUMINAMATH_GPT_combined_weight_of_parcels_l1825_182599

variable (x y z : ℕ)

theorem combined_weight_of_parcels : 
  (x + y = 132) ∧ (y + z = 135) ∧ (z + x = 140) → x + y + z = 204 :=
by 
  intros
  sorry

end NUMINAMATH_GPT_combined_weight_of_parcels_l1825_182599


namespace NUMINAMATH_GPT_find_x_eq_3_l1825_182560

noncomputable def f (x : ℝ) : ℝ := 4 * x - 9
noncomputable def f_inv (x : ℝ) : ℝ := (x + 9) / 4

theorem find_x_eq_3 : ∃ x : ℝ, f x = f_inv x ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_eq_3_l1825_182560


namespace NUMINAMATH_GPT_rhombus_area_is_160_l1825_182593

-- Define the values of the diagonals
def d1 : ℝ := 16
def d2 : ℝ := 20

-- Define the formula for the area of the rhombus
noncomputable def area_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

-- State the theorem to be proved
theorem rhombus_area_is_160 :
  area_rhombus d1 d2 = 160 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_is_160_l1825_182593


namespace NUMINAMATH_GPT_fraction_of_shaded_area_is_11_by_12_l1825_182578

noncomputable def shaded_fraction_of_square : ℚ :=
  let s : ℚ := 1 -- Assume the side length of the square is 1 for simplicity.
  let P := (0, s / 2)
  let Q := (s / 3, s)
  let V := (0, s)
  let base := s / 2
  let height := s / 3
  let triangle_area := (1 / 2) * base * height
  let square_area := s * s
  let shaded_area := square_area - triangle_area
  shaded_area / square_area

theorem fraction_of_shaded_area_is_11_by_12 : shaded_fraction_of_square = 11 / 12 :=
  sorry

end NUMINAMATH_GPT_fraction_of_shaded_area_is_11_by_12_l1825_182578
