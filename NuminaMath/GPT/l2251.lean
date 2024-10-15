import Mathlib

namespace NUMINAMATH_GPT_inequalities_region_quadrants_l2251_225183

theorem inequalities_region_quadrants:
  (∀ x y : ℝ, y > -2 * x + 3 → y > x / 2 + 1 → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0)) :=
sorry

end NUMINAMATH_GPT_inequalities_region_quadrants_l2251_225183


namespace NUMINAMATH_GPT_annie_purchases_l2251_225158

theorem annie_purchases (x y z : ℕ) 
  (h1 : x + y + z = 50) 
  (h2 : 20 * x + 400 * y + 500 * z = 5000) :
  x = 40 :=
by sorry

end NUMINAMATH_GPT_annie_purchases_l2251_225158


namespace NUMINAMATH_GPT_range_of_m_l2251_225196

def proposition_p (m : ℝ) : Prop :=
  ∀ (x : ℝ), (1 ≤ x) → (x^2 - 2*m*x + 1/2 > 0)

def proposition_q (m : ℝ) : Prop :=
  ∃ (x : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ (x^2 - m*x - 2 = 0)

theorem range_of_m (m : ℝ) (h1 : ¬ proposition_q m) (h2 : proposition_p m ∨ proposition_q m) :
  -1 < m ∧ m < 3/4 :=
  sorry

end NUMINAMATH_GPT_range_of_m_l2251_225196


namespace NUMINAMATH_GPT_trig_expression_evaluation_l2251_225167

-- Define the given conditions
axiom sin_390 : Real.sin (390 * Real.pi / 180) = 1 / 2
axiom tan_neg_45 : Real.tan (-45 * Real.pi / 180) = -1
axiom cos_360 : Real.cos (360 * Real.pi / 180) = 1

-- Formulate the theorem
theorem trig_expression_evaluation : 
  2 * Real.sin (390 * Real.pi / 180) - Real.tan (-45 * Real.pi / 180) + 5 * Real.cos (360 * Real.pi / 180) = 7 :=
by
  rw [sin_390, tan_neg_45, cos_360]
  sorry

end NUMINAMATH_GPT_trig_expression_evaluation_l2251_225167


namespace NUMINAMATH_GPT_Taylor_needs_14_jars_l2251_225127

noncomputable def standard_jar_volume : ℕ := 60
noncomputable def big_container_volume : ℕ := 840

theorem Taylor_needs_14_jars : big_container_volume / standard_jar_volume = 14 :=
by sorry

end NUMINAMATH_GPT_Taylor_needs_14_jars_l2251_225127


namespace NUMINAMATH_GPT_expansion_term_count_l2251_225194

theorem expansion_term_count 
  (A : Finset ℕ) (B : Finset ℕ) 
  (hA : A.card = 3) (hB : B.card = 4) : 
  (Finset.card (A.product B)) = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_expansion_term_count_l2251_225194


namespace NUMINAMATH_GPT_find_x_l2251_225101

-- Definitions based on the given conditions
variables {B C D : Type} (A : Type)

-- Angles in degrees
variables (angle_ACD : ℝ := 100)
variables (angle_ADB : ℝ)
variables (angle_ABD : ℝ := 2 * angle_ADB)
variables (angle_DAC : ℝ)
variables (angle_BAC : ℝ := angle_DAC)
variables (angle_ACB : ℝ := 180 - angle_ACD)
variables (y : ℝ := angle_DAC)
variables (x : ℝ := angle_ADB)

-- The proof statement
theorem find_x (h1 : B = C) (h2 : C = D) 
    (h3: angle_ACD = 100) 
    (h4: angle_ADB = x) 
    (h5: angle_ABD = 2 * x) 
    (h6: angle_DAC = angle_BAC) 
    (h7: angle_DAC = y)
    : x = 20 :=
sorry

end NUMINAMATH_GPT_find_x_l2251_225101


namespace NUMINAMATH_GPT_find_j_l2251_225105

noncomputable def f (a b c x : ℤ) : ℤ := a * x^2 + b * x + c

theorem find_j
  (a b c : ℤ)
  (h1 : f a b c 2 = 0)
  (h2 : 200 < f a b c 10 ∧ f a b c 10 < 300)
  (h3 : 400 < f a b c 9 ∧ f a b c 9 < 500)
  (j : ℤ)
  (h4 : 1000 * j < f a b c 100 ∧ f a b c 100 < 1000 * (j + 1)) :
  j = 36 := sorry

end NUMINAMATH_GPT_find_j_l2251_225105


namespace NUMINAMATH_GPT_gcd_problem_l2251_225139

theorem gcd_problem (b : ℕ) (h : ∃ k : ℕ, b = 3150 * k) :
  gcd (b^2 + 9 * b + 54) (b + 4) = 2 := by
  sorry

end NUMINAMATH_GPT_gcd_problem_l2251_225139


namespace NUMINAMATH_GPT_vegetarian_eaters_l2251_225191

-- Define the conditions
theorem vegetarian_eaters : 
  ∀ (total family_size : ℕ) 
  (only_veg only_nonveg both_veg_nonveg eat_veg : ℕ), 
  family_size = 45 → 
  only_veg = 22 → 
  only_nonveg = 15 → 
  both_veg_nonveg = 8 → 
  eat_veg = only_veg + both_veg_nonveg → 
  eat_veg = 30 :=
by
  intros total family_size only_veg only_nonveg both_veg_nonveg eat_veg
  sorry

end NUMINAMATH_GPT_vegetarian_eaters_l2251_225191


namespace NUMINAMATH_GPT_smallest_part_division_l2251_225147

theorem smallest_part_division (y : ℝ) (h1 : y > 0) :
  ∃ (x : ℝ), x = y / 9 ∧ (∃ (a b c : ℝ), a = x ∧ b = 3 * x ∧ c = 5 * x ∧ a + b + c = y) :=
sorry

end NUMINAMATH_GPT_smallest_part_division_l2251_225147


namespace NUMINAMATH_GPT_coordinates_of_F_double_prime_l2251_225109

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem coordinates_of_F_double_prime :
  let F : ℝ × ℝ := (3, 3)
  let F' := reflect_over_y_axis F
  let F'' := reflect_over_x_axis F'
  F'' = (-3, -3) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_F_double_prime_l2251_225109


namespace NUMINAMATH_GPT_beta_max_two_day_ratio_l2251_225161

noncomputable def alpha_first_day_score : ℚ := 160 / 300
noncomputable def alpha_second_day_score : ℚ := 140 / 200
noncomputable def alpha_two_day_ratio : ℚ := 300 / 500

theorem beta_max_two_day_ratio :
  ∃ (p q r : ℕ), 
  p < 300 ∧
  q < (8 * p / 15) ∧
  r < ((3500 - 7 * p) / 10) ∧
  q + r = 299 ∧
  gcd 299 500 = 1 ∧
  (299 + 500) = 799 := 
sorry

end NUMINAMATH_GPT_beta_max_two_day_ratio_l2251_225161


namespace NUMINAMATH_GPT_min_k_value_l2251_225128

noncomputable def f (k x : ℝ) : ℝ := k * (x^2 - x + 1) - x^4 * (1 - x)^4

theorem min_k_value : ∃ k : ℝ, (k = 1 / 192) ∧ ∀ x : ℝ, (0 ≤ x) → (x ≤ 1) → (f k x ≥ 0) :=
by
  existsi (1 / 192)
  sorry

end NUMINAMATH_GPT_min_k_value_l2251_225128


namespace NUMINAMATH_GPT_mario_savings_percentage_l2251_225138

-- Define the price of one ticket
def ticket_price : ℝ := sorry

-- Define the conditions
-- Condition 1: 5 tickets can be purchased for the usual price of 3 tickets
def price_for_5_tickets := 3 * ticket_price

-- Condition 2: Mario bought 5 tickets
def mario_tickets := 5 * ticket_price

-- Condition 3: Usual price for 5 tickets
def usual_price_5_tickets := 5 * ticket_price

-- Calculate the amount saved
def amount_saved := usual_price_5_tickets - price_for_5_tickets

theorem mario_savings_percentage
  (ticket_price: ℝ)
  (h1 : price_for_5_tickets = 3 * ticket_price)
  (h2 : mario_tickets = 5 * ticket_price)
  (h3 : usual_price_5_tickets = 5 * ticket_price)
  (h4 : amount_saved = usual_price_5_tickets - price_for_5_tickets):
  (amount_saved / usual_price_5_tickets) * 100 = 40 := 
by {
    -- Placeholder
    sorry
}

end NUMINAMATH_GPT_mario_savings_percentage_l2251_225138


namespace NUMINAMATH_GPT_floor_area_not_greater_than_10_l2251_225171

theorem floor_area_not_greater_than_10 (L W H : ℝ) (h_height : H = 3)
  (h_more_paint_wall1 : L * 3 > L * W)
  (h_more_paint_wall2 : W * 3 > L * W) :
  L * W ≤ 9 :=
by
  sorry

end NUMINAMATH_GPT_floor_area_not_greater_than_10_l2251_225171


namespace NUMINAMATH_GPT_part_I_part_II_l2251_225182

variable (f : ℝ → ℝ)

-- Condition 1: f is an even function
axiom even_function : ∀ x : ℝ, f (-x) = f x

-- Condition 2: f is symmetric about x = 1
axiom symmetric_about_1 : ∀ x : ℝ, f x = f (2 - x)

-- Condition 3: f(x₁ + x₂) = f(x₁) * f(x₂) for x₁, x₂ ∈ [0, 1/2]
axiom multiplicative_on_interval : ∀ x₁ x₂ : ℝ, (0 ≤ x₁ ∧ x₁ ≤ 1/2) ∧ (0 ≤ x₂ ∧ x₂ ≤ 1/2) → f (x₁ + x₂) = f x₁ * f x₂

-- Given f(1) = 2
axiom f_one : f 1 = 2

-- Part I: Prove f(1/2) = √2 and f(1/4) = 2^(1/4).
theorem part_I : f (1 / 2) = Real.sqrt 2 ∧ f (1 / 4) = Real.sqrt (Real.sqrt 2) := by
  sorry

-- Part II: Prove that f(x) is a periodic function with period 2.
theorem part_II : ∀ x : ℝ, f x = f (x + 2) := by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l2251_225182


namespace NUMINAMATH_GPT_price_of_bracelets_max_type_a_bracelets_l2251_225187

-- Part 1: Proving the prices of the bracelets
theorem price_of_bracelets :
  ∃ (x y : ℝ), (3 * x + y = 128 ∧ x + 2 * y = 76) ∧ (x = 36 ∧ y = 20) :=
sorry

-- Part 2: Proving the maximum number of type A bracelets they can buy within the budget
theorem max_type_a_bracelets :
  ∃ (m : ℕ), 36 * m + 20 * (100 - m) ≤ 2500 ∧ m = 31 :=
sorry

end NUMINAMATH_GPT_price_of_bracelets_max_type_a_bracelets_l2251_225187


namespace NUMINAMATH_GPT_racers_final_segment_l2251_225166

def final_racer_count : Nat := 9

def segment_eliminations (init_count: Nat) : Nat :=
  let seg1 := init_count - Int.toNat (Nat.sqrt init_count)
  let seg2 := seg1 - seg1 / 3
  let seg3 := seg2 - (seg2 / 4 + (2 ^ 2))
  let seg4 := seg3 - seg3 / 3
  let seg5 := seg4 / 2
  let seg6 := seg5 - (seg5 * 3 / 4)
  seg6

theorem racers_final_segment
  (init_count: Nat)
  (h: init_count = 225) :
  segment_eliminations init_count = final_racer_count :=
  by
  rw [h]
  unfold segment_eliminations
  sorry

end NUMINAMATH_GPT_racers_final_segment_l2251_225166


namespace NUMINAMATH_GPT_coordinates_of_A_l2251_225115

-- Definitions based on conditions
def origin : ℝ × ℝ := (0, 0)
def similarity_ratio : ℝ := 2
def point_A : ℝ × ℝ := (2, 3)
def point_A' (P : ℝ × ℝ) : Prop :=
  P = (similarity_ratio * point_A.1, similarity_ratio * point_A.2) ∨
  P = (-similarity_ratio * point_A.1, -similarity_ratio * point_A.2)

-- Statement of the theorem
theorem coordinates_of_A' :
  ∃ P : ℝ × ℝ, point_A' P :=
by
  use (4, 6)
  left
  sorry

end NUMINAMATH_GPT_coordinates_of_A_l2251_225115


namespace NUMINAMATH_GPT_problem_solution_l2251_225124

variable {A B : Set ℝ}

def definition_A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def definition_B : Set ℝ := {x | x ≤ 3}

theorem problem_solution : definition_A ∩ definition_B = definition_A :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l2251_225124


namespace NUMINAMATH_GPT_mean_proportion_of_3_and_4_l2251_225199

theorem mean_proportion_of_3_and_4 : ∃ x : ℝ, 3 / x = x / 4 ∧ (x = 2 * Real.sqrt 3 ∨ x = - (2 * Real.sqrt 3)) :=
by
  sorry

end NUMINAMATH_GPT_mean_proportion_of_3_and_4_l2251_225199


namespace NUMINAMATH_GPT_certain_event_l2251_225153

-- Definitions of the events as propositions
def EventA : Prop := ∃ n : ℕ, n ≥ 1 ∧ (n % 2 = 0)
def EventB : Prop := ∃ t : ℝ, t ≥ 0  -- Simplifying as the event of an advertisement airing
def EventC : Prop := ∃ w : ℕ, w ≥ 1  -- Simplifying as the event of rain in Weinan on a specific future date
def EventD : Prop := true  -- The sun rises from the east in the morning is always true

-- The statement that Event D is the only certain event among the given options
theorem certain_event : EventD ∧ ¬EventA ∧ ¬EventB ∧ ¬EventC :=
by
  sorry

end NUMINAMATH_GPT_certain_event_l2251_225153


namespace NUMINAMATH_GPT_sandbag_weight_proof_l2251_225140

-- Define all given conditions
def bag_capacity : ℝ := 250
def fill_percentage : ℝ := 0.80
def material_weight_multiplier : ℝ := 1.40 -- since 40% heavier means 1 + 0.40
def empty_bag_weight : ℝ := 0

-- Using these definitions, form the goal to prove
theorem sandbag_weight_proof : 
  (fill_percentage * bag_capacity * material_weight_multiplier) + empty_bag_weight = 280 :=
by
  sorry

end NUMINAMATH_GPT_sandbag_weight_proof_l2251_225140


namespace NUMINAMATH_GPT_car_average_speed_l2251_225120

noncomputable def average_speed (speeds : List ℝ) (distances : List ℝ) (times : List ℝ) : ℝ :=
  (distances.sum + times.sum) / times.sum

theorem car_average_speed :
  let distances := [30, 35, 35, 52 / 3, 15]
  let times := [30 / 45, 35 / 55, 30 / 60, 20 / 60, 15 / 65]
  average_speed [45, 55, 70, 52, 65] distances times = 64.82 := by
  sorry

end NUMINAMATH_GPT_car_average_speed_l2251_225120


namespace NUMINAMATH_GPT_contrapositive_example_l2251_225129

theorem contrapositive_example (x : ℝ) : 
  (x^2 - 3*x + 2 = 0 → x = 1) ↔ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) := 
by
  sorry

end NUMINAMATH_GPT_contrapositive_example_l2251_225129


namespace NUMINAMATH_GPT_arithmetic_sequence_max_sum_l2251_225137

theorem arithmetic_sequence_max_sum (a d t : ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (h1 : a > 0) 
  (h2 : (9 * t) = a + 5 * d) 
  (h3 : (11 * t) = a + 4 * d) 
  (h4 : ∀ n, S n = (n * (2 * a + (n - 1) * d)) / 2) :
  n = 10 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_max_sum_l2251_225137


namespace NUMINAMATH_GPT_max_marks_l2251_225108

theorem max_marks (M : ℝ) (h1 : 0.42 * M = 80) : M = 190 :=
by
  sorry

end NUMINAMATH_GPT_max_marks_l2251_225108


namespace NUMINAMATH_GPT_quadratic_roots_identity_l2251_225174

theorem quadratic_roots_identity
  (a b c : ℝ)
  (x1 x2 : ℝ)
  (hx1 : x1 = Real.sin (42 * Real.pi / 180))
  (hx2 : x2 = Real.sin (48 * Real.pi / 180))
  (hx2_trig_identity : x2 = Real.cos (42 * Real.pi / 180))
  (hroots : ∀ x, a * x^2 + b * x + c = 0 ↔ (x = x1 ∨ x = x2)) :
  b^2 = a^2 + 2 * a * c :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_identity_l2251_225174


namespace NUMINAMATH_GPT_max_value_of_function_f_l2251_225156

noncomputable def f (t : ℝ) : ℝ := (4^t - 2 * t) * t / 16^t

theorem max_value_of_function_f : ∃ t : ℝ, ∀ x : ℝ, f x ≤ f t ∧ f t = 1 / 8 := sorry

end NUMINAMATH_GPT_max_value_of_function_f_l2251_225156


namespace NUMINAMATH_GPT_fraction_equality_l2251_225165

theorem fraction_equality {x y : ℝ} (h : x + y ≠ 0) (h1 : x - y ≠ 0) : 
  (-x + y) / (-x - y) = (x - y) / (x + y) := 
sorry

end NUMINAMATH_GPT_fraction_equality_l2251_225165


namespace NUMINAMATH_GPT_factor_expression_l2251_225149

theorem factor_expression (x y : ℝ) :
  75 * x^10 * y^3 - 150 * x^20 * y^6 = 75 * x^10 * y^3 * (1 - 2 * x^10 * y^3) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l2251_225149


namespace NUMINAMATH_GPT_sufficient_condition_l2251_225134

theorem sufficient_condition (m : ℝ) (x : ℝ) : -3 < m ∧ m < 1 → ((m - 1) * x^2 + (m - 1) * x - 1 < 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_l2251_225134


namespace NUMINAMATH_GPT_total_is_correct_l2251_225136

-- Define the given conditions.
def dividend : ℕ := 55
def divisor : ℕ := 11
def quotient := dividend / divisor
def total := dividend + quotient + divisor

-- State the theorem to be proven.
theorem total_is_correct : total = 71 := by sorry

end NUMINAMATH_GPT_total_is_correct_l2251_225136


namespace NUMINAMATH_GPT_geometric_seq_a4_l2251_225155

variable {a : ℕ → ℝ}

-- Definition: a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Condition
axiom h : a 2 * a 6 = 4

-- Theorem that needs to be proved
theorem geometric_seq_a4 (h_seq: is_geometric_sequence a) (h: a 2 * a 6 = 4) : a 4 = 2 ∨ a 4 = -2 := by
  sorry

end NUMINAMATH_GPT_geometric_seq_a4_l2251_225155


namespace NUMINAMATH_GPT_parking_lot_wheels_l2251_225123

-- Define the total number of wheels for each type of vehicle
def car_wheels (n : ℕ) : ℕ := n * 4
def motorcycle_wheels (n : ℕ) : ℕ := n * 2
def truck_wheels (n : ℕ) : ℕ := n * 6
def van_wheels (n : ℕ) : ℕ := n * 4

-- Number of each type of guests' vehicles
def num_cars : ℕ := 5
def num_motorcycles : ℕ := 4
def num_trucks : ℕ := 3
def num_vans : ℕ := 2

-- Number of parents' vehicles and their wheels
def parents_car_wheels : ℕ := 4
def parents_jeep_wheels : ℕ := 4

-- Summing up all the wheels
def total_wheels : ℕ :=
  car_wheels num_cars +
  motorcycle_wheels num_motorcycles +
  truck_wheels num_trucks +
  van_wheels num_vans +
  parents_car_wheels +
  parents_jeep_wheels

theorem parking_lot_wheels : total_wheels = 62 := by
  sorry

end NUMINAMATH_GPT_parking_lot_wheels_l2251_225123


namespace NUMINAMATH_GPT_ribbon_original_length_l2251_225177

theorem ribbon_original_length (x : ℕ) (h1 : 11 * 35 = 7 * x) : x = 55 :=
by
  sorry

end NUMINAMATH_GPT_ribbon_original_length_l2251_225177


namespace NUMINAMATH_GPT_negation_of_universal_prop_l2251_225172

-- Define the conditions
variable (f : ℝ → ℝ)

-- Theorem statement
theorem negation_of_universal_prop : 
  (¬ ∀ x : ℝ, f x > 0) ↔ (∃ x : ℝ, f x ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_prop_l2251_225172


namespace NUMINAMATH_GPT_largest_lcm_value_is_90_l2251_225190

def lcm_vals (a b : ℕ) : ℕ := Nat.lcm a b

theorem largest_lcm_value_is_90 :
  max (lcm_vals 18 3)
      (max (lcm_vals 18 9)
           (max (lcm_vals 18 6)
                (max (lcm_vals 18 12)
                     (max (lcm_vals 18 15)
                          (lcm_vals 18 18))))) = 90 :=
by
  -- Use the fact that the calculations of LCMs are as follows:
  -- lcm(18, 3) = 18
  -- lcm(18, 9) = 18
  -- lcm(18, 6) = 18
  -- lcm(18, 12) = 36
  -- lcm(18, 15) = 90
  -- lcm(18, 18) = 18
  -- therefore, the largest value among these is 90
  sorry

end NUMINAMATH_GPT_largest_lcm_value_is_90_l2251_225190


namespace NUMINAMATH_GPT_no_such_function_exists_l2251_225159

theorem no_such_function_exists : ¬ ∃ f : ℕ → ℕ, ∀ n > 2, f (f (n - 1)) = f (n + 1) - f n :=
by {
  sorry
}

end NUMINAMATH_GPT_no_such_function_exists_l2251_225159


namespace NUMINAMATH_GPT_simplify_expression_l2251_225154

variables {K : Type*} [Field K]

theorem simplify_expression (a b c : K) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b) : 
    (a^3 - b^3) / (a * b) - (a * b - c * b) / (a * b - a^2) = (a^2 + a * b + b^2 + a * c) / (a * b) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2251_225154


namespace NUMINAMATH_GPT_range_of_b_l2251_225193

theorem range_of_b (b : ℤ) : 
  (∃ x1 x2 : ℤ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ x1 - b > 0 ∧ x2 - b > 0 ∧ (∀ x : ℤ, x < 0 ∧ x - b > 0 → (x = x1 ∨ x = x2))) ↔ (-3 ≤ b ∧ b < -2) :=
by sorry

end NUMINAMATH_GPT_range_of_b_l2251_225193


namespace NUMINAMATH_GPT_y_equals_x_l2251_225113

theorem y_equals_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x = 1 + 1 / y) (h2 : y = 1 + 1 / x) : y = x :=
sorry

end NUMINAMATH_GPT_y_equals_x_l2251_225113


namespace NUMINAMATH_GPT_simplify_polynomial_l2251_225114

variable {x : ℝ} -- Assume x is a real number

theorem simplify_polynomial :
  (3 * x^2 + 8 * x - 5) - (2 * x^2 + 6 * x - 15) = x^2 + 2 * x + 10 :=
sorry

end NUMINAMATH_GPT_simplify_polynomial_l2251_225114


namespace NUMINAMATH_GPT_max_sum_n_of_arithmetic_sequence_l2251_225197

/-- Let \( S_n \) be the sum of the first \( n \) terms of an arithmetic sequence \( \{a_n\} \) with 
a non-zero common difference, and \( a_1 > 0 \). If \( S_5 = S_9 \), then when \( S_n \) is maximum, \( n = 7 \). -/
theorem max_sum_n_of_arithmetic_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) (d : ℕ) 
  (a1_pos : a 1 > 0) (common_difference : ∀ n, a (n + 1) = a n + d)
  (s5_eq_s9 : S 5 = S 9) :
  ∃ n, (∀ m, m ≤ n → S m ≤ S n) ∧ n = 7 :=
sorry

end NUMINAMATH_GPT_max_sum_n_of_arithmetic_sequence_l2251_225197


namespace NUMINAMATH_GPT_areas_of_geometric_figures_with_equal_perimeter_l2251_225117

theorem areas_of_geometric_figures_with_equal_perimeter (l : ℝ) (h : (l > 0)) :
  let s1 := l^2 / (4 * Real.pi)
  let s2 := l^2 / 16
  let s3 := (Real.sqrt 3) * l^2 / 36
  s1 > s2 ∧ s2 > s3 := by
  sorry

end NUMINAMATH_GPT_areas_of_geometric_figures_with_equal_perimeter_l2251_225117


namespace NUMINAMATH_GPT_difference_of_fractions_l2251_225192

theorem difference_of_fractions (x y : ℝ) (h1 : x = 497) (h2 : y = 325) :
  (2/5) * (3 * x + 7 * y) - (3/5) * (x * y) = -95408.6 := by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_difference_of_fractions_l2251_225192


namespace NUMINAMATH_GPT_intersection_A_B_l2251_225116

def A (x : ℝ) : Prop := ∃ y, y = Real.log (-x^2 - 2*x + 8) ∧ -x^2 - 2*x + 8 > 0
def B (x : ℝ) : Prop := Real.log x / Real.log 2 < 1 ∧ x > 0

theorem intersection_A_B : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2251_225116


namespace NUMINAMATH_GPT_maximum_shapes_in_grid_l2251_225100

-- Define the grid size and shape properties
def grid_width : Nat := 8
def grid_height : Nat := 14
def shape_area : Nat := 3
def shape_grid_points : Nat := 8

-- Define the total grid points in the rectangular grid
def total_grid_points : Nat := (grid_width + 1) * (grid_height + 1)

-- Define the question and the condition that needs to be proved
theorem maximum_shapes_in_grid : (total_grid_points / shape_grid_points) = 16 := by
  sorry

end NUMINAMATH_GPT_maximum_shapes_in_grid_l2251_225100


namespace NUMINAMATH_GPT_triangle_area_example_l2251_225145

def point := (ℝ × ℝ)

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_example :
  let A : point := (3, -2)
  let B : point := (12, 5)
  let C : point := (3, 8)
  triangle_area A B C = 45 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_example_l2251_225145


namespace NUMINAMATH_GPT_simplify_expression_l2251_225146

theorem simplify_expression (a : ℤ) :
  ((36 * a^9)^4 * (63 * a^9)^4) = a^4 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l2251_225146


namespace NUMINAMATH_GPT_inequality_correct_l2251_225130

theorem inequality_correct (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - c > b - c := by
  sorry

end NUMINAMATH_GPT_inequality_correct_l2251_225130


namespace NUMINAMATH_GPT_factorize_l2251_225168

theorem factorize (a b : ℝ) : 2 * a ^ 2 - 8 * b ^ 2 = 2 * (a + 2 * b) * (a - 2 * b) :=
by 
  sorry

end NUMINAMATH_GPT_factorize_l2251_225168


namespace NUMINAMATH_GPT_paths_A_to_D_through_B_and_C_l2251_225133

-- Define points and paths in a grid
structure Point where
  x : ℕ
  y : ℕ

def A : Point := ⟨0, 0⟩
def B : Point := ⟨2, 3⟩
def C : Point := ⟨6, 4⟩
def D : Point := ⟨9, 6⟩

-- Calculate binomial coefficient
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.div (Nat.factorial n) ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Number of paths from one point to another in a grid
def numPaths (p1 p2 : Point) : ℕ :=
  let stepsRight := p2.x - p1.x
  let stepsDown := p2.y - p1.y
  choose (stepsRight + stepsDown) stepsRight

theorem paths_A_to_D_through_B_and_C : numPaths A B * numPaths B C * numPaths C D = 500 := by
  -- Using the conditions provided:
  -- numPaths A B = choose 5 2 = 10
  -- numPaths B C = choose 5 1 = 5
  -- numPaths C D = choose 5 2 = 10
  -- Therefore, numPaths A B * numPaths B C * numPaths C D = 10 * 5 * 10 = 500
  sorry

end NUMINAMATH_GPT_paths_A_to_D_through_B_and_C_l2251_225133


namespace NUMINAMATH_GPT_tiling_possible_if_and_only_if_one_dimension_is_integer_l2251_225184

-- Define our conditions: a, b are dimensions of the board and t is the positive dimension of the small rectangles
variable (a b : ℝ) (t : ℝ)

-- Define corresponding properties for these variables
axiom pos_t : t > 0

-- Theorem stating the condition for tiling
theorem tiling_possible_if_and_only_if_one_dimension_is_integer (a_non_int : ¬ ∃ z : ℤ, a = z) (b_non_int : ¬ ∃ z : ℤ, b = z) :
  ∃ n m : ℕ, n * 1 + m * t = a * b :=
sorry

end NUMINAMATH_GPT_tiling_possible_if_and_only_if_one_dimension_is_integer_l2251_225184


namespace NUMINAMATH_GPT_simplify_and_find_ratio_l2251_225106

theorem simplify_and_find_ratio (m : ℤ) : 
  let expr := (6 * m + 18) / 6 
  let c := 1
  let d := 3
  (c / d : ℚ) = 1 / 3 := 
by
  -- Conditions and transformations are stated here
  -- (6 * m + 18) / 6 can be simplified step-by-step
  sorry

end NUMINAMATH_GPT_simplify_and_find_ratio_l2251_225106


namespace NUMINAMATH_GPT_sabrina_total_leaves_l2251_225173

theorem sabrina_total_leaves (num_basil num_sage num_verbena : ℕ)
  (h1 : num_basil = 2 * num_sage)
  (h2 : num_sage = num_verbena - 5)
  (h3 : num_basil = 12) :
  num_sage + num_verbena + num_basil = 29 :=
by
  sorry

end NUMINAMATH_GPT_sabrina_total_leaves_l2251_225173


namespace NUMINAMATH_GPT_fraction_meaningful_l2251_225157

-- Define the condition for the fraction being meaningful
def denominator_not_zero (x : ℝ) : Prop := x + 1 ≠ 0

-- Define the statement to be proved
theorem fraction_meaningful (x : ℝ) : denominator_not_zero x ↔ x ≠ -1 :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_l2251_225157


namespace NUMINAMATH_GPT_larger_angle_measure_l2251_225141

-- Defining all conditions
def is_complementary (a b : ℝ) : Prop := a + b = 90

def angle_ratio (a b : ℝ) : Prop := a / b = 5 / 4

-- Main proof statement
theorem larger_angle_measure (a b : ℝ) (h1 : is_complementary a b) (h2 : angle_ratio a b) : a = 50 :=
by
  sorry

end NUMINAMATH_GPT_larger_angle_measure_l2251_225141


namespace NUMINAMATH_GPT_sphere_volume_l2251_225175

/-- A sphere is perfectly inscribed in a cube. 
If the edge of the cube measures 10 inches, the volume of the sphere in cubic inches is \(\frac{500}{3}\pi\). -/
theorem sphere_volume (a : ℝ) (h : a = 10) : 
  ∃ V : ℝ, V = (4 / 3) * Real.pi * (a / 2)^3 ∧ V = (500 / 3) * Real.pi :=
by
  use (4 / 3) * Real.pi * (a / 2)^3
  sorry

end NUMINAMATH_GPT_sphere_volume_l2251_225175


namespace NUMINAMATH_GPT_positive_integer_solutions_l2251_225102

theorem positive_integer_solutions :
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x + y + x * y = 2008 ∧
  ((x = 6 ∧ y = 286) ∨ (x = 286 ∧ y = 6) ∨ (x = 40 ∧ y = 48) ∨ (x = 48 ∧ y = 40)) :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_solutions_l2251_225102


namespace NUMINAMATH_GPT_exists_five_distinct_nat_numbers_l2251_225170

theorem exists_five_distinct_nat_numbers 
  (a b c d e : ℕ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h_no_div_3 : ¬(3 ∣ a) ∧ ¬(3 ∣ b) ∧ ¬(3 ∣ c) ∧ ¬(3 ∣ d) ∧ ¬(3 ∣ e))
  (h_no_div_4 : ¬(4 ∣ a) ∧ ¬(4 ∣ b) ∧ ¬(4 ∣ c) ∧ ¬(4 ∣ d) ∧ ¬(4 ∣ e))
  (h_no_div_5 : ¬(5 ∣ a) ∧ ¬(5 ∣ b) ∧ ¬(5 ∣ c) ∧ ¬(5 ∣ d) ∧ ¬(5 ∣ e)) :
  (∃ (a b c d e : ℕ),
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
    (¬(3 ∣ a) ∧ ¬(3 ∣ b) ∧ ¬(3 ∣ c) ∧ ¬(3 ∣ d) ∧ ¬(3 ∣ e)) ∧
    (¬(4 ∣ a) ∧ ¬(4 ∣ b) ∧ ¬(4 ∣ c) ∧ ¬(4 ∣ d) ∧ ¬(4 ∣ e)) ∧
    (¬(5 ∣ a) ∧ ¬(5 ∣ b) ∧ ¬(5 ∣ c) ∧ ¬(5 ∣ d) ∧ ¬(5 ∣ e)) ∧
    (∀ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z → x + y + z = a + b + c + d + e → (x + y + z) % 3 = 0) ∧
    (∀ w x y z : ℕ, w ≠ x ∧ x ≠ y ∧ y ≠ z ∧ w ≠ y ∧ w ≠ z ∧ x ≠ z → w + x + y + z = a + b + c + d + e → (w + x + y + z) % 4 = 0) ∧
    (a + b + c + d + e) % 5 = 0) :=
  sorry

end NUMINAMATH_GPT_exists_five_distinct_nat_numbers_l2251_225170


namespace NUMINAMATH_GPT_yellow_papers_count_l2251_225111

theorem yellow_papers_count (n : ℕ) (total_papers : ℕ) (periphery_papers : ℕ) (inner_papers : ℕ) 
  (h1 : n = 10) 
  (h2 : total_papers = n * n) 
  (h3 : periphery_papers = 4 * n - 4)
  (h4 : inner_papers = total_papers - periphery_papers) :
  inner_papers = 64 :=
by
  sorry

end NUMINAMATH_GPT_yellow_papers_count_l2251_225111


namespace NUMINAMATH_GPT_parallelogram_diagonal_square_l2251_225162

theorem parallelogram_diagonal_square (A B C D P Q R S : Type)
    (area_ABCD : ℝ) (proj_A_P_BD proj_C_Q_BD proj_B_R_AC proj_D_S_AC : Prop)
    (PQ RS : ℝ) (d_squared : ℝ) 
    (h_area : area_ABCD = 24)
    (h_proj_A_P : proj_A_P_BD) (h_proj_C_Q : proj_C_Q_BD)
    (h_proj_B_R : proj_B_R_AC) (h_proj_D_S : proj_D_S_AC)
    (h_PQ_length : PQ = 8) (h_RS_length : RS = 10)
    : d_squared = 62 + 20*Real.sqrt 61 := sorry

end NUMINAMATH_GPT_parallelogram_diagonal_square_l2251_225162


namespace NUMINAMATH_GPT_polygon_sides_arithmetic_progression_l2251_225152

theorem polygon_sides_arithmetic_progression
  (n : ℕ)
  (h1 : ∀ i, 1 ≤ i → i ≤ n → 172 - (i - 1) * 8 > 0) -- Each angle in the sequence is positive
  (h2 : (∀ i, 1 ≤ i → i ≤ n → (172 - (i - 1) * 8) < 180)) -- Each angle < 180 degrees
  (h3 : n * (172 - (n-1) * 4) = 180 * (n - 2)) -- Sum of interior angles formula
  : n = 10 :=
sorry

end NUMINAMATH_GPT_polygon_sides_arithmetic_progression_l2251_225152


namespace NUMINAMATH_GPT_part1_part2_l2251_225160

-- Definitions for the sets A and B
def A := {x : ℝ | x^2 - 2 * x - 8 = 0}
def B (a : ℝ) := {x : ℝ | x^2 + a * x + a^2 - 12 = 0}

-- Proof statements
theorem part1 (a : ℝ) : (A ∩ B a = A) → a = -2 :=
by
  sorry

theorem part2 (a : ℝ) : (A ∪ B a = A) → (a ≥ 4 ∨ a < -4 ∨ a = -2) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2251_225160


namespace NUMINAMATH_GPT_proof_firstExpr_proof_secondExpr_l2251_225198

noncomputable def firstExpr : ℝ :=
  Real.logb 2 (Real.sqrt (7 / 48)) + Real.logb 2 12 - (1 / 2) * Real.logb 2 42 - 1

theorem proof_firstExpr :
  firstExpr = -3 / 2 :=
by
  sorry

noncomputable def secondExpr : ℝ :=
  (Real.logb 10 2) ^ 2 + Real.logb 10 (2 * Real.logb 10 50 + Real.logb 10 25)

theorem proof_secondExpr :
  secondExpr = 0.0906 + Real.logb 10 5.004 :=
by
  sorry

end NUMINAMATH_GPT_proof_firstExpr_proof_secondExpr_l2251_225198


namespace NUMINAMATH_GPT_proof_problem_l2251_225186

-- Definitions based on the given conditions
def cond1 : Prop := 1 * 9 + 2 = 11
def cond2 : Prop := 12 * 9 + 3 = 111
def cond3 : Prop := 123 * 9 + 4 = 1111
def cond4 : Prop := 1234 * 9 + 5 = 11111
def cond5 : Prop := 12345 * 9 + 6 = 111111

-- Main statement to prove
theorem proof_problem (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) (h5 : cond5) : 
  123456 * 9 + 7 = 1111111 :=
sorry

end NUMINAMATH_GPT_proof_problem_l2251_225186


namespace NUMINAMATH_GPT_value_of_business_calculation_l2251_225121

noncomputable def value_of_business (total_shares_sold_value : ℝ) (shares_fraction_sold : ℝ) (ownership_fraction : ℝ) : ℝ :=
  (total_shares_sold_value / shares_fraction_sold) * ownership_fraction⁻¹

theorem value_of_business_calculation :
  value_of_business 45000 (3/4) (2/3) = 90000 :=
by
  sorry

end NUMINAMATH_GPT_value_of_business_calculation_l2251_225121


namespace NUMINAMATH_GPT_inequality_solution_l2251_225103

theorem inequality_solution :
  { x : ℝ // x < 2 ∨ (3 < x ∧ x < 6) ∨ (7 < x ∧ x < 8) } →
  ((x - 3) * (x - 5) * (x - 7)) / ((x - 2) * (x - 6) * (x - 8)) > 0 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2251_225103


namespace NUMINAMATH_GPT_eq1_solution_eq2_solution_l2251_225122

theorem eq1_solution (x : ℝ) : (x - 1)^2 - 1 = 15 ↔ x = 5 ∨ x = -3 := by sorry

theorem eq2_solution (x : ℝ) : (1 / 3) * (x + 3)^3 - 9 = 0 ↔ x = 0 := by sorry

end NUMINAMATH_GPT_eq1_solution_eq2_solution_l2251_225122


namespace NUMINAMATH_GPT_find_angle_A_find_sum_b_c_l2251_225119

-- Given the necessary conditions
variables (a b c : ℝ)
variables (A B C : ℝ)
variables (sin cos : ℝ → ℝ)

-- Assuming necessary trigonometric identities
axiom sin_squared_add_cos_squared : ∀ (x : ℝ), sin x * sin x + cos x * cos x = 1
axiom cos_sum : ∀ (x y : ℝ), cos (x + y) = cos x * cos y - sin x * sin y

-- Condition: 2 sin^2(A) + 3 cos(B+C) = 0
axiom condition1 : 2 * sin A * sin A + 3 * cos (B + C) = 0

-- Condition: The area of the triangle is S = 5 √3
axiom condition2 : 1 / 2 * b * c * sin A = 5 * Real.sqrt 3

-- Condition: The length of side a = √21
axiom condition3 : a = Real.sqrt 21

-- Part (1): Prove the measure of angle A
theorem find_angle_A : A = π / 3 :=
sorry

-- Part (2): Given S = 5√3 and a = √21, find b + c.
theorem find_sum_b_c : b + c = 9 :=
sorry

end NUMINAMATH_GPT_find_angle_A_find_sum_b_c_l2251_225119


namespace NUMINAMATH_GPT_fiona_observe_pairs_l2251_225110

def classroom_pairs (n : ℕ) : ℕ :=
  if n > 1 then n - 1 else 0

theorem fiona_observe_pairs :
  classroom_pairs 12 = 11 :=
by
  sorry

end NUMINAMATH_GPT_fiona_observe_pairs_l2251_225110


namespace NUMINAMATH_GPT_units_digit_of_7_pow_y_plus_6_is_9_l2251_225176

theorem units_digit_of_7_pow_y_plus_6_is_9 (y : ℕ) (hy : 0 < y) : 
  (7^y + 6) % 10 = 9 ↔ ∃ k : ℕ, y = 4 * k + 3 := by
  sorry

end NUMINAMATH_GPT_units_digit_of_7_pow_y_plus_6_is_9_l2251_225176


namespace NUMINAMATH_GPT_a_range_l2251_225125

noncomputable def f (a x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else a ^ x - a

theorem a_range (a : ℝ) : (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ a ∈ Set.Ico (1/7 : ℝ) (1/3 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_a_range_l2251_225125


namespace NUMINAMATH_GPT_product_of_second_and_fourth_term_l2251_225178

theorem product_of_second_and_fourth_term (a : ℕ → ℤ) (d : ℤ) (h₁ : a 10 = 25) (h₂ : d = 3)
  (h₃ : ∀ n, a n = a 1 + (n - 1) * d) : a 2 * a 4 = 7 :=
by
  -- Assuming necessary conditions are defined
  sorry

end NUMINAMATH_GPT_product_of_second_and_fourth_term_l2251_225178


namespace NUMINAMATH_GPT_Yoque_monthly_payment_l2251_225189

theorem Yoque_monthly_payment :
  ∃ m : ℝ, m = 15 ∧ ∀ a t : ℝ, a = 150 ∧ t = 11 ∧ (a + 0.10 * a) / t = m :=
by
  sorry

end NUMINAMATH_GPT_Yoque_monthly_payment_l2251_225189


namespace NUMINAMATH_GPT_max_cards_possible_l2251_225164

-- Define the dimensions for the cardboard and the card.
def cardboard_length : ℕ := 48
def cardboard_width : ℕ := 36
def card_length : ℕ := 16
def card_width : ℕ := 12

-- State the theorem to prove the maximum number of cards.
theorem max_cards_possible : (cardboard_length / card_length) * (cardboard_width / card_width) = 9 :=
by
  sorry -- Skip the proof, as only the statement is required.

end NUMINAMATH_GPT_max_cards_possible_l2251_225164


namespace NUMINAMATH_GPT_calculate_expression_l2251_225144

theorem calculate_expression :
  (3 + 5) * (3^2 + 5^2) * (3^4 + 5^4) * (3^8 + 5^8) * (3^16 + 5^16) * (3^32 + 5^32) * (3^64 + 5^64) = 3^128 - 5^128 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2251_225144


namespace NUMINAMATH_GPT_g_18_value_l2251_225132

-- Define the function g as taking positive integers to positive integers
variable (g : ℕ+ → ℕ+)

-- Define the conditions for the function g
axiom increasing (n : ℕ+) : g (n + 1) > g n
axiom multiplicative (m n : ℕ+) : g (m * n) = g m * g n
axiom power_property (m n : ℕ+) (h : m ≠ n ∧ m ^ (n : ℕ) = n ^ (m : ℕ)) :
  g m = n ∨ g n = m

-- Prove that g(18) is 72
theorem g_18_value : g 18 = 72 :=
sorry

end NUMINAMATH_GPT_g_18_value_l2251_225132


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2251_225118

namespace SetsIntersectionProof

def setA : Set ℝ := { x | |x| ≤ 2 }
def setB : Set ℝ := { x | x < 1 }

theorem intersection_of_A_and_B :
  setA ∩ setB = { x | -2 ≤ x ∧ x < 1 } :=
sorry

end SetsIntersectionProof

end NUMINAMATH_GPT_intersection_of_A_and_B_l2251_225118


namespace NUMINAMATH_GPT_percentage_selected_in_state_A_l2251_225150

-- Definitions
def num_candidates : ℕ := 8000
def percentage_selected_state_B : ℕ := 7
def extra_selected_candidates : ℕ := 80

-- Question
theorem percentage_selected_in_state_A :
  ∃ (P : ℕ), ((P / 100) * 8000 + 80 = 560) ∧ (P = 6) := sorry

end NUMINAMATH_GPT_percentage_selected_in_state_A_l2251_225150


namespace NUMINAMATH_GPT_solve_quadratic_equation_l2251_225107

theorem solve_quadratic_equation (x : ℝ) : 4 * (x - 1)^2 = 36 ↔ (x = 4 ∨ x = -2) :=
by sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l2251_225107


namespace NUMINAMATH_GPT_gilda_stickers_left_l2251_225151

variable (S : ℝ) (hS : S > 0)

def remaining_after_olga : ℝ := 0.70 * S
def remaining_after_sam : ℝ := 0.80 * remaining_after_olga S
def remaining_after_max : ℝ := 0.70 * remaining_after_sam S
def remaining_after_charity : ℝ := 0.90 * remaining_after_max S

theorem gilda_stickers_left :
  remaining_after_charity S / S * 100 = 35.28 := by
  sorry

end NUMINAMATH_GPT_gilda_stickers_left_l2251_225151


namespace NUMINAMATH_GPT_veronica_photo_choices_l2251_225188

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

noncomputable def choose (n k : ℕ) : ℕ :=
factorial n / (factorial k * factorial (n - k))

theorem veronica_photo_choices : choose 5 3 + choose 5 4 = 15 := by
  sorry

end NUMINAMATH_GPT_veronica_photo_choices_l2251_225188


namespace NUMINAMATH_GPT_ball_bounce_height_l2251_225180

open Real

theorem ball_bounce_height :
  ∃ k : ℕ, (20 * (2 / 3 : ℝ)^k < 2) ∧ (∀ n : ℕ, n < k → 20 * (2 / 3 : ℝ)^n ≥ 2) ∧ k = 6 :=
sorry

end NUMINAMATH_GPT_ball_bounce_height_l2251_225180


namespace NUMINAMATH_GPT_simplify_fraction_l2251_225104

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2251_225104


namespace NUMINAMATH_GPT_paula_travel_fraction_l2251_225126

theorem paula_travel_fraction :
  ∀ (f : ℚ), 
    (∀ (L_time P_time travel_total : ℚ), 
      L_time = 70 →
      P_time = 70 * f →
      travel_total = 504 →
      (L_time + 5 * L_time + P_time + P_time = travel_total) →
      f = 3/5) :=
by
  sorry

end NUMINAMATH_GPT_paula_travel_fraction_l2251_225126


namespace NUMINAMATH_GPT_robert_total_balls_l2251_225181

-- Define the conditions
def robert_initial_balls : ℕ := 25
def tim_balls : ℕ := 40

-- Mathematically equivalent proof problem
theorem robert_total_balls : 
  robert_initial_balls + (tim_balls / 2) = 45 := by
  sorry

end NUMINAMATH_GPT_robert_total_balls_l2251_225181


namespace NUMINAMATH_GPT_cat_count_l2251_225195

def initial_cats : ℕ := 2
def female_kittens : ℕ := 3
def male_kittens : ℕ := 2
def total_kittens : ℕ := female_kittens + male_kittens
def total_cats : ℕ := initial_cats + total_kittens

theorem cat_count : total_cats = 7 := by
  unfold total_cats
  unfold initial_cats total_kittens
  unfold female_kittens male_kittens
  rfl

end NUMINAMATH_GPT_cat_count_l2251_225195


namespace NUMINAMATH_GPT_ella_spent_on_video_games_last_year_l2251_225169

theorem ella_spent_on_video_games_last_year 
  (new_salary : ℝ) 
  (raise : ℝ) 
  (percentage_spent_on_video_games : ℝ) 
  (h_new_salary : new_salary = 275) 
  (h_raise : raise = 0.10) 
  (h_percentage_spent : percentage_spent_on_video_games = 0.40) :
  (new_salary / (1 + raise) * percentage_spent_on_video_games = 100) :=
by
  sorry

end NUMINAMATH_GPT_ella_spent_on_video_games_last_year_l2251_225169


namespace NUMINAMATH_GPT_lexie_crayons_count_l2251_225148

variable (number_of_boxes : ℕ) (crayons_per_box : ℕ)

theorem lexie_crayons_count (h1: number_of_boxes = 10) (h2: crayons_per_box = 8) :
  (number_of_boxes * crayons_per_box) = 80 := by
  sorry

end NUMINAMATH_GPT_lexie_crayons_count_l2251_225148


namespace NUMINAMATH_GPT_combined_percent_of_6th_graders_l2251_225112

theorem combined_percent_of_6th_graders (num_students_pineview : ℕ) 
                                        (percent_6th_pineview : ℝ) 
                                        (num_students_oakridge : ℕ)
                                        (percent_6th_oakridge : ℝ)
                                        (num_students_maplewood : ℕ)
                                        (percent_6th_maplewood : ℝ) 
                                        (total_students : ℝ) :
    num_students_pineview = 150 →
    percent_6th_pineview = 0.15 →
    num_students_oakridge = 180 →
    percent_6th_oakridge = 0.17 →
    num_students_maplewood = 170 →
    percent_6th_maplewood = 0.15 →
    total_students = 500 →
    ((percent_6th_pineview * num_students_pineview) + 
     (percent_6th_oakridge * num_students_oakridge) + 
     (percent_6th_maplewood * num_students_maplewood)) / 
    total_students * 100 = 15.72 :=
by
  sorry

end NUMINAMATH_GPT_combined_percent_of_6th_graders_l2251_225112


namespace NUMINAMATH_GPT_find_k_from_hexadecimal_to_decimal_l2251_225185

theorem find_k_from_hexadecimal_to_decimal 
  (k : ℕ) 
  (h : 1 * 6^3 + k * 6 + 5 = 239) : 
  k = 3 := by
  sorry

end NUMINAMATH_GPT_find_k_from_hexadecimal_to_decimal_l2251_225185


namespace NUMINAMATH_GPT_twenty_four_game_l2251_225135

theorem twenty_four_game : 8 / (3 - 8 / 3) = 24 := 
by
  sorry

end NUMINAMATH_GPT_twenty_four_game_l2251_225135


namespace NUMINAMATH_GPT_count_good_numbers_l2251_225163

theorem count_good_numbers : 
  (∃ (f : Fin 10 → ℕ), ∀ n, 
    (2020 % n = 22 ↔ 
      (n = f 0 ∨ n = f 1 ∨ n = f 2 ∨ n = f 3 ∨ 
       n = f 4 ∨ n = f 5 ∨ n = f 6 ∨ n = f 7 ∨ 
       n = f 8 ∨ n = f 9))) :=
sorry

end NUMINAMATH_GPT_count_good_numbers_l2251_225163


namespace NUMINAMATH_GPT_work_completion_l2251_225179

theorem work_completion (days_A : ℕ) (days_B : ℕ) (hA : days_A = 14) (hB : days_B = 35) :
  let rate_A := 1 / (days_A : ℚ)
  let rate_B := 1 / (days_B : ℚ)
  let combined_rate := rate_A + rate_B
  let days_AB := 1 / combined_rate
  days_AB = 10 := by
  sorry

end NUMINAMATH_GPT_work_completion_l2251_225179


namespace NUMINAMATH_GPT_slower_train_speed_l2251_225142

theorem slower_train_speed
  (v : ℝ) -- the speed of the slower train (kmph)
  (faster_train_speed : ℝ := 72)        -- the speed of the faster train
  (time_to_cross_man : ℝ := 18)         -- time to cross a man in the slower train (seconds)
  (faster_train_length : ℝ := 180)      -- length of the faster train (meters))
  (conversion_factor : ℝ := 5 / 18)     -- conversion factor from kmph to m/s
  (relative_speed_m_s : ℝ := ((faster_train_speed - v) * conversion_factor)) :
  ((faster_train_length : ℝ) = (relative_speed_m_s * time_to_cross_man)) →
  v = 36 :=
by
  -- the actual proof needs to be filled here
  sorry

end NUMINAMATH_GPT_slower_train_speed_l2251_225142


namespace NUMINAMATH_GPT_carrot_broccoli_ratio_l2251_225143

variables (total_earnings broccoli_earnings cauliflower_earnings spinach_earnings carrot_earnings : ℕ)

-- Define the conditions
def is_condition_satisfied :=
  total_earnings = 380 ∧
  broccoli_earnings = 57 ∧
  cauliflower_earnings = 136 ∧
  spinach_earnings = (carrot_earnings / 2 + 16)

-- Define the proof problem that checks the ratio
theorem carrot_broccoli_ratio (h : is_condition_satisfied total_earnings broccoli_earnings cauliflower_earnings spinach_earnings carrot_earnings) :
  ((carrot_earnings + ((carrot_earnings / 2) + 16)) + broccoli_earnings + cauliflower_earnings = total_earnings) →
  (carrot_earnings / broccoli_earnings = 2) :=
sorry

end NUMINAMATH_GPT_carrot_broccoli_ratio_l2251_225143


namespace NUMINAMATH_GPT_jenna_average_speed_l2251_225131

theorem jenna_average_speed (total_distance : ℕ) (total_time : ℕ) 
(first_segment_speed : ℕ) (second_segment_speed : ℕ) (third_segment_speed : ℕ) : 
  total_distance = 150 ∧ total_time = 2 ∧ first_segment_speed = 50 ∧ 
  second_segment_speed = 70 → third_segment_speed = 105 := 
by 
  intros h
  sorry

end NUMINAMATH_GPT_jenna_average_speed_l2251_225131
