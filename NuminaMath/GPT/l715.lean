import Mathlib

namespace NUMINAMATH_GPT_original_price_of_cycle_l715_71505

theorem original_price_of_cycle (SP : ℕ) (P : ℕ) (h1 : SP = 1800) (h2 : SP = 9 * P / 10) : P = 2000 :=
by
  have hSP_eq : SP = 1800 := h1
  have hSP_def : SP = 9 * P / 10 := h2
  -- Now we need to combine these to prove P = 2000
  sorry

end NUMINAMATH_GPT_original_price_of_cycle_l715_71505


namespace NUMINAMATH_GPT_eldest_boy_age_l715_71528

theorem eldest_boy_age (a b c : ℕ) (h1 : a + b + c = 45) (h2 : 3 * c = 7 * a) (h3 : 5 * c = 7 * b) : c = 21 := 
sorry

end NUMINAMATH_GPT_eldest_boy_age_l715_71528


namespace NUMINAMATH_GPT_f_800_l715_71504

-- Definitions of hypothesis from conditions given
def f : ℕ → ℤ := sorry
axiom f_mul (x y : ℕ) : f (x * y) = f x + f y
axiom f_10 : f 10 = 10
axiom f_40 : f 40 = 18

-- Proof problem statement: prove that f(800) = 32
theorem f_800 : f 800 = 32 := 
by
  sorry

end NUMINAMATH_GPT_f_800_l715_71504


namespace NUMINAMATH_GPT_first_class_circular_permutations_second_class_circular_permutations_l715_71512

section CircularPermutations

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def perm_count (a b c : ℕ) : ℕ :=
  factorial (a + b + c) / (factorial a * factorial b * factorial c)

theorem first_class_circular_permutations : perm_count 2 2 4 / 8 = 52 := by
  sorry

theorem second_class_circular_permutations : perm_count 2 2 4 / 2 / 4 = 33 := by
  sorry

end CircularPermutations

end NUMINAMATH_GPT_first_class_circular_permutations_second_class_circular_permutations_l715_71512


namespace NUMINAMATH_GPT_group_size_l715_71545

theorem group_size (boxes_per_man total_boxes : ℕ) (h1 : boxes_per_man = 2) (h2 : total_boxes = 14) :
  total_boxes / boxes_per_man = 7 := by
  -- Definitions and conditions from the problem
  have man_can_carry_2_boxes : boxes_per_man = 2 := h1
  have group_can_hold_14_boxes : total_boxes = 14 := h2
  -- Proof follows from these conditions
  sorry

end NUMINAMATH_GPT_group_size_l715_71545


namespace NUMINAMATH_GPT_bridget_apples_l715_71566

/-!
# Problem statement
Bridget bought a bag of apples. She gave half of the apples to Ann. She gave 5 apples to Cassie,
and 2 apples to Dan. She kept 6 apples for herself. Prove that Bridget originally bought 26 apples.
-/

theorem bridget_apples (x : ℕ) 
  (H1 : x / 2 + 2 * (x % 2) / 2 - 5 - 2 = 6) : x = 26 :=
sorry

end NUMINAMATH_GPT_bridget_apples_l715_71566


namespace NUMINAMATH_GPT_smallest_value_satisfies_equation_l715_71596

theorem smallest_value_satisfies_equation : ∃ x : ℝ, (|5 * x + 9| = 34) ∧ x = -8.6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_satisfies_equation_l715_71596


namespace NUMINAMATH_GPT_desired_average_sale_l715_71557

def s1 := 2500
def s2 := 4000
def s3 := 3540
def s4 := 1520
def avg := 2890

theorem desired_average_sale : (s1 + s2 + s3 + s4) / 4 = avg := by
  sorry

end NUMINAMATH_GPT_desired_average_sale_l715_71557


namespace NUMINAMATH_GPT_verify_Fermat_point_l715_71586

open Real

theorem verify_Fermat_point :
  let D := (0, 0)
  let E := (6, 4)
  let F := (3, -2)
  let Q := (2, 1)
  let distance (P₁ P₂ : ℝ × ℝ) : ℝ := sqrt ((P₂.1 - P₁.1)^2 + (P₂.2 - P₁.2)^2)
  distance D Q + distance E Q + distance F Q = 5 + sqrt 5 + sqrt 10 := by
sorry

end NUMINAMATH_GPT_verify_Fermat_point_l715_71586


namespace NUMINAMATH_GPT_gold_coin_multiple_l715_71590

theorem gold_coin_multiple (x y k : ℕ) (h₁ : x + y = 16) (h₂ : x ≠ y) (h₃ : x^2 - y^2 = k * (x - y)) : k = 16 :=
sorry

end NUMINAMATH_GPT_gold_coin_multiple_l715_71590


namespace NUMINAMATH_GPT_max_students_l715_71598

theorem max_students 
  (x : ℕ) 
  (h_lt : x < 100)
  (h_mod8 : x % 8 = 5) 
  (h_mod5 : x % 5 = 3) 
  : x = 93 := 
sorry

end NUMINAMATH_GPT_max_students_l715_71598


namespace NUMINAMATH_GPT_number_of_passed_candidates_l715_71523

-- Definitions based on conditions:
def total_candidates : ℕ := 120
def avg_total_marks : ℝ := 35
def avg_passed_marks : ℝ := 39
def avg_failed_marks : ℝ := 15

-- The number of candidates who passed the examination:
theorem number_of_passed_candidates :
  ∃ (P F : ℕ), 
    P + F = total_candidates ∧
    39 * P + 15 * F = total_candidates * avg_total_marks ∧
    P = 100 :=
by
  sorry

end NUMINAMATH_GPT_number_of_passed_candidates_l715_71523


namespace NUMINAMATH_GPT_alex_points_l715_71583

variable {x y : ℕ} -- x is the number of three-point shots, y is the number of two-point shots
variable (success_rate_3 success_rate_2 : ℚ) -- success rates for three-point and two-point shots
variable (total_shots : ℕ) -- total number of shots

def alex_total_points (x y : ℕ) (success_rate_3 success_rate_2 : ℚ) : ℚ :=
  3 * success_rate_3 * x + 2 * success_rate_2 * y

axiom condition_1 : success_rate_3 = 0.25
axiom condition_2 : success_rate_2 = 0.20
axiom condition_3 : total_shots = 40
axiom condition_4 : x + y = total_shots

theorem alex_points : alex_total_points x y 0.25 0.20 = 30 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_alex_points_l715_71583


namespace NUMINAMATH_GPT_stamp_arrangements_equals_76_l715_71556

-- Define the conditions of the problem
def stamps_available : List (ℕ × ℕ) := 
  [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), 
   (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), 
   (17, 17), (18, 18), (19, 19)]

-- Define a function to compute the number of different arrangements
noncomputable def count_stamp_arrangements : ℕ :=
  -- This is a placeholder for the actual implementation
  sorry

-- State the theorem to be proven
theorem stamp_arrangements_equals_76 : count_stamp_arrangements = 76 :=
sorry

end NUMINAMATH_GPT_stamp_arrangements_equals_76_l715_71556


namespace NUMINAMATH_GPT_average_temp_is_correct_l715_71573

-- Define the temperatures for each day
def sunday_temp : ℕ := 40
def monday_temp : ℕ := 50
def tuesday_temp : ℕ := 65
def wednesday_temp : ℕ := 36
def thursday_temp : ℕ := 82
def friday_temp : ℕ := 72
def saturday_temp : ℕ := 26

-- Define the total number of days in the week
def days_in_week : ℕ := 7

-- Define the total temperature for the week
def total_temperature : ℕ := sunday_temp + monday_temp + tuesday_temp + 
                             wednesday_temp + thursday_temp + friday_temp + 
                             saturday_temp

-- Define the average temperature calculation
def average_temperature : ℕ := total_temperature / days_in_week

-- The theorem to be proved
theorem average_temp_is_correct : average_temperature = 53 := by
  sorry

end NUMINAMATH_GPT_average_temp_is_correct_l715_71573


namespace NUMINAMATH_GPT_num_entrees_ordered_l715_71592

-- Define the conditions
def appetizer_cost: ℝ := 10
def entree_cost: ℝ := 20
def tip_rate: ℝ := 0.20
def total_spent: ℝ := 108

-- Define the theorem to prove the number of entrees ordered
theorem num_entrees_ordered : ∃ E : ℝ, (entree_cost * E) + appetizer_cost + (tip_rate * ((entree_cost * E) + appetizer_cost)) = total_spent ∧ E = 4 := 
by
  sorry

end NUMINAMATH_GPT_num_entrees_ordered_l715_71592


namespace NUMINAMATH_GPT_garden_area_in_square_meters_l715_71527

def garden_width_cm : ℕ := 500
def garden_length_cm : ℕ := 800
def conversion_factor_cm2_to_m2 : ℕ := 10000

theorem garden_area_in_square_meters : (garden_length_cm * garden_width_cm) / conversion_factor_cm2_to_m2 = 40 :=
by
  sorry

end NUMINAMATH_GPT_garden_area_in_square_meters_l715_71527


namespace NUMINAMATH_GPT_abs_neg_one_half_eq_one_half_l715_71561

theorem abs_neg_one_half_eq_one_half : abs (-1/2) = 1/2 := 
by sorry

end NUMINAMATH_GPT_abs_neg_one_half_eq_one_half_l715_71561


namespace NUMINAMATH_GPT_problem_inequality_l715_71570

theorem problem_inequality (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h₁ : 1 < a₁) (h₂ : 1 < a₂) (h₃ : 1 < a₃) (h₄ : 1 < a₄) (h₅ : 1 < a₅) :
  (1 + a₁) * (1 + a₂) * (1 + a₃) * (1 + a₄) * (1 + a₅) ≤ 16 * (a₁ * a₂ * a₃ * a₄ * a₅ + 1) :=
sorry

end NUMINAMATH_GPT_problem_inequality_l715_71570


namespace NUMINAMATH_GPT_rainfall_on_first_day_l715_71513

theorem rainfall_on_first_day (R1 R2 R3 : ℕ) 
  (hR2 : R2 = 34)
  (hR3 : R3 = R2 - 12)
  (hTotal : R1 + R2 + R3 = 82) : 
  R1 = 26 := by
  sorry

end NUMINAMATH_GPT_rainfall_on_first_day_l715_71513


namespace NUMINAMATH_GPT_math_problem_l715_71594

noncomputable def sqrt180 : ℝ := Real.sqrt 180
noncomputable def two_thirds_sqrt180 : ℝ := (2 / 3) * sqrt180
noncomputable def forty_percent_300_cubed : ℝ := (0.4 * 300)^3
noncomputable def forty_percent_180 : ℝ := 0.4 * 180
noncomputable def one_third_less_forty_percent_180 : ℝ := forty_percent_180 - (1 / 3) * forty_percent_180

theorem math_problem : 
  (two_thirds_sqrt180 * forty_percent_300_cubed) - one_third_less_forty_percent_180 = 15454377.6 :=
  by
    have h1 : sqrt180 = Real.sqrt 180 := rfl
    have h2 : two_thirds_sqrt180 = (2 / 3) * sqrt180 := rfl
    have h3 : forty_percent_300_cubed = (0.4 * 300)^3 := rfl
    have h4 : forty_percent_180 = 0.4 * 180 := rfl
    have h5 : one_third_less_forty_percent_180 = forty_percent_180 - (1 / 3) * forty_percent_180 := rfl
    sorry

end NUMINAMATH_GPT_math_problem_l715_71594


namespace NUMINAMATH_GPT_train_passes_bridge_in_expected_time_l715_71580

def train_length : ℕ := 360
def speed_kmph : ℕ := 45
def bridge_length : ℕ := 140

def speed_mps : ℚ := (speed_kmph * 1000) / 3600
def total_distance : ℕ := train_length + bridge_length
def time_to_pass : ℚ := total_distance / speed_mps

theorem train_passes_bridge_in_expected_time : time_to_pass = 40 := by
  sorry

end NUMINAMATH_GPT_train_passes_bridge_in_expected_time_l715_71580


namespace NUMINAMATH_GPT_george_hourly_rate_l715_71537

theorem george_hourly_rate (total_hours : ℕ) (total_amount : ℕ) (h1 : total_hours = 7 + 2)
  (h2 : total_amount = 45) : 
  total_amount / total_hours = 5 := 
by sorry

end NUMINAMATH_GPT_george_hourly_rate_l715_71537


namespace NUMINAMATH_GPT_mark_sandwiches_l715_71565

/--
Each day of a 6-day workweek, Mark bought either an 80-cent donut or a $1.20 sandwich. 
His total expenditure for the week was an exact number of dollars.
Prove that Mark bought exactly 3 sandwiches.
-/
theorem mark_sandwiches (s d : ℕ) (h1 : s + d = 6) (h2 : ∃ k : ℤ, 120 * s + 80 * d = 100 * k) : s = 3 :=
by
  sorry

end NUMINAMATH_GPT_mark_sandwiches_l715_71565


namespace NUMINAMATH_GPT_natural_number_factors_of_M_l715_71510

def M : ℕ := (2^3) * (3^2) * (5^5) * (7^1) * (11^2)

theorem natural_number_factors_of_M : ∃ n : ℕ, n = 432 ∧ (∀ d, d ∣ M → d > 0 → d ≤ M) :=
by
  let number_of_factors := (3 + 1) * (2 + 1) * (5 + 1) * (1 + 1) * (2 + 1)
  use number_of_factors
  sorry

end NUMINAMATH_GPT_natural_number_factors_of_M_l715_71510


namespace NUMINAMATH_GPT_relationship_between_P_and_Q_l715_71548

-- Define the sets P and Q
def P : Set ℝ := { x | x < 4 }
def Q : Set ℝ := { x | -2 < x ∧ x < 2 }

theorem relationship_between_P_and_Q : P ⊇ Q :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_P_and_Q_l715_71548


namespace NUMINAMATH_GPT_math_problem_l715_71550

variable (a b c d : ℝ)

theorem math_problem 
    (h1 : a + b + c + d = 6)
    (h2 : a^2 + b^2 + c^2 + d^2 = 12) :
    36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧
    4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 := 
by
    sorry

end NUMINAMATH_GPT_math_problem_l715_71550


namespace NUMINAMATH_GPT_ratio_of_ap_l715_71569

theorem ratio_of_ap (a d : ℕ) (h : 30 * a + 435 * d = 3 * (15 * a + 105 * d)) : a = 8 * d :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_ap_l715_71569


namespace NUMINAMATH_GPT_a5_is_16_S8_is_255_l715_71515

-- Define the sequence
def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 2 * seq n

-- Definition of the geometric sum
def geom_sum (n : ℕ) : ℕ :=
  (2 ^ (n + 1) - 1)

-- Prove that a₅ = 16
theorem a5_is_16 : seq 5 = 16 :=
  by
  unfold seq
  sorry

-- Prove that the sum of the first 8 terms, S₈ = 255
theorem S8_is_255 : geom_sum 7 = 255 :=
  by 
  unfold geom_sum
  sorry

end NUMINAMATH_GPT_a5_is_16_S8_is_255_l715_71515


namespace NUMINAMATH_GPT_sean_final_cost_l715_71524

noncomputable def totalCost (sodaCount soupCount sandwichCount saladCount : ℕ)
                            (pricePerSoda pricePerSoup pricePerSandwich pricePerSalad : ℚ)
                            (discountRate taxRate : ℚ) : ℚ :=
  let totalCostBeforeDiscount := (sodaCount * pricePerSoda) +
                                (soupCount * pricePerSoup) +
                                (sandwichCount * pricePerSandwich) +
                                (saladCount * pricePerSalad)
  let discountedTotal := totalCostBeforeDiscount * (1 - discountRate)
  let finalCost := discountedTotal * (1 + taxRate)
  finalCost

theorem sean_final_cost : 
  totalCost 4 3 2 1 
            1 (2 * 1) (4 * (2 * 1)) (2 * (4 * (2 * 1)))
            0.1 0.05 = 39.69 := 
by
  sorry

end NUMINAMATH_GPT_sean_final_cost_l715_71524


namespace NUMINAMATH_GPT_fruit_boxes_needed_l715_71558

noncomputable def fruit_boxes : ℕ × ℕ × ℕ :=
  let baskets : ℕ := 7
  let peaches_per_basket : ℕ := 23
  let apples_per_basket : ℕ := 19
  let oranges_per_basket : ℕ := 31
  let peaches_eaten : ℕ := 7
  let apples_eaten : ℕ := 5
  let oranges_eaten : ℕ := 3
  let peaches_box_size : ℕ := 13
  let apples_box_size : ℕ := 11
  let oranges_box_size : ℕ := 17

  let total_peaches := baskets * peaches_per_basket
  let total_apples := baskets * apples_per_basket
  let total_oranges := baskets * oranges_per_basket

  let remaining_peaches := total_peaches - peaches_eaten
  let remaining_apples := total_apples - apples_eaten
  let remaining_oranges := total_oranges - oranges_eaten

  let peaches_boxes := (remaining_peaches + peaches_box_size - 1) / peaches_box_size
  let apples_boxes := (remaining_apples + apples_box_size - 1) / apples_box_size
  let oranges_boxes := (remaining_oranges + oranges_box_size - 1) / oranges_box_size

  (peaches_boxes, apples_boxes, oranges_boxes)

theorem fruit_boxes_needed :
  fruit_boxes = (12, 12, 13) := by 
  sorry

end NUMINAMATH_GPT_fruit_boxes_needed_l715_71558


namespace NUMINAMATH_GPT_find_louis_age_l715_71579

variables (C L : ℕ)

-- Conditions:
-- 1. In some years, Carla will be 30 years old
-- 2. The sum of the current ages of Carla and Louis is 55

theorem find_louis_age (h1 : ∃ n, C + n = 30) (h2 : C + L = 55) : L = 25 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_louis_age_l715_71579


namespace NUMINAMATH_GPT_arithmetic_sequence_a14_l715_71511

theorem arithmetic_sequence_a14 (a : ℕ → ℤ) (h1 : a 4 = 5) (h2 : a 9 = 17) (h3 : 2 * a 9 = a 14 + a 4) : a 14 = 29 := sorry

end NUMINAMATH_GPT_arithmetic_sequence_a14_l715_71511


namespace NUMINAMATH_GPT_total_winter_clothing_l715_71593

theorem total_winter_clothing (boxes : ℕ) (scarves_per_box mittens_per_box : ℕ) (h_boxes : boxes = 8) (h_scarves : scarves_per_box = 4) (h_mittens : mittens_per_box = 6) : 
  boxes * (scarves_per_box + mittens_per_box) = 80 := 
by
  sorry

end NUMINAMATH_GPT_total_winter_clothing_l715_71593


namespace NUMINAMATH_GPT_ratio_2_10_as_percent_l715_71562

-- Define the problem conditions as given
def ratio_2_10 := 2 / 10

-- Express the question which is to show the percentage equivalent of the ratio 2:10
theorem ratio_2_10_as_percent : (ratio_2_10 * 100) = 20 :=
by
  -- Starting statement
  sorry -- Proof is not required here

end NUMINAMATH_GPT_ratio_2_10_as_percent_l715_71562


namespace NUMINAMATH_GPT_parallelogram_height_l715_71577

theorem parallelogram_height (base height area : ℝ) (h_base : base = 9) (h_area : area = 33.3) (h_formula : area = base * height) : height = 3.7 :=
by
  -- Proof goes here, but currently skipped
  sorry

end NUMINAMATH_GPT_parallelogram_height_l715_71577


namespace NUMINAMATH_GPT_thickness_and_width_l715_71589
noncomputable def channelThicknessAndWidth (L W v₀ h₀ θ g : ℝ) : ℝ × ℝ :=
let K := W * h₀ * v₀
let v := v₀ + Real.sqrt (2 * g * Real.sin θ * L)
let x := K / (v * W)
let y := K / (h₀ * v)
(x, y)

theorem thickness_and_width :
  channelThicknessAndWidth 10 3.5 1.4 0.4 (12 * Real.pi / 180) 9.81 = (0.072, 0.629) :=
by
  sorry

end NUMINAMATH_GPT_thickness_and_width_l715_71589


namespace NUMINAMATH_GPT_find_m_l715_71568

def numFactorsOf2 (k : ℕ) : ℕ :=
  k / 2 + k / 4 + k / 8 + k / 16 + k / 32 + k / 64 + k / 128 + k / 256

theorem find_m : ∃ m : ℕ, m > 1990 ^ 1990 ∧ m = 3 ^ 1990 + numFactorsOf2 m :=
by
  sorry

end NUMINAMATH_GPT_find_m_l715_71568


namespace NUMINAMATH_GPT_newLampTaller_l715_71540

-- Define the heights of the old and new lamps
def oldLampHeight : ℝ := 1
def newLampHeight : ℝ := 2.33

-- Define the proof statement
theorem newLampTaller : newLampHeight - oldLampHeight = 1.33 :=
by
  sorry

end NUMINAMATH_GPT_newLampTaller_l715_71540


namespace NUMINAMATH_GPT_mitchell_total_pages_read_l715_71529

def pages_per_chapter : ℕ := 40
def chapters_read_before : ℕ := 10
def pages_read_11th_before : ℕ := 20
def chapters_read_after : ℕ := 2

def total_pages_read := 
  pages_per_chapter * chapters_read_before + pages_read_11th_before + pages_per_chapter * chapters_read_after

theorem mitchell_total_pages_read : total_pages_read = 500 := by
  sorry

end NUMINAMATH_GPT_mitchell_total_pages_read_l715_71529


namespace NUMINAMATH_GPT_total_money_taken_in_l715_71585

-- Define the conditions as constants
def total_tickets : ℕ := 800
def advanced_ticket_price : ℝ := 14.5
def door_ticket_price : ℝ := 22.0
def door_tickets_sold : ℕ := 672
def advanced_tickets_sold : ℕ := total_tickets - door_tickets_sold
def total_revenue_advanced : ℝ := advanced_tickets_sold * advanced_ticket_price
def total_revenue_door : ℝ := door_tickets_sold * door_ticket_price
def total_revenue : ℝ := total_revenue_advanced + total_revenue_door

-- State the mathematical proof problem
theorem total_money_taken_in : total_revenue = 16640.00 := by
  sorry

end NUMINAMATH_GPT_total_money_taken_in_l715_71585


namespace NUMINAMATH_GPT_find_ordered_pair_l715_71539

-- We need to define the variables and conditions first.
variables (a c : ℝ)

-- Now we state the conditions.
def quadratic_has_one_solution : Prop :=
  a * c = 25 ∧ a + c = 12 ∧ a < c

-- Finally, we state the main goal to prove.
theorem find_ordered_pair (ha : quadratic_has_one_solution a c) :
  a = 6 - Real.sqrt 11 ∧ c = 6 + Real.sqrt 11 :=
by sorry

end NUMINAMATH_GPT_find_ordered_pair_l715_71539


namespace NUMINAMATH_GPT_children_l715_71571

theorem children's_book_pages (P : ℝ)
  (h1 : P > 0)
  (c1 : ∃ P_rem, P_rem = P - (0.2 * P))
  (c2 : ∃ P_today, P_today = (0.35 * (P - (0.2 * P))))
  (c3 : ∃ Pages_left, Pages_left = (P - (0.2 * P) - (0.35 * (P - (0.2 * P)))) ∧ Pages_left = 130) :
  P = 250 := by
  sorry

end NUMINAMATH_GPT_children_l715_71571


namespace NUMINAMATH_GPT_part1_growth_rate_part2_new_price_l715_71502

-- Definitions based on conditions
def purchase_price : ℕ := 30
def selling_price : ℕ := 40
def january_sales : ℕ := 400
def march_sales : ℕ := 576
def growth_rate (x : ℝ) : Prop := january_sales * (1 + x)^2 = march_sales

-- Part (1): Prove the monthly average growth rate
theorem part1_growth_rate : 
  ∃ (x : ℝ), growth_rate x ∧ x = 0.2 :=
by
  sorry

-- Definitions for part (2) - based on the second condition
def price_reduction (y : ℝ) : Prop := (selling_price - y - purchase_price) * (march_sales + 12 * y) = 4800

-- Part (2): Prove the new price for April
theorem part2_new_price :
  ∃ (y : ℝ), price_reduction y ∧ (selling_price - y) = 38 :=
by
  sorry

end NUMINAMATH_GPT_part1_growth_rate_part2_new_price_l715_71502


namespace NUMINAMATH_GPT_tangent_line_equation_at_x_1_intervals_of_monotonic_increase_l715_71554

noncomputable def f (x : ℝ) := x^3 - x + 3
noncomputable def df (x : ℝ) := 3 * x^2 - 1

theorem tangent_line_equation_at_x_1 : 
  let k := df 1
  let y := f 1
  (2 = k) ∧ (y = 3) ∧ ∀ x y, y - 3 = 2 * (x - 1) ↔ 2 * x - y + 1 = 0 := 
by 
  sorry

theorem intervals_of_monotonic_increase : 
  let x1 := - (Real.sqrt 3) / 3
  let x2 := (Real.sqrt 3) / 3
  ∀ x, (df x > 0 ↔ (x < x1) ∨ (x > x2)) ∧ 
       (df x < 0 ↔ (x1 < x ∧ x < x2)) := 
by 
  sorry

end NUMINAMATH_GPT_tangent_line_equation_at_x_1_intervals_of_monotonic_increase_l715_71554


namespace NUMINAMATH_GPT_actual_miles_traveled_l715_71501

def skipped_digits_odometer (digits : List ℕ) : Prop :=
  digits = [0, 1, 2, 3, 6, 7, 8, 9]

theorem actual_miles_traveled (odometer_reading : String) (actual_miles : ℕ) :
  skipped_digits_odometer [0, 1, 2, 3, 6, 7, 8, 9] →
  odometer_reading = "000306" →
  actual_miles = 134 :=
by
  intros
  sorry

end NUMINAMATH_GPT_actual_miles_traveled_l715_71501


namespace NUMINAMATH_GPT_find_numbers_l715_71530

theorem find_numbers (x y z : ℕ) :
  x + y + z = 35 → 
  2 * y = x + z + 1 → 
  y^2 = (x + 3) * z → 
  (x = 15 ∧ y = 12 ∧ z = 8) ∨ (x = 5 ∧ y = 12 ∧ z = 18) :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_l715_71530


namespace NUMINAMATH_GPT_people_eating_vegetarian_l715_71532

theorem people_eating_vegetarian (only_veg : ℕ) (both_veg_nonveg : ℕ) (total_veg : ℕ) :
  only_veg = 13 ∧ both_veg_nonveg = 6 → total_veg = 19 := 
by
  sorry

end NUMINAMATH_GPT_people_eating_vegetarian_l715_71532


namespace NUMINAMATH_GPT_count_triangles_on_cube_count_triangles_not_in_face_l715_71591

open Nat

def num_triangles_cube : ℕ := 56
def num_triangles_not_in_face : ℕ := 32

theorem count_triangles_on_cube (V : Finset ℕ) (hV : V.card = 8) :
  (V.card.choose 3 = num_triangles_cube) :=
  sorry

theorem count_triangles_not_in_face (V : Finset ℕ) (hV : V.card = 8) :
  (V.card.choose 3 - (6 * 4) = num_triangles_not_in_face) :=
  sorry

end NUMINAMATH_GPT_count_triangles_on_cube_count_triangles_not_in_face_l715_71591


namespace NUMINAMATH_GPT_Carlos_has_highest_result_l715_71588

def Alice_final_result : ℕ := 30 + 3
def Ben_final_result : ℕ := 34 + 3
def Carlos_final_result : ℕ := 13 * 3

theorem Carlos_has_highest_result : (Carlos_final_result > Alice_final_result) ∧ (Carlos_final_result > Ben_final_result) := by
  sorry

end NUMINAMATH_GPT_Carlos_has_highest_result_l715_71588


namespace NUMINAMATH_GPT_Joan_bought_72_eggs_l715_71587

def dozen := 12
def dozens_Joan_bought := 6
def eggs_Joan_bought := dozens_Joan_bought * dozen

theorem Joan_bought_72_eggs : eggs_Joan_bought = 72 := by
  sorry

end NUMINAMATH_GPT_Joan_bought_72_eggs_l715_71587


namespace NUMINAMATH_GPT_inequality_proof_l715_71526

variable {R : Type*} [LinearOrderedField R]

theorem inequality_proof (a b c x y z : R) (h1 : x^2 < a^2) (h2 : y^2 < b^2) (h3 : z^2 < c^2) :
  x^2 + y^2 + z^2 < a^2 + b^2 + c^2 ∧ x^3 + y^3 + z^3 < a^3 + b^3 + c^3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l715_71526


namespace NUMINAMATH_GPT_no_such_function_exists_l715_71520

def f (n : ℕ) : ℕ := sorry

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, n > 1 → (f n = f (f (n - 1)) + f (f (n + 1))) :=
by
  sorry

end NUMINAMATH_GPT_no_such_function_exists_l715_71520


namespace NUMINAMATH_GPT_hyperbola_focus_proof_l715_71555

noncomputable def hyperbola_focus : ℝ × ℝ :=
  (-3, 2.5 + 2 * Real.sqrt 3)

theorem hyperbola_focus_proof :
  ∃ x y : ℝ, 
  -2 * x^2 + 4 * y^2 - 12 * x - 20 * y + 5 = 0 
  → (x = -3) ∧ (y = 2.5 + 2 * Real.sqrt 3) := 
by 
  sorry

end NUMINAMATH_GPT_hyperbola_focus_proof_l715_71555


namespace NUMINAMATH_GPT_total_time_for_phd_l715_71543

def acclimation_period : ℕ := 1 -- in years
def basics_learning_phase : ℕ := 2 -- in years
def research_factor : ℝ := 1.75 -- 75% more time on research
def research_time_without_sabbaticals_and_conferences : ℝ := basics_learning_phase * research_factor
def first_sabbatical : ℝ := 0.5 -- in years (6 months)
def second_sabbatical : ℝ := 0.25 -- in years (3 months)
def first_conference : ℝ := 0.3333 -- in years (4 months)
def second_conference : ℝ := 0.4166 -- in years (5 months)
def additional_research_time : ℝ := first_sabbatical + second_sabbatical + first_conference + second_conference
def total_research_phase_time : ℝ := research_time_without_sabbaticals_and_conferences + additional_research_time
def dissertation_factor : ℝ := 0.5 -- half as long as acclimation period
def time_spent_writing_without_conference : ℝ := dissertation_factor * acclimation_period
def dissertation_conference : ℝ := 0.25 -- in years (3 months)
def total_dissertation_writing_time : ℝ := time_spent_writing_without_conference + dissertation_conference

theorem total_time_for_phd : 
  (acclimation_period + basics_learning_phase + total_research_phase_time + total_dissertation_writing_time) = 8.75 :=
by
  sorry

end NUMINAMATH_GPT_total_time_for_phd_l715_71543


namespace NUMINAMATH_GPT_coin_sum_even_odd_l715_71563

theorem coin_sum_even_odd (S : ℕ) (h : S > 1) : 
  (∃ even_count, (even_count : ℕ) ∈ [0, 2, S]) ∧ (∃ odd_count, ((odd_count : ℕ) - 1) ∈ [0, 2, S]) :=
  sorry

end NUMINAMATH_GPT_coin_sum_even_odd_l715_71563


namespace NUMINAMATH_GPT_second_hand_travel_distance_l715_71546

theorem second_hand_travel_distance (r : ℝ) (t : ℝ) : 
  r = 10 → t = 45 → 2 * t * π * r = 900 * π :=
by
  intro r_def t_def
  sorry

end NUMINAMATH_GPT_second_hand_travel_distance_l715_71546


namespace NUMINAMATH_GPT_ashton_pencils_left_l715_71559

def pencils_left (boxes : Nat) (pencils_per_box : Nat) (pencils_given : Nat) : Nat :=
  boxes * pencils_per_box - pencils_given

theorem ashton_pencils_left :
  pencils_left 2 14 6 = 22 :=
by
  sorry

end NUMINAMATH_GPT_ashton_pencils_left_l715_71559


namespace NUMINAMATH_GPT_find_b_l715_71533

theorem find_b (a b c : ℕ) (h1 : 2 * b = a + c) (h2 : b^2 = c * (a + 1)) (h3 : b^2 = a * (c + 2)) : b = 12 :=
by 
  sorry

end NUMINAMATH_GPT_find_b_l715_71533


namespace NUMINAMATH_GPT_num_triangles_from_decagon_l715_71549

-- Given condition: a decagon has 10 vertices.
def vertices_of_decagon : ℕ := 10
def k : ℕ := 3

-- We are to prove the number of triangles (combinations of 3 vertices from 10) is 120.
theorem num_triangles_from_decagon : Nat.choose vertices_of_decagon k = 120 := by
  sorry

end NUMINAMATH_GPT_num_triangles_from_decagon_l715_71549


namespace NUMINAMATH_GPT_no_integer_solutions_for_equation_l715_71595

theorem no_integer_solutions_for_equation : ¬∃ (a b c : ℤ), a^4 + b^4 = c^4 + 3 := 
  by sorry

end NUMINAMATH_GPT_no_integer_solutions_for_equation_l715_71595


namespace NUMINAMATH_GPT_probability_recruitment_l715_71547

-- Definitions for conditions
def P_A : ℚ := 2/3
def P_A_not_and_B_not : ℚ := 1/12
def P_B_and_C : ℚ := 3/8

-- Independence of A, B, and C
axiom independence_A_B_C : ∀ {P_A P_B P_C : Prop}, 
  (P_A ∧ P_B ∧ P_C) → (P_A ∧ P_B) ∧ (P_A ∧ P_C) ∧ (P_B ∧ P_C)

-- Definition of probabilities of B and C
def P_B : ℚ := 3/4
def P_C : ℚ := 1/2

-- Main theorem
theorem probability_recruitment : 
  P_A = 2/3 ∧ 
  P_A_not_and_B_not = 1/12 ∧ 
  P_B_and_C = 3/8 ∧ 
  (∀ {P_A P_B P_C : Prop}, 
    (P_A ∧ P_B ∧ P_C) → (P_A ∧ P_B) ∧ (P_A ∧ P_C) ∧ (P_B ∧ P_C)) → 
  (P_B = 3/4 ∧ P_C = 1/2) ∧ 
  (2/3 * 3/4 * 1/2 + 1/3 * 3/4 * 1/2 + 2/3 * 1/4 * 1/2 + 2/3 * 3/4 * 1/2 = 17/24) := 
by sorry

end NUMINAMATH_GPT_probability_recruitment_l715_71547


namespace NUMINAMATH_GPT_p_is_necessary_but_not_sufficient_for_q_l715_71519

variable (x : ℝ)

def p := x > 4
def q := 4 < x ∧ x < 10

theorem p_is_necessary_but_not_sufficient_for_q : (∀ x, q x → p x) ∧ ¬(∀ x, p x → q x) := by
  sorry

end NUMINAMATH_GPT_p_is_necessary_but_not_sufficient_for_q_l715_71519


namespace NUMINAMATH_GPT_find_a_l715_71560

def A : Set ℝ := {2, 3}
def B (a : ℝ) : Set ℝ := {1, 2, a}

theorem find_a (a : ℝ) : A ⊆ B a → a = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l715_71560


namespace NUMINAMATH_GPT_blanch_slices_eaten_for_dinner_l715_71582

theorem blanch_slices_eaten_for_dinner :
  ∀ (total_slices eaten_breakfast eaten_lunch eaten_snack slices_left eaten_dinner : ℕ),
  total_slices = 15 →
  eaten_breakfast = 4 →
  eaten_lunch = 2 →
  eaten_snack = 2 →
  slices_left = 2 →
  eaten_dinner = total_slices - (eaten_breakfast + eaten_lunch + eaten_snack) - slices_left →
  eaten_dinner = 5 := by
  intros total_slices eaten_breakfast eaten_lunch eaten_snack slices_left eaten_dinner
  intros h_total_slices h_eaten_breakfast h_eaten_lunch h_eaten_snack h_slices_left h_eaten_dinner
  rw [h_total_slices, h_eaten_breakfast, h_eaten_lunch, h_eaten_snack, h_slices_left] at h_eaten_dinner
  exact h_eaten_dinner

end NUMINAMATH_GPT_blanch_slices_eaten_for_dinner_l715_71582


namespace NUMINAMATH_GPT_geometric_first_term_l715_71581

theorem geometric_first_term (a r : ℝ) (h1 : a * r^3 = 720) (h2 : a * r^6 = 5040) : 
a = 720 / 7 :=
by
  sorry

end NUMINAMATH_GPT_geometric_first_term_l715_71581


namespace NUMINAMATH_GPT_correct_operation_l715_71567

theorem correct_operation : ∀ (a b : ℤ), 3 * a^2 * b - 2 * b * a^2 = a^2 * b :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l715_71567


namespace NUMINAMATH_GPT_inequality_solution_set_l715_71576

theorem inequality_solution_set :
  {x : ℝ | (0 ≤ x ∧ x < 2) ∨ (x = 0)} = {x : ℝ | 0 ≤ x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l715_71576


namespace NUMINAMATH_GPT_total_rent_calculation_l715_71552

variables (x y : ℕ) -- x: number of rooms rented for $40, y: number of rooms rented for $60
variable (rent_total : ℕ)

-- Condition: Each room at the motel was rented for either $40 or $60
-- Condition: If 10 of the rooms that were rented for $60 had instead been rented for $40, the total rent would have been reduced by 50 percent

theorem total_rent_calculation 
  (h1 : 40 * (x + 10) + 60 * (y - 10) = (40 * x + 60 * y) / 2) :
  40 * x + 60 * y = 800 :=
sorry

end NUMINAMATH_GPT_total_rent_calculation_l715_71552


namespace NUMINAMATH_GPT_ten_millions_in_hundred_million_hundred_thousands_in_million_l715_71538

theorem ten_millions_in_hundred_million :
  (100 * 10^6) / (10 * 10^6) = 10 :=
by sorry

theorem hundred_thousands_in_million :
  (1 * 10^6) / (100 * 10^3) = 10 :=
by sorry

end NUMINAMATH_GPT_ten_millions_in_hundred_million_hundred_thousands_in_million_l715_71538


namespace NUMINAMATH_GPT_cubesWithTwoColoredFaces_l715_71575

structure CuboidDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

def numberOfSmallerCubes (d : CuboidDimensions) : ℕ :=
  d.length * d.width * d.height

def numberOfCubesWithTwoColoredFaces (d : CuboidDimensions) : ℕ :=
  2 * (d.length - 2) * 2 + 2 * (d.width - 2) * 2 + 2 * (d.height - 2) * 2

theorem cubesWithTwoColoredFaces :
  numberOfCubesWithTwoColoredFaces { length := 4, width := 3, height := 3 } = 16 := by
  sorry

end NUMINAMATH_GPT_cubesWithTwoColoredFaces_l715_71575


namespace NUMINAMATH_GPT_equation_represents_pair_of_lines_l715_71521

theorem equation_represents_pair_of_lines : ∀ x y : ℝ, 9 * x^2 - 25 * y^2 = 0 → 
                    (x = (5/3) * y ∨ x = -(5/3) * y) :=
by sorry

end NUMINAMATH_GPT_equation_represents_pair_of_lines_l715_71521


namespace NUMINAMATH_GPT_tutors_schedule_l715_71541

theorem tutors_schedule :
  Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9 = 360 :=
by
  sorry

end NUMINAMATH_GPT_tutors_schedule_l715_71541


namespace NUMINAMATH_GPT_solve_equation_l715_71509

theorem solve_equation (x : ℝ) : ((x-3)^2 + 4*x*(x-3) = 0) → (x = 3 ∨ x = 3/5) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l715_71509


namespace NUMINAMATH_GPT_opposite_directions_l715_71507

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem opposite_directions (a b : V) (h : a + 4 • b = 0) : a = -4 • b := sorry

end NUMINAMATH_GPT_opposite_directions_l715_71507


namespace NUMINAMATH_GPT_determine_angle_F_l715_71572

noncomputable def sin := fun x => Real.sin x
noncomputable def cos := fun x => Real.cos x
noncomputable def arcsin := fun x => Real.arcsin x
noncomputable def angleF (D E : ℝ) := 180 - (D + E)

theorem determine_angle_F (D E F : ℝ)
  (h1 : 2 * sin D + 5 * cos E = 7)
  (h2 : 5 * sin E + 2 * cos D = 4) :
  F = arcsin (9 / 10) ∨ F = 180 - arcsin (9 / 10) :=
  sorry

end NUMINAMATH_GPT_determine_angle_F_l715_71572


namespace NUMINAMATH_GPT_simplify_sqrt_sum_l715_71551

noncomputable def sqrt_expr_1 : ℝ := Real.sqrt (12 + 8 * Real.sqrt 3)
noncomputable def sqrt_expr_2 : ℝ := Real.sqrt (12 - 8 * Real.sqrt 3)

theorem simplify_sqrt_sum : sqrt_expr_1 + sqrt_expr_2 = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_sum_l715_71551


namespace NUMINAMATH_GPT_G5_units_digit_is_0_l715_71531

def power_mod (base : ℕ) (exp : ℕ) (modulus : ℕ) : ℕ :=
  (base ^ exp) % modulus

def G (n : ℕ) : ℕ := 2 ^ (3 ^ n) + 2

theorem G5_units_digit_is_0 : (G 5) % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_G5_units_digit_is_0_l715_71531


namespace NUMINAMATH_GPT_value_of_y_l715_71597

theorem value_of_y (x y : ℝ) (h1 : x + y = 5) (h2 : x = 3) : y = 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_l715_71597


namespace NUMINAMATH_GPT_largest_triangle_perimeter_with_7_9_x_l715_71506

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def triangle_side_x_valid (x : ℕ) : Prop :=
  is_divisible_by_3 x ∧ 2 < x ∧ x < 16

theorem largest_triangle_perimeter_with_7_9_x (x : ℕ) (h : triangle_side_x_valid x) : 
  ∃ P : ℕ, P = 7 + 9 + x ∧ P = 31 :=
by
  sorry

end NUMINAMATH_GPT_largest_triangle_perimeter_with_7_9_x_l715_71506


namespace NUMINAMATH_GPT_complex_z_pow_2017_l715_71522

noncomputable def complex_number_z : ℂ := (1 + Complex.I) / (1 - Complex.I)

theorem complex_z_pow_2017 :
  (complex_number_z * (1 - Complex.I) = 1 + Complex.I) → (complex_number_z ^ 2017 = Complex.I) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_complex_z_pow_2017_l715_71522


namespace NUMINAMATH_GPT_cosine_equation_solution_count_l715_71564

open Real

noncomputable def number_of_solutions : ℕ := sorry

theorem cosine_equation_solution_count :
  number_of_solutions = 2 :=
by
  -- Let x be an angle in [0, 2π].
  sorry

end NUMINAMATH_GPT_cosine_equation_solution_count_l715_71564


namespace NUMINAMATH_GPT_value_of_each_bill_l715_71574

theorem value_of_each_bill (bank1_withdrawal bank2_withdrawal number_of_bills : ℕ)
  (h1 : bank1_withdrawal = 300) 
  (h2 : bank2_withdrawal = 300) 
  (h3 : number_of_bills = 30) : 
  (bank1_withdrawal + bank2_withdrawal) / number_of_bills = 20 :=
by
  sorry

end NUMINAMATH_GPT_value_of_each_bill_l715_71574


namespace NUMINAMATH_GPT_problem_b_value_l715_71525

theorem problem_b_value (b : ℤ)
  (h1 : 0 ≤ b)
  (h2 : b ≤ 20)
  (h3 : (3 - b) % 17 = 0) : b = 3 :=
sorry

end NUMINAMATH_GPT_problem_b_value_l715_71525


namespace NUMINAMATH_GPT_find_divisor_l715_71535

-- Define the initial number
def num := 1387

-- Define the number to subtract to make it divisible by some divisor
def least_subtract := 7

-- Define the resulting number after subtraction
def remaining_num := num - least_subtract

-- Define the divisor
def divisor := 23

-- The statement to prove: 1380 is divisible by 23
theorem find_divisor (num_subtract_div : num - least_subtract = remaining_num) 
                     (remaining_divisor : remaining_num = 1380) : 
                     ∃ k : ℕ, 1380 = k * divisor := by
  sorry

end NUMINAMATH_GPT_find_divisor_l715_71535


namespace NUMINAMATH_GPT_problem_l715_71514

-- Definitions and hypotheses based on the given conditions
variable (a b : ℝ)
def sol_set := {x : ℝ | -1/2 < x ∧ x < 1/3}
def quadratic_inequality (x : ℝ) := a * x^2 + b * x + 2

-- Statement expressing that the inequality holds for the given solution set
theorem problem
  (h : ∀ (x : ℝ), x ∈ sol_set → quadratic_inequality a b x > 0) :
  a - b = -10 :=
sorry

end NUMINAMATH_GPT_problem_l715_71514


namespace NUMINAMATH_GPT_translation_coordinates_l715_71508

-- Define starting point
def initial_point : ℤ × ℤ := (-2, 3)

-- Define the point moved up by 2 units
def move_up (p : ℤ × ℤ) (d : ℤ) : ℤ × ℤ :=
  (p.fst, p.snd + d)

-- Define the point moved right by 2 units
def move_right (p : ℤ × ℤ) (d : ℤ) : ℤ × ℤ :=
  (p.fst + d, p.snd)

-- Expected results after movements
def point_up : ℤ × ℤ := (-2, 5)
def point_right : ℤ × ℤ := (0, 3)

-- Proof statement
theorem translation_coordinates :
  move_up initial_point 2 = point_up ∧
  move_right initial_point 2 = point_right :=
by
  sorry

end NUMINAMATH_GPT_translation_coordinates_l715_71508


namespace NUMINAMATH_GPT_arithmetic_neg3_plus_4_l715_71518

theorem arithmetic_neg3_plus_4 : -3 + 4 = 1 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_neg3_plus_4_l715_71518


namespace NUMINAMATH_GPT_solve_x_squared_eq_nine_l715_71503

theorem solve_x_squared_eq_nine (x : ℝ) : x^2 = 9 → (x = 3 ∨ x = -3) :=
by
  -- Proof by sorry placeholder
  sorry

end NUMINAMATH_GPT_solve_x_squared_eq_nine_l715_71503


namespace NUMINAMATH_GPT_registration_methods_count_l715_71542

theorem registration_methods_count (students : Fin 4) (groups : Fin 3) : (3 : ℕ)^4 = 81 :=
by
  sorry

end NUMINAMATH_GPT_registration_methods_count_l715_71542


namespace NUMINAMATH_GPT_onions_total_l715_71584

-- Define the number of onions grown by Sara, Sally, and Fred
def sara_onions : ℕ := 4
def sally_onions : ℕ := 5
def fred_onions : ℕ := 9

-- Define the total onions grown
def total_onions : ℕ := sara_onions + sally_onions + fred_onions

-- Theorem stating the total number of onions grown
theorem onions_total : total_onions = 18 := by
  sorry

end NUMINAMATH_GPT_onions_total_l715_71584


namespace NUMINAMATH_GPT_order_of_logs_l715_71500

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 10 / Real.log 5
noncomputable def c : ℝ := Real.log 14 / Real.log 7

theorem order_of_logs (a_def : a = Real.log 6 / Real.log 3)
                      (b_def : b = Real.log 10 / Real.log 5)
                      (c_def : c = Real.log 14 / Real.log 7) : a > b ∧ b > c := 
by
  sorry

end NUMINAMATH_GPT_order_of_logs_l715_71500


namespace NUMINAMATH_GPT_total_time_correct_l715_71599

-- Define the base speeds and distance
def speed_boat : ℕ := 8
def speed_stream : ℕ := 6
def distance : ℕ := 210

-- Define the speeds downstream and upstream
def speed_downstream : ℕ := speed_boat + speed_stream
def speed_upstream : ℕ := speed_boat - speed_stream

-- Define the time taken for downstream and upstream
def time_downstream : ℕ := distance / speed_downstream
def time_upstream : ℕ := distance / speed_upstream

-- Define the total time taken
def total_time : ℕ := time_downstream + time_upstream

-- The theorem to be proven
theorem total_time_correct : total_time = 120 := by
  sorry

end NUMINAMATH_GPT_total_time_correct_l715_71599


namespace NUMINAMATH_GPT_exact_one_solves_l715_71553

variables (p1 p2 : ℝ)

/-- The probability that exactly one of two persons solves the problem
    when their respective probabilities are p1 and p2. -/
theorem exact_one_solves (h1 : 0 ≤ p1) (h2 : p1 ≤ 1) (h3 : 0 ≤ p2) (h4 : p2 ≤ 1) :
  (p1 * (1 - p2) + p2 * (1 - p1)) = (p1 + p2 - 2 * p1 * p2) := 
sorry

end NUMINAMATH_GPT_exact_one_solves_l715_71553


namespace NUMINAMATH_GPT_find_x_in_gp_l715_71536

theorem find_x_in_gp :
  ∃ x : ℤ, (30 + x)^2 = (10 + x) * (90 + x) ∧ x = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_x_in_gp_l715_71536


namespace NUMINAMATH_GPT_number_of_5_dollar_coins_l715_71578

-- Define the context and the proof problem
theorem number_of_5_dollar_coins (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 5 * y = 125) : y = 15 :=
by sorry

end NUMINAMATH_GPT_number_of_5_dollar_coins_l715_71578


namespace NUMINAMATH_GPT_solve_abs_inequality_l715_71517

/-- Given the inequality 2 ≤ |x - 3| ≤ 8, we want to prove that the solution is [-5 ≤ x ≤ 1] ∪ [5 ≤ x ≤ 11] --/
theorem solve_abs_inequality (x : ℝ) :
  2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 11) :=
sorry

end NUMINAMATH_GPT_solve_abs_inequality_l715_71517


namespace NUMINAMATH_GPT_maximum_value_of_sums_of_cubes_l715_71516

theorem maximum_value_of_sums_of_cubes 
  (a b c d e : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 + e^2 = 9) : 
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 27 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_sums_of_cubes_l715_71516


namespace NUMINAMATH_GPT_probability_of_selected_number_between_l715_71544

open Set

theorem probability_of_selected_number_between (s : Set ℤ) (a b x y : ℤ) 
  (h1 : a = 25) 
  (h2 : b = 925) 
  (h3 : x = 25) 
  (h4 : y = 99) 
  (h5 : s = Set.Icc a b) :
  (y - x + 1 : ℚ) / (b - a + 1 : ℚ) = 75 / 901 := 
by 
  sorry

end NUMINAMATH_GPT_probability_of_selected_number_between_l715_71544


namespace NUMINAMATH_GPT_initial_house_cats_l715_71534

theorem initial_house_cats (H : ℕ) 
  (siamese_cats : ℕ := 38) 
  (cats_sold : ℕ := 45) 
  (cats_left : ℕ := 18) 
  (initial_total_cats : ℕ := siamese_cats + H) 
  (after_sale_cats : ℕ := initial_total_cats - cats_sold) : 
  after_sale_cats = cats_left → H = 25 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_initial_house_cats_l715_71534
