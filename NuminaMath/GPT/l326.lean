import Mathlib

namespace NUMINAMATH_GPT_sum_of_roots_eq_36_l326_32688

theorem sum_of_roots_eq_36 :
  (∃ x1 x2 x3 : ℝ, (11 - x1) ^ 3 + (13 - x2) ^ 3 = (24 - 2 * x3) ^ 3 ∧ 
  (11 - x2) ^ 3 + (13 - x3) ^ 3 = (24 - 2 * x1) ^ 3 ∧ 
  (11 - x3) ^ 3 + (13 - x1) ^ 3 = (24 - 2 * x2) ^ 3 ∧
  x1 + x2 + x3 = 36) :=
sorry

end NUMINAMATH_GPT_sum_of_roots_eq_36_l326_32688


namespace NUMINAMATH_GPT_smallest_lcm_l326_32651

theorem smallest_lcm (k l : ℕ) (hk : 999 < k ∧ k < 10000) (hl : 999 < l ∧ l < 10000)
  (h_gcd : Nat.gcd k l = 5) : Nat.lcm k l = 201000 :=
sorry

end NUMINAMATH_GPT_smallest_lcm_l326_32651


namespace NUMINAMATH_GPT_number_div_0_04_eq_100_9_l326_32659

theorem number_div_0_04_eq_100_9 :
  ∃ number : ℝ, (number / 0.04 = 100.9) ∧ (number = 4.036) :=
sorry

end NUMINAMATH_GPT_number_div_0_04_eq_100_9_l326_32659


namespace NUMINAMATH_GPT_simplify_cubicroot_1600_l326_32606

theorem simplify_cubicroot_1600 : ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ (c^3 * d = 1600) ∧ (c + d = 102) := 
by 
  sorry

end NUMINAMATH_GPT_simplify_cubicroot_1600_l326_32606


namespace NUMINAMATH_GPT_parabola_distance_l326_32680

open Real

theorem parabola_distance (x₀ : ℝ) (h₁ : ∃ p > 0, (x₀^2 = 2 * p * 2) ∧ (2 + p / 2 = 5 / 2)) : abs (sqrt (x₀^2 + 4)) = 2 * sqrt 2 :=
by
  rcases h₁ with ⟨p, hp, h₀, h₂⟩
  sorry

end NUMINAMATH_GPT_parabola_distance_l326_32680


namespace NUMINAMATH_GPT_georgie_initial_avocados_l326_32624

-- Define the conditions
def avocados_needed_per_serving := 3
def servings_made := 3
def avocados_bought_by_sister := 4
def total_avocados_needed := avocados_needed_per_serving * servings_made

-- The statement to prove
theorem georgie_initial_avocados : (total_avocados_needed - avocados_bought_by_sister) = 5 :=
sorry

end NUMINAMATH_GPT_georgie_initial_avocados_l326_32624


namespace NUMINAMATH_GPT_imaginary_unit_cube_l326_32630

theorem imaginary_unit_cube (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i :=
by
  sorry

end NUMINAMATH_GPT_imaginary_unit_cube_l326_32630


namespace NUMINAMATH_GPT_total_resistance_l326_32605

theorem total_resistance (R₀ : ℝ) (h : R₀ = 10) : 
  let R₃ := R₀; let R₄ := R₀; let R₃₄ := R₃ + R₄;
  let R₂ := R₀; let R₅ := R₀; let R₂₃₄ := 1 / (1 / R₂ + 1 / R₃₄ + 1 / R₅);
  let R₁ := R₀; let R₆ := R₀; let R₁₂₃₄ := R₁ + R₂₃₄ + R₆;
  R₁₂₃₄ = 13.33 :=
by 
  sorry

end NUMINAMATH_GPT_total_resistance_l326_32605


namespace NUMINAMATH_GPT_xiaoning_comprehensive_score_l326_32640

theorem xiaoning_comprehensive_score
  (max_score : ℕ := 100)
  (midterm_weight : ℝ := 0.3)
  (final_weight : ℝ := 0.7)
  (midterm_score : ℕ := 80)
  (final_score : ℕ := 90) :
  (midterm_score * midterm_weight + final_score * final_weight) = 87 :=
by
  sorry

end NUMINAMATH_GPT_xiaoning_comprehensive_score_l326_32640


namespace NUMINAMATH_GPT_max_cake_boxes_in_carton_l326_32665

-- Define the dimensions of the carton as constants
def carton_length := 25
def carton_width := 42
def carton_height := 60

-- Define the dimensions of the cake box as constants
def box_length := 8
def box_width := 7
def box_height := 5

-- Define the volume of the carton and the volume of the cake box
def volume_carton := carton_length * carton_width * carton_height
def volume_box := box_length * box_width * box_height

-- Define the theorem statement
theorem max_cake_boxes_in_carton : 
  (volume_carton / volume_box) = 225 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_max_cake_boxes_in_carton_l326_32665


namespace NUMINAMATH_GPT_find_point_A_coordinates_l326_32696

theorem find_point_A_coordinates (A B C : ℝ × ℝ)
  (hB : B = (1, 2)) (hC : C = (3, 4))
  (trans_left : ∃ l : ℝ, A = (B.1 + l, B.2))
  (trans_up : ∃ u : ℝ, A = (C.1, C.2 - u)) :
  A = (3, 2) := 
sorry

end NUMINAMATH_GPT_find_point_A_coordinates_l326_32696


namespace NUMINAMATH_GPT_find_p_q_sum_l326_32615

-- Define the conditions
def p (q : ℤ) : ℤ := q + 20

theorem find_p_q_sum (p q : ℤ) (hp : p * q = 1764) (hq : p - q = 20) :
  p + q = 86 :=
  sorry

end NUMINAMATH_GPT_find_p_q_sum_l326_32615


namespace NUMINAMATH_GPT_frank_fence_length_l326_32623

theorem frank_fence_length (L W total_fence : ℝ) 
  (hW : W = 40) 
  (hArea : L * W = 200) 
  (htotal_fence : total_fence = 2 * L + W) : 
  total_fence = 50 := 
by 
  sorry

end NUMINAMATH_GPT_frank_fence_length_l326_32623


namespace NUMINAMATH_GPT_notched_circle_coordinates_l326_32607

variable (a b : ℝ)

theorem notched_circle_coordinates : 
  let sq_dist_from_origin := a^2 + b^2
  let A := (a, b + 5)
  let C := (a + 3, b)
  (a^2 + (b + 5)^2 = 36 ∧ (a + 3)^2 + b^2 = 36) →
  (sq_dist_from_origin = 4.16 ∧ A = (2, 5.4) ∧ C = (5, 0.4)) :=
by
  sorry

end NUMINAMATH_GPT_notched_circle_coordinates_l326_32607


namespace NUMINAMATH_GPT_unique_solution_arith_prog_system_l326_32610

theorem unique_solution_arith_prog_system (x y : ℝ) : 
  (6 * x + 9 * y = 12) ∧ (15 * x + 18 * y = 21) ↔ (x = -1) ∧ (y = 2) :=
by sorry

end NUMINAMATH_GPT_unique_solution_arith_prog_system_l326_32610


namespace NUMINAMATH_GPT_trig_identity_l326_32653

theorem trig_identity (α : ℝ) : 
  (2 * (Real.sin (4 * α))^2 - 1) / 
  (2 * (1 / Real.tan (Real.pi / 4 + 4 * α)) * (Real.cos (5 * Real.pi / 4 - 4 * α))^2) = -1 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l326_32653


namespace NUMINAMATH_GPT_snail_distance_l326_32641

def speed_A : ℝ := 10
def speed_B : ℝ := 15
def time_difference : ℝ := 0.5

theorem snail_distance : 
  ∃ (D : ℝ) (t_A t_B : ℝ), 
    D = speed_A * t_A ∧ 
    D = speed_B * t_B ∧
    t_A = t_B + time_difference ∧ 
    D = 15 := 
by
  sorry

end NUMINAMATH_GPT_snail_distance_l326_32641


namespace NUMINAMATH_GPT_imaginary_part_of_z_squared_l326_32655

-- Let i be the imaginary unit
def i : ℂ := Complex.I

-- Define the complex number (1 - 2i)
def z : ℂ := 1 - 2 * i

-- Define the expanded form of (1 - 2i)^2
def z_squared : ℂ := z^2

-- State the problem of finding the imaginary part of (1 - 2i)^2
theorem imaginary_part_of_z_squared : (z_squared).im = -4 := by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_z_squared_l326_32655


namespace NUMINAMATH_GPT_correct_pairings_l326_32625

-- Define the employees
inductive Employee
| Jia
| Yi
| Bing
deriving DecidableEq

-- Define the wives
inductive Wife
| A
| B
| C
deriving DecidableEq

-- Define the friendship and age relationships
def isGoodFriend (x y : Employee) : Prop :=
  -- A's husband is Yi's good friend.
  (x = Employee.Jia ∧ y = Employee.Yi) ∨
  (x = Employee.Yi ∧ y = Employee.Jia)

def isYoungest (x : Employee) : Prop :=
  -- Specify that Jia is the youngest
  x = Employee.Jia

def isOlder (x y : Employee) : Prop :=
  -- Bing is older than C's husband.
  x = Employee.Bing ∧ y ≠ Employee.Bing

-- The pairings of husbands and wives: Jia—A, Yi—C, Bing—B.
def pairings (x : Employee) : Wife :=
  match x with
  | Employee.Jia => Wife.A
  | Employee.Yi => Wife.C
  | Employee.Bing => Wife.B

-- Proving the given pairings fit the conditions.
theorem correct_pairings : 
  ∀ (x : Employee), 
  isGoodFriend (Employee.Jia) (Employee.Yi) ∧ 
  isYoungest Employee.Jia ∧ 
  (isOlder Employee.Bing Employee.Jia ∨ isOlder Employee.Bing Employee.Yi) → 
  pairings x = match x with
               | Employee.Jia => Wife.A
               | Employee.Yi => Wife.C
               | Employee.Bing => Wife.B :=
by
  sorry

end NUMINAMATH_GPT_correct_pairings_l326_32625


namespace NUMINAMATH_GPT_hulk_strength_l326_32631

theorem hulk_strength:
    ∃ n: ℕ, (2^(n-1) > 1000) ∧ (∀ m: ℕ, (2^(m-1) > 1000 → n ≤ m)) := sorry

end NUMINAMATH_GPT_hulk_strength_l326_32631


namespace NUMINAMATH_GPT_set_M_roster_method_l326_32685

open Set

theorem set_M_roster_method :
  {a : ℤ | ∃ (n : ℕ), 6 = n * (5 - a)} = {-1, 2, 3, 4} := by
  sorry

end NUMINAMATH_GPT_set_M_roster_method_l326_32685


namespace NUMINAMATH_GPT_expression_value_l326_32687

theorem expression_value (a b : ℤ) (h₁ : a = -5) (h₂ : b = 3) :
  -a - b^4 + a * b = -91 := by
  sorry

end NUMINAMATH_GPT_expression_value_l326_32687


namespace NUMINAMATH_GPT_find_measure_A_and_b_c_sum_l326_32639

open Real

noncomputable def triangle_abc (a b c A B C : ℝ) : Prop :=
  ∀ (A B C : ℝ),
  A + B + C = π ∧
  a = sin A ∧
  b = sin B ∧
  c = sin C ∧
  cos (A - C) - cos (A + C) = sqrt 3 * sin C

theorem find_measure_A_and_b_c_sum (a b c A B C : ℝ)
  (h_triangle : triangle_abc a b c A B C) 
  (h_area : (1/2) * b * c * (sin A) = (3 * sqrt 3) / 16) 
  (h_b_def : b = sin B) :
  A = π / 3 ∧ b + c = sqrt 3 := by
  sorry

end NUMINAMATH_GPT_find_measure_A_and_b_c_sum_l326_32639


namespace NUMINAMATH_GPT_log3_x_minus_1_increasing_l326_32613

noncomputable def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

theorem log3_x_minus_1_increasing : is_increasing_on (fun x => log_base_3 (x - 1)) (Set.Ioi 1) :=
sorry

end NUMINAMATH_GPT_log3_x_minus_1_increasing_l326_32613


namespace NUMINAMATH_GPT_train_pass_time_l326_32632

def speed_jogger := 9   -- in km/hr
def distance_ahead := 240   -- in meters
def length_train := 150   -- in meters
def speed_train := 45   -- in km/hr

noncomputable def time_to_pass_jogger : ℝ :=
  let speed_jogger_mps := speed_jogger * (1000 / 3600)
  let speed_train_mps := speed_train * (1000 / 3600)
  let relative_speed := speed_train_mps - speed_jogger_mps
  let total_distance := distance_ahead + length_train
  total_distance / relative_speed

theorem train_pass_time : time_to_pass_jogger = 39 :=
  by
    sorry

end NUMINAMATH_GPT_train_pass_time_l326_32632


namespace NUMINAMATH_GPT_calculate_expression_l326_32618

variables (a b : ℝ)

theorem calculate_expression : -a^2 * 2 * a^4 * b = -2 * (a^6) * b :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l326_32618


namespace NUMINAMATH_GPT_road_repair_completion_time_l326_32673

theorem road_repair_completion_time (L R r : ℕ) (hL : L = 100) (hR : R = 64) (hr : r = 9) :
  (L - R) / r = 5 :=
by
  sorry

end NUMINAMATH_GPT_road_repair_completion_time_l326_32673


namespace NUMINAMATH_GPT_simplify_expression_l326_32648

theorem simplify_expression (q : ℤ) : 
  (((7 * q - 2) + 2 * q * 3) * 4 + (5 + 2 / 2) * (4 * q - 6)) = 76 * q - 44 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l326_32648


namespace NUMINAMATH_GPT_least_possible_value_of_smallest_integer_l326_32608

theorem least_possible_value_of_smallest_integer {A B C D : ℤ} 
  (h_diff: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_mean: (A + B + C + D) / 4 = 68)
  (h_largest: D = 90) :
  A ≥ 5 := 
sorry

end NUMINAMATH_GPT_least_possible_value_of_smallest_integer_l326_32608


namespace NUMINAMATH_GPT_positive_real_x_condition_l326_32686

-- We define the conditions:
variables (x : ℝ)
#check (1 - x^4)
#check (1 + x^4)

-- The main proof statement:
theorem positive_real_x_condition (h1 : x > 0) 
    (h2 : (Real.sqrt (Real.sqrt (1 - x^4)) + Real.sqrt (Real.sqrt (1 + x^4)) = 1)) :
    (x^8 = 35 / 36) :=
sorry

end NUMINAMATH_GPT_positive_real_x_condition_l326_32686


namespace NUMINAMATH_GPT_team_a_took_fewer_hours_l326_32634

/-- Two dogsled teams raced across a 300-mile course. 
Team A finished the course in fewer hours than Team E. 
Team A's average speed was 5 mph greater than Team E's, which was 20 mph. 
How many fewer hours did Team A take to finish the course compared to Team E? --/

theorem team_a_took_fewer_hours :
  let distance := 300
  let speed_e := 20
  let speed_a := speed_e + 5
  let time_e := distance / speed_e
  let time_a := distance / speed_a
  time_e - time_a = 3 := by
  sorry

end NUMINAMATH_GPT_team_a_took_fewer_hours_l326_32634


namespace NUMINAMATH_GPT_domain_f_l326_32645

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 9*x + 18)

theorem domain_f :
  (∀ x : ℝ, (x ≠ -6) ∧ (x ≠ -3) → ∃ y : ℝ, y = f x) ∧
  (∀ x : ℝ, x = -6 ∨ x = -3 → ¬(∃ y : ℝ, y = f x)) :=
sorry

end NUMINAMATH_GPT_domain_f_l326_32645


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l326_32668

theorem arithmetic_sequence_common_difference :
  ∃ d : ℤ, 
    (∀ n, n ≤ 6 → 23 + (n - 1) * d > 0) ∧ 
    (∀ n, n ≥ 7 → 23 + (n - 1) * d < 0) ∧
    d = -4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l326_32668


namespace NUMINAMATH_GPT_remaining_bread_after_three_days_l326_32681

namespace BreadProblem

def InitialBreadCount : ℕ := 200

def FirstDayConsumption (bread : ℕ) : ℕ := bread / 4
def SecondDayConsumption (remainingBreadAfterFirstDay : ℕ) : ℕ := 2 * remainingBreadAfterFirstDay / 5
def ThirdDayConsumption (remainingBreadAfterSecondDay : ℕ) : ℕ := remainingBreadAfterSecondDay / 2

theorem remaining_bread_after_three_days : 
  let initialBread := InitialBreadCount 
  let breadAfterFirstDay := initialBread - FirstDayConsumption initialBread 
  let breadAfterSecondDay := breadAfterFirstDay - SecondDayConsumption breadAfterFirstDay 
  let breadAfterThirdDay := breadAfterSecondDay - ThirdDayConsumption breadAfterSecondDay 
  breadAfterThirdDay = 45 := 
by
  let initialBread := InitialBreadCount 
  let breadAfterFirstDay := initialBread - FirstDayConsumption initialBread 
  let breadAfterSecondDay := breadAfterFirstDay - SecondDayConsumption breadAfterFirstDay 
  let breadAfterThirdDay := breadAfterSecondDay - ThirdDayConsumption breadAfterSecondDay 
  have : breadAfterThirdDay = 45 := sorry
  exact this

end BreadProblem

end NUMINAMATH_GPT_remaining_bread_after_three_days_l326_32681


namespace NUMINAMATH_GPT_determine_M_l326_32604

theorem determine_M (x y z M : ℚ) 
  (h1 : x + y + z = 120)
  (h2 : x - 10 = M)
  (h3 : y + 10 = M)
  (h4 : 10 * z = M) : 
  M = 400 / 7 := 
sorry

end NUMINAMATH_GPT_determine_M_l326_32604


namespace NUMINAMATH_GPT_maxine_purchases_l326_32666

theorem maxine_purchases (x y z : ℕ) (h1 : x + y + z = 40) (h2 : 50 * x + 400 * y + 500 * z = 10000) : x = 40 :=
by
  sorry

end NUMINAMATH_GPT_maxine_purchases_l326_32666


namespace NUMINAMATH_GPT_point_on_curve_iff_F_eq_zero_l326_32611

variable (F : ℝ → ℝ → ℝ)
variable (a b : ℝ)

theorem point_on_curve_iff_F_eq_zero :
  (F a b = 0) ↔ (∃ P : ℝ × ℝ, P = (a, b) ∧ F P.1 P.2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_point_on_curve_iff_F_eq_zero_l326_32611


namespace NUMINAMATH_GPT_quadratic_equation_correct_l326_32633

theorem quadratic_equation_correct :
    (∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) ↔ (∃ x : ℝ, x^2 = 5)) ∧
    ¬(∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) ↔ (∃ x y : ℝ, x + 2 * y = 0)) ∧
    ¬(∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) ↔ (∃ x : ℝ, x^2 + 1/x = 0)) ∧
    ¬(∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) ↔ (∃ x : ℝ, x^3 + x^2 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_correct_l326_32633


namespace NUMINAMATH_GPT_find_g_l326_32658

open Real

def even (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def odd (g : ℝ → ℝ) := ∀ x, g (-x) = -g x

theorem find_g 
  (f g : ℝ → ℝ) 
  (hf : even f) 
  (hg : odd g)
  (h : ∀ x, f x + g x = exp x) :
  ∀ x, g x = exp x - exp (-x) :=
by
  sorry

end NUMINAMATH_GPT_find_g_l326_32658


namespace NUMINAMATH_GPT_geometric_sum_s9_l326_32669

variable (S : ℕ → ℝ)

theorem geometric_sum_s9
  (h1 : S 3 = 7)
  (h2 : S 6 = 63) :
  S 9 = 511 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sum_s9_l326_32669


namespace NUMINAMATH_GPT_possible_values_of_a_l326_32679

theorem possible_values_of_a :
  ∃ (a : ℤ), (∀ (b c : ℤ), (x : ℤ) → (x - a) * (x - 8) + 4 = (x + b) * (x + c)) → (a = 6 ∨ a = 10) :=
sorry

end NUMINAMATH_GPT_possible_values_of_a_l326_32679


namespace NUMINAMATH_GPT_garden_length_l326_32661

theorem garden_length 
  (W : ℕ) (small_gate_width : ℕ) (large_gate_width : ℕ) (P : ℕ)
  (hW : W = 125)
  (h_small_gate : small_gate_width = 3)
  (h_large_gate : large_gate_width = 10)
  (hP : P = 687) :
  ∃ (L : ℕ), P = 2 * L + 2 * W - (small_gate_width + large_gate_width) ∧ L = 225 := by
  sorry

end NUMINAMATH_GPT_garden_length_l326_32661


namespace NUMINAMATH_GPT_range_of_m_l326_32664

theorem range_of_m (m : ℝ) 
  (p : m < 0) 
  (q : ∀ x : ℝ, x^2 + m * x + 1 > 0) : 
  -2 < m ∧ m < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l326_32664


namespace NUMINAMATH_GPT_lower_base_length_l326_32682

variable (A B C D E : Type)
variable (AD BD BE DE : ℝ)

-- Conditions of the problem
axiom hAD : AD = 12  -- upper base
axiom hBD : BD = 18  -- height
axiom hBE_DE : BE = 2 * DE  -- ratio BE = 2 * DE

-- Define the trapezoid with given lengths and conditions
def trapezoid_exists (A B C D : Type) (AD BD BE DE : ℝ) :=
  AD = 12 ∧ BD = 18 ∧ BE = 2 * DE

-- The length of BC to be proven
def BC : ℝ := 24

-- The theorem to be proven
theorem lower_base_length (h : trapezoid_exists A B C D AD BD BE DE) : BC = 2 * AD :=
by
  sorry

end NUMINAMATH_GPT_lower_base_length_l326_32682


namespace NUMINAMATH_GPT_seq_not_square_l326_32649

open Nat

theorem seq_not_square (n : ℕ) (r : ℕ) :
  (r = 11 ∨ r = 111 ∨ r = 1111 ∨ ∃ k : ℕ, r = k * 10^(n + 1) + 1) →
  (r % 4 = 3) →
  (¬ ∃ m : ℕ, r = m^2) :=
by
  intro h_seq h_mod
  intro h_square
  sorry

end NUMINAMATH_GPT_seq_not_square_l326_32649


namespace NUMINAMATH_GPT_no_integer_solutions_l326_32662

theorem no_integer_solutions (x y : ℤ) : ¬ (x^2 + 4 * x - 11 = 8 * y) := 
by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l326_32662


namespace NUMINAMATH_GPT_distance_lines_eq_2_l326_32646

-- Define the first line in standard form
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y + 3 = 0

-- Define the second line in standard form, established based on the parallel condition
def line2 (x y : ℝ) : Prop := 6 * x + 8 * y - 14 = 0

-- Define the condition for parallel lines which gives m
axiom parallel_lines_condition : ∀ (x y : ℝ), (line1 x y) → (line2 x y)

-- Define the distance between two parallel lines formula
noncomputable def distance_between_parallel_lines (a b c1 c2 : ℝ) : ℝ :=
  abs (c2 - c1) / (Real.sqrt (a ^ 2 + b ^ 2))

-- Prove the distance between the given lines is 2
theorem distance_lines_eq_2 : distance_between_parallel_lines 3 4 (-3) 7 = 2 :=
by
  -- Details of proof are omitted, but would show how to manipulate and calculate distances
  sorry

end NUMINAMATH_GPT_distance_lines_eq_2_l326_32646


namespace NUMINAMATH_GPT_range_of_f_l326_32699

noncomputable def f (x : ℝ) : ℝ :=
  if h : x < 1 then 3^(-x) else x^2

theorem range_of_f (x : ℝ) : (f x > 9) ↔ (x < -2 ∨ x > 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l326_32699


namespace NUMINAMATH_GPT_not_perfect_square_4_2021_l326_32656

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ x : ℕ, n = x * x

-- State the non-perfect square problem for the given choices
theorem not_perfect_square_4_2021 :
  ¬ is_perfect_square (4 ^ 2021) ∧
  is_perfect_square (1 ^ 2018) ∧
  is_perfect_square (6 ^ 2020) ∧
  is_perfect_square (5 ^ 2022) :=
by
  sorry

end NUMINAMATH_GPT_not_perfect_square_4_2021_l326_32656


namespace NUMINAMATH_GPT_find_b_l326_32627

theorem find_b (a u v w : ℝ) (b : ℝ)
  (h1 : ∀ x : ℝ, 12 * x^3 + 7 * a * x^2 + 6 * b * x + b = 0 → (x = u ∨ x = v ∨ x = w))
  (h2 : 0 < u ∧ 0 < v ∧ 0 < w)
  (h3 : u ≠ v ∧ v ≠ w ∧ u ≠ w)
  (h4 : Real.log u / Real.log 3 + Real.log v / Real.log 3 + Real.log w / Real.log 3 = 3):
  b = -324 := 
sorry

end NUMINAMATH_GPT_find_b_l326_32627


namespace NUMINAMATH_GPT_smallest_x_solution_l326_32620

theorem smallest_x_solution :
  ∃ x : ℝ, x * |x| + 3 * x = 5 * x + 2 ∧ (∀ y : ℝ, y * |y| + 3 * y = 5 * y + 2 → x ≤ y)
:=
sorry

end NUMINAMATH_GPT_smallest_x_solution_l326_32620


namespace NUMINAMATH_GPT_determine_M_l326_32660

theorem determine_M (M : ℕ) (h : 12 ^ 2 * 45 ^ 2 = 15 ^ 2 * M ^ 2) : M = 36 :=
by
  sorry

end NUMINAMATH_GPT_determine_M_l326_32660


namespace NUMINAMATH_GPT_correct_statement_l326_32670

-- We assume the existence of lines and planes with certain properties.
variables {Line : Type} {Plane : Type}
variables {m n : Line} {alpha beta gamma : Plane}

-- Definitions for perpendicular and parallel relations
def perpendicular (p1 p2 : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry

-- The theorem we aim to prove given the conditions
theorem correct_statement :
  line_perpendicular_to_plane m beta ∧ line_parallel_to_plane m alpha → perpendicular alpha beta :=
by sorry

end NUMINAMATH_GPT_correct_statement_l326_32670


namespace NUMINAMATH_GPT_option_b_correct_l326_32616

theorem option_b_correct (a b c : ℝ) (hc : c ≠ 0) (h : a * c^2 > b * c^2) : a > b :=
sorry

end NUMINAMATH_GPT_option_b_correct_l326_32616


namespace NUMINAMATH_GPT_investment_after_8_years_l326_32674

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

theorem investment_after_8_years :
  let P := 500
  let r := 0.03
  let n := 8
  let A := compound_interest P r n
  round A = 633 :=
by
  sorry

end NUMINAMATH_GPT_investment_after_8_years_l326_32674


namespace NUMINAMATH_GPT_parallel_resistance_example_l326_32609

theorem parallel_resistance_example :
  ∀ (R1 R2 : ℕ), R1 = 3 → R2 = 6 → 1 / (R : ℚ) = 1 / (R1 : ℚ) + 1 / (R2 : ℚ) → R = 2 := by
  intros R1 R2 hR1 hR2 h_formula
  -- Formulation of the resistance equations and assumptions
  sorry

end NUMINAMATH_GPT_parallel_resistance_example_l326_32609


namespace NUMINAMATH_GPT_problem_l326_32642

noncomputable def f (x : ℝ) (m : ℝ) (n : ℝ) (α1 : ℝ) (α2 : ℝ) :=
  m * Real.sin (Real.pi * x + α1) + n * Real.cos (Real.pi * x + α2)

variables (m n α1 α2 : ℝ) (h_m : m ≠ 0) (h_n : n ≠ 0) (h_α1 : α1 ≠ 0) (h_α2 : α2 ≠ 0)

theorem problem (h : f 2008 m n α1 α2 = 1) : f 2009 m n α1 α2 = -1 :=
  sorry

end NUMINAMATH_GPT_problem_l326_32642


namespace NUMINAMATH_GPT_connected_geometric_seq_a10_l326_32617

noncomputable def is_kth_order_geometric (a : ℕ → ℝ) (k : ℕ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + k) = q * a n

theorem connected_geometric_seq_a10 (a : ℕ → ℝ) 
  (h : is_kth_order_geometric a 3) 
  (a1 : a 1 = 1) 
  (a4 : a 4 = 2) : 
  a 10 = 8 :=
sorry

end NUMINAMATH_GPT_connected_geometric_seq_a10_l326_32617


namespace NUMINAMATH_GPT_robins_fraction_l326_32697

theorem robins_fraction (B R J : ℕ) (h1 : R + J = B)
  (h2 : 2/3 * (R : ℚ) + 1/3 * (J : ℚ) = 7/15 * (B : ℚ)) :
  (R : ℚ) / B = 2/5 :=
by
  sorry

end NUMINAMATH_GPT_robins_fraction_l326_32697


namespace NUMINAMATH_GPT_solution_set_of_inequality_l326_32647

theorem solution_set_of_inequality 
  (f : ℝ → ℝ)
  (f_odd : ∀ x : ℝ, f (-x) = -f x)
  (f_at_2 : f 2 = 0)
  (condition : ∀ x : ℝ, 0 < x → x * (deriv f x) + f x > 0) :
  {x : ℝ | x * f x > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | 2 < x} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l326_32647


namespace NUMINAMATH_GPT_company_budget_salaries_degrees_l326_32619

theorem company_budget_salaries_degrees :
  let transportation := 0.20
  let research_and_development := 0.09
  let utilities := 0.05
  let equipment := 0.04
  let supplies := 0.02
  let total_budget := 1.0
  let total_percentage := transportation + research_and_development + utilities + equipment + supplies
  let salaries_percentage := total_budget - total_percentage
  let total_degrees := 360.0
  let degrees_salaries := salaries_percentage * total_degrees
  degrees_salaries = 216 :=
by
  sorry

end NUMINAMATH_GPT_company_budget_salaries_degrees_l326_32619


namespace NUMINAMATH_GPT_simplify_exponent_fraction_l326_32657

theorem simplify_exponent_fraction : (3 ^ 2015 + 3 ^ 2013) / (3 ^ 2015 - 3 ^ 2013) = 5 / 4 := by
  sorry

end NUMINAMATH_GPT_simplify_exponent_fraction_l326_32657


namespace NUMINAMATH_GPT_colby_mangoes_harvested_60_l326_32644

variable (kg_left kg_each : ℕ)

def totalKgMangoes (x : ℕ) : Prop :=
  ∃ x : ℕ, 
  kg_left = (x - 20) / 2 ∧ 
  kg_each * kg_left = 160 ∧
  kg_each = 8

-- Problem Statement: Prove the total kilograms of mangoes harvested is 60 given the conditions.
theorem colby_mangoes_harvested_60 (x : ℕ) (h1 : x - 20 = 2 * kg_left)
(h2 : kg_each * kg_left = 160) (h3 : kg_each = 8) : x = 60 := by
  sorry

end NUMINAMATH_GPT_colby_mangoes_harvested_60_l326_32644


namespace NUMINAMATH_GPT_min_fence_length_l326_32602

theorem min_fence_length (x : ℝ) (h : x > 0) (A : x * (64 / x) = 64) : 2 * (x + 64 / x) ≥ 32 :=
by
  have t := (2 * (x + 64 / x)) 
  sorry -- Proof omitted, only statement provided as per instructions

end NUMINAMATH_GPT_min_fence_length_l326_32602


namespace NUMINAMATH_GPT_geometric_sequence_sum_l326_32672

noncomputable def aₙ (n : ℕ) : ℝ := (2 / 3) ^ (n - 1)

noncomputable def Sₙ (n : ℕ) : ℝ := 3 * (1 - (2 / 3) ^ n)

theorem geometric_sequence_sum (n : ℕ) : Sₙ n = 3 - 2 * aₙ n := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l326_32672


namespace NUMINAMATH_GPT_molecular_weight_CuCO3_8_moles_l326_32692

-- Definitions for atomic weights
def atomic_weight_Cu : ℝ := 63.55
def atomic_weight_C : ℝ := 12.01
def atomic_weight_O : ℝ := 16.00

-- Definition for the molecular formula of CuCO3
def molecular_weight_CuCO3 :=
  atomic_weight_Cu + atomic_weight_C + 3 * atomic_weight_O

-- Number of moles
def moles : ℝ := 8

-- Total weight of 8 moles of CuCO3
def total_weight := moles * molecular_weight_CuCO3

-- Proof statement
theorem molecular_weight_CuCO3_8_moles :
  total_weight = 988.48 :=
  by
  sorry

end NUMINAMATH_GPT_molecular_weight_CuCO3_8_moles_l326_32692


namespace NUMINAMATH_GPT_profit_percentage_l326_32694

theorem profit_percentage (C S : ℝ) (hC : C = 60) (hS : S = 75) : ((S - C) / C) * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_l326_32694


namespace NUMINAMATH_GPT_train_crosses_platform_in_20s_l326_32677

noncomputable def timeToCrossPlatform (train_length : ℝ) (platform_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := train_length + platform_length
  total_distance / train_speed_mps

theorem train_crosses_platform_in_20s :
  timeToCrossPlatform 120 213.36 60 = 20 :=
by
  sorry

end NUMINAMATH_GPT_train_crosses_platform_in_20s_l326_32677


namespace NUMINAMATH_GPT_mean_combined_scores_l326_32667

theorem mean_combined_scores (M A : ℝ) (m a : ℕ) 
  (hM : M = 88) 
  (hA : A = 72) 
  (hm : (m:ℝ) / (a:ℝ) = 2 / 3) :
  (88 * m + 72 * a) / (m + a) = 78 :=
by
  sorry

end NUMINAMATH_GPT_mean_combined_scores_l326_32667


namespace NUMINAMATH_GPT_simplify_expression_l326_32654

theorem simplify_expression : 
  (1 / (1 / (1 / 2)^0 + 1 / (1 / 2)^1 + 1 / (1 / 2)^2 + 1 / (1 / 2)^3)) = 1 / 15 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l326_32654


namespace NUMINAMATH_GPT_find_n_l326_32678

theorem find_n (n : ℕ) : (1 / (n + 1 : ℝ) + 2 / (n + 1 : ℝ) + (n + 1) / (n + 1 : ℝ) = 2) → (n = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_n_l326_32678


namespace NUMINAMATH_GPT_even_function_analytic_expression_l326_32684

noncomputable def f (x : ℝ) : ℝ := 
if x ≥ 0 then Real.log (x^2 - 2 * x + 2) 
else Real.log (x^2 + 2 * x + 2)

theorem even_function_analytic_expression (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_nonneg : ∀ x : ℝ, 0 ≤ x → f x = Real.log (x^2 - 2 * x + 2)) :
  ∀ x : ℝ, x < 0 → f x = Real.log (x^2 + 2 * x + 2) :=
by
  sorry

end NUMINAMATH_GPT_even_function_analytic_expression_l326_32684


namespace NUMINAMATH_GPT_monotonic_intervals_max_min_values_l326_32698

noncomputable def f : ℝ → ℝ := λ x => (1 / 3) * x^3 + x^2 - 3 * x + 1

theorem monotonic_intervals :
  (∀ x, x < -3 → deriv f x > 0) ∧
  (∀ x, x > 1 → deriv f x > 0) ∧
  (∀ x, -3 < x ∧ x < 1 → deriv f x < 0) :=
by
  sorry

theorem max_min_values :
  f 2 = 5 / 3 ∧ f 1 = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_monotonic_intervals_max_min_values_l326_32698


namespace NUMINAMATH_GPT_Mitzi_leftover_money_l326_32638

def A := 75
def T := 30
def F := 13
def S := 23
def R := A - (T + F + S)

theorem Mitzi_leftover_money : R = 9 := by
  sorry

end NUMINAMATH_GPT_Mitzi_leftover_money_l326_32638


namespace NUMINAMATH_GPT_yvette_final_bill_l326_32671

def cost_alicia : ℝ := 7.50
def cost_brant : ℝ := 10.00
def cost_josh : ℝ := 8.50
def cost_yvette : ℝ := 9.00
def tip_rate : ℝ := 0.20

def total_cost := cost_alicia + cost_brant + cost_josh + cost_yvette
def tip := tip_rate * total_cost
def final_bill := total_cost + tip

theorem yvette_final_bill :
  final_bill = 42.00 :=
  sorry

end NUMINAMATH_GPT_yvette_final_bill_l326_32671


namespace NUMINAMATH_GPT_initial_oranges_count_l326_32614

theorem initial_oranges_count
  (initial_apples : ℕ := 50)
  (apple_cost : ℝ := 0.80)
  (orange_cost : ℝ := 0.50)
  (total_earnings : ℝ := 49)
  (remaining_apples : ℕ := 10)
  (remaining_oranges : ℕ := 6)
  : initial_oranges = 40 := 
by
  sorry

end NUMINAMATH_GPT_initial_oranges_count_l326_32614


namespace NUMINAMATH_GPT_range_of_a_l326_32635

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ x1^3 - 3*x1 + a = 0 ∧ x2^3 - 3*x2 + a = 0 ∧ x3^3 - 3*x3 + a = 0) 
  ↔ -2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l326_32635


namespace NUMINAMATH_GPT_range_of_a_l326_32693

def increasing {α : Type*} [Preorder α] (f : α → α) := ∀ x y, x ≤ y → f x ≤ f y

theorem range_of_a
  (f : ℝ → ℝ)
  (increasing_f : increasing f)
  (h_domain : ∀ x, 1 ≤ x ∧ x ≤ 5 → (f x = f x))
  (h_ineq : ∀ a, 1 ≤ a + 1 ∧ a + 1 ≤ 5 ∧ 1 ≤ 2 * a - 1 ∧ 2 * a - 1 ≤ 5 ∧ f (a + 1) < f (2 * a - 1)) :
  (2 : ℝ) < a ∧ a ≤ (3 : ℝ) := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l326_32693


namespace NUMINAMATH_GPT_Shannon_ratio_2_to_1_l326_32636

structure IceCreamCarton :=
  (scoops : ℕ)

structure PersonWants :=
  (vanilla : ℕ)
  (chocolate : ℕ)
  (strawberry : ℕ)

noncomputable def total_scoops_served (ethan_wants lucas_wants danny_wants connor_wants olivia_wants : PersonWants) : ℕ :=
  ethan_wants.vanilla + ethan_wants.chocolate +
  lucas_wants.chocolate +
  danny_wants.chocolate +
  connor_wants.chocolate +
  olivia_wants.vanilla + olivia_wants.strawberry

theorem Shannon_ratio_2_to_1 
    (cartons : List IceCreamCarton)
    (ethan_wants lucas_wants danny_wants connor_wants olivia_wants : PersonWants)
    (scoops_left : ℕ) : 
    -- Conditions
    (∀ carton ∈ cartons, carton.scoops = 10) →
    (cartons.length = 3) →
    (ethan_wants.vanilla = 1 ∧ ethan_wants.chocolate = 1) →
    (lucas_wants.chocolate = 2) →
    (danny_wants.chocolate = 2) →
    (connor_wants.chocolate = 2) →
    (olivia_wants.vanilla = 1 ∧ olivia_wants.strawberry = 1) →
    (scoops_left = 16) →
    -- To Prove
    4 / olivia_wants.vanilla + olivia_wants.strawberry = 2 := 
sorry

end NUMINAMATH_GPT_Shannon_ratio_2_to_1_l326_32636


namespace NUMINAMATH_GPT_last_three_digits_of_8_pow_1000_l326_32650

theorem last_three_digits_of_8_pow_1000 (h : 8 ^ 125 ≡ 2 [MOD 1250]) : (8 ^ 1000) % 1000 = 256 :=
by
  sorry

end NUMINAMATH_GPT_last_three_digits_of_8_pow_1000_l326_32650


namespace NUMINAMATH_GPT_find_value_of_A_l326_32603

theorem find_value_of_A (A B : ℕ) (h_ratio : A * 5 = 3 * B) (h_diff : B - A = 12) : A = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_A_l326_32603


namespace NUMINAMATH_GPT_max_number_ahn_can_get_l326_32601

theorem max_number_ahn_can_get :
  ∃ n : ℤ, (10 ≤ n ∧ n ≤ 99) ∧ ∀ m : ℤ, (10 ≤ m ∧ m ≤ 99) → (3 * (300 - n) ≥ 3 * (300 - m)) ∧ 3 * (300 - n) = 870 :=
by sorry

end NUMINAMATH_GPT_max_number_ahn_can_get_l326_32601


namespace NUMINAMATH_GPT_gas_consumption_100_l326_32612

noncomputable def gas_consumption (x : ℝ) : Prop :=
  60 * 1 + (x - 60) * 1.5 = 1.2 * x

theorem gas_consumption_100 (x : ℝ) (h : gas_consumption x) : x = 100 := 
by {
  sorry
}

end NUMINAMATH_GPT_gas_consumption_100_l326_32612


namespace NUMINAMATH_GPT_number_of_lattice_points_in_triangle_l326_32690

theorem number_of_lattice_points_in_triangle (L : ℕ) (hL : L > 1) :
  ∃ I, I = (L^2 - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_lattice_points_in_triangle_l326_32690


namespace NUMINAMATH_GPT_general_term_a_sum_Tn_l326_32626

section sequence_problem

variables {n : ℕ} (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)

-- Problem 1: General term formula for {a_n}
axiom Sn_def : ∀ n, S n = 1/4 * (a n + 1)^2
axiom a1_def : a 1 = 1
axiom an_diff : ∀ n, a (n+1) - a n = 2

theorem general_term_a : a n = 2 * n - 1 := sorry

-- Problem 2: Sum of the first n terms of sequence {b_n}
axiom an_formula : ∀ n, a n = 2 * n - 1
axiom bn_def : ∀ n, b n = 1 / (a n * a (n+1))

theorem sum_Tn : T n = n / (2 * n + 1) := sorry

end sequence_problem

end NUMINAMATH_GPT_general_term_a_sum_Tn_l326_32626


namespace NUMINAMATH_GPT_percent_decrease_to_original_price_l326_32643

variable (x : ℝ) (p : ℝ)

def new_price (x : ℝ) : ℝ := 1.35 * x

theorem percent_decrease_to_original_price :
  ∀ (x : ℝ), x ≠ 0 → (1 - (7 / 27)) * (new_price x) = x := 
sorry

end NUMINAMATH_GPT_percent_decrease_to_original_price_l326_32643


namespace NUMINAMATH_GPT_correct_discount_rate_l326_32695

def purchase_price : ℝ := 200
def marked_price : ℝ := 300
def desired_profit_percentage : ℝ := 0.20

theorem correct_discount_rate :
  ∃ (x : ℝ), 300 * x = 240 ∧ x = 0.80 := 
by
  sorry

end NUMINAMATH_GPT_correct_discount_rate_l326_32695


namespace NUMINAMATH_GPT_rolls_combinations_l326_32622

theorem rolls_combinations (x1 x2 x3 : ℕ) (h1 : x1 + x2 + x3 = 2) : 
  (Nat.choose (2 + 3 - 1) (3 - 1) = 6) :=
by
  sorry

end NUMINAMATH_GPT_rolls_combinations_l326_32622


namespace NUMINAMATH_GPT_baker_batches_chocolate_chip_l326_32637

noncomputable def number_of_batches (total_cookies : ℕ) (oatmeal_cookies : ℕ) (cookies_per_batch : ℕ) : ℕ :=
  (total_cookies - oatmeal_cookies) / cookies_per_batch

theorem baker_batches_chocolate_chip (total_cookies oatmeal_cookies cookies_per_batch : ℕ) 
  (h_total : total_cookies = 10) 
  (h_oatmeal : oatmeal_cookies = 4) 
  (h_batch : cookies_per_batch = 3) : 
  number_of_batches total_cookies oatmeal_cookies cookies_per_batch = 2 :=
by
  sorry

end NUMINAMATH_GPT_baker_batches_chocolate_chip_l326_32637


namespace NUMINAMATH_GPT_cone_volume_l326_32675

-- Define the condition
def cylinder_volume : ℝ := 30

-- Define the statement that needs to be proven
theorem cone_volume (h_cylinder_volume : cylinder_volume = 30) : cylinder_volume / 3 = 10 := 
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_cone_volume_l326_32675


namespace NUMINAMATH_GPT_olivia_did_not_sell_4_bars_l326_32621

-- Define the constants and conditions
def price_per_bar : ℕ := 3
def total_bars : ℕ := 7
def money_made : ℕ := 9

-- Calculate the number of bars sold
def bars_sold : ℕ := money_made / price_per_bar

-- Calculate the number of bars not sold
def bars_not_sold : ℕ := total_bars - bars_sold

-- Theorem to prove the answer
theorem olivia_did_not_sell_4_bars : bars_not_sold = 4 := 
by 
  sorry

end NUMINAMATH_GPT_olivia_did_not_sell_4_bars_l326_32621


namespace NUMINAMATH_GPT_division_of_powers_of_ten_l326_32689

theorem division_of_powers_of_ten :
  (10 ^ 0.7 * 10 ^ 0.4) / (10 ^ 0.2 * 10 ^ 0.6 * 10 ^ 0.3) = 1 := by
  sorry

end NUMINAMATH_GPT_division_of_powers_of_ten_l326_32689


namespace NUMINAMATH_GPT_math_problem_l326_32663

def f (x : ℝ) : ℝ := x^2 + 3
def g (x : ℝ) : ℝ := 2 * x + 5

theorem math_problem : f (g 4) - g (f 4) = 129 := by
  sorry

end NUMINAMATH_GPT_math_problem_l326_32663


namespace NUMINAMATH_GPT_book_area_correct_l326_32652

def book_length : ℝ := 5
def book_width : ℝ := 10
def book_area (length : ℝ) (width : ℝ) : ℝ := length * width

theorem book_area_correct :
  book_area book_length book_width = 50 :=
by
  sorry

end NUMINAMATH_GPT_book_area_correct_l326_32652


namespace NUMINAMATH_GPT_sum_powers_mod_5_l326_32629

theorem sum_powers_mod_5 (n : ℕ) (h : ¬ (n % 4 = 0)) : 
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_powers_mod_5_l326_32629


namespace NUMINAMATH_GPT_smallest_c_for_polynomial_l326_32676

theorem smallest_c_for_polynomial :
  ∃ r1 r2 r3 : ℕ, (r1 * r2 * r3 = 2310) ∧ (r1 + r2 + r3 = 52) := sorry

end NUMINAMATH_GPT_smallest_c_for_polynomial_l326_32676


namespace NUMINAMATH_GPT_solve_D_l326_32683

-- Define the digits represented by each letter
variable (P M T D E : ℕ)

-- Each letter represents a different digit (0-9) and should be distinct
axiom distinct_digits : (P ≠ M) ∧ (P ≠ T) ∧ (P ≠ D) ∧ (P ≠ E) ∧ 
                        (M ≠ T) ∧ (M ≠ D) ∧ (M ≠ E) ∧ 
                        (T ≠ D) ∧ (T ≠ E) ∧ 
                        (D ≠ E)

-- Each letter is a digit from 0 to 9
axiom digit_range : 0 ≤ P ∧ P ≤ 9 ∧ 0 ≤ M ∧ M ≤ 9 ∧ 
                    0 ≤ T ∧ T ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9 ∧ 
                    0 ≤ E ∧ E ≤ 9

-- Each column sums to the digit below it, considering carry overs from right to left
axiom column1 : T + T + E = E ∨ T + T + E = 10 + E
axiom column2 : E + D + T + (if T + T + E = 10 + E then 1 else 0) = P
axiom column3 : P + M + (if E + D + T + (if T + T + E = 10 + E then 1 else 0) = 10 + P then 1 else 0) = M

-- Prove that D = 4 given the above conditions
theorem solve_D : D = 4 :=
by sorry

end NUMINAMATH_GPT_solve_D_l326_32683


namespace NUMINAMATH_GPT_initial_points_count_l326_32600

theorem initial_points_count (k : ℕ) (h : (4 * k - 3) = 101): k = 26 :=
by 
  sorry

end NUMINAMATH_GPT_initial_points_count_l326_32600


namespace NUMINAMATH_GPT_ellipse_semi_minor_axis_is_2_sqrt_3_l326_32628

/-- 
  Given an ellipse with the center at (2, -1), 
  one focus at (2, -3), and one endpoint of a semi-major axis at (2, 3), 
  we prove that the semi-minor axis is 2√3.
-/
theorem ellipse_semi_minor_axis_is_2_sqrt_3 :
  let center := (2, -1)
  let focus := (2, -3)
  let endpoint := (2, 3)
  let c := Real.sqrt ((2 - 2)^2 + (-3 + 1)^2)
  let a := Real.sqrt ((2 - 2)^2 + (3 + 1)^2)
  let b2 := a^2 - c^2
  let b := Real.sqrt b2
  c = 2 ∧ a = 4 ∧ b = 2 * Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_ellipse_semi_minor_axis_is_2_sqrt_3_l326_32628


namespace NUMINAMATH_GPT_range_of_quadratic_function_l326_32691

theorem range_of_quadratic_function : 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → 0 ≤ x^2 - 4 * x + 3 ∧ x^2 - 4 * x + 3 ≤ 8 :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_range_of_quadratic_function_l326_32691
