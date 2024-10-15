import Mathlib

namespace NUMINAMATH_GPT_find_fraction_value_l1614_161440

theorem find_fraction_value (m n : ℝ) (h : 1/m - 1/n = 6) : (m * n) / (m - n) = -1/6 :=
sorry

end NUMINAMATH_GPT_find_fraction_value_l1614_161440


namespace NUMINAMATH_GPT_no_unhappy_days_l1614_161412

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end NUMINAMATH_GPT_no_unhappy_days_l1614_161412


namespace NUMINAMATH_GPT_option_c_equals_9_l1614_161421

theorem option_c_equals_9 : (3 * 3 - 3 + 3) = 9 :=
by
  sorry

end NUMINAMATH_GPT_option_c_equals_9_l1614_161421


namespace NUMINAMATH_GPT_sum_of_repeating_decimals_l1614_161474

-- Definitions of repeating decimals x and y
def x : ℚ := 25 / 99
def y : ℚ := 87 / 99

-- The assertion that the sum of these repeating decimals is equal to 112/99 as a fraction
theorem sum_of_repeating_decimals: x + y = 112 / 99 := by
  sorry

end NUMINAMATH_GPT_sum_of_repeating_decimals_l1614_161474


namespace NUMINAMATH_GPT_proof_x_square_ab_a_square_l1614_161470

variable {x b a : ℝ}

/-- Given that x < b < a < 0 where x, b, and a are real numbers, we need to prove x^2 > ab > a^2. -/
theorem proof_x_square_ab_a_square (hx : x < b) (hb : b < a) (ha : a < 0) :
  x^2 > ab ∧ ab > a^2 := 
by
  sorry

end NUMINAMATH_GPT_proof_x_square_ab_a_square_l1614_161470


namespace NUMINAMATH_GPT_g_two_eq_one_l1614_161491

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x - y) = g x * g y
axiom g_nonzero (x : ℝ) : g x ≠ 0

theorem g_two_eq_one : g 2 = 1 := by
  sorry

end NUMINAMATH_GPT_g_two_eq_one_l1614_161491


namespace NUMINAMATH_GPT_ways_to_stand_l1614_161403

-- Definitions derived from conditions
def num_steps : ℕ := 7
def max_people_per_step : ℕ := 2

-- Define a function to count the number of different ways
noncomputable def count_ways : ℕ :=
  336

-- The statement to be proven in Lean 4
theorem ways_to_stand : count_ways = 336 :=
  sorry

end NUMINAMATH_GPT_ways_to_stand_l1614_161403


namespace NUMINAMATH_GPT_base_conversion_l1614_161476

theorem base_conversion (C D : ℕ) (h₁ : 0 ≤ C ∧ C < 8) (h₂ : 0 ≤ D ∧ D < 5) (h₃ : 7 * C = 4 * D) :
  8 * C + D = 0 := by
  sorry

end NUMINAMATH_GPT_base_conversion_l1614_161476


namespace NUMINAMATH_GPT_ratio_of_people_on_buses_l1614_161467

theorem ratio_of_people_on_buses (P_2 P_3 P_4 : ℕ) 
  (h1 : P_1 = 12) 
  (h2 : P_3 = P_2 - 6) 
  (h3 : P_4 = P_1 + 9) 
  (h4 : P_1 + P_2 + P_3 + P_4 = 75) : 
  P_2 / P_1 = 2 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_people_on_buses_l1614_161467


namespace NUMINAMATH_GPT_provider_choices_count_l1614_161465

theorem provider_choices_count :
  let num_providers := 25
  let num_s_providers := 6
  let remaining_providers_after_laura := num_providers - 1
  let remaining_providers_after_brother := remaining_providers_after_laura - 1

  (num_providers * num_s_providers * remaining_providers_after_laura * remaining_providers_after_brother) = 75900 :=
by
  sorry

end NUMINAMATH_GPT_provider_choices_count_l1614_161465


namespace NUMINAMATH_GPT_Jeff_total_laps_l1614_161464

theorem Jeff_total_laps (laps_saturday : ℕ) (laps_sunday_morning : ℕ) (laps_remaining : ℕ)
  (h1 : laps_saturday = 27) (h2 : laps_sunday_morning = 15) (h3 : laps_remaining = 56) :
  (laps_saturday + laps_sunday_morning + laps_remaining) = 98 := 
by
  sorry

end NUMINAMATH_GPT_Jeff_total_laps_l1614_161464


namespace NUMINAMATH_GPT_angle_division_l1614_161492

theorem angle_division (α : ℝ) (n : ℕ) (θ : ℝ) (h : α = 78) (hn : n = 26) (ht : θ = 3) :
  α / n = θ :=
by
  sorry

end NUMINAMATH_GPT_angle_division_l1614_161492


namespace NUMINAMATH_GPT_maximum_modest_number_l1614_161447

-- Definitions and Conditions
def is_modest (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
  5 * a = b + c + d ∧
  d % 2 = 0

def G (a b c d : ℕ) : ℕ :=
  (1000 * a + 100 * b + 10 * c + d - (1000 * c + 100 * d + 10 * a + b)) / 99

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def is_divisible_by_3 (abc : ℕ) : Prop :=
  abc % 3 = 0

-- Theorem statement
theorem maximum_modest_number :
  ∃ a b c d : ℕ, is_modest a b c d ∧ is_divisible_by_11 (G a b c d) ∧ is_divisible_by_3 (100 * a + 10 * b + c) ∧ 
  (1000 * a + 100 * b + 10 * c + d) = 3816 := 
sorry

end NUMINAMATH_GPT_maximum_modest_number_l1614_161447


namespace NUMINAMATH_GPT_ratio_of_spinsters_to_cats_l1614_161408

-- Definitions for the conditions given:
def S : ℕ := 12 -- 12 spinsters
def C : ℕ := S + 42 -- 42 more cats than spinsters
def ratio (a b : ℕ) : ℚ := a / b -- Ratio definition

-- The theorem stating the required equivalence:
theorem ratio_of_spinsters_to_cats :
  ratio S C = 2 / 9 :=
by
  -- This proof has been omitted for the purpose of this exercise.
  sorry

end NUMINAMATH_GPT_ratio_of_spinsters_to_cats_l1614_161408


namespace NUMINAMATH_GPT_coordinates_at_5PM_l1614_161445

noncomputable def particle_coords_at_5PM : ℝ × ℝ :=
  let t1 : ℝ := 7  -- 7 AM
  let t2 : ℝ := 9  -- 9 AM
  let t3 : ℝ := 17  -- 5 PM in 24-hour format
  let coord1 : ℝ × ℝ := (1, 2)
  let coord2 : ℝ × ℝ := (3, -2)
  let dx : ℝ := (coord2.1 - coord1.1) / (t2 - t1)
  let dy : ℝ := (coord2.2 - coord1.2) / (t2 - t1)
  (coord2.1 + dx * (t3 - t2), coord2.2 + dy * (t3 - t2))

theorem coordinates_at_5PM
  (t1 t2 t3 : ℝ)
  (coord1 coord2 : ℝ × ℝ)
  (h_t1 : t1 = 7)
  (h_t2 : t2 = 9)
  (h_t3 : t3 = 17)
  (h_coord1 : coord1 = (1, 2))
  (h_coord2 : coord2 = (3, -2))
  (h_dx : (coord2.1 - coord1.1) / (t2 - t1) = 1)
  (h_dy : (coord2.2 - coord1.2) / (t2 - t1) = -2)
  : particle_coords_at_5PM = (11, -18) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_at_5PM_l1614_161445


namespace NUMINAMATH_GPT_boys_to_admit_or_expel_l1614_161472

-- Definitions from the conditions
def total_students : ℕ := 500

def girls_percent (x : ℕ) : ℕ := (x * total_students) / 100

-- Definition of the calculation under the new policy
def required_boys : ℕ := (total_students * 3) / 5

-- Main statement we need to prove
theorem boys_to_admit_or_expel (x : ℕ) (htotal : x + girls_percent x = total_students) :
  required_boys - x = 217 := by
  sorry

end NUMINAMATH_GPT_boys_to_admit_or_expel_l1614_161472


namespace NUMINAMATH_GPT_cos_identity_l1614_161487

noncomputable def f (x : ℝ) : ℝ :=
  let a := (2 * Real.cos x, (Real.sqrt 3) / 2)
  let b := (Real.sin (x - Real.pi / 3), 1)
  a.1 * b.1 + a.2 * b.2

theorem cos_identity (x0 : ℝ) (hx0 : x0 ∈ Set.Icc (5 * Real.pi / 12) (2 * Real.pi / 3))
  (hf : f x0 = 4 / 5) :
  Real.cos (2 * x0 - Real.pi / 12) = -7 * Real.sqrt 2 / 10 :=
sorry

end NUMINAMATH_GPT_cos_identity_l1614_161487


namespace NUMINAMATH_GPT_log_expression_value_l1614_161479

theorem log_expression_value (lg : ℕ → ℤ) :
  (lg 4 = 2 * lg 2) →
  (lg 20 = lg 4 + lg 5) →
  lg 4 + lg 5 * lg 20 + (lg 5)^2 = 2 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_log_expression_value_l1614_161479


namespace NUMINAMATH_GPT_fraction_simplification_l1614_161414

theorem fraction_simplification :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1614_161414


namespace NUMINAMATH_GPT_intersection_A_B_l1614_161443

-- Definitions of sets A and B
def A := { x : ℝ | x ≥ -1 }
def B := { y : ℝ | y < 1 }

-- Statement to prove the intersection of A and B
theorem intersection_A_B : A ∩ B = { x : ℝ | -1 ≤ x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1614_161443


namespace NUMINAMATH_GPT_find_K_l1614_161462

theorem find_K (surface_area_cube : ℝ) (volume_sphere : ℝ) (r : ℝ) (K : ℝ) 
  (cube_side_length : ℝ) (surface_area_sphere_eq : surface_area_cube = 4 * Real.pi * (r ^ 2))
  (volume_sphere_eq : volume_sphere = (4 / 3) * Real.pi * (r ^ 3)) 
  (surface_area_cube_eq : surface_area_cube = 6 * (cube_side_length ^ 2)) 
  (volume_sphere_form : volume_sphere = (K * Real.sqrt 6) / Real.sqrt Real.pi) :
  K = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_K_l1614_161462


namespace NUMINAMATH_GPT_intersection_count_l1614_161459

def M (x y : ℝ) : Prop := y^2 = x - 1
def N (x y m : ℝ) : Prop := y = 2 * x - 2 * m^2 + m - 2

theorem intersection_count (m x y : ℝ) :
  (M x y ∧ N x y m) → (∃ n : ℕ, n = 1 ∨ n = 2) :=
sorry

end NUMINAMATH_GPT_intersection_count_l1614_161459


namespace NUMINAMATH_GPT_maximize_profit_l1614_161439

noncomputable def profit (x : ℝ) : ℝ :=
  (x - 8) * (100 - 10 * (x - 10))

theorem maximize_profit :
  let max_price := 14
  let max_profit := 360
  (∀ x > 10, profit x ≤ profit max_price) ∧ profit max_price = max_profit :=
by
  let max_price := 14
  let max_profit := 360
  sorry

end NUMINAMATH_GPT_maximize_profit_l1614_161439


namespace NUMINAMATH_GPT_delores_initial_money_l1614_161471

def computer_price : ℕ := 400
def printer_price : ℕ := 40
def headphones_price : ℕ := 60
def discount_percentage : ℕ := 10
def left_money : ℕ := 10

theorem delores_initial_money :
  ∃ initial_money : ℕ,
    initial_money = printer_price + headphones_price + (computer_price - (discount_percentage * computer_price / 100)) + left_money :=
  sorry

end NUMINAMATH_GPT_delores_initial_money_l1614_161471


namespace NUMINAMATH_GPT_high_quality_chip_prob_l1614_161493

variable (chipsA chipsB chipsC : ℕ)
variable (qualityA qualityB qualityC : ℝ)
variable (totalChips : ℕ)

noncomputable def probability_of_high_quality_chip (chipsA chipsB chipsC : ℕ) (qualityA qualityB qualityC : ℝ) (totalChips : ℕ) : ℝ :=
  (chipsA / totalChips) * qualityA + (chipsB / totalChips) * qualityB + (chipsC / totalChips) * qualityC

theorem high_quality_chip_prob :
  let chipsA := 5
  let chipsB := 10
  let chipsC := 10
  let qualityA := 0.8
  let qualityB := 0.8
  let qualityC := 0.7
  let totalChips := 25
  probability_of_high_quality_chip chipsA chipsB chipsC qualityA qualityB qualityC totalChips = 0.76 :=
by
  sorry

end NUMINAMATH_GPT_high_quality_chip_prob_l1614_161493


namespace NUMINAMATH_GPT_council_revote_l1614_161413

theorem council_revote (x y x' y' m : ℝ) (h1 : x + y = 500)
    (h2 : y - x = m) (h3 : x' - y' = 1.5 * m) (h4 : x' + y' = 500) (h5 : x' = 11 / 10 * y) :
    x' - x = 156.25 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_council_revote_l1614_161413


namespace NUMINAMATH_GPT_volume_tetrahedron_OABC_correct_l1614_161499

noncomputable def volume_tetrahedron_OABC : ℝ :=
  let a := Real.sqrt 33
  let b := 4
  let c := 4 * Real.sqrt 3
  (1 / 6) * a * b * c

theorem volume_tetrahedron_OABC_correct :
  let a := Real.sqrt 33
  let b := 4
  let c := 4 * Real.sqrt 3
  let volume := (1 / 6) * a * b * c
  volume = 8 * Real.sqrt 99 / 3 :=
by
  sorry

end NUMINAMATH_GPT_volume_tetrahedron_OABC_correct_l1614_161499


namespace NUMINAMATH_GPT_train_length_l1614_161437

def speed_kmph := 72   -- Speed in kilometers per hour
def time_sec := 14     -- Time in seconds

/-- Function to convert speed from km/hr to m/s -/
def convert_speed (speed : ℕ) : ℕ :=
  speed * 1000 / 3600

/-- Function to calculate distance given speed and time -/
def calculate_distance (speed : ℕ) (time : ℕ) : ℕ :=
  speed * time

theorem train_length :
  calculate_distance (convert_speed speed_kmph) time_sec = 280 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l1614_161437


namespace NUMINAMATH_GPT_seven_digit_palindromes_count_l1614_161435

theorem seven_digit_palindromes_count : 
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  (a_choices * b_choices * c_choices * d_choices) = 9000 := by
  sorry

end NUMINAMATH_GPT_seven_digit_palindromes_count_l1614_161435


namespace NUMINAMATH_GPT_no_solution_when_k_eq_7_l1614_161458

theorem no_solution_when_k_eq_7 
  (x : ℝ) (h₁ : x ≠ 4) (h₂ : x ≠ 8) : 
  (∀ k : ℝ, (x - 3) / (x - 4) = (x - k) / (x - 8) → False) ↔ k = 7 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_when_k_eq_7_l1614_161458


namespace NUMINAMATH_GPT_simplify_expression_l1614_161400

theorem simplify_expression (x : ℝ) : 120 * x - 72 * x + 15 * x - 9 * x = 54 * x := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1614_161400


namespace NUMINAMATH_GPT_g_value_at_2_l1614_161410

def g (x : ℝ) (d : ℝ) : ℝ := 3 * x^5 - 2 * x^4 + d * x - 8

theorem g_value_at_2 (d : ℝ) (h : g (-2) d = 4) : g 2 d = -84 := by
  sorry

end NUMINAMATH_GPT_g_value_at_2_l1614_161410


namespace NUMINAMATH_GPT_find_value_added_l1614_161485

open Classical

variable (n : ℕ) (avg_initial avg_final : ℝ)

-- Initial conditions
axiom avg_then_sum (n : ℕ) (avg : ℝ) : n * avg = 600

axiom avg_after_addition (n : ℕ) (avg : ℝ) : n * avg = 825

theorem find_value_added (n : ℕ) (avg_initial avg_final : ℝ) (h1 : n * avg_initial = 600) (h2 : n * avg_final = 825) :
  avg_final - avg_initial = 15 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_value_added_l1614_161485


namespace NUMINAMATH_GPT_solveEquation_l1614_161431

noncomputable def findNonZeroSolution (z : ℝ) : Prop :=
  (5 * z) ^ 10 = (20 * z) ^ 5 ∧ z ≠ 0

theorem solveEquation : ∃ z : ℝ, findNonZeroSolution z ∧ z = 4 / 5 := by
  exists 4 / 5
  simp [findNonZeroSolution]
  sorry

end NUMINAMATH_GPT_solveEquation_l1614_161431


namespace NUMINAMATH_GPT_common_difference_l1614_161473

theorem common_difference (a : ℕ → ℝ) (d : ℝ) (h_seq : ∀ n, a n = 1 + (n - 1) * d) 
  (h_geom : (a 3) ^ 2 = (a 1) * (a 13)) (h_ne_zero: d ≠ 0) : d = 2 :=
by
  sorry

end NUMINAMATH_GPT_common_difference_l1614_161473


namespace NUMINAMATH_GPT_final_statue_weight_l1614_161406

-- Define the initial weight of the statue
def initial_weight : ℝ := 250

-- Define the percentage of weight remaining after each week
def remaining_after_week1 (w : ℝ) : ℝ := 0.70 * w
def remaining_after_week2 (w : ℝ) : ℝ := 0.80 * w
def remaining_after_week3 (w : ℝ) : ℝ := 0.75 * w

-- Define the final weight of the statue after three weeks
def final_weight : ℝ := 
  remaining_after_week3 (remaining_after_week2 (remaining_after_week1 initial_weight))

-- Prove the weight of the final statue is 105 kg
theorem final_statue_weight : final_weight = 105 := 
  by
    sorry

end NUMINAMATH_GPT_final_statue_weight_l1614_161406


namespace NUMINAMATH_GPT_evaluate_expression_l1614_161436

theorem evaluate_expression : 3 - 5 * (2^3 + 3) * 2 = -107 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1614_161436


namespace NUMINAMATH_GPT_table_tennis_total_rounds_l1614_161416

-- Mathematical equivalent proof problem in Lean 4 statement
theorem table_tennis_total_rounds
  (A_played : ℕ) (B_played : ℕ) (C_referee : ℕ) (total_rounds : ℕ)
  (hA : A_played = 5) (hB : B_played = 4) (hC : C_referee = 2) :
  total_rounds = 7 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_table_tennis_total_rounds_l1614_161416


namespace NUMINAMATH_GPT_valid_three_digit_numbers_no_seven_nine_l1614_161448

noncomputable def count_valid_three_digit_numbers : Nat := 
  let hundredsChoices := 7
  let tensAndUnitsChoices := 8
  hundredsChoices * tensAndUnitsChoices * tensAndUnitsChoices

theorem valid_three_digit_numbers_no_seven_nine : 
  count_valid_three_digit_numbers = 448 := by
  sorry

end NUMINAMATH_GPT_valid_three_digit_numbers_no_seven_nine_l1614_161448


namespace NUMINAMATH_GPT_total_number_of_parts_l1614_161442

-- Identify all conditions in the problem: sample size and probability
def sample_size : ℕ := 30
def probability : ℝ := 0.25

-- Statement of the proof problem: The total number of parts N is 120 given the conditions
theorem total_number_of_parts (N : ℕ) (h : (sample_size : ℝ) / N = probability) : N = 120 :=
sorry

end NUMINAMATH_GPT_total_number_of_parts_l1614_161442


namespace NUMINAMATH_GPT_slope_angle_of_y_eq_0_l1614_161446

theorem slope_angle_of_y_eq_0  :
  ∀ (α : ℝ), (∀ (y x : ℝ), y = 0) → α = 0 :=
by
  intros α h
  sorry

end NUMINAMATH_GPT_slope_angle_of_y_eq_0_l1614_161446


namespace NUMINAMATH_GPT_laundry_loads_l1614_161409

-- Definitions based on conditions
def num_families : ℕ := 3
def people_per_family : ℕ := 4
def num_people : ℕ := num_families * people_per_family

def days : ℕ := 7
def towels_per_person_per_day : ℕ := 1
def total_towels : ℕ := num_people * days * towels_per_person_per_day

def washing_machine_capacity : ℕ := 14

-- Statement to prove
theorem laundry_loads : total_towels / washing_machine_capacity = 6 := 
by
  sorry

end NUMINAMATH_GPT_laundry_loads_l1614_161409


namespace NUMINAMATH_GPT_integer_solutions_exist_l1614_161429

theorem integer_solutions_exist (R₀ : ℝ) : 
  ∃ (x₁ x₂ x₃ : ℤ), (x₁^2 + x₂^2 + x₃^2 = x₁ * x₂ * x₃) ∧ (R₀ < x₁) ∧ (R₀ < x₂) ∧ (R₀ < x₃) := 
sorry

end NUMINAMATH_GPT_integer_solutions_exist_l1614_161429


namespace NUMINAMATH_GPT_rectangle_perimeter_of_right_triangle_l1614_161494

-- Define the conditions for the triangle and the rectangle
def rightTriangleArea (a b c : ℕ) (h : a^2 + b^2 = c^2) : ℕ :=
  (1 / 2) * a * b

def rectanglePerimeter (width area : ℕ) : ℕ :=
  2 * ((area / width) + width)

theorem rectangle_perimeter_of_right_triangle :
  ∀ (a b c width : ℕ) (h_a : a = 5) (h_b : b = 12) (h_c : c = 13)
    (h_pyth : a^2 + b^2 = c^2) (h_width : width = 5)
    (h_area_eq : rightTriangleArea a b c h_pyth = width * (rightTriangleArea a b c h_pyth / width)),
  rectanglePerimeter width (rightTriangleArea a b c h_pyth) = 22 :=
by
  intros
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_of_right_triangle_l1614_161494


namespace NUMINAMATH_GPT_max_value_a4_a6_l1614_161483

theorem max_value_a4_a6 (a : ℕ → ℝ) (d : ℝ) (h1 : d ≥ 0) (h2 : ∀ n, a n > 0) (h3 : a 3 + 2 * a 6 = 6) :
  ∃ m, ∀ (a : ℕ → ℝ) (d : ℝ) (h1 : d ≥ 0) (h2 : ∀ n, a n > 0) (h3 : a 3 + 2 * a 6 = 6), a 4 * a 6 ≤ m :=
sorry

end NUMINAMATH_GPT_max_value_a4_a6_l1614_161483


namespace NUMINAMATH_GPT_tan_cos_identity_15deg_l1614_161422

theorem tan_cos_identity_15deg :
  (1 - (Real.tan (Real.pi / 12))^2) * (Real.cos (Real.pi / 12))^2 = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_tan_cos_identity_15deg_l1614_161422


namespace NUMINAMATH_GPT_find_some_number_l1614_161460

-- Definitions based on the given condition
def some_number : ℝ := sorry
def equation := some_number * 3.6 / (0.04 * 0.1 * 0.007) = 990.0000000000001

-- An assertion/proof that given the equation, some_number equals 7.7
theorem find_some_number (h : equation) : some_number = 7.7 :=
sorry

end NUMINAMATH_GPT_find_some_number_l1614_161460


namespace NUMINAMATH_GPT_complex_div_eq_half_sub_half_i_l1614_161434

theorem complex_div_eq_half_sub_half_i (i : ℂ) (hi : i^2 = -1) : 
  (i^3 / (1 - i)) = (1 / 2) - (1 / 2) * i :=
by
  sorry

end NUMINAMATH_GPT_complex_div_eq_half_sub_half_i_l1614_161434


namespace NUMINAMATH_GPT_employed_females_percentage_l1614_161405

theorem employed_females_percentage (total_population_percent employed_population_percent employed_males_percent : ℝ) :
  employed_population_percent = 70 → employed_males_percent = 21 →
  (employed_population_percent - employed_males_percent) / employed_population_percent * 100 = 70 :=
by
  -- Assume the total population percentage is 100%, which allows us to work directly with percentages.
  let employed_population_percent := 70
  let employed_males_percent := 21
  sorry

end NUMINAMATH_GPT_employed_females_percentage_l1614_161405


namespace NUMINAMATH_GPT_chess_team_selection_l1614_161454

theorem chess_team_selection
  (players : Finset ℕ) (twin1 twin2 : ℕ)
  (H1 : players.card = 10)
  (H2 : twin1 ∈ players)
  (H3 : twin2 ∈ players) :
  ∃ n : ℕ, n = 182 ∧ 
  (∃ team : Finset ℕ, team.card = 4 ∧
    (twin1 ∉ team ∨ twin2 ∉ team)) ∧
  n = (players.card.choose 4 - 
      ((players.erase twin1).erase twin2).card.choose 2) := sorry

end NUMINAMATH_GPT_chess_team_selection_l1614_161454


namespace NUMINAMATH_GPT_net_percentage_error_l1614_161453

noncomputable section
def calculate_percentage_error (true_side excess_error deficit_error : ℝ) : ℝ :=
  let measured_side1 := true_side * (1 + excess_error / 100)
  let measured_side2 := measured_side1 * (1 - deficit_error / 100)
  let true_area := true_side ^ 2
  let calculated_area := measured_side2 * true_side
  let percentage_error := ((true_area - calculated_area) / true_area) * 100
  percentage_error

theorem net_percentage_error 
  (S : ℝ) (h1 : S > 0) : calculate_percentage_error S 3 (-4) = 1.12 := by
  sorry

end NUMINAMATH_GPT_net_percentage_error_l1614_161453


namespace NUMINAMATH_GPT_solution_l1614_161415

noncomputable def inequality_prove (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) : Prop :=
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5)

noncomputable def equality_condition (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) : Prop :=
  (1 / (x + 3) + 1 / (y + 3) = 2 / 5) ↔ (x = 2 ∧ y = 2)

theorem solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) : 
  inequality_prove x y h1 h2 h3 ∧ equality_condition x y h1 h2 h3 := by
  sorry

end NUMINAMATH_GPT_solution_l1614_161415


namespace NUMINAMATH_GPT_min_value_x_y_l1614_161468

open Real

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / (x + 2) + 1 / (y + 2) = 1 / 6) : 
  x + y ≥ 20 :=
sorry

end NUMINAMATH_GPT_min_value_x_y_l1614_161468


namespace NUMINAMATH_GPT_quadratic_factoring_even_a_l1614_161433

theorem quadratic_factoring_even_a (a : ℤ) :
  (∃ (m p n q : ℤ), 21 * x^2 + a * x + 21 = (m * x + n) * (p * x + q) ∧ m * p = 21 ∧ n * q = 21 ∧ (∃ (k : ℤ), a = 2 * k)) :=
sorry

end NUMINAMATH_GPT_quadratic_factoring_even_a_l1614_161433


namespace NUMINAMATH_GPT_triangle_side_length_x_l1614_161428

theorem triangle_side_length_x (x : ℤ) (hpos : x > 0) (hineq1 : 7 < x^2) (hineq2 : x^2 < 17) :
    x = 3 ∨ x = 4 :=
by {
  apply sorry
}

end NUMINAMATH_GPT_triangle_side_length_x_l1614_161428


namespace NUMINAMATH_GPT_moles_of_water_formed_l1614_161432

-- Definitions (conditions)
def reaction : String := "NaOH + HCl → NaCl + H2O"

def initial_moles_NaOH : ℕ := 1
def initial_moles_HCl : ℕ := 1
def mole_ratio_NaOH_HCl : ℕ := 1
def mole_ratio_NaOH_H2O : ℕ := 1

-- The proof problem
theorem moles_of_water_formed :
  initial_moles_NaOH = mole_ratio_NaOH_HCl →
  initial_moles_HCl = mole_ratio_NaOH_HCl →
  mole_ratio_NaOH_H2O * initial_moles_NaOH = 1 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_moles_of_water_formed_l1614_161432


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l1614_161497

-- Define the equations and the problem.
def equation1 (x : ℝ) : Prop := (3 / (x^2 - 9)) + (x / (x - 3)) = 1
def equation2 (x : ℝ) : Prop := 2 - (1 / (2 - x)) = ((3 - x) / (x - 2))

-- Proof problem for the first equation: Prove that x = -4 is the solution.
theorem solve_equation1 : ∀ x : ℝ, equation1 x → x = -4 :=
by {
  sorry
}

-- Proof problem for the second equation: Prove that there are no solutions.
theorem solve_equation2 : ∀ x : ℝ, ¬equation2 x :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l1614_161497


namespace NUMINAMATH_GPT_sin_half_alpha_plus_beta_eq_sqrt2_div_2_l1614_161451

open Real

theorem sin_half_alpha_plus_beta_eq_sqrt2_div_2
  (α β : ℝ)
  (hα : α ∈ Set.Icc (π / 2) (3 * π / 2))
  (hβ : β ∈ Set.Icc (-π / 2) 0)
  (h1 : (α - π / 2)^3 - sin α - 2 = 0)
  (h2 : 8 * β^3 + 2 * (cos β)^2 + 1 = 0) :
  sin (α / 2 + β) = sqrt 2 / 2 := 
sorry

end NUMINAMATH_GPT_sin_half_alpha_plus_beta_eq_sqrt2_div_2_l1614_161451


namespace NUMINAMATH_GPT_rhombus_diagonals_not_equal_l1614_161496

-- Define what a rhombus is
structure Rhombus where
  sides_equal : ∀ a b : ℝ, a = b  -- all sides are equal
  symmetrical : Prop -- it is a symmetrical figure
  centrally_symmetrical : Prop -- it is a centrally symmetrical figure

-- Theorem to state that the diagonals of a rhombus are not necessarily equal
theorem rhombus_diagonals_not_equal (r : Rhombus) : ¬(∀ a b : ℝ, a = b) := by
  sorry

end NUMINAMATH_GPT_rhombus_diagonals_not_equal_l1614_161496


namespace NUMINAMATH_GPT_find_p_q_l1614_161495

theorem find_p_q (p q : ℤ) 
    (h1 : (3:ℤ)^5 - 2 * (3:ℤ)^4 + 3 * (3:ℤ)^3 - p * (3:ℤ)^2 + q * (3:ℤ) - 12 = 0)
    (h2 : (-1:ℤ)^5 - 2 * (-1:ℤ)^4 + 3 * (-1:ℤ)^3 - p * (-1:ℤ)^2 + q * (-1:ℤ) - 12 = 0) : 
    (p, q) = (-8, -10) :=
by
  sorry

end NUMINAMATH_GPT_find_p_q_l1614_161495


namespace NUMINAMATH_GPT_total_balloons_correct_l1614_161466

-- Define the number of blue balloons Joan and Melanie have
def Joan_balloons : ℕ := 40
def Melanie_balloons : ℕ := 41

-- Define the total number of blue balloons
def total_balloons : ℕ := Joan_balloons + Melanie_balloons

-- Prove that the total number of blue balloons is 81
theorem total_balloons_correct : total_balloons = 81 := by
  sorry

end NUMINAMATH_GPT_total_balloons_correct_l1614_161466


namespace NUMINAMATH_GPT_evaluate_expression_l1614_161475

theorem evaluate_expression : ∃ x : ℝ, (x = Real.sqrt (18 + x)) ∧ (x = (1 + Real.sqrt 73) / 2) := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1614_161475


namespace NUMINAMATH_GPT_solve_equation_l1614_161477

theorem solve_equation (x : ℤ) (h1 : x ≠ 2) : x - 8 / (x - 2) = 5 - 8 / (x - 2) → x = 5 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l1614_161477


namespace NUMINAMATH_GPT_remainder_of_876539_div_7_l1614_161489

theorem remainder_of_876539_div_7 : 876539 % 7 = 6 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_876539_div_7_l1614_161489


namespace NUMINAMATH_GPT_division_of_squares_l1614_161463

theorem division_of_squares {a b : ℕ} (h1 : a < 1000) (h2 : b > 0) (h3 : b^10 ∣ a^21) : b ∣ a^2 := 
sorry

end NUMINAMATH_GPT_division_of_squares_l1614_161463


namespace NUMINAMATH_GPT_parallel_vectors_l1614_161482

theorem parallel_vectors (m : ℝ) : (m = 1) ↔ (∃ k : ℝ, (m, 1) = k • (1, m)) := sorry

end NUMINAMATH_GPT_parallel_vectors_l1614_161482


namespace NUMINAMATH_GPT_Sam_has_walked_25_miles_l1614_161427

variables (d : ℕ) (v_fred v_sam : ℕ)

def Fred_and_Sam_meet (d : ℕ) (v_fred v_sam : ℕ) := 
  d / (v_fred + v_sam) * v_sam

theorem Sam_has_walked_25_miles :
  Fred_and_Sam_meet 50 5 5 = 25 :=
by
  sorry

end NUMINAMATH_GPT_Sam_has_walked_25_miles_l1614_161427


namespace NUMINAMATH_GPT_radius_large_circle_l1614_161480

/-- Let R be the radius of the large circle. Assume three circles of radius 2 are externally 
tangent to each other. Two of these circles are internally tangent to the larger circle, 
and the third circle is tangent to the larger circle both internally and externally. 
Prove that the radius of the large circle is 4 + 2 * sqrt 3. -/
theorem radius_large_circle (R : ℝ)
  (h1 : ∃ (C1 C2 C3 : ℝ × ℝ), 
    dist C1 C2 = 4 ∧ dist C2 C3 = 4 ∧ dist C3 C1 = 4 ∧ 
    (∃ (O : ℝ × ℝ), 
      (dist O C1 = R - 2) ∧ 
      (dist O C2 = R - 2) ∧ 
      (dist O C3 = R + 2) ∧ 
      (dist C1 C2 = 4) ∧ (dist C2 C3 = 4) ∧ (dist C3 C1 = 4))):
  R = 4 + 2 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_radius_large_circle_l1614_161480


namespace NUMINAMATH_GPT_correct_divisor_l1614_161411

theorem correct_divisor (dividend incorrect_divisor quotient correct_quotient correct_divisor : ℕ) 
  (h1 : incorrect_divisor = 63) 
  (h2 : quotient = 24) 
  (h3 : correct_quotient = 42) 
  (h4 : dividend = incorrect_divisor * quotient) 
  (h5 : dividend / correct_divisor = correct_quotient) : 
  correct_divisor = 36 := 
by 
  sorry

end NUMINAMATH_GPT_correct_divisor_l1614_161411


namespace NUMINAMATH_GPT_shaded_triangle_ratio_is_correct_l1614_161419

noncomputable def ratio_of_shaded_triangle_to_large_square (total_area : ℝ) 
  (midpoint_area_ratio : ℝ := 1 / 24) : ℝ :=
  midpoint_area_ratio * total_area

theorem shaded_triangle_ratio_is_correct 
  (shaded_area total_area : ℝ)
  (n : ℕ)
  (h1 : n = 36)
  (grid_area : ℝ)
  (condition1 : grid_area = total_area / n)
  (condition2 : shaded_area = grid_area / 2 * 3)
  : shaded_area / total_area = 1 / 24 :=
by
  sorry

end NUMINAMATH_GPT_shaded_triangle_ratio_is_correct_l1614_161419


namespace NUMINAMATH_GPT_inequality_solution_l1614_161424

theorem inequality_solution :
  {x : ℝ | (x - 1) * (x - 4) * (x - 5)^2 / ((x - 3) * (x^2 - 9)) > 0} = { x : ℝ | -3 < x ∧ x < 3 } :=
sorry

end NUMINAMATH_GPT_inequality_solution_l1614_161424


namespace NUMINAMATH_GPT_rahul_matches_played_l1614_161438

theorem rahul_matches_played
  (current_avg runs_today new_avg : ℕ)
  (h1 : current_avg = 51)
  (h2 : runs_today = 69)
  (h3 : new_avg = 54)
  : ∃ m : ℕ, ((51 * m + 69) / (m + 1) = 54) ∧ (m = 5) :=
by
  sorry

end NUMINAMATH_GPT_rahul_matches_played_l1614_161438


namespace NUMINAMATH_GPT_mistaken_multiplier_is_34_l1614_161449

-- Define the main conditions
def correct_number : ℕ := 135
def correct_multiplier : ℕ := 43
def difference : ℕ := 1215

-- Define what we need to prove
theorem mistaken_multiplier_is_34 :
  (correct_number * correct_multiplier - correct_number * x = difference) →
  x = 34 :=
by
  sorry

end NUMINAMATH_GPT_mistaken_multiplier_is_34_l1614_161449


namespace NUMINAMATH_GPT_periodic_function_l1614_161457

variable {α : Type*} [AddGroup α] {f : α → α} {a b : α}

def symmetric_around (c : α) (f : α → α) : Prop := ∀ x, f (c - x) = f (c + x)

theorem periodic_function (h1 : symmetric_around a f) (h2 : symmetric_around b f) (h_ab : a ≠ b) : ∃ T, (∀ x, f (x + T) = f x) := 
sorry

end NUMINAMATH_GPT_periodic_function_l1614_161457


namespace NUMINAMATH_GPT_businessmen_drink_neither_l1614_161481

theorem businessmen_drink_neither : 
  ∀ (total coffee tea both : ℕ), 
    total = 30 → 
    coffee = 15 → 
    tea = 13 → 
    both = 8 → 
    total - (coffee - both + tea - both + both) = 10 := 
by 
  intros total coffee tea both h_total h_coffee h_tea h_both
  sorry

end NUMINAMATH_GPT_businessmen_drink_neither_l1614_161481


namespace NUMINAMATH_GPT_isosceles_right_triangle_example_l1614_161444

theorem isosceles_right_triangle_example :
  (5 = 5) ∧ (5^2 + 5^2 = (5 * Real.sqrt 2)^2) :=
by {
  sorry
}

end NUMINAMATH_GPT_isosceles_right_triangle_example_l1614_161444


namespace NUMINAMATH_GPT_cows_sold_l1614_161450

/-- 
A man initially had 39 cows, 25 of them died last year, he sold some remaining cows, this year,
the number of cows increased by 24, he bought 43 more cows, his friend gave him 8 cows.
Now, he has 83 cows. How many cows did he sell last year?
-/
theorem cows_sold (S : ℕ) : (39 - 25 - S + 24 + 43 + 8 = 83) → S = 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cows_sold_l1614_161450


namespace NUMINAMATH_GPT_irreducible_positive_fraction_unique_l1614_161420

theorem irreducible_positive_fraction_unique
  (a b : ℕ) (h_pos : a > 0 ∧ b > 0) (h_gcd : Nat.gcd a b = 1)
  (h_eq : (a + 12) * b = 3 * a * (b + 12)) :
  a = 2 ∧ b = 9 :=
by
  sorry

end NUMINAMATH_GPT_irreducible_positive_fraction_unique_l1614_161420


namespace NUMINAMATH_GPT_smallest_positive_integer_divisible_l1614_161490

theorem smallest_positive_integer_divisible (n : ℕ) (h1 : 15 = 3 * 5) (h2 : 16 = 2 ^ 4) (h3 : 18 = 2 * 3 ^ 2) :
  n = Nat.lcm (Nat.lcm 15 16) 18 ↔ n = 720 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_divisible_l1614_161490


namespace NUMINAMATH_GPT_solve_equation_l1614_161423

theorem solve_equation :
  ∃ x : ℝ, (x - 2)^3 + (x - 6)^3 = 54 ∧ x = 7 := by
sorry

end NUMINAMATH_GPT_solve_equation_l1614_161423


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l1614_161456

noncomputable def e (a b c : ℝ) : ℝ := c / a

theorem eccentricity_of_ellipse (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : (a - c) * (a + c) = (2 * c)^2) : e a b c = (Real.sqrt 5) / 5 := 
by
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l1614_161456


namespace NUMINAMATH_GPT_difference_of_numbers_l1614_161441

theorem difference_of_numbers 
  (L S : ℤ) (hL : L = 1636) (hdiv : L = 6 * S + 10) : 
  L - S = 1365 :=
sorry

end NUMINAMATH_GPT_difference_of_numbers_l1614_161441


namespace NUMINAMATH_GPT_most_likely_outcome_l1614_161498

-- Defining the conditions
def equally_likely (n : ℕ) (k : ℕ) := (Nat.choose n k) * (1 / 2)^n

-- Defining the problem statement
theorem most_likely_outcome :
  (equally_likely 5 3 = 5 / 16 ∧ equally_likely 5 2 = 5 / 16) :=
sorry

end NUMINAMATH_GPT_most_likely_outcome_l1614_161498


namespace NUMINAMATH_GPT_distance_between_chords_l1614_161426

theorem distance_between_chords (R AB CD : ℝ) (hR : R = 15) (hAB : AB = 18) (hCD : CD = 24) : 
  ∃ d : ℝ, d = 21 :=
by 
  sorry

end NUMINAMATH_GPT_distance_between_chords_l1614_161426


namespace NUMINAMATH_GPT_rings_on_fingers_arrangement_l1614_161407

-- Definitions based on the conditions
def rings : ℕ := 5
def fingers : ℕ := 5

-- Theorem statement
theorem rings_on_fingers_arrangement : (fingers ^ rings) = 5 ^ 5 := by
  sorry  -- Proof skipped

end NUMINAMATH_GPT_rings_on_fingers_arrangement_l1614_161407


namespace NUMINAMATH_GPT_total_pages_in_book_l1614_161418

def pagesReadMonday := 23
def pagesReadTuesday := 38
def pagesReadWednesday := 61
def pagesReadThursday := 12
def pagesReadFriday := 2 * pagesReadThursday

def totalPagesRead := pagesReadMonday + pagesReadTuesday + pagesReadWednesday + pagesReadThursday + pagesReadFriday

theorem total_pages_in_book :
  totalPagesRead = 158 :=
by
  sorry

end NUMINAMATH_GPT_total_pages_in_book_l1614_161418


namespace NUMINAMATH_GPT_subset_contains_square_l1614_161417

theorem subset_contains_square {A : Finset ℕ} (hA₁ : A ⊆ Finset.range 101) (hA₂ : A.card = 50) (hA₃ : ∀ x ∈ A, ∀ y ∈ A, x + y ≠ 100) : 
  ∃ x ∈ A, ∃ k : ℕ, x = k^2 := 
sorry

end NUMINAMATH_GPT_subset_contains_square_l1614_161417


namespace NUMINAMATH_GPT_remainder_of_1234567_div_257_l1614_161430

theorem remainder_of_1234567_div_257 : 1234567 % 257 = 123 := by
  sorry

end NUMINAMATH_GPT_remainder_of_1234567_div_257_l1614_161430


namespace NUMINAMATH_GPT_cannot_determine_total_movies_l1614_161402

def number_of_books : ℕ := 22
def books_read : ℕ := 12
def books_to_read : ℕ := 10
def movies_watched : ℕ := 56

theorem cannot_determine_total_movies (n : ℕ) (h1 : books_read + books_to_read = number_of_books) : n ≠ movies_watched → n = 56 → False := 
by 
  intro h2 h3
  sorry

end NUMINAMATH_GPT_cannot_determine_total_movies_l1614_161402


namespace NUMINAMATH_GPT_original_population_before_changes_l1614_161455

open Nat

def halved_population (p: ℕ) (years: ℕ) : ℕ := p / (2^years)

theorem original_population_before_changes (P_init P_final : ℕ)
    (new_people : ℕ) (people_moved_out : ℕ) :
    new_people = 100 →
    people_moved_out = 400 →
    ∀ years, (years = 4 → halved_population P_final years = 60) →
    ∃ P_before_change, P_before_change = 780 ∧
    P_init = P_before_change + new_people - people_moved_out ∧
    halved_population P_init years = P_final := 
by
  intros
  sorry

end NUMINAMATH_GPT_original_population_before_changes_l1614_161455


namespace NUMINAMATH_GPT_systematic_sampling_first_group_l1614_161486

theorem systematic_sampling_first_group (S : ℕ) (n : ℕ) (students_per_group : ℕ) (group_number : ℕ)
(h1 : n = 160)
(h2 : students_per_group = 8)
(h3 : group_number = 16)
(h4 : S + (group_number - 1) * students_per_group = 126)
: S = 6 := by
  sorry

end NUMINAMATH_GPT_systematic_sampling_first_group_l1614_161486


namespace NUMINAMATH_GPT_find_k_l1614_161478

theorem find_k : 
  ∀ (k : ℤ), 2^4 - 6 = 3^3 + k ↔ k = -17 :=
by sorry

end NUMINAMATH_GPT_find_k_l1614_161478


namespace NUMINAMATH_GPT_plot_area_is_correct_l1614_161404

noncomputable def scaled_area_in_acres
  (scale_cm_miles : ℕ)
  (area_conversion_factor_miles_acres : ℕ)
  (bottom_cm : ℕ)
  (top_cm : ℕ)
  (height_cm : ℕ) : ℕ :=
  let area_cm_squared := (1 / 2) * (bottom_cm + top_cm) * height_cm
  let area_in_squared_miles := area_cm_squared * (scale_cm_miles * scale_cm_miles)
  area_in_squared_miles * area_conversion_factor_miles_acres

theorem plot_area_is_correct :
  scaled_area_in_acres 3 640 18 14 12 = 1105920 :=
by
  sorry

end NUMINAMATH_GPT_plot_area_is_correct_l1614_161404


namespace NUMINAMATH_GPT_combined_weight_proof_l1614_161452

-- Definitions of atomic weights
def weight_C : ℝ := 12.01
def weight_H : ℝ := 1.01
def weight_O : ℝ := 16.00
def weight_S : ℝ := 32.07

-- Definitions of molar masses of compounds
def molar_mass_C6H8O7 : ℝ := (6 * weight_C) + (8 * weight_H) + (7 * weight_O)
def molar_mass_H2SO4 : ℝ := (2 * weight_H) + weight_S + (4 * weight_O)

-- Definitions of number of moles
def moles_C6H8O7 : ℝ := 8
def moles_H2SO4 : ℝ := 4

-- Combined weight
def combined_weight : ℝ := (moles_C6H8O7 * molar_mass_C6H8O7) + (moles_H2SO4 * molar_mass_H2SO4)

theorem combined_weight_proof : combined_weight = 1929.48 :=
by
  -- calculations as explained in the problem
  let wC6H8O7 := moles_C6H8O7 * molar_mass_C6H8O7
  let wH2SO4 := moles_H2SO4 * molar_mass_H2SO4
  have h1 : wC6H8O7 = 8 * 192.14 := by sorry
  have h2 : wH2SO4 = 4 * 98.09 := by sorry
  have h3 : combined_weight = wC6H8O7 + wH2SO4 := by simp [combined_weight, wC6H8O7, wH2SO4]
  rw [h3, h1, h2]
  simp
  sorry -- finish the proof as necessary

end NUMINAMATH_GPT_combined_weight_proof_l1614_161452


namespace NUMINAMATH_GPT_solve_eq1_solve_system_l1614_161425

theorem solve_eq1 : ∃ x y : ℝ, (3 / x) + (2 / y) = 4 :=
by
  use 1
  use 2
  sorry

theorem solve_system :
  ∃ x y : ℝ,
    (3 / x + 2 / y = 4) ∧ (5 / x - 6 / y = 2) ∧ (x = 1) ∧ (y = 2) :=
by
  use 1
  use 2
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_system_l1614_161425


namespace NUMINAMATH_GPT_max_ounces_among_items_l1614_161401

theorem max_ounces_among_items
  (budget : ℝ)
  (candy_cost : ℝ)
  (candy_ounces : ℝ)
  (candy_stock : ℕ)
  (chips_cost : ℝ)
  (chips_ounces : ℝ)
  (chips_stock : ℕ)
  : budget = 7 → candy_cost = 1.25 → candy_ounces = 12 →
    candy_stock = 5 → chips_cost = 1.40 → chips_ounces = 17 → chips_stock = 4 →
    max (min ((budget / candy_cost) * candy_ounces) (candy_stock * candy_ounces))
        (min ((budget / chips_cost) * chips_ounces) (chips_stock * chips_ounces)) = 68 := 
by
  intros h_budget h_candy_cost h_candy_ounces h_candy_stock h_chips_cost h_chips_ounces h_chips_stock
  sorry

end NUMINAMATH_GPT_max_ounces_among_items_l1614_161401


namespace NUMINAMATH_GPT_existence_of_unique_root_l1614_161461

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x - 5

theorem existence_of_unique_root :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  f 0 = -4 ∧
  f 2 = Real.exp 2 - 1 →
  ∃! c, f c = 0 :=
by
  sorry

end NUMINAMATH_GPT_existence_of_unique_root_l1614_161461


namespace NUMINAMATH_GPT_line_of_intersection_canonical_form_l1614_161469

def canonical_form_of_line (A B : ℝ) (x y z : ℝ) :=
  (x / A) = (y / B) ∧ (y / B) = (z)

theorem line_of_intersection_canonical_form :
  ∀ (x y z : ℝ),
  x + y - 2*z - 2 = 0 →
  x - y + z + 2 = 0 →
  canonical_form_of_line (-1) (-3) x (y - 2) (-2) :=
by
  intros x y z h_eq1 h_eq2
  sorry

end NUMINAMATH_GPT_line_of_intersection_canonical_form_l1614_161469


namespace NUMINAMATH_GPT_dimes_count_l1614_161488

-- Definitions of types of coins and their values in cents.
def penny := 1
def nickel := 5
def dime := 10
def quarter := 25
def halfDollar := 50

-- Condition statements as assumptions
variables (num_pennies num_nickels num_dimes num_quarters num_halfDollars : ℕ)

-- Sum of all coins and their values (in cents)
def total_value := num_pennies * penny + num_nickels * nickel + num_dimes * dime + num_quarters * quarter + num_halfDollars * halfDollar

-- Total number of coins
def total_coins := num_pennies + num_nickels + num_dimes + num_quarters + num_halfDollars

-- Proving the number of dimes is 5 given the conditions.
theorem dimes_count : 
  total_value = 163 ∧ 
  total_coins = 12 ∧ 
  num_pennies ≥ 1 ∧ 
  num_nickels ≥ 1 ∧ 
  num_dimes ≥ 1 ∧ 
  num_quarters ≥ 1 ∧ 
  num_halfDollars ≥ 1 → 
  num_dimes = 5 :=
by
  sorry

end NUMINAMATH_GPT_dimes_count_l1614_161488


namespace NUMINAMATH_GPT_solve_fraction_l1614_161484

theorem solve_fraction (x : ℝ) (h₁ : x^2 - 1 = 0) (h₂ : (x - 2) * (x + 1) ≠ 0) : x = 1 := 
sorry

end NUMINAMATH_GPT_solve_fraction_l1614_161484
