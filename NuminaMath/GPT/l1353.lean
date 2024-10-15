import Mathlib

namespace NUMINAMATH_GPT_correct_technology_used_l1353_135332

-- Define the condition that the program title is "Back to the Dinosaur Era"
def program_title : String := "Back to the Dinosaur Era"

-- Define the condition that the program vividly recreated various dinosaurs and their living environments
def recreated_living_environments : Bool := true

-- Define the options for digital Earth technologies
inductive DigitalEarthTechnology
| InformationSuperhighway
| HighResolutionSatelliteTechnology
| SpatialInformationTechnology
| VisualizationAndVirtualRealityTechnology

-- Define the correct answer
def correct_technology := DigitalEarthTechnology.VisualizationAndVirtualRealityTechnology

-- The proof problem: Prove that given the conditions, the technology used is the correct one
theorem correct_technology_used
  (title : program_title = "Back to the Dinosaur Era")
  (recreated : recreated_living_environments) :
  correct_technology = DigitalEarthTechnology.VisualizationAndVirtualRealityTechnology :=
by
  sorry

end NUMINAMATH_GPT_correct_technology_used_l1353_135332


namespace NUMINAMATH_GPT_find_speeds_l1353_135300

/--
From point A to point B, which are 40 km apart, a pedestrian set out at 4:00 AM,
and a cyclist set out at 7:20 AM. The cyclist caught up with the pedestrian exactly
halfway between A and B, after which both continued their journey. A second cyclist
with the same speed as the first cyclist set out from B to A at 8:30 AM and met the
pedestrian one hour after the pedestrian's meeting with the first cyclist. Prove that
the speed of the pedestrian is 5 km/h and the speed of the cyclists is 30 km/h.
-/
theorem find_speeds (x y : ℝ) : 
  (∀ t : ℝ, (0 <= t ∧ t < (7 + (1/3)) ∨ (7 + (1/3)) <= t ∧ t <= 20) -> (x * t + 20 = y * ((7 + (1/3)) - t))) ∧ -- Midpoint and catch-up condition
  (∀ t, (8 + (1/2) <= t) -> (40 - (x * (8 + (1/2))) = y * (t - (8 + (1/2))))) -> -- Second meeting condition
  x = 5 ∧ y = 30 := 
sorry

end NUMINAMATH_GPT_find_speeds_l1353_135300


namespace NUMINAMATH_GPT_valid_assignment_l1353_135355

/-- A function to check if an expression is a valid assignment expression -/
def is_assignment (lhs : String) (rhs : String) : Prop :=
  lhs = "x" ∧ (rhs = "3" ∨ rhs = "x + 1")

theorem valid_assignment :
  (is_assignment "x" "x + 1") ∧
  ¬(is_assignment "3" "x") ∧
  ¬(is_assignment "x" "3") ∧
  ¬(is_assignment "x" "x2 + 1") :=
by
  sorry

end NUMINAMATH_GPT_valid_assignment_l1353_135355


namespace NUMINAMATH_GPT_smallest_prime_reverse_square_l1353_135389

open Nat

-- Define a function to reverse the digits of a two-digit number
def reverseDigits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

-- Define the conditions
def isTwoDigitPrime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

def isSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define the main statement
theorem smallest_prime_reverse_square : 
  ∃ P, isTwoDigitPrime P ∧ isSquare (reverseDigits P) ∧ 
       ∀ Q, isTwoDigitPrime Q ∧ isSquare (reverseDigits Q) → P ≤ Q :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_reverse_square_l1353_135389


namespace NUMINAMATH_GPT_original_two_digit_number_is_52_l1353_135321

theorem original_two_digit_number_is_52 (x : ℕ) (h1 : 10 * x + 6 = x + 474) (h2 : 10 ≤ x ∧ x < 100) : x = 52 :=
sorry

end NUMINAMATH_GPT_original_two_digit_number_is_52_l1353_135321


namespace NUMINAMATH_GPT_percent_greater_than_average_l1353_135313

variable (M N : ℝ)

theorem percent_greater_than_average (h : M > N) :
  (200 * (M - N)) / (M + N) = ((M - ((M + N) / 2)) / ((M + N) / 2)) * 100 :=
by 
  sorry

end NUMINAMATH_GPT_percent_greater_than_average_l1353_135313


namespace NUMINAMATH_GPT_rectangle_area_l1353_135309

-- Define length and width
def width : ℕ := 6
def length : ℕ := 3 * width

-- Define area of the rectangle
def area (length width : ℕ) : ℕ := length * width

-- Statement to prove
theorem rectangle_area : area length width = 108 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1353_135309


namespace NUMINAMATH_GPT_sqrt_eq_solutions_l1353_135362

theorem sqrt_eq_solutions (x : ℝ) : 
  (Real.sqrt ((2 + Real.sqrt 5) ^ x) + Real.sqrt ((2 - Real.sqrt 5) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := 
by
  sorry

end NUMINAMATH_GPT_sqrt_eq_solutions_l1353_135362


namespace NUMINAMATH_GPT_John_completes_work_alone_10_days_l1353_135356

theorem John_completes_work_alone_10_days
  (R : ℕ)
  (T : ℕ)
  (W : ℕ)
  (H1 : R = 40)
  (H2 : T = 8)
  (H3 : 1/10 = (1/R) + (1/W))
  : W = 10 := sorry

end NUMINAMATH_GPT_John_completes_work_alone_10_days_l1353_135356


namespace NUMINAMATH_GPT_factorize_expression_l1353_135339

theorem factorize_expression
  (x : ℝ) :
  ( (x^2-1)*(x^4+x^2+1)-(x^3+1)^2 ) = -2*(x + 1)*(x^2 - x + 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1353_135339


namespace NUMINAMATH_GPT_probability_points_one_unit_apart_l1353_135343

theorem probability_points_one_unit_apart :
  let total_points := 16
  let total_pairs := (total_points * (total_points - 1)) / 2
  let favorable_pairs := 12
  let probability := favorable_pairs / total_pairs
  probability = (1 : ℚ) / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_points_one_unit_apart_l1353_135343


namespace NUMINAMATH_GPT_second_hand_travel_distance_l1353_135395

theorem second_hand_travel_distance (r : ℝ) (t : ℝ) (C : ℝ) :
    r = 8 ∧ t = 45 ∧ C = 2 * Real.pi * r → 
    r * C * t = 720 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_second_hand_travel_distance_l1353_135395


namespace NUMINAMATH_GPT_omega_not_real_root_l1353_135344

theorem omega_not_real_root {ω : ℂ} (h1 : ω^3 = 1) (h2 : ω ≠ 1) (h3 : ω^2 + ω + 1 = 0) :
  (2 + 3 * ω - ω^2)^3 + (2 - 3 * ω + ω^2)^3 = -68 + 96 * ω :=
by sorry

end NUMINAMATH_GPT_omega_not_real_root_l1353_135344


namespace NUMINAMATH_GPT_probability_at_least_6_heads_l1353_135364

-- Definitions of the binomial coefficient and probability function
def binom (n k : ℕ) : ℕ := Nat.choose n k

def probability (favorable total : ℕ) : ℚ := favorable / total

-- Proof problem statement
theorem probability_at_least_6_heads (flips : ℕ) (p : ℚ) 
  (h_flips : flips = 8) 
  (h_probability : p = probability (binom 8 6 + binom 8 7 + binom 8 8) (2 ^ flips)) : 
  p = 37 / 256 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_6_heads_l1353_135364


namespace NUMINAMATH_GPT_conic_sections_l1353_135302

theorem conic_sections (x y : ℝ) : 
  y^4 - 16*x^4 = 8*y^2 - 4 → 
  (y^2 - 4 * x^2 = 4 ∨ y^2 + 4 * x^2 = 4) :=
sorry

end NUMINAMATH_GPT_conic_sections_l1353_135302


namespace NUMINAMATH_GPT_angle_measure_l1353_135334

-- Define the complement function
def complement (α : ℝ) : ℝ := 180 - α

-- Given condition
variable (α : ℝ)
variable (h : complement α = 120)

-- Theorem to prove
theorem angle_measure : α = 60 :=
by sorry

end NUMINAMATH_GPT_angle_measure_l1353_135334


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l1353_135328

-- Define the necessary variables and conditions
variable (a b c α β : ℝ)
variable (h1 : 0 < α)
variable (h2 : α < β)
variable (h3 : ∀ x : ℝ, (a * x^2 + b * x + c > 0) ↔ (α < x ∧ x < β))

-- Statement to be proved
theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, ((a + c - b) * x^2 + (b - 2 * a) * x + a > 0) ↔ ((1 / (1 + β) < x) ∧ (x < 1 / (1 + α))) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l1353_135328


namespace NUMINAMATH_GPT_max_s_value_l1353_135382

noncomputable def max_s (m n : ℝ) : ℝ := (m-1)^2 + (n-1)^2 + (m-n)^2

theorem max_s_value (m n : ℝ) (h : m^2 - 4 * n ≥ 0) : 
    ∃ s : ℝ, s = (max_s m n) ∧ s ≤ 9/8 := sorry

end NUMINAMATH_GPT_max_s_value_l1353_135382


namespace NUMINAMATH_GPT_quadratic_root_sqrt_2010_2009_l1353_135319

theorem quadratic_root_sqrt_2010_2009 :
  (∃ (a b : ℤ), a = 0 ∧ b = -(2010 + 2 * Real.sqrt 2009) ∧
  ∀ (x : ℝ), x^2 + (a : ℝ) * x + (b : ℝ) = 0 → x = Real.sqrt (2010 + 2 * Real.sqrt 2009) ∨ x = -Real.sqrt (2010 + 2 * Real.sqrt 2009)) :=
sorry

end NUMINAMATH_GPT_quadratic_root_sqrt_2010_2009_l1353_135319


namespace NUMINAMATH_GPT_actual_distance_traveled_l1353_135314

theorem actual_distance_traveled (D t : ℝ) 
  (h1 : D = 15 * t)
  (h2 : D + 50 = 35 * t) : 
  D = 37.5 :=
by
  sorry

end NUMINAMATH_GPT_actual_distance_traveled_l1353_135314


namespace NUMINAMATH_GPT_relationship_y1_y2_y3_l1353_135316

def quadratic (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 1

variable (m : ℝ) (y1 y2 y3 : ℝ)

-- Given conditions
axiom m_gt_zero : m > 0
axiom point1_on_graph : y1 = quadratic m (-1)
axiom point2_on_graph : y2 = quadratic m (5 / 2)
axiom point3_on_graph : y3 = quadratic m 6

-- Prove the relationship between y1, y2, and y3
theorem relationship_y1_y2_y3 : y3 > y1 ∧ y1 > y2 :=
by sorry

end NUMINAMATH_GPT_relationship_y1_y2_y3_l1353_135316


namespace NUMINAMATH_GPT_initial_number_2008_l1353_135349

theorem initial_number_2008 
  (numbers_on_blackboard : ℕ → Prop)
  (x : ℕ)
  (Ops : ∀ x, numbers_on_blackboard x → (numbers_on_blackboard (2 * x + 1) ∨ numbers_on_blackboard (x / (x + 2)))) 
  (initial_apearing : numbers_on_blackboard 2008) :
  numbers_on_blackboard 2008 = true :=
sorry

end NUMINAMATH_GPT_initial_number_2008_l1353_135349


namespace NUMINAMATH_GPT_printer_ratio_l1353_135374

-- Define the given conditions
def total_price_basic_computer_printer := 2500
def enhanced_computer_extra := 500
def basic_computer_price := 1500

-- The lean statement to prove the ratio of the price of the printer to the total price of the enhanced computer and printer is 1/3
theorem printer_ratio : ∀ (C_basic P C_enhanced Total_enhanced : ℕ), 
  C_basic + P = total_price_basic_computer_printer →
  C_enhanced = C_basic + enhanced_computer_extra →
  C_basic = basic_computer_price →
  C_enhanced + P = Total_enhanced →
  P / Total_enhanced = 1 / 3 := 
by
  intros C_basic P C_enhanced Total_enhanced h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_printer_ratio_l1353_135374


namespace NUMINAMATH_GPT_S_15_eq_1695_l1353_135363

open Nat

/-- Sum of the nth set described in the problem -/
def S (n : ℕ) : ℕ :=
  let first := 1 + (n * (n - 1)) / 2
  let last := first + n - 1
  (n * (first + last)) / 2

theorem S_15_eq_1695 : S 15 = 1695 :=
by
  sorry

end NUMINAMATH_GPT_S_15_eq_1695_l1353_135363


namespace NUMINAMATH_GPT_num_real_values_for_integer_roots_l1353_135305

theorem num_real_values_for_integer_roots : 
  (∃ (a : ℝ), ∀ (r s : ℤ), r + s = -a ∧ r * s = 9 * a) → ∃ (n : ℕ), n = 10 :=
by
  sorry

end NUMINAMATH_GPT_num_real_values_for_integer_roots_l1353_135305


namespace NUMINAMATH_GPT_log_expression_defined_l1353_135312

theorem log_expression_defined (x : ℝ) : ∃ c : ℝ, (∀ x > c, (x > 7^8)) :=
by
  existsi 7^8
  intro x hx
  sorry

end NUMINAMATH_GPT_log_expression_defined_l1353_135312


namespace NUMINAMATH_GPT_max_marks_paper_I_l1353_135340

theorem max_marks_paper_I (M : ℝ) (h1 : 0.40 * M = 60) : M = 150 :=
by
  sorry

end NUMINAMATH_GPT_max_marks_paper_I_l1353_135340


namespace NUMINAMATH_GPT_inheritance_amount_l1353_135397

-- Define the conditions
variable (x : ℝ) -- Let x be the inheritance amount
variable (H1 : x * 0.25 + (x * 0.75 - 5000) * 0.15 + 5000 = 16500)

-- Define the theorem to prove the inheritance amount
theorem inheritance_amount (H1 : x * 0.25 + (0.75 * x - 5000) * 0.15 + 5000 = 16500) : x = 33794 := by
  sorry

end NUMINAMATH_GPT_inheritance_amount_l1353_135397


namespace NUMINAMATH_GPT_wings_per_person_l1353_135320

-- Define the number of friends
def number_of_friends : ℕ := 15

-- Define the number of wings already cooked
def wings_already_cooked : ℕ := 7

-- Define the number of additional wings cooked
def additional_wings_cooked : ℕ := 45

-- Define the number of friends who don't eat chicken
def friends_not_eating : ℕ := 2

-- Calculate the total number of chicken wings
def total_chicken_wings : ℕ := wings_already_cooked + additional_wings_cooked

-- Calculate the number of friends who will eat chicken
def friends_eating : ℕ := number_of_friends - friends_not_eating

-- Define the statement we want to prove
theorem wings_per_person : total_chicken_wings / friends_eating = 4 := by
  sorry

end NUMINAMATH_GPT_wings_per_person_l1353_135320


namespace NUMINAMATH_GPT_slope_of_perpendicular_line_l1353_135327

theorem slope_of_perpendicular_line (m1 m2 : ℝ) : 
  (5*x - 2*y = 10) →  ∃ m2, m2 = (-2/5) :=
by sorry

end NUMINAMATH_GPT_slope_of_perpendicular_line_l1353_135327


namespace NUMINAMATH_GPT_smallest_integer_is_nine_l1353_135378

theorem smallest_integer_is_nine 
  (a b c : ℕ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0) 
  (h2 : a + b + c = 90) 
  (h3 : (a:ℝ)/b = 2/3) 
  (h4 : (b:ℝ)/c = 3/5) : 
  a = 9 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_integer_is_nine_l1353_135378


namespace NUMINAMATH_GPT_nail_polishes_total_l1353_135322

theorem nail_polishes_total :
  let k := 25
  let h := k + 8
  let r := k - 6
  h + r = 52 :=
by
  sorry

end NUMINAMATH_GPT_nail_polishes_total_l1353_135322


namespace NUMINAMATH_GPT_halfway_fraction_between_l1353_135303

theorem halfway_fraction_between (a b : ℚ) (h_a : a = 1/6) (h_b : b = 1/4) : (a + b) / 2 = 5 / 24 :=
by
  have h1 : a = (1 : ℚ) / 6 := h_a
  have h2 : b = (1 : ℚ) / 4 := h_b
  sorry

end NUMINAMATH_GPT_halfway_fraction_between_l1353_135303


namespace NUMINAMATH_GPT_good_numbers_l1353_135348

/-- Definition of a good number -/
def is_good (n : ℕ) : Prop :=
  ∃ (k_1 k_2 k_3 k_4 : ℕ), 
    (1 ≤ k_1 ∧ 1 ≤ k_2 ∧ 1 ≤ k_3 ∧ 1 ≤ k_4) ∧
    (n + k_1 ∣ n + k_1^2) ∧ 
    (n + k_2 ∣ n + k_2^2) ∧ 
    (n + k_3 ∣ n + k_3^2) ∧ 
    (n + k_4 ∣ n + k_4^2) ∧
    (k_1 ≠ k_2) ∧ (k_1 ≠ k_3) ∧ (k_1 ≠ k_4) ∧
    (k_2 ≠ k_3) ∧ (k_2 ≠ k_4) ∧ 
    (k_3 ≠ k_4)

/-- The main theorem to prove -/
theorem good_numbers : 
  is_good 58 ∧ 
  ∀ (p : ℕ), p > 2 → 
  (Prime p ∧ Prime (2 * p + 1) ↔ is_good (2 * p)) :=
by
  sorry

end NUMINAMATH_GPT_good_numbers_l1353_135348


namespace NUMINAMATH_GPT_balloon_difference_l1353_135396

def num_balloons_you := 7
def num_balloons_friend := 5

theorem balloon_difference : (num_balloons_you - num_balloons_friend) = 2 := by
  sorry

end NUMINAMATH_GPT_balloon_difference_l1353_135396


namespace NUMINAMATH_GPT_donation_percentage_correct_l1353_135311

noncomputable def percentage_donated_to_orphan_house (income remaining : ℝ) (given_to_children_percentage : ℝ) (given_to_wife_percentage : ℝ) (remaining_after_donation : ℝ)
    (before_donation_remaining : income * (1 - given_to_children_percentage / 100 - given_to_wife_percentage / 100) = remaining)
    (after_donation_remaining : remaining - remaining_after_donation * remaining = 500) : Prop :=
    100 * (remaining - 500) / remaining = 16.67

theorem donation_percentage_correct 
    (income : ℝ) 
    (child_percentage : ℝ := 10)
    (num_children : ℕ := 2)
    (wife_percentage : ℝ := 20)
    (final_amount : ℝ := 500)
    (income_value : income = 1000 ) : 
    percentage_donated_to_orphan_house income 
    (income * (1 - (child_percentage * num_children) / 100 - wife_percentage / 100)) 
    (child_percentage * num_children)
    wife_percentage 
    final_amount 
    sorry 
    sorry :=
sorry

end NUMINAMATH_GPT_donation_percentage_correct_l1353_135311


namespace NUMINAMATH_GPT_real_part_of_z1_is_zero_l1353_135326

-- Define the imaginary unit i with its property
def i := Complex.I

-- Define z1 using the given expression
noncomputable def z1 := (1 - 2 * i) / (2 + i^5)

-- State the theorem about the real part of z1
theorem real_part_of_z1_is_zero : z1.re = 0 :=
by
  sorry

end NUMINAMATH_GPT_real_part_of_z1_is_zero_l1353_135326


namespace NUMINAMATH_GPT_rice_cake_slices_length_l1353_135301

noncomputable def slice_length (cake_length : ℝ) (num_cakes : ℕ) (overlap : ℝ) (num_slices : ℕ) : ℝ :=
  let total_original_length := num_cakes * cake_length
  let total_overlap := (num_cakes - 1) * overlap
  let actual_length := total_original_length - total_overlap
  actual_length / num_slices

theorem rice_cake_slices_length : 
  slice_length 2.7 5 0.3 6 = 2.05 :=
by
  sorry

end NUMINAMATH_GPT_rice_cake_slices_length_l1353_135301


namespace NUMINAMATH_GPT_cars_on_wednesday_more_than_monday_l1353_135307

theorem cars_on_wednesday_more_than_monday:
  let cars_tuesday := 25
  let cars_monday := 0.8 * cars_tuesday
  let cars_thursday := 10
  let cars_friday := 10
  let cars_saturday := 5
  let cars_sunday := 5
  let total_cars := 97
  ∃ (cars_wednesday : ℝ), cars_wednesday - cars_monday = 2 :=
by
  sorry

end NUMINAMATH_GPT_cars_on_wednesday_more_than_monday_l1353_135307


namespace NUMINAMATH_GPT_abs_neg_three_l1353_135315

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end NUMINAMATH_GPT_abs_neg_three_l1353_135315


namespace NUMINAMATH_GPT_length_of_PR_l1353_135365

theorem length_of_PR (x y : ℝ) (h₁ : x^2 + y^2 = 250) : 
  ∃ PR : ℝ, PR = 10 * Real.sqrt 5 :=
by
  use Real.sqrt (2 * (x^2 + y^2))
  sorry

end NUMINAMATH_GPT_length_of_PR_l1353_135365


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1353_135335

theorem solution_set_of_inequality (x : ℝ) :
  2 * x ≤ -1 → x > -1 → -1 < x ∧ x ≤ -1 / 2 :=
by
  intro h1 h2
  have h3 : x ≤ -1 / 2 := by linarith
  exact ⟨h2, h3⟩

end NUMINAMATH_GPT_solution_set_of_inequality_l1353_135335


namespace NUMINAMATH_GPT_perimeter_of_figure_l1353_135359

def side_length : ℕ := 1
def num_vertical_stacks : ℕ := 2
def num_squares_per_stack : ℕ := 3
def gap_between_stacks : ℕ := 1
def squares_on_top : ℕ := 3
def squares_on_bottom : ℕ := 2

theorem perimeter_of_figure : 
  (2 * side_length * squares_on_top) + (2 * side_length * squares_on_bottom) + 
  (2 * num_squares_per_stack * num_vertical_stacks) + (2 * num_squares_per_stack * squares_on_top)
  = 22 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_figure_l1353_135359


namespace NUMINAMATH_GPT_not_square_of_expression_l1353_135361

theorem not_square_of_expression (n : ℕ) (h : n > 0) : ∀ k : ℕ, (4 * n^2 + 4 * n + 4 ≠ k^2) :=
by
  sorry

end NUMINAMATH_GPT_not_square_of_expression_l1353_135361


namespace NUMINAMATH_GPT_area_of_inscribed_rectangle_l1353_135357

theorem area_of_inscribed_rectangle
  (s : ℕ) (R_area : ℕ)
  (h1 : s = 4) 
  (h2 : 2 * 4 + 1 * 1 + R_area = s * s) :
  R_area = 7 :=
by
  sorry

end NUMINAMATH_GPT_area_of_inscribed_rectangle_l1353_135357


namespace NUMINAMATH_GPT_Piper_gym_sessions_l1353_135391

theorem Piper_gym_sessions
  (start_on_monday : Bool)
  (alternate_except_sunday : (∀ (n : ℕ), n % 2 = 1 → n % 7 ≠ 0 → Bool))
  (sessions_over_on_wednesday : Bool)
  : ∃ (n : ℕ), n = 5 :=
by 
  sorry

end NUMINAMATH_GPT_Piper_gym_sessions_l1353_135391


namespace NUMINAMATH_GPT_smallest_four_digit_multiple_of_18_l1353_135331

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, n > 999 ∧ n < 10000 ∧ 18 ∣ n ∧ (∀ m : ℕ, m > 999 ∧ m < 10000 ∧ 18 ∣ m → n ≤ m) ∧ n = 1008 :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_multiple_of_18_l1353_135331


namespace NUMINAMATH_GPT_trig_expression_l1353_135386

theorem trig_expression (α : ℝ) (h : Real.tan α = 2) : 
  (3 * Real.sin α - Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 5 / 7 :=
by
  sorry

end NUMINAMATH_GPT_trig_expression_l1353_135386


namespace NUMINAMATH_GPT_third_median_length_l1353_135329

theorem third_median_length 
  (m_A m_B : ℝ) -- lengths of the first two medians
  (area : ℝ)   -- area of the triangle
  (h_median_A : m_A = 5) -- the first median is 5 inches
  (h_median_B : m_B = 8) -- the second median is 8 inches
  (h_area : area = 6 * Real.sqrt 15) -- the area of the triangle is 6√15 square inches
  : ∃ m_C : ℝ, m_C = Real.sqrt 31 := -- the length of the third median is √31
sorry

end NUMINAMATH_GPT_third_median_length_l1353_135329


namespace NUMINAMATH_GPT_JohnReceivedDiamonds_l1353_135399

def InitialDiamonds (Bill Sam : ℕ) (John : ℕ) : Prop :=
  Bill = 12 ∧ Sam = 12

def TheftEvents (BillBefore SamBefore JohnBefore BillAfter SamAfter JohnAfter : ℕ) : Prop :=
  BillAfter = BillBefore - 1 ∧ SamAfter = SamBefore - 1 ∧ JohnAfter = JohnBefore + 1

def AverageMassChange (Bill Sam John : ℕ) (BillMassChange SamMassChange JohnMassChange : ℤ) : Prop :=
  BillMassChange = Bill - 1 ∧ SamMassChange = Sam - 2 ∧ JohnMassChange = John + 4

def JohnInitialDiamonds (John : ℕ) : Prop :=
  Exists (fun x => 4 * x = 36)

theorem JohnReceivedDiamonds : ∃ John : ℕ, 
  InitialDiamonds 12 12 John ∧
  (∃ BillBefore SamBefore JohnBefore BillAfter SamAfter JohnAfter,
      TheftEvents BillBefore SamBefore JohnBefore BillAfter SamAfter JohnAfter ∧
      AverageMassChange 12 12 12 (-12) (-24) 36) →
  John = 9 :=
sorry

end NUMINAMATH_GPT_JohnReceivedDiamonds_l1353_135399


namespace NUMINAMATH_GPT_multiple_of_5_digits_B_l1353_135373

theorem multiple_of_5_digits_B (B : ℕ) : B = 0 ∨ B = 5 ↔ 23 * 10 + B % 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_5_digits_B_l1353_135373


namespace NUMINAMATH_GPT_cross_section_equilateral_triangle_l1353_135377

-- Definitions and conditions
structure Cone where
  r : ℝ -- radius of the base circle
  R : ℝ -- radius of the semicircle
  h : ℝ -- slant height

axiom lateral_surface_unfolded (c : Cone) : c.R = 2 * c.r

def CrossSectionIsEquilateral (c : Cone) : Prop :=
  (c.h ^ 2 = (c.r * c.h)) ∧ (c.h = 2 * c.r)

-- Problem statement with conditions
theorem cross_section_equilateral_triangle (c : Cone) (h_equals_diameter : c.R = 2 * c.r) : CrossSectionIsEquilateral c :=
by
  sorry

end NUMINAMATH_GPT_cross_section_equilateral_triangle_l1353_135377


namespace NUMINAMATH_GPT_weights_balance_l1353_135371

theorem weights_balance (k : ℕ) 
    (m n : ℕ → ℝ) 
    (h1 : ∀ i : ℕ, i < k → m i > n i) 
    (h2 : ∀ i : ℕ, i < k → ∃ j : ℕ, j ≠ i ∧ (m i + n j = n i + m j 
                                               ∨ m j + n i = n j + m i)) 
    : k = 1 ∨ k = 2 := 
by sorry

end NUMINAMATH_GPT_weights_balance_l1353_135371


namespace NUMINAMATH_GPT_price_per_slice_is_five_l1353_135324

-- Definitions based on the given conditions
def pies_sold := 9
def slices_per_pie := 4
def total_revenue := 180

-- Definition derived from given conditions
def total_slices := pies_sold * slices_per_pie

-- The theorem to prove
theorem price_per_slice_is_five :
  total_revenue / total_slices = 5 :=
by
  sorry

end NUMINAMATH_GPT_price_per_slice_is_five_l1353_135324


namespace NUMINAMATH_GPT_beads_removed_l1353_135342

def total_beads (blue yellow : Nat) : Nat := blue + yellow

def beads_per_part (total : Nat) (parts : Nat) : Nat := total / parts

def beads_remaining (per_part : Nat) (removed : Nat) : Nat := per_part - removed

def doubled_beads (remaining : Nat) : Nat := 2 * remaining

theorem beads_removed {x : Nat} 
  (blue : Nat) (yellow : Nat) (parts : Nat) (final_per_part : Nat) :
  total_beads blue yellow = 39 →
  parts = 3 →
  beads_per_part 39 parts = 13 →
  doubled_beads (beads_remaining 13 x) = 6 →
  x = 10 := by
  sorry

end NUMINAMATH_GPT_beads_removed_l1353_135342


namespace NUMINAMATH_GPT_rice_grains_difference_l1353_135381

theorem rice_grains_difference : 
  3^15 - (3^1 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7 + 3^8 + 3^9 + 3^10) = 14260335 := 
by
  sorry

end NUMINAMATH_GPT_rice_grains_difference_l1353_135381


namespace NUMINAMATH_GPT_triangle_ABC_l1353_135376

theorem triangle_ABC (a b c : ℝ) (A B C : ℝ)
  (h1 : a + b = 5)
  (h2 : c = Real.sqrt 7)
  (h3 : 4 * (Real.sin ((A + B) / 2))^2 - Real.cos (2 * C) = 7 / 2) :
  (C = Real.pi / 3)
  ∧ (1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_triangle_ABC_l1353_135376


namespace NUMINAMATH_GPT_bells_ring_together_l1353_135308

theorem bells_ring_together (church school day_care library noon : ℕ) :
  church = 18 ∧ school = 24 ∧ day_care = 30 ∧ library = 35 ∧ noon = 0 →
  ∃ t : ℕ, t = 2520 ∧ ∀ n, (t - noon) % n = 0 := by
  sorry

end NUMINAMATH_GPT_bells_ring_together_l1353_135308


namespace NUMINAMATH_GPT_average_marks_physics_mathematics_l1353_135372

theorem average_marks_physics_mathematics {P C M : ℕ} (h1 : P + C + M = 180) (h2 : P = 140) (h3 : P + C = 140) : 
  (P + M) / 2 = 90 := by
  sorry

end NUMINAMATH_GPT_average_marks_physics_mathematics_l1353_135372


namespace NUMINAMATH_GPT_equation_of_line_through_points_l1353_135346

-- Definitions for the problem conditions
def point1 : ℝ × ℝ := (-1, 2)
def point2 : ℝ × ℝ := (-3, -2)

-- The theorem stating the equation of the line passing through the given points
theorem equation_of_line_through_points :
  ∃ a b c : ℝ, (a * point1.1 + b * point1.2 + c = 0) ∧ (a * point2.1 + b * point2.2 + c = 0) ∧ 
             (a = 2) ∧ (b = -1) ∧ (c = 4) :=
by
  sorry

end NUMINAMATH_GPT_equation_of_line_through_points_l1353_135346


namespace NUMINAMATH_GPT_gift_cost_l1353_135392

def ErikaSavings : ℕ := 155
def CakeCost : ℕ := 25
def LeftOver : ℕ := 5

noncomputable def CostOfGift (RickSavings : ℕ) : ℕ :=
  2 * RickSavings

theorem gift_cost (RickSavings : ℕ)
  (hRick : RickSavings = CostOfGift RickSavings / 2)
  (hTotal : ErikaSavings + RickSavings = CostOfGift RickSavings + CakeCost + LeftOver) :
  CostOfGift RickSavings = 250 :=
by
  sorry

end NUMINAMATH_GPT_gift_cost_l1353_135392


namespace NUMINAMATH_GPT_tomatoes_ruined_percentage_l1353_135358

-- The definitions from the problem conditions
def tomato_cost_per_pound : ℝ := 0.80
def tomato_selling_price_per_pound : ℝ := 0.977777777777778
def desired_profit_percent : ℝ := 0.10
def revenue_equal_cost_plus_profit_cost_fraction : ℝ := (tomato_cost_per_pound + (tomato_cost_per_pound * desired_profit_percent))

-- The theorem stating the problem and the expected result
theorem tomatoes_ruined_percentage :
  ∀ (W : ℝ) (P : ℝ),
  (0.977777777777778 * (1 - P / 100) * W = (0.80 * W + 0.08 * W)) →
  P = 10.00000000000001 :=
by
  intros W P h
  have eq1 : 0.977777777777778 * (1 - P / 100) = 0.88 := sorry
  have eq2 : 1 - P / 100 = 0.8999999999999999 := sorry
  have eq3 : P / 100 = 0.1000000000000001 := sorry
  exact sorry

end NUMINAMATH_GPT_tomatoes_ruined_percentage_l1353_135358


namespace NUMINAMATH_GPT_total_marbles_l1353_135385

theorem total_marbles (r b y : ℕ) (h_ratio : 2 * b = 3 * r) (h_ratio_alt : 4 * b = 3 * y) (h_blue_marbles : b = 24) : r + b + y = 72 :=
by
  -- By assumption, b = 24
  have h1 : b = 24 := h_blue_marbles

  -- We have the ratios 2b = 3r and 4b = 3y
  have h2 : 2 * b = 3 * r := h_ratio
  have h3 : 4 * b = 3 * y := h_ratio_alt

  -- solved by given conditions 
  sorry

end NUMINAMATH_GPT_total_marbles_l1353_135385


namespace NUMINAMATH_GPT_problem_a_problem_b_problem_c_l1353_135351

theorem problem_a : (7 * (2 / 3) + 16 * (5 / 12)) = 11.3333 := by
  sorry

theorem problem_b : (5 - (2 / (5 / 3))) = 3.8 := by
  sorry

theorem problem_c : (1 + 2 / (1 + 3 / (1 + 4))) = 2.25 := by
  sorry

end NUMINAMATH_GPT_problem_a_problem_b_problem_c_l1353_135351


namespace NUMINAMATH_GPT_randy_initial_money_l1353_135387

/--
Initially, Randy had an unknown amount of money. He was given $2000 by Smith and $900 by Michelle.
After that, Randy gave Sally a 1/4th of his total money after which he gave Jake and Harry $800 and $500 respectively.
If Randy is left with $5500 after all the transactions, prove that Randy initially had $6166.67.
-/
theorem randy_initial_money (X : ℝ) :
  (3/4 * (X + 2000 + 900) - 1300 = 5500) -> (X = 6166.67) :=
by
  sorry

end NUMINAMATH_GPT_randy_initial_money_l1353_135387


namespace NUMINAMATH_GPT_cost_of_pack_of_socks_is_5_l1353_135330

-- Conditions definitions
def shirt_price : ℝ := 12.00
def short_price : ℝ := 15.00
def trunks_price : ℝ := 14.00
def shirts_count : ℕ := 3
def shorts_count : ℕ := 2
def total_bill : ℝ := 102.00
def total_known_cost : ℝ := 3 * shirt_price + 2 * short_price + trunks_price

-- Definition of the problem statement
theorem cost_of_pack_of_socks_is_5 (S : ℝ) : total_bill = total_known_cost + S + 0.2 * (total_known_cost + S) → S = 5 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_pack_of_socks_is_5_l1353_135330


namespace NUMINAMATH_GPT_work_finished_days_earlier_l1353_135345

theorem work_finished_days_earlier
  (D : ℕ) (M : ℕ) (A : ℕ) (Work : ℕ) (D_new : ℕ) (E : ℕ)
  (hD : D = 8)
  (hM : M = 30)
  (hA : A = 10)
  (hWork : Work = M * D)
  (hTotalWork : Work = 240)
  (hD_new : D_new = Work / (M + A))
  (hDnew_calculated : D_new = 6)
  (hE : E = D - D_new)
  (hE_calculated : E = 2) : 
  E = 2 :=
by
  sorry

end NUMINAMATH_GPT_work_finished_days_earlier_l1353_135345


namespace NUMINAMATH_GPT_solve_for_x_l1353_135383

theorem solve_for_x (x : ℝ) (h : (2 / 3 - 1 / 4) = 4 / x) : x = 48 / 5 :=
by sorry

end NUMINAMATH_GPT_solve_for_x_l1353_135383


namespace NUMINAMATH_GPT_inequality_solution_set_l1353_135325

theorem inequality_solution_set (x : ℝ) : ((x - 1) * (x^2 - x + 1) > 0) ↔ (x > 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1353_135325


namespace NUMINAMATH_GPT_number_of_points_on_line_l1353_135390

theorem number_of_points_on_line (a b c d : ℕ) (h1 : a * b = 80) (h2 : c * d = 90) (h3 : a + b = c + d) :
  a + b + 1 = 22 :=
sorry

end NUMINAMATH_GPT_number_of_points_on_line_l1353_135390


namespace NUMINAMATH_GPT_proving_four_digit_number_l1353_135394

def distinct (a b c d : Nat) : Prop :=
a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def same_parity (x y : Nat) : Prop :=
(x % 2 = 0 ∧ y % 2 = 0) ∨ (x % 2 = 1 ∧ y % 2 = 1)

def different_parity (x y : Nat) : Prop :=
¬same_parity x y

theorem proving_four_digit_number :
  ∃ (A B C D : Nat),
    distinct A B C D ∧
    (different_parity A B → B ≠ 4) ∧
    (different_parity B C → C ≠ 3) ∧
    (different_parity C D → D ≠ 2) ∧
    (different_parity D A → A ≠ 1) ∧
    A + D < B + C ∧
    1000 * A + 100 * B + 10 * C + D = 2341 :=
by
  sorry

end NUMINAMATH_GPT_proving_four_digit_number_l1353_135394


namespace NUMINAMATH_GPT_necklace_ratio_l1353_135352

variable {J Q H : ℕ}

theorem necklace_ratio (h1 : H = J + 5) (h2 : H = 25) (h3 : H = Q + 15) : Q / J = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_necklace_ratio_l1353_135352


namespace NUMINAMATH_GPT_daughter_weight_l1353_135354

def main : IO Unit :=
  IO.println s!"The weight of the daughter is 50 kg."

theorem daughter_weight :
  ∀ (G D C : ℝ), G + D + C = 110 → D + C = 60 → C = (1/5) * G → D = 50 :=
by
  intros G D C h1 h2 h3
  sorry

end NUMINAMATH_GPT_daughter_weight_l1353_135354


namespace NUMINAMATH_GPT_house_trailer_payment_difference_l1353_135317

-- Define the costs and periods
def cost_house : ℕ := 480000
def cost_trailer : ℕ := 120000
def loan_period_years : ℕ := 20
def months_per_year : ℕ := 12

-- Calculate total months
def total_months : ℕ := loan_period_years * months_per_year

-- Calculate monthly payments
def monthly_payment_house : ℕ := cost_house / total_months
def monthly_payment_trailer : ℕ := cost_trailer / total_months

-- Theorem stating the difference in monthly payments
theorem house_trailer_payment_difference :
  monthly_payment_house - monthly_payment_trailer = 1500 := by sorry

end NUMINAMATH_GPT_house_trailer_payment_difference_l1353_135317


namespace NUMINAMATH_GPT_gcd_153_119_l1353_135318

theorem gcd_153_119 : Nat.gcd 153 119 = 17 :=
by
  sorry

end NUMINAMATH_GPT_gcd_153_119_l1353_135318


namespace NUMINAMATH_GPT_original_price_l1353_135388

theorem original_price (sale_price gain_percent : ℕ) (h_sale : sale_price = 130) (h_gain : gain_percent = 30) : 
    ∃ P : ℕ, (P * (1 + gain_percent / 100)) = sale_price := 
by
  use 100
  rw [h_sale, h_gain]
  norm_num
  sorry

end NUMINAMATH_GPT_original_price_l1353_135388


namespace NUMINAMATH_GPT_systematic_sampling_first_group_number_l1353_135370

-- Given conditions
def total_students := 160
def group_size := 8
def groups := total_students / group_size
def number_in_16th_group := 126

-- Theorem Statement
theorem systematic_sampling_first_group_number :
  ∃ x : ℕ, (120 + x = number_in_16th_group) ∧ x = 6 :=
by
  -- Proof can be filled here
  sorry

end NUMINAMATH_GPT_systematic_sampling_first_group_number_l1353_135370


namespace NUMINAMATH_GPT_harry_worked_16_hours_l1353_135360

-- Define the given conditions
def harrys_pay_first_30_hours (x : ℝ) : ℝ := 30 * x
def harrys_pay_additional_hours (x H : ℝ) : ℝ := (H - 30) * 2 * x
def james_pay_first_40_hours (x : ℝ) : ℝ := 40 * x
def james_pay_additional_hour (x : ℝ) : ℝ := 2 * x
def james_total_hours : ℝ := 41

-- Given that Harry and James are paid the same amount 
-- Prove that Harry worked 16 hours last week
theorem harry_worked_16_hours (x H : ℝ) 
  (h1 : harrys_pay_first_30_hours x + harrys_pay_additional_hours x H = james_pay_first_40_hours x + james_pay_additional_hour x) 
  : H = 16 :=
by
  sorry

end NUMINAMATH_GPT_harry_worked_16_hours_l1353_135360


namespace NUMINAMATH_GPT_sin_of_tan_l1353_135367

theorem sin_of_tan (A : ℝ) (hA_acute : 0 < A ∧ A < π / 2) (h_tan_A : Real.tan A = (Real.sqrt 2) / 3) :
  Real.sin A = (Real.sqrt 22) / 11 :=
sorry

end NUMINAMATH_GPT_sin_of_tan_l1353_135367


namespace NUMINAMATH_GPT_proof_candle_burn_l1353_135366

noncomputable def candle_burn_proof : Prop :=
∃ (t : ℚ),
  (t = 40 / 11) ∧
  (∀ (H_1 H_2 : ℚ → ℚ),
    (∀ t, H_1 t = 1 - t / 5) ∧
    (∀ t, H_2 t = 1 - t / 4) →
    ∃ (t : ℚ), ((1 - t / 5) = 3 * (1 - t / 4)) ∧ (t = 40 / 11))

theorem proof_candle_burn : candle_burn_proof :=
sorry

end NUMINAMATH_GPT_proof_candle_burn_l1353_135366


namespace NUMINAMATH_GPT_log_equation_solution_l1353_135333

theorem log_equation_solution (a b x : ℝ) (h : 5 * (Real.log x / Real.log b) ^ 2 + 2 * (Real.log x / Real.log a) ^ 2 = 10 * (Real.log x) ^ 2 / (Real.log a * Real.log b)) :
    b = a ^ (1 + Real.sqrt 15 / 5) ∨ b = a ^ (1 - Real.sqrt 15 / 5) :=
sorry

end NUMINAMATH_GPT_log_equation_solution_l1353_135333


namespace NUMINAMATH_GPT_calculate_value_of_expression_l1353_135310

theorem calculate_value_of_expression :
  (2523 - 2428)^2 / 121 = 75 :=
by
  -- calculation steps here
  sorry

end NUMINAMATH_GPT_calculate_value_of_expression_l1353_135310


namespace NUMINAMATH_GPT_trivia_team_points_l1353_135398

theorem trivia_team_points : 
  let member1_points := 8
  let member2_points := 12
  let member3_points := 9
  let member4_points := 5
  let member5_points := 10
  let member6_points := 7
  let member7_points := 14
  let member8_points := 11
  (member1_points + member2_points + member3_points + member4_points + member5_points + member6_points + member7_points + member8_points) = 76 :=
by
  let member1_points := 8
  let member2_points := 12
  let member3_points := 9
  let member4_points := 5
  let member5_points := 10
  let member6_points := 7
  let member7_points := 14
  let member8_points := 11
  sorry

end NUMINAMATH_GPT_trivia_team_points_l1353_135398


namespace NUMINAMATH_GPT_marcy_pets_cat_time_l1353_135336

theorem marcy_pets_cat_time (P : ℝ) (h1 : P + (1/3)*P = 16) : P = 12 :=
by
  sorry

end NUMINAMATH_GPT_marcy_pets_cat_time_l1353_135336


namespace NUMINAMATH_GPT_parabola_directrix_l1353_135347

theorem parabola_directrix (p : ℝ) (h : p > 0) (h_directrix : -p / 2 = -4) : p = 8 :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_l1353_135347


namespace NUMINAMATH_GPT_probability_non_first_class_product_l1353_135368

theorem probability_non_first_class_product (P_A P_B P_C : ℝ) (hA : P_A = 0.65) (hB : P_B = 0.2) (hC : P_C = 0.1) : 1 - P_A = 0.35 :=
by
  sorry

end NUMINAMATH_GPT_probability_non_first_class_product_l1353_135368


namespace NUMINAMATH_GPT_unit_stratified_sampling_l1353_135350

theorem unit_stratified_sampling 
  (elderly : ℕ) (middle_aged : ℕ) (young : ℕ) (selected_elderly : ℕ)
  (total : ℕ) (n : ℕ)
  (h1 : elderly = 27)
  (h2 : middle_aged = 54)
  (h3 : young = 81)
  (h4 : selected_elderly = 3)
  (h5 : total = elderly + middle_aged + young)
  (h6 : 3 / 27 = selected_elderly / elderly)
  (h7 : n / total = selected_elderly / elderly) : 
  n = 18 := 
by
  sorry

end NUMINAMATH_GPT_unit_stratified_sampling_l1353_135350


namespace NUMINAMATH_GPT_cos_C_values_l1353_135341

theorem cos_C_values (sin_A : ℝ) (cos_B : ℝ) (cos_C : ℝ) 
  (h1 : sin_A = 4 / 5) 
  (h2 : cos_B = 12 / 13) 
  : cos_C = -16 / 65 ∨ cos_C = 56 / 65 :=
by
  sorry

end NUMINAMATH_GPT_cos_C_values_l1353_135341


namespace NUMINAMATH_GPT_jordon_machine_input_l1353_135393

theorem jordon_machine_input (x : ℝ) : (3 * x - 6) / 2 + 9 = 27 → x = 14 := 
by
  sorry

end NUMINAMATH_GPT_jordon_machine_input_l1353_135393


namespace NUMINAMATH_GPT_simplify_329_mul_101_simplify_54_mul_98_plus_46_mul_98_simplify_98_mul_125_simplify_37_mul_29_plus_37_l1353_135306

theorem simplify_329_mul_101 : 329 * 101 = 33229 := by
  sorry

theorem simplify_54_mul_98_plus_46_mul_98 : 54 * 98 + 46 * 98 = 9800 := by
  sorry

theorem simplify_98_mul_125 : 98 * 125 = 12250 := by
  sorry

theorem simplify_37_mul_29_plus_37 : 37 * 29 + 37 = 1110 := by
  sorry

end NUMINAMATH_GPT_simplify_329_mul_101_simplify_54_mul_98_plus_46_mul_98_simplify_98_mul_125_simplify_37_mul_29_plus_37_l1353_135306


namespace NUMINAMATH_GPT_product_of_numbers_l1353_135304

theorem product_of_numbers (x y : ℤ) (h1 : x + y = 37) (h2 : x - y = 5) : x * y = 336 := by
  sorry

end NUMINAMATH_GPT_product_of_numbers_l1353_135304


namespace NUMINAMATH_GPT_quadratic_roots_condition_l1353_135384

theorem quadratic_roots_condition (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1*x1 + m*x1 + 4 = 0 ∧ x2*x2 + m*x2 + 4 = 0) →
  m ≤ -4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_condition_l1353_135384


namespace NUMINAMATH_GPT_greatest_three_digit_base_nine_divisible_by_seven_l1353_135380

/-- Define the problem setup -/
def greatest_three_digit_base_nine := 8 * 9^2 + 8 * 9 + 8

/-- Prove the greatest 3-digit base 9 positive integer that is divisible by 7 -/
theorem greatest_three_digit_base_nine_divisible_by_seven : 
  ∃ n : ℕ, n = greatest_three_digit_base_nine ∧ n % 7 = 0 ∧ (8 * 9^2 + 8 * 9 + 8) = 728 := by 
  sorry

end NUMINAMATH_GPT_greatest_three_digit_base_nine_divisible_by_seven_l1353_135380


namespace NUMINAMATH_GPT_wall_width_l1353_135353

theorem wall_width (V h l w : ℝ) (h_cond : h = 6 * w) (l_cond : l = 42 * w) (vol_cond : 252 * w^3 = 129024) : w = 8 := 
by
  -- Proof is omitted; required to produce lean statement only
  sorry

end NUMINAMATH_GPT_wall_width_l1353_135353


namespace NUMINAMATH_GPT_dividends_CEO_2018_l1353_135379

theorem dividends_CEO_2018 (Revenue Expenses Tax_rate Loan_payment_per_month : ℝ) 
  (Number_of_shares : ℕ) (CEO_share_percentage : ℝ)
  (hRevenue : Revenue = 2500000) 
  (hExpenses : Expenses = 1576250)
  (hTax_rate : Tax_rate = 0.2)
  (hLoan_payment_per_month : Loan_payment_per_month = 25000)
  (hNumber_of_shares : Number_of_shares = 1600)
  (hCEO_share_percentage : CEO_share_percentage = 0.35) :
  CEO_share_percentage * ((Revenue - Expenses) * (1 - Tax_rate) - Loan_payment_per_month * 12) / Number_of_shares * Number_of_shares = 153440 :=
sorry

end NUMINAMATH_GPT_dividends_CEO_2018_l1353_135379


namespace NUMINAMATH_GPT_calculate_power_expression_l1353_135338

theorem calculate_power_expression : 4 ^ 2009 * (-0.25) ^ 2008 - 1 = 3 := 
by
  -- steps and intermediate calculations go here
  sorry

end NUMINAMATH_GPT_calculate_power_expression_l1353_135338


namespace NUMINAMATH_GPT_value_of_N_l1353_135375

theorem value_of_N (N : ℕ) (h : (20 / 100) * N = (60 / 100) * 2500) : N = 7500 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_N_l1353_135375


namespace NUMINAMATH_GPT_average_of_solutions_l1353_135337

theorem average_of_solutions (a b : ℝ) :
  (∃ x1 x2 : ℝ, 3 * a * x1^2 - 6 * a * x1 + 2 * b = 0 ∧
                3 * a * x2^2 - 6 * a * x2 + 2 * b = 0 ∧
                x1 ≠ x2) →
  (1 + 1) / 2 = 1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_average_of_solutions_l1353_135337


namespace NUMINAMATH_GPT_max_value_of_function_l1353_135323

/-- Let y(x) = a^(2*x) + 2 * a^x - 1 for a positive real number a and x in [-1, 1].
    Prove that the maximum value of y on the interval [-1, 1] is 14 when a = 1/3 or a = 3. -/
theorem max_value_of_function (a : ℝ) (a_pos : 0 < a) (h : a = 1 / 3 ∨ a = 3) : 
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^(2*x) + 2 * a^x - 1 = 14 := 
sorry

end NUMINAMATH_GPT_max_value_of_function_l1353_135323


namespace NUMINAMATH_GPT_survey_no_preference_students_l1353_135369

theorem survey_no_preference_students (total_students pref_mac pref_both pref_windows : ℕ) 
    (h1 : total_students = 210) 
    (h2 : pref_mac = 60) 
    (h3 : pref_both = pref_mac / 3)
    (h4 : pref_windows = 40) : 
    total_students - (pref_mac + pref_both + pref_windows) = 90 :=
by
  sorry

end NUMINAMATH_GPT_survey_no_preference_students_l1353_135369
