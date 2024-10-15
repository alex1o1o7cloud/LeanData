import Mathlib

namespace NUMINAMATH_GPT_billiard_ball_returns_l2005_200505

theorem billiard_ball_returns
  (w h : ℕ)
  (launch_angle : ℝ)
  (reflect_angle : ℝ)
  (start_A : ℝ × ℝ)
  (h_w : w = 2021)
  (h_h : h = 4300)
  (h_launch : launch_angle = 45)
  (h_reflect : reflect_angle = 45)
  (h_in_rect : ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 2021 ∧ 0 ≤ y ∧ y ≤ 4300) :
  ∃ (bounces : ℕ), bounces = 294 :=
by
  sorry

end NUMINAMATH_GPT_billiard_ball_returns_l2005_200505


namespace NUMINAMATH_GPT_original_number_l2005_200585

theorem original_number (N : ℕ) (h : ∃ k : ℕ, N + 1 = 9 * k) : N = 8 :=
sorry

end NUMINAMATH_GPT_original_number_l2005_200585


namespace NUMINAMATH_GPT_exactly_three_correct_is_impossible_l2005_200528

theorem exactly_three_correct_is_impossible (n : ℕ) (hn : n = 5) (f : Fin n → Fin n) :
  (∃ S : Finset (Fin n), S.card = 3 ∧ ∀ i ∈ S, f i = i) → False :=
by
  intros h
  sorry

end NUMINAMATH_GPT_exactly_three_correct_is_impossible_l2005_200528


namespace NUMINAMATH_GPT_value_of_x_l2005_200575

theorem value_of_x 
  (x : ℚ) 
  (h₁ : 6 * x^2 + 19 * x - 7 = 0) 
  (h₂ : 18 * x^2 + 47 * x - 21 = 0) : 
  x = 1 / 3 := 
  sorry

end NUMINAMATH_GPT_value_of_x_l2005_200575


namespace NUMINAMATH_GPT_letters_into_mailboxes_l2005_200530

theorem letters_into_mailboxes (n m : ℕ) (h1 : n = 3) (h2 : m = 5) : m^n = 125 :=
by
  rw [h1, h2]
  exact rfl

end NUMINAMATH_GPT_letters_into_mailboxes_l2005_200530


namespace NUMINAMATH_GPT_factors_of_2550_have_more_than_3_factors_l2005_200587

theorem factors_of_2550_have_more_than_3_factors :
  ∃ n: ℕ, n = 5 ∧
    ∃ d: ℕ, d = 2550 ∧
    (∀ x < n, ∃ y: ℕ, y ∣ d ∧ (∃ z, z ∣ y ∧ z > 3)) :=
sorry

end NUMINAMATH_GPT_factors_of_2550_have_more_than_3_factors_l2005_200587


namespace NUMINAMATH_GPT_sequence_difference_constant_l2005_200542

theorem sequence_difference_constant :
  ∀ (x y : ℕ → ℕ), x 1 = 2 → y 1 = 1 →
  (∀ k, k > 1 → x k = 2 * x (k - 1) + 3 * y (k - 1)) →
  (∀ k, k > 1 → y k = x (k - 1) + 2 * y (k - 1)) →
  ∀ k, x k ^ 2 - 3 * y k ^ 2 = 1 :=
by
  -- Insert the proof steps here
  sorry

end NUMINAMATH_GPT_sequence_difference_constant_l2005_200542


namespace NUMINAMATH_GPT_find_difference_of_a_b_l2005_200511

noncomputable def a_b_are_relative_prime_and_positive (a b : ℕ) (hab_prime : Nat.gcd a b = 1) (ha_pos : a > 0) (hb_pos : b > 0) (h_gt : a > b) : Prop :=
  a ^ 3 - b ^ 3 = (131 / 5) * (a - b) ^ 3

theorem find_difference_of_a_b (a b : ℕ) 
  (hab_prime : Nat.gcd a b = 1) 
  (ha_pos : a > 0) 
  (hb_pos : b > 0) 
  (h_gt : a > b) 
  (h_eq : (a ^ 3 - b ^ 3 : ℚ) / (a - b) ^ 3 = 131 / 5) : 
  a - b = 7 :=
  sorry

end NUMINAMATH_GPT_find_difference_of_a_b_l2005_200511


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l2005_200529

variable {a : Nat → Real} -- Sequence a_n
variable {q : Real} -- Common ratio
variable (a1_pos : a 1 > 0) -- Condition a1 > 0

-- Definition of geometric sequence
def is_geometric_sequence (a : Nat → Real) (q : Real) : Prop :=
  ∀ n : Nat, a (n + 1) = a n * q

-- Definition of increasing sequence
def is_increasing_sequence (a : Nat → Real) : Prop :=
  ∀ n : Nat, a n < a (n + 1)

-- Theorem statement
theorem necessary_but_not_sufficient_condition (a : Nat → Real) (q : Real) (a1_pos : a 1 > 0) :
  is_geometric_sequence a q →
  is_increasing_sequence a →
  q > 0 ∧ ¬(q > 0 → is_increasing_sequence a) := by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l2005_200529


namespace NUMINAMATH_GPT_sequence_a_11_l2005_200516

theorem sequence_a_11 (a : ℕ → ℚ) (arithmetic_seq : ℕ → ℚ)
  (h1 : a 3 = 2)
  (h2 : a 7 = 1)
  (h_arith : ∀ n, arithmetic_seq n = 1 / (a n + 1))
  (arith_property : ∀ n, arithmetic_seq (n + 1) - arithmetic_seq n = arithmetic_seq (n + 2) - arithmetic_seq (n + 1)) :
  a 11 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_a_11_l2005_200516


namespace NUMINAMATH_GPT_tangent_line_to_ex_l2005_200581

theorem tangent_line_to_ex (b : ℝ) : (∃ x0 : ℝ, (∀ x : ℝ, (e^x - e^x0 - (x - x0) * e^x0 = 0) ↔ y = x + b)) → b = 1 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_to_ex_l2005_200581


namespace NUMINAMATH_GPT_first_bell_weight_l2005_200536

-- Given conditions from the problem
variable (x : ℕ) -- weight of the first bell in pounds
variable (total_weight : ℕ)

-- The condition as the sum of the weights
def bronze_weights (x total_weight : ℕ) : Prop :=
  x + 2 * x + 8 * 2 * x = total_weight

-- Prove that the weight of the first bell is 50 pounds given the total weight is 550 pounds
theorem first_bell_weight : bronze_weights x 550 → x = 50 := by
  intro h
  sorry

end NUMINAMATH_GPT_first_bell_weight_l2005_200536


namespace NUMINAMATH_GPT_platform_length_605_l2005_200571

noncomputable def length_of_platform (speed_kmh : ℕ) (accel : ℚ) (t_platform : ℚ) (t_man : ℚ) (dist_man_from_platform : ℚ) : ℚ :=
  let speed_ms := (speed_kmh : ℚ) * 1000 / 3600
  let distance_man := speed_ms * t_man + 0.5 * accel * t_man^2
  let train_length := distance_man - dist_man_from_platform
  let distance_platform := speed_ms * t_platform + 0.5 * accel * t_platform^2
  distance_platform - train_length

theorem platform_length_605 :
  length_of_platform 54 0.5 40 20 5 = 605 := by
  sorry

end NUMINAMATH_GPT_platform_length_605_l2005_200571


namespace NUMINAMATH_GPT_discount_percentage_l2005_200519

theorem discount_percentage (original_price sale_price : ℕ) (h₁ : original_price = 1200) (h₂ : sale_price = 1020) : 
  ((original_price - sale_price) * 100 / original_price : ℝ) = 15 :=
by
  sorry

end NUMINAMATH_GPT_discount_percentage_l2005_200519


namespace NUMINAMATH_GPT_identify_different_correlation_l2005_200593

-- Define the concept of correlation
inductive Correlation
| positive
| negative

-- Define the conditions for each option
def option_A : Correlation := Correlation.positive
def option_B : Correlation := Correlation.positive
def option_C : Correlation := Correlation.negative
def option_D : Correlation := Correlation.positive

-- The statement to prove
theorem identify_different_correlation :
  (option_A = Correlation.positive) ∧ 
  (option_B = Correlation.positive) ∧ 
  (option_D = Correlation.positive) ∧ 
  (option_C = Correlation.negative) := 
sorry

end NUMINAMATH_GPT_identify_different_correlation_l2005_200593


namespace NUMINAMATH_GPT_evaluation_of_expression_l2005_200518

theorem evaluation_of_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_GPT_evaluation_of_expression_l2005_200518


namespace NUMINAMATH_GPT_find_point_on_curve_l2005_200508

theorem find_point_on_curve :
  ∃ P : ℝ × ℝ, (P.1^3 - P.1 + 3 = P.2) ∧ (3 * P.1^2 - 1 = 2) ∧ (P = (1, 3) ∨ P = (-1, 3)) :=
sorry

end NUMINAMATH_GPT_find_point_on_curve_l2005_200508


namespace NUMINAMATH_GPT_find_quadruples_l2005_200561

def is_solution (x y z n : ℕ) : Prop :=
  x^2 + y^2 + z^2 + 1 = 2^n

theorem find_quadruples :
  ∀ x y z n : ℕ, is_solution x y z n ↔ 
  (x, y, z, n) = (1, 1, 1, 2) ∨
  (x, y, z, n) = (0, 0, 1, 1) ∨
  (x, y, z, n) = (0, 1, 0, 1) ∨
  (x, y, z, n) = (1, 0, 0, 1) ∨
  (x, y, z, n) = (0, 0, 0, 0) :=
by
  sorry

end NUMINAMATH_GPT_find_quadruples_l2005_200561


namespace NUMINAMATH_GPT_sequence_unbounded_l2005_200506

theorem sequence_unbounded 
  (a : ℕ → ℝ)
  (h1 : ∀ n, a n = |a (n + 1) - a (n + 2)|)
  (h2 : 0 < a 0)
  (h3 : 0 < a 1)
  (h4 : a 0 ≠ a 1) :
  ¬ ∃ M : ℝ, ∀ n, |a n| ≤ M := 
sorry

end NUMINAMATH_GPT_sequence_unbounded_l2005_200506


namespace NUMINAMATH_GPT_problem1_problem2_l2005_200507

-- Problem 1
theorem problem1 : 2 * Real.sqrt 12 * (Real.sqrt 3 / 4) / Real.sqrt 2 = (3 * Real.sqrt 2) / 2 :=
by sorry

-- Problem 2
theorem problem2 : (Real.sqrt 3 - Real.sqrt 2)^2 + (Real.sqrt 8 - Real.sqrt 3) * (2 * Real.sqrt 2 + Real.sqrt 3) = 10 - 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l2005_200507


namespace NUMINAMATH_GPT_eval_expression_l2005_200577

theorem eval_expression : 
  ( ( (476 * 100 + 424 * 100) * 2^3 - 4 * (476 * 100 * 424 * 100) ) * (376 - 150) ) / 250 = -7297340160 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l2005_200577


namespace NUMINAMATH_GPT_purely_imaginary_z_point_on_line_z_l2005_200520

-- Proof problem for (I)
theorem purely_imaginary_z (a : ℝ) (z : ℂ) (h : z = Complex.mk 0 (a+2)) 
: a = 2 :=
sorry

-- Proof problem for (II)
theorem point_on_line_z (a : ℝ) (x y : ℝ) (h1 : x = a^2-4) (h2 : y = a+2) (h3 : x + 2*y + 1 = 0) 
: a = -1 :=
sorry

end NUMINAMATH_GPT_purely_imaginary_z_point_on_line_z_l2005_200520


namespace NUMINAMATH_GPT_determine_f_when_alpha_l2005_200573

noncomputable def solves_functional_equation (f : ℝ → ℝ) (α : ℝ) : Prop :=
∀ (x y : ℝ), 0 < x → 0 < y → f (f x + y) = α * x + 1 / (f (1 / y))

theorem determine_f_when_alpha (α : ℝ) (f : ℝ → ℝ) :
  (α = 1 → ∀ x, 0 < x → f x = x) ∧ (α ≠ 1 → ∀ f, ¬ solves_functional_equation f α) := by
  sorry

end NUMINAMATH_GPT_determine_f_when_alpha_l2005_200573


namespace NUMINAMATH_GPT_deepak_present_age_l2005_200559

variable (R D : ℕ)

theorem deepak_present_age 
  (h1 : R + 22 = 26) 
  (h2 : R / D = 4 / 3) : 
  D = 3 := 
sorry

end NUMINAMATH_GPT_deepak_present_age_l2005_200559


namespace NUMINAMATH_GPT_interval_of_increase_of_f_l2005_200562

noncomputable def f (x : ℝ) := Real.logb (0.5) (x - x^2)

theorem interval_of_increase_of_f :
  ∀ x : ℝ, x ∈ Set.Ioo (1/2) 1 → ∃ ε > 0, ∀ y : ℝ, y ∈ Set.Ioo (x - ε) (x + ε) → f y > f x :=
  by
    sorry

end NUMINAMATH_GPT_interval_of_increase_of_f_l2005_200562


namespace NUMINAMATH_GPT_probability_of_four_digit_number_divisible_by_3_l2005_200570

def digits : List ℕ := [0, 1, 2, 3, 4, 5]

def count_valid_four_digit_numbers : Int :=
  let all_digits := digits
  let total_four_digit_numbers := 180
  let valid_four_digit_numbers := 96
  total_four_digit_numbers

def probability_divisible_by_3 : ℚ :=
  (96 : ℚ) / (180 : ℚ)

theorem probability_of_four_digit_number_divisible_by_3 :
  probability_divisible_by_3 = 8 / 15 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_four_digit_number_divisible_by_3_l2005_200570


namespace NUMINAMATH_GPT_miriam_cleaning_room_time_l2005_200553

theorem miriam_cleaning_room_time
  (laundry_time : Nat := 30)
  (bathroom_time : Nat := 15)
  (homework_time : Nat := 40)
  (total_time : Nat := 120) :
  ∃ room_time : Nat, laundry_time + bathroom_time + homework_time + room_time = total_time ∧
                  room_time = 35 := by
  sorry

end NUMINAMATH_GPT_miriam_cleaning_room_time_l2005_200553


namespace NUMINAMATH_GPT_expected_value_of_win_is_2_5_l2005_200527

noncomputable def expected_value_of_win : ℚ := 
  (1/6) * (6 - 1) + (1/6) * (6 - 2) + (1/6) * (6 - 3) + 
  (1/6) * (6 - 4) + (1/6) * (6 - 5) + (1/6) * (6 - 6)

theorem expected_value_of_win_is_2_5 : expected_value_of_win = 5 / 2 := 
by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_expected_value_of_win_is_2_5_l2005_200527


namespace NUMINAMATH_GPT_spell_AMCB_paths_equals_24_l2005_200541

def central_A_reachable_M : Nat := 4
def M_reachable_C : Nat := 2
def C_reachable_B : Nat := 3

theorem spell_AMCB_paths_equals_24 :
  central_A_reachable_M * M_reachable_C * C_reachable_B = 24 := by
  sorry

end NUMINAMATH_GPT_spell_AMCB_paths_equals_24_l2005_200541


namespace NUMINAMATH_GPT_total_heartbeats_correct_l2005_200588

-- Define the given conditions
def heartbeats_per_minute : ℕ := 160
def pace_per_mile : ℕ := 6
def race_distance : ℕ := 30

-- Define the total heartbeats during the race
def total_heartbeats_during_race : ℕ :=
  pace_per_mile * race_distance * heartbeats_per_minute

-- Theorem stating the mathematically equivalent proof problem
theorem total_heartbeats_correct :
  total_heartbeats_during_race = 28800 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_total_heartbeats_correct_l2005_200588


namespace NUMINAMATH_GPT_probability_of_winning_is_correct_l2005_200502

theorem probability_of_winning_is_correct :
  ∀ (PWin PLoss PTie : ℚ),
    PLoss = 5/12 →
    PTie = 1/6 →
    PWin + PLoss + PTie = 1 →
    PWin = 5/12 := 
by
  intros PWin PLoss PTie hLoss hTie hSum
  sorry

end NUMINAMATH_GPT_probability_of_winning_is_correct_l2005_200502


namespace NUMINAMATH_GPT_double_given_number_l2005_200540

def given_number : ℝ := 1.2 * 10^6

def double_number (x: ℝ) : ℝ := x * 2

theorem double_given_number : double_number given_number = 2.4 * 10^6 :=
by sorry

end NUMINAMATH_GPT_double_given_number_l2005_200540


namespace NUMINAMATH_GPT_incenter_circumcenter_coincide_if_and_only_if_skew_edges_equal_l2005_200564

structure Tetrahedron (α : Type*) [MetricSpace α] :=
(A B C D : α)

def Incenter {α : Type*} [MetricSpace α] (T : Tetrahedron α) : α := sorry
def Circumcenter {α : Type*} [MetricSpace α] (T : Tetrahedron α) : α := sorry

def equidistant_from_faces {α : Type*} [MetricSpace α] (T : Tetrahedron α) (I : α) : Prop := sorry
def equidistant_from_vertices {α : Type*} [MetricSpace α] (T : Tetrahedron α) (O : α) : Prop := sorry
def skew_edges_equal {α : Type*} [MetricSpace α] (T : Tetrahedron α) : Prop := sorry

theorem incenter_circumcenter_coincide_if_and_only_if_skew_edges_equal
  {α : Type*} [MetricSpace α] (T : Tetrahedron α) :
  (∃ I, ∃ O, (Incenter T = I) ∧ (Circumcenter T = O) ∧ 
            (equidistant_from_faces T I) ∧ (equidistant_from_vertices T O)) ↔ (skew_edges_equal T) := 
sorry

end NUMINAMATH_GPT_incenter_circumcenter_coincide_if_and_only_if_skew_edges_equal_l2005_200564


namespace NUMINAMATH_GPT_john_paid_more_l2005_200521

-- Define the required variables
def original_price : ℝ := 84.00000000000009
def discount_rate : ℝ := 0.10
def tip_rate : ℝ := 0.15

-- Define John and Jane's payments
def discounted_price : ℝ := original_price * (1 - discount_rate)
def johns_tip : ℝ := tip_rate * original_price
def johns_total_payment : ℝ := original_price + johns_tip
def janes_tip : ℝ := tip_rate * discounted_price
def janes_total_payment : ℝ := discounted_price + janes_tip

-- Calculate the difference
def payment_difference : ℝ := johns_total_payment - janes_total_payment

-- Statement to prove the payment difference equals $9.66
theorem john_paid_more : payment_difference = 9.66 := by
  sorry

end NUMINAMATH_GPT_john_paid_more_l2005_200521


namespace NUMINAMATH_GPT_equal_points_per_person_l2005_200568

theorem equal_points_per_person :
  let blue_eggs := 12
  let blue_points := 2
  let pink_eggs := 5
  let pink_points := 3
  let golden_eggs := 3
  let golden_points := 5
  let total_people := 4
  (blue_eggs * blue_points + pink_eggs * pink_points + golden_eggs * golden_points) / total_people = 13 :=
by
  -- place the steps based on the conditions and calculations
  sorry

end NUMINAMATH_GPT_equal_points_per_person_l2005_200568


namespace NUMINAMATH_GPT_average_stamps_per_day_l2005_200554

theorem average_stamps_per_day :
  let a1 := 8
  let d := 8
  let n := 6
  let stamps_collected : Fin n → ℕ := λ i => a1 + i * d
  -- sum the stamps collected over six days
  let S := List.sum (List.ofFn stamps_collected)
  -- calculate average
  let average := S / n
  average = 28 :=
by sorry

end NUMINAMATH_GPT_average_stamps_per_day_l2005_200554


namespace NUMINAMATH_GPT_find_unique_p_l2005_200524

theorem find_unique_p (p : ℝ) (h1 : p ≠ 0) : (∀ x : ℝ, p * x^2 - 10 * x + 2 = 0 → p = 12.5) :=
by sorry

end NUMINAMATH_GPT_find_unique_p_l2005_200524


namespace NUMINAMATH_GPT_sum_of_four_smallest_divisors_eq_11_l2005_200557

noncomputable def common_divisors_sum : ℤ :=
  let common_divisors := [1, 2, 3, 5, 6, 10, 15, 30]
  let smallest_four := common_divisors.take 4
  smallest_four.sum

theorem sum_of_four_smallest_divisors_eq_11 :
  common_divisors_sum = 11 := by
  sorry

end NUMINAMATH_GPT_sum_of_four_smallest_divisors_eq_11_l2005_200557


namespace NUMINAMATH_GPT_work_completion_days_l2005_200515

theorem work_completion_days
  (A B : ℝ)
  (h1 : A + B = 1 / 12)
  (h2 : A = 1 / 20)
  : 1 / (A + B / 2) = 15 :=
by 
  sorry

end NUMINAMATH_GPT_work_completion_days_l2005_200515


namespace NUMINAMATH_GPT_least_number_l2005_200512

theorem least_number (n : ℕ) (h1 : n % 38 = 1) (h2 : n % 3 = 1) : n = 115 :=
sorry

end NUMINAMATH_GPT_least_number_l2005_200512


namespace NUMINAMATH_GPT_intersection_of_complements_l2005_200532

variable (U : Set ℕ)
variable (A : Set ℕ)
variable (B : Set ℕ)

def complement (U A : Set ℕ) : Set ℕ := U \ A

theorem intersection_of_complements :
  U = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} →
  A = {0, 1, 3, 5, 8} →
  B = {2, 4, 5, 6, 8} →
  (complement U A ∩ complement U B) = {7, 9} :=
by
  intros hU hA hB
  sorry

end NUMINAMATH_GPT_intersection_of_complements_l2005_200532


namespace NUMINAMATH_GPT_exists_function_passing_through_point_l2005_200583

-- Define the function that satisfies f(2) = 0
theorem exists_function_passing_through_point : ∃ f : ℝ → ℝ, f 2 = 0 := 
sorry

end NUMINAMATH_GPT_exists_function_passing_through_point_l2005_200583


namespace NUMINAMATH_GPT_smallest_positive_integer_l2005_200560

theorem smallest_positive_integer (n : ℕ) (h : 629 * n ≡ 1181 * n [MOD 35]) : n = 35 :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_l2005_200560


namespace NUMINAMATH_GPT_least_gumballs_to_ensure_five_gumballs_of_same_color_l2005_200503

-- Define the number of gumballs for each color
def red_gumballs := 12
def white_gumballs := 10
def blue_gumballs := 11

-- Define the minimum number of gumballs required to ensure five of the same color
def min_gumballs_to_ensure_five_of_same_color := 13

-- Prove the question == answer given conditions
theorem least_gumballs_to_ensure_five_gumballs_of_same_color :
  (red_gumballs + white_gumballs + blue_gumballs) = 33 → min_gumballs_to_ensure_five_of_same_color = 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_least_gumballs_to_ensure_five_gumballs_of_same_color_l2005_200503


namespace NUMINAMATH_GPT_greatest_b_l2005_200599

theorem greatest_b (b : ℝ) : (-b^2 + 9 * b - 14 ≥ 0) → b ≤ 7 := sorry

end NUMINAMATH_GPT_greatest_b_l2005_200599


namespace NUMINAMATH_GPT_geometric_sequence_general_formula_l2005_200572

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a1 q : ℝ, ∀ n : ℕ, a n = a1 * q ^ (n - 1)

variables (a : ℕ → ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := a 1 + a 3 = 10
def condition2 : Prop := a 4 + a 6 = 5 / 4

-- The final statement to prove
theorem geometric_sequence_general_formula (h : geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) :
  ∀ n : ℕ, a n = 2 ^ (4 - n) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_general_formula_l2005_200572


namespace NUMINAMATH_GPT_shyne_total_plants_l2005_200586

/-- Shyne's seed packets -/
def eggplants_per_packet : ℕ := 14
def sunflowers_per_packet : ℕ := 10

/-- Seed packets purchased by Shyne -/
def eggplant_packets : ℕ := 4
def sunflower_packets : ℕ := 6

/-- Total number of plants grown by Shyne -/
def total_plants : ℕ := 116

theorem shyne_total_plants :
  eggplants_per_packet * eggplant_packets + sunflowers_per_packet * sunflower_packets = total_plants :=
by
  sorry

end NUMINAMATH_GPT_shyne_total_plants_l2005_200586


namespace NUMINAMATH_GPT_probability_collinear_dots_l2005_200549

theorem probability_collinear_dots 
  (rows : ℕ) (cols : ℕ) (total_dots : ℕ) (collinear_sets : ℕ) (total_ways : ℕ) : 
  rows = 5 → cols = 4 → total_dots = 20 → collinear_sets = 20 → total_ways = 4845 → 
  (collinear_sets : ℚ) / total_ways = 4 / 969 :=
by
  intros hrows hcols htotal_dots hcollinear_sets htotal_ways
  sorry

end NUMINAMATH_GPT_probability_collinear_dots_l2005_200549


namespace NUMINAMATH_GPT_rectangular_x_value_l2005_200582

theorem rectangular_x_value (x : ℝ)
  (h1 : ∀ (length : ℝ), length = 4 * x)
  (h2 : ∀ (width : ℝ), width = x + 10)
  (h3 : ∀ (length width : ℝ), length * width = 2 * (2 * length + 2 * width))
  : x = (Real.sqrt 41 - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_x_value_l2005_200582


namespace NUMINAMATH_GPT_tens_digit_of_6_pow_4_is_9_l2005_200543

theorem tens_digit_of_6_pow_4_is_9 : (6 ^ 4 / 10) % 10 = 9 :=
by
  sorry

end NUMINAMATH_GPT_tens_digit_of_6_pow_4_is_9_l2005_200543


namespace NUMINAMATH_GPT_a_18_value_l2005_200534

variable (a : ℕ → ℚ)

axiom a1 : a 1 = 1
axiom a2 : a 2 = 2
axiom a_rec (n : ℕ) (hn : 2 ≤ n) : 2 * n * a n = (n - 1) * a (n - 1) + (n + 1) * a (n + 1)

theorem a_18_value : a 18 = 26 / 9 :=
sorry

end NUMINAMATH_GPT_a_18_value_l2005_200534


namespace NUMINAMATH_GPT_tapA_turned_off_time_l2005_200596

noncomputable def tapA_rate := 1 / 45
noncomputable def tapB_rate := 1 / 40
noncomputable def tapB_fill_time := 23

theorem tapA_turned_off_time :
  ∃ t : ℕ, t * (tapA_rate + tapB_rate) + tapB_fill_time * tapB_rate = 1 ∧ t = 9 :=
by
  sorry

end NUMINAMATH_GPT_tapA_turned_off_time_l2005_200596


namespace NUMINAMATH_GPT_lily_spent_amount_l2005_200545

def num_years (start_year end_year : ℕ) : ℕ :=
  end_year - start_year

def total_spent (cost_per_plant num_years : ℕ) : ℕ :=
  cost_per_plant * num_years

theorem lily_spent_amount :
  let start_year := 1989
  let end_year := 2021
  let cost_per_plant := 20
  num_years start_year end_year = 32 →
  total_spent cost_per_plant 32 = 640 :=
by
  intros
  sorry

end NUMINAMATH_GPT_lily_spent_amount_l2005_200545


namespace NUMINAMATH_GPT_find_c_l2005_200569

theorem find_c (c : ℝ) (h : ∀ x y : ℝ, 5 * x + 8 * y + c = 0 ∧ x + y = 26) : c = -80 :=
sorry

end NUMINAMATH_GPT_find_c_l2005_200569


namespace NUMINAMATH_GPT_inequality_proof_l2005_200595

theorem inequality_proof {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 + 4 * a / (b + c)) * (1 + 4 * b / (c + a)) * (1 + 4 * c / (a + b)) > 25 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_inequality_proof_l2005_200595


namespace NUMINAMATH_GPT_find_a_plus_b_l2005_200576

noncomputable def lines_intersect (a b : ℝ) : Prop := 
  (∃ x y : ℝ, (x = 1/3 * y + a) ∧ (y = 1/3 * x + b) ∧ (x = 3) ∧ (y = 6))

theorem find_a_plus_b (a b : ℝ) (h : lines_intersect a b) : a + b = 6 :=
sorry

end NUMINAMATH_GPT_find_a_plus_b_l2005_200576


namespace NUMINAMATH_GPT_quadratic_form_decomposition_l2005_200565

theorem quadratic_form_decomposition (a b c : ℝ) (h : ∀ x : ℝ, 8 * x^2 + 64 * x + 512 = a * (x + b) ^ 2 + c) :
  a + b + c = 396 := 
sorry

end NUMINAMATH_GPT_quadratic_form_decomposition_l2005_200565


namespace NUMINAMATH_GPT_tenth_pair_in_twentieth_row_l2005_200522

noncomputable def pair_in_row (n k : ℕ) : ℕ × ℕ :=
  if k = 0 ∨ k > n then (0, 0) else (k, n + 1 - k)

theorem tenth_pair_in_twentieth_row : pair_in_row 20 10 = (10, 11) := by
  sorry

end NUMINAMATH_GPT_tenth_pair_in_twentieth_row_l2005_200522


namespace NUMINAMATH_GPT_right_triangle_consecutive_sides_l2005_200517

theorem right_triangle_consecutive_sides (n : ℕ) (n_pos : 0 < n) :
    (n+1)^2 + n^2 = (n+2)^2 ↔ (n = 3) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_consecutive_sides_l2005_200517


namespace NUMINAMATH_GPT_pyramid_volume_l2005_200514

-- Define the conditions
def height_vertex_to_center_base := 12 -- cm
def side_of_square_base := 10 -- cm
def base_area := side_of_square_base * side_of_square_base -- cm²
def volume := (1 / 3) * base_area * height_vertex_to_center_base -- cm³

-- State the theorem
theorem pyramid_volume : volume = 400 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_pyramid_volume_l2005_200514


namespace NUMINAMATH_GPT_simplify_complex_div_l2005_200513

theorem simplify_complex_div (a b c d : ℝ) (i : ℂ)
  (h1 : (a = 3) ∧ (b = 5) ∧ (c = -2) ∧ (d = 7) ∧ (i = Complex.I)) :
  ((Complex.mk a b) / (Complex.mk c d) = (Complex.mk (29/53) (-31/53))) :=
by
  sorry

end NUMINAMATH_GPT_simplify_complex_div_l2005_200513


namespace NUMINAMATH_GPT_group_age_analysis_l2005_200539

theorem group_age_analysis (total_members : ℕ) (average_age : ℝ) (zero_age_members : ℕ) 
  (h1 : total_members = 50) (h2 : average_age = 5) (h3 : zero_age_members = 10) :
  let total_age := total_members * average_age
  let non_zero_members := total_members - zero_age_members
  let non_zero_average_age := total_age / non_zero_members
  non_zero_members = 40 ∧ non_zero_average_age = 6.25 :=
by
  let total_age := total_members * average_age
  let non_zero_members := total_members - zero_age_members
  let non_zero_average_age := total_age / non_zero_members
  have h_non_zero_members : non_zero_members = 40 := by sorry
  have h_non_zero_average_age : non_zero_average_age = 6.25 := by sorry
  exact ⟨h_non_zero_members, h_non_zero_average_age⟩

end NUMINAMATH_GPT_group_age_analysis_l2005_200539


namespace NUMINAMATH_GPT_johns_balance_at_end_of_first_year_l2005_200547

theorem johns_balance_at_end_of_first_year (initial_deposit interest_first_year : ℝ) 
  (h1 : initial_deposit = 5000) 
  (h2 : interest_first_year = 500) :
  initial_deposit + interest_first_year = 5500 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_johns_balance_at_end_of_first_year_l2005_200547


namespace NUMINAMATH_GPT_rectangular_garden_width_l2005_200563

theorem rectangular_garden_width
  (w : ℝ)
  (h₁ : ∃ l, l = 3 * w)
  (h₂ : ∃ A, A = l * w ∧ A = 507) : 
  w = 13 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_garden_width_l2005_200563


namespace NUMINAMATH_GPT_total_slices_is_78_l2005_200574

-- Definitions based on conditions
def ratio_buzz_waiter (x : ℕ) : Prop := (5 * x) + (8 * x) = 78
def waiter_condition (x : ℕ) : Prop := (8 * x) - 20 = 28

-- Prove that the total number of slices is 78 given conditions
theorem total_slices_is_78 (x : ℕ) (h1 : ratio_buzz_waiter x) (h2 : waiter_condition x) : (5 * x) + (8 * x) = 78 :=
by
  sorry

end NUMINAMATH_GPT_total_slices_is_78_l2005_200574


namespace NUMINAMATH_GPT_arithmetic_sequence_formula_min_value_t_minus_s_max_value_k_l2005_200567

-- Definitions and theorems for the given conditions

-- (1) General formula for the arithmetic sequence
theorem arithmetic_sequence_formula (a S : Nat → Int) (n : Nat) (h1 : a 2 = -1)
  (h2 : S 9 = 5 * S 5) : 
  ∀ n, a n = -8 * n + 15 := 
sorry

-- (2) Minimum value of t - s
theorem min_value_t_minus_s (b : Nat → Rat) (T : Nat → Rat) 
  (h3 : ∀ n, b n = 1 / ((-8 * (n + 1) + 15) * (-8 * (n + 2) + 15))) 
  (h4 : ∀ n, s ≤ T n ∧ T n ≤ t) : 
  t - s = 1 / 72 := 
sorry

-- (3) Maximum value of k
theorem max_value_k (S a : Nat → Int) (k : Rat)
  (h5 : ∀ n, n ≥ 3 → S n / a n ≤ n^2 / (n + k)) :
  k = 80 / 9 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_formula_min_value_t_minus_s_max_value_k_l2005_200567


namespace NUMINAMATH_GPT_weight_of_b_l2005_200538

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45) 
  (h2 : (a + b) / 2 = 41) 
  (h3 : (b + c) / 2 = 43) 
  : b = 33 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_b_l2005_200538


namespace NUMINAMATH_GPT_chromosome_stability_due_to_meiosis_and_fertilization_l2005_200580

/-- Definition of reducing chromosome number during meiosis -/
def meiosis_reduces_chromosome_number (n : ℕ) : ℕ := n / 2

/-- Definition of restoring chromosome number during fertilization -/
def fertilization_restores_chromosome_number (n : ℕ) : ℕ := n * 2

/-- Axiom: Sexual reproduction involves meiosis and fertilization to maintain chromosome stability -/
axiom chromosome_stability (n m : ℕ) (h1 : meiosis_reduces_chromosome_number n = m) 
  (h2 : fertilization_restores_chromosome_number m = n) : n = n

/-- Theorem statement in Lean 4: The chromosome number stability in sexually reproducing organisms is maintained due to meiosis and fertilization -/
theorem chromosome_stability_due_to_meiosis_and_fertilization 
  (n : ℕ) (h_meiosis: meiosis_reduces_chromosome_number n = n / 2) 
  (h_fertilization: fertilization_restores_chromosome_number (n / 2) = n) : 
  n = n := 
by
  apply chromosome_stability
  exact h_meiosis
  exact h_fertilization

end NUMINAMATH_GPT_chromosome_stability_due_to_meiosis_and_fertilization_l2005_200580


namespace NUMINAMATH_GPT_find_A_l2005_200509

namespace PolynomialDecomposition

theorem find_A (x A B C : ℝ)
  (h : (x^3 + 2 * x^2 - 17 * x - 30)⁻¹ = A / (x - 5) + B / (x + 2) + C / ((x + 2)^2)) :
  A = 1 / 49 :=
by sorry

end PolynomialDecomposition

end NUMINAMATH_GPT_find_A_l2005_200509


namespace NUMINAMATH_GPT_div_relation_l2005_200531

theorem div_relation (a b d : ℝ) (h1 : a / b = 3) (h2 : b / d = 2 / 5) : d / a = 5 / 6 := by
  sorry

end NUMINAMATH_GPT_div_relation_l2005_200531


namespace NUMINAMATH_GPT_brother_reading_time_l2005_200579

variable (my_time_in_hours : ℕ)
variable (speed_ratio : ℕ)

theorem brother_reading_time
  (h1 : my_time_in_hours = 3)
  (h2 : speed_ratio = 4) :
  my_time_in_hours * 60 / speed_ratio = 45 := 
by
  sorry

end NUMINAMATH_GPT_brother_reading_time_l2005_200579


namespace NUMINAMATH_GPT_initial_birds_count_l2005_200510

variable (init_birds landed_birds total_birds : ℕ)

theorem initial_birds_count :
  (landed_birds = 8) →
  (total_birds = 20) →
  (init_birds + landed_birds = total_birds) →
  (init_birds = 12) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_initial_birds_count_l2005_200510


namespace NUMINAMATH_GPT_fraction_of_roots_l2005_200597

theorem fraction_of_roots (a b : ℝ) (h : a * b = -209) (h_sum : a + b = -8) : 
  (a * b) / (a + b) = 209 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_of_roots_l2005_200597


namespace NUMINAMATH_GPT_average_speed_for_trip_l2005_200592

theorem average_speed_for_trip :
  ∀ (walk_dist bike_dist drive_dist tot_dist walk_speed bike_speed drive_speed : ℝ)
  (h1 : walk_dist = 5) (h2 : bike_dist = 35) (h3 : drive_dist = 80)
  (h4 : tot_dist = 120) (h5 : walk_speed = 5) (h6 : bike_speed = 15)
  (h7 : drive_speed = 120),
  (tot_dist / (walk_dist / walk_speed + bike_dist / bike_speed + drive_dist / drive_speed)) = 30 :=
by
  intros
  sorry

end NUMINAMATH_GPT_average_speed_for_trip_l2005_200592


namespace NUMINAMATH_GPT_history_students_count_l2005_200594

theorem history_students_count
  (total_students : ℕ)
  (sample_students : ℕ)
  (physics_students_sampled : ℕ)
  (history_students_sampled : ℕ)
  (x : ℕ)
  (H1 : total_students = 1500)
  (H2 : sample_students = 120)
  (H3 : physics_students_sampled = 80)
  (H4 : history_students_sampled = sample_students - physics_students_sampled)
  (H5 : x = 1500 * history_students_sampled / sample_students) :
  x = 500 :=
by
  sorry

end NUMINAMATH_GPT_history_students_count_l2005_200594


namespace NUMINAMATH_GPT_area_ratio_of_squares_l2005_200537

theorem area_ratio_of_squares (a b : ℝ) (h : 4 * (4 * b) = 4 * a) : (a * a) / (b * b) = 16 :=
by
  sorry

end NUMINAMATH_GPT_area_ratio_of_squares_l2005_200537


namespace NUMINAMATH_GPT_min_value_of_m_l2005_200548

theorem min_value_of_m (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c = 3) :
  a^2 + b^2 + c^2 ≥ 3 :=
sorry

end NUMINAMATH_GPT_min_value_of_m_l2005_200548


namespace NUMINAMATH_GPT_average_seeds_per_apple_l2005_200598

-- Define the problem conditions and the proof statement

theorem average_seeds_per_apple
  (A : ℕ)
  (total_seeds_requirement : ℕ := 60)
  (pear_seeds_avg : ℕ := 2)
  (grape_seeds_avg : ℕ := 3)
  (num_apples : ℕ := 4)
  (num_pears : ℕ := 3)
  (num_grapes : ℕ := 9)
  (shortfall : ℕ := 3)
  (collected_seeds : ℕ := num_apples * A + num_pears * pear_seeds_avg + num_grapes * grape_seeds_avg)
  (required_seeds : ℕ := total_seeds_requirement - shortfall) :
  collected_seeds = required_seeds → A = 6 := 
by
  sorry

end NUMINAMATH_GPT_average_seeds_per_apple_l2005_200598


namespace NUMINAMATH_GPT_simplify_expr_l2005_200566

theorem simplify_expr (x : ℝ) (hx : x ≠ 0) :
  (3/4) * (8/(x^2) + 12*x - 5) = 6/(x^2) + 9*x - 15/4 := by
  sorry

end NUMINAMATH_GPT_simplify_expr_l2005_200566


namespace NUMINAMATH_GPT_find_b6_l2005_200504

def fib (b : ℕ → ℕ) : Prop :=
  ∀ n, b (n + 2) = b (n + 1) + b n

theorem find_b6 (b : ℕ → ℕ) (b1 b2 : ℕ)
  (h1 : b 1 = b1) (h2 : b 2 = b2) (h3 : b 5 = 55)
  (hfib : fib b) : b 6 = 84 :=
  sorry

end NUMINAMATH_GPT_find_b6_l2005_200504


namespace NUMINAMATH_GPT_three_times_x_not_much_different_from_two_l2005_200552

theorem three_times_x_not_much_different_from_two (x : ℝ) :
  3 * x - 2 ≤ -1 := 
sorry

end NUMINAMATH_GPT_three_times_x_not_much_different_from_two_l2005_200552


namespace NUMINAMATH_GPT_max_value_m_l2005_200578

theorem max_value_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x ≥ 3 ∧ 2 * x - 1 < m) → m ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_max_value_m_l2005_200578


namespace NUMINAMATH_GPT_largest_side_of_triangle_l2005_200533

theorem largest_side_of_triangle (x y Δ c : ℕ)
  (h1 : (x + 2 * Δ / x = y + 2 * Δ / y))
  (h2 : x = 60)
  (h3 : y = 63) :
  c = 87 :=
sorry

end NUMINAMATH_GPT_largest_side_of_triangle_l2005_200533


namespace NUMINAMATH_GPT_profit_percentage_l2005_200544

theorem profit_percentage (purchase_price sell_price : ℝ) (h1 : purchase_price = 600) (h2 : sell_price = 624) :
  ((sell_price - purchase_price) / purchase_price) * 100 = 4 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_l2005_200544


namespace NUMINAMATH_GPT_sequence_first_term_eq_three_l2005_200501

theorem sequence_first_term_eq_three
  (a : ℕ → ℕ)
  (h_rec : ∀ n : ℕ, a (n + 2) = a (n + 1) + a n)
  (h_nz : ∀ n : ℕ, 0 < a n)
  (h_a11 : a 11 = 157) :
  a 1 = 3 :=
sorry

end NUMINAMATH_GPT_sequence_first_term_eq_three_l2005_200501


namespace NUMINAMATH_GPT_kiran_money_l2005_200525

theorem kiran_money (R G K : ℕ) (h1: R / G = 6 / 7) (h2: G / K = 6 / 15) (h3: R = 36) : K = 105 := by
  sorry

end NUMINAMATH_GPT_kiran_money_l2005_200525


namespace NUMINAMATH_GPT_quadratic_real_roots_condition_sufficient_l2005_200558

theorem quadratic_real_roots_condition_sufficient (m : ℝ) : (m < 1 / 4) → ∃ x : ℝ, x^2 + x + m = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_condition_sufficient_l2005_200558


namespace NUMINAMATH_GPT_is_rectangle_l2005_200551

-- Define the points A, B, C, and D.
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (1, 6)
def C : ℝ × ℝ := (5, 4)
def D : ℝ × ℝ := (2, -2)

-- Define the vectors AB, DC, AD.
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)
def AB := vec A B
def DC := vec D C
def AD := vec A D

-- Function to compute dot product of two vectors.
def dot (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Prove that quadrilateral ABCD is a rectangle.
theorem is_rectangle : AB = DC ∧ dot AB AD = 0 := by
  sorry

end NUMINAMATH_GPT_is_rectangle_l2005_200551


namespace NUMINAMATH_GPT_second_smallest_perimeter_l2005_200500

theorem second_smallest_perimeter (a b c : ℕ) (h1 : a + 1 = b) (h2 : b + 1 = c) :
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) → 
  (a + b + c = 12) :=
by
  sorry

end NUMINAMATH_GPT_second_smallest_perimeter_l2005_200500


namespace NUMINAMATH_GPT_vector_addition_l2005_200591

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, 5)

-- State the theorem that we want to prove
theorem vector_addition : a + 3 • b = (-1, 18) :=
  sorry

end NUMINAMATH_GPT_vector_addition_l2005_200591


namespace NUMINAMATH_GPT_min_distance_from_circle_to_line_l2005_200526

-- Define the circle and line conditions
def is_on_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1
def line (x y : ℝ) : Prop := 4*x + 3*y + 5 = 0

-- The theorem to prove
theorem min_distance_from_circle_to_line (x y : ℝ) (h : is_on_circle x y) : 
  ∃ m_dist : ℝ, m_dist = 2 :=
by
  -- Place holder proof
  sorry

end NUMINAMATH_GPT_min_distance_from_circle_to_line_l2005_200526


namespace NUMINAMATH_GPT_triangle_base_and_area_l2005_200584

theorem triangle_base_and_area
  (height : ℝ)
  (h_height : height = 12)
  (height_base_ratio : ℝ)
  (h_ratio : height_base_ratio = 2 / 3) :
  ∃ (base : ℝ) (area : ℝ),
  base = height / height_base_ratio ∧
  area = base * height / 2 ∧
  base = 18 ∧
  area = 108 :=
by
  sorry

end NUMINAMATH_GPT_triangle_base_and_area_l2005_200584


namespace NUMINAMATH_GPT_correct_equation_l2005_200589

theorem correct_equation (x : ℕ) : 8 * x - 3 = 7 * x + 4 :=
by sorry

end NUMINAMATH_GPT_correct_equation_l2005_200589


namespace NUMINAMATH_GPT_min_distance_PA_l2005_200550

theorem min_distance_PA :
  let A : ℝ × ℝ := (0, 1)
  ∀ (P : ℝ × ℝ), (∃ x : ℝ, x > 0 ∧ P = (x, (x + 2) / x)) →
  ∃ d : ℝ, d = 2 ∧ ∀ Q : ℝ × ℝ, (∃ x : ℝ, x > 0 ∧ Q = (x, (x + 2) / x)) → dist A Q ≥ d :=
by
  sorry

end NUMINAMATH_GPT_min_distance_PA_l2005_200550


namespace NUMINAMATH_GPT_incorrect_statement_l2005_200555

theorem incorrect_statement (p q : Prop) (hp : ¬ p) (hq : q) : ¬ (¬ q) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_l2005_200555


namespace NUMINAMATH_GPT_function_expression_and_min_value_l2005_200556

def f (x b : ℝ) := x^2 - 2*x + b

theorem function_expression_and_min_value 
    (a b : ℝ)
    (condition1 : f (2 ^ a) b = b)
    (condition2 : f a b = 4) :
    f a b = 5 
    ∧ 
    ∃ c : ℝ, f (2^c) 5 = 4 ∧ c = 0 :=
by
  sorry

end NUMINAMATH_GPT_function_expression_and_min_value_l2005_200556


namespace NUMINAMATH_GPT_percentage_x_equals_y_l2005_200590

theorem percentage_x_equals_y (x y z : ℝ) (p : ℝ)
    (h1 : 0.45 * z = 0.39 * y)
    (h2 : z = 0.65 * x)
    (h3 : y = (p / 100) * x) : 
    p = 75 := 
sorry

end NUMINAMATH_GPT_percentage_x_equals_y_l2005_200590


namespace NUMINAMATH_GPT_find_xy_l2005_200535

theorem find_xy (x y : ℝ) :
  x^2 + y^2 = 2 ∧ (x^2 / (2 - y) + y^2 / (2 - x) = 2) → (x = 1 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_xy_l2005_200535


namespace NUMINAMATH_GPT_interpretation_of_k5_3_l2005_200523

theorem interpretation_of_k5_3 (k : ℕ) (hk : 0 < k) : (k^5)^3 = k^5 * k^5 * k^5 :=
by sorry

end NUMINAMATH_GPT_interpretation_of_k5_3_l2005_200523


namespace NUMINAMATH_GPT_tan_sum_pi_over_4_l2005_200546

open Real

theorem tan_sum_pi_over_4 {α : ℝ} (h₁ : cos (2 * α) + sin α * (2 * sin α - 1) = 2 / 5) (h₂ : π / 4 < α) (h₃ : α < π) : 
    tan (α + π / 4) = 1 / 7 := sorry

end NUMINAMATH_GPT_tan_sum_pi_over_4_l2005_200546
