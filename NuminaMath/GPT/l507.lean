import Mathlib

namespace NUMINAMATH_GPT_jessies_weight_after_first_week_l507_50730

-- Definitions from the conditions
def initial_weight : ℕ := 92
def first_week_weight_loss : ℕ := 56

-- The theorem statement
theorem jessies_weight_after_first_week : initial_weight - first_week_weight_loss = 36 := by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_jessies_weight_after_first_week_l507_50730


namespace NUMINAMATH_GPT_number_of_units_sold_l507_50780

theorem number_of_units_sold (p : ℕ) (c : ℕ) (k : ℕ) (h : p * c = k) (h₁ : c = 800) (h₂ : k = 8000) : p = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_units_sold_l507_50780


namespace NUMINAMATH_GPT_wechat_payment_meaning_l507_50791

theorem wechat_payment_meaning (initial_balance after_receive_balance : ℝ)
  (recv_amount sent_amount : ℝ)
  (h1 : recv_amount = 200)
  (h2 : initial_balance + recv_amount = after_receive_balance)
  (h3 : after_receive_balance - sent_amount = initial_balance)
  : sent_amount = 200 :=
by
  -- starting the proof becomes irrelevant
  sorry

end NUMINAMATH_GPT_wechat_payment_meaning_l507_50791


namespace NUMINAMATH_GPT_y_values_relation_l507_50721

theorem y_values_relation :
  ∀ y1 y2 y3 : ℝ,
    (y1 = (-3 + 1) ^ 2 + 1) →
    (y2 = (0 + 1) ^ 2 + 1) →
    (y3 = (2 + 1) ^ 2 + 1) →
    y2 < y1 ∧ y1 < y3 :=
by
  sorry

end NUMINAMATH_GPT_y_values_relation_l507_50721


namespace NUMINAMATH_GPT_find_product_of_variables_l507_50794

variables (a b c d : ℚ)

def system_of_equations (a b c d : ℚ) :=
  3 * a + 4 * b + 6 * c + 9 * d = 45 ∧
  4 * (d + c) = b + 1 ∧
  4 * b + 2 * c = a ∧
  2 * c - 2 = d

theorem find_product_of_variables :
  system_of_equations a b c d → a * b * c * d = 162 / 185 :=
by sorry

end NUMINAMATH_GPT_find_product_of_variables_l507_50794


namespace NUMINAMATH_GPT_amanda_jogging_distance_l507_50771

/-- Amanda's jogging path and the distance calculation. -/
theorem amanda_jogging_distance:
  let east_leg := 1.5
  let northwest_leg := 2
  let southwest_leg := 1
  -- Convert runs to displacement components
  let nw_x := northwest_leg / Real.sqrt 2
  let nw_y := northwest_leg / Real.sqrt 2
  let sw_x := southwest_leg / Real.sqrt 2
  let sw_y := southwest_leg / Real.sqrt 2
  -- Calculate net displacements
  let net_east := east_leg - (nw_x + sw_x)
  let net_north := nw_y - sw_y
  -- Final distance back to starting point
  let distance := Real.sqrt (net_east^2 + net_north^2)
  distance = Real.sqrt ((1.5 - 3 * Real.sqrt 2 / 2)^2 + (Real.sqrt 2 / 2)^2) := sorry

end NUMINAMATH_GPT_amanda_jogging_distance_l507_50771


namespace NUMINAMATH_GPT_fraction_expression_proof_l507_50739

theorem fraction_expression_proof :
  (1 / 8 * 1 / 9 * 1 / 28 = 1 / 2016) ∨ ((1 / 8 - 1 / 9) * 1 / 28 = 1 / 2016) :=
by
  sorry

end NUMINAMATH_GPT_fraction_expression_proof_l507_50739


namespace NUMINAMATH_GPT_exist_identical_2x2_squares_l507_50769

theorem exist_identical_2x2_squares : 
  ∃ sq1 sq2 : Finset (Fin 5 × Fin 5), 
    sq1.card = 4 ∧ sq2.card = 4 ∧ 
    (∀ (i : Fin 5) (j : Fin 5), 
      (i = 0 ∧ j = 0) ∨ (i = 4 ∧ j = 4) → 
      (i, j) ∈ sq1 ∧ (i, j) ∈ sq2 ∧ 
      (sq1 ≠ sq2 → ∃ p ∈ sq1, p ∉ sq2)) :=
sorry

end NUMINAMATH_GPT_exist_identical_2x2_squares_l507_50769


namespace NUMINAMATH_GPT_greatest_integer_solution_l507_50753

theorem greatest_integer_solution :
  ∃ x : ℤ, (∀ y : ℤ, (6 * (y : ℝ)^2 + 5 * (y : ℝ) - 8) < (3 * (y : ℝ)^2 - 4 * (y : ℝ) + 1) → y ≤ x) 
  ∧ (6 * (x : ℝ)^2 + 5 * (x : ℝ) - 8) < (3 * (x : ℝ)^2 - 4 * (x : ℝ) + 1) ∧ x = 0 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_solution_l507_50753


namespace NUMINAMATH_GPT_sally_total_score_l507_50762

theorem sally_total_score :
  ∀ (correct incorrect unanswered : ℕ) (score_correct score_incorrect : ℝ),
    correct = 17 →
    incorrect = 8 →
    unanswered = 5 →
    score_correct = 1 →
    score_incorrect = -0.25 →
    (correct * score_correct +
     incorrect * score_incorrect +
     unanswered * 0) = 15 :=
by
  intros correct incorrect unanswered score_correct score_incorrect
  intros h_corr h_incorr h_unan h_sc h_si
  sorry

end NUMINAMATH_GPT_sally_total_score_l507_50762


namespace NUMINAMATH_GPT_triple_hash_90_l507_50783

def hash (N : ℝ) : ℝ := 0.3 * N + 2

theorem triple_hash_90 : hash (hash (hash 90)) = 5.21 :=
by
  sorry

end NUMINAMATH_GPT_triple_hash_90_l507_50783


namespace NUMINAMATH_GPT_benjamin_distance_l507_50760

def speed := 10  -- Speed in kilometers per hour
def time := 8    -- Time in hours

def distance (s t : ℕ) := s * t  -- Distance formula

theorem benjamin_distance : distance speed time = 80 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_benjamin_distance_l507_50760


namespace NUMINAMATH_GPT_plane_angle_divides_cube_l507_50768

noncomputable def angle_between_planes (m n : ℕ) (h : m ≤ n) : ℝ :=
  Real.arctan (2 * m / (m + n))

theorem plane_angle_divides_cube (m n : ℕ) (h : m ≤ n) :
  ∃ α, α = angle_between_planes m n h :=
sorry

end NUMINAMATH_GPT_plane_angle_divides_cube_l507_50768


namespace NUMINAMATH_GPT_oscar_leap_longer_than_elmer_stride_l507_50774

theorem oscar_leap_longer_than_elmer_stride :
  ∀ (elmer_strides_per_gap oscar_leaps_per_gap gaps_between_poles : ℕ)
    (total_distance : ℝ),
  elmer_strides_per_gap = 60 →
  oscar_leaps_per_gap = 16 →
  gaps_between_poles = 60 →
  total_distance = 7920 →
  let elmer_stride_length := total_distance / (elmer_strides_per_gap * gaps_between_poles)
  let oscar_leap_length := total_distance / (oscar_leaps_per_gap * gaps_between_poles)
  oscar_leap_length - elmer_stride_length = 6.05 :=
by
  intros
  sorry

end NUMINAMATH_GPT_oscar_leap_longer_than_elmer_stride_l507_50774


namespace NUMINAMATH_GPT_greatest_expression_value_l507_50795

noncomputable def greatest_expression : ℝ := 0.9986095661846496

theorem greatest_expression_value : greatest_expression = 0.9986095661846496 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_greatest_expression_value_l507_50795


namespace NUMINAMATH_GPT_exact_time_now_l507_50723

/-- Given that it is between 9:00 and 10:00 o'clock,
and nine minutes from now, the minute hand of a watch
will be exactly opposite the place where the hour hand
was six minutes ago, show that the exact time now is 9:06
-/
theorem exact_time_now 
  (t : ℕ)
  (h1 : t < 60)
  (h2 : ∃ t, 6 * (t + 9) - (270 + 0.5 * (t - 6)) = 180 ∨ 6 * (t + 9) - (270 + 0.5 * (t - 6)) = -180) :
  t = 6 := 
sorry

end NUMINAMATH_GPT_exact_time_now_l507_50723


namespace NUMINAMATH_GPT_problem_statement_l507_50738

def f(x : ℝ) : ℝ := 3 * x - 3
def g(x : ℝ) : ℝ := x^2 + 1

theorem problem_statement : f (1 + g 2) = 15 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l507_50738


namespace NUMINAMATH_GPT_arithmetic_sequence_l507_50754

theorem arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) (h₀ : a 1 = 2) (h₁ : a 2 + a 3 = 13)
    (h₂ : ∀ n, a n = a 1 + (n - 1) * d) : a 5 = 14 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_l507_50754


namespace NUMINAMATH_GPT_washington_high_teacher_student_ratio_l507_50742

theorem washington_high_teacher_student_ratio (students teachers : ℕ) (h_students : students = 1155) (h_teachers : teachers = 42) : (students / teachers : ℚ) = 27.5 :=
by
  sorry

end NUMINAMATH_GPT_washington_high_teacher_student_ratio_l507_50742


namespace NUMINAMATH_GPT_carrots_picked_next_day_l507_50786

theorem carrots_picked_next_day :
  ∀ (initial_picked thrown_out additional_picked total : ℕ),
    initial_picked = 48 →
    thrown_out = 11 →
    total = 52 →
    additional_picked = total - (initial_picked - thrown_out) →
    additional_picked = 15 :=
by
  intros initial_picked thrown_out additional_picked total h_ip h_to h_total h_ap
  sorry

end NUMINAMATH_GPT_carrots_picked_next_day_l507_50786


namespace NUMINAMATH_GPT_peter_fraction_equiv_l507_50719

def fraction_pizza_peter_ate (total_slices : ℕ) (slices_ate_alone : ℕ) (shared_slices_brother : ℚ) (shared_slices_sister : ℚ) : ℚ :=
  (slices_ate_alone / total_slices) + (shared_slices_brother / total_slices) + (shared_slices_sister / total_slices)

theorem peter_fraction_equiv :
  fraction_pizza_peter_ate 16 3 (1/2) (1/2) = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_peter_fraction_equiv_l507_50719


namespace NUMINAMATH_GPT_max_value_fraction_l507_50712

theorem max_value_fraction (a b : ℝ) 
  (h1 : a + b - 2 ≥ 0)
  (h2 : b - a - 1 ≤ 0)
  (h3 : a ≤ 1) :
  ∃ max_val, max_val = 7 / 5 ∧ ∀ (x y : ℝ), 
    (x + y - 2 ≥ 0) → (y - x - 1 ≤ 0) → (x ≤ 1) → (x + 2*y) / (2*x + y) ≤ max_val :=
sorry

end NUMINAMATH_GPT_max_value_fraction_l507_50712


namespace NUMINAMATH_GPT_garden_perimeter_l507_50702

noncomputable def perimeter_of_garden (w l : ℝ) (h1 : l = 3 * w + 15) (h2 : w * l = 4050) : ℝ :=
  2 * l + 2 * w

theorem garden_perimeter (w l : ℝ) (h1 : l = 3 * w + 15) (h2 : w * l = 4050) :
  perimeter_of_garden w l h1 h2 = 304.64 :=
sorry

end NUMINAMATH_GPT_garden_perimeter_l507_50702


namespace NUMINAMATH_GPT_tangent_line_tangent_value_at_one_l507_50734
noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function f

theorem tangent_line_tangent_value_at_one
  (f : ℝ → ℝ)
  (hf1 : f 1 = 3 - 1 / 2)
  (hf'1 : deriv f 1 = 1 / 2)
  (tangent_eq : ∀ x, f 1 + deriv f 1 * (x - 1) = 1 / 2 * x + 2) :
  f 1 + deriv f 1 = 3 :=
by sorry

end NUMINAMATH_GPT_tangent_line_tangent_value_at_one_l507_50734


namespace NUMINAMATH_GPT_product_of_primes_95_l507_50790

theorem product_of_primes_95 (p q : Nat) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : p + q = 95) : p * q = 178 := sorry

end NUMINAMATH_GPT_product_of_primes_95_l507_50790


namespace NUMINAMATH_GPT_greatest_leftover_cookies_l507_50789

theorem greatest_leftover_cookies (n : ℕ) : ∃ k, k ≤ n ∧ k % 8 = 7 := sorry

end NUMINAMATH_GPT_greatest_leftover_cookies_l507_50789


namespace NUMINAMATH_GPT_sin_double_angle_value_l507_50750

theorem sin_double_angle_value (α : ℝ) (h1 : 0 < α ∧ α < π)
  (h2 : (1/2) * Real.cos (2 * α) = Real.sin (π/4 + α)) :
  Real.sin (2 * α) = -1 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_value_l507_50750


namespace NUMINAMATH_GPT_initial_oranges_l507_50781

open Nat

theorem initial_oranges (initial_oranges: ℕ) (eaten_oranges: ℕ) (stolen_oranges: ℕ) (returned_oranges: ℕ) (current_oranges: ℕ):
  eaten_oranges = 10 → 
  stolen_oranges = (initial_oranges - eaten_oranges) / 2 →
  returned_oranges = 5 →
  current_oranges = 30 →
  initial_oranges - eaten_oranges - stolen_oranges + returned_oranges = current_oranges →
  initial_oranges = 60 :=
by
  sorry

end NUMINAMATH_GPT_initial_oranges_l507_50781


namespace NUMINAMATH_GPT_total_soccer_games_l507_50757

theorem total_soccer_games (months : ℕ) (games_per_month : ℕ) (h_months : months = 3) (h_games_per_month : games_per_month = 9) : months * games_per_month = 27 :=
by
  sorry

end NUMINAMATH_GPT_total_soccer_games_l507_50757


namespace NUMINAMATH_GPT_negation_of_proposition_l507_50746

-- Definitions based on given conditions
def is_not_divisible_by_2 (n : ℤ) := n % 2 ≠ 0
def is_odd (n : ℤ) := n % 2 = 1

-- The negation proposition to be proved
theorem negation_of_proposition : ∃ n : ℤ, is_not_divisible_by_2 n ∧ ¬ is_odd n := 
sorry

end NUMINAMATH_GPT_negation_of_proposition_l507_50746


namespace NUMINAMATH_GPT_garden_perimeter_is_56_l507_50701

-- Define the conditions
def garden_width : ℕ := 12
def playground_length : ℕ := 16
def playground_width : ℕ := 12
def playground_area : ℕ := playground_length * playground_width
def garden_length : ℕ := playground_area / garden_width
def garden_perimeter : ℕ := 2 * (garden_length + garden_width)

-- Statement to prove
theorem garden_perimeter_is_56 :
  garden_perimeter = 56 := by
sorry

end NUMINAMATH_GPT_garden_perimeter_is_56_l507_50701


namespace NUMINAMATH_GPT_ratio_age_difference_to_pencils_l507_50758

-- Definitions of the given problem conditions
def AsafAge : ℕ := 50
def SumOfAges : ℕ := 140
def AlexanderAge : ℕ := SumOfAges - AsafAge

def PencilDifference : ℕ := 60
def TotalPencils : ℕ := 220
def AsafPencils : ℕ := (TotalPencils - PencilDifference) / 2
def AlexanderPencils : ℕ := AsafPencils + PencilDifference

-- Define the age difference and the ratio
def AgeDifference : ℕ := AlexanderAge - AsafAge
def Ratio : ℚ := AgeDifference / AsafPencils

theorem ratio_age_difference_to_pencils : Ratio = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_age_difference_to_pencils_l507_50758


namespace NUMINAMATH_GPT_johns_age_less_than_six_times_brothers_age_l507_50741

theorem johns_age_less_than_six_times_brothers_age 
  (B J : ℕ) 
  (h1 : B = 8) 
  (h2 : J + B = 10) 
  (h3 : J = 6 * B - 46) : 
  6 * B - J = 46 :=
by
  rw [h1, h3]
  exact sorry

end NUMINAMATH_GPT_johns_age_less_than_six_times_brothers_age_l507_50741


namespace NUMINAMATH_GPT_largest_n_for_factorable_polynomial_l507_50788

theorem largest_n_for_factorable_polynomial :
  (∃ (A B : ℤ), A * B = 72 ∧ ∀ (n : ℤ), n = 3 * B + A → n ≤ 217) ∧
  (∃ (A B : ℤ), A * B = 72 ∧ 3 * B + A = 217) :=
by
    sorry

end NUMINAMATH_GPT_largest_n_for_factorable_polynomial_l507_50788


namespace NUMINAMATH_GPT_polynomial_relation_l507_50785

variables {a b c : ℝ}

theorem polynomial_relation
  (h1: a ≠ 0) (h2: b ≠ 0) (h3: c ≠ 0) (h4: a + b + c = 0) :
  ((a^7 + b^7 + c^7)^2) / ((a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) * (a^4 + b^4 + c^4) * (a^5 + b^5 + c^5)) = 49 / 60 :=
sorry

end NUMINAMATH_GPT_polynomial_relation_l507_50785


namespace NUMINAMATH_GPT_correct_vector_equation_l507_50779

variables {V : Type*} [AddCommGroup V]

variables (A B C: V)

theorem correct_vector_equation : 
  (A - B) - (B - C) = A - C :=
sorry

end NUMINAMATH_GPT_correct_vector_equation_l507_50779


namespace NUMINAMATH_GPT_total_votes_cast_l507_50745

theorem total_votes_cast (V: ℕ) (invalid_votes: ℕ) (diff_votes: ℕ) 
  (H1: invalid_votes = 200) 
  (H2: diff_votes = 700) 
  (H3: (0.01 : ℝ) * V = diff_votes) 
  : (V + invalid_votes = 70200) :=
by
  sorry

end NUMINAMATH_GPT_total_votes_cast_l507_50745


namespace NUMINAMATH_GPT_jacket_final_price_l507_50792

/-- 
The initial price of the jacket is $20, 
the first discount is 40%, and the second discount is 25%. 
We need to prove that the final price of the jacket is $9.
-/
theorem jacket_final_price :
  let initial_price := 20
  let first_discount := 0.40
  let second_discount := 0.25
  let price_after_first_discount := initial_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  final_price = 9 :=
by
  sorry

end NUMINAMATH_GPT_jacket_final_price_l507_50792


namespace NUMINAMATH_GPT_no_solution_system_of_equations_l507_50767

theorem no_solution_system_of_equations :
  ¬ (∃ (x y : ℝ),
    (80 * x + 15 * y - 7) / (78 * x + 12 * y) = 1 ∧
    (2 * x^2 + 3 * y^2 - 11) / (y^2 - x^2 + 3) = 1 ∧
    78 * x + 12 * y ≠ 0 ∧
    y^2 - x^2 + 3 ≠ 0) :=
    by
      sorry

end NUMINAMATH_GPT_no_solution_system_of_equations_l507_50767


namespace NUMINAMATH_GPT_problem_statement_l507_50773

theorem problem_statement : (4^4 / 4^3) * 2^8 = 1024 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l507_50773


namespace NUMINAMATH_GPT_mary_characters_initials_l507_50751

theorem mary_characters_initials :
  ∀ (total_A total_C total_D total_E : ℕ),
  total_A = 60 / 2 →
  total_C = total_A / 2 →
  total_D = 2 * total_E →
  total_A + total_C + total_D + total_E = 60 →
  total_D = 10 :=
by
  intros total_A total_C total_D total_E hA hC hDE hSum
  sorry

end NUMINAMATH_GPT_mary_characters_initials_l507_50751


namespace NUMINAMATH_GPT_borrowed_amount_l507_50726

theorem borrowed_amount (P : ℝ) (h1 : (9 / 100) * P - (8 / 100) * P = 200) : P = 20000 :=
  by sorry

end NUMINAMATH_GPT_borrowed_amount_l507_50726


namespace NUMINAMATH_GPT_solution_set_of_inequality_l507_50729

-- Definition of the inequality and its transformation
def inequality (x : ℝ) : Prop :=
  (x - 2) / (x + 1) ≤ 0

noncomputable def transformed_inequality (x : ℝ) : Prop :=
  (x + 1) * (x - 2) ≤ 0 ∧ x + 1 ≠ 0

-- Statement of the theorem
theorem solution_set_of_inequality :
  {x : ℝ | inequality x} = {x : ℝ | -1 < x ∧ x ≤ 2} := 
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l507_50729


namespace NUMINAMATH_GPT_edward_rides_l507_50776

theorem edward_rides (total_tickets tickets_spent tickets_per_ride rides : ℕ)
    (h1 : total_tickets = 79)
    (h2 : tickets_spent = 23)
    (h3 : tickets_per_ride = 7)
    (h4 : rides = (total_tickets - tickets_spent) / tickets_per_ride) :
    rides = 8 := by sorry

end NUMINAMATH_GPT_edward_rides_l507_50776


namespace NUMINAMATH_GPT_chord_length_invalid_l507_50706

-- Define the circle radius
def radius : ℝ := 5

-- Define the maximum possible chord length in terms of the diameter
def max_chord_length (r : ℝ) : ℝ := 2 * r

-- The problem statement proving that 11 cannot be a chord length given the radius is 5
theorem chord_length_invalid : ¬ (11 ≤ max_chord_length radius) :=
by {
  sorry
}

end NUMINAMATH_GPT_chord_length_invalid_l507_50706


namespace NUMINAMATH_GPT_Haley_boxes_needed_l507_50716

theorem Haley_boxes_needed (TotalMagazines : ℕ) (MagazinesPerBox : ℕ) 
  (h1 : TotalMagazines = 63) (h2 : MagazinesPerBox = 9) : 
  TotalMagazines / MagazinesPerBox = 7 := by
sorry

end NUMINAMATH_GPT_Haley_boxes_needed_l507_50716


namespace NUMINAMATH_GPT_lisa_needs_4_weeks_to_eat_all_candies_l507_50778

-- Define the number of candies Lisa has initially.
def candies_initial : ℕ := 72

-- Define the number of candies Lisa eats per week based on the given conditions.
def candies_per_week : ℕ := (3 * 2) + (2 * 2) + (4 * 2) + 1

-- Define the number of weeks it takes for Lisa to eat all the candies.
def weeks_to_eat_all_candies (candies : ℕ) (weekly_candies : ℕ) : ℕ := 
  (candies + weekly_candies - 1) / weekly_candies

-- The theorem statement that proves Lisa needs 4 weeks to eat all 72 candies.
theorem lisa_needs_4_weeks_to_eat_all_candies :
  weeks_to_eat_all_candies candies_initial candies_per_week = 4 :=
by
  sorry

end NUMINAMATH_GPT_lisa_needs_4_weeks_to_eat_all_candies_l507_50778


namespace NUMINAMATH_GPT_first_interest_rate_is_correct_l507_50782

theorem first_interest_rate_is_correct :
  let A1 := 1500.0000000000007
  let A2 := 2500 - A1
  let yearly_income := 135
  (15.0 * (r / 100) + 6.0 * (A2 / 100) = yearly_income) -> r = 5.000000000000003 :=
sorry

end NUMINAMATH_GPT_first_interest_rate_is_correct_l507_50782


namespace NUMINAMATH_GPT_union_rational_irrational_is_real_intersection_rational_irrational_is_empty_l507_50727

section
  def A : Set ℝ := {x : ℝ | ∃ q : ℚ, x = q}
  def B : Set ℝ := {x : ℝ | ¬ ∃ q : ℚ, x = q}

  theorem union_rational_irrational_is_real : A ∪ B = Set.univ :=
  by
    sorry

  theorem intersection_rational_irrational_is_empty : A ∩ B = ∅ :=
  by
    sorry
end

end NUMINAMATH_GPT_union_rational_irrational_is_real_intersection_rational_irrational_is_empty_l507_50727


namespace NUMINAMATH_GPT_question_1_part_1_question_1_part_2_question_2_l507_50777

universe u

variables (U : Type u) [PartialOrder U]
noncomputable def A : Set ℝ := {x | (x - 2) * (x - 9) < 0}
noncomputable def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
noncomputable def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 2 - a }

theorem question_1_part_1 : A ∩ B = {x | 2 < x ∧ x ≤ 5} :=
sorry

theorem question_1_part_2 : B ∪ (Set.compl A) = {x | x ≤ 5 ∨ x ≥ 9} :=
sorry

theorem question_2 (a : ℝ) (h : C a ∪ (Set.compl B) = Set.univ) : a ≤ -3 :=
sorry

end NUMINAMATH_GPT_question_1_part_1_question_1_part_2_question_2_l507_50777


namespace NUMINAMATH_GPT_min_value_of_M_l507_50707

theorem min_value_of_M (P : ℕ → ℝ) (n : ℕ) (M : ℝ):
  (P 1 = 9 / 11) →
  (∀ n ≥ 2, P n = (3 / 4) * (P (n - 1)) + (2 / 3) * (1 - P (n - 1))) →
  (∀ n ≥ 2, P n ≤ M) →
  (M = 97 / 132) := 
sorry

end NUMINAMATH_GPT_min_value_of_M_l507_50707


namespace NUMINAMATH_GPT_polynomial_inequality_l507_50743

-- Define the polynomial P and its condition
def P (a b c : ℝ) (x : ℝ) : ℝ := 12 * x^3 + a * x^2 + b * x + c
-- Define the polynomial Q and its condition
def Q (a b c : ℝ) (x : ℝ) : ℝ := (x^2 + x + 2001)^3 + a * (x^2 + x + 2001)^2 + b * (x^2 + x + 2001) + c

-- Assumptions
axiom P_has_distinct_roots (a b c : ℝ) : ∃ p q r : ℝ, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ P a b c p = 0 ∧ P a b c q = 0 ∧ P a b c r = 0
axiom Q_has_no_real_roots (a b c : ℝ) : ¬ ∃ x : ℝ, Q a b c x = 0

-- The goal to prove
theorem polynomial_inequality (a b c : ℝ) (h1 : ∃ p q r : ℝ, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ P a b c p = 0 ∧ P a b c q = 0 ∧ P a b c r = 0)
  (h2 : ¬ ∃ x : ℝ, Q a b c x = 0) : 2001^3 + a * 2001^2 + b * 2001 + c > 1 / 64 :=
by {
  -- sorry is added to skip the proof part
  sorry
}

end NUMINAMATH_GPT_polynomial_inequality_l507_50743


namespace NUMINAMATH_GPT_angle_C_in_triangle_l507_50737

theorem angle_C_in_triangle {A B C : ℝ} 
  (h1 : A - B = 10) 
  (h2 : B = 0.5 * A) : 
  C = 150 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_angle_C_in_triangle_l507_50737


namespace NUMINAMATH_GPT_intersection_complement_l507_50766

def A : Set ℝ := {1, 2, 3, 4, 5, 6}
def B : Set ℝ := {x | 2 < x ∧ x < 5 }
def C : Set ℝ := {x | x ≤ 2 ∨ x ≥ 5 }

theorem intersection_complement :
  (A ∩ C) = {1, 2, 5, 6} :=
by sorry

end NUMINAMATH_GPT_intersection_complement_l507_50766


namespace NUMINAMATH_GPT_equation_value_l507_50736

-- Define the expressions
def a := 10 + 3
def b := 7 - 5

-- State the theorem
theorem equation_value : a^2 + b^2 = 173 := by
  sorry

end NUMINAMATH_GPT_equation_value_l507_50736


namespace NUMINAMATH_GPT_problem_solution_l507_50755

theorem problem_solution (k : ℤ) : k ≤ 0 ∧ -2 < k → k = -1 ∨ k = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l507_50755


namespace NUMINAMATH_GPT_range_x1_x2_l507_50720

noncomputable def g (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

def f (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem range_x1_x2 (a b c d x1 x2 : ℝ)
  (h1 : a ≠ 0)
  (h2 : a + 2 * b + 3 * c = 0)
  (h3 : f a b c 0 * f a b c 1 > 0)
  (hx1 : f a b c x1 = 0)
  (hx2 : f a b c x2 = 0) :
  abs (x1 - x2) ∈ Set.Ico 0 (2 / 3) :=
sorry

end NUMINAMATH_GPT_range_x1_x2_l507_50720


namespace NUMINAMATH_GPT_triangle_side_lengths_relationship_l507_50787

variable {a b c : ℝ}

def is_quadratic_mean (a b c : ℝ) : Prop :=
  (2 * b^2 = a^2 + c^2)

def is_geometric_mean (a b c : ℝ) : Prop :=
  (b * a = c^2)

theorem triangle_side_lengths_relationship (a b c : ℝ) :
  (is_quadratic_mean a b c ∧ is_geometric_mean a b c) → 
  ∃ a b c, (2 * b^2 = a^2 + c^2) ∧ (b * a = c^2) :=
sorry

end NUMINAMATH_GPT_triangle_side_lengths_relationship_l507_50787


namespace NUMINAMATH_GPT_find_a_l507_50717

-- Define the conditions and the proof goal
theorem find_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h_eq : a + a⁻¹ = 5/2) :
  a = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l507_50717


namespace NUMINAMATH_GPT_danielle_money_for_supplies_l507_50732

-- Define the conditions
def cost_of_molds := 3
def cost_of_sticks_pack := 1
def sticks_in_pack := 100
def cost_of_juice_bottle := 2
def popsicles_per_bottle := 20
def remaining_sticks := 40
def used_sticks := sticks_in_pack - remaining_sticks

-- Define number of juice bottles used
def bottles_of_juice_used : ℕ := used_sticks / popsicles_per_bottle

-- Define the total cost
def total_cost : ℕ := cost_of_molds + cost_of_sticks_pack + bottles_of_juice_used * cost_of_juice_bottle

-- Prove that Danielle had $10 for supplies
theorem danielle_money_for_supplies : total_cost = 10 := by {
  sorry
}

end NUMINAMATH_GPT_danielle_money_for_supplies_l507_50732


namespace NUMINAMATH_GPT_tan_a_eq_two_imp_cos_2a_plus_sin_2a_eq_one_fifth_l507_50759

theorem tan_a_eq_two_imp_cos_2a_plus_sin_2a_eq_one_fifth (a : ℝ) (h : Real.tan a = 2) :
  Real.cos (2 * a) + Real.sin (2 * a) = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_tan_a_eq_two_imp_cos_2a_plus_sin_2a_eq_one_fifth_l507_50759


namespace NUMINAMATH_GPT_max_temp_range_l507_50756

theorem max_temp_range (avg_temp : ℝ) (lowest_temp : ℝ) (days : ℕ) (total_temp : ℝ) (range : ℝ) : 
  avg_temp = 45 → 
  lowest_temp = 42 → 
  days = 5 → 
  total_temp = avg_temp * days → 
  range = 6 := 
by 
  sorry

end NUMINAMATH_GPT_max_temp_range_l507_50756


namespace NUMINAMATH_GPT_cost_of_cucumbers_l507_50799

theorem cost_of_cucumbers (C : ℝ) (h1 : ∀ (T : ℝ), T = 0.80 * C)
  (h2 : 2 * (0.80 * C) + 3 * C = 23) : C = 5 := by
  sorry

end NUMINAMATH_GPT_cost_of_cucumbers_l507_50799


namespace NUMINAMATH_GPT_sushi_father_lollipops_l507_50731

-- Define the conditions
def lollipops_eaten : ℕ := 5
def lollipops_left : ℕ := 7

-- Define the total number of lollipops brought
def total_lollipops := lollipops_eaten + lollipops_left

-- Proof statement
theorem sushi_father_lollipops : total_lollipops = 12 := sorry

end NUMINAMATH_GPT_sushi_father_lollipops_l507_50731


namespace NUMINAMATH_GPT_unique_zero_iff_a_in_range_l507_50796

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

theorem unique_zero_iff_a_in_range (a : ℝ) :
  (∃ x0 : ℝ, f a x0 = 0 ∧ (∀ x1 : ℝ, f a x1 = 0 → x1 = x0) ∧ x0 > 0) ↔ a < -2 :=
by sorry

end NUMINAMATH_GPT_unique_zero_iff_a_in_range_l507_50796


namespace NUMINAMATH_GPT_total_population_l507_50718

theorem total_population (x T : ℝ) (h : 128 = (x / 100) * (50 / 100) * T) : T = 25600 / x :=
by
  sorry

end NUMINAMATH_GPT_total_population_l507_50718


namespace NUMINAMATH_GPT_total_boys_in_school_l507_50724

-- Define the total percentage of boys belonging to other communities
def percentage_other_communities := 100 - (44 + 28 + 10)

-- Total number of boys in the school, represented by a variable B
def total_boys (B : ℕ) : Prop :=
0.18 * (B : ℝ) = 117

-- The theorem states that the total number of boys B is 650
theorem total_boys_in_school : ∃ B : ℕ, total_boys B ∧ B = 650 :=
sorry

end NUMINAMATH_GPT_total_boys_in_school_l507_50724


namespace NUMINAMATH_GPT_calculate_expression_l507_50761

variable {x y : ℝ}

theorem calculate_expression (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 3 / y) :
  (3 * x - 3 / x) * (3 * y + 3 / y) = 9 * x^2 - y^2 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l507_50761


namespace NUMINAMATH_GPT_remainder_and_division_l507_50704

theorem remainder_and_division (x y : ℕ) (h1 : x % y = 8) (h2 : (x / y : ℝ) = 76.4) : y = 20 :=
sorry

end NUMINAMATH_GPT_remainder_and_division_l507_50704


namespace NUMINAMATH_GPT_larger_of_two_numbers_l507_50765

theorem larger_of_two_numbers
  (A B hcf : ℕ)
  (factor1 factor2 : ℕ)
  (h_hcf : hcf = 23)
  (h_factor1 : factor1 = 9)
  (h_factor2 : factor2 = 10)
  (h_lcm : (A * B) / (hcf) = (hcf * factor1 * factor2))
  (h_A : A = hcf * 9)
  (h_B : B = hcf * 10) :
  max A B = 230 := by
  sorry

end NUMINAMATH_GPT_larger_of_two_numbers_l507_50765


namespace NUMINAMATH_GPT_walking_distance_l507_50793

theorem walking_distance (D : ℕ) (h : D / 15 = (D + 60) / 30) : D = 60 :=
by
  sorry

end NUMINAMATH_GPT_walking_distance_l507_50793


namespace NUMINAMATH_GPT_four_digit_number_l507_50763

theorem four_digit_number (a b c d : ℕ)
    (h1 : 0 ≤ a) (h2 : a ≤ 9)
    (h3 : 0 ≤ b) (h4 : b ≤ 9)
    (h5 : 0 ≤ c) (h6 : c ≤ 9)
    (h7 : 0 ≤ d) (h8 : d ≤ 9)
    (h9 : 2 * (1000 * a + 100 * b + 10 * c + d) + 1000 = 1000 * d + 100 * c + 10 * b + a)
    : (1000 * a + 100 * b + 10 * c + d) = 2996 :=
by
  sorry

end NUMINAMATH_GPT_four_digit_number_l507_50763


namespace NUMINAMATH_GPT_number_of_odd_positive_integer_triples_sum_25_l507_50772

theorem number_of_odd_positive_integer_triples_sum_25 :
  ∃ n : ℕ, (
    n = 78 ∧
    ∃ (a b c : ℕ), 
      (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 25
  ) := 
sorry

end NUMINAMATH_GPT_number_of_odd_positive_integer_triples_sum_25_l507_50772


namespace NUMINAMATH_GPT_find_a_l507_50733

noncomputable def binomial_coeff (n k : ℕ) := Nat.choose n k

theorem find_a (a : ℝ) 
  (h : ∃ (a : ℝ), a ^ 3 * binomial_coeff 8 3 = 56) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l507_50733


namespace NUMINAMATH_GPT_minimal_fraction_difference_l507_50728

theorem minimal_fraction_difference (p q : ℕ) (hp : 0 < p) (hq : 0 < q) 
  (h1 : 3 / 5 < p / q) (h2 : p / q < 2 / 3) (hmin: ∀ r s : ℕ, (3 / 5 < r / s ∧ r / s < 2 / 3 ∧ s < q) → false) :
  q - p = 11 := 
sorry

end NUMINAMATH_GPT_minimal_fraction_difference_l507_50728


namespace NUMINAMATH_GPT_train_length_correct_l507_50749

noncomputable def train_length (speed_kmh: ℝ) (time_s: ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_s

theorem train_length_correct :
  train_length 60 15 = 250.05 := 
by
  sorry

end NUMINAMATH_GPT_train_length_correct_l507_50749


namespace NUMINAMATH_GPT_find_original_number_l507_50705

theorem find_original_number (x : ℝ) : 1.5 * x = 525 → x = 350 := by
  sorry

end NUMINAMATH_GPT_find_original_number_l507_50705


namespace NUMINAMATH_GPT_num_roses_given_l507_50735

theorem num_roses_given (n : ℕ) (m : ℕ) (x : ℕ) :
  n = 28 → 
  (∀ (b g : ℕ), b + g = n → b * g = 45 * x) →
  (num_roses : ℕ) = 4 * x →
  (num_tulips : ℕ) = 10 * num_roses →
  (num_daffodils : ℕ) = x →
  num_roses = 16 :=
by
  sorry

end NUMINAMATH_GPT_num_roses_given_l507_50735


namespace NUMINAMATH_GPT_area_ratio_of_squares_l507_50744

theorem area_ratio_of_squares (s L : ℝ) 
  (H : 4 * L = 4 * 4 * s) : (L^2) = 16 * (s^2) :=
by
  -- assuming the utilization of the given condition
  sorry

end NUMINAMATH_GPT_area_ratio_of_squares_l507_50744


namespace NUMINAMATH_GPT_train_pass_man_in_16_seconds_l507_50725

noncomputable def speed_km_per_hr := 54
noncomputable def speed_m_per_s := (speed_km_per_hr * 1000) / 3600
noncomputable def time_to_pass_platform := 16
noncomputable def length_platform := 90.0072
noncomputable def length_train := speed_m_per_s * time_to_pass_platform
noncomputable def time_to_pass_man := length_train / speed_m_per_s

theorem train_pass_man_in_16_seconds :
  time_to_pass_man = 16 :=
by sorry

end NUMINAMATH_GPT_train_pass_man_in_16_seconds_l507_50725


namespace NUMINAMATH_GPT_prob_A_and_B_truth_is_0_48_l507_50764

-- Conditions: Define the probabilities
def prob_A_truth : ℝ := 0.8
def prob_B_truth : ℝ := 0.6

-- Target: Define the probability that both A and B tell the truth at the same time.
def prob_A_and_B_truth : ℝ := prob_A_truth * prob_B_truth

-- Statement: Prove that the probability that both A and B tell the truth at the same time is 0.48.
theorem prob_A_and_B_truth_is_0_48 : prob_A_and_B_truth = 0.48 := by
  sorry

end NUMINAMATH_GPT_prob_A_and_B_truth_is_0_48_l507_50764


namespace NUMINAMATH_GPT_cos_alpha_minus_pi_l507_50709

theorem cos_alpha_minus_pi (α : Real) (h : Real.sin (α / 2) = Real.sqrt 3 / 4) : 
  Real.cos (α - Real.pi) = -5 / 8 :=
sorry

end NUMINAMATH_GPT_cos_alpha_minus_pi_l507_50709


namespace NUMINAMATH_GPT_average_weight_of_a_and_b_l507_50715

-- Define the parameters in the conditions
variables (A B C : ℝ)

-- Conditions given in the problem
theorem average_weight_of_a_and_b (h1 : (A + B + C) / 3 = 45) 
                                 (h2 : (B + C) / 2 = 43) 
                                 (h3 : B = 33) : (A + B) / 2 = 41 := 
sorry

end NUMINAMATH_GPT_average_weight_of_a_and_b_l507_50715


namespace NUMINAMATH_GPT_interest_time_period_l507_50784

-- Define the constants given in the problem
def principal : ℝ := 4000
def rate1 : ℝ := 0.18
def rate2 : ℝ := 0.12
def interest_difference : ℝ := 480

-- Define the time period T
def time_period : ℝ := 2

-- Define a proof statement
theorem interest_time_period :
  (principal * rate1 * time_period) - (principal * rate2 * time_period) = interest_difference :=
by {
  -- We skip the proof since it's not required by the problem statement
  sorry
}

end NUMINAMATH_GPT_interest_time_period_l507_50784


namespace NUMINAMATH_GPT_scores_greater_than_18_l507_50748

theorem scores_greater_than_18 (scores : Fin 20 → ℝ) 
  (h_unique : Function.Injective scores)
  (h_sum : ∀ i j k : Fin 20, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i : Fin 20, scores i > 18 := 
by
  sorry

end NUMINAMATH_GPT_scores_greater_than_18_l507_50748


namespace NUMINAMATH_GPT_projection_of_point_onto_xOy_plane_l507_50747

def point := (ℝ × ℝ × ℝ)

def projection_onto_xOy_plane (P : point) : point :=
  let (x, y, z) := P
  (x, y, 0)

theorem projection_of_point_onto_xOy_plane : 
  projection_onto_xOy_plane (2, 3, 4) = (2, 3, 0) :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_projection_of_point_onto_xOy_plane_l507_50747


namespace NUMINAMATH_GPT_parabola_line_intersect_l507_50714

theorem parabola_line_intersect (a : ℝ) (b : ℝ) (h1 : a ≠ 0) (h2 : ∀ x : ℝ, (y = a * x^2) ↔ (y = 2 * x - 3) → (x, y) = (1, -1)) :
  a = -1 ∧ b = -1 ∧ ((x, y) = (-3, -9) ∨ (x, y) = (1, -1)) := by
  sorry

end NUMINAMATH_GPT_parabola_line_intersect_l507_50714


namespace NUMINAMATH_GPT_sarah_ellie_total_reflections_l507_50703

def sarah_tall_reflections : ℕ := 10
def sarah_wide_reflections : ℕ := 5
def sarah_narrow_reflections : ℕ := 8

def ellie_tall_reflections : ℕ := 6
def ellie_wide_reflections : ℕ := 3
def ellie_narrow_reflections : ℕ := 4

def tall_mirror_passages : ℕ := 3
def wide_mirror_passages : ℕ := 5
def narrow_mirror_passages : ℕ := 4

def total_reflections (sarah_tall sarah_wide sarah_narrow ellie_tall ellie_wide ellie_narrow
    tall_passages wide_passages narrow_passages : ℕ) : ℕ :=
  (sarah_tall * tall_passages + sarah_wide * wide_passages + sarah_narrow * narrow_passages) +
  (ellie_tall * tall_passages + ellie_wide * wide_passages + ellie_narrow * narrow_passages)

theorem sarah_ellie_total_reflections :
  total_reflections sarah_tall_reflections sarah_wide_reflections sarah_narrow_reflections
  ellie_tall_reflections ellie_wide_reflections ellie_narrow_reflections
  tall_mirror_passages wide_mirror_passages narrow_mirror_passages = 136 :=
by
  sorry

end NUMINAMATH_GPT_sarah_ellie_total_reflections_l507_50703


namespace NUMINAMATH_GPT_replace_movies_cost_l507_50797

theorem replace_movies_cost
  (num_movies : ℕ)
  (trade_in_value_per_vhs : ℕ)
  (cost_per_dvd : ℕ)
  (h1 : num_movies = 100)
  (h2 : trade_in_value_per_vhs = 2)
  (h3 : cost_per_dvd = 10):
  (cost_per_dvd - trade_in_value_per_vhs) * num_movies = 800 :=
by sorry

end NUMINAMATH_GPT_replace_movies_cost_l507_50797


namespace NUMINAMATH_GPT_largest_base_b_digits_not_18_l507_50740

-- Definition of the problem:
-- Let n = 12^3 in base 10
def n : ℕ := 12 ^ 3

-- Definition of the conditions:
-- In base 8, 1728 (12^3 in base 10) has its digits sum to 17
def sum_of_digits_base_8 (x : ℕ) : ℕ :=
  let digits := x.digits (8)
  digits.sum

-- Proof statement
theorem largest_base_b_digits_not_18 : ∃ b : ℕ, (max b) = 8 ∧ sum_of_digits_base_8 n ≠ 18 := by
  sorry

end NUMINAMATH_GPT_largest_base_b_digits_not_18_l507_50740


namespace NUMINAMATH_GPT_luke_plays_14_rounds_l507_50752

theorem luke_plays_14_rounds (total_points : ℕ) (points_per_round : ℕ)
  (h1 : total_points = 154) (h2 : points_per_round = 11) : 
  total_points / points_per_round = 14 := by
  sorry

end NUMINAMATH_GPT_luke_plays_14_rounds_l507_50752


namespace NUMINAMATH_GPT_polynomial_sum_equals_one_l507_50770

theorem polynomial_sum_equals_one (a a_1 a_2 a_3 a_4 : ℤ) :
  (∀ x : ℤ, (2*x + 1)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) →
  a - a_1 + a_2 - a_3 + a_4 = 1 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_sum_equals_one_l507_50770


namespace NUMINAMATH_GPT_gingerbread_price_today_is_5_l507_50722

-- Given conditions
variables {x y a b k m : ℤ}

-- Price constraints
axiom price_constraint_yesterday : 9 * x + 7 * y < 100
axiom price_constraint_today1 : 9 * a + 7 * b > 100
axiom price_constraint_today2 : 2 * a + 11 * b < 100

-- Price change constraints
axiom price_change_gingerbread : a = x + k
axiom price_change_pastries : b = y + m
axiom gingerbread_change_range : |k| ≤ 1
axiom pastries_change_range : |m| ≤ 1

theorem gingerbread_price_today_is_5 : a = 5 :=
by
  sorry

end NUMINAMATH_GPT_gingerbread_price_today_is_5_l507_50722


namespace NUMINAMATH_GPT_cake_eaten_after_four_trips_l507_50711

-- Define the fraction of the cake eaten on each trip
def fraction_eaten (n : Nat) : ℚ :=
  (1 / 3) ^ n

-- Define the total cake eaten after four trips
def total_eaten_after_four_trips : ℚ :=
  fraction_eaten 1 + fraction_eaten 2 + fraction_eaten 3 + fraction_eaten 4

-- The mathematical statement we want to prove
theorem cake_eaten_after_four_trips : total_eaten_after_four_trips = 40 / 81 := 
by
  sorry

end NUMINAMATH_GPT_cake_eaten_after_four_trips_l507_50711


namespace NUMINAMATH_GPT_solve_arcsin_cos_eq_x_over_3_l507_50700

noncomputable def arcsin (x : Real) : Real := sorry
noncomputable def cos (x : Real) : Real := sorry

theorem solve_arcsin_cos_eq_x_over_3 :
  ∀ x,
  - (3 * Real.pi / 2) ≤ x ∧ x ≤ 3 * Real.pi / 2 →
  arcsin (cos x) = x / 3 →
  x = 3 * Real.pi / 10 ∨ x = 3 * Real.pi / 8 :=
sorry

end NUMINAMATH_GPT_solve_arcsin_cos_eq_x_over_3_l507_50700


namespace NUMINAMATH_GPT_fraction_meaningful_iff_l507_50798

theorem fraction_meaningful_iff (x : ℝ) : (x ≠ 2) ↔ (x - 2 ≠ 0) := 
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_iff_l507_50798


namespace NUMINAMATH_GPT_find_m_l507_50710

theorem find_m (m : ℝ) (x : ℝ) (h : x = 1) (h_eq : (m / (2 - x)) - (1 / (x - 2)) = 3) : m = 2 :=
sorry

end NUMINAMATH_GPT_find_m_l507_50710


namespace NUMINAMATH_GPT_signup_ways_l507_50708

theorem signup_ways (students groups : ℕ) (h_students : students = 5) (h_groups : groups = 3) :
  (groups ^ students = 243) :=
by
  have calculation : 3 ^ 5 = 243 := by norm_num
  rwa [h_students, h_groups]

end NUMINAMATH_GPT_signup_ways_l507_50708


namespace NUMINAMATH_GPT_find_n_18_l507_50775

def valid_denominations (n : ℕ) : Prop :=
  ∀ k < 106, ∃ a b c : ℕ, k = 7 * a + n * b + (n + 1) * c

def cannot_form_106 (n : ℕ) : Prop :=
  ¬ ∃ a b c : ℕ, 106 = 7 * a + n * b + (n + 1) * c

theorem find_n_18 : 
  ∃ n : ℕ, valid_denominations n ∧ cannot_form_106 n ∧ ∀ m < n, ¬ (valid_denominations m ∧ cannot_form_106 m) :=
sorry

end NUMINAMATH_GPT_find_n_18_l507_50775


namespace NUMINAMATH_GPT_scout_troop_profit_l507_50713

noncomputable def buy_price_per_bar : ℚ := 3 / 4
noncomputable def sell_price_per_bar : ℚ := 2 / 3
noncomputable def num_candy_bars : ℕ := 800

theorem scout_troop_profit :
  num_candy_bars * (sell_price_per_bar : ℚ) - num_candy_bars * (buy_price_per_bar : ℚ) = -66.64 :=
by
  sorry

end NUMINAMATH_GPT_scout_troop_profit_l507_50713
