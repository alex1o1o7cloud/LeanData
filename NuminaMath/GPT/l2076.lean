import Mathlib

namespace NUMINAMATH_GPT_geom_seq_a3_value_l2076_207649

theorem geom_seq_a3_value (a_n : ℕ → ℝ) (h1 : ∃ r : ℝ, ∀ n : ℕ, a_n (n+1) = a_n (1) * r^n) 
                          (h2 : a_n (2) * a_n (4) = 2 * a_n (3) - 1) :
  a_n (3) = 1 :=
sorry

end NUMINAMATH_GPT_geom_seq_a3_value_l2076_207649


namespace NUMINAMATH_GPT_pencils_in_all_l2076_207626

/-- Eugene's initial number of pencils -/
def initial_pencils : ℕ := 51

/-- Pencils Eugene gets from Joyce -/
def additional_pencils : ℕ := 6

/-- Total number of pencils Eugene has in all -/
def total_pencils : ℕ :=
  initial_pencils + additional_pencils

/-- Proof that Eugene has 57 pencils in all -/
theorem pencils_in_all : total_pencils = 57 := by
  sorry

end NUMINAMATH_GPT_pencils_in_all_l2076_207626


namespace NUMINAMATH_GPT_equivalent_function_l2076_207634

theorem equivalent_function :
  (∀ x : ℝ, (76 * x ^ 6) ^ 7 = |x|) :=
by
  sorry

end NUMINAMATH_GPT_equivalent_function_l2076_207634


namespace NUMINAMATH_GPT_sphere_radius_l2076_207698

theorem sphere_radius (R : ℝ) (h : 4 * Real.pi * R^2 = 4 * Real.pi) : R = 1 :=
by
  sorry

end NUMINAMATH_GPT_sphere_radius_l2076_207698


namespace NUMINAMATH_GPT_local_max_2_l2076_207664

noncomputable def f (x m n : ℝ) := 2 * Real.log x - (1 / 2) * m * x^2 - n * x

theorem local_max_2 (m n : ℝ) (h : n = 1 - 2 * m) :
  ∃ m : ℝ, -1/2 < m ∧ (∀ x : ℝ, x > 0 → (∃ U : Set ℝ, IsOpen U ∧ (2 ∈ U) ∧ (∀ y ∈ U, f y m n ≤ f 2 m n))) :=
sorry

end NUMINAMATH_GPT_local_max_2_l2076_207664


namespace NUMINAMATH_GPT_subtracted_number_l2076_207635

def least_sum_is (x y z : ℤ) (a : ℤ) : Prop :=
  (x - a) * (y - 5) * (z - 2) = 1000 ∧ x + y + z = 7

theorem subtracted_number (x y z a : ℤ) (h : least_sum_is x y z a) : a = 30 :=
sorry

end NUMINAMATH_GPT_subtracted_number_l2076_207635


namespace NUMINAMATH_GPT_perimeter_of_rectangle_l2076_207662

theorem perimeter_of_rectangle (DC BC P : ℝ) (hDC : DC = 12) (hArea : 1/2 * DC * BC = 30) : P = 2 * (DC + BC) → P = 34 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_rectangle_l2076_207662


namespace NUMINAMATH_GPT_cost_per_trip_l2076_207659

theorem cost_per_trip (cost_per_pass : ℕ) (num_passes : ℕ) (trips_oldest : ℕ) (trips_youngest : ℕ) :
    cost_per_pass = 100 →
    num_passes = 2 →
    trips_oldest = 35 →
    trips_youngest = 15 →
    (cost_per_pass * num_passes) / (trips_oldest + trips_youngest) = 4 := by
  sorry

end NUMINAMATH_GPT_cost_per_trip_l2076_207659


namespace NUMINAMATH_GPT_last_two_digits_of_7_pow_10_l2076_207658

theorem last_two_digits_of_7_pow_10 :
  (7 ^ 10) % 100 = 49 := by
  sorry

end NUMINAMATH_GPT_last_two_digits_of_7_pow_10_l2076_207658


namespace NUMINAMATH_GPT_y_increase_by_18_when_x_increases_by_12_l2076_207694

theorem y_increase_by_18_when_x_increases_by_12
  (h_slope : ∀ x y: ℝ, (4 * y = 6 * x) ↔ (3 * y = 2 * x)) :
  ∀ Δx : ℝ, Δx = 12 → ∃ Δy : ℝ, Δy = 18 :=
by
  sorry

end NUMINAMATH_GPT_y_increase_by_18_when_x_increases_by_12_l2076_207694


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l2076_207673

theorem quadratic_has_two_distinct_real_roots :
  ∀ x : ℝ, ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (x^2 - 2 * x - 6 = 0 ∧ x = r1 ∨ x = r2) :=
by sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l2076_207673


namespace NUMINAMATH_GPT_minimal_fence_length_l2076_207674

-- Define the conditions as assumptions
axiom side_length : ℝ
axiom num_paths : ℕ
axiom path_length : ℝ

-- Assume the conditions given in the problem
axiom side_length_value : side_length = 50
axiom num_paths_value : num_paths = 13
axiom path_length_value : path_length = 50

-- Define the theorem to be proved
theorem minimal_fence_length : (num_paths * path_length) = 650 := by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_minimal_fence_length_l2076_207674


namespace NUMINAMATH_GPT_inverse_h_l2076_207601

-- definitions of f, g, and h
def f (x : ℝ) : ℝ := 4 * x - 5
def g (x : ℝ) : ℝ := 3 * x + 7
def h (x : ℝ) : ℝ := f (g x)

-- statement of the problem
theorem inverse_h (x : ℝ) : (∃ y : ℝ, h y = x) ∧ ∀ y : ℝ, h y = x → y = (x - 23) / 12 :=
by
  sorry

end NUMINAMATH_GPT_inverse_h_l2076_207601


namespace NUMINAMATH_GPT_john_extra_hours_l2076_207631

theorem john_extra_hours (daily_earnings : ℕ) (hours_worked : ℕ) (bonus : ℕ) (hourly_wage : ℕ) (total_earnings_with_bonus : ℕ) (total_hours_with_bonus : ℕ) : 
  daily_earnings = 80 ∧ 
  hours_worked = 8 ∧ 
  bonus = 20 ∧ 
  hourly_wage = 10 ∧ 
  total_earnings_with_bonus = daily_earnings + bonus ∧
  total_hours_with_bonus = total_earnings_with_bonus / hourly_wage → 
  total_hours_with_bonus - hours_worked = 2 := 
by 
  sorry

end NUMINAMATH_GPT_john_extra_hours_l2076_207631


namespace NUMINAMATH_GPT_ratio_m_over_n_l2076_207645

theorem ratio_m_over_n : 
  ∀ (m n : ℕ) (a b : ℝ),
  let α := (3 : ℝ) / 4
  let β := (19 : ℝ) / 20
  (a = α * b) →
  (a = β * (a * m + b * n) / (m + n)) →
  (n ≠ 0) →
  m / n = 8 / 9 :=
by
  intros m n a b α β hα hβ hn
  sorry

end NUMINAMATH_GPT_ratio_m_over_n_l2076_207645


namespace NUMINAMATH_GPT_maximum_value_of_function_l2076_207607

theorem maximum_value_of_function :
  ∀ (x : ℝ), -2 < x ∧ x < 0 → x + 1 / x ≤ -2 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_function_l2076_207607


namespace NUMINAMATH_GPT_no_such_P_exists_l2076_207686

theorem no_such_P_exists (P : Polynomial ℤ) (r : ℕ) (r_ge_3 : r ≥ 3) (a : Fin r → ℤ)
  (distinct_a : ∀ i j, i ≠ j → a i ≠ a j)
  (P_cycle : ∀ i, P.eval (a i) = a ⟨(i + 1) % r, sorry⟩)
  : False :=
sorry

end NUMINAMATH_GPT_no_such_P_exists_l2076_207686


namespace NUMINAMATH_GPT_booknote_unique_letters_count_l2076_207681

def booknote_set : Finset Char := {'b', 'o', 'k', 'n', 't', 'e'}

theorem booknote_unique_letters_count : booknote_set.card = 6 :=
by
  sorry

end NUMINAMATH_GPT_booknote_unique_letters_count_l2076_207681


namespace NUMINAMATH_GPT_factorize_x2_minus_2x_plus_1_l2076_207632

theorem factorize_x2_minus_2x_plus_1 :
  ∀ (x : ℝ), x^2 - 2 * x + 1 = (x - 1)^2 :=
by
  intro x
  linarith

end NUMINAMATH_GPT_factorize_x2_minus_2x_plus_1_l2076_207632


namespace NUMINAMATH_GPT_discount_for_multiple_rides_l2076_207666

-- Definitions based on given conditions
def ferris_wheel_cost : ℝ := 2.0
def roller_coaster_cost : ℝ := 7.0
def coupon_value : ℝ := 1.0
def total_tickets_needed : ℝ := 7.0

-- The proof problem
theorem discount_for_multiple_rides : 
  (ferris_wheel_cost + roller_coaster_cost) - (total_tickets_needed - coupon_value) = 2.0 :=
by
  sorry

end NUMINAMATH_GPT_discount_for_multiple_rides_l2076_207666


namespace NUMINAMATH_GPT_num_winners_is_4_l2076_207637

variables (A B C D : Prop)

-- Conditions
axiom h1 : A → B
axiom h2 : B → (C ∨ ¬ A)
axiom h3 : ¬ D → (A ∧ ¬ C)
axiom h4 : D → A

-- Assumptions
axiom hA : A
axiom hD : D

-- Statement to prove
theorem num_winners_is_4 : A ∧ B ∧ C ∧ D :=
by {
  sorry
}

end NUMINAMATH_GPT_num_winners_is_4_l2076_207637


namespace NUMINAMATH_GPT_student_selection_problem_l2076_207625

noncomputable def total_selections : ℕ :=
  let C := Nat.choose
  let A := Nat.factorial
  (C 3 1 * C 3 2 + C 3 2 * C 3 1 + C 3 3) * A 3

theorem student_selection_problem :
  total_selections = 114 :=
by
  sorry

end NUMINAMATH_GPT_student_selection_problem_l2076_207625


namespace NUMINAMATH_GPT_probability_same_spot_l2076_207651

theorem probability_same_spot :
  let students := ["A", "B"]
  let spots := ["Spot 1", "Spot 2"]
  let total_outcomes := [(("A", "Spot 1"), ("B", "Spot 1")),
                         (("A", "Spot 1"), ("B", "Spot 2")),
                         (("A", "Spot 2"), ("B", "Spot 1")),
                         (("A", "Spot 2"), ("B", "Spot 2"))]
  let favorable_outcomes := [(("A", "Spot 1"), ("B", "Spot 1")),
                             (("A", "Spot 2"), ("B", "Spot 2"))]
  ∀ (students : List String) (spots : List String)
    (total_outcomes favorable_outcomes : List ((String × String) × (String × String))),
  (students = ["A", "B"]) →
  (spots = ["Spot 1", "Spot 2"]) →
  (total_outcomes = [(("A", "Spot 1"), ("B", "Spot 1")),
                     (("A", "Spot 1"), ("B", "Spot 2")),
                     (("A", "Spot 2"), ("B", "Spot 1")),
                     (("A", "Spot 2"), ("B", "Spot 2"))]) →
  (favorable_outcomes = [(("A", "Spot 1"), ("B", "Spot 1")),
                         (("A", "Spot 2"), ("B", "Spot 2"))]) →
  favorable_outcomes.length / total_outcomes.length = 1 / 2 := 
by
  intros
  sorry

end NUMINAMATH_GPT_probability_same_spot_l2076_207651


namespace NUMINAMATH_GPT_arithmetic_sequence_a₄_l2076_207628

open Int

noncomputable def S (a₁ d n : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_a₄ {a₁ d : ℤ}
  (h₁ : S a₁ d 5 = 15) (h₂ : S a₁ d 9 = 63) :
  a₁ + 3 * d = 5 :=
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a₄_l2076_207628


namespace NUMINAMATH_GPT_find_fruit_cost_l2076_207616

-- Define the conditions
def muffin_cost : ℝ := 2
def francis_muffin_count : ℕ := 2
def francis_fruit_count : ℕ := 2
def kiera_muffin_count : ℕ := 2
def kiera_fruit_count : ℕ := 1
def total_cost : ℝ := 17

-- Define the cost of each fruit cup
variable (F : ℝ)

-- The statement to be proved
theorem find_fruit_cost (h : francis_muffin_count * muffin_cost 
                + francis_fruit_count * F 
                + kiera_muffin_count * muffin_cost 
                + kiera_fruit_count * F = total_cost) : 
                F = 1.80 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_fruit_cost_l2076_207616


namespace NUMINAMATH_GPT_parabola_directrix_l2076_207663

theorem parabola_directrix (x : ℝ) :
  (∃ y : ℝ, y = (x^2 - 8*x + 12) / 16) →
  ∃ directrix : ℝ, directrix = -17 / 4 :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_l2076_207663


namespace NUMINAMATH_GPT_travis_total_cost_l2076_207646

namespace TravelCost

def cost_first_leg : ℝ := 1500
def discount_first_leg : ℝ := 0.25
def fees_first_leg : ℝ := 100

def cost_second_leg : ℝ := 800
def discount_second_leg : ℝ := 0.20
def fees_second_leg : ℝ := 75

def cost_third_leg : ℝ := 1200
def discount_third_leg : ℝ := 0.35
def fees_third_leg : ℝ := 120

def discounted_cost (cost : ℝ) (discount : ℝ) : ℝ :=
  cost - (cost * discount)

def total_leg_cost (cost : ℝ) (discount : ℝ) (fees : ℝ) : ℝ :=
  (discounted_cost cost discount) + fees

def total_journey_cost : ℝ :=
  total_leg_cost cost_first_leg discount_first_leg fees_first_leg + 
  total_leg_cost cost_second_leg discount_second_leg fees_second_leg + 
  total_leg_cost cost_third_leg discount_third_leg fees_third_leg

theorem travis_total_cost : total_journey_cost = 2840 := by
  sorry

end TravelCost

end NUMINAMATH_GPT_travis_total_cost_l2076_207646


namespace NUMINAMATH_GPT_find_intersection_pair_l2076_207697

def cubic_function (x : ℝ) : ℝ := x^3 - 3*x + 2

def linear_function (x y : ℝ) : Prop := x + 4*y = 4

def intersection_points (x y : ℝ) : Prop := 
  linear_function x y ∧ y = cubic_function x

def sum_x_coord (points : List (ℝ × ℝ)) : ℝ :=
  points.map Prod.fst |>.sum

def sum_y_coord (points : List (ℝ × ℝ)) : ℝ :=
  points.map Prod.snd |>.sum

theorem find_intersection_pair (x1 x2 x3 y1 y2 y3 : ℝ) 
  (h1 : intersection_points x1 y1)
  (h2 : intersection_points x2 y2)
  (h3 : intersection_points x3 y3)
  (h_sum_x : sum_x_coord [(x1, y1), (x2, y2), (x3, y3)] = 0) :
  sum_y_coord [(x1, y1), (x2, y2), (x3, y3)] = 3 :=
sorry

end NUMINAMATH_GPT_find_intersection_pair_l2076_207697


namespace NUMINAMATH_GPT_total_cost_is_53_l2076_207660

-- Defining the costs and quantities as constants
def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def num_sandwiches : ℕ := 7
def num_sodas : ℕ := 10
def discount : ℕ := 5

-- Get the cost of sandwiches purchased
def cost_of_sandwiches : ℕ := num_sandwiches * sandwich_cost

-- Get the cost of sodas purchased
def cost_of_sodas : ℕ := num_sodas * soda_cost

-- Calculate the total cost before discount
def total_cost_before_discount : ℕ := cost_of_sandwiches + cost_of_sodas

-- Calculate the total cost after discount
def total_cost_after_discount : ℕ := total_cost_before_discount - discount

-- The theorem stating that the total cost is 53 dollars
theorem total_cost_is_53 : total_cost_after_discount = 53 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_is_53_l2076_207660


namespace NUMINAMATH_GPT_proposition_true_and_negation_false_l2076_207680

theorem proposition_true_and_negation_false (a b : ℝ) : 
  (a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1)) ∧ ¬(a + b ≥ 2 → (a < 1 ∧ b < 1)) :=
by {
  sorry
}

end NUMINAMATH_GPT_proposition_true_and_negation_false_l2076_207680


namespace NUMINAMATH_GPT_jessy_initial_reading_plan_l2076_207677

theorem jessy_initial_reading_plan (x : ℕ) (h : (7 * (3 * x + 2) = 140)) : x = 6 :=
sorry

end NUMINAMATH_GPT_jessy_initial_reading_plan_l2076_207677


namespace NUMINAMATH_GPT_theorem_incorrect_statement_D_l2076_207675

open Real

def incorrect_statement_D (φ : ℝ) (hφ : φ > 0) (x : ℝ) : Prop :=
  cos (2*x + φ) ≠ cos (2*(x - φ/2))

theorem theorem_incorrect_statement_D (φ : ℝ) (hφ : φ > 0) : 
  ∃ x : ℝ, incorrect_statement_D φ hφ x :=
by
  sorry

end NUMINAMATH_GPT_theorem_incorrect_statement_D_l2076_207675


namespace NUMINAMATH_GPT_percentage_time_in_park_l2076_207669

/-- Define the number of trips Laura takes to the park. -/
def number_of_trips : ℕ := 6

/-- Define time spent at the park per trip in hours. -/
def time_at_park_per_trip : ℝ := 2

/-- Define time spent walking per trip in hours. -/
def time_walking_per_trip : ℝ := 0.5

/-- Define the total time for all trips. -/
def total_time_for_all_trips : ℝ := (time_at_park_per_trip + time_walking_per_trip) * number_of_trips

/-- Define the total time spent in the park for all trips. -/
def total_time_in_park : ℝ := time_at_park_per_trip * number_of_trips

/-- Prove that the percentage of the total time spent in the park is 80%. -/
theorem percentage_time_in_park : total_time_in_park / total_time_for_all_trips * 100 = 80 :=
by
  sorry

end NUMINAMATH_GPT_percentage_time_in_park_l2076_207669


namespace NUMINAMATH_GPT_a_can_complete_in_6_days_l2076_207648

noncomputable def rate_b : ℚ := 1/8
noncomputable def rate_c : ℚ := 1/12
noncomputable def earnings_total : ℚ := 2340
noncomputable def earnings_b : ℚ := 780.0000000000001

theorem a_can_complete_in_6_days :
  ∃ (rate_a : ℚ), 
    (1 / rate_a) = 6 ∧
    rate_a + rate_b + rate_c = 3 * rate_b ∧
    earnings_b = (rate_b / (rate_a + rate_b + rate_c)) * earnings_total := sorry

end NUMINAMATH_GPT_a_can_complete_in_6_days_l2076_207648


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l2076_207613

noncomputable def problem1 (a b : ℝ) (A B : ℝ) (h1 : b * Real.cos A - a * Real.sin B = 0) : Real := 
  A

noncomputable def problem2 (a b c : ℝ) (A : ℝ) (area : ℝ) (h1 : b = Real.sqrt 2) (h2 : A = Real.pi / 4) (h3 : area = 1) : Real :=
  a

theorem problem1_solution (a b : ℝ) (A B : ℝ) (h1 : b * Real.cos A - a * Real.sin B = 0) :
  problem1 a b A B h1 = Real.pi / 4 :=
sorry

theorem problem2_solution (a b c : ℝ) (A : ℝ) (area : ℝ) (h1 : b = Real.sqrt 2) (h2 : A = Real.pi / 4) (h3 : area = 1) :
  problem2 a b c A area h1 h2 h3 = Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l2076_207613


namespace NUMINAMATH_GPT_fraction_absent_l2076_207682

theorem fraction_absent (total_students present_students : ℕ) (h1 : total_students = 28) (h2 : present_students = 20) : 
  (total_students - present_students) / total_students = 2 / 7 :=
by
  sorry

end NUMINAMATH_GPT_fraction_absent_l2076_207682


namespace NUMINAMATH_GPT_ratio_of_common_differences_l2076_207693

variable (x y d1 d2 : ℝ)

theorem ratio_of_common_differences (d1_nonzero : d1 ≠ 0) (d2_nonzero : d2 ≠ 0) 
  (seq1 : x + 4 * d1 = y) (seq2 : x + 5 * d2 = y) : d1 / d2 = 5 / 4 := 
sorry

end NUMINAMATH_GPT_ratio_of_common_differences_l2076_207693


namespace NUMINAMATH_GPT_divisor_value_l2076_207620

theorem divisor_value :
  ∃ D : ℕ, 
    (242 % D = 11) ∧
    (698 % D = 18) ∧
    (365 % D = 15) ∧
    (527 % D = 13) ∧
    ((242 + 698 + 365 + 527) % D = 9) ∧
    (D = 48) :=
sorry

end NUMINAMATH_GPT_divisor_value_l2076_207620


namespace NUMINAMATH_GPT_difference_of_two_numbers_l2076_207652

theorem difference_of_two_numbers 
(x y : ℝ) 
(h1 : x + y = 20) 
(h2 : x^2 - y^2 = 160) : 
  x - y = 8 := 
by 
  sorry

end NUMINAMATH_GPT_difference_of_two_numbers_l2076_207652


namespace NUMINAMATH_GPT_distance_between_D_and_E_l2076_207638

theorem distance_between_D_and_E 
  (A B C D E P : Type)
  (d_AB : ℕ) (d_BC : ℕ) (d_AC : ℕ) (d_PC : ℕ) 
  (AD_parallel_BC : Prop) (AB_parallel_CE : Prop) 
  (distance_DE : ℕ) :
  d_AB = 15 →
  d_BC = 18 → 
  d_AC = 21 → 
  d_PC = 7 → 
  AD_parallel_BC →
  AB_parallel_CE →
  distance_DE = 15 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_D_and_E_l2076_207638


namespace NUMINAMATH_GPT_matrix_pow_six_identity_l2076_207695

variable {n : Type} [Fintype n] [DecidableEq n]
variables {A B C : Matrix n n ℂ}

theorem matrix_pow_six_identity 
  (h1 : A^2 = B^2) (h2 : B^2 = C^2) (h3 : B^3 = A * B * C + 2 * (1 : Matrix n n ℂ)) : 
  A^6 = 1 :=
by 
  sorry

end NUMINAMATH_GPT_matrix_pow_six_identity_l2076_207695


namespace NUMINAMATH_GPT_sixth_student_stickers_l2076_207614

-- Define the given conditions.
def first_student_stickers := 29
def increment := 6

-- Define the number of stickers given to each subsequent student.
def stickers (n : ℕ) : ℕ :=
  first_student_stickers + n * increment

-- Theorem statement: the 6th student will receive 59 stickers.
theorem sixth_student_stickers : stickers 5 = 59 :=
by
  sorry

end NUMINAMATH_GPT_sixth_student_stickers_l2076_207614


namespace NUMINAMATH_GPT_linear_eq_m_minus_2n_zero_l2076_207657

theorem linear_eq_m_minus_2n_zero (m n : ℕ) (x y : ℝ) 
  (h1 : 2 * x ^ (m - 1) + 3 * y ^ (2 * n - 1) = 7)
  (h2 : m - 1 = 1) (h3 : 2 * n - 1 = 1) : 
  m - 2 * n = 0 := 
sorry

end NUMINAMATH_GPT_linear_eq_m_minus_2n_zero_l2076_207657


namespace NUMINAMATH_GPT_tangent_line_at_1_1_l2076_207683

noncomputable def f (x : ℝ) : ℝ := x / (2 * x - 1)

theorem tangent_line_at_1_1 :
  let m := -((2 * 1 - 1 - 2 * 1) / (2 * 1 - 1)^2) -- Derivative evaluated at x = 1
  let tangent_line (x y : ℝ) := x + y - 2
  ∀ x y : ℝ, tangent_line x y = 0 → (f x = y ∧ x = 1 → y = 1 → m = -1) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_1_1_l2076_207683


namespace NUMINAMATH_GPT_number_of_possible_values_of_a_l2076_207653

theorem number_of_possible_values_of_a :
  ∃ a_values : Finset ℕ, 
    (∀ a ∈ a_values, 5 ∣ a) ∧ 
    (∀ a ∈ a_values, a ∣ 30) ∧ 
    (∀ a ∈ a_values, 0 < a) ∧ 
    a_values.card = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_possible_values_of_a_l2076_207653


namespace NUMINAMATH_GPT_basket_white_ball_probability_l2076_207687

noncomputable def basket_problem_proof : Prop :=
  let P_A := 1 / 2
  let P_B := 1 / 2
  let P_W_given_A := 2 / 5
  let P_W_given_B := 1 / 4
  let P_W := P_A * P_W_given_A + P_B * P_W_given_B
  let P_A_given_W := (P_A * P_W_given_A) / P_W
  P_A_given_W = 8 / 13

theorem basket_white_ball_probability :
  basket_problem_proof :=
  sorry

end NUMINAMATH_GPT_basket_white_ball_probability_l2076_207687


namespace NUMINAMATH_GPT_shaded_area_equals_l2076_207619

noncomputable def area_shaded_figure (R : ℝ) : ℝ :=
  let α := (60 : ℝ) * (Real.pi / 180)
  (2 * Real.pi * R^2) / 3

theorem shaded_area_equals : ∀ R : ℝ, area_shaded_figure R = (2 * Real.pi * R^2) / 3 := sorry

end NUMINAMATH_GPT_shaded_area_equals_l2076_207619


namespace NUMINAMATH_GPT_fewest_occupied_seats_l2076_207688

theorem fewest_occupied_seats (n m : ℕ) (h₁ : n = 150) (h₂ : (m * 4 + 3 < 150)) : m = 37 :=
by
  sorry

end NUMINAMATH_GPT_fewest_occupied_seats_l2076_207688


namespace NUMINAMATH_GPT_fraction_zero_implies_x_half_l2076_207661

theorem fraction_zero_implies_x_half (x : ℝ) (h₁ : (2 * x - 1) / (x + 2) = 0) (h₂ : x ≠ -2) : x = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_fraction_zero_implies_x_half_l2076_207661


namespace NUMINAMATH_GPT_probability_of_rolling_two_exactly_four_times_in_five_rolls_l2076_207621

theorem probability_of_rolling_two_exactly_four_times_in_five_rolls :
  let p := (1 / 6)
  let q := (5 / 6)
  let n := 5
  let k := 4
  let probability := (n.choose k) * p^k * q^(n-k)
  probability = (25 / 7776) :=
by
  let p := (1 / 6)
  let q := (5 / 6)
  let n := 5
  let k := 4
  let probability := (n.choose k) * p^k * q^(n - k)
  have h : probability = (25 / 7776) := sorry
  exact h

end NUMINAMATH_GPT_probability_of_rolling_two_exactly_four_times_in_five_rolls_l2076_207621


namespace NUMINAMATH_GPT_eval_ff_ff_3_l2076_207643

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x + 1

theorem eval_ff_ff_3 : f (f (f (f 3))) = 8 :=
  sorry

end NUMINAMATH_GPT_eval_ff_ff_3_l2076_207643


namespace NUMINAMATH_GPT_negation_example_l2076_207640

theorem negation_example (p : ∀ n : ℕ, n^2 < 2^n) : 
  ¬ (∀ n : ℕ, n^2 < 2^n) ↔ ∃ n : ℕ, n^2 ≥ 2^n :=
by sorry

end NUMINAMATH_GPT_negation_example_l2076_207640


namespace NUMINAMATH_GPT_Jans_original_speed_l2076_207690

theorem Jans_original_speed
  (doubled_speed : ℕ → ℕ) (skips_after_training : ℕ) (time_in_minutes : ℕ) (original_speed : ℕ) :
  (∀ (s : ℕ), doubled_speed s = 2 * s) → 
  skips_after_training = 700 → 
  time_in_minutes = 5 → 
  (original_speed = (700 / 5) / 2) → 
  original_speed = 70 := 
by
  intros h1 h2 h3 h4
  exact h4

end NUMINAMATH_GPT_Jans_original_speed_l2076_207690


namespace NUMINAMATH_GPT_part1_x1_part1_x0_part1_xneg2_general_inequality_l2076_207633

-- Prove inequality for specific values of x
theorem part1_x1 : - (1/2 : ℝ) * (1: ℝ)^2 + 2 * (1: ℝ) < -(1: ℝ) + 5 := by
  sorry

theorem part1_x0 : - (1/2 : ℝ) * (0: ℝ)^2 + 2 * (0: ℝ) < -(0: ℝ) + 5 := by
  sorry

theorem part1_xneg2 : - (1/2 : ℝ) * (-2: ℝ)^2 + 2 * (-2: ℝ) < -(-2: ℝ) + 5 := by
  sorry

-- Prove general inequality for all real x
theorem general_inequality (x : ℝ) : - (1/2 : ℝ) * x^2 + 2 * x < -x + 5 := by
  sorry

end NUMINAMATH_GPT_part1_x1_part1_x0_part1_xneg2_general_inequality_l2076_207633


namespace NUMINAMATH_GPT_angles_in_quadrilateral_l2076_207691

theorem angles_in_quadrilateral (A B C D : ℝ)
    (h : A / B = 1 / 3 ∧ B / C = 3 / 5 ∧ C / D = 5 / 6)
    (sum_angles : A + B + C + D = 360) :
    A = 24 ∧ D = 144 := 
by
    sorry

end NUMINAMATH_GPT_angles_in_quadrilateral_l2076_207691


namespace NUMINAMATH_GPT_time_for_Harish_to_paint_alone_l2076_207699

theorem time_for_Harish_to_paint_alone (H : ℝ) (h1 : H > 0) (h2 :  (1 / 6 + 1 / H) = 1 / 2 ) : H = 3 :=
sorry

end NUMINAMATH_GPT_time_for_Harish_to_paint_alone_l2076_207699


namespace NUMINAMATH_GPT_debt_payments_l2076_207602

noncomputable def average_payment (total_amount : ℕ) (payments : ℕ) : ℕ := total_amount / payments

theorem debt_payments (x : ℕ) :
  8 * x + 44 * (x + 65) = 52 * 465 → x = 410 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_debt_payments_l2076_207602


namespace NUMINAMATH_GPT_no_solution_iff_m_range_l2076_207600

theorem no_solution_iff_m_range (m : ℝ) : 
  ¬ ∃ x : ℝ, |x-1| + |x-m| < 2*m ↔ (0 < m ∧ m < 1/3) := sorry

end NUMINAMATH_GPT_no_solution_iff_m_range_l2076_207600


namespace NUMINAMATH_GPT_average_speed_is_70_l2076_207603

theorem average_speed_is_70 
  (distance1 distance2 : ℕ) (time1 time2 : ℕ)
  (h1 : distance1 = 80) (h2 : distance2 = 60)
  (h3 : time1 = 1) (h4 : time2 = 1) :
  (distance1 + distance2) / (time1 + time2) = 70 := 
by 
  sorry

end NUMINAMATH_GPT_average_speed_is_70_l2076_207603


namespace NUMINAMATH_GPT_intersection_of_medians_x_coord_l2076_207627

def parabola (x : ℝ) : ℝ := x^2 - 4 * x - 1

theorem intersection_of_medians_x_coord (x_a x_b : ℝ) (y : ℝ) :
  (parabola x_a = y) ∧ (parabola x_b = y) ∧ (parabola 5 = parabola 5) → 
  (2 : ℝ) < ((5 + 4) / 3) :=
sorry

end NUMINAMATH_GPT_intersection_of_medians_x_coord_l2076_207627


namespace NUMINAMATH_GPT_ice_skating_rinks_and_ski_resorts_2019_l2076_207668

theorem ice_skating_rinks_and_ski_resorts_2019 (x y : ℕ) :
  x + y = 1230 →
  2 * x + 212 + y + 288 = 2560 →
  x = 830 ∧ y = 400 :=
by {
  sorry
}

end NUMINAMATH_GPT_ice_skating_rinks_and_ski_resorts_2019_l2076_207668


namespace NUMINAMATH_GPT_value_of_expression_l2076_207665

variables (u v w : ℝ)

theorem value_of_expression (h1 : u = 3 * v) (h2 : w = 5 * u) : 2 * v + u + w = 20 * v :=
by sorry

end NUMINAMATH_GPT_value_of_expression_l2076_207665


namespace NUMINAMATH_GPT_price_reduction_correct_l2076_207623

theorem price_reduction_correct :
  ∃ x : ℝ, (0.3 - x) * (500 + 4000 * x) = 180 ∧ x = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_price_reduction_correct_l2076_207623


namespace NUMINAMATH_GPT_special_discount_percentage_l2076_207654

theorem special_discount_percentage (original_price discounted_price : ℝ) (h₀ : original_price = 80) (h₁ : discounted_price = 68) : 
  ((original_price - discounted_price) / original_price) * 100 = 15 :=
by 
  sorry

end NUMINAMATH_GPT_special_discount_percentage_l2076_207654


namespace NUMINAMATH_GPT_initial_markup_percentage_l2076_207642

theorem initial_markup_percentage
  (cost_price : ℝ := 100)
  (profit_percentage : ℝ := 14)
  (discount_percentage : ℝ := 5)
  (selling_price : ℝ := cost_price * (1 + profit_percentage / 100))
  (x : ℝ := 20) :
  (cost_price + cost_price * x / 100) * (1 - discount_percentage / 100) = selling_price := by
  sorry

end NUMINAMATH_GPT_initial_markup_percentage_l2076_207642


namespace NUMINAMATH_GPT_marble_ratio_l2076_207612

theorem marble_ratio (W L M : ℕ) (h1 : W = 16) (h2 : L = W + W / 4) (h3 : W + L + M = 60) :
  M / (W + L) = 2 / 3 := 
sorry

end NUMINAMATH_GPT_marble_ratio_l2076_207612


namespace NUMINAMATH_GPT_evaluate_expression_l2076_207670

theorem evaluate_expression : (4^150 * 9^152) / 6^301 = 27 / 2 := 
by 
  -- skipping the actual proof
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2076_207670


namespace NUMINAMATH_GPT_carols_father_gave_5_peanuts_l2076_207679

theorem carols_father_gave_5_peanuts : 
  ∀ (c: ℕ) (f: ℕ), c = 2 → c + f = 7 → f = 5 :=
by
  intros c f h1 h2
  sorry

end NUMINAMATH_GPT_carols_father_gave_5_peanuts_l2076_207679


namespace NUMINAMATH_GPT_total_work_completed_in_days_l2076_207696

theorem total_work_completed_in_days (T : ℕ) :
  (amit_days amit_worked ananthu_days remaining_work : ℕ) → 
  amit_days = 3 → amit_worked = amit_days * (1 / 15) → 
  ananthu_days = 36 → 
  remaining_work = 1 - amit_worked  →
  (ananthu_days * (1 / 45)) = remaining_work →
  T = amit_days + ananthu_days →
  T = 39 := 
sorry

end NUMINAMATH_GPT_total_work_completed_in_days_l2076_207696


namespace NUMINAMATH_GPT_annies_classmates_count_l2076_207615

theorem annies_classmates_count (spent : ℝ) (cost_per_candy : ℝ) (candies_left : ℕ) (candies_per_classmate : ℕ) (expected_classmates : ℕ):
  spent = 8 ∧ cost_per_candy = 0.1 ∧ candies_left = 12 ∧ candies_per_classmate = 2 ∧ expected_classmates = 34 →
  (spent / cost_per_candy) - candies_left = (expected_classmates * candies_per_classmate) := 
by
  intros h
  sorry

end NUMINAMATH_GPT_annies_classmates_count_l2076_207615


namespace NUMINAMATH_GPT_base7_and_base13_addition_l2076_207630

def base7_to_nat (a b c : ℕ) : ℕ := a * 49 + b * 7 + c

def base13_to_nat (a b c : ℕ) : ℕ := a * 169 + b * 13 + c

theorem base7_and_base13_addition (a b c d e f : ℕ) :
  a = 5 → b = 3 → c = 6 → d = 4 → e = 12 → f = 5 →
  base7_to_nat a b c + base13_to_nat d e f = 1109 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  unfold base7_to_nat base13_to_nat
  sorry

end NUMINAMATH_GPT_base7_and_base13_addition_l2076_207630


namespace NUMINAMATH_GPT_water_fee_20_water_fee_55_l2076_207610

-- Define the water charge method as a function
def water_fee (a : ℕ) : ℝ :=
  if a ≤ 15 then 2 * a else 2.5 * a - 7.5

-- Prove the specific cases
theorem water_fee_20 :
  water_fee 20 = 42.5 :=
by sorry

theorem water_fee_55 :
  (∃ a : ℕ, water_fee a = 55) ↔ (a = 25) :=
by sorry

end NUMINAMATH_GPT_water_fee_20_water_fee_55_l2076_207610


namespace NUMINAMATH_GPT_can_form_set_l2076_207608

-- Define each group of objects based on given conditions
def famous_movie_stars : Type := sorry
def small_rivers_in_our_country : Type := sorry
def students_2012_senior_class_Panzhihua : Type := sorry
def difficult_high_school_math_problems : Type := sorry

-- Define the property of having well-defined elements
def has_definite_elements (T : Type) : Prop := sorry

-- The groups in terms of propositions
def group_A : Prop := ¬ has_definite_elements famous_movie_stars
def group_B : Prop := ¬ has_definite_elements small_rivers_in_our_country
def group_C : Prop := has_definite_elements students_2012_senior_class_Panzhihua
def group_D : Prop := ¬ has_definite_elements difficult_high_school_math_problems

-- We need to prove that group C can form a set
theorem can_form_set : group_C :=
by
  sorry

end NUMINAMATH_GPT_can_form_set_l2076_207608


namespace NUMINAMATH_GPT_paint_fraction_l2076_207609

variable (T C : ℕ) (h : T = 60) (t : ℕ) (partial_t : ℚ)

theorem paint_fraction (hT : T = 60) (ht : t = 12) : partial_t = t / T := by
  rw [ht, hT]
  norm_num
  sorry

end NUMINAMATH_GPT_paint_fraction_l2076_207609


namespace NUMINAMATH_GPT_number_99_in_column_4_l2076_207639

-- Definition of the arrangement rule
def column_of (num : ℕ) : ℕ :=
  ((num % 10) + 4) / 2 % 5 + 1

theorem number_99_in_column_4 : 
  column_of 99 = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_99_in_column_4_l2076_207639


namespace NUMINAMATH_GPT_smallest_number_divisible_l2076_207689

theorem smallest_number_divisible (n : ℕ) 
    (h1 : (n - 20) % 15 = 0) 
    (h2 : (n - 20) % 30 = 0)
    (h3 : (n - 20) % 45 = 0)
    (h4 : (n - 20) % 60 = 0) : 
    n = 200 :=
sorry

end NUMINAMATH_GPT_smallest_number_divisible_l2076_207689


namespace NUMINAMATH_GPT_range_of_a_l2076_207618

def A (a : ℝ) := ({-1, 0, a} : Set ℝ)
def B := {x : ℝ | 0 < x ∧ x < 1}

theorem range_of_a (a : ℝ) (h : A a ∩ B ≠ ∅) : 0 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2076_207618


namespace NUMINAMATH_GPT_length_of_pencils_l2076_207678

theorem length_of_pencils (length_pencil1 : ℕ) (length_pencil2 : ℕ)
  (h1 : length_pencil1 = 12) (h2 : length_pencil2 = 12) : length_pencil1 + length_pencil2 = 24 :=
by
  sorry

end NUMINAMATH_GPT_length_of_pencils_l2076_207678


namespace NUMINAMATH_GPT_contractor_fine_per_day_l2076_207656

theorem contractor_fine_per_day
    (total_days : ℕ) 
    (work_days_fine_amt : ℕ) 
    (total_amt : ℕ) 
    (absent_days : ℕ) 
    (worked_days : ℕ := total_days - absent_days)
    (earned_amt : ℕ := worked_days * work_days_fine_amt)
    (fine_per_day : ℚ)
    (total_fine : ℚ := absent_days * fine_per_day) : 
    (earned_amt - total_fine = total_amt) → 
    fine_per_day = 7.5 :=
by
  intros h
  -- proof here is omitted
  sorry

end NUMINAMATH_GPT_contractor_fine_per_day_l2076_207656


namespace NUMINAMATH_GPT_not_always_possible_to_predict_winner_l2076_207604

def football_championship (teams : Fin 16 → ℕ) : Prop :=
  ∃ i j : Fin 16, i ≠ j ∧ teams i = teams j ∧
  ∀ pairs : Fin 16 → Fin 16 × Fin 16,
  (∀ k : Fin 8, (pairs k).fst ≠ (pairs k).snd ∧
               teams (pairs k).fst ≠ teams (pairs k).snd) ∨
  ∃ k : Fin 8, (pairs k).fst = i ∧ (pairs k).snd = j

theorem not_always_possible_to_predict_winner :
  ∀ teams : Fin 16 → ℕ, (∃ i j : Fin 16, i ≠ j ∧ teams i = teams j) →
  ∃ pairs : Fin 16 → Fin 16 × Fin 16,
  (∃ k : Fin 8, teams (pairs k).fst = 15 ∧ teams (pairs k).snd = 15) ↔
  ¬ ∀ pairs : Fin 16 → Fin 16 × Fin 16,
  (∀ k : Fin 8, (pairs k).fst ≠ (pairs k).snd ∧ teams (pairs k).fst ≠ teams (pairs k).snd) :=
by
  sorry

end NUMINAMATH_GPT_not_always_possible_to_predict_winner_l2076_207604


namespace NUMINAMATH_GPT_part_I_part_II_l2076_207636

noncomputable def f (x a : ℝ) : ℝ := |x - a| - |2 * x - 1|

theorem part_I (a : ℝ) (x : ℝ) (h : a = 2) :
    f x a + 3 ≥ 0 ↔ -4 ≤ x ∧ x ≤ 2 := 
by
    -- problem restatement
    sorry

theorem part_II (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f x a ≤ 3) :
    -3 ≤ a ∧ a ≤ 5 := 
by
    -- problem restatement
    sorry

end NUMINAMATH_GPT_part_I_part_II_l2076_207636


namespace NUMINAMATH_GPT_lilly_daily_savings_l2076_207606

-- Conditions
def days_until_birthday : ℕ := 22
def flowers_to_buy : ℕ := 11
def cost_per_flower : ℕ := 4

-- Definition we want to prove
def total_cost : ℕ := flowers_to_buy * cost_per_flower
def daily_savings : ℕ := total_cost / days_until_birthday

theorem lilly_daily_savings : daily_savings = 2 := by
  sorry

end NUMINAMATH_GPT_lilly_daily_savings_l2076_207606


namespace NUMINAMATH_GPT_find_x_l2076_207624

theorem find_x (x : ℝ) (a b : ℝ) (h₀ : a * b = 4 * a - 2 * b)
  (h₁ : 3 * (6 * x) = -2) :
  x = 17 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2076_207624


namespace NUMINAMATH_GPT_product_of_two_numbers_l2076_207671

theorem product_of_two_numbers (a b : ℝ) (h1 : a + b = 70) (h2 : a - b = 10) : a * b = 1200 := 
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l2076_207671


namespace NUMINAMATH_GPT_gas_price_increase_l2076_207650

theorem gas_price_increase (P C : ℝ) (x : ℝ) 
  (h1 : P * C = P * (1 + x) * 1.10 * C * (1 - 0.27272727272727)) :
  x = 0.25 :=
by
  -- The proof will be filled here
  sorry

end NUMINAMATH_GPT_gas_price_increase_l2076_207650


namespace NUMINAMATH_GPT_debts_equal_in_25_days_l2076_207647

-- Define the initial debts and the interest rates
def Darren_initial_debt : ℝ := 200
def Darren_interest_rate : ℝ := 0.08
def Fergie_initial_debt : ℝ := 300
def Fergie_interest_rate : ℝ := 0.04

-- Define the debts as a function of days passed t
def Darren_debt (t : ℝ) : ℝ := Darren_initial_debt * (1 + Darren_interest_rate * t)
def Fergie_debt (t : ℝ) : ℝ := Fergie_initial_debt * (1 + Fergie_interest_rate * t)

-- Prove that Darren and Fergie will owe the same amount in 25 days
theorem debts_equal_in_25_days : ∃ t, Darren_debt t = Fergie_debt t ∧ t = 25 := by
  sorry

end NUMINAMATH_GPT_debts_equal_in_25_days_l2076_207647


namespace NUMINAMATH_GPT_diane_postage_problem_l2076_207622

-- Definition of stamps
def stamps : List (ℕ × ℕ) := [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]

-- Define a function to compute the number of arrangements that sums to a target value
def arrangements_sum_to (target : ℕ) (stamps : List (ℕ × ℕ)) : ℕ :=
  sorry -- Implementation detail is skipped

-- The main theorem to prove
theorem diane_postage_problem :
  arrangements_sum_to 15 stamps = 271 :=
by sorry

end NUMINAMATH_GPT_diane_postage_problem_l2076_207622


namespace NUMINAMATH_GPT_candy_total_l2076_207617

theorem candy_total (r b : ℕ) (hr : r = 145) (hb : b = 3264) : r + b = 3409 := by
  -- We can use Lean's rewrite tactic to handle the equalities, but since proof is skipped,
  -- it's not necessary to write out detailed tactics here.
  sorry

end NUMINAMATH_GPT_candy_total_l2076_207617


namespace NUMINAMATH_GPT_larger_number_of_product_56_and_sum_15_l2076_207605

theorem larger_number_of_product_56_and_sum_15 (x y : ℕ) (h1 : x * y = 56) (h2 : x + y = 15) : max x y = 8 := 
by
  sorry

end NUMINAMATH_GPT_larger_number_of_product_56_and_sum_15_l2076_207605


namespace NUMINAMATH_GPT_unoccupied_volume_in_container_l2076_207655

-- defining constants
def side_length_container := 12
def side_length_ice_cube := 3
def number_of_ice_cubes := 8
def water_fill_fraction := 3 / 4

-- defining volumes
def volume_container := side_length_container ^ 3
def volume_water := volume_container * water_fill_fraction
def volume_ice_cube := side_length_ice_cube ^ 3
def total_volume_ice := volume_ice_cube * number_of_ice_cubes
def volume_unoccupied := volume_container - (volume_water + total_volume_ice)

-- The theorem to be proved
theorem unoccupied_volume_in_container : volume_unoccupied = 216 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_unoccupied_volume_in_container_l2076_207655


namespace NUMINAMATH_GPT_plane_divided_by_n_lines_l2076_207684

-- Definition of the number of regions created by n lines in a plane
def regions (n : ℕ) : ℕ :=
  if n = 0 then 1 else (n * (n + 1)) / 2 + 1 -- Using the given formula directly

-- Theorem statement to prove the formula holds
theorem plane_divided_by_n_lines (n : ℕ) : 
  regions n = (n * (n + 1)) / 2 + 1 :=
sorry

end NUMINAMATH_GPT_plane_divided_by_n_lines_l2076_207684


namespace NUMINAMATH_GPT_range_of_f_l2076_207676

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then |x| - 1 else Real.sin x ^ 2

theorem range_of_f : Set.range f = Set.Ioi (-1) := 
  sorry

end NUMINAMATH_GPT_range_of_f_l2076_207676


namespace NUMINAMATH_GPT_slope_negative_l2076_207644

theorem slope_negative (k b m n : ℝ) (h₁ : k ≠ 0) (h₂ : m < n) 
  (ha : m = k * 1 + b) (hb : n = k * -1 + b) : k < 0 :=
by
  sorry

end NUMINAMATH_GPT_slope_negative_l2076_207644


namespace NUMINAMATH_GPT_sum_of_z_values_l2076_207611

def f (x : ℚ) : ℚ := x^2 + x + 1

theorem sum_of_z_values : ∃ z₁ z₂ : ℚ, f (4 * z₁) = 12 ∧ f (4 * z₂) = 12 ∧ (z₁ + z₂ = - 1 / 12) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_z_values_l2076_207611


namespace NUMINAMATH_GPT_find_C_l2076_207692

theorem find_C (A B C : ℕ) (hA : A = 509) (hAB : A = B + 197) (hCB : C = B - 125) : C = 187 := 
by 
  sorry

end NUMINAMATH_GPT_find_C_l2076_207692


namespace NUMINAMATH_GPT_range_of_a_l2076_207629

theorem range_of_a (a : ℝ) :
  ((∀ x : ℝ, a * x^2 + a * x - 1 < 0) ↔ (-4 < a ∧ a ≤ 0)) :=
sorry

end NUMINAMATH_GPT_range_of_a_l2076_207629


namespace NUMINAMATH_GPT_part_a_part_b_l2076_207641

-- Define the predicate ensuring that among any three consecutive symbols, there is at least one zero
def valid_sequence (s : List Char) : Prop :=
  ∀ (i : Nat), i + 2 < s.length → (s.get! i = '0' ∨ s.get! (i + 1) = '0' ∨ s.get! (i + 2) = '0')

-- Count the valid sequences given the number of 'X's and 'O's
noncomputable def count_valid_sequences (n_zeros n_crosses : Nat) : Nat :=
  sorry -- Implementation of the combinatorial counting

-- Part (a): n = 29
theorem part_a : count_valid_sequences 14 29 = 15 := by
  sorry

-- Part (b): n = 28
theorem part_b : count_valid_sequences 14 28 = 120 := by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l2076_207641


namespace NUMINAMATH_GPT_sum_of_tens_and_units_digit_of_8_pow_100_l2076_207685

noncomputable def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
noncomputable def units_digit (n : ℕ) : ℕ := n % 10
noncomputable def sum_of_digits (n : ℕ) := tens_digit n + units_digit n

theorem sum_of_tens_and_units_digit_of_8_pow_100 : sum_of_digits (8 ^ 100) = 13 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_tens_and_units_digit_of_8_pow_100_l2076_207685


namespace NUMINAMATH_GPT_tangent_line_at_one_l2076_207667

noncomputable def f (x : ℝ) := Real.log x + 2 * x^2 - 4 * x

theorem tangent_line_at_one :
  let slope := (1/x + 4*x - 4) 
  let y_val := -2 
  ∃ (A : ℝ) (B : ℝ) (C : ℝ), A = 1 ∧ B = -1 ∧ C = -3 ∧ (∀ (x y : ℝ), f x = y → A * x + B * y + C = 0) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_one_l2076_207667


namespace NUMINAMATH_GPT_time_to_cross_signal_pole_l2076_207672

-- Given conditions
def length_of_train : ℝ := 300
def time_to_cross_platform : ℝ := 39
def length_of_platform : ℝ := 1162.5

-- The question to prove
theorem time_to_cross_signal_pole :
  (length_of_train / ((length_of_train + length_of_platform) / time_to_cross_platform)) = 8 :=
by
  sorry

end NUMINAMATH_GPT_time_to_cross_signal_pole_l2076_207672
