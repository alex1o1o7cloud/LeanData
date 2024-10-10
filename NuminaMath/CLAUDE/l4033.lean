import Mathlib

namespace apple_weight_average_l4033_403311

theorem apple_weight_average (standard_weight : ℝ) (deviations : List ℝ) : 
  standard_weight = 30 →
  deviations = [0.4, -0.2, -0.8, -0.4, 1, 0.3, 0.5, -2, 0.5, -0.1] →
  (standard_weight + (deviations.sum / deviations.length)) = 29.92 := by
  sorry

end apple_weight_average_l4033_403311


namespace three_card_draw_probability_l4033_403316

/-- Represents a standard deck of 52 cards -/
def StandardDeck : Finset (Fin 52) := Finset.univ

/-- The number of diamonds in a standard deck -/
def numDiamonds : Nat := 13

/-- The number of kings in a standard deck -/
def numKings : Nat := 4

/-- The number of aces in a standard deck -/
def numAces : Nat := 4

/-- The probability of drawing a diamond as the first card, 
    a king as the second card, and an ace as the third card 
    from a standard 52-card deck -/
theorem three_card_draw_probability : 
  (numDiamonds * numKings * numAces : ℚ) / (52 * 51 * 50) = 142 / 66300 := by
  sorry

end three_card_draw_probability_l4033_403316


namespace no_win_prob_at_least_two_no_win_prob_l4033_403330

-- Define the probability of winning for a single bottle
def win_prob : ℚ := 1/6

-- Define the number of students
def num_students : ℕ := 3

-- Theorem 1: Probability that none of the three students win a prize
theorem no_win_prob : 
  (1 - win_prob) ^ num_students = 125/216 := by sorry

-- Theorem 2: Probability that at least two of the three students do not win a prize
theorem at_least_two_no_win_prob : 
  1 - (Nat.choose num_students 2 * win_prob^2 * (1 - win_prob) + win_prob^num_students) = 25/27 := by sorry

end no_win_prob_at_least_two_no_win_prob_l4033_403330


namespace platform_length_l4033_403374

/-- Given a train of length 450 meters that crosses a platform in 60 seconds
    and a signal pole in 30 seconds, prove that the length of the platform is 450 meters. -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 450)
  (h2 : platform_crossing_time = 60)
  (h3 : pole_crossing_time = 30) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 450 := by
  sorry

end platform_length_l4033_403374


namespace problem_statement_l4033_403303

theorem problem_statement (a : ℤ) (h1 : 0 < a) (h2 : a < 13) 
  (h3 : ∃ k : ℤ, 53^2016 + a = 13 * k) : a = 12 :=
by sorry

end problem_statement_l4033_403303


namespace train_crossing_bridge_time_l4033_403370

/-- The time taken for a train to cross a bridge -/
theorem train_crossing_bridge_time 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 375) 
  (h2 : train_speed_kmph = 90) 
  (h3 : bridge_length = 1250) : 
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 65 := by
sorry

end train_crossing_bridge_time_l4033_403370


namespace sqrt_10_parts_product_l4033_403367

theorem sqrt_10_parts_product (x y : ℝ) : 
  (x = ⌊Real.sqrt 10⌋) → 
  (y = Real.sqrt 10 - ⌊Real.sqrt 10⌋) → 
  y * (x + Real.sqrt 10) = 1 := by sorry

end sqrt_10_parts_product_l4033_403367


namespace opposite_numbers_l4033_403321

theorem opposite_numbers (x y z : ℝ) (h : 1/x + 1/y + 1/z = 1/(x+y+z)) :
  x + y = 0 ∨ y + z = 0 ∨ x + z = 0 := by
  sorry

end opposite_numbers_l4033_403321


namespace parallel_line_through_point_l4033_403314

/-- Given a line L1 with equation x - 2y = 0 and a point P (-3, -1),
    prove that the line L2 with equation x - 2y + 1 = 0 passes through P
    and is parallel to L1. -/
theorem parallel_line_through_point (L1 L2 : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  L1 = {(x, y) | x - 2*y = 0} →
  L2 = {(x, y) | x - 2*y + 1 = 0} →
  P = (-3, -1) →
  (P ∈ L2) ∧ (∀ (x y : ℝ), (x, y) ∈ L1 ↔ ∃ (k : ℝ), (x + k, y + k/2) ∈ L2) :=
by sorry

end parallel_line_through_point_l4033_403314


namespace cat_difference_l4033_403362

theorem cat_difference (sheridan_cats garrett_cats : ℕ) 
  (h1 : sheridan_cats = 11) 
  (h2 : garrett_cats = 24) : 
  garrett_cats - sheridan_cats = 13 := by
  sorry

end cat_difference_l4033_403362


namespace log_equation_implies_c_eq_a_to_three_halves_l4033_403307

/-- Given the equation relating logarithms of x with bases c and a, prove that c = a^(3/2) -/
theorem log_equation_implies_c_eq_a_to_three_halves
  (a c x : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_c : 0 < c)
  (h_pos_x : 0 < x)
  (h_eq : 2 * (Real.log x / Real.log c)^2 + 5 * (Real.log x / Real.log a)^2 = 12 * (Real.log x)^2 / (Real.log a * Real.log c)) :
  c = a^(3/2) := by
  sorry

end log_equation_implies_c_eq_a_to_three_halves_l4033_403307


namespace parallelogram_base_length_l4033_403354

/-- Represents a parallelogram with given properties -/
structure Parallelogram where
  area : ℝ
  base : ℝ
  altitude : ℝ
  angle : ℝ
  area_eq : area = 200
  altitude_eq : altitude = 2 * base
  angle_eq : angle = 60

/-- Theorem: The base of the parallelogram with given properties is 10 meters -/
theorem parallelogram_base_length (p : Parallelogram) : p.base = 10 := by
  sorry

end parallelogram_base_length_l4033_403354


namespace bread_calories_eq_100_l4033_403343

/-- Represents the number of calories in a serving of peanut butter -/
def peanut_butter_calories : ℕ := 200

/-- Represents the total desired calories for breakfast -/
def total_calories : ℕ := 500

/-- Represents the number of servings of peanut butter used -/
def peanut_butter_servings : ℕ := 2

/-- Calculates the calories in a piece of bread -/
def bread_calories : ℕ := total_calories - (peanut_butter_calories * peanut_butter_servings)

/-- Proves that the calories in a piece of bread equal 100 -/
theorem bread_calories_eq_100 : bread_calories = 100 := by
  sorry

end bread_calories_eq_100_l4033_403343


namespace square_area_equals_one_l4033_403331

theorem square_area_equals_one (w l : ℝ) (h1 : l = 2 * w) (h2 : w * l = 8 / 9) :
  ∃ s : ℝ, s > 0 ∧ 4 * s = 6 * w ∧ s^2 = 1 := by
  sorry

end square_area_equals_one_l4033_403331


namespace gold_checkpoint_problem_l4033_403313

theorem gold_checkpoint_problem (x : ℝ) : 
  x > 0 →
  x - x * (1/2 + 1/3 * 1/2 + 1/4 * 2/3 + 1/5 * 3/4 + 1/6 * 4/5) = 1 →
  x = 1.2 := by
sorry

end gold_checkpoint_problem_l4033_403313


namespace f_properties_l4033_403378

noncomputable def f (x : ℝ) := 4 * (Real.cos x)^2 + 4 * Real.sqrt 3 * Real.sin x * Real.cos x - 2

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ is_periodic f T ∧ ∀ T' > 0, is_periodic f T' → T ≤ T'

def has_max_at (f : ℝ → ℝ) (M : ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = M ∧ ∀ x, f x ≤ M

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem f_properties :
  (is_smallest_positive_period f Real.pi) ∧
  (∀ k : ℤ, has_max_at f 4 (k * Real.pi + Real.pi / 6)) ∧
  (∀ k : ℤ, is_increasing_on f (-Real.pi / 3 + k * Real.pi) (Real.pi / 6 + k * Real.pi)) := by
  sorry

end f_properties_l4033_403378


namespace min_value_theorem_l4033_403389

theorem min_value_theorem (m n : ℝ) (h1 : m * n > 0) (h2 : -2 * m - n + 2 = 0) :
  2 / m + 1 / n ≥ 9 / 2 :=
by sorry

end min_value_theorem_l4033_403389


namespace max_value_of_sum_and_powers_l4033_403394

theorem max_value_of_sum_and_powers (a b c d : ℝ) :
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 →
  a + b + c + d = 2 →
  ∃ (m : ℝ), m = 2 ∧ ∀ (x y z w : ℝ), 
    x ≥ 0 → y ≥ 0 → z ≥ 0 → w ≥ 0 →
    x + y + z + w = 2 →
    x + y^2 + z^3 + w^4 ≤ m :=
by sorry

end max_value_of_sum_and_powers_l4033_403394


namespace drawing_pie_satisfies_hunger_is_impossible_l4033_403325

/-- An event that involves drawing a pie and satisfying hunger -/
def drawing_pie_to_satisfy_hunger : Set (Nat × Nat) := sorry

/-- Definition of an impossible event -/
def impossible_event (E : Set (Nat × Nat)) : Prop :=
  E = ∅

/-- Theorem: Drawing a pie to satisfy hunger is an impossible event -/
theorem drawing_pie_satisfies_hunger_is_impossible :
  impossible_event drawing_pie_to_satisfy_hunger := by sorry

end drawing_pie_satisfies_hunger_is_impossible_l4033_403325


namespace find_sets_A_and_B_l4033_403357

def I : Set ℕ := {x | x ≤ 8 ∧ x > 0}

theorem find_sets_A_and_B 
  (h1 : A ∪ (I \ B) = {1, 3, 4, 5, 6, 7})
  (h2 : (I \ A) ∪ B = {1, 2, 4, 5, 6, 8})
  (h3 : (I \ A) ∩ (I \ B) = {1, 5, 6}) :
  A = {3, 4, 7} ∧ B = {2, 4, 8} := by
sorry

end find_sets_A_and_B_l4033_403357


namespace max_value_on_circle_l4033_403395

theorem max_value_on_circle (x y : ℝ) (h : x^2 + y^2 = 9) :
  ∃ (M : ℝ), M = 9 ∧ ∀ (a b : ℝ), a^2 + b^2 = 9 → 3 * |a| + 2 * |b| ≤ M :=
by sorry

end max_value_on_circle_l4033_403395


namespace sqrt_sum_equals_abs_sum_l4033_403338

theorem sqrt_sum_equals_abs_sum (x : ℝ) : 
  Real.sqrt (x^2 - 4*x + 4) + Real.sqrt (x^2 + 6*x + 9) = |x - 2| + |x + 3| := by
sorry

end sqrt_sum_equals_abs_sum_l4033_403338


namespace conic_is_parabola_l4033_403328

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  |x - 4| = Real.sqrt ((y + 3)^2 + x^2)

-- Theorem stating that the equation describes a parabola
theorem conic_is_parabola :
  ∃ (a b c d : ℝ), a ≠ 0 ∧
  ∀ (x y : ℝ), conic_equation x y ↔ y = a * x^2 + b * x + c * y + d :=
sorry

end conic_is_parabola_l4033_403328


namespace incircle_radius_of_special_triangle_l4033_403376

-- Define the triangle DEF
structure Triangle :=
  (D E F : ℝ × ℝ)

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  -- The angle at F is 90 degrees (right angle)
  sorry

def angle_E_is_45_deg (t : Triangle) : Prop :=
  -- The angle at E is 45 degrees
  sorry

def side_DF_length (t : Triangle) : ℝ :=
  -- The length of side DF
  8

-- Define the incircle radius
def incircle_radius (t : Triangle) : ℝ :=
  -- The radius of the incircle
  sorry

-- Theorem statement
theorem incircle_radius_of_special_triangle (t : Triangle) 
  (h1 : is_right_triangle t)
  (h2 : angle_E_is_45_deg t)
  (h3 : side_DF_length t = 8) :
  incircle_radius t = 8 - 4 * Real.sqrt 2 :=
sorry

end incircle_radius_of_special_triangle_l4033_403376


namespace min_stamps_proof_l4033_403324

/-- Represents the number of stamps of each denomination -/
structure StampCombination where
  three_cent : ℕ
  four_cent : ℕ
  five_cent : ℕ

/-- Calculates the total value of stamps in cents -/
def total_value (s : StampCombination) : ℕ :=
  3 * s.three_cent + 4 * s.four_cent + 5 * s.five_cent

/-- Calculates the total number of stamps -/
def total_stamps (s : StampCombination) : ℕ :=
  s.three_cent + s.four_cent + s.five_cent

/-- Checks if a stamp combination is valid (totals 50 cents) -/
def is_valid (s : StampCombination) : Prop :=
  total_value s = 50

/-- The minimum number of stamps needed -/
def min_stamps : ℕ := 10

theorem min_stamps_proof :
  (∀ s : StampCombination, is_valid s → total_stamps s ≥ min_stamps) ∧
  (∃ s : StampCombination, is_valid s ∧ total_stamps s = min_stamps) := by
  sorry

#check min_stamps_proof

end min_stamps_proof_l4033_403324


namespace terrys_spending_l4033_403309

/-- Terry's spending problem -/
theorem terrys_spending (monday : ℕ) : 
  monday = 6 →
  let tuesday := 2 * monday
  let wednesday := 2 * (monday + tuesday)
  monday + tuesday + wednesday = 54 := by
  sorry

end terrys_spending_l4033_403309


namespace division_theorem_l4033_403358

theorem division_theorem (A : ℕ) : 14 = 3 * A + 2 → A = 4 := by
  sorry

end division_theorem_l4033_403358


namespace root_difference_zero_l4033_403336

/-- The nonnegative difference between the roots of x^2 + 40x + 300 = -100 is 0 -/
theorem root_difference_zero : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 40*x + 300 + 100
  let roots := {x : ℝ | f x = 0}
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ |r₁ - r₂| = 0 :=
by sorry

end root_difference_zero_l4033_403336


namespace sum_of_squares_is_four_l4033_403383

/-- Represents a rectangle ABCD with an inscribed ellipse K and a point P on K. -/
structure RectangleWithEllipse where
  /-- Length of side AB -/
  ab : ℝ
  /-- Length of side AD -/
  ad : ℝ
  /-- Angle parameter for point P on the ellipse -/
  θ : ℝ
  /-- AB = 2 -/
  h_ab : ab = 2
  /-- AD < √2 -/
  h_ad : ad < Real.sqrt 2

/-- The sum of squares of AM and LB is always 4 -/
theorem sum_of_squares_is_four (rect : RectangleWithEllipse) :
  let x_M := (Real.sqrt 2 * (Real.cos rect.θ - 1)) / (Real.sqrt 2 - Real.sin rect.θ) + 1
  let x_L := (Real.sqrt 2 * (1 + Real.cos rect.θ)) / (Real.sqrt 2 - Real.sin rect.θ) - 1
  (1 + x_M)^2 + (1 - x_L)^2 = 4 := by
  sorry

end sum_of_squares_is_four_l4033_403383


namespace regular_polygon_sides_l4033_403384

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  n > 0 → 
  exterior_angle = 18 → 
  (n : ℝ) * exterior_angle = 360 →
  n = 20 := by
sorry

end regular_polygon_sides_l4033_403384


namespace sean_played_14_days_l4033_403337

/-- The number of days Sean played cricket -/
def sean_days (sean_minutes_per_day : ℕ) (total_minutes : ℕ) (indira_minutes : ℕ) : ℕ :=
  (total_minutes - indira_minutes) / sean_minutes_per_day

/-- Proof that Sean played cricket for 14 days -/
theorem sean_played_14_days :
  sean_days 50 1512 812 = 14 := by
  sorry

end sean_played_14_days_l4033_403337


namespace relative_complement_of_T_in_S_l4033_403322

open Set

def A₁ : Set ℕ := {0, 1}
def A₂ : Set ℕ := {1, 2}
def S : Set ℕ := A₁ ∪ A₂
def T : Set ℕ := A₁ ∩ A₂

theorem relative_complement_of_T_in_S :
  S \ T = {0, 2} := by sorry

end relative_complement_of_T_in_S_l4033_403322


namespace triangle_sine_ratio_l4033_403387

theorem triangle_sine_ratio (A B C : ℝ) (h1 : 0 < A ∧ A < π)
                                       (h2 : 0 < B ∧ B < π)
                                       (h3 : 0 < C ∧ C < π)
                                       (h4 : A + B + C = π)
                                       (h5 : Real.sin A / Real.sin B = 6/5)
                                       (h6 : Real.sin B / Real.sin C = 5/4) :
  Real.sin B = 5 * Real.sqrt 7 / 16 := by
  sorry

end triangle_sine_ratio_l4033_403387


namespace sum_of_fractions_l4033_403365

theorem sum_of_fractions : (3 : ℚ) / 7 + (5 : ℚ) / 14 = (11 : ℚ) / 14 := by
  sorry

end sum_of_fractions_l4033_403365


namespace tangent_line_at_one_l4033_403390

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x

theorem tangent_line_at_one (x : ℝ) :
  let p : ℝ × ℝ := (1, 0)
  let m : ℝ := 2
  (fun x => m * (x - p.1) + p.2) = (fun x => 2 * x - 2) :=
by sorry

end tangent_line_at_one_l4033_403390


namespace boat_speed_l4033_403371

/-- The speed of a boat in still water, given its speeds with and against a stream -/
theorem boat_speed (along_stream : ℝ) (against_stream : ℝ) 
  (h1 : along_stream = 38) 
  (h2 : against_stream = 16) : 
  (along_stream + against_stream) / 2 = 27 := by
  sorry

end boat_speed_l4033_403371


namespace geometric_series_product_l4033_403340

theorem geometric_series_product (y : ℝ) : y = 9 ↔ 
  (∑' n, (1/3)^n) * (∑' n, (-1/3)^n) = ∑' n, (1/y)^n :=
by sorry

end geometric_series_product_l4033_403340


namespace adams_school_schedule_l4033_403317

/-- Represents the number of lessons Adam had on Tuesday -/
def tuesday_lessons : ℕ := 3

theorem adams_school_schedule :
  let monday_hours : ℝ := 3
  let tuesday_hours : ℝ := tuesday_lessons
  let wednesday_hours : ℝ := 2 * tuesday_hours
  let total_hours : ℝ := 12
  monday_hours + tuesday_hours + wednesday_hours = total_hours :=
by sorry


end adams_school_schedule_l4033_403317


namespace congruence_addition_l4033_403332

theorem congruence_addition (a b c d m : ℤ) : 
  a ≡ b [ZMOD m] → c ≡ d [ZMOD m] → (a + c) ≡ (b + d) [ZMOD m] := by
  sorry

end congruence_addition_l4033_403332


namespace min_students_in_both_clubs_l4033_403398

theorem min_students_in_both_clubs
  (total_students : ℕ)
  (club1_students : ℕ)
  (club2_students : ℕ)
  (h1 : total_students = 33)
  (h2 : club1_students ≥ 24)
  (h3 : club2_students ≥ 24) :
  ∃ (intersection : ℕ), intersection ≥ 15 ∧
    intersection ≤ min club1_students club2_students ∧
    intersection ≤ total_students :=
by sorry

end min_students_in_both_clubs_l4033_403398


namespace square_difference_of_quadratic_solutions_l4033_403323

theorem square_difference_of_quadratic_solutions : ∃ α β : ℝ,
  (α ≠ β) ∧ (α^2 = 2*α + 1) ∧ (β^2 = 2*β + 1) ∧ ((α - β)^2 = 8) := by
  sorry

end square_difference_of_quadratic_solutions_l4033_403323


namespace arithmetic_sequence_theorem_l4033_403355

/-- Three numbers forming an arithmetic sequence -/
structure ArithmeticSequence :=
  (a : ℝ)
  (d : ℝ)

/-- The sum of three numbers in an arithmetic sequence -/
def sum (seq : ArithmeticSequence) : ℝ :=
  (seq.a - seq.d) + seq.a + (seq.a + seq.d)

/-- The sum of squares of three numbers in an arithmetic sequence -/
def sumOfSquares (seq : ArithmeticSequence) : ℝ :=
  (seq.a - seq.d)^2 + seq.a^2 + (seq.a + seq.d)^2

/-- Theorem: If three numbers form an arithmetic sequence with a sum of 15 and a sum of squares of 83,
    then these numbers are either 3, 5, 7 or 7, 5, 3 -/
theorem arithmetic_sequence_theorem (seq : ArithmeticSequence) :
  sum seq = 15 ∧ sumOfSquares seq = 83 →
  (seq.a = 5 ∧ (seq.d = 2 ∨ seq.d = -2)) :=
by sorry

end arithmetic_sequence_theorem_l4033_403355


namespace largest_possible_b_value_l4033_403377

theorem largest_possible_b_value (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) →
  (c < b) →
  (b < a) →
  (c = 3) →  -- c is the smallest odd prime number
  (∀ x : ℕ, (1 < x) ∧ (x < b) ∧ (x ≠ c) → (a * x * c ≠ 360)) →
  (b = 8) :=
by sorry

end largest_possible_b_value_l4033_403377


namespace trigonometric_equation_solution_l4033_403347

theorem trigonometric_equation_solution (x : ℝ) : 
  (|Real.sin x| + Real.sin (3 * x)) / (Real.cos x * Real.cos (2 * x)) = 2 / Real.sqrt 3 ↔ 
  (∃ k : ℤ, x = π / 12 + 2 * k * π ∨ x = 7 * π / 12 + 2 * k * π ∨ x = -5 * π / 6 + 2 * k * π) :=
by sorry

end trigonometric_equation_solution_l4033_403347


namespace square_sum_identity_l4033_403352

theorem square_sum_identity (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end square_sum_identity_l4033_403352


namespace compound_molecular_weight_l4033_403399

/-- Atomic weight of Potassium in g/mol -/
def atomic_weight_K : ℝ := 39.10

/-- Atomic weight of Bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- Atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- Number of Potassium atoms in the compound -/
def num_K : ℕ := 1

/-- Number of Bromine atoms in the compound -/
def num_Br : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def num_O : ℕ := 3

/-- The molecular weight of the compound -/
def molecular_weight : ℝ := num_K * atomic_weight_K + num_Br * atomic_weight_Br + num_O * atomic_weight_O

theorem compound_molecular_weight : molecular_weight = 167.00 := by
  sorry

end compound_molecular_weight_l4033_403399


namespace balloon_height_proof_l4033_403391

/-- Calculates the maximum height a helium balloon can fly given budget and costs. -/
def balloon_max_height (total_budget : ℚ) (sheet_cost rope_cost propane_cost : ℚ) 
  (helium_cost_per_oz : ℚ) (height_per_oz : ℚ) : ℚ :=
  let remaining_budget := total_budget - (sheet_cost + rope_cost + propane_cost)
  let helium_oz := remaining_budget / helium_cost_per_oz
  helium_oz * height_per_oz

/-- The maximum height of the balloon is 9,492 feet given the specified conditions. -/
theorem balloon_height_proof : 
  balloon_max_height 200 42 18 14 (3/2) 113 = 9492 := by
  sorry

end balloon_height_proof_l4033_403391


namespace derivative_of_f_l4033_403375

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.cos x

theorem derivative_of_f (x : ℝ) :
  HasDerivAt f (2*x*(Real.cos x) - x^2*(Real.sin x)) x := by
  sorry

end derivative_of_f_l4033_403375


namespace total_students_l4033_403382

theorem total_students (middle_school : ℕ) (elementary_school : ℕ) (high_school : ℕ) : 
  middle_school = 50 → 
  elementary_school = 4 * middle_school - 3 → 
  high_school = 2 * elementary_school → 
  elementary_school + middle_school + high_school = 641 := by
  sorry

end total_students_l4033_403382


namespace arithmetic_square_root_of_16_l4033_403349

theorem arithmetic_square_root_of_16 : ∃ x : ℝ, x ≥ 0 ∧ x^2 = 16 ∧ x = 4 := by
  sorry

end arithmetic_square_root_of_16_l4033_403349


namespace chord_intersection_theorem_l4033_403319

theorem chord_intersection_theorem (r : ℝ) (PT OT : ℝ) : 
  r = 7 → OT = 3 → PT = 8 → 
  ∃ (RS : ℝ), RS = 16 ∧ 
  ∃ (x : ℝ), x * (RS - x) = PT * PT ∧ 
  ∃ (n : ℕ), x * (RS - x) = n^2 :=
by sorry

end chord_intersection_theorem_l4033_403319


namespace initial_persons_count_l4033_403300

/-- The number of persons initially in the group. -/
def n : ℕ := sorry

/-- The average weight increase when a new person replaces one person. -/
def avg_weight_increase : ℚ := 5/2

/-- The weight of the replaced person. -/
def old_weight : ℕ := 65

/-- The weight of the new person. -/
def new_weight : ℕ := 85

/-- Theorem stating that the initial number of persons is 8. -/
theorem initial_persons_count : n = 8 := by
  sorry

end initial_persons_count_l4033_403300


namespace tyler_meal_choices_l4033_403301

def meat_options : ℕ := 4
def vegetable_options : ℕ := 5
def dessert_options : ℕ := 3

def meat_choices : ℕ := 2
def vegetable_choices : ℕ := 3
def dessert_choices : ℕ := 1

theorem tyler_meal_choices :
  (Nat.choose meat_options meat_choices) *
  (Nat.choose vegetable_options vegetable_choices) *
  (Nat.choose dessert_options dessert_choices) = 180 := by
  sorry

end tyler_meal_choices_l4033_403301


namespace luke_mowing_money_l4033_403393

/-- The amount of money Luke made mowing lawns -/
def mowing_money : ℝ := sorry

/-- The amount of money Luke made weed eating -/
def weed_eating_money : ℝ := 18

/-- The amount Luke spends per week -/
def weekly_spending : ℝ := 3

/-- The number of weeks the money lasts -/
def weeks_lasted : ℝ := 9

/-- Theorem stating that Luke made $9 mowing lawns -/
theorem luke_mowing_money : mowing_money = 9 := by
  sorry

end luke_mowing_money_l4033_403393


namespace polygon_diagonals_with_disconnected_vertex_l4033_403344

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals connected to a single vertex in a polygon with n sides -/
def diagonals_per_vertex (n : ℕ) : ℕ := n - 3

theorem polygon_diagonals_with_disconnected_vertex :
  diagonals 17 - diagonals_per_vertex 17 = 105 := by
  sorry

end polygon_diagonals_with_disconnected_vertex_l4033_403344


namespace cindy_envelopes_l4033_403356

theorem cindy_envelopes (friends : ℕ) (envelopes_per_friend : ℕ) (envelopes_left : ℕ) :
  friends = 5 →
  envelopes_per_friend = 3 →
  envelopes_left = 22 →
  friends * envelopes_per_friend + envelopes_left = 37 :=
by sorry

end cindy_envelopes_l4033_403356


namespace charity_event_arrangements_l4033_403363

theorem charity_event_arrangements (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 5 → k = 4 → m = 3 →
  (Nat.choose n 2) * (Nat.choose (n - 2) 1) * (Nat.choose (n - 3) 1) = 60 :=
by
  sorry

end charity_event_arrangements_l4033_403363


namespace log_216_simplification_l4033_403310

theorem log_216_simplification :
  (216 : ℝ) = 6^3 →
  Real.log 216 = 3 * (Real.log 2 + Real.log 3) := by
sorry

end log_216_simplification_l4033_403310


namespace quadratic_inequality_solution_range_l4033_403341

/-- A quadratic inequality with parameter m has exactly 3 integer solutions -/
def has_three_integer_solutions (m : ℝ) : Prop :=
  ∃! (a b c : ℤ), (m : ℝ) * (a : ℝ)^2 + (2 - m) * (a : ℝ) - 2 > 0 ∧
                   (m : ℝ) * (b : ℝ)^2 + (2 - m) * (b : ℝ) - 2 > 0 ∧
                   (m : ℝ) * (c : ℝ)^2 + (2 - m) * (c : ℝ) - 2 > 0 ∧
                   ∀ (x : ℤ), (m : ℝ) * (x : ℝ)^2 + (2 - m) * (x : ℝ) - 2 > 0 → (x = a ∨ x = b ∨ x = c)

/-- The main theorem -/
theorem quadratic_inequality_solution_range (m : ℝ) :
  has_three_integer_solutions m → -1/2 < m ∧ m ≤ -2/5 :=
sorry

end quadratic_inequality_solution_range_l4033_403341


namespace special_rectangle_ratio_l4033_403388

/-- A rectangle with the property that the square of the ratio of its short side to its long side
    is equal to the ratio of its long side to its diagonal. -/
structure SpecialRectangle where
  short : ℝ
  long : ℝ
  diagonal : ℝ
  short_positive : 0 < short
  long_positive : 0 < long
  diagonal_positive : 0 < diagonal
  pythagorean : diagonal^2 = short^2 + long^2
  special_property : (short / long)^2 = long / diagonal

/-- The ratio of the short side to the long side in a SpecialRectangle is (√5 - 1) / 3. -/
theorem special_rectangle_ratio (r : SpecialRectangle) : 
  r.short / r.long = (Real.sqrt 5 - 1) / 3 := by
  sorry

end special_rectangle_ratio_l4033_403388


namespace nomogram_relationships_l4033_403304

/-- A structure representing the scales in the nomogram -/
structure Scales where
  X : ℝ
  Y : ℝ
  Z : ℝ
  W : ℝ
  V : ℝ
  U : ℝ
  T : ℝ
  S : ℝ

/-- The theorem stating the relationships between the scales -/
theorem nomogram_relationships (scales : Scales) :
  scales.Z = (scales.X + scales.Y) / 2 ∧
  scales.W = scales.X + scales.Y ∧
  scales.Y = scales.W - scales.X ∧
  scales.V = 2 * (scales.X + scales.Z) ∧
  scales.X + scales.Z + 5 * scales.U = 0 ∧
  scales.T = (6 + scales.Y + scales.Z) / 2 ∧
  scales.Y + scales.Z + 4 * scales.S - 10 = 0 := by
  sorry

end nomogram_relationships_l4033_403304


namespace profit_is_120_l4033_403306

/-- Calculates the profit from book sales given the selling price, number of customers,
    production cost, and books per customer. -/
def calculate_profit (selling_price : ℕ) (num_customers : ℕ) (production_cost : ℕ) (books_per_customer : ℕ) : ℕ :=
  let total_books := num_customers * books_per_customer
  let revenue := selling_price * total_books
  let total_cost := production_cost * total_books
  revenue - total_cost

/-- Proves that the profit is $120 given the specified conditions. -/
theorem profit_is_120 :
  let selling_price := 20
  let num_customers := 4
  let production_cost := 5
  let books_per_customer := 2
  calculate_profit selling_price num_customers production_cost books_per_customer = 120 := by
  sorry

end profit_is_120_l4033_403306


namespace sufficient_not_necessary_contrapositive_equivalence_negation_equivalence_l4033_403396

-- Statement ②
theorem sufficient_not_necessary :
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ∧
  (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1) :=
sorry

-- Statement ③
theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) ↔
  (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) :=
sorry

-- Statement ④
theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔
  (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
sorry

end sufficient_not_necessary_contrapositive_equivalence_negation_equivalence_l4033_403396


namespace spelling_bee_contestants_l4033_403333

theorem spelling_bee_contestants (total : ℕ) 
  (h1 : (total : ℝ) * (1 - 0.6) * 0.25 = 30) : total = 300 := by
  sorry

end spelling_bee_contestants_l4033_403333


namespace shaded_area_theorem_l4033_403348

/-- The area of a shape composed of a right triangle and 12 congruent squares -/
theorem shaded_area_theorem (hypotenuse : ℝ) (num_squares : ℕ) :
  hypotenuse = 10 →
  num_squares = 12 →
  let leg := hypotenuse / Real.sqrt 2
  let triangle_area := leg * leg / 2
  let square_side := leg / 3
  let square_area := square_side * square_side
  let total_squares_area := num_squares * square_area
  triangle_area + total_squares_area = 275 / 3 := by
  sorry

end shaded_area_theorem_l4033_403348


namespace train_length_calculation_l4033_403386

/-- Calculates the length of a train given its speed, the time it takes to pass a bridge, and the length of the bridge. -/
theorem train_length_calculation (train_speed : ℝ) (bridge_length : ℝ) (passing_time : ℝ) : 
  train_speed = 45 * (1000 / 3600) →
  bridge_length = 140 →
  passing_time = 50 →
  (train_speed * passing_time) - bridge_length = 485 := by
  sorry

#check train_length_calculation

end train_length_calculation_l4033_403386


namespace functional_equation_solution_l4033_403339

/-- A function satisfying the given functional equation. -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y + f (x + y) = x * y

/-- The theorem stating that functions satisfying the equation are of the form x - 1 or x + 1. -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : SatisfiesEquation f) :
    (∀ x, f x = x - 1) ∨ (∀ x, f x = x + 1) := by
  sorry


end functional_equation_solution_l4033_403339


namespace problem_solution_l4033_403385

theorem problem_solution (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_prod : x*y + x*z + y*z ≠ 0) :
  (x^6 + y^6 + z^6) / (x*y*z * (x*y + x*z + y*z)) = -10 := by
  sorry

end problem_solution_l4033_403385


namespace qin_jiushao_algorithm_l4033_403312

theorem qin_jiushao_algorithm (n : ℕ) (x : ℝ) (h1 : n = 5) (h2 : x = 2) :
  (Finset.range (n + 1)).sum (fun i => x ^ i) = 63 := by
  sorry

end qin_jiushao_algorithm_l4033_403312


namespace circus_illumination_theorem_l4033_403320

/-- A convex figure in a plane -/
structure ConvexFigure where
  -- Define properties of a convex figure

/-- The plane -/
structure Plane where
  -- Define properties of a plane

/-- Represents the illumination of the arena -/
def Illumination (n : ℕ) := Fin n → ConvexFigure

/-- The union of a subset of convex figures -/
def UnionOfFigures (i : Illumination n) (s : Finset (Fin n)) : Set Plane :=
  sorry

/-- The entire plane is covered -/
def CoversPlaane (s : Set Plane) : Prop :=
  sorry

/-- Main theorem: For any n ≥ 2, there exists an illumination arrangement satisfying the conditions -/
theorem circus_illumination_theorem (n : ℕ) (h : n ≥ 2) :
  ∃ (i : Illumination n),
    (∀ (k : Fin n), CoversPlaane (UnionOfFigures i (Finset.erase (Finset.univ : Finset (Fin n)) k))) ∧
    (∀ (j k : Fin n), j ≠ k → ¬CoversPlaane (UnionOfFigures i (Finset.erase (Finset.erase (Finset.univ : Finset (Fin n)) j) k))) :=
  sorry

end circus_illumination_theorem_l4033_403320


namespace min_value_of_reciprocal_sum_min_value_achieved_l4033_403397

theorem min_value_of_reciprocal_sum (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) : 
  4/x + 1/y ≥ 9 := by
  sorry

theorem min_value_achieved (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ 4/x₀ + 1/y₀ = 9 := by
  sorry

end min_value_of_reciprocal_sum_min_value_achieved_l4033_403397


namespace arithmetic_mean_of_fractions_l4033_403353

theorem arithmetic_mean_of_fractions : 
  let a := 8 / 11
  let b := 5 / 6
  let c := 19 / 22
  b = (a + c) / 2 := by sorry

end arithmetic_mean_of_fractions_l4033_403353


namespace average_problem_l4033_403342

theorem average_problem (x y : ℝ) :
  (7 + 9 + x + y + 17) / 5 = 10 →
  ((x + 3) + (x + 5) + (y + 2) + 8 + (y + 18)) / 5 = 14 :=
by sorry

end average_problem_l4033_403342


namespace arrangements_with_restriction_l4033_403368

theorem arrangements_with_restriction (n : ℕ) (h : n = 6) :
  (n - 1) * Nat.factorial (n - 1) = 600 :=
by sorry

end arrangements_with_restriction_l4033_403368


namespace students_taking_one_subject_l4033_403335

theorem students_taking_one_subject (total_students : ℕ) 
  (algebra_and_drafting : ℕ) (algebra_total : ℕ) (only_drafting : ℕ) 
  (neither_subject : ℕ) :
  algebra_and_drafting = 22 →
  algebra_total = 40 →
  only_drafting = 15 →
  neither_subject = 8 →
  total_students = algebra_total + only_drafting + neither_subject →
  (algebra_total - algebra_and_drafting) + only_drafting = 33 :=
by sorry

end students_taking_one_subject_l4033_403335


namespace ship_cats_count_l4033_403305

/-- Represents the passengers on the ship --/
structure ShipPassengers where
  cats : ℕ
  sailors : ℕ
  cook : ℕ
  captain : ℕ

/-- Calculates the total number of heads on the ship --/
def totalHeads (p : ShipPassengers) : ℕ :=
  p.cats + p.sailors + p.cook + p.captain

/-- Calculates the total number of legs on the ship --/
def totalLegs (p : ShipPassengers) : ℕ :=
  4 * p.cats + 2 * p.sailors + 2 * p.cook + p.captain

/-- Theorem stating that given the conditions, the number of cats is 7 --/
theorem ship_cats_count (p : ShipPassengers) 
  (h1 : p.cook = 1) 
  (h2 : p.captain = 1) 
  (h3 : totalHeads p = 16) 
  (h4 : totalLegs p = 45) : 
  p.cats = 7 := by
  sorry


end ship_cats_count_l4033_403305


namespace circle_center_correct_l4033_403373

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 6*y = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, -3)

/-- Theorem: The center of the circle defined by circle_equation is circle_center -/
theorem circle_center_correct :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 13 :=
sorry

end circle_center_correct_l4033_403373


namespace triangle_lines_theorem_l4033_403360

/-- Triangle ABC with vertices A(-3,5), B(5,7), and C(5,1) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The given triangle -/
def ABC : Triangle := { A := (-3, 5), B := (5, 7), C := (5, 1) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The median line on side AB -/
def medianAB : LineEquation := { a := 5, b := 4, c := -29 }

/-- The line through A with equal x-axis and y-axis intercepts -/
def lineA : LineEquation := { a := 1, b := 1, c := -2 }

theorem triangle_lines_theorem (t : Triangle) (m : LineEquation) (l : LineEquation) : 
  t = ABC → m = medianAB → l = lineA → True := by sorry

end triangle_lines_theorem_l4033_403360


namespace bug_eating_ratio_l4033_403359

theorem bug_eating_ratio (gecko lizard frog toad : ℕ) : 
  gecko = 12 →
  lizard = gecko / 2 →
  toad = frog + frog / 2 →
  gecko + lizard + frog + toad = 63 →
  frog / lizard = 3 := by
  sorry

end bug_eating_ratio_l4033_403359


namespace workmen_efficiency_ratio_l4033_403334

/-- Given two workmen with different efficiencies, prove their efficiency ratio -/
theorem workmen_efficiency_ratio 
  (combined_time : ℝ) 
  (b_alone_time : ℝ) 
  (ha : combined_time = 18) 
  (hb : b_alone_time = 54) : 
  (1 / combined_time - 1 / b_alone_time) / (1 / b_alone_time) = 2 := by
  sorry

end workmen_efficiency_ratio_l4033_403334


namespace gmat_exam_correct_answers_l4033_403369

theorem gmat_exam_correct_answers 
  (total : ℕ) 
  (first_correct : ℕ) 
  (second_correct : ℕ) 
  (neither_correct : ℕ) 
  (h1 : first_correct = (85 * total) / 100)
  (h2 : second_correct = (80 * total) / 100)
  (h3 : neither_correct = (5 * total) / 100)
  : ((first_correct + second_correct - (total - neither_correct)) * 100) / total = 70 :=
by sorry

end gmat_exam_correct_answers_l4033_403369


namespace age_difference_of_children_l4033_403329

/-- Proves that the age difference between two children is 2 years given the family conditions --/
theorem age_difference_of_children (initial_members : ℕ) (initial_avg_age : ℕ) 
  (years_passed : ℕ) (current_members : ℕ) (current_avg_age : ℕ) (youngest_child_age : ℕ) :
  initial_members = 4 →
  initial_avg_age = 24 →
  years_passed = 10 →
  current_members = 6 →
  current_avg_age = 24 →
  youngest_child_age = 3 →
  ∃ (older_child_age : ℕ), 
    older_child_age - youngest_child_age = 2 ∧
    older_child_age + youngest_child_age = 
      current_members * current_avg_age - initial_members * (initial_avg_age + years_passed) :=
by
  sorry

#check age_difference_of_children

end age_difference_of_children_l4033_403329


namespace min_value_and_max_t_l4033_403346

-- Define the function f
def f (a b x : ℝ) : ℝ := |x + a| + |2*x - b|

-- State the theorem
theorem min_value_and_max_t (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (m : ℝ), m = 1 ∧ ∀ x, f a b x ≥ m) →
  (2*a + b = 2) ∧
  (∀ t, (a + 2*b ≥ t*a*b) → t ≤ 9/2) ∧
  (∃ t₀, t₀ = 9/2 ∧ a + 2*b ≥ t₀*a*b) :=
by sorry

end min_value_and_max_t_l4033_403346


namespace field_area_is_500_l4033_403380

/-- Represents the area of a field divided into two parts -/
structure FieldArea where
  small : ℝ
  large : ℝ

/-- Calculates the total area of the field -/
def total_area (f : FieldArea) : ℝ := f.small + f.large

/-- Theorem: The total area of the field is 500 hectares -/
theorem field_area_is_500 (f : FieldArea) 
  (h1 : f.small = 225)
  (h2 : f.large - f.small = (1/5) * ((f.small + f.large) / 2)) :
  total_area f = 500 := by
  sorry

end field_area_is_500_l4033_403380


namespace two_dice_prime_probability_l4033_403327

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def dice_outcomes (n : ℕ) : ℕ := 6^n

def prime_outcomes (n : ℕ) : ℕ := 
  if n = 2 then 15 else 0  -- We only define it for 2 dice as per the problem

theorem two_dice_prime_probability :
  (prime_outcomes 2 : ℚ) / (dice_outcomes 2 : ℚ) = 5/12 :=
sorry

end two_dice_prime_probability_l4033_403327


namespace solution_product_l4033_403361

theorem solution_product (a b : ℝ) : 
  (3 * a^2 + 4 * a - 7 = 0) → 
  (3 * b^2 + 4 * b - 7 = 0) → 
  (a - 2) * (b - 2) = 0 := by
sorry

end solution_product_l4033_403361


namespace equation_solutions_l4033_403392

theorem equation_solutions :
  (∀ x : ℝ, (x + 1)^2 - 16 = 0 ↔ x = 3 ∨ x = -5) ∧
  (∀ x : ℝ, -2 * (x - 1)^3 = 16 ↔ x = -1) := by
  sorry

end equation_solutions_l4033_403392


namespace machine_productivity_problem_l4033_403351

theorem machine_productivity_problem 
  (productivity_second : ℝ) 
  (productivity_first : ℝ := 1.4 * productivity_second) 
  (hours_first : ℝ := 6) 
  (hours_second : ℝ := 8) 
  (total_parts : ℕ := 820) :
  productivity_first * hours_first + productivity_second * hours_second = total_parts → 
  (productivity_first * hours_first = 420 ∧ productivity_second * hours_second = 400) :=
by
  sorry

end machine_productivity_problem_l4033_403351


namespace percentage_of_sikh_boys_l4033_403315

theorem percentage_of_sikh_boys (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (other : ℕ) :
  total = 400 →
  muslim_percent = 44 / 100 →
  hindu_percent = 28 / 100 →
  other = 72 →
  (total - (muslim_percent * total).num - (hindu_percent * total).num - other) / total * 100 = 10 := by
  sorry

end percentage_of_sikh_boys_l4033_403315


namespace graph_translation_symmetry_l4033_403302

theorem graph_translation_symmetry (m : Real) : m > 0 →
  (∀ x, 2 * Real.sin (x + m - π / 3) = 2 * Real.sin (-x + m - π / 3)) →
  m = 5 * π / 6 := by
  sorry

end graph_translation_symmetry_l4033_403302


namespace intersection_M_N_l4033_403350

def M : Set ℝ := {x | ∃ t : ℝ, x = 2^(-t)}
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}

theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by sorry

end intersection_M_N_l4033_403350


namespace fly_ceiling_distance_l4033_403366

def fly_distance (x y z : ℝ) : Prop :=
  x = 2 ∧ y = 7 ∧ x^2 + y^2 + z^2 = 10^2

theorem fly_ceiling_distance :
  ∀ x y z : ℝ, fly_distance x y z → z = Real.sqrt 47 :=
by
  sorry

end fly_ceiling_distance_l4033_403366


namespace polynomial_remainder_theorem_l4033_403326

def P (x : ℝ) : ℝ := 5*x^6 - 3*x^5 + 4*x^4 - x^3 + 6*x^2 - 5*x + 7

theorem polynomial_remainder_theorem :
  ∃ (Q : ℝ → ℝ), P = λ x ↦ (x - 3) * Q x + 3259 :=
sorry

end polynomial_remainder_theorem_l4033_403326


namespace johns_watermelon_weight_l4033_403372

theorem johns_watermelon_weight (michael_weight : ℕ) (clay_factor : ℕ) (john_factor : ℚ) :
  michael_weight = 8 →
  clay_factor = 3 →
  john_factor = 1/2 →
  (↑michael_weight * ↑clay_factor * john_factor : ℚ) = 12 := by
  sorry

end johns_watermelon_weight_l4033_403372


namespace father_son_age_sum_father_son_age_sum_proof_l4033_403364

/-- The sum of the present ages of a father and son, given specific age relationships -/
theorem father_son_age_sum : ℕ → ℕ → Prop :=
  fun son_age father_age =>
    (father_age - 18 = 3 * (son_age - 18)) ∧  -- 18 years ago relationship
    (father_age = 2 * son_age) →              -- current relationship
    son_age + father_age = 108                -- sum of present ages

/-- Proof of the father_son_age_sum theorem -/
theorem father_son_age_sum_proof : ∃ (son_age father_age : ℕ), father_son_age_sum son_age father_age := by
  sorry

#check father_son_age_sum
#check father_son_age_sum_proof

end father_son_age_sum_father_son_age_sum_proof_l4033_403364


namespace translated_function_coefficient_sum_l4033_403308

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

def translation (h : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + h)

theorem translated_function_coefficient_sum :
  ∃ (a b c : ℝ),
    (∀ x, translation 3 f x = a * x^2 + b * x + c) ∧
    a + b + c = 51 :=
by sorry

end translated_function_coefficient_sum_l4033_403308


namespace water_bottles_sold_l4033_403318

/-- The number of water bottles sold in a store, given the prices and quantities of other drinks --/
theorem water_bottles_sold : ℕ := by
  -- Define the prices of drinks
  let cola_price : ℚ := 3
  let juice_price : ℚ := 3/2
  let water_price : ℚ := 1

  -- Define the quantities of cola and juice sold
  let cola_quantity : ℕ := 15
  let juice_quantity : ℕ := 12

  -- Define the total earnings
  let total_earnings : ℚ := 88

  -- Define the function to calculate the number of water bottles
  let water_bottles (x : ℕ) : Prop :=
    cola_price * cola_quantity + juice_price * juice_quantity + water_price * x = total_earnings

  -- Prove that the number of water bottles sold is 25
  have h : water_bottles 25 := by sorry

  exact 25

end water_bottles_sold_l4033_403318


namespace ratio_first_to_last_l4033_403379

/-- An arithmetic sequence with five terms -/
structure ArithmeticSequence :=
  (a x c d b : ℚ)
  (is_arithmetic : ∃ (diff : ℚ), x = a + diff ∧ c = x + diff ∧ d = c + diff ∧ b = d + diff)
  (fourth_term : d = 3 * x)
  (fifth_term : b = 4 * x)

/-- The ratio of the first term to the last term is -1/4 -/
theorem ratio_first_to_last (seq : ArithmeticSequence) : seq.a / seq.b = -1/4 := by
  sorry

end ratio_first_to_last_l4033_403379


namespace least_integer_absolute_value_l4033_403345

theorem least_integer_absolute_value (x : ℤ) : 
  (∀ y : ℤ, y < x → ¬(|2*y + 7| ≤ 16)) ∧ (|2*x + 7| ≤ 16) → x = -11 :=
by sorry

end least_integer_absolute_value_l4033_403345


namespace sin_330_degrees_l4033_403381

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1 / 2 := by
  sorry

end sin_330_degrees_l4033_403381
