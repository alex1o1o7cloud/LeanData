import Mathlib

namespace NUMINAMATH_GPT_angle_between_AB_CD_l1709_170925

def point := (ℝ × ℝ × ℝ)

def A : point := (-3, 0, 1)
def B : point := (2, 1, -1)
def C : point := (-2, 2, 0)
def D : point := (1, 3, 2)

noncomputable def angle_between_lines (p1 p2 p3 p4 : point) : ℝ := sorry

theorem angle_between_AB_CD :
  angle_between_lines A B C D = Real.arccos (2 * Real.sqrt 105 / 35) :=
sorry

end NUMINAMATH_GPT_angle_between_AB_CD_l1709_170925


namespace NUMINAMATH_GPT_find_f_of_3_l1709_170989

theorem find_f_of_3 (a b c : ℝ) (f : ℝ → ℝ) (h1 : f 1 = 7) (h2 : f 2 = 12) (h3 : ∀ x, f x = ax + bx + c) : f 3 = 17 :=
by
  sorry

end NUMINAMATH_GPT_find_f_of_3_l1709_170989


namespace NUMINAMATH_GPT_pete_miles_walked_l1709_170979

-- Define the conditions
def maxSteps := 99999
def numFlips := 50
def finalReading := 25000
def stepsPerMile := 1500

-- Proof statement that Pete walked 3350 miles
theorem pete_miles_walked : 
  (numFlips * (maxSteps + 1) + finalReading) / stepsPerMile = 3350 := 
by 
  sorry

end NUMINAMATH_GPT_pete_miles_walked_l1709_170979


namespace NUMINAMATH_GPT_find_positive_integer_l1709_170924

theorem find_positive_integer (n : ℕ) (hn_pos : n > 0) :
  (∃ a b : ℕ, n = a^2 ∧ n + 100 = b^2) → n = 576 :=
by sorry

end NUMINAMATH_GPT_find_positive_integer_l1709_170924


namespace NUMINAMATH_GPT_eval_expression_l1709_170955

theorem eval_expression (x y z : ℝ) (hx : x = 1/3) (hy : y = 2/3) (hz : z = -9) :
  x^2 * y^3 * z = -8/27 :=
by
  subst hx
  subst hy
  subst hz
  sorry

end NUMINAMATH_GPT_eval_expression_l1709_170955


namespace NUMINAMATH_GPT_find_A_l1709_170998

def U : Set ℕ := {1, 2, 3, 4, 5}

def compl_U (A : Set ℕ) : Set ℕ := U \ A

theorem find_A (A : Set ℕ) (hU : U = {1, 2, 3, 4, 5})
  (h_compl_U : compl_U A = {2, 3}) : A = {1, 4, 5} :=
by
  sorry

end NUMINAMATH_GPT_find_A_l1709_170998


namespace NUMINAMATH_GPT_min_n_such_that_no_more_possible_l1709_170909

-- Define a seven-cell corner as a specific structure within the grid
inductive Corner
| cell7 : Corner

-- Function to count the number of cells clipped out by n corners
def clipped_cells (n : ℕ) : ℕ := 7 * n

-- Statement to be proven
theorem min_n_such_that_no_more_possible (n : ℕ) (h_n : n ≥ 3) (h_max : n < 4) :
  ¬ ∃ k : ℕ, k > n ∧ clipped_cells k ≤ 64 :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_min_n_such_that_no_more_possible_l1709_170909


namespace NUMINAMATH_GPT_find_f_21_l1709_170935

def f : ℝ → ℝ := sorry

lemma f_condition (x : ℝ) : f (2 / x + 1) = Real.log x := sorry

theorem find_f_21 : f 21 = -1 := sorry

end NUMINAMATH_GPT_find_f_21_l1709_170935


namespace NUMINAMATH_GPT_num_positive_terms_arithmetic_seq_l1709_170978

theorem num_positive_terms_arithmetic_seq :
  (∃ k : ℕ+, (∀ n : ℕ, n ≤ k → (90 - 2 * n) > 0)) → (k = 44) :=
sorry

end NUMINAMATH_GPT_num_positive_terms_arithmetic_seq_l1709_170978


namespace NUMINAMATH_GPT_exists_X_Y_l1709_170990

theorem exists_X_Y {A n : ℤ} (h_coprime : Int.gcd A n = 1) :
  ∃ X Y : ℤ, |X| < Int.sqrt n ∧ |Y| < Int.sqrt n ∧ n ∣ (A * X - Y) :=
sorry

end NUMINAMATH_GPT_exists_X_Y_l1709_170990


namespace NUMINAMATH_GPT_num_integers_n_with_properties_l1709_170996

theorem num_integers_n_with_properties :
  ∃ (N : Finset ℕ), N.card = 50 ∧
  ∀ n ∈ N, n < 150 ∧
    ∃ (m : ℕ), (∃ k, n = 2*k + 1 ∧ m = k*(k+1)) ∧ ¬ (3 ∣ m) :=
sorry

end NUMINAMATH_GPT_num_integers_n_with_properties_l1709_170996


namespace NUMINAMATH_GPT_gardener_total_expenses_l1709_170911

theorem gardener_total_expenses
  (tulips carnations roses : ℕ)
  (cost_per_flower : ℕ)
  (h1 : tulips = 250)
  (h2 : carnations = 375)
  (h3 : roses = 320)
  (h4 : cost_per_flower = 2) :
  (tulips + carnations + roses) * cost_per_flower = 1890 := 
by
  sorry

end NUMINAMATH_GPT_gardener_total_expenses_l1709_170911


namespace NUMINAMATH_GPT_A_det_nonzero_A_inv_is_correct_l1709_170920

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 4], ![2, 9]]

def A_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![9, -4], ![-2, 1]]

theorem A_det_nonzero : det A ≠ 0 := 
  sorry

theorem A_inv_is_correct : A * A_inv = 1 := 
  sorry

end NUMINAMATH_GPT_A_det_nonzero_A_inv_is_correct_l1709_170920


namespace NUMINAMATH_GPT_dart_lands_in_center_hexagon_l1709_170985

noncomputable def area_regular_hexagon (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

theorem dart_lands_in_center_hexagon {s : ℝ} (h : s > 0) :
  let A_outer := area_regular_hexagon s
  let A_inner := area_regular_hexagon (s / 2)
  (A_inner / A_outer) = 1 / 4 :=
by
  let A_outer := area_regular_hexagon s
  let A_inner := area_regular_hexagon (s / 2)
  sorry

end NUMINAMATH_GPT_dart_lands_in_center_hexagon_l1709_170985


namespace NUMINAMATH_GPT_smaller_circle_radius_l1709_170949

theorem smaller_circle_radius (r R : ℝ) (A1 A2 : ℝ) (hR : R = 5.0) (hA : A1 + A2 = 25 * Real.pi)
  (hap : A2 = A1 + 25 * Real.pi / 2) : r = 5 * Real.sqrt 2 / 2 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_smaller_circle_radius_l1709_170949


namespace NUMINAMATH_GPT_binom_18_4_eq_3060_l1709_170936

theorem binom_18_4_eq_3060 : Nat.choose 18 4 = 3060 := by
  sorry

end NUMINAMATH_GPT_binom_18_4_eq_3060_l1709_170936


namespace NUMINAMATH_GPT_sum_of_coefficients_l1709_170967

theorem sum_of_coefficients (A B C : ℤ)
  (h : ∀ x, x^3 + A * x^2 + B * x + C = (x + 3) * x * (x - 3))
  : A + B + C = -9 :=
sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1709_170967


namespace NUMINAMATH_GPT_product_relationship_l1709_170910

variable {a_1 a_2 b_1 b_2 : ℝ}

theorem product_relationship (h1 : a_1 < a_2) (h2 : b_1 < b_2) : 
  a_1 * b_1 + a_2 * b_2 > a_1 * b_2 + a_2 * b_1 := 
sorry

end NUMINAMATH_GPT_product_relationship_l1709_170910


namespace NUMINAMATH_GPT_abs_a_eq_5_and_a_add_b_eq_0_l1709_170976

theorem abs_a_eq_5_and_a_add_b_eq_0 (a b : ℤ) (h1 : |a| = 5) (h2 : a + b = 0) :
  a - b = 10 ∨ a - b = -10 :=
by
  sorry

end NUMINAMATH_GPT_abs_a_eq_5_and_a_add_b_eq_0_l1709_170976


namespace NUMINAMATH_GPT_number_of_ferns_is_six_l1709_170942

def num_fronds_per_fern : Nat := 7
def num_leaves_per_frond : Nat := 30
def total_leaves : Nat := 1260

theorem number_of_ferns_is_six :
  total_leaves = num_fronds_per_fern * num_leaves_per_frond * 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ferns_is_six_l1709_170942


namespace NUMINAMATH_GPT_geometric_sequence_sixth_term_l1709_170958

theorem geometric_sequence_sixth_term:
  ∃ q : ℝ, 
  ∀ (a₁ a₈ a₆ : ℝ), 
    a₁ = 6 ∧ a₈ = 768 ∧ a₈ = a₁ * q^7 ∧ a₆ = a₁ * q^5 
    → a₆ = 192 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sixth_term_l1709_170958


namespace NUMINAMATH_GPT_age_difference_between_brother_and_cousin_is_five_l1709_170992

variable (Lexie_age brother_age sister_age uncle_age grandma_age cousin_age : ℕ)

-- Conditions
axiom lexie_age_def : Lexie_age = 8
axiom grandma_age_def : grandma_age = 68
axiom lexie_brother_condition : Lexie_age = brother_age + 6
axiom lexie_sister_condition : sister_age = 2 * Lexie_age
axiom uncle_grandma_condition : uncle_age = grandma_age - 12
axiom cousin_brother_condition : cousin_age = brother_age + 5

-- Goal
theorem age_difference_between_brother_and_cousin_is_five : 
  Lexie_age = 8 → grandma_age = 68 → brother_age = Lexie_age - 6 → cousin_age = brother_age + 5 → cousin_age - brother_age = 5 :=
by sorry

end NUMINAMATH_GPT_age_difference_between_brother_and_cousin_is_five_l1709_170992


namespace NUMINAMATH_GPT_children_in_school_l1709_170969

theorem children_in_school (C B : ℕ) (h1 : B = 2 * C) (h2 : B = 4 * (C - 370)) : C = 740 :=
by
  sorry

end NUMINAMATH_GPT_children_in_school_l1709_170969


namespace NUMINAMATH_GPT_swimming_speed_in_still_water_l1709_170961

theorem swimming_speed_in_still_water :
  ∀ (speed_of_water person's_speed time distance: ℝ),
  speed_of_water = 8 →
  time = 1.5 →
  distance = 12 →
  person's_speed - speed_of_water = distance / time →
  person's_speed = 16 :=
by
  intro speed_of_water person's_speed time distance hw ht hd heff
  rw [hw, ht, hd] at heff
  -- steps to isolate person's_speed should be done here, but we leave it as sorry
  sorry

end NUMINAMATH_GPT_swimming_speed_in_still_water_l1709_170961


namespace NUMINAMATH_GPT_largest_number_of_stamps_per_page_l1709_170927

theorem largest_number_of_stamps_per_page :
  Nat.gcd (Nat.gcd 1200 1800) 2400 = 600 :=
sorry

end NUMINAMATH_GPT_largest_number_of_stamps_per_page_l1709_170927


namespace NUMINAMATH_GPT_div_remainder_l1709_170928

theorem div_remainder (x : ℕ) (h : x = 2^40) : 
  (2^160 + 160) % (2^80 + 2^40 + 1) = 159 :=
by
  sorry

end NUMINAMATH_GPT_div_remainder_l1709_170928


namespace NUMINAMATH_GPT_amount_given_by_mom_l1709_170931

def amount_spent_by_Mildred : ℕ := 25
def amount_spent_by_Candice : ℕ := 35
def amount_left : ℕ := 40

theorem amount_given_by_mom : 
  (amount_spent_by_Mildred + amount_spent_by_Candice + amount_left) = 100 := by
  sorry

end NUMINAMATH_GPT_amount_given_by_mom_l1709_170931


namespace NUMINAMATH_GPT_flight_cost_A_to_B_l1709_170965

-- Definitions based on conditions in the problem
def distance_AB : ℝ := 2000
def flight_cost_per_km : ℝ := 0.10
def booking_fee : ℝ := 100

-- Statement: Given the distances and cost conditions, the flight cost from A to B is $300
theorem flight_cost_A_to_B : distance_AB * flight_cost_per_km + booking_fee = 300 := by
  sorry

end NUMINAMATH_GPT_flight_cost_A_to_B_l1709_170965


namespace NUMINAMATH_GPT_rectangles_containment_existence_l1709_170902

theorem rectangles_containment_existence :
  (∃ (rects : ℕ → ℕ × ℕ), (∀ n : ℕ, (rects n).fst > 0 ∧ (rects n).snd > 0) ∧
   (∀ n m : ℕ, n ≠ m → ¬((rects n).fst ≤ (rects m).fst ∧ (rects n).snd ≤ (rects m).snd))) →
  false :=
by
  sorry

end NUMINAMATH_GPT_rectangles_containment_existence_l1709_170902


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_l1709_170971

theorem line_passes_through_fixed_point (a b c : ℝ) (h : a - b + c = 0) : a * 1 + b * (-1) + c = 0 := 
by sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_l1709_170971


namespace NUMINAMATH_GPT_small_monkey_dolls_cheaper_than_large_l1709_170956

theorem small_monkey_dolls_cheaper_than_large (S : ℕ) 
  (h1 : 300 / 6 = 50) 
  (h2 : 300 / S = 75) 
  (h3 : 75 - 50 = 25) : 
  6 - S = 2 := 
sorry

end NUMINAMATH_GPT_small_monkey_dolls_cheaper_than_large_l1709_170956


namespace NUMINAMATH_GPT_true_proposition_l1709_170995

variable (p : Prop) (q : Prop)

-- Introduce the propositions as Lean variables
def prop_p : Prop := ∀ x : ℝ, 2 ^ x > x ^ 2
def prop_q : Prop := ∀ a b : ℝ, ((a > 1 ∧ b > 1) → a * b > 1) ∧ ((a * b > 1) ∧ (¬ (a > 1 ∧ b > 1)))

-- Rewrite the main goal as a Lean statement
theorem true_proposition : ¬ prop_p ∧ prop_q := 
  sorry

end NUMINAMATH_GPT_true_proposition_l1709_170995


namespace NUMINAMATH_GPT_first_year_payment_l1709_170930

theorem first_year_payment (X : ℝ) (second_year : ℝ) (third_year : ℝ) (fourth_year : ℝ) 
    (total_payments : ℝ) 
    (h1 : second_year = X + 2)
    (h2 : third_year = X + 5)
    (h3 : fourth_year = X + 9)
    (h4 : total_payments = X + second_year + third_year + fourth_year) :
    total_payments = 96 → X = 20 :=
by
    sorry

end NUMINAMATH_GPT_first_year_payment_l1709_170930


namespace NUMINAMATH_GPT_ratio_of_first_to_second_l1709_170932

theorem ratio_of_first_to_second (x y : ℕ) 
  (h1 : x + y + (1 / 3 : ℚ) * x = 110)
  (h2 : y = 30) :
  x / y = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_first_to_second_l1709_170932


namespace NUMINAMATH_GPT_tangent_half_angle_sum_eq_product_l1709_170926

variable {α β γ : ℝ}

theorem tangent_half_angle_sum_eq_product (h : α + β + γ = 2 * Real.pi) :
  Real.tan (α / 2) + Real.tan (β / 2) + Real.tan (γ / 2) =
  Real.tan (α / 2) * Real.tan (β / 2) * Real.tan (γ / 2) :=
sorry

end NUMINAMATH_GPT_tangent_half_angle_sum_eq_product_l1709_170926


namespace NUMINAMATH_GPT_polynomial_sum_of_squares_is_23456_l1709_170999

theorem polynomial_sum_of_squares_is_23456 (p q r s t u : ℤ) :
  (∀ x, 1728 * x ^ 3 + 64 = (p * x ^ 2 + q * x + r) * (s * x ^ 2 + t * x + u)) →
  p ^ 2 + q ^ 2 + r ^ 2 + s ^ 2 + t ^ 2 + u ^ 2 = 23456 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_sum_of_squares_is_23456_l1709_170999


namespace NUMINAMATH_GPT_number_of_ways_to_choose_one_class_number_of_ways_to_choose_one_class_each_grade_number_of_ways_to_choose_two_classes_different_grades_l1709_170963

-- Define the number of classes in each grade.
def num_classes_first_year : ℕ := 14
def num_classes_second_year : ℕ := 14
def num_classes_third_year : ℕ := 15

-- Prove the number of different ways to choose students from 1 class.
theorem number_of_ways_to_choose_one_class :
  (num_classes_first_year + num_classes_second_year + num_classes_third_year) = 43 := 
by {
  -- Numerical calculation
  sorry
}

-- Prove the number of different ways to choose students from one class in each grade.
theorem number_of_ways_to_choose_one_class_each_grade :
  (num_classes_first_year * num_classes_second_year * num_classes_third_year) = 2940 := 
by {
  -- Numerical calculation
  sorry
}

-- Prove the number of different ways to choose students from 2 classes from different grades.
theorem number_of_ways_to_choose_two_classes_different_grades :
  (num_classes_first_year * num_classes_second_year + num_classes_first_year * num_classes_third_year + num_classes_second_year * num_classes_third_year) = 616 := 
by {
  -- Numerical calculation
  sorry
}

end NUMINAMATH_GPT_number_of_ways_to_choose_one_class_number_of_ways_to_choose_one_class_each_grade_number_of_ways_to_choose_two_classes_different_grades_l1709_170963


namespace NUMINAMATH_GPT_replace_movie_cost_l1709_170917

def num_popular_action_movies := 20
def num_moderate_comedy_movies := 30
def num_unpopular_drama_movies := 10
def num_popular_comedy_movies := 15
def num_moderate_action_movies := 25

def trade_in_rate_action := 3
def trade_in_rate_comedy := 2
def trade_in_rate_drama := 1

def dvd_cost_popular := 12
def dvd_cost_moderate := 8
def dvd_cost_unpopular := 5

def johns_movie_cost : Nat :=
  let total_trade_in := 
    (num_popular_action_movies + num_moderate_action_movies) * trade_in_rate_action +
    (num_moderate_comedy_movies + num_popular_comedy_movies) * trade_in_rate_comedy +
    num_unpopular_drama_movies * trade_in_rate_drama
  let total_dvd_cost :=
    (num_popular_action_movies + num_popular_comedy_movies) * dvd_cost_popular +
    (num_moderate_comedy_movies + num_moderate_action_movies) * dvd_cost_moderate +
    num_unpopular_drama_movies * dvd_cost_unpopular
  total_dvd_cost - total_trade_in

theorem replace_movie_cost : johns_movie_cost = 675 := 
by
  sorry

end NUMINAMATH_GPT_replace_movie_cost_l1709_170917


namespace NUMINAMATH_GPT_correct_equation_l1709_170922

-- Define the necessary conditions and parameters
variables (x : ℝ)

-- Length of the rectangle
def length := x 

-- Width is 6 meters less than the length
def width := x - 6

-- The area of the rectangle
def area := 720

-- Proof statement
theorem correct_equation : 
  x * (x - 6) = 720 :=
sorry

end NUMINAMATH_GPT_correct_equation_l1709_170922


namespace NUMINAMATH_GPT_Jerry_travel_time_l1709_170900

theorem Jerry_travel_time
  (speed_j speed_b distance_j distance_b time_j time_b : ℝ)
  (h_speed_j : speed_j = 40)
  (h_speed_b : speed_b = 30)
  (h_distance_b : distance_b = distance_j + 5)
  (h_time_b : time_b = time_j + 1/3)
  (h_distance_j : distance_j = speed_j * time_j)
  (h_distance_b_eq : distance_b = speed_b * time_b) :
  time_j = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_Jerry_travel_time_l1709_170900


namespace NUMINAMATH_GPT_angle_of_inclination_of_line_l1709_170908

-- Definition of the line l
def line_eq (x : ℝ) : ℝ := x + 1

-- Statement of the theorem about the angle of inclination
theorem angle_of_inclination_of_line (x : ℝ) : 
  ∃ (θ : ℝ), θ = 45 ∧ line_eq x = x + 1 := 
sorry

end NUMINAMATH_GPT_angle_of_inclination_of_line_l1709_170908


namespace NUMINAMATH_GPT_triangle_inequality_squares_l1709_170944

theorem triangle_inequality_squares (a b c : ℝ) (h₁ : a < b + c) (h₂ : b < a + c) (h₃ : c < a + b) :
  a^2 + b^2 + c^2 < 2 * (a * b + b * c + a * c) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_squares_l1709_170944


namespace NUMINAMATH_GPT_task_assignment_l1709_170988

theorem task_assignment (volunteers : ℕ) (tasks : ℕ) (selected : ℕ) (h_volunteers : volunteers = 6) (h_tasks : tasks = 4) (h_selected : selected = 4) :
  ((Nat.factorial volunteers) / (Nat.factorial (volunteers - selected))) = 360 :=
by
  rw [h_volunteers, h_selected]
  norm_num
  sorry

end NUMINAMATH_GPT_task_assignment_l1709_170988


namespace NUMINAMATH_GPT_radar_coverage_correct_l1709_170983

noncomputable def radar_coverage (r : ℝ) (width : ℝ) : ℝ × ℝ :=
  let θ := Real.pi / 7
  let distance := 40 / Real.sin θ
  let area := 1440 * Real.pi / Real.tan θ
  (distance, area)

theorem radar_coverage_correct : radar_coverage 41 18 = 
  (40 / Real.sin (Real.pi / 7), 1440 * Real.pi / Real.tan (Real.pi / 7)) :=
by
  sorry

end NUMINAMATH_GPT_radar_coverage_correct_l1709_170983


namespace NUMINAMATH_GPT_cells_at_end_of_8th_day_l1709_170972

theorem cells_at_end_of_8th_day :
  let initial_cells := 5
  let factor := 3
  let toxin_factor := 1 / 2
  let cells_after_toxin := (initial_cells * factor * factor * factor * toxin_factor : ℤ)
  let final_cells := cells_after_toxin * factor 
  final_cells = 201 :=
by
  sorry

end NUMINAMATH_GPT_cells_at_end_of_8th_day_l1709_170972


namespace NUMINAMATH_GPT_candle_ratio_proof_l1709_170913

noncomputable def candle_height_ratio := 
  ∃ (x y : ℝ), 
    (x / 6) * 3 = x / 2 ∧
    (y / 8) * 3 = 3 * y / 8 ∧
    (x / 2) = (5 * y / 8) →
    x / y = 5 / 4

theorem candle_ratio_proof : candle_height_ratio :=
by sorry

end NUMINAMATH_GPT_candle_ratio_proof_l1709_170913


namespace NUMINAMATH_GPT_jimmy_change_l1709_170939

noncomputable def change_back (pen_cost notebook_cost folder_cost highlighter_cost sticky_notes_cost total_paid discount tax : ℝ) : ℝ :=
  let total_before_discount := (5 * pen_cost) + (6 * notebook_cost) + (4 * folder_cost) + (3 * highlighter_cost) + (2 * sticky_notes_cost)
  let total_after_discount := total_before_discount * (1 - discount)
  let final_total := total_after_discount * (1 + tax)
  (total_paid - final_total)

theorem jimmy_change :
  change_back 1.65 3.95 4.35 2.80 1.75 150 0.25 0.085 = 100.16 :=
by
  sorry

end NUMINAMATH_GPT_jimmy_change_l1709_170939


namespace NUMINAMATH_GPT_exist_ordering_rectangles_l1709_170991

open Function

structure Rectangle :=
  (left_bot : ℝ × ℝ)  -- Bottom-left corner
  (right_top : ℝ × ℝ)  -- Top-right corner

def below (R1 R2 : Rectangle) : Prop :=
  ∃ g : ℝ, (∀ (x y : ℝ), R1.left_bot.1 ≤ x ∧ x ≤ R1.right_top.1 ∧ R1.left_bot.2 ≤ y ∧ y ≤ R1.right_top.2 → y < g) ∧
           (∀ (x y : ℝ), R2.left_bot.1 <= x ∧ x <= R2.right_top.1 ∧ R2.left_bot.2 <= y ∧ y <= R2.right_top.2 → y > g)

def to_right_of (R1 R2 : Rectangle) : Prop :=
  ∃ h : ℝ, (∀ (x y : ℝ), R1.left_bot.1 ≤ x ∧ x ≤ R1.right_top.1 ∧ R1.left_bot.2 ≤ y ∧ y ≤ R1.right_top.2 → x > h) ∧
           (∀ (x y : ℝ), R2.left_bot.1 <= x ∧ x <= R2.right_top.1 ∧ R2.left_bot.2 <= y ∧ y <= R2.right_top.2 → x < h)

def disjoint (R1 R2 : Rectangle) : Prop :=
  ¬ ((R1.left_bot.1 < R2.right_top.1) ∧ (R1.right_top.1 > R2.left_bot.1) ∧
     (R1.left_bot.2 < R2.right_top.2) ∧ (R1.right_top.2 > R2.left_bot.2))

theorem exist_ordering_rectangles (n : ℕ) (rectangles : Fin n → Rectangle)
  (h_disjoint : ∀ i j, i ≠ j → disjoint (rectangles i) (rectangles j)) :
  ∃ f : Fin n → Fin n, ∀ i j : Fin n, i < j → 
    (to_right_of (rectangles (f i)) (rectangles (f j)) ∨ 
    below (rectangles (f i)) (rectangles (f j))) := 
sorry

end NUMINAMATH_GPT_exist_ordering_rectangles_l1709_170991


namespace NUMINAMATH_GPT_factor_difference_of_squares_l1709_170953

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end NUMINAMATH_GPT_factor_difference_of_squares_l1709_170953


namespace NUMINAMATH_GPT_salt_quantity_l1709_170916

-- Conditions translated to Lean definitions
def cost_of_sugar_per_kg : ℝ := 1.50
def total_cost_sugar_2kg_and_salt (x : ℝ) : ℝ := 5.50
def total_cost_sugar_3kg_and_1kg_salt : ℝ := 5.00

-- Theorem statement
theorem salt_quantity (x : ℝ) : 
  2 * cost_of_sugar_per_kg + x * cost_of_sugar_per_kg / 3 = total_cost_sugar_2kg_and_salt x 
  → 3 * cost_of_sugar_per_kg + x = total_cost_sugar_3kg_and_1kg_salt 
  → x = 5 := 
sorry

end NUMINAMATH_GPT_salt_quantity_l1709_170916


namespace NUMINAMATH_GPT_simplified_expression_l1709_170973

variable (x y : ℝ)

theorem simplified_expression (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 / 5) * Real.sqrt (x * y^2) / ((-4 / 15) * Real.sqrt (y / x)) * ((-5 / 6) * Real.sqrt (x^3 * y)) =
  (15 * x^2 * y * Real.sqrt x) / 8 :=
by
  sorry

end NUMINAMATH_GPT_simplified_expression_l1709_170973


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l1709_170980

theorem repeating_decimal_to_fraction :
  (2 + (35 / 99 : ℚ)) = (233 / 99) := 
sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l1709_170980


namespace NUMINAMATH_GPT_erick_total_money_collected_l1709_170915

noncomputable def new_lemon_price (old_price increase : ℝ) : ℝ := old_price + increase
noncomputable def new_grape_price (old_price increase : ℝ) : ℝ := old_price + increase / 2

noncomputable def total_money_collected (lemons grapes : ℕ)
                                       (lemon_price grape_price lemon_increase : ℝ) : ℝ :=
  let new_lemon_price := new_lemon_price lemon_price lemon_increase
  let new_grape_price := new_grape_price grape_price lemon_increase
  lemons * new_lemon_price + grapes * new_grape_price

theorem erick_total_money_collected :
  total_money_collected 80 140 8 7 4 = 2220 := 
by
  sorry

end NUMINAMATH_GPT_erick_total_money_collected_l1709_170915


namespace NUMINAMATH_GPT_part_I_part_II_l1709_170907

-- Define the triangle and sides
structure Triangle :=
  (A B C : ℝ)   -- angles in the triangle
  (a b c : ℝ)   -- sides opposite to respective angles

-- Express given conditions in the problem
def conditions (T: Triangle) : Prop :=
  2 * (1 / (Real.tan T.A) + 1 / (Real.tan T.C)) = 1 / (Real.sin T.A) + 1 / (Real.sin T.C)

-- First theorem statement
theorem part_I (T : Triangle) : conditions T → (T.a + T.c = 2 * T.b) :=
sorry

-- Second theorem statement
theorem part_II (T : Triangle) : conditions T → (T.B ≤ Real.pi / 3) :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l1709_170907


namespace NUMINAMATH_GPT_larger_segment_of_triangle_l1709_170966

theorem larger_segment_of_triangle (a b c : ℝ) (h : ℝ) (hc : c = 100) (ha : a = 40) (hb : b = 90) 
  (h_triangle : a^2 + h^2 = x^2)
  (h_triangle2 : b^2 + h^2 = (100 - x)^2) :
  100 - x = 82.5 :=
sorry

end NUMINAMATH_GPT_larger_segment_of_triangle_l1709_170966


namespace NUMINAMATH_GPT_transformed_roots_l1709_170905

theorem transformed_roots 
  (a b c : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : a * (-1)^2 + b * (-1) + c = 0)
  (h₃ : a * 2^2 + b * 2 + c = 0) :
  (a * 0^2 + b * 0 + c = 0) ∧ (a * 3^2 + b * 3 + c = 0) :=
by 
  sorry

end NUMINAMATH_GPT_transformed_roots_l1709_170905


namespace NUMINAMATH_GPT_solve_for_k_l1709_170941

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 2

theorem solve_for_k (k : ℤ) (h1 : k % 2 = 1) (h2 : f (f (f k)) = 57) : k = 223 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_solve_for_k_l1709_170941


namespace NUMINAMATH_GPT_coo_coo_count_correct_l1709_170981

theorem coo_coo_count_correct :
  let monday_coos := 89
  let tuesday_coos := 179
  let wednesday_coos := 21
  let total_coos := monday_coos + tuesday_coos + wednesday_coos
  total_coos = 289 :=
by
  sorry

end NUMINAMATH_GPT_coo_coo_count_correct_l1709_170981


namespace NUMINAMATH_GPT_number_of_valid_ns_l1709_170987

theorem number_of_valid_ns :
  ∃ (n : ℝ), (n = 8 ∨ n = 1/2) ∧ ∀ n₁ n₂, (n₁ = 8 ∨ n₁ = 1/2) ∧ (n₂ = 8 ∨ n₂ = 1/2) → n₁ = n₂ :=
sorry

end NUMINAMATH_GPT_number_of_valid_ns_l1709_170987


namespace NUMINAMATH_GPT_speed_of_current_l1709_170946

theorem speed_of_current (v : ℝ) : 
  (∀ s, s = 3 → s / (3 - v) = 2.3076923076923075) → v = 1.7 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_speed_of_current_l1709_170946


namespace NUMINAMATH_GPT_power_eval_l1709_170994

theorem power_eval : (9^6 * 3^4) / (27^5) = 3 := by
  sorry

end NUMINAMATH_GPT_power_eval_l1709_170994


namespace NUMINAMATH_GPT_det_matrixB_eq_neg_one_l1709_170962

variable (x y : ℝ)

def matrixB : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![x, 3],
  ![-4, y]
]

theorem det_matrixB_eq_neg_one 
  (h : matrixB x y - (matrixB x y)⁻¹ = 2 • (1 : Matrix (Fin 2) (Fin 2) ℝ)) :
  Matrix.det (matrixB x y) = -1 := sorry

end NUMINAMATH_GPT_det_matrixB_eq_neg_one_l1709_170962


namespace NUMINAMATH_GPT_a_alone_time_to_complete_work_l1709_170929

theorem a_alone_time_to_complete_work :
  (W : ℝ) →
  (A : ℝ) →
  (B : ℝ) →
  (h1 : A + B = W / 6) →
  (h2 : B = W / 12) →
  A = W / 12 :=
by
  -- Given conditions
  intros W A B h1 h2
  -- Proof is not needed as per instructions
  sorry

end NUMINAMATH_GPT_a_alone_time_to_complete_work_l1709_170929


namespace NUMINAMATH_GPT_georgia_total_cost_l1709_170912

def carnation_price : ℝ := 0.50
def dozen_price : ℝ := 4.00
def teachers : ℕ := 5
def friends : ℕ := 14

theorem georgia_total_cost :
  ((dozen_price * teachers) + dozen_price + (carnation_price * (friends - 12))) = 25.00 :=
by
  sorry

end NUMINAMATH_GPT_georgia_total_cost_l1709_170912


namespace NUMINAMATH_GPT_solution_is_correct_l1709_170986

noncomputable def solve_system_of_inequalities : Prop :=
  ∃ x y : ℝ, 
    (13 * x^2 - 4 * x * y + 4 * y^2 ≤ 2) ∧ 
    (2 * x - 4 * y ≤ -3) ∧ 
    (x = -1/3) ∧ 
    (y = 2/3)

theorem solution_is_correct : solve_system_of_inequalities :=
sorry

end NUMINAMATH_GPT_solution_is_correct_l1709_170986


namespace NUMINAMATH_GPT_additional_airplanes_needed_l1709_170921

theorem additional_airplanes_needed (total_current_airplanes : ℕ) (airplanes_per_row : ℕ) 
  (h_current_airplanes : total_current_airplanes = 37) 
  (h_airplanes_per_row : airplanes_per_row = 8) : 
  ∃ additional_airplanes : ℕ, additional_airplanes = 3 ∧ 
  ((total_current_airplanes + additional_airplanes) % airplanes_per_row = 0) :=
by
  sorry

end NUMINAMATH_GPT_additional_airplanes_needed_l1709_170921


namespace NUMINAMATH_GPT_max_teams_participation_l1709_170984

theorem max_teams_participation (n : ℕ) (H : 9 * n * (n - 1) / 2 ≤ 200) : n ≤ 7 := by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_max_teams_participation_l1709_170984


namespace NUMINAMATH_GPT_existence_of_rational_solutions_a_nonexistence_of_rational_solutions_b_l1709_170943

def is_rational_non_integer (a : ℚ) : Prop :=
  ¬ (∃ n : ℤ, a = n)

theorem existence_of_rational_solutions_a :
  ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧
    ∃ k1 k2 : ℤ, 19 * x + 8 * y = k1 ∧ 8 * x + 3 * y = k2 :=
sorry

theorem nonexistence_of_rational_solutions_b :
  ¬ (∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧
    ∃ k1 k2 : ℤ, 19 * x^2 + 8 * y^2 = k1 ∧ 8 * x^2 + 3 * y^2 = k2) :=
sorry

end NUMINAMATH_GPT_existence_of_rational_solutions_a_nonexistence_of_rational_solutions_b_l1709_170943


namespace NUMINAMATH_GPT_largest_triangle_perimeter_l1709_170923

theorem largest_triangle_perimeter :
  ∀ (x : ℕ), 1 < x ∧ x < 15 → (7 + 8 + x = 29) :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_largest_triangle_perimeter_l1709_170923


namespace NUMINAMATH_GPT_find_number_l1709_170997

theorem find_number (x : ℤ) (h : 3 * x - 4 = 5) : x = 3 :=
sorry

end NUMINAMATH_GPT_find_number_l1709_170997


namespace NUMINAMATH_GPT_proof_f_value_l1709_170954

noncomputable def f : ℝ → ℝ
| x => if x ≤ 1 then 1 - x^2 else 2^x

theorem proof_f_value : f (1 / f (Real.log 6 / Real.log 2)) = 35 / 36 := by
  sorry

end NUMINAMATH_GPT_proof_f_value_l1709_170954


namespace NUMINAMATH_GPT_transformation_correct_l1709_170974

theorem transformation_correct (a b c : ℝ) (h : a / c = b / c) (hc : c ≠ 0) : a = b :=
sorry

end NUMINAMATH_GPT_transformation_correct_l1709_170974


namespace NUMINAMATH_GPT_equal_real_roots_a_value_l1709_170934

theorem equal_real_roots_a_value (a : ℝ) :
  a ≠ 0 →
  let b := -4
  let c := 3
  b * b - 4 * a * c = 0 →
  a = 4 / 3 :=
by
  intros h_nonzero h_discriminant
  sorry

end NUMINAMATH_GPT_equal_real_roots_a_value_l1709_170934


namespace NUMINAMATH_GPT_cost_of_notebook_l1709_170975

theorem cost_of_notebook (s n c : ℕ) 
    (h1 : s > 18) 
    (h2 : n ≥ 2) 
    (h3 : c > n) 
    (h4 : s * c * n = 2376) : 
    c = 11 := 
  sorry

end NUMINAMATH_GPT_cost_of_notebook_l1709_170975


namespace NUMINAMATH_GPT_polygon_perimeter_l1709_170950

theorem polygon_perimeter (side_length : ℝ) (ext_angle_deg : ℝ) (n : ℕ) (h1 : side_length = 8) 
  (h2 : ext_angle_deg = 90) (h3 : ext_angle_deg = 360 / n) : 
  4 * side_length = 32 := 
  by 
    sorry

end NUMINAMATH_GPT_polygon_perimeter_l1709_170950


namespace NUMINAMATH_GPT_shaded_area_of_joined_squares_l1709_170945

theorem shaded_area_of_joined_squares:
  ∀ (a b : ℕ) (area_of_shaded : ℝ),
  (a = 6) → (b = 8) → 
  (area_of_shaded = (6 * 6 : ℝ) + (8 * 8 : ℝ) / 2) →
  area_of_shaded = 50.24 := 
by
  intros a b area_of_shaded h1 h2 h3
  -- skipping the proof for now
  sorry

end NUMINAMATH_GPT_shaded_area_of_joined_squares_l1709_170945


namespace NUMINAMATH_GPT_part1_part2_l1709_170904

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem part1 (hx : f (-x) = 2 * f x) : f x ^ 2 = 2 / 5 := 
  sorry

theorem part2 : 
  ∀ k : ℤ, ∃ a b : ℝ, [a, b] = [2 * π * k + (5 * π / 6), 2 * π * k + (11 * π / 6)] ∧ 
  ∀ x : ℝ, x ∈ Set.Icc a b → ∀ y : ℝ, y = f (π / 12 - x) → 
  ∃ δ > 0, ∀ ε > 0, 0 < |x - y| ∧ |x - y| < δ → y < x := 
  sorry

end NUMINAMATH_GPT_part1_part2_l1709_170904


namespace NUMINAMATH_GPT_N_subset_proper_M_l1709_170918

open Set Int

def set_M : Set ℝ := {x | ∃ k : ℤ, x = (k + 2) / 4}
def set_N : Set ℝ := {x | ∃ k : ℤ, x = (2 * k + 1) / 4}

theorem N_subset_proper_M : set_N ⊂ set_M := by
  sorry

end NUMINAMATH_GPT_N_subset_proper_M_l1709_170918


namespace NUMINAMATH_GPT_shorter_base_of_isosceles_trapezoid_l1709_170940

theorem shorter_base_of_isosceles_trapezoid
  (a b : ℝ)
  (h : a > b)
  (h_division : (a + b) / 2 = (a - b) / 2 + 10) :
  b = 10 :=
by
  sorry

end NUMINAMATH_GPT_shorter_base_of_isosceles_trapezoid_l1709_170940


namespace NUMINAMATH_GPT_inequality_holds_l1709_170914

variable {a b c : ℝ}

theorem inequality_holds (h : a > 0) (h' : b > 0) (h'' : c > 0) (h_abc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l1709_170914


namespace NUMINAMATH_GPT_find_two_digit_numbers_l1709_170947

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem find_two_digit_numbers :
  ∀ (A : ℕ), (10 ≤ A ∧ A ≤ 99) →
    (sum_of_digits A)^2 = sum_of_digits (A^2) →
    (A = 11 ∨ A = 12 ∨ A = 13 ∨ A = 20 ∨ A = 21 ∨ A = 22 ∨ A = 30 ∨ A = 31 ∨ A = 50) :=
by sorry

end NUMINAMATH_GPT_find_two_digit_numbers_l1709_170947


namespace NUMINAMATH_GPT_men_seated_l1709_170960

theorem men_seated (total_passengers : ℕ) (women_ratio : ℚ) (children_count : ℕ) (men_standing_ratio : ℚ) 
  (women_with_prams : ℕ) (disabled_passengers : ℕ) 
  (h_total_passengers : total_passengers = 48) 
  (h_women_ratio : women_ratio = 2 / 3) 
  (h_children_count : children_count = 5) 
  (h_men_standing_ratio : men_standing_ratio = 1 / 8) 
  (h_women_with_prams : women_with_prams = 3) 
  (h_disabled_passengers : disabled_passengers = 2) : 
  (total_passengers * (1 - women_ratio) - total_passengers * (1 - women_ratio) * men_standing_ratio = 14) :=
by sorry

end NUMINAMATH_GPT_men_seated_l1709_170960


namespace NUMINAMATH_GPT_percentage_saved_l1709_170937

theorem percentage_saved (rent milk groceries education petrol misc savings : ℝ) 
  (salary : ℝ) 
  (h_rent : rent = 5000) 
  (h_milk : milk = 1500) 
  (h_groceries : groceries = 4500) 
  (h_education : education = 2500) 
  (h_petrol : petrol = 2000) 
  (h_misc : misc = 700) 
  (h_savings : savings = 1800) 
  (h_salary : salary = rent + milk + groceries + education + petrol + misc + savings) : 
  (savings / salary) * 100 = 10 :=
by
  sorry

end NUMINAMATH_GPT_percentage_saved_l1709_170937


namespace NUMINAMATH_GPT_geom_seq_sum_l1709_170959

theorem geom_seq_sum (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 3)
  (h2 : a 1 + a 2 + a 3 = 21)
  (h3 : ∀ n, a (n + 1) = a n * q) : a 4 + a 5 + a 6 = 168 :=
sorry

end NUMINAMATH_GPT_geom_seq_sum_l1709_170959


namespace NUMINAMATH_GPT_ramu_spent_on_repairs_l1709_170952

theorem ramu_spent_on_repairs 
    (initial_cost : ℝ) (selling_price : ℝ) (profit_percent : ℝ) (R : ℝ) 
    (h1 : initial_cost = 42000) 
    (h2 : selling_price = 64900) 
    (h3 : profit_percent = 18) 
    (h4 : profit_percent / 100 = (selling_price - (initial_cost + R)) / (initial_cost + R)) : 
    R = 13000 :=
by
  rw [h1, h2, h3] at h4
  sorry

end NUMINAMATH_GPT_ramu_spent_on_repairs_l1709_170952


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l1709_170948

theorem quadratic_inequality_solution_set :
  {x : ℝ | x * (x - 2) > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 2} :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l1709_170948


namespace NUMINAMATH_GPT_water_purification_problem_l1709_170957

variable (x : ℝ) (h : x > 0)

theorem water_purification_problem
  (h1 : ∀ (p : ℝ), p = 2400)
  (h2 : ∀ (eff : ℝ), eff = 1.2)
  (h3 : ∀ (time_saved : ℝ), time_saved = 40) :
  (2400 * 1.2 / x) - (2400 / x) = 40 := by
  sorry

end NUMINAMATH_GPT_water_purification_problem_l1709_170957


namespace NUMINAMATH_GPT_days_elapsed_l1709_170964

theorem days_elapsed
  (initial_amount : ℕ)
  (daily_spending : ℕ)
  (total_savings : ℕ)
  (doubling_factor : ℕ)
  (additional_amount : ℕ)
  :
  initial_amount = 50 →
  daily_spending = 15 →
  doubling_factor = 2 →
  additional_amount = 10 →
  2 * (initial_amount - daily_spending) * total_savings + additional_amount = 500 →
  total_savings = 7 :=
by
  intros h_initial h_spending h_doubling h_additional h_total
  sorry

end NUMINAMATH_GPT_days_elapsed_l1709_170964


namespace NUMINAMATH_GPT_find_other_endpoint_l1709_170901

theorem find_other_endpoint (x1 y1 x_m y_m x y : ℝ) 
  (h1 : (x_m, y_m) = (3, 7))
  (h2 : (x1, y1) = (0, 11)) :
  (x, y) = (6, 3) ↔ (x_m = (x1 + x) / 2 ∧ y_m = (y1 + y) / 2) :=
by
  simp at h1 h2
  simp
  sorry

end NUMINAMATH_GPT_find_other_endpoint_l1709_170901


namespace NUMINAMATH_GPT_product_of_all_possible_values_of_x_l1709_170903

def conditions (x : ℚ) : Prop := abs (18 / x - 4) = 3

theorem product_of_all_possible_values_of_x:
  ∃ x1 x2 : ℚ, conditions x1 ∧ conditions x2 ∧ ((18 * 18) / (x1 * x2) = 324 / 7) :=
sorry

end NUMINAMATH_GPT_product_of_all_possible_values_of_x_l1709_170903


namespace NUMINAMATH_GPT_band_first_set_songs_count_l1709_170919

theorem band_first_set_songs_count 
  (total_repertoire : ℕ) (second_set : ℕ) (encore : ℕ) (avg_third_fourth : ℕ)
  (h_total_repertoire : total_repertoire = 30)
  (h_second_set : second_set = 7)
  (h_encore : encore = 2)
  (h_avg_third_fourth : avg_third_fourth = 8)
  : ∃ (x : ℕ), x + second_set + encore + avg_third_fourth * 2 = total_repertoire := 
  sorry

end NUMINAMATH_GPT_band_first_set_songs_count_l1709_170919


namespace NUMINAMATH_GPT_fg_of_1_eq_15_l1709_170951

def f (x : ℝ) := 2 * x - 3
def g (x : ℝ) := (x + 2) ^ 2

theorem fg_of_1_eq_15 : f (g 1) = 15 :=
by
  sorry

end NUMINAMATH_GPT_fg_of_1_eq_15_l1709_170951


namespace NUMINAMATH_GPT_tangent_curve_line_l1709_170982

/-- Given the line y = x + 1 and the curve y = ln(x + a) are tangent, prove that the value of a is 2. -/
theorem tangent_curve_line (a : ℝ) :
  (∃ x₀ y₀, y₀ = x₀ + 1 ∧ y₀ = Real.log (x₀ + a) ∧ (1 / (x₀ + a) = 1)) → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_tangent_curve_line_l1709_170982


namespace NUMINAMATH_GPT_total_tires_l1709_170906

def cars := 15
def bicycles := 3
def pickup_trucks := 8
def tricycles := 1

def tires_per_car := 4
def tires_per_bicycle := 2
def tires_per_pickup_truck := 4
def tires_per_tricycle := 3

theorem total_tires : (cars * tires_per_car) + (bicycles * tires_per_bicycle) + (pickup_trucks * tires_per_pickup_truck) + (tricycles * tires_per_tricycle) = 101 :=
by
  sorry

end NUMINAMATH_GPT_total_tires_l1709_170906


namespace NUMINAMATH_GPT_value_of_sum_l1709_170977

theorem value_of_sum (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (hc_solution : c^2 + a * c + b = 0) (hd_solution : d^2 + a * d + b = 0)
  (ha_solution : a^2 + c * a + d = 0) (hb_solution : b^2 + c * b + d = 0)
: a + b + c + d = -2 := sorry -- The proof is omitted as requested

end NUMINAMATH_GPT_value_of_sum_l1709_170977


namespace NUMINAMATH_GPT_find_starting_number_l1709_170970

theorem find_starting_number (x : ℕ) (h1 : (50 + 250) / 2 = 150)
  (h2 : (x + 400) / 2 = 150 + 100) : x = 100 := by
  sorry

end NUMINAMATH_GPT_find_starting_number_l1709_170970


namespace NUMINAMATH_GPT_opposite_of_neg_half_l1709_170938

-- Define the opposite of a number
def opposite (x : ℝ) : ℝ := -x

-- The theorem we want to prove
theorem opposite_of_neg_half : opposite (-1/2) = 1/2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_opposite_of_neg_half_l1709_170938


namespace NUMINAMATH_GPT_solve_inequality_l1709_170993

theorem solve_inequality : {x : ℝ | 9 * x^2 + 6 * x + 1 ≤ 0} = {(-1 : ℝ) / 3} :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1709_170993


namespace NUMINAMATH_GPT_circle_line_intersection_points_l1709_170968

noncomputable def radius : ℝ := 6
noncomputable def distance : ℝ := 5

theorem circle_line_intersection_points :
  radius > distance -> number_of_intersection_points = 2 := 
by
  sorry

end NUMINAMATH_GPT_circle_line_intersection_points_l1709_170968


namespace NUMINAMATH_GPT_smallest_prime_with_conditions_l1709_170933

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n
def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def reverse_digits (n : ℕ) : ℕ := 
  let tens := n / 10 
  let units := n % 10 
  units * 10 + tens

theorem smallest_prime_with_conditions : 
  ∃ (p : ℕ), is_prime p ∧ 20 ≤ p ∧ p < 30 ∧ (reverse_digits p) < 100 ∧ is_composite (reverse_digits p) ∧ p = 23 :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_with_conditions_l1709_170933
