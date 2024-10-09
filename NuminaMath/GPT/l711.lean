import Mathlib

namespace hyperbola_equation_l711_71137

-- Fixed points F_1 and F_2
def F1 : ℝ × ℝ := (5, 0)
def F2 : ℝ × ℝ := (-5, 0)

-- Condition: The absolute value of the difference in distances from P to F1 and F2 is 6
def distance_condition (P : ℝ × ℝ) : Prop :=
  abs ((dist P F1) - (dist P F2)) = 6

theorem hyperbola_equation : 
  ∃ (a b : ℝ), a = 3 ∧ b = 4 ∧ ∀ (x y : ℝ), distance_condition (x, y) → 
  (x ^ 2) / (a ^ 2) - (y ^ 2) / (b ^ 2) = 1 :=
by
  -- We state the conditions and result derived from them
  sorry

end hyperbola_equation_l711_71137


namespace sequence_property_l711_71121

theorem sequence_property (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 9) : 7 * n * 15873 = n * 111111 :=
by sorry

end sequence_property_l711_71121


namespace train_cross_time_l711_71176

noncomputable def time_to_cross_pole (length: ℝ) (speed_kmh: ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  length / speed_ms

theorem train_cross_time :
  let length := 100
  let speed := 126
  abs (time_to_cross_pole length speed - 2.8571) < 0.0001 :=
by
  let length := 100
  let speed := 126
  have h1 : abs (time_to_cross_pole length speed - 2.8571) < 0.0001
  sorry
  exact h1

end train_cross_time_l711_71176


namespace product_ab_l711_71105

noncomputable def median_of_four_numbers (a b : ℕ) := 3
noncomputable def mean_of_four_numbers (a b : ℕ) := 4

theorem product_ab (a b : ℕ)
  (h1 : 1 + 2 + a + b = 4 * 4)
  (h2 : median_of_four_numbers a b = 3)
  (h3 : mean_of_four_numbers a b = 4) : (a * b = 36) :=
by sorry

end product_ab_l711_71105


namespace commutative_l711_71135

variable (R : Type) [NonAssocRing R]
variable (star : R → R → R)

axiom assoc : ∀ x y z : R, star (star x y) z = star x (star y z)
axiom comm_left : ∀ x y z : R, star (star x y) z = star (star y z) x
axiom distinct : ∀ {x y : R}, x ≠ y → ∃ z : R, star z x ≠ star z y

theorem commutative (x y : R) : star x y = star y x := sorry

end commutative_l711_71135


namespace find_r_k_l711_71195

theorem find_r_k :
  ∃ r k : ℚ, (∀ t : ℚ, (∃ x y : ℚ, (x = r + 3 * t ∧ y = 2 + k * t) ∧ y = 5 * x - 7)) ∧ 
            r = 9 / 5 ∧ k = -4 :=
by {
  sorry
}

end find_r_k_l711_71195


namespace find_first_number_l711_71117

-- Definitions from conditions
variable (x : ℕ) -- Let the first number be x
variable (y : ℕ) -- Let the second number be y

-- Given conditions in the problem
def condition1 : Prop := y = 43
def condition2 : Prop := x + 2 * y = 124

-- The proof target
theorem find_first_number (h1 : condition1 y) (h2 : condition2 x y) : x = 38 := by
  sorry

end find_first_number_l711_71117


namespace shuttle_speed_l711_71116

theorem shuttle_speed (speed_kps : ℕ) (conversion_factor : ℕ) (speed_kph : ℕ) :
  speed_kps = 2 → conversion_factor = 3600 → speed_kph = speed_kps * conversion_factor → speed_kph = 7200 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end shuttle_speed_l711_71116


namespace four_digit_div_by_14_l711_71127

theorem four_digit_div_by_14 (n : ℕ) (h₁ : 9450 + n < 10000) :
  (∃ k : ℕ, 9450 + n = 14 * k) ↔ (n = 8) := by
  sorry

end four_digit_div_by_14_l711_71127


namespace infinite_series_sum_l711_71148

theorem infinite_series_sum :
  ∑' (n : ℕ), (n + 1) / 10^(n + 1) = 10 / 81 :=
sorry

end infinite_series_sum_l711_71148


namespace scientific_notation_correct_l711_71124

theorem scientific_notation_correct : 1630000 = 1.63 * 10^6 :=
by sorry

end scientific_notation_correct_l711_71124


namespace point_in_second_quadrant_condition_l711_71192

theorem point_in_second_quadrant_condition (a : ℤ)
  (h1 : 3 * a - 9 < 0)
  (h2 : 10 - 2 * a > 0)
  (h3 : |3 * a - 9| = |10 - 2 * a|):
  (a + 2) ^ 2023 - 1 = 0 := 
sorry

end point_in_second_quadrant_condition_l711_71192


namespace base_six_equals_base_b_l711_71141

noncomputable def base_six_to_decimal (n : ℕ) : ℕ :=
  6 * 6 + 2

noncomputable def base_b_to_decimal (b : ℕ) : ℕ :=
  b^2 + 2 * b + 4

theorem base_six_equals_base_b (b : ℕ) : b^2 + 2 * b - 34 = 0 → b = 4 := 
by sorry

end base_six_equals_base_b_l711_71141


namespace angle_A_sides_b_c_l711_71169

noncomputable def triangle_angles (a b c : ℝ) (A B C : ℝ) : Prop :=
  a * Real.sin C - Real.sqrt 3 * c * Real.cos A = 0

theorem angle_A (a b c A B C : ℝ) (h1 : triangle_angles a b c A B C) :
  A = Real.pi / 3 :=
by sorry

noncomputable def triangle_area (a b c S : ℝ) : Prop :=
  S = Real.sqrt 3 ∧ a = 2

theorem sides_b_c (a b c S : ℝ) (h : triangle_area a b c S) :
  b = 2 ∧ c = 2 :=
by sorry

end angle_A_sides_b_c_l711_71169


namespace find_angle_between_planes_l711_71149

noncomputable def angle_between_planes (α β : ℝ) : ℝ := Real.arcsin ((Real.sqrt 6 + 1) / 5)

theorem find_angle_between_planes (α β : ℝ) (h : α = β) : 
  (∃ (cube : Type) (A B C D A₁ B₁ C₁ D₁ : cube),
    α = Real.arcsin ((Real.sqrt 6 - 1) / 5) ∨ α = Real.arcsin ((Real.sqrt 6 + 1) / 5)) 
    :=
sorry

end find_angle_between_planes_l711_71149


namespace contrapositive_statement_l711_71164

theorem contrapositive_statement (x : ℝ) : (x ≤ -3 → x < 0) → (x ≥ 0 → x > -3) := 
by
  sorry

end contrapositive_statement_l711_71164


namespace sum_mod_7_remainder_l711_71122

def sum_to (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem sum_mod_7_remainder : (sum_to 140) % 7 = 0 :=
by
  sorry

end sum_mod_7_remainder_l711_71122


namespace julie_hours_per_week_school_year_l711_71104

-- Defining the assumptions
variable (summer_hours_per_week : ℕ) (summer_weeks : ℕ) (summer_earnings : ℝ)
variable (school_year_weeks : ℕ) (school_year_earnings : ℝ)

-- Assuming the given values
def assumptions : Prop :=
  summer_hours_per_week = 36 ∧ 
  summer_weeks = 10 ∧ 
  summer_earnings = 4500 ∧ 
  school_year_weeks = 45 ∧ 
  school_year_earnings = 4500

-- Proving that Julie must work 8 hours per week during the school year to make another $4500
theorem julie_hours_per_week_school_year : 
  assumptions summer_hours_per_week summer_weeks summer_earnings school_year_weeks school_year_earnings →
  (school_year_earnings / (summer_earnings / (summer_hours_per_week * summer_weeks)) / school_year_weeks = 8) :=
by
  sorry

end julie_hours_per_week_school_year_l711_71104


namespace find_remainder_when_q_divided_by_x_plus_2_l711_71175

noncomputable def q (x : ℝ) (D E F : ℝ) := D * x^4 + E * x^2 + F * x + 5

theorem find_remainder_when_q_divided_by_x_plus_2 (D E F : ℝ) :
  q 2 D E F = 15 → q (-2) D E F = 15 :=
by
  intro h
  sorry

end find_remainder_when_q_divided_by_x_plus_2_l711_71175


namespace dave_deleted_apps_l711_71193

theorem dave_deleted_apps : 
  ∀ (a_initial a_left a_deleted : ℕ), a_initial = 16 → a_left = 5 → a_deleted = a_initial - a_left → a_deleted = 11 :=
by
  intros a_initial a_left a_deleted h_initial h_left h_deleted
  rw [h_initial, h_left] at h_deleted
  exact h_deleted

end dave_deleted_apps_l711_71193


namespace find_monthly_fee_l711_71177

-- Definitions from conditions
def monthly_fee (total_bill : ℝ) (cost_per_minute : ℝ) (minutes_used : ℝ) : ℝ :=
  total_bill - cost_per_minute * minutes_used

-- Theorem stating the question
theorem find_monthly_fee :
  let total_bill := 12.02
  let cost_per_minute := 0.25
  let minutes_used := 28.08
  total_bill - cost_per_minute * minutes_used = 5.00 :=
by
  -- Definition of variables used in the theorem
  let total_bill := 12.02
  let cost_per_minute := 0.25
  let minutes_used := 28.08
  
  -- The statement of the theorem and leaving the proof as an exercise
  show total_bill - cost_per_minute * minutes_used = 5.00
  sorry

end find_monthly_fee_l711_71177


namespace remaining_minutes_proof_l711_71143

def total_series_minutes : ℕ := 360

def first_session_end : ℕ := 17 * 60 + 44  -- in minutes
def first_session_start : ℕ := 15 * 60 + 20  -- in minutes
def second_session_end : ℕ := 20 * 60 + 40  -- in minutes
def second_session_start : ℕ := 19 * 60 + 15  -- in minutes
def third_session_end : ℕ := 22 * 60 + 30  -- in minutes
def third_session_start : ℕ := 21 * 60 + 35  -- in minutes

def first_session_duration : ℕ := first_session_end - first_session_start
def second_session_duration : ℕ := second_session_end - second_session_start
def third_session_duration : ℕ := third_session_end - third_session_start

def total_watched : ℕ := first_session_duration + second_session_duration + third_session_duration

def remaining_time : ℕ := total_series_minutes - total_watched

theorem remaining_minutes_proof : remaining_time = 76 := 
by 
  sorry  -- Proof goes here

end remaining_minutes_proof_l711_71143


namespace slope_of_asymptotes_l711_71125

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  (x - 2)^2 / 144 - (y + 3)^2 / 81 = 1

-- The theorem stating the slope of the asymptotes
theorem slope_of_asymptotes : ∀ x y : ℝ, hyperbola x y → (∃ m : ℝ, m = 3 / 4) :=
by
  sorry

end slope_of_asymptotes_l711_71125


namespace hyperbola_condition_l711_71134

theorem hyperbola_condition (k : ℝ) : (k > 1) -> ( ∀ x y : ℝ, (k - 1) * (k + 1) > 0 ↔ ( ∃ x y : ℝ, (k > 1) ∧ ((x * x) / (k - 1) - (y * y) / (k + 1)) = 1)) :=
sorry

end hyperbola_condition_l711_71134


namespace sequence_formula_and_sum_l711_71100

def arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a : ℕ → ℝ) :=
  ∀ m n k, m < n → n < k → a n^2 = a m * a k

def Sn (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

theorem sequence_formula_and_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (arithmetic_geometric_sequence a 0 ∧ a 1 = 2 ∧ geometric_sequence a → ∀ n, a n = 2) ∧
  (arithmetic_geometric_sequence a 4 ∧ a 1 = 2 ∧ geometric_sequence a → ∀ n, a n = 4 * n - 2) ∧
  (arithmetic_geometric_sequence a 4 ∧ a 1 = 2 ∧ (∀ n, S n = (n * (4 * n)) / 2) → ∃ n > 0, S n > 60 * n + 800 ∧ n = 41) ∧
  (arithmetic_geometric_sequence a 0 ∧ a 1 = 2 ∧ (∀ n, S n = 2 * n) → ∀ n > 0, ¬ (S n > 60 * n + 800)) :=
by sorry

end sequence_formula_and_sum_l711_71100


namespace solution_set_inequality_l711_71131

-- Statement of the problem
theorem solution_set_inequality :
  {x : ℝ | 1 / x < 1 / 2} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 2} :=
sorry

end solution_set_inequality_l711_71131


namespace vasya_number_l711_71113

theorem vasya_number (a b c d : ℕ) (h1 : a * b = 21) (h2 : b * c = 20) (h3 : ∃ x, x ∈ [4, 7] ∧ a ≠ c ∧ b = 7 ∧ c = 4 ∧ d = 5) : (1000 * a + 100 * b + 10 * c + d) = 3745 :=
sorry

end vasya_number_l711_71113


namespace not_perfect_square_l711_71129

theorem not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, (3^n + 2 * 17^n) = k^2 :=
by
  sorry

end not_perfect_square_l711_71129


namespace reflection_ray_equation_l711_71115

theorem reflection_ray_equation (x y : ℝ) : (y = 2 * x + 1) → (∃ (x' y' : ℝ), y' = x ∧ y = 2 * x' + 1 ∧ x - 2 * y - 1 = 0) :=
by
  intro h
  sorry

end reflection_ray_equation_l711_71115


namespace fraction_proof_l711_71167

theorem fraction_proof (a b : ℚ) (h : a / b = 3 / 4) : (a + b) / b = 7 / 4 :=
by
  sorry

end fraction_proof_l711_71167


namespace range_eq_domain_l711_71161

def f (x : ℝ) : ℝ := |x - 2| - 2

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

theorem range_eq_domain : (Set.range f) = M :=
by
  sorry

end range_eq_domain_l711_71161


namespace max_value_of_sum_on_ellipse_l711_71162

theorem max_value_of_sum_on_ellipse (x y : ℝ) (h : x^2 / 3 + y^2 = 1) : x + y ≤ 2 :=
sorry

end max_value_of_sum_on_ellipse_l711_71162


namespace main_theorem_l711_71166

def f (m: ℕ) : ℕ := m * (m + 1) / 2

lemma f_1 : f 1 = 1 := by 
  -- placeholder for proof
  sorry

lemma f_functional_eq (m n : ℕ) : f m + f n = f (m + n) - m * n := by
  -- placeholder for proof
  sorry

theorem main_theorem (m : ℕ) : f m = m * (m + 1) / 2 := by
  -- Combining the conditions to conclude the result
  sorry

end main_theorem_l711_71166


namespace min_value_p_plus_q_l711_71154

-- Definitions related to the conditions.
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def satisfies_equations (a b p q : ℕ) : Prop :=
  20 * a + 17 * b = p ∧ 17 * a + 20 * b = q ∧ is_prime p ∧ is_prime q

def distinct_positive_integers (a b : ℕ) : Prop := a > 0 ∧ b > 0 ∧ a ≠ b

-- The main proof problem.
theorem min_value_p_plus_q (a b p q : ℕ) :
  distinct_positive_integers a b →
  satisfies_equations a b p q →
  p + q = 296 :=
by
  sorry

end min_value_p_plus_q_l711_71154


namespace arun_speed_ratio_l711_71168

namespace SpeedRatio

variables (V_a V_n V_a' : ℝ)
variable (distance : ℝ := 30)
variable (original_speed_Arun : ℝ := 5)
variable (time_Arun time_Anil time_Arun_new_speed : ℝ)

-- Conditions
theorem arun_speed_ratio :
  V_a = original_speed_Arun →
  time_Arun = distance / V_a →
  time_Anil = distance / V_n →
  time_Arun = time_Anil + 2 →
  time_Arun_new_speed = distance / V_a' →
  time_Arun_new_speed = time_Anil - 1 →
  V_a' / V_a = 2 := 
by
  intros h1 h2 h3 h4 h5 h6
  simp [h1] at *
  sorry

end SpeedRatio

end arun_speed_ratio_l711_71168


namespace Johann_oranges_l711_71144

-- Define the given conditions
def initial_oranges := 60
def eaten_oranges := 10
def half_remaining_oranges := (initial_oranges - eaten_oranges) / 2
def returned_oranges := 5

-- Define the statement to prove
theorem Johann_oranges :
  initial_oranges - eaten_oranges - half_remaining_oranges + returned_oranges = 30 := by
  sorry

end Johann_oranges_l711_71144


namespace work_completion_time_equal_l711_71150

/-- Define the individual work rates of a, b, c, and d --/
def work_rate_a : ℚ := 1 / 24
def work_rate_b : ℚ := 1 / 6
def work_rate_c : ℚ := 1 / 12
def work_rate_d : ℚ := 1 / 10

/-- Define the combined work rate when they work together --/
def combined_work_rate : ℚ := work_rate_a + work_rate_b + work_rate_c + work_rate_d

/-- Define total work as one unit divided by the combined work rate --/
def total_days_to_complete : ℚ := 1 / combined_work_rate

/-- Main theorem to prove: When a, b, c, and d work together, they complete the work in 120/47 days --/
theorem work_completion_time_equal : total_days_to_complete = 120 / 47 :=
by
  sorry

end work_completion_time_equal_l711_71150


namespace board_rook_placement_l711_71139

-- Define the color function for the board
def color (n i j : ℕ) : ℕ :=
  min (i + j - 1) (2 * n - i - j + 1)

-- Conditions: It is possible to place n rooks such that no two attack each other and 
-- no two rooks stand on cells of the same color
def non_attacking_rooks (n : ℕ) (rooks : Fin n → Fin n) : Prop :=
  ∀ i j : Fin n, i ≠ j → rooks i ≠ rooks j ∧ color n i.val (rooks i).val ≠ color n j.val (rooks j).val

-- Main theorem to be proven
theorem board_rook_placement (n : ℕ) :
  (∃ rooks : Fin n → Fin n, non_attacking_rooks n rooks) →
  n % 4 = 0 ∨ n % 4 = 1 :=
by
  intros h
  sorry

end board_rook_placement_l711_71139


namespace cleaning_cost_l711_71152

theorem cleaning_cost (num_cleanings : ℕ) (chemical_cost : ℕ) (monthly_cost : ℕ) (tip_percentage : ℚ) 
  (cleaning_sessions_per_month : num_cleanings = 30 / 3)
  (monthly_chemical_cost : chemical_cost = 2 * 200)
  (total_monthly_cost : monthly_cost = 2050)
  (cleaning_cost_with_tip : monthly_cost - chemical_cost =  num_cleanings * (1 + tip_percentage) * x) : 
  x = 150 := 
by
  sorry

end cleaning_cost_l711_71152


namespace peach_trees_count_l711_71189

theorem peach_trees_count : ∀ (almond_trees: ℕ), almond_trees = 300 → 2 * almond_trees - 30 = 570 :=
by
  intros
  sorry

end peach_trees_count_l711_71189


namespace max_initial_number_l711_71180

theorem max_initial_number (n : ℕ) : 
  (∃ (a b c d e : ℕ), 
    200 = n + a + b + c + d + e ∧ 
    ¬ (n % a = 0) ∧ 
    ¬ ((n + a) % b = 0) ∧ 
    ¬ ((n + a + b) % c = 0) ∧ 
    ¬ ((n + a + b + c) % d = 0) ∧ 
    ¬ ((n + a + b + c + d) % e = 0)) → 
  n ≤ 189 := 
sorry

end max_initial_number_l711_71180


namespace min_handshakes_l711_71174

theorem min_handshakes 
  (people : ℕ) 
  (handshakes_per_person : ℕ) 
  (total_people : people = 30) 
  (handshakes_rule : handshakes_per_person = 3) 
  (unique_handshakes : people * handshakes_per_person % 2 = 0) 
  (multiple_people : people > 0):
  (people * handshakes_per_person / 2) = 45 :=
by
  sorry

end min_handshakes_l711_71174


namespace cannot_determine_both_correct_l711_71185

-- Definitions
def total_students : ℕ := 40
def answered_q1_correctly : ℕ := 30
def did_not_take_test : ℕ := 10

-- Assertion that the number of students answering both questions correctly cannot be determined
theorem cannot_determine_both_correct (answered_q2_correctly : ℕ) :
  (∃ (both_correct : ℕ), both_correct ≤ answered_q1_correctly ∧ both_correct ≤ answered_q2_correctly)  ↔ answered_q2_correctly > 0 :=
by 
 sorry

end cannot_determine_both_correct_l711_71185


namespace interest_rate_A_l711_71106

-- Definitions for the conditions
def principal : ℝ := 1000
def rate_C : ℝ := 0.115
def time_period : ℝ := 3
def gain_B : ℝ := 45

-- Main theorem to prove
theorem interest_rate_A {R : ℝ} (h1 : gain_B = (principal * rate_C * time_period - principal * (R / 100) * time_period)) : R = 10 := 
by
  sorry

end interest_rate_A_l711_71106


namespace train_speed_l711_71109

theorem train_speed (length : ℕ) (time : ℕ) (h1 : length = 1600) (h2 : time = 40) : length / time = 40 := 
by
  -- use the given conditions here
  sorry

end train_speed_l711_71109


namespace original_number_l711_71142

theorem original_number (N m a b c : ℕ) (hN : N = 3306) 
  (h_eq : 3306 + m = 222 * (a + b + c)) 
  (hm_digits : m = 100 * a + 10 * b + c) 
  (h1 : a + b + c = 15) 
  (h2 : ∃ (a b c : ℕ), a + b + c = 15 ∧ 100 * a + 10 * b + c = 78): 
  100 * a + 10 * b + c = 753 := 
by sorry

end original_number_l711_71142


namespace depth_B_is_correct_l711_71108

-- Given: Diver A is at a depth of -55 meters.
def depth_A : ℤ := -55

-- Given: Diver B is 5 meters above diver A.
def offset : ℤ := 5

-- Prove: The depth of diver B
theorem depth_B_is_correct : (depth_A + offset) = -50 :=
by
  sorry

end depth_B_is_correct_l711_71108


namespace arithmetic_sequence_max_sum_l711_71128

theorem arithmetic_sequence_max_sum (a : ℕ → ℝ) (d : ℝ) (m : ℕ) (S : ℕ → ℝ):
  (∀ n, a n = a 1 + (n - 1) * d) → 
  3 * a 8 = 5 * a m → 
  a 1 > 0 →
  (∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * d)) →
  (∀ n, S n ≤ S 20) →
  m = 13 := 
by {
  -- State the corresponding solution steps leading to the proof.
  sorry
}

end arithmetic_sequence_max_sum_l711_71128


namespace largest_sum_of_distinct_factors_l711_71184

theorem largest_sum_of_distinct_factors (A B C : ℕ) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) (h_positive : A > 0 ∧ B > 0 ∧ C > 0) (h_product : A * B * C = 3003) :
  A + B + C ≤ 105 :=
sorry  -- Proof is not required, just the statement.

end largest_sum_of_distinct_factors_l711_71184


namespace non_congruent_parallelograms_l711_71159

def side_lengths_sum (a b : ℕ) : Prop :=
  a + b = 25

def is_congruent (a b : ℕ) (a' b' : ℕ) : Prop :=
  (a = a' ∧ b = b') ∨ (a = b' ∧ b = a')

def non_congruent_count (n : ℕ) : Prop :=
  ∀ (a b : ℕ), side_lengths_sum a b → 
  ∃! (m : ℕ), is_congruent a b m b

theorem non_congruent_parallelograms :
  ∃ (n : ℕ), non_congruent_count n ∧ n = 13 :=
sorry

end non_congruent_parallelograms_l711_71159


namespace least_area_of_triangles_l711_71102

-- Define the points A, B, C, D of the unit square
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (1, 1)
def D : ℝ × ℝ := (0, 1)

-- Define the function s(M, N) as the least area of the triangles having their vertices in the set {A, B, C, D, M, N}
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

noncomputable def s (M N : ℝ × ℝ) : ℝ :=
  min (min (min (min (min (triangle_area A B M) (triangle_area A B N)) (triangle_area A C M)) (triangle_area A C N)) (min (triangle_area A D M) (triangle_area A D N)))
    (min (min (min (triangle_area B C M) (triangle_area B C N)) (triangle_area B D M)) (min (triangle_area B D N) (min (triangle_area C D M) (triangle_area C D N))))

-- Define the statement to prove
theorem least_area_of_triangles (M N : ℝ × ℝ)
  (hM : M.1 > 0 ∧ M.1 < 1 ∧ M.2 > 0 ∧ M.2 < 1)
  (hN : N.1 > 0 ∧ N.1 < 1 ∧ N.2 > 0 ∧ N.2 < 1)
  (hMN : (M ≠ A ∨ N ≠ A) ∧ (M ≠ B ∨ N ≠ B) ∧ (M ≠ C ∨ N ≠ C) ∧ (M ≠ D ∨ N ≠ D))
  : s M N ≤ 1 / 8 := 
sorry

end least_area_of_triangles_l711_71102


namespace part_a_l711_71132

theorem part_a (x : ℝ) (hx : x > 0) :
  ∃ color : ℕ, ∃ p1 p2 : ℝ × ℝ, (p1 = p2 ∨ x = dist p1 p2) :=
sorry

end part_a_l711_71132


namespace multiple_of_sum_squares_l711_71126

theorem multiple_of_sum_squares (a b c : ℕ) (h1 : a < 2017) (h2 : b < 2017) (h3 : c < 2017) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a)
    (h7 : ∃ k1, a^3 - b^3 = k1 * 2017) (h8 : ∃ k2, b^3 - c^3 = k2 * 2017) (h9 : ∃ k3, c^3 - a^3 = k3 * 2017) :
    ∃ k, a^2 + b^2 + c^2 = k * (a + b + c) :=
by
  sorry

end multiple_of_sum_squares_l711_71126


namespace european_postcards_cost_l711_71170

def price_per_postcard (country : String) : ℝ :=
  if country = "Italy" ∨ country = "Germany" then 0.10
  else if country = "Canada" then 0.07
  else if country = "Mexico" then 0.08
  else 0.0

def num_postcards (decade : Nat) (country : String) : Nat :=
  if decade = 1950 then
    if country = "Italy" then 10
    else if country = "Germany" then 5
    else if country = "Canada" then 8
    else if country = "Mexico" then 12
    else 0
  else if decade = 1960 then
    if country = "Italy" then 16
    else if country = "Germany" then 12
    else if country = "Canada" then 10
    else if country = "Mexico" then 15
    else 0
  else if decade = 1970 then
    if country = "Italy" then 12
    else if country = "Germany" then 18
    else if country = "Canada" then 13
    else if country = "Mexico" then 9
    else 0
  else 0

def total_cost (country : String) : ℝ :=
  (price_per_postcard country) * (num_postcards 1950 country)
  + (price_per_postcard country) * (num_postcards 1960 country)
  + (price_per_postcard country) * (num_postcards 1970 country)

theorem european_postcards_cost : total_cost "Italy" + total_cost "Germany" = 7.30 := by
  sorry

end european_postcards_cost_l711_71170


namespace Joe_speed_first_part_l711_71146

theorem Joe_speed_first_part
  (dist1 dist2 : ℕ)
  (speed2 avg_speed total_distance total_time : ℕ)
  (h1 : dist1 = 180)
  (h2 : dist2 = 120)
  (h3 : speed2 = 40)
  (h4 : avg_speed = 50)
  (h5 : total_distance = dist1 + dist2)
  (h6 : total_distance = 300)
  (h7 : total_time = total_distance / avg_speed)
  (h8 : total_time = 6) :
  ∃ v : ℕ, (dist1 / v + dist2 / speed2 = total_time) ∧ v = 60 :=
by
  sorry

end Joe_speed_first_part_l711_71146


namespace evaluate_expression_l711_71156

theorem evaluate_expression (b : ℚ) (h : b = 4 / 3) :
  (6 * b ^ 2 - 17 * b + 8) * (3 * b - 4) = 0 :=
by 
  -- Proof goes here
  sorry

end evaluate_expression_l711_71156


namespace suff_but_not_necess_condition_l711_71157

theorem suff_but_not_necess_condition (a b : ℝ) (h1 : a < 0) (h2 : -1 < b ∧ b < 0) : a + a * b < 0 :=
  sorry

end suff_but_not_necess_condition_l711_71157


namespace cube_of_odd_number_minus_itself_divisible_by_24_l711_71182

theorem cube_of_odd_number_minus_itself_divisible_by_24 (n : ℤ) : 
  24 ∣ ((2 * n + 1) ^ 3 - (2 * n + 1)) :=
by
  sorry

end cube_of_odd_number_minus_itself_divisible_by_24_l711_71182


namespace oliver_more_money_l711_71145

noncomputable def totalOliver : ℕ := 10 * 20 + 3 * 5
noncomputable def totalWilliam : ℕ := 15 * 10 + 4 * 5

theorem oliver_more_money : totalOliver - totalWilliam = 45 := by
  sorry

end oliver_more_money_l711_71145


namespace new_three_digit_number_l711_71133

theorem new_three_digit_number (t u : ℕ) (h1 : t < 10) (h2 : u < 10) :
  let original := 10 * t + u
  let new_number := (original * 10) + 2
  new_number = 100 * t + 10 * u + 2 :=
by
  sorry

end new_three_digit_number_l711_71133


namespace digit_divisible_by_3_l711_71118

theorem digit_divisible_by_3 (d : ℕ) (h : d < 10) : (15780 + d) % 3 = 0 ↔ d = 0 ∨ d = 3 ∨ d = 6 ∨ d = 9 := by
  sorry

end digit_divisible_by_3_l711_71118


namespace frame_width_proof_l711_71199

noncomputable section

-- Define the given conditions
def perimeter_square_opening := 60 -- cm
def perimeter_entire_frame := 180 -- cm

-- Define what we need to prove: the width of the frame
def width_of_frame : ℕ := 5 -- cm

-- Define a function to calculate the side length of a square
def side_length_of_square (perimeter : ℕ) : ℕ :=
  perimeter / 4

-- Define the side length of the square opening
def side_length_opening := side_length_of_square perimeter_square_opening

-- Use the given conditions to calculate the frame's width
-- Given formulas in the solution steps:
--  2 * (3 * side_length + 4 * d) + 2 * (side_length + 2 * d) = perimeter_entire_frame
theorem frame_width_proof (d : ℕ) (perim_square perim_frame : ℕ) :
  perim_square = perimeter_square_opening →
  perim_frame = perimeter_entire_frame →
  2 * (3 * side_length_of_square perim_square + 4 * d) 
  + 2 * (side_length_of_square perim_square + 2 * d) 
  = perim_frame →
  d = width_of_frame := 
by 
  intros h1 h2 h3
  -- The proof will go here
  sorry

end frame_width_proof_l711_71199


namespace cartons_being_considered_l711_71158

-- Definitions based on conditions
def packs_per_box : ℕ := 10
def boxes_per_carton : ℕ := 12
def price_per_pack : ℕ := 1
def total_cost : ℕ := 1440

-- Calculate total cost per carton
def cost_per_carton : ℕ := boxes_per_carton * packs_per_box * price_per_pack

-- Formulate the main theorem
theorem cartons_being_considered : (total_cost / cost_per_carton) = 12 :=
by
  -- The relevant steps would go here, but we're only providing the statement
  sorry

end cartons_being_considered_l711_71158


namespace perpendicular_lines_l711_71183

theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, 2 * x - a * y - 1 = 0 → x + 2 * y = 0) →
  (a = 1) :=
by
  sorry

end perpendicular_lines_l711_71183


namespace greatest_consecutive_integers_sum_36_l711_71173

theorem greatest_consecutive_integers_sum_36 : ∀ (x : ℤ), (x + (x + 1) + (x + 2) = 36) → (x + 2 = 13) :=
by
  sorry

end greatest_consecutive_integers_sum_36_l711_71173


namespace intersection_of_complements_l711_71191

theorem intersection_of_complements 
  (U : Set ℕ) (A B : Set ℕ)
  (hU : U = { x | x ≤ 5 }) 
  (hA : A = {1, 2, 3}) 
  (hB : B = {1, 4}) :
  ((U \ A) ∩ (U \ B)) = {0, 5} :=
by sorry

end intersection_of_complements_l711_71191


namespace length_of_second_platform_l711_71171

/-- 
Let L be the length of the second platform.
A train crosses a platform of 100 m in 15 sec.
The same train crosses another platform in 20 sec.
The length of the train is 350 m.
Prove that the length of the second platform is 250 meters.
-/
theorem length_of_second_platform (L : ℕ) (train_length : ℕ) (platform1_length : ℕ) (time1 : ℕ) (time2 : ℕ):
  train_length = 350 → platform1_length = 100 → time1 = 15 → time2 = 20 → L = 250 :=
by
  sorry

end length_of_second_platform_l711_71171


namespace quadratic_roots_product_sum_l711_71198

theorem quadratic_roots_product_sum :
  (∀ d e : ℝ, 3 * d^2 + 4 * d - 7 = 0 ∧ 3 * e^2 + 4 * e - 7 = 0 →
   (d + 1) * (e + 1) = - 8 / 3) := by
sorry

end quadratic_roots_product_sum_l711_71198


namespace geometric_sum_l711_71194

open BigOperators

noncomputable def geom_sequence (a q : ℚ) (n : ℕ) : ℚ := a * q ^ n

noncomputable def sum_geom_sequence (a q : ℚ) (n : ℕ) : ℚ := 
  if q = 1 then a * n
  else a * (1 - q ^ (n + 1)) / (1 - q)

theorem geometric_sum (a q : ℚ) (h_a : a = 1) (h_S3 : sum_geom_sequence a q 2 = 3 / 4) :
  sum_geom_sequence a q 3 = 5 / 8 :=
sorry

end geometric_sum_l711_71194


namespace molecular_weight_X_l711_71119

theorem molecular_weight_X (Ba_weight : ℝ) (total_molecular_weight : ℝ) (X_weight : ℝ) 
  (h1 : Ba_weight = 137) 
  (h2 : total_molecular_weight = 171) 
  (h3 : total_molecular_weight - Ba_weight * 1 = 2 * X_weight) : 
  X_weight = 17 :=
by
  sorry

end molecular_weight_X_l711_71119


namespace seats_on_each_bus_l711_71155

-- Define the given conditions
def totalStudents : ℕ := 45
def totalBuses : ℕ := 5

-- Define what we need to prove - 
-- that the number of seats on each bus is 9
def seatsPerBus (students : ℕ) (buses : ℕ) : ℕ := students / buses

theorem seats_on_each_bus : seatsPerBus totalStudents totalBuses = 9 := by
  -- Proof to be filled in later
  sorry

end seats_on_each_bus_l711_71155


namespace division_value_l711_71138

theorem division_value (a b c : ℝ) 
  (h1 : a / b = 5 / 3) 
  (h2 : b / c = 7 / 2) : 
  c / a = 6 / 35 := 
by
  sorry

end division_value_l711_71138


namespace calculate_expression_solve_quadratic_l711_71165

-- Problem 1
theorem calculate_expression (x : ℝ) (hx : x > 0) :
  (2 / 3) * Real.sqrt (9 * x) + 6 * Real.sqrt (x / 4) - x * Real.sqrt (1 / x) = 4 * Real.sqrt x :=
sorry

-- Problem 2
theorem solve_quadratic (x : ℝ) (h : x^2 - 4 * x + 1 = 0) :
  x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 :=
sorry

end calculate_expression_solve_quadratic_l711_71165


namespace josh_found_marbles_l711_71153

theorem josh_found_marbles :
  ∃ (F : ℕ), (F + 14 = 23) → (F = 9) :=
by
  existsi 9
  intro h
  linarith

end josh_found_marbles_l711_71153


namespace lines_intersect_at_l711_71107

noncomputable def line1 (x : ℚ) : ℚ := (-2 / 3) * x + 2
noncomputable def line2 (x : ℚ) : ℚ := -2 * x + (3 / 2)

theorem lines_intersect_at :
  ∃ (x y : ℚ), line1 x = y ∧ line2 x = y ∧ x = (3 / 8) ∧ y = (7 / 4) :=
sorry

end lines_intersect_at_l711_71107


namespace common_ratio_is_63_98_l711_71160

/-- Define the terms of the geometric series -/
def term (n : Nat) : ℚ := 
  match n with
  | 0 => 4 / 7
  | 1 => 18 / 49
  | 2 => 162 / 343
  | _ => sorry  -- For simplicity, we can define more terms if needed, but it's irrelevant for our proof

/-- Define the common ratio of the geometric series -/
def common_ratio (a b : ℚ) : ℚ := b / a

/-- The problem states that the common ratio of first two terms of the given series is equal to 63/98 -/
theorem common_ratio_is_63_98 : common_ratio (term 0) (term 1) = 63 / 98 :=
by
  -- leave the proof as sorry for now
  sorry

end common_ratio_is_63_98_l711_71160


namespace cone_volume_l711_71136

theorem cone_volume (S : ℝ) (h_S : S = 12 * Real.pi) (h_lateral : ∃ r : ℝ, S = 3 * Real.pi * r^2) :
    ∃ V : ℝ, V = (8 * Real.sqrt 3 * Real.pi / 3) :=
by
  sorry

end cone_volume_l711_71136


namespace rahim_books_from_first_shop_l711_71196

variable (books_first_shop_cost : ℕ)
variable (second_shop_books : ℕ)
variable (second_shop_books_cost : ℕ)
variable (average_price_per_book : ℕ)
variable (number_of_books_first_shop : ℕ)

theorem rahim_books_from_first_shop
  (h₁ : books_first_shop_cost = 581)
  (h₂ : second_shop_books = 20)
  (h₃ : second_shop_books_cost = 594)
  (h₄ : average_price_per_book = 25)
  (h₅ : (books_first_shop_cost + second_shop_books_cost) = (number_of_books_first_shop + second_shop_books) * average_price_per_book) :
  number_of_books_first_shop = 27 :=
sorry

end rahim_books_from_first_shop_l711_71196


namespace maura_seashells_l711_71114

theorem maura_seashells (original_seashells given_seashells remaining_seashells : ℕ)
  (h1 : original_seashells = 75) 
  (h2 : remaining_seashells = 57) 
  (h3 : given_seashells = original_seashells - remaining_seashells) :
  given_seashells = 18 := by
  -- Lean will use 'sorry' as a placeholder for the actual proof
  sorry

end maura_seashells_l711_71114


namespace number_of_children_l711_71179

-- Definitions of the conditions
def crayons_per_child : ℕ := 8
def total_crayons : ℕ := 56

-- Statement of the problem
theorem number_of_children : total_crayons / crayons_per_child = 7 := by
  sorry

end number_of_children_l711_71179


namespace exists_constant_C_inequality_for_difference_l711_71178

theorem exists_constant_C (a : ℕ → ℝ) (C : ℝ) (hC : 0 < C) :
  (a 1 = 1) →
  (a 2 = 8) →
  (∀ n : ℕ, 2 ≤ n → a (n + 1) = a (n - 1) + (4 / n) * a n) →
  (∀ n : ℕ, a n ≤ C * n^2) := sorry

theorem inequality_for_difference (a : ℕ → ℝ) :
  (a 1 = 1) →
  (a 2 = 8) →
  (∀ n : ℕ, 2 ≤ n → a (n + 1) = a (n - 1) + (4 / n) * a n) →
  (∀ n : ℕ, a (n + 1) - a n ≤ 4 * n + 3) := sorry

end exists_constant_C_inequality_for_difference_l711_71178


namespace problem1_correct_problem2_correct_l711_71151

noncomputable def problem1 := 5 + (-6) + 3 - 8 - (-4)
noncomputable def problem2 := -2^2 - 3 * (-1)^3 - (-1) / (-1 / 2)^2

theorem problem1_correct : problem1 = -2 := by
  rw [problem1]
  sorry

theorem problem2_correct : problem2 = 3 := by
  rw [problem2]
  sorry

end problem1_correct_problem2_correct_l711_71151


namespace num_candidates_appeared_each_state_l711_71101

-- Definitions
def candidates_appear : ℕ := 8000
def sel_pct_A : ℚ := 0.06
def sel_pct_B : ℚ := 0.07
def additional_selections_B : ℕ := 80

-- Proof Problem Statement
theorem num_candidates_appeared_each_state (x : ℕ) 
  (h1 : x = candidates_appear) 
  (h2 : sel_pct_A * ↑x = 0.06 * ↑x) 
  (h3 : sel_pct_B * ↑x = 0.07 * ↑x) 
  (h4 : sel_pct_B * ↑x = sel_pct_A * ↑x + additional_selections_B) : 
  x = candidates_appear := sorry

end num_candidates_appeared_each_state_l711_71101


namespace simplest_radical_l711_71112

theorem simplest_radical (r1 r2 r3 r4 : ℝ) 
  (h1 : r1 = Real.sqrt 3) 
  (h2 : r2 = Real.sqrt 4)
  (h3 : r3 = Real.sqrt 8)
  (h4 : r4 = Real.sqrt (1 / 2)) : r1 = Real.sqrt 3 :=
  by sorry

end simplest_radical_l711_71112


namespace variance_male_greater_than_female_l711_71130

noncomputable def male_scores : List ℝ := [87, 95, 89, 93, 91]
noncomputable def female_scores : List ℝ := [89, 94, 94, 89, 94]

-- Function to calculate the variance of scores
noncomputable def variance (scores : List ℝ) : ℝ :=
  let n := scores.length
  let mean := scores.sum / n
  (scores.map (λ x => (x - mean) ^ 2)).sum / n

-- We assert the problem statement
theorem variance_male_greater_than_female :
  variance male_scores > variance female_scores :=
by
  sorry

end variance_male_greater_than_female_l711_71130


namespace suggestions_difference_l711_71120

def mashed_potatoes_suggestions : ℕ := 408
def pasta_suggestions : ℕ := 305
def bacon_suggestions : ℕ := 137
def grilled_vegetables_suggestions : ℕ := 213
def sushi_suggestions : ℕ := 137

theorem suggestions_difference :
  let highest := mashed_potatoes_suggestions
  let lowest := bacon_suggestions
  highest - lowest = 271 :=
by
  sorry

end suggestions_difference_l711_71120


namespace tangent_line_through_P_line_through_P_chord_length_8_l711_71190

open Set

def circle (x y : ℝ) : Prop := x^2 + y^2 = 25

def point_P : ℝ × ℝ := (3, 4)

def tangent_line (x y : ℝ) : Prop := 3 * x + 4 * y - 25 = 0

def line_m_case1 (x : ℝ) : Prop := x = 3

def line_m_case2 (x y : ℝ) : Prop := 7 * x - 24 * y + 75 = 0

theorem tangent_line_through_P :
  tangent_line point_P.1 point_P.2 :=
sorry

theorem line_through_P_chord_length_8 :
  (∀ x y, circle x y → line_m_case1 x ∨ line_m_case2 x y) :=
sorry

end tangent_line_through_P_line_through_P_chord_length_8_l711_71190


namespace three_equal_of_four_l711_71123

theorem three_equal_of_four (a b c d : ℕ) 
  (h1 : (a + b)^2 ∣ c * d) 
  (h2 : (a + c)^2 ∣ b * d) 
  (h3 : (a + d)^2 ∣ b * c) 
  (h4 : (b + c)^2 ∣ a * d) 
  (h5 : (b + d)^2 ∣ a * c) 
  (h6 : (c + d)^2 ∣ a * b) : 
  (a = b ∧ b = c) ∨ (a = b ∧ b = d) ∨ (a = c ∧ c = d) ∨ (b = c ∧ c = d) := 
sorry

end three_equal_of_four_l711_71123


namespace factorize_expr_l711_71163

-- Define the variables a and b as elements of an arbitrary ring
variables {R : Type*} [CommRing R] (a b : R)

-- Prove the factorization identity
theorem factorize_expr : a^2 * b - b = b * (a + 1) * (a - 1) :=
by
  sorry

end factorize_expr_l711_71163


namespace range_f_subset_interval_l711_71111

-- Define the function f on real numbers
def f : ℝ → ℝ := sorry

-- The given condition for all real numbers x and y such that x > y
axiom condition (x y : ℝ) (h : x > y) : (f x)^2 ≤ f y

-- The main theorem that needs to be proven
theorem range_f_subset_interval : ∀ x, 0 ≤ f x ∧ f x ≤ 1 := 
by
  intro x
  apply And.intro
  -- Proof for 0 ≤ f x
  sorry
  -- Proof for f x ≤ 1
  sorry

end range_f_subset_interval_l711_71111


namespace contractor_fine_amount_l711_71140

def total_days := 30
def daily_earning := 25
def total_earnings := 360
def days_absent := 12
def days_worked := total_days - days_absent
def fine_per_absent_day (x : ℝ) : Prop :=
  (daily_earning * days_worked) - (x * days_absent) = total_earnings

theorem contractor_fine_amount : ∃ x : ℝ, fine_per_absent_day x := by
  use 7.5
  sorry

end contractor_fine_amount_l711_71140


namespace evaluate_expression_l711_71188

theorem evaluate_expression (α : ℝ) (h : Real.tan α = 3) :
  (2 * Real.sin (2 * α) - 3 * Real.cos (2 * α)) / (4 * Real.sin (2 * α) + 5 * Real.cos (2 * α)) = -9 / 4 :=
sorry

end evaluate_expression_l711_71188


namespace frac_difference_l711_71103

theorem frac_difference (m n : ℝ) (h : m^2 - n^2 = m * n) : (n / m) - (m / n) = -1 :=
sorry

end frac_difference_l711_71103


namespace shorter_piece_length_l711_71181

noncomputable def total_length : ℝ := 140
noncomputable def ratio : ℝ := 2 / 5

theorem shorter_piece_length (x : ℝ) (y : ℝ) (h1 : x + y = total_length) (h2 : x = ratio * y) : x = 40 :=
by
  sorry

end shorter_piece_length_l711_71181


namespace n_five_minus_n_divisible_by_30_l711_71110

theorem n_five_minus_n_divisible_by_30 (n : ℤ) : 30 ∣ (n^5 - n) :=
sorry

end n_five_minus_n_divisible_by_30_l711_71110


namespace solve_cubic_equation_l711_71187

theorem solve_cubic_equation (x : ℝ) (h : 4 * x^(1/3) - 2 * (x / x^(2/3)) = 7 + x^(1/3)) : x = 343 := by
  sorry

end solve_cubic_equation_l711_71187


namespace john_streams_hours_per_day_l711_71197

theorem john_streams_hours_per_day :
  (∃ h : ℕ, (7 - 3) * h * 10 = 160) → 
  (∃ h : ℕ, h = 4) :=
sorry

end john_streams_hours_per_day_l711_71197


namespace negate_exactly_one_even_l711_71186

variable (a b c : ℕ)

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

theorem negate_exactly_one_even :
  ¬(is_even a ∧ is_odd b ∧ is_odd c ∨ is_odd a ∧ is_even b ∧ is_odd c ∨ is_odd a ∧ is_odd b ∧ is_even c) ↔
  (is_even a ∧ is_even b ∨ is_even a ∧ is_even c ∨ is_even b ∧ is_even c ∨ is_odd a ∧ is_odd b ∧ is_odd c) := sorry

end negate_exactly_one_even_l711_71186


namespace trigonometric_identity_solution_l711_71147

theorem trigonometric_identity_solution 
  (alpha beta : ℝ)
  (h1 : π / 4 < alpha)
  (h2 : alpha < 3 * π / 4)
  (h3 : 0 < beta)
  (h4 : beta < π / 4)
  (h5 : Real.cos (π / 4 + alpha) = -4 / 5)
  (h6 : Real.sin (3 * π / 4 + beta) = 12 / 13) :
  (Real.sin (alpha + beta) = 63 / 65) ∧
  (Real.cos (alpha - beta) = -33 / 65) :=
by
  sorry

end trigonometric_identity_solution_l711_71147


namespace rect_garden_width_l711_71172

theorem rect_garden_width (w l : ℝ) (h1 : l = 3 * w) (h2 : l * w = 768) : w = 16 := by
  sorry

end rect_garden_width_l711_71172
