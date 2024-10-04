import Mathlib

namespace remainder_7_pow_700_div_100_l282_282902

theorem remainder_7_pow_700_div_100 : (7 ^ 700) % 100 = 1 := 
  by sorry

end remainder_7_pow_700_div_100_l282_282902


namespace x_y_result_l282_282232

noncomputable def x_y_value (x y : ℝ) : ℝ := x + y

theorem x_y_result (x y : ℝ) 
  (h1 : x + Real.cos y = 3009) 
  (h2 : x + 3009 * Real.sin y = 3010)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi) : 
  x_y_value x y = 3009 + Real.pi / 2 :=
by
  sorry

end x_y_result_l282_282232


namespace hotel_cost_l282_282380

/--
Let the total cost of the hotel be denoted as x dollars.
Initially, the cost for each of the original four colleagues is x / 4.
After three more colleagues joined, the cost per person becomes x / 7.
Given that the amount paid by each of the original four decreased by 15,
prove that the total cost of the hotel is 140 dollars.
-/
theorem hotel_cost (x : ℕ) (h : x / 4 - 15 = x / 7) : x = 140 := 
by
  sorry

end hotel_cost_l282_282380


namespace find_m_given_root_exists_l282_282955

theorem find_m_given_root_exists (x m : ℝ) (h : ∃ x, x ≠ 2 ∧ (x / (x - 2) - 2 = m / (x - 2))) : m = 2 :=
by
  sorry

end find_m_given_root_exists_l282_282955


namespace arthur_bakes_muffins_l282_282202

-- Definitions of the conditions
def james_muffins : ℚ := 9.58333333299999
def multiplier : ℚ := 12.0

-- Statement of the problem
theorem arthur_bakes_muffins : 
  abs (multiplier * james_muffins - 115) < 1 :=
by
  sorry

end arthur_bakes_muffins_l282_282202


namespace root_iff_coeff_sum_zero_l282_282599

theorem root_iff_coeff_sum_zero (a b c : ℝ) :
    (a * 1^2 + b * 1 + c = 0) ↔ (a + b + c = 0) := sorry

end root_iff_coeff_sum_zero_l282_282599


namespace min_rows_required_to_seat_students_l282_282467

-- Definitions based on the conditions
def seats_per_row : ℕ := 168
def total_students : ℕ := 2016
def max_students_per_school : ℕ := 40

def min_number_of_rows : ℕ :=
  -- Given that the minimum number of rows required to seat all students following the conditions is 15
  15

-- Lean statement expressing the proof problem
theorem min_rows_required_to_seat_students :
  ∃ rows : ℕ, rows = min_number_of_rows ∧
  (∀ school_sizes : List ℕ, (∀ size ∈ school_sizes, size ≤ max_students_per_school)
    → (List.sum school_sizes = total_students)
    → ∀ school_arrangement : List (List ℕ), 
        (∀ row_sizes ∈ school_arrangement, List.sum row_sizes ≤ seats_per_row) 
        → List.length school_arrangement ≤ rows) :=
sorry

end min_rows_required_to_seat_students_l282_282467


namespace final_acid_concentration_l282_282102

def volume1 : ℝ := 2
def concentration1 : ℝ := 0.40
def volume2 : ℝ := 3
def concentration2 : ℝ := 0.60

theorem final_acid_concentration :
  ((concentration1 * volume1 + concentration2 * volume2) / (volume1 + volume2)) = 0.52 :=
by
  sorry

end final_acid_concentration_l282_282102


namespace train_crosses_pole_in_15_seconds_l282_282652

theorem train_crosses_pole_in_15_seconds
    (train_speed : ℝ) (train_length_meters : ℝ) (time_seconds : ℝ) : 
    train_speed = 300 →
    train_length_meters = 1250 →
    time_seconds = 15 :=
by
  sorry

end train_crosses_pole_in_15_seconds_l282_282652


namespace number_of_boys_selected_l282_282025

theorem number_of_boys_selected {boys girls selections : ℕ} 
  (h_boys : boys = 11) (h_girls : girls = 10) (h_selections : selections = 6600) : 
  ∃ (k : ℕ), k = 2 :=
sorry

end number_of_boys_selected_l282_282025


namespace find_X_l282_282568

def tax_problem (X I T : ℝ) (income : ℝ) (total_tax : ℝ) :=
  (income = 56000) ∧ (total_tax = 8000) ∧ (T = 0.12 * X + 0.20 * (I - X))

theorem find_X :
  ∃ X : ℝ, ∀ I T : ℝ, tax_problem X I T 56000 8000 → X = 40000 := 
  by
    sorry

end find_X_l282_282568


namespace prime_a_b_l282_282831

theorem prime_a_b (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (h : a^11 + b = 2089) : 49 * b - a = 2007 :=
sorry

end prime_a_b_l282_282831


namespace fraction_one_third_between_l282_282013

theorem fraction_one_third_between (a b : ℚ) (h1 : a = 1/6) (h2 : b = 1/4) : (1/3 * (b - a) + a = 7/36) :=
by
  -- Conditions
  have ha : a = 1/6 := h1
  have hb : b = 1/4 := h2
  -- Start proof
  sorry

end fraction_one_third_between_l282_282013


namespace alex_jamie_casey_probability_l282_282923

-- Probability definitions and conditions
def alex_win_prob := 1/3
def casey_win_prob := 1/6
def jamie_win_prob := 1/2

def total_rounds := 8
def alex_wins := 4
def jamie_wins := 3
def casey_wins := 1

-- The probability computation
theorem alex_jamie_casey_probability : 
  alex_win_prob ^ alex_wins * jamie_win_prob ^ jamie_wins * casey_win_prob ^ casey_wins * (Nat.choose total_rounds (alex_wins + jamie_wins + casey_wins)) = 35 / 486 := 
sorry

end alex_jamie_casey_probability_l282_282923


namespace triangle_cos_Z_l282_282846

theorem triangle_cos_Z (X Y Z : ℝ) (hXZ : X + Y + Z = π) 
  (sinX : Real.sin X = 4 / 5) (cosY : Real.cos Y = 3 / 5) : 
  Real.cos Z = 7 / 25 := 
sorry

end triangle_cos_Z_l282_282846


namespace value_of_a_l282_282248

theorem value_of_a (a x y : ℤ) (h1 : x = 2) (h2 : y = 1) (h3 : a * x - 3 * y = 1) : a = 2 := by
  sorry

end value_of_a_l282_282248


namespace probability_A_or_not_B_l282_282706

open ProbabilityTheory

-- Definitions based on conditions in a)
def event_A (d : ℕ) : Prop := d ≤ 3
def event_B (d : ℕ) : Prop := d < 5

-- The main proof statement
theorem probability_A_or_not_B (d : ℕ) (uniform_die : ∀ d, d ∈ Finset.range 6 → Prob) : 
  (Prob (λ d, event_A d ∨ ¬ (event_B d))) = 5/6 :=
sorry

end probability_A_or_not_B_l282_282706


namespace calc_expression_l282_282268

theorem calc_expression (x y z : ℚ) (h1 : x = 1 / 3) (h2 : y = 2 / 3) (h3 : z = x * y) :
  3 * x^2 * y^5 * z^3 = 768 / 1594323 :=
by
  sorry

end calc_expression_l282_282268


namespace shorter_piece_length_l282_282188

theorem shorter_piece_length (P : ℝ) (Q : ℝ) (h1 : P + Q = 68) (h2 : Q = P + 12) : P = 28 := 
by
  sorry

end shorter_piece_length_l282_282188


namespace sum_of_three_numbers_l282_282636

variable {a b c : ℝ}

theorem sum_of_three_numbers :
  a^2 + b^2 + c^2 = 99 ∧ ab + bc + ca = 131 → a + b + c = 19 :=
by
  sorry

end sum_of_three_numbers_l282_282636


namespace polygon_sides_sum_l282_282366

theorem polygon_sides_sum :
  let triangle_sides := 3
  let square_sides := 4
  let pentagon_sides := 5
  let hexagon_sides := 6
  let heptagon_sides := 7
  let octagon_sides := 8
  let nonagon_sides := 9
  -- The sides of shapes that are adjacent on one side each (triangle and nonagon)
  let adjacent_triangle_nonagon := triangle_sides + nonagon_sides - 2
  -- The sides of the intermediate shapes that are each adjacent on two sides
  let adjacent_other_shapes := square_sides + pentagon_sides + hexagon_sides + heptagon_sides + octagon_sides - 5 * 2
  -- Summing up all the sides exposed to the outside
  adjacent_triangle_nonagon + adjacent_other_shapes = 30 := by
sorry

end polygon_sides_sum_l282_282366


namespace log_sum_exp_log_sub_l282_282905

theorem log_sum : Real.log 2 / Real.log 10 + Real.log 5 / Real.log 10 = 1 := 
by sorry

theorem exp_log_sub : Real.exp (Real.log 3 / Real.log 2 * Real.log 2) - Real.exp (Real.log 8 / 3) = 1 := 
by sorry

end log_sum_exp_log_sub_l282_282905


namespace abs_diff_one_l282_282829

theorem abs_diff_one (a b : ℤ) (h : |a| + |b| = 1) : |a - b| = 1 := sorry

end abs_diff_one_l282_282829


namespace zoe_total_songs_l282_282485

def total_songs (country_albums pop_albums songs_per_country_album songs_per_pop_album : ℕ) : ℕ :=
  country_albums * songs_per_country_album + pop_albums * songs_per_pop_album

theorem zoe_total_songs :
  total_songs 4 7 5 6 = 62 :=
by
  sorry

end zoe_total_songs_l282_282485


namespace crit_value_expr_l282_282377

theorem crit_value_expr : 
  ∃ x : ℝ, -4 < x ∧ x < 1 ∧ (x^2 - 2*x + 2) / (2*x - 2) = -1 :=
sorry

end crit_value_expr_l282_282377


namespace sufficient_but_not_necessary_condition_l282_282906

theorem sufficient_but_not_necessary_condition
  (p q r : Prop)
  (h_p_sufficient_q : p → q)
  (h_r_necessary_q : q → r)
  (h_p_not_necessary_q : ¬ (q → p))
  (h_r_not_sufficient_q : ¬ (r → q)) :
  (p → r) ∧ ¬ (r → p) :=
by
  sorry

end sufficient_but_not_necessary_condition_l282_282906


namespace temperature_fraction_l282_282257

def current_temperature : ℤ := 84
def temperature_decrease : ℤ := 21

theorem temperature_fraction :
  (current_temperature - temperature_decrease) = (3 * current_temperature / 4) := 
by
  sorry

end temperature_fraction_l282_282257


namespace sequence_arith_or_geom_l282_282959

def sequence_nature (a S : ℕ → ℝ) : Prop :=
  ∀ n, 4 * S n = (a n + 1) ^ 2

theorem sequence_arith_or_geom {a : ℕ → ℝ} {S : ℕ → ℝ} (h : sequence_nature a S) (h₁ : a 1 = 1) :
  (∃ d, ∀ n, a (n + 1) = a n + d) ∨ (∃ r, ∀ n, a (n + 1) = a n * r) :=
sorry

end sequence_arith_or_geom_l282_282959


namespace modulus_of_z_l282_282223

open Complex

theorem modulus_of_z (z : ℂ) (h : z * ⟨0, 1⟩ = ⟨2, 1⟩) : abs z = Real.sqrt 5 :=
by
  sorry

end modulus_of_z_l282_282223


namespace find_some_number_l282_282095

theorem find_some_number (x some_number : ℝ) (h1 : (27 / 4) * x - some_number = 3 * x + 27) (h2 : x = 12) :
  some_number = 18 :=
by
  sorry

end find_some_number_l282_282095


namespace volunteer_recommendations_l282_282101

def num_recommendations (boys girls : ℕ) (total_choices chosen : ℕ) : ℕ :=
  let total_combinations := Nat.choose total_choices chosen
  let invalid_combinations := Nat.choose boys chosen
  total_combinations - invalid_combinations

theorem volunteer_recommendations : num_recommendations 4 3 7 4 = 34 := by
  sorry

end volunteer_recommendations_l282_282101


namespace find_f_of_three_l282_282544

variable {f : ℝ → ℝ}

theorem find_f_of_three (h : ∀ x : ℝ, f (1 - 2 * x) = x^2 + x) : f 3 = 0 :=
by
  sorry

end find_f_of_three_l282_282544


namespace sum_of_five_primes_is_145_l282_282311

-- Condition: common difference is 12
def common_difference : ℕ := 12

-- Five prime numbers forming an arithmetic sequence with the given common difference
def a1 : ℕ := 5
def a2 : ℕ := a1 + common_difference
def a3 : ℕ := a2 + common_difference
def a4 : ℕ := a3 + common_difference
def a5 : ℕ := a4 + common_difference

-- The sum of the arithmetic sequence
def sum_of_primes : ℕ := a1 + a2 + a3 + a4 + a5

-- Prove that the sum of these five prime numbers is 145
theorem sum_of_five_primes_is_145 : sum_of_primes = 145 :=
by
  -- Proof goes here
  sorry

end sum_of_five_primes_is_145_l282_282311


namespace partition_equation_solution_l282_282055

def partition (n : ℕ) : ℕ := sorry -- defining the partition function

theorem partition_equation_solution (n : ℕ) (h : partition n + partition (n + 4) = partition (n + 2) + partition (n + 3)) :
  n = 1 ∨ n = 3 ∨ n = 5 :=
sorry

end partition_equation_solution_l282_282055


namespace probability_of_hitting_target_at_least_once_l282_282103

-- Define the constant probability of hitting the target in a single shot
def p_hit : ℚ := 2 / 3

-- Define the probability of missing the target in a single shot
def p_miss := 1 - p_hit

-- Define the probability of missing the target in all 3 shots
def p_miss_all_3 := p_miss ^ 3

-- Define the probability of hitting the target at least once in 3 shots
def p_hit_at_least_once := 1 - p_miss_all_3

-- Provide the theorem stating the solution
theorem probability_of_hitting_target_at_least_once :
  p_hit_at_least_once = 26 / 27 :=
by
  -- sorry is used to indicate the theorem needs to be proved
  sorry

end probability_of_hitting_target_at_least_once_l282_282103


namespace count_valid_numbers_correct_l282_282086

def odd_digits : Finset ℕ := {1, 3, 5, 7, 9}

def count_valid_numbers : ℕ :=
  let all_numbers := 5^5 in
  let invalid_numbers := 5 * 4^4 in
  all_numbers - invalid_numbers

theorem count_valid_numbers_correct : count_valid_numbers = 1845 := by
  sorry

end count_valid_numbers_correct_l282_282086


namespace machine_A_sprockets_per_hour_l282_282038

theorem machine_A_sprockets_per_hour :
  ∃ (A : ℝ), 
    (∃ (G : ℝ), 
      (G = 1.10 * A) ∧ 
      (∃ (T : ℝ), 
        (660 = A * (T + 10)) ∧ 
        (660 = G * T) 
      )
    ) ∧ 
    (A = 6) :=
by
  -- Conditions and variables will be introduced here...
  -- Proof can be implemented here
  sorry

end machine_A_sprockets_per_hour_l282_282038


namespace octal_to_base12_conversion_l282_282519

-- Define the computation functions required
def octalToDecimal (n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  d2 * 64 + d1 * 8 + d0

def decimalToBase12 (n : ℕ) : List ℕ :=
  let d0 := n % 12
  let n1 := n / 12
  let d1 := n1 % 12
  let n2 := n1 / 12
  let d2 := n2 % 12
  [d2, d1, d0]

-- The main theorem that combines both conversions
theorem octal_to_base12_conversion :
  decimalToBase12 (octalToDecimal 563) = [2, 6, 11] :=
sorry

end octal_to_base12_conversion_l282_282519


namespace work_completion_time_l282_282337

theorem work_completion_time :
  let a_rate := (1 : ℚ) / 11
      b_rate := (1 : ℚ) / 45
      c_rate := (1 : ℚ) / 55
      ab_rate := a_rate + b_rate
      ac_rate := a_rate + c_rate
      two_day_work := ab_rate + ac_rate
      work_done_per_two_days := two_day_work
      total_two_day_cycles := (1 : ℚ) / work_done_per_two_days
  in total_two_day_cycles ≤ 5 ∧ 2 * 5 = 10 :=
by
  let a_rate := (1 : ℚ) / 11
  let b_rate := (1 : ℚ) / 45
  let c_rate := (1 : ℚ) / 55
  let ab_rate := a_rate + b_rate
  let ac_rate := a_rate + c_rate
  let two_day_work := ab_rate + ac_rate
  let work_done_per_two_days := two_day_work
  let total_two_day_cycles := (1 : ℚ) / work_done_per_two_days
  show total_two_day_cycles ≤ 5 ∧ 2 * 5 = 10
  sorry

end work_completion_time_l282_282337


namespace sum_of_excluded_values_l282_282459

theorem sum_of_excluded_values (C D : ℝ) (h₁ : 2 * C^2 - 8 * C + 6 = 0)
    (h₂ : 2 * D^2 - 8 * D + 6 = 0) (h₃ : C ≠ D) :
    C + D = 4 :=
sorry

end sum_of_excluded_values_l282_282459


namespace correct_answer_is_B_l282_282332

-- Definitions for each set of line segments
def setA := (2, 2, 4)
def setB := (8, 6, 3)
def setC := (2, 6, 3)
def setD := (11, 4, 6)

-- Triangle inequality theorem checking function
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Statements to verify each set
lemma check_setA : ¬ is_triangle 2 2 4 := by sorry
lemma check_setB : is_triangle 8 6 3 := by sorry
lemma check_setC : ¬ is_triangle 2 6 3 := by sorry
lemma check_setD : ¬ is_triangle 11 4 6 := by sorry

-- Final theorem combining all checks to match the given problem
theorem correct_answer_is_B : 
  ¬ is_triangle 2 2 4 ∧ is_triangle 8 6 3 ∧ ¬ is_triangle 2 6 3 ∧ ¬ is_triangle 11 4 6 :=
by sorry

end correct_answer_is_B_l282_282332


namespace similar_triangles_height_ratio_l282_282478

-- Given condition: two similar triangles have a similarity ratio of 3:5
def similar_triangles (ratio : ℕ) : Prop := ratio = 3 ∧ ratio = 5

-- Goal: What is the ratio of their corresponding heights?
theorem similar_triangles_height_ratio (r : ℕ) (h : similar_triangles r) :
  r = 3 / 5 :=
sorry

end similar_triangles_height_ratio_l282_282478


namespace krista_driving_hours_each_day_l282_282413

-- Define the conditions as constants
def road_trip_days : ℕ := 3
def jade_hours_per_day : ℕ := 8
def total_hours : ℕ := 42

-- Define the function to calculate Krista's hours per day
noncomputable def krista_hours_per_day : ℕ :=
  (total_hours - road_trip_days * jade_hours_per_day) / road_trip_days

-- State the theorem to prove Krista drove 6 hours each day
theorem krista_driving_hours_each_day : krista_hours_per_day = 6 := by
  sorry

end krista_driving_hours_each_day_l282_282413


namespace yeast_counting_procedure_l282_282106

def yeast_counting_conditions (counting_method : String) (shake_test_tube_needed : Bool) (dilution_needed : Bool) : Prop :=
  counting_method = "Sampling inspection" ∧ 
  shake_test_tube_needed = true ∧ 
  dilution_needed = true

theorem yeast_counting_procedure :
  ∀ (counting_method : String) (shake_test_tube_needed : Bool) (dilution_needed : Bool),
  yeast_counting_conditions counting_method shake_test_tube_needed dilution_needed →
  counting_method = "Sampling inspection" ∧ 
  shake_test_tube_needed = true ∧ 
  dilution_needed = true :=
by
  intros counting_method shake_test_tube_needed dilution_needed h_condition
  exact h_condition

end yeast_counting_procedure_l282_282106


namespace find_x_l282_282228

theorem find_x (n : ℕ) 
  (h1 : n % 2 = 1)
  (h2 : ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ p1 * p2 * p3 = 11 * (7^n + 1) ∧ p1.prime ∧ p2.prime ∧ p3.prime): 
  7^n + 1 = 16808 :=
begin
  sorry
end

end find_x_l282_282228


namespace final_sum_after_50_passes_l282_282776

theorem final_sum_after_50_passes
  (particip: ℕ) 
  (num_passes: particip = 50) 
  (init_disp: ℕ → ℤ) 
  (initial_condition : init_disp 0 = 1 ∧ init_disp 1 = 0 ∧ init_disp 2 = -1)
  (operations: Π (i : ℕ), 
    (init_disp 0 = 1 →
    init_disp 1 = 0 →
    (i % 2 = 0 → init_disp 2 = -1) →
    (i % 2 = 1 → init_disp 2 = 1))
  )
  : init_disp 0 + init_disp 1 + init_disp 2 = 0 :=
by
  sorry

end final_sum_after_50_passes_l282_282776


namespace compute_fraction_l282_282585

theorem compute_fraction (x y z : ℝ) (h : x + y + z = 1) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = (1 - (x^2 + y^2 + z^2)) / (2 * (x^2 + y^2 + z^2)) :=
by 
  sorry

end compute_fraction_l282_282585


namespace solve_for_m_l282_282220

def A := {x : ℝ | x^2 + 3*x - 10 ≤ 0}
def B (m : ℝ) := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem solve_for_m (m : ℝ) (h : B m ⊆ A) : m < 2 :=
by
  sorry

end solve_for_m_l282_282220


namespace proof_problem_l282_282378

def operation1 (x : ℝ) := 9 - x
def operation2 (x : ℝ) := x - 9

theorem proof_problem : operation2 (operation1 15) = -15 := 
by
  sorry

end proof_problem_l282_282378


namespace ab_cd_value_l282_282089

theorem ab_cd_value (a b c d : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a + b + d = -3) 
  (h3 : a + c + d = 10) 
  (h4 : b + c + d = -1) : 
  ab + cd = -346 / 9 :=
by 
  sorry

end ab_cd_value_l282_282089


namespace rex_lesson_schedule_l282_282277

-- Define the total lessons and weeks
def total_lessons : ℕ := 40
def weeks_completed : ℕ := 6
def weeks_remaining : ℕ := 4

-- Define the proof statement
theorem rex_lesson_schedule : (weeks_completed + weeks_remaining) * 4 = total_lessons := by
  -- Proof placeholder, to be filled in 
  sorry

end rex_lesson_schedule_l282_282277


namespace expand_expression_l282_282948

theorem expand_expression : ∀ (x : ℝ), (1 + x^3) * (1 - x^4 + x^5) = 1 + x^3 - x^4 + x^5 - x^7 + x^8 :=
by
  intro x
  sorry

end expand_expression_l282_282948


namespace convert_degrees_to_radians_l282_282367

theorem convert_degrees_to_radians (deg : ℝ) (deg_eq : deg = -300) : 
  deg * (π / 180) = - (5 * π) / 3 := 
by
  rw [deg_eq]
  sorry

end convert_degrees_to_radians_l282_282367


namespace bill_due_months_l282_282612

theorem bill_due_months {TD A: ℝ} (R: ℝ) : 
  TD = 189 → A = 1764 → R = 16 → 
  ∃ M: ℕ, A - TD * (1 + (R/100) * (M/12)) = 1764 - 189 * (1 + (16/100) * (10/12)) ∧ M = 10 :=
by
  intro hTD hA hR
  use 10
  sorry

end bill_due_months_l282_282612


namespace max_mn_l282_282871

theorem max_mn (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (m n : ℝ)
  (h₂ : 2 * m + n = 2) : m * n ≤ 4 / 9 :=
by
  sorry

end max_mn_l282_282871


namespace max_value_of_f_l282_282294

noncomputable def f (x : ℝ) : ℝ := (1/5) * Real.sin (x + Real.pi/3) + Real.cos (x - Real.pi/6)

theorem max_value_of_f : ∀ x : ℝ, f x ≤ 6/5 := by
  sorry

end max_value_of_f_l282_282294


namespace circle_radius_correct_l282_282044

noncomputable def radius_of_circle 
  (side_length : ℝ)
  (angle_tangents : ℝ)
  (sin_18 : ℝ) : ℝ := 
  sorry

theorem circle_radius_correct 
  (side_length : ℝ := 6 + 2 * Real.sqrt 5)
  (angle_tangents : ℝ := 36)
  (sin_18 : ℝ := (Real.sqrt 5 - 1) / 4) :
  radius_of_circle side_length angle_tangents sin_18 = 
  2 * (2 * Real.sqrt 2 + Real.sqrt 5 - 1) :=
sorry

end circle_radius_correct_l282_282044


namespace each_friend_received_12_candies_l282_282934

-- Define the number of friends and total candies given
def num_friends : ℕ := 35
def total_candies : ℕ := 420

-- Define the number of candies each friend received
def candies_per_friend : ℕ := total_candies / num_friends

theorem each_friend_received_12_candies :
  candies_per_friend = 12 :=
by
  -- Skip the proof
  sorry

end each_friend_received_12_candies_l282_282934


namespace second_train_length_is_120_l282_282319

noncomputable def length_of_second_train
  (speed_train1_kmph : ℝ) 
  (speed_train2_kmph : ℝ) 
  (crossing_time : ℝ) 
  (length_train1_m : ℝ) : ℝ :=
  let speed_train1_mps := speed_train1_kmph * 1000 / 3600
  let speed_train2_mps := speed_train2_kmph * 1000 / 3600
  let relative_speed := speed_train1_mps + speed_train2_mps
  let distance := relative_speed * crossing_time
  distance - length_train1_m

theorem second_train_length_is_120 :
  length_of_second_train 60 40 6.119510439164867 50 = 120 :=
by
  -- Here's where the proof would go
  sorry

end second_train_length_is_120_l282_282319


namespace ratio_equivalence_l282_282983

theorem ratio_equivalence (x : ℕ) (h1 : 3 / 12 = x / 16) : x = 4 :=
by sorry

end ratio_equivalence_l282_282983


namespace is_incorrect_B_l282_282542

variable {a b c : ℝ}

theorem is_incorrect_B :
  ¬ ((a > b ∧ b > c) → (1 / (b - c)) < (1 / (a - c))) :=
sorry

end is_incorrect_B_l282_282542


namespace regular_price_of_shirt_is_50_l282_282435

-- Define all relevant conditions and given prices.
variables (P : ℝ) (shirt_price_discounted : ℝ) (total_paid : ℝ) (number_of_shirts : ℝ)

-- Define the conditions as hypotheses
def conditions :=
  (shirt_price_discounted = 0.80 * P) ∧
  (total_paid = 240) ∧
  (number_of_shirts = 6) ∧
  (total_paid = number_of_shirts * shirt_price_discounted)

-- State the theorem to prove that the regular price of the shirt is $50.
theorem regular_price_of_shirt_is_50 (h : conditions P shirt_price_discounted total_paid number_of_shirts) :
  P = 50 := 
sorry

end regular_price_of_shirt_is_50_l282_282435


namespace area_of_given_parallelogram_l282_282898

def parallelogram_base : ℝ := 24
def parallelogram_height : ℝ := 16
def parallelogram_area (b h : ℝ) : ℝ := b * h

theorem area_of_given_parallelogram : parallelogram_area parallelogram_base parallelogram_height = 384 := 
by sorry

end area_of_given_parallelogram_l282_282898


namespace find_natural_n_l282_282874

theorem find_natural_n (a : ℂ) (h₀ : a ≠ 0) (h₁ : a ≠ 1) (h₂ : a ≠ -1)
    (h₃ : a ^ 11 + a ^ 7 + a ^ 3 = 1) : a ^ 4 + a ^ 3 = a ^ 15 + 1 :=
sorry

end find_natural_n_l282_282874


namespace number_of_triangles_l282_282847

theorem number_of_triangles (OA_points OB_points : ℕ) (O : ℕ) :
    OA_points = 4 ∧ OB_points = 5 ∧ O = 1 →
    ∑ n in (Finset.range 10).powersetLen 3, if isCollinear n then 0 else 1 = 90 :=
by
    sorry

-- Assumptions and conditions definitions
def isCollinear (points : Finset ℕ) : Bool :=
    sorry  -- Define the collinearity check based on the edges.

end number_of_triangles_l282_282847


namespace three_digit_integer_divisible_by_5_l282_282305

theorem three_digit_integer_divisible_by_5 (M : ℕ) (h1 : 100 ≤ M ∧ M < 1000) (h2 : M % 10 = 5) : M % 5 = 0 := 
sorry

end three_digit_integer_divisible_by_5_l282_282305


namespace characterize_solution_pairs_l282_282796

/-- Define a set S --/
def S : Set ℝ := { x : ℝ | x > 0 ∧ x ≠ 1 }

/-- log inequality --/
def log_inequality (a b : ℝ) : Prop :=
  Real.log b / Real.log a < Real.log (b + 1) / Real.log (a + 1)

/-- Define the solution sets --/
def sol1 : Set (ℝ × ℝ) := {p | p.2 = 1 ∧ p.1 > 0 ∧ p.1 ≠ 1}
def sol2 : Set (ℝ × ℝ) := {p | p.1 > p.2 ∧ p.2 > 1}
def sol3 : Set (ℝ × ℝ) := {p | p.2 > 1 ∧ p.2 > p.1}
def sol4 : Set (ℝ × ℝ) := {p | p.1 < p.2 ∧ p.2 < 1}
def sol5 : Set (ℝ × ℝ) := {p | p.2 < 1 ∧ p.2 < p.1}

/-- Prove the log inequality and characterize the solution pairs --/
theorem characterize_solution_pairs (a b : ℝ) (h1 : a ∈ S) (h2 : b > 0) :
  log_inequality a b ↔
  (a, b) ∈ sol1 ∨ (a, b) ∈ sol2 ∨ (a, b) ∈ sol3 ∨ (a, b) ∈ sol4 ∨ (a, b) ∈ sol5 :=
sorry

end characterize_solution_pairs_l282_282796


namespace arithmetic_sequence_a1_a5_product_l282_282022

theorem arithmetic_sequence_a1_a5_product 
  (a : ℕ → ℚ) 
  (h_arithmetic : ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a3 : a 3 = 3) 
  (h_cond : (1 / a 1) + (1 / a 5) = 6 / 5) : 
  a 1 * a 5 = 5 := 
by
  sorry

end arithmetic_sequence_a1_a5_product_l282_282022


namespace evaluate_expression_1_evaluate_expression_2_l282_282493

-- Problem 1
def expression_1 (a b : Int) : Int :=
  2 * a + 3 * b - 2 * a * b - a - 4 * b - a * b

theorem evaluate_expression_1 : expression_1 6 (-1) = 25 :=
by
  sorry

-- Problem 2
def expression_2 (m n : Int) : Int :=
  m^2 + 2 * m * n + n^2

theorem evaluate_expression_2 (m n : Int) (hm : |m| = 3) (hn : |n| = 2) (hmn : m < n) : expression_2 m n = 1 :=
by
  sorry

end evaluate_expression_1_evaluate_expression_2_l282_282493


namespace functional_eq_solution_l282_282801

noncomputable def f : ℚ → ℚ := sorry

theorem functional_eq_solution (f : ℚ → ℚ)
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1):
  ∀ x : ℚ, f x = x + 1 :=
sorry

end functional_eq_solution_l282_282801


namespace cone_from_sector_l282_282634

theorem cone_from_sector
  (r : ℝ) (slant_height : ℝ)
  (radius_circle : ℝ := 10)
  (angle_sector : ℝ := 252) :
  (r = 7 ∧ slant_height = 10) :=
by
  sorry

end cone_from_sector_l282_282634


namespace painting_cost_l282_282860

theorem painting_cost (total_cost : ℕ) (num_paintings : ℕ) (price : ℕ)
  (h1 : total_cost = 104)
  (h2 : 10 < num_paintings)
  (h3 : num_paintings < 60)
  (h4 : total_cost = num_paintings * price)
  (h5 : price ∈ {d ∈ {d : ℕ | d > 0} | total_cost % d = 0}) :
  price = 2 ∨ price = 4 ∨ price = 8 :=
by
  sorry

end painting_cost_l282_282860


namespace equivalent_problem_l282_282428

theorem equivalent_problem (n : ℕ) (h₁ : 0 ≤ n) (h₂ : n < 29) (h₃ : 2 * n % 29 = 1) :
  (3^n % 29)^3 - 3 % 29 = 3 :=
sorry

end equivalent_problem_l282_282428


namespace pieces_count_l282_282885

def pieces_after_n_tears (n : ℕ) : ℕ :=
  3 * n + 1

theorem pieces_count (n : ℕ) : pieces_after_n_tears n = 3 * n + 1 :=
by
  sorry

end pieces_count_l282_282885


namespace simplify_fraction_l282_282862

variable (x y : ℝ)

theorem simplify_fraction :
  (2 * x + y) / 4 + (5 * y - 4 * x) / 6 - y / 12 = (-x + 6 * y) / 6 :=
by
  sorry

end simplify_fraction_l282_282862


namespace adam_age_l282_282921

variable (E A : ℕ)

namespace AgeProof

theorem adam_age (h1 : A = E - 5) (h2 : E + 1 = 3 * (A - 4)) : A = 9 :=
by
  sorry
end AgeProof

end adam_age_l282_282921


namespace log_ordering_l282_282685

theorem log_ordering 
  (a b c : ℝ) 
  (ha: a = Real.log 3 / Real.log 2) 
  (hb: b = Real.log 2 / Real.log 3) 
  (hc: c = Real.log 0.5 / Real.log 10) : 
  a > b ∧ b > c := 
by 
  sorry

end log_ordering_l282_282685


namespace max_pies_without_ingredients_l282_282368

theorem max_pies_without_ingredients :
  let total_pies := 36
  let chocolate_pies := total_pies / 3
  let marshmallow_pies := total_pies / 4
  let cayenne_pies := total_pies / 2
  let soy_nuts_pies := total_pies / 8
  let max_ingredient_pies := max (max chocolate_pies marshmallow_pies) (max cayenne_pies soy_nuts_pies)
  total_pies - max_ingredient_pies = 18 :=
by
  sorry

end max_pies_without_ingredients_l282_282368


namespace range_of_a_l282_282588

noncomputable def f (x : ℝ) := (1 / 2) * x ^ 2 - 16 * Real.log x

theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, a - 1 ≤ x ∧ x ≤ a + 2 → (fderiv ℝ f x) x < 0)
  ↔ (1 < a) ∧ (a ≤ 2) :=
by
  sorry

end range_of_a_l282_282588


namespace arithmetic_geometric_sequence_l282_282817

noncomputable def a_n (n : ℕ) : ℚ := 2 * n

def S (n : ℕ) : ℚ := n * a_n n / 2

def b (n : ℕ) : ℚ := 1 / ((a_n n - 1) * (a_n n + 1))

def T (n : ℕ) : ℚ := ∑ i in finset.range (n + 1), b i

theorem arithmetic_geometric_sequence (S10_eq : S 10 = 110)
    (geo_seq : (a_n 2)^2 = a_n 1 * a_n 4) : 
  (∀ n, a_n n = 2 * n) ∧ (∀ n, T n = n / (2 * n + 1)) :=
by sorry

end arithmetic_geometric_sequence_l282_282817


namespace teacherZhangAge_in_5_years_correct_l282_282620

variable (a : ℕ)

def teacherZhangAgeCurrent := 3 * a - 2

def teacherZhangAgeIn5Years := teacherZhangAgeCurrent a + 5

theorem teacherZhangAge_in_5_years_correct :
  teacherZhangAgeIn5Years a = 3 * a + 3 := by
  sorry

end teacherZhangAge_in_5_years_correct_l282_282620


namespace acute_angle_inequality_l282_282077

theorem acute_angle_inequality (α : ℝ) (h₀ : 0 < α) (h₁ : α < π / 2) :
  α < (Real.sin α + Real.tan α) / 2 := 
sorry

end acute_angle_inequality_l282_282077


namespace three_digit_integer_divisible_by_5_l282_282303

theorem three_digit_integer_divisible_by_5 (M : ℕ) (h1 : 100 ≤ M ∧ M < 1000) (h2 : M % 10 = 5) : M % 5 = 0 := 
sorry

end three_digit_integer_divisible_by_5_l282_282303


namespace original_price_of_trouser_l282_282995

theorem original_price_of_trouser (sale_price : ℝ) (discount_rate : ℝ) (original_price : ℝ) 
  (h1 : sale_price = 50) (h2 : discount_rate = 0.50) (h3 : sale_price = (1 - discount_rate) * original_price) : 
  original_price = 100 :=
sorry

end original_price_of_trouser_l282_282995


namespace sam_current_dimes_l282_282735

def original_dimes : ℕ := 8
def sister_borrowed : ℕ := 4
def friend_borrowed : ℕ := 2
def sister_returned : ℕ := 2
def friend_returned : ℕ := 1

theorem sam_current_dimes : 
  (original_dimes - sister_borrowed - friend_borrowed + sister_returned + friend_returned = 5) :=
by
  sorry

end sam_current_dimes_l282_282735


namespace sum_of_coefficients_sum_even_odd_coefficients_l282_282717

noncomputable def P (x : ℝ) : ℝ := (2 * x^2 - 2 * x + 1)^17 * (3 * x^2 - 3 * x + 1)^17

theorem sum_of_coefficients : P 1 = 1 := by
  sorry

theorem sum_even_odd_coefficients :
  (P 1 + P (-1)) / 2 = (1 + 35^17) / 2 ∧ (P 1 - P (-1)) / 2 = (1 - 35^17) / 2 := by
  sorry

end sum_of_coefficients_sum_even_odd_coefficients_l282_282717


namespace erased_angle_is_correct_l282_282792

theorem erased_angle_is_correct (n : ℕ) (x : ℝ) (h_convex : convex_polygon n) (h_sum_remaining : sum_remaining_angles = 1703) : x = 97 :=
by
  -- This is where the proof would be placed, but we'll use sorry for now
  sorry

end erased_angle_is_correct_l282_282792


namespace total_payment_correct_l282_282209

-- Define the prices of different apples.
def price_small_apple : ℝ := 1.5
def price_medium_apple : ℝ := 2.0
def price_big_apple : ℝ := 3.0

-- Define the quantities of apples bought by Donny.
def quantity_small_apples : ℕ := 6
def quantity_medium_apples : ℕ := 6
def quantity_big_apples : ℕ := 8

-- Define the conditions.
def discount_medium_apples_threshold : ℕ := 5
def discount_medium_apples_rate : ℝ := 0.20
def tax_rate : ℝ := 0.10
def big_apple_special_offer_count : ℕ := 3
def big_apple_special_offer_discount_rate : ℝ := 0.50

-- Step function to calculate discount and total cost.
noncomputable def total_cost : ℝ :=
  let cost_small := quantity_small_apples * price_small_apple
  let cost_medium := quantity_medium_apples * price_medium_apple
  let discount_medium := if quantity_medium_apples > discount_medium_apples_threshold 
                         then cost_medium * discount_medium_apples_rate else 0
  let cost_medium_after_discount := cost_medium - discount_medium
  let cost_big := quantity_big_apples * price_big_apple
  let discount_big := (quantity_big_apples / big_apple_special_offer_count) * 
                       (price_big_apple * big_apple_special_offer_discount_rate)
  let cost_big_after_discount := cost_big - discount_big
  let total_cost_before_tax := cost_small + cost_medium_after_discount + cost_big_after_discount
  let tax := total_cost_before_tax * tax_rate
  total_cost_before_tax + tax

-- Define the expected total payment.
def expected_total_payment : ℝ := 43.56

-- The theorem statement: Prove that total_cost equals the expected total payment.
theorem total_payment_correct : total_cost = expected_total_payment := sorry

end total_payment_correct_l282_282209


namespace ellipse_range_of_k_l282_282564

theorem ellipse_range_of_k (k : ℝ) :
  (4 - k > 0) → (k - 1 > 0) → (4 - k ≠ k - 1) → (1 < k ∧ k < 4 ∧ k ≠ 5 / 2) :=
by
  intros h1 h2 h3
  sorry

end ellipse_range_of_k_l282_282564


namespace AlWinsProbability_l282_282657

noncomputable def BobPlaysRandomly : 𝕜 :=
  (mk_ℙ [1/3, 1/3, 1/3]) 

noncomputable def AlPlaysRock : 𝕜 :=
  mk_ℙ [1]

theorem AlWinsProbability :
  ∀ (P_rock P_paper P_scissors : ℝ),
    P_rock = 1/3 → 
    P_paper = 1/3 →
    P_scissors = 1/3 →
    (P_rock + P_paper + P_scissors = 1) → 
    (AlPlaysRock * (P_scissors)) = (1/3) :=
by
  sorry

end AlWinsProbability_l282_282657


namespace halfway_between_one_third_and_one_fifth_l282_282532

theorem halfway_between_one_third_and_one_fifth : (1/3 + 1/5) / 2 = 4/15 := 
by 
  sorry

end halfway_between_one_third_and_one_fifth_l282_282532


namespace number_of_boys_in_biology_class_l282_282151

variable (B G : ℕ) (PhysicsClass BiologyClass : ℕ)

theorem number_of_boys_in_biology_class
  (h1 : G = 3 * B)
  (h2 : PhysicsClass = 200)
  (h3 : BiologyClass = PhysicsClass / 2)
  (h4 : BiologyClass = B + G) :
  B = 25 := by
  sorry

end number_of_boys_in_biology_class_l282_282151


namespace wrapping_paper_fraction_used_l282_282008

theorem wrapping_paper_fraction_used 
  (total_paper_used : ℚ)
  (num_presents : ℕ)
  (each_present_used : ℚ)
  (h1 : total_paper_used = 1 / 2)
  (h2 : num_presents = 5)
  (h3 : each_present_used = total_paper_used / num_presents) : 
  each_present_used = 1 / 10 := 
by
  sorry

end wrapping_paper_fraction_used_l282_282008


namespace solution_exists_l282_282373

theorem solution_exists (n p : ℕ) (hp : p.prime) (hn : 0 < n ∧ n ≤ 2 * p) :
  n^(p-1) ∣ (p-1)^n + 1 :=
sorry

end solution_exists_l282_282373


namespace pipe_B_filling_time_l282_282315

theorem pipe_B_filling_time (T_B : ℝ) 
  (A_filling_time : ℝ := 10) 
  (combined_filling_time: ℝ := 20/3)
  (A_rate : ℝ := 1 / A_filling_time)
  (combined_rate : ℝ := 1 / combined_filling_time) : 
  1 / T_B = combined_rate - A_rate → T_B = 20 := by 
  sorry

end pipe_B_filling_time_l282_282315


namespace cube_inequality_l282_282443

theorem cube_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^3 + b^3) / 2 ≥ ((a + b) / 2)^3 :=
by 
  sorry

end cube_inequality_l282_282443


namespace percentage_of_students_attend_chess_class_l282_282881

-- Definitions based on the conditions
def total_students : ℕ := 1000
def swimming_students : ℕ := 125
def chess_to_swimming_ratio : ℚ := 1 / 2

-- Problem statement
theorem percentage_of_students_attend_chess_class :
  ∃ P : ℚ, (P / 100) * total_students / 2 = swimming_students → P = 25 := by
  sorry

end percentage_of_students_attend_chess_class_l282_282881


namespace problem_x_l282_282811

theorem problem_x (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x : ℝ, f (1/2 * x - 1) = 2 * x + 3) 
  (h2 : f m = 6) : 
  m = -1/4 :=
sorry

end problem_x_l282_282811


namespace work_rate_l282_282774

theorem work_rate (R_B : ℚ) (R_A : ℚ) (R_total : ℚ) (days : ℚ)
  (h1 : R_A = (1/2) * R_B)
  (h2 : R_B = 1 / 22.5)
  (h3 : R_total = R_A + R_B)
  (h4 : days = 1 / R_total) : 
  days = 15 := 
sorry

end work_rate_l282_282774


namespace factor_tree_value_l282_282705

-- Define the values and their relationships
def A := 900
def B := 3 * (3 * 2)
def D := 3 * 2
def C := 5 * (5 * 2)
def E := 5 * 2

-- Define the theorem and provide the conditions
theorem factor_tree_value :
  (B = 3 * D) →
  (D = 3 * 2) →
  (C = 5 * E) →
  (E = 5 * 2) →
  (A = B * C) →
  A = 900 := by
  intros hB hD hC hE hA
  sorry

end factor_tree_value_l282_282705


namespace mean_of_roots_l282_282999

theorem mean_of_roots
  (a b c d k : ℤ)
  (p : ℤ → ℤ)
  (h_poly : ∀ x, p x = (x - a) * (x - b) * (x - c) * (x - d))
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_root : p k = 4) :
  k = (a + b + c + d) / 4 :=
by
  -- proof goes here
  sorry

end mean_of_roots_l282_282999


namespace sales_in_fourth_month_l282_282342

theorem sales_in_fourth_month (sale_m1 sale_m2 sale_m3 sale_m5 sale_m6 avg_sales total_months : ℕ)
    (H1 : sale_m1 = 7435) (H2 : sale_m2 = 7927) (H3 : sale_m3 = 7855) 
    (H4 : sale_m5 = 7562) (H5 : sale_m6 = 5991) (H6 : avg_sales = 7500) (H7 : total_months = 6) :
    ∃ sale_m4 : ℕ, sale_m4 = 8230 := by
  sorry

end sales_in_fourth_month_l282_282342


namespace work_rate_a_b_l282_282486

/-- a and b can do a piece of work in some days, b and c in 5 days, c and a in 15 days. If c takes 12 days to do the work, 
    prove that a and b together can complete the work in 10 days.
-/
theorem work_rate_a_b
  (A B C : ℚ) 
  (h1 : B + C = 1 / 5)
  (h2 : C + A = 1 / 15)
  (h3 : C = 1 / 12) :
  (A + B = 1 / 10) := 
sorry

end work_rate_a_b_l282_282486


namespace projection_of_sum_on_vec_a_l282_282813

open Real

noncomputable def vector_projection (a b : ℝ) (angle : ℝ) : ℝ := 
  (cos angle) * (a * b) / a

theorem projection_of_sum_on_vec_a (a b : EuclideanSpace ℝ (Fin 3)) 
  (h₁ : ‖a‖ = 2) 
  (h₂ : ‖b‖ = 2) 
  (h₃ : inner a b = (2 * 2) * (cos (π / 3))):
  (inner (a + b) a) / ‖a‖ = 3 := 
by
  sorry

end projection_of_sum_on_vec_a_l282_282813


namespace bill_due_months_l282_282613

theorem bill_due_months {TD A: ℝ} (R: ℝ) : 
  TD = 189 → A = 1764 → R = 16 → 
  ∃ M: ℕ, A - TD * (1 + (R/100) * (M/12)) = 1764 - 189 * (1 + (16/100) * (10/12)) ∧ M = 10 :=
by
  intro hTD hA hR
  use 10
  sorry

end bill_due_months_l282_282613


namespace triangle_area_is_correct_l282_282479

-- Defining the vertices of the triangle
def vertexA : ℝ × ℝ := (0, 0)
def vertexB : ℝ × ℝ := (0, 6)
def vertexC : ℝ × ℝ := (8, 10)

-- Define a function to calculate the area of a triangle given three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Statement to prove
theorem triangle_area_is_correct : triangle_area vertexA vertexB vertexC = 24.0 := 
by
  sorry

end triangle_area_is_correct_l282_282479


namespace coeff_x4_in_expansion_l282_282628

open Nat

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def coefficient_x4_term : ℕ := binom 9 4

noncomputable def constant_term : ℕ := 243 * 4

theorem coeff_x4_in_expansion : coefficient_x4_term * 972 * Real.sqrt 2 = 122472 * Real.sqrt 2 :=
by
  sorry

end coeff_x4_in_expansion_l282_282628


namespace total_animals_after_addition_l282_282595

def current_cows := 2
def current_pigs := 3
def current_goats := 6

def added_cows := 3
def added_pigs := 5
def added_goats := 2

def total_current_animals := current_cows + current_pigs + current_goats
def total_added_animals := added_cows + added_pigs + added_goats
def total_animals := total_current_animals + total_added_animals

theorem total_animals_after_addition : total_animals = 21 := by
  sorry

end total_animals_after_addition_l282_282595


namespace zoo_with_hippos_only_l282_282434

variables {Z : Type} -- The type of all zoos
variables (H R G : Set Z) -- Subsets of zoos with hippos, rhinos, and giraffes respectively

-- Conditions
def condition1 : Prop := ∀ (z : Z), z ∈ H ∧ z ∈ R → z ∉ G
def condition2 : Prop := ∀ (z : Z), z ∈ R ∧ z ∉ G → z ∈ H
def condition3 : Prop := ∀ (z : Z), z ∈ H ∧ z ∈ G → z ∈ R

-- Goal
def goal : Prop := ∃ (z : Z), z ∈ H ∧ z ∉ G ∧ z ∉ R

-- Theorem statement
theorem zoo_with_hippos_only (h1 : condition1 H R G) (h2 : condition2 H R G) (h3 : condition3 H R G) : goal H R G :=
sorry

end zoo_with_hippos_only_l282_282434


namespace remainder_4032_125_l282_282890

theorem remainder_4032_125 : 4032 % 125 = 32 := by
  sorry

end remainder_4032_125_l282_282890


namespace amanda_car_round_trip_time_l282_282198

theorem amanda_car_round_trip_time :
  (bus_time = 40) ∧ (car_time = bus_time - 5) → (round_trip_time = car_time * 2) → round_trip_time = 70 :=
by
  sorry

end amanda_car_round_trip_time_l282_282198


namespace hotel_floors_l282_282974

/-- Given:
  - Each floor has 10 identical rooms.
  - The last floor is unavailable for guests.
  - Hans could be checked into 90 different rooms.
  - There are no other guests.
 - Prove that the total number of floors in the hotel is 10.
--/
theorem hotel_floors :
  (∃ n : ℕ, n ≥ 1 ∧ 10 * (n - 1) = 90) → n = 10 :=
by 
  sorry

end hotel_floors_l282_282974


namespace distance_between_homes_l282_282724

-- Define the parameters
def maxwell_speed : ℝ := 4  -- km/h
def brad_speed : ℝ := 6     -- km/h
def maxwell_time_to_meet : ℝ := 2  -- hours
def brad_start_delay : ℝ := 1  -- hours

-- Definitions related to the timings
def brad_time_to_meet : ℝ := maxwell_time_to_meet - brad_start_delay  -- hours

-- Define the distances covered by each
def maxwell_distance : ℝ := maxwell_speed * maxwell_time_to_meet  -- km
def brad_distance : ℝ := brad_speed * brad_time_to_meet  -- km

-- Define the total distance between their homes
def total_distance : ℝ := maxwell_distance + brad_distance  -- km

-- Statement to prove
theorem distance_between_homes : total_distance = 14 :=
by
  -- The proof is omitted; add 'sorry' to indicate this.
  sorry

end distance_between_homes_l282_282724


namespace probability_red_and_at_least_one_even_l282_282340

-- Definitions based on conditions
def total_balls : ℕ := 12
def red_balls : Finset ℕ := {1, 2, 3, 4, 5, 6}
def black_balls : Finset ℕ := {7, 8, 9, 10, 11, 12}

-- Condition to check if a ball is red
def is_red (n : ℕ) : Prop := n ∈ red_balls

-- Condition to check if a ball has an even number
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Total number of ways to draw two balls with replacement
def total_ways : ℕ := total_balls * total_balls

-- Number of ways to draw both red balls
def red_red_ways : ℕ := Finset.card red_balls * Finset.card red_balls

-- Number of ways to draw both red balls with none even
def red_odd_numbers : Finset ℕ := {1, 3, 5}
def red_red_odd_ways : ℕ := Finset.card red_odd_numbers * Finset.card red_odd_numbers

-- Number of ways to draw both red balls with at least one even
def desired_outcomes : ℕ := red_red_ways - red_red_odd_ways

-- The probability
def probability : ℚ := desired_outcomes / total_ways

theorem probability_red_and_at_least_one_even :
  probability = 3 / 16 :=
by
  sorry

end probability_red_and_at_least_one_even_l282_282340


namespace wax_total_is_correct_l282_282389

-- Define the given conditions
def current_wax : ℕ := 20
def additional_wax : ℕ := 146

-- The total amount of wax required is the sum of current_wax and additional_wax
def total_wax := current_wax + additional_wax

-- The proof goal is to show that the total_wax equals 166 grams
theorem wax_total_is_correct : total_wax = 166 := by
  sorry

end wax_total_is_correct_l282_282389


namespace probability_of_x_gt_8y_l282_282727

noncomputable def probability_x_gt_8y : ℚ :=
  let rect_area := 2020 * 2030
  let tri_area := (2020 * (2020 / 8)) / 2
  tri_area / rect_area

theorem probability_of_x_gt_8y :
  probability_x_gt_8y = 255025 / 4100600 := by
  sorry

end probability_of_x_gt_8y_l282_282727


namespace symmetry_probability_l282_282148

-- Define the setting of the problem
def grid_points : ℕ := 121
def grid_size : ℕ := 11
def center_point : (ℕ × ℕ) := (6, 6)
def total_points : ℕ := grid_points - 1
def symmetric_lines : ℕ := 4
def points_per_line : ℕ := 10
def total_symmetric_points : ℕ := symmetric_lines * points_per_line
def probability : ℚ := total_symmetric_points / total_points

-- Theorem statement
theorem symmetry_probability 
  (hp: grid_points = 121) 
  (hs: grid_size = 11) 
  (hc: center_point = (6, 6))
  (htp: total_points = 120)
  (hsl: symmetric_lines = 4)
  (hpl: points_per_line = 10)
  (htsp: total_symmetric_points = 40)
  (hp: probability = 1 / 3) : 
  probability = 1 / 3 :=
by 
  sorry

end symmetry_probability_l282_282148


namespace problem_1_problem_2_problem_3_l282_282178

theorem problem_1 : 
  ∀ x : ℝ, x^2 - 2 * x + 5 = (x - 1)^2 + 4 := 
sorry

theorem problem_2 (n : ℝ) (h : ∀ x : ℝ, x^2 + 2 * n * x + 3 = (x + 5)^2 - 25 + 3) : 
  n = -5 := 
sorry

theorem problem_3 (a : ℝ) (h : ∀ x : ℝ, (x^2 + 6 * x + 9) * (x^2 - 4 * x + 4) = ((x + a)^2 + b)^2) : 
  a = -1/2 := 
sorry

end problem_1_problem_2_problem_3_l282_282178


namespace expr_undefined_iff_l282_282067

theorem expr_undefined_iff (b : ℝ) : ¬ ∃ y : ℝ, y = (b - 1) / (b^2 - 9) ↔ b = -3 ∨ b = 3 :=
by 
  sorry

end expr_undefined_iff_l282_282067


namespace domain_of_f_l282_282741

noncomputable def f (x : ℝ) : ℝ := (Real.log (x^2 - 1)) / (Real.sqrt (x^2 - x - 2))

theorem domain_of_f :
  {x : ℝ | x^2 - 1 > 0 ∧ x^2 - x - 2 > 0} = {x : ℝ | x < -1 ∨ x > 2} :=
by
  sorry

end domain_of_f_l282_282741


namespace find_x_l282_282828

def vector_dot_product (v1 v2 : ℝ × ℝ) : ℝ := 
  v1.1 * v2.1 + v1.2 * v2.2

def a : ℝ × ℝ := (1, 2)

def b (x : ℝ) : ℝ × ℝ := (x, -2)

def c (x : ℝ) : ℝ × ℝ := (1 - x, 4)

theorem find_x (x : ℝ) (h : vector_dot_product a (c x) = 0) : x = 9 :=
by
  sorry

end find_x_l282_282828


namespace wrapping_paper_each_present_l282_282003

theorem wrapping_paper_each_present (total_paper : ℚ) (num_presents : ℕ)
  (h1 : total_paper = 1 / 2) (h2 : num_presents = 5) :
  (total_paper / num_presents = 1 / 10) :=
by
  sorry

end wrapping_paper_each_present_l282_282003


namespace current_length_of_highway_l282_282498

def total_length : ℕ := 650
def miles_first_day : ℕ := 50
def miles_second_day : ℕ := 3 * miles_first_day
def miles_still_needed : ℕ := 250
def miles_built : ℕ := miles_first_day + miles_second_day

theorem current_length_of_highway :
  total_length - miles_still_needed = 400 :=
by
  sorry

end current_length_of_highway_l282_282498


namespace train_cross_time_approx_l282_282412

noncomputable def time_to_cross_pole (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (5 / 18)
  length / speed_ms

theorem train_cross_time_approx
  (d : ℝ) (v_kmh : ℝ)
  (h_d : d = 120)
  (h_v : v_kmh = 121) :
  abs (time_to_cross_pole d v_kmh - 3.57) < 0.01 :=
by {
  sorry
}

end train_cross_time_approx_l282_282412


namespace sophia_collection_value_l282_282280

-- Define the conditions
def stamps_count : ℕ := 24
def partial_stamps_count : ℕ := 8
def partial_value : ℤ := 40
def stamp_value_per_each : ℤ := partial_value / partial_stamps_count
def total_value : ℤ := stamps_count * stamp_value_per_each

-- Statement of the conclusion that needs proving
theorem sophia_collection_value :
  total_value = 120 := by
  sorry

end sophia_collection_value_l282_282280


namespace problem1_range_problem2_range_l282_282185

theorem problem1_range (x y : ℝ) (h : y = 2*|x-1| - |x-4|) : -3 ≤ y := sorry

theorem problem2_range (x a : ℝ) (h : ∀ x, 2*|x-1| - |x-a| ≥ -1) : 0 ≤ a ∧ a ≤ 2 := sorry

end problem1_range_problem2_range_l282_282185


namespace famous_sentences_correct_l282_282799

def blank_1 : String := "correct_answer_1"
def blank_2 : String := "correct_answer_2"
def blank_3 : String := "correct_answer_3"
def blank_4 : String := "correct_answer_4"
def blank_5 : String := "correct_answer_5"
def blank_6 : String := "correct_answer_6"
def blank_7 : String := "correct_answer_7"
def blank_8 : String := "correct_answer_8"

theorem famous_sentences_correct :
  blank_1 = "correct_answer_1" ∧
  blank_2 = "correct_answer_2" ∧
  blank_3 = "correct_answer_3" ∧
  blank_4 = "correct_answer_4" ∧
  blank_5 = "correct_answer_5" ∧
  blank_6 = "correct_answer_6" ∧
  blank_7 = "correct_answer_7" ∧
  blank_8 = "correct_answer_8" :=
by
  -- The proof details correspond to the part "refer to the correct solution for each blank"
  sorry

end famous_sentences_correct_l282_282799


namespace percentage_of_women_in_study_group_l282_282189

theorem percentage_of_women_in_study_group
  (W : ℝ) -- percentage of women in decimal form
  (h1 : 0 < W ∧ W ≤ 1) -- percentage of women should be between 0 and 1
  (h2 : 0.4 * W = 0.32) -- 40 percent of women are lawyers, and probability is 0.32
  : W = 0.8 :=
  sorry

end percentage_of_women_in_study_group_l282_282189


namespace isosceles_triangle_side_length_l282_282896

theorem isosceles_triangle_side_length (total_length : ℝ) (one_side_length : ℝ) (remaining_wire : ℝ) (equal_side : ℝ) :
  total_length = 20 → one_side_length = 6 → remaining_wire = total_length - one_side_length → remaining_wire / 2 = equal_side →
  equal_side = 7 :=
by
  intros h_total h_one_side h_remaining h_equal_side
  sorry

end isosceles_triangle_side_length_l282_282896


namespace rooster_ratio_l282_282122

theorem rooster_ratio (R H : ℕ) 
  (h1 : R + H = 80)
  (h2 : R + (1 / 4) * H = 35) :
  R / 80 = 1 / 4 :=
  sorry

end rooster_ratio_l282_282122


namespace even_function_m_value_l282_282701

theorem even_function_m_value {m : ℤ} (h : ∀ (x : ℝ), (m^2 - m - 1) * (-x)^m = (m^2 - m - 1) * x^m) : m = 2 := 
by
  sorry

end even_function_m_value_l282_282701


namespace superhero_distance_difference_l282_282781

theorem superhero_distance_difference :
  let t := 4 in
  let v := 100 in
  (60 / t * 10) - v = 50 :=
by
  let t := 4
  let v := 100
  sorry

end superhero_distance_difference_l282_282781


namespace pen_case_cost_l282_282345

noncomputable def case_cost (p i c : ℝ) : Prop :=
  p + i + c = 2.30 ∧
  p = 1.50 + i ∧
  c = 0.5 * i →
  c = 0.1335

theorem pen_case_cost (p i c : ℝ) : case_cost p i c :=
by
  sorry

end pen_case_cost_l282_282345


namespace bucket_full_weight_l282_282639

theorem bucket_full_weight (p q : ℝ) (x y : ℝ) 
  (h1 : x + (1 / 3) * y = p) 
  (h2 : x + (3 / 4) * y = q) : 
  x + y = (8 * q - 3 * p) / 5 := 
  by
    sorry

end bucket_full_weight_l282_282639


namespace necessary_condition_for_x_greater_than_2_l282_282233

-- Define the real number x
variable (x : ℝ)

-- The proof statement
theorem necessary_condition_for_x_greater_than_2 : (x > 2) → (x > 1) :=
by sorry

end necessary_condition_for_x_greater_than_2_l282_282233


namespace inequality_proof_l282_282956

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 + b * c) / a + (1 + c * a) / b + (1 + a * b) / c > 
  Real.sqrt (a^2 + 2) + Real.sqrt (b^2 + 2) + Real.sqrt (c^2 + 2) := 
by
  sorry

end inequality_proof_l282_282956


namespace ages_total_l282_282187

theorem ages_total (P Q : ℕ) (h1 : P - 8 = (1 / 2) * (Q - 8)) (h2 : P / Q = 3 / 4) : P + Q = 28 :=
by
  sorry

end ages_total_l282_282187


namespace evaluate_expression_l282_282832

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 7) :
  (x^5 + 3 * y^3) / 9 = 141 :=
by
  sorry

end evaluate_expression_l282_282832


namespace ring_revolutions_before_stopping_l282_282648

variable (R ω μ m g : ℝ) -- Declare the variables as real numbers

-- Statement of the theorem
theorem ring_revolutions_before_stopping
  (h_positive_R : 0 < R)
  (h_positive_ω : 0 < ω)
  (h_positive_μ : 0 < μ)
  (h_positive_m : 0 < m)
  (h_positive_g : 0 < g) :
  let N1 := m * g / (1 + μ^2)
  let N2 := μ * m * g / (1 + μ^2)
  let K_initial := (1 / 2) * m * R^2 * ω^2
  let A_friction := -2 * π * R * n * μ * (N1 + N2)
  ∃ n : ℝ, n = ω^2 * R * (1 + μ^2) / (4 * π * g * μ * (1 + μ)) :=
by sorry

end ring_revolutions_before_stopping_l282_282648


namespace molecular_weight_CaO_l282_282323

theorem molecular_weight_CaO (m : ℕ -> ℝ) (h : m 7 = 392) : m 1 = 56 :=
sorry

end molecular_weight_CaO_l282_282323


namespace compute_expression_l282_282210

theorem compute_expression :
  (143 + 29) * 2 + 25 + 13 = 382 :=
by 
  sorry

end compute_expression_l282_282210


namespace complement_union_l282_282767

open Set

theorem complement_union (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 5, 6, 8})
  (hA : A = {1, 5, 8})(hB : B = {2}) :
  (U \ A) ∪ B = {0, 2, 3, 6} :=
by
  rw [hU, hA, hB]
  -- Intermediate steps would go here
  sorry

end complement_union_l282_282767


namespace stampsLeftover_l282_282439

-- Define the number of stamps each person has
def oliviaStamps : ℕ := 52
def parkerStamps : ℕ := 66
def quinnStamps : ℕ := 23

-- Define the album's capacity in stamps
def albumCapacity : ℕ := 15

-- Define the total number of leftovers
def totalLeftover : ℕ := (oliviaStamps + parkerStamps + quinnStamps) % albumCapacity

-- Define the theorem we want to prove
theorem stampsLeftover : totalLeftover = 6 := by
  sorry

end stampsLeftover_l282_282439


namespace functional_equation_solution_l282_282527

open Function

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f (x ^ 2 + f y) = y + f x ^ 2) → (∀ x : ℝ, f x = x) :=
by
  sorry

end functional_equation_solution_l282_282527


namespace range_of_a_l282_282986

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, |x - a| + |x - 1| ≤ 4) : -3 ≤ a ∧ a ≤ 5 := 
sorry

end range_of_a_l282_282986


namespace sequence_formula_l282_282984

theorem sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : ∀ n, S n = 2 * a n + 1) : 
  ∀ n, a n = -2 ^ (n - 1) := 
by 
  sorry

end sequence_formula_l282_282984


namespace company_workers_l282_282900

theorem company_workers (W : ℕ) (H1 : (1/3 : ℚ) * W = ((1/3 : ℚ) * W)) 
  (H2 : 0.20 * ((1/3 : ℚ) * W) = ((1/15 : ℚ) * W)) 
  (H3 : 0.40 * ((2/3 : ℚ) * W) = ((4/15 : ℚ) * W)) 
  (H4 : (4/15 : ℚ) * W + (4/15 : ℚ) * W = 160)
  : (W - 160 = 140) :=
by
  sorry

end company_workers_l282_282900


namespace geometric_series_sum_eq_l282_282512

theorem geometric_series_sum_eq :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 5
  (∀ S_n, S_n = a * (1 - r^n) / (1 - r) → S_n = 1 / 3) :=
by
  intro a r n S_n
  sorry

end geometric_series_sum_eq_l282_282512


namespace probability_blue_point_l282_282916

-- Definitions of the random points
def is_random_point (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 2

-- Definition of the condition for the probability problem
def condition (x y : ℝ) : Prop :=
  x < y ∧ y < 3 * x

-- Statement of the theorem
theorem probability_blue_point (x y : ℝ) (h1 : is_random_point x) (h2 : is_random_point y) :
  ∃ p : ℝ, (p = 1 / 3) ∧ (∃ (hx : x < y) (hy : y < 3 * x), x ≤ 2 ∧ 0 ≤ x ∧ y ≤ 2 ∧ 0 ≤ y) :=
by
  sorry

end probability_blue_point_l282_282916


namespace initial_non_electrified_part_l282_282656

variables (x y : ℝ)

def electrified_fraction : Prop :=
  x + y = 1 ∧ 2 * x + 0.75 * y = 1

theorem initial_non_electrified_part (h : electrified_fraction x y) : y = 4 / 5 :=
by {
  sorry
}

end initial_non_electrified_part_l282_282656


namespace original_volume_l282_282655

theorem original_volume (V : ℝ) (h1 : V > 0) 
    (h2 : (1/16) * V = 0.75) : V = 12 :=
by sorry

end original_volume_l282_282655


namespace parallelogram_probability_l282_282857

theorem parallelogram_probability (P Q R S : ℝ × ℝ) 
  (hP : P = (4, 2)) 
  (hQ : Q = (-2, -2)) 
  (hR : R = (-6, -6)) 
  (hS : S = (0, -2)) :
  let parallelogram_area := 24 -- given the computed area based on provided geometry
  let divided_area := parallelogram_area / 2
  let not_above_x_axis_area := divided_area
  (not_above_x_axis_area / parallelogram_area) = (1 / 2) :=
by
  sorry

end parallelogram_probability_l282_282857


namespace three_pizzas_needed_l282_282795

noncomputable def masha_pizza (p : Set String) : Prop :=
  "tomatoes" ∈ p ∧ "sausage" ∉ p

noncomputable def vanya_pizza (p : Set String) : Prop :=
  "mushrooms" ∈ p

noncomputable def dasha_pizza (p : Set String) : Prop :=
  "tomatoes" ∉ p

noncomputable def nikita_pizza (p : Set String) : Prop :=
  "tomatoes" ∈ p ∧ "mushrooms" ∉ p

noncomputable def igor_pizza (p : Set String) : Prop :=
  "mushrooms" ∉ p ∧ "sausage" ∈ p

theorem three_pizzas_needed (p1 p2 p3 : Set String) :
  (∃ p1, masha_pizza p1 ∧ vanya_pizza p1 ∧ dasha_pizza p1 ∧ nikita_pizza p1 ∧ igor_pizza p1) →
  (∃ p2, masha_pizza p2 ∧ vanya_pizza p2 ∧ dasha_pizza p2 ∧ nikita_pizza p2 ∧ igor_pizza p2) →
  (∃ p3, masha_pizza p3 ∧ vanya_pizza p3 ∧ dasha_pizza p3 ∧ nikita_pizza p3 ∧ igor_pizza p3) →
  ∀ p, ¬ ((masha_pizza p ∨ dasha_pizza p) ∧ vanya_pizza p ∧ (nikita_pizza p ∨ igor_pizza p)) :=
sorry

end three_pizzas_needed_l282_282795


namespace digits_are_different_probability_l282_282649

noncomputable def prob_diff_digits : ℚ :=
  let total := 999 - 100 + 1
  let same_digits := 9
  1 - (same_digits / total)

theorem digits_are_different_probability :
  prob_diff_digits = 99 / 100 :=
by
  sorry

end digits_are_different_probability_l282_282649


namespace basketball_cost_l282_282111

-- Initial conditions
def initial_amount : Nat := 50
def cost_jerseys (n price_per_jersey : Nat) : Nat := n * price_per_jersey
def cost_shorts : Nat := 8
def remaining_amount : Nat := 14

-- Derived total spent calculation
def total_spent (initial remaining : Nat) : Nat := initial - remaining
def known_cost (jerseys shorts : Nat) : Nat := jerseys + shorts

-- Prove the cost of the basketball
theorem basketball_cost :
  let jerseys := cost_jerseys 5 2
  let shorts := cost_shorts
  let total_spent := total_spent initial_amount remaining_amount
  let known_cost := known_cost jerseys shorts
  total_spent - known_cost = 18 := 
by
  sorry

end basketball_cost_l282_282111


namespace range_of_m_l282_282244

open Set

variable (m : ℝ)
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | m < x ∧ x < m + 9}
def complementR (A : Set ℝ) : Set ℝ := {x | x ≤ -2 ∨ x ≥ 3}

theorem range_of_m (H : (complementR A) ∩ (B m) = B m) : m ≤ -11 ∨ m ≥ 3 :=
by
  sorry

end range_of_m_l282_282244


namespace find_S_l282_282633

theorem find_S (x y : ℝ) (h : x + y = 4) : 
  ∃ S, (∀ x y, x + y = 4 → 3*x^2 + y^2 = 12) → S = 6 := 
by 
  sorry

end find_S_l282_282633


namespace altitude_not_integer_l282_282444

theorem altitude_not_integer (a b c : ℕ) (H : ℚ)
  (h1 : a ^ 2 + b ^ 2 = c ^ 2)
  (coprime_ab : Nat.gcd a b = 1)
  (coprime_bc : Nat.gcd b c = 1)
  (coprime_ca : Nat.gcd c a = 1) :
  ¬ ∃ H : ℕ, a * b = c * H := 
by
  sorry

end altitude_not_integer_l282_282444


namespace rhombus_side_length_l282_282463

theorem rhombus_side_length (s : ℝ) (h : 4 * s = 32) : s = 8 :=
by
  sorry

end rhombus_side_length_l282_282463


namespace ratio_of_areas_l282_282274

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {A B C M : V}

-- Define the collinearity condition point M in the triangle plane with respect to vectors AB and AC
def point_condition (A B C M : V) : Prop :=
  5 • (M - A) = (B - A) + 3 • (C - A)

-- Define an area ratio function
def area_ratio_triangles (A B C M : V) [AddCommGroup V] [Module ℝ V] : ℝ :=
  sorry  -- Implementation of area ratio comparison, abstracted out for the given problem statement

-- The theorem to prove
theorem ratio_of_areas (hM : point_condition A B C M) : area_ratio_triangles A B C M = 3 / 5 :=
sorry

end ratio_of_areas_l282_282274


namespace papaya_production_l282_282859

theorem papaya_production (P : ℕ)
  (h1 : 2 * P + 3 * 20 = 80) :
  P = 10 := 
by sorry

end papaya_production_l282_282859


namespace prob_multiple_of_98_l282_282400

open Set

-- Given set of numbers
def S := {6, 14, 21, 28, 35, 42, 49}

-- A function to check divisibility by 98
def divisible_by_98 (a b : ℕ) : Prop :=
  98 ∣ (a * b)

-- A condition for the two numbers being distinct elements from the set S
def distinct_mem (a b : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ a ≠ b

-- The main theorem stating the desired probability
theorem prob_multiple_of_98 :
  let outcomes := { (a, b) | distinct_mem a b } in
  let favorable := { (a, b) ∈ outcomes | divisible_by_98 a b } in
  (favorable.to_finset.card : ℚ) / outcomes.to_finset.card = 1 / 7 :=
by sorry

end prob_multiple_of_98_l282_282400


namespace fiona_total_evaluations_l282_282218

theorem fiona_total_evaluations :
  ∀ n k : ℕ, (n = 13) ∧ (k = 2) → 3 * (Nat.choose n k) = 234 :=
by
  intros n k h
  cases h
  sorry

end fiona_total_evaluations_l282_282218


namespace jordan_wins_two_games_l282_282704

theorem jordan_wins_two_games 
  (Peter_wins : ℕ) 
  (Peter_losses : ℕ)
  (Emma_wins : ℕ) 
  (Emma_losses : ℕ)
  (Jordan_losses : ℕ) 
  (hPeter : Peter_wins = 5)
  (hPeterL : Peter_losses = 4)
  (hEmma : Emma_wins = 4)
  (hEmmaL : Emma_losses = 5)
  (hJordanL : Jordan_losses = 2) : ∃ (J : ℕ), J = 2 :=
by
  -- The proof will go here
  sorry

end jordan_wins_two_games_l282_282704


namespace einstein_needs_more_money_l282_282159

-- Definitions based on conditions
def pizza_price : ℝ := 12
def fries_price : ℝ := 0.3
def soda_price : ℝ := 2
def pizzas_sold : ℕ := 15
def fries_sold : ℕ := 40
def sodas_sold : ℕ := 25
def goal : ℝ := 500

-- Total amount raised calculation
def total_raised : ℝ :=
  (pizzas_sold * pizza_price) +
  (fries_sold * fries_price) +
  (sodas_sold * soda_price)

-- Proof statement
theorem einstein_needs_more_money : goal - total_raised = 258 :=
by
  sorry

end einstein_needs_more_money_l282_282159


namespace even_function_composition_is_even_l282_282584

-- Let's define what it means for a function to be even
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- The main theorem stating the evenness of the composition of an even function
theorem even_function_composition_is_even {f : ℝ → ℝ} (h : even_function f) :
  even_function (λ x, f (f x)) :=
by
  intros x
  have : f (-x) = f x := h x
  rw [←this, h (-x)]
  sorry

end even_function_composition_is_even_l282_282584


namespace superhero_vs_supervillain_distance_l282_282782

-- Definitions expressing the conditions
def superhero_speed (miles : ℕ) (minutes : ℕ) := (10 : ℕ) / (4 : ℕ)
def supervillain_speed (miles_per_hour : ℕ) := (100 : ℕ)

-- Distance calculation in 60 minutes
def superhero_distance_in_hour := 60 * superhero_speed 10 4
def supervillain_distance_in_hour := supervillain_speed 100

-- Proof statement
theorem superhero_vs_supervillain_distance :
  superhero_distance_in_hour - supervillain_distance_in_hour = (50 : ℕ) :=
by
  sorry

end superhero_vs_supervillain_distance_l282_282782


namespace smallest_area_2020th_square_l282_282919

theorem smallest_area_2020th_square (n : ℕ) :
  (∃ n : ℕ, n^2 > 2019 ∧ ∃ A : ℕ, A = n^2 - 2019 ∧ A ≠ 1) →
  (∃ A : ℕ, A = n^2 - 2019 ∧ A ≠ 1 ∧ A = 6) :=
sorry

end smallest_area_2020th_square_l282_282919


namespace probability_inequality_up_to_99_l282_282777

theorem probability_inequality_up_to_99 :
  (∀ x : ℕ, 1 ≤ x ∧ x < 100 → (2^x / x!) > x^2) →
    (∃ n : ℕ, (1 ≤ n ∧ n < 100) ∧ (2^n / n!) > n^2) →
      ∃ p : ℚ, p = 1/99 :=
by
  sorry

end probability_inequality_up_to_99_l282_282777


namespace problem_condition_relationship_l282_282587

theorem problem_condition_relationship (x : ℝ) :
  (x^2 - x - 2 > 0) → (|x - 1| > 1) := 
sorry

end problem_condition_relationship_l282_282587


namespace number_of_sandwiches_l282_282922

-- Definitions based on the conditions in the problem
def sandwich_cost : Nat := 3
def water_cost : Nat := 2
def total_cost : Nat := 11

-- Lean statement to prove the number of sandwiches bought is 3
theorem number_of_sandwiches (S : Nat) (h : sandwich_cost * S + water_cost = total_cost) : S = 3 :=
by
  sorry

end number_of_sandwiches_l282_282922


namespace intersection_is_singleton_l282_282387

-- Definitions of sets M and N
def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

-- The stated proposition we need to prove
theorem intersection_is_singleton :
  M ∩ N = {(3, -1)} :=
by {
  sorry
}

end intersection_is_singleton_l282_282387


namespace not_divisible_by_n_l282_282728

theorem not_divisible_by_n (n : ℕ) (h : n > 1) : ¬ (n ∣ (2^n - 1)) :=
by
  sorry

end not_divisible_by_n_l282_282728


namespace smallest_fraction_num_greater_than_four_fifths_with_2_digit_nums_l282_282354

theorem smallest_fraction_num_greater_than_four_fifths_with_2_digit_nums :
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ (a : ℚ) / b > 4 / 5 ∧ Int.gcd a b = 1 ∧ a = 77 :=
by {
    sorry
}

end smallest_fraction_num_greater_than_four_fifths_with_2_digit_nums_l282_282354


namespace blue_pieces_correct_l282_282884

def total_pieces : ℕ := 3409
def red_pieces : ℕ := 145
def blue_pieces : ℕ := total_pieces - red_pieces

theorem blue_pieces_correct : blue_pieces = 3264 := by
  sorry

end blue_pieces_correct_l282_282884


namespace polynomial_remainder_l282_282804

noncomputable def divisionRemainder (f g : Polynomial ℝ) : Polynomial ℝ := Polynomial.modByMonic f g

theorem polynomial_remainder :
  divisionRemainder (Polynomial.X ^ 5 + 2) (Polynomial.X ^ 2 - 4 * Polynomial.X + 7) = -29 * Polynomial.X - 54 :=
by
  sorry

end polynomial_remainder_l282_282804


namespace last_three_digits_of_primitive_polynomial_pairs_l282_282128

def is_primitive (p : Polynomial ℤ) : Prop :=
  p.coeffs.gcd = 1

def valid_coeff (n : ℤ) : Prop :=
  n ∈ {1, 2, 3, 4, 5}

theorem last_three_digits_of_primitive_polynomial_pairs :
  let polys := {p : Polynomial ℤ | (∀ i ∈ p.support, valid_coeff (p.coeff i)) ∧ is_primitive p}
  let N := (polys.card)^2
  N % 1000 = 689 :=
sorry

end last_three_digits_of_primitive_polynomial_pairs_l282_282128


namespace cos_double_angle_identity_l282_282249

theorem cos_double_angle_identity (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = 1 / 3) :
  Real.cos (2 * Real.pi / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cos_double_angle_identity_l282_282249


namespace plant_species_numbering_impossible_l282_282787

theorem plant_species_numbering_impossible :
  ∀ (n m : ℕ), 2 ≤ n ∨ n ≤ 20000 ∧ 2 ≤ m ∨ m ≤ 20000 ∧ n ≠ m → 
  ∃ x y : ℕ, 2 ≤ x ∨ x ≤ 20000 ∧ 2 ≤ y ∨ y ≤ 20000 ∧ x ≠ y ∧
  (∀ k : ℕ, gcd x k = gcd n k ∧ gcd y k = gcd m k) :=
  by sorry

end plant_species_numbering_impossible_l282_282787


namespace smallest_n_l282_282604

theorem smallest_n (n : ℕ) : (n > 0) ∧ (2^n % 30 = 1) → n = 4 :=
by
  intro h
  sorry

end smallest_n_l282_282604


namespace binomial_variance_expectation_ratio_l282_282681

open ProbabilityTheory

variable {Ω : Type*} [MeasureSpace Ω]

noncomputable def xi (n p : ℝ) : MeasureTheory.Measure Ω :=
  binomialDistribution n p 

theorem binomial_variance_expectation_ratio (n p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  let ξ := xi n p
  (variance ξ) ^ 2 / (expectation ξ) ^ 2 = (1 - p) ^ 2 := by
  sorry

end binomial_variance_expectation_ratio_l282_282681


namespace Julia_watch_collection_l282_282422

section
variable (silver_watches : ℕ) (bronze_watches : ℕ) (gold_watches : ℕ) (total_watches : ℕ)

theorem Julia_watch_collection :
  silver_watches = 20 →
  bronze_watches = 3 * silver_watches →
  gold_watches = 10 * (silver_watches + bronze_watches) / 100 →
  total_watches = silver_watches + bronze_watches + gold_watches →
  total_watches = 88 :=
by
  intros
  sorry
end

end Julia_watch_collection_l282_282422


namespace largest_possible_sum_l282_282427

theorem largest_possible_sum (clubsuit heartsuit : ℕ) (h₁ : clubsuit * heartsuit = 48) (h₂ : Even clubsuit) : 
  clubsuit + heartsuit ≤ 26 :=
sorry

end largest_possible_sum_l282_282427


namespace total_number_of_items_in_base10_l282_282500

theorem total_number_of_items_in_base10 : 
  let clay_tablets := (2 * 5^0 + 3 * 5^1 + 4 * 5^2 + 1 * 5^3)
  let bronze_sculptures := (1 * 5^0 + 4 * 5^1 + 0 * 5^2 + 2 * 5^3)
  let stone_carvings := (2 * 5^0 + 3 * 5^1 + 2 * 5^2)
  let total_items := clay_tablets + bronze_sculptures + stone_carvings
  total_items = 580 := by
  sorry

end total_number_of_items_in_base10_l282_282500


namespace n_must_be_even_l282_282290

open Nat

-- Define the system of equations:
def equation (n : ℕ) (x : ℕ → ℤ) : Prop :=
  (∀ i, 2 ≤ i ∧ i ≤ n - 1 → (-x (i-1) + 2 * x i - x (i+1) = 1)) ∧
  (2 * x 1 - x 2 = 1) ∧
  (∀ i, 1 ≤ i ∧ i ≤ n → x i > 0)

-- Define the last equation separately due to its unique form:
def last_equation (n : ℕ) (x : ℕ → ℤ) : Prop :=
  (n ≥ 1 → -x (n-1) + 2 * x n = 1)

-- The theorem to prove that n must be even:
theorem n_must_be_even (n : ℕ) (x : ℕ → ℤ) : 
  equation n x → last_equation n x → Even n :=
by
  intros h₁ h₂
  sorry

end n_must_be_even_l282_282290


namespace differential_equation_approx_solution_l282_282411

open Real

noncomputable def approximate_solution (x : ℝ) : ℝ := 0.1 * exp (x ^ 2 / 2)

theorem differential_equation_approx_solution :
  ∀ (x : ℝ), -1/2 ≤ x ∧ x ≤ 1/2 →
  ∀ (y : ℝ), -1/2 ≤ y ∧ y ≤ 1/2 →
  abs (approximate_solution x - y) < 1 / 650 :=
sorry

end differential_equation_approx_solution_l282_282411


namespace solve_equation_l282_282863

theorem solve_equation (x : ℝ) (hx : (x + 1) ≠ 0) :
  (x = -3 / 4) ∨ (x = -1) ↔ (x^3 + x^2 + x + 1) / (x + 1) = x^2 + 4 * x + 4 :=
by
  sorry

end solve_equation_l282_282863


namespace smallest_possible_n_l282_282150

theorem smallest_possible_n :
  ∃ n : ℕ, (∃ (A B C D : ℕ), (a = 110 * A ∧ b = 110 * B ∧ c = 110 * C ∧ d = 110 * D ∧
                        gcd A B C D = 1 ∧ lcm A B C D * 110 = n ∧
                        ∃ (k : ℕ), count_quadruplets_with_gcd_lcm 110 110000 = k) ∧ 
              n = 198000) := 
sorry

end smallest_possible_n_l282_282150


namespace people_per_team_l282_282928

theorem people_per_team 
  (managers : ℕ) (employees : ℕ) (teams : ℕ) 
  (h1 : managers = 23) (h2 : employees = 7) (h3 : teams = 6) :
  (managers + employees) / teams = 5 :=
by
  sorry

end people_per_team_l282_282928


namespace smallest_integer_solution_l282_282324

theorem smallest_integer_solution (x : ℤ) : 
  (10 * x * x - 40 * x + 36 = 0) → x = 2 :=
sorry

end smallest_integer_solution_l282_282324


namespace second_train_speed_l282_282887

theorem second_train_speed :
  ∃ v : ℝ, 
  (∀ t : ℝ, 20 * t = v * t + 50) ∧
  (∃ t : ℝ, 20 * t + v * t = 450) →
  v = 16 :=
by
  sorry

end second_train_speed_l282_282887


namespace dog_age_64_human_years_l282_282312

def dog_years (human_years : ℕ) : ℕ :=
if human_years = 0 then
  0
else if human_years = 1 then
  1
else if human_years = 2 then
  2
else
  2 + (human_years - 2) / 5

theorem dog_age_64_human_years : dog_years 64 = 10 :=
by 
    sorry

end dog_age_64_human_years_l282_282312


namespace hyperbola_params_l282_282870

theorem hyperbola_params (a b h k : ℝ) (h_positivity : a > 0 ∧ b > 0)
  (asymptote_1 : ∀ x : ℝ, ∃ y : ℝ, y = (3/2) * x + 4)
  (asymptote_2 : ∀ x : ℝ, ∃ y : ℝ, y = -(3/2) * x + 2)
  (passes_through : ∃ x y : ℝ, x = 2 ∧ y = 8 ∧ (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1) 
  (standard_form : ∀ x y : ℝ, ((y - k)^2 / a^2 - (x - h)^2 / b^2 = 1)) : 
  a + h = 7/3 := sorry

end hyperbola_params_l282_282870


namespace bill_due_in_months_l282_282614

noncomputable def true_discount_time (TD A R : ℝ) : ℝ :=
  let P := A - TD
  let T := TD / (P * R / 100)
  12 * T

theorem bill_due_in_months :
  ∀ (TD A R : ℝ), TD = 189 → A = 1764 → R = 16 →
  abs (true_discount_time TD A R - 10.224) < 1 :=
by
  intros TD A R hTD hA hR
  sorry

end bill_due_in_months_l282_282614


namespace stock_initial_value_l282_282949

theorem stock_initial_value (V : ℕ) (h : ∀ n ≤ 99, V + n = 200 - (99 - n)) : V = 101 :=
sorry

end stock_initial_value_l282_282949


namespace cubic_geometric_sequence_conditions_l282_282808

-- Conditions from the problem
def cubic_eq (a b c x : ℝ) : Prop := x^3 + a * x^2 + b * x + c = 0

-- The statement to be proven
theorem cubic_geometric_sequence_conditions (a b c : ℝ) :
  (∃ x q : ℝ, q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1 ∧ 
    cubic_eq a b c x ∧ cubic_eq a b c (x*q) ∧ cubic_eq a b c (x*q^2)) → 
  (b^3 = a^3 * c ∧ c ≠ 0 ∧ -a^3 < c ∧ c < a^3 / 27 ∧ a < m ∧ m < - a / 3) :=
by 
  sorry

end cubic_geometric_sequence_conditions_l282_282808


namespace range_of_m_l282_282824

def f (x : ℝ) : ℝ := x^3 + x

theorem range_of_m (m : ℝ) (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < Real.pi / 2) :
  (f (m * Real.sin θ) + f (1 - m) > 0) ↔ (m ≤ 1) :=
sorry

end range_of_m_l282_282824


namespace inequality_solution_set_l282_282749

theorem inequality_solution_set (x : ℝ) :
  ((1 / 2 - x) * (x - 1 / 3) > 0) ↔ (1 / 3 < x ∧ x < 1 / 2) :=
by 
  sorry

end inequality_solution_set_l282_282749


namespace correct_answer_is_B_l282_282331

-- Definitions for each set of line segments
def setA := (2, 2, 4)
def setB := (8, 6, 3)
def setC := (2, 6, 3)
def setD := (11, 4, 6)

-- Triangle inequality theorem checking function
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Statements to verify each set
lemma check_setA : ¬ is_triangle 2 2 4 := by sorry
lemma check_setB : is_triangle 8 6 3 := by sorry
lemma check_setC : ¬ is_triangle 2 6 3 := by sorry
lemma check_setD : ¬ is_triangle 11 4 6 := by sorry

-- Final theorem combining all checks to match the given problem
theorem correct_answer_is_B : 
  ¬ is_triangle 2 2 4 ∧ is_triangle 8 6 3 ∧ ¬ is_triangle 2 6 3 ∧ ¬ is_triangle 11 4 6 :=
by sorry

end correct_answer_is_B_l282_282331


namespace inequality_solution_l282_282892

theorem inequality_solution (x : ℝ) : (3 * x^2 - 4 * x - 4 < 0) ↔ (-2/3 < x ∧ x < 2) :=
sorry

end inequality_solution_l282_282892


namespace vector_normalization_condition_l282_282581

variables {a b : ℝ} -- Ensuring that Lean understands ℝ refers to real numbers and specifically vectors in ℝ before using it in the next parts.

-- Definitions of the vector variables
variables (a b : ℝ) (ab_non_zero : a ≠ 0 ∧ b ≠ 0)

-- Required statement
theorem vector_normalization_condition (a b : ℝ) 
(h₀ : a ≠ 0 ∧ b ≠ 0) :
  (a / abs a = b / abs b) ↔ (a = 2 * b) :=
sorry

end vector_normalization_condition_l282_282581


namespace investment_amount_l282_282981

-- Conditions and given problem rewrite in Lean 4
theorem investment_amount (P y : ℝ) (h1 : P * y * 2 / 100 = 500) (h2 : P * (1 + y / 100) ^ 2 - P = 512.50) : P = 5000 :=
sorry

end investment_amount_l282_282981


namespace least_positive_integer_condition_l282_282481

theorem least_positive_integer_condition (n : ℕ) :
  (∀ d ∈ [2, 3, 4, 5, 6, 7, 8, 9, 11], n % d = 1) → n = 10396 := 
by
  sorry

end least_positive_integer_condition_l282_282481


namespace solution_to_problem_l282_282933

def problem_statement : Prop :=
  (3^202 + 7^203)^2 - (3^202 - 7^203)^2 = 59 * 10^202

theorem solution_to_problem : problem_statement := 
  by sorry

end solution_to_problem_l282_282933


namespace num_teachers_in_Oxford_High_School_l282_282598

def classes : Nat := 15
def students_per_class : Nat := 20
def principals : Nat := 1
def total_people : Nat := 349

theorem num_teachers_in_Oxford_High_School : 
  ∃ (teachers : Nat), teachers = total_people - (classes * students_per_class + principals) :=
by
  use 48
  sorry

end num_teachers_in_Oxford_High_School_l282_282598


namespace earnings_from_roosters_l282_282942

-- Definitions from the conditions
def price_per_kg : Float := 0.50
def weight_of_rooster1 : Float := 30.0
def weight_of_rooster2 : Float := 40.0

-- The theorem we need to prove (mathematically equivalent proof problem)
theorem earnings_from_roosters (p : Float := price_per_kg)
                               (w1 : Float := weight_of_rooster1)
                               (w2 : Float := weight_of_rooster2) :
  p * w1 + p * w2 = 35.0 := 
by {
  sorry
}

end earnings_from_roosters_l282_282942


namespace inscribed_circle_radius_l282_282736

theorem inscribed_circle_radius (r : ℝ) (R : ℝ) (angle : ℝ):
  R = 6 → angle = 2 * Real.pi / 3 → r = (6 * Real.sqrt 3) / 5 :=
by
  sorry

end inscribed_circle_radius_l282_282736


namespace ratio_of_points_l282_282440

theorem ratio_of_points (B J S : ℕ) 
  (h1 : B = J + 20) 
  (h2 : B + J + S = 160) 
  (h3 : B = 45) : 
  B / S = 1 / 2 :=
  sorry

end ratio_of_points_l282_282440


namespace calculate_expression_l282_282930

theorem calculate_expression :
  5 * 6 - 2 * 3 + 7 * 4 + 9 * 2 = 70 := by
  sorry

end calculate_expression_l282_282930


namespace cyclic_points_l282_282856

variables {A B C A1 B1 C1 A2 C2 : EuclideanGeometry.Point}
variable (circumcircleABC : EuclideanGeometry.Circle)
variables (P K M : EuclideanGeometry.Point)

-- Definitions in line with conditions (a):
def segments_intersect_at_point (P : EuclideanGeometry.Point) :=
  EuclideanGeometry.LineThrough A A1 = EuclideanGeometry.LineThrough B B1 ∧ 
  EuclideanGeometry.LineThrough B B1 = EuclideanGeometry.LineThrough C C1 ∧ 
  EuclideanGeometry.LineThrough A A1 = EuclideanGeometry.LineThrough C C1

def intersection_with_circumcircle (B1A1 := EuclideanGeometry.LineThrough B1 A1) : EuclideanGeometry.Point :=
  EuclideanGeometry.line_circle_intersection circumcircleABC B1A1

def intersection_with_circumcircle' (B1C1 := EuclideanGeometry.LineThrough B1 C1) : EuclideanGeometry.Point :=
  EuclideanGeometry.line_circle_intersection circumcircleABC B1C1

-- Assuming A2 and C2 are the respective intersections from B1A1 and B1C1 with the circumcircle
axiom A2_def : intersection_with_circumcircle = A2
axiom C2_def : intersection_with_circumcircle' = C2

-- The goal (Proof statement in Lean):
theorem cyclic_points (hP : segments_intersect_at_point P) 
                        (hA2 : intersection_with_circumcircle = A2)
                        (hC2 : intersection_with_circumcircle' = C2)
                        (hK : EuclideanGeometry.LineThrough A2 C2 ∩ EuclideanGeometry.LineThrough B B1 = K)
                        (hM : EuclideanGeometry.midpoint A2 C2 = M) :
  EuclideanGeometry.CircleThrough4 A C K M :=
by sorry

end cyclic_points_l282_282856


namespace part1_part2_l282_282526

def op (a b : ℤ) := 2 * a - 3 * b

theorem part1 : op (-2) 3 = -13 := 
by
  -- Proof omitted
  sorry

theorem part2 (x : ℤ) : 
  let A := op (3 * x - 2) (x + 1)
  let B := op (-3 / 2 * x + 1) (-1 - 2 * x)
  B > A :=
by
  -- Proof omitted
  sorry

end part1_part2_l282_282526


namespace lea_total_cost_example_l282_282590

/-- Léa bought one book for $16, three binders for $2 each, and six notebooks for $1 each. -/
def total_cost (book_cost binders_cost notebooks_cost : ℕ) : ℕ :=
  book_cost + binders_cost + notebooks_cost

/-- Given the individual costs, prove the total cost of Léa's purchases is $28. -/
theorem lea_total_cost_example : total_cost 16 (3 * 2) (6 * 1) = 28 := by
  sorry

end lea_total_cost_example_l282_282590


namespace min_apples_l282_282751

theorem min_apples :
  ∃ N : ℕ, 
  (N % 3 = 2) ∧ 
  (N % 4 = 2) ∧ 
  (N % 5 = 2) ∧ 
  (N = 62) :=
by
  sorry

end min_apples_l282_282751


namespace min_value_a_plus_b_plus_c_l282_282966

theorem min_value_a_plus_b_plus_c 
  (a b c : ℕ) 
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_c_pos : c > 0)
  (x1 x2 : ℝ)
  (hx1_neg : -1 < x1)
  (hx1_pos : x1 < 0)
  (hx2_neg : 0 < x2)
  (hx2_pos : x2 < 1)
  (h_distinct : x1 ≠ x2)
  (h_eqn_x1 : a * x1^2 + b * x1 + c = 0)
  (h_eqn_x2 : a * x2^2 + b * x2 + c = 0) :
  a + b + c = 11 :=
sorry

end min_value_a_plus_b_plus_c_l282_282966


namespace percentage_of_a_l282_282836

theorem percentage_of_a (x a : ℝ) (paise_in_rupee : ℝ := 100) (a_value : a = 160 * paise_in_rupee) (h : (x / 100) * a = 80) : x = 0.5 :=
by sorry

end percentage_of_a_l282_282836


namespace julia_watches_l282_282419

theorem julia_watches (silver_watches bronze_multiplier : ℕ)
    (total_watches_percent_to_buy total_percent bronze_multiplied : ℕ) :
    silver_watches = 20 →
    bronze_multiplier = 3 →
    total_watches_percent_to_buy = 10 →
    total_percent = 100 → 
    bronze_multiplied = (silver_watches * bronze_multiplier) →
    let bronze_watches := silver_watches * bronze_multiplier,
        total_watches_before := silver_watches + bronze_watches,
        gold_watches := total_watches_percent_to_buy * total_watches_before / total_percent,
        total_watches_after := total_watches_before + gold_watches
    in
    total_watches_after = 88 :=
by
    intros silver_watches_def bronze_multiplier_def total_watches_percent_to_buy_def
    total_percent_def bronze_multiplied_def
    have bronze_watches := silver_watches * bronze_multiplier
    have total_watches_before := silver_watches + bronze_watches
    have gold_watches := total_watches_percent_to_buy * total_watches_before / total_percent
    have total_watches_after := total_watches_before + gold_watches
    simp [bronze_watches, total_watches_before, gold_watches, total_watches_after]
    exact sorry

end julia_watches_l282_282419


namespace max_oranges_donated_l282_282917

theorem max_oranges_donated (N : ℕ) : ∃ n : ℕ, n < 7 ∧ (N % 7 = n) ∧ n = 6 :=
by
  sorry

end max_oranges_donated_l282_282917


namespace zeros_in_decimal_representation_l282_282087

def term_decimal_zeros (x : ℚ) : ℕ := sorry  -- Function to calculate the number of zeros in the terminating decimal representation.

theorem zeros_in_decimal_representation :
  term_decimal_zeros (1 / (2^7 * 5^9)) = 8 :=
sorry

end zeros_in_decimal_representation_l282_282087


namespace total_boxes_correct_l282_282285

noncomputable def friday_boxes : ℕ := 40

noncomputable def saturday_boxes : ℕ := 2 * friday_boxes - 10

noncomputable def sunday_boxes : ℕ := saturday_boxes / 2

noncomputable def monday_boxes : ℕ := 
  let extra_boxes := (25 * sunday_boxes + 99) / 100 -- (25/100) * sunday_boxes rounded to nearest integer
  sunday_boxes + extra_boxes

noncomputable def total_boxes : ℕ := 
  friday_boxes + saturday_boxes + sunday_boxes + monday_boxes

theorem total_boxes_correct : total_boxes = 189 := by
  sorry

end total_boxes_correct_l282_282285


namespace cost_of_eraser_l282_282975

theorem cost_of_eraser
  (total_money: ℕ)
  (n_sharpeners n_notebooks n_erasers n_highlighters: ℕ)
  (price_sharpener price_notebook price_highlighter: ℕ)
  (heaven_spent brother_spent remaining_money final_spent: ℕ) :
  total_money = 100 →
  n_sharpeners = 2 →
  price_sharpener = 5 →
  n_notebooks = 4 →
  price_notebook = 5 →
  n_highlighters = 1 →
  price_highlighter = 30 →
  heaven_spent = n_sharpeners * price_sharpener + n_notebooks * price_notebook →
  brother_spent = 30 →
  remaining_money = total_money - heaven_spent →
  final_spent = remaining_money - brother_spent →
  final_spent = 40 →
  n_erasers = 10 →
  ∀ cost_per_eraser: ℕ, final_spent = cost_per_eraser * n_erasers →
  cost_per_eraser = 4 := by
  intros h_total_money h_n_sharpeners h_price_sharpener h_n_notebooks h_price_notebook
    h_n_highlighters h_price_highlighter h_heaven_spent h_brother_spent h_remaining_money
    h_final_spent h_n_erasers cost_per_eraser h_final_cost
  sorry

end cost_of_eraser_l282_282975


namespace three_digit_numbers_not_multiples_of_3_or_11_l282_282560

def count_multiples (a b : ℕ) (lower upper : ℕ) : ℕ :=
  (upper - lower) / b + 1

theorem three_digit_numbers_not_multiples_of_3_or_11 : 
  let total := 900
  let multiples_3 := count_multiples 3 3 102 999
  let multiples_11 := count_multiples 11 11 110 990
  let multiples_33 := count_multiples 33 33 132 990
  let multiples_3_or_11 := multiples_3 + multiples_11 - multiples_33
  total - multiples_3_or_11 = 546 := 
by 
  sorry

end three_digit_numbers_not_multiples_of_3_or_11_l282_282560


namespace common_divisor_is_19_l282_282861

theorem common_divisor_is_19 (a d : ℤ) (h1 : d ∣ (35 * a + 57)) (h2 : d ∣ (45 * a + 76)) : d = 19 :=
sorry

end common_divisor_is_19_l282_282861


namespace combustion_moles_l282_282399

-- Chemical reaction definitions
def balanced_equation : Prop :=
  ∀ (CH4 Cl2 O2 CO2 HCl H2O : ℝ),
  1 * CH4 + 4 * Cl2 + 4 * O2 = 1 * CO2 + 4 * HCl + 2 * H2O

-- Moles of substances
def moles_CH4 := 24
def moles_Cl2 := 48
def moles_O2 := 96
def moles_CO2 := 24
def moles_HCl := 48
def moles_H2O := 48

-- Prove the conditions based on the balanced equation
theorem combustion_moles :
  balanced_equation →
  (moles_O2 = 4 * moles_CH4) ∧
  (moles_H2O = 2 * moles_CH4) :=
by {
  sorry
}

end combustion_moles_l282_282399


namespace infinitely_many_gt_sqrt_l282_282477

open Real

noncomputable def sequences := ℕ → ℕ × ℕ

def strictly_increasing_ratios (seq : sequences) : Prop :=
  ∀ n : ℕ, 0 < n → (seq (n + 1)).2 / (seq (n + 1)).1 > (seq n).2 / (seq n).1

theorem infinitely_many_gt_sqrt (seq : sequences) 
  (positive_integers : ∀ n : ℕ, (seq n).1 > 0 ∧ (seq n).2 > 0) 
  (inc_ratios : strictly_increasing_ratios seq) :
  ∃ᶠ n in at_top, (seq n).2 > sqrt n :=
sorry

end infinitely_many_gt_sqrt_l282_282477


namespace soda_cans_purchase_l282_282452

noncomputable def cans_of_soda (S Q D : ℕ) : ℕ :=
  10 * D * S / Q

theorem soda_cans_purchase (S Q D : ℕ) :
  (1 : ℕ) * 10 * D / Q = (10 * D * S) / Q := by
  sorry

end soda_cans_purchase_l282_282452


namespace general_term_formula_sum_first_n_terms_l282_282962

noncomputable def a_n (n : ℕ) : ℕ := 2^(n - 1)

def S (n : ℕ) : ℕ := n * (2^(n - 1))  -- Placeholder function for the sum of the first n terms

theorem general_term_formula (a_3_eq_2a_2 : 2^(3 - 1) = 2 * 2^(2 - 1)) (S3_eq_7 : S 3 = 7) :
  ∀ n, a_n n = 2^(n - 1) :=
sorry

def T (n : ℕ) : ℕ := 4 - ((4 + 2 * n) / 2^n) -- Placeholder function for calculating T_n

theorem sum_first_n_terms (a_3_eq_2a_2 : 2^(3 - 1) = 2 * 2^(2 - 1)) (S3_eq_7 : S 3 = 7) :
  ∀ n, T n = 4 - ((4 + 2*n) / 2^n) :=
sorry

end general_term_formula_sum_first_n_terms_l282_282962


namespace area_enclosed_by_curve_l282_282457

theorem area_enclosed_by_curve :
  let s : ℝ := 3
  let arc_length : ℝ := (3 * Real.pi) / 4
  let octagon_area : ℝ := (1 + Real.sqrt 2) * s^2
  let sector_area : ℝ := (3 / 8) * Real.pi
  let total_area : ℝ := 8 * sector_area + octagon_area
  total_area = 9 + 9 * Real.sqrt 2 + 3 * Real.pi :=
by
  let s := 3
  let arc_length := (3 * Real.pi) / 4
  let r := arc_length / ((3 * Real.pi) / 4)
  have r_eq : r = 1 := by
    sorry
  let full_circle_area := Real.pi * r^2
  let sector_area := (3 / 8) * Real.pi
  have sector_area_eq : sector_area = (3 / 8) * Real.pi := by
    sorry
  let total_sector_area := 8 * sector_area
  have total_sector_area_eq : total_sector_area = 3 * Real.pi := by
    sorry
  let octagon_area := (1 + Real.sqrt 2) * s^2
  have octagon_area_eq : octagon_area = 9 * (1 + Real.sqrt 2) := by
    sorry
  let total_area := total_sector_area + octagon_area
  have total_area_eq : total_area = 9 + 9 * Real.sqrt 2 + 3 * Real.pi := by
    sorry
  exact total_area_eq

end area_enclosed_by_curve_l282_282457


namespace zero_of_sum_of_squares_eq_zero_l282_282732

theorem zero_of_sum_of_squares_eq_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by
  sorry

end zero_of_sum_of_squares_eq_zero_l282_282732


namespace correct_calculation_l282_282635

theorem correct_calculation (x : ℝ) : (2 * x^5) / (-x)^3 = -2 * x^2 :=
by sorry

end correct_calculation_l282_282635


namespace velocity_zero_at_t_eq_2_l282_282745

noncomputable def motion_equation (t : ℝ) : ℝ := -4 * t^3 + 48 * t

theorem velocity_zero_at_t_eq_2 :
  (exists t : ℝ, t > 0 ∧ deriv (motion_equation) t = 0) :=
by
  sorry

end velocity_zero_at_t_eq_2_l282_282745


namespace more_white_animals_than_cats_l282_282203

theorem more_white_animals_than_cats (C W WC : ℕ) 
  (h1 : WC = C / 3) 
  (h2 : WC = W / 6) : W = 2 * C :=
by {
  sorry
}

end more_white_animals_than_cats_l282_282203


namespace sum_of_areas_of_triangles_l282_282931

noncomputable def triangle_sum_of_box (a b c : ℝ) :=
  let face_triangles_area := 4 * ((a * b + a * c + b * c) / 2)
  let perpendicular_triangles_area := 4 * ((a * c + b * c) / 2)
  let oblique_triangles_area := 8 * Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))
  face_triangles_area + perpendicular_triangles_area + oblique_triangles_area

theorem sum_of_areas_of_triangles :
  triangle_sum_of_box 2 3 4 = 168 + k * Real.sqrt p := sorry

end sum_of_areas_of_triangles_l282_282931


namespace integer_values_of_x_in_triangle_l282_282609

theorem integer_values_of_x_in_triangle (x : ℝ) :
  (x + 14 > 38 ∧ x + 38 > 14 ∧ 14 + 38 > x) → 
  ∃ (n : ℕ), n = 27 ∧ ∀ m : ℕ, (24 < m ∧ m < 52 ↔ (m : ℝ) > 24 ∧ (m : ℝ) < 52) :=
by {
  sorry
}

end integer_values_of_x_in_triangle_l282_282609


namespace prime_in_A_l282_282453

def A (n : ℕ) : Prop :=
  ∃ a b : ℤ, b ≠ 0 ∧ n = a^2 + 2 * b^2

theorem prime_in_A {p : ℕ} (h_prime : Nat.Prime p) (h_p2_in_A : A (p^2)) : A p :=
sorry

end prime_in_A_l282_282453


namespace limit_expression_l282_282929

theorem limit_expression :
  (∀ (n : ℕ), ∃ l : ℝ, 
    ∀ ε > 0, ∃ N : ℕ, n > N → 
      abs (( (↑(n) + 1)^3 - (↑(n) - 1)^3) / ((↑(n) + 1)^2 + (↑(n) - 1)^2) - l) < ε) 
  → l = 3 :=
sorry

end limit_expression_l282_282929


namespace find_x_l282_282227

theorem find_x (n : ℕ) (h_odd : n % 2 = 1) (h_factors : ∃ (p1 p2 p3 : ℕ), p1.prime ∧ p2.prime ∧ p3.prime ∧ (7^n + 1) = p1 * p2 * p3 ∧ (p1 = 2 ∨ p2 = 2 ∨ p3 = 2) ∧ (p1 = 11 ∨ p2 = 11 ∨ p3 = 11)) :
  7^n + 1 = 16808 :=
sorry

end find_x_l282_282227


namespace trig_identity_l282_282676

theorem trig_identity (θ : ℝ) (h₁ : Real.tan θ = 2) :
  2 * Real.cos θ / (Real.sin (Real.pi / 2 + θ) + Real.sin (Real.pi + θ)) = -2 :=
by
  sorry

end trig_identity_l282_282676


namespace smallest_five_digit_divisible_by_2_3_8_9_l282_282032

-- Definitions for the conditions given in the problem
def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000
def divisible_by (n d : ℕ) : Prop := d ∣ n

-- The main theorem stating the problem
theorem smallest_five_digit_divisible_by_2_3_8_9 :
  ∃ n : ℕ, is_five_digit n ∧ divisible_by n 2 ∧ divisible_by n 3 ∧ divisible_by n 8 ∧ divisible_by n 9 ∧ n = 10008 :=
sorry

end smallest_five_digit_divisible_by_2_3_8_9_l282_282032


namespace vector_combination_l282_282968

-- Define the vectors and the conditions
def vec_a : ℝ × ℝ := (1, -2)
def vec_b (m : ℝ) : ℝ × ℝ := (2, m)
def parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ v = (k * u.1, k * u.2)

-- The main theorem to be proved
theorem vector_combination (m : ℝ) (h_parallel : parallel vec_a (vec_b m)) : 3 * vec_a + 2 * vec_b m = (7, -14) := by
  sorry

end vector_combination_l282_282968


namespace sequence_properties_l282_282072

-- Define the arithmetic sequence and its properties
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = a n + d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_seq (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

-- Given conditions
variables (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ)
  (h_arith : arithmetic_seq a 2)
  (h_sum_prop : sum_seq a S)
  (h_ratio : ∀ n, S (2 * n) / S n = 4)
  (b : ℕ → ℤ) (T : ℕ → ℤ)
  (h_b : ∀ n, b n = a n * 2 ^ (n - 1))

-- Prove the sequences
theorem sequence_properties :
  (∀ n, a n = 2 * n - 1) ∧
  (∀ n, S n = n^2) ∧
  (∀ n, T n = (2 * n - 3) * 2^n + 3) :=
by
  sorry

end sequence_properties_l282_282072


namespace select_team_of_5_l282_282272

def boys : ℕ := 7
def girls : ℕ := 9
def total_students : ℕ := boys + girls

theorem select_team_of_5 (n : ℕ := total_students) (k : ℕ := 5) :
  (Nat.choose n k) = 4368 :=
by
  sorry

end select_team_of_5_l282_282272


namespace boys_in_biology_is_25_l282_282153

-- Definition of the total number of students in the Physics class
def physics_class_students : ℕ := 200

-- Definition of the total number of students in the Biology class
def biology_class_students : ℕ := physics_class_students / 2

-- Condition that there are three times as many girls as boys in the Biology class
def girls_boys_ratio : ℕ := 3

-- Calculate the total number of "parts" in the Biology class (3 parts girls + 1 part boys)
def total_parts : ℕ := girls_boys_ratio + 1

-- The number of boys in the Biology class
def boys_in_biology : ℕ := biology_class_students / total_parts

-- The statement to prove the number of boys in the Biology class is 25
theorem boys_in_biology_is_25 : boys_in_biology = 25 := by
  sorry

end boys_in_biology_is_25_l282_282153


namespace second_part_of_sum_l282_282920

-- Defining the problem conditions
variables (x : ℚ)
def sum_parts := (2 * x) + (1/2 * x) + (1/4 * x)

theorem second_part_of_sum :
  sum_parts x = 104 →
  (1/2 * x) = 208 / 11 :=
by
  intro h
  sorry

end second_part_of_sum_l282_282920


namespace triangle_inequality_l282_282853

variables {R : Type*} [LinearOrderedField R]

theorem triangle_inequality 
  (a b c u v w : R)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) :
  (a + b + c) * (1 / u + 1 / v + 1 / w) ≤ 3 * (a / u + b / v + c / w) :=
sorry

end triangle_inequality_l282_282853


namespace correct_negation_l282_282845

-- Define a triangle with angles A, B, and C
variables (α β γ : ℝ)

-- Define properties of the angles
def is_triangle (α β γ : ℝ) : Prop := α + β + γ = 180
def is_right_angle (angle : ℝ) : Prop := angle = 90
def is_acute_angle (angle : ℝ) : Prop := angle > 0 ∧ angle < 90

-- Original statement to be negated
def original_statement (α β γ : ℝ) : Prop := 
  is_triangle α β γ ∧ is_right_angle γ → is_acute_angle α ∧ is_acute_angle β

-- Negation of the original statement
def negated_statement (α β γ : ℝ) : Prop := 
  is_triangle α β γ ∧ ¬ is_right_angle γ → ¬ (is_acute_angle α ∧ is_acute_angle β)

-- Proof statement: prove that the negated statement is the correct negation
theorem correct_negation (α β γ : ℝ) :
  negated_statement α β γ = ¬ original_statement α β γ :=
sorry

end correct_negation_l282_282845


namespace count_even_numbers_l282_282246

theorem count_even_numbers : 
  ∃ n : ℕ, n = 199 ∧ ∀ m : ℕ, (302 ≤ m ∧ m < 700 ∧ m % 2 = 0) → 
    151 ≤ ((m - 300) / 2) ∧ ((m - 300) / 2) ≤ 349 :=
sorry

end count_even_numbers_l282_282246


namespace coffee_cost_per_week_l282_282710

theorem coffee_cost_per_week 
  (number_people : ℕ) 
  (cups_per_person_per_day : ℕ) 
  (ounces_per_cup : ℝ) 
  (cost_per_ounce : ℝ) 
  (total_cost_per_week : ℝ) 
  (h₁ : number_people = 4)
  (h₂ : cups_per_person_per_day = 2)
  (h₃ : ounces_per_cup = 0.5)
  (h₄ : cost_per_ounce = 1.25)
  (h₅ : total_cost_per_week = 35) : 
  number_people * cups_per_person_per_day * ounces_per_cup * cost_per_ounce * 7 = total_cost_per_week :=
by
  sorry

end coffee_cost_per_week_l282_282710


namespace three_digit_non_multiples_of_3_or_11_l282_282559

theorem three_digit_non_multiples_of_3_or_11 : 
  ∃ (n : ℕ), n = 546 ∧ 
  (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → 
    ¬ (x % 3 = 0 ∨ x % 11 = 0) → 
    n = (900 - (300 + 81 - 27))) := 
by 
  sorry

end three_digit_non_multiples_of_3_or_11_l282_282559


namespace bill_due_in_months_l282_282615

noncomputable def true_discount_time (TD A R : ℝ) : ℝ :=
  let P := A - TD
  let T := TD / (P * R / 100)
  12 * T

theorem bill_due_in_months :
  ∀ (TD A R : ℝ), TD = 189 → A = 1764 → R = 16 →
  abs (true_discount_time TD A R - 10.224) < 1 :=
by
  intros TD A R hTD hA hR
  sorry

end bill_due_in_months_l282_282615


namespace part1_part2_l282_282835

-- Define what a double root equation is
def is_double_root_eq (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ * x₁ * a + x₁ * b + c = 0 ∧ x₂ = 2 * x₁ ∧ x₂ * x₂ * a + x₂ * b + c = 0

-- Statement for part 1: proving x^2 - 3x + 2 = 0 is a double root equation
theorem part1 : is_double_root_eq 1 (-3) 2 :=
sorry

-- Statement for part 2: finding correct values of a and b for ax^2 + bx - 6 = 0 to be a double root equation with one root 2
theorem part2 : (∃ a b : ℝ, is_double_root_eq a b (-6) ∧ (a = -3 ∧ b = 9) ∨ (a = -3/4 ∧ b = 9/2)) :=
sorry

end part1_part2_l282_282835


namespace example_function_exists_l282_282445

open Set

noncomputable def indicator_function_SVC_set : ℝ → ℝ :=
  let SVC := λ x : ℝ, x ∈ { x | 0 ≤ x ∧ x ≤ 1 } \ ⋃ (n : ℕ), (Set.Ioo ((2 * (2 ^ n - 1) + 1) / 2^(n + 1)) ((2 * (2 ^ n - 1) + 2) / 2^(n + 1)))
  in indicator SVC (λ _, 1)

theorem example_function_exists :
  ∃ f : ℝ → ℝ, (∀ x ∈ (Icc 0 1), f x = indicator_function_SVC_set x) ∧
                (∀ x ∈ (Icc 0 1), f x ≥ 0 ∧ f x ≤ 1) ∧
                measure_theory.integrable f volume ∧
                ¬ (∃ g : ℝ → ℝ, measure_theory.integrable g measure_theory.volume ∧
                                 ∀ x ∈ (Icc 0 1), ite (indicator_function_SVC_set x = 0) (g x = 0) (g x = 1)) :=
by sorry

end example_function_exists_l282_282445


namespace bicycle_cost_price_l282_282646

theorem bicycle_cost_price (CP_A : ℝ) 
    (h1 : ∀ SP_B, SP_B = 1.20 * CP_A)
    (h2 : ∀ CP_C SP_B, CP_C = 1.40 * SP_B ∧ SP_B = 1.20 * CP_A)
    (h3 : ∀ SP_D CP_C, SP_D = 1.30 * CP_C ∧ CP_C = 1.40 * 1.20 * CP_A)
    (h4 : ∀ SP_D', SP_D' = 350 / 0.90) :
    CP_A = 350 / 1.9626 :=
by
  sorry

end bicycle_cost_price_l282_282646


namespace geometric_sequence_s6_s4_l282_282224

section GeometricSequence

variables {a : ℕ → ℝ} {a1 : ℝ} {q : ℝ}
variable (h_geom : ∀ n, a (n + 1) = a n * q)
variable (h_q_ne_one : q ≠ 1)
variable (S : ℕ → ℝ)
variable (h_S : ∀ n, S n = a1 * (1 - q^(n + 1)) / (1 - q))
variable (h_ratio : S 4 / S 2 = 3)

theorem geometric_sequence_s6_s4 :
  S 6 / S 4 = 7 / 3 :=
sorry

end GeometricSequence

end geometric_sequence_s6_s4_l282_282224


namespace virginia_ends_up_with_93_eggs_l282_282320

-- Define the initial and subtracted number of eggs as conditions
def initial_eggs : ℕ := 96
def taken_eggs : ℕ := 3

-- The theorem we want to prove
theorem virginia_ends_up_with_93_eggs : (initial_eggs - taken_eggs) = 93 :=
by
  sorry

end virginia_ends_up_with_93_eggs_l282_282320


namespace part1_part2_l282_282525

def op (a b : ℤ) := 2 * a - 3 * b

theorem part1 : op (-2) 3 = -13 := 
by
  -- Proof omitted
  sorry

theorem part2 (x : ℤ) : 
  let A := op (3 * x - 2) (x + 1)
  let B := op (-3 / 2 * x + 1) (-1 - 2 * x)
  B > A :=
by
  -- Proof omitted
  sorry

end part1_part2_l282_282525


namespace black_grid_after_rotation_l282_282043
open ProbabilityTheory

noncomputable def probability_black_grid_after_rotation : ℚ := 6561 / 65536

theorem black_grid_after_rotation (p : ℚ) (h : p = 1 / 2) :
  probability_black_grid_after_rotation = (3 / 4) ^ 8 := 
sorry

end black_grid_after_rotation_l282_282043


namespace new_average_score_l282_282894

theorem new_average_score (average_initial : ℝ) (total_practices : ℕ) (highest_score lowest_score : ℝ) :
  average_initial = 87 → 
  total_practices = 10 → 
  highest_score = 95 → 
  lowest_score = 55 → 
  ((average_initial * total_practices - highest_score - lowest_score) / (total_practices - 2)) = 90 :=
by
  intros h_avg h_total h_high h_low
  sorry

end new_average_score_l282_282894


namespace total_cost_is_correct_l282_282869

def cost_per_pound : ℝ := 0.45
def weight_sugar : ℝ := 40
def weight_flour : ℝ := 16

theorem total_cost_is_correct :
  weight_sugar * cost_per_pound + weight_flour * cost_per_pound = 25.20 :=
by
  sorry

end total_cost_is_correct_l282_282869


namespace count_ways_to_choose_one_person_l282_282347

theorem count_ways_to_choose_one_person (A B : ℕ) (hA : A = 3) (hB : B = 5) : A + B = 8 :=
by
  sorry

end count_ways_to_choose_one_person_l282_282347


namespace problem_l282_282679

theorem problem (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -6) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = -150 := 
by sorry

end problem_l282_282679


namespace compute_modulo_l282_282935

theorem compute_modulo :
    (2015 % 7) = 3 ∧ (2016 % 7) = 4 ∧ (2017 % 7) = 5 ∧ (2018 % 7) = 6 →
    (2015 * 2016 * 2017 * 2018) % 7 = 3 :=
by
  intros h
  have h1 := h.left
  have h2 := h.right.left
  have h3 := h.right.right.left
  have h4 := h.right.right.right
  sorry

end compute_modulo_l282_282935


namespace solve_for_x_l282_282398

theorem solve_for_x (x y : ℝ) (h1 : y = 1 / (5 * x + 2)) (h2 : y = 2) : x = -3 / 10 :=
by
  sorry

end solve_for_x_l282_282398


namespace complex_number_pow_two_l282_282059

theorem complex_number_pow_two (i : ℂ) (hi : i^2 = -1) : (1 + i)^2 = 2 * i :=
by sorry

end complex_number_pow_two_l282_282059


namespace A_and_B_work_together_for_49_days_l282_282911

variable (A B : ℝ)
variable (d : ℝ)
variable (fraction_left : ℝ)

def work_rate_A := 1 / 15
def work_rate_B := 1 / 20
def combined_work_rate := work_rate_A + work_rate_B

def fraction_work_completed (d : ℝ) := combined_work_rate * d

theorem A_and_B_work_together_for_49_days
    (A : ℝ := 1 / 15)
    (B : ℝ := 1 / 20)
    (fraction_left : ℝ := 0.18333333333333335) :
    (d : ℝ) → (fraction_work_completed d = 1 - fraction_left) →
    d = 49 :=
by
  sorry

end A_and_B_work_together_for_49_days_l282_282911


namespace initial_sum_simple_interest_l282_282504

theorem initial_sum_simple_interest :
  ∃ P : ℝ, (P * (3/100) + P * (5/100) + P * (4/100) + P * (6/100) = 100) ∧ (P = 5000 / 9) :=
by
  sorry

end initial_sum_simple_interest_l282_282504


namespace earnings_from_roosters_l282_282943

-- Definitions from the conditions
def price_per_kg : Float := 0.50
def weight_of_rooster1 : Float := 30.0
def weight_of_rooster2 : Float := 40.0

-- The theorem we need to prove (mathematically equivalent proof problem)
theorem earnings_from_roosters (p : Float := price_per_kg)
                               (w1 : Float := weight_of_rooster1)
                               (w2 : Float := weight_of_rooster2) :
  p * w1 + p * w2 = 35.0 := 
by {
  sorry
}

end earnings_from_roosters_l282_282943


namespace quadratic_roots_shifted_l282_282586

theorem quadratic_roots_shifted (a b c : ℚ) (r s : ℚ)
  (h1_eq: 5*r^2 + 2*r - 4 = 0)
  (h2_eq: 5*s^2 + 2*s - 4 = 0)
  (h_roots: (a = 1) ∧ (r-3)*(s-3) = c) :
  c = 47/5 :=
by
  sorry

end quadratic_roots_shifted_l282_282586


namespace password_probability_l282_282645

theorem password_probability 
  (password : Fin 6 → Fin 10) 
  (attempts : ℕ) 
  (correct_digit : Fin 10) 
  (probability_first_try : ℚ := 1 / 10)
  (probability_second_try : ℚ := (9 / 10) * (1 / 9)) : 
  ((password 5 = correct_digit) ∧ attempts ≤ 2) →
  (probability_first_try + probability_second_try = 1 / 5) :=
sorry

end password_probability_l282_282645


namespace greatest_value_product_l282_282410

def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def divisible_by (m n : ℕ) : Prop := ∃ k, m = k * n

theorem greatest_value_product (a b : ℕ) : 
    is_prime a → is_prime b → a < 10 → b < 10 → divisible_by (110 + 10 * a + b) 55 → a * b = 15 :=
by
    sorry

end greatest_value_product_l282_282410


namespace table_runner_combined_area_l282_282886

theorem table_runner_combined_area
    (table_area : ℝ) (cover_percentage : ℝ) (area_two_layers : ℝ) (area_three_layers : ℝ) (A : ℝ) :
    table_area = 175 →
    cover_percentage = 0.8 →
    area_two_layers = 24 →
    area_three_layers = 28 →
    A = (cover_percentage * table_area - area_two_layers - area_three_layers) + area_two_layers + 2 * area_three_layers →
    A = 168 :=
by
  intros h_table_area h_cover_percentage h_area_two_layers h_area_three_layers h_A
  sorry

end table_runner_combined_area_l282_282886


namespace triangle_ABC_proof_l282_282576

noncomputable def sin2C_eq_sqrt3sinC (C : ℝ) : Prop := Real.sin (2 * C) = Real.sqrt 3 * Real.sin C

theorem triangle_ABC_proof (C a b c : ℝ) 
  (H1 : sin2C_eq_sqrt3sinC C) 
  (H2 : 0 < Real.sin C)
  (H3 : b = 6) 
  (H4 : a + b + c = 6*Real.sqrt 3 + 6) :
  (C = π/6) ∧ (1/2 * a * b * Real.sin C = 6*Real.sqrt 3) :=
sorry

end triangle_ABC_proof_l282_282576


namespace value_of_expression_l282_282686

variable {a b m n x : ℝ}

def opposite (a b : ℝ) : Prop := a = -b
def reciprocal (m n : ℝ) : Prop := m * n = 1
def distance_to_2 (x : ℝ) : Prop := abs (x - 2) = 3

theorem value_of_expression (h1 : opposite a b) (h2 : reciprocal m n) (h3 : distance_to_2 x) :
  (a + b - m * n) * x + (a + b)^2022 + (- m * n)^2023 = 
  if x = 5 then -6 else if x = -1 then 0 else sorry :=
by
  sorry

end value_of_expression_l282_282686


namespace profit_per_meter_correct_l282_282506

-- Define the conditions
def total_meters := 40
def total_profit := 1400

-- Define the profit per meter calculation
def profit_per_meter := total_profit / total_meters

-- Theorem stating the profit per meter is Rs. 35
theorem profit_per_meter_correct : profit_per_meter = 35 := by
  sorry

end profit_per_meter_correct_l282_282506


namespace find_angle_l282_282388

variable (a b : ℝ × ℝ) (α : ℝ)
variable (θ : ℝ)

-- Conditions provided in the problem
def condition1 := (a.1^2 + a.2^2 = 4)
def condition2 := (b = (4 * Real.cos α, -4 * Real.sin α))
def condition3 := (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0)

-- Desired result
theorem find_angle (h1 : condition1 a) (h2 : condition2 b α) (h3 : condition3 a b) :
  θ = Real.pi / 3 :=
sorry

end find_angle_l282_282388


namespace comm_ring_of_center_condition_l282_282492

variable {R : Type*} [Ring R]

def in_center (x : R) : Prop := ∀ y : R, (x * y = y * x)

def is_commutative (R : Type*) [Ring R] : Prop := ∀ a b : R, a * b = b * a

theorem comm_ring_of_center_condition (h : ∀ x : R, in_center (x^2 - x)) : is_commutative R :=
sorry

end comm_ring_of_center_condition_l282_282492


namespace non_congruent_rectangles_l282_282503

theorem non_congruent_rectangles (h w : ℕ) (hp : 2 * (h + w) = 80) :
  ∃ n, n = 20 := by
  sorry

end non_congruent_rectangles_l282_282503


namespace sum_of_fractions_l282_282352

-- Definition of the fractions
def frac1 : ℚ := 3/5
def frac2 : ℚ := 5/11
def frac3 : ℚ := 1/3

-- Main theorem stating that the sum of the fractions equals 229/165
theorem sum_of_fractions : frac1 + frac2 + frac3 = 229 / 165 := sorry

end sum_of_fractions_l282_282352


namespace probability_of_break_in_first_50_meters_l282_282179

theorem probability_of_break_in_first_50_meters (total_length favorable_length : ℝ) 
  (h_total_length : total_length = 320) 
  (h_favorable_length : favorable_length = 50) : 
  (favorable_length / total_length) = 0.15625 := 
sorry

end probability_of_break_in_first_50_meters_l282_282179


namespace total_capsules_sold_in_2_weeks_l282_282245

-- Define the conditions as constants
def Earnings100mgPerWeek := 80
def CostPer100mgCapsule := 5
def Earnings500mgPerWeek := 60
def CostPer500mgCapsule := 2

-- Theorem to prove the total number of capsules sold in 2 weeks
theorem total_capsules_sold_in_2_weeks : 
  (Earnings100mgPerWeek / CostPer100mgCapsule) * 2 + (Earnings500mgPerWeek / CostPer500mgCapsule) * 2 = 92 :=
by
  sorry

end total_capsules_sold_in_2_weeks_l282_282245


namespace part1_part2_part3_l282_282626

section Part1

variables (a b : Real)

theorem part1 : 2 * (a + b)^2 - 8 * (a + b)^2 + 3 * (a + b)^2 = -3 * (a + b)^2 :=
by
  sorry

end Part1

section Part2

variables (x y : Real)

theorem part2 (h : x^2 + 2 * y = 4) : -3 * x^2 - 6 * y + 17 = 5 :=
by
  sorry

end Part2

section Part3

variables (a b c d : Real)

theorem part3 (h1 : a - 3 * b = 3) (h2 : 2 * b - c = -5) (h3 : c - d = 9) :
  (a - c) + (2 * b - d) - (2 * b - c) = 7 :=
by
  sorry

end Part3

end part1_part2_part3_l282_282626


namespace sum_of_digits_base8_888_l282_282175

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  let base := 8 in
  let digits := [1, 5, 7, 0] in
  digits.sum = 13 := 
by
  sorry

end sum_of_digits_base8_888_l282_282175


namespace two_a_minus_b_values_l282_282222

theorem two_a_minus_b_values (a b : ℝ) (h1 : |a| = 4) (h2 : |b| = 5) (h3 : |a + b| = -(a + b)) :
  (2 * a - b = 13) ∨ (2 * a - b = -3) :=
sorry

end two_a_minus_b_values_l282_282222


namespace iced_coffee_days_per_week_l282_282121

theorem iced_coffee_days_per_week (x : ℕ) (h1 : 5 * 4 = 20)
  (h2 : 20 * 52 = 1040)
  (h3 : 2 * x = 2 * x)
  (h4 : 52 * (2 * x) = 104 * x)
  (h5 : 1040 + 104 * x = 1040 + 104 * x)
  (h6 : 1040 + 104 * x - 338 = 1040 + 104 * x - 338)
  (h7 : (0.75 : ℝ) * (1040 + 104 * x) = 780 + 78 * x) :
  x = 3 :=
by
  sorry

end iced_coffee_days_per_week_l282_282121


namespace find_monthly_salary_l282_282653

variables (x h_1 h_2 h_3 : ℕ)

theorem find_monthly_salary 
    (half_salary_bank : h_1 = x / 2)
    (half_remaining_mortgage : h_2 = (h_1 - 300) / 2)
    (half_remaining_expenses : h_3 = (h_2 + 300) / 2)
    (remaining_salary : h_3 = 800) :
  x = 7600 :=
sorry

end find_monthly_salary_l282_282653


namespace modulo_residue_l282_282889

theorem modulo_residue:
  (247 + 5 * 40 + 7 * 143 + 4 * (2^3 - 1)) % 13 = 7 :=
by
  sorry

end modulo_residue_l282_282889


namespace total_animals_correct_l282_282593

def initial_cows : ℕ := 2
def initial_pigs : ℕ := 3
def initial_goats : ℕ := 6

def added_cows : ℕ := 3
def added_pigs : ℕ := 5
def added_goats : ℕ := 2

def total_cows : ℕ := initial_cows + added_cows
def total_pigs : ℕ := initial_pigs + added_pigs
def total_goats : ℕ := initial_goats + added_goats

def total_animals : ℕ := total_cows + total_pigs + total_goats

theorem total_animals_correct : total_animals = 21 := by
  sorry

end total_animals_correct_l282_282593


namespace sum_of_digits_base8_888_l282_282176

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  (let digits := [1, 5, 7, 0] in 
    digits.sum = 13) := by
  sorry

end sum_of_digits_base8_888_l282_282176


namespace cyclic_sum_inequality_l282_282454

theorem cyclic_sum_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
    (ab / (ab + a^5 + b^5)) + (bc / (bc + b^5 + c^5)) + (ca / (ca + c^5 + a^5)) ≤ 1 := by
  sorry

end cyclic_sum_inequality_l282_282454


namespace probability_same_color_and_six_sided_die_l282_282391

theorem probability_same_color_and_six_sided_die (d1_maroon d1_teal d1_cyan d1_sparkly : ℕ) 
                                                  (d2_maroon d2_teal d2_cyan d2_sparkly : ℕ) 
                                                  (six_sided_die_outcome : Fin 6) :
  d1_maroon = 3 ∧ d1_teal = 9 ∧ d1_cyan = 7 ∧ d1_sparkly = 1 ∧ 
  d2_maroon = 5 ∧ d2_teal = 6 ∧ d2_cyan = 8 ∧ d2_sparkly = 1 ∧ 
  (six_sided_die_outcome.val > 3) →
  (63 : ℚ) / 600 = 21 / 200 :=
sorry

end probability_same_color_and_six_sided_die_l282_282391


namespace calc1_calc2_l282_282794

noncomputable def calculation1 := -4^2

theorem calc1 : calculation1 = -16 := by
  sorry

noncomputable def calculation2 := (-3) - (-6)

theorem calc2 : calculation2 = 3 := by
  sorry

end calc1_calc2_l282_282794


namespace dot_product_PA_PB_l282_282426

theorem dot_product_PA_PB (x_0 : ℝ) (h : x_0 > 0):
  let P := (x_0, x_0 + 2/x_0)
  let A := ((x_0 + 2/x_0) / 2, (x_0 + 2/x_0) / 2)
  let B := (0, x_0 + 2/x_0)
  let vector_PA := ((x_0 + 2/x_0) / 2 - x_0, (x_0 + 2/x_0) / 2 - (x_0 + 2/x_0))
  let vector_PB := (0 - x_0, (x_0 + 2/x_0) - (x_0 + 2/x_0))
  vector_PA.1 * vector_PB.1 + vector_PA.2 * vector_PB.2 = -1 := by
  sorry

end dot_product_PA_PB_l282_282426


namespace max_cosA_cosB_l282_282108

open Real

theorem max_cosA_cosB {A B : ℝ} (h_triangle : A + B < π) (h_AB : A > 0 ∧ B > 0)
  (h_sin : sin A * sin B = (2 - sqrt 3) / 4) :
  ∃ M, M = (2 + sqrt 3) / 4 ∧ ∀ (x y : ℝ), h_AB → x = A → y = B → cos x * cos y ≤ M :=
sorry

end max_cosA_cosB_l282_282108


namespace set_intersection_l282_282972

open Set

universe u

variables {U : Type u} (A B : Set ℝ) (x : ℝ)

def universal_set : Set ℝ := univ
def set_A : Set ℝ := {x | abs x < 1}
def set_B : Set ℝ := {x | x > -1/2}
def complement_B : Set ℝ := {x | x ≤ -1/2}
def intersection : Set ℝ := {x | -1 < x ∧ x ≤ -1/2}

theorem set_intersection :
  (universal_set \ set_B) ∩ set_A = {x | -1 < x ∧ x ≤ -1/2} :=
by 
  -- The actual proof steps would go here
  sorry

end set_intersection_l282_282972


namespace wrapping_paper_per_present_l282_282000

theorem wrapping_paper_per_present :
  (1 / 2) / 5 = 1 / 10 :=
by
  sorry

end wrapping_paper_per_present_l282_282000


namespace find_x_l282_282978

theorem find_x :
  (2 + 3 = 5) →
  (3 + 4 = 7) →
  (1 / (2 + 3)) * (1 / (3 + 4)) = 1 / (x + 5) →
  x = 30 :=
by
  intros
  sorry

end find_x_l282_282978


namespace change_received_l282_282875

variable (a : ℕ)

theorem change_received (h : a ≤ 30) : 100 - 3 * a = 100 - 3 * a := 
by 
  sorry

end change_received_l282_282875


namespace erased_angle_is_97_l282_282791

theorem erased_angle_is_97 (n : ℕ) (h1 : 3 ≤ n) (h2 : (n - 2) * 180 = 1703 + x) : 
  1800 - 1703 = 97 :=
by sorry

end erased_angle_is_97_l282_282791


namespace molecular_weight_l282_282483

variable (weight_moles : ℝ) (moles : ℝ)

-- Given conditions
axiom h1 : weight_moles = 699
axiom h2 : moles = 3

-- Concluding statement to prove
theorem molecular_weight : (weight_moles / moles) = 233 := sorry

end molecular_weight_l282_282483


namespace area_of_square_is_1225_l282_282742

-- Given some basic definitions and conditions
variable (s : ℝ) -- side of the square which is the radius of the circle
variable (length : ℝ := (2 / 5) * s)
variable (breadth : ℝ := 10)
variable (area_rectangle : ℝ := length * breadth)

-- Statement to prove
theorem area_of_square_is_1225 
  (h1 : length = (2 / 5) * s)
  (h2 : breadth = 10)
  (h3 : area_rectangle = 140) : 
  s^2 = 1225 := by
    sorry

end area_of_square_is_1225_l282_282742


namespace find_linear_function_and_unit_price_l282_282262

def linear_function (k b x : ℝ) : ℝ := k * x + b

def profit (cost_price : ℝ) (selling_price : ℝ) (sales_volume : ℝ) : ℝ := 
  (selling_price - cost_price) * sales_volume

theorem find_linear_function_and_unit_price
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1 = 20) (h2 : y1 = 200)
  (h3 : x2 = 25) (h4 : y2 = 150)
  (h5 : x3 = 30) (h6 : y3 = 100)
  (cost_price := 10) (desired_profit := 2160) :
  ∃ k b x : ℝ, 
    (linear_function k b x1 = y1) ∧ 
    (linear_function k b x2 = y2) ∧ 
    (profit cost_price x (linear_function k b x) = desired_profit) ∧ 
    (linear_function k b x = -10 * x + 400) ∧ 
    (x = 22) :=
by
  sorry

end find_linear_function_and_unit_price_l282_282262


namespace triangle_inequality_l282_282731

theorem triangle_inequality 
  (a b c R : ℝ) 
  (h1 : a + b > c) 
  (h2 : a + c > b) 
  (h3 : b + c > a) 
  (hR : R = (a * b * c) / (4 * Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)))) : 
  a^2 + b^2 + c^2 ≤ 9 * R^2 :=
by 
  sorry

end triangle_inequality_l282_282731


namespace billy_horses_l282_282057

theorem billy_horses (each_horse_oats_per_meal : ℕ) (meals_per_day : ℕ) (total_oats_needed : ℕ) (days : ℕ) 
    (h_each_horse_oats_per_meal : each_horse_oats_per_meal = 4)
    (h_meals_per_day : meals_per_day = 2)
    (h_total_oats_needed : total_oats_needed = 96)
    (h_days : days = 3) :
    (total_oats_needed / (days * (each_horse_oats_per_meal * meals_per_day)) = 4) :=
by
  sorry

end billy_horses_l282_282057


namespace star_evaluation_l282_282937

def star (a b : ℕ) : ℕ := 3 + b^(a + 1)

theorem star_evaluation : star (star 2 3) 2 = 3 + 2^31 :=
by {
  sorry
}

end star_evaluation_l282_282937


namespace three_digit_numbers_not_multiple_of_3_or_11_l282_282555

-- Proving the number of three-digit numbers that are multiples of neither 3 nor 11 is 547
theorem three_digit_numbers_not_multiple_of_3_or_11 : (finset.Icc 100 999).filter (λ n, ¬(3 ∣ n) ∧ ¬(11 ∣ n)).card = 547 :=
by
  -- The steps to reach the solution will be implemented here
  sorry

end three_digit_numbers_not_multiple_of_3_or_11_l282_282555


namespace find_amount_of_alcohol_l282_282048

theorem find_amount_of_alcohol (A W : ℝ) (h₁ : A / W = 4 / 3) (h₂ : A / (W + 7) = 4 / 5) : A = 14 := 
sorry

end find_amount_of_alcohol_l282_282048


namespace train_usual_time_l282_282036

theorem train_usual_time (T : ℝ) (h1 : T > 0) : 
  (4 / 5 : ℝ) * (T + 1/2) = T :=
by 
  sorry

end train_usual_time_l282_282036


namespace min_rows_for_students_l282_282475

def min_rows (total_students seats_per_row max_students_per_school : ℕ) : ℕ :=
  total_students / seats_per_row + if total_students % seats_per_row == 0 then 0 else 1

theorem min_rows_for_students :
  ∀ (total_students seats_per_row max_students_per_school : ℕ),
  (total_students = 2016) →
  (seats_per_row = 168) →
  (max_students_per_school = 40) →
  min_rows total_students seats_per_row max_students_per_school = 15 :=
by
  intros total_students seats_per_row max_students_per_school h1 h2 h3
  -- We write down the proof outline to show that 15 is the required minimum
  sorry

end min_rows_for_students_l282_282475


namespace find_k_range_of_m_l282_282386

-- Given conditions and function definition
def f (x k : ℝ) : ℝ := x^2 + (2*k-3)*x + k^2 - 7

-- Prove that k = 3 when the zeros of f(x) are -1 and -2
theorem find_k (k : ℝ) (h₁ : f (-1) k = 0) (h₂ : f (-2) k = 0) : k = 3 := 
by sorry

-- Prove the range of m such that f(x) < m for x in [-2, 2]
theorem range_of_m (m : ℝ) : (∀ x ∈ Set.Icc (-2 : ℝ) 2, x^2 + 3*x + 2 < m) ↔ 12 < m :=
by sorry

end find_k_range_of_m_l282_282386


namespace john_children_probability_l282_282416

open ProbabilityTheory

-- Define a simple binomial distribution with six trials and success probability 1/2
def binomial_distribution (n k: ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

-- Define the probability of getting at least k successes in n trials
def at_least_k_successes (n k : ℕ) (p : ℚ) : ℚ :=
  (finset.range (n + 1)).filter (λ x, x ≥ k).sum (λ j, binomial_distribution n j p)

theorem john_children_probability :
  at_least_k_successes 6 3 (1/2) = 27/32 :=
by sorry

end john_children_probability_l282_282416


namespace part1_l282_282765

theorem part1 : 2 * Real.tan (60 * Real.pi / 180) * Real.cos (30 * Real.pi / 180) - (Real.sin (45 * Real.pi / 180)) ^ 2 = 5 / 2 := 
sorry

end part1_l282_282765


namespace value_x_when_y2_l282_282396

theorem value_x_when_y2 (x : ℝ) (h1 : ∃ (x : ℝ), y = 1 / (5 * x + 2)) (h2 : y = 2) : x = -3 / 10 := by
  sorry

end value_x_when_y2_l282_282396


namespace find_x_l282_282539

theorem find_x (x : ℝ) (h : (x^2 - x - 6) / (x + 1) = (x^2 - 2*x - 3) * (0 : ℂ).im) : x = 3 :=
sorry

end find_x_l282_282539


namespace find_y_l282_282697

theorem find_y
  (x y : ℝ)
  (h1 : x^(3*y) = 8)
  (h2 : x = 2) :
  y = 1 :=
sorry

end find_y_l282_282697


namespace vertex_of_parabola_l282_282536

theorem vertex_of_parabola :
  (∃ x y : ℝ, y = -3*x^2 + 6*x + 1 ∧ (x, y) = (1, 4)) :=
sorry

end vertex_of_parabola_l282_282536


namespace problem1_problem2_l282_282362

theorem problem1 : (-1 / 2) * (-8) + (-6) = -2 := by
  sorry

theorem problem2 : -(1^4) - 2 / (-1 / 3) - abs (-9) = -4 := by
  sorry

end problem1_problem2_l282_282362


namespace min_n_for_triangle_pattern_l282_282310

/-- 
There are two types of isosceles triangles with a waist length of 1:
-  Type 1: An acute isosceles triangle with a vertex angle of 30 degrees.
-  Type 2: A right isosceles triangle with a vertex angle of 90 degrees.
They are placed around a point in a clockwise direction in a sequence such that:
- The 1st and 2nd are acute isosceles triangles (30 degrees),
- The 3rd is a right isosceles triangle (90 degrees),
- The 4th and 5th are acute isosceles triangles (30 degrees),
- The 6th is a right isosceles triangle (90 degrees), and so on.

Prove that the minimum value of n such that the nth triangle coincides exactly with
the 1st triangle is 23.
-/
theorem min_n_for_triangle_pattern : ∃ n : ℕ, n = 23 ∧ (∀ m < 23, m ≠ 23) :=
sorry

end min_n_for_triangle_pattern_l282_282310


namespace jane_ends_with_crayons_l282_282715

-- Definitions for the conditions in the problem
def initial_crayons : Nat := 87
def crayons_eaten : Nat := 7
def packs_bought : Nat := 5
def crayons_per_pack : Nat := 10
def crayons_break : Nat := 3

-- Statement to prove: Jane ends with 127 crayons
theorem jane_ends_with_crayons :
  initial_crayons - crayons_eaten + (packs_bought * crayons_per_pack) - crayons_break = 127 :=
by
  sorry

end jane_ends_with_crayons_l282_282715


namespace number_of_boys_l282_282490

-- Define the conditions given in the problem
def total_people := 41
def total_amount := 460
def boy_amount := 12
def girl_amount := 8

-- Define the proof statement that needs to be proven
theorem number_of_boys (B G : ℕ) (h1 : B + G = total_people) (h2 : boy_amount * B + girl_amount * G = total_amount) : B = 33 := 
by {
  -- The actual proof will go here
  sorry
}

end number_of_boys_l282_282490


namespace regression_equation_represents_real_relationship_maximized_l282_282608

-- Definitions from the conditions
def regression_equation (y x : ℝ) := ∃ (a b : ℝ), y = a * x + b

def represents_real_relationship_maximized (y x : ℝ) := regression_equation y x

-- The proof problem statement
theorem regression_equation_represents_real_relationship_maximized 
: ∀ (y x : ℝ), regression_equation y x → represents_real_relationship_maximized y x :=
by
  sorry

end regression_equation_represents_real_relationship_maximized_l282_282608


namespace technicians_in_workshop_l282_282256

theorem technicians_in_workshop :
  (∃ T R: ℕ, T + R = 42 ∧ 8000 * 42 = 18000 * T + 6000 * R) → ∃ T: ℕ, T = 7 :=
by
  sorry

end technicians_in_workshop_l282_282256


namespace distinguishable_triangles_count_l282_282752

def count_distinguishable_triangles (colors : ℕ) : ℕ :=
  let corner_cases := colors + (colors * (colors - 1)) + (colors * (colors - 1) * (colors - 2) / 6)
  let edge_cases := colors * colors
  let center_cases := colors
  corner_cases * edge_cases * center_cases

theorem distinguishable_triangles_count :
  count_distinguishable_triangles 8 = 61440 :=
by
  unfold count_distinguishable_triangles
  -- corner_cases = 8 + 8 * 7 + (8 * 7 * 6) / 6 = 120
  -- edge_cases = 8 * 8 = 64
  -- center_cases = 8
  -- Total = 120 * 64 * 8 = 61440
  sorry

end distinguishable_triangles_count_l282_282752


namespace calc_result_l282_282630

theorem calc_result : (377 / 13 / 29 * 1 / 4 / 2) = 0.125 := 
by sorry

end calc_result_l282_282630


namespace arithmetic_seq_inequality_l282_282961

-- Definition for the sum of the first n terms of an arithmetic sequence
def sum_arith_seq (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * a₁ + n * (n - 1) / 2 * d

theorem arithmetic_seq_inequality (a₁ : ℕ) (d : ℕ) (n : ℕ) (h : d > 0) :
  sum_arith_seq a₁ d n + sum_arith_seq a₁ d (3 * n) > 2 * sum_arith_seq a₁ d (2 * n) := by
  sorry

end arithmetic_seq_inequality_l282_282961


namespace y_A_functional_relationship_y_B_functional_relationship_cost_effective_B_cost_effective_equal_cost_effective_A_l282_282666

-- Definitions of cost functions for travel agencies
def full_ticket_price : ℕ := 240

def y_A (x : ℕ) : ℕ := 120 * x + 240
def y_B (x : ℕ) : ℕ := 144 * x + 144

-- Prove functional relationships for y_A and y_B
theorem y_A_functional_relationship (x : ℕ) : y_A x = 120 * x + 240 :=
by sorry

theorem y_B_functional_relationship (x : ℕ) : y_B x = 144 * x + 144 :=
by sorry

-- Prove conditions for cost-effectiveness
theorem cost_effective_B (x : ℕ) : x < 4 → y_A x > y_B x :=
by sorry

theorem cost_effective_equal (x : ℕ) : x = 4 → y_A x = y_B x :=
by sorry

theorem cost_effective_A (x : ℕ) : x > 4 → y_A x < y_B x :=
by sorry

end y_A_functional_relationship_y_B_functional_relationship_cost_effective_B_cost_effective_equal_cost_effective_A_l282_282666


namespace find_valid_pairs_l282_282372

open Nat

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ m : ℕ, 2 ≤ m → m ≤ p / 2 → ¬(m ∣ p)

def valid_pair (n p : ℕ) : Prop :=
  is_prime p ∧ 0 < n ∧ n ≤ 2 * p ∧ n ^ (p - 1) ∣ (p - 1) ^ n + 1

theorem find_valid_pairs (n p : ℕ) : valid_pair n p ↔ (n = 1 ∧ is_prime p) ∨ (n, p) = (2, 2) ∨ (n, p) = (3, 3) := by
  sorry

end find_valid_pairs_l282_282372


namespace number_of_subsets_number_of_sets_satisfying_conditions_l282_282684

open Finset

theorem number_of_subsets (X : Finset ℕ) :
  {1, 2} ⊆ X ∧ X ⊆ {1, 2, 3, 4, 5} ↔
  ∃ (Y : Finset ℕ), Y ⊆ {3, 4, 5} ∧ X = {1, 2} ∪ Y :=
sorry

theorem number_of_sets_satisfying_conditions :
  ∃ n : ℕ, 
  (number_of_subsets (X : Finset ℕ) → n = 8) :=
sorry

end number_of_subsets_number_of_sets_satisfying_conditions_l282_282684


namespace total_balloons_l282_282028

theorem total_balloons
  (g b y r : ℕ)  -- Number of green, blue, yellow, and red balloons respectively
  (equal_groups : g = b ∧ b = y ∧ y = r)
  (anya_took : y / 2 = 84) :
  g + b + y + r = 672 := by
sorry

end total_balloons_l282_282028


namespace percent_gain_on_transaction_l282_282914

theorem percent_gain_on_transaction
  (c : ℝ) -- cost per sheep
  (price_750_sold : ℝ := 800 * c) -- price at which 750 sheep were sold in total
  (price_per_sheep_750 : ℝ := price_750_sold / 750)
  (price_per_sheep_50 : ℝ := 1.1 * price_per_sheep_750)
  (revenue_750 : ℝ := price_per_sheep_750 * 750)
  (revenue_50 : ℝ := price_per_sheep_50 * 50)
  (total_revenue : ℝ := revenue_750 + revenue_50)
  (total_cost : ℝ := 800 * c)
  (profit : ℝ := total_revenue - total_cost)
  (percent_gain : ℝ := (profit / total_cost) * 100) :
  percent_gain = 14 :=
sorry

end percent_gain_on_transaction_l282_282914


namespace find_sinD_l282_282699

variable (DE DF : ℝ)

-- Conditions
def area_of_triangle (DE DF : ℝ) (sinD : ℝ) : Prop :=
  1 / 2 * DE * DF * sinD = 72

def geometric_mean (DE DF : ℝ) : Prop :=
  Real.sqrt (DE * DF) = 15

theorem find_sinD (DE DF sinD : ℝ) (h1 : area_of_triangle DE DF sinD) (h2 : geometric_mean DE DF) :
  sinD = 16 / 25 :=
by 
  -- Proof goes here
  sorry

end find_sinD_l282_282699


namespace roses_formula_l282_282569

open Nat

def total_roses (n : ℕ) : ℕ := 
  (choose n 4) + (choose (n - 1) 2)

theorem roses_formula (n : ℕ) (h : n ≥ 4) : 
  total_roses n = (choose n 4) + (choose (n - 1) 2) := 
by
  sorry

end roses_formula_l282_282569


namespace parallel_lines_a_value_l282_282382

theorem parallel_lines_a_value 
    (a : ℝ) 
    (l₁ : ∀ x y : ℝ, 2 * x + y - 1 = 0) 
    (l₂ : ∀ x y : ℝ, (a - 1) * x + 3 * y - 2 = 0) 
    (h_parallel : ∀ x y : ℝ, 2 / (a - 1) = 1 / 3) : 
    a = 7 := 
    sorry

end parallel_lines_a_value_l282_282382


namespace total_time_spent_l282_282592

def timeDrivingToSchool := 20
def timeAtGroceryStore := 15
def timeFillingGas := 5
def timeAtParentTeacherNight := 70
def timeAtCoffeeShop := 30
def timeDrivingHome := timeDrivingToSchool

theorem total_time_spent : 
  timeDrivingToSchool + timeAtGroceryStore + timeFillingGas + timeAtParentTeacherNight + timeAtCoffeeShop + timeDrivingHome = 160 :=
by
  sorry

end total_time_spent_l282_282592


namespace solve_equation_1_solve_equation_2_l282_282954

theorem solve_equation_1 (x : ℝ) : (2 * x - 1) ^ 2 - 25 = 0 ↔ x = 3 ∨ x = -2 := 
sorry

theorem solve_equation_2 (x : ℝ) : (1 / 3) * (x + 3) ^ 3 - 9 = 0 ↔ x = 0 := 
sorry

end solve_equation_1_solve_equation_2_l282_282954


namespace find_original_price_l282_282757

variable (original_price : ℝ)
variable (final_price : ℝ) (first_reduction_rate : ℝ) (second_reduction_rate : ℝ)

theorem find_original_price :
  final_price = 15000 →
  first_reduction_rate = 0.30 →
  second_reduction_rate = 0.40 →
  0.42 * original_price = final_price →
  original_price = 35714 := by
  intros h1 h2 h3 h4
  sorry

end find_original_price_l282_282757


namespace num_solutions_l282_282184

theorem num_solutions (h : ∀ n : ℕ, (1 ≤ n ∧ n ≤ 455) → n^3 % 455 = 1) : 
  (∃ s : Finset ℕ, (∀ n : ℕ, n ∈ s ↔ (1 ≤ n ∧ n ≤ 455) ∧ n^3 % 455 = 1) ∧ s.card = 9) :=
sorry

end num_solutions_l282_282184


namespace cos_B_plus_C_find_c_value_l282_282402

variables (A B C a b c : ℝ)
axiom triangle_angles_sum : A + B + C = Real.pi
axiom sides_opposite : a = 2 * b
axiom sine_arithmetic_sequence : 2 * Real.sin C = Real.sin A + Real.sin B
axiom area_triangle : 0.5 * b * c * Real.sin A = 3 * Real.sqrt 15 / 3
axiom sin_cos_identity : Real.sin A ^ 2 + Real.cos A ^ 2 = 1

-- Prove the value of cos(B + C)
theorem cos_B_plus_C : Real.cos (B + C) = 1 / 4 :=
by
  sorry

-- Prove the value of c given the area of the triangle
theorem find_c_value : c = 4 * Real.sqrt 2 :=
by
  sorry

end cos_B_plus_C_find_c_value_l282_282402


namespace enrico_earnings_l282_282947

theorem enrico_earnings : 
  let price_per_kg := 0.50
  let weight_rooster1 := 30
  let weight_rooster2 := 40
  let total_earnings := price_per_kg * weight_rooster1 + price_per_kg * weight_rooster2
  total_earnings = 35 := 
by
  sorry

end enrico_earnings_l282_282947


namespace julia_total_watches_l282_282417

-- Definitions based on conditions.
def silver_watches : Nat := 20
def bronze_watches : Nat := 3 * silver_watches
def total_silver_bronze_watches : Nat := silver_watches + bronze_watches
def gold_watches : Nat := total_silver_bronze_watches / 10

-- The final proof statement without providing the proof.
theorem julia_total_watches : (silver_watches + bronze_watches + gold_watches) = 88 :=
by 
  -- Since we don't need to provide the actual proof, we use sorry
  sorry

end julia_total_watches_l282_282417


namespace find_number_l282_282019

theorem find_number (x : ℝ) (h : (x - 8 - 12) / 5 = 7) : x = 55 :=
sorry

end find_number_l282_282019


namespace cats_more_than_spinsters_l282_282020

def ratio (a b : ℕ) := ∃ k : ℕ, a = b * k

theorem cats_more_than_spinsters (S C : ℕ) (h1 : ratio 2 9) (h2 : S = 12) (h3 : 2 * C = 108) :
  C - S = 42 := by 
  sorry

end cats_more_than_spinsters_l282_282020


namespace factory_processing_time_eq_l282_282773

variable (x : ℝ) (initial_rate : ℝ := x)
variable (parts : ℝ := 500)
variable (first_stage_parts : ℝ := 100)
variable (remaining_parts : ℝ := parts - first_stage_parts)
variable (total_days : ℝ := 6)
variable (new_rate : ℝ := 2 * initial_rate)

theorem factory_processing_time_eq (h : x > 0) : (first_stage_parts / initial_rate) + (remaining_parts / new_rate) = total_days := 
by
  sorry

end factory_processing_time_eq_l282_282773


namespace focus_of_given_parabola_l282_282011

-- Define the given condition as a parameter
def parabola_eq (x y : ℝ) : Prop :=
  y = - (1/2) * x^2

-- Define the property for the focus of the parabola
def is_focus_of_parabola (focus : ℝ × ℝ) : Prop :=
  focus = (0, -1/2)

-- The theorem stating that the given parabola equation has the specific focus
theorem focus_of_given_parabola : 
  (∀ x y : ℝ, parabola_eq x y) → is_focus_of_parabola (0, -1/2) :=
by
  intro h
  unfold parabola_eq at h
  unfold is_focus_of_parabola
  sorry

end focus_of_given_parabola_l282_282011


namespace seq_arithmetic_l282_282969

def seq (n : ℕ) : ℤ := 2 * n + 5

theorem seq_arithmetic :
  ∀ n : ℕ, seq (n + 1) - seq n = 2 :=
by
  intro n
  have h1 : seq (n + 1) = 2 * (n + 1) + 5 := rfl
  have h2 : seq n = 2 * n + 5 := rfl
  rw [h1, h2]
  linarith

end seq_arithmetic_l282_282969


namespace find_x_l282_282225

theorem find_x (n : ℕ) (h_odd : n % 2 = 1)
  (h_three_primes : ∃ (p1 p2 p3 : ℕ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ 
    11 = p1 ∧ (7 ^ n + 1) = p1 * p2 * p3) :
  (7 ^ n + 1) = 16808 :=
by
  sorry

end find_x_l282_282225


namespace min_ab_l282_282394

theorem min_ab (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_eq : a * b = a + b + 3) : a * b ≥ 9 :=
sorry

end min_ab_l282_282394


namespace largest_possible_integer_smallest_possible_integer_l282_282206

theorem largest_possible_integer : 3 * (15 + 20 / 4 + 1) = 63 := by
  sorry

theorem smallest_possible_integer : (3 * 15 + 20) / (4 + 1) = 13 := by
  sorry

end largest_possible_integer_smallest_possible_integer_l282_282206


namespace digits_in_number_l282_282155

def four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def contains_digits (n : ℕ) (d1 d2 d3 : ℕ) : Prop :=
  (n / 1000 = d1 ∨ n / 100 % 10 = d1 ∨ n / 10 % 10 = d1 ∨ n % 10 = d1) ∧
  (n / 1000 = d2 ∨ n / 100 % 10 = d2 ∨ n / 10 % 10 = d2 ∨ n % 10 = d2) ∧
  (n / 1000 = d3 ∨ n / 100 % 10 = d3 ∨ n / 10 % 10 = d3 ∨ n % 10 = d3)

def exactly_two_statements_true (s1 s2 s3 : Prop) : Prop :=
  (s1 ∧ s2 ∧ ¬s3) ∨ (s1 ∧ ¬s2 ∧ s3) ∨ (¬s1 ∧ s2 ∧ s3)

theorem digits_in_number (n : ℕ) 
  (h1 : four_digit_number n)
  (h2 : contains_digits n 1 4 5 ∨ contains_digits n 1 5 9 ∨ contains_digits n 7 8 9)
  (h3 : exactly_two_statements_true (contains_digits n 1 4 5) (contains_digits n 1 5 9) (contains_digits n 7 8 9)) :
  contains_digits n 1 4 5 ∧ contains_digits n 1 5 9 :=
sorry

end digits_in_number_l282_282155


namespace no_sequences_periodic_l282_282637

-- Definition of sequence A
def seqA : ℕ → ℤ
| 0 := 1
| 1 := 1
| 2 := 0
| 3 := 1
| n := if h : ∃ k, n = k + (k + 1) then 0 else 0

-- Definition of sequence B
def seqB : ℕ → ℤ
| 0 := 1
| 1 := 2
| 2 := 1
| 3 := 2
| n := if h : ∃ k, n = k*2 + k then 3 else 1

-- Definition of sequence C (adding corresponding elements of seqA and seqB)
def seqC (n : ℕ) : ℤ := seqA n + seqB n

-- The main theorem to prove that none of the sequences A, B, or C are periodic
theorem no_sequences_periodic :
  ¬(∃ P, ∀ n, seqA (n + P) = seqA n) ∧
  ¬(∃ P, ∀ n, seqB (n + P) = seqB n) ∧
  ¬(∃ P, ∀ n, seqC (n + P) = seqC n) :=
sorry

end no_sequences_periodic_l282_282637


namespace initial_balloons_blown_up_l282_282606
-- Import necessary libraries

-- Define the statement
theorem initial_balloons_blown_up (x : ℕ) (hx : x + 13 = 60) : x = 47 :=
by
  sorry

end initial_balloons_blown_up_l282_282606


namespace min_value_x2y2z2_l282_282269

open Real

noncomputable def condition (x y z : ℝ) : Prop := (1 / x + 1 / y + 1 / z = 3)

theorem min_value_x2y2z2 (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_cond : condition x y z) :
  x^2 * y^2 * z^2 ≥ 1 / 64 :=
by
  sorry

end min_value_x2y2z2_l282_282269


namespace symmetric_point_correct_l282_282740

-- Define the point and the symmetry operation
structure Point :=
  (x : ℝ)
  (y : ℝ)

def symmetric_with_respect_to_x_axis (p : Point) : Point :=
  {x := p.x, y := -p.y}

-- Define the specific point M
def M : Point := {x := 1, y := 2}

-- Define the expected answer point M'
def M' : Point := {x := 1, y := -2}

-- Prove that the symmetric point with respect to the x-axis is as expected
theorem symmetric_point_correct :
  symmetric_with_respect_to_x_axis M = M' :=
by sorry

end symmetric_point_correct_l282_282740


namespace seth_spent_more_l282_282600

theorem seth_spent_more : 
  let ice_cream_cartons := 20
  let yogurt_cartons := 2
  let ice_cream_price := 6
  let yogurt_price := 1
  let ice_cream_discount := 0.10
  let yogurt_discount := 0.20
  let total_ice_cream_cost := ice_cream_cartons * ice_cream_price
  let total_yogurt_cost := yogurt_cartons * yogurt_price
  let discounted_ice_cream_cost := total_ice_cream_cost * (1 - ice_cream_discount)
  let discounted_yogurt_cost := total_yogurt_cost * (1 - yogurt_discount)
  discounted_ice_cream_cost - discounted_yogurt_cost = 106.40 :=
by
  sorry

end seth_spent_more_l282_282600


namespace ratio_of_a_to_b_l282_282140

theorem ratio_of_a_to_b (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
    (h_x : x = 1.25 * a) (h_m : m = 0.40 * b) (h_ratio : m / x = 0.4) 
    : (a / b) = 4 / 5 := by
  sorry

end ratio_of_a_to_b_l282_282140


namespace relationship_among_a_b_and_ab_l282_282231

noncomputable def a : ℝ := Real.log 0.4 / Real.log 0.2
noncomputable def b : ℝ := 1 - (1 / (Real.log 4 / Real.log 10))

theorem relationship_among_a_b_and_ab : a * b < a + b ∧ a + b < 0 := by
  sorry

end relationship_among_a_b_and_ab_l282_282231


namespace max_value_expression_l282_282066

theorem max_value_expression (x : ℝ) : 
  (∃ y : ℝ, y = x^4 / (x^8 + 2 * x^6 + 4 * x^4 + 4 * x^2 + 16) ∧ 
                ∀ z : ℝ, 
                (∃ x : ℝ, z = x^4 / (x^8 + 2 * x^6 + 4 * x^4 + 4 * x^2 + 16)) → 
                y ≥ z) → 
  ∃ y : ℝ, y = 1 / 16 := 
sorry

end max_value_expression_l282_282066


namespace equation_of_line_l282_282012

theorem equation_of_line (x y : ℝ) :
  (∃ (x1 y1 : ℝ), (x1 = 0) ∧ (y1= 2) ∧ (y - y1 = 2 * (x - x1))) → (y = 2 * x + 2) :=
by
  sorry

end equation_of_line_l282_282012


namespace combined_length_of_trains_is_correct_l282_282318

noncomputable def combined_length_of_trains : ℕ :=
  let speed_A := 120 * 1000 / 3600 -- speed of train A in m/s
  let speed_B := 100 * 1000 / 3600 -- speed of train B in m/s
  let speed_motorbike := 64 * 1000 / 3600 -- speed of motorbike in m/s
  let relative_speed_A := (120 - 64) * 1000 / 3600 -- relative speed of train A with respect to motorbike in m/s
  let relative_speed_B := (100 - 64) * 1000 / 3600 -- relative speed of train B with respect to motorbike in m/s
  let length_A := relative_speed_A * 75 -- length of train A in meters
  let length_B := relative_speed_B * 90 -- length of train B in meters
  length_A + length_B

theorem combined_length_of_trains_is_correct :
  combined_length_of_trains = 2067 :=
  by
  sorry

end combined_length_of_trains_is_correct_l282_282318


namespace students_taking_art_l282_282771

theorem students_taking_art :
  ∀ (total_students music_students both_music_art neither_music_art : ℕ),
  total_students = 500 →
  music_students = 30 →
  both_music_art = 10 →
  neither_music_art = 470 →
  (total_students - neither_music_art) - (music_students - both_music_art) - both_music_art = 10 :=
by
  intros total_students music_students both_music_art neither_music_art h_total h_music h_both h_neither
  sorry

end students_taking_art_l282_282771


namespace daniel_video_games_l282_282520

/--
Daniel has a collection of some video games. 80 of them, Daniel bought for $12 each.
Of the rest, 50% were bought for $7. All others had a price of $3 each.
Daniel spent $2290 on all the games in his collection.
Prove that the total number of video games in Daniel's collection is 346.
-/
theorem daniel_video_games (n : ℕ) (r : ℕ)
    (h₀ : 80 * 12 = 960)
    (h₁ : 2290 - 960 = 1330)
    (h₂ : r / 2 * 7 + r / 2 * 3 = 1330):
    n = 80 + r → n = 346 :=
by
  intro h_total
  have r_eq : r = 266 := by sorry
  rw [r_eq] at h_total
  exact h_total

end daniel_video_games_l282_282520


namespace find_x_l282_282226

open Nat

def has_three_distinct_prime_factors (x : ℕ) : Prop :=
  ∃ a b c : ℕ, Prime a ∧ Prime b ∧ Prime c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ x = a * b * c

theorem find_x (n : ℕ) (h₁ : Odd n) (h₂ : 7^n + 1 = x)
  (h₃ : has_three_distinct_prime_factors x) (h₄ : 11 ∣ x) : x = 16808 := by
  sorry

end find_x_l282_282226


namespace number_of_cute_integers_l282_282531

def is_cute (n : ℕ) : Prop :=
  let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  (digits.perm [1, 2, 3, 4]) ∧
  ((n / 1000) % 1 = 0) ∧  -- Divisible by 1 (trivially true)
  ((n / 100) % 2 = 0) ∧   -- First 2 digits divisible by 2
  ((n / 10) % 3 = 0) ∧    -- First 3 digits divisible by 3
  (n % 4 = 0)             -- All 4 digits divisible by 4

theorem number_of_cute_integers : ∃! (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ is_cute n :=
sorry

end number_of_cute_integers_l282_282531


namespace probability_at_least_one_deciphers_l282_282822

theorem probability_at_least_one_deciphers (P_A P_B : ℚ) (hA : P_A = 1/2) (hB : P_B = 1/3) :
    P_A + P_B - P_A * P_B = 2/3 := by
  sorry

end probability_at_least_one_deciphers_l282_282822


namespace angle_B_in_triangle_tan_A_given_c_eq_3a_l282_282965

theorem angle_B_in_triangle (a b c A B C : ℝ) (h1 : a^2 + c^2 - b^2 = ac) : B = π / 3 := 
sorry

theorem tan_A_given_c_eq_3a (a b c A B C : ℝ) (h1 : a^2 + c^2 - b^2 = ac) (h2 : c = 3 * a) : 
(Real.tan A) = Real.sqrt 3 / 5 :=
sorry

end angle_B_in_triangle_tan_A_given_c_eq_3a_l282_282965


namespace find_b_of_sin_l282_282510

theorem find_b_of_sin (a b c d : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
                       (h_period : (2 * Real.pi) / b = Real.pi / 2) : b = 4 := by
  sorry

end find_b_of_sin_l282_282510


namespace min_sum_rect_box_l282_282134

-- Define the main theorem with the given constraints
theorem min_sum_rect_box (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_vol : a * b * c = 2002) : a + b + c ≥ 38 :=
  sorry

end min_sum_rect_box_l282_282134


namespace efficiency_of_worker_p_more_than_q_l282_282196

noncomputable def worker_p_rate : ℚ := 1 / 22
noncomputable def combined_rate : ℚ := 1 / 12

theorem efficiency_of_worker_p_more_than_q
  (W_p : ℚ) (W_q : ℚ)
  (h1 : W_p = worker_p_rate)
  (h2 : W_p + W_q = combined_rate) : (W_p / W_q) = 6 / 5 :=
by
  sorry

end efficiency_of_worker_p_more_than_q_l282_282196


namespace length_of_ae_l282_282336

-- Definitions for lengths of segments
variable {ab bc cd de ac ae : ℝ}

-- Given conditions as assumptions
axiom h1 : bc = 3 * cd
axiom h2 : de = 8
axiom h3 : ab = 5
axiom h4 : ac = 11

-- The main theorem to prove
theorem length_of_ae : ae = ab + bc + cd + de → bc = ac - ab → bc = 6 → cd = bc / 3 → ae = 21 :=
by sorry

end length_of_ae_l282_282336


namespace julia_watches_l282_282420

theorem julia_watches (silver_watches bronze_multiplier : ℕ)
    (total_watches_percent_to_buy total_percent bronze_multiplied : ℕ) :
    silver_watches = 20 →
    bronze_multiplier = 3 →
    total_watches_percent_to_buy = 10 →
    total_percent = 100 → 
    bronze_multiplied = (silver_watches * bronze_multiplier) →
    let bronze_watches := silver_watches * bronze_multiplier,
        total_watches_before := silver_watches + bronze_watches,
        gold_watches := total_watches_percent_to_buy * total_watches_before / total_percent,
        total_watches_after := total_watches_before + gold_watches
    in
    total_watches_after = 88 :=
by
    intros silver_watches_def bronze_multiplier_def total_watches_percent_to_buy_def
    total_percent_def bronze_multiplied_def
    have bronze_watches := silver_watches * bronze_multiplier
    have total_watches_before := silver_watches + bronze_watches
    have gold_watches := total_watches_percent_to_buy * total_watches_before / total_percent
    have total_watches_after := total_watches_before + gold_watches
    simp [bronze_watches, total_watches_before, gold_watches, total_watches_after]
    exact sorry

end julia_watches_l282_282420


namespace greatest_decimal_is_7391_l282_282035

noncomputable def decimal_conversion (n d : ℕ) : ℝ :=
  n / d

noncomputable def forty_two_percent_of (r : ℝ) : ℝ :=
  0.42 * r

theorem greatest_decimal_is_7391 :
  let a := forty_two_percent_of (decimal_conversion 7 11)
  let b := decimal_conversion 17 23
  let c := 0.7391
  let d := decimal_conversion 29 47
  a < b ∧ a < c ∧ a < d ∧ b = c ∧ d < b :=
by
  have dec1 := forty_two_percent_of (decimal_conversion 7 11)
  have dec2 := decimal_conversion 17 23
  have dec3 := 0.7391
  have dec4 := decimal_conversion 29 47
  sorry

end greatest_decimal_is_7391_l282_282035


namespace new_recipe_water_l282_282141

theorem new_recipe_water (flour water sugar : ℕ)
  (h_orig : flour = 10 ∧ water = 6 ∧ sugar = 3)
  (h_new : ∀ (new_flour new_water new_sugar : ℕ), 
            new_flour = 10 ∧ new_water = 3 ∧ new_sugar = 3)
  (h_sugar : sugar = 4) :
  new_water = 4 := 
  sorry

end new_recipe_water_l282_282141


namespace x_ge_3_is_necessary_but_not_sufficient_for_x_gt_3_l282_282904

theorem x_ge_3_is_necessary_but_not_sufficient_for_x_gt_3 :
  (∀ x : ℝ, x > 3 → x ≥ 3) ∧ (∃ x : ℝ, x ≥ 3 ∧ ¬ (x > 3)) :=
by
  sorry

end x_ge_3_is_necessary_but_not_sufficient_for_x_gt_3_l282_282904


namespace min_rows_needed_l282_282471

-- Define the basic conditions
def total_students := 2016
def seats_per_row := 168
def max_students_per_school := 40

-- Define the minimum number of rows required to accommodate all conditions
noncomputable def min_required_rows (students : ℕ) (seats : ℕ) (max_per_school : ℕ) : ℕ := 15

-- Lean theorem asserting the truth of the above definition under given conditions
theorem min_rows_needed : min_required_rows total_students seats_per_row max_students_per_school = 15 :=
by
  -- Proof omitted
  sorry

end min_rows_needed_l282_282471


namespace ammunition_explosion_probability_l282_282494

theorem ammunition_explosion_probability :
  (let P_A : ℝ := 0.025,
       P_B : ℝ := 0.1,
       P_C : ℝ := 0.1,
       P_D := 1 - (1 - P_A) * (1 - P_B) * (1 - P_C) in
   P_D = 0.21025) :=
by sorry

end ammunition_explosion_probability_l282_282494


namespace find_values_of_ABC_l282_282024

-- Define the given conditions
def condition1 (A B C : ℕ) : Prop := A + B + C = 36
def condition2 (A B C : ℕ) : Prop := 
  (A + B) * 3 * 4 = (B + C) * 2 * 4 ∧ 
  (B + C) * 2 * 4 = (A + C) * 2 * 3

-- State the problem
theorem find_values_of_ABC (A B C : ℕ) 
  (h1 : condition1 A B C) 
  (h2 : condition2 A B C) : 
  A = 12 ∧ B = 4 ∧ C = 20 :=
sorry

end find_values_of_ABC_l282_282024


namespace percentage_students_receive_valentine_l282_282437

/-- Given the conditions:
  1. There are 30 students.
  2. Mo wants to give a Valentine to some percentage of them.
  3. Each Valentine costs $2.
  4. Mo has $40.
  5. Mo will spend 90% of his money on Valentines.
Prove that the percentage of students receiving a Valentine is 60%.
-/
theorem percentage_students_receive_valentine :
  let total_students := 30
  let valentine_cost := 2
  let total_money := 40
  let spent_percentage := 0.90
  ∃ (cards : ℕ), 
    let money_spent := total_money * spent_percentage
    let cards_bought := money_spent / valentine_cost
    let percentage_students := (cards_bought / total_students) * 100
    percentage_students = 60 := 
by
  sorry

end percentage_students_receive_valentine_l282_282437


namespace complement_U_A_l282_282589

def U := {x : ℝ | x < 2}
def A := {x : ℝ | x^2 < x}

theorem complement_U_A :
  (U \ A) = {x : ℝ | x ≤ 0 ∨ (1 ≤ x ∧ x < 2)} :=
sorry

end complement_U_A_l282_282589


namespace longest_side_of_triangle_l282_282743

theorem longest_side_of_triangle (x : ℝ) (a b c : ℝ)
  (h1 : a = 5)
  (h2 : b = 2 * x + 3)
  (h3 : c = 3 * x - 2)
  (h4 : a + b + c = 41) :
  c = 19 :=
by
  sorry

end longest_side_of_triangle_l282_282743


namespace multiply_square_expression_l282_282511

theorem multiply_square_expression (x : ℝ) : ((-3 * x) ^ 2) * (2 * x) = 18 * x ^ 3 := by
  sorry

end multiply_square_expression_l282_282511


namespace compute_avg_interest_rate_l282_282642

variable (x : ℝ)

/-- The total amount of investment is $5000 - x at 3% and x at 7%. The incomes are equal 
thus we are asked to compute the average rate of interest -/
def avg_interest_rate : Prop :=
  let i_3 := 0.03 * (5000 - x)
  let i_7 := 0.07 * x
  i_3 = i_7 ∧
  (2 * i_3) / 5000 = 0.042

theorem compute_avg_interest_rate 
  (condition : ∃ x : ℝ, 0.03 * (5000 - x) = 0.07 * x) :
  avg_interest_rate x :=
by
  sorry

end compute_avg_interest_rate_l282_282642


namespace ch4_contains_most_atoms_l282_282925

def molecule_atoms (molecule : String) : Nat :=
  match molecule with
  | "O₂"   => 2
  | "NH₃"  => 4
  | "CO"   => 2
  | "CH₄"  => 5
  | _      => 0

theorem ch4_contains_most_atoms :
  ∀ (a b c d : Nat), 
  a = molecule_atoms "O₂" →
  b = molecule_atoms "NH₃" →
  c = molecule_atoms "CO" →
  d = molecule_atoms "CH₄" →
  d > a ∧ d > b ∧ d > c :=
by
  intros
  sorry

end ch4_contains_most_atoms_l282_282925


namespace share_difference_l282_282926

theorem share_difference (F V R P E : ℕ)
  (hratio : 3 * V = 5 * F ∧ 9 * V = 5 * R ∧ 7 * V = 5 * P ∧ 11 * V = 5 * E)
  (hV : V = 3000) :
  (F + R + E) - (V + P) = 6600 :=
by
  sorry

end share_difference_l282_282926


namespace min_value_x_3y_6z_l282_282716

theorem min_value_x_3y_6z (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0 ∧ xyz = 27) : x + 3 * y + 6 * z ≥ 27 :=
sorry

end min_value_x_3y_6z_l282_282716


namespace sum_of_squares_of_roots_eq_21_l282_282957

theorem sum_of_squares_of_roots_eq_21 (a : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 + x2^2 = 21 ∧ x1 + x2 = -a ∧ x1 * x2 = 2*a) ↔ a = -3 :=
by
  sorry

end sum_of_squares_of_roots_eq_21_l282_282957


namespace greatest_possible_difference_l282_282182

def is_reverse (q r : ℕ) : Prop :=
  let q_tens := q / 10
  let q_units := q % 10
  let r_tens := r / 10
  let r_units := r % 10
  (q_tens = r_units) ∧ (q_units = r_tens)

theorem greatest_possible_difference (q r : ℕ) (hq1 : q ≥ 10) (hq2 : q < 100)
  (hr1 : r ≥ 10) (hr2 : r < 100) (hrev : is_reverse q r) (hpos_diff : q - r < 30) :
  q - r ≤ 27 :=
by
  sorry

end greatest_possible_difference_l282_282182


namespace alex_bought_3_bags_of_chips_l282_282672

theorem alex_bought_3_bags_of_chips (x : ℝ) : 
    (1 * x + 5 + 73) / x = 27 → x = 3 := by sorry

end alex_bought_3_bags_of_chips_l282_282672


namespace triangle_angle_sum_l282_282759

theorem triangle_angle_sum (y : ℝ) (h : 40 + 3 * y + (y + 10) = 180) : y = 32.5 :=
by
  sorry

end triangle_angle_sum_l282_282759


namespace least_prime_factor_of_5_to_the_3_minus_5_to_the_2_l282_282482

theorem least_prime_factor_of_5_to_the_3_minus_5_to_the_2 : 
  Nat.minFac (5^3 - 5^2) = 2 := by
  sorry

end least_prime_factor_of_5_to_the_3_minus_5_to_the_2_l282_282482


namespace wrapping_paper_fraction_used_l282_282007

theorem wrapping_paper_fraction_used 
  (total_paper_used : ℚ)
  (num_presents : ℕ)
  (each_present_used : ℚ)
  (h1 : total_paper_used = 1 / 2)
  (h2 : num_presents = 5)
  (h3 : each_present_used = total_paper_used / num_presents) : 
  each_present_used = 1 / 10 := 
by
  sorry

end wrapping_paper_fraction_used_l282_282007


namespace calculate_lassis_from_nine_mangoes_l282_282660

variable (mangoes_lassis_ratio : ℕ → ℕ → Prop)
variable (cost_per_mango : ℕ)

def num_lassis (mangoes : ℕ) : ℕ :=
  5 * mangoes
  
theorem calculate_lassis_from_nine_mangoes
  (h1 : mangoes_lassis_ratio 15 3)
  (h2 : cost_per_mango = 2) :
  num_lassis 9 = 45 :=
by
  sorry

end calculate_lassis_from_nine_mangoes_l282_282660


namespace quadratic_b_value_l282_282518

theorem quadratic_b_value (b : ℝ) (n : ℝ) (h_b_neg : b < 0) 
  (h_equiv : ∀ x : ℝ, (x + n)^2 + 1 / 16 = x^2 + b * x + 1 / 4) : 
  b = - (Real.sqrt 3) / 2 := 
sorry

end quadratic_b_value_l282_282518


namespace questionnaires_drawn_from_unit_D_l282_282725

theorem questionnaires_drawn_from_unit_D 
  (total_sample: ℕ) 
  (sample_from_B: ℕ) 
  (d: ℕ) 
  (h_total_sample: total_sample = 150) 
  (h_sample_from_B: sample_from_B = 30) 
  (h_arithmetic_sequence: (30 - d) + 30 + (30 + d) + (30 + 2 * d) = total_sample) 
  : 30 + 2 * d = 60 :=
by 
  sorry

end questionnaires_drawn_from_unit_D_l282_282725


namespace least_number_divisible_by_11_l282_282166

theorem least_number_divisible_by_11 (n : ℕ) (k : ℕ) (h₁ : n = 2520 * k + 1) (h₂ : 11 ∣ n) : n = 12601 :=
sorry

end least_number_divisible_by_11_l282_282166


namespace sqrt8_sub_sqrt2_eq_sqrt2_l282_282207

theorem sqrt8_sub_sqrt2_eq_sqrt2 : Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end sqrt8_sub_sqrt2_eq_sqrt2_l282_282207


namespace concentric_circles_area_difference_l282_282029

/-- Two concentric circles with radii 12 cm and 7 cm have an area difference of 95π cm² between them. -/
theorem concentric_circles_area_difference :
  let r1 := 12
  let r2 := 7
  let area_larger := Real.pi * r1^2
  let area_smaller := Real.pi * r2^2
  let area_difference := area_larger - area_smaller
  area_difference = 95 * Real.pi := by
sorry

end concentric_circles_area_difference_l282_282029


namespace minimum_oranges_to_profit_l282_282495

/-- 
A boy buys 4 oranges for 12 cents and sells 6 oranges for 25 cents. 
Calculate the minimum number of oranges he needs to sell to make a profit of 150 cents.
--/
theorem minimum_oranges_to_profit (cost_oranges : ℕ) (cost_cents : ℕ)
  (sell_oranges : ℕ) (sell_cents : ℕ) (desired_profit : ℚ) :
  cost_oranges = 4 → cost_cents = 12 →
  sell_oranges = 6 → sell_cents = 25 →
  desired_profit = 150 →
  (∃ n : ℕ, n = 129) :=
by
  sorry

end minimum_oranges_to_profit_l282_282495


namespace wayne_took_cards_l282_282361

-- Let's define the problem context
variable (initial_cards : ℕ := 76)
variable (remaining_cards : ℕ := 17)

-- We need to show that Wayne took away 59 cards
theorem wayne_took_cards (x : ℕ) (h : x = initial_cards - remaining_cards) : x = 59 :=
by
  sorry

end wayne_took_cards_l282_282361


namespace geom_sequence_second_term_l282_282640

noncomputable def geom_sequence_term (a r : ℕ) (n : ℕ) : ℕ := a * r^(n-1)

theorem geom_sequence_second_term 
  (a1 a5: ℕ) (r: ℕ) 
  (h1: a1 = 5)
  (h2: a5 = geom_sequence_term a1 r 5)
  (h3: a5 = 320)
  (h_r: r^4 = 64): 
  geom_sequence_term a1 r 2 = 10 :=
by
  sorry

end geom_sequence_second_term_l282_282640


namespace incorrect_statement_S9_lt_S10_l282_282385

variable {a : ℕ → ℝ} -- Sequence
variable {S : ℕ → ℝ} -- Sum of the first n terms
variable {d : ℝ}     -- Common difference

-- Arithmetic sequence definition
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Sum of the first n terms
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * a 0 + n * (n-1) * d / 2)

-- Given conditions
variable 
  (arith_seq : arithmetic_sequence a d)
  (sum_terms : sum_of_first_n_terms a S)
  (H1 : S 9 < S 8)
  (H2 : S 8 = S 7)

-- Prove the statement
theorem incorrect_statement_S9_lt_S10 : 
  ¬ (S 9 < S 10) := 
sorry

end incorrect_statement_S9_lt_S10_l282_282385


namespace value_of_4_ampersand_neg3_l282_282522

-- Define the operation '&'
def ampersand (x y : Int) : Int :=
  x * (y + 2) + x * y

-- State the theorem
theorem value_of_4_ampersand_neg3 : ampersand 4 (-3) = -16 :=
by
  sorry

end value_of_4_ampersand_neg3_l282_282522


namespace cups_filled_l282_282027

def total_tea : ℕ := 1050
def tea_per_cup : ℕ := 65

theorem cups_filled : Nat.floor (total_tea / (tea_per_cup : ℚ)) = 16 :=
by
  sorry

end cups_filled_l282_282027


namespace problem_1_problem_2_l282_282815

-- Define the functions f and g.
def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp x - a * x) * (x * Real.exp x + Real.sqrt 2)
def g (x : ℝ) : ℝ := x * Real.exp x + Real.sqrt 2

-- Problem 1: Prove that a = 0 given the slope of the tangent line to y = f(x) at (0, f(0))
theorem problem_1 (a : ℝ) (f : ℝ → ℝ) (h : f 0 = (Real.sqrt 2 + 1) ∧ f'(0) = (Real.sqrt 2 + 1)) : a = 0 :=
sorry

-- Problem 2: Prove that g(x) > 1 for all x in ℝ
theorem problem_2 (g : ℝ → ℝ) (h : ∀ x : ℝ, g(x) = x * Real.exp x + Real.sqrt 2) : ∀ x : ℝ, g x > 1 :=
sorry

end problem_1_problem_2_l282_282815


namespace wrapping_paper_per_present_l282_282001

theorem wrapping_paper_per_present :
  (1 / 2) / 5 = 1 / 10 :=
by
  sorry

end wrapping_paper_per_present_l282_282001


namespace range_of_3a_minus_b_l282_282069

theorem range_of_3a_minus_b (a b : ℝ) (ha : -5 < a) (ha' : a < 2) (hb : 1 < b) (hb' : b < 4) : 
  -19 < 3 * a - b ∧ 3 * a - b < 5 :=
by
  sorry

end range_of_3a_minus_b_l282_282069


namespace exists_real_x_for_sequence_floor_l282_282456

open Real

theorem exists_real_x_for_sequence_floor (a : Fin 1998 → ℕ)
  (h1 : ∀ n : Fin 1998, 0 ≤ a n)
  (h2 : ∀ (i j : Fin 1998), (i.val + j.val ≤ 1997) → (a i + a j ≤ a ⟨i.val + j.val, sorry⟩ ∧ a ⟨i.val + j.val, sorry⟩ ≤ a i + a j + 1)) :
  ∃ x : ℝ, ∀ n : Fin 1998, a n = ⌊(n.val + 1) * x⌋ :=
sorry

end exists_real_x_for_sequence_floor_l282_282456


namespace rational_expression_iff_rational_square_l282_282300

theorem rational_expression_iff_rational_square (x : ℝ) :
  (∃ r : ℚ, x^2 + (Real.sqrt (x^4 + 1)) - 1 / (x^2 + (Real.sqrt (x^4 + 1))) = r) ↔
  (∃ q : ℚ, x^2 = q) := by
  sorry

end rational_expression_iff_rational_square_l282_282300


namespace smaller_integer_of_two_digits_l282_282065

theorem smaller_integer_of_two_digits (a b : ℕ) (ha : 10 ≤ a ∧ a ≤ 99) (hb: 10 ≤ b ∧ b ≤ 99) (h_diff : a ≠ b)
  (h_eq : (a + b) / 2 = a + b / 100) : a = 49 ∨ b = 49 := 
by
  sorry

end smaller_integer_of_two_digits_l282_282065


namespace arithmetic_sums_l282_282611

theorem arithmetic_sums (d : ℤ) (p q : ℤ) (S : ℤ → ℤ)
  (hS : ∀ n, S n = p * n^2 + q * n)
  (h_eq : S 20 = S 40) : S 60 = 0 :=
by
  sorry

end arithmetic_sums_l282_282611


namespace find_function_l282_282668

def satisfies_functional_eqn (f : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x * f y) = f (x * y^2) - 2 * x^2 * f y - f x - 1

theorem find_function (f : ℝ → ℝ) :
  satisfies_functional_eqn f → (∀ y : ℝ, f y = y^2 - 1) :=
by
  intro h
  sorry

end find_function_l282_282668


namespace staffing_correct_l282_282363

-- The number of ways to staff a battle station with constraints.
def staffing_ways (total_applicants unsuitable_fraction: ℕ) (job_openings: ℕ): ℕ :=
  let suitable_candidates := total_applicants * (1 - unsuitable_fraction)
  if suitable_candidates < job_openings then
    0 
  else
    (List.range' (suitable_candidates - job_openings + 1) job_openings).prod

-- Definitions of the problem conditions
def total_applicants := 30
def unsuitable_fraction := 2/3
def job_openings := 5
-- Expected result
def expected_result := 30240

-- The theorem to prove the number of ways to staff the battle station equals the given result.
theorem staffing_correct : staffing_ways total_applicants unsuitable_fraction job_openings = expected_result := by
  sorry

end staffing_correct_l282_282363


namespace no_divisor_form_24k_20_l282_282788

theorem no_divisor_form_24k_20 (n : ℕ) : ¬ ∃ k : ℕ, 24 * k + 20 ∣ 3^n + 1 :=
sorry

end no_divisor_form_24k_20_l282_282788


namespace conditional_prob_l282_282040

-- Define the conditions
def set := {1, 2, 3, 4, 5}
def is_odd (n : ℕ) : Prop := n % 2 = 1
def event_A : Set ℕ := { x | x ∈ set ∧ is_odd x }
def event_B {x} : Set ℕ := { y | y ∈ set ∧ is_odd y ∧ y ≠ x }

-- Define the probabilities
def C {α : Type*} [Fintype α] [DecidableEq α] (s : Finset α) (k : ℕ) : ℕ :=
  Nat.choose s.card k

noncomputable def P (s t : Set ℕ) : ℚ :=
  ↑(s.inter t).card / ↑s.card

/-- To prove that the conditional probability P(B|A) equals to 1/2 -/
theorem conditional_prob :
  P (event_A) set = 3 / 5 ∧ P (event_A.inter (event_B 1)) set = 3 / 10 →
  P (event_B 1 | event_A) = 1 / 2 :=
by
  intros h
  sorry

end conditional_prob_l282_282040


namespace hyperbola_eccentricity_is_sqrt_5_l282_282082

noncomputable def hyperbola_eccentricity (m : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  real.sqrt (1 + (b/a)^2) -- Defining eccentricity

theorem hyperbola_eccentricity_is_sqrt_5
  (m : ℝ) (h : m = 4) : 
  hyperbola_eccentricity m (1/2) 1 = real.sqrt 5 :=
  by {
      sorry
  } 

end hyperbola_eccentricity_is_sqrt_5_l282_282082


namespace reduced_price_l282_282897

theorem reduced_price (P R : ℝ) (h1 : R = 0.8 * P) (h2 : 600 = (600 / P + 4) * R) : R = 30 := 
by
  sorry

end reduced_price_l282_282897


namespace students_in_only_one_subject_l282_282866

variables (A B C : ℕ) 
variables (A_inter_B A_inter_C B_inter_C A_inter_B_inter_C : ℕ)

def students_in_one_subject (A B C A_inter_B A_inter_C B_inter_C A_inter_B_inter_C : ℕ) : ℕ :=
  A + B + C - A_inter_B - A_inter_C - B_inter_C + A_inter_B_inter_C - 2 * A_inter_B_inter_C

theorem students_in_only_one_subject :
  ∀ (A B C A_inter_B A_inter_C B_inter_C A_inter_B_inter_C : ℕ),
    A = 29 →
    B = 28 →
    C = 27 →
    A_inter_B = 13 →
    A_inter_C = 12 →
    B_inter_C = 11 →
    A_inter_B_inter_C = 5 →
    students_in_one_subject A B C A_inter_B A_inter_C B_inter_C A_inter_B_inter_C = 27 :=
by
  intros A B C A_inter_B A_inter_C B_inter_C A_inter_B_inter_C hA hB hC hAB hAC hBC hABC
  unfold students_in_one_subject
  rw [hA, hB, hC, hAB, hAC, hBC, hABC]
  norm_num
  sorry

end students_in_only_one_subject_l282_282866


namespace harry_total_expenditure_l282_282915

theorem harry_total_expenditure :
  let pumpkin_price := 2.50
  let tomato_price := 1.50
  let chili_pepper_price := 0.90
  let pumpkin_packets := 3
  let tomato_packets := 4
  let chili_pepper_packets := 5
  (pumpkin_packets * pumpkin_price) + (tomato_packets * tomato_price) + (chili_pepper_packets * chili_pepper_price) = 18.00 :=
by
  sorry

end harry_total_expenditure_l282_282915


namespace jiwon_walk_distance_l282_282415

theorem jiwon_walk_distance : 
  (13 * 90) * 0.45 = 526.5 := by
  sorry

end jiwon_walk_distance_l282_282415


namespace original_numbers_geometric_sequence_l282_282622

theorem original_numbers_geometric_sequence (a q : ℝ) :
  (2 * (a * q + 8) = a + a * q^2) →
  ((a * q + 8) ^ 2 = a * (a * q^2 + 64)) →
  (a, a * q, a * q^2) = (4, 12, 36) ∨ (a, a * q, a * q^2) = (4 / 9, -20 / 9, 100 / 9) :=
by {
  sorry
}

end original_numbers_geometric_sequence_l282_282622


namespace sequence_u5_eq_27_l282_282282

theorem sequence_u5_eq_27 (u : ℕ → ℝ) 
  (h_recurrence : ∀ n, u (n + 2) = 3 * u (n + 1) - 2 * u n)
  (h_u3 : u 3 = 15)
  (h_u6 : u 6 = 43) :
  u 5 = 27 :=
  sorry

end sequence_u5_eq_27_l282_282282


namespace greatest_divisor_form_p_plus_1_l282_282266

theorem greatest_divisor_form_p_plus_1 (n : ℕ) (hn : 0 < n):
  (∀ p : ℕ, Nat.Prime p → p % 3 = 2 → ¬ (p ∣ n) → 6 ∣ (p + 1)) ∧
  (∀ d : ℕ, (∀ p : ℕ, Nat.Prime p → p % 3 = 2 → ¬ (p ∣ n) → d ∣ (p + 1)) → d ≤ 6) :=
by {
  sorry
}

end greatest_divisor_form_p_plus_1_l282_282266


namespace tan_half_angle_product_zero_l282_282393

theorem tan_half_angle_product_zero (a b : ℝ) 
  (h: 6 * (Real.cos a + Real.cos b) + 3 * (Real.cos a * Real.cos b + 1) = 0) 
  : Real.tan (a / 2) * Real.tan (b / 2) = 0 := 
by 
  sorry

end tan_half_angle_product_zero_l282_282393


namespace probability_red_white_green_probability_any_order_l282_282358

-- Definitions based on the conditions
def total_balls := 28
def red_balls := 15
def white_balls := 9
def green_balls := 4

-- Part (a): Probability of first red, second white, third green
theorem probability_red_white_green : 
  (red_balls / total_balls) * (white_balls / (total_balls - 1)) * (green_balls / (total_balls - 2)) = 5 / 182 :=
by 
  sorry

-- Part (b): Probability of red, white, and green in any order
theorem probability_any_order :
  6 * ((red_balls / total_balls) * (white_balls / (total_balls - 1)) * (green_balls / (total_balls - 2))) = 15 / 91 :=
by
  sorry

end probability_red_white_green_probability_any_order_l282_282358


namespace part1_part2_l282_282524

def op (a b : ℝ) : ℝ := 2 * a - 3 * b

theorem part1 : op (-2) 3 = -13 :=
by
  -- sorry step to skip proof
  sorry

theorem part2 (x : ℝ) :
  let A := op (3 * x - 2) (x + 1)
  let B := op (-3 / 2 * x + 1) (-1 - 2 * x)
  B > A :=
by
  -- sorry step to skip proof
  sorry

end part1_part2_l282_282524


namespace unit_digit_of_product_is_4_l282_282146

theorem unit_digit_of_product_is_4 :
  let expr := ((2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)) - 1 in
  expr % 10 = 4 :=
by
  -- define the expression 
  let expr : ℕ := ((2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)) - 1
  -- ensure the equivalence of unit digit
  show expr % 10 = 4
  sorry -- proof goes here

end unit_digit_of_product_is_4_l282_282146


namespace find_time_for_products_maximize_salary_l282_282762

-- Assume the conditions and definitions based on the given problem
variables (x y a : ℝ)

-- Condition 1: Time to produce 6 type A and 4 type B products is 170 minutes
axiom cond1 : 6 * x + 4 * y = 170

-- Condition 2: Time to produce 10 type A and 10 type B products is 350 minutes
axiom cond2 : 10 * x + 10 * y = 350


-- Question 1: Validating the time to produce one type A product and one type B product
theorem find_time_for_products : 
  x = 15 ∧ y = 20 := by
  sorry

-- Variables for calculation of Zhang's daily salary
variables (m : ℕ) (base_salary : ℝ := 100) (daily_work: ℝ := 480)

-- Conditions for the piece-rate wages
variables (a_condition: 2 < a ∧ a < 3) 
variables (num_products: m + (28 - m) = 28)

-- Question 2: Finding optimal production plan to maximize daily salary
theorem maximize_salary :
  (2 < a ∧ a < 2.5) → m = 16 ∨ 
  (a = 2.5) → true ∨
  (2.5 < a ∧ a < 3) → m = 28 := by
  sorry

end find_time_for_products_maximize_salary_l282_282762


namespace blue_whale_tongue_weight_in_tons_l282_282872

-- Define the conditions
def weight_of_tongue_pounds : ℕ := 6000
def pounds_per_ton : ℕ := 2000

-- Define the theorem stating the question and its answer
theorem blue_whale_tongue_weight_in_tons :
  (weight_of_tongue_pounds / pounds_per_ton) = 3 :=
by sorry

end blue_whale_tongue_weight_in_tons_l282_282872


namespace problem_dividing_remainder_l282_282909

-- The conditions exported to Lean
def tiling_count (n : ℕ) : ℕ :=
  -- This function counts the number of valid tilings for a board size n with all colors used
  sorry

def remainder_when_divide (num divisor : ℕ) : ℕ := num % divisor

-- The statement problem we need to prove
theorem problem_dividing_remainder :
  remainder_when_divide (tiling_count 9) 1000 = 545 := 
sorry

end problem_dividing_remainder_l282_282909


namespace trapezoid_equilateral_triangle_ratio_l282_282927

theorem trapezoid_equilateral_triangle_ratio (s d : ℝ) (AB CD : ℝ) 
  (h1 : AB = s) 
  (h2 : CD = 2 * d)
  (h3 : d = s) : 
  AB / CD = 1 / 2 := 
by
  sorry

end trapezoid_equilateral_triangle_ratio_l282_282927


namespace part1_part2_l282_282523

def op (a b : ℝ) : ℝ := 2 * a - 3 * b

theorem part1 : op (-2) 3 = -13 :=
by
  -- sorry step to skip proof
  sorry

theorem part2 (x : ℝ) :
  let A := op (3 * x - 2) (x + 1)
  let B := op (-3 / 2 * x + 1) (-1 - 2 * x)
  B > A :=
by
  -- sorry step to skip proof
  sorry

end part1_part2_l282_282523


namespace side_length_of_square_l282_282562

theorem side_length_of_square (P : ℝ) (h1 : P = 12 / 25) : 
  P / 4 = 0.12 := 
by
  sorry

end side_length_of_square_l282_282562


namespace first_part_lent_years_l282_282780

theorem first_part_lent_years (P P1 P2 : ℝ) (rate1 rate2 : ℝ) (years2 : ℝ) (interest1 interest2 : ℝ) (t : ℝ) 
  (h1 : P = 2717)
  (h2 : P2 = 1672)
  (h3 : P1 = P - P2)
  (h4 : rate1 = 0.03)
  (h5 : rate2 = 0.05)
  (h6 : years2 = 3)
  (h7 : interest1 = P1 * rate1 * t)
  (h8 : interest2 = P2 * rate2 * years2)
  (h9 : interest1 = interest2) :
  t = 8 :=
sorry

end first_part_lent_years_l282_282780


namespace toothpaste_usage_l282_282880

-- Define the variables involved
variables (t : ℕ) -- total toothpaste in grams
variables (d : ℕ) -- grams used by dad per brushing
variables (m : ℕ) -- grams used by mom per brushing
variables (b : ℕ) -- grams used by Anne + brother per brushing
variables (r : ℕ) -- brushing rate per day
variables (days : ℕ) -- days for toothpaste to run out
variables (N : ℕ) -- family members

-- Given conditions
variables (ht : t = 105)         -- Total toothpaste is 105 grams
variables (hd : d = 3)           -- Dad uses 3 grams per brushing
variables (hm : m = 2)           -- Mom uses 2 grams per brushing
variables (hr : r = 3)           -- Each member brushes three times a day
variables (hdays : days = 5)     -- Toothpaste runs out in 5 days

-- Additional calculations
variable (total_brushing : ℕ)
variable (total_usage_d: ℕ)
variable (total_usage_m: ℕ)
variable (total_usage_parents: ℕ)
variable (total_usage_family: ℕ)

-- Helper expressions
def total_brushing_expr := days * r * 2
def total_usage_d_expr := d * r
def total_usage_m_expr := m * r
def total_usage_parents_expr := (total_usage_d_expr + total_usage_m_expr) * days
def total_usage_family_expr := t - total_usage_parents_expr

-- Assume calculations
variables (h1: total_usage_d = total_usage_d_expr)  
variables (h2: total_usage_m = total_usage_m_expr)
variables (h3: total_usage_parents = total_usage_parents_expr)
variables (h4: total_usage_family = total_usage_family_expr)
variables (h5 : total_brushing = total_brushing_expr)

-- Define the proof
theorem toothpaste_usage : 
  b = total_usage_family / total_brushing := 
  sorry

end toothpaste_usage_l282_282880


namespace simplify_and_evaluate_l282_282451

variable (a : ℚ)
variable (a_val : a = -1/2)

theorem simplify_and_evaluate : (4 - 3 * a) * (1 + 2 * a) - 3 * a * (1 - 2 * a) = 3 := by
  sorry

end simplify_and_evaluate_l282_282451


namespace value_of_x2_minus_y2_l282_282833

-- Define the variables x and y as real numbers
variables (x y : ℝ)

-- State the conditions
def condition1 : Prop := (x + y) / 2 = 5
def condition2 : Prop := (x - y) / 2 = 2

-- State the theorem to prove
theorem value_of_x2_minus_y2 (h1 : condition1 x y) (h2 : condition2 x y) : x^2 - y^2 = 40 :=
by
  sorry

end value_of_x2_minus_y2_l282_282833


namespace factor_x8_minus_81_l282_282058

theorem factor_x8_minus_81 (x : ℝ) : x^8 - 81 = (x^2 - 3) * (x^2 + 3) * (x^4 + 9) := 
by 
  sorry

end factor_x8_minus_81_l282_282058


namespace customer_payment_eq_3000_l282_282296

theorem customer_payment_eq_3000 (cost_price : ℕ) (markup_percentage : ℕ) (payment : ℕ)
  (h1 : cost_price = 2500)
  (h2 : markup_percentage = 20)
  (h3 : payment = cost_price + (markup_percentage * cost_price / 100)) :
  payment = 3000 :=
by
  sorry

end customer_payment_eq_3000_l282_282296


namespace smaller_angle_parallelogram_l282_282253

theorem smaller_angle_parallelogram (x : ℕ) (h1 : ∀ a b : ℕ, a ≠ b ∧ a + b = 180) (h2 : ∃ y : ℕ, y = x + 70) : x = 55 :=
by
  sorry

end smaller_angle_parallelogram_l282_282253


namespace simplify_expression_l282_282515

theorem simplify_expression (x : ℝ) : x^2 * x^4 + x * x^2 * x^3 = 2 * x^6 := by
  sorry

end simplify_expression_l282_282515


namespace einstein_needs_more_money_l282_282160

-- Definitions based on conditions
def pizza_price : ℝ := 12
def fries_price : ℝ := 0.3
def soda_price : ℝ := 2
def pizzas_sold : ℕ := 15
def fries_sold : ℕ := 40
def sodas_sold : ℕ := 25
def goal : ℝ := 500

-- Total amount raised calculation
def total_raised : ℝ :=
  (pizzas_sold * pizza_price) +
  (fries_sold * fries_price) +
  (sodas_sold * soda_price)

-- Proof statement
theorem einstein_needs_more_money : goal - total_raised = 258 :=
by
  sorry

end einstein_needs_more_money_l282_282160


namespace greatest_common_factor_of_two_digit_palindromes_is_11_l282_282322

-- Define a two-digit palindrome
def is_two_digit_palindrome (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ (n / 10 = n % 10)

-- Define the GCD of the set of all such numbers
def GCF_two_digit_palindromes : ℕ :=
  gcd (11 * 1) (gcd (11 * 2) (gcd (11 * 3) (gcd (11 * 4)
  (gcd (11 * 5) (gcd (11 * 6) (gcd (11 * 7) (gcd (11 * 8) (11 * 9))))))))

-- The statement to prove
theorem greatest_common_factor_of_two_digit_palindromes_is_11 :
  GCF_two_digit_palindromes = 11 :=
by
  sorry

end greatest_common_factor_of_two_digit_palindromes_is_11_l282_282322


namespace mul_equiv_l282_282037

theorem mul_equiv :
  (213 : ℝ) * 16 = 3408 →
  (16 : ℝ) * 21.3 = 340.8 :=
by
  sorry

end mul_equiv_l282_282037


namespace find_S_l282_282041

variable {R S T c : ℝ}

theorem find_S
  (h1 : R = 2)
  (h2 : T = 1/2)
  (h3 : S = 4)
  (h4 : R = c * S / T)
  (h5 : R = 8)
  (h6 : T = 1/3) :
  S = 32 / 3 :=
by
  sorry

end find_S_l282_282041


namespace find_positive_integer_solutions_l282_282529

def is_solution (x y : ℕ) : Prop :=
  4 * x^3 + 4 * x^2 * y - 15 * x * y^2 - 18 * y^3 - 12 * x^2 + 6 * x * y + 36 * y^2 + 5 * x - 10 * y = 0

theorem find_positive_integer_solutions :
  ∀ x y : ℕ, 0 < x ∧ 0 < y → (is_solution x y ↔ (x = 1 ∧ y = 1) ∨ (∃ y', y = y' ∧ x = 2 * y' ∧ 0 < y')) :=
by
  intros x y hxy
  sorry

end find_positive_integer_solutions_l282_282529


namespace largest_share_received_l282_282010

theorem largest_share_received (total_profit : ℝ) (ratios : List ℝ) (h_ratios : ratios = [1, 2, 2, 3, 4, 5]) 
  (h_profit : total_profit = 51000) : 
  let parts := ratios.sum 
  let part_value := total_profit / parts
  let largest_share := 5 * part_value 
  largest_share = 15000 := 
by 
  sorry

end largest_share_received_l282_282010


namespace intersection_complement_eq_l282_282854

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}
def P : Finset ℕ := {1, 2, 3, 4}
def Q : Finset ℕ := {3, 4, 5}
def U_complement_Q : Finset ℕ := U \ Q

theorem intersection_complement_eq : P ∩ U_complement_Q = {1, 2} :=
by {
  sorry
}

end intersection_complement_eq_l282_282854


namespace max_lambda_inequality_l282_282837

theorem max_lambda_inequality (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  (1 / Real.sqrt (20 * a + 23 * b) + 1 / Real.sqrt (23 * a + 20 * b)) ≥ (2 / Real.sqrt 43 / Real.sqrt (a + b)) :=
by
  sorry

end max_lambda_inequality_l282_282837


namespace jason_arms_tattoos_l282_282351

variable (x : ℕ)

def jason_tattoos (x : ℕ) : ℕ := 2 * x + 3 * 2

def adam_tattoos (x : ℕ) : ℕ := 3 + 2 * (jason_tattoos x)

theorem jason_arms_tattoos : adam_tattoos x = 23 → x = 2 := by
  intro h
  sorry

end jason_arms_tattoos_l282_282351


namespace speed_of_man_in_still_water_l282_282338

def upstream_speed : ℝ := 32
def downstream_speed : ℝ := 48

theorem speed_of_man_in_still_water :
  (upstream_speed + (downstream_speed - upstream_speed) / 2) = 40 :=
by
  sorry

end speed_of_man_in_still_water_l282_282338


namespace problem1_problem2_problem3_l282_282237

noncomputable 
def f (x : ℝ) : ℝ := Real.exp x

theorem problem1 
  (a b : ℝ)
  (h1 : f 1 = a) 
  (h2 : b = 0) : f x = Real.exp x :=
sorry

theorem problem2 
  (k : ℝ) 
  (h : ∀ x : ℝ, f x ≥ k * x) : 0 ≤ k ∧ k ≤ Real.exp 1 :=
sorry

theorem problem3 
  (t : ℝ)
  (h : t ≤ 2) : ∀ x : ℝ, f x > t + Real.log x :=
sorry

end problem1_problem2_problem3_l282_282237


namespace cubed_gt_if_gt_l282_282677

theorem cubed_gt_if_gt {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end cubed_gt_if_gt_l282_282677


namespace squared_greater_abs_greater_l282_282543

theorem squared_greater_abs_greater {a b : ℝ} : a^2 > b^2 ↔ |a| > |b| :=
by sorry

end squared_greater_abs_greater_l282_282543


namespace sum_products_of_chords_l282_282570

variable {r x y u v : ℝ}

theorem sum_products_of_chords (h1 : x * y = u * v) (h2 : 4 * r^2 = (x + y)^2 + (u + v)^2) :
  x * (x + y) + u * (u + v) = 4 * r^2 := by
sorry

end sum_products_of_chords_l282_282570


namespace unit_digit_of_expression_is_4_l282_282147

theorem unit_digit_of_expression_is_4 :
  Nat.unitsDigit ((2+1) * (2^2+1) * (2^4+1) * (2^8+1) * (2^16+1) * (2^32+1) - 1) = 4 :=
by
  sorry

end unit_digit_of_expression_is_4_l282_282147


namespace toms_crab_buckets_l282_282161

def crabs_per_bucket := 12
def price_per_crab := 5
def weekly_earnings := 3360

theorem toms_crab_buckets : (weekly_earnings / (crabs_per_bucket * price_per_crab)) = 56 := by
  sorry

end toms_crab_buckets_l282_282161


namespace min_rows_needed_l282_282473

-- Define the basic conditions
def total_students := 2016
def seats_per_row := 168
def max_students_per_school := 40

-- Define the minimum number of rows required to accommodate all conditions
noncomputable def min_required_rows (students : ℕ) (seats : ℕ) (max_per_school : ℕ) : ℕ := 15

-- Lean theorem asserting the truth of the above definition under given conditions
theorem min_rows_needed : min_required_rows total_students seats_per_row max_students_per_school = 15 :=
by
  -- Proof omitted
  sorry

end min_rows_needed_l282_282473


namespace jake_spent_more_l282_282353

def cost_of_balloons (helium_count : ℕ) (foil_count : ℕ) (helium_price : ℝ) (foil_price : ℝ) : ℝ :=
  helium_count * helium_price + foil_count * foil_price

theorem jake_spent_more 
  (allan_helium : ℕ) (allan_foil : ℕ) (jake_helium : ℕ) (jake_foil : ℕ)
  (helium_price : ℝ) (foil_price : ℝ)
  (h_allan_helium : allan_helium = 2) (h_allan_foil : allan_foil = 3) 
  (h_jake_helium : jake_helium = 4) (h_jake_foil : jake_foil = 2)
  (h_helium_price : helium_price = 1.5) (h_foil_price : foil_price = 2.5) :
  cost_of_balloons jake_helium jake_foil helium_price foil_price - 
  cost_of_balloons allan_helium allan_foil helium_price foil_price = 0.5 := 
by
  sorry

end jake_spent_more_l282_282353


namespace days_wages_l282_282647

theorem days_wages (S W_a W_b : ℝ) 
    (h1 : S = 28 * W_b) 
    (h2 : S = 12 * (W_a + W_b)) 
    (h3 : S = 21 * W_a) : 
    true := 
by sorry

end days_wages_l282_282647


namespace correct_quotient_divide_8_l282_282501

theorem correct_quotient_divide_8 (N : ℕ) (Q : ℕ) 
  (h1 : N = 7 * 12 + 5) 
  (h2 : N / 8 = Q) : 
  Q = 11 := 
by
  sorry

end correct_quotient_divide_8_l282_282501


namespace percentage_of_loss_is_15_percent_l282_282779

/-- 
Given:
  SP₁ = 168 -- Selling price when gaining 20%
  Gain = 20% 
  SP₂ = 119 -- Selling price when calculating loss

Prove:
  The percentage of loss when the article is sold for Rs. 119 is 15%
--/

noncomputable def percentage_loss (CP SP₂: ℝ) : ℝ :=
  ((CP - SP₂) / CP) * 100

theorem percentage_of_loss_is_15_percent (CP SP₂ SP₁: ℝ) (Gain: ℝ):
  CP = 140 ∧ SP₁ = 168 ∧ SP₂ = 119 ∧ Gain = 20 → percentage_loss CP SP₂ = 15 :=
by
  intro h
  sorry

end percentage_of_loss_is_15_percent_l282_282779


namespace solve_for_y_l282_282695

theorem solve_for_y (y : ℝ) (h : (1 / 4) - (1 / 6) = 2 / y) : y = 24 :=
sorry

end solve_for_y_l282_282695


namespace XAXAXA_divisible_by_seven_l282_282275

theorem XAXAXA_divisible_by_seven (X A : ℕ) (hX : X < 10) (hA : A < 10) : 
  (101010 * X + 10101 * A) % 7 = 0 := 
by 
  sorry

end XAXAXA_divisible_by_seven_l282_282275


namespace maximum_marks_l282_282051

-- Definitions based on the conditions
def passing_percentage : ℝ := 0.5
def student_marks : ℝ := 200
def marks_to_pass : ℝ := student_marks + 20

-- Lean 4 statement for the proof problem
theorem maximum_marks (M : ℝ) 
  (h1 : marks_to_pass = 220)
  (h2 : passing_percentage * M = marks_to_pass) :
  M = 440 :=
sorry

end maximum_marks_l282_282051


namespace julia_total_watches_l282_282418

-- Definitions based on conditions.
def silver_watches : Nat := 20
def bronze_watches : Nat := 3 * silver_watches
def total_silver_bronze_watches : Nat := silver_watches + bronze_watches
def gold_watches : Nat := total_silver_bronze_watches / 10

-- The final proof statement without providing the proof.
theorem julia_total_watches : (silver_watches + bronze_watches + gold_watches) = 88 :=
by 
  -- Since we don't need to provide the actual proof, we use sorry
  sorry

end julia_total_watches_l282_282418


namespace quadratic_expression_value_l282_282582

-- Given conditions
variables (a : ℝ) (h : 2 * a^2 + 3 * a - 2022 = 0)

-- Prove the main statement
theorem quadratic_expression_value :
  2 - 6 * a - 4 * a^2 = -4042 :=
sorry

end quadratic_expression_value_l282_282582


namespace probability_reach_2_3_in_7_steps_l282_282602

theorem probability_reach_2_3_in_7_steps (q : ℚ) (m n : ℕ) (h_rel_prime : Nat.coprime m n)
  (h_q : q = 179 / 8192) (h_frac : Rat.mk_nat m n = q) :
  m + n = 8371 := by
  sorry

end probability_reach_2_3_in_7_steps_l282_282602


namespace Hilltown_Volleyball_Club_Members_l282_282017

-- Definitions corresponding to the conditions
def knee_pad_cost : ℕ := 6
def uniform_cost : ℕ := 14
def total_expenditure : ℕ := 4000

-- Definition of total cost per member
def cost_per_member : ℕ := 2 * (knee_pad_cost + uniform_cost)

-- Proof statement
theorem Hilltown_Volleyball_Club_Members :
  total_expenditure % cost_per_member = 0 ∧ total_expenditure / cost_per_member = 100 := by
    sorry

end Hilltown_Volleyball_Club_Members_l282_282017


namespace maximum_value_expression_maximum_value_expression_achieved_l282_282952

theorem maximum_value_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  (1 / (x^2 - 4 * x + 9) + 1 / (y^2 - 4 * y + 9) + 1 / (z^2 - 4 * z + 9)) ≤ 7 / 18 :=
sorry

theorem maximum_value_expression_achieved :
  (1 / (0^2 - 4 * 0 + 9) + 1 / (0^2 - 4 * 0 + 9) + 1 / (1^2 - 4 * 1 + 9)) = 7 / 18 :=
sorry

end maximum_value_expression_maximum_value_expression_achieved_l282_282952


namespace students_with_same_grade_l282_282144

theorem students_with_same_grade :
  let total_students := 40
  let students_with_same_A := 3
  let students_with_same_B := 2
  let students_with_same_C := 6
  let students_with_same_D := 1
  let total_same_grade_students := students_with_same_A + students_with_same_B + students_with_same_C + students_with_same_D
  total_same_grade_students = 12 →
  (total_same_grade_students / total_students) * 100 = 30 :=
by
  sorry

end students_with_same_grade_l282_282144


namespace original_strength_of_class_l282_282867

-- Definitions from the problem conditions
def average_age_original (x : ℕ) : ℕ := 40 * x
def total_students (x : ℕ) : ℕ := x + 17
def total_age_new_students : ℕ := 17 * 32
def new_average_age : ℕ := 36

-- Lean statement to prove that the original strength of the class is 17.
theorem original_strength_of_class :
  ∃ x : ℕ, average_age_original x + total_age_new_students = total_students x * new_average_age ∧ x = 17 :=
by
  sorry

end original_strength_of_class_l282_282867


namespace k_values_for_perpendicular_lines_l282_282460

-- Definition of perpendicular condition for lines
def perpendicular_lines (k : ℝ) : Prop :=
  k * (k - 1) + (1 - k) * (2 * k + 3) = 0

-- Lean 4 statement representing the math proof problem
theorem k_values_for_perpendicular_lines (k : ℝ) :
  perpendicular_lines k ↔ k = -3 ∨ k = 1 :=
by
  sorry

end k_values_for_perpendicular_lines_l282_282460


namespace smallest_number_condition_l282_282631

def smallest_number := 1621432330
def primes := [29, 53, 37, 41, 47, 61]
def lcm_of_primes := primes.prod

theorem smallest_number_condition :
  ∃ k : ℕ, 5 * (smallest_number + 11) = k * lcm_of_primes ∧
          (∀ y, (∃ m : ℕ, 5 * (y + 11) = m * lcm_of_primes) → smallest_number ≤ y) :=
by
  -- The proof goes here
  sorry

#print smallest_number_condition

end smallest_number_condition_l282_282631


namespace total_profit_l282_282654

theorem total_profit (C_profit : ℝ) (x : ℝ) (h1 : 4 * x = 48000) : 12 * x = 144000 :=
by
  sorry

end total_profit_l282_282654


namespace total_animals_correct_l282_282594

def initial_cows : ℕ := 2
def initial_pigs : ℕ := 3
def initial_goats : ℕ := 6

def added_cows : ℕ := 3
def added_pigs : ℕ := 5
def added_goats : ℕ := 2

def total_cows : ℕ := initial_cows + added_cows
def total_pigs : ℕ := initial_pigs + added_pigs
def total_goats : ℕ := initial_goats + added_goats

def total_animals : ℕ := total_cows + total_pigs + total_goats

theorem total_animals_correct : total_animals = 21 := by
  sorry

end total_animals_correct_l282_282594


namespace sum_of_digits_base_8_rep_of_888_l282_282171

theorem sum_of_digits_base_8_rep_of_888 : 
  let n := 888
  let base_8_rep := (1, 5, 7, 0)
  let digit_sum := 1 + 5 + 7 + 0
  is_base_8_rep (n : ℕ) (base_8_rep : ℕ × ℕ × ℕ × ℕ) → 
  digit_sum = 13 := 
by
  sorry

end sum_of_digits_base_8_rep_of_888_l282_282171


namespace denomination_of_checks_l282_282784

-- Definitions based on the conditions.
def total_checks := 30
def total_worth := 1800
def checks_spent := 24
def average_remaining := 100

-- Statement to be proven.
theorem denomination_of_checks :
  ∃ x : ℝ, (total_checks - checks_spent) * average_remaining + checks_spent * x = total_worth ∧ x = 40 :=
by
  sorry

end denomination_of_checks_l282_282784


namespace algebraic_expression_value_l282_282094

theorem algebraic_expression_value (m : ℝ) (h : m^2 - m - 2 = 0) : 2 * m^2 - 2 * m = 4 := by
  sorry

end algebraic_expression_value_l282_282094


namespace total_stamps_collected_l282_282085

-- Conditions
def harry_stamps : ℕ := 180
def sister_stamps : ℕ := 60
def harry_three_times_sister : harry_stamps = 3 * sister_stamps := 
  by
  sorry  -- Proof will show that 180 = 3 * 60 (provided for completeness)

-- Statement to prove
theorem total_stamps_collected : harry_stamps + sister_stamps = 240 :=
  by
  sorry

end total_stamps_collected_l282_282085


namespace no_positive_integer_solutions_l282_282797

theorem no_positive_integer_solutions (x y z : ℕ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) : x^2 + y^2 ≠ 7 * z^2 := by
  sorry

end no_positive_integer_solutions_l282_282797


namespace smallest_fraction_num_greater_than_four_fifths_with_2_digit_nums_l282_282355

theorem smallest_fraction_num_greater_than_four_fifths_with_2_digit_nums :
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ (a : ℚ) / b > 4 / 5 ∧ Int.gcd a b = 1 ∧ a = 77 :=
by {
    sorry
}

end smallest_fraction_num_greater_than_four_fifths_with_2_digit_nums_l282_282355


namespace wrapping_paper_per_present_l282_282002

theorem wrapping_paper_per_present :
  (1 / 2) / 5 = 1 / 10 :=
by
  sorry

end wrapping_paper_per_present_l282_282002


namespace change_received_l282_282877

variable (a : ℝ)

theorem change_received (h : a ≤ 30) : 100 - 3 * a = 100 - 3 * a :=
by
  sorry

end change_received_l282_282877


namespace wrapping_paper_each_present_l282_282004

theorem wrapping_paper_each_present (total_paper : ℚ) (num_presents : ℕ)
  (h1 : total_paper = 1 / 2) (h2 : num_presents = 5) :
  (total_paper / num_presents = 1 / 10) :=
by
  sorry

end wrapping_paper_each_present_l282_282004


namespace math_problem_l282_282997

def f (x : ℝ) : ℝ := sorry

theorem math_problem (n s : ℕ)
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x - f y))
  (hn : n = 1)
  (hs : s = 6) :
  n * s = 6 := by
  sorry

end math_problem_l282_282997


namespace three_digit_numbers_not_multiple_of_3_or_11_l282_282556

theorem three_digit_numbers_not_multiple_of_3_or_11 : 
  let total_three_digit_numbers := 999 - 100 + 1 in
  let multiples_of_3 := 333 - 34 + 1 in
  let multiples_of_11 := 90 - 10 + 1 in
  let multiples_of_33 := 30 - 4 + 1 in
  let multiples_of_3_or_11 := multiples_of_3 + multiples_of_11 - multiples_of_33 in
  total_three_digit_numbers - multiples_of_3_or_11 = 546 :=
by
  let total_three_digit_numbers := 999 - 100 + 1
  let multiples_of_3 := 333 - 34 + 1
  let multiples_of_11 := 90 - 10 + 1
  let multiples_of_33 := 30 - 4 + 1
  let multiples_of_3_or_11 := multiples_of_3 + multiples_of_11 - multiples_of_33
  show total_three_digit_numbers - multiples_of_3_or_11 = 546 from sorry

end three_digit_numbers_not_multiple_of_3_or_11_l282_282556


namespace division_and_multiplication_l282_282888

theorem division_and_multiplication (a b c d : ℝ) : (a / b / c * d) = 30 :=
by 
  let a := 120
  let b := 6
  let c := 2
  let d := 3
  sorry

end division_and_multiplication_l282_282888


namespace total_chess_games_played_l282_282183

theorem total_chess_games_played : finset.card (finset.pairs_of_card 2 (finset.range 10)) = 45 := by
  sorry

end total_chess_games_played_l282_282183


namespace Carly_applications_l282_282516

theorem Carly_applications (x : ℕ) (h1 : ∀ y, y = 2 * x) (h2 : x + 2 * x = 600) : x = 200 :=
sorry

end Carly_applications_l282_282516


namespace clock_shows_l282_282798

-- Definitions for the hands and their positions
variables {A B C : ℕ} -- Representing hands A, B, and C as natural numbers for simplicity

-- Conditions based on the problem description:
-- 1. Hands A and B point exactly at the hour markers.
-- 2. Hand C is slightly off from an hour marker.
axiom hand_A_hour_marker : A % 12 = A
axiom hand_B_hour_marker : B % 12 = B
axiom hand_C_slightly_off : C % 12 ≠ C

-- Theorem stating that given these conditions, the clock shows the time 4:50
theorem clock_shows (h1: A % 12 = A) (h2: B % 12 = B) (h3: C % 12 ≠ C) : A = 50 ∧ B = 12 ∧ C = 4 :=
sorry

end clock_shows_l282_282798


namespace students_table_tennis_not_basketball_l282_282618

variable (total_students : ℕ)
variable (students_like_basketball : ℕ)
variable (students_like_table_tennis : ℕ)
variable (students_dislike_both : ℕ)

theorem students_table_tennis_not_basketball 
  (h_total : total_students = 40)
  (h_basketball : students_like_basketball = 17)
  (h_table_tennis : students_like_table_tennis = 20)
  (h_dislike : students_dislike_both = 8) : 
  ∃ (students_table_tennis_not_basketball : ℕ), students_table_tennis_not_basketball = 15 :=
by
  sorry

end students_table_tennis_not_basketball_l282_282618


namespace partition_cities_l282_282404

theorem partition_cities (k : ℕ) (V : Type) [fintype V] [decidable_eq V]
  (E : V → V → Prop) [decidable_rel E]
  (flight_company : E → fin k)
  (common_endpoint : ∀ e1 e2 : E, flight_company e1 = flight_company e2 → ∃ v : V, E v v) :
  ∃ (partition : fin (k + 2) → set V), ∀ (i : fin (k + 2)), ∀ (v1 v2 : V), 
  v1 ∈ partition i → v2 ∈ partition i → ¬E v1 v2 :=
sorry

end partition_cities_l282_282404


namespace square_area_ratio_l282_282723

theorem square_area_ratio (n : ℕ) (s₁ s₂: ℕ) (h1 : s₁ = 1) (h2 : s₂ = n^2) (h3 : 2 * s₂ - 1 = 17) :
  s₂ = 81 := 
sorry

end square_area_ratio_l282_282723


namespace molecular_weight_of_BaBr2_l282_282167

theorem molecular_weight_of_BaBr2 
    (atomic_weight_Ba : ℝ)
    (atomic_weight_Br : ℝ)
    (moles : ℝ)
    (hBa : atomic_weight_Ba = 137.33)
    (hBr : atomic_weight_Br = 79.90) 
    (hmol : moles = 8) :
    (atomic_weight_Ba + 2 * atomic_weight_Br) * moles = 2377.04 :=
by 
  sorry

end molecular_weight_of_BaBr2_l282_282167


namespace sum_of_fractions_l282_282932

theorem sum_of_fractions :
  (3 / 15) + (6 / 15) + (9 / 15) + (12 / 15) + (15 / 15) + 
  (18 / 15) + (21 / 15) + (24 / 15) + (27 / 15) + (75 / 15) = 14 :=
by
  sorry

end sum_of_fractions_l282_282932


namespace sum_of_digits_base_8_888_is_13_l282_282172

noncomputable def sum_of_digits_base_8_rep_of_888 : ℕ :=
  let n : ℕ := 888
  let base_8_rep : ℕ := 1 * 8^3 + 5 * 8^2 + 7 * 8^1 + 0 * 8^0
  in 1 + 5 + 7 + 0

theorem sum_of_digits_base_8_888_is_13 : sum_of_digits_base_8_rep_of_888 = 13 :=
  sorry

end sum_of_digits_base_8_888_is_13_l282_282172


namespace soybeans_to_oil_kg_l282_282186

-- Define initial data
def kgSoybeansToTofu : ℕ := 3
def kgSoybeansToOil : ℕ := 6
def kgTofuCostPerKg : ℕ := 3
def kgOilCostPerKg : ℕ := 15
def batchSoybeansKg : ℕ := 460
def totalRevenue : ℕ := 1800

-- Define problem statement
theorem soybeans_to_oil_kg (x y : ℕ) (h : x + y = batchSoybeansKg) 
  (hRevenue : 3 * kgTofuCostPerKg * x + (kgOilCostPerKg * y) / (kgSoybeansToOil) = totalRevenue) : 
  y = 360 :=
sorry

end soybeans_to_oil_kg_l282_282186


namespace team_combination_count_l282_282496

theorem team_combination_count (n k : ℕ) (hn : n = 7) (hk : k = 4) :
  ∃ m, m = Nat.choose n k ∧ m = 35 :=
by
  sorry

end team_combination_count_l282_282496


namespace max_g_value_range_of_m_compare_values_l282_282240

-- Proof for maximum value of g(x)
theorem max_g_value :
  ∀ x > 0, ln (x) - x + 2 ≤ 1 := 
sorry

-- Proof for the range of m
theorem range_of_m (m : ℝ) :
  (∀ x ≥ 1, m * ln x ≥ (x - 1) / (x + 1)) → m ≥ 1 / 2 :=
sorry

-- Proof for comparing f(tan(alpha)) and -cos(2*alpha)
theorem compare_values (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  let x := tan α in 
  if hα1 : α < π / 4 then ln x < -cos (2 * α)
  else if hα2 : α = π / 4 then ln x = -cos (2 * α)
  else ln x > -cos (2 * α) :=
sorry

end max_g_value_range_of_m_compare_values_l282_282240


namespace connie_marbles_l282_282364

-- Define the initial number of marbles that Connie had
def initial_marbles : ℝ := 73.5

-- Define the number of marbles that Connie gave away
def marbles_given : ℝ := 70.3

-- Define the expected number of marbles remaining
def marbles_remaining : ℝ := 3.2

-- State the theorem: prove that initial_marbles - marbles_given = marbles_remaining
theorem connie_marbles :
  initial_marbles - marbles_given = marbles_remaining :=
sorry

end connie_marbles_l282_282364


namespace interval_of_x_l282_282953

theorem interval_of_x (x : ℝ) : (4 * x > 2) ∧ (4 * x < 5) ∧ (5 * x > 2) ∧ (5 * x < 5) ↔ (x > 1/2) ∧ (x < 1) := 
by 
  sorry

end interval_of_x_l282_282953


namespace polynomial_factor_pq_l282_282980

theorem polynomial_factor_pq (p q : ℝ) (h : ∀ x : ℝ, (x^2 + 2*x + 5) ∣ (x^4 + p*x^2 + q)) : p + q = 31 :=
sorry

end polynomial_factor_pq_l282_282980


namespace find_n_0_l282_282807

open Real

def sequence_a_n (n : ℕ) : ℕ :=
  ⌊log 2 n⌋₊

def sum_S_n (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, sequence_a_n (i + 1)

theorem find_n_0 : ∃ n_0 : ℕ, sum_S_n n_0 > 2018 ∧ ∀ n < n_0, sum_S_n n ≤ 2018 :=
  by
  use 316
  split
  { -- Prove sum_S_n 316 > 2018
    sorry }
  { -- Prove for all n < 316, sum_S_n n ≤ 2018
    sorry }

end find_n_0_l282_282807


namespace einstein_fundraising_l282_282158

def boxes_of_pizza : Nat := 15
def packs_of_potato_fries : Nat := 40
def cans_of_soda : Nat := 25
def price_per_box : ℝ := 12
def price_per_pack : ℝ := 0.3
def price_per_can : ℝ := 2
def goal_amount : ℝ := 500

theorem einstein_fundraising : goal_amount - (boxes_of_pizza * price_per_box + packs_of_potato_fries * price_per_pack + cans_of_soda * price_per_can) = 258 := by
  sorry

end einstein_fundraising_l282_282158


namespace exists_positive_integer_special_N_l282_282848

theorem exists_positive_integer_special_N : 
  ∃ (N : ℕ), 
    (∃ (m : ℕ), N = 1990 * (m + 995)) ∧ 
    (∀ (n : ℕ), (∃ (m : ℕ), 2 * N = (n + 1) * (2 * m + n)) ↔ (3980 = 2 * 1990)) := by
  sorry

end exists_positive_integer_special_N_l282_282848


namespace smallest_four_digit_equivalent_6_mod_7_l282_282632

theorem smallest_four_digit_equivalent_6_mod_7 :
  (∃ (n : ℕ), n >= 1000 ∧ n < 10000 ∧ n % 7 = 6 ∧ (∀ (m : ℕ), m >= 1000 ∧ m < 10000 ∧ m % 7 = 6 → m >= n)) ∧ ∃ (n : ℕ), n = 1000 :=
sorry

end smallest_four_digit_equivalent_6_mod_7_l282_282632


namespace arithmetic_sequence_sum_range_l282_282118

noncomputable def S_n (a d : ℝ) (n : ℕ) : ℝ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_range 
  (a d : ℝ)
  (h1 : 1 ≤ a + 3 * d) 
  (h2 : a + 3 * d ≤ 4)
  (h3 : 2 ≤ a + 4 * d)
  (h4 : a + 4 * d ≤ 3) 
  : 0 ≤ S_n a d 6 ∧ S_n a d 6 ≤ 30 := 
sorry

end arithmetic_sequence_sum_range_l282_282118


namespace ted_alex_age_ratio_l282_282313

theorem ted_alex_age_ratio (t a : ℕ) 
  (h1 : t - 3 = 4 * (a - 3))
  (h2 : t - 5 = 5 * (a - 5)) : 
  ∃ x : ℕ, (t + x) / (a + x) = 3 ∧ x = 1 :=
by
  sorry

end ted_alex_age_ratio_l282_282313


namespace range_m_l282_282127

theorem range_m (m : ℝ) : 
  (∀ x : ℝ, ((m * x - 1) * (x - 2) > 0) ↔ (1/m < x ∧ x < 2)) → m < 0 :=
by
  sorry

end range_m_l282_282127


namespace problem_M_plus_N_l282_282088

theorem problem_M_plus_N (M N : ℝ) (H1 : 4/7 = M/77) (H2 : 4/7 = 98/(N^2)) : M + N = 57.1 := 
sorry

end problem_M_plus_N_l282_282088


namespace can_form_triangle_l282_282330

theorem can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

example : can_form_triangle 8 6 3 := by
  sorry

end can_form_triangle_l282_282330


namespace find_a8_l282_282068

theorem find_a8 (a : ℕ → ℤ) (x : ℤ) :
  (1 + x)^10 = a 0 + a 1 * (1 - x) + a 2 * (1 - x)^2 + a 3 * (1 - x)^3 +
               a 4 * (1 - x)^4 + a 5 * (1 - x)^5 + a 6 * (1 - x)^6 +
               a 7 * (1 - x)^7 + a 8 * (1 - x)^8 + a 9 * (1 - x)^9 +
               a 10 * (1 - x)^10 → a 8 = 180 := by
  sorry

end find_a8_l282_282068


namespace find_unknown_blankets_rate_l282_282641

noncomputable def unknown_blankets_rate : ℝ :=
  let total_cost_3_blankets := 3 * 100
  let discount := 0.10 * total_cost_3_blankets
  let cost_3_blankets_after_discount := total_cost_3_blankets - discount
  let cost_1_blanket := 150
  let tax := 0.15 * cost_1_blanket
  let cost_1_blanket_after_tax := cost_1_blanket + tax
  let total_avg_price_per_blanket := 150
  let total_blankets := 6
  let total_cost := total_avg_price_per_blanket * total_blankets
  (total_cost - cost_3_blankets_after_discount - cost_1_blanket_after_tax) / 2

theorem find_unknown_blankets_rate : unknown_blankets_rate = 228.75 :=
  by
    sorry

end find_unknown_blankets_rate_l282_282641


namespace Amanda_car_round_trip_time_l282_282199

theorem Amanda_car_round_trip_time (bus_time : ℕ) (car_reduction : ℕ) (bus_one_way_trip : bus_time = 40) (car_time_reduction : car_reduction = 5) : 
  (2 * (bus_time - car_reduction)) = 70 := 
by
  sorry

end Amanda_car_round_trip_time_l282_282199


namespace sine_wave_solution_l282_282056

theorem sine_wave_solution (a b c : ℝ) (h_pos_a : a > 0) 
  (h_amp : a = 3) 
  (h_period : (2 * Real.pi) / b = Real.pi) 
  (h_peak : (Real.pi / (2 * b)) - (c / b) = Real.pi / 6) : 
  a = 3 ∧ b = 2 ∧ c = Real.pi / 6 :=
by
  -- Lean code to construct the proof will appear here
  sorry

end sine_wave_solution_l282_282056


namespace LindasOriginalSavings_l282_282120

theorem LindasOriginalSavings : 
  (∃ S : ℝ, (1 / 4) * S = 200) ∧ 
  (3 / 4) * S = 600 ∧ 
  (∀ F : ℝ, 0.80 * F = 600 → F = 750) → 
  S = 800 :=
by
  sorry

end LindasOriginalSavings_l282_282120


namespace circle_equation_l282_282064

theorem circle_equation 
  (x y : ℝ)
  (passes_origin : (x, y) = (0, 0))
  (intersects_line : ∃ (x y : ℝ), 2 * x - y + 1 = 0)
  (intersects_circle : ∃ (x y :ℝ), x^2 + y^2 - 2 * x - 15 = 0) : 
  x^2 + y^2 + 28 * x - 15 * y = 0 :=
sorry

end circle_equation_l282_282064


namespace min_rows_required_to_seat_students_l282_282466

-- Definitions based on the conditions
def seats_per_row : ℕ := 168
def total_students : ℕ := 2016
def max_students_per_school : ℕ := 40

def min_number_of_rows : ℕ :=
  -- Given that the minimum number of rows required to seat all students following the conditions is 15
  15

-- Lean statement expressing the proof problem
theorem min_rows_required_to_seat_students :
  ∃ rows : ℕ, rows = min_number_of_rows ∧
  (∀ school_sizes : List ℕ, (∀ size ∈ school_sizes, size ≤ max_students_per_school)
    → (List.sum school_sizes = total_students)
    → ∀ school_arrangement : List (List ℕ), 
        (∀ row_sizes ∈ school_arrangement, List.sum row_sizes ≤ seats_per_row) 
        → List.length school_arrangement ≤ rows) :=
sorry

end min_rows_required_to_seat_students_l282_282466


namespace two_people_same_birthday_l282_282730

noncomputable def population : ℕ := 6000000000

noncomputable def max_age_seconds : ℕ := 150 * 366 * 24 * 60 * 60

theorem two_people_same_birthday :
  ∃ (a b : ℕ) (ha : a < population) (hb : b < population) (hab : a ≠ b),
  (∃ (t : ℕ) (ht_a : t < max_age_seconds) (ht_b : t < max_age_seconds), true) :=
by
  sorry

end two_people_same_birthday_l282_282730


namespace subsets_bound_l282_282270

variable {n : ℕ} (S : Finset (Fin n)) (m : ℕ) (A : ℕ → Finset (Fin n))

theorem subsets_bound {n : ℕ} (hn : n ≥ 2) (hA : ∀ i, 1 ≤ i ∧ i ≤ m → (A i).card ≥ 2)
  (h_inter : ∀ i j k, 1 ≤ i ∧ i ≤ m → 1 ≤ j ∧ j ≤ m → 1 ≤ k ∧ k ≤ m →
    (A i) ∩ (A j) ≠ ∅ ∧ (A i) ∩ (A k) ≠ ∅ ∧ (A j) ∩ (A k) ≠ ∅ → (A i) ∩ (A j) ∩ (A k) ≠ ∅) :
  m ≤ 2 ^ (n - 1) - 1 := 
sorry

end subsets_bound_l282_282270


namespace probability_same_color_l282_282390

/-- Define the number of green plates. -/
def green_plates : ℕ := 7

/-- Define the number of yellow plates. -/
def yellow_plates : ℕ := 5

/-- Define the total number of plates. -/
def total_plates : ℕ := green_plates + yellow_plates

/-- Calculate the binomial coefficient for choosing k items from a set of n items. -/
def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Prove that the probability of selecting three plates of the same color is 9/44. -/
theorem probability_same_color :
  (binomial_coeff green_plates 3 + binomial_coeff yellow_plates 3) / binomial_coeff total_plates 3 = 9 / 44 :=
by
  sorry

end probability_same_color_l282_282390


namespace probability_same_color_is_one_third_l282_282164

-- Define a type for colors
inductive Color 
| red 
| white 
| blue 

open Color

-- Define the function to calculate the probability of the same color selection
def sameColorProbability : ℚ :=
  let total_outcomes := 3 * 3
  let favorable_outcomes := 3
  favorable_outcomes / total_outcomes

-- Theorem stating that the probability is 1/3
theorem probability_same_color_is_one_third : sameColorProbability = 1 / 3 :=
by
  -- Steps of proof will be provided here
  sorry

end probability_same_color_is_one_third_l282_282164


namespace math_problem_solution_l282_282425

noncomputable def math_problem (a b c d : ℝ) (h1 : a^2 + b^2 - c^2 - d^2 = 0) (h2 : a^2 - b^2 - c^2 + d^2 = (56 / 53) * (b * c + a * d)) : ℝ :=
  (a * b + c * d) / (b * c + a * d)

theorem math_problem_solution (a b c d : ℝ) (h1 : a^2 + b^2 - c^2 - d^2 = 0) (h2 : a^2 - b^2 - c^2 + d^2 = (56 / 53) * (b * c + a * d)) :
  math_problem a b c d h1 h2 = 45 / 53 := sorry

end math_problem_solution_l282_282425


namespace proof_solution_l282_282643

def proof_problem : Prop :=
  ∀ (s c p d : ℝ), 
  4 * s + 8 * c + p + 2 * d = 5.00 → 
  5 * s + 11 * c + p + 3 * d = 6.50 → 
  s + c + p + d = 1.50

theorem proof_solution : proof_problem :=
  sorry

end proof_solution_l282_282643


namespace remainder_of_product_mod_10_l282_282168

theorem remainder_of_product_mod_10 :
  (1265 * 4233 * 254 * 1729) % 10 = 0 := by
  sorry

end remainder_of_product_mod_10_l282_282168


namespace minimum_value_of_y_l282_282461

-- Define the function y
noncomputable def y (x : ℝ) := 2 + 4 * x + 1 / x

-- Prove that the minimum value is 6 for x > 0
theorem minimum_value_of_y : ∃ (x : ℝ), x > 0 ∧ (∀ (y : ℝ), (2 + 4 * x + 1 / x) ≤ y) ∧ (2 + 4 * x + 1 / x) = 6 := 
sorry

end minimum_value_of_y_l282_282461


namespace sector_area_given_angle_radius_sector_max_area_perimeter_l282_282547

open Real

theorem sector_area_given_angle_radius :
  ∀ (α : ℝ) (R : ℝ), α = 60 * (π / 180) ∧ R = 10 →
  (α / 360 * 2 * π * R) = 10 * π / 3 ∧ 
  (α * π * R^2 / 360) = 50 * π / 3 :=
by
  intros α R h
  rcases h with ⟨hα, hR⟩
  sorry

theorem sector_max_area_perimeter :
  ∀ (r α: ℝ), (2 * r + r * α) = 8 →
  α = 2 →
  r = 2 :=
by
  intros r α h ha
  sorry

end sector_area_given_angle_radius_sector_max_area_perimeter_l282_282547


namespace distinct_painted_cubes_l282_282497

-- Define the context of the problem
def num_faces : ℕ := 6

def total_paintings : ℕ := num_faces.factorial

def num_rotations : ℕ := 24

-- Statement of the theorem
theorem distinct_painted_cubes (h1 : total_paintings = 720) (h2 : num_rotations = 24) : 
  total_paintings / num_rotations = 30 := by
  sorry

end distinct_painted_cubes_l282_282497


namespace change_received_l282_282878

variable (a : ℝ)

theorem change_received (h : a ≤ 30) : 100 - 3 * a = 100 - 3 * a :=
by
  sorry

end change_received_l282_282878


namespace slope_of_l4_l282_282271

open Real

def line1 (x y : ℝ) : Prop := 4 * x - 3 * y = 6
def pointD : ℝ × ℝ := (0, -2)
def line2 (y : ℝ) : Prop := y = -1
def area_triangle_DEF := 4

theorem slope_of_l4 
  (l4_slope : ℝ)
  (H1 : ∃ x, line1 x (-1))
  (H2 : ∀ x y, 
         x ≠ 0 ∧
         y ≠ -2 ∧
         y ≠ -1 →
         line2 y →
         l4_slope = (y - (-2)) / (x - 0) →
         (1/2) * |(y + 1)| * (sqrt ((x-0) * (x-0) + (y-(-2)) * (y-(-2)))) = area_triangle_DEF ) :
  l4_slope = 1 / 8 :=
sorry

end slope_of_l4_l282_282271


namespace sufficient_and_necessary_condition_l282_282964

variable {a_n : ℕ → ℝ}

-- Defining the geometric sequence and the given conditions
def is_geometric_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a_n (n + 1) = a_n n * r

def is_increasing_sequence (a_n : ℕ → ℝ) : Prop :=
  ∀ n, a_n n < a_n (n + 1)

def condition (a_n : ℕ → ℝ) : Prop := a_n 0 < a_n 1 ∧ a_n 1 < a_n 2

-- The proof statement
theorem sufficient_and_necessary_condition (a_n : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a_n) :
  condition a_n ↔ is_increasing_sequence a_n :=
sorry

end sufficient_and_necessary_condition_l282_282964


namespace quadratic_equal_real_roots_l282_282982

theorem quadratic_equal_real_roots (m : ℝ) (h : ∃ x : ℝ, x^2 - 4 * x + m = 1 ∧ 
                              (∀ y : ℝ, y ≠ x → y^2 - 4 * y + m ≠ 1)) : m = 5 :=
by sorry

end quadratic_equal_real_roots_l282_282982


namespace solve_chair_table_fraction_l282_282912

def chair_table_fraction : Prop :=
  ∃ (C T : ℝ), T = 140 ∧ (T + 4 * C = 220) ∧ (C / T = 1 / 7)

theorem solve_chair_table_fraction : chair_table_fraction :=
  sorry

end solve_chair_table_fraction_l282_282912


namespace krista_egg_sales_l282_282579

-- Define the conditions
def hens : ℕ := 10
def eggs_per_hen_per_week : ℕ := 12
def price_per_dozen : ℕ := 3
def weeks : ℕ := 4

-- Define the total money made as the value we want to prove
def total_money_made : ℕ := 120

-- State the theorem
theorem krista_egg_sales : 
  (hens * eggs_per_hen_per_week * weeks / 12) * price_per_dozen = total_money_made :=
by
  sorry

end krista_egg_sales_l282_282579


namespace symmetric_line_eq_l282_282384

-- Defining a structure for a line using its standard equation form "ax + by + c = 0"
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Definition: A line is symmetric with respect to y-axis if it can be obtained
-- by replacing x with -x in its equation form.

def isSymmetricToYAxis (l₁ l₂ : Line) : Prop :=
  l₂.a = -l₁.a ∧ l₂.b = l₁.b ∧ l₂.c = l₁.c

-- The given condition: line1 is 4x - 3y + 5 = 0
def line1 : Line := { a := 4, b := -3, c := 5 }

-- The expected line l symmetric to y-axis should satisfy our properties
def expected_line_l : Line := { a := 4, b := 3, c := -5 }

-- The theorem we need to prove
theorem symmetric_line_eq : ∃ l : Line,
  isSymmetricToYAxis line1 l ∧ l = { a := 4, b := 3, c := -5 } :=
by
  sorry

end symmetric_line_eq_l282_282384


namespace students_total_l282_282360

theorem students_total (T : ℝ) (h₁ : 0.675 * T = 594) : T = 880 :=
sorry

end students_total_l282_282360


namespace perfect_square_condition_l282_282690

theorem perfect_square_condition (m : ℤ) : 
  (∃ k : ℤ, (x - 1) * (x + 3) * (x - 4) * (x - 8) + m = k^2) ↔ m = 196 :=
by sorry

end perfect_square_condition_l282_282690


namespace tree_height_at_year_3_l282_282195

theorem tree_height_at_year_3 :
  ∃ h₃ : ℕ, h₃ = 27 ∧
  (∃ h₇ h₆ h₅ h₄ : ℕ,
   h₇ = 648 ∧
   h₆ = h₇ / 2 ∧
   h₅ = h₆ / 2 ∧
   h₄ = h₅ / 2 ∧
   h₄ = 3 * h₃) :=
by
  sorry

end tree_height_at_year_3_l282_282195


namespace var_of_or_l282_282242

theorem var_of_or (p q : Prop) (h : ¬ (p ∧ q)) : (p ∨ q = true) ∨ (p ∨ q = false) :=
by
  sorry

end var_of_or_l282_282242


namespace arithmetic_geometric_sum_l282_282115

theorem arithmetic_geometric_sum (a : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ) (n : ℕ) (h1 : a 1 = 2)
  (h2 : a 3 = a 1 + 2 * d) (h3 : a 5 = a 1 + 4 * d) (h4 : (a 3) ^ 2 = a 1 * a 5)
  (h5 : d ≠ 0) : S n = (n^2 + 7 * n) / 4 := sorry

end arithmetic_geometric_sum_l282_282115


namespace runs_in_last_match_l282_282344

theorem runs_in_last_match (W : ℕ) (R x : ℝ) 
    (hW : W = 85) 
    (hR : R = 12.4 * W) 
    (new_average : (R + x) / (W + 5) = 12) : 
    x = 26 := 
by 
  sorry

end runs_in_last_match_l282_282344


namespace max_value_y_eq_neg10_l282_282062

open Real

theorem max_value_y_eq_neg10 (x : ℝ) (hx : x > 0) : 
  ∃ y, y = 2 - 9 * x - 4 / x ∧ (∀ z, (∃ (x' : ℝ), x' > 0 ∧ z = 2 - 9 * x' - 4 / x') → z ≤ y) ∧ y = -10 :=
by
  sorry

end max_value_y_eq_neg10_l282_282062


namespace multiples_of_3_or_4_probability_l282_282018

theorem multiples_of_3_or_4_probability :
  let total_cards := 36
  let multiples_of_3 := 12
  let multiples_of_4 := 9
  let multiples_of_both := 3
  let favorable_outcomes := multiples_of_3 + multiples_of_4 - multiples_of_both
  let probability := (favorable_outcomes : ℚ) / total_cards
  probability = 1 / 2 :=
by
  sorry

end multiples_of_3_or_4_probability_l282_282018


namespace john_makes_200_profit_l282_282265

noncomputable def john_profit (num_woodburnings : ℕ) (price_per_woodburning : ℕ) (cost_of_wood : ℕ) : ℕ :=
  (num_woodburnings * price_per_woodburning) - cost_of_wood

theorem john_makes_200_profit :
  john_profit 20 15 100 = 200 :=
by
  sorry

end john_makes_200_profit_l282_282265


namespace find_a_l282_282137

theorem find_a (a x_0 : ℝ) (h_tangent: (ax_0^3 + 1 = x_0) ∧ (3 * a * x_0^2 = 1)) : a = 4 / 27 :=
sorry

end find_a_l282_282137


namespace smallest_fraction_greater_than_4_over_5_l282_282357

theorem smallest_fraction_greater_than_4_over_5 :
  ∃ (b : ℕ), 10 ≤ b ∧ b < 100 ∧ 77 * 5 > 4 * b ∧ Int.gcd 77 b = 1 ∧
  ∀ (a : ℕ), 10 ≤ a ∧ a < 77 → ¬ ∃ (b' : ℕ), 10 ≤ b' ∧ b' < 100 ∧ a * 5 > 4 * b' ∧ Int.gcd a b' = 1 := by
  sorry

end smallest_fraction_greater_than_4_over_5_l282_282357


namespace correct_calculation_l282_282392

theorem correct_calculation (x : ℕ) (h : 954 - x = 468) : 954 + x = 1440 := by
  sorry

end correct_calculation_l282_282392


namespace jake_reaches_ground_later_by_2_seconds_l282_282204

noncomputable def start_floor : ℕ := 12
noncomputable def steps_per_floor : ℕ := 25
noncomputable def jake_steps_per_second : ℕ := 3
noncomputable def elevator_B_time : ℕ := 90

noncomputable def total_steps_jake := (start_floor - 1) * steps_per_floor
noncomputable def time_jake := (total_steps_jake + jake_steps_per_second - 1) / jake_steps_per_second
noncomputable def time_difference := time_jake - elevator_B_time

theorem jake_reaches_ground_later_by_2_seconds :
  time_difference = 2 := by
  sorry

end jake_reaches_ground_later_by_2_seconds_l282_282204


namespace simplify_and_evaluate_l282_282737

theorem simplify_and_evaluate :
  ∀ (a b : ℚ), a = 2 → b = -1/2 → (a - 2 * (a - b^2) + 3 * (-a + b^2) = -27/4) :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end simplify_and_evaluate_l282_282737


namespace complement_U_M_l282_282119

theorem complement_U_M :
  let U := {x : ℤ | ∃ k : ℤ, x = 2 * k}
  let M := {x : ℤ | ∃ k : ℤ, x = 4 * k}
  {x | x ∈ U ∧ x ∉ M} = {x : ℤ | ∃ k : ℤ, x = 4 * k - 2} :=
by
  sorry

end complement_U_M_l282_282119


namespace problem_statement_l282_282826

open Set

noncomputable def U : Set ℝ := univ
noncomputable def M : Set ℝ := { x : ℝ | abs x < 2 }
noncomputable def N : Set ℝ := { y : ℝ | ∃ x : ℝ, y = 2^x - 1 }

theorem problem_statement :
  compl M ∪ compl N = Iic (-1) ∪ Ici 2 :=
by {
  sorry
}

end problem_statement_l282_282826


namespace fraction_of_bag_spent_on_lunch_l282_282733

-- Definitions of conditions based on the problem
def initial_amount : ℕ := 158
def price_of_shoes : ℕ := 45
def price_of_bag : ℕ := price_of_shoes - 17
def amount_left : ℕ := 78
def money_before_lunch := amount_left + price_of_shoes + price_of_bag
def money_spent_on_lunch := initial_amount - money_before_lunch 

-- Statement of the problem in Lean
theorem fraction_of_bag_spent_on_lunch :
  (money_spent_on_lunch : ℚ) / price_of_bag = 1 / 4 :=
by
  -- Conditions decoded to match the solution provided
  have h1 : price_of_bag = 28 := by sorry
  have h2 : money_before_lunch = 151 := by sorry
  have h3 : money_spent_on_lunch = 7 := by sorry
  -- The main theorem statement
  exact sorry

end fraction_of_bag_spent_on_lunch_l282_282733


namespace number_of_boys_in_biology_class_l282_282152

variable (B G : ℕ) (PhysicsClass BiologyClass : ℕ)

theorem number_of_boys_in_biology_class
  (h1 : G = 3 * B)
  (h2 : PhysicsClass = 200)
  (h3 : BiologyClass = PhysicsClass / 2)
  (h4 : BiologyClass = B + G) :
  B = 25 := by
  sorry

end number_of_boys_in_biology_class_l282_282152


namespace prob_select_math_books_l282_282273

theorem prob_select_math_books :
  let total_books := 5
  let math_books := 3
  let total_ways_select_2 := Nat.choose total_books 2
  let ways_select_2_math := Nat.choose math_books 2
  let probability := (ways_select_2_math : ℚ) / total_ways_select_2
  probability = 3 / 10 :=
by
  sorry

end prob_select_math_books_l282_282273


namespace base6_to_base10_product_zero_l282_282131

theorem base6_to_base10_product_zero
  (c d e : ℕ)
  (h : (5 * 6^2 + 3 * 6^1 + 2 * 6^0) = (100 * c + 10 * d + e)) :
  (c * e) / 10 = 0 :=
by
  sorry

end base6_to_base10_product_zero_l282_282131


namespace polynomial_identity_l282_282039

variable (x y : ℝ)

theorem polynomial_identity :
    (x + y^2) * (x - y^2) * (x^2 + y^4) = x^4 - y^8 :=
sorry

end polynomial_identity_l282_282039


namespace part1_l282_282834

-- Definition of a double root equation
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, (x₁ ≠ x₂ ∧ x₂ = 2 * x₁ ∧ a * x₁ ^ 2 + b * x₁ + c = 0 ∧ a * x₂ ^ 2 + b * x₂ + c = 0)

-- Part 1: Proof that x^2 - 3x + 2 = 0 is a double root equation
theorem part1 : is_double_root_equation 1 (-3) 2 :=
by {
  use [1, 2],
  split,
  { intros h,
    linarith, },
  split,
  { refl, },
  split;
  { simp, linarith, }
}

end part1_l282_282834


namespace smallest_integer_equal_costs_l282_282754

-- Definitions based directly on conditions
def decimal_cost (n : ℕ) : ℕ :=
  (n.digits 10).sum * 2

def binary_cost (n : ℕ) : ℕ :=
  (n.digits 2).sum

-- The main statement to prove
theorem smallest_integer_equal_costs : ∃ n : ℕ, n < 2000 ∧ decimal_cost n = binary_cost n ∧ n = 255 :=
by 
  sorry

end smallest_integer_equal_costs_l282_282754


namespace victor_percentage_80_l282_282625

def percentage_of_marks (marks_obtained : ℕ) (maximum_marks : ℕ) : ℕ :=
  (marks_obtained * 100) / maximum_marks

theorem victor_percentage_80 :
  percentage_of_marks 240 300 = 80 := by
  sorry

end victor_percentage_80_l282_282625


namespace sum_of_ab_conditions_l282_282448

theorem sum_of_ab_conditions (a b : ℝ) (h : a^3 + b^3 = 1 - 3 * a * b) : a + b = 1 ∨ a + b = -2 := 
by
  sorry

end sum_of_ab_conditions_l282_282448


namespace enrico_earnings_l282_282946

theorem enrico_earnings : 
  let price_per_kg := 0.50
  let weight_rooster1 := 30
  let weight_rooster2 := 40
  let total_earnings := price_per_kg * weight_rooster1 + price_per_kg * weight_rooster2
  total_earnings = 35 := 
by
  sorry

end enrico_earnings_l282_282946


namespace distance_each_player_runs_l282_282287

-- Definitions based on conditions
def length : ℝ := 100
def width : ℝ := 50
def laps : ℝ := 6

def perimeter (l w : ℝ) : ℝ := 2 * (l + w)

def total_distance (l w laps : ℝ) : ℝ := laps * perimeter l w

-- Theorem statement
theorem distance_each_player_runs :
  total_distance length width laps = 1800 := 
by 
  sorry

end distance_each_player_runs_l282_282287


namespace subgroups_of_integers_l282_282450

theorem subgroups_of_integers (G : AddSubgroup ℤ) : ∃ (d : ℤ), G = AddSubgroup.zmultiples d := 
sorry

end subgroups_of_integers_l282_282450


namespace avg_b_c_weight_l282_282739

theorem avg_b_c_weight (a b c : ℝ) (H1 : (a + b + c) / 3 = 45) (H2 : (a + b) / 2 = 40) (H3 : b = 39) : (b + c) / 2 = 47 :=
by
  sorry

end avg_b_c_weight_l282_282739


namespace addition_problem_l282_282259

theorem addition_problem (m n p q : ℕ) (Hm : m = 2) (Hn : 2 + n + 7 + 5 = 20) (Hp : 1 + 6 + p + 8 = 24) (Hq : 3 + 2 + q = 12) (Hpositives : 0 < m ∧ 0 < n ∧ 0 < p ∧ 0 < q) :
  m + n + p + q = 24 :=
sorry

end addition_problem_l282_282259


namespace solution_is_D_l282_282327

-- Definitions of the equations
def eqA (x : ℝ) := 3 * x + 6 = 0
def eqB (x : ℝ) := 2 * x + 4 = 0
def eqC (x : ℝ) := (1 / 2) * x = -4
def eqD (x : ℝ) := 2 * x - 4 = 0

-- Theorem stating that only eqD has a solution x = 2
theorem solution_is_D : 
  ¬ eqA 2 ∧ ¬ eqB 2 ∧ ¬ eqC 2 ∧ eqD 2 := 
by
  sorry

end solution_is_D_l282_282327


namespace num_possible_values_a_l282_282283

theorem num_possible_values_a (a : ℕ) :
  (9 ∣ a) ∧ (a ∣ 18) ∧ (0 < a) → ∃ n : ℕ, n = 2 :=
by
  sorry

end num_possible_values_a_l282_282283


namespace amanda_car_round_trip_time_l282_282197

theorem amanda_car_round_trip_time :
  (bus_time = 40) ∧ (car_time = bus_time - 5) → (round_trip_time = car_time * 2) → round_trip_time = 70 :=
by
  sorry

end amanda_car_round_trip_time_l282_282197


namespace geometric_series_sum_l282_282891

theorem geometric_series_sum :
  let a := 3
  let r := 2
  let n := 8
  let S := (a * (1 - r^n)) / (1 - r)
  (3 + 6 + 12 + 24 + 48 + 96 + 192 + 384 = S) → S = 765 :=
by
  -- conditions
  let a := 3
  let r := 2
  let n := 8
  let S := (a * (1 - r^n)) / (1 - r)
  have h : 3 * (1 - 2^n) / (1 - 2) = 765 := sorry
  sorry

end geometric_series_sum_l282_282891


namespace solve_fraction_equation_l282_282143

theorem solve_fraction_equation : ∀ (x : ℝ), (x + 2) / (2 * x - 1) = 1 → x = 3 :=
by
  intros x h
  sorry

end solve_fraction_equation_l282_282143


namespace minor_axis_length_l282_282292

theorem minor_axis_length {x y : ℝ} (h : x^2 / 16 + y^2 / 9 = 1) : 6 = 6 :=
by
  sorry

end minor_axis_length_l282_282292


namespace sum_S_15_22_31_l282_282970

-- Define the sequence \{a_n\} with the sum of the first n terms S_n
def S : ℕ → ℤ
| 0 => 0
| n + 1 => S n + (-1: ℤ)^n * (4 * (n + 1) - 3)

-- The statement to prove: S_{15} + S_{22} - S_{31} = -76
theorem sum_S_15_22_31 : S 15 + S 22 - S 31 = -76 :=
sorry

end sum_S_15_22_31_l282_282970


namespace rational_numbers_cubic_sum_l282_282683

theorem rational_numbers_cubic_sum
  (a b c : ℚ)
  (h1 : a - b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 3) :
  a^3 + b^3 + c^3 = 1 :=
by
  sorry

end rational_numbers_cubic_sum_l282_282683


namespace find_y_l282_282844

-- Definitions of the angles
def angle_ABC : ℝ := 80
def angle_BAC : ℝ := 70
def angle_BCA : ℝ := 180 - angle_ABC - angle_BAC -- calculation of third angle in triangle ABC

-- Right angle in triangle CDE
def angle_ECD : ℝ := 90

-- Defining the proof problem
theorem find_y (y : ℝ) : 
  angle_BCA = 30 →
  angle_CDE = angle_BCA →
  angle_CDE + y + angle_ECD = 180 → 
  y = 60 := by
  intro h1 h2 h3
  sorry

end find_y_l282_282844


namespace count_positive_integers_l282_282534

theorem count_positive_integers (n : ℕ) (x : ℝ) (h1 : n ≤ 1500) :
  (∃ x : ℝ, n = ⌊x⌋ + ⌊3*x⌋ + ⌊5*x⌋) ↔ n = 668 :=
by
  sorry

end count_positive_integers_l282_282534


namespace find_x_l282_282217

theorem find_x (x : ℕ) (a : ℕ) (h₁: a = 450) (h₂: (15^x * 8^3) / 256 = a) : x = 2 :=
by
  sorry

end find_x_l282_282217


namespace find_income_l282_282015

variable (x : ℝ)

def income : ℝ := 5 * x
def expenditure : ℝ := 4 * x
def savings : ℝ := income x - expenditure x

theorem find_income (h : savings x = 4000) : income x = 20000 :=
by
  rw [savings, income, expenditure] at h
  sorry

end find_income_l282_282015


namespace percent_not_filler_l282_282913

theorem percent_not_filler (sandwich_weight filler_weight : ℕ) (h_sandwich : sandwich_weight = 180) (h_filler : filler_weight = 45) : 
  (sandwich_weight - filler_weight) * 100 / sandwich_weight = 75 :=
by
  -- proof here
  sorry

end percent_not_filler_l282_282913


namespace train_cross_time_l282_282348

noncomputable def train_speed_kmph : ℝ := 72
noncomputable def train_speed_mps : ℝ := 20
noncomputable def platform_length : ℝ := 320
noncomputable def time_cross_platform : ℝ := 34
noncomputable def train_length : ℝ := 360

theorem train_cross_time (v_kmph : ℝ) (v_mps : ℝ) (p_len : ℝ) (t_cross : ℝ) (t_len : ℝ) :
  v_kmph = 72 ∧ v_mps = 20 ∧ p_len = 320 ∧ t_cross = 34 ∧ t_len = 360 →
  (t_len / v_mps) = 18 :=
by
  intros
  sorry

end train_cross_time_l282_282348


namespace problem_proof_l282_282998

-- Problem statement
variable (f : ℕ → ℕ)

-- Condition: if f(k) ≥ k^2 then f(k+1) ≥ (k+1)^2
variable (h : ∀ k, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2)

-- Additional condition: f(4) ≥ 25
variable (h₀ : f 4 ≥ 25)

-- To prove: ∀ k ≥ 4, f(k) ≥ k^2
theorem problem_proof : ∀ k ≥ 4, f k ≥ k^2 :=
by
  sorry

end problem_proof_l282_282998


namespace support_percentage_l282_282255

theorem support_percentage (men women : ℕ) (support_men_percentage support_women_percentage : ℝ) 
(men_support women_support total_support : ℕ)
(hmen : men = 150) 
(hwomen : women = 850) 
(hsupport_men_percentage : support_men_percentage = 0.55) 
(hsupport_women_percentage : support_women_percentage = 0.70) 
(hmen_support : men_support = 83) 
(hwomen_support : women_support = 595)
(htotal_support : total_support = men_support + women_support) :
  ((total_support : ℝ) / (men + women) * 100) = 68 :=
by
  -- Insert the proof here to verify each step of the calculation and rounding
  sorry

end support_percentage_l282_282255


namespace find_f_of_minus_five_l282_282135

theorem find_f_of_minus_five (a b : ℝ) (f : ℝ → ℝ) (h1 : f 5 = 7) (h2 : ∀ x, f x = a * x + b * Real.sin x + 1) : f (-5) = -5 :=
by
  sorry

end find_f_of_minus_five_l282_282135


namespace robyn_packs_l282_282734

-- Define the problem conditions
def total_packs : ℕ := 76
def lucy_packs : ℕ := 29

-- Define the goal to be proven
theorem robyn_packs : total_packs - lucy_packs = 47 := 
by
  sorry

end robyn_packs_l282_282734


namespace angles_in_order_l282_282241

-- α1, α2, α3 are real numbers representing the angles of inclination of lines
variable (α1 α2 α3 : ℝ)

-- Conditions given in the problem
axiom tan_α1 : Real.tan α1 = 1
axiom tan_α2 : Real.tan α2 = -1
axiom tan_α3 : Real.tan α3 = -2

-- Theorem to prove
theorem angles_in_order : α1 < α3 ∧ α3 < α2 := 
by
  sorry

end angles_in_order_l282_282241


namespace vector_sum_length_l282_282821

open Real

noncomputable def vector_length (v : ℝ × ℝ) : ℝ :=
sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2

noncomputable def vector_angle_cosine (v w : ℝ × ℝ) : ℝ :=
dot_product v w / (vector_length v * vector_length w)

theorem vector_sum_length (a b : ℝ × ℝ)
  (ha : vector_length a = 2)
  (hb : vector_length b = 2)
  (hab_angle : vector_angle_cosine a b = cos (π / 3)):
  vector_length (a.1 + b.1, a.2 + b.2) = 2 * sqrt 3 :=
by sorry

end vector_sum_length_l282_282821


namespace inequality_solution_condition_necessary_but_not_sufficient_l282_282812

theorem inequality_solution_condition (a : ℝ) :
  (∃ x : ℝ, x^2 + 2 * a * x - a ≤ 0) ↔ (a ≥ 0 ∨ a ≤ -1) := sorry

theorem necessary_but_not_sufficient (a : ℝ) :
  (a > 0 ∨ a < -1) → (∃ x : ℝ, x^2 + 2 * a * x - a ≤ 0) ∧ ¬(∃ x : ℝ, x^2 + 2 * a * x - a ≤ 0 → (a > 0 ∨ a < -1)) := sorry

end inequality_solution_condition_necessary_but_not_sufficient_l282_282812


namespace divisor_greater_than_8_l282_282405

-- Define the condition that remainder is 8
def remainder_is_8 (n m : ℕ) : Prop :=
  n % m = 8

-- Theorem: If n divided by m has remainder 8, then m must be greater than 8
theorem divisor_greater_than_8 (m : ℕ) (hm : m ≤ 8) : ¬ exists n, remainder_is_8 n m :=
by
  sorry

end divisor_greater_than_8_l282_282405


namespace solve_r_l282_282662

def E (a : ℝ) (b : ℝ) (c : ℕ) : ℝ := a * b^c

theorem solve_r : ∃ (r : ℝ), E r r 5 = 1024 ∧ r = 2^(5/3) :=
by
  sorry

end solve_r_l282_282662


namespace part1_part2_l282_282941

-- Definition for f(x)
def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

-- The first proof problem: Solve the inequality f(x) > 0
theorem part1 {x : ℝ} : f x > 0 ↔ x > 1 ∨ x < -5 :=
sorry

-- The second proof problem: Finding the range of m
theorem part2 {m : ℝ} : (∀ x, f x + 3 * |x - 4| ≥ m) → m ≤ 9 :=
sorry

end part1_part2_l282_282941


namespace value_of_t5_l282_282764

noncomputable def t_5_value (t1 t2 : ℚ) (r : ℚ) (a : ℚ) : ℚ := a * r^4

theorem value_of_t5 
  (a r : ℚ)
  (h1 : a > 0)  -- condition: each term is positive
  (h2 : a + a * r = 15 / 2)  -- condition: sum of first two terms is 15/2
  (h3 : a^2 + (a * r)^2 = 153 / 4)  -- condition: sum of squares of first two terms is 153/4
  (h4 : r > 0)  -- ensuring positivity of r
  (h5 : r < 1)  -- ensuring t1 > t2
  : t_5_value a (a * r) r a = 3 / 128 :=
sorry

end value_of_t5_l282_282764


namespace find_X_l282_282567

-- Define the variables for income, tax, and the variable X
def income := 58000
def tax := 8000

-- Define the tax formula as per the problem
def tax_formula (X : ℝ) : ℝ :=
  0.11 * X + 0.20 * (income - X)

-- The theorem we want to prove
theorem find_X :
  ∃ X : ℝ, tax_formula X = tax ∧ X = 40000 :=
sorry

end find_X_l282_282567


namespace maximum_marks_l282_282858

theorem maximum_marks (M : ℝ) (mark_obtained failed_by : ℝ) (pass_percentage : ℝ) 
  (h1 : pass_percentage = 0.6) (h2 : mark_obtained = 250) (h3 : failed_by = 50) :
  (pass_percentage * M = mark_obtained + failed_by) → M = 500 :=
by 
  sorry

end maximum_marks_l282_282858


namespace value_x_when_y2_l282_282395

theorem value_x_when_y2 (x : ℝ) (h1 : ∃ (x : ℝ), y = 1 / (5 * x + 2)) (h2 : y = 2) : x = -3 / 10 := by
  sorry

end value_x_when_y2_l282_282395


namespace average_words_per_minute_l282_282918

theorem average_words_per_minute 
  (total_words : ℕ) 
  (total_hours : ℕ) 
  (h_words : total_words = 30000) 
  (h_hours : total_hours = 100) : 
  (total_words / total_hours / 60 = 5) := by
  sorry

end average_words_per_minute_l282_282918


namespace compensation_problem_l282_282260

namespace CompensationProof

variables (a b c : ℝ)

def geometric_seq_with_ratio_1_by_2 (a b c : ℝ) : Prop :=
  c = (1/2) * b ∧ b = (1/2) * a

def total_compensation_eq (a b c : ℝ) : Prop :=
  4 * c + 2 * b + a = 50

theorem compensation_problem :
  total_compensation_eq a b c ∧ geometric_seq_with_ratio_1_by_2 a b c → c = 50 / 7 :=
sorry

end CompensationProof

end compensation_problem_l282_282260


namespace fraction_calculation_l282_282793

theorem fraction_calculation : 
  (1/2 - 1/3) / (3/7 * 2/8) = 14/9 :=
by
  sorry

end fraction_calculation_l282_282793


namespace triangle_side_m_l282_282985

theorem triangle_side_m (a b m : ℝ) (ha : a = 2) (hb : b = 3) (h1 : a + b > m) (h2 : a + m > b) (h3 : b + m > a) :
  (1 < m ∧ m < 5) → m = 3 :=
by
  sorry

end triangle_side_m_l282_282985


namespace vector_AD_length_l282_282234

open Real EuclideanSpace

noncomputable def problem_statement
  (m n : ℝ) (angle_mn : ℝ) (norm_m : ℝ) (norm_n : ℝ) (AB AC : ℝ) (AD : ℝ) : Prop :=
  angle_mn = π / 6 ∧ 
  norm_m = sqrt 3 ∧ 
  norm_n = 2 ∧ 
  AB = 2 * m + 2 * n ∧ 
  AC = 2 * m - 6 * n ∧ 
  AD = 2 * m - 2 * n ∧
  sqrt ((AD) * (AD)) = 2

theorem vector_AD_length 
  (m n : ℝ) (angle_mn : ℝ) (norm_m : ℝ) (norm_n : ℝ) (AB AC AD : ℝ) :
  problem_statement m n angle_mn norm_m norm_n AB AC AD :=
by
  unfold problem_statement
  sorry

end vector_AD_length_l282_282234


namespace initial_pencils_count_l282_282619

variables {pencils_taken : ℕ} {pencils_left : ℕ} {initial_pencils : ℕ}

theorem initial_pencils_count 
  (h1 : pencils_taken = 4)
  (h2 : pencils_left = 5) :
  initial_pencils = 9 :=
by 
  sorry

end initial_pencils_count_l282_282619


namespace monomials_like_terms_l282_282839

theorem monomials_like_terms {m n : ℕ} (hm : m = 3) (hn : n = 1) : m - n = 2 :=
by
  rw [hm, hn]
  rfl

end monomials_like_terms_l282_282839


namespace remainder_when_divided_by_2_l282_282489

-- Define the main parameters
def n : ℕ := sorry  -- n is a positive integer
def k : ℤ := sorry  -- Provided for modular arithmetic context

-- Conditions
axiom h1 : n > 0  -- n is a positive integer
axiom h2 : (n + 1) % 6 = 4  -- When n + 1 is divided by 6, the remainder is 4

-- The theorem statement
theorem remainder_when_divided_by_2 : n % 2 = 1 :=
by
  sorry

end remainder_when_divided_by_2_l282_282489


namespace arrangement_problem_l282_282104

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem arrangement_problem 
  (p1 p2 p3 p4 p5 : Type)  -- Representing the five people
  (youngest : p1)         -- Specifying the youngest
  (oldest : p5)           -- Specifying the oldest
  (unique_people : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ p3 ≠ p4 ∧ p3 ≠ p5 ∧ p4 ≠ p5) -- Ensuring five unique people
  : (factorial 5) - (factorial 4 * 2) = 72 :=
by sorry

end arrangement_problem_l282_282104


namespace positive_integer_solution_l282_282882

/-- Given that x, y, and t are all equal to 1, and x + y + z + t = 10, we need to prove that z = 7. -/
theorem positive_integer_solution {x y z t : ℕ} (hx : x = 1) (hy : y = 1) (ht : t = 1) (h : x + y + z + t = 10) : z = 7 :=
by {
  -- We would provide the proof here, but for now, we use sorry
  sorry
}

end positive_integer_solution_l282_282882


namespace scaled_multiplication_l282_282550

theorem scaled_multiplication (h : 268 * 74 = 19832) : 2.68 * 0.74 = 1.9832 :=
by
  -- proof steps would go here
  sorry

end scaled_multiplication_l282_282550


namespace quadrant_of_angle_l282_282692

theorem quadrant_of_angle (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  ∃ n : ℤ, n = 1 ∧ α = (n * π + π / 2) :=
sorry

end quadrant_of_angle_l282_282692


namespace ways_to_sum_420_l282_282707

theorem ways_to_sum_420 : 
  (∃ n k : ℕ, n ≥ 2 ∧ 2 * k + n - 1 > 0 ∧ n * (2 * k + n - 1) = 840) → (∃ c, c = 11) :=
by
  sorry

end ways_to_sum_420_l282_282707


namespace pump_fill_time_without_leak_l282_282049

theorem pump_fill_time_without_leak
    (P : ℝ)
    (h1 : 2 + 1/7 = (15:ℝ)/7)
    (h2 : 1 / P - 1 / 30 = 7 / 15) :
  P = 2 := by
  sorry

end pump_fill_time_without_leak_l282_282049


namespace coffee_cost_per_week_l282_282712

theorem coffee_cost_per_week 
  (people_in_house : ℕ) 
  (drinks_per_person_per_day : ℕ) 
  (ounces_per_cup : ℝ) 
  (cost_per_ounce : ℝ) 
  (num_days_in_week : ℕ) 
  (h1 : people_in_house = 4) 
  (h2 : drinks_per_person_per_day = 2)
  (h3 : ounces_per_cup = 0.5)
  (h4 : cost_per_ounce = 1.25)
  (h5 : num_days_in_week = 7) :
  people_in_house * drinks_per_person_per_day * ounces_per_cup * cost_per_ounce * num_days_in_week = 35 := 
by
  sorry

end coffee_cost_per_week_l282_282712


namespace ab_cd_value_l282_282090

theorem ab_cd_value (a b c d : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a + b + d = -3) 
  (h3 : a + c + d = 10) 
  (h4 : b + c + d = -1) : 
  ab + cd = -346 / 9 :=
by 
  sorry

end ab_cd_value_l282_282090


namespace polynomial_pairs_l282_282061

noncomputable def degree (p : Polynomial ℝ) : ℕ := sorry -- Placeholder for degree function

theorem polynomial_pairs (f g : Polynomial ℝ)
  (H1 : ∀ x, x^2 * g.eval x = f.eval (g.eval x)) :
  (degree f = 3 ∧ degree g = 1) ∨ (degree f = 2 ∧ degree g = 2) :=
sorry

end polynomial_pairs_l282_282061


namespace number_of_geese_more_than_ducks_l282_282309

theorem number_of_geese_more_than_ducks (geese ducks : ℝ) (h1 : geese = 58.0) (h2 : ducks = 37.0) :
  geese - ducks = 21.0 :=
by
  sorry

end number_of_geese_more_than_ducks_l282_282309


namespace arrangements_with_gap_l282_282747

theorem arrangements_with_gap :
  ∃ (arrangements : ℕ), arrangements = 36 :=
by
  sorry

end arrangements_with_gap_l282_282747


namespace original_price_dish_l282_282901

-- Conditions
variables (P : ℝ) -- Original price of the dish
-- Discount and tips
def john_discounted_and_tip := 0.9 * P + 0.15 * P
def jane_discounted_and_tip := 0.9 * P + 0.135 * P

-- Condition of payment difference
def payment_difference := john_discounted_and_tip P = jane_discounted_and_tip P + 0.36

-- The theorem to prove
theorem original_price_dish : payment_difference P → P = 24 :=
by
  intro h
  sorry

end original_price_dish_l282_282901


namespace triangle_area_less_than_sqrt3_div_3_l282_282729

-- Definitions for a triangle and its properties
structure Triangle :=
  (a b c : ℝ)
  (ha hb hc : ℝ)
  (area : ℝ)

def valid_triangle (Δ : Triangle) : Prop :=
  0 < Δ.a ∧ 0 < Δ.b ∧ 0 < Δ.c ∧ Δ.ha < 1 ∧ Δ.hb < 1 ∧ Δ.hc < 1

theorem triangle_area_less_than_sqrt3_div_3 (Δ : Triangle) (h : valid_triangle Δ) : Δ.area < (Real.sqrt 3) / 3 :=
sorry

end triangle_area_less_than_sqrt3_div_3_l282_282729


namespace line_length_after_erasing_l282_282278

-- Definition of the initial length in meters and the erased length in centimeters
def initial_length_meters : ℝ := 1.5
def erased_length_centimeters : ℝ := 15.25

-- Conversion factor from meters to centimeters
def meters_to_centimeters (m : ℝ) : ℝ := m * 100

-- Definition of the initial length in centimeters
def initial_length_centimeters : ℝ := meters_to_centimeters initial_length_meters

-- Statement of the theorem
theorem line_length_after_erasing :
  initial_length_centimeters - erased_length_centimeters = 134.75 :=
by
  -- The proof would go here
  sorry

end line_length_after_erasing_l282_282278


namespace parallel_lines_perpendicular_lines_l282_282908

theorem parallel_lines (t s k : ℝ) :
  (∀ t, ∃ s, (1 - 2 * t = s) ∧ (2 + k * t = 1 - 2 * s)) →
  k = 4 :=
by
  sorry

theorem perpendicular_lines (t s k : ℝ) :
  (∀ t, ∃ s, (1 - 2 * t = s) ∧ (2 + k * t = 1 - 2 * s)) →
  k = -1 :=
by
  sorry

end parallel_lines_perpendicular_lines_l282_282908


namespace lena_can_form_rectangles_vasya_can_form_rectangles_lena_and_vasya_can_be_right_l282_282113

def total_area_of_triangles_and_quadrilateral (A B Q : ℝ) : ℝ :=
  A + B + Q

def lena_triangles_and_quadrilateral_area (A B Q : ℝ) : Prop :=
  (24 : ℝ) = total_area_of_triangles_and_quadrilateral A B Q

def total_area_of_triangles_and_pentagon (C D P : ℝ) : ℝ :=
  C + D + P

def vasya_triangles_and_pentagon_area (C D P : ℝ) : Prop :=
  (24 : ℝ) = total_area_of_triangles_and_pentagon C D P

theorem lena_can_form_rectangles (A B Q : ℝ) (h : lena_triangles_and_quadrilateral_area A B Q) :
  lena_triangles_and_quadrilateral_area A B Q :=
by 
-- We assume the definition holds as given
sorry

theorem vasya_can_form_rectangles (C D P : ℝ) (h : vasya_triangles_and_pentagon_area C D P) :
  vasya_triangles_and_pentagon_area C D P :=
by 
-- We assume the definition holds as given
sorry

theorem lena_and_vasya_can_be_right (A B Q C D P : ℝ)
  (hlena : lena_triangles_and_quadrilateral_area A B Q)
  (hvasya : vasya_triangles_and_pentagon_area C D P) :
  lena_triangles_and_quadrilateral_area A B Q ∧ vasya_triangles_and_pentagon_area C D P :=
by 
-- Combining both assumptions
exact ⟨hlena, hvasya⟩

end lena_can_form_rectangles_vasya_can_form_rectangles_lena_and_vasya_can_be_right_l282_282113


namespace largest_divisor_of_m_square_minus_n_square_l282_282996

theorem largest_divisor_of_m_square_minus_n_square (m n : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 1) (h : n < m) :
  ∃ k : ℤ, k = 8 ∧ ∀ a b : ℤ, a % 2 = 1 → b % 2 = 1 → a > b → 8 ∣ (a^2 - b^2) := 
by
  sorry

end largest_divisor_of_m_square_minus_n_square_l282_282996


namespace einstein_fundraising_l282_282157

def boxes_of_pizza : Nat := 15
def packs_of_potato_fries : Nat := 40
def cans_of_soda : Nat := 25
def price_per_box : ℝ := 12
def price_per_pack : ℝ := 0.3
def price_per_can : ℝ := 2
def goal_amount : ℝ := 500

theorem einstein_fundraising : goal_amount - (boxes_of_pizza * price_per_box + packs_of_potato_fries * price_per_pack + cans_of_soda * price_per_can) = 258 := by
  sorry

end einstein_fundraising_l282_282157


namespace add_three_to_both_sides_l282_282541

variable {a b : ℝ}

theorem add_three_to_both_sides (h : a < b) : 3 + a < 3 + b :=
by
  sorry

end add_three_to_both_sides_l282_282541


namespace unit_price_correct_minimum_cost_l282_282772

-- Given conditions for the unit prices
-- ∀ x (the unit price of type A), if 110/(x) = 120/(x+1) then x = 11
theorem unit_price_correct :
  ∀ (x : ℝ), (110 / x = 120 / (x + 1)) → (x = 11) :=
by
  assume x,
  intro h,
  have h_eq : 110 * (x + 1) = 120 * x := (div_eq_iff h).mp rfl,
  have h_simpl: 110 * x + 110 = 120 * x := by linarith,
  have h_final: x = 11 := by linarith,
  exact h_final

-- Proving minimum cost
-- ∀ a : ℕ (the number of type A notebooks), b : ℕ (the number of type B notebooks),
-- if a + b = 100, b ≤ 3 * a, then the minimum cost w = 11a + 12(100 - a) is 1100
theorem minimum_cost :
  ∀ (a b : ℕ), (a + b = 100) → (b ≤ 3 * a) → (-a + 1200 = 1100) :=
by
  assume a b,
  intro h1,
  intro h2,
  have h1_rewrite: b = 100 - a := by linarith,
  have h_cost: 11 * a + 12 * (100 - a) = -a + 1200 := by linarith,
  exact h_cost

end unit_price_correct_minimum_cost_l282_282772


namespace isosceles_triangle_of_cosine_condition_l282_282401

theorem isosceles_triangle_of_cosine_condition
  (A B C : ℝ)
  (h : 2 * Real.cos A * Real.cos B = 1 - Real.cos C) :
  A = B ∨ A = π - B :=
  sorry

end isosceles_triangle_of_cosine_condition_l282_282401


namespace remainder_of_x7_plus_2_div_x_plus_1_l282_282939

def f (x : ℤ) := x^7 + 2

theorem remainder_of_x7_plus_2_div_x_plus_1 : 
  (f (-1) = 1) := sorry

end remainder_of_x7_plus_2_div_x_plus_1_l282_282939


namespace enrico_earnings_l282_282945

theorem enrico_earnings : 
  let price_per_kg := 0.50
  let weight_rooster1 := 30
  let weight_rooster2 := 40
  let total_earnings := price_per_kg * weight_rooster1 + price_per_kg * weight_rooster2
  total_earnings = 35 := 
by
  sorry

end enrico_earnings_l282_282945


namespace find_function_l282_282580

theorem find_function (f : ℕ → ℕ) (k : ℕ) :
  (∀ n : ℕ, f n < f (n + 1)) →
  (∀ n : ℕ, f (f n) = n + 2 * k) →
  ∀ n : ℕ, f n = n + k := 
by
  intro h1 h2
  sorry

end find_function_l282_282580


namespace exists_root_between_l282_282079

-- Given definitions and conditions
variables (a b c : ℝ)
variables (ha : a ≠ 0)
variables (x1 x2 : ℝ)
variable (h1 : a * x1^2 + b * x1 + c = 0)    -- root of the first equation
variable (h2 : -a * x2^2 + b * x2 + c = 0)   -- root of the second equation

-- Proof statement
theorem exists_root_between (a b c : ℝ) (ha : a ≠ 0) (x1 x2 : ℝ)
    (h1 : a * x1^2 + b * x1 + c = 0) (h2 : -a * x2^2 + b * x2 + c = 0) :
    ∃ x3 : ℝ, 
      (x1 ≤ x3 ∧ x3 ≤ x2 ∨ x1 ≥ x3 ∧ x3 ≥ x2) ∧ 
      (1 / 2 * a * x3^2 + b * x3 + c = 0) :=
sorry

end exists_root_between_l282_282079


namespace find_x_squared_l282_282528

theorem find_x_squared :
  ∃ x : ℕ, (x^2 >= 2525 * 10^8) ∧ (x^2 < 2526 * 10^8) ∧ (x % 100 = 17 ∨ x % 100 = 33 ∨ x % 100 = 67 ∨ x % 100 = 83) ∧
    (x = 502517 ∨ x = 502533 ∨ x = 502567 ∨ x = 502583) :=
sorry

end find_x_squared_l282_282528


namespace find_smallest_angle_l282_282302

theorem find_smallest_angle 
  (x y : ℝ)
  (hx : x + y = 45)
  (hy : y = x - 5)
  (hz : x > 0 ∧ y > 0 ∧ x + y < 180) :
  min x y = 20 := 
sorry

end find_smallest_angle_l282_282302


namespace gcd_problem_l282_282376

theorem gcd_problem : Nat.gcd 12740 220 - 10 = 10 :=
by
  sorry

end gcd_problem_l282_282376


namespace sum_of_base8_digits_888_l282_282173

def sumDigits (n : ℕ) (base : ℕ) : ℕ :=
  if n < base then n else
  sumDigits (n / base) base + (n % base)

theorem sum_of_base8_digits_888 : sumDigits 888 8 = 13 := by
  sorry

end sum_of_base8_digits_888_l282_282173


namespace polynomial_factors_sum_l282_282433

open Real

theorem polynomial_factors_sum
  (a b c : ℝ)
  (h1 : ∀ x, (x^2 + x + 2) * (a * x + b - a) + (c - a - b) * x + 5 + 2 * a - 2 * b = 0)
  (h2 : a * (1/2)^3 + b * (1/2)^2 + c * (1/2) - 25/16 = 0) :
  a + b + c = 45 / 11 :=
by
  sorry

end polynomial_factors_sum_l282_282433


namespace jerry_age_l282_282855

theorem jerry_age (M J : ℕ) (h1 : M = 4 * J + 10) (h2 : M = 30) : J = 5 := by
  sorry

end jerry_age_l282_282855


namespace petya_correct_square_l282_282893

theorem petya_correct_square :
  ∃ x a b : ℕ, (1 ≤ x ∧ x ≤ 9) ∧
              (x^2 = 10 * a + b) ∧ 
              (2 * x = 10 * b + a) ∧
              (x^2 = 81) :=
by
  sorry

end petya_correct_square_l282_282893


namespace max_area_of_rectangular_garden_l282_282446

-- Definitions corresponding to the conditions in the problem
def length1 (x : ℕ) := x
def length2 (x : ℕ) := 75 - x

-- Definition of the area
def area (x : ℕ) := x * (75 - x)

-- Statement to prove: there exists natural numbers x and y such that x + y = 75 and x * y = 1406
theorem max_area_of_rectangular_garden :
  ∃ (x : ℕ), (x + (75 - x) = 75) ∧ (x * (75 - x) = 1406) := 
by
  -- Due to the nature of this exercise, the actual proof is omitted.
  sorry

end max_area_of_rectangular_garden_l282_282446


namespace find_puppy_weight_l282_282778

noncomputable def weight_problem (a b c : ℕ) : Prop :=
  a + b + c = 36 ∧ a + c = 3 * b ∧ a + b = c + 6

theorem find_puppy_weight (a b c : ℕ) (h : weight_problem a b c) : a = 12 :=
sorry

end find_puppy_weight_l282_282778


namespace find_angle_A_find_area_l282_282819

noncomputable theory

-- Define the given conditions of the triangle
variables (A B C : ℝ) (a b c : ℝ)

-- Define the problem conditions
axiom angle_conditions : A + B + C = π
axiom side_conditions : a = 2 * √3 ∧ b + c = 4
axiom equation_condition : a * cos C + c * cos A = -2 * b * cos A

-- Prove the required results
theorem find_angle_A : A = 2 * π / 3 := sorry

theorem find_area : 1 / 2 * b * c * sin A = √3 := sorry

end find_angle_A_find_area_l282_282819


namespace sequence_arithmetic_difference_neg1_l282_282381

variable (a : ℕ → ℝ)

theorem sequence_arithmetic_difference_neg1 (h : ∀ n, a (n + 1) + 1 = a n) : ∀ n, a (n + 1) - a n = -1 :=
by
  intro n
  specialize h n
  linarith

-- Assuming natural numbers starting from 1 (ℕ^*), which is not directly available in Lean.
-- So we use assumptions accordingly.

end sequence_arithmetic_difference_neg1_l282_282381


namespace arithmetic_sequence_terms_l282_282235

theorem arithmetic_sequence_terms (a : ℕ → ℝ) (n : ℕ) (S : ℝ) 
  (h1 : a 0 + a 1 + a 2 = 4) 
  (h2 : a (n-3) + a (n-2) + a (n-1) = 7) 
  (h3 : (n * (a 0 + a (n-1)) / 2) = 22) : 
  n = 12 :=
sorry

end arithmetic_sequence_terms_l282_282235


namespace expand_polynomials_l282_282667

def p (z : ℝ) : ℝ := 3 * z ^ 2 + 4 * z - 7
def q (z : ℝ) : ℝ := 4 * z ^ 3 - 3 * z + 2

theorem expand_polynomials :
  (p z) * (q z) = 12 * z ^ 5 + 16 * z ^ 4 - 37 * z ^ 3 - 6 * z ^ 2 + 29 * z - 14 := by
  sorry

end expand_polynomials_l282_282667


namespace tourists_originally_in_group_l282_282505

theorem tourists_originally_in_group (x : ℕ) (h₁ : 220 / x - 220 / (x + 1) = 2) : x = 10 := 
by
  sorry

end tourists_originally_in_group_l282_282505


namespace sara_change_l282_282335

-- Define the costs of individual items
def cost_book_1 : ℝ := 5.5
def cost_book_2 : ℝ := 6.5
def cost_notebook : ℝ := 3
def cost_bookmarks : ℝ := 2

-- Define the discounts and taxes
def discount_books : ℝ := 0.10
def sales_tax : ℝ := 0.05

-- Define the payment amount
def amount_given : ℝ := 20

-- Calculate the total cost, discount, and final amount
def discounted_book_cost := (cost_book_1 + cost_book_2) * (1 - discount_books)
def subtotal := discounted_book_cost + cost_notebook + cost_bookmarks
def total_with_tax := subtotal * (1 + sales_tax)
def change := amount_given - total_with_tax

-- State the theorem
theorem sara_change : change = 3.41 := by
  sorry

end sara_change_l282_282335


namespace soccer_balls_are_20_l282_282149

variable (S : ℕ)
variable (num_baseballs : ℕ) (num_volleyballs : ℕ)
variable (condition_baseballs : num_baseballs = 5 * S)
variable (condition_volleyballs : num_volleyballs = 3 * S)
variable (condition_total : num_baseballs + num_volleyballs = 160)

theorem soccer_balls_are_20 :
  S = 20 :=
by
  sorry

end soccer_balls_are_20_l282_282149


namespace pies_can_be_made_l282_282605

def total_apples : Nat := 51
def apples_handout : Nat := 41
def apples_per_pie : Nat := 5

theorem pies_can_be_made :
  ((total_apples - apples_handout) / apples_per_pie) = 2 := by
  sorry

end pies_can_be_made_l282_282605


namespace negation_of_proposition_l282_282295

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x < 0 → x^3 - x^2 + 1 ≤ 0)) ↔ (∃ x : ℝ, x < 0 ∧ x^3 - x^2 + 1 > 0) :=
by
  sorry

end negation_of_proposition_l282_282295


namespace problem1_problem2_l282_282083

-- Problem (1)
theorem problem1 (x : ℝ) : (2 * |x - 1| ≥ 1) ↔ (x ≤ 1/2 ∨ x ≥ 3/2) := sorry

-- Problem (2)
theorem problem2 (a : ℝ) (h : a > 0) : (∀ x : ℝ, |a * x - 1| + |a * x - a| ≥ 1) ↔ a ≥ 2 := sorry

end problem1_problem2_l282_282083


namespace num_subsets_of_abc_eq_eight_l282_282301

theorem num_subsets_of_abc_eq_eight : 
  (∃ (s : Finset ℕ), s = {1, 2, 3} ∧ s.powerset.card = 8) :=
sorry

end num_subsets_of_abc_eq_eight_l282_282301


namespace min_value_of_x_l282_282842

-- Define the conditions and state the problem
theorem min_value_of_x (x : ℝ) : (∀ a : ℝ, a > 0 → x^2 < 1 + a) → x ≥ -1 :=
by
  sorry

end min_value_of_x_l282_282842


namespace area_of_triangle_l282_282409

-- Definitions of the conditions
def hypotenuse_AC (a b c : ℝ) : Prop := c = 50
def sum_of_legs (a b : ℝ) : Prop := a + b = 70
def pythagorean_theorem (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- The main theorem statement
theorem area_of_triangle (a b c : ℝ) (h1 : hypotenuse_AC a b c)
  (h2 : sum_of_legs a b) (h3 : pythagorean_theorem a b c) : 
  (1/2) * a * b = 300 := 
by
  sorry

end area_of_triangle_l282_282409


namespace least_integer_sol_l282_282480

theorem least_integer_sol (x : ℤ) (h : |(2 : ℤ) * x + 7| ≤ 16) : x ≥ -11 := sorry

end least_integer_sol_l282_282480


namespace terminal_side_in_second_quadrant_l282_282549

theorem terminal_side_in_second_quadrant (α : ℝ) (h : (Real.tan α < 0) ∧ (Real.cos α < 0)) :
  (2 < α / (π / 2)) ∧ (α / (π / 2) < 3) :=
by
  sorry

end terminal_side_in_second_quadrant_l282_282549


namespace coefficient_a7_l282_282071

theorem coefficient_a7 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ) (x : ℝ) 
  (h : x^9 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 
          + a_4 * (x - 1)^4 + a_5 * (x - 1)^5 + a_6 * (x - 1)^6 + a_7 * (x - 1)^7 
          + a_8 * (x - 1)^8 + a_9 * (x - 1)^9) : 
  a_7 = 36 := 
by
  sorry

end coefficient_a7_l282_282071


namespace max_a_monotonic_f_l282_282540

theorem max_a_monotonic_f {a : ℝ} (h1 : 0 < a)
  (h2 : ∀ x ≥ 1, 0 ≤ (3 * x^2 - a)) : a ≤ 3 := by
  -- Proof to be provided
  sorry

end max_a_monotonic_f_l282_282540


namespace composition_of_homotheties_l282_282126

-- Define points A1 and A2 and the coefficients k1 and k2
variables (A1 A2 : ℂ) (k1 k2 : ℂ)

-- Definition of homothety
def homothety (A : ℂ) (k : ℂ) (z : ℂ) : ℂ := k * (z - A) + A

-- Translation vector in case 1
noncomputable def translation_vector (A1 A2 : ℂ) (k1 k2 : ℂ) : ℂ :=
  if k1 * k2 = 1 then (1 - k1) * A1 + (k1 - 1) * A2 else 0 

-- Center A in case 2
noncomputable def center (A1 A2 : ℂ) (k1 k2 : ℂ) : ℂ :=
  if k1 * k2 ≠ 1 then (k2 * (1 - k1) * A1 + (1 - k2) * A2) / (k1 * k2 - 1) else 0

-- The final composition of two homotheties
noncomputable def composition (A1 A2 : ℂ) (k1 k2 : ℂ) (z : ℂ) : ℂ :=
  if k1 * k2 = 1 then z + translation_vector A1 A2 k1 k2
  else homothety (center A1 A2 k1 k2) (k1 * k2) z

-- The theorem to prove
theorem composition_of_homotheties 
  (A1 A2 : ℂ) (k1 k2 : ℂ) : ∀ z : ℂ,
  composition A1 A2 k1 k2 z = if k1 * k2 = 1 then z + translation_vector A1 A2 k1 k2
                              else homothety (center A1 A2 k1 k2) (k1 * k2) z := 
by sorry

end composition_of_homotheties_l282_282126


namespace find_m_l282_282825

-- Defining the sets and conditions
def A (m : ℝ) : Set ℝ := {1, m-2}
def B : Set ℝ := {x | x = 2}

theorem find_m (m : ℝ) (h : A m ∩ B = {2}) : m = 4 := by
  sorry

end find_m_l282_282825


namespace tangent_line_circle_l282_282700

theorem tangent_line_circle (m : ℝ) (h : ∀ x y : ℝ, (x + y = 0) → ((x - m)^2 + y^2 = 2)) : m = 2 :=
sorry

end tangent_line_circle_l282_282700


namespace total_distance_run_l282_282286

theorem total_distance_run (length : ℕ) (width : ℕ) (laps : ℕ) (h_length : length = 100) (h_width : width = 50) (h_laps : laps = 6) : 
  let perimeter := 2 * length + 2 * width in
  let distance := laps * perimeter in
  distance = 1800 :=
by
  sorry

end total_distance_run_l282_282286


namespace correct_propositions_l282_282789

-- Definitions of parallel and perpendicular
def parallel (x y : Type) : Prop := sorry
def perpendicular (x y : Type) : Prop := sorry

-- Main theorem
theorem correct_propositions (m n α β γ : Type) :
  ( (parallel m α ∧ parallel n β ∧ parallel α β → parallel m n) ∧
    (parallel α γ ∧ parallel β γ → parallel α β) ∧
    (perpendicular m α ∧ perpendicular n β ∧ parallel α β → parallel m n) ∧
    (perpendicular α γ ∧ perpendicular β γ → parallel α β) ) →
  ( (parallel α γ ∧ parallel β γ → parallel α β) ∧
    (perpendicular m α ∧ perpendicular n β ∧ parallel α β → parallel m n) ) :=
  sorry

end correct_propositions_l282_282789


namespace find_integer_solutions_l282_282371

noncomputable def integer_solutions (x y z w : ℤ) : Prop :=
  x * y * z / w + y * z * w / x + z * w * x / y + w * x * y / z = 4

theorem find_integer_solutions :
  { (x, y, z, w) : ℤ × ℤ × ℤ × ℤ |
    integer_solutions x y z w } =
  {(1,1,1,1), (-1,-1,-1,-1), (-1,-1,1,1), (-1,1,-1,1),
   (-1,1,1,-1), (1,-1,-1,1), (1,-1,1,-1), (1,1,-1,-1)} :=
by
  sorry

end find_integer_solutions_l282_282371


namespace sequence_is_increasing_l282_282021

def S (n : ℕ) : ℤ :=
  n^2 + 2 * n - 2

def a : ℕ → ℤ
| 0       => 0
| 1       => 1
| n + 1   => S (n + 1) - S n

theorem sequence_is_increasing : ∀ n m : ℕ, n < m → a n < a m :=
  sorry

end sequence_is_increasing_l282_282021


namespace angle_sum_around_point_l282_282325

theorem angle_sum_around_point (y : ℕ) (h1 : 210 + 3 * y = 360) : y = 50 := 
by 
  sorry

end angle_sum_around_point_l282_282325


namespace median_mean_l282_282744

theorem median_mean (n : ℕ) (h : n + 4 = 8) : (4 + 6 + 8 + 14 + 16) / 5 = 9.6 := by
  sorry

end median_mean_l282_282744


namespace sum_squares_not_perfect_square_l282_282903

theorem sum_squares_not_perfect_square (x y z : ℤ) (h : x^2 + y^2 + z^2 = 1993) : ¬ ∃ a : ℤ, x + y + z = a^2 :=
sorry

end sum_squares_not_perfect_square_l282_282903


namespace calculate_expression_l282_282514

variable (a : ℝ)

theorem calculate_expression : 2 * a - 7 * a + 4 * a = -a := by
  sorry

end calculate_expression_l282_282514


namespace prob_truth_same_time_l282_282899

theorem prob_truth_same_time (pA pB : ℝ) (hA : pA = 0.85) (hB : pB = 0.60) :
  pA * pB = 0.51 :=
by
  rw [hA, hB]
  norm_num

end prob_truth_same_time_l282_282899


namespace existence_of_function_values_around_k_l282_282517

-- Define the function f(n, m) with the given properties
def is_valid_function (f : ℤ × ℤ → ℤ) : Prop :=
  ∀ n m : ℤ, f (n, m) = (f (n-1, m) + f (n+1, m) + f (n, m-1) + f (n, m+1)) / 4

-- Theorem to prove the existence of such a function
theorem existence_of_function :
  ∃ (f : ℤ × ℤ → ℤ), is_valid_function f :=
sorry

-- Theorem to prove that for any k in ℤ, f(n, m) has values both greater and less than k
theorem values_around_k (k : ℤ) :
  ∃ (f : ℤ × ℤ → ℤ), is_valid_function f ∧ (∃ n1 m1 n2 m2, f (n1, m1) > k ∧ f (n2, m2) < k) :=
sorry

end existence_of_function_values_around_k_l282_282517


namespace fair_bets_allocation_l282_282442

theorem fair_bets_allocation (p_a : ℚ) (p_b : ℚ) (coins : ℚ) 
  (h_prob : p_a = 3 / 4 ∧ p_b = 1 / 4) (h_coins : coins = 96) : 
  (coins * p_a = 72) ∧ (coins * p_b = 24) :=
by 
  sorry

end fair_bets_allocation_l282_282442


namespace number_divisors_product_l282_282533

theorem number_divisors_product :
  ∃ N : ℕ, (∃ a b : ℕ, N = 3^a * 5^b ∧ (N^((a+1)*(b+1) / 2)) = 3^30 * 5^40) ∧ N = 3^3 * 5^4 :=
sorry

end number_divisors_product_l282_282533


namespace bug_meeting_point_l282_282992
-- Import the necessary library

-- Define the side lengths of the triangle
variables (DE EF FD : ℝ)
variables (bugs_meet : ℝ)

-- State the conditions and the result
theorem bug_meeting_point
  (h1 : DE = 6)
  (h2 : EF = 8)
  (h3 : FD = 10)
  (h4 : bugs_meet = 1 / 2 * (DE + EF + FD)) :
  bugs_meet - DE = 6 :=
by
  sorry

end bug_meeting_point_l282_282992


namespace smallest_value_of_n_l282_282317

theorem smallest_value_of_n 
  (n : ℕ) 
  (h1 : ∀ θ : ℝ, θ = (n - 2) * 180 / n) 
  (h2 : ∀ α : ℝ, α = 360 / n) 
  (h3 : 28 = 180 / n) :
  n = 45 :=
sorry

end smallest_value_of_n_l282_282317


namespace at_least_six_stones_empty_l282_282991

def frogs_on_stones (a : Fin 23 → Fin 23) (k : Nat) : Fin 22 → Fin 23 :=
  fun i => (a i + i.1 * k) % 23

theorem at_least_six_stones_empty 
  (a : Fin 22 → Fin 23) :
  ∃ k : Nat, ∀ (s : Fin 23), ∃ (j : Fin 22), frogs_on_stones (fun i => a i) k j ≠ s ↔ ∃! t : Fin 23, ∃! j, (frogs_on_stones (fun i => a i) k j) = t := 
  sorry

end at_least_six_stones_empty_l282_282991


namespace abs_ineq_solution_range_l282_282251

theorem abs_ineq_solution_range (a : ℝ) :
  (∃ x : ℝ, |x + 1| - |x - 2| > a) → a < 3 :=
by
  sorry

end abs_ineq_solution_range_l282_282251


namespace integer_divisibility_l282_282491

theorem integer_divisibility
  (x y z : ℤ)
  (h : 11 ∣ (7 * x + 2 * y - 5 * z)) :
  11 ∣ (3 * x - 7 * y + 12 * z) :=
sorry

end integer_divisibility_l282_282491


namespace ab_plus_cd_eq_neg_346_over_9_l282_282092

theorem ab_plus_cd_eq_neg_346_over_9 (a b c d : ℝ)
  (h1 : a + b + c = 5)
  (h2 : a + b + d = -3)
  (h3 : a + c + d = 10)
  (h4 : b + c + d = -1) :
  a * b + c * d = -346 / 9 := 
sorry

end ab_plus_cd_eq_neg_346_over_9_l282_282092


namespace find_b_l282_282097

-- Define the given hyperbola equation and conditions
def hyperbola (x y : ℝ) (b : ℝ) : Prop := x^2 - y^2 / b^2 = 1
def asymptote_line (x y : ℝ) : Prop := 2 * x - y = 0

-- State the theorem to prove
theorem find_b (b : ℝ) (hb : b > 0) :
    (∀ x y : ℝ, hyperbola x y b → asymptote_line x y) → b = 2 :=
by 
  sorry

end find_b_l282_282097


namespace min_rows_for_students_l282_282476

def min_rows (total_students seats_per_row max_students_per_school : ℕ) : ℕ :=
  total_students / seats_per_row + if total_students % seats_per_row == 0 then 0 else 1

theorem min_rows_for_students :
  ∀ (total_students seats_per_row max_students_per_school : ℕ),
  (total_students = 2016) →
  (seats_per_row = 168) →
  (max_students_per_school = 40) →
  min_rows total_students seats_per_row max_students_per_school = 15 :=
by
  intros total_students seats_per_row max_students_per_school h1 h2 h3
  -- We write down the proof outline to show that 15 is the required minimum
  sorry

end min_rows_for_students_l282_282476


namespace no_second_quadrant_l282_282098

theorem no_second_quadrant (k : ℝ) :
  (∀ x : ℝ, (x < 0 → 3 * x + k - 2 ≤ 0)) → k ≤ 2 :=
by
  intro h
  sorry

end no_second_quadrant_l282_282098


namespace Priya_driving_speed_l282_282124

/-- Priya's driving speed calculation -/
theorem Priya_driving_speed
  (time_XZ : ℝ) (rate_back : ℝ) (time_ZY : ℝ)
  (midway_condition : time_XZ = 5)
  (speed_back_condition : rate_back = 60)
  (time_back_condition : time_ZY = 2.0833333333333335) :
  ∃ speed_XZ : ℝ, speed_XZ = 50 :=
by
  have distance_ZY : ℝ := rate_back * time_ZY
  have distance_XZ : ℝ := 2 * distance_ZY
  have speed_XZ : ℝ := distance_XZ / time_XZ
  existsi speed_XZ
  sorry

end Priya_driving_speed_l282_282124


namespace find_d_l282_282753

open Nat

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ (∀ k : Nat, k > 1 → k < n → n % k ≠ 0)

def less_than_10_primes (n : Nat) : Prop :=
  n < 10 ∧ is_prime n

theorem find_d (d e f : Nat) (hd : less_than_10_primes d) (he : less_than_10_primes e) (hf : less_than_10_primes f) :
  d + e = f → d < e → d = 2 :=
by
  sorry

end find_d_l282_282753


namespace sum_of_decimals_l282_282786

theorem sum_of_decimals :
  let a := 0.35
  let b := 0.048
  let c := 0.0072
  a + b + c = 0.4052 := by
  sorry

end sum_of_decimals_l282_282786


namespace maximal_intersection_area_of_rectangles_l282_282597

theorem maximal_intersection_area_of_rectangles :
  ∀ (a b : ℕ), a * b = 2015 ∧ a < b →
  ∀ (c d : ℕ), c * d = 2016 ∧ c > d →
  ∃ (max_area : ℕ), max_area = 1302 ∧ ∀ intersection_area, intersection_area ≤ 1302 := 
by
  sorry

end maximal_intersection_area_of_rectangles_l282_282597


namespace smallest_number_greater_than_500000_has_56_positive_factors_l282_282429

/-- Let n be the smallest number greater than 500,000 
    that is the product of the first four terms of both
    an arithmetic sequence and a geometric sequence.
    Prove that n has 56 positive factors. -/
theorem smallest_number_greater_than_500000_has_56_positive_factors :
  ∃ n : ℕ,
    (500000 < n) ∧
    (∀ a d b r, a > 0 → d > 0 → b > 0 → r > 0 →
      n = (a * (a + d) * (a + 2 * d) * (a + 3 * d)) ∧
          n = (b * (b * r) * (b * r^2) * (b * r^3))) ∧
    (n.factors.length = 56) :=
by sorry

end smallest_number_greater_than_500000_has_56_positive_factors_l282_282429


namespace tim_scored_sum_first_8_even_numbers_l282_282763

-- Define the first 8 even numbers.
def first_8_even_numbers : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16]

-- Define the sum of those numbers.
def sum_first_8_even_numbers : ℕ := List.sum first_8_even_numbers

-- The theorem stating the problem.
theorem tim_scored_sum_first_8_even_numbers : sum_first_8_even_numbers = 72 := by
  sorry

end tim_scored_sum_first_8_even_numbers_l282_282763


namespace cost_of_eight_memory_cards_l282_282621

theorem cost_of_eight_memory_cards (total_cost_of_three: ℕ) (h: total_cost_of_three = 45) : 8 * (total_cost_of_three / 3) = 120 := by
  sorry

end cost_of_eight_memory_cards_l282_282621


namespace bookstore_shoe_store_common_sales_l282_282769

-- Define the conditions
def bookstore_sale_days (d: ℕ) : Prop := d % 4 = 0 ∧ d >= 4 ∧ d <= 28
def shoe_store_sale_days (d: ℕ) : Prop := (d - 2) % 6 = 0 ∧ d >= 2 ∧ d <= 26

-- Define the question to be proven as a theorem
theorem bookstore_shoe_store_common_sales : 
  ∃ (n: ℕ), n = 2 ∧ (
    ∀ (d: ℕ), 
      ((bookstore_sale_days d ∧ shoe_store_sale_days d) → n = 2) 
      ∧ (d < 4 ∨ d > 28 ∨ d < 2 ∨ d > 26 → n = 2)
  ) :=
sorry

end bookstore_shoe_store_common_sales_l282_282769


namespace margie_driving_distance_l282_282591

-- Define the constants given in the conditions
def mileage_per_gallon : ℝ := 40
def cost_per_gallon : ℝ := 5
def total_money : ℝ := 25

-- Define the expected result/answer
def expected_miles : ℝ := 200

-- The theorem that needs to be proved
theorem margie_driving_distance :
  (total_money / cost_per_gallon) * mileage_per_gallon = expected_miles :=
by
  -- proof goes here
  sorry

end margie_driving_distance_l282_282591


namespace solve_problem_l282_282678

variable (a b : ℝ)

def condition1 : Prop := a + b = 1
def condition2 : Prop := ab = -6

theorem solve_problem (h1 : condition1 a b) (h2 : condition2 a b) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = -150 :=
by
  sorry

end solve_problem_l282_282678


namespace group_combinations_l282_282408

theorem group_combinations (men women : ℕ) (h_men : men = 5) (h_women : women = 4) :
  (∃ (group4_men group4_women : ℕ), group4_men + group4_women = 4 ∧ group4_men ≥ 1 ∧ group4_women ≥ 1) →
  ((nat.choose men 2) * (nat.choose women 2) + (nat.choose men 1) * (nat.choose women 3)) = 80 :=
by
  intros group4_criteria
  simp [h_men, h_women]
  sorry

end group_combinations_l282_282408


namespace sum_of_digits_base8_l282_282170

theorem sum_of_digits_base8 (n : ℕ) (h : n = 888) : 
  (let d0 := (n % 8) 
       n1 := (n / 8)
       d1 := (n1 % 8) 
       n2 := (n1 / 8)
       d2 := (n2 % 8)
       n3 := (n2 / 8) in
   d0 + d1 + d2 + n3) = 13 :=
by {
  have h₁ : n = 888,
  { exact h, },
  let b8 := 8,
  let step1 := n % b8,
  let n1 := n / b8,
  let step2 := n1 % b8,
  let n2 := n1 / b8,
  let step3 := n2 % b8,
  let n3 := n2 / b8,
  have h₂ : step1 = 0,
  { rw h₁, norm_num, },
  have h₃ : step2 = 7,
  { rw h₁, norm_num, },
  have h₄ : step3 = 5,
  { rw h₁, norm_num, },
  have h₅ : n3 = 1,
  { rw h₁, norm_num, },
  rw [h₂, h₃, h₄, h₅],
  norm_num,
  sorry
}

end sum_of_digits_base8_l282_282170


namespace farmer_land_l282_282123

variable (T : ℝ) -- Total land owned by the farmer

def is_cleared (T : ℝ) : ℝ := 0.90 * T
def cleared_barley (T : ℝ) : ℝ := 0.80 * is_cleared T
def cleared_potato (T : ℝ) : ℝ := 0.10 * is_cleared T
def cleared_tomato : ℝ := 90
def cleared_land (T : ℝ) : ℝ := cleared_barley T + cleared_potato T + cleared_tomato

theorem farmer_land (T : ℝ) (h : cleared_land T = is_cleared T) : T = 1000 := sorry

end farmer_land_l282_282123


namespace hyperbola_asymptote_slopes_l282_282665

theorem hyperbola_asymptote_slopes:
  (∀ (x y : ℝ), (x^2 / 144 - y^2 / 81 = 1) → (y = (3 / 4) * x ∨ y = -(3 / 4) * x)) :=
by
  sorry

end hyperbola_asymptote_slopes_l282_282665


namespace triangle_angle_contradiction_l282_282314

theorem triangle_angle_contradiction (A B C : ℝ) (h_sum : A + B + C = 180) (h_lt_60 : A < 60 ∧ B < 60 ∧ C < 60) : false := 
sorry

end triangle_angle_contradiction_l282_282314


namespace number_of_correct_propositions_l282_282236

noncomputable def arithmetic_sequence (a n : ℕ → ℝ) := ∃ d, ∀ n, a (n + 1) = a n + d
noncomputable def geometric_sequence (a n : ℕ → ℝ) := ∃ r, ∀ n, a (n + 1) = a n * r

def sum_seq (a : ℕ → ℝ) (n : ℕ) := ∑ i in range (n + 1), a i

noncomputable def seq_prop_1 (a : ℕ → ℝ) :=
  (arithmetic_sequence a) →
  collinear [(10 : ℕ, sum_seq a 10 / 10), 
             (100 : ℕ, sum_seq a 100 / 100), 
             (110 : ℕ, sum_seq a 110 / 110)]
 
noncomputable def seq_prop_2 (a : ℕ → ℝ) :=
  (geometric_sequence a) →
  ∀ m : ℕ+, geometric_sequence (λ n, [sum_seq a m, 
                                       sum_seq a (2 * m) - sum_seq a m, 
                                       sum_seq a (3 * m) - sum_seq a (2 * m)].nth n.succ)

noncomputable def seq_prop_3 (a : ℕ → ℝ) :=
  (geometric_sequence a) →
  (∀ n : ℕ+, (n, sum_seq a n) ∈ set_of (λ (p : ℕ × ℝ), ∃ b r, b ≠ 0 ∧ b ≠ 1 ∧ p.snd = b ^ p.fst + r)) →
  ∃ (r : ℝ), r = -1

noncomputable def seq_prop_4 (a : ℕ → ℝ) :=
  (a 1 = 2) → 
  (∀ n : ℕ, a (n + 1) - a n = 2 ^ n) →
  ∀ n, sum_seq a n = 2^(n+1) - 2

theorem number_of_correct_propositions :
  (∃ a1 a2 a3 a4 : ℕ → ℝ, seq_prop_1 a1 ∧ seq_prop_2 a2 ∧ seq_prop_3 a3 ∧ seq_prop_4 a4)
  → 3 = 3 :=
by sorry

end number_of_correct_propositions_l282_282236


namespace intersection_complement_eq_l282_282638

open Set

namespace MathProof

variable (U A B : Set ℕ)

theorem intersection_complement_eq :
  U = {1, 2, 3, 4, 5, 6, 7} →
  A = {3, 4, 5} →
  B = {1, 3, 6} →
  A ∩ (U \ B) = {4, 5} :=
by
  intros hU hA hB
  sorry

end MathProof

end intersection_complement_eq_l282_282638


namespace largest_number_among_l282_282201

theorem largest_number_among (π: ℝ) (sqrt_2: ℝ) (neg_2: ℝ) (three: ℝ)
  (h1: 3.14 ≤ π)
  (h2: 1 < sqrt_2 ∧ sqrt_2 < 2)
  (h3: neg_2 < 1)
  (h4: 3 < π) :
  (neg_2 < sqrt_2) ∧ (sqrt_2 < 3) ∧ (3 < π) :=
by {
  sorry
}

end largest_number_among_l282_282201


namespace cookies_per_bag_l282_282895

theorem cookies_per_bag (n_bags : ℕ) (total_cookies : ℕ) (n_candies : ℕ) (h_bags : n_bags = 26) (h_cookies : total_cookies = 52) (h_candies : n_candies = 15) : (total_cookies / n_bags) = 2 :=
by sorry

end cookies_per_bag_l282_282895


namespace fraction_integer_l282_282720

theorem fraction_integer (x y : ℤ) (h₁ : ∃ k : ℤ, 3 * x + 4 * y = 5 * k) : ∃ m : ℤ, 4 * x - 3 * y = 5 * m :=
by
  sorry

end fraction_integer_l282_282720


namespace sum_of_base_8_digits_888_l282_282174

def base_8_representation (n : ℕ) : ℕ := 
  let d0 := n % 8
  let n  := n / 8
  let d1 := n % 8
  let n  := n / 8
  let d2 := n % 8
  let n  := n / 8
  let d3 := n % 8
  d3 * 1000 + d2 * 100 + d1 * 10 + d0

def sum_of_digits (n : ℕ) : ℕ :=
  n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

theorem sum_of_base_8_digits_888 : 
  sum_of_digits (base_8_representation 888) = 13 :=
by
  sorry

end sum_of_base_8_digits_888_l282_282174


namespace boys_in_biology_is_25_l282_282154

-- Definition of the total number of students in the Physics class
def physics_class_students : ℕ := 200

-- Definition of the total number of students in the Biology class
def biology_class_students : ℕ := physics_class_students / 2

-- Condition that there are three times as many girls as boys in the Biology class
def girls_boys_ratio : ℕ := 3

-- Calculate the total number of "parts" in the Biology class (3 parts girls + 1 part boys)
def total_parts : ℕ := girls_boys_ratio + 1

-- The number of boys in the Biology class
def boys_in_biology : ℕ := biology_class_students / total_parts

-- The statement to prove the number of boys in the Biology class is 25
theorem boys_in_biology_is_25 : boys_in_biology = 25 := by
  sorry

end boys_in_biology_is_25_l282_282154


namespace three_digit_numbers_not_multiple_of_3_or_11_l282_282554

-- Proving the number of three-digit numbers that are multiples of neither 3 nor 11 is 547
theorem three_digit_numbers_not_multiple_of_3_or_11 : (finset.Icc 100 999).filter (λ n, ¬(3 ∣ n) ∧ ¬(11 ∣ n)).card = 547 :=
by
  -- The steps to reach the solution will be implemented here
  sorry

end three_digit_numbers_not_multiple_of_3_or_11_l282_282554


namespace find_angle_A_find_area_l282_282818

-- Definitions
variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
def law_c1 (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * Real.cos C + c * Real.cos A = -2 * b * Real.cos A

def law_c2 (a : ℝ) : Prop := a = 2 * Real.sqrt 3
def law_c3 (b c : ℝ) : Prop := b + c = 4

-- Questions
theorem find_angle_A (A B C : ℝ) (a b c : ℝ) (h1 : law_c1 A B C a b c) (h2 : law_c2 a) (h3 : law_c3 b c) : 
  A = 2 * Real.pi / 3 :=
sorry

theorem find_area (A B C : ℝ) (a b c : ℝ) (h1 : law_c1 A B C a b c) (h2 : law_c2 a) (h3 : law_c3 b c)
  (hA : A = 2 * Real.pi / 3) : 
  (1/2) * b * c * Real.sin A = Real.sqrt 3 :=
sorry

end find_angle_A_find_area_l282_282818


namespace ordered_pair_A_B_l282_282014

open Polynomial

noncomputable def system_sum_of_roots : Prop :=
  let y := (x ^ 3 - 3 * x + 2)
  let eq1 := (2 * x + 3 * y = 3)
  ∃ x1 x2 x3 y1 y2 y3 : ℝ, (y1 = x1 ^ 3 - 3 * x1 + 2) ∧ (y2 = x2 ^ 3 - 3 * x2 + 2) ∧ (y3 = x3 ^ 3 - 3 * x3 + 2) ∧ 
                         (2 * x1 + 3 * y1 = 3) ∧ (2 * x2 + 3 * y2 = 3) ∧ (2 * x3 + 3 * y3 = 3) ∧ 
                         (x1 + x2 + x3 = 0) ∧ (y1 + y2 + y3 = 3)

theorem ordered_pair_A_B : 
  system_sum_of_roots :=
sorry

end ordered_pair_A_B_l282_282014


namespace loss_percentage_l282_282339

theorem loss_percentage (cost_price selling_price : ℝ) (h_cost : cost_price = 1500) (h_sell : selling_price = 1260) : 
  (cost_price - selling_price) / cost_price * 100 = 16 := 
by
  sorry

end loss_percentage_l282_282339


namespace intercepts_of_line_l282_282215

theorem intercepts_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) :
  (∃ x_intercept : ℝ, x_intercept = 7 ∧ (4 * x_intercept + 7 * 0 = 28)) ∧
  (∃ y_intercept : ℝ, y_intercept = 4 ∧ (4 * 0 + 7 * y_intercept = 28)) :=
by
  sorry

end intercepts_of_line_l282_282215


namespace non_parallel_lines_implies_unique_solution_l282_282194

variable (a1 b1 c1 a2 b2 c2 : ℝ)

def system_of_equations (x y : ℝ) := a1 * x + b1 * y = c1 ∧ a2 * x + b2 * y = c2

def lines_not_parallel := a1 * b2 ≠ a2 * b1

theorem non_parallel_lines_implies_unique_solution :
  lines_not_parallel a1 b1 a2 b2 → ∃! (x y : ℝ), system_of_equations a1 b1 c1 a2 b2 c2 x y :=
sorry

end non_parallel_lines_implies_unique_solution_l282_282194


namespace min_value_of_expression_l282_282117

theorem min_value_of_expression (x y : ℝ) : 
  ∃ m : ℝ, m = (xy - 1)^2 + (x + y)^2 ∧ (∀ x y : ℝ, (xy - 1)^2 + (x + y)^2 ≥ m) := 
sorry

end min_value_of_expression_l282_282117


namespace ab_plus_cd_eq_neg_346_over_9_l282_282091

theorem ab_plus_cd_eq_neg_346_over_9 (a b c d : ℝ)
  (h1 : a + b + c = 5)
  (h2 : a + b + d = -3)
  (h3 : a + c + d = 10)
  (h4 : b + c + d = -1) :
  a * b + c * d = -346 / 9 := 
sorry

end ab_plus_cd_eq_neg_346_over_9_l282_282091


namespace trig_identity_l282_282074

open Real

theorem trig_identity (α : ℝ) (h : tan α = 2) :
  2 * cos (2 * α) + 3 * sin (2 * α) - sin (α) ^ 2 = 2 / 5 :=
by sorry

end trig_identity_l282_282074


namespace count_prime_sum_112_l282_282760

noncomputable def primeSum (primes : List ℕ) : ℕ :=
  if H : ∀ p ∈ primes, Nat.Prime p ∧ p > 10 then primes.sum else 0

theorem count_prime_sum_112 :
  ∃ (primes : List ℕ), primeSum primes = 112 ∧ primes.length = 6 := by
  sorry

end count_prime_sum_112_l282_282760


namespace quotient_of_even_and_odd_composites_l282_282535

theorem quotient_of_even_and_odd_composites:
  (4 * 6 * 8 * 10 * 12) / (9 * 15 * 21 * 25 * 27) = 512 / 28525 := by
sorry

end quotient_of_even_and_odd_composites_l282_282535


namespace math_problem_l282_282252

noncomputable def f (x : ℝ) : ℝ := 
  2 * Real.sin x * Real.cos (Real.pi / 2 - x) - 
  Real.sqrt 3 * Real.sin (Real.pi + x) * Real.cos x + 
  Real.sin (Real.pi / 2 + x) * Real.cos x

theorem math_problem
  (a b c A B C : ℝ)
  (h1 : c * Real.sin A = Real.sqrt 3 * a * Real.cos C)
  (h2 : (a - c) * (a + c) = b * (b - c))
  (h3 : 0 < A ∧ A < Real.pi)
  (h4 : 0 < C ∧ C < Real.pi)
  (hB : B = Real.pi - A - C) : 
  ∃ T, T = Real.pi ∧ (f B = 5 / 2) := 
sorry

end math_problem_l282_282252


namespace travel_time_l282_282462

-- Definitions from problem conditions
def scale := 3000000
def map_distance_cm := 6
def conversion_factor_cm_to_km := 30000 -- derived from 1 cm on the map equals 30,000 km in reality
def speed_kmh := 30

-- The travel time we want to prove
theorem travel_time : (map_distance_cm * conversion_factor_cm_to_km / speed_kmh) = 6000 := 
by
  sorry

end travel_time_l282_282462


namespace baseball_to_football_ratio_l282_282109

theorem baseball_to_football_ratio (total_cards : ℕ) (baseball_cards : ℕ) (football_cards : ℕ)
  (h_total : total_cards = 125)
  (h_baseball : baseball_cards = 95)
  (h_football : football_cards = total_cards - baseball_cards) :
  (baseball_cards : ℚ) / football_cards = 19 / 6 :=
by
  sorry

end baseball_to_football_ratio_l282_282109


namespace house_spirits_elevator_l282_282571

-- Define the given conditions
def first_floor_domovoi := 1
def middle_floor_domovoi := 2
def last_floor_domovoi := 1
def total_floors := 7
def spirits_per_cycle := first_floor_domovoi + 5 * middle_floor_domovoi + last_floor_domovoi

-- Prove the statement
theorem house_spirits_elevator (n : ℕ) (floor : ℕ) (h1 : total_floors = 7) (h2 : spirits_per_cycle = 12) (h3 : n = 1000) :
  floor = 4 :=
by
  sorry

end house_spirits_elevator_l282_282571


namespace negation_correct_l282_282746

variable {α : Type*} (A B : Set α)

-- Define the original proposition
def original_proposition : Prop := A ∪ B = A → A ∩ B = B

-- Define the negation of the original proposition
def negation_proposition : Prop := A ∪ B ≠ A → A ∩ B ≠ B

-- State that the negation of the original proposition is equivalent to the negation proposition
theorem negation_correct : ¬(original_proposition A B) ↔ negation_proposition A B := by sorry

end negation_correct_l282_282746


namespace largest_digit_divisible_by_6_l282_282031

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem largest_digit_divisible_by_6 (N : ℕ) (hN : N ≤ 9) :
  (∃ m : ℕ, 56780 + N = m * 6) ∧ is_even N ∧ is_divisible_by_3 (26 + N) → N = 4 := by
  sorry

end largest_digit_divisible_by_6_l282_282031


namespace min_expression_value_l282_282671

theorem min_expression_value (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (∃ (min_val : ℝ), min_val = 12 ∧ (∀ (x y : ℝ), (x > 1) → (y > 1) →
  ((x^2 / (y - 1)) + (y^2 / (x - 1)) + (x + y) ≥ min_val))) :=
by
  sorry

end min_expression_value_l282_282671


namespace largest_time_for_77_degrees_l282_282100

-- Define the initial conditions of the problem
def temperature_eqn (t : ℝ) : ℝ := -t^2 + 14 * t + 40

-- Define the proposition we want to prove
theorem largest_time_for_77_degrees : ∃ t, temperature_eqn t = 77 ∧ t = 11 := 
sorry

end largest_time_for_77_degrees_l282_282100


namespace like_terms_monomials_m_n_l282_282838

theorem like_terms_monomials_m_n (m n : ℕ) (h1 : 3 * x ^ m * y = - x ^ 3 * y ^ n) :
  m - n = 2 :=
by
  sorry

end like_terms_monomials_m_n_l282_282838


namespace cost_of_gas_used_l282_282578

theorem cost_of_gas_used (initial_odometer final_odometer fuel_efficiency cost_per_gallon : ℝ)
  (h₀ : initial_odometer = 82300)
  (h₁ : final_odometer = 82335)
  (h₂ : fuel_efficiency = 22)
  (h₃ : cost_per_gallon = 3.80) :
  (final_odometer - initial_odometer) / fuel_efficiency * cost_per_gallon = 6.04 :=
by
  sorry

end cost_of_gas_used_l282_282578


namespace functional_equation_solution_l282_282951

theorem functional_equation_solution (f g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, g (f (x + y)) = f x + 2 * (x + y) * g y) : 
  (∀ x : ℝ, f x = 0) ∧ (∀ x : ℝ, g x = 0) :=
sorry

end functional_equation_solution_l282_282951


namespace total_animals_after_addition_l282_282596

def current_cows := 2
def current_pigs := 3
def current_goats := 6

def added_cows := 3
def added_pigs := 5
def added_goats := 2

def total_current_animals := current_cows + current_pigs + current_goats
def total_added_animals := added_cows + added_pigs + added_goats
def total_animals := total_current_animals + total_added_animals

theorem total_animals_after_addition : total_animals = 21 := by
  sorry

end total_animals_after_addition_l282_282596


namespace problem_statement_l282_282680

variable {α : Type*} [LinearOrderedCommRing α]

theorem problem_statement (a b c d e : α) (h : a * b^2 * c^3 * d^4 * e^5 < 0) : a * b^2 * c * d^4 * e < 0 :=
by
  sorry

end problem_statement_l282_282680


namespace quadrilateral_area_l282_282669

variable (d : ℝ) (o₁ : ℝ) (o₂ : ℝ)

theorem quadrilateral_area (h₁ : d = 28) (h₂ : o₁ = 8) (h₃ : o₂ = 2) : 
  (1 / 2 * d * o₁) + (1 / 2 * d * o₂) = 140 := 
  by
    rw [h₁, h₂, h₃]
    sorry

end quadrilateral_area_l282_282669


namespace Amanda_car_round_trip_time_l282_282200

theorem Amanda_car_round_trip_time (bus_time : ℕ) (car_reduction : ℕ) (bus_one_way_trip : bus_time = 40) (car_time_reduction : car_reduction = 5) : 
  (2 * (bus_time - car_reduction)) = 70 := 
by
  sorry

end Amanda_car_round_trip_time_l282_282200


namespace minimum_rows_required_l282_282469

theorem minimum_rows_required
  (seats_per_row : ℕ)
  (total_students : ℕ)
  (max_students_per_school : ℕ)
  (H1 : seats_per_row = 168)
  (H2 : total_students = 2016)
  (H3 : max_students_per_school = 40)
  : ∃ n : ℕ, n = 15 ∧ (∀ configuration : List (List ℕ), configuration.length = n ∧ 
       (∀ school_students, school_students ∈ configuration → school_students.length ≤ seats_per_row) ∧
       ∀ i, ∃ (c : ℕ) (school_students : ℕ), school_students ≤ max_students_per_school ∧
         i < total_students - ∑ configuration.head! length → 
         true) :=
sorry

end minimum_rows_required_l282_282469


namespace simplify_expression_l282_282129

noncomputable def problem_expression : ℝ :=
  (0.25)^(-2) + 8^(2/3) - real.log 25 / real.log 10 - 2 * (real.log 2 / real.log 10)

theorem simplify_expression : problem_expression = 18 :=
by
  sorry

end simplify_expression_l282_282129


namespace numerator_of_first_fraction_l282_282987

theorem numerator_of_first_fraction (y : ℝ) (h : y > 0) (x : ℝ) 
  (h_eq : (x / y) * y + (3 * y) / 10 = 0.35 * y) : x = 32 := 
by
  sorry

end numerator_of_first_fraction_l282_282987


namespace sum_of_digits_base8_888_l282_282177

/-- Definition of the function to convert a number to a specified base -/
def convert_to_base (n : ℕ) (b : ℕ) : list ℕ := 
  if b ≤ 1 then [n] 
  else let rec aux (n: ℕ) (acc: list ℕ) : list ℕ :=
    if n = 0 then acc 
    else aux (n / b) ((n % b) :: acc) 
  in aux n []

/-- Definition of the function to sum the digits in a list -/
def sum_of_digits (digits : list ℕ) : ℕ :=
  digits.sum

/-- Problem statement: sum of the digits in the base 8 representation of 888 -/
theorem sum_of_digits_base8_888 : sum_of_digits (convert_to_base 888 8) = 13 := 
  sorry

end sum_of_digits_base8_888_l282_282177


namespace sum_of_bases_is_20_l282_282574

theorem sum_of_bases_is_20
  (B1 B2 : ℕ)
  (G1 : ℚ)
  (G2 : ℚ)
  (hG1_B1 : G1 = (4 * B1 + 5) / (B1^2 - 1))
  (hG2_B1 : G2 = (5 * B1 + 4) / (B1^2 - 1))
  (hG1_B2 : G1 = (3 * B2) / (B2^2 - 1))
  (hG2_B2 : G2 = (6 * B2) / (B2^2 - 1)) :
  B1 + B2 = 20 :=
sorry

end sum_of_bases_is_20_l282_282574


namespace minimum_filtration_process_l282_282785

noncomputable def filtration_process (n : ℕ) : Prop :=
  (0.8 : ℝ) ^ n < 0.05

theorem minimum_filtration_process : ∃ n : ℕ, filtration_process n ∧ n ≥ 14 := 
  sorry

end minimum_filtration_process_l282_282785


namespace find_k_l282_282180

-- Define the conditions and the question
theorem find_k (t k : ℝ) (h1 : t = 50) (h2 : t = (5 / 9) * (k - 32)) : k = 122 := by
  -- Proof will go here
  sorry

end find_k_l282_282180


namespace percentage_difference_liliane_alice_l282_282263

theorem percentage_difference_liliane_alice :
  let J := 200
  let L := 1.30 * J
  let A := 1.15 * J
  (L - A) / A * 100 = 13.04 :=
by
  sorry

end percentage_difference_liliane_alice_l282_282263


namespace points_on_circle_l282_282219

theorem points_on_circle (t : ℝ) : ∃ x y : ℝ, x = Real.cos t ∧ y = Real.sin t ∧ x^2 + y^2 = 1 :=
by
  sorry

end points_on_circle_l282_282219


namespace number_of_carbons_l282_282047

-- Definitions of given conditions
def molecular_weight (total_c total_h total_o c_weight h_weight o_weight : ℕ) := 
    total_c * c_weight + total_h * h_weight + total_o * o_weight

-- Given values
def num_hydrogen_atoms : ℕ := 8
def num_oxygen_atoms : ℕ := 2
def molecular_wt : ℕ := 88
def atomic_weight_c : ℕ := 12
def atomic_weight_h : ℕ := 1
def atomic_weight_o : ℕ := 16

-- The theorem to be proved
theorem number_of_carbons (num_carbons : ℕ) 
    (H_hydrogen : num_hydrogen_atoms = 8)
    (H_oxygen : num_oxygen_atoms = 2)
    (H_molecular_weight : molecular_wt = 88)
    (H_atomic_weight_c : atomic_weight_c = 12)
    (H_atomic_weight_h : atomic_weight_h = 1)
    (H_atomic_weight_o : atomic_weight_o = 16) :
    molecular_weight num_carbons num_hydrogen_atoms num_oxygen_atoms atomic_weight_c atomic_weight_h atomic_weight_o = molecular_wt → 
    num_carbons = 4 :=
by
  intros h
  sorry 

end number_of_carbons_l282_282047


namespace point_in_which_quadrant_l282_282441

theorem point_in_which_quadrant (x y : ℝ) (h1 : y = 2 * x + 3) (h2 : abs x = abs y) :
  (x < 0 ∧ y < 0) ∨ (x < 0 ∧ y > 0) :=
by
  -- Proof omitted
  sorry

end point_in_which_quadrant_l282_282441


namespace geometric_sum_is_correct_l282_282936

theorem geometric_sum_is_correct : 
  let a := 1
  let r := 5
  let n := 6
  a * (r^n - 1) / (r - 1) = 3906 := by
  sorry

end geometric_sum_is_correct_l282_282936


namespace part1_part2_part3_l282_282081

-- Define the function f(x)
def f (x a b : ℝ) : ℝ := log x - a * x + b / x

-- Condition: f(x) + f(1/x) = 0
def condition (x a b : ℝ) : Prop := f x a b + f (1/x) a b = 0

-- Prove a = -2 when f has a tangent at x=1 passing through (0, -5)
theorem part1 (x a b : ℝ) (h : condition x a b) (tangent_line : ∃ m, ∀ (y : ℝ), y = f 1 a b + m * (y - 1) → (0, -5) ∈ tangent_line) :
  a = -2 := sorry

-- Prove f(a^2 / 2) > 0 given 0 < a < 1
theorem part2 (a : ℝ) (h : 0 < a ∧ a < 1) : f (a^2 / 2) a a > 0 := sorry

-- Prove the range of a when f has three distinct zeros
theorem part3 (a b : ℝ) (h : condition a a b) (h2 : number_of_zeros f = 3) :
  0 < a ∧ a < 1 / 2 := sorry

end part1_part2_part3_l282_282081


namespace mystical_mountain_creatures_l282_282708

-- Definitions for conditions
def nineHeadedBirdHeads : Nat := 9
def nineHeadedBirdTails : Nat := 1
def nineTailedFoxHeads : Nat := 1
def nineTailedFoxTails : Nat := 9

-- Prove the number of Nine-Tailed Foxes
theorem mystical_mountain_creatures (x y : Nat)
  (h1 : 9 * x + (y - 1) = 36 * (y - 1) + 4 * x)
  (h2 : 9 * (x - 1) + y = 3 * (9 * y + (x - 1))) :
  x = 14 :=
by
  sorry

end mystical_mountain_creatures_l282_282708


namespace find_quotient_l282_282254

theorem find_quotient (divisor remainder dividend : ℕ) (h_divisor : divisor = 24) (h_remainder : remainder = 5) (h_dividend : dividend = 1565) : 
  (dividend - remainder) / divisor = 65 :=
by
  sorry

end find_quotient_l282_282254


namespace range_of_m_l282_282814

/-- Given the conditions:
- \( \left|1 - \frac{x - 2}{3}\right| \leq 2 \)
- \( x^2 - 2x + 1 - m^2 \leq 0 \) where \( m > 0 \)
- \( \neg \left( \left|1 - \frac{x - 2}{3}\right| \leq 2 \right) \) is a necessary but not sufficient condition for \( x^2 - 2x + 1 - m^2 \leq 0 \)

Prove that the range of \( m \) is \( m \geq 10 \).
-/
theorem range_of_m (m : ℝ) (x : ℝ)
  (h1 : ∀ x, ¬(abs (1 - (x - 2) / 3) ≤ 2) → x < -1 ∨ x > 11)
  (h2 : ∀ x, ∀ m > 0, x^2 - 2 * x + 1 - m^2 ≤ 0)
  : m ≥ 10 :=
sorry

end range_of_m_l282_282814


namespace three_digit_non_multiples_of_3_or_11_l282_282558

theorem three_digit_non_multiples_of_3_or_11 : 
  ∃ (n : ℕ), n = 546 ∧ 
  (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → 
    ¬ (x % 3 = 0 ∨ x % 11 = 0) → 
    n = (900 - (300 + 81 - 27))) := 
by 
  sorry

end three_digit_non_multiples_of_3_or_11_l282_282558


namespace divisible_by_factorial_l282_282758

def f : ℕ → ℕ → ℕ
| 0, 0 => 1
| 0, _ => 0
| _, 0 => 0
| n + 1, k + 1 => (n + 1) * (f (n + 1) k + f n k)

theorem divisible_by_factorial (n k : ℕ) : n! ∣ f n k := by sorry

end divisible_by_factorial_l282_282758


namespace keith_attended_games_l282_282156

def total_games : ℕ := 8
def missed_games : ℕ := 4
def attended_games (total : ℕ) (missed : ℕ) : ℕ := total - missed

theorem keith_attended_games : attended_games total_games missed_games = 4 := by
  sorry

end keith_attended_games_l282_282156


namespace equidistant_point_l282_282374

/-- 
  Find the point in the xz-plane that is equidistant from the points (1, 0, 0), 
  (0, -2, 3), and (4, 2, -2). The point in question is \left( \frac{41}{7}, 0, -\frac{19}{14} \right).
-/
theorem equidistant_point :
  ∃ (x z : ℚ), 
    (x - 1)^2 + z^2 = x^2 + 4 + (z - 3)^2 ∧
    (x - 1)^2 + z^2 = (x - 4)^2 + 4 + (z + 2)^2 ∧
    x = 41 / 7 ∧ z = -19 / 14 :=
by
  sorry

end equidistant_point_l282_282374


namespace min_rows_needed_l282_282472

-- Define the basic conditions
def total_students := 2016
def seats_per_row := 168
def max_students_per_school := 40

-- Define the minimum number of rows required to accommodate all conditions
noncomputable def min_required_rows (students : ℕ) (seats : ℕ) (max_per_school : ℕ) : ℕ := 15

-- Lean theorem asserting the truth of the above definition under given conditions
theorem min_rows_needed : min_required_rows total_students seats_per_row max_students_per_school = 15 :=
by
  -- Proof omitted
  sorry

end min_rows_needed_l282_282472


namespace sum_largest_smallest_gx_l282_282719

noncomputable def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2 * x - 8| + 3

theorem sum_largest_smallest_gx : (∀ x, 1 ≤ x ∧ x ≤ 10 → True) → ∀ (a b : ℝ), (∃ x, 1 ≤ x ∧ x ≤ 10 ∧ g x = a) → (∃ y, 1 ≤ y ∧ y ≤ 10 ∧ g y = b) → a + b = -1 :=
by
  intro h x y hx hy
  sorry

end sum_largest_smallest_gx_l282_282719


namespace change_received_l282_282876

variable (a : ℕ)

theorem change_received (h : a ≤ 30) : 100 - 3 * a = 100 - 3 * a := 
by 
  sorry

end change_received_l282_282876


namespace wilson_buys_3_bottles_of_cola_l282_282334

theorem wilson_buys_3_bottles_of_cola
    (num_hamburgers : ℕ := 2) 
    (cost_per_hamburger : ℕ := 5) 
    (cost_per_cola : ℕ := 2) 
    (discount : ℕ := 4) 
    (total_paid : ℕ := 12) :
    num_hamburgers * cost_per_hamburger - discount + x * cost_per_cola = total_paid → x = 3 :=
by
  sorry

end wilson_buys_3_bottles_of_cola_l282_282334


namespace find_m_l282_282973

/-- Given vectors \(\overrightarrow{OA} = (1, m)\) and \(\overrightarrow{OB} = (m-1, 2)\), if 
\(\overrightarrow{OA} \perp \overrightarrow{AB}\), then \(m = \frac{1}{3}\). -/
theorem find_m (m : ℝ) (h : (1, m).1 * (m - 1 - 1, 2 - m).1 + (1, m).2 * (m - 1 - 1, 2 - m).2 = 0) :
  m = 1 / 3 :=
sorry

end find_m_l282_282973


namespace maximum_value_l282_282114

variables (a b c : ℝ)
variables (a_vec b_vec c_vec : EuclideanSpace ℝ (Fin 3))

axiom norm_a : ‖a_vec‖ = 2
axiom norm_b : ‖b_vec‖ = 3
axiom norm_c : ‖c_vec‖ = 4

theorem maximum_value : 
  (‖(a_vec - (3:ℝ) • b_vec)‖^2 + ‖(b_vec - (3:ℝ) • c_vec)‖^2 + ‖(c_vec - (3:ℝ) • a_vec)‖^2) ≤ 377 :=
by
  sorry

end maximum_value_l282_282114


namespace triangle_perimeter_l282_282610

-- Define the ratios
def ratio1 : ℚ := 1 / 2
def ratio2 : ℚ := 1 / 3
def ratio3 : ℚ := 1 / 4

-- Define the longest side
def longest_side : ℚ := 48

-- Compute the perimeter given the conditions
theorem triangle_perimeter (ratio1 ratio2 ratio3 : ℚ) (longest_side : ℚ) 
  (h_ratio1 : ratio1 = 1 / 2) (h_ratio2 : ratio2 = 1 / 3) (h_ratio3 : ratio3 = 1 / 4)
  (h_longest_side : longest_side = 48) : 
  (longest_side * 6/ (ratio1 * 12 + ratio2 * 12 + ratio3 * 12)) = 104 := by
  sorry

end triangle_perimeter_l282_282610


namespace problem1_l282_282601

variable (x y : ℝ)
variable (h1 : x = Real.sqrt 3 + Real.sqrt 5)
variable (h2 : y = Real.sqrt 3 - Real.sqrt 5)

theorem problem1 : 2 * x^2 - 4 * x * y + 2 * y^2 = 40 :=
by sorry

end problem1_l282_282601


namespace minimum_value_128_l282_282116

theorem minimum_value_128 (a b c : ℝ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) (h_prod: a * b * c = 8) : 
  (2 * a + b) * (a + 3 * c) * (b * c + 2) ≥ 128 := 
by
  sorry

end minimum_value_128_l282_282116


namespace technicians_count_l282_282573

theorem technicians_count 
    (total_workers : ℕ) (avg_salary_all : ℕ) (avg_salary_technicians : ℕ) (avg_salary_rest : ℕ)
    (h_workers : total_workers = 28) (h_avg_all : avg_salary_all = 8000) 
    (h_avg_tech : avg_salary_technicians = 14000) (h_avg_rest : avg_salary_rest = 6000) : 
    ∃ T R : ℕ, T + R = total_workers ∧ (avg_salary_technicians * T + avg_salary_rest * R = avg_salary_all * total_workers) ∧ T = 7 :=
by
  sorry

end technicians_count_l282_282573


namespace find_circle_center_l282_282802

noncomputable def circle_center : (ℝ × ℝ) :=
  let x_center := 5
  let y_center := 4
  (x_center, y_center)

theorem find_circle_center (x y : ℝ) (h : x^2 - 10 * x + y^2 - 8 * y = 16) :
  circle_center = (5, 4) := by
  sorry

end find_circle_center_l282_282802


namespace arithmetic_mean_location_l282_282316

theorem arithmetic_mean_location (a b : ℝ) : 
    abs ((a + b) / 2 - a) = abs (b - (a + b) / 2) := 
by 
    sorry

end arithmetic_mean_location_l282_282316


namespace algebraic_expression_simplification_l282_282431

theorem algebraic_expression_simplification (k x : ℝ) (h : (x - k * x) * (2 * x - k * x) - 3 * x * (2 * x - k * x) = 5 * x^2) :
  k = 3 ∨ k = -3 :=
by {
  sorry
}

end algebraic_expression_simplification_l282_282431


namespace double_even_l282_282583

-- Define even function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Lean statement of the mathematically equivalent proof problem
theorem double_even (f : ℝ → ℝ) (h : is_even_function f) : is_even_function (f ∘ f) :=
by
  sorry

end double_even_l282_282583


namespace saucepan_capacity_l282_282288

-- Define the conditions
variable (x : ℝ)
variable (h : 0.28 * x = 35)

-- State the theorem
theorem saucepan_capacity : x = 125 :=
by
  sorry

end saucepan_capacity_l282_282288


namespace unit_digit_of_expression_l282_282145

theorem unit_digit_of_expression :
  let expr := (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)
  (expr - 1) % 10 = 4 :=
by
  let expr := (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)
  sorry

end unit_digit_of_expression_l282_282145


namespace sum_of_cubes_divisible_by_nine_l282_282756

theorem sum_of_cubes_divisible_by_nine (n : ℕ) (h : 0 < n) : 9 ∣ (n^3 + (n + 1)^3 + (n + 2)^3) :=
by sorry

end sum_of_cubes_divisible_by_nine_l282_282756


namespace total_cost_l282_282298

theorem total_cost :
  ∀ (cost_caramel cost_candy cost_cotton : ℕ),
  (cost_candy = 2 * cost_caramel) →
  (cost_cotton = (1 / 2) * (4 * cost_candy)) →
  (cost_caramel = 3) →
  6 * cost_candy + 3 * cost_caramel + cost_cotton = 57 :=
begin
  intro cost_caramel, intro cost_candy, intro cost_cotton,
  assume h1 : cost_candy = 2 * cost_caramel,
  assume h2 : cost_cotton = (1 / 2) * (4 * cost_candy),
  assume h3 : cost_caramel = 3,
  sorry
end

end total_cost_l282_282298


namespace sum_of_digits_base8_888_l282_282169

def base8_representation (n : Nat) : List Nat :=
  let rec helper (n : Nat) (acc : List Nat) : List Nat :=
    if n == 0 then acc else helper (n / 8) ((n % 8) :: acc)
  helper n []

def sum_of_list (l : List Nat) : Nat :=
  l.foldl (· + ·) 0

theorem sum_of_digits_base8_888 : 
  sum_of_list (base8_representation 888) = 13 := 
  by
    sorry

end sum_of_digits_base8_888_l282_282169


namespace math_problem_l282_282852

theorem math_problem (n a b : ℕ) (hn_pos : n > 0) (h1 : 3 * n + 1 = a^2) (h2 : 5 * n - 1 = b^2) :
  (∃ x y: ℕ, 7 * n + 13 = x * y ∧ 1 < x ∧ 1 < y) ∧
  (∃ p q: ℕ, 8 * (17 * n^2 + 3 * n) = p^2 + q^2) :=
  sorry

end math_problem_l282_282852


namespace min_rows_for_students_l282_282474

def min_rows (total_students seats_per_row max_students_per_school : ℕ) : ℕ :=
  total_students / seats_per_row + if total_students % seats_per_row == 0 then 0 else 1

theorem min_rows_for_students :
  ∀ (total_students seats_per_row max_students_per_school : ℕ),
  (total_students = 2016) →
  (seats_per_row = 168) →
  (max_students_per_school = 40) →
  min_rows total_students seats_per_row max_students_per_school = 15 :=
by
  intros total_students seats_per_row max_students_per_school h1 h2 h3
  -- We write down the proof outline to show that 15 is the required minimum
  sorry

end min_rows_for_students_l282_282474


namespace stock_return_to_original_l282_282698

theorem stock_return_to_original (x : ℝ) (h : x > 0) :
  ∃ d : ℝ, d = 3 / 13 ∧ (x * 1.30 * (1 - d)) = x :=
by sorry

end stock_return_to_original_l282_282698


namespace largest_y_coordinate_l282_282365

theorem largest_y_coordinate (x y : ℝ) (h : x^2 / 49 + (y - 3)^2 / 25 = 0) : y = 3 :=
sorry

end largest_y_coordinate_l282_282365


namespace Jack_can_form_rectangle_l282_282110

theorem Jack_can_form_rectangle : 
  ∃ (a b : ℕ), 
  3 * a = 2016 ∧ 
  4 * a = 2016 ∧ 
  4 * b = 2016 ∧ 
  3 * b = 2016 ∧ 
  (503 * 4 + 3 * 9 = 2021) ∧ 
  (2 * 3 = 4) :=
by 
  sorry

end Jack_can_form_rectangle_l282_282110


namespace determine_n_l282_282938

theorem determine_n (n : ℕ) (h : n ≥ 2)
    (condition : ∀ i j : ℕ, i ≤ n → j ≤ n → (i + j) % 2 = (Nat.choose n i + Nat.choose n j) % 2) :
    ∃ k : ℕ, k ≥ 2 ∧ n = 2^k - 2 := 
sorry

end determine_n_l282_282938


namespace remainder_when_divided_by_7_l282_282326

theorem remainder_when_divided_by_7
  (x : ℤ) (k : ℤ) (h : x = 52 * k + 19) : x % 7 = 5 :=
sorry

end remainder_when_divided_by_7_l282_282326


namespace find_M_l282_282052

theorem find_M (M : ℤ) (h1 : 22 < M) (h2 : M < 24) : M = 23 := by
  sorry

end find_M_l282_282052


namespace find_polynomials_l282_282816

noncomputable theory

open Polynomial

theorem find_polynomials {
  (n : ℕ) (a : Fin (n + 1) → ℤ) (f : Polynomial ℤ) 
  (roots_real : ∀ z ∈ (f.roots.map (algebraMap ℚ ℝ)).to_finset, z.im = 0)
  (coeff_one_neg_one : ∀ i, a i = 1 ∨ a i = -1)
}
  (h_f : f = ∑ i in Finset.range (n + 1), (C (a i) * X^i)) :
  (n = 1 ∧ (f = X - 1 ∨ f = -X + 1 ∨ f = X + 1 ∨ f = -X - 1))
  ∨ (n = 2 ∧ (f = X^2 + X - 1 ∨ f = -X^2 - X + 1 ∨ f = X^2 - X - 1 ∨ f = -X^2 + X + 1))
  ∨ (n = 3 ∧ (f = X^3 + X^2 - X - 1 ∨ f = -X^3 - X^2 + X + 1 ∨ f = X^3 - X^2 - X + 1 ∨ f = -X^3 + X^2 + X - 1)) :=
sorry

end find_polynomials_l282_282816


namespace diameter_outer_boundary_correct_l282_282045

noncomputable def diameter_outer_boundary 
  (D_fountain : ℝ)
  (w_gardenRing : ℝ)
  (w_innerPath : ℝ)
  (w_outerPath : ℝ) : ℝ :=
  let R_fountain := D_fountain / 2
  let R_innerPath := R_fountain + w_gardenRing
  let R_outerPathInner := R_innerPath + w_innerPath
  let R_outerPathOuter := R_outerPathInner + w_outerPath
  2 * R_outerPathOuter

theorem diameter_outer_boundary_correct :
  diameter_outer_boundary 10 12 3 4 = 48 := by
  -- skipping proof
  sorry

end diameter_outer_boundary_correct_l282_282045


namespace sum_consecutive_equals_prime_l282_282879

theorem sum_consecutive_equals_prime (m k p : ℕ) (h_prime : Nat.Prime p) :
  (∃ S, S = (m * (2 * k + m - 1)) / 2 ∧ S = p) →
  m = 1 ∨ m = 2 :=
sorry

end sum_consecutive_equals_prime_l282_282879


namespace difference_cubics_divisible_by_24_l282_282369

theorem difference_cubics_divisible_by_24 
    (a b : ℤ) (h : ∃ k : ℤ, a - b = 3 * k) : 
    ∃ k : ℤ, (2 * a + 1)^3 - (2 * b + 1)^3 = 24 * k :=
by
  sorry

end difference_cubics_divisible_by_24_l282_282369


namespace zoe_total_songs_l282_282761

-- Define the number of country albums Zoe bought
def country_albums : Nat := 3

-- Define the number of pop albums Zoe bought
def pop_albums : Nat := 5

-- Define the number of songs per album
def songs_per_album : Nat := 3

-- Define the total number of albums
def total_albums : Nat := country_albums + pop_albums

-- Define the total number of songs
def total_songs : Nat := total_albums * songs_per_album

-- Theorem statement asserting the total number of songs
theorem zoe_total_songs : total_songs = 24 := by
  -- Proof will be inserted here (currently skipped)
  sorry

end zoe_total_songs_l282_282761


namespace point_on_circle_l282_282258

noncomputable def distance_from_origin (x : ℝ) (y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

theorem point_on_circle : distance_from_origin (-3) 4 = 5 := by
  sorry

end point_on_circle_l282_282258


namespace probability_all_co_captains_l282_282617

-- Define the number of students in each team
def students_team1 : ℕ := 4
def students_team2 : ℕ := 6
def students_team3 : ℕ := 7
def students_team4 : ℕ := 9

-- Define the probability of selecting each team
def prob_selecting_team : ℚ := 1 / 4

-- Define the probability of selecting three co-captains from each team
def prob_team1 : ℚ := 1 / Nat.choose students_team1 3
def prob_team2 : ℚ := 1 / Nat.choose students_team2 3
def prob_team3 : ℚ := 1 / Nat.choose students_team3 3
def prob_team4 : ℚ := 1 / Nat.choose students_team4 3

-- Define the total probability
def total_prob : ℚ :=
  prob_selecting_team * (prob_team1 + prob_team2 + prob_team3 + prob_team4)

theorem probability_all_co_captains :
  total_prob = 59 / 1680 := by
  sorry

end probability_all_co_captains_l282_282617


namespace mobot_coloring_six_colorings_l282_282537

theorem mobot_coloring_six_colorings (n m : ℕ) (h : n ≥ 3 ∧ m ≥ 3) :
  (∃ mobot, mobot = (1, 1)) ↔ (∃ colorings : ℕ, colorings = 6) :=
sorry

end mobot_coloring_six_colorings_l282_282537


namespace g_inv_g_inv_14_l282_282864

def g (x : ℝ) : ℝ := 5 * x - 3

noncomputable def g_inv (y : ℝ) : ℝ := (y + 3) / 5

theorem g_inv_g_inv_14 : g_inv (g_inv 14) = 32 / 25 :=
by
  sorry

end g_inv_g_inv_14_l282_282864


namespace jessica_probability_at_least_two_correct_l282_282264

open Set
open Function
open Probability

noncomputable def jessica_quiz_probability : ℚ :=
  let p_wrong := (2 / 3 : ℚ)
  let p_correct := (1 / 3 : ℚ)
  let p0 := (p_wrong ^ 6) -- Probability of getting exactly 0 correct
  let p1 := (6 * p_correct * (p_wrong ^ 5)) -- Probability of getting exactly 1 correct
  1 - (p0 + p1)

theorem jessica_probability_at_least_two_correct :
  jessica_quiz_probability = 473 / 729 := by
  sorry

end jessica_probability_at_least_two_correct_l282_282264


namespace l_shape_area_l282_282343

theorem l_shape_area (large_length large_width small_length small_width : ℕ)
  (large_rect_area : large_length = 10 ∧ large_width = 7)
  (small_rect_area : small_length = 3 ∧ small_width = 2) :
  (large_length * large_width) - 2 * (small_length * small_width) = 58 :=
by 
  sorry

end l_shape_area_l282_282343


namespace range_of_a_l282_282243

variable (a : ℝ)

def proposition_p (a : ℝ) : Prop := 0 < a ∧ a < 1

def proposition_q (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 - x + a > 0 ∧ 1 - 4 * a^2 < 0

theorem range_of_a : (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) →
  (0 < a ∧ a ≤ 1/2 ∨ a ≥ 1) := 
by
  sorry

end range_of_a_l282_282243


namespace pm_star_eq_6_l282_282521

open Set

-- Definitions based on the conditions
def universal_set : Set ℕ := univ
def M : Set ℕ := {1, 2, 3, 4, 5}
def P : Set ℕ := {2, 3, 6}
def star (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- The theorem to prove
theorem pm_star_eq_6 : star P M = {6} :=
sorry

end pm_star_eq_6_l282_282521


namespace inequaliy_pos_real_abc_l282_282721

theorem inequaliy_pos_real_abc (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_abc : a * b * c = 1) : 
  (a / (a * b + 1)) + (b / (b * c + 1)) + (c / (c * a + 1)) ≥ (3 / 2) := 
by
  sorry

end inequaliy_pos_real_abc_l282_282721


namespace intersection_distance_zero_l282_282293

noncomputable def A : Type := ℝ × ℝ

def P : A := (2, 0)

def line_intersects_parabola (x y : ℝ) : Prop :=
  y - 2 * x + 5 = 0 ∧ y^2 = 3 * x + 4

def distance (p1 p2 : A) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem intersection_distance_zero :
  ∀ (A1 A2 : A),
  line_intersects_parabola A1.1 A1.2 ∧ line_intersects_parabola A2.1 A2.2 →
  (abs (distance A1 P - distance A2 P) = 0) :=
sorry

end intersection_distance_zero_l282_282293


namespace halfway_miles_proof_l282_282163

def groceries_miles : ℕ := 10
def haircut_miles : ℕ := 15
def doctor_miles : ℕ := 5

def total_miles : ℕ := groceries_miles + haircut_miles + doctor_miles

theorem halfway_miles_proof : total_miles / 2 = 15 := by
  -- calculation to follow
  sorry

end halfway_miles_proof_l282_282163


namespace dimensions_multiple_of_three_l282_282193

theorem dimensions_multiple_of_three (a b c : ℤ) (h : a * b * c = (a + 1) * (b + 1) * (c - 2)) :
  (a % 3 = 0) ∨ (b % 3 = 0) ∨ (c % 3 = 0) :=
sorry

end dimensions_multiple_of_three_l282_282193


namespace real_root_solution_l282_282379

theorem real_root_solution (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  ∃ x1 x2 : ℝ, 
    (x1 < b ∧ b < x2) ∧
    (1 / (x1 - a) + 1 / (x1 - b) + 1 / (x1 - c) = 0) ∧ 
    (1 / (x2 - a) + 1 / (x2 - b) + 1 / (x2 - c) = 0) :=
by
  sorry

end real_root_solution_l282_282379


namespace tan_alpha_sin_double_angle_l282_282675

theorem tan_alpha_sin_double_angle (α : ℝ) (h : Real.tan α = 3/4) : Real.sin (2 * α) = 24/25 :=
by
  sorry

end tan_alpha_sin_double_angle_l282_282675


namespace trigonometric_identity_l282_282963

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) : 2 * Real.sin θ + Real.sin θ * Real.cos θ = 2 := by
  sorry

end trigonometric_identity_l282_282963


namespace number_of_juniors_l282_282406

variable (J S x y : ℕ)

-- Conditions given in the problem
axiom total_students : J + S = 40
axiom junior_debate_team : 3 * J / 10 = x
axiom senior_debate_team : S / 5 = y
axiom equal_debate_team : x = y

-- The theorem to prove 
theorem number_of_juniors : J = 16 :=
by
  sorry

end number_of_juniors_l282_282406


namespace calculate_change_l282_282341

theorem calculate_change : 
  let bracelet_cost := 15
  let necklace_cost := 10
  let mug_cost := 20
  let num_bracelets := 3
  let num_necklaces := 2
  let num_mugs := 1
  let discount := 0.10
  let total_cost := (num_bracelets * bracelet_cost) + (num_necklaces * necklace_cost) + (num_mugs * mug_cost)
  let discount_amount := total_cost * discount
  let final_amount := total_cost - discount_amount
  let payment := 100
  let change := payment - final_amount
  change = 23.50 :=
by
  -- Intentionally skipping the proof
  sorry

end calculate_change_l282_282341


namespace range_of_c_l282_282076

def P (c : ℝ) : Prop := ∀ x1 x2 : ℝ, x1 < x2 → (c ^ x1) > (c ^ x2)
def q (c : ℝ) : Prop := ∀ x : ℝ, x > (1 / 2) → (2 * c * x - c) > 0

theorem range_of_c (c : ℝ) (h1 : c > 0) (h2 : c ≠ 1)
  (h3 : ¬ (P c ∧ q c)) (h4 : (P c ∨ q c)) :
  (1 / 2) < c ∧ c < 1 :=
by
  sorry

end range_of_c_l282_282076


namespace inverse_proportional_fraction_l282_282509

theorem inverse_proportional_fraction (N : ℝ) (d f : ℝ) (h : N ≠ 0):
  d * f = N :=
sorry

end inverse_proportional_fraction_l282_282509


namespace second_coloring_book_pictures_l282_282447

theorem second_coloring_book_pictures (P1 P2 P_colored P_left : ℕ) (h1 : P1 = 23) (h2 : P_colored = 44) (h3 : P_left = 11) (h4 : P1 + P2 = P_colored + P_left) :
  P2 = 32 :=
by
  rw [h1, h2, h3] at h4
  linarith

end second_coloring_book_pictures_l282_282447


namespace domain_is_all_real_l282_282211

-- Definitions and conditions
def quadratic_expression (x : ℝ) : ℝ := x^2 - 8 * x + 18

def domain_of_f (x : ℝ) : Prop := ∃ (y : ℝ), y = 1 / (⌊quadratic_expression x⌋)

-- Theorem statement
theorem domain_is_all_real : ∀ x : ℝ, domain_of_f x :=
by
  sorry

end domain_is_all_real_l282_282211


namespace amplitude_of_cosine_function_is_3_l282_282205

variable (a b : ℝ)
variable (h_a : a > 0)
variable (h_b : b > 0)
variable (h_max : ∀ x : ℝ, a * Real.cos (b * x) ≤ 3)
variable (h_cycle : ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ (∀ x : ℝ, a * Real.cos (b * (x + 2 * Real.pi)) = a * Real.cos (b * x)))

theorem amplitude_of_cosine_function_is_3 :
  a = 3 :=
sorry

end amplitude_of_cosine_function_is_3_l282_282205


namespace product_of_two_numbers_l282_282748

theorem product_of_two_numbers (x y : ℝ) (h₁ : x + y = 23) (h₂ : x^2 + y^2 = 289) : x * y = 120 := by
  sorry

end product_of_two_numbers_l282_282748


namespace gcd_168_54_264_l282_282670

theorem gcd_168_54_264 : Nat.gcd (Nat.gcd 168 54) 264 = 6 :=
by
  -- proof goes here and ends with sorry for now
  sorry

end gcd_168_54_264_l282_282670


namespace price_of_sundae_l282_282487

variable (num_ice_cream_bars num_sundaes : ℕ)
variable (total_price : ℚ)
variable (price_per_ice_cream_bar : ℚ)
variable (price_per_sundae : ℚ)

theorem price_of_sundae :
  num_ice_cream_bars = 125 →
  num_sundaes = 125 →
  total_price = 225 →
  price_per_ice_cream_bar = 0.60 →
  price_per_sundae = (total_price - (num_ice_cream_bars * price_per_ice_cream_bar)) / num_sundaes →
  price_per_sundae = 1.20 :=
by
  intros
  sorry

end price_of_sundae_l282_282487


namespace trajectory_of_moving_circle_l282_282499

noncomputable def circle_eq_form (x y k: ℝ) := x^2 + y^2 + k = 0

theorem trajectory_of_moving_circle :
  let C : ℝ → ℝ → Prop := λ x y, circle_eq_form (x - 3) y 1
  let O : ℝ → ℝ → Prop := λ x y, circle_eq_form x y 1
  let r : ℝ
  MO (x y : ℝ) (M : ℝ×ℝ) := (M.fst - x)^2 + (M.snd - y)^2 = (r + 1)^2
  MC (x y : ℝ) (M : ℝ×ℝ) := (M.fst - x)^2 + (M.snd - y)^2 = (r - 1)^2
  in ∃ M : ℝ × ℝ, (MO 0 0 M → MC 3 0 M → (trajectory M = Hyperbola)) :=
sorry

end trajectory_of_moving_circle_l282_282499


namespace prob_two_out_of_three_A_prob_at_least_one_A_and_B_l282_282766

open ProbabilityTheory

-- Define the probability of buses arriving on time
def prob_bus_A : ℝ := 0.7
def prob_bus_B : ℝ := 0.75

-- 1. Probability that exactly two out of three tourists taking bus A arrive on time
theorem prob_two_out_of_three_A : 
  (C 3 2) * (prob_bus_A ^ 2) * ((1 - prob_bus_A) ^ 1) = 0.441 :=
sorry

-- 2. Probability that at least one out of two tourists, one taking bus A, and the other taking bus B, arrives on time
theorem prob_at_least_one_A_and_B :
  1 - ((1 - prob_bus_A) * (1 - prob_bus_B)) = 0.925 :=
sorry

end prob_two_out_of_three_A_prob_at_least_one_A_and_B_l282_282766


namespace part1_tangent_line_max_min_values_l282_282239

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x
def tangent_line_at (a : ℝ) (x y : ℝ) : ℝ := 9 * x + y - 4

theorem part1 (a : ℝ) : f' a 1 = -9 → a = -6 :=
by
  sorry

theorem tangent_line (a : ℝ) (x y : ℝ) : a = -6 → f a 1 = -5 → tangent_line_at a 1 (-5) = 0 :=
by
  sorry

def interval := Set.Icc (-5 : ℝ) 5

theorem max_min_values (a : ℝ) : a = -6 →
  (∀ x ∈ interval, f a (-5) = -275 ∨ f a 0 = 0 ∨ f a 4 = -32 ∨ f a 5 = -25) →
  (∀ x ∈ interval, f a x ≤ 0 ∧ f a x ≥ -275) :=
by
  sorry

end part1_tangent_line_max_min_values_l282_282239


namespace arithmetic_square_root_16_l282_282738

theorem arithmetic_square_root_16 : ∃ (x : ℝ), x * x = 16 ∧ x ≥ 0 ∧ x = 4 := by
  sorry

end arithmetic_square_root_16_l282_282738


namespace tina_earned_more_l282_282436

def candy_bar_problem_statement : Prop :=
  let type_a_price := 2
  let type_b_price := 3
  let marvin_type_a_sold := 20
  let marvin_type_b_sold := 15
  let tina_type_a_sold := 70
  let tina_type_b_sold := 35
  let marvin_discount_per_5_type_a := 1
  let tina_discount_per_10_type_b := 2
  let tina_returns_type_b := 2
  let marvin_total_earnings := 
    (marvin_type_a_sold * type_a_price) + 
    (marvin_type_b_sold * type_b_price) -
    (marvin_type_a_sold / 5 * marvin_discount_per_5_type_a)
  let tina_total_earnings := 
    (tina_type_a_sold * type_a_price) + 
    (tina_type_b_sold * type_b_price) -
    (tina_type_b_sold / 10 * tina_discount_per_10_type_b) -
    (tina_returns_type_b * type_b_price)
  let difference := tina_total_earnings - marvin_total_earnings
  difference = 152

theorem tina_earned_more :
  candy_bar_problem_statement :=
by
  sorry

end tina_earned_more_l282_282436


namespace smallest_b_l282_282455

theorem smallest_b (a b : ℕ) (pos_a : 0 < a) (pos_b : 0 < b)
    (h1 : a - b = 4)
    (h2 : gcd ((a^3 + b^3) / (a + b)) (a * b) = 4) : b = 2 :=
sorry

end smallest_b_l282_282455


namespace inequality_a4b_to_abcd_l282_282073

theorem inequality_a4b_to_abcd (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a^4 * b + b^4 * c + c^4 * d + d^4 * a ≥ a * b * c * d * (a + b + c + d) :=
by
  sorry

end inequality_a4b_to_abcd_l282_282073


namespace find_2023rd_letter_l282_282627

def seq : List Char := ['A', 'B', 'C', 'D', 'D', 'C', 'B', 'A']

theorem find_2023rd_letter : seq.get! ((2023 % seq.length) - 1) = 'B' :=
by
  sorry

end find_2023rd_letter_l282_282627


namespace angle_ADB_is_45_degrees_l282_282261

open Real EuclideanGeometry

def convex_pentagon (A B C D E : Point) : Prop :=
  ConvexPolygon A B C D E ∧
  ∠A B C = 90 ∧
  ∠B C D = 90 ∧
  ∠D A E = 90 ∧
  Inscribable A B C D E

theorem angle_ADB_is_45_degrees 
{A B C D E : Point} 
(h_convex : convex_pentagon A B C D E) 
: ∠A D B = 45 := 
sorry

end angle_ADB_is_45_degrees_l282_282261


namespace quadratic_inequality_solution_l282_282809

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 50 * x + 601 ≤ 9} = {x : ℝ | 19.25545 ≤ x ∧ x ≤ 30.74455} :=
by 
  sorry

end quadratic_inequality_solution_l282_282809


namespace triangle_overlap_angle_is_30_l282_282623

noncomputable def triangle_rotation_angle (hypotenuse : ℝ) (overlap_ratio : ℝ) :=
  if hypotenuse = 10 ∧ overlap_ratio = 0.5 then 30 else sorry

theorem triangle_overlap_angle_is_30 :
  triangle_rotation_angle 10 0.5 = 30 :=
sorry

end triangle_overlap_angle_is_30_l282_282623


namespace minimum_bus_door_height_l282_282139

-- Definitions based on the problem conditions
def normal_distribution_height : Real → Real → Real → Real := sorry  -- Placeholder for the PDF of the normal distribution

def mu : Real := 170  -- mean
def sigma : Real := 7  -- standard deviation

-- Given probabilities
axiom prob_mu_minus_sigma_to_mu_plus_sigma : 0.6826 = 
  (normal_distribution_height mu sigma (mu - sigma)) -
  (normal_distribution_height mu sigma (mu + sigma))

axiom prob_mu_minus_2sigma_to_mu_plus_2sigma : 0.9544 = 
  (normal_distribution_height mu sigma (mu - 2 * sigma)) -
  (normal_distribution_height mu sigma (mu + 2 * sigma))

axiom prob_mu_minus_3sigma_to_mu_plus_3sigma : 0.9974 =
  (normal_distribution_height mu sigma (mu - 3 * sigma)) -
  (normal_distribution_height mu sigma (mu + 3 * sigma))

-- Prove the required height for the bus doors
theorem minimum_bus_door_height : ∃ h : Real, h = 184 ∧ 
  ((1 - (normal_distribution_height mu sigma h)) ≤ 0.0228) :=
by
  sorry

end minimum_bus_door_height_l282_282139


namespace sum_consecutive_not_power_of_two_l282_282208

theorem sum_consecutive_not_power_of_two :
  ∀ n k : ℕ, ∀ x : ℕ, n > 0 → k > 0 → (n * (n + 2 * k - 1)) / 2 ≠ 2 ^ x := by
  sorry

end sum_consecutive_not_power_of_two_l282_282208


namespace geometric_seq_neither_necess_nor_suff_l282_282284

theorem geometric_seq_neither_necess_nor_suff (a_1 q : ℝ) (h₁ : a_1 ≠ 0) (h₂ : q ≠ 0) :
  ¬ (∀ n : ℕ, (a_1 * q > 0 → a_1 * q ^ n < a_1 * q ^ (n + 1)) ∧ (∀ n : ℕ, (a_1 * q ^ n < a_1 * q ^ (n + 1)) → a_1 * q > 0)) :=
by
  sorry

end geometric_seq_neither_necess_nor_suff_l282_282284


namespace greatest_two_digit_with_product_9_l282_282629

theorem greatest_two_digit_with_product_9 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ a b : ℕ, n = 10 * a + b ∧ a * b = 9) ∧ (∀ m : ℕ, 10 ≤ m ∧ m < 100 ∧ (∃ c d : ℕ, m = 10 * c + d ∧ c * d = 9) → m ≤ 91) :=
by
  sorry

end greatest_two_digit_with_product_9_l282_282629


namespace rational_x_of_rational_x3_and_x2_add_x_l282_282424

variable {x : ℝ}

theorem rational_x_of_rational_x3_and_x2_add_x (hx3 : ∃ a : ℚ, x^3 = a)
  (hx2_add_x : ∃ b : ℚ, x^2 + x = b) : ∃ r : ℚ, x = r :=
sorry

end rational_x_of_rational_x3_and_x2_add_x_l282_282424


namespace total_games_played_l282_282346

-- Define the number of teams
def num_teams : ℕ := 12

-- Define the number of games each team plays with each other team
def games_per_pair : ℕ := 4

-- The theorem stating the total number of games played
theorem total_games_played : num_teams * (num_teams - 1) / 2 * games_per_pair = 264 :=
by
  sorry

end total_games_played_l282_282346


namespace james_units_per_semester_l282_282714

theorem james_units_per_semester
  (cost_per_unit : ℕ)
  (total_cost : ℕ)
  (num_semesters : ℕ)
  (payment_per_semester : ℕ)
  (units_per_semester : ℕ)
  (H1 : cost_per_unit = 50)
  (H2 : total_cost = 2000)
  (H3 : num_semesters = 2)
  (H4 : payment_per_semester = total_cost / num_semesters)
  (H5 : units_per_semester = payment_per_semester / cost_per_unit) :
  units_per_semester = 20 :=
sorry

end james_units_per_semester_l282_282714


namespace binary_predecessor_l282_282691

theorem binary_predecessor (N : ℕ) (hN : N = 0b11000) : 0b10111 + 1 = N := 
by
  sorry

end binary_predecessor_l282_282691


namespace glucose_solution_volume_l282_282191

theorem glucose_solution_volume
  (h1 : 6.75 / 45 = 15 / x) :
  x = 100 :=
by
  sorry

end glucose_solution_volume_l282_282191


namespace relationship_between_a_b_c_l282_282070

theorem relationship_between_a_b_c (a b c : ℕ) (h1 : a = 2^40) (h2 : b = 3^32) (h3 : c = 4^24) : a < c ∧ c < b := by
  -- Definitions as per conditions
  have ha : a = 32^8 := by sorry
  have hb : b = 81^8 := by sorry
  have hc : c = 64^8 := by sorry
  -- Comparisons involving the bases
  have h : 32 < 64 := by sorry
  have h' : 64 < 81 := by sorry
  -- Resultant comparison
  exact ⟨by sorry, by sorry⟩

end relationship_between_a_b_c_l282_282070


namespace wrapping_paper_each_present_l282_282005

theorem wrapping_paper_each_present (total_paper : ℚ) (num_presents : ℕ)
  (h1 : total_paper = 1 / 2) (h2 : num_presents = 5) :
  (total_paper / num_presents = 1 / 10) :=
by
  sorry

end wrapping_paper_each_present_l282_282005


namespace simplify_expression_l282_282674

-- Definitions of intermediate calculations
def a : ℤ := 3 + 5 + 6 - 2
def b : ℚ := a * 2 / 4
def c : ℤ := 3 * 4 + 6 - 4
def d : ℚ := c / 3

-- The statement to be proved
theorem simplify_expression : b + d = 32 / 3 := by
  sorry

end simplify_expression_l282_282674


namespace gcd_lcm_1365_910_l282_282803

theorem gcd_lcm_1365_910 :
  gcd 1365 910 = 455 ∧ lcm 1365 910 = 2730 :=
by
  sorry

end gcd_lcm_1365_910_l282_282803


namespace symmetry_of_g_function_l282_282940

def g (x : ℝ) : ℝ := |⌊x⌋| - |⌊2 - x⌋|

theorem symmetry_of_g_function : ∀ x : ℝ, g(x) = g(2 - x) :=
by
  sorry

end symmetry_of_g_function_l282_282940


namespace second_hand_angle_after_2_minutes_l282_282333

theorem second_hand_angle_after_2_minutes :
  ∀ angle_in_radians, (∀ rotations:ℝ, rotations = 2 → one_full_circle = 2 * Real.pi → angle_in_radians = - (rotations * one_full_circle)) →
  angle_in_radians = -4 * Real.pi :=
by
  intros
  sorry

end second_hand_angle_after_2_minutes_l282_282333


namespace trigonometric_identity_l282_282430

theorem trigonometric_identity (x y z : ℝ)
  (h1 : Real.cos x + Real.cos y + Real.cos z = 1)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 1) :
  Real.cos (2 * x) + Real.cos (2 * y) + 2 * Real.cos (2 * z) = 2 :=
by
  sorry

end trigonometric_identity_l282_282430


namespace determine_function_l282_282213

open Real

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (1/2) * (x^2 + (1/x)) else 0

theorem determine_function (f: ℝ → ℝ) (h : ∀ x ≠ 0, (1/x) * f (-x) + f (1/x) = x ) :
  ∀ x ≠ 0, f x = (1/2) * (x^2 + (1/x)) :=
by
  sorry

end determine_function_l282_282213


namespace mixed_doubles_pairing_l282_282750

theorem mixed_doubles_pairing: 
  let males := 5
  let females := 4
  let choose_males := Nat.choose males 2
  let choose_females := Nat.choose females 2
  let arrangements := Nat.factorial 2
  choose_males * choose_females * arrangements = 120 := by
  sorry

end mixed_doubles_pairing_l282_282750


namespace ratio_of_p_q_l282_282624

theorem ratio_of_p_q (b : ℝ) (p q : ℝ) (h1 : p = -b / 8) (h2 : q = -b / 12) : p / q = 3 / 2 := 
by
  sorry

end ratio_of_p_q_l282_282624


namespace tank_plastering_cost_l282_282783

noncomputable def plastering_cost (L W D : ℕ) (cost_per_sq_meter : ℚ) : ℚ :=
  let A_bottom := L * W
  let A_long_walls := 2 * (L * D)
  let A_short_walls := 2 * (W * D)
  let A_total := A_bottom + A_long_walls + A_short_walls
  A_total * cost_per_sq_meter

theorem tank_plastering_cost :
  plastering_cost 25 12 6 0.25 = 186 := by
  sorry

end tank_plastering_cost_l282_282783


namespace final_position_l282_282060

structure Position where
  base : ℝ × ℝ
  stem : ℝ × ℝ

def rotate180 (pos : Position) : Position :=
  { base := (-pos.base.1, -pos.base.2),
    stem := (-pos.stem.1, -pos.stem.2) }

def reflectX (pos : Position) : Position :=
  { base := (pos.base.1, -pos.base.2),
    stem := (pos.stem.1, -pos.stem.2) }

def rotateHalfTurn (pos : Position) : Position :=
  { base := (-pos.base.1, -pos.base.2),
    stem := (-pos.stem.1, -pos.stem.2) }

def reflectY (pos : Position) : Position :=
  { base := (-pos.base.1, pos.base.2),
    stem := (-pos.stem.1, pos.stem.2) }

theorem final_position : 
  let initial_pos := Position.mk (1, 0) (0, 1)
  let pos1 := rotate180 initial_pos
  let pos2 := reflectX pos1
  let pos3 := rotateHalfTurn pos2
  let final_pos := reflectY pos3
  final_pos = { base := (-1, 0), stem := (0, -1) } :=
by
  sorry

end final_position_l282_282060


namespace find_principal_amount_l282_282702

theorem find_principal_amount
  (P r : ℝ) -- P for Principal amount, r for interest rate
  (simple_interest : 800 = P * r / 100 * 2) -- Condition 1: Simple Interest Formula
  (compound_interest : 820 = P * ((1 + r / 100) ^ 2 - 1)) -- Condition 2: Compound Interest Formula
  : P = 8000 := 
sorry

end find_principal_amount_l282_282702


namespace inequality_proof_l282_282383

theorem inequality_proof 
  (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (sum_eq_one : a + b + c + d = 1) :
  (a^2 / (1 + a)) + (b^2 / (1 + b)) + (c^2 / (1 + c)) + (d^2 / (1 + d)) ≥ 1/5 := 
by {
  sorry
}

end inequality_proof_l282_282383


namespace no_distinct_nat_numbers_eq_l282_282125

theorem no_distinct_nat_numbers_eq (x y z t : ℕ) (hxy : x ≠ y) (hxz : x ≠ z) (hxt : x ≠ t) 
  (hyz : y ≠ z) (hyt : y ≠ t) (hzt : z ≠ t) : x ^ x + y ^ y ≠ z ^ z + t ^ t := 
by 
  sorry

end no_distinct_nat_numbers_eq_l282_282125


namespace simplify_expression_l282_282130

theorem simplify_expression (a: ℤ) (h₁: a ≠ 0) (h₂: a ≠ 1) (h₃: a ≠ -3) :
  (2 * a = 4) → a = 2 :=
by
  sorry

end simplify_expression_l282_282130


namespace distance_travelled_downstream_l282_282768

def speed_boat_still_water : ℕ := 24
def speed_stream : ℕ := 4
def time_downstream : ℕ := 6

def effective_speed_downstream : ℕ := speed_boat_still_water + speed_stream
def distance_downstream : ℕ := effective_speed_downstream * time_downstream

theorem distance_travelled_downstream : distance_downstream = 168 := by
  sorry

end distance_travelled_downstream_l282_282768


namespace coffee_cost_per_week_l282_282711

theorem coffee_cost_per_week 
  (number_people : ℕ) 
  (cups_per_person_per_day : ℕ) 
  (ounces_per_cup : ℝ) 
  (cost_per_ounce : ℝ) 
  (total_cost_per_week : ℝ) 
  (h₁ : number_people = 4)
  (h₂ : cups_per_person_per_day = 2)
  (h₃ : ounces_per_cup = 0.5)
  (h₄ : cost_per_ounce = 1.25)
  (h₅ : total_cost_per_week = 35) : 
  number_people * cups_per_person_per_day * ounces_per_cup * cost_per_ounce * 7 = total_cost_per_week :=
by
  sorry

end coffee_cost_per_week_l282_282711


namespace total_saplings_l282_282726

theorem total_saplings (a_efficiency b_efficiency : ℝ) (A B T n : ℝ) 
  (h1 : a_efficiency = (3/4))
  (h2 : b_efficiency = 1)
  (h3 : B = n + 36)
  (h4 : T = 2 * n + 36)
  (h5 : n * (4/3) = n + 36)
  : T = 252 :=
by {
  sorry
}

end total_saplings_l282_282726


namespace batman_game_cost_l282_282162

theorem batman_game_cost (total_spent superman_cost : ℝ) 
  (H1 : total_spent = 18.66) (H2 : superman_cost = 5.06) :
  total_spent - superman_cost = 13.60 :=
by
  sorry

end batman_game_cost_l282_282162


namespace smallest_number_is_D_l282_282924

-- Define the given numbers in Lean
def A := 25
def B := 111
def C := 16 + 4 + 2  -- since 10110_{(2)} equals 22 in base 10
def D := 16 + 2 + 1  -- since 10011_{(2)} equals 19 in base 10

-- The Lean statement for the proof problem
theorem smallest_number_is_D : min (min A B) (min C D) = D := by
  sorry

end smallest_number_is_D_l282_282924


namespace calculate_V3_at_2_l282_282513

def polynomial (x : ℕ) : ℕ :=
  (((((2 * x + 5) * x + 6) * x + 23) * x - 8) * x + 10) * x - 3

theorem calculate_V3_at_2 : polynomial 2 = 71 := by
  sorry

end calculate_V3_at_2_l282_282513


namespace common_factor_of_polynomial_l282_282132

theorem common_factor_of_polynomial :
  ∀ (x : ℝ), (2 * x^2 - 8 * x) = 2 * x * (x - 4) := by
  sorry

end common_factor_of_polynomial_l282_282132


namespace greatest_sum_x_y_l282_282464

theorem greatest_sum_x_y (x y : ℤ) (h : x^2 + y^2 = 36) : (x + y ≤ 9) := sorry

end greatest_sum_x_y_l282_282464


namespace inequality_div_l282_282694

theorem inequality_div (m n : ℝ) (h : m > n) : (m / 5) > (n / 5) :=
sorry

end inequality_div_l282_282694


namespace cube_of_odd_sum_l282_282693

theorem cube_of_odd_sum (a : ℕ) (h1 : 1 < a) (h2 : ∃ (n : ℕ), (n = (a - 1) + 2 * (a - 1) + 1) ∧ n = 1979) : a = 44 :=
sorry

end cube_of_odd_sum_l282_282693


namespace no_tangent_to_x_axis_max_integer_a_for_inequality_l282_282080

noncomputable def f (x a : ℝ) : ℝ := (x - 1) * Real.exp x - (a / 2) * x^2

theorem no_tangent_to_x_axis (a : ℝ) : ¬∃ t : ℝ, f t a = 0 ∧ (t * Real.exp t - a * t) = 0 := sorry

theorem max_integer_a_for_inequality : 
  (∃ a : ℤ, (∀ x1 x2 : ℝ, x2 > 0 → f (x1 + x2) a - f (x1 - x2) a > -2 * x2) ∧ 
             (∀ b : ℤ, b > a → ∃ x1 x2 : ℝ, x2 > 0 ∧ f (x1 + x2) b - f (x1 - x2) b ≤ -2 * x2)) ∧ a = ↑3 := sorry

end no_tangent_to_x_axis_max_integer_a_for_inequality_l282_282080


namespace Julia_watch_collection_l282_282421

section
variable (silver_watches : ℕ) (bronze_watches : ℕ) (gold_watches : ℕ) (total_watches : ℕ)

theorem Julia_watch_collection :
  silver_watches = 20 →
  bronze_watches = 3 * silver_watches →
  gold_watches = 10 * (silver_watches + bronze_watches) / 100 →
  total_watches = silver_watches + bronze_watches + gold_watches →
  total_watches = 88 :=
by
  intros
  sorry
end

end Julia_watch_collection_l282_282421


namespace find_c_value_l282_282138

theorem find_c_value (x1 y1 x2 y2 : ℝ) (h1 : x1 = 1) (h2 : y1 = 4) (h3 : x2 = 5) (h4 : y2 = 0) (c : ℝ)
  (h5 : 3 * ((x1 + x2) / 2) - 2 * ((y1 + y2) / 2) = c) : c = 5 :=
sorry

end find_c_value_l282_282138


namespace souvenirs_expenses_l282_282449

/--
  Given:
  1. K = T + 146.00
  2. T + K = 548.00
  Prove: 
  - K = 347.00
-/
theorem souvenirs_expenses (T K : ℝ) (h1 : K = T + 146) (h2 : T + K = 548) : K = 347 :=
  sorry

end souvenirs_expenses_l282_282449


namespace quadrilateral_not_parallelogram_l282_282688

-- Definitions based on the given conditions
structure Quadrilateral :=
  (a b c d : ℝ) -- sides of the quadrilateral
  (parallel : Prop) -- one pair of parallel sides
  (equal_sides : Prop) -- another pair of equal sides

-- Problem statement
theorem quadrilateral_not_parallelogram (q : Quadrilateral) 
  (h1 : q.parallel) 
  (h2 : q.equal_sides) : 
  ¬ (∃ p : Quadrilateral, p = q) :=
sorry

end quadrilateral_not_parallelogram_l282_282688


namespace root_expression_value_l282_282093

theorem root_expression_value 
  (m : ℝ) 
  (h : 2 * m^2 - 3 * m - 1 = 0) : 
  6 * m^2 - 9 * m + 2021 = 2024 := 
by 
  sorry

end root_expression_value_l282_282093


namespace divide_composite_products_l282_282063

theorem divide_composite_products :
  let first_three := [4, 6, 8]
  let next_three := [9, 10, 12]
  let prod_first_three := first_three.prod
  let prod_next_three := next_three.prod
  (prod_first_three : ℚ) / prod_next_three = 8 / 45 :=
by
  sorry

end divide_composite_products_l282_282063


namespace jed_change_l282_282577

theorem jed_change :
  ∀ (num_games : ℕ) (cost_per_game : ℕ) (payment : ℕ) (bill_value : ℕ),
  num_games = 6 →
  cost_per_game = 15 →
  payment = 100 →
  bill_value = 5 →
  (payment - num_games * cost_per_game) / bill_value = 2 :=
by
  intros num_games cost_per_game payment bill_value
  sorry

end jed_change_l282_282577


namespace shortest_routes_l282_282359

theorem shortest_routes
  (side_length : ℝ)
  (refuel_distance : ℝ)
  (total_distance : ℝ)
  (shortest_paths : ℕ) :
  side_length = 10 ∧
  refuel_distance = 30 ∧
  total_distance = 180 →
  shortest_paths = 18 :=
sorry

end shortest_routes_l282_282359


namespace quadratic_function_expression_quadratic_function_inequality_l282_282546

noncomputable def f (x : ℝ) : ℝ := x^2 - x + 1

theorem quadratic_function_expression (a b c : ℝ) (h₀ : a ≠ 0) 
  (h₁ : ∀ x : ℝ, f (x + 1) - f x = 2 * x) 
  (h₂ : f 0 = 1) : 
  (f x = x^2 - x + 1) := 
by {
  sorry
}

theorem quadratic_function_inequality (m : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x > 2 * x + m) ↔ m < -1 := 
by {
  sorry
}

end quadratic_function_expression_quadratic_function_inequality_l282_282546


namespace percentage_boys_from_school_A_is_20_l282_282403

-- Definitions and conditions based on the problem
def total_boys : ℕ := 200
def non_science_boys_from_A : ℕ := 28
def science_ratio : ℝ := 0.30
def non_science_ratio : ℝ := 1 - science_ratio

-- To prove: The percentage of the total boys that are from school A is 20%
theorem percentage_boys_from_school_A_is_20 :
  ∃ (x : ℝ), x = 20 ∧ 
  (non_science_ratio * (x / 100 * total_boys) = non_science_boys_from_A) := 
sorry

end percentage_boys_from_school_A_is_20_l282_282403


namespace solve_for_x_l282_282397

theorem solve_for_x (x y : ℝ) (h1 : y = 1 / (5 * x + 2)) (h2 : y = 2) : x = -3 / 10 :=
by
  sorry

end solve_for_x_l282_282397


namespace differential_savings_l282_282488

def original_tax_rate : ℝ := 0.45
def new_tax_rate : ℝ := 0.30
def annual_income : ℝ := 48000

theorem differential_savings : (original_tax_rate * annual_income) - (new_tax_rate * annual_income) = 7200 := by
  sorry

end differential_savings_l282_282488


namespace find_packs_of_yellow_bouncy_balls_l282_282414

noncomputable def packs_of_yellow_bouncy_balls (red_packs : ℕ) (balls_per_pack : ℕ) (extra_balls : ℕ) : ℕ :=
  (red_packs * balls_per_pack - extra_balls) / balls_per_pack

theorem find_packs_of_yellow_bouncy_balls :
  packs_of_yellow_bouncy_balls 5 18 18 = 4 := 
by
  sorry

end find_packs_of_yellow_bouncy_balls_l282_282414


namespace smallest_fraction_greater_than_4_over_5_l282_282356

theorem smallest_fraction_greater_than_4_over_5 :
  ∃ (b : ℕ), 10 ≤ b ∧ b < 100 ∧ 77 * 5 > 4 * b ∧ Int.gcd 77 b = 1 ∧
  ∀ (a : ℕ), 10 ≤ a ∧ a < 77 → ¬ ∃ (b' : ℕ), 10 ≤ b' ∧ b' < 100 ∧ a * 5 > 4 * b' ∧ Int.gcd a b' = 1 := by
  sorry

end smallest_fraction_greater_than_4_over_5_l282_282356


namespace chess_tournament_l282_282042

theorem chess_tournament :
  ∀ (n : ℕ), (∃ (players : ℕ) (total_games : ℕ),
  players = 8 ∧ total_games = 56 ∧ total_games = (players * (players - 1) * n) / 2) →
  n = 2 :=
by
  intros n h
  rcases h with ⟨players, total_games, h_players, h_total_games, h_eq⟩
  have := h_eq
  sorry

end chess_tournament_l282_282042


namespace invisible_trees_in_square_l282_282873

theorem invisible_trees_in_square (n : ℕ) : 
  ∃ (N M : ℕ), ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → 
  Nat.gcd (N + i) (M + j) ≠ 1 :=
by
  sorry

end invisible_trees_in_square_l282_282873


namespace find_a_l282_282971

-- Define the sets A and B based on the conditions
def A (a : ℝ) : Set ℝ := {a ^ 2, a + 1, -3}
def B (a : ℝ) : Set ℝ := {a - 3, a ^ 2 + 1, 2 * a - 1}

-- Statement: Prove that a = -1 satisfies the condition A ∩ B = {-3}
theorem find_a (a : ℝ) (h : A a ∩ B a = {-3}) : a = -1 :=
by
  sorry

end find_a_l282_282971


namespace quadratic_real_solutions_l282_282099

theorem quadratic_real_solutions (m : ℝ) :
  (∃ (x : ℝ), m * x^2 + 2 * x + 1 = 0) ↔ (m ≤ 1 ∧ m ≠ 0) :=
by
  sorry

end quadratic_real_solutions_l282_282099


namespace contrapositive_abc_l282_282868

theorem contrapositive_abc (a b c : ℝ) : (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) → (abc ≠ 0) := 
sorry

end contrapositive_abc_l282_282868


namespace probability_red_chips_drawn_first_l282_282651

def probability_all_red_drawn (total_chips : Nat) (red_chips : Nat) (green_chips : Nat) : ℚ :=
  let total_arrangements := Nat.choose total_chips green_chips
  let favorable_arrangements := Nat.choose (total_chips - 1) (green_chips - 1)
  favorable_arrangements / total_arrangements

theorem probability_red_chips_drawn_first :
  probability_all_red_drawn 9 5 4 = 4 / 9 :=
by
  sorry

end probability_red_chips_drawn_first_l282_282651


namespace ellipse_chord_equation_l282_282096

theorem ellipse_chord_equation 
  (chord_bisected : (∃ A B : ℝ × ℝ, ((A.1^2 / 36) + (A.2^2 / 9) = 1 ∧ (B.1^2 / 36) + (B.2^2 / 9) = 1) ∧ (A.1 + B.1) / 2 = 4 ∧ (A.2 + B.2) / 2 = 2)) : 
  ∃ (a b c : ℝ), a * 4 + b * 2 + c = 0 ∧  a = 1 ∧ b = 2 ∧ c = -8 :=
by
  sorry

end ellipse_chord_equation_l282_282096


namespace michael_twice_jacob_in_11_years_l282_282709

-- Definitions
def jacob_age_4_years := 5
def jacob_current_age := jacob_age_4_years - 4
def michael_current_age := jacob_current_age + 12

-- Theorem to prove
theorem michael_twice_jacob_in_11_years :
  ∀ (x : ℕ), jacob_current_age + x = 1 →
    michael_current_age + x = 13 →
    michael_current_age + (11 : ℕ) = 2 * (jacob_current_age + (11 : ℕ)) :=
by
  intros x h1 h2
  sorry

end michael_twice_jacob_in_11_years_l282_282709


namespace prob_divisible_by_5_l282_282306

theorem prob_divisible_by_5 (M: ℕ) (h1: 100 ≤ M ∧ M < 1000) (h2: M % 10 = 5): 
  (∃ (k: ℕ), M = 5 * k) :=
by
  sorry

end prob_divisible_by_5_l282_282306


namespace three_digit_integer_divisible_by_5_l282_282304

theorem three_digit_integer_divisible_by_5 (M : ℕ) (h1 : 100 ≤ M ∧ M < 1000) (h2 : M % 10 = 5) : M % 5 = 0 := 
sorry

end three_digit_integer_divisible_by_5_l282_282304


namespace smallest_prime_factor_of_5_pow_5_minus_5_pow_3_l282_282033

theorem smallest_prime_factor_of_5_pow_5_minus_5_pow_3 : Nat.Prime 2 ∧ (∀ p : ℕ, Nat.Prime p ∧ p ∣ (5^5 - 5^3) → p ≥ 2) := by
  sorry

end smallest_prime_factor_of_5_pow_5_minus_5_pow_3_l282_282033


namespace steven_falls_correct_l282_282603

/-
  We will model the problem where we are given the conditions about the falls of Steven, Stephanie,
  and Sonya, and then prove that the number of times Steven fell is 3.
-/

variables (S : ℕ) -- Steven's falls

-- Conditions
def stephanie_falls := S + 13
def sonya_falls := 6 
def sonya_condition := 6 = (stephanie_falls / 2) - 2

-- Theorem statement
theorem steven_falls_correct : S = 3 :=
by {
  -- Note: the actual proof steps would go here, but are omitted per instructions
  sorry
}

end steven_falls_correct_l282_282603


namespace union_A_B_l282_282827

def setA : Set ℝ := {x | (x + 1) * (x - 2) < 0}
def setB : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem union_A_B : setA ∪ setB = {x | -1 < x ∧ x ≤ 3} :=
by
  sorry

end union_A_B_l282_282827


namespace total_reptiles_l282_282142

theorem total_reptiles 
  (reptiles_in_s1 : ℕ := 523)
  (reptiles_in_s2 : ℕ := 689)
  (reptiles_in_s3 : ℕ := 784)
  (reptiles_in_s4 : ℕ := 392)
  (reptiles_in_s5 : ℕ := 563)
  (reptiles_in_s6 : ℕ := 842) :
  reptiles_in_s1 + reptiles_in_s2 + reptiles_in_s3 + reptiles_in_s4 + reptiles_in_s5 + reptiles_in_s6 = 3793 :=
by
  sorry

end total_reptiles_l282_282142


namespace abc_divisibility_l282_282800

theorem abc_divisibility (a b c : ℕ) (h1 : a^2 * b ∣ a^3 + b^3 + c^3) (h2 : b^2 * c ∣ a^3 + b^3 + c^3) (h3 : c^2 * a ∣ a^3 + b^3 + c^3) :
  a = 1 ∧ b = 1 ∧ c = 1 :=
sorry

end abc_divisibility_l282_282800


namespace range_of_expression_l282_282267

noncomputable def expression (a b c d : ℝ) : ℝ :=
  Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt (b^2 + (2 - c)^2) + 
  Real.sqrt (c^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2)

theorem range_of_expression (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 2)
  (h3 : 0 ≤ b) (h4 : b ≤ 2) (h5 : 0 ≤ c) (h6 : c ≤ 2)
  (h7 : 0 ≤ d) (h8 : d ≤ 2) :
  4 * Real.sqrt 2 ≤ expression a b c d ∧ expression a b c d ≤ 16 :=
by
  sorry

end range_of_expression_l282_282267


namespace infinite_corners_have_subset_l282_282850

open Finset

noncomputable def is_corner {α : Type*} [linear_order α] (n : ℕ) (S : Finset (Vector α n)) : Prop :=
∀ (a b : Vector α n), a ∈ S → (∀ i, a.nth i ≥ b.nth i) → b ∈ S

theorem infinite_corners_have_subset (n : ℕ) (C : Set (Finset (Vector ℕ n)))
  (h_inf : C.infinite) (h_corner : ∀ (S ∈ C), is_corner n S) :
  ∃ (A B ∈ C), A ⊆ B :=
sorry

end infinite_corners_have_subset_l282_282850


namespace cos_A_equals_one_third_l282_282988

-- Noncomputable context as trigonometric functions are involved.
noncomputable def cosA_in_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  let law_of_cosines : (a * Real.cos B) = (3 * c - b) * Real.cos A := sorry
  (Real.cos A = 1 / 3)

-- Define the problem statement to be proved
theorem cos_A_equals_one_third (a b c A B C : ℝ) 
  (h1 : a = Real.cos B)
  (h2 : a * Real.cos B = (3 * c - b) * Real.cos A) :
  Real.cos A = 1 / 3 := 
by 
  -- Placeholder for the actual proof
  sorry

end cos_A_equals_one_third_l282_282988


namespace exists_monotonic_subsequence_l282_282960

open Function -- For function related definitions
open Finset -- For finite set operations

-- Defining the theorem with the given conditions and the goal to be proved
theorem exists_monotonic_subsequence (a : Fin 10 → ℝ) (h : ∀ i j : Fin 10, i ≠ j → a i ≠ a j) :
  ∃ (i1 i2 i3 i4 : Fin 10), i1 < i2 ∧ i2 < i3 ∧ i3 < i4 ∧
  ((a i1 < a i2 ∧ a i2 < a i3 ∧ a i3 < a i4) ∨ (a i1 > a i2 ∧ a i2 > a i3 ∧ a i3 > a i4)) :=
by
  sorry -- Proof is omitted as per the instructions

end exists_monotonic_subsequence_l282_282960


namespace net_pay_is_correct_l282_282566

-- Define the gross pay and taxes paid as constants
def gross_pay : ℕ := 450
def taxes_paid : ℕ := 135

-- Define net pay as a function of gross pay and taxes paid
def net_pay (gross : ℕ) (taxes : ℕ) : ℕ := gross - taxes

-- The proof statement
theorem net_pay_is_correct : net_pay gross_pay taxes_paid = 315 := by
  sorry -- The proof goes here

end net_pay_is_correct_l282_282566


namespace fred_final_baseball_cards_l282_282810

-- Conditions
def initial_cards : ℕ := 25
def sold_to_melanie : ℕ := 7
def traded_with_kevin : ℕ := 3
def bought_from_alex : ℕ := 5

-- Proof statement (Lean theorem)
theorem fred_final_baseball_cards : initial_cards - sold_to_melanie - traded_with_kevin + bought_from_alex = 20 := by
  sorry

end fred_final_baseball_cards_l282_282810


namespace coffee_cost_per_week_l282_282713

theorem coffee_cost_per_week 
  (people_in_house : ℕ) 
  (drinks_per_person_per_day : ℕ) 
  (ounces_per_cup : ℝ) 
  (cost_per_ounce : ℝ) 
  (num_days_in_week : ℕ) 
  (h1 : people_in_house = 4) 
  (h2 : drinks_per_person_per_day = 2)
  (h3 : ounces_per_cup = 0.5)
  (h4 : cost_per_ounce = 1.25)
  (h5 : num_days_in_week = 7) :
  people_in_house * drinks_per_person_per_day * ounces_per_cup * cost_per_ounce * num_days_in_week = 35 := 
by
  sorry

end coffee_cost_per_week_l282_282713


namespace keith_and_jason_books_l282_282423

theorem keith_and_jason_books :
  let K := 20
  let J := 21
  K + J = 41 :=
by
  sorry

end keith_and_jason_books_l282_282423


namespace problem_solution_l282_282548

-- Definitions based on conditions
def p (a b : ℝ) : Prop := a > b → a^2 > b^2
def neg_p (a b : ℝ) : Prop := a > b → a^2 ≤ b^2
def disjunction (p q : Prop) : Prop := p ∨ q
def suff_but_not_nec (x : ℝ) : Prop := x > 2 → x > 1 ∧ ¬(x > 1 → x > 2)
def congruent_triangles (T1 T2 : Prop) : Prop := T1 → T2
def neg_congruent_triangles (T1 T2 : Prop) : Prop := ¬(T1 → T2)

-- Mathematical problem as Lean statements
theorem problem_solution :
  ( (∀ a b : ℝ, p a b = (a > b → a^2 > b^2) ∧ neg_p a b = (a > b → a^2 ≤ b^2)) ∧
    (∀ p q : Prop, (disjunction p q) = false → p = false ∧ q = false) ∧
    (∀ x : ℝ, suff_but_not_nec x = (x > 2 → x > 1 ∧ ¬(x > 1 → x > 2))) ∧
    (∀ T1 T2 : Prop, (neg_congruent_triangles T1 T2) = true ↔ ¬(T1 → T2)) ) →
  ( (∀ a b : ℝ, neg_p a b = (a > b → a^2 ≤ b^2)) ∧
    (∀ p q : Prop, (disjunction p q) = false → p = false ∧ q = false) ∧
    (∀ x : ℝ, suff_but_not_nec x = (x > 2 → x > 1 ∧ ¬(x > 1 → x > 2))) ∧
    (∀ T1 T2 : Prop, (neg_congruent_triangles T1 T2) = false) ) :=
sorry

end problem_solution_l282_282548


namespace range_of_a_l282_282840

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (1 - a) * x + 1 < 0) → a ∈ Set.Icc (-1 : ℝ) 3 :=
by
  sorry

end range_of_a_l282_282840


namespace fraction_halfway_between_l282_282291

theorem fraction_halfway_between : 
  ∃ (x : ℚ), (x = (1 / 6 + 1 / 4) / 2) ∧ x = 5 / 24 :=
by
  sorry

end fraction_halfway_between_l282_282291


namespace solved_distance_l282_282990

variable (D : ℝ) 

-- Time for A to cover the distance
variable (tA : ℝ) (tB : ℝ)
variable (dA : ℝ) (dB : ℝ := D - 26)

-- A covers the distance in 36 seconds
axiom hA : tA = 36

-- B covers the distance in 45 seconds
axiom hB : tB = 45

-- A beats B by 26 meters implies B covers (D - 26) in the time A covers D
axiom h_diff : dB = dA - 26

theorem solved_distance :
  D = 130 := 
by 
  sorry

end solved_distance_l282_282990


namespace sufficient_conditions_for_sum_positive_l282_282552

variable {a b : ℝ}

theorem sufficient_conditions_for_sum_positive (h₃ : a + b > 2) (h₄ : a > 0 ∧ b > 0) : a + b > 0 :=
by {
  sorry
}

end sufficient_conditions_for_sum_positive_l282_282552


namespace marys_age_l282_282276

variable (M R : ℕ) -- Define M (Mary's current age) and R (Rahul's current age) as natural numbers

theorem marys_age
  (h1 : R = M + 40)       -- Rahul is 40 years older than Mary
  (h2 : R + 30 = 3 * (M + 30))  -- In 30 years, Rahul will be three times as old as Mary
  : M = 20 := 
sorry  -- The proof goes here

end marys_age_l282_282276


namespace fifteenth_term_arithmetic_sequence_l282_282616

theorem fifteenth_term_arithmetic_sequence (a d : ℤ) : 
  (a + 20 * d = 17) ∧ (a + 21 * d = 20) → (a + 14 * d = -1) := by
  sorry

end fifteenth_term_arithmetic_sequence_l282_282616


namespace minimum_rows_required_l282_282468

theorem minimum_rows_required
  (seats_per_row : ℕ)
  (total_students : ℕ)
  (max_students_per_school : ℕ)
  (H1 : seats_per_row = 168)
  (H2 : total_students = 2016)
  (H3 : max_students_per_school = 40)
  : ∃ n : ℕ, n = 15 ∧ (∀ configuration : List (List ℕ), configuration.length = n ∧ 
       (∀ school_students, school_students ∈ configuration → school_students.length ≤ seats_per_row) ∧
       ∀ i, ∃ (c : ℕ) (school_students : ℕ), school_students ≤ max_students_per_school ∧
         i < total_students - ∑ configuration.head! length → 
         true) :=
sorry

end minimum_rows_required_l282_282468


namespace find_num_biology_books_l282_282026

-- Given conditions
def num_chemistry_books : ℕ := 8
def total_ways_to_pick : ℕ := 2548

-- Function to calculate combinations
def combination (n k : ℕ) := n.choose k

-- Statement to be proved
theorem find_num_biology_books (B : ℕ) (h1 : combination num_chemistry_books 2 = 28) 
  (h2 : combination B 2 * 28 = total_ways_to_pick) : B = 14 :=
by 
  -- Proof goes here
  sorry

end find_num_biology_books_l282_282026


namespace count_primes_1021_eq_one_l282_282538

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_primes_1021_eq_one :
  (∃ n : ℕ, 3 ≤ n ∧ is_prime (n^3 + 2*n + 1) ∧
  ∀ m : ℕ, (3 ≤ m ∧ m ≠ n) → ¬ is_prime (m^3 + 2*m + 1)) :=
sorry

end count_primes_1021_eq_one_l282_282538


namespace total_supervisors_l282_282016

theorem total_supervisors (buses : ℕ) (supervisors_per_bus : ℕ) (h1 : buses = 7) (h2 : supervisors_per_bus = 3) :
  buses * supervisors_per_bus = 21 :=
by
  sorry

end total_supervisors_l282_282016


namespace calculate_ab_plus_cd_l282_282851

theorem calculate_ab_plus_cd (a b c d : ℝ) 
  (h1 : a + b + c = 5)
  (h2 : a + b + d = -1)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = 12) :
  a * b + c * d = 27 :=
by
  sorry -- Proof to be filled in.

end calculate_ab_plus_cd_l282_282851


namespace bucket_p_fill_time_l282_282658

theorem bucket_p_fill_time (capacity_P capacity_Q drum_capacity turns : ℕ)
  (h1 : capacity_P = 3 * capacity_Q)
  (h2 : drum_capacity = 45 * (capacity_P + capacity_Q))
  (h3 : bucket_fill_turns = drum_capacity / capacity_P) :
  bucket_fill_turns = 60 :=
by
  sorry

end bucket_p_fill_time_l282_282658


namespace waiter_tables_l282_282350

theorem waiter_tables (init_customers : ℕ) (left_customers : ℕ) (people_per_table : ℕ) (remaining_customers : ℕ) (num_tables : ℕ) :
  init_customers = 44 →
  left_customers = 12 →
  people_per_table = 8 →
  remaining_customers = init_customers - left_customers →
  num_tables = remaining_customers / people_per_table →
  num_tables = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end waiter_tables_l282_282350


namespace product_of_two_numbers_l282_282607

theorem product_of_two_numbers
  (x y : ℝ)
  (h1 : x - y = 12)
  (h2 : x^2 + y^2 = 106) :
  x * y = 32 := by 
  sorry

end product_of_two_numbers_l282_282607


namespace sequence_problem_l282_282682

noncomputable def exists_integers (a : ℕ → ℕ) (hbij : Function.Bijective a) : Prop :=
  ∃ (ℓ m : ℕ), 1 < ℓ ∧ ℓ < m ∧ a 1 + a m = 2 * a ℓ

theorem sequence_problem
  (a : ℕ → ℕ) (hbij : Function.Bijective a) : exists_integers a hbij :=
sorry

end sequence_problem_l282_282682


namespace percentage_taxed_on_excess_income_l282_282989

noncomputable def pct_taxed_on_first_40k : ℝ := 0.11
noncomputable def first_40k_income : ℝ := 40000
noncomputable def total_income : ℝ := 58000
noncomputable def total_tax : ℝ := 8000

theorem percentage_taxed_on_excess_income :
  ∃ P : ℝ, (total_tax - pct_taxed_on_first_40k * first_40k_income = P * (total_income - first_40k_income)) ∧ P * 100 = 20 := 
by
  sorry

end percentage_taxed_on_excess_income_l282_282989


namespace total_sacks_after_6_days_l282_282830

-- Define the conditions
def sacks_per_day : ℕ := 83
def days : ℕ := 6

-- Prove the total number of sacks after 6 days is 498
theorem total_sacks_after_6_days : sacks_per_day * days = 498 := by
  -- Proof Content Placeholder
  sorry

end total_sacks_after_6_days_l282_282830


namespace earnings_from_roosters_l282_282944

-- Definitions from the conditions
def price_per_kg : Float := 0.50
def weight_of_rooster1 : Float := 30.0
def weight_of_rooster2 : Float := 40.0

-- The theorem we need to prove (mathematically equivalent proof problem)
theorem earnings_from_roosters (p : Float := price_per_kg)
                               (w1 : Float := weight_of_rooster1)
                               (w2 : Float := weight_of_rooster2) :
  p * w1 + p * w2 = 35.0 := 
by {
  sorry
}

end earnings_from_roosters_l282_282944


namespace parking_fines_l282_282502

theorem parking_fines (total_citations littering_citations offleash_dog_citations parking_fines : ℕ) 
  (h1 : total_citations = 24) 
  (h2 : littering_citations = 4) 
  (h3 : offleash_dog_citations = 4) 
  (h4 : total_citations = littering_citations + offleash_dog_citations + parking_fines) : 
  parking_fines = 16 := 
by 
  sorry

end parking_fines_l282_282502


namespace cost_of_four_books_l282_282165

theorem cost_of_four_books
  (H : 2 * book_cost = 36) :
  4 * book_cost = 72 :=
by
  sorry

end cost_of_four_books_l282_282165


namespace jack_sugar_usage_l282_282993

theorem jack_sugar_usage (initial_sugar bought_sugar final_sugar x : ℕ) 
  (h1 : initial_sugar = 65) 
  (h2 : bought_sugar = 50) 
  (h3 : final_sugar = 97) 
  (h4 : final_sugar = initial_sugar - x + bought_sugar) : 
  x = 18 := 
by 
  sorry

end jack_sugar_usage_l282_282993


namespace sum_of_cube_faces_l282_282212

theorem sum_of_cube_faces :
  ∃ (a b c d e f : ℕ), 
    (a = 12) ∧ 
    (b = a + 3) ∧ 
    (c = b + 3) ∧ 
    (d = c + 3) ∧ 
    (e = d + 3) ∧ 
    (f = e + 3) ∧ 
    (a + f = 39) ∧ 
    (b + e = 39) ∧ 
    (c + d = 39) ∧ 
    (a + b + c + d + e + f = 117) :=
by
  let a := 12
  let b := a + 3
  let c := b + 3
  let d := c + 3
  let e := d + 3
  let f := e + 3
  have h1 : a + f = 39 := sorry
  have h2 : b + e = 39 := sorry
  have h3 : c + d = 39 := sorry
  have sum : a + b + c + d + e + f = 117 := sorry
  exact ⟨a, b, c, d, e, f, rfl, rfl, rfl, rfl, rfl, rfl, h1, h2, h3, sum⟩

end sum_of_cube_faces_l282_282212


namespace find_x2_plus_y2_l282_282950

theorem find_x2_plus_y2 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : x * y + x + y = 90) 
  (h2 : x^2 * y + x * y^2 = 1122) : 
  x^2 + y^2 = 1044 :=
sorry

end find_x2_plus_y2_l282_282950


namespace three_digit_numbers_not_multiples_of_3_or_11_l282_282561

def count_multiples (a b : ℕ) (lower upper : ℕ) : ℕ :=
  (upper - lower) / b + 1

theorem three_digit_numbers_not_multiples_of_3_or_11 : 
  let total := 900
  let multiples_3 := count_multiples 3 3 102 999
  let multiples_11 := count_multiples 11 11 110 990
  let multiples_33 := count_multiples 33 33 132 990
  let multiples_3_or_11 := multiples_3 + multiples_11 - multiples_33
  total - multiples_3_or_11 = 546 := 
by 
  sorry

end three_digit_numbers_not_multiples_of_3_or_11_l282_282561


namespace roots_exist_for_all_K_l282_282673

theorem roots_exist_for_all_K (K : ℝ) : ∃ x : ℝ, x = K^3 * (x - 1) * (x - 3) :=
by
  -- Applied conditions and approach
  sorry

end roots_exist_for_all_K_l282_282673


namespace keychain_arrangement_count_l282_282575

-- Definitions of the keys
inductive Key
| house
| car
| office
| other1
| other2

-- Function to count the number of distinct arrangements on a keychain
noncomputable def distinct_keychain_arrangements : ℕ :=
  sorry -- This will be the placeholder for the proof

-- The ultimate theorem stating the solution
theorem keychain_arrangement_count : distinct_keychain_arrangements = 2 :=
  sorry -- This will be the placeholder for the proof

end keychain_arrangement_count_l282_282575


namespace contrapositive_even_sum_l282_282133

theorem contrapositive_even_sum (a b : ℕ) :
  (¬(a % 2 = 0 ∧ b % 2 = 0) → ¬(a + b) % 2 = 0) ↔ (¬((a + b) % 2 = 0) → ¬(a % 2 = 0 ∧ b % 2 = 0)) :=
by
  sorry

end contrapositive_even_sum_l282_282133


namespace find_B_and_C_l282_282806

def values_of_B_and_C (B C : ℤ) : Prop :=
  5 * B - 3 = 32 ∧ 2 * B + 2 * C = 18

theorem find_B_and_C : ∃ B C : ℤ, values_of_B_and_C B C ∧ B = 7 ∧ C = 2 := by
  sorry

end find_B_and_C_l282_282806


namespace technicians_count_l282_282572

theorem technicians_count (
    workers_total : ℕ,
    avg_salary_all : ℕ,
    avg_salary_tech : ℕ,
    avg_salary_rest : ℕ,
    total_salary : ℕ,
    technicians : ℕ,
    rest : ℕ,
    h1 : workers_total = 28,
    h2 : avg_salary_all = 8000,
    h3 : avg_salary_tech = 14000,
    h4 : avg_salary_rest = 6000,
    h5 : total_salary = avg_salary_all * workers_total,
    h6 : total_salary = avg_salary_tech * technicians + avg_salary_rest * rest,
    h7 : technicians + rest = workers_total
  ) : technicians = 7 :=
by 
    sorry

end technicians_count_l282_282572


namespace total_spectators_after_halftime_l282_282790

theorem total_spectators_after_halftime
  (initial_boys : ℕ := 300)
  (initial_girls : ℕ := 400)
  (initial_adults : ℕ := 300)
  (total_people : ℕ := 1000)
  (quarter_boys_leave_fraction : ℚ := 1 / 4)
  (quarter_girls_leave_fraction : ℚ := 1 / 8)
  (quarter_adults_leave_fraction : ℚ := 1 / 5)
  (halftime_new_boys : ℕ := 50)
  (halftime_new_girls : ℕ := 90)
  (halftime_adults_leave_fraction : ℚ := 3 / 100) :
  let boys_after_first_quarter := initial_boys - initial_boys * quarter_boys_leave_fraction
  let girls_after_first_quarter := initial_girls - initial_girls * quarter_girls_leave_fraction
  let adults_after_first_quarter := initial_adults - initial_adults * quarter_adults_leave_fraction
  let boys_after_halftime := boys_after_first_quarter + halftime_new_boys
  let girls_after_halftime := girls_after_first_quarter + halftime_new_girls
  let adults_after_halftime := adults_after_first_quarter * (1 - halftime_adults_leave_fraction)
  boys_after_halftime + girls_after_halftime + adults_after_halftime = 948 :=
by sorry

end total_spectators_after_halftime_l282_282790


namespace reciprocal_geometric_sum_l282_282661

variable (n : ℕ) (r s : ℝ)
variable (h_r_nonzero : r ≠ 0)
variable (h_sum_original : (1 - r^(2 * n)) / (1 - r^2) = s^3)

theorem reciprocal_geometric_sum (n : ℕ) (r s : ℝ) (h_r_nonzero : r ≠ 0)
  (h_sum_original : (1 - r^(2 * n)) / (1 - r^2) = s^3) :
  ((1 - (1 / r^2)^n) / (1 - 1 / r^2)) = s^3 / r^2 :=
sorry

end reciprocal_geometric_sum_l282_282661


namespace two_digit_sum_divisible_by_17_l282_282407

theorem two_digit_sum_divisible_by_17 :
  ∃ A : ℕ, A ≥ 10 ∧ A < 100 ∧ ∃ B : ℕ, B = (A % 10) * 10 + (A / 10) ∧ (A + B) % 17 = 0 ↔ A = 89 ∨ A = 98 := 
sorry

end two_digit_sum_divisible_by_17_l282_282407


namespace forty_percent_of_number_l282_282181

theorem forty_percent_of_number (N : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 17) : 0.4 * N = 204 :=
sorry

end forty_percent_of_number_l282_282181


namespace largest_lcm_among_given_pairs_l282_282664

theorem largest_lcm_among_given_pairs : 
  max (max (max (max (max (Nat.lcm 18 3) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 12)) (Nat.lcm 18 15)) (Nat.lcm 18 18) = 90 :=
by
  sorry

end largest_lcm_among_given_pairs_l282_282664


namespace can_form_triangle_l282_282329

theorem can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

example : can_form_triangle 8 6 3 := by
  sorry

end can_form_triangle_l282_282329


namespace walking_rate_ratio_l282_282770

variables (R R' : ℝ)

theorem walking_rate_ratio (h₁ : R * 21 = R' * 18) : R' / R = 7 / 6 :=
by {
  sorry
}

end walking_rate_ratio_l282_282770


namespace correct_exponentiation_rule_l282_282328

theorem correct_exponentiation_rule (x y : ℝ) : ((x^2)^3 = x^6) :=
  by sorry

end correct_exponentiation_rule_l282_282328


namespace problem1_problem2_l282_282687

variable (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
def f (x : ℝ) : ℝ := abs (x - a) + 2 * abs (x + b)

theorem problem1 (h3 : ∃ x, f x = 1) : a + b = 1 := sorry

theorem problem2 (h4 : a + b = 1) (m : ℝ) (h5 : ∀ m, m ≤ 1/a + 2/b)
: m ≤ 3 + 2 * Real.sqrt 2 := sorry

end problem1_problem2_l282_282687


namespace prob_divisible_by_5_l282_282308

theorem prob_divisible_by_5 (M: ℕ) (h1: 100 ≤ M ∧ M < 1000) (h2: M % 10 = 5): 
  (∃ (k: ℕ), M = 5 * k) :=
by
  sorry

end prob_divisible_by_5_l282_282308


namespace wrapping_paper_area_correct_l282_282046

noncomputable def wrapping_paper_area (l w h : ℝ) (hlw : l ≥ w) : ℝ :=
  (l + 2*h)^2

theorem wrapping_paper_area_correct (l w h : ℝ) (hlw : l ≥ w) :
  wrapping_paper_area l w h hlw = (l + 2*h)^2 :=
by
  sorry

end wrapping_paper_area_correct_l282_282046


namespace base4_sum_correct_l282_282805

/-- Define the base-4 numbers as natural numbers. -/
def a := 3 * 4^2 + 1 * 4^1 + 2 * 4^0
def b := 3 * 4^1 + 1 * 4^0
def c := 3 * 4^0

/-- Define their sum in base 10. -/
def sum_base_10 := a + b + c

/-- Define the target sum in base 4 as a natural number. -/
def target := 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0

/-- Prove that the sum of the base-4 numbers equals the target sum in base 4. -/
theorem base4_sum_correct : sum_base_10 = target := by
  sorry

end base4_sum_correct_l282_282805


namespace balanced_number_example_l282_282650

/--
A number is balanced if it is a three-digit number, all digits are different,
and it equals the sum of all possible two-digit numbers composed from its different digits.
-/
def isBalanced (n : ℕ) : Prop :=
  (n / 100 ≠ (n / 10) % 10) ∧ (n / 100 ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10) ∧
  (n = (10 * (n / 100) + (n / 10) % 10) + (10 * (n / 100) + n % 10) +
    (10 * ((n / 10) % 10) + n / 100) + (10 * ((n / 10) % 10) + n % 10) +
    (10 * (n % 10) + n / 100) + (10 * (n % 10) + ((n / 10) % 10)))

theorem balanced_number_example : isBalanced 132 :=
  sorry

end balanced_number_example_l282_282650


namespace range_of_x_l282_282230

-- Defining the propositions p and q
def p (x : ℝ) : Prop := x^2 + 2 * x - 3 > 0
def q (x : ℝ) : Prop := 1 / (3 - x) > 1

-- Theorem statement
theorem range_of_x (x : ℝ) : (¬ q x ∧ p x) → (x ≥ 3 ∨ (1 < x ∧ x ≤ 2) ∨ x < -3) :=
by
  sorry

end range_of_x_l282_282230


namespace line_in_slope_intercept_form_l282_282775

-- Given the condition
def line_eq (x y : ℝ) : Prop :=
  (2 * (x - 3)) - (y + 4) = 0

-- Prove that the line equation can be expressed as y = 2x - 10.
theorem line_in_slope_intercept_form (x y : ℝ) :
  line_eq x y ↔ y = 2 * x - 10 := 
sorry

end line_in_slope_intercept_form_l282_282775


namespace arithmetic_seq_n_possible_values_l282_282843

theorem arithmetic_seq_n_possible_values
  (a1 : ℕ) (a_n : ℕ → ℕ) (d : ℕ) (n : ℕ):
  a1 = 1 → 
  (∀ n, n ≥ 3 → a_n n = 100) → 
  (∃ d : ℕ, ∀ n, n ≥ 3 → a_n n = a1 + (n - 1) * d) → 
  (n = 4 ∨ n = 10 ∨ n = 12 ∨ n = 34 ∨ n = 100) := by
  sorry

end arithmetic_seq_n_possible_values_l282_282843


namespace highest_possible_relocation_preference_l282_282438

theorem highest_possible_relocation_preference
  (total_employees : ℕ)
  (relocated_to_X_percent : ℝ)
  (relocated_to_Y_percent : ℝ)
  (prefer_X_percent : ℝ)
  (prefer_Y_percent : ℝ)
  (htotal : total_employees = 200)
  (hrelocated_to_X_percent : relocated_to_X_percent = 0.30)
  (hrelocated_to_Y_percent : relocated_to_Y_percent = 0.70)
  (hprefer_X_percent : prefer_X_percent = 0.60)
  (hprefer_Y_percent : prefer_Y_percent = 0.40) :
  ∃ (max_relocated_with_preference : ℕ), max_relocated_with_preference = 140 :=
by
  sorry

end highest_possible_relocation_preference_l282_282438


namespace range_of_values_for_m_l282_282722

theorem range_of_values_for_m (m : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 4| < m) → m > 1 :=
by
  sorry

end range_of_values_for_m_l282_282722


namespace ratio_e_a_l282_282247

theorem ratio_e_a (a b c d e : ℚ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 3)
  (h4 : d / e = 1 / 4) :
  e / a = 8 / 15 := 
by
  sorry

end ratio_e_a_l282_282247


namespace minimum_rows_required_l282_282470

theorem minimum_rows_required
  (seats_per_row : ℕ)
  (total_students : ℕ)
  (max_students_per_school : ℕ)
  (H1 : seats_per_row = 168)
  (H2 : total_students = 2016)
  (H3 : max_students_per_school = 40)
  : ∃ n : ℕ, n = 15 ∧ (∀ configuration : List (List ℕ), configuration.length = n ∧ 
       (∀ school_students, school_students ∈ configuration → school_students.length ≤ seats_per_row) ∧
       ∀ i, ∃ (c : ℕ) (school_students : ℕ), school_students ≤ max_students_per_school ∧
         i < total_students - ∑ configuration.head! length → 
         true) :=
sorry

end minimum_rows_required_l282_282470


namespace total_cost_of_items_l282_282297

-- Definitions based on conditions in a)
def price_of_caramel : ℕ := 3
def price_of_candy_bar : ℕ := 2 * price_of_caramel
def price_of_cotton_candy : ℕ := (4 * price_of_candy_bar) / 2
def cost_of_6_candy_bars : ℕ := 6 * price_of_candy_bar
def cost_of_3_caramels : ℕ := 3 * price_of_caramel

-- Problem statement to be proved
theorem total_cost_of_items : cost_of_6_candy_bars + cost_of_3_caramels + price_of_cotton_candy = 57 :=
by
  sorry

end total_cost_of_items_l282_282297


namespace cone_apex_angle_l282_282565

theorem cone_apex_angle (R : ℝ) 
  (h1 : ∀ (θ : ℝ), (∃ (r : ℝ), r = R / 2 ∧ 2 * π * r = π * R)) :
  ∀ (θ : ℝ), θ = π / 3 :=
by
  sorry

end cone_apex_angle_l282_282565


namespace min_value_inequality_l282_282545

theorem min_value_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 3^x + 9^y ≥ 2 * Real.sqrt 3 := 
by
  sorry

end min_value_inequality_l282_282545


namespace prob_divisible_by_5_l282_282307

theorem prob_divisible_by_5 (M: ℕ) (h1: 100 ≤ M ∧ M < 1000) (h2: M % 10 = 5): 
  (∃ (k: ℕ), M = 5 * k) :=
by
  sorry

end prob_divisible_by_5_l282_282307


namespace remainder_of_polynomial_l282_282375

   def polynomial_division_remainder (x : ℝ) : ℝ := x^4 - 4*x^2 + 7

   theorem remainder_of_polynomial : polynomial_division_remainder 1 = 4 :=
   by
     -- This placeholder indicates that the proof is omitted.
     sorry
   
end remainder_of_polynomial_l282_282375


namespace each_worker_paid_40_l282_282192

variable (n_orchids : ℕ) (price_per_orchid : ℕ)
variable (n_money_plants : ℕ) (price_per_money_plant : ℕ)
variable (new_pots_cost : ℕ) (leftover_money : ℕ)
variable (n_workers : ℕ)

noncomputable def total_earnings : ℤ :=
  n_orchids * price_per_orchid + n_money_plants * price_per_money_plant

noncomputable def total_spent : ℤ :=
  new_pots_cost + leftover_money

noncomputable def amount_paid_to_workers : ℤ :=
  total_earnings n_orchids price_per_orchid n_money_plants price_per_money_plant - 
  total_spent new_pots_cost leftover_money

noncomputable def amount_paid_to_each_worker : ℤ :=
  amount_paid_to_workers n_orchids price_per_orchid n_money_plants price_per_money_plant 
    new_pots_cost leftover_money / n_workers

theorem each_worker_paid_40 :
  amount_paid_to_each_worker 20 50 15 25 150 1145 2 = 40 := by
  sorry

end each_worker_paid_40_l282_282192


namespace blue_pieces_correct_l282_282883

def total_pieces : ℕ := 3409
def red_pieces : ℕ := 145
def blue_pieces : ℕ := total_pieces - red_pieces

theorem blue_pieces_correct : blue_pieces = 3264 := by
  sorry

end blue_pieces_correct_l282_282883


namespace smallest_m_for_triangle_sides_l282_282229

noncomputable def is_triangle_sides (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem smallest_m_for_triangle_sides (a b c : ℝ) (h : is_triangle_sides a b c) :
  (a^2 + c^2) / (b + c)^2 < 1 / 2 := sorry

end smallest_m_for_triangle_sides_l282_282229


namespace rhombus_area_outside_circle_l282_282644

theorem rhombus_area_outside_circle (d : ℝ) (r : ℝ) (h_d : d = 10) (h_r : r = 3) : 
  (d * d / 2 - 9 * Real.pi) > 9 :=
by
  sorry

end rhombus_area_outside_circle_l282_282644


namespace fill_in_the_blank_correct_option_l282_282030

-- Assume each option is defined
def options := ["the other", "some", "another", "other"]

-- Define a helper function to validate the correct option
def is_correct_option (opt: String) : Prop :=
  opt = "another"

-- The main problem statement
theorem fill_in_the_blank_correct_option :
  (∀ opt, opt ∈ options → is_correct_option opt → opt = "another") :=
by
  intro opt h_option h_correct
  simp [is_correct_option] at h_correct
  exact h_correct

-- Test case to check the correct option
example : is_correct_option "another" :=
by
  simp [is_correct_option]

end fill_in_the_blank_correct_option_l282_282030


namespace airplane_rows_l282_282054

theorem airplane_rows (R : ℕ) 
  (h1 : ∀ n, n = 5) 
  (h2 : ∀ s, s = 7) 
  (h3 : ∀ f, f = 2) 
  (h4 : ∀ p, p = 1400):
  (2 * 5 * 7 * R = 1400) → R = 20 :=
by
  -- Assuming the given equation 2 * 5 * 7 * R = 1400
  sorry

end airplane_rows_l282_282054


namespace min_distance_origin_to_line_l282_282078

theorem min_distance_origin_to_line (a b : ℝ) (h : a + 2 * b = Real.sqrt 5) : 
  Real.sqrt (a^2 + b^2) ≥ 1 :=
sorry

end min_distance_origin_to_line_l282_282078


namespace Jesse_read_pages_l282_282994

theorem Jesse_read_pages (total_pages : ℝ) (h : (2 / 3) * total_pages = 166) :
  (1 / 3) * total_pages = 83 :=
sorry

end Jesse_read_pages_l282_282994


namespace find_f11_l282_282820

-- Define the odd function properties
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define the functional equation property
def functional_eqn (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = -f x

-- Define the specific values of the function on (0,2)
def specific_values (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2

-- The main theorem that needs to be proved
theorem find_f11 (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : functional_eqn f) (h3 : specific_values f) : 
  f 11 = -2 :=
sorry

end find_f11_l282_282820


namespace halfway_between_one_sixth_and_one_twelfth_is_one_eighth_l282_282216

theorem halfway_between_one_sixth_and_one_twelfth_is_one_eighth : 
  (1 / 6 + 1 / 12) / 2 = 1 / 8 := 
by
  sorry

end halfway_between_one_sixth_and_one_twelfth_is_one_eighth_l282_282216


namespace Jori_water_left_l282_282112

theorem Jori_water_left (a b : ℚ) (h1 : a = 7/2) (h2 : b = 7/4) : a - b = 7/4 := by
  sorry

end Jori_water_left_l282_282112


namespace find_k_collinear_l282_282084

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-2, -1)
def c : ℝ × ℝ := (1, 2)

theorem find_k_collinear : ∃ k : ℝ, (1 - 2 * k, 3 - k) = (-k, k) * c ∧ k = -1/3 :=
by
  sorry

end find_k_collinear_l282_282084


namespace seating_arrangements_count_l282_282508

open Finset

-- Let A, B, C, D, and E represent Alice, Bob, Carla, Derek, and Eric respectively.
inductive Person
| A | B | C | D | E 
open Person

-- Function to check if a seating arrangement meets the constraints
def valid_seating (seating : Vector Person 5) : Prop :=
  (¬ (seating.anyp (= A 
   ∧ (seating.get! 0 = B ∨ seating.get! 1 = B ∨
       seating.get! 2 = D ∨ seating.get! 3 = D ∨ seating.get! 4 = D))))
  ∧ (¬ (seating.anyp (= C ∧ seating.anyp (= D ∧ (seating.get! 0 = D ∨ seating.get! 4 = D)))))

noncomputable def seating_arrangements : Finset (Vector Person 5) :=
  filter valid_seating (univ : Finset (Vector Person 5))

theorem seating_arrangements_count : seating_arrangements.card = 15 :=
sorry

end seating_arrangements_count_l282_282508


namespace number_of_multiples_143_l282_282553

theorem number_of_multiples_143
  (h1 : 143 = 11 * 13)
  (h2 : ∀ i j : ℕ, 10^j - 10^i = 10^i * (10^(j-i) - 1))
  (h3 : ∀ i : ℕ, gcd (10^i) 143 = 1)
  (h4 : ∀ k : ℕ, 143 ∣ 10^k - 1 ↔ k % 6 = 0)
  (h5 : ∀ i j : ℕ, 0 ≤ i ∧ i < j ∧ j ≤ 99)
  : ∃ n : ℕ, n = 784 :=
by
  sorry

end number_of_multiples_143_l282_282553


namespace quadratic_equal_roots_l282_282551

theorem quadratic_equal_roots (k : ℝ) :
  (∀ x : ℝ, x^2 + k * x + 1 = 0 → x = -k / 2) ↔ (k = 2 ∨ k = -2) :=
by
  sorry

end quadratic_equal_roots_l282_282551


namespace box_volume_l282_282289

theorem box_volume (x y z : ℕ) 
  (h1 : 2 * x + 2 * y = 26)
  (h2 : x + z = 10)
  (h3 : y + z = 7) :
  x * y * z = 80 :=
by
  sorry

end box_volume_l282_282289


namespace binomial_variance_l282_282841

theorem binomial_variance
  {X : Type}
  (n : ℕ)
  (h_binom : ∀ k : ℕ, Pr(X = k) = (n.choose k) * (1/3)^k * (2/3)^(n-k))
  (h_expect : E(X) = 5/3)
  (h_var_formula : D(X) = n * (1/3) * (1 - 1/3)) :
  D(X) = 10 / 9 :=
by
  sorry

end binomial_variance_l282_282841


namespace sum_of_ratios_eq_one_l282_282507

open Triangle Circle

variable {α β γ: ℝ}

-- Conditions
def triangle_inscribed_in_circle (A B C O: Point) (R: ℝ): Prop :=
  ∃ (C : Circle), C.radius = R ∧ C.contains A ∧ C.contains B ∧ C.contains C

def ratio_radii_circles_tangent (O: Point) (R: ℝ): Prop :=
  ∃ (α β γ < 1), true  -- Here, we assume α, β, and γ exist and are < 1

-- Main statement to prove
theorem sum_of_ratios_eq_one (A B C O: Point) (R: ℝ) (α β γ: ℝ)
  (h1: triangle_inscribed_in_circle A B C O R)
  (h2: ratio_radii_circles_tangent O R):
  α + β + γ = 1 :=
sorry

end sum_of_ratios_eq_one_l282_282507


namespace problem_solution_l282_282250

variable (α : ℝ)

/-- If $\sin\alpha = 2\cos\alpha$, then the function $f(x) = 2^x - \tan\alpha$ satisfies $f(0) = -1$. -/
theorem problem_solution (h : Real.sin α = 2 * Real.cos α) : (2^0 - Real.tan α) = -1 := by
  sorry

end problem_solution_l282_282250


namespace value_of_expression_l282_282221

theorem value_of_expression (x y : ℝ) (h₁ : x * y = 3) (h₂ : x + y = 4) : x ^ 2 + y ^ 2 - 3 * x * y = 1 := 
by
  sorry

end value_of_expression_l282_282221


namespace least_colors_hexagon_tiling_l282_282009

open SimpleGraph

-- Condition setup
def hexagon_tiling : SimpleGraph ℕ := {
  adj := λ a b, -- adjacency represents if two hexagons share a side
    -- (this needs an appropriate representation of the hexagon grid adjacency)
    sorry,
  sym := sorry, -- adjacency is symmetric
  loopless := sorry -- no loops in the graph, as a hexagon cannot share an edge with itself
}

-- Main statement
theorem least_colors_hexagon_tiling : chromatic_number hexagon_tiling = 3 := 
sorry

end least_colors_hexagon_tiling_l282_282009


namespace min_value_expr_l282_282432

theorem min_value_expr (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
  ∃ x : ℝ, x = 6 ∧ x = (2 * a + b) / c + (2 * a + c) / b + (2 * b + c) / a :=
by
  sorry

end min_value_expr_l282_282432


namespace min_rows_required_to_seat_students_l282_282465

-- Definitions based on the conditions
def seats_per_row : ℕ := 168
def total_students : ℕ := 2016
def max_students_per_school : ℕ := 40

def min_number_of_rows : ℕ :=
  -- Given that the minimum number of rows required to seat all students following the conditions is 15
  15

-- Lean statement expressing the proof problem
theorem min_rows_required_to_seat_students :
  ∃ rows : ℕ, rows = min_number_of_rows ∧
  (∀ school_sizes : List ℕ, (∀ size ∈ school_sizes, size ≤ max_students_per_school)
    → (List.sum school_sizes = total_students)
    → ∀ school_arrangement : List (List ℕ), 
        (∀ row_sizes ∈ school_arrangement, List.sum row_sizes ≤ seats_per_row) 
        → List.length school_arrangement ≤ rows) :=
sorry

end min_rows_required_to_seat_students_l282_282465


namespace distance_center_to_line_l282_282105

-- Definition of the parametric equation of the line l in the Cartesian coordinate system
def line_param_eq (t : ℝ) : ℝ × ℝ :=
  (3 - (real.sqrt 2 / 2) * t, real.sqrt 5 + (real.sqrt 2 / 2) * t)

-- Definition of the polar equation of circle C in the polar coordinate system
def circle_polar_eq (theta : ℝ) : ℝ :=
  2 * real.sqrt 5 * real.sin theta

-- Definitions for the circle in Cartesian coordinates
def circle_center : ℝ × ℝ := (0, real.sqrt 5)
def radius : ℝ := real.sqrt 5

-- Parametric equation in Cartesian converted form
noncomputable def line_in_cartesian (x y : ℝ) : Prop :=
  x + y - real.sqrt 5 = 3

-- Distance from a point to a line in Cartesian form
noncomputable def point_line_distance (x0 y0 a b c : ℝ) : ℝ :=
  abs (a * x0 + b * y0 + c) / real.sqrt (a ^ 2 + b ^ 2)

-- Parameters of line equation for distance calculation
def a : ℝ := 1
def b : ℝ := 1
def c : ℝ := -(real.sqrt 5 + 3)
def point : ℝ × ℝ := (0, real.sqrt 5)

-- Thm (I): Distance from the center of the circle to the line
theorem distance_center_to_line : point_line_distance (0) (real.sqrt 5) a b c = (3 * real.sqrt 2) / 2 :=
  sorry

-- Thm (II): Sum of distances |PA| + |PB|
noncomputable theorem sum_distances_PA_PB (t : ℝ) : 
  let P := (3, real.sqrt 5),
    A := line_param_eq t, 
    B := line_param_eq (-t) 
  in (abs (t) + abs (-t)) = 3 * real.sqrt 2 :=
  sorry

end distance_center_to_line_l282_282105


namespace hamburger_varieties_l282_282689

-- Define the problem conditions as Lean definitions.
def condiments := 9  -- There are 9 condiments
def patty_choices := 3  -- Choices of 1, 2, or 3 patties

-- The goal is to prove that the number of different kinds of hamburgers is 1536.
theorem hamburger_varieties : (3 * 2^9) = 1536 := by
  sorry

end hamburger_varieties_l282_282689


namespace simplify_expression_l282_282279

variable (a b : ℝ)

theorem simplify_expression : 
  (a^(2/3) * b^(1/2)) * (-3 * a^(1/2) * b^(1/3)) / (1/3 * a^(1/6) * b^(5/6)) = -9 * a := 
  sorry

end simplify_expression_l282_282279


namespace product_ge_one_l282_282718

variable (a b : ℝ)
variable (x1 x2 x3 x4 x5 : ℝ)

theorem product_ge_one
  (ha : 0 < a)
  (hb : 0 < b)
  (h_ab : a + b = 1)
  (hx1 : 0 < x1)
  (hx2 : 0 < x2)
  (hx3 : 0 < x3)
  (hx4 : 0 < x4)
  (hx5 : 0 < x5)
  (h_prod_xs : x1 * x2 * x3 * x4 * x5 = 1) :
  (a * x1 + b) * (a * x2 + b) * (a * x3 + b) * (a * x4 + b) * (a * x5 + b) ≥ 1 :=
by
  sorry

end product_ge_one_l282_282718


namespace smallest_t_l282_282281

theorem smallest_t (p q r : ℕ) (h₁ : 0 < p) (h₂ : 0 < q) (h₃ : 0 < r) (h₄ : p + q + r = 2510) 
                   (k : ℕ) (t : ℕ) (h₅ : p! * q! * r! = k * 10^t) (h₆ : ¬(10 ∣ k)) : t = 626 := 
by sorry

end smallest_t_l282_282281


namespace three_digit_numbers_not_multiple_of_3_or_11_l282_282557

theorem three_digit_numbers_not_multiple_of_3_or_11 : 
  let total_three_digit_numbers := 999 - 100 + 1 in
  let multiples_of_3 := 333 - 34 + 1 in
  let multiples_of_11 := 90 - 10 + 1 in
  let multiples_of_33 := 30 - 4 + 1 in
  let multiples_of_3_or_11 := multiples_of_3 + multiples_of_11 - multiples_of_33 in
  total_three_digit_numbers - multiples_of_3_or_11 = 546 :=
by
  let total_three_digit_numbers := 999 - 100 + 1
  let multiples_of_3 := 333 - 34 + 1
  let multiples_of_11 := 90 - 10 + 1
  let multiples_of_33 := 30 - 4 + 1
  let multiples_of_3_or_11 := multiples_of_3 + multiples_of_11 - multiples_of_33
  show total_three_digit_numbers - multiples_of_3_or_11 = 546 from sorry

end three_digit_numbers_not_multiple_of_3_or_11_l282_282557


namespace sum_of_central_squares_is_34_l282_282053

-- Defining the parameters and conditions
def is_adjacent (i j : ℕ) : Prop := 
  (i = j + 1 ∨ i = j - 1 ∨ i = j + 4 ∨ i = j - 4)

def valid_matrix (M : Fin 4 → Fin 4 → ℕ) : Prop := 
  ∀ (i j : Fin 4), 
  i < 3 ∧ j < 3 → is_adjacent (M i j) (M (i + 1) j) ∧ is_adjacent (M i j) (M i (j + 1))

def corners_sum_to_34 (M : Fin 4 → Fin 4 → ℕ) : Prop :=
  M 0 0 + M 0 3 + M 3 0 + M 3 3 = 34

-- Stating the proof problem
theorem sum_of_central_squares_is_34 :
  ∃ (M : Fin 4 → Fin 4 → ℕ), valid_matrix M ∧ corners_sum_to_34 M → 
  (M 1 1 + M 1 2 + M 2 1 + M 2 2 = 34) :=
by
  sorry

end sum_of_central_squares_is_34_l282_282053


namespace avg_wx_l282_282696

theorem avg_wx (w x y : ℝ) (h1 : 3 / w + 3 / x = 3 / y) (h2 : w * x = y) : (w + x) / 2 = 1 / 2 :=
by
  -- omitted proof
  sorry

end avg_wx_l282_282696


namespace simplify_expression_l282_282370

theorem simplify_expression (a : ℝ) (h₀ : a ≥ 0) (h₁ : a ≠ 1) (h₂ : a ≠ 1 + Real.sqrt 2) (h₃ : a ≠ 1 - Real.sqrt 2) :
  (1 + 2 * a ^ (1 / 4) - a ^ (1 / 2)) / (1 - a + 4 * a ^ (3 / 4) - 4 * a ^ (1 / 2)) +
  (a ^ (1 / 4) - 2) / (a ^ (1 / 4) - 1) ^ 2 = 1 / (a ^ (1 / 4) - 1) :=
by
  sorry

end simplify_expression_l282_282370


namespace kenny_trumpet_hours_l282_282849

variables (x y : ℝ)
def basketball_hours := 10
def running_hours := 2 * basketball_hours
def trumpet_hours := 2 * running_hours

theorem kenny_trumpet_hours (x y : ℝ) (H : basketball_hours + running_hours + trumpet_hours = x + y) :
  trumpet_hours = 40 :=
by
  sorry

end kenny_trumpet_hours_l282_282849


namespace option_d_always_correct_l282_282075

variable {a b : ℝ}

theorem option_d_always_correct (h1 : a < b) (h2 : b < 0) (h3 : a < 0) :
  (a + 1 / b)^2 > (b + 1 / a)^2 :=
by
  -- Lean proof code would go here.
  sorry

end option_d_always_correct_l282_282075


namespace part1_part2_l282_282823

open Set

variable (a : ℝ)

def real_universe := @univ ℝ

def set_A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}
def set_B : Set ℝ := {x | 2 < x ∧ x < 10}
def set_C (a : ℝ) : Set ℝ := {x | x ≤ a}

noncomputable def complement_A := (real_universe \ set_A)

theorem part1 : (complement_A ∩ set_B) = { x | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10) } :=
by sorry

theorem part2 : set_A ⊆ set_C a → a > 7 :=
by sorry

end part1_part2_l282_282823


namespace solution_set_I_range_of_a_II_l282_282238

def f (x a : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2*x - 1|

theorem solution_set_I (x : ℝ) (a : ℝ) (h : a = 2) :
  f x a ≤ 6 ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

theorem range_of_a_II (a : ℝ) :
  (∀ x : ℝ, f x a + g x ≥ 3) ↔ 2 ≤ a := by
  sorry

end solution_set_I_range_of_a_II_l282_282238


namespace solutions_x_y_l282_282530

-- Helper definitions for the rational solutions
noncomputable def rational_sol (p : ℚ) :=
  ( (1 + 1/p) ^ (p + 1), (1 + 1/p) ^ p )

-- Main theorem statement
theorem solutions_x_y (x y : ℚ) :

  -- Natural numbers case
  ((x ∈ ℕ ∧ y ∈ ℕ) → (x = y ∨ (x = 4 ∧ y = 2)) ) ∧

  -- Rational numbers case
  ((x ∈ ℚ ∧ y ∈ ℚ) → ( ∃ p : ℚ, p ≠ 0 ∧ (x, y) = rational_sol p )) :=
sorry

end solutions_x_y_l282_282530


namespace henry_present_age_l282_282023

theorem henry_present_age (H J : ℕ) (h1 : H + J = 41) (h2 : H - 7 = 2 * (J - 7)) : H = 25 :=
sorry

end henry_present_age_l282_282023


namespace find_eq_thirteen_l282_282977

open Real

theorem find_eq_thirteen
  (a x b y c z : ℝ)
  (h1 : x / a + y / b + z / c = 5)
  (h2 : a / x + b / y + c / z = 6) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 13 := 
sorry

end find_eq_thirteen_l282_282977


namespace percentage_problem_l282_282563

theorem percentage_problem (P : ℝ) :
  (P / 100) * 600 = (40 / 100) * 1050 → P = 70 :=
by
  intro h
  sorry

end percentage_problem_l282_282563


namespace natalie_blueberry_bushes_l282_282214

-- Definitions of the conditions
def bushes_yield_containers (bushes containers : ℕ) : Prop :=
  containers = bushes * 7

def containers_exchange_zucchinis (containers zucchinis : ℕ) : Prop :=
  zucchinis = containers * 3 / 7

-- Theorem statement
theorem natalie_blueberry_bushes (zucchinis_needed : ℕ) (zucchinis_per_trade containers_per_trade bushes_per_container : ℕ) 
  (h1 : zucchinis_per_trade = 3) (h2 : containers_per_trade = 7) (h3 : bushes_per_container = 7) 
  (h4 : zucchinis_needed = 63) : 
  ∃ bushes_needed : ℕ, bushes_needed = 21 := 
by
  sorry

end natalie_blueberry_bushes_l282_282214


namespace wrapping_paper_fraction_used_l282_282006

theorem wrapping_paper_fraction_used 
  (total_paper_used : ℚ)
  (num_presents : ℕ)
  (each_present_used : ℚ)
  (h1 : total_paper_used = 1 / 2)
  (h2 : num_presents = 5)
  (h3 : each_present_used = total_paper_used / num_presents) : 
  each_present_used = 1 / 10 := 
by
  sorry

end wrapping_paper_fraction_used_l282_282006


namespace function_characterization_l282_282663

theorem function_characterization (f : ℤ → ℤ)
  (h : ∀ a b : ℤ, ∃ k : ℤ, f (f a - b) + b * f (2 * a) = k ^ 2) :
  (∀ n : ℤ, (n % 2 = 0 → f n = 0) ∧ (n % 2 ≠ 0 → ∃ k: ℤ, f n = k ^ 2))
  ∨ (∀ n : ℤ, ∃ k: ℤ, f n = k ^ 2 ∧ k = n) :=
sorry

end function_characterization_l282_282663


namespace tree_height_at_end_of_4_years_l282_282349

theorem tree_height_at_end_of_4_years 
  (initial_growth : ℕ → ℕ)
  (height_7_years : initial_growth 7 = 64)
  (growth_pattern : ∀ n, initial_growth (n + 1) = 2 * initial_growth n) :
  initial_growth 4 = 8 :=
by
  sorry

end tree_height_at_end_of_4_years_l282_282349


namespace log_expression_value_l282_282034

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_expression_value :
  log10 8 + 3 * log10 4 - 2 * log10 2 + 4 * log10 25 + log10 16 = 11 := by
  sorry

end log_expression_value_l282_282034


namespace cash_refund_per_bottle_l282_282976

-- Define the constants based on the conditions
def bottles_per_month : ℕ := 15
def cost_per_bottle : ℝ := 3.0
def bottles_can_buy_with_refund : ℕ := 6
def months_per_year : ℕ := 12

-- Define the total number of bottles consumed in a year
def total_bottles_per_year : ℕ := bottles_per_month * months_per_year

-- Define the total refund in dollars after 1 year
def total_refund_amount : ℝ := bottles_can_buy_with_refund * cost_per_bottle

-- Define the statement we need to prove
theorem cash_refund_per_bottle :
  total_refund_amount / total_bottles_per_year = 0.10 :=
by
  -- This is where the steps would be completed to prove the theorem
  sorry

end cash_refund_per_bottle_l282_282976


namespace discriminant_of_quadratic_eq_l282_282458

/-- The discriminant of a quadratic equation -/
def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem discriminant_of_quadratic_eq : discriminant 1 3 (-1) = 13 := by
  sorry

end discriminant_of_quadratic_eq_l282_282458


namespace average_probable_weight_l282_282703

-- Definitions based on the conditions
def ArunOpinion (w : ℝ) : Prop := 65 < w ∧ w < 72
def BrotherOpinion (w : ℝ) : Prop := 60 < w ∧ w < 70
def MotherOpinion (w : ℝ) : Prop := w ≤ 68

-- The actual statement we want to prove
theorem average_probable_weight : 
  (∀ (w : ℝ), ArunOpinion w → BrotherOpinion w → MotherOpinion w → 65 < w ∧ w ≤ 68) →
  (65 + 68) / 2 = 66.5 :=
by 
  intros h1
  sorry

end average_probable_weight_l282_282703


namespace six_digit_numbers_l282_282484

def isNonPerfectPower (n : ℕ) : Prop :=
  ∀ m k : ℕ, m ≥ 2 → k ≥ 2 → m^k ≠ n

theorem six_digit_numbers : ∃ x : ℕ, 
  100000 ≤ x ∧ x < 1000000 ∧ 
  (∃ a b c: ℕ, x = (a^3 * b)^2 ∧ isNonPerfectPower a ∧ isNonPerfectPower b ∧ isNonPerfectPower c ∧ 
    (∃ k : ℤ, k > 1 ∧ 
      (x: ℤ) / (k^3 : ℤ) < 1 ∧ 
      ∃ num denom: ℕ, num < denom ∧ 
      num = n^3 ∧ denom = d^2 ∧ 
      isNonPerfectPower n ∧ isNonPerfectPower d)) := 
sorry

end six_digit_numbers_l282_282484


namespace P_subset_Q_l282_282907

def P : Set ℕ := {1, 2, 4}
def Q : Set ℕ := {1, 2, 4, 8}

theorem P_subset_Q : P ⊂ Q := by
  sorry

end P_subset_Q_l282_282907


namespace range_of_a_l282_282107

variable (a : ℝ)
variable (x : ℝ)

noncomputable def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (h : ∀ x, otimes (x - a) (x + a) < 1) : - 1 / 2 < a ∧ a < 3 / 2 :=
sorry

end range_of_a_l282_282107


namespace trig_expression_value_l282_282958

theorem trig_expression_value (α : ℝ) (h₁ : Real.tan (α + π / 4) = -1/2) (h₂ : π / 2 < α ∧ α < π) :
  (Real.sin (2 * α) - 2 * (Real.cos α)^2) / Real.sin (α - π / 4) = - (2 * Real.sqrt 5) / 5 :=
by
  sorry

end trig_expression_value_l282_282958


namespace initial_oranges_count_l282_282910

theorem initial_oranges_count 
  (O : ℕ)
  (h1 : 10 = O - 13) : 
  O = 23 := 
sorry

end initial_oranges_count_l282_282910


namespace sale_price_correct_l282_282755

variable (x : ℝ)

-- Conditions
def decreased_price (x : ℝ) : ℝ :=
  0.9 * x

def final_sale_price (decreased_price : ℝ) : ℝ :=
  0.7 * decreased_price

-- Proof statement
theorem sale_price_correct : final_sale_price (decreased_price x) = 0.63 * x := by
  sorry

end sale_price_correct_l282_282755


namespace depth_of_well_l282_282050

theorem depth_of_well
  (d : ℝ)
  (h1 : ∃ t1 t2 : ℝ, 18 * t1^2 = d ∧ t2 = d / 1150 ∧ t1 + t2 = 8) :
  d = 33.18 :=
sorry

end depth_of_well_l282_282050


namespace find_fake_coin_l282_282321

theorem find_fake_coin (k : ℕ) :
  ∃ (weighings : ℕ), (weighings ≤ 3 * k + 1) :=
sorry

end find_fake_coin_l282_282321


namespace parallel_planes_perpendicular_planes_l282_282967

variables {A1 B1 C1 D1 A2 B2 C2 D2 : ℝ}

-- Parallelism Condition
theorem parallel_planes (h₁ : A1 ≠ 0) (h₂ : B1 ≠ 0) (h₃ : C1 ≠ 0) (h₄ : A2 ≠ 0) (h₅ : B2 ≠ 0) (h₆ : C2 ≠ 0) :
  (A1 / A2 = B1 / B2 ∧ B1 / B2 = C1 / C2) ↔ (∃ k : ℝ, (A1 = k * A2) ∧ (B1 = k * B2) ∧ (C1 = k * C2)) :=
sorry

-- Perpendicularity Condition
theorem perpendicular_planes :
  A1 * A2 + B1 * B2 + C1 * C2 = 0 :=
sorry

end parallel_planes_perpendicular_planes_l282_282967


namespace calculate_value_of_squares_difference_l282_282659

theorem calculate_value_of_squares_difference : 305^2 - 301^2 = 2424 :=
by {
  sorry
}

end calculate_value_of_squares_difference_l282_282659


namespace solve_system_l282_282136

noncomputable def f (a b x : ℝ) : ℝ := a^x + b

theorem solve_system (a b : ℝ) :
  (f a b 1 = 4) ∧ (f a b 0 = 2) →
  a = 3 ∧ b = 1 :=
by
  sorry

end solve_system_l282_282136


namespace max_value_of_m_l282_282979

theorem max_value_of_m :
  (∃ (t : ℝ), ∀ (x : ℝ), 2 ≤ x ∧ x ≤ m → (x + t)^2 ≤ 2 * x) → m ≤ 8 :=
sorry

end max_value_of_m_l282_282979


namespace concentration_sequences_and_min_operations_l282_282190

theorem concentration_sequences_and_min_operations :
  (a_1 = 1.55 ∧ b_1 = 0.65) ∧
  (∀ n ≥ 1, a_n - b_n = 0.9 * (1 / 2)^(n - 1)) ∧
  (∃ n, 0.9 * (1 / 2)^(n - 1) < 0.01 ∧ n = 8) :=
by
  sorry

end concentration_sequences_and_min_operations_l282_282190


namespace cylinder_lateral_surface_area_l282_282299

theorem cylinder_lateral_surface_area (r l : ℝ) (A : ℝ) (h_r : r = 1) (h_l : l = 2) : A = 4 * Real.pi :=
by
  sorry

end cylinder_lateral_surface_area_l282_282299


namespace largest_of_consecutive_non_prime_integers_l282_282865

-- Definition of a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m:ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of consecutive non-prime sequence condition
def consecutive_non_prime_sequence (start : ℕ) : Prop :=
  ∀ i : ℕ, 0 ≤ i → i < 10 → ¬ is_prime (start + i)

theorem largest_of_consecutive_non_prime_integers :
  (∃ start, start + 9 < 50 ∧ consecutive_non_prime_sequence start) →
  (∃ start, start + 9 = 47) :=
by
  sorry

end largest_of_consecutive_non_prime_integers_l282_282865
