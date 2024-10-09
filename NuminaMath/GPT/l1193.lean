import Mathlib

namespace third_podcast_length_correct_l1193_119302

def first_podcast_length : ℕ := 45
def fourth_podcast_length : ℕ := 60
def next_podcast_length : ℕ := 60
def total_drive_time : ℕ := 360

def second_podcast_length := 2 * first_podcast_length

def total_time_other_than_third := first_podcast_length + second_podcast_length + fourth_podcast_length + next_podcast_length

theorem third_podcast_length_correct :
  total_drive_time - total_time_other_than_third = 105 := by
  -- Proof goes here
  sorry

end third_podcast_length_correct_l1193_119302


namespace gcd_459_357_polynomial_at_neg4_l1193_119343

-- Statement for the GCD problem
theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

-- Definition of the polynomial
def f (x : Int) : Int :=
  12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6

-- Statement for the polynomial evaluation problem
theorem polynomial_at_neg4 : f (-4) = 3392 := by
  sorry

end gcd_459_357_polynomial_at_neg4_l1193_119343


namespace ants_of_species_X_on_day_6_l1193_119376

/-- Given the initial populations of Species X and Species Y and their growth rates,
    prove the number of Species X ants on Day 6. -/
theorem ants_of_species_X_on_day_6 
  (x y : ℕ)  -- Number of Species X and Y ants on Day 0
  (h1 : x + y = 40)  -- Total number of ants on Day 0
  (h2 : 64 * x + 4096 * y = 21050)  -- Total number of ants on Day 6
  :
  64 * x = 2304 := 
sorry

end ants_of_species_X_on_day_6_l1193_119376


namespace log_sum_correct_l1193_119334

noncomputable def log_sum : Prop :=
  let x := (3/2)
  let y := (5/3)
  (x + y) = (19/6)

theorem log_sum_correct : log_sum :=
by
  sorry

end log_sum_correct_l1193_119334


namespace count_ways_to_choose_one_person_l1193_119388

theorem count_ways_to_choose_one_person (A B : ℕ) (hA : A = 3) (hB : B = 5) : A + B = 8 :=
by
  sorry

end count_ways_to_choose_one_person_l1193_119388


namespace trapezium_other_side_length_l1193_119304

theorem trapezium_other_side_length :
  ∃ (x : ℝ), 1/2 * (18 + x) * 17 = 323 ∧ x = 20 :=
by
  sorry

end trapezium_other_side_length_l1193_119304


namespace range_of_x_plus_y_l1193_119385

theorem range_of_x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x * y - (x + y) = 1) : 
  x + y ≥ 2 + 2 * Real.sqrt 2 :=
sorry

end range_of_x_plus_y_l1193_119385


namespace solve_equation_l1193_119325

theorem solve_equation (x : ℝ) : 4 * (x - 1) ^ 2 = 9 ↔ x = 5 / 2 ∨ x = -1 / 2 := 
by 
  sorry

end solve_equation_l1193_119325


namespace greatest_possible_red_points_l1193_119339

theorem greatest_possible_red_points (R B : ℕ) (h1 : R + B = 25)
    (h2 : ∀ r1 r2, r1 < R → r2 < R → r1 ≠ r2 → ∃ (n : ℕ), (∃ b1 : ℕ, b1 < B) ∧ ¬∃ b2 : ℕ, b2 < B) :
  R ≤ 13 :=
by {
  sorry
}

end greatest_possible_red_points_l1193_119339


namespace faster_train_length_225_l1193_119387

noncomputable def length_of_faster_train (speed_slower speed_faster : ℝ) (time : ℝ) : ℝ :=
  let relative_speed_kmph := speed_slower + speed_faster
  let relative_speed_mps := (relative_speed_kmph * 1000) / 3600
  relative_speed_mps * time

theorem faster_train_length_225 :
  length_of_faster_train 36 45 10 = 225 := by
  sorry

end faster_train_length_225_l1193_119387


namespace quadratic_distinct_real_roots_l1193_119362

theorem quadratic_distinct_real_roots (k : ℝ) : k < 1 / 2 ∧ k ≠ 0 ↔ (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (k * x1^2 - 2 * x1 + 2 = 0) ∧ (k * x2^2 - 2 * x2 + 2 = 0)) := 
by 
  sorry

end quadratic_distinct_real_roots_l1193_119362


namespace evaluate_expression_l1193_119399

theorem evaluate_expression :
  let a := 24
  let b := 7
  3 * (a + b) ^ 2 - (a ^ 2 + b ^ 2) = 2258 :=
by
  let a := 24
  let b := 7
  sorry

end evaluate_expression_l1193_119399


namespace wire_length_ratio_l1193_119395

noncomputable def total_wire_length_bonnie (pieces : Nat) (length_per_piece : Nat) := 
  pieces * length_per_piece

noncomputable def volume_of_cube (edge_length : Nat) := 
  edge_length ^ 3

noncomputable def wire_length_roark_per_cube (edges_per_cube : Nat) (length_per_edge : Nat) (num_cubes : Nat) :=
  edges_per_cube * length_per_edge * num_cubes

theorem wire_length_ratio : 
  let bonnie_pieces := 12
  let bonnie_length_per_piece := 8
  let bonnie_edge_length := 8
  let roark_length_per_edge := 2
  let roark_edges_per_cube := 12
  let bonnie_wire_length := total_wire_length_bonnie bonnie_pieces bonnie_length_per_piece
  let bonnie_cube_volume := volume_of_cube bonnie_edge_length
  let roark_num_cubes := bonnie_cube_volume
  let roark_wire_length := wire_length_roark_per_cube roark_edges_per_cube roark_length_per_edge roark_num_cubes
  bonnie_wire_length / roark_wire_length = 1 / 128 :=
by
  sorry

end wire_length_ratio_l1193_119395


namespace compute_fraction_l1193_119314

theorem compute_fraction : (1922^2 - 1913^2) / (1930^2 - 1905^2) = (9 : ℚ) / 25 := by
  sorry

end compute_fraction_l1193_119314


namespace abc_correct_and_c_not_true_l1193_119383

theorem abc_correct_and_c_not_true (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  a^2 > b^2 ∧ ab > b^2 ∧ (1/(a+b) > 1/a) ∧ ¬(1/a < 1/b) :=
  sorry

end abc_correct_and_c_not_true_l1193_119383


namespace no_one_is_always_largest_l1193_119367

theorem no_one_is_always_largest (a b c d : ℝ) :
  a - 2 = b + 3 ∧ a - 2 = c * 2 ∧ a - 2 = d + 5 →
  ∀ x, (x = a ∨ x = b ∨ x = c ∨ x = d) → (x ≤ c ∨ x ≤ a) :=
by
  -- The proof requires assuming the conditions and showing that no variable is always the largest.
  intro h cond
  sorry

end no_one_is_always_largest_l1193_119367


namespace ratio_eq_l1193_119332

variable (a b c d : ℚ)

theorem ratio_eq :
  (a / b = 5 / 2) →
  (c / d = 7 / 3) →
  (d / b = 5 / 4) →
  (a / c = 6 / 7) :=
by
  intros h1 h2 h3
  sorry

end ratio_eq_l1193_119332


namespace rank_identity_l1193_119301

theorem rank_identity (n p : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) 
  (h1: 2 ≤ n) (h2: 2 ≤ p) (h3: A^(p+1) = A) : 
  Matrix.rank A + Matrix.rank (1 - A^p) = n := 
  sorry

end rank_identity_l1193_119301


namespace smallest_integer_in_range_l1193_119393

-- Given conditions
def is_congruent_6 (n : ℕ) : Prop := n % 6 = 1
def is_congruent_7 (n : ℕ) : Prop := n % 7 = 1
def is_congruent_8 (n : ℕ) : Prop := n % 8 = 1

-- Lean statement for the proof problem
theorem smallest_integer_in_range :
  ∃ n : ℕ, (n > 1) ∧ is_congruent_6 n ∧ is_congruent_7 n ∧ is_congruent_8 n ∧ (n = 169) ∧ (120 ≤ n ∧ n < 210) :=
by
  sorry

end smallest_integer_in_range_l1193_119393


namespace parallel_lines_implies_value_of_m_l1193_119329

theorem parallel_lines_implies_value_of_m :
  ∀ (m : ℝ), (∀ (x y : ℝ), 3 * x + 2 * y - 2 = 0) ∧ (∀ (x y : ℝ), (2 * m - 1) * x + m * y + 1 = 0) → 
  m = 2 := 
by
  sorry

end parallel_lines_implies_value_of_m_l1193_119329


namespace least_integer_value_l1193_119357

theorem least_integer_value :
  ∃ x : ℤ, (∀ x' : ℤ, (|3 * x' + 4| <= 18) → (x' >= x)) ∧ (|3 * x + 4| <= 18) ∧ x = -7 := 
sorry

end least_integer_value_l1193_119357


namespace ratio_solution_l1193_119394

theorem ratio_solution (x : ℚ) : (1 : ℚ) / 3 = 5 / 3 / x → x = 5 := 
by
  intro h
  sorry

end ratio_solution_l1193_119394


namespace back_wheel_revolutions_calculation_l1193_119353

noncomputable def front_diameter : ℝ := 3 -- Diameter of the front wheel in feet
noncomputable def back_diameter : ℝ := 0.5 -- Diameter of the back wheel in feet
noncomputable def no_slippage : Prop := true -- No slippage condition
noncomputable def front_revolutions : ℕ := 150 -- Number of front wheel revolutions

theorem back_wheel_revolutions_calculation 
  (d_f : ℝ) (d_b : ℝ) (slippage : Prop) (n_f : ℕ) : 
  slippage → d_f = front_diameter → d_b = back_diameter → 
  n_f = front_revolutions → 
  ∃ n_b : ℕ, n_b = 900 := 
by
  sorry

end back_wheel_revolutions_calculation_l1193_119353


namespace pyramid_addition_totals_l1193_119326

theorem pyramid_addition_totals 
  (initial_faces : ℕ) (initial_edges : ℕ) (initial_vertices : ℕ)
  (first_pyramid_new_faces : ℕ) (first_pyramid_new_edges : ℕ) (first_pyramid_new_vertices : ℕ)
  (second_pyramid_new_faces : ℕ) (second_pyramid_new_edges : ℕ) (second_pyramid_new_vertices : ℕ)
  (cancelling_faces_first : ℕ) (cancelling_faces_second : ℕ) :
  initial_faces = 5 → 
  initial_edges = 9 → 
  initial_vertices = 6 → 
  first_pyramid_new_faces = 3 →
  first_pyramid_new_edges = 3 →
  first_pyramid_new_vertices = 1 →
  second_pyramid_new_faces = 4 →
  second_pyramid_new_edges = 4 →
  second_pyramid_new_vertices = 1 →
  cancelling_faces_first = 1 →
  cancelling_faces_second = 1 →
  initial_faces + first_pyramid_new_faces - cancelling_faces_first 
  + second_pyramid_new_faces - cancelling_faces_second 
  + initial_edges + first_pyramid_new_edges + second_pyramid_new_edges
  + initial_vertices + first_pyramid_new_vertices + second_pyramid_new_vertices 
  = 34 := by sorry

end pyramid_addition_totals_l1193_119326


namespace accurate_measurement_l1193_119335

-- Define the properties of Dr. Sharadek's tape
structure SharadekTape where
  startsWithHalfCM : Bool -- indicates if the tape starts with a half-centimeter bracket
  potentialError : ℝ -- potential measurement error

-- Define the conditions as an instance of the structure
noncomputable def drSharadekTape : SharadekTape :=
  { startsWithHalfCM := true,
    potentialError := 0.5 }

-- Define a segment with a known precise measurement
structure Segment where
  length : ℝ

noncomputable def AB (N : ℕ) : Segment :=
  { length := N + 0.5 }

-- The theorem stating the correct answer under the given conditions
theorem accurate_measurement (N : ℕ) : 
  ∃ AB : Segment, AB.length = N + 0.5 :=
by
  existsi AB N
  exact rfl

end accurate_measurement_l1193_119335


namespace prime_square_sub_one_divisible_by_24_l1193_119310

theorem prime_square_sub_one_divisible_by_24 (p : ℕ) (hp : p ≥ 5) (prime_p : Nat.Prime p) : 24 ∣ p^2 - 1 := by
  sorry

end prime_square_sub_one_divisible_by_24_l1193_119310


namespace weather_on_july_15_l1193_119347

theorem weather_on_july_15 
  (T: ℝ) (sunny: Prop) (W: ℝ) (crowded: Prop) 
  (h1: (T ≥ 85 ∧ sunny ∧ W < 15) → crowded) 
  (h2: ¬ crowded) : (T < 85 ∨ ¬ sunny ∨ W ≥ 15) :=
sorry

end weather_on_july_15_l1193_119347


namespace fifth_term_of_arithmetic_sequence_is_minus_three_l1193_119368

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

theorem fifth_term_of_arithmetic_sequence_is_minus_three (a d : ℤ) :
  (arithmetic_sequence a d 11 = 25) ∧ (arithmetic_sequence a d 12 = 29) →
  (arithmetic_sequence a d 4 = -3) :=
by 
  intros h
  sorry

end fifth_term_of_arithmetic_sequence_is_minus_three_l1193_119368


namespace central_angle_measure_l1193_119381

-- Given conditions
def radius : ℝ := 2
def area : ℝ := 4

-- Central angle α
def central_angle : ℝ := 2

-- Theorem statement: The central angle measure is 2 radians
theorem central_angle_measure :
  ∃ α : ℝ, α = central_angle ∧ area = (1/2) * (α * radius) := 
sorry

end central_angle_measure_l1193_119381


namespace geometric_sequence_n_value_l1193_119351

theorem geometric_sequence_n_value (a₁ : ℕ) (q : ℕ) (a_n : ℕ) (n : ℕ) (h1 : a₁ = 1) (h2 : q = 2) (h3 : a_n = 64) (h4 : a_n = a₁ * q^(n-1)) : n = 7 :=
by
  sorry

end geometric_sequence_n_value_l1193_119351


namespace max_value_of_f_on_interval_l1193_119307

noncomputable def f (x : ℝ) : ℝ := 2^x + x * Real.log (1/4)

theorem max_value_of_f_on_interval :
  ∃ x ∈ Set.Icc (-2:ℝ) 2, f x = (1/4:ℝ) + 4 * Real.log 2 := 
sorry

end max_value_of_f_on_interval_l1193_119307


namespace total_missed_questions_l1193_119333

-- Definitions
def missed_by_you : ℕ := 36
def missed_by_friend : ℕ := 7
def missed_by_you_friends : ℕ := missed_by_you + missed_by_friend

-- Theorem
theorem total_missed_questions (h1 : missed_by_you = 5 * missed_by_friend) :
  missed_by_you_friends = 43 :=
by
  sorry

end total_missed_questions_l1193_119333


namespace quadratic_to_square_form_l1193_119312

theorem quadratic_to_square_form (x : ℝ) :
  (x^2 - 6*x + 7 = 0) ↔ ((x - 3)^2 = 2) :=
sorry

end quadratic_to_square_form_l1193_119312


namespace total_students_l1193_119386

-- Define the conditions
def ratio_girls_boys (G B : ℕ) : Prop := G / B = 1 / 2
def ratio_math_girls (M N : ℕ) : Prop := M / N = 3 / 1
def ratio_sports_boys (S T : ℕ) : Prop := S / T = 4 / 1

-- Define the problem statement
theorem total_students (G B M N S T : ℕ) 
  (h1 : ratio_girls_boys G B)
  (h2 : ratio_math_girls M N)
  (h3 : ratio_sports_boys S T)
  (h4 : M = 12)
  (h5 : G = M + N)
  (h6 : G = 16) 
  (h7 : B = 32) : 
  G + B = 48 :=
sorry

end total_students_l1193_119386


namespace g_of_2_l1193_119354

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : x * g y = 2 * y * g x 
axiom g_of_10 : g 10 = 5

theorem g_of_2 : g 2 = 2 :=
by
    sorry

end g_of_2_l1193_119354


namespace total_area_equals_total_frequency_l1193_119327

-- Definition of frequency and frequency distribution histogram
def frequency_distribution_histogram (frequencies : List ℕ) := ∀ i, (i < frequencies.length) → ℕ

-- Definition that the total area of the small rectangles is the sum of the frequencies
def total_area_of_rectangles (frequencies : List ℕ) : ℕ := frequencies.sum

-- Theorem stating the equivalence
theorem total_area_equals_total_frequency (frequencies : List ℕ) :
  total_area_of_rectangles frequencies = frequencies.sum := 
by
  sorry

end total_area_equals_total_frequency_l1193_119327


namespace evaluate_fx_plus_2_l1193_119390

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)

theorem evaluate_fx_plus_2 (x : ℝ) (h : x ^ 2 ≠ 1) : 
  f (x + 2) = (x + 3) / (x + 1) :=
by
  sorry

end evaluate_fx_plus_2_l1193_119390


namespace union_complement_eq_l1193_119350

open Set

variable (I A B : Set ℤ)
variable (I_def : I = {-3, -2, -1, 0, 1, 2})
variable (A_def : A = {-1, 1, 2})
variable (B_def : B = {-2, -1, 0})

theorem union_complement_eq :
  A ∪ (I \ B) = {-3, -1, 1, 2} :=
by 
  rw [I_def, A_def, B_def]
  sorry

end union_complement_eq_l1193_119350


namespace find_minimum_value_l1193_119379

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 + 6*y^2 - 2*x*y - 14*x - 6*y + 72

theorem find_minimum_value :
  let x := 9
  let y := 2
  (∀ x y : ℝ, f x y ≥ 3) ∧ (f 9 2 = 3) :=
by
  sorry

end find_minimum_value_l1193_119379


namespace problem_f_2004_l1193_119352

noncomputable def f (x : ℝ) (a : ℝ) (α : ℝ) (b : ℝ) (β : ℝ) : ℝ := 
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem problem_f_2004 (a α b β : ℝ) 
  (h_non_zero : a ≠ 0 ∧ b ≠ 0 ∧ α ≠ 0 ∧ β ≠ 0) 
  (h_condition : f 2003 a α b β = 6) : 
  f 2004 a α b β = 2 := 
by
  sorry

end problem_f_2004_l1193_119352


namespace min_distance_to_circle_l1193_119365

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2))

def is_on_circle (Q : ℝ × ℝ) : Prop :=
  (Q.1 - 1)^2 + Q.2^2 = 4

def P : ℝ × ℝ := (-2, -3)
def center : ℝ × ℝ := (1, 0)
def radius : ℝ := 2

theorem min_distance_to_circle : ∃ Q : ℝ × ℝ, is_on_circle Q ∧ distance P Q = 3 * (Real.sqrt 2) - radius :=
by
  sorry

end min_distance_to_circle_l1193_119365


namespace count_positive_integers_l1193_119320

theorem count_positive_integers (n : ℕ) (m : ℕ) :
  (∀ (k : ℕ), 1 ≤ k ∧ k < 100 ∧ (∃ (n : ℕ), n = 2 * k + 1 ∧ n < 200) 
  ∧ (∃ (m : ℤ), m = k * (k + 1) ∧ m % 5 = 0)) → 
  ∃ (cnt : ℕ), cnt = 20 :=
by
  sorry

end count_positive_integers_l1193_119320


namespace max_value_quadratic_l1193_119398

theorem max_value_quadratic :
  (∃ x : ℝ, ∀ y : ℝ, -3*y^2 + 9*y + 24 ≤ -3*x^2 + 9*x + 24) ∧ (∃ x : ℝ, x = 3/2) :=
sorry

end max_value_quadratic_l1193_119398


namespace find_x_value_l1193_119319

theorem find_x_value (x : ℝ) (y : ℝ) (h1 : y = (x^2 - 9) / (x - 3)) (h2 : y = 3 * x - 4) : x = 7 / 2 := 
sorry

end find_x_value_l1193_119319


namespace worksheets_already_graded_l1193_119308

theorem worksheets_already_graded {total_worksheets problems_per_worksheet problems_left_to_grade : ℕ} :
  total_worksheets = 9 →
  problems_per_worksheet = 4 →
  problems_left_to_grade = 16 →
  (total_worksheets - (problems_left_to_grade / problems_per_worksheet)) = 5 :=
by
  intros h1 h2 h3
  sorry

end worksheets_already_graded_l1193_119308


namespace sum_due_is_correct_l1193_119355

-- Define constants for Banker's Discount and True Discount
def BD : ℝ := 288
def TD : ℝ := 240

-- Define Banker's Gain as the difference between BD and TD
def BG : ℝ := BD - TD

-- Define the sum due (S.D.) as the face value including True Discount and Banker's Gain
def SD : ℝ := TD + BG

-- Create a theorem to prove the sum due is Rs. 288
theorem sum_due_is_correct : SD = 288 :=
by
  -- Skipping proof with sorry; expect this statement to be true based on given conditions 
  sorry

end sum_due_is_correct_l1193_119355


namespace find_m_l1193_119397

theorem find_m (x1 x2 m : ℝ)
  (h1 : ∀ x, x^2 - 4 * x + m = 0 → x = x1 ∨ x = x2)
  (h2 : x1 + x2 - x1 * x2 = 1) :
  m = 3 :=
sorry

end find_m_l1193_119397


namespace max_value_expression_l1193_119373

theorem max_value_expression : ∀ (a b c d : ℝ), 
  a ∈ Set.Icc (-4.5) 4.5 → 
  b ∈ Set.Icc (-4.5) 4.5 → 
  c ∈ Set.Icc (-4.5) 4.5 → 
  d ∈ Set.Icc (-4.5) 4.5 → 
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 90 :=
by sorry

end max_value_expression_l1193_119373


namespace satisfies_conditions_l1193_119315

theorem satisfies_conditions : ∃ (n : ℤ), 0 ≤ n ∧ n < 31 ∧ -250 % 31 = n % 31 ∧ n = 29 :=
by
  sorry

end satisfies_conditions_l1193_119315


namespace fish_per_black_duck_l1193_119345

theorem fish_per_black_duck :
  ∀ (W_d B_d M_d : ℕ) (fish_per_W fish_per_M total_fish : ℕ),
    (fish_per_W = 5) →
    (fish_per_M = 12) →
    (W_d = 3) →
    (B_d = 7) →
    (M_d = 6) →
    (total_fish = 157) →
    (total_fish - (W_d * fish_per_W + M_d * fish_per_M)) = 70 →
    (70 / B_d) = 10 :=
by
  intros W_d B_d M_d fish_per_W fish_per_M total_fish hW hM hW_d hB_d hM_d htotal_fish hcalculation
  sorry

end fish_per_black_duck_l1193_119345


namespace value_of_b_plus_c_l1193_119371

theorem value_of_b_plus_c 
  (b c : ℝ) 
  (f : ℝ → ℝ)
  (h_def : ∀ x, f x = x^2 + 2 * b * x + c)
  (h_solution_set : ∀ x, f x ≤ 0 ↔ -1 ≤ x ∧ x ≤ 1) :
  b + c = -1 :=
sorry

end value_of_b_plus_c_l1193_119371


namespace minimum_value_of_PQ_l1193_119378

theorem minimum_value_of_PQ {x y : ℝ} (P : ℝ × ℝ) (h₁ : (P.1 - 3)^2 + (P.2 - 4)^2 > 4)
  (h₂ : ∀ Q : ℝ × ℝ, (Q.1 - 3)^2 + (Q.2 - 4)^2 = 4 → (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (P.1)^2 + (P.2)^2) :
  ∃ PQ_min : ℝ, PQ_min = 17/2 := by
  sorry

end minimum_value_of_PQ_l1193_119378


namespace anya_can_obtain_any_composite_number_l1193_119322

theorem anya_can_obtain_any_composite_number (n : ℕ) (h : ∃ k, k > 1 ∧ k < n ∧ n % k = 0) : ∃ m ≥ 4, ∀ k, k > 1 → k < m → m % k = 0 → m = n :=
by
  sorry

end anya_can_obtain_any_composite_number_l1193_119322


namespace student_tickets_second_day_l1193_119375

variable (S T x: ℕ)

theorem student_tickets_second_day (hT : T = 9) (h_eq1 : 4 * S + 3 * T = 79) (h_eq2 : 12 * S + x * T = 246) : x = 10 :=
by
  sorry

end student_tickets_second_day_l1193_119375


namespace max_sum_square_pyramid_addition_l1193_119361

def square_pyramid_addition_sum (faces edges vertices : ℕ) : ℕ :=
  let new_faces := faces - 1 + 4
  let new_edges := edges + 4
  let new_vertices := vertices + 1
  new_faces + new_edges + new_vertices

theorem max_sum_square_pyramid_addition :
  square_pyramid_addition_sum 6 12 8 = 34 :=
by
  sorry

end max_sum_square_pyramid_addition_l1193_119361


namespace pairs_of_polygons_with_angle_difference_l1193_119331

theorem pairs_of_polygons_with_angle_difference :
  ∃ (pairs : ℕ), pairs = 52 ∧ ∀ (n k : ℕ), n > k ∧ (360 / k - 360 / n = 1) :=
sorry

end pairs_of_polygons_with_angle_difference_l1193_119331


namespace largest_n_factors_l1193_119323

theorem largest_n_factors (n : ℤ) :
  (∃ A B : ℤ, 3 * B + A = n ∧ A * B = 72) → n ≤ 217 :=
by {
  sorry
}

end largest_n_factors_l1193_119323


namespace total_paid_is_201_l1193_119316

def adult_ticket_price : ℕ := 8
def child_ticket_price : ℕ := 5
def total_tickets : ℕ := 33
def child_tickets : ℕ := 21
def adult_tickets : ℕ := total_tickets - child_tickets
def total_paid : ℕ := (child_tickets * child_ticket_price) + (adult_tickets * adult_ticket_price)

theorem total_paid_is_201 : total_paid = 201 :=
by
  sorry

end total_paid_is_201_l1193_119316


namespace find_a_l1193_119374

theorem find_a (a : ℝ) (h_pos : a > 0) :
  (∀ x y : ℤ, x^2 - a * (x : ℝ) + 4 * a = 0) →
  a = 25 ∨ a = 18 ∨ a = 16 :=
by
  sorry

end find_a_l1193_119374


namespace trig_identity_1_trig_identity_2_l1193_119396

theorem trig_identity_1 (θ : ℝ) (h₀ : 0 < θ ∧ θ < π / 2) (h₁ : Real.tan θ = 2) :
  (Real.sin (π - θ) + Real.sin (3 * π / 2 + θ)) / 
  (3 * Real.sin (π / 2 - θ) - 2 * Real.sin (π + θ)) = 1 / 7 :=
by sorry

theorem trig_identity_2 (θ : ℝ) (h₀ : 0 < θ ∧ θ < π / 2) (h₁ : Real.tan θ = 2) :
  (1 - Real.cos (2 * θ)) / 
  (Real.sin (2 * θ) + Real.cos (2 * θ)) = 8 :=
by sorry

end trig_identity_1_trig_identity_2_l1193_119396


namespace mass_percentage_O_is_correct_l1193_119369

noncomputable def molar_mass_Al : ℝ := 26.98
noncomputable def molar_mass_O : ℝ := 16.00
noncomputable def num_Al_atoms : ℕ := 2
noncomputable def num_O_atoms : ℕ := 3

noncomputable def molar_mass_Al2O3 : ℝ :=
  (num_Al_atoms * molar_mass_Al) + (num_O_atoms * molar_mass_O)

noncomputable def mass_percentage_O_in_Al2O3 : ℝ :=
  ((num_O_atoms * molar_mass_O) / molar_mass_Al2O3) * 100

theorem mass_percentage_O_is_correct :
  mass_percentage_O_in_Al2O3 = 47.07 :=
by
  sorry

end mass_percentage_O_is_correct_l1193_119369


namespace no_intersection_tangent_graph_l1193_119328

theorem no_intersection_tangent_graph (k : ℝ) (m : ℤ) : 
  (∀ x: ℝ, x = (k * Real.pi) / 2 → (¬ 4 * k ≠ 4 * m + 1)) → 
  (-1 ≤ k ∧ k ≤ 1) →
  (k = 1 / 4 ∨ k = -3 / 4) :=
sorry

end no_intersection_tangent_graph_l1193_119328


namespace number_of_classmates_l1193_119309

theorem number_of_classmates (n : ℕ) (Alex Aleena : ℝ) 
  (h_Alex : Alex = 1/11) (h_Aleena : Aleena = 1/14) 
  (h_bound : ∀ (x : ℝ), Aleena ≤ x ∧ x ≤ Alex → ∃ c : ℕ, n - 2 > 0 ∧ c = n - 2) : 
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_l1193_119309


namespace num_ways_to_convert_20d_l1193_119360

theorem num_ways_to_convert_20d (n d q : ℕ) (h : 5 * n + 10 * d + 25 * q = 2000) (hn : n ≥ 2) (hq : q ≥ 1) :
    ∃ k : ℕ, k = 130 := sorry

end num_ways_to_convert_20d_l1193_119360


namespace cubic_eq_solutions_l1193_119349

theorem cubic_eq_solutions (x : ℝ) :
  x^3 - 4 * x = 0 ↔ x = 0 ∨ x = -2 ∨ x = 2 := by
  sorry

end cubic_eq_solutions_l1193_119349


namespace minimum_rows_required_l1193_119359

theorem minimum_rows_required (total_students : ℕ) (max_students_per_school : ℕ) (seats_per_row : ℕ) (num_schools : ℕ) 
    (h_total_students : total_students = 2016) 
    (h_max_students_per_school : max_students_per_school = 45) 
    (h_seats_per_row : seats_per_row = 168) 
    (h_num_schools : num_schools = 46) : 
    ∃ (min_rows : ℕ), min_rows = 16 := 
by 
  -- Proof omitted
  sorry

end minimum_rows_required_l1193_119359


namespace smallest_positive_integer_l1193_119321

theorem smallest_positive_integer (x : ℕ) (hx_pos : x > 0) (h : x < 15) : x = 1 :=
by
  sorry

end smallest_positive_integer_l1193_119321


namespace total_games_is_24_l1193_119341

-- Definitions of conditions
def games_this_month : Nat := 9
def games_last_month : Nat := 8
def games_next_month : Nat := 7

-- Total games attended
def total_games_attended : Nat :=
  games_this_month + games_last_month + games_next_month

-- Problem statement
theorem total_games_is_24 : total_games_attended = 24 := by
  sorry

end total_games_is_24_l1193_119341


namespace simplify_fraction_l1193_119324

theorem simplify_fraction : 
  ((2^12)^2 - (2^10)^2) / ((2^11)^2 - (2^9)^2) = 4 := 
by sorry

end simplify_fraction_l1193_119324


namespace hawks_score_l1193_119336

theorem hawks_score (x y : ℕ) (h1 : x + y = 82) (h2 : x - y = 18) : y = 32 :=
sorry

end hawks_score_l1193_119336


namespace probability_of_drawing_3_black_and_2_white_l1193_119340

noncomputable def total_ways_to_draw_5_balls : ℕ := Nat.choose 27 5
noncomputable def ways_to_choose_3_black : ℕ := Nat.choose 10 3
noncomputable def ways_to_choose_2_white : ℕ := Nat.choose 12 2
noncomputable def favorable_outcomes : ℕ := ways_to_choose_3_black * ways_to_choose_2_white
noncomputable def desired_probability : ℚ := favorable_outcomes / total_ways_to_draw_5_balls

theorem probability_of_drawing_3_black_and_2_white :
  desired_probability = 132 / 1345 := by
  sorry

end probability_of_drawing_3_black_and_2_white_l1193_119340


namespace thabo_total_books_l1193_119318

-- Definitions and conditions mapped from the problem
def H : ℕ := 35
def P_NF : ℕ := H + 20
def P_F : ℕ := 2 * P_NF
def total_books : ℕ := H + P_NF + P_F

-- The theorem proving the total number of books
theorem thabo_total_books : total_books = 200 := by
  -- Proof goes here.
  sorry

end thabo_total_books_l1193_119318


namespace min_value_of_expression_l1193_119356

theorem min_value_of_expression (x : ℝ) (hx : 0 < x) : 4 * x + 1 / x ^ 6 ≥ 5 :=
sorry

end min_value_of_expression_l1193_119356


namespace pushups_percentage_l1193_119348

def total_exercises : ℕ := 12 + 8 + 20

def percentage_pushups (total_ex: ℕ) : ℕ := (8 * 100) / total_ex

theorem pushups_percentage (h : total_exercises = 40) : percentage_pushups total_exercises = 20 :=
by
  sorry

end pushups_percentage_l1193_119348


namespace cost_hour_excess_is_1point75_l1193_119384

noncomputable def cost_per_hour_excess (x : ℝ) : Prop :=
  let total_hours := 9
  let initial_cost := 15
  let excess_hours := total_hours - 2
  let total_cost := initial_cost + excess_hours * x
  let average_cost_per_hour := 3.0277777777777777
  (total_cost / total_hours) = average_cost_per_hour

theorem cost_hour_excess_is_1point75 : cost_per_hour_excess 1.75 :=
by
  sorry

end cost_hour_excess_is_1point75_l1193_119384


namespace bert_made_1_dollar_l1193_119337

def bert_earnings (selling_price tax_rate markup : ℝ) : ℝ :=
  selling_price - (tax_rate * selling_price) - (selling_price - markup)

theorem bert_made_1_dollar :
  bert_earnings 90 0.1 10 = 1 :=
by 
  sorry

end bert_made_1_dollar_l1193_119337


namespace quadratic_distinct_real_roots_l1193_119346

theorem quadratic_distinct_real_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x^2 + m*x + (m + 3) = 0)) ↔ (m < -2 ∨ m > 6) := 
sorry

end quadratic_distinct_real_roots_l1193_119346


namespace min_value_theorem_l1193_119330

noncomputable def min_value (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) : ℝ :=
  1/a + 2/b

theorem min_value_theorem (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) :
  min_value a b h₀ h₁ h₂ ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_theorem_l1193_119330


namespace calculation_l1193_119313

theorem calculation :
  5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 :=
  by
    sorry

end calculation_l1193_119313


namespace cut_ribbon_l1193_119377

theorem cut_ribbon
    (length_ribbon : ℝ)
    (points : ℝ × ℝ × ℝ × ℝ × ℝ)
    (h_length : length_ribbon = 5)
    (h_points : points = (1, 2, 3, 4, 5)) :
    points.2.1 = (11 / 15) * length_ribbon :=
by
    sorry

end cut_ribbon_l1193_119377


namespace c_value_for_infinite_solutions_l1193_119392

theorem c_value_for_infinite_solutions :
  ∀ (c : ℝ), (∀ (x : ℝ), 3 * (5 + c * x) = 15 * x + 15) ↔ c = 5 :=
by
  -- Proof
  sorry

end c_value_for_infinite_solutions_l1193_119392


namespace find_k_l1193_119382

theorem find_k (k : ℝ) :
    (∀ x : ℝ, 4 * x^2 + k * x + 4 ≠ 0) → k = 8 :=
sorry

end find_k_l1193_119382


namespace negate_universal_prop_l1193_119391

theorem negate_universal_prop :
  (¬ ∀ x : ℝ, x^2 - 2*x + 2 > 0) ↔ ∃ x : ℝ, x^2 - 2*x + 2 ≤ 0 :=
sorry

end negate_universal_prop_l1193_119391


namespace complex_imaginary_unit_sum_l1193_119389

theorem complex_imaginary_unit_sum (i : ℂ) (h : i^2 = -1) : i + i^2 + i^3 = -1 := 
by sorry

end complex_imaginary_unit_sum_l1193_119389


namespace average_score_of_seniors_l1193_119305

theorem average_score_of_seniors
    (total_students : ℕ)
    (average_score_all : ℚ)
    (num_seniors num_non_seniors : ℕ)
    (mean_score_senior mean_score_non_senior : ℚ)
    (h1 : total_students = 120)
    (h2 : average_score_all = 84)
    (h3 : num_non_seniors = 2 * num_seniors)
    (h4 : mean_score_senior = 2 * mean_score_non_senior)
    (h5 : num_seniors + num_non_seniors = total_students)
    (h6 : num_seniors * mean_score_senior + num_non_seniors * mean_score_non_senior = total_students * average_score_all) :
  mean_score_senior = 126 :=
by
  sorry

end average_score_of_seniors_l1193_119305


namespace sufficient_not_necessary_l1193_119338

theorem sufficient_not_necessary (a b : ℝ) (h : b > a ∧ a > 0) : (1 / a > 1 / b) :=
by {
  sorry -- the proof steps are intentionally omitted
}

end sufficient_not_necessary_l1193_119338


namespace no_function_satisfies_condition_l1193_119342

theorem no_function_satisfies_condition :
  ¬ ∃ f : ℤ → ℤ, ∀ x y : ℤ, f (x + f y) = f x - y :=
sorry

end no_function_satisfies_condition_l1193_119342


namespace trigonometric_cos_value_l1193_119366

open Real

theorem trigonometric_cos_value (α : ℝ) (h : sin (α + π / 6) = 1 / 3) : 
  cos (2 * α - 2 * π / 3) = -7 / 9 := 
sorry

end trigonometric_cos_value_l1193_119366


namespace mouse_jump_frog_jump_diff_l1193_119358

open Nat

theorem mouse_jump_frog_jump_diff :
  let grasshopper_jump := 19
  let frog_jump := grasshopper_jump + 10
  let mouse_jump := grasshopper_jump + 30
  mouse_jump - frog_jump = 20 :=
by
  let grasshopper_jump := 19
  let frog_jump := grasshopper_jump + 10
  let mouse_jump := grasshopper_jump + 30
  have h1 : frog_jump = 29 := by decide
  have h2 : mouse_jump = 49 := by decide
  have h3 : mouse_jump - frog_jump = 20 := by decide
  exact h3

end mouse_jump_frog_jump_diff_l1193_119358


namespace original_cost_price_l1193_119306

theorem original_cost_price ( C S : ℝ )
  (h1 : S = 1.05 * C)
  (h2 : S - 3 = 1.10 * 0.95 * C)
  : C = 600 :=
sorry

end original_cost_price_l1193_119306


namespace marked_price_l1193_119344

theorem marked_price (original_price : ℝ) 
                     (discount1_rate : ℝ) 
                     (profit_rate : ℝ) 
                     (discount2_rate : ℝ)
                     (marked_price : ℝ) : 
                     original_price = 40 → 
                     discount1_rate = 0.15 → 
                     profit_rate = 0.25 → 
                     discount2_rate = 0.10 → 
                     marked_price = 47.20 := 
by
  intros h₁ h₂ h₃ h₄
  sorry

end marked_price_l1193_119344


namespace power_function_value_at_neg2_l1193_119303

theorem power_function_value_at_neg2 
  (f : ℝ → ℝ)
  (a : ℝ)
  (h1 : ∀ x : ℝ, f x = x^a)
  (h2 : f 2 = 1 / 4) 
  : f (-2) = 1 / 4 := by
  sorry

end power_function_value_at_neg2_l1193_119303


namespace percentage_answered_first_correctly_l1193_119372

-- Defining the given conditions
def percentage_answered_second_correctly : ℝ := 0.25
def percentage_answered_neither_correctly : ℝ := 0.20
def percentage_answered_both_correctly : ℝ := 0.20

-- Lean statement for the proof problem
theorem percentage_answered_first_correctly :
  ∃ a : ℝ, a + percentage_answered_second_correctly - percentage_answered_both_correctly = 0.80 ∧ a = 0.75 := by
  sorry

end percentage_answered_first_correctly_l1193_119372


namespace general_term_formula_sum_first_n_terms_l1193_119364

noncomputable def a_n (n : ℕ) : ℕ := 2^(n - 1)

def S (n : ℕ) : ℕ := n * (2^(n - 1))  -- Placeholder function for the sum of the first n terms

theorem general_term_formula (a_3_eq_2a_2 : 2^(3 - 1) = 2 * 2^(2 - 1)) (S3_eq_7 : S 3 = 7) :
  ∀ n, a_n n = 2^(n - 1) :=
sorry

def T (n : ℕ) : ℕ := 4 - ((4 + 2 * n) / 2^n) -- Placeholder function for calculating T_n

theorem sum_first_n_terms (a_3_eq_2a_2 : 2^(3 - 1) = 2 * 2^(2 - 1)) (S3_eq_7 : S 3 = 7) :
  ∀ n, T n = 4 - ((4 + 2*n) / 2^n) :=
sorry

end general_term_formula_sum_first_n_terms_l1193_119364


namespace total_fish_sold_l1193_119311

-- Define the conditions
def w1 : ℕ := 50
def w2 : ℕ := 3 * w1

-- Define the statement to prove
theorem total_fish_sold : w1 + w2 = 200 := by
  -- Insert the proof here 
  -- (proof omitted as per the instructions)
  sorry

end total_fish_sold_l1193_119311


namespace sufficient_but_not_necessary_l1193_119300

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, (0 < x ∧ x < 2) → (x < 2)) ∧ ¬(∀ x : ℝ, (0 < x ∧ x < 2) ↔ (x < 2)) :=
sorry

end sufficient_but_not_necessary_l1193_119300


namespace sequence_remainder_mod_10_l1193_119380

def T : ℕ → ℕ := sorry -- Since the actual recursive definition is part of solution steps, we abstract it.
def remainder (n k : ℕ) : ℕ := n % k

theorem sequence_remainder_mod_10 (n : ℕ) (h: n = 2023) : remainder (T n) 10 = 6 :=
by 
  sorry

end sequence_remainder_mod_10_l1193_119380


namespace geometric_sequence_a6_l1193_119317

theorem geometric_sequence_a6 (a : ℕ → ℝ) (a1 r : ℝ) (h1 : ∀ n, a n = a1 * r ^ (n - 1)) (h2 : (a 2) * (a 4) * (a 12) = 64) : a 6 = 4 :=
sorry

end geometric_sequence_a6_l1193_119317


namespace total_eggs_l1193_119363

def e0 : ℝ := 47.0
def ei : ℝ := 5.0

theorem total_eggs : e0 + ei = 52.0 := by
  sorry

end total_eggs_l1193_119363


namespace simplify_fraction_sum_l1193_119370

variable (a b c : ℝ)
variable (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)

theorem simplify_fraction_sum (x : ℝ) (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ( (x + a) ^ 2 / ((a - b) * (a - c))
  + (x + b) ^ 2 / ((b - a) * (b - c))
  + (x + c) ^ 2 / ((c - a) * (c - b)) )
  = a * x + b * x + c * x - a - b - c :=
sorry

end simplify_fraction_sum_l1193_119370
