import Mathlib

namespace total_yen_l902_90245

-- Define the given conditions in Lean 4
def bal_bahamian_dollars : ℕ := 5000
def bal_us_dollars : ℕ := 2000
def bal_euros : ℕ := 3000

def exchange_rate_bahamian_to_yen : ℝ := 122.13
def exchange_rate_us_to_yen : ℝ := 110.25
def exchange_rate_euro_to_yen : ℝ := 128.50

def check_acc1 : ℕ := 15000
def check_acc2 : ℕ := 6359
def sav_acc1 : ℕ := 5500
def sav_acc2 : ℕ := 3102

def stocks : ℕ := 200000
def bonds : ℕ := 150000
def mutual_funds : ℕ := 120000

-- Prove the total amount of yen the family has
theorem total_yen : 
  bal_bahamian_dollars * exchange_rate_bahamian_to_yen + 
  bal_us_dollars * exchange_rate_us_to_yen + 
  bal_euros * exchange_rate_euro_to_yen
  + (check_acc1 + check_acc2 + sav_acc1 + sav_acc2 : ℝ)
  + (stocks + bonds + mutual_funds : ℝ) = 1716611 := 
by
  sorry

end total_yen_l902_90245


namespace mul_pos_neg_eq_neg_l902_90232

theorem mul_pos_neg_eq_neg (a : Int) : 3 * (-2) = -6 := by
  sorry

end mul_pos_neg_eq_neg_l902_90232


namespace men_left_the_job_l902_90231

theorem men_left_the_job
    (work_rate_20men : 20 * 4 = 30)
    (work_rate_remaining : 6 * 6 = 36) :
    4 = 20 - (20 * 4) / (6 * 6)  :=
by
  sorry

end men_left_the_job_l902_90231


namespace problem_proof_l902_90296

-- Define the conditions
def a (n : ℕ) : Real := sorry  -- a is some real number, so it's non-deterministic here

def a_squared (n : ℕ) : Real := a n ^ (2 * n)  -- a^(2n)

-- Main theorem to prove
theorem problem_proof (n : ℕ) (h : a_squared n = 3) : 2 * (a n ^ (6 * n)) - 1 = 53 :=
by
  sorry  -- Proof to be completed

end problem_proof_l902_90296


namespace cos_arithmetic_sequence_result_l902_90263

-- Define an arithmetic sequence as a function
def arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem cos_arithmetic_sequence_result (a d : ℝ) 
  (h : arithmetic_seq a d 1 + arithmetic_seq a d 5 + arithmetic_seq a d 9 = 8 * Real.pi) :
  Real.cos (arithmetic_seq a d 3 + arithmetic_seq a d 7) = -1 / 2 := by
  sorry

end cos_arithmetic_sequence_result_l902_90263


namespace total_teaching_hours_l902_90283

-- Define the durations of the classes
def eduardo_math_classes : ℕ := 3
def eduardo_science_classes : ℕ := 4
def eduardo_history_classes : ℕ := 2

def math_class_duration : ℕ := 1
def science_class_duration : ℚ := 1.5
def history_class_duration : ℕ := 2

-- Define Eduardo's teaching time
def eduardo_total_time : ℚ :=
  eduardo_math_classes * math_class_duration +
  eduardo_science_classes * science_class_duration +
  eduardo_history_classes * history_class_duration

-- Define Frankie's teaching time (double the classes of Eduardo)
def frankie_total_time : ℚ :=
  2 * (eduardo_math_classes * math_class_duration) +
  2 * (eduardo_science_classes * science_class_duration) +
  2 * (eduardo_history_classes * history_class_duration)

-- Define the total teaching time for both Eduardo and Frankie
def total_teaching_time : ℚ :=
  eduardo_total_time + frankie_total_time

-- Theorem statement that both their total teaching time is 39 hours
theorem total_teaching_hours : total_teaching_time = 39 :=
by
  -- skipping the proof using sorry
  sorry

end total_teaching_hours_l902_90283


namespace alcohol_percentage_new_mixture_l902_90270

theorem alcohol_percentage_new_mixture :
  let initial_alcohol_percentage := 0.90
  let initial_solution_volume := 24
  let added_water_volume := 16
  let total_new_volume := initial_solution_volume + added_water_volume
  let initial_alcohol_amount := initial_solution_volume * initial_alcohol_percentage
  let new_alcohol_percentage := (initial_alcohol_amount / total_new_volume) * 100
  new_alcohol_percentage = 54 := by
    sorry

end alcohol_percentage_new_mixture_l902_90270


namespace decagon_side_length_in_rectangle_l902_90204

theorem decagon_side_length_in_rectangle
  (AB CD : ℝ)
  (AE FB : ℝ)
  (s : ℝ)
  (cond1 : AB = 10)
  (cond2 : CD = 15)
  (cond3 : AE = 5)
  (cond4 : FB = 5)
  (regular_decagon : ℝ → Prop)
  (h : regular_decagon s) : 
  s = 5 * (Real.sqrt 2 - 1) :=
by 
  sorry

end decagon_side_length_in_rectangle_l902_90204


namespace best_selling_price_70_l902_90252

-- Definitions for the conditions in the problem
def purchase_price : ℕ := 40
def initial_selling_price : ℕ := 50
def initial_sales_volume : ℕ := 50

-- The profit function
def profit (x : ℕ) : ℕ :=
(50 + x - purchase_price) * (initial_sales_volume - x)

-- The problem statement to be proved
theorem best_selling_price_70 :
  ∃ x : ℕ, 0 < x ∧ x < 50 ∧ profit x = 900 ∧ (initial_selling_price + x) = 70 :=
by
  sorry

end best_selling_price_70_l902_90252


namespace angle_A_is_120_max_sin_B_plus_sin_C_l902_90293

-- Define the measures in degrees using real numbers
variable (a b c R : Real)
variable (A B C : ℝ) (sin cos : ℝ → ℝ)

-- Question 1: Prove A = 120 degrees given the initial condition
theorem angle_A_is_120
  (H1 : 2 * a * (sin A) = (2 * b + c) * (sin B) + (2 * c + b) * (sin C)) :
  A = 120 :=
by
  sorry

-- Question 2: Given the angles sum to 180 degrees and A = 120 degrees, prove the max value of sin B + sin C is 1
theorem max_sin_B_plus_sin_C
  (H2 : A + B + C = 180)
  (H3 : A = 120) :
  (sin B) + (sin C) ≤ 1 :=
by
  sorry

end angle_A_is_120_max_sin_B_plus_sin_C_l902_90293


namespace maximum_sum_of_factors_exists_maximum_sum_of_factors_l902_90275

theorem maximum_sum_of_factors {A B C : ℕ} (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C)
  (h4 : A * B * C = 2023) : A + B + C ≤ 297 :=
sorry

theorem exists_maximum_sum_of_factors : ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2023 ∧ A + B + C = 297 :=
sorry

end maximum_sum_of_factors_exists_maximum_sum_of_factors_l902_90275


namespace largest_c_value_l902_90257

theorem largest_c_value (c : ℝ) (h : -2 * c^2 + 8 * c - 6 ≥ 0) : c ≤ 3 := 
sorry

end largest_c_value_l902_90257


namespace largest_angle_in_triangle_l902_90286

theorem largest_angle_in_triangle : 
  ∀ (A B C : ℝ), A + B + C = 180 ∧ A + B = 105 ∧ (A = B + 40)
  → (C = 75) :=
by
  sorry

end largest_angle_in_triangle_l902_90286


namespace max_sum_of_squares_eq_100_l902_90235

theorem max_sum_of_squares_eq_100 : 
  ∃ (x y : ℤ), x^2 + y^2 = 100 ∧ 
  (∀ (x y : ℤ), x^2 + y^2 = 100 → x + y ≤ 14) ∧ 
  (∃ (x y : ℕ), x^2 + y^2 = 100 ∧ x + y = 14) :=
by {
  sorry
}

end max_sum_of_squares_eq_100_l902_90235


namespace cost_price_proof_l902_90217

noncomputable def cost_price_per_bowl : ℚ := 1400 / 103

theorem cost_price_proof
  (total_bowls: ℕ) (sold_bowls: ℕ) (selling_price_per_bowl: ℚ)
  (percentage_gain: ℚ) 
  (total_bowls_eq: total_bowls = 110)
  (sold_bowls_eq: sold_bowls = 100)
  (selling_price_per_bowl_eq: selling_price_per_bowl = 14)
  (percentage_gain_eq: percentage_gain = 300 / 11) :
  (selling_price_per_bowl * sold_bowls - (sold_bowls + 3) * (selling_price_per_bowl / (3 * percentage_gain / 100))) = cost_price_per_bowl :=
by
  sorry

end cost_price_proof_l902_90217


namespace area_of_triangle_ABC_l902_90262

noncomputable def area_triangle_ABC (AF BE : ℝ) (angle_FGB : ℝ) : ℝ :=
  let FG := AF / 3
  let BG := (2 / 3) * BE
  let area_FGB := (1 / 2) * FG * BG * Real.sin angle_FGB
  6 * area_FGB

theorem area_of_triangle_ABC
  (AF BE : ℕ) (hAF : AF = 10) (hBE : BE = 15)
  (angle_FGB : ℝ) (h_angle_FGB : angle_FGB = Real.pi / 3) :
  area_triangle_ABC AF BE angle_FGB = 50 * Real.sqrt 3 :=
by
  simp [area_triangle_ABC, hAF, hBE, h_angle_FGB]
  sorry

end area_of_triangle_ABC_l902_90262


namespace eval_polynomial_l902_90271

theorem eval_polynomial (x : ℝ) (h : x^2 - 3 * x - 9 = 0) : x^3 - 3 * x^2 - 9 * x + 27 = 27 := 
by
  sorry

end eval_polynomial_l902_90271


namespace parabola_transformation_l902_90243

-- Defining the original parabola
def original_parabola (x : ℝ) : ℝ :=
  3 * x^2

-- Condition: Transformation 1 -> Translation 4 units to the right
def translated_right_parabola (x : ℝ) : ℝ :=
  original_parabola (x - 4)

-- Condition: Transformation 2 -> Translation 1 unit upwards
def translated_up_parabola (x : ℝ) : ℝ :=
  translated_right_parabola x + 1

-- Statement that needs to be proved
theorem parabola_transformation :
  ∀ x : ℝ, translated_up_parabola x = 3 * (x - 4)^2 + 1 :=
by
  intros x
  sorry

end parabola_transformation_l902_90243


namespace container_weight_l902_90216

noncomputable def weight_in_pounds : ℝ := 57 + 3/8
noncomputable def weight_in_ounces : ℝ := weight_in_pounds * 16
noncomputable def number_of_containers : ℝ := 7
noncomputable def ounces_per_container : ℝ := weight_in_ounces / number_of_containers

theorem container_weight :
  ounces_per_container = 131.142857 :=
by sorry

end container_weight_l902_90216


namespace x1_x2_lt_one_l902_90274

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * x - log x
noncomputable def g (x : ℝ) : ℝ := x / exp x

theorem x1_x2_lt_one (k : ℝ) (x1 x2 : ℝ) (h : f x1 1 + g x1 - k = 0) (h2 : f x2 1 + g x2 - k = 0) (hx1 : 0 < x1) (hx2 : x1 < x2) : x1 * x2 < 1 :=
by
  sorry

end x1_x2_lt_one_l902_90274


namespace B_is_werewolf_l902_90272

def is_werewolf (x : Type) : Prop := sorry
def is_knight (x : Type) : Prop := sorry
def is_liar (x : Type) : Prop := sorry

variables (A B : Type)

-- Conditions
axiom one_is_werewolf : is_werewolf A ∨ is_werewolf B
axiom only_one_werewolf : ¬ (is_werewolf A ∧ is_werewolf B)
axiom A_statement : is_werewolf A → is_knight A
axiom B_statement : is_werewolf B → is_liar B

theorem B_is_werewolf : is_werewolf B := 
by
  sorry

end B_is_werewolf_l902_90272


namespace minimum_value_proof_l902_90215

noncomputable def minValue (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 2) : ℝ := 
  (x + 8 * y) / (x * y)

theorem minimum_value_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 2) : 
  minValue x y hx hy h = 9 := 
by
  sorry

end minimum_value_proof_l902_90215


namespace no_real_roots_of_quadratic_l902_90209

def quadratic (a b c : ℝ) : ℝ × ℝ × ℝ := (a^2, b^2 + a^2 - c^2, b^2)

def discriminant (A B C : ℝ) : ℝ := B^2 - 4 * A * C

theorem no_real_roots_of_quadratic (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : c > 0)
  (h3 : |a - b| < c)
  : (discriminant (a^2) (b^2 + a^2 - c^2) (b^2)) < 0 :=
sorry

end no_real_roots_of_quadratic_l902_90209


namespace contact_alignment_possible_l902_90268

/-- A vacuum tube has seven contacts arranged in a circle and is inserted into a socket that has seven holes.
Prove that it is possible to number the tube's contacts and the socket's holes in such a way that:
in any insertion of the tube, at least one contact will align with its corresponding hole (i.e., the hole with the same number). -/
theorem contact_alignment_possible : ∃ (f : Fin 7 → Fin 7), ∀ (rotation : Fin 7 → Fin 7), ∃ k : Fin 7, f k = rotation k := 
sorry

end contact_alignment_possible_l902_90268


namespace min_choir_members_l902_90260

theorem min_choir_members (n : ℕ) : 
  (∀ (m : ℕ), m % 9 = 0 ∧ m % 10 = 0 ∧ m % 11 = 0 → m ≥ n) → 
  n = 990 :=
by
  sorry

end min_choir_members_l902_90260


namespace linear_equation_solution_l902_90222

theorem linear_equation_solution (m n : ℤ) (x y : ℤ)
  (h1 : x + 2 * y = 5)
  (h2 : x + y = 7)
  (h3 : x = -m)
  (h4 : y = -n) :
  (3 * m + 2 * n) / (5 * m - n) = 11 / 14 :=
by
  sorry

end linear_equation_solution_l902_90222


namespace total_dolls_48_l902_90208

def dolls_sister : ℕ := 8

def dolls_hannah : ℕ := 5 * dolls_sister

def total_dolls : ℕ := dolls_hannah + dolls_sister

theorem total_dolls_48 : total_dolls = 48 := 
by
  unfold total_dolls dolls_hannah dolls_sister
  rfl

end total_dolls_48_l902_90208


namespace normal_trip_distance_l902_90273

variable (S D : ℝ)

-- Conditions
axiom h1 : D = 3 * S
axiom h2 : D + 50 = 5 * S

theorem normal_trip_distance : D = 75 :=
by
  sorry

end normal_trip_distance_l902_90273


namespace count_two_digit_integers_remainder_3_div_9_l902_90289

theorem count_two_digit_integers_remainder_3_div_9 :
  ∃ (N : ℕ), N = 10 ∧ ∀ k, (10 ≤ 9 * k + 3 ∧ 9 * k + 3 < 100) ↔ (1 ≤ k ∧ k ≤ 10) :=
by
  sorry

end count_two_digit_integers_remainder_3_div_9_l902_90289


namespace arithmetic_sequence_nth_term_l902_90200

theorem arithmetic_sequence_nth_term (a₁ : ℤ) (d : ℤ) (n : ℕ) :
  (a₁ = 11) →
  (d = -3) →
  (-49 = a₁ + (n - 1) * d) →
  (n = 21) :=
by 
  intros h₁ h₂ h₃
  sorry

end arithmetic_sequence_nth_term_l902_90200


namespace length_of_RS_l902_90221

-- Define the lengths of the edges of the tetrahedron
def edge_lengths : List ℕ := [9, 16, 22, 31, 39, 48]

-- Given the edge PQ has length 48
def PQ_length : ℕ := 48

-- We need to prove that the length of edge RS is 9
theorem length_of_RS :
  ∃ (RS : ℕ), RS = 9 ∧
  ∃ (PR QR PS SQ : ℕ),
  [PR, QR, PS, SQ] ⊆ edge_lengths ∧
  PR + QR > PQ_length ∧
  PR + PQ_length > QR ∧
  QR + PQ_length > PR ∧
  PS + SQ > PQ_length ∧
  PS + PQ_length > SQ ∧
  SQ + PQ_length > PS :=
by
  sorry

end length_of_RS_l902_90221


namespace jade_more_transactions_l902_90202

theorem jade_more_transactions (mabel_transactions : ℕ) (anthony_percentage : ℕ) (cal_fraction_numerator : ℕ) 
  (cal_fraction_denominator : ℕ) (jade_transactions : ℕ) (h1 : mabel_transactions = 90) 
  (h2 : anthony_percentage = 10) (h3 : cal_fraction_numerator = 2) (h4 : cal_fraction_denominator = 3) 
  (h5 : jade_transactions = 83) :
  jade_transactions - (2 * (90 + (90 * 10 / 100)) / 3) = 17 := 
by
  sorry

end jade_more_transactions_l902_90202


namespace speed_of_car_in_second_hour_l902_90201

theorem speed_of_car_in_second_hour
(speed_in_first_hour : ℝ)
(average_speed : ℝ)
(total_time : ℝ)
(speed_in_second_hour : ℝ)
(h1 : speed_in_first_hour = 100)
(h2 : average_speed = 65)
(h3 : total_time = 2)
(h4 : average_speed = (speed_in_first_hour + speed_in_second_hour) / total_time) :
  speed_in_second_hour = 30 :=
by {
  sorry
}

end speed_of_car_in_second_hour_l902_90201


namespace min_value_one_over_a_plus_two_over_b_l902_90230

/-- Given a > 0, b > 0, 2a + b = 1, prove that the minimum value of (1/a) + (2/b) is 8 --/
theorem min_value_one_over_a_plus_two_over_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (1 / a) + (2 / b) ≥ 8 :=
sorry

end min_value_one_over_a_plus_two_over_b_l902_90230


namespace original_number_of_coins_in_first_pile_l902_90223

noncomputable def originalCoinsInFirstPile (x y z : ℕ) : ℕ :=
  if h : (2 * (x - y) = 16) ∧ (2 * y - z = 16) ∧ (2 * z - (x + y) = 16) then x else 0

theorem original_number_of_coins_in_first_pile (x y z : ℕ) (h1 : 2 * (x - y) = 16) 
                                              (h2 : 2 * y - z = 16) 
                                              (h3 : 2 * z - (x + y) = 16) : x = 22 :=
by sorry

end original_number_of_coins_in_first_pile_l902_90223


namespace find_n_l902_90214

theorem find_n (a b n : ℕ) (k l m : ℤ) 
  (ha : a % n = 2) 
  (hb : b % n = 3) 
  (h_ab : a > b) 
  (h_ab_mod : (a - b) % n = 5) : 
  n = 6 := 
sorry

end find_n_l902_90214


namespace find_square_side_length_l902_90206

theorem find_square_side_length
  (a CF AE : ℝ)
  (h_CF : CF = 2 * a)
  (h_AE : AE = 3.5 * a)
  (h_sum : CF + AE = 91) :
  a = 26 := by
  sorry

end find_square_side_length_l902_90206


namespace sum_of_squares_base_b_l902_90203

theorem sum_of_squares_base_b (b : ℕ) (h : (b + 4)^2 + (b + 8)^2 + (2 * b)^2 = 2 * b^3 + 8 * b^2 + 5 * b) :
  (4 * b + 12 : ℕ) = 62 :=
by
  sorry

end sum_of_squares_base_b_l902_90203


namespace length_of_tunnel_l902_90250

theorem length_of_tunnel
    (length_of_train : ℕ)
    (speed_kmh : ℕ)
    (crossing_time_seconds : ℕ)
    (distance_covered : ℕ)
    (length_of_tunnel : ℕ) :
    length_of_train = 1200 →
    speed_kmh = 96 →
    crossing_time_seconds = 90 →
    distance_covered = (speed_kmh * 1000 / 3600) * crossing_time_seconds →
    length_of_train + length_of_tunnel = distance_covered →
    length_of_tunnel = 6000 :=
by
  sorry

end length_of_tunnel_l902_90250


namespace find_t_value_l902_90205

theorem find_t_value (k t : ℤ) (h1 : 0 < k) (h2 : k < 10) (h3 : 0 < t) (h4 : t < 10) : t = 6 :=
by
  sorry

end find_t_value_l902_90205


namespace nails_for_smaller_planks_l902_90284

def total_large_planks := 13
def nails_per_plank := 17
def total_nails := 229

def nails_for_large_planks : ℕ :=
  total_large_planks * nails_per_plank

theorem nails_for_smaller_planks :
  total_nails - nails_for_large_planks = 8 :=
by
  -- Proof goes here
  sorry

end nails_for_smaller_planks_l902_90284


namespace total_pints_l902_90269

-- Define the given conditions as constants
def annie_picked : Int := 8
def kathryn_picked : Int := annie_picked + 2
def ben_picked : Int := kathryn_picked - 3

-- State the main theorem to prove
theorem total_pints : annie_picked + kathryn_picked + ben_picked = 25 := by
  sorry

end total_pints_l902_90269


namespace grandson_age_is_5_l902_90224

-- Definitions based on the conditions
def grandson_age_months_eq_grandmother_years (V B : ℕ) : Prop := B = 12 * V
def combined_age_eq_65 (V B : ℕ) : Prop := B + V = 65

-- Main theorem stating that under these conditions, the grandson's age is 5 years
theorem grandson_age_is_5 (V B : ℕ) (h₁ : grandson_age_months_eq_grandmother_years V B) (h₂ : combined_age_eq_65 V B) : V = 5 :=
by sorry

end grandson_age_is_5_l902_90224


namespace probability_consecutive_computer_scientists_l902_90278

theorem probability_consecutive_computer_scientists :
  let n := 12
  let k := 5
  let total_permutations := Nat.factorial (n - 1)
  let consecutive_permutations := Nat.factorial (7) * Nat.factorial (5)
  let probability := consecutive_permutations / total_permutations
  probability = (1 / 66) :=
by
  sorry

end probability_consecutive_computer_scientists_l902_90278


namespace dogs_prevent_wolf_escape_l902_90227

theorem dogs_prevent_wolf_escape
  (wolf_speed dog_speed : ℝ)
  (at_center: True)
  (dogs_at_vertices: True)
  (wolf_all_over_field: True)
  (dogs_on_perimeter: True)
  (wolf_handles_one_dog: ∀ (d : ℕ), d = 1 → True)
  (wolf_handles_two_dogs: ∀ (d : ℕ), d = 2 → False)
  (dog_faster_than_wolf: dog_speed = 1.5 * wolf_speed) : 
  ∀ (wolf_position : ℝ × ℝ) (boundary_position : ℝ × ℝ), 
  wolf_position != boundary_position → dog_speed > wolf_speed → 
  False := 
by sorry

end dogs_prevent_wolf_escape_l902_90227


namespace monthlyShoeSales_l902_90297

-- Defining the conditions
def pairsSoldLastWeek := 27
def pairsSoldThisWeek := 12
def pairsNeededToMeetGoal := 41

-- Defining the question as a statement to prove
theorem monthlyShoeSales : pairsSoldLastWeek + pairsSoldThisWeek + pairsNeededToMeetGoal = 80 := by
  sorry

end monthlyShoeSales_l902_90297


namespace minimum_x_for_g_maximum_l902_90249

theorem minimum_x_for_g_maximum :
  ∃ x > 0, ∀ k m: ℤ, (x = 1440 * k + 360 ∧ x = 2520 * m + 630) -> x = 7560 :=
by
  sorry

end minimum_x_for_g_maximum_l902_90249


namespace distance_from_Q_to_AD_l902_90287

-- Define the square $ABCD$ with side length 6
def square_ABCD (A B C D : ℝ × ℝ) : Prop :=
  A = (0, 6) ∧ B = (6, 6) ∧ C = (6, 0) ∧ D = (0, 0)

-- Define point $N$ as the midpoint of $\overline{CD}$
def midpoint_CD (C D N : ℝ × ℝ) : Prop :=
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

-- Define the intersection condition of the circles centered at $N$ and $A$
def intersect_circles (N A Q D : ℝ × ℝ) : Prop :=
  (Q = D ∨ (∃ r₁ r₂, (Q.1 - N.1)^2 + Q.2^2 = r₁ ∧ Q.1^2 + (Q.2 - A.2)^2 = r₂))

-- Prove the distance from $Q$ to $\overline{AD}$ equals 12/5
theorem distance_from_Q_to_AD (A B C D N Q : ℝ × ℝ)
  (h_square : square_ABCD A B C D)
  (h_midpoint : midpoint_CD C D N)
  (h_intersect : intersect_circles N A Q D) :
  Q.2 = 12 / 5 :=
sorry

end distance_from_Q_to_AD_l902_90287


namespace convince_the_king_l902_90246

/-- Define the types of inhabitants -/
inductive Inhabitant
| Knight
| Liar
| Normal

/-- Define the king's preference -/
def K (inhabitant : Inhabitant) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => False
  | Inhabitant.Liar => False
  | Inhabitant.Normal => True

/-- All knights tell the truth -/
def tells_truth (inhabitant : Inhabitant) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => True
  | Inhabitant.Liar => False
  | Inhabitant.Normal => False

/-- All liars always lie -/
def tells_lie (inhabitant : Inhabitant) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => False
  | Inhabitant.Liar => True
  | Inhabitant.Normal => False

/-- Normal persons can tell both truths and lies -/
def can_tell_both (inhabitant : Inhabitant) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => False
  | Inhabitant.Liar => False
  | Inhabitant.Normal => True

/-- Prove there exists a true statement and a false statement to convince the king -/
theorem convince_the_king (p : Inhabitant) :
  (∃ S : Prop, (S ↔ tells_truth p) ∧ K p) ∧ (∃ S' : Prop, (¬ S' ↔ tells_lie p) ∧ K p) :=
by
  sorry

end convince_the_king_l902_90246


namespace find_x_for_g_statement_l902_90211

noncomputable def g (x : ℝ) : ℝ := (x + 4) ^ (1/3) / 5 ^ (1/3)

theorem find_x_for_g_statement (x : ℝ) : g (3 * x) = 3 * g x ↔ x = -13 / 3 := by
  sorry

end find_x_for_g_statement_l902_90211


namespace quadruple_exists_unique_l902_90237

def digits (x : Nat) : Prop := x ≤ 9

theorem quadruple_exists_unique :
  ∃ (A B C D: Nat),
    digits A ∧ digits B ∧ digits C ∧ digits D ∧
    A > B ∧ B > C ∧ C > D ∧
    (A * 1000 + B * 100 + C * 10 + D) -
    (D * 1000 + C * 100 + B * 10 + A) =
    (B * 1000 + D * 100 + A * 10 + C) ∧
    (A, B, C, D) = (7, 6, 4, 1) :=
by
  sorry

end quadruple_exists_unique_l902_90237


namespace max_val_proof_l902_90225

noncomputable def max_val (p q r x y z : ℝ) : ℝ :=
  1 / (p + q) + 1 / (p + r) + 1 / (q + r) + 1 / (x + y) + 1 / (x + z) + 1 / (y + z)

theorem max_val_proof {p q r x y z : ℝ}
  (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_pos_r : 0 < r)
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_sum_pqr : p + q + r = 2) (h_sum_xyz : x + y + z = 1) :
  max_val p q r x y z = 27 / 4 :=
sorry

end max_val_proof_l902_90225


namespace measure_of_angle_C_range_of_sum_ab_l902_90207

-- Proof problem (1): Prove the measure of angle C
theorem measure_of_angle_C (a b c : ℝ) (A B C : ℝ) 
  (h1 : 2 * c * Real.sin C = (2 * b + a) * Real.sin B + (2 * a - 3 * b) * Real.sin A) :
  C = Real.pi / 3 := by 
  sorry

-- Proof problem (2): Prove the range of possible values of a + b
theorem range_of_sum_ab (a b : ℝ) (c : ℝ) (h1 : c = 4) (h2 : 16 = a^2 + b^2 - a * b) :
  4 < a + b ∧ a + b ≤ 8 := by 
  sorry

end measure_of_angle_C_range_of_sum_ab_l902_90207


namespace sum_of_digits_of_N_l902_90229

open Nat

theorem sum_of_digits_of_N (T : ℕ) (hT : T = 3003) :
  ∃ N : ℕ, (N * (N + 1)) / 2 = T ∧ (digits 10 N).sum = 14 :=
by 
  sorry

end sum_of_digits_of_N_l902_90229


namespace range_of_a_for_inequality_l902_90239

open Real

theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, ¬(a*x^2 - |x + 1| + 2*a < 0)) ↔ a ≥ (sqrt 3 + 1) / 4 := 
by
  sorry

end range_of_a_for_inequality_l902_90239


namespace carnival_total_cost_l902_90254

def morning_costs (under18_cost over18_cost : ℕ) : ℕ :=
  under18_cost + over18_cost

def afternoon_costs (under18_cost over18_cost : ℕ) : ℕ :=
  under18_cost + 1 + over18_cost + 1

noncomputable def mara_cost : ℕ :=
  let bumper_car_cost := morning_costs 2 0 + afternoon_costs 2 0
  let ferris_wheel_cost := morning_costs 5 5 + 5
  bumper_car_cost + ferris_wheel_cost

noncomputable def riley_cost : ℕ :=
  let space_shuttle_cost := morning_costs 0 5 + afternoon_costs 0 5
  let ferris_wheel_cost := morning_costs 0 6 + (6 + 1)
  space_shuttle_cost + ferris_wheel_cost

theorem carnival_total_cost :
  mara_cost + riley_cost = 61 := by
  sorry

end carnival_total_cost_l902_90254


namespace find_angle_C_l902_90294

variable {A B C a b c : ℝ}

theorem find_angle_C (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : 0 < C) (h6 : C < π)
  (h7 : A + B + C = π) (h8 : a > 0) (h9 : b > 0) (h10 : c > 0) 
  (h11 : (a + b - c) * (a + b + c) = a * b) : C = 2 * π / 3 :=
by
  sorry

end find_angle_C_l902_90294


namespace percentage_increase_l902_90291

variable (x y p : ℝ)

theorem percentage_increase (h : x = y + (p / 100) * y) : p = 100 * ((x - y) / y) := 
by 
  sorry

end percentage_increase_l902_90291


namespace walter_fraction_fewer_bananas_l902_90256

theorem walter_fraction_fewer_bananas (f : ℚ) (h1 : 56 + (56 - 56 * f) = 98) : f = 1 / 4 :=
sorry

end walter_fraction_fewer_bananas_l902_90256


namespace exists_x0_in_interval_l902_90266

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem exists_x0_in_interval :
  ∃ x0 : ℝ, 0 < x0 ∧ x0 < 4 ∧ f x0 = 0 ∧ 2 < x0 ∧ x0 < 3 :=
sorry

end exists_x0_in_interval_l902_90266


namespace tan_symmetric_about_k_pi_over_2_min_value_cos2x_plus_sinx_l902_90210

theorem tan_symmetric_about_k_pi_over_2 (k : ℤ) : 
  (∀ x : ℝ, Real.tan (x + k * Real.pi / 2) = Real.tan x) := 
sorry

theorem min_value_cos2x_plus_sinx : 
  (∀ x : ℝ, Real.cos x ^ 2 + Real.sin x ≥ -1) ∧ (∃ x : ℝ, Real.cos x ^ 2 + Real.sin x = -1) :=
sorry

end tan_symmetric_about_k_pi_over_2_min_value_cos2x_plus_sinx_l902_90210


namespace arithmetic_mean_correct_l902_90255

noncomputable def arithmetic_mean (n : ℕ) (h : n > 1) : ℝ :=
  let one_minus_one_div_n := 1 - (1 / n : ℝ)
  let rest_ones := (n - 1 : ℕ) • 1
  let total_sum : ℝ := rest_ones + one_minus_one_div_n
  total_sum / n

theorem arithmetic_mean_correct (n : ℕ) (h : n > 1) :
  arithmetic_mean n h = 1 - (1 / (n * n : ℝ)) := sorry

end arithmetic_mean_correct_l902_90255


namespace nm_odd_if_squares_sum_odd_l902_90242

theorem nm_odd_if_squares_sum_odd
  (n m : ℤ)
  (h : (n^2 + m^2) % 2 = 1) :
  (n * m) % 2 = 1 :=
sorry

end nm_odd_if_squares_sum_odd_l902_90242


namespace chemical_reaction_l902_90267

def reaction_balanced (koh nh4i ki nh3 h2o : ℕ) : Prop :=
  koh = nh4i ∧ nh4i = ki ∧ ki = nh3 ∧ nh3 = h2o

theorem chemical_reaction
  (KOH NH4I : ℕ)
  (h1 : KOH = 3)
  (h2 : NH4I = 3)
  (balanced : reaction_balanced KOH NH4I 3 3 3) :
  (∃ (NH3 KI H2O : ℕ),
    NH3 = 3 ∧ KI = 3 ∧ H2O = 3 ∧ 
    NH3 = NH4I - NH4I ∧
    KI = KOH - KOH ∧
    H2O = KOH - KOH) ∧
  (KOH = NH4I) := 
by sorry

end chemical_reaction_l902_90267


namespace sum_of_three_digit_even_naturals_correct_l902_90219

noncomputable def sum_of_three_digit_even_naturals : ℕ := 
  let a := 100
  let l := 998
  let d := 2
  let n := (l - a) / d + 1
  n / 2 * (a + l)

theorem sum_of_three_digit_even_naturals_correct : 
  sum_of_three_digit_even_naturals = 247050 := by 
  sorry

end sum_of_three_digit_even_naturals_correct_l902_90219


namespace lesser_of_two_numbers_l902_90240

theorem lesser_of_two_numbers (x y : ℝ) (h1 : x + y = 50) (h2 : x - y = 6) : y = 22 :=
by
  sorry

end lesser_of_two_numbers_l902_90240


namespace seven_times_one_fifth_cubed_l902_90251

theorem seven_times_one_fifth_cubed : 7 * (1 / 5) ^ 3 = 7 / 125 := 
by 
  sorry

end seven_times_one_fifth_cubed_l902_90251


namespace ball_hits_ground_at_5_over_2_l902_90299

noncomputable def ball_height (t : ℝ) : ℝ := -16 * t^2 + 40 * t + 60

theorem ball_hits_ground_at_5_over_2 :
  ∃ t : ℝ, t = 5 / 2 ∧ ball_height t = 0 :=
sorry

end ball_hits_ground_at_5_over_2_l902_90299


namespace ab_value_l902_90259

theorem ab_value (a b : ℝ) (h1 : a = Real.exp (2 - a)) (h2 : 1 + Real.log b = Real.exp (1 - Real.log b)) : 
  a * b = Real.exp 1 :=
sorry

end ab_value_l902_90259


namespace moles_CH3COOH_equiv_l902_90282

theorem moles_CH3COOH_equiv (moles_NaOH moles_NaCH3COO : ℕ)
    (h1 : moles_NaOH = 1)
    (h2 : moles_NaCH3COO = 1) :
    moles_NaOH = moles_NaCH3COO :=
by
  sorry

end moles_CH3COOH_equiv_l902_90282


namespace price_comparison_2010_l902_90220

def X_initial : ℝ := 4.20
def Y_initial : ℝ := 6.30
def r_X : ℝ := 0.45
def r_Y : ℝ := 0.20
def n : ℕ := 9

theorem price_comparison_2010: 
  X_initial + r_X * n > Y_initial + r_Y * n := by
  sorry

end price_comparison_2010_l902_90220


namespace apples_found_l902_90264

theorem apples_found (start_apples : ℕ) (end_apples : ℕ) (h_start : start_apples = 7) (h_end : end_apples = 81) : 
  end_apples - start_apples = 74 := 
by 
  sorry

end apples_found_l902_90264


namespace intersection_M_N_l902_90285

def M : Set ℝ := { x | -5 < x ∧ x < 3 }
def N : Set ℝ := { x | -2 < x ∧ x < 4 }

theorem intersection_M_N : M ∩ N = { x | -2 < x ∧ x < 3 } := 
by sorry

end intersection_M_N_l902_90285


namespace intersect_inverse_l902_90290

theorem intersect_inverse (c d : ℤ) (h1 : 2 * (-4) + c = d) (h2 : 2 * d + c = -4) : d = -4 := 
by
  sorry

end intersect_inverse_l902_90290


namespace percentage_of_students_70_79_l902_90280

def tally_90_100 := 6
def tally_80_89 := 9
def tally_70_79 := 8
def tally_60_69 := 6
def tally_50_59 := 3
def tally_below_50 := 1

def total_students := tally_90_100 + tally_80_89 + tally_70_79 + tally_60_69 + tally_50_59 + tally_below_50

theorem percentage_of_students_70_79 : (tally_70_79 : ℚ) / total_students = 8 / 33 :=
by
  sorry

end percentage_of_students_70_79_l902_90280


namespace max_imaginary_part_angle_l902_90236

def poly (z : Complex) : Complex := z^6 - z^4 + z^2 - 1

theorem max_imaginary_part_angle :
  ∃ θ : Real, θ = 45 ∧ 
  (∃ z : Complex, poly z = 0 ∧ ∀ w : Complex, poly w = 0 → w.im ≤ z.im)
:= sorry

end max_imaginary_part_angle_l902_90236


namespace boat_crossing_l902_90288

theorem boat_crossing (students teacher trips people_in_boat : ℕ) (h_students : students = 13) (h_teacher : teacher = 1) (h_boat_capacity : people_in_boat = 5) :
  trips = (students + teacher + people_in_boat - 1) / (people_in_boat - 1) :=
by
  sorry

end boat_crossing_l902_90288


namespace two_quadratic_residue_mod_p_p_congruent_one_mod_2_pow_n_plus_two_l902_90279

variable (n : ℕ) (F : ℕ → ℕ) (p : ℕ)

-- Condition: F_n = 2^{2^n} + 1
def F_n (n : ℕ) : ℕ := 2^(2^n) + 1

-- Assuming n >= 2
def n_ge_two (n : ℕ) : Prop := n ≥ 2

-- Assuming p is a prime factor of F_n
def prime_factor_of_F_n (p : ℕ) (n : ℕ) : Prop := p ∣ (F_n n) ∧ Prime p

-- Part a: 2 is a quadratic residue modulo p
theorem two_quadratic_residue_mod_p (n : ℕ) (p : ℕ) (hn : n_ge_two n) (hp : prime_factor_of_F_n p n) :
  ∃ x : ℕ, x^2 ≡ 2 [MOD p] := sorry

-- Part b: p ≡ 1 (mod 2^(n+2))
theorem p_congruent_one_mod_2_pow_n_plus_two (n : ℕ) (p : ℕ) (hn : n_ge_two n) (hp : prime_factor_of_F_n p n) :
  p ≡ 1 [MOD 2^(n+2)] := sorry

end two_quadratic_residue_mod_p_p_congruent_one_mod_2_pow_n_plus_two_l902_90279


namespace inequality_solution_set_l902_90277

theorem inequality_solution_set :
  {x : ℝ | (x^2 + 2*x + 2) / (x + 2) > 1} = {x : ℝ | (-2 < x ∧ x < -1) ∨ (0 < x)} :=
sorry

end inequality_solution_set_l902_90277


namespace maximum_value_quadratic_l902_90265

def quadratic_function (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

theorem maximum_value_quadratic :
  ∃ x : ℝ, quadratic_function x = 2 ∧ ∀ y : ℝ, quadratic_function y ≤ 2 :=
sorry

end maximum_value_quadratic_l902_90265


namespace factorable_polynomial_l902_90295

theorem factorable_polynomial (m : ℤ) :
  (∃ A B C D E F : ℤ, 
    (A * D = 1 ∧ E + B = 4 ∧ C + F = 2 ∧ F + 3 * E + C = m + m^2 - 16)
    ∧ ((A * x + B * y + C) * (D * x + E * y + F) = x^2 + 4 * x * y + 2 * x + m * y + m^2 - 16)) ↔
  (m = 5 ∨ m = -6) :=
by
  sorry

end factorable_polynomial_l902_90295


namespace new_average_daily_production_l902_90248

theorem new_average_daily_production (n : ℕ) (avg_past : ℕ) (production_today : ℕ) (new_avg : ℕ)
  (h1 : n = 9)
  (h2 : avg_past = 50)
  (h3 : production_today = 100)
  (h4 : new_avg = (avg_past * n + production_today) / (n + 1)) :
  new_avg = 55 :=
by
  -- Using the provided conditions, it will be shown in the proof stage that new_avg equals 55
  sorry

end new_average_daily_production_l902_90248


namespace number_of_students_in_class_l902_90213

theorem number_of_students_in_class
  (x : ℕ)
  (S : ℝ)
  (incorrect_score correct_score : ℝ)
  (incorrect_score_mistake : incorrect_score = 85)
  (correct_score_corrected : correct_score = 78)
  (average_difference : ℝ)
  (average_difference_value : average_difference = 0.75)
  (test_attendance : ℕ)
  (test_attendance_value : test_attendance = x - 3)
  (average_difference_condition : (S + incorrect_score) / test_attendance - (S + correct_score) / test_attendance = average_difference) :
  x = 13 :=
by
  sorry

end number_of_students_in_class_l902_90213


namespace units_digit_sum_squares_of_odd_integers_l902_90241

theorem units_digit_sum_squares_of_odd_integers :
  let first_2005_odd_units := [802, 802, 401] -- counts for units 1, 9, 5 respectively
  let extra_squares_last_6 := [9, 1, 3, 9, 5, 9] -- units digits of the squares of the last 6 numbers
  let total_sum :=
        (first_2005_odd_units[0] * 1 + 
         first_2005_odd_units[1] * 9 + 
         first_2005_odd_units[2] * 5) +
        (extra_squares_last_6.sum)
  (total_sum % 10) = 1 :=
by
  sorry

end units_digit_sum_squares_of_odd_integers_l902_90241


namespace complement_union_eq_l902_90228

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x ^ 2 - 4 * x + 3 = 0}

theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 0} :=
by
  sorry

end complement_union_eq_l902_90228


namespace find_m_l902_90234

theorem find_m (m : ℤ) (h : 3 ∈ ({1, m + 2} : Set ℤ)) : m = 1 :=
sorry

end find_m_l902_90234


namespace total_litter_pieces_l902_90212

-- Define the number of glass bottles and aluminum cans as constants.
def glass_bottles : ℕ := 10
def aluminum_cans : ℕ := 8

-- State the theorem that the sum of glass bottles and aluminum cans is 18.
theorem total_litter_pieces : glass_bottles + aluminum_cans = 18 := by
  sorry

end total_litter_pieces_l902_90212


namespace find_smaller_number_l902_90218

theorem find_smaller_number (a b : ℕ) 
  (h1 : a + b = 15) 
  (h2 : 3 * (a - b) = 21) : b = 4 :=
by
  sorry

end find_smaller_number_l902_90218


namespace sum_of_remainders_l902_90253

theorem sum_of_remainders (a b c : ℕ) (h₁ : a % 30 = 15) (h₂ : b % 30 = 7) (h₃ : c % 30 = 18) : 
    (a + b + c) % 30 = 10 := 
by
  sorry

end sum_of_remainders_l902_90253


namespace sum_MN_MK_eq_14_sqrt4_3_l902_90281

theorem sum_MN_MK_eq_14_sqrt4_3
  (MN MK : ℝ)
  (area: ℝ)
  (angle_LMN : ℝ)
  (h_area : area = 49)
  (h_angle_LMN : angle_LMN = 30) :
  MN + MK = 14 * (Real.sqrt (Real.sqrt 3)) :=
by
  sorry

end sum_MN_MK_eq_14_sqrt4_3_l902_90281


namespace smallest_number_of_students_l902_90298

theorem smallest_number_of_students (a b c : ℕ) (h1 : 4 * c = 3 * a) (h2 : 7 * b = 5 * a) (h3 : 10 * c = 9 * b) : a + b + c = 66 := sorry

end smallest_number_of_students_l902_90298


namespace min_draws_to_ensure_20_of_one_color_l902_90244

-- Define the total number of balls for each color
def red_balls : ℕ := 30
def green_balls : ℕ := 25
def yellow_balls : ℕ := 22
def blue_balls : ℕ := 15
def white_balls : ℕ := 12
def black_balls : ℕ := 10

-- Define the minimum number of balls to guarantee at least one color reaches 20 balls
def min_balls_needed : ℕ := 95

-- Theorem to state the problem mathematically in Lean
theorem min_draws_to_ensure_20_of_one_color :
  ∀ (r g y b w bl : ℕ),
    r = 30 → g = 25 → y = 22 → b = 15 → w = 12 → bl = 10 →
    (∃ n : ℕ, n ≥ min_balls_needed ∧
    ∀ (r_draw g_draw y_draw b_draw w_draw bl_draw : ℕ),
      r_draw + g_draw + y_draw + b_draw + w_draw + bl_draw = n →
      (r_draw > 19 ∨ g_draw > 19 ∨ y_draw > 19 ∨ b_draw > 19 ∨ w_draw > 19 ∨ bl_draw > 19)) :=
by
  intros r g y b w bl hr hg hy hb hw hbl
  use min_balls_needed
  sorry

end min_draws_to_ensure_20_of_one_color_l902_90244


namespace silk_pieces_count_l902_90261

theorem silk_pieces_count (S C : ℕ) (h1 : S = 2 * C) (h2 : S + C + 2 = 13) : S = 7 :=
by
  sorry

end silk_pieces_count_l902_90261


namespace number_of_rectangles_l902_90247

theorem number_of_rectangles (m n : ℕ) (h1 : m = 8) (h2 : n = 10) : (m - 1) * (n - 1) = 63 := by
  sorry

end number_of_rectangles_l902_90247


namespace non_neg_int_solutions_m_value_integer_values_of_m_l902_90292

-- 1. Non-negative integer solutions of x + 2y = 3
theorem non_neg_int_solutions (x y : ℕ) :
  x + 2 * y = 3 ↔ (x = 3 ∧ y = 0) ∨ (x = 1 ∧ y = 1) :=
sorry

-- 2. If (x, y) = (1, 1) satisfies both x + 2y = 3 and x + y = 2, then m = -4
theorem m_value (m : ℝ) :
  (1 + 2 * 1 = 3) ∧ (1 + 1 = 2) ∧ (1 - 2 * 1 + m * 1 = -5) → m = -4 :=
sorry

-- 3. Given n = 3, integer values of m are -2 or 0
theorem integer_values_of_m (m : ℤ) :
  ∃ x y : ℤ, 3 * x + 4 * y = 5 ∧ x - 2 * y + m * x = -5 → m = -2 ∨ m = 0 :=
sorry

end non_neg_int_solutions_m_value_integer_values_of_m_l902_90292


namespace sum_of_digits_10pow97_minus_97_l902_90226

-- Define a function that computes the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main statement we want to prove
theorem sum_of_digits_10pow97_minus_97 :
  sum_of_digits (10^97 - 97) = 858 :=
by
  sorry

end sum_of_digits_10pow97_minus_97_l902_90226


namespace shaded_region_area_l902_90258

theorem shaded_region_area (a b : ℕ) (H : a = 2) (K : b = 4) :
  let s := a + b
  let area_square_EFGH := s * s
  let area_smaller_square_FG := a * a
  let area_smaller_square_EF := b * b
  let shaded_area := area_square_EFGH - (area_smaller_square_FG + area_smaller_square_EF)
  shaded_area = 16 := 
by
  sorry

end shaded_region_area_l902_90258


namespace point_not_on_line_l902_90238

theorem point_not_on_line
  (p q : ℝ)
  (h : p * q > 0) :
  ¬ (∃ (x y : ℝ), x = 2023 ∧ y = 0 ∧ y = p * x + q) :=
by
  sorry

end point_not_on_line_l902_90238


namespace four_racers_meet_l902_90276

/-- In a circular auto race, four racers participate. Their cars start simultaneously from 
the same point and move at constant speeds, and for any three cars, there is a moment 
when they meet. Prove that after the start of the race, there will be a moment when all 
four cars meet. (Assume the race continues indefinitely in time.) -/
theorem four_racers_meet (V1 V2 V3 V4 : ℝ) (L : ℝ) (t : ℝ) 
  (h1 : 0 ≤ t) 
  (h2 : V1 ≤ V2 ∧ V2 ≤ V3 ∧ V3 ≤ V4)
  (h3 : ∀ t1 t2 t3, ∃ t, t1 * V1 = t ∧ t2 * V2 = t ∧ t3 * V3 = t) :
  ∃ t, t > 0 ∧ ∃ t', V1 * t' % L = 0 ∧ V2 * t' % L = 0 ∧ V3 * t' % L = 0 ∧ V4 * t' % L = 0 :=
sorry

end four_racers_meet_l902_90276


namespace negation_of_existence_l902_90233

theorem negation_of_existence : 
  (¬ ∃ x : ℝ, Real.exp x = x - 1) = (∀ x : ℝ, Real.exp x ≠ x - 1) :=
by
  sorry

end negation_of_existence_l902_90233
