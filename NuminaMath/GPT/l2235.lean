import Mathlib

namespace salon_revenue_l2235_223574

noncomputable def revenue (num_customers first_visit second_visit third_visit : ℕ) (first_charge second_charge : ℕ) : ℕ :=
  num_customers * first_charge + second_visit * second_charge + third_visit * second_charge

theorem salon_revenue : revenue 100 100 30 10 10 8 = 1320 :=
by
  unfold revenue
  -- The proof will continue here.
  sorry

end salon_revenue_l2235_223574


namespace profit_percentage_before_decrease_l2235_223553

-- Defining the conditions as Lean definitions
def newManufacturingCost : ℝ := 50
def oldManufacturingCost : ℝ := 80
def profitPercentageNew : ℝ := 0.5

-- Defining the problem as a theorem in Lean
theorem profit_percentage_before_decrease
  (P : ℝ)
  (hP : profitPercentageNew * P = P - newManufacturingCost) :
  ((P - oldManufacturingCost) / P) * 100 = 20 := 
by
  sorry

end profit_percentage_before_decrease_l2235_223553


namespace opposite_of_neg_five_is_five_l2235_223530

-- Define the condition for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that the opposite of -5 is 5
theorem opposite_of_neg_five_is_five : is_opposite (-5) 5 :=
by
  -- Insert the proof here
  sorry

end opposite_of_neg_five_is_five_l2235_223530


namespace preimage_of_5_1_is_2_3_l2235_223541

-- Define the mapping function
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, 2*p.1 - p.2)

-- Define the pre-image condition for (5, 1)
theorem preimage_of_5_1_is_2_3 : ∃ p : ℝ × ℝ, f p = (5, 1) ∧ p = (2, 3) :=
by
  -- Here we state that such a point p exists with the required properties.
  sorry

end preimage_of_5_1_is_2_3_l2235_223541


namespace part1_part2_l2235_223544

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem part1 (x : ℝ) : f x ≥ 3 ↔ x ≤ -3 / 2 ∨ x ≥ 3 / 2 := 
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x ≥ a^2 - a) ↔ -1 ≤ a ∧ a ≤ 2 :=
  sorry

end part1_part2_l2235_223544


namespace cos_315_deg_l2235_223533

noncomputable def cos_315 : ℝ :=
  Real.cos (315 * Real.pi / 180)

theorem cos_315_deg : cos_315 = Real.sqrt 2 / 2 :=
by
  sorry

end cos_315_deg_l2235_223533


namespace average_chemistry_mathematics_l2235_223538

-- Define the conditions 
variable {P C M : ℝ} -- Marks in Physics, Chemistry, and Mathematics

-- The given condition in the problem
theorem average_chemistry_mathematics (h : P + C + M = P + 130) : (C + M) / 2 = 65 := 
by
  -- This will be the main proof block (we use 'sorry' to omit the actual proof)
  sorry

end average_chemistry_mathematics_l2235_223538


namespace pavan_total_distance_l2235_223570

theorem pavan_total_distance:
  ∀ (D : ℝ),
  (∃ Time1 Time2,
    Time1 = (D / 2) / 30 ∧
    Time2 = (D / 2) / 25 ∧
    Time1 + Time2 = 11)
  → D = 150 :=
by
  intros D h
  sorry

end pavan_total_distance_l2235_223570


namespace bowling_ball_weight_l2235_223519

theorem bowling_ball_weight (b c : ℕ) 
  (h1 : 5 * b = 3 * c) 
  (h2 : 3 * c = 105) : 
  b = 21 := 
  sorry

end bowling_ball_weight_l2235_223519


namespace total_amount_distributed_l2235_223597

theorem total_amount_distributed (A : ℝ) :
  (∀ A, (A / 14 = A / 18 + 80) → A = 5040) :=
by
  sorry

end total_amount_distributed_l2235_223597


namespace original_number_is_45_l2235_223593

theorem original_number_is_45 (x y : ℕ) (h1 : x + y = 9) (h2 : 10 * y + x = 10 * x + y + 9) : 10 * x + y = 45 := by
  sorry

end original_number_is_45_l2235_223593


namespace divides_equiv_l2235_223502

theorem divides_equiv (m n : ℤ) : 
  (17 ∣ (2 * m + 3 * n)) ↔ (17 ∣ (9 * m + 5 * n)) :=
by
  sorry

end divides_equiv_l2235_223502


namespace count_paths_COMPUTER_l2235_223506

theorem count_paths_COMPUTER : 
  let possible_paths (n : ℕ) := 2 ^ n 
  possible_paths 7 + possible_paths 7 + 1 = 257 :=
by sorry

end count_paths_COMPUTER_l2235_223506


namespace carolyn_sum_correct_l2235_223542

def initial_sequence := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def carolyn_removes : List ℕ := [4, 8, 10, 9]

theorem carolyn_sum_correct : carolyn_removes.sum = 31 :=
by
  sorry

end carolyn_sum_correct_l2235_223542


namespace sphere_volume_from_surface_area_l2235_223561

theorem sphere_volume_from_surface_area (S : ℝ) (V : ℝ) (R : ℝ) (h1 : S = 36 * Real.pi) (h2 : S = 4 * Real.pi * R ^ 2) (h3 : V = (4 / 3) * Real.pi * R ^ 3) : V = 36 * Real.pi :=
by
  sorry

end sphere_volume_from_surface_area_l2235_223561


namespace inequality_solution_l2235_223500

theorem inequality_solution (x : ℝ) :
  (2 / (x^2 + 2*x + 1) + 4 / (x^2 + 8*x + 7) > 3/2) ↔
  (x < -7 ∨ (-7 < x ∧ x < -1) ∨ (-1 < x)) :=
by sorry

end inequality_solution_l2235_223500


namespace probability_black_then_red_l2235_223518

/-- Definition of a standard deck -/
def standard_deck := {cards : Finset (Fin 52) // cards.card = 52}

/-- Definition of black cards in the deck -/
def black_cards := {cards : Finset (Fin 52) // cards.card = 26}

/-- Definition of red cards in the deck -/
def red_cards := {cards : Finset (Fin 52) // cards.card = 26}

/-- Probability of drawing the top card as black and the second card as red -/
def prob_black_then_red (deck : standard_deck) (black : black_cards) (red : red_cards) : ℚ :=
  (26 * 26) / (52 * 51)

theorem probability_black_then_red (deck : standard_deck) (black : black_cards) (red : red_cards) :
  prob_black_then_red deck black red = 13 / 51 :=
sorry

end probability_black_then_red_l2235_223518


namespace function_relationship_value_of_x_when_y_is_1_l2235_223550

variable (x y : ℝ) (k : ℝ)

-- Conditions
def inverse_proportion (x y : ℝ) : Prop :=
  ∃ k : ℝ, y = k / (x - 3)

axiom condition_1 : inverse_proportion x y
axiom condition_2 : y = 5 ∧ x = 4

-- Statements to be proved
theorem function_relationship :
  ∃ k : ℝ, (y = k / (x - 3)) ∧ (y = 5 ∧ x = 4 → k = 5) :=
by
  sorry

theorem value_of_x_when_y_is_1 (hy : y = 1) :
  ∃ x : ℝ, (y = 5 / (x - 3)) ∧ x = 8 :=
by
  sorry

end function_relationship_value_of_x_when_y_is_1_l2235_223550


namespace construct_segment_length_l2235_223562

theorem construct_segment_length (a b : ℝ) (h : a > b) : 
  ∃ c : ℝ, c = (a^2 + b^2) / (a - b) :=
by
  sorry

end construct_segment_length_l2235_223562


namespace find_term_of_sequence_l2235_223524

theorem find_term_of_sequence :
  ∀ (a d n : ℤ), a = -5 → d = -4 → (-4)*n + 1 = -401 → n = 100 :=
by
  intros a d n h₁ h₂ h₃
  sorry

end find_term_of_sequence_l2235_223524


namespace find_x_when_perpendicular_l2235_223555

def a : ℝ × ℝ := (1, -2)
def b (x: ℝ) : ℝ × ℝ := (x, 1)
def are_perpendicular (a b: ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem find_x_when_perpendicular (x: ℝ) (h: are_perpendicular a (b x)) : x = 2 :=
by
  sorry

end find_x_when_perpendicular_l2235_223555


namespace find_S_2013_l2235_223587

variable {a : ℕ → ℤ} -- the arithmetic sequence
variable {S : ℕ → ℤ} -- the sum of the first n terms

-- Conditions
axiom a1_eq_neg2011 : a 1 = -2011
axiom sum_sequence : ∀ n, S n = (n * (a 1 + a n)) / 2
axiom condition_eq : (S 2012 / 2012) - (S 2011 / 2011) = 1

-- The Lean statement to prove that S 2013 = 2013
theorem find_S_2013 : S 2013 = 2013 := by
  sorry

end find_S_2013_l2235_223587


namespace ab_value_l2235_223501

-- Define sets A and B
def A : Set ℝ := {-1.3}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b = 0}

-- The proof statement: Given A = B, prove ab = 0.104
theorem ab_value (a b : ℝ) (h : A = B a b) : a * b = 0.104 :=
by
  sorry

end ab_value_l2235_223501


namespace greatest_of_consecutive_even_numbers_l2235_223543

theorem greatest_of_consecutive_even_numbers (n : ℤ) (h : ((n - 4) + (n - 2) + n + (n + 2) + (n + 4)) / 5 = 35) : n + 4 = 39 :=
by
  sorry

end greatest_of_consecutive_even_numbers_l2235_223543


namespace convergent_inequalities_l2235_223596

theorem convergent_inequalities (α : ℝ) (P Q : ℕ → ℤ) (h_convergent : ∀ n ≥ 1, abs (α - P n / Q n) < 1 / (2 * (Q n) ^ 2) ∨ abs (α - P (n - 1) / Q (n - 1)) < 1 / (2 * (Q (n - 1))^2))
  (h_continued_fraction : ∀ n ≥ 1, P (n-1) * Q n - P n * Q (n-1) = (-1)^(n-1)) :
  ∃ p q : ℕ, 0 < q ∧ abs (α - p / q) < 1 / (2 * q^2) :=
sorry

end convergent_inequalities_l2235_223596


namespace todd_ate_cupcakes_l2235_223578

def total_cupcakes_baked := 68
def packages := 6
def cupcakes_per_package := 6
def total_packaged_cupcakes := packages * cupcakes_per_package
def remaining_cupcakes := total_cupcakes_baked - total_packaged_cupcakes

theorem todd_ate_cupcakes : total_cupcakes_baked - remaining_cupcakes = 36 := by
  sorry

end todd_ate_cupcakes_l2235_223578


namespace faster_train_speed_correct_l2235_223577

noncomputable def speed_of_faster_train (V_s_kmph : ℝ) (length_faster_train_m : ℝ) (time_s : ℝ) : ℝ :=
  let V_s_mps := V_s_kmph * (1000 / 3600)
  let V_r_mps := length_faster_train_m / time_s
  let V_f_mps := V_r_mps - V_s_mps
  V_f_mps * (3600 / 1000)

theorem faster_train_speed_correct : 
  speed_of_faster_train 36 90.0072 4 = 45.00648 := 
by
  sorry

end faster_train_speed_correct_l2235_223577


namespace find_coprime_pairs_l2235_223563

theorem find_coprime_pairs :
  ∀ (x y : ℕ), x > 0 → y > 0 → x.gcd y = 1 →
    (x ∣ y^2 + 210) →
    (y ∣ x^2 + 210) →
    (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 211) ∨ 
    (∃ n : ℕ, n > 0 ∧ n = 1 ∧ n = 1 ∧ 
      (x = 212*n - n - 1 ∨ y = 212*n - n - 1)) := sorry

end find_coprime_pairs_l2235_223563


namespace average_bacterial_count_closest_to_true_value_l2235_223536

-- Define the conditions
variables (dilution_spread_plate_method : Prop)
          (count_has_randomness : Prop)
          (count_not_uniform : Prop)

-- State the theorem
theorem average_bacterial_count_closest_to_true_value
  (h1: dilution_spread_plate_method)
  (h2: count_has_randomness)
  (h3: count_not_uniform)
  : true := sorry

end average_bacterial_count_closest_to_true_value_l2235_223536


namespace find_multiplier_l2235_223548

theorem find_multiplier (n k : ℤ) (h1 : n + 4 = 15) (h2 : 3 * n = k * (n + 4) + 3) : k = 2 :=
  sorry

end find_multiplier_l2235_223548


namespace chord_cos_theta_condition_l2235_223568

open Real

-- Translation of the given conditions and proof problem
theorem chord_cos_theta_condition
  (a b x y θ : ℝ)
  (h1 : a^2 = b^2 + 2) :
  x * y = cos θ := 
sorry

end chord_cos_theta_condition_l2235_223568


namespace exponent_multiplication_l2235_223532

theorem exponent_multiplication (a : ℝ) (m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 4) :
  a^(m + n) = 8 := by
  sorry

end exponent_multiplication_l2235_223532


namespace int_cubed_bound_l2235_223503

theorem int_cubed_bound (a : ℤ) (h : 0 < a^3 ∧ a^3 < 9) : a = 1 ∨ a = 2 :=
sorry

end int_cubed_bound_l2235_223503


namespace i_pow_2006_l2235_223590

-- Definitions based on given conditions
def i : ℂ := Complex.I

-- Cyclic properties of i (imaginary unit)
axiom i_pow_1 : i^1 = i
axiom i_pow_2 : i^2 = -1
axiom i_pow_3 : i^3 = -i
axiom i_pow_4 : i^4 = 1
axiom i_pow_5 : i^5 = i

-- The proof statement
theorem i_pow_2006 : (i^2006 = -1) :=
by
  sorry

end i_pow_2006_l2235_223590


namespace tank_length_l2235_223599

variable (rate : ℝ)
variable (time : ℝ)
variable (width : ℝ)
variable (depth : ℝ)
variable (volume : ℝ)
variable (length : ℝ)

-- Given conditions
axiom rate_cond : rate = 5 -- cubic feet per hour
axiom time_cond : time = 60 -- hours
axiom width_cond : width = 6 -- feet
axiom depth_cond : depth = 5 -- feet

-- Derived volume from the rate and time
axiom volume_cond : volume = rate * time

-- Definition of length from volume, width, and depth
axiom length_def : length = volume / (width * depth)

-- The proof problem to show
theorem tank_length : length = 10 := by
  -- conditions provided and we expect the length to be computed
  sorry

end tank_length_l2235_223599


namespace quadratic_roots_sum_product_l2235_223521

theorem quadratic_roots_sum_product :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0 ∧ x2^2 - 2 * x2 - 8 = 0 ∧ x1 ≠ x2) →
    (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end quadratic_roots_sum_product_l2235_223521


namespace max_overlap_l2235_223591

variable (A : Type) [Fintype A] [DecidableEq A]
variable (P1 P2 : A → Prop)

theorem max_overlap (hP1 : ∃ X : Finset A, (X.card : ℝ) / Fintype.card A = 0.20 ∧ ∀ a ∈ X, P1 a)
                    (hP2 : ∃ Y : Finset A, (Y.card : ℝ) / Fintype.card A = 0.70 ∧ ∀ a ∈ Y, P2 a) :
  ∃ Z : Finset A, (Z.card : ℝ) / Fintype.card A = 0.20 ∧ ∀ a ∈ Z, P1 a ∧ P2 a :=
sorry

end max_overlap_l2235_223591


namespace single_point_graph_l2235_223572

theorem single_point_graph (d : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 8 * y + d = 0 → x = -1 ∧ y = 4) → d = 19 :=
by
  sorry

end single_point_graph_l2235_223572


namespace mink_ratio_set_free_to_total_l2235_223504

-- Given conditions
def coats_needed_per_skin : ℕ := 15
def minks_bought : ℕ := 30
def babies_per_mink : ℕ := 6
def coats_made : ℕ := 7

-- Question as a proof problem
theorem mink_ratio_set_free_to_total :
  let total_minks := minks_bought * (1 + babies_per_mink)
  let minks_used := coats_made * coats_needed_per_skin
  let minks_set_free := total_minks - minks_used
  minks_set_free * 2 = total_minks :=
by
  sorry

end mink_ratio_set_free_to_total_l2235_223504


namespace solve_for_q_l2235_223586

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 14) (h2 : 6 * p + 5 * q = 17) : q = -1 / 11 :=
by
  sorry

end solve_for_q_l2235_223586


namespace council_counts_l2235_223539

theorem council_counts 
    (total_classes : ℕ := 20)
    (students_per_class : ℕ := 5)
    (total_students : ℕ := 100)
    (petya_class_council : ℕ × ℕ := (1, 4))  -- (boys, girls)
    (equal_boys_girls : 2 * 50 = total_students)  -- Equal number of boys and girls
    (more_girls_classes : ℕ := 15)
    (min_girls_each : ℕ := 3)
    (remaining_classes : ℕ := 4)
    (remaining_students : ℕ := 20)
    : (19, 1) = (19, 1) :=
by
    -- actual proof goes here
    sorry

end council_counts_l2235_223539


namespace non_congruent_triangles_count_l2235_223523

def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

def count_non_congruent_triangles : ℕ :=
  let a_values := [1, 2]
  let b_values := [2, 3]
  let triangles := [(1, 2, 2), (2, 2, 2), (2, 2, 3)]
  triangles.length

theorem non_congruent_triangles_count : count_non_congruent_triangles = 3 :=
  by
    -- Proof would go here
    sorry

end non_congruent_triangles_count_l2235_223523


namespace factor_of_change_l2235_223579

-- Given conditions
def avg_marks_before : ℕ := 45
def avg_marks_after : ℕ := 90
def num_students : ℕ := 30

-- Prove the factor F by which marks are changed
theorem factor_of_change : ∃ F : ℕ, avg_marks_before * F = avg_marks_after := 
by
  use 2
  have h1 : 30 * avg_marks_before = 30 * 45 := rfl
  have h2 : 30 * avg_marks_after = 30 * 90 := rfl
  sorry

end factor_of_change_l2235_223579


namespace best_chart_for_temperature_changes_l2235_223567

def Pie_chart := "Represent the percentage of parts in the whole."
def Line_chart := "Represent changes over time."
def Bar_chart := "Show the specific number of each item."

theorem best_chart_for_temperature_changes : 
  "The best statistical chart to use for understanding temperature changes throughout a day" = Line_chart :=
by
  sorry

end best_chart_for_temperature_changes_l2235_223567


namespace diamonds_count_l2235_223566

-- Definitions based on the conditions given in the problem
def totalGems : Nat := 5155
def rubies : Nat := 5110
def diamonds (total rubies : Nat) : Nat := total - rubies

-- Statement of the proof problem
theorem diamonds_count : diamonds totalGems rubies = 45 := by
  sorry

end diamonds_count_l2235_223566


namespace tan_13pi_div_3_eq_sqrt_3_l2235_223575

theorem tan_13pi_div_3_eq_sqrt_3 : Real.tan (13 * Real.pi / 3) = Real.sqrt 3 :=
  sorry

end tan_13pi_div_3_eq_sqrt_3_l2235_223575


namespace count_solutions_cos2x_plus_3sin2x_eq_1_l2235_223511

open Real

theorem count_solutions_cos2x_plus_3sin2x_eq_1 :
  ∀ x : ℝ, (-10 < x ∧ x < 45 → cos x ^ 2 + 3 * sin x ^ 2 = 1) → 
  ∃! n : ℕ, n = 18 := 
by
  intro x hEq
  sorry

end count_solutions_cos2x_plus_3sin2x_eq_1_l2235_223511


namespace fill_bucket_time_l2235_223598

-- Problem statement:
-- Prove that the time taken to fill the bucket completely is 150 seconds
-- given that two-thirds of the bucket is filled in 100 seconds.

theorem fill_bucket_time (t : ℕ) (h : (2 / 3) * t = 100) : t = 150 :=
by
  -- Proof should be here
  sorry

end fill_bucket_time_l2235_223598


namespace travel_time_without_paddles_l2235_223512

variables (A B : Type) (v v_r S : ℝ)
noncomputable def time_to_travel (distance velocity : ℝ) := distance / velocity

-- Condition: The travel time from A to B is 3 times the travel time from B to A
axiom travel_condition : (time_to_travel S (v + v_r)) = 3 * (time_to_travel S (v - v_r))

-- Condition: We are considering travel from B to A by canoe without paddles
noncomputable def time_without_paddles := time_to_travel S v_r

-- Proving that without paddles it takes 3 times longer than usual (using canoes with paddles)
theorem travel_time_without_paddles :
  time_without_paddles S v_r = 3 * (time_to_travel S (v - v_r)) :=
sorry

end travel_time_without_paddles_l2235_223512


namespace cesaro_sum_100_terms_l2235_223571

noncomputable def cesaro_sum (A : List ℝ) : ℝ :=
  let n := A.length
  (List.sum A) / n

theorem cesaro_sum_100_terms :
  ∀ (A : List ℝ), A.length = 99 →
  cesaro_sum A = 1000 →
  cesaro_sum (1 :: A) = 991 :=
by
  intros A h1 h2
  sorry

end cesaro_sum_100_terms_l2235_223571


namespace five_digit_palindromes_count_l2235_223516

theorem five_digit_palindromes_count : ∃ n : ℕ, n = 900 ∧
  ∀ (a b c : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
    ∃ (x : ℕ), x = a * 10^4 + b * 10^3 + c * 10^2 + b * 10 + a := sorry

end five_digit_palindromes_count_l2235_223516


namespace valentines_count_l2235_223576

theorem valentines_count (x y : ℕ) (h : x * y = x + y + 42) : x * y = 88 := by
  sorry

end valentines_count_l2235_223576


namespace initial_rotations_l2235_223554

-- Given conditions as Lean definitions
def rotations_per_block : ℕ := 200
def blocks_to_ride : ℕ := 8
def additional_rotations_needed : ℕ := 1000

-- Question translated to proof statement
theorem initial_rotations (rotations : ℕ) :
  rotations + additional_rotations_needed = rotations_per_block * blocks_to_ride → rotations = 600 :=
by
  intros h
  sorry

end initial_rotations_l2235_223554


namespace smallest_b_value_l2235_223583

theorem smallest_b_value (a b c : ℕ) (h0 : a > 0) (h1 : b > 0) (h2 : c > 0)
  (h3 : (31 : ℚ) / 72 = (a : ℚ) / 8 + (b : ℚ) / 9 - c) :
  b = 5 :=
sorry

end smallest_b_value_l2235_223583


namespace largest_five_digit_number_with_product_120_l2235_223537

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def prod_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldr (· * ·) 1

def max_five_digit_prod_120 : ℕ := 85311

theorem largest_five_digit_number_with_product_120 :
  is_five_digit max_five_digit_prod_120 ∧ prod_of_digits max_five_digit_prod_120 = 120 :=
by
  sorry

end largest_five_digit_number_with_product_120_l2235_223537


namespace initial_strawberries_l2235_223517

-- Define the conditions
def strawberries_eaten : ℝ := 42.0
def strawberries_left : ℝ := 36.0

-- State the theorem
theorem initial_strawberries :
  strawberries_eaten + strawberries_left = 78 :=
by
  sorry

end initial_strawberries_l2235_223517


namespace difference_of_squares_153_147_l2235_223592

theorem difference_of_squares_153_147 : (153^2 - 147^2) = 1800 := by
  sorry

end difference_of_squares_153_147_l2235_223592


namespace main_theorem_l2235_223510

-- Define the function f(x)
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Condition: f is symmetric about x = 1
def symmetric_about_one (a b c : ℝ) : Prop := 
  ∀ x : ℝ, f a b c (1 - x) = f a b c (1 + x)

-- Main statement
theorem main_theorem (a b c : ℝ) (h₁ : 0 < a) (h₂ : symmetric_about_one a b c) :
  ∀ x : ℝ, f a b c (2^x) > f a b c (3^x) :=
sorry

end main_theorem_l2235_223510


namespace radius_of_larger_circle_15_l2235_223557

def radius_larger_circle (r1 r2 r3 r : ℝ) : Prop :=
  ∃ (A B C O : EuclideanSpace ℝ (Fin 2)), 
    dist A B = r1 + r2 ∧
    dist B C = r2 + r3 ∧
    dist A C = r1 + r3 ∧
    dist O A = r - r1 ∧
    dist O B = r - r2 ∧
    dist O C = r - r3 ∧
    (dist O A + r1 = r ∧
    dist O B + r2 = r ∧
    dist O C + r3 = r)

theorem radius_of_larger_circle_15 :
  radius_larger_circle 10 3 2 15 :=
by
  sorry

end radius_of_larger_circle_15_l2235_223557


namespace integer_between_squares_l2235_223520

theorem integer_between_squares (a b c d: ℝ) (h₀: 0 < a) (h₁: 0 < b) (h₂: 0 < c) (h₃: 0 < d) (h₄: c * d = 1) : 
  ∃ n : ℤ, ab ≤ n^2 ∧ n^2 ≤ (a + c) * (b + d) := 
by 
  sorry

end integer_between_squares_l2235_223520


namespace fraction_in_orange_tin_l2235_223514

variables {C : ℕ} -- assume total number of cookies as a natural number

theorem fraction_in_orange_tin (h1 : 11 / 12 = (1 / 6) + (5 / 12) + w)
  (h2 : 1 - (11 / 12) = 1 / 12) :
  w = 1 / 3 :=
by
  sorry

end fraction_in_orange_tin_l2235_223514


namespace length_of_segment_l2235_223580

theorem length_of_segment (x : ℤ) (hx : |x - 3| = 4) : 
  let a := 7
  let b := -1
  a - b = 8 := by
    sorry

end length_of_segment_l2235_223580


namespace unique_root_when_abs_t_gt_2_l2235_223515

theorem unique_root_when_abs_t_gt_2 (t : ℝ) (h : |t| > 2) :
  ∃! x : ℝ, x^3 - 3 * x = t ∧ |x| > 2 :=
sorry

end unique_root_when_abs_t_gt_2_l2235_223515


namespace area_of_triangle_ABC_l2235_223551

open Real

-- Defining the conditions as per the problem
def triangle_side_equality (AB AC : ℝ) : Prop := AB = AC
def angle_relation (angleBAC angleBTC : ℝ) : Prop := angleBAC = 2 * angleBTC
def side_length_BT (BT : ℝ) : Prop := BT = 70
def side_length_AT (AT : ℝ) : Prop := AT = 37

-- Proving the area of triangle ABC given the conditions
theorem area_of_triangle_ABC
  (AB AC : ℝ)
  (angleBAC angleBTC : ℝ)
  (BT AT : ℝ)
  (h1 : triangle_side_equality AB AC)
  (h2 : angle_relation angleBAC angleBTC)
  (h3 : side_length_BT BT)
  (h4 : side_length_AT AT) 
  : ∃ area : ℝ, area = 420 :=
sorry

end area_of_triangle_ABC_l2235_223551


namespace domain_shift_l2235_223556

theorem domain_shift (f : ℝ → ℝ) (dom_f : ∀ x, 1 ≤ x ∧ x ≤ 4 → f x = f x) :
  ∀ x, -1 ≤ x ∧ x ≤ 2 ↔ (1 ≤ x + 2 ∧ x + 2 ≤ 4) :=
by
  sorry

end domain_shift_l2235_223556


namespace rhombus_diagonal_length_l2235_223522

theorem rhombus_diagonal_length (d1 d2 : ℝ) (area : ℝ) 
(h_d2 : d2 = 18) (h_area : area = 126) (h_formula : area = (d1 * d2) / 2) : 
d1 = 14 :=
by
  -- We're skipping the proof steps.
  sorry

end rhombus_diagonal_length_l2235_223522


namespace base10_to_base7_conversion_l2235_223569

theorem base10_to_base7_conversion : 2023 = 5 * 7^3 + 6 * 7^2 + 2 * 7^1 + 0 * 7^0 :=
  sorry

end base10_to_base7_conversion_l2235_223569


namespace average_speed_to_SF_l2235_223505

theorem average_speed_to_SF (v d : ℝ) (h1 : d ≠ 0) (h2 : v ≠ 0) :
  (2 * d / ((d / v) + (2 * d / v)) = 34) → v = 51 :=
by
  -- proof goes here
  sorry

end average_speed_to_SF_l2235_223505


namespace quadrilateral_pyramid_volume_l2235_223559

theorem quadrilateral_pyramid_volume (h Q : ℝ) : 
  ∃ V : ℝ, V = (2 / 3 : ℝ) * h * (Real.sqrt (h^2 + 4 * Q^2) - h^2) :=
by
  sorry

end quadrilateral_pyramid_volume_l2235_223559


namespace price_reduction_l2235_223588

theorem price_reduction (original_price final_price : ℝ) (x : ℝ) 
  (h : original_price = 289) (h2 : final_price = 256) :
  289 * (1 - x) ^ 2 = 256 := sorry

end price_reduction_l2235_223588


namespace intersection_is_singleton_l2235_223534

-- Definitions of sets M and N
def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

-- The stated proposition we need to prove
theorem intersection_is_singleton :
  M ∩ N = {(3, -1)} :=
by {
  sorry
}

end intersection_is_singleton_l2235_223534


namespace playerA_winning_conditions_l2235_223529

def playerA_has_winning_strategy (n : ℕ) : Prop :=
  (n % 4 = 0) ∨ (n % 4 = 3)

theorem playerA_winning_conditions (n : ℕ) (h : n ≥ 2) : 
  playerA_has_winning_strategy n ↔ (n % 4 = 0 ∨ n % 4 = 3) :=
by sorry

end playerA_winning_conditions_l2235_223529


namespace y_intercept_of_line_l2235_223573

theorem y_intercept_of_line : ∀ (x y : ℝ), (3 * x - 4 * y = 12) → (x = 0) → (y = -3) :=
by
  intros x y h_eq h_x0
  sorry

end y_intercept_of_line_l2235_223573


namespace ribbon_tape_needed_l2235_223528

theorem ribbon_tape_needed 
  (total_length : ℝ) (num_boxes : ℕ) (ribbon_per_box : ℝ)
  (h1 : total_length = 82.04)
  (h2 : num_boxes = 28)
  (h3 : total_length / num_boxes = ribbon_per_box)
  : ribbon_per_box = 2.93 :=
sorry

end ribbon_tape_needed_l2235_223528


namespace solve_system_of_equations_l2235_223558

theorem solve_system_of_equations (x y : ℝ) (hx : x + y + Real.sqrt (x * y) = 28)
  (hy : x^2 + y^2 + x * y = 336) : (x = 4 ∧ y = 16) ∨ (x = 16 ∧ y = 4) :=
sorry

end solve_system_of_equations_l2235_223558


namespace average_incorrect_answers_is_correct_l2235_223595

-- Definitions
def total_items : ℕ := 60
def liza_correct_answers : ℕ := (90 * total_items) / 100
def rose_correct_answers : ℕ := liza_correct_answers + 2
def max_correct_answers : ℕ := liza_correct_answers - 5

def liza_incorrect_answers : ℕ := total_items - liza_correct_answers
def rose_incorrect_answers : ℕ := total_items - rose_correct_answers
def max_incorrect_answers : ℕ := total_items - max_correct_answers

def average_incorrect_answers : ℚ :=
  (liza_incorrect_answers + rose_incorrect_answers + max_incorrect_answers) / 3

-- Theorem statement
theorem average_incorrect_answers_is_correct : average_incorrect_answers = 7 := by
  -- Proof goes here
  sorry

end average_incorrect_answers_is_correct_l2235_223595


namespace option_c_is_incorrect_l2235_223560

/-- Define the temperature data -/
def temps : List Int := [-20, -10, 0, 10, 20, 30]

/-- Define the speed of sound data corresponding to the temperatures -/
def speeds : List Int := [318, 324, 330, 336, 342, 348]

/-- The speed of sound at 10 degrees Celsius -/
def speed_at_10 : Int := 336

/-- The incorrect claim in option C -/
def incorrect_claim : Prop := (speed_at_10 * 4 ≠ 1334)

/-- Prove that the claim in option C is incorrect -/
theorem option_c_is_incorrect : incorrect_claim :=
by {
  sorry
}

end option_c_is_incorrect_l2235_223560


namespace black_piece_is_option_C_l2235_223540

-- Definitions for the problem conditions
def rectangular_prism (cubes : Nat) := cubes = 16
def block (small_cubes : Nat) := small_cubes = 4
def piece_containing_black_shape_is_partially_seen (rows : Nat) := rows = 2

-- Hypotheses and conditions
variable (rect_prism : Nat) (block1 block2 block3 block4 : Nat)
variable (visibility_block1 visibility_block2 visibility_block3 : Bool)
variable (visible_in_back_row : Bool)

-- Given conditions based on the problem statement
axiom h1 : rectangular_prism rect_prism
axiom h2 : block block1
axiom h3 : block block2
axiom h4 : block block3
axiom h5 : block block4
axiom h6 : visibility_block1 = true
axiom h7 : visibility_block2 = true
axiom h8 : visibility_block3 = true
axiom h9 : visible_in_back_row = true

-- Prove the configuration matches Option C
theorem black_piece_is_option_C :
  ∀ (config : Char), (config = 'C') :=
by
  intros
  -- Proof incomplete intentionally.
  sorry

end black_piece_is_option_C_l2235_223540


namespace anand_income_l2235_223552

theorem anand_income
  (x y : ℕ)
  (h1 : 5 * x - 3 * y = 800)
  (h2 : 4 * x - 2 * y = 800) : 
  5 * x = 2000 := 
sorry

end anand_income_l2235_223552


namespace combine_like_terms_l2235_223581

theorem combine_like_terms (a : ℝ) : 2 * a + 3 * a = 5 * a := 
by sorry

end combine_like_terms_l2235_223581


namespace find_y_intercept_l2235_223584

def line_y_intercept (m x y : ℝ) (pt : ℝ × ℝ) : ℝ :=
  let y_intercept := pt.snd - m * pt.fst
  y_intercept

theorem find_y_intercept (m x y b : ℝ) (pt : ℝ × ℝ) (h1 : m = 2) (h2 : pt = (498, 998)) :
  line_y_intercept m x y pt = 2 :=
by
  sorry

end find_y_intercept_l2235_223584


namespace solution_set_of_inequality_l2235_223546

theorem solution_set_of_inequality :
  ∀ x : ℝ, (x^2 - 3*x - 4 ≤ 0) ↔ (-1 ≤ x ∧ x ≤ 4) :=
by sorry

end solution_set_of_inequality_l2235_223546


namespace each_bug_ate_1_5_flowers_l2235_223564

-- Define the conditions given in the problem
def bugs : ℝ := 2.0
def flowers : ℝ := 3.0

-- The goal is to prove that the number of flowers each bug ate is 1.5
theorem each_bug_ate_1_5_flowers : (flowers / bugs) = 1.5 :=
by
  sorry

end each_bug_ate_1_5_flowers_l2235_223564


namespace cost_per_kg_after_30_l2235_223545

theorem cost_per_kg_after_30 (l m : ℝ) 
  (hl : l = 20) 
  (h1 : 30 * l + 3 * m = 663) 
  (h2 : 30 * l + 6 * m = 726) : 
  m = 21 :=
by
  -- Proof will be written here
  sorry

end cost_per_kg_after_30_l2235_223545


namespace speed_increase_percentage_l2235_223527

variable (T : ℚ)  -- usual travel time in minutes
variable (v : ℚ)  -- usual speed

-- Conditions
-- Ivan usually arrives at 9:00 AM, traveling for T minutes at speed v.
-- When Ivan leaves 40 minutes late and drives 1.6 times his usual speed, he arrives at 8:35 AM
def usual_arrival_time : ℚ := 9 * 60  -- 9:00 AM in minutes

def time_when_late : ℚ := (9 * 60) + 40 - (25 + 40)  -- 8:35 AM in minutes

def increased_speed := 1.6 * v -- 60% increase in speed

def time_taken_with_increased_speed := T - 65

theorem speed_increase_percentage :
  ((T / (T - 40)) = 1.3) :=
by
-- assume the equation for usual time T in terms of increased speed is known
-- Use provided conditions and solve the equation to derive the result.
  sorry

end speed_increase_percentage_l2235_223527


namespace ellipse_foci_y_axis_range_l2235_223582

noncomputable def is_ellipse_with_foci_on_y_axis (k : ℝ) : Prop :=
  (k > 5) ∧ (k < 10) ∧ (10 - k > k - 5)

theorem ellipse_foci_y_axis_range (k : ℝ) :
  is_ellipse_with_foci_on_y_axis k ↔ 5 < k ∧ k < 7.5 := 
by
  sorry

end ellipse_foci_y_axis_range_l2235_223582


namespace length_AB_eight_l2235_223508

-- Define parabola and line
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def line (x y k : ℝ) : Prop := y = k * x - k

-- Define intersection points A and B
def intersects (p1 p2 : ℝ × ℝ) (k : ℝ) : Prop :=
  parabola p1.1 p1.2 ∧ line p1.1 p1.2 k ∧
  parabola p2.1 p2.2 ∧ line p2.1 p2.2 k

-- Define midpoint distance condition
def midpoint_condition (p1 p2 : ℝ × ℝ) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint.1 = 3

-- The main theorem statement
theorem length_AB_eight (k : ℝ) (A B : ℝ × ℝ) (h1 : intersects A B k)
  (h2 : midpoint_condition A B) : abs ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 64 := 
sorry

end length_AB_eight_l2235_223508


namespace sequence_a_100_l2235_223513

theorem sequence_a_100 (a : ℕ → ℤ) (h₁ : a 1 = 3) (h₂ : ∀ n : ℕ, a (n + 1) = a n - 2) : a 100 = -195 :=
by
  sorry

end sequence_a_100_l2235_223513


namespace range_of_m_value_of_x_l2235_223547

noncomputable def a : ℝ := 3 / 2

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log a

-- Statement for the range of m
theorem range_of_m :
  ∀ m : ℝ, f (3 * m - 2) < f (2 * m + 5) ↔ (2 / 3) < m ∧ m < 7 :=
by
  intro m
  sorry

-- Value of x
theorem value_of_x :
  ∃ x : ℝ, f (x - 2 / x) = Real.log (7 / 2) / Real.log (3 / 2) ∧ x > 0 ∧ x = 4 :=
by
  use 4
  sorry

end range_of_m_value_of_x_l2235_223547


namespace simplify_and_evaluate_expression_l2235_223507

   variable (x : ℝ)

   theorem simplify_and_evaluate_expression (h : x = 2 * Real.sqrt 5 - 1) :
     (1 / (x ^ 2 + 2 * x + 1) * (1 + 3 / (x - 1)) / ((x + 2) / (x ^ 2 - 1))) = Real.sqrt 5 / 10 :=
   sorry
   
end simplify_and_evaluate_expression_l2235_223507


namespace average_cd_l2235_223589

theorem average_cd (c d: ℝ) (h: (4 + 6 + 8 + c + d) / 5 = 18) : (c + d) / 2 = 36 :=
by sorry

end average_cd_l2235_223589


namespace survey_participants_l2235_223525

-- Total percentage for option A and option B in bytes
def percent_A : ℝ := 0.50
def percent_B : ℝ := 0.30

-- Number of participants who chose option A
def participants_A : ℕ := 150

-- Target number of participants who chose option B (to be proved)
def participants_B : ℕ := 90

-- The theorem to prove the number of participants who chose option B
theorem survey_participants :
  (participants_B : ℝ) = participants_A * (percent_B / percent_A) :=
by
  sorry

end survey_participants_l2235_223525


namespace area_of_large_rectangle_l2235_223531

noncomputable def areaEFGH : ℕ :=
  let shorter_side := 3
  let longer_side := 2 * shorter_side
  let width_EFGH := shorter_side + shorter_side
  let length_EFGH := longer_side + longer_side
  width_EFGH * length_EFGH

theorem area_of_large_rectangle :
  areaEFGH = 72 := by
  sorry

end area_of_large_rectangle_l2235_223531


namespace total_instruments_correct_l2235_223565

def numberOfFlutesCharlie : ℕ := 1
def numberOfHornsCharlie : ℕ := 2
def numberOfHarpsCharlie : ℕ := 1
def numberOfDrumsCharlie : ℕ := 5

def numberOfFlutesCarli : ℕ := 3 * numberOfFlutesCharlie
def numberOfHornsCarli : ℕ := numberOfHornsCharlie / 2
def numberOfDrumsCarli : ℕ := 2 * numberOfDrumsCharlie
def numberOfHarpsCarli : ℕ := 0

def numberOfFlutesNick : ℕ := 2 * numberOfFlutesCarli - 1
def numberOfHornsNick : ℕ := numberOfHornsCharlie + numberOfHornsCarli
def numberOfDrumsNick : ℕ := 4 * numberOfDrumsCarli - 2
def numberOfHarpsNick : ℕ := 0

def numberOfFlutesDaisy : ℕ := numberOfFlutesNick * numberOfFlutesNick
def numberOfHornsDaisy : ℕ := (numberOfHornsNick - numberOfHornsCarli) / 2
def numberOfDrumsDaisy : ℕ := (numberOfDrumsCharlie + numberOfDrumsCarli + numberOfDrumsNick) / 3
def numberOfHarpsDaisy : ℕ := numberOfHarpsCharlie

def numberOfInstrumentsCharlie : ℕ := numberOfFlutesCharlie + numberOfHornsCharlie + numberOfHarpsCharlie + numberOfDrumsCharlie
def numberOfInstrumentsCarli : ℕ := numberOfFlutesCarli + numberOfHornsCarli + numberOfDrumsCarli
def numberOfInstrumentsNick : ℕ := numberOfFlutesNick + numberOfHornsNick + numberOfDrumsNick
def numberOfInstrumentsDaisy : ℕ := numberOfFlutesDaisy + numberOfHornsDaisy + numberOfHarpsDaisy + numberOfDrumsDaisy

def totalInstruments : ℕ := numberOfInstrumentsCharlie + numberOfInstrumentsCarli + numberOfInstrumentsNick + numberOfInstrumentsDaisy

theorem total_instruments_correct : totalInstruments = 113 := by
  sorry

end total_instruments_correct_l2235_223565


namespace next_performance_together_in_90_days_l2235_223585

theorem next_performance_together_in_90_days :
  Nat.lcm (Nat.lcm 5 6) (Nat.lcm 9 10) = 90 := by
  sorry

end next_performance_together_in_90_days_l2235_223585


namespace b_k_divisible_by_11_is_5_l2235_223535

def b (n : ℕ) : ℕ :=
  -- Function to concatenate numbers from 1 to n
  let digits := List.join (List.map (λ x => Nat.digits 10 x) (List.range' 1 n.succ))
  digits.foldl (λ acc d => acc * 10 + d) 0

def g (n : ℕ) : ℤ :=
  let digits := Nat.digits 10 n
  digits.enum.foldl (λ acc ⟨i, d⟩ => if i % 2 = 0 then acc + Int.ofNat d else acc - Int.ofNat d) 0

def isDivisibleBy11 (n : ℕ) : Bool :=
  g n % 11 = 0

def count_b_k_divisible_by_11 : ℕ :=
  List.length (List.filter isDivisibleBy11 (List.map b (List.range' 1 51)))

theorem b_k_divisible_by_11_is_5 : count_b_k_divisible_by_11 = 5 := by
  sorry

end b_k_divisible_by_11_is_5_l2235_223535


namespace A_empty_iff_a_gt_9_over_8_A_one_element_l2235_223526

-- Definition of A based on a given condition
def A (a : ℝ) : Set ℝ := {x | a * x^2 - 3 * x + 2 = 0}

-- Problem 1: Prove that if A is empty, then a > 9/8
theorem A_empty_iff_a_gt_9_over_8 {a : ℝ} : 
  (A a = ∅) ↔ (a > 9 / 8) := 
sorry

-- Problem 2: Prove the elements in A when it contains only one element
theorem A_one_element {a : ℝ} : 
  (∃! x, x ∈ A a) ↔ (a = 0 ∧ (A a = {2 / 3})) ∨ (a = 9 / 8 ∧ (A a = {4 / 3})) := 
sorry

end A_empty_iff_a_gt_9_over_8_A_one_element_l2235_223526


namespace engineer_days_l2235_223509

theorem engineer_days (x : ℕ) (k : ℕ) (d : ℕ) (n : ℕ) (m : ℕ) (e : ℕ)
  (h1 : k = 10) -- Length of the road in km
  (h2 : d = 15) -- Total days to complete the project
  (h3 : n = 30) -- Initial number of men
  (h4 : m = 2) -- Length of the road completed in x days
  (h5 : e = n + 30) -- New number of men
  (h6 : (4 : ℚ) / x = (8 : ℚ) / (d - x)) : x = 5 :=
by
  -- The proof would go here.
  sorry

end engineer_days_l2235_223509


namespace difference_in_combined_area_l2235_223594

-- Define the dimensions of the two rectangular sheets of paper
def paper1_length : ℝ := 11
def paper1_width : ℝ := 17
def paper2_length : ℝ := 8.5
def paper2_width : ℝ := 11

-- Define the areas of one side of each sheet
def area1 : ℝ := paper1_length * paper1_width -- 187
def area2 : ℝ := paper2_length * paper2_width -- 93.5

-- Define the combined areas of front and back of each sheet
def combined_area1 : ℝ := 2 * area1 -- 374
def combined_area2 : ℝ := 2 * area2 -- 187

-- Prove that the difference in combined area is 187
theorem difference_in_combined_area : combined_area1 - combined_area2 = 187 :=
by 
  -- Using the definitions above to simplify the goal
  sorry

end difference_in_combined_area_l2235_223594


namespace question_solution_l2235_223549

theorem question_solution 
  (hA : -(-1) = abs (-1))
  (hB : ¬ (∃ n : ℤ, ∀ m : ℤ, n < m ∧ m < 0))
  (hC : (-2)^3 = -2^3)
  (hD : ∃ q : ℚ, q = 0) :
  ¬ (∀ q : ℚ, q > 0 ∨ q < 0) := 
by {
  sorry
}

end question_solution_l2235_223549
