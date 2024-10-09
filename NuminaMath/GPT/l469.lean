import Mathlib

namespace number_of_small_slices_l469_46932

-- Define the given conditions
variables (S L : ℕ)
axiom total_slices : S + L = 5000
axiom total_revenue : 150 * S + 250 * L = 1050000

-- State the problem we need to prove
theorem number_of_small_slices : S = 1500 :=
by sorry

end number_of_small_slices_l469_46932


namespace projection_of_a_onto_b_l469_46985

open Real

def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (-2, 4)

theorem projection_of_a_onto_b :
  let dot_product := vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2
  let magnitude_b_squared := vector_b.1 ^ 2 + vector_b.2 ^ 2
  let scalar_projection := dot_product / magnitude_b_squared
  let proj_vector := (scalar_projection * vector_b.1, scalar_projection * vector_b.2)
  proj_vector = (-4/5, 8/5) :=
by
  sorry

end projection_of_a_onto_b_l469_46985


namespace vector_addition_correct_l469_46918

open Matrix

-- Define the vectors as 3x1 matrices
def v1 : Matrix (Fin 3) (Fin 1) ℤ := ![![3], ![-5], ![1]]
def v2 : Matrix (Fin 3) (Fin 1) ℤ := ![![-1], ![4], ![-2]]
def v3 : Matrix (Fin 3) (Fin 1) ℤ := ![![2], ![-1], ![3]]

-- Define the scalar multiples
def scaled_v1 := (2 : ℤ) • v1
def scaled_v2 := (3 : ℤ) • v2
def neg_v3 := (-1 : ℤ) • v3

-- Define the summation result
def result := scaled_v1 + scaled_v2 + neg_v3

-- Define the expected result for verification
def expected_result : Matrix (Fin 3) (Fin 1) ℤ := ![![1], ![3], ![-7]]

-- The proof statement (without the proof itself)
theorem vector_addition_correct :
  result = expected_result := by
  sorry

end vector_addition_correct_l469_46918


namespace time_difference_alice_bob_l469_46966

theorem time_difference_alice_bob
  (alice_speed : ℕ) (bob_speed : ℕ) (distance : ℕ)
  (h_alice_speed : alice_speed = 7)
  (h_bob_speed : bob_speed = 9)
  (h_distance : distance = 12) :
  (bob_speed * distance - alice_speed * distance) = 24 :=
by
  sorry

end time_difference_alice_bob_l469_46966


namespace pupils_correct_l469_46976

def totalPeople : ℕ := 676
def numberOfParents : ℕ := 22
def numberOfPupils : ℕ := totalPeople - numberOfParents

theorem pupils_correct :
  numberOfPupils = 654 := 
by
  sorry

end pupils_correct_l469_46976


namespace sum_of_squares_of_roots_l469_46978

theorem sum_of_squares_of_roots :
  (∃ r1 r2 : ℝ, (r1 + r2 = 10 ∧ r1 * r2 = 16) ∧ (r1^2 + r2^2 = 68)) :=
by
  sorry

end sum_of_squares_of_roots_l469_46978


namespace gcd_3_666666666_equals_3_l469_46914

theorem gcd_3_666666666_equals_3 :
  Nat.gcd 33333333 666666666 = 3 := by
  sorry

end gcd_3_666666666_equals_3_l469_46914


namespace solve_apples_problem_l469_46998

def apples_problem (marin_apples donald_apples total_apples : ℕ) : Prop :=
  marin_apples = 9 ∧ total_apples = 11 → donald_apples = 2

theorem solve_apples_problem : apples_problem 9 2 11 := by
  sorry

end solve_apples_problem_l469_46998


namespace distance_between_starting_points_l469_46986

theorem distance_between_starting_points :
  let speed1 := 70
  let speed2 := 80
  let start_time := 10 -- in hours (10 am)
  let meet_time := 14 -- in hours (2 pm)
  let travel_time := meet_time - start_time
  let distance1 := speed1 * travel_time
  let distance2 := speed2 * travel_time
  distance1 + distance2 = 600 :=
by
  sorry

end distance_between_starting_points_l469_46986


namespace find_x_eq_14_4_l469_46930

theorem find_x_eq_14_4 (x : ℝ) (h : ⌈x⌉ * x = 216) : x = 14.4 :=
by
  sorry

end find_x_eq_14_4_l469_46930


namespace probability_of_drawing_white_ball_l469_46919

theorem probability_of_drawing_white_ball (P_A P_B P_C : ℝ) 
    (hA : P_A = 0.4) 
    (hB : P_B = 0.25)
    (hSum : P_A + P_B + P_C = 1) : 
    P_C = 0.35 :=
by
    -- Placeholder for the proof
    sorry

end probability_of_drawing_white_ball_l469_46919


namespace evaluate_f_at_2_l469_46990

def f (x : ℝ) : ℝ := x^2 - 3 * x

theorem evaluate_f_at_2 :
  f 2 = -2 :=
by
  sorry

end evaluate_f_at_2_l469_46990


namespace unique_numbers_l469_46954

theorem unique_numbers (x y : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (S : x + y = 17) 
  (Q : x^2 + y^2 = 145) 
  : x = 8 ∧ y = 9 ∨ x = 9 ∧ y = 8 :=
by
  sorry

end unique_numbers_l469_46954


namespace not_dividable_by_wobbly_l469_46906

-- Define a wobbly number
def is_wobbly_number (n : ℕ) : Prop :=
  n > 0 ∧ (∀ k : ℕ, k < (Nat.log 10 n) → 
    (n / (10^k) % 10 ≠ 0 → n / (10^(k+1)) % 10 = 0) ∧
    (n / (10^k) % 10 = 0 → n / (10^(k+1)) % 10 ≠ 0))

-- Define sets of multiples of 10 and 25
def multiples_of (m : ℕ) (k : ℕ): Prop :=
  ∃ q : ℕ, k = q * m

def is_multiple_of_10 (k : ℕ) : Prop := multiples_of 10 k
def is_multiple_of_25 (k : ℕ) : Prop := multiples_of 25 k

theorem not_dividable_by_wobbly (n : ℕ) : 
  ¬ ∃ w : ℕ, is_wobbly_number w ∧ n ∣ w ↔ is_multiple_of_10 n ∨ is_multiple_of_25 n :=
by
  sorry

end not_dividable_by_wobbly_l469_46906


namespace intersection_A_compB_l469_46953

-- Define the sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 7}

-- Define the complement of B relative to ℝ
def comp_B : Set ℝ := {x | x ≤ 2 ∨ x ≥ 7}

-- State the main theorem to prove
theorem intersection_A_compB : A ∩ comp_B = {x | -3 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_A_compB_l469_46953


namespace tokens_per_pitch_l469_46989

theorem tokens_per_pitch 
  (tokens_macy : ℕ) (tokens_piper : ℕ)
  (hits_macy : ℕ) (hits_piper : ℕ)
  (misses_total : ℕ) (p : ℕ)
  (h1 : tokens_macy = 11)
  (h2 : tokens_piper = 17)
  (h3 : hits_macy = 50)
  (h4 : hits_piper = 55)
  (h5 : misses_total = 315)
  (h6 : 28 * p = hits_macy + hits_piper + misses_total) :
  p = 15 := 
by 
  sorry

end tokens_per_pitch_l469_46989


namespace johns_average_speed_l469_46912

-- Conditions
def biking_time_minutes : ℝ := 45
def biking_speed_mph : ℝ := 20
def walking_time_minutes : ℝ := 120
def walking_speed_mph : ℝ := 3

-- Proof statement
theorem johns_average_speed :
  let biking_time_hours := biking_time_minutes / 60
  let biking_distance := biking_speed_mph * biking_time_hours
  let walking_time_hours := walking_time_minutes / 60
  let walking_distance := walking_speed_mph * walking_time_hours
  let total_distance := biking_distance + walking_distance
  let total_time := biking_time_hours + walking_time_hours
  let average_speed := total_distance / total_time
  average_speed = 7.64 :=
by
  sorry

end johns_average_speed_l469_46912


namespace price_of_coffee_table_l469_46997

-- Define the given values
def price_sofa : ℕ := 1250
def price_armchair : ℕ := 425
def num_armchairs : ℕ := 2
def total_invoice : ℕ := 2430

-- Define the target value (price of the coffee table)
def price_coffee_table : ℕ := 330

-- The theorem to prove
theorem price_of_coffee_table :
  total_invoice = price_sofa + num_armchairs * price_armchair + price_coffee_table :=
by sorry

end price_of_coffee_table_l469_46997


namespace best_marksman_score_l469_46967

theorem best_marksman_score (n : ℕ) (hypothetical_score : ℕ) (average_if_hypothetical : ℕ) (actual_total_score : ℕ) (H1 : n = 8) (H2 : hypothetical_score = 92) (H3 : average_if_hypothetical = 84) (H4 : actual_total_score = 665) :
    ∃ (actual_best_score : ℕ), actual_best_score = 77 :=
by
    have hypothetical_total_score : ℕ := 7 * average_if_hypothetical + hypothetical_score
    have difference : ℕ := hypothetical_total_score - actual_total_score
    use hypothetical_score - difference
    sorry

end best_marksman_score_l469_46967


namespace least_positive_number_of_24x_plus_16y_is_8_l469_46925

theorem least_positive_number_of_24x_plus_16y_is_8 :
  ∃ (x y : ℤ), 24 * x + 16 * y = 8 :=
by
  sorry

end least_positive_number_of_24x_plus_16y_is_8_l469_46925


namespace john_personal_payment_l469_46955

-- Definitions of the conditions
def cost_of_one_hearing_aid : ℕ := 2500
def number_of_hearing_aids : ℕ := 2
def insurance_coverage_percent : ℕ := 80

-- Derived definitions based on conditions
def total_cost : ℕ := cost_of_one_hearing_aid * number_of_hearing_aids
def insurance_coverage_amount : ℕ := total_cost * insurance_coverage_percent / 100
def johns_share : ℕ := total_cost - insurance_coverage_amount

-- Theorem statement (proof not included)
theorem john_personal_payment : johns_share = 1000 :=
sorry

end john_personal_payment_l469_46955


namespace charlie_cortland_apples_l469_46999

/-- Given that Charlie picked 0.17 bags of Golden Delicious apples, 0.17 bags of Macintosh apples, 
   and a total of 0.67 bags of fruit, prove that the number of bags of Cortland apples picked by Charlie is 0.33. -/
theorem charlie_cortland_apples :
  let golden_delicious := 0.17
  let macintosh := 0.17
  let total_fruit := 0.67
  total_fruit - (golden_delicious + macintosh) = 0.33 :=
by
  sorry

end charlie_cortland_apples_l469_46999


namespace hyperbola_eccentricity_is_5_over_3_l469_46961

noncomputable def hyperbola_asymptote_condition (a b : ℝ) : Prop :=
  a / b = 3 / 4

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity_is_5_over_3 (a b : ℝ) (h : hyperbola_asymptote_condition a b) :
  hyperbola_eccentricity a b = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_is_5_over_3_l469_46961


namespace expression_evaluation_l469_46920

theorem expression_evaluation : abs (abs (-abs (-2 + 1) - 2) + 2) = 5 := 
by  
  sorry

end expression_evaluation_l469_46920


namespace no_positive_integer_pairs_l469_46950

theorem no_positive_integer_pairs (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) : ¬ (x^2 + y^2 = x^3 + 2 * y) :=
by sorry

end no_positive_integer_pairs_l469_46950


namespace original_price_l469_46942

theorem original_price (x : ℝ) (h : 0.9504 * x = 108) : x = 10800 / 9504 :=
by
  sorry

end original_price_l469_46942


namespace consecutive_odd_split_l469_46940

theorem consecutive_odd_split (m : ℕ) (hm : m > 1) : (∃ n : ℕ, n = 2015 ∧ n < ((m + 2) * (m - 1)) / 2) → m = 45 :=
by
  sorry

end consecutive_odd_split_l469_46940


namespace birds_in_sanctuary_l469_46945

theorem birds_in_sanctuary (x y : ℕ) 
    (h1 : x + y = 200)
    (h2 : 2 * x + 4 * y = 590) : 
    x = 105 :=
by
  sorry

end birds_in_sanctuary_l469_46945


namespace systematic_sampling_l469_46957

theorem systematic_sampling (total_employees groups group_size draw_5th draw_10th : ℕ)
  (h1 : total_employees = 200)
  (h2 : groups = 40)
  (h3 : group_size = total_employees / groups)
  (h4 : draw_5th = 22)
  (h5 : ∃ x : ℕ, draw_5th = (5-1) * group_size + x)
  (h6 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ groups → draw_10th = (k-1) * group_size + x) :
  draw_10th = 47 := 
by
  sorry

end systematic_sampling_l469_46957


namespace geometric_sequence_increasing_l469_46931

theorem geometric_sequence_increasing {a : ℕ → ℝ} (r : ℝ) (h_pos : 0 < r) (h_geometric : ∀ n, a (n + 1) = r * a n) :
  (a 0 < a 1 ∧ a 1 < a 2) ↔ ∀ n m, n < m → a n < a m :=
by sorry

end geometric_sequence_increasing_l469_46931


namespace monotonically_increasing_range_of_a_l469_46958

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4 * x - 5)

theorem monotonically_increasing_range_of_a :
  ∀ (a : ℝ), (∀ x : ℝ, x > a → f x > f a) ↔ a ≥ 5 :=
by
  intro a
  unfold f
  sorry

end monotonically_increasing_range_of_a_l469_46958


namespace greatest_two_digit_multiple_of_7_l469_46903

theorem greatest_two_digit_multiple_of_7 : ∃ n, 10 ≤ n ∧ n < 100 ∧ n % 7 = 0 ∧ ∀ m, 10 ≤ m ∧ m < 100 ∧ m % 7 = 0 → n ≥ m := 
by
  sorry

end greatest_two_digit_multiple_of_7_l469_46903


namespace polar_equation_graph_l469_46973

theorem polar_equation_graph :
  ∀ (ρ θ : ℝ), (ρ > 0) → ((ρ - 1) * (θ - π) = 0) ↔ (ρ = 1 ∨ θ = π) :=
by
  sorry

end polar_equation_graph_l469_46973


namespace product_of_de_l469_46975

theorem product_of_de (d e : ℤ) (h1: ∀ (r : ℝ), r^2 - r - 1 = 0 → r^6 - (d : ℝ) * r - (e : ℝ) = 0) : 
  d * e = 40 :=
by
  sorry

end product_of_de_l469_46975


namespace inequality_solution_set_l469_46977

theorem inequality_solution_set : 
  {x : ℝ | (x - 2) * (x + 1) ≤ 0} = {x : ℝ | -1 < x ∧ x ≤ 2} :=
by
  sorry

end inequality_solution_set_l469_46977


namespace violet_balloons_remaining_l469_46926

def initial_count : ℕ := 7
def lost_count : ℕ := 3

theorem violet_balloons_remaining : initial_count - lost_count = 4 :=
by sorry

end violet_balloons_remaining_l469_46926


namespace distinct_intersections_count_l469_46946

theorem distinct_intersections_count :
  (∃ (x y : ℝ), (x + 2 * y = 7 ∧ 3 * x - 4 * y + 8 = 0) ∨ (x + 2 * y = 7 ∧ 4 * x + 5 * y - 20 = 0) ∨
                (x - 2 * y - 1 = 0 ∧ 3 * x - 4 * y = 8) ∨ (x - 2 * y - 1 = 0 ∧ 4 * x + 5 * y - 20 = 0)) ∧
  ∃ count : ℕ, count = 3 :=
by sorry

end distinct_intersections_count_l469_46946


namespace John_sells_each_wig_for_five_dollars_l469_46911

theorem John_sells_each_wig_for_five_dollars
  (plays : ℕ)
  (acts_per_play : ℕ)
  (wigs_per_act : ℕ)
  (wig_cost : ℕ)
  (total_cost : ℕ)
  (sold_wigs_cost : ℕ)
  (remaining_wigs_cost : ℕ) :
  plays = 3 ∧
  acts_per_play = 5 ∧
  wigs_per_act = 2 ∧
  wig_cost = 5 ∧
  total_cost = 150 ∧
  remaining_wigs_cost = 110 ∧
  total_cost - remaining_wigs_cost = sold_wigs_cost →
  (sold_wigs_cost / (plays * acts_per_play * wigs_per_act - remaining_wigs_cost / wig_cost)) = wig_cost :=
by sorry

end John_sells_each_wig_for_five_dollars_l469_46911


namespace evaluate_expression_l469_46947

theorem evaluate_expression (a b : ℝ) (h : (1/2 * a * (1:ℝ)^3 - 3 * b * 1 + 4 = 9)) :
  (1/2 * a * (-1:ℝ)^3 - 3 * b * (-1) + 4 = -1) := by
sorry

end evaluate_expression_l469_46947


namespace sequence_formula_l469_46988

theorem sequence_formula (S : ℕ → ℕ) (a : ℕ → ℕ) (h₁ : ∀ n, S n = n^2 + 1) :
  (a 1 = 2) ∧ (∀ n ≥ 2, a n = 2 * n - 1) := 
by
  sorry

end sequence_formula_l469_46988


namespace neg_int_solution_l469_46904

theorem neg_int_solution (x : ℤ) : -2 * x < 4 ↔ x = -1 :=
by
  sorry

end neg_int_solution_l469_46904


namespace gcd_39_91_l469_46962
-- Import the Mathlib library to ensure all necessary functions and theorems are available

-- Lean statement for proving the GCD of 39 and 91 is 13.
theorem gcd_39_91 : Nat.gcd 39 91 = 13 := by
  sorry

end gcd_39_91_l469_46962


namespace opposite_sqrt3_l469_46968

def opposite (x : ℝ) : ℝ := -x

theorem opposite_sqrt3 :
  opposite (Real.sqrt 3) = -Real.sqrt 3 :=
by
  sorry

end opposite_sqrt3_l469_46968


namespace david_is_30_l469_46979

-- Definitions representing the conditions
def uncleBobAge : ℕ := 60
def emilyAge : ℕ := (2 * uncleBobAge) / 3
def davidAge : ℕ := emilyAge - 10

-- Statement that represents the equivalence to be proven
theorem david_is_30 : davidAge = 30 :=
by
  sorry

end david_is_30_l469_46979


namespace probability_within_sphere_correct_l469_46971

noncomputable def probability_within_sphere : ℝ :=
  let cube_volume := (2 : ℝ) * (2 : ℝ) * (2 : ℝ)
  let sphere_volume := (4 * Real.pi / 3) * (0.5) ^ 3
  sphere_volume / cube_volume

theorem probability_within_sphere_correct (x y z : ℝ) 
  (hx1 : -1 ≤ x) (hx2 : x ≤ 1) 
  (hy1 : -1 ≤ y) (hy2 : y ≤ 1) 
  (hz1 : -1 ≤ z) (hz2 : z ≤ 1) 
  (hx_sq : x^2 ≤ 0.5) 
  (hxyz : x^2 + y^2 + z^2 ≤ 0.25) : 
  probability_within_sphere = Real.pi / 48 :=
by
  sorry

end probability_within_sphere_correct_l469_46971


namespace slope_points_eq_l469_46933

theorem slope_points_eq (m : ℚ) (h : ((m + 2) / (3 - m) = 2)) : m = 4 / 3 :=
sorry

end slope_points_eq_l469_46933


namespace coordinates_of_P_l469_46900

def A : ℝ × ℝ := (3, -4)
def B : ℝ × ℝ := (-9, 2)
def P : ℝ × ℝ := (-1, -2)

theorem coordinates_of_P : P = (1 / 3 • (B.1 - A.1) + 2 / 3 • A.1, 1 / 3 • (B.2 - A.2) + 2 / 3 • A.2) :=
by
    rw [A, B, P]
    sorry

end coordinates_of_P_l469_46900


namespace average_primes_4_to_15_l469_46965

theorem average_primes_4_to_15 :
  (5 + 7 + 11 + 13) / 4 = 9 :=
by sorry

end average_primes_4_to_15_l469_46965


namespace sally_gave_joan_5_balloons_l469_46948

theorem sally_gave_joan_5_balloons (x : ℕ) (h1 : 9 + x - 2 = 12) : x = 5 :=
by
  -- Proof is skipped
  sorry

end sally_gave_joan_5_balloons_l469_46948


namespace reflection_twice_is_identity_l469_46987

-- Define the reflection matrix R over the vector (1, 2)
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  -- Note: The specific definition of the reflection matrix over (1, 2) is skipped as we only need the final proof statement.
  sorry

-- Assign the reflection matrix R to variable R
def R := reflection_matrix

-- Prove that R^2 = I
theorem reflection_twice_is_identity : R * R = 1 := by
  sorry

end reflection_twice_is_identity_l469_46987


namespace sum_of_coefficients_l469_46936

theorem sum_of_coefficients (f : ℕ → ℕ) :
  (5 * 1 + 2)^7 = 823543 :=
by
  sorry

end sum_of_coefficients_l469_46936


namespace percentage_increase_l469_46910

theorem percentage_increase (P : ℕ) (x y : ℕ) (h1 : x = 5) (h2 : y = 7) 
    (h3 : (x * (1 + P / 100) / (y * (1 - 10 / 100))) = 20 / 21) : 
    P = 20 :=
by
  sorry

end percentage_increase_l469_46910


namespace man_year_of_birth_l469_46982

theorem man_year_of_birth (x : ℕ) (hx1 : (x^2 + x >= 1850)) (hx2 : (x^2 + x < 1900)) : (1850 + (x^2 + x - x)) = 1892 :=
by {
  sorry
}

end man_year_of_birth_l469_46982


namespace value_two_stddevs_less_l469_46929

theorem value_two_stddevs_less (μ σ : ℝ) (hμ : μ = 16.5) (hσ : σ = 1.5) : μ - 2 * σ = 13.5 :=
by
  rw [hμ, hσ]
  norm_num

end value_two_stddevs_less_l469_46929


namespace base_6_units_digit_l469_46928

def num1 : ℕ := 217
def num2 : ℕ := 45
def base : ℕ := 6

theorem base_6_units_digit :
  (num1 % base) * (num2 % base) % base = (num1 * num2) % base :=
by
  sorry

end base_6_units_digit_l469_46928


namespace set_complement_intersection_l469_46951

open Set

variable (U A B : Set ℕ)

theorem set_complement_intersection :
  U = {2, 3, 5, 7, 8} →
  A = {2, 8} →
  B = {3, 5, 8} →
  (U \ A) ∩ B = {3, 5} :=
by
  intros
  sorry

end set_complement_intersection_l469_46951


namespace radius_ratio_of_smaller_to_larger_l469_46922

noncomputable def ratio_of_radii (v_large v_small : ℝ) (R r : ℝ) (h_large : (4/3) * Real.pi * R^3 = v_large) (h_small : v_small = 0.25 * v_large) (h_small_sphere : (4/3) * Real.pi * r^3 = v_small) : ℝ :=
  let ratio := r / R
  ratio

theorem radius_ratio_of_smaller_to_larger (v_large : ℝ) (R r : ℝ) (h_large : (4/3) * Real.pi * R^3 = 576 * Real.pi) (h_small_sphere : (4/3) * Real.pi * r^3 = 0.25 * 576 * Real.pi) : r / R = 1 / (2^(2/3)) :=
by
  sorry

end radius_ratio_of_smaller_to_larger_l469_46922


namespace volume_of_remaining_solid_l469_46915

noncomputable def volume_cube_with_cylindrical_hole 
  (side_length : ℝ) (hole_diameter : ℝ) (π : ℝ := 3.141592653589793) : ℝ :=
  let V_cube := side_length^3
  let radius := hole_diameter / 2
  let height := side_length
  let V_cylinder := π * radius^2 * height
  V_cube - V_cylinder

theorem volume_of_remaining_solid 
  (side_length : ℝ)
  (hole_diameter : ℝ)
  (h₁ : side_length = 6) 
  (h₂ : hole_diameter = 3)
  (π : ℝ := 3.141592653589793) : 
  abs (volume_cube_with_cylindrical_hole side_length hole_diameter π - 173.59) < 0.01 :=
by
  sorry

end volume_of_remaining_solid_l469_46915


namespace necessarily_negative_expression_l469_46917

theorem necessarily_negative_expression
  (x y z : ℝ)
  (hx : 0 < x ∧ x < 1)
  (hy : -1 < y ∧ y < 0)
  (hz : 0 < z ∧ z < 1)
  : y - z < 0 :=
sorry

end necessarily_negative_expression_l469_46917


namespace days_considered_l469_46905

theorem days_considered (visitors_current : ℕ) (visitors_previous : ℕ) (total_visitors : ℕ)
  (h1 : visitors_current = 132) (h2 : visitors_previous = 274) (h3 : total_visitors = 406)
  (h_total : visitors_current + visitors_previous = total_visitors) :
  2 = 2 :=
by
  sorry

end days_considered_l469_46905


namespace proposition_not_true_at_9_l469_46983

variable {P : ℕ → Prop}

theorem proposition_not_true_at_9 (h : ∀ k : ℕ, k ≥ 1 → P k → P (k + 1)) (h10 : ¬P 10) : ¬P 9 :=
by
  sorry

end proposition_not_true_at_9_l469_46983


namespace escher_prints_probability_l469_46992

theorem escher_prints_probability :
  let total_pieces := 12
  let escher_pieces := 4
  let total_permutations := Nat.factorial total_pieces
  let block_permutations := 9 * Nat.factorial (total_pieces - escher_pieces)
  let probability := block_permutations / (total_pieces * Nat.factorial (total_pieces - 1))
  probability = 1 / 1320 :=
by
  let total_pieces := 12
  let escher_pieces := 4
  let total_permutations := Nat.factorial total_pieces
  let block_permutations := 9 * Nat.factorial (total_pieces - escher_pieces)
  let probability := block_permutations / (total_pieces * Nat.factorial (total_pieces - 1))
  sorry

end escher_prints_probability_l469_46992


namespace janet_family_needs_91_tickets_l469_46927

def janet_family_tickets (adults: ℕ) (children: ℕ) (roller_coaster_adult_tickets: ℕ) (roller_coaster_child_tickets: ℕ) 
  (giant_slide_adult_tickets: ℕ) (giant_slide_child_tickets: ℕ) (num_roller_coaster_rides_adult: ℕ) 
  (num_roller_coaster_rides_child: ℕ) (num_giant_slide_rides_adult: ℕ) (num_giant_slide_rides_child: ℕ) : ℕ := 
  (adults * roller_coaster_adult_tickets * num_roller_coaster_rides_adult) + 
  (children * roller_coaster_child_tickets * num_roller_coaster_rides_child) + 
  (1 * giant_slide_adult_tickets * num_giant_slide_rides_adult) + 
  (1 * giant_slide_child_tickets * num_giant_slide_rides_child)

theorem janet_family_needs_91_tickets :
  janet_family_tickets 2 2 7 5 4 3 3 2 5 3 = 91 := 
by 
  -- Calculations based on the given conditions (skipped in this statement)
  sorry

end janet_family_needs_91_tickets_l469_46927


namespace smallest_angle_of_trapezoid_l469_46923

theorem smallest_angle_of_trapezoid 
  (a d : ℝ) 
  (h1 : a + 3 * d = 140)
  (h2 : ∀ i j k l : ℝ, i + j = k + l → i + j = 180 ∧ k + l = 180) :
  a = 40 :=
by
  sorry

end smallest_angle_of_trapezoid_l469_46923


namespace minimum_jumps_to_cover_circle_l469_46974

/--
Given 2016 points arranged in a circle and the ability to jump either 2 or 3 points clockwise,
prove that the minimum number of jumps required to visit every point at least once and return to the starting 
point is 2017.
-/
theorem minimum_jumps_to_cover_circle (n : Nat) (h : n = 2016) : 
  ∃ (a b : Nat), 2 * a + 3 * b = n ∧ (a + b) = 2017 := 
sorry

end minimum_jumps_to_cover_circle_l469_46974


namespace angle_2016_216_in_same_quadrant_l469_46909

noncomputable def angle_in_same_quadrant (a b : ℝ) : Prop :=
  let normalized (x : ℝ) := x % 360
  normalized a = normalized b

theorem angle_2016_216_in_same_quadrant : angle_in_same_quadrant 2016 216 := by
  sorry

end angle_2016_216_in_same_quadrant_l469_46909


namespace tangents_form_rectangle_l469_46969

-- Define the first ellipse
def ellipse1 (a b x y : ℝ) : Prop := x^2 / a^4 + y^2 / b^4 = 1

-- Define the second ellipse
def ellipse2 (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define conjugate diameters through lines
def conjugate_diameters (a b m : ℝ) : Prop := True -- (You might want to further define what conjugate diameters imply here)

-- Prove the main statement
theorem tangents_form_rectangle
  (a b m : ℝ)
  (x1 y1 x2 y2 k1 k2 : ℝ)
  (h1 : ellipse1 a b x1 y1)
  (h2 : ellipse1 a b x2 y2)
  (h3 : ellipse2 a b x1 y1)
  (h4 : ellipse2 a b x2 y2)
  (conj1 : conjugate_diameters a b m)
  (tangent_slope1 : k1 = -b^2 / a^2 * (1 / m))
  (conj2 : conjugate_diameters a b (-b^4/a^4 * 1/m))
  (tangent_slope2 : k2 = -b^4 / a^4 * (1 / (-b^4/a^4 * (1/m))))
: k1 * k2 = -1 :=
sorry

end tangents_form_rectangle_l469_46969


namespace a_perfect_square_l469_46941

theorem a_perfect_square (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_div : 2 * a * b ∣ a^2 + b^2 - a) : ∃ k : ℕ, a = k^2 := 
sorry

end a_perfect_square_l469_46941


namespace prove_a_eq_0_prove_monotonic_decreasing_prove_m_ge_4_l469_46964

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (4 * x + a) / (x^2 + 1)

-- 1. Prove that a = 0 given that f(x) is an odd function
theorem prove_a_eq_0 (a : ℝ) (h : ∀ x : ℝ, f (-x) a = - f x a) : a = 0 := sorry

-- 2. Prove that f(x) = 4x / (x^2 + 1) is monotonically decreasing on [1, +∞) for x > 0
theorem prove_monotonic_decreasing (x : ℝ) (hx : x > 0) :
  ∀ (x1 x2 : ℝ), 1 ≤ x1 → x1 < x2 → (f x1 0) > (f x2 0) := sorry

-- 3. Prove that |f(x1) - f(x2)| ≤ m for all x1, x2 ∈ R implies m ≥ 4
theorem prove_m_ge_4 (m : ℝ) 
  (h : ∀ x1 x2 : ℝ, |f x1 0 - f x2 0| ≤ m) : m ≥ 4 := sorry

end prove_a_eq_0_prove_monotonic_decreasing_prove_m_ge_4_l469_46964


namespace correct_option_is_B_l469_46902

theorem correct_option_is_B :
  (∃ (A B C D : String), A = "√49 = -7" ∧ B = "√((-3)^2) = 3" ∧ C = "-√((-5)^2) = 5" ∧ D = "√81 = ±9" ∧
    (B = "√((-3)^2) = 3")) :=
by
  sorry

end correct_option_is_B_l469_46902


namespace unique_cubic_coefficients_l469_46949

noncomputable def cubic_function (a b c : ℝ) (x : ℝ) : ℝ := 4 * x^3 + a * x^2 + b * x + c

theorem unique_cubic_coefficients
  (a b c : ℝ)
  (h1 : ∀ x, -1 ≤ x ∧ x ≤ 1 → -1 ≤ cubic_function a b c x ∧ cubic_function a b c x ≤ 1) :
  (a = 0 ∧ b = -3 ∧ c = 0) :=
by
  sorry

end unique_cubic_coefficients_l469_46949


namespace expression_eq_16x_l469_46995

variable (x y z w : ℝ)

theorem expression_eq_16x
  (h1 : y = 2 * x)
  (h2 : z = 3 * y)
  (h3 : w = z + x) :
  x + y + z + w = 16 * x :=
sorry

end expression_eq_16x_l469_46995


namespace sqrt_sum_eq_fraction_l469_46939

-- Definitions as per conditions
def w : ℕ := 4
def x : ℕ := 9
def z : ℕ := 25

-- Main theorem statement
theorem sqrt_sum_eq_fraction : (Real.sqrt (w / x) + Real.sqrt (x / z) = 19 / 15) := by
  sorry

end sqrt_sum_eq_fraction_l469_46939


namespace interest_rate_l469_46924

variable (P : ℝ) (T : ℝ) (SI : ℝ)

theorem interest_rate (h_P : P = 535.7142857142857) (h_T : T = 4) (h_SI : SI = 75) :
    (SI / (P * T)) * 100 = 3.5 := by
  sorry

end interest_rate_l469_46924


namespace salary_proof_l469_46959

-- Defining the monthly salaries of the officials
def D_Dupon : ℕ := 6000
def D_Duran : ℕ := 8000
def D_Marten : ℕ := 5000

-- Defining the statements made by each official
def Dupon_statement1 : Prop := D_Dupon = 6000
def Dupon_statement2 : Prop := D_Duran = D_Dupon + 2000
def Dupon_statement3 : Prop := D_Marten = D_Dupon - 1000

def Duran_statement1 : Prop := D_Duran > D_Marten
def Duran_statement2 : Prop := D_Duran - D_Marten = 3000
def Duran_statement3 : Prop := D_Marten = 9000

def Marten_statement1 : Prop := D_Marten < D_Dupon
def Marten_statement2 : Prop := D_Dupon = 7000
def Marten_statement3 : Prop := D_Duran = D_Dupon + 3000

-- Defining the constraints about the number of truth and lies
def Told_the_truth_twice_and_lied_once : Prop :=
  (Dupon_statement1 ∧ Dupon_statement2 ∧ ¬Dupon_statement3) ∨
  (Dupon_statement1 ∧ ¬Dupon_statement2 ∧ Dupon_statement3) ∨
  (¬Dupon_statement1 ∧ Dupon_statement2 ∧ Dupon_statement3) ∨
  (Duran_statement1 ∧ Duran_statement2 ∧ ¬Duran_statement3) ∨
  (Duran_statement1 ∧ ¬Duran_statement2 ∧ Duran_statement3) ∨
  (¬Duran_statement1 ∧ Duran_statement2 ∧ Duran_statement3) ∨
  (Marten_statement1 ∧ Marten_statement2 ∧ ¬Marten_statement3) ∨
  (Marten_statement1 ∧ ¬Marten_statement2 ∧ Marten_statement3) ∨
  (¬Marten_statement1 ∧ Marten_statement2 ∧ Marten_statement3)

-- The final proof goal
theorem salary_proof : Told_the_truth_twice_and_lied_once →
  D_Dupon = 6000 ∧ D_Duran = 8000 ∧ D_Marten = 5000 := by 
  sorry

end salary_proof_l469_46959


namespace exp_13_pi_i_over_2_eq_i_l469_46913

theorem exp_13_pi_i_over_2_eq_i : Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end exp_13_pi_i_over_2_eq_i_l469_46913


namespace probability_of_four_card_success_l469_46991

example (cards : Fin 4) (pins : Fin 4) {attempts : ℕ}
  (h1 : ∀ (c : Fin 4) (p : Fin 4), attempts ≤ 3)
  (h2 : ∀ (c : Fin 4), ∃ (p : Fin 4), p ≠ c ∧ attempts ≤ 3) :
  ∃ (three_cards : Fin 3), attempts ≤ 3 :=
sorry

noncomputable def probability_success :
  ℚ := 23 / 24

theorem probability_of_four_card_success :
  probability_success = 23 / 24 :=
sorry

end probability_of_four_card_success_l469_46991


namespace find_x_l469_46980

def side_of_square_eq_twice_radius_of_larger_circle (s: ℝ) (r_l: ℝ) : Prop :=
  s = 2 * r_l

def radius_of_larger_circle_eq_x_minus_third_radius_of_smaller_circle (r_l: ℝ) (x: ℝ) (r_s: ℝ) : Prop :=
  r_l = x - (1 / 3) * r_s

def circumference_of_smaller_circle_eq (r_s: ℝ) (circumference: ℝ) : Prop :=
  2 * Real.pi * r_s = circumference

def side_squared_eq_area (s: ℝ) (area: ℝ) : Prop :=
  s^2 = area

noncomputable def value_of_x (r_s r_l: ℝ) : ℝ :=
  14 + 4 / (3 * Real.pi)

theorem find_x 
  (s r_l r_s x: ℝ)
  (h1: side_squared_eq_area s 784)
  (h2: side_of_square_eq_twice_radius_of_larger_circle s r_l)
  (h3: radius_of_larger_circle_eq_x_minus_third_radius_of_smaller_circle r_l x r_s)
  (h4: circumference_of_smaller_circle_eq r_s 8) :
  x = value_of_x r_s r_l :=
sorry

end find_x_l469_46980


namespace assoc_mul_l469_46944

-- Conditions from the problem
variables (x y z : Type) [Mul x] [Mul y] [Mul z]

theorem assoc_mul (a b c : x) : (a * b) * c = a * (b * c) := by sorry

end assoc_mul_l469_46944


namespace compare_points_l469_46952

def parabola (x : ℝ) : ℝ := -x^2 - 4 * x + 1

theorem compare_points (y₁ y₂ : ℝ) :
  parabola (-3) = y₁ →
  parabola (-2) = y₂ →
  y₁ < y₂ :=
by
  intros hy₁ hy₂
  sorry

end compare_points_l469_46952


namespace new_arithmetic_mean_l469_46907

theorem new_arithmetic_mean
  (seq : List ℝ)
  (h_seq_len : seq.length = 60)
  (h_mean : (seq.sum / 60 : ℝ) = 42)
  (h_removed : ∃ a b, a ∈ seq ∧ b ∈ seq ∧ a = 50 ∧ b = 60) :
  ((seq.erase 50).erase 60).sum / 58 = 41.55 := 
sorry

end new_arithmetic_mean_l469_46907


namespace range_of_m_l469_46981

theorem range_of_m (m : ℝ) : (2 + m > 0) ∧ (1 - m > 0) ∧ (2 + m > 1 - m) → -1/2 < m ∧ m < 1 :=
by
  intros h
  sorry

end range_of_m_l469_46981


namespace sum_of_geometric_numbers_l469_46972

def is_geometric (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  ∃ r : ℕ, r > 0 ∧ 
  (d2 = d1 * r) ∧ 
  (d3 = d2 * r) ∧ 
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3

theorem sum_of_geometric_numbers : 
  (∃ smallest largest : ℕ,
    (smallest = 124) ∧ 
    (largest = 972) ∧ 
    is_geometric (smallest) ∧ 
    is_geometric (largest)
  ) →
  124 + 972 = 1096 :=
by
  sorry

end sum_of_geometric_numbers_l469_46972


namespace determine_alpha_l469_46921

theorem determine_alpha (α : ℝ) (y : ℝ → ℝ) (h : ∀ x, y x = x^α) (hp : y 2 = Real.sqrt 2) : α = 1 / 2 :=
sorry

end determine_alpha_l469_46921


namespace weight_of_b_l469_46970

/--
Given:
1. The sum of weights (a, b, c) is 129 kg.
2. The sum of weights (a, b) is 80 kg.
3. The sum of weights (b, c) is 86 kg.

Prove that the weight of b is 37 kg.
-/
theorem weight_of_b (a b c : ℝ) 
  (h1 : a + b + c = 129) 
  (h2 : a + b = 80) 
  (h3 : b + c = 86) : 
  b = 37 :=
sorry

end weight_of_b_l469_46970


namespace least_possible_z_minus_x_l469_46935

theorem least_possible_z_minus_x (x y z : ℕ) 
  (hx_prime : Nat.Prime x) (hy_prime : Nat.Prime y) (hz_prime : Nat.Prime z)
  (hxy : x < y) (hyz : y < z) (hyx_gt_3: y - x > 3)
  (hx_even : x % 2 = 0) (hy_odd : y % 2 = 1) (hz_odd : z % 2 = 1) :
  z - x = 9 :=
sorry

end least_possible_z_minus_x_l469_46935


namespace monotonic_range_l469_46938

theorem monotonic_range (a : ℝ) :
  (∀ x y, 2 ≤ x ∧ x ≤ 3 ∧ 2 ≤ y ∧ y ≤ 3 ∧ x < y → (x^2 - 2*a*x + 3) < (y^2 - 2*a*y + 3))
  ∨ (∀ x y, 2 ≤ x ∧ x ≤ 3 ∧ 2 ≤ y ∧ y ≤ 3 ∧ x < y → (x^2 - 2*a*x + 3) > (y^2 - 2*a*y + 3))
  ↔ (a ≤ 2 ∨ a ≥ 3) :=
by
  sorry

end monotonic_range_l469_46938


namespace find_roots_range_l469_46994

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x ^ 2 + b * x + c

theorem find_roots_range 
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hx : -1 < -1/2 ∧ -1/2 < 0 ∧ 0 < 1/2 ∧ 1/2 < 1 ∧ 1 < 3/2 ∧ 3/2 < 2 ∧ 2 < 5/2 ∧ 5/2 < 3)
  (hy : ∀ {x : ℝ}, x = -1 → quadratic_function a b c x = -2 ∧
                   x = -1/2 → quadratic_function a b c x = -1/4 ∧
                   x = 0 → quadratic_function a b c x = 1 ∧
                   x = 1/2 → quadratic_function a b c x = 7/4 ∧
                   x = 1 → quadratic_function a b c x = 2 ∧
                   x = 3/2 → quadratic_function a b c x = 7/4 ∧
                   x = 2 → quadratic_function a b c x = 1 ∧
                   x = 5/2 → quadratic_function a b c x = -1/4 ∧
                   x = 3 → quadratic_function a b c x = -2) :
  ∃ x1 x2 : ℝ, -1/2 < x1 ∧ x1 < 0 ∧ 2 < x2 ∧ x2 < 5/2 ∧ quadratic_function a b c x1 = 0 ∧ quadratic_function a b c x2 = 0 :=
by sorry

end find_roots_range_l469_46994


namespace total_red_marbles_l469_46993

-- Definitions derived from the problem conditions
def Jessica_red_marbles : ℕ := 3 * 12
def Sandy_red_marbles : ℕ := 4 * Jessica_red_marbles
def Alex_red_marbles : ℕ := Jessica_red_marbles + 2 * 12

-- Statement we need to prove that total number of marbles is 240
theorem total_red_marbles : 
  Jessica_red_marbles + Sandy_red_marbles + Alex_red_marbles = 240 := by
  -- We provide the proof later
  sorry

end total_red_marbles_l469_46993


namespace LemonadeCalories_l469_46960

noncomputable def total_calories (lemon_juice sugar water honey : ℕ) (cal_per_100g_lemon_juice cal_per_100g_sugar cal_per_100g_honey : ℕ) : ℝ :=
  (lemon_juice / 100) * cal_per_100g_lemon_juice +
  (sugar / 100) * cal_per_100g_sugar +
  (honey / 100) * cal_per_100g_honey

noncomputable def calories_in_250g (total_calories : ℝ) (total_weight : ℕ) : ℝ :=
  (total_calories / total_weight) * 250

theorem LemonadeCalories :
  let lemon_juice := 150
  let sugar := 200
  let water := 300
  let honey := 50
  let cal_per_100g_lemon_juice := 25
  let cal_per_100g_sugar := 386
  let cal_per_100g_honey := 64
  let total_weight := lemon_juice + sugar + water + honey
  let total_cal := total_calories lemon_juice sugar water honey cal_per_100g_lemon_juice cal_per_100g_sugar cal_per_100g_honey
  calories_in_250g total_cal total_weight = 301 :=
by
  sorry

end LemonadeCalories_l469_46960


namespace tv_episode_length_l469_46937

theorem tv_episode_length :
  ∀ (E : ℕ), 
    600 = 3 * E + 270 + 2 * 105 + 45 → 
    E = 25 :=
by
  intros E h
  sorry

end tv_episode_length_l469_46937


namespace measure_angle_E_l469_46956

-- Definitions based on conditions
variables {p q : Type} {A B E : ℝ}

noncomputable def measure_A (A B : ℝ) : ℝ := A
noncomputable def measure_B (A B : ℝ) : ℝ := 9 * A
noncomputable def parallel_lines (p q : Type) : Prop := true

-- Condition: measure of angle A is 1/9 of the measure of angle B
axiom angle_condition : A = (1 / 9) * B

-- Condition: p is parallel to q
axiom parallel_condition : parallel_lines p q

-- Prove that the measure of angle E is 18 degrees
theorem measure_angle_E (y : ℝ) (h1 : A = y) (h2 : B = 9 * y) : E = 18 :=
by
  sorry

end measure_angle_E_l469_46956


namespace train_speeds_l469_46996

-- Definitions used in conditions
def initial_distance : ℝ := 300
def time_elapsed : ℝ := 2
def remaining_distance : ℝ := 40
def speed_difference : ℝ := 10

-- Stating the problem in Lean
theorem train_speeds :
  ∃ (v_fast v_slow : ℝ),
    v_slow + speed_difference = v_fast ∧
    (2 * (v_slow + v_fast)) = (initial_distance - remaining_distance) ∧
    v_slow = 60 ∧
    v_fast = 70 :=
by
  sorry

end train_speeds_l469_46996


namespace inequality_proof_l469_46963

open Real

theorem inequality_proof (x y z: ℝ) (hx: 0 ≤ x) (hy: 0 ≤ y) (hz: 0 ≤ z) : 
  (x^3 / (x^3 + 2 * y^2 * sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * sqrt (y * z)) 
  ) ≥ 1 := 
by
  sorry

end inequality_proof_l469_46963


namespace triangle_is_isosceles_l469_46916

theorem triangle_is_isosceles
    (a b c : ℝ)
    (A B C : ℝ)
    (h1 : a = 2 * c * Real.cos B)
    (h2 : b = c * Real.cos A) 
    (h3 : c = a * Real.cos C) 
    : a = b := 
sorry

end triangle_is_isosceles_l469_46916


namespace austin_tax_l469_46934

theorem austin_tax 
  (number_of_robots : ℕ)
  (cost_per_robot change_left starting_amount : ℚ) 
  (h1 : number_of_robots = 7)
  (h2 : cost_per_robot = 8.75)
  (h3 : change_left = 11.53)
  (h4 : starting_amount = 80) : 
  ∃ tax : ℚ, tax = 7.22 :=
by
  sorry

end austin_tax_l469_46934


namespace find_first_term_l469_46901

variable {a r : ℚ}

theorem find_first_term (h1 : a / (1 - r) = 30) (h2 : a^2 / (1 - r^2) = 120) : a = 240 / 7 :=
by
  sorry

end find_first_term_l469_46901


namespace find_d_l469_46943

variable (d x : ℕ)
axiom balls_decomposition : d = x + (x + 1) + (x + 2)
axiom probability_condition : (x : ℚ) / (d : ℚ) < 1 / 6

theorem find_d : d = 3 := sorry

end find_d_l469_46943


namespace point_returns_to_original_after_seven_steps_l469_46908

-- Define a structure for a triangle and a point inside it
structure Triangle :=
  (A B C : Point)

structure Point :=
  (x y : ℝ)

-- Given a triangle and a point inside it
variable (ABC : Triangle)
variable (M : Point)

-- Define the set of movements and the intersection points
def move_parallel_to_BC (M : Point) (ABC : Triangle) : Point := sorry
def move_parallel_to_AB (M : Point) (ABC : Triangle) : Point := sorry
def move_parallel_to_AC (M : Point) (ABC : Triangle) : Point := sorry

-- Function to perform the stepwise movement through 7 steps
def move_M_seven_times (M : Point) (ABC : Triangle) : Point :=
  let M1 := move_parallel_to_BC M ABC
  let M2 := move_parallel_to_AB M1 ABC 
  let M3 := move_parallel_to_AC M2 ABC
  let M4 := move_parallel_to_BC M3 ABC
  let M5 := move_parallel_to_AB M4 ABC
  let M6 := move_parallel_to_AC M5 ABC
  let M7 := move_parallel_to_BC M6 ABC
  M7

-- The theorem stating that after 7 steps, point M returns to its original position
theorem point_returns_to_original_after_seven_steps :
  move_M_seven_times M ABC = M := sorry

end point_returns_to_original_after_seven_steps_l469_46908


namespace hexagonal_tile_difference_l469_46984

theorem hexagonal_tile_difference :
  let initial_blue_tiles := 15
  let initial_green_tiles := 9
  let new_green_border_tiles := 18
  let new_blue_border_tiles := 18
  let total_green_tiles := initial_green_tiles + new_green_border_tiles
  let total_blue_tiles := initial_blue_tiles + new_blue_border_tiles
  total_blue_tiles - total_green_tiles = 6 := by {
    sorry
  }

end hexagonal_tile_difference_l469_46984
