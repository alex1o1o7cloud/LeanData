import Mathlib

namespace prove_inequality_holds_equality_condition_l761_761859

noncomputable def inequality_holds (m : ℕ) (x : Fin m → ℝ) : Prop :=
  (m - 1)^(m - 1) * (∑ i, x i ^ m)  ≥ (∑ i, x i) ^ m - m^m * (∏ i, x i)

theorem prove_inequality_holds (m : ℕ) (x : Fin m → ℝ) (hm : 2 ≤ m) (hx : ∀ i, 0 ≤ x i) : 
  inequality_holds m x := 
sorry

theorem equality_condition (m : ℕ) (x : Fin m → ℝ) (hm : 2 ≤ m) (hx : ∀ i, 0 ≤ x i) :
  inequality_holds m x ↔ ∀ i, x i = 1 / m := 
sorry

end prove_inequality_holds_equality_condition_l761_761859


namespace abs_negative_five_l761_761272

theorem abs_negative_five : abs (-5) = 5 :=
by
  sorry

end abs_negative_five_l761_761272


namespace simplify_expression_l761_761003

variable (m : ℝ)

theorem simplify_expression (h₁ : m ≠ 2) (h₂ : m ≠ 3) :
  (m - (4 * m - 9) / (m - 2)) / ((m ^ 2 - 9) / (m - 2)) = (m - 3) / (m + 3) := 
sorry

end simplify_expression_l761_761003


namespace unique_first_place_probability_at_least_half_l761_761169

theorem unique_first_place_probability_at_least_half {n : ℕ} : 
  (∃! c : fin (2 ^ n), true) → 
  (∀ x : fin n, ∃! g : fin (2 * n), true) → 
  ∃ (p : nnreal), p ≥ 1 / 2 := 
begin
  sorry
end

end unique_first_place_probability_at_least_half_l761_761169


namespace part_I_part_II_part_III_l761_761628

-- Definition of sequences and property P
def is_sequence_with_property_P (A : List ℕ) (n : ℕ) : Prop :=
  A.length = n ∧ (∀ a ∈ A, 1 ≤ a ∧ a ≤ n) ∧ A.nodup

-- Definition of T(A)
def T (A : List ℕ) : List ℕ :=
  List.map (λ (k : ℕ) => if A[k] < A[k + 1] then 1 else 0) (List.range (A.length - 1))

-- Part I
theorem part_I (A : List ℕ) (h_A : is_sequence_with_property_P A 4) (h_T : T A = [0, 1, 1]) :
  A = [4, 1, 2, 3] ∨ A = [3, 1, 2, 4] ∨ A = [2, 1, 3, 4] :=
sorry

-- Part II
theorem part_II (E : List ℕ) (h_E : E.length ≥ 1 ∧ ∀ e ∈ E, e ∈ {0, 1}) :
  ∃ A : List ℕ, (is_sequence_with_property_P A (E.length + 1)) ∧ (T A = E) :=
sorry

-- Part III
theorem part_III (A : List ℕ) (n : ℕ) (h_A : is_sequence_with_property_P A n) (h_n_ge_5 : n ≥ 5)
  (h_diff_abs : abs (A[0] - A[n-1]) = 1)
  (h_T_alternating : ∀ i, i < n - 1 →
    (i % 2 = 0 → T A[i] = 0) ∧ (i % 2 = 1 → T A[i] = 1)) :
  ∃ k, 2 * k = (List.permutations A).length :=
sorry

end part_I_part_II_part_III_l761_761628


namespace greatest_integer_less_than_or_equal_l761_761388

noncomputable def problem_expression : ℚ :=  (5^80 + 4^130) / (5^75 + 4^125)

theorem greatest_integer_less_than_or_equal :
  floor problem_expression = 3125 :=
sorry

end greatest_integer_less_than_or_equal_l761_761388


namespace angle_sum_bounds_l761_761437

variable (A B C D X : Point) (θ : ℝ)
variable [Field ℝ] [Fact (-1 ≤ θ ∧ θ ≤ 1)] -- θ represents an acute angle

-- Given conditions
noncomputable def points_subtend_angle (p q r s t : Point) (θ : ℝ) : Prop :=
  angle_subtended (p, q) t = θ ∧
  angle_subtended (q, r) t = θ ∧
  angle_subtended (r, s) t = θ ∧
  angle_subtended (s, p) t = θ

-- Main theorem
theorem angle_sum_bounds (h : points_subtend_angle A B C D X θ) :
  (∠AXC + ∠BXD = 0) ∨
  (∠AXC + ∠BXD = 2 * Real.arccos (2 * Real.cos θ - 1)) :=
sorry

end angle_sum_bounds_l761_761437


namespace average_sqft_per_person_texas_l761_761936

theorem average_sqft_per_person_texas :
  let population := 17000000
  let area_sqmiles := 268596
  let usable_land_percentage := 0.8
  let sqfeet_per_sqmile := 5280 * 5280
  let total_sqfeet := area_sqmiles * sqfeet_per_sqmile
  let usable_sqfeet := usable_land_percentage * total_sqfeet
  let avg_sqfeet_per_person := usable_sqfeet / population
  352331 <= avg_sqfeet_per_person ∧ avg_sqfeet_per_person < 500000 :=
by
  sorry

end average_sqft_per_person_texas_l761_761936


namespace radius_of_large_circle_l761_761499

-- Define a larger circle with centers and their relations
variables {R : ℝ} -- Radius of the large circle
variables (C₁ C₂ C₃ C₄ O : ℂ) -- Centers of the circles

-- Assumptions based on the problem statement
def is_tangent (x y : ℂ) (r₁ r₂ : ℝ) : Prop := abs (x - y) = r₁ + r₂

-- Given conditions
variable (radius_small : ℝ := 2) -- Radius of smaller circles

-- External tangency conditions between small circles
axiom tangent_C₁_C₂ : is_tangent C₁ C₂ radius_small radius_small
axiom tangent_C₂_C₃ : is_tangent C₂ C₃ radius_small radius_small
axiom tangent_C₃_C₄ : is_tangent C₃ C₄ radius_small radius_small
axiom tangent_C₄_C₁ : is_tangent C₄ C₁ radius_small radius_small

-- Internal tangency conditions to the large circle
axiom tangent_O_C₁ : is_tangent O C₁ R radius_small
axiom tangent_O_C₂ : is_tangent O C₂ R radius_small
axiom tangent_O_C₃ : is_tangent O C₃ R radius_small
axiom tangent_O_C₄ : is_tangent O C₄ R radius_small

-- The theorem to be proved
theorem radius_of_large_circle : R = 2 * (Real.sqrt 2 + 1) :=
sorry -- The proof goes here

end radius_of_large_circle_l761_761499


namespace powers_of_7_units_digit_cyclic_units_digit_2137_pow_753_l761_761717

-- Definition for units digit operation
def units_digit (a : ℕ) : ℕ := a % 10

-- Cyclic property observed in powers of 7
theorem powers_of_7_units_digit_cyclic : ∀ (n : ℕ), ∃ r < 4, units_digit (7 ^ n) = list.nth_le [1, 7, 9, 3] r (by norm_num) :=
by sorry

-- Main theorem
theorem units_digit_2137_pow_753 : units_digit (2137 ^ 753) = 7 :=
by sorry

end powers_of_7_units_digit_cyclic_units_digit_2137_pow_753_l761_761717


namespace engineer_walk_duration_l761_761448

variables (D : ℕ) (S : ℕ) (v : ℕ) (t : ℕ) (t1 : ℕ)

-- Stating the conditions
-- The time car normally takes to travel distance D
-- Speed (S) times the time (t) equals distance (D)
axiom speed_distance_relation : S * t = D

-- Engineer arrives at station at 7:00 AM and walks towards the car
-- They meet at t1 minutes past 7:00 AM, and the car covers part of the distance
-- Engineer reaches factory 20 minutes earlier than usual
-- Therefore, the car now meets the engineer covering less distance and time
axiom car_meets_engineer : S * t1 + v * t1 = D

-- The total travel time to the factory is reduced by 20 minutes
axiom travel_time_reduction : t - t1 = (t - 20 / 60)

-- Mathematically equivalent proof problem
theorem engineer_walk_duration : t1 = 50 := by
  sorry

end engineer_walk_duration_l761_761448


namespace solve_right_angled_parallelograms_l761_761814

def right_angled_parallelograms (S p s s' : ℝ) : Prop :=
  ∃ (x y z u : ℝ),
  (xy + zu = S) ∧
  (x + z = p) ∧
  (zy = s) ∧
  (xu = s') ∧
  (x = (p * (2 * s' + S) + sqrt (p^2 * (2 * s' + S)^2 - 4 * p^2 * s' * (s + s' + S))) / (2 * (s + s' + S))) ∨
  (x = (p * (2 * s' + S) - sqrt (p^2 * (2 * s' + S)^2 - 4 * p^2 * s' * (s + s' + S))) / (2 * (s + s' + S))) ∧
  (z = (p * (2 * s + S) - sqrt (S^2 - 4 * s * s')) / (2 * (s + s' + S))) ∨
  (z = (p * (2 * s + S) + sqrt (S^2 - 4 * s * s')) / (2 * (s + s' + S))) ∧
  (y = s / z) ∧
  (u = s' / x)

theorem solve_right_angled_parallelograms (S p s s' : ℝ) : right_angled_parallelograms S p s s' :=
  sorry

end solve_right_angled_parallelograms_l761_761814


namespace find_x_for_abs_expression_zero_l761_761815

theorem find_x_for_abs_expression_zero (x : ℚ) : |5 * x - 2| = 0 → x = 2 / 5 := by
  sorry

end find_x_for_abs_expression_zero_l761_761815


namespace three_digit_multiples_of_25_not_75_count_l761_761136

-- Definitions from conditions.
def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000
def is_multiple_of_25 (n : ℕ) : Prop := n % 25 = 0
def is_multiple_of_75 (n : ℕ) : Prop := n % 75 = 0

-- The theorem statement.
theorem three_digit_multiples_of_25_not_75_count : 
  let count := (finset.filter (λ n, is_three_digit n ∧ is_multiple_of_25 n ∧ ¬ is_multiple_of_75 n) (finset.range 1000)).card
  in count = 24 :=
by
  sorry

end three_digit_multiples_of_25_not_75_count_l761_761136


namespace field_trip_classrooms_count_l761_761312

variable (students : ℕ) (seats_per_bus : ℕ) (number_of_buses : ℕ) (total_classrooms : ℕ)

def fieldTrip 
    (students := 58)
    (seats_per_bus := 2)
    (number_of_buses := 29)
    (total_classrooms := 2) : Prop :=
  students = seats_per_bus * number_of_buses  ∧ total_classrooms = students / (students / total_classrooms)

theorem field_trip_classrooms_count : fieldTrip := by
  -- Proof goes here
  sorry

end field_trip_classrooms_count_l761_761312


namespace Tim_change_l761_761699

theorem Tim_change :
  ∀ (initial_amount paid_amount change : ℕ),
    initial_amount = 50 →
    paid_amount = 45 →
    change = initial_amount - paid_amount →
    change = 5 :=
by
  intros initial_amount paid_amount change h_initial h_paid h_change
  rw [h_initial, h_paid] at h_change
  simp at h_change
  assumption

# Emacs-Lisp

end Tim_change_l761_761699


namespace delta_inequality_of_minimal_covering_matrix_space_exists_minimal_covering_matrix_space_l761_761966

section covering_matrix_space

variables {m p : ℕ}
variables {𝔽 : Type*} [Field 𝔽]
variables {M : Type*} [add_comm_group M] [module 𝔽 M]

-- Define the vector space of m x p matrices
def matrix_space (m p : ℕ) := vector (vector ℝ p) m

-- Definitions according to conditions given
def delta (S : set (matrix_space m p)) : ℕ := 
  vector_space.dim ℝ (span ℝ (⋃ (A ∈ S), column_space A))

def is_covering_matrix_space (T : set (matrix_space m p)) : Prop :=
  (⋃ (A ∈ T) (hA : A ≠ 0), ker A) = set.univ

def is_minimal_covering_matrix_space (T : set (matrix_space m p)) : Prop :=
  is_covering_matrix_space T ∧ ∀ (S : set (matrix_space m p)), 
    is_covering_matrix_space S → S ⊆ T → S = T

variable (T : set (matrix_space m p))

-- Problem (a)
theorem delta_inequality_of_minimal_covering_matrix_space 
  (hTmin : is_minimal_covering_matrix_space T) 
  (n : ℕ) (h_dim : vector_space.dim ℝ (span ℝ T) = n) :
  delta T ≤ n.choose 2 :=
sorry

-- Problem (b)
theorem exists_minimal_covering_matrix_space (n : ℕ) :
  ∃ (m p : ℕ) (T : set (matrix_space m p)), 
    vector_space.dim ℝ (span ℝ T) = n ∧
    is_minimal_covering_matrix_space T ∧ 
    delta T = n.choose 2 :=
sorry

end covering_matrix_space

end delta_inequality_of_minimal_covering_matrix_space_exists_minimal_covering_matrix_space_l761_761966


namespace cos_angle_AMB_l761_761209

noncomputable section

open Real

variables (s : ℝ)
def A : EuclideanSpace 3 := ![0, 0, 0]
def B : EuclideanSpace 3 := ![s, s, s]
def E : EuclideanSpace 3 := ![s, 0, s]
def F : EuclideanSpace 3 := ![s, s, 0]
def M : EuclideanSpace 3 := ![s, s / 2, s / 2]

def AM : EuclideanSpace 3 := M - A
def BM : EuclideanSpace 3 := M - B

noncomputable def cosAngleAMB : ℝ :=
  let am_dot_bm := (AM s) ⬝ (BM s)
  let norm_am := ∥AM s∥
  let norm_bm := ∥BM s∥
  am_dot_bm / (norm_am * norm_bm)

theorem cos_angle_AMB :
  cosAngleAMB s = -1 / (3 * sqrt 2) :=  
sorry

end cos_angle_AMB_l761_761209


namespace graph_of_9x2_minus_16y2_is_pair_of_straight_lines_l761_761666

theorem graph_of_9x2_minus_16y2_is_pair_of_straight_lines :
  (let A := 9
    let B := 0
    let C := -16
    let equation : _ := A * x^2 + C * y^2
    let delta := B^2 - 4 * A * C
     in delta > 0) →
  ∃ a b c : ℝ, ∀ x y : ℝ, 9 * x ^ 2 - 16 * y ^ 2 = 0 ↔ (3 * x + 4 * y = 0) ∨ (3 * x - 4 * y = 0) :=
by
  sorry

end graph_of_9x2_minus_16y2_is_pair_of_straight_lines_l761_761666


namespace fraction_of_oranges_is_correct_l761_761321

variable (O P A : ℕ)
variable (total_fruit : ℕ := 56)

theorem fraction_of_oranges_is_correct:
  (A = 35) →
  (P = O / 2) →
  (A = 5 * P) →
  (O + P + A = total_fruit) →
  (O / total_fruit = 1 / 4) :=
by
  -- proof to be filled in 
  sorry

end fraction_of_oranges_is_correct_l761_761321


namespace keiko_speed_is_pi_over_2_l761_761612
-- Import the necessary library

-- Define the conditions
def outer_track_time (inner_radius : ℝ) (width : ℝ) (extra_time : ℝ) (keiko_speed : ℝ) : Prop :=
  let inner_semi_circumference := π * inner_radius
  let outer_semi_circumference := π * (inner_radius + width)
  let inner_quarter_circumference := (π / 2) * inner_radius
  let outer_quarter_circumference := (π / 2) * (inner_radius + width)
  let distance_difference := (outer_semi_circumference + outer_quarter_circumference) - (inner_semi_circumference + inner_quarter_circumference)
  distance_difference / keiko_speed = extra_time

-- State the proof problem
theorem keiko_speed_is_pi_over_2 (r : ℝ) (extra_time : ℝ) (w : ℝ) : 
  outer_track_time r w extra_time (π / 2) :=
by
  dsimp [outer_track_time]
  sorry

end keiko_speed_is_pi_over_2_l761_761612


namespace chloe_profit_l761_761459

theorem chloe_profit 
  (cost_per_dozen : ℕ)
  (selling_price_per_half_dozen : ℕ)
  (dozens_sold : ℕ)
  (h1 : cost_per_dozen = 50)
  (h2 : selling_price_per_half_dozen = 30)
  (h3 : dozens_sold = 50) : 
  (selling_price_per_half_dozen - cost_per_dozen / 2) * (dozens_sold * 2) = 500 :=
by 
  sorry

end chloe_profit_l761_761459


namespace shaded_area_l761_761181

noncomputable def semicircle_area (diameter : ℝ) : ℝ :=
  (π * diameter^2) / 8

def smaller_semicircle_area : ℝ := semicircle_area 5
def larger_semicircle_area : ℝ := semicircle_area 10
def largest_semicircle_area : ℝ := semicircle_area 30

theorem shaded_area :
  let total_area := 4 * smaller_semicircle_area + 2 * larger_semicircle_area
  in largest_semicircle_area - total_area = (175 / 2) * π :=
by
  -- Placeholder for the proof
  sorry

end shaded_area_l761_761181


namespace sine_shift_l761_761327

theorem sine_shift (x : ℝ) : sin (x + π / 3) = sin (x + π / 3) :=
by sorry

end sine_shift_l761_761327


namespace range_of_f_on_interval_l761_761889

def f (x : ℝ) : ℝ := x^2 - 6*x - 9

theorem range_of_f_on_interval :
  set.range (λ x, f x) ∩ set.Ioo (1 : ℝ) 4 = set.Ioo (-18 : ℝ) (-14) :=
by
  sorry

end range_of_f_on_interval_l761_761889


namespace max_profit_l761_761168

def fixed_cost : ℝ := 300
def revenue (x : ℝ) : ℝ := 80 * x
def g (x : ℝ) : ℝ := if 0 < x ∧ x ≤ 4 then 20 else x^2 + 40 * x - 100

def W (x : ℝ) : ℝ := 
if 0 < x ∧ x ≤ 4 then revenue x - (g x + fixed_cost)
else revenue x - (g x + fixed_cost)

theorem max_profit : 
  (∀ x : ℝ, 0 < x ∧ x ≤ 4 → W x = 80 * x - 320) ∧ 
  (∀ x : ℝ, x > 4 → W x = -x^2 + 40 * x - 200) ∧
  (W 20 = 200) :=
by 
  sorry

end max_profit_l761_761168


namespace greatest_power_of_3_in_factorial_l761_761724

theorem greatest_power_of_3_in_factorial :
  ∃ k : ℕ, (∀ n : ℕ, (∃ m : ℕ, 30! = 3^m * n → m ≤ k) ∧ k = 14) :=
by
  sorry

end greatest_power_of_3_in_factorial_l761_761724


namespace ball_reaches_height_l761_761409

theorem ball_reaches_height (h₀ : ℝ) (ratio : ℝ) (target_height : ℝ) (bounces : ℕ) 
  (initial_height : h₀ = 16) 
  (bounce_ratio : ratio = 1/3) 
  (target : target_height = 2) 
  (bounce_count : bounces = 7) :
  h₀ * (ratio ^ bounces) < target_height := 
sorry

end ball_reaches_height_l761_761409


namespace sum_real_solutions_sqrt_eq_l761_761148

theorem sum_real_solutions_sqrt_eq (b : ℝ) (hb : b > 4) :
  (∃ x : ℝ, (√(b - √(b + x)) = x + 1) ∧ (∀ y : ℝ, (√(b - √(b + y)) = y + 1) → x = y)) →
  x + y = √(b - 1) - 1 :=
sorry

end sum_real_solutions_sqrt_eq_l761_761148


namespace vasya_numbers_l761_761381

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : 
  (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) ∨ (x = y ∧ x = 0) :=
by 
  -- Placeholder to show where the proof would go
  sorry

end vasya_numbers_l761_761381


namespace find_real_roots_l761_761821

theorem find_real_roots : 
  {x : ℝ | x^9 + (9 / 8) * x^6 + (27 / 64) * x^3 - x + (219 / 512) = 0} =
  {1 / 2, (-1 + Real.sqrt 13) / 4, (-1 - Real.sqrt 13) / 4} :=
by
  sorry

end find_real_roots_l761_761821


namespace complex_inequality_l761_761221

theorem complex_inequality (n : ℕ) (a : Fin n → ℝ) (z : Fin n → ℂ) :
  abs (∑ j, a j * z j) ^ 2 ≤ 
  (1/2) * (∑ j, a j ^ 2) * ((∑ j, abs (z j) ^ 2) + abs (∑ j, z j ^ 2)) :=
by
  sorry

end complex_inequality_l761_761221


namespace binom_150_1_eq_150_l761_761014

/-- Definition of factorial -/
def fact : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * fact n

/-- Definition of binomial coefficient -/
def binom (n k : ℕ) : ℕ :=
fact n / (fact k * fact (n - k))

/-- Theorem stating the specific binomial coefficient calculation -/
theorem binom_150_1_eq_150 : binom 150 1 = 150 := by
  sorry

end binom_150_1_eq_150_l761_761014


namespace sum_of_first_50_odd_numbers_l761_761818

theorem sum_of_first_50_odd_numbers (h : (∑ k in finset.range 75, 2 * k + 1) = 5625) : 
  (∑ k in finset.range 50, 2 * k + 1) = 2500 :=
by
  sorry

end sum_of_first_50_odd_numbers_l761_761818


namespace positive_difference_of_perimeters_is_zero_l761_761804

-- Definitions of given conditions
def rect1_length : ℕ := 5
def rect1_width : ℕ := 1
def rect2_first_rect_length : ℕ := 3
def rect2_first_rect_width : ℕ := 2
def rect2_second_rect_length : ℕ := 1
def rect2_second_rect_width : ℕ := 2

-- Perimeter calculation functions
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)
def rect1_perimeter := perimeter rect1_length rect1_width
def rect2_extended_length : ℕ := rect2_first_rect_length + rect2_second_rect_length
def rect2_extended_width : ℕ := rect2_first_rect_width
def rect2_perimeter := perimeter rect2_extended_length rect2_extended_width

-- The positive difference of the perimeters
def positive_difference (a b : ℕ) : ℕ := if a > b then a - b else b - a

-- The Lean 4 statement to be proven
theorem positive_difference_of_perimeters_is_zero :
    positive_difference rect1_perimeter rect2_perimeter = 0 := by
  sorry

end positive_difference_of_perimeters_is_zero_l761_761804


namespace parabola_chord_length_l761_761483

theorem parabola_chord_length (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (h1 : y1^2 = 4 * x1) (h2 : y2^2 = 4 * x2) 
  (hx : x1 + x2 = 9) 
  (focus_line : ∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b → y^2 = 4 * x) :
  |(x1 - 1, y1) - (x2 - 1, y2)| = 11 := 
sorry

end parabola_chord_length_l761_761483


namespace prob1_part1_prob1_part2_l761_761992

noncomputable def U : Set ℝ := Set.univ
noncomputable def A : Set ℝ := {x | -2 < x ∧ x < 5}
noncomputable def B (a : ℝ) : Set ℝ := {x | 2 - a < x ∧ x < 1 + 2 * a}

theorem prob1_part1 (a : ℝ) (ha : a = 3) :
  A ∪ B a = {x | -2 < x ∧ x < 7} ∧ A ∩ B a = {x | -1 < x ∧ x < 5} :=
by {
  sorry
}

theorem prob1_part2 (h : ∀ x, x ∈ A → x ∈ B a) :
  ∀ a : ℝ, a ≤ 2 :=
by {
  sorry
}

end prob1_part1_prob1_part2_l761_761992


namespace Vasya_numbers_l761_761358

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 :=
by
  sorry

end Vasya_numbers_l761_761358


namespace num_distinct_convex_polygons_l761_761056

theorem num_distinct_convex_polygons (n : ℕ) (h : n = 15) :
  let total_subsets := 2 ^ n in
  let subsets_with_0_points := (nat.choose n 0) in
  let subsets_with_1_point := (nat.choose n 1) in
  let subsets_with_2_points := (nat.choose n 2) in
  total_subsets - subsets_with_0_points - subsets_with_1_point - subsets_with_2_points = 32647 :=
by
  have h1 : total_subsets = 2 ^ n := rfl
  have h2 : subsets_with_0_points = (nat.choose n 0) := rfl
  have h3 : subsets_with_1_point = (nat.choose n 1) := rfl
  have h4 : subsets_with_2_points = (nat.choose n 2) := rfl
  rw [h, h1, h2, h3, h4]
  norm_num
  sorry

end num_distinct_convex_polygons_l761_761056


namespace arithmetic_general_formula_sum_of_first_n_terms_l761_761534

-- Given conditions
variables {a : ℕ → ℕ} {b : ℕ → ℚ} 

axiom arithmetic_sequence (d : ℕ) :
  (a 1 + a 3 = 8) ∧ (a 2 + a 4 = 12)

-- Questions to prove 
theorem arithmetic_general_formula (d : ℕ) (h : arithmetic_sequence d) :
  ∀ n, a n = 2 * n :=
sorry

theorem sum_of_first_n_terms (d : ℕ) (h : arithmetic_sequence d) :
  ∀ n, let b n := (a n : ℚ) / 2^n in (finset.range n).sum b = 4 - (n + 2) / (2^(n-1)) :=
sorry

end arithmetic_general_formula_sum_of_first_n_terms_l761_761534


namespace find_g_of_conditions_l761_761292

theorem find_g_of_conditions (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, x * g y = 2 * y * g x)
  (g_10 : g 10 = 15) : g 2 = 6 :=
sorry

end find_g_of_conditions_l761_761292


namespace number_of_ways_to_choose_one_book_l761_761683

-- Defining the conditions
def num_chinese_books : ℕ := 5
def num_math_books : ℕ := 4

-- Statement of the theorem
theorem number_of_ways_to_choose_one_book : num_chinese_books + num_math_books = 9 :=
by
  -- Skipping the proof as instructed
  sorry

end number_of_ways_to_choose_one_book_l761_761683


namespace units_digit_of_1583_pow_1246_l761_761908

theorem units_digit_of_1583_pow_1246 : 
  (1583^1246) % 10 = 9 := 
sorry

end units_digit_of_1583_pow_1246_l761_761908


namespace count_lattice_points_in_segment_l761_761473

noncomputable def gcd (a b : ℤ) : ℤ := if b = 0 then a else gcd b (a % b)

theorem count_lattice_points_in_segment :
  let x1 := 5
  let y1 := 5
  let x2 := 65
  let y2 := 290
  let dx := x2 - x1
  let dy := y2 - y1
  let n := gcd dx dy
  n + 1 = 16 :=
by
  -- The definitions and calculations happen here.
  sorry

end count_lattice_points_in_segment_l761_761473


namespace hyperbola_asymptote_correct_l761_761898

noncomputable def hyperbola_asymptote (m : ℝ) : Prop :=
  ∃ (P : ℝ × ℝ) (F : ℝ × ℝ), P.1^2 - (P.2^2 / m) = 1 ∧ P.2^2 = 8 * P.1 ∧
    dist P F = 5 ∧ m = 3 → (√3 * P.1 = P.2 ∨ √3 * P.1 = -P.2)

theorem hyperbola_asymptote_correct : ∀ (P : ℝ × ℝ) (F : ℝ × ℝ), 
  hyperbola_asymptote 3 :=
by
  sorry

end hyperbola_asymptote_correct_l761_761898


namespace mountain_descent_speed_increase_l761_761787

/--
Chrystal’s vehicle speed is 30 miles per hour. Ascending the mountain decreases its speed by fifty percent, 
and descending the mountain increases its speed by a certain percentage. 
If the distance going to the top of the mountain is 60 miles and the distance going down to the foot of the mountain is 72 miles, 
and it takes her 6 hours to pass the whole mountain, then the percentage increase in speed while descending the mountain is 20%. 
-/
theorem mountain_descent_speed_increase :
  ∀ (v₀ d_up d_down t_total: ℝ) (h: d_up / (v₀ * 0.5) + d_down / v₀ = t_total),
  (v₀ = 30) → (d_up = 60) → (d_down = 72) → (t_total = 6) →
  let v_down := d_down / (t_total - d_up / (v₀ * 0.5))
  in (v_down - v₀) / v₀ * 100 = 20 :=
begin
  intros v₀ d_up d_down t_total h hv₀ hd_up hd_down ht_total,
  let v_down := d_down / (t_total - d_up / (v₀ * 0.5)),
  have : v_down = 36,
   sorry,
  have : (v_down - v₀) / v₀ * 100 = 20,
   sorry,
  exact this,
end

end mountain_descent_speed_increase_l761_761787


namespace fruit_display_total_l761_761686

-- Define the number of bananas
def bananas : ℕ := 5

-- Define the number of oranges, twice the number of bananas
def oranges : ℕ := 2 * bananas

-- Define the number of apples, twice the number of oranges
def apples : ℕ := 2 * oranges

-- Define the total number of fruits
def total_fruits : ℕ := bananas + oranges + apples

-- Theorem statement: Prove that the total number of fruits is 35
theorem fruit_display_total : total_fruits = 35 := 
by
  -- proof will be skipped, but all necessary conditions and definitions are included to ensure the statement is valid
  sorry

end fruit_display_total_l761_761686


namespace martin_speed_first_half_l761_761233

variable (v : ℝ) -- speed during the first half of the trip

theorem martin_speed_first_half
    (trip_duration : ℝ := 8)              -- The trip lasted 8 hours
    (speed_second_half : ℝ := 85)          -- Speed during the second half of the trip
    (total_distance : ℝ := 620)            -- Total distance traveled
    (time_each_half : ℝ := trip_duration / 2) -- Each half of the trip took half of the total time
    (distance_second_half : ℝ := speed_second_half * time_each_half)
    (distance_first_half : ℝ := total_distance - distance_second_half) :
    v = distance_first_half / time_each_half :=
by
  sorry

end martin_speed_first_half_l761_761233


namespace matrix_multiplication_correct_l761_761464

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 1], ![4, -2]]
def matrix2 : Matrix (Fin 2) (Fin 2) ℤ := ![![7, -3], ![2, 2]]
def result_matrix : Matrix (Fin 2) (Fin 2) ℤ := ![![23, -7], ![24, -16]]

theorem matrix_multiplication_correct :
  matrix1.mul matrix2 = result_matrix :=
by
  sorry

end matrix_multiplication_correct_l761_761464


namespace prime_factor_of_sum_of_consecutive_integers_l761_761907

theorem prime_factor_of_sum_of_consecutive_integers (n : ℤ) : ∃ p : ℕ, Prime p ∧ p = 2 ∧ (p ∣ ((n - 1) + n + (n + 1) + (n + 2))) :=
by
  sorry

end prime_factor_of_sum_of_consecutive_integers_l761_761907


namespace defective_percentage_m1_l761_761170

theorem defective_percentage_m1 :
  ∀ (total_production : ℕ)
    (m1_percentage m2_percentage m3_percentage : ℕ)
    (m2_defective_percentage m3_defective_percentage : ℕ)
    (total_non_defective_percentage : ℕ),
    m1_percentage = 25 →
    m2_percentage = 35 →
    m3_percentage = 40 →
    m2_defective_percentage = 4 →
    m3_defective_percentage = 5 →
    total_non_defective_percentage = 961 / 10 →
    total_production = 100 →
    let m1_production := total_production * m1_percentage / 100,
        m2_production := total_production * m2_percentage / 100,
        m3_production := total_production * m3_percentage / 100,
        total_defective_percentage := 100 - total_non_defective_percentage,
        m2_defective_units := m2_production * m2_defective_percentage / 100,
        m3_defective_units := m3_production * m3_defective_percentage / 100,
        total_defective_units := total_production * total_defective_percentage / 100,
        m1_defective_units := total_defective_units - (m2_defective_units + m3_defective_units)
    in 
    m1_defective_units * 100 / m1_production = 2 :=
by
  intros
  sorry

end defective_percentage_m1_l761_761170


namespace area_of_triangle_eccentricity_range_θ_eccentricity_product_range_l761_761083

variables {c a1 b1 a2 b2 e1 e2 : ℝ}
variables {M : ℝ × ℝ}
variables {θ : ℝ}

-- Conditions
def is_focus1 (F1 : ℝ × ℝ) : Prop := F1 = (-c, 0)
def is_focus2 (F2 : ℝ × ℝ) : Prop := F2 = (c, 0)
def is_ellipse (C1 : set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ C1 ↔ (x^2) / (a1^2) + (y^2) / (b1^2) = 1 ∧ a1 > b1 ∧ b1 > 0
def is_hyperbola (C2 : set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ C2 ↔ (x^2) / (a2^2) - (y^2) / (b2^2) = 1 ∧ a2 > 0 ∧ b2 > 0
def is_intersection (M : ℝ × ℝ) (C1 C2 : set (ℝ × ℝ)) : Prop := M ∈ C1 ∧ M ∈ C2
def eccentricity1 (C1 : set (ℝ × ℝ)) (e1 : ℝ) : Prop := e1 = sqrt (1 - (b1^2) / (a1^2))
def eccentricity2 (C2 : set (ℝ × ℝ)) (e2 : ℝ) : Prop := e2 = sqrt (1 + (b2^2) / (a2^2))

-- Proof Statements
theorem area_of_triangle (F1 F2 : ℝ × ℝ) (H1 : is_focus1 F1) (H2 : is_focus2 F2)
  (H_ellipse : is_ellipse (λ p => (p.1^2) / (a1^2) + (p.2^2) / (b1^2) = 1)) 
  (H_hyperbola : is_hyperbola (λ p => (p.1^2) / (a2^2) - (p.2^2) / (b2^2) = 1))
  (H_intersection : is_intersection M (λ p => (p.1^2) / (a1^2) + (p.2^2) / (b1^2) = 1) (λ p => (p.1^2) / (a2^2) - (p.2^2) = 1))
  : (1/2) * sqrt ((2 * b1^2) * sin θ / (1 + cos θ)) * sqrt ((2 * b2^2) * sin θ / (1 - cos θ)) = b1 * b2 :=
sorry

theorem eccentricity_range_θ (θ : ℝ) (H_ellipse : is_ellipse (λ p => (p.1^2) / (a1^2) + (p.2^2) / (b1^2) = 1)) :
  e1 > sin(θ / 2) ∧ e1 < 1 :=
sorry

theorem eccentricity_product_range (θ : ℝ) (H_ellipse : is_ellipse (λ p => (p.1^2) / (a1^2) + (p.2^2) / (b1^2) = 1))
  (H_hyperbola : is_hyperbola (λ p => (p.1^2) / (a2^2) - (p.2^2) / (b2^2) = 1))
  (H_angle : θ = 2 * Real.pi / 3) :
  e1 * e2 ≥ sqrt 3 / 2 ∧ e1^2 + e2^2 > 2 :=
sorry

end area_of_triangle_eccentricity_range_θ_eccentricity_product_range_l761_761083


namespace number_between_sasha_and_yulia_l761_761073

theorem number_between_sasha_and_yulia : 
  ∀ (Rita Yulia Sasha Natasha Alina : ℕ),
  Rita = 1 ∧ Yulia = 2 ∧ Sasha = 3 ∧ Natasha = 4 ∧ Alina = 5 →
  (Yulia < Sasha) →
  Sasha - Yulia = 1 →
  0 = 0 :=
begin
  intros Rita Yulia Sasha Natasha Alina h_pos h_order h_subtract,
  exact rfl
end 

end number_between_sasha_and_yulia_l761_761073


namespace min_distance_parabola_line_l761_761618

def parabola (x : ℝ) : ℝ := x^2 - 6 * x + 12
def line (x : ℝ) : ℝ := 2 * x - 5

theorem min_distance_parabola_line :
  ∃ (a : ℝ),
  a >= -real.sqrt (31/2) ∧ a <= real.sqrt (31/2) ∧
  (∀ (b : ℝ), ∃ (a : ℝ), abs ((line b) - (parabola a)) ≥ 1 / real.sqrt 5) :=
sorry

end min_distance_parabola_line_l761_761618


namespace find_Q_l761_761488

theorem find_Q (Q : ℕ) (sum_100th_group : 100 * Q = (100/2) * (2 * (∑ k in finset.range 101, k) + (2 * (∑ k in finset.range 100, k) - 1))) :
  Q = 10001 :=
by
  sorry

end find_Q_l761_761488


namespace binom_150_1_l761_761015

-- Definition of the binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_150_1 : binom 150 1 = 150 := by
  -- The proof is skipped and marked as sorry
  sorry

end binom_150_1_l761_761015


namespace angle_APB_l761_761948

theorem angle_APB (PA_tangent_SAR : Prop) (PB_tangent_RBT : Prop) (SRT_straight : Prop) 
  (arc_AS : Real := 70) (arc_BT : Real := 45) : 
  ∠ APB = 115 := by
  sorry

end angle_APB_l761_761948


namespace domain_range_g_l761_761228

variable (f : ℝ → ℝ) 

noncomputable def g (x : ℝ) := 2 - f (x + 1)

theorem domain_range_g :
  (∀ x, 0 ≤ x → x ≤ 3 → 0 ≤ f x → f x ≤ 1) →
  (∀ x, -1 ≤ x → x ≤ 2) ∧ (∀ y, 1 ≤ y → y ≤ 2) :=
sorry

end domain_range_g_l761_761228


namespace coby_travel_time_l761_761795

def travel_time (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

theorem coby_travel_time :
  let wash_to_idaho_distance := 640
  let idaho_to_nevada_distance := 550
  let wash_to_idaho_speed := 80
  let idaho_to_nevada_speed := 50
  travel_time wash_to_idaho_distance wash_to_idaho_speed + travel_time idaho_to_nevada_distance idaho_to_nevada_speed = 19 := by
  sorry

end coby_travel_time_l761_761795


namespace bridge_length_correct_l761_761433

-- Definitions based on conditions
def train_length : ℝ := 200 -- meters
def train_speed : ℝ := 60 * 1000 / 3600 -- converting 60 km/h to m/s
def deceleration : ℝ := -2 -- m/s²
def headwind_speed : ℝ := 10 * 1000 / 3600 -- converting 10 km/h to m/s
def final_velocity : ℝ := 0 -- m/s, since the train stops

-- Effective initial velocity considering headwind
def effective_initial_velocity : ℝ := train_speed - headwind_speed

-- Using kinematic equation to calculate distance traveled while decelerating
def stopping_distance : ℝ := (final_velocity ^ 2 - effective_initial_velocity ^ 2) / (2 * deceleration)

-- The length of the bridge is the stopping distance plus the length of the train
def bridge_length : ℝ := stopping_distance + train_length

-- Main theorem to prove the length of the bridge is as required
theorem bridge_length_correct : bridge_length ≈ 248.30 := by sorry

end bridge_length_correct_l761_761433


namespace bin101_to_decimal_l761_761036

-- Define the binary representation of 101 (base 2)
def bin101 : ℕ := 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- State the theorem that asserts the decimal value of 101 (base 2) is 5
theorem bin101_to_decimal : bin101 = 5 := by
  sorry

end bin101_to_decimal_l761_761036


namespace variance_transformed_sample_l761_761552

-- Given conditions 
variable (n : ℕ) (x : Fin n → ℝ)

-- Assume the sample variance of x₁, x₂, ..., xₙ is 2
axiom sample_variance_x (h1 : ∑ i, (x i - (∑ j, x j) / n)^2 / n = 2)

-- The mathematical proof problem in Lean 4
theorem variance_transformed_sample :
  (∑ i, ((3 * x i + 2) - (∑ j, (3 * x j + 2)) / n)^2 / n = 18 := sorry

end variance_transformed_sample_l761_761552


namespace star_commutative_star_not_associative_l761_761838

variable (x y z k : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)

def star (x y : ℝ) : ℝ := (k * x * y) / (x + y)

theorem star_commutative : star k x y = star k y x :=
by
  sorry

theorem star_not_associative : star k (star k x y) z ≠ star k x (star k y z) :=
by
  sorry

end star_commutative_star_not_associative_l761_761838


namespace box_volume_l761_761754

theorem box_volume
  (l w h : ℝ)
  (h1 : l * w = 30)
  (h2 : w * h = 20)
  (h3 : l * h = 12)
  (h4 : l = h + 1) :
  l * w * h = 120 := 
sorry

end box_volume_l761_761754


namespace processing_box_function_l761_761293

theorem processing_box_function (A B C D : Prop) (hA : A = "To indicate the start of an algorithm")
                                                    (hB : B = "To indicate an algorithm input")
                                                    (hC : C = "Assignment and calculation")
                                                    (hD : D = "To judge whether a condition is met") :
  C = "Assignment and calculation" :=
by
  -- Proof steps here
  sorry

end processing_box_function_l761_761293


namespace length_DE_l761_761280

open Classical

noncomputable def triangle_base_length (ABC_base : ℝ) : ℝ :=
15

noncomputable def is_parallel (DE BC : ℝ) : Prop :=
DE = BC

noncomputable def area_ratio (triangle_small triangle_large : ℝ) : ℝ :=
0.25

theorem length_DE 
  (ABC_base : ℝ)
  (DE : ℝ)
  (BC : ℝ)
  (triangle_small : ℝ)
  (triangle_large : ℝ)
  (h_base : triangle_base_length ABC_base = 15)
  (h_parallel : is_parallel DE BC)
  (h_area : area_ratio triangle_small triangle_large = 0.25)
  (h_similar : true):
  DE = 7.5 :=
by
  sorry

end length_DE_l761_761280


namespace solution_l761_761880

noncomputable section
open BigOperators

variable {a b c : ℕ → ℕ}

def geometric_sequence (a : ℕ → ℕ) (q : ℕ) :=
∀ n, a (n + 1) = q * a n

def increasing_sequence (a : ℕ → ℕ) :=
∀ n, a n < a (n + 1)

def sum_first_n_terms (seq_sum : ℕ → ℕ) (b : ℕ → ℕ) :=
∀ n, seq_sum n = ∑ i in Finset.range n, b i

def condition_1 (a : ℕ → ℕ) : Prop :=
increasing_sequence a ∧ a 5 ^ 2 = a 10

def condition_2 (a : ℕ → ℕ) : Prop :=
∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)

def condition_3 (b : ℕ → ℕ) : Prop :=
b 1 = 1 ∧ ∀ n, b n ≠ 0

def condition_4 (sn : ℕ → ℕ) (b : ℕ → ℕ) : Prop :=
sum_first_n_terms sn b ∧ ∀ n, b n * b (n + 1) = 4 * sn n - 1

theorem solution (a b sn : ℕ → ℕ) 
  (cond1 : condition_1 a)
  (cond2 : condition_2 a)
  (cond3 : condition_3 b)
  (cond4 : condition_4 sn b)
  : (∀ n, a n = 2 ^ n) ∧ (∀ n, b n = 2 * n - 1) ∧ (∃ tn : ℕ → ℕ, tn = λ n, (2 * n - 3) * 2 ^ (n + 1) + 6) :=
by 
  sorry

end solution_l761_761880


namespace sum_quotient_dividend_divisor_l761_761153

theorem sum_quotient_dividend_divisor (n : ℕ) (d : ℕ) (h : n = 45) (h1 : d = 3) : 
  (n / d) + n + d = 63 :=
by
  sorry

end sum_quotient_dividend_divisor_l761_761153


namespace dataset_mean_and_mode_l761_761855

noncomputable def dataset : List ℕ := [60, 30, 50, 40, 50, 70]

def mean (data : List ℕ) : ℚ :=
  (data.sum : ℚ) / data.length

def mode (data : List ℕ) : ℕ :=
  data.maxBy (λ n, data.count n) (by simp)

theorem dataset_mean_and_mode : mean dataset = 50 ∧ mode dataset = 50 := 
  by 
    sorry

end dataset_mean_and_mode_l761_761855


namespace triangle_side_sum_l761_761696

theorem triangle_side_sum (a : ℝ) (angle_B : ℝ) (angle_A : ℝ) (angle_C : ℝ) :
  angle_A = 50 ∧ angle_B = 40 ∧ angle_C = 90 ∧ a = 8 →
  sin (40 * Real.pi / 180) * 8 + cos (40 * Real.pi / 180) * 8 = 11.3 :=
by
  intro h
  cases h with h1 h2,
  cases h2 with h3 h4,
  sorry

end triangle_side_sum_l761_761696


namespace complex_pure_imaginary_l761_761159

theorem complex_pure_imaginary (a : ℝ) 
  (h : (1 + a * complex.I) * (3 - complex.I) = complex.I * (3 * a - 1)) : 
  a = -3 :=
  sorry

end complex_pure_imaginary_l761_761159


namespace determine_a_l761_761816

-- Define the collinearity condition
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let slope (p1 p2 : ℝ × ℝ) := (p2.2 - p1.2) / (p2.1 - p1.1)
  slope p1 p2 = slope p2 p3

-- Points (3,1), (6,a), and (8,10)
def p1 := (3, 1) : ℝ × ℝ
def p3 := (8, 10) : ℝ × ℝ

-- Main theorem
theorem determine_a (a : ℝ) : 
  collinear p1 (6, a) p3 ↔ a = 6.4 :=
sorry

end determine_a_l761_761816


namespace regions_divided_by_7_tangents_l761_761192

-- Define the recursive function R for the number of regions divided by n tangents
def R : ℕ → ℕ
| 0       => 1
| (n + 1) => R n + (n + 1)

-- The theorem stating the specific case of the problem
theorem regions_divided_by_7_tangents : R 7 = 29 := by
  sorry

end regions_divided_by_7_tangents_l761_761192


namespace regular_hexagon_area_l761_761840

theorem regular_hexagon_area (a : ℝ) (r : ℝ) 
  (ha : r = (3 / 2) * a) :
  let s := r in
  let area := (3 * Real.sqrt 3 / 2) * s^2 in
  area = (27 * a^2 * Real.sqrt 3) / 8 := by
  -- Proof will be provided here
  sorry

end regular_hexagon_area_l761_761840


namespace area_ratio_four_small_to_one_large_l761_761171

theorem area_ratio_four_small_to_one_large (s : ℝ) :
  let area_small := (sqrt 3 / 4) * s ^ 2,
      perimeter_small := 3 * s,
      large_side := 3 * s,
      area_large := (sqrt 3 / 4) * (large_side) ^ 2,
      total_area_small := 4 * area_small
  in total_area_small / area_large = 4 / 9 :=
by
  sorry

end area_ratio_four_small_to_one_large_l761_761171


namespace inequality_area_correct_l761_761400

noncomputable def inequality_area : ℝ :=
  let area_half := (1 / 2) * ((-(-2) + 1)) * (2 + (1 / 2)) in
  2 * area_half

theorem inequality_area_correct :
  inequality_area = 15 / 2 :=
by
  sorry

end inequality_area_correct_l761_761400


namespace probability_of_different_tens_digits_l761_761660

open Nat

theorem probability_of_different_tens_digits (S : Finset ℤ) (h : ∀ x ∈ S, 20 ≤ x ∧ x ≤ 89) :
  (∃ T : Finset ℤ, (∀ t ∈ T, 20 ≤ t ∧ t ≤ 89 ∧ ∃ d ∈ Range 10, x / 10 = d ∧ T.card = 6 ∧ T.val.filter (λ x, x / 10 = 7) ≠ []))
    → (6 ≤ S.card ∧ ∀ x y ∈ S, x ≠ y → x / 10 ≠ y / 10)
    → ∑ x in S, 1 / (∑ a in Finset.filter (λ x, x / 10 ≠ 7) S, (1 : ℚ)) = 2000 / 342171 := by sorry

end probability_of_different_tens_digits_l761_761660


namespace f_of_f_of_neg1_l761_761087

-- Define the function f(x) as per the conditions
def f (x : ℝ) : ℝ :=
  if x ≥ 0 then real.log 2 (x^2) + 1 else (1 / 3)^x + 1

-- State the theorem to prove that f(f(-1)) = 5
theorem f_of_f_of_neg1 : f (f (-1)) = 5 :=
by
  -- Proof omitted; includes necessary placeholder for compilation
  sorry

end f_of_f_of_neg1_l761_761087


namespace necessary_but_not_sufficient_for_lt_l761_761503

variable {a b : ℝ}

theorem necessary_but_not_sufficient_for_lt (h : a < b + 1) : a < b := 
sorry

end necessary_but_not_sufficient_for_lt_l761_761503


namespace cost_of_two_pencils_and_one_pen_l761_761286

variables (a b : ℝ)

theorem cost_of_two_pencils_and_one_pen
  (h1 : 3 * a + b = 3.00)
  (h2 : 3 * a + 4 * b = 7.50) :
  2 * a + b = 2.50 :=
sorry

end cost_of_two_pencils_and_one_pen_l761_761286


namespace vasya_numbers_l761_761338

theorem vasya_numbers :
  ∃ x y : ℝ, x + y = x * y ∧ x + y = x / y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l761_761338


namespace sequence_value_l761_761594

theorem sequence_value :
  (∀ n : ℕ, a_n = n^2 - 2n + 3) → a_5 = 18 :=
by
  intros h
  sorry

end sequence_value_l761_761594


namespace product_frac_equality_l761_761005

-- Define the product function for the required interval
def product_frac (n : ℕ) : ℚ := ∏ i in (Finset.range n).map (λ x, x + 1), (i + 5) / i

-- The goal is to prove that the product from 1 to 30 equals 326159
theorem product_frac_equality : product_frac 30 = 326159 := 
by 
  sorry

end product_frac_equality_l761_761005


namespace smallest_positive_multiple_of_17_6_more_than_multiple_of_73_l761_761713

theorem smallest_positive_multiple_of_17_6_more_than_multiple_of_73 :
  ∃ b : ℤ, (17 * b ≡ 6 [MOD 73]) ∧ 17 * b = 663 :=
begin
  sorry
end

end smallest_positive_multiple_of_17_6_more_than_multiple_of_73_l761_761713


namespace classify_triangle_l761_761533

theorem classify_triangle (m : ℕ) (h₁ : m > 1) (h₂ : 3 * m + 3 = 180) :
  (m < 60) ∧ (m + 1 < 90) ∧ (m + 2 < 90) :=
by
  sorry

end classify_triangle_l761_761533


namespace union_of_A_and_B_l761_761627

def A := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 3)}
def B := {y : ℝ | ∃ (x : ℝ), y = Real.exp x}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | x > 0} := by
sorry

end union_of_A_and_B_l761_761627


namespace negation_of_P_l761_761120

open Classical

variable (x : ℝ)

def P (x : ℝ) : Prop :=
  x^2 + 2 > 2 * x

theorem negation_of_P : (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x := by
  sorry

end negation_of_P_l761_761120


namespace binary_to_decimal_l761_761023

theorem binary_to_decimal :
  ∀ n : ℕ, n = 101 →
  ∑ i in Finset.range 3, (n / 10^i % 10) * 2^i = 5 :=
by
  sorry

end binary_to_decimal_l761_761023


namespace roots_are_2_i_neg_i_l761_761990

noncomputable def roots_satisfy_conditions (a b c : ℂ) : Prop :=
  a + b + c = 3 ∧ ab + ac + bc = 3 ∧ abc = -1

theorem roots_are_2_i_neg_i (a b c : ℂ) (h : roots_satisfy_conditions a b c) :
  (a = 2 ∨ a = complex.I ∨ a = -complex.I) ∧
  (b = 2 ∨ b = complex.I ∨ b = -complex.I) ∧
  (c = 2 ∨ c = complex.I ∨ c = -complex.I) :=
sorry

end roots_are_2_i_neg_i_l761_761990


namespace rectangle_triangle_problem_l761_761589

theorem rectangle_triangle_problem (a b : ℝ) (x : ℝ)
  (h1 : 4 * (3/2) * x^2 = 288)
  (h2 : PQ = a - 2 * x)
  (h3 : PR = b - 6 * x)
  (h4 : b = 60)
  (h5 : x = sqrt 48) : 
  PR = 60 - 24 * sqrt 3 :=
by 
  sorry

end rectangle_triangle_problem_l761_761589


namespace geometric_sequence_a9_l761_761091

theorem geometric_sequence_a9 :
  ∃ q : ℚ, ∃ a_1 a_2 a_5 a_8 a_9 : ℚ,
    a_1 = 1 / 2 ∧
    a_2 = a_1 * q ∧
    a_5 = a_1 * q^4 ∧
    a_8 = a_1 * q^7 ∧
    a_2 * a_8 = 2 * a_5 + 3 ∧
    a_9 = a_1 * q^8 ∧
    a_9 = 18 :=
begin
  sorry
end

end geometric_sequence_a9_l761_761091


namespace problem_expression_eq_zero_l761_761468

variable {x y : ℝ}

theorem problem_expression_eq_zero (h : x * y ≠ 0) : 
    ( ( (x^2 - 1) / x ) * ( (y^2 - 1) / y ) ) - 
    ( ( (x^2 - 1) / y ) * ( (y^2 - 1) / x ) ) = 0 :=
by
  sorry

end problem_expression_eq_zero_l761_761468


namespace Matt_received_more_pencils_than_Lauren_l761_761659

-- Definitions based on conditions
def total_pencils := 2 * 12
def pencils_to_Lauren := 6
def pencils_after_Lauren := total_pencils - pencils_to_Lauren
def pencils_left := 9
def pencils_to_Matt := pencils_after_Lauren - pencils_left

-- Formulate the problem statement
theorem Matt_received_more_pencils_than_Lauren (total_pencils := 24) (pencils_to_Lauren := 6) (pencils_after_Lauren := 18) (pencils_left := 9) (correct_answer := 3) :
  pencils_to_Matt - pencils_to_Lauren = correct_answer := 
by 
  sorry

end Matt_received_more_pencils_than_Lauren_l761_761659


namespace volume_of_pond_rect_prism_l761_761583

-- Define the problem as a proposition
theorem volume_of_pond_rect_prism :
  let l := 28
  let w := 10
  let h := 5
  V = l * w * h →
  V = 1400 :=
by
  intros l w h h1
  -- Here, the theorem states the equivalence of the volume given the defined length, width, and height being equal to 1400 cubic meters.
  have : V = 28 * 10 * 5 := by sorry
  exact this

end volume_of_pond_rect_prism_l761_761583


namespace expenditure_representation_l761_761237

def income_represented_pos (income : ℤ) : Prop := income > 0

def expenditure_represented_neg (expenditure : ℤ) : Prop := expenditure < 0

theorem expenditure_representation (income expenditure : ℤ) (h_income: income_represented_pos income) (exp_value: expenditure = 3) : expenditure_represented_neg expenditure := 
sorry

end expenditure_representation_l761_761237


namespace Vasya_numbers_l761_761372

theorem Vasya_numbers : ∃ (x y : ℝ), x + y = xy ∧ xy = x / y ∧ (x, y) = (1 / 2, -1) :=
by
  sorry

end Vasya_numbers_l761_761372


namespace car_Y_win_probability_l761_761940

theorem car_Y_win_probability
  (P : Type -> ℝ)
  (P_X : P unit = 1 / 2)
  (P_Z : P unit = 1 / 3)
  (P_sum : P unit + P unit + P unit = 13 / 12) :
  ∃ P_Y : ℝ, P_Y = 1 / 4 :=
by 
    have hP_Y_is_fraction := P_sum - P_X - P_Z,
    rw hP_Y_is_fraction,
    exact 1 / 4
    sorry

end car_Y_win_probability_l761_761940


namespace ratio_of_new_r_to_original_r_l761_761183

theorem ratio_of_new_r_to_original_r
  (r₁ r₂ : ℝ)
  (a₁ a₂ : ℝ)
  (h₁ : a₁ = (2 * r₁)^3)
  (h₂ : a₂ = (2 * r₂)^3)
  (h : a₂ = 0.125 * a₁) :
  r₂ / r₁ = 1 / 2 :=
by
  sorry

end ratio_of_new_r_to_original_r_l761_761183


namespace collinear_midpoint_O_of_circumscribed_quadrilateral_l761_761972

noncomputable def is_circumscribed_quadrilateral (A B C D : Point)
    (ω : Circle) (O : Point) : Prop :=
  ω.center = O ∧
  ¬(Segment (A, B) ∩ ω).empty ∧
  ¬(Segment (B, C) ∩ ω).empty ∧
  ¬(Segment (C, D) ∩ ω).empty ∧
  ¬(Segment (D, A) ∩ ω).empty ∧
  collinear [A, B, C, D]

def midpoint (P Q : Point) : Point :=
  Point.mk ((P.x + Q.x) / 2) ((P.y + Q.y) / 2)

theorem collinear_midpoint_O_of_circumscribed_quadrilateral
    (A B C D K L X O : Point)
    (ω ω₁ ω₂ : Circle)
    (O_center : ω.center = O)
    (circumscribed : is_circumscribed_quadrilateral A B C D ω O)
    (HX : intersection(Line.mk A B, Line.mk C D) = {X})
    (HK : is_tangent ω₁ (Line.extension A B) (Line.extension C D) (Segment.mk A D) K)
    (HL : is_tangent ω₂ (Line.extension A B) (Line.extension C D) (Segment.mk B C) L)
    (collinear_XKL : collinear [X, K, L]) :
  let M := midpoint A D,
      N := midpoint B C
  in collinear [O, M, N] := 
sorry

end collinear_midpoint_O_of_circumscribed_quadrilateral_l761_761972


namespace combined_degrees_l761_761265

variable (Summer_deg Jolly_deg : ℕ)

def Summer_has_150_degrees := Summer_deg = 150

def Summer_has_5_more_degrees_than_Jolly := Summer_deg = Jolly_deg + 5

theorem combined_degrees (h1 : Summer_has_150_degrees Summer_deg) (h2 : Summer_has_5_more_degrees_than_Jolly Summer_deg Jolly_deg) :
  Summer_deg + Jolly_deg = 295 :=
by
  sorry

end combined_degrees_l761_761265


namespace expression_evaluation_l761_761053

theorem expression_evaluation :
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 := by
  sorry

end expression_evaluation_l761_761053


namespace cannot_determine_log23_l761_761867

-- Definitions using the conditions provided
def log2 : ℝ := 0.3010
def log5 : ℝ := 0.6990

-- Conditions
axiom log2_is_approx : (real.log 2) ≈ log2
axiom log5_is_approx : (real.log 5) ≈ log5

-- Lean statement for the problem
theorem cannot_determine_log23 :
  ¬ (∃ log23 : ℝ, (real.log 23) ≈ log23 ⊓ ∀ (log2 log5 : ℝ), log2 ≈ 0.3010 → log5 ≈ 0.6990 → 
  (real.log 23) ≈ f log2 log5) :=
sorry

end cannot_determine_log23_l761_761867


namespace problem1_problem2_l761_761784

theorem problem1 (x : ℝ) : 2 * (x - 1) ^ 2 = 18 ↔ x = 4 ∨ x = -2 := by
  sorry

theorem problem2 (x : ℝ) : x ^ 2 - 4 * x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 := by
  sorry

end problem1_problem2_l761_761784


namespace find_m_for_arithmetic_progression_roots_l761_761476

theorem find_m_for_arithmetic_progression_roots :
  ∀ m : ℝ,
    (∀ x : ℝ, x^4 - (3 * m + 2) * x^2 + m^2 = 0 →
                 (∃ a b : ℝ, [a, b, -b, -a] = list.sort (≤) [x_i | x_i is roots of polynomial] ∧
                     (a - b = 2 * b) ∧ (a = 3 * b))
              → m = 6 ∨ m = -6 / 19) :=
sorry

end find_m_for_arithmetic_progression_roots_l761_761476


namespace combined_degrees_l761_761260

theorem combined_degrees (summer_degrees jolly_degrees : ℕ) (h1 : summer_degrees = jolly_degrees + 5) (h2 : summer_degrees = 150) : summer_degrees + jolly_degrees = 295 := 
by
  sorry

end combined_degrees_l761_761260


namespace linear_regression_fixed_point_l761_761298

variable (b a x y : ℝ)
variable (x_bar y_bar : ℝ)

-- Define the linear regression equation
def linear_regression_eqn (x y : ℝ) : Prop := y = b * x + a

-- Assume the fixed point for linear regression equation
axiom fixed_point_condition : linear_regression_eqn x_bar y_bar

-- Statement to prove that the linear regression equation passes through the fixed point (x_bar, y_bar)
theorem linear_regression_fixed_point : linear_regression_eqn x_bar y_bar :=
by 
  -- Proof goes here, sorry placeholder indicates proof is not provided
  sorry

end linear_regression_fixed_point_l761_761298


namespace smallest_lambda_l761_761985

-- Definitions of the sets and the conditions
def Q (A : Finset ℝ) : Finset ℝ := 
  {r | ∃ a b c d ∈ A, c ≠ d ∧ r = (a - b) / (c - d)}

noncomputable def lambda := 1 / 2

-- The final statement to prove
theorem smallest_lambda (A : Finset ℝ) (hA : 2 ≤ A.card) : 
  Q(A).card ≤ lambda * A.card ^ 4 :=
begin
  sorry
end

end smallest_lambda_l761_761985


namespace total_travel_time_l761_761792

-- Define the necessary distances and speeds
def distance_Washington_to_Idaho : ℝ := 640
def speed_Washington_to_Idaho : ℝ := 80
def distance_Idaho_to_Nevada : ℝ := 550
def speed_Idaho_to_Nevada : ℝ := 50

-- Definitions for time calculations
def time_Washington_to_Idaho : ℝ := distance_Washington_to_Idaho / speed_Washington_to_Idaho
def time_Idaho_to_Nevada : ℝ := distance_Idaho_to_Nevada / speed_Idaho_to_Nevada

-- Problem statement to prove
theorem total_travel_time : time_Washington_to_Idaho + time_Idaho_to_Nevada = 19 := 
by
  sorry

end total_travel_time_l761_761792


namespace cone_lateral_surface_area_eq_l761_761741

variables {R : ℝ} (h_positive : R > 0)

noncomputable def volume_hemisphere (R : ℝ) : ℝ :=
  (2 / 3) * π * R^3

noncomputable def volume_cone (R h : ℝ) : ℝ :=
  (1 / 3) * π * R^2 * h

noncomputable def slant_height (R h : ℝ) : ℝ :=
  Real.sqrt (R^2 + h^2)

noncomputable def lateral_surface_area (R l : ℝ) : ℝ :=
  π * R * l

theorem cone_lateral_surface_area_eq (V_eq : volume_hemisphere R = volume_cone R (2 * R)) :
  lateral_surface_area R (slant_height R (2 * R)) = π * R^2 * Real.sqrt 5 :=
by
  sorry

end cone_lateral_surface_area_eq_l761_761741


namespace fraction_increase_by_three_l761_761933

variables (a b : ℝ)

theorem fraction_increase_by_three : 
  3 * (2 * a * b / (3 * a - 4 * b)) = 2 * (3 * a * 3 * b) / (3 * (3 * a) - 4 * (3 * b)) :=
by
  sorry

end fraction_increase_by_three_l761_761933


namespace scientific_notation_correct_l761_761961

-- The given number
def given_number : ℕ := 9000000000

-- The correct answer in scientific notation
def correct_sci_not : ℕ := 9 * (10 ^ 9)

-- The theorem to prove
theorem scientific_notation_correct :
  given_number = correct_sci_not :=
by
  sorry

end scientific_notation_correct_l761_761961


namespace find_b_for_tangent_line_l761_761537

theorem find_b_for_tangent_line (m b : ℝ) :
  ∀ (x y : ℝ),
  (x^2 - 2*x + y^2 - 2*m*y + 2*m - 1 = 0) →
  m = 1 →
  (y = x + b) →
  (abs b = sqrt 2) :=
by sorry

end find_b_for_tangent_line_l761_761537


namespace midpoint_polar_coord_correct_l761_761939

noncomputable def midpoint_polar_coordinates (A B : ℝ × ℝ) : ℝ × ℝ :=
let x_A := A.1 * Real.cos(A.2),
    y_A := A.1 * Real.sin(A.2),
    x_B := B.1 * Real.cos(B.2),
    y_B := B.1 * Real.sin(B.2),
    x_M := (x_A + x_B) / 2,
    y_M := (y_A + y_B) / 2,
    r_M := Real.sqrt(x_M ^ 2 + y_M ^ 2),
    θ_M := Real.arctan2 y_M x_M
in (r_M, θ_M)

theorem midpoint_polar_coord_correct :
  midpoint_polar_coordinates (10, Real.pi / 4) (10, 3 * Real.pi / 4) = (5 * Real.sqrt 2, Real.pi / 2) := 
by
  sorry

end midpoint_polar_coord_correct_l761_761939


namespace geometric_sequence_sum_l761_761090

theorem geometric_sequence_sum (q : ℝ) (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h2 : a 3 + a 5 = 6) :
  a 5 + a 7 + a 9 = 28 :=
  sorry

end geometric_sequence_sum_l761_761090


namespace modulus_z2_is_five_over_four_l761_761874

noncomputable def z1 : ℂ := 3 + 4 * Complex.i
noncomputable def z2 (t : ℝ) : ℂ := t + Complex.i

theorem modulus_z2_is_five_over_four (t : ℝ)
  (h1 : z1 * Complex.conj (z2 t) ∈ ℝ) : Complex.abs (z2 (3 / 4)) = 5 / 4 := by
  sorry

end modulus_z2_is_five_over_four_l761_761874


namespace slices_per_friend_l761_761774

theorem slices_per_friend (total_slices friends : ℕ) (h1 : total_slices = 16) (h2 : friends = 4) : (total_slices / friends) = 4 :=
by
  sorry

end slices_per_friend_l761_761774


namespace problem_1_problem_2_problem_3_problem_4_l761_761785

-- Problem 1
theorem problem_1 : 4.7 + (-2.5) - (-5.3) - 7.5 = 0 := by
  sorry

-- Problem 2
theorem problem_2 : 18 + 48 / (-2)^2 - (-4)^2 * 5 = -50 := by
  sorry

-- Problem 3
theorem problem_3 : -1^4 + (-2)^2 / 4 * (5 - (-3)^2) = -5 := by
  sorry

-- Problem 4
theorem problem_4 : (-19 + 15 / 16) * 8 = -159 + 1 / 2 := by
  sorry

end problem_1_problem_2_problem_3_problem_4_l761_761785


namespace time_to_fill_tub_l761_761250

def tubVolume : ℕ := 120
def leakyRate : ℕ := 1
def flowRate : ℕ := 12
def fillCycleTime : ℕ := 2
def netGainPerCycle : ℕ := (flowRate - leakyRate) - leakyRate

theorem time_to_fill_tub : 
    ∃ (time_in_minutes : ℕ), (time_in_minutes = 24) ∧ (tubVolume = 12 * netGainPerCycle * fillCycleTime) :=
begin
  sorry
end

end time_to_fill_tub_l761_761250


namespace find_first_term_l761_761520

variable {a : ℕ → ℕ}

-- Given conditions
axiom arithmetic_seq : ∀ n : ℕ, a (n + 1) + a n = 4 * n

-- Question to prove
theorem find_first_term : a 0 = 1 :=
sorry

end find_first_term_l761_761520


namespace f_is_even_if_g_is_even_and_f_def_l761_761977

variables (g : ℝ → ℝ)

def is_even_function (h : ℝ → ℝ) : Prop :=
∀ x : ℝ, h(-x) = h(x)

def is_even (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f(-x) = f(x)

theorem f_is_even_if_g_is_even_and_f_def (h_even : is_even_function g) :
  is_even (λ x : ℝ, |g(x^2)|) :=
sorry

end f_is_even_if_g_is_even_and_f_def_l761_761977


namespace positive_integers_sum_divide_7n_count_l761_761496

theorem positive_integers_sum_divide_7n_count : 
  ∃ n : ℕ, ∀ i ∈ [1, 6, 13], 
    (1 + 2 + ... + i) ∣ (7 * i) :=
by sorry

end positive_integers_sum_divide_7n_count_l761_761496


namespace sum_function_values_l761_761924

variable (f : ℕ → ℕ)
variable (hf : ∀ (a b : ℕ), f (a + b) = f a * f b)
variable (hf1 : f 1 = 2)

theorem sum_function_values :
  (∑ k in Finset.range 1007, f (2 * k + 2) / f (2 * k + 1)) = 2014 := by
  sorry

end sum_function_values_l761_761924


namespace compute_expression_l761_761216

noncomputable def given_cubic (x : ℝ) : Prop :=
  x ^ 3 - 7 * x ^ 2 + 12 * x = 18

theorem compute_expression (a b c : ℝ) (ha : given_cubic a) (hb : given_cubic b) (hc : given_cubic c) :
  (a + b + c = 7) → 
  (a * b + b * c + c * a = 12) → 
  (a * b * c = 18) → 
  (a * b / c + b * c / a + c * a / b = -6) :=
by 
  sorry

end compute_expression_l761_761216


namespace team_total_score_l761_761472

theorem team_total_score (Connor_score Amy_score Jason_score : ℕ)
  (h1 : Connor_score = 2)
  (h2 : Amy_score = Connor_score + 4)
  (h3 : Jason_score = 2 * Amy_score) :
  Connor_score + Amy_score + Jason_score = 20 :=
by
  sorry

end team_total_score_l761_761472


namespace solve_7_at_8_l761_761917

theorem solve_7_at_8 : (7 * 8) / (7 + 8 + 3) = 28 / 9 := by
  sorry

end solve_7_at_8_l761_761917


namespace defective_probability_l761_761444

variable (total_products defective_products qualified_products : ℕ)
variable (first_draw_defective second_draw_defective : Prop)

-- Definitions of the problem
def total_prods := 10
def def_prods := 4
def qual_prods := 6
def p_A := def_prods / total_prods
def p_AB := (def_prods / total_prods) * ((def_prods - 1) / (total_prods - 1))
def p_B_given_A := p_AB / p_A

-- Theorem: The probability of drawing a defective product on the second draw given the first was defective is 1/3.
theorem defective_probability 
  (hp1 : total_products = total_prods)
  (hp2 : defective_products = def_prods)
  (hp3 : qualified_products = qual_prods)
  (pA_eq : p_A = 2 / 5)
  (pAB_eq : p_AB = 2 / 15) : 
  p_B_given_A = 1 / 3 := sorry

end defective_probability_l761_761444


namespace log_seq_ar_eq_l761_761911

open Real

theorem log_seq_ar_eq (x : ℝ) (h : ∀ (x : ℝ), (\lg 2), (\lg (2^x - 1)), (\lg (2^x + 3)) form_ar_squence): x = (log 5) / (log 2) :=
sorry

end log_seq_ar_eq_l761_761911


namespace matrix_det_cos_is_zero_l761_761467

theorem matrix_det_cos_is_zero :
  det ![
    ![\cos 2, \cos 4, \cos 6],
    ![\cos 8, \cos 10, \cos 12],
    ![\cos 14, \cos 16, \cos 18]
  ] = 0 :=
by
  -- The actual proof goes here
  sorry

end matrix_det_cos_is_zero_l761_761467


namespace irreducible_positive_fraction_unique_l761_761057

theorem irreducible_positive_fraction_unique
  (a b : ℕ) (h_pos : a > 0 ∧ b > 0) (h_gcd : Nat.gcd a b = 1)
  (h_eq : (a + 12) * b = 3 * a * (b + 12)) :
  a = 2 ∧ b = 9 :=
by
  sorry

end irreducible_positive_fraction_unique_l761_761057


namespace binom_150_1_eq_150_l761_761013

/-- Definition of factorial -/
def fact : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * fact n

/-- Definition of binomial coefficient -/
def binom (n k : ℕ) : ℕ :=
fact n / (fact k * fact (n - k))

/-- Theorem stating the specific binomial coefficient calculation -/
theorem binom_150_1_eq_150 : binom 150 1 = 150 := by
  sorry

end binom_150_1_eq_150_l761_761013


namespace find_value_l761_761557

theorem find_value (x : ℝ) (h : 2^(2*x) = 16) : 2^(-x) + (Real.log 8 / Real.log 2) = 13 / 4 := by
  sorry

end find_value_l761_761557


namespace angle_between_BE_ABC_l761_761852

noncomputable def range_of_angle (P A B C D E : Point) (α : Real) : Prop :=
  let length_of_tetrahedron := 1
  let D := midpoint P C
  let E ∈ segment AD
  ∃ angle_formed_by_BE_ABC : ℝ, 
    α = angle_formed_by_BE_ABC ∧
    (0 ≤ α ∧ α ≤ Real.arctan (Real.sqrt 14 / 7))

theorem angle_between_BE_ABC 
  (P A B C D E : Point) (α : Real) 
  (tetrahedron: regular_tetrahedron P A B C)
  (D_def: midpoint P C D)
  (E_def: ∃t, 0 ≤ t ∧ t ≤ 1 ∧ E = P + t • (D - P)) : 
  range_of_angle P A B C D E α :=
begin
  sorry
end

end angle_between_BE_ABC_l761_761852


namespace mike_books_l761_761236

theorem mike_books : 51 - 45 = 6 := 
by 
  rfl

end mike_books_l761_761236


namespace alphabet_value_l761_761296

def letter_value (c : Char) : Int :=
  let n := c.to_nat - 'a'.to_nat + 1
  match n % 10 with
  | 1 => 1
  | 2 => 2
  | 3 => 0
  | 4 => -1
  | 5 => -2
  | 6 => -1
  | 7 => 0
  | 8 => 1
  | 9 => 2
  | _ => 1  -- covers the case of 0 or multiples of 10

def word_value (word : String) : Int :=
  word.foldl (λ acc c => acc + letter_value c) 0

theorem alphabet_value : word_value "alphabet" = 4 := by
  sorry

end alphabet_value_l761_761296


namespace parallelogram_sum_l761_761048

structure Point where
  x : ℝ
  y : ℝ

structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

theorem parallelogram_sum (p : Parallelogram) 
  (h1 : p.v1 = ⟨1, 3⟩) 
  (h2 : p.v2 = ⟨6, 8⟩) 
  (h3 : p.v3 = ⟨13, 8⟩) 
  (h4 : p.v4 = ⟨8, 3⟩) :
  let side1 := distance p.v1 p.v2
  let side2 := p.v3.x - p.v2.x
  let perimeter := 2 * (side1 + side2)
  let height  := p.v2.y - p.v4.y
  let area := side2 * height
  perimeter + area = 10 * Real.sqrt 2 + 49 :=
by
  sorry

end parallelogram_sum_l761_761048


namespace probability_at_least_half_girls_l761_761610

theorem probability_at_least_half_girls (n : ℕ) (hn : n = 6) :
  (probability (λ (s : vector bool n), s.foldr (λ b acc, if b then acc + 1 else acc) 0 ≥ n/2))
  = 21 / 32 := by
  sorry

end probability_at_least_half_girls_l761_761610


namespace product_frac_equality_l761_761006

-- Define the product function for the required interval
def product_frac (n : ℕ) : ℚ := ∏ i in (Finset.range n).map (λ x, x + 1), (i + 5) / i

-- The goal is to prove that the product from 1 to 30 equals 326159
theorem product_frac_equality : product_frac 30 = 326159 := 
by 
  sorry

end product_frac_equality_l761_761006


namespace max_ratio_of_quadrilateral_l761_761117

variable (a b : ℝ) (S₁ S₂ : ℝ)

noncomputable def hyperbola1 (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
noncomputable def hyperbola2 (x y : ℝ) : Prop := (y^2 / b^2) - (x^2 / a^2) = 1

theorem max_ratio_of_quadrilateral
  (ha : a > 0)
  (hb : b > 0)
  (h_intersect : ∃ x y, hyperbola1 x y ∧ hyperbola2 x y)
  (h_S₁ : S₁ = 2 * a * b)
  (h_S₂ : S₂ = 2 * (a^2 + b^2))
  : (S₁ / S₂) ≤ 1 / 2 := sorry

end max_ratio_of_quadrilateral_l761_761117


namespace sum_of_integers_l761_761675

-- Definitions for the conditions
variables {x y : ℝ}

-- Condition 1: x^2 + y^2 = 250
def condition1 := x^2 + y^2 = 250

-- Condition 2: xy = 120
def condition2 := x * y = 120

-- Condition 3: x^2 - y^2 = 130
def condition3 := x^2 - y^2 = 130

-- The theorem to prove the sum of x and y
theorem sum_of_integers (x y : ℝ) (h1 : condition1) (h2 : condition2) (h3 : condition3) : 
  x + y = 10 * Real.sqrt 4.9 :=
sorry

end sum_of_integers_l761_761675


namespace dance_team_members_l761_761441

theorem dance_team_members (a b c : ℕ)
  (h1 : a + b + c = 100)
  (h2 : b = 2 * a)
  (h3 : c = 2 * a + 10) :
  c = 46 := by
  sorry

end dance_team_members_l761_761441


namespace similar_triangle_angles_l761_761581

theorem similar_triangle_angles (α β γ : ℝ) (h1 : α + β + γ = Real.pi) (h2 : α + β/2 + γ/2 = Real.pi):
  ∃ (k : ℝ), α = k ∧ β = 2 * k ∧ γ = 4 * k ∧ k = Real.pi / 7 := 
sorry

end similar_triangle_angles_l761_761581


namespace sum_seq_l761_761810

noncomputable def seq (n : ℕ) : ℤ :=
  if n = 0 then 2009
  else -2009 * (∑ k in Finset.range n, seq k) / n

theorem sum_seq : (∑ n in Finset.range 2010, 2^n * seq n) = 2009 := by
  sorry

end sum_seq_l761_761810


namespace find_g_of_conditions_l761_761291

theorem find_g_of_conditions (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, x * g y = 2 * y * g x)
  (g_10 : g 10 = 15) : g 2 = 6 :=
sorry

end find_g_of_conditions_l761_761291


namespace product_of_intersection_points_l761_761881

variables {a b c k m : ℝ}

theorem product_of_intersection_points (h : ∀ x, mx + k = ax^2 + bx + c → mx + k = ax^2 + bx + c) :
  let f := λ x : ℝ, ax^2 + (b - m) * x + (c - k) in
  let roots := (h : Poly.root_disjoint (polynomial C (mx + k) - polynomial C (ax^2 + bx + c))) in
  (roots.1 * roots.2) = (c - k) / a :=
sorry

end product_of_intersection_points_l761_761881


namespace distance_reflection_xy_plane_l761_761941

-- Define point P
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Given point P
def P : Point3D := ⟨2, -3, 1⟩

-- Define the reflection of a point about the xy-plane
def reflect_xy_plane (P : Point3D) : Point3D :=
  ⟨P.x, P.y, -P.z⟩

-- Define vector distance function
def distance (P1 P2 : Point3D) : ℝ :=
  (Real.sqrt (((P2.x - P1.x) ^ 2) + ((P2.y - P1.y) ^ 2) + ((P2.z - P1.z) ^ 2)))

-- Here is the proof problem statement
theorem distance_reflection_xy_plane :
  distance P (reflect_xy_plane P) = 2 := by
  sorry

end distance_reflection_xy_plane_l761_761941


namespace mono_increasing_necessary_not_sufficient_problem_statement_l761_761982

-- Define the function
def f (x : ℝ) (m : ℝ) : ℝ := x^3 + 2*x^2 + m*x + 1

-- Define the first condition of p: f(x) is monotonically increasing in (-∞, +∞)
def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x ≤ f y

-- Define the second condition q: m > 4/3
def m_gt_4_over_3 (m : ℝ) : Prop := m > 4/3

-- State the theorem: 
theorem mono_increasing_necessary_not_sufficient (m : ℝ):
  is_monotonically_increasing (f x) → m_gt_4_over_3 m → 
  (is_monotonically_increasing (f x) ↔ m ≥ 4/3) ∧ (¬ is_monotonically_increasing (f x) → m > 4/3) := 
by
  sorry

-- Main theorem tying the conditions to the conclusion
theorem problem_statement (m : ℝ):
  is_monotonically_increasing (f x) → m_gt_4_over_3 m → 
  (is_monotonically_increasing (f x) ↔ m ≥ 4/3) ∧ (¬ is_monotonically_increasing (f x) → m > 4/3) :=
  by sorry

end mono_increasing_necessary_not_sufficient_problem_statement_l761_761982


namespace geometric_sequence_analogy_l761_761883

variables {a_n b_n : ℕ → ℕ} {S T : ℕ → ℕ}

-- Conditions for the arithmetic sequence
def is_arithmetic_sequence_sum (S : ℕ → ℕ) :=
  S 8 - S 4 = 2 * (S 4) ∧ S 12 - S 8 = 2 * (S 8 - S 4)

-- Conditions for the geometric sequence
def is_geometric_sequence_product (T : ℕ → ℕ) :=
  (T 8 / T 4) = (T 4) ∧ (T 12 / T 8) = (T 8 / T 4)

-- Statement of the proof problem
theorem geometric_sequence_analogy
  (h_arithmetic : is_arithmetic_sequence_sum S)
  (h_geometric_nil : is_geometric_sequence_product T) :
  T 4 / T 4 = 1 ∧
  (T 8 / T 4) / (T 8 / T 4) = 1 ∧
  (T 12 / T 8) / (T 12 / T 8) = 1 := 
by
  sorry

end geometric_sequence_analogy_l761_761883


namespace sequence_increasing_l761_761536

open Nat Real

noncomputable def a_n (n : ℕ) (λ : ℝ) : ℝ := 2 * n + λ

noncomputable def S_n (n : ℕ) (λ : ℝ) : ℝ := n^2 + (λ + 1) * n

theorem sequence_increasing (λ : ℝ) : (∀ n >= 7, S_n n λ > S_n (n - 1) λ) ↔ λ > -16 :=
by
  sorry

end sequence_increasing_l761_761536


namespace semicircle_perimeter_l761_761671

theorem semicircle_perimeter (r : ℝ) (π : ℝ) (hr : r = 27.23) (hπ : π ≈ 3.14159) :
  let P := π * r + 2 * r in
  P ≈ 140.06 :=
by sorry

end semicircle_perimeter_l761_761671


namespace optimal_cafeteria_location_l761_761698

theorem optimal_cafeteria_location {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
    (d_AC : dist A C) (d_BC : dist B C) :
    (∀ O : Type, dist O A * 10 + dist O B * 20 + dist O C * 30 ≥ dist C A * 10 + dist C B * 20) :=
begin
  -- Proof goes here
  sorry
end

end optimal_cafeteria_location_l761_761698


namespace num_of_terms_in_arithmetic_sequence_l761_761906

-- Define the arithmetic sequence
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the first term, common difference, and last term of the sequence
def a : ℕ := 15
def d : ℕ := 4
def last_term : ℕ := 99

-- Define the number of terms in the sequence
def n : ℕ := 22

-- State the theorem
theorem num_of_terms_in_arithmetic_sequence : arithmetic_seq a d n = last_term :=
by
  sorry

end num_of_terms_in_arithmetic_sequence_l761_761906


namespace sum_of_divisors_of_2_pow_2007_l761_761070

theorem sum_of_divisors_of_2_pow_2007 : (finset.range 2008).sum (λ k, 2^k) = 2^2008 - 1 :=
by
  sorry

end sum_of_divisors_of_2_pow_2007_l761_761070


namespace chord_equidistant_proof_l761_761547

variables {p q s m : ℝ}
variable (Q : ℝ × ℝ)
variable (intersects_parabola : ℝ × ℝ → Prop)
variable (equidistant : ℝ → ℝ × ℝ → Prop)

def parabola (x y : ℝ) : Prop :=
  y^2 = 2 * p * x

def point_Q_condition (Q : ℝ × ℝ) : Prop :=
  Q = (q, 0) ∧ q < 0

def line_condition (x : ℝ) : Prop :=
  x = s ∧ s > 0

def chord_slope_condition (m : ℝ) : Prop :=
  |m| = sqrt (p / (s - q))

theorem chord_equidistant_proof :
  ∀ (Q : ℝ × ℝ) (p q s : ℝ), parabola Q.1 Q.2 → point_Q_condition Q → line_condition s →
  (∃ ℝ x1 x2, intersects_parabola (x1, Q.2) ∧ intersects_parabola (x2, Q.2) ∧ 
   equidistant s (x1, Q.2) ∧ equidistant s (x2, Q.2) → chord_slope_condition m) :=
by
  intros Q p q s h_parabola h_point_Q h_line_condition h_exists
  sorry

end chord_equidistant_proof_l761_761547


namespace ratio_of_averages_l761_761800

def rectangular_array (a : ℕ → ℕ → ℝ) := (∀ i j, 1 ≤ i ∧ i ≤ 50 ∧ 1 ≤ j ∧ j ≤ 60)

def modified_row_sum (a : ℕ → ℕ → ℝ) (i : ℕ) := 2 * (∑ j in Finset.range 60, a i j)
def modified_column_sum (a : ℕ → ℕ → ℝ) (j : ℕ) := 5 * (∑ i in Finset.range 50, a i j)

def average_row_sum (a : ℕ → ℕ → ℝ) : ℝ := 
  (∑ i in Finset.range 50, modified_row_sum a i) / 50 

def average_column_sum (a : ℕ → ℕ → ℝ) : ℝ := 
  (∑ j in Finset.range 60, modified_column_sum a j) / 60

theorem ratio_of_averages (a : ℕ → ℕ → ℝ) (h : rectangular_array a) :
  (average_row_sum a) / (average_column_sum a) = (12 : ℝ) / (25 : ℝ) :=
by
  sorry

end ratio_of_averages_l761_761800


namespace are_correct_statements_l761_761394

theorem are_correct_statements : ∀ A B C D : Prop,
  (A ↔ ¬(∀ x > 0, x^2 + 1 > 0) ↔ ∃ x > 0, x^2 + 1 ≤ 0) →
  (B ↔ ∀ m : ℝ, (f(x) = (m^2 - 3*m + 3)*x^(3*m - 4) (∀ x > 0, f(x) < 0 → m = 1))) →
  (C ↔ ∀ x : ℝ → x ≠ -2 ∧ f(x) = (x + 1) / (x + 2) ↔ ¬symmetric_about (x = -2, y = -1)) →
  (D ↔ ∀ x > 0, max (x^2 - x + 4) / x = -3) →
  (B ∧ D) := by sorry

end are_correct_statements_l761_761394


namespace problem_I_problem_II_l761_761405

theorem problem_I (x : ℝ) (h : x^(1/2) + x^(-1/2) = 3) :
  (x^2 + x^(-2) - 7) / (x + x^(-1) + 3) = 4 :=
sorry

theorem problem_II :
  (1 / 27)^(1 / 3) - (6.25)^(1 / 2) + (2 * sqrt 2)^(-2 / 3) + real.pi^0 - 3^(-1) = -1 :=
sorry

end problem_I_problem_II_l761_761405


namespace maximum_interval_length_l761_761268

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem maximum_interval_length 
  (m n : ℕ)
  (h1 : 0 < m)
  (h2 : m < n)
  (h3 : ∃ k : ℕ, ∀ i : ℕ, 0 ≤ i → i < k → ¬ is_multiple_of (m + i) 2000 ∧ (m + i) % 2021 = 0):
  n - m = 1999 :=
sorry

end maximum_interval_length_l761_761268


namespace masking_tape_needed_l761_761052

theorem masking_tape_needed {a b : ℕ} (h1 : a = 4) (h2 : b = 6) :
  2 * a + 2 * b = 20 :=
by
  rw [h1, h2]
  norm_num

end masking_tape_needed_l761_761052


namespace triangle_cosine_length_l761_761187

theorem triangle_cosine_length (LM MN : ℝ) (hLM : LM = 15) (hcosM : cos M = 3/5) : MN = 9 :=
by
  sorry

end triangle_cosine_length_l761_761187


namespace odd_factors_count_of_n_l761_761980

def n : ℕ := 2^4 * 3^3 * 5 * 7

theorem odd_factors_count_of_n : 
  let odd_factors (n : ℕ) := 
    (∏ p in (nat.factors n).erase 2, p)^((nat.factors n).count (∏ p in (nat.factors n).erase 2, p)) 
  in
  ∑_ (d in divisors n).filter (λ d, ¬ 2 ∣ d) 1 = 16 :=
by sorry

end odd_factors_count_of_n_l761_761980


namespace g_even_l761_761195

def g (x : ℝ) : ℝ := 4 / (3 * x^8 - 7)

theorem g_even : ∀ x : ℝ, g (-x) = g x := by
  -- here we would provide the proof, but we'll use sorry for now as specified
  sorry

end g_even_l761_761195


namespace montague_fraction_l761_761587

noncomputable def fraction_montague (M C : ℝ) : Prop :=
  M + C = 1 ∧
  (0.70 * C) / (0.20 * M + 0.70 * C) = 7 / 11

theorem montague_fraction : ∃ M C : ℝ, fraction_montague M C ∧ M = 2 / 3 :=
by sorry

end montague_fraction_l761_761587


namespace sum_elements_of_set_mul_l761_761230

def set_mul (A B : Set ℕ) : Set ℕ := { x | ∃ a ∈ A, ∃ b ∈ B, x = a * b }

noncomputable def A : Set ℕ := {1, 2}
noncomputable def B : Set ℕ := {0, 2}

theorem sum_elements_of_set_mul : (∑ x in (set_mul A B).toFinset, x) = 6 := by
  sorry

end sum_elements_of_set_mul_l761_761230


namespace problem1_problem2_l761_761864

noncomputable def f (x a b : ℝ) : ℝ := |x - a| - |x + b|
noncomputable def g (x a b : ℝ) : ℝ := -x^2 - a*x - b

-- Problem 1: Prove that a + b = 3
theorem problem1 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : ∀ x, f x a b ≤ 3) : a + b = 3 := 
sorry

-- Problem 2: Prove that 1/2 < a < 3
theorem problem2 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 3) 
  (h₃ : ∀ x, x ≥ a → g x a b < f x a b) : 1/2 < a ∧ a < 3 := 
sorry

end problem1_problem2_l761_761864


namespace inequality_f_l761_761077

-- Definitions and conditions
variables {X : Type} [metric_space X]
variables (A B C H I G A1 B1 C1 : X)
variable (f : X → ℝ)
-- Acute-angled triangle
axiom acute_triangle : ∀ (X : X), -- define an axiom or condition that ensures the triangle is acute

-- Inequalities for function f at H, I, and G
axiom f_def :
  f H = 1 / (cos (angle A B C) * cos (angle B C A) * cos (angle C A B)) ∧
  f I = 1 / (dist A B * dist A C * dist B C) * (dist A B + dist B C) * (dist B C + dist A C) * (dist A C + dist A B) ∧
  f G = 8

-- Proof statement
theorem inequality_f : f H ≥ f I ∧ f I ≥ f G :=
begin
  sorry
end

end inequality_f_l761_761077


namespace supremum_implies_infimum_n_equals_one_least_squares_l761_761986

variable {n : ℕ}
variable {X : ℝ^n}
variable {DX : Matrix n n ℝ} [NonDegenerateCovarianceMatrix DX]
variable {A : Set (ℝ^n)}
variable {Sigma : Matrix n n ℝ}
variable {a : ℝ^n}

noncomputable def normal_density (X : ℝ^n) (a : ℝ^n) (Sigma : Matrix n n ℝ) : ℝ := sorry

theorem supremum_implies_infimum (hX : RandomVector X)
    (hf : ∀ a, ∀ Sigma, normal_density X a Sigma = f X a Sigma)
    (hSigma : Sigma.Symmetric ∧ Sigma.PositiveDefinite)
    (hA : A ⊆ (ℝ^n)) :
  (sup a Sigma, (E (ln (normal_density X a Sigma)))) = inf a, (det (E (X - a) ⬝ (X - a)^T)) := sorry

theorem n_equals_one_least_squares (hX : RandomVector X)
    (hf : ∀ a, ∀ Sigma, normal_density X a Sigma = f X a Sigma)
    (hSigma : Sigma.Symmetric ∧ Sigma.PositiveDefinite)
    (hA : A ⊆ (ℝ^n))
    (hn : n = 1):
  (sup a Sigma, (E (ln (normal_density X a Sigma)))) = inf a, (E ((X - a)^2)) := sorry

end supremum_implies_infimum_n_equals_one_least_squares_l761_761986


namespace rectangle_side_length_l761_761249

theorem rectangle_side_length (a b c d : ℕ) 
  (h₁ : a = 3) 
  (h₂ : b = 6) 
  (h₃ : a / c = 3 / 4) : 
  c = 4 := 
by
  sorry

end rectangle_side_length_l761_761249


namespace range_of_a_l761_761229

variable (a : ℝ)

def proposition_p := ∀ x : ℝ, a * x^2 - 2 * x + 1 > 0
def proposition_q := ∀ x : ℝ, x ∈ Set.Icc (1/2 : ℝ) (2 : ℝ) → x + (1 / x) > a

theorem range_of_a (hp : proposition_p a) (hq : proposition_q a) : 1 < a ∧ a < 2 :=
sorry

end range_of_a_l761_761229


namespace area_triangle_ABD_l761_761955

theorem area_triangle_ABD {A B C D : Point} (hABC : Triangle A B C) 
  (h_angleA : angle A = 30) (h_angleB : angle B = 60) (h_angleC : angle C = 90)
  (h_area_ABC : area hABC = 25 * sqrt 3) (h_trisect : trisects BAC at D) : 
  area (triangle A B D) = 2.5 :=
sorry

end area_triangle_ABD_l761_761955


namespace parallel_line_through_point_eq_l761_761064

theorem parallel_line_through_point_eq (x y : ℝ) (a b : ℝ) (h1 : a ≠ 0 ∨ b ≠ 0) :
  (a * 2 + b * 1 + c = 0 → ∃ (c' : ℝ), a * x + b * y + c' = 0 ∧ 2 * x + 1 * y - 1 = 0) :=
by
  sorry

# Given conditions:
-- a = 1, b = -1 (from the equation x - y = 0, which has the same slope as x - y + 2 = 0)
-- The line passes through the point (2, 1), so c' = -1 when plugging in (2, 1) --> c' = 1

end parallel_line_through_point_eq_l761_761064


namespace matrix_multiplication_correct_l761_761465

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 1], ![4, -2]]
def matrix2 : Matrix (Fin 2) (Fin 2) ℤ := ![![7, -3], ![2, 2]]
def result_matrix : Matrix (Fin 2) (Fin 2) ℤ := ![![23, -7], ![24, -16]]

theorem matrix_multiplication_correct :
  matrix1.mul matrix2 = result_matrix :=
by
  sorry

end matrix_multiplication_correct_l761_761465


namespace Vasya_numbers_l761_761374

theorem Vasya_numbers : ∃ (x y : ℝ), x + y = xy ∧ xy = x / y ∧ (x, y) = (1 / 2, -1) :=
by
  sorry

end Vasya_numbers_l761_761374


namespace not_777_integers_l761_761076

theorem not_777_integers (p : ℕ) (hp : Nat.Prime p) :
  ¬ (∃ count : ℕ, count = 777 ∧ ∀ n : ℕ, ∃ k : ℕ, (n ^ 3 + n * p + 1 = k * (n + p + 1))) :=
by
  sorry

end not_777_integers_l761_761076


namespace range_of_a_l761_761149

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 1 → Real.exp (a * x) ≥ 2 * Real.log x + x^2 - a * x) ↔ 0 ≤ a :=
sorry

end range_of_a_l761_761149


namespace length_BO_l761_761639

theorem length_BO (A B C D E O: Point)
  (hAC_diameter : is_diameter (circle C) (line A C))
  (hCircle: is_circle (circle C) (line A C))
  (hIntersect_AB : intersects_on (circle C) (line A B) D)
  (hIntersect_BC : intersects_on (circle C) (line B C) E)
  (hAngle_EDC: ∠ E D C = 30°)
  (hLength_AE: length (segment A E) = √3)
  (hArea_ratio: area (triangle D B E) / area (triangle A B C) = 1/2)
  (hIntersect_AE_CD: intersects_on (line A E) (line C D) O):
  length (segment B O) = 2 := sorry

end length_BO_l761_761639


namespace symmetric_point_coordinates_l761_761285

-- Define the type for 3D points
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the symmetric point function with respect to the x-axis
def symmetricPointWithRespectToXAxis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

-- Define the specific point
def givenPoint : Point3D := { x := 2, y := 3, z := 4 }

-- State the theorem to be proven
theorem symmetric_point_coordinates : 
  symmetricPointWithRespectToXAxis givenPoint = { x := 2, y := -3, z := -4 } :=
by
  sorry

end symmetric_point_coordinates_l761_761285


namespace tangent_sum_formula_application_l761_761834

-- Define the problem's parameters and statement
noncomputable def thirty_three_degrees_radian := Real.pi * 33 / 180
noncomputable def seventeen_degrees_radian := Real.pi * 17 / 180
noncomputable def twenty_eight_degrees_radian := Real.pi * 28 / 180

theorem tangent_sum_formula_application :
  Real.tan seventeen_degrees_radian + Real.tan twenty_eight_degrees_radian + Real.tan seventeen_degrees_radian * Real.tan twenty_eight_degrees_radian = 1 := 
sorry

end tangent_sum_formula_application_l761_761834


namespace binom_150_1_l761_761016

-- Definition of the binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_150_1 : binom 150 1 = 150 := by
  -- The proof is skipped and marked as sorry
  sorry

end binom_150_1_l761_761016


namespace three_digit_multiples_of_25_not_75_count_l761_761138

-- Definitions from conditions.
def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000
def is_multiple_of_25 (n : ℕ) : Prop := n % 25 = 0
def is_multiple_of_75 (n : ℕ) : Prop := n % 75 = 0

-- The theorem statement.
theorem three_digit_multiples_of_25_not_75_count : 
  let count := (finset.filter (λ n, is_three_digit n ∧ is_multiple_of_25 n ∧ ¬ is_multiple_of_75 n) (finset.range 1000)).card
  in count = 24 :=
by
  sorry

end three_digit_multiples_of_25_not_75_count_l761_761138


namespace log_condition_l761_761914

noncomputable def is_non_square_non_cube_non_integral_rational (x : ℝ) : Prop :=
  ¬∃ n : ℤ, x = n^2 ∨ x = n^3 ∨ (x.denom = 1)

theorem log_condition (x : ℝ) (h : log (3 * x) 343 = x) : is_non_square_non_cube_non_integral_rational x := 
sorry

end log_condition_l761_761914


namespace parking_lot_motorcycles_l761_761938

theorem parking_lot_motorcycles
  (x y : ℕ)
  (h1 : x + y = 24)
  (h2 : 3 * x + 4 * y = 86) : x = 10 :=
by
  sorry

end parking_lot_motorcycles_l761_761938


namespace Vasya_numbers_l761_761364

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 :=
by
  sorry

end Vasya_numbers_l761_761364


namespace solution_set_quadratic_ineq_l761_761161

theorem solution_set_quadratic_ineq (a m : ℝ) (h : a > 0) 
  (h1: ∀ x, 1 < x ∧ x < m ↔ ax^2 - 6x + a^2 < 0) : m = 2 :=
sorry

end solution_set_quadratic_ineq_l761_761161


namespace top_angle_degrees_l761_761097

def isosceles_triangle_with_angle_ratio (x : ℝ) (a b c : ℝ) : Prop :=
  a = x ∧ b = 4 * x ∧ a + b + c = 180 ∧ (a = b ∨ a = c ∨ b = c)

theorem top_angle_degrees (x : ℝ) (a b c : ℝ) :
  isosceles_triangle_with_angle_ratio x a b c → c = 20 ∨ c = 120 :=
by
  sorry

end top_angle_degrees_l761_761097


namespace min_distance_time_and_distance_correct_l761_761736

noncomputable def min_dist_time_and_distance : ℝ × ℝ :=
let 
  t_min := 5 / 2, 
  dist_min := 37.5 
in (t_min, dist_min)

theorem min_distance_time_and_distance_correct :
  ∃ t d : ℝ, 
  let 
      initial_distance : ℝ := 100, 
      speed_A : ℝ := 50, 
      accel_B : ℝ := 20 in 
  t = 5 / 2 ∧ d = 37.5 ∧
  (∀ t' : ℝ, (100 + (1 / 2) * 20 * t'^2 - 50 * t' + initial_distance) ≥ 37.5) :=
begin
  sorry
end

end min_distance_time_and_distance_correct_l761_761736


namespace max_1200th_day_celebration_l761_761634

theorem max_1200th_day_celebration : 
  ∃ day : nat, 
    birth_day = 5 ∧
    ∀ n : ℕ, celebration_day = 1200 → 
    nat.cycle_day birth_day celebration_day = day ∧ day = 6
:= 
begin
  -- Given: Max was born on Friday, January 1st, 2010.
  -- birth_day == Friday (5th day of the week if week starts on Sunday)
  let birth_day := 5,

  -- Celebration on the 1200th day of Max's life
  let celebration_day := 1200,

  -- The calculation part would proceed but since it's not required we'll put sorry for now.
  sorry,
end


end max_1200th_day_celebration_l761_761634


namespace angle_sum_x_y_l761_761949

def angle_A := 36
def angle_B := 80
def angle_C := 24

def target_sum : ℕ := 140

theorem angle_sum_x_y (angle_A angle_B angle_C : ℕ) (x y : ℕ) : 
  angle_A = 36 → angle_B = 80 → angle_C = 24 → x + y = 140 := by 
  intros _ _ _
  sorry

end angle_sum_x_y_l761_761949


namespace dot_product_parallel_angle_perpendicular_l761_761508

variables (a b : ℝ^3) (θ : ℝ)
-- Conditions
axiom norm_a : ‖a‖ = 1
axiom norm_b : ‖b‖ = sqrt 2
axiom angle_ab : θ = angle_between a b

-- Problem 1
-- Proof that if a ∥ b then a • b = ± sqrt 2
theorem dot_product_parallel (h_par : a ∥ b) : a ⬝ b = sqrt 2 ∨ a ⬝ b = -sqrt 2 := 
sorry

-- Problem 2
-- Proof that if (a - b) ⊥ a then θ = π/4
theorem angle_perpendicular (h_perp : (a - b) ⊥ a) : θ = π / 4 := 
sorry

end dot_product_parallel_angle_perpendicular_l761_761508


namespace count_multiples_of_25_but_not_75_3_digit_l761_761135

theorem count_multiples_of_25_but_not_75_3_digit :
  let is_3_digit (n : ℕ) := (100 ≤ n) ∧ (n ≤ 999)
  let is_multiple_of_25 (n : ℕ) := ∃ k : ℕ, n = 25 * k
  let is_multiple_of_75 (n : ℕ) := ∃ m : ℕ, n = 75 * m
  (finset.filter (λ n : ℕ, is_3_digit n ∧ is_multiple_of_25 n ∧ ¬ is_multiple_of_75 n) (finset.range 1000)).card = 24 := by
  sorry

end count_multiples_of_25_but_not_75_3_digit_l761_761135


namespace problem1_problem2_l761_761893

section Problem1

def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x + 1

def g (k : ℝ) (x : ℝ) : ℝ := k * x + 1 - Real.log x

def h (k : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then f x else g k x

theorem problem1 (k : ℝ) (hk : k < 0) :
  (k < -1 → ∃ x, h k x = 0 ∧ h k x ≠ 1) ∧
  (-1 ≤ k ∧ k < 0 → ∃ x1 x2, x1 ≠ x2 ∧ h k x1 = 0 ∧ h k x2 = 0) :=
sorry

end Problem1

section Problem2

def H (a t : ℝ) : ℝ := 4 * t ^ 3 - 3 * t ^ 2 - 6 * t ^ 2 * a + 6 * t * a - 5

def H' (a t : ℝ) : ℝ := 12 * t ^ 2 - 6 * t - 12 * t * a + 6 * a

theorem problem2 (a : ℝ) :
  (∃ t1 t2 t3, t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ H a t1 = 0 ∧ H a t2 = 0 ∧ H a t3 = 0)
  ↔ (a < -1 ∨ a > 7 / 2) :=
sorry

end Problem2

end problem1_problem2_l761_761893


namespace third_side_length_l761_761582

theorem third_side_length (a b : ℕ) (c : ℕ) (h₁ : a = 3) (h₂ : b = 6) (h₃ : odd c) (h₄ : a + b > c) (h₅ : b + c > a) (h₆ : c + a > b) : c = 5 ∨ c = 7 :=
by
  subst h₁
  subst h₂
  -- Apply the triangle inequalities and derive that c must be either 5 or 7
  sorry

end third_side_length_l761_761582


namespace rectangle_cut_square_l761_761806

theorem rectangle_cut_square (a b r h: ℕ) (ha: a = 9) (hb: b = 12) (hr: r = 1) (hh: h = 8)
    (total_area: a * b = 108) (hole_area: r * h = 8) (usable_area: total_area - hole_area = 100) :
    ∃ s: ℕ, (s * s = 100) ∧ (50 + 50 = 100) ∧
    ((∃ p: ℕ, ∃ q: ℕ, ∃ x: ℕ, ∃ y: ℕ, p * x = 50 ∧ q * y = 50) ∧ (∃ n : ℕ, n * n = 100)) := 
sorry

end rectangle_cut_square_l761_761806


namespace B_profit_percentage_l761_761756

theorem B_profit_percentage (cost_price_A : ℝ) (profit_A : ℝ) (selling_price_C : ℝ) 
  (h1 : cost_price_A = 154) 
  (h2 : profit_A = 0.20) 
  (h3 : selling_price_C = 231) : 
  (selling_price_C - (cost_price_A * (1 + profit_A))) / (cost_price_A * (1 + profit_A)) * 100 = 25 :=
by
  sorry

end B_profit_percentage_l761_761756


namespace complement_union_l761_761125

theorem complement_union (U A B complement_U_A : Set Int) (hU : U = {-1, 0, 1, 2}) 
  (hA : A = {-1, 2}) (hB : B = {0, 2}) (hC : complement_U_A = {0, 1}) :
  complement_U_A ∪ B = {0, 1, 2} := by
  sorry

end complement_union_l761_761125


namespace train_length_l761_761434

-- Define the train crossing problem as a theorem
theorem train_length (L : ℝ) : 
  (∃ (S : ℝ), S = (L + 350) / 15 ∧ S = (L + 500) / 20) -> L = 100 :=
begin
  sorry
end

end train_length_l761_761434


namespace geom_seq_b_ac_l761_761920

theorem geom_seq_b_ac (a b c : ℝ) (h_geom : -2, a, b, c, -8 forms a geometric sequence) :
    b = -4 ∧ a * c = 16 :=
by
  -- Proof will go here
  sorry

end geom_seq_b_ac_l761_761920


namespace find_m_and_k_l761_761512

noncomputable def f (a : ℝ) (m : ℝ) (x : ℝ) := a^(2 * x) + m * a^(-2 * x)
noncomputable def g (f : ℝ → ℝ) (a : ℝ) (k : ℝ) (x : ℝ) := f x - 2 * k * f (x / 2) + 2 * a^(-2 * x)

theorem find_m_and_k (a : ℝ) (m : ℝ) (k : ℝ) :
  (∀ x : ℝ, f a m (-x) = -f a m x) →
  a ≠ 0 →
  a ≠ 1 →
  f a m 1 = 15 / 4 →
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ g (f a (-1)) a k x = 2) →
  m = -1 ∧ (∀ k : ℝ, (k ∈ set.Iic 0)) :=
by
  sorry

end find_m_and_k_l761_761512


namespace geometric_sequence_s4_l761_761868

noncomputable def geometric_sequence_sum : ℕ → ℝ → ℝ → ℝ
| 0, a1, q => 0
| (n+1), a1, q => a1 * (1 - q^(n+1)) / (1 - q)

variable (a1 q : ℝ) (n : ℕ)

theorem geometric_sequence_s4  (h1 : a1 * (q^1) * (q^3) = 16) (h2 : geometric_sequence_sum 2 a1 q + a1 * (q^2) = 7) :
  geometric_sequence_sum 3 a1 q = 15 :=
sorry

end geometric_sequence_s4_l761_761868


namespace rebate_amount_l761_761636

theorem rebate_amount : 
  let cost_polo_shirts := 3 * 26,
      cost_necklaces := 2 * 83,
      cost_computer_game := 90,
      total_cost_before_rebate := cost_polo_shirts + cost_necklaces + cost_computer_game,
      total_cost_after_rebate := 322,
      rebate := total_cost_before_rebate - total_cost_after_rebate
  in rebate = 12 :=
by
  let cost_polo_shirts := 3 * 26
  let cost_necklaces := 2 * 83
  let cost_computer_game := 90
  let total_cost_before_rebate := cost_polo_shirts + cost_necklaces + cost_computer_game
  let total_cost_after_rebate := 322
  let rebate := total_cost_before_rebate - total_cost_after_rebate
  show rebate = 12
  sorry

end rebate_amount_l761_761636


namespace initial_percentage_of_milk_l761_761322

theorem initial_percentage_of_milk (P : ℝ) :
  (P / 100) * 60 = (68 / 100) * 74.11764705882354 → P = 84 :=
by
  sorry

end initial_percentage_of_milk_l761_761322


namespace find_leftmost_vertex_l761_761317

open Real

/-- Define the vertices of the quadrilateral on the graph of y = e^x. -/
def vertices (m : ℕ) : list (ℝ × ℝ) :=
  [(m : ℝ, exp m), (m+1, exp (m+1)), (m+2, exp (m+2)), (m+3, exp (m+3))]

/-- Calculate the area using the Shoelace Theorem. -/
noncomputable def shoelace_area (v : list (ℝ × ℝ)) : ℝ :=
  0.5 * abs (v.nth 0 v.nth 2 v.nth 3 - v.nth 1 v.nth 3 v.nth 2)

/-- The main theorem: Find the x-coordinate of the leftmost vertex. -/
theorem find_leftmost_vertex (m : ℕ) (hm : m ∈ {1, 2, 3, 4, 5}) :
  shoelace_area (vertices m) = (exp 2 - 1) / exp 1 := 
sorry

end find_leftmost_vertex_l761_761317


namespace elijah_needs_20_meters_of_tape_l761_761050

def wall_width_4m_2 (n : Nat) : Prop :=
  n = 2 * 4

def wall_width_6m_2 (n : Nat) : Prop :=
  n = 2 * 6

def total_masking_tape (tape : Nat) : Prop :=
  ∃ n1 n2, wall_width_4m_2 n1 ∧ wall_width_6m_2 n2 ∧ tape = n1 + n2

theorem elijah_needs_20_meters_of_tape : total_masking_tape 20 :=
by
  unfold total_masking_tape
  apply Exists.intro 8
  apply Exists.intro 12
  unfold wall_width_4m_2 wall_width_6m_2
  split
  . rfl
  split
  . rfl
  . rfl

end elijah_needs_20_meters_of_tape_l761_761050


namespace weather_forecast_probability_l761_761307

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem weather_forecast_probability :
  binomial_probability 3 2 0.8 = 0.384 :=
by
  sorry

end weather_forecast_probability_l761_761307


namespace correct_proposition_l761_761770

theorem correct_proposition :
  -- Proposition 1: If \(\alpha > \beta\), then \(\sin\alpha > \sin\beta\)
  (∀ (α β : ℝ), α > β → sin α > sin β) = false ∧

  -- Proposition 2: The negation of "For all \(x > 1\), \(x^2 > 1\)" is
  -- "There exists \(x \leq 1\), \(x^2 \leq 1\)"
  (¬ (∀ (x : ℝ), (x > 1 → x^2 > 1)) = ∃ (x : ℝ), x ≤ 1 ∧ x^2 ≤ 1) = false ∧

  -- Proposition 3: The necessary and sufficient condition for the lines \(ax + y + 2 = 0\)
  -- and \(ax - y + 4 = 0\) to be perpendicular is \(a = \pm1\)
  (∀ (a : ℝ), (∃ (m n : ℝ), m * n = -(a * a) ∧ m * n + 1 = 0) ↔ a = 1 ∨ a = -1) = true ∧

  -- Proposition 4: The contrapositive of "If \(xy = 0\), then \(x = 0\) or \(y = 0\)"
  -- is "If \(x \neq 0\) or \(y \neq 0\), then \(xy \neq 0\)"
  (∀ (x y : ℝ), (x * y = 0 → x = 0 ∨ y = 0) = (¬ (x ≠ 0 ∧ y ≠ 0) → x * y ≠ 0)) = false := 
sorry

end correct_proposition_l761_761770


namespace perfect_square_l761_761860

-- Define natural numbers m and n and the condition mn ∣ m^2 + n^2 + m
variables (m n : ℕ)

-- Define the condition as a hypothesis
def condition (m n : ℕ) : Prop := (m * n) ∣ (m ^ 2 + n ^ 2 + m)

-- The main theorem statement: if the condition holds, then m is a perfect square
theorem perfect_square (m n : ℕ) (h : condition m n) : ∃ k : ℕ, m = k ^ 2 :=
sorry

end perfect_square_l761_761860


namespace total_travel_time_l761_761790

noncomputable def washingtonToIdahoDistance : ℕ := 640
noncomputable def idahoToNevadaDistance : ℕ := 550
noncomputable def washingtonToIdahoSpeed : ℕ := 80
noncomputable def idahoToNevadaSpeed : ℕ := 50

theorem total_travel_time :
  (washingtonToIdahoDistance / washingtonToIdahoSpeed) + (idahoToNevadaDistance / idahoToNevadaSpeed) = 19 :=
by
  sorry

end total_travel_time_l761_761790


namespace vasya_numbers_l761_761382

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : 
  (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) ∨ (x = y ∧ x = 0) :=
by 
  -- Placeholder to show where the proof would go
  sorry

end vasya_numbers_l761_761382


namespace max_attendees_l761_761047

-- Define the days of the week.
inductive Day
| Mon | Tues | Wed | Thurs | Fri

open Day

-- Define the unavailability of each person.
def isUnvailable_Alice : Day → Bool :=
  fun d => d = Mon ∨ d = Wed ∨ d = Fri

def isUnvailable_Bob : Day → Bool :=
  fun d => d = Tues ∨ d = Thurs

def isUnvailable_Charlie : Day → Bool :=
  fun d => d = Mon ∨ d = Tues ∨ d = Fri

def isUnvailable_Diana : Day → Bool :=
  fun d => d = Wed ∨ d = Thurs

-- Implementation of available attendees on a particular day.
def availableAttendees (d : Day) : Nat :=
  (if ¬isUnvailable_Alice d then 1 else 0) +
  (if ¬isUnvailable_Bob d then 1 else 0) +
  (if ¬isUnvailable_Charlie d then 1 else 0) +
  (if ¬isUnvailable_Diana d then 1 else 0)

-- The theorem indicating the maximum number of available attendees on any given day.
theorem max_attendees : (d : Day) → availableAttendees d = 2 :=
by
  intro d
  cases d
  case Mon =>
    simp [availableAttendees, isUnvailable_Alice, isUnvailable_Bob, isUnvailable_Charlie, isUnvailable_Diana]
  case Tues =>
    simp [availableAttendees, isUnvailable_Alice, isUnvailable_Bob, isUnvailable_Charlie, isUnvailable_Diana]
  case Wed =>
    simp [availableAttendees, isUnvailable_Alice, isUnvailable_Bob, isUnvailable_Charlie, isUnvailable_Diana]
  case Thurs =>
    simp [availableAttendees, isUnvailable_Alice, isUnvailable_Bob, isUnvailable_Charlie, isUnvailable_Diana]
  case Fri =>
    simp [availableAttendees, isUnvailable_Alice, isUnvailable_Bob, isUnvailable_Charlie, isUnvailable_Diana]
  done

end max_attendees_l761_761047


namespace inv_f_eq_four_l761_761669

def f (x : ℝ) : ℝ := (3 * x + 2) / (x - 4)

theorem inv_f_eq_four : 
  ∀ a b c d : ℝ, 
  (∀ x : ℝ, f x = (3 * x + 2) / (x - 4)) ->
  (f⁻¹ x = (a * x + b) / (c * x + d)) -> 
  (a / c = 4) :=
by
  -- Proof omitted
  sorry

end inv_f_eq_four_l761_761669


namespace pitcher_fill_four_glasses_l761_761392

variable (P G : ℚ) -- P: Volume of pitcher, G: Volume of one glass
variable (h : P / 2 = 3 * G)

theorem pitcher_fill_four_glasses : (4 * G = 2 * P / 3) :=
by
  sorry

end pitcher_fill_four_glasses_l761_761392


namespace x_plus_3y_value_l761_761127

theorem x_plus_3y_value (x y : ℝ) (h1 : x + y = 19) (h2 : x + 2y = 10) : x + 3y = 1 :=
by
  -- The proof is omitted as per instructions
  sorry

end x_plus_3y_value_l761_761127


namespace max_value_k_l761_761081

def is_in_set (x y : Int) : Prop :=
  (|x| - 1) ^ 2 + (|y| - 1) ^ 2 < 4

def M := {p : Int × Int | is_in_set p.1 p.2}

def count_points_with_property (k : Int) :=
  (M : Set (Int × Int)).count (λ p, p.1 * p.2 ≥ k)

theorem max_value_k (k : Int) (h₀ : k > 0) (h₁ : count_points_with_property k = 6) :
  k = 2 :=
sorry

end max_value_k_l761_761081


namespace price_of_other_frisbees_l761_761428

-- Lean 4 Statement
theorem price_of_other_frisbees (P : ℝ) (x : ℕ) (h1 : x ≥ 40) (h2 : P * x + 4 * (60 - x) = 200) :
  P = 3 := 
  sorry

end price_of_other_frisbees_l761_761428


namespace number_of_arrangements_with_CD_adjacent_l761_761829

theorem number_of_arrangements_with_CD_adjacent :
  let n := 6 in
  let units := n - 1 in
  let ways_to_arrange_units := Nat.factorial units in
  let ways_to_arrange_CD := 2 in
  ways_to_arrange_units * ways_to_arrange_CD = 240 := by
  sorry

end number_of_arrangements_with_CD_adjacent_l761_761829


namespace age_sum_is_47_l761_761722

theorem age_sum_is_47 (a b c : ℕ) (b_def : b = 18) 
  (a_def : a = b + 2) (c_def : c = b / 2) : a + b + c = 47 :=
by
  sorry

end age_sum_is_47_l761_761722


namespace hyperbola_eccentricity_l761_761545

noncomputable def eccentricity_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : Prop :=
  let c := Real.sqrt (a^2 + b^2) in
  a + c < b^2 / a

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h : eccentricity_range a b h1 h2) : 
  let e := Real.sqrt (1 + b^2 / a^2) in
  e > 2 := 
by
  let c := Real.sqrt (a^2 + b^2)
  sorry

end hyperbola_eccentricity_l761_761545


namespace unique_fib_sum_representation_l761_761647

def fib : ℕ → ℕ
| 0 := 1
| 1 := 1
| (n + 2) := fib (n + 1) + fib n

theorem unique_fib_sum_representation (n : ℕ) (h : n > 0) : 
  ∃! (fib_indices : list ℕ), 
    (∀ i, i ∈ fib_indices → i ≥ 2) ∧ 
    (∀ (i j : ℕ), i ∈ fib_indices → j ∈ fib_indices → i ≠ j → abs (i - j) ≠ 1) ∧ 
    (list.sum (fib_indices.map fib) = n) := 
sorry

end unique_fib_sum_representation_l761_761647


namespace coby_travel_time_l761_761797

def travel_time (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

theorem coby_travel_time :
  let wash_to_idaho_distance := 640
  let idaho_to_nevada_distance := 550
  let wash_to_idaho_speed := 80
  let idaho_to_nevada_speed := 50
  travel_time wash_to_idaho_distance wash_to_idaho_speed + travel_time idaho_to_nevada_distance idaho_to_nevada_speed = 19 := by
  sorry

end coby_travel_time_l761_761797


namespace line_equation_through_point_l761_761823

theorem line_equation_through_point 
  (x y : ℝ)
  (h1 : (5, 2) ∈ {p : ℝ × ℝ | p.2 = p.1 * (2 / 5)})
  (h2 : (5, 2) ∈ {p : ℝ × ℝ | p.1 / 6 + p.2 / 12 = 1}) 
  (h3 : (5,2) ∈ {p : ℝ × ℝ | 2 * p.1 = p.2 }) :
  (2 * x + y - 12 = 0 ∨ 
   2 * x - 5 * y = 0) := 
sorry

end line_equation_through_point_l761_761823


namespace geraldine_banana_count_l761_761842

variable (b : ℕ) -- the number of bananas Geraldine ate on June 1

theorem geraldine_banana_count 
    (h1 : (5 * b + 80 = 150)) 
    : (b + 32 = 46) :=
by
  sorry

end geraldine_banana_count_l761_761842


namespace find_smallest_n_l761_761068

-- Define the trigonometric functions used in the problem.
noncomputable def trig_sum : ℝ :=
  (finset.range 59).sum (λ k, 1 / (real.sin (30 + k) * real.sin (31 + k)))

-- The main theorem to prove
theorem find_smallest_n : ∃ n : ℕ, n > 0 ∧ trig_sum = 1 / real.cos (n:ℝ) ∧ n = 1 :=
by
  use 1
  -- Proof placeholder:
  sorry

end find_smallest_n_l761_761068


namespace seven_pencils_same_color_l761_761320

open Classical

theorem seven_pencils_same_color (pencils : Fin 25 → ℕ) 
  (h : ∀ (s : Finset (Fin 25)), s.card = 5 → ∃ (x : Fin 25) (y : Fin 25), x ≠ y ∧ pencils x = pencils y) :
  ∃ c, 7 ≤ (Finset.univ.filter (λ i, pencils i = c)).card :=
by {
  sorry
}

end seven_pencils_same_color_l761_761320


namespace ratio_c_d_approx_l761_761901

theorem ratio_c_d_approx (a b c d e f : ℝ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : d / e = 3)
  (h4 : e / f = 1 / 2)
  (h5 : (a * b * c) / (d * e * f) = 0.75) :
  c / d ≈ 0.4333 := sorry

end ratio_c_d_approx_l761_761901


namespace min_value_expr_l761_761621

theorem min_value_expr (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) :
  (x - 2)^2 + ((y / x) - 1)^2 + ((z / y) - 1)^2 + ((5 / z) - 1)^2 = 9 :=
sorry

end min_value_expr_l761_761621


namespace find_k_l761_761178

def P_b (b n : ℕ) : ℝ := Real.log b ((n + 1) / n : ℝ)

theorem find_k (k : ℕ) (h1 : k ∈ {k : ℕ | k > 0}) (h2 : k ≤ 20) :
  (∑ n in Finset.range (21 - k), P_b 10 (k + n)) = (Real.log 2 21 - Real.log 2 3) / (1 + Real.log 2 5) ↔ k = 3 := 
sorry

end find_k_l761_761178


namespace find_n_value_l761_761085

theorem find_n_value (n : ℕ) (a b : ℕ → ℝ) (h_a : ∀ n, a n = 2 ^ (-n + 3)) (h_b : ∀ n, b n = 2 ^ (n - 1)) :
  (a n * b n + 1 > a n + b n) ↔ n = 2 :=
by
  sorry

end find_n_value_l761_761085


namespace cyclic_sum_minimum_value_l761_761861

variable (a b c d : ℝ)

theorem cyclic_sum_minimum_value 
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) (h₄ : a + b + c + d = 4) :
  (cyclic_sum : ℝ) (cyclic_sum = (b + 3) / (a ^ 2 + 4) + (c + 3) / (b ^ 2 + 4) + (d + 3) / (c ^ 2 + 4) + (a + 3) / (d ^ 2 + 4)) ->
  cyclic_sum ≥ 3 := by
  sorry

end cyclic_sum_minimum_value_l761_761861


namespace cubic_polynomial_tangent_rational_roots_l761_761967

variable {f : ℚ[X]}

theorem cubic_polynomial_tangent_rational_roots (h_cubic: degree f = 3)
  (h_tangent: ∃ r : ℚ, multiplicity r f ≥ 2) : ∀ r : ℚ, IsRoot f r → ∃ s : ℚ, f = polynomial.C s * (X - polynomial.C r)^2 * (X - polynomial.C s) :=
sorry

end cubic_polynomial_tangent_rational_roots_l761_761967


namespace prod_fraction_eq_l761_761007

theorem prod_fraction_eq :
  (∏ n in Finset.range 30, (n + 6) / (n + 1)) = 326284 :=
by
  sorry

end prod_fraction_eq_l761_761007


namespace abs_diff_of_two_numbers_l761_761676

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 45) (h2 : x * y = 504) : |x - y| = 3 :=
by
  sorry

end abs_diff_of_two_numbers_l761_761676


namespace find_kn_l761_761585

-- Define the arithmetic sequence
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

-- Define the conditions
axiom d_ne_0 : ∃ d : ℤ, d ≠ 0
axiom a1 : ∃ a : ℤ, ∃ d : ℤ, d ≠ 0
axiom a2_geometric_mean_a1_a4 : ∀ (a d : ℤ), d ≠ 0 → (a + d)^2 = a * (a + 3 * d)

-- Define the problem of finding the general term of k_n
theorem find_kn (k_n : ℕ → ℤ) : 
  (∀ (a d : ℤ), d ≠ 0 → (a + d)^2 = a * (a + 3 * d)) → 
  (∀ (d : ℤ), d ≠ 0 → 
  k_n = λ n, 3^(n-1)) :=
by sorry

end find_kn_l761_761585


namespace time_to_fill_tub_l761_761251

def tubVolume : ℕ := 120
def leakyRate : ℕ := 1
def flowRate : ℕ := 12
def fillCycleTime : ℕ := 2
def netGainPerCycle : ℕ := (flowRate - leakyRate) - leakyRate

theorem time_to_fill_tub : 
    ∃ (time_in_minutes : ℕ), (time_in_minutes = 24) ∧ (tubVolume = 12 * netGainPerCycle * fillCycleTime) :=
begin
  sorry
end

end time_to_fill_tub_l761_761251


namespace john_trip_time_l761_761611

theorem john_trip_time (x : ℝ) (h : x + 2 * x + 2 * x = 10) : x = 2 :=
by
  sorry

end john_trip_time_l761_761611


namespace problem_statement_l761_761912

-- Define conditions
def is_solution (x : ℝ) : Prop :=
  real.log (343) / real.log (3 * x) = x

-- Formulate what we need to prove about the solution
def is_non_square_non_cube_non_integral_rational (x : ℝ) : Prop :=
  x ∈ set_of (λ x : ℚ, ¬is_square x ∧ ¬is_cube x ∧ frac x ≠ 0)

-- The main statement: Prove that x, satisfying the conditions, has the specified properties
theorem problem_statement (x : ℝ) (hx : is_solution x) : is_non_square_non_cube_non_integral_rational x :=
sorry

end problem_statement_l761_761912


namespace bin101_to_decimal_l761_761034

-- Define the binary representation of 101 (base 2)
def bin101 : ℕ := 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- State the theorem that asserts the decimal value of 101 (base 2) is 5
theorem bin101_to_decimal : bin101 = 5 := by
  sorry

end bin101_to_decimal_l761_761034


namespace MN_parallel_BH_l761_761597

-- Definitions of points and lines in triangle ABC satisfying the conditions
variables {A B C H B1 M N A1 C1 : Type}

-- Assume relations and intersections given in the conditions
variables [IsAltitude BH A B C]
variables [IsMedian BB_1 A B C]
variables [IsMidline A1 C1 A B C]
variables [PointOnLine A1 BC]
variables [PointOnLine C1 AB]
variables [IntersectAt A1 C1 BB_1 M]
variables [IntersectAt C1 B1 A1 H N]

-- Goal: Prove that MN and BH are parallel
theorem MN_parallel_BH : Parallel MN BH :=
by
  sorry

end MN_parallel_BH_l761_761597


namespace number_of_proper_subsets_of_set_A_l761_761670

theorem number_of_proper_subsets_of_set_A : 
  ∃ A : set ℕ, A = {0, 1, 2} ∧ (set_proper_subsets_count A = 7) :=
by {
  let A := {0, 1, 2},
  have h1 : A = {0, 1, 2} := rfl,
  have h2 : set_proper_subsets_count A = 7 := sorry,
  exact ⟨A, h1, h2⟩
}

end number_of_proper_subsets_of_set_A_l761_761670


namespace value_of_transformed_product_of_roots_l761_761212

theorem value_of_transformed_product_of_roots 
  (a b : ℚ)
  (h1 : 3 * a^2 + 4 * a - 7 = 0)
  (h2 : 3 * b^2 + 4 * b - 7 = 0)
  (h3 : a ≠ b) : 
  (a - 2) * (b - 2) = 13 / 3 :=
by
  -- The exact proof would be completed here.
  sorry

end value_of_transformed_product_of_roots_l761_761212


namespace minimize_PA_PB_PN_l761_761522

-- Define points A, B, and the line l
variables (A B : Point) (l : Line)

-- Define the midpoint M of segment AB
def midpoint (A B : Point) : Point :=
  Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2)

-- Define the foot of the perpendicular from a point P to line l
def foot_of_perpendicular (P : Point) (l : Line) : Point :=
  -- Function to get the foot of the perpendicular (pseudo-definition)
  sorry

-- Define the distance squared between two points
def dist_sq (P Q : Point) : ℝ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

-- Define the condition that GC = 2MG
def GC_eq_2MG (G : Point) (M C : Point) : Prop :=
  dist_sq G C = 2 * dist_sq M G

-- Define the minimizing point P
def minimizing_point (A B : Point) (l : Line) : Point :=
  let M := midpoint A B in
  let N := foot_of_perpendicular M l in
  let G := sorry in -- Function to construct G such that GC = 2MG
  G

-- Prove the point P minimizes the given expression
theorem minimize_PA_PB_PN (P : Point) (l : Line) (A B : Point) :
  let N := foot_of_perpendicular P l in
  let M := midpoint A B in
  let G := minimizing_point A B l in
  P = G ↔ dist_sq P A + dist_sq P B + dist_sq P N = dist_sq G A + dist_sq G B + dist_sq G N := 
sorry

end minimize_PA_PB_PN_l761_761522


namespace values_of_a_b_extreme_values_l761_761114

noncomputable def f (a b x : ℝ) : ℝ := 3 * (a * x^3 + b * x^2)

theorem values_of_a_b (a b : ℝ) :
  f a b 1 = 3 ∧ (9 * a * 1^2 + 6 * b * 1 = 0) → (a = -2 ∧ b = 3) :=
by
  intro h
  cases h with h1 h2
  sorry

noncomputable def fx (x : ℝ) : ℝ := -6 * x^3 + 9 * x^2

theorem extreme_values :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → fx x ≤ 15 ∧ fx x ≥ -81) ∧
  (fx (-1) = 15) ∧
  (fx (3) = -81) :=
by
  sorry

end values_of_a_b_extreme_values_l761_761114


namespace find_a_for_positive_root_l761_761570

theorem find_a_for_positive_root (h : ∃ x > 0, (1 - x) / (x - 2) = a / (2 - x) - 2) : a = 1 :=
sorry

end find_a_for_positive_root_l761_761570


namespace total_wash_time_l761_761638

theorem total_wash_time (clothes_time : ℕ) (towels_time : ℕ) (sheets_time : ℕ) (total_time : ℕ) 
  (h1 : clothes_time = 30) 
  (h2 : towels_time = 2 * clothes_time) 
  (h3 : sheets_time = towels_time - 15) 
  (h4 : total_time = clothes_time + towels_time + sheets_time) : 
  total_time = 135 := 
by 
  sorry

end total_wash_time_l761_761638


namespace range_of_m_l761_761530

variable {f : ℝ → ℝ}

theorem range_of_m 
  (even_f : ∀ x : ℝ, f x = f (-x))
  (mono_f : ∀ {x y : ℝ}, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h : f (m + 1) < f (3 * m - 1)) :
  m > 1 ∨ m < 0 :=
sorry

end range_of_m_l761_761530


namespace proof_OQ_perp_EF_iff_QE_eq_QF_l761_761766

-- Definitions from the problem statement
variable {ABC : Type} [Triangle ABC] (A B C : Point) (h_iso: AB = AC) -- isosceles triangle
variable (M : Point) (h_M : midpoint M B C) -- M is midpoint of BC
variable (O : Point) (h_OB : perpendicular OB AB) (h_O : O ∈ AM) -- O is on AM and OB ⊥ AB
variable (Q : Point) (h_Q : Q ∈ BC) -- Q is on BC different from B and C
variable (E : Point) (F : Point) (h_EF_collinear : collinear E Q F) -- E lies on AB, F lies on AC, E, Q, F distinct and collinear

-- The statement to prove
theorem proof_OQ_perp_EF_iff_QE_eq_QF :
  (perpendicular (OQ : Line) (EF : Line)) ↔ (distance Q E = distance Q F) :=
by
  sorry

end proof_OQ_perp_EF_iff_QE_eq_QF_l761_761766


namespace action_figures_more_than_books_l761_761198

variable (initialActionFigures : Nat) (newActionFigures : Nat) (books : Nat)

def totalActionFigures (initialActionFigures newActionFigures : Nat) : Nat :=
  initialActionFigures + newActionFigures

theorem action_figures_more_than_books :
  initialActionFigures = 5 → newActionFigures = 7 → books = 9 →
  totalActionFigures initialActionFigures newActionFigures - books = 3 :=
by
  intros h_initial h_new h_books
  rw [h_initial, h_new, h_books]
  sorry

end action_figures_more_than_books_l761_761198


namespace solve_x4_minus_16_eq_0_l761_761490

theorem solve_x4_minus_16_eq_0 {x : ℂ} : (x ^ 4 - 16 = 0) ↔ (x = 2 ∨ x = -2 ∨ x = 2 * complex.I ∨ x = -2 * complex.I) :=
by
  sorry

end solve_x4_minus_16_eq_0_l761_761490


namespace seats_in_hall_l761_761577

theorem seats_in_hall (S : ℝ) (h1 : 0.50 * S = 300) : S = 600 :=
by
  sorry

end seats_in_hall_l761_761577


namespace girls_more_than_boys_l761_761746

theorem girls_more_than_boys (total_students boys : ℕ) (h : total_students = 466) (b : boys = 127) (gt : total_students - boys > boys) :
  total_students - 2 * boys = 212 := by
  sorry

end girls_more_than_boys_l761_761746


namespace chelsea_victory_shots_l761_761131

theorem chelsea_victory_shots (k : ℕ) (n : ℕ) :
  (∃ n ≥ 49, ∀ x : ℕ, x < n → x = k + 10 * n + 5 * (60 - n) > k + 540) ∧ (k ≥ 300) :=
by
  sorry

end chelsea_victory_shots_l761_761131


namespace calculate_sum_of_inverses_l761_761218

noncomputable section

variables {p q z1 z2 z3 : ℂ}

-- Conditions
def is_root (a : ℂ) (p : ℂ[X]) := p.eval a = 0

def roots_cond : Prop := 
  is_root z1 (X^3 + C p * X + C q) ∧ 
  is_root z2 (X^3 + C p * X + C q) ∧ 
  is_root z3 (X^3 + C p * X + C q)

-- Main theorem
theorem calculate_sum_of_inverses (h : roots_cond) :
  (1 / z1^2) + (1 / z2^2) + (1 / z3^2) = (p^2) / (q^2) :=
sorry

end calculate_sum_of_inverses_l761_761218


namespace Vasya_numbers_l761_761378

theorem Vasya_numbers : ∃ (x y : ℝ), x + y = xy ∧ xy = x / y ∧ (x, y) = (1 / 2, -1) :=
by
  sorry

end Vasya_numbers_l761_761378


namespace binary_to_decimal_l761_761021

theorem binary_to_decimal :
  ∀ n : ℕ, n = 101 →
  ∑ i in Finset.range 3, (n / 10^i % 10) * 2^i = 5 :=
by
  sorry

end binary_to_decimal_l761_761021


namespace general_term_formula_l761_761591

-- Definitions:
def q : ℝ := 4
def sum_of_first_three_terms (a₁ : ℝ) : ℝ := a₁ + q * a₁ + q^2 * a₁

-- Conditions:
axiom sum_condition (a₁ : ℝ) : sum_of_first_three_terms a₁ = 21

-- Statement to Prove:
theorem general_term_formula (a₁ : ℝ) (n : ℕ) (h : sum_condition a₁) : 
  (a₁ = 1) → a_n = 4^(n-1) :=
sorry

end general_term_formula_l761_761591


namespace solve_system_of_equations_l761_761257

theorem solve_system_of_equations (x y : ℝ) : 
  (x + y = x^2 + 2 * x * y + y^2) ∧ (x - y = x^2 - 2 * x * y + y^2) ↔ 
  (x = 0 ∧ y = 0) ∨ 
  (x = 1/2 ∧ y = 1/2) ∨ 
  (x = 1/2 ∧ y = -1/2) ∨ 
  (x = 1 ∧ y = 0) :=
by
  sorry

end solve_system_of_equations_l761_761257


namespace imaginary_part_of_z_l761_761160

theorem imaginary_part_of_z (z : ℂ) (h : complex.I * z = (1 + complex.I) / 2) : z.im = -1/2 :=
sorry

end imaginary_part_of_z_l761_761160


namespace find_z_l761_761039

variable (a b : ℝ) (z : ℂ)
def z_def: ℂ := a + b * Complex.I
def z_conj_def: ℂ := a - b * Complex.I
def equation : Prop := 3 * z_def + 2 * Complex.I * z_conj_def = -5 + 4 * Complex.I

theorem find_z :
  equation a b → z_def = -7/13 + (22/13) * Complex.I :=
by
  sorry

end find_z_l761_761039


namespace vasya_numbers_l761_761367

theorem vasya_numbers (x y : ℚ) (h : x + y = xy ∧ xy = x / y) : x = 1 / 2 ∧ y = -1 :=
by {
  sorry
}

end vasya_numbers_l761_761367


namespace remainder_of_factorial_sum_mod_30_l761_761163

theorem remainder_of_factorial_sum_mod_30 :
  (Finset.sum (Finset.range 101) (λ n, Nat.factorial n)) % 30 = 3 :=
  sorry

end remainder_of_factorial_sum_mod_30_l761_761163


namespace domain_of_w_l761_761041

-- Define the function w(x)
def w (x : ℝ) : ℝ := real.sqrt (2 * (x - 1)) + real.sqrt (4 - 2 * x)

-- Statement: Prove that the domain of w(x) is [1, 2]
theorem domain_of_w : {x : ℝ | 0 ≤ 2 * (x - 1) ∧ 0 ≤ 4 - 2 * x} = set.Icc 1 2 := 
by
  sorry

end domain_of_w_l761_761041


namespace number_of_vertical_asymptotes_l761_761478

def has_vertical_asymptotes (f : ℚ → ℚ) (x : ℚ) : Prop :=
  ∃ ε > 0, ∀ δ > 0, ∃ x' ∈ set.Ioo (x - ε) (x + ε), abs (f x') > δ

theorem number_of_vertical_asymptotes (f : ℚ → ℚ)
  (num := λ x, x - 2)
  (denom := λ x, x^2 + 8 * x - 9)
  (f_def : ∀ x, f x = num x / denom x)
  : (∃ x1 x2 : ℚ, has_vertical_asymptotes f x1 ∧ has_vertical_asymptotes f x2 ∧ x1 ≠ x2) ∧ 
    (∀ x : ℚ, has_vertical_asymptotes f x → (x = 1 ∨ x = -9)) :=
by
  -- Placeholder proof
  sorry

end number_of_vertical_asymptotes_l761_761478


namespace mean_and_median_change_l761_761174

def original_scores := [35, 40, 42, 48, 30]
def corrected_scores := [35, 40, 47, 48, 30]

def mean (lst : List ℕ) : ℚ :=
  (lst.sum : ℚ) / (lst.length : ℚ)

def median (lst : List ℕ) : ℕ :=
  let sorted_lst := lst.qsort (· ≤ ·)
  sorted_lst.get! ((lst.length - 1) / 2)

theorem mean_and_median_change :
  mean corrected_scores = mean original_scores + 1 ∧ 
  median corrected_scores = median original_scores :=
by
  sorry

end mean_and_median_change_l761_761174


namespace measure_of_angle_y_l761_761827

def is_straight_angle (a : ℝ) := a = 180

theorem measure_of_angle_y (angle_ABC angle_ADB angle_BDA y : ℝ) 
  (h1 : angle_ABC = 117)
  (h2 : angle_ADB = 31)
  (h3 : angle_BDA = 28)
  (h4 : is_straight_angle (angle_ABC + (180 - angle_ABC)))
  : y = 86 := 
by 
  sorry

end measure_of_angle_y_l761_761827


namespace gcd_max_digits_l761_761826

theorem gcd_max_digits (a b : ℕ) (h_a : a < 10^7) (h_b : b < 10^7) (h_lcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) : Nat.gcd a b < 10^3 :=
by
  sorry

end gcd_max_digits_l761_761826


namespace radius_of_diameter_l761_761710

theorem radius_of_diameter (D : ℕ) (hD : D = 26) : ∃ R, R = D / 2 ∧ R = 13 := by
  use 13
  split
  { rw hD
    norm_num }
  { refl }

end radius_of_diameter_l761_761710


namespace enclosed_area_calculation_l761_761991

theorem enclosed_area_calculation :
  (∀ x : ℝ, (1/x + x^2)^3 = ∑ k in range 4, (choose 3 k) * (1/x)^(3-k) * (x^2)^k) →
  (∀ x : ℝ, ∃ a : ℝ, 3 = a) →
  (∫ x in 0..3, (3 * x - x^2) dx = 9/2) :=
by
  sorry

end enclosed_area_calculation_l761_761991


namespace percentage_increase_l761_761406

theorem percentage_increase (initial final : ℝ) (h_initial : initial = 200) (h_final : final = 250) :
  ((final - initial) / initial) * 100 = 25 := 
sorry

end percentage_increase_l761_761406


namespace czech_slovak_olympiad_problem_l761_761658

theorem czech_slovak_olympiad_problem (x y z : ℝ) :
  x^4 + y^2 + 4 = 5 * y * z ∧ 
  y^4 + z^2 + 4 = 5 * z * x ∧ 
  z^4 + x^2 + 4 = 5 * x * y ↔ 
  (x, y, z) = (sqrt 2, sqrt 2, sqrt 2) ∨ 
  (x, y, z) = (sqrt 2, sqrt 2, -sqrt 2) ∨ 
  (x, y, z) = (sqrt 2, -sqrt 2, sqrt 2) ∨ 
  (x, y, z) = (sqrt 2, -sqrt 2, -sqrt 2) ∨ 
  (x, y, z) = (-sqrt 2, sqrt 2, sqrt 2) ∨ 
  (x, y, z) = (-sqrt 2, sqrt 2, -sqrt 2) ∨ 
  (x, y, z) = (-sqrt 2, -sqrt 2, sqrt 2) ∨ 
  (x, y, z) = (-sqrt 2, -sqrt 2, -sqrt 2) := sorry

end czech_slovak_olympiad_problem_l761_761658


namespace green_face_probability_l761_761661

def probability_of_green_face (total_faces green_faces : Nat) : ℚ :=
  green_faces / total_faces

theorem green_face_probability :
  let total_faces := 10
  let green_faces := 3
  let blue_faces := 5
  let red_faces := 2
  probability_of_green_face total_faces green_faces = 3/10 :=
by
  sorry

end green_face_probability_l761_761661


namespace arithmetic_sequence_fifth_term_l761_761664

theorem arithmetic_sequence_fifth_term (x y : ℚ) (h₁ : 2x + 3y - (2x - 3y) = -6y)
  (h₂ : 2x - 3y - 6y = 2xy) (h₃ : 2x - 9y - 6y = 2x / y) : 
  2x - 21y = 63 / 10 := 
by
  sorry

end arithmetic_sequence_fifth_term_l761_761664


namespace largest_number_divisible_by_11_l761_761244

theorem largest_number_divisible_by_11 : 
    ∃ n: ℕ, (∀ d, d ∈ [1,2,3,4,5,6,7,8,9] → n.digits 10 = d ∧
    (n % 11 = 0 ∧ n = 987652413)) sorry

end largest_number_divisible_by_11_l761_761244


namespace total_travel_time_l761_761789

noncomputable def washingtonToIdahoDistance : ℕ := 640
noncomputable def idahoToNevadaDistance : ℕ := 550
noncomputable def washingtonToIdahoSpeed : ℕ := 80
noncomputable def idahoToNevadaSpeed : ℕ := 50

theorem total_travel_time :
  (washingtonToIdahoDistance / washingtonToIdahoSpeed) + (idahoToNevadaDistance / idahoToNevadaSpeed) = 19 :=
by
  sorry

end total_travel_time_l761_761789


namespace binary101_is_5_l761_761030

theorem binary101_is_5 : 
  let binary101 := [1, 0, 1] in
  let decimal := binary101[0] * 2^2 + binary101[1] * 2^1 + binary101[2] * 2^0 in
  decimal = 5 :=
by
  let binary101 := [1, 0, 1]
  let decimal := binary101[0] * 2^2 + binary101[1] * 2^1 + binary101[2] * 2^0
  show decimal = 5
  sorry

end binary101_is_5_l761_761030


namespace coby_travel_time_l761_761796

def travel_time (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

theorem coby_travel_time :
  let wash_to_idaho_distance := 640
  let idaho_to_nevada_distance := 550
  let wash_to_idaho_speed := 80
  let idaho_to_nevada_speed := 50
  travel_time wash_to_idaho_distance wash_to_idaho_speed + travel_time idaho_to_nevada_distance idaho_to_nevada_speed = 19 := by
  sorry

end coby_travel_time_l761_761796


namespace total_fruits_on_display_l761_761691

-- Declare the variables and conditions as hypotheses
variables (apples oranges bananas : ℕ)
variables (h1 : apples = 2 * oranges)
variables (h2 : oranges = 2 * bananas)
variables (h3 : bananas = 5)

-- Define what we want to prove
theorem total_fruits_on_display : apples + oranges + bananas = 35 :=
by sorry

end total_fruits_on_display_l761_761691


namespace find_k_l761_761553

variable (k : ℝ)
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, 2)

theorem find_k 
  (h : (k * a.1 - b.1, k * a.2 - b.2) = (k - 1, k - 2)) 
  (perp_cond : (k * a.1 - b.1, k * a.2 - b.2).fst * (b.1 + a.1) + (k * a.1 - b.1, k * a.2 - b.2).snd * (b.2 + a.2) = 0) :
  k = 8 / 5 :=
sorry

end find_k_l761_761553


namespace bin101_to_decimal_l761_761035

-- Define the binary representation of 101 (base 2)
def bin101 : ℕ := 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- State the theorem that asserts the decimal value of 101 (base 2) is 5
theorem bin101_to_decimal : bin101 = 5 := by
  sorry

end bin101_to_decimal_l761_761035


namespace tens_digit_of_8_pow_103_l761_761387

theorem tens_digit_of_8_pow_103 : 
  (let sequence := [08, 64, 12, 96, 68, 44, 52, 16, 28, 24, 92, 36, 88, 04, 32, 56, 48, 84, 72, 76] in
  sequence[(103 % 20) - 1] / 10) % 10 = 1 :=
by
  sorry

end tens_digit_of_8_pow_103_l761_761387


namespace total_bottles_and_fruits_l761_761415

theorem total_bottles_and_fruits :
  let regular_soda := 130
  let diet_soda := 88
  let sparkling_water := 65
  let orange_juice := 47
  let cranberry_juice := 27
  let apples := 102
  let oranges := 88
  let bananas := 74
  let pears := 45
  total_bottles_and_fruits = (regular_soda + diet_soda + sparkling_water + orange_juice + cranberry_juice) + (apples + oranges + bananas + pears) -> total_bottles_and_fruits = 666 := 
  by sorry

end total_bottles_and_fruits_l761_761415


namespace journey_time_l761_761313

-- Defining constants and conditions
def speed_boat_still_water : ℝ := 12
def speed_stream_outward : ℝ := 2
def distance_to_place : ℝ := 180
def speed_stream_return : ℝ := 4

-- Calculate the effective speeds
def speed_downstream := speed_boat_still_water + speed_stream_outward
def speed_upstream := speed_boat_still_water - speed_stream_return

-- Calculate times
def time_downstream := distance_to_place / speed_downstream
def time_upstream := distance_to_place / speed_upstream

-- Calculate total time
def total_time := time_downstream + time_upstream

theorem journey_time : total_time ≈ 35.36 := by
  sorry

end journey_time_l761_761313


namespace scientific_notation_correct_l761_761962

-- The given number
def given_number : ℕ := 9000000000

-- The correct answer in scientific notation
def correct_sci_not : ℕ := 9 * (10 ^ 9)

-- The theorem to prove
theorem scientific_notation_correct :
  given_number = correct_sci_not :=
by
  sorry

end scientific_notation_correct_l761_761962


namespace total_cost_of_books_and_pencils_l761_761921

variable (a b : ℕ)

theorem total_cost_of_books_and_pencils (a b : ℕ) : 5 * a + 2 * b = 5 * a + 2 * b := by
  sorry

end total_cost_of_books_and_pencils_l761_761921


namespace binary101_is_5_l761_761031

theorem binary101_is_5 : 
  let binary101 := [1, 0, 1] in
  let decimal := binary101[0] * 2^2 + binary101[1] * 2^1 + binary101[2] * 2^0 in
  decimal = 5 :=
by
  let binary101 := [1, 0, 1]
  let decimal := binary101[0] * 2^2 + binary101[1] * 2^1 + binary101[2] * 2^0
  show decimal = 5
  sorry

end binary101_is_5_l761_761031


namespace sum_prime_numbers_in_interval_l761_761854

/-- Define the sequence a_n. -/
def a (n : ℕ) : Real := Real.log (n + 2) / Real.log (n + 1)

/-- Define the condition for n to be a "prime number". -/
def is_prime_number (n : ℕ) : Prop := (Real.log (n + 2) / Real.log 2).den = 1

/-- Define the interval of interest. -/
def in_interval (n : ℕ) : Prop := 1 < n ∧ n ≤ 2016

/-- The sum of all "prime numbers" in the interval (1, 2016] is 2026. -/
theorem sum_prime_numbers_in_interval : 
  (∑ n in Finset.filter (λ n, is_prime_number n) (Finset.Icc 2 2016).val) = 2026 :=
  sorry

end sum_prime_numbers_in_interval_l761_761854


namespace distance_between_stations_l761_761397

/-- Two trains start at the same time from two stations and proceed towards each other. 
    The first train travels at 20 km/hr and the second train travels at 25 km/hr. 
    When they meet, the second train has traveled 60 km more than the first train. -/
theorem distance_between_stations
    (t : ℝ) -- The time in hours when they meet
    (x : ℝ) -- The distance traveled by the slower train
    (d1 d2 : ℝ) -- Distances traveled by the two trains respectively
    (h1 : 20 * t = x)
    (h2 : 25 * t = x + 60) :
  d1 + d2 = 540 :=
by
  sorry

end distance_between_stations_l761_761397


namespace power_mod_1000_l761_761648

theorem power_mod_1000 (N : ℤ) (h : Int.gcd N 10 = 1) : (N ^ 101 ≡ N [ZMOD 1000]) :=
  sorry

end power_mod_1000_l761_761648


namespace right_angles_l761_761401

open Real

-- Define the ellipse equation and constraints on a and b
structure Ellipse (a b : ℝ) (ha : a > b) (hb : b > 0) :=
  (x y : ℝ)
  (on_ellipse : (x^2 / a^2) + (y^2 / b^2) = 1)

-- Define the foci
def Foci (a b : ℝ) (ellipse : Ellipse a b) : ℝ := 
  sqrt (a^2 - b^2)

-- Define point P on the ellipse with parametric representation
structure PointOnEllipse (a b α : ℝ) (ha : a > b) (hb : b > 0) extends Ellipse a b ha hb :=
  (cos_alpha_sin_alpha : (x = a * cos α) ∧ (y = b * sin α))

-- Define the tangent line passing through point P
def TangentLineAtP (a b α : ℝ) (P : PointOnEllipse a b α) : Prop :=
  ∀ x y, (x * cos α / a) + (y * sin α / b) = 1

-- Assume P is the point (a * cos_alpha, b * sin_alpha)
def P := (a * cos α, b * sin α)

-- The coordinates of intersections (dummy definitions to simulate proof)
def C := (0, 0)
def D := (0, 0)

-- Problem statement in Lean
theorem right_angles (a b : ℝ) (ha : a > b) (hb : b > 0)
  (P : PointOnEllipse a b α ha hb) : 
  ∃ C D : ℝ × ℝ, -- Points where tangent intersections occur
  ∠(C, P, D) = 90 ∧ ∠(C, -P, D) = 90 := 
sorry

end right_angles_l761_761401


namespace omitted_decimal_sum_is_integer_l761_761430

def numbers : List ℝ := [1.05, 1.15, 1.25, 1.4, 1.5, 1.6, 1.75, 1.85, 1.95]

theorem omitted_decimal_sum_is_integer :
  1.05 + 1.15 + 1.25 + 1.4 + (15 : ℝ) + 1.6 + 1.75 + 1.85 + 1.95 = 27 :=
by sorry

end omitted_decimal_sum_is_integer_l761_761430


namespace part1_part2_part3_l761_761879

section Problem1
variable (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ)
variable (n : ℕ)

axiom a_def : ∀ n, a n = 2^(n-1)
axiom b_arith : ∀ n, b n = b 1 + (n-1) * (b 2 - b 1)
axiom cn_eq_n : ∀ n, c n = n
axiom union_set : ∀ x, c x = a x ∨ c x = b x

theorem part1 : (∀ n, c n == n) → (∀ n, b n == n) := sorry
end Problem1

section Problem2
variable (b : ℕ → ℝ) (d : ℕ → ℝ) (n : ℕ) (λ : ℝ)
axiom bn_def : ∀ n, b n = Real.sqrt 2 * n
axiom dn_def : ∀ λ n, d n = 3^n + (-1)^n * ( (b n / n)^(2 * n) * λ)
axiom d_increasing : ∀ (n : ℕ), d (n + 1) > d n

theorem part2 : ∀ λ, (∀ n, b n = Real.sqrt 2 * n) → (∀ (n : ℕ), d (n + 1) > d n) → (-1 < λ ∧ λ < 3 / 2) := sorry
end Problem2

section Problem3
variable (a : ℕ → ℕ) (b : ℕ → ℚ) (c : ℕ → ℕ)
variable (A B : Set ℕ)

axiom a_def : ∀ n, a n = 2^(n-1)
axiom b_arith : ∀ n, b n = Real.sqrt 2 * n
axiom c_geometric : ∀ n, c n = 1 * (2^(n-1))
axiom c1_eq1 : c 1 = 1
axiom c9_eq8 : c 9 = 8
axiom disjoint : A ∩ B = ∅

theorem part3 : (A ∩ B = ∅) → (c 1 = 1 ∧ c 9 = 8) → (∀ n, b n = Real.sqrt 2 * n) := sorry
end Problem3

end part1_part2_part3_l761_761879


namespace oxygen_atoms_l761_761740

theorem oxygen_atoms (x : ℤ) (h : 27 + 16 * x + 3 = 78) : x = 3 := 
by 
  sorry

end oxygen_atoms_l761_761740


namespace largest_prime_divisor_360_231_l761_761709

theorem largest_prime_divisor_360_231 :
  ∃ p : ℕ, (prime p) ∧ (p ∣ 360) ∧ (p ∣ 231) ∧ (∀ q : ℕ, (prime q) ∧ (q ∣ 360) ∧ (q ∣ 231) → q ≤ p) :=
sorry 

end largest_prime_divisor_360_231_l761_761709


namespace remainder_of_xyz_l761_761563

theorem remainder_of_xyz {x y z : ℕ} (hx: x < 9) (hy: y < 9) (hz: z < 9)
  (h1: (x + 3*y + 2*z) % 9 = 0)
  (h2: (2*x + 2*y + z) % 9 = 7)
  (h3: (x + 2*y + 3*z) % 9 = 5) :
  (x * y * z) % 9 = 5 :=
sorry

end remainder_of_xyz_l761_761563


namespace positive_difference_time_l761_761702

def time_difference_tom_linda : ℝ :=
  let v_linda := 3           -- Linda's speed in miles per hour
  let v_tom := 8             -- Tom's speed in miles per hour
  let t_start_tom := 1       -- Time (in hours) after which Tom starts
  let d_linda_first_hour := v_linda * t_start_tom
  let time_tom_half_distance := (d_linda_first_hour / 2) / v_tom
  let time_tom_twice_distance := (d_linda_first_hour * 2) / v_tom
  let time_difference_hours := time_tom_twice_distance - time_tom_half_distance
  time_difference_hours * 60 -- Convert hours to minutes

theorem positive_difference_time : time_difference_tom_linda = 33.75 :=
  by
  -- Skipping the proof
  sorry

end positive_difference_time_l761_761702


namespace AF_over_AC_l761_761947

variables {A B C D E F G : Type*}
variables [Rect A B C D] [Rect A E F G]
variables (F_diagonal_AC : F ∈ AC)
variables (area_FRS_eq : ∀ (area_TRIA : ℝ), area area_TRIA = (1 / 18) * area (Rect A E F G))

noncomputable def AF_AC_eq (AF AC : ℝ) : Prop :=
  AF / AC = 3 / 5

theorem AF_over_AC (AF_AC_eq : AF_AC_eq AF AC) : AF / AC = 3 / 5 := by
  intro h₁ h₂ h₃ h₄
  have h : AF / AC = 3 / 5 := sorry
  exact h

end AF_over_AC_l761_761947


namespace triangle_area_from_subareas_l761_761324

-- Definitions related to the problem
variable (t1 t2 t3 : ℝ) (ht1 : t1 > 0) (ht2 : t2 > 0) (ht3 : t3 > 0)

-- The proof statement where we need to confirm that triangle area T is as calculated
theorem triangle_area_from_subareas (t1 t2 t3 : ℝ) (ht1 : t1 > 0) (ht2 : t2 > 0) (ht3 : t3 > 0) :
  ∃ T : ℝ, T = (Real.sqrt t1 + Real.sqrt t2 + Real.sqrt t3) ^ 2 := 
begin
  use (Real.sqrt t1 + Real.sqrt t2 + Real.sqrt t3) ^ 2,
  dsimp,
  ring,
end

end triangle_area_from_subareas_l761_761324


namespace cut_scene_length_proof_l761_761425

noncomputable def original_length : ℕ := 60
noncomputable def final_length : ℕ := 57
noncomputable def cut_scene_length := original_length - final_length

theorem cut_scene_length_proof : cut_scene_length = 3 := by
  sorry

end cut_scene_length_proof_l761_761425


namespace negation_statement_l761_761711

open Set

variable {S : Set ℝ}

theorem negation_statement (h : ∀ x ∈ S, 3 * x - 5 > 0) : ∃ x ∈ S, 3 * x - 5 ≤ 0 :=
sorry

end negation_statement_l761_761711


namespace inequality_of_weighted_squares_equality_conditions_of_weighted_squares_l761_761475

theorem inequality_of_weighted_squares
  (x y a b : ℝ)
  (h_sum : a + b = 1)
  (h_nonneg_a : a ≥ 0)
  (h_nonneg_b : b ≥ 0) :
  (a * x + b * y)^2 ≤ a * x^2 + b * y^2 :=
by sorry

theorem equality_conditions_of_weighted_squares
  (x y a b : ℝ)
  (h_sum : a + b = 1)
  (h_nonneg_a : a ≥ 0)
  (h_nonneg_b : b ≥ 0) :
  (a * x + b * y)^2 = a * x^2 + b * y^2
  ↔ (a = 0 ∨ b = 0 ∨ x = y) :=
by sorry

end inequality_of_weighted_squares_equality_conditions_of_weighted_squares_l761_761475


namespace sum_of_a_theta_eq_l761_761207

noncomputable def m : ℕ := sorry -- m is a positive odd integer greater than 1
noncomputable def n : ℕ := 2 * m
noncomputable def θ : ℂ := Complex.exp (2 * Real.pi * Complex.I / n)

def a (i : ℕ) : ℤ :=
  if even i then 1 else -1

theorem sum_of_a_theta_eq :
  m > 1 ∧ odd m →
  ∑ i in Finset.range (m-2), a i * θ^i = 1 / (1 - θ) :=
by
  sorry

end sum_of_a_theta_eq_l761_761207


namespace calculate_f_f_2_l761_761086

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 3 * x ^ 2 - 4
else if x = 0 then 2
else -1

theorem calculate_f_f_2 : f (f 2) = 188 :=
by
  sorry

end calculate_f_f_2_l761_761086


namespace exists_epsilon_divisible_by_1001_l761_761227

theorem exists_epsilon_divisible_by_1001 (a : Fin 10 → ℤ) :
  ∃ (ε : Fin 10 → ℤ), (∀ i, ε i ∈ {0, 1, -1}) ∧ 1001 ∣ ∑ i, ε i * a i := 
by 
  sorry

end exists_epsilon_divisible_by_1001_l761_761227


namespace angle_between_vectors_eq_2pi_over_3_l761_761862

open Real
open InnerProductSpace

theorem angle_between_vectors_eq_2pi_over_3 (a b : ℝ^3) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : ∥a∥ = ∥b∥ ∧ ∥a + b∥ = ∥a∥) :
  angle a b = 2 * π / 3 :=
sorry

end angle_between_vectors_eq_2pi_over_3_l761_761862


namespace min_value_of_f_l761_761873

noncomputable def f (x a : ℝ) := Real.exp (x - a) - Real.log (x + a) - 1

theorem min_value_of_f (a : ℝ) : 
  (0 < a) → (∃ x : ℝ, f x a = 0) ↔ a = 1 / 2 :=
by
  sorry

end min_value_of_f_l761_761873


namespace carla_marbles_l761_761786

theorem carla_marbles (before now bought : ℝ) (h_before : before = 187.0) (h_now : now = 321) : bought = 134 :=
by
  sorry

end carla_marbles_l761_761786


namespace distance_traveled_l761_761199

theorem distance_traveled (speed1 speed2 hours1 hours2 : ℝ)
  (h1 : speed1 = 45) (h2 : hours1 = 2) (h3 : speed2 = 50) (h4 : hours2 = 3) :
  speed1 * hours1 + speed2 * hours2 = 240 := by
  sorry

end distance_traveled_l761_761199


namespace paul_crayons_l761_761641

def initial_crayons : ℝ := 479.0
def additional_crayons : ℝ := 134.0
def total_crayons : ℝ := initial_crayons + additional_crayons

theorem paul_crayons : total_crayons = 613.0 :=
by
  sorry

end paul_crayons_l761_761641


namespace vasya_numbers_l761_761357

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 := 
by 
  sorry

end vasya_numbers_l761_761357


namespace hyperbola_eccentricity_l761_761491

theorem hyperbola_eccentricity (a b e : ℝ) (h : b = 3 * a) (h_hyperbola : e = Real.sqrt (1 + (b^2 / a^2))) : e = Real.sqrt 10 :=
by
  -- Since b = 3 * a:
  have hb : b^2 = (3 * a)^2 := by sorry
  -- Therefore:
  rw h at hb
  -- Substituting into the formula:
  have h1 : e = Real.sqrt (1 + ((3 * a)^2 / a^2)) := by sorry
  -- Simplifying we get:
  rw [←mul_self_div_self a.ne_zero, mul_self_three] at h1
  -- Therefore:
  exact h1

end hyperbola_eccentricity_l761_761491


namespace james_remaining_money_after_tickets_l761_761602

def parking_tickets_cost (ticket1 ticket2 ticket3 : ℕ) : ℕ :=
  ticket1 + ticket2 + ticket3

def roommate_share (total_cost : ℕ) : ℕ :=
  total_cost / 2

theorem james_remaining_money_after_tickets
  (initial_money : ℕ)
  (ticket1 : ℕ)
  (ticket2 : ℕ)
  (ticket3 : ℕ)
  (ticket_cost : ticket1 = 150)
  (ticket_cost2 : ticket2 = 150)
  (ticket_cost3 : ticket3 = ticket1 / 3) :
  let total_cost := parking_tickets_cost ticket1 ticket2 ticket3 in
  let james_share := roommate_share total_cost in
  (initial_money - james_share) = 325 :=
by
  sorry

end james_remaining_money_after_tickets_l761_761602


namespace log_inequality_l761_761620

theorem log_inequality
  (a : ℝ := Real.log 4 / Real.log 5)
  (b : ℝ := (Real.log 3 / Real.log 5)^2)
  (c : ℝ := Real.log 5 / Real.log 4) :
  b < a ∧ a < c :=
by
  sorry

end log_inequality_l761_761620


namespace range_of_squared_sum_l761_761847

theorem range_of_squared_sum (x y : ℝ) (h : x^2 + 1 / y^2 = 2) : ∃ z, z = x^2 + y^2 ∧ z ≥ 1 / 2 :=
by
  sorry

end range_of_squared_sum_l761_761847


namespace range_of_f_l761_761288

noncomputable def f (x : ℝ) : ℝ := x^4 + 6 * x^2 + 9

theorem range_of_f : set.Icc (9 : ℝ) ∞ = {y : ℝ | ∃ x : ℝ, 0 ≤ x ∧ f x = y} :=
by {
    sorry
}

end range_of_f_l761_761288


namespace sum_abs_roots_l761_761833

noncomputable def polynomial := (λ x:ℂ, x^4 - 4 * x^3 + 8 * x^2 - 16 * x + 15)

theorem sum_abs_roots :
    (roots polynomial).sum (λ r, abs r) = 2 * real.sqrt 3 + 2 * real.sqrt 5 :=
  sorry

end sum_abs_roots_l761_761833


namespace range_of_m_l761_761084

theorem range_of_m (m : ℝ) (P Q : set ℝ) 
  (hP: P = {x : ℝ | x^2 - 4 * x - 12 ≤ 0})
  (hQ: Q = {x : ℝ | |x - m| ≤ m^2}) 
  (hneq : ∀ x, x ∉ P → x ∈ Q) :
  (m ∈ (-∞, -3] ∪ (2, ∞)) :=
sorry

end range_of_m_l761_761084


namespace max_n_arithmetic_sequences_l761_761775

theorem max_n_arithmetic_sequences (a b : ℕ → ℤ) 
  (ha : ∀ n, a n = 1 + (n - 1) * 1)  -- Assuming x = 1 for simplicity, as per solution x = y = 1
  (hb : ∀ n, b n = 1 + (n - 1) * 1)  -- Assuming y = 1
  (a1 : a 1 = 1)
  (b1 : b 1 = 1)
  (a2_leq_b2 : a 2 ≤ b 2)
  (hn : ∃ n, a n * b n = 1764) :
  ∃ n, n = 44 ∧ a n * b n = 1764 :=
by
  sorry

end max_n_arithmetic_sequences_l761_761775


namespace combined_degrees_l761_761261

-- Definitions based on conditions
def summer_degrees : ℕ := 150
def jolly_degrees (summer_degrees : ℕ) : ℕ := summer_degrees - 5

-- Theorem stating the combined degrees
theorem combined_degrees : summer_degrees + jolly_degrees summer_degrees = 295 :=
by
  sorry

end combined_degrees_l761_761261


namespace base_nine_to_base_ten_conversion_l761_761701

theorem base_nine_to_base_ten_conversion : 
  (2 * 9^3 + 8 * 9^2 + 4 * 9^1 + 7 * 9^0 = 2149) := 
by 
  sorry

end base_nine_to_base_ten_conversion_l761_761701


namespace g_is_even_l761_761193

def g (x : ℝ) : ℝ := 4 / (3 * x^8 - 7)

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  sorry

end g_is_even_l761_761193


namespace total_travel_time_l761_761794

-- Define the necessary distances and speeds
def distance_Washington_to_Idaho : ℝ := 640
def speed_Washington_to_Idaho : ℝ := 80
def distance_Idaho_to_Nevada : ℝ := 550
def speed_Idaho_to_Nevada : ℝ := 50

-- Definitions for time calculations
def time_Washington_to_Idaho : ℝ := distance_Washington_to_Idaho / speed_Washington_to_Idaho
def time_Idaho_to_Nevada : ℝ := distance_Idaho_to_Nevada / speed_Idaho_to_Nevada

-- Problem statement to prove
theorem total_travel_time : time_Washington_to_Idaho + time_Idaho_to_Nevada = 19 := 
by
  sorry

end total_travel_time_l761_761794


namespace couscous_problem_l761_761408

def total_couscous (S1 S2 S3 : ℕ) : ℕ :=
  S1 + S2 + S3

def couscous_per_dish (total : ℕ) (dishes : ℕ) : ℕ :=
  total / dishes

theorem couscous_problem 
  (S1 S2 S3 : ℕ) (dishes : ℕ) 
  (h1 : S1 = 7) (h2 : S2 = 13) (h3 : S3 = 45) (h4 : dishes = 13) :
  couscous_per_dish (total_couscous S1 S2 S3) dishes = 5 := by  
  sorry

end couscous_problem_l761_761408


namespace order_of_values_l761_761501

noncomputable def a : ℝ := 21.2
noncomputable def b : ℝ := Real.sqrt 450 - 0.8
noncomputable def c : ℝ := 2 * Real.logb 5 2

theorem order_of_values : c < b ∧ b < a := by 
  sorry

end order_of_values_l761_761501


namespace angle_A_is_pi_div_3_min_area_is_sqrt3_div_3_l761_761175

-- Definition of the problem
variable (ABC : Type) [Triangle ABC] (a b c : ℝ) (A B C : ℝ)
  (hacute : IsAcuteTriangle ABC A B C)
  (hsides : SidesOppositeAngles ABC a b c A B C)
  (heqn : 4 * (Real.sin ((B + C)/2))^2 - Real.cos (2*A) = 7/2)
  (halt : AltitudeOnSide 'BC' 1)

-- Statements to prove
theorem angle_A_is_pi_div_3 : A = π / 3 := 
sorry

theorem min_area_is_sqrt3_div_3 : 
  let area := 1 / 2 * b * c * Real.sin A 
  in area ≥ sqrt 3 / 3 :=
sorry

end angle_A_is_pi_div_3_min_area_is_sqrt3_div_3_l761_761175


namespace john_longest_continuous_run_distance_l761_761200

structure RunnerConditions where
  initial_duration : ℝ -- in hours
  duration_increase_percent : ℝ -- percentage
  initial_speed : ℝ -- in mph
  speed_increase : ℝ -- in mph
  elevation_gain : ℝ -- in feet
  speed_decrease_percent : ℝ -- percentage
  elevation_gain_factor_per_thousand_feet : ℝ -- percentage per 1000 feet

constant john_conditions : RunnerConditions :=
{ initial_duration := 8,
  duration_increase_percent := 75,
  initial_speed := 8,
  speed_increase := 4,
  elevation_gain := 5500,
  speed_decrease_percent := 10,
  elevation_gain_factor_per_thousand_feet := 25 }

noncomputable def calculate_distance (cond : RunnerConditions) : ℝ :=
  let increased_duration := cond.initial_duration * (1 + cond.duration_increase_percent / 100)
  let increased_speed := cond.initial_speed + cond.speed_increase
  let actual_speed := increased_speed * (1 - cond.speed_decrease_percent / 100)
  let distance_flat := actual_speed * increased_duration
  let elevation_gain_factor := (cond.elevation_gain / 1000) * (1 + cond.elevation_gain_factor_per_thousand_feet / 100)
  distance_flat * elevation_gain_factor

theorem john_longest_continuous_run_distance :
  calculate_distance john_conditions = 207.9 :=
by
  sorry

end john_longest_continuous_run_distance_l761_761200


namespace sin_double_angle_value_l761_761103

theorem sin_double_angle_value (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2)
  (h : 2 * cos (2 * α) = cos (α - π / 4)) : sin (2 * α) = 7 / 8 :=
by 
  sorry

end sin_double_angle_value_l761_761103


namespace Vasya_numbers_l761_761349

theorem Vasya_numbers : 
  ∃ x y : ℚ, (x + y = x * y) ∧ (x * y = x / y) ∧ x = 1 / 2 ∧ y = -1 :=
by 
  use (1 / 2)
  use (-1)
  split
  case left {
    calc 
      (1 / 2) + (-1) = -1 / 2 : by norm_num,
      -1 / 2 = ((1 / 2) * (-1)) : by norm_num,
      (1 / 2) + (-1) = (1 / 2) * (-1) : by norm_num,
  },
  case right {
    calc 
      (1 / 2) * (-1) = -1 / 2 : by norm_num,
      -1 / 2 = (1 / 2) / (-1) : by norm_num,
      (1 / 2) * (-1) = (1 / 2) / (-1) : by norm_num,
  }

end Vasya_numbers_l761_761349


namespace mass_percentage_O_in_CaOH2_is_approx_43_19_l761_761066

noncomputable def calcium_hydroxide_mass_percentage_of_oxygen : ℝ :=
let molar_mass_Ca := 40.08
let molar_mass_O := 16.00
let molar_mass_H := 1.01
let molar_mass_CaOH2 := (1 * molar_mass_Ca) + (2 * molar_mass_O) + (2 * molar_mass_H)
let mass_percentage_O := (2 * molar_mass_O / molar_mass_CaOH2) * 100
in mass_percentage_O

theorem mass_percentage_O_in_CaOH2_is_approx_43_19 :
  abs (calcium_hydroxide_mass_percentage_of_oxygen - 43.19) < 1e-2 :=
sorry

end mass_percentage_O_in_CaOH2_is_approx_43_19_l761_761066


namespace matrix_multiplication_correct_l761_761466

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 1], ![4, -2]]
def matrix2 : Matrix (Fin 2) (Fin 2) ℤ := ![![7, -3], ![2, 2]]
def result_matrix : Matrix (Fin 2) (Fin 2) ℤ := ![![23, -7], ![24, -16]]

theorem matrix_multiplication_correct :
  matrix1.mul matrix2 = result_matrix :=
by
  sorry

end matrix_multiplication_correct_l761_761466


namespace vasya_numbers_l761_761341

theorem vasya_numbers :
  ∃ x y : ℝ, x + y = x * y ∧ x + y = x / y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l761_761341


namespace part1_harmonious_part2_t_range_l761_761544

-- Part 1
def harmonious_fun (f₁ f₂ h: ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, h(x) = a * f₁(x) + b * f₂(x)

theorem part1_harmonious 
  (f₁ f₂ h : ℝ → ℝ)
  (hf₁ : ∀ x, f₁ x = x - 1)
  (hf₂ : ∀ x, f₂ x = 3 * x + 1)
  (hh : ∀ x, h x = 2 * x + 2) :
  harmonious_fun f₁ f₂ h :=
sorry

-- Part 2
def solution_exists (h : ℝ → ℝ) (t : ℝ) : Prop :=
  ∃ x ∈ set.Icc (3 : ℝ) 9, h(9 * x) + t * h(3 * x) = 0

theorem part2_t_range
  (f₁ f₂ h : ℝ → ℝ)
  (hf₁ : ∀ x, f₁ x = real.log x / real.log 3)
  (hf₂: ∀ x, f₂ x = real.log x / real.log (1 / 3))
  (hf : ∀ x, h x = 2 * f₁ x + f₂ x) :
  set.Icc (-3 / 2) (-4 / 3) = {t : ℝ | solution_exists h t} :=
sorry

end part1_harmonious_part2_t_range_l761_761544


namespace expectation_of_X_variance_of_3X_plus_2_l761_761121

open ProbabilityTheory

namespace Proof

def X : Distribution ℝ := binom 4 (1/3)

theorem expectation_of_X :
  E[X] = 4 * (1/3) := by
  sorry

theorem variance_of_3X_plus_2 :
  let D (X : Distribution ℝ) := Var[X]
  D[3 * X + 2] = 9 * D[X] := by
  sorry

end Proof

end expectation_of_X_variance_of_3X_plus_2_l761_761121


namespace new_sphere_radius_l761_761776

theorem new_sphere_radius (R r h: ℝ) (pi: ℝ) 
  (hR: R = 20) 
  (hr: r = 12) 
  (hh: h = 2*R): 
  let original_volume := (4/3) * pi * R^3 in
  let cylinder_volume := pi * r^2 * h in
  let remaining_volume := original_volume - cylinder_volume in
  ∃ (new_r: ℝ), (4/3) * pi * new_r^3 = remaining_volume := 
by
  sorry

end new_sphere_radius_l761_761776


namespace tetrahedron_edge_length_l761_761072

-- Define the conditions for the problem
def radius := 2
def side_length := 2 * radius
def diagonal_length := side_length * Real.sqrt 2
def height := 2 * radius

-- Calculate the edge length of the tetrahedron
noncomputable def edge_length : ℝ :=
  Real.sqrt (diagonal_length^2 + height^2)

-- Prove that the edge length is equal to 2 * sqrt(5)
theorem tetrahedron_edge_length :
  edge_length = 2 * Real.sqrt 5 :=
by
  rw [edge_length, Real.sqrt_eq_rpow, Real.sqrt_eq_rpow, Real.rpow_two, ← Real.sqrt_mul, Real.sqrt_eq_rpow, Real.rpow_add, Real.two_mul, Real.pow_four, Real.mul_rpow, Real.mul_self_sqrt]
  sorry

end tetrahedron_edge_length_l761_761072


namespace cos_2pi_minus_alpha_l761_761104

theorem cos_2pi_minus_alpha (alpha : ℝ) (h1 : α ∈ set.Ioo (π / 2) (3 * π / 2))
  (h2 : real.tan α = -12 / 5) : real.cos (2 * π - α) = -5 / 13 :=
sorry

end cos_2pi_minus_alpha_l761_761104


namespace combined_degrees_l761_761264

variable (Summer_deg Jolly_deg : ℕ)

def Summer_has_150_degrees := Summer_deg = 150

def Summer_has_5_more_degrees_than_Jolly := Summer_deg = Jolly_deg + 5

theorem combined_degrees (h1 : Summer_has_150_degrees Summer_deg) (h2 : Summer_has_5_more_degrees_than_Jolly Summer_deg Jolly_deg) :
  Summer_deg + Jolly_deg = 295 :=
by
  sorry

end combined_degrees_l761_761264


namespace factorial_sum_remainder_mod_30_l761_761165

theorem factorial_sum_remainder_mod_30 :
  (∑ n in Finset.range 101, Nat.factorial n) % 30 = 3 :=
by
  sorry

end factorial_sum_remainder_mod_30_l761_761165


namespace find_center_number_l761_761442

def is_adjacency_valid (grid : List (List ℕ)) : Prop :=
  ∀ i j, (i > 0 → (grid[i][j] = grid[i-1][j] + 1 ∨ grid[i][j] = grid[i-1][j] - 1)) ∧
         (i < 2 → (grid[i][j] = grid[i+1][j] + 1 ∨ grid[i][j] = grid[i+1][j] - 1)) ∧
         (j > 0 → (grid[i][j] = grid[i][j-1] + 1 ∨ grid[i][j] = grid[i][j-1] - 1)) ∧
         (j < 2 → (grid[i][j] = grid[i][j+1] + 1 ∨ grid[i][j] = grid[i][j+1] - 1))

def sum_of_corners (grid : List (List ℕ)) : ℕ :=
  grid[0][0] + grid[0][2] + grid[2][0] + grid[2][2]

def product_of_diagonal_corners (grid : List (List ℕ)) : ℕ :=
  grid[0][0] * grid[2][2]

theorem find_center_number (grid : List (List ℕ)) :
  is_adjacency_valid grid →
  sum_of_corners grid = 20 →
  product_of_diagonal_corners grid = 9 →
  grid[1][1] = 5 :=
by
  sorry

end find_center_number_l761_761442


namespace range_of_a_l761_761108

noncomputable def f (a x : ℝ) : ℝ := a / x - 1 + Real.log x

theorem range_of_a {a : ℝ} (h : ∃ x : ℝ, x > 0 ∧ f a x ≤ 0) : a ≤ 1 :=
by
  cases' h with x hx
  let g := λ x : ℝ, x - x * Real.log x
  have : ∀ x > 0, f a x ≤ 0 → a ≤ g x := 
    by 
      intro x x_pos hx_le
      have := calc 
        a / x - 1 + Real.log x ≤ 0   : hx_le
        a ≤ x - x * Real.log x       : by linarith
      exact this
      
  have g_max : ∀ x : ℝ, 0 < x → g x ≤ 1 :=
    by 
      intro x hx_pos 
      have : (λ x, by simp [g, ← Real.exp_le_exp_iff] : ∀ (x > 0), -Real.log x ≥ 0 → x ≤ 1)
        by 
          intro x hx 
          linarith

  exact le_trans (this x hx.1 hx.2) (g_max x hx.1)
  
sorry

end range_of_a_l761_761108


namespace arrange_programs_l761_761075

/-- There are 36 different arrangements of the program list {A, B, C, D, E} 
    such that programs A and B are not adjacent, and the last program is either A or B. -/
theorem arrange_programs :
  let programs := ["A", "B", "C", "D", "E"]
  in ∃ (arrangements : List (List String)),
      (∀ arr ∈ arrangements,
        (arr.length = 5 ∧
         (arr.getLast? = some "A" ∨ arr.getLast? = some "B") ∧
         ∀ i, i < 4 → (arr[i] != "A" ∨ arr[i + 1] != "B") ∧ (arr[i] != "B" ∨ arr[i + 1] != "A"))) ∧
      arrangements.length = 36 :=
by
  sorry

end arrange_programs_l761_761075


namespace hamburgers_purchased_l761_761079

theorem hamburgers_purchased (total_revenue : ℕ) (hamburger_price : ℕ) (additional_hamburgers : ℕ) 
  (target_amount : ℕ) (h1 : total_revenue = 50) (h2 : hamburger_price = 5) (h3 : additional_hamburgers = 4) 
  (h4 : target_amount = 50) :
  (target_amount - (additional_hamburgers * hamburger_price)) / hamburger_price = 6 := 
by 
  sorry

end hamburgers_purchased_l761_761079


namespace cube_cannot_cover_5x5_square_l761_761411

theorem cube_cannot_cover_5x5_square {Cube : Type} (position : ℤ × ℤ) (faces : Cube → List (ℤ × ℤ)) (move : Cube → Cube) :
  ∀ initial_face : Cube, ∀ steps : List Cube, 
  (steps.head = initial_face) ∧ (∀ step ∈ steps, ℕ ≤ 6 ∧ move step = step + 1) →
  (length steps < 25) :=
begin
  sorry
end

end cube_cannot_cover_5x5_square_l761_761411


namespace find_a_minus_b_l761_761222

variable {g : ℝ → ℝ}

theorem find_a_minus_b
  (hg : ∀ x y : ℝ, x < y → g x < g y)
  (h_range : ∀ t : ℝ, g (2 * t^2 + t + 5) < g (t^2 - 3 * t + 2) ↔ -3 < t ∧ t < -1) :
  let a := -1
  let b := -3
  in a - b = 2 := 
by
  intros
  sorry

end find_a_minus_b_l761_761222


namespace find_y_l761_761144

-- Declare the variables and conditions
variable (x y : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := 1.5 * x = 0.3 * y
def condition2 : Prop := x = 20

-- State the theorem that given these conditions, y must be 100
theorem find_y (h1 : condition1 x y) (h2 : condition2 x) : y = 100 :=
by sorry

end find_y_l761_761144


namespace degree_h_is_5_l761_761846

def f (x : ℝ) : ℝ := -9 * x^5 + 2 * x^3 + 4 * x - 6

def h (p : ℝ → ℝ) (x : ℝ) : ℝ := 
  p(x) -- Define h(x) as any polynomial p(x)

theorem degree_h_is_5 
  (h_poly : h (λ x, 9 * x^5 + 0 * x^4 + 0 * x^3 + c * x^2 + d * x + e))
  (hf_h_degree_2 : degree (λ x, f(x) + h (λ x, 9 * x^5 + 0 * x^4 + 0 * x^3 + c * x^2 + d * x + e)) = 2) :
  degree (h (λ x, 9 * x^5 + 0 * x^4 + 0 * x^3 + c * x^2 + d * x + e)) = 5 :=
sorry

end degree_h_is_5_l761_761846


namespace Kolya_result_l761_761432

-- Define the list of numbers
def numbers := [1.05, 1.15, 1.25, 1.4, 1.5, 1.6, 1.75, 1.85, 1.95]

-- Defining the missing decimal point case
def mistaken_number := 15

-- Calculate the sum correctly and with a missing decimal point
noncomputable def correct_sum := numbers.sum
noncomputable def mistaken_sum := correct_sum + (mistaken_number - 1.5)

-- Prove the result is as expected
theorem Kolya_result : mistaken_sum = 27 := by
  sorry

end Kolya_result_l761_761432


namespace complementary_sets_count_l761_761819

noncomputable def number_of_complementary_sets : Nat := 
  let case2 := 4 * (3^3) * 3!
  let case3 := 6 * (3^2) * (3!)^2
  let case4 := 4 * 3 * (3!)^3
  let case5 := (3!)^4
  case2 + case3 + case4 + case5

theorem complementary_sets_count : number_of_complementary_sets = 4536 := by
  sorry

end complementary_sets_count_l761_761819


namespace maria_picks_correct_number_of_integers_l761_761996

noncomputable def numDivisorsMariaCanPick : ℕ :=
  (List.range' 1 720).countp (λ n, 720 % n = 0)

theorem maria_picks_correct_number_of_integers :
  numDivisorsMariaCanPick = 30 := 
sorry

end maria_picks_correct_number_of_integers_l761_761996


namespace factorize_m_cubed_minus_9_m_l761_761055

theorem factorize_m_cubed_minus_9_m (m : ℝ) : m^3 - 9 * m = m * (m + 3) * (m - 3) :=
by
  sorry

end factorize_m_cubed_minus_9_m_l761_761055


namespace circle_radius_probability_l761_761018

theorem circle_radius_probability 
  (square_area : ℝ := 4040 * 4040)
  (lattice_point_prob : ℝ := 3 / 4) 
  (π : ℝ := Real.pi)
  (circle_area : ℝ := λ (d : ℝ), π * d * d) :
  (Float.ofReal (sqrt (lattice_point_prob / π))).round = 0.5 :=
by
  sorry

end circle_radius_probability_l761_761018


namespace determine_r_l761_761481

theorem determine_r (r : ℚ) (h : 32 = 5^(2 * r + 3)) : r = -1/2 := 
by {
  sorry
}

end determine_r_l761_761481


namespace count_exactly_one_between_l761_761074

theorem count_exactly_one_between (A B C D E : Type) : 
  let people := [A, B, C, D, E] in
  ∃ ans, ans = 36 :=
begin
  sorry
end

end count_exactly_one_between_l761_761074


namespace part1_part2_l761_761105

-- Part (1)
theorem part1 (θ : ℝ) (h1 : cos θ = -1 / √ 3) (h2 : sin θ = √ 2 / √ 3) : 
  ( -cos(3 * π / 2 + θ) + √2 * sin(π / 2 + θ) ) / ( sin(2 * π - θ) - 2 * √2 * cos (-θ) ) = -2 :=
by sorry

-- Part (2)
theorem part2 (θ α : ℝ) (h1 : cos θ = -1 / √ 3) (h2 : sin θ = √ 2 / √ 3) (h3 : cos α = 1 / √ 3) 
  (h4 : sin α = √ 2 / √ 3) (h_symmetry : α = π - θ) : 
  sin(α - π / 6) = (3 * √ 2 - √ 3) / 6 :=
by sorry

end part1_part2_l761_761105


namespace percentage_difference_is_20_l761_761203

/-
Barry can reach apples that are 5 feet high.
Larry is 5 feet tall.
When Barry stands on Larry's shoulders, they can reach 9 feet high.
-/
def Barry_height : ℝ := 5
def Larry_height : ℝ := 5
def Combined_height : ℝ := 9

/-
Prove the percentage difference between Larry's full height and his shoulder height is 20%.
-/
theorem percentage_difference_is_20 :
  ((Larry_height - (Combined_height - Barry_height)) / Larry_height) * 100 = 20 :=
by
  sorry

end percentage_difference_is_20_l761_761203


namespace range_of_a_minus_b_l761_761556

theorem range_of_a_minus_b {a b : ℝ} (h1 : -2 < a) (h2 : a < 3) (h3 : 1 < b) (h4 : b < 2) : 
  -4 < a - b ∧ a - b < 2 := 
by
  sorry

end range_of_a_minus_b_l761_761556


namespace quadruples_solution_l761_761061

theorem quadruples_solution (a b c d : ℝ) :
  (a * b + c * d = 6) ∧
  (a * c + b * d = 3) ∧
  (a * d + b * c = 2) ∧
  (a + b + c + d = 6) ↔
  (a = 0 ∧ b = 1 ∧ c = 2 ∧ d = 3) ∨
  (a = 2 ∧ b = 3 ∧ c = 0 ∧ d = 1) ∨
  (a = 1 ∧ b = 0 ∧ c = 3 ∧ d = 2) ∨
  (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 0) :=
sorry

end quadruples_solution_l761_761061


namespace ratio_of_sums_l761_761562

theorem ratio_of_sums (a b c u v w : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_pos_u : 0 < u) (h_pos_v : 0 < v) (h_pos_w : 0 < w)
    (h1 : a^2 + b^2 + c^2 = 9) (h2 : u^2 + v^2 + w^2 = 49) (h3 : a * u + b * v + c * w = 21) : 
    (a + b + c) / (u + v + w) = 3 / 7 := 
by
  sorry

end ratio_of_sums_l761_761562


namespace Aunt_Zhang_expenditure_is_negative_l761_761240

-- Define variables for the problem
def income_yuan : ℤ := 5
def expenditure_yuan : ℤ := 3

-- The theorem stating Aunt Zhang's expenditure in financial terms
theorem Aunt_Zhang_expenditure_is_negative :
  (- expenditure_yuan) = -3 :=
by
  sorry

end Aunt_Zhang_expenditure_is_negative_l761_761240


namespace jane_earnings_l761_761197

def earnings_per_bulb : ℝ := 0.50
def tulip_bulbs : ℕ := 20
def iris_bulbs : ℕ := tulip_bulbs / 2
def daffodil_bulbs : ℕ := 30
def crocus_bulbs : ℕ := daffodil_bulbs * 3
def total_earnings : ℝ := (tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs) * earnings_per_bulb

theorem jane_earnings : total_earnings = 75.0 := by
  sorry

end jane_earnings_l761_761197


namespace inverse_function_correct_inequality_solution_l761_761113

noncomputable def f (x : ℝ) : ℝ := 1 - (2 / (2^x + 1))

noncomputable def f_inv (y : ℝ) : ℝ := Real.log (1 + y) / Real.log (1 - y)

theorem inverse_function_correct (x : ℝ) (hx : -1 < x ∧ x < 1) :
  f_inv (f x) = x :=
sorry

theorem inequality_solution :
  ∀ x, (1 / 2 < x ∧ x < 1) ↔ (f_inv x > Real.log (1 + x) + 1) :=
sorry

end inverse_function_correct_inequality_solution_l761_761113


namespace base_angle_of_isosceles_l761_761449

theorem base_angle_of_isosceles (vertex_angle : ℝ) (base_angle : ℝ) (h1 : vertex_angle = 30)
  (h2 : ∑ angle : list ℝ := [vertex_angle, base_angle, base_angle], angle = 180) : base_angle = 75 := 
by
  sorry

end base_angle_of_isosceles_l761_761449


namespace owen_saved_amount_l761_761836

/-- Prove that Owen saved 360 dollars in June given the conditions. ---/
theorem owen_saved_amount :
  let daily_burgers := 2
  let burger_cost := 12
  let june_days := 30
  let burgers_bought := daily_burgers * june_days
  let free_burgers := burgers_bought / 2
  let total_cost := burgers_bought * burger_cost
  let cost_with_deal := (burgers_bought - free_burgers) * burger_cost
  let saved_amount := total_cost - cost_with_deal
  saved_amount = 360 :=
by
  unfold daily_burgers
  unfold burger_cost
  unfold june_days
  unfold burgers_bought
  unfold free_burgers
  unfold total_cost
  unfold cost_with_deal
  unfold saved_amount
  sorry

end owen_saved_amount_l761_761836


namespace binomial_expansion_sum_abs_eq_2187_l761_761082

theorem binomial_expansion_sum_abs_eq_2187 :
  let a_0 := (1 : ℝ)
  let a_1 := -14
  let a_2 := 84
  let a_3 := -280
  let a_4 := 560
  let a_5 := -672
  let a_6 := 448
  let a_7 := -128
  ∑ i in (finset.range 8), |coeff (mv_polynomial.C (1 - 2 * (x : ℝ)) ^ 7) i| = 2187 := sorry

end binomial_expansion_sum_abs_eq_2187_l761_761082


namespace tan_C_half_area_triangle_l761_761188

-- Define the trigonometric relationship and conditions
variables {A B C : ℝ} -- Angles of triangle ABC
variables {a b c : ℝ} -- Sides opposite to angles A, B, and C respectively

-- Given conditions
axiom trig_condition : (sin A / a) + (sin B / b) = (cos C / c)
axiom pythagorean_condition : a^2 + b^2 - c^2 = 8

-- Task 1: Prove tan C = 1/2
theorem tan_C_half : tan C = 1 / 2 :=
sorry

-- Task 2: Prove the area of triangle ABC is 1
theorem area_triangle : let S := (1 / 2) * a * b * sin C in S = 1 :=
sorry

end tan_C_half_area_triangle_l761_761188


namespace prime_exponent_mod_pq_l761_761981

theorem prime_exponent_mod_pq (p q a n : ℕ) (hp : nat.prime p) (hq : nat.prime q)
  (h : a ≡ 1 [MOD (p-1)*(q-1)]) : (n^a) % (p*q) = n % (p*q) :=
by
  sorry

end prime_exponent_mod_pq_l761_761981


namespace find_a5_l761_761101

variable {α : Type*}

def arithmetic_sequence (a : ℕ → α) := ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a5 (a : ℕ → ℤ) (h1 : arithmetic_sequence a) (h2 : a 2 + a 8 = 12) : a 5 = 6 :=
by
  sorry

end find_a5_l761_761101


namespace vector_addition_proof_l761_761210

def u : ℝ × ℝ × ℝ := (-3, 2, 5)
def v : ℝ × ℝ × ℝ := (4, -7, 1)
def result : ℝ × ℝ × ℝ := (-2, -3, 11)

theorem vector_addition_proof : (2 • u + v) = result := by
  sorry

end vector_addition_proof_l761_761210


namespace meaningful_range_l761_761145

theorem meaningful_range (x : ℝ) : (∃ y : ℝ, y = (sqrt (x + 3)) / (x - 1)) ↔ (x ≥ -3 ∧ x ≠ 1) :=
by 
  sorry

end meaningful_range_l761_761145


namespace decreasing_f_interval_l761_761894

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (3 * a - 1) * x + 4 * a else log a x

theorem decreasing_f_interval (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) ↔ (1 / 7 ≤ a ∧ a < 1 / 3) := 
sorry

end decreasing_f_interval_l761_761894


namespace Alok_ordered_9_plates_of_mixed_vegetable_l761_761768

theorem Alok_ordered_9_plates_of_mixed_vegetable (chapatis rice total_paid cost_chapati cost_rice cost_mixed_vegetable : ℕ) 
(h1 : chapatis = 16) 
(h2 : rice = 5) 
(h3 : total_paid = 1015)
(h4 : cost_chapati = 6) 
(h5 : cost_rice = 45) 
(h6 : cost_mixed_vegetable = 70) : 
  ∃ (mixed_vegetable : ℕ), mixed_vegetable = 9 := 
by
  -- let cost_chapatis = chapatis * cost_chapati
  -- have h_cost_chapatis : cost_chapatis = 96, from sorry,
  -- let cost_rice = rice * cost_rice
  -- have h_cost_rice : cost_rice = 225, from sorry,
  -- let total_cost = cost_chapatis + cost_rice 
  -- have h_total_cost : total_cost = 321, from sorry,
  -- let remaining_amount = total_paid - total_cost
  -- have h_remaining_amount : remaining_amount = 694, from sorry,
  -- let mixed_vegetable = remaining_amount / cost_mixed_vegetable
  -- have h_mixed_vegetable : mixed_vegetable = 9, from sorry,
  existsi 9,
  rfl

end Alok_ordered_9_plates_of_mixed_vegetable_l761_761768


namespace number_of_students_l761_761279

theorem number_of_students (S N : ℕ) (h1 : S = 15 * N)
                           (h2 : (8 * 14) = 112)
                           (h3 : (6 * 16) = 96)
                           (h4 : 17 = 17)
                           (h5 : S = 225) : N = 15 :=
by sorry

end number_of_students_l761_761279


namespace proof_main_statement_l761_761206

noncomputable def proof_problem : Prop :=
  ∀ (f : ℝ → ℝ) (x0 : ℝ),
    (∀ x, continuous_at f x) ∧
    (∀ x ≠ x0, differentiable ℝ f) ∧
    ∃ L R, is_finite_limit (deriv (λ x, f x) x0) L ∧
           is_finite_limit (deriv (λ x, f x) x0) R →
    ∃ (g h : ℝ → ℝ) (α : ℤ), 
      differentiable ℝ g ∧
      is_linear_map ℝ h ∧
      α ∈ {-1, 0, 1} ∧
      ∀ x, f x = g x + α * |h x|

theorem proof_main_statement : proof_problem := 
  sorry

end proof_main_statement_l761_761206


namespace factor_of_increase_l761_761399

noncomputable def sum_arithmetic_progression (a1 d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem factor_of_increase (a1 d n : ℕ) (h1 : a1 > 0) (h2 : (sum_arithmetic_progression a1 (3 * d) n = 2 * sum_arithmetic_progression a1 d n)) :
  sum_arithmetic_progression a1 (4 * d) n = (5 / 2) * sum_arithmetic_progression a1 d n :=
sorry

end factor_of_increase_l761_761399


namespace brian_time_l761_761780

theorem brian_time (todd_time : ℕ) (h1 : todd_time = 88) (h2 : todd_time = brian_time - 8) : brian_time = 96 :=
by
  sorry

end brian_time_l761_761780


namespace Nancy_folders_l761_761999

def n_initial : ℕ := 43
def n_deleted : ℕ := 31
def n_per_folder : ℕ := 6
def n_folders : ℕ := (n_initial - n_deleted) / n_per_folder

theorem Nancy_folders : n_folders = 2 := by
  sorry

end Nancy_folders_l761_761999


namespace dividend_calculation_l761_761750

theorem dividend_calculation
(total_investment : ℝ)
(nominal_value : ℝ)
(premium_percent : ℝ)
(declared_dividend_percent : ℝ)
(h_total_investment : total_investment = 14400)
(h_nominal_value : nominal_value = 100)
(h_premium_percent : premium_percent = 0.20)
(h_declared_dividend_percent : declared_dividend_percent = 0.05) :
let cost_per_share := nominal_value * (1 + premium_percent),
    number_of_shares := total_investment / cost_per_share,
    dividend_per_share := nominal_value * declared_dividend_percent,
    total_dividend := number_of_shares * dividend_per_share in
total_dividend = 600 :=
by
  sorry

end dividend_calculation_l761_761750


namespace graph_represents_snail_and_rabbit_l761_761576

-- Variables
variable (t : ℝ) -- Time variable
variable d_snail d_rabbit : ℝ → ℝ -- Distance functions for snail and rabbit

-- Conditions
def snail_constant_speed (t : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ ∀ t', d_snail t' = k * t'

def rabbit_late_start_sprint_rest (t : ℝ) : Prop :=
  ∃ k₁ k₂ k₃ t₀ t₁ t₂ t₃ : ℝ,
  0 < k₂ ∧ 0 < k₃ ∧ t₁ > t₀ ∧ t₂ > t₁ ∧ t₃ > t₂ ∧ 
  t ≥ t₀ ∧ t₀ = 2 ∧ 
  (∀ t', t₀ ≤ t' ∧ t' < t₁ → d_rabbit t' = 0) ∧ -- Late start
  (∀ t', t₁ ≤ t' ∧ t' < t₂ → d_rabbit t' = k₁ * (t' - t₀)) ∧ -- Sprint
  (∀ t', t₂ ≤ t' ∧ t' < t₃ → d_rabbit t' = k₁ * (t₂ - t₀)) ∧ -- Rest
  (∀ t', t₃ ≤ t' ∧ t' ≤ t → d_rabbit t' = k₁ * (t₂ - t₀) + k₂ * (t' - t₂)) -- Sprint to finish

def snail_finishes_first : Prop :=
  ∀ t, t > 0 → d_snail t > d_rabbit t

-- Problem statement
theorem graph_represents_snail_and_rabbit :
  snail_constant_speed t →
  rabbit_late_start_sprint_rest t →
  snail_finishes_first →
  -- The appropriate graph that represents these conditions.
  -- In this context, we cannot define Graph A directly, but we can state that the conditions
  -- suitably represent the characteristics that match Graph A.
  true :=
by
  sorry

end graph_represents_snail_and_rabbit_l761_761576


namespace ab_range_l761_761925

theorem ab_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a * b = a + b) : 1 / 4 ≤ a * b :=
sorry

end ab_range_l761_761925


namespace option_b_is_correct_l761_761923

def is_parallel (μ v : ℝ × ℝ × ℝ) : Prop :=
∃ k : ℝ, k ≠ 0 ∧ μ = (k * v.1, k * v.2, k * v.3)

def is_perpendicular (μ v : ℝ × ℝ × ℝ) : Prop :=
μ.1 * v.1 + μ.2 * v.2 + μ.3 * v.3 = 0

def not_parallel_nor_perpendicular (μ v : ℝ × ℝ × ℝ) : Prop :=
¬ is_parallel μ v ∧ ¬ is_perpendicular μ v

theorem option_b_is_correct :
  not_parallel_nor_perpendicular (3, 0, -1) (0, 0, 2) :=
sorry

end option_b_is_correct_l761_761923


namespace locus_C_l761_761297

noncomputable def midpoint (A B : ℝ) : ℝ := (A + B) / 2
noncomputable def radius (A B : ℝ) : ℝ := abs (A - B) / 2
noncomputable def distance (x y : ℝ) : ℝ := abs (x - y)
noncomputable def circle_eq {A B : ℝ} := ∀ C : ℝ, distance C (midpoint A B) = radius A B

theorem locus_C (A B e : ℝ) 
  (h1 : ¬ (A = B)) 
  (circle_O : (O : ℝ) → distance O A = 0) 
  (circle_O_prime : (O' : ℝ) → distance O' B = 0) 
  (tangent_C : (C : ℝ) → distance C O = distance C O')
  : circle_eq := 
by 
  sorry

end locus_C_l761_761297


namespace tan_beta_expression_max_tan_beta_l761_761500

variable (α β : ℝ)
variable (h1 : 0 < α ∧ α < (π / 2))
variable (h2 : 0 < β ∧ β < (π / 2))
variable (h3 : α + β ≠ π / 2)
variable (h4 : sin β = sin α * cos (α + β))

theorem tan_beta_expression (h1 h2 h3 h4) :
  tan β = tan α / (1 + 2 * tan α ^ 2) :=
sorry

theorem max_tan_beta (h1 h2 h3 h4) :
  ∃ x, tan β = x ∧ x = (sqrt 2) / 4 :=
sorry

end tan_beta_expression_max_tan_beta_l761_761500


namespace major_premise_of_e_irrat_l761_761733

def is_non_repeating_infinite_decimal (x : Real) : Prop :=
  ∀ (repeats : ℕ → ℕ → Prop), ¬(∃ (n m : ℕ), repeats n m)

def is_irrational (x : Real) : Prop :=
  ¬(∃ (a b : ℤ), b ≠ 0 ∧ x = a / b)

theorem major_premise_of_e_irrat (e : Real) 
  (h1 : is_non_repeating_infinite_decimal e) : 
  ∃ (p : Real → Prop), p e ∧ (∀ x, p x ↔ is_irrational x) :=
by 
  use is_non_repeating_infinite_decimal
  split
  { exact h1 }
  { split
    { intro h
      exact sorry }
    { intro h
      exact sorry } }

end major_premise_of_e_irrat_l761_761733


namespace pyramid_volume_proof_l761_761649

open Real

noncomputable def pyramid_volume (AB BC PA PB : ℝ) (h1 : AB = 10) (h2 : BC = 6) (h3 : PB = 20) (h4 : PA = sqrt (20^2 - 10^2)) : ℝ := 
  (1 / 3) * (10 * 6) * PA

theorem pyramid_volume_proof : 
  ∀ (AB BC PA PB : ℝ), AB = 10 → BC = 6 → PB = 20 → PA = sqrt (20^2 - 10^2) → 
  pyramid_volume AB BC PA PB 10 6 20 (sqrt (400 - 100)) = 200 * sqrt 3 :=
by
  intros
  rw [h, h_1, h_2, h_3]
  sorry

end pyramid_volume_proof_l761_761649


namespace total_travel_time_l761_761793

-- Define the necessary distances and speeds
def distance_Washington_to_Idaho : ℝ := 640
def speed_Washington_to_Idaho : ℝ := 80
def distance_Idaho_to_Nevada : ℝ := 550
def speed_Idaho_to_Nevada : ℝ := 50

-- Definitions for time calculations
def time_Washington_to_Idaho : ℝ := distance_Washington_to_Idaho / speed_Washington_to_Idaho
def time_Idaho_to_Nevada : ℝ := distance_Idaho_to_Nevada / speed_Idaho_to_Nevada

-- Problem statement to prove
theorem total_travel_time : time_Washington_to_Idaho + time_Idaho_to_Nevada = 19 := 
by
  sorry

end total_travel_time_l761_761793


namespace domain_of_tangent_sqrt_l761_761287

noncomputable def domain_of_f : set ℝ := {x : ℝ | x ∈ (0, (Real.pi / 4)) ∨ x ∈ ((Real.pi / 4), 1)}

theorem domain_of_tangent_sqrt (x : ℝ) : 
(∀ k : ℤ, 2*x ≠ k * Real.pi + (Real.pi / 2)) → 
(x - x^2 > 0) → 
x ∈ (0, Real.pi / 4) ∨ x ∈ (Real.pi / 4, 1) := 
by
  intro h1 h2
  sorry

end domain_of_tangent_sqrt_l761_761287


namespace vasya_numbers_l761_761365

theorem vasya_numbers (x y : ℚ) (h : x + y = xy ∧ xy = x / y) : x = 1 / 2 ∧ y = -1 :=
by {
  sorry
}

end vasya_numbers_l761_761365


namespace problem_l761_761526

theorem problem (x y : ℝ) (h1 : 2 * x + y = 4) (h2 : x + 2 * y = 5) : 5 * x ^ 2 + 8 * x * y + 5 * y ^ 2 = 41 := 
by 
  sorry

end problem_l761_761526


namespace vector_subtraction_l761_761882

-- Define the vectors and the relationship that C is the midpoint of AB.
variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables (A B C : V)
variables (AC CB AB BC : V)

-- Define conditions 
def midpoint_condition (C : V) (A B : V) : Prop :=
  C = (A + B) / 2

def ac_equals_cb (AC CB : V) : Prop :=
  AC = CB

-- State the theorem
theorem vector_subtraction (C A B : V) (midpoint_cond : midpoint_condition C A B) (ac_eq_cb : ac_equals_cb (C - A) (B - C)) : 
  (B - A) - (C - B) = (C - A) :=
sorry

end vector_subtraction_l761_761882


namespace battery_lasts_12_hours_more_l761_761630

-- Define the battery consumption rates
def standby_consumption_rate : ℚ := 1 / 36
def active_consumption_rate : ℚ := 1 / 4

-- Define the usage times
def total_time_hours : ℚ := 12
def active_use_time_hours : ℚ := 1.5
def standby_time_hours : ℚ := total_time_hours - active_use_time_hours

-- Define the total battery used during standby and active use
def standby_battery_used : ℚ := standby_time_hours * standby_consumption_rate
def active_battery_used : ℚ := active_use_time_hours * active_consumption_rate
def total_battery_used : ℚ := standby_battery_used + active_battery_used

-- Define the remaining battery
def remaining_battery : ℚ := 1 - total_battery_used

-- Define how long the remaining battery will last on standby
def remaining_standby_time : ℚ := remaining_battery / standby_consumption_rate

-- Theorem stating the correct answer
theorem battery_lasts_12_hours_more :
  remaining_standby_time = 12 := 
sorry

end battery_lasts_12_hours_more_l761_761630


namespace martha_kept_nuts_l761_761452

theorem martha_kept_nuts (total_nuts : ℕ)
    (Tommy_received : ℕ)
    (Bessie_received : ℕ)
    (Bob_received : ℕ)
    (Jessie_received : ℕ)
    (boys_more_than_girls : Tommy_received + Bob_received - (Bessie_received + Jessie_received) = 100) :
    let martha_kept := total_nuts - (Tommy_received + Bessie_received + Bob_received + Jessie_received) in
    martha_kept = 321 :=
sorry

end martha_kept_nuts_l761_761452


namespace find_value_of_m_l761_761092

noncomputable def parabola := { p : ℝ // p > 0 }

def lies_on_parabola (A : ℝ × ℝ) (parabola : parabola) : Prop :=
  A.2 ^ 2 = 2 * parabola.val * A.1

def radius (A : ℝ × ℝ) (F : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - F.1) ^ 2 + (A.2 - F.2) ^ 2)

def chord_length (r : ℝ) (d : ℝ) : ℝ :=
  real.sqrt (r ^ 2 - d ^ 2) * 2

theorem find_value_of_m (m : ℝ) (p : parabola) (A : ℝ × ℝ)
  (hA : A = (m, 2 * real.sqrt 2))
  (h_on_parabola : lies_on_parabola A p)
  (h_F : p = ⟨4 / m, sorry⟩) :
  chord_length (radius A (4 / m, 0)) m = 2 * real.sqrt 7 →
  m = 2 * real.sqrt 3 / 3 :=
sorry

end find_value_of_m_l761_761092


namespace geometric_progression_x_geometric_progression_sum_l761_761045

theorem geometric_progression_x (x : ℝ) : (30 + x) ^ 2 = (10 + x) * (90 + x) → x = 0 := by
  assume h,
  have h1 : (30 + x) ^ 2 = 900 + 60 * x + x ^ 2 := by sorry,
  have h2 : (10 + x) * (90 + x) = 900 + 100 * x + x ^ 2 := by sorry,
  rw [h1, h2] at h,
  have eq : 900 + 60 * x + x ^ 2 = 900 + 100 * x + x ^ 2 := h,
  simp at eq,
  exact eq.symm

theorem geometric_progression_sum : 10 + 30 + 90 = 130 := by
  norm_num

end geometric_progression_x_geometric_progression_sum_l761_761045


namespace trig_identity_l761_761844

theorem trig_identity (θ : ℝ) (h : Real.tan (θ - Real.pi) = 2) :
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 :=
  sorry

end trig_identity_l761_761844


namespace vasya_numbers_l761_761356

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 := 
by 
  sorry

end vasya_numbers_l761_761356


namespace inscribed_circle_radius_PQR_l761_761389

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def heron_formula (a b c : ℝ) : ℝ :=
  let s := semiperimeter a b c
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def radius_inscribed_circle (a b c : ℝ) : ℝ :=
  let s := semiperimeter a b c
  let K := heron_formula a b c
  in K / s

theorem inscribed_circle_radius_PQR :
  radius_inscribed_circle 30 26 28 = 8 :=
by
  sorry

end inscribed_circle_radius_PQR_l761_761389


namespace problem1_problem2_l761_761783

theorem problem1 (x : ℝ) : 2 * (x - 1) ^ 2 = 18 ↔ x = 4 ∨ x = -2 := by
  sorry

theorem problem2 (x : ℝ) : x ^ 2 - 4 * x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 := by
  sorry

end problem1_problem2_l761_761783


namespace Vasya_numbers_l761_761375

theorem Vasya_numbers : ∃ (x y : ℝ), x + y = xy ∧ xy = x / y ∧ (x, y) = (1 / 2, -1) :=
by
  sorry

end Vasya_numbers_l761_761375


namespace required_string_length_l761_761479

/-- The length of the string required to draw an ellipse using the given method. -/
theorem required_string_length (a b : ℝ) (h1 : a = 12) (h2 : b = 8) : 2 * a = 24 :=
by
  rw [h1]
  norm_num
  sorry

end required_string_length_l761_761479


namespace percent_more_proof_l761_761102

-- Define the conditions
def y := 150
def x := 120
def is_percent_more (y x p : ℕ) : Prop := y = (1 + p / 100) * x

-- The proof problem statement
theorem percent_more_proof : ∃ p : ℕ, is_percent_more y x p ∧ p = 25 := by
  sorry

end percent_more_proof_l761_761102


namespace smallest_fraction_division_l761_761832

theorem smallest_fraction_division (a b : ℕ) (h_coprime : Nat.gcd a b = 1) 
(h1 : ∃ n, (25 * a = n * 21 * b)) (h2 : ∃ m, (15 * a = m * 14 * b)) : (a = 42) ∧ (b = 5) := 
sorry

end smallest_fraction_division_l761_761832


namespace first_term_exceeding_10000_l761_761665

/-- Definition of the sequence described in the problem --/
noncomputable def seq : ℕ → ℕ
| 0     := 0
| 1     := 3
| (n+2) := seq (n+1) + seq (n+1).sum

/-- Prove that the first term in the sequence to exceed 10000 is 49152 --/
theorem first_term_exceeding_10000 : ∃ n, seq n > 10000 ∧ seq n = 49152 :=
begin
  sorry
end

end first_term_exceeding_10000_l761_761665


namespace isolating_line_unique_l761_761929

noncomputable def f (x : ℝ) := x^2
noncomputable def g (a x : ℝ) := a * log x

theorem isolating_line_unique (a : ℝ) (hx : ∀ x, f x ≥ g a x ∧ g a x ≥ f x) :
  a = 2 * real.exp 1 := 
sorry

end isolating_line_unique_l761_761929


namespace square_area_10_or_17_l761_761517

open Real

/-- Given a square ABCD with vertices lying on the curve of the function f(x) = x^3 - 9/2 x + 1,
  prove that the area of the square is either 10 or 17.
-/
theorem square_area_10_or_17 (f : ℝ → ℝ) (h_f : ∀ x, f x = x^3 - 9/2 * x + 1)
  (h1 : ∃ A B C D : ℝ × ℝ, A ∈ (set_of_l exists (fun (x : ℝ) => f x = snd A)) ∧
                            B ∈ (set_of_l exists (fun (x : ℝ) => f x = snd B)) ∧
                            C ∈ (set_of_l exists (fun (x : ℝ) => f x = snd C)) ∧
                            D ∈ (set_of_l exists (fun (x : ℝ) => f x = snd D)) ∧
                            (A.1, A.2) = square_center(f) ∧
                            (B.1, B.2) = square_center(f) ∧
                            (C.1, C.2) = square_center(f) ∧
                            (D.1, D.2) = square_center(f)) :
  ∃ area, area = 10 ∨ area = 17 :=
by 
  sorry

end square_area_10_or_17_l761_761517


namespace Vasya_numbers_l761_761360

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 :=
by
  sorry

end Vasya_numbers_l761_761360


namespace find_m_l761_761099

open Real

theorem find_m (m : ℝ) (a : ℝ × ℝ × ℝ) (b : ℝ × ℝ × ℝ) (h : a = (-1, m, 2)) (k : b = (-1, 2, -1)) 
  (dot_prod : a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = -3) : 
  m = -1 := 
by
  -- sorry is a placeholder for the actual proof
  sorry

end find_m_l761_761099


namespace division_modulus_l761_761253

-- Definitions using the conditions
def a : ℕ := 8 * (10^9)
def b : ℕ := 4 * (10^4)
def n : ℕ := 10^6

-- Lean statement to prove the problem
theorem division_modulus (a b n : ℕ) (h : a = 8 * (10^9) ∧ b = 4 * (10^4) ∧ n = 10^6) : 
  ((a / b) % n) = 200000 := 
by 
  sorry

end division_modulus_l761_761253


namespace volume_of_cube_l761_761753

-- Conditions
def base_side_length : ℝ := 2
def lateral_faces_isosceles_right : Prop := true -- Placeholder to indicate property
def cube_bottom_on_base : Prop := true -- Placeholder to indicate property
def cube_vertices_touch_midpoints : Prop := true -- Placeholder to indicate property

-- Given that conditions hold, the volume of the cube is 1.
theorem volume_of_cube (h₁ : base_side_length = 2) 
  (h₂ : lateral_faces_isosceles_right) 
  (h₃ : cube_bottom_on_base) 
  (h₄ : cube_vertices_touch_midpoints) : 
  ∃ V : ℝ, V = 1 :=
by
  use 1
  sorry

end volume_of_cube_l761_761753


namespace vasya_numbers_l761_761355

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 := 
by 
  sorry

end vasya_numbers_l761_761355


namespace v_n_zero_for_all_n_l761_761615

variable (p : ℕ) (h_odd_prime : p.prime ∧ ¬ even p)
variable (u : ℕ → ℤ)

def binomial (n k : ℕ) : ℕ := Nat.binomial n k

noncomputable def v (n : ℕ) : ℤ := 
  (finset.range (n + 1)).sum (λ i, binomial n i * (p^i) * (u i))

theorem v_n_zero_for_all_n (h_inf_zeros : ∃ᶠ n in filter.at_top, v p u n = 0) : 
  ∀ n > 0, v p u n = 0 :=
sorry

end v_n_zero_for_all_n_l761_761615


namespace total_red_and_green_peaches_l761_761407

def red_peaches : ℕ := 6
def green_peaches : ℕ := 16

theorem total_red_and_green_peaches :
  red_peaches + green_peaches = 22 :=
  by 
    sorry

end total_red_and_green_peaches_l761_761407


namespace series_sum_eq_l761_761010

noncomputable def series_term (n : ℕ) : ℝ :=
  (4 * ↑n + 3) / ((4 * ↑n + 1)^2 * (4 * ↑n + 5)^2)

theorem series_sum_eq :
  (∑' n, series_term n) = 1 / 800 :=
by
  sorry

end series_sum_eq_l761_761010


namespace trigonometric_identity_l761_761506

-- Statement of the problem in Lean
theorem trigonometric_identity
  (x : ℝ)
  (h1 : x ∈ Ioo (-π / 2) 0)
  (h2 : tan x = - (4 / 3)) :
  sin (x + π) = 4 / 5 :=
sorry

end trigonometric_identity_l761_761506


namespace rationalize_denominator_l761_761248

theorem rationalize_denominator :
  ( ( √18 + √8 ) / ( √12 + √8 ) ) = ( 2.5 * √6 - 4 ) :=
by
  sorry

end rationalize_denominator_l761_761248


namespace Vasya_numbers_l761_761359

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 :=
by
  sorry

end Vasya_numbers_l761_761359


namespace bullets_probability_l761_761331

theorem bullets_probability (p q : ℚ) (h1 : p = 2 / 3) (h2 : q = 3 / 4) : 
  let k := 100 * p / q in
  ∃ (N : ℕ), N = 89 ∧ N ≥ k := 
by
  sorry

end bullets_probability_l761_761331


namespace smallest_multiple_of_17_more_than_6_of_73_l761_761715

theorem smallest_multiple_of_17_more_than_6_of_73 : 
  ∃ a : ℕ, a > 0 ∧ a % 17 = 0 ∧ a % 73 = 6 ∧ a = 663 := 
begin
  use 663,
  split,
  { exact nat.succ_pos' _ },
  split,
  { exact nat.mod_eq_zero_of_dvd (by norm_num) },
  split,
  { exact nat.modeq.symm (nat.modeq.eq_iff_modeq.mp (by norm_num)) },
  { refl }
end

end smallest_multiple_of_17_more_than_6_of_73_l761_761715


namespace _l761_761788

noncomputable def radius_of_circle_B : ℝ :=
by
  let r : ℝ := radius_B
  let rC : ℝ := 2 * r -- Radius of circle C
  let rA : ℝ := 2 -- Radius of circle A
  let rD : ℝ := 2 * rA -- Radius of circle D since circle A passes through D

  -- Using Pythagorean theorem on the appropriate triangle
  have h : (rA + r) ^ 2 = (rA - rC) ^ 2 + (rD - r) ^ 2 := by sorry

  -- Solving the quadratic equation 7 * r^2 - 20 * r + 12 = 0
  have quadratic_solver : r = (20 + 8) / 14 ∨ r = (20 - 8) / 14 := by sorry

  exact r = 6 / 7

axiom radius_B: ℝ -- This would be given in the actual proof script instead of being stated as an axiom

end _l761_761788


namespace linearly_dependent_k_l761_761122

theorem linearly_dependent_k (k : ℝ) : 
  (∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ a • (⟨2, 3⟩ : ℝ × ℝ) + b • (⟨1, k⟩ : ℝ × ℝ) = (0, 0)) ↔ k = 3 / 2 :=
by
  sorry

end linearly_dependent_k_l761_761122


namespace chloe_profit_l761_761457

def cost_per_dozen : ℕ := 50
def sell_per_half_dozen : ℕ := 30
def total_dozens_sold : ℕ := 50

def total_cost (n: ℕ) : ℕ := n * cost_per_dozen
def total_revenue (n: ℕ) : ℕ := n * (sell_per_half_dozen * 2)
def profit (cost revenue : ℕ) : ℕ := revenue - cost

theorem chloe_profit : 
  profit (total_cost total_dozens_sold) (total_revenue total_dozens_sold) = 500 := 
by
  sorry

end chloe_profit_l761_761457


namespace cost_per_book_l761_761703

-- Definitions and conditions
def number_of_books : ℕ := 8
def amount_tommy_has : ℕ := 13
def amount_tommy_needs_to_save : ℕ := 27

-- Total money Tommy needs to buy the books
def total_amount_needed : ℕ := amount_tommy_has + amount_tommy_needs_to_save

-- Proven statement
theorem cost_per_book : (total_amount_needed / number_of_books) = 5 := by
  -- Skip proof
  sorry

end cost_per_book_l761_761703


namespace sum_two_smallest_prime_factors_450_l761_761391

theorem sum_two_smallest_prime_factors_450 :
  let prime_factors := [2, 3, 5] in
  (prime_factors.take 2).sum = 5 :=
by
  have prime_factors : List ℕ := [2, 3, 5]
  have eq_factors : prime_factors.take 2 = [2, 3] := rfl
  have eq_sum : (prime_factors.take 2).sum = 2 + 3 := rfl
  rw [eq_factors, eq_sum]
  norm_num

end sum_two_smallest_prime_factors_450_l761_761391


namespace range_of_m_l761_761109

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 1 - (m * real.exp x) / (x^2 + x + 1)

theorem range_of_m (h : ∃ x0 : ℕ, 0 < x0 ∧ f x0 m ≥ 0 ∧ (∀ x > 0, f x m < 0 → x = x0)) :
  (7 / real.exp 2 : ℝ) < m ∧ m ≤ 3 / real.exp 1 := 
sorry

end range_of_m_l761_761109


namespace remainder_of_factorial_sum_mod_30_l761_761162

theorem remainder_of_factorial_sum_mod_30 :
  (Finset.sum (Finset.range 101) (λ n, Nat.factorial n)) % 30 = 3 :=
  sorry

end remainder_of_factorial_sum_mod_30_l761_761162


namespace vasya_numbers_l761_761369

theorem vasya_numbers (x y : ℚ) (h : x + y = xy ∧ xy = x / y) : x = 1 / 2 ∧ y = -1 :=
by {
  sorry
}

end vasya_numbers_l761_761369


namespace Marissa_has_21_more_marbles_than_Jonny_l761_761232

noncomputable def Mara_marbles (bags : ℕ) (marbles : ℕ) : ℕ :=
bags * marbles

noncomputable def Markus_marbles (bags : ℕ) (marbles : ℕ) : ℕ :=
bags * marbles

noncomputable def Jonny_marbles (total_marbles : ℕ) (bags : ℕ) : ℕ :=
total_marbles

noncomputable def Marissa_marbles (bags1 : ℕ) (marbles1 : ℕ) (bags2 : ℕ) (marbles2 : ℕ) : ℕ :=
(bags1 * marbles1) + (bags2 * marbles2)

noncomputable def Jonny : ℕ := Jonny_marbles 18 3

noncomputable def Marissa : ℕ := Marissa_marbles 3 5 3 8

theorem Marissa_has_21_more_marbles_than_Jonny : (Marissa - Jonny) = 21 :=
by
  sorry

end Marissa_has_21_more_marbles_than_Jonny_l761_761232


namespace Vasya_numbers_l761_761373

theorem Vasya_numbers : ∃ (x y : ℝ), x + y = xy ∧ xy = x / y ∧ (x, y) = (1 / 2, -1) :=
by
  sorry

end Vasya_numbers_l761_761373


namespace evaluate_expression_l761_761561

variables (a b c d m : ℝ)

lemma opposite_is_zero (h1 : a + b = 0) : a + b = 0 := h1

lemma reciprocals_equal_one (h2 : c * d = 1) : c * d = 1 := h2

lemma abs_value_two (h3 : |m| = 2) : |m| = 2 := h3

theorem evaluate_expression (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 2) :
  m + c * d + (a + b) / m = 3 ∨ m + c * d + (a + b) / m = -1 :=
by
  sorry

end evaluate_expression_l761_761561


namespace vasya_numbers_l761_761352

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 := 
by 
  sorry

end vasya_numbers_l761_761352


namespace laura_walk_distance_l761_761204

theorem laura_walk_distance 
  (east_blocks : ℕ) 
  (north_blocks : ℕ) 
  (block_length_miles : ℕ → ℝ) 
  (h_east_blocks : east_blocks = 8) 
  (h_north_blocks : north_blocks = 14) 
  (h_block_length_miles : ∀ b : ℕ, b = 1 → block_length_miles b = 1 / 4) 
  : (east_blocks + north_blocks) * block_length_miles 1 = 5.5 := 
by 
  sorry

end laura_walk_distance_l761_761204


namespace b_2015_eq_l761_761623

noncomputable def b : ℕ → ℝ
| 1     := 3 + Real.sqrt 11
| n + 1 := if n ≥ 1 then b n / b (n - 1) else b (n / n ) -- Handling n >= 1 condition inside the definition

-- sequence containment condition
def sequence_condition (b : ℕ → ℝ) := 
  ∀ n ≥ 2, b n = b (n - 1) * b (n + 1)

-- specific initial condition
def initial_conditions (b : ℕ → ℝ) := 
  b 1 = 3 + Real.sqrt 11 ∧ b 1987 = 17 + Real.sqrt 11

theorem b_2015_eq : sequence_condition b ∧ initial_conditions b → b 2015 = (3 - Real.sqrt 11) / 8 :=
sorry

end b_2015_eq_l761_761623


namespace rubiks_cube_path_impossible_l761_761600

-- Define the number of squares and vertices on the surface of the Rubik's cube
def num_squares : ℕ := 54
def num_vertices : ℕ := 56

-- Non-self-intersecting path on the surface of the Rubik's cube
def non_self_intersecting_path (squares vertices : ℕ) : Prop := 
  ∀ (path : list (ℕ × ℕ)), path.length = squares ∧ 
  (∀ p ∈ path, p.1 < vertices ∧ p.2 < vertices) ∧ 
  (∀ p1 p2 ∈ path, p1 ≠ p2 → p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2)

-- Main theorem statement: such a path does not exist
theorem rubiks_cube_path_impossible : 
  ¬ (∃ path, non_self_intersecting_path num_squares num_vertices path) :=
sorry

end rubiks_cube_path_impossible_l761_761600


namespace cube_roots_not_all_rounded_correctly_l761_761172

def approx (x y : ℝ) (eps : ℝ) : Prop :=
  abs (x - y) < eps

noncomputable def cube_root_approx_correct (   a2 a16 a54 a128 a250 a432 a686 a1024 : ℝ) : Prop :=
  approx (a2) (2^(1/3)) (10^-7) ∧
  approx (a16) (16^(1/3)) (10^-7) ∧
  approx (a54) (54^(1/3)) (10^-7) ∧
  approx (a128) (128^(1/3)) (10^-7) ∧
  approx (a250) (250^(1/3)) (10^-7) ∧
  approx (a432) (432^(1/3)) (10^-7) ∧
  approx (a686) (686^(1/3)) (10^-7) ∧
  approx (a1024) (1024^(1/3)) (10^-7)

theorem cube_roots_not_all_rounded_correctly :
  let a250' : ℝ := 6.2996053 in
  let a686' : ℝ := 8.8194474 in
  let a2 := 1.2599210 in
  let a16 := 2.5198421 in
  let a54 := 3.7797631 in
  let a128 := 5.0396842 in
  let a432 := 7.5595263 in
  let a250_bad := 6.2996053 in
  let a686_bad := 8.8194474 in
  let a1024 := 10.0793684 in
  ¬cube_root_approx_correct a2 a16 a54 a128 a250_bad a432 a686_bad a1024 ∧ 
   (approx a250' (250^(1/3)) (10^-7) ∧ approx a686' (686^(1/3)) (10^-7)) :=
by sorry

end cube_roots_not_all_rounded_correctly_l761_761172


namespace sums_of_digits_not_all_equal_l761_761735

theorem sums_of_digits_not_all_equal : 
  ∀ (groups : ℕ → list (list ℕ)) (g : ℕ) (n : ℕ), 
  (n = 72) → 
  (g = 18) → 
  (∀ i, groups i ≠ ∅) → 
  (∀ j < g, list.length (groups j) = 4) → 
  (∀ i j < g, i ≠ j → groups i ≠ groups j) → 
  (∀ i j ∈ list.range n, ∃ dep, (i ∈ dep ∧ j ∈ dep) → dep ∈ (list.range n)) → 
  (∃ (sum_digit : ℕ → ℕ), (∀ i < g, sum_digit (list.prod (groups i)) = sum_digit (list.prod (groups j)) → False) sorry

end sums_of_digits_not_all_equal_l761_761735


namespace prob_at_least_one_female_science_correct_expected_value_xi_correct_l761_761326

def total_students_science : ℕ := 8
def total_students_humanities : ℕ := 4

def males_science : ℕ := 5
def females_science : ℕ := 3
def males_humanities : ℕ := 1
def females_humanities : ℕ := 3

def total_selected : ℕ := 3
def selected_from_science : ℕ := 2
def selected_from_humanities : ℕ := 1

def probability_at_least_one_female_science : ℚ :=
  (C(5, 1) * C(3, 1) + C(3, 2)) / C(8, 2)

theorem prob_at_least_one_female_science_correct :
  probability_at_least_one_female_science = 9 / 14 :=
by
  sorry

def prob_distribution_xi : Fin 3 → ℚ
| 0 => (C(5, 0) * C(3, 2)) / C(8, 2) * (C(1, 1)) / C(4, 1)
| 1 => ((C(5, 1) * C(3, 1)) / C(8, 2) + (C(3, 2)) / C(8, 2)) * (C(1, 1)) / C(4, 1)
| 2 => (C(5, 2)) / C(8, 2) * (C(1, 1)) / C(4, 1)

def expected_value_xi : ℚ :=
  0 * prob_distribution_xi 0 + 1 * prob_distribution_xi 1 + 2 * prob_distribution_xi 2

theorem expected_value_xi_correct :
  expected_value_xi = 19 / 56 :=
by
  sorry

end prob_at_least_one_female_science_correct_expected_value_xi_correct_l761_761326


namespace find_a4_l761_761884

variable {a_n : ℕ → ℝ}
variable (S_n : ℕ → ℝ)

noncomputable def Sn := 1/2 * 5 * (a_n 1 + a_n 5)

axiom h1 : S_n 5 = 25
axiom h2 : a_n 2 = 3

theorem find_a4 : a_n 4 = 5 := sorry

end find_a4_l761_761884


namespace probability_at_least_half_girls_l761_761607

-- Conditions
def six_children : ℕ := 6
def prob_girl : ℝ := 0.5

-- Statement to prove
theorem probability_at_least_half_girls :
  (∑ k in finset.range (six_children + 1), if 3 ≤ k then ↑(nat.binomial six_children k) * (prob_girl ^ k) * ((1 - prob_girl) ^ (six_children - k)) else 0) = 21 / 32 :=
by sorry

end probability_at_least_half_girls_l761_761607


namespace angle_ACB_is_44_l761_761180

theorem angle_ACB_is_44
  (DC_par_AB : Parallel DC AB)
  (angle_DCA : ∠ DCA = 50)
  (angle_ABC : ∠ ABC = 68) :
  ∠ ACB = 44 := by
  sorry

end angle_ACB_is_44_l761_761180


namespace similarity_Oa_Ob_Oc_ABC_perpendicular_bisectors_intersect_l761_761417

variables {A B C A1 B1 C1 : Type*} [triangle A B C]
variables (O Oa Ob Oc : circumcenter A B C A1 B1 C1) (H Ha Hb Hc : orthocenter A B C A1 B1 C1)

-- Part (a)
theorem similarity_Oa_Ob_Oc_ABC :
  ∆ Oa Ob Oc ~ ∆ A B C := 
sorry

-- Part (b)
theorem perpendicular_bisectors_intersect :
  intersect_perpendicular_bisectors O H Oa Ha Ob Hb Oc Hc := 
sorry

end similarity_Oa_Ob_Oc_ABC_perpendicular_bisectors_intersect_l761_761417


namespace minimum_vertical_segment_length_l761_761294

open Real

theorem minimum_vertical_segment_length :
  let f := λ x : ℝ, |x|
  let g := λ x : ℝ, -x^2 - 2*x - 1
  ∃ x : ℝ, x < 0 ∧ (f x - g x) = 3 / 4 :=
by sorry

end minimum_vertical_segment_length_l761_761294


namespace smallest_multiple_of_17_more_than_6_of_73_l761_761716

theorem smallest_multiple_of_17_more_than_6_of_73 : 
  ∃ a : ℕ, a > 0 ∧ a % 17 = 0 ∧ a % 73 = 6 ∧ a = 663 := 
begin
  use 663,
  split,
  { exact nat.succ_pos' _ },
  split,
  { exact nat.mod_eq_zero_of_dvd (by norm_num) },
  split,
  { exact nat.modeq.symm (nat.modeq.eq_iff_modeq.mp (by norm_num)) },
  { refl }
end

end smallest_multiple_of_17_more_than_6_of_73_l761_761716


namespace weights_are_equal_l761_761839

variable {n : ℕ}
variables {a : Fin (2 * n + 1) → ℝ}

def weights_condition
    (a : Fin (2 * n + 1) → ℝ) : Prop :=
  ∀ i : Fin (2 * n + 1), ∃ (A B : Finset (Fin (2 * n + 1))),
    A.card = n ∧ B.card = n ∧ A ∩ B = ∅ ∧
    A ∪ B = Finset.univ.erase i ∧
    (A.sum a = B.sum a)

theorem weights_are_equal
    (h : weights_condition a) :
  ∃ k : ℝ, ∀ i : Fin (2 * n + 1), a i = k :=
  sorry

end weights_are_equal_l761_761839


namespace parallelogram_relation_l761_761965

-- Definitions based on the conditions in a)
variable (A B C D E F : Point)
variable [parallelogram : Parallelogram A B C D]
variable [line_through_C : ∃ l : Line, l.contains C ∧ l.produced.contains E ∧ l.produced.contains F]

-- The theorem statement we want to prove
theorem parallelogram_relation :
  AC^2 + CE * CF = AB * AE + AD * AF :=
by sorry

end parallelogram_relation_l761_761965


namespace simplify_fraction_l761_761255

theorem simplify_fraction (a b : ℝ) (h : a ≠ b) : 
  (a ^ -6 - b ^ -6) / (a ^ -3 - b ^ -3) = a ^ -6 + a ^ -3 * b ^ -3 + b ^ -6 :=
by sorry

end simplify_fraction_l761_761255


namespace irreducible_positive_fraction_unique_l761_761058

theorem irreducible_positive_fraction_unique
  (a b : ℕ) (h_pos : a > 0 ∧ b > 0) (h_gcd : Nat.gcd a b = 1)
  (h_eq : (a + 12) * b = 3 * a * (b + 12)) :
  a = 2 ∧ b = 9 :=
by
  sorry

end irreducible_positive_fraction_unique_l761_761058


namespace omitted_decimal_sum_is_integer_l761_761429

def numbers : List ℝ := [1.05, 1.15, 1.25, 1.4, 1.5, 1.6, 1.75, 1.85, 1.95]

theorem omitted_decimal_sum_is_integer :
  1.05 + 1.15 + 1.25 + 1.4 + (15 : ℝ) + 1.6 + 1.75 + 1.85 + 1.95 = 27 :=
by sorry

end omitted_decimal_sum_is_integer_l761_761429


namespace chord_length_l761_761281

variable (x y : ℝ)

/--
The chord length cut by the line y = 2x - 2 on the circle (x-2)^2 + (y-2)^2 = 25 is 10.
-/
theorem chord_length (h₁ : y = 2 * x - 2) (h₂ : (x - 2)^2 + (y - 2)^2 = 25) : 
  ∃ length : ℝ, length = 10 :=
sorry

end chord_length_l761_761281


namespace series_sum_eq_l761_761009

noncomputable def series_term (n : ℕ) : ℝ :=
  (4 * ↑n + 3) / ((4 * ↑n + 1)^2 * (4 * ↑n + 5)^2)

theorem series_sum_eq :
  (∑' n, series_term n) = 1 / 800 :=
by
  sorry

end series_sum_eq_l761_761009


namespace longest_diagonal_of_rhombus_l761_761421

noncomputable def length_of_longest_diagonal (area : ℝ) (ratio : ℝ) :=
  (let x := (area * 8 / (ratio + 1)^2).sqrt in 4 * x)

theorem longest_diagonal_of_rhombus :
  length_of_longest_diagonal 144  (4 / 3) = 8 * Real.sqrt 6 :=
by
  sorry

end longest_diagonal_of_rhombus_l761_761421


namespace common_measure_largest_l761_761182

theorem common_measure_largest {a b : ℕ} (h_a : a = 15) (h_b : b = 12): 
  (∀ c : ℕ, c ∣ a ∧ c ∣ b → c ≤ Nat.gcd a b) ∧ Nat.gcd a b = 3 := 
by
  sorry

end common_measure_largest_l761_761182


namespace math_problem_solution_l761_761897

noncomputable def hyperbola_eq {a b : ℝ} (ha : a > 0) (hb : b > 0) (eccentricity : ℝ) (pA : ℝ × ℝ) 
  (hpA : pA = (2, 3)) (he : eccentricity = 2) : Prop := 
  ∃ a b : ℝ, (a > 0) ∧ (b = sqrt 3 * a) ∧ (pA = (2, 3)) ∧ (a^2 = 1) ∧ (he = (c / a)) 
  ∧ (frac (b^2) (a^2) = 3)
  ∧ (x^2 - (y^2 / 3) = 1)

noncomputable def line_eq {m : ℝ} (slope : ℝ) (pP pQ pM : ℝ × ℝ) 
  (hpQ_mid : pQ = (pP + pM) / 2) (hslope : slope = sqrt 5 / 5) : Prop :=
  ∃ m : ℝ, (slope = sqrt 5 / 5) ∧ (14 * y^2 + 6 * sqrt 5 * t * y + 3 * (t^2 - 1) = 0) 
  ∧ (t^2 = 21) ∧ (line_eq = x - sqrt 5 * y ± sqrt 21)

theorem math_problem_solution {a b : ℝ} {ha : a > 0} {hb : b > 0} {eccentricity : ℝ} {pA : ℝ × ℝ} 
  {hpA : pA = (2, 3)} {he : eccentricity = 2} {m : ℝ} {slope : ℝ} {pP pQ pM : ℝ × ℝ} 
  {hpQ_mid : pQ = (pP + pM) / 2} {hslope : slope = sqrt 5 / 5} : 
  hyperbola_eq ha hb he pA hpA he ∧ line_eq slope pP pQ pM hpQ_mid hslope :=
sorry

end math_problem_solution_l761_761897


namespace part1_part2_l761_761514

-- Given conditions
variable {a : ℕ → ℝ}
variable (λ : ℝ) (hλ : λ > 0)
variable h₁ : a 1 = λ
variable h₂ : ∀ n : ℕ, a n * a (n + 1) = 2^(7 - 2*n)

-- Part 1: Prove that for n ≥ 2, (a (n + 1)) / (a (n - 1)) = 1/4
theorem part1 (n : ℕ) (hn : n ≥ 2) : (a (n + 1)) / (a (n - 1)) = 1 / 4 :=
by sorry

-- Part 2: Show the existence of λ such that the sequence {a_n} is geometric
theorem part2 : ∃ λ > 0, λ = 8 ∧ (∀ n : ℕ, if n = 1 then a n = λ 
                                                         else if n % 2 = 0 then a n = 2^(4 - 2*n / 2)
                                                         else a n = 2^(4 - (2*n - 1) / 2)) :=
by sorry

end part1_part2_l761_761514


namespace at_least_one_hits_l761_761243

open ProbabilityTheory

def prob_person_A_hits : ℝ := 0.8
def prob_person_B_hits : ℝ := 0.8

theorem at_least_one_hits : 
  let prob_at_least_one_hits := 1 - (1 - prob_person_A_hits) * (1 - prob_person_B_hits)
  in prob_at_least_one_hits = 0.96 :=
sorry

end at_least_one_hits_l761_761243


namespace common_chord_length_of_two_overlapping_circles_eq_12sqrt3_l761_761704

def radius := 12
def is_equilateral_triangle (a b c : ℝ) := a = b ∧ b = c
def chord_length (r : ℝ) := 2 * r * Math.sqrt 3

theorem common_chord_length_of_two_overlapping_circles_eq_12sqrt3 : 
  ∀ (r : ℝ), 
  r = 12 → 
  let a := r in let b := r in let c := r in
  is_equilateral_triangle a b c →
  chord_length r = 12 * Math.sqrt 3 := by
  intros r hr_eq a b c h_triangle
  sorry

end common_chord_length_of_two_overlapping_circles_eq_12sqrt3_l761_761704


namespace abs_neg_five_is_five_l761_761277

theorem abs_neg_five_is_five : abs (-5) = 5 := 
by 
  sorry

end abs_neg_five_is_five_l761_761277


namespace find_angle_and_area_inscribed_circle_l761_761598

theorem find_angle_and_area_inscribed_circle 
  (a b c : ℝ) (A B C : ℝ)
  (h1 : a + b = 13)
  (h2 : c = 7)
  (h3 : 4 * Real.sin (A + B) / 2 ^ 2 - Real.cos (2 * C) = 7 / 2) :
  C = Real.pi / 3 ∧ (let S := 1 / 2 * a * b * Real.sin C
                         r := 2 * S / (a + b + c) 
                     in Real.pi * r ^ 2 = 3 * Real.pi) :=
by
  sorry

end find_angle_and_area_inscribed_circle_l761_761598


namespace robins_hair_cut_l761_761652

theorem robins_hair_cut (x : ℕ) : 16 - x + 12 = 17 → x = 11 := by
  sorry

end robins_hair_cut_l761_761652


namespace combined_degrees_l761_761258

theorem combined_degrees (summer_degrees jolly_degrees : ℕ) (h1 : summer_degrees = jolly_degrees + 5) (h2 : summer_degrees = 150) : summer_degrees + jolly_degrees = 295 := 
by
  sorry

end combined_degrees_l761_761258


namespace count_3_digit_multiples_of_25_not_75_l761_761139

theorem count_3_digit_multiples_of_25_not_75 : 
  (finset.Icc 100 975).filter (λ n, n % 25 = 0 ∧ n % 75 ≠ 0).card = 24 :=
by
  sorry

end count_3_digit_multiples_of_25_not_75_l761_761139


namespace store_cost_relation_cost_effectiveness_l761_761333

variables (x : ℕ) (y1 y2 : ℝ)

def storeA_price (x : ℕ) : ℝ :=
  if x = 1 then
    200
  else
    140 * x + 60

def storeB_price (x : ℕ) : ℝ :=
  150 * x

theorem store_cost_relation (h : x ≥ 1) :
  storeA_price x = 140 * x + 60 ∧ storeB_price x = 150 * x :=
begin
  split,
  { rw storeA_price,
    split_ifs,
    { simp [h] },
    { refl } },
  { rw storeB_price }
end

theorem cost_effectiveness (h : x ≥ 1) :
  (x < 6 → storeA_price x < storeB_price x) ∧ (x > 6 → storeA_price x > storeB_price x) :=
begin
  split,
  { intro h1,
    calc
      storeA_price x = 140 * x + 60 : by simp [storeA_price, h]
      ... < 150 * x : by linarith [h1] },
  { intro h2,
    calc
      storeA_price x = 140 * x + 60 : by simp [storeA_price, h]
      ... > 150 * x : by linarith [h2] }
end

end store_cost_relation_cost_effectiveness_l761_761333


namespace even_function_behavior_l761_761808

noncomputable def f : ℝ → ℝ := sorry /- define f(x) accordingly -/

theorem even_function_behavior
  (even_f : ∀ x, f x = f (-x))
  (monotonic_dec : ∀ x1 x2 ∈ Iic (0 : ℝ), x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0) :
  f 1 < f (-2) ∧ f (-2) < f (-3) :=
begin
  sorry
end

end even_function_behavior_l761_761808


namespace no_repeating_odd_pair_adj_123456_l761_761805

theorem no_repeating_odd_pair_adj_123456 :
  arrangements ([1, 2, 3, 4, 5, 6], {1, 3, 5}) = 432 :=
  sorry

end no_repeating_odd_pair_adj_123456_l761_761805


namespace domain_f1_correct_f2_correct_f2_at_3_l761_761403

noncomputable def f1 (x : ℝ) : ℝ := Real.sqrt (4 - 2 * x) + 1 + 1 / (x + 1)

noncomputable def domain_f1 : Set ℝ := {x | 4 - 2 * x ≥ 0} \ (insert 1 (insert (-1) {}))

theorem domain_f1_correct : domain_f1 = { x | x ≤ 2 ∧ x ≠ 1 ∧ x ≠ -1 } :=
by
  sorry

noncomputable def f2 (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem f2_correct : ∀ x, f2 (x + 1) = x^2 - 2 * x :=
by
  sorry

theorem f2_at_3 : f2 3 = 0 :=
by
  sorry

end domain_f1_correct_f2_correct_f2_at_3_l761_761403


namespace find_spherical_coordinates_of_A_l761_761673

theorem find_spherical_coordinates_of_A :
  let x := (3 * Real.sqrt 3) / 2,
      y := 9 / 2,
      z := 3,
      r := Real.sqrt ((x^2) + (y^2) + (z^2)),
      theta := Real.arctan (y / x),
      phi := Real.arccos (z / r)
  in (r, theta, phi) = (6, Real.pi / 3, Real.pi / 3) := by
  sorry

end find_spherical_coordinates_of_A_l761_761673


namespace four_digit_flippies_div_by_4_l761_761419

def is_flippy (n : ℕ) : Prop := 
  let digits := [4, 6]
  n / 1000 ∈ digits ∧
  (n / 100 % 10) ∈ digits ∧
  ((n / 10 % 10) = if (n / 100 % 10) = 4 then 6 else 4) ∧
  (n % 10) = if (n / 1000) = 4 then 6 else 4

def is_divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

theorem four_digit_flippies_div_by_4 : 
  ∃! n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ is_flippy n ∧ is_divisible_by_4 n :=
by
  sorry

end four_digit_flippies_div_by_4_l761_761419


namespace spinner_points_east_l761_761599

def initial_direction : ℤ := 0 -- North
def clockwise_revolutions : ℚ := 7 / 2 -- 3 1/2 revolutions
def counterclockwise_revolutions : ℚ := 17 / 4 -- 4 1/4 revolutions

def net_movement : ℚ := clockwise_revolutions - counterclockwise_revolutions

-- Convert net movement to the final direction
-- We know 0 represents north, 1/4 represents west, 1/2 represents south, and 3/4 represents east.
def direction_after_moves : ℚ := initial_direction + net_movement

theorem spinner_points_east 
  (initial_direction = 0)
  (clockwise_revolutions = 7 / 2)
  (counterclockwise_revolutions = 17 / 4)
  : direction_after_moves = 3 / 4 :=
by
  sorry

end spinner_points_east_l761_761599


namespace find_lambda_l761_761128

noncomputable def vector_a : List ℤ := [2, 3, -1]
noncomputable def vector_b (λ : ℤ) : List ℤ := [4, λ, -2]

theorem find_lambda (λ : ℤ) (h : 0 ≠ (2 * 4 + 3 * λ + (-1) * (-2))) :
  λ = (-10 / 3) :=
by
  sorry

end find_lambda_l761_761128


namespace binary_to_decimal_l761_761022

theorem binary_to_decimal :
  ∀ n : ℕ, n = 101 →
  ∑ i in Finset.range 3, (n / 10^i % 10) * 2^i = 5 :=
by
  sorry

end binary_to_decimal_l761_761022


namespace part_I_part_II_part_III_l761_761096

noncomputable def ellipse_params : (ℝ × ℝ) :=
  let a := 2
  let b := sqrt 3
  (a, b)

def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

theorem part_I :
  ∃ (a b : ℝ), (a = 2) ∧ (b = sqrt 3) ∧ (∀ (x y : ℝ), ellipse a b x y ↔ x^2 / 4 + y^2 / 3 = 1) :=
  sorry

theorem part_II :
  ∀ (M N : ℝ × ℝ), (∃ (x y : ℝ), M = (x, y) ∧ N = (x, -y)) ∧ (let (a, b) := ellipse_params in ellipse a b (fst M) (snd M)) ∧ 
  ((0, fst M, snd M) • (0, fst N, snd N) = -2) →
  ∃ (k : ℝ), (y = k (x - 1) ∨ y = -k (x - 1)) :=
  sorry

theorem part_III :
  ∀ (A B M N : ℝ × ℝ), (∃ k : ℝ, y = k x) ∧ (MN ∥ AB) →
  ∃ c : ℝ, c = 4 ∧ ∀ |A - B|^2 / |M - N| = c :=
  sorry

end part_I_part_II_part_III_l761_761096


namespace miles_driven_l761_761635

noncomputable def total_paid : ℝ := 95.74
noncomputable def rental_fee : ℝ := 20.99
noncomputable def charge_per_mile : ℝ := 0.25

theorem miles_driven :
  let miles := (total_paid - rental_fee) / charge_per_mile 
  in miles = 299 :=
by
  let miles := (total_paid - rental_fee) / charge_per_mile
  have h1 : miles = 299 := sorry
  exact h1

end miles_driven_l761_761635


namespace binary101_is_5_l761_761032

theorem binary101_is_5 : 
  let binary101 := [1, 0, 1] in
  let decimal := binary101[0] * 2^2 + binary101[1] * 2^1 + binary101[2] * 2^0 in
  decimal = 5 :=
by
  let binary101 := [1, 0, 1]
  let decimal := binary101[0] * 2^2 + binary101[1] * 2^1 + binary101[2] * 2^0
  show decimal = 5
  sorry

end binary101_is_5_l761_761032


namespace sum_first_5_terms_l761_761173

variable {a r : ℝ}

theorem sum_first_5_terms (h1 : a * (1 + r + r^2) = 13)
                          (h2 : a * (1 + r + r^2 + r^3 + r^4 + r^5 + r^6) = 183) :
    a * (1 + r + r^2 + r^3 + r^4) = ? := 
sorry

end sum_first_5_terms_l761_761173


namespace problem_1_problem_2_problem_3_l761_761093

-- Definitions for sequences
def S (a : ℕ → ℝ) (n : ℕ) : ℝ := (finset.range n).sum (λ k, a k + 1)
def a : ℕ → ℝ
| 0 := 2
| (n + 1) := 2 * (a n)
def b : ℕ → ℝ
| 0 := 3 / 2
| (n + 1) := if (n + 1) ≥ 2 then (-1)^(n + 1) * (1 / 2^(n + 1) + 1) else 3 / 2
def c (λ : ℝ) (n : ℕ) := 2^n + λ * b n

-- Conditions and problems
theorem problem_1 : ∀ n : ℕ, S a n = 2 * a n - 2 := sorry
theorem problem_2 : ∀ n : ℕ, ∑ k in finset.range (n+1), (-1)^(k+1) * b k / (2^k + 1) = 1 / a n := sorry 
theorem problem_3 : ∃ λ : ℝ, (∀ n : ℕ, n > 0 → c λ n > c λ (n - 1)) ↔ λ ∈ (-128 / 35, 32 / 19) := sorry

end problem_1_problem_2_problem_3_l761_761093


namespace tetrahedron_volume_formula_l761_761988

variables (r₀ S₀ S₁ S₂ S₃ V : ℝ)

theorem tetrahedron_volume_formula
  (h : V = (1/3) * (S₁ + S₂ + S₃ - S₀) * r₀) :
  V = (1/3) * (S₁ + S₂ + S₃ - S₀) * r₀ :=
by { sorry }

end tetrahedron_volume_formula_l761_761988


namespace possible_values_of_r_l761_761435

noncomputable def triangle_area (r : ℝ) : ℝ :=
  let base := real.sqrt (2 * (r + 2))
  let height := r + 2
  (1 / 2) * base * height

theorem possible_values_of_r (r : ℝ) : 32 ≤ triangle_area r ∧ triangle_area r ≤ 128 ↔ 9.31 ≤ r ∧ r ≤ 20.63 :=
begin
  sorry
end

end possible_values_of_r_l761_761435


namespace alternative_plan_cost_is_eleven_l761_761807

-- Defining current cost
def current_cost : ℕ := 12

-- Defining the alternative plan cost in terms of current cost
def alternative_cost : ℕ := current_cost - 1

-- Theorem stating the alternative cost is $11
theorem alternative_plan_cost_is_eleven : alternative_cost = 11 :=
by
  -- This is the proof, which we are skipping with sorry
  sorry

end alternative_plan_cost_is_eleven_l761_761807


namespace census_suitable_survey_l761_761447

theorem census_suitable_survey (A B C D : Prop) : 
  D := 
sorry

end census_suitable_survey_l761_761447


namespace coordinates_of_foci_l761_761063

-- Define the constants used in the proof
def a_squared : ℝ := 1 / 2
def b_squared : ℝ := 1 / 3

-- Define the condition of the given ellipse equation in standard form
noncomputable def ellipse_standard_form_equation (x y : ℝ) : Prop :=
  (x^2 / a_squared) + (y^2 / b_squared) = 1

-- Define the calculation of 'c' as the distance from the center to a focus
noncomputable def c_value : ℝ := Real.sqrt (a_squared - b_squared)

-- Prove the coordinates of the foci based on the given conditions
theorem coordinates_of_foci :
  (∀ x y : ℝ, ellipse_standard_form_equation x y ↔ (2*x^2 + 3*y^2 = 1)) →
  (c_value = Real.sqrt (1/6)) →
  (c_value = (Real.sqrt 6) / 6) →
  ∃ (fx₁ fx₂ : ℝ), fx₁ = -c_value ∧ fx₂ = c_value ∧ 
  ∀ x y : ℝ, ellipse_standard_form_equation x y → (x, y) = (fx₁, 0) ∨ (x, y) = (fx₂, 0) :=
sorry

end coordinates_of_foci_l761_761063


namespace solution_to_fraction_problem_l761_761059

noncomputable def fraction_problem : Prop :=
  ∀ (a b : Nat), a > 0 -> b > 0 -> Nat.gcd a b = 1 ->
    ((a + 12) * b = 3 * a * (b + 12)) -> a = 2 ∧ b = 9

theorem solution_to_fraction_problem : fraction_problem :=
sorry

end solution_to_fraction_problem_l761_761059


namespace maximum_value_of_function_l761_761290

theorem maximum_value_of_function :
  ∀ (x : ℝ), -2 < x ∧ x < 0 → x + 1 / x ≤ -2 :=
by
  sorry

end maximum_value_of_function_l761_761290


namespace circle_polar_equation_and_intersection_l761_761954

-- Unpacking the conditions
def parametric_circle (φ : ℝ) : ℝ × ℝ := (1 + Real.cos φ, Real.sin φ)

-- Cartesian equation derived from parametric equations
def cartesian_circle (x y : ℝ) := (x - 1)^2 + y^2 = 1

-- Polar conversion function
def convert_to_polar (x y : ℝ) : ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2)
  let θ := Real.atan2 y x
  (ρ, θ)

-- Defining the ray OM: θ = π/4
def ray_OM(θ : ℝ) : θ = π / 4

-- Defining the intersection point P in Cartesian coordinates
def point_P := (1, 1)

-- Specifying the corresponding polar coordinates
def polar_coordinates_P := (Real.sqrt 2, π / 4)

-- The lean theorem proving the polar equation of circle C and intersection point P
theorem circle_polar_equation_and_intersection :
  (∀ φ, let (x, y) := parametric_circle φ in cartesian_circle x y) ∧
  (let P := point_P in convert_to_polar P.1 P.2 = polar_coordinates_P) :=
by
  sorry -- Proof is omitted

end circle_polar_equation_and_intersection_l761_761954


namespace value_2_stddev_less_than_mean_l761_761278

theorem value_2_stddev_less_than_mean :
  let mean := 17.5
  let stddev := 2.5
  mean - 2 * stddev = 12.5 :=
by
  sorry

end value_2_stddev_less_than_mean_l761_761278


namespace find_dot_AP_BC_l761_761958

-- Defining the lengths of the sides of the triangle.
def length_AB : ℝ := 13
def length_BC : ℝ := 14
def length_CA : ℝ := 15

-- Defining the provided dot product conditions at point P.
def dot_BP_CA : ℝ := 18
def dot_CP_BA : ℝ := 32

-- The target is to prove the final dot product.
theorem find_dot_AP_BC :
  ∃ (AP BC : ℝ), BC = 14 → dot_BP_CA = 18 → dot_CP_BA = 32 → (AP * BC = 14) :=
by
  -- proof goes here
  sorry

end find_dot_AP_BC_l761_761958


namespace nonzero_sequence_l761_761899

theorem nonzero_sequence {a : ℕ → ℤ} (h1 : a 1 = 1) (h2 : a 2 = 2)
  (hrec : ∀ n : ℕ, (a n * a (n + 1)).even → a (n + 2) = 5 * a (n + 1) - 3 * a n ∧
                   ¬(a n * a (n + 1)).even → a (n + 2) = a (n + 1) - a n) :
  ∀ n : ℕ, a n ≠ 0 :=
by
  sorry

end nonzero_sequence_l761_761899


namespace median_water_slide_times_l761_761314

theorem median_water_slide_times :
  let data := [70, 90, 110, 125, 140, 150, 170, 175, 180, 190, 205, 215, 225, 250, 260, 270]
  List.median data = 185 :=
by
  sorry

end median_water_slide_times_l761_761314


namespace no_such_natural_numbers_l761_761817

theorem no_such_natural_numbers :
  ¬ ∃ a b c d e f g h : ℕ,
    (∃ s1 : Finset ℕ, s1.card = 1 ∧ ∀ x ∈ s1, x % 8 = 0) ∧
    (∃ s2 : Finset ℕ, s2.card = 2 ∧ ∀ x ∈ s2, x % 7 = 0) ∧
    (∃ s3 : Finset ℕ, s3.card = 3 ∧ ∀ x ∈ s3, x % 6 = 0) ∧
    (∃ s4 : Finset ℕ, s4.card = 4 ∧ ∀ x ∈ s4, x % 5 = 0) ∧
    (∃ s5 : Finset ℕ, s5.card = 5 ∧ ∀ x ∈ s5, x % 4 = 0) ∧
    (∃ s6 : Finset ℕ, s6.card = 6 ∧ ∀ x ∈ s6, x % 3 = 0) ∧
    (∃ s7 : Finset ℕ, s7.card = 7 ∧ ∀ x ∈ s7, x % 2 = 0) ∧
    ({a, b, c, d, e, f, g, h} = s1 ∪ s2 ∪ s3 ∪ s4 ∪ s5 ∪ s6 ∪ s7) :=
sorry

end no_such_natural_numbers_l761_761817


namespace next_divisor_after_221_l761_761606

-- Given conditions
def even_four_digit_number (m : ℕ) : Prop :=
  1000 ≤ m ∧ m < 10000 ∧ m % 2 = 0

def is_divisor (a b : ℕ) : Prop :=
  b % a = 0

def sorted_divisors (m : ℕ) : List ℕ :=
  List.filter (λ d, is_divisor d m) (List.range (m + 1))

-- Problem statement as a Lean proof
theorem next_divisor_after_221 (m : ℕ) (h1 : even_four_digit_number m) (h2 : is_divisor 221 m) :
  List.nthLe (sorted_divisors m) (List.indexOf 221 (sorted_divisors m) + 1) sorry = 247 := sorry

end next_divisor_after_221_l761_761606


namespace minimum_cubes_to_match_views_l761_761413

-- Front view condition represented as a 2x2 grid with top right missing
def front_view_condition (cubes: List (ℕ × ℕ × ℕ)) : Prop :=
  (1, 1, _ ∈ cubes) ∧ (1, 2, _ ∈ cubes) ∧ (2, 1, _ ∈ cubes)

-- Side view condition represented as a vertical stack of three cubes
def side_view_condition (cubes: List (ℕ × ℕ × ℕ)) : Prop :=
  (1, _ , 1 ∈ cubes) ∧ (1, _, 2 ∈ cubes) ∧ (1, _, 3 ∈ cubes)

-- Connected cubes condition
def connected_cubes_condition (cubes: List (ℕ × ℕ × ℕ)) : Prop := 
  ∀ c ∈ cubes, ∃ c' ∈ cubes, c ≠ c' ∧ (c.1 = c'.1 ∨ c.2 = c'.2 ∨ c.3 = c'.3) -- Every cube shares a face with another cube

-- The main problem statement
theorem minimum_cubes_to_match_views : ∃ cubes: List (ℕ × ℕ × ℕ), 
  front_view_condition cubes ∧ side_view_condition cubes ∧ connected_cubes_condition cubes ∧ cubes.length = 4 := 
sorry

end minimum_cubes_to_match_views_l761_761413


namespace min_val_range_l761_761531

theorem min_val_range (a b : ℝ) (h : a * (Real.sqrt 3 / 2) + b * (1 / 2) = 1) :
  ∃ m ∈ Set.Icc (-∞ : ℝ) (-1), ∀ x, a * Real.sin x + b * Real.cos x ≥ m := 
begin
  sorry
end

end min_val_range_l761_761531


namespace simplify_f_value_at_neg_31_pi_over_3_l761_761504

-- The function given in the problem
def f (α : Real) : Real :=
  (sin (π - α) * cos (2 * π - α) * tan (π + α)) / (tan (-π - α) * sin (-π - α))

-- Proof Problem 1: Simplify the function to cos(α)
theorem simplify_f (α : Real) : f(α) = cos(α) := by
  sorry

-- Proof Problem 2: Prove the function value for α = -31π/3
theorem value_at_neg_31_pi_over_3 : f(-31 * π / 3) = 1 / 2 := by
  sorry

end simplify_f_value_at_neg_31_pi_over_3_l761_761504


namespace analytical_expression_of_f_monotonic_intervals_of_f_l761_761119

def f (x : ℝ) : ℝ :=
if x > 0 then -x^2 + 2 * x
else if x = 0 then 0
else x^2 + 2 * x

theorem analytical_expression_of_f :
  f x = if x > 0 then -x^2 + 2 * x else if x = 0 then 0 else x^2 + 2 * x :=
sorry

theorem monotonic_intervals_of_f :
  (∀ x y : ℝ, -1 ≤ x ∧ x < y ∧ y ≤ 1 → f x ≤ f y) ∧
  (∀ x y : ℝ, x < y ∧ (y < -1 ∨ 1 < x) → f y < f x) :=
sorry

end analytical_expression_of_f_monotonic_intervals_of_f_l761_761119


namespace circular_garden_area_l761_761325

theorem circular_garden_area :
  ∀ (A B C D : ℝ) (AB DC : ℝ),
    AB = 20 → DC = 15 → D = A + (1 / 3) * (B - A) →
    let AD := (1 / 3) * AB in
    let CD_squared := DC ^ 2 - AD ^ 2 in
    let radius := Real.sqrt CD_squared in
    let area := Real.pi * radius ^ 2 in
    area = 180.556 * Real.pi := 
by
  intro A B C D AB DC hAB hDC hD AD CD_squared radius area
  rw [hAB, hDC, hD]
  simp [AD, CD_squared, radius, area]
  sorry

end circular_garden_area_l761_761325


namespace total_number_of_people_l761_761402

-- Conditions
def number_of_parents : ℕ := 105
def number_of_pupils : ℕ := 698

-- Theorem stating the total number of people is 803 given the conditions
theorem total_number_of_people : 
  number_of_parents + number_of_pupils = 803 :=
by
  sorry

end total_number_of_people_l761_761402


namespace GF_perpendicular_DE_l761_761185

theorem GF_perpendicular_DE 
  (A B C D E F G : Point)
  (h_iso : IsoscelesTriangle A B C)
  (h_D_on_AB : PointOnLineExtension D A B)
  (h_E_on_AC : PointOnLine E A C)
  (h_CE_BD : SegmentEqual (CE E C) (BD B D))
  (h_DE_intersect_F : PointOnLineIntersection F D E B C)
  (h_circle_BDF : CirclePassingThroughPoints (circ B D F G))
  (h_circle_ABC : CirclePassingThroughPoints (circ A B C G))
  : Perpendicular (Line G F) (Line D E) :=
sorry

end GF_perpendicular_DE_l761_761185


namespace minimize_quadratic_nonnegative_x_l761_761067

theorem minimize_quadratic_nonnegative_x :
  ∀ x : ℝ, 0 ≤ x → (∀ y : ℝ, 0 ≤ y → x^2 + 13 * x + 4 ≤ y^2 + 13 * y + 4) → x = 0 :=
by
  intros x hx hmin
  have h₀ : (0:ℝ)^2 + 13 * 0 + 4 = 4 := by norm_num
  have hₓ : x^2 + 13 * x + 4 = (x + 13 / 2)^2 - 153 / 4 := by {
    calc x^2 + 13 * x + 4
        = (x + 13 / 2)^2 - (13 / 2)^2 + 4   : by sorry
    ... = (x + 13 / 2)^2 - 169 / 4 + 4      : by norm_num
    ... = (x + 13 / 2)^2 - 169 / 4 + 16 / 4 : by { congr, norm_num }
    ... = (x + 13 / 2)^2 - 153 / 4          : by norm_num
  }
  sorry

end minimize_quadratic_nonnegative_x_l761_761067


namespace determine_y_l761_761482

theorem determine_y (y : ℝ) (h1 : 0 < y) (h2 : y * (⌊y⌋ : ℝ) = 90) : y = 10 :=
sorry

end determine_y_l761_761482


namespace find_value_of_D_l761_761568

theorem find_value_of_D (C : ℕ) (D : ℕ) (k : ℕ) (h : C = (10^D) * k) (hD : k % 10 ≠ 0) : D = 69 := by
  sorry

end find_value_of_D_l761_761568


namespace Vasya_numbers_l761_761377

theorem Vasya_numbers : ∃ (x y : ℝ), x + y = xy ∧ xy = x / y ∧ (x, y) = (1 / 2, -1) :=
by
  sorry

end Vasya_numbers_l761_761377


namespace range_of_f_inv_l761_761308

noncomputable def f (x : ℝ) : ℝ := 2 - Real.logb 2 x

theorem range_of_f_inv :
  { y : ℝ | ∃ x : ℝ, f x = y ∧ 1 < y } = set.Ioo 0 2 :=
by
  sorry

end range_of_f_inv_l761_761308


namespace direction_vector_correct_l761_761299

noncomputable def projection_matrix : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![1/18,  1/9, -1/6],
    ![ 1/9,  1/6, -1/3],
    ![-1/6, -1/3,  2/3]]

def direction_vector (v : Vector ℚ (Fin 3)) (a b c : ℚ) : Prop :=
  ∃ k : ℚ, k ≠ 0 ∧ (v = k • ![a, b, c])

theorem direction_vector_correct :
  direction_vector (projection_matrix.mulVec ![1, 0, 0]) 1 2 -3 ∧
  (1 > 0 ∧ Int.gcd 1 (Int.gcd 2 (-3)) = 1) := by
  sorry

end direction_vector_correct_l761_761299


namespace intersection_A_B_l761_761123

def A := {x : ℝ | x^2 - x - 2 ≤ 0}
def B := {y : ℝ | ∃ x : ℝ, y = 2^x}

theorem intersection_A_B :
  A ∩ {x : ℝ | x > 0} = {x : ℝ | 0 < x ∧ x ≤ 2} :=
sorry

end intersection_A_B_l761_761123


namespace complex_number_problem_l761_761849

theorem complex_number_problem {z : ℂ} (h : (z - 2 * complex.I) / z = 2 + complex.I) :
  z.im = -1 ∧ z ^ 6 = -8 * complex.I :=
by
  sorry

end complex_number_problem_l761_761849


namespace expression_in_parentheses_l761_761564

theorem expression_in_parentheses (x y : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) :
  ∃ expr : ℝ, xy * expr = -x^3 * y^2 ∧ expr = -x^2 * y :=
by
  sorry

end expression_in_parentheses_l761_761564


namespace interquartile_range_baskets_l761_761654

-- Define the list of basket prices
def basket_prices : List ℝ := [3, 5, 6, 7, 8, 9, 10, 12, 13, 15, 18, 20]

-- Define the function to calculate the interquartile range
def interquartileRange (prices : List ℝ) : ℝ :=
  let sorted_prices := prices.sorted
  let n := sorted_prices.length
  let lower_half := sorted_prices.take (n / 2)
  let upper_half := sorted_prices.drop (n / 2)
  let q1 := lower_half.get ((lower_half.length - 1) / 2) + lower_half.get (lower_half.length / 2) / 2
  let q3 := upper_half.get ((upper_half.length - 1) / 2) + upper_half.get (upper_half.length / 2) / 2
  q3 - q1

-- Define the theorem we want to prove
theorem interquartile_range_baskets : interquartileRange basket_prices = 7.5 := by
  sorry

end interquartile_range_baskets_l761_761654


namespace increasing_function_unique_root_proof_l761_761878

noncomputable def increasing_function_unique_root (f : ℝ → ℝ) :=
  (∀ x y : ℝ, x < y → f x ≤ f y) -- condition for increasing function
  ∧ ∃! x : ℝ, f x = 0 -- exists exactly one root

theorem increasing_function_unique_root_proof
  (f : ℝ → ℝ)
  (h_inc : ∀ x y : ℝ, x < y → f x ≤ f y)
  (h_ex : ∃ x : ℝ, f x = 0) :
  ∃! x : ℝ, f x = 0 := sorry

end increasing_function_unique_root_proof_l761_761878


namespace candies_per_basket_l761_761772

noncomputable def chocolate_bars : ℕ := 5
noncomputable def mms : ℕ := 7 * chocolate_bars
noncomputable def marshmallows : ℕ := 6 * mms
noncomputable def total_candies : ℕ := chocolate_bars + mms + marshmallows
noncomputable def baskets : ℕ := 25

theorem candies_per_basket : total_candies / baskets = 10 :=
by
  sorry

end candies_per_basket_l761_761772


namespace false_proposition_A_l761_761497

theorem false_proposition_A 
  (a b : ℝ)
  (root1_eq_1 : ∀ x, x^2 + a * x + b = 0 → x = 1)
  (root2_eq_3 : ∀ x, x^2 + a * x + b = 0 → x = 3)
  (sum_of_roots_eq_2 : -a = 2)
  (opposite_sign_roots : ∀ x1 x2, x1 * x2 < 0) :
  ∃ prop, prop = "A" :=
sorry

end false_proposition_A_l761_761497


namespace product_of_two_numbers_l761_761679

theorem product_of_two_numbers (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : a * b = 875 :=
sorry

end product_of_two_numbers_l761_761679


namespace set_B_forms_triangle_l761_761393

theorem set_B_forms_triangle (a b c : ℝ) (h1 : a = 25) (h2 : b = 24) (h3 : c = 7):
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  sorry

end set_B_forms_triangle_l761_761393


namespace wall_ratio_l761_761318

theorem wall_ratio (V : ℝ) (B : ℝ) (H : ℝ) (x : ℝ) (L : ℝ) :
  V = 12.8 →
  B = 0.4 →
  H = 5 * B →
  L = x * H →
  V = B * H * L →
  x = 4 ∧ L / H = 4 :=
by
  intros hV hB hH hL hVL
  sorry

end wall_ratio_l761_761318


namespace combined_degrees_l761_761266

variable (Summer_deg Jolly_deg : ℕ)

def Summer_has_150_degrees := Summer_deg = 150

def Summer_has_5_more_degrees_than_Jolly := Summer_deg = Jolly_deg + 5

theorem combined_degrees (h1 : Summer_has_150_degrees Summer_deg) (h2 : Summer_has_5_more_degrees_than_Jolly Summer_deg Jolly_deg) :
  Summer_deg + Jolly_deg = 295 :=
by
  sorry

end combined_degrees_l761_761266


namespace matrix_multiplication_l761_761461

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 1], ![4, -2]]

def matrix2 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![7, -3], ![2, 2]]

def product_matrix : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![23, -7], ![24, -16]]

theorem matrix_multiplication :
  matrix1 ⬝ matrix2 = product_matrix := by
  sorry

end matrix_multiplication_l761_761461


namespace num_accompanying_year_2022_l761_761231

theorem num_accompanying_year_2022 : 
  ∃ N : ℤ, (N = 2) ∧ 
    (∀ n : ℤ, (100 * n + 22) % n = 0 ∧ 10 ≤ n ∧ n < 100 → n = 11 ∨ n = 22) :=
by 
  sorry

end num_accompanying_year_2022_l761_761231


namespace points_P_Q_R_collinear_l761_761518

-- Define the setup of the problem
variable (α : Type)
variable [ordered_field α] [plane α] {A B C D E F L M N P Q R : point α}

-- Incircle touches sides at D, E, F
def incircle_touches_sides (A B C D E F : point α) : Prop :=
  touches_incircle A B C D E F

-- Definition for reflecting a point over a line
def reflection_over_line (p l : point α) : point α := reflection p l

-- Relevant points definitions
def L := reflection_over_line D (line_through E F)
def M := reflection_over_line E (line_through F D)
def N := reflection_over_line F (line_through D E)

-- Lines intersecting sides definition
def intersect_side (p q : point α) (l : line α) : point α := intersection_point p q l

def P := intersect_side (line_through A L) (line_through B C)
def Q := intersect_side (line_through B M) (line_through C A)
def R := intersect_side (line_through C N) (line_through A B)

-- Definition of collinear points
def collinear (p q r : point α) : Prop := lies_on_same_line p q r

-- Lean statement
theorem points_P_Q_R_collinear
  (A B C D E F : point α)
  (h1 : ∀ {X : point α}, X ≠ A ∧ X ≠ B ∧ X ≠ C)
  (h2 : incircle_touches_sides A B C D E F)
  (L := reflection_over_line D (line_through E F))
  (M := reflection_over_line E (line_through F D))
  (N := reflection_over_line F (line_through D E))
  (P := intersect_side (line_through A L) (line_through B C))
  (Q := intersect_side (line_through B M) (line_through C A))
  (R := intersect_side (line_through C N) (line_through A B)) :
  collinear P Q R := 
  sorry

end points_P_Q_R_collinear_l761_761518


namespace like_term_example_l761_761446

def is_like_term (t1 t2 : String) : Prop :=
  let letters_and_exponents (term : String) := -- function that parses a term into its letters and exponents
    sorry
  letters_and_exponents t1 = letters_and_exponents t2

theorem like_term_example :
  is_like_term "3a^2b" "a^2b" :=
by
  -- Expected parsed forms (for illustrative purposes, not executable)
  -- letters_and_exponents "3a^2b" = [("a", 2), ("b", 1)]
  -- letters_and_exponents "a^2b" = [("a", 2), ("b", 1)]
  sorry

end like_term_example_l761_761446


namespace vasya_numbers_l761_761380

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : 
  (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) ∨ (x = y ∧ x = 0) :=
by 
  -- Placeholder to show where the proof would go
  sorry

end vasya_numbers_l761_761380


namespace max_salary_21_players_l761_761426

noncomputable def max_player_salary (num_players : ℕ) (min_salary : ℕ) (total_salary_cap : ℕ) : ℕ :=
  total_salary_cap - (num_players - 1) * min_salary

theorem max_salary_21_players :
  max_player_salary 21 15000 700000 = 400000 :=
by simp [max_player_salary]; norm_num; sorry

end max_salary_21_players_l761_761426


namespace number_of_factors_l761_761226

-- Define the LCM of the numbers from 1 to 20
noncomputable def L : ℕ := Nat.lcmList (List.range' 1 20)

-- State the prime factorizations of L
noncomputable def L_prime_factors : List (ℕ × ℕ) := [(2, 4), (3, 2), (5, 1), (7, 1), (11, 1), (13, 1), (17, 1), (19, 1)]

-- The problem statement asserts that the number of positive factors of L that are divisible by exactly 18 of the numbers 
-- from 1 to 20 is 13.
theorem number_of_factors (hL : L = 2^4 * 3^2 * 5 * 7 * 11 * 13 * 17 * 19) :
  (∃ (n : ℕ), (n ∣ L ∧ ∀ m ∈ List.range' 1 21, Prime m → m ∣ L → (¬m ∣ n ↔ List.count m (List.range' 1 21) = 2))) := 13 := by
  sorry

end number_of_factors_l761_761226


namespace trig_expression_value_l761_761453

theorem trig_expression_value :
  let sin_1200 := -Real.sin (60 * Real.pi / 180),
      cos_1290 := -Real.cos (30 * Real.pi / 180),
      cos_1020 := Real.cos (60 * Real.pi / 180),
      sin_1050 := Real.sin (30 * Real.pi / 180),
      tan_945 := 1
  in sin_1200 * cos_1290 + cos_1020 * sin_1050 + tan_945 = 2 :=
by
  sorry

end trig_expression_value_l761_761453


namespace triangle_arithmetic_sequence_l761_761189

-- Define the proof problem
theorem triangle_arithmetic_sequence 
  (A B C : ℝ) (a b c : ℝ)
  (h : (1 / (a + b) + 1 / (b + c) = 3 / (a + b + c)))
  (h_triangle : A + B + C = 180) 
  (h_a : a = sqrt (c^2 + b^2 - 2 * b * c * cos A))
  (h_b : b = sqrt (a^2 + c^2 - 2 * a * c * cos B))
  (h_c : c = sqrt (a^2 + b^2 - 2 * a * b * cos C)) :
  A = 60 ∧ B = 60 ∧ C = 60 :=
  sorry

end triangle_arithmetic_sequence_l761_761189


namespace probability_at_least_half_girls_l761_761609

theorem probability_at_least_half_girls (n : ℕ) (hn : n = 6) :
  (probability (λ (s : vector bool n), s.foldr (λ b acc, if b then acc + 1 else acc) 0 ≥ n/2))
  = 21 / 32 := by
  sorry

end probability_at_least_half_girls_l761_761609


namespace pasta_sauce_salad_time_ratio_l761_761604

-- Variables and definitions
variables (initial_temp boil_temp time_per_degree cooking_time total_time : ℕ)
-- Assumptions based on the conditions
def initial_temp := 41
def boil_temp := 212
def time_per_degree := 3
def cooking_time := 12
def total_time := 73

-- Define the proof problem
theorem pasta_sauce_salad_time_ratio :
  (total_time - (boil_temp - initial_temp) / time_per_degree - cooking_time) = total_time - ((boil_temp - initial_temp) / time_per_degree  + cooking_time) ∧ 
  (((total_time - ((boil_temp - initial_temp) / time_per_degree + cooking_time)) : ℚ) / (cooking_time : ℚ) = 1 / 3) := 
by 
  sorry

end pasta_sauce_salad_time_ratio_l761_761604


namespace conjugate_of_div_l761_761283

theorem conjugate_of_div :
  let i := Complex.I
  let z := (1 + 2 * i) / i
  let conj_z := Complex.conj z
  conj_z = 2 + i :=
by 
  sorry

end conjugate_of_div_l761_761283


namespace find_prob_indep_l761_761157

variable {Ω : Type*} [ProbabilitySpace Ω]

def indep (A B : Event Ω) : Prop := ∀ ω, Pr (A ∩ B) = (Pr A) * (Pr B)

theorem find_prob_indep (A B : Event Ω) (hA : Pr A = 5 / 7) (hB : Pr B = 2 / 5) (hIndep : indep A B) : 
  Pr (A ∩ B) = 2 / 7 := by
  sorry

end find_prob_indep_l761_761157


namespace AP_AQ_leq_AB_l761_761184

-- Definitions of conditions
variable (A B C D E F P Q : Point) 
variable (l BC : Line)
variable [HasMem A l] [HasMem D l] 
variable [Parallel l BC]
variable [IsIsoscelesTriangle ABC AB AC]
variable [Perpendicular A BD E] [Perpendicular A CD F]
variable [OnImage l E P] [OnImage l F Q]
 
-- Statement of the proof
theorem AP_AQ_leq_AB 
  (h_eq: AB = AC)
  (h_parallel: ∀ x : Point, (x ∈ l) ↔ (x ∈ l.parallel BC))
  (h_perpendicular1: ∀ x : Point, Perpendicular A BD x ↔ (x = E))
  (h_perpendicular2: ∀ x : Point, Perpendicular A CD x ↔ (x = F))
  (h_image1: ∀ x : Point, OnImage l x P ↔ (x = E))
  (h_image2: ∀ x : Point, OnImage l x Q ↔ (x = F)) :
  AP + AQ ≤ AB :=
by
  sorry

end AP_AQ_leq_AB_l761_761184


namespace prod_fraction_eq_l761_761008

theorem prod_fraction_eq :
  (∏ n in Finset.range 30, (n + 6) / (n + 1)) = 326284 :=
by
  sorry

end prod_fraction_eq_l761_761008


namespace largest_power_of_two_divides_n_l761_761813

noncomputable def largestPowerOfTwo (n : ℤ) : ℤ :=
  let v2 := PadicValuation 2 (padicVal 2 n)
  v2

theorem largest_power_of_two_divides_n (a b : ℕ) (ha : a = 15) (hb : b = 13) :
  largestPowerOfTwo (a^4 - b^4) = 16 := by
  -- Proof would go here
  sorry

end largest_power_of_two_divides_n_l761_761813


namespace find_m_if_divisible_by_11_l761_761569

theorem find_m_if_divisible_by_11 : ∃ m : ℕ, m < 10 ∧ (734000000 + m*100000 + 8527) % 11 = 0 ↔ m = 6 :=
by {
    sorry
}

end find_m_if_divisible_by_11_l761_761569


namespace T_5_value_l761_761094

-- Definitions based on the given conditions
def arithmetic_sequence (n : ℕ) : ℕ := n + 1
def sum_arithmetic_sequence (n : ℕ) : ℚ := ↑(n * (n + 1)) / 2
def reciprocal_sum_arithmetic_sequence (n : ℕ) : ℚ := 2 / (↑n * (↑n + 1))

noncomputable def T (n : ℕ) : ℚ :=
  ∑ i in Finset.range n, reciprocal_sum_arithmetic_sequence (i + 1)

-- The proof problem statement
theorem T_5_value : T 5 = 5 / 3 := by
  sorry

end T_5_value_l761_761094


namespace vasya_numbers_l761_761354

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 := 
by 
  sorry

end vasya_numbers_l761_761354


namespace inequality_proof_l761_761088

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  (1 / (1 - x^2)) + (1 / (1 - y^2)) ≥ (2 / (1 - x * y)) :=
by sorry

end inequality_proof_l761_761088


namespace bisect_proof_l761_761580

variable {A B C D H K : Type} [Point : Type]
variables [Triangle : triangle A B C]
variables [Altitude : altitude A D B C] [Orthocenter : orthocenter H A B C]
variables [Center : center D K H B]

noncomputable def bisects : Prop := 
  let J := midpoint A C in
  ∃ K, center D K H B ∧ line DK ∧ DK ∩ AC = J

theorem bisect_proof : bisects :=
sorry

end bisect_proof_l761_761580


namespace total_jumps_l761_761662

def taehyung_jumps_per_day : ℕ := 56
def taehyung_days : ℕ := 3
def namjoon_jumps_per_day : ℕ := 35
def namjoon_days : ℕ := 4

theorem total_jumps : taehyung_jumps_per_day * taehyung_days + namjoon_jumps_per_day * namjoon_days = 308 :=
by
  sorry

end total_jumps_l761_761662


namespace unique_solution_for_a_eq_1_l761_761498

def equation (a x : ℝ) : Prop :=
  5^(x^2 - 6 * a * x + 9 * a^2) = a * x^2 - 6 * a^2 * x + 9 * a^3 + a^2 - 6 * a + 6

theorem unique_solution_for_a_eq_1 :
  (∃! x : ℝ, equation 1 x) ∧ 
  (∀ a : ℝ, (∃! x : ℝ, equation a x) → a = 1) :=
sorry

end unique_solution_for_a_eq_1_l761_761498


namespace find_lambda_l761_761126

open Real EuclideanSpace

noncomputable theory

def vec_a : ℝ × ℝ := (3, 4)
def vec_b : ℝ × ℝ := (4, 3)
def vec_c (λ : ℝ) : ℝ × ℝ := (3 * λ - 4, 4 * λ - 3)

-- Defining the inner product for 2D vectors
def inner_product (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2
-- Defining the norm (magnitude) of a 2D vector
def norm (v : ℝ × ℝ) : ℝ := real.sqrt (inner_product v v)

-- The problem statement that needs to be proved
theorem find_lambda : ∃ (λ : ℝ), λ = -1 ∧
  (inner_product (vec_c λ) vec_a) / (norm (vec_c λ) * norm vec_a) =
  (inner_product (vec_c λ) vec_b) / (norm (vec_c λ) * norm vec_b) :=
begin
  use -1,
  split,
  { -- Proving λ = -1
    refl },
  { -- The angle condition
    sorry
  }
end

end find_lambda_l761_761126


namespace find_abc_l761_761975

theorem find_abc (a b c : ℝ)
  (h1 : ∀ x : ℝ, (x < -6 ∨ (|x - 31| ≤ 1)) ↔ (x - a) * (x - b) / (x - c) ≤ 0)
  (h2 : a < b) :
  a + 2 * b + 3 * c = 76 :=
sorry

end find_abc_l761_761975


namespace binomial_sum_identity_l761_761224

-- Let n be a natural number.
variable (n : ℕ)

-- Theorem statement
theorem binomial_sum_identity (n : ℕ) :
  (∑ i in Finset.range (2 * n + 1), (-1) ^ i * (Nat.choose (2 * n) i) ^ 2) = 
  (-1) ^ n * Nat.choose (2 * n) n := 
sorry

end binomial_sum_identity_l761_761224


namespace line_through_orthocenter_l761_761617

open EuclideanGeometry

-- Definitions for the points and shapes involved
variables (A B C D P Q X Y : Point)

-- Rectangle condition
def is_rectangle (A B C D : Point) : Prop :=
  is_parallelogram A B C D ∧
  angle_eq (A, B, C) (B, C, D) π / 2 ∧
  angle_eq (C, D, A) (D, A, B) π / 2

-- Orthogonal projection definition
def orthogonal_projection (P Q C D : Point) : Prop :=
  is_foot Q P (line C D)

-- Circle passing through three points
def passes_through_circle (A B Q : Point) : Set Point :=
  { P : Point | dist P A = dist P B ∧ dist P B = dist P Q }

-- Orthocenter definition for triangle
def orthocenter (C D P : Point) (H : Point) : Prop :=
  ∃ K, H = intersection_point (altitude C D P) (altitude D P C)

-- Problem statement
theorem line_through_orthocenter
  (h_rect : is_rectangle A B C D)
  (h_P_on_AB : P ∈ line_segment A B)
  (h_Q_proj : orthogonal_projection P Q C D)
  (h_circle : ∃ cir, passes_through_circle A B Q cir)
  (h_X : ∃ cir, X ∈ intersection_point cir (line A D))
  (h_Y : ∃ cir, Y ∈ intersection_point cir (line B C)) :
  ∃ H, orthocenter C D P H ∧ H ∈ line X Y :=
sorry

end line_through_orthocenter_l761_761617


namespace M_eq_N_l761_761124

def M : set ℤ := {-1, 0, 1}
def N : set ℤ := {x | ∃ a b ∈ M, x = a * b}

theorem M_eq_N : M = N :=
sorry

end M_eq_N_l761_761124


namespace value_of_a_eq_half_l761_761865

theorem value_of_a_eq_half
  (n : ℕ)
  (a : ℝ)
  (h_expansion : (ax + 1)^n = a_0 + a_1 * x + a_2 * x^2 + ... + a_n * x^n)
  (h_a1 : a_1 = 4)
  (h_a2 : a_2 = 7) :
  a = 1 / 2 :=
sorry

end value_of_a_eq_half_l761_761865


namespace bird_count_difference_l761_761719

theorem bird_count_difference :
    ∃ B1 B2 B3 : ℕ,
    B1 = 300 ∧
    B3 = B1 - 200 ∧
    B1 + B2 + B3 = 1300 ∧
    B2 - B1 = 600 :=
begin
  sorry
end

end bird_count_difference_l761_761719


namespace ellipse_standard_eq_hyperbola_standard_eq_l761_761404

open Real

noncomputable def ellipse_lhs : ℝ → ℝ → ℝ := λ (x y : ℝ), 9 * x^2 + 4 * y^2
noncomputable def hyperbola_lhs : ℝ → ℝ → ℝ := λ (x y : ℝ), x^2 / 2 - y^2

theorem ellipse_standard_eq :
  (∃ (x y : ℝ), (ellipse_lhs x y = 36) ∧ (x, y) = (-2, 3)) →
  ∀ (x y : ℝ), x^2 / 10 + y^2 / 15 = 1 :=
sorry

theorem hyperbola_standard_eq :
  (∃ (x y : ℝ), (hyperbola_lhs x y = 1) ∧ (x, y) = (2, -2)) →
  ∀ (x y : ℝ), y^2 / 2 - x^2 / 4 = 1 :=
sorry

end ellipse_standard_eq_hyperbola_standard_eq_l761_761404


namespace remainder_of_3x_plus_5y_l761_761993

-- Conditions and parameter definitions
def x (k : ℤ) := 13 * k + 7
def y (m : ℤ) := 17 * m + 11

-- Proof statement
theorem remainder_of_3x_plus_5y (k m : ℤ) : (3 * x k + 5 * y m) % 221 = 76 := by
  sorry

end remainder_of_3x_plus_5y_l761_761993


namespace digits_sum_l761_761595

noncomputable def solve_digits (P Q R : ℕ) :=
  P + Q + R

theorem digits_sum : 
  ∃ (P Q R : ℕ), (1 ≤ P ∧ P ≤ 9) ∧ (1 ≤ Q ∧ Q ≤ 9) ∧ (1 ≤ R ∧ R ≤ 9) 
  ∧ (Q * R + 2 * (111 * P) = 2022) ∧ (solve_digits P Q R = 15) :=
begin
  sorry
end

end digits_sum_l761_761595


namespace fruit_display_total_l761_761688

-- Define the number of bananas
def bananas : ℕ := 5

-- Define the number of oranges, twice the number of bananas
def oranges : ℕ := 2 * bananas

-- Define the number of apples, twice the number of oranges
def apples : ℕ := 2 * oranges

-- Define the total number of fruits
def total_fruits : ℕ := bananas + oranges + apples

-- Theorem statement: Prove that the total number of fruits is 35
theorem fruit_display_total : total_fruits = 35 := 
by
  -- proof will be skipped, but all necessary conditions and definitions are included to ensure the statement is valid
  sorry

end fruit_display_total_l761_761688


namespace segment_PS_length_correct_l761_761177

-- Defining the points and properties of the quadrilateral
variables (P Q R S T : ℝ)
variables (hPQ : P = 7) (hQR : Q = 10) (hRS : R = 25)
variables (hAngleQ : ∀ (a b c : ℝ), angle a b c = π/2) (hAngleR : ∀ (a b c : ℝ), angle a b c = π/2)
variables (hParallelPTQR : T = Q)

noncomputable def segment_PS_length : ℝ := sqrt (10^2 + 15^2)

theorem segment_PS_length_correct : segment_PS_length = 5 * sqrt 13 := by
  -- Utilizing the conditions provided
  have hPT : T = 10 := by sorry
  have hTS : S - T = 15 := by sorry
  have hPS : sqrt (hPT^2 + hTS^2) = sqrt (10^2 + 15^2) := by sorry
  show sqrt (10^2 + 15^2) = 5 * sqrt 13 from sorry

end segment_PS_length_correct_l761_761177


namespace team_total_score_l761_761471

theorem team_total_score (Connor_score Amy_score Jason_score : ℕ)
  (h1 : Connor_score = 2)
  (h2 : Amy_score = Connor_score + 4)
  (h3 : Jason_score = 2 * Amy_score) :
  Connor_score + Amy_score + Jason_score = 20 :=
by
  sorry

end team_total_score_l761_761471


namespace at_most_three_real_lambda_l761_761219

-- Given conditions: P and Q are relatively prime nonconstant polynomials in ℝ[x]
variables {P Q : Polynomial ℝ}

-- Definition of "relatively prime" for polynomials
def relatively_prime (P Q : Polynomial ℝ) : Prop := P.gcd Q = 1

-- Main theorem statement
theorem at_most_three_real_lambda (hP : P.degree > 0) (hQ : Q.degree > 0) (h_rel_prime : relatively_prime P Q) :
  {λ : ℝ | ∃ R : Polynomial ℝ, P + λ • Q = R ^ 2 }.finite ∧ (λs : {λ : ℝ | ∃ R : Polynomial ℝ, P + λ • Q = R ^ 2}.toFinset).card ≤ 3 :=
sorry

end at_most_three_real_lambda_l761_761219


namespace repeating_decimal_denominators_l761_761269

theorem repeating_decimal_denominators (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) (h4 : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) (h5 : a ≠ 9 ∨ b ≠ 9 ∨ c ≠ 9) :
  ∃ D : Finset ℕ, D.card = 7 ∧ ∀ d ∈ D, ∃ n : ℕ, (n / d).denominator = d ∧ n % d ≠ 0 :=
sorry

end repeating_decimal_denominators_l761_761269


namespace find_sin_beta_l761_761525

-- Define the conditions
variables (α β : ℝ)
variables (h1 : 0 < α ∧ α < π / 2)
variables (h2 : -π / 2 < β ∧ β < 0)
variables (h3 : cos (α - β) = -3 / 5)
variables (h4 : tan α = 4 / 3)

-- State the theorem
theorem find_sin_beta : sin β = -24 / 25 :=
by
  -- Proof goes here
  sorry

end find_sin_beta_l761_761525


namespace sin_C_in_right_triangle_l761_761167

theorem sin_C_in_right_triangle (k : ℝ) (A B C : RealPlanePoint)
  (h_angle_A : angle A B C = π / 2)
  (h_tan_C : tan (angle B A C) = 3 / 2) :
  sin (angle B A C) = 3 * sqrt 13 / 13 :=
sorry

end sin_C_in_right_triangle_l761_761167


namespace monotonicity_intervals_f_max_min_values_f_range_abs_inverses_l761_761890

noncomputable def f (x a : ℝ) := x^2 - x * |x - a| - 3 * a

theorem monotonicity_intervals_f (a : ℝ) (h : a = 1) :
  ∃ c : ℝ, ∀ x : ℝ, (x < c → f x a < f (c + ε) a) ∧ (x > c → f x a > f (c + ε) a) :=
sorry

theorem max_min_values_f {a : ℝ} (h : a > 0) :
  ∃ (ξ η : ℝ), f ξ a = min (f 0 a) (f 3 a) ∧ f η a = max (f 0 a) (f 3 a) :=
sorry

theorem range_abs_inverses (a : ℝ) (h1 : 0 < a) (h2 : a < 3) :
  ∃ x1 x2 : ℝ, f x1 a = 0 ∧ f x2 a = 0 ∧ x1 ≠ x2 ∧ 
  ∃ lower upper : ℝ, 1 < lower ∧ ∀ (x1 x2 : ℝ), f x1 a = 0 → f x2 a = 0 → x1 ≠ x2 → 
    lower ≤ | 1 / x1 - 1 / x2 | ∧ | 1 / x1 - 1 / x2 | < upper :=
sorry

end monotonicity_intervals_f_max_min_values_f_range_abs_inverses_l761_761890


namespace rain_three_days_l761_761306

def P (event : Prop) := ℝ

variables (F S S_sun : Prop)

-- defining probabilities
def P_F : P F := 3 / 10
def P_S : P S := 1 / 2
def P_S_sun_given_S : P S_sun := 7 / 10

theorem rain_three_days :
  (P F * P S * P S_sun = 21 / 200) :=
by
  sorry

end rain_three_days_l761_761306


namespace max_distance_is_sqrt_3_l761_761186

noncomputable def line_l (ρ θ : ℝ) : Prop :=
  sqrt 2 * ρ * cos (θ + π / 4) = 1

noncomputable def curve_C (α x y : ℝ) : Prop :=
  x = 1 + sqrt 3 * cos α ∧ y = sin α

noncomputable def max_distance_from_curve_to_line_l : ℝ :=
  sqrt 3

theorem max_distance_is_sqrt_3 :
  ∀ (M : ℝ × ℝ), 
  (∃ α : ℝ, curve_C α M.1 M.2) →
  (∀ ρ θ : ℝ, line_l ρ θ → True) →
  ∃ d : ℝ, d = sqrt 3 :=
sorry

end max_distance_is_sqrt_3_l761_761186


namespace remaining_movies_l761_761685

-- Definitions based on the problem's conditions
def total_movies : ℕ := 8
def watched_movies : ℕ := 4

-- Theorem statement to prove that you still have 4 movies left to watch
theorem remaining_movies : total_movies - watched_movies = 4 :=
by
  sorry

end remaining_movies_l761_761685


namespace bin101_to_decimal_l761_761033

-- Define the binary representation of 101 (base 2)
def bin101 : ℕ := 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- State the theorem that asserts the decimal value of 101 (base 2) is 5
theorem bin101_to_decimal : bin101 = 5 := by
  sorry

end bin101_to_decimal_l761_761033


namespace c_investment_ratio_l761_761438

-- Conditions as definitions
variables (x : ℕ) (m : ℕ) (total_profit a_share : ℕ)
variables (h_total_profit : total_profit = 19200)
variables (h_a_share : a_share = 6400)

-- Definition of total investment (investments weighted by time)
def total_investment (x m : ℕ) : ℕ :=
  (12 * x) + (6 * 2 * x) + (4 * m * x)

-- Definition of A's share in terms of total investment
def a_share_in_terms_of_total_investment (x : ℕ) (total_investment : ℕ) (total_profit : ℕ) : ℕ :=
  (12 * x * total_profit) / total_investment

-- The theorem stating the ratio of C's investment to A's investment
theorem c_investment_ratio (x m total_profit a_share : ℕ) (h_total_profit : total_profit = 19200)
  (h_a_share : a_share = 6400) (h_a_share_eq : a_share_in_terms_of_total_investment x (total_investment x m) total_profit = a_share) :
  m = 3 :=
by sorry

end c_investment_ratio_l761_761438


namespace cupcakes_per_package_l761_761234

theorem cupcakes_per_package (total_cupcakes baked eaten packages : ℕ) (h1 : baked = 71) (h2 : eaten = 43) (h3 : packages = 4) :
  (baked - eaten) / packages = 7 :=
by
  rw [h1, h2, h3]
  sorry

end cupcakes_per_package_l761_761234


namespace average_sales_is_96_l761_761270

-- Definitions for the sales data
def january_sales : ℕ := 110
def february_sales : ℕ := 80
def march_sales : ℕ := 70
def april_sales : ℕ := 130
def may_sales : ℕ := 90

-- Number of months
def num_months : ℕ := 5

-- Total sales calculation
def total_sales : ℕ := january_sales + february_sales + march_sales + april_sales + may_sales

-- Average sales per month calculation
def average_sales_per_month : ℕ := total_sales / num_months

-- Proposition to prove that the average sales per month is 96
theorem average_sales_is_96 : average_sales_per_month = 96 :=
by
  -- We use 'sorry' here to skip the proof, as the problem requires only the statement
  sorry

end average_sales_is_96_l761_761270


namespace count_3_digit_multiples_of_25_not_75_l761_761140

theorem count_3_digit_multiples_of_25_not_75 : 
  (finset.Icc 100 975).filter (λ n, n % 25 = 0 ∧ n % 75 ≠ 0).card = 24 :=
by
  sorry

end count_3_digit_multiples_of_25_not_75_l761_761140


namespace binary_to_decimal_101_l761_761028

theorem binary_to_decimal_101 : ∑ (i : Fin 3), (Nat.digit 2 ⟨i, sorry⟩ (list.nth_le [1, 0, 1] i sorry)) * (2 ^ i) = 5 := 
  sorry

end binary_to_decimal_101_l761_761028


namespace perimeter_of_triangle_ABF2_l761_761540

open Real

noncomputable def perimeter_triangle_ABF2 
  (a b c : ℝ)
  (h1 : b = 4)
  (h2 : c = (3/5) * a)
  (h3 : a^2 = b^2 + c^2) : ℝ :=
  4 * a

theorem perimeter_of_triangle_ABF2 : perimeter_triangle_ABF2 5 4 3 4 (3/5 * 5) (5^2 = 4^2 + (3/5 * 5)^2) = 20 :=
by sorry

end perimeter_of_triangle_ABF2_l761_761540


namespace purely_imaginary_iff_x_equals_one_l761_761158

theorem purely_imaginary_iff_x_equals_one (x : ℝ) :
  ((x^2 - 1) + (x + 1) * Complex.I).re = 0 → x = 1 :=
by
  sorry

end purely_imaginary_iff_x_equals_one_l761_761158


namespace calculate_A_minus_B_l761_761436

variable (A B : ℝ)
variable (h1 : A + B + B = 814.8)
variable (h2 : 10 * B = A)

theorem calculate_A_minus_B : A - B = 611.1 :=
by
  sorry

end calculate_A_minus_B_l761_761436


namespace ellipse_equation_line_slope_elliptic_l761_761856

-- Define the ellipse and its properties including eccentricity
variables {a b x y : ℝ}
def ellipse (a b : ℝ) := (x : ℝ) -> (y : ℝ) -> (x^2 / a^2 + y^2 / b^2 = 1)

def eccentricity (a b : ℝ) := (c : ℝ) -> (c = sqrt (a^2 - b^2)) -> (c / a = sqrt(3) / 2)

-- Conditions given in the problem
variables (area_quad : ℝ)
axiom area_condition : (a * b * 4 = 16) -- Area of the quadrilateral
axiom eccentricity_condition : eccentricity a b (sqrt (a^2 - b^2))

-- Definitions for geometric sequence condition
variables {P M N : ℝ} (slope : ℝ)
def geo_seq_condition (k : ℝ) (P M N : ℝ) : Prop :=
|PM|^2 = |PN| * |MN|

-- Prove the equation of the ellipse and the slope
theorem ellipse_equation : ellipse 4 2 :=
sorry

theorem line_slope_elliptic : exists l, geo_seq_condition l P M N ∧ l = 1 / (4 * sqrt(5)) :=
sorry

end ellipse_equation_line_slope_elliptic_l761_761856


namespace find_original_faculty_count_l761_761640

variable (F : ℝ)
variable (final_count : ℝ := 195)
variable (first_year_reduction : ℝ := 0.075)
variable (second_year_increase : ℝ := 0.125)
variable (third_year_reduction : ℝ := 0.0325)
variable (fourth_year_increase : ℝ := 0.098)
variable (fifth_year_reduction : ℝ := 0.1465)

theorem find_original_faculty_count (h : F * (1 - first_year_reduction)
                                        * (1 + second_year_increase)
                                        * (1 - third_year_reduction)
                                        * (1 + fourth_year_increase)
                                        * (1 - fifth_year_reduction) = final_count) :
  F = 244 :=
by sorry

end find_original_faculty_count_l761_761640


namespace hexagon_side_length_proof_l761_761242

noncomputable def hexagon_side_length (d : ℝ) (h : d = 10) : ℝ := 
  let side_length := d * 2 / real.sqrt 3 in 
  side_length

theorem hexagon_side_length_proof : hexagon_side_length 10 (by refl) = 20 * real.sqrt 3 / 3 := 
  by {
    unfold hexagon_side_length,
    rw [←mul_assoc, mul_div_mul_left, div_eq_mul_inv, mul_assoc, real.mul_sqrt, mul_div_assoc, real.sqrt_eq_rpow, real.sq_sqrt],
    any_goals {norm_num [real.rpow_nat_cast]},
    {
      convert congr_arg (λ x, x * (1 : ℝ)) _,
      unfold_coes,
      simp [real.div_sqrt],
    }
  }

end hexagon_side_length_proof_l761_761242


namespace sum_of_all_possible_values_of_R_area_l761_761416

noncomputable def larger_square_side : ℝ := 4
noncomputable def rectangle_side_lengths : ℝ × ℝ := (2, 4)
noncomputable def smaller_square_side : ℝ := 2
noncomputable def circle_radius : ℝ := 1

theorem sum_of_all_possible_values_of_R_area :
  let area_of_larger_square := larger_square_side ^ 2 in
  let area_of_rectangle := (rectangle_side_lengths.1 * rectangle_side_lengths.2) in
  let area_of_smaller_square := smaller_square_side ^ 2 in
  let area_of_circle := Real.pi * circle_radius ^ 2 in
  let total_occupied_area := area_of_rectangle + area_of_smaller_square + area_of_circle in
  let remaining_area := area_of_larger_square - total_occupied_area in
  remaining_area = 4 - Real.pi :=
by
  sorry

end sum_of_all_possible_values_of_R_area_l761_761416


namespace compute_tan_sum_arctan_roots_l761_761984

theorem compute_tan_sum_arctan_roots :
  let z : ℂ → ℂ := λ z, z^10 - 2*z^9 + 4*z^8 - 8*z^7 + 16*z^6 - 32*z^5 + 64*z^4 - 128*z^3 + 256*z^2 - 256*z + 256
  let roots := {z | z ≠ 0 ∧ z = z}
  let z_roots : Set ℂ := {z_i | z i = 0 ∧ z_i ∈ roots}
  (∀ z_i ∈ z_roots, z z_i = 0) →
  tan (∑ k in z_roots, arctan (z_roots k)) = 821 / 410 :=
by
  sorry

end compute_tan_sum_arctan_roots_l761_761984


namespace abs_neg_five_is_five_l761_761275

theorem abs_neg_five_is_five : abs (-5) = 5 := 
by 
  sorry

end abs_neg_five_is_five_l761_761275


namespace delivery_capacity_equation_l761_761720

-- Define the conditions as given in the problem statement
variables (x : ℝ) -- Average number of items delivered per week before the change
variables (c : ℝ) -- Number of couriers (constant)

-- Define the delivery capacities before and after the change
def delivery_before := 3000
def delivery_after := 4200

-- Define the per person delivery rates before and after the change
def rate_per_person_before := x
def rate_per_person_after := x + 40

-- Express the condition of constant number of couriers
def num_couriers_before := delivery_before / rate_per_person_before
def num_couriers_after := delivery_after / rate_per_person_after

-- State the theorem: the two expressions for the number of couriers are equal
theorem delivery_capacity_equation :
  num_couriers_before = num_couriers_after :=
by
  sorry

end delivery_capacity_equation_l761_761720


namespace team_total_points_l761_761469

theorem team_total_points (Connor_score Amy_score Jason_score : ℕ) :
  Connor_score = 2 →
  Amy_score = Connor_score + 4 →
  Jason_score = 2 * Amy_score →
  Connor_score + Amy_score + Jason_score = 20 :=
by
  intros
  sorry

end team_total_points_l761_761469


namespace petya_board_problem_l761_761643

variable (A B Z : ℕ)

theorem petya_board_problem (h1 : A + B + Z = 10) (h2 : A * B = 15) : Z = 2 := sorry

end petya_board_problem_l761_761643


namespace infinite_descendant_sequence_l761_761779

-- Assumptions
variable (Human : Type)
variable (alive_today : Human → Prop)
variable (finite_lifespan : Human → Prop)
variable (finite_children : Human → Prop)
variable (descendants : Human → Set Human)

axiom humanity_never_extinct (h : Human) : ∃ y, y ∈ descendants h

axiom finite_lifespan_humans (h : Human) : finite_lifespan h

axiom finite_children_humans (h : Human) : ∃ s : Set Human, finite s ∧ s = descendants h

-- Theorem Statement
theorem infinite_descendant_sequence :
  (∃ (E : ℕ → Human), alive_today (E 0) ∧
                      (∀ n, E (n + 1) ∈ descendants (E n))) :=
sorry

end infinite_descendant_sequence_l761_761779


namespace problem_one_problem_two_l761_761657

-- Problem 1: Simplify the given polynomial expression in terms of a.
theorem problem_one (a : ℝ) : 
  a^2 - 3a + 1 - a^2 + 6a - 7 = 3a - 6 :=
by
  sorry

-- Problem 2: Simplify the given polynomial expression in terms of m and n.
theorem problem_two (m n : ℝ) : 
  (3 * m^2 * n - 5 * m * n) - 3 * (4 * m^2 * n - 5 * m * n) = -9 * m^2 * n + 10 * m * n :=
by
  sorry

end problem_one_problem_two_l761_761657


namespace exists_equal_chord_l761_761089

def convex_figure := sorry -- Placeholder for an actual convex figure definition
def point := sorry -- Placeholder for an actual point definition

-- Condition: A convex figure and a point A inside it
variable (C : convex_figure) (A : point)

-- Statement: There exists a chord passing through point A dividing it into two equal parts
theorem exists_equal_chord (hC : C.is_convex) (hA : C.contains A) : 
  ∃ (P Q : point), P ≠ Q ∧ C.boundary_contains P ∧ C.boundary_contains Q ∧ A ∈ segment P Q ∧ dist A P = dist A Q :=
begin
  sorry
end

end exists_equal_chord_l761_761089


namespace area_PQR_eq_one_fourth_area_DEF_l761_761973

theorem area_PQR_eq_one_fourth_area_DEF 
  {A B C D E F P Q R : Type*}
  [IsTriangle A B C]
  (hD: PointOnLine D A B)
  (hE: PointOnLine E B C)
  (hF: PointOnLine F C A)
  (hP: Midpoint P A E)
  (hQ: Midpoint Q B F)
  (hR: Midpoint R C D) :
  Area P Q R = (1 / 4) * Area D E F :=
sorry

end area_PQR_eq_one_fourth_area_DEF_l761_761973


namespace length_of_AB_l761_761588

/-- A triangle ABC lies between two parallel lines where AC = 5 cm. Prove that AB = 10 cm. -/
noncomputable def triangle_is_between_two_parallel_lines : Prop := sorry

noncomputable def segmentAC : ℝ := 5

theorem length_of_AB :
  ∃ (AB : ℝ), triangle_is_between_two_parallel_lines ∧ segmentAC = 5 ∧ AB = 10 :=
sorry

end length_of_AB_l761_761588


namespace max_area_inscribed_rectangle_l761_761951

theorem max_area_inscribed_rectangle :
  let A := (0 : ℝ, 0 : ℝ),
      B := (1 : ℝ, 0 : ℝ),
      C := (0 : ℝ, 1 : ℝ),
      AB := dist A B,
      AC := dist A C,
      ABC_right_isosceles := (AB = 1) ∧ (AC = 1) ∧ (angle A B C = π / 2),
      rectangle_pts := ∃ (E : ℝ × ℝ) (F : ℝ × ℝ) (G : ℝ × ℝ) (H : ℝ × ℝ),
                        E.2 = 0 ∧ F.1 = 0 ∧ E.1 = F.2 ∧ G.1 = G.2 ∧ H.1 = H.2 ∧
                        G.1 = 1 - E.1 ∧ H.2 = 1 - F.2 ∧ G.2 = F.2 ∧ H.1 = E.1,
      rectangle_area := ∀ (E : ℝ × ℝ) (F : ℝ × ℝ) (G : ℝ × ℝ) (H : ℝ × ℝ),
                         G.1 = G.2 → H.1 = H.2 → 
                         G.1 = 1 - E.1 → H.2 = 1 - F.2 →
                         (E.2 = 0 ∧ F.1 = 0 ∧ E.1 = F.2) →
                         (G.2 = F.2 ∧ H.1 = E.1) → 
                         (dist A B = 1) ∧ (dist A C = 1) 
                          → E.1 = 0.5 → E.2 = 0 → F.1 = 0 → F.2 = 0.5
                          → G.1 = 1 - E.1 → H.2 = 1 - F.2
                          → max (E.1 * F.2) (E.1 * F.2) = 1 / 4,
  (ABC_right_isosceles ∧ rectangle_pts) →
  ∃ E F G H, rectangle_area E F G H.
sorry

end max_area_inscribed_rectangle_l761_761951


namespace triangle_ratio_implies_identity_l761_761191

theorem triangle_ratio_implies_identity {A B C a b c R : ℝ}
  (hA : A = 4 * (π / 7))
  (hB : B = 2 * (π / 7))
  (hC : C = π / 7)
  (hTriangle : A + B + C = π)
  (ha : a = 2 * R * Math.sin A)
  (hb : b = 2 * R * Math.sin B)
  (hc : c = 2 * R * Math.sin C) :
  (1 / a + 1 / b = 1 / c) := by
  sorry

end triangle_ratio_implies_identity_l761_761191


namespace hotel_price_difference_l761_761725

variable (R G P : ℝ)

def condition1 : Prop := P = 0.80 * R
def condition2 : Prop := P = 0.90 * G
def to_prove : Prop := (R / G - 1) * 100 = 12.5

theorem hotel_price_difference (h1 : condition1 R G P) (h2 : condition2 R G P) : to_prove R G :=
by
  sorry

end hotel_price_difference_l761_761725


namespace sum_T_eq_73_75_l761_761017

theorem sum_T_eq_73_75 :
  let T := ∑ k in finset.range 50, (3 + 5 * (k + 1 : ℕ)) / 3^(51 - (k + 1) : ℕ)
  in T = 73.75 :=
by
  let T : ℝ := ∑ k in finset.range 50, (3 + 5 * (k + 1 : ℕ)) / 3^(51 - (k + 1) : ℕ)
  have h : T = 73.75 := sorry
  exact h

end sum_T_eq_73_75_l761_761017


namespace range_of_a_l761_761978

-- Definitions
variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Conditions
noncomputable def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

noncomputable def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f (x)

-- Statement
theorem range_of_a (h_odd : is_odd f)
  (h_periodic : is_periodic f 3)
  (h_f1_gt1 : f 1 > 1)
  (h_f2_eq_a : f 2 = a) :
  a ∈ Ioi (-1) :=
  sorry

end range_of_a_l761_761978


namespace transformation_sum_l761_761527

theorem transformation_sum (A a ω φ : ℝ) (hω : ω > 0) (hA : A > 0) (ha : a > 0) (hφ : 0 < φ ∧ φ < π)
  (ht : ∀ x, 3 * sin (2 * x - π / 6) + 1 = A * sin (ω * x - φ) + a) :
  A + a + ω + φ = 16 / 3 + 11 * π / 12 :=
sorry

end transformation_sum_l761_761527


namespace ab_value_l761_761869

theorem ab_value (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 30) (h4 : 2 * a * b + 12 * a = 3 * b + 270) :
  a * b = 216 := by
  sorry

end ab_value_l761_761869


namespace binomial_8_5_permutation_5_3_l761_761798

theorem binomial_8_5 : Nat.choose 8 5 = 56 := by
  sorry

theorem permutation_5_3 : Nat.perm 5 3 = 60 := by
  sorry

end binomial_8_5_permutation_5_3_l761_761798


namespace smallest_n_for_sqrt_50n_is_integer_l761_761529

theorem smallest_n_for_sqrt_50n_is_integer :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, (50 * n) = k * k) ∧ n = 2 :=
by
  sorry

end smallest_n_for_sqrt_50n_is_integer_l761_761529


namespace negation_proposition_correct_l761_761480

theorem negation_proposition_correct : 
  (∀ x : ℝ, 0 < x → x + 4 / x ≥ 4) :=
by
  intro x hx
  sorry

end negation_proposition_correct_l761_761480


namespace men_employed_l761_761723

theorem men_employed (M : ℕ) (W : ℕ)
  (h1 : W = M * 9)
  (h2 : W = (M + 10) * 6) : M = 20 := by
  sorry

end men_employed_l761_761723


namespace abs_negative_five_l761_761273

theorem abs_negative_five : abs (-5) = 5 :=
by
  sorry

end abs_negative_five_l761_761273


namespace basketball_players_l761_761574

theorem basketball_players {total : ℕ} (total_boys : total = 22) 
                           (football_boys : ℕ) (football_boys_count : football_boys = 15) 
                           (neither_boys : ℕ) (neither_boys_count : neither_boys = 3) 
                           (both_boys : ℕ) (both_boys_count : both_boys = 18) : 
                           (total - neither_boys = 19) := 
by
  sorry

end basketball_players_l761_761574


namespace h_maxima_f_max_at_pi_div_3_g_bound_l761_761252

-- Definition of h(x)
def h (x : ℝ) : ℝ := (Real.sin x) ^ 2 * Real.sin (2 * x)

-- Proof that h(x) has maxima at π/3 and 4π/3
theorem h_maxima (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2 * Real.pi) : 
  (x = Real.pi / 3 ∨ x = 4 * Real.pi / 3) → 
  h x = (3 * Real.sqrt 3) / 16 := 
sorry

-- Definition of f_n(x)
def f (x : ℝ) (n : ℕ) : ℝ := 
  ∣ Real.sin x ^ 2 * 
  (∏ i in Finset.range (n-1), (Real.sin (2 ^ (i+1) * x)) ^ 3) * 
  Real.sin (2 ^ n * x) ^ 2 ^ n ∣

-- Prove that at x = π/3, f achieves its maximum value
theorem f_max_at_pi_div_3 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2 * Real.pi) (n : ℕ) : 
  f (Real.pi / 3) n ≥ f x n:= 
sorry

-- Definition of g(x)
def g (x : ℝ) (n : ℕ) : ℝ := 
  ∏ i in Finset.range (n+1), (Real.sin (2 ^ i * x)) ^ 2

-- Prove that g(x) <= (3/4)^n
theorem g_bound (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2 * Real.pi) (n : ℕ) : 
  g x n ≤ (3 / 4) ^ n := 
sorry

end h_maxima_f_max_at_pi_div_3_g_bound_l761_761252


namespace sequence_value_a_l761_761549

theorem sequence_value_a (a : ℚ) (a_n : ℕ → ℚ)
  (h1 : a_n 1 = a) (h2 : a_n 2 = a)
  (h3 : ∀ n ≥ 3, a_n n = a_n (n - 1) + a_n (n - 2))
  (h4 : a_n 8 = 34) :
  a = 34 / 21 :=
by sorry

end sequence_value_a_l761_761549


namespace option_C_l761_761445

def p (a b : ℝ) : Prop := a > b → 1/a < 1/b
def q : Prop := {x : ℝ | |x| > x} = set.Iio 0

theorem option_C (a b : ℝ) : ¬p a b ∧ q :=
by
  -- Conditions
  have h1 : ¬p a b := sorry
  have h2 : q := sorry
  -- Prove (p ∨ q) is true and (p ∧ q) is false
  exact ⟨h1, h2⟩

end option_C_l761_761445


namespace slope_of_midpoints_l761_761712

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

theorem slope_of_midpoints :
  let p1 := (1, 2)
  let p2 := (3, 6)
  let p3 := (4, 3)
  let p4 := (7, 9)
  slope (midpoint p1 p2) (midpoint p3 p4) = 4 / 7 :=
by
  sorry

end slope_of_midpoints_l761_761712


namespace angle_between_altitude_and_bisector_l761_761579

-- Define the right triangle with given sides
structure RightTriangle (a b c : ℝ) :=
  (hypotenuse_condition : c^2 = a^2 + b^2)

-- Instantiate a specific right triangle with given side lengths
def specific_triangle : RightTriangle 2 (2 * Real.sqrt 3) 4 :=
{ hypotenuse_condition := by 
  rw [sq, sq, sq, sq, sq, mul_eq_mul_right_iff];
  norm_num;
  rw [Real.sqrt_mul_self] <|> sorry }

-- Define the calculation for the altitude and angle bisector
def altitude (a b c : ℝ) (rt : RightTriangle a b c) :=
  (2 * a * b) / c

def angle_bisector_degree := 45.0 -- specifically for right triangle's 90-degree bisector

-- The main theorem: The angle between the altitude and the angle bisector drawn from the right angle is 15 degrees
theorem angle_between_altitude_and_bisector : 
  ∀ (a b c : ℝ) (rt : RightTriangle a b c),
  a = 2 → b = 2 * Real.sqrt 3 → c = 4 →
  let h := altitude a b c rt in
  let θ := 45.0 in -- angle bisector for the 90-degree angle
  15.0 = θ - 30.0 :=
by
  intros a b c rt a_eq b_eq c_eq h θ,
  norm_num,
  sorry

end angle_between_altitude_and_bisector_l761_761579


namespace arithmetic_mean_of_fractions_l761_761019

theorem arithmetic_mean_of_fractions :
  let a := (5 : ℚ) / 8
  let b := (9 : ℚ) / 16
  let c := (11 : ℚ) / 16
  a = (b + c) / 2 := by
  sorry

end arithmetic_mean_of_fractions_l761_761019


namespace unique_isolating_line_a_eq_2e_l761_761931

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (a x : ℝ) : ℝ := a * Real.log x

theorem unique_isolating_line_a_eq_2e (a : ℝ) (h : a > 0) :
  (∃ k b, ∀ x : ℝ, f x ≥ k * x + b ∧ k * x + b ≥ g a x) → a = 2 * Real.exp 1 :=
sorry

end unique_isolating_line_a_eq_2e_l761_761931


namespace hyperbola_asymptotes_angle_l761_761877

-- Define the given conditions and the proof problem
theorem hyperbola_asymptotes_angle (a b c : ℝ) (e : ℝ) (h1 : e = 2) 
  (h2 : e = c / a) (h3 : c = 2 * a) (h4 : b^2 + a^2 = c^2) : 
  ∃ θ : ℝ, θ = 60 :=
by 
  sorry -- Proof is omitted

end hyperbola_asymptotes_angle_l761_761877


namespace value_of_n_l761_761267

theorem value_of_n 
    (n : ℕ)
    (x y z : Fin n → Bool)
    (hx : ∀ i, x i = true ∨ x i = false)
    (hy : ∀ i, y i = true ∨ y i = false)
    (hz : ∀ i, z i = true ∨ z i = false)
    (hxy0 : ∑ i, (if x i then 1 else -1) * (if y i then 1 else -1) = 0)
    (hxz0 : ∑ i, (if x i then 1 else -1) * (if z i then 1 else -1) = 0)
    (hyz0 : ∑ i, (if y i then 1 else -1) * (if z i then 1 else -1) = 0) :
    ∃ k, n = 4 * k :=
by
  sorry

end value_of_n_l761_761267


namespace Vasya_numbers_l761_761350

theorem Vasya_numbers : 
  ∃ x y : ℚ, (x + y = x * y) ∧ (x * y = x / y) ∧ x = 1 / 2 ∧ y = -1 :=
by 
  use (1 / 2)
  use (-1)
  split
  case left {
    calc 
      (1 / 2) + (-1) = -1 / 2 : by norm_num,
      -1 / 2 = ((1 / 2) * (-1)) : by norm_num,
      (1 / 2) + (-1) = (1 / 2) * (-1) : by norm_num,
  },
  case right {
    calc 
      (1 / 2) * (-1) = -1 / 2 : by norm_num,
      -1 / 2 = (1 / 2) / (-1) : by norm_num,
      (1 / 2) * (-1) = (1 / 2) / (-1) : by norm_num,
  }

end Vasya_numbers_l761_761350


namespace intersection_A_complement_B_range_of_a_l761_761551

variables (x a : Real)

-- Definition of sets A and B for the first problem
def A1 := {x : Real | -2 ≤ x ∧ x ≤ -2 + 3}
def B := {x : Real | x < -1 ∨ x > 5}
def complement_B := {x : Real | -1 ≤ x ∧ x ≤ 5}

-- Problem 1 Statement
theorem intersection_A_complement_B : A1 ∩ complement_B = {x : Real | -1 ≤ x ∧ x ≤ 1} :=
sorry

-- Definitions for the second problem
def A2 := {x : Real | a ≤ x ∧ x ≤ a + 3}

-- Problem 2 Statement
theorem range_of_a (a : Real) : (A2 ⊆ B) ↔ (a < -4 ∨ a > 5) :=
sorry

end intersection_A_complement_B_range_of_a_l761_761551


namespace toucan_weights_l761_761697

namespace Toucans

def num_Toco_toucans : ℕ := 2
def num_Keel_billed_toucans : ℕ := 3
def num_Plate_billed_toucan : ℕ := 1

def weight_Toco_toucan : ℝ := 680
def weight_Keel_billed_toucan : ℝ := 450
def weight_Plate_billed_toucan : ℝ := 320

def total_weight : ℝ := (num_Toco_toucans * weight_Toco_toucan) +
                         (num_Keel_billed_toucans * weight_Keel_billed_toucan) +
                         (num_Plate_billed_toucan * weight_Plate_billed_toucan)

def total_number : ℕ := num_Toco_toucans + num_Keel_billed_toucans + num_Plate_billed_toucan

def average_weight : ℝ := total_weight / total_number

theorem toucan_weights :
  total_weight = 3030 ∧ average_weight = 505 := by
  sorry

end Toucans

end toucan_weights_l761_761697


namespace binary_to_decimal_101_l761_761027

theorem binary_to_decimal_101 : ∑ (i : Fin 3), (Nat.digit 2 ⟨i, sorry⟩ (list.nth_le [1, 0, 1] i sorry)) * (2 ^ i) = 5 := 
  sorry

end binary_to_decimal_101_l761_761027


namespace vasya_numbers_l761_761351

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 := 
by 
  sorry

end vasya_numbers_l761_761351


namespace average_median_nonempty_subsets_S_l761_761220

def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2008}

def m (A : Set ℕ) : ℝ :=
if H : A.nonempty ∧ (A ∩ S).finite
then let l := (A ∩ S).toFinset.sort (· ≤ ·)
     if h : l.length % 2 = 1
     then l.get ⟨l.length / 2, by linarith⟩
     else (l.get ⟨l.length / 2 - 1, by linarith⟩ + l.get ⟨l.length / 2, by linarith⟩) / 2
else 0

theorem average_median_nonempty_subsets_S : 
  (∑ A in (Finset.powersetUniv S.toFinset).filter (λ A, A ≠ ∅), m (A : Set ℕ) : ℝ) / 
  ((Finset.powersetUniv S.toFinset).filter (λ A, A ≠ ∅)).card = 1004.5 :=
sorry

end average_median_nonempty_subsets_S_l761_761220


namespace initial_crayons_correct_l761_761692

-- Define the variables and given conditions
variable (initial_crayons : ℕ)
variable (benny_crayons : ℕ := 3)
variable (total_crayons : ℕ := 12)

-- Theorem: Prove that the number of initial crayons is 9
theorem initial_crayons_correct : initial_crayons + benny_crayons = total_crayons → initial_crayons = 9 :=
by
  intro h
  have h1 : initial_crayons + 3 = 12 := h
  have h2 : initial_crayons = 12 - 3 := by
    linarith
  exact h2

end initial_crayons_correct_l761_761692


namespace balls_in_boxes_l761_761555

theorem balls_in_boxes :
  ∃ (n k : ℕ), n = 6 ∧ k = 3 ∧ n.choose (k - 1) = 28 :=
by {
  use [6, 3],
  split,
  { refl },
  split,
  { refl },
  {
    calc 6.choose (3 - 1) = 8.choose 2 : by rw [nat.choose_succ_succ]
                   ...  = 28 : by norm_num
  }
}

end balls_in_boxes_l761_761555


namespace ratio_AB_AD_l761_761651

theorem ratio_AB_AD (a x y : ℝ) (h1 : 0.3 * a^2 = 0.7 * x * y) (h2 : y = a / 10) : x / y = 43 :=
by
  sorry

end ratio_AB_AD_l761_761651


namespace evaluate_expression_l761_761560

variables (a b c d m : ℝ)

lemma opposite_is_zero (h1 : a + b = 0) : a + b = 0 := h1

lemma reciprocals_equal_one (h2 : c * d = 1) : c * d = 1 := h2

lemma abs_value_two (h3 : |m| = 2) : |m| = 2 := h3

theorem evaluate_expression (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 2) :
  m + c * d + (a + b) / m = 3 ∨ m + c * d + (a + b) / m = -1 :=
by
  sorry

end evaluate_expression_l761_761560


namespace min_inverse_sum_l761_761509

theorem min_inverse_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  (1 / x + 1 / y) ≥ 4 :=
sorry

example : ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ x + y = 1 ∧ (1 / x + 1 / y) = 4 :=
begin
  use [0.5, 0.5],
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num },
end

end min_inverse_sum_l761_761509


namespace find_probability_l761_761575

noncomputable def probability_within_0_80 (η : ℝ → ℝ) (δ : ℝ) : Prop :=
  ∀ η, (η : NormalDist 100 (δ ^ 2)) → (h₁ : ∀ x, 80 < x ∧ x < 120 → prob η x = 0.6) →
  (h₂ : δ > 0) → (prob η (0, 80) = 0.2)

-- Lean 4 statement expressing the equivalent of the mathematical problem
theorem find_probability (η : ℝ → ℝ) (δ : ℝ) (h₁ : ∀ x, 80 < x ∧ x < 120 → prob η x = 0.6)
  (h₂ : δ > 0) (h₃ : η = NormalDist 100 (δ ^ 2)) : 
  prob η (0, 80) = 0.2 :=
sorry  -- Proof not included, as per instructions

end find_probability_l761_761575


namespace find_angle_C_l761_761596

noncomputable def angle_C_value (A B : ℝ) : ℝ :=
  180 - A - B

theorem find_angle_C (A B : ℝ) 
  (h1 : 3 * Real.sin A + 4 * Real.cos B = 6)
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) :
  angle_C_value A B = 30 :=
sorry

end find_angle_C_l761_761596


namespace length_BM_zero_l761_761208

-- Define the setup conditions
variables {O : Type*} [MetricSpace O]
variables {A B C M : O}
variable {s : ℝ}

-- Define properties of the equilateral triangle and circle
def equilateral_triangle (A B C : O) (s : ℝ) : Prop :=
  dist A B = s ∧ dist B C = s ∧ dist C A = s

def point_on_arc (M A C : O) : Prop :=
  ∃ (O_center : O), (∃ (r : ℝ), dist O_center A = r ∧ dist O_center C = r ∧ dist O_center M = r) ∧
  ¬ ∃ (B_near_arc : O), dist O_center B_near_arc < dist O_center A

-- Declare the main theorem
theorem length_BM_zero 
  (h1 : equilateral_triangle A B C s) 
  (h2 : point_on_arc M A C) 
  (h3 : dist B O ≠ dist M O) :
  dist B M = 0 :=
sorry

end length_BM_zero_l761_761208


namespace range_of_a_l761_761111

variable {a : ℝ}
def A : set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def B : set ℝ := {x : ℝ | a < x ∧ x < 2 * a - 1}

theorem range_of_a : (A ∩ B = B) → a ∈ Iic 2 := by
  intro h
  sorry

end range_of_a_l761_761111


namespace vasya_numbers_l761_761366

theorem vasya_numbers (x y : ℚ) (h : x + y = xy ∧ xy = x / y) : x = 1 / 2 ∧ y = -1 :=
by {
  sorry
}

end vasya_numbers_l761_761366


namespace complex_number_not_in_first_quadrant_l761_761282

open Complex

theorem complex_number_not_in_first_quadrant (m : ℝ) : 
  let z := (m - 2 * Complex.I) / (1 + 2 * Complex.I) in
  ¬((z.re > 0) ∧ (z.im > 0)) :=
by 
  sorry

end complex_number_not_in_first_quadrant_l761_761282


namespace vasya_numbers_l761_761385

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : 
  (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) ∨ (x = y ∧ x = 0) :=
by 
  -- Placeholder to show where the proof would go
  sorry

end vasya_numbers_l761_761385


namespace cara_sitting_pairs_l761_761456

theorem cara_sitting_pairs : ∀ (n : ℕ), n = 7 → ∃ (pairs : ℕ), pairs = 6 :=
by
  intros n hn
  have h : n - 1 = 6 := sorry
  exact ⟨n - 1, h⟩

end cara_sitting_pairs_l761_761456


namespace sixtieth_number_is_18_l761_761668

theorem sixtieth_number_is_18 :
  let row n := list.replicate (3 * n) (3 * n),
      series Σ := list.concat (list.range (series))
  in series !! 59 = some 18 :=
by
  sorry

end sixtieth_number_is_18_l761_761668


namespace total_spent_l761_761755

def cost_sandwich : ℕ := 2
def cost_hamburger : ℕ := 2
def cost_hotdog : ℕ := 1
def cost_fruit_juice : ℕ := 2

def selene_sandwiches : ℕ := 3
def selene_fruit_juice : ℕ := 1
def tanya_hamburgers : ℕ := 2
def tanya_fruit_juice : ℕ := 2

def total_selene_spent : ℕ := (selene_sandwiches * cost_sandwich) + (selene_fruit_juice * cost_fruit_juice)
def total_tanya_spent : ℕ := (tanya_hamburgers * cost_hamburger) + (tanya_fruit_juice * cost_fruit_juice)

theorem total_spent : total_selene_spent + total_tanya_spent = 16 := by
  sorry

end total_spent_l761_761755


namespace value_of_a_minus_2_b_minus_2_l761_761214

theorem value_of_a_minus_2_b_minus_2 :
  ∀ (a b : ℝ), (a + b = -4/3 ∧ a * b = -7/3) → ((a - 2) * (b - 2) = 0) := by
  sorry

end value_of_a_minus_2_b_minus_2_l761_761214


namespace tom_watches_movies_total_duration_l761_761945

-- Define the running times for each movie
def M := 120
def A := M - 30
def B := A + 10
def D := 2 * B - 20

-- Define the number of times Tom watches each movie
def watch_B := 2
def watch_A := 3
def watch_M := 1
def watch_D := 4

-- Calculate the total time spent watching each movie
def total_time_B := watch_B * B
def total_time_A := watch_A * A
def total_time_M := watch_M * M
def total_time_D := watch_D * D

-- Calculate the total duration Tom spends watching these movies in a week
def total_duration := total_time_B + total_time_A + total_time_M + total_time_D

-- The statement to prove
theorem tom_watches_movies_total_duration :
  total_duration = 1310 := 
by
  sorry

end tom_watches_movies_total_duration_l761_761945


namespace cos_eq_cos_of_n_l761_761824

theorem cos_eq_cos_of_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : Real.cos (n * Real.pi / 180) = Real.cos (283 * Real.pi / 180)) : n = 77 :=
by sorry

end cos_eq_cos_of_n_l761_761824


namespace product_of_two_numbers_l761_761677

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := 
by {
  sorry
}

end product_of_two_numbers_l761_761677


namespace max_min_diff_value_l761_761976

noncomputable def max_min_diff_c (a b c : ℝ) (h1 : a + b + c = 2) (h2 : a^2 + b^2 + c^2 = 12) : ℝ :=
  (10 / 3) - (-2)

theorem max_min_diff_value (a b c : ℝ) (h1 : a + b + c = 2) (h2 : a^2 + b^2 + c^2 = 12) : 
  max_min_diff_c a b c h1 h2 = 16 / 3 := 
by 
  sorry

end max_min_diff_value_l761_761976


namespace longest_diagonal_length_l761_761424

-- Defining conditions
variable (d1 d2 : ℝ)
variable (x : ℝ)
variable (area : ℝ)
variable (h_area : area = 144)
variable (h_ratio : d1 = 4 * x)
variable (h_ratio' : d2 = 3 * x)
variable (h_area_eq : area = 1 / 2 * d1 * d2)

-- The Lean statement, asserting the length of the longest diagonal is 8 * sqrt(6)
theorem longest_diagonal_length (x : ℝ) (h_area : 1 / 2 * (4 * x) * (3 * x) = 144) :
  4 * x = 8 * Real.sqrt 6 := by
sorry

end longest_diagonal_length_l761_761424


namespace expenditure_representation_l761_761238

def income_represented_pos (income : ℤ) : Prop := income > 0

def expenditure_represented_neg (expenditure : ℤ) : Prop := expenditure < 0

theorem expenditure_representation (income expenditure : ℤ) (h_income: income_represented_pos income) (exp_value: expenditure = 3) : expenditure_represented_neg expenditure := 
sorry

end expenditure_representation_l761_761238


namespace roots_quadratic_eq_sum_prod_l761_761150

theorem roots_quadratic_eq_sum_prod (r s p q : ℝ) (hr : r + s = p) (hq : r * s = q) : r^2 + s^2 = p^2 - 2 * q :=
by
  sorry

end roots_quadratic_eq_sum_prod_l761_761150


namespace geometric_sequence_seventh_term_l761_761414

theorem geometric_sequence_seventh_term (a r: ℤ) (h1 : a = 3) (h2 : a * r ^ 5 = 729) : a * r ^ 6 = 2187 :=
by sorry

end geometric_sequence_seventh_term_l761_761414


namespace exists_x0_in_interval_l761_761624

theorem exists_x0_in_interval 
  (a b : ℝ) : ∃ x0 ∈ Icc (-1:ℝ) 1, |(x0 ^ 2 + a * x0 + b)| + a ≥ 0 :=
by sorry

end exists_x0_in_interval_l761_761624


namespace middle_term_in_expansion_sum_of_odd_coefficients_max_coefficient_terms_l761_761100

noncomputable def binomial_expansion (m : ℕ) : ℕ → ℚ
| r => (nat.choose m r) * ((1 / 2) ^ r)

def is_arithmetic_seq (a0 a1 a2 : ℚ) : Prop := 2 * a1 = a0 + a2

theorem middle_term_in_expansion :
  ∀ a0 a1 a2 x : ℚ, ∀ m : ℕ, is_arithmetic_seq a0 a1 a2 →
  m = 8 → (1 + 1 / 2 * x)^m =
  a0 + a1 * x + a2 * x^2 + (binomial_expansion m 4) * x ^ 4 :=
by sorry

theorem sum_of_odd_coefficients :
  ∀ a0 a1 a2 x : ℚ, ∀ m : ℕ, is_arithmetic_seq a0 a1 a2 →
  m = 8 → (1 + 1 / 2 * x)^m =
  a0 + a1 * x + a2 * x^2 + (3/2)^8 - (1/2)^8 =
  205 / 16 :=
by sorry

theorem max_coefficient_terms :
  ∀ a0 a1 a2 x : ℚ, ∀ m : ℕ, is_arithmetic_seq a0 a1 a2 →
  m = 14 → (1 + 1 / 2 * x)^(m+6) =
  a0 + a1 * x + a2 * x^2 + (binomial_expansion 14 5) * x ^ 5 :=
  (binomial_expansion 14 6) * x ^ 6 :=
by sorry

end middle_term_in_expansion_sum_of_odd_coefficients_max_coefficient_terms_l761_761100


namespace percentage_salt_in_first_solution_l761_761758

theorem percentage_salt_in_first_solution (S : ℝ) (total_weight : ℝ) (replacement_fraction : ℝ) (second_solution_salt_percentage : ℝ) (resulting_solution_salt_percentage : ℝ) :
  resulting_solution_salt_percentage = 16 →
  replacement_fraction = 1/4 →
  second_solution_salt_percentage = 31 →
  total_weight = 100 →
  let total_salt := (S / 100) * (total_weight * (1 - replacement_fraction)) + (second_solution_salt_percentage / 100) * (total_weight * replacement_fraction)
  in total_salt = resulting_solution_salt_percentage * total_weight / 100 →
  S = 11 :=
by
  sorry

end percentage_salt_in_first_solution_l761_761758


namespace vasya_numbers_l761_761353

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 := 
by 
  sorry

end vasya_numbers_l761_761353


namespace correct_sum_of_integers_l761_761334

theorem correct_sum_of_integers :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a - b = 3 ∧ a * b = 63 ∧ a + b = 17 :=
by 
  sorry

end correct_sum_of_integers_l761_761334


namespace count_3_digit_multiples_of_25_not_75_l761_761141

theorem count_3_digit_multiples_of_25_not_75 : 
  (finset.Icc 100 975).filter (λ n, n % 25 = 0 ∧ n % 75 ≠ 0).card = 24 :=
by
  sorry

end count_3_digit_multiples_of_25_not_75_l761_761141


namespace total_fruits_on_display_l761_761689

-- Declare the variables and conditions as hypotheses
variables (apples oranges bananas : ℕ)
variables (h1 : apples = 2 * oranges)
variables (h2 : oranges = 2 * bananas)
variables (h3 : bananas = 5)

-- Define what we want to prove
theorem total_fruits_on_display : apples + oranges + bananas = 35 :=
by sorry

end total_fruits_on_display_l761_761689


namespace probability_of_divisibility_l761_761926

-- Definitions based on given conditions

def digits : Finset ℕ := {1, 2, 3, 5, 5, 8, 0}

noncomputable def probability_divisible_by_30 : ℚ :=
  5 / 21

theorem probability_of_divisibility :
  let arrangements := digits.to_finset.permutations.map (λ l, l.foldl (λ acc d, acc * 10 + d) 0)
  let divisible_by_30 := arrangements.filter (λ n, n % 30 = 0)
  (divisible_by_30.card : ℚ) / (arrangements.card : ℚ) = probability_divisible_by_30 :=
by
  sorry

end probability_of_divisibility_l761_761926


namespace find_time_eating_dinner_l761_761450

def total_flight_time : ℕ := 11 * 60 + 20
def time_reading : ℕ := 2 * 60
def time_watching_movies : ℕ := 4 * 60
def time_listening_radio : ℕ := 40
def time_playing_games : ℕ := 1 * 60 + 10
def time_nap : ℕ := 3 * 60

theorem find_time_eating_dinner : 
  total_flight_time - (time_reading + time_watching_movies + time_listening_radio + time_playing_games + time_nap) = 30 := 
by
  sorry

end find_time_eating_dinner_l761_761450


namespace factor_polynomial_real_coeffs_l761_761247

-- Define the polynomial P with real coefficients
theorem factor_polynomial_real_coeffs (P : Polynomial ℝ) :
  ∃ L Q : list (Polynomial ℝ), 
    (∀ p ∈ L, degree p = 1) ∧ (∀ q ∈ Q, degree q = 2) ∧
    P = (L.product * Q.product) :=
sorry

end factor_polynomial_real_coeffs_l761_761247


namespace distance_PQ_l761_761593

-- Definitions for points in polar coordinates
structure PolarCoord :=
  (r : ℝ)
  (theta : ℝ)

def P : PolarCoord := ⟨1, Real.pi / 6⟩
def Q : PolarCoord := ⟨2, Real.pi / 2⟩

-- Function to convert polar coordinates to rectangular coordinates
def polarToRect (p : PolarCoord) : ℝ × ℝ :=
  (p.r * Real.cos p.theta, p.r * Real.sin p.theta)

-- Distance formula between two points in rectangular coordinates
def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Theorem statement: distance between points P and Q in polar coordinates is √3
theorem distance_PQ : distance (polarToRect P) (polarToRect Q) = Real.sqrt 3 := by
  sorry

end distance_PQ_l761_761593


namespace grandpa_movie_time_l761_761130

theorem grandpa_movie_time
  (each_movie_time : ℕ := 90)
  (max_movies_2_days : ℕ := 9)
  (x_movies_tuesday : ℕ)
  (movies_wednesday := 2 * x_movies_tuesday)
  (total_movies := x_movies_tuesday + movies_wednesday)
  (h : total_movies = max_movies_2_days) :
  90 * x_movies_tuesday = 270 :=
by
  sorry

end grandpa_movie_time_l761_761130


namespace circles_on_parabola_pass_through_focus_l761_761858

theorem circles_on_parabola_pass_through_focus : 
  ∀ (P : ℝ × ℝ), (P.snd + 2)^2 = 4 * (P.fst - 1) →
  (∀ r : ℝ, circle P r ∧ tangent_y_axis P r → ∃ Q : ℝ × ℝ, Q = (2, -2)) :=
begin
  sorry
end

end circles_on_parabola_pass_through_focus_l761_761858


namespace greatest_n_le_5_value_ge_2525_l761_761155

theorem greatest_n_le_5_value_ge_2525 (n : ℤ) (V : ℤ) 
  (h1 : 101 * n^2 ≤ V) 
  (h2 : ∀ k : ℤ, (101 * k^2 ≤ V) → (k ≤ 5)) : 
  V ≥ 2525 := 
sorry

end greatest_n_le_5_value_ge_2525_l761_761155


namespace tan_pi_minus_alpha_l761_761528

theorem tan_pi_minus_alpha 
  (α : ℝ) 
  (h1 : Real.sin α = 1 / 3) 
  (h2 : π / 2 < α) 
  (h3 : α < π) :
  Real.tan (π - α) = Real.sqrt 2 / 4 :=
by
  sorry

end tan_pi_minus_alpha_l761_761528


namespace problem_statement_l761_761913

-- Define conditions
def is_solution (x : ℝ) : Prop :=
  real.log (343) / real.log (3 * x) = x

-- Formulate what we need to prove about the solution
def is_non_square_non_cube_non_integral_rational (x : ℝ) : Prop :=
  x ∈ set_of (λ x : ℚ, ¬is_square x ∧ ¬is_cube x ∧ frac x ≠ 0)

-- The main statement: Prove that x, satisfying the conditions, has the specified properties
theorem problem_statement (x : ℝ) (hx : is_solution x) : is_non_square_non_cube_non_integral_rational x :=
sorry

end problem_statement_l761_761913


namespace vasya_numbers_l761_761379

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : 
  (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) ∨ (x = y ∧ x = 0) :=
by 
  -- Placeholder to show where the proof would go
  sorry

end vasya_numbers_l761_761379


namespace probability_of_second_term_is_one_l761_761619

def permutation_set : set (list ℕ) := {l | (l.length = 6) ∧
                        (∀ i ∈ l, i ∈ [1, 2, 3, 4, 5, 6]) ∧
                        (l.head ∈ [2, 4, 6]) ∧
                        (l !! 2 ∈ [1, 3, 5])}

def favorable_permutations : list (list ℕ) := 
  [ [e, 1, o] ++ rest | 
    e ∈ [2, 4, 6], 
    o ∈ [1, 3, 5],
    rest ∈ list.permutations [1, 2, 3, 4, 5, 6].erase e.ContinouP (1),
    list.disjoint (e::o::[]) rest ]

noncomputable def probability_second_term_is_one :=
  (favorable_permutations.length : ℚ) / (permutation_set.size : ℚ)

theorem probability_of_second_term_is_one : probability_second_term_is_one = 1 / 6 :=
by
  sorry

end probability_of_second_term_is_one_l761_761619


namespace probability_of_exactly_one_pair_l761_761554

theorem probability_of_exactly_one_pair :
  let N := 10
  let colors := 5
  let pairs := 2
  let draws := 5
  let total_combinations := Nat.binom N draws
  let favorable_combinations := 5 * 4 * 8
  total_combinations = 252 ∧ favorable_combinations = 160 →
  let probability := (160 : ℚ) / 252
  probability = 40 / 63 := 
by
  intros
  sorry

end probability_of_exactly_one_pair_l761_761554


namespace combined_degrees_l761_761259

theorem combined_degrees (summer_degrees jolly_degrees : ℕ) (h1 : summer_degrees = jolly_degrees + 5) (h2 : summer_degrees = 150) : summer_degrees + jolly_degrees = 295 := 
by
  sorry

end combined_degrees_l761_761259


namespace no_positive_n_for_prime_expr_l761_761811

noncomputable def is_prime (p : ℤ) : Prop := p > 1 ∧ (∀ m : ℤ, 1 < m → m < p → ¬ (m ∣ p))

theorem no_positive_n_for_prime_expr : 
  ∀ n : ℕ, 0 < n → ¬ is_prime (n^3 - 9 * n^2 + 23 * n - 17) := by
  sorry

end no_positive_n_for_prime_expr_l761_761811


namespace david_speed_29_17_l761_761474

theorem david_speed_29_17
    (distance_chennai_hyderabad : ℝ)
    (lewis_speed : ℝ)
    (meeting_distance_from_chennai : ℝ)
    (distance_to_and_fro : ℝ := distance_chennai_hyderabad + meeting_distance_from_chennai)
    (lewis_travel_time : ℝ := distance_to_and_fro / lewis_speed) :
  let david_speed := meeting_distance_from_chennai / lewis_travel_time in
  distance_chennai_hyderabad = 350 →
  lewis_speed = 70 →
  meeting_distance_from_chennai = 250 →
  david_speed = (175 / 6) :=
by
  intros
  sorry

end david_speed_29_17_l761_761474


namespace evaluate_g_at_pi_div_12_l761_761667

def f (x : ℝ) : ℝ := 6 * Real.sin (2 * x - Real.pi / 3)

def g (x : ℝ) : ℝ := f (x - Real.pi / 12)

theorem evaluate_g_at_pi_div_12 : g (Real.pi / 12) = -3 * Real.sqrt 3 := by
  sorry

end evaluate_g_at_pi_div_12_l761_761667


namespace Kolya_result_l761_761431

-- Define the list of numbers
def numbers := [1.05, 1.15, 1.25, 1.4, 1.5, 1.6, 1.75, 1.85, 1.95]

-- Defining the missing decimal point case
def mistaken_number := 15

-- Calculate the sum correctly and with a missing decimal point
noncomputable def correct_sum := numbers.sum
noncomputable def mistaken_sum := correct_sum + (mistaken_number - 1.5)

-- Prove the result is as expected
theorem Kolya_result : mistaken_sum = 27 := by
  sorry

end Kolya_result_l761_761431


namespace find_angle_BAC_is_7_l761_761727

theorem find_angle_BAC_is_7 :
  ∀ {A B C D E : Type} [euclidean_geometry A B C D E],
    isosceles A D E 37 ∧ congruent A B C E B D → ∠A B C = 7 :=
by
  intros A B C D E h
  cases h with isosceles_ADE congruence_triangles
  sorry

end find_angle_BAC_is_7_l761_761727


namespace vasya_numbers_l761_761340

theorem vasya_numbers :
  ∃ x y : ℝ, x + y = x * y ∧ x + y = x / y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l761_761340


namespace vasya_numbers_l761_761370

theorem vasya_numbers (x y : ℚ) (h : x + y = xy ∧ xy = x / y) : x = 1 / 2 ∧ y = -1 :=
by {
  sorry
}

end vasya_numbers_l761_761370


namespace class_scheduling_l761_761241

theorem class_scheduling :
  ∃ (schedules : ℕ), schedules = 18 ∧
  ∀ (class1 class2 class3 class4 : String),
  class1 ≠ "Physical Education" →
  {class2, class3, class4} = {"Chinese", "Mathematics", "Computer Science"} →
  schedules = 3 * (3 * 2 * 1) :=
by
  sorry

end class_scheduling_l761_761241


namespace distinct_positive_integers_sum_one_l761_761655

def sum_of_reciprocals (n : ℕ) (s : Set ℕ) := ∑ x in s, (1 : ℚ) / x

theorem distinct_positive_integers_sum_one (n : ℕ) (hn : n > 2) :
  ∃ s : Set ℕ, s.finite ∧ s.card = n ∧ (∀ x ∈ s, 0 < x) ∧ sum_of_reciprocals n s = 1 :=
sorry

end distinct_positive_integers_sum_one_l761_761655


namespace x_squared_plus_inverse_squared_l761_761398

theorem x_squared_plus_inverse_squared (x : ℝ) (h : x + 1/x = 3.5) : x^2 + (1/x)^2 = 10.25 :=
by sorry

end x_squared_plus_inverse_squared_l761_761398


namespace Michelle_initial_crayons_l761_761998

variable (M : ℕ)  -- M is the number of crayons Michelle initially has
variable (J : ℕ := 2)  -- Janet has 2 crayons
variable (final_crayons : ℕ := 4)  -- After Janet gives her crayons to Michelle, Michelle has 4 crayons

theorem Michelle_initial_crayons : M + J = final_crayons → M = 2 :=
by
  intro h1
  sorry

end Michelle_initial_crayons_l761_761998


namespace relationship_of_f_l761_761510

def f (x : ℝ) : ℝ := x^2 - 2 * Real.cos x

theorem relationship_of_f (x1 x2 x3 : ℝ) (h1 : x1 < -(1 / 3)) (h2 : x2 = -(1 / 3)) (h3 : x3 = 2 / 5) :
  f 0 < f x1 ∧ f x1 = f x2 ∧ f x2 < f x3 :=
  by
  sorry

end relationship_of_f_l761_761510


namespace max_h_value_l761_761542

noncomputable def f (x : ℝ) := log (x + 1) / log 2
noncomputable def g (x : ℝ) := (1 / 2) * log (3 * x + 1) / log 2
noncomputable def h (x : ℝ) := g x - f x

theorem max_h_value : 
  (∃ x : ℝ, h x = ((1 / 2) * log (9 / 8) / log 2)) :=
begin
  sorry
end

end max_h_value_l761_761542


namespace gambler_initial_games_l761_761744

theorem gambler_initial_games (x : ℕ)
  (h1 : ∀ x, ∃ (wins : ℝ), wins = 0.40 * x) 
  (h2 : ∀ x, ∃ (total_games : ℕ), total_games = x + 30)
  (h3 : ∀ x, ∃ (total_wins : ℝ), total_wins = 0.40 * x + 24)
  (h4 : ∀ x, ∃ (final_win_rate : ℝ), final_win_rate = (0.40 * x + 24) / (x + 30))
  (h5 : ∃ (final_win_rate_target : ℝ), final_win_rate_target = 0.60) :
  x = 30 :=
by
  sorry

end gambler_initial_games_l761_761744


namespace arc_measure_BN_l761_761777

variables (M N C A B P : Point)

noncomputable def circle_semicircle (M N C A B P : Point) : Prop :=
  ∃ γ : Circle, (γ.center = C ∧ diameter γ M N) ∧
  (γ.on_circle A ∧ γ.on_circle B) ∧
  (C = midpoint M N) ∧
  (P ∈ line_through C N) ∧
  (∠ C A P = 10) ∧ (∠ C B P = 10) ∧
  (arc_measure γ M A = 40)

theorem arc_measure_BN 
  (M N C A B P : Point) (h : circle_semicircle M N C A B P) :
  arc_measure (Circle.mk C (dist C M)) B N = 20 :=
by 
  sorry

end arc_measure_BN_l761_761777


namespace area_of_figure_l761_761590
-- Import necessary libraries

-- Define the conditions as functions/constants
def length_left : ℕ := 7
def width_top : ℕ := 6
def height_middle : ℕ := 3
def width_middle : ℕ := 4
def height_right : ℕ := 5
def width_right : ℕ := 5

-- State the problem as a theorem
theorem area_of_figure : 
  (length_left * width_top) + 
  (width_middle * height_middle) + 
  (width_right * height_right) = 79 := 
  by
  sorry

end area_of_figure_l761_761590


namespace parallelogram_area_l761_761038

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def vector_sub (p q : Point3D) : Point3D :=
  { x := p.x - q.x, y := p.y - q.y, z := p.z - q.z }

def cross_product (u v : Point3D) : Point3D :=
  { x := u.y * v.z - u.z * v.y,
    y := u.z * v.x - u.x * v.z,
    z := u.x * v.y - u.y * v.x }

def magnitude (p : Point3D) : ℝ :=
  real.sqrt (p.x^2 + p.y^2 + p.z^2)

theorem parallelogram_area (P Q R S : Point3D) 
  (hP : P = { x := 1, y := -2, z := 1 })
  (hQ : Q = { x := 4, y := -7, z := 4 })
  (hR : R = { x := 2, y := -1, z := -1 })
  (hS : S = { x := 5, y := -6, z := 2 }) :
  let v1 := vector_sub Q P in
  let v2 := vector_sub S R in
  let v3 := vector_sub R P in
  v1 = v2 ∧ magnitude (cross_product v1 v3) = real.sqrt 194 :=
by
  sorry

end parallelogram_area_l761_761038


namespace temperature_at_summit_l761_761439

-- State the given conditions as definitions
def temperature_drop_per_100_meters := 0.6 -- degrees Celsius per 100 meters
def altitude_school := 300 -- meters
def temp_at_school := 35 -- degrees Celsius
def altitude_mountain := 1600 -- meters

-- State the proof problem
theorem temperature_at_summit :
  let altitude_difference := altitude_mountain - altitude_school,
      temperature_drop := (temperature_drop_per_100_meters * altitude_difference) / 100,
      temp_at_summit := temp_at_school - temperature_drop in
  temp_at_summit = 27.2 :=
by
  sorry

end temperature_at_summit_l761_761439


namespace number_of_divisors_not_multiples_of_14_l761_761217

theorem number_of_divisors_not_multiples_of_14 
  (n : ℕ)
  (h1: ∃ k : ℕ, n = 2 * k * k)
  (h2: ∃ k : ℕ, n = 3 * k * k * k)
  (h3: ∃ k : ℕ, n = 5 * k * k * k * k * k)
  (h4: ∃ k : ℕ, n = 7 * k * k * k * k * k * k * k)
  : 
  ∃ num_divisors : ℕ, num_divisors = 19005 ∧ (∀ d : ℕ, d ∣ n → ¬(14 ∣ d)) := sorry

end number_of_divisors_not_multiples_of_14_l761_761217


namespace f_of_7_l761_761743

theorem f_of_7 (f : ℝ → ℝ) (h : ∀ (x : ℝ), f (4 * x - 1) = x^2 + 2 * x + 2) :
    f 7 = 10 := by
  sorry

end f_of_7_l761_761743


namespace fruit_display_total_l761_761687

-- Define the number of bananas
def bananas : ℕ := 5

-- Define the number of oranges, twice the number of bananas
def oranges : ℕ := 2 * bananas

-- Define the number of apples, twice the number of oranges
def apples : ℕ := 2 * oranges

-- Define the total number of fruits
def total_fruits : ℕ := bananas + oranges + apples

-- Theorem statement: Prove that the total number of fruits is 35
theorem fruit_display_total : total_fruits = 35 := 
by
  -- proof will be skipped, but all necessary conditions and definitions are included to ensure the statement is valid
  sorry

end fruit_display_total_l761_761687


namespace first_round_knockout_percentage_l761_761653

/-- Rocky's boxing statistics -/
def rocky_fights : ℕ := 190
def knockout_percentage : ℚ := 0.50
def first_round_knockouts : ℕ := 19

/-- Prove that 20% of Rocky's knockouts were in the first round. -/
theorem first_round_knockout_percentage
  (total_fights : ℕ)
  (ko_percentage : ℚ)
  (first_round_ko : ℕ)
  (total_knockouts : ℕ :=
    (ko_percentage * total_fights))
  : (first_round_ko / total_knockouts) * 100 = 20 :=
by
  sorry

#eval first_round_knockout_percentage rocky_fights knockout_percentage first_round_knockouts

end first_round_knockout_percentage_l761_761653


namespace count_multiples_2_or_3_not_4_or_5_l761_761142

theorem count_multiples_2_or_3_not_4_or_5 :
  (finset.filter (λ n : ℕ, (n ≤ 200) ∧ ((n % 2 = 0 ∨ n % 3 = 0) ∧ (n % 4 ≠ 0 ∧ n % 5 ≠ 0))) (finset.range 201)).card = 53 :=
by sorry

end count_multiples_2_or_3_not_4_or_5_l761_761142


namespace petya_can_guarantee_k_pastries_l761_761684

-- Conditions
def num_plates := 2019
def initial_pastries := 1
def max_move := 16
def multiples (n : ℕ) := n * 32

-- Proving Petya's strategy to guarantee k pastries on one plate
theorem petya_can_guarantee_k_pastries :
  ∃ k, k = 32 ∧ ∀ (moves : fin num_plates → ℕ), 
       (∀ i, moves i ≤ max_move) →
       (∃ plate, num_plates ≤ plate ∧ k ≤ (initial_pastries * num_plates) / 32) :=
by
  -- Proof should be provided here
  sorry

end petya_can_guarantee_k_pastries_l761_761684


namespace constant_term_in_expansion_l761_761822

theorem constant_term_in_expansion : 
  (∃ c : ℤ, c = 60 ∧ ∀ x ≠ 0, ∃ r, (sqrt(x) - 2/x)^6 = (sqrt(x) - 2/x)^6 + c) := sorry

end constant_term_in_expansion_l761_761822


namespace initial_crayons_correct_l761_761693

-- Define the variables and given conditions
variable (initial_crayons : ℕ)
variable (benny_crayons : ℕ := 3)
variable (total_crayons : ℕ := 12)

-- Theorem: Prove that the number of initial crayons is 9
theorem initial_crayons_correct : initial_crayons + benny_crayons = total_crayons → initial_crayons = 9 :=
by
  intro h
  have h1 : initial_crayons + 3 = 12 := h
  have h2 : initial_crayons = 12 - 3 := by
    linarith
  exact h2

end initial_crayons_correct_l761_761693


namespace batting_average_is_60_l761_761410

-- Definitions for conditions:
def highest_score : ℕ := 179
def difference_highest_lowest : ℕ := 150
def average_44_innings : ℕ := 58
def innings_excluding_highest_lowest : ℕ := 44
def total_innings : ℕ := 46

-- Lowest score
def lowest_score : ℕ := highest_score - difference_highest_lowest

-- Total runs in 44 innings
def total_runs_44 : ℕ := average_44_innings * innings_excluding_highest_lowest

-- Total runs in 46 innings
def total_runs_46 : ℕ := total_runs_44 + highest_score + lowest_score

-- Batting average in 46 innings
def batting_average_46 : ℕ := total_runs_46 / total_innings

-- The theorem to prove
theorem batting_average_is_60 :
  batting_average_46 = 60 :=
sorry

end batting_average_is_60_l761_761410


namespace elijah_needs_20_meters_of_tape_l761_761049

def wall_width_4m_2 (n : Nat) : Prop :=
  n = 2 * 4

def wall_width_6m_2 (n : Nat) : Prop :=
  n = 2 * 6

def total_masking_tape (tape : Nat) : Prop :=
  ∃ n1 n2, wall_width_4m_2 n1 ∧ wall_width_6m_2 n2 ∧ tape = n1 + n2

theorem elijah_needs_20_meters_of_tape : total_masking_tape 20 :=
by
  unfold total_masking_tape
  apply Exists.intro 8
  apply Exists.intro 12
  unfold wall_width_4m_2 wall_width_6m_2
  split
  . rfl
  split
  . rfl
  . rfl

end elijah_needs_20_meters_of_tape_l761_761049


namespace average_age_l761_761944

theorem average_age (h1 : ∀ (n : ℕ), n = 8 → (MrBernard : ℕ) = 3 * 20) (h2 : (Luke : ℕ) = 20) : (avg_age : ℕ) = 36 := 
by
  -- Define Mr. Bernard's age in 8 years
  let MrBernard_in_eight_years := 3 * Luke
  -- Define Mr. Bernard's current age
  let MrBernard := MrBernard_in_eight_years - 8
  -- Define the sum of ages
  let sum_ages := Luke + MrBernard
  -- Define the average age
  let avg_age := sum_ages / 2
  -- Assertion based on conditions
  assert h3: avg_age = 36
  sorry

end average_age_l761_761944


namespace chloe_profit_l761_761458

def cost_per_dozen : ℕ := 50
def sell_per_half_dozen : ℕ := 30
def total_dozens_sold : ℕ := 50

def total_cost (n: ℕ) : ℕ := n * cost_per_dozen
def total_revenue (n: ℕ) : ℕ := n * (sell_per_half_dozen * 2)
def profit (cost revenue : ℕ) : ℕ := revenue - cost

theorem chloe_profit : 
  profit (total_cost total_dozens_sold) (total_revenue total_dozens_sold) = 500 := 
by
  sorry

end chloe_profit_l761_761458


namespace composite_has_at_least_three_divisors_l761_761739

def is_composite (n : ℕ) : Prop := ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

theorem composite_has_at_least_three_divisors (n : ℕ) (h : is_composite n) : ∃ a b c, a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c :=
sorry

end composite_has_at_least_three_divisors_l761_761739


namespace propA_propB_relation_l761_761646

variable (x y : ℤ)

theorem propA_propB_relation :
  (x + y ≠ 5 → x ≠ 2 ∨ y ≠ 3) ∧ ¬(x ≠ 2 ∨ y ≠ 3 → x + y ≠ 5) :=
by
  sorry

end propA_propB_relation_l761_761646


namespace r_filling_l761_761726

-- Definitions for the conditions in problem
def K_radius : ℝ := 1
def r_list := [1, 0.5, 2 * Real.sqrt 3 - 3, Real.sqrt 2 - 1, 
                     Real.sqrt 5 / (Real.sqrt 5 + Real.sqrt (10 + 2 * Real.sqrt 5)), 1/3, 1/3]

-- Conditions for the non-intersecting circles
def condition_a (r1 r2 : ℝ) : Prop := 
  r1 > 1/2 ∧ r2 > 1/2 → ¬ (r1 + r2 < 1)

def condition_b (r1 r2 r3 : ℝ) : Prop := 
  r1 > 2 * Real.sqrt 3 - 3 ∧ r2 > 2 * Real.sqrt 3 - 3 ∧ r3 > 2 * Real.sqrt 3 - 3 → 
  ¬ ((r1 + r2 + r3) / 3 < 4 - 2 * Real.sqrt 3)

-- Theorem for the radii filling
theorem r_filling (n : ℕ) : ℝ :=
match n with
| 1 => r_list.head!
| 2 => r_list.nth 1 |>.getD 0
| 3 => r_list.nth 2 |>.getD 0
| 4 => r_list.nth 3 |>.getD 0
| 5 => r_list.nth 4 |>.getD 0
| 6 => r_list.nth 5 |>.getD 0
| 7 => r_list.nth 6 |>.getD 0
| _ => 0
-- Return 0 for cases n > 7 since they are not provided

-- Proof of Claim A
lemma proof_a (r1 r2 : ℝ): condition_a r1 r2 := sorry

-- Proof of Claim B
lemma proof_b (r1 r2 r3 : ℝ): condition_b r1 r2 r3 := sorry

end r_filling_l761_761726


namespace marcus_dog_time_l761_761995

-- Definitions of the problem conditions
def bath_time : ℕ := 20
def blow_dry_time : ℕ := 10
def fetch_time : ℕ := 15
def training_time : ℕ := 10

-- Walk times based on terrain and speeds
def walk_flat_time : ℕ := (1 * 60) / 6
def walk_uphill_time : ℕ := (1 * 60) / 4
def walk_downhill_time : ℕ := (1 * 60) / 8
def walk_sandy_time : ℕ := (1 * 60) / 3

-- Let's sum up all the times
def total_time := bath_time + blow_dry_time + fetch_time + training_time
               + walk_flat_time + walk_uphill_time + walk_downhill_time + walk_sandy_time

-- Statement of the theorem to prove total time
theorem marcus_dog_time : total_time = 107.5 := by
  sorry

end marcus_dog_time_l761_761995


namespace base6_add_sub_l761_761831

theorem base6_add_sub (a b c : ℕ) (ha : a = 5 * 6^2 + 5 * 6^1 + 5 * 6^0)
  (hb : b = 6 * 6^1 + 5 * 6^0) (hc : c = 1 * 6^1 + 1 * 6^0) :
  (a + b - c) = 1 * 6^3 + 0 * 6^2 + 5 * 6^1 + 3 * 6^0 :=
by
  -- We should translate the problem context into equivalence
  -- but this part of the actual proof is skipped with sorry.
  sorry

end base6_add_sub_l761_761831


namespace constant_term_binomial_expansion_l761_761284

theorem constant_term_binomial_expansion : 
  let general_term (r : ℕ) : ℕ := Nat.choose 6 r * x ^ (6 - 2 * r) in
  ∃ (r : ℕ), 6 - 2 * r = 0 ∧ general_term r = 20 :=
sorry

end constant_term_binomial_expansion_l761_761284


namespace Vasya_numbers_l761_761376

theorem Vasya_numbers : ∃ (x y : ℝ), x + y = xy ∧ xy = x / y ∧ (x, y) = (1 / 2, -1) :=
by
  sorry

end Vasya_numbers_l761_761376


namespace problem_statement_l761_761190

-- Definitions of the side lengths' relationships in triangle ABC
variables {a b c : ℝ} {A B C : ℝ} {BD : ℝ}

-- Conditions
def conditions : Prop :=
  (BD = sqrt 31) ∧
  (a / 2 = b / 3) ∧
  (b / 3 = c / 4)

-- Statements to prove
def tan_C_is : Prop := 
  ∀ {a b c : ℝ}, 
  (a / 2 = b / 3) ∧ (b / 3 = c / 4) ∧ (BD = sqrt 31) → 
  tan C = -sqrt 15

def area_triangle_ABC : ℝ :=
  3 * sqrt 15

theorem problem_statement : conditions → tan_C_is ∧ (S_triangle_ABC = 3 * sqrt 15) := 
  by 
    sorry

end problem_statement_l761_761190


namespace tan_sum_of_angles_eq_neg_sqrt_three_l761_761519

theorem tan_sum_of_angles_eq_neg_sqrt_three 
  (A B C : ℝ)
  (h1 : B - A = C - B)
  (h2 : A + B + C = Real.pi) :
  Real.tan (A + C) = -Real.sqrt 3 :=
sorry

end tan_sum_of_angles_eq_neg_sqrt_three_l761_761519


namespace Vasya_numbers_l761_761346

theorem Vasya_numbers : 
  ∃ x y : ℚ, (x + y = x * y) ∧ (x * y = x / y) ∧ x = 1 / 2 ∧ y = -1 :=
by 
  use (1 / 2)
  use (-1)
  split
  case left {
    calc 
      (1 / 2) + (-1) = -1 / 2 : by norm_num,
      -1 / 2 = ((1 / 2) * (-1)) : by norm_num,
      (1 / 2) + (-1) = (1 / 2) * (-1) : by norm_num,
  },
  case right {
    calc 
      (1 / 2) * (-1) = -1 / 2 : by norm_num,
      -1 / 2 = (1 / 2) / (-1) : by norm_num,
      (1 / 2) * (-1) = (1 / 2) / (-1) : by norm_num,
  }

end Vasya_numbers_l761_761346


namespace count_multiples_of_25_but_not_75_3_digit_l761_761133

theorem count_multiples_of_25_but_not_75_3_digit :
  let is_3_digit (n : ℕ) := (100 ≤ n) ∧ (n ≤ 999)
  let is_multiple_of_25 (n : ℕ) := ∃ k : ℕ, n = 25 * k
  let is_multiple_of_75 (n : ℕ) := ∃ m : ℕ, n = 75 * m
  (finset.filter (λ n : ℕ, is_3_digit n ∧ is_multiple_of_25 n ∧ ¬ is_multiple_of_75 n) (finset.range 1000)).card = 24 := by
  sorry

end count_multiples_of_25_but_not_75_3_digit_l761_761133


namespace find_m_value_l761_761558

theorem find_m_value (m : ℝ) (h : (m + 1) ≠ 0) (hq : m (m - 2) - 1 = 2) : m = 3 := by
  -- Since it's an informal theorem statement, we include sketch/outline of solution
  -- Step 1: Expand hq to form quadratic equation: m(m-2) - 1 = 2
  -- Step 2: Solve m^2 - 2m - 3 = 0
  -- Step 3: Factorize to (m - 3)(m + 1) = 0
  -- Step 4: Exclude m = -1 because m + 1 ≠ 0
  -- Step 5: Thus, m = 3
  sorry

end find_m_value_l761_761558


namespace sum_of_valid_x_l761_761997

theorem sum_of_valid_x :
  ∑ x in {x | ∃ y, x * y = 360 ∧ x ≥ 18 ∧ y ≥ 12}, x = 92 := by
  sorry

end sum_of_valid_x_l761_761997


namespace g_is_even_l761_761194

def g (x : ℝ) : ℝ := 4 / (3 * x^8 - 7)

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  sorry

end g_is_even_l761_761194


namespace BP_times_BQ_independent_of_choice_of_p_l761_761752

-- Define the geometrical setup

variables {R : Type*} [real_field R]
variables (O A B : point R) (sphere_center : point R) (sphere_radius : R)

-- Assuming the sphere center \(O\) and radius
def is_sphere (O : point R) (R : ℝ) (P : point R) : Prop :=
  dist O P = R

-- Define the condition that \(A\) and \(B\) are opposite sides of the plane
variable (plane : set (point R))

-- Define \( C \) as a circle that is the intersection of the sphere and a plane
def is_circle (C : set (point R)) : Prop :=
  ∃ (O : point R), O ∈ plane ∧ ∀ P ∈ C, dist O P = some_radius

-- Define that the line joining \( A \) to the center of the sphere is normal to the plane
def normal_to_plane (A O : point R) : Prop :=
  ∀ P ∈ plane, ⟪(O - A), (P - O)⟫ = 0

-- Define another plane \( p \) that intersects AB and the circle at \( P \) and \( Q \)
variables (p : set (point R)) (P Q : point R)

def intersects_AB (p : set (point R)) (A B : point R) : Prop :=
  ∃ X ∈ (segment A B), X ∈ p

def intersects_C (p : set (point R)) (C : set (point R)) (P Q : point R) : Prop :=
  P ∈ C ∧ Q ∈ C ∧ P ∈ p ∧ Q ∈ p

-- Formulate the theorem
theorem BP_times_BQ_independent_of_choice_of_p
  (is_sphere : is_sphere O sphere_radius) 
  (opposite_sides : ∀ P ∈ plane, A ∈ P ↔ ¬ B ∈ P)
  (normal : normal_to_plane A O)
  (intersects_AB : intersects_AB p A B)
  (intersects_C : intersects_C p C P Q) :
  ∃ k, ∀ (p : set (point R)), (intersects_AB p A B) → (intersects_C p C P Q) → BP P B * BQ Q B = k := 
sorry

end BP_times_BQ_independent_of_choice_of_p_l761_761752


namespace negation_of_exists_x_squared_gt_one_l761_761301

-- Negation of the proposition
theorem negation_of_exists_x_squared_gt_one :
  ¬ (∃ x : ℝ, x^2 > 1) ↔ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 :=
by
  sorry

end negation_of_exists_x_squared_gt_one_l761_761301


namespace pets_beds_calculation_l761_761632

theorem pets_beds_calculation
  (initial_beds : ℕ)
  (additional_beds : ℕ)
  (total_pets : ℕ)
  (H1 : initial_beds = 12)
  (H2 : additional_beds = 8)
  (H3 : total_pets = 10) :
  (initial_beds + additional_beds) / total_pets = 2 := 
by 
  sorry

end pets_beds_calculation_l761_761632


namespace find_smallest_z_l761_761565

theorem find_smallest_z (x y z : ℤ) (h1 : 7 < x) (h2 : x < 9) (h3 : x < y) (h4 : y < z) 
  (h5 : y - x = 7) : z = 16 :=
by
  sorry

end find_smallest_z_l761_761565


namespace smallest_n_integer_series_l761_761845

theorem smallest_n_integer_series : 
  let a := Real.pi / 2015 in
  ∃ n : ℕ, ( ∀ m : ℕ, m < n → ¬(2 * ∑ k in Finset.range (m+1), Real.cos ((k*(k+2))*a) * Real.sin (k*a)).isInteger) ∧
           (2 * ∑ k in Finset.range (n+1), Real.cos ((k*(k+2))*a) * Real.sin (k*a)).isInteger ∧
           n = 31 :=
by
  sorry

end smallest_n_integer_series_l761_761845


namespace travel_time_difference_l761_761737

theorem travel_time_difference 
  (speed : ℝ) (d1 d2 : ℝ) (h_speed : speed = 50) (h_d1 : d1 = 475) (h_d2 : d2 = 450) : 
  (d1 - d2) / speed * 60 = 30 := 
by 
  sorry

end travel_time_difference_l761_761737


namespace total_fruits_on_display_l761_761690

-- Declare the variables and conditions as hypotheses
variables (apples oranges bananas : ℕ)
variables (h1 : apples = 2 * oranges)
variables (h2 : oranges = 2 * bananas)
variables (h3 : bananas = 5)

-- Define what we want to prove
theorem total_fruits_on_display : apples + oranges + bananas = 35 :=
by sorry

end total_fruits_on_display_l761_761690


namespace Vasya_numbers_l761_761363

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 :=
by
  sorry

end Vasya_numbers_l761_761363


namespace circle_chord_intersection_l761_761943

theorem circle_chord_intersection
  (O : Type) [hd : metric_space O] [hc : circle O]
  (A B C D P Q : O)
  (h1 : diameter CD)
  (h2 : chord AB)
  (h3 : chord AQ)
  (h4 : P = intersection AB CD)
  (h5 : P = intersection AQ CD)
  (h6 : perpendicular CD AB at P)
  (h7 : midpoint P CD):
  (segment_length AP) * (segment_length AQ) = (segment_length CD / 2) ^ 2 :=
by sorry

end circle_chord_intersection_l761_761943


namespace infinite_quadratic_polynomials_l761_761044

namespace Polynomials

theorem infinite_quadratic_polynomials {a b c : ℝ}
  (h : a ≠ 0) (rs_eq_sum : a + b + c = (roots a b c).prod)
  : ∃ (infinitely_many : ℕ → ℝ), ∀ n, polynomial.degree (x^2 - x + (infinitely_many n)) = 2 :=
sorry

end Polynomials

end infinite_quadratic_polynomials_l761_761044


namespace slope_of_line_is_correct_l761_761799

noncomputable def parabola_equation : ℙ := { x, y // y^2 = 4 * x }

noncomputable def focus : ℕ := (1 : ℕ, 0 : ℕ)

def line_passing_focus (k : ℝ) (x : ℝ) : ℝ := k * (x - 1)

def midpoint_ab (x1 y1 x2 y2 : ℝ) : ℝ × ℝ := 
((x1 + x2) / 2, (y1 + y2) / 2)

def perpendicular_through_m (k : ℝ) (M_x M_y : ℝ) (x : ℝ) : ℝ := 
-M_y / (M_x - x) 

def directrix : ℝ := -1

theorem slope_of_line_is_correct (k : ℝ) :
  (
    ∃ (A B : ℝ × ℝ), 
    A.1^2 = 4 * A.2 ∧ B.1^2 = 4 * B.2 ∧ 
    ∃ (M : ℝ × ℝ), 
    M = midpoint_ab A.1 A.2 B.1 B.2 ∧ 
    let N := (directrix, perpendicular_through_m k M.1 M.2 directrix) in 
    |N.2 - A.2| = √((M.1 - A.1)^2 + (M.2 - B.2)^2) ∧
    |M.1 - B.1| = √((N.1 - M.1)^2 + (N.2 - M.2)^2)
  )
  → (k = (√3) / 3) :=
sorry

end slope_of_line_is_correct_l761_761799


namespace exist_point_with_distance_condition_l761_761843

theorem exist_point_with_distance_condition 
  (n : ℕ) (points : Fin 2n → Real × Real) (lines : Fin 3n → (Real × Real) × (Real × Real)) :
  ∃ P : Real × Real, 
    (∑ i in Finset.range (3n), distance_line_point (lines i) P) < 
    (∑ i in Finset.range (2n), distance_point_point (points i) P) := 
sorry

def distance_line_point : ((Real × Real) × (Real × Real)) → (Real × Real) → Real := 
  λ line P, ...  -- placeholder for the distance function from a point to a line

def distance_point_point : (Real × Real) → (Real × Real) → Real := 
  λ Q P, ...  -- placeholder for the distance function from a point to a point

end exist_point_with_distance_condition_l761_761843


namespace distinct_flags_count_l761_761742

/- Define the set of available colors -/
inductive Color
| red | white | blue | green | yellow

open Color

/- Define a strip as a combination of three colors with the given restrictions -/
structure Flag :=
(top middle bottom : Color)
(no_adjacent_same : top ≠ middle ∧ middle ≠ bottom)
(not_more_than_twice : (top = middle ∧ top ≠ bottom) ∨ 
                     (top ≠ middle ∧ middle = bottom) ∨ 
                     (top ≠ middle ∧ middle ≠ bottom ∧ top ≠ bottom))

/- Define the total number of flags -/
def num_distinct_flags : Nat :=
  80

/- State the theorem to be proved -/
theorem distinct_flags_count : 
  ∃ (f : Set Flag), f.card = num_distinct_flags := sorry

end distinct_flags_count_l761_761742


namespace exponentiation_problem_l761_761721

theorem exponentiation_problem : 2^3 * 2^2 * 3^3 * 3^2 = 6^5 :=
by sorry

end exponentiation_problem_l761_761721


namespace hyperbola_equation_l761_761876

-- Defining the ellipse and hyperbola conditions
def ellipse_eq (x y : ℝ) := (x^2)/27 + (y^2)/36 = 1
def intersects_at (x y : ℝ) := (x, y) = (Real.sqrt 15, 4)

-- Standard equation of the hyperbola we want to prove
def hyperbola_eq (x y : ℝ) := (y^2)/4 - (x^2)/5 = 1

-- The Lean theorem statement
theorem hyperbola_equation :
  (∀ x y : ℝ, ellipse_eq x y → intersects_at x y) →
  (hyperbola_eq (Real.sqrt 15) 4) :=
by
  intros h1 h2
  -- Proof would go here
  sorry

end hyperbola_equation_l761_761876


namespace y_range_for_conditions_l761_761919

theorem y_range_for_conditions (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 72) : -9 ≤ y ∧ y < -8 :=
sorry

end y_range_for_conditions_l761_761919


namespace caleb_spent_more_on_ice_cream_l761_761455

theorem caleb_spent_more_on_ice_cream :
  let num_ic_cream := 10
  let cost_ic_cream := 4
  let num_frozen_yog := 4
  let cost_frozen_yog := 1
  (num_ic_cream * cost_ic_cream - num_frozen_yog * cost_frozen_yog) = 36 := 
by
  sorry

end caleb_spent_more_on_ice_cream_l761_761455


namespace minimum_omega_l761_761042

open Real

theorem minimum_omega (ω : ℕ) (h_ω_pos : ω > 0) :
  (∃ k : ℤ, ω * (π / 6) + (π / 6) = k * π + (π / 2)) → ω = 2 :=
by
  sorry

end minimum_omega_l761_761042


namespace g_even_l761_761196

def g (x : ℝ) : ℝ := 4 / (3 * x^8 - 7)

theorem g_even : ∀ x : ℝ, g (-x) = g x := by
  -- here we would provide the proof, but we'll use sorry for now as specified
  sorry

end g_even_l761_761196


namespace find_ratio_l761_761546

-- Given that the tangent of angle θ (inclination angle) is -2
def tan_theta (θ : Real) : Prop := Real.tan θ = -2

theorem find_ratio (θ : Real) (h : tan_theta θ) :
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 1 / 3 := by
  sorry

end find_ratio_l761_761546


namespace max_cart_length_l761_761778

-- Definition of the problem setup
def corridor_width : ℝ := 1.5
def cart_width : ℝ := 1

-- The theorem to be proven
theorem max_cart_length : 
  ∃ (AD : ℝ), (∀ (t : ℝ), (1 < t ∧ t ≤ sqrt 2) → AD ≤ (3 * √2 - 2)) ∧
              AD = 3 * √2 - 2 :=
by sorry

end max_cart_length_l761_761778


namespace odd_prime_p_unique_l761_761489

def set_A (p : ℕ) : Finset ℕ :=
  ((Finset.range ((p - 1) / 2)).map (λ k, (k^2 + 1) % p))

def set_B (p g : ℕ) : Finset ℕ :=
  ((Finset.range ((p - 1) / 2)).map (λ k, g^k % p))

theorem odd_prime_p_unique (p g : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1)
  (hsets_eq : set_A p = set_B p g) : p = 3 := 
  sorry

end odd_prime_p_unique_l761_761489


namespace tobias_shoveled_7_driveways_l761_761328

theorem tobias_shoveled_7_driveways :
  let original_price := 95
  let discount_rate := 0.10
  let tax_rate := 0.05
  let monthly_allowance := 5
  let lawn_charge := 15
  let driveway_charge := 7
  let hourly_wage := 8
  let part_time_hours := 10
  let lawns_mowed := 4
  let change_left := 15

  let discounted_price := original_price * (1 - discount_rate)
  let total_price := discounted_price * (1 + tax_rate)
  
  let total_money_before_purchase := total_price + change_left

  let allowance := 3 * monthly_allowance
  let lawn_income := lawns_mowed * lawn_charge
  let part_time_income := part_time_hours * hourly_wage

  let total_earned := allowance + lawn_income + part_time_income

  let shoveling_income := total_earned - total_money_before_purchase
  let driveways_shoveled := shoveling_income / driveway_charge
  
  floor driveways_shoveled = 7 :=
by
  sorry

end tobias_shoveled_7_driveways_l761_761328


namespace cube_diagonals_angle_l761_761573

-- Given conditions
variable {V : Type} [inner_product_space ℝ V] [finite_dimensional ℝ V] [dim_eq : finite_dimensional.finrank ℝ V = 3]

-- Define a cube and the relevant space diagonals
structure cube (s : ℝ) (origin : V) :=
(is_cube : ∀ v1 v2 ∈ set.univ, inner_product_space.is_orthonormal_basis ℝ ![v1, v2])

-- Prove that the angle between the space diagonals of two adjacent faces of the cube is 60 degrees
theorem cube_diagonals_angle (c : cube s V):
  ∃ θ, θ = 60 ∧ ∀ v1 v2 ∈ set.univ, ∠(v1, v2) = θ := 
sorry

end cube_diagonals_angle_l761_761573


namespace angle_paq_gt_150_l761_761572

theorem angle_paq_gt_150
    (A B C E F P Q : Type)
    [Add A]
    [Add B]
    [Add C]
    [Add E]
    [Add F]
    [Add P]
    [Add Q]
    (condition1 : ∠A = 60)
    (condition2 : AngleBisectorsIntersectAt E F A B C)
    (condition3 : IsParallelogram B F P E)
    (condition4 : IsParallelogram C E Q F) :
  ∠PAQ > 150 :=
sorry

end angle_paq_gt_150_l761_761572


namespace right_triangle_hypotenuse_inequality_l761_761154

theorem right_triangle_hypotenuse_inequality
  (a b c m : ℝ)
  (h_right_triangle : c^2 = a^2 + b^2)
  (h_area_relation : a * b = c * m) :
  m + c > a + b :=
by
  sorry

end right_triangle_hypotenuse_inequality_l761_761154


namespace distance_EF_parabola_through_E_distance_DF_l761_761650

-- Define points E, F, C
structure Point :=
(x : ℝ)
(y : ℝ)

def E : Point := ⟨3, 5⟩
def F : Point := ⟨0, 11 / 4⟩
def C : Point := ⟨0, 2⟩

-- Define the distance formula
def distance (A B : Point) : ℝ :=
  real.sqrt ((B.x - A.x) ^ 2 + (B.y - A.y) ^ 2)

-- Distance between E and F
theorem distance_EF : distance E F = 15 / 4 :=
by sorry

-- Analytical expression of the parabola
def parabola (x : ℝ) : ℝ :=
  (1 / 3) * x^2 + 2

theorem parabola_through_E : parabola 3 = 5 :=
by sorry

-- Point D on the parabola with x-coordinate -2
def D : Point := ⟨-2, parabola (-2)⟩

theorem distance_DF : distance D F = 25 / 12 :=
by sorry

end distance_EF_parabola_through_E_distance_DF_l761_761650


namespace ellipse_circle_intersect_l761_761586

theorem ellipse_circle_intersect (k : ℝ) :
  (∃ (z : ℂ), (|z - 4| = 3 * |z + 4| ∧ |z| = k) ∧ 
  ∀ z1 z2 : ℂ, ((|z1 - 4| = 3 * |z1 + 4| ∧ |z1| = k) ∧ (|z2 - 4| = 3 * |z2 + 4| ∧ |z2| = k)) → z1 = z2) → k = 3 :=
by
  sorry

end ellipse_circle_intersect_l761_761586


namespace RSA_next_challenge_digits_l761_761271

theorem RSA_next_challenge_digits (previous_digits : ℕ) (prize_increase : ℕ) :
  previous_digits = 193 ∧ prize_increase > 10000 → ∃ N : ℕ, N = 212 :=
by {
  sorry -- Proof is omitted
}

end RSA_next_challenge_digits_l761_761271


namespace exists_palindrome_in_bases_l761_761626

theorem exists_palindrome_in_bases (K d : ℕ) (hK : 0 < K) (hd : 0 < d) :
  ∃ (n : ℕ) (b : Fin K → ℕ),
    (∀ (i : Fin K), is_palindrome_in_base n (b i) d) :=
sorry

end exists_palindrome_in_bases_l761_761626


namespace ellipse_equation_proof_max_min_values_of_TS_line_through_fixed_point_l761_761857

-- Given conditions
def ellipse_eq (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ a = 2 * b ∧ (x, y) = (-√3, 1/2) ∧ 
  (x^2 / a^2 + y^2 / b^2 = 1)

def lower_vertex (y : ℝ) : Prop := 
  y = -1

def intersection_products (x₁ y₁ x₂ y₂ k m : ℝ) : Prop := 
  k ≠ 0 ∧ m ≠ -1 ∧
  ∀ (x : ℝ), (x, k*x + m) ∈ set_of (λ p:ℝ × ℝ, (p.fst^2 / 4) + p.snd^2 = 1) →
  (x₁, y₁) = rect_intersection x k m ∧ (x₂, y₂) = rect_intersection x k m ∧
  ((x₁ / (y₁ + 1)) * (x₂ / (y₂ + 1)) = 2)

-- Prove statements
theorem ellipse_equation_proof : 
  (∃ (x y : ℝ), ellipse_eq x y) → ∃ x y : ℝ, x^2 / 4 + y^2 = 1 :=
sorry

theorem max_min_values_of_TS :
  (∃ x y : ℝ, ellipse_eq x y) → 
  let T := (λ x₀ y₀ : ℝ, x₀^2 / 4 + y₀^2 = 1) in
  let S := (1, 0) in
  ∃ min max : ℝ, 
    (∃ x₀ y₀, T x₀ y₀ → sqrt ((1 - x₀)^2 + y₀^2) = min) ∧
    (∃ x₀ y₀, T x₀ y₀ → sqrt ((1 - x₀)^2 + y₀^2) = max) ∧
    min = √(2 / 3) ∧ max = 3 :=
sorry

theorem line_through_fixed_point :
  (∃ (x₁ y₁ x₂ y₂ k m : ℝ), intersection_products x₁ y₁ x₂ y₂ k m) → 
  ∃ (x y : ℝ), y = 3 :=
sorry

end ellipse_equation_proof_max_min_values_of_TS_line_through_fixed_point_l761_761857


namespace range_of_x_l761_761507

noncomputable def A (x : ℝ) : ℤ := Int.ceil x

theorem range_of_x (x : ℝ) (h₁ : x > 0) (h₂ : A (2 * x * A x) = 5) : x ∈ Set.Ioc 1 (5 / 4 : ℝ) :=
sorry

end range_of_x_l761_761507


namespace average_of_divisibles_by_4_l761_761062

theorem average_of_divisibles_by_4 (a b : ℕ) (H1 : a = 6) (H2 : b = 38) : 
  (∑ i in Finset.filter (λ x, x % 4 = 0) (Finset.range (b + 1)) / 
    Finset.card (Finset.filter (λ x, x % 4 = 0) (Finset.range (b + 1)))) = 22 :=
by
  sorry

end average_of_divisibles_by_4_l761_761062


namespace problem1_problem2_l761_761905

noncomputable def vector_a (θ : ℝ) := (Real.cos θ, Real.sin θ)
noncomputable def vector_b := (2, -1)
noncomputable def dot_product (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2
noncomputable def vector_norm (v : ℝ × ℝ) := Real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Problem 1
theorem problem1 (θ : ℝ) (h : dot_product (vector_a θ) vector_b = 0) :
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 1 / 3 :=
sorry

-- Problem 2
theorem problem2 (θ : ℝ) (h : vector_norm ((vector_a θ).1 - 2, (vector_a θ).2 + 1) = 2) (hθ : 0 < θ ∧ θ < Real.pi / 2) :
  Real.sin (θ + Real.pi / 4) = 7 * Real.sqrt 2 / 10 :=
sorry

end problem1_problem2_l761_761905


namespace contrapositive_l761_761548

theorem contrapositive (p q : Prop) (h : p → q) : ¬q → ¬p :=
by
  sorry

end contrapositive_l761_761548


namespace angle_bisector_parallelogram_l761_761584

open_locale classical

variable {α : Type*} [field α] 

variables {A B C D M N Q : α} 
variables (AB BC CD DA : α) (AM CN : α)

-- Assume that ABCD is a parallelogram with given points M and N
def is_parallelogram (A B C D : α) : Prop := 
  (A + C = B + D ∧ AB = CD ∧ BC = DA)

-- M and N are such that AM = CN
def points_on_sides (M N : α) (AM CN : α) : Prop := 
  AM = CN

-- Q is the intersection of AN and CM
def intersection_point (A N C M Q : α) : Prop := 
  -- Roughly saying that Q lies on both line segments AN and CM
  A * N = C * M

-- Angle bisector property
def angle_bisector (D Q: α) : Prop :=
  -- Roughly say that DQ bisects angle D.
  sorry

theorem angle_bisector_parallelogram 
  (ABCD_parallelogram : is_parallelogram A B C D)
  (MN_on_sides : points_on_sides M N AM CN)
  (Q_intersection : intersection_point A N C M Q) :
  angle_bisector D Q :=
sorry

end angle_bisector_parallelogram_l761_761584


namespace equal_angles_l761_761080

-- Define the basic geometric elements and conditions
variables {A B C P C1 B1 : Type}
[EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C]
[EuclideanGeometry P] [EuclideanGeometry C1] [EuclideanGeometry B1]
(hacute : acute_angle A B C)
(hP : inside_angle P A B C)

-- Define the perpendicular conditions
(hPC1 : IsPerpendicular P C1 A B)
(hPB1 : IsPerpendicular P B1 A C)

-- Define right angle conditions
(hRightPC1 : ∠P C1 A = 90)
(hRightPB1 : ∠P B1 A = 90)

-- Statement to prove that the given angles are equal 
theorem equal_angles (hP : inside_angle P A B C) 
                     (hPC1 : IsPerpendicular P C1 A B)
                     (hPB1 : IsPerpendicular P B1 A C)
                     (hRightPC1 : ∠P C1 A = 90)
                     (hRightPB1 : ∠P B1 A = 90) :
  ∠C1 A P = ∠C1 B1 P :=
sorry

end equal_angles_l761_761080


namespace longest_diagonal_length_l761_761423

-- Defining conditions
variable (d1 d2 : ℝ)
variable (x : ℝ)
variable (area : ℝ)
variable (h_area : area = 144)
variable (h_ratio : d1 = 4 * x)
variable (h_ratio' : d2 = 3 * x)
variable (h_area_eq : area = 1 / 2 * d1 * d2)

-- The Lean statement, asserting the length of the longest diagonal is 8 * sqrt(6)
theorem longest_diagonal_length (x : ℝ) (h_area : 1 / 2 * (4 * x) * (3 * x) = 144) :
  4 * x = 8 * Real.sqrt 6 := by
sorry

end longest_diagonal_length_l761_761423


namespace length_MD_l761_761329

theorem length_MD {DE EF FD : ℝ} (h1 : DE = 10) (h2 : EF = 24) (h3 : FD = 26)
  (M : ℝ) (h4 : ∃ ω3 ω4 : set (ℝ × ℝ), tangent ω3 DF D ∧ tangent ω4 DE D ∧ passes_through ω3 E ∧ passes_through ω4 F ∧ intersection_not_D ω3 ω4 M):
  M = (25 * real.sqrt 364) / 364 :=
sorry

end length_MD_l761_761329


namespace unique_isolating_line_a_eq_2e_l761_761932

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (a x : ℝ) : ℝ := a * Real.log x

theorem unique_isolating_line_a_eq_2e (a : ℝ) (h : a > 0) :
  (∃ k b, ∀ x : ℝ, f x ≥ k * x + b ∧ k * x + b ≥ g a x) → a = 2 * Real.exp 1 :=
sorry

end unique_isolating_line_a_eq_2e_l761_761932


namespace sum_of_fractions_l761_761850

theorem sum_of_fractions (n : ℕ) (h : n ≥ 2) : 
  ∑ a b, (a < b ∧ b ≤ n ∧ a + b > n ∧ Nat.coprime a b) → (1 : ℚ) / (a * b) = 1 / 2 := 
  by 
  sorry

end sum_of_fractions_l761_761850


namespace min_m_value_l761_761979

def f (x : Real) : Real := Real.sin (x - Real.pi / 6)

theorem min_m_value :
  ∃ m, (∀ α ∈ Set.Icc (-(5 * Real.pi / 6)) (-(Real.pi / 2)), 
    ∃! β ∈ Set.Icc 0 m, f(α) + f(β) = 0)
  ∧ m = Real.pi / 2 := 
sorry

end min_m_value_l761_761979


namespace last_locker_opened_l761_761773

theorem last_locker_opened (k : ℕ) (h : k = 10) : ℕ :=
  let L₀ := 1
  let L₁ := 2
  let L := λ n : ℕ, if n = 0 then L₀ else if n = 1 then L₁ else (by exact (L (n - 2)) * 4 - 2)
  by 
  have h_even : 2 ∣ k := by sorry
  let result := if (k % 2 = 0) then ((4 ^ (k / 2) + 2) / 3) else ((4 ^ ((k + 1) / 2) + 2) / 3)
  have : result = 342 := by sorry
  exact this

end last_locker_opened_l761_761773


namespace find_standard_equation_and_slope_l761_761539

-- Define the ellipse and the points
variable (a b : ℝ)
variable (A : ℝ × ℝ := (0, -1))
variable (C : ℝ × ℝ → Prop := λ p, p.1^2 / a^2 + p.2^2 / b^2 = 1)
variable (P1 : ℝ × ℝ := (sqrt 3, 1 / 2))
variable (P2 : ℝ × ℝ := (1, sqrt 3 / 2))

-- Define the lines l1 and l2
variable (k1 : ℝ)
variable (l1 : ℝ → ℝ := λ x, k1 * x - 1)
variable (l2 : ℝ → ℝ := λ x, -1 / k1 * x - 1)

-- Define the intersections E and F
variable (E : ℝ × ℝ := ((1 / (k1 - 1)), (1 / (k1 - 1))))
variable (F : ℝ × ℝ := (1 / ((-1 / k1) - 1), 1 / ((-1 / k1) - 1)))

-- Define the condition that OE = OF
axiom oe_eq_of : abs (1 / (k1 - 1)) = abs (1 / ((-1 / k1) - 1))

-- State the theorem
theorem find_standard_equation_and_slope :
  C P1 ∧ C P2 ∧ a > b ∧ b > 0 ∧ oe_eq_of →
  (C = (λ p, p.1^2 / 4 + p.2^2 = 1)) ∧ (k1 = 1 + sqrt 2 ∨ k1 = 1 - sqrt 2) :=
by
  sorry

end find_standard_equation_and_slope_l761_761539


namespace compare_expressions_l761_761004

theorem compare_expressions (x : ℝ) : (x - 2) * (x + 3) > x^2 + x - 7 :=
by {
  -- below proof is left as an exercise
  sorry
}

end compare_expressions_l761_761004


namespace train_length_proof_l761_761763

-- Definitions for the given conditions
def train_speed_kmh : ℝ := 144
def train_crossing_time_sec : ℝ := 5

-- The goal is to prove the length of the train
def length_of_train : ℝ := 200

-- Convert speed from km/h to m/s
def train_speed_ms := train_speed_kmh * (1000 / 3600)

-- Prove the length of the train
theorem train_length_proof : train_speed_ms * train_crossing_time_sec = length_of_train :=
by
  sorry

end train_length_proof_l761_761763


namespace select_group_odd_number_of_girl_friends_l761_761738

variables {Girl Boy : Type}
variable (friends_with : Boy → Girl → Prop)

axiom each_boy_has_girl_friend : ∀ b : Boy, ∃ g : Girl, friends_with b g

theorem select_group_odd_number_of_girl_friends:
  ∃ (group : Finset (Girl ⊕ Boy)), 
  (group.card * 2 ≥ (Finset.univ : Finset (Girl ⊕ Boy)).card) ∧ 
  (∀ (b : Boy), b ∈ group →
    (group.filter_sum_left Girl Boy).filter (λ g : Girl, friends_with b g).card % 2 = 1) :=
sorry

end select_group_odd_number_of_girl_friends_l761_761738


namespace even_function_solution_l761_761571

theorem even_function_solution :
  ∀ (m : ℝ), (∀ x : ℝ, (m+1) * x^2 + (m-2) * x = (m+1) * x^2 - (m-2) * x) → (m = 2 ∧ ∀ x : ℝ, (2+1) * x^2 + (2-2) * x = 3 * x^2) :=
by
  sorry

end even_function_solution_l761_761571


namespace abs_neg_five_is_five_l761_761276

theorem abs_neg_five_is_five : abs (-5) = 5 := 
by 
  sorry

end abs_neg_five_is_five_l761_761276


namespace Vasya_numbers_l761_761344

theorem Vasya_numbers : 
  ∃ x y : ℚ, (x + y = x * y) ∧ (x * y = x / y) ∧ x = 1 / 2 ∧ y = -1 :=
by 
  use (1 / 2)
  use (-1)
  split
  case left {
    calc 
      (1 / 2) + (-1) = -1 / 2 : by norm_num,
      -1 / 2 = ((1 / 2) * (-1)) : by norm_num,
      (1 / 2) + (-1) = (1 / 2) * (-1) : by norm_num,
  },
  case right {
    calc 
      (1 / 2) * (-1) = -1 / 2 : by norm_num,
      -1 / 2 = (1 / 2) / (-1) : by norm_num,
      (1 / 2) * (-1) = (1 / 2) / (-1) : by norm_num,
  }

end Vasya_numbers_l761_761344


namespace unique_elements_in_set_l761_761902

theorem unique_elements_in_set (a : ℤ) (h : a ∈ ({0, 1, 2} : Finset ℤ)) :
  ({1, a^2 - a - 1, a^2 - 2*a + 2} : Finset ℤ).card = 3 ↔ a = 0 := by
  sorry

end unique_elements_in_set_l761_761902


namespace vasya_numbers_l761_761383

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : 
  (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) ∨ (x = y ∧ x = 0) :=
by 
  -- Placeholder to show where the proof would go
  sorry

end vasya_numbers_l761_761383


namespace distinct_meeting_points_l761_761332

theorem distinct_meeting_points (h1 : ∀ t (t ≥ 60), (∃ n,  n * 5 = t) ∧ (∃ m,  m * 8 = t)) :
  19 :=
by
  -- the proof goes here
  sorry

end distinct_meeting_points_l761_761332


namespace number_in_scientific_notation_l761_761960

/-- Condition: A constant corresponding to the number we are converting. -/
def number : ℕ := 9000000000

/-- Condition: The correct answer we want to prove. -/
def correct_answer : ℕ := 9 * 10^9

/-- Proof Problem: Prove that the number equals the correct_answer when expressed in scientific notation. -/
theorem number_in_scientific_notation : number = correct_answer := by
  sorry

end number_in_scientific_notation_l761_761960


namespace vasya_numbers_l761_761337

theorem vasya_numbers :
  ∃ x y : ℝ, x + y = x * y ∧ x + y = x / y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l761_761337


namespace find_n_find_m_l761_761674

noncomputable def a (n : Nat) : Nat :=
  Nat.recOn (λ _ => Nat) 1 (λ k ak => ak + 3) n

theorem find_n :
  (∃ n, a n = 700) ↔ (∃ n : Nat, n = 234) :=
by
  split
  case mp =>
    intro h
    obtain ⟨n, hn⟩ := h
    sorry
  case mpr =>
    intro h
    exact Exists.intro 234 sorry

theorem find_m :
  ∃ m : Nat, m = 60 :=
by
  sorry

end find_n_find_m_l761_761674


namespace opposite_numbers_abs_l761_761147

theorem opposite_numbers_abs (a b : ℤ) (h : a + b = 0) : |a - 2014 + b| = 2014 :=
by
  -- proof here
  sorry

end opposite_numbers_abs_l761_761147


namespace find_a_for_min_value_l761_761115

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 1 / x + a * real.log x 

theorem find_a_for_min_value (a : ℝ) (h : ∀ x ∈ set.Icc (1/2:ℝ) 1, f x a ≥ 0) (hx_min : ∃ x ∈ set.Icc (1/2:ℝ) 1, f x a = 0) : 
  a = 2 / real.log 2 :=
sorry

end find_a_for_min_value_l761_761115


namespace max_size_of_NBalancedSet_l761_761964

variable (n : ℕ)

structure NBalancedSet (B : Finset ℕ) : Prop :=
  (n_pos : 0 < n)
  (subset_of_three : ∀ (s : Finset ℕ), s.card = 3 → s ⊆ B → ∃ (a b : ℕ), a ∈ s ∧ b ∈ s ∧ a ≠ b ∧ knows a b)
  (subset_of_n : ∀ (s : Finset ℕ), s.card = n → s ⊆ B → ∃ (a b : ℕ), a ∈ s ∧ b ∈ s ∧ ¬ knows a b)

theorem max_size_of_NBalancedSet (B : Finset ℕ) (h : NBalancedSet n B) :
  B.card ≤ (n - 1) * (n + 2) / 2 := sorry

end max_size_of_NBalancedSet_l761_761964


namespace spiritual_connection_probability_l761_761705

def in_range (x : ℕ) (a b : ℕ) : Prop :=
  a ≤ x ∧ x ≤ b

def num_possible_outcomes : ℕ := 3 * 3
def num_favorable_outcomes : ℕ := 7

/-- 
Prove that the probability of two people having a spiritual connection,
given that they pick numbers randomly from the set {1, 2, 3}, 
is 7/9.
-/
theorem spiritual_connection_probability : 
  ∀ (a b : ℕ), in_range a 1 3 → in_range b 1 3 → (|a - b| ≤ 1) → 
  (num_favorable_outcomes : ℚ) / (num_possible_outcomes : ℚ) = 7 / 9 :=
by sorry

end spiritual_connection_probability_l761_761705


namespace Vasya_numbers_l761_761361

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 :=
by
  sorry

end Vasya_numbers_l761_761361


namespace sum_digits_10_pow_85_minus_85_l761_761484

-- Define the function that computes the sum of the digits
def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

-- Define the specific problem for n = 10^85 - 85
theorem sum_digits_10_pow_85_minus_85 : 
  sum_of_digits (10^85 - 85) = 753 :=
by
  sorry

end sum_digits_10_pow_85_minus_85_l761_761484


namespace problem_statement_l761_761888

theorem problem_statement (x : Set ℝ) (n : ℕ) (s2 : ℝ) (mean : ℝ)
  (h1 : n > 0) 
  (h2 : s2 = 0 → ∃ m : ℝ, ∀ i ∈ x, i = m)
  (data : List ℝ) (h3 : data = [2, 3, 5, 7, 8, 9, 9, 11])
  (histogram : List ℝ) (h4 : is_unimodal_and_symmetric histogram) : 
  A and C are correct :=
sorry

end problem_statement_l761_761888


namespace series_result_l761_761011

noncomputable def series_sum (u : ℕ → ℚ) (s : ℚ) : Prop :=
  ∑' n, u n = s

def nth_term (n : ℕ) : ℚ := (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

theorem series_result : series_sum nth_term (1 / 200) := by
  sorry

end series_result_l761_761011


namespace product_of_two_numbers_l761_761678

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := 
by {
  sorry
}

end product_of_two_numbers_l761_761678


namespace range_of_a_l761_761523

variable (a : ℝ)

def p : Prop := ∀ x : ℝ, x^2 - 2 * a * x + 4 > 0

def q : Prop := ∀ x : ℝ, 0 < x → 4 - 2 * a < 1 / x

theorem range_of_a
  (h1 : p ∨ q)
  (h2 : ¬ (p ∧ q))
  : a ∈ Set.Iic (-2) ∪ Set.Ioi 1.5 :=
sorry

end range_of_a_l761_761523


namespace union_complement_A_B_l761_761550

def A : set ℝ := { x | x^2 - 4 * x - 12 < 0 }

def B : set ℝ := { x | x < 2 }

def complement_B : set ℝ := { x | x ≥ 2 }

theorem union_complement_A_B : A ∪ complement_B = { x | x > -2 } :=
by
  sorry

end union_complement_A_B_l761_761550


namespace coefficient_x_term_l761_761918

noncomputable def m := ∫ x in 0..(π/2), ( √2 * sin (x + π/4) )

theorem coefficient_x_term : 
  let term := (√x - m / √x) ^ 6 in
  ∃ c : ℝ, (term.coeff 1) = c ∧ c = 60 :=
  by
    let term := (√x - m / √x) ^ 6
    sorry

end coefficient_x_term_l761_761918


namespace triangle_BC_length_l761_761443

theorem triangle_BC_length
  (y_eq_2x2 : ∀ (x : ℝ), ∃ (y : ℝ), y = 2 * x ^ 2)
  (area_ABC : ∃ (A B C : ℝ × ℝ), 
    A = (0, 0) ∧ (∃ (a : ℝ), B = (a, 2 * a ^ 2) ∧ C = (-a, 2 * a ^ 2) ∧ 2 * a ^ 3 = 128))
  : ∃ (a : ℝ), 2 * a = 8 := 
sorry

end triangle_BC_length_l761_761443


namespace Xiao_Hong_steps_per_kcal_l761_761708

-- Define the condition: Xiao Ming consumes the same energy for 1200 steps as Xiao Hong for 9000 steps
variable (x : ℝ) -- steps per 1 kcal for Xiao Hong
constant h₁ : (1200 / (x + 2)) = (9000 / x)

-- Define the required theorem to prove
theorem Xiao_Hong_steps_per_kcal :
  ∃ x : ℝ, (1200 / (x + 2)) = (9000 / x) :=
begin
  use sorry
end

end Xiao_Hong_steps_per_kcal_l761_761708


namespace difference_of_roots_eq_four_l761_761812

theorem difference_of_roots_eq_four (p : ℝ) :
  let f := fun x => x^2 - 2*p*x + (p^2 - 4) in
  ∃ r s : ℝ, (f r = 0 ∧ f s = 0 ∧ r ≥ s ∧ r - s = 4) :=
sorry

end difference_of_roots_eq_four_l761_761812


namespace number_of_acute_triangles_l761_761771

def num_triangles : ℕ := 7
def right_triangles : ℕ := 2
def obtuse_triangles : ℕ := 3

theorem number_of_acute_triangles :
  num_triangles - right_triangles - obtuse_triangles = 2 := by
  sorry

end number_of_acute_triangles_l761_761771


namespace product_series_eq_l761_761000

theorem product_series_eq :
  (∏ k in Finset.range 249, (4 * (k + 1)) / (4 * (k + 1) + 4)) = (1 / 250) :=
sorry

end product_series_eq_l761_761000


namespace angle_identity_at_point_l761_761107

noncomputable def distance (x y : ℝ) := real.sqrt (x^2 + y^2)

theorem angle_identity_at_point 
  {α : ℝ} (h_vertex_origin : true)
  (h_initial_side : true)
  (h_point_on_terminal_side : let P := (-2 : ℝ, -1 : ℝ) in 
    (P.1 = -2) ∧ (P.2 = -1)) :
  2*(cos α)^2 - sin (π - 2*α) = 4/5 :=
sorry

end angle_identity_at_point_l761_761107


namespace average_speed_approx_l761_761751

noncomputable def average_speed (distance1 distance2 distance3 speed1 speed2 speed3 : ℝ) : ℝ :=
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let time3 := distance3 / speed3
  let total_dist := distance1 + distance2 + distance3
  let total_time := time1 + time2 + time3
  total_dist / total_time

theorem average_speed_approx :
  average_speed (1 / 3) (1 / 3) (1 / 3) 4 10 6 ≈ 5.81 :=
by
  sorry

end average_speed_approx_l761_761751


namespace find_number_l761_761493

theorem find_number (x : ℕ) (h : x * 9999 = 724817410) : x = 72492 :=
sorry

end find_number_l761_761493


namespace inequality_proof_l761_761645

noncomputable def problem_statement (x y z : ℝ) (positive_x : 0 < x) (positive_y : 0 < y) (positive_z : 0 < z) 
  (condition : x * y * z + x * y + y * z + z * x = x + y + z + 1) : Prop :=
  (1 / 3) * (Real.sqrt ((1 + x^2) / (1 + x)) + Real.sqrt ((1 + y^2) / (1 + y)) + Real.sqrt ((1 + z^2) / (1 + z))) ≤ 
  ((x + y + z) / 3) ^ (5 / 8)

-- The statement below is what needs to be proven.
theorem inequality_proof (x y z : ℝ) (positive_x : 0 < x) (positive_y : 0 < y) (positive_z : 0 < z) 
  (condition : x * y * z + x * y + y * z + z * x = x + y + z + 1) : problem_statement x y z positive_x positive_y positive_z condition :=
sorry

end inequality_proof_l761_761645


namespace Aunt_Zhang_expenditure_is_negative_l761_761239

-- Define variables for the problem
def income_yuan : ℤ := 5
def expenditure_yuan : ℤ := 3

-- The theorem stating Aunt Zhang's expenditure in financial terms
theorem Aunt_Zhang_expenditure_is_negative :
  (- expenditure_yuan) = -3 :=
by
  sorry

end Aunt_Zhang_expenditure_is_negative_l761_761239


namespace number_of_valid_boards_l761_761386

def valid_board (n : ℕ) (board : List (List ℤ)) : Prop :=
  List.length board = n ∧ ∀ row, row ∈ board → List.length row = n ∧
  ∀ i j, i < n ∧ j < n → abs (board[i][j]) = 1 ∧ abs (board[i][j] + board[i][j+1] + board[i+1][j] + board[i+1][j+1]) ≤ 1

def count_valid_boards (n : ℕ) : ℕ :=
  -- Here we would need a predicate for counting valid boards, this is just a placeholder
  if n = 2007 then 2^2007 - 2 else 0

theorem number_of_valid_boards :
  count_valid_boards 2007 = 2^2007 - 2 := by
  sorry

end number_of_valid_boards_l761_761386


namespace factorize_x4_minus_64_l761_761487

theorem factorize_x4_minus_64 (x : ℝ) : (x^4 - 64) = (x^2 - 8) * (x^2 + 8) :=
by sorry

end factorize_x4_minus_64_l761_761487


namespace jake_weight_loss_l761_761601

theorem jake_weight_loss:
  ∀ (J S x : ℝ),
  (J + S = 224 ∧ J = 156) →
  (J - x = 2 * (S - x) ↔ x = 20) :=
by
  intros J S x h
  cases h with h1 h2
  rw [h2] -- Replace J with 156
  simp only [Real.add_eq_add_iff] at h1
  replace h1 : S = 68 := by linarith
  rw [h1]
  split
  { intro hx
    linarith }
  { intro hx
    subst hx
    linarith }
  sorry

end jake_weight_loss_l761_761601


namespace harmonic_mean_of_3_and_2048_is_6_l761_761295

theorem harmonic_mean_of_3_and_2048_is_6 :
  let a := 3
  let b := 2048
  let harmonic_mean := (2 * a * b) / (a + b)
  abs (harmonic_mean - 6) < 1 :=
by
  let a := 3
  let b := 2048
  let harmonic_mean := (2 * a * b) / (a + b)
  have abs (harmonic_mean - 6) < 1
  sorry

end harmonic_mean_of_3_and_2048_is_6_l761_761295


namespace independent_event_prob_l761_761156

theorem independent_event_prob (p : set α → ℝ) (a b : set α) (h_independent : ∀ s t : set α, p (s ∩ t) = p s * p t)
  (ha : p a = 4 / 5) (hb : p b = 2 / 5) : p (a ∩ b) = 8 / 25 :=
by
  have h := h_independent a b
  rw [ha, hb] at h
  exact h

end independent_event_prob_l761_761156


namespace perpendicular_ED_FD_l761_761956

-- Definitions of the conditions
variables {A B C D E F M : Type} [geometry (A B C D E F M)]

-- Equilateral condition
axiom equilateral_ABC : AB = AC

-- Midpoint of BC
axiom midpoint_D : midpoint D B C

-- CE perpendicular to AB and BE = BD
axiom perp_CE_AB : perp CE AB
axiom equal_BE_BD : BE = BD

-- Midpoint of BE
axiom midpoint_BE_M : midpoint M B E

-- F lies on minor arc of circumcircle of triangle ABD and MF perpendicular to BE
axiom arc_ADF : F ∈ arc_minor (circumcircle A B D) A D
axiom perp_MF_BE : perp MF BE

-- Theorem: Proving ED is perpendicular to FD
theorem perpendicular_ED_FD : perp ED FD :=
sorry

end perpendicular_ED_FD_l761_761956


namespace probability_of_Ravi_l761_761330

variable (P_Ram P_Ravi P_Ram_and_Ravi : ℝ)

-- Conditions
axiom P_Ram_val : P_Ram = (2 / 7)
axiom P_Ram_and_Ravi_val : P_Ram_and_Ravi = 0.05714285714285714
axiom independence : ∀ (P_Ram P_Ravi : ℝ), P_Ram_and_Ravi = P_Ram * P_Ravi

-- To prove
theorem probability_of_Ravi :
  P_Ravi = 0.2 :=
by
  have h : P_Ram * P_Ravi = 0.05714285714285714 := by rw [P_Ram_val, P_Ram_and_Ravi_val]
  sorry

end probability_of_Ravi_l761_761330


namespace shopping_expenses_l761_761605

-- Define the original price variables for Ms. Li and Ms. Zhang
def original_price_li : ℝ := 190
def original_price_zhang : ℝ := 390

-- Define the discounted price calculations
def discounted_price_li (x : ℝ) : ℝ := 0.9 * x
def discounted_price_zhang (y : ℝ) : ℝ := 0.9 * 300 + 0.8 * (y - 300)

-- The total paid by Ms. Li
def paid_li : ℝ := discounted_price_li original_price_li

-- The total paid by Ms. Zhang
def paid_zhang : ℝ := discounted_price_zhang original_price_zhang

-- The combined total they would have paid without any discount
def total_without_discount : ℝ := original_price_li + original_price_zhang

-- The additional savings if paid together
def additional_savings (x y : ℝ) : ℝ := (0.9 * x + 0.8 * (y - 300) + 300 * 0.9) - (300 * 0.9 + 0.8 * (x + y - 300))

-- The increased amount if no discount was applied
def increased_amount : ℝ := total_without_discount - (paid_li + paid_zhang)

theorem shopping_expenses : 
  (paid_li = 171) ∧ 
  (paid_zhang = 342) ∧ 
  (additional_savings original_price_li original_price_zhang = 19) ∧ 
  (increased_amount = 67) :=
by
  -- We assume the conditions are correct for given discounts and savings
  have h1 : paid_li = 0.9 * original_price_li, by rfl,
  have h2 : paid_li = 171, by norm_num [h1, original_price_li],
  have h3 : paid_zhang = 0.9 * 300 + 0.8 * (original_price_zhang - 300), by rfl,
  have h4 : paid_zhang = 342, by norm_num [h3, original_price_zhang],
  have h5 : additional_savings original_price_li original_price_zhang = 19, by norm_num,
  have h6 : increased_amount = 67, by norm_num [total_without_discount, paid_li, paid_zhang],
  exact ⟨h2, h4, h5, h6⟩,
sorry

end shopping_expenses_l761_761605


namespace extreme_value_f_for_m_eq_2_min_integer_m_for_inequality_l761_761116

open Real

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := log x + x - (1 / 2) * m * x^2

theorem extreme_value_f_for_m_eq_2 :
  (∃ x : ℝ, f x 2 = ∂x f x 2 = 0) ∧
  f 1 2 = max (λ x, f x 2) :=
sorry

theorem min_integer_m_for_inequality :
  ∀ x : ℝ, x > 0 → f x (2 : ℝ) ≤ 2 * x - 1 :=
sorry

end extreme_value_f_for_m_eq_2_min_integer_m_for_inequality_l761_761116


namespace num_correct_statements_is_one_l761_761502

noncomputable def num_correct_statements (a b l : Line) (alpha beta : Plane) : Nat :=
  let s1 := ¬(alpha ⊥ beta ∧ a ∈ β → a ⊥ alpha)
  let s2 := (alpha ∥ beta ∧ a ∈ alpha ∧ b ∈ beta → (∃ p : ℝ → ℝ → Point, ∀ t s : ℝ, p t s ∈ α ∧ p t s ∈ β ∧ (a ≠ l ∧ b ≠ l → a ⊥ b)))
  let s3 := (a ⊥ l ∧ b ⊥ l → ¬(a ⊥ b))
  cond s1 1 0 + cond s2 1 0 + cond s3 1 0

theorem num_correct_statements_is_one (a b l : Line) (alpha beta : Plane) :
  num_correct_statements a b l alpha beta = 1 := by
  sorry

end num_correct_statements_is_one_l761_761502


namespace factorial_mod_prime_l761_761152

theorem factorial_mod_prime (a b : ℕ) (p : ℕ) (hp : nat.prime p) (h : p = a + b + 1) :
  (p ∣ (a! * b! + 1)) ∨ (p ∣ (a! * b! - 1)) :=
sorry

end factorial_mod_prime_l761_761152


namespace ice_cream_flavors_l761_761143

theorem ice_cream_flavors (scoops flavors : ℕ) (hq : scoops = 5) (hf : flavors = 3) : (nat.choose (scoops + flavors - 1) (flavors - 1)) = 21 := by
  sorry

end ice_cream_flavors_l761_761143


namespace tan_cos_X_l761_761957

-- Geometric definitions and auxiliary functions
variables {X Y Z : Type*} [metric_space X] [metric_space Y] [metric_space Z]

-- Conditions
def angle_Z_eq_90 (h : triangle X Y Z) : Prop :=
sorry

def XY_eq_13 (h : triangle X Y Z) : Prop :=
dist X Y = 13

def YZ_eq_5 (h : triangle X Y Z) : Prop :=
dist Y Z = 5

-- Proving tan and cos of angle X
theorem tan_cos_X (h : triangle X Y Z) (hz90 : angle_Z_eq_90 h) (hXY : XY_eq_13 h) (hYZ : YZ_eq_5 h) :
  tan_angle X = 5 / 12 ∧ cos_angle X = 12 / 13 :=
sorry

end tan_cos_X_l761_761957


namespace ducks_percentage_non_heron_birds_l761_761614

theorem ducks_percentage_non_heron_birds
  (total_birds : ℕ)
  (geese_percent pelicans_percent herons_percent ducks_percent : ℝ)
  (H_geese : geese_percent = 20 / 100)
  (H_pelicans: pelicans_percent = 40 / 100)
  (H_herons : herons_percent = 15 / 100)
  (H_ducks : ducks_percent = 25 / 100)
  (hnz : total_birds ≠ 0) :
  (ducks_percent / (1 - herons_percent)) * 100 = 30 :=
by
  sorry

end ducks_percentage_non_heron_birds_l761_761614


namespace expression_integrality_l761_761078

def comb (m l : ℕ) : ℕ := (Nat.factorial m) / ((Nat.factorial l) * (Nat.factorial (m - l)))

theorem expression_integrality (l m : ℕ) (hlm : 1 ≤ l ∧ l < m) :
  let C := comb m l in
  (m + 8) % (l + 2) = 0 ↔ ∃ k : ℕ, (m - 3*l + 2) * C = k * (l + 2) * C := 
by
  sorry

end expression_integrality_l761_761078


namespace lake_coverage_day_17_l761_761757

-- Define the state of lake coverage as a function of day
def lake_coverage (day : ℕ) : ℝ :=
  if day ≤ 20 then 2 ^ (day - 20) else 0

-- Prove that on day 17, the lake was covered by 12.5% algae
theorem lake_coverage_day_17 : lake_coverage 17 = 0.125 :=
by
  sorry

end lake_coverage_day_17_l761_761757


namespace num_of_integers_l761_761302

theorem num_of_integers (n:int) (hn : 1 / 7 ≤ 6 / (n:ℝ) ∧ 6 / (n:ℝ) ≤ 1 / 4) : 
  24 ≤ n ∧ n ≤ 42 ∧ (n:ℕ).card = 19 :=
by
  sorry

end num_of_integers_l761_761302


namespace running_problem_l761_761642

variables (x y : ℝ)

theorem running_problem :
  (5 * x = 5 * y + 10) ∧ (4 * x = 4 * y + 2 * y) :=
by
  sorry

end running_problem_l761_761642


namespace value_of_square_l761_761567

theorem value_of_square :
  ∃ (x : ℕ), 9210 - 9124 = 210 - x ∧ x = 124 :=
begin
  sorry
end

end value_of_square_l761_761567


namespace find_a_value_median_A_correct_mode_B_correct_variance_B_correct_stability_comparison_l761_761176

-- Define the scores of A and B
def scores_A : List ℕ := [8, 9, 7, 9, 8, 6, 7, 8, 10, 8]
def scores_B : List ℕ := [6, 7, 9, 7, 9, 10, 8, 7, 7, 10]

-- Define the average of scores
def average : ℕ := 8

-- Given the scores and average, prove the hypothesis about a
theorem find_a_value (a : ℕ) :
  (8 + 9 + 7 + 9 + 8 + 6 + 7 + a + 10 + 8) / 10 = 8 → a = 8 := sorry

-- Define the median of A's scores
def median_A : ℕ := 8

-- Given the scores of A, prove the median is correct
theorem median_A_correct : median_A = 8 := sorry

-- Define the mode of B's scores
def mode_B : ℕ := 7

-- Given the scores of B, prove the mode is correct
theorem mode_B_correct : mode_B = 7 := sorry

-- Define the variance of A's scores
def variance_A : ℕ := 1.2

-- Define the variance of B's scores
def variance_B : ℕ := 2.4

-- Given the scores of B, prove the variance is correct
theorem variance_B_correct : variance_B = 2.4 := sorry

-- Prove that A's scores are more stable than B's scores
theorem stability_comparison : variance_A < variance_B → "A's scores are more stable" := sorry

end find_a_value_median_A_correct_mode_B_correct_variance_B_correct_stability_comparison_l761_761176


namespace integer_multiplied_by_b_l761_761151

variable (a b : ℤ) (x : ℤ)

theorem integer_multiplied_by_b (h1 : -11 * a < 0) (h2 : x < 0) (h3 : (-11 * a * x) * (x * b) + a * b = 89) :
  x = -1 :=
by
  sorry

end integer_multiplied_by_b_l761_761151


namespace find_a_l761_761891

-- Definition of the piecewise function f
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a * x

-- Condition given in the problem
theorem find_a (a : ℝ) : 
  (f a) (f a 0) = 3 * a → a = 4 := 
by
  sorry

end find_a_l761_761891


namespace trader_marked_price_theorem_l761_761762

-- Define the necessary parameters and conditions
variables (CP MP SP : ℝ)
variables (x : ℝ)  -- x is the percentage above the cost price
variables (discount loss : ℝ)

-- Define the conditions
def condition1 : Prop := discount = 0.10
def condition2 : Prop := loss = 0.01
def marked_price : Prop := MP = ((1 + x / 100) * CP)
def selling_price : Prop := SP = (1 - discount) * MP
def selling_price_condition : Prop := SP = (1 - loss) * CP

-- The theorem to prove
theorem trader_marked_price_theorem 
  (CP MP SP : ℝ)
  (x : ℝ)
  (discount loss : ℝ)
  (cond1 : condition1 discount)
  (cond2 : condition2 loss)
  (mp_cond : marked_price CP MP x)
  (sp_cond : selling_price CP MP SP discount)
  (sp_loss_cond : selling_price_condition CP SP loss) 
  : x = 10 := 
sorry

end trader_marked_price_theorem_l761_761762


namespace intersection_shape_is_rectangle_l761_761672

noncomputable def hyperbola (x y : ℝ) : Prop := x * y = 20
noncomputable def circle (x y : ℝ) : Prop := x^2 + y^2 = 50

theorem intersection_shape_is_rectangle :
  ∃ (points : list (ℝ × ℝ)), (∀ (p : ℝ × ℝ), p ∈ points → hyperbola p.1 p.2 ∧ circle p.1 p.2) ∧ 
  shape_formed_by points = rectangle :=
sorry

end intersection_shape_is_rectangle_l761_761672


namespace tangent_length_l761_761761

theorem tangent_length (r d : ℕ) (h₁ : r = 36) (h₂ : d = 85) : ∃ l : ℕ, l = 77 ∧ l^2 = d^2 - r^2 :=
by 
  existsi 77
  split
  sorry

end tangent_length_l761_761761


namespace b_finish_remaining_work_in_5_days_l761_761395

theorem b_finish_remaining_work_in_5_days
  (a_work_rate: ℕ → ℚ) (b_work_rate: ℕ → ℚ) :
  (a_work_rate 4 = 1) → (b_work_rate 14 = 1) →
  (∀ (d: ℕ), d = (2 : ℕ) → (a_work_rate 4 + b_work_rate 14) * d = 9/14) →
  (∀ (remaining_work: ℚ), remaining_work = 1 - 9/14 → 
  (remaining_work / (b_work_rate 14)) = 5) :=
begin
  intros h1 h2 h3 h4,
  sorry
end

end b_finish_remaining_work_in_5_days_l761_761395


namespace value_of_a_minus_2_b_minus_2_l761_761215

theorem value_of_a_minus_2_b_minus_2 :
  ∀ (a b : ℝ), (a + b = -4/3 ∧ a * b = -7/3) → ((a - 2) * (b - 2) = 0) := by
  sorry

end value_of_a_minus_2_b_minus_2_l761_761215


namespace spring_membership_decrease_l761_761764

theorem spring_membership_decrease (init_members : ℝ) (increase_percent : ℝ) (total_change_percent : ℝ) 
  (fall_members := init_members * (1 + increase_percent / 100)) 
  (spring_members := init_members * (1 + total_change_percent / 100)) :
  increase_percent = 8 → total_change_percent = -12.52 → 
  (fall_members - spring_members) / fall_members * 100 = 19 :=
by
  intros h1 h2
  -- The complicated proof goes here.
  sorry

end spring_membership_decrease_l761_761764


namespace max_norm_of_linear_combination_l761_761863

noncomputable section

variables {ℝ : Type} [Nontrivial ℝ] [NormedSpace ℝ ℝ]

theorem max_norm_of_linear_combination
  (a b : ℝ)
  (m n : ℝ)
  (λ1 λ2 : ℝ)
  (ha : ∥a∥ = m)
  (hb : ∥b∥ = n)
  (h_neq_zero_a : a ≠ 0)
  (h_neq_zero_b : b ≠ 0) :
  ∥λ1 • a + λ2 • b∥ ≤ |λ1| * m + |λ2| * n :=
sorry

end max_norm_of_linear_combination_l761_761863


namespace team_total_points_l761_761470

theorem team_total_points (Connor_score Amy_score Jason_score : ℕ) :
  Connor_score = 2 →
  Amy_score = Connor_score + 4 →
  Jason_score = 2 * Amy_score →
  Connor_score + Amy_score + Jason_score = 20 :=
by
  intros
  sorry

end team_total_points_l761_761470


namespace solve_for_y_l761_761809

def star (x y : ℝ) : ℝ := 5 * x - 2 * y + 3 * x * y

theorem solve_for_y (y : ℝ) : star 2 y = 10 → y = 0 := by
  intro h
  sorry

end solve_for_y_l761_761809


namespace factorial_sum_remainder_mod_30_l761_761164

theorem factorial_sum_remainder_mod_30 :
  (∑ n in Finset.range 101, Nat.factorial n) % 30 = 3 :=
by
  sorry

end factorial_sum_remainder_mod_30_l761_761164


namespace correct_regression_equation_l761_761535

noncomputable def mean_x : ℝ := 4
noncomputable def mean_y : ℝ := 6.5

-- This represents the fact that x and y are negatively correlated.
def negatively_correlated (x y : ℝ) : Prop := sorry

def regression_equation (x y : ℝ) : ℝ := -2 * x + 14.5

theorem correct_regression_equation 
  (hx : mean_x = 4) 
  (hy : mean_y = 6.5)
  (h_corr : negatively_correlated mean_x mean_y) :
  ∃ (b : ℝ) (a : ℝ), (∀ x, y = a * x + b) :=
begin
  use [14.5, -2],
  sorry
end

end correct_regression_equation_l761_761535


namespace beautiful_equation_probability_correct_l761_761420

def is_beautiful (a b m : ℕ) : Prop :=
  m ∈ {1, 2, 3, 4} ∧ (a - m) * (m - b) = 0

def beautiful_equation_probability : ℚ :=
  let outcomes := [(1, 2), (2, 3), (3, 4)]
  in outcomes.length / 16

theorem beautiful_equation_probability_correct :
  beautiful_equation_probability = 3 / 16 :=
by sorry

end beautiful_equation_probability_correct_l761_761420


namespace binary_to_decimal_101_l761_761025

theorem binary_to_decimal_101 : ∑ (i : Fin 3), (Nat.digit 2 ⟨i, sorry⟩ (list.nth_le [1, 0, 1] i sorry)) * (2 ^ i) = 5 := 
  sorry

end binary_to_decimal_101_l761_761025


namespace weight_of_purple_ring_l761_761201

noncomputable section

def orange_ring_weight : ℝ := 0.08333333333333333
def white_ring_weight : ℝ := 0.4166666666666667
def total_weight : ℝ := 0.8333333333

theorem weight_of_purple_ring :
  total_weight - orange_ring_weight - white_ring_weight = 0.3333333333 :=
by
  -- We'll place the statement here, leave out the proof for skipping.
  sorry

end weight_of_purple_ring_l761_761201


namespace binary_addition_correct_l761_761001

-- define the binary numbers as natural numbers using their binary representations
def bin_1010 : ℕ := 0b1010
def bin_10 : ℕ := 0b10
def bin_sum : ℕ := 0b1100

-- state the theorem that needs to be proved
theorem binary_addition_correct : bin_1010 + bin_10 = bin_sum := by
  sorry

end binary_addition_correct_l761_761001


namespace trig_proof_l761_761559

theorem trig_proof (α : ℝ) (h : sqrt 3 * sin α + cos α = 1 / 2) :
  cos (2 * α + 4 * π / 3) = -7 / 8 :=
by
  sorry

end trig_proof_l761_761559


namespace calc_expr_l761_761002

theorem calc_expr :
  (-1) * (-3) + 3^2 / (8 - 5) = 6 :=
by
  sorry

end calc_expr_l761_761002


namespace omi_age_l761_761613

theorem omi_age: 
  ∃ (O K A : ℕ), 
  K = 28 ∧ A = (3 * K) / 4 ∧ (K + A + O) / 3 = 35 ∧ O = 56 := 
by
  trivial

end omi_age_l761_761613


namespace opposite_angles_obtuse_diagonal_shorter_l761_761728

theorem opposite_angles_obtuse_diagonal_shorter {A B C D : Point} (hquad : Quadrilateral A B C D)
  (hB : obtuse_angle A B C) (hD : obtuse_angle A D C) : 
  dist B D < dist A C :=
sorry

end opposite_angles_obtuse_diagonal_shorter_l761_761728


namespace product_of_two_numbers_l761_761680

theorem product_of_two_numbers (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : a * b = 875 :=
sorry

end product_of_two_numbers_l761_761680


namespace triangle_lengths_and_area_l761_761934

theorem triangle_lengths_and_area
  (A B C : Type)
  [preorder A] [decidable_eq A] [decidable_rel ((<) : A → A → Prop)]
  [has_zero A] [has_add A] [has_mul A] [has_div A] [has_sub A]
  (angle_A : A)
  (sin_B : A)
  (len_AC : A)
  (H1 : angle_A = 90)
  (H2 : sin_B = 3 / 5)
  (H3 : len_AC = 15) :
  ∃ (AB BC : A) (area : A), AB = 9 ∧ BC = 12 ∧ area = 54 :=
by
  sorry

end triangle_lengths_and_area_l761_761934


namespace unique_solution_inequality_l761_761477

def valid_a : ℝ := -1

theorem unique_solution_inequality (a : ℕ) (h : valid_a a) :
  (∀ x : ℝ, |x^2 + 2 * a * x + 4 * a| ≤ 2) ↔ a = -1 :=
by sorry

end unique_solution_inequality_l761_761477


namespace jane_crayons_l761_761603

theorem jane_crayons :
  let start := 87
  let eaten := 7
  start - eaten = 80 :=
by
  sorry

end jane_crayons_l761_761603


namespace Vasya_numbers_l761_761348

theorem Vasya_numbers : 
  ∃ x y : ℚ, (x + y = x * y) ∧ (x * y = x / y) ∧ x = 1 / 2 ∧ y = -1 :=
by 
  use (1 / 2)
  use (-1)
  split
  case left {
    calc 
      (1 / 2) + (-1) = -1 / 2 : by norm_num,
      -1 / 2 = ((1 / 2) * (-1)) : by norm_num,
      (1 / 2) + (-1) = (1 / 2) * (-1) : by norm_num,
  },
  case right {
    calc 
      (1 / 2) * (-1) = -1 / 2 : by norm_num,
      -1 / 2 = (1 / 2) / (-1) : by norm_num,
      (1 / 2) * (-1) = (1 / 2) / (-1) : by norm_num,
  }

end Vasya_numbers_l761_761348


namespace geometric_sequence_mean_l761_761592

theorem geometric_sequence_mean (a : ℕ → ℝ) (q : ℝ) (h_q : q = -2) 
  (h_condition : a 3 * a 7 = 4 * a 4) : 
  ((a 8 + a 11) / 2 = -56) 
:= sorry

end geometric_sequence_mean_l761_761592


namespace geometry_proof_problem_l761_761935

/-
In Δ XYZ, XY = 12, ∠X = 45°, and ∠Z = 60°. Let H, D, and M be points on the line YZ such that XH ⊥ YZ, ∠YXD = ∠DXZ, and YM = MZ. Point N is the midpoint of the segment HM, and point P is on ray XD such that PN ⊥ YZ. Then XP^2 = m/n, where m and n are relatively prime positive integers. Prove that m + n = 13.
-/
theorem geometry_proof_problem
  (X Y Z H D M N P : Type)
  (XY : ℕ)
  (angle_X : ℕ)
  (angle_Z : ℕ)
  (XY_eq_12 : XY = 12)
  (angle_X_eq_45 : angle_X = 45)
  (angle_Z_eq_60 : angle_Z = 60)
  (XH_perp_YZ : XH ⊥ YZ)
  (angle_YXD_eq_angle_DXZ : ∠YXD = ∠DXZ)
  (YM_eq_MZ : YM = MZ)
  (H_midpoint_of_HM : H.is_midpoint N M)
  (PN_perp_YZ : PN ⊥ YZ)
  (m n : ℕ)
  (rel_prime : gcd m n = 1)
  (XP_sq : XP^2 = (m : ℚ) / (n : ℚ)) :
  m + n = 13 :=
by
  sorry

end geometry_proof_problem_l761_761935


namespace positive_real_triangle_inequality_l761_761731

theorem positive_real_triangle_inequality
    (a b c : ℝ)
    (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
    (h : 5 * a * b * c > a^3 + b^3 + c^3) :
    a + b > c ∧ a + c > b ∧ b + c > a :=
by
  sorry

end positive_real_triangle_inequality_l761_761731


namespace series_result_l761_761012

noncomputable def series_sum (u : ℕ → ℚ) (s : ℚ) : Prop :=
  ∑' n, u n = s

def nth_term (n : ℕ) : ℚ := (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

theorem series_result : series_sum nth_term (1 / 200) := by
  sorry

end series_result_l761_761012


namespace part1_solution_part2_solution_l761_761803

noncomputable def f (x : ℝ) := |x - 1| + |x - 3|

theorem part1_solution (x : ℝ) : 
  f(x) ≤ x + 1 ↔ (1 ≤ x ∧ x ≤ 5) :=
sorry

theorem part2_solution (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : 
  (a^2 / (a + 1)) + (b^2 / (b + 1)) ≥ 1 :=
sorry

end part1_solution_part2_solution_l761_761803


namespace y_intercept_of_line_l761_761759
noncomputable def slope_intercept_form (m x b : ℝ) : ℝ := m * x + b

theorem y_intercept_of_line :
  ∃ b : ℝ, ∀ (m x y : ℝ), m = 9.9 → x = 100 → y = 1000 → y = slope_intercept_form m x b ∧ b = 10 :=
begin
  sorry
end

end y_intercept_of_line_l761_761759


namespace probability_at_least_half_girls_l761_761608

-- Conditions
def six_children : ℕ := 6
def prob_girl : ℝ := 0.5

-- Statement to prove
theorem probability_at_least_half_girls :
  (∑ k in finset.range (six_children + 1), if 3 ≤ k then ↑(nat.binomial six_children k) * (prob_girl ^ k) * ((1 - prob_girl) ^ (six_children - k)) else 0) = 21 / 32 :=
by sorry

end probability_at_least_half_girls_l761_761608


namespace jane_needs_9_more_days_l761_761963

def jane_rate : ℕ := 16
def mark_rate : ℕ := 20
def mark_days : ℕ := 3
def total_vases : ℕ := 248

def vases_by_mark_in_3_days : ℕ := mark_rate * mark_days
def vases_by_jane_and_mark_in_3_days : ℕ := (jane_rate + mark_rate) * mark_days
def remaining_vases_after_3_days : ℕ := total_vases - vases_by_jane_and_mark_in_3_days
def days_jane_needs_alone : ℕ := (remaining_vases_after_3_days + jane_rate - 1) / jane_rate

theorem jane_needs_9_more_days :
  days_jane_needs_alone = 9 :=
by
  sorry

end jane_needs_9_more_days_l761_761963


namespace RookGameOptimalPlay_l761_761311

theorem RookGameOptimalPlay :
  ∀ (rook : ℕ × ℕ) (player_turn : ℕ), rook = (1, 1) ∧ player_turn = 1 → (∃ k, rook = (8, 8)) → (optimal_play rook player_turn = 2) := 
by 
  sorry

end RookGameOptimalPlay_l761_761311


namespace conjugate_of_complex_number_l761_761538

noncomputable def complex_number : ℂ := (1 + 3 * complex.i) / (1 - complex.i)

theorem conjugate_of_complex_number :
  complex.conj complex_number = -1 - 2 * complex.i := by
sorry

end conjugate_of_complex_number_l761_761538


namespace ellipse_foci_condition_l761_761887

theorem ellipse_foci_condition (a b : ℝ) (hx : a > b) (hy : a > 0) (hz : b > 0) : 
  (∀ x y : ℝ, (x^2 / a + y^2 / b = 1) → ellipse_foci_on_x_axis_condition hx hy hz) :=
sorry

def ellipse_foci_on_x_axis_condition (hx hy hz : Prop) : Prop :=
  (hx ∧ hy ∧ hz)

end ellipse_foci_condition_l761_761887


namespace combined_degrees_l761_761263

-- Definitions based on conditions
def summer_degrees : ℕ := 150
def jolly_degrees (summer_degrees : ℕ) : ℕ := summer_degrees - 5

-- Theorem stating the combined degrees
theorem combined_degrees : summer_degrees + jolly_degrees summer_degrees = 295 :=
by
  sorry

end combined_degrees_l761_761263


namespace matrix_multiplication_l761_761463

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 1], ![4, -2]]

def matrix2 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![7, -3], ![2, 2]]

def product_matrix : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![23, -7], ![24, -16]]

theorem matrix_multiplication :
  matrix1 ⬝ matrix2 = product_matrix := by
  sorry

end matrix_multiplication_l761_761463


namespace symmetric_intersection_points_eq_y_axis_l761_761928

theorem symmetric_intersection_points_eq_y_axis (k : ℝ) :
  (∀ x y : ℝ, (y = k * x + 1) ∧ (x^2 + y^2 + k * x - y - 9 = 0) → (∃ x' : ℝ, y = k * (-x') + 1 ∧ (x'^2 + y^2 + k * x' - y - 9 = 0) ∧ x' = -x)) →
  k = 0 :=
by
  sorry

end symmetric_intersection_points_eq_y_axis_l761_761928


namespace math_problem_l761_761946

-- Definitions based on given conditions
def parametric_curve (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, Real.sin α)

def polar_line (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - π / 4) = √2 / 2

def point_Q : ℝ × ℝ := (2, 3)

-- Prove the key results as stated in the problem
theorem math_problem :
  (∀ x y α, (x, y) = parametric_curve α → (x^2) / 4 + y^2 = 1) ∧ 
  (∀ ρ θ, polar_line ρ θ → θ = π / 4) ∧ 
  (∃ A B : ℝ × ℝ,
    (A = (0, 1) ∧ B = (-8/5, -3/5)) ∧
    let QA := Real.sqrt ((point_Q.1 - 0)^2 + (point_Q.2 - 1)^2),
        QB := Real.sqrt ((point_Q.1 + 8/5)^2 + (point_Q.2 + 3/5)^2)
    in QA + QB = 28 * Real.sqrt 2 / 5) :=
by
  sorry

end math_problem_l761_761946


namespace find_coefficients_and_extrema_l761_761110

noncomputable def f (a b c x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_coefficients_and_extrema (a b c : ℝ) :
  let f' := λ x : ℝ, 3 * x^2 + 2 * a * x + b,
      f1 := f a b c 1,
      fTangentLine := 3 * 1 - f1 + 1 = 0,
      fPrime1 := f' 1,
      slope_tangent := f' 1 = 3,
      extreme_value := f' 1 = 0 in
  (∀ x, f' x = 3 * x^2 + 2 * a * x + b) ∧ 
  (2 * a + b = 0) ∧ 
  (3 + 2 * a + b = 0) ∧ 
  f a b c 1 = 4 ∧ 
  f (-3) = 8 ∧ 
  f (-2) = 13 ∧ 
  (max (f (-3)) (f (-2)) = 13) ∧ 
  a = 2 ∧ 
  b = -4 ∧ 
  c = 5 := 
sorry

end find_coefficients_and_extrema_l761_761110


namespace solve_problem_1_solve_problem_2_l761_761782

/-
Problem 1:
Given the equation 2(x - 1)^2 = 18, prove that x = 4 or x = -2.
-/
theorem solve_problem_1 (x : ℝ) : 2 * (x - 1)^2 = 18 → (x = 4 ∨ x = -2) :=
by
  sorry

/-
Problem 2:
Given the equation x^2 - 4x - 3 = 0, prove that x = 2 + √7 or x = 2 - √7.
-/
theorem solve_problem_2 (x : ℝ) : x^2 - 4 * x - 3 = 0 → (x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) :=
by
  sorry

end solve_problem_1_solve_problem_2_l761_761782


namespace Adam_total_candy_l761_761440

theorem Adam_total_candy :
  (2 + 5) * 4 = 28 := 
by 
  sorry

end Adam_total_candy_l761_761440


namespace part1_part2_l761_761524

def quadratic_inequality_A (x m : ℝ) := -x^2 + 2 * m * x + 4 - m^2 ≥ 0
def quadratic_inequality_B (x : ℝ) := 2 * x^2 - 5 * x - 7 < 0

theorem part1 (m : ℝ) :
  (∀ x, quadratic_inequality_A x m ∧ quadratic_inequality_B x ↔ 0 ≤ x ∧ x < 7 / 2) →
  m = 2 := by sorry

theorem part2 (m : ℝ) :
  (∀ x, quadratic_inequality_B x → ¬ quadratic_inequality_A x m) →
  m ≤ -3 ∨ 11 / 2 ≤ m := by sorry

end part1_part2_l761_761524


namespace common_chord_circle_equation_l761_761289

theorem common_chord_circle_equation :
  let C1 := ∀ x y : ℝ, x^2 + y^2 - 12 * x - 2 * y - 13 = 0
  let C2 := ∀ x y : ℝ, x^2 + y^2 + 12 * x + 16 * y - 25 = 0
  ∃ (x0 y0 r : ℝ), ((x - x0)^2 + (y - y0)^2 = r) ∧
  (∃ x y : ℝ, C1 x y) ∧
  (∃ x y : ℝ, C2 x y) :=
sorry

end common_chord_circle_equation_l761_761289


namespace tan_theta_minus_pi_over_4_l761_761909

theorem tan_theta_minus_pi_over_4 (θ : ℝ) (h : Real.cos θ - 3 * Real.sin θ = 0) :
  Real.tan (θ - Real.pi / 4) = -1 / 2 :=
sorry

end tan_theta_minus_pi_over_4_l761_761909


namespace base_conversion_subtraction_l761_761054

def base6_to_nat (d0 d1 d2 d3 d4 : ℕ) : ℕ :=
  d4 * 6^4 + d3 * 6^3 + d2 * 6^2 + d1 * 6^1 + d0 * 6^0

def base7_to_nat (d0 d1 d2 d3 : ℕ) : ℕ :=
  d3 * 7^3 + d2 * 7^2 + d1 * 7^1 + d0 * 7^0

theorem base_conversion_subtraction :
  base6_to_nat 1 2 3 5 4 - base7_to_nat 1 2 3 4 = 4851 := by
  sorry

end base_conversion_subtraction_l761_761054


namespace isolating_line_unique_l761_761930

noncomputable def f (x : ℝ) := x^2
noncomputable def g (a x : ℝ) := a * log x

theorem isolating_line_unique (a : ℝ) (hx : ∀ x, f x ≥ g a x ∧ g a x ≥ f x) :
  a = 2 * real.exp 1 := 
sorry

end isolating_line_unique_l761_761930


namespace find_value_of_expression_l761_761071

theorem find_value_of_expression (x : ℝ) (h : 5 * x - 3 = 7) : 3 * x^2 + 2 = 14 :=
by {
  sorry
}

end find_value_of_expression_l761_761071


namespace count_integers_divisible_by_neither_5_nor_7_l761_761303

theorem count_integers_divisible_by_neither_5_nor_7 :
  (∃ n : ℕ, n = 343 ∧ ∀ x : ℕ, x < 500 → (x % 5 ≠ 0) ∧ (x % 7 ≠ 0) ↔ x ∈ (finset.range 500).filter (λ k, k % 5 ≠ 0 ∧ k % 7 ≠ 0)).count = 343 :=
by sorry

end count_integers_divisible_by_neither_5_nor_7_l761_761303


namespace total_distance_traveled_l761_761396

noncomputable def total_distance (d v1 v2 v3 time_total : ℝ) : ℝ :=
  3 * d

theorem total_distance_traveled
  (d : ℝ)
  (v1 : ℝ := 3)
  (v2 : ℝ := 6)
  (v3 : ℝ := 9)
  (time_total : ℝ := 11 / 60)
  (h : d / v1 + d / v2 + d / v3 = time_total) :
  total_distance d v1 v2 v3 time_total = 0.9 :=
by
  sorry

end total_distance_traveled_l761_761396


namespace minimum_value_of_sum_of_squares_l761_761983

theorem minimum_value_of_sum_of_squares :
  ∃ p q r s t u v w x : ℤ,
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧ p ≠ x ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧ q ≠ x ∧
  r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧ r ≠ x ∧
  s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧ s ≠ x ∧
  t ≠ u ∧ t ≠ v ∧ t ≠ w ∧ t ≠ x ∧
  u ≠ v ∧ u ≠ w ∧ u ≠ x ∧
  v ≠ w ∧ v ≠ x ∧
  w ≠ x ∧
  p ∈ {-8, -6, -4, -1, 1, 3, 5, 7, 9} ∧
  q ∈ {-8, -6, -4, -1, 1, 3, 5, 7, 9} ∧
  r ∈ {-8, -6, -4, -1, 1, 3, 5, 7, 9} ∧
  s ∈ {-8, -6, -4, -1, 1, 3, 5, 7, 9} ∧
  t ∈ {-8, -6, -4, -1, 1, 3, 5, 7, 9} ∧
  u ∈ {-8, -6, -4, -1, 1, 3, 5, 7, 9} ∧
  v ∈ {-8, -6, -4, -1, 1, 3, 5, 7, 9} ∧
  w ∈ {-8, -6, -4, -1, 1, 3, 5, 7, 9} ∧
  x ∈ {-8, -6, -4, -1, 1, 3, 5, 7, 9} ∧
  (p + q + r + s) + (t + u + v + w + x) = 6 ∧
  (p + q + r + s)^2 + (t + u + v + w + x)^2 = 18 :=
sorry

end minimum_value_of_sum_of_squares_l761_761983


namespace vasya_numbers_l761_761384

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : 
  (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) ∨ (x = y ∧ x = 0) :=
by 
  -- Placeholder to show where the proof would go
  sorry

end vasya_numbers_l761_761384


namespace DO_angle_bisector_AOB_l761_761970

open EuclideanGeometry

variables {A B C O D : Point}

-- Conditions
variables (triangle_ABC : Triangle A B C) 
          (angle_ABC_gt_90 : Angle B A C > 90) 
          (circumcenter_O : IsCircumcenter O A B C) 
          (second_intersection_D : SecondIntersectionPointLineCircle A C O B C D) 

-- Statement as per the conditions provided
theorem DO_angle_bisector_AOB : AngleBisector O D (Segment A O) (Segment B O) :=
by
  sorry

end DO_angle_bisector_AOB_l761_761970


namespace ryan_learning_hours_l761_761486

variables (total_hours english_hours chinese_hours : ℕ)

theorem ryan_learning_hours
  (h_total : total_hours = 3)
  (h_english : english_hours = 2) :
  chinese_hours = total_hours - english_hours := by
  have h_chinese := 1
  rw [h_total, h_english] at h_chinese
  exact h_chinese

end ryan_learning_hours_l761_761486


namespace calculate_miles_walked_l761_761246

theorem calculate_miles_walked :
  let initial_reading := 0
  let flip_count := 55
  let final_reading := 30000
  let steps_per_flip := 90000
  let steps_per_mile := 1800
  let total_steps := flip_count * steps_per_flip + final_reading
  let miles_walked := total_steps / steps_per_mile
  miles_walked.floor = 2767 :=
by
  let initial_reading := 0
  let flip_count := 55
  let final_reading := 30000
  let steps_per_flip := 90000
  let steps_per_mile := 1800
  let total_steps := flip_count * steps_per_flip + final_reading
  let miles_walked := total_steps / steps_per_mile
  have h1 : miles_walked = 2766 + 2/3 := sorry
  have h2 : (2766 + 2/3).floor = 2767 := sorry
  exact h2.symm.trans h1.ge_floor

end calculate_miles_walked_l761_761246


namespace cos_theta_acu_l761_761749

def cos_theta : ℝ :=
  let d1 := (4, 5)
  let d2 := (2, -1)
  let dot_product := d1.1 * d2.1 + d1.2 * d2.2
  let norm_d1 := Real.sqrt (d1.1 ^ 2 + d1.2 ^ 2)
  let norm_d2 := Real.sqrt (d2.1 ^ 2 + d2.2 ^ 2)
  dot_product / (norm_d1 * norm_d2)

theorem cos_theta_acu : cos_theta = 3 / Real.sqrt 205 :=
  sorry

end cos_theta_acu_l761_761749


namespace domain_of_y_range_of_a_l761_761543

noncomputable theory

def f (x : ℝ) := (Real.log x / Real.log 2) ^ 2 - 2 * (Real.log x / Real.log (1 / 2)) + 1 
def g (x : ℝ) (a : ℝ) := x ^ 2 - a * x + 1

theorem domain_of_y {x : ℝ} (k : ℤ) : 
  (cos (x - π / 3) > 0) ↔ (2 * k * π - π / 6 < x ∧ x < 2 * k * π + 5 * π / 6) := sorry

theorem range_of_a (a : ℝ) :
  (∀ (x1 : ℝ), x1 ∈ set.Icc (1 / 8) 2 → 
    ∃! (x0 : ℝ), x0 ∈ set.Icc (-1) 2 ∧ f x1 = g x0 a) ↔ (a ≤ -2 ∨ a > 5 / 2) := sorry

end domain_of_y_range_of_a_l761_761543


namespace number_in_scientific_notation_l761_761959

/-- Condition: A constant corresponding to the number we are converting. -/
def number : ℕ := 9000000000

/-- Condition: The correct answer we want to prove. -/
def correct_answer : ℕ := 9 * 10^9

/-- Proof Problem: Prove that the number equals the correct_answer when expressed in scientific notation. -/
theorem number_in_scientific_notation : number = correct_answer := by
  sorry

end number_in_scientific_notation_l761_761959


namespace vasya_numbers_l761_761339

theorem vasya_numbers :
  ∃ x y : ℝ, x + y = x * y ∧ x + y = x / y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l761_761339


namespace interest_rate_A_l761_761748

def principal : ℝ := 3500
def rate_C : ℝ := 11 / 100 -- Interest rate per annum that B lends to C
def time : ℕ := 3 -- Time in years
def gain_B : ℝ := 105

/-- The interest rate at which A lent the money to B is 10% per annum. -/
theorem interest_rate_A :
  ∃ R : ℝ, (R = 10 / 100) ∧
  let interest_from_C := principal * rate_C * time in
  let interest_paid_to_A := interest_from_C - gain_B in
  interest_paid_to_A = principal * R * time :=
begin
  sorry
end

end interest_rate_A_l761_761748


namespace Arun_lower_limit_l761_761451

-- Define the weight of Arun
def Weight : Type := ℝ
variable (W : Weight)

-- Conditions
def Arun_estimation (L : Weight) := L < W ∧ W < 72
def Brother_estimation := 60 < W ∧ W < 70
def Mother_estimation := W ≤ 68
def Average_weight := W = 67

-- Problem statement
theorem Arun_lower_limit (L : Weight) (h1 : Arun_estimation W L) (h2 : Brother_estimation W) 
    (h3 : Mother_estimation W) (h4 : Average_weight W) : 60 < W := sorry

end Arun_lower_limit_l761_761451


namespace find_a_l761_761971

def A : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def B (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }

theorem find_a (a : ℝ) : (A ∩ B a = B a) → (a = 0 ∨ a = 1 / 2 ∨ a = 1 / 3) := by
  sorry

end find_a_l761_761971


namespace vasya_numbers_l761_761342

theorem vasya_numbers :
  ∃ x y : ℝ, x + y = x * y ∧ x + y = x / y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l761_761342


namespace functional_equation_solution_l761_761989

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y z : ℝ, x + y + z = 0 → f(x^3) + f(y)^3 + f(z)^3 = 3 * x * y * z) →
  (∀ x : ℝ, f(x) = x) :=
by
  intros hf x
  sorry

end functional_equation_solution_l761_761989


namespace infinite_series_sum_l761_761211

theorem infinite_series_sum
  (a b : ℝ)
  (h1 : (∑' n : ℕ, a / (b ^ (n + 1))) = 4) :
  (∑' n : ℕ, a / ((a + b) ^ (n + 1))) = 4 / 5 := 
sorry

end infinite_series_sum_l761_761211


namespace point_inside_circle_l761_761532

theorem point_inside_circle (O A : Type) (r OA : ℝ) (h1 : r = 6) (h2 : OA = 5) :
  OA < r :=
by
  sorry

end point_inside_circle_l761_761532


namespace monotonicity_two_zeros_implies_a_range_l761_761112

noncomputable def f (a x : ℝ) : ℝ := (1/2) * a * x^2 - Real.log x - 2

theorem monotonicity (a : ℝ) :
  (∀ x > 0, deriv (λ x, f a x) x < 0 ↔ a ≤ 0) ∧
  (∀ x > 0, (deriv (λ x, f a x) x = 0 ↔ x = (Real.sqrt a)/a) → 
    (∀ x ∈ Ioo 0 (Real.sqrt a / a), deriv (λ x, f a x) x < 0) ∧ 
    (∀ x ∈ Ioi (Real.sqrt a / a), deriv (λ x, f a x) x > 0)) :=
sorry

theorem two_zeros_implies_a_range (a : ℝ) :
  (∀ x > 0, ((f a x = 0) → (Real.exists_two_points Ioi (λ x, f a x = 0))) ↔ a ∈ Ioo 0 (Real.exp 3)) :=
sorry

end monotonicity_two_zeros_implies_a_range_l761_761112


namespace combination_equality_l761_761485

open nat

noncomputable def combination (n k : ℕ) : ℕ := nat.choose n k

theorem combination_equality : ∀ (n : ℕ),
  (0 ≤ 5 - n ∧ 5 - n ≤ n) →
  (0 ≤ 10 - n ∧ 10 - n ≤ n + 1) →
  combination n (5 - n) + combination (n + 1) (10 - n) = 7 :=
begin
  intros n h1 h2,
  -- proof goes here
  sorry
end

end combination_equality_l761_761485


namespace sin_theta_value_l761_761885

theorem sin_theta_value (a : ℝ) (h : a ≠ 0) (h_tan : Real.tan θ = -a) (h_point : P = (a, -1)) : Real.sin θ = -Real.sqrt 2 / 2 :=
sorry

end sin_theta_value_l761_761885


namespace isosceles_right_triangle_l761_761872

theorem isosceles_right_triangle
  (a b c : ℝ)
  (h : sqrt (c^2 - a^2 - b^2) + abs (a - b) = 0) :
  c^2 = a^2 + b^2 ∧ a = b :=
by
  sorry

end isosceles_right_triangle_l761_761872


namespace minimum_3a_2b_l761_761870

-- Definitions based on conditions
variables (a b : ℝ)
hypothesis a_pos : 0 < a
hypothesis b_pos : 0 < b

-- Condition from the problem statement
hypothesis condition : 1 / (a + b) + 1 / (a - b) = 1

-- Statement to prove
theorem minimum_3a_2b : 3 * a + 2 * b = 3 + Real.sqrt 5 := 
  sorry

end minimum_3a_2b_l761_761870


namespace equal_zero_b_l761_761969

theorem equal_zero_b (n : ℕ) (a : Fin n → ℝ) (b : Fin n → ℝ) (h1 : ∀ i, 0 ≤ a i ∧ a i ≤ π) 
  (h2 : ∀ k : ℕ, |∑ i in Finset.finRange n, b i * Real.cos (k * a i)| < 1 / k) : 
  ∀ i, b i = 0 := 
by 
  sorry

end equal_zero_b_l761_761969


namespace proof_of_unit_prices_proof_of_minimum_cost_l761_761700

variable (unit_price_TC unit_price_RE quantity_TC quantity_RE total_cost_TC total_cost_RE : ℝ)

-- Define conditions
def conditions : Prop :=
  total_cost_RE = 14000 ∧
  total_cost_TC = 7000 ∧
  unit_price_RE = 1.4 * unit_price_TC ∧
  quantity_RE = quantity_TC + 300 ∧
  unit_price_TC > 0

-- Define variables for minimizing total cost
variable (a : ℝ)  -- quantity of "Traditional Culture" classic textbooks to be ordered
variable (total_quantity : ℝ := 1000)
variable (max_quantity_TC : ℝ := 400)
variable (max_total_cost : ℝ := 12880)

-- Inequalities for minimization task
def min_cost_conditions : Prop :=
  a ≤ 400 ∧
  10 * a + 14 * (total_quantity - a) ≤ max_total_cost

-- Minimum cost
def minimum_cost : ℝ := 10 * max_quantity_TC + 14 * (total_quantity - max_quantity_TC)

-- The proof statements
theorem proof_of_unit_prices (h : conditions): 
  unit_price_TC = 10 ∧ unit_price_RE = 14 :=
sorry

theorem proof_of_minimum_cost (h : min_cost_conditions):
  minimum_cost = 12400 :=
sorry

end proof_of_unit_prices_proof_of_minimum_cost_l761_761700


namespace inequality_proof_l761_761729

noncomputable section

variables (a b θ : ℝ)

theorem inequality_proof :
    |a| + |b| ≤ sqrt (a^2 * cos θ ^ 2 + b^2 * sin θ ^ 2) + sqrt (a^2 * sin θ ^ 2 + b^2 * cos θ ^ 2) ∧ 
    sqrt (a^2 * cos θ ^ 2 + b^2 * sin θ ^ 2) + sqrt (a^2 * sin θ ^ 2 + b^2 * cos θ ^ 2) ≤ sqrt (2 * (a^2 + b^2)) := 
sorry

end inequality_proof_l761_761729


namespace log_condition_l761_761915

noncomputable def is_non_square_non_cube_non_integral_rational (x : ℝ) : Prop :=
  ¬∃ n : ℤ, x = n^2 ∨ x = n^3 ∨ (x.denom = 1)

theorem log_condition (x : ℝ) (h : log (3 * x) 343 = x) : is_non_square_non_cube_non_integral_rational x := 
sorry

end log_condition_l761_761915


namespace pascal_second_number_57_elements_row_l761_761390

theorem pascal_second_number_57_elements_row :
  ∃ n : ℕ, (n + 1 = 57 ∧ (nat.choose n 1 = 56)) :=
begin
  use 56,
  split,
  { norm_num, },
  { apply nat.choose_one_right, }
end

end pascal_second_number_57_elements_row_l761_761390


namespace Vasya_numbers_l761_761345

theorem Vasya_numbers : 
  ∃ x y : ℚ, (x + y = x * y) ∧ (x * y = x / y) ∧ x = 1 / 2 ∧ y = -1 :=
by 
  use (1 / 2)
  use (-1)
  split
  case left {
    calc 
      (1 / 2) + (-1) = -1 / 2 : by norm_num,
      -1 / 2 = ((1 / 2) * (-1)) : by norm_num,
      (1 / 2) + (-1) = (1 / 2) * (-1) : by norm_num,
  },
  case right {
    calc 
      (1 / 2) * (-1) = -1 / 2 : by norm_num,
      -1 / 2 = (1 / 2) / (-1) : by norm_num,
      (1 / 2) * (-1) = (1 / 2) / (-1) : by norm_num,
  }

end Vasya_numbers_l761_761345


namespace diameter_of_inscribed_circle_is_8_l761_761040

noncomputable def diameter_of_inscribed_circle
  (DE DF EF : ℝ) (h1 : DE = 13) (h2 : DF = 14) (h3 : EF = 15) : ℝ :=
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let r := K / s
  let d := 2 * r
  d

theorem diameter_of_inscribed_circle_is_8 : diameter_of_inscribed_circle 13 14 15 13 14 15 = 8 :=
  by
  sorry

end diameter_of_inscribed_circle_is_8_l761_761040


namespace initial_crayons_count_l761_761695

theorem initial_crayons_count : 
  ∀ (initial_crayons added_crayons total_crayons : ℕ), 
  added_crayons = 3 ∧ total_crayons = 12 → initial_crayons + added_crayons = total_crayons → initial_crayons = 9 := 
by
  intros initial_crayons added_crayons total_crayons
  intros h1 h2
  cases h1 with h_added h_total
  rw [h_added, h_total] at h2
  linarith

end initial_crayons_count_l761_761695


namespace probability_D_within_E_l761_761886

-- Define the regions D and E
def region_D (x y : ℝ) : Prop := (y = x^2) ∨ (y = 1)
def region_E (x y : ℝ) : Prop := (-1 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1)

-- Define the combined area for region E
def area_E : ℝ := 2

-- Define the integral to calculate the area of D
def integral_D : ℝ := ∫ x in -1..1, (1 - x^2) -- implictly assuming a definite integral over the interval [-1, 1]

-- State the final theorem
theorem probability_D_within_E : 
  (integral_D / area_E) = (2 / 3) := 
by 
  sorry

end probability_D_within_E_l761_761886


namespace total_travel_time_l761_761791

noncomputable def washingtonToIdahoDistance : ℕ := 640
noncomputable def idahoToNevadaDistance : ℕ := 550
noncomputable def washingtonToIdahoSpeed : ℕ := 80
noncomputable def idahoToNevadaSpeed : ℕ := 50

theorem total_travel_time :
  (washingtonToIdahoDistance / washingtonToIdahoSpeed) + (idahoToNevadaDistance / idahoToNevadaSpeed) = 19 :=
by
  sorry

end total_travel_time_l761_761791


namespace points_opposite_sides_l761_761718

theorem points_opposite_sides (m : ℝ) : (-2 < m ∧ m < -1) ↔ ((2 - 3 * 1 - m) * (1 - 3 * 1 - m) < 0) := by
  sorry

end points_opposite_sides_l761_761718


namespace lisa_eats_one_candy_on_other_days_l761_761631

def candies_total : ℕ := 36
def candies_per_day_on_mondays_and_wednesdays : ℕ := 2
def weeks : ℕ := 4
def days_in_a_week : ℕ := 7
def mondays_and_wednesdays_in_4_weeks : ℕ := 2 * weeks
def total_candies_mondays_and_wednesdays : ℕ := mondays_and_wednesdays_in_4_weeks * candies_per_day_on_mondays_and_wednesdays
def total_other_candies : ℕ := candies_total - total_candies_mondays_and_wednesdays
def total_other_days : ℕ := weeks * (days_in_a_week - 2)
def candies_per_other_day : ℕ := total_other_candies / total_other_days

theorem lisa_eats_one_candy_on_other_days :
  candies_per_other_day = 1 :=
by
  -- Prove the theorem with conditions defined
  sorry

end lisa_eats_one_candy_on_other_days_l761_761631


namespace letters_into_mailboxes_l761_761132

theorem letters_into_mailboxes (n m : ℕ) (h1 : n = 3) (h2 : m = 5) : m^n = 125 :=
by
  rw [h1, h2]
  exact rfl

end letters_into_mailboxes_l761_761132


namespace smallest_positive_multiple_of_17_6_more_than_multiple_of_73_l761_761714

theorem smallest_positive_multiple_of_17_6_more_than_multiple_of_73 :
  ∃ b : ℤ, (17 * b ≡ 6 [MOD 73]) ∧ 17 * b = 663 :=
begin
  sorry
end

end smallest_positive_multiple_of_17_6_more_than_multiple_of_73_l761_761714


namespace masking_tape_needed_l761_761051

theorem masking_tape_needed {a b : ℕ} (h1 : a = 4) (h2 : b = 6) :
  2 * a + 2 * b = 20 :=
by
  rw [h1, h2]
  norm_num

end masking_tape_needed_l761_761051


namespace value_2sin_cos_l761_761106

variables {m : ℝ} (hm : m ≠ 0)

def point_P := (-4 * m, 3 * m)

noncomputable def r := real.sqrt ((-4 * m)^2 + (3 * m)^2)

def α_sin := 3 * m / r
def α_cos := -4 * m / r

theorem value_2sin_cos (hm : m ≠ 0) : 
  (2 * α_sin hm + α_cos hm = 2 / 5 ∨ 2 * α_sin hm + α_cos hm = -2 / 5) := sorry

end value_2sin_cos_l761_761106


namespace geometric_sequence_angle_count_l761_761043

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ
noncomputable def csc (θ : ℝ) : ℝ := 1 / Real.sin θ
noncomputable def cot (θ : ℝ) : ℝ := Real.cos θ / Real.sin θ

def valid_angle (θ : ℝ) : Prop := θ ∈ Set.Icc 0 (2 * Real.pi) ∧ 
  Real.cos θ ≠ 0 ∧ Real.sin θ ≠ 0

theorem geometric_sequence_angle_count :
  (Finset.card (Finset.filter valid_angle (Finset.Icc 0 (2 * Real.pi))))
  = 2 := sorry

end geometric_sequence_angle_count_l761_761043


namespace therapists_next_meeting_day_l761_761767

theorem therapists_next_meeting_day : Nat.lcm (Nat.lcm 5 2) (Nat.lcm 9 3) = 90 := by
  -- Given that Alex works every 5 days,
  -- Brice works every 2 days,
  -- Emma works every 9 days,
  -- and Fiona works every 3 days, we need to show that the LCM of these numbers is 90.
  sorry

end therapists_next_meeting_day_l761_761767


namespace probability_of_common_books_l761_761637

theorem probability_of_common_books (total_books : ℕ) (books_to_select : ℕ) :
  total_books = 12 → books_to_select = 6 →
  let total_ways := Nat.choose 12 6 * Nat.choose 12 6 in
  let successful_ways := Nat.choose 12 3 * Nat.choose 9 3 * Nat.choose 9 3 in
  (successful_ways : ℚ) / total_ways = 220 / 153 :=
by
  intros ht12 hs6
  let total_ways := Nat.choose 12 6 * Nat.choose 12 6
  let successful_ways := Nat.choose 12 3 * Nat.choose 9 3 * Nat.choose 9 3
  have htotal_ways : total_ways = 924 * 924 := by sorry
  have hsuccessful_ways : successful_ways = 220 * 84 * 84 := by sorry
  rw [ht12, hs6, htotal_ways, hsuccessful_ways]
  norm_num
  exact @eq.refl(ℚ) (220 / 153)

end probability_of_common_books_l761_761637


namespace distance_between_Z1_Z2_l761_761875

-- Define the complex numbers z1 and z2
def z1 := Complex.mk 1 (-1)
def z2 := Complex.mk 3 (-5)

-- Define the points Z1 and Z2 corresponding to z1 and z2 in the complex plane
def Z1 := (1, -1) : ℝ × ℝ
def Z2 := (3, -5) : ℝ × ℝ

-- Define the distance formula between two points in Cartesian coordinates
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Statement of the theorem to prove
theorem distance_between_Z1_Z2 : distance Z1 Z2 = 2 * Real.sqrt 5 := by
  -- Proof goes here
  sorry

end distance_between_Z1_Z2_l761_761875


namespace simplest_quadratic_radical_value_l761_761916

theorem simplest_quadratic_radical_value (x : ℝ) (h1 : sqrt (x + 3) = sqrt 5) : x = 2 := 
by sorry

end simplest_quadratic_radical_value_l761_761916


namespace S_n_sum_not_geometric_seq_l761_761853

-- Definitions
noncomputable def a_seq (a : ℕ → ℝ) := ∀ n : ℕ, n > 0 → a (n + 1) = 2 * a n - n + 1
noncomputable def is_arithmetic_seq (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, n > 0 → a (n + 1) = a n + d

-- Problems
theorem S_n_sum (a : ℕ → ℝ) (h_arith : is_arithmetic_seq a) (h_seq : a_seq a) :
  ∀ n : ℕ, n > 0 → ∑ k in range n, 1 / (a k * a (k + 1)) = n / (n + 1) :=
sorry

theorem not_geometric_seq (a : ℕ → ℝ) (h_seq : a_seq a) :
  ¬ ∀ n : ℕ, n > 0 → (a n + 2) * (a (n + 2) + 2) = (a (n + 1) + 2) * (a (n + 1) + 2) :=
sorry

end S_n_sum_not_geometric_seq_l761_761853


namespace count_multiples_of_25_but_not_75_3_digit_l761_761134

theorem count_multiples_of_25_but_not_75_3_digit :
  let is_3_digit (n : ℕ) := (100 ≤ n) ∧ (n ≤ 999)
  let is_multiple_of_25 (n : ℕ) := ∃ k : ℕ, n = 25 * k
  let is_multiple_of_75 (n : ℕ) := ∃ m : ℕ, n = 75 * m
  (finset.filter (λ n : ℕ, is_3_digit n ∧ is_multiple_of_25 n ∧ ¬ is_multiple_of_75 n) (finset.range 1000)).card = 24 := by
  sorry

end count_multiples_of_25_but_not_75_3_digit_l761_761134


namespace max_right_triangles_l761_761848

open_locale classical

variables (L : set (set (ℝ × ℝ))) (n : ℕ)
noncomputable theory 

-- Conditions
def is_line (l : set (ℝ × ℝ)) := ∃ a b c : ℝ, (a, b) ≠ (0, 0) ∧ ∀ p : ℝ × ℝ, p ∈ l ↔ a * p.1 + b * p.2 + c = 0
def is_right_triangle (l1 l2 l3 : set (ℝ × ℝ)) : Prop :=
  l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 ∧ is_line l1 ∧ is_line l2 ∧ is_line l3 ∧
  ∃ p1 p2 p3 : ℝ × ℝ, 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    p1 ∈ l1 ∧ p1 ∈ l2 ∧
    p2 ∈ l2 ∧ p2 ∈ l3 ∧
    p3 ∈ l3 ∧ p3 ∈ l1 ∧
    (let v1 := (p2.1 - p1.1, p2.2 - p1.2),
         v2 := (p3.1 - p2.1, p3.2 - p2.2) in
    v1.1 * v2.1 + v1.2 * v2.2 = 0)

-- Given conditions
axiom h_lines : set.finite L
axiom lines_card : L.to_finset.card = 100

-- Set of right-angled triangles
def T : set (set (set (ℝ × ℝ))) := {t : set (set (ℝ × ℝ)) | ∃ l1 l2 l3 ∈ L, t = {l1, l2, l3} ∧ is_right_triangle l1 l2 l3}

-- The Lean statement for the proof problem
theorem max_right_triangles : |T| ≤ 62500 :=
sorry

end max_right_triangles_l761_761848


namespace find_a_b_l761_761494

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * |cos x| + b * |sin x|

theorem find_a_b (a b : ℝ) :
  (∀ x : ℝ, f a b x = a * |cos x| + b * |sin x| ∧ (∃ x = -π / 3, is_local_min (f a b) x))
  ∧ (∫ x in -π/2..π/2, (f a b x)^2 = 2) 
  → a = -1/√π ∧ b = √3/√π :=
sorry

end find_a_b_l761_761494


namespace roots_count_l761_761037

noncomputable def g : ℤ → ℝ := sorry

axiom g_symmetry1 (x : ℤ) : g (3 + x) = g (3 - x)
axiom g_symmetry2 (x : ℤ) : g (5 + x) = g (5 - x)
axiom g_initial : g 1 = 0

theorem roots_count : (set_of (λ x : ℤ, g x = 0) ∩ set.Icc (-1000 : ℤ) 1000).finite ∧ 
                      (set_of (λ x : ℤ, g x = 0) ∩ set.Icc (-1000 : ℤ) 1000).to_finset.card ≥ 250 := 
by sorry

end roots_count_l761_761037


namespace expected_xi_eq_l761_761820

-- Definitions based on conditions
variable (Team : Type) [Fintype Team]
variable (CanSing : Team → Prop)
variable (CanDance : Team → Prop)
variable (CanSingAndDance : Team → Prop := λ t => CanSing t ∧ CanDance t)
variable [DecidablePred CanSing]
variable [DecidablePred CanDance]
variable [DecidablePred CanSingAndDance]

axiom TwoCanSing : Fintype.card { t // CanSing t } = 2
axiom FiveCanDance : Fintype.card { t // CanDance t } = 5

-- Given condition probability defined
axiom ProbXiGreaterThanZero : (2 : ℝ) * (∑ x in Finset.univ.filter CanSingAndDance, (1 / Fintype.card Team : ℝ)) * (∑ y in Finset.univ.filter CanSingAndDance, (1 / Fintype.card Team : ℝ - ite (x = y) (1 / (Fintype.card Team - 1) : ℝ) 0)) = 7 / 10

-- To prove
theorem expected_xi_eq : (E : ℝ) (Xi : Type) [Fintype Xi] [HasZero Xi] [HasOne Xi] := E (λ (selected : Finset Team ↪ Xi), Fintype.card (selected.image CanSingAndDance)) = 0.8 :=
sorry

end expected_xi_eq_l761_761820


namespace rectangular_diagonal_length_l761_761316

theorem rectangular_diagonal_length (x y z : ℝ) 
  (h_surface_area : 2 * (x * y + y * z + z * x) = 11)
  (h_edge_sum : x + y + z = 6) :
  Real.sqrt (x^2 + y^2 + z^2) = 5 := 
by
  sorry

end rectangular_diagonal_length_l761_761316


namespace sum_series_induction_l761_761335

theorem sum_series_induction (k : ℕ) :
  (∑ i in Finset.range ((3 * (k + 1) + 2)), i + 1) = (3 * k + 2) + (3 * k + 3) + (3 * k + 4) + (∑ i in Finset.range ((3 * k) + 2), i + 1) :=
by
  sorry

end sum_series_induction_l761_761335


namespace roots_of_quadratic_l761_761245

theorem roots_of_quadratic :
  ∃ (b c : ℝ), ( ∀ (x : ℝ), x^2 + b * x + c = 0 ↔ x = 1 ∨ x = -2) :=
sorry

end roots_of_quadratic_l761_761245


namespace find_a_b_monotonicity_l761_761892

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (x^2 + a * x + b) / x

theorem find_a_b (a b : ℝ) (h_odd : ∀ x ≠ 0, f (-x) a b = -f x a b) (h_eq : f 1 a b = f 4 a b) :
  a = 0 ∧ b = 4 := by sorry

theorem monotonicity (f : ℝ → ℝ) (h : ∀ x ≠ 0, f x = x + 4 / x) :
  (∀ x1 x2, 0 < x1 ∧ x1 ≤ 2 ∧ x1 < x2 ∧ x2 ≤ 2 → f x1 > f x2) ∧
  (∀ x1 x2, 2 < x1 ∧ x1 < x2 → f x1 < f x2) := by sorry

end find_a_b_monotonicity_l761_761892


namespace sufficient_not_necessary_condition_l761_761146

variable (a : ℝ)

theorem sufficient_not_necessary_condition :
  (1 < a ∧ a < 2) → (a^2 - 3 * a ≤ 0) := by
  intro h
  sorry

end sufficient_not_necessary_condition_l761_761146


namespace num_valid_lists_l761_761622

-- Define a predicate for a list to satisfy the given constraints
def valid_list (l : List ℕ) : Prop :=
  l = List.range' 1 12 ∧ ∀ i, 1 < i ∧ i ≤ 12 → (l.indexOf (l.get! (i - 1) + 1) < i - 1 ∨ l.indexOf (l.get! (i - 1) - 1) < i - 1) ∧ ¬(l.indexOf (l.get! (i - 1) + 1) < i - 1 ∧ l.indexOf (l.get! (i - 1) - 1) < i - 1)

-- Prove that there is exactly one valid list of such nature
theorem num_valid_lists : ∃! l : List ℕ, valid_list l :=
  sorry

end num_valid_lists_l761_761622


namespace projection_of_unit_vectors_l761_761904

open Real

variables {a b : Vector ℝ} -- Declare the vectors a and b

-- Asserting the conditions of the problem
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (condition : ‖a + b‖ = 1)

-- Define the projection function
def projection (u v : Vector ℝ) : Vector ℝ := (u • v) / ‖v‖^2 • v

theorem projection_of_unit_vectors :
  projection a b = -0.5 • b := by
  -- Given conditions
  have hab : a • b = -0.5, from sorry,
  -- Conclude the projection result
  sorry

end projection_of_unit_vectors_l761_761904


namespace complement_union_l761_761903

def M := { x : ℝ | (x + 3) * (x - 1) < 0 }
def N := { x : ℝ | x ≤ -3 }
def union_set := M ∪ N

theorem complement_union :
  ∀ x : ℝ, x ∈ (⊤ \ union_set) ↔ x ≥ 1 :=
by
  sorry

end complement_union_l761_761903


namespace monotonic_increasing_interval_l761_761300

noncomputable def f (x : ℝ) := log (x^2 - 3 * x)

theorem monotonic_increasing_interval :
  {x : ℝ | 3 < x} = {x : ℝ | f' x > 0 ∧ x^2 - 3 * x > 0} :=
sorry

end monotonic_increasing_interval_l761_761300


namespace fill_tank_time_l761_761644

theorem fill_tank_time :
  ∀ (capacity rateA rateB rateC timeA timeB timeC : ℕ),
  capacity = 1000 →
  rateA = 200 →
  rateB = 50 →
  rateC = 25 →
  timeA = 1 →
  timeB = 2 →
  timeC = 2 →
  let net_fill := rateA * timeA + rateB * timeB - rateC * timeC in
  let total_cycles := capacity / net_fill in
  let cycle_time := timeA + timeB + timeC in
  let total_time := total_cycles * cycle_time in
  total_time = 20 := sorry

end fill_tank_time_l761_761644


namespace surface_area_of_prism_l761_761166

noncomputable def volume_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

noncomputable def volume_prism (l w h : ℝ) : ℝ :=
  l * w * h

noncomputable def surface_area_prism (l w h : ℝ) : ℝ :=
  2 * (l * w + w * h + l * h)

theorem surface_area_of_prism :
  let r := Real.cbrt (36 / Real.pi)
  let l := 6
  let w := 4
  let h := (volume_sphere r) / (l * w)
  surface_area_prism l w h = 88 :=
by
  sorry

end surface_area_of_prism_l761_761166


namespace smallest_possible_value_of_m_l761_761225

theorem smallest_possible_value_of_m 
  (m : ℕ) (y : Fin m → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ y i)
  (h_sum : ∑ i, y i = 1)
  (h_sum_sq : ∑ i, (y i)^2 ≤ 1 / 50) : m ≥ 50 := sorry

end smallest_possible_value_of_m_l761_761225


namespace variance_scaling_l761_761516

variable {ι : Type*} [Fintype ι]
variable (x : ι → ℝ) (a b : ℝ)

theorem variance_scaling (h_var_x : ∑ i, (x i - (∑ j, x j) / Fintype.card ι)^2 = 3)
  (h_var_axb : ∑ i, (a * x i + b - (∑ j, (a * x j + b)) / Fintype.card ι)^2 = 12) : 
  a = 2 ∨ a = -2 :=
by
  sorry

end variance_scaling_l761_761516


namespace number_of_small_spheres_l761_761747

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem number_of_small_spheres
  (d_large : ℝ) (d_small : ℝ)
  (h1 : d_large = 6) (h2 : d_small = 2) :
  let V_large := volume_of_sphere (d_large / 2)
  let V_small := volume_of_sphere (d_small / 2)
  V_large / V_small = 27 := 
by
  sorry

end number_of_small_spheres_l761_761747


namespace ab_div_c_eq_2_l761_761205

variable (a b c : ℝ)

def condition1 (a b c : ℝ) : Prop := a * b - c = 3
def condition2 (a b c : ℝ) : Prop := a * b * c = 18

theorem ab_div_c_eq_2 (h1 : condition1 a b c) (h2 : condition2 a b c) : a * b / c = 2 :=
by sorry

end ab_div_c_eq_2_l761_761205


namespace calculate_monthly_rent_l761_761765

theorem calculate_monthly_rent (P : ℝ) (R : ℝ) (T : ℝ) (M : ℝ) (rent : ℝ) :
  P = 12000 →
  R = 0.06 →
  T = 400 →
  M = 0.1 →
  rent = 103.70 :=
by
  intros hP hR hT hM
  sorry

end calculate_monthly_rent_l761_761765


namespace water_height_l761_761682

noncomputable def cone_radius : ℝ := 20
noncomputable def cone_height : ℝ := 60
noncomputable def water_volume_fraction : ℝ := 0.5

theorem water_height (a b : ℕ) (h1 : (20 : ℝ) = cone_radius)
                     (h2 : (60 : ℝ) = cone_height)
                     (h3 : (0.5 : ℝ) = water_volume_fraction)
                     (h4 : water_volume_fraction * (cone_radius ^ 2 * cone_height / 3) = cone_radius ^ 2 * cone_height / 2 / 3):
  a ∈ ℕ ∧ b ∈ ℕ ∧ b ≠ 1 ∧ (60 * (1 / 2)^(1 / 3) = a * (b)^(1 / 3)) → a + b = 32 :=
sorry

end water_height_l761_761682


namespace overlapping_segments_length_l761_761310

theorem overlapping_segments_length:
  (red_segments_total : ℕ) (actual_distance : ℕ) (num_overlaps : ℕ) (x : ℝ)
  (h1 : red_segments_total = 98)
  (h2 : actual_distance = 83)
  (h3 : num_overlaps = 6)
  (h4 : red_segments_total - actual_distance = num_overlaps * x) :
  x = 2.5 :=
by
  sorry

end overlapping_segments_length_l761_761310


namespace chloe_profit_l761_761460

theorem chloe_profit 
  (cost_per_dozen : ℕ)
  (selling_price_per_half_dozen : ℕ)
  (dozens_sold : ℕ)
  (h1 : cost_per_dozen = 50)
  (h2 : selling_price_per_half_dozen = 30)
  (h3 : dozens_sold = 50) : 
  (selling_price_per_half_dozen - cost_per_dozen / 2) * (dozens_sold * 2) = 500 :=
by 
  sorry

end chloe_profit_l761_761460


namespace part1_strictly_increasing_part2_parity_even_part2_parity_neither_l761_761896

def f (x : ℝ) (a : ℝ) : ℝ := 2 * x^2 + a / x

theorem part1_strictly_increasing (a : ℝ) (x₁ x₂ : ℝ) (h₁ : 1 ≤ x₁) (h₂ : x₁ ≤ x₂) (ha : a = 1) : 
  f x₁ a < f x₂ a :=
by
  sorry

theorem part2_parity_even (a : ℝ) (x : ℝ) (ha : a = 0) : f (-x) a = f x a :=
by
  sorry

theorem part2_parity_neither (a : ℝ) (x : ℝ) (ha : a ≠ 0) : ¬(f (-x) a = f x a) ∧ ¬(f (-x) a = -f x a) :=
by
  sorry

end part1_strictly_increasing_part2_parity_even_part2_parity_neither_l761_761896


namespace binary_to_decimal_101_l761_761026

theorem binary_to_decimal_101 : ∑ (i : Fin 3), (Nat.digit 2 ⟨i, sorry⟩ (list.nth_le [1, 0, 1] i sorry)) * (2 ^ i) = 5 := 
  sorry

end binary_to_decimal_101_l761_761026


namespace combined_degrees_l761_761262

-- Definitions based on conditions
def summer_degrees : ℕ := 150
def jolly_degrees (summer_degrees : ℕ) : ℕ := summer_degrees - 5

-- Theorem stating the combined degrees
theorem combined_degrees : summer_degrees + jolly_degrees summer_degrees = 295 :=
by
  sorry

end combined_degrees_l761_761262


namespace probability_interval_one_to_two_l761_761305

def p (x : ℝ) : ℝ := 
  if x ≤ 0 then 0 
  else if x > 2 then 0 
  else x / 2

theorem probability_interval_one_to_two :
  ∫ x in 1..2, p x = 3 / 4 :=
by
  sorry

end probability_interval_one_to_two_l761_761305


namespace five_a1_plus_a7_l761_761095

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a n = a 0 + n * d

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(n + 1) * a 0 + (n * (n + 1) / 2) * ((a 1) - a 0)

theorem five_a1_plus_a7 (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) (S3_eq_6 : S 3 = 6)
  (seq_arithmetic : arithmetic_sequence a d) (Sn_def : ∀ n : ℕ, S n = sum_first_n_terms a n) :
  (5 * a 0 + a 7) = 12 :=
by
  sorry

end five_a1_plus_a7_l761_761095


namespace triangles_similar_l761_761994

noncomputable def triangle {α : Type*} [Field α] (A B C : α) : α × α × α := (A, B, C)

variables {α : Type*} [Field α]

def line (P Q : α) := ∃ a b c : α, ¬ (a = 0 ∧ b = 0) ∧ (a * P + b * Q + c = 0)

noncomputable def intersection (AP AB : line α) : α := sorry

def angle (P Q R S : α) : α := sorry  -- placeholder for angle definition

def circumcircle (ABC : α × α × α) : α := sorry  -- placeholder for circumcircle definition

variables (A B C P A_1 B_1 C_1 A_2 B_2 C_2 : α)
variables (hAP : line A P)
variables (hBP : line B P)
variables (hCP : line C P)

variables (hIntA : intersection hAP (circumcircle (triangle A B C)) = A_1)
variables (hIntB : intersection hBP (circumcircle (triangle A B C)) = B_1)
variables (hIntC : intersection hCP (circumcircle (triangle A B C)) = C_1)

variables (hA2 : line B C A_2)
variables (hB2 : line C A B_2)
variables (hC2 : line A B C_2)

variables (hAngleA2 : angle P A_2 B C = angle P B_2 C A)
variables (hAngleB2 : angle P B_2 C A = angle P C_2 A B)

theorem triangles_similar : triangle A_2 B_2 C_2 ∼ triangle A_1 B_1 C_1 := 
sorry

end triangles_similar_l761_761994


namespace three_digit_multiples_of_25_not_75_count_l761_761137

-- Definitions from conditions.
def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000
def is_multiple_of_25 (n : ℕ) : Prop := n % 25 = 0
def is_multiple_of_75 (n : ℕ) : Prop := n % 75 = 0

-- The theorem statement.
theorem three_digit_multiples_of_25_not_75_count : 
  let count := (finset.filter (λ n, is_three_digit n ∧ is_multiple_of_25 n ∧ ¬ is_multiple_of_75 n) (finset.range 1000)).card
  in count = 24 :=
by
  sorry

end three_digit_multiples_of_25_not_75_count_l761_761137


namespace lamps_on_after_n2_minus_1_lamps_on_after_n2_minus_n_plus_1_l761_761968

def lamps_on_again (n : ℕ) (steps : ℕ → Bool → Bool) : ∃ M : ℕ, ∀ s, (s ≥ M) → (n > 1 → ∀ i : ℕ, steps i true = true) := 
sorry

theorem lamps_on_after_n2_minus_1 (n : ℕ) (k : ℕ) (hk : n = 2^k) (steps : ℕ → Bool → Bool) : 
∀ s, (s ≥ n^2 - 1) → (n > 1 → ∀ i : ℕ, steps i true = true) := 
sorry

theorem lamps_on_after_n2_minus_n_plus_1 (n : ℕ) (k : ℕ) (hk : n = 2^k + 1) (steps : ℕ → Bool → Bool) : 
∀ s, (s ≥ n^2 - n + 1) → (n > 1 → ∀ i : ℕ, steps i true = true) := 
sorry

end lamps_on_after_n2_minus_1_lamps_on_after_n2_minus_n_plus_1_l761_761968


namespace range_of_a_l761_761541

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h_def : ∀ x, f x = x^2 - 2 * a * x + a^2 - 1) 
(h_sol : ∀ x, f (f x) ≥ 0) : a ≤ -2 :=
sorry

end range_of_a_l761_761541


namespace min_value_expression_l761_761974

theorem min_value_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  ∃ x, (x = a^2 + b^2 + c^2 + 1/a^2 + b/a + c/b) ∧ x = sqrt 6 :=
by 
  sorry

end min_value_expression_l761_761974


namespace neq_necessary_but_not_sufficient_l761_761732

theorem neq_necessary_but_not_sufficient (x y : ℝ) : ¬(x ≠ y ↔ ¬|x| = |y|) :=
by
  -- Key definitions
  have h := abs_eq_abs,
  -- Statement of necessary but not sufficient condition
  sorry

end neq_necessary_but_not_sufficient_l761_761732


namespace equal_area_of_quadrilaterals_l761_761734

variables {A B C D P Q X Y Z T O : Type}
variables [AddCommGroup A] [Module ℝ A]
variables (A B C D P Q X Y Z T O : A)
variables [AddCommGroup B C D A]
variables (h1 : ∀ (A B C D : A), midpoint ℝ A B = P ∧ midpoint ℝ C D = Q)
variables (h2 : ∀ (P Q : A), line_through P (parallel_to (diagonal A C)) ∧ line_through Q (parallel_to (diagonal B D)) → intersect_at O)
variables (h3 : ∀ (X Y Z T : A), is_midpoint_of A B X ∧ is_midpoint_of B C Y ∧ is_midpoint_of C D Z ∧ is_midpoint_of D A T)

theorem equal_area_of_quadrilaterals :
  area (quadrilateral O X B Y) = area (quadrilateral O Y C Z) ∧
  area (quadrilateral O Z D T) = area (quadrilateral O T A X) :=
by
  sorry

end equal_area_of_quadrilaterals_l761_761734


namespace range_of_y_0_l761_761513

theorem range_of_y_0 (x_0 y_0 : ℝ) (h_point_on_parabola : x_0^2 = 8 * y_0) (h_focus : (0, 2)) (h_directrix: ∀ y, y = -2) (h_circle_radius: |(x_0, y_0) - (0, 2)| = y_0 + 2) (h_intersects_directrix: ∃ y, y = -2 ∧ x^2 + (y - 2)^2 = (y_0 + 2)^2) :
  2 < y_0 :=
sorry

end range_of_y_0_l761_761513


namespace right_triangle_XZ_length_l761_761065

theorem right_triangle_XZ_length (X Y Z : Type) [triangle X Y Z]
  (hypotenuse : XY = 13)
  (right_angle : angle X = 90)
  (angle_Y : angle Y = 60) :
  length XZ = (13 * real.sqrt 3 / 2) := sorry

end right_triangle_XZ_length_l761_761065


namespace vasya_numbers_l761_761371

theorem vasya_numbers (x y : ℚ) (h : x + y = xy ∧ xy = x / y) : x = 1 / 2 ∧ y = -1 :=
by {
  sorry
}

end vasya_numbers_l761_761371


namespace luke_total_points_l761_761633

theorem luke_total_points (rounds : ℕ) (points_per_round : ℕ) (total_points : ℕ) 
  (h1 : rounds = 177) (h2 : points_per_round = 46) : 
  total_points = 8142 := by
  have h : total_points = rounds * points_per_round := by sorry
  rw [h1, h2] at h
  exact h

end luke_total_points_l761_761633


namespace matrix_multiplication_l761_761462

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 1], ![4, -2]]

def matrix2 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![7, -3], ![2, 2]]

def product_matrix : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![23, -7], ![24, -16]]

theorem matrix_multiplication :
  matrix1 ⬝ matrix2 = product_matrix := by
  sorry

end matrix_multiplication_l761_761462


namespace bubble_pass_probability_l761_761801

-- Define the conditions and question
variable (s : Fin 35 → ℝ)
variable (distinct : ∀ i ≠ j, s i ≠ s j)

-- Define the event of a single bubble pass
-- For simplicity, we do not model the complete bubble pass algorithm,
-- but define what needs to be shown: the probability calculation outcome.

theorem bubble_pass_probability (p q : ℕ) (h : p / q = 1 / 1650 ∧ Int.gcd p q = 1) :
  p + q = 1651 :=
by
  sorry

end bubble_pass_probability_l761_761801


namespace prob_no_ball_in_own_box_l761_761835

def num_ways_valid_placement : ℕ := 84
def num_ways_total_placement : ℕ := 240

theorem prob_no_ball_in_own_box 
  (balls : Fin 5)
  (boxes : Fin 4)
  (no_box_empty : ∀ b : Fin 4, ∃ x : Fin 5, x ∈ balls)
  (no_ball_in_own : ∀ b : Fin 4, b ≠ balls b) :
  (num_ways_valid_placement : ℚ) / (num_ways_total_placement : ℚ) = 7 / 20 :=
begin
  sorry
end

end prob_no_ball_in_own_box_l761_761835


namespace vasya_numbers_l761_761343

theorem vasya_numbers :
  ∃ x y : ℝ, x + y = x * y ∧ x + y = x / y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l761_761343


namespace arithmetic_mean_alpha_X_l761_761629

-- Define the set M
def M : Set ℕ := {n | 1 ≤ n ∧ n ≤ 1000}

-- Auxiliary functions to compute α_X
def alpha_X (X : Set ℕ) [DecidablePred X] : ℕ := 
  let max_X := Finset.sup (X.toFinset) id
  let min_X := Finset.inf (X.toFinset) id
  max_X + min_X

-- Main theorem to be proven
theorem arithmetic_mean_alpha_X : 
  let non_empty_subsets := {X : Set ℕ | X ≠ ∅ ∧ X ⊆ M}
  (1 / (2^1000 - 1) * ∑ X in non_empty_subsets, alpha_X X) = 1001 :=
by
  sorry

end arithmetic_mean_alpha_X_l761_761629


namespace initial_crayons_count_l761_761694

theorem initial_crayons_count : 
  ∀ (initial_crayons added_crayons total_crayons : ℕ), 
  added_crayons = 3 ∧ total_crayons = 12 → initial_crayons + added_crayons = total_crayons → initial_crayons = 9 := 
by
  intros initial_crayons added_crayons total_crayons
  intros h1 h2
  cases h1 with h_added h_total
  rw [h_added, h_total] at h2
  linarith

end initial_crayons_count_l761_761694


namespace min_different_first_digits_l761_761837

def first_decimal_digit (n : ℕ) : ℕ :=
  (n / 10 ^ ((n : ℕ).digits 10).length.pred) % 10

theorem min_different_first_digits (n : ℕ) (h_pos : n > 0) :
  let digits := list.map (λ k : ℕ, first_decimal_digit (k * n)) (list.range' 1 9) in
  list.nodup digits → list.length (list.dedup digits) = 4 :=
begin
  sorry
end

end min_different_first_digits_l761_761837


namespace perimeter_of_triangle_DEF2_l761_761866

-- Definition of the ellipse and its parameters
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 9) = 1 

-- Definition of the foci of the ellipse
noncomputable def F1 : ℝ × ℝ := (-√7, 0)
noncomputable def F2 : ℝ × ℝ := (√7, 0)

-- Definition of the chord passing through the left focus F1
def passes_through_focus_F1 (x y : ℝ) : Prop :=
  ellipse x y ∧ (x = -√7)

-- Proving the perimeter of triangle DEF2
theorem perimeter_of_triangle_DEF2 :
  ∀ (D E : ℝ × ℝ), passes_through_focus_F1 D.1 D.2 → passes_through_focus_F1 E.1 E.2 →
  let a := 4 in
  (2 * a) + (2 * a) = 16
:= by
  intros D E hD hE a
  have : a = 4 := rfl
  sorry

end perimeter_of_triangle_DEF2_l761_761866


namespace Vasya_numbers_l761_761362

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 :=
by
  sorry

end Vasya_numbers_l761_761362


namespace angle_between_altitude_and_bisector_l761_761578

-- Define the right triangle with given sides
structure RightTriangle (a b c : ℝ) :=
  (hypotenuse_condition : c^2 = a^2 + b^2)

-- Instantiate a specific right triangle with given side lengths
def specific_triangle : RightTriangle 2 (2 * Real.sqrt 3) 4 :=
{ hypotenuse_condition := by 
  rw [sq, sq, sq, sq, sq, mul_eq_mul_right_iff];
  norm_num;
  rw [Real.sqrt_mul_self] <|> sorry }

-- Define the calculation for the altitude and angle bisector
def altitude (a b c : ℝ) (rt : RightTriangle a b c) :=
  (2 * a * b) / c

def angle_bisector_degree := 45.0 -- specifically for right triangle's 90-degree bisector

-- The main theorem: The angle between the altitude and the angle bisector drawn from the right angle is 15 degrees
theorem angle_between_altitude_and_bisector : 
  ∀ (a b c : ℝ) (rt : RightTriangle a b c),
  a = 2 → b = 2 * Real.sqrt 3 → c = 4 →
  let h := altitude a b c rt in
  let θ := 45.0 in -- angle bisector for the 90-degree angle
  15.0 = θ - 30.0 :=
by
  intros a b c rt a_eq b_eq c_eq h θ,
  norm_num,
  sorry

end angle_between_altitude_and_bisector_l761_761578


namespace solution_to_fraction_problem_l761_761060

noncomputable def fraction_problem : Prop :=
  ∀ (a b : Nat), a > 0 -> b > 0 -> Nat.gcd a b = 1 ->
    ((a + 12) * b = 3 * a * (b + 12)) -> a = 2 ∧ b = 9

theorem solution_to_fraction_problem : fraction_problem :=
sorry

end solution_to_fraction_problem_l761_761060


namespace binary101_is_5_l761_761029

theorem binary101_is_5 : 
  let binary101 := [1, 0, 1] in
  let decimal := binary101[0] * 2^2 + binary101[1] * 2^1 + binary101[2] * 2^0 in
  decimal = 5 :=
by
  let binary101 := [1, 0, 1]
  let decimal := binary101[0] * 2^2 + binary101[1] * 2^1 + binary101[2] * 2^0
  show decimal = 5
  sorry

end binary101_is_5_l761_761029


namespace odd_and_monotonically_decreasing_l761_761769

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≥ f y

theorem odd_and_monotonically_decreasing :
  is_odd (fun x : ℝ => -x^3) ∧ is_monotonically_decreasing (fun x : ℝ => -x^3) :=
by
  sorry

end odd_and_monotonically_decreasing_l761_761769


namespace blueBirdChessTeam72_l761_761663

def blueBirdChessTeamArrangements : Nat :=
  let boys_girls_ends := 3 * 3 + 3 * 3
  let alternate_arrangements := 2 * 2
  boys_girls_ends * alternate_arrangements

theorem blueBirdChessTeam72 : blueBirdChessTeamArrangements = 72 := by
  unfold blueBirdChessTeamArrangements
  sorry

end blueBirdChessTeam72_l761_761663


namespace series_sum_is_81_l761_761454

theorem series_sum_is_81 :
  10 * (∑ k in Finset.range 9, (k^2 + k - 1) / (k * (k + 1))) = 81 :=
begin
  sorry
end

end series_sum_is_81_l761_761454


namespace sofa_love_seat_cost_l761_761427

theorem sofa_love_seat_cost (love_seat_cost : ℕ) (sofa_cost : ℕ) 
    (h₁ : love_seat_cost = 148) (h₂ : sofa_cost = 2 * love_seat_cost) :
    love_seat_cost + sofa_cost = 444 := 
by
  sorry

end sofa_love_seat_cost_l761_761427


namespace minimize_PR_plus_RQ_l761_761098

-- Lean statement defining the proof problem
theorem minimize_PR_plus_RQ (m : ℚ) : 
  let P := (-2 : ℚ, -3 : ℚ)
  let Q := (5 : ℚ, 3 : ℚ)
  let R := (2 : ℚ, m)
  ∃ (m_val : ℚ), m_val = 3 / 7 ∧ 
  (∀ m', ((HasDist.dist P (2, m') + HasDist.dist (2, m') Q) ≥ (HasDist.dist P (2, m_val) + HasDist.dist (2, m_val) Q))) :=
by
  sorry

end minimize_PR_plus_RQ_l761_761098


namespace meredith_distance_l761_761235

-- Define the vector components for each leg of the journey
def vector_miles (a b : ℝ) : ℝ × ℝ := (a * Math.cos (45 * Math.pi / 180), b * Math.sin (45 * Math.pi / 180))

-- Meredith's journey vectors
def leg1 : ℝ × ℝ := vector_miles 5 5
def leg2 : ℝ × ℝ := vector_miles 15 (-15)
def leg3 : ℝ × ℝ := vector_miles (-25) (-25)
def leg4 : ℝ × ℝ := vector_miles (-35) 35
def leg5 : ℝ × ℝ := vector_miles 20 20

-- Compute the sum of vectors
def sum_vectors : ℝ × ℝ := (leg1.1 + leg2.1 + leg3.1 + leg4.1 + leg5.1, leg1.2 + leg2.2 + leg3.2 + leg4.2 + leg5.2)

-- Calculate the distance from the origin
def distance_from_origin : ℝ := Real.sqrt (sum_vectors.1 ^ 2 + sum_vectors.2 ^ 2)

-- The final theorem statement
theorem meredith_distance : distance_from_origin = 20 :=
by
  sorry

end meredith_distance_l761_761235


namespace longest_diagonal_of_rhombus_l761_761422

noncomputable def length_of_longest_diagonal (area : ℝ) (ratio : ℝ) :=
  (let x := (area * 8 / (ratio + 1)^2).sqrt in 4 * x)

theorem longest_diagonal_of_rhombus :
  length_of_longest_diagonal 144  (4 / 3) = 8 * Real.sqrt 6 :=
by
  sorry

end longest_diagonal_of_rhombus_l761_761422


namespace parabola_has_two_distinct_roots_l761_761871

theorem parabola_has_two_distinct_roots
  (a b c : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (p : ℝ → ℝ), 
  (p = (λ x, a*x^2 + 2*b*x + c) ∨ 
   p = (λ x, b*x^2 + 2*c*x + a) ∨ 
   p = (λ x, c*x^2 + 2*a*x + b)) ∧ 
  (∃ Δ : ℝ, Δ > 0 ∧ Δ = ?m*) :=
sorry

end parabola_has_two_distinct_roots_l761_761871


namespace value_of_transformed_product_of_roots_l761_761213

theorem value_of_transformed_product_of_roots 
  (a b : ℚ)
  (h1 : 3 * a^2 + 4 * a - 7 = 0)
  (h2 : 3 * b^2 + 4 * b - 7 = 0)
  (h3 : a ≠ b) : 
  (a - 2) * (b - 2) = 13 / 3 :=
by
  -- The exact proof would be completed here.
  sorry

end value_of_transformed_product_of_roots_l761_761213


namespace min_value_of_f_l761_761505

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 - 2 * x + 16) / (2 * x - 1)

theorem min_value_of_f :
  ∃ x : ℝ, x ≥ 1 ∧ f x = 9 ∧ (∀ y : ℝ, y ≥ 1 → f y ≥ 9) :=
by { sorry }

end min_value_of_f_l761_761505


namespace heights_inscribed_circle_inequality_l761_761223

theorem heights_inscribed_circle_inequality
  {h₁ h₂ r : ℝ} (h₁_pos : 0 < h₁) (h₂_pos : 0 < h₂) (r_pos : 0 < r)
  (triangle_heights : ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a * h₁ = b * h₂ ∧ 
                                       a + b > c ∧ h₁ = 2 * r * (a + b + c) / (a * b)):
  (1 / (2 * r) < 1 / h₁ + 1 / h₂ ∧ 1 / h₁ + 1 / h₂ < 1 / r) :=
sorry

end heights_inscribed_circle_inequality_l761_761223


namespace max_median_sum_l761_761046

open Finset

variable {α : Type*}

-- Define the 10 groups and the conditions on their medians.
def group_medians (groups : list (finset nat)) : Prop :=
  (∀ (g : finset nat), g ∈ groups → g.card = 5) ∧ 
  (⋃₀ (list.to_finset groups)) = finset.range 51 \ {0}

theorem max_median_sum : 
  ∀ (groups : list (finset nat)), group_medians groups → 
  ∑ (g : finset nat) in (list.to_finset groups), median g = 345 :=
sorry

end max_median_sum_l761_761046


namespace radius_of_middle_circle_l761_761950

theorem radius_of_middle_circle (r : ℝ) (r1 r2 r3 r4 r5 : ℝ) :
  r1 = 10 → r5 = 24 → r1 + r2 + r3 + r4 + r5 = 70 →
  r2 = r1 * r → r3 = r2 * r → r4 = r3 * r → r = 1.2 →
  10 * (1.2 ^ 2) = 14.4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h4, h5, h6, h7]
  exact rfl

end radius_of_middle_circle_l761_761950


namespace february_first_day_of_week_l761_761566

theorem february_first_day_of_week 
  (feb13_is_wednesday : ∃ day, day = 13 ∧ day_of_week = "Wednesday") :
  ∃ day, day = 1 ∧ day_of_week = "Friday" :=
sorry

end february_first_day_of_week_l761_761566


namespace problem_proof_l761_761910

theorem problem_proof (M N : ℕ) 
  (h1 : 4 * 63 = 7 * M) 
  (h2 : 4 * N = 7 * 84) : 
  M + N = 183 :=
sorry

end problem_proof_l761_761910


namespace simplify_result_l761_761254

noncomputable def simplify_expression : ℂ :=
  6 * (2 - complex.i) + 4 * complex.i * (6 - complex.i)

theorem simplify_result : simplify_expression = 16 + 18 * complex.i := 
by 
  unfold simplify_expression
  have : complex.i ^ 2 = -1 := complex.i_sq
  calc
    6 * (2 - complex.i) + 4 * complex.i * (6 - complex.i)
      = 6 * 2 - 6 * complex.i + 4 * complex.i * 6 - 4 * complex.i ^ 2 : by ring
    ... = 12 - 6 * complex.i + 24 * complex.i - 4 * (-1)       : by rw this
    ... = 12 - 6 * complex.i + 24 * complex.i + 4
    ... = 16 + 18 * complex.i


end simplify_result_l761_761254


namespace exists_n_digit_number_with_one_appended_and_once_one_l761_761802

theorem exists_n_digit_number_with_one_appended_and_once_one (n : ℕ) : 
  ∃ (x : ℕ), (x < 3^n) ∧ (∀ i, digit_at x i ∈ {1, 2, 3}) ∧ (digit_count x 1 = 1) ∧ (digit_at x n = 1) :=
by
  -- Definitions here
  def digit_at (x : ℕ) (i : ℕ) : ℕ := sorry  -- Function to get the digit at position i
  def digit_count (x : ℕ) (d : ℕ) : ℕ := sorry  -- Function to count the occurrences of digit d in number x
  sorry

end exists_n_digit_number_with_one_appended_and_once_one_l761_761802


namespace sum_of_possible_values_of_f1_l761_761987

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, f(f(x + y)) = f(x) * f(y) + f(x) + f(y) + x * y

theorem sum_of_possible_values_of_f1 : f(1) = 1 := by
  sorry

end sum_of_possible_values_of_f1_l761_761987


namespace Galia_number_problem_l761_761841

theorem Galia_number_problem :
  ∀ k : ℤ, ∃ N : ℤ, ((k * N + N) / N - N = k - 100) → N = 101 :=
by
  intros k N h
  sorry

end Galia_number_problem_l761_761841


namespace ratio_x_y_half_l761_761323

variable (x y z : ℝ)

theorem ratio_x_y_half (h1 : (x + 4) / 2 = (y + 9) / (z - 3))
                      (h2 : (y + 9) / (z - 3) = (x + 5) / (z - 5)) :
  x / y = 1 / 2 :=
by
  sorry

end ratio_x_y_half_l761_761323


namespace power_of_a_point_zero_and_concyclic_l761_761319

noncomputable theory

open_locale classical

variables (P : fin 6 → (ℝ × ℝ)) (k : ℝ)

-- Given conditions:
-- 1. There are 6 points on the plane such that no three of them are collinear.
def no_three_collinear : Prop := ∀ (i j l : fin 6), i ≠ j → j ≠ l → i ≠ l →
  let A := P i, B := P j, C := P l in (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) ≠ 0

-- 2. For any 4 points among these 6, there exists a point whose power with respect to 
-- the circle passing through the other three points is a constant value k.
def power_of_a_point (O : ℝ × ℝ) (R : ℝ) (P : ℝ × ℝ) : ℝ := (P.1 - O.1) ^ 2 + (P.2 - O.2) ^ 2 - R ^ 2

def power_condition : Prop :=
  ∀ (a b c d : fin 6) (h : list.nodup [a, b, c, d]), ∃ (p ∈ [a, b, c, d].to_finset), ∃ (O : ℝ × ℝ) (R : ℝ),
  {P a, P b, P c, P d}.to_finset = insert (P p) {P a, P b, P c, P d}.erase (P p) ∧
  power_of_a_point O R (P p) = k

-- Prove that 
theorem power_of_a_point_zero_and_concyclic
  (h_no_three_collinear : no_three_collinear P)
  (h_power_condition : power_condition P k) :
  k = 0 ∧ ∃ (O : ℝ × ℝ) (R : ℝ), ∀ i, (P i).1 ^ 2 + (P i).2 ^ 2 = R ^ 2 + 2 * O.1 * (P i).1 + 2 * O.2 * (P i).2 :=
sorry

end power_of_a_point_zero_and_concyclic_l761_761319


namespace length_of_second_train_is_153_l761_761706

def speed_first_train_kmh : ℝ := 80
def speed_second_train_kmh : ℝ := 65
def length_first_train_meter : ℝ := 121
def time_seconds : ℝ := 6.802214443534172

def relative_speed_m_s : ℝ :=
  (speed_first_train_kmh + speed_second_train_kmh) * 1000 / 3600

def total_distance_meter : ℝ :=
  relative_speed_m_s * time_seconds

def length_second_train_meter : ℝ :=
  total_distance_meter - length_first_train_meter

theorem length_of_second_train_is_153 :
  length_second_train_meter ≈ 153 :=
sorry

end length_of_second_train_is_153_l761_761706


namespace sum_first_10_terms_abs_a_n_l761_761521

noncomputable def a_n (n : ℕ) : ℤ :=
  if n = 0 then 0 else 3 * n - 7

def abs_a_n (n : ℕ) : ℤ :=
  if n = 1 ∨ n = 2 then -3 * n + 7 else 3 * n - 7

def sum_abs_a_n (n : ℕ) : ℤ :=
  if n = 0 then 0 else List.sum (List.map abs_a_n (List.range n))

theorem sum_first_10_terms_abs_a_n : sum_abs_a_n 10 = 105 := 
  sorry

end sum_first_10_terms_abs_a_n_l761_761521


namespace polygonal_chain_has_edge_of_cube_l761_761681

-- Definitions for vertices, edges, and polygonal chains
variables {V : Type} [Fintype V]

def vertices_of_cube (V : Type) [Fintype V] : Finset (V) := sorry
def edge_of_cube (v1 v2 : V) : Prop := sorry
def polygonal_chain (seq : list V) : Prop := sorry
def non_self_intersecting (seq : list V) : Prop := sorry
def closed_chain (seq : list V) : Prop := sorry

theorem polygonal_chain_has_edge_of_cube
  {V : Type} [Fintype V] 
  (seq : list V)
  (h1 : closed_chain seq)
  (h2 : non_self_intersecting seq)
  (h3 : (polygonal_chain seq) ∧ (list.length seq = 8))
  (h4 : ∀ v ∈ seq, v ∈ vertices_of_cube V) :
  ∃ (v1 v2 : V), edge_of_cube v1 v2 ∧ (v1, v2) ∈ list.to_finset (list.zip seq (list.tail seq ++ [list.head seq .get_or_else seq.head])) := 
sorry

end polygonal_chain_has_edge_of_cube_l761_761681


namespace dirichlet_poisson_solution_l761_761256

noncomputable def solution (x y z : ℝ) : ℝ :=
  (x^2 + y^2 + z^2 - 4) * x * y + 14

theorem dirichlet_poisson_solution (u : ℝ → ℝ → ℝ → ℝ) :
  (∀ x y z : ℝ, 
    
    -- Given condition: Poisson equation
    (∂^2 / ∂x^2 + ∂^2 / ∂y^2 + ∂^2 / ∂z^2) (u x y z) = 14 * x * y) 
    ∧ 
    
    -- Given condition: Dirichlet boundary
    (∀ (x y z : ℝ), (x^2 + y^2 + z^2) = 4 → (u x y z) = 14) 
    → 
    
    -- Conclusion: Solution function
    (u = solution) := 
sorry

end dirichlet_poisson_solution_l761_761256


namespace swap_rows_matrix_exists_l761_761825

theorem swap_rows_matrix_exists (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (M : Matrix (Fin 2) (Fin 2) ℝ) : 
  (N = ![![0, 1], ![1, 0]]) →
  (N ⬝ M = λ a b c d, ![![c, d], ![a, b]]) :=
by
  sorry

end swap_rows_matrix_exists_l761_761825


namespace arrange_numbers_condition_l761_761625

theorem arrange_numbers_condition (n : ℕ) (h : 0 < n):
  ∃ (s : fin n → ℕ), 
    (∀ i j : fin n, i ≠ j → 
    (let mean := (s i + s j) / 2 in 
    ∀ k : fin n, s k ≠ mean)) := by
  sorry

end arrange_numbers_condition_l761_761625


namespace find_general_term_a_l761_761515

-- Define the sequence and conditions
noncomputable def S (n : ℕ) : ℚ :=
  if n = 0 then 0 else (n - 1) / (n * (n + 1))

-- General term to prove
def a (n : ℕ) : ℚ := 1 / (2^n) - 1 / (n * (n + 1))

theorem find_general_term_a :
  ∀ n : ℕ, n > 0 → S n + a n = (n - 1) / (n * (n + 1)) :=
by
  intro n hn
  sorry -- Proof omitted

end find_general_term_a_l761_761515


namespace part1_part2_part3_l761_761895

-- Define the function f and its derivative f'
def f (a : ℝ) (x : ℝ) := a * x + x * real.log x
def f' (a : ℝ) (x : ℝ) := a + real.log x + 1

-- The first part: Given f'(e) = 3, prove a = 1
theorem part1 (h : f' a real.exp 1 = 3) : a = 1 := sorry

-- The second part: Finding the monotonic intervals
theorem part2 : 
  (∀ x : ℝ, (0 < x ∧ x < real.exp (-2)) → f' 1 x < 0) ∧ 
  (∀ x : ℝ, (real.exp (-2) < x) → f' 1 x > 0) := sorry

-- The third part:
-- Given that the inequality f(x) - kx + k > 0 holds for any x ∈ (1, +∞), prove the maximum integer k is 3
theorem part3 (h : ∀ x : ℝ, 1 < x → f 1 x - k * x + k > 0) : k < 4 := sorry

end part1_part2_part3_l761_761895


namespace girls_more_than_boys_l761_761745

theorem girls_more_than_boys (total_students boys : ℕ) (h : total_students = 466) (b : boys = 127) (gt : total_students - boys > boys) :
  total_students - 2 * boys = 212 := by
  sorry

end girls_more_than_boys_l761_761745


namespace general_term_a_n_sum_T_n_l761_761900

open Real BigOperators Nat

noncomputable def radius (a_n : ℕ → ℝ) (n : ℕ) := sqrt (2 * a_n n + n)
noncomputable def distance (n : ℕ) := sqrt n
noncomputable def next_a_n (a_n : ℕ → ℝ) (n : ℕ) : ℝ := (1 / 4) * (radius a_n n ^ 2 - distance n ^ 2)
noncomputable def a_seq (n : ℕ) : ℝ := (2 ^ (n - 1) : ℝ)

theorem general_term_a_n : ∀ (a_n : ℕ → ℝ) (n : ℕ), 
  (a_n 1 = 1) → 
  ((∀ n : ℕ, a_n (n + 1) = next_a_n a_n n) → 
  ∀ n : ℕ, a_n n = a_seq n) :=
sorry

noncomputable def b_seq (a_n : ℕ → ℝ) (n : ℕ) : ℝ := (n / (4 * a_n n))
noncomputable def T_seq (a_n : ℕ → ℝ) (n : ℕ) : ℝ := (∑ k in range n, b_seq a_n (k + 1))

theorem sum_T_n : ∀ (a_n : ℕ → ℝ) (n : ℕ),
  (a_n 1 = 1) → 
  ((∀ n : ℕ, a_n (n + 1) = next_a_n a_n n) → 
  T_seq a_n n = 1 - ((n + 2) / (2 ^ (n + 1) : ℝ))) :=
sorry

end general_term_a_n_sum_T_n_l761_761900


namespace quadratic_factor_n_l761_761851

theorem quadratic_factor_n (n : ℤ) (h : ∃ m : ℤ, (x + 5) * (x + m) = x^2 + 7 * x + n) : n = 10 :=
sorry

end quadratic_factor_n_l761_761851


namespace not_factorable_l761_761656

theorem not_factorable (n : ℕ) : n = 2^1000 →
  ¬ ∃ (f g : ℤ[X]), f.degree > 0 ∧ g.degree > 0 ∧ (x^2 + x)^(2^n) + 1 = f * g :=
by {
  intro hn,
  rw hn,
  sorry,
}

end not_factorable_l761_761656


namespace perpendicular_lines_l761_761922

theorem perpendicular_lines (a : ℝ) : 
  let l1 := λ (x y : ℝ), a * x + 2 * y + 6 = 0 in
  let l2 := λ (x y : ℝ), x + (a - 1) * y + a^2 - 1 = 0 in
  (∀ x1 y1 x2 y2 : ℝ, l1 x1 y1 → l2 x2 y2 → x1 * x2 + y1 * y2 = 0) → 
  a = 2 / 3 := 
by {
  sorry
}

end perpendicular_lines_l761_761922


namespace infinite_solutions_l761_761828

theorem infinite_solutions (x y : ℤ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x^2 + y^2 = xy + x^3) :
  infinite {p : ℤ × ℤ | let (a, b) := p in 0 < a ∧ 0 < b ∧ a^2 + b^2 = a * b + a^3} :=
sorry

end infinite_solutions_l761_761828


namespace number_of_lines_passing_through_point_and_forming_given_area_l761_761927

theorem number_of_lines_passing_through_point_and_forming_given_area :
  ∃ l : ℝ → ℝ, (∀ x y : ℝ, l 1 = 1) ∧ (∃ (a b : ℝ), abs ((1/2) * a * b) = 2)
  → (∃ n : ℕ, n = 4) :=
by
  sorry

end number_of_lines_passing_through_point_and_forming_given_area_l761_761927


namespace Vasya_numbers_l761_761347

theorem Vasya_numbers : 
  ∃ x y : ℚ, (x + y = x * y) ∧ (x * y = x / y) ∧ x = 1 / 2 ∧ y = -1 :=
by 
  use (1 / 2)
  use (-1)
  split
  case left {
    calc 
      (1 / 2) + (-1) = -1 / 2 : by norm_num,
      -1 / 2 = ((1 / 2) * (-1)) : by norm_num,
      (1 / 2) + (-1) = (1 / 2) * (-1) : by norm_num,
  },
  case right {
    calc 
      (1 / 2) * (-1) = -1 / 2 : by norm_num,
      -1 / 2 = (1 / 2) / (-1) : by norm_num,
      (1 / 2) * (-1) = (1 / 2) / (-1) : by norm_num,
  }

end Vasya_numbers_l761_761347


namespace f_neg_2_equals_neg_12_l761_761511

def is_symmetric_about (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x : ℝ, f (2 * c - x) = f x

def f (x : ℝ) : ℝ := 
  if x ≥ 1 then x * (1 - x) else sorry

theorem f_neg_2_equals_neg_12 (h_symm : is_symmetric_about f 1) : f (-2) = -12 :=
by
  -- actual proof steps go here
  sorry

end f_neg_2_equals_neg_12_l761_761511


namespace find_principal_amount_l761_761830

theorem find_principal_amount
  (A5 : ℝ)
  (r1 r2 r3 r4 r5 : ℝ)
  (P : ℝ)
  (hA5 : A5 = 1120)
  (h_r1 : r1 = 0.05)
  (h_r2 : r2 = 0.03)
  (h_r3 : r3 = 0.06)
  (h_r4 : r4 = 0.04)
  (h_r5 : r5 = 0.035)
  (hP : P = 908.34) :
  let P4 := A5 / (1 + r5),
      P3 := P4 / (1 + r4),
      P2 := P3 / (1 + r3),
      P1 := P2 / (1 + r2),
      P0 := P1 / (1 + r1)
  in P0 = P :=
by
  sorry

end find_principal_amount_l761_761830


namespace infinite_solutions_l761_761495

theorem infinite_solutions (n : ℤ) :
  (√(n + 2) ≤ √(3 * n + 1)) ∧ (√(3 * n + 1) < √(4 * n - 7)) ↔ ∃ (k : ℤ), ∀ n, n ≥ k :=
by
  sorry

end infinite_solutions_l761_761495


namespace rhombus_area_l761_761304

theorem rhombus_area (ABCD : Type*) [rhombus ABCD] 
  (perimeter_ABCD : ℝ) (AC : ℝ) (h1 : perimeter_ABCD = 8 * Real.sqrt 5) (h2 : AC = 4) :
  rhombus_area ABCD = 16 :=
by
  sorry

end rhombus_area_l761_761304


namespace integral_x_squared_sub_x_integral_abs_x_sub_2_integral_sqrt_1_sub_x_squared_l761_761336

-- Problem 1: Prove the integral of (x^2 - x) from 0 to 1 is -1/6
theorem integral_x_squared_sub_x :
  ∫ x in 0..1, (x^2 - x) = -1/6 := 
sorry

-- Problem 2: Prove the integral of |x-2| from 1 to 3 is 1
theorem integral_abs_x_sub_2 :
  ∫ x in 1..3, |x - 2| = 1 := 
sorry

-- Problem 3: Prove the integral of sqrt(1 - x^2) from 0 to 1 is π/4
theorem integral_sqrt_1_sub_x_squared :
  ∫ x in 0..1, (sqrt (1 - x^2)) = π/4 := 
sorry

end integral_x_squared_sub_x_integral_abs_x_sub_2_integral_sqrt_1_sub_x_squared_l761_761336


namespace count_sets_of_values_l761_761730

theorem count_sets_of_values (a b c d : ℕ) (h1 : a > b > c > d) 
    (h2 : a + b + c + d = 2010) (h3 : a ^ 2 - b ^ 2 + c ^ 2 - d ^ 2 = 2010) : 
    ∃ n, n = 501 ∧ ∃ (s : Finset (ℕ × ℕ × ℕ × ℕ)), 
    s = (Finset.univ.filter (λ x, let ⟨a, b, c, d⟩ := x in 
      a > b ∧ b > c ∧ c > d ∧ a + b + c + d = 2010 ∧ a ^ 2 - b ^ 2 + c ^ 2 - d ^ 2 = 2010)) ∧ s.card = 501 :=
by
  sorry

end count_sets_of_values_l761_761730


namespace points_concyclic_l761_761942

variable (A B C D E F G H I : Type*)
variable [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F] [Inhabited G] [Inhabited H] [Inhabited I]

-- Definitions of the points F, G, I, H, intersections and conditions
variable (triABC : Triangle A B C)
variable (altAD : Altitude A D)
variable (altBE : Altitude B E)
variable (pointF : Segment A D)
variable (pointG : Segment B E)
variable (ratioAF_FD : Ratio AF FD)
variable (ratioBG_GE : Ratio BG GE)
variable (lineCF : Line C F)
variable (lineCG : Line C G)
variable (intersectionCF_BE : Intersect CF BE H)
variable (intersectionCG_AD : Intersect CG AD I)

theorem points_concyclic :
  acute_triangle triABC → 
  altitude triABC altAD D → 
  altitude triABC altBE E → 
  on_segment pointF AD F → 
  on_segment pointG BE G → 
  ratioAF_FD = ratioBG_GE → 
  intersect CF BE H → 
  intersect CG AD I → 
  concyclic_points F G I H :=
sorry

end points_concyclic_l761_761942


namespace maximum_n_satisfies_Sn_pos_l761_761179

theorem maximum_n_satisfies_Sn_pos (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 16 > 0) (h2 : a 17 < 0) (h3 : a 16 > |a 17|) : 
  ∃ n, n = 32 ∧ (∀ m < 32, S m a > 0) :=
  sorry

end maximum_n_satisfies_Sn_pos_l761_761179


namespace complex_magnitude_l761_761492

noncomputable def complex_magnitude_expression : ℂ :=
  (1 / 3 - (2 / 3) * complex.I) ^ 4.5

theorem complex_magnitude :
  |complex_magnitude_expression| = (5:ℂ) ^ 2.25 / (27 * complex.sqrt 3) :=
by
  sorry

end complex_magnitude_l761_761492


namespace solve_problem_1_solve_problem_2_l761_761781

/-
Problem 1:
Given the equation 2(x - 1)^2 = 18, prove that x = 4 or x = -2.
-/
theorem solve_problem_1 (x : ℝ) : 2 * (x - 1)^2 = 18 → (x = 4 ∨ x = -2) :=
by
  sorry

/-
Problem 2:
Given the equation x^2 - 4x - 3 = 0, prove that x = 2 + √7 or x = 2 - √7.
-/
theorem solve_problem_2 (x : ℝ) : x^2 - 4 * x - 3 = 0 → (x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) :=
by
  sorry

end solve_problem_1_solve_problem_2_l761_761781


namespace distance_MN_equal_2sqrt5_l761_761952

/-- Define the curve C in rectangular coordinates -/
def curve_C (x y : ℝ) : Prop := y = |x|

/-- Define the line l in rectangular coordinates -/
def line_l (x y : ℝ) : Prop := x - 2 * y - 2 = 0

/-- Define the distance between a point and the line -/
def distance_from_point_to_line (x1 y1 : ℝ) : ℝ :=
  abs (x1 - 2 * y1 - 2) / sqrt (1^2 + (-2)^2)

/-- M and N are distinct points on curve C such that their distances to line l are √5 -/
def points_on_curve_C_with_distance_to_line_l : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((-1, 1), (3, 3))

/-- Statement to prove the distance |MN| given the conditions -/
theorem distance_MN_equal_2sqrt5 :
  let M := points_on_curve_C_with_distance_to_line_l.1,
      N := points_on_curve_C_with_distance_to_line_l.2 in
  curve_C M.1 M.2 ∧ curve_C N.1 N.2 ∧
  line_l M.1 M.2 ∧ line_l N.1 N.2 ∧
  distance_from_point_to_line M.1 M.2 = sqrt 5 ∧ 
  distance_from_point_to_line N.1 N.2 = sqrt 5 →
  dist (M.1, M.2) (N.1, N.2) = 2 * sqrt 5 :=
by
  sorry

end distance_MN_equal_2sqrt5_l761_761952


namespace abs_negative_five_l761_761274

theorem abs_negative_five : abs (-5) = 5 :=
by
  sorry

end abs_negative_five_l761_761274


namespace minimum_positive_m_is_sqrt7_div_2_l761_761118

noncomputable def minimum_positive_value_of_m 
  (line : ℝ → ℝ → Prop)
  (parabola : ℝ → ℝ → Prop) 
  (A B : ℝ × ℝ) 
  (orthogonal : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop) : ℝ :=
  ∃ m > 0, ∀ (x y : ℝ), 
    (line x y → parabola x y → ∃ P, orthogonal P A B) → 
    m = (Real.sqrt 7 / 2)

theorem minimum_positive_m_is_sqrt7_div_2 :
  minimum_positive_value_of_m 
    (λ x y, x - 4 * y + 1 = 0) 
    (λ x y, y = x^2) 
    (0, 2 + Real.sqrt 7 / 2) 
    (0, 2 - Real.sqrt 7 / 2) 
    (λ P A B, (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = -1) := 
sorry

end minimum_positive_m_is_sqrt7_div_2_l761_761118


namespace sum_of_angles_AOB_COD_sum_of_angles_BOC_DOA_l761_761953

noncomputable theory

variables {α β γ δ : ℝ} 

def are_angles_of_quadrilateral (α β γ δ : ℝ) : Prop :=
  α + β + γ + δ = 360

theorem sum_of_angles_AOB_COD {α β γ δ : ℝ} (h : are_angles_of_quadrilateral α β γ δ) :
  (180 - (α + β) / 2) + (180 - (γ + δ) / 2) = 180 :=
by sorry

theorem sum_of_angles_BOC_DOA {α β γ δ : ℝ} (h : are_angles_of_quadrilateral α β γ δ) :
  (180 - (β + γ) / 2) + (180 - (δ + α) / 2) = 180 :=
by sorry

end sum_of_angles_AOB_COD_sum_of_angles_BOC_DOA_l761_761953


namespace count_apollonian_problems_l761_761020

theorem count_apollonian_problems : 
  let n := 3 in  -- number of types (point, plane, finite-radius sphere)
  let k := 4 in  -- number of elements chosen (4 given objects)
  Nat.binomial (n + k - 1) k = 15 := -- combinatorial calculation
by
  let n := 3
  let k := 4
  calc
    Nat.binomial (n + k - 1) k = Nat.binomial 6 4       := rfl
    ... = 15                                            := by norm_num


end count_apollonian_problems_l761_761020


namespace responseRateChange_l761_761760

def responseRate (responses : ℕ) (customers : ℕ) : ℝ :=
  (responses.toReal / customers.toReal) * 100

theorem responseRateChange :
  let originalResponses := 10
  let originalCustomers := 100
  let finalResponses := 27
  let finalCustomers := 90
  let originalRate := responseRate originalResponses originalCustomers
  let finalRate := responseRate finalResponses finalCustomers
  (finalRate - originalRate) / originalRate * 100 = 200 := 
by
  sorry

end responseRateChange_l761_761760


namespace sequence_inequality_l761_761616

theorem sequence_inequality
  (x : ℕ → ℝ)
  (n : ℕ)
  (h1 : ∀ k, 1 ≤ k → k ≤ n → 0 < x k)
  (h2 : ∀ k, 1 ≤ k → k ≤ n → x k < 1)
  (h3 : ∀ k1 k2, 1 ≤ k1 → k1 ≤ k2 → k2 ≤ n → x k1 ≤ x k2):
  (∑ k in (finset.range n).map (λ i, i + 1), x k ^ (2 * k) / (1 - x k ^ (k + 1))^2) <
  n / ((n + 1) * (1 - x n)^2) := 
by
  sorry

end sequence_inequality_l761_761616


namespace positive_difference_enrollment_l761_761707

theorem positive_difference_enrollment 
  (highest_enrollment : ℕ)
  (lowest_enrollment : ℕ)
  (h_highest : highest_enrollment = 2150)
  (h_lowest : lowest_enrollment = 980) :
  highest_enrollment - lowest_enrollment = 1170 :=
by {
  -- Proof to be added here
  sorry
}

end positive_difference_enrollment_l761_761707


namespace smallest_sum_is_neg10_l761_761069

noncomputable def smallest_sum (s : Finset ℤ) : ℤ :=
  if h : 3 ≤ s.card then
    (s.val.sort (· < ·)).take 3 |>.sum
  else
    0

theorem smallest_sum_is_neg10 : 
  smallest_sum (Finset.ofList [10, 30, -12, 15, -8]) = -10 :=
by
  sorry

end smallest_sum_is_neg10_l761_761069


namespace keaton_harvest_frequency_l761_761202

def keaton_harvest (total_earnings yearly_earnings_apples earnings_per_apple_harvest earnings_per_orange_harvest: ℕ) : ℕ :=
let apple_harvests_per_year := 12 / 3 in
let yearly_earnings_apples := apple_harvests_per_year * earnings_per_apple_harvest in
let yearly_earnings_oranges := total_earnings - yearly_earnings_apples in
let orange_harvests_per_year := yearly_earnings_oranges / earnings_per_orange_harvest in
12 / orange_harvests_per_year

theorem keaton_harvest_frequency :
  keaton_harvest 420 120 30 50 = 2 :=
by
  -- Omit
  sorry

end keaton_harvest_frequency_l761_761202


namespace range_of_function_l761_761309

theorem range_of_function :
  ∀ (x : ℝ), -1 < (-1 + 4 / (1 + 2^x)) ∧ (-1 + 4 / (1 + 2^x)) < 3 :=
by
  sorry

end range_of_function_l761_761309


namespace vasya_numbers_l761_761368

theorem vasya_numbers (x y : ℚ) (h : x + y = xy ∧ xy = x / y) : x = 1 / 2 ∧ y = -1 :=
by {
  sorry
}

end vasya_numbers_l761_761368


namespace binary_to_decimal_l761_761024

theorem binary_to_decimal :
  ∀ n : ℕ, n = 101 →
  ∑ i in Finset.range 3, (n / 10^i % 10) * 2^i = 5 :=
by
  sorry

end binary_to_decimal_l761_761024


namespace sum_of_scores_in_two_ways_between_0_and_210_l761_761937

-- Define the variables and the conditions
def c : ℕ := sorry -- number of correct answers
def u : ℕ := sorry -- number of unanswered questions
def i : ℕ := 30 - c - u -- number of incorrect answers

def S (c u : ℕ) : ℝ := 7 * c + 1.5 * u

-- Prove the sum of scores that can be obtained in exactly two ways is 195
theorem sum_of_scores_in_two_ways_between_0_and_210 :
  (∑ s in (finset.filter (λ s, (set.card { (c, u) : ℕ × ℕ | S c u = s ∧ 0 ≤ c ∧ c ≤ 30 ∧ 0 ≤ u ∧ u ≤ 30 ∧ 0 ≤ 30 - c - u}.to_finset.card = 2) (finset.range 211)), (λ s, s)) = 195 := sorry

end sum_of_scores_in_two_ways_between_0_and_210_l761_761937


namespace average_price_of_blankets_l761_761418

noncomputable def averagePrice (p1 p2 p3 q1 q2 q3 : ℕ) : ℕ := 
  (p1 * q1 + p2 * q2 + p3 * q3) / (q1 + q2 + q3)

theorem average_price_of_blankets :
  ∀ (p1 p2 p3 q1 q2 q3 : ℕ),
  p1 = 100 → q1 = 4 →
  p2 = 150 → q2 = 5 →
  p3 = 350 → q3 = 2 →
  averagePrice p1 p2 p3 q1 q2 q3 = 168 :=
by
  intros
  unfold averagePrice
  rw [h, h_1, h_2, h_3, h_4, h_5]
  norm_num
  sorry

end average_price_of_blankets_l761_761418


namespace total_sand_volume_l761_761412

noncomputable def cone_diameter : ℝ := 10
noncomputable def cone_radius : ℝ := cone_diameter / 2
noncomputable def cone_height : ℝ := 0.75 * cone_diameter
noncomputable def cylinder_height : ℝ := 0.5 * cone_diameter
noncomputable def total_volume : ℝ := (1 / 3 * Real.pi * cone_radius^2 * cone_height) + (Real.pi * cone_radius^2 * cylinder_height)

theorem total_sand_volume : total_volume = 187.5 * Real.pi := 
by
  sorry

end total_sand_volume_l761_761412


namespace part1_part2_l761_761129

noncomputable def vector (a b : ℝ) := (a, b)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def a := vector 1 2
def b := vector (-3) 2

def x (k : ℝ) := vector (k * 1 + (-3)) (k * 2 + 2)
def y := vector (1 + 3 * 3) (2 - 3 * 2)

theorem part1 (k : ℝ) : dot_product (x k) y = 0 → k = 19 := sorry

theorem part2 (k : ℝ) : k < 19 ∧ k ≠ -1/3 → dot_product (x k) y < 0 := sorry

end part1_part2_l761_761129


namespace product_of_perimeters_l761_761315

theorem product_of_perimeters (x y : ℝ) (h1 : x^2 + y^2 = 130) (h2 : x^2 - y^2 = 58) :
  4 * real.sqrt 94 * 24 = 96 * real.sqrt 94 :=
by
  sorry

end product_of_perimeters_l761_761315
