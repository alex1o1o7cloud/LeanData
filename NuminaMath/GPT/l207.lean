import Mathlib

namespace new_person_weight_l207_207245

variable {W : ℝ} -- Total weight of the original group of 15 people
variable {N : ℝ} -- Weight of the new person

theorem new_person_weight
  (avg_increase : (W - 90 + N) / 15 = (W - 90) / 14 + 3.7)
  : N = 55.5 :=
sorry

end new_person_weight_l207_207245


namespace positive_integers_n_l207_207403

theorem positive_integers_n (n a b : ℕ) (h1 : 2 < n) (h2 : n = a ^ 3 + b ^ 3) 
  (h3 : ∀ d, d > 1 ∧ d ∣ n → a ≤ d) (h4 : b ∣ n) : n = 16 ∨ n = 72 ∨ n = 520 :=
sorry

end positive_integers_n_l207_207403


namespace find_b_plus_k_l207_207154

open Real

noncomputable def semi_major_axis (f1 f2 : ℝ × ℝ) (p : ℝ × ℝ) : ℝ :=
  dist p f1 + dist p f2

def c_squared (a : ℝ) (b : ℝ) : ℝ :=
  a ^ 2 - b ^ 2

theorem find_b_plus_k :
  ∀ (f1 f2 : ℝ × ℝ) (p : ℝ × ℝ) (h k : ℝ) (a b : ℝ),
  f1 = (-2, 0) →
  f2 = (2, 0) →
  p = (6, 0) →
  (∃ a b, semi_major_axis f1 f2 p = 2 * a ∧ c_squared a b = 4) →
  h = 0 →
  k = 0 →
  b = 4 * sqrt 2 →
  b + k = 4 * sqrt 2 :=
by
  intros f1 f2 p h k a b f1_def f2_def p_def maj_axis_def h_def k_def b_def
  rw [b_def, k_def]
  exact add_zero (4 * sqrt 2)

end find_b_plus_k_l207_207154


namespace max_abs_diff_f_l207_207045

noncomputable def f (x : ℝ) : ℝ := (x + 1) ^ 2 * Real.exp x

theorem max_abs_diff_f (k : ℝ) (h₁ : -3 ≤ k) (h₂ : k ≤ -1) (x₁ x₂ : ℝ) (h₃ : k ≤ x₁) (h₄ : x₁ ≤ k + 2) (h₅ : k ≤ x₂) (h₆ : x₂ ≤ k + 2) :
  |f x₁ - f x₂| ≤ 4 * Real.exp 1 := sorry

end max_abs_diff_f_l207_207045


namespace regular_polygon_sides_l207_207567

theorem regular_polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (160 : ℝ) = (180 * (n - 2)) / n) : n = 18 := 
by 
  sorry

end regular_polygon_sides_l207_207567


namespace denis_neighbors_l207_207114

-- Define positions
inductive Position
| P1 | P2 | P3 | P4 | P5

open Position

-- Declare the children
inductive Child
| Anya | Borya | Vera | Gena | Denis

open Child

def next_to (p1 p2 : Position) : Prop := 
  (p1 = P1 ∧ p2 = P2) ∨ (p1 = P2 ∧ p2 = P1) ∨
  (p1 = P2 ∧ p2 = P3) ∨ (p1 = P3 ∧ p2 = P2) ∨
  (p1 = P3 ∧ p2 = P4) ∨ (p1 = P4 ∧ p2 = P3) ∨
  (p1 = P4 ∧ p2 = P5) ∨ (p1 = P5 ∧ p2 = P4)

variables (pos : Child → Position)

-- Given conditions
axiom borya_beginning : pos Borya = P1
axiom vera_next_to_anya : next_to (pos Vera) (pos Anya)
axiom vera_not_next_to_gena : ¬ next_to (pos Vera) (pos Gena)
axiom no_two_next_to : ∀ (c1 c2 : Child), 
  c1 ∈ [Anya, Borya, Gena] → c2 ∈ [Anya, Borya, Gena] → c1 ≠ c2 → ¬ next_to (pos c1) (pos c2)

-- Prove the result
theorem denis_neighbors : next_to (pos Denis) (pos Anya) ∧ next_to (pos Denis) (pos Gena) :=
sorry

end denis_neighbors_l207_207114


namespace min_value_expr_l207_207741

theorem min_value_expr (x : ℝ) (h : x > -3) : ∃ m, (∀ y > -3, 2 * y + (1 / (y + 3)) ≥ m) ∧ m = 2 * Real.sqrt 2 - 6 :=
by
  sorry

end min_value_expr_l207_207741


namespace find_DG_l207_207630

theorem find_DG (a b k l : ℕ) (h1 : a * k = 37 * (a + b)) (h2 : b * l = 37 * (a + b)) : 
  k = 1406 :=
by
  sorry

end find_DG_l207_207630


namespace customer_can_receive_exact_change_l207_207853

theorem customer_can_receive_exact_change (k : ℕ) (hk : k ≤ 1000) :
  ∃ change : ℕ, change + k = 1000 ∧ change ≤ 1999 :=
by
  sorry

end customer_can_receive_exact_change_l207_207853


namespace geometric_series_common_ratio_l207_207981

theorem geometric_series_common_ratio (a r : ℝ) (h₁ : r ≠ 1)
    (h₂ : a / (1 - r) = 64 * (a * r^4) / (1 - r)) : r = 1/2 :=
by
  have h₃ : 1 = 64 * r^4 := by
    have : 1 - r ≠ 0 := by linarith
    field_simp at h₂; assumption
  sorry

end geometric_series_common_ratio_l207_207981


namespace find_number_l207_207508

theorem find_number (n : ℝ) (h : 1 / 2 * n + 7 = 17) : n = 20 :=
by
  sorry

end find_number_l207_207508


namespace chocolate_bar_cost_l207_207703

theorem chocolate_bar_cost 
  (x : ℝ)  -- cost of each bar in dollars
  (total_bars : ℕ)  -- total number of bars in the box
  (sold_bars : ℕ)  -- number of bars sold
  (amount_made : ℝ)  -- amount made in dollars
  (h1 : total_bars = 9)  -- condition: total bars in the box is 9
  (h2 : sold_bars = total_bars - 3)  -- condition: Wendy sold all but 3 bars
  (h3 : amount_made = 18)  -- condition: Wendy made $18
  (h4 : amount_made = sold_bars * x)  -- condition: amount made from selling sold bars
  : x = 3 := 
sorry

end chocolate_bar_cost_l207_207703


namespace irr_sqrt6_l207_207399

open Real

theorem irr_sqrt6 : ¬ ∃ (q : ℚ), (↑q : ℝ) = sqrt 6 := by
  sorry

end irr_sqrt6_l207_207399


namespace f_nested_seven_l207_207042

-- Definitions for the given conditions
variables (f : ℝ → ℝ) (odd_f : ∀ x, f (-x) = -f x)
variables (period_f : ∀ x, f (x + 4) = f x)
variables (f_one : f 1 = 4)

theorem f_nested_seven (f : ℝ → ℝ)
  (odd_f : ∀ x, f (-x) = -f x)
  (period_f : ∀ x, f (x + 4) = f x)
  (f_one : f 1 = 4) :
  f (f 7) = 0 :=
sorry

end f_nested_seven_l207_207042


namespace vector_dot_product_parallel_l207_207305

theorem vector_dot_product_parallel (m : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h_a : a = (1, 2))
  (h_b : b = (m, -4))
  (h_parallel : a.1 * b.2 = a.2 * b.1) :
  (a.1 * b.1 + a.2 * b.2) = -10 := by
  sorry

end vector_dot_product_parallel_l207_207305


namespace regular_polygon_sides_l207_207578

-- Define the problem conditions based on the given problem.
variables (n : ℕ)
def sum_of_interior_angles (n : ℕ) : ℤ := 180 * (n - 2)
def interior_angle (n : ℕ) : ℤ := 160
def total_interior_angle (n : ℕ) : ℤ := 160 * n

-- State the problem and expected result.
theorem regular_polygon_sides (h : 180 * (n - 2) = 160 * n) : n = 18 := 
by {
  -- This is to only state the theorem for now, proof will be crafted separately.
  sorry
}

end regular_polygon_sides_l207_207578


namespace fraction_of_plot_occupied_by_beds_l207_207936

-- Define the conditions based on plot area and number of beds
def plot_area : ℕ := 64
def total_beds : ℕ := 13
def outer_beds : ℕ := 12
def central_bed_area : ℕ := 4 * 4

-- The proof statement showing that fraction of the plot occupied by the beds is 15/32
theorem fraction_of_plot_occupied_by_beds : 
  (central_bed_area + (plot_area - central_bed_area)) / plot_area = 15 / 32 := 
sorry

end fraction_of_plot_occupied_by_beds_l207_207936


namespace has_local_maximum_l207_207344

noncomputable def func (x : ℝ) : ℝ := (1 / 3) * x ^ 3 - 4 * x + 4

theorem has_local_maximum :
  ∃ x, x = -2 ∧ func x = 28 / 3 :=
by
  sorry

end has_local_maximum_l207_207344


namespace length_of_qr_l207_207796

theorem length_of_qr (Q : ℝ) (PQ QR : ℝ) 
  (h1 : Real.sin Q = 0.6)
  (h2 : PQ = 15) :
  QR = 18.75 :=
by
  sorry

end length_of_qr_l207_207796


namespace num_divisors_not_divisible_by_2_of_360_l207_207053

def is_divisor (n d : ℕ) : Prop := d ∣ n

def is_prime (p : ℕ) : Prop := Nat.Prime p

noncomputable def prime_factors (n : ℕ) : List ℕ := sorry -- To be implemented if needed

def count_divisors_not_divisible_by_2 (n : ℕ) : ℕ :=
  let factors : List ℕ := prime_factors 360
  let a := 0
  let b_choices := [0, 1, 2]
  let c_choices := [0, 1]
  (b_choices.length) * (c_choices.length)

theorem num_divisors_not_divisible_by_2_of_360 :
  count_divisors_not_divisible_by_2 360 = 6 :=
by sorry

end num_divisors_not_divisible_by_2_of_360_l207_207053


namespace combined_mpg_is_30_l207_207628

-- Define the constants
def ray_efficiency : ℕ := 50 -- miles per gallon
def tom_efficiency : ℕ := 25 -- miles per gallon
def ray_distance : ℕ := 100 -- miles
def tom_distance : ℕ := 200 -- miles

-- Define the combined miles per gallon calculation and the proof statement.
theorem combined_mpg_is_30 :
  (ray_distance + tom_distance) /
  ((ray_distance / ray_efficiency) + (tom_distance / tom_efficiency)) = 30 :=
by
  -- All proof steps are skipped using sorry
  sorry

end combined_mpg_is_30_l207_207628


namespace two_f_one_lt_f_four_l207_207591

theorem two_f_one_lt_f_four
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (x + 2) = f (-x - 2))
  (h2 : ∀ x, x > 2 → x * (deriv f x) > 2 * (deriv f x) + f x) :
  2 * f 1 < f 4 :=
sorry

end two_f_one_lt_f_four_l207_207591


namespace smallest_x_l207_207661

theorem smallest_x (x : ℕ) (h : 450 * x % 648 = 0) : x = 36 := 
sorry

end smallest_x_l207_207661


namespace xy_range_l207_207176

theorem xy_range (x y : ℝ)
  (h1 : x + y = 1)
  (h2 : 1 / 3 ≤ x ∧ x ≤ 2 / 3) :
  2 / 9 ≤ x * y ∧ x * y ≤ 1 / 4 :=
sorry

end xy_range_l207_207176


namespace remainder_7459_div_9_l207_207841

theorem remainder_7459_div_9 : 7459 % 9 = 7 := 
by
  sorry

end remainder_7459_div_9_l207_207841


namespace probability_compare_l207_207359

-- Conditions
def v : ℝ := 0.1
def n : ℕ := 998

-- Binomial distribution formula
noncomputable def binom_prob (n k : ℕ) (v : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * (v ^ k) * ((1 - v) ^ (n - k))

-- Theorem to prove
theorem probability_compare :
  binom_prob n 99 v > binom_prob n 100 v :=
by
  sorry

end probability_compare_l207_207359


namespace cos_seven_pi_over_four_proof_l207_207007

def cos_seven_pi_over_four : Prop := (Real.cos (7 * Real.pi / 4) = 1 / Real.sqrt 2)

theorem cos_seven_pi_over_four_proof : cos_seven_pi_over_four :=
by
  sorry

end cos_seven_pi_over_four_proof_l207_207007


namespace scientific_notation_120_million_l207_207799

theorem scientific_notation_120_million :
  120000000 = 1.2 * 10^7 :=
by
  sorry

end scientific_notation_120_million_l207_207799


namespace fraction_equation_solution_l207_207354

theorem fraction_equation_solution (a : ℤ) (hpos : a > 0) (h : (a : ℝ) / (a + 50) = 0.870) : a = 335 :=
by {
  sorry
}

end fraction_equation_solution_l207_207354


namespace part_I_period_part_I_monotonicity_interval_part_II_range_l207_207301

noncomputable def f (x : ℝ) : ℝ :=
  4 * Real.cos x * Real.sin (x + Real.pi / 6) - 1

theorem part_I_period : ∀ x, f (x + Real.pi) = f x := by
  sorry

theorem part_I_monotonicity_interval (k : ℤ) :
  ∀ x, k * Real.pi + Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + 2 * Real.pi / 3 → f (x + Real.pi) = f x := by
  sorry

theorem part_II_range :
  ∀ x, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 4 → f x ∈ Set.Icc (-1) 2 := by
  sorry

end part_I_period_part_I_monotonicity_interval_part_II_range_l207_207301


namespace scheduling_competitions_l207_207016

-- Define the problem conditions
def scheduling_conditions (gyms : ℕ) (sports : ℕ) (max_sports_per_gym : ℕ) : Prop :=
  gyms = 4 ∧ sports = 3 ∧ max_sports_per_gym = 2

-- Define the main statement
theorem scheduling_competitions :
  scheduling_conditions 4 3 2 →
  (number_of_arrangements = 60) :=
by
  sorry

end scheduling_competitions_l207_207016


namespace cos_B_third_quadrant_l207_207902

theorem cos_B_third_quadrant (B : ℝ) (hB1 : π < B ∧ B < 3 * π / 2) (hB2 : sin B = -5 / 13) : cos B = -12 / 13 :=
by
  sorry

end cos_B_third_quadrant_l207_207902


namespace circle_through_origin_and_point_l207_207807

theorem circle_through_origin_and_point (a r : ℝ) :
  (∃ a r : ℝ, (a^2 + (5 - 3 * a)^2 = r^2) ∧ ((a - 3)^2 + (3 * a - 6)^2 = r^2)) →
  a = 5/3 ∧ r^2 = 25/9 :=
sorry

end circle_through_origin_and_point_l207_207807


namespace max_consecutive_integers_sum_lt_1000_l207_207997

theorem max_consecutive_integers_sum_lt_1000 :
  ∃ n : ℕ, (∀ m : ℕ, m ≤ n → m * (m + 1) / 2 < 1000) ∧ (n * (n + 1) / 2 < 1000) ∧ ¬((n + 1) * (n + 2) / 2 < 1000) :=
sorry

end max_consecutive_integers_sum_lt_1000_l207_207997


namespace probability_girl_selection_l207_207145

-- Define the conditions
def total_candidates : ℕ := 3 + 1
def girl_candidates : ℕ := 1

-- Define the question in terms of probability
def probability_of_selecting_girl (total: ℕ) (girl: ℕ) : ℚ :=
  girl / total

-- Lean statement to prove
theorem probability_girl_selection : probability_of_selecting_girl total_candidates girl_candidates = 1 / 4 :=
by
  sorry

end probability_girl_selection_l207_207145


namespace juice_cost_l207_207063

theorem juice_cost (J : ℝ) (h1 : 15 * 3 + 25 * 1 + 12 * J = 88) : J = 1.5 :=
by
  sorry

end juice_cost_l207_207063


namespace unique_triangle_solution_l207_207151

noncomputable def triangle_solutions (a b A : ℝ) : ℕ :=
sorry -- Placeholder for actual function calculating number of solutions

theorem unique_triangle_solution : triangle_solutions 30 25 150 = 1 :=
sorry -- Proof goes here

end unique_triangle_solution_l207_207151


namespace rotation_90_ccw_l207_207674

-- Define the complex number before the rotation
def initial_complex : ℂ := -4 - 2 * Complex.I

-- Define the resulting complex number after a 90-degree counter-clockwise    rotation
def result_complex : ℂ := 2 - 4 * Complex.I

-- State the theorem to be proved
theorem rotation_90_ccw (z : ℂ) (h : z = initial_complex) :
  Complex.I * z = result_complex :=
by sorry

end rotation_90_ccw_l207_207674


namespace lucia_hiphop_classes_l207_207338

def cost_hiphop_class : Int := 10
def cost_ballet_class : Int := 12
def cost_jazz_class : Int := 8
def num_ballet_classes : Int := 2
def num_jazz_classes : Int := 1
def total_cost : Int := 52

def num_hiphop_classes : Int := (total_cost - (num_ballet_classes * cost_ballet_class + num_jazz_classes * cost_jazz_class)) / cost_hiphop_class

theorem lucia_hiphop_classes : num_hiphop_classes = 2 := by
  sorry

end lucia_hiphop_classes_l207_207338


namespace age_proof_l207_207820

theorem age_proof (A B C D k m : ℕ)
  (h1 : A + B + C + D = 76)
  (h2 : A - 3 = k)
  (h3 : B - 3 = 2*k)
  (h4 : C - 3 = 3*k)
  (h5 : A - 5 = 3*m)
  (h6 : D - 5 = 4*m)
  (h7 : B - 5 = 5*m) :
  A = 11 := 
sorry

end age_proof_l207_207820


namespace fourth_intersection_point_exists_l207_207760

noncomputable def find_fourth_intersection_point : Prop :=
  let points := [(4, 1/2), (-6, -1/3), (1/4, 8), (-2/3, -3)]
  ∃ (h k r : ℝ), 
  ∀ (x y : ℝ), (x, y) ∈ points → (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2

theorem fourth_intersection_point_exists :
  find_fourth_intersection_point :=
by
  sorry

end fourth_intersection_point_exists_l207_207760


namespace root_of_equation_l207_207056

theorem root_of_equation : 
  ∀ (f : ℝ → ℝ), (∀ x : ℝ, f x = (x - 1) / x) →
  f (4 * (1 / 2)) = (1 / 2) :=
by
  sorry

end root_of_equation_l207_207056


namespace flour_maximum_weight_l207_207256

/-- Given that the bag of flour is marked with 25kg + 50g, prove that the maximum weight of the flour is 25.05kg. -/
theorem flour_maximum_weight :
  let weight_kg := 25
  let weight_g := 50
  (weight_kg + (weight_g / 1000 : ℝ)) = 25.05 :=
by 
  -- provide definitions
  let weight_kg := 25
  let weight_g := 50
  have : (weight_kg + (weight_g / 1000 : ℝ)) = 25.05 := sorry
  exact this

end flour_maximum_weight_l207_207256


namespace blocks_left_l207_207467

theorem blocks_left (initial_blocks used_blocks : ℕ) (h_initial : initial_blocks = 59) (h_used : used_blocks = 36) : initial_blocks - used_blocks = 23 :=
by
  -- proof here
  sorry

end blocks_left_l207_207467


namespace sqrt_23_range_l207_207279

theorem sqrt_23_range : 4.5 < Real.sqrt 23 ∧ Real.sqrt 23 < 5 := by
  sorry

end sqrt_23_range_l207_207279


namespace non_drinkers_count_l207_207690

-- Define the total number of businessmen and the sets of businessmen drinking each type of beverage.
def total_businessmen : ℕ := 30
def coffee_drinkers : ℕ := 15
def tea_drinkers : ℕ := 12
def soda_drinkers : ℕ := 8
def coffee_tea_drinkers : ℕ := 7
def tea_soda_drinkers : ℕ := 3
def coffee_soda_drinkers : ℕ := 2
def all_three_drinkers : ℕ := 1

-- Statement to prove:
theorem non_drinkers_count :
  total_businessmen - (coffee_drinkers + tea_drinkers + soda_drinkers - coffee_tea_drinkers - tea_soda_drinkers - coffee_soda_drinkers + all_three_drinkers) = 6 :=
by
  -- Skip the proof for now.
  sorry

end non_drinkers_count_l207_207690


namespace exercise_hobby_gender_independence_l207_207513

-- Define given contingency table details
def num_males_hobby := 30
def num_females_no_hobby := 10
def total_num_employees := 100

-- Define the full contingency table
def contingency_table := 
  ⟨50, 50, 
    { hobby := 70, no_hobby := 30, total := total_num_employees },
    { hobby :=  { males := num_males_hobby, females := 40 },
      no_hobby := { males := 20, females := num_females_no_hobby } }⟩

-- Define probability values
def P_X_0 := 3 / 29
def P_X_1 := 40 / 87
def P_X_2 := 38 / 87

-- Define distribution table
def distribution_table := [(0, P_X_0), (1, P_X_1), (2, P_X_2)]

-- Define expectation
def E_X : ℝ := 4 / 3

-- Theorem statement
theorem exercise_hobby_gender_independence :
  let χ2 := 4.76 in
  let critical_value := 6.635 in
  χ2 < critical_value →
  -- Completing the contingency table assertion
  contingency_table =
    ⟨50, 50, 
      { hobby := 70, no_hobby := 30, total := total_num_employees },
      { hobby :=  { males := num_males_hobby, females := 40 },
        no_hobby := { males := 20, females := num_females_no_hobby } }⟩ ∧
  -- Conclusion of independence test assertion
  ∀ α (h : α = 0.01), χ2 < critical_value ∧ 
  -- Distribution and expectation assertion
  distribution_table = [(0, P_X_0), (1, P_X_1), (2, P_X_2)] ∧ 
  E_X = 4 / 3 :=
sorry

end exercise_hobby_gender_independence_l207_207513


namespace heartbeats_during_race_l207_207152

theorem heartbeats_during_race 
  (heart_rate : ℕ) -- average heartbeats per minute
  (pace : ℕ) -- minutes per mile
  (distance : ℕ) -- distance in miles
  (heart_rate_avg : heart_rate = 160) -- condition (1)
  (pace_per_mile : pace = 6) -- condition (2)
  (race_distance : distance = 20) -- condition (3)
  : heart_rate * (pace * distance) = 19200 := 
by
  rw [heart_rate_avg, pace_per_mile, race_distance]
  exact eq.refl 19200

end heartbeats_during_race_l207_207152


namespace find_N_l207_207896

theorem find_N : ∀ N : ℕ, (991 + 993 + 995 + 997 + 999 = 5000 - N) → N = 25 :=
by
  intro N h
  sorry

end find_N_l207_207896


namespace max_consecutive_integers_sum_l207_207993

theorem max_consecutive_integers_sum (S_n : ℕ → ℕ) : (∀ n, S_n n = n * (n + 1) / 2) → ∀ n, (S_n n < 1000 ↔ n ≤ 44) :=
by
  intros H n
  split
  · intro H1
    have H2 : n * (n + 1) < 2000 := by
      rw [H n] at H1
      exact H1
    sorry
  · intro H1
    have H2 : n ≤ 44 := H1
    have H3 : n * (n + 1) < 2000 := by
      sorry
    have H4 : S_n n < 1000 := by
      rw [H n]
      exact H3
    exact H4

end max_consecutive_integers_sum_l207_207993


namespace correct_formulas_l207_207511

theorem correct_formulas (n : ℕ) :
  ((2 * n - 1)^2 - 4 * (n * (n - 1)) / 2) = (2 * n^2 - 2 * n + 1) ∧ 
  (1 + ((n - 1) * n) / 2 * 4) = (2 * n^2 - 2 * n + 1) ∧ 
  ((n - 1)^2 + n^2) = (2 * n^2 - 2 * n + 1) := by
  sorry

end correct_formulas_l207_207511


namespace two_trucks_carry_2_tons_l207_207265

theorem two_trucks_carry_2_tons :
  ∀ (truck_capacity : ℕ), truck_capacity = 999 →
  (truck_capacity * 2) / 1000 = 2 :=
by
  intros truck_capacity h_capacity
  rw [h_capacity]
  exact sorry

end two_trucks_carry_2_tons_l207_207265


namespace sally_found_more_balloons_l207_207632

def sally_original_balloons : ℝ := 9.0
def sally_new_balloons : ℝ := 11.0

theorem sally_found_more_balloons :
  sally_new_balloons - sally_original_balloons = 2.0 :=
by
  -- math proof goes here
  sorry

end sally_found_more_balloons_l207_207632


namespace line_passes_through_point_has_correct_equation_l207_207423

theorem line_passes_through_point_has_correct_equation :
  (∃ (L : ℝ × ℝ → Prop), (L (-2, 5)) ∧ (∃ m : ℝ, m = -3 / 4 ∧ ∀ (x y : ℝ), L (x, y) ↔ y - 5 = -3 / 4 * (x + 2))) →
  ∀ x y : ℝ, (3 * x + 4 * y - 14 = 0) ↔ (y - 5 = -3 / 4 * (x + 2)) :=
by
  intro h_L
  sorry

end line_passes_through_point_has_correct_equation_l207_207423


namespace sqrt_eq_sum_iff_l207_207422

open Real

theorem sqrt_eq_sum_iff (a b : ℝ) : sqrt (a^2 + b^2) = a + b ↔ (a * b = 0) ∧ (a + b ≥ 0) :=
by
  sorry

end sqrt_eq_sum_iff_l207_207422


namespace circle_eq_l207_207095

theorem circle_eq (D E : ℝ) :
  (∀ {x y : ℝ}, (x = 0 ∧ y = 0) ∨
               (x = 4 ∧ y = 0) ∨
               (x = -1 ∧ y = 1) → 
               x^2 + y^2 + D * x + E * y = 0) →
  (D = -4 ∧ E = -6) :=
by
  intros h
  have h1 : 0^2 + 0^2 + D * 0 + E * 0 = 0 := by exact h (Or.inl ⟨rfl, rfl⟩)
  have h2 : 4^2 + 0^2 + D * 4 + E * 0 = 0 := by exact h (Or.inr (Or.inl ⟨rfl, rfl⟩))
  have h3 : (-1)^2 + 1^2 + D * (-1) + E * 1 = 0 := by exact h (Or.inr (Or.inr ⟨rfl, rfl⟩))
  sorry -- proof steps would go here to eventually show D = -4 and E = -6

end circle_eq_l207_207095


namespace choir_females_correct_l207_207489

noncomputable def number_of_females_in_choir : ℕ :=
  let orchestra_males := 11
  let orchestra_females := 12
  let orchestra_musicians := orchestra_males + orchestra_females
  let band_males := 2 * orchestra_males
  let band_females := 2 * orchestra_females
  let band_musicians := 2 * orchestra_musicians
  let total_musicians := 98
  let choir_males := 12
  let choir_musicians := total_musicians - (orchestra_musicians + band_musicians)
  let choir_females := choir_musicians - choir_males
  choir_females

theorem choir_females_correct : number_of_females_in_choir = 17 := by
  sorry

end choir_females_correct_l207_207489


namespace quad_roots_expression_l207_207454

theorem quad_roots_expression (x1 x2 : ℝ) (h1 : x1 * x1 + 2019 * x1 + 1 = 0) (h2 : x2 * x2 + 2019 * x2 + 1 = 0) :
  x1 * x2 - x1 - x2 = 2020 :=
sorry

end quad_roots_expression_l207_207454


namespace fraction_of_groups_with_a_and_b_l207_207516

/- Definitions based on the conditions -/
def total_persons : ℕ := 6
def group_size : ℕ := 3
def person_a : ℕ := 1  -- arbitrary assignment for simplicity
def person_b : ℕ := 2  -- arbitrary assignment for simplicity

/- Hypotheses based on conditions -/
axiom six_persons (n : ℕ) : n = total_persons
axiom divided_into_two_groups (grp_size : ℕ) : grp_size = group_size
axiom a_and_b_included (a b : ℕ) : a = person_a ∧ b = person_b

/- The theorem to prove -/
theorem fraction_of_groups_with_a_and_b
    (total_groups : ℕ := Nat.choose total_persons group_size)
    (groups_with_a_b : ℕ := Nat.choose 4 1) :
    groups_with_a_b / total_groups = 1 / 5 :=
by
    sorry

end fraction_of_groups_with_a_and_b_l207_207516


namespace tens_digit_N_pow_20_is_7_hundreds_digit_N_pow_200_is_3_l207_207455

def tens_digit_N_pow_20 (N : ℕ) : Nat :=
if (N % 2 = 0 ∧ N % 10 ≠ 0) then
  if (N % 5 = 1 ∨ N % 5 = 2 ∨ N % 5 = 3 ∨ N % 5 = 4) then
    (N^20 % 100) / 10  -- tens digit of last two digits
  else
    sorry  -- N should be in form of 5k±1 or 5k±2
else
  sorry  -- N not satisfying conditions

def hundreds_digit_N_pow_200 (N : ℕ) : Nat :=
if (N % 2 = 0 ∧ N % 10 ≠ 0) then
  (N^200 % 1000) / 100  -- hundreds digit of the last three digits
else
  sorry  -- N not satisfying conditions

theorem tens_digit_N_pow_20_is_7 (N : ℕ) (h1 : N % 2 = 0) (h2 : N % 10 ≠ 0) : 
  tens_digit_N_pow_20 N = 7 := sorry

theorem hundreds_digit_N_pow_200_is_3 (N : ℕ) (h1 : N % 2 = 0) (h2 : N % 10 ≠ 0) : 
  hundreds_digit_N_pow_200 N = 3 := sorry

end tens_digit_N_pow_20_is_7_hundreds_digit_N_pow_200_is_3_l207_207455


namespace function_property_l207_207355

noncomputable def f (x : ℝ) : ℝ := sorry
variable (a x1 x2 : ℝ)

-- Conditions
axiom f_defined_on_R : ∀ x : ℝ, f x ≠ 0
axiom f_increasing_on_left_of_a : ∀ x y : ℝ, x < y → y < a → f x < f y
axiom f_even_shifted_by_a : ∀ x : ℝ, f (x + a) = f (-(x + a))
axiom ordering : x1 < a ∧ a < x2
axiom distance_comp : |x1 - a| < |x2 - a|

-- Proof Goal
theorem function_property : f (2 * a - x1) > f (2 * a - x2) :=
by
  sorry

end function_property_l207_207355


namespace difference_of_numbers_l207_207987

/-- Given two natural numbers a and 10a whose sum is 23,320,
prove that the difference between them is 19,080. -/
theorem difference_of_numbers (a : ℕ) (h : a + 10 * a = 23320) : 10 * a - a = 19080 := by
  sorry

end difference_of_numbers_l207_207987


namespace area_of_sector_one_radian_l207_207803

theorem area_of_sector_one_radian (r θ : ℝ) (hθ : θ = 1) (hr : r = 1) : 
  (1/2 * (r * θ) * r) = 1/2 :=
by
  sorry

end area_of_sector_one_radian_l207_207803


namespace minimum_time_to_replace_shades_l207_207752

theorem minimum_time_to_replace_shades :
  ∀ (C : ℕ) (S : ℕ) (T : ℕ) (E : ℕ),
  ((C = 60) ∧ (S = 4) ∧ (T = 5) ∧ (E = 48)) →
  ((C * S * T) / E = 25) :=
by
  intros C S T E h
  rcases h with ⟨hC, hS, hT, hE⟩
  sorry

end minimum_time_to_replace_shades_l207_207752


namespace relationship_between_line_and_circle_l207_207294

variables {a b r : ℝ} (M : ℝ × ℝ) (l m : ℝ → ℝ)

def point_inside_circle_not_on_axes 
    (M : ℝ × ℝ) (r : ℝ) : Prop := 
    (M.fst^2 + M.snd^2 < r^2) ∧ (M.fst ≠ 0) ∧ (M.snd ≠ 0)

def line_eq (a b r : ℝ) (x y : ℝ) : Prop := 
    a * x + b * y = r^2

def chord_midpoint (M : ℝ × ℝ) (m : ℝ → ℝ) : Prop := 
    ∃ x1 y1 x2 y2, 
    (M.fst = (x1 + x2) / 2 ∧ M.snd = (y1 + y2) / 2) ∧ 
    (m x1 = y1 ∧ m x2 = y2)

def circle_external (O : ℝ → ℝ) (l : ℝ → ℝ) : Prop := 
    ∀ x y, O x = y → l x ≠ y

theorem relationship_between_line_and_circle
    (M_inside : point_inside_circle_not_on_axes M r)
    (M_chord : chord_midpoint M m)
    (line_eq_l : line_eq a b r M.fst M.snd) :
    (m (M.fst) = - (a / b) * M.snd) ∧ 
    (∀ x, l x ≠ m x) :=
sorry

end relationship_between_line_and_circle_l207_207294


namespace regular_polygon_sides_l207_207561

theorem regular_polygon_sides (n : ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → (interior_angle i = 160)) : n = 18 :=
by
  -- Proof goes here
  sorry

end regular_polygon_sides_l207_207561


namespace convex_cyclic_quadrilaterals_perimeter_40_l207_207435

theorem convex_cyclic_quadrilaterals_perimeter_40 :
  ∃ (n : ℕ), n = 750 ∧ ∀ (a b c d : ℕ), a + b + c + d = 40 → a ≥ b → b ≥ c → c ≥ d →
  (a < b + c + d) ∧ (b < a + c + d) ∧ (c < a + b + d) ∧ (d < a + b + c) :=
sorry

end convex_cyclic_quadrilaterals_perimeter_40_l207_207435


namespace one_third_pow_3_eq_3_pow_nineteen_l207_207193

theorem one_third_pow_3_eq_3_pow_nineteen (y : ℤ) (h : (1 / 3 : ℝ) * (3 ^ 20) = 3 ^ y) : y = 19 :=
by
  sorry

end one_third_pow_3_eq_3_pow_nineteen_l207_207193


namespace minimum_value_l207_207739

theorem minimum_value (x : ℝ) (h : x > -3) : 2 * x + (1 / (x + 3)) ≥ 2 * Real.sqrt 2 - 6 :=
sorry

end minimum_value_l207_207739


namespace geometric_series_ratio_half_l207_207960

theorem geometric_series_ratio_half (a r S : ℝ) (hS : S = a / (1 - r)) 
  (h_ratio : (ar^4) / (1 - r) = S / 64) : r = 1 / 2 :=
by
  sorry

end geometric_series_ratio_half_l207_207960


namespace bike_average_speed_l207_207381

theorem bike_average_speed (distance time : ℕ)
    (h1 : distance = 48)
    (h2 : time = 6) :
    distance / time = 8 := 
  by
    sorry

end bike_average_speed_l207_207381


namespace value_of_f_is_negative_l207_207300

theorem value_of_f_is_negative {a b c : ℝ} (h1 : a + b < 0) (h2 : b + c < 0) (h3 : c + a < 0) :
  2 * a ^ 3 + 4 * a + 2 * b ^ 3 + 4 * b + 2 * c ^ 3 + 4 * c < 0 := by
sorry

end value_of_f_is_negative_l207_207300


namespace probability_two_cards_diff_suits_l207_207774

def prob_two_cards_diff_suits {deck_size suits cards_per_suit : ℕ} (h1 : deck_size = 40) (h2 : suits = 4) (h3 : cards_per_suit = 10) : ℚ :=
  let total_cards := deck_size
  let cards_same_suit := cards_per_suit - 1
  let cards_diff_suit := total_cards - 1 - cards_same_suit 
  cards_diff_suit / (total_cards - 1)

theorem probability_two_cards_diff_suits (h1 : 40 = 40) (h2 : 4 = 4) (h3 : 10 = 10) :
  prob_two_cards_diff_suits h1 h2 h3 = 10 / 13 :=
by
  sorry

end probability_two_cards_diff_suits_l207_207774


namespace plane_equation_l207_207390

def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ :=
  (3 + s + 2 * t, 4 - 2 * s, 1 - s + t)

def normal_vector : ℝ × ℝ × ℝ :=
  (-2, -3, 4)

def point_on_plane : ℝ × ℝ × ℝ :=
  (3, 4, 1)

theorem plane_equation : ∀ (x y z : ℝ),
  (∃ (s t : ℝ), (x, y, z) = parametric_plane s t) ↔
  2 * x + 3 * y - 4 * z - 14 = 0 :=
sorry

end plane_equation_l207_207390


namespace floor_neg_sqrt_eval_l207_207022

theorem floor_neg_sqrt_eval :
  ⌊-(Real.sqrt (64 / 9))⌋ = -3 :=
by
  sorry

end floor_neg_sqrt_eval_l207_207022


namespace time_to_cross_signal_post_l207_207673

def train_length := 600 -- in meters
def bridge_length := 5400 -- in meters (5.4 kilometers)
def crossing_time_bridge := 6 * 60 -- in seconds (6 minutes)
def speed := bridge_length / crossing_time_bridge -- in meters per second

theorem time_to_cross_signal_post : 
  (600 / speed) = 40 :=
by
  sorry

end time_to_cross_signal_post_l207_207673


namespace who_is_next_to_denis_l207_207109

-- Define the five positions
inductive Pos : Type
| p1 | p2 | p3 | p4 | p5
open Pos

-- Define the people
inductive Person : Type
| Anya | Borya | Vera | Gena | Denis
open Person

-- Define the conditions as predicates
def isAtStart (p : Person) : Prop := p = Borya

def areNextTo (p1 p2 : Person) (posMap : Person → Pos) : Prop := 
  match posMap p1, posMap p2 with
  | p1, p2 => (abs (p1.to_nat - p2.to_nat)) = 1

def notNextTo (p1 p2 : Person) (posMap : Person → Pos) : Prop :=
  match posMap p1, posMap p2 with
  | p1, p2 => (abs (p1.to_nat - p2.to_nat)) ≠ 1

-- Define the adjacency conditions
def conditions (posMap : Person → Pos) :=
  isAtStart (posMap Borya) ∧
  areNextTo Vera Anya posMap ∧
  notNextTo Vera Gena posMap ∧
  notNextTo Anya Borya posMap ∧
  notNextTo Anya Gena posMap ∧
  notNextTo Borya Gena posMap

-- The final goal: proving Denis is next to Anya and Gena
theorem who_is_next_to_denis (posMap : Person → Pos)
  (h : conditions posMap) :
  areNextTo Denis Anya posMap ∧ areNextTo Denis Gena posMap :=
sorry

end who_is_next_to_denis_l207_207109


namespace overall_average_runs_l207_207606

theorem overall_average_runs 
  (test_matches: ℕ) (test_avg: ℕ) 
  (odi_matches: ℕ) (odi_avg: ℕ) 
  (t20_matches: ℕ) (t20_avg: ℕ)
  (h_test_matches: test_matches = 25)
  (h_test_avg: test_avg = 48)
  (h_odi_matches: odi_matches = 20)
  (h_odi_avg: odi_avg = 38)
  (h_t20_matches: t20_matches = 15)
  (h_t20_avg: t20_avg = 28) :
  (25 * 48 + 20 * 38 + 15 * 28) / (25 + 20 + 15) = 39.67 :=
sorry

end overall_average_runs_l207_207606


namespace find_a_l207_207592

theorem find_a (a x : ℝ) (h1 : 3 * a - x = x / 2 + 3) (h2 : x = 2) : a = 2 := 
by
  sorry

end find_a_l207_207592


namespace speed_of_goods_train_l207_207259

theorem speed_of_goods_train
  (length_train : ℕ)
  (length_platform : ℕ)
  (time_crossing : ℕ)
  (h_length_train : length_train = 240)
  (h_length_platform : length_platform = 280)
  (h_time_crossing : time_crossing = 26)
  : (length_train + length_platform) / time_crossing * (3600 / 1000) = 72 := 
by sorry

end speed_of_goods_train_l207_207259


namespace alyssa_kittens_l207_207528

theorem alyssa_kittens (original_kittens given_away: ℕ) (h1: original_kittens = 8) (h2: given_away = 4) :
  original_kittens - given_away = 4 :=
by
  sorry

end alyssa_kittens_l207_207528


namespace simplify_and_evaluate_expression_l207_207087

theorem simplify_and_evaluate_expression (x y : ℝ) (h1 : x = 1/2) (h2 : y = -2) :
  ((x + 2 * y) ^ 2 - (x + y) * (x - y)) / (2 * y) = -4 := by
  sorry

end simplify_and_evaluate_expression_l207_207087


namespace student_score_variance_l207_207680

noncomputable def variance_student_score : ℝ :=
  let number_of_questions := 25
  let probability_correct := 0.8
  let score_correct := 4
  let variance_eta := number_of_questions * probability_correct * (1 - probability_correct)
  let variance_xi := (score_correct ^ 2) * variance_eta
  variance_xi

theorem student_score_variance : variance_student_score = 64 := by
  sorry

end student_score_variance_l207_207680


namespace regular_polygon_sides_l207_207576

-- Define the problem conditions based on the given problem.
variables (n : ℕ)
def sum_of_interior_angles (n : ℕ) : ℤ := 180 * (n - 2)
def interior_angle (n : ℕ) : ℤ := 160
def total_interior_angle (n : ℕ) : ℤ := 160 * n

-- State the problem and expected result.
theorem regular_polygon_sides (h : 180 * (n - 2) = 160 * n) : n = 18 := 
by {
  -- This is to only state the theorem for now, proof will be crafted separately.
  sorry
}

end regular_polygon_sides_l207_207576


namespace myOperation_identity_l207_207285

variable {R : Type*} [LinearOrderedField R]

def myOperation (a b : R) : R := (a - b) ^ 2

theorem myOperation_identity (x y : R) : myOperation ((x - y) ^ 2) ((y - x) ^ 2) = 0 := 
by 
  sorry

end myOperation_identity_l207_207285


namespace explicit_expression_for_f_l207_207431

variable (f : ℕ → ℕ)

-- Define the condition
axiom h : ∀ x : ℕ, f (x + 1) = 3 * x + 2

-- State the theorem
theorem explicit_expression_for_f (x : ℕ) : f x = 3 * x - 1 :=
by {
  sorry
}

end explicit_expression_for_f_l207_207431


namespace find_f2_l207_207619

noncomputable def f (x a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 0) : f 2 a b = -16 :=
by {
  sorry
}

end find_f2_l207_207619


namespace value_of_expression_l207_207601

theorem value_of_expression (m n : ℝ) (h : m + n = 4) : 2 * m^2 + 4 * m * n + 2 * n^2 - 5 = 27 :=
  sorry

end value_of_expression_l207_207601


namespace find_b_l207_207945

theorem find_b (b : ℕ) (h1 : 0 ≤ b) (h2 : b ≤ 20) (h3 : (746392847 - b) % 17 = 0) : b = 16 :=
sorry

end find_b_l207_207945


namespace parallel_lines_implies_a_eq_one_l207_207888

theorem parallel_lines_implies_a_eq_one 
(h_parallel: ∀ (a : ℝ), ∀ (x y : ℝ), (x + a * y = 2 * a + 2) → (a * x + y = a + 1) → -1/a = -a) :
  ∀ (a : ℝ), a = 1 := by
  sorry

end parallel_lines_implies_a_eq_one_l207_207888


namespace ratio_of_black_to_white_after_border_l207_207392

def original_tiles (black white : ℕ) : Prop := black = 14 ∧ white = 21
def original_dimensions (length width : ℕ) : Prop := length = 5 ∧ width = 7

def border_added (length width l w : ℕ) : Prop := l = length + 2 ∧ w = width + 2

def total_white_tiles (initial_white new_white total_white : ℕ) : Prop :=
  total_white = initial_white + new_white

def black_white_ratio (black_tiles white_tiles : ℕ) (ratio : ℚ) : Prop :=
  ratio = black_tiles / white_tiles

theorem ratio_of_black_to_white_after_border 
  (black_white_tiles : ℕ → ℕ → Prop)
  (dimensions : ℕ → ℕ → Prop)
  (border : ℕ → ℕ → ℕ → ℕ → Prop)
  (total_white : ℕ → ℕ → ℕ → Prop)
  (ratio : ℕ → ℕ → ℚ → Prop)
  (black_tiles white_tiles initial_white total_white_new length width l w : ℕ)
  (rat : ℚ) :
  black_white_tiles black_tiles initial_white →
  dimensions length width →
  border length width l w →
  total_white initial_white (l * w - length * width) white_tiles →
  ratio black_tiles white_tiles rat →
  rat = 2 / 7 :=
by
  intros
  sorry

end ratio_of_black_to_white_after_border_l207_207392


namespace lena_more_candy_bars_than_nicole_l207_207073

theorem lena_more_candy_bars_than_nicole
  (Lena Kevin Nicole : ℕ)
  (h1 : Lena = 16)
  (h2 : Lena + 5 = 3 * Kevin)
  (h3 : Kevin + 4 = Nicole) :
  Lena - Nicole = 5 :=
by
  sorry

end lena_more_candy_bars_than_nicole_l207_207073


namespace James_beat_record_by_72_l207_207208

-- Define the conditions as given in the problem
def touchdowns_per_game : ℕ := 4
def points_per_touchdown : ℕ := 6
def games_in_season : ℕ := 15
def conversions : ℕ := 6
def points_per_conversion : ℕ := 2
def old_record : ℕ := 300

-- Define the necessary calculations based on the conditions
def points_from_touchdowns_per_game : ℕ := touchdowns_per_game * points_per_touchdown
def points_from_touchdowns_in_season : ℕ := games_in_season * points_from_touchdowns_per_game
def points_from_conversions : ℕ := conversions * points_per_conversion
def total_points_in_season : ℕ := points_from_touchdowns_in_season + points_from_conversions
def points_above_old_record : ℕ := total_points_in_season - old_record

-- State the proof problem
theorem James_beat_record_by_72 : points_above_old_record = 72 :=
by
  sorry

end James_beat_record_by_72_l207_207208


namespace not_possible_cut_l207_207449

theorem not_possible_cut (n : ℕ) : 
  let chessboard_area := 8 * 8
  let rectangle_area := 3
  let rectangles_needed := chessboard_area / rectangle_area
  rectangles_needed ≠ n :=
by
  sorry

end not_possible_cut_l207_207449


namespace adam_total_cost_l207_207150

theorem adam_total_cost 
    (sandwiches_count : ℕ)
    (sandwiches_price : ℝ)
    (chips_count : ℕ)
    (chips_price : ℝ)
    (water_count : ℕ)
    (water_price : ℝ)
    (sandwich_discount : sandwiches_count = 4 ∧ sandwiches_price = 4 ∧ sandwiches_count = 3 + 1)
    (tax_rate : ℝ)
    (initial_tax_rate : tax_rate = 0.10)
    (chips_cost : chips_count = 3 ∧ chips_price = 3.50)
    (water_cost : water_count = 2 ∧ water_price = 2) : 
  (3 * sandwiches_price + chips_count * chips_price + water_count * water_price) * (1 + tax_rate) = 29.15 := 
by
  sorry

end adam_total_cost_l207_207150


namespace negation_of_universal_proposition_l207_207357

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0 :=
by
  sorry

end negation_of_universal_proposition_l207_207357


namespace number_of_cakes_sold_l207_207272

-- Definitions based on the conditions provided
def cakes_made : ℕ := 173
def cakes_bought : ℕ := 103
def cakes_left : ℕ := 190

-- Calculate the initial total number of cakes
def initial_cakes : ℕ := cakes_made + cakes_bought

-- Calculate the number of cakes sold
def cakes_sold : ℕ := initial_cakes - cakes_left

-- The proof statement
theorem number_of_cakes_sold : cakes_sold = 86 :=
by
  unfold cakes_sold initial_cakes cakes_left cakes_bought cakes_made
  rfl

end number_of_cakes_sold_l207_207272


namespace rope_purchases_l207_207329

theorem rope_purchases (last_week_rope_feet : ℕ) (less_rope : ℕ) (feet_to_inches : ℕ) 
  (h1 : last_week_rope_feet = 6) 
  (h2 : less_rope = 4) 
  (h3 : feet_to_inches = 12) : 
  (last_week_rope_feet * feet_to_inches) + ((last_week_rope_feet - less_rope) * feet_to_inches) = 96 := 
by
  sorry

end rope_purchases_l207_207329


namespace carpet_dimensions_problem_l207_207395

def carpet_dimensions (width1 width2 : ℕ) (l : ℕ) :=
  ∃ x y : ℕ, width1 = 38 ∧ width2 = 50 ∧ l = l ∧ x = 25 ∧ y = 50

theorem carpet_dimensions_problem (l : ℕ) :
  carpet_dimensions 38 50 l :=
by
  sorry

end carpet_dimensions_problem_l207_207395


namespace min_scalar_product_l207_207751

open Real

variable {a b : ℝ → ℝ}

-- Definitions used as conditions in the problem
def condition (a b : ℝ → ℝ) : Prop :=
  |2 * a - b| ≤ 3

-- The goal to prove based on the conditions and the correct answer
theorem min_scalar_product (h : condition a b) : 
  (a x) * (b x) ≥ -9 / 8 :=
sorry

end min_scalar_product_l207_207751


namespace abs_neg_two_equals_two_l207_207693

theorem abs_neg_two_equals_two : |(-2 : ℤ)| = 2 := 
by 
  sorry

end abs_neg_two_equals_two_l207_207693


namespace smallest_number_to_add_l207_207494

theorem smallest_number_to_add:
  ∃ x : ℕ, x = 119 ∧ (2714 + x) % 169 = 0 :=
by
  sorry

end smallest_number_to_add_l207_207494


namespace probability_king_ace_correct_l207_207122

noncomputable def probability_king_ace : ℚ :=
  4 / 663

theorem probability_king_ace_correct :
  ∀ (deck : Finset ℕ), deck.card = 52 →
  (∃ (f : ℕ → Finset ℕ) , (f 1).card = 51 ∧ (f 2).card = 50 ∧ (f 51).card = 1) →
  (∃ (g : ℕ → ℕ), g 1 = 4 ∧ g 2 = 3) →
  probability_king_ace = 4 / 663 := by
  intros deck hdeck hfex hgex
  sorry

end probability_king_ace_correct_l207_207122


namespace problem_1_l207_207374

theorem problem_1
  (α : ℝ)
  (h : Real.tan α = -1/2) :
  1 / (Real.sin α ^ 2 - Real.sin α * Real.cos α - 2 * Real.cos α ^ 2) = -1 := 
sorry

end problem_1_l207_207374


namespace pencil_eraser_cost_l207_207767

theorem pencil_eraser_cost (p e : ℕ) (h_eq : 10 * p + 4 * e = 120) (h_gt : p > e) : p + e = 15 :=
by sorry

end pencil_eraser_cost_l207_207767


namespace speed_of_stream_l207_207818

theorem speed_of_stream 
  (v : ℝ)
  (boat_speed : ℝ)
  (distance_downstream : ℝ)
  (distance_upstream : ℝ)
  (H1 : boat_speed = 12)
  (H2 : distance_downstream = 32)
  (H3 : distance_upstream = 16)
  (H4 : distance_downstream / (boat_speed + v) = distance_upstream / (boat_speed - v)) :
  v = 4 :=
by
  sorry

end speed_of_stream_l207_207818


namespace lena_nicole_candy_difference_l207_207072

variables (L K N : ℕ)

theorem lena_nicole_candy_difference
  (hL : L = 16)
  (hLK : L + 5 = 3 * K)
  (hKN : K = N - 4) :
  L - N = 5 :=
sorry

end lena_nicole_candy_difference_l207_207072


namespace part1_part2_l207_207050

-- Definition of Set A
def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 6 }

-- Definition of Set B
def B : Set ℝ := { x | x ≥ 3 }

-- The Complement of the Intersection of A and B
def C_R (S : Set ℝ) : Set ℝ := { x | ¬ (x ∈ S) }

-- Set C
def C (a : ℝ) : Set ℝ := { x | x ≤ a }

-- Lean statement for part 1
theorem part1 : C_R (A ∩ B) = { x | x < 3 ∨ x > 6 } :=
by sorry

-- Lean statement for part 2
theorem part2 (a : ℝ) (hA_C : A ⊆ C a) : a ≥ 6 :=
by sorry

end part1_part2_l207_207050


namespace positive_integer_condition_l207_207254

theorem positive_integer_condition (n : ℕ) (h : 15 * n = n^2 + 56) : n = 8 :=
sorry

end positive_integer_condition_l207_207254


namespace total_students_l207_207931

theorem total_students (S : ℕ) (H1 : S / 2 = S - 15) : S = 30 :=
sorry

end total_students_l207_207931


namespace exists_subset_no_three_ap_l207_207333

-- Define the set S_n
def S (n : ℕ) : Finset ℕ := (Finset.range ((3^n + 1) / 2 + 1)).image (λ i => i + 1)

-- Define the property of no three elements forming an arithmetic progression
def no_three_form_ap (M : Finset ℕ) : Prop :=
  ∀ a b c, a ∈ M → b ∈ M → c ∈ M → a < b → b < c → 2 * b ≠ a + c

-- Define the theorem statement
theorem exists_subset_no_three_ap (n : ℕ) :
  ∃ M : Finset ℕ, M ⊆ S n ∧ M.card = 2^n ∧ no_three_form_ap M :=
sorry

end exists_subset_no_three_ap_l207_207333


namespace range_of_values_for_a_l207_207596

theorem range_of_values_for_a 
  (f : ℝ → ℝ)
  (h1 : ∀ x > 0, f x = x - 1/x - a * Real.log x)
  (h2 : ∀ x > 0, (x^2 - a * x + 1) ≥ 0) : 
  a ≤ 2 :=
sorry

end range_of_values_for_a_l207_207596


namespace extremely_powerful_count_l207_207275

def is_extremely_powerful (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ b % 2 = 1 ∧ a^b = n

noncomputable def count_extremely_powerful_below (m : ℕ) : ℕ :=
  Nat.card { n : ℕ | is_extremely_powerful n ∧ n < m }

theorem extremely_powerful_count : count_extremely_powerful_below 5000 = 19 :=
by
  sorry

end extremely_powerful_count_l207_207275


namespace value_of_c_l207_207771

theorem value_of_c (a b c : ℕ) (hab : b = 1) (hd : a ≠ b ∧ a ≠ c ∧ b ≠ c) (h_pow : (10 * a + b)^2 = 100 * c + 10 * c + b) (h_gt : 100 * c + 10 * c + b > 300) : 
  c = 4 :=
sorry

end value_of_c_l207_207771


namespace teacher_total_score_l207_207149

/-- Conditions -/
def written_test_score : ℝ := 80
def interview_score : ℝ := 60
def written_test_weight : ℝ := 0.6
def interview_weight : ℝ := 0.4

/-- Prove the total score -/
theorem teacher_total_score :
  written_test_score * written_test_weight + interview_score * interview_weight = 72 :=
by
  sorry

end teacher_total_score_l207_207149


namespace petya_time_l207_207537

variable (a V : ℝ)

noncomputable def planned_time := a / V
noncomputable def real_time := (a / (2.5 * V)) + (a / (1.6 * V))

theorem petya_time (hV : V > 0) (ha : a > 0) : real_time a V > planned_time a V :=
by
  sorry

end petya_time_l207_207537


namespace nancy_hourly_wage_l207_207779

theorem nancy_hourly_wage 
  (tuition_per_semester : ℕ := 22000) 
  (parents_cover : ℕ := tuition_per_semester / 2) 
  (scholarship : ℕ := 3000) 
  (student_loan : ℕ := 2 * scholarship) 
  (work_hours : ℕ := 200) 
  (remaining_tuition : ℕ := parents_cover - scholarship - student_loan) :
  (remaining_tuition / work_hours = 10) :=
  by
  sorry

end nancy_hourly_wage_l207_207779


namespace smallest_gcd_bc_l207_207311

theorem smallest_gcd_bc (a b c : ℕ) (h1 : Nat.gcd a b = 240) (h2 : Nat.gcd a c = 1001) : Nat.gcd b c = 1 :=
sorry

end smallest_gcd_bc_l207_207311


namespace who_is_next_to_denis_l207_207110

-- Define the five positions
inductive Pos : Type
| p1 | p2 | p3 | p4 | p5
open Pos

-- Define the people
inductive Person : Type
| Anya | Borya | Vera | Gena | Denis
open Person

-- Define the conditions as predicates
def isAtStart (p : Person) : Prop := p = Borya

def areNextTo (p1 p2 : Person) (posMap : Person → Pos) : Prop := 
  match posMap p1, posMap p2 with
  | p1, p2 => (abs (p1.to_nat - p2.to_nat)) = 1

def notNextTo (p1 p2 : Person) (posMap : Person → Pos) : Prop :=
  match posMap p1, posMap p2 with
  | p1, p2 => (abs (p1.to_nat - p2.to_nat)) ≠ 1

-- Define the adjacency conditions
def conditions (posMap : Person → Pos) :=
  isAtStart (posMap Borya) ∧
  areNextTo Vera Anya posMap ∧
  notNextTo Vera Gena posMap ∧
  notNextTo Anya Borya posMap ∧
  notNextTo Anya Gena posMap ∧
  notNextTo Borya Gena posMap

-- The final goal: proving Denis is next to Anya and Gena
theorem who_is_next_to_denis (posMap : Person → Pos)
  (h : conditions posMap) :
  areNextTo Denis Anya posMap ∧ areNextTo Denis Gena posMap :=
sorry

end who_is_next_to_denis_l207_207110


namespace union_complement_A_B_eq_U_l207_207732

-- Define the universal set U, set A, and set B
def U : Set ℕ := {1, 2, 3, 4, 5, 7}
def A : Set ℕ := {4, 7}
def B : Set ℕ := {1, 3, 4, 7}

-- Define the complement of A with respect to U (C_U A)
def C_U_A : Set ℕ := U \ A
-- Define the complement of B with respect to U (C_U B)
def C_U_B : Set ℕ := U \ B

-- The theorem to prove
theorem union_complement_A_B_eq_U : (C_U_A ∪ B) = U := by
  sorry

end union_complement_A_B_eq_U_l207_207732


namespace independence_equivalence_l207_207505

theorem independence_equivalence (A B : Set Ω) (P : Measure Ω) :
  (P(A|B) = P(A|Bᶜ) ∨ P(A) = P(A|B)) →
  P(A ∩ B) = P(A) * P(B) :=
  sorry

end independence_equivalence_l207_207505


namespace real_root_exists_l207_207404

theorem real_root_exists (a : ℝ) : 
    (∃ x : ℝ, x^4 - a * x^3 - x^2 - a * x + 1 = 0) ↔ (-1 / 2 ≤ a) := by
  sorry

end real_root_exists_l207_207404


namespace bad_carrots_count_l207_207401

-- Define the number of carrots each person picked and the number of good carrots
def carol_picked := 29
def mom_picked := 16
def good_carrots := 38

-- Define the total number of carrots picked and the total number of bad carrots
def total_carrots := carol_picked + mom_picked
def bad_carrots := total_carrots - good_carrots

-- State the theorem that the number of bad carrots is 7
theorem bad_carrots_count :
  bad_carrots = 7 :=
by
  sorry

end bad_carrots_count_l207_207401


namespace determine_lunch_break_duration_lunch_break_duration_in_minutes_l207_207765

noncomputable def painter_lunch_break_duration (j h L : ℝ) : Prop :=
  (10 - L) * (j + h) = 0.6 ∧
  (8 - L) * h = 0.3 ∧
  (5 - L) * j = 0.1

theorem determine_lunch_break_duration (j h : ℝ) :
  ∃ L : ℝ, painter_lunch_break_duration j h L ∧ L = 0.8 :=
by sorry

theorem lunch_break_duration_in_minutes (j h : ℝ) :
  ∃ L : ℝ, painter_lunch_break_duration j h L ∧ L * 60 = 48 :=
by sorry

end determine_lunch_break_duration_lunch_break_duration_in_minutes_l207_207765


namespace y_intercept_of_line_l207_207646

theorem y_intercept_of_line (m : ℝ) (x₀ y₀ : ℝ) (h₁ : m = -3) (h₂ : x₀ = 7) (h₃ : y₀ = 0) :
  ∃ (b : ℝ), (0, b) = (0, 21) :=
by
  -- Our goal is to prove the y-intercept is (0, 21)
  sorry

end y_intercept_of_line_l207_207646


namespace scientific_notation_of_3300000_l207_207762

theorem scientific_notation_of_3300000 : 3300000 = 3.3 * 10^6 :=
by
  sorry

end scientific_notation_of_3300000_l207_207762


namespace earnings_per_puppy_l207_207277

def daily_pay : ℝ := 40
def total_earnings : ℝ := 76
def num_puppies : ℕ := 16

theorem earnings_per_puppy : (total_earnings - daily_pay) / num_puppies = 2.25 := by
  sorry

end earnings_per_puppy_l207_207277


namespace price_of_orange_l207_207239

-- Define relevant conditions
def price_apple : ℝ := 1.50
def morning_apples : ℕ := 40
def morning_oranges : ℕ := 30
def afternoon_apples : ℕ := 50
def afternoon_oranges : ℕ := 40
def total_sales : ℝ := 205

-- Define the proof problem
theorem price_of_orange (O : ℝ) 
  (h : (morning_apples * price_apple + morning_oranges * O) + 
       (afternoon_apples * price_apple + afternoon_oranges * O) = total_sales) : 
  O = 1 :=
by
  sorry

end price_of_orange_l207_207239


namespace total_cost_l207_207243

noncomputable def cost_sandwich : ℝ := 2.44
noncomputable def quantity_sandwich : ℕ := 2
noncomputable def cost_soda : ℝ := 0.87
noncomputable def quantity_soda : ℕ := 4

noncomputable def total_cost_sandwiches : ℝ := cost_sandwich * quantity_sandwich
noncomputable def total_cost_sodas : ℝ := cost_soda * quantity_soda

theorem total_cost (total_cost_sandwiches total_cost_sodas : ℝ) : (total_cost_sandwiches + total_cost_sodas = 8.36) :=
by
  sorry

end total_cost_l207_207243


namespace find_x_average_is_60_l207_207225

theorem find_x_average_is_60 : 
  ∃ x : ℕ, (54 + 55 + 57 + 58 + 59 + 62 + 62 + 63 + x) / 9 = 60 ∧ x = 70 :=
by
  existsi 70
  sorry

end find_x_average_is_60_l207_207225


namespace simplify_fractions_l207_207469

-- Define the fractions and their product.
def fraction1 : ℚ := 14 / 3
def fraction2 : ℚ := 9 / -42

-- Define the product of the fractions with scalar multiplication by 5.
def product : ℚ := 5 * fraction1 * fraction2

-- The target theorem to prove the equivalence.
theorem simplify_fractions : product = -5 := 
sorry  -- Proof is omitted

end simplify_fractions_l207_207469


namespace total_sweaters_knit_l207_207217

-- Definitions from condition a)
def monday_sweaters : ℕ := 8
def tuesday_sweaters : ℕ := monday_sweaters + 2
def wednesday_sweaters : ℕ := tuesday_sweaters - 4
def thursday_sweaters : ℕ := wednesday_sweaters
def friday_sweaters : ℕ := monday_sweaters / 2

-- Theorem statement
theorem total_sweaters_knit : 
  monday_sweaters + tuesday_sweaters + wednesday_sweaters + thursday_sweaters + friday_sweaters = 34 :=
  by
    sorry

end total_sweaters_knit_l207_207217


namespace circumcenter_coords_l207_207037

-- Define the given points A, B, and C
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (-5, 1)
def C : ℝ × ℝ := (3, -5)

-- The target statement to prove
theorem circumcenter_coords :
  ∃ x y : ℝ, (x - 2)^2 + (y - 2)^2 = (x + 5)^2 + (y - 1)^2 ∧
             (x - 2)^2 + (y - 2)^2 = (x - 3)^2 + (y + 5)^2 ∧
             x = -1 ∧ y = -2 :=
by
  sorry

end circumcenter_coords_l207_207037


namespace garden_fencing_cost_l207_207685

theorem garden_fencing_cost (x y : ℝ) (h1 : x^2 + y^2 = 900) (h2 : x * y = 200)
    (cost_per_meter : ℝ) (h3 : cost_per_meter = 15) : 
    cost_per_meter * (2 * x + y) = 300 * Real.sqrt 7 + 150 * Real.sqrt 2 :=
by
  sorry

end garden_fencing_cost_l207_207685


namespace petya_time_comparison_l207_207539

theorem petya_time_comparison (V : ℝ) (a : ℝ) (hV_pos : 0 < V) (ha_pos : 0 < a) :
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  T_real > T_planned :=
by
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  have T_real_eq : T_real = (a / (1.25 * V)) + (a / (0.8 * V)) := by sorry
  have T_real_simplified : T_real = (2 * a / 5 / V) + (5 * a / 8 / V) := by sorry
  have T_real_combined : T_real = (41 * a) / (40 * V) := by sorry
  have planned_vs_real : T_real > a / V := by sorry
  exact planned_vs_real

end petya_time_comparison_l207_207539


namespace prism_surface_area_l207_207478

-- Define the base of the prism as an isosceles trapezoid ABCD
structure Trapezoid :=
(AB CD : ℝ)
(BC : ℝ)
(AD : ℝ)

-- Define the properties of the prism
structure Prism :=
(base : Trapezoid)
(diagonal_cross_section_area : ℝ)

-- Define the specific isosceles trapezoid from the problem
def myTrapezoid : Trapezoid :=
{ AB := 13, CD := 13, BC := 11, AD := 21 }

-- Define the specific prism from the problem with the given conditions
noncomputable def myPrism : Prism :=
{ base := myTrapezoid, diagonal_cross_section_area := 180 }

-- Define the total surface area as a function
noncomputable def total_surface_area (p : Prism) : ℝ :=
2 * (1 / 2 * (p.base.AD + p.base.BC) * (Real.sqrt ((p.base.CD) ^ 2 - ((p.base.AD - p.base.BC) / 2) ^ 2))) +
(p.base.AB + p.base.BC + p.base.CD + p.base.AD) * (p.diagonal_cross_section_area / (Real.sqrt ((1 / 2 * (p.base.AD + p.base.BC)) ^ 2 + (Real.sqrt ((p.base.CD) ^ 2 - ((p.base.AD - p.base.BC) / 2) ^ 2)) ^ 2)))

-- The proof problem in Lean
theorem prism_surface_area :
  total_surface_area myPrism = 906 :=
sorry

end prism_surface_area_l207_207478


namespace probability_unique_tens_digits_l207_207474

theorem probability_unique_tens_digits :
  let num_ways := 10^6 in
  let total_combinations := Nat.choose 70 6 in
  (num_ways : ℚ) / total_combinations = 625 / 74440775 :=
by 
  sorry

end probability_unique_tens_digits_l207_207474


namespace solve_quadratic_equation_l207_207350

theorem solve_quadratic_equation (x : ℝ) : 4 * (2 * x + 1) ^ 2 = 9 * (x - 3) ^ 2 ↔ x = -11 ∨ x = 1 := 
by sorry

end solve_quadratic_equation_l207_207350


namespace equation_of_circle_correct_l207_207098

open Real

def equation_of_circle_through_points (x y : ℝ) :=
  x^2 + y^2 - 4 * x - 6 * y

theorem equation_of_circle_correct :
  ∀ (x y : ℝ),
    (equation_of_circle_through_points (0 : ℝ) (0 : ℝ) = 0) →
    (equation_of_circle_through_points (4 : ℝ) (0 : ℝ) = 0) →
    (equation_of_circle_through_points (-1 : ℝ) (1 : ℝ) = 0) →
    (equation_of_circle_through_points x y = 0) :=
by 
  sorry

end equation_of_circle_correct_l207_207098


namespace functional_equation_solution_l207_207334

-- The mathematical problem statement in Lean 4

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h_monotonic : ∀ x y : ℝ, (f x) * (f y) = f (x + y))
  (h_mono : ∀ x y : ℝ, x < y → f x < f y ∨ f x > f y) :
  ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, f x = a^x :=
sorry

end functional_equation_solution_l207_207334


namespace pete_total_miles_l207_207627

-- Definitions based on conditions
def flip_step_count : ℕ := 89999
def steps_full_cycle : ℕ := 90000
def total_flips : ℕ := 52
def end_year_reading : ℕ := 55555
def steps_per_mile : ℕ := 1900

-- Total steps Pete walked
def total_steps_pete_walked (flips : ℕ) (end_reading : ℕ) : ℕ :=
  flips * steps_full_cycle + end_reading

-- Total miles Pete walked
def total_miles_pete_walked (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  steps / steps_per_mile

-- Given the parameters, closest number of miles Pete walked should be 2500
theorem pete_total_miles : total_miles_pete_walked (total_steps_pete_walked total_flips end_year_reading) steps_per_mile = 2500 :=
by
  sorry

end pete_total_miles_l207_207627


namespace positive_integers_expressible_l207_207017

theorem positive_integers_expressible :
  ∃ (x y : ℕ), (x > 0) ∧ (y > 0) ∧ (x^2 + y) / (x * y + 1) = 1 ∧
  ∃ (x' y' : ℕ), (x' > 0) ∧ (y' > 0) ∧ (x' ≠ x ∨ y' ≠ y) ∧ (x'^2 + y') / (x' * y' + 1) = 1 :=
by
  sorry

end positive_integers_expressible_l207_207017


namespace root_in_interval_l207_207293

noncomputable def f (a b x : ℝ) : ℝ := a^x + x - b

theorem root_in_interval (a b : ℝ) (ha : a > 1) (hb : 0 < b ∧ b < 1) : 
  ∃ x : ℝ, -1 < x ∧ x < 0 ∧ f a b x = 0 :=
by {
  sorry
}

end root_in_interval_l207_207293


namespace oscar_leap_difference_in_feet_l207_207705

theorem oscar_leap_difference_in_feet 
  (strides_per_gap : ℕ) 
  (leaps_per_gap : ℕ) 
  (total_distance : ℕ) 
  (num_poles : ℕ)
  (h1 : strides_per_gap = 54) 
  (h2 : leaps_per_gap = 15) 
  (h3 : total_distance = 5280) 
  (h4 : num_poles = 51) 
  : (total_distance / (strides_per_gap * (num_poles - 1)) -
       total_distance / (leaps_per_gap * (num_poles - 1)) = 5) :=
by
  sorry

end oscar_leap_difference_in_feet_l207_207705


namespace find_2a_plus_b_l207_207456

open Real

variables {a b : ℝ}

-- Conditions
def angles_in_first_quadrant (a b : ℝ) : Prop := 
  0 < a ∧ a < π / 2 ∧ 0 < b ∧ b < π / 2

def cos_condition (a b : ℝ) : Prop :=
  5 * cos a ^ 2 + 3 * cos b ^ 2 = 2

def sin_condition (a b : ℝ) : Prop :=
  5 * sin (2 * a) + 3 * sin (2 * b) = 0

-- Problem statement
theorem find_2a_plus_b (a b : ℝ) 
  (h1 : angles_in_first_quadrant a b)
  (h2 : cos_condition a b)
  (h3 : sin_condition a b) :
  2 * a + b = π / 2 := 
sorry

end find_2a_plus_b_l207_207456


namespace boundary_points_distance_probability_l207_207220

theorem boundary_points_distance_probability
  (a b c : ℕ)
  (h1 : ∀ (x y : ℝ), x ∈ [0, 4] → y ∈ [0, 4] → (|x - y| ≥ 1 / 2 → True))
  (h2 : ∀ (x y : ℝ), x ∈ [0, 4] → y ∈ [0, 4] → True)
  (h3 : ∃ a b c : ℕ, a - b * Real.pi = 2 ∧ c = 4 ∧ Int.gcd (Int.ofNat a) (Int.gcd (Int.ofNat b) (Int.ofNat c)) = 1) :
  (a + b + c = 62) := sorry

end boundary_points_distance_probability_l207_207220


namespace point_3_units_away_l207_207784

theorem point_3_units_away (x : ℤ) (h : abs (x + 1) = 3) : x = 2 ∨ x = -4 :=
by
  sorry

end point_3_units_away_l207_207784


namespace regular_polygon_sides_l207_207566

theorem regular_polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (160 : ℝ) = (180 * (n - 2)) / n) : n = 18 := 
by 
  sorry

end regular_polygon_sides_l207_207566


namespace minimum_additional_squares_needed_to_achieve_symmetry_l207_207868

def initial_grid : List (ℕ × ℕ) := [(1, 4), (4, 1)] -- Initial shaded squares

def is_symmetric (grid : List (ℕ × ℕ)) : Prop :=
  ∀ (x y : ℕ × ℕ), x ∈ grid → y ∈ grid →
    ((x.1 = 2 * 2 - y.1 ∧ x.2 = y.2) ∨
     (x.1 = y.1 ∧ x.2 = 5 - y.2) ∨
     (x.1 = 2 * 2 - y.1 ∧ x.2 = 5 - y.2))

def additional_squares_needed : ℕ :=
  6 -- As derived in the solution steps, 6 additional squares are needed to achieve symmetry

theorem minimum_additional_squares_needed_to_achieve_symmetry :
  ∀ (initial_shades : List (ℕ × ℕ)),
    initial_shades = initial_grid →
    ∃ (additional : List (ℕ × ℕ)),
      initial_shades ++ additional = symmetric_grid ∧
      additional.length = additional_squares_needed :=
by 
-- skip the proof
sorry

end minimum_additional_squares_needed_to_achieve_symmetry_l207_207868


namespace complete_the_square_l207_207246

theorem complete_the_square (x : ℝ) :
  x^2 - 8 * x + 5 = 0 ↔ (x - 4)^2 = 11 :=
by
  sorry

end complete_the_square_l207_207246


namespace probability_sum_sixteen_l207_207409

-- Define the probabilities involved
def probability_of_coin_fifteen := 1 / 2
def probability_of_die_one := 1 / 6

-- Define the combined probability
def combined_probability : ℚ := probability_of_coin_fifteen * probability_of_die_one

theorem probability_sum_sixteen : combined_probability = 1 / 12 := by
  sorry

end probability_sum_sixteen_l207_207409


namespace frustum_midsection_area_relation_l207_207636

theorem frustum_midsection_area_relation 
  (S₁ S₂ S₀ : ℝ) 
  (h₁: 0 ≤ S₁ ∧ 0 ≤ S₂ ∧ 0 ≤ S₀)
  (h₂: ∃ a h, (a / (a + 2 * h))^2 = S₂ / S₁ ∧ (a / (a + h))^2 = S₂ / S₀) :
  2 * Real.sqrt S₀ = Real.sqrt S₁ + Real.sqrt S₂ := 
sorry

end frustum_midsection_area_relation_l207_207636


namespace gervais_avg_mileage_l207_207033
variable (x : ℤ)

def gervais_daily_mileage : Prop := ∃ (x : ℤ), (3 * x = 1250 - 305) ∧ x = 315

theorem gervais_avg_mileage : gervais_daily_mileage :=
by
  sorry

end gervais_avg_mileage_l207_207033


namespace smallest_n_inequality_l207_207875

theorem smallest_n_inequality :
  ∃ (n : ℕ), ∀ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4) ∧ n = 4 :=
by
  -- Proof steps would go here
  sorry

end smallest_n_inequality_l207_207875


namespace john_total_feet_climbed_l207_207451

def first_stair_steps : ℕ := 20
def second_stair_steps : ℕ := 2 * first_stair_steps
def third_stair_steps : ℕ := second_stair_steps - 10
def step_height : ℝ := 0.5

theorem john_total_feet_climbed : 
  (first_stair_steps + second_stair_steps + third_stair_steps) * step_height = 45 :=
by
  sorry

end john_total_feet_climbed_l207_207451


namespace petya_time_l207_207538

variable (a V : ℝ)

noncomputable def planned_time := a / V
noncomputable def real_time := (a / (2.5 * V)) + (a / (1.6 * V))

theorem petya_time (hV : V > 0) (ha : a > 0) : real_time a V > planned_time a V :=
by
  sorry

end petya_time_l207_207538


namespace sum_of_fractions_bounds_l207_207839

theorem sum_of_fractions_bounds (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_sum_numerators : a + c = 1000) (h_sum_denominators : b + d = 1000) :
  (999 / 969 + 1 / 31) ≤ (a / b + c / d) ∧ (a / b + c / d) ≤ (999 + 1 / 999) :=
by
  sorry

end sum_of_fractions_bounds_l207_207839


namespace evaluate_expression_l207_207580

theorem evaluate_expression : 
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3 / 3) := 
by
  sorry

end evaluate_expression_l207_207580


namespace necessary_condition_not_sufficient_condition_main_l207_207586

example (x : ℝ) : (x^2 - 3 * x > 0) → (x > 4) ∨ (x < 0 ∧ x > 0) := by
  sorry

theorem necessary_condition (x : ℝ) :
  (x^2 - 3 * x > 0) → (x > 4) :=
by
  sorry

theorem not_sufficient_condition (x : ℝ) :
  ¬ (x > 4) → (x^2 - 3 * x > 0) :=
by
  sorry

theorem main (x : ℝ) :
  (x^2 - 3 * x > 0) ↔ ¬ (x > 4) :=
by
  sorry

end necessary_condition_not_sufficient_condition_main_l207_207586


namespace votes_cast_l207_207127

theorem votes_cast (V : ℝ) (candidate_votes : ℝ) (rival_margin : ℝ)
  (h1 : candidate_votes = 0.30 * V)
  (h2 : rival_margin = 4000)
  (h3 : 0.30 * V + (0.30 * V + rival_margin) = V) :
  V = 10000 := 
by 
  sorry

end votes_cast_l207_207127


namespace probability_fewer_heads_than_tails_is_793_over_2048_l207_207241

noncomputable def probability_fewer_heads_than_tails (n : ℕ) : ℝ :=
(793 / 2048 : ℚ)

theorem probability_fewer_heads_than_tails_is_793_over_2048 :
  probability_fewer_heads_than_tails 12 = (793 / 2048 : ℚ) :=
sorry

end probability_fewer_heads_than_tails_is_793_over_2048_l207_207241


namespace calculate_diff_of_squares_l207_207696

noncomputable def diff_of_squares (a b : ℕ) : ℕ :=
  a^2 - b^2

theorem calculate_diff_of_squares :
  diff_of_squares 601 597 = 4792 :=
by
  sorry

end calculate_diff_of_squares_l207_207696


namespace double_acute_angle_l207_207295

theorem double_acute_angle (α : ℝ) (h : 0 < α ∧ α < π / 2) : 0 < 2 * α ∧ 2 * α < π :=
by
  sorry

end double_acute_angle_l207_207295


namespace sequence_a_n_sum_T_n_l207_207720

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (b : ℕ → ℕ)
variable (T : ℕ → ℕ)

theorem sequence_a_n (n : ℕ) (hS : ∀ n, S n = 2 * a n - n) :
  a n = 2 ^ n - 1 :=
sorry

theorem sum_T_n (n : ℕ) (hb : ∀ n, b n = (2 * n + 1) * (a n + 1)) 
  (ha : ∀ n, a n = 2 ^ n - 1) :
  T n = 2 + (2 * n - 1) * 2 ^ (n + 1) :=
sorry

end sequence_a_n_sum_T_n_l207_207720


namespace inequality_solution_set_nonempty_range_l207_207893

theorem inequality_solution_set_nonempty_range (a : ℝ) :
  (∃ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0) ↔ (a ≤ -2 ∨ a ≥ 6 / 5) :=
by
  -- Proof is omitted
  sorry

end inequality_solution_set_nonempty_range_l207_207893


namespace coloring_hexagonal_pyramids_l207_207009

def color_ways : ℕ := 405

theorem coloring_hexagonal_pyramids :
  ∃ n : ℕ, n = color_ways :=
by {
  use 405,
  sorry
}

end coloring_hexagonal_pyramids_l207_207009


namespace trapezoid_DC_length_l207_207707

theorem trapezoid_DC_length 
  (AB DC: ℝ) (BC: ℝ) 
  (angle_BCD angle_CDA: ℝ)
  (h1: AB = 8)
  (h2: BC = 4 * Real.sqrt 3)
  (h3: angle_BCD = 60)
  (h4: angle_CDA = 45)
  (h5: AB = DC):
  DC = 14 + 4 * Real.sqrt 2 :=
sorry

end trapezoid_DC_length_l207_207707


namespace maximize_profit_l207_207388

variables (a x : ℝ) (t : ℝ := 5 - 12 / (x + 3)) (cost : ℝ := 10 + 2 * t) 
  (price : ℝ := 5 + 20 / t) (profit : ℝ := 2 * (price * t - cost - x))

-- Assume non-negativity and upper bound on promotional cost
variable (h_a_nonneg : 0 ≤ a)
variable (h_a_pos : 0 < a)

noncomputable def profit_function (x : ℝ) : ℝ := 20 - 4 / x - x

-- Prove the maximum promotional cost that maximizes the profit
theorem maximize_profit : 
  (if a ≥ 2 then ∃ y, y = 2 ∧ profit_function y = profit_function 2 
   else ∃ y, y = a ∧ profit_function y = profit_function a) := 
sorry

end maximize_profit_l207_207388


namespace tan_of_angle_in_third_quadrant_l207_207421

theorem tan_of_angle_in_third_quadrant (α : ℝ) (h1 : Real.sin α = -12 / 13) (h2 : π < α ∧ α < 3 * π / 2) : Real.tan α = 12 / 5 := 
sorry

end tan_of_angle_in_third_quadrant_l207_207421


namespace solve_arithmetic_sequence_l207_207734

theorem solve_arithmetic_sequence (x : ℝ) 
  (term1 term2 term3 : ℝ)
  (h1 : term1 = 3 / 4)
  (h2 : term2 = 2 * x - 3)
  (h3 : term3 = 7 * x) 
  (h_arith : term2 - term1 = term3 - term2) :
  x = -9 / 4 :=
by
  sorry

end solve_arithmetic_sequence_l207_207734


namespace geometric_series_common_ratio_l207_207982

theorem geometric_series_common_ratio (a r S : ℝ) 
  (hS : S = a / (1 - r)) 
  (h64 : (a * r^4) / (1 - r) = S / 64) : 
  r = 1 / 2 :=
by
  sorry

end geometric_series_common_ratio_l207_207982


namespace simplify_expression_l207_207369

theorem simplify_expression : (3 + 3 + 5) / 2 - 1 / 2 = 5 := by
  sorry

end simplify_expression_l207_207369


namespace man_profit_doubled_l207_207846

noncomputable def percentage_profit (C SP1 SP2 : ℝ) : ℝ :=
  (SP2 - C) / C * 100

theorem man_profit_doubled (C SP1 SP2 : ℝ) (h1 : SP1 = 1.30 * C) (h2 : SP2 = 2 * SP1) :
  percentage_profit C SP1 SP2 = 160 := by
  sorry

end man_profit_doubled_l207_207846


namespace min_value_M_l207_207297

theorem min_value_M (a b c : ℝ) (h1 : a < b) (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0): 
  ∃ M : ℝ, M = 8 ∧ M = (a + 2 * b + 4 * c) / (b - a) :=
sorry

end min_value_M_l207_207297


namespace find_x_and_verify_l207_207750

theorem find_x_and_verify (x : ℤ) (h : (x - 14) / 10 = 4) : (x - 5) / 7 = 7 := 
by 
  sorry

end find_x_and_verify_l207_207750


namespace initial_percentage_female_workers_l207_207235

theorem initial_percentage_female_workers
(E : ℕ) (F : ℝ) 
(h1 : E + 30 = 360) 
(h2 : (F / 100) * E = (55 / 100) * (E + 30)) :
F = 60 :=
by
  -- proof omitted
  sorry

end initial_percentage_female_workers_l207_207235


namespace median_of_right_triangle_l207_207062

theorem median_of_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) : 
  c / 2 = 5 :=
by
  rw [h3]
  norm_num

end median_of_right_triangle_l207_207062


namespace squares_difference_l207_207695

theorem squares_difference (a b : ℕ) (h₁ : a = 601) (h₂ : b = 597) : a^2 - b^2 = 4792 := by
  rw [h₁, h₂]
  -- insert actual proof here
  sorry

end squares_difference_l207_207695


namespace vector_addition_subtraction_identity_l207_207088

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (BC AB AC : V)

theorem vector_addition_subtraction_identity : BC + AB - AC = 0 := 
by sorry

end vector_addition_subtraction_identity_l207_207088


namespace josh_daily_hours_l207_207766

-- Definitions of the parameters
def hours_josh_per_day : ℕ := sorry
def hours_carl_per_day := hours_josh_per_day - 2
def days_per_week : ℕ := 5
def weeks_per_month : ℕ := 4
def hourly_wage_josh : ℝ := 9
def hourly_wage_carl := hourly_wage_josh / 2
def total_monthly_payment : ℝ := 1980

-- Lean statement proving the number of hours Josh works per day
theorem josh_daily_hours :
  let month_hours_josh := weeks_per_month * days_per_week * hours_josh_per_day in
  let month_hours_carl := weeks_per_month * days_per_week * hours_carl_per_day in
  let monthly_earnings_josh := month_hours_josh * hourly_wage_josh in
  let monthly_earnings_carl := month_hours_carl * hourly_wage_carl in
  monthly_earnings_josh + monthly_earnings_carl = total_monthly_payment → 
  hours_josh_per_day = 8 :=
by 
  sorry

end josh_daily_hours_l207_207766


namespace factor_polynomial_l207_207870

theorem factor_polynomial 
(a b c d : ℝ) :
  a^3 * (b^2 - d^2) + b^3 * (c^2 - a^2) + c^3 * (d^2 - b^2) + d^3 * (a^2 - c^2)
  = (a - b) * (b - c) * (c - d) * (d - a) * (a^2 + ab + ac + ad + b^2 + bc + bd + c^2 + cd + d^2) :=
sorry

end factor_polynomial_l207_207870


namespace product_of_two_special_numbers_is_perfect_square_l207_207884

-- Define the structure of the required natural numbers
structure SpecialNumber where
  m : ℕ
  n : ℕ
  value : ℕ := 2^m * 3^n

-- The main theorem to be proved
theorem product_of_two_special_numbers_is_perfect_square :
  ∀ (a b c d e : SpecialNumber),
  ∃ x y : SpecialNumber, ∃ k : ℕ, (x.value * y.value) = k * k :=
by
  sorry

end product_of_two_special_numbers_is_perfect_square_l207_207884


namespace sum_a1_a3_a5_l207_207641

-- Definitions
variable (a : ℕ → ℕ)
variable (b : ℕ → ℕ)

-- Conditions
axiom initial_condition : a 1 = 16
axiom relationship_ak_bk : ∀ k, b k = a k / 2
axiom ak_next : ∀ k, a (k + 1) = a k + 2 * (b k)

-- Theorem Statement
theorem sum_a1_a3_a5 : a 1 + a 3 + a 5 = 336 :=
by
  sorry

end sum_a1_a3_a5_l207_207641


namespace problem_ACD_l207_207729

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 - x

theorem problem_ACD (a : ℝ) :
  (f a 0 = (2/3) ∧
  ¬(∀ x, f a x ≥ 0 → ((a ≥ 1) ∨ (a ≤ -1))) ∧
  (∃ x1 x2, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) ∧
  (∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0)) :=
sorry

end problem_ACD_l207_207729


namespace geometric_series_common_ratio_l207_207962

theorem geometric_series_common_ratio (a r S : ℝ) (h₁ : S = a / (1 - r)) (h₂ : ar^4 / (1 - r) = S / 64) : r = 1 / 2 :=
  by
  sorry

end geometric_series_common_ratio_l207_207962


namespace rebecca_has_more_eggs_than_marbles_l207_207343

-- Given conditions
def eggs : Int := 20
def marbles : Int := 6

-- Mathematically equivalent statement to prove
theorem rebecca_has_more_eggs_than_marbles :
    eggs - marbles = 14 :=
by
    sorry

end rebecca_has_more_eggs_than_marbles_l207_207343


namespace ratio_of_numbers_l207_207487

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a + b = 7 * (a - b)) :
  a / b = 4 / 3 :=
by
  sorry

end ratio_of_numbers_l207_207487


namespace total_sweaters_knit_l207_207216

-- Definitions from condition a)
def monday_sweaters : ℕ := 8
def tuesday_sweaters : ℕ := monday_sweaters + 2
def wednesday_sweaters : ℕ := tuesday_sweaters - 4
def thursday_sweaters : ℕ := wednesday_sweaters
def friday_sweaters : ℕ := monday_sweaters / 2

-- Theorem statement
theorem total_sweaters_knit : 
  monday_sweaters + tuesday_sweaters + wednesday_sweaters + thursday_sweaters + friday_sweaters = 34 :=
  by
    sorry

end total_sweaters_knit_l207_207216


namespace meeting_time_l207_207223

/--
The Racing Magic takes 150 seconds to circle the racing track once.
The Charging Bull makes 40 rounds of the track in an hour.
Prove that Racing Magic and Charging Bull meet at the starting point for the second time 
after 300 minutes.
-/
theorem meeting_time (rac_magic_time : ℕ) (chrg_bull_rounds_hour : ℕ)
  (h1 : rac_magic_time = 150) (h2 : chrg_bull_rounds_hour = 40) : 
  ∃ t: ℕ, t = 300 := 
by
  sorry

end meeting_time_l207_207223


namespace solve_for_x_l207_207792

noncomputable def solve_equation (x : ℝ) : Prop := 
  (6 * x + 2) / (3 * x^2 + 6 * x - 4) = 3 * x / (3 * x - 2) ∧ x ≠ 2 / 3

theorem solve_for_x (x : ℝ) (h : solve_equation x) : x = (Real.sqrt 6) / 3 ∨ x = - (Real.sqrt 6) / 3 := 
  sorry

end solve_for_x_l207_207792


namespace Eric_white_marbles_l207_207706

theorem Eric_white_marbles (total_marbles blue_marbles green_marbles : ℕ) (h1 : total_marbles = 20) (h2 : blue_marbles = 6) (h3 : green_marbles = 2) : 
  total_marbles - (blue_marbles + green_marbles) = 12 := by
  sorry

end Eric_white_marbles_l207_207706


namespace percentage_not_red_roses_l207_207200

-- Definitions for the conditions
def roses : Nat := 25
def tulips : Nat := 40
def daisies : Nat := 60
def lilies : Nat := 15
def sunflowers : Nat := 10
def totalFlowers : Nat := roses + tulips + daisies + lilies + sunflowers -- 150
def redRoses : Nat := roses / 2 -- 12 (considering integer division)

-- Statement to prove
theorem percentage_not_red_roses : 
  ((totalFlowers - redRoses) * 100 / totalFlowers) = 92 := by
  sorry

end percentage_not_red_roses_l207_207200


namespace focal_length_ellipse_l207_207480

theorem focal_length_ellipse :
  let a := 2
  let b := Real.sqrt 3
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 2 :=
by
  sorry

end focal_length_ellipse_l207_207480


namespace KimSweaterTotal_l207_207210

theorem KimSweaterTotal :
  let monday := 8
  let tuesday := monday + 2
  let wednesday := tuesday - 4
  let thursday := wednesday
  let friday := monday / 2
  monday + tuesday + wednesday + thursday + friday = 34 := by
  sorry

end KimSweaterTotal_l207_207210


namespace geometric_series_common_ratio_l207_207968

theorem geometric_series_common_ratio (a r : ℝ) (h : a / (1 - r) = 64 * (a * r^4 / (1 - r))) : r = 1/2 :=
by {
  sorry
}

end geometric_series_common_ratio_l207_207968


namespace annual_decrease_rate_l207_207644

theorem annual_decrease_rate (P₀ P₂ : ℝ) (r : ℝ) (h₀ : P₀ = 8000) (h₂ : P₂ = 5120) :
  P₂ = P₀ * (1 - r / 100) ^ 2 → r = 20 :=
by
  intros h
  have h₀' : P₀ = 8000 := h₀
  have h₂' : P₂ = 5120 := h₂
  sorry

end annual_decrease_rate_l207_207644


namespace general_term_formula_sum_inequality_l207_207721

noncomputable def a (n : ℕ) : ℝ := if n > 0 then (-1)^(n-1) * 3 / 2^n else 0

noncomputable def S (n : ℕ) : ℝ := if n > 0 then 1 - (-1/2)^n else 0

theorem general_term_formula (n : ℕ) (hn : n > 0) :
  a n = (-1)^(n-1) * (3/2^n) :=
by sorry

theorem sum_inequality (n : ℕ) (hn : n > 0) :
  S n + 1 / S n ≤ 13 / 6 :=
by sorry

end general_term_formula_sum_inequality_l207_207721


namespace students_helped_on_third_day_l207_207857

theorem students_helped_on_third_day (books_total : ℕ) (books_per_student : ℕ) (students_day1 : ℕ) (students_day2 : ℕ) (students_day4 : ℕ) (books_day3 : ℕ) :
  books_total = 120 →
  books_per_student = 5 →
  students_day1 = 4 →
  students_day2 = 5 →
  students_day4 = 9 →
  books_day3 = books_total - ((students_day1 + students_day2 + students_day4) * books_per_student) →
  books_day3 / books_per_student = 6 :=
by
  sorry

end students_helped_on_third_day_l207_207857


namespace cosine_of_angle_in_third_quadrant_l207_207904

theorem cosine_of_angle_in_third_quadrant (B : ℝ) (hB : B ∈ Set.Ioo (π : ℝ) (3 * π / 2)) (hSinB : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
sorry

end cosine_of_angle_in_third_quadrant_l207_207904


namespace probability_sum_two_balls_less_than_five_l207_207828

theorem probability_sum_two_balls_less_than_five :
  let balls := {1, 2, 3, 4, 5}
  let bag1 := balls
  let bag2 := balls
  let events := { (x, y) | x ∈ bag1 ∧ y ∈ bag2 }
  let favorable_events := { (x, y) | x ∈ bag1 ∧ y ∈ bag2 ∧ x + y < 5 }
  let total_events := card events
  let favorable_count := card favorable_events
  let probability := (favorable_count : ℚ) / (total_events : ℚ)
  probability = 6 / 25 :=
by
  sorry -- Proof to be filled in

end probability_sum_two_balls_less_than_five_l207_207828


namespace simplify_expression_l207_207348

theorem simplify_expression (x y : ℝ) (h1 : x = 1) (h2 : y = 2) : 
  ((x + y) * (x - y) - (x - y)^2 + 2 * y * (x - y)) / (4 * y) = -1 :=
by
  sorry

end simplify_expression_l207_207348


namespace min_value_expr_l207_207742

theorem min_value_expr (x : ℝ) (h : x > -3) : ∃ m, (∀ y > -3, 2 * y + (1 / (y + 3)) ≥ m) ∧ m = 2 * Real.sqrt 2 - 6 :=
by
  sorry

end min_value_expr_l207_207742


namespace subset_implies_bound_l207_207419

def setA := {x : ℝ | x < 2}
def setB (m : ℝ) := {x : ℝ | x < m}

theorem subset_implies_bound (m : ℝ) (h : setB m ⊆ setA) : m ≤ 2 :=
by 
  sorry

end subset_implies_bound_l207_207419


namespace apples_to_cucumbers_l207_207061

theorem apples_to_cucumbers (a b c : ℕ) 
    (h₁ : 10 * a = 5 * b) 
    (h₂ : 3 * b = 4 * c) : 
    (24 * a) = 16 * c := 
by
  sorry

end apples_to_cucumbers_l207_207061


namespace wire_cutting_l207_207013

theorem wire_cutting : 
  ∃ (n : ℕ), n = 33 ∧ (∀ (x y : ℕ), 3 * x + y = 100 → x > 0 ∧ y > 0 → ∃ m : ℕ, m = n) :=
by {
  sorry
}

end wire_cutting_l207_207013


namespace total_heartbeats_during_race_l207_207153

namespace Heartbeats

def avg_heart_beats_per_minute : ℕ := 160
def pace_minutes_per_mile : ℕ := 6
def race_distance_miles : ℕ := 20

theorem total_heartbeats_during_race :
  (race_distance_miles * pace_minutes_per_mile * avg_heart_beats_per_minute = 19200) :=
by
  sorry

end Heartbeats

end total_heartbeats_during_race_l207_207153


namespace probability_six_integers_diff_tens_l207_207472

-- Defining the range and conditions for the problem
def set_of_integers : Finset ℤ := Finset.range 70 \ Finset.range 10

def has_different_tens_digit (s : Finset ℤ) : Prop :=
  (s.card = 6) ∧ (∀ x y ∈ s, x ≠ y → (x / 10) ≠ (y / 10))

noncomputable def num_ways_choose_six_diff_tens : ℚ :=
  ((7 : ℚ) * (10^6 : ℚ))

noncomputable def total_ways_choose_six : ℚ :=
  (Nat.choose 70 6 : ℚ)

noncomputable def probability_diff_tens : ℚ :=
  num_ways_choose_six_diff_tens / total_ways_choose_six

-- Statement claiming the required probability
theorem probability_six_integers_diff_tens :
  probability_diff_tens = 1750 / 2980131 :=
by
  sorry

end probability_six_integers_diff_tens_l207_207472


namespace probability_heads_all_three_tosses_l207_207238

theorem probability_heads_all_three_tosses :
  (1 / 2) * (1 / 2) * (1 / 2) = 1 / 8 := 
sorry

end probability_heads_all_three_tosses_l207_207238


namespace rakesh_gross_salary_before_tax_l207_207086

variable (S : ℝ)
variable (net_salary_after_tax fd_amount remaining_amount groceries_expenses utilities_expenses vacation_expense total_expenses : ℝ)

def deductions_and_expenses : Prop :=
  net_salary_after_tax = S * 0.93 ∧
  fd_amount = S * 0.15 + 200 ∧
  remaining_amount = net_salary_after_tax - fd_amount ∧
  groceries_expenses = 0.30 * remaining_amount ∧        -- Groceries expenses might fluctuate by 2%, not covered here directly for simplicity.
  utilities_expenses = 0.20 * remaining_amount ∧
  vacation_expense = 0.05 * remaining_amount ∧
  total_expenses = groceries_expenses + utilities_expenses + vacation_expense + 1500

theorem rakesh_gross_salary_before_tax (h : deductions_and_expenses S net_salary_after_tax fd_amount remaining_amount groceries_expenses utilities_expenses vacation_expense total_expenses) :
  remaining_amount - total_expenses = 2380 → S ≈ 11305.41 :=
by
  intros h391
  sorry

end rakesh_gross_salary_before_tax_l207_207086


namespace find_principal_amount_l207_207266

-- Given conditions
def SI : ℝ := 4016.25
def R : ℝ := 0.14
def T : ℕ := 5

-- Question: What is the principal amount P?
theorem find_principal_amount : (SI / (R * T) = 5737.5) :=
sorry

end find_principal_amount_l207_207266


namespace topaz_sapphire_value_equal_l207_207083

/-
  Problem statement: Given the following conditions:
  1. One sapphire and two topazes are three times more valuable than an emerald: S + 2T = 3E
  2. Seven sapphires and one topaz are eight times more valuable than an emerald: 7S + T = 8E
  
  Prove that the value of one topaz is equal to the value of one sapphire (T = S).
-/

theorem topaz_sapphire_value_equal
  (S T E : ℝ) 
  (h1 : S + 2 * T = 3 * E) 
  (h2 : 7 * S + T = 8 * E) :
  T = S := 
  sorry

end topaz_sapphire_value_equal_l207_207083


namespace arrange_letters_of_unique_word_l207_207736

-- Define the problem parameters
def unique_word := ["M₁", "I₁", "S₁", "S₂", "I₂", "P₁", "P₂", "I₃"]
def word_length := unique_word.length
def arrangement_count := Nat.factorial word_length

-- Theorem statement corresponding to the problem
theorem arrange_letters_of_unique_word :
  arrangement_count = 40320 :=
by
  sorry

end arrange_letters_of_unique_word_l207_207736


namespace curve_touches_x_axis_at_most_three_times_l207_207882

theorem curve_touches_x_axis_at_most_three_times
  (a b c d : ℝ) :
  ∃ (x : ℝ), (x^4 - x^5 + a * x^3 + b * x^2 + c * x + d = 0) → ∃ (y : ℝ), (y = 0) → 
  ∃(n : ℕ), (n ≤ 3) :=
by sorry

end curve_touches_x_axis_at_most_three_times_l207_207882


namespace find_value_of_m_l207_207337

/-- Given the universal set U, set A, and the complement of A in U, we prove that m = -2. -/
theorem find_value_of_m (m : ℤ) (U : Set ℤ) (A : Set ℤ) (complement_U_A : Set ℤ) 
  (h1 : U = {2, 3, m^2 + m - 4})
  (h2 : A = {m, 2})
  (h3 : complement_U_A = {3}) 
  (h4 : U = A ∪ complement_U_A) 
  (h5 : A ∩ complement_U_A = ∅) 
  : m = -2 :=
sorry

end find_value_of_m_l207_207337


namespace who_is_next_to_Denis_l207_207106

-- Definitions according to conditions
def SchoolChildren := ["Anya", "Borya", "Vera", "Gena", "Denis"]

def Position : Type := ℕ

def is_next_to (a b : Position) := abs (a - b) = 1

def valid_positions (positions : SchoolChildren → Position) :=
  positions "Borya" = 1 ∧
  is_next_to (positions "Vera") (positions "Anya") ∧
  ¬is_next_to (positions "Vera") (positions "Gena") ∧
  ¬is_next_to (positions "Anya") (positions "Borya") ∧
  ¬is_next_to (positions "Gena") (positions "Borya") ∧
  ¬is_next_to (positions "Anya") (positions "Gena")

theorem who_is_next_to_Denis:
  ∃ positions : SchoolChildren → Position, valid_positions positions ∧ 
  (is_next_to (positions "Denis") (positions "Anya") ∧
   is_next_to (positions "Denis") (positions "Gena")) :=
by
  sorry

end who_is_next_to_Denis_l207_207106


namespace polynomial_root_problem_l207_207927

theorem polynomial_root_problem (a b c d : ℤ) (r1 r2 r3 r4 : ℕ)
  (h_roots : ∀ x, x^4 + a * x^3 + b * x^2 + c * x + d = (x + r1) * (x + r2) * (x + r3) * (x + r4))
  (h_sum : a + b + c + d = 2009) :
  d = 528 := 
by
  sorry

end polynomial_root_problem_l207_207927


namespace proof_inequality_l207_207038

noncomputable def proof_problem (p q : ℝ) (m n : ℕ) (hpq : p + q = 1) (hp : 0 < p) (hq : 0 < q) : Prop :=
  (1 - p^m)^n + (1 - q^n)^m ≥ 1

theorem proof_inequality (p q : ℝ) (m n : ℕ) (hpq : p + q = 1) (hp : 0 < p) (hq : 0 < q) :
  (1 - p^m)^n + (1 - q^n)^m ≥ 1 :=
by
  sorry

end proof_inequality_l207_207038


namespace david_account_amount_l207_207250

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem david_account_amount : compound_interest 5000 0.06 2 1 = 5304.50 := by
  sorry

end david_account_amount_l207_207250


namespace cubic_root_identity_l207_207769

theorem cubic_root_identity (r : ℝ) (h : (r^(1/3)) - (1/(r^(1/3))) = 2) : r^3 - (1/r^3) = 14 := 
by 
  sorry

end cubic_root_identity_l207_207769


namespace bleach_to_detergent_ratio_changed_factor_l207_207954

theorem bleach_to_detergent_ratio_changed_factor :
  let original_bleach : ℝ := 4
  let original_detergent : ℝ := 40
  let original_water : ℝ := 100
  let altered_detergent : ℝ := 60
  let altered_water : ℝ := 300

  -- Calculate the factor by which the volume increased
  let original_total_volume := original_detergent + original_water
  let altered_total_volume := altered_detergent + altered_water
  let volume_increase_factor := altered_total_volume / original_total_volume

  -- The calculated factor of the ratio change
  let original_ratio_bleach_to_detergent := original_bleach / original_detergent

  altered_detergent > 0 → altered_water > 0 →
  volume_increase_factor * original_ratio_bleach_to_detergent = 2.5714 :=
by
  -- Insert proof here
  sorry

end bleach_to_detergent_ratio_changed_factor_l207_207954


namespace like_terms_monomials_l207_207194

theorem like_terms_monomials (m n : ℕ) (h₁ : m = 2) (h₂ : n = 1) : m + n = 3 := 
by
  sorry

end like_terms_monomials_l207_207194


namespace circle_passing_through_points_eq_l207_207094

theorem circle_passing_through_points_eq :
  ∃ D E F, (∀ x y, x^2 + y^2 + D*x + E*y + F = 0 ↔ 
    (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) ∧
  (D = -4 ∧ E = -6 ∧ F = 0) :=
begin
  sorry
end

end circle_passing_through_points_eq_l207_207094


namespace probability_sum_greater_than_five_l207_207650

-- Definitions for the conditions
def die_faces := {1, 2, 3, 4, 5, 6}
def possible_outcomes := (die_faces × die_faces).to_finset
def favorable_outcomes := possible_outcomes.filter (λ (pair : ℕ × ℕ), pair.fst + pair.snd > 5)
def probability_of_sum_greater_than_five := (favorable_outcomes.card : ℚ) / possible_outcomes.card

-- Problem statement
theorem probability_sum_greater_than_five :
  probability_of_sum_greater_than_five = 13 / 18 :=
sorry

end probability_sum_greater_than_five_l207_207650


namespace min_frac_sum_l207_207722

noncomputable def min_value (x y : ℝ) : ℝ :=
  if (x + y = 1 ∧ x > 0 ∧ y > 0) then 1/x + 4/y else 0

theorem min_frac_sum (x y : ℝ) (h₁ : x + y = 1) (h₂: x > 0) (h₃: y > 0) : 
  min_value x y = 9 :=
sorry

end min_frac_sum_l207_207722


namespace inequality_correct_l207_207169

noncomputable def a : ℝ := Real.exp (-0.5)
def b : ℝ := 0.5
noncomputable def c : ℝ := Real.log 1.5

theorem inequality_correct : a > b ∧ b > c :=
by
  sorry

end inequality_correct_l207_207169


namespace problem_l207_207460

def vec_a : ℝ × ℝ := (5, 3)
def vec_b : ℝ × ℝ := (1, -2)
def two_vec_b : ℝ × ℝ := (2 * 1, 2 * -2)
def expected_result : ℝ × ℝ := (3, 7)

theorem problem : (vec_a.1 - two_vec_b.1, vec_a.2 - two_vec_b.2) = expected_result :=
by
  sorry

end problem_l207_207460


namespace VehicleB_travel_time_l207_207493

theorem VehicleB_travel_time 
    (v_A v_B : ℝ)
    (d : ℝ)
    (h1 : d = 3 * (v_A + v_B))
    (h2 : 3 * v_A = d / 2)
    (h3 : ∀ t ≤ 3.5 , d - t * v_B - 0.5 * v_A = 0)
    : d / v_B = 7.2 :=
by
  sorry

end VehicleB_travel_time_l207_207493


namespace probability_of_different_tens_digits_l207_207475

open Finset

-- Define the basic setup
def integers (n : ℕ) : Finset ℕ := {i in (range n) | i ≥ 10 ∧ i ≤ 79}

def tens_digit (n : ℕ) : ℕ := n / 10

def six_integers_with_different_tens_digits (s : Finset ℕ) : Prop :=
  s.card = 6 ∧ (s.map ⟨tens_digit, by simp⟩).card = 6

def favorable_ways : ℕ :=
  7 * 10^6

def total_ways : ℕ :=
  nat.choose 70 6

noncomputable def probability : ℚ :=
  favorable_ways / total_ways

-- The main statement
theorem probability_of_different_tens_digits :
  ∀ (s : Finset ℕ), six_integers_with_different_tens_digits s → 
  probability = 175 / 2980131 :=
begin
  intros s h,
  sorry
end

end probability_of_different_tens_digits_l207_207475


namespace shekar_marks_in_math_l207_207347

theorem shekar_marks_in_math (M : ℕ) : 
  (65 + 82 + 67 + 75 + M) / 5 = 73 → M = 76 :=
by
  intros h
  sorry

end shekar_marks_in_math_l207_207347


namespace range_of_a_l207_207956

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 * a * x + 3 * x > 2 * a + 3) ↔ (x < 1)) → (a < -3 / 2) :=
by
  intro h
  sorry

end range_of_a_l207_207956


namespace tangential_circle_radius_l207_207833

theorem tangential_circle_radius (R r x : ℝ) (hR : R > r) (hx : x = 4 * R * r / (R + r)) :
  ∃ x, x = 4 * R * r / (R + r) := by
sorry

end tangential_circle_radius_l207_207833


namespace sin_minus_cos_value_l207_207041

theorem sin_minus_cos_value
  (α : ℝ)
  (h1 : Real.tan α = (Real.sqrt 3) / 3)
  (h2 : π < α ∧ α < 3 * π / 2) :
  Real.sin α - Real.cos α = -1/2 + Real.sqrt 3 / 2 :=
by
  sorry

end sin_minus_cos_value_l207_207041


namespace sqrt_expr_eq_l207_207373

theorem sqrt_expr_eq : (Real.sqrt 2 + Real.sqrt 3)^2 - (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) = 6 + 2 * Real.sqrt 6 :=
by sorry

end sqrt_expr_eq_l207_207373


namespace store_A_more_cost_effective_100_cost_expressions_for_x_most_cost_effective_plan_l207_207759

-- Definitions and conditions
def cost_per_soccer : ℕ := 200
def cost_per_basketball : ℕ := 80
def discount_A_soccer (n : ℕ) : ℕ := n * cost_per_soccer
def discount_A_basketball (n : ℕ) : ℕ := if n > 100 then (n - 100) * cost_per_basketball else 0
def discount_B_soccer (n : ℕ) : ℕ := n * cost_per_soccer * 8 / 10
def discount_B_basketball (n : ℕ) : ℕ := n * cost_per_basketball * 8 / 10

-- For x = 100
def total_cost_A_100 : ℕ := discount_A_soccer 100 + discount_A_basketball 100
def total_cost_B_100 : ℕ := discount_B_soccer 100 + discount_B_basketball 100

-- Prove that for x = 100, Store A is more cost-effective
theorem store_A_more_cost_effective_100 : total_cost_A_100 < total_cost_B_100 :=
by sorry

-- For x > 100, express costs in terms of x
def total_cost_A (x : ℕ) : ℕ := 80 * x + 12000
def total_cost_B (x : ℕ) : ℕ := 64 * x + 16000

-- Prove the expressions for costs
theorem cost_expressions_for_x (x : ℕ) (h : x > 100) : 
  total_cost_A x = 80 * x + 12000 ∧ total_cost_B x = 64 * x + 16000 :=
by sorry

-- For x = 300, most cost-effective plan
def combined_A_100_B_200 : ℕ := (discount_A_soccer 100 + cost_per_soccer * 100) + (200 * cost_per_basketball * 8 / 10)
def only_A_300 : ℕ := discount_A_soccer 100 + (300 - 100) * cost_per_basketball
def only_B_300 : ℕ := discount_B_soccer 100 + 300 * cost_per_basketball * 8 / 10

-- Prove the most cost-effective plan for x = 300
theorem most_cost_effective_plan : combined_A_100_B_200 < only_B_300 ∧ combined_A_100_B_200 < only_A_300 :=
by sorry

end store_A_more_cost_effective_100_cost_expressions_for_x_most_cost_effective_plan_l207_207759


namespace equal_sharing_of_chicken_wings_l207_207130

theorem equal_sharing_of_chicken_wings 
  (initial_wings : ℕ) (additional_wings : ℕ) (number_of_friends : ℕ)
  (total_wings : ℕ) (wings_per_person : ℕ)
  (h_initial : initial_wings = 8)
  (h_additional : additional_wings = 10)
  (h_number : number_of_friends = 3)
  (h_total : total_wings = initial_wings + additional_wings)
  (h_division : wings_per_person = total_wings / number_of_friends) :
  wings_per_person = 6 := 
  by
  sorry

end equal_sharing_of_chicken_wings_l207_207130


namespace seashells_problem_l207_207396

theorem seashells_problem
  (F : ℕ)
  (h : (150 - F) / 2 = 55) :
  F = 40 :=
  sorry

end seashells_problem_l207_207396


namespace solve_fractional_equation_l207_207817

theorem solve_fractional_equation
  (x : ℝ)
  (h1 : x ≠ 0)
  (h2 : x ≠ 2)
  (h_eq : 2 / x - 1 / (x - 2) = 0) : 
  x = 4 := by
  sorry

end solve_fractional_equation_l207_207817


namespace pirate_islands_probability_l207_207140

open Finset

/-- There are 7 islands.
There is a 1/5 chance of finding an island with treasure only (no traps).
There is a 1/10 chance of finding an island with treasure and traps.
There is a 1/10 chance of finding an island with traps only (no treasure).
There is a 3/5 chance of finding an island with neither treasure nor traps.
We want to prove that the probability of finding exactly 3 islands
with treasure only and the remaining 4 islands with neither treasure
nor traps is 81/2225. -/
theorem pirate_islands_probability :
  (Nat.choose 7 3 : ℚ) * ((1/5)^3) * ((3/5)^4) = 81 / 2225 :=
by
  /- Here goes the proof -/
  sorry

end pirate_islands_probability_l207_207140


namespace arcsin_cos_arcsin_arccos_sin_arccos_l207_207029

-- Define the statement
theorem arcsin_cos_arcsin_arccos_sin_arccos (x : ℝ) 
  (h1 : -1 ≤ x) 
  (h2 : x ≤ 1) :
  (Real.arcsin (Real.cos (Real.arcsin x)) + Real.arccos (Real.sin (Real.arccos x)) = Real.pi / 2) := 
sorry

end arcsin_cos_arcsin_arccos_sin_arccos_l207_207029


namespace fans_received_all_items_l207_207715

def multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, m = n * k

theorem fans_received_all_items :
  (∀ n, multiple_of 100 n → multiple_of 40 n ∧ multiple_of 60 n ∧ multiple_of 24 n ∧ n ≤ 7200 → ∃ k, n = 600 * k) →
  (∃ k : ℕ, 7200 / 600 = k ∧ k = 12) :=
by
  sorry

end fans_received_all_items_l207_207715


namespace parameterized_line_segment_problem_l207_207813

theorem parameterized_line_segment_problem
  (p q r s : ℝ)
  (hq : q = 1)
  (hs : s = 2)
  (hpq : p + q = 6)
  (hrs : r + s = 9) :
  p^2 + q^2 + r^2 + s^2 = 79 := 
sorry

end parameterized_line_segment_problem_l207_207813


namespace petya_time_comparison_l207_207541

theorem petya_time_comparison (V : ℝ) (a : ℝ) (hV_pos : 0 < V) (ha_pos : 0 < a) :
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  T_real > T_planned :=
by
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  have T_real_eq : T_real = (a / (1.25 * V)) + (a / (0.8 * V)) := by sorry
  have T_real_simplified : T_real = (2 * a / 5 / V) + (5 * a / 8 / V) := by sorry
  have T_real_combined : T_real = (41 * a) / (40 * V) := by sorry
  have planned_vs_real : T_real > a / V := by sorry
  exact planned_vs_real

end petya_time_comparison_l207_207541


namespace cos_third_quadrant_l207_207899

theorem cos_third_quadrant (B : ℝ) (hB: π < B ∧ B < 3 * π / 2) (hSinB : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
by
  sorry

end cos_third_quadrant_l207_207899


namespace arith_seq_a4_a10_l207_207067

variable {a : ℕ → ℕ}
axiom hp1 : a 1 + a 2 + a 3 = 32
axiom hp2 : a 11 + a 12 + a 13 = 118

theorem arith_seq_a4_a10 :
  a 4 + a 10 = 50 :=
by
  have h1 : a 2 = 32 / 3 := sorry
  have h2 : a 12 = 118 / 3 := sorry
  have h3 : a 2 + a 12 = 50 := sorry
  exact sorry

end arith_seq_a4_a10_l207_207067


namespace ball_and_ring_problem_l207_207584

theorem ball_and_ring_problem (x y : ℕ) (m_x m_y : ℕ) : 
  m_x + 2 = y ∧ 
  m_y = x + 2 ∧
  x * m_x + y * m_y - 800 = 2 * (y - x) ∧
  x^2 + y^2 = 881 →
  (x = 25 ∧ y = 16) ∨ (x = 16 ∧ y = 25) := 
by 
  sorry

end ball_and_ring_problem_l207_207584


namespace lines_are_perpendicular_l207_207307

-- Definitions of the conditions
variables {Point : Type*} [AffineSpace Point ℝ] 
variables (Γ₁ Γ₂ : Set Point) -- circles
variables (A B O C : Point) -- intersections and center
variables (D E : Point) -- intersections with lines

def is_center (O : Point) (Γ : Set Point) := sorry -- requires definition of a circle center
def is_on_circle (P : Point) (Γ : Set Point) := sorry -- requires definition of being on a circle

-- Problem setup as conditions
variables (hI : ∃ (A B : Point), A ≠ B ∧ A ∈ Γ₁ ∧ B ∈ Γ₁ ∧ A ∈ Γ₂ ∧ B ∈ Γ₂)
variables (hO : is_center O Γ₁)
variables (hC : C ∈ Γ₁ ∧ C ≠ A ∧ C ≠ B)
variables (hD : ∃ (P : Point), P ∈ Γ₂ ∧ P ∈ line_span ℝ {A, C})
variables (hE : ∃ (Q : Point), Q ∈ Γ₂ ∧ Q ∈ line_span ℝ {B, C})

theorem lines_are_perpendicular
  (hOc : line_span ℝ {O, C} = line_span ℝ {O, C})
  (hDE : ∃ (D E : Point), line_span ℝ {D, E} = line_span ℝ {D, E}) : 
  ⊥ (line_span ℝ {O, C}) (line_span ℝ {D, E}) := 
sorry

end lines_are_perpendicular_l207_207307


namespace discount_percentage_in_february_l207_207138

theorem discount_percentage_in_february (C : ℝ) (h1 : C > 0) 
(markup1 : ℝ) (markup2 : ℝ) (profit : ℝ) (D : ℝ) :
  markup1 = 0.20 → markup2 = 0.25 → profit = 0.125 →
  1.50 * C * (1 - D) = 1.125 * C → D = 0.25 :=
by
  intros
  sorry

end discount_percentage_in_february_l207_207138


namespace verify_digits_l207_207251

theorem verify_digits :
  ∀ (a b c d e f g h : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
  f ≠ g ∧ f ≠ h ∧
  g ≠ h ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 →
  (10 * a + b) - (10 * c + d) = 10 * e + d →
  e * f = 10 * d + c →
  (10 * g + d) + (10 * g + b) = 10 * h + c →
  a = 9 ∧ b = 8 ∧ c = 2 ∧ d = 4 ∧ e = 7 ∧ f = 6 ∧ g = 1 ∧ h = 3 :=
by
  intros a b c d e f g h
  intros h1 h2 h3
  sorry

end verify_digits_l207_207251


namespace water_drain_rate_l207_207356

theorem water_drain_rate
  (total_volume : ℕ)
  (total_time : ℕ)
  (H1 : total_volume = 300)
  (H2 : total_time = 25) :
  total_volume / total_time = 12 := 
by
  sorry

end water_drain_rate_l207_207356


namespace smallest_possible_value_l207_207746

theorem smallest_possible_value 
  (a : ℂ)
  (h : 8 * a^2 + 6 * a + 2 = 0) :
  ∃ z : ℂ, z = 3 * a + 1 ∧ z.re = -1 / 8 :=
by
  sorry

end smallest_possible_value_l207_207746


namespace seq_expression_l207_207486

noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ := n^2 * a n

theorem seq_expression (a : ℕ → ℝ) (h₁ : a 1 = 2) (h₂ : ∀ n ≥ 1, S n a = n^2 * a n) :
  ∀ n ≥ 1, a n = 4 / (n * (n + 1)) :=
by
  sorry

end seq_expression_l207_207486


namespace max_stamps_without_discount_theorem_l207_207912

def total_money := 5000
def price_per_stamp := 50
def max_stamps_without_discount := 100

theorem max_stamps_without_discount_theorem :
  price_per_stamp * max_stamps_without_discount ≤ total_money ∧
  ∀ n, n > max_stamps_without_discount → price_per_stamp * n > total_money := by
  sorry

end max_stamps_without_discount_theorem_l207_207912


namespace petya_time_comparison_l207_207531

theorem petya_time_comparison (V a : ℝ) (hV : V > 0) (ha : a > 0) :
  (a / V) < ((a / (2.5 * V)) + (a / (1.6 * V))) := by
  have T_planned : ℝ := a / V
  have T_first_half : ℝ := a / (2.5 * V)
  have T_second_half : ℝ := a / (1.6 * V)
  have T_real : ℝ := T_first_half + T_second_half
  sorry

end petya_time_comparison_l207_207531


namespace math_evening_problem_l207_207003

theorem math_evening_problem
  (S : ℕ)
  (r : ℕ)
  (fifth_graders_per_row : ℕ := 3)
  (sixth_graders_per_row : ℕ := r - fifth_graders_per_row)
  (total_number_of_students : ℕ := r * r) :
  70 < total_number_of_students ∧ total_number_of_students < 90 → 
  r = 9 ∧ 
  6 * r = 54 ∧
  3 * r = 27 :=
sorry

end math_evening_problem_l207_207003


namespace regular_polygon_sides_l207_207575

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ α, α = 160 → ∑ x in range n, (180 - (180 - 160)) = 360) : n = 18 :=
by
  sorry

end regular_polygon_sides_l207_207575


namespace pentagon_areas_l207_207937

noncomputable def find_areas (x A_total : ℝ) : (ℝ × ℝ) :=
  let y := (A_total - x) / 10 in
  (y, y)

theorem pentagon_areas (x y z A_total : ℝ) (hx : x > 0) (hy : y = z) (hz : A_total = x + 10 * y) :
  y = z ∧ y = (A_total - x) / 10 :=
by
  sorry

end pentagon_areas_l207_207937


namespace regular_polygon_sides_l207_207571

-- Define the number of sides
def n : ℕ := sorry

-- The interior angle condition
def interior_angle_condition (n : ℕ) : Prop := 
  let interior_angle := (180 * (n - 2)) / n
  interior_angle = 160

-- Prove the statement
theorem regular_polygon_sides (h : interior_angle_condition n) : n = 18 := 
  sorry

end regular_polygon_sides_l207_207571


namespace actual_time_greater_than_planned_time_l207_207548

def planned_time (a V : ℝ) : ℝ := a / V

def actual_time (a V : ℝ) : ℝ := (a / (2.5 * V)) + (a / (1.6 * V))

theorem actual_time_greater_than_planned_time (a V : ℝ) (hV : V > 0) : 
  actual_time a V > planned_time a V :=
by 
  sorry

end actual_time_greater_than_planned_time_l207_207548


namespace cosine_in_third_quadrant_l207_207907

theorem cosine_in_third_quadrant (B : Real) 
  (h1 : Real.sin B = -5/13) 
  (h2 : π < B ∧ B < 3 * π / 2) : Real.cos B = -12/13 := 
sorry

end cosine_in_third_quadrant_l207_207907


namespace lowest_score_l207_207010

theorem lowest_score (score1 score2 : ℕ) (max_score : ℕ) (desired_mean : ℕ) (lowest_possible_score : ℕ) 
  (h_score1 : score1 = 82) (h_score2 : score2 = 75) (h_max_score : max_score = 100) (h_desired_mean : desired_mean = 85)
  (h_lowest_possible_score : lowest_possible_score = 83) : 
  ∃ x1 x2 : ℕ, x1 = max_score ∧ x2 = lowest_possible_score ∧ (score1 + score2 + x1 + x2) / 4 = desired_mean := by
  sorry

end lowest_score_l207_207010


namespace max_consecutive_integers_lt_1000_l207_207999

theorem max_consecutive_integers_lt_1000 : 
  ∃ n : ℕ, (n * (n + 1)) / 2 < 1000 ∧ ∀ m : ℕ, m > n → (m * (m + 1)) / 2 ≥ 1000 :=
sorry

end max_consecutive_integers_lt_1000_l207_207999


namespace combined_weight_of_student_and_sister_l207_207191

theorem combined_weight_of_student_and_sister
  (S : ℝ) (R : ℝ)
  (h1 : S = 90)
  (h2 : S - 6 = 2 * R) :
  S + R = 132 :=
by
  sorry

end combined_weight_of_student_and_sister_l207_207191


namespace olivia_cookies_total_l207_207935

def cookies_total (baggie_cookie_count : ℝ) (chocolate_chip_cookies : ℝ) 
                  (baggies_oatmeal_cookies : ℝ) (total_cookies : ℝ) : Prop :=
  let oatmeal_cookies := baggies_oatmeal_cookies * baggie_cookie_count
  oatmeal_cookies + chocolate_chip_cookies = total_cookies

theorem olivia_cookies_total :
  cookies_total 9.0 13.0 3.111111111 41.0 :=
by
  -- Proof goes here
  sorry

end olivia_cookies_total_l207_207935


namespace ratio_of_triangle_areas_l207_207201

theorem ratio_of_triangle_areas (kx ky k : ℝ)
(n m : ℕ) (h1 : n > 0) (h2 : m > 0) :
  let A := (1 / 2) * (ky / m) * (kx / 2)
  let B := (1 / 2) * (kx / n) * (ky / 2)
  (A / B) = (n / m) :=
by
  sorry

end ratio_of_triangle_areas_l207_207201


namespace non_zero_x_satisfies_equation_l207_207842

theorem non_zero_x_satisfies_equation :
  ∃ (x : ℝ), (x ≠ 0) ∧ (7 * x)^5 = (14 * x)^4 ∧ x = 16 / 7 :=
by {
  sorry
}

end non_zero_x_satisfies_equation_l207_207842


namespace woman_l207_207268

-- Define the variables and given conditions
variables (W S X : ℕ)
axiom s_eq : S = 27
axiom sum_eq : W + S = 84
axiom w_eq : W = 2 * S + X

theorem woman's_age_more_years : X = 3 :=
by
  -- Proof goes here
  sorry

end woman_l207_207268


namespace find_m_l207_207918

theorem find_m
  (m : ℝ)
  (A B : ℝ × ℝ × ℝ)
  (hA : A = (m, 2, 3))
  (hB : B = (1, -1, 1))
  (h_dist : (Real.sqrt ((m - 1) ^ 2 + (2 - (-1)) ^ 2 + (3 - 1) ^ 2) = Real.sqrt 13)) :
  m = 1 := 
sorry

end find_m_l207_207918


namespace find_a_and_a100_l207_207447

def seq (a : ℝ) (n : ℕ) : ℝ := (-1)^n * n + a

theorem find_a_and_a100 :
  ∃ a : ℝ, (seq a 1 + seq a 4 = 3 * seq a 2) ∧ (seq a 100 = 97) :=
by
  sorry

end find_a_and_a100_l207_207447


namespace addition_of_decimals_l207_207823

theorem addition_of_decimals :
  0.9 + 0.99 = 1.89 :=
by
  sorry

end addition_of_decimals_l207_207823


namespace total_slides_used_l207_207407

theorem total_slides_used (duration : ℕ) (initial_slides : ℕ) (initial_time : ℕ) (constant_rate : ℕ) (total_time: ℕ)
  (H1 : duration = 50)
  (H2 : initial_slides = 4)
  (H3 : initial_time = 2)
  (H4 : constant_rate = initial_slides / initial_time)
  (H5 : total_time = duration) 
  : (constant_rate * total_time) = 100 := 
by
  sorry

end total_slides_used_l207_207407


namespace find_BE_l207_207610

-- Definitions from the conditions
variable {A B C D E : Point}
variable (AB BC CA BD BE CE : ℝ)
variable (angleBAE angleCAD : Real.Angle)

-- Given conditions
axiom h1 : AB = 12
axiom h2 : BC = 17
axiom h3 : CA = 15
axiom h4 : BD = 7
axiom h5 : angleBAE = angleCAD

-- Required proof statement
theorem find_BE :
  BE = 1632 / 201 := by
  sorry

end find_BE_l207_207610


namespace white_area_of_painting_l207_207952

theorem white_area_of_painting (s : ℝ) (total_gray_area : ℝ) (gray_area_squares : ℕ)
  (h1 : ∀ t, t = 3 * s) -- The frame is 3 times the smaller square's side length.
  (h2 : total_gray_area = 62) -- The gray area is 62 cm^2.
  (h3 : gray_area_squares = 31) -- The gray area is composed of 31 smaller squares.
  : ∃ white_area, white_area = 10 := 
  sorry

end white_area_of_painting_l207_207952


namespace extreme_point_l207_207639

noncomputable def f (x : ℝ) : ℝ := (x^4 / 4) - (x^3 / 3)
noncomputable def f_prime (x : ℝ) : ℝ := deriv f x

theorem extreme_point (x : ℝ) : f_prime 1 = 0 ∧
  (∀ y, y < 1 → f_prime y < 0) ∧
  (∀ z, z > 1 → f_prime z > 0) :=
by
  sorry

end extreme_point_l207_207639


namespace find_k_l207_207179

theorem find_k (k : ℝ) (h : 0.5 * |-2 * k| * |k| = 1) : k = 1 ∨ k = -1 :=
sorry

end find_k_l207_207179


namespace regular_polygon_sides_l207_207565

theorem regular_polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (160 : ℝ) = (180 * (n - 2)) / n) : n = 18 := 
by 
  sorry

end regular_polygon_sides_l207_207565


namespace royalties_amount_l207_207320

/--
Given the following conditions:
1. No tax for royalties up to 800 yuan.
2. For royalties exceeding 800 yuan but not exceeding 4000 yuan, tax is levied at 14% on the amount exceeding 800 yuan.
3. For royalties exceeding 4000 yuan, tax is levied at 11% of the total royalties.

If someone has paid 420 yuan in taxes for publishing a book, prove that their royalties amount to 3800 yuan.
-/
theorem royalties_amount (r : ℝ) (h₁ : ∀ r, r ≤ 800 → 0 = r * 0 / 100)
  (h₂ : ∀ r, 800 < r ∧ r ≤ 4000 → 0.14 * (r - 800) = r * 0.14 / 100)
  (h₃ : ∀ r, r > 4000 → 0.11 * r = 420) : r = 3800 := sorry

end royalties_amount_l207_207320


namespace brushes_cost_l207_207339

-- Define the conditions
def canvas_cost (B : ℝ) : ℝ := 3 * B
def paint_cost : ℝ := 5 * 8
def total_material_cost (B : ℝ) : ℝ := B + canvas_cost B + paint_cost
def earning_from_sale : ℝ := 200 - 80

-- State the question as a theorem in Lean
theorem brushes_cost (B : ℝ) (h : total_material_cost B = earning_from_sale) : B = 20 :=
sorry

end brushes_cost_l207_207339


namespace equal_areas_of_ngons_l207_207121

noncomputable def area_of_ngon (n : ℕ) (sides : Fin n → ℝ) (radius : ℝ) (circumference : ℝ) : ℝ := sorry

theorem equal_areas_of_ngons 
  (n : ℕ) 
  (sides1 sides2 : Fin n → ℝ) 
  (radius : ℝ) 
  (circumference : ℝ)
  (h_sides : ∀ i : Fin n, ∃ j : Fin n, sides1 i = sides2 j)
  (h_inscribed1 : area_of_ngon n sides1 radius circumference = area_of_ngon n sides1 radius circumference)
  (h_inscribed2 : area_of_ngon n sides2 radius circumference = area_of_ngon n sides2 radius circumference) :
  area_of_ngon n sides1 radius circumference = area_of_ngon n sides2 radius circumference :=
sorry

end equal_areas_of_ngons_l207_207121


namespace regular_polygon_sides_l207_207579

-- Define the problem conditions based on the given problem.
variables (n : ℕ)
def sum_of_interior_angles (n : ℕ) : ℤ := 180 * (n - 2)
def interior_angle (n : ℕ) : ℤ := 160
def total_interior_angle (n : ℕ) : ℤ := 160 * n

-- State the problem and expected result.
theorem regular_polygon_sides (h : 180 * (n - 2) = 160 * n) : n = 18 := 
by {
  -- This is to only state the theorem for now, proof will be crafted separately.
  sorry
}

end regular_polygon_sides_l207_207579


namespace Petya_time_comparison_l207_207544

-- Define the conditions
variables (a V : ℝ) (hV : V > 0)

noncomputable def T_planned := a / V

noncomputable def T1 := a / (2.5 * V)

noncomputable def T2 := a / (1.6 * V)

noncomputable def T_real := T1 + T2

-- State the main theorem
theorem Petya_time_comparison (ha : a > 0) : T_real a V > T_planned a V :=
by
  -- Proof to be filled in
  sorry

end Petya_time_comparison_l207_207544


namespace problem1_problem2_l207_207047

def f (x : ℝ) : ℝ := |2 * x + 1| + |x - 1|

theorem problem1 (x : ℝ) : f x ≥ 4 ↔ x ≤ -4/3 ∨ x ≥ 4/3 := 
  sorry

theorem problem2 (a : ℝ) : (∀ x : ℝ, f x > a) ↔ a < 3/2 := 
  sorry

end problem1_problem2_l207_207047


namespace no_solution_for_given_eqn_l207_207230

open Real

noncomputable def no_real_solution (x : ℝ) : Prop :=
1 - log (sin x) = cos x 

theorem no_solution_for_given_eqn :
  ¬ ∃ x : ℝ, 1 - log (sin x) = cos x := 
by {
  sorry
}

end no_solution_for_given_eqn_l207_207230


namespace license_plates_count_l207_207387

def number_of_license_plates : ℕ :=
  let digit_choices := 10^5
  let letter_block_choices := 3 * 26^2
  let block_positions := 6
  digit_choices * letter_block_choices * block_positions

theorem license_plates_count : number_of_license_plates = 1216800000 := by
  -- proof steps here
  sorry

end license_plates_count_l207_207387


namespace train_speed_in_km_hr_l207_207394

-- Definitions based on conditions
def train_length : ℝ := 150  -- meters
def crossing_time : ℝ := 6  -- seconds

-- Definition for conversion factor
def meters_per_second_to_km_per_hour (speed_mps : ℝ) : ℝ := speed_mps * 3.6

-- Main theorem
theorem train_speed_in_km_hr : meters_per_second_to_km_per_hour (train_length / crossing_time) = 90 :=
by
  sorry

end train_speed_in_km_hr_l207_207394


namespace ratio_JL_JM_l207_207629

theorem ratio_JL_JM (s w h : ℝ) (shared_area_25 : 0.25 * s^2 = 0.4 * w * h) (jm_eq_s : h = s) :
  w / h = 5 / 8 :=
by
  -- Proof will go here
  sorry

end ratio_JL_JM_l207_207629


namespace part1_part2_l207_207915

theorem part1 (p : ℝ) (h : p = 2 / 5) : 
  (p^2 + 2 * (3 / 5) * p^2) = 0.352 :=
by 
  rw [h]
  sorry

theorem part2 (p : ℝ) (h : p = 2 / 5) : 
  (4 * (1 / (11.32 * p^4)) + 5 * (2.4 / (11.32 * p^4)) + 6 * (3.6 / (11.32 * p^4)) + 7 * (2.16 / (11.32 * p^4))) = 4.834 :=
by 
  rw [h]
  sorry

end part1_part2_l207_207915


namespace vector_parallel_l207_207433

theorem vector_parallel (x : ℝ) :
  let a : ℝ × ℝ := (2 * x + 1, 4)
  let b : ℝ × ℝ := (2 - x, 3)
  (3 * (2 * x + 1) - 4 * (2 - x) = 0) → (x = 1 / 2) :=
by
  intros a b h
  sorry

end vector_parallel_l207_207433


namespace geometric_series_common_ratio_l207_207986

theorem geometric_series_common_ratio (a r S : ℝ) 
  (hS : S = a / (1 - r)) 
  (h64 : (a * r^4) / (1 - r) = S / 64) : 
  r = 1 / 2 :=
by
  sorry

end geometric_series_common_ratio_l207_207986


namespace hypotenuse_length_l207_207370

-- Let a and b be the lengths of the non-hypotenuse sides of a right triangle.
-- We are given that a = 6 and b = 8, and we need to prove that the hypotenuse c is 10.
theorem hypotenuse_length (a b c : ℕ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c ^ 2 = a ^ 2 + b ^ 2) : c = 10 :=
by
  -- The proof goes here.
  sorry

end hypotenuse_length_l207_207370


namespace geometric_series_common_ratio_l207_207983

theorem geometric_series_common_ratio (a r S : ℝ) 
  (hS : S = a / (1 - r)) 
  (h64 : (a * r^4) / (1 - r) = S / 64) : 
  r = 1 / 2 :=
by
  sorry

end geometric_series_common_ratio_l207_207983


namespace cherries_count_l207_207136

theorem cherries_count (b s r c : ℝ) 
  (h1 : b + s + r + c = 360)
  (h2 : s = 2 * b)
  (h3 : r = 4 * s)
  (h4 : c = 2 * r) : 
  c = 640 / 3 :=
by 
  sorry

end cherries_count_l207_207136


namespace logarithmic_expression_evaluation_l207_207406

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem logarithmic_expression_evaluation : 
  log_base_10 (5 / 2) + 2 * log_base_10 2 - (1/2)⁻¹ = -1 := 
by 
  sorry

end logarithmic_expression_evaluation_l207_207406


namespace number_of_parallel_lines_l207_207745

/-- 
Given 10 parallel lines in the first set and the fact that the intersection 
of two sets of parallel lines forms 1260 parallelograms, 
prove that the second set contains 141 parallel lines.
-/
theorem number_of_parallel_lines (n : ℕ) (h₁ : 10 - 1 = 9) (h₂ : 9 * (n - 1) = 1260) : n = 141 :=
sorry

end number_of_parallel_lines_l207_207745


namespace jellybean_total_l207_207379

theorem jellybean_total 
    (blackBeans : ℕ)
    (greenBeans : ℕ)
    (orangeBeans : ℕ)
    (h1 : blackBeans = 8)
    (h2 : greenBeans = blackBeans + 2)
    (h3 : orangeBeans = greenBeans - 1) :
    blackBeans + greenBeans + orangeBeans = 27 :=
by
    -- The proof will be placed here
    sorry

end jellybean_total_l207_207379


namespace no_twelve_consecutive_primes_in_ap_l207_207789

theorem no_twelve_consecutive_primes_in_ap (d : ℕ) (h : d < 2000) :
  ∀ a : ℕ, ¬(∀ n : ℕ, n < 12 → (Prime (a + n * d))) :=
sorry

end no_twelve_consecutive_primes_in_ap_l207_207789


namespace students_know_mothers_birthday_l207_207231

-- Defining the given conditions
def total_students : ℕ := 40
def A : ℕ := 10
def B : ℕ := 12
def C : ℕ := 22
def D : ℕ := 26

-- Statement to prove
theorem students_know_mothers_birthday : (B + C) = 22 :=
by
  sorry

end students_know_mothers_birthday_l207_207231


namespace cos_B_third_quadrant_l207_207901

theorem cos_B_third_quadrant (B : ℝ) (hB1 : π < B ∧ B < 3 * π / 2) (hB2 : sin B = -5 / 13) : cos B = -12 / 13 :=
by
  sorry

end cos_B_third_quadrant_l207_207901


namespace wuxi_GDP_scientific_notation_l207_207269

theorem wuxi_GDP_scientific_notation :
  14800 = 1.48 * 10^4 :=
sorry

end wuxi_GDP_scientific_notation_l207_207269


namespace misha_second_attempt_points_l207_207622

/--
Misha made a homemade dartboard at his summer cottage. The round board is 
divided into several sectors by circles, and you can throw darts at it. 
Points are awarded based on the sector hit.

Misha threw 8 darts three times. In his second attempt, he scored twice 
as many points as in his first attempt, and in his third attempt, he scored 
1.5 times more points than in his second attempt. How many points did he 
score in his second attempt?
-/
theorem misha_second_attempt_points:
  ∀ (x : ℕ), 
  (x ≥ 24) →
  (2 * x ≥ 48) →
  (3 * x = 72) →
  (2 * x = 48) :=
by
  intros x h1 h2 h3
  sorry

end misha_second_attempt_points_l207_207622


namespace right_triangle_candidate_l207_207397

theorem right_triangle_candidate :
  (∃ a b c : ℕ, (a, b, c) = (1, 2, 3) ∧ a^2 + b^2 = c^2) ∨
  (∃ a b c : ℕ, (a, b, c) = (2, 3, 4) ∧ a^2 + b^2 = c^2) ∨
  (∃ a b c : ℕ, (a, b, c) = (3, 4, 5) ∧ a^2 + b^2 = c^2) ∨
  (∃ a b c : ℕ, (a, b, c) = (4, 5, 6) ∧ a^2 + b^2 = c^2) ↔
  (∃ a b c : ℕ, (a, b, c) = (3, 4, 5) ∧ a^2 + b^2 = c^2) :=
by
  sorry

end right_triangle_candidate_l207_207397


namespace product_of_roots_is_neg4_l207_207076

-- Define the polynomial
def poly : Polynomial ℤ := 3 * Polynomial.X^4 - 8 * Polynomial.X^3 + Polynomial.X^2 + 4 * Polynomial.X - 12

-- Define the proof problem
theorem product_of_roots_is_neg4 : 
  let a := 3 in
  let e := -12 in
  (a, b, c, d : ℤ) -> 0 = poly.eval a -> Polynomial.eval (poly) b = 0 -> Polynomial.eval (poly) c = 0 -> Polynomial.eval (poly) d = 0 -> (a * b * c * d) = -4 :=
by
  sorry

end product_of_roots_is_neg4_l207_207076


namespace pure_imaginary_k_l207_207669

theorem pure_imaginary_k (k : ℝ) :
  (2 * k^2 - 3 * k - 2 = 0) → (k^2 - 2 * k ≠ 0) → k = -1 / 2 :=
by
  intro hr hi
  -- Proof will go here.
  sorry

end pure_imaginary_k_l207_207669


namespace sheets_paper_150_l207_207530

def num_sheets_of_paper (S : ℕ) (E : ℕ) : Prop :=
  (S - E = 50) ∧ (3 * E - S = 150)

theorem sheets_paper_150 (S E : ℕ) : num_sheets_of_paper S E → S = 150 :=
by
  sorry

end sheets_paper_150_l207_207530


namespace fermats_little_theorem_for_q_plus_1_l207_207426

theorem fermats_little_theorem_for_q_plus_1 (q : ℕ) (h1 : Nat.Prime q) (h2 : q % 2 = 1) :
  (q + 1)^(q - 1) % q = 1 := by
  sorry

end fermats_little_theorem_for_q_plus_1_l207_207426


namespace Nancy_hourly_wage_l207_207783

def tuition_cost := 22000
def parents_coverage := tuition_cost / 2
def scholarship := 3000
def loan := 2 * scholarship
def working_hours := 200
def remaining_tuition := tuition_cost - parents_coverage - scholarship - loan
def hourly_wage_required := remaining_tuition / working_hours

theorem Nancy_hourly_wage : hourly_wage_required = 10 := by
  sorry

end Nancy_hourly_wage_l207_207783


namespace cos_sq_minus_sin_sq_l207_207880

variable (α β : ℝ)

theorem cos_sq_minus_sin_sq (h : Real.cos (α + β) * Real.cos (α - β) = 1 / 3) :
  Real.cos α ^ 2 - Real.sin β ^ 2 = 1 / 3 :=
sorry

end cos_sq_minus_sin_sq_l207_207880


namespace teacher_total_score_l207_207146

variable (written_score : ℕ)
variable (interview_score : ℕ)
variable (weight_written : ℝ)
variable (weight_interview : ℝ)

theorem teacher_total_score :
  (written_score = 80) → (interview_score = 60) → (weight_written = 0.6) → (weight_interview = 0.4) →
  (written_score * weight_written + interview_score * weight_interview = 72) :=
by
  sorry

end teacher_total_score_l207_207146


namespace geometric_series_common_ratio_l207_207980

theorem geometric_series_common_ratio (a r : ℝ) (h₁ : r ≠ 1)
    (h₂ : a / (1 - r) = 64 * (a * r^4) / (1 - r)) : r = 1/2 :=
by
  have h₃ : 1 = 64 * r^4 := by
    have : 1 - r ≠ 0 := by linarith
    field_simp at h₂; assumption
  sorry

end geometric_series_common_ratio_l207_207980


namespace hyperbola_eccentricity_l207_207192

theorem hyperbola_eccentricity
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b = -4 * a / 3)
  (hc : c = (Real.sqrt (a ^ 2 + b ^ 2)))
  (point_on_asymptote : ∃ x y : ℝ, x = 3 ∧ y = -4 ∧ (y = b / a * x ∨ y = -b / a * x)) :
  (c / a) = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_l207_207192


namespace petya_time_comparison_l207_207532

theorem petya_time_comparison (V a : ℝ) (hV : V > 0) (ha : a > 0) :
  (a / V) < ((a / (2.5 * V)) + (a / (1.6 * V))) := by
  have T_planned : ℝ := a / V
  have T_first_half : ℝ := a / (2.5 * V)
  have T_second_half : ℝ := a / (1.6 * V)
  have T_real : ℝ := T_first_half + T_second_half
  sorry

end petya_time_comparison_l207_207532


namespace connie_s_problem_l207_207160

theorem connie_s_problem (y : ℕ) (h : 3 * y = 90) : y / 3 = 10 :=
by
  sorry

end connie_s_problem_l207_207160


namespace equation1_solution_equation2_no_solution_l207_207794

theorem equation1_solution (x: ℝ) (h: x ≠ -1/2 ∧ x ≠ 1):
  (1 / (x - 1) = 5 / (2 * x + 1)) ↔ (x = 2) :=
sorry

theorem equation2_no_solution (x: ℝ) (h: x ≠ 1 ∧ x ≠ -1):
  ¬ ( (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1 ) :=
sorry

end equation1_solution_equation2_no_solution_l207_207794


namespace martin_boxes_l207_207081

theorem martin_boxes (total_crayons : ℕ) (crayons_per_box : ℕ) (number_of_boxes : ℕ) 
  (h1 : total_crayons = 56) (h2 : crayons_per_box = 7) 
  (h3 : total_crayons = crayons_per_box * number_of_boxes) : 
  number_of_boxes = 8 :=
by 
  sorry

end martin_boxes_l207_207081


namespace hannah_age_is_48_l207_207308

-- Define the ages of the brothers
def num_brothers : ℕ := 3
def age_each_brother : ℕ := 8

-- Define the sum of brothers' ages
def sum_brothers_ages : ℕ := num_brothers * age_each_brother

-- Define the age of Hannah
def hannah_age : ℕ := 2 * sum_brothers_ages

-- The theorem to prove Hannah's age is 48 years
theorem hannah_age_is_48 : hannah_age = 48 := by
  sorry

end hannah_age_is_48_l207_207308


namespace g_of_g_of_g_of_g_of_3_l207_207925

def g (x : ℕ) : ℕ :=
if x % 3 = 0 then x / 3 else x^2 + 2

theorem g_of_g_of_g_of_g_of_3 : g (g (g (g 3))) = 3 :=
by sorry

end g_of_g_of_g_of_g_of_3_l207_207925


namespace n_eq_14_l207_207883

variable {a : ℕ → ℕ}  -- the arithmetic sequence
variable {S : ℕ → ℕ}  -- the sum function of the first n terms
variable {d : ℕ}      -- the common difference of the arithmetic sequence

-- Given Conditions
axiom Sn_eq_4 : S 4 = 40
axiom Sn_eq_210 : ∃ (n : ℕ), S n = 210
axiom Sn_minus_4_eq_130 : ∃ (n : ℕ), S (n - 4) = 130

-- Main theorem to prove
theorem n_eq_14 : ∃ (n : ℕ),  S n = 210 ∧ S (n - 4) = 130 ∧ n = 14 :=
by
  sorry

end n_eq_14_l207_207883


namespace total_sweaters_knit_l207_207218

-- Definitions from condition a)
def monday_sweaters : ℕ := 8
def tuesday_sweaters : ℕ := monday_sweaters + 2
def wednesday_sweaters : ℕ := tuesday_sweaters - 4
def thursday_sweaters : ℕ := wednesday_sweaters
def friday_sweaters : ℕ := monday_sweaters / 2

-- Theorem statement
theorem total_sweaters_knit : 
  monday_sweaters + tuesday_sweaters + wednesday_sweaters + thursday_sweaters + friday_sweaters = 34 :=
  by
    sorry

end total_sweaters_knit_l207_207218


namespace denis_neighbors_l207_207117

theorem denis_neighbors :
  ∃ (positions : ℕ → String), 
  (positions 1 = "Borya") ∧ 
  (positions 2 ≠ "Gena") ∧ (positions 2 = "Vera" → positions 3 = "Anya" ∨ positions 3 = "Gena") ∧ 
  (positions 3 ≠ "Borya") ∧ (positions 3 ≠ "Gena") ∧ 
  (positions 5 ≠ "Borya") ∧ (positions 5 ≠ "Anya") → 
  (positions 4 = "Denis" → 
    (positions 3 = "Anya" ∨ positions 5 = "Gena") ∧ 
    (positions 3 ≠ "Gena" ∨ positions 5 = "Anya")) :=
by
  sorry

end denis_neighbors_l207_207117


namespace gcd_459_357_l207_207654

theorem gcd_459_357 : gcd 459 357 = 51 := 
sorry

end gcd_459_357_l207_207654


namespace Denis_next_to_Anya_Gena_l207_207115

inductive Person
| Anya
| Borya
| Vera
| Gena
| Denis

open Person

def isNextTo (p1 p2 : Person) (line : List Person) : Prop :=
  ∃ n, line.nth n = some p1 ∧ line.nth (n + 1) = some p2 ∨ line.nth (n - 1) = some p2

def notNextTo (p1 p2 : Person) (line : List Person) : Prop :=
  ¬ isNextTo p1 p2 line

theorem Denis_next_to_Anya_Gena :
  ∀ (line : List Person),
    line.length = 5 →
    line.head = some Borya →
    isNextTo Vera Anya line →
    notNextTo Vera Gena line →
    notNextTo Anya Borya line →
    notNextTo Gena Borya line →
    notNextTo Anya Gena line →
    isNextTo Denis Anya line ∧ isNextTo Denis Gena line :=
by
  intros line length_five borya_head vera_next_anya vera_not_next_gena anya_not_next_borya gena_not_next_borya anya_not_next_gena
  sorry

end Denis_next_to_Anya_Gena_l207_207115


namespace hall_breadth_l207_207385

theorem hall_breadth (l : ℝ) (w_s l_s b : ℝ) (n : ℕ)
  (hall_length : l = 36)
  (stone_width : w_s = 0.4)
  (stone_length : l_s = 0.5)
  (num_stones : n = 2700)
  (area_paving : l * b = n * (w_s * l_s)) :
  b = 15 := by
  sorry

end hall_breadth_l207_207385


namespace no_snow_probability_l207_207157

theorem no_snow_probability (p1 p2 p3 p4 : ℚ) 
  (h1 : p1 = 2 / 3) 
  (h2 : p2 = 3 / 4) 
  (h3 : p3 = 5 / 6) 
  (h4 : p4 = 1 / 2) : 
  (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4) = 1 / 144 :=
by
  sorry

end no_snow_probability_l207_207157


namespace exam_total_questions_l207_207066

/-- 
In an examination, a student scores 4 marks for every correct answer 
and loses 1 mark for every wrong answer. The student secures 140 marks 
in total. Given that the student got 40 questions correct, 
prove that the student attempted a total of 60 questions. 
-/
theorem exam_total_questions (C W T : ℕ) 
  (score_correct : C = 40)
  (total_score : 4 * C - W = 140)
  (total_questions : T = C + W) : 
  T = 60 := 
by 
  -- Proof omitted
  sorry

end exam_total_questions_l207_207066


namespace find_sale_in_second_month_l207_207260

def sale_in_second_month (sale1 sale3 sale4 sale5 sale6 target_average : ℕ) (S : ℕ) : Prop :=
  sale1 + S + sale3 + sale4 + sale5 + sale6 = target_average * 6

theorem find_sale_in_second_month :
  sale_in_second_month 5420 6200 6350 6500 7070 6200 5660 :=
by
  sorry

end find_sale_in_second_month_l207_207260


namespace bacteria_growth_time_l207_207091

theorem bacteria_growth_time (n0 : ℕ) (n : ℕ) (rate : ℕ) (time_step : ℕ) (final : ℕ)
  (h0 : n0 = 200)
  (h1 : rate = 3)
  (h2 : time_step = 5)
  (h3 : n = n0 * rate ^ final)
  (h4 : n = 145800) :
  final = 30 := 
sorry

end bacteria_growth_time_l207_207091


namespace quadratic_trinomial_form_l207_207129

noncomputable def quadratic_form (a b c : ℝ) (h : a ≠ 0) : Prop :=
  ∀ x : ℝ, 
    (a * (3.8 * x - 1)^2 + b * (3.8 * x - 1) + c) = (a * (-3.8 * x)^2 + b * (-3.8 * x) + c)

theorem quadratic_trinomial_form (a b c : ℝ) (h : a ≠ 0) : b = a → quadratic_form a b c h :=
by
  intro hba
  unfold quadratic_form
  intro x
  rw [hba]
  sorry

end quadratic_trinomial_form_l207_207129


namespace calculate_expression_l207_207552

theorem calculate_expression :
  50 * 24.96 * 2.496 * 500 = (1248)^2 :=
by
  sorry

end calculate_expression_l207_207552


namespace union_of_sets_l207_207336

def A (x : ℤ) : Set ℤ := {x^2, 2*x - 1, -4}
def B (x : ℤ) : Set ℤ := {x - 5, 1 - x, 9}

theorem union_of_sets (x : ℤ) (hx : x = -3) (h_inter : A x ∩ B x = {9}) :
  A x ∪ B x = {-8, -4, 4, -7, 9} :=
by
  sorry

end union_of_sets_l207_207336


namespace regular_polygon_sides_l207_207572

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ α, α = 160 → ∑ x in range n, (180 - (180 - 160)) = 360) : n = 18 :=
by
  sorry

end regular_polygon_sides_l207_207572


namespace opposite_of_2023_is_minus_2023_l207_207481

def opposite (x y : ℤ) : Prop := x + y = 0

theorem opposite_of_2023_is_minus_2023 : opposite 2023 (-2023) :=
by
  sorry

end opposite_of_2023_is_minus_2023_l207_207481


namespace geometric_sequence_product_l207_207171

theorem geometric_sequence_product :
  ∀ (a : ℕ → ℝ), (∀ n, a n > 0) →
  (∃ (a_1 a_99 : ℝ), (a_1 + a_99 = 10) ∧ (a_1 * a_99 = 16) ∧ a 1 = a_1 ∧ a 99 = a_99) →
  a 20 * a 50 * a 80 = 64 :=
by
  intro a hpos hex
  sorry

end geometric_sequence_product_l207_207171


namespace find_integer_pairs_l207_207281

theorem find_integer_pairs : 
  ∀ (x y : Int), x^3 = y^3 + 2 * y^2 + 1 ↔ (x, y) = (1, 0) ∨ (x, y) = (1, -2) ∨ (x, y) = (-2, -3) :=
by
  intros x y
  sorry

end find_integer_pairs_l207_207281


namespace students_in_class_l207_207753

-- Define the relevant variables and conditions
variables (P H W T A S : ℕ)

-- Given conditions
axiom poetry_club : P = 22
axiom history_club : H = 27
axiom writing_club : W = 28
axiom two_clubs : T = 6
axiom all_clubs : A = 6

-- Statement to prove
theorem students_in_class
  (poetry_club : P = 22)
  (history_club : H = 27)
  (writing_club : W = 28)
  (two_clubs : T = 6)
  (all_clubs : A = 6) :
  S = P + H + W - T - 2 * A :=
sorry

end students_in_class_l207_207753


namespace sufficient_but_not_necessary_l207_207862

theorem sufficient_but_not_necessary (a b : ℝ) :
  (a > b + 1) → (a > b) ∧ ¬(a > b → a > b + 1) :=
by
  sorry

end sufficient_but_not_necessary_l207_207862


namespace co_presidents_included_probability_l207_207827

-- Let the number of students in each club
def club_sizes : List ℕ := [6, 8, 9, 10]

-- Function to calculate binomial coefficient
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Function to calculate probability for a given club size
noncomputable def co_president_probability (n : ℕ) : ℚ :=
  (choose (n - 2) 2 : ℚ) / (choose n 4)

-- List of probabilities for each club
noncomputable def probabilities : List ℚ :=
  List.map co_president_probability club_sizes

-- Aggregate total probability by averaging the individual probabilities
noncomputable def total_probability : ℚ :=
  (1 / 4 : ℚ) * probabilities.sum

-- The proof problem: proving the total probability equals 119/700
theorem co_presidents_included_probability :
  total_probability = 119 / 700 := by
  sorry

end co_presidents_included_probability_l207_207827


namespace overall_average_score_l207_207198

structure Club where
  members : Nat
  average_score : Nat

def ClubA : Club := { members := 40, average_score := 90 }
def ClubB : Club := { members := 50, average_score := 81 }

theorem overall_average_score : 
  (ClubA.members * ClubA.average_score + ClubB.members * ClubB.average_score) / 
  (ClubA.members + ClubB.members) = 85 :=
by
  sorry

end overall_average_score_l207_207198


namespace hyperbola_equation_l207_207049

theorem hyperbola_equation
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (e : ℝ) (he : e = 2 * Real.sqrt 3 / 3)
  (dist_from_origin : ∀ A B : ℝ × ℝ, A = (0, -b) ∧ B = (a, 0) →
    abs (a * b) / Real.sqrt (a^2 + b^2) = Real.sqrt 3 / 2) :
  (a^2 = 3 ∧ b^2 = 1) → (∀ x y : ℝ, (x^2 / 3 - y^2 = 1)) := 
sorry

end hyperbola_equation_l207_207049


namespace cookie_difference_l207_207529

def AlyssaCookies : ℕ := 129
def AiyannaCookies : ℕ := 140
def Difference : ℕ := 11

theorem cookie_difference : AiyannaCookies - AlyssaCookies = Difference := by
  sorry

end cookie_difference_l207_207529


namespace find_g_neg3_l207_207634

variable (g : ℚ → ℚ)

-- Given condition
axiom condition : ∀ x : ℚ, x ≠ 0 → 4 * g (1/x) + (3 * g x) / x = 3 * x^2

-- Theorem statement
theorem find_g_neg3 : g (-3) = -27 / 2 := 
by 
  sorry

end find_g_neg3_l207_207634


namespace molly_age_l207_207631

theorem molly_age
  (S M : ℕ)
  (h_ratio : S / M = 4 / 3)
  (h_sandy_future : S + 6 = 42)
  : M = 27 :=
sorry

end molly_age_l207_207631


namespace problem_statement_l207_207006

theorem problem_statement : 2456 + 144 / 12 * 5 - 256 = 2260 := 
by
  -- statements and proof steps would go here
  sorry

end problem_statement_l207_207006


namespace ivy_covering_the_tree_l207_207158

def ivy_stripped_per_day := 6
def ivy_grows_per_night := 2
def days_to_strip := 10
def net_ivy_stripped_per_day := ivy_stripped_per_day - ivy_grows_per_night

theorem ivy_covering_the_tree : net_ivy_stripped_per_day * days_to_strip = 40 := by
  have h1 : net_ivy_stripped_per_day = 4 := by
    unfold net_ivy_stripped_per_day
    rfl
  rw [h1]
  show 4 * 10 = 40
  rfl

end ivy_covering_the_tree_l207_207158


namespace cosine_in_third_quadrant_l207_207908

theorem cosine_in_third_quadrant (B : Real) 
  (h1 : Real.sin B = -5/13) 
  (h2 : π < B ∧ B < 3 * π / 2) : Real.cos B = -12/13 := 
sorry

end cosine_in_third_quadrant_l207_207908


namespace rectangle_area_l207_207581

-- Define the given dimensions
def length : ℝ := 1.5
def width : ℝ := 0.75
def expected_area : ℝ := 1.125

-- State the problem
theorem rectangle_area (l w : ℝ) (h_l : l = length) (h_w : w = width) : l * w = expected_area :=
by sorry

end rectangle_area_l207_207581


namespace quotient_larger_than_dividend_l207_207261

-- Define the problem conditions
variables {a b : ℝ}

-- State the theorem corresponding to the problem
theorem quotient_larger_than_dividend (h : b ≠ 0) : ¬ (∀ a : ℝ, ∀ b : ℝ, (a / b > a) ) :=
by
  sorry

end quotient_larger_than_dividend_l207_207261


namespace sets_satisfy_union_l207_207735

theorem sets_satisfy_union (A : Set Int) : (A ∪ {-1, 1} = {-1, 0, 1}) → 
  (∃ (X : Finset (Set Int)), X.card = 4 ∧ ∀ B ∈ X, A = B) :=
  sorry

end sets_satisfy_union_l207_207735


namespace Louisa_total_travel_time_l207_207082

theorem Louisa_total_travel_time :
  ∀ (v : ℝ), v > 0 → (200 / v) + 4 = (360 / v) → (200 / v) + (360 / v) = 14 :=
by
  intros v hv eqn
  sorry

end Louisa_total_travel_time_l207_207082


namespace find_a1_l207_207446

-- Define the arithmetic sequence and the given conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_mean (x y z : ℝ) : Prop :=
  y^2 = x * z

def problem_statement (a : ℕ → ℝ) (d : ℝ) : Prop :=
  (arithmetic_sequence a d) ∧ (geometric_mean (a 1) (a 2) (a 4))

theorem find_a1 (a : ℕ → ℝ) (d : ℝ) (h : problem_statement a d) : a 1 = 1 := by
  have h_seq : arithmetic_sequence a d := h.1
  have h_geom : geometric_mean (a 1) (a 2) (a 4) := h.2
  sorry

end find_a1_l207_207446


namespace solve_chestnut_problem_l207_207773

def chestnut_problem : Prop :=
  ∃ (P M L : ℕ), (M = 2 * P) ∧ (L = P + 2) ∧ (P + M + L = 26) ∧ (M = 12)

theorem solve_chestnut_problem : chestnut_problem :=
by 
  sorry

end solve_chestnut_problem_l207_207773


namespace no_solutions_interval_length_l207_207417

theorem no_solutions_interval_length : 
  (∀ x a : ℝ, |x| ≠ ax - 2) → ([-1, 1].length = 2) :=
by {
  sorry
}

end no_solutions_interval_length_l207_207417


namespace amusement_park_ticket_length_l207_207524

theorem amusement_park_ticket_length (Area Width Length : ℝ) (h₀ : Area = 1.77) (h₁ : Width = 3) (h₂ : Area = Width * Length) : Length = 0.59 :=
by
  -- Proof will go here
  sorry

end amusement_park_ticket_length_l207_207524


namespace james_total_earnings_l207_207612

-- Define the earnings for January
def januaryEarnings : ℕ := 4000

-- Define the earnings for February based on January
def februaryEarnings : ℕ := 2 * januaryEarnings

-- Define the earnings for March based on February
def marchEarnings : ℕ := februaryEarnings - 2000

-- Define the total earnings including January, February, and March
def totalEarnings : ℕ := januaryEarnings + februaryEarnings + marchEarnings

-- State the theorem: total earnings should be 18000
theorem james_total_earnings : totalEarnings = 18000 := by
  sorry

end james_total_earnings_l207_207612


namespace geometric_series_common_ratio_l207_207963

theorem geometric_series_common_ratio (a r S : ℝ) (h₁ : S = a / (1 - r)) (h₂ : ar^4 / (1 - r) = S / 64) : r = 1 / 2 :=
  by
  sorry

end geometric_series_common_ratio_l207_207963


namespace closest_integer_to_cube_root_of_150_l207_207502

theorem closest_integer_to_cube_root_of_150 : 
  ∃ (n : ℤ), ∀ m : ℤ, abs (150 - 5 ^ 3) < abs (150 - m ^ 3) → n = 5 :=
by
  sorry

end closest_integer_to_cube_root_of_150_l207_207502


namespace tangent_line_eq_at_x_is_1_range_of_sum_extreme_values_l207_207288

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * Real.log x - x
noncomputable def g (x m : ℝ) : ℝ := f x + m * x^2
noncomputable def tangentLineEq (x y : ℝ) : Prop := x + 2 * y + 1 = 0
noncomputable def rangeCondition (x₁ x₂ m : ℝ) : Prop := g x₁ m + g x₂ m < -3 / 2

theorem tangent_line_eq_at_x_is_1 :
  tangentLineEq 1 (f 1) := 
sorry

theorem range_of_sum_extreme_values (h : 0 < m ∧ m < 1 / 4) (x₁ x₂ : ℝ) :
  rangeCondition x₁ x₂ m := 
sorry

end tangent_line_eq_at_x_is_1_range_of_sum_extreme_values_l207_207288


namespace gross_profit_value_l207_207645

theorem gross_profit_value
  (SP : ℝ) (C : ℝ) (GP : ℝ)
  (h1 : SP = 81)
  (h2 : GP = 1.7 * C)
  (h3 : SP = C + GP) :
  GP = 51 :=
by
  sorry

end gross_profit_value_l207_207645


namespace smallest_root_of_g_l207_207283

-- Define the function g(x)
def g (x : ℝ) : ℝ := 10 * x^4 - 17 * x^2 + 7

-- The main statement: proving the smallest root of g(x) is -sqrt(7/5)
theorem smallest_root_of_g : ∃ x : ℝ, g x = 0 ∧ ∀ y : ℝ, g y = 0 → x ≤ y := 
sorry

end smallest_root_of_g_l207_207283


namespace geometric_series_common_ratio_l207_207985

theorem geometric_series_common_ratio (a r S : ℝ) 
  (hS : S = a / (1 - r)) 
  (h64 : (a * r^4) / (1 - r) = S / 64) : 
  r = 1 / 2 :=
by
  sorry

end geometric_series_common_ratio_l207_207985


namespace area_of_square_l207_207340

-- Defining the points A and B as given in the conditions.
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (4, 6)

-- Theorem statement: proving that the area of the square given the endpoints A and B is 12.5.
theorem area_of_square : 
  ∀ (A B : ℝ × ℝ),
  A = (1, 2) → B = (4, 6) → 
  ∃ (area : ℝ), area = 12.5 := 
by
  intros A B hA hB
  sorry

end area_of_square_l207_207340


namespace hyperbola_focal_coordinates_l207_207808

theorem hyperbola_focal_coordinates:
  ∀ (x y : ℝ), x^2 / 16 - y^2 / 9 = 1 → ∃ c : ℝ, c = 5 ∧ (x = -c ∨ x = c) ∧ y = 0 :=
by
  intro x y
  sorry

end hyperbola_focal_coordinates_l207_207808


namespace find_angle_A_find_perimeter_l207_207763

-- Given problem conditions as Lean definitions
def triangle_sides (a b c : ℝ) : Prop :=
  ∃ B : ℝ, c = a * (Real.cos B + Real.sqrt 3 * Real.sin B)

def triangle_area (S a : ℝ) : Prop :=
  S = Real.sqrt 3 / 4 ∧ a = 1

-- Prove angle A
theorem find_angle_A (a b c S : ℝ) (hc : triangle_sides a b c) (ha : triangle_area S a) :
  ∃ A : ℝ, A = Real.pi / 6 := 
sorry

-- Prove perimeter
theorem find_perimeter (a b c S : ℝ) (hc : triangle_sides a b c) (ha : triangle_area S a) :
  ∃ P : ℝ, P = Real.sqrt 3 + 2 := 
sorry

end find_angle_A_find_perimeter_l207_207763


namespace total_pages_book_l207_207919

-- Define the conditions
def reading_speed1 : ℕ := 10 -- pages per day for first half
def reading_speed2 : ℕ := 5 -- pages per day for second half
def total_days : ℕ := 75 -- total days spent reading

-- This is the main theorem we seek to prove:
theorem total_pages_book (P : ℕ) 
  (h1 : ∃ D1 D2 : ℕ, D1 + D2 = total_days ∧ D1 * reading_speed1 = P / 2 ∧ D2 * reading_speed2 = P / 2) : 
  P = 500 :=
by
  sorry

end total_pages_book_l207_207919


namespace a_squared_plus_b_squared_less_than_c_squared_l207_207084

theorem a_squared_plus_b_squared_less_than_c_squared 
  (a b c : Real) 
  (h : a^2 + b^2 + a * b + b * c + c * a < 0) : 
  a^2 + b^2 < c^2 := 
  by 
  sorry

end a_squared_plus_b_squared_less_than_c_squared_l207_207084


namespace find_y_l207_207950

theorem find_y 
  (h : (5 + 8 + 17) / 3 = (12 + y) / 2) : y = 8 :=
sorry

end find_y_l207_207950


namespace cosine_in_third_quadrant_l207_207909

theorem cosine_in_third_quadrant (B : Real) 
  (h1 : Real.sin B = -5/13) 
  (h2 : π < B ∧ B < 3 * π / 2) : Real.cos B = -12/13 := 
sorry

end cosine_in_third_quadrant_l207_207909


namespace box_volume_correct_l207_207353

variables (length width height : ℕ)

def volume_of_box (length width height : ℕ) : ℕ :=
  length * width * height

theorem box_volume_correct :
  volume_of_box 20 15 10 = 3000 :=
by
  -- This is where the proof would go
  sorry 

end box_volume_correct_l207_207353


namespace squares_difference_l207_207694

theorem squares_difference (a b : ℕ) (h₁ : a = 601) (h₂ : b = 597) : a^2 - b^2 = 4792 := by
  rw [h₁, h₂]
  -- insert actual proof here
  sorry

end squares_difference_l207_207694


namespace probability_sum_greater_than_five_l207_207649

theorem probability_sum_greater_than_five (dice_outcomes : List (ℕ × ℕ)) (h: dice_outcomes = [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (3,1), (3,2), (4,1), (5,1), (2,4)] ++ 
                              [(1,5), (2,6), (3,3), (3,4), (3,5), (3,6), (4,2), (4,3), (4,4), (4,5), (4,6), 
                               (5,2), (5,3), (5,4), (5,5), (5,6), (6,1), (6,2), (6,3), (6,4), (6,5), (6,6)]) :
  p_greater_5 = 2 / 3 := 
by
  sorry

end probability_sum_greater_than_five_l207_207649


namespace circle_eq_l207_207096

theorem circle_eq (D E : ℝ) :
  (∀ {x y : ℝ}, (x = 0 ∧ y = 0) ∨
               (x = 4 ∧ y = 0) ∨
               (x = -1 ∧ y = 1) → 
               x^2 + y^2 + D * x + E * y = 0) →
  (D = -4 ∧ E = -6) :=
by
  intros h
  have h1 : 0^2 + 0^2 + D * 0 + E * 0 = 0 := by exact h (Or.inl ⟨rfl, rfl⟩)
  have h2 : 4^2 + 0^2 + D * 4 + E * 0 = 0 := by exact h (Or.inr (Or.inl ⟨rfl, rfl⟩))
  have h3 : (-1)^2 + 1^2 + D * (-1) + E * 1 = 0 := by exact h (Or.inr (Or.inr ⟨rfl, rfl⟩))
  sorry -- proof steps would go here to eventually show D = -4 and E = -6

end circle_eq_l207_207096


namespace distance_between_points_l207_207207

open Real -- opening real number namespace

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * cos θ, r * sin θ)

theorem distance_between_points :
  let A := polar_to_cartesian 2 (π / 3)
  let B := polar_to_cartesian 2 (2 * π / 3)
  dist A B = 2 :=
by
  sorry

end distance_between_points_l207_207207


namespace max_consecutive_integers_sum_lt_1000_l207_207998

theorem max_consecutive_integers_sum_lt_1000 :
  ∃ n : ℕ, (∀ m : ℕ, m ≤ n → m * (m + 1) / 2 < 1000) ∧ (n * (n + 1) / 2 < 1000) ∧ ¬((n + 1) * (n + 2) / 2 < 1000) :=
sorry

end max_consecutive_integers_sum_lt_1000_l207_207998


namespace spherical_coords_eq_l207_207869

theorem spherical_coords_eq :
  let x := 4
  let y := -4 * Real.sqrt 3
  let z := 4
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let θ := Real.atan2 y x
  let φ := Real.acos (z / ρ)
  (ρ, θ, φ) = (4 * Real.sqrt 5, 5 * Real.pi / 3, Real.acos (1 / Real.sqrt 5)) :=
by
  let x := 4
  let y := -4 * Real.sqrt 3
  let z := 4
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let θ := Real.atan2 y x
  let φ := Real.acos (z / ρ)
  have ρ_pos : 0 < ρ := by sorry
  have θ_range : 0 ≤ θ ∧ θ < 2 * Real.pi := by sorry
  have φ_range : 0 ≤ φ ∧ φ ≤ Real.pi := by sorry
  have h1 : ρ = 4 * Real.sqrt 5 := by sorry
  have h2 : θ = 5 * Real.pi / 3 := by sorry
  have h3 : φ = Real.acos (1 / Real.sqrt 5) := by sorry
  exact ⟨h1, h2, h3⟩

end spherical_coords_eq_l207_207869


namespace largest_sum_is_5_over_6_l207_207866

def sum_1 := (1/3) + (1/7)
def sum_2 := (1/3) + (1/8)
def sum_3 := (1/3) + (1/2)
def sum_4 := (1/3) + (1/9)
def sum_5 := (1/3) + (1/4)

theorem largest_sum_is_5_over_6 : (sum_3 = 5/6) ∧ ((sum_3 > sum_1) ∧ (sum_3 > sum_2) ∧ (sum_3 > sum_4) ∧ (sum_3 > sum_5)) :=
by
  sorry

end largest_sum_is_5_over_6_l207_207866


namespace atomic_weight_of_calcium_l207_207712

theorem atomic_weight_of_calcium (Ca I : ℝ) (h1 : 294 = Ca + 2 * I) (h2 : I = 126.9) : Ca = 40.2 :=
by
  sorry

end atomic_weight_of_calcium_l207_207712


namespace function_is_even_with_period_pi_div_2_l207_207811

noncomputable def f (x : ℝ) : ℝ := (1 + Real.cos (2 * x)) * (Real.sin x) ^ 2

theorem function_is_even_with_period_pi_div_2 : 
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, f (x + (π / 2)) = f x) :=
by
  sorry

end function_is_even_with_period_pi_div_2_l207_207811


namespace petya_time_l207_207536

variable (a V : ℝ)

noncomputable def planned_time := a / V
noncomputable def real_time := (a / (2.5 * V)) + (a / (1.6 * V))

theorem petya_time (hV : V > 0) (ha : a > 0) : real_time a V > planned_time a V :=
by
  sorry

end petya_time_l207_207536


namespace altitude_eq_4r_l207_207322

variable (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]

-- We define the geometrical relations and constraints
def AC_eq_BC (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] (AC BC : ℝ) : Prop :=
AC = BC

def in_circle_radius_eq_r (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] (incircle_radius r : ℝ) : Prop :=
incircle_radius = r

def ex_circle_radius_eq_r (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] (excircle_radius r : ℝ) : Prop :=
excircle_radius = r

-- Main theorem to prove
theorem altitude_eq_4r 
  (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]
  (AC BC : ℝ) (r : ℝ)
  (h : ℝ)
  (H1 : AC_eq_BC A B C D AC BC)
  (H2 : in_circle_radius_eq_r A B C D r r)
  (H3 : ex_circle_radius_eq_r A B C D r r) :
  h = 4 * r :=
  sorry

end altitude_eq_4r_l207_207322


namespace mary_probability_at_least_three_correct_l207_207930

noncomputable def probability_correct_guesses (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
Finset.sum (Finset.range (k + 1)) (λ i, (nat.choose n i : ℚ) * p^i * (1 - p)^(n - i))

theorem mary_probability_at_least_three_correct :
  probability_correct_guesses 5 2 (1/4) = 53/512 :=
by
  unfold probability_correct_guesses
  norm_cast
  sorry

end mary_probability_at_least_three_correct_l207_207930


namespace right_triangle_width_l207_207990

theorem right_triangle_width (height : ℝ) (side_square : ℝ) (width : ℝ) (n_triangles : ℕ) 
  (triangle_right : height = 2)
  (fit_inside_square : side_square = 2)
  (number_triangles : n_triangles = 2) :
  width = 2 :=
sorry

end right_triangle_width_l207_207990


namespace initial_average_correct_l207_207477

theorem initial_average_correct (A : ℕ) 
  (num_students : ℕ) (wrong_mark : ℕ) (correct_mark : ℕ)
  (wrong_avg : ℕ) (correct_avg : ℕ) 
  (h1 : num_students = 30)
  (h2 : wrong_mark = 70)
  (h3 : correct_mark = 10)
  (h4 : correct_avg = 98)
  (h5 : num_students * correct_avg = (num_students * A) - (wrong_mark - correct_mark)) :
  A = 100 := 
sorry

end initial_average_correct_l207_207477


namespace find_m_l207_207620

def g (x : ℤ) (A : ℤ) (B : ℤ) (C : ℤ) : ℤ := A * x^2 + B * x + C

theorem find_m (A B C m : ℤ) 
  (h1 : g 2 A B C = 0)
  (h2 : 100 < g 9 A B C ∧ g 9 A B C < 110)
  (h3 : 150 < g 10 A B C ∧ g 10 A B C < 160)
  (h4 : 10000 * m < g 200 A B C ∧ g 200 A B C < 10000 * (m + 1)) : 
  m = 16 :=
sorry

end find_m_l207_207620


namespace petya_time_comparison_l207_207534

theorem petya_time_comparison (V a : ℝ) (hV : V > 0) (ha : a > 0) :
  (a / V) < ((a / (2.5 * V)) + (a / (1.6 * V))) := by
  have T_planned : ℝ := a / V
  have T_first_half : ℝ := a / (2.5 * V)
  have T_second_half : ℝ := a / (1.6 * V)
  have T_real : ℝ := T_first_half + T_second_half
  sorry

end petya_time_comparison_l207_207534


namespace percentage_who_do_not_have_job_of_choice_have_university_diploma_l207_207758

theorem percentage_who_do_not_have_job_of_choice_have_university_diploma :
  ∀ (total_population university_diploma job_of_choice no_diploma_job_of_choice : ℝ),
    total_population = 100 →
    job_of_choice = 40 →
    no_diploma_job_of_choice = 10 →
    university_diploma = 48 →
    ((university_diploma - (job_of_choice - no_diploma_job_of_choice)) / (total_population - job_of_choice)) * 100 = 30 :=
by
  intros total_population university_diploma job_of_choice no_diploma_job_of_choice h1 h2 h3 h4
  sorry

end percentage_who_do_not_have_job_of_choice_have_university_diploma_l207_207758


namespace average_weight_increase_l207_207317

theorem average_weight_increase (A : ℝ) :
  let initial_total_weight := 10 * A
  let new_total_weight := initial_total_weight - 65 + 97
  let new_average := new_total_weight / 10
  let increase := new_average - A
  increase = 3.2 :=
by
  sorry

end average_weight_increase_l207_207317


namespace log_relationship_l207_207718

theorem log_relationship :
  let a := Real.logb 0.3 0.2
  let b := Real.logb 3 2
  let c := Real.logb 0.2 3
  c < b ∧ b < a :=
by
  let a := Real.logb 0.3 0.2
  let b := Real.logb 3 2
  let c := Real.logb 0.2 3
  sorry

end log_relationship_l207_207718


namespace consecutive_primes_sum_square_is_prime_l207_207416

-- Defining what it means for three numbers to be consecutive primes
def consecutive_primes (p q r : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  ((p < q ∧ q < r) ∨ (p < q ∧ q < r ∧ r < p) ∨ 
   (r < p ∧ p < q) ∨ (q < p ∧ p < r) ∨ 
   (q < r ∧ r < p) ∨ (r < q ∧ q < p))

-- Defining our main problem statement
theorem consecutive_primes_sum_square_is_prime :
  ∀ p q r : ℕ, consecutive_primes p q r → Nat.Prime (p^2 + q^2 + r^2) ↔ (p = 3 ∧ q = 5 ∧ r = 7) :=
by
  -- Sorry is used to skip the proof.
  sorry

end consecutive_primes_sum_square_is_prime_l207_207416


namespace nth_equation_proof_l207_207934

theorem nth_equation_proof (n : ℕ) (hn : n > 0) :
  (1 : ℝ) + (1 / (n : ℝ)) - (2 / (2 * n - 1)) = (2 * n^2 + n + 1) / (n * (2 * n - 1)) :=
by
  sorry

end nth_equation_proof_l207_207934


namespace evaluate_expression_l207_207411

-- Define the base value
def base := 3000

-- Define the exponential expression
def exp_value := base ^ base

-- Prove that base * exp_value equals base ^ (1 + base)
theorem evaluate_expression : base * exp_value = base ^ (1 + base) := by
  sorry

end evaluate_expression_l207_207411


namespace like_terms_monomials_l207_207195

theorem like_terms_monomials (m n : ℕ) (h₁ : m = 2) (h₂ : n = 1) : m + n = 3 := 
by
  sorry

end like_terms_monomials_l207_207195


namespace geometric_series_common_ratio_l207_207969

theorem geometric_series_common_ratio (a r : ℝ) (h : a / (1 - r) = 64 * (a * r^4 / (1 - r))) : r = 1/2 :=
by {
  sorry
}

end geometric_series_common_ratio_l207_207969


namespace fg_neg_two_l207_207432

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := 2 * x + 3

theorem fg_neg_two : f (g (-2)) = 2 := by
  sorry

end fg_neg_two_l207_207432


namespace best_fitting_model_is_model1_l207_207445

noncomputable def model1_R2 : ℝ := 0.98
noncomputable def model2_R2 : ℝ := 0.80
noncomputable def model3_R2 : ℝ := 0.54
noncomputable def model4_R2 : ℝ := 0.35

theorem best_fitting_model_is_model1 :
  model1_R2 > model2_R2 ∧ model1_R2 > model3_R2 ∧ model1_R2 > model4_R2 :=
by
  sorry

end best_fitting_model_is_model1_l207_207445


namespace Trumpington_marching_band_max_l207_207488

theorem Trumpington_marching_band_max (n : ℕ) (k : ℕ) 
  (h1 : 20 * n % 26 = 4)
  (h2 : n = 8 + 13 * k)
  (h3 : 20 * n < 1000) 
  : 20 * (8 + 13 * 3) = 940 := 
by
  sorry

end Trumpington_marching_band_max_l207_207488


namespace spacy_subsets_15_l207_207551

def spacy (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => 2
  | 2     => 3
  | n + 3 => spacy n + spacy (n-2)

theorem spacy_subsets_15 : spacy 15 = 406 := 
  sorry

end spacy_subsets_15_l207_207551


namespace shorter_piece_length_l207_207852

theorem shorter_piece_length (total_len : ℝ) (ratio : ℝ) (shorter_len : ℝ) (longer_len : ℝ) 
  (h1 : total_len = 49) (h2 : ratio = 2/5) (h3 : shorter_len = x) 
  (h4 : longer_len = (5/2) * x) (h5 : shorter_len + longer_len = total_len) : 
  shorter_len = 14 := 
by
  sorry

end shorter_piece_length_l207_207852


namespace russom_greatest_number_of_envelopes_l207_207492

theorem russom_greatest_number_of_envelopes :
  ∃ n, n > 0 ∧ 18 % n = 0 ∧ 12 % n = 0 ∧ ∀ m, m > 0 ∧ 18 % m = 0 ∧ 12 % m = 0 → m ≤ n :=
sorry

end russom_greatest_number_of_envelopes_l207_207492


namespace problem_l207_207928

theorem problem (k : ℕ) (hk : 0 < k) (n : ℕ) : 
  (∃ p : ℕ, n = 2 * 3 ^ (k - 1) * p ∧ 0 < p) ↔ 3^k ∣ (2^n - 1) := 
by 
  sorry

end problem_l207_207928


namespace find_f_six_l207_207719

theorem find_f_six (f : ℕ → ℤ) (h : ∀ (x : ℕ), f (x + 1) = x^2 - 4) : f 6 = 21 :=
by
sorry

end find_f_six_l207_207719


namespace max_consecutive_integers_sum_l207_207994

theorem max_consecutive_integers_sum (S_n : ℕ → ℕ) : (∀ n, S_n n = n * (n + 1) / 2) → ∀ n, (S_n n < 1000 ↔ n ≤ 44) :=
by
  intros H n
  split
  · intro H1
    have H2 : n * (n + 1) < 2000 := by
      rw [H n] at H1
      exact H1
    sorry
  · intro H1
    have H2 : n ≤ 44 := H1
    have H3 : n * (n + 1) < 2000 := by
      sorry
    have H4 : S_n n < 1000 := by
      rw [H n]
      exact H3
    exact H4

end max_consecutive_integers_sum_l207_207994


namespace range_of_m_for_roots_greater_than_2_l207_207955

theorem range_of_m_for_roots_greater_than_2 :
  ∀ m : ℝ, (∀ x : ℝ, x^2 + (m-2)*x + 5 - m = 0 → x > 2) ↔ (-5 < m ∧ m ≤ -4) :=
  sorry

end range_of_m_for_roots_greater_than_2_l207_207955


namespace largest_possible_value_of_m_l207_207453

theorem largest_possible_value_of_m :
  ∃ (X Y Z : ℕ), 0 ≤ X ∧ X ≤ 7 ∧ 0 ≤ Y ∧ Y ≤ 7 ∧ 0 ≤ Z ∧ Z ≤ 7 ∧
                 (64 * X + 8 * Y + Z = 475) ∧ 
                 (144 * Z + 12 * Y + X = 475) := 
sorry

end largest_possible_value_of_m_l207_207453


namespace probability_at_least_one_die_less_3_l207_207124

-- Definitions
def total_outcomes_dice : ℕ := 64
def outcomes_no_die_less_3 : ℕ := 36
def favorable_outcomes : ℕ := total_outcomes_dice - outcomes_no_die_less_3
def probability : ℚ := favorable_outcomes / total_outcomes_dice

-- Theorem statement
theorem probability_at_least_one_die_less_3 :
  probability = 7 / 16 :=
by
  -- Proof would go here
  sorry

end probability_at_least_one_die_less_3_l207_207124


namespace fraction_simplifiable_by_7_l207_207786

theorem fraction_simplifiable_by_7 (a b c : ℤ) (h : (100 * a + 10 * b + c) % 7 = 0) : 
  ((10 * b + c + 16 * a) % 7 = 0) ∧ ((10 * b + c - 61 * a) % 7 = 0) :=
by
  sorry

end fraction_simplifiable_by_7_l207_207786


namespace geometric_series_common_ratio_l207_207966

theorem geometric_series_common_ratio (a r S : ℝ) (h₁ : S = a / (1 - r)) (h₂ : ar^4 / (1 - r) = S / 64) : r = 1 / 2 :=
  by
  sorry

end geometric_series_common_ratio_l207_207966


namespace sum_of_tetrahedron_properties_eq_14_l207_207325

-- Define the regular tetrahedron properties
def regular_tetrahedron_edges : ℕ := 6
def regular_tetrahedron_vertices : ℕ := 4
def regular_tetrahedron_faces : ℕ := 4

-- State the theorem that needs to be proven
theorem sum_of_tetrahedron_properties_eq_14 :
  regular_tetrahedron_edges + regular_tetrahedron_vertices + regular_tetrahedron_faces = 14 :=
by
  sorry

end sum_of_tetrahedron_properties_eq_14_l207_207325


namespace no_sum_of_three_squares_l207_207468

theorem no_sum_of_three_squares (a k : ℕ) : 
  ¬ ∃ x y z : ℤ, 4^a * (8*k + 7) = x^2 + y^2 + z^2 :=
by
  sorry

end no_sum_of_three_squares_l207_207468


namespace sum_of_solutions_l207_207059

theorem sum_of_solutions (a b : ℝ) (h1 : a * (a - 4) = 12) (h2 : b * (b - 4) = 12) (h3 : a ≠ b) : a + b = 4 := 
by {
  -- missing proof part
  sorry
}

end sum_of_solutions_l207_207059


namespace count_valid_numbers_l207_207310

-- Define the range of numbers we are analyzing
def range := finset.Icc 1 500

-- Define the condition for being a multiple of both 4 and 6 (i.e., multiple of 12)
def is_multiple_of_12 (n : ℕ) : Prop := n % 12 = 0

-- Define the condition for not being a multiple of 5
def not_multiple_of_5 (n : ℕ) : Prop := ¬ (n % 5 = 0)

-- Define the condition for not being a multiple of 9
def not_multiple_of_9 (n : ℕ) : Prop := ¬ (n % 9 = 0)

-- Define the final set of numbers according to the conditions specified
def valid_numbers := range.filter (λ n, is_multiple_of_12 n ∧ not_multiple_of_5 n ∧ not_multiple_of_9 n)

-- Define the theorem we want to prove
theorem count_valid_numbers : valid_numbers.card = 26 :=
by
  sorry

end count_valid_numbers_l207_207310


namespace prob_product_not_gt_4_prob_diff_less_2_l207_207237

open MeasureTheory

noncomputable def part_i (balls : Finset ℕ) : ℚ :=
  let outcomes := {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}.to_finset in
  let favorable := {(1, 2), (1, 3), (1, 4)}.to_finset in
  (favorable.card : ℚ) / outcomes.card

theorem prob_product_not_gt_4 : part_i {1, 2, 3, 4}.to_finset = 1/2 := by
  sorry

noncomputable def part_ii (balls : Finset ℕ) : ℚ :=
  let outcomes := (balls.product balls).filter (λ p, |p.1 - p.2| < 2) in
  (outcomes.card : ℚ) / (balls.card * balls.card)

theorem prob_diff_less_2 : part_ii {1, 2, 3, 4}.to_finset = 5/8 := by
  sorry

end prob_product_not_gt_4_prob_diff_less_2_l207_207237


namespace smallest_rational_in_set_l207_207271

theorem smallest_rational_in_set : 
  ∀ (a b c d : ℚ), 
    a = -2/3 → b = -1 → c = 0 → d = 1 → 
    (a > b ∧ b < c ∧ c < d) → b = -1 := 
by
  intros a b c d ha hb hc hd h
  sorry

end smallest_rational_in_set_l207_207271


namespace find_f_zero_l207_207302

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x + b

theorem find_f_zero (a b : ℝ)
  (h1 : f 3 a b = 7)
  (h2 : f 5 a b = -1) : f 0 a b = 19 :=
by
  sorry

end find_f_zero_l207_207302


namespace perpendicular_condition_l207_207726

-- Definitions based on the conditions
def line_l1 (m : ℝ) (x y : ℝ) : Prop := (m + 1) * x + (1 - m) * y - 1 = 0
def line_l2 (m : ℝ) (x y : ℝ) : Prop := (m - 1) * x + (2 * m + 1) * y + 4 = 0

-- Perpendicularity condition based on the definition in conditions
def perpendicular (m : ℝ) : Prop :=
  (m + 1) * (m - 1) + (1 - m) * (2 * m + 1) = 0

-- Sufficient but not necessary condition
def sufficient_but_not_necessary (m : ℝ) : Prop :=
  m = 0

-- Final statement to prove
theorem perpendicular_condition :
  sufficient_but_not_necessary 0 -> perpendicular 0 :=
by
  sorry

end perpendicular_condition_l207_207726


namespace geometric_seq_condition_l207_207890

-- Defining a geometric sequence
def is_geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Defining an increasing sequence
def is_increasing_seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

-- The condition to be proved
theorem geometric_seq_condition (a : ℕ → ℝ) (h_geo : is_geometric_seq a) :
  (a 0 < a 1 → is_increasing_seq a) ∧ (is_increasing_seq a → a 0 < a 1) :=
by 
  sorry

end geometric_seq_condition_l207_207890


namespace minimum_choir_members_l207_207676

def choir_members_min (n : ℕ) : Prop :=
  (n % 8 = 0) ∧ 
  (n % 9 = 0) ∧ 
  (n % 10 = 0) ∧ 
  (n % 11 = 0)

theorem minimum_choir_members : ∃ n, choir_members_min n ∧ (∀ m, choir_members_min m → n ≤ m) :=
sorry

end minimum_choir_members_l207_207676


namespace probability_after_5_rounds_l207_207527

def initial_coins : ℕ := 5
def rounds : ℕ := 5
def final_probability : ℚ := 1 / 2430000

structure Player :=
  (name : String)
  (initial_coins : ℕ)
  (final_coins : ℕ)

def Abby : Player := ⟨"Abby", 5, 5⟩
def Bernardo : Player := ⟨"Bernardo", 4, 3⟩
def Carl : Player := ⟨"Carl", 3, 3⟩
def Debra : Player := ⟨"Debra", 4, 5⟩

def check_final_state (players : List Player) : Prop :=
  ∀ (p : Player), p ∈ players →
  (p.name = "Abby" ∧ p.final_coins = 5 ∨
   p.name = "Bernardo" ∧ p.final_coins = 3 ∨
   p.name = "Carl" ∧ p.final_coins = 3 ∨
   p.name = "Debra" ∧ p.final_coins = 5)

theorem probability_after_5_rounds :
  ∃ prob : ℚ, prob = final_probability ∧ check_final_state [Abby, Bernardo, Carl, Debra] :=
sorry

end probability_after_5_rounds_l207_207527


namespace geometric_series_common_ratio_l207_207984

theorem geometric_series_common_ratio (a r S : ℝ) 
  (hS : S = a / (1 - r)) 
  (h64 : (a * r^4) / (1 - r) = S / 64) : 
  r = 1 / 2 :=
by
  sorry

end geometric_series_common_ratio_l207_207984


namespace compound_interest_calculation_l207_207126

-- Given conditions
def P : ℝ := 20000
def r : ℝ := 0.03
def t : ℕ := 5

-- The amount after t years with compound interest
def A := P * (1 + r) ^ t

-- Prove the total amount is as given in choice B
theorem compound_interest_calculation : 
  A = 20000 * (1 + 0.03) ^ 5 :=
by
  sorry

end compound_interest_calculation_l207_207126


namespace factorize_expression1_factorize_expression2_l207_207026

variable {R : Type*} [CommRing R]

theorem factorize_expression1 (x y : R) : x^2 + 2 * x + 1 - y^2 = (x + y + 1) * (x - y + 1) :=
  sorry

theorem factorize_expression2 (m n p : R) : m^2 - n^2 - 2 * n * p - p^2 = (m + n + p) * (m - n - p) :=
  sorry

end factorize_expression1_factorize_expression2_l207_207026


namespace abs_neg_two_eq_two_neg_two_pow_zero_eq_one_l207_207698

theorem abs_neg_two_eq_two : |(-2)| = 2 :=
sorry

theorem neg_two_pow_zero_eq_one : (-2)^0 = 1 :=
sorry

end abs_neg_two_eq_two_neg_two_pow_zero_eq_one_l207_207698


namespace regular_polygon_sides_l207_207568

-- Define the number of sides
def n : ℕ := sorry

-- The interior angle condition
def interior_angle_condition (n : ℕ) : Prop := 
  let interior_angle := (180 * (n - 2)) / n
  interior_angle = 160

-- Prove the statement
theorem regular_polygon_sides (h : interior_angle_condition n) : n = 18 := 
  sorry

end regular_polygon_sides_l207_207568


namespace jungkook_red_balls_l207_207327

-- Definitions from conditions
def num_boxes : ℕ := 2
def red_balls_per_box : ℕ := 3

-- Theorem stating the problem
theorem jungkook_red_balls : (num_boxes * red_balls_per_box) = 6 :=
by sorry

end jungkook_red_balls_l207_207327


namespace Addison_High_School_college_attendance_l207_207270

theorem Addison_High_School_college_attendance:
  ∀ (G B : ℕ) (pG_not_college p_total_college : ℚ),
  G = 200 →
  B = 160 →
  pG_not_college = 0.4 →
  p_total_college = 0.6667 →
  ((B * 100) / 160) = 75 := 
by
  intro G B pG_not_college p_total_college G_eq B_eq pG_not_college_eq p_total_college_eq
  -- skipped proof
  sorry

end Addison_High_School_college_attendance_l207_207270


namespace certain_number_is_47_l207_207312

theorem certain_number_is_47 (x : ℤ) (h : 34 + x - 53 = 28) : x = 47 :=
by
  sorry

end certain_number_is_47_l207_207312


namespace sum_at_simple_interest_l207_207686

theorem sum_at_simple_interest 
  (P R : ℕ)
  (h : ((P * (R + 1) * 3) / 100) - ((P * R * 3) / 100) = 69) : 
  P = 2300 :=
by sorry

end sum_at_simple_interest_l207_207686


namespace geometric_series_common_ratio_l207_207967

theorem geometric_series_common_ratio (a r : ℝ) (h : a / (1 - r) = 64 * (a * r^4 / (1 - r))) : r = 1/2 :=
by {
  sorry
}

end geometric_series_common_ratio_l207_207967


namespace perfect_square_trinomial_l207_207602

theorem perfect_square_trinomial (m : ℤ) (h : ∃ b : ℤ, (x : ℤ) → x^2 - 10 * x + m = (x + b)^2) : m = 25 :=
sorry

end perfect_square_trinomial_l207_207602


namespace Louisa_travel_distance_l207_207463

variables (D : ℕ)

theorem Louisa_travel_distance : 
  (200 / 50 + 3 = D / 50) → D = 350 :=
by
  intros h
  sorry

end Louisa_travel_distance_l207_207463


namespace find_m_l207_207507

theorem find_m (m : ℕ) (h1 : List ℕ := [27, 32, 39, m, 46, 47])
            (h2 : List ℕ := [30, 31, 34, 41, 42, 45])
            (h3 : (39 + m) / 2 = 42) :
            m = 45 :=
by {
  sorry
}

end find_m_l207_207507


namespace range_of_function_l207_207253

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem range_of_function :
  ∀ (x : ℝ), 1 ≤ x ∧ x < 5 → -4 ≤ f x ∧ f x < 5 :=
by
  intro x hx
  sorry

end range_of_function_l207_207253


namespace problem_statement_l207_207189

-- Definition of the conditions
variables {a : ℝ} (h₀ : a > 0) (h₁ : a ≠ 1)

-- The Lean 4 statement for the problem
theorem problem_statement (h : 0 < a ∧ a < 1) : 
  (∀ x y : ℝ, x < y → a^x > a^y) → 
  (∀ x : ℝ, (2 - a) * x^3 > 0) ∧ 
  (∀ x : ℝ, (2 - a) * x^3 > 0 → 0 < a ∧ a < 2 ∧ (∀ x y : ℝ, x < y → a^x > a^y) → False) :=
by
  intros
  sorry

end problem_statement_l207_207189


namespace claudia_filled_5oz_glasses_l207_207867

theorem claudia_filled_5oz_glasses :
  ∃ (n : ℕ), n = 6 ∧ 4 * 8 + 15 * 4 + n * 5 = 122 :=
by
  sorry

end claudia_filled_5oz_glasses_l207_207867


namespace books_sold_wednesday_l207_207922

-- Define the conditions of the problem
def total_books : Nat := 1200
def sold_monday : Nat := 75
def sold_tuesday : Nat := 50
def sold_thursday : Nat := 78
def sold_friday : Nat := 135
def percentage_not_sold : Real := 66.5

-- Define the statement to be proved
theorem books_sold_wednesday : 
  let books_sold := total_books * (1 - percentage_not_sold / 100)
  let known_sales := sold_monday + sold_tuesday + sold_thursday + sold_friday
  books_sold - known_sales = 64 :=
by
  sorry

end books_sold_wednesday_l207_207922


namespace money_after_purchase_l207_207014

def initial_money : ℕ := 4
def cost_of_candy_bar : ℕ := 1
def money_left : ℕ := 3

theorem money_after_purchase :
  initial_money - cost_of_candy_bar = money_left := by
  sorry

end money_after_purchase_l207_207014


namespace range_of_a_l207_207723

noncomputable def interval1 (a : ℝ) : Prop := -2 < a ∧ a <= 1 / 2
noncomputable def interval2 (a : ℝ) : Prop := a >= 2

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, x + |x - 2 * a| > 1

theorem range_of_a (a : ℝ) (h1 : ∀ x : ℝ, p a ∨ q a) (h2 : ¬ (∀ x : ℝ, p a ∧ q a)) : 
  interval1 a ∨ interval2 a :=
sorry

end range_of_a_l207_207723


namespace next_to_Denis_l207_207107

def student := {name : String}

def Borya : student := {name := "Borya"}
def Anya : student := {name := "Anya"}
def Vera : student := {name := "Vera"}
def Gena : student := {name := "Gena"}
def Denis : student := {name := "Denis"}

variables l : list student

axiom h₁ : l.head = Borya
axiom h₂ : ∃ i, l.get? i = some Vera ∧ (l.get? (i - 1) = some Anya ∨ l.get? (i + 1) = some Anya) ∧ l.get? (i - 1) ≠ some Gena ∧ l.get? (i + 1) ≠ some Gena
axiom h₃ : ∀ i j, l.get? i ∈ [some Anya, some Gena, some Borya] → l.get? j ∈ [some Anya, some Gena, some Borya] → abs (i - j) ≠ 1

theorem next_to_Denis : ∃ i, l.get? i = some Denis → (l.get? (i - 1) = some Anya ∨ l.get? (i + 1) = some Anya) ∧ (l.get? (i - 1) = some Gena ∨ l.get? (i + 1) = some Gena) :=
sorry

end next_to_Denis_l207_207107


namespace fraction_multiplication_result_l207_207244

theorem fraction_multiplication_result :
  (5 * 7) / 8 = 4 + 3 / 8 :=
by
  sorry

end fraction_multiplication_result_l207_207244


namespace problem1_problem2_problem3_l207_207418

def A : Set ℝ := Set.Icc (-1) 1
def B : Set ℝ := Set.Icc (-2) 2
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 + m * x - 1
def g (a m x : ℝ) : ℝ := 2 * abs (x - a) - x^2 - m * x

theorem problem1 (m : ℝ) : (∀ x, f m x ≤ 0 → x ∈ A) → m ∈ Set.Icc (-1) 1 :=
sorry

theorem problem2 (f_eq : ∀ x, f (-4) (1-x) = f (-4) (1+x)) : 
  Set.range (f (-4) ∘ id) ⊆ Set.Icc (-3) 15 :=
sorry

theorem problem3 (a : ℝ) (m : ℝ) :
  (a ≤ -1 → ∃ x, f m x + g a m x = -2*a - 2) ∧
  (-1 < a ∧ a < 1 → ∃ x, f m x + g a m x = a^2 - 1) ∧
  (a ≥ 1 → ∃ x, f m x + g a m x = 2*a - 2) :=
sorry

end problem1_problem2_problem3_l207_207418


namespace measure_angle_C_and_area_l207_207316

noncomputable def triangleProblem (a b c A B C : ℝ) :=
  (a + b = 5) ∧ (c = Real.sqrt 7) ∧ (4 * Real.sin ((A + B) / 2)^2 - Real.cos (2 * C) = 7 / 2)

theorem measure_angle_C_and_area (a b c A B C : ℝ) (h: triangleProblem a b c A B C) :
  C = Real.pi / 3 ∧ (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
by
  obtain ⟨ha, hb, hc⟩ := h
  sorry

end measure_angle_C_and_area_l207_207316


namespace number_of_ways_to_choose_positions_l207_207318

-- Definition of the problem conditions
def number_of_people : ℕ := 8

-- Statement of the proof problem
theorem number_of_ways_to_choose_positions : 
  (number_of_people) * (number_of_people - 1) * (number_of_people - 2) = 336 := by
  -- skipping the proof itself
  sorry

end number_of_ways_to_choose_positions_l207_207318


namespace calculate_g_g_2_l207_207600

def g (x : ℤ) : ℤ := 2 * x^2 + 2 * x - 1

theorem calculate_g_g_2 : g (g 2) = 263 :=
by
  sorry

end calculate_g_g_2_l207_207600


namespace functional_equation_solution_l207_207015

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f x * f y + f (x + y) = x * y) →
  (∀ x : ℝ, f x = x - 1 ∨ f x = -x - 1) :=
by
  sorry

end functional_equation_solution_l207_207015


namespace schoolchildren_lineup_l207_207104

theorem schoolchildren_lineup :
  ∀ (line : list string),
  (line.length = 5) →
  (line.head = some "Borya") →
  ((∃ i j, i ≠ j ∧ (line.nth i = some "Vera") ∧ (line.nth j = some "Anya") ∧ (i + 1 = j ∨ i = j + 1)) ∧ 
   (∀ i j, (line.nth i = some "Vera") → (line.nth j = some "Gena") → ((i + 1 ≠ j) ∧ (i ≠ j + 1)))) →
  (∀ i j k, (line.nth i = some "Anya") → (line.nth j = some "Borya") → (line.nth k = some "Gena") → 
            ((i + 1 ≠ j) ∧ (i ≠ j + 1) ∧ (i + 1 ≠ k) ∧ (i ≠ k + 1))) →
  ∃ i, (line.nth i = some "Denis") ∧ 
       (((line.nth (i + 1) = some "Anya") ∨ (line.nth (i - 1) = some "Anya")) ∧ 
        ((line.nth (i + 1) = some "Gena") ∨ (line.nth (i - 1) = some "Gena")))
:= by
 sorry

end schoolchildren_lineup_l207_207104


namespace length_segment_l207_207485

/--
Given a cylinder with a radius of 5 units capped with hemispheres at each end and having a total volume of 900π,
prove that the length of the line segment AB is 88/3 units.
-/
theorem length_segment (r : ℝ) (V : ℝ) (h : ℝ) : r = 5 ∧ V = 900 * Real.pi → h = 88 / 3 := by
  sorry

end length_segment_l207_207485


namespace range_of_a_l207_207911

theorem range_of_a (a : ℝ) :
  (∀ x : ℕ, 0 < x ∧ 3*x + a ≤ 2 → x = 1 ∨ x = 2) ↔ (-7 < a ∧ a ≤ -4) :=
sorry

end range_of_a_l207_207911


namespace inequality_proof_l207_207335

theorem inequality_proof 
  (a b c d : ℝ)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (d_pos : 0 < d)
  (sum_eq : a + b + c + d = 3) : 
  1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 ≤ 1 / (a * b * c * d)^2 := 
by 
  sorry

end inequality_proof_l207_207335


namespace part1_solution_part2_solution_l207_207044

variables (x y m : ℤ)

-- Given the system of equations
def system_of_equations (x y m : ℤ) : Prop :=
  (2 * x - y = m) ∧ (3 * x + 2 * y = m + 7)

-- Part (1) m = 0, find x = 1, y = 2
theorem part1_solution : system_of_equations x y 0 → x = 1 ∧ y = 2 :=
sorry

-- Part (2) point A(-2,3) in the second quadrant with distances 3 and 2, find m = -7
def is_in_second_quadrant (x y : ℤ) : Prop :=
  x < 0 ∧ y > 0

def distance_to_axes (x y dx dy : ℤ) : Prop :=
  y = dy ∧ x = -dx

theorem part2_solution : is_in_second_quadrant x y →
  distance_to_axes x y 2 3 →
  system_of_equations x y m →
  m = -7 :=
sorry

end part1_solution_part2_solution_l207_207044


namespace math_problem_proof_l207_207202

def ratio_area_BFD_square_ABCE (x : ℝ) (AF FE DE CD : ℝ) (h1 : AF = FE / 3) (h2 : CD = 3 * DE) : Prop :=
  let AE := (AF + FE)
  let area_square := (AE)^2
  let area_triangle_BFD := area_square - (1/2 * AF * (AE - FE) + 1/2 * (AE - FE) * FE + 1/2 * DE * CD)
  (area_triangle_BFD / area_square) = (1/16)
  
theorem math_problem_proof (x AF FE DE CD : ℝ) (h1 : AF = FE / 3) (h2 : CD = 3 * DE) (area_ratio : area_triangle_BFD / area_square = 1/16) : ratio_area_BFD_square_ABCE x AF FE DE CD h1 h2 :=
sorry

end math_problem_proof_l207_207202


namespace peanut_butter_sandwich_days_l207_207070

theorem peanut_butter_sandwich_days 
  (H : ℕ)
  (total_days : ℕ)
  (probability_ham_and_cake : ℚ)
  (ham_probability : ℚ)
  (cake_probability : ℚ)
  (Ham_days : H = 3)
  (Total_days : total_days = 5)
  (Ham_probability_val : ham_probability = H / 5)
  (Cake_probability_val : cake_probability = 1 / 5)
  (Probability_condition : ham_probability * cake_probability = 0.12) :
  5 - H = 2 :=
by 
  sorry

end peanut_butter_sandwich_days_l207_207070


namespace necessary_and_sufficient_condition_l207_207733

variable (x a : ℝ)

-- Condition 1: For all x in [1, 2], x^2 - a ≥ 0
def condition1 (x a : ℝ) : Prop := 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

-- Condition 2: There exists an x in ℝ such that x^2 + 2ax + 2 - a = 0
def condition2 (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

-- Proof problem: The necessary and sufficient condition for p ∧ q is a ≤ -2 ∨ a = 1
theorem necessary_and_sufficient_condition (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0) ↔ (a ≤ -2 ∨ a = 1) :=
sorry

end necessary_and_sufficient_condition_l207_207733


namespace polynomial_condition_l207_207708

theorem polynomial_condition {P : Polynomial ℝ} :
  (∀ (a b c : ℝ), a * b + b * c + c * a = 0 → P.eval (a - b) + P.eval (b - c) + P.eval (c - a) = 2 * P.eval (a + b + c)) →
    ∃ α β : ℝ, P = Polynomial.C α * Polynomial.X^4 + Polynomial.C β * Polynomial.X^2 :=
by
  intro h
  sorry

end polynomial_condition_l207_207708


namespace k_even_l207_207057

theorem k_even (n a b k : ℕ) (h1 : 2^n - 1 = a * b) (h2 : 2^k ∣ 2^(n-2) + a - b):
  k % 2 = 0 :=
sorry

end k_even_l207_207057


namespace calories_in_300g_l207_207079

/-
Define the conditions of the problem.
-/

def lemon_juice_grams := 150
def sugar_grams := 200
def lime_juice_grams := 50
def water_grams := 500

def lemon_juice_calories_per_100g := 30
def sugar_calories_per_100g := 390
def lime_juice_calories_per_100g := 20
def water_calories := 0

/-
Define the total weight of the beverage.
-/
def total_weight := lemon_juice_grams + sugar_grams + lime_juice_grams + water_grams

/-
Define the total calories of the beverage.
-/
def total_calories := 
  (lemon_juice_calories_per_100g * lemon_juice_grams / 100) + 
  (sugar_calories_per_100g * sugar_grams / 100) + 
  (lime_juice_calories_per_100g * lime_juice_grams / 100) + 
  water_calories

/-
Prove the number of calories in 300 grams of the beverage.
-/
theorem calories_in_300g : (total_calories / total_weight) * 300 = 278 := by
  sorry

end calories_in_300g_l207_207079


namespace average_computer_time_per_person_is_95_l207_207365

def people : ℕ := 8
def computers : ℕ := 5
def work_time : ℕ := 152 -- total working day minutes

def total_computer_time : ℕ := work_time * computers
def average_time_per_person : ℕ := total_computer_time / people

theorem average_computer_time_per_person_is_95 :
  average_time_per_person = 95 := 
by
  sorry

end average_computer_time_per_person_is_95_l207_207365


namespace Petya_time_comparison_l207_207546

-- Define the conditions
variables (a V : ℝ) (hV : V > 0)

noncomputable def T_planned := a / V

noncomputable def T1 := a / (2.5 * V)

noncomputable def T2 := a / (1.6 * V)

noncomputable def T_real := T1 + T2

-- State the main theorem
theorem Petya_time_comparison (ha : a > 0) : T_real a V > T_planned a V :=
by
  -- Proof to be filled in
  sorry

end Petya_time_comparison_l207_207546


namespace two_times_x_equals_two_l207_207439

theorem two_times_x_equals_two (x : ℝ) (h : x = 1) : 2 * x = 2 := by
  sorry

end two_times_x_equals_two_l207_207439


namespace teacher_total_score_l207_207148

/-- Conditions -/
def written_test_score : ℝ := 80
def interview_score : ℝ := 60
def written_test_weight : ℝ := 0.6
def interview_weight : ℝ := 0.4

/-- Prove the total score -/
theorem teacher_total_score :
  written_test_score * written_test_weight + interview_score * interview_weight = 72 :=
by
  sorry

end teacher_total_score_l207_207148


namespace sets_equal_l207_207332

theorem sets_equal :
  {u : ℤ | ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l} =
  {u : ℤ | ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r} := 
sorry

end sets_equal_l207_207332


namespace calculate_diff_of_squares_l207_207697

noncomputable def diff_of_squares (a b : ℕ) : ℕ :=
  a^2 - b^2

theorem calculate_diff_of_squares :
  diff_of_squares 601 597 = 4792 :=
by
  sorry

end calculate_diff_of_squares_l207_207697


namespace Mike_can_play_300_minutes_l207_207621

-- Define the weekly earnings, spending, and costs as conditions
def weekly_earnings : ℕ := 100
def half_spent_at_arcade : ℕ := weekly_earnings / 2
def food_cost : ℕ := 10
def token_cost_per_hour : ℕ := 8
def hour_in_minutes : ℕ := 60

-- Define the remaining money after buying food
def money_for_tokens : ℕ := half_spent_at_arcade - food_cost

-- Define the hours he can play
def hours_playable : ℕ := money_for_tokens / token_cost_per_hour

-- Define the total minutes he can play
def total_minutes_playable : ℕ := hours_playable * hour_in_minutes

-- Prove that with his expenditure, Mike can play for 300 minutes
theorem Mike_can_play_300_minutes : total_minutes_playable = 300 := 
by
  sorry -- Proof will be filled here

end Mike_can_play_300_minutes_l207_207621


namespace probability_girl_selection_l207_207144

-- Define the conditions
def total_candidates : ℕ := 3 + 1
def girl_candidates : ℕ := 1

-- Define the question in terms of probability
def probability_of_selecting_girl (total: ℕ) (girl: ℕ) : ℚ :=
  girl / total

-- Lean statement to prove
theorem probability_girl_selection : probability_of_selecting_girl total_candidates girl_candidates = 1 / 4 :=
by
  sorry

end probability_girl_selection_l207_207144


namespace cos_third_quadrant_l207_207898

theorem cos_third_quadrant (B : ℝ) (hB: π < B ∧ B < 3 * π / 2) (hSinB : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
by
  sorry

end cos_third_quadrant_l207_207898


namespace avg_annual_growth_rate_l207_207000

variable (x : ℝ)

/-- Initial GDP in 2020 is 43903.89 billion yuan and GDP in 2022 is 53109.85 billion yuan. 
    Prove that the average annual growth rate x satisfies the equation 43903.89 * (1 + x)^2 = 53109.85 -/
theorem avg_annual_growth_rate (x : ℝ) :
  43903.89 * (1 + x)^2 = 53109.85 :=
sorry

end avg_annual_growth_rate_l207_207000


namespace regular_polygon_sides_l207_207574

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ α, α = 160 → ∑ x in range n, (180 - (180 - 160)) = 360) : n = 18 :=
by
  sorry

end regular_polygon_sides_l207_207574


namespace find_number_l207_207262

-- Define a constant to represent the number
def c : ℝ := 1002 / 20.04

-- Define the main theorem
theorem find_number (x : ℝ) (h : x - c = 2984) : x = 3034 := by
  -- The proof will be placed here
  sorry

end find_number_l207_207262


namespace minimum_number_of_guests_l207_207100

def total_food : ℤ := 327
def max_food_per_guest : ℤ := 2

theorem minimum_number_of_guests :
  ∀ (n : ℤ), total_food ≤ n * max_food_per_guest → n = 164 :=
by
  sorry

end minimum_number_of_guests_l207_207100


namespace denis_neighbors_l207_207118

theorem denis_neighbors :
  ∃ (positions : ℕ → String), 
  (positions 1 = "Borya") ∧ 
  (positions 2 ≠ "Gena") ∧ (positions 2 = "Vera" → positions 3 = "Anya" ∨ positions 3 = "Gena") ∧ 
  (positions 3 ≠ "Borya") ∧ (positions 3 ≠ "Gena") ∧ 
  (positions 5 ≠ "Borya") ∧ (positions 5 ≠ "Anya") → 
  (positions 4 = "Denis" → 
    (positions 3 = "Anya" ∨ positions 5 = "Gena") ∧ 
    (positions 3 ≠ "Gena" ∨ positions 5 = "Anya")) :=
by
  sorry

end denis_neighbors_l207_207118


namespace frustum_radius_l207_207226

theorem frustum_radius (C1 C2 l: ℝ) (S_lateral: ℝ) (r: ℝ) :
  (C1 = 2 * r * π) ∧ (C2 = 6 * r * π) ∧ (l = 3) ∧ (S_lateral = 84 * π) → (r = 7) :=
by
  sorry

end frustum_radius_l207_207226


namespace range_of_a_l207_207046

noncomputable def f (a x : ℝ) : ℝ := 
  if x < 1 then a^x else (a-3)*x + 4*a

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) : 
  0 < a ∧ a ≤ 3/4 :=
by {
  sorry
}

end range_of_a_l207_207046


namespace trajectory_eq_l207_207321

theorem trajectory_eq :
  ∀ (x y : ℝ), abs x * abs y = 1 → (x * y = 1 ∨ x * y = -1) :=
by
  intro x y h
  sorry

end trajectory_eq_l207_207321


namespace union_M_N_l207_207589

def M : Set ℝ := { x | -3 < x ∧ x ≤ 5 }
def N : Set ℝ := { x | x > 3 }

theorem union_M_N : M ∪ N = { x | x > -3 } :=
by
  sorry

end union_M_N_l207_207589


namespace number_of_8th_graders_l207_207754

variable (x y : ℕ)
variable (y_valid : 0 ≤ y)

theorem number_of_8th_graders (h : x * (x + 3 - 2 * y) = 14) :
  x = 7 :=
by 
  sorry

end number_of_8th_graders_l207_207754


namespace polynomial_value_l207_207180

theorem polynomial_value (x y : ℝ) (h : x - 2 * y + 3 = 8) : x - 2 * y = 5 :=
by
  sorry

end polynomial_value_l207_207180


namespace sqrt_sum_inequality_l207_207458

theorem sqrt_sum_inequality (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h_sum : a + b + c = 3) :
  (Real.sqrt a + Real.sqrt b + Real.sqrt c ≥ a * b + b * c + c * a) :=
by
  sorry

end sqrt_sum_inequality_l207_207458


namespace circle_condition_l207_207405

theorem circle_condition (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 4 * m * x - 2 * y + 5 * m = 0) ↔ (m < 1 / 4 ∨ m > 1) :=
sorry

end circle_condition_l207_207405


namespace actual_time_greater_than_planned_time_l207_207549

def planned_time (a V : ℝ) : ℝ := a / V

def actual_time (a V : ℝ) : ℝ := (a / (2.5 * V)) + (a / (1.6 * V))

theorem actual_time_greater_than_planned_time (a V : ℝ) (hV : V > 0) : 
  actual_time a V > planned_time a V :=
by 
  sorry

end actual_time_greater_than_planned_time_l207_207549


namespace find_C_l207_207077

noncomputable def A : Set ℝ := {x | x^2 - 5 * x + 6 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x < a}
def isSolutionC (C : Set ℝ) : Prop := C = {2, 3}

theorem find_C : ∃ C : Set ℝ, isSolutionC C ∧ ∀ a, (A ∪ B a = A) ↔ a ∈ C :=
by
  sorry

end find_C_l207_207077


namespace find_a_range_l207_207594

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then -(x - 1) ^ 2 else (3 - a) * x + 4 * a

theorem find_a_range (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → (f x₁ a - f x₂ a) / (x₁ - x₂) > 0) ↔ (-1 ≤ a ∧ a < 3) :=
sorry

end find_a_range_l207_207594


namespace range_of_x_l207_207482

variable (x : ℝ)

theorem range_of_x (h1 : 2 - x > 0) (h2 : x - 1 ≥ 0) : 1 ≤ x ∧ x < 2 := by
  sorry

end range_of_x_l207_207482


namespace purely_imaginary_m_no_m_in_fourth_quadrant_l207_207496

def z (m : ℝ) : ℂ := ⟨m^2 - 8 * m + 15, m^2 - 5 * m⟩

theorem purely_imaginary_m :
  (∀ m : ℝ, z m = ⟨0, m^2 - 5 * m⟩ ↔ m = 3) :=
by
  sorry

theorem no_m_in_fourth_quadrant :
  ¬ ∃ m : ℝ, (m^2 - 8 * m + 15 > 0) ∧ (m^2 - 5 * m < 0) :=
by
  sorry

end purely_imaginary_m_no_m_in_fourth_quadrant_l207_207496


namespace doors_per_apartment_l207_207677

def num_buildings : ℕ := 2
def num_floors_per_building : ℕ := 12
def num_apt_per_floor : ℕ := 6
def total_num_doors : ℕ := 1008

theorem doors_per_apartment : total_num_doors / (num_buildings * num_floors_per_building * num_apt_per_floor) = 7 :=
by
  sorry

end doors_per_apartment_l207_207677


namespace geometric_series_common_ratio_l207_207977

theorem geometric_series_common_ratio (a r : ℝ) (h₁ : r ≠ 1)
    (h₂ : a / (1 - r) = 64 * (a * r^4) / (1 - r)) : r = 1/2 :=
by
  have h₃ : 1 = 64 * r^4 := by
    have : 1 - r ≠ 0 := by linarith
    field_simp at h₂; assumption
  sorry

end geometric_series_common_ratio_l207_207977


namespace g_at_5_l207_207810

def g (x : ℝ) : ℝ := sorry

axiom functional_equation : ∀ (x : ℝ), g x + 2 * g (1 - x) = x^2 + 2 * x

theorem g_at_5 : g 5 = -19 / 3 :=
by {
  sorry
}

end g_at_5_l207_207810


namespace train_length_l207_207526

-- Defining the conditions
def speed_kmh : ℕ := 64
def speed_m_per_s : ℚ := (64 * 1000) / 3600 -- 64 km/h converted to m/s
def time_to_cross_seconds : ℕ := 9 

-- The theorem to prove the length of the train
theorem train_length : speed_m_per_s * time_to_cross_seconds = 160 := 
by 
  unfold speed_m_per_s 
  norm_num
  sorry -- Placeholder for actual proof

end train_length_l207_207526


namespace num_distinct_sums_of_three_distinct_elements_l207_207895

noncomputable def arith_seq_sum_of_three_distinct : Nat :=
  let a (i : Nat) : Nat := 3 * i + 1
  let lower_bound := 21
  let upper_bound := 129
  (upper_bound - lower_bound) / 3 + 1

theorem num_distinct_sums_of_three_distinct_elements : arith_seq_sum_of_three_distinct = 37 := by
  -- We are skipping the proof by using sorry
  sorry

end num_distinct_sums_of_three_distinct_elements_l207_207895


namespace kim_knit_sweaters_total_l207_207215

theorem kim_knit_sweaters_total :
  ∀ (M T W R F : ℕ), 
    M = 8 →
    T = M + 2 →
    W = T - 4 →
    R = T - 4 →
    F = M / 2 →
    M + T + W + R + F = 34 :=
by
  intros M T W R F hM hT hW hR hF
  rw [hM, hT, hW, hR, hF]
  norm_num
  sorry

end kim_knit_sweaters_total_l207_207215


namespace prob_pass_kth_intersection_l207_207386

variable {n k : ℕ}

-- Definitions based on problem conditions
def prob_approach_highway (n : ℕ) : ℚ := 1 / n
def prob_exit_highway (n : ℕ) : ℚ := 1 / n

-- Theorem stating the required probability
theorem prob_pass_kth_intersection (h_n : n > 0) (h_k : k > 0) (h_k_le_n : k ≤ n) :
  (prob_approach_highway n) * (prob_exit_highway n * n) * (2 * k - 1) / n ^ 2 = 
  (2 * k * n - 2 * k ^ 2 + 2 * k - 1) / n ^ 2 := sorry

end prob_pass_kth_intersection_l207_207386


namespace factor_polynomial_l207_207699

theorem factor_polynomial (x y : ℝ) : 
  (x^2 - 2*x*y + y^2 - 16) = (x - y + 4) * (x - y - 4) :=
sorry

end factor_polynomial_l207_207699


namespace probability_at_least_one_die_less_than_3_l207_207123

theorem probability_at_least_one_die_less_than_3 :
  let total_outcomes := 8 * 8,
      favorable_outcomes := total_outcomes - (6 * 6)
  in (favorable_outcomes / total_outcomes : ℚ) = 7 / 16 := by
  sorry

end probability_at_least_one_die_less_than_3_l207_207123


namespace probability_different_tens_digit_l207_207471

open Nat

theorem probability_different_tens_digit :
  let total_ways := choose 70 6,
      favorable_ways := 7 * 10^6
  in 
    (favorable_ways : ℝ) / total_ways = (2000 / 3405864 : ℝ) :=
by
  have h1 : total_ways = 70.choose 6 := rfl
  have h2 : favorable_ways = 7 * 10^6 := rfl
  rw [h1, h2]
  sorry

end probability_different_tens_digit_l207_207471


namespace difference_of_numbers_l207_207821

variables (x y : ℝ)

-- Definitions corresponding to the conditions
def sum_of_numbers (x y : ℝ) : Prop := x + y = 30
def product_of_numbers (x y : ℝ) : Prop := x * y = 200

-- The proof statement in Lean
theorem difference_of_numbers (x y : ℝ) 
  (h1: sum_of_numbers x y) 
  (h2: product_of_numbers x y) : x - y = 10 ∨ y - x = 10 :=
by
  sorry

end difference_of_numbers_l207_207821


namespace inverse_linear_intersection_l207_207887

theorem inverse_linear_intersection (m n : ℝ) 
  (h1 : n = 2 / m) 
  (h2 : n = m + 3) 
  : (1 / m) - (1 / n) = 3 / 2 := 
by sorry

end inverse_linear_intersection_l207_207887


namespace parameterized_line_solution_l207_207812

theorem parameterized_line_solution :
  ∃ s l : ℝ, s = 1 / 2 ∧ l = -10 ∧
    ∀ t : ℝ, ∃ x y : ℝ,
      (x = -7 + t * l → y = s + t * (-5)) ∧ (y = (1 / 2) * x + 4) :=
by
  sorry

end parameterized_line_solution_l207_207812


namespace star_value_when_c_2_d_3_l207_207183

def star (c d : ℕ) : ℕ := c^3 + 3*c^2*d + 3*c*d^2 + d^3

theorem star_value_when_c_2_d_3 :
  star 2 3 = 125 :=
by
  sorry

end star_value_when_c_2_d_3_l207_207183


namespace completing_the_square_l207_207125

theorem completing_the_square :
  ∃ d, (∀ x: ℝ, (x^2 - 6 * x + 5 = 0) → ((x - 3)^2 = d)) ∧ d = 4 :=
by
  -- proof goes here
  sorry

end completing_the_square_l207_207125


namespace find_k_l207_207174

theorem find_k (k : ℝ) : (∀ x y : ℝ, (x + k * y - 2 * k = 0) → (k * x - (k - 2) * y + 1 = 0) → x * k + y * (-1 / k) + y * 2 = 0) →
  (k = 0 ∨ k = 3) :=
by
  sorry

end find_k_l207_207174


namespace largest_whole_number_l207_207582

theorem largest_whole_number :
  ∃ x : ℕ, 9 * x - 8 < 130 ∧ (∀ y : ℕ, 9 * y - 8 < 130 → y ≤ x) ∧ x = 15 :=
sorry

end largest_whole_number_l207_207582


namespace find_number_l207_207132

noncomputable def percentage_of (p : ℝ) (n : ℝ) := p / 100 * n

noncomputable def fraction_of (f : ℝ) (n : ℝ) := f * n

theorem find_number :
  ∃ x : ℝ, percentage_of 40 60 = fraction_of (4/5) x + 4 ∧ x = 25 :=
by
  sorry

end find_number_l207_207132


namespace teacher_total_score_l207_207147

variable (written_score : ℕ)
variable (interview_score : ℕ)
variable (weight_written : ℝ)
variable (weight_interview : ℝ)

theorem teacher_total_score :
  (written_score = 80) → (interview_score = 60) → (weight_written = 0.6) → (weight_interview = 0.4) →
  (written_score * weight_written + interview_score * weight_interview = 72) :=
by
  sorry

end teacher_total_score_l207_207147


namespace line_tangent_to_circle_l207_207939

theorem line_tangent_to_circle (x y : ℝ) :
  (3 * x - 4 * y + 25 = 0) ∧ (x^2 + y^2 = 25) → (x = -3 ∧ y = 4) :=
by sorry

end line_tangent_to_circle_l207_207939


namespace problem1_problem2_solution_l207_207850

noncomputable def trig_expr : ℝ :=
  3 * Real.tan (30 * Real.pi / 180) - (Real.tan (45 * Real.pi / 180))^2 + 2 * Real.sin (60 * Real.pi / 180)

theorem problem1 : trig_expr = 2 * Real.sqrt 3 - 1 :=
by
  -- Proof omitted
  sorry

noncomputable def quad_eq (x : ℝ) : Prop := 
  (3*x - 1) * (x + 2) = 11*x - 4

theorem problem2_solution (x : ℝ) : quad_eq x ↔ (x = (3 + Real.sqrt 3) / 3 ∨ x = (3 - Real.sqrt 3) / 3) :=
by
  -- Proof omitted
  sorry

end problem1_problem2_solution_l207_207850


namespace next_to_Denis_l207_207108

def student := {name : String}

def Borya : student := {name := "Borya"}
def Anya : student := {name := "Anya"}
def Vera : student := {name := "Vera"}
def Gena : student := {name := "Gena"}
def Denis : student := {name := "Denis"}

variables l : list student

axiom h₁ : l.head = Borya
axiom h₂ : ∃ i, l.get? i = some Vera ∧ (l.get? (i - 1) = some Anya ∨ l.get? (i + 1) = some Anya) ∧ l.get? (i - 1) ≠ some Gena ∧ l.get? (i + 1) ≠ some Gena
axiom h₃ : ∀ i j, l.get? i ∈ [some Anya, some Gena, some Borya] → l.get? j ∈ [some Anya, some Gena, some Borya] → abs (i - j) ≠ 1

theorem next_to_Denis : ∃ i, l.get? i = some Denis → (l.get? (i - 1) = some Anya ∨ l.get? (i + 1) = some Anya) ∧ (l.get? (i - 1) = some Gena ∨ l.get? (i + 1) = some Gena) :=
sorry

end next_to_Denis_l207_207108


namespace milk_after_three_operations_l207_207665

-- Define the initial amount of milk and the proportion replaced each step
def initial_milk : ℝ := 100
def proportion_replaced : ℝ := 0.2

-- Define the amount of milk after each replacement operation
noncomputable def milk_after_n_operations (n : ℕ) (milk : ℝ) : ℝ :=
  if n = 0 then milk
  else (1 - proportion_replaced) * milk_after_n_operations (n - 1) milk

-- Define the statement about the amount of milk after three operations
theorem milk_after_three_operations : milk_after_n_operations 3 initial_milk = 51.2 :=
by
  sorry

end milk_after_three_operations_l207_207665


namespace y_completion_time_l207_207510

noncomputable def work_done (days : ℕ) (rate : ℚ) : ℚ := days * rate

theorem y_completion_time (X_days Y_remaining_days : ℕ) (X_rate Y_days : ℚ) :
  X_days = 40 →
  work_done 8 (1 / X_days) = 1 / 5 →
  work_done Y_remaining_days (4 / 5 / Y_remaining_days) = 4 / 5 →
  Y_days = 35 :=
by
  intros hX hX_work_done hY_work_done
  -- With the stated conditions, we should be able to conclude that Y_days is 35.
  sorry

end y_completion_time_l207_207510


namespace descent_time_on_moving_escalator_standing_l207_207384

theorem descent_time_on_moving_escalator_standing (l v_mont v_ek t : ℝ)
  (H1 : l / v_mont = 42)
  (H2 : l / (v_mont + v_ek) = 24)
  : t = 56 := by
  sorry

end descent_time_on_moving_escalator_standing_l207_207384


namespace stripe_area_l207_207382

-- Definitions based on conditions
def diameter : ℝ := 40
def stripe_width : ℝ := 4
def revolutions : ℝ := 3

-- The statement we want to prove
theorem stripe_area (π : ℝ) : 
  (revolutions * π * diameter * stripe_width) = 480 * π :=
by
  sorry

end stripe_area_l207_207382


namespace population_increase_l207_207761

theorem population_increase (P : ℕ)
  (birth_rate1_per_1000 : ℕ := 25)
  (death_rate1_per_1000 : ℕ := 12)
  (immigration_rate1 : ℕ := 15000)
  (birth_rate2_per_1000 : ℕ := 30)
  (death_rate2_per_1000 : ℕ := 8)
  (immigration_rate2 : ℕ := 30000)
  (pop_increase1_perc : ℤ := 200)
  (pop_increase2_perc : ℤ := 300) :
  (12 * P - P) / P * 100 = 1100 := by
  sorry

end population_increase_l207_207761


namespace product_equals_sum_only_in_two_cases_l207_207085

theorem product_equals_sum_only_in_two_cases (x y : ℤ) : 
  x * y = x + y ↔ (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = 2) :=
by 
  sorry

end product_equals_sum_only_in_two_cases_l207_207085


namespace circle_passing_through_points_eq_l207_207093

theorem circle_passing_through_points_eq :
  ∃ D E F, (∀ x y, x^2 + y^2 + D*x + E*y + F = 0 ↔ 
    (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) ∧
  (D = -4 ∧ E = -6 ∧ F = 0) :=
begin
  sorry
end

end circle_passing_through_points_eq_l207_207093


namespace optimal_garden_dimensions_l207_207345

theorem optimal_garden_dimensions :
  ∃ (l w : ℝ), (2 * l + 2 * w = 400 ∧
                l ≥ 100 ∧
                w ≥ 0 ∧ 
                l * w = 10000) :=
by
  sorry

end optimal_garden_dimensions_l207_207345


namespace mark_siblings_l207_207080

theorem mark_siblings (total_eggs : ℕ) (eggs_per_person : ℕ) (persons_including_mark : ℕ) (h1 : total_eggs = 24) (h2 : eggs_per_person = 6) (h3 : persons_including_mark = total_eggs / eggs_per_person) : persons_including_mark - 1 = 3 :=
by 
  sorry

end mark_siblings_l207_207080


namespace f_satisfies_conditions_l207_207854

def g (n : Int) : Int :=
  if n >= 1 then 1 else 0

def f (n m : Int) : Int :=
  if m = 0 then n
  else n % m

theorem f_satisfies_conditions (n m : Int) : 
  (f 0 m = 0) ∧ 
  (f (n + 1) m = (1 - g m + g m * g (m - 1 - f n m)) * (1 + f n m)) := by
  sorry

end f_satisfies_conditions_l207_207854


namespace rectangle_area_in_cm_l207_207290

theorem rectangle_area_in_cm (length_in_m : ℝ) (width_in_m : ℝ) 
  (h_length : length_in_m = 0.5) (h_width : width_in_m = 0.36) : 
  (100 * length_in_m) * (100 * width_in_m) = 1800 :=
by
  -- We skip the proof for now
  sorry

end rectangle_area_in_cm_l207_207290


namespace sufficient_condition_for_perpendicular_l207_207292

variables (m n : Line) (α β : Plane)

def are_parallel (l1 l2 : Line) : Prop := sorry
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry

theorem sufficient_condition_for_perpendicular :
  (are_parallel m n) ∧ (line_perpendicular_to_plane n α) → (line_perpendicular_to_plane m α) :=
sorry

end sufficient_condition_for_perpendicular_l207_207292


namespace regular_polygon_sides_l207_207562

theorem regular_polygon_sides (n : ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → (interior_angle i = 160)) : n = 18 :=
by
  -- Proof goes here
  sorry

end regular_polygon_sides_l207_207562


namespace nancy_hourly_wage_l207_207778

theorem nancy_hourly_wage 
  (tuition_per_semester : ℕ := 22000) 
  (parents_cover : ℕ := tuition_per_semester / 2) 
  (scholarship : ℕ := 3000) 
  (student_loan : ℕ := 2 * scholarship) 
  (work_hours : ℕ := 200) 
  (remaining_tuition : ℕ := parents_cover - scholarship - student_loan) :
  (remaining_tuition / work_hours = 10) :=
  by
  sorry

end nancy_hourly_wage_l207_207778


namespace not_possible_odd_sum_l207_207747

theorem not_possible_odd_sum (m n : ℤ) (h : (m ^ 2 + n ^ 2) % 2 = 0) : (m + n) % 2 ≠ 1 :=
sorry

end not_possible_odd_sum_l207_207747


namespace trapezoid_inscribed_circles_radii_l207_207424

open Real

variables (a b m n : ℝ)
noncomputable def r := (a * sqrt b) / (sqrt a + sqrt b)
noncomputable def R := (b * sqrt a) / (sqrt a + sqrt b)

theorem trapezoid_inscribed_circles_radii
  (h : a < b)
  (hM : m = sqrt (a * b))
  (hN : m = sqrt (a * b)) :
  (r a b = (a * sqrt b) / (sqrt a + sqrt b)) ∧
  (R a b = (b * sqrt a) / (sqrt a + sqrt b)) :=
by
  sorry

end trapezoid_inscribed_circles_radii_l207_207424


namespace probability_A_does_not_lose_l207_207366

theorem probability_A_does_not_lose (p_tie p_A_win : ℚ) (h_tie : p_tie = 1 / 2) (h_A_win : p_A_win = 1 / 3) :
  p_tie + p_A_win = 5 / 6 :=
by sorry

end probability_A_does_not_lose_l207_207366


namespace two_digit_numbers_non_repeating_l207_207598

-- The set of available digits is given as 0, 1, 2, 3, 4
def digits : List ℕ := [0, 1, 2, 3, 4]

-- Ensure the tens place digits are subset of 1, 2, 3, 4 (exclude 0)
def valid_tens : List ℕ := [1, 2, 3, 4]

theorem two_digit_numbers_non_repeating :
  let num_tens := valid_tens.length
  let num_units := (digits.length - 1)
  num_tens * num_units = 16 :=
by
  -- Observe num_tens = 4, since valid_tens = [1, 2, 3, 4]
  -- Observe num_units = 4, since digits.length = 5 and we exclude the tens place digit
  sorry

end two_digit_numbers_non_repeating_l207_207598


namespace fraction_1790s_l207_207920

def total_states : ℕ := 30
def states_1790s : ℕ := 16

theorem fraction_1790s : (states_1790s / total_states : ℚ) = 8 / 15 :=
by
  -- We claim that the fraction of states admitted during the 1790s is exactly 8/15
  sorry

end fraction_1790s_l207_207920


namespace Denis_next_to_Anya_Gena_l207_207116

inductive Person
| Anya
| Borya
| Vera
| Gena
| Denis

open Person

def isNextTo (p1 p2 : Person) (line : List Person) : Prop :=
  ∃ n, line.nth n = some p1 ∧ line.nth (n + 1) = some p2 ∨ line.nth (n - 1) = some p2

def notNextTo (p1 p2 : Person) (line : List Person) : Prop :=
  ¬ isNextTo p1 p2 line

theorem Denis_next_to_Anya_Gena :
  ∀ (line : List Person),
    line.length = 5 →
    line.head = some Borya →
    isNextTo Vera Anya line →
    notNextTo Vera Gena line →
    notNextTo Anya Borya line →
    notNextTo Gena Borya line →
    notNextTo Anya Gena line →
    isNextTo Denis Anya line ∧ isNextTo Denis Gena line :=
by
  intros line length_five borya_head vera_next_anya vera_not_next_gena anya_not_next_borya gena_not_next_borya anya_not_next_gena
  sorry

end Denis_next_to_Anya_Gena_l207_207116


namespace exists_fraction_x_only_and_f_of_1_is_0_l207_207652

theorem exists_fraction_x_only_and_f_of_1_is_0 : ∃ f : ℚ → ℚ, (∀ x : ℚ, f x = (x - 1) / x) ∧ f 1 = 0 := 
by
  sorry

end exists_fraction_x_only_and_f_of_1_is_0_l207_207652


namespace hexagon_colorings_correct_l207_207011

def valid_hexagon_colorings : Prop :=
  ∃ (colors : Fin 6 → Fin 7),
    (colors 0 ≠ colors 1) ∧
    (colors 1 ≠ colors 2) ∧
    (colors 2 ≠ colors 3) ∧
    (colors 3 ≠ colors 4) ∧
    (colors 4 ≠ colors 5) ∧
    (colors 5 ≠ colors 0) ∧
    (colors 0 ≠ colors 2) ∧
    (colors 1 ≠ colors 3) ∧
    (colors 2 ≠ colors 4) ∧
    (colors 3 ≠ colors 5) ∧
    ∃! (n : Nat), n = 12600

theorem hexagon_colorings_correct : valid_hexagon_colorings :=
sorry

end hexagon_colorings_correct_l207_207011


namespace same_solution_m_l207_207877

theorem same_solution_m (m x : ℤ) : 
  (8 - m = 2 * (x + 1)) ∧ (2 * (2 * x - 3) - 1 = 1 - 2 * x) → m = 10 / 3 :=
by
  sorry

end same_solution_m_l207_207877


namespace cosine_of_angle_in_third_quadrant_l207_207905

theorem cosine_of_angle_in_third_quadrant (B : ℝ) (hB : B ∈ Set.Ioo (π : ℝ) (3 * π / 2)) (hSinB : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
sorry

end cosine_of_angle_in_third_quadrant_l207_207905


namespace problem1_problem2_l207_207863

theorem problem1 :
  0.064 ^ (-1 / 3) - (-7 / 8) ^ 0 + 16 ^ 0.75 + 0.01 ^ (1 / 2) = 48 / 5 :=
by sorry

theorem problem2 :
  2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 
  - 25 ^ (Real.log 3 / Real.log 5) = -7 :=
by sorry

end problem1_problem2_l207_207863


namespace max_bus_capacity_l207_207065

-- Definitions and conditions
def left_side_regular_seats := 12
def left_side_priority_seats := 3
def right_side_regular_seats := 9
def right_side_priority_seats := 2
def right_side_wheelchair_space := 1
def regular_seat_capacity := 3
def priority_seat_capacity := 2
def back_row_seat_capacity := 7
def standing_capacity := 14

-- Definition of total bus capacity
def total_bus_capacity : ℕ :=
  (left_side_regular_seats * regular_seat_capacity) + 
  (left_side_priority_seats * priority_seat_capacity) + 
  (right_side_regular_seats * regular_seat_capacity) + 
  (right_side_priority_seats * priority_seat_capacity) + 
  back_row_seat_capacity + 
  standing_capacity

-- Theorem to prove
theorem max_bus_capacity : total_bus_capacity = 94 := by
  -- skipping the proof
  sorry

end max_bus_capacity_l207_207065


namespace combinedHeightCorrect_l207_207408

def empireStateBuildingHeightToTopFloor : ℕ := 1250
def empireStateBuildingAntennaHeight : ℕ := 204

def willisTowerHeightToTopFloor : ℕ := 1450
def willisTowerAntennaHeight : ℕ := 280

def oneWorldTradeCenterHeightToTopFloor : ℕ := 1368
def oneWorldTradeCenterAntennaHeight : ℕ := 408

def totalHeightEmpireStateBuilding := empireStateBuildingHeightToTopFloor + empireStateBuildingAntennaHeight
def totalHeightWillisTower := willisTowerHeightToTopFloor + willisTowerAntennaHeight
def totalHeightOneWorldTradeCenter := oneWorldTradeCenterHeightToTopFloor + oneWorldTradeCenterAntennaHeight

def combinedHeight := totalHeightEmpireStateBuilding + totalHeightWillisTower + totalHeightOneWorldTradeCenter

theorem combinedHeightCorrect : combinedHeight = 4960 := by
  sorry

end combinedHeightCorrect_l207_207408


namespace simplify_expression_l207_207791

theorem simplify_expression :
  (2^5 + 4^3) * (2^2 - (-2)^3)^8 = 96 * 12^8 :=
by
  sorry

end simplify_expression_l207_207791


namespace C_pow_eq_target_l207_207770

open Matrix

-- Define the specific matrix C
def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; -4, -1]

-- Define the target matrix for the formula we need to prove
def C_power_50 : Matrix (Fin 2) (Fin 2) ℤ := !![101, 50; -200, -99]

-- Prove that C^50 equals to the target matrix
theorem C_pow_eq_target (n : ℕ) (h : n = 50) : C ^ n = C_power_50 := by
  rw [h]
  sorry

end C_pow_eq_target_l207_207770


namespace geometric_series_common_ratio_l207_207973

theorem geometric_series_common_ratio :
  ∀ (a r : ℝ), (r ≠ 1) → 
  (∑' n, a * r^n = 64 * ∑' n, a * r^(n+4)) →
  r = 1 / 2 :=
by
  intros a r hnr heq
  have hsum1 : ∑' n, a * r^n = a / (1 - r) := sorry
  have hsum2 : ∑' n, a * r^(n+4) = a * r^4 / (1 - r) := sorry
  rw [hsum1, hsum2] at heq
  -- Further steps to derive r = 1/2 are omitted
  sorry

end geometric_series_common_ratio_l207_207973


namespace range_of_x_l207_207304

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) + 1

theorem range_of_x (x : ℝ) (h : f (2 * x - 1) + f (4 - x^2) > 2) : x ∈ Set.Ioo (-1 : ℝ) 3 :=
by
  sorry

end range_of_x_l207_207304


namespace parabola_vertex_l207_207638

theorem parabola_vertex (y x : ℝ) : y^2 - 4*y + 3*x + 7 = 0 → (x = -1 ∧ y = 2) := 
sorry

end parabola_vertex_l207_207638


namespace base9_first_digit_is_4_l207_207092

-- Define the base three representation of y
def y_base3 : Nat := 112211

-- Function to convert a given number from base 3 to base 10
def base3_to_base10 (n : Nat) : Nat :=
  let rec convert (n : Nat) (acc : Nat) (place : Nat) : Nat :=
    if n = 0 then acc
    else convert (n / 10) (acc + (n % 10) * (3 ^ place)) (place + 1)
  convert n 0 0

-- Compute the base 10 representation of y
def y_base10 : Nat := base3_to_base10 y_base3

-- Function to convert a given number from base 10 to base 9
def base10_to_base9 (n : Nat) : List Nat :=
  let rec convert (n : Nat) (acc : List Nat) : List Nat :=
    if n = 0 then acc
    else convert (n / 9) ((n % 9) :: acc)
  convert n []

-- Compute the base 9 representation of y as a list of digits
def y_base9 : List Nat := base10_to_base9 y_base10

-- Get the first digit (most significant digit) of the base 9 representation of y
def first_digit_base9 (digits : List Nat) : Nat :=
  digits.headD 0

-- The statement to prove
theorem base9_first_digit_is_4 : first_digit_base9 y_base9 = 4 := by sorry

end base9_first_digit_is_4_l207_207092


namespace remaining_students_average_l207_207805

theorem remaining_students_average
  (N : ℕ) (A : ℕ) (M : ℕ) (B : ℕ) (E : ℕ)
  (h1 : N = 20)
  (h2 : A = 80)
  (h3 : M = 5)
  (h4 : B = 50)
  (h5 : E = (N - M))
  : (N * A - M * B) / E = 90 :=
by
  -- Using sorries to skip the proof
  sorry

end remaining_students_average_l207_207805


namespace range_of_k_l207_207173

theorem range_of_k (k : ℝ) : (∀ x : ℝ, |x + 1| + |x - 2| > k) → k > 3 := 
sorry

end range_of_k_l207_207173


namespace prob_at_least_one_goes_l207_207164

-- Conditions: probabilities of the persons going to Beijing and independence
variable (P_A P_B P_C : ℚ) -- Probabilities of A, B, C going
variable (independence : independent {P_A, P_B, P_C})

-- The given probabilities
def P_A_prob : P_A = 1 / 3 := sorry
def P_B_prob : P_B = 1 / 4 := sorry
def P_C_prob : P_C = 1 / 5 := sorry

-- Proof goal: the probability of at least one of A, B, or C going to Beijing
theorem prob_at_least_one_goes (P_A P_B P_C : ℚ) (independence : independent {P_A, P_B, P_C})
    (h1 : P_A = 1 / 3) (h2 : P_B = 1 / 4) (h3 : P_C = 1 / 5) :
    1 - ((1 - P_A) * (1 - P_B) * (1 - P_C)) = 3 / 5 := by
  sorry

end prob_at_least_one_goes_l207_207164


namespace yen_per_pound_l207_207323

theorem yen_per_pound 
  (pounds_initial : ℕ) 
  (euros : ℕ) 
  (yen_initial : ℕ) 
  (pounds_per_euro : ℕ) 
  (yen_total : ℕ) 
  (hp : pounds_initial = 42) 
  (he : euros = 11) 
  (hy : yen_initial = 3000) 
  (hpe : pounds_per_euro = 2) 
  (hy_total : yen_total = 9400) 
  : (yen_total - yen_initial) / (pounds_initial + euros * pounds_per_euro) = 100 := 
by
  sorry

end yen_per_pound_l207_207323


namespace square_area_increase_l207_207815

theorem square_area_increase (s : ℕ) (h : (s = 5) ∨ (s = 10) ∨ (s = 15)) :
  (1.35^2 - 1) * 100 = 82.25 :=
by
  sorry

end square_area_increase_l207_207815


namespace nancy_hourly_wage_l207_207776

-- Definitions based on conditions
def tuition_per_semester : ℕ := 22000
def parents_contribution : ℕ := tuition_per_semester / 2
def scholarship_amount : ℕ := 3000
def loan_amount : ℕ := 2 * scholarship_amount
def nancy_contributions : ℕ := parents_contribution + scholarship_amount + loan_amount
def remaining_tuition : ℕ := tuition_per_semester - nancy_contributions
def total_working_hours : ℕ := 200

-- Theorem to prove based on the formulated problem
theorem nancy_hourly_wage :
  (remaining_tuition / total_working_hours) = 10 :=
by
  sorry

end nancy_hourly_wage_l207_207776


namespace distinct_9_pointed_stars_l207_207946

-- Define a function to count the distinct n-pointed stars for a given n
def count_distinct_stars (n : ℕ) : ℕ :=
  -- Functionality to count distinct stars will be implemented here
  sorry

-- Theorem stating the number of distinct 9-pointed stars
theorem distinct_9_pointed_stars : count_distinct_stars 9 = 2 :=
  sorry

end distinct_9_pointed_stars_l207_207946


namespace find_degree_of_alpha_l207_207605

theorem find_degree_of_alpha
  (x : ℝ)
  (alpha : ℝ := x + 40)
  (beta : ℝ := 3 * x - 40)
  (h_parallel : alpha + beta = 180) :
  alpha = 85 :=
by
  sorry

end find_degree_of_alpha_l207_207605


namespace x_y_divisible_by_3_l207_207943

theorem x_y_divisible_by_3
    (x y z t : ℤ)
    (h : x^3 + y^3 = 3 * (z^3 + t^3)) :
    (3 ∣ x) ∧ (3 ∣ y) :=
by sorry

end x_y_divisible_by_3_l207_207943


namespace evaluate_expression_l207_207402

theorem evaluate_expression : 8 * ((1 : ℚ) / 3)^3 - 1 = -19 / 27 := by
  sorry

end evaluate_expression_l207_207402


namespace minimum_sugar_amount_l207_207155

theorem minimum_sugar_amount (f s : ℕ) (h1 : f ≥ 9 + s / 2) (h2 : f ≤ 3 * s) : s ≥ 4 :=
by
  -- Provided conditions: f ≥ 9 + s / 2 and f ≤ 3 * s
  -- Goal: s ≥ 4
  sorry

end minimum_sugar_amount_l207_207155


namespace imaginary_part_of_conjugate_l207_207806

def complex_conjugate (z : ℂ) : ℂ := ⟨z.re, -z.im⟩

theorem imaginary_part_of_conjugate :
  ∀ (z : ℂ), z = (1+i)^2 / (1-i) → (complex_conjugate z).im = -1 :=
by
  sorry

end imaginary_part_of_conjugate_l207_207806


namespace square_binomial_unique_a_l207_207871

theorem square_binomial_unique_a (a : ℝ) : 
  (∃ r s : ℝ, (ax^2 - 8*x + 16) = (r*x + s)^2) ↔ a = 1 :=
by
  sorry

end square_binomial_unique_a_l207_207871


namespace right_triangle_BD_length_l207_207608

theorem right_triangle_BD_length (BC AC AD BD : ℝ ) (h_bc: BC = 1) (h_ac: AC = b) (h_ad: AD = 2) :
  BD = Real.sqrt (b^2 - 3) :=
by
  sorry

end right_triangle_BD_length_l207_207608


namespace product_mod_25_l207_207797

theorem product_mod_25 (m : ℕ) (h : 0 ≤ m ∧ m < 25) : 
  43 * 67 * 92 % 25 = 2 :=
by
  sorry

end product_mod_25_l207_207797


namespace dogwood_trees_initial_count_l207_207831

theorem dogwood_trees_initial_count 
  (dogwoods_today : ℕ) 
  (dogwoods_tomorrow : ℕ) 
  (final_dogwoods : ℕ)
  (total_planted : ℕ := dogwoods_today + dogwoods_tomorrow)
  (initial_dogwoods := final_dogwoods - total_planted)
  (h : dogwoods_today = 41)
  (h1 : dogwoods_tomorrow = 20)
  (h2 : final_dogwoods = 100) : 
  initial_dogwoods = 39 := 
by sorry

end dogwood_trees_initial_count_l207_207831


namespace minimum_value_of_expr_l207_207744

noncomputable def expr (x : ℝ) := 2 * x + 1 / (x + 3)

theorem minimum_value_of_expr (x : ℝ) (h : x > -3) :
  ∃ y, y = 2 * real.sqrt 2 - 6 ∧ ∀ z, z > -3 → expr z ≥ y := sorry

end minimum_value_of_expr_l207_207744


namespace matrix_not_invertible_l207_207163

noncomputable def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem matrix_not_invertible (x : ℝ) :
  determinant (2*x + 1) 9 (4 - x) 10 = 0 ↔ x = 26/29 := by
  sorry

end matrix_not_invertible_l207_207163


namespace raja_journey_distance_l207_207509

theorem raja_journey_distance
  (T : ℝ) (D : ℝ)
  (H1 : T = 10)
  (H2 : ∀ t1 t2, t1 = D / 42 ∧ t2 = D / 48 → T = t1 + t2) :
  D = 224 :=
by
  sorry

end raja_journey_distance_l207_207509


namespace nancy_hourly_wage_l207_207775

-- Definitions based on conditions
def tuition_per_semester : ℕ := 22000
def parents_contribution : ℕ := tuition_per_semester / 2
def scholarship_amount : ℕ := 3000
def loan_amount : ℕ := 2 * scholarship_amount
def nancy_contributions : ℕ := parents_contribution + scholarship_amount + loan_amount
def remaining_tuition : ℕ := tuition_per_semester - nancy_contributions
def total_working_hours : ℕ := 200

-- Theorem to prove based on the formulated problem
theorem nancy_hourly_wage :
  (remaining_tuition / total_working_hours) = 10 :=
by
  sorry

end nancy_hourly_wage_l207_207775


namespace probability_A_or_B_complement_l207_207443

-- Define the sample space for rolling a die
def sample_space : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define Event A: the outcome is an even number not greater than 4
def event_A : Finset ℕ := {2, 4}

-- Define Event B: the outcome is less than 6
def event_B : Finset ℕ := {1, 2, 3, 4, 5}

-- Define the complement of Event B
def event_B_complement : Finset ℕ := {6}

-- Mutually exclusive property of events A and B_complement
axiom mutually_exclusive (A B_complement: Finset ℕ) : A ∩ B_complement = ∅

-- Define the probability function
def probability (events: Finset ℕ) : ℚ := (events.card : ℚ) / (sample_space.card : ℚ)

-- Theorem stating the probability of event (A + B_complement)
theorem probability_A_or_B_complement : probability (event_A ∪ event_B_complement) = 1 / 2 :=
by 
  sorry

end probability_A_or_B_complement_l207_207443


namespace cube_construction_symmetry_l207_207518

/-- 
 The number of distinct ways to construct a 3x3x3 cube using 9 red, 9 white, and 9 blue unit 
 cubes (considering two constructions identical if one can be rotated to match the other) 
 is equal to the result given by applying Burnside's Lemma to the symmetry group of the cube.
-/ 
theorem cube_construction_symmetry :
  let G := SymmetricGroup.group 3;
  let fixed_points (g : G) : ℕ := sorry;
  (1 / G.card) * ∑ (g : G), fixed_points g = sorry := 
sorry

end cube_construction_symmetry_l207_207518


namespace kim_knit_sweaters_total_l207_207214

theorem kim_knit_sweaters_total :
  ∀ (M T W R F : ℕ), 
    M = 8 →
    T = M + 2 →
    W = T - 4 →
    R = T - 4 →
    F = M / 2 →
    M + T + W + R + F = 34 :=
by
  intros M T W R F hM hT hW hR hF
  rw [hM, hT, hW, hR, hF]
  norm_num
  sorry

end kim_knit_sweaters_total_l207_207214


namespace exponent_division_l207_207692

theorem exponent_division (h1 : 27 = 3^3) : 3^18 / 27^3 = 19683 := by
  sorry

end exponent_division_l207_207692


namespace geometric_series_ratio_half_l207_207958

theorem geometric_series_ratio_half (a r S : ℝ) (hS : S = a / (1 - r)) 
  (h_ratio : (ar^4) / (1 - r) = S / 64) : r = 1 / 2 :=
by
  sorry

end geometric_series_ratio_half_l207_207958


namespace floor_of_neg_sqrt_frac_l207_207025

theorem floor_of_neg_sqrt_frac :
  (Int.floor (-Real.sqrt (64 / 9)) = -3) :=
by
  sorry

end floor_of_neg_sqrt_frac_l207_207025


namespace part1_solution_part2_solution_l207_207929

noncomputable def f (x a : ℝ) := |x + a| + |x - a|

theorem part1_solution : (∀ x : ℝ, f x 1 ≥ 4 ↔ x ∈ Set.Iic (-2) ∨ x ∈ Set.Ici 2) := by
  sorry

theorem part2_solution : (∀ x : ℝ, f x a ≥ 6 → a ∈ Set.Iic (-3) ∨ a ∈ Set.Ici 3) := by
  sorry

end part1_solution_part2_solution_l207_207929


namespace digit_expression_equals_2021_l207_207663

theorem digit_expression_equals_2021 :
  ∃ (f : ℕ → ℕ), 
  (f 0 = 0 ∧
   f 1 = 1 ∧
   f 2 = 2 ∧
   f 3 = 3 ∧
   f 4 = 4 ∧
   f 5 = 5 ∧
   f 6 = 6 ∧
   f 7 = 7 ∧
   f 8 = 8 ∧
   f 9 = 9 ∧
   43 * (8 * 5 + 7) + 0 * 1 * 2 * 6 * 9 = 2021) :=
sorry

end digit_expression_equals_2021_l207_207663


namespace range_of_a_l207_207089

open Set

theorem range_of_a (a : ℝ) : (-3 < a ∧ a < -1) ↔ (∀ x, x < -1 ∨ 5 < x ∨ (a < x ∧ x < a+8)) :=
sorry

end range_of_a_l207_207089


namespace percentage_forgot_homework_l207_207102

def total_students_group_A : ℕ := 30
def total_students_group_B : ℕ := 50
def forget_percentage_A : ℝ := 0.20
def forget_percentage_B : ℝ := 0.12

theorem percentage_forgot_homework :
  let num_students_forgot_A := forget_percentage_A * total_students_group_A
  let num_students_forgot_B := forget_percentage_B * total_students_group_B
  let total_students_forgot := num_students_forgot_A + num_students_forgot_B
  let total_students := total_students_group_A + total_students_group_B
  let percentage_forgot := (total_students_forgot / total_students) * 100
  percentage_forgot = 15 := sorry

end percentage_forgot_homework_l207_207102


namespace milk_replacement_problem_l207_207860

theorem milk_replacement_problem (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 90)
  (h3 : (90 - x) - ((90 - x) * x / 90) = 72.9) : x = 9 :=
sorry

end milk_replacement_problem_l207_207860


namespace sequence_2018_value_l207_207185

theorem sequence_2018_value (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n + 1) - a n = (-1 / 2) ^ n) :
  a 2018 = (2 * (1 - (1 / 2) ^ 2018)) / 3 :=
by sorry

end sequence_2018_value_l207_207185


namespace minimum_value_of_g_gm_equal_10_implies_m_is_5_l207_207048

/-- Condition: Definition of the function y in terms of x and m -/
def y (x m : ℝ) : ℝ := x^2 + m * x - 4

/-- Theorem about finding the minimum value of g(m) -/
theorem minimum_value_of_g (m : ℝ) :
  ∃ g : ℝ, g = (if m ≥ -4 then 2 * m
      else if -8 < m ∧ m < -4 then -m^2 / 4 - 4
      else 4 * m + 12) := by
  sorry

/-- Theorem that if the minimum value of g(m) is 10, then m must be 5 -/
theorem gm_equal_10_implies_m_is_5 :
  ∃ m, (if m ≥ -4 then 2 * m
       else if -8 < m ∧ m < -4 then -m^2 / 4 - 4
       else 4 * m + 12) = 10 := by
  use 5
  sorry

end minimum_value_of_g_gm_equal_10_implies_m_is_5_l207_207048


namespace triangle_lengths_ce_l207_207203

theorem triangle_lengths_ce (AE BE CE : ℝ) (angle_AEB angle_BEC angle_CED : ℝ) (h1 : angle_AEB = 30)
  (h2 : angle_BEC = 45) (h3 : angle_CED = 45) (h4 : AE = 30) (h5 : BE = AE / 2) (h6 : CE = BE) : CE = 15 :=
by sorry

end triangle_lengths_ce_l207_207203


namespace mooncake_packaging_problem_l207_207364

theorem mooncake_packaging_problem :
  ∃ x y : ℕ, 9 * x + 4 * y = 35 ∧ x + y = 5 :=
by
  -- Proof is omitted
  sorry

end mooncake_packaging_problem_l207_207364


namespace polynomial_has_at_most_one_integer_root_l207_207926

theorem polynomial_has_at_most_one_integer_root (k : ℝ) :
  ∀ x y : ℤ, (x^3 - 24 * x + k = 0) ∧ (y^3 - 24 * y + k = 0) → x = y :=
by
  intros x y h
  sorry

end polynomial_has_at_most_one_integer_root_l207_207926


namespace problem_statement_l207_207910

def A := {x : ℝ | x * (x - 1) < 0}
def B := {y : ℝ | ∃ x : ℝ, y = x^2}

theorem problem_statement : A ⊆ {y : ℝ | y ≥ 0} :=
sorry

end problem_statement_l207_207910


namespace geometric_sequence_tenth_term_l207_207559

theorem geometric_sequence_tenth_term :
  let a := 4
  let r := (4 / 3 : ℚ)
  a * r ^ 9 = (1048576 / 19683 : ℚ) :=
by
  sorry

end geometric_sequence_tenth_term_l207_207559


namespace closest_integer_to_cube_root_of_150_l207_207497

theorem closest_integer_to_cube_root_of_150 : ∃ (n : ℤ), abs ((n: ℝ)^3 - 150) ≤ abs (((n + 1 : ℤ) : ℝ)^3 - 150) ∧
  abs ((n: ℝ)^3 - 150) ≤ abs (((n - 1 : ℤ) : ℝ)^3 - 150) ∧ n = 5 :=
by
  sorry

end closest_integer_to_cube_root_of_150_l207_207497


namespace boatman_current_speed_and_upstream_time_l207_207255

variables (v : ℝ) (v_T : ℝ) (t_up : ℝ) (t_total : ℝ) (dist : ℝ) (d1 : ℝ) (d2 : ℝ)

theorem boatman_current_speed_and_upstream_time
  (h1 : dist = 12.5)
  (h2 : d1 = 3)
  (h3 : d2 = 5)
  (h4 : t_total = 8)
  (h5 : ∀ t, t = d1 / (v - v_T))
  (h6 : ∀ t, t = d2 / (v + v_T))
  (h7 : dist / (v - v_T) + dist / (v + v_T) = t_total) :
  v_T = 5 / 6 ∧ t_up = 5 := by
  sorry

end boatman_current_speed_and_upstream_time_l207_207255


namespace no_playful_two_digit_numbers_l207_207859

def is_playful (a b : ℕ) : Prop := 10 * a + b = a^3 + b^2

theorem no_playful_two_digit_numbers :
  (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → ¬ is_playful a b) :=
by {
  sorry
}

end no_playful_two_digit_numbers_l207_207859


namespace ratio_diff_squares_eq_16_l207_207400

theorem ratio_diff_squares_eq_16 (x y : ℕ) (h1 : x + y = 16) (h2 : x ≠ y) :
  (x^2 - y^2) / (x - y) = 16 :=
by
  sorry

end ratio_diff_squares_eq_16_l207_207400


namespace range_of_a_l207_207303

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(1 + a * x) - x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f a (f a x) - x

theorem range_of_a (a : ℝ) : (F a 0 = 0 → F a e = 0) → 
  (0 < a ∧ a < (1 / (Real.exp 1 * Real.log 2))) :=
by
  sorry

end range_of_a_l207_207303


namespace train_speed_l207_207687

theorem train_speed (length : ℝ) (time : ℝ)
  (length_pos : length = 160) (time_pos : time = 8) : 
  (length / time) * 3.6 = 72 :=
by
  sorry

end train_speed_l207_207687


namespace range_and_intervals_of_f_l207_207713

noncomputable def f (x : ℝ) : ℝ := (1/3)^(x^2 - 2 * x - 3)

theorem range_and_intervals_of_f :
  (∀ y, y > 0 → y ≤ 81 → (∃ x : ℝ, f x = y)) ∧
  (∀ x y, x ≤ y → f x ≥ f y) ∧
  (∀ x y, x ≥ y → f x ≤ f y) :=
by
  sorry

end range_and_intervals_of_f_l207_207713


namespace measure_of_angle_A_l207_207177

variable (a b c A B C : ℝ)

noncomputable def problem_statement : Prop :=
  (a + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C →
  A = Real.pi / 3

theorem measure_of_angle_A (h : problem_statement a b c A B C) : A = Real.pi / 3 :=
sorry

end measure_of_angle_A_l207_207177


namespace find_a3_l207_207289

noncomputable def geometric_term (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * q^(n-1)

noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * (q^n - 1) / (q - 1)

theorem find_a3 (a : ℝ) (q : ℝ) (h_q : q = 3)
  (h_sum : geometric_sum a q 3 + geometric_sum a q 4 = 53 / 3) :
  geometric_term a q 3 = 3 :=
by
  sorry

end find_a3_l207_207289


namespace clothing_prices_and_purchase_plans_l207_207607

theorem clothing_prices_and_purchase_plans :
  ∃ (x y : ℕ) (a : ℤ), 
  x + y = 220 ∧
  6 * x = 5 * y ∧
  120 * a + 100 * (150 - a) ≤ 17000 ∧
  (90 ≤ a ∧ a ≤ 100) ∧
  x = 100 ∧
  y = 120 ∧
  (∀ b : ℤ, (90 ≤ b ∧ b ≤ 100) → 120 * b + 100 * (150 - b) ≥ 16800)
  :=
sorry

end clothing_prices_and_purchase_plans_l207_207607


namespace socks_selection_l207_207519

theorem socks_selection :
  let red_socks := 120
  let green_socks := 90
  let blue_socks := 70
  let black_socks := 50
  let yellow_socks := 30
  let total_socks :=  red_socks + green_socks + blue_socks + black_socks + yellow_socks 
  (∀ k : ℕ, k ≥ 1 → k ≤ total_socks → (∃ p : ℕ, p = 12 → (p ≥ k / 2)) → k = 28) :=
by
  sorry

end socks_selection_l207_207519


namespace minimum_value_of_f_l207_207711

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3) * Real.exp x

theorem minimum_value_of_f : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = -2 * Real.exp 1 :=
by
  sorry

end minimum_value_of_f_l207_207711


namespace kim_knit_sweaters_total_l207_207213

theorem kim_knit_sweaters_total :
  ∀ (M T W R F : ℕ), 
    M = 8 →
    T = M + 2 →
    W = T - 4 →
    R = T - 4 →
    F = M / 2 →
    M + T + W + R + F = 34 :=
by
  intros M T W R F hM hT hW hR hF
  rw [hM, hT, hW, hR, hF]
  norm_num
  sorry

end kim_knit_sweaters_total_l207_207213


namespace first_discount_percentage_l207_207258

theorem first_discount_percentage (d : ℝ) (h : d > 0) :
  (∃ x : ℝ, (0 < x) ∧ (x < 100) ∧ 0.6 * d = (d * (1 - x / 100)) * 0.8) → x = 25 :=
by
  sorry

end first_discount_percentage_l207_207258


namespace meal_cost_before_tax_and_tip_l207_207916

theorem meal_cost_before_tax_and_tip (total_expenditure : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (base_meal_cost : ℝ):
  total_expenditure = 35.20 →
  tax_rate = 0.08 →
  tip_rate = 0.18 →
  base_meal_cost * (1 + tax_rate + tip_rate) = total_expenditure →
  base_meal_cost = 28 :=
by
  intros h_total h_tax h_tip h_eq
  sorry

end meal_cost_before_tax_and_tip_l207_207916


namespace initial_number_of_boarders_l207_207814

theorem initial_number_of_boarders (B D : ℕ) (h1 : B / D = 2 / 5) (h2 : (B + 15) / D = 1 / 2) : B = 60 :=
by
  -- Proof needs to be provided here
  sorry

end initial_number_of_boarders_l207_207814


namespace jane_exercises_per_day_l207_207324

-- Conditions
variables (total_hours : ℕ) (total_weeks : ℕ) (days_per_week : ℕ)
variable (goal_achieved : total_hours = 40 ∧ total_weeks = 8 ∧ days_per_week = 5)

-- Statement
theorem jane_exercises_per_day : ∃ hours_per_day : ℕ, hours_per_day = (total_hours / total_weeks) / days_per_week :=
by
  sorry

end jane_exercises_per_day_l207_207324


namespace simplify_division_l207_207941

noncomputable def a := 5 * 10 ^ 10
noncomputable def b := 2 * 10 ^ 4 * 10 ^ 2

theorem simplify_division : a / b = 25000 := by
  sorry

end simplify_division_l207_207941


namespace axis_of_symmetry_parabola_l207_207894

theorem axis_of_symmetry_parabola : 
  ∀ (x : ℝ), 2 * (x - 3)^2 - 5 = 2 * (x - 3)^2 - 5 → (∃ h : ℝ, h = 3 ∧ ∀ x : ℝ, h = 3) :=
by
  sorry

end axis_of_symmetry_parabola_l207_207894


namespace value_of_f_neg_a_l207_207640

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by
  sorry

end value_of_f_neg_a_l207_207640


namespace maria_tom_weather_probability_l207_207228

noncomputable def probability_exactly_two_clear_days (p : ℝ) (n : ℕ) : ℝ :=
  (n.choose 2) * (p ^ (n - 2)) * ((1 - p) ^ 2)

theorem maria_tom_weather_probability :
  probability_exactly_two_clear_days 0.6 5 = 1080 / 3125 :=
by
  sorry

end maria_tom_weather_probability_l207_207228


namespace find_a_interval_l207_207709

theorem find_a_interval :
  ∀ {a : ℝ}, (∃ b x y : ℝ, x = abs (y + a) + 4 / a ∧ x^2 + y^2 + 24 + b * (2 * y + b) = 10 * x) ↔ (a < 0 ∨ a ≥ 2 / 3) :=
by {
  sorry
}

end find_a_interval_l207_207709


namespace regular_polygon_sides_l207_207573

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ α, α = 160 → ∑ x in range n, (180 - (180 - 160)) = 360) : n = 18 :=
by
  sorry

end regular_polygon_sides_l207_207573


namespace nancy_hourly_wage_l207_207780

theorem nancy_hourly_wage 
  (tuition_per_semester : ℕ := 22000) 
  (parents_cover : ℕ := tuition_per_semester / 2) 
  (scholarship : ℕ := 3000) 
  (student_loan : ℕ := 2 * scholarship) 
  (work_hours : ℕ := 200) 
  (remaining_tuition : ℕ := parents_cover - scholarship - student_loan) :
  (remaining_tuition / work_hours = 10) :=
  by
  sorry

end nancy_hourly_wage_l207_207780


namespace miniature_tower_height_l207_207555

theorem miniature_tower_height
  (actual_height : ℝ)
  (actual_volume : ℝ)
  (miniature_volume : ℝ)
  (actual_height_eq : actual_height = 60)
  (actual_volume_eq : actual_volume = 200000)
  (miniature_volume_eq : miniature_volume = 0.2) :
  ∃ (miniature_height : ℝ), miniature_height = 0.6 :=
by
  sorry

end miniature_tower_height_l207_207555


namespace regular_polygon_sides_l207_207560

theorem regular_polygon_sides (n : ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → (interior_angle i = 160)) : n = 18 :=
by
  -- Proof goes here
  sorry

end regular_polygon_sides_l207_207560


namespace nancy_hourly_wage_l207_207777

-- Definitions based on conditions
def tuition_per_semester : ℕ := 22000
def parents_contribution : ℕ := tuition_per_semester / 2
def scholarship_amount : ℕ := 3000
def loan_amount : ℕ := 2 * scholarship_amount
def nancy_contributions : ℕ := parents_contribution + scholarship_amount + loan_amount
def remaining_tuition : ℕ := tuition_per_semester - nancy_contributions
def total_working_hours : ℕ := 200

-- Theorem to prove based on the formulated problem
theorem nancy_hourly_wage :
  (remaining_tuition / total_working_hours) = 10 :=
by
  sorry

end nancy_hourly_wage_l207_207777


namespace refills_needed_l207_207556

theorem refills_needed 
  (cups_per_day : ℕ)
  (bottle_capacity_oz : ℕ)
  (oz_per_cup : ℕ)
  (total_oz : ℕ)
  (refills : ℕ)
  (h1 : cups_per_day = 12)
  (h2 : bottle_capacity_oz = 16)
  (h3 : oz_per_cup = 8)
  (h4 : total_oz = cups_per_day * oz_per_cup)
  (h5 : refills = total_oz / bottle_capacity_oz) :
  refills = 6 :=
by
  sorry

end refills_needed_l207_207556


namespace set_union_is_correct_l207_207051

noncomputable def M (a : ℝ) : Set ℝ := {3, 2^a}
noncomputable def N (a b : ℝ) : Set ℝ := {a, b}

variable (a b : ℝ)
variable (h₁ : M a ∩ N a b = {2})
variable (h₂ : ∃ a b, N a b = {1, 2} ∧ M a = {3, 2} ∧ M a ∪ N a b = {1, 2, 3})

theorem set_union_is_correct :
  M 1 ∪ N 1 2 = {1, 2, 3} :=
by
  sorry

end set_union_is_correct_l207_207051


namespace regular_polygon_sides_l207_207564

theorem regular_polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (160 : ℝ) = (180 * (n - 2)) / n) : n = 18 := 
by 
  sorry

end regular_polygon_sides_l207_207564


namespace find_m_l207_207427

theorem find_m (f : ℝ → ℝ) (m : ℝ) 
  (h_even : ∀ x, f (-x) = f x) 
  (h_fx : ∀ x, 0 < x → f x = 4^(m - x)) 
  (h_f_neg2 : f (-2) = 1/8) : 
  m = 1/2 := 
by 
  sorry

end find_m_l207_207427


namespace Petya_time_comparison_l207_207545

-- Define the conditions
variables (a V : ℝ) (hV : V > 0)

noncomputable def T_planned := a / V

noncomputable def T1 := a / (2.5 * V)

noncomputable def T2 := a / (1.6 * V)

noncomputable def T_real := T1 + T2

-- State the main theorem
theorem Petya_time_comparison (ha : a > 0) : T_real a V > T_planned a V :=
by
  -- Proof to be filled in
  sorry

end Petya_time_comparison_l207_207545


namespace roots_negative_and_bounds_find_possible_values_of_b_and_c_l207_207593

theorem roots_negative_and_bounds
  (b c x₁ x₂ x₁' x₂' : ℤ) 
  (h1 : x₁ * x₂ > 0) 
  (h2 : x₁' * x₂' > 0)
  (h3 : x₁^2 + b * x₁ + c = 0) 
  (h4 : x₂^2 + b * x₂ + c = 0) 
  (h5 : x₁'^2 + c * x₁' + b = 0) 
  (h6 : x₂'^2 + c * x₂' + b = 0) :
  x₁ < 0 ∧ x₂ < 0 ∧ x₁' < 0 ∧ x₂' < 0 ∧ (b - 1 ≤ c ∧ c ≤ b + 1) :=
by
  sorry


theorem find_possible_values_of_b_and_c 
  (b c : ℤ) 
  (h's : ∃ x₁ x₂ x₁' x₂', 
    x₁ * x₂ > 0 ∧ 
    x₁' * x₂' > 0 ∧ 
    (x₁^2 + b * x₁ + c = 0) ∧ 
    (x₂^2 + b * x₂ + c = 0) ∧ 
    (x₁'^2 + c * x₁' + b = 0) ∧ 
    (x₂'^2 + c * x₂' + b = 0)) :
  (b = 4 ∧ c = 4) ∨ 
  (b = 5 ∧ c = 6) ∨ 
  (b = 6 ∧ c = 5) :=
by
  sorry

end roots_negative_and_bounds_find_possible_values_of_b_and_c_l207_207593


namespace width_of_Carols_rectangle_l207_207864

theorem width_of_Carols_rectangle 
  (w : ℝ) 
  (h1 : 15 * w = 6 * 50) : w = 20 := 
by 
  sorry

end width_of_Carols_rectangle_l207_207864


namespace smallest_gcd_six_l207_207099

theorem smallest_gcd_six (x : ℕ) (hx1 : 70 ≤ x) (hx2 : x ≤ 90) (hx3 : Nat.gcd 24 x = 6) : x = 78 :=
by
  sorry

end smallest_gcd_six_l207_207099


namespace count_positive_multiples_of_7_ending_in_5_below_1500_l207_207054

theorem count_positive_multiples_of_7_ending_in_5_below_1500 : 
  ∃ n : ℕ, n = 21 ∧ (∀ k : ℕ, (k < 1500) → ((k % 7 = 0) ∧ (k % 10 = 5) → (∃ m : ℕ, k = 35 + 70 * m) ∧ (0 ≤ m) ∧ (m < 21))) :=
sorry

end count_positive_multiples_of_7_ending_in_5_below_1500_l207_207054


namespace mrs_hilt_additional_rocks_l207_207222

-- Definitions from the conditions
def total_rocks : ℕ := 125
def rocks_she_has : ℕ := 64
def additional_rocks_needed : ℕ := total_rocks - rocks_she_has

-- The theorem to prove the question equals the answer given the conditions
theorem mrs_hilt_additional_rocks : additional_rocks_needed = 61 := 
by
  sorry

end mrs_hilt_additional_rocks_l207_207222


namespace scientific_notation_120_million_l207_207800

theorem scientific_notation_120_million :
  120000000 = 1.2 * 10^7 :=
by
  sorry

end scientific_notation_120_million_l207_207800


namespace geometric_series_common_ratio_l207_207976

theorem geometric_series_common_ratio :
  ∀ (a r : ℝ), (r ≠ 1) → 
  (∑' n, a * r^n = 64 * ∑' n, a * r^(n+4)) →
  r = 1 / 2 :=
by
  intros a r hnr heq
  have hsum1 : ∑' n, a * r^n = a / (1 - r) := sorry
  have hsum2 : ∑' n, a * r^(n+4) = a * r^4 / (1 - r) := sorry
  rw [hsum1, hsum2] at heq
  -- Further steps to derive r = 1/2 are omitted
  sorry

end geometric_series_common_ratio_l207_207976


namespace problem_statement_l207_207031

-- Definition of operation nabla
def nabla (a b : ℕ) : ℕ :=
  (b * (2 * a + b - 1)) / 2

-- Main theorem statement
theorem problem_statement : nabla 2 (nabla 0 (nabla 1 7)) = 71859 :=
by
  -- Computational proof
  sorry

end problem_statement_l207_207031


namespace partI_partII_l207_207730

theorem partI (m : ℝ) (h1 : ∀ t : ℝ, |t + 3| - |t - 2| ≤ 6 * m - m^2) :
  1 ≤ m ∧ m ≤ 5 :=
sorry

noncomputable def lambda : ℝ := 5

theorem partII (x y z : ℝ) (h2 : 3 * x + 4 * y + 5 * z = lambda) :
  x^2 + y^2 + z^2 ≥ 1/2 :=
sorry

end partI_partII_l207_207730


namespace standing_next_to_Denis_l207_207111

def students := ["Anya", "Borya", "Vera", "Gena", "Denis"]

def positions (line : list string) := 
  ∀ (student : string), student ∈ students → ∃ (i : ℕ), line.nth i = some student

theorem standing_next_to_Denis (line : list string) 
  (h1 : ∃ (i : ℕ), line.nth i = some "Borya" ∧ i = 0)
  (h2 : ∃ (i j k : ℕ), line.nth i = some "Anya" ∧ line.nth j = some "Vera" ∧ abs (i - j) = 1 ∧
       line.nth k = some "Gena" ∧ abs (j - k) ≠ 1)
  (h3 : ∃ (i j k : ℕ), line.nth i = some "Anya" ∧ line.nth k = some "Gena" ∧
       (abs (i - 0) ≥ 2 ∧ abs (k - 0) ≥ 2) ∧ abs (i - k) ≥ 2) :
  ∃ (a b : ℕ), line.nth a = some "Anya" ∧ line.nth b = some "Gena" ∧ 
  ∀ (d : ℕ), line.nth d = some "Denis" ∧ (abs (d - a) = 1 ∨ abs (d - b) = 1) :=
sorry

end standing_next_to_Denis_l207_207111


namespace no_such_integers_exists_l207_207873

theorem no_such_integers_exists (a b c : ℤ) (h_gcd : Int.gcd (Int.gcd a b) c = 1) :
  ¬ (a^2 + b^2 = 3 * c^2) := 
by {
  sorry
}

end no_such_integers_exists_l207_207873


namespace position_of_2017_in_arithmetic_sequence_l207_207657

theorem position_of_2017_in_arithmetic_sequence :
  ∀ (n : ℕ), 4 + 3 * (n - 1) = 2017 → n = 672 :=
by
  intros n h
  sorry

end position_of_2017_in_arithmetic_sequence_l207_207657


namespace geometric_series_ratio_half_l207_207957

theorem geometric_series_ratio_half (a r S : ℝ) (hS : S = a / (1 - r)) 
  (h_ratio : (ar^4) / (1 - r) = S / 64) : r = 1 / 2 :=
by
  sorry

end geometric_series_ratio_half_l207_207957


namespace water_remainder_l207_207678

theorem water_remainder (n : ℕ) (f : ℕ → ℚ) (h_init : f 1 = 1) 
  (h_recursive : ∀ k, k ≥ 2 → f k = f (k - 1) * (k^2 - 1) / k^2) :
  f 7 = 1 / 50 := 
sorry

end water_remainder_l207_207678


namespace solve_equation_l207_207851

variable (x : ℝ)

theorem solve_equation (h : x * (x - 4) = x - 6) : x = 2 ∨ x = 3 := 
sorry

end solve_equation_l207_207851


namespace ratio_mercedes_jonathan_l207_207326

theorem ratio_mercedes_jonathan (M : ℝ) (J : ℝ) (D : ℝ) 
  (h1 : J = 7.5) 
  (h2 : D = M + 2) 
  (h3 : M + D = 32) : M / J = 2 :=
by
  sorry

end ratio_mercedes_jonathan_l207_207326


namespace Petya_time_comparison_l207_207543

-- Define the conditions
variables (a V : ℝ) (hV : V > 0)

noncomputable def T_planned := a / V

noncomputable def T1 := a / (2.5 * V)

noncomputable def T2 := a / (1.6 * V)

noncomputable def T_real := T1 + T2

-- State the main theorem
theorem Petya_time_comparison (ha : a > 0) : T_real a V > T_planned a V :=
by
  -- Proof to be filled in
  sorry

end Petya_time_comparison_l207_207543


namespace range_of_expression_l207_207175

open Real

theorem range_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) :
  1 ≤ x^2 + y^2 + sqrt (x * y) ∧ x^2 + y^2 + sqrt (x * y) ≤ 9 / 8 :=
sorry

end range_of_expression_l207_207175


namespace c_share_of_profit_l207_207249

-- Definitions for the investments and total profit
def investments_a := 800
def investments_b := 1000
def investments_c := 1200
def total_profit := 1000

-- Definition for the share of profits based on the ratio of investments
def share_of_c : ℕ :=
  let ratio_a := 4
  let ratio_b := 5
  let ratio_c := 6
  let total_ratio := ratio_a + ratio_b + ratio_c
  (ratio_c * total_profit) / total_ratio

-- The theorem to be proved
theorem c_share_of_profit : share_of_c = 400 := by
  sorry

end c_share_of_profit_l207_207249


namespace volume_of_cone_l207_207273

theorem volume_of_cone (d h : ℝ) (d_eq : d = 12) (h_eq : h = 9) : 
  (1 / 3) * π * (d / 2)^2 * h = 108 * π := 
by 
  rw [d_eq, h_eq] 
  sorry

end volume_of_cone_l207_207273


namespace tangent_at_5_eqn_l207_207809

noncomputable def f : ℝ → ℝ := sorry

axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom f_period : ∀ x : ℝ, f (x + 2) = f (2 - x)
axiom tangent_at_neg1 : ∀ x y : ℝ, x - y + 3 = 0 → x = -1 → y = f x

theorem tangent_at_5_eqn : 
  ∀ x y : ℝ, x = 5 → y = f x → x + y - 7 = 0 :=
sorry

end tangent_at_5_eqn_l207_207809


namespace next_perfect_square_l207_207036

theorem next_perfect_square (n : ℤ) (hn : Even n) (x : ℤ) (hx : x = n^2) : 
  ∃ y : ℤ, y = x + 2 * n + 1 ∧ (∃ m : ℤ, y = m^2) ∧ m > n :=
by
  sorry

end next_perfect_square_l207_207036


namespace sock_pairs_l207_207832

theorem sock_pairs (n : ℕ) (h : ((2 * n) * (2 * n - 1)) / 2 = 90) : n = 10 :=
sorry

end sock_pairs_l207_207832


namespace trig_identity_proof_l207_207879

theorem trig_identity_proof
  (α : ℝ)
  (h : Real.sin (α - π / 6) = 3 / 5) :
  Real.cos (2 * π / 3 - α) = 3 / 5 :=
sorry

end trig_identity_proof_l207_207879


namespace sequence_geometric_l207_207452

noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  3 * a n - 2

theorem sequence_geometric (a : ℕ → ℝ) (h : ∀ n, S n a = 3 * a n - 2) :
  ∀ n, a n = (3/2)^(n-1) :=
by
  intro n
  sorry

end sequence_geometric_l207_207452


namespace jellybean_count_l207_207377

def black_beans : Nat := 8
def green_beans : Nat := black_beans + 2
def orange_beans : Nat := green_beans - 1
def total_jelly_beans : Nat := black_beans + green_beans + orange_beans

theorem jellybean_count : total_jelly_beans = 27 :=
by
  -- proof steps would go here.
  sorry

end jellybean_count_l207_207377


namespace slope_of_intersection_points_l207_207167

theorem slope_of_intersection_points : 
  (∀ t : ℝ, ∃ x y : ℝ, (2 * x + 3 * y = 10 * t + 4) ∧ (x + 4 * y = 3 * t + 3)) → 
  (∀ t1 t2 : ℝ, t1 ≠ t2 → ((2 * ((10 * t1 + 4)  / 2) + 3 * ((-5/3 * t1 - 2/3)) = (10 * t1 + 4)) ∧ (2 * ((10 * t2 + 4) / 2) + 3 * ((-5/3 * t2 - 2/3)) = (10 * t2 + 4))) → 
  (31 * (((-5/3 * t1 - 2/3) - (-5/3 * t2 - 2/3)) / ((10 * t1 + 4) / 2 - (10 * t2 + 4) / 2)) = -4)) :=
sorry

end slope_of_intersection_points_l207_207167


namespace geometric_series_common_ratio_l207_207965

theorem geometric_series_common_ratio (a r S : ℝ) (h₁ : S = a / (1 - r)) (h₂ : ar^4 / (1 - r) = S / 64) : r = 1 / 2 :=
  by
  sorry

end geometric_series_common_ratio_l207_207965


namespace dispatch_3_male_2_female_dispatch_at_least_2_male_l207_207236

-- Define the number of male and female drivers
def male_drivers : ℕ := 5
def female_drivers : ℕ := 4
def total_drivers_needed : ℕ := 5

-- Define the combination formula (binomial coefficient)
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- First part of the problem
theorem dispatch_3_male_2_female : 
  combination male_drivers 3 * combination female_drivers 2 = 60 :=
by sorry

-- Second part of the problem
theorem dispatch_at_least_2_male : 
  combination male_drivers 2 * combination female_drivers 3 + 
  combination male_drivers 3 * combination female_drivers 2 + 
  combination male_drivers 4 * combination female_drivers 1 + 
  combination male_drivers 5 * combination female_drivers 0 = 121 :=
by sorry

end dispatch_3_male_2_female_dispatch_at_least_2_male_l207_207236


namespace intersection_of_A_and_B_l207_207440

def setA (x : Real) : Prop := -1 < x ∧ x < 3
def setB (x : Real) : Prop := -2 < x ∧ x < 2

theorem intersection_of_A_and_B : {x : Real | setA x} ∩ {x : Real | setB x} = {x : Real | -1 < x ∧ x < 2} := 
by
  sorry

end intersection_of_A_and_B_l207_207440


namespace solve_eq1_solve_eq2_l207_207470

-- Define the first equation
def eq1 (x : ℝ) : Prop := x^2 - 2 * x - 1 = 0

-- Define the second equation
def eq2 (x : ℝ) : Prop := (x - 2)^2 = 2 * x - 4

-- State the first theorem
theorem solve_eq1 (x : ℝ) : eq1 x ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
by sorry

-- State the second theorem
theorem solve_eq2 (x : ℝ) : eq2 x ↔ (x = 2 ∨ x = 4) :=
by sorry

end solve_eq1_solve_eq2_l207_207470


namespace ratio_of_saturday_to_friday_customers_l207_207367

def tips_per_customer : ℝ := 2.0
def customers_friday : ℕ := 28
def customers_sunday : ℕ := 36
def total_tips : ℝ := 296

theorem ratio_of_saturday_to_friday_customers :
  let tips_friday := customers_friday * tips_per_customer
  let tips_sunday := customers_sunday * tips_per_customer
  let tips_friday_and_sunday := tips_friday + tips_sunday
  let tips_saturday := total_tips - tips_friday_and_sunday
  let customers_saturday := tips_saturday / tips_per_customer
  (customers_saturday / customers_friday : ℝ) = 3 := 
by
  sorry

end ratio_of_saturday_to_friday_customers_l207_207367


namespace max_crystalline_polyhedron_volume_l207_207684

theorem max_crystalline_polyhedron_volume (n : ℕ) (R : ℝ) (h_n : n > 1) :
  ∃ V : ℝ, 
    V = (32 / 81) * (n - 1) * (R ^ 3) * Real.sin (2 * Real.pi / (n - 1)) :=
sorry

end max_crystalline_polyhedron_volume_l207_207684


namespace cos_pi_over_8_cos_5pi_over_8_l207_207553

theorem cos_pi_over_8_cos_5pi_over_8 :
  (Real.cos (Real.pi / 8)) * (Real.cos (5 * Real.pi / 8)) = - (Real.sqrt 2 / 4) :=
by
  sorry

end cos_pi_over_8_cos_5pi_over_8_l207_207553


namespace proof_angles_proof_area_l207_207442

noncomputable def angles_in_ABC (A B C: ℝ) (a b c: ℝ) :=
  A = π / 4 ∧
  b * sin (π / 4 + C) - c * sin (π / 4 + B) = a ∧
  B = 5 * π / 8 ∧
  C = π / 8

noncomputable def area_of_ABC (A B C: ℝ) (a b c: ℝ) :=
  a = 2 * sqrt 2 ∧
  angles_in_ABC A B C a b c →
  let area := 1 / 2 * a * b * sin C
  in area = 2

theorem proof_angles (A B C: ℝ) (a b c: ℝ) :
  angles_in_ABC A B C a b c := 
sorry

theorem proof_area (A B C: ℝ) (a b c: ℝ) :
  area_of_ABC A B C a b c :=
sorry

end proof_angles_proof_area_l207_207442


namespace trig_identity_l207_207514

theorem trig_identity : 4 * Real.cos (50 * Real.pi / 180) - Real.tan (40 * Real.pi / 180) = Real.sqrt 3 :=
by sorry

end trig_identity_l207_207514


namespace smallest_x_for_multiple_of_450_and_648_l207_207659

theorem smallest_x_for_multiple_of_450_and_648 (x : ℕ) (hx : x > 0) :
  ∃ (y : ℕ), (450 * 36) = y ∧ (450 * 36) % 648 = 0 :=
by
  use (450 / gcd 450 648 * 648 / gcd 450 648)
  sorry

end smallest_x_for_multiple_of_450_and_648_l207_207659


namespace jason_worked_hours_on_saturday_l207_207764

def hours_jason_works (x y : ℝ) : Prop :=
  (4 * x + 6 * y = 88) ∧ (x + y = 18)

theorem jason_worked_hours_on_saturday (x y : ℝ) : hours_jason_works x y → y = 8 := 
by 
  sorry

end jason_worked_hours_on_saturday_l207_207764


namespace rectangular_prism_volume_l207_207141

theorem rectangular_prism_volume
  (l w h : ℝ)
  (face1 : l * w = 6)
  (face2 : w * h = 8)
  (face3 : l * h = 12) : l * w * h = 24 := sorry

end rectangular_prism_volume_l207_207141


namespace Kiera_envelopes_total_l207_207924

-- Define variables for different colored envelopes
def E_b : ℕ := 120
def E_y : ℕ := E_b - 25
def E_g : ℕ := 5 * E_y
def E_r : ℕ := (E_b + E_y) / 2  -- integer division in lean automatically rounds down
def E_p : ℕ := E_r + 71
def E_total : ℕ := E_b + E_y + E_g + E_r + E_p

-- The statement to be proven
theorem Kiera_envelopes_total : E_total = 975 := by
  -- intentionally put the sorry to mark the proof as unfinished
  sorry

end Kiera_envelopes_total_l207_207924


namespace part1_solution_part2_solution_l207_207043

-- Definitions based on the given conditions
def eq1 (x y m : ℝ) := 2 * x - y = m
def eq2 (x y m : ℝ) := 3 * x + 2 * y = m + 7

-- Problem Part 1: When m = 0, the solution to the system of equations
theorem part1_solution :
  ∃ (x y : ℝ), eq1 x y 0 ∧ eq2 x y 0 ∧ x = 1 ∧ y = 2 :=
by
  existsi 1
  existsi 2
  apply And.intro
  show eq1 1 2 0, by sorry
  apply And.intro
  show eq2 1 2 0, by sorry
  apply And.intro
  show 1 = 1, by rfl
  show 2 = 2, by rfl

-- Problem Part 2: When A(-2, 3), the value of m that satisfies the equations
theorem part2_solution :
  let x := -2
  let y := 3
  ∃ (m : ℝ), eq1 x y m ∧ m = -7 :=
by
  existsi (-7 : ℝ)
  apply And.intro
  show eq1 (-2) 3 (-7), by sorry
  show -7 = -7, by rfl

end part1_solution_part2_solution_l207_207043


namespace convert_to_scientific_notation_l207_207802

-- Problem statement: convert 120 million to scientific notation and validate the format.
theorem convert_to_scientific_notation :
  120000000 = 1.2 * 10^7 :=
sorry

end convert_to_scientific_notation_l207_207802


namespace range_of_e_l207_207039

theorem range_of_e (a b c d e : ℝ)
  (h1 : a + b + c + d + e = 8)
  (h2 : a^2 + b^2 + c^2 + d^2 + e^2 = 16) :
  0 ≤ e ∧ e ≤ 16 / 5 :=
by
  sorry

end range_of_e_l207_207039


namespace find_starting_number_l207_207988

-- Define that there are 15 even integers between a starting number and 40
def even_integers_range (n : ℕ) : Prop :=
  ∃ k : ℕ, (1 ≤ k) ∧ (k = 15) ∧ (n + 2*(k-1) = 40)

-- Proof statement
theorem find_starting_number : ∃ n : ℕ, even_integers_range n ∧ n = 12 :=
by
  sorry

end find_starting_number_l207_207988


namespace remainder_of_sum_l207_207075

theorem remainder_of_sum (a b c : ℕ) (h₁ : a * b * c % 7 = 1) (h₂ : 2 * c % 7 = 5) (h₃ : 3 * b % 7 = (4 + b) % 7) :
  (a + b + c) % 7 = 6 := by
  sorry

end remainder_of_sum_l207_207075


namespace cylinder_cone_volume_l207_207134

theorem cylinder_cone_volume (V_total : ℝ) (Vc Vcone : ℝ)
  (h1 : V_total = 48)
  (h2 : V_total = Vc + Vcone)
  (h3 : Vc = 3 * Vcone) :
  Vc = 36 ∧ Vcone = 12 :=
by
  sorry

end cylinder_cone_volume_l207_207134


namespace geometric_series_common_ratio_l207_207975

theorem geometric_series_common_ratio :
  ∀ (a r : ℝ), (r ≠ 1) → 
  (∑' n, a * r^n = 64 * ∑' n, a * r^(n+4)) →
  r = 1 / 2 :=
by
  intros a r hnr heq
  have hsum1 : ∑' n, a * r^n = a / (1 - r) := sorry
  have hsum2 : ∑' n, a * r^(n+4) = a * r^4 / (1 - r) := sorry
  rw [hsum1, hsum2] at heq
  -- Further steps to derive r = 1/2 are omitted
  sorry

end geometric_series_common_ratio_l207_207975


namespace abs_neg_one_ninth_l207_207476

theorem abs_neg_one_ninth : abs (- (1 / 9)) = 1 / 9 := by
  sorry

end abs_neg_one_ninth_l207_207476


namespace geometric_series_ratio_half_l207_207959

theorem geometric_series_ratio_half (a r S : ℝ) (hS : S = a / (1 - r)) 
  (h_ratio : (ar^4) / (1 - r) = S / 64) : r = 1 / 2 :=
by
  sorry

end geometric_series_ratio_half_l207_207959


namespace floor_of_negative_sqrt_l207_207020

noncomputable def eval_expr : ℚ := -real.sqrt (64 / 9)

theorem floor_of_negative_sqrt : ⌊eval_expr⌋ = -3 :=
by
  -- skip proof
  sorry

end floor_of_negative_sqrt_l207_207020


namespace subtraction_888_55_555_55_l207_207944

theorem subtraction_888_55_555_55 : 888.88 - 555.55 = 333.33 :=
by
  sorry

end subtraction_888_55_555_55_l207_207944


namespace cos_B_third_quadrant_l207_207903

theorem cos_B_third_quadrant (B : ℝ) (hB1 : π < B ∧ B < 3 * π / 2) (hB2 : sin B = -5 / 13) : cos B = -12 / 13 :=
by
  sorry

end cos_B_third_quadrant_l207_207903


namespace regular_polygon_sides_l207_207577

-- Define the problem conditions based on the given problem.
variables (n : ℕ)
def sum_of_interior_angles (n : ℕ) : ℤ := 180 * (n - 2)
def interior_angle (n : ℕ) : ℤ := 160
def total_interior_angle (n : ℕ) : ℤ := 160 * n

-- State the problem and expected result.
theorem regular_polygon_sides (h : 180 * (n - 2) = 160 * n) : n = 18 := 
by {
  -- This is to only state the theorem for now, proof will be crafted separately.
  sorry
}

end regular_polygon_sides_l207_207577


namespace students_can_do_both_l207_207829

variable (total_students swimmers gymnasts neither : ℕ)

theorem students_can_do_both (h1 : total_students = 60)
                             (h2 : swimmers = 27)
                             (h3 : gymnasts = 28)
                             (h4 : neither = 15) : 
                             total_students - (total_students - swimmers + total_students - gymnasts - neither) = 10 := 
by 
  sorry

end students_can_do_both_l207_207829


namespace probability_two_identical_l207_207653

-- Define the number of ways to choose 3 out of 4 attractions
def choose_3_out_of_4 := Nat.choose 4 3

-- Define the total number of ways for both tourists to choose 3 attractions out of 4
def total_basic_events := choose_3_out_of_4 * choose_3_out_of_4

-- Define the number of ways to choose exactly 2 identical attractions
def ways_to_choose_2_identical := Nat.choose 4 2 * Nat.choose 2 1 * Nat.choose 1 1

-- The probability that they choose exactly 2 identical attractions
def probability : ℚ := ways_to_choose_2_identical / total_basic_events

-- Prove that this probability is 3/4
theorem probability_two_identical : probability = 3 / 4 := by
  have h1 : choose_3_out_of_4 = 4 := by sorry
  have h2 : total_basic_events = 16 := by sorry
  have h3 : ways_to_choose_2_identical = 12 := by sorry
  rw [probability, h2, h3]
  norm_num

end probability_two_identical_l207_207653


namespace beneficial_for_kati_l207_207616

variables (n : ℕ) (x y : ℝ)

theorem beneficial_for_kati (hn : n > 0) (hx : x ≥ 0) (hy : y ≥ 0) :
  (x + y) / (n + 2) > (x + y / 2) / (n + 1) :=
sorry

end beneficial_for_kati_l207_207616


namespace value_of_a_minus_b_l207_207737

theorem value_of_a_minus_b (a b : ℝ) (h1 : 2 * a - b = 5) (h2 : a - 2 * b = 4) : a - b = 3 :=
by
  sorry

end value_of_a_minus_b_l207_207737


namespace product_xy_eq_3_l207_207178

variable {x y : ℝ}
variables (h₀ : x ≠ y) (h₁ : x ≠ 0) (h₂ : y ≠ 0)
variable (h₃ : x + (3 / x) = y + (3 / y))

theorem product_xy_eq_3 : x * y = 3 := by
  sorry

end product_xy_eq_3_l207_207178


namespace arithmetic_sequence_a1_l207_207172

theorem arithmetic_sequence_a1 
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ (k : ℕ), a (k + 1) = a k + d)
  (sum_first_100 : ∑ k in Finset.range 100, a k = 100)
  (sum_last_100 : ∑ k in Finset.range 100, a (k + 900) = 1000) :
  a 0 = 0.505 :=
sorry

end arithmetic_sequence_a1_l207_207172


namespace normal_distribution_interval_probability_l207_207199

noncomputable def normal_cdf (μ σ : ℝ) (x : ℝ) : ℝ :=
sorry

theorem normal_distribution_interval_probability
  (σ : ℝ) (hσ : σ > 0)
  (hprob : normal_cdf 1 σ 2 - normal_cdf 1 σ 0 = 0.8) :
  (normal_cdf 1 σ 2 - normal_cdf 1 σ 1) = 0.4 :=
sorry

end normal_distribution_interval_probability_l207_207199


namespace negative_comparison_l207_207558

theorem negative_comparison : -2023 > -2024 :=
sorry

end negative_comparison_l207_207558


namespace probability_selecting_girl_l207_207142

def boys : ℕ := 3
def girls : ℕ := 1
def total_candidates : ℕ := boys + girls
def favorable_outcomes : ℕ := girls

theorem probability_selecting_girl : 
  ∃ p : ℚ, p = (favorable_outcomes : ℚ) / (total_candidates : ℚ) ∧ p = 1 / 4 :=
sorry

end probability_selecting_girl_l207_207142


namespace apples_per_classmate_l207_207603

theorem apples_per_classmate 
  (total_apples : ℕ) 
  (people : ℕ) 
  (h : total_apples = 15) 
  (p : people = 3) : 
  total_apples / people = 5 :=
by
  rw [h, p]
  norm_num

end apples_per_classmate_l207_207603


namespace probability_sum_16_l207_207410

open ProbabilityTheory

noncomputable def coin_flip_probs : Finset ℚ := {5 , 15}
noncomputable def die_probs : Finset ℚ := {1, 2, 3, 4, 5, 6}

def fair_coin (x : ℚ) : ℚ := if x = 5 ∨ x = 15 then (1 : ℚ) / 2 else 0
def fair_die (x : ℚ) : ℚ := if x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 then (1 : ℚ) / 6 else 0

theorem probability_sum_16 : ∑ x in coin_flip_probs, ∑ y in die_probs, (if x + y = 16 then fair_coin x * fair_die y else 0) = 1 / 12 := 
    sorry

end probability_sum_16_l207_207410


namespace zero_point_six_six_six_is_fraction_l207_207168

def is_fraction (x : ℝ) : Prop := ∃ (n d : ℤ), d ≠ 0 ∧ x = (n : ℝ) / (d : ℝ)

theorem zero_point_six_six_six_is_fraction:
  let sqrt_2_div_3 := (Real.sqrt 2) / 3
  let neg_sqrt_4 := - Real.sqrt 4
  let zero_point_six_six_six := 0.666
  let one_seventh := 1 / 7
  is_fraction zero_point_six_six_six :=
by sorry

end zero_point_six_six_six_is_fraction_l207_207168


namespace total_cost_correct_l207_207844

/-- Define the base car rental cost -/
def rental_cost : ℝ := 150

/-- Define cost per mile -/
def cost_per_mile : ℝ := 0.5

/-- Define miles driven on Monday -/
def miles_monday : ℝ := 620

/-- Define miles driven on Thursday -/
def miles_thursday : ℝ := 744

/-- Define the total cost Zach spent -/
def total_cost : ℝ := rental_cost + (miles_monday * cost_per_mile) + (miles_thursday * cost_per_mile)

/-- Prove that the total cost Zach spent is 832 dollars -/
theorem total_cost_correct : total_cost = 832 := by
  sorry

end total_cost_correct_l207_207844


namespace h_comp_h_3_l207_207897

def h (x : ℕ) : ℕ := 3 * x * x + 5 * x - 3

theorem h_comp_h_3 : h (h 3) = 4755 := by
  sorry

end h_comp_h_3_l207_207897


namespace alberto_vs_bjorn_distance_difference_l207_207358

noncomputable def alberto_distance (t : ℝ) : ℝ := (3.75 / 5) * t
noncomputable def bjorn_distance (t : ℝ) : ℝ := (3.4375 / 5) * t

theorem alberto_vs_bjorn_distance_difference :
  alberto_distance 5 - bjorn_distance 5 = 0.3125 :=
by
  -- proof goes here
  sorry

end alberto_vs_bjorn_distance_difference_l207_207358


namespace regular_polygon_sides_l207_207570

-- Define the number of sides
def n : ℕ := sorry

-- The interior angle condition
def interior_angle_condition (n : ℕ) : Prop := 
  let interior_angle := (180 * (n - 2)) / n
  interior_angle = 160

-- Prove the statement
theorem regular_polygon_sides (h : interior_angle_condition n) : n = 18 := 
  sorry

end regular_polygon_sides_l207_207570


namespace number_is_3034_l207_207263

theorem number_is_3034 (number : ℝ) (h : number - 1002 / 20.04 = 2984) : number = 3034 :=
sorry

end number_is_3034_l207_207263


namespace sum_f_neg_l207_207299

noncomputable def f (x : ℝ) : ℝ := -x - x^3

theorem sum_f_neg {x1 x2 x3 : ℝ}
  (h1 : x1 + x2 > 0)
  (h2 : x2 + x3 > 0)
  (h3 : x3 + x1 > 0) :
  f x1 + f x2 + f x3 < 0 :=
by
  sorry

end sum_f_neg_l207_207299


namespace combined_weight_of_elephant_and_donkey_l207_207923

theorem combined_weight_of_elephant_and_donkey 
  (tons_to_pounds : ℕ → ℕ)
  (elephant_weight_tons : ℕ) 
  (donkey_percentage : ℕ) : 
  tons_to_pounds elephant_weight_tons * (1 + donkey_percentage / 100) = 6600 :=
by
  let tons_to_pounds (t : ℕ) := 2000 * t
  let elephant_weight_tons := 3
  let donkey_percentage := 10
  sorry

end combined_weight_of_elephant_and_donkey_l207_207923


namespace lines_intersection_example_l207_207834

theorem lines_intersection_example (m b : ℝ) 
  (h1 : 8 = m * 4 + 2) 
  (h2 : 8 = 4 * 4 + b) : 
  b + m = -13 / 2 := 
by
  sorry

end lines_intersection_example_l207_207834


namespace buildings_subset_count_l207_207989

theorem buildings_subset_count :
  let buildings := Finset.range (16 + 1) \ {0}
  ∃ S ⊆ buildings, ∀ (a b : ℕ), a ≠ b ∧ a ∈ S ∧ b ∈ S → ∃ k, (b - a = 2 * k + 1) ∨ (a - b = 2 * k + 1) ∧ Finset.card S = 510 :=
sorry

end buildings_subset_count_l207_207989


namespace first_year_payment_l207_207351

theorem first_year_payment (x : ℝ) 
  (second_year : ℝ := x + 2)
  (third_year : ℝ := x + 5)
  (fourth_year : ℝ := x + 9)
  (total_payment : ℝ := x + second_year + third_year + fourth_year)
  (h : total_payment = 96) : x = 20 := 
by
  sorry

end first_year_payment_l207_207351


namespace sum_lent_borrowed_l207_207523

-- Define the given conditions and the sum lent
def sum_lent (P r t : ℝ) (I : ℝ) : Prop :=
  I = P * r * t / 100 ∧ I = P - 1540

-- Define the main theorem to be proven
theorem sum_lent_borrowed : 
  ∃ P : ℝ, sum_lent P 8 10 ((4 * P) / 5) ∧ P = 7700 :=
by
  sorry

end sum_lent_borrowed_l207_207523


namespace petya_time_comparison_l207_207542

theorem petya_time_comparison (V : ℝ) (a : ℝ) (hV_pos : 0 < V) (ha_pos : 0 < a) :
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  T_real > T_planned :=
by
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  have T_real_eq : T_real = (a / (1.25 * V)) + (a / (0.8 * V)) := by sorry
  have T_real_simplified : T_real = (2 * a / 5 / V) + (5 * a / 8 / V) := by sorry
  have T_real_combined : T_real = (41 * a) / (40 * V) := by sorry
  have planned_vs_real : T_real > a / V := by sorry
  exact planned_vs_real

end petya_time_comparison_l207_207542


namespace range_of_a_l207_207885

-- Defining the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, (1 ≤ x ∧ x ≤ 5) → x^2 - (a + 1) * x + a ≤ 0
def q (a : ℝ) : Prop := 3 < a ∧ a < 6

-- Main theorem statement
theorem range_of_a (a : ℝ) (h : ¬(p a) ∧ q a) : 3 < a ∧ a ≤ 5 :=
by
  sorry

end range_of_a_l207_207885


namespace servant_leaving_months_l207_207434

-- The given conditions
def total_salary_year : ℕ := 90 + 110
def monthly_salary (months: ℕ) : ℕ := (months * total_salary_year) / 12
def total_received : ℕ := 40 + 110

-- The theorem to prove
theorem servant_leaving_months (months : ℕ) (h : monthly_salary months = total_received) : months = 9 :=
by {
    sorry
}

end servant_leaving_months_l207_207434


namespace geometric_series_common_ratio_l207_207964

theorem geometric_series_common_ratio (a r S : ℝ) (h₁ : S = a / (1 - r)) (h₂ : ar^4 / (1 - r) = S / 64) : r = 1 / 2 :=
  by
  sorry

end geometric_series_common_ratio_l207_207964


namespace min_weighings_to_find_counterfeit_l207_207825

-- Definition of the problem conditions.
def coin_is_genuine (coins : Fin 10 → ℝ) (n : Fin 10) : Prop :=
  ∀ m : Fin 10, m ≠ n → coins m = coins (Fin.mk 0 sorry)

def counterfit_coin_is_lighter (coins : Fin 10 → ℝ) (n : Fin 10) : Prop :=
  ∀ m : Fin 10, m ≠ n → coins n < coins m

-- The theorem statement
theorem min_weighings_to_find_counterfeit :
  (∀ coins : Fin 10 → ℝ, ∃ n : Fin 10, coin_is_genuine coins n ∧ counterfit_coin_is_lighter coins n → ∃ min_weighings : ℕ, min_weighings = 3) :=
by {
  sorry
}

end min_weighings_to_find_counterfeit_l207_207825


namespace abs_diff_of_two_numbers_l207_207822

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 216) : |x - y| = 6 := 
sorry

end abs_diff_of_two_numbers_l207_207822


namespace count_multiples_4_6_not_5_9_l207_207309

/-- The number of integers between 1 and 500 that are multiples of both 4 and 6 but not of either 5 or 9 is 22. -/
theorem count_multiples_4_6_not_5_9 :
  let lcm_4_6 := (Nat.lcm 4 6)
  let lcm_4_6_5 := (Nat.lcm lcm_4_6 5)
  let lcm_4_6_9 := (Nat.lcm lcm_4_6 9)
  let lcm_4_6_5_9 := (Nat.lcm lcm_4_6_5 9)
  let count_multiples (x : Nat) := (500 / x)
  count_multiples lcm_4_6 - count_multiples lcm_4_6_5 - count_multiples lcm_4_6_9 + count_multiples lcm_4_6_5_9 = 22 :=
by
  let lcm_4_6 := (Nat.lcm 4 6)
  let lcm_4_6_5 := (Nat.lcm lcm_4_6 5)
  let lcm_4_6_9 := (Nat.lcm lcm_4_6 9)
  let lcm_4_6_5_9 := (Nat.lcm lcm_4_6_5 9)
  let count_multiples (x : Nat) := (500 / x)
  show count_multiples lcm_4_6 - count_multiples lcm_4_6_5 - count_multiples lcm_4_6_9 + count_multiples lcm_4_6_5_9 = 22
  sorry

end count_multiples_4_6_not_5_9_l207_207309


namespace second_field_full_rows_l207_207520

theorem second_field_full_rows 
    (rows_field1 : ℕ) (cobs_per_row : ℕ) (total_cobs : ℕ)
    (H1 : rows_field1 = 13)
    (H2 : cobs_per_row = 4)
    (H3 : total_cobs = 116) : 
    (total_cobs - rows_field1 * cobs_per_row) / cobs_per_row = 16 :=
by sorry

end second_field_full_rows_l207_207520


namespace abs_eq_neg_iff_nonpositive_l207_207438

theorem abs_eq_neg_iff_nonpositive (x : ℝ) : |x| = -x ↔ x ≤ 0 := by
  sorry

end abs_eq_neg_iff_nonpositive_l207_207438


namespace range_of_a_l207_207441

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (a - 1) * x < 1 ↔ x > 1 / (a - 1)) → a < 1 :=
by 
  sorry

end range_of_a_l207_207441


namespace perimeter_of_rectangle_l207_207204

theorem perimeter_of_rectangle (DC BC P : ℝ) (hDC : DC = 12) (hArea : 1/2 * DC * BC = 30) : P = 2 * (DC + BC) → P = 34 :=
by
  sorry

end perimeter_of_rectangle_l207_207204


namespace problem_I_problem_II_problem_III_l207_207892

-- The function f(x)
noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) := (1/2) * x^2 - a * Real.log x + b

-- Tangent line at x = 1
def tangent_condition (a : ℝ) (b : ℝ) :=
  1 - a = 3 ∧ f 1 a b = 0

-- Extreme point at x = 1
def extreme_condition (a : ℝ) :=
  1 - a = 0 

-- Monotonicity and minimum m
def inequality_condition (a m : ℝ) :=
  -2 ≤ a ∧ a < 0 ∧ ∀ (x1 x2 : ℝ), 0 < x1 ∧ x1 ≤ 2 ∧ 0 < x2 ∧ x2 ≤ 2 → 
  |f x1 a (0 : ℝ) - f x2 a 0| ≤ m * |1 / x1 - 1 / x2|

-- Proof problem 1
theorem problem_I : ∃ (a b : ℝ), tangent_condition a b → a = -2 ∧ b = -0.5 := sorry

-- Proof problem 2
theorem problem_II : ∃ (a : ℝ), extreme_condition a → a = 1 := sorry

-- Proof problem 3
theorem problem_III : ∃ (m : ℝ), inequality_condition (-2 : ℝ) m → m = 12 := sorry

end problem_I_problem_II_problem_III_l207_207892


namespace ticket_difference_l207_207248

theorem ticket_difference (V G : ℕ) (h1 : V + G = 320) (h2 : 45 * V + 20 * G = 7500) :
  G - V = 232 :=
by
  sorry

end ticket_difference_l207_207248


namespace isosceles_trapezoid_sides_length_l207_207804

theorem isosceles_trapezoid_sides_length (b1 b2 A : ℝ) (h s : ℝ) 
  (hb1 : b1 = 11) (hb2 : b2 = 17) (hA : A = 56) :
  (A = 1/2 * (b1 + b2) * h) →
  (s ^ 2 = h ^ 2 + (b2 - b1) ^ 2 / 4) →
  s = 5 :=
by
  intro
  sorry

end isosceles_trapezoid_sides_length_l207_207804


namespace area_of_smallest_square_containing_circle_l207_207840

theorem area_of_smallest_square_containing_circle (r : ℝ) (h : r = 5) : 
  ∃ (a : ℝ), a = 100 :=
by
  sorry

end area_of_smallest_square_containing_circle_l207_207840


namespace find_F_l207_207190

theorem find_F (C F : ℝ) 
  (h1 : C = 7 / 13 * (F - 40))
  (h2 : C = 26) :
  F = 88.2857 :=
by
  sorry

end find_F_l207_207190


namespace min_value_prime_factorization_l207_207219

/-- Let x and y be positive integers and assume 5 * x ^ 7 = 13 * y ^ 11.
  If x has a prime factorization of the form a ^ c * b ^ d, then the minimum possible value of a + b + c + d is 31. -/
theorem min_value_prime_factorization (x y a b c d : ℕ) (hx_pos : x > 0) (hy_pos: y > 0) (ha_pos : a > 0) (hb_pos : b > 0) (hc_pos: c > 0) (hd_pos: d > 0)
    (h_eq : 5 * x ^ 7 = 13 * y ^ 11) (h_fact : x = a^c * b^d) : a + b + c + d = 31 :=
by
  sorry

end min_value_prime_factorization_l207_207219


namespace volume_ratio_proof_l207_207824

-- Definitions based on conditions
def edge_ratio (a b : ℝ) : Prop := a = 3 * b
def volume_ratio (V_large V_small : ℝ) : Prop := V_large = 27 * V_small

-- Problem statement
theorem volume_ratio_proof (e V_small V_large : ℝ) 
  (h1 : edge_ratio (3 * e) e)
  (h2 : volume_ratio V_large V_small) : 
  V_large / V_small = 27 := 
by sorry

end volume_ratio_proof_l207_207824


namespace number_of_true_statements_is_two_l207_207881

def line_plane_geometry : Type :=
  -- Types representing lines and planes
  sorry

def l : line_plane_geometry := sorry
def alpha : line_plane_geometry := sorry
def m : line_plane_geometry := sorry
def beta : line_plane_geometry := sorry

def is_perpendicular (x y : line_plane_geometry) : Prop := sorry
def is_parallel (x y : line_plane_geometry) : Prop := sorry
def is_contained_in (x y : line_plane_geometry) : Prop := sorry

axiom l_perpendicular_alpha : is_perpendicular l alpha
axiom m_contained_in_beta : is_contained_in m beta

def statement_1 : Prop := is_parallel alpha beta → is_perpendicular l m
def statement_2 : Prop := is_perpendicular alpha beta → is_parallel l m
def statement_3 : Prop := is_parallel l m → is_perpendicular alpha beta

theorem number_of_true_statements_is_two : 
  (statement_1 ↔ true) ∧ (statement_2 ↔ false) ∧ (statement_3 ↔ true) := 
sorry

end number_of_true_statements_is_two_l207_207881


namespace Terry_driving_speed_is_40_l207_207798

-- Conditions
def distance_home_to_workplace : ℕ := 60
def total_time_driving : ℕ := 3

-- Computation for total distance
def total_distance := distance_home_to_workplace * 2

-- Desired speed computation
def driving_speed := total_distance / total_time_driving

-- Problem statement to prove
theorem Terry_driving_speed_is_40 : driving_speed = 40 :=
by 
  sorry -- proof not required as per instructions

end Terry_driving_speed_is_40_l207_207798


namespace evaluate_fraction_l207_207280

theorem evaluate_fraction : ∃ p q : ℤ, gcd p q = 1 ∧ (2023 : ℤ) / (2022 : ℤ) - 2 * (2022 : ℤ) / (2023 : ℤ) = (p : ℚ) / (q : ℚ) ∧ p = -(2022^2 : ℤ) + 4045 :=
by
  sorry

end evaluate_fraction_l207_207280


namespace probability_is_correct_l207_207264

-- Define the properties of the islands
def hasTreasureAndNoTraps : ℚ := 1 / 3
def hasTrapsAndNoTreasure : ℚ := 1 / 6
def hasNeither : ℚ := 1 / 2

-- Define the total number of islands
def totalIslands : ℕ := 7

-- Define the probability calculation function
noncomputable def probabilityOfExactlyFourTreasures : ℚ :=
  let combinations := Nat.choose totalIslands 4 in
  let probFourTreasures := (hasTreasureAndNoTraps ^ 4) in
  let probThreeNoTrapsNoTreasure := (hasNeither ^ 3) in
  combinations * probFourTreasures * probThreeNoTrapsNoTreasure

-- The statement that needs to be proven
theorem probability_is_correct :
  probabilityOfExactlyFourTreasures = 35 / 648 :=
sorry

end probability_is_correct_l207_207264


namespace max_consecutive_integers_sum_l207_207995

theorem max_consecutive_integers_sum (S_n : ℕ → ℕ) : (∀ n, S_n n = n * (n + 1) / 2) → ∀ n, (S_n n < 1000 ↔ n ≤ 44) :=
by
  intros H n
  split
  · intro H1
    have H2 : n * (n + 1) < 2000 := by
      rw [H n] at H1
      exact H1
    sorry
  · intro H1
    have H2 : n ≤ 44 := H1
    have H3 : n * (n + 1) < 2000 := by
      sorry
    have H4 : S_n n < 1000 := by
      rw [H n]
      exact H3
    exact H4

end max_consecutive_integers_sum_l207_207995


namespace compute_a1d1_a2d2_a3d3_eq_1_l207_207618

theorem compute_a1d1_a2d2_a3d3_eq_1 {a1 a2 a3 d1 d2 d3 : ℝ}
  (h : ∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3)) :
  a1 * d1 + a2 * d2 + a3 * d3 = 1 := by
  sorry

end compute_a1d1_a2d2_a3d3_eq_1_l207_207618


namespace oranges_ratio_l207_207461

theorem oranges_ratio (T : ℕ) (h1 : 100 + T + 70 = 470) : T / 100 = 3 := by
  -- The solution steps are omitted.
  sorry

end oranges_ratio_l207_207461


namespace sphere_radius_eq_three_l207_207197

theorem sphere_radius_eq_three (r : ℝ) (h : (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2) : r = 3 := 
sorry

end sphere_radius_eq_three_l207_207197


namespace gold_to_brown_ratio_l207_207656

theorem gold_to_brown_ratio :
  ∃ (num_gold num_brown : ℕ), 
  num_brown = 4 ∧ 
  (∃ (num_blue : ℕ), 
  num_blue = 60 ∧ 
  num_blue = 5 * num_gold) ∧ 
  (num_gold : ℚ) / num_brown = 3 :=
by
  sorry

end gold_to_brown_ratio_l207_207656


namespace cos_A_minus_cos_C_l207_207448

-- Definitions representing the conditions
variables (A B C : ℝ) (a b c : ℝ)
variables (h₁ : 4 * b * Real.sin A = Real.sqrt 7 * a)
variables (h₂ : 2 * b = a + c) (h₃ : A < B) (h₄ : B < C)

-- Statement of the proof problem
theorem cos_A_minus_cos_C (A B C a b c : ℝ)
  (h₁ : 4 * b * Real.sin A = Real.sqrt 7 * a)
  (h₂ : 2 * b = a + c)
  (h₃ : A < B)
  (h₄ : B < C) :
  Real.cos A - Real.cos C = Real.sqrt 7 / 2 :=
by
  sorry

end cos_A_minus_cos_C_l207_207448


namespace exist_matrices_with_dets_l207_207588

noncomputable section

open Matrix BigOperators

variables {α : Type} [Field α] [DecidableEq α]

theorem exist_matrices_with_dets (m n : ℕ) (h₁ : 1 < m) (h₂ : 1 < n)
  (αs : Fin m → α) (β : α) :
  ∃ (A : Fin m → Matrix (Fin n) (Fin n) α), (∀ i, det (A i) = αs i) ∧ det (∑ i, A i) = β :=
sorry

end exist_matrices_with_dets_l207_207588


namespace closest_integer_to_cube_root_of_150_l207_207500

theorem closest_integer_to_cube_root_of_150 : 
  let cbrt := (150: ℝ)^(1/3) in
  abs (cbrt - 6) < abs (cbrt - 5) :=
by
  sorry

end closest_integer_to_cube_root_of_150_l207_207500


namespace gcd_linear_combination_l207_207938

theorem gcd_linear_combination (a b : ℤ) : Int.gcd (5 * a + 3 * b) (13 * a + 8 * b) = Int.gcd a b := by
  sorry

end gcd_linear_combination_l207_207938


namespace working_mom_work_percent_l207_207847

theorem working_mom_work_percent :
  let awake_hours := 16
  let work_hours := 8
  (work_hours / awake_hours) * 100 = 50 :=
by
  sorry

end working_mom_work_percent_l207_207847


namespace pairs_m_n_l207_207874

theorem pairs_m_n (m n : ℤ) : n ^ 2 - 3 * m * n + m - n = 0 ↔ (m = 0 ∧ n = 0) ∨ (m = 0 ∧ n = 1) :=
by sorry

end pairs_m_n_l207_207874


namespace actual_time_greater_than_planned_time_l207_207547

def planned_time (a V : ℝ) : ℝ := a / V

def actual_time (a V : ℝ) : ℝ := (a / (2.5 * V)) + (a / (1.6 * V))

theorem actual_time_greater_than_planned_time (a V : ℝ) (hV : V > 0) : 
  actual_time a V > planned_time a V :=
by 
  sorry

end actual_time_greater_than_planned_time_l207_207547


namespace Nancy_hourly_wage_l207_207781

def tuition_cost := 22000
def parents_coverage := tuition_cost / 2
def scholarship := 3000
def loan := 2 * scholarship
def working_hours := 200
def remaining_tuition := tuition_cost - parents_coverage - scholarship - loan
def hourly_wage_required := remaining_tuition / working_hours

theorem Nancy_hourly_wage : hourly_wage_required = 10 := by
  sorry

end Nancy_hourly_wage_l207_207781


namespace sin_210_eq_neg_half_l207_207252

theorem sin_210_eq_neg_half : Real.sin (210 * Real.pi / 180) = -1 / 2 :=
by 
  sorry

end sin_210_eq_neg_half_l207_207252


namespace floor_of_neg_sqrt_frac_l207_207024

theorem floor_of_neg_sqrt_frac :
  (Int.floor (-Real.sqrt (64 / 9)) = -3) :=
by
  sorry

end floor_of_neg_sqrt_frac_l207_207024


namespace wallpaper_three_layers_l207_207647

theorem wallpaper_three_layers
  (A B C : ℝ)
  (hA : A = 300)
  (hB : B = 30)
  (wall_area : ℝ)
  (h_wall_area : wall_area = 180)
  (hC : C = A - (wall_area - B) - B)
  : C = 120 := by
  sorry

end wallpaper_three_layers_l207_207647


namespace statue_of_liberty_ratio_l207_207819

theorem statue_of_liberty_ratio :
  let H_statue := 305 -- height in feet
  let H_model := 10 -- height in inches
  H_statue / H_model = 30.5 := 
by
  let H_statue := 305
  let H_model := 10
  sorry

end statue_of_liberty_ratio_l207_207819


namespace max_lambda_l207_207414

theorem max_lambda {
  λ : ℝ, 
  ∀ a b : ℝ, λ * a^2 * b^2 * (a + b)^2 ≤ (a^2 + a * b + b^2)^3
} : λ ≤ 27 / 4 :=
begin
  sorry
end

end max_lambda_l207_207414


namespace closest_integer_to_cube_root_of_150_l207_207501

theorem closest_integer_to_cube_root_of_150 : 
  ∃ (n : ℤ), ∀ m : ℤ, abs (150 - 5 ^ 3) < abs (150 - m ^ 3) → n = 5 :=
by
  sorry

end closest_integer_to_cube_root_of_150_l207_207501


namespace vector_addition_example_l207_207184

def vector_addition (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)

theorem vector_addition_example : vector_addition (1, -1) (-1, 2) = (0, 1) := 
by 
  unfold vector_addition 
  simp
  sorry

end vector_addition_example_l207_207184


namespace parabola_intersections_l207_207835

theorem parabola_intersections :
  ∃ y1 y2, (∀ x y, (y = 2 * x^2 + 5 * x + 1 ∧ y = - x^2 + 4 * x + 6) → 
     (x = ( -1 + Real.sqrt 61) / 6 ∧ y = y1) ∨ (x = ( -1 - Real.sqrt 61) / 6 ∧ y = y2)) := 
by
  sorry

end parabola_intersections_l207_207835


namespace negation_of_P_l207_207229

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n)

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def P : Prop := ∀ n : ℕ, is_prime n → is_odd n

theorem negation_of_P : ¬ P ↔ ∃ n : ℕ, is_prime n ∧ ¬ is_odd n :=
by sorry

end negation_of_P_l207_207229


namespace find_N_l207_207876

def consecutive_product_sum_condition (a : ℕ) : Prop :=
  a*(a + 1)*(a + 2) = 8*(a + (a + 1) + (a + 2))

theorem find_N : ∃ (N : ℕ), N = 120 ∧ ∃ (a : ℕ), a > 0 ∧ consecutive_product_sum_condition a := by
  sorry

end find_N_l207_207876


namespace max_consecutive_integers_sum_lt_1000_l207_207996

theorem max_consecutive_integers_sum_lt_1000 :
  ∃ n : ℕ, (∀ m : ℕ, m ≤ n → m * (m + 1) / 2 < 1000) ∧ (n * (n + 1) / 2 < 1000) ∧ ¬((n + 1) * (n + 2) / 2 < 1000) :=
sorry

end max_consecutive_integers_sum_lt_1000_l207_207996


namespace simplify_decimal_l207_207506

theorem simplify_decimal : (3416 / 1000 : ℚ) = 427 / 125 := by
  sorry

end simplify_decimal_l207_207506


namespace student_papers_count_l207_207393

theorem student_papers_count {F n k: ℝ}
  (h1 : 35 * k = 0.6 * n * F)
  (h2 : 5 * k > 0.5 * F)
  (h3 : 6 * k > 0.5 * F)
  (h4 : 7 * k > 0.5 * F)
  (h5 : 8 * k > 0.5 * F)
  (h6 : 9 * k > 0.5 * F) :
  n = 5 :=
by
  sorry

end student_papers_count_l207_207393


namespace find_b_l207_207234

theorem find_b (a b c : ℝ) (h1 : a + b + c = 150) (h2 : a + 10 = c^2) (h3 : b - 5 = c^2) : 
  b = (1322 - 2 * Real.sqrt 1241) / 16 := 
by 
  sorry

end find_b_l207_207234


namespace proof_quadratic_conclusions_l207_207843

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Given points on the graph
def points_on_graph (a b c : ℝ) : Prop :=
  quadratic_function a b c (-1) = -2 ∧
  quadratic_function a b c 0 = -3 ∧
  quadratic_function a b c 1 = -4 ∧
  quadratic_function a b c 2 = -3 ∧
  quadratic_function a b c 3 = 0

-- Assertions based on the problem statement
def assertion_A (a b : ℝ) : Prop := 2 * a + b = 0

def assertion_C (a b c : ℝ) : Prop :=
  quadratic_function a b c 3 = 0 ∧ quadratic_function a b c (-1) = 0

def assertion_D (a b c : ℝ) (m : ℝ) (y1 y2 : ℝ) : Prop :=
  (quadratic_function a b c (m - 1) = y1) → 
  (quadratic_function a b c m = y2) → 
  (y1 < y2) → 
  (m > 3 / 2)

-- Final theorem statement to be proven
theorem proof_quadratic_conclusions (a b c : ℝ) (m y1 y2 : ℝ) :
  points_on_graph a b c →
  assertion_A a b →
  assertion_C a b c →
  assertion_D a b c m y1 y2 :=
by
  sorry

end proof_quadratic_conclusions_l207_207843


namespace terminating_decimal_expansion_of_17_div_200_l207_207702

theorem terminating_decimal_expansion_of_17_div_200 :
  (17 / 200 : ℚ) = 34 / 10000 := sorry

end terminating_decimal_expansion_of_17_div_200_l207_207702


namespace f_strictly_increasing_solve_inequality_l207_207891

variable (f : ℝ → ℝ)

-- Conditions
axiom ax1 : ∀ a b : ℝ, f (a + b) = f a + f b - 1
axiom ax2 : ∀ x : ℝ, x > 0 → f x > 1
axiom ax3 : f 4 = 3

-- Prove monotonicity
theorem f_strictly_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := by
  sorry

-- Solve the inequality
theorem solve_inequality (m : ℝ) : -2/3 < m ∧ m < 2 ↔ f (3 * m^2 - m - 2) < 2 := by
  sorry

end f_strictly_increasing_solve_inequality_l207_207891


namespace Robert_has_taken_more_photos_l207_207772

variables (C L R : ℕ) -- Claire's, Lisa's, and Robert's photos

-- Conditions definitions:
def ClairePhotos : Prop := C = 8
def LisaPhotos : Prop := L = 3 * C
def RobertPhotos : Prop := R > C

-- The proof problem statement:
theorem Robert_has_taken_more_photos (h1 : ClairePhotos C) (h2 : LisaPhotos C L) : RobertPhotos C R :=
by { sorry }

end Robert_has_taken_more_photos_l207_207772


namespace ratio_of_a_to_b_in_arithmetic_sequence_l207_207701

theorem ratio_of_a_to_b_in_arithmetic_sequence (a x b : ℝ) (h : a = 0 ∧ b = 2 * x) : (a / b) = 0 :=
  by sorry

end ratio_of_a_to_b_in_arithmetic_sequence_l207_207701


namespace number_of_elements_in_sequence_l207_207436

theorem number_of_elements_in_sequence :
  ∀ (a₀ d : ℕ) (n : ℕ), 
  a₀ = 4 →
  d = 2 →
  n = 64 →
  (a₀ + (n - 1) * d = 130) →
  n = 64 := 
by
  -- We will skip the proof steps as indicated
  sorry

end number_of_elements_in_sequence_l207_207436


namespace price_per_exercise_book_is_correct_l207_207664

-- Define variables and conditions from the problem statement
variables (xM xH booksM booksH pricePerBook : ℝ)
variables (xH_gives_xM : ℝ)

-- Conditions set up from the problem statement
axiom pooled_money : xM = xH
axiom books_ming : booksM = 8
axiom books_hong : booksH = 12
axiom amount_given : xH_gives_xM = 1.1

-- Problem statement to prove
theorem price_per_exercise_book_is_correct :
  (8 + 12) * pricePerBook / 2 = 1.1 → pricePerBook = 0.55 := by
  sorry

end price_per_exercise_book_is_correct_l207_207664


namespace education_budget_l207_207363

-- Definitions of the conditions
def total_budget : ℕ := 32 * 10^6  -- 32 million
def policing_budget : ℕ := total_budget / 2
def public_spaces_budget : ℕ := 4 * 10^6  -- 4 million

-- The theorem statement
theorem education_budget :
  total_budget - (policing_budget + public_spaces_budget) = 12 * 10^6 :=
by
  sorry

end education_budget_l207_207363


namespace regular_discount_rate_l207_207139

theorem regular_discount_rate (MSRP : ℝ) (s : ℝ) (sale_price : ℝ) (d : ℝ) :
  MSRP = 35 ∧ s = 0.20 ∧ sale_price = 19.6 → d = 0.3 :=
by
  intro h
  sorry

end regular_discount_rate_l207_207139


namespace solve_for_x_l207_207012

theorem solve_for_x (x z : ℝ) (h : z = 3 * x) :
  (4 * z^2 + z + 5 = 3 * (8 * x^2 + z + 3)) ↔ 
  (x = (1 + Real.sqrt 19) / 4 ∨ x = (1 - Real.sqrt 19) / 4) := by
  sorry

end solve_for_x_l207_207012


namespace jellybean_total_l207_207380

theorem jellybean_total 
    (blackBeans : ℕ)
    (greenBeans : ℕ)
    (orangeBeans : ℕ)
    (h1 : blackBeans = 8)
    (h2 : greenBeans = blackBeans + 2)
    (h3 : orangeBeans = greenBeans - 1) :
    blackBeans + greenBeans + orangeBeans = 27 :=
by
    -- The proof will be placed here
    sorry

end jellybean_total_l207_207380


namespace solve_for_w_squared_l207_207604

-- Define the original equation
def eqn (w : ℝ) := 2 * (w + 15)^2 = (4 * w + 9) * (3 * w + 6)

-- Define the goal to prove w^2 = 6.7585 based on the given equation
theorem solve_for_w_squared : ∃ w : ℝ, eqn w ∧ w^2 = 6.7585 :=
by
  sorry

end solve_for_w_squared_l207_207604


namespace product_of_solutions_l207_207484

theorem product_of_solutions :
  ∀ x : ℝ, (x + 3) / (2 * x + 3) = (4 * x + 4) / (7 * x + 4) →
  (∀ x1 x2 : ℝ, (x1 ≠ x2) → (x = x1 ∨ x = x2) → x1 * x2 = 0) :=
by
  sorry

end product_of_solutions_l207_207484


namespace sum_reciprocals_of_squares_eq_fifty_over_forty_nine_l207_207667

theorem sum_reciprocals_of_squares_eq_fifty_over_forty_nine (a b : ℕ) (h : a * b = 7) :
  (1 / (a:ℚ)^2 + 1 / (b:ℚ)^2) = 50 / 49 :=
by {
  sorry
}

end sum_reciprocals_of_squares_eq_fifty_over_forty_nine_l207_207667


namespace medical_team_selection_l207_207491

theorem medical_team_selection : 
  let male_doctors := 6
  let female_doctors := 5
  let choose_male := Nat.choose male_doctors 2
  let choose_female := Nat.choose female_doctors 1
  choose_male * choose_female = 75 := 
by 
  sorry

end medical_team_selection_l207_207491


namespace shirts_not_washed_l207_207633

def total_shortsleeve_shirts : Nat := 40
def total_longsleeve_shirts : Nat := 23
def washed_shirts : Nat := 29

theorem shirts_not_washed :
  (total_shortsleeve_shirts + total_longsleeve_shirts) - washed_shirts = 34 :=
by
  sorry

end shirts_not_washed_l207_207633


namespace inequality_holds_l207_207032

theorem inequality_holds (a : ℝ) : 
  (∀ x : ℝ, a*x^2 + 2*a*x - 2 < 0) ↔ a ∈ Set.Icc (-2 : ℝ) (0 : ℝ) :=
sorry

end inequality_holds_l207_207032


namespace binary_div_mul_l207_207412

-- Define the binary numbers
def a : ℕ := 0b101110
def b : ℕ := 0b110100
def c : ℕ := 0b110

-- Statement to prove the given problem
theorem binary_div_mul : (a * b) / c = 0b101011100 := by
  -- Skipping the proof
  sorry

end binary_div_mul_l207_207412


namespace number_of_divisors_of_n_l207_207186

def n : ℕ := 2^3 * 3^4 * 5^3 * 7^2

theorem number_of_divisors_of_n : ∃ d : ℕ, d = 240 ∧ ∀ k : ℕ, k ∣ n ↔ ∃ a b c d : ℕ, 0 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 4 ∧ 0 ≤ c ∧ c ≤ 3 ∧ 0 ≤ d ∧ d ≤ 2 := 
sorry

end number_of_divisors_of_n_l207_207186


namespace regression_estimate_l207_207286

theorem regression_estimate:
  ∀ (x : ℝ), (1.43 * x + 257 = 400) → x = 100 :=
by
  intro x
  intro h
  sorry

end regression_estimate_l207_207286


namespace problem_l207_207889

noncomputable def f : ℝ → ℝ := sorry

theorem problem :
  (∀ x : ℝ, f (x) + f (x + 2) = 0) →
  (f (1) = -2) →
  (f (2019) + f (2018) = 2) :=
by
  intro h1 h2
  sorry

end problem_l207_207889


namespace closest_integer_to_cube_root_of_150_l207_207499

theorem closest_integer_to_cube_root_of_150 : 
  let cbrt := (150: ℝ)^(1/3) in
  abs (cbrt - 6) < abs (cbrt - 5) :=
by
  sorry

end closest_integer_to_cube_root_of_150_l207_207499


namespace question_eq_answer_l207_207331

theorem question_eq_answer (w x y z k : ℕ) 
  (h : 2^w * 3^x * 5^y * 7^z * 11^k = 2520) : 
  2 * w + 3 * x + 5 * y + 7 * z + 11 * k = 24 :=
sorry

end question_eq_answer_l207_207331


namespace simplify_expression_l207_207349

theorem simplify_expression (t : ℝ) (t_ne_zero : t ≠ 0) : (t^5 * t^3) / t^4 = t^4 := 
by
  sorry

end simplify_expression_l207_207349


namespace sara_museum_visit_l207_207346

theorem sara_museum_visit (S : Finset ℕ) (hS : S.card = 6) :
  ∃ count : ℕ, count = 720 ∧ 
  (∀ M A : Finset ℕ, M.card = 3 → A.card = 3 → M ∪ A = S → 
    count = (S.card.choose M.card) * M.card.factorial * A.card.factorial) :=
by
  sorry

end sara_museum_visit_l207_207346


namespace problem_solution_l207_207626

theorem problem_solution (s t : ℕ) (hpos_s : 0 < s) (hpos_t : 0 < t) (h_eq : s * (s - t) = 29) : s + t = 57 :=
by
  sorry

end problem_solution_l207_207626


namespace car_b_speed_l207_207554

theorem car_b_speed :
  ∀ (v : ℕ),
    (232 - 4 * v = 32) →
    v = 50 :=
  by
  sorry

end car_b_speed_l207_207554


namespace closest_integer_to_cube_root_of_150_l207_207498

theorem closest_integer_to_cube_root_of_150 : ∃ (n : ℤ), abs ((n: ℝ)^3 - 150) ≤ abs (((n + 1 : ℤ) : ℝ)^3 - 150) ∧
  abs ((n: ℝ)^3 - 150) ≤ abs (((n - 1 : ℤ) : ℝ)^3 - 150) ∧ n = 5 :=
by
  sorry

end closest_integer_to_cube_root_of_150_l207_207498


namespace tangent_line_circle_l207_207749

theorem tangent_line_circle (m : ℝ) (h : ∀ x y : ℝ,  (x + y + m = 0) → (x^2 + y^2 = m) → m = 2) : m = 2 :=
sorry

end tangent_line_circle_l207_207749


namespace catches_difference_is_sixteen_l207_207450

noncomputable def joe_catches : ℕ := 23
noncomputable def derek_catches : ℕ := 2 * joe_catches - 4
noncomputable def tammy_catches : ℕ := 30
noncomputable def one_third_derek : ℕ := derek_catches / 3
noncomputable def difference : ℕ := tammy_catches - one_third_derek

theorem catches_difference_is_sixteen :
  difference = 16 := 
by
  sorry

end catches_difference_is_sixteen_l207_207450


namespace coordinates_of_point_P_in_third_quadrant_l207_207643

noncomputable def distance_from_y_axis (P : ℝ × ℝ) : ℝ := abs P.1
noncomputable def distance_from_x_axis (P : ℝ × ℝ) : ℝ := abs P.2

theorem coordinates_of_point_P_in_third_quadrant : 
  ∃ P : ℝ × ℝ, P.1 < 0 ∧ P.2 < 0 ∧ distance_from_x_axis P = 2 ∧ distance_from_y_axis P = 5 ∧ P = (-5, -2) :=
by
  sorry

end coordinates_of_point_P_in_third_quadrant_l207_207643


namespace eq_of_divides_l207_207515

theorem eq_of_divides (a b : ℕ) (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b :=
sorry

end eq_of_divides_l207_207515


namespace find_x_5pi_over_4_l207_207028

open Real

theorem find_x_5pi_over_4 (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * π) (h3 : sin x + cos x = -sqrt 2) : x = 5 * π / 4 := 
sorry

end find_x_5pi_over_4_l207_207028


namespace value_of_b_l207_207368

theorem value_of_b (a b : ℕ) (q : ℝ)
  (h1 : q = 0.5)
  (h2 : a = 2020)
  (h3 : q = a / b) : b = 4040 := by
  sorry

end value_of_b_l207_207368


namespace soap_bubble_thickness_scientific_notation_l207_207101

theorem soap_bubble_thickness_scientific_notation :
  (0.0007 * 0.001) = 7 * 10^(-7) := by
sorry

end soap_bubble_thickness_scientific_notation_l207_207101


namespace equation_of_circle_correct_l207_207097

open Real

def equation_of_circle_through_points (x y : ℝ) :=
  x^2 + y^2 - 4 * x - 6 * y

theorem equation_of_circle_correct :
  ∀ (x y : ℝ),
    (equation_of_circle_through_points (0 : ℝ) (0 : ℝ) = 0) →
    (equation_of_circle_through_points (4 : ℝ) (0 : ℝ) = 0) →
    (equation_of_circle_through_points (-1 : ℝ) (1 : ℝ) = 0) →
    (equation_of_circle_through_points x y = 0) :=
by 
  sorry

end equation_of_circle_correct_l207_207097


namespace problem_statement_period_property_symmetry_property_zero_property_l207_207181

noncomputable def f (x : ℝ) : ℝ := Real.cos (x + Real.pi / 3)

theorem problem_statement : ¬(∀ x : ℝ, (Real.pi / 2 < x ∧ x < Real.pi) → f x > f (x + ε))
  → ∃ x : ℝ, f (x + Real.pi) = 0 :=
by
  intro h
  use Real.pi / 6
  sorry

theorem period_property : ∀ k : ℤ, f (x + 2 * k * Real.pi) = f x :=
by
  intro k
  sorry

theorem symmetry_property : ∀ y : ℝ, f (8 * Real.pi / 3 - y) = f (8 * Real.pi / 3 + y) :=
by
  intro y
  sorry

theorem zero_property : f (Real.pi / 6 + Real.pi) = 0 :=
by
  sorry

end problem_statement_period_property_symmetry_property_zero_property_l207_207181


namespace factorial_expression_l207_207495

namespace FactorialProblem

-- Definition of factorial function.
def factorial : ℕ → ℕ 
| 0 => 1
| (n+1) => (n+1) * factorial n

-- Theorem stating the problem equivalently.
theorem factorial_expression : (factorial 12 - factorial 10) / factorial 8 = 11790 := by
  sorry

end FactorialProblem

end factorial_expression_l207_207495


namespace total_cost_correct_l207_207845

/-- Define the base car rental cost -/
def rental_cost : ℝ := 150

/-- Define cost per mile -/
def cost_per_mile : ℝ := 0.5

/-- Define miles driven on Monday -/
def miles_monday : ℝ := 620

/-- Define miles driven on Thursday -/
def miles_thursday : ℝ := 744

/-- Define the total cost Zach spent -/
def total_cost : ℝ := rental_cost + (miles_monday * cost_per_mile) + (miles_thursday * cost_per_mile)

/-- Prove that the total cost Zach spent is 832 dollars -/
theorem total_cost_correct : total_cost = 832 := by
  sorry

end total_cost_correct_l207_207845


namespace steel_mill_production_2010_l207_207858

noncomputable def steel_mill_production (P : ℕ → ℕ) : Prop :=
  (P 1990 = 400000) ∧ (P 2000 = 500000) ∧ ∀ n, (P n) = (P (n-1)) + (500000 - 400000) / 10

theorem steel_mill_production_2010 (P : ℕ → ℕ) (h : steel_mill_production P) : P 2010 = 630000 :=
by
  sorry -- proof omitted

end steel_mill_production_2010_l207_207858


namespace seating_arrangements_l207_207623

-- Definitions for conditions
def num_parents : ℕ := 2
def num_children : ℕ := 3
def num_front_seats : ℕ := 2
def num_back_seats : ℕ := 3
def num_family_members : ℕ := num_parents + num_children

-- The statement we need to prove
theorem seating_arrangements : 
  (num_parents * -- choices for driver
  (num_family_members - 1) * -- choices for the front passenger
  (num_back_seats.factorial)) = 48 := -- arrangements for the back seats
by
  sorry

end seating_arrangements_l207_207623


namespace average_income_B_and_C_l207_207224

variables (A_income B_income C_income : ℝ)

noncomputable def average_monthly_income_B_and_C (A_income : ℝ) :=
  (B_income + C_income) / 2

theorem average_income_B_and_C
  (h1 : (A_income + B_income) / 2 = 5050)
  (h2 : (A_income + C_income) / 2 = 5200)
  (h3 : A_income = 4000) :
  average_monthly_income_B_and_C 4000 = 6250 :=
by
  sorry

end average_income_B_and_C_l207_207224


namespace sum_a_b_eq_34_over_3_l207_207599

theorem sum_a_b_eq_34_over_3 (a b: ℚ)
  (h1 : 2 * a + 5 * b = 43)
  (h2 : 8 * a + 2 * b = 50) :
  a + b = 34 / 3 :=
sorry

end sum_a_b_eq_34_over_3_l207_207599


namespace find_x_l207_207375

variable (x : ℕ)  -- we'll use natural numbers to avoid negative values

-- initial number of children
def initial_children : ℕ := 21

-- number of children who got off
def got_off : ℕ := 10

-- total children after some got on
def total_children : ℕ := 16

-- statement to prove x is the number of children who got on the bus
theorem find_x : initial_children - got_off + x = total_children → x = 5 :=
by
  sorry

end find_x_l207_207375


namespace problem1_problem2_l207_207274

variable (x y a b c d : ℝ)
variable (h_a : a ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0)

-- Problem 1: Prove (x + y) * (x^2 - x * y + y^2) = x^3 + y^3
theorem problem1 : (x + y) * (x^2 - x * y + y^2) = x^3 + y^3 := sorry

-- Problem 2: Prove ((a^2 * b) / (-c * d^3))^3 / (2 * a / d^3) * (c / (2 * a))^2 = - (a^3 * b^3) / (8 * c * d^6)
theorem problem2 (a b c d : ℝ) (h_a : a ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) : 
  ((a^2 * b) / (-c * d^3))^3 / (2 * a / d^3) * (c / (2 * a))^2 = - (a^3 * b^3) / (8 * c * d^6) := 
  sorry

end problem1_problem2_l207_207274


namespace sequence_term_l207_207362

noncomputable def S (n : ℕ) : ℤ := n^2 - 3 * n

theorem sequence_term (n : ℕ) (h : n ≥ 1) : 
  ∃ a : ℕ → ℤ, a n = 2 * n - 4 := 
  sorry

end sequence_term_l207_207362


namespace regular_pentagonal_pyramid_angle_l207_207319

noncomputable def angle_between_slant_height_and_non_intersecting_edge (base_edge_slant_height : ℝ) : ℝ :=
  -- Assuming the base edge and slant height are given as input and equal
  if base_edge_slant_height > 0 then 36 else 0

theorem regular_pentagonal_pyramid_angle
  (base_edge_slant_height : ℝ)
  (h : base_edge_slant_height > 0) :
  angle_between_slant_height_and_non_intersecting_edge base_edge_slant_height = 36 :=
by
  -- omitted proof steps
  sorry

end regular_pentagonal_pyramid_angle_l207_207319


namespace observation_count_l207_207119

theorem observation_count (mean_before mean_after : ℝ) 
  (wrong_value : ℝ) (correct_value : ℝ) (n : ℝ) :
  mean_before = 36 →
  correct_value = 60 →
  wrong_value = 23 →
  mean_after = 36.5 →
  n = 74 :=
by
  intros h_mean_before h_correct_value h_wrong_value h_mean_after
  sorry

end observation_count_l207_207119


namespace trigonometric_identity_l207_207738

theorem trigonometric_identity
  (x : ℝ) 
  (h_tan : Real.tan x = -1/2) :
  (3 * Real.sin x ^ 2 - 2) / (Real.sin x * Real.cos x) = 7 / 2 := 
by
  sorry

end trigonometric_identity_l207_207738


namespace geometric_series_common_ratio_l207_207978

theorem geometric_series_common_ratio (a r : ℝ) (h₁ : r ≠ 1)
    (h₂ : a / (1 - r) = 64 * (a * r^4) / (1 - r)) : r = 1/2 :=
by
  have h₃ : 1 = 64 * r^4 := by
    have : 1 - r ≠ 0 := by linarith
    field_simp at h₂; assumption
  sorry

end geometric_series_common_ratio_l207_207978


namespace intersection_A_B_l207_207886

-- Definitions for sets A and B
def A : Set ℝ := { x | ∃ y : ℝ, x + y^2 = 1 }
def B : Set ℝ := { y | ∃ x : ℝ, y = x^2 - 1 }

-- The proof goal to show the intersection of sets A and B
theorem intersection_A_B : A ∩ B = { z | -1 ≤ z ∧ z ≤ 1 } :=
by
  sorry

end intersection_A_B_l207_207886


namespace mrs_petersons_change_l207_207933

-- Define the conditions
def num_tumblers : ℕ := 10
def cost_per_tumbler : ℕ := 45
def discount_rate : ℚ := 0.10
def num_bills : ℕ := 5
def value_per_bill : ℕ := 100

-- Formulate the proof statement
theorem mrs_petersons_change :
  let total_cost_before_discount := num_tumblers * cost_per_tumbler
  let discount_amount := total_cost_before_discount * discount_rate
  let total_cost_after_discount := total_cost_before_discount - discount_amount
  let total_amount_paid := num_bills * value_per_bill
  let change_received := total_amount_paid - total_cost_after_discount
  change_received = 95 := by sorry

end mrs_petersons_change_l207_207933


namespace probability_selecting_girl_l207_207143

def boys : ℕ := 3
def girls : ℕ := 1
def total_candidates : ℕ := boys + girls
def favorable_outcomes : ℕ := girls

theorem probability_selecting_girl : 
  ∃ p : ℚ, p = (favorable_outcomes : ℚ) / (total_candidates : ℚ) ∧ p = 1 / 4 :=
sorry

end probability_selecting_girl_l207_207143


namespace monthly_installment_amount_l207_207947

theorem monthly_installment_amount (total_cost : ℝ) (down_payment_percentage : ℝ) (additional_down_payment : ℝ) 
  (balance_after_months : ℝ) (months : ℕ) (monthly_installment : ℝ) : 
    total_cost = 1000 → 
    down_payment_percentage = 0.20 → 
    additional_down_payment = 20 → 
    balance_after_months = 520 → 
    months = 4 → 
    monthly_installment = 65 :=
by
  intros
  sorry

end monthly_installment_amount_l207_207947


namespace hyperbola_equiv_l207_207030

-- The existing hyperbola
def hyperbola1 (x y : ℝ) : Prop := y^2 / 4 - x^2 = 1

-- The new hyperbola with same asymptotes passing through (2, 2) should have this form
def hyperbola2 (x y : ℝ) : Prop := (x^2 / 3 - y^2 / 12 = 1)

theorem hyperbola_equiv (x y : ℝ) :
  (hyperbola1 2 2) →
  (y^2 / 4 - x^2 / 4 = -3) →
  (hyperbola2 x y) :=
by
  intros h1 h2
  sorry

end hyperbola_equiv_l207_207030


namespace lena_more_candy_bars_than_nicole_l207_207074

theorem lena_more_candy_bars_than_nicole
  (Lena Kevin Nicole : ℕ)
  (h1 : Lena = 16)
  (h2 : Lena + 5 = 3 * Kevin)
  (h3 : Kevin + 4 = Nicole) :
  Lena - Nicole = 5 :=
by
  sorry

end lena_more_candy_bars_than_nicole_l207_207074


namespace solution_l207_207816

-- Define the equations and their solution sets
def eq1 (x p : ℝ) : Prop := x^2 - p * x + 6 = 0
def eq2 (x q : ℝ) : Prop := x^2 + 6 * x - q = 0

-- Define the condition that the solution sets intersect at {2}
def intersect_at_2 (p q : ℝ) : Prop :=
  eq1 2 p ∧ eq2 2 q

-- The main theorem stating the value of p + q given the conditions
theorem solution (p q : ℝ) (h : intersect_at_2 p q) : p + q = 21 :=
by
  sorry

end solution_l207_207816


namespace Vins_total_miles_l207_207838

theorem Vins_total_miles : 
  let dist_library_one_way := 6
  let dist_school_one_way := 5
  let dist_friend_one_way := 8
  let extra_miles := 1
  let shortcut_miles := 2
  let days_per_week := 7
  let weeks := 4

  -- Calculate weekly miles
  let library_round_trip := (dist_library_one_way + dist_library_one_way + extra_miles)
  let total_library_weekly := library_round_trip * 3

  let school_round_trip := (dist_school_one_way + dist_school_one_way + extra_miles)
  let total_school_weekly := school_round_trip * 2

  let friend_round_trip := dist_friend_one_way + (dist_friend_one_way - shortcut_miles)
  let total_friend_weekly := friend_round_trip / 2 -- Every two weeks

  let total_weekly := total_library_weekly + total_school_weekly + total_friend_weekly

  -- Calculate total miles over the weeks
  let total_miles := total_weekly * weeks

  total_miles = 272 := sorry

end Vins_total_miles_l207_207838


namespace cone_height_l207_207257

theorem cone_height (r_sphere : ℝ) (r_cone : ℝ) (waste_percentage : ℝ) 
  (h : ℝ) (V_sphere : ℝ) (V_cone : ℝ) : 
  r_sphere = 9 → r_cone = 9 → waste_percentage = 0.75 → 
  V_sphere = (4 / 3) * Real.pi * r_sphere^3 → 
  V_cone = (1 / 3) * Real.pi * r_cone^2 * h → 
  V_cone = waste_percentage * V_sphere → 
  h = 27 :=
by
  intros r_sphere_eq r_cone_eq waste_eq V_sphere_eq V_cone_eq V_cone_waste_eq
  sorry

end cone_height_l207_207257


namespace king_william_probability_l207_207328

theorem king_william_probability :
  let m := 2
  let n := 15
  m + n = 17 :=
by
  sorry

end king_william_probability_l207_207328


namespace total_distinct_symbols_l207_207757

def numSequences (n : ℕ) : ℕ := 3^n

theorem total_distinct_symbols :
  numSequences 1 + numSequences 2 + numSequences 3 + numSequences 4 = 120 :=
by
  sorry

end total_distinct_symbols_l207_207757


namespace length_of_AB_l207_207625

theorem length_of_AB (A B P Q : ℝ) 
  (hp : 0 < P) (hp' : P < 1) 
  (hq : 0 < Q) (hq' : Q < 1) 
  (H1 : P = 3 / 7) (H2 : Q = 5 / 12)
  (H3 : P * (1 - Q) + Q * (1 - P) = 4) : 
  (B - A) = 336 / 11 :=
by
  sorry

end length_of_AB_l207_207625


namespace exponentiation_multiplication_identity_l207_207008

theorem exponentiation_multiplication_identity :
  (-4)^(2010) * (-0.25)^(2011) = -0.25 :=
by
  sorry

end exponentiation_multiplication_identity_l207_207008


namespace geometric_sequence_condition_l207_207287

theorem geometric_sequence_condition (a b c d : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :
  (∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) → (a * d = b * c) ∧ 
  ¬ (∀ a b c d : ℝ, a * d = b * c → ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) := 
by
  sorry

end geometric_sequence_condition_l207_207287


namespace cyclist_average_rate_l207_207372

noncomputable def average_rate_round_trip (D : ℝ) : ℝ :=
  let time_to_travel := D / 10
  let time_to_return := D / 9
  let total_distance := 2 * D
  let total_time := time_to_travel + time_to_return
  (total_distance / total_time)

theorem cyclist_average_rate (D : ℝ) (hD : D > 0) :
  average_rate_round_trip D = 180 / 19 :=
by
  sorry

end cyclist_average_rate_l207_207372


namespace initial_typists_count_l207_207060

theorem initial_typists_count
  (letters_per_20_min : Nat)
  (letters_total_1_hour : Nat)
  (letters_typists_count : Nat)
  (n_typists_init : Nat)
  (h1 : letters_per_20_min = 46)
  (h2 : letters_typists_count = 30)
  (h3 : letters_total_1_hour = 207) :
  n_typists_init = 20 :=
by {
  sorry
}

end initial_typists_count_l207_207060


namespace limit_expression_equals_half_second_derivative_l207_207428

variable {f : ℝ → ℝ}
variable {hf : DifferentiableAt ℝ f 1}

theorem limit_expression_equals_half_second_derivative :
  (𝓝[≠] 0).lim (λ Δx : ℝ, (f(1 + Δx) - f(1)) / (-2 * Δx)) = - (1 / 2) * deriv (deriv f) 1 :=
by
  sorry

end limit_expression_equals_half_second_derivative_l207_207428


namespace convert_cost_to_usd_l207_207001

def sandwich_cost_gbp : Float := 15.0
def conversion_rate : Float := 1.3

theorem convert_cost_to_usd :
  (Float.round ((sandwich_cost_gbp * conversion_rate) * 100) / 100) = 19.50 :=
by
  sorry

end convert_cost_to_usd_l207_207001


namespace John_used_16_bulbs_l207_207069

variable (X : ℕ)

theorem John_used_16_bulbs
  (h1 : 40 - X = 2 * 12) :
  X = 16 := 
sorry

end John_used_16_bulbs_l207_207069


namespace quadratic_has_solution_zero_l207_207587

theorem quadratic_has_solution_zero (k : ℝ) : 
  (∃ x : ℝ, (k - 2) * x^2 + 3 * x + k^2 - 4 = 0) →
  ((k - 2) ≠ 0) → k = -2 := 
by 
  sorry

end quadratic_has_solution_zero_l207_207587


namespace volume_of_right_prism_with_trapezoid_base_l207_207787

variable (S1 S2 H a b h: ℝ)

theorem volume_of_right_prism_with_trapezoid_base 
  (hS1 : S1 = a * H) 
  (hS2 : S2 = b * H) 
  (h_trapezoid : a ≠ b) : 
  1 / 2 * (S1 + S2) * h = (1 / 2 * (a + b) * h) * H :=
by 
  sorry

end volume_of_right_prism_with_trapezoid_base_l207_207787


namespace find_a_l207_207590

theorem find_a
  (a b c : ℝ) 
  (h1 : ∀ x : ℝ, x = 1 ∨ x = 2 → a * x * (x + 1) + b * x * (x + 2) + c * (x + 1) * (x + 2) = 0)
  (h2 : a + b + c = 2) : 
  a = 12 := 
sorry

end find_a_l207_207590


namespace geometric_series_common_ratio_l207_207974

theorem geometric_series_common_ratio :
  ∀ (a r : ℝ), (r ≠ 1) → 
  (∑' n, a * r^n = 64 * ∑' n, a * r^(n+4)) →
  r = 1 / 2 :=
by
  intros a r hnr heq
  have hsum1 : ∑' n, a * r^n = a / (1 - r) := sorry
  have hsum2 : ∑' n, a * r^(n+4) = a * r^4 / (1 - r) := sorry
  rw [hsum1, hsum2] at heq
  -- Further steps to derive r = 1/2 are omitted
  sorry

end geometric_series_common_ratio_l207_207974


namespace who_is_next_to_Denis_l207_207105

-- Definitions according to conditions
def SchoolChildren := ["Anya", "Borya", "Vera", "Gena", "Denis"]

def Position : Type := ℕ

def is_next_to (a b : Position) := abs (a - b) = 1

def valid_positions (positions : SchoolChildren → Position) :=
  positions "Borya" = 1 ∧
  is_next_to (positions "Vera") (positions "Anya") ∧
  ¬is_next_to (positions "Vera") (positions "Gena") ∧
  ¬is_next_to (positions "Anya") (positions "Borya") ∧
  ¬is_next_to (positions "Gena") (positions "Borya") ∧
  ¬is_next_to (positions "Anya") (positions "Gena")

theorem who_is_next_to_Denis:
  ∃ positions : SchoolChildren → Position, valid_positions positions ∧ 
  (is_next_to (positions "Denis") (positions "Anya") ∧
   is_next_to (positions "Denis") (positions "Gena")) :=
by
  sorry

end who_is_next_to_Denis_l207_207105


namespace smallest_prime_sum_l207_207583

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_sum_of_distinct_primes (n k : ℕ) (s : List ℕ) : Prop :=
  s.length = k ∧ (∀ x ∈ s, is_prime x) ∧ (∀ (x y : ℕ), x ≠ y → x ∈ s → y ∈ s → x ≠ y) ∧ s.sum = n

theorem smallest_prime_sum :
  (is_prime 61) ∧ 
  (∃ s2, is_sum_of_distinct_primes 61 2 s2) ∧ 
  (∃ s3, is_sum_of_distinct_primes 61 3 s3) ∧ 
  (∃ s4, is_sum_of_distinct_primes 61 4 s4) ∧ 
  (∃ s5, is_sum_of_distinct_primes 61 5 s5) ∧ 
  (∃ s6, is_sum_of_distinct_primes 61 6 s6) :=
by
  sorry

end smallest_prime_sum_l207_207583


namespace integer_solutions_count_l207_207951

theorem integer_solutions_count : 
  ∃ (s : Finset ℤ), 
    (∀ x ∈ s, 2 * x + 1 > -3 ∧ -x + 3 ≥ 0) ∧ 
    s.card = 5 := 
by 
  sorry

end integer_solutions_count_l207_207951


namespace fraction_value_l207_207296

theorem fraction_value (a b : ℝ) (h : 1 / a - 1 / b = 4) : 
    (a - 2 * a * b - b) / (2 * a + 7 * a * b - 2 * b) = 6 :=
by
  sorry

end fraction_value_l207_207296


namespace minimize_perimeter_of_sector_l207_207727

theorem minimize_perimeter_of_sector (r θ: ℝ) (h₁: (1 / 2) * θ * r^2 = 16) (h₂: 2 * r + θ * r = 2 * r + 32 / r): θ = 2 :=
by
  sorry

end minimize_perimeter_of_sector_l207_207727


namespace max_difference_in_masses_of_two_flour_bags_l207_207341

theorem max_difference_in_masses_of_two_flour_bags :
  ∀ (x y : ℝ), (24.8 ≤ x ∧ x ≤ 25.2) → (24.8 ≤ y ∧ y ≤ 25.2) → |x - y| ≤ 0.4 :=
by
  sorry

end max_difference_in_masses_of_two_flour_bags_l207_207341


namespace work_completion_time_l207_207383

theorem work_completion_time (A B C : ℝ) (hA : A = 1 / 4) (hB : B = 1 / 12) (hAC : A + C = 1 / 2) :
  1 / (B + C) = 3 :=
by
  -- The proof goes here
  sorry

end work_completion_time_l207_207383


namespace find_constant_term_l207_207429

theorem find_constant_term (q' : ℝ → ℝ) (c : ℝ) (h1 : ∀ q : ℝ, q' q = 3 * q - c)
  (h2 : q' (q' 7) = 306) : c = 252 :=
by
  sorry

end find_constant_term_l207_207429


namespace slope_angle_of_y_eq_0_l207_207196

theorem slope_angle_of_y_eq_0  :
  ∀ (α : ℝ), (∀ (y x : ℝ), y = 0) → α = 0 :=
by
  intros α h
  sorry

end slope_angle_of_y_eq_0_l207_207196


namespace det_matrix_4x4_l207_207512

def matrix_4x4 : Matrix (Fin 4) (Fin 4) ℤ :=
  ![
    ![3, 0, 2, 0],
    ![2, 3, -1, 4],
    ![0, 4, -2, 3],
    ![5, 2, 0, 1]
  ]

theorem det_matrix_4x4 : Matrix.det matrix_4x4 = -84 :=
by
  sorry

end det_matrix_4x4_l207_207512


namespace regular_polygon_sides_l207_207569

-- Define the number of sides
def n : ℕ := sorry

-- The interior angle condition
def interior_angle_condition (n : ℕ) : Prop := 
  let interior_angle := (180 * (n - 2)) / n
  interior_angle = 160

-- Prove the statement
theorem regular_polygon_sides (h : interior_angle_condition n) : n = 18 := 
  sorry

end regular_polygon_sides_l207_207569


namespace minimum_value_of_expr_l207_207743

noncomputable def expr (x : ℝ) := 2 * x + 1 / (x + 3)

theorem minimum_value_of_expr (x : ℝ) (h : x > -3) :
  ∃ y, y = 2 * real.sqrt 2 - 6 ∧ ∀ z, z > -3 → expr z ≥ y := sorry

end minimum_value_of_expr_l207_207743


namespace fraction_doubling_unchanged_l207_207914

theorem fraction_doubling_unchanged (x y : ℝ) (h : x ≠ y) : 
  (3 * (2 * x)) / (2 * x - 2 * y) = (3 * x) / (x - y) :=
by
  sorry

end fraction_doubling_unchanged_l207_207914


namespace closest_integer_to_cuberoot_150_l207_207504

theorem closest_integer_to_cuberoot_150 : 
  let cube5 := 5^3 in 
  let cube6 := 6^3 in 
  let midpoint := (cube5 + cube6) / 2 in 
  125 < 150 ∧ 150 < 216 ∧ 150 < midpoint → 
  5 = round (150^(1/3)) := 
by 
  intro h
  sorry

end closest_integer_to_cuberoot_150_l207_207504


namespace liters_conversion_hours_to_days_cubic_meters_to_cubic_cm_l207_207671

-- Define the conversions and the corresponding proofs
theorem liters_conversion : 8.32 = 8 + 320 / 1000 := sorry

theorem hours_to_days : 6 = 1 / 4 * 24 := sorry

theorem cubic_meters_to_cubic_cm : 0.75 * 10^6 = 750000 := sorry

end liters_conversion_hours_to_days_cubic_meters_to_cubic_cm_l207_207671


namespace initial_decaf_percentage_l207_207679

theorem initial_decaf_percentage 
  (x : ℝ)
  (h1 : 0 ≤ x) (h2 : x ≤ 100) 
  (h3 : (x / 100 * 400) + 60 = 220) :
  x = 40 :=
by sorry

end initial_decaf_percentage_l207_207679


namespace complement_of_angle_l207_207637

theorem complement_of_angle (x : ℝ) (h : 90 - x = 3 * x + 10) : x = 20 := by
  sorry

end complement_of_angle_l207_207637


namespace c_zero_roots_arithmetic_seq_range_f1_l207_207595

section problem

variable (b : ℝ)
def f (x : ℝ) := x^3 + 3 * b * x^2 + 0 * x + (-2 * b^3)
def f' (x : ℝ) := 3 * x^2 + 6 * b * x + 0

-- Proving c = 0 if f(x) is increasing on (-∞, 0) and decreasing on (0, 2)
theorem c_zero (h_inc : ∀ x < 0, f' b x > 0) (h_dec : ∀ x > 0, f' b x < 0) : 0 = 0 := sorry

-- Proving f(x) = 0 has two other distinct real roots x1 and x2 different from -b, forming an arithmetic sequence
theorem roots_arithmetic_seq (hb : ∀ x : ℝ, f b x = 0 → (x = -b ∨ -b ≠ x)) : 
    ∃ (x1 x2 : ℝ), x1 ≠ -b ∧ x2 ≠ -b ∧ x1 + x2 = -2 * b := sorry

-- Proving the range of values for f(1) when the maximum value of f(x) is less than 16
theorem range_f1 (h_max : ∀ x : ℝ, f b x < 16 ) : 0 ≤ f b 1 ∧ f b 1 < 11 := sorry

end problem

end c_zero_roots_arithmetic_seq_range_f1_l207_207595


namespace odd_squares_diff_divisible_by_8_l207_207790

theorem odd_squares_diff_divisible_by_8 (m n : ℤ) (a b : ℤ) (hm : a = 2 * m + 1) (hn : b = 2 * n + 1) : (a^2 - b^2) % 8 = 0 := sorry

end odd_squares_diff_divisible_by_8_l207_207790


namespace find_k_l207_207420

theorem find_k (x y k : ℤ) (h₁ : x = -3) (h₂ : y = 2) (h₃ : 2 * x + k * y = 6) : k = 6 :=
by
  rw [h₁, h₂] at h₃
  -- Substitute x and y in the equation
  -- 2 * (-3) + k * 2 = 6
  sorry

end find_k_l207_207420


namespace daniel_initial_noodles_l207_207131

variable (give : ℕ)
variable (left : ℕ)
variable (initial : ℕ)

theorem daniel_initial_noodles (h1 : give = 12) (h2 : left = 54) (h3 : initial = left + give) : initial = 66 := by
  sorry

end daniel_initial_noodles_l207_207131


namespace simplify_polynomial_l207_207940

theorem simplify_polynomial :
  (3 * x^3 + 4 * x^2 + 8 * x - 5) - (2 * x^3 + x^2 + 6 * x - 7) = x^3 + 3 * x^2 + 2 * x + 2 := 
by
  sorry

end simplify_polynomial_l207_207940


namespace cos_sum_formula_l207_207267

open Real

theorem cos_sum_formula (A B C : ℝ) (h1 : sin A + sin B + sin C = 0) (h2 : cos A + cos B + cos C = 0) :
  cos (A - B) + cos (B - C) + cos (C - A) = -3 / 2 :=
by
  sorry

end cos_sum_formula_l207_207267


namespace runner_speed_comparison_l207_207651

theorem runner_speed_comparison
  (t1 t2 : ℕ → ℝ) -- function to map lap-time.
  (s v1 v2 : ℝ)  -- speed of runners v1 and v2 respectively, and the street distance s.
  (h1 : t1 1 < t2 1) -- first runner overtakes the second runner twice implying their lap-time comparison.
  (h2 : ∀ n, t1 (n + 1) = t1 n + t1 1) -- lap time consistency for runner 1
  (h3 : ∀ n, t2 (n + 1) = t2 n + t2 1) -- lap time consistency for runner 2
  (h4 : t1 3 < t2 2) -- first runner completes 3 laps faster than second runner completes 2 laps
   : 2 * v2 ≤ v1 := sorry

end runner_speed_comparison_l207_207651


namespace find_r4_l207_207206

-- Definitions of the problem conditions
variable (r1 r2 r3 r4 r5 r6 r7 : ℝ)
-- Given radius of the smallest circle
axiom smallest_circle : r1 = 6
-- Given radius of the largest circle
axiom largest_circle : r7 = 24
-- Given that radii of circles form a geometric sequence
axiom geometric_sequence : r2 = r1 * (r7 / r1)^(1/6) ∧ 
                            r3 = r1 * (r7 / r1)^(2/6) ∧
                            r4 = r1 * (r7 / r1)^(3/6) ∧
                            r5 = r1 * (r7 / r1)^(4/6) ∧
                            r6 = r1 * (r7 / r1)^(5/6)

-- Statement to prove
theorem find_r4 : r4 = 12 :=
by
  sorry

end find_r4_l207_207206


namespace complement_of_A_in_U_l207_207597

noncomputable def U := {x : ℝ | Real.exp x > 1}

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x - 1)

def A := { x : ℝ | x > 1 }

def compl (U A : Set ℝ) := { x : ℝ | x ∈ U ∧ x ∉ A }

theorem complement_of_A_in_U : compl U A = { x : ℝ | 0 < x ∧ x ≤ 1 } := sorry

end complement_of_A_in_U_l207_207597


namespace jellybean_count_l207_207378

def black_beans : Nat := 8
def green_beans : Nat := black_beans + 2
def orange_beans : Nat := green_beans - 1
def total_jelly_beans : Nat := black_beans + green_beans + orange_beans

theorem jellybean_count : total_jelly_beans = 27 :=
by
  -- proof steps would go here.
  sorry

end jellybean_count_l207_207378


namespace dmitry_black_socks_l207_207849

theorem dmitry_black_socks :
  let blue_socks := 10
  let initial_black_socks := 22
  let white_socks := 12
  let total_initial_socks := blue_socks + initial_black_socks + white_socks
  ∀ x : ℕ,
    let total_socks := total_initial_socks + x
    let black_socks := initial_black_socks + x
    (black_socks : ℚ) / (total_socks : ℚ) = 2 / 3 → x = 22 :=
by
  sorry

end dmitry_black_socks_l207_207849


namespace negative_comparison_l207_207557

theorem negative_comparison : -2023 > -2024 :=
sorry

end negative_comparison_l207_207557


namespace geometric_sequence_sum_l207_207064

theorem geometric_sequence_sum (a_1 q n S : ℕ) (h1 : a_1 = 2) (h2 : q = 2) (h3 : S = 126) 
    (h4 : S = (a_1 * (1 - q^n)) / (1 - q)) : 
    n = 6 :=
by
  sorry

end geometric_sequence_sum_l207_207064


namespace selection_count_Group3_selection_count_Group4_selection_count_Group5_probability_A_or_B_l207_207462

/-
  Conditions:
-/
def Group3 : ℕ := 18
def Group4 : ℕ := 12
def Group5 : ℕ := 6
def TotalParticipantsToSelect : ℕ := 12
def TotalFromGroups345 : ℕ := Group3 + Group4 + Group5

/-
  Questions:
  1. Prove that the number of people to be selected from each group using stratified sampling:
\ 2. Prove that the probability of selecting at least one of A or B from Group 5 is 3/5.
-/

theorem selection_count_Group3 : 
  (Group3 * TotalParticipantsToSelect / TotalFromGroups345) = 6 := 
  by sorry

theorem selection_count_Group4 : 
  (Group4 * TotalParticipantsToSelect / TotalFromGroups345) = 4 := 
  by sorry

theorem selection_count_Group5 : 
  (Group5 * TotalParticipantsToSelect / TotalFromGroups345) = 2 := 
  by sorry

noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_A_or_B : 
  (combination 6 2 - combination 4 2) / combination 6 2 = 3 / 5 := 
  by sorry

end selection_count_Group3_selection_count_Group4_selection_count_Group5_probability_A_or_B_l207_207462


namespace symmetric_function_exists_l207_207768

-- Define the main sets A and B with given cardinalities
def A := { n : ℕ // n < 2011^2 }
def B := { n : ℕ // n < 2010 }

-- The main theorem to prove
theorem symmetric_function_exists :
  ∃ (f : A × A → B), 
  (∀ x y, f (x, y) = f (y, x)) ∧ 
  (∀ g : A → B, ∃ (a1 a2 : A), g a1 = f (a1, a2) ∧ g a2 = f (a1, a2) ∧ a1 ≠ a2) :=
sorry

end symmetric_function_exists_l207_207768


namespace car_meeting_distance_l207_207648

theorem car_meeting_distance
  (distance_AB : ℝ)
  (speed_A : ℝ)
  (speed_B : ℝ)
  (midpoint_C : ℝ)
  (meeting_distance_from_C : ℝ) 
  (h1 : distance_AB = 245)
  (h2 : speed_A = 70)
  (h3 : speed_B = 90)
  (h4 : midpoint_C = distance_AB / 2) :
  meeting_distance_from_C = 15.31 := 
sorry

end car_meeting_distance_l207_207648


namespace problem_statement_l207_207170

theorem problem_statement (n : ℕ) : (-1 : ℤ) ^ n * (-1) ^ (2 * n + 1) * (-1) ^ (n + 1) = 1 := 
by
  sorry

end problem_statement_l207_207170


namespace geometric_series_common_ratio_l207_207972

theorem geometric_series_common_ratio :
  ∀ (a r : ℝ), (r ≠ 1) → 
  (∑' n, a * r^n = 64 * ∑' n, a * r^(n+4)) →
  r = 1 / 2 :=
by
  intros a r hnr heq
  have hsum1 : ∑' n, a * r^n = a / (1 - r) := sorry
  have hsum2 : ∑' n, a * r^(n+4) = a * r^4 / (1 - r) := sorry
  rw [hsum1, hsum2] at heq
  -- Further steps to derive r = 1/2 are omitted
  sorry

end geometric_series_common_ratio_l207_207972


namespace boat_sinking_weight_range_l207_207133

theorem boat_sinking_weight_range
  (L_min L_max : ℝ)
  (B_min B_max : ℝ)
  (D_min D_max : ℝ)
  (sink_rate : ℝ)
  (down_min down_max : ℝ)
  (min_weight max_weight : ℝ)
  (condition1 : 3 ≤ L_min ∧ L_max ≤ 5)
  (condition2 : 2 ≤ B_min ∧ B_max ≤ 3)
  (condition3 : 1 ≤ D_min ∧ D_max ≤ 2)
  (condition4 : sink_rate = 0.01)
  (condition5 : 0.03 ≤ down_min ∧ down_max ≤ 0.06)
  (condition6 : ∀ D, D_min ≤ D ∧ D ≤ D_max → (D - down_max) ≥ 0.5)
  (condition7 : min_weight = down_min * (10 / 0.01))
  (condition8 : max_weight = down_max * (10 / 0.01)) :
  min_weight = 30 ∧ max_weight = 60 := 
sorry

end boat_sinking_weight_range_l207_207133


namespace problem1_problem2_l207_207731

-- Proof Problem 1: Prove that when \( k = 5 \), \( x^2 - 5x + 4 > 0 \) holds for \( \{x \mid x < 1 \text{ or } x > 4\} \).
theorem problem1 (x : ℝ) (h : x^2 - 5 * x + 4 > 0) : x < 1 ∨ x > 4 :=
sorry

-- Proof Problem 2: Prove that the range of values for \( k \) such that \( x^2 - kx + 4 > 0 \) holds for all real numbers \( x \) is \( (-4, 4) \).
theorem problem2 (k : ℝ) : (∀ x : ℝ, x^2 - k * x + 4 > 0) ↔ -4 < k ∧ k < 4 :=
sorry

end problem1_problem2_l207_207731


namespace schoolchildren_lineup_l207_207103

theorem schoolchildren_lineup :
  ∀ (line : list string),
  (line.length = 5) →
  (line.head = some "Borya") →
  ((∃ i j, i ≠ j ∧ (line.nth i = some "Vera") ∧ (line.nth j = some "Anya") ∧ (i + 1 = j ∨ i = j + 1)) ∧ 
   (∀ i j, (line.nth i = some "Vera") → (line.nth j = some "Gena") → ((i + 1 ≠ j) ∧ (i ≠ j + 1)))) →
  (∀ i j k, (line.nth i = some "Anya") → (line.nth j = some "Borya") → (line.nth k = some "Gena") → 
            ((i + 1 ≠ j) ∧ (i ≠ j + 1) ∧ (i + 1 ≠ k) ∧ (i ≠ k + 1))) →
  ∃ i, (line.nth i = some "Denis") ∧ 
       (((line.nth (i + 1) = some "Anya") ∨ (line.nth (i - 1) = some "Anya")) ∧ 
        ((line.nth (i + 1) = some "Gena") ∨ (line.nth (i - 1) = some "Gena")))
:= by
 sorry

end schoolchildren_lineup_l207_207103


namespace bus_children_count_l207_207826

theorem bus_children_count
  (initial_count : ℕ)
  (first_stop_add : ℕ)
  (second_stop_add : ℕ)
  (second_stop_remove : ℕ)
  (third_stop_remove : ℕ)
  (third_stop_add : ℕ)
  (final_count : ℕ)
  (h1 : initial_count = 18)
  (h2 : first_stop_add = 5)
  (h3 : second_stop_remove = 4)
  (h4 : third_stop_remove = 3)
  (h5 : third_stop_add = 5)
  (h6 : final_count = 25)
  (h7 : initial_count + first_stop_add = 23)
  (h8 : 23 + second_stop_add - second_stop_remove - third_stop_remove + third_stop_add = final_count) :
  second_stop_add = 4 :=
by
  sorry

end bus_children_count_l207_207826


namespace probability_six_integers_unique_tens_digit_l207_207473

theorem probability_six_integers_unique_tens_digit :
  (∃ (x1 x2 x3 x4 x5 x6 : ℕ),
    10 ≤ x1 ∧ x1 ≤ 79 ∧
    10 ≤ x2 ∧ x2 ≤ 79 ∧
    10 ≤ x3 ∧ x3 ≤ 79 ∧
    10 ≤ x4 ∧ x4 ≤ 79 ∧
    10 ≤ x5 ∧ x5 ≤ 79 ∧
    10 ≤ x6 ∧ x6 ≤ 79 ∧
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x1 ≠ x5 ∧ x1 ≠ x6 ∧
    x2 ≠ x3 ∧ x2 ≠ x4 ∧ x2 ≠ x5 ∧ x2 ≠ x6 ∧
    x3 ≠ x4 ∧ x3 ≠ x5 ∧ x3 ≠ x6 ∧
    x4 ≠ x5 ∧ x4 ≠ x6 ∧
    x5 ≠ x6 ∧
    tens_digit x1 ≠ tens_digit x2 ∧
    tens_digit x1 ≠ tens_digit x3 ∧
    tens_digit x1 ≠ tens_digit x4 ∧
    tens_digit x1 ≠ tens_digit x5 ∧
    tens_digit x1 ≠ tens_digit x6 ∧
    tens_digit x2 ≠ tens_digit x3 ∧
    tens_digit x2 ≠ tens_digit x4 ∧
    tens_digit x2 ≠ tens_digit x5 ∧
    tens_digit x2 ≠ tens_digit x6 ∧
    tens_digit x3 ≠ tens_digit x4 ∧
    tens_digit x3 ≠ tens_digit x5 ∧
    tens_digit x3 ≠ tens_digit x6 ∧
    tens_digit x4 ≠ tens_digit x5 ∧
    tens_digit x4 ≠ tens_digit x6 ∧
    tens_digit x5 ≠ tens_digit x6)
    →
  (probability := \(\frac{4375}{744407}\)).sorry

end probability_six_integers_unique_tens_digit_l207_207473


namespace KimSweaterTotal_l207_207212

theorem KimSweaterTotal :
  let monday := 8
  let tuesday := monday + 2
  let wednesday := tuesday - 4
  let thursday := wednesday
  let friday := monday / 2
  monday + tuesday + wednesday + thursday + friday = 34 := by
  sorry

end KimSweaterTotal_l207_207212


namespace smallest_x_l207_207660

theorem smallest_x (x : ℕ) (h : 450 * x % 648 = 0) : x = 36 := 
sorry

end smallest_x_l207_207660


namespace inscribed_circle_radius_l207_207233

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := 10
noncomputable def c : ℝ := 20

noncomputable def r : ℝ := 1 / (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))

theorem inscribed_circle_radius :
  r = 20 / (3.5 + 2 * Real.sqrt 14) :=
sorry

end inscribed_circle_radius_l207_207233


namespace cos_third_quadrant_l207_207900

theorem cos_third_quadrant (B : ℝ) (hB: π < B ∧ B < 3 * π / 2) (hSinB : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
by
  sorry

end cos_third_quadrant_l207_207900


namespace solution_set_of_x_squared_geq_four_l207_207284

theorem solution_set_of_x_squared_geq_four :
  {x : ℝ | x^2 ≥ 4} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} :=
sorry

end solution_set_of_x_squared_geq_four_l207_207284


namespace total_oranges_is_correct_l207_207078

/-- Define the number of boxes and the number of oranges per box -/
def boxes : ℕ := 7
def oranges_per_box : ℕ := 6

/-- Prove that the total number of oranges is 42 -/
theorem total_oranges_is_correct : boxes * oranges_per_box = 42 := 
by 
  sorry

end total_oranges_is_correct_l207_207078


namespace cube_side_length_is_30_l207_207352

theorem cube_side_length_is_30
  (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (total_cost : ℝ) (s : ℝ)
  (h1 : cost_per_kg = 40)
  (h2 : coverage_per_kg = 20)
  (h3 : total_cost = 10800)
  (total_surface_area : ℝ) (W : ℝ) (C : ℝ)
  (h4 : total_surface_area = 6 * s^2)
  (h5 : W = total_surface_area / coverage_per_kg)
  (h6 : C = W * cost_per_kg)
  (h7 : C = total_cost) :
  s = 30 :=
by
  sorry

end cube_side_length_is_30_l207_207352


namespace length_of_plot_l207_207949

-- Definitions of the given conditions, along with the question.
def breadth (b : ℝ) : Prop := 2 * (b + 32) + 2 * b = 5300 / 26.50
def length (b : ℝ) := b + 32

theorem length_of_plot (b : ℝ) (h : breadth b) : length b = 66 := by 
  sorry

end length_of_plot_l207_207949


namespace sum_of_remainders_l207_207371

theorem sum_of_remainders
  (a b c : ℕ)
  (h₁ : a % 36 = 15)
  (h₂ : b % 36 = 22)
  (h₃ : c % 36 = 9) :
  (a + b + c) % 36 = 10 :=
by
  sorry

end sum_of_remainders_l207_207371


namespace lena_nicole_candy_difference_l207_207071

variables (L K N : ℕ)

theorem lena_nicole_candy_difference
  (hL : L = 16)
  (hLK : L + 5 = 3 * K)
  (hKN : K = N - 4) :
  L - N = 5 :=
sorry

end lena_nicole_candy_difference_l207_207071


namespace sequence_formula_l207_207425

theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 2) (h2 : ∀ n : ℕ, a (n + 1) = 3 * a n - 2) :
  ∀ n : ℕ, a n = 3^(n - 1) + 1 :=
by sorry

end sequence_formula_l207_207425


namespace distinct_values_count_l207_207276

theorem distinct_values_count :
  ∃ (S : Finset ℕ), S.card = 104 ∧
    (∀ p q : ℕ, p ∈ Finset.range 16 → p > 0 → q ∈ Finset.range 16 → q > 0 → (p * q + p + q) ∈ S) := 
sorry

end distinct_values_count_l207_207276


namespace no_solution_a_solution_b_l207_207166

def f (n : ℕ) : ℕ :=
  if n = 0 then
    0
  else
    n / 7 + f (n / 7)

theorem no_solution_a :
  ¬ ∃ n : ℕ, 7 ^ 399 ∣ n! ∧ ¬ 7 ^ 400 ∣ n! := sorry

theorem solution_b :
  {n : ℕ | 7 ^ 400 ∣ n! ∧ ¬ 7 ^ 401 ∣ n!} = {2401, 2402, 2403, 2404, 2405, 2406, 2407} := sorry

end no_solution_a_solution_b_l207_207166


namespace quadratic_one_real_root_l207_207913

theorem quadratic_one_real_root (m : ℝ) : 
  (∃ x : ℝ, (x^2 - 6*m*x + 2*m = 0) ∧ 
    (∀ y : ℝ, (y^2 - 6*m*y + 2*m = 0) → y = x)) → 
  m = 2 / 9 :=
by
  sorry

end quadratic_one_real_root_l207_207913


namespace minimum_weighings_for_counterfeit_coin_l207_207830

/-- Given 9 coins, where 8 have equal weight and 1 is heavier (the counterfeit coin), prove that the 
minimum number of weighings required on a balance scale without weights to find the counterfeit coin is 2. -/
theorem minimum_weighings_for_counterfeit_coin (n : ℕ) (coins : Fin n → ℝ) 
  (h_n : n = 9) 
  (h_real : ∃ w : ℝ, ∀ i : Fin n, i.val < 8 → coins i = w) 
  (h_counterfeit : ∃ i : Fin n, ∀ j : Fin n, j ≠ i → coins i > coins j) : 
  ∃ k : ℕ, k = 2 :=
by
  sorry

end minimum_weighings_for_counterfeit_coin_l207_207830


namespace inequality_proof_l207_207457

theorem inequality_proof (x1 x2 x3 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) :
  (x1^2 + x2^2 + x3^2)^3 / (x1^3 + x2^3 + x3^3)^2 ≤ 3 :=
sorry

end inequality_proof_l207_207457


namespace jeff_total_jars_l207_207068

theorem jeff_total_jars (x : ℕ) : 
  16 * x + 28 * x + 40 * x + 52 * x = 2032 → 4 * x = 56 :=
by
  intro h
  -- additional steps to solve the problem would go here.
  sorry

end jeff_total_jars_l207_207068


namespace find_principal_amount_l207_207227

-- Definitions of the conditions
def rate_of_interest : ℝ := 0.20
def time_period : ℕ := 2
def interest_difference : ℝ := 144

-- Definitions for Simple Interest (SI) and Compound Interest (CI)
def simple_interest (P : ℝ) : ℝ := P * rate_of_interest * time_period
def compound_interest (P : ℝ) : ℝ := P * (1 + rate_of_interest)^time_period - P

-- Statement to prove the principal amount given the conditions
theorem find_principal_amount (P : ℝ) : 
    compound_interest P - simple_interest P = interest_difference → P = 3600 := by
    sorry

end find_principal_amount_l207_207227


namespace piper_gym_sessions_l207_207465

-- Define the conditions and the final statement as a theorem
theorem piper_gym_sessions (session_count : ℕ) (week_days : ℕ) (start_day : ℕ) 
  (alternate_day : ℕ) (skip_day : ℕ): (session_count = 35) ∧ (week_days = 7) ∧ 
  (start_day = 1) ∧ (alternate_day = 2) ∧ (skip_day = 7) → 
  (start_day + ((session_count - 1) / 3) * week_days + ((session_count - 1) % 3) * alternate_day) % week_days = 3 := 
by 
  sorry

end piper_gym_sessions_l207_207465


namespace factorization_problem_l207_207398

theorem factorization_problem :
  (∃ (h : D), 
    (¬ ∃ (a b : ℝ) (x y : ℝ), a * (x - y) = a * x - a * y) ∧
    (¬ ∃ (x : ℝ), x^2 - 2 * x + 3 = x * (x - 2) + 3) ∧
    (¬ ∃ (x : ℝ), (x - 1) * (x + 4) = x^2 + 3 * x - 4) ∧
    (∃ (x : ℝ), x^3 - 2 * x^2 + x = x * (x - 1)^2)) :=
  sorry

end factorization_problem_l207_207398


namespace find_x_l207_207027

theorem find_x (x : ℕ) : (x > 20) ∧ (x < 120) ∧ (∃ y : ℕ, x = y^2) ∧ (x % 3 = 0) ↔ (x = 36) ∨ (x = 81) :=
by
  sorry

end find_x_l207_207027


namespace geometric_series_common_ratio_l207_207970

theorem geometric_series_common_ratio (a r : ℝ) (h : a / (1 - r) = 64 * (a * r^4 / (1 - r))) : r = 1/2 :=
by {
  sorry
}

end geometric_series_common_ratio_l207_207970


namespace triangle_ab_length_triangle_roots_quadratic_l207_207291

open Real

noncomputable def right_angled_triangle_length_ab (p s : ℝ) : ℝ :=
  (p / 2) - sqrt ((p / 2)^2 - 2 * s)

noncomputable def right_angled_triangle_quadratic (p s : ℝ) : Polynomial ℝ :=
  Polynomial.X^2 - Polynomial.C ((p / 2) + sqrt ((p / 2)^2 - 2 * s)) * Polynomial.X
    + Polynomial.C (2 * s)

theorem triangle_ab_length (p s : ℝ) :
  ∃ (AB : ℝ), AB = right_angled_triangle_length_ab p s ∧
    ∃ (AC BC : ℝ), (AC + BC + AB = p) ∧ (1 / 2 * BC * AC = s) :=
by
  use right_angled_triangle_length_ab p s
  sorry

theorem triangle_roots_quadratic (p s : ℝ) :
  ∃ (AC BC : ℝ), AC + BC = (p / 2) + sqrt ((p / 2)^2 - 2 * s) ∧
    AC * BC = 2 * s ∧
    (Polynomial.aeval AC (right_angled_triangle_quadratic p s) = 0) ∧
    (Polynomial.aeval BC (right_angled_triangle_quadratic p s) = 0) :=
by
  sorry

end triangle_ab_length_triangle_roots_quadratic_l207_207291


namespace remainder_of_3024_l207_207415

theorem remainder_of_3024 (M : ℤ) (hM1 : M = 3024) (h_condition : ∃ k : ℤ, M = 24 * k + 13) :
  M % 1821 = 1203 :=
by
  sorry

end remainder_of_3024_l207_207415


namespace initial_candies_is_720_l207_207855

-- Definitions according to the conditions
def candies_remaining_after_day_n (initial_candies : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 1 => initial_candies / 2
  | 2 => (initial_candies / 2) / 3
  | 3 => (initial_candies / 2) / 3 / 4
  | 4 => (initial_candies / 2) / 3 / 4 / 5
  | 5 => (initial_candies / 2) / 3 / 4 / 5 / 6
  | _ => 0 -- For days beyond the fifth, this is nonsensical

-- Proof statement
theorem initial_candies_is_720 : ∀ (initial_candies : ℕ), candies_remaining_after_day_n initial_candies 5 = 1 → initial_candies = 720 :=
by
  intros initial_candies h
  sorry

end initial_candies_is_720_l207_207855


namespace sum_of_squares_l207_207058

theorem sum_of_squares (x y : ℝ) (h1 : y + 6 = (x - 3)^2) (h2 : x + 6 = (y - 3)^2) (hxy : x ≠ y) : x^2 + y^2 = 43 :=
sorry

end sum_of_squares_l207_207058


namespace sequence_value_at_99_l207_207609

theorem sequence_value_at_99 :
  ∃ a : ℕ → ℚ, (a 1 = 2) ∧ (∀ n : ℕ, a (n + 1) = a n + n / 2) ∧ (a 99 = 2427.5) :=
by
  sorry

end sequence_value_at_99_l207_207609


namespace smallest_n_l207_207040

theorem smallest_n (n : ℕ) (h : 0 < n) : 
  (1 / (n : ℝ)) - (1 / (n + 1 : ℝ)) < 1 / 15 → n = 4 := sorry

end smallest_n_l207_207040


namespace total_fencing_l207_207232

open Real

def playground_side_length : ℝ := 27
def garden_length : ℝ := 12
def garden_width : ℝ := 9
def flower_bed_radius : ℝ := 5
def sandpit_side1 : ℝ := 7
def sandpit_side2 : ℝ := 10
def sandpit_side3 : ℝ := 13

theorem total_fencing : 
    4 * playground_side_length + 
    2 * (garden_length + garden_width) + 
    2 * Real.pi * flower_bed_radius + 
    (sandpit_side1 + sandpit_side2 + sandpit_side3) = 211.42 := 
    by sorry

end total_fencing_l207_207232


namespace inequality_unique_solution_l207_207314

theorem inequality_unique_solution (p : ℝ) :
  (∃ x : ℝ, 0 ≤ x^2 + p * x + 5 ∧ x^2 + p * x + 5 ≤ 1) →
  (∃ x : ℝ, x^2 + p * x + 4 = 0) → p = 4 ∨ p = -4 :=
sorry

end inequality_unique_solution_l207_207314


namespace ABCD_eq_neg1_l207_207724

noncomputable def A := (Real.sqrt 2013 + Real.sqrt 2012)
noncomputable def B := (- Real.sqrt 2013 - Real.sqrt 2012)
noncomputable def C := (Real.sqrt 2013 - Real.sqrt 2012)
noncomputable def D := (Real.sqrt 2012 - Real.sqrt 2013)

theorem ABCD_eq_neg1 : A * B * C * D = -1 :=
by sorry

end ABCD_eq_neg1_l207_207724


namespace cowboy_cost_problem_l207_207120

/-- The cost of a sandwich, a cup of coffee, and a donut adds up to 0.40 dollars given the expenditure details of two cowboys. -/
theorem cowboy_cost_problem (S C D : ℝ) (h1 : 4 * S + C + 10 * D = 1.69) (h2 : 3 * S + C + 7 * D = 1.26) :
  S + C + D = 0.40 :=
by
  sorry

end cowboy_cost_problem_l207_207120


namespace rachel_milk_correct_l207_207019

-- Define the initial amount of milk Don has
def don_milk : ℚ := 1 / 5

-- Define the fraction of milk Rachel drinks
def rachel_drinks_fraction : ℚ := 2 / 3

-- Define the total amount of milk Rachel drinks
def rachel_milk : ℚ := rachel_drinks_fraction * don_milk

-- The goal is to prove that Rachel drinks a specific amount of milk
theorem rachel_milk_correct : rachel_milk = 2 / 15 :=
by
  -- The proof would be here
  sorry

end rachel_milk_correct_l207_207019


namespace standing_next_to_Denis_l207_207112

def students := ["Anya", "Borya", "Vera", "Gena", "Denis"]

def positions (line : list string) := 
  ∀ (student : string), student ∈ students → ∃ (i : ℕ), line.nth i = some student

theorem standing_next_to_Denis (line : list string) 
  (h1 : ∃ (i : ℕ), line.nth i = some "Borya" ∧ i = 0)
  (h2 : ∃ (i j k : ℕ), line.nth i = some "Anya" ∧ line.nth j = some "Vera" ∧ abs (i - j) = 1 ∧
       line.nth k = some "Gena" ∧ abs (j - k) ≠ 1)
  (h3 : ∃ (i j k : ℕ), line.nth i = some "Anya" ∧ line.nth k = some "Gena" ∧
       (abs (i - 0) ≥ 2 ∧ abs (k - 0) ≥ 2) ∧ abs (i - k) ≥ 2) :
  ∃ (a b : ℕ), line.nth a = some "Anya" ∧ line.nth b = some "Gena" ∧ 
  ∀ (d : ℕ), line.nth d = some "Denis" ∧ (abs (d - a) = 1 ∨ abs (d - b) = 1) :=
sorry

end standing_next_to_Denis_l207_207112


namespace range_of_x_l207_207483

noncomputable def function_domain (x : ℝ) : Prop :=
x + 2 > 0 ∧ x ≠ 1

theorem range_of_x {x : ℝ} (h : function_domain x) : x > -2 ∧ x ≠ 1 :=
by
  sorry

end range_of_x_l207_207483


namespace actual_time_greater_than_planned_time_l207_207550

def planned_time (a V : ℝ) : ℝ := a / V

def actual_time (a V : ℝ) : ℝ := (a / (2.5 * V)) + (a / (1.6 * V))

theorem actual_time_greater_than_planned_time (a V : ℝ) (hV : V > 0) : 
  actual_time a V > planned_time a V :=
by 
  sorry

end actual_time_greater_than_planned_time_l207_207550


namespace quadratic_solution_l207_207361

theorem quadratic_solution :
  ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 := sorry

end quadratic_solution_l207_207361


namespace constant_term_equality_l207_207330

theorem constant_term_equality (a : ℝ) 
  (h1 : ∃ T, T = (x : ℝ)^2 + 2 / x ∧ T^9 = 64 * ↑(Nat.choose 9 6)) 
  (h2 : ∃ T, T = (x : ℝ) + a / (x^2) ∧ T^9 = a^3 * ↑(Nat.choose 9 3)):
  a = 4 := 
sorry

end constant_term_equality_l207_207330


namespace values_of_n_l207_207700

theorem values_of_n (a b d : ℕ) :
  7 * a + 77 * b + 7777 * d = 6700 →
  ∃ n : ℕ, ∃ (count : ℕ), count = 107 ∧ n = a + 2 * b + 4 * d := 
by
  sorry

end values_of_n_l207_207700


namespace tablecloth_diameter_l207_207716

theorem tablecloth_diameter (r : ℝ) (h : r = 5) : 2 * r = 10 :=
by
  simp [h]
  sorry

end tablecloth_diameter_l207_207716


namespace unique_midpoints_are_25_l207_207992

/-- Define the properties of a parallelogram with marked points such as vertices, midpoints of sides, and intersection point of diagonals --/
structure Parallelogram :=
(vertices : Set ℝ)
(midpoints : Set ℝ)
(diagonal_intersection : ℝ)

def congruent_parallelograms (P P' : Parallelogram) : Prop :=
  P.vertices = P'.vertices ∧ P.midpoints = P'.midpoints ∧ P.diagonal_intersection = P'.diagonal_intersection

def unique_midpoints_count (P P' : Parallelogram) : ℕ := sorry

theorem unique_midpoints_are_25
  (P P' : Parallelogram)
  (h_congruent : congruent_parallelograms P P') :
  unique_midpoints_count P P' = 25 := sorry

end unique_midpoints_are_25_l207_207992


namespace major_axis_length_l207_207683

theorem major_axis_length (r : ℝ) (minor_axis : ℝ) (major_axis : ℝ) 
  (h1 : r = 2) 
  (h2 : minor_axis = 2 * r) 
  (h3 : major_axis = minor_axis + 0.75 * minor_axis) : 
  major_axis = 7 := 
by 
  sorry

end major_axis_length_l207_207683


namespace power_sum_identity_l207_207459

theorem power_sum_identity (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 0) : 
  (a^7 + b^7 + c^7)^2 / ((a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) * (a^4 + b^4 + c^4) * (a^5 + b^5 + c^5)) = 49 / 60 := 
by
  sorry

end power_sum_identity_l207_207459


namespace sum_of_absolute_slopes_l207_207479

theorem sum_of_absolute_slopes (P Q R S : ℤ × ℤ)
  (h1 : P = (30, 200))
  (h2 : S = (31, 215))
  (h3 : ∀ (Q R : ℤ × ℤ), Q ≠ R → 
        (Q.1 - P.1) * (R.2 - P.2) ≠ (Q.2 - P.2) * (R.1 - P.1))
  (h4 : ∀ (Q R : ℤ × ℤ), Q.1 ≠ S.1 → 
        (Q.1 - P.1) * (S.2 - R.2) ≠ (Q.2 - P.2) * (S.1 - R.1)) :
  let slopes := {7, 1 / 2, -7, -1 / 2, 1, -1, 7 / 2, -7 / 2} in
  ∑ s in slopes, abs s = 255 / 2 ∧ 255 / 2 = (255 : ℚ) / (2 : ℚ) :=
sorry

end sum_of_absolute_slopes_l207_207479


namespace problem_solution_l207_207748

theorem problem_solution (x y : ℝ) (h1 : x * y = 6) (h2 : x^2 * y + x * y^2 + x + y = 63) : x^2 + y^2 = 69 :=
by
  sorry

end problem_solution_l207_207748


namespace Joey_SAT_Weeks_l207_207613

theorem Joey_SAT_Weeks
    (hours_per_night : ℕ) (nights_per_week : ℕ)
    (hours_per_weekend_day : ℕ) (days_per_weekend : ℕ)
    (total_hours : ℕ) (weekly_hours : ℕ) (weeks : ℕ)
    (h1 : hours_per_night = 2) (h2 : nights_per_week = 5)
    (h3 : hours_per_weekend_day = 3) (h4 : days_per_weekend = 2)
    (h5 : total_hours = 96) (h6 : weekly_hours = 16)
    (h7 : weekly_hours = (hours_per_night * nights_per_week) + (hours_per_weekend_day * days_per_weekend)) :
  weeks = total_hours / weekly_hours :=
sorry

end Joey_SAT_Weeks_l207_207613


namespace cleaning_cost_l207_207614

theorem cleaning_cost (num_cleanings : ℕ) (chemical_cost : ℕ) (monthly_cost : ℕ) (tip_percentage : ℚ) 
  (cleaning_sessions_per_month : num_cleanings = 30 / 3)
  (monthly_chemical_cost : chemical_cost = 2 * 200)
  (total_monthly_cost : monthly_cost = 2050)
  (cleaning_cost_with_tip : monthly_cost - chemical_cost =  num_cleanings * (1 + tip_percentage) * x) : 
  x = 150 := 
by
  sorry

end cleaning_cost_l207_207614


namespace seats_scientific_notation_l207_207090

theorem seats_scientific_notation : 
  (13000 = 1.3 * 10^4) := 
by 
  sorry 

end seats_scientific_notation_l207_207090


namespace train_length_l207_207836

noncomputable def relative_speed_kmh (vA vB : ℝ) : ℝ :=
  vA - vB

noncomputable def relative_speed_mps (relative_speed_kmh : ℝ) : ℝ :=
  relative_speed_kmh * (5 / 18)

noncomputable def distance_covered (relative_speed_mps : ℝ) (time_s : ℝ) : ℝ :=
  relative_speed_mps * time_s

theorem train_length (vA_kmh : ℝ) (vB_kmh : ℝ) (time_s : ℝ) (L : ℝ) 
  (h1 : vA_kmh = 42) (h2 : vB_kmh = 36) (h3 : time_s = 36) 
  (h4 : distance_covered (relative_speed_mps (relative_speed_kmh vA_kmh vB_kmh)) time_s = 2 * L) :
  L = 30 :=
by
  sorry

end train_length_l207_207836


namespace petya_time_comparison_l207_207540

theorem petya_time_comparison (V : ℝ) (a : ℝ) (hV_pos : 0 < V) (ha_pos : 0 < a) :
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  T_real > T_planned :=
by
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  have T_real_eq : T_real = (a / (1.25 * V)) + (a / (0.8 * V)) := by sorry
  have T_real_simplified : T_real = (2 * a / 5 / V) + (5 * a / 8 / V) := by sorry
  have T_real_combined : T_real = (41 * a) / (40 * V) := by sorry
  have planned_vs_real : T_real > a / V := by sorry
  exact planned_vs_real

end petya_time_comparison_l207_207540


namespace solve_quadratic_completing_square_l207_207793

theorem solve_quadratic_completing_square (x : ℝ) :
  x^2 - 4 * x + 3 = 0 → (x - 2)^2 = 1 :=
by sorry

end solve_quadratic_completing_square_l207_207793


namespace smallest_positive_period_symmetry_axis_not_even_function_decreasing_interval_l207_207182

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (4 * Real.pi / 3))

theorem smallest_positive_period (T : ℝ) : T = Real.pi ↔ (∀ x : ℝ, f (x + T) = f x) := by
  sorry

theorem symmetry_axis (x : ℝ) : x = (7 * Real.pi / 12) ↔ (∀ y : ℝ, f (2 * x - y) = f y) := by
  sorry

theorem not_even_function : ¬ (∀ x : ℝ, f (x + (Real.pi / 3)) = f (-x - (Real.pi / 3))) := by
  sorry

theorem decreasing_interval (k : ℤ) (x : ℝ) : (k * Real.pi - (5 * Real.pi / 12) ≤ x ∧ x ≤ k * Real.pi + (Real.pi / 12)) ↔ (∀ x1 x2 : ℝ, x1 < x2 → f x1 ≥ f x2) := by
  sorry

end smallest_positive_period_symmetry_axis_not_even_function_decreasing_interval_l207_207182


namespace probability_of_at_least_two_consecutive_heads_equals_11_over_16_l207_207135

-- Definitions of the conditions
def fair_coin_toss_outcome_space : ℕ := 2^4

def unfavorable_outcomes : fin 5 := ⟨[0, 1, 2, 4, 8], sorry⟩ -- positions of TTTT, TTTH, TTHT, THTT, HTTT as bit representations

def probability_of_unfavorable : ℚ := 5 * (1 / fair_coin_toss_outcome_space)

-- Theorem statement
theorem probability_of_at_least_two_consecutive_heads_equals_11_over_16 :
  1 - probability_of_unfavorable = 11 / 16 :=
sorry

end probability_of_at_least_two_consecutive_heads_equals_11_over_16_l207_207135


namespace first_dilution_volume_l207_207675

theorem first_dilution_volume (x : ℝ) (V : ℝ) (red_factor : ℝ) (p : ℝ) :
  V = 1000 →
  red_factor = 25 / 3 →
  (1000 - 2 * x) * (1000 - x) = 1000 * 1000 * (3 / 25) →
  x = 400 :=
by
  intros hV hred hf
  sorry

end first_dilution_volume_l207_207675


namespace install_time_for_windows_l207_207522

theorem install_time_for_windows
  (total_windows installed_windows hours_per_window : ℕ)
  (h1 : total_windows = 200)
  (h2 : installed_windows = 65)
  (h3 : hours_per_window = 12) :
  (total_windows - installed_windows) * hours_per_window = 1620 :=
by
  sorry

end install_time_for_windows_l207_207522


namespace geometric_series_common_ratio_l207_207979

theorem geometric_series_common_ratio (a r : ℝ) (h₁ : r ≠ 1)
    (h₂ : a / (1 - r) = 64 * (a * r^4) / (1 - r)) : r = 1/2 :=
by
  have h₃ : 1 = 64 * r^4 := by
    have : 1 - r ≠ 0 := by linarith
    field_simp at h₂; assumption
  sorry

end geometric_series_common_ratio_l207_207979


namespace non_degenerate_ellipse_condition_l207_207948

theorem non_degenerate_ellipse_condition (k : ℝ) :
  (∃ x y : ℝ, 3 * x^2 + 6 * y^2 - 12 * x + 18 * y = k) ↔ k > -51 / 2 :=
sorry

end non_degenerate_ellipse_condition_l207_207948


namespace train_speed_l207_207672

theorem train_speed (length_of_train : ℝ) (time_to_cross : ℝ) (speed_of_man_km_hr : ℝ) 
  (h_length : length_of_train = 420)
  (h_time : time_to_cross = 62.99496040316775)
  (h_man_speed : speed_of_man_km_hr = 6) :
  ∃ speed_of_train_km_hr : ℝ, speed_of_train_km_hr = 30 :=
by
  sorry

end train_speed_l207_207672


namespace square_length_QP_l207_207755

theorem square_length_QP (r1 r2 dist : ℝ) (h_r1 : r1 = 10) (h_r2 : r2 = 7) (h_dist : dist = 15)
  (x : ℝ) (h_equal_chords: QP = PR) :
  x ^ 2 = 65 :=
sorry

end square_length_QP_l207_207755


namespace chuck_total_play_area_l207_207159

noncomputable def chuck_play_area (leash_radius : ℝ) : ℝ :=
  let middle_arc_area := (1 / 2) * Real.pi * leash_radius^2
  let corner_arc_area := 2 * (1 / 4) * Real.pi * leash_radius^2
  middle_arc_area + corner_arc_area

theorem chuck_total_play_area (leash_radius : ℝ) (shed_width shed_length : ℝ) 
  (h_radius : leash_radius = 4) (h_width : shed_width = 4) (h_length : shed_length = 6) :
  chuck_play_area leash_radius = 16 * Real.pi :=
by
  sorry

end chuck_total_play_area_l207_207159


namespace exists_parallelogram_C1D1_C2D2_l207_207517

open Function

-- Definitions corresponding to the conditions in the problem
variables {ω : Circle} {A B : Point}
hypothesis (A_interior : ω.contains_interior A)
hypothesis (B_on_circle : ω.contains B)

-- Statement of the theorem
theorem exists_parallelogram_C1D1_C2D2 :
  ∃ (C1 D1 C2 D2 : Point),
  ω.contains C1 ∧ ω.contains D1 ∧ ω.contains C2 ∧ ω.contains D2 ∧
  is_parallelogram A B C1 D1 ∧ is_parallelogram A B C2 D2 :=
sorry

end exists_parallelogram_C1D1_C2D2_l207_207517


namespace totalCats_l207_207682

def whiteCats : Nat := 2
def blackCats : Nat := 10
def grayCats : Nat := 3

theorem totalCats : whiteCats + blackCats + grayCats = 15 := by
  sorry

end totalCats_l207_207682


namespace number_of_red_balls_l207_207444

theorem number_of_red_balls (W R T : ℕ) (hW : W = 12) (h_freq : (R : ℝ) / (T : ℝ) = 0.25) (hT : T = W + R) : R = 4 :=
by
  sorry

end number_of_red_balls_l207_207444


namespace quadratic_solution_exists_l207_207585

-- Define the conditions
variables (a b : ℝ) (h₀ : a ≠ 0)
-- The condition that the first quadratic equation has at most one solution
def has_at_most_one_solution (a b : ℝ) : Prop :=
  b^2 + 4*a*(a - 3) <= 0

-- The second quadratic equation
def second_equation (a b x : ℝ) : ℝ :=
  (b - 3) * x^2 + (a - 2 * b) * x + 3 * a + 3
  
-- The proof problem invariant in Lean 4
theorem quadratic_solution_exists (h₁ : has_at_most_one_solution a b) :
  ∃ x : ℝ, second_equation a b x = 0 :=
by
  sorry

end quadratic_solution_exists_l207_207585


namespace distance_from_LV_to_LA_is_273_l207_207689

-- Define the conditions
def distance_SLC_to_LV : ℝ := 420
def total_time : ℝ := 11
def avg_speed : ℝ := 63

-- Define the total distance covered given the average speed and time
def total_distance : ℝ := avg_speed * total_time

-- Define the distance from Las Vegas to Los Angeles
def distance_LV_to_LA : ℝ := total_distance - distance_SLC_to_LV

-- Now state the theorem we want to prove
theorem distance_from_LV_to_LA_is_273 :
  distance_LV_to_LA = 273 :=
sorry

end distance_from_LV_to_LA_is_273_l207_207689


namespace unit_digit_4137_pow_754_l207_207128

theorem unit_digit_4137_pow_754 : (4137 ^ 754) % 10 = 9 := by
  sorry

end unit_digit_4137_pow_754_l207_207128


namespace range_of_BC_in_triangle_l207_207313

theorem range_of_BC_in_triangle 
  (A B C : ℝ) 
  (a c BC : ℝ)
  (h1 : c = Real.sqrt 2)
  (h2 : a * Real.cos C = c * Real.sin A)
  (h3 : 0 < C ∧ C < Real.pi)
  (h4 : BC = 2 * Real.sin A)
  (h5 : ∃ A1 A2, 0 < A1 ∧ A1 < Real.pi / 2 ∧ Real.pi / 2 < A2 ∧ A2 < Real.pi ∧ Real.sin A = Real.sin A1 ∧ Real.sin A = Real.sin A2)
  : BC ∈ Set.Ioo (Real.sqrt 2) 2 :=
sorry

end range_of_BC_in_triangle_l207_207313


namespace julia_played_tag_with_4_kids_on_tuesday_l207_207209

variable (k_monday : ℕ) (k_diff : ℕ)

theorem julia_played_tag_with_4_kids_on_tuesday
  (h_monday : k_monday = 16)
  (h_diff : k_monday = k_tuesday + 12) :
  k_tuesday = 4 :=
by
  sorry

end julia_played_tag_with_4_kids_on_tuesday_l207_207209


namespace find_number_l207_207670

theorem find_number (x : ℝ) (h : 45 * 7 = 0.35 * x) : x = 900 :=
by
  -- Proof (skipped with sorry)
  sorry

end find_number_l207_207670


namespace max_value_q_l207_207617

noncomputable def q (A M C : ℕ) : ℕ :=
  A * M * C + A * M + M * C + C * A + A + M + C

theorem max_value_q : ∀ A M C : ℕ, A + M + C = 15 → q A M C ≤ 215 :=
by 
  sorry

end max_value_q_l207_207617


namespace probability_of_F_l207_207861

theorem probability_of_F (P : String → ℚ) (hD : P "D" = 1/4) (hE : P "E" = 1/3) (hG : P "G" = 1/6) (total : P "D" + P "E" + P "F" + P "G" = 1) :
  P "F" = 1/4 :=
by
  sorry

end probability_of_F_l207_207861


namespace sum_first_50_natural_numbers_l207_207242

-- Define the sum of the first n natural numbers
def sum_natural (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Prove that the sum of the first 50 natural numbers is 1275
theorem sum_first_50_natural_numbers : sum_natural 50 = 1275 := 
by
  -- Skipping proof details
  sorry

end sum_first_50_natural_numbers_l207_207242


namespace inequality_holds_l207_207714

theorem inequality_holds (c : ℝ) : (∀ x : ℝ, 3 * Real.sin x - 4 * Real.cos x + c > 0) → c > 5 := by sorry

end inequality_holds_l207_207714


namespace original_average_of_15_numbers_l207_207921

theorem original_average_of_15_numbers (A : ℝ) (h1 : 15 * A + 15 * 12 = 52 * 15) :
  A = 40 :=
sorry

end original_average_of_15_numbers_l207_207921


namespace delivery_time_is_40_minutes_l207_207953

-- Define the conditions
def total_pizzas : Nat := 12
def two_pizza_stops : Nat := 2
def pizzas_per_stop_with_two_pizzas : Nat := 2
def time_per_stop_minutes : Nat := 4

-- Define the number of pizzas covered by stops with two pizzas
def pizzas_covered_by_two_pizza_stops : Nat := two_pizza_stops * pizzas_per_stop_with_two_pizzas

-- Define the number of single pizza stops
def single_pizza_stops : Nat := total_pizzas - pizzas_covered_by_two_pizza_stops

-- Define the total number of stops
def total_stops : Nat := two_pizza_stops + single_pizza_stops

-- Total time to deliver all pizzas
def total_delivery_time_minutes : Nat := total_stops * time_per_stop_minutes

theorem delivery_time_is_40_minutes : total_delivery_time_minutes = 40 := by
  sorry

end delivery_time_is_40_minutes_l207_207953


namespace geometric_series_common_ratio_l207_207971

theorem geometric_series_common_ratio (a r : ℝ) (h : a / (1 - r) = 64 * (a * r^4 / (1 - r))) : r = 1/2 :=
by {
  sorry
}

end geometric_series_common_ratio_l207_207971


namespace ratio_of_x_to_y_l207_207681

theorem ratio_of_x_to_y (x y : ℝ) (h : y = 0.20 * x) : x / y = 5 :=
by
  sorry

end ratio_of_x_to_y_l207_207681


namespace sum_faces_of_pentahedron_l207_207662

def pentahedron := {f : ℕ // f = 5}

theorem sum_faces_of_pentahedron (p : pentahedron) : p.val = 5 := 
by
  sorry

end sum_faces_of_pentahedron_l207_207662


namespace remainder_div_14_l207_207247

variables (x k : ℕ)

theorem remainder_div_14 (h : x = 142 * k + 110) : x % 14 = 12 := by 
  sorry

end remainder_div_14_l207_207247


namespace arithmetic_sequence_general_term_and_sum_max_l207_207917

-- Definitions and conditions
def a1 : ℤ := 4
def d : ℤ := -2
def a (n : ℕ) : ℤ := a1 + (n - 1) * d
def Sn (n : ℕ) : ℤ := n * (a1 + (a n)) / 2

-- Prove the general term formula and maximum value
theorem arithmetic_sequence_general_term_and_sum_max :
  (∀ n, a n = -2 * n + 6) ∧ (∃ n, Sn n = 6) :=
by
  sorry

end arithmetic_sequence_general_term_and_sum_max_l207_207917


namespace functional_eq_is_odd_function_l207_207725

theorem functional_eq_is_odd_function (f : ℝ → ℝ)
  (hf_nonzero : ∃ x : ℝ, f x ≠ 0)
  (hf_eq : ∀ a b : ℝ, f (a * b) = a * f b + b * f a) :
  ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

end functional_eq_is_odd_function_l207_207725


namespace rectangle_area_ratio_l207_207391

theorem rectangle_area_ratio (x d : ℝ) (h_ratio : 5 * x / (2 * x) = 5 / 2) (h_diag : d = 13) :
  ∃ k : ℝ, 10 * x^2 = k * d^2 ∧ k = 10 / 29 :=
by
  sorry

end rectangle_area_ratio_l207_207391


namespace five_peso_coins_count_l207_207187

theorem five_peso_coins_count (x y : ℕ) (h1 : x + y = 56) (h2 : 10 * x + 5 * y = 440) (h3 : x = 24 ∨ y = 24) : y = 24 :=
by sorry

end five_peso_coins_count_l207_207187


namespace number_of_trees_planted_l207_207490

-- Definition of initial conditions
def initial_trees : ℕ := 22
def final_trees : ℕ := 55

-- Theorem stating the number of trees planted
theorem number_of_trees_planted : final_trees - initial_trees = 33 := by
  sorry

end number_of_trees_planted_l207_207490


namespace base7_to_base10_conversion_l207_207240

def convert_base_7_to_10 := 243

namespace Base7toBase10

theorem base7_to_base10_conversion :
  2 * 7^2 + 4 * 7^1 + 3 * 7^0 = 129 := by
  -- The original number 243 in base 7 is expanded and evaluated to base 10.
  sorry

end Base7toBase10

end base7_to_base10_conversion_l207_207240


namespace geq_solution_l207_207756

def geom_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ (a (n+1) / a n) = (a 1 / a 0)

theorem geq_solution
  (a : ℕ → ℝ)
  (h_seq : geom_seq a)
  (h_cond : a 0 * a 2 + 2 * a 1 * a 3 + a 1 * a 5 = 9) :
  a 1 + a 3 = 3 :=
sorry

end geq_solution_l207_207756


namespace four_points_all_edges_red_l207_207137

def color := ℕ -- Using ℕ to represent colors, where 0 can be red and 1 can be blue.

def is_red (c : color) : Prop := c = 0

def has_red_edge (E : Finset (Finset ℕ)) (e : Finset ℕ) (col : ℕ → ℕ → color) : Prop :=
  ∃ x ∈ e, ∃ y ∈ e, x ≠ y ∧ is_red (col x y)

noncomputable def exists_red_K4 (V : Finset ℕ) (E : Finset (Finset ℕ)) (col : ℕ → ℕ → color) :=
  (V.card = 9) ∧
  (E.card = 36) ∧
  (∀ t ∈ V.powersetLen 3, has_red_edge E t col) →
  ∃ S ∈ V.powersetLen 4, ∀ e ∈ S.powersetLen 2, is_red (col e.1 e.2)

theorem four_points_all_edges_red (V : Finset ℕ) (E : Finset (Finset ℕ)) (col : ℕ → ℕ → color) :
  exists_red_K4 V E col :=
sorry

end four_points_all_edges_red_l207_207137


namespace greyhound_catches_hare_l207_207691

theorem greyhound_catches_hare {a b : ℝ} (h_speed : b < a) : ∃ t : ℝ, ∀ s : ℝ, ∃ n : ℕ, (n * t * (a - b)) > s + t * (a + b) :=
by
  sorry

end greyhound_catches_hare_l207_207691


namespace simple_interest_rate_l207_207005

-- Define the entities and conditions
variables (P A T : ℝ) (R : ℝ)

-- Conditions given in the problem
def principal := P = 12500
def amount := A = 16750
def time := T = 8

-- Result that needs to be proved
def correct_rate := R = 4.25

-- Main statement to be proven: Given the conditions, the rate is 4.25%
theorem simple_interest_rate :
  principal P → amount A → time T → (A - P = (P * R * T) / 100) → correct_rate R :=
by
  intros hP hA hT hSI
  sorry

end simple_interest_rate_l207_207005


namespace tan_identity_proof_l207_207034

noncomputable def tan_add_pi_over_3 (α β : ℝ) : ℝ :=
  Real.tan (α + Real.pi / 3)

theorem tan_identity_proof 
  (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3 / 5)
  (h2 : Real.tan (β - Real.pi / 3) = 1 / 4) :
  tan_add_pi_over_3 α β = 7 / 23 := 
sorry

end tan_identity_proof_l207_207034


namespace cosine_of_angle_in_third_quadrant_l207_207906

theorem cosine_of_angle_in_third_quadrant (B : ℝ) (hB : B ∈ Set.Ioo (π : ℝ) (3 * π / 2)) (hSinB : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
sorry

end cosine_of_angle_in_third_quadrant_l207_207906


namespace total_robodinos_in_shipment_l207_207525

-- Definitions based on the conditions:
def percentage_on_display : ℝ := 0.30
def percentage_in_storage : ℝ := 0.70
def stored_robodinos : ℕ := 168

-- The main statement to prove:
theorem total_robodinos_in_shipment (T : ℝ) : (percentage_in_storage * T = stored_robodinos) → T = 240 := by
  sorry

end total_robodinos_in_shipment_l207_207525


namespace Emily_total_points_l207_207278

-- Definitions of the points scored in each round
def round1_points := 16
def round2_points := 32
def round3_points := -27
def round4_points := 92
def round5_points := 4

-- Total points calculation in Lean
def total_points := round1_points + round2_points + round3_points + round4_points + round5_points

-- Lean statement to prove total points at the end of the game
theorem Emily_total_points : total_points = 117 :=
by 
  -- Unfold the definition of total_points and simplify
  unfold total_points round1_points round2_points round3_points round4_points round5_points
  -- Simplify the expression
  sorry

end Emily_total_points_l207_207278


namespace find_a3_plus_a9_l207_207728

noncomputable def arithmetic_sequence (a : ℕ → ℕ) : Prop := 
∀ n m : ℕ, a (n + m) = a n + a m

theorem find_a3_plus_a9 (a : ℕ → ℕ) 
  (is_arithmetic : arithmetic_sequence a)
  (h : a 1 + a 6 + a 11 = 3) : 
  a 3 + a 9 = 2 :=
sorry

end find_a3_plus_a9_l207_207728


namespace g_minus3_is_correct_l207_207635

theorem g_minus3_is_correct (g : ℚ → ℚ) (h : ∀ x : ℚ, x ≠ 0 → 4 * g (1 / x) + 3 * g x / x = 3 * x^2) : 
  g (-3) = 247 / 39 :=
by
  sorry

end g_minus3_is_correct_l207_207635


namespace count_valid_three_digit_numbers_l207_207437

def three_digit_number (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a * 100 + b * 10 + c < 1000) ∧
  (a * 100 + b * 10 + c >= 100) ∧
  (c = 2 * (b - a) + a)

theorem count_valid_three_digit_numbers : ∃ n : ℕ, n = 90 ∧
  ∃ (a b c : ℕ), three_digit_number a b c :=
by
  sorry

end count_valid_three_digit_numbers_l207_207437


namespace petya_time_comparison_l207_207533

theorem petya_time_comparison (V a : ℝ) (hV : V > 0) (ha : a > 0) :
  (a / V) < ((a / (2.5 * V)) + (a / (1.6 * V))) := by
  have T_planned : ℝ := a / V
  have T_first_half : ℝ := a / (2.5 * V)
  have T_second_half : ℝ := a / (1.6 * V)
  have T_real : ℝ := T_first_half + T_second_half
  sorry

end petya_time_comparison_l207_207533


namespace base_for_195₁₀_four_digit_even_final_digit_l207_207878

theorem base_for_195₁₀_four_digit_even_final_digit :
  ∃ b : ℕ, (b^3 ≤ 195 ∧ 195 < b^4) ∧ (∃ d : ℕ, 195 % b = d ∧ d % 2 = 0) ∧ b = 5 :=
by {
  sorry
}

end base_for_195₁₀_four_digit_even_final_digit_l207_207878


namespace james_total_earnings_l207_207611

-- Define the earnings for January
def januaryEarnings : ℕ := 4000

-- Define the earnings for February based on January
def februaryEarnings : ℕ := 2 * januaryEarnings

-- Define the earnings for March based on February
def marchEarnings : ℕ := februaryEarnings - 2000

-- Define the total earnings including January, February, and March
def totalEarnings : ℕ := januaryEarnings + februaryEarnings + marchEarnings

-- State the theorem: total earnings should be 18000
theorem james_total_earnings : totalEarnings = 18000 := by
  sorry

end james_total_earnings_l207_207611


namespace problem_solution_l207_207055

theorem problem_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end problem_solution_l207_207055


namespace number_of_books_about_trains_l207_207162

theorem number_of_books_about_trains
  (books_animals : ℕ)
  (books_outer_space : ℕ)
  (book_cost : ℕ)
  (total_spent : ℕ)
  (T : ℕ)
  (hyp1 : books_animals = 8)
  (hyp2 : books_outer_space = 6)
  (hyp3 : book_cost = 6)
  (hyp4 : total_spent = 102)
  (hyp5 : total_spent = (books_animals + books_outer_space + T) * book_cost)
  : T = 3 := by
  sorry

end number_of_books_about_trains_l207_207162


namespace dima_age_l207_207342

variable (x : ℕ)

-- Dima's age is x years
def age_of_dima := x

-- Dima's age is twice his brother's age
def age_of_brother := x / 2

-- Dima's age is three times his sister's age
def age_of_sister := x / 3

-- The average age of Dima, his sister, and his brother is 11 years
def average_age := (x + age_of_brother x + age_of_sister x) / 3 = 11

theorem dima_age (h1 : age_of_brother x = x / 2) 
                 (h2 : age_of_sister x = x / 3) 
                 (h3 : average_age x) : x = 18 := 
by sorry

end dima_age_l207_207342


namespace candy_given_away_l207_207932

-- Define the conditions
def pieces_per_student := 2
def number_of_students := 9

-- Define the problem statement as a theorem
theorem candy_given_away : pieces_per_student * number_of_students = 18 := by
  -- This is where the proof would go, but we omit it with sorry.
  sorry

end candy_given_away_l207_207932


namespace length_of_AB_l207_207466

theorem length_of_AB 
  (P Q A B : ℝ)
  (h_P_on_AB : P > 0 ∧ P < B)
  (h_Q_on_AB : Q > P ∧ Q < B)
  (h_ratio_P : P = 3 / 7 * B)
  (h_ratio_Q : Q = 4 / 9 * B)
  (h_PQ : Q - P = 3) 
: B = 189 := 
sorry

end length_of_AB_l207_207466


namespace find_intersection_point_l207_207282

theorem find_intersection_point :
  ∃ (x y z : ℝ), 
    ((∃ t : ℝ, x = 1 + 2 * t ∧ y = 1 - t ∧ z = -2 + 3 * t) ∧ 
    (4 * x + 2 * y - z - 11 = 0)) ∧ 
    (x = 3 ∧ y = 0 ∧ z = 1) :=
by
  sorry

end find_intersection_point_l207_207282


namespace hexagon_midpoints_equilateral_l207_207856

noncomputable def inscribed_hexagon_midpoints_equilateral (r : ℝ) (h : ℝ) 
  (hex : ∀ (A B C D E F : ℝ) (O : ℝ), 
    true) : Prop :=
  ∀ (M N P : ℝ), 
    true

theorem hexagon_midpoints_equilateral (r : ℝ) (h : ℝ) 
  (hex : ∀ (A B C D E F : ℝ) (O : ℝ), 
    true) : 
  inscribed_hexagon_midpoints_equilateral r h hex :=
sorry

end hexagon_midpoints_equilateral_l207_207856


namespace division_proof_l207_207666

-- Define the given condition
def given_condition : Prop :=
  2084.576 / 135.248 = 15.41

-- Define the problem statement we want to prove
def problem_statement : Prop :=
  23.8472 / 13.5786 = 1.756

-- Main theorem stating that under the given condition, the problem statement holds
theorem division_proof (h : given_condition) : problem_statement :=
by sorry

end division_proof_l207_207666


namespace fraction_spent_on_fruits_l207_207004

theorem fraction_spent_on_fruits (M : ℕ) (hM : M = 24) :
  (M - (M / 3 + M / 6) - 6) / M = 1 / 4 :=
by
  sorry

end fraction_spent_on_fruits_l207_207004


namespace proof1_proof2_l207_207035

noncomputable def a (n : ℕ) : ℝ := (n^2 + 1) * 3^n

def recurrence_relation : Prop :=
  ∀ n, a (n + 3) - 9 * a (n + 2) + 27 * a (n + 1) - 27 * a n = 0

noncomputable def series_sum (x : ℝ) : ℝ :=
  ∑' n, a n * x^n

def series_evaluation (x : ℝ) : Prop :=
  series_sum x = (1 - 3*x + 18*x^2) / (1 - 3*x)^3

theorem proof1 : recurrence_relation := 
  by sorry

theorem proof2 : ∀ x : ℝ, series_evaluation x := 
  by sorry

end proof1_proof2_l207_207035


namespace jason_tattoos_on_each_leg_l207_207688

-- Define the basic setup
variable (x : ℕ)

-- Define the number of tattoos Jason has on each leg
def tattoos_on_each_leg := x

-- Define the total number of tattoos Jason has
def total_tattoos_jason := 2 + 2 + 2 * x

-- Define the total number of tattoos Adam has
def total_tattoos_adam := 23

-- Define the relation between Adam's and Jason's tattoos
def relation := 2 * total_tattoos_jason + 3 = total_tattoos_adam

-- The proof statement we need to show
theorem jason_tattoos_on_each_leg : tattoos_on_each_leg = 3  :=
by
  sorry

end jason_tattoos_on_each_leg_l207_207688


namespace solve_z_l207_207942

variable (z : ℂ) -- Define the variable z in the complex number system
variable (i : ℂ) -- Define the variable i in the complex number system

-- State the conditions: 2 - 3i * z = 4 + 5i * z and i^2 = -1
axiom cond1 : 2 - 3 * i * z = 4 + 5 * i * z
axiom cond2 : i^2 = -1

-- The theorem to prove: z = i / 4
theorem solve_z : z = i / 4 :=
by
  sorry

end solve_z_l207_207942


namespace area_triangle_ABC_correct_l207_207161

noncomputable def rectangle_area : ℝ := 42

noncomputable def area_triangle_outside_I : ℝ := 9
noncomputable def area_triangle_outside_II : ℝ := 3.5
noncomputable def area_triangle_outside_III : ℝ := 12

noncomputable def area_triangle_ABC : ℝ :=
  rectangle_area - (area_triangle_outside_I + area_triangle_outside_II + area_triangle_outside_III)

theorem area_triangle_ABC_correct : area_triangle_ABC = 17.5 := by 
  sorry

end area_triangle_ABC_correct_l207_207161


namespace range_of_independent_variable_l207_207360

theorem range_of_independent_variable (x : ℝ) (hx : 1 - 2 * x ≥ 0) : x ≤ 0.5 :=
sorry

end range_of_independent_variable_l207_207360


namespace avg_stoppage_time_is_20_minutes_l207_207165

noncomputable def avg_stoppage_time : Real :=
let train1 := (60, 40) -- without stoppages, with stoppages (in kmph)
let train2 := (75, 50) -- without stoppages, with stoppages (in kmph)
let train3 := (90, 60) -- without stoppages, with stoppages (in kmph)
let time1 := (train1.1 - train1.2 : Real) / train1.1
let time2 := (train2.1 - train2.2 : Real) / train2.1
let time3 := (train3.1 - train3.2 : Real) / train3.1
let total_time := time1 + time2 + time3
(total_time / 3) * 60 -- convert hours to minutes

theorem avg_stoppage_time_is_20_minutes :
  avg_stoppage_time = 20 :=
sorry

end avg_stoppage_time_is_20_minutes_l207_207165


namespace paul_crayons_left_l207_207785

theorem paul_crayons_left (initial_crayons lost_crayons : ℕ) 
  (h_initial : initial_crayons = 253) 
  (h_lost : lost_crayons = 70) : (initial_crayons - lost_crayons) = 183 := 
by
  sorry

end paul_crayons_left_l207_207785


namespace max_E_l207_207710

def E (x₁ x₂ x₃ x₄ : ℝ) : ℝ :=
  x₁ + x₂ + x₃ + x₄ -
  x₁ * x₂ - x₁ * x₃ - x₁ * x₄ -
  x₂ * x₃ - x₂ * x₄ - x₃ * x₄ +
  x₁ * x₂ * x₃ + x₁ * x₂ * x₄ +
  x₁ * x₃ * x₄ + x₂ * x₃ * x₄ -
  x₁ * x₂ * x₃ * x₄

theorem max_E (x₁ x₂ x₃ x₄ : ℝ) (h₁ : 0 ≤ x₁) (h₂ : x₁ ≤ 1) (h₃ : 0 ≤ x₂) (h₄ : x₂ ≤ 1) (h₅ : 0 ≤ x₃) (h₆ : x₃ ≤ 1) (h₇ : 0 ≤ x₄) (h₈ : x₄ ≤ 1) : 
  E x₁ x₂ x₃ x₄ ≤ 1 :=
sorry

end max_E_l207_207710


namespace binomial_pmf_l207_207188

open ProbabilityTheory
open Finset

variables (X : ℕ → ℕ) [H : measure_theory.MeasureSpace (sample_space X)]

noncomputable def binomial_pmf (n : ℕ) (p : ℚ) : pmf (fin (n+1)) :=
pmf.uniform_of_finset (finset.range (n + 1)) (λ k, (finset.choose n k : ℚ) * p^k * (1 - p)^(n - k))

variable (n : ℕ)
variable (p : ℚ)
variable (h₁ : X ~ binomial_pmf n p)
variable (h₂ : (X 1) = 6)
variable (h₃ : (X 1) = 3)

theorem binomial_pmf.X_eq_1_prob : 
  (pmf.prob (binomial_pmf 12 (1 / 2)) (1 : fin 13)) = 3 / 2^10 := 
sorry

end binomial_pmf_l207_207188


namespace total_weight_l207_207788

def weight_of_blue_ball : ℝ := 6.0
def weight_of_brown_ball : ℝ := 3.12

theorem total_weight (_ : weight_of_blue_ball = 6.0) (_ : weight_of_brown_ball = 3.12) : 
  weight_of_blue_ball + weight_of_brown_ball = 9.12 :=
by
  sorry

end total_weight_l207_207788


namespace total_prayers_in_a_week_l207_207624

def prayers_per_week (pastor_prayers : ℕ → ℕ) : ℕ :=
  (pastor_prayers 0) + (pastor_prayers 1) + (pastor_prayers 2) +
  (pastor_prayers 3) + (pastor_prayers 4) + (pastor_prayers 5) + (pastor_prayers 6)

def pastor_paul (day : ℕ) : ℕ :=
  if day = 6 then 40 else 20

def pastor_bruce (day : ℕ) : ℕ :=
  if day = 6 then 80 else 10

def pastor_caroline (day : ℕ) : ℕ :=
  if day = 6 then 30 else 10

theorem total_prayers_in_a_week :
  prayers_per_week pastor_paul + prayers_per_week pastor_bruce + prayers_per_week pastor_caroline = 390 :=
sorry

end total_prayers_in_a_week_l207_207624


namespace find_intersection_point_l207_207521

/-- Definition of the parabola -/
def parabola (y : ℝ) : ℝ := -3 * y ^ 2 - 4 * y + 7

/-- Condition for intersection at exactly one point -/
def discriminant (m : ℝ) : ℝ := 4 ^ 2 - 4 * 3 * (m - 7)

/-- Main theorem stating the proof problem -/
theorem find_intersection_point (m : ℝ) :
  (discriminant m = 0) → m = 25 / 3 :=
by
  sorry

end find_intersection_point_l207_207521


namespace relative_error_comparison_l207_207002

theorem relative_error_comparison :
  let error1 := 0.05
  let length1 := 25
  let error2 := 0.25
  let length2 := 125
  (error1 / length1) = (error2 / length2) :=
by
  sorry

end relative_error_comparison_l207_207002


namespace floor_of_negative_sqrt_l207_207021

noncomputable def eval_expr : ℚ := -real.sqrt (64 / 9)

theorem floor_of_negative_sqrt : ⌊eval_expr⌋ = -3 :=
by
  -- skip proof
  sorry

end floor_of_negative_sqrt_l207_207021


namespace problem1_problem2_l207_207795

-- Problem 1: Solution set for x(7 - x) >= 12
theorem problem1 (x : ℝ) : x * (7 - x) ≥ 12 ↔ (3 ≤ x ∧ x ≤ 4) :=
by
  sorry

-- Problem 2: Solution set for x^2 > 2(x - 1)
theorem problem2 (x : ℝ) : x^2 > 2 * (x - 1) ↔ true :=
by
  sorry

end problem1_problem2_l207_207795


namespace part1_solution_part2_solution_part3_solution_l207_207865

-- Define the basic conditions
variables (x y m : ℕ)

-- Part 1: Number of pieces of each type purchased (Proof for 10 pieces of A, 20 pieces of B)
theorem part1_solution (h1 : x + y = 30) (h2 : 28 * x + 22 * y = 720) :
  (x = 10) ∧ (y = 20) :=
sorry

-- Part 2: Maximize sales profit for the second purchase
theorem part2_solution (h1 : 28 * m + 22 * (80 - m) ≤ 2000) :
  m = 40 ∧ (max_profit = 1040) :=
sorry

-- Variables for Part 3
variables (a : ℕ)
-- Profit equation for type B apples with adjusted selling price
theorem part3_solution (h : (4 + 2 * a) * (34 - a - 22) = 90) :
  (a = 7) ∧ (selling_price = 27) :=
sorry

end part1_solution_part2_solution_part3_solution_l207_207865


namespace rational_range_l207_207315

theorem rational_range (a : ℚ) (h : a - |a| = 2 * a) : a ≤ 0 := 
sorry

end rational_range_l207_207315


namespace range_a_and_inequality_l207_207221

noncomputable def f (x a : ℝ) : ℝ := x^2 - a * Real.log (x + 2)
noncomputable def f' (x a : ℝ) : ℝ := 2 * x - a / (x + 2)

theorem range_a_and_inequality (a x1 x2 : ℝ) (h_deriv: ∀ (x : ℝ), f' x a = 0 → x = x1 ∨ x = x2) (h_lt: x1 < x2) (h_extreme: f (x1) a = f (x2) a):
  (-2 < a ∧ a < 0) → 
  (f (x1) a / x2 + 1 < 0) :=
by
  sorry

end range_a_and_inequality_l207_207221


namespace term_value_in_sequence_l207_207306

theorem term_value_in_sequence (a : ℕ → ℕ) (n : ℕ) (h : ∀ n, a n = n * (n + 2) / 2) (h_val : a n = 220) : n = 20 :=
  sorry

end term_value_in_sequence_l207_207306


namespace regular_polygon_sides_l207_207563

theorem regular_polygon_sides (n : ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → (interior_angle i = 160)) : n = 18 :=
by
  -- Proof goes here
  sorry

end regular_polygon_sides_l207_207563


namespace closest_integer_to_cuberoot_150_l207_207503

theorem closest_integer_to_cuberoot_150 : 
  let cube5 := 5^3 in 
  let cube6 := 6^3 in 
  let midpoint := (cube5 + cube6) / 2 in 
  125 < 150 ∧ 150 < 216 ∧ 150 < midpoint → 
  5 = round (150^(1/3)) := 
by 
  intro h
  sorry

end closest_integer_to_cuberoot_150_l207_207503


namespace units_digit_35_87_plus_93_49_l207_207668

theorem units_digit_35_87_plus_93_49 : (35^87 + 93^49) % 10 = 8 := by
  sorry

end units_digit_35_87_plus_93_49_l207_207668


namespace maximum_lambda_l207_207413

theorem maximum_lambda (a b : ℝ) : (27 / 4) * a^2 * b^2 * (a + b)^2 ≤ (a^2 + a * b + b^2)^3 := 
sorry

end maximum_lambda_l207_207413


namespace problem_statement_l207_207376

noncomputable def probability_different_colors : ℚ :=
  let p_red := 7 / 11
  let p_green := 4 / 11
  (p_red * p_green) + (p_green * p_red)

theorem problem_statement :
  let p_red := 7 / 11
  let p_green := 4 / 11
  (p_red * p_green) + (p_green * p_red) = 56 / 121 := by
  sorry

end problem_statement_l207_207376


namespace KimSweaterTotal_l207_207211

theorem KimSweaterTotal :
  let monday := 8
  let tuesday := monday + 2
  let wednesday := tuesday - 4
  let thursday := wednesday
  let friday := monday / 2
  monday + tuesday + wednesday + thursday + friday = 34 := by
  sorry

end KimSweaterTotal_l207_207211


namespace fraction_of_area_above_line_l207_207642

open Real

-- Define the points and the line between them
noncomputable def pointA : (ℝ × ℝ) := (2, 3)
noncomputable def pointB : (ℝ × ℝ) := (5, 1)

-- Define the vertices of the square
noncomputable def square_vertices : List (ℝ × ℝ) := [(2, 1), (5, 1), (5, 4), (2, 4)]

-- Define the equation of the line
noncomputable def line_eq (x : ℝ) : ℝ :=
  (-2/3) * x + 13/3

-- Define the vertical and horizontal boundaries
noncomputable def x_min : ℝ := 2
noncomputable def x_max : ℝ := 5
noncomputable def y_min : ℝ := 1
noncomputable def y_max : ℝ := 4

-- Calculate the area of the triangle formed below the line
noncomputable def triangle_area : ℝ := 0.5 * 2 * 3

-- Calculate the area of the square
noncomputable def square_area : ℝ := 3 * 3

-- The fraction of the area above the line
noncomputable def area_fraction_above : ℝ := (square_area - triangle_area) / square_area

-- Prove the fraction of the area of the square above the line is 2/3
theorem fraction_of_area_above_line : area_fraction_above = 2 / 3 :=
  sorry

end fraction_of_area_above_line_l207_207642


namespace petya_time_l207_207535

variable (a V : ℝ)

noncomputable def planned_time := a / V
noncomputable def real_time := (a / (2.5 * V)) + (a / (1.6 * V))

theorem petya_time (hV : V > 0) (ha : a > 0) : real_time a V > planned_time a V :=
by
  sorry

end petya_time_l207_207535


namespace total_pennies_l207_207464

theorem total_pennies (rachelle_pennies : ℕ) (gretchen_pennies : ℕ) (rocky_pennies : ℕ)
  (h1 : rachelle_pennies = 180)
  (h2 : gretchen_pennies = rachelle_pennies / 2)
  (h3 : rocky_pennies = gretchen_pennies / 3) :
  rachelle_pennies + gretchen_pennies + rocky_pennies = 300 :=
by
  sorry

end total_pennies_l207_207464


namespace total_value_of_coins_is_correct_l207_207872

def rolls_dollars : ℕ := 6
def rolls_half_dollars : ℕ := 5
def rolls_quarters : ℕ := 7
def rolls_dimes : ℕ := 4
def rolls_nickels : ℕ := 3
def rolls_pennies : ℕ := 2

def coins_per_dollar_roll : ℕ := 20
def coins_per_half_dollar_roll : ℕ := 25
def coins_per_quarter_roll : ℕ := 40
def coins_per_dime_roll : ℕ := 50
def coins_per_nickel_roll : ℕ := 40
def coins_per_penny_roll : ℕ := 50

def value_per_dollar : ℚ := 1
def value_per_half_dollar : ℚ := 0.5
def value_per_quarter : ℚ := 0.25
def value_per_dime : ℚ := 0.10
def value_per_nickel : ℚ := 0.05
def value_per_penny : ℚ := 0.01

theorem total_value_of_coins_is_correct : 
  rolls_dollars * coins_per_dollar_roll * value_per_dollar +
  rolls_half_dollars * coins_per_half_dollar_roll * value_per_half_dollar +
  rolls_quarters * coins_per_quarter_roll * value_per_quarter +
  rolls_dimes * coins_per_dime_roll * value_per_dime +
  rolls_nickels * coins_per_nickel_roll * value_per_nickel +
  rolls_pennies * coins_per_penny_roll * value_per_penny = 279.50 := 
sorry

end total_value_of_coins_is_correct_l207_207872


namespace diameter_of_large_circle_is_19_312_l207_207704

noncomputable def diameter_large_circle (r_small : ℝ) (n : ℕ) : ℝ :=
  let side_length_inner_octagon := 2 * r_small
  let radius_inner_octagon := side_length_inner_octagon / (2 * Real.sin (Real.pi / n)) / 2
  let radius_large_circle := radius_inner_octagon + r_small
  2 * radius_large_circle

theorem diameter_of_large_circle_is_19_312 :
  diameter_large_circle 4 8 = 19.312 :=
by
  sorry

end diameter_of_large_circle_is_19_312_l207_207704


namespace school_children_count_l207_207848

theorem school_children_count (C B : ℕ) (h1 : B = 2 * C) (h2 : B = 4 * (C - 370)) : C = 740 :=
by sorry

end school_children_count_l207_207848


namespace range_of_a_l207_207430

-- Defining the function f
noncomputable def f (x a : ℝ) : ℝ :=
  (Real.exp x) * (2 * x - 1) - a * x + a

-- Main statement
theorem range_of_a (a : ℝ)
  (h1 : a < 1)
  (h2 : ∃ x0 x1 : ℤ, x0 ≠ x1 ∧ f x0 a ≤ 0 ∧ f x1 a ≤ 0) :
  (5 / (3 * Real.exp 2)) < a ∧ a ≤ (3 / (2 * Real.exp 1)) :=
sorry

end range_of_a_l207_207430


namespace gcd_102_238_l207_207837

theorem gcd_102_238 : Nat.gcd 102 238 = 34 :=
by
  sorry

end gcd_102_238_l207_207837


namespace perpendicular_probability_l207_207717

def S : Set ℚ := {-3, -5/4, -1/2, 0, 1/3, 1, 4/5, 2}

def is_perpendicular_pair (a b : ℚ) : Prop :=
  a * b = -1

def pairs (s : Set ℚ) : Set (ℚ × ℚ) :=
  { p | p.1 ∈ s ∧ p.2 ∈ s ∧ p.1 ≠ p.2 }

def perpendicular_pairs (s : Set ℚ) : Set (ℚ × ℚ) :=
  { p ∈ pairs s | is_perpendicular_pair p.1 p.2 }

def total_pairs_count (s : Set ℚ) : ℕ :=
  (s.card.choose 2)

def favorable_pairs_count (s : Set ℚ) : ℕ :=
  (perpendicular_pairs s).card

theorem perpendicular_probability :
  let favorable_count := favorable_pairs_count S
  let total_count := total_pairs_count S
  total_count ≠ 0 →
  (favorable_count : ℚ) / total_count = 3 / 28 :=
by
  let favorable_count := favorable_pairs_count S
  let total_count := total_pairs_count S
  intro h
  sorry

end perpendicular_probability_l207_207717


namespace smallest_x_for_multiple_of_450_and_648_l207_207658

theorem smallest_x_for_multiple_of_450_and_648 (x : ℕ) (hx : x > 0) :
  ∃ (y : ℕ), (450 * 36) = y ∧ (450 * 36) % 648 = 0 :=
by
  use (450 / gcd 450 648 * 648 / gcd 450 648)
  sorry

end smallest_x_for_multiple_of_450_and_648_l207_207658


namespace initial_weight_of_mixture_eq_20_l207_207389

theorem initial_weight_of_mixture_eq_20
  (W : ℝ) (h1 : 0.1 * W + 4 = 0.25 * (W + 4)) :
  W = 20 :=
by
  sorry

end initial_weight_of_mixture_eq_20_l207_207389


namespace denis_neighbors_l207_207113

-- Define positions
inductive Position
| P1 | P2 | P3 | P4 | P5

open Position

-- Declare the children
inductive Child
| Anya | Borya | Vera | Gena | Denis

open Child

def next_to (p1 p2 : Position) : Prop := 
  (p1 = P1 ∧ p2 = P2) ∨ (p1 = P2 ∧ p2 = P1) ∨
  (p1 = P2 ∧ p2 = P3) ∨ (p1 = P3 ∧ p2 = P2) ∨
  (p1 = P3 ∧ p2 = P4) ∨ (p1 = P4 ∧ p2 = P3) ∨
  (p1 = P4 ∧ p2 = P5) ∨ (p1 = P5 ∧ p2 = P4)

variables (pos : Child → Position)

-- Given conditions
axiom borya_beginning : pos Borya = P1
axiom vera_next_to_anya : next_to (pos Vera) (pos Anya)
axiom vera_not_next_to_gena : ¬ next_to (pos Vera) (pos Gena)
axiom no_two_next_to : ∀ (c1 c2 : Child), 
  c1 ∈ [Anya, Borya, Gena] → c2 ∈ [Anya, Borya, Gena] → c1 ≠ c2 → ¬ next_to (pos c1) (pos c2)

-- Prove the result
theorem denis_neighbors : next_to (pos Denis) (pos Anya) ∧ next_to (pos Denis) (pos Gena) :=
sorry

end denis_neighbors_l207_207113


namespace minimum_value_l207_207740

theorem minimum_value (x : ℝ) (h : x > -3) : 2 * x + (1 / (x + 3)) ≥ 2 * Real.sqrt 2 - 6 :=
sorry

end minimum_value_l207_207740


namespace Nancy_hourly_wage_l207_207782

def tuition_cost := 22000
def parents_coverage := tuition_cost / 2
def scholarship := 3000
def loan := 2 * scholarship
def working_hours := 200
def remaining_tuition := tuition_cost - parents_coverage - scholarship - loan
def hourly_wage_required := remaining_tuition / working_hours

theorem Nancy_hourly_wage : hourly_wage_required = 10 := by
  sorry

end Nancy_hourly_wage_l207_207782


namespace points_on_x_eq_3_is_vertical_line_points_with_x_lt_3_points_with_x_gt_3_points_on_y_eq_2_is_horizontal_line_points_with_y_gt_2_l207_207655

open Set

-- Define the point in the coordinate plane as a product of real numbers
def Point := ℝ × ℝ

-- Prove points with x = 3 form a vertical line
theorem points_on_x_eq_3_is_vertical_line : {p : Point | p.1 = 3} = {p : Point | ∀ y : ℝ, (3, y) = p} := sorry

-- Prove points with x < 3 lie to the left of x = 3
theorem points_with_x_lt_3 : {p : Point | p.1 < 3} = {p : Point | ∀ x y : ℝ, x < 3 → p = (x, y)} := sorry

-- Prove points with x > 3 lie to the right of x = 3
theorem points_with_x_gt_3 : {p : Point | p.1 > 3} = {p : Point | ∀ x y : ℝ, x > 3 → p = (x, y)} := sorry

-- Prove points with y = 2 form a horizontal line
theorem points_on_y_eq_2_is_horizontal_line : {p : Point | p.2 = 2} = {p : Point | ∀ x : ℝ, (x, 2) = p} := sorry

-- Prove points with y > 2 lie above y = 2
theorem points_with_y_gt_2 : {p : Point | p.2 > 2} = {p : Point | ∀ x y : ℝ, y > 2 → p = (x, y)} := sorry

end points_on_x_eq_3_is_vertical_line_points_with_x_lt_3_points_with_x_gt_3_points_on_y_eq_2_is_horizontal_line_points_with_y_gt_2_l207_207655


namespace convert_to_scientific_notation_l207_207801

-- Problem statement: convert 120 million to scientific notation and validate the format.
theorem convert_to_scientific_notation :
  120000000 = 1.2 * 10^7 :=
sorry

end convert_to_scientific_notation_l207_207801


namespace k_range_for_two_zeros_of_f_l207_207298

noncomputable def f (x k : ℝ) : ℝ := x^2 - x * (Real.log x) - k * (x + 2) + 2

theorem k_range_for_two_zeros_of_f :
  ∀ k : ℝ, (∃ x1 x2 : ℝ, (1/2 < x1) ∧ (x1 < x2) ∧ f x1 k = 0 ∧ f x2 k = 0) ↔ 1 < k ∧ k ≤ (9 + 2 * Real.log 2) / 10 :=
by
  sorry

end k_range_for_two_zeros_of_f_l207_207298


namespace meaningful_fraction_condition_l207_207991

theorem meaningful_fraction_condition (x : ℝ) : (4 - 2 * x ≠ 0) ↔ (x ≠ 2) :=
by {
  sorry
}

end meaningful_fraction_condition_l207_207991


namespace bruce_bank_ratio_l207_207156

noncomputable def bruce_aunt : ℝ := 75
noncomputable def bruce_grandfather : ℝ := 150
noncomputable def bruce_bank : ℝ := 45
noncomputable def bruce_total : ℝ := bruce_aunt + bruce_grandfather
noncomputable def bruce_ratio : ℝ := bruce_bank / bruce_total

theorem bruce_bank_ratio :
  bruce_ratio = 1 / 5 :=
by
  -- proof goes here
  sorry

end bruce_bank_ratio_l207_207156


namespace complement_of_M_l207_207052

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 - 4 ≤ 0}

theorem complement_of_M :
  ∀ x, x ∈ U \ M ↔ x < -2 ∨ x > 2 :=
by
  sorry

end complement_of_M_l207_207052


namespace marbles_lost_l207_207615

theorem marbles_lost (m_initial m_current : ℕ) (h_initial : m_initial = 19) (h_current : m_current = 8) : m_initial - m_current = 11 :=
by {
  sorry
}

end marbles_lost_l207_207615


namespace Diana_friends_count_l207_207018

theorem Diana_friends_count (totalErasers : ℕ) (erasersPerFriend : ℕ) 
  (h1: totalErasers = 3840) (h2: erasersPerFriend = 80) : 
  totalErasers / erasersPerFriend = 48 := 
by 
  sorry

end Diana_friends_count_l207_207018


namespace geometric_series_ratio_half_l207_207961

theorem geometric_series_ratio_half (a r S : ℝ) (hS : S = a / (1 - r)) 
  (h_ratio : (ar^4) / (1 - r) = S / 64) : r = 1 / 2 :=
by
  sorry

end geometric_series_ratio_half_l207_207961


namespace problem_part1_problem_part2_l207_207205

open ProbabilityTheory

noncomputable def total_questions : ℕ := 8
noncomputable def listening_questions : ℕ := 3
noncomputable def written_response_questions : ℕ := 5

-- The probability that Student A draws a listening question and Student B draws a written response question
def prob_A_listening_B_written : ℚ :=
  (listening_questions * written_response_questions) / (total_questions * (total_questions - 1))

-- The probability that at least one of the students draws a listening question
def prob_at_least_one_listening : ℚ :=
  1 - (written_response_questions * (written_response_questions - 1)) / (total_questions * (total_questions - 1))

theorem problem_part1 : prob_A_listening_B_written = 15 / 56 := sorry

theorem problem_part2 : prob_at_least_one_listening = 9 / 14 := sorry

end problem_part1_problem_part2_l207_207205


namespace floor_neg_sqrt_eval_l207_207023

theorem floor_neg_sqrt_eval :
  ⌊-(Real.sqrt (64 / 9))⌋ = -3 :=
by
  sorry

end floor_neg_sqrt_eval_l207_207023
