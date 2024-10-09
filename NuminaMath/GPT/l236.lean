import Mathlib

namespace smallest_five_digit_int_equiv_mod_l236_23622

theorem smallest_five_digit_int_equiv_mod (n : ℕ) (h1 : 10000 ≤ n) (h2 : n % 9 = 4) : n = 10003 := 
sorry

end smallest_five_digit_int_equiv_mod_l236_23622


namespace steps_to_school_l236_23607

-- Define the conditions as assumptions
def distance : Float := 900
def step_length : Float := 0.45

-- Define the statement to be proven
theorem steps_to_school (x : Float) : step_length * x = distance → x = 2000 := by
  intro h
  sorry

end steps_to_school_l236_23607


namespace expression_value_l236_23690

def a : ℕ := 45
def b : ℕ := 18
def c : ℕ := 10

theorem expression_value :
  (a + b)^2 - (a^2 + b^2 + c) = 1610 := by
  sorry

end expression_value_l236_23690


namespace trigonometric_identity_l236_23641

open Real

theorem trigonometric_identity (α : ℝ) (h : sin (α - (π / 12)) = 1 / 3) :
  cos (α + (17 * π / 12)) = 1 / 3 :=
sorry

end trigonometric_identity_l236_23641


namespace max_even_integers_for_odd_product_l236_23624

theorem max_even_integers_for_odd_product (a b c d e f g : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) (h6 : 0 < f) (h7 : 0 < g) 
  (h_prod_odd : a * b * c * d * e * f * g % 2 = 1) : a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ e % 2 = 1 ∧ f % 2 = 1 ∧ g % 2 = 1 :=
sorry

end max_even_integers_for_odd_product_l236_23624


namespace cost_of_each_toy_car_l236_23623

theorem cost_of_each_toy_car (S M C A B : ℕ) (hS : S = 53) (hM : M = 7) (hA : A = 10) (hB : B = 14) 
(hTotalSpent : S - M = C + A + B) (hTotalCars : 2 * C / 2 = 11) : 
C / 2 = 11 :=
by
  rw [hS, hM, hA, hB] at hTotalSpent
  sorry

end cost_of_each_toy_car_l236_23623


namespace total_rope_length_l236_23644

theorem total_rope_length 
  (longer_side : ℕ) (shorter_side : ℕ) 
  (h1 : longer_side = 28) (h2 : shorter_side = 22) : 
  2 * longer_side + 2 * shorter_side = 100 := by
  sorry

end total_rope_length_l236_23644


namespace convex_polygons_count_l236_23634

def binomial (n k : ℕ) : ℕ := if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def count_convex_polygons_with_two_acute_angles (m n : ℕ) : ℕ :=
  if 4 < m ∧ m < n then
    (2 * n + 1) * (binomial (n + 1) (m - 1) + binomial n (m - 1))
  else 0

theorem convex_polygons_count (m n : ℕ) (h : 4 < m ∧ m < n) :
  count_convex_polygons_with_two_acute_angles m n = 
  (2 * n + 1) * (binomial (n + 1) (m - 1) + binomial n (m - 1)) :=
by sorry

end convex_polygons_count_l236_23634


namespace calculate_expression_l236_23619

theorem calculate_expression : 
  let a := (-1 : Int) ^ 2023
  let b := (-8 : Int) / (-4)
  let c := abs (-5)
  a + b - c = -4 := 
by
  sorry

end calculate_expression_l236_23619


namespace tripod_height_l236_23653

-- Define the conditions of the problem
structure Tripod where
  leg_length : ℝ
  angle_equal : Bool
  top_height : ℝ
  broken_length : ℝ

def m : ℕ := 27
def n : ℕ := 10

noncomputable def h : ℝ := m / Real.sqrt n

theorem tripod_height :
  ∀ (t : Tripod),
  t.leg_length = 6 →
  t.angle_equal = true →
  t.top_height = 3 →
  t.broken_length = 2 →
  (h = m / Real.sqrt n) →
  (⌊m + Real.sqrt n⌋ = 30) :=
by
  intros
  sorry

end tripod_height_l236_23653


namespace find_x_intercept_l236_23657

-- Define the equation of the line
def line_eq (x y : ℝ) : Prop := 4 * x + 7 * y = 28

-- Define the x-intercept point when y = 0
def x_intercept (x : ℝ) : Prop := line_eq x 0

-- Prove that for the x-intercept, when y = 0, x = 7
theorem find_x_intercept : x_intercept 7 :=
by
  -- proof would go here
  sorry

end find_x_intercept_l236_23657


namespace fraction_zero_if_abs_x_eq_one_l236_23651

theorem fraction_zero_if_abs_x_eq_one (x : ℝ) : 
  (|x| - 1) = 0 → (x^2 - 2 * x + 1 ≠ 0) → x = -1 := 
by 
  sorry

end fraction_zero_if_abs_x_eq_one_l236_23651


namespace second_solution_concentration_l236_23654

def volume1 : ℝ := 5
def concentration1 : ℝ := 0.04
def volume2 : ℝ := 2.5
def concentration_final : ℝ := 0.06
def total_silver1 : ℝ := volume1 * concentration1
def total_volume : ℝ := volume1 + volume2
def total_silver_final : ℝ := total_volume * concentration_final

theorem second_solution_concentration :
  ∃ (C2 : ℝ), total_silver1 + volume2 * C2 = total_silver_final ∧ C2 = 0.1 := 
by 
  sorry

end second_solution_concentration_l236_23654


namespace ratio_bc_cd_l236_23684

-- Definitions based on given conditions.
variable (a b c d e : ℝ)
variable (h_ab : b - a = 5)
variable (h_ac : c - a = 11)
variable (h_de : e - d = 8)
variable (h_ae : e - a = 22)

-- The theorem to prove bc : cd = 2 : 1.
theorem ratio_bc_cd (h_ab : b - a = 5) (h_ac : c - a = 11) (h_de : e - d = 8) (h_ae : e - a = 22) :
  (c - b) / (d - c) = 2 :=
by
  sorry

end ratio_bc_cd_l236_23684


namespace maximize_distance_l236_23620

noncomputable def maxTotalDistance (x : ℕ) (y : ℕ) (cityMPG highwayMPG : ℝ) (totalGallons : ℝ) : ℝ :=
  let cityDistance := cityMPG * ((x / 100.0) * totalGallons)
  let highwayDistance := highwayMPG * ((y / 100.0) * totalGallons)
  cityDistance + highwayDistance

theorem maximize_distance (x y : ℕ) (hx : x + y = 100) :
  maxTotalDistance x y 7.6 12.2 24.0 = 7.6 * (x / 100.0 * 24.0) + 12.2 * ((100.0 - x) / 100.0 * 24.0) :=
by
  sorry

end maximize_distance_l236_23620


namespace arithmetic_sequence_sum_l236_23655

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (hS3 : S 3 = 12) (hS6 : S 6 = 42) 
  (h_arith_seq : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) :
  a 10 + a 11 + a 12 = 66 :=
sorry

end arithmetic_sequence_sum_l236_23655


namespace binomial_coefficient_middle_term_l236_23647

theorem binomial_coefficient_middle_term :
  let n := 11
  let sum_odd := 1024
  sum_odd = 2^(n-1) →
  let binom_coef := Nat.choose n (n / 2 - 1)
  binom_coef = 462 :=
by
  intro n
  let n := 11
  intro sum_odd
  let sum_odd := 1024
  intro h
  let binom_coef := Nat.choose n (n / 2 - 1)
  have : binom_coef = 462 := sorry
  exact this

end binomial_coefficient_middle_term_l236_23647


namespace Robe_savings_l236_23687

-- Define the conditions and question in Lean 4
theorem Robe_savings 
  (repair_fee : ℕ)
  (corner_light_cost : ℕ)
  (brake_disk_cost : ℕ)
  (total_remaining_savings : ℕ)
  (total_savings_before : ℕ)
  (h1 : repair_fee = 10)
  (h2 : corner_light_cost = 2 * repair_fee)
  (h3 : brake_disk_cost = 3 * corner_light_cost)
  (h4 : total_remaining_savings = 480)
  (h5 : total_savings_before = total_remaining_savings + (repair_fee + corner_light_cost + 2 * brake_disk_cost)) :
  total_savings_before = 630 :=
by
  -- Proof steps to be filled
  sorry

end Robe_savings_l236_23687


namespace sum_of_below_avg_l236_23672

-- Define class averages
def a1 := 75
def a2 := 85
def a3 := 90
def a4 := 65

-- Define the overall average
def avg : ℚ := (a1 + a2 + a3 + a4) / 4

-- Define a predicate indicating if a class average is below the overall average
def below_avg (a : ℚ) : Prop := a < avg

-- The theorem to prove the required sum of averages below the overall average
theorem sum_of_below_avg : a1 < avg ∧ a4 < avg → a1 + a4 = 140 :=
by
  sorry

end sum_of_below_avg_l236_23672


namespace arithmetic_sequence_difference_l236_23649

def arithmetic_sequence (a d n : ℕ) : ℤ :=
  a + (n - 1) * d

theorem arithmetic_sequence_difference :
  let a := 3
  let d := 7
  let a₁₀₀₀ := arithmetic_sequence a d 1000
  let a₁₀₀₃ := arithmetic_sequence a d 1003
  abs (a₁₀₀₃ - a₁₀₀₀) = 21 :=
by
  sorry

end arithmetic_sequence_difference_l236_23649


namespace necessary_not_sufficient_condition_l236_23637

theorem necessary_not_sufficient_condition (a : ℝ) :
  (a < 2) ∧ (a^2 - 4 < 0) ↔ (a < 2) ∧ (a > -2) :=
by
  sorry

end necessary_not_sufficient_condition_l236_23637


namespace david_age_l236_23676

theorem david_age (A B C D : ℕ)
  (h1 : A = B - 5)
  (h2 : B = C + 2)
  (h3 : D = C + 4)
  (h4 : A = 12) : D = 19 :=
sorry

end david_age_l236_23676


namespace circle_intersects_y_axis_with_constraints_l236_23679

theorem circle_intersects_y_axis_with_constraints {m n : ℝ} 
    (H1 : n = m ^ 2 + 2 * m + 2) 
    (H2 : abs m <= 2) : 
    1 ≤ n ∧ n < 10 :=
sorry

end circle_intersects_y_axis_with_constraints_l236_23679


namespace westbound_cyclist_speed_increase_l236_23665

def eastbound_speed : ℕ := 18
def travel_time : ℕ := 6
def total_distance : ℕ := 246

theorem westbound_cyclist_speed_increase (x : ℕ) :
  eastbound_speed * travel_time + (eastbound_speed + x) * travel_time = total_distance →
  x = 5 :=
by
  sorry

end westbound_cyclist_speed_increase_l236_23665


namespace part_I_solution_set_part_II_range_of_a_l236_23626

-- Given function definition
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a*x + 6

-- (I) Prove the solution set of f(x) < 0 when a = 5
theorem part_I_solution_set : 
  (∀ x : ℝ, f x 5 < 0 ↔ (-3 < x ∧ x < -2)) := by
  sorry

-- (II) Prove the range of a such that f(x) > 0 for all x ∈ ℝ 
theorem part_II_range_of_a :
  (∀ a : ℝ, (∀ x : ℝ, f x a > 0) ↔ (-2*Real.sqrt 6 < a ∧ a < 2*Real.sqrt 6)) := by
  sorry

end part_I_solution_set_part_II_range_of_a_l236_23626


namespace p_add_inv_p_gt_two_l236_23638

theorem p_add_inv_p_gt_two {p : ℝ} (hp_pos : p > 0) (hp_neq_one : p ≠ 1) : p + 1 / p > 2 :=
by
  sorry

end p_add_inv_p_gt_two_l236_23638


namespace restore_original_problem_l236_23698

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1 / X) * (Y + 1 / 2) = 43 :=
by
  rw [hX, hY]
  -- Continue the proof steps here
  sorry

end restore_original_problem_l236_23698


namespace total_price_of_hats_l236_23625

variables (total_hats : ℕ) (blue_hat_cost : ℕ) (green_hat_cost : ℕ) (green_hats : ℕ) (total_price : ℕ)

def total_number_of_hats := 85
def cost_per_blue_hat := 6
def cost_per_green_hat := 7
def number_of_green_hats := 30

theorem total_price_of_hats :
  (number_of_green_hats * cost_per_green_hat) + ((total_number_of_hats - number_of_green_hats) * cost_per_blue_hat) = 540 :=
sorry

end total_price_of_hats_l236_23625


namespace tony_will_have_4_dollars_in_change_l236_23646

def tony_change : ℕ :=
  let bucket_capacity : ℕ := 2
  let sandbox_depth : ℕ := 2
  let sandbox_width : ℕ := 4
  let sandbox_length : ℕ := 5
  let sand_weight_per_cubic_foot : ℕ := 3
  let water_consumed_per_drink : ℕ := 3
  let trips_per_drink : ℕ := 4
  let bottle_capacity : ℕ := 15
  let bottle_cost : ℕ := 2
  let initial_money : ℕ := 10

  let sandbox_volume := sandbox_depth * sandbox_width * sandbox_length
  let total_sand_weight := sandbox_volume * sand_weight_per_cubic_foot
  let number_of_trips := total_sand_weight / bucket_capacity
  let number_of_drinks := number_of_trips / trips_per_drink
  let total_water_consumed := number_of_drinks * water_consumed_per_drink
  let number_of_bottles := total_water_consumed / bottle_capacity
  let total_water_cost := number_of_bottles * bottle_cost
  let change := initial_money - total_water_cost

  change

theorem tony_will_have_4_dollars_in_change : tony_change = 4 := by
  sorry

end tony_will_have_4_dollars_in_change_l236_23646


namespace original_ratio_l236_23685

namespace OilBill

-- Definitions based on conditions
def JanuaryBill : ℝ := 179.99999999999991

def FebruaryBillWith30More (F : ℝ) : Prop := 
  3 * (F + 30) = 900

-- Statement of the problem proving the original ratio
theorem original_ratio (F : ℝ) (hF : FebruaryBillWith30More F) : 
  F / JanuaryBill = 3 / 2 :=
by
  -- This will contain the proof steps
  sorry

end OilBill

end original_ratio_l236_23685


namespace no_such_integers_l236_23645

theorem no_such_integers (a b : ℤ) : 
  ¬ (∃ a b : ℤ, ∃ k₁ k₂ : ℤ, a^5 * b + 3 = k₁^3 ∧ a * b^5 + 3 = k₂^3) :=
by 
  sorry

end no_such_integers_l236_23645


namespace find_number_of_girls_l236_23615

theorem find_number_of_girls (B G : ℕ) 
  (h1 : B + G = 604) 
  (h2 : 12 * B + 11 * G = 47 * 604 / 4) : 
  G = 151 :=
by
  sorry

end find_number_of_girls_l236_23615


namespace even_n_ineq_l236_23636

theorem even_n_ineq (n : ℕ) (h : ∀ x : ℝ, 3 * x^n + n * (x + 2) - 3 ≥ n * x^2) : Even n :=
  sorry

end even_n_ineq_l236_23636


namespace problem1_problem2_l236_23659

theorem problem1 : 2 * Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 27 = 3 * Real.sqrt 3 := by
  sorry

theorem problem2 : (Real.sqrt 18 - Real.sqrt 3) * Real.sqrt 12 = 6 * Real.sqrt 6 - 6 := by
  sorry

end problem1_problem2_l236_23659


namespace find_real_solutions_l236_23640

variable (x : ℝ)

theorem find_real_solutions :
  (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 8) ↔ 
  (x = -2 * Real.sqrt 14 ∨ x = 2 * Real.sqrt 14) := 
sorry

end find_real_solutions_l236_23640


namespace calculate_selling_price_l236_23663

theorem calculate_selling_price (cost_price : ℝ) (profit_percentage : ℝ) (selling_price : ℝ) 
  (h1 : cost_price = 83.33) 
  (h2 : profit_percentage = 20) : 
  selling_price = 100 := by
  sorry

end calculate_selling_price_l236_23663


namespace unique_pair_not_opposite_l236_23689

def QuantumPair (a b : String): Prop := ∃ oppositeMeanings : Bool, a ≠ b ∧ oppositeMeanings

theorem unique_pair_not_opposite :
  ∃ (a b : String), 
    (a = "increase of 2 years" ∧ b = "decrease of 2 liters") ∧ 
    (¬ QuantumPair a b) :=
by 
  sorry

end unique_pair_not_opposite_l236_23689


namespace smallest_digits_to_append_l236_23674

theorem smallest_digits_to_append (n : ℕ) : ∃ d, d ≤ 4 ∧ ∃ k, (2014 * 10^d + k) % 2520 = 0 :=
  sorry

end smallest_digits_to_append_l236_23674


namespace quadratic_equation_conditions_l236_23662

theorem quadratic_equation_conditions :
  ∃ (a b c : ℝ), a = 3 ∧ c = 1 ∧ (a * x^2 + b * x + c = 0 ↔ 3 * x^2 + 1 = 0) :=
by
  use 3, 0, 1
  sorry

end quadratic_equation_conditions_l236_23662


namespace bullseye_points_l236_23616

theorem bullseye_points (B : ℝ) (h : B + B / 2 = 75) : B = 50 :=
by
  sorry

end bullseye_points_l236_23616


namespace lyssa_fewer_correct_l236_23680

-- Define the total number of items in the exam
def total_items : ℕ := 75

-- Define the number of mistakes made by Lyssa
def lyssa_mistakes : ℕ := total_items * 20 / 100  -- 20% of 75

-- Define the number of correct answers by Lyssa
def lyssa_correct : ℕ := total_items - lyssa_mistakes

-- Define the number of mistakes made by Precious
def precious_mistakes : ℕ := 12

-- Define the number of correct answers by Precious
def precious_correct : ℕ := total_items - precious_mistakes

-- Statement to prove Lyssa got 3 fewer correct answers than Precious
theorem lyssa_fewer_correct : (precious_correct - lyssa_correct) = 3 := by
  sorry

end lyssa_fewer_correct_l236_23680


namespace problem_solution_l236_23695

variable (α : ℝ)

/-- If $\sin\alpha = 2\cos\alpha$, then the function $f(x) = 2^x - \tan\alpha$ satisfies $f(0) = -1$. -/
theorem problem_solution (h : Real.sin α = 2 * Real.cos α) : (2^0 - Real.tan α) = -1 := by
  sorry

end problem_solution_l236_23695


namespace number_of_new_trailer_homes_l236_23643

-- Definitions coming from the conditions
def initial_trailers : ℕ := 30
def initial_avg_age : ℕ := 15
def years_passed : ℕ := 5
def current_avg_age : ℕ := initial_avg_age + years_passed

-- Let 'n' be the number of new trailer homes added five years ago
variable (n : ℕ)

def new_trailer_age : ℕ := years_passed
def total_trailers : ℕ := initial_trailers + n
def total_ages : ℕ := (initial_trailers * current_avg_age) + (n * new_trailer_age)
def combined_avg_age := total_ages / total_trailers

theorem number_of_new_trailer_homes (h : combined_avg_age = 12) : n = 34 := 
sorry

end number_of_new_trailer_homes_l236_23643


namespace surface_area_bound_l236_23681

theorem surface_area_bound
  (a b c d : ℝ)
  (h1: 0 ≤ a) (h2: 0 ≤ b) (h3: 0 ≤ c) (h4: 0 ≤ d) 
  (h_quad: a + b + c > d) : 
  2 * (a * b + b * c + c * a) ≤ (a + b + c) ^ 2 - (d ^ 2) / 3 :=
sorry

end surface_area_bound_l236_23681


namespace total_points_scored_l236_23630

theorem total_points_scored (n m T : ℕ) 
  (h1 : T = 2 * n + 5 * m) 
  (h2 : n = m + 3 ∨ m = n + 3)
  : T = 20 :=
sorry

end total_points_scored_l236_23630


namespace correct_operation_l236_23628

theorem correct_operation (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end correct_operation_l236_23628


namespace percentage_parents_agree_l236_23601

def total_parents : ℕ := 800
def disagree_parents : ℕ := 640

theorem percentage_parents_agree : 
  ((total_parents - disagree_parents) / total_parents : ℚ) * 100 = 20 := 
by 
  sorry

end percentage_parents_agree_l236_23601


namespace pentomino_reflectional_count_l236_23668

def is_reflectional (p : Pentomino) : Prop := sorry -- Define reflectional symmetry property
def is_rotational (p : Pentomino) : Prop := sorry -- Define rotational symmetry property

theorem pentomino_reflectional_count :
  ∀ (P : Finset Pentomino),
  P.card = 15 →
  (∃ (R : Finset Pentomino), R.card = 2 ∧ (∀ p ∈ R, is_rotational p ∧ ¬ is_reflectional p)) →
  (∃ (S : Finset Pentomino), S.card = 7 ∧ (∀ p ∈ S, is_reflectional p)) :=
by
  sorry -- Proof not required as per instructions

end pentomino_reflectional_count_l236_23668


namespace quadratic_linear_term_l236_23611

theorem quadratic_linear_term (m : ℝ) 
  (h : 2 * m = 6) : -4 * (x : ℝ) + m * x = -x := by 
  sorry

end quadratic_linear_term_l236_23611


namespace sum_of_first_9_terms_of_arithmetic_sequence_l236_23656

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

theorem sum_of_first_9_terms_of_arithmetic_sequence 
  (h1 : is_arithmetic_sequence a) 
  (h2 : a 2 + a 8 = 18) 
  (h3 : sum_of_first_n_terms a S) :
  S 9 = 81 :=
sorry

end sum_of_first_9_terms_of_arithmetic_sequence_l236_23656


namespace distinct_complex_numbers_no_solution_l236_23664

theorem distinct_complex_numbers_no_solution :
  ¬∃ (a b c d : ℂ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d ∧
  (a^3 - b * c * d = b^3 - c * d * a) ∧ 
  (b^3 - c * d * a = c^3 - d * a * b) ∧ 
  (c^3 - d * a * b = d^3 - a * b * c) := 
by {
  sorry
}

end distinct_complex_numbers_no_solution_l236_23664


namespace sum_interior_angles_of_regular_polygon_l236_23612

theorem sum_interior_angles_of_regular_polygon (exterior_angle : ℝ) (sum_exterior_angles : ℝ) (n : ℝ)
  (h1 : exterior_angle = 45)
  (h2 : sum_exterior_angles = 360)
  (h3 : n = sum_exterior_angles / exterior_angle) :
  180 * (n - 2) = 1080 :=
by
  sorry

end sum_interior_angles_of_regular_polygon_l236_23612


namespace jogged_distance_is_13_point_5_l236_23677

noncomputable def jogger_distance (x t d : ℝ) : Prop :=
  d = x * t ∧
  d = (x + 3/4) * (3 * t / 4) ∧
  d = (x - 3/4) * (t + 3)

theorem jogged_distance_is_13_point_5:
  ∃ (x t d : ℝ), jogger_distance x t d ∧ d = 13.5 :=
by
  sorry

end jogged_distance_is_13_point_5_l236_23677


namespace integer_diff_of_two_squares_l236_23697

theorem integer_diff_of_two_squares (m : ℤ) : 
  (∃ x y : ℤ, m = x^2 - y^2) ↔ (∃ k : ℤ, m ≠ 4 * k + 2) := by
  sorry

end integer_diff_of_two_squares_l236_23697


namespace smaller_angle_in_parallelogram_l236_23650

theorem smaller_angle_in_parallelogram 
  (opposite_angles : ∀ A B C D : ℝ, A = C ∧ B = D)
  (adjacent_angles_supplementary : ∀ A B : ℝ, A + B = π)
  (angle_diff : ∀ A B : ℝ, B = A + π/9) :
  ∃ θ : ℝ, θ = 4 * π / 9 :=
by
  sorry

end smaller_angle_in_parallelogram_l236_23650


namespace flower_position_after_50_beats_l236_23694

-- Define the number of students
def num_students : Nat := 7

-- Define the initial position of the flower
def initial_position : Nat := 1

-- Define the number of drum beats
def drum_beats : Nat := 50

-- Theorem stating that after 50 drum beats, the flower will be with the 2nd student
theorem flower_position_after_50_beats : 
  (initial_position + (drum_beats % num_students)) % num_students = 2 := by
  -- Start the proof (this part usually would contain the actual proof logic)
  sorry

end flower_position_after_50_beats_l236_23694


namespace mixed_number_calculation_l236_23632

theorem mixed_number_calculation :
  47 * (4 + 3/7 - (5 + 1/3)) / (3 + 1/2 + (2 + 1/5)) = -7 - 119/171 := by
  sorry

end mixed_number_calculation_l236_23632


namespace odd_square_minus_one_divisible_by_eight_l236_23666

theorem odd_square_minus_one_divisible_by_eight (n : ℤ) : ∃ k : ℤ, ((2 * n + 1) ^ 2 - 1) = 8 * k := 
by
  sorry

end odd_square_minus_one_divisible_by_eight_l236_23666


namespace find_cost_price_l236_23608

theorem find_cost_price (SP : ℝ) (loss_percent : ℝ) (CP : ℝ) (h1 : SP = 1260) (h2 : loss_percent = 16) : CP = 1500 :=
by
  sorry

end find_cost_price_l236_23608


namespace center_of_circle_l236_23635

theorem center_of_circle (x y : ℝ) (h : x^2 + y^2 = 4 * x - 6 * y + 9) : x + y = -1 := 
by 
  sorry

end center_of_circle_l236_23635


namespace find_Q_x_l236_23639

noncomputable def Q : ℝ → ℝ := sorry

variables (Q0 Q1 Q2 : ℝ)

axiom Q_def : ∀ x, Q x = Q0 + Q1 * x + Q2 * x^2
axiom Q_minus_2 : Q (-2) = -3

theorem find_Q_x : ∀ x, Q x = (3 / 5) * (1 + x - x^2) :=
by 
  -- Proof to be completed
  sorry

end find_Q_x_l236_23639


namespace find_y_l236_23683

theorem find_y (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) 
  (h : (2 * a) ^ (2 * b ^ 2) = (a ^ b + y ^ b) ^ 2) : y = 4 * a ^ 2 - a := 
sorry

end find_y_l236_23683


namespace diameter_of_circle_A_l236_23692

theorem diameter_of_circle_A
  (diameter_B : ℝ)
  (r : ℝ)
  (h1 : diameter_B = 16)
  (h2 : r^2 = (r / 8)^2 * 4):
  2 * (r / 2) = 8 :=
by
  sorry

end diameter_of_circle_A_l236_23692


namespace sin_C_in_right_triangle_l236_23604

-- Triangle ABC with angle B = 90 degrees and tan A = 3/4
theorem sin_C_in_right_triangle (A C : ℝ) (h1 : A + C = π / 2) (h2 : Real.tan A = 3 / 4) : Real.sin C = 4 / 5 := by
  sorry

end sin_C_in_right_triangle_l236_23604


namespace certain_number_is_84_l236_23648

/-
The least number by which 72 must be multiplied in order to produce a multiple of a certain number is 14.
What is that certain number?
-/

theorem certain_number_is_84 (x : ℕ) (h: 72 * 14 % x = 0 ∧ ∀ y : ℕ, 1 ≤ y → y < 14 → 72 * y % x ≠ 0) : x = 84 :=
sorry

end certain_number_is_84_l236_23648


namespace largest_multiple_of_11_neg_greater_minus_210_l236_23613

theorem largest_multiple_of_11_neg_greater_minus_210 :
  ∃ (x : ℤ), x % 11 = 0 ∧ -x < -210 ∧ ∀ y, y % 11 = 0 ∧ -y < -210 → y ≤ x :=
sorry

end largest_multiple_of_11_neg_greater_minus_210_l236_23613


namespace exists_n_for_perfect_square_l236_23669

theorem exists_n_for_perfect_square (k : ℕ) (hk_pos : k > 0) :
  ∃ n : ℕ, n > 0 ∧ ∃ a : ℕ, a^2 = n * 2^k - 7 :=
by
  sorry

end exists_n_for_perfect_square_l236_23669


namespace sequence_increasing_range_l236_23667

theorem sequence_increasing_range (a : ℝ) (h : ∀ n : ℕ, (n - a) ^ 2 < (n + 1 - a) ^ 2) :
  a < 3 / 2 :=
by
  sorry

end sequence_increasing_range_l236_23667


namespace paper_thickness_after_2_folds_l236_23660

theorem paper_thickness_after_2_folds:
  ∀ (initial_thickness : ℝ) (folds : ℕ),
  initial_thickness = 0.1 →
  folds = 2 →
  (initial_thickness * 2^folds = 0.4) :=
by
  intros initial_thickness folds h_initial h_folds
  sorry

end paper_thickness_after_2_folds_l236_23660


namespace equivalent_sets_l236_23621

-- Definitions of the condition and expected result
def condition_set : Set ℕ := { x | x - 3 < 2 }
def expected_set : Set ℕ := {0, 1, 2, 3, 4}

-- Theorem statement
theorem equivalent_sets : condition_set = expected_set := 
by
  sorry

end equivalent_sets_l236_23621


namespace other_asymptote_of_hyperbola_l236_23629

theorem other_asymptote_of_hyperbola (a b : ℝ) : 
  (∀ x : ℝ, y = 2 * x + 3) → 
  (∃ y : ℝ, x = -4) → 
  (∀ x : ℝ, y = - (1 / 2) * x - 7) := 
by {
  -- The proof will go here
  sorry
}

end other_asymptote_of_hyperbola_l236_23629


namespace candy_game_win_l236_23603

def winning_player (A B : ℕ) : String :=
  if (A % B = 0 ∨ B % A = 0) then "Player with forcing checks" else "No inevitable winner"

theorem candy_game_win :
  winning_player 1000 2357 = "Player with forcing checks" :=
by
  sorry

end candy_game_win_l236_23603


namespace trigonometric_identity_l236_23600

theorem trigonometric_identity 
  (x : ℝ)
  (h : Real.sin (2 * x + Real.pi / 5) = Real.sqrt 3 / 3) : 
  Real.sin (4 * Real.pi / 5 - 2 * x) + Real.sin (3 * Real.pi / 10 - 2 * x)^2 = (2 + Real.sqrt 3) / 3 :=
by
  sorry

end trigonometric_identity_l236_23600


namespace four_ab_eq_four_l236_23602

theorem four_ab_eq_four {a b : ℝ} (h : a * b = 1) : 4 * a * b = 4 :=
by
  sorry

end four_ab_eq_four_l236_23602


namespace probability_not_above_y_axis_l236_23606

-- Define the vertices
structure Point :=
  (x : ℝ)
  (y : ℝ)

def P := Point.mk (-1) 5
def Q := Point.mk 2 (-3)
def R := Point.mk (-5) (-3)
def S := Point.mk (-8) 5

-- Define predicate for being above the y-axis
def is_above_y_axis (p : Point) : Prop := p.y > 0

-- Define the parallelogram region (this is theoretical as defining a whole region 
-- can be complex, but we state the region as a property)
noncomputable def in_region_of_parallelogram (p : Point) : Prop := sorry

-- Define the probability calculation statement
theorem probability_not_above_y_axis (p : Point) :
  in_region_of_parallelogram p → ¬is_above_y_axis p := sorry

end probability_not_above_y_axis_l236_23606


namespace Alex_failing_implies_not_all_hw_on_time_l236_23614

-- Definitions based on the conditions provided
variable (Alex_submits_all_hw_on_time : Prop)
variable (Alex_passes_course : Prop)

-- Given condition: Submitting all homework assignments implies passing the course
axiom Mrs_Thompson_statement : Alex_submits_all_hw_on_time → Alex_passes_course

-- The problem: Prove that if Alex failed the course, then he did not submit all homework assignments on time
theorem Alex_failing_implies_not_all_hw_on_time (h : ¬Alex_passes_course) : ¬Alex_submits_all_hw_on_time :=
  by
  sorry

end Alex_failing_implies_not_all_hw_on_time_l236_23614


namespace total_players_on_ground_l236_23671

theorem total_players_on_ground 
  (cricket_players : ℕ) (hockey_players : ℕ) (football_players : ℕ) (softball_players : ℕ)
  (hcricket : cricket_players = 16) (hhokey : hockey_players = 12) 
  (hfootball : football_players = 18) (hsoftball : softball_players = 13) :
  cricket_players + hockey_players + football_players + softball_players = 59 :=
by
  sorry

end total_players_on_ground_l236_23671


namespace a_eq_1_sufficient_not_necessary_l236_23642

theorem a_eq_1_sufficient_not_necessary (a : ℝ) :
  (∀ x : ℝ, x ≤ 1 → |x - 1| ≤ |x - a|) ∧ ¬(∀ x : ℝ, x ≤ 1 → |x - 1| = |x - a|) :=
by
  sorry

end a_eq_1_sufficient_not_necessary_l236_23642


namespace range_equality_of_f_and_f_f_l236_23618

noncomputable def f (x a : ℝ) := x * Real.log x - x + 2 * a

theorem range_equality_of_f_and_f_f (a : ℝ) :
  (∀ x : ℝ, 0 < x → 1 < f x a) ∧ (∀ x : ℝ, 0 < x → f x a ≤ 1) →
  (∃ I : Set ℝ, (Set.range (λ x => f x a) = I) ∧ (Set.range (λ x => f (f x a) a) = I)) → 
  (1/2 < a ∧ a ≤ 1) :=
by 
  sorry

end range_equality_of_f_and_f_f_l236_23618


namespace sequence_expression_l236_23696

theorem sequence_expression (a : ℕ → ℝ) (h_base : a 1 = 2)
  (h_rec : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * (n + 1) * a n / (a n + n)) :
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = (n + 1) * 2^(n + 1) / (2^(n + 1) - 1) :=
by
  sorry

end sequence_expression_l236_23696


namespace length_of_rest_of_body_l236_23688

theorem length_of_rest_of_body (h : ℝ) (legs : ℝ) (head : ℝ) (rest_of_body : ℝ) :
  h = 60 → legs = (1 / 3) * h → head = (1 / 4) * h → rest_of_body = h - (legs + head) → rest_of_body = 25 := by
  sorry

end length_of_rest_of_body_l236_23688


namespace number_of_days_l236_23627

noncomputable def days_to_lay_bricks (b c f : ℕ) : ℕ :=
(b * b) / f

theorem number_of_days (b c f : ℕ) (h_nonzero_f : f ≠ 0) (h_bc_pos : b > 0 ∧ c > 0) :
  days_to_lay_bricks b c f = (b * b) / f :=
by 
  sorry

end number_of_days_l236_23627


namespace boys_trees_l236_23652

theorem boys_trees (avg_per_person trees_per_girl trees_per_boy : ℕ) :
  avg_per_person = 6 →
  trees_per_girl = 15 →
  (1 / trees_per_boy + 1 / trees_per_girl = 1 / avg_per_person) →
  trees_per_boy = 10 :=
by
  intros h_avg h_girl h_eq
  -- We will provide the proof here eventually
  sorry

end boys_trees_l236_23652


namespace min_value_AP_AQ_l236_23661

noncomputable def min_distance (A P Q : ℝ × ℝ) : ℝ := dist A P + dist A Q

theorem min_value_AP_AQ :
  ∀ (A P Q : ℝ × ℝ),
    (∀ (x : ℝ), A = (x, 0)) →
    ((P.1 - 1) ^ 2 + (P.2 - 3) ^ 2 = 1) →
    ((Q.1 - 7) ^ 2 + (Q.2 - 5) ^ 2 = 4) →
    min_distance A P Q = 7 :=
by
  intros A P Q hA hP hQ
  -- Proof is to be provided here
  sorry

end min_value_AP_AQ_l236_23661


namespace ball_distributions_l236_23610

theorem ball_distributions (p q : ℚ) (h1 : p = (Nat.choose 5 1 * Nat.choose 4 1 * Nat.choose 20 2 * Nat.choose 18 6 * Nat.choose 12 4 * Nat.choose 8 4 * Nat.choose 4 4) / Nat.choose 20 20)
                            (h2 : q = (Nat.choose 20 4 * Nat.choose 16 4 * Nat.choose 12 4 * Nat.choose 8 4 * Nat.choose 4 4) / Nat.choose 20 20) :
  p / q = 10 :=
by
  sorry

end ball_distributions_l236_23610


namespace minimum_value_of_function_l236_23617

theorem minimum_value_of_function : ∀ x : ℝ, x ≥ 0 → (4 * x^2 + 12 * x + 25) / (6 * (1 + x)) ≥ 8 / 3 := by
  sorry

end minimum_value_of_function_l236_23617


namespace prime_pair_perfect_square_l236_23693

theorem prime_pair_perfect_square (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h : ∃ a : ℕ, p^2 + p * q + q^2 = a^2) : (p = 3 ∧ q = 5) ∨ (p = 5 ∧ q = 3) := 
sorry

end prime_pair_perfect_square_l236_23693


namespace sum_of_rationals_eq_l236_23633

theorem sum_of_rationals_eq (a1 a2 a3 a4 : ℚ)
  (h : {x : ℚ | ∃ i j, 1 ≤ i ∧ i < j ∧ j ≤ 4 ∧ x = a1 * a2 ∧ x = a1 * a3 ∧ x = a1 * a4 ∧ x = a2 * a3 ∧ x = a2 * a4 ∧ x = a3 * a4} = {-24, -2, -3/2, -1/8, 1, 3}) :
  a1 + a2 + a3 + a4 = 9/4 ∨ a1 + a2 + a3 + a4 = -9/4 :=
sorry

end sum_of_rationals_eq_l236_23633


namespace pie_difference_l236_23631

theorem pie_difference:
  ∀ (a b c d : ℚ), a = 6 / 7 → b = 3 / 4 → (a - b) = c → c = 3 / 28 :=
by
  sorry

end pie_difference_l236_23631


namespace units_digit_of_five_consecutive_product_is_zero_l236_23605

theorem units_digit_of_five_consecutive_product_is_zero (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 10 = 0 :=
by
  sorry

end units_digit_of_five_consecutive_product_is_zero_l236_23605


namespace triangle_height_l236_23678

theorem triangle_height (base height area : ℝ) (h_base : base = 3) (h_area : area = 6) (h_formula : area = (1/2) * base * height) : height = 4 :=
by
  sorry

end triangle_height_l236_23678


namespace rate_of_interest_first_year_l236_23609

-- Define the conditions
def principal : ℝ := 9000
def rate_second_year : ℝ := 0.05
def total_amount_after_2_years : ℝ := 9828

-- Define the problem statement which we need to prove
theorem rate_of_interest_first_year (R : ℝ) :
  (principal + (principal * R / 100)) + 
  ((principal + (principal * R / 100)) * rate_second_year) = 
  total_amount_after_2_years → 
  R = 4 := 
by
  sorry

end rate_of_interest_first_year_l236_23609


namespace range_of_function_l236_23675

theorem range_of_function (x y z : ℝ)
  (h : x^2 + y^2 + x - y = 1) :
  ∃ a b : ℝ, (a = (3 * Real.sqrt 6 + Real.sqrt 6) / 2) ∧ (b = (-3 * Real.sqrt 2 + Real.sqrt 6) / 2) ∧
    ∀ f : ℝ, f = (x - 1) * Real.cos z + (y + 1) * Real.sin z →
              b ≤ f ∧ f ≤ a := 
by
  sorry

end range_of_function_l236_23675


namespace three_subsets_equal_sum_l236_23658

theorem three_subsets_equal_sum (n : ℕ) (h1 : n ≡ 0 [MOD 3] ∨ n ≡ 2 [MOD 3]) (h2 : 5 ≤ n) :
  ∃ (A B C : Finset ℕ), A ∪ B ∪ C = Finset.range (n + 1) ∧
                        A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ C ∩ A = ∅ ∧
                        A.sum id = B.sum id ∧ B.sum id = C.sum id ∧ C.sum id = A.sum id :=
sorry

end three_subsets_equal_sum_l236_23658


namespace pyramid_volume_l236_23682

/-- Given the vertices of a triangle and its midpoints, calculate the volume of the folded triangular pyramid. -/
theorem pyramid_volume
  (A B C : ℝ × ℝ)
  (D E F : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (24, 0))
  (hC : C = (12, 16))
  (hD : D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hE : E = ((A.1 + C.1) / 2, (A.2 + C.2) / 2))
  (hF : F = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (area_ABC : ℝ)
  (h_area : area_ABC = 192)
  : (1 / 3) * area_ABC * 8 = 512 :=
by sorry

end pyramid_volume_l236_23682


namespace sum_and_difference_repeating_decimals_l236_23699

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_9 : ℚ := 1
noncomputable def repeating_decimal_3 : ℚ := 1 / 3

theorem sum_and_difference_repeating_decimals :
  repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_9 + repeating_decimal_3 = 2 / 9 := 
by 
  sorry

end sum_and_difference_repeating_decimals_l236_23699


namespace designed_height_correct_l236_23673
noncomputable def designed_height_of_lower_part (H : ℝ) (L : ℝ) : Prop :=
  H = 2 ∧ (H - L) / L = L / H

theorem designed_height_correct : ∃ L, designed_height_of_lower_part 2 L ∧ L = Real.sqrt 5 - 1 :=
by
  sorry

end designed_height_correct_l236_23673


namespace sum_over_term_is_two_l236_23691

-- Definitions of conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = n / 2 * (2 * a 0 + (n - 1) * (a 1 - a 0))

def seq_sn_over_an_arithmetic (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∃ dS : ℝ, ∀ n : ℕ, (S (n + 1)) / (a (n + 1)) = (S n) / (a n) + dS

-- The theorem to prove
theorem sum_over_term_is_two (a S : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : sum_first_n_terms a S)
  (h3 : seq_sn_over_an_arithmetic S a) :
  S 3 / a 3 = 2 :=
sorry

end sum_over_term_is_two_l236_23691


namespace max_value_of_y_l236_23686

noncomputable def maxY (x y : ℝ) : ℝ :=
  if x^2 + y^2 = 10 * x + 60 * y then y else 0

theorem max_value_of_y (x y : ℝ) (h : x^2 + y^2 = 10 * x + 60 * y) : 
  y ≤ 30 + 5 * Real.sqrt 37 :=
sorry

end max_value_of_y_l236_23686


namespace jaysons_moms_age_l236_23670

theorem jaysons_moms_age (jayson's_age dad's_age mom's_age : ℕ) 
  (h1 : jayson's_age = 10)
  (h2 : dad's_age = 4 * jayson's_age)
  (h3 : mom's_age = dad's_age - 2) :
  mom's_age - jayson's_age = 28 := 
by
  sorry

end jaysons_moms_age_l236_23670
