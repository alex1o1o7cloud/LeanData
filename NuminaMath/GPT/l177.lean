import Mathlib

namespace geometric_sequence_common_ratio_l177_177243

theorem geometric_sequence_common_ratio
  (S : ℕ → ℝ)
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : S 1 = a 1)
  (h2 : S 2 = a 1 + a 1 * q)
  (h3 : a 2 = a 1 * q)
  (h4 : a 3 = a 1 * q^2)
  (h5 : 3 * S 2 = a 3 - 2)
  (h6 : 3 * S 1 = a 2 - 2) :
  q = 4 :=
sorry

end geometric_sequence_common_ratio_l177_177243


namespace reciprocal_of_fraction_subtraction_l177_177645

theorem reciprocal_of_fraction_subtraction : (1 / ((2 / 3) - (3 / 4))) = -12 := by
  sorry

end reciprocal_of_fraction_subtraction_l177_177645


namespace inverse_proportion_function_l177_177224

theorem inverse_proportion_function (m x : ℝ) (h : (m ≠ 0)) (A : (m, m / 8) ∈ {p : ℝ × ℝ | p.snd = (m / p.fst)}) :
    ∃ f : ℝ → ℝ, (∀ x, f x = 8 / x) :=
by
  use (fun x => 8 / x)
  intros x
  rfl

end inverse_proportion_function_l177_177224


namespace grace_earnings_september_l177_177346

def charge_small_lawn_per_hour := 6
def charge_large_lawn_per_hour := 10
def charge_pull_small_weeds_per_hour := 11
def charge_pull_large_weeds_per_hour := 15
def charge_small_mulch_per_hour := 9
def charge_large_mulch_per_hour := 13

def hours_small_lawn := 20
def hours_large_lawn := 43
def hours_small_weeds := 4
def hours_large_weeds := 5
def hours_small_mulch := 6
def hours_large_mulch := 4

def earnings_small_lawn := hours_small_lawn * charge_small_lawn_per_hour
def earnings_large_lawn := hours_large_lawn * charge_large_lawn_per_hour
def earnings_small_weeds := hours_small_weeds * charge_pull_small_weeds_per_hour
def earnings_large_weeds := hours_large_weeds * charge_pull_large_weeds_per_hour
def earnings_small_mulch := hours_small_mulch * charge_small_mulch_per_hour
def earnings_large_mulch := hours_large_mulch * charge_large_mulch_per_hour

def total_earnings : ℕ :=
  earnings_small_lawn + earnings_large_lawn + earnings_small_weeds + earnings_large_weeds +
  earnings_small_mulch + earnings_large_mulch

theorem grace_earnings_september : total_earnings = 775 :=
by
  sorry

end grace_earnings_september_l177_177346


namespace sum_of_numbers_l177_177736

theorem sum_of_numbers (a b : ℕ) (h : a + 4 * b = 30) : a + b = 12 :=
sorry

end sum_of_numbers_l177_177736


namespace factorization_correct_l177_177356

theorem factorization_correct (c : ℝ) : (x : ℝ) → x^2 - x + c = (x + 2) * (x - 3) → c = -6 := by
  intro x h
  sorry

end factorization_correct_l177_177356


namespace cos_B_of_triangle_l177_177442

theorem cos_B_of_triangle (A B : ℝ) (a b : ℝ) (h1 : A = 2 * B) (h2 : a = 6) (h3 : b = 4) :
  Real.cos B = 3 / 4 :=
by
  sorry

end cos_B_of_triangle_l177_177442


namespace correct_calculation_l177_177067

theorem correct_calculation :
  3 * Real.sqrt 2 - (Real.sqrt 2 / 2) = (5 / 2) * Real.sqrt 2 :=
by
  -- To proceed with the proof, we need to show:
  -- 3 * sqrt(2) - (sqrt(2) / 2) = (5 / 2) * sqrt(2)
  sorry

end correct_calculation_l177_177067


namespace find_k_value_l177_177910

theorem find_k_value (k : ℕ) :
  3 * 6 * 4 * k = Nat.factorial 8 → k = 560 :=
by
  sorry

end find_k_value_l177_177910


namespace min_denominator_of_sum_600_700_l177_177468

def is_irreducible_fraction (a : ℕ) (b : ℕ) : Prop := 
  Nat.gcd a b = 1

def min_denominator_of_sum (d1 d2 : ℕ) (a b : ℕ) : ℕ :=
  let lcm := Nat.lcm d1 d2
  let sum_numerator := a * (lcm / d1) + b * (lcm / d2)
  Nat.gcd sum_numerator lcm

theorem min_denominator_of_sum_600_700 (a b : ℕ) (h1 : is_irreducible_fraction a 600) (h2 : is_irreducible_fraction b 700) :
  min_denominator_of_sum 600 700 a b = 168 := sorry

end min_denominator_of_sum_600_700_l177_177468


namespace weight_of_raisins_proof_l177_177057

-- Define the conditions
def weight_of_peanuts : ℝ := 0.1
def total_weight_of_snacks : ℝ := 0.5

-- Theorem to prove that the weight of raisins equals 0.4 pounds
theorem weight_of_raisins_proof : total_weight_of_snacks - weight_of_peanuts = 0.4 := by
  sorry

end weight_of_raisins_proof_l177_177057


namespace arithmetic_problem_l177_177973

theorem arithmetic_problem : 
  let x := 512.52 
  let y := 256.26 
  let diff := x - y 
  let result := diff * 3 
  result = 768.78 := 
by 
  sorry

end arithmetic_problem_l177_177973


namespace four_distinct_real_roots_l177_177899

noncomputable def f (x c : ℝ) : ℝ := x^2 + 4 * x + c

-- We need to prove that if c is in the interval (-1, 3), f(f(x)) has exactly 4 distinct real roots
theorem four_distinct_real_roots (c : ℝ) : (-1 < c) ∧ (c < 3) → 
  ∃ (x₁ x₂ x₃ x₄ : ℝ), (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) 
  ∧ (f (f x₁ c) c = 0 ∧ f (f x₂ c) c = 0 ∧ f (f x₃ c) c = 0 ∧ f (f x₄ c) c = 0) :=
by sorry

end four_distinct_real_roots_l177_177899


namespace recurring_decimal_to_fraction_l177_177050

theorem recurring_decimal_to_fraction : (∃ (x : ℚ), x = 3 + 56 / 99) :=
by
  have x : ℚ := 3 + 56 / 99
  exists x
  sorry

end recurring_decimal_to_fraction_l177_177050


namespace arithmetic_seq_sum_l177_177740

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h_arith : is_arithmetic_sequence a)
  (h1 : a 1 + a 4 + a 7 = 39) (h2 : a 2 + a 5 + a 8 = 33) :
  a 3 + a 6 + a 9 = 27 :=
sorry

end arithmetic_seq_sum_l177_177740


namespace differentiable_function_zero_l177_177715

noncomputable def f : ℝ → ℝ := sorry

theorem differentiable_function_zero (f : ℝ → ℝ) (h_diff : ∀ x ≥ 0, DifferentiableAt ℝ f x)
  (h_f0 : f 0 = 0) (h_fun : ∀ x ≥ 0, ∀ y ≥ 0, (x = y^2) → deriv f x = f y) : 
  ∀ x ≥ 0, f x = 0 :=
by
  sorry

end differentiable_function_zero_l177_177715


namespace proof_problem1_proof_problem2_l177_177043

noncomputable def problem1_lhs : ℝ := 
  1 / (Real.sqrt 3 + 1) - Real.sin (Real.pi / 3) + Real.sqrt 32 * Real.sqrt (1 / 8)

noncomputable def problem1_rhs : ℝ := 3 / 2

theorem proof_problem1 : problem1_lhs = problem1_rhs :=
by 
  sorry

noncomputable def problem2_lhs : ℝ := 
  2^(-2 : ℤ) - Real.sqrt ((-2)^2) + 6 * Real.sin (Real.pi / 4) - Real.sqrt 18

noncomputable def problem2_rhs : ℝ := -7 / 4

theorem proof_problem2 : problem2_lhs = problem2_rhs :=
by 
  sorry

end proof_problem1_proof_problem2_l177_177043


namespace sum_of_hundreds_and_tens_digits_of_product_l177_177998

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def seq_num (a : ℕ) (x : ℕ) := List.foldr (λ _ acc => acc * 1000 + a) 0 (List.range x)

noncomputable def num_a := seq_num 707 101
noncomputable def num_b := seq_num 909 101

noncomputable def product := num_a * num_b

theorem sum_of_hundreds_and_tens_digits_of_product :
  hundreds_digit product + tens_digit product = 8 := by
  sorry

end sum_of_hundreds_and_tens_digits_of_product_l177_177998


namespace chess_tournament_participants_l177_177927

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 105) : n = 15 := by
  sorry

end chess_tournament_participants_l177_177927


namespace tan_half_angle_l177_177165

theorem tan_half_angle {α β : ℝ} (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) (h1 : Real.tan α = 2) (h2 : Real.tan β = 3) :
  Real.tan ((α + β) / 2) = 1 + Real.sqrt 2 := 
sorry

end tan_half_angle_l177_177165


namespace total_passengers_transportation_l177_177791

theorem total_passengers_transportation : 
  let passengers_one_way := 100
  let passengers_return := 60
  let first_trip_total := passengers_one_way + passengers_return
  let additional_trips := 3
  let additional_trips_total := additional_trips * first_trip_total
  let total_passengers := first_trip_total + additional_trips_total
  total_passengers = 640 := 
by
  sorry

end total_passengers_transportation_l177_177791


namespace isosceles_triangle_vertex_angle_l177_177662

theorem isosceles_triangle_vertex_angle (B : ℝ) (V : ℝ) (h1 : B = 70) (h2 : B = B) (h3 : V + 2 * B = 180) : V = 40 ∨ V = 70 :=
by {
  sorry
}

end isosceles_triangle_vertex_angle_l177_177662


namespace total_cans_collected_l177_177349

theorem total_cans_collected :
  let cans_in_first_bag := 5
  let cans_in_second_bag := 7
  let cans_in_third_bag := 12
  let cans_in_fourth_bag := 4
  let cans_in_fifth_bag := 8
  let cans_in_sixth_bag := 10
  let cans_in_seventh_bag := 15
  let cans_in_eighth_bag := 6
  let cans_in_ninth_bag := 5
  let cans_in_tenth_bag := 13
  let total_cans := cans_in_first_bag + cans_in_second_bag + cans_in_third_bag + cans_in_fourth_bag + cans_in_fifth_bag + cans_in_sixth_bag + cans_in_seventh_bag + cans_in_eighth_bag + cans_in_ninth_bag + cans_in_tenth_bag
  total_cans = 85 :=
by
  sorry

end total_cans_collected_l177_177349


namespace greatest_positive_integer_x_l177_177922

theorem greatest_positive_integer_x (x : ℕ) (h₁ : x^2 < 12) (h₂ : ∀ y: ℕ, y^2 < 12 → y ≤ x) : 
  x = 3 := 
by
  sorry

end greatest_positive_integer_x_l177_177922


namespace rectangle_width_l177_177467

theorem rectangle_width (L W : ℝ) (h₁ : 2 * L + 2 * W = 54) (h₂ : W = L + 3) : W = 15 :=
sorry

end rectangle_width_l177_177467


namespace frank_maze_time_l177_177815

theorem frank_maze_time 
    (n mazes : ℕ)
    (avg_time_per_maze completed_time total_allowable_time remaining_maze_time extra_time_inside current_time : ℕ) 
    (h1 : mazes = 5)
    (h2 : avg_time_per_maze = 60)
    (h3 : completed_time = 200)
    (h4 : total_allowable_time = mazes * avg_time_per_maze)
    (h5 : total_allowable_time = 300)
    (h6 : remaining_maze_time = total_allowable_time - completed_time) 
    (h7 : extra_time_inside = 55)
    (h8 : current_time + extra_time_inside ≤ remaining_maze_time) :
  current_time = 45 :=
by
  sorry

end frank_maze_time_l177_177815


namespace mary_screws_l177_177046

theorem mary_screws (S : ℕ) (h : S + 2 * S = 24) : S = 8 :=
by sorry

end mary_screws_l177_177046


namespace find_b_l177_177861

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 35 * b) : b = 63 := 
by 
  sorry

end find_b_l177_177861


namespace arrangement_plans_count_l177_177660

noncomputable def number_of_arrangement_plans (num_teachers : ℕ) (num_students : ℕ) : ℕ :=
if num_teachers = 2 ∧ num_students = 4 then 12 else 0

theorem arrangement_plans_count :
  number_of_arrangement_plans 2 4 = 12 :=
by 
  sorry

end arrangement_plans_count_l177_177660


namespace quadratic_has_distinct_real_roots_l177_177473

-- Definitions for the quadratic equation coefficients
def a : ℝ := 3
def b : ℝ := -4
def c : ℝ := 1

-- Definition of the discriminant
def Δ : ℝ := b^2 - 4 * a * c

-- Statement of the problem: Prove that the quadratic equation has two distinct real roots
theorem quadratic_has_distinct_real_roots (hΔ : Δ = 4) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) :=
by
  sorry

end quadratic_has_distinct_real_roots_l177_177473


namespace cloth_sold_worth_l177_177937

-- Define the commission rate and commission received
def commission_rate := 0.05
def commission_received := 12.50

-- State the theorem to be proved
theorem cloth_sold_worth : commission_received / commission_rate = 250 :=
by
  sorry

end cloth_sold_worth_l177_177937


namespace paul_walking_time_l177_177045

variable (P : ℕ)

def is_walking_time (P : ℕ) : Prop :=
  P + 7 * (P + 2) = 46

theorem paul_walking_time (h : is_walking_time P) : P = 4 :=
by sorry

end paul_walking_time_l177_177045


namespace harry_to_sue_nuts_ratio_l177_177286

-- Definitions based on conditions
def sue_nuts : ℕ := 48
def bill_nuts (harry_nuts : ℕ) : ℕ := 6 * harry_nuts
def total_nuts (harry_nuts : ℕ) : ℕ := bill_nuts harry_nuts + harry_nuts

-- Proving the ratio
theorem harry_to_sue_nuts_ratio (H : ℕ) (h1 : sue_nuts = 48) (h2 : bill_nuts H + H = 672) : H / sue_nuts = 2 :=
by
  sorry

end harry_to_sue_nuts_ratio_l177_177286


namespace isosceles_triangle_side_length_l177_177532

theorem isosceles_triangle_side_length :
  let a := 1
  let b := Real.sqrt 3
  let right_triangle_area := (1 / 2) * a * b
  let isosceles_triangle_area := right_triangle_area / 3
  ∃ s, s = Real.sqrt 109 / 6 ∧ 
    (∀ (base height : ℝ), 
      (base = a / 3 ∨ base = b / 3) ∧
      height = (2 * isosceles_triangle_area) / base → 
      1 / 2 * base * height = isosceles_triangle_area) :=
by
  sorry

end isosceles_triangle_side_length_l177_177532


namespace find_C_l177_177428

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

noncomputable def A : ℕ := sum_of_digits (4568 ^ 7777)
noncomputable def B : ℕ := sum_of_digits A
noncomputable def C : ℕ := sum_of_digits B

theorem find_C : C = 5 :=
by
  sorry

end find_C_l177_177428


namespace nine_op_ten_l177_177338

def op (A B : ℕ) : ℚ := (1 : ℚ) / (A * B) + (1 : ℚ) / ((A + 1) * (B + 2))

theorem nine_op_ten : op 9 10 = 7 / 360 := by
  sorry

end nine_op_ten_l177_177338


namespace proposition_is_false_l177_177183

noncomputable def false_proposition : Prop :=
¬(∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), Real.sin x + Real.cos x ≥ 2)

theorem proposition_is_false : false_proposition :=
by
  sorry

end proposition_is_false_l177_177183


namespace student_count_incorrect_l177_177413

theorem student_count_incorrect :
  ∀ k : ℕ, 2012 ≠ 18 + 17 * k :=
by
  intro k
  sorry

end student_count_incorrect_l177_177413


namespace contrapositive_example_l177_177423

theorem contrapositive_example (a b : ℝ) : (a ≠ 0 ∨ b ≠ 0) → (a^2 + b^2 ≠ 0) :=
by
  sorry

end contrapositive_example_l177_177423


namespace algebraic_expression_eval_l177_177515

theorem algebraic_expression_eval (a b : ℝ) 
  (h_eq : ∀ (x : ℝ), ¬(x ≠ 0 ∧ x ≠ 1 ∧ (x / (x - 1) + (x - 1) / x = (a + b * x) / (x^2 - x)))) :
  8 * a + 4 * b - 5 = 27 := 
sorry

end algebraic_expression_eval_l177_177515


namespace find_pair_not_satisfying_equation_l177_177397

theorem find_pair_not_satisfying_equation :
  ¬ (187 * 314 - 104 * 565 = 41) :=
by
  sorry

end find_pair_not_satisfying_equation_l177_177397


namespace distance_between_stripes_l177_177882

theorem distance_between_stripes
  (curb_distance : ℝ) (length_curb : ℝ) (stripe_length : ℝ) (distance_stripes : ℝ)
  (h1 : curb_distance = 60)
  (h2 : length_curb = 20)
  (h3 : stripe_length = 50)
  (h4 : distance_stripes = (length_curb * curb_distance) / stripe_length) :
  distance_stripes = 24 :=
by
  sorry

end distance_between_stripes_l177_177882


namespace broadway_show_total_amount_collected_l177_177271

theorem broadway_show_total_amount_collected (num_adults num_children : ℕ) 
  (adult_ticket_price child_ticket_ratio : ℕ) 
  (child_ticket_price : ℕ) 
  (h1 : num_adults = 400) 
  (h2 : num_children = 200) 
  (h3 : adult_ticket_price = 32) 
  (h4 : child_ticket_ratio = 2) 
  (h5 : adult_ticket_price = child_ticket_ratio * child_ticket_price) : 
  num_adults * adult_ticket_price + num_children * child_ticket_price = 16000 := 
  by 
    sorry

end broadway_show_total_amount_collected_l177_177271


namespace commute_times_abs_diff_l177_177774

def commute_times_avg (x y : ℝ) : Prop := (x + y + 7 + 8 + 9) / 5 = 8
def commute_times_var (x y : ℝ) : Prop := ((x - 8)^2 + (y - 8)^2 + (7 - 8)^2 + (8 - 8)^2 + (9 - 8)^2) / 5 = 4

theorem commute_times_abs_diff (x y : ℝ) (h_avg : commute_times_avg x y) (h_var : commute_times_var x y) :
  |x - y| = 6 :=
sorry

end commute_times_abs_diff_l177_177774


namespace cabin_charges_per_night_l177_177140

theorem cabin_charges_per_night 
  (total_lodging_cost : ℕ)
  (hostel_cost_per_night : ℕ)
  (hostel_days : ℕ)
  (total_cabin_days : ℕ)
  (friends_sharing_expenses : ℕ)
  (jimmy_lodging_expense : ℕ) 
  (total_cost_paid_by_jimmy : ℕ) :
  total_lodging_cost = total_cost_paid_by_jimmy →
  hostel_cost_per_night = 15 →
  hostel_days = 3 →
  total_cabin_days = 2 →
  friends_sharing_expenses = 3 →
  jimmy_lodging_expense = 75 →
  ∃ cabin_cost_per_night, cabin_cost_per_night = 45 :=
by
  sorry

end cabin_charges_per_night_l177_177140


namespace smallest_integer_for_polynomial_div_l177_177627

theorem smallest_integer_for_polynomial_div (x : ℤ) : 
  (∃ k : ℤ, x = 6) ↔ ∃ y, y * (x - 5) = x^2 + 4 * x + 7 := 
by 
  sorry

end smallest_integer_for_polynomial_div_l177_177627


namespace operation_difference_l177_177597

def operation (x y : ℕ) : ℕ := x * y - 3 * x + y

theorem operation_difference : operation 5 9 - operation 9 5 = 16 :=
by
  sorry

end operation_difference_l177_177597


namespace cylinder_volume_l177_177025

-- Definitions based on conditions
def lateral_surface_to_rectangle (generatrix_a generatrix_b : ℝ) (volume : ℝ) :=
  -- Condition: Rectangle with sides 8π and 4π
  (generatrix_a = 8 * Real.pi ∧ volume = 32 * Real.pi^2) ∨
  (generatrix_a = 4 * Real.pi ∧ volume = 64 * Real.pi^2)

-- Statement
theorem cylinder_volume (generatrix_a generatrix_b : ℝ)
  (h : (generatrix_a = 8 * Real.pi ∨ generatrix_b = 4 * Real.pi) ∧ (generatrix_b = 4 * Real.pi ∨ generatrix_b = 8 * Real.pi)) :
  ∃ (volume : ℝ), lateral_surface_to_rectangle generatrix_a generatrix_b volume :=
sorry

end cylinder_volume_l177_177025


namespace equality_of_expressions_l177_177402

theorem equality_of_expressions (a b c : ℝ) (h : a = b + c + 2) : 
  a + b * c = (a + b) * (a + c) ↔ a = 0 ∨ a = 1 :=
by sorry

end equality_of_expressions_l177_177402


namespace total_machine_operation_time_l177_177463

theorem total_machine_operation_time 
  (num_dolls : ℕ) 
  (shoes_per_doll bags_per_doll cosmetics_per_doll hats_per_doll : ℕ) 
  (time_per_doll time_per_accessory : ℕ)
  (num_shoes num_bags num_cosmetics num_hats num_accessories : ℕ) 
  (total_doll_time total_accessory_time total_time : ℕ) :
  num_dolls = 12000 →
  shoes_per_doll = 2 →
  bags_per_doll = 3 →
  cosmetics_per_doll = 1 →
  hats_per_doll = 5 →
  time_per_doll = 45 →
  time_per_accessory = 10 →
  num_shoes = num_dolls * shoes_per_doll →
  num_bags = num_dolls * bags_per_doll →
  num_cosmetics = num_dolls * cosmetics_per_doll →
  num_hats = num_dolls * hats_per_doll →
  num_accessories = num_shoes + num_bags + num_cosmetics + num_hats →
  total_doll_time = num_dolls * time_per_doll →
  total_accessory_time = num_accessories * time_per_accessory →
  total_time = total_doll_time + total_accessory_time →
  total_time = 1860000 := 
sorry

end total_machine_operation_time_l177_177463


namespace roots_opposite_signs_l177_177929

theorem roots_opposite_signs (p : ℝ) (hp : p > 0) :
  ( ∃ (x₁ x₂ : ℝ), (x₁ * x₂ < 0) ∧ (5 * x₁^2 - 4 * (p + 3) * x₁ + 4 = p^2) ∧  
      (5 * x₂^2 - 4 * (p + 3) * x₂ + 4 = p^2) ) ↔ p > 2 :=
by {
  sorry
}

end roots_opposite_signs_l177_177929


namespace quotient_multiple_of_y_l177_177053

theorem quotient_multiple_of_y (x y m : ℤ) (h1 : x = 11 * y + 4) (h2 : 2 * x = 8 * m * y + 3) (h3 : 13 * y - x = 1) : m = 3 :=
by
  sorry

end quotient_multiple_of_y_l177_177053


namespace abs_inequality_solution_l177_177192

theorem abs_inequality_solution (x : ℝ) : |x - 1| + |x - 3| < 8 ↔ -2 < x ∧ x < 6 :=
by sorry

end abs_inequality_solution_l177_177192


namespace average_salary_increase_l177_177854

theorem average_salary_increase 
  (average_salary : ℕ) (manager_salary : ℕ)
  (n : ℕ) (initial_count : ℕ) (new_count : ℕ) (initial_average : ℕ)
  (total_salary : ℕ) (new_total_salary : ℕ) (new_average : ℕ)
  (salary_increase : ℕ) :
  initial_average = 1500 →
  manager_salary = 3600 →
  initial_count = 20 →
  new_count = initial_count + 1 →
  total_salary = initial_count * initial_average →
  new_total_salary = total_salary + manager_salary →
  new_average = new_total_salary / new_count →
  salary_increase = new_average - initial_average →
  salary_increase = 100 := by
  sorry

end average_salary_increase_l177_177854


namespace circle_equation_from_parabola_l177_177369

theorem circle_equation_from_parabola :
  let F := (2, 0)
  let A := (2, 4)
  let B := (2, -4)
  let diameter := 8
  let center := F
  let radius_squared := diameter^2 / 4
  (x - center.1)^2 + y^2 = radius_squared :=
by
  sorry

end circle_equation_from_parabola_l177_177369


namespace ratio_a_b_l177_177447

theorem ratio_a_b (a b c d : ℝ) 
  (h1 : b / c = 7 / 9) 
  (h2 : c / d = 5 / 7)
  (h3 : a / d = 5 / 12) : 
  a / b = 3 / 4 :=
  sorry

end ratio_a_b_l177_177447


namespace value_a6_l177_177988

noncomputable def seq (a : ℕ → ℕ) : Prop :=
a 1 = 1 ∧ ∀ n ≥ 2, a n - a (n - 1) = n - 1

theorem value_a6 : ∃ a : ℕ → ℕ, seq a ∧ a 6 = 16 := by
  sorry

end value_a6_l177_177988


namespace find_coordinates_of_C_l177_177496

def Point := (ℝ × ℝ)

def A : Point := (-2, -1)
def B : Point := (4, 7)

/-- A custom definition to express that point C divides the segment AB in the ratio 2:1 from point B. -/
def is_point_C (C : Point) : Prop :=
  ∃ k : ℝ, k = 2 / 3 ∧
  C = (k * A.1 + (1 - k) * B.1, k * A.2 + (1 - k) * B.2)

theorem find_coordinates_of_C (C : Point) (h : is_point_C C) : 
  C = (2, 13 / 3) :=
sorry

end find_coordinates_of_C_l177_177496


namespace find_f_of_2_l177_177828

theorem find_f_of_2 : ∃ (f : ℤ → ℤ), (∀ x : ℤ, f (x+1) = x^2 - 1) ∧ f 2 = 0 :=
by
  sorry

end find_f_of_2_l177_177828


namespace present_age_of_son_l177_177090

/-- A man is 46 years older than his son and in two years, the man's age will be twice the age of his son. Prove that the present age of the son is 44. -/
theorem present_age_of_son (M S : ℕ) (h1 : M = S + 46) (h2 : M + 2 = 2 * (S + 2)) : S = 44 :=
by {
  sorry
}

end present_age_of_son_l177_177090


namespace find_z_value_l177_177576

variables {BD FC GC FE : Prop}
variables {a b c d e f g z : ℝ}

-- Assume all given conditions
axiom BD_is_straight : BD
axiom FC_is_straight : FC
axiom GC_is_straight : GC
axiom FE_is_straight : FE
axiom sum_is_z : z = a + b + c + d + e + f + g

-- Goal to prove
theorem find_z_value : z = 540 :=
by
  sorry

end find_z_value_l177_177576


namespace total_distance_is_correct_l177_177231

noncomputable def boat_speed : ℝ := 20 -- boat speed in still water (km/hr)
noncomputable def current_speed_first : ℝ := 5 -- current speed for the first 6 minutes (km/hr)
noncomputable def current_speed_second : ℝ := 8 -- current speed for the next 6 minutes (km/hr)
noncomputable def current_speed_third : ℝ := 3 -- current speed for the last 6 minutes (km/hr)
noncomputable def time_in_hours : ℝ := 6 / 60 -- 6 minutes in hours (0.1 hours)

noncomputable def total_distance_downstream := 
  (boat_speed + current_speed_first) * time_in_hours +
  (boat_speed + current_speed_second) * time_in_hours +
  (boat_speed + current_speed_third) * time_in_hours

theorem total_distance_is_correct : total_distance_downstream = 7.6 :=
  by 
  sorry

end total_distance_is_correct_l177_177231


namespace repeating_decimal_eq_fraction_l177_177616

theorem repeating_decimal_eq_fraction :
  let a := (85 : ℝ) / 100
  let r := (1 : ℝ) / 100
  (∑' n : ℕ, a * (r ^ n)) = 85 / 99 := by
  let a := (85 : ℝ) / 100
  let r := (1 : ℝ) / 100
  exact sorry

end repeating_decimal_eq_fraction_l177_177616


namespace trig_identity_l177_177799

theorem trig_identity (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) 
  : Real.cos (5 / 6 * π + α) + (Real.cos (4 * π / 3 + α))^2 = (2 - Real.sqrt 3) / 3 := 
sorry

end trig_identity_l177_177799


namespace compute_operation_l177_177247

def operation_and (x : ℝ) := 10 - x
def operation_and_prefix (x : ℝ) := x - 10

theorem compute_operation (x : ℝ) : operation_and_prefix (operation_and 15) = -15 :=
by
  sorry

end compute_operation_l177_177247


namespace main_theorem_l177_177970

noncomputable def proof_problem (α : ℝ) (h1 : 0 < α) (h2 : α < π) : Prop :=
  2 * Real.sin (2 * α) ≤ Real.cos (α / 2)

noncomputable def equality_case (α : ℝ) (h1 : 0 < α) (h2 : α < π) : Prop :=
  α = π / 3 → 2 * Real.sin (2 * α) = Real.cos (α / 2)

theorem main_theorem (α : ℝ) (h1 : 0 < α) (h2 : α < π) :
  proof_problem α h1 h2 ∧ equality_case α h1 h2 :=
by
  sorry

end main_theorem_l177_177970


namespace angles_equal_l177_177566

theorem angles_equal {α β γ α1 β1 γ1 : ℝ} (h1 : α + β + γ = 180) (h2 : α1 + β1 + γ1 = 180) 
  (h_eq_or_sum_to_180 : (α = α1 ∨ α + α1 = 180) ∧ (β = β1 ∨ β + β1 = 180) ∧ (γ = γ1 ∨ γ + γ1 = 180)) :
  α = α1 ∧ β = β1 ∧ γ = γ1 := 
by 
  sorry

end angles_equal_l177_177566


namespace smallest_m_for_reflection_l177_177305

noncomputable def theta : Real := Real.arctan (1 / 3)
noncomputable def pi_8 : Real := Real.pi / 8
noncomputable def pi_12 : Real := Real.pi / 12
noncomputable def pi_4 : Real := Real.pi / 4
noncomputable def pi_6 : Real := Real.pi / 6

/-- The smallest positive integer m such that R^(m)(l) = l
where the transformation R(l) is described as:
l is reflected in l1 (angle pi/8), then the resulting line is
reflected in l2 (angle pi/12) -/
theorem smallest_m_for_reflection :
  ∃ (m : ℕ), m > 0 ∧ ∀ (k : ℤ), m = 12 * k + 12 := by
sorry

end smallest_m_for_reflection_l177_177305


namespace g_f_neg2_l177_177622

def f (x : ℤ) : ℤ := x^3 + 3

def g (x : ℤ) : ℤ := 2*x^2 + 2*x + 1

theorem g_f_neg2 : g (f (-2)) = 41 :=
by {
  -- proof steps skipped
  sorry
}

end g_f_neg2_l177_177622


namespace units_digit_expression_mod_10_l177_177498

theorem units_digit_expression_mod_10 : ((2 ^ 2023) * (5 ^ 2024) * (11 ^ 2025)) % 10 = 0 := 
by 
  -- Proof steps would go here
  sorry

end units_digit_expression_mod_10_l177_177498


namespace percentage_honda_red_l177_177920

theorem percentage_honda_red (total_cars : ℕ) (honda_cars : ℕ) (percentage_red_total : ℚ)
  (percentage_red_non_honda : ℚ) (percentage_red_honda : ℚ) :
  total_cars = 9000 →
  honda_cars = 5000 →
  percentage_red_total = 0.60 →
  percentage_red_non_honda = 0.225 →
  percentage_red_honda = 0.90 →
  ((honda_cars * percentage_red_honda) / total_cars) * 100 = ((total_cars * percentage_red_total - (total_cars - honda_cars) * percentage_red_non_honda) / honda_cars) * 100 :=
by
  sorry

end percentage_honda_red_l177_177920


namespace investment_amount_correct_l177_177085

-- Lean statement definitions based on conditions
def cost_per_tshirt : ℕ := 3
def selling_price_per_tshirt : ℕ := 20
def tshirts_sold : ℕ := 83
def total_revenue : ℕ := tshirts_sold * selling_price_per_tshirt
def total_cost_of_tshirts : ℕ := tshirts_sold * cost_per_tshirt
def investment_in_equipment : ℕ := total_revenue - total_cost_of_tshirts

-- Theorem statement
theorem investment_amount_correct : investment_in_equipment = 1411 := by
  sorry

end investment_amount_correct_l177_177085


namespace inequality_my_problem_l177_177021

theorem inequality_my_problem (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a * b + b * c + c * a = 1) :
  (Real.sqrt ((1 / a) + 6 * b)) + (Real.sqrt ((1 / b) + 6 * c)) + (Real.sqrt ((1 / c) + 6 * a)) ≤ (1 / (a * b * c)) :=
  sorry

end inequality_my_problem_l177_177021


namespace range_of_x_satisfying_inequality_l177_177354

noncomputable def f : ℝ → ℝ := sorry -- f is some even and monotonically increasing function

theorem range_of_x_satisfying_inequality :
  (∀ x, f (-x) = f x) ∧ (∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y) → {x : ℝ | f x < f 1} = {x : ℝ | -1 < x ∧ x < 1} :=
by
  intro h
  sorry

end range_of_x_satisfying_inequality_l177_177354


namespace camera_filter_kit_savings_l177_177244

variable (kit_price : ℝ) (single_prices : List ℝ)
variable (correct_saving_amount : ℝ)

theorem camera_filter_kit_savings
    (h1 : kit_price = 145.75)
    (h2 : single_prices = [3 * 9.50, 2 * 15.30, 1 * 20.75, 2 * 25.80])
    (h3 : correct_saving_amount = -14.30) :
    (single_prices.sum - kit_price = correct_saving_amount) :=
by
  sorry

end camera_filter_kit_savings_l177_177244


namespace problem_quadratic_radicals_l177_177493

theorem problem_quadratic_radicals (x y : ℝ) (h : 3 * y = x + 2 * y + 2) : x - y = -2 :=
sorry

end problem_quadratic_radicals_l177_177493


namespace mutually_exclusive_but_not_complementary_l177_177190

-- Definitions for the problem conditions
inductive Card
| red | black | white | blue

inductive Person
| A | B | C | D

open Card Person

-- The statement of the proof
theorem mutually_exclusive_but_not_complementary : 
  (∃ (f : Person → Card), (f A = red) ∧ (f B ≠ red)) ∧ (∃ (f : Person → Card), (f B = red) ∧ (f A ≠ red)) :=
sorry

end mutually_exclusive_but_not_complementary_l177_177190


namespace complex_number_location_second_quadrant_l177_177549

theorem complex_number_location_second_quadrant (z : ℂ) (h : z / (1 + I) = I) : z.re < 0 ∧ z.im > 0 :=
by sorry

end complex_number_location_second_quadrant_l177_177549


namespace parabola_distance_l177_177618

theorem parabola_distance (P : ℝ × ℝ) (hP : P.2^2 = 4 * P.1)
  (h_distance_focus : (P.1 - 1)^2 + P.2^2 = 9) : 
  Real.sqrt (P.1^2 + P.2^2) = 2 * Real.sqrt 3 :=
by
  sorry

end parabola_distance_l177_177618


namespace calculate_nested_expression_l177_177954

theorem calculate_nested_expression :
  3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2 = 1457 :=
by
  sorry

end calculate_nested_expression_l177_177954


namespace average_of_numbers_in_range_l177_177638

-- Define the set of numbers we are considering
def numbers_in_range : List ℕ := [10, 15, 20, 25, 30]

-- Define the sum of these numbers
def sum_in_range : ℕ := 10 + 15 + 20 + 25 + 30

-- Define the number of elements in our range
def count_in_range : ℕ := 5

-- Prove that the average of numbers in the range is 20
theorem average_of_numbers_in_range : (sum_in_range / count_in_range) = 20 := by
  -- TODO: Proof to be written, for now we use sorry as a placeholder
  sorry

end average_of_numbers_in_range_l177_177638


namespace points_of_third_l177_177592

noncomputable def points_of_first : ℕ := 11
noncomputable def points_of_second : ℕ := 7
noncomputable def points_of_fourth : ℕ := 2
noncomputable def johns_total_points : ℕ := 38500

theorem points_of_third :
  ∃ x : ℕ, (points_of_first * points_of_second * x * points_of_fourth ∣ johns_total_points) ∧
    (johns_total_points / (points_of_first * points_of_second * points_of_fourth)) = x := 
sorry

end points_of_third_l177_177592


namespace tan_theta_solution_l177_177553

theorem tan_theta_solution (θ : ℝ)
  (h : 2 * Real.sin (θ + Real.pi / 3) = 3 * Real.sin (Real.pi / 3 - θ)) :
  Real.tan θ = Real.sqrt 3 / 5 := sorry

end tan_theta_solution_l177_177553


namespace cubes_with_one_face_painted_cubes_with_two_faces_painted_size_of_new_cube_l177_177696

def cube (n : ℕ) : Type := ℕ × ℕ × ℕ

-- Define a 4x4x4 cube and the painting conditions
def four_by_four_cube := cube 4

-- Determine the number of small cubes with exactly one face painted
theorem cubes_with_one_face_painted : 
  ∃ (count : ℕ), count = 24 :=
by
  -- proof goes here
  sorry

-- Determine the number of small cubes with exactly two faces painted
theorem cubes_with_two_faces_painted : 
  ∃ (count : ℕ), count = 24 :=
by
  -- proof goes here
  sorry

-- Given condition and find the size of the new cube
theorem size_of_new_cube (n : ℕ) : 
  (n - 2) ^ 3 = 3 * 12 * (n - 2) → n = 8 :=
by
  -- proof goes here
  sorry

end cubes_with_one_face_painted_cubes_with_two_faces_painted_size_of_new_cube_l177_177696


namespace shares_distribution_correct_l177_177303

def shares_distributed (a b c d e : ℕ) : Prop :=
  a = 50 ∧ b = 100 ∧ c = 300 ∧ d = 150 ∧ e = 600

theorem shares_distribution_correct (a b c d e : ℕ) :
  (a = (1/2 : ℚ) * b)
  ∧ (b = (1/3 : ℚ) * c)
  ∧ (c = 2 * d)
  ∧ (d = (1/4 : ℚ) * e)
  ∧ (a + b + c + d + e = 1200) → shares_distributed a b c d e :=
sorry

end shares_distribution_correct_l177_177303


namespace original_inhabitants_proof_l177_177186

noncomputable def original_inhabitants (final_population : ℕ) : ℝ :=
  final_population / (0.75 * 0.9)

theorem original_inhabitants_proof :
  original_inhabitants 5265 = 7800 :=
by
  sorry

end original_inhabitants_proof_l177_177186


namespace ellipse_eccentricity_l177_177023

theorem ellipse_eccentricity (m : ℝ) (e : ℝ) : 
  (∀ x y : ℝ, (x^2 / m) + (y^2 / 4) = 1) ∧ foci_y_axis ∧ e = 1 / 2 → m = 3 :=
by
  sorry

end ellipse_eccentricity_l177_177023


namespace find_base_l177_177725

-- Definitions based on the conditions of the problem
def is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
def is_perfect_cube (n : ℕ) := ∃ m : ℕ, m * m * m = n
def is_perfect_fourth (n : ℕ) := ∃ m : ℕ, m * m * m * m = n

-- Define the number A in terms of base a
def A (a : ℕ) : ℕ := 4 * a * a + 4 * a + 1

-- Problem statement: find a base a > 4 such that A is both a perfect cube and a perfect fourth power
theorem find_base (a : ℕ)
  (ha : a > 4)
  (h_square : is_perfect_square (A a)) :
  is_perfect_cube (A a) ∧ is_perfect_fourth (A a) :=
sorry

end find_base_l177_177725


namespace intersect_point_l177_177363

-- Definitions as per conditions
def f (x : ℝ) (b : ℝ) : ℝ := 4 * x + b
def f_inv (x : ℝ) (a : ℝ) : ℝ := a -- We define inverse as per given (4, a)

-- Variables for the conditions
variables (a b : ℤ)

-- Theorems to prove the conditions match the answers
theorem intersect_point : ∃ a b : ℤ, f 4 b = a ∧ f_inv 4 a = 4 ∧ a = 4 := by
  sorry

end intersect_point_l177_177363


namespace packages_per_box_l177_177136

theorem packages_per_box (P : ℕ) (h1 : 192 > 0) (h2 : 2 > 0) (total_soaps : 2304 > 0) (h : 2 * P * 192 = 2304) : P = 6 :=
by
  sorry

end packages_per_box_l177_177136


namespace kayla_scores_on_sixth_level_l177_177148

-- Define the sequence of points scored in each level
def points (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | 1 => 3
  | 2 => 5
  | 3 => 8
  | 4 => 12
  | n + 5 => points (n + 4) + (n + 1) + 1

-- Statement to prove that Kayla scores 17 points on the sixth level
theorem kayla_scores_on_sixth_level : points 5 = 17 :=
by
  sorry

end kayla_scores_on_sixth_level_l177_177148


namespace missing_angle_in_convex_polygon_l177_177762

theorem missing_angle_in_convex_polygon (n : ℕ) (x : ℝ) 
  (h1 : n ≥ 5) 
  (h2 : 180 * (n - 2) - 3 * x = 3330) : 
  x = 54 := 
by 
  sorry

end missing_angle_in_convex_polygon_l177_177762


namespace circle_standard_form1_circle_standard_form2_l177_177033

theorem circle_standard_form1 (x y : ℝ) :
  x^2 + y^2 - 4 * x + 6 * y - 3 = 0 ↔ (x - 2)^2 + (y + 3)^2 = 16 :=
by
  sorry

theorem circle_standard_form2 (x y : ℝ) :
  4 * x^2 + 4 * y^2 - 8 * x + 4 * y - 11 = 0 ↔ (x - 1)^2 + (y + 1 / 2)^2 = 4 :=
by
  sorry

end circle_standard_form1_circle_standard_form2_l177_177033


namespace abs_cube_root_neg_64_l177_177950

-- Definitions required for the problem
def cube_root (x : ℝ) : ℝ := x^(1/3)
def abs_value (x : ℝ) : ℝ := abs x

-- The statement of the problem
theorem abs_cube_root_neg_64 : abs_value (cube_root (-64)) = 4 :=
by sorry

end abs_cube_root_neg_64_l177_177950


namespace money_distribution_l177_177632

theorem money_distribution (a : ℕ) (h1 : 5 * a = 1500) : 7 * a - 3 * a = 1200 := by
  sorry

end money_distribution_l177_177632


namespace ratio_longer_to_shorter_side_l177_177073

-- Definitions of the problem
variables (l s : ℝ)
def rect_sheet_fold : Prop :=
  l = Real.sqrt (s^2 + (s^2 / l)^2)

-- The to-be-proved theorem
theorem ratio_longer_to_shorter_side (h : rect_sheet_fold l s) :
  l / s = Real.sqrt ((2 : ℝ) / (Real.sqrt 5 - 1)) :=
sorry

end ratio_longer_to_shorter_side_l177_177073


namespace find_constant_k_l177_177860

theorem find_constant_k (k : ℝ) :
  (-x^2 - (k + 9) * x - 8 = - (x - 2) * (x - 4)) → k = -15 :=
by 
  sorry

end find_constant_k_l177_177860


namespace balls_per_color_l177_177089

theorem balls_per_color (total_balls : ℕ) (total_colors : ℕ)
  (h1 : total_balls = 350) (h2 : total_colors = 10) : 
  total_balls / total_colors = 35 :=
by
  sorry

end balls_per_color_l177_177089


namespace valid_n_value_l177_177422

theorem valid_n_value (n : ℕ) (a : ℕ → ℕ)
    (h1 : ∀ k : ℕ, 1 ≤ k ∧ k < n → k ∣ a k)
    (h2 : ¬ n ∣ a n)
    (h3 : 2 ≤ n) :
    ∃ (p : ℕ) (α : ℕ), (Nat.Prime p) ∧ (n = p ^ α) ∧ (α ≥ 1) :=
by sorry

end valid_n_value_l177_177422


namespace pool_width_l177_177595

-- Define the given conditions
def hose_rate : ℝ := 60 -- cubic feet per minute
def drain_time : ℝ := 2000 -- minutes
def pool_length : ℝ := 150 -- feet
def pool_depth : ℝ := 10 -- feet

-- Calculate the total volume drained
def total_volume := hose_rate * drain_time -- cubic feet

-- Define a variable for the pool width
variable (W : ℝ)

-- The statement to prove
theorem pool_width :
  (total_volume = pool_length * W * pool_depth) → W = 80 :=
by
  sorry

end pool_width_l177_177595


namespace percentage_of_x_eq_y_l177_177342

theorem percentage_of_x_eq_y
  (x y : ℝ) 
  (h : 0.60 * (x - y) = 0.20 * (x + y)) :
  y = 0.50 * x := 
sorry

end percentage_of_x_eq_y_l177_177342


namespace calculate_drift_l177_177187

theorem calculate_drift (w v t : ℝ) (hw : w = 400) (hv : v = 10) (ht : t = 50) : v * t - w = 100 :=
by
  sorry

end calculate_drift_l177_177187


namespace initial_seashells_l177_177272

-- Definitions based on the problem conditions
def gave_to_joan : ℕ := 6
def left_with_jessica : ℕ := 2

-- Theorem statement to prove the number of seashells initially found by Jessica
theorem initial_seashells : gave_to_joan + left_with_jessica = 8 := by
  -- Proof goes here
  sorry

end initial_seashells_l177_177272


namespace maximum_sum_of_digits_difference_l177_177605

-- Definition of the sum of the digits of a number
-- For the purpose of this statement, we'll assume the existence of a function sum_of_digits

def sum_of_digits (n : ℕ) : ℕ :=
  sorry -- Assume the function is defined elsewhere

-- Statement of the problem
theorem maximum_sum_of_digits_difference :
  ∃ x : ℕ, sum_of_digits (x + 2019) - sum_of_digits x = 12 :=
sorry

end maximum_sum_of_digits_difference_l177_177605


namespace k_less_than_two_l177_177852

theorem k_less_than_two
    (x : ℝ)
    (k : ℝ)
    (y : ℝ)
    (h : y = (2 - k) / x)
    (h1 : (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) : k < 2 :=
by
  sorry

end k_less_than_two_l177_177852


namespace xy_yz_zx_value_l177_177472

namespace MathProof

theorem xy_yz_zx_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x^2 + x * y + y^2 = 147) 
  (h2 : y^2 + y * z + z^2 = 16) 
  (h3 : z^2 + x * z + x^2 = 163) :
  x * y + y * z + z * x = 56 := 
sorry      

end MathProof

end xy_yz_zx_value_l177_177472


namespace expand_and_simplify_l177_177784

theorem expand_and_simplify (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * (14 / x^3 + 15 * x - 6 * x^5) = (6 / x^3) + (45 * x / 7) - (18 * x^5 / 7) :=
by
  sorry

end expand_and_simplify_l177_177784


namespace parabola_hyperbola_focus_l177_177100

/-- 
Proof problem: If the focus of the parabola y^2 = 2px coincides with the right focus of the hyperbola x^2/3 - y^2/1 = 1, then p = 2.
-/
theorem parabola_hyperbola_focus (p : ℝ) :
    ∀ (focus_parabola : ℝ × ℝ) (focus_hyperbola : ℝ × ℝ),
      (focus_parabola = (p, 0)) →
      (focus_hyperbola = (2, 0)) →
      (focus_parabola = focus_hyperbola) →
        p = 2 :=
by
  intros focus_parabola focus_hyperbola h1 h2 h3
  sorry

end parabola_hyperbola_focus_l177_177100


namespace simplify_tan_expression_simplify_complex_expression_l177_177348

-- Problem 1
theorem simplify_tan_expression (α : ℝ) (hα : 0 < α ∧ α < 2 * π ∧ α > 3 / 2 * π) : 
  (Real.tan α + Real.sqrt ((1 / (Real.cos α)^2) - 1) + 2 * (Real.sin α)^2 + 2 * (Real.cos α)^2 = 2) :=
sorry

-- Problem 2
theorem simplify_complex_expression (α : ℝ) (hα : 0 < α ∧ α < 2 * π ∧ α > 3 / 2 * π) : 
  (Real.sin (α + π) * Real.tan (π - α) * Real.cos (2 * π - α) / (Real.sin (π - α) * Real.sin (π / 2 + α)) + Real.cos (5 * π / 2) = - Real.cos α) :=
sorry

end simplify_tan_expression_simplify_complex_expression_l177_177348


namespace factorization_correct_l177_177064

theorem factorization_correct (a : ℝ) : a^2 - 2 * a - 15 = (a + 3) * (a - 5) := 
by 
  sorry

end factorization_correct_l177_177064


namespace multiple_of_kids_finishing_early_l177_177158

-- Definitions based on conditions
def num_10_percent_kids (total_kids : ℕ) : ℕ := (total_kids * 10) / 100

def num_remaining_kids (total_kids kids_less_6 kids_more_14 : ℕ) : ℕ := total_kids - kids_less_6 - kids_more_14

def num_multiple_finishing_less_8 (total_kids : ℕ) (multiple : ℕ) : ℕ := multiple * num_10_percent_kids total_kids

-- Main theorem statement
theorem multiple_of_kids_finishing_early 
  (total_kids : ℕ)
  (h_total_kids : total_kids = 40)
  (kids_more_14 : ℕ)
  (h_kids_more_14 : kids_more_14 = 4)
  (h_1_6_remaining : kids_more_14 = num_remaining_kids total_kids (num_10_percent_kids total_kids) kids_more_14 / 6)
  : (num_multiple_finishing_less_8 total_kids 3) = (total_kids - num_10_percent_kids total_kids - kids_more_14) := 
by 
  sorry

end multiple_of_kids_finishing_early_l177_177158


namespace max_chocolates_l177_177714

theorem max_chocolates (b c k : ℕ) (h1 : b + c = 36) (h2 : c = k * b) (h3 : k > 0) : b ≤ 18 :=
sorry

end max_chocolates_l177_177714


namespace opposite_of_one_half_l177_177676

theorem opposite_of_one_half : -((1:ℚ)/2) = -1/2 := by
  -- Skipping the proof using sorry
  sorry

end opposite_of_one_half_l177_177676


namespace cats_after_purchasing_l177_177449

/-- Mrs. Sheridan's total number of cats after purchasing more -/
theorem cats_after_purchasing (a b : ℕ) (h₀ : a = 11) (h₁ : b = 43) : a + b = 54 := by
  sorry

end cats_after_purchasing_l177_177449


namespace xiao_zhang_complete_task_l177_177482

open Nat

def xiaoZhangCharacters (n : ℕ) : ℕ :=
match n with
| 0 => 0
| (n+1) => 2 * (xiaoZhangCharacters n)

theorem xiao_zhang_complete_task :
  ∀ (total_chars : ℕ), (total_chars > 0) → 
  (xiaoZhangCharacters 5 = (total_chars / 3)) →
  (xiaoZhangCharacters 6 = total_chars) :=
by
  sorry

end xiao_zhang_complete_task_l177_177482


namespace largest_number_is_C_l177_177294

theorem largest_number_is_C (A B C D E : ℝ) 
  (hA : A = 0.989) 
  (hB : B = 0.9098) 
  (hC : C = 0.9899) 
  (hD : D = 0.9009) 
  (hE : E = 0.9809) : 
  C > A ∧ C > B ∧ C > D ∧ C > E := 
by 
  sorry

end largest_number_is_C_l177_177294


namespace length_of_short_pieces_l177_177713

def total_length : ℕ := 27
def long_piece_length : ℕ := 4
def number_of_long_pieces : ℕ := total_length / long_piece_length
def remainder_length : ℕ := total_length % long_piece_length
def number_of_short_pieces : ℕ := 3

theorem length_of_short_pieces (h1 : remainder_length = 3) : (remainder_length / number_of_short_pieces) = 1 :=
by
  sorry

end length_of_short_pieces_l177_177713


namespace solve_for_a_l177_177232

theorem solve_for_a (a : ℚ) (h : a + a / 3 = 8 / 3) : a = 2 :=
sorry

end solve_for_a_l177_177232


namespace find_price_per_craft_l177_177907

-- Definitions based on conditions
def price_per_craft (x : ℝ) : Prop :=
  let crafts_sold := 3
  let extra_money := 7
  let deposit := 18
  let remaining_money := 25
  let total_before_deposit := 43
  3 * x + extra_money = total_before_deposit

-- Statement of the problem to prove x = 12 given conditions
theorem find_price_per_craft : ∃ x : ℝ, price_per_craft x ∧ x = 12 :=
by
  sorry

end find_price_per_craft_l177_177907


namespace ratio_of_sum_and_difference_l177_177321

theorem ratio_of_sum_and_difference (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : (x + y) / (x - y) = x / y) : x / y = 1 + Real.sqrt 2 :=
sorry

end ratio_of_sum_and_difference_l177_177321


namespace solution_set_of_inequality_l177_177980

theorem solution_set_of_inequality (a x : ℝ) (h1 : a < 2) (h2 : a * x > 2 * x + a - 2) : x < 1 :=
sorry

end solution_set_of_inequality_l177_177980


namespace car_payment_months_l177_177644

theorem car_payment_months 
    (total_price : ℕ) 
    (initial_payment : ℕ)
    (monthly_payment : ℕ) 
    (h_total_price : total_price = 13380) 
    (h_initial_payment : initial_payment = 5400) 
    (h_monthly_payment : monthly_payment = 420) 
    : total_price - initial_payment = 7980 
    ∧ (total_price - initial_payment) / monthly_payment = 19 := 
by 
  sorry

end car_payment_months_l177_177644


namespace ratio_S15_S5_l177_177076

-- Definition of a geometric sequence sum and the given ratio S10/S5 = 1/2
noncomputable def geom_sum : ℕ → ℕ := sorry
axiom ratio_S10_S5 : geom_sum 10 / geom_sum 5 = 1 / 2

-- The goal is to prove that the ratio S15/S5 = 3/4
theorem ratio_S15_S5 : geom_sum 15 / geom_sum 5 = 3 / 4 :=
by sorry

end ratio_S15_S5_l177_177076


namespace sam_new_crime_books_l177_177822

theorem sam_new_crime_books (used_adventure_books : ℝ) (used_mystery_books : ℝ) (total_books : ℝ) :
  used_adventure_books = 13.0 →
  used_mystery_books = 17.0 →
  total_books = 45.0 →
  total_books - (used_adventure_books + used_mystery_books) = 15.0 :=
by
  intros ha hm ht
  rw [ha, hm, ht]
  norm_num
  -- sorry

end sam_new_crime_books_l177_177822


namespace product_of_intersection_coordinates_l177_177156

noncomputable def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 5)^2 = 1
noncomputable def circle2 (x y : ℝ) : Prop := (x - 5)^2 + (y - 5)^2 = 4

theorem product_of_intersection_coordinates :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y ∧ x * y = 15 :=
by
  sorry

end product_of_intersection_coordinates_l177_177156


namespace probability_at_least_two_worth_visiting_l177_177856

theorem probability_at_least_two_worth_visiting :
  let total_caves := 8
  let worth_visiting := 3
  let select_caves := 4
  let worth_select_2 := Nat.choose worth_visiting 2 * Nat.choose (total_caves - worth_visiting) 2
  let worth_select_3 := Nat.choose worth_visiting 3 * Nat.choose (total_caves - worth_visiting) 1
  let total_select := Nat.choose total_caves select_caves
  let probability := (worth_select_2 + worth_select_3) / total_select
  probability = 1 / 2 := sorry

end probability_at_least_two_worth_visiting_l177_177856


namespace confetti_left_correct_l177_177986

-- Define the number of pieces of red and green confetti collected by Eunji
def red_confetti : ℕ := 1
def green_confetti : ℕ := 9

-- Define the total number of pieces of confetti collected by Eunji
def total_confetti : ℕ := red_confetti + green_confetti

-- Define the number of pieces of confetti given to Yuna
def given_to_Yuna : ℕ := 4

-- Define the number of pieces of confetti left with Eunji
def confetti_left : ℕ :=  red_confetti + green_confetti - given_to_Yuna

-- Goal to prove
theorem confetti_left_correct : confetti_left = 6 := by
  -- Here the steps proving the equality would go, but we add sorry to skip the proof
  sorry

end confetti_left_correct_l177_177986


namespace solve_for_y_l177_177375

theorem solve_for_y (y : ℝ) (h : y ≠ 2) :
  (7 * y / (y - 2) - 4 / (y - 2) = 3 / (y - 2)) → y = 1 :=
by
  intro h_eq
  sorry

end solve_for_y_l177_177375


namespace quadratic_graph_nature_l177_177945

theorem quadratic_graph_nature (a b : Real) (h : a ≠ 0) :
  ∀ (x : Real), (a * x^2 + b * x + (b^2 / (2 * a)) > 0) ∨ (a * x^2 + b * x + (b^2 / (2 * a)) < 0) :=
by
  sorry

end quadratic_graph_nature_l177_177945


namespace slope_of_line_l177_177902

theorem slope_of_line : ∀ (x y : ℝ), (x - y + 1 = 0) → (1 = 1) :=
by
  intros x y h
  sorry

end slope_of_line_l177_177902


namespace max_XYZ_plus_terms_l177_177844

theorem max_XYZ_plus_terms {X Y Z : ℕ} (h : X + Y + Z = 15) :
  X * Y * Z + X * Y + Y * Z + Z * X ≤ 200 :=
sorry

end max_XYZ_plus_terms_l177_177844


namespace intersection_A_B_l177_177691

def A := {x : ℝ | x^2 - x - 2 ≤ 0}
def B := {y : ℝ | ∃ x : ℝ, y = 2^x}

theorem intersection_A_B :
  A ∩ {x : ℝ | x > 0} = {x : ℝ | 0 < x ∧ x ≤ 2} :=
sorry

end intersection_A_B_l177_177691


namespace deposit_percentage_is_10_l177_177593

-- Define the deposit and remaining amount
def deposit := 120
def remaining := 1080

-- Define total cost
def total_cost := deposit + remaining

-- Define deposit percentage calculation
def deposit_percentage := (deposit / total_cost) * 100

-- Theorem to prove the deposit percentage is 10%
theorem deposit_percentage_is_10 : deposit_percentage = 10 := by
  -- Since deposit, remaining and total_cost are defined explicitly,
  -- the proof verification of final result is straightforward.
  sorry

end deposit_percentage_is_10_l177_177593


namespace henry_needs_30_dollars_l177_177147

def henry_action_figures_completion (current_figures total_figures cost_per_figure : ℕ) : ℕ :=
  (total_figures - current_figures) * cost_per_figure

theorem henry_needs_30_dollars : henry_action_figures_completion 3 8 6 = 30 := by
  sorry

end henry_needs_30_dollars_l177_177147


namespace combined_shoe_size_l177_177726

-- Definitions based on conditions
def Jasmine_size : ℕ := 7
def Alexa_size : ℕ := 2 * Jasmine_size
def Clara_size : ℕ := 3 * Jasmine_size

-- Statement to prove
theorem combined_shoe_size : Jasmine_size + Alexa_size + Clara_size = 42 :=
by
  sorry

end combined_shoe_size_l177_177726


namespace input_statement_is_INPUT_l177_177574

namespace ProgrammingStatements

-- Definitions of each type of statement
def PRINT_is_output : Prop := True
def INPUT_is_input : Prop := True
def THEN_is_conditional : Prop := True
def END_is_termination : Prop := True

-- The proof problem
theorem input_statement_is_INPUT :
  INPUT_is_input := by
  sorry

end ProgrammingStatements

end input_statement_is_INPUT_l177_177574


namespace Danny_finishes_first_l177_177591

-- Definitions based on the conditions
variables (E D F : ℝ)    -- Garden areas for Emily, Danny, Fiona
variables (e d f : ℝ)    -- Mowing rates for Emily, Danny, Fiona
variables (start_time : ℝ)

-- Condition definitions
def emily_garden_size := E = 3 * D
def emily_garden_size_fiona := E = 5 * F
def fiona_mower_speed_danny := f = (1/4) * d
def fiona_mower_speed_emily := f = (1/5) * e

-- Prove Danny finishes first
theorem Danny_finishes_first 
  (h1 : emily_garden_size E D)
  (h2 : emily_garden_size_fiona E F)
  (h3 : fiona_mower_speed_danny f d)
  (h4 : fiona_mower_speed_emily f e) : 
  (start_time ≤ (5/12) * (start_time + E/d) ∧ start_time ≤ (E/f)) -> (start_time + E/d < start_time + E/e) -> 
  true := 
sorry -- proof is omitted

end Danny_finishes_first_l177_177591


namespace range_of_x_l177_177219

theorem range_of_x (x : ℝ) : (x^2 - 9*x + 14 < 0) ∧ (2*x + 3 > 0) ↔ (2 < x) ∧ (x < 7) := 
by 
  sorry

end range_of_x_l177_177219


namespace greg_total_earnings_correct_l177_177350

def charge_per_dog := 20
def charge_per_minute := 1

def earnings_one_dog := charge_per_dog + charge_per_minute * 10
def earnings_two_dogs := 2 * (charge_per_dog + charge_per_minute * 7)
def earnings_three_dogs := 3 * (charge_per_dog + charge_per_minute * 9)

def total_earnings := earnings_one_dog + earnings_two_dogs + earnings_three_dogs

theorem greg_total_earnings_correct : total_earnings = 171 := by
  sorry

end greg_total_earnings_correct_l177_177350


namespace sat_marking_problem_l177_177978

-- Define the recurrence relation for the number of ways to mark questions without consecutive markings of the same letter.
def f : ℕ → ℕ
| 0     => 1
| 1     => 2
| 2     => 3
| (n+2) => f (n+1) + f n

-- Define that each letter marking can be done in 32 different ways.
def markWays : ℕ := 32

-- Define the number of questions to be 10.
def numQuestions : ℕ := 10

-- Calculate the number of sequences of length numQuestions with no consecutive same markings.
def numWays := f numQuestions

-- Prove that the number of ways results in 2^20 * 3^10 and compute 100m + n + p where m = 20, n = 10, p = 3.
theorem sat_marking_problem :
  (numWays ^ 5 = 2 ^ 20 * 3 ^ 10) ∧ (100 * 20 + 10 + 3 = 2013) :=
by
  sorry

end sat_marking_problem_l177_177978


namespace volume_of_prism_l177_177831

theorem volume_of_prism (a b c : ℝ) (h₁ : a * b = 48) (h₂ : b * c = 36) (h₃ : a * c = 50) : 
    (a * b * c = 170) :=
by
  sorry

end volume_of_prism_l177_177831


namespace test_completion_days_l177_177269

theorem test_completion_days :
  let barbara_days := 10
  let edward_days := 9
  let abhinav_days := 11
  let alex_days := 12
  let barbara_rate := 1 / barbara_days
  let edward_rate := 1 / edward_days
  let abhinav_rate := 1 / abhinav_days
  let alex_rate := 1 / alex_days
  let one_cycle_work := barbara_rate + edward_rate + abhinav_rate + alex_rate
  let cycles_needed := (1 : ℚ) / one_cycle_work
  Nat.ceil cycles_needed = 3 :=
by
  sorry

end test_completion_days_l177_177269


namespace new_cases_first_week_l177_177807

theorem new_cases_first_week
  (X : ℕ)
  (second_week_cases : X / 2 = X / 2)
  (third_week_cases : X / 2 + 2000 = (X / 2) + 2000)
  (total_cases : X + X / 2 + (X / 2 + 2000) = 9500) :
  X = 3750 := 
by sorry

end new_cases_first_week_l177_177807


namespace max_length_interval_l177_177454

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := ((m ^ 2 + m) * x - 1) / (m ^ 2 * x)

theorem max_length_interval (a b m : ℝ) (h1 : m ≠ 0) (h2 : ∀ x, f m x = x → x ∈ Set.Icc a b) :
  |b - a| = (2 * Real.sqrt 3) / 3 := sorry

end max_length_interval_l177_177454


namespace polynomial_roots_bc_product_l177_177061

theorem polynomial_roots_bc_product : ∃ (b c : ℤ), 
  (∀ x, (x^2 - 2*x - 1 = 0 → x^5 - b*x^3 - c*x^2 = 0)) ∧ (b * c = 348) := by 
  sorry

end polynomial_roots_bc_product_l177_177061


namespace find_series_sum_l177_177407

noncomputable def series_sum (s : ℝ) : ℝ := ∑' n : ℕ, (n+1) * s^(4*n + 3)

theorem find_series_sum (s : ℝ) (h : s^4 - s - 1/2 = 0) : series_sum s = -4 := by
  sorry

end find_series_sum_l177_177407


namespace lucas_initial_money_l177_177961

theorem lucas_initial_money : (3 * 2 + 14 = 20) := by sorry

end lucas_initial_money_l177_177961


namespace first_valve_time_l177_177151

noncomputable def first_valve_filling_time (V1 V2 : ℝ) (pool_capacity : ℝ) (combined_time : ℝ) : ℝ :=
  pool_capacity / V1

theorem first_valve_time :
  ∀ (V1 : ℝ) (V2 : ℝ) (pool_capacity : ℝ) (combined_time : ℝ),
    V2 = V1 + 50 →
    V1 + V2 = pool_capacity / combined_time →
    combined_time = 48 →
    pool_capacity = 12000 →
    first_valve_filling_time V1 V2 pool_capacity combined_time / 60 = 2 :=
  
by
  intros V1 V2 pool_capacity combined_time h1 h2 h3 h4
  sorry

end first_valve_time_l177_177151


namespace find_grape_juice_l177_177028

variables (milk water: ℝ) (limit total_before_test grapejuice: ℝ)

-- Conditions
def milk_amt: ℝ := 8
def water_amt: ℝ := 8
def limit_amt: ℝ := 32

-- The total liquid consumed before the test can be computed
def total_before_test_amt (milk water: ℝ) : ℝ := limit_amt - water_amt

-- The given total liquid consumed must be (milk + grape juice)
def total_consumed (milk grapejuice: ℝ) : ℝ := milk + grapejuice

theorem find_grape_juice :
    total_before_test_amt milk_amt water_amt = total_consumed milk_amt grapejuice →
    grapejuice = 16 :=
by
    unfold total_before_test_amt total_consumed
    sorry

end find_grape_juice_l177_177028


namespace distance_not_all_odd_l177_177928

theorem distance_not_all_odd (A B C D : ℝ × ℝ) : 
  ∃ (P Q : ℝ × ℝ), dist P Q % 2 = 0 := by sorry

end distance_not_all_odd_l177_177928


namespace kelly_needs_to_give_away_l177_177182

-- Definition of initial number of Sony games and desired number of Sony games left
def initial_sony_games : ℕ := 132
def desired_remaining_sony_games : ℕ := 31

-- The main theorem: The number of Sony games Kelly needs to give away to have 31 left
theorem kelly_needs_to_give_away : initial_sony_games - desired_remaining_sony_games = 101 := by
  sorry

end kelly_needs_to_give_away_l177_177182


namespace total_swordfish_caught_l177_177590

theorem total_swordfish_caught (fishing_trips : ℕ) (shelly_each_trip : ℕ) (sam_each_trip : ℕ) : 
  shelly_each_trip = 3 → 
  sam_each_trip = 2 → 
  fishing_trips = 5 → 
  (shelly_each_trip + sam_each_trip) * fishing_trips = 25 :=
by
  sorry

end total_swordfish_caught_l177_177590


namespace total_profit_is_64000_l177_177981

-- Definitions for investments and periods
variables (IB IA TB TA Profit_B Profit_A Total_Profit : ℕ)

-- Conditions from the problem
def condition1 := IA = 5 * IB
def condition2 := TA = 3 * TB
def condition3 := Profit_B = 4000
def condition4 := Profit_A / Profit_B = (IA * TA) / (IB * TB)

-- Target statement to be proved
theorem total_profit_is_64000 (IB IA TB TA Profit_B Profit_A Total_Profit : ℕ) :
  condition1 IA IB → condition2 TA TB → condition3 Profit_B → condition4 IA TA IB TB Profit_A Profit_B → 
  Total_Profit = Profit_A + Profit_B → Total_Profit = 64000 :=
by {
  sorry
}

end total_profit_is_64000_l177_177981


namespace relationship_a_b_c_l177_177285

theorem relationship_a_b_c (x y a b c : ℝ) (h1 : x + y = a)
  (h2 : x^2 + y^2 = b) (h3 : x^3 + y^3 = c) : a^3 - 3*a*b + 2*c = 0 := by
  sorry

end relationship_a_b_c_l177_177285


namespace exam_students_count_l177_177506

theorem exam_students_count (failed_students : ℕ) (failed_percentage : ℝ) (total_students : ℕ) 
    (h1 : failed_students = 260) 
    (h2 : failed_percentage = 0.65) 
    (h3 : (failed_percentage * total_students : ℝ) = (failed_students : ℝ)) : 
    total_students = 400 := 
by 
    sorry

end exam_students_count_l177_177506


namespace log2_125_eq_9y_l177_177912

theorem log2_125_eq_9y (y : ℝ) (h : Real.log 5 / Real.log 8 = y) : Real.log 125 / Real.log 2 = 9 * y :=
by
  sorry

end log2_125_eq_9y_l177_177912


namespace remaining_time_for_P_l177_177071

theorem remaining_time_for_P 
  (P_rate : ℝ) (Q_rate : ℝ) (together_time : ℝ) (remaining_time_minutes : ℝ)
  (hP_rate : P_rate = 1 / 3) 
  (hQ_rate : Q_rate = 1 / 18) 
  (h_together_time : together_time = 2) 
  (h_remaining_time_minutes : remaining_time_minutes = 40) :
  (((P_rate + Q_rate) * together_time) + P_rate * (remaining_time_minutes / 60)) = 1 :=
by  rw [hP_rate, hQ_rate, h_together_time, h_remaining_time_minutes]
    admit

end remaining_time_for_P_l177_177071


namespace lola_dora_allowance_l177_177365

variable (total_cost deck_cost sticker_cost sticker_count packs_each : ℕ)
variable (allowance : ℕ)

theorem lola_dora_allowance 
  (h1 : deck_cost = 10)
  (h2 : sticker_cost = 2)
  (h3 : packs_each = 2)
  (h4 : sticker_count = 2 * packs_each)
  (h5 : total_cost = deck_cost + sticker_count * sticker_cost)
  (h6 : total_cost = 18) :
  allowance = 9 :=
sorry

end lola_dora_allowance_l177_177365


namespace work_completion_time_l177_177971

theorem work_completion_time (A_works_in : ℕ) (A_works_days : ℕ) (B_works_remainder_in : ℕ) (total_days : ℕ) :
  (A_works_in = 60) → (A_works_days = 15) → (B_works_remainder_in = 30) → (total_days = 24) := 
by
  intros hA_work hA_days hB_work
  sorry

end work_completion_time_l177_177971


namespace power_function_passes_point_l177_177682

noncomputable def f (k α x : ℝ) : ℝ := k * x^α

theorem power_function_passes_point (k α : ℝ) (h1 : f k α (1/2) = (Real.sqrt 2)/2) : 
  k + α = 3/2 :=
sorry

end power_function_passes_point_l177_177682


namespace mean_of_three_is_90_l177_177117

-- Given conditions as Lean definitions
def mean_twelve (s : ℕ) : Prop := s = 12 * 40
def added_sum (x y z : ℕ) (s : ℕ) : Prop := s + x + y + z = 15 * 50
def z_value (x z : ℕ) : Prop := z = x + 10

-- Theorem statement to prove the mean of x, y, and z is 90
theorem mean_of_three_is_90 (x y z s : ℕ) : 
  (mean_twelve s) → (z_value x z) → (added_sum x y z s) → 
  (x + y + z) / 3 = 90 := 
by 
  intros h1 h2 h3 
  sorry

end mean_of_three_is_90_l177_177117


namespace first_term_geometric_sequence_l177_177111

variable {a : ℕ → ℝ} -- Define the geometric sequence a_n
variable (q : ℝ) -- Define the common ratio q which is a real number

-- Conditions given in the problem
def geom_seq_first_term (a : ℕ → ℝ) (q : ℝ) :=
  a 3 = 2 ∧ a 4 = 4 ∧ (∀ n : ℕ, a (n+1) = a n * q)

-- Assert that if these conditions hold, then the first term is 1/2
theorem first_term_geometric_sequence (hq : geom_seq_first_term a q) : a 1 = 1/2 :=
by
  sorry

end first_term_geometric_sequence_l177_177111


namespace negation_of_universal_statement_l177_177277

theorem negation_of_universal_statement :
  ¬(∀ a : ℝ, ∃ x : ℝ, x > 0 ∧ a * x^2 - 3 * x - a = 0) ↔ ∃ a : ℝ, ∀ x : ℝ, ¬(x > 0 ∧ a * x^2 - 3 * x - a = 0) :=
by sorry

end negation_of_universal_statement_l177_177277


namespace remainder_8_pow_2023_mod_5_l177_177810

theorem remainder_8_pow_2023_mod_5 :
  8 ^ 2023 % 5 = 2 :=
by
  sorry

end remainder_8_pow_2023_mod_5_l177_177810


namespace find_line_equation_of_ellipse_intersection_l177_177678

-- Defining the ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 2 = 1

-- Defining the line intersects points
def line_intersects (A B : ℝ × ℝ) : Prop := 
  ∃ x1 y1 x2 y2 : ℝ, A = (x1, y1) ∧ B = (x2, y2) ∧ 
  (ellipse x1 y1) ∧ (ellipse x2 y2) ∧ 
  ((x1 + x2) / 2 = 1 / 2) ∧ ((y1 + y2) / 2 = -1)

-- Statement to prove the equation of the line
theorem find_line_equation_of_ellipse_intersection (A B : ℝ × ℝ)
  (h : line_intersects A B) : 
  ∃ m b : ℝ, (∀ x y : ℝ, y = m * x + b ↔ x - 4*y - (9/2) = 0) :=
sorry

end find_line_equation_of_ellipse_intersection_l177_177678


namespace max_value_of_t_l177_177968

variable (n r t : ℕ)
variable (A : Finset (Finset (Fin n)))
variable (h₁ : n ≤ 2 * r)
variable (h₂ : ∀ s ∈ A, Finset.card s = r)
variable (h₃ : Finset.card A = t)

theorem max_value_of_t : 
  (n < 2 * r → t ≤ Nat.choose n r) ∧ 
  (n = 2 * r → t ≤ Nat.choose n r / 2) :=
by
  sorry

end max_value_of_t_l177_177968


namespace inequality_proof_l177_177749

theorem inequality_proof (x1 x2 y1 y2 z1 z2 : ℝ)
  (hx1 : x1 > 0) (hx2 : x2 > 0)
  (hx1y1 : x1 * y1 - z1^2 > 0) (hx2y2 : x2 * y2 - z2^2 > 0) :
  8 / ((x1 + x2) * (y1 + y2) - (z1 - z2)^2) ≤ 1 / (x1 * y1 - z1^2) + 1 / (x2 * y2 - z2^2) :=
sorry

end inequality_proof_l177_177749


namespace calculate_coeffs_l177_177935

noncomputable def quadratic_coeffs (p q : ℝ) : Prop :=
  if p = 1 then true else if p = -2 then q = -1 else false

theorem calculate_coeffs (p q : ℝ) :
    (∃ p q, (x^2 + p * x + q = 0) ∧ (x^2 - p^2 * x + p * q = 0)) →
    quadratic_coeffs p q :=
by sorry

end calculate_coeffs_l177_177935


namespace simplify_logarithmic_expression_l177_177340

theorem simplify_logarithmic_expression :
  (1 / (Real.logb 12 3 + 1) + 1 / (Real.logb 8 2 + 1) + 1 / (Real.logb 18 9 + 1) = 1) :=
sorry

end simplify_logarithmic_expression_l177_177340


namespace no_valid_sequence_of_integers_from_1_to_2004_l177_177589

theorem no_valid_sequence_of_integers_from_1_to_2004 :
  ¬ ∃ (a : ℕ → ℕ), 
    (∀ i, 1 ≤ a i ∧ a i ≤ 2004) ∧ 
    (∀ i j, i ≠ j → a i ≠ a j) ∧ 
    (∀ k, 1 ≤ k ∧ k + 9 ≤ 2004 → 
      (a k + a (k + 1) + a (k + 2) + a (k + 3) + a (k + 4) + a (k + 5) + 
       a (k + 6) + a (k + 7) + a (k + 8) + a (k + 9)) % 10 = 0) :=
  sorry

end no_valid_sequence_of_integers_from_1_to_2004_l177_177589


namespace min_cuts_for_payment_7_days_l177_177917

theorem min_cuts_for_payment_7_days (n : ℕ) (h : n = 7) : ∃ k, k = 1 :=
by sorry

end min_cuts_for_payment_7_days_l177_177917


namespace trivia_team_students_per_group_l177_177792

theorem trivia_team_students_per_group (total_students : ℕ) (not_picked : ℕ) (num_groups : ℕ) 
  (h1 : total_students = 58) (h2 : not_picked = 10) (h3 : num_groups = 8) :
  (total_students - not_picked) / num_groups = 6 :=
by
  sorry

end trivia_team_students_per_group_l177_177792


namespace coloring_circle_impossible_l177_177673

theorem coloring_circle_impossible (n : ℕ) (h : n = 2022) : 
  ¬ (∃ (coloring : ℕ → ℕ), (∀ i, 0 ≤ coloring i ∧ coloring i < 3) ∧ (∀ i, coloring ((i + 1) % n) ≠ coloring i)) :=
sorry

end coloring_circle_impossible_l177_177673


namespace rectangle_area_288_l177_177507

/-- A rectangle contains eight circles arranged in a 2x4 grid. Each circle has a radius of 3 inches.
    We are asked to prove that the area of the rectangle is 288 square inches. --/
noncomputable def circle_radius : ℝ := 3
noncomputable def circles_per_width : ℕ := 2
noncomputable def circles_per_length : ℕ := 4
noncomputable def circle_diameter : ℝ := 2 * circle_radius
noncomputable def rectangle_width : ℝ := circles_per_width * circle_diameter
noncomputable def rectangle_length : ℝ := circles_per_length * circle_diameter
noncomputable def rectangle_area : ℝ := rectangle_length * rectangle_width

theorem rectangle_area_288 :
  rectangle_area = 288 :=
by
  -- Proof of the area will be filled in here.
  sorry

end rectangle_area_288_l177_177507


namespace trapezoid_CD_length_l177_177957

theorem trapezoid_CD_length (AB CD AD BC : ℝ) (P : ℝ) 
  (h₁ : AB = 12) 
  (h₂ : AD = 5) 
  (h₃ : BC = 7) 
  (h₄ : P = 40) : CD = 16 :=
by
  sorry

end trapezoid_CD_length_l177_177957


namespace MrsHiltRows_l177_177059

theorem MrsHiltRows :
  let (a : ℕ) := 16
  let (b : ℕ) := 14
  let (r : ℕ) := 5
  (a + b) / r = 6 := by
  sorry

end MrsHiltRows_l177_177059


namespace shaded_area_l177_177328

theorem shaded_area (r : ℝ) (α : ℝ) (β : ℝ) (h1 : r = 4) (h2 : α = 1/2) :
  β = 64 - 16 * Real.pi := by sorry

end shaded_area_l177_177328


namespace sum_of_ages_now_l177_177185

variable (D A Al B : ℝ)

noncomputable def age_condition (D : ℝ) : Prop :=
  D = 16

noncomputable def alex_age_condition (A : ℝ) : Prop :=
  A = 60 - (30 - 16)

noncomputable def allison_age_condition (Al : ℝ) : Prop :=
  Al = 15 - (30 - 16)

noncomputable def bernard_age_condition (B A Al : ℝ) : Prop :=
  B = (A + Al) / 2

noncomputable def sum_of_ages (A Al B : ℝ) : ℝ :=
  A + Al + B

theorem sum_of_ages_now :
  age_condition D →
  alex_age_condition A →
  allison_age_condition Al →
  bernard_age_condition B A Al →
  sum_of_ages A Al B = 70.5 := by
  sorry

end sum_of_ages_now_l177_177185


namespace Joe_team_wins_eq_1_l177_177330

-- Definition for the points a team gets for winning a game.
def points_per_win := 3
-- Definition for the points a team gets for a tie game.
def points_per_tie := 1

-- Given conditions
def Joe_team_draws := 3
def first_place_wins := 2
def first_place_ties := 2
def points_difference := 2

def first_place_points := (first_place_wins * points_per_win) + (first_place_ties * points_per_tie)

def Joe_team_total_points := first_place_points - points_difference
def Joe_team_points_from_ties := Joe_team_draws * points_per_tie
def Joe_team_points_from_wins := Joe_team_total_points - Joe_team_points_from_ties

-- To prove: number of games Joe's team won
theorem Joe_team_wins_eq_1 : (Joe_team_points_from_wins / points_per_win) = 1 :=
by
  sorry

end Joe_team_wins_eq_1_l177_177330


namespace rectangle_length_l177_177838

theorem rectangle_length (L W : ℝ) 
  (h1 : L + W = 23) 
  (h2 : L^2 + W^2 = 289) : 
  L = 15 :=
by 
  sorry

end rectangle_length_l177_177838


namespace express_in_scientific_notation_l177_177610

theorem express_in_scientific_notation :
  102200 = 1.022 * 10^5 :=
sorry

end express_in_scientific_notation_l177_177610


namespace unique_two_digit_u_l177_177888

theorem unique_two_digit_u:
  ∃! u : ℤ, 10 ≤ u ∧ u < 100 ∧ 
            (15 * u) % 100 = 45 ∧ 
            u % 17 = 7 :=
by
  -- To be completed in proof
  sorry

end unique_two_digit_u_l177_177888


namespace carlos_and_dana_rest_days_l177_177742

structure Schedule where
  days_of_cycle : ℕ
  work_days : ℕ
  rest_days : ℕ

def carlos : Schedule := ⟨7, 5, 2⟩
def dana : Schedule := ⟨13, 9, 4⟩

def days_both_rest (days_count : ℕ) (sched1 sched2 : Schedule) : ℕ :=
  let lcm_cycle := Nat.lcm sched1.days_of_cycle sched2.days_of_cycle
  let coincidences_in_cycle := 2  -- As derived from the solution
  let full_cycles := days_count / lcm_cycle
  coincidences_in_cycle * full_cycles

theorem carlos_and_dana_rest_days :
  days_both_rest 1500 carlos dana = 32 := by
  sorry

end carlos_and_dana_rest_days_l177_177742


namespace aaron_total_amount_owed_l177_177444

def total_cost (monthly_payment : ℤ) (months : ℤ) : ℤ :=
  monthly_payment * months

def interest_fee (amount : ℤ) (rate : ℤ) : ℤ :=
  amount * rate / 100

def total_amount_owed (monthly_payment : ℤ) (months : ℤ) (rate : ℤ) : ℤ :=
  let amount := total_cost monthly_payment months
  let fee := interest_fee amount rate
  amount + fee

theorem aaron_total_amount_owed :
  total_amount_owed 100 12 10 = 1320 :=
by
  sorry

end aaron_total_amount_owed_l177_177444


namespace cristina_pace_is_4_l177_177477

-- Definitions of the conditions
def head_start : ℝ := 36
def nicky_pace : ℝ := 3
def time : ℝ := 36

-- Definition of the distance Nicky runs
def distance_nicky_runs : ℝ := nicky_pace * time

-- Definition of the total distance Cristina ran to catch up
def distance_cristina_runs : ℝ := distance_nicky_runs + head_start

-- Lean 4 theorem statement to prove Cristina's pace
theorem cristina_pace_is_4 :
  (distance_cristina_runs / time) = 4 := 
by sorry

end cristina_pace_is_4_l177_177477


namespace christina_age_half_in_five_years_l177_177396

theorem christina_age_half_in_five_years (C Y : ℕ) 
  (h1 : C + 5 = Y / 2)
  (h2 : 21 = 3 * C / 5) :
  Y = 80 :=
sorry

end christina_age_half_in_five_years_l177_177396


namespace distance_against_current_l177_177581

theorem distance_against_current (V_b V_c : ℝ) (h1 : V_b + V_c = 2) (h2 : V_b = 1.5) : 
  (V_b - V_c) * 3 = 3 := by
  sorry

end distance_against_current_l177_177581


namespace scrap_cookie_radius_l177_177412

theorem scrap_cookie_radius (r: ℝ) (r_cookies: ℝ) (A_scrap: ℝ) (r_large: ℝ) (A_large: ℝ) (A_total_small: ℝ):
  r_cookies = 1.5 ∧
  r_large = r_cookies + 2 * r_cookies ∧
  A_large = π * r_large^2 ∧
  A_total_small = 8 * (π * r_cookies^2) ∧
  A_scrap = A_large - A_total_small ∧
  A_scrap = π * r^2
  → r = r_cookies
  :=
by
  intro h
  rcases h with ⟨hcookies, hrlarge, halarge, hatotalsmall, hascrap, hpi⟩
  sorry

end scrap_cookie_radius_l177_177412


namespace chemistry_textbook_weight_l177_177287

theorem chemistry_textbook_weight (G C : ℝ) 
  (h1 : G = 0.625) 
  (h2 : C = G + 6.5) : 
  C = 7.125 := 
by 
  sorry

end chemistry_textbook_weight_l177_177287


namespace exists_infinite_n_ωn_less_ωn_add1_less_ωn_add2_l177_177629

def omega (n : Nat) : Nat :=
  if n = 1 then 0 else n.factors.toFinset.card

theorem exists_infinite_n_ωn_less_ωn_add1_less_ωn_add2 :
  ∃ᶠ n in atTop, ∃ k : Nat, n = 2^k ∧
    omega n < omega (n + 1) ∧
    omega (n + 1) < omega (n + 2) :=
sorry

end exists_infinite_n_ωn_less_ωn_add1_less_ωn_add2_l177_177629


namespace selection_methods_correct_l177_177167

-- Define the number of students in each year
def first_year_students : ℕ := 3
def second_year_students : ℕ := 5
def third_year_students : ℕ := 4

-- Define the total number of different selection methods
def total_selection_methods : ℕ := first_year_students + second_year_students + third_year_students

-- Lean statement to prove the question is equivalent to the answer
theorem selection_methods_correct :
  total_selection_methods = 12 := by
  sorry

end selection_methods_correct_l177_177167


namespace Rahul_savings_l177_177457

variable (total_savings ppf_savings nsc_savings x : ℝ)

theorem Rahul_savings
  (h1 : total_savings = 180000)
  (h2 : ppf_savings = 72000)
  (h3 : nsc_savings = total_savings - ppf_savings)
  (h4 : x * nsc_savings = 0.5 * ppf_savings) :
  x = 1 / 3 :=
by
  -- Proof goes here
  sorry

end Rahul_savings_l177_177457


namespace simplify_fraction_l177_177579

theorem simplify_fraction : (2^5 + 2^3) / (2^4 - 2^2) = 10 / 3 := 
by 
  sorry

end simplify_fraction_l177_177579


namespace other_root_l177_177723

-- Define the condition that one root of the quadratic equation is -3
def is_root (a b c : ℤ) (x : ℚ) : Prop := a * x^2 + b * x + c = 0

-- Define the quadratic equation 7x^2 + mx - 6 = 0
def quadratic_eq (m : ℤ) (x : ℚ) : Prop := is_root 7 m (-6) x

-- Prove that the other root is 2/7 given that one root is -3
theorem other_root (m : ℤ) (h : quadratic_eq m (-3)) : quadratic_eq m (2 / 7) :=
by
  sorry

end other_root_l177_177723


namespace cubic_roots_a_b_third_root_l177_177252

theorem cubic_roots_a_b_third_root (a b : ℝ) :
  (∀ x, x^3 + a * x^2 + b * x + 6 = 0 → (x = 2 ∨ x = 3 ∨ x = -1)) →
  a = -4 ∧ b = 1 :=
by
  intro h
  -- We're skipping the proof steps and focusing on definite the goal
  sorry

end cubic_roots_a_b_third_root_l177_177252


namespace sufficient_but_not_necessary_l177_177729

def quadratic_real_roots (a : ℝ) : Prop :=
  (∃ x : ℝ, x^2 - 2 * x + a = 0)

theorem sufficient_but_not_necessary (a : ℝ) :
  (quadratic_real_roots 1) ∧ (∀ a > 1, ¬ quadratic_real_roots a) :=
sorry

end sufficient_but_not_necessary_l177_177729


namespace Batman_game_cost_l177_177776

theorem Batman_game_cost (football_cost strategy_cost total_spent batman_cost : ℝ)
  (h₁ : football_cost = 14.02)
  (h₂ : strategy_cost = 9.46)
  (h₃ : total_spent = 35.52)
  (h₄ : total_spent = football_cost + strategy_cost + batman_cost) :
  batman_cost = 12.04 := by
  sorry

end Batman_game_cost_l177_177776


namespace parallelogram_sticks_l177_177663

theorem parallelogram_sticks (a : ℕ) (h₁ : ∃ l₁ l₂, l₁ = 5 ∧ l₂ = 5 ∧ 
                                (l₁ = l₂) ∧ (a = 7)) : a = 7 :=
by sorry

end parallelogram_sticks_l177_177663


namespace extra_people_got_on_the_train_l177_177572

-- Definitions corresponding to the conditions
def initial_people_on_train : ℕ := 78
def people_got_off : ℕ := 27
def current_people_on_train : ℕ := 63

-- The mathematical equivalent proof problem
theorem extra_people_got_on_the_train :
  (initial_people_on_train - people_got_off + extra_people = current_people_on_train) → (extra_people = 12) :=
by
  sorry

end extra_people_got_on_the_train_l177_177572


namespace geometric_sequence_nec_not_suff_l177_177550

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n ≠ 0 → (a (n + 1) / a n) = (a (n + 2) / a (n + 1))

noncomputable def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 3) = a (n + 1) * a (n + 2)

theorem geometric_sequence_nec_not_suff (a : ℕ → ℝ) (hn : ∀ n : ℕ, a n ≠ 0) : 
  (is_geometric_sequence a → satisfies_condition a) ∧ ¬(satisfies_condition a → is_geometric_sequence a) :=
by
  sorry

end geometric_sequence_nec_not_suff_l177_177550


namespace infinitude_of_composite_z_l177_177519

theorem infinitude_of_composite_z (a : ℕ) (h : ∃ k : ℕ, k > 1 ∧ a = 4 * k^4) : 
  ∀ n : ℕ, ¬ Prime (n^4 + a) :=
by sorry

end infinitude_of_composite_z_l177_177519


namespace largest_sum_of_digits_l177_177485

theorem largest_sum_of_digits (a b c : ℕ) (y : ℕ) (h1: a < 10) (h2: b < 10) (h3: c < 10) (h4: 0 < y ∧ y ≤ 12) (h5: 1000 * y = abc) :
  a + b + c = 8 := by
  sorry

end largest_sum_of_digits_l177_177485


namespace simplified_expression_l177_177958

theorem simplified_expression (x : ℝ) : 
  x * (3 * x^2 - 2) - 5 * (x^2 - 2 * x + 7) = 3 * x^3 - 5 * x^2 + 8 * x - 35 := 
by
  sorry

end simplified_expression_l177_177958


namespace round_robin_pairing_possible_l177_177018

def players : Set String := {"A", "B", "C", "D", "E", "F"}

def is_pairing (pairs : List (String × String)) : Prop :=
  ∀ (p : String × String), p ∈ pairs → p.1 ≠ p.2 ∧ p.1 ∈ players ∧ p.2 ∈ players

def unique_pairs (rounds : List (List (String × String))) : Prop :=
  ∀ r, r ∈ rounds → is_pairing r ∧ (∀ p1 p2, p1 ∈ r → p2 ∈ r → p1 ≠ p2 → p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2)

def all_players_paired (rounds : List (List (String × String))) : Prop :=
  ∀ p, p ∈ players →
  (∀ q, q ∈ players → p ≠ q → 
    (∃ r, r ∈ rounds ∧ (p,q) ∈ r ∨ (q,p) ∈ r))

theorem round_robin_pairing_possible : 
  ∃ rounds, List.length rounds = 5 ∧ unique_pairs rounds ∧ all_players_paired rounds :=
  sorry

end round_robin_pairing_possible_l177_177018


namespace custom_op_3_7_l177_177398

-- Define the custom operation (a # b)
def custom_op (a b : ℕ) : ℕ := a * b - b + b^2

-- State the theorem that proves the result
theorem custom_op_3_7 : custom_op 3 7 = 63 := by
  sorry

end custom_op_3_7_l177_177398


namespace sin_cos_plus_one_l177_177534

theorem sin_cos_plus_one (x : ℝ) (h : Real.tan x = 1 / 3) : Real.sin x * Real.cos x + 1 = 13 / 10 :=
by
  sorry

end sin_cos_plus_one_l177_177534


namespace neg_one_power_zero_l177_177359

theorem neg_one_power_zero : (-1: ℤ)^0 = 1 := 
sorry

end neg_one_power_zero_l177_177359


namespace all_metals_conduct_electricity_l177_177864

def Gold_conducts : Prop := sorry
def Silver_conducts : Prop := sorry
def Copper_conducts : Prop := sorry
def Iron_conducts : Prop := sorry
def inductive_reasoning : Prop := sorry

theorem all_metals_conduct_electricity (g: Gold_conducts) (s: Silver_conducts) (c: Copper_conducts) (i: Iron_conducts) : inductive_reasoning := 
sorry

end all_metals_conduct_electricity_l177_177864


namespace observation_count_l177_177865

theorem observation_count (n : ℤ) (mean_initial : ℝ) (erroneous_value correct_value : ℝ) (mean_corrected : ℝ) :
  mean_initial = 36 →
  erroneous_value = 20 →
  correct_value = 34 →
  mean_corrected = 36.45 →
  n ≥ 0 →
  ∃ n : ℤ, (n * mean_initial + (correct_value - erroneous_value) = n * mean_corrected) ∧ (n = 31) :=
by
  intros h1 h2 h3 h4 h5
  use 31
  sorry

end observation_count_l177_177865


namespace circle_range_of_a_l177_177024

theorem circle_range_of_a (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2 * a * x - 4 * y + (a^2 + a) = 0 → (x - h)^2 + (y - k)^2 = r^2) ↔ (a < 4) :=
sorry

end circle_range_of_a_l177_177024


namespace profit_growth_rate_and_expected_profit_l177_177694

theorem profit_growth_rate_and_expected_profit
  (profit_April : ℕ)
  (profit_June : ℕ)
  (months : ℕ)
  (avg_growth_rate : ℝ)
  (profit_July : ℕ) :
  profit_April = 6000 ∧ profit_June = 7260 ∧ months = 2 ∧ 
  (profit_April : ℝ) * (1 + avg_growth_rate)^months = profit_June →
  avg_growth_rate = 0.1 ∧ 
  (profit_June : ℝ) * (1 + avg_growth_rate) = profit_July →
  profit_July = 7986 := 
sorry

end profit_growth_rate_and_expected_profit_l177_177694


namespace a_five_minus_a_divisible_by_five_l177_177649

theorem a_five_minus_a_divisible_by_five (a : ℤ) : 5 ∣ (a^5 - a) :=
by
  -- proof steps
  sorry

end a_five_minus_a_divisible_by_five_l177_177649


namespace sandy_saved_last_year_percentage_l177_177974

theorem sandy_saved_last_year_percentage (S : ℝ) (P : ℝ) :
  (this_year_salary: ℝ) → (this_year_savings: ℝ) → 
  (this_year_saved_percentage: ℝ) → (saved_last_year_percentage: ℝ) → 
  this_year_salary = 1.1 * S → 
  this_year_saved_percentage = 6 →
  this_year_savings = (this_year_saved_percentage / 100) * this_year_salary →
  (this_year_savings / ((P / 100) * S)) = 0.66 →
  P = 10 :=
by
  -- The proof is to be filled in here.
  sorry

end sandy_saved_last_year_percentage_l177_177974


namespace real_solutions_l177_177248

-- Given the condition (equation)
def quadratic_equation (x y : ℝ) : Prop :=
  x^2 + 2 * x * Real.sin (x * y) + 1 = 0

-- The main theorem statement proving the solutions for x and y
theorem real_solutions (x y : ℝ) (k : ℤ) :
  quadratic_equation x y ↔
  (x = 1 ∧ (y = (Real.pi / 2 + 2 * k * Real.pi) ∨ y = (-Real.pi / 2 + 2 * k * Real.pi))) ∨
  (x = -1 ∧ (y = (-Real.pi / 2 + 2 * k * Real.pi) ∨ y = (Real.pi / 2 + 2 * k * Real.pi))) :=
by
  sorry

end real_solutions_l177_177248


namespace correct_option_l177_177939

-- Define the variable 'a' as a real number
variable (a : ℝ)

-- Define propositions for each option
def option_A : Prop := 5 * a ^ 2 - 4 * a ^ 2 = 1
def option_B : Prop := (a ^ 7) / (a ^ 4) = a ^ 3
def option_C : Prop := (a ^ 3) ^ 2 = a ^ 5
def option_D : Prop := a ^ 2 * a ^ 3 = a ^ 6

-- State the main proposition asserting that option B is correct and others are incorrect
theorem correct_option :
  option_B a ∧ ¬option_A a ∧ ¬option_C a ∧ ¬option_D a :=
  by sorry

end correct_option_l177_177939


namespace cubes_not_arithmetic_progression_l177_177630

theorem cubes_not_arithmetic_progression (x y z : ℤ) (h1 : y = (x + z) / 2) (h2 : x ≠ y) (h3 : y ≠ z) : x^3 + z^3 ≠ 2 * y^3 :=
by
  sorry

end cubes_not_arithmetic_progression_l177_177630


namespace problem1_problem2_l177_177783

noncomputable def f (x a b c : ℝ) : ℝ := abs (x + a) + abs (x - b) + c

theorem problem1 (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c)
  (h₃ : ∃ x, f x a b c = 4) : a + b + c = 4 :=
sorry

theorem problem2 (a b c : ℝ) (h : a + b + c = 4) : (1 / a) + (1 / b) + (1 / c) ≥ 9 / 4 :=
sorry

end problem1_problem2_l177_177783


namespace smallest_sum_of_consecutive_primes_divisible_by_5_l177_177764

def consecutive_primes (n : Nat) : Prop :=
  -- Define what it means to be 4 consecutive prime numbers
  Nat.Prime n ∧ Nat.Prime (n + 2) ∧ Nat.Prime (n + 6) ∧ Nat.Prime (n + 8)

def sum_of_consecutive_primes (n : Nat) : Nat :=
  n + (n + 2) + (n + 6) + (n + 8)

theorem smallest_sum_of_consecutive_primes_divisible_by_5 :
  ∃ n, n > 10 ∧ consecutive_primes n ∧ sum_of_consecutive_primes n % 5 = 0 ∧ sum_of_consecutive_primes n = 60 :=
by
  sorry

end smallest_sum_of_consecutive_primes_divisible_by_5_l177_177764


namespace log_max_reciprocal_min_l177_177816

open Real

-- Definitions for the conditions
variables (x y : ℝ)
def conditions (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ 2 * x + 5 * y = 20

-- Theorem statement for the first question
theorem log_max (x y : ℝ) (h : conditions x y) : log x + log y ≤ 1 :=
sorry

-- Theorem statement for the second question
theorem reciprocal_min (x y : ℝ) (h : conditions x y) : (1 / x) + (1 / y) ≥ (7 + 2 * sqrt 10) / 20 :=
sorry

end log_max_reciprocal_min_l177_177816


namespace exponent_problem_l177_177540

theorem exponent_problem : (5 ^ 6 * 5 ^ 9 * 5) / 5 ^ 3 = 5 ^ 13 := 
by
  sorry

end exponent_problem_l177_177540


namespace geometric_sequence_and_general_formula_l177_177846

theorem geometric_sequence_and_general_formula (a : ℕ → ℝ) (h : ∀ n, a (n+1) = (2/3) * a n + 2) (ha1 : a 1 = 7) : 
  ∃ r : ℝ, ∀ n, a n = r ^ (n-1) + 6 :=
sorry

end geometric_sequence_and_general_formula_l177_177846


namespace find_principal_l177_177336

-- Conditions as definitions
def amount : ℝ := 1120
def rate : ℝ := 0.05
def time : ℝ := 2

-- Required to add noncomputable due to the use of division and real numbers
noncomputable def principal : ℝ := amount / (1 + rate * time)

-- The main theorem statement which needs to be proved
theorem find_principal :
  principal = 1018.18 :=
sorry  -- Proof is not required; it is left as sorry

end find_principal_l177_177336


namespace isosceles_triangle_perimeter_l177_177233

/-- 
Prove that the perimeter of an isosceles triangle with sides 6 cm and 8 cm, 
and an area of 12 cm², is 20 cm.
--/
theorem isosceles_triangle_perimeter (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (S : ℝ) (h3 : S = 12) :
  a ≠ b →
  a = c ∨ b = c →
  ∃ P : ℝ, P = 20 := sorry

end isosceles_triangle_perimeter_l177_177233


namespace correct_subtraction_result_l177_177000

-- Definition of numbers:
def tens_digit := 2
def ones_digit := 4
def correct_number := 10 * tens_digit + ones_digit
def incorrect_number := 59
def incorrect_result := 14
def Z := incorrect_result + incorrect_number

-- Statement of the theorem
theorem correct_subtraction_result : Z - correct_number = 49 :=
by
  sorry

end correct_subtraction_result_l177_177000


namespace area_of_isosceles_triangle_l177_177787

open Real

theorem area_of_isosceles_triangle 
  (PQ PR QR : ℝ) (PQ_eq_PR : PQ = PR) (PQ_val : PQ = 13) (QR_val : QR = 10) : 
  1 / 2 * QR * sqrt (PQ^2 - (QR / 2)^2) = 60 := 
by 
sorry

end area_of_isosceles_triangle_l177_177787


namespace alternative_plan_cost_is_eleven_l177_177873

-- Defining current cost
def current_cost : ℕ := 12

-- Defining the alternative plan cost in terms of current cost
def alternative_cost : ℕ := current_cost - 1

-- Theorem stating the alternative cost is $11
theorem alternative_plan_cost_is_eleven : alternative_cost = 11 :=
by
  -- This is the proof, which we are skipping with sorry
  sorry

end alternative_plan_cost_is_eleven_l177_177873


namespace problem_remainder_P2017_mod_1000_l177_177246

def P (x : ℤ) : ℤ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

theorem problem_remainder_P2017_mod_1000 :
  (P 2017) % 1000 = 167 :=
by
  -- this proof examines \( P(2017) \) modulo 1000
  sorry

end problem_remainder_P2017_mod_1000_l177_177246


namespace food_consumption_reduction_l177_177199

noncomputable def reduction_factor (n p : ℝ) : ℝ :=
  (n * p) / ((n - 0.05 * n) * (p + 0.2 * p))

theorem food_consumption_reduction (n p : ℝ) (h : n > 0 ∧ p > 0) :
  (1 - reduction_factor n p) * 100 = 12.28 := by
  sorry

end food_consumption_reduction_l177_177199


namespace sequence_increasing_and_bounded_sequence_decreasing_and_bounded_l177_177108

def recurrence_relation (x : ℕ → ℝ) (n : ℕ) : Prop :=
  x (n + 1) = (2 * x n ^ 2 - x n) / (3 * (x n - 2))

-- For the first problem
theorem sequence_increasing_and_bounded (x : ℕ → ℝ) (h_rec : ∀ n, recurrence_relation x n)
  (h_initial : 4 < x 0 ∧ x 0 < 5) : ∀ n, x n < x (n + 1) ∧ x (n + 1) < 5 :=
by
  sorry

-- For the second problem
theorem sequence_decreasing_and_bounded (x : ℕ → ℝ) (h_rec : ∀ n, recurrence_relation x n)
  (h_initial : x 0 > 5) : ∀ n, 5 < x (n + 1) ∧ x (n + 1) < x n :=
by
  sorry

end sequence_increasing_and_bounded_sequence_decreasing_and_bounded_l177_177108


namespace largest_common_divisor_of_product_l177_177953

theorem largest_common_divisor_of_product (n : ℕ) (h_even : n % 2 = 0) (h_pos : 0 < n) :
  ∃ d : ℕ, d = 105 ∧ ∀ k : ℕ, k = (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) → d ∣ k :=
by
  sorry

end largest_common_divisor_of_product_l177_177953


namespace calories_consumed_l177_177213

-- Define the conditions
def pages_written : ℕ := 12
def pages_per_donut : ℕ := 2
def calories_per_donut : ℕ := 150

-- Define the theorem to be proved
theorem calories_consumed (pages_written : ℕ) (pages_per_donut : ℕ) (calories_per_donut : ℕ) : ℕ :=
  (pages_written / pages_per_donut) * calories_per_donut

-- Ensure the theorem corresponds to the correct answer
example : calories_consumed 12 2 150 = 900 := by
  sorry

end calories_consumed_l177_177213


namespace ab_condition_l177_177249

theorem ab_condition (a b : ℝ) : ¬((a + b > 1 → a^2 + b^2 > 1) ∧ (a^2 + b^2 > 1 → a + b > 1)) :=
by {
  -- This proof problem states that the condition "a + b > 1" is neither sufficient nor necessary for "a^2 + b^2 > 1".
  sorry
}

end ab_condition_l177_177249


namespace star_angle_sum_l177_177710

-- Define variables and angles for Petya's and Vasya's stars.
variables {α β γ δ ε : ℝ}
variables {φ χ ψ ω : ℝ}
variables {a b c d e : ℝ}

-- Conditions
def all_acute (a b c d e : ℝ) : Prop := a < 90 ∧ b < 90 ∧ c < 90 ∧ d < 90 ∧ e < 90
def one_obtuse (a b c d e : ℝ) : Prop := (a > 90 ∨ b > 90 ∨ c > 90 ∨ d > 90 ∨ e > 90)

-- Question: Prove the sum of the angles at the vertices of both stars is equal
theorem star_angle_sum : all_acute α β γ δ ε → one_obtuse φ χ ψ ω α → 
  α + β + γ + δ + ε = φ + χ + ψ + ω + α := 
by sorry

end star_angle_sum_l177_177710


namespace original_price_of_article_l177_177161

theorem original_price_of_article (P : ℝ) : 
  (P - 0.30 * P) * (1 - 0.20) = 1120 → P = 2000 :=
by
  intro h
  -- h represents the given condition for the problem
  sorry  -- proof will go here

end original_price_of_article_l177_177161


namespace number_of_terms_arithmetic_sequence_l177_177669

theorem number_of_terms_arithmetic_sequence
  (a₁ d n : ℝ)
  (h1 : a₁ + (a₁ + d) + (a₁ + 2 * d) = 34)
  (h2 : (a₁ + (n-3) * d) + (a₁ + (n-2) * d) + (a₁ + (n-1) * d) = 146)
  (h3 : n / 2 * (2 * a₁ + (n-1) * d) = 390) :
  n = 11 :=
by sorry

end number_of_terms_arithmetic_sequence_l177_177669


namespace geometric_sequence_a11_l177_177956

theorem geometric_sequence_a11
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h3 : a 3 = 4)
  (h7 : a 7 = 12) : 
  a 11 = 36 :=
by
  sorry

end geometric_sequence_a11_l177_177956


namespace sum_of_solutions_l177_177690

theorem sum_of_solutions (x : ℝ) :
  (4 * x + 6) * (3 * x - 12) = 0 → (x = -3 / 2 ∨ x = 4) →
  (-3 / 2 + 4) = 5 / 2 :=
by
  intros Hsol Hsols
  sorry

end sum_of_solutions_l177_177690


namespace base_eight_conversion_l177_177748

theorem base_eight_conversion :
  (1 * 8^2 + 3 * 8^1 + 2 * 8^0 = 90) := by
  sorry

end base_eight_conversion_l177_177748


namespace probability_A_score_not_less_than_135_l177_177841

/-- A certain school organized a competition with the following conditions:
  - The test has 25 multiple-choice questions, each with 4 options.
  - Each correct answer scores 6 points, each unanswered question scores 2 points, and each wrong answer scores 0 points.
  - Both candidates answered the first 20 questions correctly.
  - Candidate A will attempt only the last 3 questions, and for each, A can eliminate 1 wrong option,
    hence the probability of answering any one question correctly is 1/3.
  - A gives up the last 2 questions.
  - Prove that the probability that A's total score is not less than 135 points is equal to 7/27.
-/
theorem probability_A_score_not_less_than_135 :
  let prob_success := 1 / 3
  let prob_2_successes := (3 * (prob_success^2) * (2/3))
  let prob_3_successes := (prob_success^3)
  prob_2_successes + prob_3_successes = 7 / 27 := 
by
  sorry

end probability_A_score_not_less_than_135_l177_177841


namespace negation_of_P_l177_177120

variable (x : ℝ)

def P : Prop := ∀ x : ℝ, x^2 + 2*x + 3 ≥ 0

theorem negation_of_P : ¬P ↔ ∃ x : ℝ, x^2 + 2*x + 3 < 0 :=
by sorry

end negation_of_P_l177_177120


namespace range_of_m_l177_177030

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → -3 ≤ m ∧ m ≤ 5 :=
by
  sorry

end range_of_m_l177_177030


namespace larinjaitis_age_l177_177092

theorem larinjaitis_age : 
  ∀ (birth_year : ℤ) (death_year : ℤ), birth_year = -30 → death_year = 30 → (death_year - birth_year + 1) = 1 :=
by
  intros birth_year death_year h_birth h_death
  sorry

end larinjaitis_age_l177_177092


namespace beautiful_association_number_part1_beautiful_association_number_part2_beautiful_association_number_part3_l177_177871

def beautiful_association_number (x y a t : ℚ) : Prop :=
  |x - a| + |y - a| = t

theorem beautiful_association_number_part1 :
  beautiful_association_number (-3) 5 2 8 :=
by sorry

theorem beautiful_association_number_part2 (x : ℚ) :
  beautiful_association_number x 2 3 4 ↔ x = 6 ∨ x = 0 :=
by sorry

theorem beautiful_association_number_part3 (x0 x1 x2 x3 x4 : ℚ) :
  beautiful_association_number x0 x1 1 1 ∧ 
  beautiful_association_number x1 x2 2 1 ∧ 
  beautiful_association_number x2 x3 3 1 ∧ 
  beautiful_association_number x3 x4 4 1 →
  x1 + x2 + x3 + x4 = 10 :=
by sorry

end beautiful_association_number_part1_beautiful_association_number_part2_beautiful_association_number_part3_l177_177871


namespace sufficient_but_not_necessary_l177_177494

theorem sufficient_but_not_necessary (x : ℝ) : 
  (x = 1 → x^2 = 1) ∧ ¬(x^2 = 1 → x = 1) :=
by
  sorry

end sufficient_but_not_necessary_l177_177494


namespace total_junk_mail_l177_177814

-- Definitions for conditions
def houses_per_block : Nat := 17
def pieces_per_house : Nat := 4
def blocks : Nat := 16

-- Theorem stating that the mailman gives out 1088 pieces of junk mail in total
theorem total_junk_mail : houses_per_block * pieces_per_house * blocks = 1088 := by
  sorry

end total_junk_mail_l177_177814


namespace multiply_polynomials_l177_177113

theorem multiply_polynomials (x : ℝ) : 
  (x^6 + 64 * x^3 + 4096) * (x^3 - 64) = x^9 - 262144 :=
by
  sorry

end multiply_polynomials_l177_177113


namespace smallest_pos_integer_n_l177_177931

theorem smallest_pos_integer_n 
  (x y : ℤ)
  (hx: ∃ k : ℤ, x = 8 * k - 2)
  (hy : ∃ l : ℤ, y = 8 * l + 2) :
  ∃ n : ℤ, n > 0 ∧ ∃ (m : ℤ), x^2 - x*y + y^2 + n = 8 * m ∧ n = 4 := by
  sorry

end smallest_pos_integer_n_l177_177931


namespace largest_fraction_of_three_l177_177170

theorem largest_fraction_of_three (a b c : Nat) (h1 : Nat.gcd a 6 = 1)
  (h2 : Nat.gcd b 15 = 1) (h3 : Nat.gcd c 20 = 1)
  (h4 : (a * b * c) = 60) :
  max (a / 6) (max (b / 15) (c / 20)) = 5 / 6 :=
by
  sorry

end largest_fraction_of_three_l177_177170


namespace point_on_line_iff_l177_177373

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given points O, A, B, and X in a vector space V, prove that X lies on the line AB if and only if
there exists a scalar t such that the position vector of X is a linear combination of the position vectors
of A and B with respect to O. -/
theorem point_on_line_iff (O A B X : V) :
  (∃ t : ℝ, X - O = t • (A - O) + (1 - t) • (B - O)) ↔ (∃ t : ℝ, ∃ (t : ℝ), X - O = (1 - t) • (A - O) + t • (B - O)) :=
sorry

end point_on_line_iff_l177_177373


namespace max_area_of_rectangular_garden_l177_177078

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

end max_area_of_rectangular_garden_l177_177078


namespace manufacturing_cost_eq_210_l177_177130

theorem manufacturing_cost_eq_210 (transport_cost : ℝ) (shoecount : ℕ) (selling_price : ℝ) (gain : ℝ) (M : ℝ) :
  transport_cost = 500 / 100 →
  shoecount = 100 →
  selling_price = 258 →
  gain = 0.20 →
  M = (selling_price / (1 + gain)) - (transport_cost) :=
by
  intros
  sorry

end manufacturing_cost_eq_210_l177_177130


namespace find_number_of_3cm_books_l177_177654

-- Define the conditions
def total_books : ℕ := 46
def total_thickness : ℕ := 200
def thickness_3cm : ℕ := 3
def thickness_5cm : ℕ := 5

-- Let x be the number of 3 cm thick books, y be the number of 5 cm thick books
variable (x y : ℕ)

-- Define the system of equations based on the given conditions
axiom total_books_eq : x + y = total_books
axiom total_thickness_eq : thickness_3cm * x + thickness_5cm * y = total_thickness

-- The theorem to prove: x = 15
theorem find_number_of_3cm_books : x = 15 :=
by
  sorry

end find_number_of_3cm_books_l177_177654


namespace Heesu_has_greatest_sum_l177_177048

-- Define the numbers collected by each individual
def Sora_collected : (Nat × Nat) := (4, 6)
def Heesu_collected : (Nat × Nat) := (7, 5)
def Jiyeon_collected : (Nat × Nat) := (3, 8)

-- Calculate the sums
def Sora_sum : Nat := Sora_collected.1 + Sora_collected.2
def Heesu_sum : Nat := Heesu_collected.1 + Heesu_collected.2
def Jiyeon_sum : Nat := Jiyeon_collected.1 + Jiyeon_collected.2

-- The theorem to prove that Heesu has the greatest sum
theorem Heesu_has_greatest_sum :
  Heesu_sum > Sora_sum ∧ Heesu_sum > Jiyeon_sum :=
by
  sorry

end Heesu_has_greatest_sum_l177_177048


namespace weight_of_each_bag_l177_177995

theorem weight_of_each_bag (empty_weight loaded_weight : ℕ) (number_of_bags : ℕ) (weight_per_bag : ℕ)
    (h1 : empty_weight = 500)
    (h2 : loaded_weight = 1700)
    (h3 : number_of_bags = 20)
    (h4 : loaded_weight - empty_weight = number_of_bags * weight_per_bag) :
    weight_per_bag = 60 :=
by
  sorry

end weight_of_each_bag_l177_177995


namespace square_rectangle_area_ratio_l177_177739

theorem square_rectangle_area_ratio (l1 l2 : ℕ) (h1 : l1 = 32) (h2 : l2 = 64) (p : ℕ) (s : ℕ) 
  (h3 : p = 256) (h4 : s = p / 4)  :
  (s * s) / (l1 * l2) = 2 := 
by
  sorry

end square_rectangle_area_ratio_l177_177739


namespace binomial_coefficient_divisible_by_prime_binomial_coefficient_extreme_cases_l177_177300

-- Definitions and lemma statement
theorem binomial_coefficient_divisible_by_prime
  {p k : ℕ} (hp : Prime p) (hk : 0 < k) (hkp : k < p) :
  p ∣ Nat.choose p k := 
sorry

-- Theorem for k = 0 and k = p cases
theorem binomial_coefficient_extreme_cases {p : ℕ} (hp : Prime p) :
  Nat.choose p 0 = 1 ∧ Nat.choose p p = 1 :=
sorry

end binomial_coefficient_divisible_by_prime_binomial_coefficient_extreme_cases_l177_177300


namespace min_value_perpendicular_vectors_l177_177522

theorem min_value_perpendicular_vectors (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (hperp : x + 3 * y = 1) : (1 / x + 1 / (3 * y)) = 4 :=
by sorry

end min_value_perpendicular_vectors_l177_177522


namespace tetrahedron_volume_l177_177222

theorem tetrahedron_volume (h_1 h_2 h_3 : ℝ) (V : ℝ)
  (h1_pos : 0 < h_1) (h2_pos : 0 < h_2) (h3_pos : 0 < h_3)
  (V_nonneg : 0 ≤ V) : 
  V ≥ (1 / 3) * h_1 * h_2 * h_3 := sorry

end tetrahedron_volume_l177_177222


namespace find_x_l177_177217

def otimes (m n : ℝ) : ℝ := m^2 - 2*m*n

theorem find_x (x : ℝ) (h : otimes (x + 1) (x - 2) = 5) : x = 0 ∨ x = 4 := 
by
  sorry

end find_x_l177_177217


namespace women_fraction_half_l177_177081

theorem women_fraction_half
  (total_people : ℕ)
  (married_fraction : ℝ)
  (max_unmarried_women : ℕ)
  (total_people_eq : total_people = 80)
  (married_fraction_eq : married_fraction = 1 / 2)
  (max_unmarried_women_eq : max_unmarried_women = 32) :
  (∃ (women_fraction : ℝ), women_fraction = 1 / 2) :=
by
  sorry

end women_fraction_half_l177_177081


namespace adam_total_spending_l177_177355

def first_laptop_cost : ℤ := 500
def second_laptop_cost : ℤ := 3 * first_laptop_cost
def total_cost : ℤ := first_laptop_cost + second_laptop_cost

theorem adam_total_spending : total_cost = 2000 := by
  sorry

end adam_total_spending_l177_177355


namespace new_area_of_card_l177_177102

-- Conditions from the problem
def original_length : ℕ := 5
def original_width : ℕ := 7
def shortened_length := original_length - 2
def shortened_width := original_width - 1

-- Statement of the proof problem
theorem new_area_of_card : shortened_length * shortened_width = 18 :=
by
  sorry

end new_area_of_card_l177_177102


namespace set_representation_l177_177668

def is_nat_star (n : ℕ) : Prop := n > 0
def satisfies_eqn (x y : ℕ) : Prop := y = 6 / (x + 3)

theorem set_representation :
  {p : ℕ × ℕ | is_nat_star p.fst ∧ is_nat_star p.snd ∧ satisfies_eqn p.fst p.snd } = { (3, 1) } :=
by
  sorry

end set_representation_l177_177668


namespace remaining_money_after_purchases_l177_177273

def initial_amount : ℝ := 100
def bread_cost : ℝ := 4
def candy_cost : ℝ := 3
def cereal_cost : ℝ := 6
def fruit_percentage : ℝ := 0.2
def milk_cost_each : ℝ := 4.50
def turkey_fraction : ℝ := 0.25

-- Calculate total spent on initial purchases
def initial_spent : ℝ := bread_cost + (2 * candy_cost) + cereal_cost

-- Remaining amount after initial purchases
def remaining_after_initial : ℝ := initial_amount - initial_spent

-- Spend 20% on fruits
def spent_on_fruits : ℝ := fruit_percentage * remaining_after_initial
def remaining_after_fruits : ℝ := remaining_after_initial - spent_on_fruits

-- Spend on two gallons of milk
def spent_on_milk : ℝ := 2 * milk_cost_each
def remaining_after_milk : ℝ := remaining_after_fruits - spent_on_milk

-- Spend 1/4 on turkey
def spent_on_turkey : ℝ := turkey_fraction * remaining_after_milk
def final_remaining : ℝ := remaining_after_milk - spent_on_turkey

theorem remaining_money_after_purchases : final_remaining = 43.65 := by
  sorry

end remaining_money_after_purchases_l177_177273


namespace probability_of_matching_pair_l177_177575

/-!
# Probability of Selecting a Matching Pair of Shoes

Given:
- 12 pairs of sneakers, each with a 4% probability of being chosen.
- 15 pairs of boots, each with a 3% probability of being chosen.
- 18 pairs of dress shoes, each with a 2% probability of being chosen.

If two shoes are selected from the warehouse without replacement, prove that the probability 
of selecting a matching pair of shoes is 52.26%.
-/

namespace ShoeWarehouse

def prob_sneakers_first : ℝ := 0.48
def prob_sneakers_second : ℝ := 0.44
def prob_boots_first : ℝ := 0.45
def prob_boots_second : ℝ := 0.42
def prob_dress_first : ℝ := 0.36
def prob_dress_second : ℝ := 0.34

theorem probability_of_matching_pair :
  (prob_sneakers_first * prob_sneakers_second) +
  (prob_boots_first * prob_boots_second) +
  (prob_dress_first * prob_dress_second) = 0.5226 :=
sorry

end ShoeWarehouse

end probability_of_matching_pair_l177_177575


namespace sum_diameters_eq_sum_legs_l177_177212

theorem sum_diameters_eq_sum_legs 
  (a b c R r : ℝ)
  (h_right_triangle : a^2 + b^2 = c^2)
  (h_circum_radius : R = c / 2)
  (h_incircle_radius : r = (a + b - c) / 2) :
  2 * R + 2 * r = a + b :=
by 
  sorry

end sum_diameters_eq_sum_legs_l177_177212


namespace total_jumps_is_400_l177_177870

-- Define the variables according to the conditions 
def Ronald_jumps := 157
def Rupert_jumps := Ronald_jumps + 86

-- Prove the total jumps
theorem total_jumps_is_400 : Ronald_jumps + Rupert_jumps = 400 := by
  sorry

end total_jumps_is_400_l177_177870


namespace ball_bounce_height_l177_177840

noncomputable def min_bounces (h₀ h_min : ℝ) (bounce_factor : ℝ) := 
  Nat.ceil (Real.log (h_min / h₀) / Real.log bounce_factor)

theorem ball_bounce_height :
  min_bounces 512 40 (3/4) = 8 :=
by
  sorry

end ball_bounce_height_l177_177840


namespace comb_identity_l177_177770

theorem comb_identity (n : Nat) (h : 0 < n) (h_eq : Nat.choose n 2 = Nat.choose (n-1) 2 + Nat.choose (n-1) 3) : n = 5 := by
  sorry

end comb_identity_l177_177770


namespace mixture_ratio_l177_177306

variables (p q V W : ℝ)

-- Condition summaries:
-- - First jar has volume V, ratio of alcohol to water is p:1.
-- - Second jar has volume W, ratio of alcohol to water is q:2.

theorem mixture_ratio (hp : p > 0) (hq : q > 0) (hV : V > 0) (hW : W > 0) : 
  (p * V * (p + 2) + q * W * (p + 1)) / ((p + 1) * (q + 2) * (V + 2 * W)) =
  (p * V) / (p + 1) + (q * W) / (q + 2) :=
sorry

end mixture_ratio_l177_177306


namespace hexahedron_has_six_faces_l177_177094

-- Definition based on the condition
def is_hexahedron (P : Type) := 
  ∃ (f : P → ℕ), ∀ (x : P), f x = 6

-- Theorem statement based on the question and correct answer
theorem hexahedron_has_six_faces (P : Type) (h : is_hexahedron P) : 
  ∀ (x : P), ∃ (f : P → ℕ), f x = 6 :=
by 
  sorry

end hexahedron_has_six_faces_l177_177094


namespace only_D_is_quadratic_l177_177093

-- Conditions
def eq_A (x : ℝ) : Prop := x^2 + 1/x - 1 = 0
def eq_B (x : ℝ) : Prop := (2*x + 1) + x = 0
def eq_C (m x : ℝ) : Prop := 2*m^2 + x = 3
def eq_D (x : ℝ) : Prop := x^2 - x = 0

-- Proof statement
theorem only_D_is_quadratic :
  ∃ (x : ℝ), eq_D x ∧ 
  (¬(∃ x : ℝ, eq_A x) ∧ ¬(∃ x : ℝ, eq_B x) ∧ ¬(∃ (m x : ℝ), eq_C m x)) :=
by
  sorry

end only_D_is_quadratic_l177_177093


namespace find_angle_A_l177_177612

def triangle_ABC_angle_A (a b : ℝ) (B A : ℝ) (acute : Prop)
  (ha : a = 2 * Real.sqrt 3)
  (hb : b = 2 * Real.sqrt 2)
  (hB: B = Real.pi / 4)
  (hacute: acute) : Prop :=
  A = Real.pi / 3

theorem find_angle_A 
  (a b A B : ℝ) (acute : Prop)
  (ha : a = 2 * Real.sqrt 3)
  (hb : b = 2 * Real.sqrt 2)
  (hB: B = Real.pi / 4)
  (hacute: acute)
  (h_conditions : triangle_ABC_angle_A a b B A acute ha hb hB hacute) : 
  A = Real.pi / 3 := 
sorry

end find_angle_A_l177_177612


namespace C_share_of_profit_l177_177552

def A_investment : ℕ := 12000
def B_investment : ℕ := 16000
def C_investment : ℕ := 20000
def total_profit : ℕ := 86400

theorem C_share_of_profit: 
  (C_investment / (A_investment + B_investment + C_investment) * total_profit) = 36000 :=
by
  sorry

end C_share_of_profit_l177_177552


namespace m_perp_beta_l177_177474

variable {Point Line Plane : Type}
variable {belongs : Point → Line → Prop}
variable {perp : Line → Plane → Prop}
variable {intersect : Plane → Plane → Line}

variable (α β γ : Plane)
variable (m n l : Line)

-- Conditions for the problem
axiom n_perp_α : perp n α
axiom n_perp_β : perp n β
axiom m_perp_α : perp m α

-- Proof goal: proving m is perpendicular to β
theorem m_perp_beta : perp m β :=
by
  sorry

end m_perp_beta_l177_177474


namespace triangle_side_b_l177_177851

-- Define the conditions and state the problem
theorem triangle_side_b (A C : ℕ) (a b c : ℝ)
  (h1 : C = 4 * A)
  (h2 : a = 36)
  (h3 : c = 60) :
  b = 45 := by
  sorry

end triangle_side_b_l177_177851


namespace parabola_focus_coordinates_l177_177837

theorem parabola_focus_coordinates (x y p : ℝ) (h : y^2 = 8 * x) : 
  p = 2 → (p, 0) = (2, 0) := 
by 
  sorry

end parabola_focus_coordinates_l177_177837


namespace angle_between_vectors_l177_177388

def vector (α : Type) [Field α] := (α × α)

theorem angle_between_vectors
    (a : vector ℝ)
    (b : vector ℝ)
    (ha : a = (4, 0))
    (hb : b = (-1, Real.sqrt 3)) :
  let dot_product (v w : vector ℝ) : ℝ := (v.1 * w.1 + v.2 * w.2)
  let norm (v : vector ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)
  let cos_theta := dot_product a b / (norm a * norm b)
  ∀ theta, Real.cos theta = cos_theta → theta = 2 * Real.pi / 3 :=
by
  sorry

end angle_between_vectors_l177_177388


namespace number_of_players_in_tournament_l177_177964

theorem number_of_players_in_tournament (n : ℕ) (h : 2 * 30 = n * (n - 1)) : n = 10 :=
sorry

end number_of_players_in_tournament_l177_177964


namespace swimming_lane_length_l177_177557

theorem swimming_lane_length (round_trips : ℕ) (total_distance : ℕ) (lane_length : ℕ) 
  (h1 : round_trips = 4) (h2 : total_distance = 800) 
  (h3 : total_distance = lane_length * (round_trips * 2)) : 
  lane_length = 100 := 
by
  sorry

end swimming_lane_length_l177_177557


namespace rick_bought_30_guppies_l177_177319

theorem rick_bought_30_guppies (G : ℕ) (T C : ℕ) 
  (h1 : T = 4 * C) 
  (h2 : C = 2 * G) 
  (h3 : G + C + T = 330) : 
  G = 30 := 
by 
  sorry

end rick_bought_30_guppies_l177_177319


namespace system_solution_l177_177401

theorem system_solution (x y z : ℚ) 
  (h1 : x + y + x * y = 19) 
  (h2 : y + z + y * z = 11) 
  (h3 : z + x + z * x = 14) :
    (x = 4 ∧ y = 3 ∧ z = 2) ∨ (x = -6 ∧ y = -5 ∧ z = -4) :=
by
  sorry

end system_solution_l177_177401


namespace focus_coordinates_of_hyperbola_l177_177820

theorem focus_coordinates_of_hyperbola (x y : ℝ) :
  (∃ c : ℝ, (c = 5 ∧ y = 10) ∧ (c = 5 + Real.sqrt 97)) ↔ 
  (x, y) = (5 + Real.sqrt 97, 10) :=
by
  sorry

end focus_coordinates_of_hyperbola_l177_177820


namespace salad_dressing_percentage_l177_177872

variable (P Q : ℝ) -- P and Q are the amounts of dressings P and Q in grams

-- Conditions
variable (h1 : 0.3 * P + 0.1 * Q = 12) -- The combined vinegar percentage condition
variable (h2 : P + Q = 100)            -- The total weight condition

-- Statement to prove
theorem salad_dressing_percentage (P_percent : ℝ) 
    (h1 : 0.3 * P + 0.1 * Q = 12) (h2 : P + Q = 100) : 
    P / (P + Q) * 100 = 10 :=
sorry

end salad_dressing_percentage_l177_177872


namespace count_ways_to_write_2010_l177_177724

theorem count_ways_to_write_2010 : ∃ N : ℕ, 
  (∀ (a_3 a_2 a_1 a_0 : ℕ), a_0 ≤ 99 ∧ a_1 ≤ 99 ∧ a_2 ≤ 99 ∧ a_3 ≤ 99 → 
    2010 = a_3 * 10^3 + a_2 * 10^2 + a_1 * 10 + a_0) ∧ 
    N = 202 :=
sorry

end count_ways_to_write_2010_l177_177724


namespace cost_function_segments_l177_177967

def C (n : ℕ) : ℕ :=
  if h : 1 ≤ n ∧ n ≤ 10 then 10 * n
  else if h : 10 < n then 8 * n - 40
  else 0

theorem cost_function_segments :
  (∀ n, 1 ≤ n ∧ n ≤ 10 → C n = 10 * n) ∧
  (∀ n, 10 < n → C n = 8 * n - 40) ∧
  (∀ n, C n = if (1 ≤ n ∧ n ≤ 10) then 10 * n else if (10 < n) then 8 * n - 40 else 0) ∧
  ∃ n₁ n₂, (1 ≤ n₁ ∧ n₁ ≤ 10) ∧ (10 < n₂ ∧ n₂ ≤ 20) ∧ C n₁ = 10 * n₁ ∧ C n₂ = 8 * n₂ - 40 :=
by
  sorry

end cost_function_segments_l177_177967


namespace cuboid_surface_area_l177_177143

-- Definitions
def Length := 12  -- meters
def Breadth := 14  -- meters
def Height := 7  -- meters

-- Surface area of a cuboid formula
def surfaceAreaOfCuboid (l b h : Nat) : Nat :=
  2 * (l * b + l * h + b * h)

-- Proof statement
theorem cuboid_surface_area : surfaceAreaOfCuboid Length Breadth Height = 700 := by
  sorry

end cuboid_surface_area_l177_177143


namespace repeating_decimals_sum_l177_177180

theorem repeating_decimals_sum :
  let x := (246 : ℚ) / 999
  let y := (135 : ℚ) / 999
  let z := (579 : ℚ) / 999
  x - y + z = (230 : ℚ) / 333 :=
by
  sorry

end repeating_decimals_sum_l177_177180


namespace pyramid_base_side_length_l177_177782

theorem pyramid_base_side_length (A : ℝ) (h : ℝ) (s : ℝ)
  (hA : A = 200)
  (hh : h = 40)
  (hface : A = (1 / 2) * s * h) : 
  s = 10 :=
by
  sorry

end pyramid_base_side_length_l177_177782


namespace locus_of_point_P_l177_177393

noncomputable def ellipse_locus
  (r : ℝ) (u v : ℝ) : Prop :=
  ∃ x1 y1 : ℝ,
    (x1^2 + y1^2 = r^2) ∧ (u - x1)^2 + v^2 = y1^2

theorem locus_of_point_P {r u v : ℝ} :
  (ellipse_locus r u v) ↔ ((u^2 / (2 * r^2)) + (v^2 / r^2) ≤ 1) :=
by sorry

end locus_of_point_P_l177_177393


namespace shaded_area_l177_177256

theorem shaded_area (PR PV PQ QR : ℝ) (hPR : PR = 20) (hPV : PV = 12) (hPQ_QR : PQ + QR = PR) :
  PR * PV - 1 / 2 * 12 * PR = 120 :=
by
  -- Definitions used earlier
  have h_area_rectangle : PR * PV = 240 := by
    rw [hPR, hPV]
    norm_num
  have h_half_total_unshaded : (1 / 2) * 12 * PR = 120 := by
    rw [hPR]
    norm_num
  rw [h_area_rectangle, h_half_total_unshaded]
  norm_num

end shaded_area_l177_177256


namespace cubic_feet_per_bag_l177_177517

-- Definitions
def length_bed := 8 -- in feet
def width_bed := 4 -- in feet
def height_bed := 1 -- in feet
def number_of_beds := 2
def number_of_bags := 16

-- Theorem statement
theorem cubic_feet_per_bag : 
  (length_bed * width_bed * height_bed * number_of_beds) / number_of_bags = 4 :=
by
  sorry

end cubic_feet_per_bag_l177_177517


namespace pet_store_satisfaction_l177_177258

theorem pet_store_satisfaction :
  let puppies := 15
  let kittens := 6
  let hamsters := 8
  let friends := 3
  puppies * kittens * hamsters * friends.factorial = 4320 := by
  sorry

end pet_store_satisfaction_l177_177258


namespace height_relationship_l177_177297

theorem height_relationship (r1 r2 h1 h2 : ℝ) (h_radii : r2 = 1.2 * r1) (h_volumes : π * r1^2 * h1 = π * r2^2 * h2) : h1 = 1.44 * h2 :=
by
  sorry

end height_relationship_l177_177297


namespace price_of_tray_l177_177137

noncomputable def price_per_egg : ℕ := 50
noncomputable def tray_eggs : ℕ := 30
noncomputable def discount_per_egg : ℕ := 10

theorem price_of_tray : (price_per_egg - discount_per_egg) * tray_eggs / 100 = 12 :=
by
  sorry

end price_of_tray_l177_177137


namespace find_matrix_M_l177_177801

-- Define the given matrix with real entries
def matrix_M : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 2], ![-1, 0]]

-- Define the function for matrix operations
def M_calc (M : Matrix (Fin 2) (Fin 2) ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  (M * M * M) - (M * M) + (2 • M)

-- Define the target matrix
def target_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 3], ![-2, 0]]

-- Problem statement: The matrix M should satisfy the given matrix equation
theorem find_matrix_M (M : Matrix (Fin 2) (Fin 2) ℝ) :
  M_calc M = target_matrix ↔ M = matrix_M :=
sorry

end find_matrix_M_l177_177801


namespace max_sum_clock_digits_l177_177936

theorem max_sum_clock_digits : ∃ t : ℕ, 0 ≤ t ∧ t < 24 ∧ 
  (∃ h1 h2 m1 m2 : ℕ, t = h1 * 10 + h2 + m1 * 10 + m2 ∧ 
   (0 ≤ h1 ∧ h1 ≤ 2) ∧ (0 ≤ h2 ∧ h2 ≤ 9) ∧ (0 ≤ m1 ∧ m1 ≤ 5) ∧ (0 ≤ m2 ∧ m2 ≤ 9) ∧ 
   h1 + h2 + m1 + m2 = 24) := sorry

end max_sum_clock_digits_l177_177936


namespace average_height_Heidi_Lola_l177_177098

theorem average_height_Heidi_Lola :
  (2.1 + 1.4) / 2 = 1.75 := by
  sorry

end average_height_Heidi_Lola_l177_177098


namespace correct_options_l177_177415

-- Given conditions
def f : ℝ → ℝ := sorry -- We will assume there is some function f that satisfies the conditions

axiom xy_identity (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = x * f y + y * f x
axiom f_positive (x : ℝ) (hx : 1 < x) : 0 < f x

-- Proof of the required conclusion
theorem correct_options (h1 : f 1 = 0) (h2 : ∀ x y, f (x * y) ≠ f x * f y)
  (h3 : ∀ x, 1 < x → ∀ y, 1 < y → x < y → f x < f y)
  (h4 : ∀ x, 2 ≤ x → x * f (x - 3 / 2) ≥ (3 / 2 - x) * f x) : 
  f 1 = 0 ∧ (∀ x y, f (x * y) ≠ f x * f y) ∧ (∀ x, 1 < x → ∀ y, 1 < y → x < y → f x < f y) ∧ (∀ x, 2 ≤ x → x * f (x - 3 / 2) ≥ (3 / 2 - x) * f x) :=
sorry

end correct_options_l177_177415


namespace x_cubed_plus_square_plus_lin_plus_a_l177_177264

theorem x_cubed_plus_square_plus_lin_plus_a (a b x : ℝ) (h : b / x^3 + 1 / x^2 + 1 / x + 1 = 0) :
  x^3 + x^2 + x + a = a - b :=
by {
  sorry
}

end x_cubed_plus_square_plus_lin_plus_a_l177_177264


namespace squares_overlap_ratio_l177_177259

theorem squares_overlap_ratio (a b : ℝ) (h1 : 0.52 * a^2 = a^2 - (a^2 - 0.52 * a^2))
                             (h2 : 0.73 * b^2 = b^2 - (b^2 - 0.73 * b^2)) :
                             a / b = 3 / 4 := by
sorry

end squares_overlap_ratio_l177_177259


namespace volume_of_prism_l177_177203

theorem volume_of_prism (a : ℝ) (h_pos : 0 < a) (h_lat : ∀ S_lat, S_lat = a ^ 2) : 
  ∃ V, V = (a ^ 3 * (Real.sqrt 2 - 1)) / 4 :=
by
  sorry

end volume_of_prism_l177_177203


namespace min_expression_l177_177011

theorem min_expression (a b c d e f : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
  (h_sum : a + b + c + d + e + f = 10) : 
  (1 / a + 9 / b + 16 / c + 25 / d + 36 / e + 49 / f) ≥ 67.6 :=
sorry

end min_expression_l177_177011


namespace find_larger_number_l177_177530

variables (x y : ℝ)

def sum_cond : Prop := x + y = 17
def diff_cond : Prop := x - y = 7

theorem find_larger_number (h1 : sum_cond x y) (h2 : diff_cond x y) : x = 12 :=
sorry

end find_larger_number_l177_177530


namespace initial_marbles_count_l177_177551

-- Leo's initial conditions and quantities
def initial_packs := 40
def marbles_per_pack := 10
def given_Manny (P: ℕ) := P / 4
def given_Neil (P: ℕ) := P / 8
def kept_by_Leo := 25

-- The equivalent proof problem stated in Lean
theorem initial_marbles_count (P: ℕ) (Manny_packs: ℕ) (Neil_packs: ℕ) (kept_packs: ℕ) :
  Manny_packs = given_Manny P → Neil_packs = given_Neil P → kept_packs = kept_by_Leo → P = initial_packs → P * marbles_per_pack = 400 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_marbles_count_l177_177551


namespace two_distinct_nonzero_complex_numbers_l177_177308

noncomputable def count_distinct_nonzero_complex_numbers_satisfying_conditions : ℕ :=
sorry

theorem two_distinct_nonzero_complex_numbers :
  count_distinct_nonzero_complex_numbers_satisfying_conditions = 2 :=
sorry

end two_distinct_nonzero_complex_numbers_l177_177308


namespace set_intersection_complement_l177_177657

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}
noncomputable def A : Set ℕ := {1, 3}
noncomputable def B : Set ℕ := {2, 3}

theorem set_intersection_complement :
  A ∩ (U \ B) = {1} :=
sorry

end set_intersection_complement_l177_177657


namespace minimum_value_of_expression_l177_177505

theorem minimum_value_of_expression (a b : ℝ) (h1 : a > 0) (h2 : a > b) (h3 : ab = 1) :
  (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2 := 
sorry

end minimum_value_of_expression_l177_177505


namespace Alejandra_overall_score_l177_177406

theorem Alejandra_overall_score :
  let score1 := (60/100 : ℝ) * 20
  let score2 := (75/100 : ℝ) * 30
  let score3 := (85/100 : ℝ) * 40
  let total_score := score1 + score2 + score3
  let total_questions := 90
  let overall_percentage := (total_score / total_questions) * 100
  round overall_percentage = 77 :=
by
  sorry

end Alejandra_overall_score_l177_177406


namespace count_sum_or_diff_squares_at_least_1500_l177_177563

theorem count_sum_or_diff_squares_at_least_1500 : 
  (∃ (n : ℕ), 1 ≤ n ∧ n ≤ 2000 ∧ (∃ (x y : ℕ), n = x^2 + y^2 ∨ n = x^2 - y^2)) → 
  1500 ≤ 2000 :=
by
  sorry

end count_sum_or_diff_squares_at_least_1500_l177_177563


namespace area_of_BEIH_l177_177299

def calculate_area_of_quadrilateral (A B C D E F I H : (ℝ × ℝ)) : ℝ := 
  sorry

theorem area_of_BEIH : 
  let A := (0, 3)
  let B := (0, 0)
  let C := (3, 0)
  let D := (3, 3)
  let E := (0, 1.5)
  let F := (1, 0)
  let I := (3 / 5, 9 / 5)
  let H := (3 / 4, 3 / 4)
  calculate_area_of_quadrilateral A B C D E F I H = 27 / 40 :=
sorry

end area_of_BEIH_l177_177299


namespace triangle_angle_bisectors_l177_177737

theorem triangle_angle_bisectors (α β γ : ℝ) 
  (h1 : α + β + γ = 180)
  (h2 : α = 100) 
  (h3 : β = 30) 
  (h4 : γ = 50) :
  ∃ α' β' γ', α' = 40 ∧ β' = 65 ∧ γ' = 75 :=
sorry

end triangle_angle_bisectors_l177_177737


namespace total_cost_l177_177150

noncomputable def cost_sandwich : ℝ := 2.44
noncomputable def quantity_sandwich : ℕ := 2
noncomputable def cost_soda : ℝ := 0.87
noncomputable def quantity_soda : ℕ := 4

noncomputable def total_cost_sandwiches : ℝ := cost_sandwich * quantity_sandwich
noncomputable def total_cost_sodas : ℝ := cost_soda * quantity_soda

theorem total_cost (total_cost_sandwiches total_cost_sodas : ℝ) : (total_cost_sandwiches + total_cost_sodas = 8.36) :=
by
  sorry

end total_cost_l177_177150


namespace marble_color_197th_l177_177465

theorem marble_color_197th (n : ℕ) (total_marbles : ℕ) (marble_color : ℕ → ℕ)
                          (h_total : total_marbles = 240)
                          (h_pattern : ∀ k, marble_color (k + 15) = marble_color k)
                          (h_colors : ∀ i, (0 ≤ i ∧ i < 15) →
                                   (marble_color i = if i < 6 then 1
                                   else if i < 11 then 2
                                   else if i < 15 then 3
                                   else 0)) :
  marble_color 197 = 1 := sorry

end marble_color_197th_l177_177465


namespace smallest_k_for_polygon_l177_177919

-- Definitions and conditions
def equiangular_decagon_interior_angle : ℝ := 144

-- Question transformation into a proof problem
theorem smallest_k_for_polygon (k : ℕ) (hk : k > 1) :
  (∀ (n2 : ℕ), n2 = 10 * k → ∃ (interior_angle : ℝ), interior_angle = k * equiangular_decagon_interior_angle ∧
  n2 ≥ 3) → k = 2 :=
by
  sorry

end smallest_k_for_polygon_l177_177919


namespace combination_addition_l177_177197

noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem combination_addition :
  combination 13 11 + 3 = 81 :=
by
  sorry

end combination_addition_l177_177197


namespace Brian_age_in_eight_years_l177_177014

-- Definitions based on conditions
variable {Christian Brian : ℕ}
variable (h1 : Christian = 2 * Brian)
variable (h2 : Christian + 8 = 72)

-- Target statement to prove Brian's age in eight years
theorem Brian_age_in_eight_years : (Brian + 8) = 40 :=
by 
  sorry

end Brian_age_in_eight_years_l177_177014


namespace determine_OQ_l177_177504

theorem determine_OQ (l m n p O A B C D Q : ℝ) (h0 : O = 0)
  (hA : A = l) (hB : B = m) (hC : C = n) (hD : D = p)
  (hQ : l ≤ Q ∧ Q ≤ m)
  (h_ratio : (|C - Q| / |Q - D|) = (|B - Q| / |Q - A|)) :
  Q = (l + m) / 2 :=
sorry

end determine_OQ_l177_177504


namespace axis_of_symmetry_parabola_l177_177914

theorem axis_of_symmetry_parabola : 
  ∀ (x : ℝ), ∃ y : ℝ, y = (x - 5)^2 → x = 5 := 
by 
  sorry

end axis_of_symmetry_parabola_l177_177914


namespace fifth_friend_paid_40_l177_177462

variable (x1 x2 x3 x4 x5 : ℝ)

def conditions : Prop :=
  (x1 = 1/3 * (x2 + x3 + x4 + x5)) ∧
  (x2 = 1/4 * (x1 + x3 + x4 + x5)) ∧
  (x3 = 1/5 * (x1 + x2 + x4 + x5)) ∧
  (x4 = 1/6 * (x1 + x2 + x3 + x5)) ∧
  (x1 + x2 + x3 + x4 + x5 = 120)

theorem fifth_friend_paid_40 (h : conditions x1 x2 x3 x4 x5) : x5 = 40 := by
  sorry

end fifth_friend_paid_40_l177_177462


namespace combined_mpg_correct_l177_177446

def ray_mpg := 30
def tom_mpg := 15
def alice_mpg := 60
def distance_each := 120

-- Total gasoline consumption
def ray_gallons := distance_each / ray_mpg
def tom_gallons := distance_each / tom_mpg
def alice_gallons := distance_each / alice_mpg

def total_gallons := ray_gallons + tom_gallons + alice_gallons
def total_distance := 3 * distance_each

def combined_mpg := total_distance / total_gallons

theorem combined_mpg_correct :
  combined_mpg = 26 :=
by
  -- All the necessary calculations would go here.
  sorry

end combined_mpg_correct_l177_177446


namespace ciphertext_to_plaintext_l177_177523

theorem ciphertext_to_plaintext :
  ∃ (a b c d : ℕ), (a + 2 * b = 14) ∧ (2 * b + c = 9) ∧ (2 * c + 3 * d = 23) ∧ (4 * d = 28) ∧ a = 6 ∧ b = 4 ∧ c = 1 ∧ d = 7 :=
by 
  sorry

end ciphertext_to_plaintext_l177_177523


namespace employees_in_january_l177_177347

theorem employees_in_january (E : ℝ) (h : 500 = 1.15 * E) : E = 500 / 1.15 :=
by
  sorry

end employees_in_january_l177_177347


namespace total_items_8_l177_177949

def sandwiches_cost : ℝ := 5.0
def soft_drinks_cost : ℝ := 1.5
def total_money : ℝ := 40.0

noncomputable def total_items (s : ℕ) (d : ℕ) : ℕ := s + d

theorem total_items_8 :
  ∃ (s d : ℕ), 5 * (s : ℝ) + 1.5 * (d : ℝ) = 40 ∧ s + d = 8 := 
by
  sorry

end total_items_8_l177_177949


namespace williams_farm_tax_l177_177072

variables (T : ℝ)
variables (tax_collected : ℝ := 3840)
variables (percentage_williams_land : ℝ := 0.5)
variables (percentage_taxable_land : ℝ := 0.25)

theorem williams_farm_tax : (percentage_williams_land * tax_collected) = 1920 := by
  sorry

end williams_farm_tax_l177_177072


namespace regina_earnings_l177_177426

-- Definitions based on conditions
def num_cows := 20
def num_pigs := 4 * num_cows
def price_per_pig := 400
def price_per_cow := 800

-- Total earnings calculation based on definitions
def earnings_from_cows := num_cows * price_per_cow
def earnings_from_pigs := num_pigs * price_per_pig
def total_earnings := earnings_from_cows + earnings_from_pigs

-- Proof statement
theorem regina_earnings : total_earnings = 48000 := by
  sorry

end regina_earnings_l177_177426


namespace bob_needs_additional_weeks_l177_177992

-- Definitions based on conditions
def weekly_prize : ℕ := 100
def initial_weeks_won : ℕ := 2
def total_prize_won : ℕ := initial_weeks_won * weekly_prize
def puppy_cost : ℕ := 1000
def additional_weeks_needed : ℕ := (puppy_cost - total_prize_won) / weekly_prize

-- Statement of the theorem
theorem bob_needs_additional_weeks : additional_weeks_needed = 8 := by
  -- Proof here
  sorry

end bob_needs_additional_weeks_l177_177992


namespace derek_percentage_difference_l177_177214

-- Definitions and assumptions based on conditions
def average_score_first_test (A : ℝ) : ℝ := A

def derek_score_first_test (D1 : ℝ) (A : ℝ) : Prop := D1 = 0.5 * A

def derek_score_second_test (D2 : ℝ) (D1 : ℝ) : Prop := D2 = 1.5 * D1

-- Theorem statement
theorem derek_percentage_difference (A D1 D2 : ℝ)
  (h1 : derek_score_first_test D1 A)
  (h2 : derek_score_second_test D2 D1) :
  (A - D2) / A * 100 = 25 :=
by
  -- Placeholder for the proof
  sorry

end derek_percentage_difference_l177_177214


namespace probability_one_letter_from_each_l177_177511

theorem probability_one_letter_from_each
  (total_cards : ℕ)
  (adam_cards : ℕ)
  (brian_cards : ℕ)
  (h1 : total_cards = 12)
  (h2 : adam_cards = 4)
  (h3 : brian_cards = 6)
  : (4/12 * 6/11) + (6/12 * 4/11) = 4/11 := by
  sorry

end probability_one_letter_from_each_l177_177511


namespace optimal_solution_range_l177_177760

theorem optimal_solution_range (a : ℝ) (x y : ℝ) :
  (x + y - 4 ≥ 0) → (2 * x - y - 5 ≤ 0) → (x = 1) → (y = 3) →
  (-2 < a) ∧ (a < 1) :=
by
  intros h1 h2 hx hy
  sorry

end optimal_solution_range_l177_177760


namespace modulus_of_z_l177_177017

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := sorry

theorem modulus_of_z 
  (hz : i * z = (1 - 2 * i)^2) : 
  Complex.abs z = 5 := by
  sorry

end modulus_of_z_l177_177017


namespace sum_lent_is_1050_l177_177684

-- Define the variables for the problem
variable (P : ℝ) -- Sum lent
variable (r : ℝ) -- Interest rate
variable (t : ℝ) -- Time period
variable (I : ℝ) -- Interest

-- Define the conditions
def conditions := 
  r = 0.06 ∧ 
  t = 6 ∧ 
  I = P - 672 ∧ 
  I = P * (r * t)

-- Define the main theorem
theorem sum_lent_is_1050 (P r t I : ℝ) (h : conditions P r t I) : P = 1050 :=
  sorry

end sum_lent_is_1050_l177_177684


namespace largest_n_crates_same_number_oranges_l177_177518

theorem largest_n_crates_same_number_oranges (total_crates : ℕ) 
  (crate_min_oranges : ℕ) (crate_max_oranges : ℕ) 
  (h1 : total_crates = 200) (h2 : crate_min_oranges = 100) (h3 : crate_max_oranges = 130) 
  : ∃ n : ℕ, n = 7 ∧ ∀ orange_count, crate_min_oranges ≤ orange_count ∧ orange_count ≤ crate_max_oranges → ∃ k, k = n ∧ ∃ t, t ≤ total_crates ∧ t ≥ k := 
sorry

end largest_n_crates_same_number_oranges_l177_177518


namespace inequality_solution_set_l177_177171

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^3 + (a-1)*x^2

theorem inequality_solution_set (a : ℝ) (ha : ∀ x : ℝ, f x a = -f (-x) a) :
  {x : ℝ | f (a*x) a > f (a-x) a} = {x : ℝ | x > 1/2} :=
by
  sorry

end inequality_solution_set_l177_177171


namespace master_bedroom_and_bath_area_l177_177716

-- Definitions of the problem conditions
def guest_bedroom_area : ℕ := 200
def two_guest_bedrooms_area : ℕ := 2 * guest_bedroom_area
def kitchen_guest_bath_living_area : ℕ := 600
def total_rent : ℕ := 3000
def cost_per_sq_ft : ℕ := 2
def total_area_of_house : ℕ := total_rent / cost_per_sq_ft
def expected_master_bedroom_and_bath_area : ℕ := 500

-- Theorem statement to prove the desired area
theorem master_bedroom_and_bath_area :
  total_area_of_house - (two_guest_bedrooms_area + kitchen_guest_bath_living_area) = expected_master_bedroom_and_bath_area :=
by
  sorry

end master_bedroom_and_bath_area_l177_177716


namespace min_price_floppy_cd_l177_177599

theorem min_price_floppy_cd (x y : ℝ) (h1 : 4 * x + 5 * y ≥ 20) (h2 : 6 * x + 3 * y ≤ 24) : 3 * x + 9 * y ≥ 22 :=
by
  -- The proof is not provided as per the instructions.
  sorry

end min_price_floppy_cd_l177_177599


namespace original_numbers_geometric_sequence_l177_177891

theorem original_numbers_geometric_sequence (a q : ℝ) :
  (2 * (a * q + 8) = a + a * q^2) →
  ((a * q + 8) ^ 2 = a * (a * q^2 + 64)) →
  (a, a * q, a * q^2) = (4, 12, 36) ∨ (a, a * q, a * q^2) = (4 / 9, -20 / 9, 100 / 9) :=
by {
  sorry
}

end original_numbers_geometric_sequence_l177_177891


namespace shooter_hit_rate_l177_177441

noncomputable def shooter_prob := 2 / 3

theorem shooter_hit_rate:
  ∀ (x : ℚ), (1 - x)^4 = 1 / 81 → x = shooter_prob :=
by
  intro x h
  -- Proof is omitted
  sorry

end shooter_hit_rate_l177_177441


namespace sum_of_decimals_l177_177987

theorem sum_of_decimals :
  let a := 0.3
  let b := 0.08
  let c := 0.007
  a + b + c = 0.387 :=
by
  sorry

end sum_of_decimals_l177_177987


namespace counting_five_digit_numbers_l177_177116

theorem counting_five_digit_numbers :
  ∃ (M : ℕ), 
    (∃ (b : ℕ), (∃ (y : ℕ), 10000 * b + y = 8 * y ∧ 10000 * b = 7 * y ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1429 ≤ y ∧ y ≤ 9996)) ∧ 
    (M = 1224) := 
by
  sorry

end counting_five_digit_numbers_l177_177116


namespace focus_of_parabola_l177_177863

-- Define the given parabola equation
def given_parabola (x : ℝ) : ℝ := 4 * x^2

-- Define what it means to be the focus of this parabola
def is_focus (focus : ℝ × ℝ) : Prop :=
  focus = (0, 1 / 16)

-- The theorem to prove
theorem focus_of_parabola : ∃ focus : ℝ × ℝ, is_focus focus :=
  by 
    use (0, 1 / 16)
    exact sorry

end focus_of_parabola_l177_177863


namespace average_percentage_increase_l177_177608

def initial_income_A : ℝ := 60
def new_income_A : ℝ := 80
def initial_income_B : ℝ := 100
def new_income_B : ℝ := 130
def hours_worked_C : ℝ := 20
def initial_rate_C : ℝ := 8
def new_rate_C : ℝ := 10

theorem average_percentage_increase :
  let initial_weekly_income_C := hours_worked_C * initial_rate_C
  let new_weekly_income_C := hours_worked_C * new_rate_C
  let percentage_increase_A := (new_income_A - initial_income_A) / initial_income_A * 100
  let percentage_increase_B := (new_income_B - initial_income_B) / initial_income_B * 100
  let percentage_increase_C := (new_weekly_income_C - initial_weekly_income_C) / initial_weekly_income_C * 100
  let average_percentage_increase := (percentage_increase_A + percentage_increase_B + percentage_increase_C) / 3
  average_percentage_increase = 29.44 :=
by sorry

end average_percentage_increase_l177_177608


namespace rearrange_digits_to_perfect_square_l177_177228

theorem rearrange_digits_to_perfect_square :
  ∃ n : ℤ, 2601 = n ^ 2 ∧ (∃ (perm : List ℤ), perm = [2, 0, 1, 6] ∧ perm.permutations ≠ List.nil) :=
by
  sorry

end rearrange_digits_to_perfect_square_l177_177228


namespace Abby_has_17_quarters_l177_177829

theorem Abby_has_17_quarters (q n : ℕ) (h1 : q + n = 23) (h2 : 25 * q + 5 * n = 455) : q = 17 :=
sorry

end Abby_has_17_quarters_l177_177829


namespace tens_digit_of_23_pow_1987_l177_177720

theorem tens_digit_of_23_pow_1987 : (23 ^ 1987 % 100 / 10) % 10 = 4 :=
by
  sorry

end tens_digit_of_23_pow_1987_l177_177720


namespace relay_race_total_time_l177_177753

noncomputable def mary_time (susan_time : ℕ) : ℕ := 2 * susan_time
noncomputable def susan_time (jen_time : ℕ) : ℕ := jen_time + 10
def jen_time : ℕ := 30
noncomputable def tiffany_time (mary_time : ℕ) : ℕ := mary_time - 7

theorem relay_race_total_time :
  let mary_time := mary_time (susan_time jen_time)
  let susan_time := susan_time jen_time
  let tiffany_time := tiffany_time mary_time
  mary_time + susan_time + jen_time + tiffany_time = 223 := by
  sorry

end relay_race_total_time_l177_177753


namespace range_of_a_l177_177509

-- Function definition
def f (x a : ℝ) : ℝ := -x^3 + 3 * a^2 * x - 4 * a

-- Main theorem statement
theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x, f x a = 0) ↔ (a ∈ Set.Ioi (Real.sqrt 2)) :=
sorry

end range_of_a_l177_177509


namespace expected_value_is_90_l177_177159

noncomputable def expected_value_coins_heads : ℕ :=
  let nickel := 5
  let quarter := 25
  let half_dollar := 50
  let dollar := 100
  1/2 * (nickel + quarter + half_dollar + dollar)

theorem expected_value_is_90 : expected_value_coins_heads = 90 := by
  sorry

end expected_value_is_90_l177_177159


namespace units_digit_is_seven_l177_177235

-- Defining the structure of the three-digit number and its properties
def original_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
def reversed_number (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

def four_times_original (a b c : ℕ) : ℕ := 4 * original_number a b c
def subtract_reversed (a b c : ℕ) : ℕ := four_times_original a b c - reversed_number a b c

-- Theorem statement: Given the condition, what is the units digit of the result?
theorem units_digit_is_seven (a b c : ℕ) (h : a = c + 3) : (subtract_reversed a b c) % 10 = 7 :=
by
  sorry

end units_digit_is_seven_l177_177235


namespace vanAubel_theorem_l177_177332

variables (A B C O A1 B1 C1 : Type)
variables (CA1 A1B CB1 B1A CO OC1 : ℝ)

-- Given Conditions
axiom condition1 : CB1 / B1A = 1
axiom condition2 : CO / OC1 = 2

-- Van Aubel's theorem statement
theorem vanAubel_theorem : (CO / OC1) = (CA1 / A1B) + (CB1 / B1A) := by
  sorry

end vanAubel_theorem_l177_177332


namespace find_min_y_l177_177019

theorem find_min_y (x y : ℕ) (hx : x = y + 8) 
    (h : Nat.gcd ((x^3 + y^3) / (x + y)) (x * y) = 16) : 
    y = 4 :=
sorry

end find_min_y_l177_177019


namespace wedge_product_correct_l177_177194

variables {a1 a2 b1 b2 : ℝ}
def a : ℝ × ℝ := (a1, a2)
def b : ℝ × ℝ := (b1, b2)

def wedge_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.2 - v.2 * w.1

theorem wedge_product_correct (a b : ℝ × ℝ) :
  wedge_product a b = a.1 * b.2 - a.2 * b.1 :=
by
  -- Proof is omitted, theorem statement only
  sorry

end wedge_product_correct_l177_177194


namespace hiking_trip_rate_ratio_l177_177685

theorem hiking_trip_rate_ratio 
  (rate_up : ℝ) (time_up : ℝ) (distance_down : ℝ) (time_down : ℝ)
  (h1 : rate_up = 7) 
  (h2 : time_up = 2) 
  (h3 : distance_down = 21) 
  (h4 : time_down = 2) : 
  (distance_down / time_down) / rate_up = 1.5 :=
by
  -- skip the proof as per instructions
  sorry

end hiking_trip_rate_ratio_l177_177685


namespace number_of_pink_cookies_l177_177114

def total_cookies : ℕ := 86
def red_cookies : ℕ := 36

def pink_cookies (total red : ℕ) : ℕ := total - red

theorem number_of_pink_cookies : pink_cookies total_cookies red_cookies = 50 :=
by
  sorry

end number_of_pink_cookies_l177_177114


namespace volume_of_cube_l177_177409

-- Define the conditions
def surface_area (a : ℝ) : ℝ := 6 * a^2
def side_length (a : ℝ) (SA : ℝ) : Prop := SA = 6 * a^2
def volume (a : ℝ) : ℝ := a^3

-- State the theorem
theorem volume_of_cube (a : ℝ) (SA : surface_area a = 150) : volume a = 125 := 
sorry

end volume_of_cube_l177_177409


namespace inscribed_circle_radius_l177_177524

variable (AB AC BC : ℝ) (r : ℝ)

theorem inscribed_circle_radius 
  (h1 : AB = 9) 
  (h2 : AC = 9) 
  (h3 : BC = 8) : r = (4 * Real.sqrt 65) / 13 := 
sorry

end inscribed_circle_radius_l177_177524


namespace smoking_lung_disease_confidence_l177_177290

/-- Prove that given the conditions, the correct statement is C:
   If it is concluded from the statistic that there is a 95% confidence 
   that smoking is related to lung disease, then there is a 5% chance of
   making a wrong judgment. -/
theorem smoking_lung_disease_confidence 
  (P Q : Prop) 
  (confidence_level : ℝ) 
  (h_conf : confidence_level = 0.95) 
  (h_PQ : P → (Q → true)) :
  ¬Q → (confidence_level = 1 - 0.05) :=
by
  sorry

end smoking_lung_disease_confidence_l177_177290


namespace monthly_expenses_last_month_was_2888_l177_177884

def basic_salary : ℕ := 1250
def commission_rate : ℚ := 0.10
def total_sales : ℕ := 23600
def savings_rate : ℚ := 0.20

theorem monthly_expenses_last_month_was_2888 :
  let commission := commission_rate * total_sales
  let total_earnings := basic_salary + commission
  let savings := savings_rate * total_earnings
  let monthly_expenses := total_earnings - savings
  monthly_expenses = 2888 := by
  sorry

end monthly_expenses_last_month_was_2888_l177_177884


namespace cost_price_of_article_l177_177704

theorem cost_price_of_article (x : ℝ) (h : 57 - x = x - 43) : x = 50 :=
by sorry

end cost_price_of_article_l177_177704


namespace problem1_problem2_l177_177664

-- Problem 1
theorem problem1 (a b : ℝ) (h : 2 * (a + 1) * (b + 1) = (a + b) * (a + b + 2)) : a^2 + b^2 = 2 := sorry

-- Problem 2
theorem problem2 (a b c : ℝ) (h : a^2 + c^2 = 2 * b^2) : (a + b) * (a + c) + (c + a) * (c + b) = 2 * (b + a) * (b + c) := sorry

end problem1_problem2_l177_177664


namespace arithmetic_sequence_general_term_l177_177372

theorem arithmetic_sequence_general_term
    (a : ℕ → ℤ)
    (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
    (h_mean_26 : (a 2 + a 6) / 2 = 5)
    (h_mean_37 : (a 3 + a 7) / 2 = 7) :
    ∀ n, a n = 2 * n - 3 := 
by
  sorry

end arithmetic_sequence_general_term_l177_177372


namespace find_m_independent_quadratic_term_l177_177395

def quadratic_poly (m : ℝ) (x : ℝ) : ℝ :=
  -3 * x^2 + m * x^2 - x + 3

theorem find_m_independent_quadratic_term (m : ℝ) :
  (∀ x, quadratic_poly m x = -x + 3) → m = 3 :=
by 
  sorry

end find_m_independent_quadratic_term_l177_177395


namespace weight_of_dry_grapes_l177_177933

theorem weight_of_dry_grapes (w_fresh : ℝ) (perc_water_fresh perc_water_dried : ℝ) (w_non_water : ℝ) (w_dry : ℝ) :
  w_fresh = 5 →
  perc_water_fresh = 0.90 →
  perc_water_dried = 0.20 →
  w_non_water = w_fresh * (1 - perc_water_fresh) →
  w_non_water = w_dry * (1 - perc_water_dried) →
  w_dry = 0.625 :=
by sorry

end weight_of_dry_grapes_l177_177933


namespace additional_dividend_amount_l177_177448

theorem additional_dividend_amount
  (E : ℝ) (Q : ℝ) (expected_extra_per_earnings : ℝ) (half_of_extra_per_earnings_to_dividend : ℝ) 
  (expected : E = 0.80) (quarterly_earnings : Q = 1.10)
  (extra_per_earnings : expected_extra_per_earnings = 0.30)
  (half_dividend : half_of_extra_per_earnings_to_dividend = 0.15):
  Q - E = expected_extra_per_earnings ∧ 
  expected_extra_per_earnings / 2 = half_of_extra_per_earnings_to_dividend :=
by sorry

end additional_dividend_amount_l177_177448


namespace well_depth_is_2000_l177_177848

-- Given conditions
def total_time : ℝ := 10
def stone_law (t₁ : ℝ) : ℝ := 20 * t₁^2
def sound_velocity : ℝ := 1120

-- Statement to be proven
theorem well_depth_is_2000 :
  ∃ (d t₁ t₂ : ℝ), 
    d = stone_law t₁ ∧ t₂ = d / sound_velocity ∧ t₁ + t₂ = total_time :=
sorry

end well_depth_is_2000_l177_177848


namespace tan_A_value_l177_177918

open Real

theorem tan_A_value (A : ℝ) (h1 : sin A * (sin A + sqrt 3 * cos A) = -1 / 2) (h2 : 0 < A ∧ A < π) :
  tan A = -sqrt 3 / 3 :=
sorry

end tan_A_value_l177_177918


namespace max_value_of_expr_l177_177453

open Classical
open Real

theorem max_value_of_expr 
  (x y : ℝ) 
  (h₁ : 0 < x) 
  (h₂ : 0 < y) 
  (h₃ : x^2 - 2 * x * y + 3 * y^2 = 10) : 
  ∃ a b c d : ℝ, 
    (x^2 + 2 * x * y + 3 * y^2 = 20 + 10 * sqrt 3) ∧ 
    (a = 20) ∧ 
    (b = 10) ∧ 
    (c = 3) ∧ 
    (d = 2) := 
sorry

end max_value_of_expr_l177_177453


namespace total_road_length_l177_177254

theorem total_road_length (L : ℚ) : (1/3) * L + (2/5) * (2/3) * L = 135 → L = 225 := 
by
  intro h
  sorry

end total_road_length_l177_177254


namespace cuberoot_condition_l177_177088

/-- If \(\sqrt[3]{x-1}=3\), then \((x-1)^2 = 729\). -/
theorem cuberoot_condition (x : ℝ) (h : (x - 1)^(1/3) = 3) : (x - 1)^2 = 729 := 
  sorry

end cuberoot_condition_l177_177088


namespace negation_exists_equiv_forall_l177_177821

theorem negation_exists_equiv_forall :
  (¬ (∃ x : ℤ, x^2 + 2*x - 1 < 0)) ↔ (∀ x : ℤ, x^2 + 2*x - 1 ≥ 0) :=
by
  sorry

end negation_exists_equiv_forall_l177_177821


namespace Mike_given_total_cookies_l177_177128

-- All given conditions
variables (total Tim fridge Mike Anna : Nat)
axiom h1 : total = 256
axiom h2 : Tim = 15
axiom h3 : fridge = 188
axiom h4 : Anna = 2 * Tim
axiom h5 : total = Tim + Anna + fridge + Mike

-- The goal of the proof
theorem Mike_given_total_cookies : Mike = 23 :=
by
  sorry

end Mike_given_total_cookies_l177_177128


namespace arithmetic_sequence_a3_l177_177080

theorem arithmetic_sequence_a3 :
  ∃ (a : ℕ → ℝ) (d : ℝ), 
    (∀ n, a n = 2 + (n - 1) * d) ∧
    (a 1 = 2) ∧
    (a 5 = a 4 + 2) →
    a 3 = 6 :=
sorry

end arithmetic_sequence_a3_l177_177080


namespace simplify_and_sum_coefficients_l177_177173

theorem simplify_and_sum_coefficients :
  (∃ A B C D : ℤ, (∀ x : ℝ, x ≠ D → (x^3 + 6 * x^2 + 11 * x + 6) / (x + 1) = A * x^2 + B * x + C) ∧ A + B + C + D = 11) :=
sorry

end simplify_and_sum_coefficients_l177_177173


namespace probability_of_forming_triangle_l177_177900

def segment_lengths : List ℕ := [1, 3, 5, 7, 9]
def valid_combinations : List (ℕ × ℕ × ℕ) := [(3, 5, 7), (3, 7, 9), (5, 7, 9)]
def total_combinations := Nat.choose 5 3

theorem probability_of_forming_triangle :
  (valid_combinations.length : ℚ) / total_combinations = 3 / 10 := 
by
  sorry

end probability_of_forming_triangle_l177_177900


namespace min_transport_cost_l177_177281

-- Definitions based on conditions
def total_washing_machines : ℕ := 100
def typeA_max_count : ℕ := 4
def typeB_max_count : ℕ := 8
def typeA_cost : ℕ := 400
def typeA_capacity : ℕ := 20
def typeB_cost : ℕ := 300
def typeB_capacity : ℕ := 10

-- Minimum transportation cost calculation
def min_transportation_cost : ℕ :=
  let typeA_trucks_used := min typeA_max_count (total_washing_machines / typeA_capacity)
  let remaining_washing_machines := total_washing_machines - typeA_trucks_used * typeA_capacity
  let typeB_trucks_used := min typeB_max_count (remaining_washing_machines / typeB_capacity)
  typeA_trucks_used * typeA_cost + typeB_trucks_used * typeB_cost

-- Lean 4 statement to prove the minimum transportation cost
theorem min_transport_cost : min_transportation_cost = 2200 := by
  sorry

end min_transport_cost_l177_177281


namespace car_distance_after_y_begins_l177_177628

theorem car_distance_after_y_begins (v_x v_y : ℝ) (t_y_start t_x_after_y : ℝ) (d_x_before_y : ℝ) :
  v_x = 35 → v_y = 50 → t_y_start = 1.2 → d_x_before_y = v_x * t_y_start → t_x_after_y = 2.8 →
  (d_x_before_y + v_x * t_x_after_y = 98) :=
by
  intros h_vx h_vy h_ty_start h_dxbefore h_txafter
  simp [h_vx, h_vy, h_ty_start, h_dxbefore, h_txafter]
  sorry

end car_distance_after_y_begins_l177_177628


namespace actual_time_before_storm_l177_177620

-- Define valid hour digit ranges before the storm
def valid_first_digit (d : ℕ) : Prop := d = 1 ∨ d = 2 ∨ d = 3
def valid_second_digit (d : ℕ) : Prop := d = 9 ∨ d = 0 ∨ d = 1

-- Define valid minute digit ranges before the storm
def valid_third_digit (d : ℕ) : Prop := d = 4 ∨ d = 5 ∨ d = 6
def valid_fourth_digit (d : ℕ) : Prop := d = 9 ∨ d = 0 ∨ d = 1

-- Define a valid time in HH:MM format
def valid_time (hh mm : ℕ) : Prop :=
  hh < 24 ∧ mm < 60

-- The proof problem
theorem actual_time_before_storm (hh hh' mm mm' : ℕ) 
  (h1 : valid_first_digit hh) (h2 : valid_second_digit hh') 
  (h3 : valid_third_digit mm) (h4 : valid_fourth_digit mm') 
  (h_valid : valid_time (hh * 10 + hh') (mm * 10 + mm')) 
  (h_display : (hh + 1) * 10 + (hh' - 1) = 20 ∧ (mm + 1) * 10 + (mm' - 1) = 50) :
  hh * 10 + hh' = 19 ∧ mm * 10 + mm' = 49 :=
by
  sorry

end actual_time_before_storm_l177_177620


namespace geese_count_l177_177223

variables (k n : ℕ)

theorem geese_count (h1 : k * n = (k + 20) * (n - 75)) (h2 : k * n = (k - 15) * (n + 100)) : n = 300 :=
by
  sorry

end geese_count_l177_177223


namespace Z_real_Z_imaginary_Z_pure_imaginary_l177_177687

-- Definitions

def Z (a : ℝ) : ℂ := (a^2 - 9 : ℝ) + (a^2 - 2 * a - 15 : ℂ)

-- Statement for the proof problems

theorem Z_real (a : ℝ) : 
  (Z a).im = 0 ↔ a = 5 ∨ a = -3 := sorry

theorem Z_imaginary (a : ℝ) : 
  (Z a).re = 0 ↔ a ≠ 5 ∧ a ≠ -3 := sorry

theorem Z_pure_imaginary (a : ℝ) : 
  (Z a).re = 0 ∧ (Z a).im ≠ 0 ↔ a = 3 := sorry

end Z_real_Z_imaginary_Z_pure_imaginary_l177_177687


namespace train_length_proof_l177_177230

-- Define the conditions
def train_speed_kmph := 72
def platform_length_m := 290
def crossing_time_s := 26

-- Conversion factor
def kmph_to_mps := 5 / 18

-- Convert speed to m/s
def train_speed_mps := train_speed_kmph * kmph_to_mps

-- Distance covered by train while crossing the platform (in meters)
def distance_covered := train_speed_mps * crossing_time_s

-- Length of the train (in meters)
def train_length := distance_covered - platform_length_m

-- The theorem to be proved
theorem train_length_proof : train_length = 230 :=
by 
  -- proof would be placed here 
  sorry

end train_length_proof_l177_177230


namespace eval_expression_l177_177084

theorem eval_expression : 
  (8^5) / (4 * 2^5 + 16) = 2^11 / 9 :=
by
  sorry

end eval_expression_l177_177084


namespace negate_statement_l177_177105

variable (Students Teachers : Type)
variable (Patient : Students → Prop)
variable (PatientT : Teachers → Prop)
variable (a : ∀ t : Teachers, PatientT t)
variable (b : ∃ t : Teachers, PatientT t)
variable (c : ∀ s : Students, ¬ Patient s)
variable (d : ∀ s : Students, ¬ Patient s)
variable (e : ∃ s : Students, ¬ Patient s)
variable (f : ∀ s : Students, Patient s)

theorem negate_statement : (∃ s : Students, ¬ Patient s) ↔ ¬ (∀ s : Students, Patient s) :=
by sorry

end negate_statement_l177_177105


namespace tan_pi_div_4_sub_theta_l177_177666

theorem tan_pi_div_4_sub_theta (theta : ℝ) (h : Real.tan theta = 1 / 2) : 
  Real.tan (π / 4 - theta) = 1 / 3 := 
sorry

end tan_pi_div_4_sub_theta_l177_177666


namespace certain_number_sum_421_l177_177903

theorem certain_number_sum_421 :
  ∃ n, (∃ k, n = 423 * k) ∧ k = 2 →
  n + 421 = 1267 :=
by
  sorry

end certain_number_sum_421_l177_177903


namespace amoebas_after_ten_days_l177_177707

def amoeba_split_fun (n : Nat) : Nat := 3^n

theorem amoebas_after_ten_days : amoeba_split_fun 10 = 59049 := by
  have h : 3 ^ 10 = 59049 := by norm_num
  exact h

end amoebas_after_ten_days_l177_177707


namespace assertion1_false_assertion2_true_assertion3_false_assertion4_false_l177_177717

section

-- Assertion 1: ∀ x ∈ ℝ, x ≥ 1 is false
theorem assertion1_false : ¬(∀ x : ℝ, x ≥ 1) := 
sorry

-- Assertion 2: ∃ x ∈ ℕ, x ∈ ℝ is true
theorem assertion2_true : ∃ x : ℕ, (x : ℝ) = x := 
sorry

-- Assertion 3: ∀ x ∈ ℝ, x > 2 → x ≥ 3 is false
theorem assertion3_false : ¬(∀ x : ℝ, x > 2 → x ≥ 3) := 
sorry

-- Assertion 4: ∃ n ∈ ℤ, ∀ x ∈ ℝ, n ≤ x < n + 1 is false
theorem assertion4_false : ¬(∃ n : ℤ, ∀ x : ℝ, n ≤ x ∧ x < n + 1) := 
sorry

end

end assertion1_false_assertion2_true_assertion3_false_assertion4_false_l177_177717


namespace unit_digit_of_15_pow_l177_177420

-- Define the conditions
def base_number : ℕ := 15
def base_unit_digit : ℕ := 5

-- State the question and objective in Lean 4
theorem unit_digit_of_15_pow (X : ℕ) (h : 0 < X) : (15^X) % 10 = 5 :=
sorry

end unit_digit_of_15_pow_l177_177420


namespace find_complex_Z_l177_177195

open Complex

theorem find_complex_Z (Z : ℂ) (h : (2 + 4 * I) / Z = 1 - I) : 
  Z = -1 + 3 * I :=
by
  sorry

end find_complex_Z_l177_177195


namespace width_of_park_l177_177615

theorem width_of_park (L : ℕ) (A_lawn : ℕ) (w_road : ℕ) (W : ℚ) :
  L = 60 → A_lawn = 2109 → w_road = 3 →
  60 * W - 2 * 60 * 3 = 2109 →
  W = 41.15 :=
by
  intros hL hA_lawn hw_road hEq
  -- The proof will go here
  sorry

end width_of_park_l177_177615


namespace taco_truck_profit_l177_177617

-- Definitions and conditions
def pounds_of_beef : ℕ := 100
def beef_per_taco : ℝ := 0.25
def price_per_taco : ℝ := 2
def cost_per_taco : ℝ := 1.5

-- Desired profit result
def expected_profit : ℝ := 200

-- The proof statement (to be completed)
theorem taco_truck_profit :
  let tacos := pounds_of_beef / beef_per_taco;
  let revenue := tacos * price_per_taco;
  let cost := tacos * cost_per_taco;
  let profit := revenue - cost;
  profit = expected_profit :=
by
  sorry

end taco_truck_profit_l177_177617


namespace factor_product_l177_177728

theorem factor_product : 2^2 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end factor_product_l177_177728


namespace find_prime_pair_l177_177754

noncomputable def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def has_integer_root (p q : ℕ) : Prop :=
  ∃ x : ℤ, x^4 + p * x^3 - q = 0

theorem find_prime_pair :
  ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ has_integer_root p q ∧ p = 2 ∧ q = 3 := by
  sorry

end find_prime_pair_l177_177754


namespace average_of_a_and_b_l177_177181

theorem average_of_a_and_b (a b c : ℝ) 
  (h₁ : (b + c) / 2 = 90)
  (h₂ : c - a = 90) :
  (a + b) / 2 = 45 :=
sorry

end average_of_a_and_b_l177_177181


namespace initial_speed_100kmph_l177_177812

theorem initial_speed_100kmph (v x : ℝ) (h1 : 0 < v) (h2 : 100 - x = v / 2) 
  (h3 : (80 - x) / (v - 10) - 20 / (v - 20) = 1 / 12) : v = 100 :=
by 
  sorry

end initial_speed_100kmph_l177_177812


namespace part1_part2_l177_177280

open Set

def A (a : ℝ) : Set ℝ := { x | x^2 - a * x + a^2 - 19 = 0 }
def B : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def C : Set ℝ := { x | x^2 + 2 * x - 8 = 0 }

theorem part1 (a : ℝ) : (A a ∩ B = A a ∪ B) → a = 5 :=
by
  sorry

theorem part2 (a : ℝ) : (A a ∩ B ≠ ∅) ∧ (A a ∩ C = ∅) → a = -2 :=
by
  sorry

end part1_part2_l177_177280


namespace simplify_expression_l177_177189

theorem simplify_expression (a : ℝ) (h : a ≠ 2) : 
  (a^2 / (a - 2)) - (4 / (a - 2)) = a + 2 :=
by 
  sorry

end simplify_expression_l177_177189


namespace quadratic_solution_unique_l177_177242

noncomputable def solve_quad_eq (a : ℝ) (h : a ≠ 0) (h_uniq : (36^2 - 4 * a * 12) = 0) : ℝ :=
-2 / 3

theorem quadratic_solution_unique (a : ℝ) (h : a ≠ 0) (h_uniq : (36^2 - 4 * a * 12) = 0) :
  (∃! x : ℝ, a * x^2 + 36 * x + 12 = 0) ∧ (solve_quad_eq a h h_uniq) = -2 / 3 :=
by
  sorry

end quadratic_solution_unique_l177_177242


namespace arithmetic_sequence_common_difference_l177_177769

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) :
  (a 5 = 8) → (a 1 + a 2 + a 3 = 6) → (∀ n : ℕ, a (n + 1) = a 1 + n * d) → d = 2 :=
by
  intros ha5 hsum harr
  sorry

end arithmetic_sequence_common_difference_l177_177769


namespace penny_canoe_l177_177295

theorem penny_canoe (P : ℕ)
  (h1 : 140 * (2/3 : ℚ) * P + 35 = 595) : P = 6 :=
sorry

end penny_canoe_l177_177295


namespace M_Mobile_cheaper_than_T_Mobile_l177_177431

def T_Mobile_total_cost (lines : ℕ) : ℕ :=
  if lines <= 2 then 50
  else 50 + (lines - 2) * 16

def M_Mobile_total_cost (lines : ℕ) : ℕ :=
  if lines <= 2 then 45
  else 45 + (lines - 2) * 14

theorem M_Mobile_cheaper_than_T_Mobile : 
  T_Mobile_total_cost 5 - M_Mobile_total_cost 5 = 11 :=
by
  sorry

end M_Mobile_cheaper_than_T_Mobile_l177_177431


namespace comm_ring_of_center_condition_l177_177385

variable {R : Type*} [Ring R]

def in_center (x : R) : Prop := ∀ y : R, (x * y = y * x)

def is_commutative (R : Type*) [Ring R] : Prop := ∀ a b : R, a * b = b * a

theorem comm_ring_of_center_condition (h : ∀ x : R, in_center (x^2 - x)) : is_commutative R :=
sorry

end comm_ring_of_center_condition_l177_177385


namespace range_of_m_l177_177670

def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (m : ℝ) (h : ¬ (p m ∨ q m)) : m ≥ 2 :=
by
  sorry

end range_of_m_l177_177670


namespace range_of_a_l177_177032

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (0 ≤ a ∧ a < 3) :=
sorry

end range_of_a_l177_177032


namespace non_degenerate_ellipse_l177_177800

theorem non_degenerate_ellipse (k : ℝ) : 
    (∃ x y : ℝ, x^2 + 9 * y^2 - 6 * x + 18 * y = k) ↔ k > -18 :=
sorry

end non_degenerate_ellipse_l177_177800


namespace prime_quadruple_solution_l177_177994

-- Define the problem statement in Lean
theorem prime_quadruple_solution :
  ∀ (p q r : ℕ) (n : ℕ),
    Prime p → Prime q → Prime r → n > 0 →
    p^2 = q^2 + r^n →
    (p = 3 ∧ q = 2 ∧ r = 5 ∧ n = 1) ∨ (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 4) :=
by
  sorry -- Proof omitted

end prime_quadruple_solution_l177_177994


namespace real_solution_count_l177_177794

/-- Given \( \lfloor x \rfloor \) is the greatest integer less than or equal to \( x \),
prove that the number of real solutions to the equation \( 9x^2 - 36\lfloor x \rfloor + 20 = 0 \) is 2. --/
theorem real_solution_count (x : ℝ) (h : ⌊x⌋ = Int.floor x) :
  ∃ (S : Finset ℝ), S.card = 2 ∧ ∀ a ∈ S, 9 * a^2 - 36 * ⌊a⌋ + 20 = 0 :=
sorry

end real_solution_count_l177_177794


namespace problem_l177_177135

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function definition

theorem problem :
  (∀ x, f (-x) = -f x) → -- f is odd
  (∀ x, f (x + 2) = -1 / f x) → -- Functional equation
  (∀ x, 0 < x ∧ x < 1 → f x = 3 ^ x) → -- Definition on interval (0,1)
  f (Real.log (54) / Real.log 3) = -3 / 2 := sorry

end problem_l177_177135


namespace triangle_side_length_l177_177585

variables {BC AC : ℝ} {α β γ : ℝ}

theorem triangle_side_length :
  α = 45 ∧ β = 75 ∧ AC = 6 ∧ α + β + γ = 180 →
  BC = 6 * (Real.sqrt 3 - 1) :=
by
  intros h
  sorry

end triangle_side_length_l177_177585


namespace francie_remaining_money_l177_177042

noncomputable def total_savings_before_investment : ℝ :=
  (5 * 8) + (6 * 6) + 20

noncomputable def investment_return : ℝ :=
  0.05 * 10

noncomputable def total_savings_after_investment : ℝ :=
  total_savings_before_investment + investment_return

noncomputable def spent_on_clothes : ℝ :=
  total_savings_after_investment / 2

noncomputable def remaining_after_clothes : ℝ :=
  total_savings_after_investment - spent_on_clothes

noncomputable def amount_remaining : ℝ :=
  remaining_after_clothes - 35

theorem francie_remaining_money : amount_remaining = 13.25 := 
  sorry

end francie_remaining_money_l177_177042


namespace sequence_formula_l177_177164

theorem sequence_formula (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 3)
  (h3 : a 3 = 6)
  (h4 : a 4 = 10)
  (h5 : ∀ n > 0, a (n + 1) - a n = n + 1) :
  ∀ n, a n = n * (n + 1) / 2 :=
by 
  sorry

end sequence_formula_l177_177164


namespace trigonometric_expression_eval_l177_177634

-- Conditions
variable (α : Real) (h1 : ∃ x : Real, 3 * x^2 - x - 2 = 0 ∧ x = Real.cos α) (h2 : α > π ∧ α < 3 * π / 2)

-- Question and expected answer
theorem trigonometric_expression_eval :
  (Real.sin (-α + 3 * π / 2) * Real.cos (3 * π / 2 + α) * Real.tan (π - α)^2) /
  (Real.cos (π / 2 + α) * Real.sin (π / 2 - α)) = 5 / 4 := sorry

end trigonometric_expression_eval_l177_177634


namespace unique_solution_for_k_l177_177302

theorem unique_solution_for_k : 
  ∃! k : ℚ, k ≠ 0 ∧ (∀ x : ℚ, (x + 3) / (k * x - 2) = x ↔ x = -2) :=
by
  sorry

end unique_solution_for_k_l177_177302


namespace triangle_sides_consecutive_obtuse_l177_177763

/-- Given the sides of a triangle are consecutive natural numbers 
    and the largest angle is obtuse, 
    the lengths of the sides in ascending order are 2, 3, 4. -/
theorem triangle_sides_consecutive_obtuse 
    (x : ℕ) (hx : x > 1) 
    (cos_alpha_neg : (x - 4) < 0) 
    (x_lt_4 : x < 4) :
    (x = 3) → (∃ a b c : ℕ, a < b ∧ b < c ∧ a + b > c ∧ a = 2 ∧ b = 3 ∧ c = 4) :=
by
  intro hx3
  use 2, 3, 4
  repeat {split}
  any_goals {linarith}
  all_goals {sorry}

end triangle_sides_consecutive_obtuse_l177_177763


namespace if_a_gt_abs_b_then_a2_gt_b2_l177_177667

theorem if_a_gt_abs_b_then_a2_gt_b2 (a b : ℝ) (h : a > abs b) : a^2 > b^2 :=
by sorry

end if_a_gt_abs_b_then_a2_gt_b2_l177_177667


namespace student_distribution_l177_177847

theorem student_distribution (a b : ℕ) (h1 : a + b = 81) (h2 : a = b - 9) : a = 36 ∧ b = 45 := 
by
  sorry

end student_distribution_l177_177847


namespace otimes_h_h_h_eq_h_l177_177008

variable (h : ℝ)

def otimes (x y : ℝ) : ℝ := x^3 - y

theorem otimes_h_h_h_eq_h : otimes h (otimes h h) = h := by
  -- Proof goes here, but is omitted
  sorry

end otimes_h_h_h_eq_h_l177_177008


namespace maximum_height_when_isosceles_l177_177489

variable (c : ℝ) (c1 c2 : ℝ)

def right_angled_triangle (c1 c2 c : ℝ) : Prop :=
  c1 * c1 + c2 * c2 = c * c

def isosceles_right_triangle (c1 c2 : ℝ) : Prop :=
  c1 = c2

noncomputable def height_relative_to_hypotenuse (c : ℝ) : ℝ :=
  c / 2

theorem maximum_height_when_isosceles 
  (c1 c2 c : ℝ) 
  (h_right : right_angled_triangle c1 c2 c) 
  (h_iso : isosceles_right_triangle c1 c2) :
  height_relative_to_hypotenuse c = c / 2 :=
  sorry

end maximum_height_when_isosceles_l177_177489


namespace tan_seventeen_pi_over_four_l177_177697

theorem tan_seventeen_pi_over_four : Real.tan (17 * Real.pi / 4) = 1 := by
  sorry

end tan_seventeen_pi_over_four_l177_177697


namespace inscribed_circle_diameter_l177_177095

theorem inscribed_circle_diameter (PQ PR QR : ℝ) (h₁ : PQ = 13) (h₂ : PR = 14) (h₃ : QR = 15) :
  ∃ d : ℝ, d = 8 :=
by
  sorry

end inscribed_circle_diameter_l177_177095


namespace largest_of_five_consecutive_integers_with_product_15120_eq_9_l177_177384

theorem largest_of_five_consecutive_integers_with_product_15120_eq_9 :
  ∃ n : ℕ, (n + 0) * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 15120 ∧ n + 4 = 9 :=
by
  sorry

end largest_of_five_consecutive_integers_with_product_15120_eq_9_l177_177384


namespace tan_45_degrees_l177_177497

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l177_177497


namespace num_sets_B_l177_177897

open Set

theorem num_sets_B (A B : Set ℕ) (hA : A = {1, 2}) (h_union : A ∪ B = {1, 2, 3}) : ∃ n, n = 4 :=
by
  sorry

end num_sets_B_l177_177897


namespace parabola_x_intercept_y_intercept_point_l177_177564

theorem parabola_x_intercept_y_intercept_point (a b w : ℝ) 
  (h1 : a = -1) 
  (h2 : b = 4) 
  (h3 : ∀ x : ℝ, x = 0 → w = 8): 
  ∃ (w : ℝ), w = 8 := 
by
  sorry

end parabola_x_intercept_y_intercept_point_l177_177564


namespace find_sum_of_a_b_c_l177_177001

def a := 8
def b := 2
def c := 2

theorem find_sum_of_a_b_c : a + b + c = 12 :=
by
  have ha : a = 8 := rfl
  have hb : b = 2 := rfl
  have hc : c = 2 := rfl
  sorry

end find_sum_of_a_b_c_l177_177001


namespace Z_is_divisible_by_10001_l177_177317

theorem Z_is_divisible_by_10001
    (Z : ℕ) (a b c d : ℕ) (ha : a ≠ 0)
    (hZ : Z = 1000 * 10001 * a + 100 * 10001 * b + 10 * 10001 * c + 10001 * d)
    : 10001 ∣ Z :=
by {
    -- Proof omitted
    sorry
}

end Z_is_divisible_by_10001_l177_177317


namespace inequality_satisfied_equality_condition_l177_177268

theorem inequality_satisfied (x y : ℝ) : x^2 + y^2 + 1 ≥ 2 * (x * y - x + y) :=
sorry

theorem equality_condition (x y : ℝ) : (x^2 + y^2 + 1 = 2 * (x * y - x + y)) ↔ (x = y - 1) :=
sorry

end inequality_satisfied_equality_condition_l177_177268


namespace find_a_l177_177751

-- Define the lines as given
def line1 (x y : ℝ) := 2 * x + y - 5 = 0
def line2 (x y : ℝ) := x - y - 1 = 0
def line3 (a x y : ℝ) := a * x + y - 3 = 0

-- Define the condition that they intersect at a single point
def lines_intersect_at_point (x y a : ℝ) := line1 x y ∧ line2 x y ∧ line3 a x y

-- To prove: If lines intersect at a certain point, then a = 1
theorem find_a (a : ℝ) : (∃ x y, lines_intersect_at_point x y a) → a = 1 :=
by
  sorry

end find_a_l177_177751


namespace door_height_eight_l177_177166

theorem door_height_eight (x : ℝ) (h w : ℝ) (H1 : w = x - 4) (H2 : h = x - 2) (H3 : x^2 = (x - 4)^2 + (x - 2)^2) : h = 8 :=
by
  sorry

end door_height_eight_l177_177166


namespace find_increase_x_l177_177906

noncomputable def initial_radius : ℝ := 7
noncomputable def initial_height : ℝ := 5
variable (x : ℝ)

theorem find_increase_x (hx : x > 0)
  (volume_eq : π * (initial_radius + x) ^ 2 * initial_height =
               π * initial_radius ^ 2 * (initial_height + 2 * x)) :
  x = 28 / 5 :=
by
  sorry

end find_increase_x_l177_177906


namespace sequence_converges_and_limit_l177_177681

theorem sequence_converges_and_limit {a : ℝ} (m : ℕ) (h_a_pos : 0 < a) (h_m_pos : 0 < m) :
  (∃ (x : ℕ → ℝ), 
  (x 1 = 1) ∧ 
  (x 2 = a) ∧ 
  (∀ n : ℕ, x (n + 2) = (x (n + 1) ^ m * x n) ^ (↑(1 : ℕ) / (m + 1))) ∧ 
  ∃ l : ℝ, (∀ ε > 0, ∃ N, ∀ n > N, |x n - l| < ε) ∧ l = a ^ (↑(m + 1) / ↑(m + 2))) :=
sorry

end sequence_converges_and_limit_l177_177681


namespace percentage_increase_expenditure_l177_177215

variable (I : ℝ) -- original income
variable (E : ℝ) -- original expenditure
variable (I_new : ℝ) -- new income
variable (S : ℝ) -- original savings
variable (S_new : ℝ) -- new savings

-- a) Conditions
def initial_spend (I : ℝ) : ℝ := 0.75 * I
def income_increased (I : ℝ) : ℝ := 1.20 * I
def savings_increased (S : ℝ) : ℝ := 1.4999999999999996 * S

-- b) Definitions relating formulated conditions
def new_expenditure (I : ℝ) : ℝ := 1.20 * I - 0.3749999999999999 * I
def original_expenditure (I : ℝ) : ℝ := 0.75 * I

-- c) Proof statement
theorem percentage_increase_expenditure :
  initial_spend I = E →
  income_increased I = I_new →
  savings_increased (0.25 * I) = S_new →
  ((new_expenditure I - original_expenditure I) / original_expenditure I) * 100 = 10 := 
by 
  intros h1 h2 h3
  sorry

end percentage_increase_expenditure_l177_177215


namespace intersection_nonempty_l177_177793

open Nat

theorem intersection_nonempty (a : ℕ) (ha : a ≥ 2) :
  ∃ (b : ℕ), b = 1 ∨ b = a ∧
  ∃ y, (∃ x, y = a^x ∧ x ≥ 1) ∧
       (∃ x, y = (a + 1)^x + b ∧ x ≥ 1) :=
by sorry

end intersection_nonempty_l177_177793


namespace largest_two_digit_divisible_by_6_ending_in_4_l177_177890

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, is_two_digit n ∧ is_divisible_by_6 n ∧ ends_in_4 n ∧
  ∀ m : ℕ, is_two_digit m ∧ is_divisible_by_6 m ∧ ends_in_4 m → m ≤ n :=
  sorry

end largest_two_digit_divisible_by_6_ending_in_4_l177_177890


namespace ferris_wheel_capacity_l177_177112

theorem ferris_wheel_capacity :
  let num_seats := 4
  let people_per_seat := 4
  num_seats * people_per_seat = 16 := 
by
  let num_seats := 4
  let people_per_seat := 4
  sorry

end ferris_wheel_capacity_l177_177112


namespace bob_weight_l177_177643

variable (j b : ℕ)

theorem bob_weight :
  j + b = 210 →
  b - j = b / 3 →
  b = 126 :=
by
  intros h1 h2
  sorry

end bob_weight_l177_177643


namespace factor_expression_l177_177686

noncomputable def numerator (a b c : ℝ) : ℝ := 
(|a^2 + b^2|^3 + |b^2 + c^2|^3 + |c^2 + a^2|^3)

noncomputable def denominator (a b c : ℝ) : ℝ := 
(|a + b|^3 + |b + c|^3 + |c + a|^3)

theorem factor_expression (a b c : ℝ) : 
  (denominator a b c) ≠ 0 → 
  (numerator a b c) / (denominator a b c) = 1 :=
by
  sorry

end factor_expression_l177_177686


namespace cost_price_marked_price_ratio_l177_177226

theorem cost_price_marked_price_ratio (x : ℝ) (hx : x > 0) :
  let selling_price := (2 / 3) * x
  let cost_price := (3 / 4) * selling_price 
  cost_price / x = 1 / 2 := 
by
  let selling_price := (2 / 3) * x 
  let cost_price := (3 / 4) * selling_price 
  have hs : selling_price = (2 / 3) * x := rfl 
  have hc : cost_price = (3 / 4) * selling_price := rfl 
  have ratio := hc.symm 
  simp [ratio, hs]
  sorry

end cost_price_marked_price_ratio_l177_177226


namespace find_divisor_l177_177399

theorem find_divisor (x : ℝ) (h : 1152 / x - 189 = 3) : x = 6 :=
by
  sorry

end find_divisor_l177_177399


namespace suff_not_necess_cond_perpendicular_l177_177842

theorem suff_not_necess_cond_perpendicular (m : ℝ) :
  (m = 1 → ∀ x y : ℝ, x - y = 0 ∧ x + y = 0) ∧
  (m ≠ 1 → ∃ (x y : ℝ), ¬ (x - y = 0 ∧ x + y = 0)) :=
sorry

end suff_not_necess_cond_perpendicular_l177_177842


namespace ordered_pair_solution_l177_177641

theorem ordered_pair_solution :
  ∃ (x y : ℤ), 
    (x + y = (7 - x) + (7 - y)) ∧ 
    (x - y = (x - 2) + (y - 2)) ∧ 
    (x = 5 ∧ y = 2) :=
by
  sorry

end ordered_pair_solution_l177_177641


namespace total_cost_correct_l177_177650

def shirt_price : ℕ := 5
def hat_price : ℕ := 4
def jeans_price : ℕ := 10
def jacket_price : ℕ := 20
def shoes_price : ℕ := 15

def num_shirts : ℕ := 4
def num_jeans : ℕ := 3
def num_hats : ℕ := 4
def num_jackets : ℕ := 3
def num_shoes : ℕ := 2

def third_jacket_discount : ℕ := jacket_price / 2
def discount_per_two_shirts : ℕ := 2
def free_hat : ℕ := if num_jeans ≥ 3 then 1 else 0
def shoes_discount : ℕ := (num_shirts / 2) * discount_per_two_shirts

def total_cost : ℕ :=
  (num_shirts * shirt_price) +
  (num_jeans * jeans_price) +
  ((num_hats - free_hat) * hat_price) +
  ((num_jackets - 1) * jacket_price + third_jacket_discount) +
  (num_shoes * shoes_price - shoes_discount)

theorem total_cost_correct : total_cost = 138 := by
  sorry

end total_cost_correct_l177_177650


namespace peaches_total_l177_177975

def peaches_in_basket (a b : Nat) : Nat :=
  a + b 

theorem peaches_total (a b : Nat) (h1 : a = 20) (h2 : b = 25) : peaches_in_basket a b = 45 := 
by
  sorry

end peaches_total_l177_177975


namespace find_geometric_sequence_l177_177756

def geometric_sequence (b1 b2 b3 b4 : ℤ) :=
  ∃ q : ℤ, b2 = b1 * q ∧ b3 = b1 * q^2 ∧ b4 = b1 * q^3

theorem find_geometric_sequence :
  ∃ b1 b2 b3 b4 : ℤ, 
    geometric_sequence b1 b2 b3 b4 ∧
    (b1 + b4 = -49) ∧
    (b2 + b3 = 14) ∧ 
    ((b1, b2, b3, b4) = (7, -14, 28, -56) ∨ (b1, b2, b3, b4) = (-56, 28, -14, 7)) :=
by
  sorry

end find_geometric_sequence_l177_177756


namespace probability_of_team_A_winning_is_11_over_16_l177_177701

noncomputable def prob_A_wins_series : ℚ :=
  let total_games := 5
  let wins_needed_A := 2
  let wins_needed_B := 3
  -- Assuming equal probability for each game being won by either team
  let equal_chance_of_winning := 0.5
  -- Calculation would follow similar steps omitted for brevity
  -- Assuming the problem statement proven by external logical steps
  11 / 16

theorem probability_of_team_A_winning_is_11_over_16 :
  prob_A_wins_series = 11 / 16 := 
  sorry

end probability_of_team_A_winning_is_11_over_16_l177_177701


namespace total_cows_on_farm_l177_177850

-- Defining the conditions
variables (X H : ℕ) -- X is the number of cows per herd, H is the total number of herds
axiom half_cows_counted : 2800 = X * H / 2

-- The theorem stating the total number of cows on the entire farm
theorem total_cows_on_farm (X H : ℕ) (h1 : 2800 = X * H / 2) : 5600 = X * H := 
by 
  sorry

end total_cows_on_farm_l177_177850


namespace find_n_l177_177484

theorem find_n (n : ℝ) (h1 : ∀ m : ℝ, m = 4 → m^(m/2) = 4) : 
  n^(n/2) = 8 ↔ n = 2^Real.sqrt 6 :=
by
  sorry

end find_n_l177_177484


namespace find_a_l177_177418

theorem find_a (a : ℚ) (A : Set ℚ) (h : 3 ∈ A) (hA : A = {a + 2, 2 * a^2 + a}) : a = 3 / 2 := 
by
  sorry

end find_a_l177_177418


namespace count_polynomials_with_three_integer_roots_l177_177196

def polynomial_with_roots (n: ℕ) : Nat :=
  have h: n = 8 := by
    sorry
  if n = 8 then
    -- Apply the combinatorial argument as discussed
    52
  else
    -- Case for other n
    0

theorem count_polynomials_with_three_integer_roots:
  polynomial_with_roots 8 = 52 := 
  sorry

end count_polynomials_with_three_integer_roots_l177_177196


namespace square_perimeter_ratio_l177_177806

theorem square_perimeter_ratio (x y : ℝ)
(h : (x / y) ^ 2 = 16 / 25) : (4 * x) / (4 * y) = 4 / 5 :=
by sorry

end square_perimeter_ratio_l177_177806


namespace num_valid_configurations_l177_177036

-- Definitions used in the problem
def grid := (Fin 8) × (Fin 8)
def knights_tell_truth := true
def knaves_lie := true
def statement (i j : Fin 8) (r c : grid → ℕ) := (c ⟨0,j⟩ > r ⟨i,0⟩)

-- The theorem statement to prove
theorem num_valid_configurations : ∃ n : ℕ, n = 255 :=
sorry

end num_valid_configurations_l177_177036


namespace jake_present_weight_l177_177417

variables (J S : ℕ)

theorem jake_present_weight :
  (J - 33 = 2 * S) ∧ (J + S = 153) → J = 113 :=
by
  sorry

end jake_present_weight_l177_177417


namespace nabla_four_seven_l177_177809

def nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem nabla_four_seven : nabla 4 7 = 11 / 29 :=
by
  sorry

end nabla_four_seven_l177_177809


namespace river_depth_conditions_l177_177455

noncomputable def depth_beginning_may : ℝ := 15
noncomputable def depth_increase_june : ℝ := 11.25

theorem river_depth_conditions (d k : ℝ)
  (h1 : ∃ d, d = depth_beginning_may) 
  (h2 : 1.5 * d + k = 45)
  (h3 : k = 0.75 * d) :
  d = depth_beginning_may ∧ k = depth_increase_june :=
by
  have H : d = 15 := sorry
  have K : k = 11.25 := sorry
  exact ⟨H, K⟩

end river_depth_conditions_l177_177455


namespace prob_both_standard_prob_only_one_standard_l177_177027

-- Given conditions
axiom prob_A1 : ℝ
axiom prob_A2 : ℝ
axiom prob_A1_std : prob_A1 = 0.95
axiom prob_A2_std : prob_A2 = 0.95
axiom prob_not_A1 : ℝ
axiom prob_not_A2 : ℝ
axiom prob_not_A1_std : prob_not_A1 = 0.05
axiom prob_not_A2_std : prob_not_A2 = 0.05
axiom independent_A1_A2 : prob_A1 * prob_A2 = prob_A1 * prob_A2

-- Definitions of events
def event_A1 := true -- Event that the first product is standard
def event_A2 := true -- Event that the second product is standard
def event_not_A1 := not event_A1
def event_not_A2 := not event_A2

-- Proof problems
theorem prob_both_standard :
  prob_A1 * prob_A2 = 0.9025 := by sorry

theorem prob_only_one_standard :
  (prob_A1 * prob_not_A2) + (prob_not_A1 * prob_A2) = 0.095 := by sorry

end prob_both_standard_prob_only_one_standard_l177_177027


namespace sum_series_eq_l177_177068

theorem sum_series_eq : 
  ∑' n : ℕ, (n + 1) * (1 / 3 : ℝ)^n = 9 / 4 :=
by sorry

end sum_series_eq_l177_177068


namespace find_common_difference_l177_177174

theorem find_common_difference (AB BC AC : ℕ) (x y z d : ℕ) 
  (h1 : AB = 300) (h2 : BC = 350) (h3 : AC = 400) 
  (hx : x = (2 * d) / 5) (hy : y = (7 * d) / 15) (hz : z = (8 * d) / 15) 
  (h_sum : x + y + z = 750) : 
  d = 536 :=
by
  -- Proof goes here
  sorry

end find_common_difference_l177_177174


namespace next_two_equations_l177_177456

-- Definitions based on the conditions in the problem
def pattern1 (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Statement to prove the continuation of the pattern
theorem next_two_equations 
: pattern1 9 40 41 ∧ pattern1 11 60 61 :=
by
  sorry

end next_two_equations_l177_177456


namespace find_levels_satisfying_surface_area_conditions_l177_177066

theorem find_levels_satisfying_surface_area_conditions (n : ℕ) :
  let A_total_lateral := n * (n + 1) * Real.pi
  let A_total_vertical := Real.pi * n^2
  let A_total := n * (3 * n + 1) * Real.pi
  A_total_lateral = 0.35 * A_total → n = 13 :=
by
  intros A_total_lateral A_total_vertical A_total h
  sorry

end find_levels_satisfying_surface_area_conditions_l177_177066


namespace opposite_of_2023_is_neg_2023_l177_177122

-- Definitions based on conditions
def is_additive_inverse (x y : Int) : Prop := x + y = 0

-- The proof statement
theorem opposite_of_2023_is_neg_2023 : is_additive_inverse 2023 (-2023) :=
by
  -- This is where the proof would go, but it is marked as sorry for now
  sorry

end opposite_of_2023_is_neg_2023_l177_177122


namespace general_form_identity_expression_simplification_l177_177626

section
variable (a b x y : ℝ)

theorem general_form_identity : (a + b) * (a^2 - a * b + b^2) = a^3 + b^3 :=
by
  sorry

theorem expression_simplification : (x + y) * (x^2 - x * y + y^2) - (x - y) * (x^2 + x * y + y^2) = 2 * y^3 :=
by
  sorry
end

end general_form_identity_expression_simplification_l177_177626


namespace samantha_lost_pieces_l177_177562

theorem samantha_lost_pieces (total_pieces_on_board : ℕ) (arianna_lost : ℕ) (initial_pieces_per_player : ℕ) :
  total_pieces_on_board = 20 →
  arianna_lost = 3 →
  initial_pieces_per_player = 16 →
  (initial_pieces_per_player - (total_pieces_on_board - (initial_pieces_per_player - arianna_lost))) = 9 :=
by
  intros h1 h2 h3
  sorry

end samantha_lost_pieces_l177_177562


namespace trigonometric_identity_l177_177267

theorem trigonometric_identity (x : ℝ) (h : Real.sin (x + Real.pi / 4) = 1 / 3) :
  Real.sin (4 * x) - 2 * Real.cos (3 * x) * Real.sin x = -7 / 9 :=
by
  sorry

end trigonometric_identity_l177_177267


namespace average_age_decrease_l177_177240

theorem average_age_decrease (N : ℕ) (T : ℝ) 
  (h1 : T = 40 * N) 
  (h2 : ∀ new_average_age : ℝ, (T + 12 * 34) / (N + 12) = new_average_age → new_average_age = 34) :
  ∃ decrease : ℝ, decrease = 6 :=
by
  sorry

end average_age_decrease_l177_177240


namespace arnaldo_billion_difference_l177_177461

theorem arnaldo_billion_difference :
  (10 ^ 12) - (10 ^ 9) = 999000000000 :=
by
  sorry

end arnaldo_billion_difference_l177_177461


namespace complete_the_square_d_l177_177149

theorem complete_the_square_d (x : ℝ) : (∃ c d : ℝ, x^2 + 6 * x - 4 = 0 → (x + c)^2 = d) ∧ d = 13 :=
by
  sorry

end complete_the_square_d_l177_177149


namespace workers_time_l177_177500

variables (x y: ℝ)

theorem workers_time (h1 : (x > 0) ∧ (y > 0)) 
                     (h2 : (3/x + 2/y = 11/20)) 
                     (h3 : (1/x + 1/y = 1/2)) :
                     (x = 10 ∧ y = 8) := 
by
  sorry

end workers_time_l177_177500


namespace power_addition_rule_l177_177825

variable {a : ℝ}
variable {m n : ℕ}

theorem power_addition_rule (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + n) = 6 := by
  sorry

end power_addition_rule_l177_177825


namespace measure_of_angle_S_l177_177535

-- Define the angles in the pentagon PQRST
variables (P Q R S T : ℝ)
-- Assume the conditions from the problem
variables (h1 : P = Q)
variables (h2 : Q = R)
variables (h3 : S = T)
variables (h4 : P = S - 30)
-- Assume the sum of angles in a pentagon is 540 degrees
variables (h5 : P + Q + R + S + T = 540)

theorem measure_of_angle_S :
  S = 126 := by
  -- placeholder for the actual proof
  sorry

end measure_of_angle_S_l177_177535


namespace range_of_m_l177_177741

open Real

theorem range_of_m (m : ℝ) : (¬ ∃ x₀ : ℝ, m * x₀^2 + m * x₀ + 1 ≤ 0) ↔ (0 ≤ m ∧ m < 4) := by
  sorry

end range_of_m_l177_177741


namespace unique_solution_nat_triplet_l177_177633

theorem unique_solution_nat_triplet (x y l : ℕ) (h : x^3 + y^3 - 53 = 7^l) : (x, y, l) = (3, 3, 0) :=
sorry

end unique_solution_nat_triplet_l177_177633


namespace average_income_QR_l177_177859

theorem average_income_QR 
  (P Q R : ℝ)
  (h1 : (P + Q) / 2 = 2050)
  (h2 : (P + R) / 2 = 6200)
  (h3 : P = 3000) :
  (Q + R) / 2 = 5250 :=
  sorry

end average_income_QR_l177_177859


namespace max_eq_zero_max_two_solutions_l177_177026

theorem max_eq_zero_max_two_solutions {a b : Fin 10 → ℝ}
  (h : ∀ i, a i ≠ 0) : 
  ∃ (solution_count : ℕ), solution_count <= 2 ∧
  ∃ (solutions : Fin solution_count → ℝ), 
    ∀ (x : ℝ), (∀ i, max (a i * x + b i) = 0) ↔ ∃ j, x = solutions j := sorry

end max_eq_zero_max_two_solutions_l177_177026


namespace intersection_of_A_and_B_l177_177115

namespace IntersectionProblem

def setA : Set ℝ := {0, 1, 2}
def setB : Set ℝ := {x | x^2 - x ≤ 0}
def intersection : Set ℝ := {0, 1}

theorem intersection_of_A_and_B : A ∩ B = intersection := sorry

end IntersectionProblem

end intersection_of_A_and_B_l177_177115


namespace abs_neg_three_l177_177049

theorem abs_neg_three : |(-3 : ℝ)| = 3 := 
by
  -- The proof would go here, but we skip it for this exercise.
  sorry

end abs_neg_three_l177_177049


namespace num_balls_box_l177_177695

theorem num_balls_box (n : ℕ) (balls : Fin n → ℕ) (red blue : Fin n → Prop)
  (h_colors : ∀ i, red i ∨ blue i)
  (h_constraints : ∀ i j k,  red i ∨ red j ∨ red k ∧ blue i ∨ blue j ∨ blue k) : 
  n = 4 := 
sorry

end num_balls_box_l177_177695


namespace divisor_of_635_l177_177546

theorem divisor_of_635 (p : ℕ) (h1 : Nat.Prime p) (k : ℕ) (h2 : 635 = 7 * k * p + 11) : p = 89 :=
sorry

end divisor_of_635_l177_177546


namespace letters_per_large_envelope_l177_177647

theorem letters_per_large_envelope
  (total_letters : ℕ)
  (small_envelope_letters : ℕ)
  (large_envelopes : ℕ)
  (large_envelopes_count : ℕ)
  (h1 : total_letters = 80)
  (h2 : small_envelope_letters = 20)
  (h3 : large_envelopes_count = 30)
  (h4 : total_letters - small_envelope_letters = large_envelopes)
  : large_envelopes / large_envelopes_count = 2 :=
by
  sorry

end letters_per_large_envelope_l177_177647


namespace system_of_equations_solution_l177_177887

theorem system_of_equations_solution :
  ∃ x y : ℝ, 7 * x - 3 * y = 2 ∧ 2 * x + y = 8 ∧ x = 2 ∧ y = 4 :=
by
  use 2
  use 4
  sorry

end system_of_equations_solution_l177_177887


namespace find_n_l177_177544

variable (a r : ℚ) (n : ℕ)

def geom_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- Given conditions
axiom seq_first_term : a = 1 / 3
axiom seq_common_ratio : r = 1 / 3
axiom sum_of_first_n_terms_eq : geom_sum a r n = 80 / 243

-- Prove that n = 5
theorem find_n : n = 5 := by
  sorry

end find_n_l177_177544


namespace value_of_7th_term_l177_177110

noncomputable def arithmetic_sequence_a1_d_n (a1 d n a7 : ℝ) : Prop := 
  ((5 * a1 + 10 * d = 68) ∧ 
   (5 * (a1 + (n - 1) * d) - 10 * d = 292) ∧
   (n / 2 * (2 * a1 + (n - 1) * d) = 234) ∧ 
   (a1 + 6 * d = a7))

theorem value_of_7th_term (a1 d n a7 : ℝ) : 
  arithmetic_sequence_a1_d_n a1 d n 18 := 
by
  simp [arithmetic_sequence_a1_d_n]
  sorry

end value_of_7th_term_l177_177110


namespace bee_loss_rate_l177_177930

theorem bee_loss_rate (initial_bees : ℕ) (days : ℕ) (remaining_bees : ℕ) :
  initial_bees = 80000 → 
  days = 50 → 
  remaining_bees = initial_bees / 4 → 
  (initial_bees - remaining_bees) / days = 1200 :=
by
  intros h₁ h₂ h₃
  sorry

end bee_loss_rate_l177_177930


namespace probability_of_first_spade_or_ace_and_second_ace_l177_177999

theorem probability_of_first_spade_or_ace_and_second_ace :
  let deck_size := 52
  let aces := 4
  let spades := 13
  let non_ace_spades := spades - 1
  let other_aces := aces - 1
  let prob_first_non_ace_spade := non_ace_spades / deck_size
  let prob_first_ace_not_spade := (aces - 1) / deck_size
  let prob_first_ace_spade := 1 / deck_size
  let prob_second_ace_after_non_ace_spade := aces / (deck_size - 1)
  let prob_second_ace_after_ace_not_spade := (aces - 1) / (deck_size - 1)
  let prob_second_ace_after_ace_spade := (aces - 1) / (deck_size - 1)
  ((prob_first_non_ace_spade * prob_second_ace_after_non_ace_spade) +
   (prob_first_ace_not_spade * prob_second_ace_after_ace_not_spade) +
   (prob_first_ace_spade * prob_second_ace_after_ace_spade)) = 5 / 221 :=
by
  let deck_size := 52
  let aces := 4
  let spades := 13
  let non_ace_spades := spades - 1
  let other_aces := aces - 1
  let prob_first_non_ace_spade := non_ace_spades / deck_size
  let prob_first_ace_not_spade := (aces - 1) / deck_size
  let prob_first_ace_spade := 1 / deck_size
  let prob_second_ace_after_non_ace_spade := aces / (deck_size - 1)
  let prob_second_ace_after_ace_not_spade := (aces - 1) / (deck_size - 1)
  let prob_second_ace_after_ace_spade := (aces - 1) / (deck_size - 1)
  sorry

end probability_of_first_spade_or_ace_and_second_ace_l177_177999


namespace simplify_expression_l177_177188

theorem simplify_expression : 
  (Real.sqrt 12) + (Real.sqrt 4) * ((Real.sqrt 5 - Real.pi) ^ 0) - (abs (-2 * Real.sqrt 3)) = 2 := 
by 
  sorry

end simplify_expression_l177_177188


namespace irrational_number_l177_177758

noncomputable def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem irrational_number : 
  is_rational (Real.sqrt 4) ∧ 
  is_rational (22 / 7 : ℝ) ∧ 
  is_rational (1.0101 : ℝ) ∧ 
  ¬ is_rational (Real.pi / 3) 
  :=
sorry

end irrational_number_l177_177758


namespace sphere_radius_ratio_l177_177698

theorem sphere_radius_ratio (R r : ℝ) (h₁ : (4 / 3) * Real.pi * R ^ 3 = 450 * Real.pi) (h₂ : (4 / 3) * Real.pi * r ^ 3 = 0.25 * 450 * Real.pi) :
  r / R = 1 / 2 :=
sorry

end sphere_radius_ratio_l177_177698


namespace total_paper_clips_l177_177383

/-
Given:
- The number of cartons: c = 3
- The number of boxes: b = 4
- The number of bags: p = 2
- The number of paper clips in each carton: paper_clips_per_carton = 300
- The number of paper clips in each box: paper_clips_per_box = 550
- The number of paper clips in each bag: paper_clips_per_bag = 1200

Prove that the total number of paper clips is 5500.
-/

theorem total_paper_clips :
  let c := 3
  let paper_clips_per_carton := 300
  let b := 4
  let paper_clips_per_box := 550
  let p := 2
  let paper_clips_per_bag := 1200
  (c * paper_clips_per_carton + b * paper_clips_per_box + p * paper_clips_per_bag) = 5500 :=
by
  sorry

end total_paper_clips_l177_177383


namespace count_valid_Q_l177_177661

noncomputable def P (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 5)

def Q_degree (Q : Polynomial ℝ) : Prop :=
  Q.degree = 2

def R_degree (R : Polynomial ℝ) : Prop :=
  R.degree = 3

def P_Q_relation (Q R : Polynomial ℝ) : Prop :=
  ∀ x, P (Q.eval x) = P x * R.eval x

theorem count_valid_Q : 
  (∃ Qs : Finset (Polynomial ℝ), ∀ Q ∈ Qs, Q_degree Q ∧ (∃ R, R_degree R ∧ P_Q_relation Q R) 
    ∧ Qs.card = 22) :=
sorry

end count_valid_Q_l177_177661


namespace similar_triangle_longest_side_length_l177_177693

-- Given conditions as definitions 
def originalTriangleSides (a b c : ℕ) : Prop := a = 8 ∧ b = 10 ∧ c = 12
def similarTrianglePerimeter (P : ℕ) : Prop := P = 150

-- Statement to be proved using the given conditions
theorem similar_triangle_longest_side_length (a b c P : ℕ) 
  (h1 : originalTriangleSides a b c) 
  (h2 : similarTrianglePerimeter P) : 
  ∃ x : ℕ, P = (a + b + c) * x ∧ 12 * x = 60 :=
by
  -- Proof would go here
  sorry

end similar_triangle_longest_side_length_l177_177693


namespace prism_aligns_l177_177827

theorem prism_aligns (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ prism_dimensions = (a * 5, b * 10, c * 20) :=
by
  sorry

end prism_aligns_l177_177827


namespace student_19_in_sample_l177_177458

-- Definitions based on conditions
def total_students := 52
def sample_size := 4
def sampling_interval := 13

def selected_students := [6, 32, 45]

-- The theorem to prove
theorem student_19_in_sample : 19 ∈ selected_students ∨ ∃ k : ℕ, 13 * k + 6 = 19 :=
by
  sorry

end student_19_in_sample_l177_177458


namespace product_of_five_consecutive_integers_not_square_l177_177335

theorem product_of_five_consecutive_integers_not_square (a : ℕ) :
  ¬ ∃ b c d e : ℕ, b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ ∃ k : ℕ, (a * b * c * d * e) = k^2 :=
by
  sorry

end product_of_five_consecutive_integers_not_square_l177_177335


namespace length_first_train_correct_l177_177004

noncomputable def length_first_train 
    (speed_train1_kmph : ℕ := 120)
    (speed_train2_kmph : ℕ := 80)
    (length_train2_m : ℝ := 290.04)
    (time_sec : ℕ := 9) 
    (conversion_factor : ℝ := (5 / 18)) : ℝ :=
  let relative_speed_kmph := speed_train1_kmph + speed_train2_kmph
  let relative_speed_mps := relative_speed_kmph * conversion_factor
  let total_distance_m := relative_speed_mps * time_sec
  let length_train1_m := total_distance_m - length_train2_m
  length_train1_m

theorem length_first_train_correct 
    (L1_approx : ℝ := 210) :
    length_first_train = L1_approx :=
  by
  sorry

end length_first_train_correct_l177_177004


namespace notebook_cost_3_dollars_l177_177559

def cost_of_notebook (total_spent backpack_cost pen_cost pencil_cost num_notebooks : ℕ) : ℕ := 
  (total_spent - (backpack_cost + pen_cost + pencil_cost)) / num_notebooks

theorem notebook_cost_3_dollars 
  (total_spent : ℕ := 32) 
  (backpack_cost : ℕ := 15) 
  (pen_cost : ℕ := 1) 
  (pencil_cost : ℕ := 1) 
  (num_notebooks : ℕ := 5) 
  : cost_of_notebook total_spent backpack_cost pen_cost pencil_cost num_notebooks = 3 :=
by
  sorry

end notebook_cost_3_dollars_l177_177559


namespace ruler_cost_l177_177607

variable {s c r : ℕ}

theorem ruler_cost (h1 : s > 18) (h2 : r > 1) (h3 : c > r) (h4 : s * c * r = 1729) : c = 13 :=
by
  sorry

end ruler_cost_l177_177607


namespace geometric_progression_fourth_term_l177_177938

theorem geometric_progression_fourth_term (x : ℚ)
  (h : (3 * x + 3) / x = (5 * x + 5) / (3 * x + 3)) :
  (5 / 3) * (5 * x + 5) = -125/12 :=
by
  sorry

end geometric_progression_fourth_term_l177_177938


namespace total_heads_l177_177727

/-- There are H hens and C cows. Each hen has 1 head and 2 feet, and each cow has 1 head and 4 feet.
Given that the total number of feet is 140 and there are 26 hens, prove that the total number of heads is 48. -/
theorem total_heads (H C : ℕ) (h1 : 2 * H + 4 * C = 140) (h2 : H = 26) : H + C = 48 := by
  sorry

end total_heads_l177_177727


namespace evaluate_expression_l177_177952

theorem evaluate_expression (x : ℤ) (h : x + 1 = 4) : 
  (-3)^3 + (-3)^2 + (-3 * x) + 3 * x + 3^2 + 3^3 = 18 :=
by
  -- Since we know the condition x + 1 = 4
  have hx : x = 3 := by linarith
  -- Substitution x = 3 into the expression
  rw [hx]
  -- The expression after substitution and simplification
  sorry

end evaluate_expression_l177_177952


namespace Edmund_can_wrap_15_boxes_every_3_days_l177_177869

-- We define the conditions as Lean definitions
def inches_per_gift_box : ℕ := 18
def inches_per_day : ℕ := 90

-- We state the theorem to prove the question (15 gift boxes every 3 days)
theorem Edmund_can_wrap_15_boxes_every_3_days :
  (inches_per_day / inches_per_gift_box) * 3 = 15 :=
by
  sorry

end Edmund_can_wrap_15_boxes_every_3_days_l177_177869


namespace original_number_unique_l177_177948

theorem original_number_unique (x : ℝ) (h_pos : 0 < x) 
  (h_condition : 100 * x = 9 / x) : x = 3 / 10 :=
by
  sorry

end original_number_unique_l177_177948


namespace helium_min_cost_l177_177227

noncomputable def W (x : ℝ) : ℝ :=
  if x < 4 then 40 * (4 * x + 16 / x + 100)
  else 40 * (9 / (x * x) - 3 / x + 117)

theorem helium_min_cost :
  (∀ x, W x ≥ 4640) ∧ (W 2 = 4640) :=
by {
  sorry
}

end helium_min_cost_l177_177227


namespace custom_op_eval_l177_177573

def custom_op (a b : ℝ) : ℝ := 4 * a + 5 * b - a^2 * b

theorem custom_op_eval :
  custom_op 3 4 = -4 :=
by
  sorry

end custom_op_eval_l177_177573


namespace hyperbola_condition_l177_177537

theorem hyperbola_condition (a : ℝ) (h : a > 0)
  (e : ℝ) (h_e : e = Real.sqrt (1 + 4 / (a^2))) :
  (e > Real.sqrt 2) ↔ (0 < a ∧ a < 1) := 
sorry

end hyperbola_condition_l177_177537


namespace opposite_of_2023_l177_177288

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l177_177288


namespace number_of_yellow_marbles_l177_177041

/-- 
 In a jar with blue, red, and yellow marbles:
  - there are 7 blue marbles
  - there are 11 red marbles
  - the probability of picking a yellow marble is 1/4
 Show that the number of yellow marbles is 6.
-/
theorem number_of_yellow_marbles 
  (blue red y : ℕ) 
  (h_blue : blue = 7) 
  (h_red : red = 11) 
  (h_prob : y / (18 + y) = 1 / 4) : 
  y = 6 := 
sorry

end number_of_yellow_marbles_l177_177041


namespace karen_total_cost_l177_177327

noncomputable def calculate_total_cost (burger_price sandwich_price smoothie_price : ℝ) (num_smoothies : ℕ)
  (discount_rate tax_rate : ℝ) (order_time : ℕ) : ℝ :=
  let total_cost_before_discount := burger_price + sandwich_price + (num_smoothies * smoothie_price)
  let discount := if total_cost_before_discount > 15 ∧ order_time ≥ 1400 ∧ order_time ≤ 1600 then total_cost_before_discount * discount_rate else 0
  let reduced_price := total_cost_before_discount - discount
  let tax := reduced_price * tax_rate
  reduced_price + tax

theorem karen_total_cost :
  calculate_total_cost 5.75 4.50 4.25 2 0.20 0.12 1545 = 16.80 :=
by
  sorry

end karen_total_cost_l177_177327


namespace largest_fraction_l177_177619

theorem largest_fraction (x y z w : ℝ) (hx : 0 < x) (hxy : x < y) (hyz : y < z) (hzw : z < w) :
  max (max (max (max ((x + y) / (z + w)) ((x + w) / (y + z))) ((y + z) / (x + w))) ((y + w) / (x + z))) ((z + w) / (x + y)) = (z + w) / (x + y) :=
by sorry

end largest_fraction_l177_177619


namespace hyperbola_asymptote_l177_177568

theorem hyperbola_asymptote (m : ℝ) : 
  (∀ x y : ℝ, (y^2 / 16 - x^2 / 9 = 1) ↔ (y = m * x ∨ y = -m * x)) → 
  (m = 4 / 3) :=
by
  sorry

end hyperbola_asymptote_l177_177568


namespace line_eq_l177_177921

theorem line_eq (P : ℝ × ℝ) (hP : P = (1, 2)) (h_perp : ∀ x y : ℝ, 2 * x + y - 1 = 0 → x - 2 * y + c = 0) : 
  ∃ c : ℝ, (x - 2 * y + c = 0 ∧ P ∈ {(x, y) | x - 2 * y + c = 0}) ∧ c = 3 :=
  sorry

end line_eq_l177_177921


namespace commuting_time_equation_l177_177536

-- Definitions based on the conditions
def distance_to_cemetery : ℝ := 15
def cyclists_speed (x : ℝ) : ℝ := x
def car_speed (x : ℝ) : ℝ := 2 * x
def cyclists_start_time_earlier : ℝ := 0.5

-- The statement we need to prove
theorem commuting_time_equation (x : ℝ) (h : x > 0) :
  distance_to_cemetery / cyclists_speed x =
  (distance_to_cemetery / car_speed x) + cyclists_start_time_earlier :=
by
  sorry

end commuting_time_equation_l177_177536


namespace ratio_x_y_z_l177_177639

theorem ratio_x_y_z (x y z : ℝ) (h1 : 0.10 * x = 0.20 * y) (h2 : 0.30 * y = 0.40 * z) :
  ∃ k : ℝ, x = 8 * k ∧ y = 4 * k ∧ z = 3 * k :=
by                         
  sorry

end ratio_x_y_z_l177_177639


namespace value_of_expression_l177_177491

theorem value_of_expression (m : ℝ) (h : 1 / (m - 2) = 1) : (2 / (m - 2)) - m + 2 = 1 :=
sorry

end value_of_expression_l177_177491


namespace three_pow_two_digits_count_l177_177959

theorem three_pow_two_digits_count : 
  ∃ n_set : Finset ℕ, (∀ n ∈ n_set, 10 ≤ 3^n ∧ 3^n < 100) ∧ n_set.card = 2 := 
sorry

end three_pow_two_digits_count_l177_177959


namespace condo_floors_l177_177830

theorem condo_floors (F P : ℕ) (h1: 12 * F + 2 * P = 256) (h2 : P = 2) : F + P = 23 :=
by
  sorry

end condo_floors_l177_177830


namespace product_of_x_y_l177_177361

theorem product_of_x_y (x y : ℝ) (h1 : -3 * x + 4 * y = 28) (h2 : 3 * x - 2 * y = 8) : x * y = 264 :=
by
  sorry

end product_of_x_y_l177_177361


namespace marla_colors_green_squares_l177_177056

-- Condition 1: Grid dimensions
def num_rows : ℕ := 10
def num_cols : ℕ := 15

-- Condition 2: Red squares
def red_rows : ℕ := 4
def red_squares_per_row : ℕ := 6
def red_squares : ℕ := red_rows * red_squares_per_row

-- Condition 3: Blue rows (first 2 and last 2)
def blue_rows : ℕ := 2 + 2
def blue_squares_per_row : ℕ := num_cols
def blue_squares : ℕ := blue_rows * blue_squares_per_row

-- Derived information
def total_squares : ℕ := num_rows * num_cols
def non_green_squares : ℕ := red_squares + blue_squares

-- The Lemma to prove
theorem marla_colors_green_squares : total_squares - non_green_squares = 66 := by
  sorry

end marla_colors_green_squares_l177_177056


namespace middle_joints_capacity_l177_177699

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def bamboo_tube_capacity (a : ℕ → ℝ) : Prop :=
  a 0 + a 1 + a 2 = 4.5 ∧ a 6 + a 7 + a 8 = 2.5 ∧ arithmetic_seq a (a 1 - a 0)

theorem middle_joints_capacity (a : ℕ → ℝ) (d : ℝ) (h : bamboo_tube_capacity a) : 
  a 3 + a 4 + a 5 = 3.5 :=
by
  sorry

end middle_joints_capacity_l177_177699


namespace simplify_arithmetic_expr1_simplify_arithmetic_expr2_l177_177658

-- Problem 1 Statement
theorem simplify_arithmetic_expr1 (x y : ℝ) : 
  (x - 3 * y) - (y - 2 * x) = 3 * x - 4 * y :=
sorry

-- Problem 2 Statement
theorem simplify_arithmetic_expr2 (a b : ℝ) : 
  5 * a * b^2 - 3 * (2 * a^2 * b - 2 * (a^2 * b - 2 * a * b^2)) = -7 * a * b^2 :=
sorry

end simplify_arithmetic_expr1_simplify_arithmetic_expr2_l177_177658


namespace remainder_of_3_pow_108_plus_5_l177_177733

theorem remainder_of_3_pow_108_plus_5 :
  (3^108 + 5) % 10 = 6 := by
  sorry

end remainder_of_3_pow_108_plus_5_l177_177733


namespace simplify_expression_l177_177221

theorem simplify_expression (x : ℝ) (h₁ : x ≠ -1) (h₂: x ≠ -3) :
  (x - 1 - 8 / (x + 1)) / ( (x + 3) / (x + 1) ) = x - 3 :=
by
  sorry

end simplify_expression_l177_177221


namespace part_one_part_two_l177_177979

-- Definitions for the propositions
def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*x + m ≥ 0

def proposition_q (m : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (∀ x y : ℝ, (x^2)/(a) + (y^2)/(b) = 1)

-- Theorems for the answers
theorem part_one (m : ℝ) : ¬ proposition_p m → m < 1 :=
by sorry

theorem part_two (m : ℝ) : ¬ (proposition_p m ∧ proposition_q m) ∧ (proposition_p m ∨ proposition_q m) → m < 1 ∨ (4 ≤ m ∧ m ≤ 6) :=
by sorry

end part_one_part_two_l177_177979


namespace volume_conversion_l177_177653

theorem volume_conversion (V_ft : ℕ) (h_V : V_ft = 216) (conversion_factor : ℕ) (h_cf : conversion_factor = 27) :
  V_ft / conversion_factor = 8 :=
by
  sorry

end volume_conversion_l177_177653


namespace simplify_expression_l177_177209

theorem simplify_expression :
  (20^4 + 625) * (40^4 + 625) * (60^4 + 625) * (80^4 + 625) /
  (10^4 + 625) * (30^4 + 625) * (50^4 + 625) * (70^4 + 625) = 7 := 
sorry

end simplify_expression_l177_177209


namespace necessary_condition_for_positive_on_interval_l177_177805

theorem necessary_condition_for_positive_on_interval (a b : ℝ) (h : a + 2 * b > 0) :
  (∀ x, 0 ≤ x → x ≤ 1 → (a * x + b) > 0) ↔ ∃ c, 0 < c ∧ c ≤ 1 ∧ a + 2 * b > 0 ∧ ¬∀ d, 0 < d ∧ d ≤ 1 → a * d + b > 0 := 
by 
  sorry

end necessary_condition_for_positive_on_interval_l177_177805


namespace average_speed_of_train_l177_177898

theorem average_speed_of_train (d1 d2: ℝ) (t1 t2: ℝ) (h_d1: d1 = 250) (h_d2: d2 = 350) (h_t1: t1 = 2) (h_t2: t2 = 4) :
  (d1 + d2) / (t1 + t2) = 100 := by
  sorry

end average_speed_of_train_l177_177898


namespace no_possible_path_l177_177600

theorem no_possible_path (n : ℕ) (h1 : n > 0) :
  ¬ ∃ (path : ℕ × ℕ → ℕ × ℕ), 
    (∀ (i : ℕ × ℕ), path i = if (i.1 < n - 1 ∧ i.2 < n - 1) then (i.1 + 1, i.2) else if i.2 < n - 1 then (i.1, i.2 + 1) else (i.1 - 1, i.2 - 1)) ∧
    (∀ (i j : ℕ × ℕ), i ≠ j → path i ≠ path j) ∧
    path (0, 0) = (0, 1) ∧
    path (n-1, n-1) = (n-1, 0) :=
sorry

end no_possible_path_l177_177600


namespace evaluated_expression_l177_177451

noncomputable def evaluation_problem (x a y z c d : ℝ) : ℝ :=
  (2 * x^3 - 3 * a^4) / (y^2 + 4 * z^5) + c^4 - d^2

theorem evaluated_expression :
  evaluation_problem 0.66 0.1 0.66 0.1 0.066 0.1 = 1.309091916 :=
by
  sorry

end evaluated_expression_l177_177451


namespace gcd_97_pow_10_plus_1_and_97_pow_10_plus_97_pow_3_plus_1_l177_177777

theorem gcd_97_pow_10_plus_1_and_97_pow_10_plus_97_pow_3_plus_1 :
  Int.gcd (97 ^ 10 + 1) (97 ^ 10 + 97 ^ 3 + 1) = 1 := sorry

end gcd_97_pow_10_plus_1_and_97_pow_10_plus_97_pow_3_plus_1_l177_177777


namespace exponents_product_as_cube_l177_177293

theorem exponents_product_as_cube :
  (3^12 * 3^3) = 243^3 :=
sorry

end exponents_product_as_cube_l177_177293


namespace average_of_numbers_between_6_and_36_divisible_by_7_l177_177464

noncomputable def average_of_divisibles_by_seven : ℕ :=
  let numbers := [7, 14, 21, 28, 35]
  let sum := numbers.sum
  let count := numbers.length
  sum / count

theorem average_of_numbers_between_6_and_36_divisible_by_7 : average_of_divisibles_by_seven = 21 :=
by
  sorry

end average_of_numbers_between_6_and_36_divisible_by_7_l177_177464


namespace system1_solution_l177_177118

theorem system1_solution (x y : ℝ) (h1 : 4 * x - 3 * y = 1) (h2 : 3 * x - 2 * y = -1) : x = -5 ∧ y = 7 :=
sorry

end system1_solution_l177_177118


namespace proof_x_squared_minus_y_squared_l177_177811

theorem proof_x_squared_minus_y_squared (x y : ℚ) (h1 : x + y = 9 / 14) (h2 : x - y = 3 / 14) :
  x^2 - y^2 = 27 / 196 := by
  sorry

end proof_x_squared_minus_y_squared_l177_177811


namespace triangular_number_is_perfect_square_l177_177344

def is_triangular_number (T : ℕ) : Prop :=
∃ n : ℕ, T = n * (n + 1) / 2

def is_perfect_square (T : ℕ) : Prop :=
∃ y : ℕ, T = y * y

theorem triangular_number_is_perfect_square:
  ∀ (x_k : ℕ), 
    ((∃ n y : ℕ, (2 * n + 1)^2 - 8 * y^2 = 1 ∧ T_n = n * (n + 1) / 2 ∧ T_n = x_k^2 - 1 / 8) →
    (is_triangular_number T_n → is_perfect_square T_n)) :=
by
  sorry

end triangular_number_is_perfect_square_l177_177344


namespace max_value_a_l177_177598

def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → |a - 2| ≤ |x + 1 / x|

theorem max_value_a : ∃ (a : ℝ), condition a ∧ (∀ b : ℝ, condition b → b ≤ 4) :=
  sorry

end max_value_a_l177_177598


namespace A_inter_B_domain_l177_177570

def A_domain : Set ℝ := {x : ℝ | x^2 + x - 2 >= 0}
def B_domain : Set ℝ := {x : ℝ | (2*x + 6)/(3 - x) >= 0 ∧ x ≠ -2}

theorem A_inter_B_domain :
  (A_domain ∩ B_domain) = {x : ℝ | (1 <= x ∧ x < 3) ∨ (-3 <= x ∧ x < -2)} :=
by
  sorry

end A_inter_B_domain_l177_177570


namespace simplify_fraction_lemma_l177_177836

noncomputable def simplify_fraction (a : ℝ) (h : a ≠ 5) : ℝ :=
  (a^2 - 5 * a) / (a - 5)

theorem simplify_fraction_lemma (a : ℝ) (h : a ≠ 5) : simplify_fraction a h = a := by
  sorry

end simplify_fraction_lemma_l177_177836


namespace max_value_of_expression_l177_177153

theorem max_value_of_expression :
  ∀ r : ℝ, -3 * r^2 + 30 * r + 8 ≤ 83 :=
by
  -- Proof needed
  sorry

end max_value_of_expression_l177_177153


namespace avg_combined_is_2a_plus_3b_l177_177062

variables {x1 x2 x3 y1 y2 y3 a b : ℝ}

-- Given conditions
def avg_x_is_a (x1 x2 x3 a : ℝ) : Prop := (x1 + x2 + x3) / 3 = a
def avg_y_is_b (y1 y2 y3 b : ℝ) : Prop := (y1 + y2 + y3) / 3 = b

-- The statement to be proved
theorem avg_combined_is_2a_plus_3b
    (hx : avg_x_is_a x1 x2 x3 a) 
    (hy : avg_y_is_b y1 y2 y3 b) :
    ((2 * x1 + 3 * y1) + (2 * x2 + 3 * y2) + (2 * x3 + 3 * y3)) / 3 = 2 * a + 3 * b := 
by
  sorry

end avg_combined_is_2a_plus_3b_l177_177062


namespace score_difference_l177_177315

theorem score_difference 
  (x y z w : ℝ)
  (h1 : x = 2 + (y + z + w) / 3)
  (h2 : y = (x + z + w) / 3 - 3)
  (h3 : z = 3 + (x + y + w) / 3) :
  (x + y + z) / 3 - w = 2 :=
by {
  sorry
}

end score_difference_l177_177315


namespace product_units_tens_not_divisible_by_8_l177_177087

theorem product_units_tens_not_divisible_by_8 :
  ¬ (1834 % 8 = 0) → (4 * 3 = 12) :=
by
  intro h
  exact (by norm_num : 4 * 3 = 12)

end product_units_tens_not_divisible_by_8_l177_177087


namespace lisa_score_is_85_l177_177427

def score_formula (c w : ℕ) : ℕ := 30 + 4 * c - w

theorem lisa_score_is_85 (c w : ℕ) 
  (score_equality : 85 = score_formula c w)
  (non_neg_w : w ≥ 0)
  (total_questions : c + w ≤ 30) :
  (c = 14 ∧ w = 1) :=
by
  sorry

end lisa_score_is_85_l177_177427


namespace adam_spent_money_on_ferris_wheel_l177_177309

def tickets_bought : ℕ := 13
def tickets_left : ℕ := 4
def ticket_cost : ℕ := 9
def tickets_used : ℕ := tickets_bought - tickets_left

theorem adam_spent_money_on_ferris_wheel :
  tickets_used * ticket_cost = 81 :=
by
  sorry

end adam_spent_money_on_ferris_wheel_l177_177309


namespace percentage_goods_lost_eq_l177_177635

-- Define the initial conditions
def initial_value : ℝ := 100
def profit_margin : ℝ := 0.10 * initial_value
def selling_price : ℝ := initial_value + profit_margin
def loss_percentage : ℝ := 0.12

-- Define the correct answer as a constant
def correct_percentage_loss : ℝ := 13.2

-- Define the target theorem
theorem percentage_goods_lost_eq : (0.12 * selling_price / initial_value * 100) = correct_percentage_loss := 
by
  -- sorry is used to skip the proof part as per instructions
  sorry

end percentage_goods_lost_eq_l177_177635


namespace odd_function_iff_a2_b2_zero_l177_177839

noncomputable def f (x a b : ℝ) : ℝ := x * |x - a| + b

theorem odd_function_iff_a2_b2_zero {a b : ℝ} :
  (∀ x, f x a b = - f (-x) a b) ↔ a^2 + b^2 = 0 := by
  sorry

end odd_function_iff_a2_b2_zero_l177_177839


namespace percentage_of_water_in_juice_l177_177392

-- Define the initial condition for tomato puree water percentage
def puree_water_percentage : ℝ := 0.20

-- Define the volume of tomato puree produced from tomato juice
def volume_puree : ℝ := 3.75

-- Define the volume of tomato juice used to produce the puree
def volume_juice : ℝ := 30

-- Given conditions and definitions, prove the percentage of water in tomato juice
theorem percentage_of_water_in_juice :
  ((volume_juice - (volume_puree - puree_water_percentage * volume_puree)) / volume_juice) * 100 = 90 :=
by sorry

end percentage_of_water_in_juice_l177_177392


namespace rectangle_square_ratio_l177_177781

theorem rectangle_square_ratio (s x y : ℝ) (h1 : 0.1 * s ^ 2 = 0.25 * x * y) (h2 : y = s / 4) :
  x / y = 6 := 
sorry

end rectangle_square_ratio_l177_177781


namespace pats_stick_length_correct_l177_177391

noncomputable def jane_stick_length : ℕ := 22
noncomputable def sarah_stick_length : ℕ := jane_stick_length + 24
noncomputable def uncovered_pats_stick : ℕ := sarah_stick_length / 2
noncomputable def covered_pats_stick : ℕ := 7
noncomputable def total_pats_stick : ℕ := uncovered_pats_stick + covered_pats_stick

theorem pats_stick_length_correct : total_pats_stick = 30 := by
  sorry

end pats_stick_length_correct_l177_177391


namespace solve_inequality_l177_177609

theorem solve_inequality (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ ((4 / 3) < x ∧ x < 2) ∨ (2 < x) :=
by {
  sorry
}

end solve_inequality_l177_177609


namespace percentage_not_even_integers_l177_177527

variable (T : ℝ) (E : ℝ)
variables (h1 : 0.36 * T = E * 0.60) -- Condition 1 translated: 36% of T are even multiples of 3.
variables (h2 : E * 0.40)            -- Condition 2 translated: 40% of E are not multiples of 3.

theorem percentage_not_even_integers : 0.40 * T = T - E :=
by
  sorry

end percentage_not_even_integers_l177_177527


namespace trisha_total_distance_walked_l177_177476

def d1 : ℝ := 0.1111111111111111
def d2 : ℝ := 0.1111111111111111
def d3 : ℝ := 0.6666666666666666

theorem trisha_total_distance_walked :
  d1 + d2 + d3 = 0.8888888888888888 := 
sorry

end trisha_total_distance_walked_l177_177476


namespace parameter_range_exists_solution_l177_177613

theorem parameter_range_exists_solution :
  (∃ b : ℝ, -14 < b ∧ b < 9 ∧ ∃ a : ℝ, ∃ x y : ℝ,
    x^2 + y^2 + 2 * b * (b + x + y) = 81 ∧ y = 5 / ((x - a)^2 + 1)) :=
sorry

end parameter_range_exists_solution_l177_177613


namespace quadratic_monotonic_range_l177_177700

theorem quadratic_monotonic_range {t : ℝ} (h : ∀ x1 x2 : ℝ, (1 < x1 ∧ x1 < 3) → (1 < x2 ∧ x2 < 3) → x1 < x2 → (x1^2 - 2 * t * x1 + 1 ≤ x2^2 - 2 * t * x2 + 1)) : 
  t ≤ 1 ∨ t ≥ 3 :=
by
  sorry

end quadratic_monotonic_range_l177_177700


namespace remainder_145_mul_155_div_12_l177_177099

theorem remainder_145_mul_155_div_12 : (145 * 155) % 12 = 11 := by
  sorry

end remainder_145_mul_155_div_12_l177_177099


namespace distinct_values_f_in_interval_l177_177198

noncomputable def f (x : ℝ) : ℤ :=
  ⌊x⌋ + ⌊2 * x⌋ + ⌊(5 * x) / 3⌋ + ⌊3 * x⌋ + ⌊4 * x⌋

theorem distinct_values_f_in_interval : 
  ∃ n : ℕ, n = 734 ∧ 
    ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 100 ∧ 0 ≤ y ∧ y ≤ 100 → 
      f x = f y → x = y :=
sorry

end distinct_values_f_in_interval_l177_177198


namespace cube_volumes_total_l177_177433

theorem cube_volumes_total :
  let v1 := 5^3
  let v2 := 6^3
  let v3 := 7^3
  v1 + v2 + v3 = 684 := by
  -- Here will be the proof using Lean's tactics
  sorry

end cube_volumes_total_l177_177433


namespace spider_legs_total_l177_177743

-- Definitions based on given conditions
def spiders : ℕ := 4
def legs_per_spider : ℕ := 8

-- Theorem statement
theorem spider_legs_total : (spiders * legs_per_spider) = 32 := by
  sorry

end spider_legs_total_l177_177743


namespace domain_of_f_l177_177843

noncomputable def f (x : ℝ) := 1 / Real.log (x + 1) + Real.sqrt (9 - x^2)

theorem domain_of_f :
  {x : ℝ | 0 < x + 1} ∩ {x : ℝ | x ≠ 0} ∩ {x : ℝ | 9 - x^2 ≥ 0} = (Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioc 0 (3 : ℝ)) :=
by
  sorry

end domain_of_f_l177_177843


namespace sin_inequality_l177_177855

open Real

theorem sin_inequality (a b : ℝ) (n : ℕ) (ha : 0 < a) (haq : a < π/4) (hb : 0 < b) (hbq : b < π/4) (hn : 0 < n) :
  (sin a)^n + (sin b)^n / (sin a + sin b)^n ≥ (sin (2 * a))^n + (sin (2 * b))^n / (sin (2 * a) + sin (2* b))^n :=
sorry

end sin_inequality_l177_177855


namespace roots_quadratic_eq_a2_b2_l177_177624

theorem roots_quadratic_eq_a2_b2 (a b : ℝ) (h1 : a^2 - 5 * a + 5 = 0) (h2 : b^2 - 5 * b + 5 = 0) : a^2 + b^2 = 15 :=
by
  sorry

end roots_quadratic_eq_a2_b2_l177_177624


namespace triangle_DEF_area_10_l177_177722

-- Definitions of vertices and line
def D : ℝ × ℝ := (4, 0)
def E : ℝ × ℝ := (0, 4)
def line (x y : ℝ) : Prop := x + y = 9

-- Definition of point F lying on the given line
axiom F_on_line (F : ℝ × ℝ) : line (F.1) (F.2)

-- The proof statement of the area of triangle DEF being 10
theorem triangle_DEF_area_10 : ∃ F : ℝ × ℝ, line F.1 F.2 ∧ 
  (1 / 2) * abs (D.1 - F.1) * abs E.2 = 10 :=
by
  sorry

end triangle_DEF_area_10_l177_177722


namespace difference_square_consecutive_l177_177909

theorem difference_square_consecutive (x : ℕ) (h : x * (x + 1) = 812) : (x + 1)^2 - x = 813 :=
sorry

end difference_square_consecutive_l177_177909


namespace count_zeros_in_decimal_rep_l177_177539

theorem count_zeros_in_decimal_rep (n : ℕ) (h : n = 2^3 * 5^7) : 
  ∀ (a b : ℕ), (∃ (a : ℕ) (b : ℕ), n = 10^b ∧ a < 10^b) → 
  6 = b - 1 := by
  sorry

end count_zeros_in_decimal_rep_l177_177539


namespace numberOfBookshelves_l177_177320

-- Define the conditions as hypotheses
def numBooks : ℕ := 23
def numMagazines : ℕ := 61
def totalItems : ℕ := 2436

-- Define the number of items per bookshelf
def itemsPerBookshelf : ℕ := numBooks + numMagazines

-- State the theorem to be proven
theorem numberOfBookshelves (bookshelves : ℕ) :
  itemsPerBookshelf * bookshelves = totalItems → 
  bookshelves = 29 :=
by
  -- placeholder for proof
  sorry

end numberOfBookshelves_l177_177320


namespace ratio_of_x_to_y_l177_177508

-- Given condition: The percentage that y is less than x is 83.33333333333334%.
def percentage_less_than (x y : ℝ) : Prop := (x - y) / x = 0.8333333333333334

-- Prove: The ratio R = x / y is 1/6.
theorem ratio_of_x_to_y (x y : ℝ) (h : percentage_less_than x y) : x / y = 6 := 
by sorry

end ratio_of_x_to_y_l177_177508


namespace passengers_taken_at_second_station_l177_177132

noncomputable def initial_passengers : ℕ := 270
noncomputable def passengers_dropped_first_station := initial_passengers / 3
noncomputable def passengers_after_first_station := initial_passengers - passengers_dropped_first_station + 280
noncomputable def passengers_dropped_second_station := passengers_after_first_station / 2
noncomputable def passengers_after_second_station (x : ℕ) := passengers_after_first_station - passengers_dropped_second_station + x
noncomputable def passengers_at_third_station := 242

theorem passengers_taken_at_second_station : ∃ x : ℕ,
  passengers_after_second_station x = passengers_at_third_station ∧ x = 12 :=
by
  sorry

end passengers_taken_at_second_station_l177_177132


namespace rectangle_perimeter_of_divided_square_l177_177680

theorem rectangle_perimeter_of_divided_square
  (s : ℝ)
  (hs : 4 * s = 100) :
  let l := s
  let w := s / 2
  2 * (l + w) = 75 :=
by
  let l := s
  let w := s / 2
  sorry

end rectangle_perimeter_of_divided_square_l177_177680


namespace power_function_properties_l177_177778

theorem power_function_properties (m : ℤ) :
  (m^2 - 2 * m - 2 ≠ 0) ∧ (m^2 + 4 * m < 0) ∧ (m^2 + 4 * m % 2 = 1) → m = -1 := by
  intro h
  sorry

end power_function_properties_l177_177778


namespace factorization_correct_l177_177145

theorem factorization_correct {c d : ℤ} (h1 : c + 4 * d = 4) (h2 : c * d = -32) :
  c - d = 12 :=
by
  sorry

end factorization_correct_l177_177145


namespace local_minimum_at_two_l177_177394

def f (x : ℝ) : ℝ := x^3 - 12 * x

theorem local_minimum_at_two : ∃ a : ℝ, a = 2 ∧ ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, (0 < |x - a| ∧ |x - a| < δ) → f x > f a :=
by sorry

end local_minimum_at_two_l177_177394


namespace quadratic_has_distinct_real_roots_l177_177567

theorem quadratic_has_distinct_real_roots (m : ℝ) (hm : m ≠ 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (m * x1^2 - 2 * x1 + 3 = 0) ∧ (m * x2^2 - 2 * x2 + 3 = 0) ↔ 0 < m ∧ m < (1 / 3) :=
by
  sorry

end quadratic_has_distinct_real_roots_l177_177567


namespace tony_water_intake_l177_177802

theorem tony_water_intake (yesterday water_two_days_ago : ℝ) 
    (h1 : yesterday = 48) (h2 : yesterday = 0.96 * water_two_days_ago) :
    water_two_days_ago = 50 :=
by
  sorry

end tony_water_intake_l177_177802


namespace solution_set_f_pos_l177_177245

open Set Function

variables (f : ℝ → ℝ)
variables (h_even : ∀ x : ℝ, f (-x) = f x)
variables (h_diff : ∀ x ≠ 0, DifferentiableAt ℝ f x)
variables (h_pos : ∀ x : ℝ, x > 0 → f x + x * (f' x) > 0)
variables (h_at_2 : f 2 = 0)

theorem solution_set_f_pos :
  {x : ℝ | f x > 0} = (Iio (-2)) ∪ (Ioi 2) :=
by 
  sorry

end solution_set_f_pos_l177_177245


namespace proof1_proof2_l177_177584

open Real

noncomputable def problem1 (a b c : ℝ) (A : ℝ) (S : ℝ) :=
  ∃ (a b : ℝ), A = π / 3 ∧ c = 2 ∧ S = sqrt 3 / 2 ∧ S = 1/2 * b * 2 * sin (π / 3) ∧
  a^2 = b^2 + c^2 - 2 * b * c * cos (π / 3) ∧ b = 1 ∧ a = sqrt 3

noncomputable def problem2 (a b c : ℝ) (A B : ℝ) :=
  c = a * cos B ∧ (a + b + c) * (a + b - c) = (2 + sqrt 2) * a * b ∧ 
  B = π / 4 ∧ A = π / 2 → 
  ∃ C, C = π / 4 ∧ C = B

theorem proof1 : problem1 (sqrt 3) 1 2 (π / 3) (sqrt 3 / 2) :=
by
  sorry

theorem proof2 : problem2 (sqrt 3) 1 2 (π / 2) (π / 4) :=
by
  sorry

end proof1_proof2_l177_177584


namespace find_a6_l177_177368

variable {a : ℕ → ℝ}
variable {q : ℝ}
variable {a₁ : ℝ}

/-- The sequence is a geometric sequence -/
axiom geom_seq (n : ℕ) : a n = a₁ * q ^ (n - 1)

/-- The sum of the first three terms is 168 -/
axiom sum_of_first_three_terms : a₁ + a₁ * q + a₁ * q ^ 2 = 168

/-- The difference between the 2nd and the 5th terms is 42 -/
axiom difference_a2_a5 : a₁ * q - a₁ * q ^ 4 = 42

theorem find_a6 : a 6 = 3 :=
by
  -- Proof goes here
  sorry

end find_a6_l177_177368


namespace find_lower_percentage_l177_177460

theorem find_lower_percentage (P : ℝ) : 
  (12000 * 0.15 * 2 - 720 = 12000 * (P / 100) * 2) → P = 12 := by
  sorry

end find_lower_percentage_l177_177460


namespace surface_area_of_rectangular_prism_l177_177984

theorem surface_area_of_rectangular_prism :
  ∀ (length width height : ℝ), length = 8 → width = 4 → height = 2 → 
    2 * (length * width + length * height + width * height) = 112 :=
by
  intros length width height h_length h_width h_height
  rw [h_length, h_width, h_height]
  sorry

end surface_area_of_rectangular_prism_l177_177984


namespace water_tank_capacity_l177_177003

-- Define the variables and conditions
variables (T : ℝ) (h : 0.35 * T = 36)

-- State the theorem
theorem water_tank_capacity : T = 103 :=
by
  -- Placeholder for proof
  sorry

end water_tank_capacity_l177_177003


namespace area_of_trapezoid_RSQT_l177_177750

theorem area_of_trapezoid_RSQT
  (PR PQ : ℝ)
  (PR_eq_PQ : PR = PQ)
  (small_triangle_area : ℝ)
  (total_area : ℝ)
  (num_of_small_triangles : ℕ)
  (num_of_triangles_in_trapezoid : ℕ)
  (area_of_trapezoid : ℝ)
  (is_isosceles_triangle : ∀ (a b c : ℝ), a = b → b = c → a = c)
  (are_similar_triangles : ∀ {A B C D E F : ℝ}, 
    A / B = D / E → A / C = D / F → B / A = E / D → C / A = F / D)
  (smallest_triangle_areas : ∀ {n : ℕ}, n = 9 → small_triangle_area = 2 → num_of_small_triangles = 9)
  (triangle_total_area : ∀ (a : ℝ), a = 72 → total_area = 72)
  (contains_3_small_triangles : ∀ (n : ℕ), n = 3 → num_of_triangles_in_trapezoid = 3)
  (parallel_ST_to_PQ : ∀ {x y z : ℝ}, x = z → y = z → x = y)
  : area_of_trapezoid = 39 :=
sorry

end area_of_trapezoid_RSQT_l177_177750


namespace cleaner_for_dog_stain_l177_177962

theorem cleaner_for_dog_stain (D : ℝ) (H : 6 * D + 3 * 4 + 1 * 1 = 49) : D = 6 :=
by 
  -- Proof steps would go here, but we are skipping the proof.
  sorry

end cleaner_for_dog_stain_l177_177962


namespace find_x_l177_177134

theorem find_x (x : ℝ) (h : 49 / x = 700) : x = 0.07 :=
sorry

end find_x_l177_177134


namespace complement_of_intersection_eq_l177_177876

-- Definitions of sets with given conditions
def U : Set ℝ := {x | 0 ≤ x ∧ x < 10}
def A : Set ℝ := {x | 2 < x ∧ x ≤ 4}
def B : Set ℝ := {x | 3 < x ∧ x ≤ 5}

-- Complement of a set with respect to U
def complement_U (S : Set ℝ) : Set ℝ := {x | x ∈ U ∧ x ∉ S}

-- Intersect two sets
def intersection (S1 S2 : Set ℝ) : Set ℝ := {x | x ∈ S1 ∧ x ∈ S2}

theorem complement_of_intersection_eq :
  complement_U (intersection A B) = {x | (0 ≤ x ∧ x ≤ 2) ∨ (5 < x ∧ x < 10)} := 
by
  sorry

end complement_of_intersection_eq_l177_177876


namespace range_of_a_l177_177291

theorem range_of_a (a : ℝ) :
  ¬ (∃ x0 : ℝ, 2^x0 - 2 ≤ a^2 - 3 * a) → (1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l177_177291


namespace quadratic_eq_zero_l177_177985

theorem quadratic_eq_zero (x a b : ℝ) (h : x = a ∨ x = b) : x^2 - (a + b) * x + a * b = 0 :=
by sorry

end quadratic_eq_zero_l177_177985


namespace find_a_20_l177_177808

-- Definitions
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ (a₀ d : ℤ), ∀ n, a n = a₀ + n * d

def sum_first_n (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = n * (a 0 + a (n - 1)) / 2

-- Conditions and question
theorem find_a_20 (a S : ℕ → ℤ) (a₀ d : ℤ) :
  arithmetic_seq a ∧ sum_first_n a S ∧ 
  S 6 = 8 * (S 3) ∧ a 3 - a 5 = 8 → a 20 = -74 :=
by
  sorry

end find_a_20_l177_177808


namespace raccoon_hid_nuts_l177_177377

theorem raccoon_hid_nuts :
  ∃ (r p : ℕ), r + p = 25 ∧ (p = r - 3) ∧ 5 * r = 6 * p ∧ 5 * r = 70 :=
by
  sorry

end raccoon_hid_nuts_l177_177377


namespace unique_intersection_of_A_and_B_l177_177721

-- Define the sets A and B with their respective conditions
def A : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ x^2 + y^2 = 4 }

def B (r : ℝ) : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (x - 3)^2 + (y - 4)^2 = r^2 ∧ r > 0 }

-- Define the main theorem statement
theorem unique_intersection_of_A_and_B (r : ℝ) (h : r > 0) : 
  (∃! p, p ∈ A ∧ p ∈ B r) ↔ r = 3 ∨ r = 7 :=
sorry

end unique_intersection_of_A_and_B_l177_177721


namespace square_perimeter_l177_177339

-- Define the area of the square
def square_area := 720

-- Define the side length of the square
noncomputable def side_length := Real.sqrt square_area

-- Define the perimeter of the square
noncomputable def perimeter := 4 * side_length

-- Statement: Prove that the perimeter is 48 * sqrt(5)
theorem square_perimeter : perimeter = 48 * Real.sqrt 5 :=
by
  -- The proof is omitted as instructed
  sorry

end square_perimeter_l177_177339


namespace problem_l177_177386

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- Conditions
def condition1 : a + b = 1 := sorry
def condition2 : a^2 + b^2 = 3 := sorry
def condition3 : a^3 + b^3 = 4 := sorry
def condition4 : a^4 + b^4 = 7 := sorry

-- Question and proof
theorem problem : a^10 + b^10 = 123 :=
by
  have h1 : a + b = 1 := condition1
  have h2 : a^2 + b^2 = 3 := condition2
  have h3 : a^3 + b^3 = 4 := condition3
  have h4 : a^4 + b^4 = 7 := condition4
  sorry

end problem_l177_177386


namespace sector_max_area_l177_177250

-- Define the problem conditions
variables (α : ℝ) (R : ℝ)
variables (h_perimeter : 2 * R + R * α = 40)
variables (h_positive_radius : 0 < R)

-- State the theorem
theorem sector_max_area (h_alpha : α = 2) : 
  1/2 * α * (40 - 2 * R) * R = 100 := 
sorry

end sector_max_area_l177_177250


namespace cirrus_to_cumulus_is_four_l177_177204

noncomputable def cirrus_to_cumulus_ratio (Ci Cu Cb : ℕ) : ℕ :=
  Ci / Cu

theorem cirrus_to_cumulus_is_four :
  ∀ (Ci Cu Cb : ℕ), (Cb = 3) → (Cu = 12 * Cb) → (Ci = 144) → cirrus_to_cumulus_ratio Ci Cu Cb = 4 :=
by
  intros Ci Cu Cb hCb hCu hCi
  sorry

end cirrus_to_cumulus_is_four_l177_177204


namespace ranges_of_a_and_m_l177_177993

open Set Real

def A : Set Real := {x | x^2 - 3*x + 2 = 0}
def B (a : Real) : Set Real := {x | x^2 - a*x + a - 1 = 0}
def C (m : Real) : Set Real := {x | x^2 - m*x + 2 = 0}

theorem ranges_of_a_and_m (a m : Real) :
  A ∪ B a = A → A ∩ C m = C m → (a = 2 ∨ a = 3) ∧ (m = 3 ∨ (-2*sqrt 2 < m ∧ m < 2*sqrt 2)) :=
by
  have hA : A = {1, 2} := sorry
  sorry

end ranges_of_a_and_m_l177_177993


namespace profit_is_1500_l177_177478

def cost_per_charm : ℕ := 15
def charms_per_necklace : ℕ := 10
def sell_price_per_necklace : ℕ := 200
def necklaces_sold : ℕ := 30

def cost_per_necklace : ℕ := cost_per_charm * charms_per_necklace
def profit_per_necklace : ℕ := sell_price_per_necklace - cost_per_necklace
def total_profit : ℕ := profit_per_necklace * necklaces_sold

theorem profit_is_1500 : total_profit = 1500 :=
by
  sorry

end profit_is_1500_l177_177478


namespace longest_side_of_rectangular_solid_l177_177208

theorem longest_side_of_rectangular_solid 
  (x y z : ℝ) 
  (h1 : x * y = 20) 
  (h2 : y * z = 15) 
  (h3 : x * z = 12) 
  (h4 : x * y * z = 60) : 
  max (max x y) z = 10 := 
by sorry

end longest_side_of_rectangular_solid_l177_177208


namespace outer_circle_radius_l177_177387

theorem outer_circle_radius (C_inner : ℝ) (w : ℝ) (r_outer : ℝ) (h1 : C_inner = 440) (h2 : w = 14) :
  r_outer = (440 / (2 * Real.pi)) + 14 :=
by 
  have h_r_inner : r_outer = (440 / (2 * Real.pi)) + 14 := by sorry
  exact h_r_inner

end outer_circle_radius_l177_177387


namespace local_value_of_4_in_564823_l177_177480

def face_value (d : ℕ) : ℕ := d
def place_value_of_thousands : ℕ := 1000
def local_value (d : ℕ) (p : ℕ) : ℕ := d * p

theorem local_value_of_4_in_564823 :
  local_value (face_value 4) place_value_of_thousands = 4000 :=
by 
  sorry

end local_value_of_4_in_564823_l177_177480


namespace min_omega_value_l177_177031

theorem min_omega_value (ω : ℝ) (hω : ω > 0)
  (f : ℝ → ℝ)
  (hf_def : ∀ x, f x = Real.cos (ω * x - (Real.pi / 6))) :
  (∀ x, f x ≤ f (Real.pi / 4)) → ω = 2 / 3 :=
by
  sorry

end min_omega_value_l177_177031


namespace aqua_park_earnings_l177_177142

def admission_cost : ℕ := 12
def tour_cost : ℕ := 6
def group1_size : ℕ := 10
def group2_size : ℕ := 5

theorem aqua_park_earnings :
  (group1_size * admission_cost + group1_size * tour_cost) + (group2_size * admission_cost) = 240 :=
by
  sorry

end aqua_park_earnings_l177_177142


namespace train_speed_is_60_0131_l177_177659

noncomputable def train_speed (speed_of_man_kmh : ℝ) (length_of_train_m : ℝ) (time_s : ℝ) : ℝ :=
  let speed_of_man_ms := speed_of_man_kmh * 1000 / 3600
  let relative_speed := length_of_train_m / time_s
  let train_speed_ms := relative_speed - speed_of_man_ms
  train_speed_ms * 3600 / 1000

theorem train_speed_is_60_0131 :
  train_speed 6 330 17.998560115190788 = 60.0131 := by
  sorry

end train_speed_is_60_0131_l177_177659


namespace simplify_expression_l177_177916

variable (b : ℝ)

theorem simplify_expression :
  (2 * b + 6 - 5 * b) / 2 = -3 / 2 * b + 3 :=
sorry

end simplify_expression_l177_177916


namespace min_sum_y1_y2_l177_177709

theorem min_sum_y1_y2 (y : ℕ → ℕ) (h_seq : ∀ n ≥ 1, y (n+2) = (y n + 2013)/(1 + y (n+1))) : 
  ∃ y1 y2, y1 + y2 = 94 ∧ (∀ n, y n > 0) ∧ (y 1 = y1) ∧ (y 2 = y2) := 
sorry

end min_sum_y1_y2_l177_177709


namespace perpendicular_lines_parallel_lines_l177_177744

-- Define the given lines
def l1 (m : ℝ) (x y : ℝ) : ℝ := (m-2)*x + 3*y + 2*m
def l2 (m x y : ℝ) : ℝ := x + m*y + 6

-- The slope conditions for the lines to be perpendicular
def slopes_perpendicular (m : ℝ) : Prop :=
  (m - 2) * m = 3

-- The slope conditions for the lines to be parallel
def slopes_parallel (m : ℝ) : Prop :=
  m = -1

-- Perpendicular lines proof statement
theorem perpendicular_lines (m : ℝ) (x y : ℝ)
  (h1 : l1 m x y = 0)
  (h2 : l2 m x y = 0) :
  slopes_perpendicular m :=
sorry

-- Parallel lines proof statement
theorem parallel_lines (m : ℝ) (x y : ℝ)
  (h1 : l1 m x y = 0)
  (h2 : l2 m x y = 0) :
  slopes_parallel m :=
sorry

end perpendicular_lines_parallel_lines_l177_177744


namespace competition_result_l177_177193

variables (Olya Oleg Pasha : ℕ)

theorem competition_result 
  (h1 : Olya ≠ 1 → Olya ≠ 3 → False)
  (h2 : (Oleg = 1 ∨ Oleg = 3) → Olya = 3)
  (h3 : (Oleg ≠ 1 → (Olya = 2 ∨ Olya = 3)))
  (h4 : Olya ≠ 1 ∧ Oleg ≠ 2 ∧ Pasha ≠ 3) :
  Oleg = 1 ∧ Pasha = 2 ∧ Olya = 3 := 
by {
  sorry
}

end competition_result_l177_177193


namespace arithmetic_sequence_a7_l177_177798

variable {a : ℕ → ℚ}

def isArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (h_arith : isArithmeticSequence a) (h_a1 : a 1 = 2) (h_a3_a5 : a 3 + a 5 = 8) :
  a 7 = 6 :=
sorry

end arithmetic_sequence_a7_l177_177798


namespace cos2theta_sin2theta_l177_177424

theorem cos2theta_sin2theta (θ : ℝ) (h : 2 * Real.cos θ + Real.sin θ = 0) :
  Real.cos (2 * θ) + (1 / 2) * Real.sin (2 * θ) = -1 :=
sorry

end cos2theta_sin2theta_l177_177424


namespace average_interest_rate_equal_4_09_percent_l177_177886

-- Define the given conditions
def investment_total : ℝ := 5000
def interest_rate_at_3_percent : ℝ := 0.03
def interest_rate_at_5_percent : ℝ := 0.05
def return_relationship (x : ℝ) : Prop := 
  interest_rate_at_5_percent * x = 2 * interest_rate_at_3_percent * (investment_total - x)

-- Define the final statement
theorem average_interest_rate_equal_4_09_percent :
  ∃ x : ℝ, return_relationship x ∧ 
  ((interest_rate_at_5_percent * x + interest_rate_at_3_percent * (investment_total - x)) / investment_total) = 0.04091 := 
by
  sorry

end average_interest_rate_equal_4_09_percent_l177_177886


namespace necessarily_positive_y_plus_z_l177_177035

theorem necessarily_positive_y_plus_z
  (x y z : ℝ)
  (hx : 0 < x ∧ x < 0.5) 
  (hy : -0.5 < y ∧ y < 0) 
  (hz : 0.5 < z ∧ z < 1) :
  y + z > 0 := 
by
  sorry

end necessarily_positive_y_plus_z_l177_177035


namespace repeating_block_length_of_three_elevens_l177_177803

def smallest_repeating_block_length (x : ℚ) : ℕ :=
  sorry -- Definition to compute smallest repeating block length if needed

theorem repeating_block_length_of_three_elevens :
  smallest_repeating_block_length (3 / 11) = 2 :=
sorry

end repeating_block_length_of_three_elevens_l177_177803


namespace average_annual_cost_reduction_l177_177990

theorem average_annual_cost_reduction (x : ℝ) (h : (1 - x) ^ 2 = 0.64) : x = 0.2 :=
sorry

end average_annual_cost_reduction_l177_177990


namespace david_remaining_money_l177_177172

-- Given conditions
def hourly_rate : ℕ := 14
def hours_per_day : ℕ := 2
def days_in_week : ℕ := 7
def weekly_earnings : ℕ := hourly_rate * hours_per_day * days_in_week
def cost_of_shoes : ℕ := weekly_earnings / 2
def remaining_after_shoes : ℕ := weekly_earnings - cost_of_shoes
def given_to_mom : ℕ := remaining_after_shoes / 2
def remaining_after_gift : ℕ := remaining_after_shoes - given_to_mom

-- Theorem
theorem david_remaining_money : remaining_after_gift = 49 := by
  sorry

end david_remaining_money_l177_177172


namespace solve_inequality_l177_177878

theorem solve_inequality 
  (k_0 k b m n : ℝ)
  (hM1 : -1 = k_0 * m + b) (hM2 : -1 = k^2 / m)
  (hN1 : 2 = k_0 * n + b) (hN2 : 2 = k^2 / n) :
  {x : ℝ | x^2 > k_0 * k^2 + b * x} = {x : ℝ | x < -1 ∨ x > 2} :=
  sorry

end solve_inequality_l177_177878


namespace cone_base_radius_l177_177218

/--
Given a cone with the following properties:
1. The surface area of the cone is \(3\pi\).
2. The lateral surface of the cone unfolds into a semicircle (which implies the slant height is twice the radius of the base).
Prove that the radius of the base of the cone is \(1\).
-/
theorem cone_base_radius 
  (S : ℝ)
  (r l : ℝ)
  (h1 : S = 3 * Real.pi)
  (h2 : l = 2 * r)
  : r = 1 := 
  sorry

end cone_base_radius_l177_177218


namespace tan_x_min_x_div_x_min_sin_x_gt_two_range_of_a_l177_177718

open Real

-- Part 1
theorem tan_x_min_x_div_x_min_sin_x_gt_two (x : ℝ) (hx1 : 0 < x) (hx2 : x < π / 2) :
  (tan x - x) / (x - sin x) > 2 :=
sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < π / 2 → tan x + 2 * sin x - a * x > 0) → a ≤ 3 :=
sorry

end tan_x_min_x_div_x_min_sin_x_gt_two_range_of_a_l177_177718


namespace cube_volume_l177_177316

theorem cube_volume (s : ℝ) (h : s ^ 2 = 64) : s ^ 3 = 512 :=
sorry

end cube_volume_l177_177316


namespace expression_equality_l177_177005

theorem expression_equality : (2 + Real.sqrt 2 + 1 / (2 + Real.sqrt 2) + 1 / (Real.sqrt 2 - 2) = 2) :=
sorry

end expression_equality_l177_177005


namespace gravitational_force_at_384000km_l177_177560

theorem gravitational_force_at_384000km
  (d1 d2 : ℝ)
  (f1 f2 : ℝ)
  (k : ℝ)
  (h1 : d1 = 6400)
  (h2 : d2 = 384000)
  (h3 : f1 = 800)
  (h4 : f1 * d1^2 = k)
  (h5 : f2 * d2^2 = k) :
  f2 = 2 / 9 :=
by
  sorry

end gravitational_force_at_384000km_l177_177560


namespace crayons_lost_or_given_away_correct_l177_177541

def initial_crayons : ℕ := 606
def remaining_crayons : ℕ := 291
def crayons_lost_or_given_away : ℕ := initial_crayons - remaining_crayons

theorem crayons_lost_or_given_away_correct :
  crayons_lost_or_given_away = 315 :=
by
  sorry

end crayons_lost_or_given_away_correct_l177_177541


namespace second_part_of_ratio_l177_177357

-- Define the conditions
def ratio_percent := 20
def first_part := 4

-- Define the proof statement using the conditions
theorem second_part_of_ratio (ratio_percent : ℕ) (first_part : ℕ) : 
  ∃ second_part : ℕ, (first_part * 100) = ratio_percent * second_part :=
by
  -- Let the second part be 20 and verify the condition
  use 20
  -- Clear the proof (details are not required)
  sorry

end second_part_of_ratio_l177_177357


namespace ali_ate_half_to_percent_l177_177835

theorem ali_ate_half_to_percent : (1 / 2 : ℚ) * 100 = 50 := by
  sorry

end ali_ate_half_to_percent_l177_177835


namespace average_marks_l177_177646

-- Conditions
def marks_english : ℕ := 73
def marks_mathematics : ℕ := 69
def marks_physics : ℕ := 92
def marks_chemistry : ℕ := 64
def marks_biology : ℕ := 82
def number_of_subjects : ℕ := 5

-- Problem Statement
theorem average_marks :
  (marks_english + marks_mathematics + marks_physics + marks_chemistry + marks_biology) / number_of_subjects = 76 :=
by
  sorry

end average_marks_l177_177646


namespace length_of_side_d_l177_177492

variable (a b c d : ℕ)
variable (h_ratio1 : a / c = 3 / 4)
variable (h_ratio2 : b / d = 3 / 4)
variable (h_a : a = 3)
variable (h_b : b = 6)

theorem length_of_side_d (a b c d : ℕ)
  (h_ratio1 : a / c = 3 / 4)
  (h_ratio2 : b / d = 3 / 4)
  (h_a : a = 3)
  (h_b : b = 6) : d = 8 := 
sorry

end length_of_side_d_l177_177492


namespace equal_circles_common_point_l177_177824

theorem equal_circles_common_point (n : ℕ) (r : ℝ) 
  (centers : Fin n → ℝ × ℝ)
  (h : ∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k →
    ∃ (p : ℝ × ℝ),
      dist p (centers i) = r ∧
      dist p (centers j) = r ∧
      dist p (centers k) = r) :
  ∃ O : ℝ × ℝ, ∀ i : Fin n, dist O (centers i) = r := sorry

end equal_circles_common_point_l177_177824


namespace simplify_division_l177_177322

noncomputable def a := 5 * 10 ^ 10
noncomputable def b := 2 * 10 ^ 4 * 10 ^ 2

theorem simplify_division : a / b = 25000 := by
  sorry

end simplify_division_l177_177322


namespace jack_weight_l177_177735

-- Define weights and conditions
def weight_of_rocks : ℕ := 5 * 4
def weight_of_anna : ℕ := 40
def weight_of_jack : ℕ := weight_of_anna - weight_of_rocks

-- Prove that Jack's weight is 20 pounds
theorem jack_weight : weight_of_jack = 20 := by
  sorry

end jack_weight_l177_177735


namespace incorrect_statement_b_l177_177587

-- Defining the equation of the circle
def is_on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 25

-- Defining the point not on the circle
def is_not_on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 ≠ 25

-- The proposition to be proved
theorem incorrect_statement_b : ¬ ∀ p : ℝ × ℝ, is_not_on_circle p.1 p.2 → ¬ is_on_circle p.1 p.2 :=
by
  -- Here we should provide the proof, but this is not required based on the instructions.
  sorry

end incorrect_statement_b_l177_177587


namespace simplify_and_evaluate_l177_177168

noncomputable 
def expr (a b : ℚ) := 2*(a^2*b - 2*a*b) - 3*(a^2*b - 3*a*b) + a^2*b

theorem simplify_and_evaluate :
  let a := (-2 : ℚ) 
  let b := (1/3 : ℚ)
  expr a b = -10/3 :=
by
  sorry

end simplify_and_evaluate_l177_177168


namespace units_digit_of_7_pow_6_cubed_l177_177333

-- Define the repeating cycle of unit digits for powers of 7
def unit_digit_of_power_of_7 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 0 -- This case is actually unreachable given the modulus operation

-- Define the main problem statement
theorem units_digit_of_7_pow_6_cubed : unit_digit_of_power_of_7 (6 ^ 3) = 1 :=
by
  sorry

end units_digit_of_7_pow_6_cubed_l177_177333


namespace average_halfway_l177_177555

theorem average_halfway (a b : ℚ) (h_a : a = 1/8) (h_b : b = 1/3) : (a + b) / 2 = 11 / 48 := by
  sorry

end average_halfway_l177_177555


namespace molecular_weight_is_44_02_l177_177602

-- Definition of atomic weights and the number of atoms
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def count_N : ℕ := 2
def count_O : ℕ := 1

-- The compound's molecular weight calculation
def molecular_weight : ℝ := (count_N * atomic_weight_N) + (count_O * atomic_weight_O)

-- The proof statement that the molecular weight of the compound is approximately 44.02 amu
theorem molecular_weight_is_44_02 : molecular_weight = 44.02 := 
by
  sorry

#eval molecular_weight  -- Should output 44.02 (not part of the theorem, just for checking)

end molecular_weight_is_44_02_l177_177602


namespace boys_more_than_girls_l177_177470

theorem boys_more_than_girls
  (x y a b : ℕ)
  (h1 : x > y)
  (h2 : x * a + y * b = x * b + y * a - 1) :
  x = y + 1 :=
sorry

end boys_more_than_girls_l177_177470


namespace num_koi_fish_after_3_weeks_l177_177996

noncomputable def total_days : ℕ := 3 * 7

noncomputable def koi_fish_added_per_day : ℕ := 2
noncomputable def goldfish_added_per_day : ℕ := 5

noncomputable def total_fish_initial : ℕ := 280
noncomputable def goldfish_final : ℕ := 200

noncomputable def koi_fish_total_after_3_weeks : ℕ :=
  let koi_fish_added := koi_fish_added_per_day * total_days
  let goldfish_added := goldfish_added_per_day * total_days
  let goldfish_initial := goldfish_final - goldfish_added
  let koi_fish_initial := total_fish_initial - goldfish_initial
  koi_fish_initial + koi_fish_added

theorem num_koi_fish_after_3_weeks :
  koi_fish_total_after_3_weeks = 227 := by
  sorry

end num_koi_fish_after_3_weeks_l177_177996


namespace tangent_of_7pi_over_4_l177_177834

   theorem tangent_of_7pi_over_4 : Real.tan (7 * Real.pi / 4) = -1 := 
   sorry
   
end tangent_of_7pi_over_4_l177_177834


namespace tank_capacity_l177_177785

theorem tank_capacity (w c : ℝ) (h1 : w / c = 1 / 6) (h2 : (w + 5) / c = 1 / 3) : c = 30 :=
by
  sorry

end tank_capacity_l177_177785


namespace ratio_of_dogs_to_cats_l177_177786

theorem ratio_of_dogs_to_cats (D C : ℕ) (hC : C = 40) (h : D + 20 = 2 * C) :
  D / Nat.gcd D C = 3 ∧ C / Nat.gcd D C = 2 :=
by
  sorry

end ratio_of_dogs_to_cats_l177_177786


namespace upstream_travel_time_l177_177125

-- Define the given conditions
def downstream_time := 1 -- 1 hour
def stream_speed := 3 -- 3 kmph
def boat_speed_still_water := 15 -- 15 kmph

-- Compute the downstream speed
def downstream_speed : Nat := boat_speed_still_water + stream_speed

-- Compute the distance covered downstream
def distance_downstream : Nat := downstream_speed * downstream_time

-- Compute the upstream speed
def upstream_speed : Nat := boat_speed_still_water - stream_speed

-- The goal is to prove the time it takes to cover the distance upstream is 1.5 hours
theorem upstream_travel_time : (distance_downstream : Real) / upstream_speed = 1.5 := by
  sorry

end upstream_travel_time_l177_177125


namespace problem_solution_l177_177934

open Set

def A : Set ℕ := {x | x > 0 ∧ x < 9}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {3, 4, 5, 6}

theorem problem_solution :
  A ∩ B = {1, 2, 3} ∧
  A ∩ C = {3, 4, 5, 6} ∧
  A ∩ (B ∪ C) = {1, 2, 3, 4, 5, 6} ∧
  A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8} :=
by
  sorry

end problem_solution_l177_177934


namespace simon_legos_l177_177989

theorem simon_legos (Kent_legos : ℕ) (hk : Kent_legos = 40)
                    (Bruce_legos : ℕ) (hb : Bruce_legos = Kent_legos + 20)
                    (Simon_legos : ℕ) (hs : Simon_legos = Bruce_legos + Bruce_legos / 5) :
    Simon_legos = 72 := 
sorry

end simon_legos_l177_177989


namespace A_work_days_l177_177379

theorem A_work_days (x : ℝ) (h1 : 1 / 15 + 1 / x = 1 / 8.571428571428571) : x = 20 :=
by
  sorry

end A_work_days_l177_177379


namespace apothem_comparison_l177_177925

noncomputable def pentagon_side_length : ℝ := 4 / Real.tan (54 * Real.pi / 180)

noncomputable def pentagon_apothem : ℝ := pentagon_side_length / (2 * Real.tan (54 * Real.pi / 180))

noncomputable def hexagon_side_length : ℝ := 4 / Real.sqrt 3

noncomputable def hexagon_apothem : ℝ := (Real.sqrt 3 / 2) * hexagon_side_length

theorem apothem_comparison : pentagon_apothem = 1.06 * hexagon_apothem :=
by
  sorry

end apothem_comparison_l177_177925


namespace binary_sum_l177_177434

-- Define the binary representations in terms of their base 10 equivalent.
def binary_111111111 := 511
def binary_1111111 := 127

-- State the proof problem.
theorem binary_sum : binary_111111111 + binary_1111111 = 638 :=
by {
  -- placeholder for proof
  sorry
}

end binary_sum_l177_177434


namespace sum_of_100_and_98_consecutive_diff_digits_l177_177533

def S100 (n : ℕ) : ℕ := 50 * (2 * n + 99)
def S98 (n : ℕ) : ℕ := 49 * (2 * n + 297)

theorem sum_of_100_and_98_consecutive_diff_digits (n : ℕ) :
  ¬ (S100 n % 10 = S98 n % 10) :=
sorry

end sum_of_100_and_98_consecutive_diff_digits_l177_177533


namespace multiples_sum_squared_l177_177362

theorem multiples_sum_squared :
  let a := 4
  let b := 4
  ((a + b)^2) = 64 :=
by
  sorry

end multiples_sum_squared_l177_177362


namespace find_price_of_fourth_variety_theorem_l177_177503

-- Define the variables and conditions
variables (P1 P2 P3 P4 : ℝ) (Q1 Q2 Q3 Q4 : ℝ) (P_avg : ℝ)

-- Given conditions
def price_of_fourth_variety : Prop :=
  P1 = 126 ∧
  P2 = 135 ∧
  P3 = 156 ∧
  P_avg = 165 ∧
  Q1 / Q2 = 2 / 3 ∧
  Q1 / Q3 = 2 / 4 ∧
  Q1 / Q4 = 2 / 5 ∧
  (P1 * Q1 + P2 * Q2 + P3 * Q3 + P4 * Q4) / (Q1 + Q2 + Q3 + Q4) = P_avg

-- Prove that the price of the fourth variety of tea is Rs. 205.8 per kg
theorem find_price_of_fourth_variety_theorem : price_of_fourth_variety P1 P2 P3 P4 Q1 Q2 Q3 Q4 P_avg → P4 = 205.8 :=
by {
  sorry
}

end find_price_of_fourth_variety_theorem_l177_177503


namespace probability_diagonals_intersect_l177_177502

theorem probability_diagonals_intersect {n : ℕ} :
  (2 * n + 1 > 2) → 
  ∀ (total_diagonals : ℕ) (total_combinations : ℕ) (intersecting_pairs : ℕ),
    total_diagonals = 2 * n^2 - n - 1 →
    total_combinations = (total_diagonals * (total_diagonals - 1)) / 2 →
    intersecting_pairs = ((2 * n + 1) * n * (2 * n - 1) * (n - 1)) / 6 →
    (intersecting_pairs : ℚ) / (total_combinations : ℚ) = n * (2 * n - 1) / (3 * (2 * n^2 - n - 2)) := sorry

end probability_diagonals_intersect_l177_177502


namespace find_t_plus_a3_l177_177270

noncomputable def geometric_sequence_sum (n : ℕ) (t : ℤ) : ℤ :=
  3 ^ n + t

noncomputable def a_1 (t : ℤ) : ℤ :=
  geometric_sequence_sum 1 t

noncomputable def a_2 (t : ℤ) : ℤ :=
  geometric_sequence_sum 2 t - geometric_sequence_sum 1 t

noncomputable def a_3 (t : ℤ) : ℤ :=
  geometric_sequence_sum 3 t - geometric_sequence_sum 2 t

theorem find_t_plus_a3 (t : ℤ) : t + a_3 t = 17 :=
sorry

end find_t_plus_a3_l177_177270


namespace value_of_expression_l177_177708

theorem value_of_expression (a b c : ℝ) (h1 : 4 * a = 5 * b) (h2 : 5 * b = 30) (h3 : a + b + c = 15) : 40 * a * b / c = 1200 :=
by
  sorry

end value_of_expression_l177_177708


namespace arun_weight_lower_limit_l177_177571

variable {W B : ℝ}

theorem arun_weight_lower_limit
  (h1 : 64 < W ∧ W < 72)
  (h2 : B < W ∧ W < 70)
  (h3 : W ≤ 67)
  (h4 : (64 + 67) / 2 = 66) :
  64 < B :=
by sorry

end arun_weight_lower_limit_l177_177571


namespace find_ABC_l177_177047

theorem find_ABC {A B C : ℕ} (h₀ : ∀ n : ℕ, n ≤ 9 → n ≤ 9) (h₁ : 0 ≤ A) (h₂ : A ≤ 9) 
  (h₃ : 0 ≤ B) (h₄ : B ≤ 9) (h₅ : 0 ≤ C) (h₆ : C ≤ 9) (h₇ : 100 * A + 10 * B + C = B^C - A) :
  100 * A + 10 * B + C = 127 := by {
  sorry
}

end find_ABC_l177_177047


namespace middle_rectangle_frequency_l177_177965

theorem middle_rectangle_frequency (S A : ℝ) (h1 : S + A = 100) (h2 : A = S / 3) : A = 25 :=
by
  sorry

end middle_rectangle_frequency_l177_177965


namespace range_of_x_l177_177352

theorem range_of_x (x : ℝ) (h : |2 * x + 1| + |2 * x - 5| = 6) : -1 / 2 ≤ x ∧ x ≤ 5 / 2 := by
  sorry

end range_of_x_l177_177352


namespace depth_of_water_is_60_l177_177490

def dean_height : ℕ := 6
def depth_multiplier : ℕ := 10
def water_depth : ℕ := depth_multiplier * dean_height

theorem depth_of_water_is_60 : water_depth = 60 := by
  -- mathematical equivalent proof problem
  sorry

end depth_of_water_is_60_l177_177490


namespace fraction_sum_l177_177292

theorem fraction_sum : (1 / 3 : ℚ) + (2 / 7) + (3 / 8) = 167 / 168 := by
  sorry

end fraction_sum_l177_177292


namespace solution_set_f_inequality_l177_177016

variable (f : ℝ → ℝ)

axiom domain_of_f : ∀ x : ℝ, true
axiom avg_rate_of_f : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 3
axiom f_at_5 : f 5 = 18

theorem solution_set_f_inequality : {x : ℝ | f (3 * x - 1) > 9 * x} = {x : ℝ | x > 2} :=
by
  sorry

end solution_set_f_inequality_l177_177016


namespace fencing_required_l177_177371

theorem fencing_required (L W : ℝ) (hL : L = 20) (hArea : L * W = 60) : (L + 2 * W) = 26 := 
by
  sorry

end fencing_required_l177_177371


namespace factor_expression_l177_177207

variable (x : ℕ)

theorem factor_expression : 12 * x^3 + 6 * x^2 = 6 * x^2 * (2 * x + 1) := by
  sorry

end factor_expression_l177_177207


namespace wendy_pictures_l177_177127

theorem wendy_pictures (album1_pics rest_albums albums each_album_pics : ℕ)
    (h1 : album1_pics = 44)
    (h2 : rest_albums = 5)
    (h3 : each_album_pics = 7)
    (h4 : albums = rest_albums * each_album_pics)
    (h5 : albums = 5 * 7):
  album1_pics + albums = 79 :=
by
  -- We leave the proof as an exercise
  sorry

end wendy_pictures_l177_177127


namespace minute_hand_angle_backward_l177_177216

theorem minute_hand_angle_backward (backward_minutes : ℝ) (h : backward_minutes = 10) :
  (backward_minutes / 60) * (2 * Real.pi) = Real.pi / 3 := by
  sorry

end minute_hand_angle_backward_l177_177216


namespace mrs_wilsborough_vip_tickets_l177_177866

theorem mrs_wilsborough_vip_tickets:
  let S := 500 -- Initial savings
  let PVIP := 100 -- Price per VIP ticket
  let preg := 50 -- Price per regular ticket
  let nreg := 3 -- Number of regular tickets
  let R := 150 -- Remaining savings after purchase
  
  -- The total amount spent on tickets is S - R
  S - R = PVIP * 2 + preg * nreg := 
by sorry

end mrs_wilsborough_vip_tickets_l177_177866


namespace arithmetic_geometric_seq_l177_177712

theorem arithmetic_geometric_seq (a : ℕ → ℤ) (d : ℤ)
  (h_arith : ∀ n : ℕ, a (n + 1) = a n + d)
  (h_diff : d = 2)
  (h_geom : (a 1)^2 = a 0 * (a 0 + 6)) :
  a 1 = -6 :=
by 
  sorry

end arithmetic_geometric_seq_l177_177712


namespace time_difference_for_x_miles_l177_177404

def time_old_shoes (n : Nat) : Int := 10 * n
def time_new_shoes (n : Nat) : Int := 13 * n
def time_difference_for_5_miles : Int := time_new_shoes 5 - time_old_shoes 5

theorem time_difference_for_x_miles (x : Nat) (h : time_difference_for_5_miles = 15) : 
  time_new_shoes x - time_old_shoes x = 3 * x := 
by
  sorry

end time_difference_for_x_miles_l177_177404


namespace sufficient_not_necessary_l177_177260

theorem sufficient_not_necessary (a b : ℝ) :
  (a = -1 ∧ b = 2 → a * b = -2) ∧ (a * b = -2 → ¬(a = -1 ∧ b = 2)) :=
by
  sorry

end sufficient_not_necessary_l177_177260


namespace jack_sugar_l177_177683

theorem jack_sugar (initial_sugar : ℕ) (sugar_used : ℕ) (sugar_bought : ℕ) (final_sugar : ℕ) 
  (h1 : initial_sugar = 65) (h2 : sugar_used = 18) (h3 : sugar_bought = 50) : 
  final_sugar = initial_sugar - sugar_used + sugar_bought := 
sorry

end jack_sugar_l177_177683


namespace arithmetic_mean_first_n_positive_integers_l177_177775

theorem arithmetic_mean_first_n_positive_integers (n : ℕ) (Sn : ℕ) (h : Sn = n * (n + 1) / 2) : 
  (Sn / n) = (n + 1) / 2 := by
  -- proof steps would go here
  sorry

end arithmetic_mean_first_n_positive_integers_l177_177775


namespace tangent_line_eq_l177_177029

theorem tangent_line_eq : 
  ∀ (x y: ℝ), y = x^3 - x + 3 → (x = 1 ∧ y = 3) → (2 * x - y - 1 = 0) :=
by
  intros x y h_curve h_point
  sorry

end tangent_line_eq_l177_177029


namespace circle_center_l177_177972

theorem circle_center (x y: ℝ) : 
  (x + 2)^2 + (y + 3)^2 = 29 ↔ (∃ c1 c2 : ℝ, c1 = -2 ∧ c2 = -3) :=
by sorry

end circle_center_l177_177972


namespace shelter_cats_incoming_l177_177403

theorem shelter_cats_incoming (x : ℕ) (h : x + x / 2 - 3 + 5 - 1 = 19) : x = 12 :=
by
  sorry

end shelter_cats_incoming_l177_177403


namespace number_of_pairs_l177_177583

open Nat

theorem number_of_pairs :
  ∃ n, n = 9 ∧
    (∃ x y : ℕ,
      x > 0 ∧ y > 0 ∧
      x + y = 150 ∧
      x % 3 = 0 ∧
      y % 5 = 0 ∧
      (∃! (x y : ℕ), x + y = 150 ∧ x % 3 = 0 ∧ y % 5 = 0 ∧ x > 0 ∧ y > 0)) := sorry

end number_of_pairs_l177_177583


namespace dacid_average_l177_177202

noncomputable def average (a b : ℕ) : ℚ :=
(a + b) / 2

noncomputable def overall_average (a b c d e : ℕ) : ℚ :=
(a + b + c + d + e) / 5

theorem dacid_average :
  ∀ (english mathematics physics chemistry biology : ℕ),
  english = 86 →
  mathematics = 89 →
  physics = 82 →
  chemistry = 87 →
  biology = 81 →
  (average english mathematics < 90) ∧
  (average english physics < 90) ∧
  (average english chemistry < 90) ∧
  (average english biology < 90) ∧
  (average mathematics physics < 90) ∧
  (average mathematics chemistry < 90) ∧
  (average mathematics biology < 90) ∧
  (average physics chemistry < 90) ∧
  (average physics biology < 90) ∧
  (average chemistry biology < 90) ∧
  overall_average english mathematics physics chemistry biology = 85 := by
  intros english mathematics physics chemistry biology
  intros h_english h_mathematics h_physics h_chemistry h_biology
  simp [average, overall_average]
  rw [h_english, h_mathematics, h_physics, h_chemistry, h_biology]
  sorry

end dacid_average_l177_177202


namespace smaller_number_l177_177471

theorem smaller_number (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) : x = 3 ∨ y = 3 :=
by
  sorry

end smaller_number_l177_177471


namespace largest_initial_number_l177_177006

theorem largest_initial_number (n : ℕ) (h : (∃ a b c d e : ℕ, n ≠ 0 ∧ n + a + b + c + d + e = 200 
                                              ∧ n % a ≠ 0 ∧ n % b ≠ 0 ∧ n % c ≠ 0 ∧ n % d ≠ 0 ∧ n % e ≠ 0)) 
: n ≤ 189 :=
sorry

end largest_initial_number_l177_177006


namespace total_games_played_l177_177411

-- Define the function for combinations
def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

-- Given conditions
def teams : ℕ := 20
def games_per_pair : ℕ := 10

-- Proposition stating the target result
theorem total_games_played : 
  (combination teams 2 * games_per_pair) = 1900 :=
by
  sorry

end total_games_played_l177_177411


namespace greatest_possible_value_of_q_minus_r_l177_177351

theorem greatest_possible_value_of_q_minus_r :
  ∃ q r : ℕ, 0 < q ∧ 0 < r ∧ 852 = 21 * q + r ∧ q - r = 28 :=
by
  -- Proof goes here
  sorry

end greatest_possible_value_of_q_minus_r_l177_177351


namespace max_value_l177_177251

open Real

/-- Given vectors a, b, and c, and real numbers m and n such that m * a + n * b = c,
prove that the maximum value for (m - 3)^2 + n^2 is 16. --/
theorem max_value
  (α : ℝ)
  (a : ℝ × ℝ) (b : ℝ × ℝ) (c : ℝ × ℝ)
  (m n : ℝ)
  (ha : a = (1, 1))
  (hb : b = (1, -1))
  (hc : c = (sqrt 2 * cos α, sqrt 2 * sin α))
  (h : m * a.1 + n * b.1 = c.1 ∧ m * a.2 + n * b.2 = c.2) :
  (m - 3)^2 + n^2 ≤ 16 :=
by
  sorry

end max_value_l177_177251


namespace cookies_on_third_plate_l177_177283

theorem cookies_on_third_plate :
  ∀ (a5 a7 a14 a19 a25 : ℕ),
  (a5 = 5) ∧ (a7 = 7) ∧ (a14 = 14) ∧ (a19 = 19) ∧ (a25 = 25) →
  ∃ (a12 : ℕ), a12 = 12 :=
by
  sorry

end cookies_on_third_plate_l177_177283


namespace rectangle_diagonal_length_l177_177284

theorem rectangle_diagonal_length (p : ℝ) (r_lw : ℝ) (l w d : ℝ) 
    (h_p : p = 84) 
    (h_ratio : r_lw = 5 / 2) 
    (h_l : l = 5 * (p / 2) / 7) 
    (h_w : w = 2 * (p / 2) / 7) 
    (h_d : d = Real.sqrt (l ^ 2 + w ^ 2)) :
  d = 2 * Real.sqrt 261 :=
by
  sorry

end rectangle_diagonal_length_l177_177284


namespace right_triangle_hypotenuse_consecutive_even_l177_177580

theorem right_triangle_hypotenuse_consecutive_even (x : ℕ) (h : x ≠ 0) :
  ∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ ((a, b, c) = (x - 2, x, x + 2) ∨ (a, b, c) = (x, x - 2, x + 2) ∨ (a, b, c) = (x + 2, x, x - 2)) ∧ c = 10 := 
by
  sorry

end right_triangle_hypotenuse_consecutive_even_l177_177580


namespace problem_solution_l177_177520

theorem problem_solution :
  2 ^ 2000 - 3 * 2 ^ 1999 + 2 ^ 1998 - 2 ^ 1997 + 2 ^ 1996 = -5 * 2 ^ 1996 :=
by  -- initiate the proof script
  sorry  -- means "proof is omitted"

end problem_solution_l177_177520


namespace speed_on_way_home_l177_177378

theorem speed_on_way_home (d : ℝ) (v_up : ℝ) (v_avg : ℝ) (v_home : ℝ) 
  (h1 : v_up = 110) 
  (h2 : v_avg = 91)
  (h3 : 91 = (2 * d) / (d / 110 + d / v_home)) : 
  v_home = 10010 / 129 := 
sorry

end speed_on_way_home_l177_177378


namespace area_of_triangle_l177_177169

open Real

theorem area_of_triangle (a b c : ℝ) (A B C : ℝ) (h1 : sin A = sqrt 3 * sin C)
                        (h2 : B = π / 6) (h3 : b = 2) :
    1 / 2 * a * c * sin B = sqrt 3 :=
by
  sorry

end area_of_triangle_l177_177169


namespace fatima_donates_75_sq_inches_l177_177531

/-- Fatima starts with 100 square inches of cloth and cuts it in half twice.
    The total amount of cloth she donates should be 75 square inches. -/
theorem fatima_donates_75_sq_inches:
  ∀ (cloth_initial cloth_after_first_cut cloth_after_second_cut cloth_donated_first cloth_donated_second: ℕ),
  cloth_initial = 100 → 
  cloth_after_first_cut = cloth_initial / 2 →
  cloth_donated_first = cloth_initial / 2 →
  cloth_after_second_cut = cloth_after_first_cut / 2 →
  cloth_donated_second = cloth_after_first_cut / 2 →
  cloth_donated_first + cloth_donated_second = 75 := 
by
  intros cloth_initial cloth_after_first_cut cloth_after_second_cut cloth_donated_first cloth_donated_second
  intros h_initial h_after_first h_donated_first h_after_second h_donated_second
  sorry

end fatima_donates_75_sq_inches_l177_177531


namespace bus_driver_total_hours_l177_177155

variables (R OT : ℕ)

-- Conditions
def regular_rate := 16
def overtime_rate := 28
def max_regular_hours := 40
def total_compensation := 864

-- Proof goal: total hours worked is 48
theorem bus_driver_total_hours :
  (regular_rate * R + overtime_rate * OT = total_compensation) →
  (R ≤ max_regular_hours) →
  (R + OT = 48) :=
by
  sorry

end bus_driver_total_hours_l177_177155


namespace treasure_chest_age_l177_177706

theorem treasure_chest_age (n : ℕ) (h : n = 3 * 8^2 + 4 * 8^1 + 7 * 8^0) : n = 231 :=
by
  sorry

end treasure_chest_age_l177_177706


namespace Freddy_age_l177_177438

noncomputable def M : ℕ := 11
noncomputable def R : ℕ := M - 2
noncomputable def F : ℕ := M + 4

theorem Freddy_age : F = 15 :=
  by
    sorry

end Freddy_age_l177_177438


namespace hanging_spheres_ratio_l177_177983

theorem hanging_spheres_ratio (m1 m2 g T_B T_H : ℝ)
  (h1 : T_B = 3 * T_H)
  (h2 : T_H = m2 * g)
  (h3 : T_B = m1 * g + T_H)
  : m1 / m2 = 2 :=
by
  sorry

end hanging_spheres_ratio_l177_177983


namespace min_total_rope_cut_l177_177513

theorem min_total_rope_cut (len1 len2 len3 p1 p2 p3 p4: ℕ) (hl1 : len1 = 52) (hl2 : len2 = 37)
  (hl3 : len3 = 25) (hp1 : p1 = 7) (hp2 : p2 = 3) (hp3 : p3 = 1) 
  (hp4 : ∃ x y z : ℕ, x * p1 + y * p2 + z * p3 = len1 + len2 - len3 ∧ x + y + z ≤ 25) :
  p4 = 82 := 
sorry

end min_total_rope_cut_l177_177513


namespace largest_and_next_largest_difference_l177_177797

theorem largest_and_next_largest_difference (a b c : ℕ) (h1: a = 10) (h2: b = 11) (h3: c = 12) : 
  let largest := max a (max b c)
  let next_largest := min (max a b) (max (min a b) c)
  largest - next_largest = 1 :=
by
  -- Proof to be filled in for verification
  sorry

end largest_and_next_largest_difference_l177_177797


namespace investment_simple_compound_l177_177144

theorem investment_simple_compound (P y : ℝ) 
    (h1 : 600 = P * y * 2 / 100)
    (h2 : 615 = P * (1 + y/100)^2 - P) : 
    P = 285.71 :=
by
    sorry

end investment_simple_compound_l177_177144


namespace sheep_per_herd_l177_177131

theorem sheep_per_herd (herds : ℕ) (total_sheep : ℕ) (h_herds : herds = 3) (h_total_sheep : total_sheep = 60) : 
  (total_sheep / herds) = 20 :=
by
  sorry

end sheep_per_herd_l177_177131


namespace composite_product_quotient_l177_177298

def first_seven_composite := [4, 6, 8, 9, 10, 12, 14]
def next_eight_composite := [15, 16, 18, 20, 21, 22, 24, 25]

noncomputable def product {α : Type*} [Monoid α] (l : List α) : α :=
  l.foldl (· * ·) 1

theorem composite_product_quotient : 
  (product first_seven_composite : ℚ) / (product next_eight_composite : ℚ) = 1 / 2475 := 
by 
  sorry

end composite_product_quotient_l177_177298


namespace seven_thousand_twenty_two_is_7022_l177_177044

-- Define the translations of words to numbers
def seven_thousand : ℕ := 7000
def twenty_two : ℕ := 22

-- Define the full number by summing its parts
def seven_thousand_twenty_two : ℕ := seven_thousand + twenty_two

theorem seven_thousand_twenty_two_is_7022 : seven_thousand_twenty_two = 7022 := by
  sorry

end seven_thousand_twenty_two_is_7022_l177_177044


namespace geometric_sequence_sum_l177_177010

noncomputable def seq (a : ℕ → ℝ) : Prop :=
∀ n ≥ 2, a n ^ 2 = a (n - 1) * a (n + 1)

theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h_seq : seq a)
  (h_a2 : a 2 = 3)
  (h_sum : a 2 + a 4 + a 6 = 21) :
  (a 4 + a 6 + a 8) = 42 :=
sorry

end geometric_sequence_sum_l177_177010


namespace number_of_pairings_l177_177265

-- Definitions for conditions.
def bowls : Finset String := {"red", "blue", "yellow", "green"}
def glasses : Finset String := {"red", "blue", "yellow", "green"}

-- The theorem statement
theorem number_of_pairings : bowls.card * glasses.card = 16 := by
  sorry

end number_of_pairings_l177_177265


namespace program_arrangement_possible_l177_177889

theorem program_arrangement_possible (initial_programs : ℕ) (additional_programs : ℕ) 
  (h_initial: initial_programs = 6) (h_additional: additional_programs = 2) : 
  ∃ arrangements, arrangements = 56 :=
by
  sorry

end program_arrangement_possible_l177_177889


namespace weight_per_linear_foot_l177_177205

theorem weight_per_linear_foot 
  (length_of_log : ℕ) 
  (cut_length : ℕ) 
  (piece_weight : ℕ) 
  (h1 : length_of_log = 20) 
  (h2 : cut_length = length_of_log / 2) 
  (h3 : piece_weight = 1500) 
  (h4 : length_of_log / 2 = 10) 
  : piece_weight / cut_length = 150 := 
  by 
  sorry

end weight_per_linear_foot_l177_177205


namespace problem_l177_177374

theorem problem (x y : ℕ) (hy : y > 3) (h : x^2 + y^4 = 2 * ((x-6)^2 + (y+1)^2)) : x^2 + y^4 = 1994 := by
  sorry

end problem_l177_177374


namespace johnny_fishes_l177_177674

theorem johnny_fishes (total_fishes sony_multiple j : ℕ) (h1 : total_fishes = 120) (h2 : sony_multiple = 7) (h3 : total_fishes = j + sony_multiple * j) : j = 15 :=
by sorry

end johnny_fishes_l177_177674


namespace cost_effectiveness_order_l177_177238

variables {cS cM cL qS qM qL : ℝ}
variables (h1 : cM = 2 * cS)
variables (h2 : qM = 0.7 * qL)
variables (h3 : qL = 3 * qS)
variables (h4 : cL = 1.2 * cM)

theorem cost_effectiveness_order :
  (cL / qL <= cM / qM) ∧ (cM / qM <= cS / qS) :=
by
  sorry

end cost_effectiveness_order_l177_177238


namespace travel_time_reduction_l177_177601

theorem travel_time_reduction
  (original_speed : ℝ)
  (new_speed : ℝ)
  (time : ℝ)
  (distance : ℝ)
  (new_time : ℝ)
  (h1 : original_speed = 80)
  (h2 : new_speed = 50)
  (h3 : time = 3)
  (h4 : distance = original_speed * time)
  (h5 : new_time = distance / new_speed) :
  new_time = 4.8 := 
sorry

end travel_time_reduction_l177_177601


namespace intersecting_lines_solution_l177_177819

theorem intersecting_lines_solution (a b : ℝ) :
  (∃ (a b : ℝ), 
    ((a^2 + 1) * 2 - 2 * b * (-3) = 4) ∧ 
    ((1 - a) * 2 + b * (-3) = 9)) →
  (a, b) = (4, -5) ∨ (a, b) = (-2, -1) :=
by
  sorry

end intersecting_lines_solution_l177_177819


namespace expansion_coefficient_a2_l177_177940

theorem expansion_coefficient_a2 : 
  (∃ (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ), 
    (1 - 2*x)^7 = a + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 + a_6*x^6 + a_7*x^7 -> 
    a_2 = 84) :=
sorry

end expansion_coefficient_a2_l177_177940


namespace eleven_y_minus_x_eq_one_l177_177040

theorem eleven_y_minus_x_eq_one 
  (x y : ℤ) 
  (hx_pos : x > 0)
  (h1 : x = 7 * y + 3)
  (h2 : 2 * x = 6 * (3 * y) + 2) : 
  11 * y - x = 1 := 
by 
  sorry

end eleven_y_minus_x_eq_one_l177_177040


namespace weight_loss_percentage_l177_177390

variables (W : ℝ) (x : ℝ)

def weight_loss_challenge :=
  W - W * x / 100 + W * 2 / 100 = W * 86.7 / 100

theorem weight_loss_percentage (h : weight_loss_challenge W x) : x = 15.3 :=
by sorry

end weight_loss_percentage_l177_177390


namespace jerry_remaining_money_l177_177941

-- Define initial money
def initial_money := 18

-- Define amount spent on video games
def spent_video_games := 6

-- Define amount spent on a snack
def spent_snack := 3

-- Define total amount spent
def total_spent := spent_video_games + spent_snack

-- Define remaining money after spending
def remaining_money := initial_money - total_spent

theorem jerry_remaining_money : remaining_money = 9 :=
by
  sorry

end jerry_remaining_money_l177_177941


namespace find_f_19_l177_177109

noncomputable def f : ℝ → ℝ := sorry -- placeholder for the given function

-- Define the conditions
axiom even_function : ∀ x : ℝ, f x = f (-x) 
axiom periodicity : ∀ x : ℝ, f (x + 2) = -f x

-- The statement we need to prove
theorem find_f_19 : f 19 = 0 := 
by
  sorry -- placeholder for the proof

end find_f_19_l177_177109


namespace average_speed_correct_l177_177129

-- Definitions for the conditions
def distance1 : ℚ := 40
def speed1 : ℚ := 8
def time1 : ℚ := distance1 / speed1

def distance2 : ℚ := 20
def speed2 : ℚ := 40
def time2 : ℚ := distance2 / speed2

def total_distance : ℚ := distance1 + distance2
def total_time : ℚ := time1 + time2

-- Definition of average speed
def average_speed : ℚ := total_distance / total_time

-- Proof statement that needs to be proven
theorem average_speed_correct : average_speed = 120 / 11 :=
by 
  -- The details for the proof will be filled here
  sorry

end average_speed_correct_l177_177129


namespace incorrect_conclusion_l177_177139

noncomputable def quadratic (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 2) * x + 2

theorem incorrect_conclusion (m : ℝ) (hx : m - 2 = 0) :
  ¬(∀ x : ℝ, quadratic m x = 2 ↔ x = 2) :=
by
  sorry

end incorrect_conclusion_l177_177139


namespace smallest_sum_is_S5_l177_177266

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Definitions of arithmetic sequence sum
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Conditions
axiom h1 : a 3 + a 8 > 0
axiom h2 : S 9 < 0

-- Statements relating terms and sums in arithmetic sequence
axiom h3 : ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem smallest_sum_is_S5 (seq_a : arithmetic_sequence a) : S 5 ≤ S 1 ∧ S 5 ≤ S 2 ∧ S 5 ≤ S 3 ∧ S 5 ≤ S 4 ∧ S 5 ≤ S 6 ∧ S 5 ≤ S 7 ∧ S 5 ≤ S 8 ∧ S 5 ≤ S 9 :=
by {
    sorry
}

end smallest_sum_is_S5_l177_177266


namespace intersection_M_N_l177_177257

-- Define the sets M and N according to the conditions given in the problem
def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

-- State the theorem to prove the intersection of M and N
theorem intersection_M_N : (M ∩ N) = {0, 1} := 
  sorry

end intersection_M_N_l177_177257


namespace orange_marbles_l177_177220

-- Definitions based on the given conditions
def total_marbles : ℕ := 24
def blue_marbles : ℕ := total_marbles / 2
def red_marbles : ℕ := 6

-- The statement to prove: the number of orange marbles is 6
theorem orange_marbles : (total_marbles - (blue_marbles + red_marbles)) = 6 := 
  by 
  sorry

end orange_marbles_l177_177220


namespace minimum_value_of_function_l177_177849

noncomputable def y (x : ℝ) : ℝ := 4 * x + 25 / x

theorem minimum_value_of_function : ∃ x > 0, y x = 20 :=
by
  sorry

end minimum_value_of_function_l177_177849


namespace least_four_digit_multiple_3_5_7_l177_177752

theorem least_four_digit_multiple_3_5_7 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ n = 1050 :=
by
  use 1050
  repeat {sorry}

end least_four_digit_multiple_3_5_7_l177_177752


namespace problem_inequality_l177_177804

variable (a b : ℝ)

theorem problem_inequality (h1 : a < 0) (h2 : -1 < b) (h3 : b < 0) : ab > ab^2 ∧ ab^2 > a :=
by
  sorry

end problem_inequality_l177_177804


namespace triangle_has_side_property_l177_177052

theorem triangle_has_side_property (a b c : ℝ) (A B C : ℝ) 
  (h₀ : 3 * b * Real.cos C + 3 * c * Real.cos B = a^2)
  (h₁ : A + B + C = Real.pi)
  (h₂ : a = 3) :
  a = 3 := 
sorry

end triangle_has_side_property_l177_177052


namespace arithmetic_seq_formula_l177_177311

variable (a : ℕ → ℤ)

-- Given conditions
axiom h1 : a 1 + a 2 + a 3 = 0
axiom h2 : a 4 + a 5 + a 6 = 18

-- Goal: general formula for the arithmetic sequence
theorem arithmetic_seq_formula (n : ℕ) : a n = 2 * n - 4 := by
  sorry

end arithmetic_seq_formula_l177_177311


namespace sqrt_neg_square_real_l177_177440

theorem sqrt_neg_square_real : ∃! (x : ℝ), -(x + 2) ^ 2 = 0 := by
  sorry

end sqrt_neg_square_real_l177_177440


namespace james_total_matches_l177_177012

-- Define the conditions
def dozen : Nat := 12
def boxes_per_dozen : Nat := 5
def matches_per_box : Nat := 20

-- Calculate the expected number of matches
def expected_matches : Nat := boxes_per_dozen * dozen * matches_per_box

-- State the theorem to be proved
theorem james_total_matches : expected_matches = 1200 := by
  sorry

end james_total_matches_l177_177012


namespace number_of_diagonals_in_hexagon_l177_177275

-- Define the number of sides of the hexagon
def sides_of_hexagon : ℕ := 6

-- Define the formula for the number of diagonals in an n-sided polygon
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- The theorem we want to prove
theorem number_of_diagonals_in_hexagon : number_of_diagonals sides_of_hexagon = 9 :=
by
  sorry

end number_of_diagonals_in_hexagon_l177_177275


namespace cube_volume_ratio_l177_177982

theorem cube_volume_ratio (edge1 edge2 : ℕ) (h1 : edge1 = 10) (h2 : edge2 = 36) :
  (edge1^3 : ℚ) / (edge2^3) = 125 / 5832 :=
by
  sorry

end cube_volume_ratio_l177_177982


namespace gather_all_candies_l177_177857

theorem gather_all_candies (n : ℕ) (h₁ : n ≥ 4) (candies : ℕ) (h₂ : candies ≥ 4)
    (plates : Fin n → ℕ) :
    ∃ plate : Fin n, ∀ i : Fin n, i ≠ plate → plates i = 0 :=
sorry

end gather_all_candies_l177_177857


namespace cars_produced_total_l177_177969

theorem cars_produced_total :
  3884 + 2871 = 6755 :=
by
  sorry

end cars_produced_total_l177_177969


namespace initial_weasels_count_l177_177588

theorem initial_weasels_count (initial_rabbits : ℕ) (foxes : ℕ) (weasels_per_fox : ℕ) (rabbits_per_fox : ℕ) 
                              (weeks : ℕ) (remaining_rabbits_weasels : ℕ) (initial_weasels : ℕ) 
                              (total_rabbits_weasels : ℕ) : 
    initial_rabbits = 50 → foxes = 3 → weasels_per_fox = 4 → rabbits_per_fox = 2 → weeks = 3 → 
    remaining_rabbits_weasels = 96 → total_rabbits_weasels = initial_rabbits + initial_weasels → initial_weasels = 100 :=
by
  sorry

end initial_weasels_count_l177_177588


namespace striped_jerseys_count_l177_177488

noncomputable def totalSpent : ℕ := 80
noncomputable def longSleevedJerseyCost : ℕ := 15
noncomputable def stripedJerseyCost : ℕ := 10
noncomputable def numberOfLongSleevedJerseys : ℕ := 4

theorem striped_jerseys_count :
  (totalSpent - numberOfLongSleevedJerseys * longSleevedJerseyCost) / stripedJerseyCost = 2 := by
  sorry

end striped_jerseys_count_l177_177488


namespace double_grandfather_pension_l177_177779

-- Define the total family income and individual contributions
def total_income (masha mother father grandfather : ℝ) : ℝ :=
  masha + mother + father + grandfather

-- Define the conditions provided in the problem
variables
  (masha mother father grandfather : ℝ)
  (cond1 : 2 * masha = total_income masha mother father grandfather * 1.05)
  (cond2 : 2 * mother = total_income masha mother father grandfather * 1.15)
  (cond3 : 2 * father = total_income masha mother father grandfather * 1.25)

-- Define the statement to be proved
theorem double_grandfather_pension :
  2 * grandfather = total_income masha mother father grandfather * 1.55 :=
by
  -- Proof placeholder
  sorry

end double_grandfather_pension_l177_177779


namespace paving_cost_l177_177705

def length : Real := 5.5
def width : Real := 3.75
def rate : Real := 700
def area : Real := length * width
def cost : Real := area * rate

theorem paving_cost :
  cost = 14437.50 :=
by
  -- Proof steps go here
  sorry

end paving_cost_l177_177705


namespace bus_arrival_time_at_first_station_l177_177405

noncomputable def time_to_first_station (start_time end_time first_station_to_work: ℕ) : ℕ :=
  (end_time - start_time) - first_station_to_work

theorem bus_arrival_time_at_first_station :
  time_to_first_station 360 540 140 = 40 :=
by
  -- provide the proof here, which has been omitted per the instructions
  sorry

end bus_arrival_time_at_first_station_l177_177405


namespace distance_between_A_and_B_l177_177818

theorem distance_between_A_and_B 
  (v t t1 : ℝ)
  (h1 : 5 * v * t + 4 * v * t = 9 * v * t)
  (h2 : t1 = 10 / (4.8 * v))
  (h3 : 10 / 4.8 = 25 / 12):
  (9 * v * t + 4 * v * t1) = 450 :=
by 
  -- Proof to be completed
  sorry

end distance_between_A_and_B_l177_177818


namespace tim_income_less_than_juan_l177_177874

-- Definitions of the conditions
variables {T J M : ℝ}
def mart_income_condition1 (M T : ℝ) : Prop := M = 1.40 * T
def mart_income_condition2 (M J : ℝ) : Prop := M = 0.84 * J

-- The proof goal
theorem tim_income_less_than_juan (T J M : ℝ) 
(h1: mart_income_condition1 M T) 
(h2: mart_income_condition2 M J) : 
T = 0.60 * J :=
by
  sorry

end tim_income_less_than_juan_l177_177874


namespace certain_number_eq_neg17_l177_177768

theorem certain_number_eq_neg17 (x : Int) : 47 + x = 30 → x = -17 := by
  intro h
  have : x = 30 - 47 := by
    sorry  -- This is just to demonstrate the proof step. Actual manipulation should prove x = -17
  simp [this]

end certain_number_eq_neg17_l177_177768


namespace team_A_more_uniform_l177_177963

noncomputable def average_height : ℝ := 2.07

variables (S_A S_B : ℝ) (h_variance : S_A^2 < S_B^2)

theorem team_A_more_uniform : true ∧ false :=
by
  sorry

end team_A_more_uniform_l177_177963


namespace trig_identity_solution_l177_177091

theorem trig_identity_solution
  (x : ℝ)
  (h : Real.sin (x + Real.pi / 4) = 1 / 3) :
  Real.sin (4 * x) - 2 * Real.cos (3 * x) * Real.sin x = -7 / 9 :=
by
  sorry

end trig_identity_solution_l177_177091


namespace necklace_length_l177_177274

-- Given conditions as definitions in Lean
def num_pieces : ℕ := 16
def piece_length : ℝ := 10.4
def overlap_length : ℝ := 3.5
def effective_length : ℝ := piece_length - overlap_length
def total_length : ℝ := effective_length * num_pieces

-- The theorem to prove
theorem necklace_length :
  total_length = 110.4 :=
by
  -- Proof omitted
  sorry

end necklace_length_l177_177274


namespace coordinates_of_C_l177_177416

structure Point :=
  (x : Int)
  (y : Int)

def reflect_over_x_axis (p : Point) : Point :=
  {x := p.x, y := -p.y}

def reflect_over_y_axis (p : Point) : Point :=
  {x := -p.x, y := p.y}

def C : Point := {x := 2, y := 2}

noncomputable def C'_reflected_x := reflect_over_x_axis C
noncomputable def C''_reflected_y := reflect_over_y_axis C'_reflected_x

theorem coordinates_of_C'' : C''_reflected_y = {x := -2, y := -2} :=
by
  sorry

end coordinates_of_C_l177_177416


namespace multiplier_for_ab_to_equal_1800_l177_177665

variable (a b m : ℝ)
variable (h1 : 4 * a = 30)
variable (h2 : 5 * b = 30)
variable (h3 : a * b = 45)
variable (h4 : m * (a * b) = 1800)

theorem multiplier_for_ab_to_equal_1800 (h1 : 4 * a = 30) (h2 : 5 * b = 30) (h3 : a * b = 45) (h4 : m * (a * b) = 1800) :
  m = 40 :=
sorry

end multiplier_for_ab_to_equal_1800_l177_177665


namespace morio_current_age_l177_177896

-- Given conditions
def teresa_current_age : ℕ := 59
def morio_age_when_michiko_born : ℕ := 38
def teresa_age_when_michiko_born : ℕ := 26

-- Definitions derived from the conditions
def michiko_age : ℕ := teresa_current_age - teresa_age_when_michiko_born

-- Statement to prove Morio's current age
theorem morio_current_age : (michiko_age + morio_age_when_michiko_born) = 71 :=
by
  sorry

end morio_current_age_l177_177896


namespace tan_2016_l177_177034

-- Define the given condition
def sin_36 (a : ℝ) : Prop := Real.sin (36 * Real.pi / 180) = a

-- Prove the required statement given the condition
theorem tan_2016 (a : ℝ) (h : sin_36 a) : Real.tan (2016 * Real.pi / 180) = a / Real.sqrt (1 - a^2) :=
sorry

end tan_2016_l177_177034


namespace ratio_of_areas_l177_177020

theorem ratio_of_areas (s : ℝ) (h_s_pos : 0 < s) :
  let small_triangle_area := (s^2 * Real.sqrt 3) / 4
  let total_small_triangles_area := 6 * small_triangle_area
  let large_triangle_side := 6 * s
  let large_triangle_area := (large_triangle_side^2 * Real.sqrt 3) / 4
  total_small_triangles_area / large_triangle_area = 1 / 6 :=
by
  let small_triangle_area := (s^2 * Real.sqrt 3) / 4
  let total_small_triangles_area := 6 * small_triangle_area
  let large_triangle_side := 6 * s
  let large_triangle_area := (large_triangle_side^2 * Real.sqrt 3) / 4
  sorry
 
end ratio_of_areas_l177_177020


namespace fries_sold_l177_177960

theorem fries_sold (small_fries large_fries : ℕ) (h1 : small_fries = 4) (h2 : large_fries = 5 * small_fries) :
  small_fries + large_fries = 24 :=
  by
    sorry

end fries_sold_l177_177960


namespace keith_score_l177_177904

theorem keith_score (K : ℕ) (h : K + 3 * K + (3 * K + 5) = 26) : K = 3 :=
by
  sorry

end keith_score_l177_177904


namespace outfit_count_l177_177236

section OutfitProblem

-- Define the number of each type of shirts, pants, and hats
def num_red_shirts : ℕ := 7
def num_blue_shirts : ℕ := 5
def num_green_shirts : ℕ := 8

def num_pants : ℕ := 10

def num_green_hats : ℕ := 10
def num_red_hats : ℕ := 6
def num_blue_hats : ℕ := 7

-- The main theorem to prove the number of outfits where shirt and hat are not the same color
theorem outfit_count : 
  (num_red_shirts * num_pants * (num_green_hats + num_blue_hats) +
  num_blue_shirts * num_pants * (num_green_hats + num_red_hats) +
  num_green_shirts * num_pants * (num_red_hats + num_blue_hats)) = 3030 :=
  sorry

end OutfitProblem

end outfit_count_l177_177236


namespace relationship_between_y_values_l177_177895

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + |b| * x + c

theorem relationship_between_y_values
  (a b c y1 y2 y3 : ℝ)
  (h1 : quadratic_function a b c (-14 / 3) = y1)
  (h2 : quadratic_function a b c (5 / 2) = y2)
  (h3 : quadratic_function a b c 3 = y3)
  (axis_symmetry : -(|b| / (2 * a)) = -1)
  (h_pos : 0 < a) :
  y2 < y1 ∧ y1 < y3 :=
by
  sorry

end relationship_between_y_values_l177_177895


namespace fourth_power_sum_l177_177430

theorem fourth_power_sum
  (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 19.5 := 
sorry

end fourth_power_sum_l177_177430


namespace aquafaba_needed_for_cakes_l177_177636

def tablespoons_of_aquafaba_for_egg_whites (n_egg_whites : ℕ) : ℕ :=
  2 * n_egg_whites

def total_egg_whites_needed (cakes : ℕ) (egg_whites_per_cake : ℕ) : ℕ :=
  cakes * egg_whites_per_cake

theorem aquafaba_needed_for_cakes (cakes : ℕ) (egg_whites_per_cake : ℕ) :
  tablespoons_of_aquafaba_for_egg_whites (total_egg_whites_needed cakes egg_whites_per_cake) = 32 :=
by
  have h1 : cakes = 2 := sorry
  have h2 : egg_whites_per_cake = 8 := sorry
  sorry

end aquafaba_needed_for_cakes_l177_177636


namespace n_gon_partition_l177_177101

-- Define a function to determine if an n-gon can be partitioned as required
noncomputable def canBePartitioned (n : ℕ) (h : n ≥ 3) : Prop :=
  n ≠ 4 ∧ n ≥ 3

theorem n_gon_partition (n : ℕ) (h : n ≥ 3) : canBePartitioned n h ↔ (n = 3 ∨ n ≥ 5) :=
by sorry

end n_gon_partition_l177_177101


namespace cost_per_pack_l177_177191

variable (total_amount : ℕ) (number_of_packs : ℕ)

theorem cost_per_pack (h1 : total_amount = 132) (h2 : number_of_packs = 11) : 
  total_amount / number_of_packs = 12 := by
  sorry

end cost_per_pack_l177_177191


namespace place_two_in_front_l177_177516

-- Define the conditions: the original number has hundreds digit h, tens digit t, and units digit u.
variables (h t u : ℕ)

-- Define the function representing the placement of the digit 2 before the three-digit number.
def new_number (h t u : ℕ) : ℕ :=
  2000 + 100 * h + 10 * t + u

-- State the theorem that proves the new number formed is as stated.
theorem place_two_in_front : new_number h t u = 2000 + 100 * h + 10 * t + u :=
by sorry

end place_two_in_front_l177_177516


namespace model_to_statue_scale_l177_177703

theorem model_to_statue_scale
  (statue_height_ft : ℕ)
  (model_height_in : ℕ)
  (ft_to_in : ℕ)
  (statue_height_in : ℕ)
  (scale : ℕ)
  (h1 : statue_height_ft = 120)
  (h2 : model_height_in = 6)
  (h3 : ft_to_in = 12)
  (h4 : statue_height_in = statue_height_ft * ft_to_in)
  (h5 : scale = (statue_height_in / model_height_in) / ft_to_in) : scale = 20 := 
  sorry

end model_to_statue_scale_l177_177703


namespace inversely_proportional_example_l177_177419

theorem inversely_proportional_example (x y k : ℝ) (h₁ : x * y = k) (h₂ : x = 30) (h₃ : y = 8) :
  y = 24 → x = 10 :=
by
  sorry

end inversely_proportional_example_l177_177419


namespace triangular_region_area_l177_177381

noncomputable def area_of_triangle (f g h : ℝ → ℝ) : ℝ :=
  let (x1, y1) := (-3, f (-3))
  let (x2, y2) := (7/3, g (7/3))
  let (x3, y3) := (15/11, f (15/11))
  let base := abs (x2 - x1)
  let height := abs (y3 - 2)
  (1/2) * base * height

theorem triangular_region_area :
  let f x := (2/3) * x + 4
  let g x := -3 * x + 9
  let h x := (2 : ℝ)
  area_of_triangle f g h = 256/33 :=  -- Given conditions
by
  sorry  -- Proof to be supplied

end triangular_region_area_l177_177381


namespace tangent_line_eq_l177_177501

theorem tangent_line_eq (x y : ℝ) (h : y = e^(-5 * x) + 2) :
  ∀ (t : ℝ), t = 0 → y = 3 → y = -5 * x + 3 :=
by
  sorry

end tangent_line_eq_l177_177501


namespace sequence_monotonically_increasing_l177_177337

noncomputable def a (n : ℕ) : ℝ := (n - 1 : ℝ) / (n + 1 : ℝ)

theorem sequence_monotonically_increasing : ∀ n : ℕ, a (n + 1) > a n :=
by
  sorry

end sequence_monotonically_increasing_l177_177337


namespace train_start_time_l177_177443

theorem train_start_time (D PQ : ℝ) (S₁ S₂ : ℝ) (T₁ T₂ meet : ℝ) :
  PQ = 110  -- Distance between stations P and Q
  ∧ S₁ = 20  -- Speed of the first train
  ∧ S₂ = 25  -- Speed of the second train
  ∧ T₂ = 8  -- Start time of the second train
  ∧ meet = 10 -- Meeting time
  ∧ T₁ + T₂ = meet → -- Meeting time condition
  T₁ = 7.5 := -- Answer: first train start time
by
sorry

end train_start_time_l177_177443


namespace num_bicycles_l177_177079

theorem num_bicycles (spokes_per_wheel wheels_per_bicycle total_spokes : ℕ) (h1 : spokes_per_wheel = 10) (h2 : total_spokes = 80) (h3 : wheels_per_bicycle = 2) : total_spokes / spokes_per_wheel / wheels_per_bicycle = 4 := by
  sorry

end num_bicycles_l177_177079


namespace total_volume_of_cubes_l177_177069

theorem total_volume_of_cubes (s : ℕ) (n : ℕ) (h_s : s = 5) (h_n : n = 4) : 
  n * s^3 = 500 :=
by
  sorry

end total_volume_of_cubes_l177_177069


namespace profit_per_meter_is_15_l177_177376

def sellingPrice (meters : ℕ) : ℕ := 
    if meters = 85 then 8500 else 0

def costPricePerMeter : ℕ := 85

def totalCostPrice (meters : ℕ) : ℕ := 
    meters * costPricePerMeter

def totalProfit (meters : ℕ) (sellingPrice : ℕ) (costPrice : ℕ) : ℕ := 
    sellingPrice - costPrice

def profitPerMeter (profit : ℕ) (meters : ℕ) : ℕ := 
    profit / meters

theorem profit_per_meter_is_15 : profitPerMeter (totalProfit 85 (sellingPrice 85) (totalCostPrice 85)) 85 = 15 := 
by sorry

end profit_per_meter_is_15_l177_177376


namespace angle_bisector_length_l177_177331

variable (a b : ℝ) (α l : ℝ)

theorem angle_bisector_length (ha : 0 < a) (hb : 0 < b) (hα : 0 < α) (hl : l = (2 * a * b * Real.cos (α / 2)) / (a + b)) :
  l = (2 * a * b * Real.cos (α / 2)) / (a + b) := by
  -- problem assumptions
  have h1 : a > 0 := ha
  have h2 : b > 0 := hb
  have h3 : α > 0 := hα
  -- conclusion
  exact hl

end angle_bisector_length_l177_177331


namespace intersection_of_sets_l177_177732

open Set Real

theorem intersection_of_sets :
  let M := {x : ℝ | x ≤ 4}
  let N := {x : ℝ | x > 0}
  M ∩ N = {x : ℝ | 0 < x ∧ x ≤ 4} :=
by
  sorry

end intersection_of_sets_l177_177732


namespace sum_digits_10_pow_100_minus_100_l177_177772

open Nat

/-- Define the condition: 10^100 - 100 as an expression. -/
def subtract_100_from_power_10 (n : ℕ) : ℕ :=
  10^n - 100

/-- Sum the digits of a natural number. -/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- The goal is to prove the sum of the digits of 10^100 - 100 equals 882. -/
theorem sum_digits_10_pow_100_minus_100 :
  sum_of_digits (subtract_100_from_power_10 100) = 882 :=
by
  sorry

end sum_digits_10_pow_100_minus_100_l177_177772


namespace poodle_terrier_bark_ratio_l177_177253

theorem poodle_terrier_bark_ratio :
  ∀ (P T : ℕ),
  (T = 12) →
  (P = 24) →
  (P / T = 2) :=
by intros P T hT hP
   sorry

end poodle_terrier_bark_ratio_l177_177253


namespace right_triangle_hypotenuse_l177_177893

theorem right_triangle_hypotenuse (a b c : ℕ) (h : a = 3) (h' : b = 4) (hc : c^2 = a^2 + b^2) : c = 5 := 
by
  -- proof goes here
  sorry

end right_triangle_hypotenuse_l177_177893


namespace chessboard_not_divisible_by_10_l177_177104

theorem chessboard_not_divisible_by_10 :
  ∀ (B : ℕ × ℕ → ℕ), 
  (∀ x y, B (x, y) < 10) ∧ 
  (∀ x y, x ≥ 0 ∧ x < 8 ∧ y ≥ 0 ∧ y < 8) →
  ¬ ( ∃ k : ℕ, ∀ x y, (B (x, y) + k) % 10 = 0 ) :=
by
  intros
  sorry

end chessboard_not_divisible_by_10_l177_177104


namespace classroom_needs_more_money_l177_177651

theorem classroom_needs_more_money 
    (goal : ℕ) 
    (raised_from_two_families : ℕ) 
    (raised_from_eight_families : ℕ) 
    (raised_from_ten_families : ℕ) 
    (H : goal = 200) 
    (H1 : raised_from_two_families = 2 * 20) 
    (H2 : raised_from_eight_families = 8 * 10) 
    (H3 : raised_from_ten_families = 10 * 5) 
    (total_raised : ℕ := raised_from_two_families + raised_from_eight_families + raised_from_ten_families) : 
    (goal - total_raised) = 30 := 
by 
  sorry

end classroom_needs_more_money_l177_177651


namespace greatest_value_of_sum_l177_177304

variable (x y : ℝ)

-- Conditions
axiom sum_of_squares : x^2 + y^2 = 130
axiom product : x * y = 36

-- Statement to prove
theorem greatest_value_of_sum : x + y ≤ Real.sqrt 202 := sorry

end greatest_value_of_sum_l177_177304


namespace problem_conditions_l177_177905

noncomputable def f (x : ℝ) : ℝ := -x - x^3

variables (x₁ x₂ : ℝ)

theorem problem_conditions (h₁ : x₁ + x₂ ≤ 0) :
  (f x₁ * f (-x₁) ≤ 0) ∧
  (¬ (f x₂ * f (-x₂) > 0)) ∧
  (¬ (f x₁ + f x₂ ≤ f (-x₁) + f (-x₂))) ∧
  (f x₁ + f x₂ ≥ f (-x₁) + f (-x₂)) :=
sorry

end problem_conditions_l177_177905


namespace find_f2_l177_177642

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^5 - a * x^3 + b * x - 6

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -22 :=
by
  sorry

end find_f2_l177_177642


namespace tan_double_angle_l177_177761

theorem tan_double_angle (α : ℝ) (h : Real.tan (π - α) = 2) : Real.tan (2 * α) = 4 / 3 :=
by
  sorry

end tan_double_angle_l177_177761


namespace max_value_of_quadratic_l177_177282

theorem max_value_of_quadratic : 
  ∃ x : ℝ, (∃ M : ℝ, ∀ y : ℝ, (-3 * y^2 + 15 * y + 9 <= M)) ∧ M = 111 / 4 :=
by
  sorry

end max_value_of_quadratic_l177_177282


namespace Liam_cycling_speed_l177_177780

theorem Liam_cycling_speed :
  ∀ (Eugene_speed Claire_speed Liam_speed : ℝ),
    Eugene_speed = 6 →
    Claire_speed = (3/4) * Eugene_speed →
    Liam_speed = (4/3) * Claire_speed →
    Liam_speed = 6 :=
by
  intros
  sorry

end Liam_cycling_speed_l177_177780


namespace train_cross_time_l177_177543

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

end train_cross_time_l177_177543


namespace balls_in_boxes_l177_177007

/-- Prove that the number of ways to put 6 distinguishable balls in 3 distinguishable boxes is 729 (which is 3^6). -/
theorem balls_in_boxes : (3 ^ 6) = 729 :=
by 
  sorry

end balls_in_boxes_l177_177007


namespace worker_cellphone_surveys_l177_177594

theorem worker_cellphone_surveys 
  (regular_rate : ℕ) 
  (num_surveys : ℕ) 
  (higher_rate : ℕ)
  (total_earnings : ℕ) 
  (earned : ℕ → ℕ → ℕ)
  (higher_earned : ℕ → ℕ → ℕ) 
  (h1 : regular_rate = 10) 
  (h2 : num_surveys = 50) 
  (h3 : higher_rate = 13) 
  (h4 : total_earnings = 605) 
  (h5 : ∀ x, earned regular_rate (num_surveys - x) + higher_earned higher_rate x = total_earnings)
  : (∃ x, x = 35 ∧ earned regular_rate (num_surveys - x) + higher_earned higher_rate x = total_earnings) :=
sorry

end worker_cellphone_surveys_l177_177594


namespace expand_and_simplify_l177_177913

theorem expand_and_simplify :
  ∀ (x : ℝ), 5 * (6 * x^3 - 3 * x^2 + 4 * x - 2) = 30 * x^3 - 15 * x^2 + 20 * x - 10 :=
by
  intro x
  sorry

end expand_and_simplify_l177_177913


namespace shifted_parabola_sum_l177_177789

theorem shifted_parabola_sum :
  let f (x : ℝ) := 3 * x^2 - 2 * x + 5
  let g (x : ℝ) := 3 * (x - 3)^2 - 2 * (x - 3) + 5
  let a := 3
  let b := -20
  let c := 38
  a + b + c = 21 :=
by
  sorry

end shifted_parabola_sum_l177_177789


namespace gum_pieces_per_package_l177_177796

theorem gum_pieces_per_package (packages : ℕ) (extra : ℕ) (total : ℕ) (pieces_per_package : ℕ) :
    packages = 43 → extra = 8 → total = 997 → 43 * pieces_per_package + extra = total → pieces_per_package = 23 :=
by
  intros hpkg hextra htotal htotal_eq
  sorry

end gum_pieces_per_package_l177_177796


namespace circle_condition_l177_177261

theorem circle_condition (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 4 * m * x - 2 * y + 5 * m = 0) ↔ (m < 1 / 4 ∨ m > 1) :=
sorry

end circle_condition_l177_177261


namespace largest_number_not_sum_of_two_composites_l177_177640

-- Definitions related to composite numbers and natural numbers
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b)

-- Lean statement of the problem
theorem largest_number_not_sum_of_two_composites : ∀ n : ℕ,
  (∀ m : ℕ, m > n → cannot_be_sum_of_two_composites m → false) → n = 11 :=
by
  sorry

end largest_number_not_sum_of_two_composites_l177_177640


namespace math_problem_l177_177977

variable (x b : ℝ)
variable (h1 : x < b)
variable (h2 : b < 0)
variable (h3 : b = -2)

theorem math_problem : x^2 > b * x ∧ b * x > b^2 :=
by
  sorry

end math_problem_l177_177977


namespace sum_of_transformed_roots_equals_one_l177_177911

theorem sum_of_transformed_roots_equals_one 
  {α β γ : ℝ} 
  (hα : α^3 - α - 1 = 0) 
  (hβ : β^3 - β - 1 = 0) 
  (hγ : γ^3 - γ - 1 = 0) : 
  (1 - α) / (1 + α) + (1 - β) / (1 + β) + (1 - γ) / (1 + γ) = 1 :=
sorry

end sum_of_transformed_roots_equals_one_l177_177911


namespace max_three_digit_divisible_by_4_sequence_l177_177528

theorem max_three_digit_divisible_by_4_sequence (a : ℕ → ℕ) (n : ℕ) (h1 : ∀ k ≤ n - 2, a (k + 2) = 3 * a (k + 1) - 2 * a k - 2)
(h2 : ∀ k1 k2, k1 < k2 → a k1 < a k2) (ha2022 : ∃ k, a k = 2022) (hn : n ≥ 3) :
  ∃ m : ℕ, ∀ k, 100 ≤ a k ∧ a k ≤ 999 → a k % 4 = 0 → m ≤ 225 := by
  sorry

end max_three_digit_divisible_by_4_sequence_l177_177528


namespace sqrt_seven_l177_177521

theorem sqrt_seven (x : ℝ) : x^2 = 7 ↔ x = Real.sqrt 7 ∨ x = -Real.sqrt 7 := by
  sorry

end sqrt_seven_l177_177521


namespace quadratic_transformation_l177_177075

theorem quadratic_transformation :
  ∀ x : ℝ, (x^2 - 6 * x - 5 = 0) → ((x - 3)^2 = 14) :=
by
  intros x h
  sorry

end quadratic_transformation_l177_177075


namespace find_longer_diagonal_l177_177832

-- Define the necessary conditions
variables (d1 d2 : ℝ)
variable (A : ℝ)
axiom ratio_condition : d1 / d2 = 2 / 3
axiom area_condition : A = 12

-- Define the problem of finding the length of the longer diagonal
theorem find_longer_diagonal : ∃ (d : ℝ), d = d2 → d = 6 :=
by 
  sorry

end find_longer_diagonal_l177_177832


namespace fraction_to_decimal_l177_177060

theorem fraction_to_decimal : (58 : ℚ) / 160 = 0.3625 := 
by sorry

end fraction_to_decimal_l177_177060


namespace bailey_rawhide_bones_l177_177421

variable (dog_treats : ℕ) (chew_toys : ℕ) (total_items : ℕ)
variable (credit_cards : ℕ) (items_per_card : ℕ)

theorem bailey_rawhide_bones :
  (dog_treats = 8) →
  (chew_toys = 2) →
  (credit_cards = 4) →
  (items_per_card = 5) →
  (total_items = credit_cards * items_per_card) →
  (total_items - (dog_treats + chew_toys) = 10) :=
by
  intros
  sorry

end bailey_rawhide_bones_l177_177421


namespace solution_l177_177586

def p (x : ℝ) : Prop := x^2 + 2 * x - 3 < 0
def q (x : ℝ) : Prop := x ∈ Set.univ

theorem solution (x : ℝ) (hx : p x ∧ q x) : x = -2 ∨ x = -1 ∨ x = 0 := 
by
  sorry

end solution_l177_177586


namespace solve_equation_l177_177867

theorem solve_equation (x : ℝ) (h : x ≠ 1) :
  (3 * x) / (x - 1) = 2 + 1 / (x - 1) → x = -1 :=
by
  sorry

end solve_equation_l177_177867


namespace solve_fraction_zero_l177_177746

theorem solve_fraction_zero (x : ℕ) (h : x ≠ 0) (h_eq : (x - 1) / x = 0) : x = 1 := by 
  sorry

end solve_fraction_zero_l177_177746


namespace sum_of_functions_positive_l177_177437

open Real

noncomputable def f (x : ℝ) : ℝ := (exp x - exp (-x)) / 2

theorem sum_of_functions_positive (x1 x2 x3 : ℝ) (h1 : x1 + x2 > 0) (h2 : x2 + x3 > 0) (h3 : x3 + x1 > 0) : f x1 + f x2 + f x3 > 0 := by
  sorry

end sum_of_functions_positive_l177_177437


namespace base_7_minus_base_8_to_decimal_l177_177946

theorem base_7_minus_base_8_to_decimal : 
  (5 * 7^4 + 4 * 7^3 + 3 * 7^2 + 2 * 7^1 + 1 * 7^0) - (1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0) = 8190 :=
by sorry

end base_7_minus_base_8_to_decimal_l177_177946


namespace five_twos_make_24_l177_177877

theorem five_twos_make_24 :
  ∃ a b c d e : ℕ, a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 ∧ e = 2 ∧
  ((a + b + c) * (d + e) = 24) :=
by
  sorry

end five_twos_make_24_l177_177877


namespace common_solution_l177_177765

theorem common_solution (x : ℚ) : 
  (8 * x^2 + 7 * x - 1 = 0) ∧ (40 * x^2 + 89 * x - 9 = 0) → x = 1 / 8 :=
by { sorry }

end common_solution_l177_177765


namespace min_max_F_l177_177229

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

def F (x y : ℝ) : ℝ := x^2 + y^2

theorem min_max_F (x y : ℝ) (h1 : f (y^2 - 6 * y + 11) + f (x^2 - 8 * x + 10) ≤ 0) (h2 : y ≥ 3) :
  ∃ (min_val max_val : ℝ), min_val = 13 ∧ max_val = 49 ∧
    min_val ≤ F x y ∧ F x y ≤ max_val :=
sorry

end min_max_F_l177_177229


namespace simplest_common_denominator_l177_177353

theorem simplest_common_denominator (x y : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (d : ℤ), d = x^2 * y^2 ∧ ∀ (a b : ℤ), 
    (∃ (k : ℤ), a = k * (x^2 * y)) ∧ (∃ (m : ℤ), b = m * (x * y^2)) → d = lcm a b :=
by
  sorry

end simplest_common_denominator_l177_177353


namespace solve_system_part1_solve_system_part3_l177_177603

noncomputable def solution_part1 : Prop :=
  ∃ (x y : ℝ), (x + y = 2) ∧ (5 * x - 2 * (x + y) = 6) ∧ (x = 2) ∧ (y = 0)

-- Part (1) Statement
theorem solve_system_part1 : solution_part1 := sorry

noncomputable def solution_part3 : Prop :=
  ∃ (a b c : ℝ), (a + b = 3) ∧ (5 * a + 3 * c = 1) ∧ (a + b + c = 0) ∧ (a = 2) ∧ (b = 1) ∧ (c = -3)

-- Part (3) Statement
theorem solve_system_part3 : solution_part3 := sorry

end solve_system_part1_solve_system_part3_l177_177603


namespace students_count_rental_cost_l177_177790

theorem students_count (k m : ℕ) (n : ℕ) 
  (h1 : n = 35 * k)
  (h2 : n = 55 * (m - 1) + 45) : 
  n = 175 := 
by {
  sorry
}

theorem rental_cost (x y : ℕ) 
  (total_buses : x + y = 4)
  (cost_limit : 35 * x + 55 * y ≤ 1500) : 
  320 * x + 400 * y = 1440 := 
by {
  sorry 
}

end students_count_rental_cost_l177_177790


namespace monotonic_range_of_t_l177_177086

noncomputable def f (x : ℝ) := (x^2 - 3 * x + 3) * Real.exp x

def is_monotonic_on_interval (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨ (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y)

theorem monotonic_range_of_t (t : ℝ) (ht : t > -2) :
  is_monotonic_on_interval (-2) t f ↔ (-2 < t ∧ t ≤ 0) :=
sorry

end monotonic_range_of_t_l177_177086


namespace find_pairs_l177_177435

theorem find_pairs (x y : ℕ) (h : x > 0 ∧ y > 0) (d : ℕ) (gcd_cond : d = Nat.gcd x y)
  (eqn_cond : x * y * d = x + y + d ^ 2) : (x, y) = (2, 2) ∨ (x, y) = (2, 3) ∨ (x, y) = (3, 2) :=
by {
  sorry
}

end find_pairs_l177_177435


namespace Mike_owes_Laura_l177_177625

theorem Mike_owes_Laura :
  let rate_per_room := (13 : ℚ) / 3
  let rooms_cleaned := (8 : ℚ) / 5
  let total_amount := (104 : ℚ) / 15
  rate_per_room * rooms_cleaned = total_amount :=
by
  sorry

end Mike_owes_Laura_l177_177625


namespace integer_pairs_satisfying_equation_and_nonnegative_product_l177_177526

theorem integer_pairs_satisfying_equation_and_nonnegative_product :
  ∃ (pairs : List (ℤ × ℤ)), 
    (∀ p ∈ pairs, p.1 * p.2 ≥ 0 ∧ p.1^3 + p.2^3 + 99 * p.1 * p.2 = 33^3) ∧ 
    pairs.length = 35 :=
by sorry

end integer_pairs_satisfying_equation_and_nonnegative_product_l177_177526


namespace num_boys_l177_177771

variable (B G : ℕ)

def ratio_boys_girls (B G : ℕ) : Prop := B = 7 * G
def total_students (B G : ℕ) : Prop := B + G = 48

theorem num_boys (B G : ℕ) (h1 : ratio_boys_girls B G) (h2 : total_students B G) : 
  B = 42 :=
by
  sorry

end num_boys_l177_177771


namespace kids_left_playing_l177_177152

-- Define the conditions
def initial_kids : ℝ := 22.0
def kids_went_home : ℝ := 14.0

-- Theorem statement: Prove that the number of kids left playing is 8.0
theorem kids_left_playing : initial_kids - kids_went_home = 8.0 :=
by
  sorry -- Proof is left as an exercise

end kids_left_playing_l177_177152


namespace evaluate_fraction_l177_177262

theorem evaluate_fraction : 
  ( (20 - 19) + (18 - 17) + (16 - 15) + (14 - 13) + (12 - 11) + (10 - 9) + (8 - 7) + (6 - 5) + (4 - 3) + (2 - 1) ) 
  / 
  ( (1 - 2) + (3 - 4) + (5 - 6) + (7 - 8) + (9 - 10) + (11 - 12) + 13 ) 
  = (10 / 7) := 
by
  -- proof skipped
  sorry

end evaluate_fraction_l177_177262


namespace find_x_squared_plus_one_over_x_squared_l177_177858

theorem find_x_squared_plus_one_over_x_squared (x : ℝ) (h : x + 1/x = 4) : x^2 + 1/x^2 = 14 := by
  sorry

end find_x_squared_plus_one_over_x_squared_l177_177858


namespace exponential_function_decreasing_l177_177301

theorem exponential_function_decreasing {a : ℝ} 
  (h : ∀ x y : ℝ, x > y → (a-1)^x < (a-1)^y) : 1 < a ∧ a < 2 :=
by sorry

end exponential_function_decreasing_l177_177301


namespace tetrahedron_equal_reciprocal_squares_l177_177013

noncomputable def tet_condition_heights (h_1 h_2 h_3 h_4 : ℝ) : Prop :=
True

noncomputable def tet_condition_distances (d_1 d_2 d_3 : ℝ) : Prop :=
True

theorem tetrahedron_equal_reciprocal_squares
  (h_1 h_2 h_3 h_4 d_1 d_2 d_3 : ℝ)
  (hc_hts : tet_condition_heights h_1 h_2 h_3 h_4)
  (hc_dsts : tet_condition_distances d_1 d_2 d_3) :
  1 / (h_1 ^ 2) + 1 / (h_2 ^ 2) + 1 / (h_3 ^ 2) + 1 / (h_4 ^ 2) =
  1 / (d_1 ^ 2) + 1 / (d_2 ^ 2) + 1 / (d_3 ^ 2) :=
sorry

end tetrahedron_equal_reciprocal_squares_l177_177013


namespace cannot_form_right_triangle_l177_177558

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem cannot_form_right_triangle : ¬ is_right_triangle 40 50 60 := 
by
  sorry

end cannot_form_right_triangle_l177_177558


namespace partition_pos_integers_100_subsets_l177_177951

theorem partition_pos_integers_100_subsets :
  ∃ (P : (ℕ+ → Fin 100)), ∀ a b c : ℕ+, (a + 99 * b = c) → P a = P c ∨ P a = P b ∨ P b = P c :=
sorry

end partition_pos_integers_100_subsets_l177_177951


namespace range_of_a_l177_177307

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 4 → x^2 - 2 * x + 1 - a^2 < 0) ↔ (a < -3 ∨ a > 3) :=
sorry

end range_of_a_l177_177307


namespace sqrt_product_eq_sixty_sqrt_two_l177_177123

theorem sqrt_product_eq_sixty_sqrt_two : (Real.sqrt 50) * (Real.sqrt 18) * (Real.sqrt 8) = 60 * (Real.sqrt 2) := 
by 
  sorry

end sqrt_product_eq_sixty_sqrt_two_l177_177123


namespace negation_proposition_l177_177055

open Classical

theorem negation_proposition :
  ¬ (∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) ↔ ∀ x : ℝ, x^2 - 2*x + 1 > 0 :=
by
  sorry

end negation_proposition_l177_177055


namespace roots_diff_eq_4_l177_177606

theorem roots_diff_eq_4 {r s : ℝ} (h₁ : r ≠ s) (h₂ : r > s) (h₃ : r^2 - 10 * r + 21 = 0) (h₄ : s^2 - 10 * s + 21 = 0) : r - s = 4 := 
by
  sorry

end roots_diff_eq_4_l177_177606


namespace final_sale_price_is_correct_l177_177323

-- Define the required conditions
def original_price : ℝ := 1200.00
def first_discount_rate : ℝ := 0.10
def second_discount_rate : ℝ := 0.20
def final_discount_rate : ℝ := 0.05

-- Define the expression to calculate the sale price after the discounts
def first_discount_price := original_price * (1 - first_discount_rate)
def second_discount_price := first_discount_price * (1 - second_discount_rate)
def final_sale_price := second_discount_price * (1 - final_discount_rate)

-- Prove that the final sale price equals $820.80
theorem final_sale_price_is_correct : final_sale_price = 820.80 := by
  sorry

end final_sale_price_is_correct_l177_177323


namespace total_commission_l177_177554

-- Define the commission rate
def commission_rate : ℝ := 0.02

-- Define the sale prices of the three houses
def sale_price1 : ℝ := 157000
def sale_price2 : ℝ := 499000
def sale_price3 : ℝ := 125000

-- Total commission calculation
theorem total_commission :
  (commission_rate * sale_price1 + commission_rate * sale_price2 + commission_rate * sale_price3) = 15620 := 
by
  sorry

end total_commission_l177_177554


namespace solution_set_of_inequality_l177_177133

theorem solution_set_of_inequality (x : ℝ) : x * (x + 2) ≥ 0 ↔ x ≤ -2 ∨ x ≥ 0 := 
sorry

end solution_set_of_inequality_l177_177133


namespace roots_in_intervals_l177_177499

theorem roots_in_intervals {a b c : ℝ} (h₁ : a < b) (h₂ : b < c) :
  let f (x : ℝ) := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)
  -- statement that the roots are in the intervals (a, b) and (b, c)
  ∃ r₁ r₂, (a < r₁ ∧ r₁ < b) ∧ (b < r₂ ∧ r₂ < c) ∧ f r₁ = 0 ∧ f r₂ = 0 := 
sorry

end roots_in_intervals_l177_177499


namespace line_through_intersection_points_l177_177459

def first_circle (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def second_circle (x y : ℝ) : Prop := x^2 + y^2 = 5

theorem line_through_intersection_points (x y : ℝ) :
  (first_circle x y ∧ second_circle x y) → x - y - 3 = 0 :=
by
  sorry

end line_through_intersection_points_l177_177459


namespace min_frac_sum_l177_177833

open Real

noncomputable def minValue (m n : ℝ) : ℝ := 1 / m + 2 / n

theorem min_frac_sum (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) :
  minValue m n = 3 + 2 * sqrt 2 := by
  sorry

end min_frac_sum_l177_177833


namespace prove_county_growth_condition_l177_177569

variable (x : ℝ)
variable (investment2014 : ℝ) (investment2016 : ℝ)

def county_growth_condition
  (h1 : investment2014 = 2500)
  (h2 : investment2016 = 3500) : Prop :=
  investment2014 * (1 + x)^2 = investment2016

theorem prove_county_growth_condition
  (x : ℝ)
  (h1 : investment2014 = 2500)
  (h2 : investment2016 = 3500) : county_growth_condition x investment2014 investment2016 h1 h2 :=
by
  sorry

end prove_county_growth_condition_l177_177569


namespace actual_revenue_percentage_of_projected_l177_177318

theorem actual_revenue_percentage_of_projected (R : ℝ) (hR : R > 0) :
  (0.75 * R) / (1.2 * R) * 100 = 62.5 := 
by
  sorry

end actual_revenue_percentage_of_projected_l177_177318


namespace xyz_abs_eq_one_l177_177278

theorem xyz_abs_eq_one (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x)
  (cond : x + 1/y = y + 1/z ∧ y + 1/z = z + 1/x + 1) : |x * y * z| = 1 :=
sorry

end xyz_abs_eq_one_l177_177278


namespace minimum_discount_l177_177479

theorem minimum_discount (x : ℝ) (hx : x ≤ 10) : 
  let cost_price := 400 
  let selling_price := 500
  let discount_price := selling_price - (selling_price * (x / 100))
  let gross_profit := discount_price - cost_price 
  gross_profit ≥ cost_price * 0.125 :=
sorry

end minimum_discount_l177_177479


namespace relationship_between_n_and_m_l177_177862

variable {n m : ℕ}
variable {x y : ℝ}
variable {a : ℝ}
variable {z : ℝ}

def mean_sample_combined (n m : ℕ) (x y z a : ℝ) : Prop :=
  z = a * x + (1 - a) * y ∧ a > 1 / 2

theorem relationship_between_n_and_m 
  (hx : ∀ (i : ℕ), i < n → x = x)
  (hy : ∀ (j : ℕ), j < m → y = y)
  (hz : mean_sample_combined n m x y z a)
  (hne : x ≠ y) : n < m :=
sorry

end relationship_between_n_and_m_l177_177862


namespace candy_bar_reduction_l177_177542

variable (W P x : ℝ)
noncomputable def percent_reduction := (x / W) * 100

theorem candy_bar_reduction (h_weight_reduced : W > 0) 
                            (h_price_same : P > 0) 
                            (h_price_increase : P / (W - x) = (5 / 3) * (P / W)) :
    percent_reduction W x = 40 := 
sorry

end candy_bar_reduction_l177_177542


namespace x_eq_y_sufficient_not_necessary_abs_l177_177022

theorem x_eq_y_sufficient_not_necessary_abs (x y : ℝ) : (x = y → |x| = |y|) ∧ (|x| = |y| → x = y ∨ x = -y) :=
by {
  sorry
}

end x_eq_y_sufficient_not_necessary_abs_l177_177022


namespace xiao_xuan_wins_l177_177200

def cards_game (n : ℕ) (min_take : ℕ) (max_take : ℕ) (initial_turn : String) : String :=
  if initial_turn = "Xiao Liang" then "Xiao Xuan" else "Xiao Liang"

theorem xiao_xuan_wins :
  cards_game 17 1 2 "Xiao Liang" = "Xiao Xuan" :=
sorry

end xiao_xuan_wins_l177_177200


namespace arithmetic_sequence_sum_l177_177845

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h₁ : a 1 = -2012)
  (h₂ : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1)))
  (h₃ : (S 12) / 12 - (S 10) / 10 = 2) :
  S 2012 = -2012 := by
  sorry

end arithmetic_sequence_sum_l177_177845


namespace find_a_b_l177_177826

theorem find_a_b (a b : ℝ) (h : (⟨1, 2⟩ * (a : ℂ) + b = ⟨0, 2⟩)) : a = 1 ∧ b = -1 := 
by
  sorry

end find_a_b_l177_177826


namespace girls_attending_picnic_l177_177002

theorem girls_attending_picnic (g b : ℕ) (h1 : g + b = 1200) (h2 : (2 * g) / 3 + b / 2 = 730) : (2 * g) / 3 = 520 :=
by
  -- The proof steps would go here.
  sorry

end girls_attending_picnic_l177_177002


namespace repeated_digit_squares_l177_177039

theorem repeated_digit_squares :
  {n : ℕ | ∃ d : Fin 10, n = d ^ 2 ∧ (∀ m < n, m % 10 = d % 10)} ⊆ {0, 1, 4, 9} := by
  sorry

end repeated_digit_squares_l177_177039


namespace avg_visitors_other_days_l177_177868

-- Definitions for average visitors on Sundays and average visitors over the month
def avg_visitors_on_sundays : ℕ := 600
def avg_visitors_over_month : ℕ := 300
def days_in_month : ℕ := 30

-- Given conditions
def num_sundays_in_month : ℕ := 5
def total_days : ℕ := days_in_month
def total_visitors_over_month : ℕ := avg_visitors_over_month * days_in_month

-- Goal: Calculate the average number of visitors on other days (Monday to Saturday)
theorem avg_visitors_other_days :
  (avg_visitors_on_sundays * num_sundays_in_month + (total_days - num_sundays_in_month) * 240) = total_visitors_over_month :=
by
  -- Proof expected here, but skipped according to the instructions
  sorry

end avg_visitors_other_days_l177_177868


namespace domain_of_composite_l177_177163

theorem domain_of_composite (f : ℝ → ℝ) (x : ℝ) (hf : ∀ y, (0 ≤ y ∧ y ≤ 1) → f y = f y) :
  (0 ≤ x ∧ x ≤ 1) → (0 ≤ x ∧ x ≤ 1) → (0 ≤ x ∧ x ≤ 1) →
  0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ 2*x ∧ 2*x ≤ 1 ∧ 0 ≤ x + 1/3 ∧ x + 1/3 ≤ 1 →
  0 ≤ x ∧ x ≤ 1/2 :=
by
  intro h1 h2 h3 h4
  have h5: 0 ≤ 2*x ∧ 2*x ≤ 1 := sorry
  have h6: 0 ≤ x + 1/3 ∧ x + 1/3 ≤ 1 := sorry
  sorry

end domain_of_composite_l177_177163


namespace smallest_possible_b_l177_177976

theorem smallest_possible_b 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a - b = 8) 
  (h4 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : 
  b = 4 := 
sorry

end smallest_possible_b_l177_177976


namespace cube_volume_and_diagonal_from_surface_area_l177_177255

theorem cube_volume_and_diagonal_from_surface_area
    (A : ℝ) (h : A = 150) :
    ∃ (V : ℝ) (d : ℝ), V = 125 ∧ d = 5 * Real.sqrt 3 :=
by
  sorry

end cube_volume_and_diagonal_from_surface_area_l177_177255


namespace loss_percent_l177_177813

theorem loss_percent (cost_price selling_price loss_percent : ℝ) 
  (h_cost_price : cost_price = 600)
  (h_selling_price : selling_price = 550)
  (h_loss_percent : loss_percent = 8.33) : 
  (loss_percent = ((cost_price - selling_price) / cost_price) * 100) := 
by
  rw [h_cost_price, h_selling_price]
  sorry

end loss_percent_l177_177813


namespace remainder_a52_div_52_l177_177341

def a_n (n : ℕ) : ℕ := 
  (List.range (n + 1)).foldl (λ acc x => acc * 10 ^ (Nat.digits 10 x).length + x) 0

theorem remainder_a52_div_52 : (a_n 52) % 52 = 28 := 
  by
  sorry

end remainder_a52_div_52_l177_177341


namespace box_volume_correct_l177_177162

def volume_of_box (x : ℝ) : ℝ := (16 - 2 * x) * (12 - 2 * x) * x

theorem box_volume_correct {x : ℝ} (h1 : 1 ≤ x) (h2 : x ≤ 3) : 
  volume_of_box x = 4 * x^3 - 56 * x^2 + 192 * x := 
by 
  unfold volume_of_box 
  sorry

end box_volume_correct_l177_177162


namespace smallest_k_for_inequality_l177_177370

theorem smallest_k_for_inequality :
  ∃ k : ℕ, (∀ m : ℕ, m < k → 64^m ≤ 7) ∧ 64^k > 7 :=
by
  sorry

end smallest_k_for_inequality_l177_177370


namespace find_a_plus_b_l177_177757

theorem find_a_plus_b {f : ℝ → ℝ} (a b : ℝ) :
  (∀ x, f x = x^3 + 3*x^2 + 6*x + 14) →
  f a = 1 →
  f b = 19 →
  a + b = -2 :=
by
  sorry

end find_a_plus_b_l177_177757


namespace solution_set_f_x_sq_gt_2f_x_plus_1_l177_177677

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_f_x_sq_gt_2f_x_plus_1
  (h_domain : ∀ x, 0 < x → ∃ y, f y = f x)
  (h_func_equation : ∀ x y, 0 < x → 0 < y → f (x + y) = f x * f y)
  (h_greater_than_2 : ∀ x, 1 < x → f x > 2)
  (h_f2 : f 2 = 4) :
  ∀ x, x^2 > x + 2 → x > 2 :=
by
  intros x h
  sorry

end solution_set_f_x_sq_gt_2f_x_plus_1_l177_177677


namespace divisible_by_11_and_smallest_n_implies_77_l177_177510

theorem divisible_by_11_and_smallest_n_implies_77 (n : ℕ) (h₁ : n = 7) : ∃ m : ℕ, m = 11 * n := 
sorry

end divisible_by_11_and_smallest_n_implies_77_l177_177510


namespace students_after_last_stop_on_mondays_and_wednesdays_students_after_last_stop_on_tuesdays_and_thursdays_students_after_last_stop_on_fridays_l177_177103

structure BusRoute where
  first_stop : Nat
  second_stop_on : Nat
  second_stop_off : Nat
  third_stop_on : Nat
  third_stop_off : Nat
  fourth_stop_on : Nat
  fourth_stop_off : Nat

def mondays_and_wednesdays := BusRoute.mk 39 29 12 35 18 27 15
def tuesdays_and_thursdays := BusRoute.mk 39 33 10 5 0 8 4
def fridays := BusRoute.mk 39 25 10 40 20 10 5

def students_after_last_stop (route : BusRoute) : Nat :=
  let stop1 := route.first_stop
  let stop2 := stop1 + route.second_stop_on - route.second_stop_off
  let stop3 := stop2 + route.third_stop_on - route.third_stop_off
  stop3 + route.fourth_stop_on - route.fourth_stop_off

theorem students_after_last_stop_on_mondays_and_wednesdays :
  students_after_last_stop mondays_and_wednesdays = 85 := by
  sorry

theorem students_after_last_stop_on_tuesdays_and_thursdays :
  students_after_last_stop tuesdays_and_thursdays = 71 := by
  sorry

theorem students_after_last_stop_on_fridays :
  students_after_last_stop fridays = 79 := by
  sorry

end students_after_last_stop_on_mondays_and_wednesdays_students_after_last_stop_on_tuesdays_and_thursdays_students_after_last_stop_on_fridays_l177_177103


namespace mostSuitableForComprehensiveSurvey_l177_177334

-- Definitions of conditions
def optionA := "Understanding the sleep time of middle school students nationwide"
def optionB := "Understanding the water quality of a river"
def optionC := "Surveying the vision of all classmates"
def optionD := "Surveying the number of fish in a pond"

-- Define the notion of being the most suitable option for a comprehensive survey
def isSuitableForComprehensiveSurvey (option : String) := option = optionC

-- The theorem statement
theorem mostSuitableForComprehensiveSurvey : isSuitableForComprehensiveSurvey optionC := by
  -- This is the Lean 4 statement where we accept the hypotheses
  -- and conclude the theorem. Proof is omitted with "sorry".
  sorry

end mostSuitableForComprehensiveSurvey_l177_177334


namespace determinant_scaling_l177_177314

variable (p q r s : ℝ)

theorem determinant_scaling 
  (h : Matrix.det ![![p, q], ![r, s]] = 3) : 
  Matrix.det ![![2 * p, 2 * p + 5 * q], ![2 * r, 2 * r + 5 * s]] = 30 :=
by 
  sorry

end determinant_scaling_l177_177314


namespace michael_students_l177_177382

theorem michael_students (M N : ℕ) (h1 : M = 5 * N) (h2 : M + N + 300 = 3500) : M = 2667 := 
by 
  -- This to be filled later
  sorry

end michael_students_l177_177382


namespace coloringBooks_shelves_l177_177313

variables (initialStock soldBooks shelves : ℕ)

-- Given conditions
def initialBooks : initialStock = 87 := sorry
def booksSold : soldBooks = 33 := sorry
def numberOfShelves : shelves = 9 := sorry

-- Number of coloring books per shelf
def coloringBooksPerShelf (remainingBooksResult : ℕ) (booksPerShelfResult : ℕ) : Prop :=
  remainingBooksResult = initialStock - soldBooks ∧ booksPerShelfResult = remainingBooksResult / shelves

-- Prove the number of coloring books per shelf is 6
theorem coloringBooks_shelves (remainingBooksResult booksPerShelfResult : ℕ) : 
  coloringBooksPerShelf initialStock soldBooks shelves remainingBooksResult booksPerShelfResult →
  booksPerShelfResult = 6 :=
sorry

end coloringBooks_shelves_l177_177313


namespace blueBirdChessTeam72_l177_177755

def blueBirdChessTeamArrangements : Nat :=
  let boys_girls_ends := 3 * 3 + 3 * 3
  let alternate_arrangements := 2 * 2
  boys_girls_ends * alternate_arrangements

theorem blueBirdChessTeam72 : blueBirdChessTeamArrangements = 72 := by
  unfold blueBirdChessTeamArrangements
  sorry

end blueBirdChessTeam72_l177_177755


namespace plane_point_to_center_ratio_l177_177452

variable (a b c p q r : ℝ)

theorem plane_point_to_center_ratio :
  (a / p) + (b / q) + (c / r) = 2 ↔ 
  (∀ (α β γ : ℝ), α = 2 * p ∧ β = 2 * q ∧ γ = 2 * r ∧ (α, 0, 0) = (a, b, c) → 
  (a / (2 * p)) + (b / (2 * q)) + (c / (2 * r)) = 1) :=
by {
  sorry
}

end plane_point_to_center_ratio_l177_177452


namespace sequence_a_100_l177_177875

theorem sequence_a_100 : 
  (∃ a : ℕ → ℕ, a 1 = 2 ∧ (∀ n ≥ 1, a (n + 1) = a n + 2 * n) ∧ a 100 = 9902) :=
by
  sorry

end sequence_a_100_l177_177875


namespace gcd_18_30_is_6_gcd_18_30_is_even_l177_177141

def gcd_18_30 : ℕ := Nat.gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 := by
  sorry

theorem gcd_18_30_is_even : Even gcd_18_30 := by
  sorry

end gcd_18_30_is_6_gcd_18_30_is_even_l177_177141


namespace expected_value_of_10_sided_die_l177_177565

-- Definition of the conditions
def num_faces : ℕ := 10
def face_values : List ℕ := List.range' 2 num_faces

-- Theorem statement: The expected value of a roll of this die is 6.5
theorem expected_value_of_10_sided_die : 
  (List.sum face_values : ℚ) / num_faces = 6.5 := 
sorry

end expected_value_of_10_sided_die_l177_177565


namespace eccentricity_of_hyperbola_l177_177263

theorem eccentricity_of_hyperbola
  (a b c e : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c = a * e)
  (h4 : c^2 = a^2 + b^2)
  (h5 : ∀ B : ℝ × ℝ, B = (0, b))
  (h6 : ∀ F : ℝ × ℝ, F = (c, 0))
  (h7 : ∀ m_FB m_asymptote : ℝ, m_FB * m_asymptote = -1 → (m_FB = -b / c) ∧ (m_asymptote = b / a)) :
  e = (1 + Real.sqrt 5) / 2 :=
sorry

end eccentricity_of_hyperbola_l177_177263


namespace inequality_holds_l177_177126

theorem inequality_holds (a : ℝ) : (∀ x : ℝ, a * x ^ 2 - 2 * a * x + 1 > 0) ↔ (0 < a ∧ a < 1) :=
sorry

end inequality_holds_l177_177126


namespace flagpole_break_height_l177_177082

theorem flagpole_break_height (h h_break distance : ℝ) (h_pos : 0 < h) (h_break_pos : 0 < h_break)
  (h_flagpole : h = 8) (d_distance : distance = 3) (h_relationship : (h_break ^ 2 + distance^2) = (h - h_break)^2) :
  h_break = Real.sqrt 3 :=
  sorry

end flagpole_break_height_l177_177082


namespace find_order_amount_l177_177711

noncomputable def unit_price : ℝ := 100

def discount_rate (x : ℕ) : ℝ :=
  if x < 250 then 0
  else if x < 500 then 0.05
  else if x < 1000 then 0.10
  else 0.15

theorem find_order_amount (T : ℝ) (x : ℕ)
    (hx : x = 980) (hT : T = 88200) :
  T = unit_price * x * (1 - discount_rate x) :=
by
  rw [hx, hT]
  sorry

end find_order_amount_l177_177711


namespace divisors_remainder_5_l177_177880

theorem divisors_remainder_5 (d : ℕ) : d ∣ 2002 ∧ d > 5 ↔ d = 7 ∨ d = 11 ∨ d = 13 ∨ d = 14 ∨ 
                                      d = 22 ∨ d = 26 ∨ d = 77 ∨ d = 91 ∨ 
                                      d = 143 ∨ d = 154 ∨ d = 182 ∨ d = 286 ∨ 
                                      d = 1001 ∨ d = 2002 :=
by sorry

end divisors_remainder_5_l177_177880


namespace negation_of_implication_l177_177631

theorem negation_of_implication (x : ℝ) : x^2 + x - 6 < 0 → x ≤ 2 :=
by
  -- proof goes here
  sorry

end negation_of_implication_l177_177631


namespace smallest_x_l177_177175

theorem smallest_x (x : ℕ) (M : ℕ) (h : 1800 * x = M^3) :
  x = 30 :=
by
  sorry

end smallest_x_l177_177175


namespace simplify_and_evaluate_l177_177414

variable (a : ℝ)
variable (b : ℝ)

theorem simplify_and_evaluate (h : b = -1/3) : (a + b)^2 - a * (2 * b + a) = 1/9 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_l177_177414


namespace strictly_increasing_function_exists_l177_177096

noncomputable def exists_strictly_increasing_function (f : ℕ → ℕ) :=
  (∀ n : ℕ, n = 1 → f n = 2) ∧
  (∀ n : ℕ, f (f n) = f n + n) ∧
  (∀ m n : ℕ, m < n → f m < f n)

theorem strictly_increasing_function_exists : 
  ∃ f : ℕ → ℕ,
  exists_strictly_increasing_function f :=
sorry

end strictly_increasing_function_exists_l177_177096


namespace proof_problem_l177_177070

variables (α : ℝ)

-- Condition: tan(α) = 2
def tan_condition : Prop := Real.tan α = 2

-- First expression: (sin α + 2 cos α) / (4 cos α - sin α) = 2
def expression1 : Prop := (Real.sin α + 2 * Real.cos α) / (4 * Real.cos α - Real.sin α) = 2

-- Second expression: sqrt(2) * sin(2α + π/4) + 1 = 6/5
def expression2 : Prop := Real.sqrt 2 * Real.sin (2 * α + Real.pi / 4) + 1 = 6 / 5

-- Theorem: Prove the expressions given the condition
theorem proof_problem :
  tan_condition α → expression1 α ∧ expression2 α :=
by
  intro tan_cond
  have h1 : expression1 α := sorry
  have h2 : expression2 α := sorry
  exact ⟨h1, h2⟩

end proof_problem_l177_177070


namespace find_m_n_l177_177343

theorem find_m_n (x m n : ℝ) : (x + 4) * (x - 2) = x^2 + m * x + n → m = 2 ∧ n = -8 := 
by
  intro h
  -- Steps to prove the theorem would be here
  sorry

end find_m_n_l177_177343


namespace find_radius_of_base_of_cone_l177_177239

noncomputable def radius_of_cone (CSA : ℝ) (l : ℝ) : ℝ :=
  CSA / (Real.pi * l)

theorem find_radius_of_base_of_cone :
  radius_of_cone 527.7875658030853 14 = 12 :=
by
  sorry

end find_radius_of_base_of_cone_l177_177239


namespace intersection_in_quadrants_I_and_II_l177_177942

open Set

def in_quadrants_I_and_II (x y : ℝ) : Prop :=
  (0 ≤ x ∧ 0 ≤ y) ∨ (x ≤ 0 ∧ 0 ≤ y)

theorem intersection_in_quadrants_I_and_II :
  ∀ (x y : ℝ),
    y > 3 * x → y > -2 * x + 3 → in_quadrants_I_and_II x y :=
by
  intros x y h1 h2
  sorry

end intersection_in_quadrants_I_and_II_l177_177942


namespace problem_range_of_a_l177_177432

theorem problem_range_of_a (a : ℝ) :
  (∀ x : ℝ, |2 - x| + |3 + x| ≥ a^2 - 4 * a) ↔ -1 ≤ a ∧ a ≤ 5 :=
by
  sorry

end problem_range_of_a_l177_177432


namespace optimal_price_l177_177058

def monthly_sales (p : ℝ) : ℝ := 150 - 6 * p
def break_even (p : ℝ) : Prop := 40 ≤ monthly_sales p
def revenue (p : ℝ) : ℝ := p * monthly_sales p

theorem optimal_price : ∃ p : ℝ, p = 13 ∧ p ≤ 30 ∧ break_even p ∧ ∀ q : ℝ, q ≤ 30 → break_even q → revenue p ≥ revenue q := 
by
  sorry

end optimal_price_l177_177058


namespace solve_Q1_l177_177106

noncomputable def Q1 (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y + y * f x) = f x + f y + x * f y

theorem solve_Q1 :
  ∀ f : ℝ → ℝ, Q1 f → f = (id : ℝ → ℝ) :=
  by sorry

end solve_Q1_l177_177106


namespace intersection_M_N_l177_177178

open Set Real

def M := {x : ℝ | x^2 < 4}
def N := {x : ℝ | ∃ α : ℝ, x = sin α}
def IntersectSet := {x : ℝ | -1 ≤ x ∧ x ≤ 1}

theorem intersection_M_N : M ∩ N = IntersectSet := by
  sorry

end intersection_M_N_l177_177178


namespace largest_integer_same_cost_l177_177450

def sum_decimal_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def sum_ternary_digits (n : ℕ) : ℕ :=
  n.digits 3 |>.sum

theorem largest_integer_same_cost :
  ∃ n : ℕ, n < 1000 ∧ sum_decimal_digits n = sum_ternary_digits n ∧ ∀ m : ℕ, m < 1000 ∧ sum_decimal_digits m = sum_ternary_digits m → m ≤ n := 
  sorry

end largest_integer_same_cost_l177_177450


namespace f_at_three_bounds_l177_177556

theorem f_at_three_bounds (a c : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a * x^2 - c)
  (h2 : -4 ≤ f 1 ∧ f 1 ≤ -1) (h3 : -1 ≤ f 2 ∧ f 2 ≤ 5) : -1 ≤ f 3 ∧ f 3 ≤ 20 :=
sorry

end f_at_three_bounds_l177_177556


namespace xiao_li_hits_bullseye_14_times_l177_177408

theorem xiao_li_hits_bullseye_14_times
  (initial_rifle_bullets : ℕ := 10)
  (initial_pistol_bullets : ℕ := 14)
  (reward_per_bullseye_rifle : ℕ := 2)
  (reward_per_bullseye_pistol : ℕ := 4)
  (xiao_wang_bullseyes : ℕ := 30)
  (total_bullets : ℕ := initial_rifle_bullets + xiao_wang_bullseyes * reward_per_bullseye_rifle) :
  ∃ (xiao_li_bullseyes : ℕ), total_bullets = initial_pistol_bullets + xiao_li_bullseyes * reward_per_bullseye_pistol ∧ xiao_li_bullseyes = 14 :=
by sorry

end xiao_li_hits_bullseye_14_times_l177_177408


namespace total_ages_l177_177578

def Kate_age : ℕ := 19
def Maggie_age : ℕ := 17
def Sue_age : ℕ := 12

theorem total_ages : Kate_age + Maggie_age + Sue_age = 48 := sorry

end total_ages_l177_177578


namespace actual_revenue_percentage_l177_177901

def last_year_revenue (R : ℝ) := R
def projected_revenue (R : ℝ) := 1.25 * R
def actual_revenue (R : ℝ) := 0.75 * R

theorem actual_revenue_percentage (R : ℝ) : 
  (actual_revenue R / projected_revenue R) * 100 = 60 :=
by
  sorry

end actual_revenue_percentage_l177_177901


namespace time_difference_l177_177577

theorem time_difference (speed_Xanthia speed_Molly book_pages : ℕ) (minutes_in_hour : ℕ) :
  speed_Xanthia = 120 ∧ speed_Molly = 40 ∧ book_pages = 360 ∧ minutes_in_hour = 60 →
  (book_pages / speed_Molly - book_pages / speed_Xanthia) * minutes_in_hour = 360 := by
  sorry

end time_difference_l177_177577


namespace smallest_digit_not_in_odd_units_l177_177747

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l177_177747


namespace evaluation_expression_l177_177538

theorem evaluation_expression : 
  20 * (10 - 10.5 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5))) = 192.6 := 
by
  sorry

end evaluation_expression_l177_177538


namespace gcd_impossible_l177_177908

-- Define the natural numbers a, b, and c
variable (a b c : ℕ)

-- Define the factorial values
def fact_30 := Nat.factorial 30
def fact_40 := Nat.factorial 40
def fact_50 := Nat.factorial 50

-- Define the gcd values to be checked
def gcd_ab := fact_30 + 111
def gcd_bc := fact_40 + 234
def gcd_ca := fact_50 + 666

-- The main theorem to prove the impossibility
theorem gcd_impossible (h1 : Nat.gcd a b = gcd_ab) (h2 : Nat.gcd b c = gcd_bc) (h3 : Nat.gcd c a = gcd_ca) : False :=
by
  -- Proof omitted
  sorry

end gcd_impossible_l177_177908


namespace bicyclist_speed_first_100_km_l177_177237

theorem bicyclist_speed_first_100_km (v : ℝ) :
  (16 = 400 / ((100 / v) + 20)) →
  v = 20 :=
by
  sorry

end bicyclist_speed_first_100_km_l177_177237


namespace opposite_of_2021_l177_177107

theorem opposite_of_2021 : ∃ y : ℝ, 2021 + y = 0 ∧ y = -2021 :=
by
  sorry

end opposite_of_2021_l177_177107


namespace hyperbola_asymptotes_l177_177324

theorem hyperbola_asymptotes (x y : ℝ) (h : y^2 / 16 - x^2 / 9 = (1 : ℝ)) :
  ∃ (m : ℝ), (m = 4 / 3) ∨ (m = -4 / 3) :=
sorry

end hyperbola_asymptotes_l177_177324


namespace sum_squares_mod_13_is_zero_l177_177445

def sum_squares_mod_13 : ℕ :=
  (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2 + 10^2 + 11^2 + 12^2) % 13

theorem sum_squares_mod_13_is_zero : sum_squares_mod_13 = 0 := by
  sorry

end sum_squares_mod_13_is_zero_l177_177445


namespace step_of_induction_l177_177823

theorem step_of_induction (k : ℕ) (h : ∃ m : ℕ, 5^k - 2^k = 3 * m) :
  5^(k+1) - 2^(k+1) = 5 * (5^k - 2^k) + 3 * 2^k := 
by
  sorry

end step_of_induction_l177_177823


namespace evaTotalMarksCorrect_l177_177466

-- Definition of marks scored by Eva in each subject across semesters
def evaMathsMarksSecondSemester : Nat := 80
def evaArtsMarksSecondSemester : Nat := 90
def evaScienceMarksSecondSemester : Nat := 90

def evaMathsMarksFirstSemester : Nat := evaMathsMarksSecondSemester + 10
def evaArtsMarksFirstSemester : Nat := evaArtsMarksSecondSemester - 15
def evaScienceMarksFirstSemester : Nat := evaScienceMarksSecondSemester - (evaScienceMarksSecondSemester / 3)

-- Total marks in each semester
def totalMarksFirstSemester : Nat := evaMathsMarksFirstSemester + evaArtsMarksFirstSemester + evaScienceMarksFirstSemester
def totalMarksSecondSemester : Nat := evaMathsMarksSecondSemester + evaArtsMarksSecondSemester + evaScienceMarksSecondSemester

-- Combined total
def evaTotalMarks : Nat := totalMarksFirstSemester + totalMarksSecondSemester

-- Statement to prove
theorem evaTotalMarksCorrect : evaTotalMarks = 485 := 
by
  -- This needs to be proved as per the conditions and calculations above
  sorry

end evaTotalMarksCorrect_l177_177466


namespace car_second_hour_speed_l177_177719

theorem car_second_hour_speed (x : ℝ) 
  (first_hour_speed : ℝ := 20)
  (average_speed : ℝ := 40) 
  (total_time : ℝ := 2)
  (total_distance : ℝ := first_hour_speed + x) 
  : total_distance / total_time = average_speed → x = 60 :=
by
  intro h
  sorry

end car_second_hour_speed_l177_177719


namespace collinear_points_sum_xy_solution_l177_177926

theorem collinear_points_sum_xy_solution (x y : ℚ)
  (h1 : (B : ℚ × ℚ) = (-2, y))
  (h2 : (A : ℚ × ℚ) = (x, 5))
  (h3 : (C : ℚ × ℚ) = (1, 1))
  (h4 : dist (B.1, B.2) (C.1, C.2) = 2 * dist (A.1, A.2) (C.1, C.2))
  (h5 : (y - 5) / (-2 - x) = (1 - 5) / (1 - x)) :
  x + y = -9 / 2 ∨ x + y = 17 / 2 :=
by sorry

end collinear_points_sum_xy_solution_l177_177926


namespace polynomial_coefficients_sum_even_odd_coefficients_difference_square_l177_177881

theorem polynomial_coefficients_sum (a : Fin 8 → ℝ):
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) = 3^7 - 1 :=
by
  -- assume the polynomial (1 + 2x)^7 has coefficients a 0, a 1, ..., a 7
  -- such that (1 + 2x)^7 = a 0 + a 1 * x + a 2 * x^2 + ... + a 7 * x^7
  sorry

theorem even_odd_coefficients_difference_square (a : Fin 8 → ℝ):
  (a 0 + a 2 + a 4 + a 6)^2 - (a 1 + a 3 + a 5 + a 7)^2 = -3^7 :=
by
  -- assume the polynomial (1 + 2x)^7 has coefficients a 0, a 1, ..., a 7
  -- such that (1 + 2x)^7 = a 0 + a 1 * x + a 2 * x^2 + ... + a 7 * x^7
  sorry

end polynomial_coefficients_sum_even_odd_coefficients_difference_square_l177_177881


namespace measure_of_angle_Q_l177_177425

-- Given conditions
variables (α β γ δ : ℝ)
axiom h1 : α = 130
axiom h2 : β = 95
axiom h3 : γ = 110
axiom h4 : δ = 104

-- Statement of the problem
theorem measure_of_angle_Q (Q : ℝ) (h5 : Q + α + β + γ + δ = 540) : Q = 101 := 
sorry

end measure_of_angle_Q_l177_177425


namespace dave_guitar_strings_l177_177894

noncomputable def strings_per_night : ℕ := 2
noncomputable def shows_per_week : ℕ := 6
noncomputable def weeks : ℕ := 12

theorem dave_guitar_strings : 
  (strings_per_night * shows_per_week * weeks) = 144 := 
by
  sorry

end dave_guitar_strings_l177_177894


namespace final_song_count_l177_177121

theorem final_song_count {init_songs added_songs removed_songs doubled_songs final_songs : ℕ} 
    (h1 : init_songs = 500)
    (h2 : added_songs = 500)
    (h3 : doubled_songs = (init_songs + added_songs) * 2)
    (h4 : removed_songs = 50)
    (h_final : final_songs = doubled_songs - removed_songs) : 
    final_songs = 2950 :=
by
  sorry

end final_song_count_l177_177121


namespace actual_plot_area_l177_177367

noncomputable def area_of_triangle_in_acres : Real :=
  let base_cm : Real := 8
  let height_cm : Real := 5
  let area_cm2 : Real := 0.5 * base_cm * height_cm
  let conversion_factor_cm2_to_km2 : Real := 25
  let area_km2 : Real := area_cm2 * conversion_factor_cm2_to_km2
  let conversion_factor_km2_to_acres : Real := 247.1
  area_km2 * conversion_factor_km2_to_acres

theorem actual_plot_area :
  area_of_triangle_in_acres = 123550 :=
by
  sorry

end actual_plot_area_l177_177367


namespace determine_a_l177_177671

open Real

theorem determine_a :
  (∃ a : ℝ, |x^2 + a*x + 4*a| ≤ 3 → x^2 + a*x + 4*a = 3) ↔ (a = 8 + 2*sqrt 13 ∨ a = 8 - 2*sqrt 13) :=
by
  sorry

end determine_a_l177_177671


namespace abs_diff_eq_seven_l177_177234

theorem abs_diff_eq_seven (m n : ℤ) (h1 : |m| = 5) (h2 : |n| = 2) (h3 : m * n < 0) : |m - n| = 7 := 
sorry

end abs_diff_eq_seven_l177_177234


namespace calculate_result_l177_177759

theorem calculate_result (x : ℝ) : (-x^3)^3 = -x^9 :=
by {
  sorry  -- Proof not required per instructions
}

end calculate_result_l177_177759


namespace distinct_lines_isosceles_not_equilateral_l177_177326

-- Define a structure for an isosceles triangle that is not equilateral
structure IsoscelesButNotEquilateralTriangle :=
  (a b c : ℕ)    -- sides of the triangle
  (h₁ : a = b)   -- two equal sides
  (h₂ : a ≠ c)   -- not equilateral (not all three sides are equal)

-- Define that the number of distinct lines representing altitudes, medians, and interior angle bisectors is 5
theorem distinct_lines_isosceles_not_equilateral (T : IsoscelesButNotEquilateralTriangle) : 
  ∃ (n : ℕ), n = 5 :=
by
  sorry

end distinct_lines_isosceles_not_equilateral_l177_177326


namespace quadratic_y1_gt_y2_l177_177637

theorem quadratic_y1_gt_y2 {a b c y1 y2 : ℝ} (ha : a > 0) (hy1 : y1 = a * (-1)^2 + b * (-1) + c) (hy2 : y2 = a * 2^2 + b * 2 + c) : y1 > y2 :=
  sorry

end quadratic_y1_gt_y2_l177_177637


namespace number_of_boys_and_girls_l177_177436

theorem number_of_boys_and_girls (b g : ℕ) 
    (h1 : ∀ n : ℕ, (n ≥ 1) → ∃ (a_n : ℕ), a_n = 2 * n + 1)
    (h2 : (2 * b + 1 = g))
    : b = (g - 1) / 2 :=
by
  sorry

end number_of_boys_and_girls_l177_177436


namespace problem_sol_l177_177179

theorem problem_sol (a b : ℝ) (h : ∀ x, (x > -1 ∧ x < 1/3) ↔ (ax^2 + bx + 1 > 0)) : a * b = 6 :=
sorry

end problem_sol_l177_177179


namespace solveForX_l177_177009

theorem solveForX : ∃ (x : ℚ), x + 5/8 = 7/24 + 1/4 ∧ x = -1/12 := by
  sorry

end solveForX_l177_177009


namespace ship_navigation_avoid_reefs_l177_177679

theorem ship_navigation_avoid_reefs (a : ℝ) (h : a > 0) :
  (10 * a) * 40 / Real.sqrt ((10 * a) ^ 2 + 40 ^ 2) > 20 ↔
  a > (4 * Real.sqrt 3 / 3) :=
by
  sorry

end ship_navigation_avoid_reefs_l177_177679


namespace supplement_of_supplement_of_58_l177_177325

theorem supplement_of_supplement_of_58 (α : ℝ) (h : α = 58) : 180 - (180 - α) = 58 :=
by
  sorry

end supplement_of_supplement_of_58_l177_177325


namespace max_value_frac_l177_177885
noncomputable section

open Real

variables (a b x y : ℝ)

theorem max_value_frac :
  a > 1 → b > 1 → 
  a^x = 2 → b^y = 2 →
  a + sqrt b = 4 →
  (2/x + 1/y) ≤ 4 :=
by
  intros ha hb hax hby hab
  sorry

end max_value_frac_l177_177885


namespace monday_to_sunday_ratio_l177_177529

-- Define the number of pints Alice bought on Sunday
def sunday_pints : ℕ := 4

-- Define the number of pints Alice bought on Monday as a multiple of Sunday
def monday_pints (k : ℕ) : ℕ := 4 * k

-- Define the number of pints Alice bought on Tuesday
def tuesday_pints (k : ℕ) : ℚ := (4 * k) / 3

-- Define the number of pints Alice returned on Wednesday
def wednesday_return (k : ℕ) : ℚ := (2 * k) / 3

-- Define the total number of pints Alice had on Wednesday before returning the expired ones
def total_pre_return (k : ℕ) : ℚ := 18 + (2 * k) / 3

-- Define the total number of pints purchased from Sunday to Tuesday
def total_pints (k : ℕ) : ℚ := 4 + 4 * k + (4 * k) / 3

-- The statement to be proven
theorem monday_to_sunday_ratio : ∃ k : ℕ, 
  (4 * k + (4 * k) / 3 + 4 = 18 + (2 * k) / 3) ∧
  (4 * k) / 4 = 3 :=
by 
  sorry

end monday_to_sunday_ratio_l177_177529


namespace value_of_C_l177_177157

theorem value_of_C (k : ℝ) (C : ℝ) (h : k = 0.4444444444444444) :
  (2 * k * 0 ^ 2 + 6 * k * 0 + C = 0) ↔ C = 2 :=
by {
  sorry
}

end value_of_C_l177_177157


namespace max_gcd_of_consecutive_terms_seq_b_l177_177702

-- Define the sequence b_n
def sequence_b (n : ℕ) : ℕ := n.factorial + 3 * n

-- Define the gcd function for two terms in the sequence
def gcd_two_terms (n : ℕ) : ℕ := Nat.gcd (sequence_b n) (sequence_b (n + 1))

-- Define the condition of n being greater than or equal to 0
def n_ge_zero (n : ℕ) : Prop := n ≥ 0

-- The theorem statement
theorem max_gcd_of_consecutive_terms_seq_b : ∃ n : ℕ, n_ge_zero n ∧ gcd_two_terms n = 14 := 
sorry

end max_gcd_of_consecutive_terms_seq_b_l177_177702


namespace inequality_arith_geo_mean_l177_177176

variable (a k : ℝ)
variable (h1 : 1 ≤ k)
variable (h2 : k ≤ 3)
variable (h3 : 0 < k)

theorem inequality_arith_geo_mean (h1 : 1 ≤ k) (h2 : k ≤ 3) (h3 : 0 < k):
    ( (a + k * a) / 2 ) ^ 2 ≥ ( (a * (k * a)) ^ (1/2) ) ^ 2 :=
by
  sorry

end inequality_arith_geo_mean_l177_177176


namespace tim_took_rulers_l177_177675

theorem tim_took_rulers (initial_rulers : ℕ) (remaining_rulers : ℕ) (rulers_taken : ℕ) :
  initial_rulers = 46 → remaining_rulers = 21 → rulers_taken = initial_rulers - remaining_rulers → rulers_taken = 25 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end tim_took_rulers_l177_177675


namespace apples_equation_l177_177077

variable {A J H : ℕ}

theorem apples_equation:
    A + J = 12 →
    H = A + J + 9 →
    A = J + 8 →
    H = 21 :=
by
  intros h1 h2 h3
  sorry

end apples_equation_l177_177077


namespace pounds_of_fish_to_ship_l177_177656

theorem pounds_of_fish_to_ship (crates_weight : ℕ) (cost_per_crate : ℝ) (total_cost : ℝ) :
  crates_weight = 30 → cost_per_crate = 1.5 → total_cost = 27 → 
  (total_cost / cost_per_crate) * crates_weight = 540 :=
by
  intros h1 h2 h3
  sorry

end pounds_of_fish_to_ship_l177_177656


namespace A_fraction_simplification_l177_177766

noncomputable def A : ℚ := 
  ((3/8) * (13/5)) / ((5/2) * (6/5)) +
  ((5/8) * (8/5)) / (3 * (6/5) * (25/6)) +
  (20/3) * (3/25) +
  28 +
  (1 / 9) / 7 +
  (1/5) / (9 * 22)

theorem A_fraction_simplification :
  let num := 1901
  let denom := 3360
  (A = num / denom) :=
sorry

end A_fraction_simplification_l177_177766


namespace sum_reciprocals_roots_l177_177177

theorem sum_reciprocals_roots :
  (∃ p q : ℝ, p + q = 10 ∧ p * q = 3) →
  (∃ p q : ℝ, p ≠ 0 ∧ q ≠ 0 → (1 / p) + (1 / q) = 10 / 3) :=
by
  sorry

end sum_reciprocals_roots_l177_177177


namespace decreasing_interval_0_pi_over_4_l177_177364

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.cos (x + φ)

theorem decreasing_interval_0_pi_over_4 (φ : ℝ) (hφ1 : 0 < |φ| ∧ |φ| < Real.pi / 2)
  (hodd : ∀ x : ℝ, f (x + Real.pi / 4) φ = -f (-x + Real.pi / 4) φ) :
  ∀ x : ℝ, 0 < x ∧ x < Real.pi / 4 → f x φ > f (x + 1e-6) φ :=
by sorry

end decreasing_interval_0_pi_over_4_l177_177364


namespace point_P_coordinates_l177_177966

-- Definitions based on conditions
def in_fourth_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 < 0

def distance_to_x_axis (P : ℝ × ℝ) (d : ℝ) : Prop :=
  abs P.2 = d

def distance_to_y_axis (P : ℝ × ℝ) (d : ℝ) : Prop :=
  abs P.1 = d

-- The theorem statement based on the proof problem
theorem point_P_coordinates :
  ∃ P : ℝ × ℝ, 
    in_fourth_quadrant P ∧ 
    distance_to_x_axis P 2 ∧ 
    distance_to_y_axis P 3 ∧ 
    P = (3, -2) :=
by
  sorry

end point_P_coordinates_l177_177966


namespace cannot_afford_laptop_l177_177731

theorem cannot_afford_laptop (P_0 : ℝ) : 56358 < P_0 * (1.06)^2 :=
by
  sorry

end cannot_afford_laptop_l177_177731


namespace correct_factorization_l177_177038

theorem correct_factorization (a : ℝ) : 
  (a ^ 2 + 4 * a ≠ a ^ 2 * (a + 4)) ∧ 
  (a ^ 2 - 9 ≠ (a + 9) * (a - 9)) ∧ 
  (a ^ 2 + 4 * a + 2 ≠ (a + 2) ^ 2) → 
  (a ^ 2 - 2 * a + 1 = (a - 1) ^ 2) :=
by sorry

end correct_factorization_l177_177038


namespace integer_values_satisfying_square_root_condition_l177_177692

theorem integer_values_satisfying_square_root_condition :
  ∃ (s : Finset ℤ), s.card = 6 ∧ ∀ x ∈ s, 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 6 := sorry

end integer_values_satisfying_square_root_condition_l177_177692


namespace robert_ate_7_chocolates_l177_177734

-- Define the number of chocolates Nickel ate
def nickel_chocolates : ℕ := 5

-- Define the number of chocolates Robert ate
def robert_chocolates : ℕ := nickel_chocolates + 2

-- Prove that Robert ate 7 chocolates
theorem robert_ate_7_chocolates : robert_chocolates = 7 := by
    sorry

end robert_ate_7_chocolates_l177_177734


namespace distance_from_tee_to_hole_l177_177561

-- Define the constants based on the problem conditions
def s1 : ℕ := 180
def s2 : ℕ := (1 / 2 * s1 + 20 - 20)

-- Define the total distance calculation
def total_distance := s1 + s2

-- State the ultimate theorem that needs to be proved
theorem distance_from_tee_to_hole : total_distance = 270 := by
  sorry

end distance_from_tee_to_hole_l177_177561


namespace area_of_triangle_l177_177648

theorem area_of_triangle {A B C : ℝ} {a b c : ℝ}
  (h1 : b = 2) (h2 : c = 2 * Real.sqrt 2) (h3 : C = Real.pi / 4) :
  1 / 2 * b * c * Real.sin (Real.pi - C - (1 / 2 * Real.pi / 3)) = Real.sqrt 3 + 1 :=
by
  sorry

end area_of_triangle_l177_177648


namespace longer_side_length_l177_177495

-- Define the relevant entities: radius, area of the circle, and rectangle conditions.
noncomputable def radius : ℝ := 6
noncomputable def area_circle : ℝ := Real.pi * radius^2
noncomputable def area_rectangle : ℝ := 3 * area_circle
noncomputable def shorter_side : ℝ := 2 * radius

-- Prove that the length of the longer side of the rectangle is 9π cm.
theorem longer_side_length : ∃ (l : ℝ), (area_rectangle = l * shorter_side) → (l = 9 * Real.pi) :=
by
  sorry

end longer_side_length_l177_177495


namespace probability_neither_event_l177_177054

theorem probability_neither_event (P_A P_B P_A_and_B : ℝ)
  (h1 : P_A = 0.25)
  (h2 : P_B = 0.40)
  (h3 : P_A_and_B = 0.20) :
  1 - (P_A + P_B - P_A_and_B) = 0.55 :=
by
  sorry

end probability_neither_event_l177_177054


namespace geometric_progression_solution_l177_177481

theorem geometric_progression_solution (x : ℝ) :
  (2 * x + 10) ^ 2 = x * (5 * x + 10) → x = 15 + 5 * Real.sqrt 5 :=
by
  intro h
  sorry

end geometric_progression_solution_l177_177481


namespace measure_of_angle_x_l177_177429

theorem measure_of_angle_x :
  ∀ (angle_ABC angle_BDE angle_DBE angle_ABD x : ℝ),
    angle_ABC = 132 ∧
    angle_BDE = 31 ∧
    angle_DBE = 30 ∧
    angle_ABD = 180 - 132 →
    x = 180 - (angle_BDE + angle_DBE) →
    x = 119 :=
by
  intros angle_ABC angle_BDE angle_DBE angle_ABD x h h2
  sorry

end measure_of_angle_x_l177_177429


namespace find_k_l177_177345

noncomputable def f (x : ℝ) : ℝ := 7 * x^3 - 1 / x + 5
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := x^3 - k

theorem find_k (k : ℝ) : 
  f 3 - g 3 k = 5 → k = - 485 / 3 :=
by
  sorry

end find_k_l177_177345


namespace calculate_expression_l177_177611

theorem calculate_expression :
  427 / 2.68 * 16 * 26.8 / 42.7 * 16 = 25600 :=
sorry

end calculate_expression_l177_177611


namespace twice_not_square_l177_177400

theorem twice_not_square (m : ℝ) : 2 * m ≠ m * m := by
  sorry

end twice_not_square_l177_177400


namespace breadth_of_boat_l177_177944

theorem breadth_of_boat :
  ∀ (L : ℝ) (h : ℝ) (m : ℝ) (g : ℝ) (rho : ℝ),
    L = 8 → h = 0.01 → m = 160 → g = 9.81 → rho = 1000 →
    (L * 2 * h = (m * g) / (rho * g)) :=
by
  intros L h m g rho hL hh hm hg hrho
  sorry

end breadth_of_boat_l177_177944


namespace pet_store_cages_l177_177853

theorem pet_store_cages 
  (initial_puppies : ℕ) 
  (sold_puppies : ℕ) 
  (puppies_per_cage : ℕ) 
  (h_initial_puppies : initial_puppies = 45) 
  (h_sold_puppies : sold_puppies = 11) 
  (h_puppies_per_cage : puppies_per_cage = 7) 
  : (initial_puppies - sold_puppies + puppies_per_cage - 1) / puppies_per_cage = 5 :=
by sorry

end pet_store_cages_l177_177853


namespace find_g5_l177_177545

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Define the conditions
axiom cond1 : g 1 > 1
axiom cond2 : ∀ (x y : ℤ), g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom cond3 : ∀ (x : ℤ), 3 * g x = g (x + 1) + 2 * x - 1

-- The statement we need to prove
theorem find_g5 : g 5 = 248 := by
  sorry

end find_g5_l177_177545


namespace no_real_roots_other_than_zero_l177_177296

theorem no_real_roots_other_than_zero (k : ℝ) (h : k ≠ 0):
  ¬(∃ x : ℝ, x^2 + 2 * k * x + 3 * k^2 = 0) :=
by
  sorry

end no_real_roots_other_than_zero_l177_177296


namespace payment_to_C_l177_177883

-- Work rates definition
def work_rate_A : ℚ := 1 / 6
def work_rate_B : ℚ := 1 / 8
def combined_work_rate_A_B : ℚ := work_rate_A + work_rate_B
def combined_work_rate_A_B_C : ℚ := 1 / 3

-- C's work rate calculation
def work_rate_C : ℚ := combined_work_rate_A_B_C - combined_work_rate_A_B

-- Payment calculation
def total_payment : ℚ := 3200
def C_payment_ratio : ℚ := work_rate_C / combined_work_rate_A_B_C
def C_payment : ℚ := total_payment * C_payment_ratio

-- Theorem stating the result
theorem payment_to_C : C_payment = 400 := by
  sorry

end payment_to_C_l177_177883


namespace tan_3theta_l177_177773

theorem tan_3theta (θ : ℝ) (h : Real.tan θ = 3 / 4) : Real.tan (3 * θ) = -12.5 :=
sorry

end tan_3theta_l177_177773


namespace find_multiplier_l177_177358

variable {a b : ℝ} 

theorem find_multiplier (h1 : 3 * a = x * b) (h2 : a ≠ 0 ∧ b ≠ 0) (h3 : a / 4 = b / 3) : x = 4 :=
by
  sorry

end find_multiplier_l177_177358


namespace probability_different_color_and_label_sum_more_than_3_l177_177487

-- Definitions for the conditions:
structure Coin :=
  (color : Bool) -- True for Yellow, False for Green
  (label : Nat)

def coins : List Coin := [
  Coin.mk true 1,
  Coin.mk true 2,
  Coin.mk false 1,
  Coin.mk false 2,
  Coin.mk false 3
]

def outcomes : List (Coin × Coin) :=
  [(coins[0], coins[1]), (coins[0], coins[2]), (coins[0], coins[3]), (coins[0], coins[4]),
   (coins[1], coins[2]), (coins[1], coins[3]), (coins[1], coins[4]),
   (coins[2], coins[3]), (coins[2], coins[4]), (coins[3], coins[4])]

def different_color_and_label_sum_more_than_3 (c1 c2 : Coin) : Bool :=
  c1.color ≠ c2.color ∧ (c1.label + c2.label > 3)

def valid_outcomes : List (Coin × Coin) :=
  outcomes.filter (λ p => different_color_and_label_sum_more_than_3 p.fst p.snd)

-- Proof statement:
theorem probability_different_color_and_label_sum_more_than_3 :
  (valid_outcomes.length : ℚ) / (outcomes.length : ℚ) = 3 / 10 :=
by
  sorry

end probability_different_color_and_label_sum_more_than_3_l177_177487


namespace correct_option_l177_177655

-- Defining the conditions for each option
def optionA (m n : ℝ) : Prop := (m / n)^7 = m^7 * n^(1/7)
def optionB : Prop := (4)^(4/12) = (-3)^(1/3)
def optionC (x y : ℝ) : Prop := ((x^3 + y^3)^(1/4)) = (x + y)^(3/4)
def optionD : Prop := (9)^(1/6) = 3^(1/3)

-- Asserting that option D is correct
theorem correct_option : optionD :=
by
  sorry

end correct_option_l177_177655


namespace sum_of_squares_ways_l177_177439

theorem sum_of_squares_ways : 
  ∃ ways : ℕ, ways = 2 ∧
    (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = 100) ∧ 
    (∃ (x y z w : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ x^2 + y^2 + z^2 + w^2 = 100) := 
sorry

end sum_of_squares_ways_l177_177439


namespace train_length_1080_l177_177310

def length_of_train (speed time : ℕ) : ℕ := speed * time

theorem train_length_1080 (speed time : ℕ) (h1 : speed = 108) (h2 : time = 10) : length_of_train speed time = 1080 := by
  sorry

end train_length_1080_l177_177310


namespace intersection_M_N_union_complements_M_N_l177_177788

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x ≥ 1}
def N : Set ℝ := {x | 0 ≤ x ∧ x < 5}

theorem intersection_M_N :
  M ∩ N = {x | 1 ≤ x ∧ x < 5} :=
by {
  sorry
}

theorem union_complements_M_N :
  (compl M) ∪ (compl N) = {x | x < 1 ∨ x ≥ 5} :=
by {
  sorry
}

end intersection_M_N_union_complements_M_N_l177_177788


namespace average_age_of_coaches_l177_177688

variables 
  (total_members : ℕ) (avg_age_total : ℕ) 
  (num_girls : ℕ) (num_boys : ℕ) (num_coaches : ℕ) 
  (avg_age_girls : ℕ) (avg_age_boys : ℕ)

theorem average_age_of_coaches 
  (h1 : total_members = 50) 
  (h2 : avg_age_total = 18)
  (h3 : num_girls = 25) 
  (h4 : num_boys = 20) 
  (h5 : num_coaches = 5)
  (h6 : avg_age_girls = 16)
  (h7 : avg_age_boys = 17) : 
  (900 - (num_girls * avg_age_girls + num_boys * avg_age_boys)) / num_coaches = 32 :=
by
  sorry

end average_age_of_coaches_l177_177688


namespace circumference_base_of_cone_l177_177738

-- Define the given conditions
def radius_circle : ℝ := 6
def angle_sector : ℝ := 300

-- Define the problem to prove the circumference of the base of the resulting cone in terms of π
theorem circumference_base_of_cone :
  (angle_sector / 360) * (2 * π * radius_circle) = 10 * π := by
sorry

end circumference_base_of_cone_l177_177738


namespace cond_prob_B_given_A_l177_177289

-- Definitions based on the conditions
def eventA := {n : ℕ | n > 4 ∧ n ≤ 6}
def eventB := {k : ℕ × ℕ | (k.1 + k.2) = 7}

-- Probability of event A
def probA := (2 : ℚ) / 6

-- Joint probability of events A and B
def probAB := (1 : ℚ) / (6 * 6)

-- Conditional probability P(B|A)
def cond_prob := probAB / probA

-- The final statement to prove
theorem cond_prob_B_given_A : cond_prob = 1 / 6 := by
  sorry

end cond_prob_B_given_A_l177_177289


namespace sin_240_deg_l177_177097

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 :=
by
  sorry

end sin_240_deg_l177_177097


namespace mike_games_l177_177160

theorem mike_games (initial_money spent_money game_cost remaining_games : ℕ)
  (h1 : initial_money = 101)
  (h2 : spent_money = 47)
  (h3 : game_cost = 6)
  (h4 : remaining_games = (initial_money - spent_money) / game_cost) :
  remaining_games = 9 := by
  sorry

end mike_games_l177_177160


namespace octagon_diagonals_l177_177997

def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem octagon_diagonals : number_of_diagonals 8 = 20 :=
by
  sorry

end octagon_diagonals_l177_177997


namespace find_e_l177_177730

variables (j p t b a : ℝ) (e : ℝ)

theorem find_e
  (h1 : j = 0.75 * p)
  (h2 : j = 0.80 * t)
  (h3 : t = p - (e / 100) * p)
  (h4 : b = 1.40 * j)
  (h5 : a = 0.85 * b)
  (h6 : e = 2 * ((p - a) / p) * 100) :
  e = 21.5 := by
  sorry

end find_e_l177_177730


namespace cricket_run_rate_l177_177279

theorem cricket_run_rate (run_rate_first_10_overs : ℝ) (target_runs : ℝ) (first_overs : ℝ) (remaining_overs : ℝ):
  run_rate_first_10_overs = 6.2 → 
  target_runs = 282 →
  first_overs = 10 →
  remaining_overs = 40 →
  (target_runs - run_rate_first_10_overs * first_overs) / remaining_overs = 5.5 :=
by
  intros h1 h2 h3 h4
  -- Insert proof here
  sorry

end cricket_run_rate_l177_177279


namespace quadratic_trinomial_positive_c_l177_177991

theorem quadratic_trinomial_positive_c
  (a b c : ℝ)
  (h1 : b^2 < 4 * a * c)
  (h2 : a + b + c > 0) :
  c > 0 :=
sorry

end quadratic_trinomial_positive_c_l177_177991


namespace same_asymptotes_hyperbolas_l177_177548

theorem same_asymptotes_hyperbolas (M : ℝ) :
  (∀ x y : ℝ, ((x^2 / 9) - (y^2 / 16) = 1) ↔ ((y^2 / 32) - (x^2 / M) = 1)) →
  M = 18 :=
by
  sorry

end same_asymptotes_hyperbolas_l177_177548


namespace count_five_digit_numbers_with_digit_8_l177_177210

theorem count_five_digit_numbers_with_digit_8 : 
    let total_numbers := 99999 - 10000 + 1
    let without_8 := 8 * (9 ^ 4)
    90000 - without_8 = 37512 := by
    let total_numbers := 99999 - 10000 + 1 -- Total number of five-digit numbers
    let without_8 := 8 * (9 ^ 4) -- Number of five-digit numbers without any '8'
    show total_numbers - without_8 = 37512
    sorry

end count_five_digit_numbers_with_digit_8_l177_177210


namespace rectangle_cut_dimensions_l177_177915

-- Define the original dimensions of the rectangle as constants.
def original_length : ℕ := 12
def original_height : ℕ := 6

-- Define the dimensions of the new rectangle after slicing parallel to the longer side.
def new_length := original_length / 2
def new_height := original_height

-- The theorem statement.
theorem rectangle_cut_dimensions :
  new_length = 6 ∧ new_height = 6 :=
by
  sorry

end rectangle_cut_dimensions_l177_177915


namespace correct_statement_about_K_l177_177943

-- Defining the possible statements about the chemical equilibrium constant K
def K (n : ℕ) : String :=
  match n with
  | 1 => "The larger the K, the smaller the conversion rate of the reactants."
  | 2 => "K is related to the concentration of the reactants."
  | 3 => "K is related to the concentration of the products."
  | 4 => "K is related to temperature."
  | _ => "Invalid statement"

-- Given that the correct answer is that K is related to temperature
theorem correct_statement_about_K : K 4 = "K is related to temperature." :=
by
  rfl

end correct_statement_about_K_l177_177943


namespace neither_necessary_nor_sufficient_l177_177621

def set_M : Set ℝ := {x | x > 2}
def set_P : Set ℝ := {x | x < 3}

theorem neither_necessary_nor_sufficient (x : ℝ) :
  (x ∈ set_M ∨ x ∈ set_P) ↔ (x ∉ set_M ∩ set_P) :=
sorry

end neither_necessary_nor_sufficient_l177_177621


namespace find_larger_number_l177_177119

theorem find_larger_number (L S : ℕ)
  (h1 : L - S = 1370)
  (h2 : L = 6 * S + 15) :
  L = 1641 := sorry

end find_larger_number_l177_177119


namespace tickets_not_went_to_concert_l177_177525

theorem tickets_not_went_to_concert :
  let total_tickets := 900
  let before_start := total_tickets * 3 / 4
  let remaining_after_start := total_tickets - before_start
  let after_first_song := remaining_after_start * 5 / 9
  let during_middle := 80
  remaining_after_start - (after_first_song + during_middle) = 20 := 
by
  let total_tickets := 900
  let before_start := total_tickets * 3 / 4
  let remaining_after_start := total_tickets - before_start
  let after_first_song := remaining_after_start * 5 / 9
  let during_middle := 80
  show remaining_after_start - (after_first_song + during_middle) = 20
  sorry

end tickets_not_went_to_concert_l177_177525


namespace area_of_triangle_is_18_l177_177652

-- Define the vertices of the triangle
def point1 : ℝ × ℝ := (1, 4)
def point2 : ℝ × ℝ := (7, 4)
def point3 : ℝ × ℝ := (1, 10)

-- Define a function to calculate the area of a triangle given three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2)

-- Statement of the problem
theorem area_of_triangle_is_18 :
  triangle_area point1 point2 point3 = 18 :=
by
  -- skipping the proof
  sorry

end area_of_triangle_is_18_l177_177652


namespace Cody_money_final_l177_177514

-- Define the initial amount of money Cody had
def Cody_initial : ℝ := 45.0

-- Define the birthday gift amount
def birthday_gift : ℝ := 9.0

-- Define the amount spent on the game
def game_expense : ℝ := 19.0

-- Define the percentage of remaining money spent on clothes as a fraction
def clothes_spending_fraction : ℝ := 0.40

-- Define the late birthday gift received
def late_birthday_gift : ℝ := 4.5

-- Define the final amount of money Cody has
def Cody_final : ℝ :=
  let after_birthday := Cody_initial + birthday_gift
  let after_game := after_birthday - game_expense
  let spent_on_clothes := clothes_spending_fraction * after_game
  let after_clothes := after_game - spent_on_clothes
  after_clothes + late_birthday_gift

theorem Cody_money_final : Cody_final = 25.5 := by
  sorry

end Cody_money_final_l177_177514


namespace candy_given_l177_177486

theorem candy_given (A R G : ℕ) (h1 : A = 15) (h2 : R = 9) : G = 6 :=
by
  sorry

end candy_given_l177_177486


namespace a_18_value_l177_177483

-- Define the concept of an "Equally Summed Sequence"
def equallySummedSequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  ∀ n, a n + a (n + 1) = c

-- Define the specific conditions for a_1 and the common sum
def specific_sequence (a : ℕ → ℝ) : Prop :=
  equallySummedSequence a 5 ∧ a 1 = 2

-- The theorem we want to prove
theorem a_18_value (a : ℕ → ℝ) (h : specific_sequence a) : a 18 = 3 :=
sorry

end a_18_value_l177_177483


namespace isosceles_triangle_perimeter_l177_177475

-- Definitions of the side lengths
def side1 : ℝ := 8
def side2 : ℝ := 4

-- Theorem to prove the perimeter of the isosceles triangle
theorem isosceles_triangle_perimeter (side1 side2 : ℝ) (h1 : side1 = 8 ∨ side2 = 8) (h2 : side1 = 4 ∨ side2 = 4) : ∃ p : ℝ, p = 20 :=
by
  -- We omit the proof using sorry
  sorry

end isosceles_triangle_perimeter_l177_177475


namespace ken_house_distance_condition_l177_177329

noncomputable def ken_distance_to_dawn : ℕ := 4 -- This is the correct answer

theorem ken_house_distance_condition (K M : ℕ) (h1 : K = 2 * M) (h2 : K + M + M + K = 12) :
  K = ken_distance_to_dawn :=
  by
  sorry

end ken_house_distance_condition_l177_177329


namespace fraction_part_of_twenty_five_l177_177923

open Nat

def eighty_percent (x : ℕ) : ℕ := (85 * x) / 100

theorem fraction_part_of_twenty_five (x y : ℕ) (h1 : eighty_percent 40 = 34) (h2 : 34 - y = 14) (h3 : y = (4 * 25) / 5) : y = 20 :=
by 
  -- Given h1: eighty_percent 40 = 34
  -- And h2: 34 - y = 14
  -- And h3: y = (4 * 25) / 5
  -- Show y = 20
  sorry

end fraction_part_of_twenty_five_l177_177923


namespace mrs_hilt_apples_l177_177241

theorem mrs_hilt_apples (hours : ℕ := 3) (rate : ℕ := 5) : 
  (rate * hours) = 15 := 
by sorry

end mrs_hilt_apples_l177_177241


namespace temperature_range_l177_177389

-- Define the problem conditions
def highest_temp := 26
def lowest_temp := 12

-- The theorem stating the range of temperature change
theorem temperature_range : ∀ t : ℝ, lowest_temp ≤ t ∧ t ≤ highest_temp :=
by sorry

end temperature_range_l177_177389


namespace combined_stickers_count_l177_177037

theorem combined_stickers_count :
  let initial_june_stickers := 76
  let initial_bonnie_stickers := 63
  let stickers_given := 25
  let june_total := initial_june_stickers + stickers_given
  let bonnie_total := initial_bonnie_stickers + stickers_given
  june_total + bonnie_total = 189 :=
by
  -- Definitions
  let initial_june_stickers := 76
  let initial_bonnie_stickers := 63
  let stickers_given := 25
  
  -- Calculations
  let june_total := initial_june_stickers + stickers_given
  let bonnie_total := initial_bonnie_stickers + stickers_given

  -- Proof is omitted
  sorry

end combined_stickers_count_l177_177037


namespace pie_crusts_flour_l177_177614

theorem pie_crusts_flour (initial_crusts : ℕ)
  (initial_flour_per_crust : ℚ)
  (new_crusts : ℕ)
  (total_flour : ℚ)
  (h1 : initial_crusts = 40)
  (h2 : initial_flour_per_crust = 1/8)
  (h3 : new_crusts = 25)
  (h4 : total_flour = initial_crusts * initial_flour_per_crust) :
  (new_crusts * (total_flour / new_crusts) = total_flour) :=
by
  sorry

end pie_crusts_flour_l177_177614


namespace new_person_weight_l177_177146

theorem new_person_weight
    (avg_weight_20 : ℕ → ℕ)
    (total_weight_20 : ℕ)
    (avg_weight_21 : ℕ)
    (count_20 : ℕ)
    (count_21 : ℕ) :
    avg_weight_20 count_20 = 58 →
    total_weight_20 = count_20 * avg_weight_20 count_20 →
    avg_weight_21 = 53 →
    count_21 = count_20 + 1 →
    ∃ (W : ℕ), total_weight_20 + W = count_21 * avg_weight_21 ∧ W = 47 := 
by 
  sorry

end new_person_weight_l177_177146


namespace probability_unit_square_not_touch_central_2x2_square_l177_177360

-- Given a 6x6 checkerboard with a marked 2x2 square at the center,
-- prove that the probability of choosing a unit square that does not touch
-- the marked 2x2 square is 2/3.

theorem probability_unit_square_not_touch_central_2x2_square : 
    let total_squares := 36
    let touching_squares := 12
    let squares_not_touching := total_squares - touching_squares
    (squares_not_touching : ℚ) / (total_squares : ℚ) = 2 / 3 := by
  sorry

end probability_unit_square_not_touch_central_2x2_square_l177_177360


namespace largest_constant_inequality_equality_condition_l177_177932

theorem largest_constant_inequality (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆) ^ 2 ≥
    3 * (x₁ * (x₂ + x₃) + x₂ * (x₃ + x₄) + x₃ * (x₄ + x₅) + x₄ * (x₅ + x₆) + x₅ * (x₆ + x₁) + x₆ * (x₁ + x₂)) :=
sorry

theorem equality_condition (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₄ = x₂ + x₅) ∧ (x₂ + x₅ = x₃ + x₆) :=
sorry

end largest_constant_inequality_equality_condition_l177_177932


namespace olivia_race_time_l177_177947

variable (O E : ℕ)

theorem olivia_race_time (h1 : O + E = 112) (h2 : E = O - 4) : O = 58 :=
sorry

end olivia_race_time_l177_177947


namespace find_A_l177_177817

-- Define the four-digit number being a multiple of 9 and the sum of its digits condition
def digit_sum_multiple_of_9 (A : ℤ) : Prop :=
  (3 + A + A + 1) % 9 = 0

-- The Lean statement for the proof problem
theorem find_A (A : ℤ) (h : digit_sum_multiple_of_9 A) : A = 7 :=
sorry

end find_A_l177_177817


namespace plan_y_more_cost_effective_l177_177410

theorem plan_y_more_cost_effective (m : Nat) : 2500 + 7 * m < 15 * m → 313 ≤ m :=
by
  intro h
  sorry

end plan_y_more_cost_effective_l177_177410


namespace pairs_satisfy_inequality_l177_177689

section inequality_problem

variables (a b : ℝ)

-- Conditions
variable (hb1 : b ≠ -1)
variable (hb2 : b ≠ 0)

-- Inequalities to check
def inequality (a b : ℝ) : Prop :=
  (1 + a) ^ 2 / (1 + b) ≤ 1 + a ^ 2 / b

-- Main theorem
theorem pairs_satisfy_inequality :
  (b > 0 ∨ b < -1 → ∀ a, a ≠ b → inequality a b) ∧
  (∀ a, a ≠ -1 ∧ a ≠ 0 → inequality a a) :=
by
  sorry

end inequality_problem

end pairs_satisfy_inequality_l177_177689


namespace number_of_girls_joined_l177_177767

-- Define the initial conditions
def initial_girls := 18
def initial_boys := 15
def boys_quit := 4
def total_children_after_changes := 36

-- Define the changes
def boys_after_quit := initial_boys - boys_quit
def girls_after_changes := total_children_after_changes - boys_after_quit
def girls_joined := girls_after_changes - initial_girls

-- State the theorem
theorem number_of_girls_joined :
  girls_joined = 7 :=
by
  sorry

end number_of_girls_joined_l177_177767


namespace SammyFinishedProblems_l177_177201

def initial : ℕ := 9 -- number of initial math problems
def remaining : ℕ := 7 -- number of remaining math problems
def finished (init rem : ℕ) : ℕ := init - rem -- defining number of finished problems

theorem SammyFinishedProblems : finished initial remaining = 2 := by
  sorry -- placeholder for proof

end SammyFinishedProblems_l177_177201


namespace pills_in_a_week_l177_177124

def insulin_pills_per_day : Nat := 2
def blood_pressure_pills_per_day : Nat := 3
def anticonvulsant_pills_per_day : Nat := 2 * blood_pressure_pills_per_day

def total_pills_per_day : Nat := insulin_pills_per_day + blood_pressure_pills_per_day + anticonvulsant_pills_per_day

theorem pills_in_a_week : total_pills_per_day * 7 = 77 := by
  sorry

end pills_in_a_week_l177_177124


namespace max_value_of_a_le_2_and_ge_neg_2_max_a_is_2_l177_177469

theorem max_value_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) : a ≤ 2 :=
by
  -- Proof omitted
  sorry

theorem le_2_and_ge_neg_2 (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) : -2 ≤ a :=
by
  -- Proof omitted
  sorry

theorem max_a_is_2 (a : ℝ) (h3 : a ≤ 2) (h4 : -2 ≤ a) : a = 2 :=
by
  -- Proof omitted
  sorry

end max_value_of_a_le_2_and_ge_neg_2_max_a_is_2_l177_177469


namespace B1F_base16_to_base10_is_2847_l177_177955

theorem B1F_base16_to_base10_is_2847 : 
  let B := 11
  let one := 1
  let F := 15
  let base := 16
  B * base^2 + one * base^1 + F * base^0 = 2847 := 
by
  sorry

end B1F_base16_to_base10_is_2847_l177_177955


namespace right_triangle_345_l177_177225

theorem right_triangle_345 : 
  (∃ a b c : ℕ, a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2) ∧ 
  ¬(∃ a b c : ℕ, a = 2 ∧ b = 3 ∧ c = 4 ∧ a^2 + b^2 = c^2) ∧ 
  ¬(∃ a b c : ℕ, a = 4 ∧ b = 5 ∧ c = 6 ∧ a^2 + b^2 = c^2) ∧ 
  ¬(∃ a b c : ℕ, a = 6 ∧ b = 8 ∧ c = 9 ∧ a^2 + b^2 = c^2) :=
by {
  sorry
}

end right_triangle_345_l177_177225


namespace walking_time_l177_177065

noncomputable def time_to_reach_destination (mr_harris_speed : ℝ) (mr_harris_time_to_store : ℝ) (your_speed : ℝ) (distance_factor : ℝ) : ℝ :=
  let store_distance := mr_harris_speed * mr_harris_time_to_store
  let your_destination_distance := distance_factor * store_distance
  your_destination_distance / your_speed

theorem walking_time (mr_harris_speed your_speed : ℝ) (mr_harris_time_to_store : ℝ) (distance_factor : ℝ) (h_speed : your_speed = 2 * mr_harris_speed) (h_time : mr_harris_time_to_store = 2) (h_factor : distance_factor = 3) :
  time_to_reach_destination mr_harris_speed mr_harris_time_to_store your_speed distance_factor = 3 :=
by
  rw [h_time, h_speed, h_factor]
  -- calculations based on given conditions
  sorry

end walking_time_l177_177065


namespace cheese_stick_problem_l177_177184

theorem cheese_stick_problem (cheddar pepperjack mozzarella : ℕ) (total : ℕ)
    (h1 : cheddar = 15)
    (h2 : pepperjack = 45)
    (h3 : 2 * pepperjack = total)
    (h4 : total = cheddar + pepperjack + mozzarella) :
    mozzarella = 30 :=
by
    sorry

end cheese_stick_problem_l177_177184


namespace num_ways_to_write_360_as_increasing_seq_l177_177051

def is_consecutive_sum (n k : ℕ) : Prop :=
  let seq_sum := k * n + k * (k - 1) / 2
  seq_sum = 360

def valid_k (k : ℕ) : Prop :=
  k ≥ 2 ∧ k ∣ 360 ∧ (k = 2 ∨ (k - 1) % 2 = 0)

noncomputable def count_consecutive_sums : ℕ :=
  Nat.card {k // valid_k k ∧ ∃ n : ℕ, is_consecutive_sum n k}

theorem num_ways_to_write_360_as_increasing_seq : count_consecutive_sums = 4 :=
sorry

end num_ways_to_write_360_as_increasing_seq_l177_177051


namespace range_of_a_l177_177672

noncomputable def f (a x : ℝ) : ℝ :=
  Real.exp (x-2) + (1/3) * x^3 - (3/2) * x^2 + 2 * x - Real.log (x-1) + a

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, (1 < x → f a x = y) ↔ ∃ z : ℝ, 1 < z → f a (f a z) = y) →
  a ≤ 1/3 :=
sorry

end range_of_a_l177_177672


namespace gasoline_added_l177_177924

theorem gasoline_added (total_capacity : ℝ) (initial_fraction final_fraction : ℝ) 
(h1 : initial_fraction = 3 / 4)
(h2 : final_fraction = 9 / 10)
(h3 : total_capacity = 29.999999999999996) : 
(final_fraction * total_capacity - initial_fraction * total_capacity = 4.499999999999999) :=
by sorry

end gasoline_added_l177_177924


namespace prob_exactly_one_hits_is_one_half_prob_at_least_one_hits_is_two_thirds_l177_177366

def person_A_hits : ℚ := 1 / 2
def person_B_hits : ℚ := 1 / 3

def person_A_misses : ℚ := 1 - person_A_hits
def person_B_misses : ℚ := 1 - person_B_hits

def exactly_one_hits : ℚ := (person_A_hits * person_B_misses) + (person_B_hits * person_A_misses)
def at_least_one_hits : ℚ := 1 - (person_A_misses * person_B_misses)

theorem prob_exactly_one_hits_is_one_half : exactly_one_hits = 1 / 2 := sorry

theorem prob_at_least_one_hits_is_two_thirds : at_least_one_hits = 2 / 3 := sorry

end prob_exactly_one_hits_is_one_half_prob_at_least_one_hits_is_two_thirds_l177_177366


namespace path_length_cube_dot_l177_177138

-- Define the edge length of the cube
def edge_length : ℝ := 2

-- Define the distance of the dot from the center of the top face
def dot_distance_from_center : ℝ := 0.5

-- Define the number of complete rolls
def complete_rolls : ℕ := 2

-- Calculate the constant c such that the path length of the dot is c * π
theorem path_length_cube_dot : ∃ c : ℝ, dot_distance_from_center = 2.236 :=
by
  sorry

end path_length_cube_dot_l177_177138


namespace surveys_from_retired_is_12_l177_177206

-- Define the given conditions
def ratio_retired : ℕ := 2
def ratio_current : ℕ := 8
def ratio_students : ℕ := 40
def total_surveys : ℕ := 300
def total_ratio : ℕ := ratio_retired + ratio_current + ratio_students

-- Calculate the expected number of surveys from retired faculty
def number_of_surveys_retired : ℕ := total_surveys * ratio_retired / total_ratio

-- Lean 4 statement for proof
theorem surveys_from_retired_is_12 :
  number_of_surveys_retired = 12 :=
by
  -- Proof to be filled in
  sorry

end surveys_from_retired_is_12_l177_177206


namespace possible_values_of_t_l177_177276

theorem possible_values_of_t
  (theta : ℝ) 
  (x y t : ℝ) :
  x = Real.cos theta →
  y = Real.sin theta →
  t = (Real.sin theta) ^ 2 + (Real.cos theta) ^ 2 →
  x^2 + y^2 = 1 →
  t = 1 := by
  sorry

end possible_values_of_t_l177_177276


namespace travel_time_equation_l177_177582

theorem travel_time_equation
 (d : ℝ) (x t_saved factor : ℝ) 
 (h : d = 202) 
 (h1 : t_saved = 1.8) 
 (h2 : factor = 1.6)
 : (d / x) * factor = d / (x - t_saved) := sorry

end travel_time_equation_l177_177582


namespace first_player_wins_if_not_power_of_two_l177_177074

/-- 
  Prove that the first player can guarantee a win if and only if $n$ is not a power of two, under the given conditions. 
-/
theorem first_player_wins_if_not_power_of_two
  (n : ℕ) (h : n > 1) :
  (∃ k : ℕ, n = 2^k) ↔ false :=
sorry

end first_player_wins_if_not_power_of_two_l177_177074


namespace square_side_length_l177_177063

theorem square_side_length {s : ℝ} (h1 : 4 * s = 60) : s = 15 := 
by
  linarith

end square_side_length_l177_177063


namespace sum_m_n_l177_177312

-- Declare the namespaces and definitions for the problem
namespace DelegateProblem

-- Condition: total number of delegates
def total_delegates : Nat := 12

-- Condition: number of delegates from each country
def delegates_per_country : Nat := 4

-- Computation of m and n such that their sum is 452
-- This follows from the problem statement and the solution provided
def m : Nat := 221
def n : Nat := 231

-- Theorem statement in Lean for proving m + n = 452
theorem sum_m_n : m + n = 452 := by
  -- Algebraic proof omitted
  sorry

end DelegateProblem

end sum_m_n_l177_177312


namespace radius_of_spherical_circle_correct_l177_177154

noncomputable def radius_of_spherical_circle (rho theta phi : ℝ) : ℝ :=
  if rho = 1 ∧ phi = Real.pi / 4 then Real.sqrt 2 / 2 else 0

theorem radius_of_spherical_circle_correct :
  ∀ (theta : ℝ), radius_of_spherical_circle 1 theta (Real.pi / 4) = Real.sqrt 2 / 2 :=
by
  sorry

end radius_of_spherical_circle_correct_l177_177154


namespace number_of_dogs_per_box_l177_177380

-- Definition of the problem
def num_boxes : ℕ := 7
def total_dogs : ℕ := 28

-- Statement of the theorem to prove
theorem number_of_dogs_per_box (x : ℕ) (h : num_boxes * x = total_dogs) : x = 4 :=
by
  sorry

end number_of_dogs_per_box_l177_177380


namespace find_circle_center_l177_177795

def circle_center_eq : Prop :=
  ∃ (x y : ℝ), (x^2 - 6 * x + y^2 + 2 * y - 12 = 0) ∧ (x = 3) ∧ (y = -1)

theorem find_circle_center : circle_center_eq :=
sorry

end find_circle_center_l177_177795


namespace min_a_for_monotonic_increase_l177_177623

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x ^ 3 + 2 * a * x ^ 2 + 2

theorem min_a_for_monotonic_increase :
  ∀ a : ℝ, (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → x^2 + 4 * a * x ≥ 0) ↔ a ≥ -1/4 := sorry

end min_a_for_monotonic_increase_l177_177623


namespace arithmetic_geometric_sequence_S6_l177_177547

variables (S : ℕ → ℕ)

-- Definitions of conditions from a)
def S2 := S 2 = 3
def S4 := S 4 = 15

-- Main proof statement
theorem arithmetic_geometric_sequence_S6 (S : ℕ → ℕ) (h1 : S 2 = 3) (h2 : S 4 = 15) :
  S 6 = 63 :=
sorry

end arithmetic_geometric_sequence_S6_l177_177547


namespace area_of_closed_shape_l177_177892

theorem area_of_closed_shape :
  ∫ y in (-2 : ℝ)..3, ((2:ℝ)^y + 2 - (2:ℝ)^y) = 10 := by
  sorry

end area_of_closed_shape_l177_177892


namespace fifteenth_term_arithmetic_sequence_l177_177879

theorem fifteenth_term_arithmetic_sequence (a d : ℤ) : 
  (a + 20 * d = 17) ∧ (a + 21 * d = 20) → (a + 14 * d = -1) := by
  sorry

end fifteenth_term_arithmetic_sequence_l177_177879


namespace rectangle_area_diagonal_ratio_l177_177596

theorem rectangle_area_diagonal_ratio (d : ℝ) (x : ℝ) (h_ratio : 5 * x ≥ 0 ∧ 2 * x ≥ 0)
  (h_diagonal : d^2 = (5 * x)^2 + (2 * x)^2) :
  ∃ k : ℝ, (5 * x) * (2 * x) = k * d^2 ∧ k = 10 / 29 :=
by
  sorry

end rectangle_area_diagonal_ratio_l177_177596


namespace minimum_value_of_f_l177_177745

noncomputable def f (x : ℝ) : ℝ := sorry

theorem minimum_value_of_f :
  (∀ x : ℝ, f (x + 1) + f (x - 1) = 2 * x^2 - 4 * x) →
  ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ m = -2 :=
by
  sorry

end minimum_value_of_f_l177_177745


namespace greatest_possible_mean_BC_l177_177211

theorem greatest_possible_mean_BC :
  ∀ (A_n B_n C_weight C_n : ℕ),
    (A_n > 0) ∧ (B_n > 0) ∧ (C_n > 0) ∧
    (40 * A_n + 50 * B_n) / (A_n + B_n) = 43 ∧
    (40 * A_n + C_weight) / (A_n + C_n) = 44 →
    ∃ k : ℕ, ∃ n : ℕ, 
      A_n = 7 * k ∧ B_n = 3 * k ∧ 
      C_weight = 28 * k + 44 * n ∧ 
      44 + 46 * k / (3 * k + n) ≤ 59 :=
sorry

end greatest_possible_mean_BC_l177_177211


namespace pages_share_units_digit_l177_177015

def units_digit (n : Nat) : Nat :=
  n % 10

theorem pages_share_units_digit :
  (∃ (x_set : Finset ℕ), (∀ (x : ℕ), x ∈ x_set ↔ (1 ≤ x ∧ x ≤ 63 ∧ units_digit x = units_digit (64 - x))) ∧ x_set.card = 13) :=
by
  sorry

end pages_share_units_digit_l177_177015


namespace rectangle_area_l177_177604

noncomputable def side_of_square : ℝ := Real.sqrt 625

noncomputable def radius_of_circle : ℝ := side_of_square

noncomputable def length_of_rectangle : ℝ := (2 / 5) * radius_of_circle

def breadth_of_rectangle : ℝ := 10

theorem rectangle_area :
  length_of_rectangle * breadth_of_rectangle = 100 := 
by
  simp [length_of_rectangle, breadth_of_rectangle, radius_of_circle, side_of_square]
  sorry

end rectangle_area_l177_177604


namespace negation_proposition_l177_177512

theorem negation_proposition (m : ℤ) :
  ¬(∃ x : ℤ, x^2 + 2*x + m < 0) ↔ ∀ x : ℤ, x^2 + 2*x + m ≥ 0 :=
by
  sorry

end negation_proposition_l177_177512


namespace find_a_b_find_range_of_x_l177_177083

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (Real.log x / Real.log 2)^2 - 2 * a * (Real.log x / Real.log 2) + b

theorem find_a_b (a b : ℝ) :
  (f (1/4) a b = -1) → (a = -2 ∧ b = 3) :=
by
  sorry

theorem find_range_of_x (a b : ℝ) :
  a = -2 → b = 3 →
  ∀ x : ℝ, (f x a b < 0) → (1/8 < x ∧ x < 1/2) :=
by
  sorry

end find_a_b_find_range_of_x_l177_177083
