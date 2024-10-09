import Mathlib

namespace sin_150_equals_half_l1794_179477

theorem sin_150_equals_half : Real.sin (150 * Real.pi / 180) = 1 / 2 := 
by 
  sorry

end sin_150_equals_half_l1794_179477


namespace can_form_triangle_l1794_179413

theorem can_form_triangle (a b c : ℕ) (h1 : a = 5) (h2 : b = 6) (h3 : c = 10) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  rw [h1, h2, h3]
  repeat {sorry}

end can_form_triangle_l1794_179413


namespace sum_of_reciprocal_transformed_roots_l1794_179404

theorem sum_of_reciprocal_transformed_roots :
  ∀ (a b c : ℝ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    -1 < a ∧ a < 1 ∧
    -1 < b ∧ b < 1 ∧
    -1 < c ∧ c < 1 ∧
    (45 * a ^ 3 - 70 * a ^ 2 + 28 * a - 2 = 0) ∧
    (45 * b ^ 3 - 70 * b ^ 2 + 28 * b - 2 = 0) ∧
    (45 * c ^ 3 - 70 * c ^ 2 + 28 * c - 2 = 0)
  → (1 - a)⁻¹ + (1 - b)⁻¹ + (1 - c)⁻¹ = 13 / 9 := 
by 
  sorry

end sum_of_reciprocal_transformed_roots_l1794_179404


namespace sum_odd_digits_from_1_to_200_l1794_179464

/-- Function to compute the sum of odd digits of a number -/
def odd_digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.filter (fun d => d % 2 = 1) |>.sum

/-- Statement of the problem to prove the sum of the odd digits of numbers from 1 to 200 is 1000 -/
theorem sum_odd_digits_from_1_to_200 : (Finset.range 200).sum odd_digit_sum = 1000 := 
  sorry

end sum_odd_digits_from_1_to_200_l1794_179464


namespace infinite_cube_volume_sum_l1794_179456

noncomputable def sum_of_volumes_of_infinite_cubes (a : ℝ) : ℝ :=
  ∑' n, (((a / (3 ^ n))^3))

theorem infinite_cube_volume_sum (a : ℝ) : sum_of_volumes_of_infinite_cubes a = (27 / 26) * a^3 :=
sorry

end infinite_cube_volume_sum_l1794_179456


namespace direction_vector_of_line_l1794_179498

theorem direction_vector_of_line : 
  ∃ v : ℝ × ℝ, 
  (∀ x y : ℝ, 2 * y + x = 3 → v = (-2, -1)) :=
by
  sorry

end direction_vector_of_line_l1794_179498


namespace red_before_green_probability_l1794_179495

open Classical

noncomputable def probability_red_before_green (total_chips : ℕ) (red_chips : ℕ) (green_chips : ℕ) : ℚ :=
  let total_arrangements := (Nat.choose (total_chips - 1) green_chips)
  let favorable_arrangements := Nat.choose (total_chips - red_chips - 1) (green_chips - 1)
  favorable_arrangements / total_arrangements

theorem red_before_green_probability :
  probability_red_before_green 8 4 3 = 3 / 7 :=
sorry

end red_before_green_probability_l1794_179495


namespace complement_union_correct_l1794_179466

open Set

theorem complement_union_correct :
  let P : Set ℕ := { x | x * (x - 3) ≥ 0 }
  let Q : Set ℕ := {2, 4}
  (compl P) ∪ Q = {1, 2, 4} :=
by
  let P : Set ℕ := { x | x * (x - 3) ≥ 0 }
  let Q : Set ℕ := {2, 4}
  have h : (compl P) ∪ Q = {1, 2, 4} := sorry
  exact h

end complement_union_correct_l1794_179466


namespace range_of_a_l1794_179499

noncomputable def f (x a b : ℝ) : ℝ := (2 * x^2 - a * x + b) * Real.log (x - 1)

theorem range_of_a (a b : ℝ) (h1 : ∀ x > 1, f x a b ≥ 0) : a ≤ 6 :=
by 
  let x := 2
  have hb_eq : b = 2 * a - 8 :=
    by sorry
  have ha_le_6 : a ≤ 6 :=
    by sorry
  exact ha_le_6

end range_of_a_l1794_179499


namespace polarEquationOfCircleCenter1_1Radius1_l1794_179497

noncomputable def circleEquationInPolarCoordinates (θ : ℝ) : ℝ := 2 * Real.cos (θ - 1)

theorem polarEquationOfCircleCenter1_1Radius1 (ρ θ : ℝ) 
  (h : Real.sqrt ((ρ * Real.cos θ - Real.cos 1)^2 + (ρ * Real.sin θ - Real.sin 1)^2) = 1) :
  ρ = circleEquationInPolarCoordinates θ :=
by sorry

end polarEquationOfCircleCenter1_1Radius1_l1794_179497


namespace find_a_l1794_179438

noncomputable def exists_nonconstant_function (a : ℝ) : Prop :=
  ∃ f : ℝ → ℝ, (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 ≠ f x2) ∧ 
  (∀ x : ℝ, f (a * x) = a^2 * f x) ∧
  (∀ x : ℝ, f (f x) = a * f x)

theorem find_a :
  ∀ (a : ℝ), exists_nonconstant_function a → (a = 0 ∨ a = 1) :=
by
  sorry

end find_a_l1794_179438


namespace tour_group_size_l1794_179476

def adult_price : ℕ := 8
def child_price : ℕ := 3
def total_spent : ℕ := 44

theorem tour_group_size :
  ∃ (x y : ℕ), adult_price * x + child_price * y = total_spent ∧ (x + y = 8 ∨ x + y = 13) :=
by
  sorry

end tour_group_size_l1794_179476


namespace count_3_digit_numbers_divisible_by_5_l1794_179450

theorem count_3_digit_numbers_divisible_by_5 :
  let a := 100
  let l := 995
  let d := 5
  let n := (l - a) / d + 1
  n = 180 :=
by
  sorry

end count_3_digit_numbers_divisible_by_5_l1794_179450


namespace odd_function_solution_l1794_179403

def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem odd_function_solution (f : ℝ → ℝ) (h1 : is_odd f) (h2 : ∀ x : ℝ, x > 0 → f x = x^3 + x + 1) :
  ∀ x : ℝ, x < 0 → f x = x^3 + x - 1 :=
by
  sorry

end odd_function_solution_l1794_179403


namespace number_of_pints_of_paint_l1794_179471

-- Statement of the problem
theorem number_of_pints_of_paint (A B : ℝ) (N : ℕ) 
  (large_cube_paint : ℝ) (hA : A = 4) (hB : B = 2) (hN : N = 125) 
  (large_cube_paint_condition : large_cube_paint = 1) : 
  (N * (B / A) ^ 2 * large_cube_paint = 31.25) :=
by {
  -- Given the conditions
  sorry
}

end number_of_pints_of_paint_l1794_179471


namespace fat_rings_per_group_l1794_179463

theorem fat_rings_per_group (F : ℕ)
  (h1 : ∀ F, (70 * (F + 4)) = (40 * (F + 4)) + 180)
  : F = 2 :=
sorry

end fat_rings_per_group_l1794_179463


namespace lina_collects_stickers_l1794_179491

theorem lina_collects_stickers :
  let a := 3
  let d := 2
  let n := 10
  let a_n := a + (n - 1) * d
  let S_n := (n / 2) * (a + a_n)
  S_n = 120 :=
by
  sorry

end lina_collects_stickers_l1794_179491


namespace find_9b_l1794_179457

variable (a b : ℚ)

theorem find_9b (h1 : 7 * a + 3 * b = 0) (h2 : a = b - 4) : 9 * b = 126 / 5 := 
by
  sorry

end find_9b_l1794_179457


namespace slope_of_AB_is_1_l1794_179467

noncomputable def circle1 := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 4 * p.1 + 2 * p.2 - 11 = 0 }
noncomputable def circle2 := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 14 * p.1 + 12 * p.2 + 60 = 0 }
def is_on_circle1 (p : ℝ × ℝ) := p ∈ circle1
def is_on_circle2 (p : ℝ × ℝ) := p ∈ circle2

theorem slope_of_AB_is_1 :
  ∃ A B : ℝ × ℝ,
  is_on_circle1 A ∧ is_on_circle2 A ∧
  is_on_circle1 B ∧ is_on_circle2 B ∧
  (B.2 - A.2) / (B.1 - A.1) = 1 :=
sorry

end slope_of_AB_is_1_l1794_179467


namespace negativity_of_c_plus_b_l1794_179446

variable (a b c : ℝ)

def isWithinBounds : Prop := (1 < a ∧ a < 2) ∧ (0 < b ∧ b < 1) ∧ (-2 < c ∧ c < -1)

theorem negativity_of_c_plus_b (h : isWithinBounds a b c) : c + b < 0 :=
sorry

end negativity_of_c_plus_b_l1794_179446


namespace fraction_division_l1794_179468

-- Definition of fractions involved
def frac1 : ℚ := 4 / 9
def frac2 : ℚ := 5 / 8

-- Statement of the proof problem
theorem fraction_division :
  (frac1 / frac2) = 32 / 45 :=
by {
  sorry
}

end fraction_division_l1794_179468


namespace sum_of_numbers_ge_1_1_l1794_179415

theorem sum_of_numbers_ge_1_1 :
  let numbers := [1.4, 0.9, 1.2, 0.5, 1.3]
  let threshold := 1.1
  let filtered_numbers := numbers.filter (fun x => x >= threshold)
  let sum_filtered := filtered_numbers.sum
  sum_filtered = 3.9 :=
by {
  sorry
}

end sum_of_numbers_ge_1_1_l1794_179415


namespace sqrt_nat_or_irrational_l1794_179441

theorem sqrt_nat_or_irrational {n : ℕ} : 
  (∃ m : ℕ, m^2 = n) ∨ (¬ ∃ q r : ℕ, r ≠ 0 ∧ (q^2 = n * r^2 ∧ r * r ≠ n * n)) :=
sorry

end sqrt_nat_or_irrational_l1794_179441


namespace min_value_of_expression_l1794_179452

noncomputable def f (m : ℝ) : ℝ :=
  let x1 := -m - (m^2 + 3 * m - 2)
  let x2 := -2 * m - x1
  x1 * (x2 + x1) + x2^2

theorem min_value_of_expression :
  ∃ m : ℝ, f m = 3 * (m - 1/2)^2 + 5/4 ∧ f m ≥ f (1/2) := by
  sorry

end min_value_of_expression_l1794_179452


namespace number_of_true_propositions_eq_2_l1794_179472

theorem number_of_true_propositions_eq_2 :
  (¬(∀ (a b : ℝ), a < 0 → b > 0 → a + b < 0)) ∧
  (∀ (α β : ℝ), α = 90 → β = 90 → α = β) ∧
  (∀ (α β : ℝ), α + β = 90 → (∀ (γ : ℝ), γ + α = 90 → β = γ)) ∧
  (¬(∀ (ℓ m n : ℕ), (ℓ ≠ m ∧ ℓ ≠ n ∧ m ≠ n) → (∀ (α β : ℝ), α = β))) →
  2 = 2 :=
by
  sorry

end number_of_true_propositions_eq_2_l1794_179472


namespace evaluate_expression_l1794_179400

theorem evaluate_expression (a : ℕ) (h : a = 2) : a^3 * a^4 = 128 := 
by
  sorry

end evaluate_expression_l1794_179400


namespace trains_cross_each_other_in_given_time_l1794_179462

noncomputable def trains_crossing_time (length1 length2 speed1_kmph speed2_kmph : ℝ) : ℝ :=
  let speed1 := (speed1_kmph * 1000) / 3600
  let speed2 := (speed2_kmph * 1000) / 3600
  let relative_speed := speed1 + speed2
  let total_distance := length1 + length2
  total_distance / relative_speed

theorem trains_cross_each_other_in_given_time :
  trains_crossing_time 300 400 36 18 = 46.67 :=
by
  -- expected proof here
  sorry

end trains_cross_each_other_in_given_time_l1794_179462


namespace youngest_child_age_possible_l1794_179459

theorem youngest_child_age_possible 
  (total_bill : ℝ) (mother_charge : ℝ) 
  (yearly_charge_per_child : ℝ) (minimum_charge_per_child : ℝ) 
  (num_children : ℤ) (children_total_bill : ℝ)
  (total_years : ℤ)
  (youngest_possible_age : ℤ) :
  total_bill = 15.30 →
  mother_charge = 6 →
  yearly_charge_per_child = 0.60 →
  minimum_charge_per_child = 0.90 →
  num_children = 3 →
  children_total_bill = total_bill - mother_charge →
  children_total_bill - num_children * minimum_charge_per_child = total_years * yearly_charge_per_child →
  total_years = 11 →
  youngest_possible_age = 1 :=
sorry

end youngest_child_age_possible_l1794_179459


namespace min_value_3x_4y_l1794_179475

theorem min_value_3x_4y
  (x y : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  3 * x + 4 * y = 21 :=
sorry

end min_value_3x_4y_l1794_179475


namespace solve_rings_l1794_179486

variable (B : ℝ) (S : ℝ)

def conditions := (S = (5/8) * (Real.sqrt B)) ∧ (S + B = 52)

theorem solve_rings : conditions B S → (S + B = 52) := by
  intros h
  sorry

end solve_rings_l1794_179486


namespace product_xyz_l1794_179480

theorem product_xyz (x y z : ℝ) (h1 : x = y) (h2 : x = 2 * z) (h3 : x = 7.999999999999999) :
    x * y * z = 255.9999999999998 := by
  sorry

end product_xyz_l1794_179480


namespace faster_train_speed_l1794_179429

theorem faster_train_speed (v : ℝ) (h_total_length : 100 + 100 = 200) 
  (h_cross_time : 8 = 8) (h_speeds : 3 * v = 200 / 8) : 2 * v = 50 / 3 :=
sorry

end faster_train_speed_l1794_179429


namespace ab_value_l1794_179421

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a * b = 9 :=
by
  sorry

end ab_value_l1794_179421


namespace fifth_equation_l1794_179442

noncomputable def equation_1 : Prop := 2 * 1 = 2
noncomputable def equation_2 : Prop := 2 ^ 2 * 1 * 3 = 3 * 4
noncomputable def equation_3 : Prop := 2 ^ 3 * 1 * 3 * 5 = 4 * 5 * 6

theorem fifth_equation
  (h1 : equation_1)
  (h2 : equation_2)
  (h3 : equation_3) :
  2 ^ 5 * 1 * 3 * 5 * 7 * 9 = 6 * 7 * 8 * 9 * 10 :=
by {
  sorry
}

end fifth_equation_l1794_179442


namespace repay_loan_with_interest_l1794_179453

theorem repay_loan_with_interest (amount_borrowed : ℝ) (interest_rate : ℝ) (total_payment : ℝ) 
  (h1 : amount_borrowed = 100) (h2 : interest_rate = 0.10) :
  total_payment = amount_borrowed + (amount_borrowed * interest_rate) :=
by sorry

end repay_loan_with_interest_l1794_179453


namespace cleaning_time_together_l1794_179487

theorem cleaning_time_together (t : ℝ) (h_t : 3 = t / 3) (h_john_time : 6 = 6) : 
  (5 / (1 / 6 + 1 / 9)) = 3.6 :=
by
  sorry

end cleaning_time_together_l1794_179487


namespace sum_of_cubes_is_zero_l1794_179407

theorem sum_of_cubes_is_zero 
  (a b : ℝ) 
  (h1 : a + b = 0) 
  (h2 : a * b = -1) : 
  a^3 + b^3 = 0 := by
  sorry

end sum_of_cubes_is_zero_l1794_179407


namespace smallest_solution_l1794_179488

theorem smallest_solution (x : ℝ) (h : x^2 + 10 * x - 24 = 0) : x = -12 :=
sorry

end smallest_solution_l1794_179488


namespace optimal_discount_sequence_saves_more_l1794_179430

theorem optimal_discount_sequence_saves_more :
  (let initial_price := 30
   let flat_discount := 5
   let percent_discount := 0.25
   let first_seq_price := ((initial_price - flat_discount) * (1 - percent_discount))
   let second_seq_price := ((initial_price * (1 - percent_discount)) - flat_discount)
   first_seq_price - second_seq_price = 1.25) :=
by
  sorry

end optimal_discount_sequence_saves_more_l1794_179430


namespace pairs_satisfying_condition_l1794_179431

theorem pairs_satisfying_condition :
  (∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 1000 ∧ 1 ≤ y ∧ y ≤ 1000 ∧ (x^2 + y^2) % 7 = 0) → 
  (∃ n : ℕ, n = 20164) :=
sorry

end pairs_satisfying_condition_l1794_179431


namespace no_discrepancy_l1794_179489

-- Definitions based on the conditions
def t1_hours : ℝ := 1.5 -- time taken clockwise in hours
def t2_minutes : ℝ := 90 -- time taken counterclockwise in minutes

-- Lean statement to prove the equivalence
theorem no_discrepancy : t1_hours * 60 = t2_minutes :=
by sorry

end no_discrepancy_l1794_179489


namespace mark_initial_money_l1794_179461

theorem mark_initial_money (X : ℝ) 
  (h1 : X = (1/2) * X + 14 + (1/3) * X + 16) : X = 180 := 
  by
  sorry

end mark_initial_money_l1794_179461


namespace find_number_l1794_179422

theorem find_number (x : ℤ) (h : 2 * x - 8 = -12) : x = -2 :=
by
  sorry

end find_number_l1794_179422


namespace slope_of_parallel_line_l1794_179410

theorem slope_of_parallel_line (m : ℚ) (b : ℚ) :
  (∀ x y : ℚ, 5 * x - 3 * y = 21 → y = (5 / 3) * x + b) →
  m = 5 / 3 :=
by
  intros hyp
  sorry

end slope_of_parallel_line_l1794_179410


namespace volume_increase_is_79_4_percent_l1794_179444

noncomputable def original_volume (L B H : ℝ) : ℝ := L * B * H

noncomputable def new_volume (L B H : ℝ) : ℝ :=
  (L * 1.15) * (B * 1.30) * (H * 1.20)

noncomputable def volume_increase (L B H : ℝ) : ℝ :=
  new_volume L B H - original_volume L B H

theorem volume_increase_is_79_4_percent (L B H : ℝ) :
  volume_increase L B H = 0.794 * original_volume L B H := by
  sorry

end volume_increase_is_79_4_percent_l1794_179444


namespace zog_words_count_l1794_179423

-- Defining the number of letters in the Zoggian alphabet
def num_letters : ℕ := 6

-- Function to calculate the number of words with n letters
def words_with_n_letters (n : ℕ) : ℕ := num_letters ^ n

-- Definition to calculate the total number of words with at most 4 letters
def total_words : ℕ :=
  (words_with_n_letters 1) +
  (words_with_n_letters 2) +
  (words_with_n_letters 3) +
  (words_with_n_letters 4)

-- Theorem statement
theorem zog_words_count : total_words = 1554 := by
  sorry

end zog_words_count_l1794_179423


namespace complex_subtraction_l1794_179458

def z1 : ℂ := 3 + (1 : ℂ)
def z2 : ℂ := 2 - (1 : ℂ)

theorem complex_subtraction : z1 - z2 = 1 + 2 * (1 : ℂ) :=
by
  sorry

end complex_subtraction_l1794_179458


namespace largest_number_with_two_moves_l1794_179411

theorem largest_number_with_two_moves (n : Nat) (matches_limit : Nat) (initial_number : Nat)
  (h_n : initial_number = 1405) (h_limit: matches_limit = 2) : n = 7705 :=
by
  sorry

end largest_number_with_two_moves_l1794_179411


namespace poly_has_two_distinct_negative_real_roots_l1794_179469

-- Definition of the polynomial equation
def poly_eq (p x : ℝ) : Prop :=
  x^4 + 4*p*x^3 + 2*x^2 + 4*p*x + 1 = 0

-- Theorem statement that needs to be proved
theorem poly_has_two_distinct_negative_real_roots (p : ℝ) :
  p > 1 → ∃ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ poly_eq p x1 ∧ poly_eq p x2 :=
by
  sorry

end poly_has_two_distinct_negative_real_roots_l1794_179469


namespace B_should_be_paid_2307_69_l1794_179473

noncomputable def A_work_per_day : ℚ := 1 / 15
noncomputable def B_work_per_day : ℚ := 1 / 10
noncomputable def C_work_per_day : ℚ := 1 / 20
noncomputable def combined_work_per_day : ℚ := A_work_per_day + B_work_per_day + C_work_per_day
noncomputable def total_work : ℚ := 1
noncomputable def total_wages : ℚ := 5000
noncomputable def time_taken : ℚ := total_work / combined_work_per_day
noncomputable def B_share_of_work : ℚ := B_work_per_day / combined_work_per_day
noncomputable def B_share_of_wages : ℚ := B_share_of_work * total_wages

theorem B_should_be_paid_2307_69 : B_share_of_wages = 2307.69 := by
  sorry

end B_should_be_paid_2307_69_l1794_179473


namespace travel_time_l1794_179428

theorem travel_time (time_Ngapara_Zipra : ℝ) 
  (h1 : time_Ngapara_Zipra = 60) 
  (h2 : ∃ time_Ningi_Zipra, time_Ningi_Zipra = 0.8 * time_Ngapara_Zipra) 
  : ∃ total_travel_time, total_travel_time = time_Ningi_Zipra + time_Ngapara_Zipra ∧ total_travel_time = 108 := 
by
  sorry

end travel_time_l1794_179428


namespace area_of_combined_rectangle_l1794_179409

theorem area_of_combined_rectangle
  (short_side : ℝ) (num_small_rectangles : ℕ) (total_area : ℝ)
  (h1 : num_small_rectangles = 4)
  (h2 : short_side = 7)
  (h3 : total_area = (3 * short_side + short_side) * (2 * short_side)) :
  total_area = 392 := by
  sorry

end area_of_combined_rectangle_l1794_179409


namespace total_bees_is_25_l1794_179494

def initial_bees : ℕ := 16
def additional_bees : ℕ := 9

theorem total_bees_is_25 : initial_bees + additional_bees = 25 := by
  sorry

end total_bees_is_25_l1794_179494


namespace remainder_division_l1794_179454

def f (x : ℝ) : ℝ := x^3 - 4 * x + 7

theorem remainder_division (x : ℝ) : f 3 = 22 := by
  sorry

end remainder_division_l1794_179454


namespace A_not_divisible_by_B_l1794_179448

variable (A B : ℕ)
variable (h1 : A ≠ B)
variable (h2 : (∀ i, (1 ≤ i ∧ i ≤ 7) → (∃! j, (1 ≤ j ∧ j ≤ 7) ∧ (j = i))))
variable (h3 : (∀ i, (1 ≤ i ∧ i ≤ 7) → (∃! j, (1 ≤ j ∧ j ≤ 7) ∧ (j = i))))

theorem A_not_divisible_by_B : ¬ (A % B = 0) :=
sorry

end A_not_divisible_by_B_l1794_179448


namespace minimum_value_expression_l1794_179418

theorem minimum_value_expression (x y : ℝ) : ∃ (m : ℝ), ∀ x y : ℝ, x^2 + 3 * x * y + y^2 ≥ m ∧ m = 0 :=
by
  use 0
  sorry

end minimum_value_expression_l1794_179418


namespace probability_draw_l1794_179490

theorem probability_draw (pA_win pA_not_lose : ℝ) (h1 : pA_win = 0.3) (h2 : pA_not_lose = 0.8) :
  pA_not_lose - pA_win = 0.5 :=
by 
  sorry

end probability_draw_l1794_179490


namespace find_a10_of_arithmetic_sequence_l1794_179436

theorem find_a10_of_arithmetic_sequence (a : ℕ → ℚ)
  (h_seq : ∀ n : ℕ, ∃ d : ℚ, ∀ m : ℕ, a (n + m + 1) = a (n + m) + d)
  (h_a1 : a 1 = 1)
  (h_a4 : a 4 = 4) :
  a 10 = -4 / 5 :=
sorry

end find_a10_of_arithmetic_sequence_l1794_179436


namespace circle_area_l1794_179427

/-
Circle A has a diameter equal to the radius of circle B.
The area of circle A is 16π square units.
Prove the area of circle B is 64π square units.
-/

theorem circle_area (rA dA rB : ℝ) (h1 : dA = 2 * rA) (h2 : rB = dA) (h3 : π * rA ^ 2 = 16 * π) : π * rB ^ 2 = 64 * π :=
by
  sorry

end circle_area_l1794_179427


namespace loraine_wax_usage_proof_l1794_179470

-- Conditions
variables (large_animals small_animals : ℕ)
variable (wax : ℕ)

-- Definitions based on conditions
def large_animal_wax := 4
def small_animal_wax := 2
def total_sticks := 20
def small_animals_wax := 12
def small_to_large_ratio := 3

-- Proof statement
theorem loraine_wax_usage_proof (h1 : small_animals_wax = small_animals * small_animal_wax)
  (h2 : small_animals = large_animals * small_to_large_ratio)
  (h3 : wax = small_animals_wax + large_animals * large_animal_wax) :
  wax = total_sticks := by
  sorry

end loraine_wax_usage_proof_l1794_179470


namespace correct_quadratic_equation_l1794_179481

def is_quadratic_with_one_variable (eq : String) : Prop :=
  eq = "x^2 + 1 = 0"

theorem correct_quadratic_equation :
  is_quadratic_with_one_variable "x^2 + 1 = 0" :=
by {
  sorry
}

end correct_quadratic_equation_l1794_179481


namespace colorful_family_children_count_l1794_179406

theorem colorful_family_children_count 
    (B W S x : ℕ)
    (h1 : B = W) (h2 : W = S)
    (h3 : (B - x) + W = 10)
    (h4 : W + (S + x) = 18) :
    B + W + S = 21 :=
by
  sorry

end colorful_family_children_count_l1794_179406


namespace geometric_sequence_problem_l1794_179445

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

theorem geometric_sequence_problem (a : ℕ → ℝ) (h1 : a 1 = 2)
  (h2 : a 1 + a 3 + a 5 = 14) (h_seq : geometric_sequence a) :
  (1 / a 1) + (1 / a 3) + (1 / a 5) = 7 / 8 := sorry

end geometric_sequence_problem_l1794_179445


namespace min_ratio_cyl_inscribed_in_sphere_l1794_179460

noncomputable def min_surface_area_to_volume_ratio (R r : ℝ) : ℝ :=
  let h := 2 * Real.sqrt (R^2 - r^2)
  let A := 2 * Real.pi * r * (h + r)
  let V := Real.pi * r^2 * h
  A / V

theorem min_ratio_cyl_inscribed_in_sphere (R : ℝ) :
  ∃ r h, h = 2 * Real.sqrt (R^2 - r^2) ∧
         min_surface_area_to_volume_ratio R r = (Real.sqrt (Real.sqrt 4 + 1))^3 / R := 
by {
  sorry
}

end min_ratio_cyl_inscribed_in_sphere_l1794_179460


namespace triangle_area_l1794_179401

theorem triangle_area (a b c : ℕ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c = 10)
  (right_triangle : a^2 + b^2 = c^2) : (1 / 2 : ℝ) * (a * b) = 24 := by
  sorry

end triangle_area_l1794_179401


namespace clock_hands_straight_twenty_four_hours_l1794_179419

noncomputable def hands_straight_per_day : ℕ :=
  2 * 22

theorem clock_hands_straight_twenty_four_hours :
  hands_straight_per_day = 44 :=
by
  sorry

end clock_hands_straight_twenty_four_hours_l1794_179419


namespace find_k_l1794_179492

theorem find_k {k : ℝ} (h : (∃ α β : ℝ, α ≠ 0 ∧ β ≠ 0 ∧ α / β = 3 / 1 ∧ α + β = -10 ∧ α * β = k)) : k = 18.75 :=
sorry

end find_k_l1794_179492


namespace plum_cost_l1794_179416

theorem plum_cost
  (total_fruits : ℕ)
  (total_cost : ℕ)
  (peach_cost : ℕ)
  (plums_bought : ℕ)
  (peaches_bought : ℕ)
  (P : ℕ) :
  total_fruits = 32 →
  total_cost = 52 →
  peach_cost = 1 →
  plums_bought = 20 →
  peaches_bought = total_fruits - plums_bought →
  total_cost = 20 * P + peaches_bought * peach_cost →
  P = 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end plum_cost_l1794_179416


namespace points_in_quadrants_l1794_179451

theorem points_in_quadrants (x y : ℝ) (h_line : 4 * x + 7 * y = 28)
  (h_equidistant : |x| = |y|) : 
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end points_in_quadrants_l1794_179451


namespace find_original_price_l1794_179425

-- Define the conditions for the problem
def original_price (P : ℝ) : Prop :=
  0.90 * P = 1620

-- Prove the original price P
theorem find_original_price (P : ℝ) (h : original_price P) : P = 1800 :=
by
  -- The proof goes here
  sorry

end find_original_price_l1794_179425


namespace square_form_l1794_179414

theorem square_form (m n : ℤ) : 
  ∃ k l : ℤ, (2 * m^2 + n^2)^2 = 2 * k^2 + l^2 :=
by
  let x := (2 * m^2 + n^2)
  let y := x^2
  let k := 2 * m * n
  let l := 2 * m^2 - n^2
  use k, l
  sorry

end square_form_l1794_179414


namespace problem_c_l1794_179420

noncomputable def M (a b : ℝ) := (a^4 + b^4) * (a^2 + b^2)
noncomputable def N (a b : ℝ) := (a^3 + b^3) ^ 2

theorem problem_c (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_neq : a ≠ b) : M a b > N a b := 
by
  -- Proof goes here
  sorry

end problem_c_l1794_179420


namespace workbooks_needed_l1794_179435

theorem workbooks_needed (classes : ℕ) (workbooks_per_class : ℕ) (spare_workbooks : ℕ) (total_workbooks : ℕ) :
  classes = 25 → workbooks_per_class = 144 → spare_workbooks = 80 → total_workbooks = 25 * 144 + 80 → 
  total_workbooks = classes * workbooks_per_class + spare_workbooks :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  exact h4

end workbooks_needed_l1794_179435


namespace fraction_of_total_money_spent_on_dinner_l1794_179479

-- Definitions based on conditions
def aaron_savings : ℝ := 40
def carson_savings : ℝ := 40
def total_savings : ℝ := aaron_savings + carson_savings

def ice_cream_cost_per_scoop : ℝ := 1.5
def scoops_each : ℕ := 6
def total_ice_cream_cost : ℝ := 2 * scoops_each * ice_cream_cost_per_scoop

def total_left : ℝ := 2

def total_spent : ℝ := total_savings - total_left
def dinner_cost : ℝ := total_spent - total_ice_cream_cost

-- Target statement
theorem fraction_of_total_money_spent_on_dinner : 
  (dinner_cost = 60) ∧ (total_savings = 80) → dinner_cost / total_savings = 3 / 4 :=
by
  intros h
  sorry

end fraction_of_total_money_spent_on_dinner_l1794_179479


namespace red_cards_count_l1794_179484

theorem red_cards_count (R B : ℕ) (h1 : R + B = 20) (h2 : 3 * R + 5 * B = 84) : R = 8 :=
sorry

end red_cards_count_l1794_179484


namespace inequality_am_gm_l1794_179449

theorem inequality_am_gm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 1/a + 1/b + 1/c ≥ a + b + c) : a + b + c ≥ 3 * a * b * c :=
sorry

end inequality_am_gm_l1794_179449


namespace additional_people_needed_l1794_179474

def total_days := 50
def initial_people := 40
def days_passed := 25
def work_completed := 0.40

theorem additional_people_needed : 
  ∃ additional_people : ℕ, additional_people = 8 :=
by
  -- Placeholder for the actual proof skipped with 'sorry'
  sorry

end additional_people_needed_l1794_179474


namespace not_all_inequalities_true_l1794_179408

theorem not_all_inequalities_true (a b c : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : 0 < b ∧ b < 1) (h₂ : 0 < c ∧ c < 1) :
  ¬(a * (1 - b) > 1 / 4 ∧ b * (1 - c) > 1 / 4 ∧ c * (1 - a) > 1 / 4) :=
  sorry

end not_all_inequalities_true_l1794_179408


namespace parabola_midpoint_length_squared_l1794_179437

theorem parabola_midpoint_length_squared :
  ∀ (A B : ℝ × ℝ), 
  (∃ (x y : ℝ), A = (x, 3*x^2 + 4*x + 2) ∧ B = (-x, -(3*x^2 + 4*x + 2)) ∧ ((A.1 + B.1) / 2 = 0) ∧ ((A.2 + B.2) / 2 = 0)) →
  dist A B^2 = 8 :=
by
  sorry

end parabola_midpoint_length_squared_l1794_179437


namespace no_member_of_T_divisible_by_9_but_some_member_divisible_by_4_l1794_179432

def sum_of_squares_of_four_consecutive_integers (n : ℤ) : ℤ :=
  (n - 2) ^ 2 + (n - 1) ^ 2 + n ^ 2 + (n + 1) ^ 2

def is_divisible_by (a b : ℤ) : Prop := b ≠ 0 ∧ a % b = 0

theorem no_member_of_T_divisible_by_9_but_some_member_divisible_by_4 :
  ¬ (∃ n : ℤ, is_divisible_by (sum_of_squares_of_four_consecutive_integers n) 9) ∧
  (∃ n : ℤ, is_divisible_by (sum_of_squares_of_four_consecutive_integers n) 4) :=
by 
  sorry

end no_member_of_T_divisible_by_9_but_some_member_divisible_by_4_l1794_179432


namespace jason_worked_hours_on_saturday_l1794_179485

def hours_jason_works (x y : ℝ) : Prop :=
  (4 * x + 6 * y = 88) ∧ (x + y = 18)

theorem jason_worked_hours_on_saturday (x y : ℝ) : hours_jason_works x y → y = 8 := 
by 
  sorry

end jason_worked_hours_on_saturday_l1794_179485


namespace object_travel_distance_in_one_hour_l1794_179493

/-- If an object travels at 3 feet per second, then it travels 10800 feet in one hour. -/
theorem object_travel_distance_in_one_hour
  (speed : ℕ) (seconds_in_minute : ℕ) (minutes_in_hour : ℕ)
  (h_speed : speed = 3)
  (h_seconds_in_minute : seconds_in_minute = 60)
  (h_minutes_in_hour : minutes_in_hour = 60) :
  (speed * (seconds_in_minute * minutes_in_hour) = 10800) :=
by
  sorry

end object_travel_distance_in_one_hour_l1794_179493


namespace find_percentage_decrease_l1794_179455

noncomputable def initialPrice : ℝ := 100
noncomputable def priceAfterJanuary : ℝ := initialPrice * 1.30
noncomputable def priceAfterFebruary : ℝ := priceAfterJanuary * 0.85
noncomputable def priceAfterMarch : ℝ := priceAfterFebruary * 1.10

theorem find_percentage_decrease :
  ∃ (y : ℝ), (priceAfterMarch * (1 - y / 100) = initialPrice) ∧ abs (y - 18) < 1 := 
sorry

end find_percentage_decrease_l1794_179455


namespace net_change_salary_l1794_179447

/-- Given an initial salary S and a series of percentage changes:
    20% increase, 10% decrease, 15% increase, and 5% decrease,
    prove that the net change in salary is 17.99%. -/
theorem net_change_salary (S : ℝ) :
  (1.20 * 0.90 * 1.15 * 0.95 - 1) * S = 0.1799 * S :=
sorry

end net_change_salary_l1794_179447


namespace price_reduction_l1794_179405

theorem price_reduction (C : ℝ) (h1 : C > 0) :
  let first_discounted_price := 0.7 * C
  let final_discounted_price := 0.8 * first_discounted_price
  let reduction := 1 - final_discounted_price / C
  reduction = 0.44 :=
by
  sorry

end price_reduction_l1794_179405


namespace gap_between_rails_should_be_12_24_mm_l1794_179482

noncomputable def initial_length : ℝ := 15
noncomputable def temperature_initial : ℝ := -8
noncomputable def temperature_max : ℝ := 60
noncomputable def expansion_coefficient : ℝ := 0.000012
noncomputable def change_in_temperature : ℝ := temperature_max - temperature_initial
noncomputable def final_length : ℝ := initial_length * (1 + expansion_coefficient * change_in_temperature)
noncomputable def gap : ℝ := (final_length - initial_length) * 1000  -- converted to mm

theorem gap_between_rails_should_be_12_24_mm
  : gap = 12.24 := by
  sorry

end gap_between_rails_should_be_12_24_mm_l1794_179482


namespace garden_area_l1794_179426

theorem garden_area 
  (property_width : ℕ)
  (property_length : ℕ)
  (garden_width_ratio : ℚ)
  (garden_length_ratio : ℚ)
  (width_ratio_eq : garden_width_ratio = (1 : ℚ) / 8)
  (length_ratio_eq : garden_length_ratio = (1 : ℚ) / 10)
  (property_width_eq : property_width = 1000)
  (property_length_eq : property_length = 2250) :
  (property_width * garden_width_ratio * property_length * garden_length_ratio = 28125) :=
  sorry

end garden_area_l1794_179426


namespace dot_product_square_ABCD_l1794_179412

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point := ⟨Q.x - P.x, Q.y - P.y⟩

def dot_product (v w : Point) : ℝ := v.x * w.x + v.y * w.y

def square_ABCD : Prop :=
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨2, 0⟩
  let C : Point := ⟨2, 2⟩
  let D : Point := ⟨0, 2⟩
  let E : Point := ⟨1, 0⟩  -- E is the midpoint of AB
  let EC := vector E C
  let ED := vector E D
  dot_product EC ED = 3

theorem dot_product_square_ABCD : square_ABCD := by
  sorry

end dot_product_square_ABCD_l1794_179412


namespace jake_delay_l1794_179433

-- Define the conditions as in a)
def floors_jake_descends : ℕ := 8
def steps_per_floor : ℕ := 30
def steps_per_second_jake : ℕ := 3
def elevator_time_seconds : ℕ := 60 -- 1 minute = 60 seconds

-- Define the statement based on c)
theorem jake_delay (floors : ℕ) (steps_floor : ℕ) (steps_second : ℕ) (elevator_time : ℕ) :
  (floors = floors_jake_descends) →
  (steps_floor = steps_per_floor) →
  (steps_second = steps_per_second_jake) →
  (elevator_time = elevator_time_seconds) →
  (floors * steps_floor / steps_second - elevator_time = 20) :=
by
  intros
  sorry

end jake_delay_l1794_179433


namespace intersection_of_sets_l1794_179478

open Set

theorem intersection_of_sets (A B : Set ℕ) (hA : A = {0, 1, 2, 3, 4, 5}) (hB : B = {2, 4, 6}) :
  A ∩ B = {2, 4} :=
by
  sorry

end intersection_of_sets_l1794_179478


namespace number_machine_output_l1794_179443

def machine (x : ℕ) : ℕ := x + 15 - 6

theorem number_machine_output : machine 68 = 77 := by
  sorry

end number_machine_output_l1794_179443


namespace fifth_term_sum_of_powers_of_4_l1794_179440

theorem fifth_term_sum_of_powers_of_4 :
  (4^0 + 4^1 + 4^2 + 4^3 + 4^4) = 341 := 
by
  sorry

end fifth_term_sum_of_powers_of_4_l1794_179440


namespace rabbits_and_raccoons_l1794_179434

variable (b_r t_r x : ℕ)

theorem rabbits_and_raccoons : 
  2 * b_r = x ∧ 3 * t_r = x ∧ b_r = t_r + 3 → x = 18 := 
by
  sorry

end rabbits_and_raccoons_l1794_179434


namespace inequality_proof_l1794_179424

theorem inequality_proof (a b c : ℝ) (h : a > b) : a / (c ^ 2 + 1) > b / (c ^ 2 + 1) :=
by
  sorry

end inequality_proof_l1794_179424


namespace original_cost_price_l1794_179465

theorem original_cost_price (SP : ℝ) (loss_percentage : ℝ) (C : ℝ) 
  (h1 : SP = 1275) 
  (h2 : loss_percentage = 15) 
  (h3 : SP = (1 - loss_percentage / 100) * C) : 
  C = 1500 := 
by 
  sorry

end original_cost_price_l1794_179465


namespace infinite_k_values_l1794_179402

theorem infinite_k_values (k : ℕ) : (∃ k, ∀ (a b c : ℕ),
  (a = 64 ∧ b ≥ 0 ∧ c = 0 ∧ k = 2^a * 3^b * 5^c) ↔
  Nat.lcm (Nat.lcm (2^8) (2^24 * 3^12)) k = 2^64) →
  ∃ (b : ℕ), true :=
by
  sorry

end infinite_k_values_l1794_179402


namespace third_number_pascals_triangle_61_numbers_l1794_179483

theorem third_number_pascals_triangle_61_numbers : (Nat.choose 60 2) = 1770 := by
  sorry

end third_number_pascals_triangle_61_numbers_l1794_179483


namespace amplitude_of_f_phase_shift_of_f_vertical_shift_of_f_l1794_179496

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (4 * x - Real.pi / 2) + 1

theorem amplitude_of_f : (∀ x y : ℝ, |f x - f y| ≤ 2 * |x - y|) := sorry

theorem phase_shift_of_f : (∃ φ : ℝ, φ = -Real.pi / 8) := sorry

theorem vertical_shift_of_f : (∃ v : ℝ, v = 1) := sorry

end amplitude_of_f_phase_shift_of_f_vertical_shift_of_f_l1794_179496


namespace statement_a_statement_b_statement_c_statement_d_l1794_179439

open Real

-- Statement A (incorrect)
theorem statement_a (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : ¬ (a*c > b*d) := sorry

-- Statement B (correct)
theorem statement_b (a b : ℝ) (h1 : b < a) (h2 : a < 0) : (1 / a < 1 / b) := sorry

-- Statement C (incorrect)
theorem statement_c (a b : ℝ) (h : 1 / (a^2) < 1 / (b^2)) : ¬ (a > abs b) := sorry

-- Statement D (correct)
theorem statement_d (a b m : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : m > 0) : (a + m) / (b + m) > a / b := sorry

end statement_a_statement_b_statement_c_statement_d_l1794_179439


namespace fraction_simplification_l1794_179417

theorem fraction_simplification : (98 / 210 : ℚ) = 7 / 15 := 
by 
  sorry

end fraction_simplification_l1794_179417
