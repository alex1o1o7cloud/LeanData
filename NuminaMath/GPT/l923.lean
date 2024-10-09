import Mathlib

namespace abs_expression_value_l923_92343

theorem abs_expression_value (x : ℤ) (h : x = -2023) : 
  abs (abs (abs (abs x - 2 * x) - abs x) - x) = 6069 :=
by sorry

end abs_expression_value_l923_92343


namespace sum_f_eq_28743_l923_92394

def f (n : ℕ) : ℕ := 4 * n ^ 3 - 6 * n ^ 2 + 4 * n + 13

theorem sum_f_eq_28743 : (Finset.range 13).sum (λ n => f (n + 1)) = 28743 :=
by
  -- Placeholder for actual proof
  sorry

end sum_f_eq_28743_l923_92394


namespace prob_task1_and_not_task2_l923_92326

def prob_task1_completed : ℚ := 5 / 8
def prob_task2_completed : ℚ := 3 / 5

theorem prob_task1_and_not_task2 : 
  ((prob_task1_completed) * (1 - prob_task2_completed)) = 1 / 4 := 
by 
  sorry

end prob_task1_and_not_task2_l923_92326


namespace solve_equation_simplify_expression_l923_92303

-- Problem (1)
theorem solve_equation : ∀ x : ℝ, x * (x + 6) = 8 * (x + 3) ↔ x = 6 ∨ x = -4 := by
  sorry

-- Problem (2)
theorem simplify_expression : ∀ a b : ℝ, a ≠ b → (a ≠ 0 ∧ b ≠ 0) →
  (3 * a ^ 2 - 3 * b ^ 2) / (a ^ 2 * b + a * b ^ 2) /
  (1 - (a ^ 2 + b ^ 2) / (2 * a * b)) = -6 / (a - b) := by
  sorry

end solve_equation_simplify_expression_l923_92303


namespace isosceles_triangle_of_cosine_condition_l923_92373

theorem isosceles_triangle_of_cosine_condition
  (A B C : ℝ)
  (h : 2 * Real.cos A * Real.cos B = 1 - Real.cos C) :
  A = B ∨ A = π - B :=
  sorry

end isosceles_triangle_of_cosine_condition_l923_92373


namespace matt_and_peter_worked_together_days_l923_92352

variables (W : ℝ) -- Represents total work
noncomputable def work_rate_peter := W / 35
noncomputable def work_rate_together := W / 20

theorem matt_and_peter_worked_together_days (x : ℝ) :
  (x / 20) + (14 / 35) = 1 → x = 12 :=
by {
  sorry
}

end matt_and_peter_worked_together_days_l923_92352


namespace complex_problem_solution_l923_92323

noncomputable def complex_problem (c d : ℂ) (h1 : c ≠ 0) (h2 : d ≠ 0) (h3 : c^2 - c * d + d^2 = 0) : ℂ :=
  (c^12 + d^12) / (c + d)^12

theorem complex_problem_solution (c d : ℂ) (h1 : c ≠ 0) (h2 : d ≠ 0) (h3 : c^2 - c * d + d^2 = 0) :
  complex_problem c d h1 h2 h3 = 2 / 81 := 
sorry

end complex_problem_solution_l923_92323


namespace points_three_units_away_from_neg3_l923_92336

theorem points_three_units_away_from_neg3 (x : ℝ) : (abs (x + 3) = 3) ↔ (x = 0 ∨ x = -6) :=
by
  sorry

end points_three_units_away_from_neg3_l923_92336


namespace students_on_field_trip_l923_92347

theorem students_on_field_trip 
    (vans : ℕ)
    (van_capacity : ℕ)
    (adults : ℕ)
    (students : ℕ)
    (H1 : vans = 3)
    (H2 : van_capacity = 8)
    (H3 : adults = 2)
    (H4 : students = vans * van_capacity - adults) :
    students = 22 := 
by 
  sorry

end students_on_field_trip_l923_92347


namespace same_color_probability_l923_92396

-- Define the total number of balls
def total_balls : ℕ := 4 + 6 + 5

-- Define the number of each color of balls
def white_balls : ℕ := 4
def black_balls : ℕ := 6
def red_balls : ℕ := 5

-- Define the events and probabilities
def pr_event (n : ℕ) (total : ℕ) : ℚ := n / total
def pr_cond_event (n : ℕ) (total : ℕ) : ℚ := n / total

-- Define the probabilities for each compound event
def pr_C1 : ℚ := pr_event white_balls total_balls * pr_cond_event (white_balls - 1) (total_balls - 1)
def pr_C2 : ℚ := pr_event black_balls total_balls * pr_cond_event (black_balls - 1) (total_balls - 1)
def pr_C3 : ℚ := pr_event red_balls total_balls * pr_cond_event (red_balls - 1) (total_balls - 1)

-- Define the total probability
def pr_C : ℚ := pr_C1 + pr_C2 + pr_C3

-- The goal is to prove that the total probability pr_C is equal to 31 / 105
theorem same_color_probability : pr_C = 31 / 105 := 
  by sorry

end same_color_probability_l923_92396


namespace jumps_correct_l923_92391

def R : ℕ := 157
def X : ℕ := 86
def total_jumps (R X : ℕ) : ℕ := R + (R + X)

theorem jumps_correct : total_jumps R X = 400 := by
  sorry

end jumps_correct_l923_92391


namespace sufficient_condition_perpendicular_l923_92350

variables {Plane Line : Type}
variables (l : Line) (α β : Plane)

-- Definitions for perpendicularity and parallelism
def perp (l : Line) (α : Plane) : Prop := sorry
def parallel (α β : Plane) : Prop := sorry

theorem sufficient_condition_perpendicular
  (h1 : perp l α) 
  (h2 : parallel α β) : 
  perp l β :=
sorry

end sufficient_condition_perpendicular_l923_92350


namespace product_neg_int_add_five_l923_92370

theorem product_neg_int_add_five:
  let x := -11 
  let y := -8 
  x * y + 5 = 93 :=
by
  -- Proof omitted
  sorry

end product_neg_int_add_five_l923_92370


namespace sum_mod_18_l923_92341

theorem sum_mod_18 :
  (65 + 66 + 67 + 68 + 69 + 70 + 71 + 72) % 18 = 8 :=
by
  sorry

end sum_mod_18_l923_92341


namespace find_y_l923_92377

open Complex

theorem find_y (y : ℝ) (h₁ : (3 : ℂ) + (↑y : ℂ) * I = z₁) 
  (h₂ : (2 : ℂ) - I = z₂) 
  (h₃ : z₁ / z₂ = 1 + I) 
  (h₄ : z₁ = (3 : ℂ) + (↑y : ℂ) * I) 
  (h₅ : z₂ = (2 : ℂ) - I)
  : y = 1 :=
sorry


end find_y_l923_92377


namespace mul_equiv_l923_92375

theorem mul_equiv :
  (213 : ℝ) * 16 = 3408 →
  (16 : ℝ) * 21.3 = 340.8 :=
by
  sorry

end mul_equiv_l923_92375


namespace parallel_perpendicular_implies_perpendicular_l923_92310

-- Definitions of the geometric relationships
variables {Line Plane : Type}
variables (a b : Line) (alpha beta : Plane)

-- Conditions as per the problem statement
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

-- Lean statement of the proof problem
theorem parallel_perpendicular_implies_perpendicular
  (h1 : parallel_line_plane a alpha)
  (h2 : perpendicular_line_plane b alpha) :
  perpendicular_lines a b :=  
sorry

end parallel_perpendicular_implies_perpendicular_l923_92310


namespace jerome_contact_list_count_l923_92398

theorem jerome_contact_list_count :
  (let classmates := 20
   let out_of_school_friends := classmates / 2
   let family := 3 -- two parents and one sister
   let total_contacts := classmates + out_of_school_friends + family
   total_contacts = 33) :=
by
  let classmates := 20
  let out_of_school_friends := classmates / 2
  let family := 3
  let total_contacts := classmates + out_of_school_friends + family
  show total_contacts = 33
  sorry

end jerome_contact_list_count_l923_92398


namespace average_percentage_of_15_students_l923_92362

open Real

theorem average_percentage_of_15_students :
  ∀ (x : ℝ),
  (15 + 10 = 25) →
  (10 * 90 = 900) →
  (25 * 84 = 2100) →
  (15 * x + 900 = 2100) →
  x = 80 :=
by
  intro x h_sum h_10_avg h_25_avg h_total
  sorry

end average_percentage_of_15_students_l923_92362


namespace rotated_and_shifted_line_eq_l923_92334

theorem rotated_and_shifted_line_eq :
  let rotate_line_90 (x y : ℝ) := ( -y, x )
  let shift_right (x y : ℝ) := (x + 1, y)
  ∃ (new_a new_b new_c : ℝ), 
  (∀ (x y : ℝ), (y = 3 * x → x * new_a + y * new_b + new_c = 0)) ∧ 
  (new_a = 1) ∧ (new_b = 3) ∧ (new_c = -1) := by
  sorry

end rotated_and_shifted_line_eq_l923_92334


namespace remainder_division_l923_92357

-- Definition of the number in terms of its components
def num : ℤ := 98 * 10^6 + 76 * 10^4 + 54 * 10^2 + 32

-- The modulus
def m : ℤ := 25

-- The given problem restated as a hypothesis and goal
theorem remainder_division : num % m = 7 :=
by
  sorry

end remainder_division_l923_92357


namespace value_of_c_in_base8_perfect_cube_l923_92382

theorem value_of_c_in_base8_perfect_cube (c : ℕ) (h : 0 ≤ c ∧ c < 8) :
  4 * 8^2 + c * 8 + 3 = x^3 → c = 0 := by
  sorry

end value_of_c_in_base8_perfect_cube_l923_92382


namespace find_product_l923_92330

theorem find_product
  (a b c d : ℝ) :
  3 * a + 2 * b + 4 * c + 6 * d = 60 →
  4 * (d + c) = b^2 →
  4 * b + 2 * c = a →
  c - 2 = d →
  a * b * c * d = 0 :=
by
  sorry

end find_product_l923_92330


namespace Trevor_future_age_when_brother_is_three_times_now_l923_92322

def Trevor_current_age := 11
def Brother_current_age := 20

theorem Trevor_future_age_when_brother_is_three_times_now :
  ∃ (X : ℕ), Brother_current_age + (X - Trevor_current_age) = 3 * Trevor_current_age :=
by
  use 24
  sorry

end Trevor_future_age_when_brother_is_three_times_now_l923_92322


namespace max_k_l923_92355

-- Definitions and conditions
def original_number (A B : ℕ) : ℕ := 10 * A + B
def new_number (A C B : ℕ) : ℕ := 100 * A + 10 * C + B

theorem max_k (A C B k : ℕ) (hA : A ≠ 0) (h1 : 0 ≤ A ∧ A ≤ 9) (h2 : 0 ≤ B ∧ B ≤ 9) (h3: 0 ≤ C ∧ C ≤ 9) :
  ((original_number A B) * k = (new_number A C B)) → 
  (∀ (A: ℕ), 1 ≤ k) → 
  k ≤ 19 :=
by
  sorry

end max_k_l923_92355


namespace find_k_l923_92332

-- Definitions
variable (m n k : ℝ)

-- Given conditions
def on_line_1 : Prop := m = 2 * n + 5
def on_line_2 : Prop := (m + 5) = 2 * (n + k) + 5

-- Desired conclusion
theorem find_k (h1 : on_line_1 m n) (h2 : on_line_2 m n k) : k = 2.5 :=
sorry

end find_k_l923_92332


namespace three_irrational_numbers_l923_92359

theorem three_irrational_numbers (a b c d e : ℝ) 
  (ha : ¬ ∃ q1 q2 : ℚ, a = q1 + q2) 
  (hb : ¬ ∃ q1 q2 : ℚ, b = q1 + q2) 
  (hc : ¬ ∃ q1 q2 : ℚ, c = q1 + q2) 
  (hd : ¬ ∃ q1 q2 : ℚ, d = q1 + q2) 
  (he : ¬ ∃ q1 q2 : ℚ, e = q1 + q2) : 
  ∃ x y z, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) 
  ∧ (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) 
  ∧ (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e)
  ∧ (¬ ∃ q1 q2 : ℚ, x + y = q1 + q2) 
  ∧ (¬ ∃ q1 q2 : ℚ, y + z = q1 + q2) 
  ∧ (¬ ∃ q1 q2 : ℚ, z + x = q1 + q2) :=
sorry

end three_irrational_numbers_l923_92359


namespace max_sequence_sum_l923_92368

variable {α : Type*} [LinearOrderedField α]

noncomputable def arithmeticSequence (a1 d : α) (n : ℕ) : α :=
  a1 + d * n

noncomputable def sequenceSum (a1 d : α) (n : ℕ) : α :=
  n * (a1 + (a1 + d * (n - 1))) / 2

theorem max_sequence_sum (a1 d : α) (n : ℕ) (hn : 5 ≤ n ∧ n ≤ 10)
    (h1 : d < 0) (h2 : sequenceSum a1 d 5 = sequenceSum a1 d 10) :
    n = 7 ∨ n = 8 :=
  sorry

end max_sequence_sum_l923_92368


namespace triangle_has_three_altitudes_l923_92313

-- Assuming a triangle in ℝ² space
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Definition of an altitude in the context of Lean
def altitude (T : Triangle) (p : ℝ × ℝ) := 
  ∃ (a : ℝ) (b : ℝ), T.A.1 * p.1 + T.A.2 * p.2 = a * p.1 + b -- Placeholder, real definition of altitude may vary

-- Prove that a triangle has exactly 3 altitudes
theorem triangle_has_three_altitudes (T : Triangle) : ∃ (p₁ p₂ p₃ : ℝ × ℝ), 
  altitude T p₁ ∧ altitude T p₂ ∧ altitude T p₃ :=
sorry

end triangle_has_three_altitudes_l923_92313


namespace age_difference_l923_92302

theorem age_difference (john_age father_age mother_age : ℕ) 
    (h1 : john_age * 2 = father_age) 
    (h2 : father_age = mother_age + 4) 
    (h3 : father_age = 40) :
    mother_age - john_age = 16 :=
by
  sorry

end age_difference_l923_92302


namespace georg_can_identify_fake_coins_l923_92360

theorem georg_can_identify_fake_coins :
  ∀ (coins : ℕ) (baron : ℕ → ℕ → ℕ) (queries : ℕ),
    coins = 100 →
    ∃ (fake_count : ℕ → ℕ) (exaggeration : ℕ),
      (∀ group_size : ℕ, 10 ≤ group_size ∧ group_size ≤ 20) →
      (∀ (show_coins : ℕ), show_coins ≤ group_size → fake_count show_coins = baron show_coins exaggeration) →
      queries < 120 :=
by
  sorry

end georg_can_identify_fake_coins_l923_92360


namespace polynomial_divisibility_by_120_l923_92361

theorem polynomial_divisibility_by_120 (n : ℤ) : 120 ∣ (n^5 - 5 * n^3 + 4 * n) :=
by
  sorry

end polynomial_divisibility_by_120_l923_92361


namespace find_x_such_that_fraction_eq_l923_92399

theorem find_x_such_that_fraction_eq 
  (x : ℚ) (h₁ : x ≠ 1) (h₂ : x ≠ 5) : 
  (x^2 - 4 * x + 3) / (x^2 - 6 * x + 5) = (x^2 - 3 * x - 10) / (x^2 - 2 * x - 15) ↔ 
  x = -19 / 3 :=
sorry

end find_x_such_that_fraction_eq_l923_92399


namespace relationship_between_sets_l923_92300

def M (x : ℤ) : Prop := ∃ k : ℤ, x = 5 * k - 2
def P (x : ℤ) : Prop := ∃ n : ℤ, x = 5 * n + 3
def S (x : ℤ) : Prop := ∃ m : ℤ, x = 10 * m + 3

theorem relationship_between_sets :
  (∀ x, S x → P x) ∧ (∀ x, P x → M x) ∧ (∀ x, M x → P x) :=
by
  sorry

end relationship_between_sets_l923_92300


namespace inequality_solution_l923_92388

theorem inequality_solution (x : ℝ) : 
  (x + 10) / (x^2 + 2 * x + 5) ≥ 0 ↔ x ∈ Set.Ici (-10) :=
sorry

end inequality_solution_l923_92388


namespace total_books_in_bookcase_l923_92365

def num_bookshelves := 8
def num_layers_per_bookshelf := 5
def books_per_layer := 85

theorem total_books_in_bookcase : 
  (num_bookshelves * num_layers_per_bookshelf * books_per_layer) = 3400 := by
  sorry

end total_books_in_bookcase_l923_92365


namespace probability_three_same_color_is_one_seventeenth_l923_92318

def standard_deck := {cards : Finset ℕ // cards.card = 52 ∧ ∃ reds blacks, reds.card = 26 ∧ blacks.card = 26 ∧ (reds ∪ blacks = cards)}

def num_ways_to_pick_3_same_color : ℕ :=
  (26 * 25 * 24) + (26 * 25 * 24)

def total_ways_to_pick_3 : ℕ :=
  52 * 51 * 50

def probability_top_three_same_color := (num_ways_to_pick_3_same_color / total_ways_to_pick_3 : ℚ)

theorem probability_three_same_color_is_one_seventeenth :
  probability_top_three_same_color = (1 / 17 : ℚ) := by sorry

end probability_three_same_color_is_one_seventeenth_l923_92318


namespace cars_to_sell_l923_92324

theorem cars_to_sell (n : ℕ) 
  (h1 : ∀ c, c ∈ {c' : ℕ | c' ≤ n} → ∃ m, m = 3)
  (h2 : ∀ c, c ∈ {c' : ℕ | c' ≤ n} → c ∈ {c' : ℕ | c' < 3})
  (h3 : 15 * 3 = 45)
  (h4 : ∀ n, n * 3 = 45 → n = 15):
  n = 15 := 
  by
    have n_eq: n * 3 = 45 := sorry
    exact h4 n n_eq

end cars_to_sell_l923_92324


namespace gcd_136_1275_l923_92346

theorem gcd_136_1275 : Nat.gcd 136 1275 = 17 := by
sorry

end gcd_136_1275_l923_92346


namespace proof_of_a_neg_two_l923_92333

theorem proof_of_a_neg_two (a : ℝ) (i : ℂ) (h_i : i^2 = -1) (h_real : (1 + i)^2 - a / i = (a + 2) * i → ∃ r : ℝ, (1 + i)^2 - a / i = r) : a = -2 :=
sorry

end proof_of_a_neg_two_l923_92333


namespace coordinates_with_respect_to_origin_l923_92329

theorem coordinates_with_respect_to_origin (x y : ℤ) (h : (x, y) = (2, -6)) : (x, y) = (2, -6) :=
by
  sorry

end coordinates_with_respect_to_origin_l923_92329


namespace verify_a_l923_92366

def g (x : ℝ) : ℝ := 5 * x - 7

theorem verify_a (a : ℝ) : g a = 0 ↔ a = 7 / 5 := by
  sorry

end verify_a_l923_92366


namespace percentage_B_to_C_l923_92364

variables (total_students : ℕ)
variables (pct_A pct_B pct_C pct_A_to_C pct_B_to_C : ℝ)

-- Given conditions
axiom total_students_eq_100 : total_students = 100
axiom pct_A_eq_60 : pct_A = 60
axiom pct_B_eq_40 : pct_B = 40
axiom pct_A_to_C_eq_30 : pct_A_to_C = 30
axiom pct_C_eq_34 : pct_C = 34

-- Proof goal
theorem percentage_B_to_C :
  pct_B_to_C = 40 :=
sorry

end percentage_B_to_C_l923_92364


namespace inequality_tangents_l923_92387

def f (x : ℝ) (a b : ℝ) : ℝ := x^3 - a * x - b

theorem inequality_tangents (a b : ℝ) (h1 : 0 < a)
  (h2 : ∃ x0 : ℝ, 2 * x0^3 - 3 * a * x0^2 + a^2 + 2 * b = 0): 
  -a^2 / 2 < b ∧ b < f a a b :=
by
  sorry

end inequality_tangents_l923_92387


namespace no_solution_iff_discriminant_l923_92353

theorem no_solution_iff_discriminant (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 1 ≥ 0) ↔ -2 ≤ k ∧ k ≤ 2 := by
  sorry

end no_solution_iff_discriminant_l923_92353


namespace range_of_a_l923_92340

-- Defining the function f(x)
def f (a x : ℝ) := x^2 + (a^2 - 1) * x + (a - 2)

-- The statement of the problem in Lean 4
theorem range_of_a (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 > 1 ∧ x2 < 1 ∧ f a x1 = 0 ∧ f a x2 = 0) : -2 < a ∧ a < 1 :=
by
  sorry -- Proof is omitted

end range_of_a_l923_92340


namespace percentage_error_in_square_area_l923_92321

-- Given an error of 1% in excess while measuring the side of a square,
-- prove that the percentage of error in the calculated area of the square is 2.01%.

theorem percentage_error_in_square_area (s : ℝ) (h : s ≠ 0) :
  let measured_side := 1.01 * s
  let actual_area := s ^ 2
  let calculated_area := (1.01 * s) ^ 2
  let error_in_area := calculated_area - actual_area
  let percentage_error := (error_in_area / actual_area) * 100
  percentage_error = 2.01 :=
by {
  let measured_side := 1.01 * s;
  let actual_area := s ^ 2;
  let calculated_area := (1.01 * s) ^ 2;
  let error_in_area := calculated_area - actual_area;
  let percentage_error := (error_in_area / actual_area) * 100;
  sorry
}

end percentage_error_in_square_area_l923_92321


namespace boat_speed_in_still_water_l923_92379

theorem boat_speed_in_still_water (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 5) : b = 8 := 
by
  /- The proof steps would go here -/
  sorry

end boat_speed_in_still_water_l923_92379


namespace sujis_age_l923_92320

theorem sujis_age (x : ℕ) (Abi Suji : ℕ)
  (h1 : Abi = 5 * x)
  (h2 : Suji = 4 * x)
  (h3 : (Abi + 3) / (Suji + 3) = 11 / 9) : 
  Suji = 24 := 
by 
  sorry

end sujis_age_l923_92320


namespace tg_half_product_l923_92325

open Real

variable (α β : ℝ)

theorem tg_half_product (h1 : sin α + sin β = 2 * sin (α + β))
                        (h2 : ∀ n : ℤ, α + β ≠ 2 * π * n) :
  tan (α / 2) * tan (β / 2) = 1 / 3 := by
  sorry

end tg_half_product_l923_92325


namespace rational_solutions_iff_k_equals_8_l923_92380

theorem rational_solutions_iff_k_equals_8 {k : ℕ} (hk : k > 0) :
  (∃ (x : ℚ), k * x^2 + 16 * x + k = 0) ↔ k = 8 :=
by
  sorry

end rational_solutions_iff_k_equals_8_l923_92380


namespace solve_system_eq_l923_92337

theorem solve_system_eq (x y : ℝ) :
  x^2 * y - x * y^2 - 5 * x + 5 * y + 3 = 0 ∧
  x^3 * y - x * y^3 - 5 * x^2 + 5 * y^2 + 15 = 0 ↔
  x = 4 ∧ y = 1 :=
sorry

end solve_system_eq_l923_92337


namespace Jason_toys_correct_l923_92393

variable (R Jn Js : ℕ)

def Rachel_toys : ℕ := 1

def John_toys (R : ℕ) : ℕ := R + 6

def Jason_toys (Jn : ℕ) : ℕ := 3 * Jn

theorem Jason_toys_correct (hR : R = 1) (hJn : Jn = John_toys R) (hJs : Js = Jason_toys Jn) : Js = 21 :=
by
  sorry

end Jason_toys_correct_l923_92393


namespace smallest_integer_m_l923_92304

theorem smallest_integer_m (m : ℕ) : m > 1 ∧ m % 13 = 2 ∧ m % 5 = 2 ∧ m % 3 = 2 → m = 197 := 
by 
  sorry

end smallest_integer_m_l923_92304


namespace spadesuit_calculation_l923_92301

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_calculation : spadesuit 2 (spadesuit 6 1) = -1221 := by
  sorry

end spadesuit_calculation_l923_92301


namespace entrants_total_l923_92309

theorem entrants_total (N : ℝ) (h1 : N > 800)
  (h2 : 0.35 * N = NumFemales)
  (h3 : 0.65 * N = NumMales)
  (h4 : NumMales - NumFemales = 252) :
  N = 840 := 
sorry

end entrants_total_l923_92309


namespace line_equation_l923_92389

variable (t : ℝ)
variable (x y : ℝ)

def param_x (t : ℝ) : ℝ := 3 * t + 2
def param_y (t : ℝ) : ℝ := 5 * t - 7

theorem line_equation :
  ∃ m b : ℝ, ∀ t : ℝ, y = param_y t ∧ x = param_x t → y = m * x + b := by
  use (5 / 3)
  use (-31 / 3)
  sorry

end line_equation_l923_92389


namespace eval_expression_l923_92372

theorem eval_expression (x : ℝ) (h₀ : x = 3) :
  let initial_expr : ℝ := (2 * x + 2) / (x - 2)
  let replaced_expr : ℝ := (2 * initial_expr + 2) / (initial_expr - 2)
  replaced_expr = 8 :=
by
  sorry

end eval_expression_l923_92372


namespace exist_n_div_k_l923_92348

open Function

theorem exist_n_div_k (k : ℕ) (h1 : k ≥ 1) (h2 : Nat.gcd k 6 = 1) :
  ∃ n : ℕ, n ≥ 0 ∧ k ∣ (2^n + 3^n + 6^n - 1) := 
sorry

end exist_n_div_k_l923_92348


namespace max_negatives_l923_92376

theorem max_negatives (a b c d e f : ℤ) (h : ab + cdef < 0) : ∃ w : ℤ, w = 4 := 
sorry

end max_negatives_l923_92376


namespace area_circle_l923_92311

-- Define the given condition
def polar_eq (r θ : ℝ) : Prop :=
  r = 3 * Real.cos θ - 4 * Real.sin θ

-- The goal is to prove the area of the circle described by the polar equation
theorem area_circle {r θ : ℝ} (h : polar_eq r θ) :
  ∃ A, A = π * (5 / 2) ^ 2 :=
sorry

end area_circle_l923_92311


namespace ratio_of_milk_water_in_larger_vessel_l923_92316

-- Definitions of conditions
def volume1 (V : ℝ) : ℝ := 3 * V
def volume2 (V : ℝ) : ℝ := 5 * V

def ratio_milk_water_1 : ℝ × ℝ := (1, 2)
def ratio_milk_water_2 : ℝ × ℝ := (3, 2)

-- Define the problem statement
theorem ratio_of_milk_water_in_larger_vessel (V : ℝ) (hV : V > 0) :
  (volume1 V / (ratio_milk_water_1.1 + ratio_milk_water_1.2)) = V ∧ 
  2 * (volume1 V / (ratio_milk_water_1.1 + ratio_milk_water_1.2)) = 2 * V ∧ 
  3 * (volume2 V / (ratio_milk_water_2.1 + ratio_milk_water_2.2)) = 3 * V ∧ 
  2 * (volume2 V / (ratio_milk_water_2.1 + ratio_milk_water_2.2)) = 2 * V →
  (4 * V) / (4 * V) = 1 :=
sorry

end ratio_of_milk_water_in_larger_vessel_l923_92316


namespace sum_reciprocals_of_roots_l923_92395

-- Problem statement: Prove that the sum of the reciprocals of the roots of the quadratic equation x^2 - 11x + 6 = 0 is 11/6.
theorem sum_reciprocals_of_roots : 
  ∀ (p q : ℝ), p + q = 11 → p * q = 6 → (1 / p + 1 / q = 11 / 6) :=
by
  intro p q hpq hprod
  sorry

end sum_reciprocals_of_roots_l923_92395


namespace sameTypeTerm_l923_92328

variable (a b : ℝ) -- Assume a and b are real numbers 

-- Definitions for each term in the conditions
def term1 : ℝ := 2 * a * b^2
def term2 : ℝ := -a^2 * b
def term3 : ℝ := -2 * a * b
def term4 : ℝ := 5 * a^2

-- The term we are comparing against
def compareTerm : ℝ := 3 * a^2 * b

-- The condition we want to prove
theorem sameTypeTerm : term2 = compareTerm :=
  sorry


end sameTypeTerm_l923_92328


namespace initial_blue_balls_l923_92317

-- Define the initial conditions
variables (B : ℕ) (total_balls : ℕ := 15) (removed_blue_balls : ℕ := 3)
variable (prob_after_removal : ℚ := 1 / 3)
variable (remaining_balls : ℕ := total_balls - removed_blue_balls)
variable (remaining_blue_balls : ℕ := B - removed_blue_balls)

-- State the theorem
theorem initial_blue_balls : 
  remaining_balls = 12 → remaining_blue_balls = remaining_balls * prob_after_removal → B = 7 :=
by
  intros h1 h2
  sorry

end initial_blue_balls_l923_92317


namespace height_of_box_l923_92384

theorem height_of_box (h : ℝ) :
  (∃ (h : ℝ),
    (∀ (x y z : ℝ), (x = 3) ∧ (y = 3) ∧ (z = h / 2) → true) ∧
    (∀ (x y z : ℝ), (x = 1) ∧ (y = 1) ∧ (z = 1) → true) ∧
    h = 6) :=
sorry

end height_of_box_l923_92384


namespace earnings_percentage_difference_l923_92331

-- Defining the conditions
def MikeEarnings : ℕ := 12
def PhilEarnings : ℕ := 6

-- Proving the percentage difference
theorem earnings_percentage_difference :
  ((MikeEarnings - PhilEarnings: ℕ) * 100 / MikeEarnings = 50) :=
by 
  sorry

end earnings_percentage_difference_l923_92331


namespace walters_exceptional_days_l923_92369

variable (b w : ℕ)
variable (days_total dollars_total : ℕ)
variable (normal_earn exceptional_earn : ℕ)
variable (at_least_exceptional_days : ℕ)

-- Conditions
def conditions : Prop :=
  days_total = 15 ∧
  dollars_total = 70 ∧
  normal_earn = 4 ∧
  exceptional_earn = 6 ∧
  at_least_exceptional_days = 5 ∧
  b + w = days_total ∧
  normal_earn * b + exceptional_earn * w = dollars_total ∧
  w ≥ at_least_exceptional_days

-- Theorem to prove the number of exceptional days is 5
theorem walters_exceptional_days (h : conditions b w days_total dollars_total normal_earn exceptional_earn at_least_exceptional_days) : w = 5 :=
sorry

end walters_exceptional_days_l923_92369


namespace sin_135_degree_l923_92306

theorem sin_135_degree : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_135_degree_l923_92306


namespace total_money_difference_l923_92385

-- Define the number of quarters each sibling has
def quarters_Karen : ℕ := 32
def quarters_Christopher : ℕ := 64
def quarters_Emily : ℕ := 20
def quarters_Michael : ℕ := 12

-- Define the value of each quarter
def value_per_quarter : ℚ := 0.25

-- Prove that the total money difference between the pairs of siblings is $16.00
theorem total_money_difference : 
  (quarters_Karen - quarters_Emily) * value_per_quarter + 
  (quarters_Christopher - quarters_Michael) * value_per_quarter = 16 := by
sorry

end total_money_difference_l923_92385


namespace bananas_left_l923_92397

theorem bananas_left (original_bananas : ℕ) (bananas_eaten : ℕ) 
  (h1 : original_bananas = 12) (h2 : bananas_eaten = 4) : 
  original_bananas - bananas_eaten = 8 := 
by
  sorry

end bananas_left_l923_92397


namespace fewest_coach_handshakes_l923_92356

theorem fewest_coach_handshakes (n_A n_B k_A k_B : ℕ) (h1 : n_A = n_B + 2)
    (h2 : ((n_A * (n_A - 1)) / 2) + ((n_B * (n_B - 1)) / 2) + (n_A * n_B) + k_A + k_B = 620) :
  k_A + k_B = 189 := 
sorry

end fewest_coach_handshakes_l923_92356


namespace lines_perpendicular_slope_l923_92344

theorem lines_perpendicular_slope (k : ℝ) :
  (∀ (x : ℝ), k * 2 = -1) → k = (-1:ℝ)/2 :=
by
  sorry

end lines_perpendicular_slope_l923_92344


namespace triangle_inequality_part_a_triangle_inequality_part_b_l923_92354

variable {a b c S : ℝ}

/-- Part (a): Prove that for any triangle ABC, the inequality a^2 + b^2 + c^2 ≥ 4 √3 S holds
    where equality holds if and only if ABC is an equilateral triangle. -/
theorem triangle_inequality_part_a (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (area_S : S > 0) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S :=
sorry

/-- Part (b): Prove that for any triangle ABC,
    the inequality a^2 + b^2 + c^2 - (a - b)^2 - (b - c)^2 - (c - a)^2 ≥ 4 √3 S
    holds where equality also holds if and only if a = b = c. -/
theorem triangle_inequality_part_b (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (area_S : S > 0) :
  a^2 + b^2 + c^2 - (a - b)^2 - (b - c)^2 - (c - a)^2 ≥ 4 * Real.sqrt 3 * S :=
sorry

end triangle_inequality_part_a_triangle_inequality_part_b_l923_92354


namespace unit_vector_perpendicular_l923_92358

theorem unit_vector_perpendicular (x y : ℝ)
  (h1 : 4 * x + 2 * y = 0) 
  (h2 : x^2 + y^2 = 1) :
  (x = (Real.sqrt 5) / 5 ∧ y = -(2 * (Real.sqrt 5) / 5)) ∨ 
  (x = -(Real.sqrt 5) / 5 ∧ y = 2 * (Real.sqrt 5) / 5) :=
sorry

end unit_vector_perpendicular_l923_92358


namespace max_ab_is_nine_l923_92374

noncomputable def f (a b x : ℝ) : ℝ := 4 * x^3 - a * x^2 - 2 * b * x + 2

/-- If a > 0, b > 0, and the function f(x) = 4x^3 - ax^2 - 2bx + 2 has an extremum at x = 1, then the maximum value of ab is 9. -/
theorem max_ab_is_nine {a b : ℝ}
  (ha : a > 0) (hb : b > 0)
  (extremum_x1 : deriv (f a b) 1 = 0) :
  a * b ≤ 9 :=
sorry

end max_ab_is_nine_l923_92374


namespace probability_below_8_l923_92305

def prob_hit_10 := 0.20
def prob_hit_9 := 0.30
def prob_hit_8 := 0.10

theorem probability_below_8 : (1 - (prob_hit_10 + prob_hit_9 + prob_hit_8) = 0.40) :=
by
  sorry

end probability_below_8_l923_92305


namespace total_earnings_correct_l923_92392

-- Define the earnings of Terrence
def TerrenceEarnings : ℕ := 30

-- Define the difference in earnings between Jermaine and Terrence
def JermaineEarningsDifference : ℕ := 5

-- Define the earnings of Jermaine
def JermaineEarnings : ℕ := TerrenceEarnings + JermaineEarningsDifference

-- Define the earnings of Emilee
def EmileeEarnings : ℕ := 25

-- Define the total earnings
def TotalEarnings : ℕ := TerrenceEarnings + JermaineEarnings + EmileeEarnings

theorem total_earnings_correct : TotalEarnings = 90 := by
  sorry

end total_earnings_correct_l923_92392


namespace vinny_fifth_month_loss_l923_92312

theorem vinny_fifth_month_loss (start_weight : ℝ) (end_weight : ℝ) (first_month_loss : ℝ) (second_month_loss : ℝ) (third_month_loss : ℝ) (fourth_month_loss : ℝ) (total_loss : ℝ):
  start_weight = 300 ∧
  first_month_loss = 20 ∧
  second_month_loss = first_month_loss / 2 ∧
  third_month_loss = second_month_loss / 2 ∧
  fourth_month_loss = third_month_loss / 2 ∧
  (start_weight - end_weight) = total_loss ∧
  end_weight = 250.5 →
  (total_loss - (first_month_loss + second_month_loss + third_month_loss + fourth_month_loss)) = 12 :=
by
  sorry

end vinny_fifth_month_loss_l923_92312


namespace solution_set_l923_92307

open Real

noncomputable def condition (x : ℝ) := x ≥ 2

noncomputable def eq_1 (x : ℝ) := sqrt (x + 5 - 6 * sqrt (x - 2)) + sqrt (x + 12 - 8 * sqrt (x - 2)) = 2

theorem solution_set :
  {x : ℝ | condition x ∧ eq_1 x} = {x : ℝ | 11 ≤ x ∧ x ≤ 18} :=
by sorry

end solution_set_l923_92307


namespace find_n_from_exponent_equation_l923_92363

theorem find_n_from_exponent_equation (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by
  sorry

end find_n_from_exponent_equation_l923_92363


namespace sum_of_possible_values_of_x_l923_92351

theorem sum_of_possible_values_of_x :
  let sq_side := (x - 4)
  let rect_length := (x - 5)
  let rect_width := (x + 6)
  let sq_area := (sq_side)^2
  let rect_area := rect_length * rect_width
  (3 * (sq_area) = rect_area) → ∃ (x1 x2 : ℝ), (3 * (x1 - 4) ^ 2 = (x1 - 5) * (x1 + 6)) ∧ (3 * (x2 - 4) ^ 2 = (x2 - 5) * (x2 + 6)) ∧ (x1 + x2 = 12.5) := 
by
  sorry

end sum_of_possible_values_of_x_l923_92351


namespace arith_seq_a15_l923_92345

variable {α : Type} [LinearOrderedField α]

def is_arith_seq (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arith_seq_a15 (a : ℕ → α) (k l m : ℕ) (x y : α) 
  (h_seq : is_arith_seq a)
  (h_k : a k = x)
  (h_l : a l = y) :
  a (l + (l - k)) = 2 * y - x := 
  sorry

end arith_seq_a15_l923_92345


namespace converse_xy_implies_x_is_true_l923_92367

/-- Prove that the converse of the proposition "If \(xy = 0\), then \(x = 0\)" is true. -/
theorem converse_xy_implies_x_is_true {x y : ℝ} (h : x = 0) : x * y = 0 :=
by sorry

end converse_xy_implies_x_is_true_l923_92367


namespace rectangular_garden_length_l923_92314

theorem rectangular_garden_length (P B L : ℕ) (h1 : P = 1800) (h2 : B = 400) (h3 : P = 2 * (L + B)) : L = 500 :=
sorry

end rectangular_garden_length_l923_92314


namespace insurance_not_covered_percentage_l923_92349

noncomputable def insurance_monthly_cost : ℝ := 20
noncomputable def insurance_months : ℝ := 24
noncomputable def procedure_cost : ℝ := 5000
noncomputable def amount_saved : ℝ := 3520

theorem insurance_not_covered_percentage :
  ((procedure_cost - amount_saved - (insurance_monthly_cost * insurance_months)) / procedure_cost) * 100 = 20 :=
by
  sorry

end insurance_not_covered_percentage_l923_92349


namespace problem_conditions_l923_92378

theorem problem_conditions (m : ℝ) (hf_pow : m^2 - m - 1 = 1) (hf_inc : m > 0) : m = 2 :=
sorry

end problem_conditions_l923_92378


namespace impossible_sequence_l923_92390

theorem impossible_sequence (a : ℕ → ℝ) (c : ℝ) (a1 : ℝ)
  (h_periodic : ∀ n, a (n + 3) = a n)
  (h_det : ∀ n, a n * a (n + 3) - a (n + 1) * a (n + 2) = c)
  (ha1 : a 1 = 2) (hc : c = 2) : false :=
by
  sorry

end impossible_sequence_l923_92390


namespace equality_of_costs_l923_92335

theorem equality_of_costs (x : ℕ) :
  (800 + 30 * x = 500 + 35 * x) ↔ x = 60 := by
  sorry

end equality_of_costs_l923_92335


namespace smallest_positive_divisible_by_111_has_last_digits_2004_l923_92381

theorem smallest_positive_divisible_by_111_has_last_digits_2004 :
  ∃ (X : ℕ), (∃ (A : ℕ), X = A * 10^4 + 2004) ∧ 111 ∣ X ∧ X = 662004 := by
  sorry

end smallest_positive_divisible_by_111_has_last_digits_2004_l923_92381


namespace audrey_peaches_l923_92371

variable (A : ℕ)
variable (P : ℕ := 48)
variable (D : ℕ := 22)

theorem audrey_peaches : A - P = D → A = 70 :=
by
  intro h
  sorry

end audrey_peaches_l923_92371


namespace corridor_length_correct_l923_92339

/-- Scale representation in the blueprint: 1 cm represents 10 meters. --/
def scale_cm_to_m (cm: ℝ): ℝ := cm * 10

/-- Length of the corridor in the blueprint. --/
def blueprint_length_cm: ℝ := 9.5

/-- Real-life length of the corridor. --/
def real_life_length: ℝ := 95

/-- Proof that the real-life length of the corridor is correctly calculated. --/
theorem corridor_length_correct :
  scale_cm_to_m blueprint_length_cm = real_life_length :=
by
  sorry

end corridor_length_correct_l923_92339


namespace scoops_of_natural_seedless_raisins_l923_92338

theorem scoops_of_natural_seedless_raisins 
  (cost_natural : ℝ := 3.45) 
  (cost_golden : ℝ := 2.55) 
  (num_golden : ℝ := 20) 
  (cost_mixture : ℝ := 3) : 
  ∃ x : ℝ, (3.45 * x + 20 * 2.55 = 3 * (x + 20)) ∧ x = 20 :=
sorry

end scoops_of_natural_seedless_raisins_l923_92338


namespace same_color_probability_l923_92315

def sides := 12
def violet_sides := 3
def orange_sides := 4
def lime_sides := 5

def prob_violet := violet_sides / sides
def prob_orange := orange_sides / sides
def prob_lime := lime_sides / sides

theorem same_color_probability :
  (prob_violet * prob_violet) + (prob_orange * prob_orange) + (prob_lime * prob_lime) = 25 / 72 :=
by
  sorry

end same_color_probability_l923_92315


namespace tablets_of_medicine_A_l923_92386

-- Given conditions as definitions
def B_tablets : ℕ := 16

def min_extracted_tablets : ℕ := 18

-- Question and expected answer encapsulated in proof statement
theorem tablets_of_medicine_A (A_tablets : ℕ) (h : A_tablets + B_tablets - 2 >= min_extracted_tablets) : A_tablets = 3 :=
sorry

end tablets_of_medicine_A_l923_92386


namespace maximize_daily_profit_l923_92342

noncomputable def daily_profit : ℝ → ℝ → ℝ
| x, c => if h : 0 < x ∧ x ≤ c then (3 * (9 * x - 2 * x^2)) / (2 * (6 - x)) else 0

theorem maximize_daily_profit (c : ℝ) (x : ℝ) (h1 : 0 < c) (h2 : c < 6) :
  (y = daily_profit x c) ∧
  (if 0 < c ∧ c < 3 then x = c else if 3 ≤ c ∧ c < 6 then x = 3 else False) :=
by
  sorry

end maximize_daily_profit_l923_92342


namespace fixed_point_of_line_l923_92308

theorem fixed_point_of_line (m : ℝ) : 
  (m - 1) * (7 / 2) - (m + 3) * (5 / 2) - (m - 11) = 0 :=
by
  sorry

end fixed_point_of_line_l923_92308


namespace unique_function_satisfying_conditions_l923_92383

theorem unique_function_satisfying_conditions :
  ∀ f : ℚ → ℚ, (f 1 = 2) → (∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) → (∀ x : ℚ, f x = x + 1) :=
by
  intro f h1 hCond
  sorry

end unique_function_satisfying_conditions_l923_92383


namespace inequality_proof_l923_92327

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ( (2 + x) / (1 + x) )^2 + ( (2 + y) / (1 + y) )^2 ≥ 9 / 2 := 
by 
  sorry

end inequality_proof_l923_92327


namespace sector_area_l923_92319

noncomputable def area_of_sector (r : ℝ) (theta : ℝ) : ℝ :=
  1 / 2 * r * r * theta

theorem sector_area (r : ℝ) (theta : ℝ) (h_r : r = Real.pi) (h_theta : theta = 2 * Real.pi / 3) :
  area_of_sector r theta = Real.pi^3 / 6 :=
by
  sorry

end sector_area_l923_92319
