import Mathlib

namespace C3PO_Optimal_Play_Wins_l86_8690

def initial_number : List ℕ := [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]

-- Conditions for the game
structure GameConditions where
  number : List ℕ
  robots : List String
  cannot_swap : List (ℕ × ℕ) -- Pair of digits that cannot be swapped again
  cannot_start_with_zero : Bool
  c3po_starts : Bool

-- Define the initial conditions
def initial_conditions : GameConditions :=
{
  number := initial_number,
  robots := ["C3PO", "R2D2"],
  cannot_swap := [],
  cannot_start_with_zero := true,
  c3po_starts := true
}

-- Define the winning condition for C3PO
def C3PO_wins : Prop :=
  ∀ game : GameConditions, game = initial_conditions → ∃ is_c3po_winner : Bool, is_c3po_winner = true

-- The theorem statement
theorem C3PO_Optimal_Play_Wins : C3PO_wins :=
by
  sorry

end C3PO_Optimal_Play_Wins_l86_8690


namespace find_other_person_weight_l86_8674

theorem find_other_person_weight
    (initial_avg_weight : ℕ)
    (final_avg_weight : ℕ)
    (initial_group_size : ℕ)
    (new_person_weight : ℕ)
    (final_group_size : ℕ)
    (initial_total_weight : ℕ)
    (final_total_weight : ℕ)
    (new_total_weight : ℕ)
    (other_person_weight : ℕ) :
  initial_avg_weight = 48 →
  final_avg_weight = 51 →
  initial_group_size = 23 →
  final_group_size = 25 →
  new_person_weight = 93 →
  initial_total_weight = initial_group_size * initial_avg_weight →
  final_total_weight = final_group_size * final_avg_weight →
  new_total_weight = initial_total_weight + new_person_weight + other_person_weight →
  final_total_weight = new_total_weight →
  other_person_weight = 78 :=
by
  sorry

end find_other_person_weight_l86_8674


namespace greatest_decimal_is_7391_l86_8642

noncomputable def decimal_conversion (n d : ℕ) : ℝ :=
  n / d

noncomputable def forty_two_percent_of (r : ℝ) : ℝ :=
  0.42 * r

theorem greatest_decimal_is_7391 :
  let a := forty_two_percent_of (decimal_conversion 7 11)
  let b := decimal_conversion 17 23
  let c := 0.7391
  let d := decimal_conversion 29 47
  a < b ∧ a < c ∧ a < d ∧ b = c ∧ d < b :=
by
  have dec1 := forty_two_percent_of (decimal_conversion 7 11)
  have dec2 := decimal_conversion 17 23
  have dec3 := 0.7391
  have dec4 := decimal_conversion 29 47
  sorry

end greatest_decimal_is_7391_l86_8642


namespace cricket_team_rh_players_l86_8603

theorem cricket_team_rh_players (total_players throwers non_throwers lh_non_throwers rh_non_throwers rh_players : ℕ)
    (h1 : total_players = 58)
    (h2 : throwers = 37)
    (h3 : non_throwers = total_players - throwers)
    (h4 : lh_non_throwers = non_throwers / 3)
    (h5 : rh_non_throwers = non_throwers - lh_non_throwers)
    (h6 : rh_players = throwers + rh_non_throwers) :
  rh_players = 51 := by
  sorry

end cricket_team_rh_players_l86_8603


namespace smaller_of_two_digit_product_l86_8622

theorem smaller_of_two_digit_product (a b : ℕ) (ha : 10 ≤ a) (hb : 10 ≤ b) (ha' : a < 100) (hb' : b < 100) 
  (hprod : a * b = 4680) : min a b = 52 :=
by
  sorry

end smaller_of_two_digit_product_l86_8622


namespace difference_of_squares_divisible_by_9_l86_8625

theorem difference_of_squares_divisible_by_9 (a b : ℤ) : 
  9 ∣ ((3 * a + 2)^2 - (3 * b + 2)^2) :=
by
  sorry

end difference_of_squares_divisible_by_9_l86_8625


namespace proof_problem_l86_8615

-- Define the problem space
variables (x y : ℝ)

-- Define the conditions
def satisfies_condition (x y : ℝ) : Prop :=
  (0 < x) ∧ (0 < y) ∧ (4 * Real.log x + 2 * Real.log (2 * y) ≥ x^2 + 8 * y - 4)

-- The theorem statement
theorem proof_problem (hx : 0 < x) (hy : 0 < y) (hcond : satisfies_condition x y) :
  x + 2 * y = 1/2 + Real.sqrt 2 :=
sorry

end proof_problem_l86_8615


namespace find_second_number_l86_8639

def average (nums : List ℕ) : ℕ :=
  nums.sum / nums.length

theorem find_second_number (nums : List ℕ) (a b : ℕ) (avg : ℕ) :
  average [10, 70, 28] = 36 ∧ average (10 :: 70 :: 28 :: []) + 4 = avg ∧ average (a :: b :: nums) = avg ∧ a = 20 ∧ b = 60 → b = 60 :=
by
  sorry

end find_second_number_l86_8639


namespace range_m_l86_8649

namespace MathProof

noncomputable def f (x m : ℝ) : ℝ := x^3 - 3 * x + 2 + m

theorem range_m
  (m : ℝ)
  (h : m > 0)
  (a b c : ℝ)
  (ha : 0 ≤ a ∧ a ≤ 2)
  (hb : 0 ≤ b ∧ b ≤ 2)
  (hc : 0 ≤ c ∧ c ≤ 2)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (h_triangle : f a m ^ 2 + f b m ^ 2 = f c m ^ 2 ∨
                f a m ^ 2 + f c m ^ 2 = f b m ^ 2 ∨
                f b m ^ 2 + f c m ^ 2 = f a m ^ 2) :
  0 < m ∧ m < 3 + 4 * Real.sqrt 2 :=
by
  sorry

end MathProof

end range_m_l86_8649


namespace common_chord_of_circles_l86_8653

theorem common_chord_of_circles :
  ∀ (x y : ℝ), (x^2 + y^2 - 4 * x = 0) ∧ (x^2 + y^2 - 4 * y = 0) → (x = y) :=
by
  intros x y h
  sorry

end common_chord_of_circles_l86_8653


namespace cos_B_arithmetic_sequence_sin_A_sin_C_geometric_sequence_l86_8635

theorem cos_B_arithmetic_sequence (A B C : ℝ) (h1 : 2 * B = A + C) (h2 : A + B + C = 180) :
  Real.cos B = 1 / 2 :=
by
  sorry

theorem sin_A_sin_C_geometric_sequence (A B C a b c : ℝ) (h1 : 2 * B = A + C) (h2 : A + B + C = 180)
  (h3 : b^2 = a * c) (h4 : b^2 = a^2 + c^2 - 2 * a * c * Real.cos B) :
  Real.sin A * Real.sin C = 3 / 4 :=
by
  sorry

end cos_B_arithmetic_sequence_sin_A_sin_C_geometric_sequence_l86_8635


namespace segments_form_pentagon_l86_8699

theorem segments_form_pentagon (a b c d e : ℝ) 
  (h_sum : a + b + c + d + e = 2)
  (h_a : a > 1/10)
  (h_b : b > 1/10)
  (h_c : c > 1/10)
  (h_d : d > 1/10)
  (h_e : e > 1/10) :
  a + b + c + d > e ∧ a + b + c + e > d ∧ a + b + d + e > c ∧ a + c + d + e > b ∧ b + c + d + e > a := 
sorry

end segments_form_pentagon_l86_8699


namespace v_function_expression_f_max_value_l86_8604

noncomputable def v (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 4 then 2
else if 4 < x ∧ x ≤ 20 then - (1/8) * x + (5/2)
else 0

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 4 then 2 * x
else if 4 < x ∧ x ≤ 20 then - (1/8) * x^2 + (5/2) * x
else 0

theorem v_function_expression :
  ∀ x, 0 < x ∧ x ≤ 20 → 
  v x = (if 0 < x ∧ x ≤ 4 then 2 else if 4 < x ∧ x ≤ 20 then - (1/8) * x + (5/2) else 0) :=
by sorry

theorem f_max_value :
  ∃ x, 0 < x ∧ x ≤ 20 ∧ f x = 12.5 :=
by sorry

end v_function_expression_f_max_value_l86_8604


namespace greatest_whole_number_solution_l86_8614

theorem greatest_whole_number_solution (x : ℤ) (h : 6 * x - 5 < 7 - 3 * x) : x ≤ 1 :=
sorry

end greatest_whole_number_solution_l86_8614


namespace rhombus_diagonal_length_l86_8634

theorem rhombus_diagonal_length (d1 d2 : ℝ) (Area : ℝ) 
  (h1 : d1 = 12) (h2 : Area = 60) 
  (h3 : Area = (d1 * d2) / 2) : d2 = 10 := 
by
  sorry

end rhombus_diagonal_length_l86_8634


namespace find_sum_l86_8657

variable (a b c d : ℝ)

theorem find_sum (h1 : a * b + b * c + c * d + d * a = 48) (h2 : b + d = 6) : a + c = 8 :=
sorry

end find_sum_l86_8657


namespace ratio_of_height_to_radius_min_surface_area_l86_8617

theorem ratio_of_height_to_radius_min_surface_area 
  (r h : ℝ)
  (V : ℝ := 500)
  (volume_cond : π * r^2 * h = V)
  (surface_area : ℝ := 2 * π * r^2 + 2 * π * r * h) : 
  h / r = 2 :=
by
  sorry

end ratio_of_height_to_radius_min_surface_area_l86_8617


namespace daily_salmon_l86_8663

-- Definitions of the daily consumption of trout and total fish
def daily_trout : ℝ := 0.2
def daily_total_fish : ℝ := 0.6

-- Theorem statement that the daily consumption of salmon is 0.4 buckets
theorem daily_salmon : daily_total_fish - daily_trout = 0.4 := 
by
  -- Skipping the proof, as required
  sorry

end daily_salmon_l86_8663


namespace train_length_is_correct_l86_8618

noncomputable def length_of_train (speed_train_kmh : ℝ) (speed_man_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let relative_speed_kmh := speed_train_kmh + speed_man_kmh
  let relative_speed_ms := relative_speed_kmh * (5/18)
  let length := relative_speed_ms * time_s
  length

theorem train_length_is_correct (h1 : 84 = 84) (h2 : 6 = 6) (h3 : 4.399648028157747 = 4.399648028157747) :
  length_of_train 84 6 4.399648028157747 = 110.991201 := by
  dsimp [length_of_train]
  norm_num
  sorry

end train_length_is_correct_l86_8618


namespace pure_imaginary_complex_number_l86_8692

theorem pure_imaginary_complex_number (m : ℝ) (h : (m^2 - 3*m) = 0) :
  (m^2 - 5*m + 6) ≠ 0 → m = 0 :=
by
  intro h_im
  have h_fact : (m = 0) ∨ (m = 3) := by
    sorry -- This is where the factorization steps would go
  cases h_fact with
  | inl h0 =>
    assumption
  | inr h3 =>
    exfalso
    have : (3^2 - 5*3 + 6) = 0 := by
      sorry -- Simplify to check that m = 3 is not a valid solution
    contradiction

end pure_imaginary_complex_number_l86_8692


namespace number_of_knights_l86_8658

def traveler := Type
def is_knight (t : traveler) : Prop := sorry
def is_liar (t : traveler) : Prop := sorry

axiom total_travelers : Finset traveler
axiom vasily : traveler
axiom  h_total : total_travelers.card = 16

axiom kn_lie (t : traveler) : is_knight t ∨ is_liar t

axiom vasily_liar : is_liar vasily
axiom contradictory_statements_in_room (rooms: Finset (Finset traveler)):
  (∀ room ∈ rooms, ∃ t ∈ room, (is_liar t ∧ is_knight t))
  ∧
  (∀ room ∈ rooms, ∃ t ∈ room, (is_knight t ∧ is_liar t))

theorem number_of_knights : 
  ∃ k, k = 9 ∧ (∃ l, l = 7 ∧ ∀ t ∈ total_travelers, (is_knight t ∨ is_liar t)) :=
sorry

end number_of_knights_l86_8658


namespace price_reduction_percentage_price_increase_amount_l86_8696

theorem price_reduction_percentage (x : ℝ) (hx : 50 * (1 - x)^2 = 32) : x = 0.2 := 
sorry

theorem price_increase_amount (y : ℝ) 
  (hy1 : 0 < y ∧ y ≤ 8) 
  (hy2 : 6000 = (10 + y) * (500 - 20 * y)) : y = 5 := 
sorry

end price_reduction_percentage_price_increase_amount_l86_8696


namespace find_a3_l86_8666

theorem find_a3 (a : ℝ) (a0 a1 a2 a3 a4 a5 a6 a7 : ℝ) 
    (h1 : (1 + x) * (a - x)^6 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7)
    (h2 : a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 = 0) :
  a = 1 → a3 = -5 := 
by 
  sorry

end find_a3_l86_8666


namespace opposite_numbers_A_l86_8664

theorem opposite_numbers_A :
  let A1 := -((-1)^2)
  let A2 := abs (-1)

  let B1 := (-2)^3
  let B2 := -(2^3)
  
  let C1 := 2
  let C2 := 1 / 2
  
  let D1 := -(-1)
  let D2 := 1
  
  (A1 = -A2 ∧ A2 = 1) ∧ ¬(B1 = -B2) ∧ ¬(C1 = -C2) ∧ ¬(D1 = -D2)
:= by
  let A1 := -((-1)^2)
  let A2 := abs (-1)

  let B1 := (-2)^3
  let B2 := -(2^3)
  
  let C1 := 2
  let C2 := 1 / 2
  
  let D1 := -(-1)
  let D2 := 1

  sorry

end opposite_numbers_A_l86_8664


namespace largest_num_pencils_in_package_l86_8651

theorem largest_num_pencils_in_package (Ming_pencils Catherine_pencils : ℕ) 
  (Ming_pencils := 40) 
  (Catherine_pencils := 24) 
  (H : ∃ k, Ming_pencils = k * a ∧ Catherine_pencils = k * b) :
  gcd Ming_pencils Catherine_pencils = 8 :=
by
  sorry

end largest_num_pencils_in_package_l86_8651


namespace distance_between_consecutive_trees_l86_8623

theorem distance_between_consecutive_trees 
  (yard_length : ℕ) (num_trees : ℕ) (tree_at_each_end : yard_length > 0 ∧ num_trees ≥ 2) 
  (equal_distances : ∀ k, k < num_trees - 1 → (yard_length / (num_trees - 1) : ℝ) = 12) :
  yard_length = 360 → num_trees = 31 → (yard_length / (num_trees - 1) : ℝ) = 12 := 
by
  sorry

end distance_between_consecutive_trees_l86_8623


namespace johns_weight_l86_8662

theorem johns_weight (j m : ℝ) (h1 : j + m = 240) (h2 : j - m = j / 3) : j = 144 :=
by
  sorry

end johns_weight_l86_8662


namespace eccentricity_ratio_l86_8624

noncomputable def ellipse_eccentricity (m n : ℝ) : ℝ := (1 - (1 / n) / (1 / m))^(1/2)

theorem eccentricity_ratio (m n : ℝ) (h : ellipse_eccentricity m n = 1 / 2) :
  m / n = 3 / 4 :=
by
  sorry

end eccentricity_ratio_l86_8624


namespace students_all_three_classes_l86_8611

variables (H M E HM HE ME HME : ℕ)

-- Conditions from the problem
def student_distribution : Prop :=
  H = 12 ∧
  M = 17 ∧
  E = 36 ∧
  HM + HE + ME = 3 ∧
  86 = H + M + E - (HM + HE + ME) + HME

-- Prove the number of students registered for all three classes
theorem students_all_three_classes (h : student_distribution H M E HM HE ME HME) : HME = 24 :=
  by sorry

end students_all_three_classes_l86_8611


namespace sqrt_of_sixteen_l86_8655

theorem sqrt_of_sixteen : ∃ x : ℤ, x^2 = 16 ∧ (x = 4 ∨ x = -4) := by
  sorry

end sqrt_of_sixteen_l86_8655


namespace min_L_pieces_correct_l86_8689

noncomputable def min_L_pieces : ℕ :=
  have pieces : Nat := 11
  pieces

theorem min_L_pieces_correct :
  min_L_pieces = 11 := 
by
  sorry

end min_L_pieces_correct_l86_8689


namespace negative_number_unique_l86_8638

theorem negative_number_unique (a b c d : ℚ) (h₁ : a = 1) (h₂ : b = 0) (h₃ : c = 1/2) (h₄ : d = -2) :
  ∃! x : ℚ, x < 0 ∧ (x = a ∨ x = b ∨ x = c ∨ x = d) :=
by 
  sorry

end negative_number_unique_l86_8638


namespace Dima_broke_more_l86_8640

theorem Dima_broke_more (D F : ℕ) (h : 2 * D + 7 * F = 3 * (D + F)) : D = 4 * F :=
sorry

end Dima_broke_more_l86_8640


namespace triangle_PQ_length_l86_8677

theorem triangle_PQ_length (RP PQ : ℝ) (n : ℕ) (h_rp : RP = 2.4) (h_n : n = 25) : RP = 2.4 → PQ = 3 := by
  sorry

end triangle_PQ_length_l86_8677


namespace log_product_evaluation_l86_8654

noncomputable def evaluate_log_product : ℝ :=
  Real.log 9 / Real.log 2 * Real.log 16 / Real.log 3 * Real.log 27 / Real.log 7

theorem log_product_evaluation : evaluate_log_product = 24 := 
  sorry

end log_product_evaluation_l86_8654


namespace angle_C_measurement_l86_8641

variables (A B C : ℝ)

theorem angle_C_measurement
  (h1 : A + C = 2 * B)
  (h2 : C - A = 80)
  (h3 : A + B + C = 180) :
  C = 100 :=
by sorry

end angle_C_measurement_l86_8641


namespace usual_time_to_cover_distance_l86_8693

theorem usual_time_to_cover_distance (S T : ℝ) (h1 : 0.75 * S = S / (T + 24)) (h2 : S * T = 0.75 * S * (T + 24)) : T = 72 :=
by
  sorry

end usual_time_to_cover_distance_l86_8693


namespace expression_divisible_by_9_for_any_int_l86_8667

theorem expression_divisible_by_9_for_any_int (a b : ℤ) : 9 ∣ ((3 * a + 2)^2 - (3 * b + 2)^2) := 
by 
  sorry

end expression_divisible_by_9_for_any_int_l86_8667


namespace sum_of_first_two_digits_of_repeating_decimal_l86_8632

theorem sum_of_first_two_digits_of_repeating_decimal (c d : ℕ) (h : (c, d) = (3, 5)) : c + d = 8 :=
by 
  sorry

end sum_of_first_two_digits_of_repeating_decimal_l86_8632


namespace remainder_of_p_div_x_minus_3_l86_8656

def p (x : ℝ) : ℝ := x^4 - x^3 - 4 * x + 7

theorem remainder_of_p_div_x_minus_3 : 
  let remainder := p 3 
  remainder = 49 := 
by
  sorry

end remainder_of_p_div_x_minus_3_l86_8656


namespace landscape_length_l86_8600

theorem landscape_length (b : ℝ) 
  (h1 : ∀ (l : ℝ), l = 8 * b) 
  (A : ℝ)
  (h2 : A = 8 * b^2)
  (Playground_area : ℝ)
  (h3 : Playground_area = 1200)
  (h4 : Playground_area = (1 / 6) * A) :
  ∃ (l : ℝ), l = 240 :=
by 
  sorry

end landscape_length_l86_8600


namespace evaluate_absolute_value_l86_8670

theorem evaluate_absolute_value (π : ℝ) (h : π < 5.5) : |5.5 - π| = 5.5 - π :=
by
  sorry

end evaluate_absolute_value_l86_8670


namespace arrangeable_sequence_l86_8616

theorem arrangeable_sequence (n : Fin 2017 → ℤ) :
  (∀ i : Fin 2017, ∃ (perm : Fin 5 → Fin 5),
    let a := n ((i + perm 0) % 2017)
    let b := n ((i + perm 1) % 2017)
    let c := n ((i + perm 2) % 2017)
    let d := n ((i + perm 3) % 2017)
    let e := n ((i + perm 4) % 2017)
    a - b + c - d + e = 29) →
  (∀ i : Fin 2017, n i = 29) :=
by
  sorry

end arrangeable_sequence_l86_8616


namespace range_of_m_l86_8646

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem range_of_m (m : ℝ) (h : ∀ x > 0, f x > m * x) : m ≤ 2 := sorry

end range_of_m_l86_8646


namespace measure_of_angle_C_l86_8607

variable (A B C : Real)

theorem measure_of_angle_C (h1 : 4 * Real.sin A + 2 * Real.cos B = 4) 
                           (h2 : (1/2) * Real.sin B + Real.cos A = Real.sqrt 3 / 2) :
                           C = Real.pi / 6 :=
by
  sorry

end measure_of_angle_C_l86_8607


namespace valid_set_example_l86_8637

def is_valid_set (S : Set ℝ) : Prop :=
  ∀ x ∈ S, ∃ y ∈ S, x ≠ y

theorem valid_set_example : is_valid_set { x : ℝ | x > Real.sqrt 2 } :=
sorry

end valid_set_example_l86_8637


namespace tv_price_reduction_percentage_l86_8650

noncomputable def price_reduction (x : ℝ) : Prop :=
  (1 - x / 100) * 1.80 = 1.44000000000000014

theorem tv_price_reduction_percentage : price_reduction 20 :=
by
  sorry

end tv_price_reduction_percentage_l86_8650


namespace circles_intersect_l86_8633

variable (r1 r2 d : ℝ)
variable (h1 : r1 = 4)
variable (h2 : r2 = 5)
variable (h3 : d = 7)

theorem circles_intersect : 1 < d ∧ d < r1 + r2 :=
by sorry

end circles_intersect_l86_8633


namespace combined_mpg_19_l86_8602

theorem combined_mpg_19 (m: ℕ) (h: m = 100) :
  let ray_car_mpg := 50
  let tom_car_mpg := 25
  let jerry_car_mpg := 10
  let ray_gas_used := m / ray_car_mpg
  let tom_gas_used := m / tom_car_mpg
  let jerry_gas_used := m / jerry_car_mpg
  let total_gas_used := ray_gas_used + tom_gas_used + jerry_gas_used
  let total_miles := 3 * m
  let combined_mpg := total_miles * 25 / (4 * m)
  combined_mpg = 19 := 
by {
  sorry
}

end combined_mpg_19_l86_8602


namespace perfect_square_n_l86_8680

theorem perfect_square_n (n : ℕ) (hn_pos : n > 0) :
  (∃ (m : ℕ), m * m = (n^2 + 11 * n - 4) * n.factorial + 33 * 13^n + 4) ↔ n = 1 ∨ n = 2 :=
by sorry

end perfect_square_n_l86_8680


namespace ordered_pairs_count_l86_8672

theorem ordered_pairs_count : 
  (∀ (b c : ℕ), b > 0 ∧ b ≤ 6 ∧ c > 0 ∧ c ≤ 6 ∧ b^2 - 4 * c < 0 ∧ c^2 - 4 * b < 0 → 
  ((b = 1 ∧ (c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6)) ∨ 
  (b = 2 ∧ (c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6)) ∨ 
  (b = 3 ∧ (c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6)) ∨ 
  (b = 4 ∧ (c = 5 ∨ c = 6)))) ∧
  (∃ (n : ℕ), n = 15) := sorry

end ordered_pairs_count_l86_8672


namespace oranges_equivalency_l86_8683

theorem oranges_equivalency :
  ∀ (w_orange w_apple w_pear : ℕ), 
  (9 * w_orange = 6 * w_apple + w_pear) →
  (36 * w_orange = 24 * w_apple + 4 * w_pear) :=
by
  -- The proof will go here; for now, we'll use sorry to skip it
  sorry

end oranges_equivalency_l86_8683


namespace find_t_l86_8695

theorem find_t (t : ℕ) : 
  t > 3 ∧ (3 * t - 10) * (4 * t - 9) = (t + 12) * (2 * t + 1) → t = 6 := 
by
  intro h
  have h1 : t > 3 := h.1
  have h2 : (3 * t - 10) * (4 * t - 9) = (t + 12) * (2 * t + 1) := h.2
  sorry

end find_t_l86_8695


namespace translation_symmetric_graphs_l86_8684

/-- The graph of the function f(x)=sin(x/π + φ) is translated to the right by θ (θ>0) units to obtain the graph of the function g(x).
    On the graph of f(x), point A is translated to point B, let x_A and x_B be the abscissas of points A and B respectively.
    If the axes of symmetry of the graphs of f(x) and g(x) coincide, then the real values that can be taken as x_A - x_B are -2π² or -π². -/
theorem translation_symmetric_graphs (θ : ℝ) (hθ : θ > 0) (x_A x_B : ℝ) (φ : ℝ) :
  ((x_A - x_B = -2 * π^2) ∨ (x_A - x_B = -π^2)) :=
sorry

end translation_symmetric_graphs_l86_8684


namespace DavidCrunchesLessThanZachary_l86_8610

-- Definitions based on conditions
def ZacharyPushUps : ℕ := 44
def ZacharyCrunches : ℕ := 17
def DavidPushUps : ℕ := ZacharyPushUps + 29
def DavidCrunches : ℕ := 4

-- Problem statement we need to prove:
theorem DavidCrunchesLessThanZachary : DavidCrunches = ZacharyCrunches - 13 :=
by
  -- Proof will go here
  sorry

end DavidCrunchesLessThanZachary_l86_8610


namespace front_wheel_revolutions_l86_8688

theorem front_wheel_revolutions (P_front P_back : ℕ) (R_back : ℕ) (H1 : P_front = 30) (H2 : P_back = 20) (H3 : R_back = 360) :
  ∃ F : ℕ, F = 240 := by
  sorry

end front_wheel_revolutions_l86_8688


namespace race_head_start_l86_8652

theorem race_head_start
  (v_A v_B L x : ℝ)
  (h1 : v_A = (4 / 3) * v_B)
  (h2 : L / v_A = (L - x * L) / v_B) :
  x = 1 / 4 :=
sorry

end race_head_start_l86_8652


namespace books_per_week_l86_8673

-- Define the conditions
def total_books_read : ℕ := 20
def weeks : ℕ := 5

-- Define the statement to be proved
theorem books_per_week : (total_books_read / weeks) = 4 := by
  -- Proof omitted
  sorry

end books_per_week_l86_8673


namespace original_price_l86_8686

noncomputable def original_selling_price (CP : ℝ) : ℝ := CP * 1.25
noncomputable def selling_price_at_loss (CP : ℝ) : ℝ := CP * 0.5

theorem original_price (CP : ℝ) (h : selling_price_at_loss CP = 320) : original_selling_price CP = 800 :=
by
  sorry

end original_price_l86_8686


namespace bananas_left_correct_l86_8619

def initial_bananas : ℕ := 12
def eaten_bananas : ℕ := 1
def bananas_left (initial eaten : ℕ) := initial - eaten

theorem bananas_left_correct : bananas_left initial_bananas eaten_bananas = 11 :=
by
  sorry

end bananas_left_correct_l86_8619


namespace cat_clothing_probability_l86_8668

-- Define the conditions as Lean definitions
def n_items : ℕ := 3
def total_legs : ℕ := 4
def favorable_outcomes_per_leg : ℕ := 1
def possible_outcomes_per_leg : ℕ := (n_items.factorial : ℕ)
def probability_per_leg : ℚ := favorable_outcomes_per_leg / possible_outcomes_per_leg

-- Theorem statement to show the combined probability for all legs
theorem cat_clothing_probability
    (n_items_eq : n_items = 3)
    (total_legs_eq : total_legs = 4)
    (fact_n_items : (n_items.factorial) = 6)
    (prob_leg_eq : probability_per_leg = 1 / 6) :
    (probability_per_leg ^ total_legs = 1 / 1296) := by
    sorry

end cat_clothing_probability_l86_8668


namespace number_is_40_l86_8631

theorem number_is_40 (N : ℝ) (h : N = (3/8) * N + (1/4) * N + 15) : N = 40 :=
by
  sorry

end number_is_40_l86_8631


namespace jovana_bucket_shells_l86_8691

theorem jovana_bucket_shells :
  let a0 := 5.2
  let a1 := a0 + 15.7
  let a2 := a1 + 17.5
  let a3 := a2 - 4.3
  let a4 := 3 * a3
  a4 = 102.3 := 
by
  sorry

end jovana_bucket_shells_l86_8691


namespace length_of_train_is_125_l86_8661

noncomputable def speed_kmph : ℝ := 90
noncomputable def time_sec : ℝ := 5
noncomputable def speed_mps : ℝ := speed_kmph * (1000 / 3600)
noncomputable def length_train : ℝ := speed_mps * time_sec

theorem length_of_train_is_125 :
  length_train = 125 := 
by
  sorry

end length_of_train_is_125_l86_8661


namespace actual_number_of_children_l86_8681

-- Define the conditions of the problem
def condition1 (C B : ℕ) : Prop := B = 2 * C
def condition2 : ℕ := 320
def condition3 (C B : ℕ) : Prop := B = 4 * (C - condition2)

-- Define the statement to be proved
theorem actual_number_of_children (C B : ℕ) 
  (h1 : condition1 C B) (h2 : condition3 C B) : C = 640 :=
by 
  -- Proof will be added here
  sorry

end actual_number_of_children_l86_8681


namespace ben_and_sue_answer_l86_8613

theorem ben_and_sue_answer :
  let x := 8
  let y := 3 * (x + 2)
  let z := 3 * (y - 2)
  z = 84
:= by
  let x := 8
  let y := 3 * (x + 2)
  let z := 3 * (y - 2)
  show z = 84
  sorry

end ben_and_sue_answer_l86_8613


namespace female_democrats_l86_8608

theorem female_democrats (F M : ℕ) 
    (h₁ : F + M = 990)
    (h₂ : F / 2 + M / 4 = 330) : F / 2 = 275 := 
by sorry

end female_democrats_l86_8608


namespace complete_square_example_l86_8645

theorem complete_square_example :
  ∃ c : ℝ, ∃ d : ℝ, (∀ x : ℝ, x^2 + 12 * x + 4 = (x + c)^2 - d) ∧ d = 32 := by
  sorry

end complete_square_example_l86_8645


namespace total_acres_cleaned_l86_8687

theorem total_acres_cleaned (A D : ℕ) (h1 : (D - 1) * 90 + 30 = A) (h2 : D * 80 = A) : A = 480 :=
sorry

end total_acres_cleaned_l86_8687


namespace right_triangle_hypotenuse_45_deg_4_inradius_l86_8612

theorem right_triangle_hypotenuse_45_deg_4_inradius : 
  ∀ (R : ℝ) (hypotenuse_length : ℝ), R = 4 ∧ 
  (∀ (A B C : ℝ), A = 45 ∧ B = 45 ∧ C = 90) →
  hypotenuse_length = 8 :=
by
  sorry

end right_triangle_hypotenuse_45_deg_4_inradius_l86_8612


namespace unemployment_percentage_next_year_l86_8679

theorem unemployment_percentage_next_year (E U : ℝ) (h1 : E > 0) :
  ( (0.91 * (0.056 * E)) / (1.04 * E) ) * 100 = 4.9 := by
  sorry

end unemployment_percentage_next_year_l86_8679


namespace valid_k_values_l86_8621

theorem valid_k_values
  (k : ℝ)
  (h : k = -7 ∨ k = -5 ∨ k = 1 ∨ k = 4) :
  (∀ x, -4 < x ∧ x < 1 → (x < k ∨ x > k + 2)) → (k = -7 ∨ k = 1 ∨ k = 4) :=
by sorry

end valid_k_values_l86_8621


namespace calculate_expression_l86_8665

theorem calculate_expression : (Real.pi - 2023)^0 - |1 - Real.sqrt 2| + 2 * Real.cos (Real.pi / 4) - (1 / 2)⁻¹ = 0 :=
by
  sorry

end calculate_expression_l86_8665


namespace geometric_sequence_b_eq_neg3_l86_8609

theorem geometric_sequence_b_eq_neg3 (a b c : ℝ) : 
  (∃ r : ℝ, -1 = r * a ∧ a = r * b ∧ b = r * c ∧ c = r * (-9)) → b = -3 :=
by
  intro h
  obtain ⟨r, h1, h2, h3, h4⟩ := h
  -- Proof to be filled in later.
  sorry

end geometric_sequence_b_eq_neg3_l86_8609


namespace two_digit_number_tens_place_l86_8605

theorem two_digit_number_tens_place (x y : Nat) (hx1 : 0 ≤ x) (hx2 : x ≤ 9) (hy1 : 0 ≤ y) (hy2 : y ≤ 9)
    (h : (x + y) * 3 = 10 * x + y - 2) : x = 2 := 
sorry

end two_digit_number_tens_place_l86_8605


namespace Robert_can_read_one_book_l86_8675

def reading_speed : ℕ := 100 -- pages per hour
def book_length : ℕ := 350 -- pages
def available_time : ℕ := 5 -- hours

theorem Robert_can_read_one_book :
  (available_time * reading_speed) >= book_length ∧ 
  (available_time * reading_speed) < 2 * book_length :=
by {
  -- The proof steps are omitted as instructed.
  sorry
}

end Robert_can_read_one_book_l86_8675


namespace sufficient_condition_for_inequality_l86_8644

theorem sufficient_condition_for_inequality (m : ℝ) : (m ≥ 2) → (∀ x : ℝ, x^2 - 2 * x + m ≥ 0) :=
by
  sorry

end sufficient_condition_for_inequality_l86_8644


namespace students_needed_to_fill_buses_l86_8628

theorem students_needed_to_fill_buses (n : ℕ) (c : ℕ) (h_n : n = 254) (h_c : c = 30) : 
  (c * ((n + c - 1) / c) - n) = 16 :=
by
  sorry

end students_needed_to_fill_buses_l86_8628


namespace hexagon_shaded_area_correct_l86_8648

theorem hexagon_shaded_area_correct :
  let side_length := 3
  let semicircle_radius := side_length / 2
  let central_circle_radius := 1
  let hexagon_area := (3 * Real.sqrt 3 / 2) * side_length ^ 2
  let semicircle_area := (π * (semicircle_radius ^ 2)) / 2
  let total_semicircle_area := 6 * semicircle_area
  let central_circle_area := π * (central_circle_radius ^ 2)
  let shaded_area := hexagon_area - (total_semicircle_area + central_circle_area)
  shaded_area = 13.5 * Real.sqrt 3 - 7.75 * π := by
  sorry

end hexagon_shaded_area_correct_l86_8648


namespace employees_6_or_more_percentage_is_18_l86_8620

-- Defining the employee counts for different year ranges
def count_less_than_1 (y : ℕ) : ℕ := 4 * y
def count_1_to_2 (y : ℕ) : ℕ := 6 * y
def count_2_to_3 (y : ℕ) : ℕ := 7 * y
def count_3_to_4 (y : ℕ) : ℕ := 4 * y
def count_4_to_5 (y : ℕ) : ℕ := 3 * y
def count_5_to_6 (y : ℕ) : ℕ := 3 * y
def count_6_to_7 (y : ℕ) : ℕ := 2 * y
def count_7_to_8 (y : ℕ) : ℕ := 2 * y
def count_8_to_9 (y : ℕ) : ℕ := y
def count_9_to_10 (y : ℕ) : ℕ := y

-- Sum of all employees T
def total_employees (y : ℕ) : ℕ := count_less_than_1 y + count_1_to_2 y + count_2_to_3 y +
                                    count_3_to_4 y + count_4_to_5 y + count_5_to_6 y +
                                    count_6_to_7 y + count_7_to_8 y + count_8_to_9 y +
                                    count_9_to_10 y

-- Employees with 6 years or more E
def employees_6_or_more (y : ℕ) : ℕ := count_6_to_7 y + count_7_to_8 y + count_8_to_9 y + count_9_to_10 y

-- Calculate percentage
def percentage (y : ℕ) : ℚ := (employees_6_or_more y : ℚ) / (total_employees y : ℚ) * 100

-- Proving the final statement
theorem employees_6_or_more_percentage_is_18 (y : ℕ) (hy : y ≠ 0) : percentage y = 18 :=
by
  sorry

end employees_6_or_more_percentage_is_18_l86_8620


namespace radius_of_semi_circle_l86_8698

variable (r w l : ℝ)

def rectangle_inscribed_semi_circle (w l : ℝ) := 
  l = 3*w ∧ 
  2*l + 2*w = 126 ∧ 
  (∃ r, l = 2*r)

theorem radius_of_semi_circle :
  (∃ w l r, rectangle_inscribed_semi_circle w l ∧ l = 2*r) → r = 23.625 :=
by
  sorry

end radius_of_semi_circle_l86_8698


namespace total_distance_is_10_miles_l86_8636

noncomputable def total_distance_back_to_town : ℕ :=
  let distance1 := 3
  let distance2 := 3
  let distance3 := 4
  distance1 + distance2 + distance3

theorem total_distance_is_10_miles :
  total_distance_back_to_town = 10 :=
by
  sorry

end total_distance_is_10_miles_l86_8636


namespace two_digit_numbers_condition_l86_8678

theorem two_digit_numbers_condition (a b : ℕ) (h1 : a ≠ 0) (h2 : 1 ≤ a ∧ a ≤ 9) (h3 : 0 ≤ b ∧ b ≤ 9) :
  (a + 1) * (b + 1) = 10 * a + b + 1 ↔ b = 9 := 
sorry

end two_digit_numbers_condition_l86_8678


namespace arithmetic_sum_example_l86_8647

def S (n : ℕ) (a1 d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

def a (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d

theorem arithmetic_sum_example (a1 d : ℤ) 
  (S20_eq_340 : S 20 a1 d = 340) :
  a 6 a1 d + a 9 a1 d + a 11 a1 d + a 16 a1 d = 68 :=
by
  sorry

end arithmetic_sum_example_l86_8647


namespace quadratic_root_proof_l86_8629

noncomputable def root_condition (p q m n : ℝ) :=
  ∃ x : ℝ, x^2 + p * x + q = 0 ∧ x ≠ 0 ∧ (1/x)^2 + m * (1/x) + n = 0

theorem quadratic_root_proof (p q m n : ℝ) (h : root_condition p q m n) :
  (pn - m) * (qm - p) = (qn - 1)^2 :=
sorry

end quadratic_root_proof_l86_8629


namespace margo_total_distance_l86_8694

-- Definitions based on the conditions
def time_to_friends_house_min : ℕ := 15
def time_to_return_home_min : ℕ := 25
def total_walking_time_min : ℕ := time_to_friends_house_min + time_to_return_home_min
def total_walking_time_hours : ℚ := total_walking_time_min / 60
def average_walking_rate_mph : ℚ := 3
def total_distance_miles : ℚ := average_walking_rate_mph * total_walking_time_hours

-- The statement of the proof problem
theorem margo_total_distance : total_distance_miles = 2 := by
  sorry

end margo_total_distance_l86_8694


namespace total_cows_l86_8697

variable (D C : ℕ)

-- The conditions of the problem translated to Lean definitions
def total_heads := D + C
def total_legs := 2 * D + 4 * C 

-- The main theorem based on the conditions and the result to prove
theorem total_cows (h1 : total_legs D C = 2 * total_heads D C + 40) : C = 20 :=
by
  sorry


end total_cows_l86_8697


namespace flour_needed_for_two_loaves_l86_8606

-- Define the amount of flour needed for one loaf.
def flour_per_loaf : ℝ := 2.5

-- Define the number of loaves.
def number_of_loaves : ℕ := 2

-- Define the total amount of flour needed for the given number of loaves.
def total_flour_needed : ℝ := flour_per_loaf * number_of_loaves

-- The theorem statement: Prove that the total amount of flour needed is 5 cups.
theorem flour_needed_for_two_loaves : total_flour_needed = 5 := by
  sorry

end flour_needed_for_two_loaves_l86_8606


namespace find_X_l86_8671

-- Defining the given conditions and what we need to prove
theorem find_X (X : ℝ) (h : (X + 43 / 151) * 151 = 2912) : X = 19 :=
sorry

end find_X_l86_8671


namespace multiply_expression_l86_8685

theorem multiply_expression (x : ℝ) : (x^4 + 12 * x^2 + 144) * (x^2 - 12) = x^6 - 1728 := by
  sorry

end multiply_expression_l86_8685


namespace sum_of_legs_le_sqrt2_hypotenuse_l86_8601

theorem sum_of_legs_le_sqrt2_hypotenuse
  (a b c : ℝ)
  (h : a^2 + b^2 = c^2) :
  a + b ≤ Real.sqrt 2 * c :=
sorry

end sum_of_legs_le_sqrt2_hypotenuse_l86_8601


namespace part_I_solution_part_II_solution_l86_8676

-- Defining f(x) given parameters a and b
def f (x a b : ℝ) := |x - a| + |x + b|

-- Part (I): Given a = 1 and b = 2, solve the inequality f(x) ≤ 5
theorem part_I_solution (x : ℝ) : 
  (f x 1 2) ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := 
by
  sorry

-- Part (II): Given the minimum value of f(x) is 3, find min (a^2 / b + b^2 / a)
theorem part_II_solution (a b : ℝ) (h : 3 = |a| + |b|) (ha : a > 0) (hb : b > 0) : 
  (min (a^2 / b + b^2 / a)) = 3 := 
by
  sorry

end part_I_solution_part_II_solution_l86_8676


namespace interest_rate_first_part_l86_8659

theorem interest_rate_first_part 
  (total_amount : ℤ) 
  (amount_at_first_rate : ℤ) 
  (amount_at_second_rate : ℤ) 
  (rate_second_part : ℤ) 
  (total_annual_interest : ℤ) 
  (r : ℤ) 
  (h_split : total_amount = amount_at_first_rate + amount_at_second_rate) 
  (h_second : rate_second_part = 5)
  (h_interest : (amount_at_first_rate * r) / 100 + (amount_at_second_rate * rate_second_part) / 100 = total_annual_interest) :
  r = 3 := 
by 
  sorry

end interest_rate_first_part_l86_8659


namespace original_square_perimeter_l86_8626

theorem original_square_perimeter (P : ℝ) (x : ℝ) (h1 : 4 * x * 2 + 4 * x = 56) : P = 32 :=
by
  sorry

end original_square_perimeter_l86_8626


namespace integer_solutions_exist_l86_8669

theorem integer_solutions_exist (k : ℤ) :
  (∃ x : ℤ, 9 * x - 3 = k * x + 14) ↔ (k = 8 ∨ k = 10 ∨ k = -8 ∨ k = 26) :=
by
  sorry

end integer_solutions_exist_l86_8669


namespace xiao_qian_has_been_to_great_wall_l86_8682

-- Define the four students
inductive Student
| XiaoZhao
| XiaoQian
| XiaoSun
| XiaoLi

open Student

-- Define the relations for their statements
def has_been (s : Student) : Prop :=
  match s with
  | XiaoZhao => false
  | XiaoQian => true
  | XiaoSun => true
  | XiaoLi => false

def said (s : Student) : Prop :=
  match s with
  | XiaoZhao => ¬has_been XiaoZhao
  | XiaoQian => has_been XiaoLi
  | XiaoSun => has_been XiaoQian
  | XiaoLi => ¬has_been XiaoLi

axiom only_one_lying : ∃ l : Student, ∀ s : Student, said s → (s ≠ l)

theorem xiao_qian_has_been_to_great_wall : has_been XiaoQian :=
by {
  sorry -- Proof elided
}

end xiao_qian_has_been_to_great_wall_l86_8682


namespace BillCookingTime_l86_8643

-- Definitions corresponding to the conditions
def chopTimePepper : Nat := 3  -- minutes to chop one pepper
def chopTimeOnion : Nat := 4   -- minutes to chop one onion
def grateTimeCheese : Nat := 1 -- minutes to grate cheese for one omelet
def cookTimeOmelet : Nat := 5  -- minutes to assemble and cook one omelet

def numberOfPeppers : Nat := 4  -- number of peppers Bill needs to chop
def numberOfOnions : Nat := 2   -- number of onions Bill needs to chop
def numberOfOmelets : Nat := 5  -- number of omelets Bill prepares

-- Calculations based on conditions
def totalChopTimePepper : Nat := numberOfPeppers * chopTimePepper
def totalChopTimeOnion : Nat := numberOfOnions * chopTimeOnion
def totalGrateTimeCheese : Nat := numberOfOmelets * grateTimeCheese
def totalCookTimeOmelet : Nat := numberOfOmelets * cookTimeOmelet

-- Total preparation and cooking time
def totalTime : Nat := totalChopTimePepper + totalChopTimeOnion + totalGrateTimeCheese + totalCookTimeOmelet

-- Theorem statement
theorem BillCookingTime :
  totalTime = 50 := by
  sorry

end BillCookingTime_l86_8643


namespace smallest_third_term_geometric_l86_8630

theorem smallest_third_term_geometric (d : ℝ) : 
  (∃ d, (7 + d) ^ 2 = 4 * (26 + 2 * d)) → ∃ g3, (g3 = 10 ∨ g3 = 36) ∧ g3 = min (10) (36) :=
by
  sorry

end smallest_third_term_geometric_l86_8630


namespace absolute_value_inequality_l86_8627

theorem absolute_value_inequality (x : ℝ) : ¬ (|x - 3| + |x + 4| < 6) :=
sorry

end absolute_value_inequality_l86_8627


namespace andrew_correct_answer_l86_8660

variable {x : ℕ}

theorem andrew_correct_answer (h : (x - 8) / 7 = 15) : (x - 5) / 11 = 10 :=
by
  sorry

end andrew_correct_answer_l86_8660
