import Mathlib

namespace major_axis_length_l224_22403

/-- Defines the properties of the ellipse we use in this problem. --/
def ellipse (x y : ℝ) : Prop :=
  let f1 := (5, 1 + Real.sqrt 8)
  let f2 := (5, 1 - Real.sqrt 8)
  let tangent_line_at_y := y = 1
  let tangent_line_at_x := x = 1
  tangent_line_at_y ∧ tangent_line_at_x ∧
  ((x - f1.1)^2 + (y - f1.2)^2) + ((x - f2.1)^2 + (y - f2.2)^2) = 4

/-- Proves the length of the major axis of the specific ellipse --/
theorem major_axis_length : ∃ l : ℝ, l = 4 :=
  sorry

end major_axis_length_l224_22403


namespace product_of_differences_of_squares_is_diff_of_square_l224_22421

-- Define when an integer is a difference of squares of positive integers
def diff_of_squares (n : ℕ) : Prop :=
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ n = x^2 - y^2

-- State the main theorem
theorem product_of_differences_of_squares_is_diff_of_square 
  (a b c d : ℕ) (h₁ : diff_of_squares a) (h₂ : diff_of_squares b) (h₃ : diff_of_squares c) (h₄ : diff_of_squares d) : 
  diff_of_squares (a * b * c * d) := by
  sorry

end product_of_differences_of_squares_is_diff_of_square_l224_22421


namespace students_chose_apples_l224_22446

theorem students_chose_apples (total students choosing_bananas : ℕ) (h1 : students_choosing_bananas = 168) 
  (h2 : 3 * total = 4 * students_choosing_bananas) : (total / 4) = 56 :=
  by
  sorry

end students_chose_apples_l224_22446


namespace company_bought_gravel_l224_22442

def weight_of_gravel (total_weight_of_materials : ℝ) (weight_of_sand : ℝ) : ℝ :=
  total_weight_of_materials - weight_of_sand

theorem company_bought_gravel :
  weight_of_gravel 14.02 8.11 = 5.91 := 
by
  sorry

end company_bought_gravel_l224_22442


namespace total_time_watching_videos_l224_22459

theorem total_time_watching_videos 
  (cat_video_length : ℕ)
  (dog_video_length : ℕ)
  (gorilla_video_length : ℕ)
  (h1 : cat_video_length = 4)
  (h2 : dog_video_length = 2 * cat_video_length)
  (h3 : gorilla_video_length = 2 * (cat_video_length + dog_video_length)) :
  cat_video_length + dog_video_length + gorilla_video_length = 36 :=
  by
  sorry

end total_time_watching_videos_l224_22459


namespace scientific_notation_proof_l224_22433

-- Given number is 657,000
def number : ℕ := 657000

-- Scientific notation of the given number
def scientific_notation (n : ℕ) : Prop :=
    n = 657000 ∧ (6.57 : ℝ) * (10 : ℝ)^5 = 657000

theorem scientific_notation_proof : scientific_notation number :=
by 
  sorry

end scientific_notation_proof_l224_22433


namespace smallest_n_l224_22419

theorem smallest_n (n : ℕ) (h1 : n ≥ 1)
  (h2 : ∃ k : ℕ, 2002 * n = k ^ 3)
  (h3 : ∃ m : ℕ, n = 2002 * m ^ 2) :
  n = 2002^5 := sorry

end smallest_n_l224_22419


namespace not_p_and_pq_false_not_necessarily_p_or_q_l224_22496

theorem not_p_and_pq_false_not_necessarily_p_or_q (p q : Prop) 
  (h1 : ¬p) 
  (h2 : ¬(p ∧ q)) : ¬(p ∨ q) ∨ (p ∨ q) := by
  sorry

end not_p_and_pq_false_not_necessarily_p_or_q_l224_22496


namespace mike_spent_on_mower_blades_l224_22414

theorem mike_spent_on_mower_blades (x : ℝ) 
  (initial_money : ℝ := 101) 
  (cost_of_games : ℝ := 54) 
  (games : ℝ := 9) 
  (price_per_game : ℝ := 6) 
  (h1 : 101 - x = 54) :
  x = 47 := 
by
  sorry

end mike_spent_on_mower_blades_l224_22414


namespace inequality_proof_l224_22410

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  2 * (a + b + c) + 9 / (a * b + b * c + c * a)^2 ≥ 7 :=
by
  sorry

end inequality_proof_l224_22410


namespace value_of_place_ratio_l224_22464

theorem value_of_place_ratio :
  let d8_pos := 10000
  let d6_pos := 0.1
  d8_pos = 100000 * d6_pos :=
by
  let d8_pos := 10000
  let d6_pos := 0.1
  sorry

end value_of_place_ratio_l224_22464


namespace image_center_after_reflection_and_translation_l224_22412

def circle_center_before_translation : ℝ × ℝ := (3, -4)

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-x, y)

def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (x, y + d)

theorem image_center_after_reflection_and_translation :
  translate_up (reflect_y_axis circle_center_before_translation) 5 = (-3, 1) :=
by
  -- The detail proof goes here.
  sorry

end image_center_after_reflection_and_translation_l224_22412


namespace exists_n_such_that_n_pow_n_plus_n_plus_one_pow_n_divisible_by_1987_l224_22438

theorem exists_n_such_that_n_pow_n_plus_n_plus_one_pow_n_divisible_by_1987 :
  ∃ n : ℕ, n ^ n + (n + 1) ^ n ≡ 0 [MOD 1987] := sorry

end exists_n_such_that_n_pow_n_plus_n_plus_one_pow_n_divisible_by_1987_l224_22438


namespace evaluate_expression_l224_22457

theorem evaluate_expression :
  let a := Real.sqrt 2 ^ 2 + Real.sqrt 3 + Real.sqrt 5
  let b := - Real.sqrt 2 ^ 2 + Real.sqrt 3 + Real.sqrt 5
  let c := Real.sqrt 2 ^ 2 - Real.sqrt 3 + Real.sqrt 5
  let d := - Real.sqrt 2 ^ 2 - Real.sqrt 3 + Real.sqrt 5
  (1/a + 1/b + 1/c + 1/d)^2 = 5 :=
by
  sorry

end evaluate_expression_l224_22457


namespace n_square_divisible_by_144_l224_22451

theorem n_square_divisible_by_144 (n : ℤ) (hn : n > 0)
  (hw : ∃ k : ℤ, n = 12 * k) : ∃ m : ℤ, n^2 = 144 * m :=
by {
  sorry
}

end n_square_divisible_by_144_l224_22451


namespace number_of_students_speaking_two_languages_l224_22406

variables (G H M GH GM HM GHM N : ℕ)

def students_speaking_two_languages (G H M GH GM HM GHM N : ℕ) : ℕ :=
  G + H + M - (GH + GM + HM) + GHM

theorem number_of_students_speaking_two_languages 
  (h_total : N = 22)
  (h_G : G = 6)
  (h_H : H = 15)
  (h_M : M = 6)
  (h_GHM : GHM = 1)
  (h_students : N = students_speaking_two_languages G H M GH GM HM GHM N): 
  GH + GM + HM = 6 := 
by 
  unfold students_speaking_two_languages at h_students 
  sorry

end number_of_students_speaking_two_languages_l224_22406


namespace find_y_value_l224_22444

theorem find_y_value : (15^3 * 7^4) / 5670 = 1428.75 := by
  sorry

end find_y_value_l224_22444


namespace ratio_problem_l224_22456

theorem ratio_problem 
  (a b c d : ℚ)
  (h₁ : a / b = 8)
  (h₂ : c / b = 5)
  (h₃ : c / d = 1 / 3) : 
  d / a = 15 / 8 := 
by 
  sorry

end ratio_problem_l224_22456


namespace sheets_in_stack_l224_22469

theorem sheets_in_stack (sheets : ℕ) (thickness : ℝ) (h1 : sheets = 400) (h2 : thickness = 4) :
    let thickness_per_sheet := thickness / sheets
    let stack_height := 6
    (stack_height / thickness_per_sheet = 600) :=
by
  sorry

end sheets_in_stack_l224_22469


namespace largest_power_dividing_factorial_l224_22426

theorem largest_power_dividing_factorial (n : ℕ) (h : n = 2015) : ∃ k : ℕ, (2015^k ∣ n!) ∧ k = 67 :=
by
  sorry

end largest_power_dividing_factorial_l224_22426


namespace sum_of_variables_is_233_l224_22475

-- Define A, B, C, D, E, F with their corresponding values.
def A : ℤ := 13
def B : ℤ := 9
def C : ℤ := -3
def D : ℤ := -2
def E : ℕ := 165
def F : ℕ := 51

-- Define the main theorem to prove the sum of A, B, C, D, E, F equals 233.
theorem sum_of_variables_is_233 : A + B + C + D + E + F = 233 := 
by {
  -- Proof is not required according to problem statement, hence using sorry.
  sorry
}

end sum_of_variables_is_233_l224_22475


namespace mask_distribution_l224_22440

theorem mask_distribution (x : ℕ) (total_masks_3 : ℕ) (total_masks_4 : ℕ)
    (h1 : total_masks_3 = 3 * x + 20)
    (h2 : total_masks_4 = 4 * x - 25) :
    3 * x + 20 = 4 * x - 25 :=
by
  sorry

end mask_distribution_l224_22440


namespace negative_expression_b_negative_expression_c_negative_expression_e_l224_22432

theorem negative_expression_b:
  3 * Real.sqrt 11 - 10 < 0 := 
sorry

theorem negative_expression_c:
  18 - 5 * Real.sqrt 13 < 0 := 
sorry

theorem negative_expression_e:
  10 * Real.sqrt 26 - 51 < 0 := 
sorry

end negative_expression_b_negative_expression_c_negative_expression_e_l224_22432


namespace option_C_correct_l224_22430

theorem option_C_correct (a b : ℝ) : (2 * a * b^2)^2 = 4 * a^2 * b^4 := 
by 
  sorry

end option_C_correct_l224_22430


namespace yolanda_walking_rate_l224_22431

theorem yolanda_walking_rate 
  (d_xy : ℕ) (bob_start_after_yolanda : ℕ) (bob_distance_walked : ℕ) 
  (bob_rate : ℕ) (y : ℕ) 
  (bob_distance_to_time : bob_rate ≠ 0 ∧ bob_distance_walked / bob_rate = 2) 
  (yolanda_distance_walked : d_xy - bob_distance_walked = 9 ∧ y = 9 / 3) : 
  y = 3 :=
by 
  sorry

end yolanda_walking_rate_l224_22431


namespace minimum_bail_rate_l224_22415

theorem minimum_bail_rate 
  (distance : ℝ)
  (leak_rate : ℝ)
  (max_water : ℝ)
  (rowing_speed : ℝ)
  (bail_rate : ℝ)
  (time_to_shore : ℝ) :
  distance = 2 ∧
  leak_rate = 15 ∧
  max_water = 60 ∧
  rowing_speed = 3 ∧
  time_to_shore = distance / rowing_speed * 60 →
  bail_rate = (leak_rate * time_to_shore - max_water) / time_to_shore →
  bail_rate = 13.5 :=
by
  intros
  sorry

end minimum_bail_rate_l224_22415


namespace slope_of_line_l224_22488

-- Definitions of the conditions in the problem
def line_eq (a : ℝ) (x y : ℝ) : Prop := x + a * y + 1 = 0

def y_intercept (l : ℝ → ℝ → Prop) (b : ℝ) : Prop :=
  l 0 b

-- The statement of the proof problem
theorem slope_of_line (a : ℝ) (h : y_intercept (line_eq a) (-2)) : 
  ∃ (m : ℝ), m = -2 :=
sorry

end slope_of_line_l224_22488


namespace intersection_eq_l224_22429

-- defining the set A
def A := {x : ℝ | x^2 + 2*x - 3 ≤ 0}

-- defining the set B
def B := {y : ℝ | ∃ x ∈ A, y = x^2 + 4*x + 3}

-- The proof problem statement: prove that A ∩ B = [-1, 1]
theorem intersection_eq : A ∩ B = {y : ℝ | -1 ≤ y ∧ y ≤ 1} :=
by sorry

end intersection_eq_l224_22429


namespace waynes_son_time_to_shovel_l224_22466

-- Definitions based on the conditions
variables (S W : ℝ) (son_rate : S = 1 / 21) (wayne_rate : W = 6 * S) (together_rate : 3 * (S + W) = 1)

theorem waynes_son_time_to_shovel : 
  1 / S = 21 :=
by
  -- Proof will be provided later
  sorry

end waynes_son_time_to_shovel_l224_22466


namespace player_placing_third_won_against_seventh_l224_22416

theorem player_placing_third_won_against_seventh :
  ∃ (s : Fin 8 → ℚ),
    -- Condition 1: Scores are different
    (∀ i j, i ≠ j → s i ≠ s j) ∧
    -- Condition 2: Second place score equals the sum of the bottom four scores
    (s 1 = s 4 + s 5 + s 6 + s 7) ∧
    -- Result: Third player won against the seventh player
    (s 2 > s 6) :=
sorry

end player_placing_third_won_against_seventh_l224_22416


namespace no_xy_term_implies_k_eq_4_l224_22463

theorem no_xy_term_implies_k_eq_4 (k : ℝ) :
  (∀ x y : ℝ, (x + 2 * y) * (2 * x - k * y - 1) = 2 * x^2 + (4 - k) * x * y - x - 2 * k * y^2 - 2 * y) →
  ((4 - k) = 0) →
  k = 4 := 
by
  intros h1 h2
  sorry

end no_xy_term_implies_k_eq_4_l224_22463


namespace total_spent_on_toys_l224_22411

-- Definition of the costs
def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59

-- The theorem to prove the total amount spent on toys
theorem total_spent_on_toys : football_cost + marbles_cost = 12.30 :=
by sorry

end total_spent_on_toys_l224_22411


namespace min_filtration_cycles_l224_22497

theorem min_filtration_cycles {c₀ : ℝ} (initial_concentration : c₀ = 225)
  (max_concentration : ℝ := 7.5) (reduction_factor : ℝ := 1 / 3)
  (log2 : ℝ := 0.3010) (log3 : ℝ := 0.4771) :
  ∃ n : ℕ, (c₀ * (reduction_factor ^ n) ≤ max_concentration ∧ n ≥ 9) :=
sorry

end min_filtration_cycles_l224_22497


namespace range_of_t_l224_22492

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, x < y → f y < f x

variable {f : ℝ → ℝ}

theorem range_of_t (h_odd : odd_function f) 
  (h_decreasing : decreasing_function f)
  (h_ineq : ∀ t, -1 < t → t < 1 → f (1 - t) + f (1 - t^2) < 0) 
  : ∀ t, 0 < t → t < 1 :=
by sorry

end range_of_t_l224_22492


namespace sample_size_is_200_l224_22498
-- Define the total number of students and the number of students surveyed
def total_students : ℕ := 3600
def students_surveyed : ℕ := 200

-- Define the sample size
def sample_size := students_surveyed

-- Prove the sample size is 200
theorem sample_size_is_200 : sample_size = 200 :=
by
  -- Placeholder for the actual proof
  sorry

end sample_size_is_200_l224_22498


namespace simplify_expression1_simplify_expression2_l224_22404

section
variables (a b : ℝ)

theorem simplify_expression1 : -b*(2*a - b) + (a + b)^2 = a^2 + 2*b^2 :=
sorry
end

section
variables (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2)

theorem simplify_expression2 : (1 - (x/(2 + x))) / ((x^2 - 4)/(x^2 + 4*x + 4)) = 2/(x - 2) :=
sorry
end

end simplify_expression1_simplify_expression2_l224_22404


namespace number_of_chairs_is_40_l224_22409

-- Define the conditions
variables (C : ℕ) -- Total number of chairs
variables (capacity_per_chair : ℕ := 2) -- Each chair's capacity is 2 people
variables (occupied_ratio : ℚ := 3 / 5) -- Ratio of occupied chairs
variables (attendees : ℕ := 48) -- Number of attendees

theorem number_of_chairs_is_40
  (h1 : ∀ c : ℕ, capacity_per_chair * c = attendees)
  (h2 : occupied_ratio * C * capacity_per_chair = attendees) : 
  C = 40 := sorry

end number_of_chairs_is_40_l224_22409


namespace sahil_purchase_price_l224_22470

def purchase_price (P : ℝ) : Prop :=
  let repair_cost := 5000
  let transportation_charges := 1000
  let total_cost := repair_cost + transportation_charges
  let selling_price := 27000
  let profit_factor := 1.5
  profit_factor * (P + total_cost) = selling_price

theorem sahil_purchase_price : ∃ P : ℝ, purchase_price P ∧ P = 12000 :=
by
  use 12000
  unfold purchase_price
  simp
  sorry

end sahil_purchase_price_l224_22470


namespace trig_relation_l224_22417

theorem trig_relation (a b c : ℝ) 
  (h1 : a = Real.sin 2) 
  (h2 : b = Real.cos 2) 
  (h3 : c = Real.tan 2) : c < b ∧ b < a := 
by
  sorry

end trig_relation_l224_22417


namespace remainder_of_product_l224_22450

theorem remainder_of_product (a b n : ℕ) (h1 : a = 2431) (h2 : b = 1587) (h3 : n = 800) : 
  (a * b) % n = 397 := 
by
  sorry

end remainder_of_product_l224_22450


namespace sin_110_correct_tan_945_correct_cos_25pi_over_4_correct_l224_22481

noncomputable def sin_110_degrees : ℝ := Real.sin (110 * Real.pi / 180)
noncomputable def tan_945_degrees_reduction : ℝ := Real.tan (945 * Real.pi / 180 - 5 * Real.pi)
noncomputable def cos_25pi_over_4_reduction : ℝ := Real.cos (25 * Real.pi / 4 - 6 * 2 * Real.pi)

theorem sin_110_correct : sin_110_degrees = Real.sin (110 * Real.pi / 180) :=
by
  sorry

theorem tan_945_correct : tan_945_degrees_reduction = 1 :=
by 
  sorry

theorem cos_25pi_over_4_correct : cos_25pi_over_4_reduction = Real.cos (Real.pi / 4) :=
by 
  sorry

end sin_110_correct_tan_945_correct_cos_25pi_over_4_correct_l224_22481


namespace complement_union_eq_l224_22439

-- Definitions / Conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }

-- Statement of the theorem
theorem complement_union_eq {x : ℝ} :
  {x | x ≥ 2} = (U \ (M ∪ N)) := sorry

end complement_union_eq_l224_22439


namespace instructors_teach_together_in_360_days_l224_22434

def Felicia_teaches_every := 5
def Greg_teaches_every := 3
def Hannah_teaches_every := 9
def Ian_teaches_every := 2
def Joy_teaches_every := 8

def lcm_multiple (a b c d e : ℕ) : ℕ := Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d e)))

theorem instructors_teach_together_in_360_days :
  lcm_multiple Felicia_teaches_every
               Greg_teaches_every
               Hannah_teaches_every
               Ian_teaches_every
               Joy_teaches_every = 360 :=
by
  -- Since the real proof is omitted, we close with sorry
  sorry

end instructors_teach_together_in_360_days_l224_22434


namespace monomial_addition_l224_22458

-- Definition of a monomial in Lean
def isMonomial (p : ℕ → ℝ) : Prop := ∃ c n, ∀ x, p x = c * x^n

theorem monomial_addition (A : ℕ → ℝ) :
  (isMonomial (fun x => -3 * x + A x)) → isMonomial A :=
sorry

end monomial_addition_l224_22458


namespace combined_loss_l224_22427

variable (initial : ℕ) (donation : ℕ) (prize : ℕ) (final : ℕ) (lottery_winning : ℕ) (X : ℕ)

theorem combined_loss (h1 : initial = 10) (h2 : donation = 4) (h3 : prize = 90) 
                      (h4 : final = 94) (h5 : lottery_winning = 65) :
                      (initial - donation + prize - X + lottery_winning = final) ↔ (X = 67) :=
by
  -- proof steps will go here
  sorry

end combined_loss_l224_22427


namespace smallest_b_value_is_6_l224_22479

noncomputable def smallest_b_value (a b c : ℝ) (h_arith : a + c = 2 * b) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 216) : ℝ :=
b

theorem smallest_b_value_is_6 (a b c : ℝ) (h_arith : a + c = 2 * b) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 216) : 
  smallest_b_value a b c h_arith h_pos h_prod = 6 :=
sorry

end smallest_b_value_is_6_l224_22479


namespace root_equation_value_l224_22472

theorem root_equation_value (m : ℝ) (h : m^2 - 2 * m - 3 = 0) : 2026 - m^2 + 2 * m = 2023 :=
sorry

end root_equation_value_l224_22472


namespace container_capacity_l224_22461

variable (C : ℝ)
variable (h1 : 0.30 * C + 27 = (3/4) * C)

theorem container_capacity : C = 60 := by
  sorry

end container_capacity_l224_22461


namespace sam_investment_time_l224_22482

theorem sam_investment_time (P r : ℝ) (n A t : ℕ) (hP : P = 8000) (hr : r = 0.10) (hn : n = 2) (hA : A = 8820) :
  A = P * (1 + r / n) ^ (n * t) → t = 1 :=
by
  sorry

end sam_investment_time_l224_22482


namespace maximize_farmer_profit_l224_22436

theorem maximize_farmer_profit :
  ∃ x y : ℝ, x + y ≤ 2 ∧ 3 * x + y ≤ 5 ∧ x ≥ 0 ∧ y ≥ 0 ∧ x = 1.5 ∧ y = 0.5 ∧ 
  (∀ x' y' : ℝ, x' + y' ≤ 2 ∧ 3 * x' + y' ≤ 5 ∧ x' ≥ 0 ∧ y' ≥ 0 → 14400 * x + 6300 * y ≥ 14400 * x' + 6300 * y') :=
by
  sorry

end maximize_farmer_profit_l224_22436


namespace x_eq_sum_of_squares_of_two_consecutive_integers_l224_22428

noncomputable def x_seq (n : ℕ) : ℝ :=
  1 / 4 * ((2 + Real.sqrt 3) ^ (2 * n - 1) + (2 - Real.sqrt 3) ^ (2 * n - 1))

theorem x_eq_sum_of_squares_of_two_consecutive_integers (n : ℕ) : 
  ∃ y : ℤ, x_seq n = (y:ℝ)^2 + (y + 1)^2 :=
sorry

end x_eq_sum_of_squares_of_two_consecutive_integers_l224_22428


namespace find_C_probability_within_r_l224_22401

noncomputable def probability_density (x y R : ℝ) (C : ℝ) : ℝ :=
if x^2 + y^2 <= R^2 then C * (R - Real.sqrt (x^2 + y^2)) else 0

noncomputable def total_integral (R : ℝ) (C : ℝ) : ℝ :=
∫ (x : ℝ) in -R..R, ∫ (y : ℝ) in -R..R, probability_density x y R C

theorem find_C (R : ℝ) (hR : 0 < R) : 
  (∫ (x : ℝ) in -R..R, ∫ (y : ℝ) in -R..R, probability_density x y R C) = 1 ↔ 
  C = 3 / (π * R^3) := 
by 
  sorry

theorem probability_within_r (R r : ℝ) 
  (hR : 0 < R) (hr : 0 < r) (hrR : r <= R) (P : ℝ) : 
  (∫ (x : ℝ) in -r..r, ∫ (y : ℝ) in -r..r, probability_density x y R (3 / (π * R^3))) = P ↔ 
  (R = 2 ∧ r = 1 → P = 1 / 2) := 
by 
  sorry

end find_C_probability_within_r_l224_22401


namespace real_solutions_l224_22485

open Real

theorem real_solutions (x : ℝ) : (x - 2) ^ 4 + (2 - x) ^ 4 = 50 ↔ 
  x = 2 + sqrt (-12 + 3 * sqrt 17) ∨ x = 2 - sqrt (-12 + 3 * sqrt 17) :=
by
  sorry

end real_solutions_l224_22485


namespace mechanism_completion_times_l224_22499

theorem mechanism_completion_times :
  ∃ (x y : ℝ), (1 / x + 1 / y = 1 / 30) ∧ (6 * (1 / x + 1 / y) + 40 * (1 / y) = 1) ∧ x = 75 ∧ y = 50 :=
by {
  sorry
}

end mechanism_completion_times_l224_22499


namespace probability_white_ball_l224_22491

def num_white_balls : ℕ := 5
def num_black_balls : ℕ := 6
def total_balls : ℕ := num_white_balls + num_black_balls

theorem probability_white_ball : (num_white_balls : ℚ) / total_balls = 5 / 11 := by
  sorry

end probability_white_ball_l224_22491


namespace simplify_and_evaluate_l224_22423

theorem simplify_and_evaluate (x : ℝ) (h₁ : x ≠ 0) (h₂ : x = 2) : 
  (1 + 1 / x) / ((x^2 - 1) / x) = 1 := 
by 
  sorry

end simplify_and_evaluate_l224_22423


namespace third_side_length_l224_22437

/-- Given two sides of a triangle with lengths 4cm and 9cm, prove that the valid length of the third side must be 9cm. -/
theorem third_side_length (a b c : ℝ) (h₀ : a = 4) (h₁ : b = 9) :
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) → (c = 9) :=
by {
  sorry
}

end third_side_length_l224_22437


namespace solve_geometric_sequence_product_l224_22487

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ r : ℝ, a (n + 1) = a n * r

theorem solve_geometric_sequence_product (a : ℕ → ℝ) (h_geom : geometric_sequence a)
  (h_a35 : a 3 * a 5 = 4) : 
  a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7 = 128 :=
sorry

end solve_geometric_sequence_product_l224_22487


namespace max_tickets_l224_22452

theorem max_tickets (ticket_price normal_discounted_price budget : ℕ) (h1 : ticket_price = 15) (h2 : normal_discounted_price = 13) (h3 : budget = 180) :
  ∃ n : ℕ, ((n ≤ 10 → ticket_price * n ≤ budget) ∧ (n > 10 → normal_discounted_price * n ≤ budget)) ∧ ∀ m : ℕ, ((m ≤ 10 → ticket_price * m ≤ budget) ∧ (m > 10 → normal_discounted_price * m ≤ budget)) → m ≤ 13 :=
by
  sorry

end max_tickets_l224_22452


namespace arithmetic_sequence_product_l224_22441

theorem arithmetic_sequence_product 
  (a d : ℤ)
  (h1 : a + 6 * d = 20)
  (h2 : d = 2) : 
  a * (a + d) * (a + 2 * d) = 960 := 
by
  -- proof goes here
  sorry

end arithmetic_sequence_product_l224_22441


namespace factor_1024_into_three_factors_l224_22465

theorem factor_1024_into_three_factors :
  ∃ (factors : Finset (Finset ℕ)), factors.card = 14 ∧
  ∀ f ∈ factors, ∃ a b c : ℕ, a + b + c = 10 ∧ a ≥ b ∧ b ≥ c ∧ (2 ^ a) * (2 ^ b) * (2 ^ c) = 1024 :=
sorry

end factor_1024_into_three_factors_l224_22465


namespace probability_of_two_white_balls_l224_22405

-- Define the total number of balls
def total_balls : ℕ := 11

-- Define the number of white balls
def white_balls : ℕ := 5

-- Define the number of ways to choose 2 out of n (combinations)
def choose (n r : ℕ) : ℕ := n.choose r

-- Define the total combinations of drawing 2 balls out of 11
def total_combinations : ℕ := choose total_balls 2

-- Define the combinations of drawing 2 white balls out of 5
def white_combinations : ℕ := choose white_balls 2

-- Define the probability of drawing 2 white balls
noncomputable def probability_white : ℚ := (white_combinations : ℚ) / (total_combinations : ℚ)

-- Now, state the theorem that states the desired result
theorem probability_of_two_white_balls : probability_white = 2 / 11 := sorry

end probability_of_two_white_balls_l224_22405


namespace parallel_lines_eq_l224_22474

theorem parallel_lines_eq {a x y : ℝ} :
  (∀ x y : ℝ, x + a * y = 2 * a + 2) ∧ (∀ x y : ℝ, a * x + y = a + 1) →
  a = 1 :=
by
  sorry

end parallel_lines_eq_l224_22474


namespace my_op_identity_l224_22476

def my_op (a b : ℕ) : ℕ := a + b + a * b

theorem my_op_identity (a : ℕ) : my_op (my_op a 1) 2 = 6 * a + 5 :=
by
  sorry

end my_op_identity_l224_22476


namespace gcd_7392_15015_l224_22443

-- Define the two numbers
def num1 : ℕ := 7392
def num2 : ℕ := 15015

-- State the theorem and use sorry to omit the proof
theorem gcd_7392_15015 : Nat.gcd num1 num2 = 1 := 
  by sorry

end gcd_7392_15015_l224_22443


namespace melissa_gave_x_books_l224_22455

-- Define the initial conditions as constants
def initial_melissa_books : ℝ := 123
def initial_jordan_books : ℝ := 27
def final_melissa_books (x : ℝ) : ℝ := initial_melissa_books - x
def final_jordan_books (x : ℝ) : ℝ := initial_jordan_books + x

-- The main theorem to prove how many books Melissa gave to Jordan
theorem melissa_gave_x_books : ∃ x : ℝ, final_melissa_books x = 3 * final_jordan_books x ∧ x = 10.5 :=
sorry

end melissa_gave_x_books_l224_22455


namespace convert_13_to_binary_l224_22413

theorem convert_13_to_binary : (13 : ℕ) = 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 :=
by sorry

end convert_13_to_binary_l224_22413


namespace condition_M_intersect_N_N_l224_22420

theorem condition_M_intersect_N_N (a : ℝ) :
  (∀ (x y : ℝ), (x^2 + (y - a)^2 ≤ 1 → y ≥ x^2)) ↔ (a ≥ 5 / 4) :=
sorry

end condition_M_intersect_N_N_l224_22420


namespace eccentricity_of_hyperbola_l224_22402

theorem eccentricity_of_hyperbola (a b c : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_c : c = Real.sqrt (a^2 + b^2))
  (F1 : ℝ × ℝ := (-c, 0))
  (A B : ℝ × ℝ)
  (slope_of_AB : ∀ (x y : ℝ), y = x + c)
  (asymptotes_eqn : ∀ (x : ℝ), x = a ∨ x = -a)
  (intersections : A = (-(a * c / (a - b)), -(b * c / (a - b))) ∧ B = (-(a * c / (a + b)), (b * c / (a + b))))
  (AB_eq_2BF1 : 2 * (F1 - B) = A - B) :
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 5 :=
sorry

end eccentricity_of_hyperbola_l224_22402


namespace cuberoot_eq_3_implies_cube_eq_19683_l224_22418

theorem cuberoot_eq_3_implies_cube_eq_19683 (x : ℝ) (h : (x + 6)^(1/3) = 3) : (x + 6)^3 = 19683 := by
  sorry

end cuberoot_eq_3_implies_cube_eq_19683_l224_22418


namespace value_of_otimes_difference_l224_22467

def otimes (a b : ℚ) : ℚ := (a^3) / (b^2)

theorem value_of_otimes_difference :
  otimes (otimes 2 3) 4 - otimes 2 (otimes 3 4) = - 1184 / 243 := 
by
  sorry

end value_of_otimes_difference_l224_22467


namespace cube_surface_area_l224_22473

theorem cube_surface_area (s : ℝ) (h : s = 8) : 6 * s^2 = 384 :=
by
  sorry

end cube_surface_area_l224_22473


namespace probability_AC_adjacent_l224_22453

noncomputable def probability_AC_adjacent_given_AB_adjacent : ℚ :=
  let total_permutations_with_AB_adjacent := 48
  let permutations_with_ABC_adjacent := 12
  permutations_with_ABC_adjacent / total_permutations_with_AB_adjacent

theorem probability_AC_adjacent :  
  probability_AC_adjacent_given_AB_adjacent = 1 / 4 :=
by
  sorry

end probability_AC_adjacent_l224_22453


namespace KarenEggRolls_l224_22478

-- Definitions based on conditions
def OmarEggRolls : ℕ := 219
def TotalEggRolls : ℕ := 448

-- The statement to be proved
theorem KarenEggRolls : (TotalEggRolls - OmarEggRolls = 229) :=
by {
    -- Proof step goes here
    sorry
}

end KarenEggRolls_l224_22478


namespace Lily_books_on_Wednesday_l224_22400

noncomputable def booksMike : ℕ := 45

noncomputable def booksCorey : ℕ := 2 * booksMike

noncomputable def booksMikeGivenToLily : ℕ := 13

noncomputable def booksCoreyGivenToLily : ℕ := booksMikeGivenToLily + 5

noncomputable def booksEmma : ℕ := 28

noncomputable def booksEmmaGivenToLily : ℕ := booksEmma / 4

noncomputable def totalBooksLilyGot : ℕ := booksMikeGivenToLily + booksCoreyGivenToLily + booksEmmaGivenToLily

theorem Lily_books_on_Wednesday : totalBooksLilyGot = 38 := by
  sorry

end Lily_books_on_Wednesday_l224_22400


namespace salt_percentage_in_first_solution_l224_22407

theorem salt_percentage_in_first_solution
    (S : ℝ)
    (h1 : ∀ w : ℝ, w ≥ 0 → ∃ q : ℝ, q = w)  -- One fourth of the first solution was replaced by the second solution
    (h2 : ∀ w1 w2 w3 : ℝ,
            w1 + w2 = w3 →
            (w1 / w3 * S + w2 / w3 * 25 = 16)) :  -- Resulting solution was 16 percent salt by weight
  S = 13 :=   -- Correct answer
sorry

end salt_percentage_in_first_solution_l224_22407


namespace brooke_initial_l224_22448

variable (B : ℕ)

def brooke_balloons_initially (B : ℕ) :=
  let brooke_balloons := B + 8
  let tracy_balloons_initial := 6
  let tracy_added_balloons := 24
  let tracy_balloons := tracy_balloons_initial + tracy_added_balloons
  let tracy_popped_balloons := tracy_balloons / 2 -- Tracy having half her balloons popped.
  (brooke_balloons + tracy_popped_balloons = 35)

theorem brooke_initial (h : brooke_balloons_initially B) : B = 12 :=
  sorry

end brooke_initial_l224_22448


namespace Anna_needs_308_tulips_l224_22468

-- Define conditions as assertions or definitions
def number_of_eyes := 2
def red_tulips_per_eye := 8 
def number_of_eyebrows := 2
def purple_tulips_per_eyebrow := 5
def red_tulips_for_nose := 12
def red_tulips_for_smile := 18
def yellow_tulips_background := 9 * red_tulips_for_smile
def additional_purple_tulips_eyebrows := 4 * number_of_eyes * red_tulips_per_eye - number_of_eyebrows * purple_tulips_per_eyebrow
def yellow_tulips_for_nose := 3 * red_tulips_for_nose

-- Define total number of tulips for each color
def total_red_tulips := number_of_eyes * red_tulips_per_eye + red_tulips_for_nose + red_tulips_for_smile
def total_purple_tulips := number_of_eyebrows * purple_tulips_per_eyebrow + additional_purple_tulips_eyebrows
def total_yellow_tulips := yellow_tulips_background + yellow_tulips_for_nose

-- Define the total number of tulips
def total_tulips := total_red_tulips + total_purple_tulips + total_yellow_tulips

theorem Anna_needs_308_tulips :
  total_tulips = 308 :=
sorry

end Anna_needs_308_tulips_l224_22468


namespace problem1_problem2_l224_22483

-- Problem 1: Prove that 2023 * 2023 - 2024 * 2022 = 1
theorem problem1 : 2023 * 2023 - 2024 * 2022 = 1 := 
by 
  sorry

-- Problem 2: Prove that (-4 * x * y^3) * (1/2 * x * y) + (-3 * x * y^2)^2 = 7 * x^2 * y^4
theorem problem2 (x y : ℝ) : (-4 * x * y^3) * ((1/2) * x * y) + (-3 * x * y^2)^2 = 7 * x^2 * y^4 := 
by 
  sorry

end problem1_problem2_l224_22483


namespace sum_of_coefficients_256_l224_22480

theorem sum_of_coefficients_256 (n : ℕ) (h : (3 + 1)^n = 256) : n = 4 :=
sorry

end sum_of_coefficients_256_l224_22480


namespace solve_for_y_l224_22486

theorem solve_for_y (y : ℕ) : (1000^4 = 10^y) → y = 12 :=
by {
  sorry
}

end solve_for_y_l224_22486


namespace rectangle_area_l224_22493

theorem rectangle_area (b : ℕ) (side radius length : ℕ) 
    (h1 : side * side = 1296)
    (h2 : radius = side)
    (h3 : length = radius / 6) :
    length * b = 6 * b :=
by
  sorry

end rectangle_area_l224_22493


namespace product_of_two_numbers_l224_22489

theorem product_of_two_numbers (a b : ℕ) (h_lcm : lcm a b = 48) (h_gcd : gcd a b = 8) : a * b = 384 :=
by sorry

end product_of_two_numbers_l224_22489


namespace largest_divisible_n_l224_22490

/-- Largest positive integer n for which n^3 + 10 is divisible by n + 1 --/
theorem largest_divisible_n (n : ℕ) :
  n = 0 ↔ ∀ m : ℕ, (m > n) → ¬ ((m^3 + 10) % (m + 1) = 0) :=
by
  sorry

end largest_divisible_n_l224_22490


namespace charlie_more_apples_than_bella_l224_22425

variable (D : ℝ) 

theorem charlie_more_apples_than_bella 
    (hC : C = 1.75 * D)
    (hB : B = 1.50 * D) :
    (C - B) / B = 0.1667 := 
by
  sorry

end charlie_more_apples_than_bella_l224_22425


namespace proof1_proof2_proof3_l224_22471

variables (x m n : ℝ)

theorem proof1 (x : ℝ) : (-3 * x - 5) * (5 - 3 * x) = 9 * x^2 - 25 :=
sorry

theorem proof2 (x : ℝ) : (-3 * x - 5) * (5 + 3 * x) = - (3 * x + 5) ^ 2 :=
sorry

theorem proof3 (m n : ℝ) : (2 * m - 3 * n + 1) * (2 * m + 1 + 3 * n) = (2 * m + 1) ^ 2 - (3 * n) ^ 2 :=
sorry

end proof1_proof2_proof3_l224_22471


namespace triangle_side_y_values_l224_22462

theorem triangle_side_y_values (y : ℕ) : (4 < y^2 ∧ y^2 < 20) ↔ (y = 3 ∨ y = 4) :=
by
  sorry

end triangle_side_y_values_l224_22462


namespace sum_of_three_numbers_l224_22435

theorem sum_of_three_numbers :
  ((3 : ℝ) / 8) + 0.125 + 9.51 = 10.01 :=
sorry

end sum_of_three_numbers_l224_22435


namespace total_pages_in_book_l224_22454

theorem total_pages_in_book (P : ℕ) 
  (h1 : 7 / 13 * P = P - 96 - 5 / 9 * (P - 7 / 13 * P))
  (h2 : 96 = 4 / 9 * (P - 7 / 13 * P)) : 
  P = 468 :=
 by 
    sorry

end total_pages_in_book_l224_22454


namespace arithmetic_sequence_ninth_term_l224_22495

theorem arithmetic_sequence_ninth_term (a d : ℤ) (h1 : a + 2 * d = 20) (h2 : a + 5 * d = 26) : a + 8 * d = 32 :=
sorry

end arithmetic_sequence_ninth_term_l224_22495


namespace restaurant_problem_l224_22447

theorem restaurant_problem (A K : ℕ) (h1 : A + K = 11) (h2 : 8 * A = 72) : K = 2 :=
by
  sorry

end restaurant_problem_l224_22447


namespace gcd_of_g_y_and_y_l224_22424

theorem gcd_of_g_y_and_y (y : ℤ) (h : 9240 ∣ y) : Int.gcd ((5 * y + 3) * (11 * y + 2) * (17 * y + 8) * (4 * y + 7)) y = 168 := by
  sorry

end gcd_of_g_y_and_y_l224_22424


namespace best_choice_for_square_formula_l224_22460

theorem best_choice_for_square_formula : 
  (89.8^2 = (90 - 0.2)^2) :=
by sorry

end best_choice_for_square_formula_l224_22460


namespace faye_pencils_l224_22494

theorem faye_pencils (rows crayons : ℕ) (pencils_per_row : ℕ) (h1 : rows = 7) (h2 : pencils_per_row = 5) : 
  (rows * pencils_per_row) = 35 :=
by {
  sorry
}

end faye_pencils_l224_22494


namespace ian_number_is_1021_l224_22477

-- Define the sequences each student skips
def alice_skips (n : ℕ) := ∃ k : ℕ, n = 4 * k
def barbara_skips (n : ℕ) := ∃ k : ℕ, n = 16 * (k + 1)
def candice_skips (n : ℕ) := ∃ k : ℕ, n = 64 * (k + 1)
-- Similar definitions for Debbie, Eliza, Fatima, Greg, and Helen

-- Define the condition under which Ian says a number
def ian_says (n : ℕ) :=
  ¬(alice_skips n) ∧ ¬(barbara_skips n) ∧ ¬(candice_skips n) -- and so on for Debbie, Eliza, Fatima, Greg, Helen

theorem ian_number_is_1021 : ian_says 1021 :=
by
  sorry

end ian_number_is_1021_l224_22477


namespace total_tickets_used_l224_22484

theorem total_tickets_used :
  let shooting_game_cost := 5
  let carousel_cost := 3
  let jen_games := 2
  let russel_rides := 3
  let jen_total := shooting_game_cost * jen_games
  let russel_total := carousel_cost * russel_rides
  jen_total + russel_total = 19 :=
by
  -- proof goes here
  sorry

end total_tickets_used_l224_22484


namespace cost_price_of_cricket_bat_for_A_l224_22408

-- Define the cost price of the cricket bat for A as a variable
variable (CP_A : ℝ)

-- Define the conditions given in the problem
def condition1 := CP_A * 1.20 -- B buys at 20% profit
def condition2 := CP_A * 1.20 * 1.25 -- B sells at 25% profit
def totalCost := 231 -- C pays $231

-- The theorem we need to prove
theorem cost_price_of_cricket_bat_for_A : (condition2 = totalCost) → CP_A = 154 := by
  intros h
  sorry

end cost_price_of_cricket_bat_for_A_l224_22408


namespace longest_boat_length_l224_22422

theorem longest_boat_length (a : ℝ) (c : ℝ) 
  (parallel_banks : ∀ x y : ℝ, (x = y) ∨ (x = -y)) 
  (right_angle_bend : ∃ b : ℝ, b = a) :
  c = 2 * a * Real.sqrt 2 := by
  sorry

end longest_boat_length_l224_22422


namespace all_integers_appear_exactly_once_l224_22449

noncomputable def sequence_of_integers (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, ∃ m : ℕ, a m > 0 ∧ ∃ m' : ℕ, a m' < 0

noncomputable def distinct_modulo_n (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, (∀ i j : ℕ, i < j ∧ j < n → a i % n ≠ a j % n)

theorem all_integers_appear_exactly_once
  (a : ℕ → ℤ)
  (h_seq : sequence_of_integers a)
  (h_distinct : distinct_modulo_n a) :
  ∀ x : ℤ, ∃! i : ℕ, a i = x := 
sorry

end all_integers_appear_exactly_once_l224_22449


namespace attendees_on_monday_is_10_l224_22445

-- Define the given conditions
def attendees_tuesday : ℕ := 15
def attendees_wed_thru_fri : ℕ := 10
def days_wed_thru_fri : ℕ := 3
def average_attendance : ℕ := 11
def total_days : ℕ := 5

-- Define the number of people who attended class on Monday
def attendees_tuesday_to_friday : ℕ := attendees_tuesday + attendees_wed_thru_fri * days_wed_thru_fri
def total_attendance : ℕ := average_attendance * total_days
def attendees_monday : ℕ := total_attendance - attendees_tuesday_to_friday

-- State the theorem
theorem attendees_on_monday_is_10 : attendees_monday = 10 :=
by
  -- Proof omitted
  sorry

end attendees_on_monday_is_10_l224_22445
