import Mathlib

namespace cost_price_of_computer_table_l1133_113357

theorem cost_price_of_computer_table (SP : ℝ) (CP : ℝ) (h : SP = CP * 1.24) (h_SP : SP = 8215) : CP = 6625 :=
by
  -- Start the proof block
  sorry -- Proof is not required as per the instructions

end cost_price_of_computer_table_l1133_113357


namespace sum_of_squares_l1133_113329

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 20) (h2 : a * b + b * c + c * a = 131) : 
  a^2 + b^2 + c^2 = 138 := 
sorry

end sum_of_squares_l1133_113329


namespace distance_between_trees_l1133_113317

-- Lean 4 statement for the proof problem
theorem distance_between_trees (n : ℕ) (yard_length : ℝ) (h_n : n = 26) (h_length : yard_length = 600) :
  yard_length / (n - 1) = 24 :=
by
  sorry

end distance_between_trees_l1133_113317


namespace find_value_of_m_l1133_113335

theorem find_value_of_m (x m : ℤ) (h₁ : x = 2) (h₂ : y = m) (h₃ : 3 * x + 2 * y = 10) : m = 2 := 
by
  sorry

end find_value_of_m_l1133_113335


namespace cardinal_transitivity_l1133_113378

variable {α β γ : Cardinal}

theorem cardinal_transitivity (h1 : α < β) (h2 : β < γ) : α < γ :=
  sorry

end cardinal_transitivity_l1133_113378


namespace sum_of_digits_base_2_315_l1133_113358

theorem sum_of_digits_base_2_315 :
  let binary_representation := "100111011"
  let digits_sum := binary_representation.toList.map (λ c => c.toNat - '0'.toNat)
  let sum_of_digits := digits_sum.sum
  sum_of_digits = 6 :=
by
  sorry

end sum_of_digits_base_2_315_l1133_113358


namespace distance_geologists_probability_l1133_113347

theorem distance_geologists_probability :
  let speed := 4 -- km/h
  let n_roads := 6
  let travel_time := 1 -- hour
  let distance_traveled := speed * travel_time -- km
  let distance_threshold := 6 -- km
  let n_outcomes := n_roads * n_roads
  let favorable_outcomes := 18 -- determined from the solution steps
  let probability := favorable_outcomes / n_outcomes
  probability = 0.5 := by
  sorry

end distance_geologists_probability_l1133_113347


namespace determine_a_l1133_113349

theorem determine_a (a : ℝ): (∃ b : ℝ, 27 * x^3 + 9 * x^2 + 36 * x + a = (3 * x + b)^3) → a = 8 := 
by
  sorry

end determine_a_l1133_113349


namespace rhombus_diagonal_l1133_113309

theorem rhombus_diagonal (side : ℝ) (short_diag : ℝ) (long_diag : ℝ) 
  (h1 : side = 37) (h2 : short_diag = 40) :
  long_diag = 62 :=
sorry

end rhombus_diagonal_l1133_113309


namespace sum_due_is_l1133_113382

-- Definitions and conditions from the problem
def BD : ℤ := 288
def TD : ℤ := 240
def face_value (FV : ℤ) : Prop := BD = TD + (TD * TD) / FV

-- Proof statement
theorem sum_due_is (FV : ℤ) (h : face_value FV) : FV = 1200 :=
sorry

end sum_due_is_l1133_113382


namespace decryption_proof_l1133_113395

-- Definitions
def Original_Message := "МОСКВА"
def Encrypted_Text_1 := "ТПЕОИРВНТМОЛАРГЕИАНВИЛЕДНМТААГТДЬТКУБЧКГЕИШНЕИАЯРЯ"
def Encrypted_Text_2 := "ЛСИЕМГОРТКРОМИТВАВКНОПКРАСЕОГНАЬЕП"
def Encrypted_Text_3 := "РТПАИОМВСВТИЕОБПРОЕННИИГЬКЕЕАМТАЛВТДЬСОУМЧШСЕОНШЬИАЯК"

noncomputable def Encrypted_Message_1 := "ЙМЫВОТСЬЛКЪГВЦАЯЯ"
noncomputable def Encrypted_Message_2 := "УКМАПОЧСРКЩВЗАХ"
noncomputable def Encrypted_Message_3 := "ШМФЭОГЧСЙЪКФЬВЫЕАКК"

def Decrypted_Message_1_and_3 := "ПОВТОРЕНИЕМАТЬУЧЕНИЯ"
def Decrypted_Message_2 := "СМОТРИВКОРЕНЬ"

-- Theorem statement
theorem decryption_proof :
  (Encrypted_Text_1 = Encrypted_Text_3 ∧ Original_Message = "МОСКВА" ∧ Encrypted_Message_1 = Encrypted_Message_3) →
  (Decrypted_Message_1_and_3 = "ПОВТОРЕНИЕМАТЬУЧЕНИЯ" ∧ Decrypted_Message_2 = "СМОТРИВКОРЕНЬ") :=
by 
  sorry

end decryption_proof_l1133_113395


namespace solve_for_x_l1133_113385

theorem solve_for_x (x : ℝ) (h :  9 / x^2 = x / 25) : x = 5 :=
by 
  sorry

end solve_for_x_l1133_113385


namespace total_canoes_built_by_End_of_May_l1133_113362

noncomputable def total_canoes_built (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem total_canoes_built_by_End_of_May :
  total_canoes_built 7 2 5 = 217 :=
by
  -- The proof would go here.
  sorry

end total_canoes_built_by_End_of_May_l1133_113362


namespace unknown_cube_edge_length_l1133_113354

theorem unknown_cube_edge_length (a b c x : ℕ) (h_a : a = 6) (h_b : b = 10) (h_c : c = 12) : a^3 + b^3 + x^3 = c^3 → x = 8 :=
by
  sorry

end unknown_cube_edge_length_l1133_113354


namespace length_of_AB_l1133_113367

open Real

noncomputable def line (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, (sqrt 3 / 2) * t)
noncomputable def curve (θ : ℝ) : ℝ × ℝ := (cos θ, sin θ)

theorem length_of_AB :
  ∃ A B : ℝ × ℝ, (∃ t : ℝ, line t = A) ∧ (∃ θ : ℝ, curve θ = A) ∧
                 (∃ t : ℝ, line t = B) ∧ (∃ θ : ℝ, curve θ = B) ∧
                 dist A B = 1 :=
by
  sorry

end length_of_AB_l1133_113367


namespace total_number_of_employees_l1133_113340
  
def part_time_employees : ℕ := 2041
def full_time_employees : ℕ := 63093
def total_employees : ℕ := part_time_employees + full_time_employees

theorem total_number_of_employees : total_employees = 65134 := by
  sorry

end total_number_of_employees_l1133_113340


namespace bake_sale_donation_l1133_113311

theorem bake_sale_donation :
  let total_earning := 400
  let cost_of_ingredients := 100
  let donation_homeless_piggy := 10
  let total_donation_homeless := 160
  let donation_homeless := total_donation_homeless - donation_homeless_piggy
  let available_for_donation := total_earning - cost_of_ingredients
  let donation_food_bank := available_for_donation - donation_homeless
  (donation_homeless / donation_food_bank) = 1 := 
by
  sorry

end bake_sale_donation_l1133_113311


namespace pickle_to_tomato_ratio_l1133_113389

theorem pickle_to_tomato_ratio 
  (mushrooms : ℕ) 
  (cherry_tomatoes : ℕ) 
  (pickles : ℕ) 
  (bacon_bits : ℕ) 
  (red_bacon_bits : ℕ) 
  (h1 : mushrooms = 3) 
  (h2 : cherry_tomatoes = 2 * mushrooms)
  (h3 : red_bacon_bits = 32)
  (h4 : bacon_bits = 3 * red_bacon_bits)
  (h5 : bacon_bits = 4 * pickles) : 
  pickles/cherry_tomatoes = 4 :=
by
  sorry

end pickle_to_tomato_ratio_l1133_113389


namespace total_dollars_l1133_113315

theorem total_dollars (mark_dollars : ℚ) (carolyn_dollars : ℚ) (mark_money : mark_dollars = 7 / 8) (carolyn_money : carolyn_dollars = 2 / 5) :
  mark_dollars + carolyn_dollars = 1.275 := sorry

end total_dollars_l1133_113315


namespace first_negative_term_position_l1133_113380

def a1 : ℤ := 1031
def d : ℤ := -3
def nth_term (n : ℕ) : ℤ := a1 + (n - 1 : ℤ) * d

theorem first_negative_term_position : ∃ n : ℕ, nth_term n < 0 ∧ n = 345 := 
by 
  -- Placeholder for proof
  sorry

end first_negative_term_position_l1133_113380


namespace total_votes_l1133_113364

-- Define the conditions
variable (V : ℝ) -- total number of votes polled
variable (w : ℝ) -- votes won by the winning candidate
variable (l : ℝ) -- votes won by the losing candidate
variable (majority : ℝ) -- majority votes

-- Define the specific values for the problem
def candidate_win_percentage (V : ℝ) : ℝ := 0.70 * V
def candidate_lose_percentage (V : ℝ) : ℝ := 0.30 * V

-- Define the majority condition
def majority_condition (V : ℝ) : Prop := (candidate_win_percentage V - candidate_lose_percentage V) = 240

-- The proof statement
theorem total_votes (V : ℝ) (h : majority_condition V) : V = 600 := by
  sorry

end total_votes_l1133_113364


namespace find_k_l1133_113350

noncomputable def S (n : ℕ) : ℤ := n^2 - 8 * n
noncomputable def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem find_k (k : ℕ) (h : a k = 5) : k = 7 := by
  sorry

end find_k_l1133_113350


namespace incorrect_arrangements_hello_l1133_113316

-- Given conditions: the word "hello" with letters 'h', 'e', 'l', 'l', 'o'
def letters : List Char := ['h', 'e', 'l', 'l', 'o']

-- The number of permutations of the letters in "hello" excluding the correct order
-- We need to prove that the number of incorrect arrangements is 59.
theorem incorrect_arrangements_hello : 
  (List.permutations letters).length - 1 = 59 := 
by sorry

end incorrect_arrangements_hello_l1133_113316


namespace probability_of_log2_condition_l1133_113302

noncomputable def probability_log_condition : ℝ :=
  let a := 0
  let b := 9
  let log_lower_bound := 1
  let log_upper_bound := 2
  let exp_lower_bound := 2^log_lower_bound
  let exp_upper_bound := 2^log_upper_bound
  (exp_upper_bound - exp_lower_bound) / (b - a)

theorem probability_of_log2_condition :
  probability_log_condition = 2 / 9 :=
by
  sorry

end probability_of_log2_condition_l1133_113302


namespace ratio_of_volumes_cone_cylinder_l1133_113374

theorem ratio_of_volumes_cone_cylinder (r h_cylinder : ℝ) (h_cone : ℝ) (h_radius : r = 4) (h_height_cylinder : h_cylinder = 12) (h_height_cone : h_cone = h_cylinder / 2) :
  ((1/3) * (π * r^2 * h_cone)) / (π * r^2 * h_cylinder) = 1 / 6 :=
by
  -- Definitions and assumptions are directly included from the conditions.
  sorry

end ratio_of_volumes_cone_cylinder_l1133_113374


namespace no_such_integers_l1133_113369

theorem no_such_integers :
  ¬ (∃ a b c d : ℤ, a * 19^3 + b * 19^2 + c * 19 + d = 1 ∧ a * 62^3 + b * 62^2 + c * 62 + d = 2) :=
by
  sorry

end no_such_integers_l1133_113369


namespace true_propositions_l1133_113339

theorem true_propositions :
  (∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 + 2*x - m = 0) ∧            -- Condition 1
  ((∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ∧                    -- Condition 2
   (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1) ) ∧
  (∀ x y : ℝ, (x * y ≠ 0) → (x ≠ 0 ∧ y ≠ 0)) ∧              -- Condition 3
  ¬ ( (∀ p q : Prop, ¬p → ¬ (p ∧ q)) ∧ (¬ ¬p → p ∧ q) ) ∧   -- Condition 4
  (∃ x : ℝ, x^2 + x + 3 ≤ 0)                                 -- Condition 5
:= by {
  sorry
}

end true_propositions_l1133_113339


namespace fraction_of_males_l1133_113332

theorem fraction_of_males (M F : ℝ) (h1 : M + F = 1) (h2 : (7/8 * M + 9/10 * (1 - M)) = 0.885) :
  M = 0.6 :=
sorry

end fraction_of_males_l1133_113332


namespace min_value_2a_b_c_l1133_113321

theorem min_value_2a_b_c (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * (a + b + c) + b * c = 4) : 
  2 * a + b + c ≥ 4 :=
sorry

end min_value_2a_b_c_l1133_113321


namespace valid_votes_for_candidate_a_l1133_113363

theorem valid_votes_for_candidate_a (total_votes : ℕ) (invalid_percentage : ℝ) (candidate_a_percentage : ℝ) (valid_votes_a : ℝ) :
  total_votes = 560000 ∧ invalid_percentage = 0.15 ∧ candidate_a_percentage = 0.80 →
  valid_votes_a = (candidate_a_percentage * (1 - invalid_percentage) * total_votes) := 
sorry

end valid_votes_for_candidate_a_l1133_113363


namespace root_of_quadratic_eq_l1133_113384

theorem root_of_quadratic_eq : 
  ∃ x₁ x₂ : ℝ, (x₁ = 0 ∧ x₂ = 2) ∧ ∀ x : ℝ, x^2 - 2 * x = 0 → (x = x₁ ∨ x = x₂) :=
by
  sorry

end root_of_quadratic_eq_l1133_113384


namespace runner_distance_l1133_113318

theorem runner_distance (track_length race_length : ℕ) (A_speed B_speed C_speed : ℚ)
  (h1 : track_length = 400) (h2 : race_length = 800)
  (h3 : A_speed = 1) (h4 : B_speed = 8 / 7) (h5 : C_speed = 6 / 7) :
  ∃ distance_from_finish : ℚ, distance_from_finish = 200 :=
by {
  -- We are not required to provide the actual proof steps, just setting up the definitions and initial statements for the proof.
  sorry
}

end runner_distance_l1133_113318


namespace train_speed_l1133_113361

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 630) (h_time : time = 36) :
  (length / 1000) / (time / 3600) = 63 :=
by
  rw [h_length, h_time]
  sorry

end train_speed_l1133_113361


namespace square_projection_exists_l1133_113328

structure Point :=
(x y : Real)

structure Line :=
(a b c : Real) -- Line equation ax + by + c = 0

def is_on_line (P : Point) (l : Line) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

theorem square_projection_exists (P : Point) (l : Line) :
  ∃ (A B C D : Point), 
  is_on_line A l ∧ 
  is_on_line B l ∧
  (A.x + B.x) / 2 = P.x ∧ 
  (A.y + B.y) / 2 = P.y ∧ 
  (A.x = B.x ∨ A.y = B.y) ∧ -- assuming one of the sides lies along the line
  (C.x + D.x) / 2 = P.x ∧ 
  (C.y + D.y) / 2 = P.y ∧ 
  C ≠ A ∧ C ≠ B ∧ D ≠ A ∧ D ≠ B :=
sorry

end square_projection_exists_l1133_113328


namespace max_distinct_values_is_two_l1133_113325

-- Definitions of non-negative numbers and conditions
variable (a b c d : ℝ)
variable (ha : 0 ≤ a)
variable (hb : 0 ≤ b)
variable (hc : 0 ≤ c)
variable (hd : 0 ≤ d)
variable (h1 : Real.sqrt (a + b) + Real.sqrt (c + d) = Real.sqrt (a + c) + Real.sqrt (b + d))
variable (h2 : Real.sqrt (a + c) + Real.sqrt (b + d) = Real.sqrt (a + d) + Real.sqrt (b + c))

-- Theorem stating that the maximum number of distinct values among a, b, c, d is 2.
theorem max_distinct_values_is_two : 
  ∃ (u v : ℝ), 0 ≤ u ∧ 0 ≤ v ∧ (u = a ∨ u = b ∨ u = c ∨ u = d) ∧ (v = a ∨ v = b ∨ v = c ∨ v = d) ∧ 
  ∀ (x y : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d) → (y = a ∨ y = b ∨ y = c ∨ y = d) → x = y ∨ x = u ∨ x = v :=
sorry

end max_distinct_values_is_two_l1133_113325


namespace base_area_functional_relationship_base_area_when_height_4_8_l1133_113320

noncomputable def cylinder_base_area (h : ℝ) : ℝ := 24 / h

theorem base_area_functional_relationship (h : ℝ) (H : h ≠ 0) :
  cylinder_base_area h = 24 / h := by
  unfold cylinder_base_area
  rfl

theorem base_area_when_height_4_8 :
  cylinder_base_area 4.8 = 5 := by
  unfold cylinder_base_area
  norm_num

end base_area_functional_relationship_base_area_when_height_4_8_l1133_113320


namespace dagger_example_l1133_113326

def dagger (m n p q : ℚ) : ℚ := 2 * m * p * (q / n)

theorem dagger_example : dagger 5 8 3 4 = 15 := by
  sorry

end dagger_example_l1133_113326


namespace square_area_from_circles_l1133_113390

theorem square_area_from_circles :
  (∀ (r : ℝ), r = 7 → ∀ (n : ℕ), n = 4 → (∃ (side_length : ℝ), side_length = 2 * (2 * r))) →
  ∀ (side_length : ℝ), side_length = 28 →
  (∃ (area : ℝ), area = side_length * side_length ∧ area = 784) :=
sorry

end square_area_from_circles_l1133_113390


namespace remainder_when_divided_l1133_113306

theorem remainder_when_divided (P K Q R K' Q' S' T : ℕ)
  (h1 : P = K * Q + R)
  (h2 : Q = K' * Q' + S')
  (h3 : R * Q' = T) :
  P % (K * K') = K * S' + (T / Q') :=
by
  sorry

end remainder_when_divided_l1133_113306


namespace N_is_composite_l1133_113397

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ (Nat.Prime N) :=
by
  sorry

end N_is_composite_l1133_113397


namespace find_a_l1133_113310

theorem find_a (a : ℝ) (h1 : ∀ x : ℝ, a^(2*x - 4) ≤ 2^(x^2 - 2*x)) (ha_pos : a > 0) (ha_neq1 : a ≠ 1) : a = 2 :=
sorry

end find_a_l1133_113310


namespace triangle_inequality_equivalence_l1133_113330

theorem triangle_inequality_equivalence
    (a b c : ℝ) :
  (a < b + c ∧ b < a + c ∧ c < a + b) ↔
  (|b - c| < a ∧ a < b + c ∧ |a - c| < b ∧ b < a + c ∧ |a - b| < c ∧ c < a + b) ∧
  (max a (max b c) < b + c ∧ max a (max b c) < a + c ∧ max a (max b c) < a + b) :=
by sorry

end triangle_inequality_equivalence_l1133_113330


namespace solution1_solution2_solution3_l1133_113371

noncomputable def problem1 : Nat :=
  (1) * (2 - 1) * (2 + 1)

theorem solution1 : problem1 = 3 := by
  sorry

noncomputable def problem2 : Nat :=
  (2) * (2 + 1) * (2^2 + 1)

theorem solution2 : problem2 = 15 := by
  sorry

noncomputable def problem3 : Nat :=
  (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)

theorem solution3 : problem3 = 2^64 - 1 := by
  sorry

end solution1_solution2_solution3_l1133_113371


namespace composite_number_N_l1133_113300

theorem composite_number_N (y : ℕ) (hy : y > 1) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = (y ^ 125 - 1) / (3 ^ 22 - 1) :=
by
  -- use sorry to skip the proof
  sorry

end composite_number_N_l1133_113300


namespace solve_eqn_l1133_113308

noncomputable def root_expr (a b k x : ℝ) : ℝ := Real.sqrt ((a + b * Real.sqrt k)^x)

theorem solve_eqn: {x : ℝ | root_expr 3 2 2 x + root_expr 3 (-2) 2 x = 6} = {2, -2} :=
by
  sorry

end solve_eqn_l1133_113308


namespace known_number_is_24_l1133_113370

noncomputable def HCF (a b : ℕ) : ℕ := sorry
noncomputable def LCM (a b : ℕ) : ℕ := sorry

theorem known_number_is_24 (A B : ℕ) (h1 : B = 182)
  (h2 : HCF A B = 14)
  (h3 : LCM A B = 312) : A = 24 := by
  sorry

end known_number_is_24_l1133_113370


namespace product_in_M_l1133_113342

def M : Set ℤ := {x | ∃ (a b : ℤ), x = a^2 - b^2}

theorem product_in_M (p q : ℤ) (hp : p ∈ M) (hq : q ∈ M) : p * q ∈ M :=
by
  sorry

end product_in_M_l1133_113342


namespace number_of_plants_l1133_113303

--- The given problem conditions and respective proof setup
axiom green_leaves_per_plant : ℕ
axiom yellow_turn_fall_off : ℕ
axiom green_leaves_total : ℕ

def one_third (n : ℕ) : ℕ := n / 3

-- Specify the given conditions
axiom leaves_per_plant_cond : green_leaves_per_plant = 18
axiom fall_off_cond : yellow_turn_fall_off = one_third green_leaves_per_plant
axiom total_leaves_cond : green_leaves_total = 36

-- Proof statement for the number of tea leaf plants
theorem number_of_plants : 
  (green_leaves_per_plant - yellow_turn_fall_off) * 3 = green_leaves_total :=
by
  sorry

end number_of_plants_l1133_113303


namespace range_of_m_l1133_113337

theorem range_of_m {m : ℝ} (h : ∀ x : ℝ, (3 * m - 1) ^ x = (3 * m - 1) ^ x ∧ (3 * m - 1) > 0 ∧ (3 * m - 1) < 1) :
  1 / 3 < m ∧ m < 2 / 3 :=
by
  sorry

end range_of_m_l1133_113337


namespace martin_family_ice_cream_cost_l1133_113336

theorem martin_family_ice_cream_cost (R : ℤ)
  (kiddie_scoop_cost : ℤ) (double_scoop_cost : ℤ)
  (total_cost : ℤ) :
  kiddie_scoop_cost = 3 → 
  double_scoop_cost = 6 → 
  total_cost = 32 →
  2 * R + 2 * kiddie_scoop_cost + 3 * double_scoop_cost = total_cost →
  R = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end martin_family_ice_cream_cost_l1133_113336


namespace tangent_line_values_l1133_113392

theorem tangent_line_values (m : ℝ) :
  (∃ s : ℝ, 3 * s^2 = 12 ∧ 12 * s + m = s^3 - 2) ↔ (m = -18 ∨ m = 14) :=
by
  sorry

end tangent_line_values_l1133_113392


namespace problem_statement_l1133_113365

noncomputable def theta (h1 : 2 * Real.cos θ + Real.sin θ = 0) (h2 : 0 < θ ∧ θ < Real.pi) : Real :=
θ

noncomputable def varphi (h4 : Real.pi / 2 < φ ∧ φ < Real.pi) : Real :=
φ

theorem problem_statement
  (θ : Real) (φ : Real)
  (h1 : 2 * Real.cos θ + Real.sin θ = 0)
  (h2 : 0 < θ ∧ θ < Real.pi)
  (h3 : Real.sin (θ - φ) = Real.sqrt 10 / 10)
  (h4 : Real.pi / 2 < φ ∧ φ < Real.pi) :
  Real.tan θ = -2 ∧
  Real.sin θ = (2 * Real.sqrt 5) / 5 ∧
  Real.cos θ = -Real.sqrt 5 / 5 ∧
  Real.cos φ = -Real.sqrt 2 / 10 :=
by
  sorry

end problem_statement_l1133_113365


namespace angela_height_l1133_113375

def height_of_Amy : ℕ := 150
def height_of_Helen : ℕ := height_of_Amy + 3
def height_of_Angela : ℕ := height_of_Helen + 4

theorem angela_height : height_of_Angela = 157 := by
  sorry

end angela_height_l1133_113375


namespace n_plus_floor_sqrt2_plus1_pow_n_is_odd_l1133_113381

theorem n_plus_floor_sqrt2_plus1_pow_n_is_odd (n : ℕ) (h : n > 0) : 
  Odd (n + ⌊(Real.sqrt 2 + 1) ^ n⌋) :=
by sorry

end n_plus_floor_sqrt2_plus1_pow_n_is_odd_l1133_113381


namespace determine_k_l1133_113352

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + 2 * k * x + 1

-- State the problem
theorem determine_k (k : ℝ) :
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, f k x ≤ 4) ∧ (∃ x ∈ Set.Icc (-3 : ℝ) 2, f k x = 4)
  ↔ (k = 3 / 8 ∨ k = -3) :=
by
  sorry

end determine_k_l1133_113352


namespace min_triangle_area_l1133_113360

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 2 = 1
noncomputable def circle_with_diameter_passing_origin (A B : ℝ × ℝ) : Prop :=
  let O := (0, 0)
  let d := (A.1 - B.1)^2 + (A.2 - B.2)^2
  let center := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  center.1^2 + center.2^2 = d / 4

theorem min_triangle_area (A B : ℝ × ℝ)
    (hA : hyperbola A.1 A.2)
    (hB : hyperbola B.1 B.2)
    (hc : circle_with_diameter_passing_origin A B) : 
    ∃ (S : ℝ), S = 2 :=
sorry

end min_triangle_area_l1133_113360


namespace niko_total_profit_l1133_113343

noncomputable def calculate_total_profit : ℝ :=
  let pairs := 9
  let price_per_pair := 2
  let discount_rate := 0.10
  let shipping_cost := 5
  let profit_4_pairs := 0.25
  let profit_5_pairs := 0.20
  let tax_rate := 0.05
  let cost_socks := pairs * price_per_pair
  let discount := discount_rate * cost_socks
  let cost_after_discount := cost_socks - discount
  let total_cost := cost_after_discount + shipping_cost
  let resell_price_4_pairs := (price_per_pair * (1 + profit_4_pairs)) * 4
  let resell_price_5_pairs := (price_per_pair * (1 + profit_5_pairs)) * 5
  let total_resell_price := resell_price_4_pairs + resell_price_5_pairs
  let sales_tax := tax_rate * total_resell_price
  let total_resell_price_after_tax := total_resell_price + sales_tax
  let total_profit := total_resell_price_after_tax - total_cost
  total_profit

theorem niko_total_profit : calculate_total_profit = 0.85 :=
by
  sorry

end niko_total_profit_l1133_113343


namespace part1_part2_l1133_113366

-- Part 1: Number of k-tuples of ordered subsets with empty intersection
theorem part1 (S : Finset α) (n k : ℕ) (h : S.card = n) (hk : 0 < k) :
  (∃ (f : Fin (n) → Fin (2^k - 1)), true) :=
sorry

-- Part 2: Number of k-tuples of subsets with chain condition
theorem part2 (S : Finset α) (n k : ℕ) (h : S.card = n) (hk : 0 < k) :
  (S.card = (k + 1)^n) :=
sorry

end part1_part2_l1133_113366


namespace painting_time_l1133_113377

-- Definitions translated from conditions
def total_weight_tons := 5
def weight_per_ball_kg := 4
def number_of_students := 10
def balls_per_student_per_6_minutes := 5

-- Derived Definitions
def total_weight_kg := total_weight_tons * 1000
def total_balls := total_weight_kg / weight_per_ball_kg
def balls_painted_by_all_students_per_6_minutes := number_of_students * balls_per_student_per_6_minutes
def required_intervals := total_balls / balls_painted_by_all_students_per_6_minutes
def total_time_minutes := required_intervals * 6

-- The theorem statement
theorem painting_time : total_time_minutes = 150 := by
  sorry

end painting_time_l1133_113377


namespace fine_per_day_of_absence_l1133_113386

theorem fine_per_day_of_absence :
  ∃ x: ℝ, ∀ (total_days work_wage total_received_days absent_days: ℝ),
  total_days = 30 →
  work_wage = 10 →
  total_received_days = 216 →
  absent_days = 7 →
  (total_days - absent_days) * work_wage - (absent_days * x) = total_received_days :=
sorry

end fine_per_day_of_absence_l1133_113386


namespace kitty_vacuum_time_l1133_113379

theorem kitty_vacuum_time
  (weekly_toys : ℕ := 5)
  (weekly_windows : ℕ := 15)
  (weekly_furniture : ℕ := 10)
  (total_cleaning_time : ℕ := 200)
  (weeks : ℕ := 4)
  : (weekly_toys + weekly_windows + weekly_furniture) * weeks < total_cleaning_time ∧ ((total_cleaning_time - ((weekly_toys + weekly_windows + weekly_furniture) * weeks)) / weeks = 20)
  := by
  sorry

end kitty_vacuum_time_l1133_113379


namespace largest_e_l1133_113388

variable (a b c d e : ℤ)

theorem largest_e 
  (h1 : a - 1 = b + 2) 
  (h2 : a - 1 = c - 3)
  (h3 : a - 1 = d + 4)
  (h4 : a - 1 = e - 6) 
  : e > a ∧ e > b ∧ e > c ∧ e > d := 
sorry

end largest_e_l1133_113388


namespace width_of_carton_is_25_l1133_113353

-- Definitions for the given problem
def carton_width := 25
def carton_length := 60
def width_or_height := min carton_width carton_length

theorem width_of_carton_is_25 : width_or_height = 25 := by
  sorry

end width_of_carton_is_25_l1133_113353


namespace unique_integer_sequence_l1133_113331

theorem unique_integer_sequence (a : ℕ → ℤ) :
  a 1 = 1 ∧ a 2 > 1 ∧ (∀ n ≥ 1, a (n + 1)^3 + 1 = a n * a (n + 2)) →
  ∃! (a : ℕ → ℤ), a 1 = 1 ∧ a 2 > 1 ∧ (∀ n ≥ 1, a (n + 1)^3 + 1 = a n * a (n + 2)) :=
sorry

end unique_integer_sequence_l1133_113331


namespace inequality_b_2pow_a_a_2pow_neg_b_l1133_113368

theorem inequality_b_2pow_a_a_2pow_neg_b (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : 
  b * 2^a + a * 2^(-b) ≥ a + b :=
sorry

end inequality_b_2pow_a_a_2pow_neg_b_l1133_113368


namespace Amy_balloons_l1133_113394

-- Defining the conditions
def James_balloons : ℕ := 1222
def more_balloons : ℕ := 208

-- Defining Amy's balloons as a proof goal
theorem Amy_balloons : ∀ (Amy_balloons : ℕ), James_balloons - more_balloons = Amy_balloons → Amy_balloons = 1014 :=
by
  intros Amy_balloons h
  sorry

end Amy_balloons_l1133_113394


namespace cos_theta_plus_5π_div_6_l1133_113391

theorem cos_theta_plus_5π_div_6 (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2)
  (hcond : Real.sin (θ / 2 + π / 6) = 3 / 5) :
  Real.cos (θ + 5 * π / 6) = -24 / 25 :=
by
  sorry -- Proof is skipped as instructed

end cos_theta_plus_5π_div_6_l1133_113391


namespace abs_gt_implies_nec_not_suff_l1133_113313

theorem abs_gt_implies_nec_not_suff {a b : ℝ} : 
  (|a| > b) → (∀ (a b : ℝ), a > b → |a| > b) ∧ ¬(∀ (a b : ℝ), |a| > b → a > b) :=
by
  sorry

end abs_gt_implies_nec_not_suff_l1133_113313


namespace solution1_solution2_l1133_113383

noncomputable def problem1 (a : ℝ) : Prop :=
  (∃ x : ℝ, -2 < x ∧ x < 2 ∧ x^2 - 2*x - a = 0) ∨
  (∃ x : ℝ, x^2 + (a-1)*x + 4 < 0)

theorem solution1 (a : ℝ) : problem1 a ↔ a < -3 ∨ a ≥ -1 := 
  sorry

noncomputable def problem2 (a : ℝ) (x : ℝ) : Prop :=
  (-2 < x ∧ x < 2 ∧ x^2 - 2*x - a = 0)

noncomputable def condition2 (a x : ℝ) : Prop :=
  (2*a < x ∧ x < a+1)

theorem solution2 (a : ℝ) : (∀ x, condition2 a x → problem2 a x) → a ≥ -1/2 :=
  sorry

end solution1_solution2_l1133_113383


namespace sides_of_regular_polygon_l1133_113305

theorem sides_of_regular_polygon 
    (sum_interior_angles : ∀ n : ℕ, (n - 2) * 180 = 1440) :
  ∃ n : ℕ, n = 10 :=
by
  sorry

end sides_of_regular_polygon_l1133_113305


namespace johns_tour_program_days_l1133_113398

/-- John has Rs 360 for his expenses. If he exceeds his days by 4 days, he must cut down daily expenses by Rs 3. Prove that the number of days of John's tour program is 20. -/
theorem johns_tour_program_days
    (d e : ℕ)
    (h1 : 360 = e * d)
    (h2 : 360 = (e - 3) * (d + 4)) : 
    d = 20 := 
  sorry

end johns_tour_program_days_l1133_113398


namespace necessary_but_not_sufficient_condition_for_inequality_l1133_113393

theorem necessary_but_not_sufficient_condition_for_inequality 
    {a b c : ℝ} (h : a * c^2 ≥ b * c^2) : ¬(a > b → (a * c^2 < b * c^2)) :=
by
  sorry

end necessary_but_not_sufficient_condition_for_inequality_l1133_113393


namespace new_ticket_price_l1133_113301

theorem new_ticket_price (a : ℕ) (x : ℝ) (initial_price : ℝ) (revenue_increase : ℝ) (spectator_increase : ℝ)
  (h₀ : initial_price = 25)
  (h₁ : spectator_increase = 1.5)
  (h₂ : revenue_increase = 1.14)
  (h₃ : x = 0.76):
  initial_price * x = 19 :=
by
  sorry

end new_ticket_price_l1133_113301


namespace linear_equation_l1133_113314

noncomputable def is_linear (k : ℝ) : Prop :=
  2 * (|k|) = 1 ∧ k ≠ 1

theorem linear_equation (k : ℝ) : is_linear k ↔ k = -1 :=
by
  sorry

end linear_equation_l1133_113314


namespace sum_zero_opposites_l1133_113334

theorem sum_zero_opposites {a b : ℝ} (h : a + b = 0) : a = -b :=
by sorry

end sum_zero_opposites_l1133_113334


namespace equal_roots_h_l1133_113355

theorem equal_roots_h (h : ℝ) : (∀ x : ℝ, 3 * x^2 - 4 * x + h / 3 = 0) ↔ h = 4 := by
  -- proof goes here
  sorry

end equal_roots_h_l1133_113355


namespace concert_tickets_full_price_revenue_l1133_113344

theorem concert_tickets_full_price_revenue :
  ∃ (f p d : ℕ), f + d = 200 ∧ f * p + d * (p / 3) = 2688 ∧ f * p = 2128 :=
by
  -- We need to find the solution steps are correct to establish the existence
  sorry

end concert_tickets_full_price_revenue_l1133_113344


namespace g_g_g_3_equals_107_l1133_113307

def g (x : ℝ) : ℝ := 3 * x + 2

theorem g_g_g_3_equals_107 : g (g (g 3)) = 107 := 
by 
  sorry

end g_g_g_3_equals_107_l1133_113307


namespace right_angled_triangle_lines_l1133_113348

theorem right_angled_triangle_lines (m : ℝ) :
  (∀ x y : ℝ, 2 * x - y + 4 = 0 → x - 2 * y + 5 = 0 → m * x - 3 * y + 12 = 0 → 
    (exists x₁ y₁ : ℝ, 2 * x₁ - 1 * y₁ + 4 = 0 ∧ (x₁ - 5) ^ 2 / 4 + y₁ / (4) = (2^(1/2))^2) ∨ 
    (exists x₂ y₂ : ℝ, 1/2 * x₂ * y₂ - y₂ / 3 + 1 / 6 = 0 ∧ (x₂ + 5) ^ 2 / 9 + y₂ / 4 = small)) → 
    (m = -3 / 2 ∨ m = -6) :=
sorry

end right_angled_triangle_lines_l1133_113348


namespace find_z_l1133_113322

-- Given conditions as Lean definitions
def consecutive (x y z : ℕ) : Prop := x = z + 2 ∧ y = z + 1 ∧ x > y ∧ y > z
def equation (x y z : ℕ) : Prop := 2 * x + 3 * y + 3 * z = 5 * y + 11

-- The statement to be proven
theorem find_z (x y z : ℕ) (h1 : consecutive x y z) (h2 : equation x y z) : z = 3 :=
sorry

end find_z_l1133_113322


namespace ellipse_equation_x_axis_foci_ellipse_equation_y_axis_foci_l1133_113312

theorem ellipse_equation_x_axis_foci (x y : ℝ) (h : 3 * x + 4 * y = 12) :
  ∃ a b c : ℝ, c = 4 ∧ b = 3 ∧ a = 5 ∧ (x^2 / a^2) + (y^2 / b^2) = 1 := by
  sorry

theorem ellipse_equation_y_axis_foci (x y : ℝ) (h : 3 * x + 4 * y = 12) :
  ∃ a b c : ℝ, c = 3 ∧ b = 4 ∧ a = 5 ∧ (x^2 / b^2) + (y^2 / a^2) = 1 := by
  sorry

end ellipse_equation_x_axis_foci_ellipse_equation_y_axis_foci_l1133_113312


namespace find_digit_l1133_113319

theorem find_digit {x : ℕ} (hx : x = 7) : (10 * (x - 3) + x) = 47 :=
by
  sorry

end find_digit_l1133_113319


namespace g_f_3_eq_1476_l1133_113324

def f (x : ℝ) : ℝ := x^3 - 2 * x + 1
def g (x : ℝ) : ℝ := 3 * x^2 + x + 2

theorem g_f_3_eq_1476 : g (f 3) = 1476 :=
by
  sorry

end g_f_3_eq_1476_l1133_113324


namespace study_days_needed_l1133_113387

theorem study_days_needed :
  let math_chapters := 4
  let math_worksheets := 7
  let physics_chapters := 5
  let physics_worksheets := 9
  let chemistry_chapters := 6
  let chemistry_worksheets := 8

  let math_chapter_hours := 2.5
  let math_worksheet_hours := 1.5
  let physics_chapter_hours := 3.0
  let physics_worksheet_hours := 2.0
  let chemistry_chapter_hours := 3.5
  let chemistry_worksheet_hours := 1.75

  let daily_study_hours := 7.0
  let breaks_first_3_hours := 3 * 10 / 60.0
  let breaks_next_3_hours := 3 * 15 / 60.0
  let breaks_final_hour := 1 * 20 / 60.0
  let snack_breaks := 2 * 20 / 60.0
  let lunch_break := 45 / 60.0

  let break_time_per_day := breaks_first_3_hours + breaks_next_3_hours + breaks_final_hour + snack_breaks + lunch_break
  let effective_study_time_per_day := daily_study_hours - break_time_per_day

  let total_math_hours := (math_chapters * math_chapter_hours) + (math_worksheets * math_worksheet_hours)
  let total_physics_hours := (physics_chapters * physics_chapter_hours) + (physics_worksheets * physics_worksheet_hours)
  let total_chemistry_hours := (chemistry_chapters * chemistry_chapter_hours) + (chemistry_worksheets * chemistry_worksheet_hours)

  let total_study_hours := total_math_hours + total_physics_hours + total_chemistry_hours
  let total_study_days := total_study_hours / effective_study_time_per_day
  
  total_study_days.ceil = 23 := by sorry

end study_days_needed_l1133_113387


namespace binom_20_10_l1133_113323

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k + 1 => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

theorem binom_20_10 :
  binom 18 8 = 43758 →
  binom 18 9 = 48620 →
  binom 18 10 = 43758 →
  binom 20 10 = 184756 :=
by
  intros h₁ h₂ h₃
  sorry

end binom_20_10_l1133_113323


namespace stratified_sampling_l1133_113376

theorem stratified_sampling :
  let total_employees := 150
  let middle_managers := 30
  let senior_managers := 10
  let selected_employees := 30
  let selection_probability := selected_employees / total_employees
  let selected_middle_managers := middle_managers * selection_probability
  let selected_senior_managers := senior_managers * selection_probability
  selected_middle_managers = 6 ∧ selected_senior_managers = 2 :=
by
  sorry

end stratified_sampling_l1133_113376


namespace largest_whole_x_l1133_113346

theorem largest_whole_x (x : ℕ) (h : 11 * x < 150) : x ≤ 13 :=
sorry

end largest_whole_x_l1133_113346


namespace visits_per_hour_l1133_113304

open Real

theorem visits_per_hour (price_per_visit : ℝ) (hours_per_day : ℕ) (days_per_month : ℕ) (total_earnings : ℝ) 
  (h_price : price_per_visit = 0.10)
  (h_hours : hours_per_day = 24)
  (h_days : days_per_month = 30)
  (h_earnings : total_earnings = 3600) :
  (total_earnings / (price_per_visit * hours_per_day * days_per_month) : ℝ) = 50 :=
by
  sorry

end visits_per_hour_l1133_113304


namespace find_quadratic_function_find_vertex_find_range_l1133_113399

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

def satisfies_points (a b c : ℝ) : Prop :=
  quadratic_function a b c (-1) = 0 ∧
  quadratic_function a b c 0 = -3 ∧
  quadratic_function a b c 2 = -3

theorem find_quadratic_function : ∃ a b c, satisfies_points a b c ∧ (a = 1 ∧ b = -2 ∧ c = -3) :=
sorry

theorem find_vertex (a b c : ℝ) (h : a = 1 ∧ b = -2 ∧ c = -3) :
  ∃ x y, x = 1 ∧ y = -4 ∧ ∀ x', x' > 1 → quadratic_function a b c x' > quadratic_function a b c x :=
sorry

theorem find_range (a b c : ℝ) (h : a = 1 ∧ b = -2 ∧ c = -3) :
  ∀ x, -1 < x ∧ x < 2 → -4 < quadratic_function a b c x ∧ quadratic_function a b c x < 0 :=
sorry

end find_quadratic_function_find_vertex_find_range_l1133_113399


namespace cleaning_times_l1133_113356

theorem cleaning_times (A B C : ℕ) (hA : A = 40) (hB : B = A / 4) (hC : C = 2 * B) : 
  B = 10 ∧ C = 20 := by
  sorry

end cleaning_times_l1133_113356


namespace cost_of_book_l1133_113338

-- Definitions based on the conditions
def cost_pen : ℕ := 4
def cost_ruler : ℕ := 1
def fifty_dollar_bill : ℕ := 50
def change_received : ℕ := 20
def total_spent : ℕ := fifty_dollar_bill - change_received

-- Problem Statement: Prove the cost of the book
theorem cost_of_book : ∀ (cost_pen cost_ruler total_spent : ℕ), 
  total_spent = 50 - 20 → cost_pen = 4 → cost_ruler = 1 →
  (total_spent - (cost_pen + cost_ruler) = 25) :=
by
  intros cost_pen cost_ruler total_spent h1 h2 h3
  sorry

end cost_of_book_l1133_113338


namespace num_natural_a_l1133_113333

theorem num_natural_a (a b : ℕ) : 
  (a^2 + a + 100 = b^2) → ∃ n : ℕ, n = 4 := sorry

end num_natural_a_l1133_113333


namespace john_needs_more_usd_l1133_113396

noncomputable def additional_usd (needed_eur needed_sgd : ℝ) (has_usd has_jpy : ℝ) : ℝ :=
  let eur_to_usd := 1 / 0.84
  let sgd_to_usd := 1 / 1.34
  let jpy_to_usd := 1 / 110.35
  let total_needed_usd := needed_eur * eur_to_usd + needed_sgd * sgd_to_usd
  let total_has_usd := has_usd + has_jpy * jpy_to_usd
  total_needed_usd - total_has_usd

theorem john_needs_more_usd :
  ∀ (needed_eur needed_sgd : ℝ) (has_usd has_jpy : ℝ),
    needed_eur = 7.50 → needed_sgd = 5.00 → has_usd = 2.00 → has_jpy = 500 →
    additional_usd needed_eur needed_sgd has_usd has_jpy = 6.13 :=
by
  intros needed_eur needed_sgd has_usd has_jpy
  intros hneeded_eur hneeded_sgd hhas_usd hhas_jpy
  unfold additional_usd
  rw [hneeded_eur, hneeded_sgd, hhas_usd, hhas_jpy]
  sorry

end john_needs_more_usd_l1133_113396


namespace smallest_x_solution_l1133_113345

def smallest_x_condition (x : ℝ) : Prop :=
  (x^2 - 5 * x - 84 = (x - 12) * (x + 7)) ∧
  (x ≠ 9) ∧
  (x ≠ -7) ∧
  ((x^2 - 5 * x - 84) / (x - 9) = 4 / (x + 7))

theorem smallest_x_solution :
  ∃ x : ℝ, smallest_x_condition x ∧ ∀ y : ℝ, smallest_x_condition y → x ≤ y :=
sorry

end smallest_x_solution_l1133_113345


namespace range_of_x_in_sqrt_x_plus_3_l1133_113372

theorem range_of_x_in_sqrt_x_plus_3 (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 :=
by
  sorry

end range_of_x_in_sqrt_x_plus_3_l1133_113372


namespace range_of_p_l1133_113359

theorem range_of_p 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n : ℕ, S n = (-1 : ℝ)^n * a n + 1/(2^n) + n - 3)
  (h2 : ∀ n : ℕ, (a (n + 1) - p) * (a n - p) < 0) :
  -3/4 < p ∧ p < 11/4 :=
sorry

end range_of_p_l1133_113359


namespace tailor_charges_30_per_hour_l1133_113373

noncomputable def tailor_hourly_rate (shirts pants : ℕ) (shirt_hours pant_hours total_cost : ℝ) :=
  total_cost / (shirts * shirt_hours + pants * pant_hours)

theorem tailor_charges_30_per_hour :
  tailor_hourly_rate 10 12 1.5 3 1530 = 30 := by
  sorry

end tailor_charges_30_per_hour_l1133_113373


namespace amount_of_money_l1133_113327

theorem amount_of_money (x y : ℝ) 
  (h1 : x + 1/2 * y = 50) 
  (h2 : 2/3 * x + y = 50) : 
  (x + 1/2 * y = 50) ∧ (2/3 * x + y = 50) :=
by
  exact ⟨h1, h2⟩ 

end amount_of_money_l1133_113327


namespace m_value_if_linear_l1133_113341

theorem m_value_if_linear (m : ℝ) (x : ℝ) (h : (m + 2) * x^(|m| - 1) + 8 = 0) (linear : |m| - 1 = 1) : m = 2 :=
sorry

end m_value_if_linear_l1133_113341


namespace two_false_propositions_l1133_113351

theorem two_false_propositions (a : ℝ) :
  (¬((a > -3) → (a > -6))) ∧ (¬((a > -6) → (a > -3))) → (¬(¬(a > -3) → ¬(a > -6))) :=
by
  sorry

end two_false_propositions_l1133_113351
