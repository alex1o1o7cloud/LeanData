import Mathlib

namespace simplify_expr_l1086_108600

theorem simplify_expr (a b : ℝ) (h₁ : a + b = 0) (h₂ : a ≠ b) : (1 - a) + (1 - b) = 2 := by
  sorry

end simplify_expr_l1086_108600


namespace melted_ice_cream_depth_l1086_108622

noncomputable def radius_sphere : ℝ := 3
noncomputable def radius_cylinder : ℝ := 10
noncomputable def height_cylinder : ℝ := 36 / 100

theorem melted_ice_cream_depth :
  (4 / 3) * Real.pi * radius_sphere^3 = Real.pi * radius_cylinder^2 * height_cylinder :=
by
  sorry

end melted_ice_cream_depth_l1086_108622


namespace min_cost_to_package_fine_arts_collection_l1086_108660

theorem min_cost_to_package_fine_arts_collection :
  let box_length := 20
  let box_width := 20
  let box_height := 12
  let cost_per_box := 0.50
  let required_volume := 1920000
  let volume_of_one_box := box_length * box_width * box_height
  let number_of_boxes := required_volume / volume_of_one_box
  let total_cost := number_of_boxes * cost_per_box
  total_cost = 200 := 
by
  sorry

end min_cost_to_package_fine_arts_collection_l1086_108660


namespace ned_initial_lives_l1086_108625

-- Define the initial number of lives Ned had
def initial_lives (start_lives current_lives lost_lives : ℕ) : ℕ :=
  current_lives + lost_lives

-- Define the conditions
def current_lives := 70
def lost_lives := 13

-- State the theorem
theorem ned_initial_lives : initial_lives current_lives current_lives lost_lives = 83 := by
  sorry

end ned_initial_lives_l1086_108625


namespace find_f2_g2_l1086_108698

def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x
def equation (f g : ℝ → ℝ) : Prop := ∀ x : ℝ, f x - g x = x^3 + 2^(-x)

theorem find_f2_g2 (f g : ℝ → ℝ)
  (h1 : even_function f)
  (h2 : odd_function g)
  (h3 : equation f g) :
  f 2 + g 2 = -2 :=
sorry

end find_f2_g2_l1086_108698


namespace find_s5_l1086_108659

noncomputable def s (a b x y : ℝ) (n : ℕ) : ℝ :=
if n = 1 then (a * x + b * y) else
if n = 2 then (a * x^2 + b * y^2) else
if n = 3 then (a * x^3 + b * y^3) else
if n = 4 then (a * x^4 + b * y^4) else
if n = 5 then (a * x^5 + b * y^5) else 0

theorem find_s5 
  (a b x y : ℝ) :
  s a b x y 1 = 5 →
  s a b x y 2 = 11 →
  s a b x y 3 = 24 →
  s a b x y 4 = 58 →
  s a b x y 5 = 262.88 :=
by
  intros h1 h2 h3 h4
  sorry

end find_s5_l1086_108659


namespace price_increase_percentage_l1086_108697

variables
  (coffees_daily_before : ℕ := 4)
  (price_per_coffee_before : ℝ := 2)
  (coffees_daily_after : ℕ := 2)
  (price_increase_savings : ℝ := 2)
  (spending_before := coffees_daily_before * price_per_coffee_before)
  (spending_after := spending_before - price_increase_savings)
  (price_per_coffee_after := spending_after / coffees_daily_after)

theorem price_increase_percentage :
  ((price_per_coffee_after - price_per_coffee_before) / price_per_coffee_before) * 100 = 50 :=
by
  sorry

end price_increase_percentage_l1086_108697


namespace sum_le_two_of_cubics_sum_to_two_l1086_108692

theorem sum_le_two_of_cubics_sum_to_two (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^3 + b^3 = 2) : a + b ≤ 2 := 
sorry

end sum_le_two_of_cubics_sum_to_two_l1086_108692


namespace arithmetic_sequence_sum_l1086_108646

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_a7 : a 7 = 12) :
  a 3 + a 11 = 24 :=
by
  sorry

end arithmetic_sequence_sum_l1086_108646


namespace geese_in_marsh_l1086_108606

theorem geese_in_marsh (number_of_ducks : ℕ) (total_number_of_birds : ℕ) (number_of_geese : ℕ) (h1 : number_of_ducks = 37) (h2 : total_number_of_birds = 95) : 
  number_of_geese = 58 := 
by
  sorry

end geese_in_marsh_l1086_108606


namespace principal_amount_l1086_108645

theorem principal_amount (SI : ℝ) (R : ℝ) (T : ℕ) (P : ℝ) :
  SI = 3.45 → R = 0.05 → T = 3 → SI = P * R * T → P = 23 :=
by
  -- The proof steps would go here but are omitted as specified.
  sorry

end principal_amount_l1086_108645


namespace no_distinct_positive_integers_l1086_108638

noncomputable def P (x : ℕ) : ℕ := x^2000 - x^1000 + 1

theorem no_distinct_positive_integers (a : Fin 2001 → ℕ) (h_distinct : Function.Injective a) :
  ¬ (∀ i j, i ≠ j → a i * a j ∣ P (a i) * P (a j)) :=
sorry

end no_distinct_positive_integers_l1086_108638


namespace machines_initially_working_l1086_108613

theorem machines_initially_working (N x : ℕ) (h1 : N * 4 * R = x)
  (h2 : 20 * 6 * R = 3 * x) : N = 10 :=
by
  sorry

end machines_initially_working_l1086_108613


namespace evaluate_expression_l1086_108691

theorem evaluate_expression : 4 * 12 + 5 * 11 + 6^2 + 7 * 9 = 202 :=
by sorry

end evaluate_expression_l1086_108691


namespace problem_statement_l1086_108680

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

variable (f g : ℝ → ℝ)

axiom f_odd : odd_function f
axiom f_neg : ∀ x : ℝ, x < 0 → f x = x^3 - 1
axiom f_pos : ∀ x : ℝ, x > 0 → f x = g x

theorem problem_statement : f (-1) + g 2 = 7 :=
by
  sorry

end problem_statement_l1086_108680


namespace friends_count_l1086_108688

-- Define that Laura has 28 blocks
def blocks := 28

-- Define that each friend gets 7 blocks
def blocks_per_friend := 7

-- The proof statement we want to prove
theorem friends_count : blocks / blocks_per_friend = 4 := by
  sorry

end friends_count_l1086_108688


namespace inverse_proposition_l1086_108669

   theorem inverse_proposition (x a b : ℝ) :
     (x ≥ a^2 + b^2 → x ≥ 2 * a * b) →
     (x ≥ 2 * a * b → x ≥ a^2 + b^2) :=
   sorry
   
end inverse_proposition_l1086_108669


namespace one_meter_to_leaps_l1086_108654

theorem one_meter_to_leaps 
  (x y z w u v : ℕ)
  (h1 : x * leaps = y * strides) 
  (h2 : z * bounds = w * leaps) 
  (h3 : u * bounds = v * meters) :
  1 * meters = (uw / vz) * leaps :=
sorry

end one_meter_to_leaps_l1086_108654


namespace find_line_eq_of_given_conditions_l1086_108601

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * y + 5 = 0
def line_perpendicular (a b : ℝ) : Prop := a + b + 1 = 0
def is_center (x y : ℝ) : Prop := (x, y) = (0, 3)
def is_eq_of_line (x y : ℝ) : Prop := x - y + 3 = 0

theorem find_line_eq_of_given_conditions (x y : ℝ) (h1 : circle_eq x y) (h2 : line_perpendicular x y) (h3 : is_center x y) : is_eq_of_line x y :=
by
  sorry

end find_line_eq_of_given_conditions_l1086_108601


namespace tan_theta_is_sqrt3_div_5_l1086_108619

open Real

theorem tan_theta_is_sqrt3_div_5 (theta : ℝ) (h : 2 * sin (theta + π / 3) = 3 * sin (π / 3 - theta)) :
  tan theta = sqrt 3 / 5 :=
sorry

end tan_theta_is_sqrt3_div_5_l1086_108619


namespace proof_ineq_l1086_108634

noncomputable def P (f g : ℤ → ℤ) (m n k : ℕ) :=
  (∀ x y : ℤ, -1000 ≤ x ∧ x ≤ 1000 ∧ -1000 ≤ y ∧ y ≤ 1000 ∧ f x = g y → m = m + 1) ∧
  (∀ x y : ℤ, -1000 ≤ x ∧ x ≤ 1000 ∧ -1000 ≤ y ∧ y ≤ 1000 ∧ f x = f y → n = n + 1) ∧
  (∀ x y : ℤ, -1000 ≤ x ∧ x ≤ 1000 ∧ -1000 ≤ y ∧ y ≤ 1000 ∧ g x = g y → k = k + 1)

theorem proof_ineq (f g : ℤ → ℤ) (m n k : ℕ) (h : P f g m n k) : 
  2 * m ≤ n + k :=
  sorry

end proof_ineq_l1086_108634


namespace game_show_possible_guesses_l1086_108624

theorem game_show_possible_guesses : 
  (∃ A B C : ℕ, 
    A + B + C = 8 ∧ 
    A > 0 ∧ B > 0 ∧ C > 0 ∧ 
    (A = 1 ∨ A = 4) ∧
    (B = 1 ∨ B = 4) ∧
    (C = 1 ∨ C = 4) ) →
  (number_of_possible_guesses : ℕ) = 210 :=
sorry

end game_show_possible_guesses_l1086_108624


namespace part_I_part_II_l1086_108636

open Real

noncomputable def f (x : ℝ) : ℝ := log ((2 / (x + 1)) - 1)

def g (x a : ℝ) : ℝ := -x^2 + 2 * x + a

-- Domain of function f
def A : Set ℝ := {x | -1 < x ∧ x < 1}

-- Range of function g with a given condition on x
def B (a : ℝ) : Set ℝ := {y | ∃ x, 0 ≤ x ∧ x ≤ 3 ∧ y = g x a}

theorem part_I : f (1 / 2015) + f (-1 / 2015) = 0 := sorry

theorem part_II (a : ℝ) : (A ∩ B a) = ∅ ↔ a ≤ -2 ∨ a ≥ 4 := sorry

end part_I_part_II_l1086_108636


namespace find_angle_C_l1086_108678

theorem find_angle_C 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : 10 * a * Real.cos B = 3 * b * Real.cos A) 
  (h2 : Real.cos A = (5 * Real.sqrt 26) / 26) 
  (h3 : A + B + C = π) : 
  C = (3 * π) / 4 :=
sorry

end find_angle_C_l1086_108678


namespace right_triangle_sin_sum_l1086_108648

/--
In a right triangle ABC with ∠A = 90°, prove that sin A + sin^2 B + sin^2 C = 2.
-/
theorem right_triangle_sin_sum (A B C : ℝ) (hA : A = 90) (hABC : A + B + C = 180) :
  Real.sin (A * π / 180) + Real.sin (B * π / 180) ^ 2 + Real.sin (C * π / 180) ^ 2 = 2 :=
sorry

end right_triangle_sin_sum_l1086_108648


namespace max_value_of_squares_l1086_108674

theorem max_value_of_squares (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 8) :
  a^2 + b^2 + c^2 + d^2 ≤ 4 :=
sorry

end max_value_of_squares_l1086_108674


namespace total_students_surveyed_l1086_108676

-- Define the constants for liked and disliked students.
def liked_students : ℕ := 235
def disliked_students : ℕ := 165

-- The theorem to prove the total number of students surveyed.
theorem total_students_surveyed : liked_students + disliked_students = 400 :=
by
  -- The proof will go here.
  sorry

end total_students_surveyed_l1086_108676


namespace determine_set_A_l1086_108611

variable (U : Set ℕ) (A : Set ℕ)

theorem determine_set_A (hU : U = {0, 1, 2, 3}) (hcompl : U \ A = {2}) :
  A = {0, 1, 3} :=
by
  sorry

end determine_set_A_l1086_108611


namespace rectangle_area_l1086_108694

theorem rectangle_area :
  ∃ (a b : ℕ), a ≠ b ∧ Even a ∧ (a * b = 3 * (2 * a + 2 * b)) ∧ (a * b = 162) :=
by
  sorry

end rectangle_area_l1086_108694


namespace burt_net_profit_l1086_108631

theorem burt_net_profit
  (cost_seeds : ℝ := 2.00)
  (cost_soil : ℝ := 8.00)
  (num_plants : ℕ := 20)
  (price_per_plant : ℝ := 5.00) :
  let total_cost := cost_seeds + cost_soil
  let total_revenue := num_plants * price_per_plant
  let net_profit := total_revenue - total_cost
  net_profit = 90.00 :=
by sorry

end burt_net_profit_l1086_108631


namespace median_a_sq_correct_sum_of_medians_sq_l1086_108690

noncomputable def median_a_sq (a b c : ℝ) := (2 * b^2 + 2 * c^2 - a^2) / 4
noncomputable def median_b_sq (a b c : ℝ) := (2 * a^2 + 2 * c^2 - b^2) / 4
noncomputable def median_c_sq (a b c : ℝ) := (2 * a^2 + 2 * b^2 - c^2) / 4

theorem median_a_sq_correct (a b c : ℝ) : 
  median_a_sq a b c = (2 * b^2 + 2 * c^2 - a^2) / 4 :=
sorry

theorem sum_of_medians_sq (a b c : ℝ) :
  median_a_sq a b c + median_b_sq a b c + median_c_sq a b c = 
  3 * (a^2 + b^2 + c^2) / 4 :=
sorry

end median_a_sq_correct_sum_of_medians_sq_l1086_108690


namespace law_of_sines_proof_l1086_108609

noncomputable def law_of_sines (a b c α β γ : ℝ) :=
  (a / Real.sin α = b / Real.sin β) ∧
  (b / Real.sin β = c / Real.sin γ) ∧
  (α + β + γ = Real.pi)

theorem law_of_sines_proof (a b c α β γ : ℝ) (h : law_of_sines a b c α β γ) :
  (a = b * Real.cos γ + c * Real.cos β) ∧
  (b = c * Real.cos α + a * Real.cos γ) ∧
  (c = a * Real.cos β + b * Real.cos α) :=
sorry

end law_of_sines_proof_l1086_108609


namespace remainder_division_39_l1086_108683

theorem remainder_division_39 (N : ℕ) (k m R1 : ℕ) (hN1 : N = 39 * k + R1) (hN2 : N % 13 = 5) (hR1_lt_39 : R1 < 39) :
  R1 = 5 :=
by sorry

end remainder_division_39_l1086_108683


namespace cost_of_parts_l1086_108684

theorem cost_of_parts (C : ℝ) 
  (h1 : ∀ n ∈ List.range 60, (1.4 * C * n) = (1.4 * C * 60))
  (h2 : 5000 + 3000 = 8000)
  (h3 : 60 * C * 1.4 - (60 * C + 8000) = 11200) : 
  C = 800 := by
  sorry

end cost_of_parts_l1086_108684


namespace power_addition_l1086_108664

theorem power_addition {a m n : ℝ} (h1 : a^m = 2) (h2 : a^n = 8) : a^(m + n) = 16 :=
sorry

end power_addition_l1086_108664


namespace tickets_needed_l1086_108656

def tickets_per_roller_coaster : ℕ := 5
def tickets_per_giant_slide : ℕ := 3
def roller_coaster_rides : ℕ := 7
def giant_slide_rides : ℕ := 4

theorem tickets_needed : tickets_per_roller_coaster * roller_coaster_rides + tickets_per_giant_slide * giant_slide_rides = 47 := 
by
  sorry

end tickets_needed_l1086_108656


namespace complex_arithmetic_l1086_108682

def Q : ℂ := 7 + 3 * Complex.I
def E : ℂ := 2 * Complex.I
def D : ℂ := 7 - 3 * Complex.I
def F : ℂ := 1 + Complex.I

theorem complex_arithmetic : (Q * E * D) + F = 1 + 117 * Complex.I := by
  sorry

end complex_arithmetic_l1086_108682


namespace sufficient_but_not_necessary_l1086_108643

variables {p q : Prop}

theorem sufficient_but_not_necessary :
  (p → q) ∧ (¬q → ¬p) ∧ ¬(q → p) → (¬q → ¬p) ∧ (¬(q → p)) :=
by
  sorry

end sufficient_but_not_necessary_l1086_108643


namespace find_original_number_l1086_108677

/-- Given that one less than the reciprocal of a number is 5/2, the original number must be -2/3. -/
theorem find_original_number (y : ℚ) (h : 1 - 1 / y = 5 / 2) : y = -2 / 3 :=
sorry

end find_original_number_l1086_108677


namespace max_digits_product_l1086_108642

def digitsProduct (A B : ℕ) : ℕ := A * B

theorem max_digits_product 
  (A B : ℕ) 
  (h1 : A + B + 5 ≡ 0 [MOD 9]) 
  (h2 : 0 ≤ A ∧ A ≤ 9) 
  (h3 : 0 ≤ B ∧ B ≤ 9) 
  : digitsProduct A B = 42 := 
sorry

end max_digits_product_l1086_108642


namespace total_steps_needed_l1086_108629

def cycles_needed (dist : ℕ) : ℕ := dist
def steps_per_cycle : ℕ := 5
def effective_steps_per_pattern : ℕ := 1

theorem total_steps_needed (dist : ℕ) (h : dist = 66) : 
  steps_per_cycle * cycles_needed dist = 330 :=
by 
  -- Placeholder for proof
  sorry

end total_steps_needed_l1086_108629


namespace find_a_l1086_108696

theorem find_a :
  ∃ a : ℝ, 
    (∀ x : ℝ, f x = 3 * x + a * x^3) ∧ 
    (f 1 = a + 3) ∧ 
    (∃ k : ℝ, k = 6 ∧ k = deriv f 1 ∧ ((∀ x : ℝ, deriv f x = 3 + 3 * a * x^2))) → 
    a = 1 :=
by sorry

end find_a_l1086_108696


namespace arrange_abc_l1086_108639

noncomputable def a : ℝ := Real.log (4) / Real.log (0.3)
noncomputable def b : ℝ := Real.log (0.2) / Real.log (0.3)
noncomputable def c : ℝ := (1 / Real.exp 1) ^ Real.pi

theorem arrange_abc (a := a) (b := b) (c := c) : b > c ∧ c > a := by
  sorry

end arrange_abc_l1086_108639


namespace minutes_before_4_angle_same_as_4_l1086_108633

def hour_hand_angle_at_4 := 120
def minute_hand_angle_at_4 := 0
def minute_hand_angle_per_minute := 6
def hour_hand_angle_per_minute := 0.5

theorem minutes_before_4_angle_same_as_4 :
  ∃ m : ℚ, abs (hour_hand_angle_at_4 - 5.5 * m) = hour_hand_angle_at_4 ∧ 
           (60 - m) = 21 + 9 / 11 := by
  sorry

end minutes_before_4_angle_same_as_4_l1086_108633


namespace largest_is_C_l1086_108614

def A : ℝ := 0.978
def B : ℝ := 0.9719
def C : ℝ := 0.9781
def D : ℝ := 0.917
def E : ℝ := 0.9189

theorem largest_is_C : 
  (C > A) ∧ 
  (C > B) ∧ 
  (C > D) ∧ 
  (C > E) := by
  sorry

end largest_is_C_l1086_108614


namespace tangent_line_at_1_tangent_line_through_2_3_l1086_108666

-- Define the function
def f (x : ℝ) : ℝ := x^3

-- Derivative of the function
def f' (x : ℝ) : ℝ := 3 * x^2

-- Problem 1: Prove that the tangent line at point (1, 1) is y = 3x - 2
theorem tangent_line_at_1 (x y : ℝ) (h : y = f 1 + f' 1 * (x - 1)) : y = 3 * x - 2 := 
sorry

-- Problem 2: Prove that the tangent line passing through (2/3, 0) is either y = 0 or y = 3x - 2
theorem tangent_line_through_2_3 (x y x0 : ℝ) 
  (hx0 : y = f x0 + f' x0 * (x - x0))
  (hp : 0 = f' x0 * (2/3 - x0)) :
  y = 0 ∨ y = 3 * x - 2 := 
sorry

end tangent_line_at_1_tangent_line_through_2_3_l1086_108666


namespace find_N_l1086_108644

theorem find_N : ∃ (N : ℕ), (1000 ≤ N ∧ N < 10000) ∧ (N^2 % 10000 = N) ∧ (N % 16 = 7) ∧ N = 3751 := 
by sorry

end find_N_l1086_108644


namespace survey_is_sample_of_population_l1086_108679

-- Definitions based on the conditions in a)
def population_size := 50000
def sample_size := 2000
def is_comprehensive_survey := false
def is_sampling_survey := true
def is_population_student (n : ℕ) : Prop := n ≤ population_size
def is_individual_unit (n : ℕ) : Prop := n ≤ sample_size

-- Theorem that encapsulates the proof problem
theorem survey_is_sample_of_population : is_sampling_survey ∧ ∃ n, is_individual_unit n :=
by
  sorry

end survey_is_sample_of_population_l1086_108679


namespace intersection_is_correct_l1086_108672

def setA := {x : ℝ | 3 * x - x^2 > 0}
def setB := {x : ℝ | x ≤ 1}

theorem intersection_is_correct : 
  setA ∩ setB = {x | 0 < x ∧ x ≤ 1} :=
sorry

end intersection_is_correct_l1086_108672


namespace benny_spent_on_baseball_gear_l1086_108621

theorem benny_spent_on_baseball_gear (initial_amount left_over spent : ℕ) 
  (h_initial : initial_amount = 67) 
  (h_left : left_over = 33) 
  (h_spent : spent = initial_amount - left_over) : 
  spent = 34 :=
by
  rw [h_initial, h_left] at h_spent
  exact h_spent

end benny_spent_on_baseball_gear_l1086_108621


namespace marbles_total_l1086_108650

def marbles_initial := 22
def marbles_given := 20

theorem marbles_total : marbles_initial + marbles_given = 42 := by
  sorry

end marbles_total_l1086_108650


namespace inequality_relations_l1086_108640

variable {R : Type} [OrderedAddCommGroup R]
variables (x y z : R)

theorem inequality_relations (h1 : x - y > x + z) (h2 : x + y < y + z) : y < -z ∧ x < z :=
by
  sorry

end inequality_relations_l1086_108640


namespace karl_sticker_count_l1086_108655

theorem karl_sticker_count : 
  ∀ (K R B : ℕ), 
    (R = K + 20) → 
    (B = R - 10) → 
    (K + R + B = 105) → 
    K = 25 := 
by
  intros K R B hR hB hSum
  sorry

end karl_sticker_count_l1086_108655


namespace simplify_expression_l1086_108616

theorem simplify_expression (x : ℝ) : (x + 1) ^ 2 + x * (x - 2) = 2 * x ^ 2 + 1 :=
by
  sorry

end simplify_expression_l1086_108616


namespace wombat_clawing_l1086_108665

variable (W : ℕ)
variable (R : ℕ := 1)

theorem wombat_clawing :
    (9 * W + 3 * R = 39) → (W = 4) :=
by 
  sorry

end wombat_clawing_l1086_108665


namespace point_in_or_on_circle_l1086_108626

theorem point_in_or_on_circle (θ : Real) :
  let P := (5 * Real.cos θ, 4 * Real.sin θ)
  let C_eq := ∀ (x y : Real), x^2 + y^2 = 25
  25 * Real.cos θ ^ 2 + 16 * Real.sin θ ^ 2 ≤ 25 := 
by 
  sorry

end point_in_or_on_circle_l1086_108626


namespace value_of_k_l1086_108608

theorem value_of_k (k : ℤ) : (1/2)^(22) * (1/(81 : ℝ))^k = 1/(18 : ℝ)^(22) → k = 11 :=
by
  sorry

end value_of_k_l1086_108608


namespace repeating_decimals_sum_l1086_108670

theorem repeating_decimals_sum :
  let x := (0.2222222222 : ℚ)
          -- Repeating decimal 0.222... represented up to some precision in rational form.
          -- Of course, internally it is understood with perpetuity.
  let y := (0.0303030303 : ℚ)
          -- Repeating decimal 0.0303... represented up to some precision in rational form.
  x + y = 25 / 99 :=
by
  let x := 2 / 9
  let y := 1 / 33
  sorry

end repeating_decimals_sum_l1086_108670


namespace train_speed_excluding_stoppages_l1086_108695

-- Define the speed of the train excluding stoppages and including stoppages
variables (S : ℕ) -- S is the speed of the train excluding stoppages
variables (including_stoppages_speed : ℕ := 40) -- The speed including stoppages is 40 kmph

-- The train stops for 20 minutes per hour. This means it runs for (60 - 20) minutes per hour.
def running_time_per_hour := 40

-- Converting 40 minutes to hours
def running_fraction_of_hour : ℚ := 40 / 60

-- Formulate the main theorem:
theorem train_speed_excluding_stoppages
    (H1 : including_stoppages_speed = 40)
    (H2 : running_fraction_of_hour = 2 / 3) :
    S = 60 :=
by
    sorry

end train_speed_excluding_stoppages_l1086_108695


namespace area_of_rectangle_l1086_108602

-- Define the problem statement and conditions
theorem area_of_rectangle (p d : ℝ) :
  ∃ A : ℝ, (∀ (x y : ℝ), 2 * x + 2 * y = p ∧ x^2 + y^2 = d^2 → A = x * y) →
  A = (p^2 - 4 * d^2) / 8 :=
by 
  sorry

end area_of_rectangle_l1086_108602


namespace set_intersection_l1086_108612

theorem set_intersection (M N : Set ℝ) (hM : M = {x | x < 3}) (hN : N = {x | x > 2}) :
  M ∩ N = {x | 2 < x ∧ x < 3} :=
sorry

end set_intersection_l1086_108612


namespace cube_sum_identity_l1086_108675

theorem cube_sum_identity (p q r : ℝ)
  (h₁ : p + q + r = 4)
  (h₂ : pq + qr + rp = 6)
  (h₃ : pqr = -8) :
  p^3 + q^3 + r^3 = 64 := 
by
  sorry

end cube_sum_identity_l1086_108675


namespace smallest_n_with_digits_315_l1086_108673

-- Defining the conditions
def relatively_prime (m n : ℕ) := Nat.gcd m n = 1
def valid_fraction (m n : ℕ) := (m < n) ∧ relatively_prime m n

-- Predicate for the sequence 3, 1, 5 in the decimal representation of m/n
def contains_digits_315 (m n : ℕ) : Prop :=
  ∃ k d : ℕ, 10^k * m % n = 315 * 10^(d - 3) ∧ d ≥ 3

-- The main theorem: smallest n for which the conditions are satisfied
theorem smallest_n_with_digits_315 :
  ∃ n : ℕ, valid_fraction m n ∧ contains_digits_315 m n ∧ n = 159 :=
sorry

end smallest_n_with_digits_315_l1086_108673


namespace total_tiles_number_l1086_108632

-- Define the conditions based on the problem statement
def square_floor_tiles (s : ℕ) : ℕ := s * s

def black_tiles_count (s : ℕ) : ℕ := 3 * s - 3

-- The main theorem statement: given the number of black tiles as 201,
-- prove that the total number of tiles is 4624
theorem total_tiles_number (s : ℕ) (h₁ : black_tiles_count s = 201) : 
  square_floor_tiles s = 4624 :=
by
  -- This is where the proof would go
  sorry

end total_tiles_number_l1086_108632


namespace largest_difference_l1086_108623

def A := 3 * 1005^1006
def B := 1005^1006
def C := 1004 * 1005^1005
def D := 3 * 1005^1005
def E := 1005^1005
def F := 1005^1004

theorem largest_difference : 
  A - B > B - C ∧ 
  A - B > C - D ∧ 
  A - B > D - E ∧ 
  A - B > E - F :=
by
  sorry

end largest_difference_l1086_108623


namespace ratio_of_distances_l1086_108687

-- Define the speeds and times for ferries P and Q
def speed_P : ℝ := 8
def time_P : ℝ := 3
def speed_Q : ℝ := speed_P + 1
def time_Q : ℝ := time_P + 5

-- Define the distances covered by ferries P and Q
def distance_P : ℝ := speed_P * time_P
def distance_Q : ℝ := speed_Q * time_Q

-- The statement to prove: the ratio of the distances
theorem ratio_of_distances : distance_Q / distance_P = 3 :=
sorry

end ratio_of_distances_l1086_108687


namespace area_of_circle_below_line_l1086_108661

theorem area_of_circle_below_line (x y : ℝ) :
  (x - 3)^2 + (y - 5)^2 = 9 →
  y ≤ 8 →
  ∃ (A : ℝ), A = 9 * Real.pi :=
sorry

end area_of_circle_below_line_l1086_108661


namespace no_even_sum_of_four_consecutive_in_circle_l1086_108653

theorem no_even_sum_of_four_consecutive_in_circle (n : ℕ) (h1 : n = 2018) :
  ¬ ∃ (f : ℕ → ℕ), (∀ i, 1 ≤ f i ∧ f i ≤ n) ∧ (∀ i, i < n → (f (i % n) + f ((i + 1) % n) + f ((i + 2) % n) + f ((i + 3) % n)) % 2 = 1) :=
by { sorry }

end no_even_sum_of_four_consecutive_in_circle_l1086_108653


namespace count_divisible_by_90_four_digit_numbers_l1086_108651

theorem count_divisible_by_90_four_digit_numbers :
  ∃ (n : ℕ), (n = 10) ∧ (∀ (x : ℕ), 1000 ≤ x ∧ x < 10000 ∧ x % 90 = 0 ∧ x % 100 = 90 → (x = 1890 ∨ x = 2790 ∨ x = 3690 ∨ x = 4590 ∨ x = 5490 ∨ x = 6390 ∨ x = 7290 ∨ x = 8190 ∨ x = 9090 ∨ x = 9990)) :=
by
  sorry

end count_divisible_by_90_four_digit_numbers_l1086_108651


namespace max_k_no_real_roots_max_integer_value_k_no_real_roots_l1086_108604

-- Define the quadratic equation with the condition on the discriminant.
theorem max_k_no_real_roots : ∀ k : ℤ, (4 + 4 * (k : ℝ) < 0) ↔ k < -1 := sorry

-- Prove that the maximum integer value of k satisfying this condition is -2.
theorem max_integer_value_k_no_real_roots : ∃ k_max : ℤ, k_max ∈ { k : ℤ | 4 + 4 * (k : ℝ) < 0 } ∧ ∀ k' : ℤ, k' ∈ { k : ℤ | 4 + 4 * (k : ℝ) < 0 } → k' ≤ k_max :=
sorry

end max_k_no_real_roots_max_integer_value_k_no_real_roots_l1086_108604


namespace nine_a_minus_six_b_l1086_108641

-- Define the variables and conditions.
variables (a b : ℚ)

-- Assume the given conditions.
def condition1 : Prop := 3 * a + 4 * b = 0
def condition2 : Prop := a = 2 * b - 3

-- Formalize the statement to prove.
theorem nine_a_minus_six_b (h1 : condition1 a b) (h2 : condition2 a b) : 9 * a - 6 * b = -81 / 5 :=
sorry

end nine_a_minus_six_b_l1086_108641


namespace davonte_ran_further_than_mercedes_l1086_108671

-- Conditions
variable (jonathan_distance : ℝ) (mercedes_distance : ℝ) (davonte_distance : ℝ)

-- Given conditions
def jonathan_ran := jonathan_distance = 7.5
def mercedes_ran_twice_jonathan := mercedes_distance = 2 * jonathan_distance
def mercedes_and_davonte_total := mercedes_distance + davonte_distance = 32

-- Prove the distance Davonte ran farther than Mercedes is 2 kilometers
theorem davonte_ran_further_than_mercedes :
  jonathan_ran jonathan_distance ∧
  mercedes_ran_twice_jonathan jonathan_distance mercedes_distance ∧
  mercedes_and_davonte_total mercedes_distance davonte_distance →
  davonte_distance - mercedes_distance = 2 :=
by
  sorry

end davonte_ran_further_than_mercedes_l1086_108671


namespace weight_of_apples_l1086_108610

-- Definitions based on conditions
def total_weight : ℕ := 10
def weight_orange : ℕ := 1
def weight_grape : ℕ := 3
def weight_strawberry : ℕ := 3

-- Prove that the weight of apples is 3 kilograms
theorem weight_of_apples : (total_weight - (weight_orange + weight_grape + weight_strawberry)) = 3 :=
by
  sorry

end weight_of_apples_l1086_108610


namespace find_a_5_l1086_108686

def arithmetic_sequence (a : ℕ → ℤ) := 
  ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d

def sum_first_n (a : ℕ → ℤ) (S : ℕ → ℤ) := 
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem find_a_5 {a : ℕ → ℤ} {S : ℕ → ℤ}
  (h_seq : arithmetic_sequence a)
  (h_S6 : S 6 = 3)
  (h_a4 : a 4 = 2)
  (h_sum_first_n : sum_first_n a S) :
  a 5 = 5 := 
sorry

end find_a_5_l1086_108686


namespace ratio_new_average_to_original_l1086_108689

theorem ratio_new_average_to_original (scores : List ℝ) (h_len : scores.length = 50) :
  let A := (scores.sum / scores.length : ℝ)
  let new_sum := scores.sum + 2 * A
  let new_avg := new_sum / (scores.length + 2)
  new_avg / A = 1 := 
by
  sorry

end ratio_new_average_to_original_l1086_108689


namespace avg_three_numbers_l1086_108628

theorem avg_three_numbers (A B C : ℝ) 
  (h1 : A + B = 53)
  (h2 : B + C = 69)
  (h3 : A + C = 58) : 
  (A + B + C) / 3 = 30 := 
by
  sorry

end avg_three_numbers_l1086_108628


namespace cross_number_puzzle_digit_star_l1086_108685

theorem cross_number_puzzle_digit_star :
  ∃ N₁ N₂ N₃ N₄ : ℕ,
    N₁ % 1000 / 100 = 4 ∧ N₁ % 10 = 1 ∧ ∃ n : ℕ, N₁ = n ^ 2 ∧
    N₃ % 1000 / 100 = 6 ∧ ∃ m : ℕ, N₃ = m ^ 4 ∧
    ∃ p : ℕ, N₂ = 2 * p ^ 5 ∧ 100 ≤ N₂ ∧ N₂ < 1000 ∧
    N₄ % 10 = 5 ∧ ∃ q : ℕ, N₄ = q ^ 3 ∧ 100 ≤ N₄ ∧ N₄ < 1000 ∧
    (N₁ % 10 = 4) :=
by
  sorry

end cross_number_puzzle_digit_star_l1086_108685


namespace annual_income_earned_by_both_investments_l1086_108647

noncomputable def interest (principal: ℝ) (rate: ℝ) (time: ℝ) : ℝ :=
  principal * rate * time

theorem annual_income_earned_by_both_investments :
  let total_amount := 8000
  let first_investment := 3000
  let first_interest_rate := 0.085
  let second_interest_rate := 0.064
  let second_investment := total_amount - first_investment
  interest first_investment first_interest_rate 1 + interest second_investment second_interest_rate 1 = 575 :=
by
  sorry

end annual_income_earned_by_both_investments_l1086_108647


namespace aisha_probability_l1086_108630

noncomputable def prob_one_head (prob_tail : ℝ) (num_coins : ℕ) : ℝ :=
  1 - (prob_tail ^ num_coins)

theorem aisha_probability : 
  prob_one_head (1/2) 4 = 15 / 16 := 
by 
  sorry

end aisha_probability_l1086_108630


namespace bottle_cap_cost_l1086_108618

-- Define the conditions given in the problem.
def caps_cost (n : ℕ) (cost : ℝ) : Prop := n * cost = 12

-- Prove that the cost of each bottle cap is $2 given 6 bottle caps cost $12.
theorem bottle_cap_cost (h : caps_cost 6 cost) : cost = 2 :=
sorry

end bottle_cap_cost_l1086_108618


namespace brushes_cost_l1086_108662

-- Define the conditions
def canvas_cost (B : ℝ) : ℝ := 3 * B
def paint_cost : ℝ := 5 * 8
def total_material_cost (B : ℝ) : ℝ := B + canvas_cost B + paint_cost
def earning_from_sale : ℝ := 200 - 80

-- State the question as a theorem in Lean
theorem brushes_cost (B : ℝ) (h : total_material_cost B = earning_from_sale) : B = 20 :=
sorry

end brushes_cost_l1086_108662


namespace tournament_teams_matches_l1086_108693

theorem tournament_teams_matches (teams : Fin 10 → ℕ) 
  (h : ∀ i, teams i ≤ 9) : 
  ∃ i j : Fin 10, i ≠ j ∧ teams i = teams j := 
by 
  sorry

end tournament_teams_matches_l1086_108693


namespace power_function_through_point_l1086_108649

-- Define the condition that the power function passes through the point (2, 8)
theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) (h : ∀ x, f x = x^α) (h₂ : f 2 = 8) :
  α = 3 ∧ ∀ x, f x = x^3 :=
by
  -- Proof will be provided here
  sorry

end power_function_through_point_l1086_108649


namespace boxes_of_orange_crayons_l1086_108681

theorem boxes_of_orange_crayons
  (n_orange_boxes : ℕ)
  (orange_crayons_per_box : ℕ := 8)
  (blue_boxes : ℕ := 7) (blue_crayons_per_box : ℕ := 5)
  (red_boxes : ℕ := 1) (red_crayons_per_box : ℕ := 11)
  (total_crayons : ℕ := 94)
  (h_total_crayons : (n_orange_boxes * orange_crayons_per_box) + (blue_boxes * blue_crayons_per_box) + (red_boxes * red_crayons_per_box) = total_crayons):
  n_orange_boxes = 6 := 
by sorry

end boxes_of_orange_crayons_l1086_108681


namespace average_visitors_per_day_is_276_l1086_108657

-- Define the number of days in the month
def num_days_in_month : ℕ := 30

-- Define the number of Sundays in the month
def num_sundays_in_month : ℕ := 4

-- Define the number of other days in the month
def num_other_days_in_month : ℕ := num_days_in_month - num_sundays_in_month * 7 / 7 + 2

-- Define the average visitors on Sundays
def avg_visitors_sunday : ℕ := 510

-- Define the average visitors on other days
def avg_visitors_other_days : ℕ := 240

-- Calculate total visitors on Sundays
def total_visitors_sundays : ℕ := num_sundays_in_month * avg_visitors_sunday

-- Calculate total visitors on other days
def total_visitors_other_days : ℕ := num_other_days_in_month * avg_visitors_other_days

-- Calculate total visitors in the month
def total_visitors_in_month : ℕ := total_visitors_sundays + total_visitors_other_days

-- Given conditions, prove average visitors per day in a month
theorem average_visitors_per_day_is_276 :
  total_visitors_in_month / num_days_in_month = 276 := by
  sorry

end average_visitors_per_day_is_276_l1086_108657


namespace line_equation_parametric_to_implicit_l1086_108667

theorem line_equation_parametric_to_implicit (t : ℝ) :
  ∀ x y : ℝ, (x = 3 * t + 6 ∧ y = 5 * t - 7) → y = (5 / 3) * x - 17 :=
by
  intros x y h
  obtain ⟨hx, hy⟩ := h
  sorry

end line_equation_parametric_to_implicit_l1086_108667


namespace basis_vetors_correct_options_l1086_108652

def is_basis (e1 e2 : ℝ × ℝ) : Prop :=
  e1 ≠ (0, 0) ∧ e2 ≠ (0, 0) ∧ e1.1 * e2.2 - e1.2 * e2.1 ≠ 0

def option_A : ℝ × ℝ := (0, 0)
def option_A' : ℝ × ℝ := (1, 2)

def option_B : ℝ × ℝ := (2, -1)
def option_B' : ℝ × ℝ := (1, 2)

def option_C : ℝ × ℝ := (-1, -2)
def option_C' : ℝ × ℝ := (1, 2)

def option_D : ℝ × ℝ := (1, 1)
def option_D' : ℝ × ℝ := (1, 2)

theorem basis_vetors_correct_options:
  ¬ is_basis option_A option_A' ∧ ¬ is_basis option_C option_C' ∧ 
  is_basis option_B option_B' ∧ is_basis option_D option_D' := 
by
  sorry

end basis_vetors_correct_options_l1086_108652


namespace James_final_assets_correct_l1086_108605

/-- Given the following initial conditions:
- James starts with 60 gold bars.
- He pays 10% in tax.
- He loses half of what is left in a divorce.
- He invests 25% of the remaining gold bars in a stock market and earns an additional gold bar.
- On Monday, he exchanges half of his remaining gold bars at a rate of 5 silver bars for 1 gold bar.
- On Tuesday, he exchanges half of his remaining gold bars at a rate of 7 silver bars for 1 gold bar.
- On Wednesday, he exchanges half of his remaining gold bars at a rate of 3 silver bars for 1 gold bar.

We need to determine:
- The number of silver bars James has,
- The number of remaining gold bars James has, and
- The number of gold bars worth from the stock investment James has after these transactions.
-/
noncomputable def James_final_assets (init_gold : ℕ) : ℕ × ℕ × ℕ :=
  let tax := init_gold / 10
  let gold_after_tax := init_gold - tax
  let gold_after_divorce := gold_after_tax / 2
  let invest_gold := gold_after_divorce * 25 / 100
  let remaining_gold_after_invest := gold_after_divorce - invest_gold
  let gold_after_stock := remaining_gold_after_invest + 1
  let monday_gold_exchanged := gold_after_stock / 2
  let monday_silver := monday_gold_exchanged * 5
  let remaining_gold_after_monday := gold_after_stock - monday_gold_exchanged
  let tuesday_gold_exchanged := remaining_gold_after_monday / 2
  let tuesday_silver := tuesday_gold_exchanged * 7
  let remaining_gold_after_tuesday := remaining_gold_after_monday - tuesday_gold_exchanged
  let wednesday_gold_exchanged := remaining_gold_after_tuesday / 2
  let wednesday_silver := wednesday_gold_exchanged * 3
  let remaining_gold_after_wednesday := remaining_gold_after_tuesday - wednesday_gold_exchanged
  let total_silver := monday_silver + tuesday_silver + wednesday_silver
  (total_silver, remaining_gold_after_wednesday, invest_gold)

theorem James_final_assets_correct : James_final_assets 60 = (99, 3, 6) := 
sorry

end James_final_assets_correct_l1086_108605


namespace simplify_fraction_l1086_108627

theorem simplify_fraction (x y : ℝ) (h : x ≠ y) : ((x^2 - y^2) / (x - y)) = x + y :=
by
  -- This is a placeholder for the actual proof
  sorry

end simplify_fraction_l1086_108627


namespace carpet_cost_calculation_l1086_108607

theorem carpet_cost_calculation
  (length_feet : ℕ)
  (width_feet : ℕ)
  (feet_to_yards : ℕ)
  (cost_per_square_yard : ℕ)
  (h_length : length_feet = 15)
  (h_width : width_feet = 12)
  (h_convert : feet_to_yards = 3)
  (h_cost : cost_per_square_yard = 10) :
  (length_feet / feet_to_yards) *
  (width_feet / feet_to_yards) *
  cost_per_square_yard = 200 := by
  sorry

end carpet_cost_calculation_l1086_108607


namespace largest_value_expression_l1086_108668

theorem largest_value_expression (a b c : ℝ) (ha : a ∈ ({1, 2, 4} : Set ℝ)) (hb : b ∈ ({1, 2, 4} : Set ℝ)) (hc : c ∈ ({1, 2, 4} : Set ℝ)) (habc_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (a / 2) / (b / c) ≤ 4 :=
sorry

end largest_value_expression_l1086_108668


namespace find_side_a_find_area_l1086_108617

-- Definitions from the conditions
variables {A B C : ℝ} 
variables {a b c : ℝ}
variable (angle_B: B = 120 * Real.pi / 180)
variable (side_b: b = Real.sqrt 7)
variable (side_c: c = 1)

-- The first proof problem: Prove that a = 2 given the above conditions
theorem find_side_a (h_angle_B: B = 120 * Real.pi / 180)
  (h_side_b: b = Real.sqrt 7) (h_side_c: c = 1)
  (h_cos_formula: b^2 = a^2 + c^2 - 2 * a * c * Real.cos B) : a = 2 :=
  by
  sorry

-- The second proof problem: Prove that the area is sqrt(3)/2 given the above conditions
theorem find_area (h_angle_B: B = 120 * Real.pi / 180)
  (h_side_b: b = Real.sqrt 7) (h_side_c: c = 1)
  (h_side_a: a = 2) : (1 / 2) * a * c * Real.sin B = Real.sqrt 3 / 2 :=
  by
  sorry

end find_side_a_find_area_l1086_108617


namespace quadratic_inequality_l1086_108615

theorem quadratic_inequality (a b c : ℝ) (h : a^2 + a * b + a * c < 0) : b^2 > 4 * a * c := 
sorry

end quadratic_inequality_l1086_108615


namespace pythagorean_triple_divisibility_l1086_108658

theorem pythagorean_triple_divisibility (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (∃ k₃, k₃ ∣ a ∨ k₃ ∣ b) ∧
  (∃ k₄, k₄ ∣ a ∨ k₄ ∣ b ∧ 2 ∣ k₄) ∧
  (∃ k₅, k₅ ∣ a ∨ k₅ ∣ b ∨ k₅ ∣ c) :=
by
  sorry

end pythagorean_triple_divisibility_l1086_108658


namespace painted_cells_solutions_l1086_108620

def painted_cells (k l : ℕ) : ℕ := (2 * k + 1) * (2 * l + 1) - 74

theorem painted_cells_solutions : ∃ k l : ℕ, k * l = 74 ∧ (painted_cells k l = 373 ∨ painted_cells k l = 301) :=
by
  sorry

end painted_cells_solutions_l1086_108620


namespace ratio_of_awards_l1086_108699

theorem ratio_of_awards 
  (Scott_awards : ℕ) (Scott_awards_eq : Scott_awards = 4)
  (Jessie_awards : ℕ) (Jessie_awards_eq : Jessie_awards = 3 * Scott_awards)
  (rival_awards : ℕ) (rival_awards_eq : rival_awards = 24) :
  rival_awards / Jessie_awards = 2 :=
by sorry

end ratio_of_awards_l1086_108699


namespace total_number_of_notes_l1086_108637

-- The total amount of money in Rs.
def total_amount : ℕ := 400

-- The number of each type of note is equal.
variable (n : ℕ)

-- The total value equation given the number of each type of note.
def total_value : ℕ := n * 1 + n * 5 + n * 10

-- Prove that if the total value equals 400, the total number of notes is 75.
theorem total_number_of_notes : total_value n = total_amount → 3 * n = 75 :=
by
  sorry

end total_number_of_notes_l1086_108637


namespace winning_candidate_votes_l1086_108663

-- Define the conditions as hypotheses in Lean.
def two_candidates (candidates : ℕ) : Prop := candidates = 2
def winner_received_62_percent (V : ℝ) (votes_winner : ℝ) : Prop := votes_winner = 0.62 * V
def winning_margin (V : ℝ) : Prop := 0.24 * V = 384

-- The main theorem to prove: the winner candidate received 992 votes.
theorem winning_candidate_votes (V votes_winner : ℝ) (candidates : ℕ) 
  (h1 : two_candidates candidates) 
  (h2 : winner_received_62_percent V votes_winner)
  (h3 : winning_margin V) : 
  votes_winner = 992 :=
by
  sorry

end winning_candidate_votes_l1086_108663


namespace min_fraction_value_l1086_108603

noncomputable def min_value_fraction (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h₁ : x^2 + y^2 + z^2 = 1) : ℝ :=
  (z+1)^2 / (2 * x * y * z)

theorem min_fraction_value (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h₁ : x^2 + y^2 + z^2 = 1) :
  min_value_fraction x y z h h₁ = 3 + 2 * Real.sqrt 2 :=
  sorry

end min_fraction_value_l1086_108603


namespace Grant_score_is_100_l1086_108635

/-- Definition of scores --/
def Hunter_score : ℕ := 45

def John_score (H : ℕ) : ℕ := 2 * H

def Grant_score (J : ℕ) : ℕ := J + 10

/-- Theorem to prove Grant's score --/
theorem Grant_score_is_100 : Grant_score (John_score Hunter_score) = 100 := 
  sorry

end Grant_score_is_100_l1086_108635
